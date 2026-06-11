# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the MPS-based Krotov-inspired discrete adjoint optimizer.

The tests validate the transfer of the gate-local Krotov scheme to the MPS backend
against dense state-vector references:
  - Forward propagation of parameterized circuits matches dense matrix propagation.
  - The batch Krotov direction equals the finite-difference gradient of the
    empirical loss (exact-gradient property) for both supported loss models.
  - The online stale-adjoint sweep matches a dense re-implementation.
  - Hybrid training reduces the empirical loss on a small parity task.
  - The Qiskit converter reproduces bound-parameter statevectors.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.optimization import (
    KrotovOptions,
    KrotovReadout,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    empirical_loss,
    sample_contribution,
    train_krotov_batch,
    train_krotov_hybrid,
)
from mqt.yaqs.optimization.krotov import (
    backward_costates,
    forward_states,
    terminal_costate,
)

rng = np.random.default_rng(42)

_I2 = np.eye(2, dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def embed_one_site(mat: np.ndarray, site: int, num_qubits: int) -> np.ndarray:
    """Embed a 2x2 operator into the full register (site 0 = least significant bit).

    Returns:
        The full ``2^n x 2^n`` operator.
    """
    full = np.array([[1.0]], dtype=np.complex128)
    for q in range(num_qubits - 1, -1, -1):
        full = np.kron(full, mat if q == site else _I2)
    return full


def embed_two_site(mat4: np.ndarray, site_i: int, site_j: int, num_qubits: int) -> np.ndarray:
    """Embed a 4x4 operator on ascending sites ``(site_i, site_j)`` with ``site_i`` as the
    more significant local factor (site 0 = least significant register bit).

    Returns:
        The full ``2^n x 2^n`` operator.
    """
    dim = 2**num_qubits
    full = np.zeros((dim, dim), dtype=np.complex128)
    for col in range(dim):
        bits = [(col >> q) & 1 for q in range(num_qubits)]
        local_col = 2 * bits[site_i] + bits[site_j]
        for local_row in range(4):
            amp = mat4[local_row, local_col]
            if amp == 0.0:
                continue
            out_bits = bits.copy()
            out_bits[site_i] = (local_row >> 1) & 1
            out_bits[site_j] = local_row & 1
            row = sum(bit << q for q, bit in enumerate(out_bits))
            full[row, col] += amp
    return full


def embed_gate(circuit: ParameterizedCircuit, gate: ParameterizedGate, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Embed one circuit factor into the full register.

    Returns:
        The full ``2^n x 2^n`` unitary of the factor.
    """
    matrix, sites = circuit.gate_matrix(gate, theta, x)
    if len(sites) == 1:
        return embed_one_site(matrix, sites[0], circuit.num_qubits)
    return embed_two_site(matrix, sites[0], sites[1], circuit.num_qubits)


def dense_forward(circuit: ParameterizedCircuit, theta: np.ndarray, x: np.ndarray, psi0: np.ndarray) -> np.ndarray:
    """Propagate a dense statevector through the circuit.

    Returns:
        The final statevector.
    """
    psi = psi0.copy()
    for gate in circuit.gates:
        psi = embed_gate(circuit, gate, theta, x) @ psi
    return psi


def build_test_circuit() -> ParameterizedCircuit:
    """Build a 3-qubit circuit covering all supported gate types.

    Includes single-qubit rotations, a Hadamard, nearest-neighbor and long-range
    entanglers (also with reversed control/target order), a long-range ``rzz``,
    a data-encoded rotation, and a scaled angle map.

    Returns:
        The test circuit with four trainable parameters.
    """
    gates = [
        ParameterizedGate("h", (0,)),
        ParameterizedGate("rx", (0,), param_index=0),
        ParameterizedGate("ry", (1,), param_index=1, angle_scale=1.7, angle_offset=0.2),
        ParameterizedGate("rz", (2,), param_index=2),
        ParameterizedGate("cx", (0, 1)),
        ParameterizedGate("cx", (2, 0)),
        ParameterizedGate("rzz", (0, 2), param_index=3),
        ParameterizedGate("ry", (2,), data_map=lambda x: float(x[0])),
        ParameterizedGate("cp", (1, 0), param_index=0),
    ]
    return ParameterizedCircuit(num_qubits=3, gates=gates)


def test_forward_states_match_dense_statevector() -> None:
    """The MPS forward sweep reproduces dense matrix propagation for every prefix."""
    circuit = build_test_circuit()
    theta = rng.normal(size=circuit.num_params)
    x = np.array([0.37])

    states = forward_states(circuit, theta, x, MPS(circuit.num_qubits), KrotovTruncation())

    psi = np.zeros(2**circuit.num_qubits, dtype=np.complex128)
    psi[0] = 1.0
    np.testing.assert_allclose(states[0].to_vec(), psi, atol=1e-12)
    for k, gate in enumerate(circuit.gates):
        psi = embed_gate(circuit, gate, theta, x) @ psi
        np.testing.assert_allclose(states[k + 1].to_vec(), psi, atol=1e-10)


@pytest.mark.parametrize(("loss", "use_bias", "label"), [("mse", True, -1.0), ("mse", False, 1.0), ("bce", False, 1.0)])
def test_batch_direction_equals_finite_difference_gradient(
    loss: Literal["mse", "bce"], use_bias: bool, label: float
) -> None:
    """The batch Krotov direction equals the exact gradient of the empirical loss."""
    circuit = build_test_circuit()
    readout = KrotovReadout(observable=Observable("z", 0), loss=loss, use_bias=use_bias)
    theta = rng.normal(size=circuit.num_params)
    bias = 0.1 if use_bias else 0.0
    truncation = KrotovTruncation()

    inputs = np.array([[0.37], [-0.6]])
    labels = np.array([label, -label if loss == "mse" else 1.0 - label])

    contribution = np.zeros(circuit.num_params)
    for x, y in zip(inputs, labels):
        sample_grad, _, _ = sample_contribution(
            circuit, theta, x, float(y), readout, bias, MPS(circuit.num_qubits), truncation
        )
        contribution += sample_grad
    contribution /= len(inputs)

    eps = 1e-6
    for idx in range(circuit.num_params):
        shifted_plus = theta.copy()
        shifted_plus[idx] += eps
        shifted_minus = theta.copy()
        shifted_minus[idx] -= eps
        numeric = (
            empirical_loss(circuit, shifted_plus, inputs, labels, readout, bias)
            - empirical_loss(circuit, shifted_minus, inputs, labels, readout, bias)
        ) / (2 * eps)
        assert contribution[idx] == pytest.approx(numeric, abs=1e-6)


def test_backward_costates_match_dense_adjoint() -> None:
    """Backward-propagated costates match the dense adjoint propagation."""
    circuit = build_test_circuit()
    readout = KrotovReadout(observable=Observable("z", 0), loss="mse")
    theta = rng.normal(size=circuit.num_params)
    x = np.array([0.5])
    truncation = KrotovTruncation()

    states = forward_states(circuit, theta, x, MPS(circuit.num_qubits), truncation)
    chi_terminal, _, z = terminal_costate(states[-1], readout, 1.0, 0.0)
    costates = backward_costates(circuit, theta, x, chi_terminal, truncation)

    psi0 = np.zeros(2**circuit.num_qubits, dtype=np.complex128)
    psi0[0] = 1.0
    psi_final = dense_forward(circuit, theta, x, psi0)
    obs_full = embed_one_site(_Z, 0, circuit.num_qubits)
    z_dense = float(np.real(psi_final.conj() @ obs_full @ psi_final))
    assert z == pytest.approx(z_dense, abs=1e-10)

    chi_dense = 2.0 * (z_dense - 1.0) * (obs_full @ psi_final)
    np.testing.assert_allclose(costates[-1].to_vec(), chi_dense, atol=1e-10)
    for k in range(len(circuit.gates) - 1, -1, -1):
        chi_dense = embed_gate(circuit, circuit.gates[k], theta, x).conj().T @ chi_dense
        np.testing.assert_allclose(costates[k].to_vec(), chi_dense, atol=1e-10)


def test_online_sweep_matches_dense_reference() -> None:
    """One online stale-adjoint sweep matches a dense re-implementation."""
    from mqt.yaqs.optimization.krotov import _online_sample_update  # noqa: PLC0415, PLC2701

    circuit = build_test_circuit()
    readout = KrotovReadout(observable=Observable("z", 0), loss="mse")
    theta = rng.normal(size=circuit.num_params)
    x = np.array([0.21])
    y = -1.0
    step_size = 0.17
    truncation = KrotovTruncation()

    new_theta, _ = _online_sample_update(
        circuit, theta, x, y, readout, 0.0, step_size, MPS(circuit.num_qubits), truncation
    )

    # Dense reference of the same stale-adjoint sweep.
    num_qubits = circuit.num_qubits
    psi0 = np.zeros(2**num_qubits, dtype=np.complex128)
    psi0[0] = 1.0
    psi_final = dense_forward(circuit, theta, x, psi0)
    obs_full = embed_one_site(_Z, 0, num_qubits)
    z = float(np.real(psi_final.conj() @ obs_full @ psi_final))
    chi = 2.0 * (z - y) * (obs_full @ psi_final)

    num_gates = len(circuit.gates)
    chis: list[np.ndarray] = [chi] * (num_gates + 1)
    for k in range(num_gates - 1, -1, -1):
        chis[k] = embed_gate(circuit, circuit.gates[k], theta, x).conj().T @ chis[k + 1]

    dense_theta = theta.copy()
    current = psi0.copy()
    for k, gate in enumerate(circuit.gates):
        if gate.param_index is not None:
            gate_output = embed_gate(circuit, gate, dense_theta, x) @ current
            operator, sites = circuit.derivative_operator(gate)
            if len(sites) == 1:
                derivative_full = embed_one_site(operator, sites[0], num_qubits)
            else:
                derivative_full = embed_two_site(operator, sites[0], sites[1], num_qubits)
            signal = gate.angle_scale * 2.0 * float(np.real(chis[k + 1].conj() @ derivative_full @ gate_output))
            dense_theta[gate.param_index] -= step_size * signal
        current = embed_gate(circuit, gate, dense_theta, x) @ current

    np.testing.assert_allclose(new_theta, dense_theta, atol=1e-10)


def test_hybrid_training_reduces_parity_loss() -> None:
    """Hybrid Krotov training solves a small 3-bit parity task."""
    num_qubits = 3
    gates: list[ParameterizedGate] = []
    param = 0
    for _layer in range(2):
        for qubit in range(num_qubits):
            gates.append(ParameterizedGate("rz", (qubit,), param_index=param))
            gates.append(ParameterizedGate("ry", (qubit,), param_index=param + 1))
            gates.append(ParameterizedGate("rz", (qubit,), param_index=param + 2))
            param += 3
        gates.extend([
            ParameterizedGate("cx", (0, 1)),
            ParameterizedGate("cx", (1, 2)),
            ParameterizedGate("cx", (2, 0)),
        ])
    circuit = ParameterizedCircuit(num_qubits=num_qubits, gates=gates)
    readout = KrotovReadout(observable=Observable("z", 0), loss="mse", use_bias=True)

    inputs = np.array([[(idx >> q) & 1 for q in range(num_qubits)] for idx in range(2**num_qubits)], dtype=float)
    labels = np.array([1.0 if int(np.sum(bits)) % 2 else -1.0 for bits in inputs])

    def state_prep(x: np.ndarray) -> MPS:
        basis = "".join(str(int(bit)) for bit in x)
        return MPS(num_qubits, state="basis", basis_string=basis)

    initial_theta = np.random.default_rng(0).normal(scale=0.5, size=circuit.num_params)
    options = KrotovOptions(max_iterations=20, switch_iteration=12, online_step_size=0.3, batch_step_size=0.3, seed=1)

    result = train_krotov_hybrid(
        circuit,
        readout,
        inputs,
        labels,
        initial_theta=initial_theta,
        options=options,
        initial_state=state_prep,
    )

    losses = np.asarray(result.trace["loss"], dtype=float)
    assert losses[-1] < 0.05
    assert losses[-1] < 0.5 * losses[0]
    assert result.trace["phase"][1] == "online"
    assert result.trace["phase"][-1] == "batch"


def test_batch_training_monotonic_descent() -> None:
    """Full-batch Krotov descent with a small step size is monotone."""
    circuit = build_test_circuit()
    readout = KrotovReadout(observable=Observable("z", 0), loss="mse")
    inputs = np.array([[0.3], [-0.4], [0.9]])
    labels = np.array([1.0, -1.0, 1.0])
    initial_theta = rng.normal(scale=0.3, size=circuit.num_params)
    options = KrotovOptions(max_iterations=8, batch_step_size=0.05)

    result = train_krotov_batch(circuit, readout, inputs, labels, initial_theta=initial_theta, options=options)

    losses = np.asarray(result.trace["loss"], dtype=float)
    assert np.all(np.diff(losses) <= 1e-12)


def test_from_qiskit_matches_statevector() -> None:
    """The Qiskit converter reproduces bound-parameter statevectors and angle maps."""
    alpha = Parameter("alpha")
    beta = Parameter("beta")
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.rx(2.0 * alpha + 0.3, 0)
    qc.ry(beta, 1)
    qc.cx(0, 1)
    qc.rz(0.7, 2)
    qc.rzz(beta, 0, 2)
    qc.cx(2, 1)

    circuit = ParameterizedCircuit.from_qiskit(qc, parameters=[alpha, beta])
    assert circuit.num_params == 2

    theta = np.array([0.45, -0.8])
    states = forward_states(circuit, theta, np.array([]), MPS(3), KrotovTruncation())

    bound = qc.assign_parameters({alpha: theta[0], beta: theta[1]})
    reference = np.asarray(Statevector.from_instruction(bound).data, dtype=np.complex128)

    overlap = np.vdot(reference, states[-1].to_vec())
    assert abs(overlap) == pytest.approx(1.0, abs=1e-10)


def test_validation_errors() -> None:
    """Invalid configurations are rejected."""
    with pytest.raises(ValueError, match="bias is only supported"):
        KrotovReadout(observable=Observable("z", 0), loss="bce", use_bias=True)

    with pytest.raises(ValueError, match="not a supported one-parameter gate"):
        ParameterizedCircuit(2, [ParameterizedGate("cx", (0, 1), param_index=0)])

    with pytest.raises(ValueError, match="outside range"):
        ParameterizedCircuit(2, [ParameterizedGate("rx", (2,), param_index=0)])

    with pytest.raises(ValueError, match="duplicate sites"):
        ParameterizedCircuit(2, [ParameterizedGate("rzz", (1, 1), param_index=0)])
