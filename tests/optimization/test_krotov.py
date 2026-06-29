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
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.circuit_library import (
    brickwall_matrix_product_disentangler_bonds,
    create_brickwall_matrix_product_disentangler_ansatz_circuit,
    create_sequential_matrix_product_disentangler_circuit,
)
from mqt.yaqs.optimization import (
    KrotovNoiseMap,
    KrotovOptions,
    KrotovReadout,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    brickwall_matrix_product_disentangler_num_parameters,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
    create_sequential_matrix_product_disentangler_parameterized_circuit,
    empirical_loss,
    noisy_sample_loss,
    noisy_state_preparation_contribution,
    noisy_state_preparation_cross_contribution,
    noisy_state_preparation_metrics,
    sample_contribution,
    state_preparation_contribution,
    state_preparation_loss,
    state_preparation_metrics,
    train_krotov_batch,
    train_krotov_hybrid,
    train_krotov_noisy_state_preparation_batch,
    train_krotov_noisy_state_preparation_hybrid,
    train_krotov_state_preparation_batch,
    train_krotov_state_preparation_hybrid,
)
from mqt.yaqs.optimization.krotov import (
    backward_costates,
    forward_states,
    forward_tjm_trajectory,
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
    """Embed a 4x4 operator on ascending sites ``(site_i, site_j)``.

    The lower site is the more significant local factor (site 0 = least significant register bit).

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
            if np.isclose(amp, 0.0):
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


def u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """Return the Qiskit U3 matrix used by the dense BMPD helper.

    Returns:
        Dense ``2 x 2`` unitary matrix.
    """
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    return np.array(
        [
            [cos, -np.exp(1j * lam) * sin],
            [np.exp(1j * phi) * sin, np.exp(1j * (phi + lam)) * cos],
        ],
        dtype=np.complex128,
    )


def random_unitary(dim: int, seed: int) -> np.ndarray:
    """Build a deterministic random dense unitary.

    Returns:
        Dense unitary matrix.
    """
    local_rng = np.random.default_rng(seed)
    matrix = local_rng.normal(size=(dim, dim)) + 1j * local_rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r) / np.abs(np.diag(r))
    return np.asarray(q * phases, dtype=np.complex128)


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


def build_state_preparation_test_circuit() -> ParameterizedCircuit:
    """Build a 3-qubit data-free circuit for target-state gradient tests.

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
    loss: Literal["mse", "bce"],
    use_bias: bool,  # noqa: FBT001
    label: float,
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
    for x, y in zip(inputs, labels, strict=False):
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
            gates.extend((
                ParameterizedGate("rz", (qubit,), param_index=param),
                ParameterizedGate("ry", (qubit,), param_index=param + 1),
                ParameterizedGate("rz", (qubit,), param_index=param + 2),
            ))
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


def test_bmpd_parameterized_circuit_uses_scalar_gate_parameters() -> None:
    """The Krotov BMPD constructor emits only supported one-parameter gates."""
    circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(4, 1)

    assert circuit.num_params == brickwall_matrix_product_disentangler_num_parameters(4, 1)
    assert circuit.num_params == 27
    assert len(circuit.gates) == 27
    assert [(gate.name, gate.sites, gate.param_index, gate.angle_scale) for gate in circuit.gates[:3]] == [
        ("rxx", (1, 2), 24, -1.0),
        ("ryy", (1, 2), 25, -1.0),
        ("rzz", (1, 2), 26, -1.0),
    ]
    assert all(gate.is_trainable and gate.is_parametric for gate in circuit.gates)


def test_bmpd_parameterized_circuit_matches_dense_ansatz() -> None:
    """The scalar Krotov BMPD ansatz matches the dense UnitaryGate BMPD ansatz."""
    num_qubits = 4
    depth = 1
    initial_param_count = 3 * num_qubits
    num_bmpd_rows = sum(len(layer) for layer in brickwall_matrix_product_disentangler_bonds(num_qubits, depth))

    circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(
        num_qubits,
        depth,
        initial_single_qubit_layer=True,
    )
    theta = rng.normal(scale=0.2, size=circuit.num_params)

    initial_unitaries = [(u3_matrix(*theta[3 * site : 3 * site + 3]), (site,)) for site in range(num_qubits)]
    dense_circuit = create_brickwall_matrix_product_disentangler_ansatz_circuit(
        num_qubits,
        depth,
        theta[initial_param_count:].reshape(num_bmpd_rows, 9),
        initial_single_qubit_unitaries=initial_unitaries,
    )

    states = forward_states(circuit, theta, np.array([]), MPS(num_qubits), KrotovTruncation())
    reference = np.asarray(Statevector(dense_circuit).data, dtype=np.complex128)
    overlap = np.vdot(reference, states[-1].to_vec())
    assert abs(overlap) == pytest.approx(1.0, abs=1e-10)


def test_smpd_kak_parameterized_circuit_matches_dense_gate_sequence() -> None:
    """The KAK-expanded SMPD circuit matches the dense SMPD UnitaryGate circuit."""
    num_qubits = 3
    dense_gates = [
        (random_unitary(2, seed=10), (0,)),
        (random_unitary(4, seed=11), (0, 1)),
        (random_unitary(4, seed=12), (2, 1)),
        (random_unitary(2, seed=13), (2,)),
    ]

    circuit, initial_theta = create_sequential_matrix_product_disentangler_parameterized_circuit(
        num_qubits,
        dense_gates,
    )
    dense_circuit = create_sequential_matrix_product_disentangler_circuit(num_qubits, dense_gates)

    assert circuit.num_params == len(initial_theta)
    assert all(gate.name != "unitary" for gate in circuit.gates)
    assert all((gate.param_index is None) or gate.is_parametric for gate in circuit.gates)

    states = forward_states(circuit, initial_theta, np.array([]), MPS(num_qubits), KrotovTruncation())
    reference = np.asarray(Statevector(dense_circuit).data, dtype=np.complex128)
    overlap = np.vdot(reference, states[-1].to_vec())
    assert abs(overlap) == pytest.approx(1.0, abs=1e-10)


def test_smpd_kak_parameterized_circuit_supports_krotov_gradient() -> None:
    """The KAK-expanded SMPD circuit can be used in the deterministic Krotov adjoint."""
    num_qubits = 2
    dense_gates = [(random_unitary(4, seed=21), (0, 1))]
    circuit, theta = create_sequential_matrix_product_disentangler_parameterized_circuit(num_qubits, dense_gates)
    readout = KrotovReadout(observable=Observable("z", 0), loss="mse")
    x = np.array([])
    y = 1.0

    contribution, _, _ = sample_contribution(circuit, theta, x, y, readout, 0.0, MPS(num_qubits), KrotovTruncation())

    eps = 1e-6
    for idx in range(circuit.num_params):
        shifted_plus = theta.copy()
        shifted_plus[idx] += eps
        shifted_minus = theta.copy()
        shifted_minus[idx] -= eps
        numeric = (
            empirical_loss(circuit, shifted_plus, np.array([x]), np.array([y]), readout)
            - empirical_loss(circuit, shifted_minus, np.array([x]), np.array([y]), readout)
        ) / (2 * eps)
        assert contribution[idx] == pytest.approx(numeric, abs=1e-6)


def test_state_preparation_direction_equals_finite_difference_gradient() -> None:
    """The target-state Krotov direction equals the infidelity gradient."""
    circuit = build_state_preparation_test_circuit()
    theta = rng.normal(scale=0.4, size=circuit.num_params)
    target = rng.normal(size=2**circuit.num_qubits) + 1j * rng.normal(size=2**circuit.num_qubits)
    target /= np.linalg.norm(target)

    contribution, loss_value, fidelity = state_preparation_contribution(
        circuit,
        theta,
        target,
        MPS(circuit.num_qubits),
        KrotovTruncation(),
    )
    assert loss_value == pytest.approx(1.0 - fidelity, abs=1e-12)

    eps = 1e-6
    for idx in range(circuit.num_params):
        shifted_plus = theta.copy()
        shifted_plus[idx] += eps
        shifted_minus = theta.copy()
        shifted_minus[idx] -= eps
        numeric = (
            state_preparation_loss(circuit, shifted_plus, target)
            - state_preparation_loss(circuit, shifted_minus, target)
        ) / (2 * eps)
        assert contribution[idx] == pytest.approx(numeric, abs=1e-6)


def test_state_preparation_has_zero_gradient_at_perfect_fidelity() -> None:
    """The target-state objective is stationary when the circuit already prepares the target."""
    circuit = build_state_preparation_test_circuit()
    theta = rng.normal(scale=0.3, size=circuit.num_params)
    target = forward_states(circuit, theta, np.array([]), MPS(circuit.num_qubits), KrotovTruncation())[-1]

    contribution, loss_value, fidelity = state_preparation_contribution(
        circuit,
        theta,
        target,
        MPS(circuit.num_qubits),
        KrotovTruncation(),
    )

    assert fidelity == pytest.approx(1.0, abs=1e-10)
    assert loss_value == pytest.approx(0.0, abs=1e-10)
    assert np.linalg.norm(contribution) == pytest.approx(0.0, abs=1e-9)


def test_state_preparation_batch_training_increases_target_fidelity() -> None:
    """Batch Krotov training prepares a small target state from a dense statevector."""
    circuit = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("rz", (0,), param_index=0),
            ParameterizedGate("ry", (0,), param_index=1),
            ParameterizedGate("rz", (0,), param_index=2),
        ],
    )
    target_theta = np.array([0.8, 1.1, -0.4])
    target = forward_states(circuit, target_theta, np.array([]), MPS(1), KrotovTruncation())[-1].to_vec()
    initial_theta = np.array([-0.6, 0.2, 0.5])
    options = KrotovOptions(max_iterations=35, batch_step_size=0.35)

    initial_loss, initial_fidelity = state_preparation_metrics(circuit, initial_theta, target)
    result = train_krotov_state_preparation_batch(
        circuit,
        target,
        initial_theta=initial_theta,
        options=options,
    )
    final_loss = float(result.trace["loss"][-1])
    final_fidelity = float(result.trace["fidelity"][-1])

    assert result.bias == pytest.approx(0.0, abs=1e-15)
    assert final_loss < initial_loss
    assert final_fidelity > initial_fidelity
    assert final_fidelity > 0.99
    assert result.trace["phase"][1] == "batch"


def test_noisy_state_preparation_independent_hybrid_noise_off_matches_noiseless_training() -> None:
    """Noisy hybrid state preparation reduces to deterministic hybrid training without noise."""
    circuit = build_state_preparation_test_circuit()
    initial_theta = rng.normal(scale=0.2, size=circuit.num_params)
    target_theta = rng.normal(scale=0.3, size=circuit.num_params)
    target = forward_states(circuit, target_theta, np.array([]), MPS(circuit.num_qubits), KrotovTruncation())[
        -1
    ].to_vec()
    options = KrotovOptions(
        max_iterations=4,
        switch_iteration=2,
        online_step_size=0.1,
        batch_step_size=0.1,
        seed=9,
    )

    deterministic = train_krotov_state_preparation_hybrid(
        circuit,
        target,
        initial_theta=initial_theta,
        options=options,
    )
    noisy = train_krotov_noisy_state_preparation_hybrid(
        circuit,
        target,
        None,
        KrotovTJMOptions(num_trajectories=3, random_seed=11, trajectory_update="independent"),
        initial_theta=initial_theta,
        options=options,
    )

    np.testing.assert_allclose(noisy.theta, deterministic.theta, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(noisy.trace["fidelity"], dtype=np.float64),
        np.asarray(deterministic.trace["fidelity"], dtype=np.float64),
        atol=1e-10,
    )
    assert noisy.trace["phase"] == deterministic.trace["phase"]


def test_noisy_state_preparation_noise_off_matches_noiseless_contribution() -> None:
    """Noisy trajectory mode with one noiseless trajectory reduces to deterministic Krotov."""
    circuit = build_state_preparation_test_circuit()
    theta = rng.normal(scale=0.25, size=circuit.num_params)
    target = rng.normal(size=2**circuit.num_qubits) + 1j * rng.normal(size=2**circuit.num_qubits)
    target /= np.linalg.norm(target)
    truncation = KrotovTruncation()

    deterministic, det_loss, det_fidelity = state_preparation_contribution(
        circuit,
        theta,
        target,
        MPS(circuit.num_qubits),
        truncation,
    )
    noisy, noisy_loss, noisy_fidelity, trajectories = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        None,
        KrotovTJMOptions(num_trajectories=1),
        MPS(circuit.num_qubits),
        truncation,
    )

    np.testing.assert_allclose(noisy, deterministic, atol=1e-10)
    assert noisy_loss == pytest.approx(det_loss, abs=1e-10)
    assert noisy_fidelity == pytest.approx(det_fidelity, abs=1e-10)
    assert all(noise_map.operators == () for noise_map in trajectories[0].noise_maps)


def test_cross_trajectory_noise_off_matches_noiseless_contribution() -> None:
    """Cross-trajectory mode with one noiseless trajectory reduces to deterministic Krotov."""
    circuit = build_state_preparation_test_circuit()
    theta = rng.normal(scale=0.25, size=circuit.num_params)
    target = rng.normal(size=2**circuit.num_qubits) + 1j * rng.normal(size=2**circuit.num_qubits)
    target /= np.linalg.norm(target)
    truncation = KrotovTruncation()

    deterministic, det_loss, det_fidelity = state_preparation_contribution(
        circuit,
        theta,
        target,
        MPS(circuit.num_qubits),
        truncation,
    )
    cross, cross_loss, cross_fidelity, trajectories = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        None,
        KrotovTJMOptions(num_trajectories=1, trajectory_update="cross"),
        MPS(circuit.num_qubits),
        truncation,
    )

    np.testing.assert_allclose(cross, deterministic, atol=1e-10)
    assert cross_loss == pytest.approx(det_loss, abs=1e-10)
    assert cross_fidelity == pytest.approx(det_fidelity, abs=1e-10)
    assert all(noise_map.operators == () for noise_map in trajectories[0].noise_maps)


def test_noisy_state_preparation_fixed_seed_reproducibility_and_mean_fidelity() -> None:
    """Fixed seeds reproduce trajectory contributions and metrics average fidelities."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.37])
    target = np.array([0.2, np.sqrt(1.0 - 0.2**2)], dtype=np.complex128)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.07}])
    tjm_options = KrotovTJMOptions(num_trajectories=4, random_seed=11)
    truncation = KrotovTruncation()

    first = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        noise_model,
        tjm_options,
        MPS(1),
        truncation,
        iteration=3,
    )
    second = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        noise_model,
        tjm_options,
        MPS(1),
        truncation,
        iteration=3,
    )

    np.testing.assert_allclose(first[0], second[0], atol=1e-12)
    assert first[1] == pytest.approx(second[1], abs=1e-12)
    assert first[2] == pytest.approx(second[2], abs=1e-12)
    assert [[m.jump_process_index for m in t.noise_maps] for t in first[3]] == [
        [m.jump_process_index for m in t.noise_maps] for t in second[3]
    ]

    loss, mean_fidelity, fidelities = noisy_state_preparation_metrics(
        circuit,
        theta,
        target,
        noise_model,
        tjm_options,
        iteration=3,
    )
    assert mean_fidelity == pytest.approx(np.mean(fidelities), abs=1e-12)
    assert loss == pytest.approx(1.0 - mean_fidelity, abs=1e-12)


def test_cross_trajectory_contribution_matches_dense_double_sum() -> None:
    """Cross mode evaluates the explicit density-matrix trajectory double sum."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.37])
    target = np.array([np.cos(0.8 / 2), np.sin(0.8 / 2)], dtype=np.complex128)
    x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    fixed_maps = [[KrotovNoiseMap()], [KrotovNoiseMap(operators=((x_gate, (0,)),))]]
    tjm_options = KrotovTJMOptions(num_trajectories=2, trajectory_update="cross")

    contribution, loss_value, mean_fidelity, trajectories = noisy_state_preparation_cross_contribution(
        circuit,
        theta,
        target,
        None,
        tjm_options,
        MPS(1),
        KrotovTruncation(),
        fixed_noise_maps=fixed_maps,
    )

    y_gate = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    derivative = -0.5j * y_gate
    phi = np.array([np.cos(theta[0] / 2), np.sin(theta[0] / 2)], dtype=np.complex128)
    forward_states = [phi, phi]
    backward_states = [target, x_gate.conj().T @ target]
    expected = 0.0
    for xi_state in backward_states:
        for phi_state in forward_states:
            derivative_overlap = np.vdot(xi_state, derivative @ phi_state)
            density_overlap = np.vdot(phi_state, xi_state)
            expected -= 2.0 * float(np.real(derivative_overlap * density_overlap)) / 4.0

    np.testing.assert_allclose(contribution, np.array([expected]), atol=1e-12)
    assert mean_fidelity == pytest.approx(
        np.mean([abs(np.vdot(target, trajectory.states[-1].to_vec())) ** 2 for trajectory in trajectories]),
        abs=1e-12,
    )
    assert loss_value == pytest.approx(1.0 - mean_fidelity, abs=1e-12)


def test_tjm_noisy_gate_indices_apply_noise_only_at_selected_gates() -> None:
    """Logical-gate noise masks avoid repeated noise on decomposed primitives."""
    circuit = ParameterizedCircuit(
        1,
        [
            ParameterizedGate("rx", (0,), param_index=0),
            ParameterizedGate("ry", (0,), param_index=1),
            ParameterizedGate("rz", (0,), param_index=2),
        ],
    )
    theta = np.array([0.1, 0.2, 0.3])
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 2.0}])
    tjm_options = KrotovTJMOptions(num_trajectories=1, random_seed=3, noisy_gate_indices=(1,))

    trajectory = forward_tjm_trajectory(
        circuit,
        theta,
        np.array([], dtype=np.float64),
        MPS(1),
        KrotovTruncation(),
        noise_model,
        tjm_options,
        np.random.default_rng(7),
    )

    assert trajectory.noise_maps[0].operators == ()
    assert trajectory.noise_maps[1].operators != ()
    assert trajectory.noise_maps[1].jump_process_index == 0
    assert trajectory.noise_maps[2].operators == ()


def test_noisy_state_preparation_fixed_trajectory_gradient_matches_finite_difference() -> None:
    """Fixed realized drift maps give a finite-difference pathwise gradient."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.41])
    target = np.array([np.cos(0.8 / 2), np.sin(0.8 / 2)], dtype=np.complex128)
    strength = 0.05
    drift = np.exp(-0.5 * strength) * np.eye(2, dtype=np.complex128)
    fixed_maps = [[KrotovNoiseMap(operators=((drift, (0,)),))], [KrotovNoiseMap(operators=((drift, (0,)),))]]
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": strength}])
    tjm_options = KrotovTJMOptions(num_trajectories=2)
    truncation = KrotovTruncation()

    contribution, loss_value, mean_fidelity, _trajectories = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        noise_model,
        tjm_options,
        MPS(1),
        truncation,
        fixed_noise_maps=fixed_maps,
    )
    assert loss_value == pytest.approx(1.0 - mean_fidelity, abs=1e-12)

    eps = 1e-6
    plus = theta + np.array([eps])
    minus = theta - np.array([eps])
    numeric = (
        noisy_state_preparation_metrics(
            circuit,
            plus,
            target,
            noise_model,
            tjm_options,
            fixed_noise_maps=fixed_maps,
        )[0]
        - noisy_state_preparation_metrics(
            circuit,
            minus,
            target,
            noise_model,
            tjm_options,
            fixed_noise_maps=fixed_maps,
        )[0]
    ) / (2 * eps)
    assert contribution[0] == pytest.approx(numeric, abs=1e-6)


def test_normalized_noise_map_preserves_independent_update_phase() -> None:
    """Normalizing a realized map must not flip the trajectory update phase."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.41])
    target = np.array([np.cos(0.8 / 2), np.sin(0.8 / 2)], dtype=np.complex128)
    strength = 0.05
    drift = np.exp(-0.5 * strength) * np.eye(2, dtype=np.complex128)
    fixed_maps = [[KrotovNoiseMap(operators=((drift, (0,)),), normalized=True)]]
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": strength}])
    tjm_options = KrotovTJMOptions(num_trajectories=1)

    contribution, _loss_value, _mean_fidelity, _trajectories = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        noise_model,
        tjm_options,
        MPS(1),
        KrotovTruncation(),
        fixed_noise_maps=fixed_maps,
    )

    eps = 1e-6
    numeric = (
        noisy_state_preparation_metrics(
            circuit,
            theta + np.array([eps]),
            target,
            noise_model,
            tjm_options,
            fixed_noise_maps=fixed_maps,
        )[0]
        - noisy_state_preparation_metrics(
            circuit,
            theta - np.array([eps]),
            target,
            noise_model,
            tjm_options,
            fixed_noise_maps=fixed_maps,
        )[0]
    ) / (2 * eps)

    assert np.sign(contribution[0]) == np.sign(numeric)
    assert abs(contribution[0]) == pytest.approx(abs(numeric) * np.exp(-0.5 * strength), abs=1e-6)


def test_sampled_pauli_tjm_maps_replay_as_fixed_gradient() -> None:
    """Sampled normalized Pauli maps are stored as replayable physical maps."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.41])
    target = np.array([np.cos(0.8 / 2), np.sin(0.8 / 2)], dtype=np.complex128)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 2.0}])
    tjm_options = KrotovTJMOptions(num_trajectories=4, random_seed=11)
    truncation = KrotovTruncation()

    _contribution, _loss_value, _mean_fidelity, trajectories = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        noise_model,
        tjm_options,
        MPS(1),
        truncation,
    )
    fixed_maps = [trajectory.noise_maps for trajectory in trajectories]
    realized_maps = [maps[0] for maps in fixed_maps]

    assert any(noise_map.jump_process_index is not None for noise_map in realized_maps)
    assert all(not noise_map.normalized for noise_map in realized_maps)
    assert all(len(noise_map.operators) <= 1 for noise_map in realized_maps)

    contribution, _loss_value, _mean_fidelity, _trajectories = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        noise_model,
        tjm_options,
        MPS(1),
        truncation,
        fixed_noise_maps=fixed_maps,
    )

    eps = 1e-6
    numeric = (
        noisy_state_preparation_metrics(
            circuit,
            theta + np.array([eps]),
            target,
            noise_model,
            tjm_options,
            fixed_noise_maps=fixed_maps,
        )[0]
        - noisy_state_preparation_metrics(
            circuit,
            theta - np.array([eps]),
            target,
            noise_model,
            tjm_options,
            fixed_noise_maps=fixed_maps,
        )[0]
    ) / (2 * eps)
    assert contribution[0] == pytest.approx(numeric, abs=1e-6)


def test_noisy_expectation_loss_averages_readout_before_loss() -> None:
    """Noisy supervised readout applies loss after trajectory averaging."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.0])
    readout = KrotovReadout(observable=Observable("z", 0), loss="mse")
    x_gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    fixed_maps = [[KrotovNoiseMap()], [KrotovNoiseMap(operators=((x_gate, (0,)),))]]
    tjm_options = KrotovTJMOptions(num_trajectories=2)

    loss_value, zbar, trajectory_readouts = noisy_sample_loss(
        circuit,
        theta,
        np.array([]),
        1.0,
        readout,
        0.0,
        None,
        tjm_options,
        fixed_noise_maps=fixed_maps,
    )

    assert trajectory_readouts == pytest.approx([1.0, -1.0], abs=1e-12)
    assert zbar == pytest.approx(0.0, abs=1e-12)
    assert loss_value == pytest.approx(1.0, abs=1e-12)
    per_trajectory_loss_mean = np.mean([(z - 1.0) ** 2 for z in trajectory_readouts])
    assert per_trajectory_loss_mean == pytest.approx(2.0, abs=1e-12)
    assert loss_value != pytest.approx(per_trajectory_loss_mean)


def test_noisy_state_preparation_training_does_not_mutate_noise_model() -> None:
    """Noisy Krotov updates only circuit parameters, not noise configuration."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    target = np.array([np.cos(0.6 / 2), np.sin(0.6 / 2)], dtype=np.complex128)
    theta = np.array([0.1])
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.03}])
    original_strength = noise_model.processes[0]["strength"]
    original_matrix = noise_model.processes[0]["matrix"].copy()

    result = train_krotov_noisy_state_preparation_batch(
        circuit,
        target,
        noise_model,
        KrotovTJMOptions(num_trajectories=2, random_seed=5),
        initial_theta=theta,
        options=KrotovOptions(max_iterations=1, batch_step_size=0.1),
    )

    assert noise_model.processes[0]["strength"] == original_strength
    np.testing.assert_allclose(noise_model.processes[0]["matrix"], original_matrix)
    assert result.theta[0] != pytest.approx(theta[0])


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


def test_noisy_state_preparation_crn_monotonic_descent() -> None:
    """A noisy state preparation batch with CRN produces a monotonically increasing fidelity trace."""
    np.random.seed(42)
    num_qubits = 2
    statevec = np.random.randn(2**num_qubits) + 1j * np.random.randn(2**num_qubits)
    statevec /= np.linalg.norm(statevec)
    target_state = statevec

    circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, 1)
    initial_theta = np.random.randn(circuit.num_params) * 0.1

    processes = [{"name": "pauli_x", "sites": [i], "strength": 0.1} for i in range(num_qubits)]
    noise_model = NoiseModel(processes)

    tjm_options = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=True)
    options = KrotovOptions(max_iterations=10, batch_step_size=0.1, seed=123)

    result = train_krotov_noisy_state_preparation_batch(
        circuit, target_state, noise_model, tjm_options, initial_theta=initial_theta, options=options
    )

    fidelities = result.trace["fidelity"]
    
    # Assert monotonic increase
    for i in range(len(fidelities) - 1):
        assert fidelities[i + 1] >= fidelities[i] - 1e-12
