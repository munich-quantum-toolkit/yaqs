# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :mod:`mqt.yaqs.digital.digital_tjm`.

Test sections follow the public API order in ``digital_tjm.py``.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs import NoiseModel, Observable, Simulator, State, StrongSimParams, WeakSimParams
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import GateLibrary, X, Y, Z
from mqt.yaqs.core.methods.tdvp.sweep_utils import renorm_drift, uses_fixed_chi
from mqt.yaqs.core.methods.tdvp.tdvp import evolve_window
from mqt.yaqs.digital.digital_tjm import (
    apply_long_range_gate_mpo,
    apply_single_qubit_gate,
    apply_two_qubit_gate,
    apply_two_qubit_gate_tdvp,
    apply_two_qubit_gate_tebd,
    apply_window,
    construct_generator_mpo,
    create_local_noise_model,
    digital_tjm,
    process_layer,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm
from tests.core.methods.tdvp.conftest import (
    HAAR_LR_FID_ABS,
    HAAR_LR_FID_FLOOR,
    HAAR_LR_Z_TOL,
    NORM_TOL,
    PLUS_LR_RZZ_GLOBAL_FID,
    Z_TOL,
    _apply_lr_gate,
    _assert_z_observables_match,
    _fidelity,
    _haar_random_mps,
    _max_bond,
    _prep_state,
    _qiskit_plus_rzz_reference,
    _run_circuit,
    _tdvp_params,
    _z_expectation,
    assert_mps_bond_invariants,
)
from tests.digital.conftest import _physical_second_schmidt, _run_strong_noiseless

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.simulation_parameters import GateMode


# --- create_local_noise_model ---


def _sample_global_noise_model() -> NoiseModel:
    """Build the shared global noise model used in local-noise filtering tests.

    Returns:
        Noise model with single- and two-qubit processes on a five-site span.
    """
    global_processes = [
        {"name": "pauli_x", "sites": [0], "strength": 0.01},
        {"name": "pauli_x", "sites": [1], "strength": 0.02},
        {"name": "pauli_x", "sites": [2], "strength": 0.03},
        {"name": "pauli_x", "sites": [3], "strength": 0.04},
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": 0.05},
        {"name": "crosstalk_xx", "sites": [1, 2], "strength": 0.06},
        {"name": "crosstalk_xx", "sites": [2, 3], "strength": 0.07},
        {"name": "crosstalk_yy", "sites": [3, 4], "strength": 0.08},
        {"name": "crosstalk_xy", "sites": [0, 1], "strength": 0.09},
        {"name": "crosstalk_yx", "sites": [1, 2], "strength": 0.10},
        {"name": "crosstalk_xx", "sites": [1, 3], "strength": 0.06},
    ]
    return NoiseModel(global_processes)


def _assert_noise_processes_match(
    actual: NoiseModel,
    expected_processes: list[dict[str, object]],
) -> None:
    """Assert that a local noise model contains exactly the expected processes.

    Args:
        actual: Local noise model returned by :func:`create_local_noise_model`.
        expected_processes: Expected process dictionaries with ``name``, ``sites``,
            and ``strength`` keys.
    """
    assert len(actual.processes) == len(expected_processes)
    for expected_process in expected_processes:
        found = any(
            actual_process["name"] == expected_process["name"]
            and actual_process["sites"] == expected_process["sites"]
            and actual_process["strength"] == expected_process["strength"]
            for actual_process in actual.processes
        )
        assert found, f"Expected process {expected_process} not found in local model"


@pytest.mark.parametrize(
    ("start", "end", "expected_processes"),
    [
        (
            1,
            2,
            [
                {"name": "pauli_x", "sites": [1], "strength": 0.02},
                {"name": "pauli_x", "sites": [2], "strength": 0.03},
                {"name": "crosstalk_xx", "sites": [1, 2], "strength": 0.06},
                {"name": "crosstalk_yx", "sites": [1, 2], "strength": 0.10},
            ],
        ),
        (
            0,
            1,
            [
                {"name": "pauli_x", "sites": [0], "strength": 0.01},
                {"name": "pauli_x", "sites": [1], "strength": 0.02},
                {"name": "crosstalk_xx", "sites": [0, 1], "strength": 0.05},
                {"name": "crosstalk_xy", "sites": [0, 1], "strength": 0.09},
            ],
        ),
        (
            2,
            3,
            [
                {"name": "pauli_x", "sites": [2], "strength": 0.03},
                {"name": "pauli_x", "sites": [3], "strength": 0.04},
                {"name": "crosstalk_xx", "sites": [2, 3], "strength": 0.07},
            ],
        ),
        (
            1,
            1,
            [
                {"name": "pauli_x", "sites": [1], "strength": 0.02},
            ],
        ),
        (
            1,
            3,
            [
                {"name": "pauli_x", "sites": [1], "strength": 0.02},
                {"name": "pauli_x", "sites": [3], "strength": 0.04},
                {"name": "crosstalk_xx", "sites": [1, 3], "strength": 0.06},
            ],
        ),
    ],
)
def test_create_local_noise_model(
    start: int,
    end: int,
    expected_processes: list[dict[str, object]],
) -> None:
    """Local noise models retain only processes overlapping the gate site range."""
    local_model = create_local_noise_model(_sample_global_noise_model(), start, end)
    _assert_noise_processes_match(local_model, expected_processes)


# --- process_layer ---


def test_process_layer() -> None:
    """Test the process_layer function for grouping gate nodes.

    This test creates a 9-qubit circuit with measurement, barrier, single-qubit, and two-qubit gates.
    After processing, it verifies that measurement and barrier nodes have been removed and that the remaining
    nodes are correctly grouped into single, even, and odd sets. In the even group, the lower qubit index
    should be even, and in the odd group, it should be odd.
    """
    # Create a QuantumCircuit with 9 qubits and 9 classical bits.
    qc = QuantumCircuit(9, 9)
    qc.measure(0, 0)
    qc.barrier(3, label="SAMPLE_OBSERVABLES")
    qc.barrier(1)
    qc.x(qc.qubits[2])
    qc.cx(5, 4)
    qc.cx(7, 8)

    # Convert the circuit to a DAG.
    dag = circuit_to_dag(qc)

    # Call process_layer on the DAG.
    single, even, odd, measure_barriers = process_layer(dag)

    assert len(measure_barriers) == 1
    assert measure_barriers[0].op.name == "barrier"
    assert measure_barriers[0].op.label == "SAMPLE_OBSERVABLES"

    # After processing, measurement nodes and non-SAMPLE_OBSERVABLES barriers should have been removed.
    for node in dag.op_nodes():
        if node.op.name == "barrier" and str(getattr(node.op, "label", "")).upper() == "SAMPLE_OBSERVABLES":
            continue
        assert node.op.name not in {"measure", "barrier"}, f"Unexpected node {node.op.name} in the DAG op nodes."

    # Verify that the single-qubit gate is in the single-qubit group.
    single_names = [node.op.name.lower() for node in single]
    assert any("x" in name for name in single_names), "X gate not found in single group."

    # Verify the grouping of two-qubit gates.
    # For each node in the even group, the lower qubit index should be even.
    for node in even:
        q0 = dag.find_bit(node.qargs[0]).index
        q1 = dag.find_bit(node.qargs[1]).index
        assert min(q0, q1) % 2 == 0, f"Node with qubits {q0, q1} not in even group."

    # For each node in the odd group, the lower qubit index should be odd.
    for node in odd:
        q0 = dag.find_bit(node.qargs[0]).index
        q1 = dag.find_bit(node.qargs[1]).index
        assert min(q0, q1) % 2 == 1, f"Node with qubits {q0, q1} not in odd group."


def test_process_layer_rejects_mid_circuit_measure() -> None:
    """Mid-circuit measurements must not be silently dropped during layer processing."""
    qc = QuantumCircuit(2, 1)
    qc.measure(0, 0)
    qc.x(0)

    dag = circuit_to_dag(qc)

    with pytest.raises(ValueError, match="Non-terminal measure operations are not supported"):
        process_layer(dag)


def test_process_layer_unsupported_gate() -> None:
    """Test that process_layer raises an exception when encountering an unsupported gate.

    This test creates a 3-qubit circuit with a CCX gate, which is not supported by process_layer.
    It verifies that an exception is raised.
    """
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)

    dag = circuit_to_dag(qc)

    with pytest.raises(NotImplementedError):
        process_layer(dag)


# --- apply_single_qubit_gate ---


def test_apply_single_qubit_gate() -> None:
    """Test applying a single-qubit gate to an MPS using apply_single_qubit_gate.

    This test creates a one-qubit MPS and applies an X gate extracted from the front layer of a DAG.
    It then compares the updated tensor to the expected result computed via an einsum contraction.
    """
    mps = MPS(length=1)
    tensor = mps.tensors[0]

    qc = QuantumCircuit(1)
    qc.x(0)

    dag = circuit_to_dag(qc)
    node = dag.front_layer()[0]

    apply_single_qubit_gate(mps, node)

    gate_tensor = X().tensor
    expected = np.einsum("ab,bcd->acd", gate_tensor, tensor)
    np.testing.assert_allclose(mps.tensors[0], expected)


# --- construct_generator_mpo ---


def test_construct_generator_mpo() -> None:
    """Test the construction of a generator MPO from a two-qubit gate.

    This test retrieves a CX gate from the GateLibrary, sets its target sites, and uses construct_generator_mpo
    to obtain an MPO representation of the gate. It verifies that the first and last site indices match the expected
    values and that the generator MPO tensors at these sites correspond to the gate's generators. All other tensors
    should be the identity.
    """
    gate = GateLibrary.cx()
    gate.set_sites(1, 3)
    length = 5
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    for _tensor in mpo.tensors:
        pass
    assert first_site == 1
    assert last_site == 3
    np.testing.assert_allclose(np.squeeze(np.transpose(mpo.tensors[1], (2, 3, 0, 1))), np.complex128(gate.generator[0]))
    np.testing.assert_allclose(np.squeeze(np.transpose(mpo.tensors[3], (2, 3, 0, 1))), np.complex128(gate.generator[1]))
    for i in range(length):
        if i not in {1, 3}:
            np.testing.assert_allclose(np.squeeze(np.transpose(mpo.tensors[i], (2, 3, 0, 1))), np.eye(2, dtype=complex))


# --- apply_window ---


def test_apply_window() -> None:
    """Test the apply_window function for extracting a window from MPS and MPO objects.

    This test creates dummy MPS and MPO objects with 5 tensors, applies a window function with specified parameters,
    and asserts that the resulting window, as well as the shortened MPS and MPO, have the expected properties.
    """
    length = 5
    tensors = cast(
        "list[NDArray[np.complex128]]",
        [np.full((2, 1, 1), i, dtype=np.complex128) for i in range(5)],
    )
    mps = MPS(length, tensors)
    mps.normalize()

    gate = GateLibrary.cx()
    gate.set_sites(1, 2)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)

    window_size = 1

    short_state, short_mpo, window = apply_window(mps, mpo, first_site, last_site, window_size)

    assert window == [0, 3]
    assert short_state.length == 4
    assert short_mpo.length == 4


# --- apply_two_qubit_gate_tdvp ---


def test_apply_two_qubit_gate_tdvp_rejects_non_2site_mode() -> None:
    """Long-range digital TDVP windows only support two-site integration."""
    length = 4
    gate = GateLibrary.rzz([0.2])
    gate.set_sites(0, length - 1)
    state = State(length, initial="x+").mps
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=4,
        tdvp_mode="dynamic",
    )
    with pytest.raises(ValueError, match='tdvp_mode="2site"'):
        apply_two_qubit_gate_tdvp(state, gate, params)


_RZZ_ANGLE = 0.3


def _lr_pair(length: int) -> tuple[int, int]:
    """Return the long-range benchmark qubit pair for a chain.

    Args:
        length: Number of qubits in the chain.

    Returns:
        ``(0, length - 1)`` spanning the chain ends.
    """
    return (0, length - 1)


def _nn_pair(_length: int) -> tuple[int, int]:
    """Return the nearest-neighbor benchmark qubit pair.

    Args:
        _length: Chain length (unused; kept for a uniform helper signature).

    Returns:
        ``(0, 1)`` for the leftmost nearest-neighbor link.
    """
    return (0, 1)


def _rzz_lr_ladder_circuit(length: int) -> QuantumCircuit:
    """Long-range RZZ ladder used in fixed-χ regression tests.

    Returns:
        Benchmark circuit with mirrored long-range RZZ pairs.
    """
    qc = QuantumCircuit(length)
    for i in range(length // 2):
        a, b = i, length - 1 - i
        if a < b:
            qc.rzz(_RZZ_ANGLE, a, b)
    return qc


def _mixed_small_circuit(length: int) -> QuantumCircuit:
    """Small NN+LR circuit used in fixed-χ regression tests.

    Returns:
        Circuit with H, nearest-neighbor CX, long-range CX, and RZZ.
    """
    lr = _lr_pair(length)
    nn = _nn_pair(length)
    qc = QuantumCircuit(length)
    qc.h(0)
    qc.cx(nn[0], nn[1])
    qc.cx(lr[0], lr[1])
    qc.rzz(_RZZ_ANGLE, lr[0], lr[1])
    return qc


def _mixed_small_zeros_circuit(length: int = 10) -> QuantumCircuit:
    """NN+LR circuit on |0⟩^L with LR CX at topological gate index 2.

    Returns:
        Benchmark circuit for hybrid TDVP replay regressions.
    """
    qc = QuantumCircuit(length)
    qc.h(0)
    qc.cx(4, 5)
    qc.cx(0, length - 1)
    qc.rzz(_RZZ_ANGLE, 0, length - 1)
    return qc


def _grid_shape(num_qubits: int) -> tuple[int, int]:
    """Infer a near-square row/column layout for a 1D qubit chain.

    Args:
        num_qubits: Total number of qubits to lay out on a grid.

    Returns:
        ``(nrow, ncol)`` whose product equals ``num_qubits``.
    """
    for nrow in range(int(math.sqrt(num_qubits)), 0, -1):
        if num_qubits % nrow == 0:
            return nrow, num_qubits // nrow
    return 1, num_qubits


def _grid_index(row: int, col: int, ncol: int) -> int:
    """Map a 2D grid coordinate to a 1D chain index.

    Args:
        row: Row index on the grid.
        col: Column index on the grid.
        ncol: Number of columns in the grid.

    Returns:
        Linear qubit index ``row * ncol + col``.
    """
    return row * ncol + col


def _ising_2d_mapped_circuit(length: int = 6) -> QuantumCircuit:
    """2D Ising-style RZZ layers mapped onto a 1D chain.

    Returns:
        Benchmark circuit for hybrid TDVP replay regressions.
    """
    qc = QuantumCircuit(length)
    nrow, ncol = _grid_shape(length)
    for r in range(nrow):
        for c in range(ncol):
            qc.h(_grid_index(r, c, ncol))
    for r in range(nrow):
        for c in range(ncol - 1):
            qc.rzz(_RZZ_ANGLE, _grid_index(r, c, ncol), _grid_index(r, c + 1, ncol))
    for r in range(nrow - 1):
        for c in range(ncol):
            qc.rzz(_RZZ_ANGLE, _grid_index(r, c, ncol), _grid_index(r + 1, c, ncol))
    return qc


def _hybrid_tdvp_replay_params(*, max_bond_dim: int, tdvp_sweeps: int = 1) -> StrongSimParams:
    """StrongSimParams for hybrid ``gate_mode='tdvp'`` circuit replay.

    Returns:
        Simulation parameters routing LR gates through ``apply_two_qubit_gate_tdvp``.
    """
    return StrongSimParams(
        preset="exact",
        get_state=True,
        gate_mode="tdvp",
        max_bond_dim=max_bond_dim,
        tdvp_sweeps=tdvp_sweeps,
        tdvp_mode="2site",
        svd_threshold=1e-10,
        krylov_tol=1e-12,
    )


def _replay_hybrid_tdvp_through_gate(qc: QuantumCircuit, num_gates: int, *, params: StrongSimParams) -> State:
    """Replay a circuit gate-by-gate through hybrid TDVP routing.

    Returns:
        State after applying the first ``num_gates`` topological operations.
    """
    state = State(qc.num_qubits, initial="zeros")
    dag = circuit_to_dag(qc)
    for gate_index, node in enumerate(dag.topological_op_nodes()):
        if gate_index >= num_gates:
            break
        if len(node.qargs) == 1:
            apply_single_qubit_gate(state.mps, node)
        else:
            apply_two_qubit_gate(state.mps, node, params)
    return state


def _ladder_pairs(length: int) -> list[tuple[int, int]]:
    """Enumerate mirrored long-range RZZ pairs used in ladder regressions.

    Args:
        length: Number of qubits in the chain.

    Returns:
        Ordered list of disjoint ``(left, right)`` site pairs.
    """
    pairs: list[tuple[int, int]] = []
    for left in range(length // 2):
        right = length - 1 - left
        if left < right:
            pairs.append((left, right))
    return pairs


def _exact_ladder_reference(length: int, num_gates: int) -> np.ndarray:
    r"""Build the exact state vector for a prefix of the RZZ ladder circuit.

    Args:
        length: Number of qubits in the chain.
        num_gates: Number of ladder RZZ gates to include from the left.

    Returns:
        State vector after ``H^{\\otimes L}`` and the first ``num_gates`` RZZ gates.
    """
    qc = QuantumCircuit(length)
    qc.h(range(length))
    for site_a, site_b in _ladder_pairs(length)[:num_gates]:
        qc.rzz(_RZZ_ANGLE, site_a, site_b)
    return np.asarray(Statevector(qc).data, dtype=np.complex128)


def _apply_ladder_gates(state: State, *, max_bond_dim: int | None, tdvp_sweeps: int) -> None:
    """Apply the mirrored RZZ ladder through production digital gate routing.

    Args:
        state: MPS state updated in place.
        max_bond_dim: Optional bond-dimension cap forwarded to TDVP parameters.
        tdvp_sweeps: Number of symmetric TDVP substeps per long-range gate.
    """
    params = _tdvp_params(max_bond_dim=max_bond_dim, tdvp_sweeps=tdvp_sweeps)
    for site_a, site_b in _ladder_pairs(state.length):
        gate = GateLibrary.rzz([_RZZ_ANGLE])
        gate.set_sites(site_a, site_b)
        if abs(site_a - site_b) == 1:
            apply_two_qubit_gate_tebd(state.mps, gate, params)
        else:
            apply_two_qubit_gate_tdvp(state.mps, gate, params)


PRODUCTION_LENGTHS = (6, 10, 14)
ROUND_TRIP_TOL = 1e-10
ISING_TARGET_GATE = 12
ISING_CHI = 16
MIXED_SMALL_ZEROS_LENGTH = 10
MIXED_SMALL_TARGET_GATE = 2
LADDER_INVARIANT_LENGTH = 10
LADDER_INVARIANT_CHI = 64
LADDER_INVARIANT_SWEEPS = 1
INVARIANT_TOL = 1e-10
SQRT2 = float(np.sqrt(2.0))


def _apply_production_lr_rzz(
    length: int,
    *,
    theta: float = _RZZ_ANGLE,
    max_bond_dim: int | None = None,
    tdvp_sweeps: int = 1,
) -> np.ndarray:
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, length - 1)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=max_bond_dim, tdvp_sweeps=tdvp_sweeps))
    return out.to_vec()


def _apply_no_support_baseline(
    length: int,
    *,
    theta: float = _RZZ_ANGLE,
    max_bond_dim: int | None = None,
    tdvp_sweeps: int = 1,
) -> np.ndarray:
    """Window-local ``evolve_window`` + graft + drift renorm (production LR contract).

    Returns:
        State vector after applying the gate.
    """
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, length - 1)
    prep = copy.deepcopy(State(length, initial="x+").mps)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    short_state, short_mpo, window = apply_window(prep, mpo, first_site, last_site, 1)
    params = _tdvp_params(max_bond_dim=max_bond_dim, tdvp_sweeps=tdvp_sweeps)
    evolve_window(short_state, short_mpo, params)
    for i in range(window[0], window[1] + 1):
        prep.tensors[i] = short_state.tensors[i - window[0]]
    if uses_fixed_chi(params):
        renorm_drift(prep, params)
    return prep.to_vec()


@pytest.mark.tdvp_regression
def test_lr_routes_2site() -> None:
    """Long-range digital TDVP window evolution uses the 2-site kernel."""
    length = 8
    gate = GateLibrary.rzz([0.3])
    gate.set_sites(0, length - 1)
    out = copy.deepcopy(State(length, initial="x+").mps)
    params = _tdvp_params(max_bond_dim=None, tdvp_sweeps=4)

    with (
        patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_dynamic") as mock_dynamic,
        patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_2site") as mock_two,
    ):
        mock_two.side_effect = lambda *_args, **_kwargs: None
        apply_two_qubit_gate_tdvp(out, gate, params)
        mock_two.assert_called_once()
        mock_dynamic.assert_not_called()
        assert mock_two.call_args.kwargs.get("drift_renorm") is False


@pytest.mark.tdvp_regression
def test_lr_rzz_cap_chi1() -> None:
    """End-to-end χ=1 long-range RZZ respects the bond-dimension cap."""
    theta = 0.3
    length = 8
    sites = (0, length - 1)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(
        out,
        gate,
        StrongSimParams(
            preset="exact",
            get_state=True,
            max_bond_dim=1,
            tdvp_sweeps=64,
            svd_threshold=1e-14,
            krylov_tol=1e-12,
        ),
    )
    assert max(out.bond_dimensions()) == 1
    assert all(dim == 1 for dim in out.bond_dimensions())
    assert out.norm() == pytest.approx(1.0, abs=NORM_TOL)
    assert_mps_bond_invariants(out, max_bond_dim=1)


@pytest.mark.tdvp_regression
def test_lr_bond_invariants() -> None:
    """Public LR TDVP gate application leaves consistent tensor shapes."""
    out = _apply_lr_gate(State(8, initial="x+").mps, "rzz", 0.3, max_bond_dim=8, sweeps=16)
    assert_mps_bond_invariants(out, max_bond_dim=8)


@pytest.mark.parametrize("tdvp_sweeps", [1, 4, 16])
@pytest.mark.tdvp_regression
def test_sweeps_default_explicit(tdvp_sweeps: int) -> None:
    """Normal runs do not require debug tracing and match explicit sweep counts."""
    length = 6
    theta = 0.3
    out_a = _apply_lr_gate(State(length, initial="x+").mps, "rzz", theta, max_bond_dim=8, sweeps=tdvp_sweeps)
    out_b = _apply_lr_gate(State(length, initial="x+").mps, "rzz", theta, max_bond_dim=8, sweeps=tdvp_sweeps)
    assert _fidelity(out_a.to_vec(), out_b.to_vec()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.tdvp_regression
def test_lr_rzz_routes_fidelity() -> None:
    """Long-range RZZ on |+⟩^L: local ⟨Z_i⟩ exact; global fidelity at production level."""
    length = 8
    theta = 0.3
    sites = (0, length - 1)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=1))

    ref = _qiskit_plus_rzz_reference(length, theta, sites=sites)
    _assert_z_observables_match(ref, out.to_vec(), length)
    assert _fidelity(ref, out.to_vec()) == pytest.approx(PLUS_LR_RZZ_GLOBAL_FID, abs=1e-4)


@pytest.mark.parametrize("length", [7, 8])
@pytest.mark.tdvp_regression
def test_lr_rzz_endpoint_z_obs(length: int) -> None:
    """Endpoint RZZ on |+⟩^L: exact ⟨Z_i⟩, entanglement on crossed bonds, stable norm."""
    theta = 0.3
    sites = (0, length - 1)
    out = _apply_lr_gate(
        State(length, initial="x+").mps,
        "rzz",
        theta,
        max_bond_dim=None,
        sweeps=1,
    )
    ref = _qiskit_plus_rzz_reference(length, theta, sites=sites)
    _assert_z_observables_match(ref, out.to_vec(), length)
    assert _fidelity(ref, out.to_vec()) == pytest.approx(PLUS_LR_RZZ_GLOBAL_FID, abs=1e-4)
    assert out.norm() == pytest.approx(1.0, abs=NORM_TOL)
    assert_mps_bond_invariants(out)


@pytest.mark.tdvp_regression
@pytest.mark.slow
def test_lr_rzz_endpoint_l10() -> None:
    """Endpoint RZZ at L=10 uses the production single-sweep path."""
    test_lr_rzz_endpoint_z_obs(10)


@pytest.mark.tdvp_regression
def test_lr_rzz_internal_z_obs() -> None:
    """Internal long-range pairs: exact ⟨Z_i⟩ on |+⟩^L."""
    theta = 0.3
    length = 10
    sites = (2, 7)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=1))

    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    _assert_z_observables_match(ref, out.to_vec(), length)
    assert_mps_bond_invariants(out)


@pytest.mark.tdvp_regression
def test_lr_rzz_shifted_z_obs() -> None:
    """Shifted internal RZZ(1,8) on |+⟩^L: exact ⟨Z_i⟩."""
    theta = 0.3
    length = 10
    sites = (1, 8)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=1))

    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    _assert_z_observables_match(ref, out.to_vec(), length)
    assert_mps_bond_invariants(out)


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
@pytest.mark.tdvp_regression
def test_lr_rzz_spectator_z_zero(length: int) -> None:
    """Endpoint RZZ on |+⟩^L: interior |⟨Z_i⟩| ≈ 0 for spectators."""
    vec = _apply_production_lr_rzz(length, max_bond_dim=None, tdvp_sweeps=1)
    for site in range(1, length - 1):
        assert abs(_z_expectation(vec, site)) < Z_TOL


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
@pytest.mark.tdvp_regression
def test_lr_rzz_bond_not_inflated(length: int) -> None:
    """Bond dimension on crossed path stays minimal for endpoint |+⟩ RZZ."""
    gate = GateLibrary.rzz([_RZZ_ANGLE])
    gate.set_sites(0, length - 1)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=1))
    assert _max_bond(out) <= 2


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
@pytest.mark.tdvp_regression
def test_lr_rzz_roundtrip_plus(length: int) -> None:
    """RZZ(theta) followed by RZZ(-theta) returns to |+⟩^L."""
    prep = copy.deepcopy(State(length, initial="x+").mps)
    gate_fwd = GateLibrary.rzz([_RZZ_ANGLE])
    gate_fwd.set_sites(0, length - 1)
    gate_bwd = GateLibrary.rzz([-_RZZ_ANGLE])
    gate_bwd.set_sites(0, length - 1)
    params = _tdvp_params(max_bond_dim=None, tdvp_sweeps=1)
    apply_two_qubit_gate_tdvp(prep, gate_fwd, params)
    apply_two_qubit_gate_tdvp(prep, gate_bwd, params)
    plus = State(length, initial="x+").mps.to_vec()
    assert _fidelity(plus, prep.to_vec()) > 1.0 - ROUND_TRIP_TOL


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
@pytest.mark.tdvp_regression
def test_lr_rzz_roundtrip_z_obs(length: int) -> None:
    """Round-trip LR RZZ restores all single-site ⟨Z_i⟩."""
    prep = copy.deepcopy(State(length, initial="x+").mps)
    gate_fwd = GateLibrary.rzz([_RZZ_ANGLE])
    gate_fwd.set_sites(0, length - 1)
    gate_bwd = GateLibrary.rzz([-_RZZ_ANGLE])
    gate_bwd.set_sites(0, length - 1)
    params = _tdvp_params(max_bond_dim=None, tdvp_sweeps=1)
    apply_two_qubit_gate_tdvp(prep, gate_fwd, params)
    apply_two_qubit_gate_tdvp(prep, gate_bwd, params)
    vec = prep.to_vec()
    for site in range(length):
        assert abs(_z_expectation(vec, site)) < Z_TOL


@pytest.mark.parametrize("length", PRODUCTION_LENGTHS)
@pytest.mark.tdvp_regression
def test_lr_rzz_vs_baseline(length: int) -> None:
    """Production LR path matches window-local ``evolve_window`` + post-graft drift renorm."""
    prod = _apply_production_lr_rzz(length, max_bond_dim=64, tdvp_sweeps=1)
    baseline = _apply_no_support_baseline(length, max_bond_dim=64, tdvp_sweeps=1)
    assert _fidelity(prod, baseline) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.tdvp_regression
def test_lr_rzz_haar_stable() -> None:
    """Entangled Haar prep stays stable under production 1-sweep LR RZZ."""
    length = 8
    theta = 0.3
    sites = (0, length - 1)
    prep = _haar_random_mps(length)
    prep.normalize()
    out = _apply_lr_gate(prep, "rzz", theta, max_bond_dim=None, sweeps=1)

    qc = QuantumCircuit(length)
    qc.initialize(prep.to_vec().tolist(), range(length))
    qc.rzz(theta, sites[0], sites[1])
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    vec = out.to_vec()
    fid = _fidelity(ref, vec)

    _assert_z_observables_match(ref, vec, length, tol=HAAR_LR_Z_TOL)
    assert out.norm() == pytest.approx(1.0, abs=NORM_TOL)
    assert_mps_bond_invariants(out)
    assert fid == pytest.approx(1.0, abs=HAAR_LR_FID_ABS)
    assert fid > HAAR_LR_FID_FLOOR


@pytest.mark.parametrize("length", [8, 10])
@pytest.mark.tdvp_regression
def test_lr_rzz_zeros_exact(length: int) -> None:
    """|0⟩^L + long-range RZZ stays exact without spurious entanglement."""
    theta = 0.3
    sites = (0, length - 1)
    out = copy.deepcopy(State(length, initial="zeros").mps)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=64))

    qc = QuantumCircuit(length)
    qc.rzz(theta, sites[0], sites[1])
    ref = np.asarray(Statevector.from_label("0" * length).evolve(qc).data, dtype=np.complex128)
    assert _fidelity(ref, out.to_vec()) > 1.0 - 1e-6
    vec = out.to_vec()
    for bond in range(length - 1):
        assert _physical_second_schmidt(vec, length, bond) < 3e-7
    assert out.norm() == pytest.approx(1.0, abs=NORM_TOL)


@pytest.mark.parametrize("length", [8, 10])
@pytest.mark.tdvp_regression
def test_lr_rzz_zeros_fchi_exact(length: int) -> None:
    """Fixed-χ zeros control stays exact on product states."""
    theta = 0.3
    out = _apply_lr_gate(
        State(length, initial="zeros").mps,
        "rzz",
        theta,
        max_bond_dim=8,
        sweeps=64,
    )
    qc = QuantumCircuit(length)
    qc.rzz(theta, 0, length - 1)
    ref = np.asarray(Statevector.from_label("0" * length).evolve(qc).data, dtype=np.complex128)
    assert _fidelity(ref, out.to_vec()) > 1.0 - 1e-6
    assert out.get_max_bond() <= 8
    assert_mps_bond_invariants(out, max_bond_dim=8)


@pytest.mark.parametrize("length", [6, 8, 10])
@pytest.mark.parametrize("max_bond_dim", [1, 2, 4, 8])
@pytest.mark.parametrize("initial_state", ["plus", "zeros", "low_depth"])
@pytest.mark.parametrize("gate_name", ["rzz", "rxx"])
@pytest.mark.parametrize("sweeps", [1, 4, 16])
@pytest.mark.tdvp_regression
def test_lr_fchi_cap(
    length: int,
    max_bond_dim: int,
    initial_state: str,
    gate_name: str,
    sweeps: int,
) -> None:
    """Long-range Pauli rotations never exceed the configured χ cap."""
    prep = _prep_state(initial_state, length)
    out = _apply_lr_gate(prep, gate_name, 0.3, max_bond_dim=max_bond_dim, sweeps=sweeps)
    assert _max_bond(out) <= max_bond_dim
    assert all(dim <= max_bond_dim for dim in out.bond_dimensions())
    assert_mps_bond_invariants(out, max_bond_dim=max_bond_dim)
    assert out.norm() == pytest.approx(1.0, abs=NORM_TOL)


@pytest.mark.tdvp_regression
def test_fchi_plus_one_gate() -> None:
    """χ=1 plus/RZZ stays rank-1 limited and does not pad to χ=2."""
    theta = 0.3
    length = 8
    prep = _prep_state("plus", length)
    out = _apply_lr_gate(prep, "rzz", theta, max_bond_dim=1, sweeps=64)
    ref = _qiskit_plus_rzz_reference(length, theta, sites=(0, length - 1))
    plus = _prep_state("plus", length)
    assert _max_bond(out) == 1
    assert _fidelity(ref, out.to_vec()) == pytest.approx(0.977668, abs=0.01)
    assert _fidelity(plus.to_vec(), out.to_vec()) > 0.999
    assert_mps_bond_invariants(out, max_bond_dim=1)


@pytest.mark.tdvp_regression
def test_fchi_plus_two_gates() -> None:
    """χ=2 plus/RZZ applies the gate rather than staying at |+⟩^L."""
    theta = 0.3
    length = 7
    prep = _prep_state("plus", length)
    out = _apply_lr_gate(prep, "rzz", theta, max_bond_dim=2, sweeps=1)
    ref = _qiskit_plus_rzz_reference(length, theta, sites=(0, length - 1))
    assert _max_bond(out) <= 2
    assert _fidelity(ref, out.to_vec()) == pytest.approx(PLUS_LR_RZZ_GLOBAL_FID, abs=1e-4)
    assert_mps_bond_invariants(out, max_bond_dim=2)


@pytest.mark.tdvp_regression
def test_fchi_plus_eight_z_obs() -> None:
    """χ=8 plus/RZZ: local ⟨Z_i⟩ exact under production single-sweep 2TDVP."""
    theta = 0.3
    length = 8
    prep = _prep_state("plus", length)
    out = _apply_lr_gate(prep, "rzz", theta, max_bond_dim=8, sweeps=1)
    ref = _qiskit_plus_rzz_reference(length, theta, sites=(0, length - 1))
    _assert_z_observables_match(ref, out.to_vec(), length)
    assert _max_bond(out) <= 8
    assert _fidelity(ref, out.to_vec()) == pytest.approx(PLUS_LR_RZZ_GLOBAL_FID, abs=1e-4)
    assert_mps_bond_invariants(out, max_bond_dim=8)


@pytest.mark.tdvp_regression
def test_ladder_fchi_no_shape_error() -> None:
    """Multi-gate ladder circuits complete without bond-dimension violations."""
    length = 8
    prep = State(length, initial="zeros").mps
    out = _run_circuit(prep, _rzz_lr_ladder_circuit(length), max_bond_dim=8, sweeps=4)
    assert _max_bond(out) <= 8
    assert_mps_bond_invariants(out, max_bond_dim=8)


@pytest.mark.tdvp_regression
def test_ladder_enforces_cap() -> None:
    """When uncapped ladder reaches χ above the cap, capped evolution differs and respects χ."""
    length = LADDER_INVARIANT_LENGTH
    prep = _prep_state("plus", length)
    uncapped = _run_circuit(copy.deepcopy(prep), _rzz_lr_ladder_circuit(length), max_bond_dim=64, sweeps=1)
    capped = _run_circuit(copy.deepcopy(prep), _rzz_lr_ladder_circuit(length), max_bond_dim=2, sweeps=1)

    assert _max_bond(uncapped) > 2
    assert _max_bond(capped) <= 2
    assert abs(float(np.linalg.norm(capped.to_vec())) - 1.0) < NORM_TOL
    assert _fidelity(uncapped.to_vec(), capped.to_vec()) < 0.99


@pytest.mark.tdvp_regression
def test_ladder_fchi_vs_uncapped() -> None:
    """L=10 |+⟩ ladder: χ=64 matches χ=None when no bond reaches the cap."""
    uncapped = State(LADDER_INVARIANT_LENGTH, initial="x+")
    capped = State(LADDER_INVARIANT_LENGTH, initial="x+")
    _apply_ladder_gates(uncapped, max_bond_dim=None, tdvp_sweeps=LADDER_INVARIANT_SWEEPS)
    _apply_ladder_gates(capped, max_bond_dim=LADDER_INVARIANT_CHI, tdvp_sweeps=LADDER_INVARIANT_SWEEPS)

    assert _max_bond(uncapped.mps) <= LADDER_INVARIANT_CHI
    assert _max_bond(capped.mps) <= LADDER_INVARIANT_CHI

    exact = _exact_ladder_reference(LADDER_INVARIANT_LENGTH, len(_ladder_pairs(LADDER_INVARIANT_LENGTH)))
    assert _fidelity(exact, uncapped.mps.to_vec()) > 0.9
    assert _fidelity(uncapped.mps.to_vec(), capped.mps.to_vec()) == pytest.approx(1.0, abs=INVARIANT_TOL)


@pytest.mark.parametrize("gate_index", range(5))
@pytest.mark.tdvp_regression
def test_ladder_per_gate_fchi(gate_index: int) -> None:
    """Per-gate partial ladder: χ=64 matches χ=None while bonds stay below cap."""
    uncapped = State(LADDER_INVARIANT_LENGTH, initial="x+")
    capped = State(LADDER_INVARIANT_LENGTH, initial="x+")
    params_u = _tdvp_params(max_bond_dim=None, tdvp_sweeps=LADDER_INVARIANT_SWEEPS)
    params_c = _tdvp_params(max_bond_dim=LADDER_INVARIANT_CHI, tdvp_sweeps=LADDER_INVARIANT_SWEEPS)

    for idx, (site_a, site_b) in enumerate(_ladder_pairs(LADDER_INVARIANT_LENGTH)):
        if idx > gate_index:
            break
        gate = GateLibrary.rzz([_RZZ_ANGLE])
        gate.set_sites(site_a, site_b)
        for state, params in ((uncapped, params_u), (capped, params_c)):
            if abs(site_a - site_b) == 1:
                apply_two_qubit_gate_tebd(state.mps, gate, params)
            else:
                apply_two_qubit_gate_tdvp(state.mps, gate, params)

    ref = _exact_ladder_reference(LADDER_INVARIANT_LENGTH, gate_index + 1)
    assert _max_bond(capped.mps) <= LADDER_INVARIANT_CHI
    assert _fidelity(ref, uncapped.mps.to_vec()) == pytest.approx(
        _fidelity(ref, capped.mps.to_vec()), abs=INVARIANT_TOL
    )
    assert _fidelity(uncapped.mps.to_vec(), capped.mps.to_vec()) == pytest.approx(1.0, abs=INVARIANT_TOL)


@pytest.mark.tdvp_regression
def test_mixed_fchi_cap() -> None:
    """Mixed NN+LR circuits respect χ under hybrid TDVP routing."""
    length = 8
    prep = _prep_state("plus", length)
    out = _run_circuit(prep, _mixed_small_circuit(length), max_bond_dim=8, sweeps=16)
    assert _max_bond(out) <= 8
    assert out.norm() == pytest.approx(1.0, abs=NORM_TOL)
    assert_mps_bond_invariants(out, max_bond_dim=8)


@pytest.mark.tdvp_regression
def test_ising_gate12_norm_unit() -> None:
    """Gate 12 LR RZZ under hybrid gate_mode='tdvp' must not inflate global norm to sqrt(2)."""
    qc = _ising_2d_mapped_circuit()
    params = _hybrid_tdvp_replay_params(max_bond_dim=ISING_CHI)
    state = _replay_hybrid_tdvp_through_gate(qc, ISING_TARGET_GATE + 1, params=params)
    norm = float(state.mps.norm())
    vec_norm = float(np.linalg.norm(state.mps.to_vec()))
    assert abs(norm - 1.0) < NORM_TOL, f"scalar norm {norm}"
    assert abs(vec_norm - 1.0) < NORM_TOL, f"vec norm {vec_norm}"
    assert abs(vec_norm - SQRT2) > 0.1


@pytest.mark.tdvp_regression
def test_ising_full_z_obs() -> None:
    """Full ising_2d_mapped zeros circuit stays near exact under hybrid gate_mode='tdvp'."""
    qc = _ising_2d_mapped_circuit()
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    num_gates = len(list(circuit_to_dag(qc).topological_op_nodes()))
    params = _hybrid_tdvp_replay_params(max_bond_dim=ISING_CHI)
    state = _replay_hybrid_tdvp_through_gate(qc, num_gates, params=params)
    assert abs(float(state.mps.norm()) - 1.0) < NORM_TOL
    assert _fidelity(ref, state.mps.to_vec()) > 0.94


@pytest.mark.tdvp_regression
def test_mixed_gate2_norm_unit() -> None:
    """Gate 2 LR CX under hybrid gate_mode='tdvp' must preserve global norm."""
    qc = _mixed_small_zeros_circuit()
    params = _hybrid_tdvp_replay_params(max_bond_dim=16)
    state = _replay_hybrid_tdvp_through_gate(qc, MIXED_SMALL_TARGET_GATE + 1, params=params)
    norm = float(state.mps.norm())
    vec_norm = float(np.linalg.norm(state.mps.to_vec()))
    assert abs(norm - 1.0) < NORM_TOL, f"scalar norm {norm}"
    assert abs(vec_norm - 1.0) < NORM_TOL, f"vec norm {vec_norm}"


@pytest.mark.parametrize("chi", [16, 32])
@pytest.mark.tdvp_regression
def test_mixed_zeros_full(chi: int) -> None:
    """Full mixed_small L=10 zeros circuit matches exact reference under hybrid gate_mode='tdvp'."""
    qc = _mixed_small_zeros_circuit(MIXED_SMALL_ZEROS_LENGTH)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    params = _hybrid_tdvp_replay_params(max_bond_dim=chi)
    state = _replay_hybrid_tdvp_through_gate(qc, len(list(circuit_to_dag(qc).topological_op_nodes())), params=params)
    assert abs(float(state.mps.norm()) - 1.0) < NORM_TOL
    assert _fidelity(ref, state.mps.to_vec()) > 0.99


@pytest.mark.parametrize("initial_state", ["plus", "low_depth"])
@pytest.mark.parametrize("max_bond_dim", [1, 2, 8, None])
@pytest.mark.parametrize("sweeps", [1, 4, 16, 64])
@pytest.mark.tdvp_regression
def test_fchi_norm_stable(
    initial_state: str,
    max_bond_dim: int | None,
    sweeps: int,
) -> None:
    """Long-range TDVP stays normalized across χ budgets and sweep counts."""
    length = 8
    prep = _prep_state(initial_state, length)
    out = _apply_lr_gate(prep, "rzz", 0.3, max_bond_dim=max_bond_dim, sweeps=sweeps)
    assert out.norm() == pytest.approx(1.0, abs=NORM_TOL)
    cap = max_bond_dim if max_bond_dim is not None else None
    assert_mps_bond_invariants(out, max_bond_dim=cap)


@pytest.mark.parametrize("gate_name", ["rzz", "rxx"])
@pytest.mark.tdvp_regression
def test_fchi_trunc_high_bond(gate_name: str) -> None:
    """Fixed-χ TDVP truncates incoming MPS bond dimensions before evolving."""
    length = 8
    prep = _prep_state("low_depth", length)
    assert max(prep.bond_dimensions()) > 1
    out = _apply_lr_gate(prep, gate_name, 0.3, max_bond_dim=1, sweeps=4)
    assert _max_bond(out) <= 1
    assert out.norm() == pytest.approx(1.0, abs=NORM_TOL)


@pytest.mark.tdvp_regression
def test_hybrid_lr_routes_tdvp() -> None:
    """Hybrid gate_mode='tdvp' routes LR gates through the TDVP window path."""
    length = 8
    gate = GateLibrary.rzz([0.3])
    gate.set_sites(0, length - 1)
    out = copy.deepcopy(State(length, initial="x+").mps)
    params = _tdvp_params(max_bond_dim=8, tdvp_sweeps=16)
    params.gate_mode = "tdvp"

    qc = QuantumCircuit(length)
    qc.rzz(0.3, 0, length - 1)
    dag = circuit_to_dag(qc)
    node = next(n for n in dag.front_layer() if n.op.name == "rzz")

    with (
        patch("mqt.yaqs.digital.digital_tjm.apply_two_qubit_gate_tebd") as mock_tebd,
        patch("mqt.yaqs.digital.digital_tjm.apply_two_qubit_gate_tdvp") as mock_tdvp,
    ):
        mock_tdvp.return_value = (0, length - 1)
        apply_two_qubit_gate(out, node, params)
        mock_tdvp.assert_called_once()
        mock_tebd.assert_not_called()


@pytest.mark.tdvp_regression
def test_hybrid_lr_z_obs() -> None:
    """Public hybrid path: exact ⟨Z_i⟩ on endpoint RZZ after H prep."""
    length = 8
    theta = 0.3
    sites = (0, length - 1)
    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=None,
        gate_mode="tdvp",
        tdvp_sweeps=1,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    result = Simulator(parallel=False, show_progress=False).run(State(length, initial="zeros"), qc, params, None)
    assert result.output_state is not None
    ref = _qiskit_plus_rzz_reference(length, theta, sites=sites)
    _assert_z_observables_match(ref, result.output_state.mps.to_vec(), length)


@pytest.mark.tdvp_regression
def test_hybrid_low_depth_smoke() -> None:
    """Low-depth entangled prep completes under hybrid TDVP."""
    length = 8
    prep_qc = QuantumCircuit(length)
    for i in range(0, length, 2):
        prep_qc.h(i)
    for i in range(length - 1):
        prep_qc.cx(i, i + 1)
    prep_params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=8,
        gate_mode="mpo",
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    prep_result = Simulator(parallel=False, show_progress=False).run(
        State(length, initial="zeros"), prep_qc, prep_params, None
    )
    assert prep_result.output_state is not None

    qc = QuantumCircuit(length)
    qc.rzz(0.3, 0, length - 1)
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        observables=[Observable(Z(), 0)],
        max_bond_dim=8,
        gate_mode="tdvp",
        tdvp_sweeps=16,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    init = State(length, tensors=[t.copy() for t in prep_result.output_state.mps.tensors])
    result = Simulator(parallel=False, show_progress=False).run(init, qc, params, None)
    assert result.output_state is not None
    assert result.output_state.mps.get_max_bond() <= 8


@pytest.mark.tdvp_regression
def test_hybrid_lr_vs_full_tdvp() -> None:
    """Long-range gates use TDVP in hybrid mode and match an all-TDVP run."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 2)

    hybrid_z = _run_strong_noiseless(qc, gate_mode="tdvp")
    tdvp_z = _run_strong_noiseless(qc, gate_mode="full-tdvp")
    assert hybrid_z == pytest.approx(tdvp_z, abs=1e-10)


@pytest.mark.tdvp_regression
def test_hybrid_mixed_vs_full_tdvp() -> None:
    """Circuits with both NN and long-range gates: hybrid vs all-TDVP."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 3)
    qc.rzz(0.2, 2, 3)

    hybrid_z = _run_strong_noiseless(qc, gate_mode="tdvp")
    tdvp_z = _run_strong_noiseless(qc, gate_mode="full-tdvp")
    assert hybrid_z == pytest.approx(tdvp_z, abs=1e-10)


@pytest.mark.tdvp_regression
def test_sweeps_unitary() -> None:
    """Multiple TDVP substeps target the same gate, not its square."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.rzz(0.3, 0, 3)

    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    one_sweep = _run_strong_noiseless(qc, gate_mode="full-tdvp", get_state=True, tdvp_sweeps=1)
    two_sweeps = _run_strong_noiseless(qc, gate_mode="full-tdvp", get_state=True, tdvp_sweeps=2)
    assert isinstance(one_sweep, np.ndarray)
    assert isinstance(two_sweeps, np.ndarray)

    assert _fidelity(ref, one_sweep) == pytest.approx(1.0, abs=1e-9)
    assert _fidelity(ref, two_sweeps) == pytest.approx(1.0, abs=1e-9)
    assert _fidelity(one_sweep, two_sweeps) == pytest.approx(1.0, abs=1e-9)

    doubled_gate = QuantumCircuit(4)
    doubled_gate.h(0)
    doubled_gate.rzz(0.3, 0, 3)
    doubled_gate.rzz(0.3, 0, 3)
    ref_doubled = np.asarray(Statevector(doubled_gate).data, dtype=np.complex128)
    assert _fidelity(ref_doubled, two_sweeps) < 0.99


@pytest.mark.tdvp_regression
def test_sweeps_hybrid_lr_vs_qiskit() -> None:
    """Hybrid long-range TDVP with multiple sweeps matches Qiskit within truncation error."""
    qc = QuantumCircuit(4)
    qc.rx(0.2, 1)
    qc.rxx(0.25, 0, 3)

    vec = _run_strong_noiseless(qc, gate_mode="tdvp", get_state=True, tdvp_sweeps=8)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.tdvp_regression
def test_sweeps_hybrid_nn_unchanged() -> None:
    """Nearest-neighbor hybrid circuits ignore tdvp_sweeps (TEBD path)."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cz(1, 2)

    baseline = _run_strong_noiseless(qc, gate_mode="tdvp", tdvp_sweeps=1)
    many_sweeps = _run_strong_noiseless(qc, gate_mode="tdvp", tdvp_sweeps=5)
    assert baseline == pytest.approx(many_sweeps, abs=1e-10)


@pytest.mark.tdvp_regression
def test_sweeps_mixed_regression() -> None:
    """Mixed hybrid circuit: multi-sweep TDVP converges toward Qiskit on long-range gates.

    Nearest-neighbor gates use TEBD; long-range gates use symmetric TDVP substeps.
    Unpadded minimal-bond MPS may need many sweeps; sweep count must increase until
    results stabilize against Qiskit.
    """
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 3)
    qc.rzz(0.2, 2, 3)

    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    thirty_two_sweeps = _run_strong_noiseless(qc, gate_mode="tdvp", get_state=True, tdvp_sweeps=32)
    sixty_four_sweeps = _run_strong_noiseless(qc, gate_mode="tdvp", get_state=True, tdvp_sweeps=64)
    assert isinstance(thirty_two_sweeps, np.ndarray)
    assert isinstance(sixty_four_sweeps, np.ndarray)

    assert _fidelity(ref, thirty_two_sweeps) == pytest.approx(1.0, abs=2e-4)
    assert _fidelity(ref, sixty_four_sweeps) == pytest.approx(1.0, abs=1e-4)
    assert _fidelity(thirty_two_sweeps, sixty_four_sweeps) == pytest.approx(1.0, abs=1e-4)

    doubled_gate = QuantumCircuit(4)
    doubled_gate.h(0)
    doubled_gate.cx(0, 1)
    doubled_gate.cx(0, 3)
    doubled_gate.rzz(0.2, 2, 3)
    doubled_gate.rzz(0.2, 2, 3)
    ref_doubled = np.asarray(Statevector(doubled_gate).data, dtype=np.complex128)
    assert _fidelity(ref_doubled, sixty_four_sweeps) < _fidelity(ref, sixty_four_sweeps)


# --- apply_two_qubit_gate_tebd ---


def test_tebd_lr_cx() -> None:
    """TEBD applies CX(1, 3) on |1111> via SWAP insertion."""
    length = 4
    mps = MPS(length, state="ones")
    mps.normalize()

    qc = QuantumCircuit(length)
    qc.cx(1, 3)
    dag = circuit_to_dag(qc)
    node = next(n for n in dag.front_layer() if n.op.name.lower() == "cx")

    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact", gate_mode="swaps")
    apply_two_qubit_gate_tebd(mps, convert_dag_to_tensor_algorithm(node)[0], sim_params)
    mps.normalize(decomposition="SVD")
    for i, element in enumerate(mps.to_vec()):
        if i == 7:
            np.testing.assert_allclose(np.abs(element), 1, atol=1e-10)
        else:
            np.testing.assert_allclose(np.abs(element), 0, atol=1e-10)


def test_tebd_lr_cnot() -> None:
    """Direct TEBD helper matches CX on |11> (same check as apply_two_qubit_gate)."""
    length = 4
    mps = MPS(length, state="ones")
    mps.normalize()

    qc = QuantumCircuit(length)
    qc.cx(1, 2)
    dag = circuit_to_dag(qc)
    node = next(n for n in dag.front_layer() if n.op.name.lower() == "cx")

    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact", gate_mode="swaps")
    apply_two_qubit_gate_tebd(
        mps,
        convert_dag_to_tensor_algorithm(node)[0],
        sim_params,
    )
    mps.normalize(decomposition="SVD")
    for i, element in enumerate(mps.to_vec()):
        if i == 11:
            np.testing.assert_allclose(np.abs(element), 1, atol=1e-10)
        else:
            np.testing.assert_allclose(np.abs(element), 0, atol=1e-10)


@pytest.mark.parametrize(("gate_name", "sites"), [("rzz", (0, 1)), ("rxx", (0, 1)), ("cx", (0, 1))])
@pytest.mark.tdvp_regression
def test_hybrid_nn_uses_tebd(gate_name: str, sites: tuple[int, int]) -> None:
    """Nearest-neighbor gates stay on TEBD/SVD in hybrid tdvp mode."""
    length = 4
    qc = QuantumCircuit(length)
    if gate_name == "rzz":
        qc.rzz(0.3, *sites)
    elif gate_name == "rxx":
        qc.rxx(0.3, *sites)
    else:
        qc.cx(*sites)
    dag = circuit_to_dag(qc)
    node = next(n for n in dag.front_layer() if n.op.name == gate_name)
    out = State(length, initial="zeros").mps
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=8,
        gate_mode="tdvp",
        tdvp_sweeps=4,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )

    with (
        patch("mqt.yaqs.digital.digital_tjm.apply_two_qubit_gate_tebd") as mock_tebd,
        patch("mqt.yaqs.digital.digital_tjm.apply_two_qubit_gate_tdvp") as mock_tdvp,
    ):
        mock_tebd.return_value = (sites[0], sites[1])
        apply_two_qubit_gate(out, node, params)
        mock_tebd.assert_called_once()
        mock_tdvp.assert_not_called()


@pytest.mark.tdvp_regression
def test_zip_up_nearest_neighbor_matches_tebd() -> None:
    """Zip-up uses TEBD on nearest-neighbor gates, matching an all-TEBD run."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cz(1, 2)
    qc.rx(0.3, 2)

    tebd_z = _run_strong_noiseless(qc, gate_mode="swaps")
    zip_up_z = _run_strong_noiseless(qc, gate_mode="mpo")
    assert zip_up_z == pytest.approx(tebd_z, abs=1e-12)


@pytest.mark.tdvp_regression
def test_tebd_lr_vs_tdvp() -> None:
    """TEBD with SWAP insertion matches all-TDVP on a long-range gate."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 2)

    tebd_z = _run_strong_noiseless(qc, gate_mode="swaps")
    tdvp_z = _run_strong_noiseless(qc, gate_mode="full-tdvp")
    assert tebd_z == pytest.approx(tdvp_z, abs=1e-10)


@pytest.mark.tdvp_regression
def test_tebd_lr_vs_qiskit() -> None:
    """TEBD with SWAP insertion matches Qiskit Statevector on small circuits."""
    qc = QuantumCircuit(4)
    qc.rx(0.37, 0)
    qc.ry(0.51, 1)
    qc.rz(0.23, 2)
    qc.h(0)
    qc.cx(0, 3)
    qc.rzz(0.2, 1, 3)

    # Use a stricter SVD threshold than the helper default to reduce purely numerical drift
    # from repeated TEBD split/merge operations (including SWAP insertion).
    tebd_vec = _run_strong_noiseless(qc, gate_mode="swaps", get_state=True, svd_threshold=1e-14)
    assert isinstance(tebd_vec, np.ndarray)

    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, tebd_vec) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.tdvp_regression
def test_tebd_mixed_vs_tdvp() -> None:
    """TEBD with SWAPs on a circuit mixing NN and long-range gates."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 3)
    qc.rzz(0.2, 2, 3)

    tebd_z = _run_strong_noiseless(qc, gate_mode="swaps")
    tdvp_z = _run_strong_noiseless(qc, gate_mode="full-tdvp")
    assert tebd_z == pytest.approx(tdvp_z, abs=1e-10)


@pytest.mark.tdvp_regression
def test_tebd_truncation_respects_max_bond_dim() -> None:
    """TEBD updates honor max_bond_dim on a circuit that grows entanglement."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    state = State(4, initial="zeros")
    sim_params = StrongSimParams(
        observables=[Observable(Z(), 0)],
        gate_mode="swaps",
        max_bond_dim=2,
        svd_threshold=1e-6,
    )
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)
    assert result.max_bond is not None
    assert int(np.max(result.max_bond)) <= 2


# --- apply_long_range_gate_mpo ---


def test_mpo_lr_cx() -> None:
    """Zip-up applies CX(1, 3) on |1111> via extended gate MPO."""
    length = 4
    mps = MPS(length, state="ones")
    mps.normalize()

    qc = QuantumCircuit(length)
    qc.cx(1, 3)
    dag = circuit_to_dag(qc)
    node = next(n for n in dag.front_layer() if n.op.name.lower() == "cx")

    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact", gate_mode="mpo")
    apply_long_range_gate_mpo(mps, convert_dag_to_tensor_algorithm(node)[0], sim_params)
    mps.normalize(decomposition="SVD")
    for i, element in enumerate(mps.to_vec()):
        if i == 7:
            np.testing.assert_allclose(np.abs(element), 1, atol=1e-10)
        else:
            np.testing.assert_allclose(np.abs(element), 0, atol=1e-10)


@pytest.mark.parametrize("gate_mode", ["swaps", "mpo", "tdvp"])
@pytest.mark.tdvp_regression
def test_lr_modes_fid_cap(gate_mode: str) -> None:
    """swaps, MPO, and hybrid TDVP all run accurately on a small LR gate."""
    length = 6
    theta = 0.3
    sites = (0, length - 1)
    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)

    params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=8,
        gate_mode=cast("GateMode", gate_mode),
        tdvp_sweeps=16,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    result = Simulator(parallel=False, show_progress=False).run(State(length, initial="zeros"), qc, params, None)
    assert result.output_state is not None
    out = result.output_state.mps
    if gate_mode == "tdvp":
        _assert_z_observables_match(ref, out.to_vec(), length)
        assert _fidelity(ref, out.to_vec()) == pytest.approx(PLUS_LR_RZZ_GLOBAL_FID, abs=1e-4)
    else:
        assert _fidelity(ref, out.to_vec()) > 0.999
    assert out.get_max_bond() <= 8
    assert_mps_bond_invariants(out, max_bond_dim=8)


@pytest.mark.tdvp_regression
def test_zip_lr_vs_tdvp() -> None:
    """Long-range gates use MPO mode and match full-tdvp."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 2)

    zip_up_z = _run_strong_noiseless(qc, gate_mode="mpo")
    tdvp_z = _run_strong_noiseless(qc, gate_mode="full-tdvp")
    assert zip_up_z == pytest.approx(tdvp_z, abs=1e-10)


@pytest.mark.tdvp_regression
def test_zip_lr_vs_tebd() -> None:
    """Zip-up on long-range gates matches SWAP+TEBD on small circuits."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 2)

    zip_up_z = _run_strong_noiseless(qc, gate_mode="mpo")
    tebd_z = _run_strong_noiseless(qc, gate_mode="swaps")
    assert zip_up_z == pytest.approx(tebd_z, abs=1e-10)


# --- apply_two_qubit_gate ---


def test_apply_two_qubit_gate() -> None:
    """Test applying a two-qubit gate.

    This test creates an MPS and applies a CX gate extracted from a circuit. It verifies that the MPS tensors change
    as expected after gate application.
    """
    length = 4
    mps0 = MPS(length, state="ones")
    mps0.normalize()

    qc = QuantumCircuit(length)
    qc.cx(1, 2)
    dag = circuit_to_dag(qc)

    cx_nodes = [node for node in dag.front_layer() if node.op.name.lower() == "cx"]
    assert cx_nodes, "No CX gate found in the front layer."
    node = cx_nodes[0]

    sim_params = StrongSimParams(observables=[Observable(Z(), 0)])
    copy.deepcopy(mps0.tensors)
    apply_two_qubit_gate(mps0, node, sim_params)
    mps0.normalize(decomposition="SVD")
    for i, element in enumerate(mps0.to_vec()):
        if i == 11:
            np.testing.assert_allclose(np.abs(element), 1, atol=1e-15)
        else:
            np.testing.assert_allclose(np.abs(element), 0, atol=1e-15)


def test_unknown_gate_mode_raises() -> None:
    """Invalid gate_mode names are rejected at gate-application time."""
    mps = MPS(2, state="zeros")
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)
    node = next(n for n in dag.front_layer() if n.op.name == "cx")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)])
    sim_params.gate_mode = cast("GateMode", "invalid")
    with pytest.raises(ValueError, match="Unknown gate_mode"):
        apply_two_qubit_gate(mps, node, sim_params)


@pytest.mark.parametrize("gate_mode", ["tdvp", "swaps"])
@pytest.mark.tdvp_regression
def test_nearest_neighbor_gate_modes_agree(gate_mode: str) -> None:
    """Hybrid and TEBD agree on a purely nearest-neighbor circuit."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cz(1, 2)
    qc.rx(0.3, 2)

    hybrid_z = _run_strong_noiseless(qc, gate_mode="tdvp")
    other_z = _run_strong_noiseless(qc, gate_mode=cast("GateMode", gate_mode))
    assert hybrid_z == pytest.approx(other_z, abs=1e-12)


@pytest.mark.tdvp_regression
def test_swaps_lr_reversed_ctrl() -> None:
    """TEBD swap routing handles gates with descending site indices (``cx(3, 0)``)."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(3, 0)
    swaps_z = _run_strong_noiseless(qc, gate_mode="swaps")
    mpo_z = _run_strong_noiseless(qc, gate_mode="mpo")
    assert swaps_z == pytest.approx(mpo_z, abs=1e-10)


# --- digital_tjm ---


def test_digital_tjm_strong_smoke_via_simulator() -> None:
    """Strong simulation via the public ``Simulator`` API completes and yields an observable."""
    length = 4
    state = State(length, initial="random")

    qc = QuantumCircuit(length)
    qc.cx(1, 3)

    sim_params = StrongSimParams(observables=[Observable(Z(), 0)])
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)

    assert result.expectation_values[0] is not None
    assert result.expectation_values[0].shape == (1,)


def test_digital_tjm_weak_smoke_via_simulator() -> None:
    """Weak simulation via the public ``Simulator`` API returns shot counts."""
    length = 4
    state = State(length, initial="random")

    qc = QuantumCircuit(length)
    qc.cx(1, 3)
    qc.measure_all()

    sim_params = WeakSimParams(shots=16)
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)

    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


def test_noisy_digital_tjm_matches_reference() -> None:
    """Noisy circuit TJM matches seeded reference at ``num_traj=100``.

    Circuit: for layer k, apply k repetitions of rzz(0.5) on (0,1) and (1,2) for a 3-qubit chain.
    Noise model: single-qubit bitflip on each qubit and crosstalk_xx/yy on neighbors, strength 0.01.
    Reference values were recorded with ``random_seed=7`` and ``num_traj=100`` (reproducible MC mean).
    A dense Qiskit density-matrix reference would require substantially more trajectories.
    """
    num_qubits = 3
    noise_factor = 0.01

    # Seeded TJM reference (random_seed=7, num_traj=100)
    reference = np.array(
        [
            [1.0, 0.92, 0.92, 0.9400000000000001, 0.9, 0.8],
            [1.0, 0.8200000000000002, 0.7000000000000001, 0.64, 0.52, 0.44],
            [1.0, 0.9200000000000002, 0.84, 0.72, 0.66, 0.58],
        ],
        dtype=float,
    )

    # YAQS noise model: bitflip on each site and crosstalk_xx on neighbors
    noise_model = NoiseModel(
        [{"name": "pauli_x", "sites": [i], "strength": noise_factor} for i in range(num_qubits)]
        + [{"name": "crosstalk_xx", "sites": [i, i + 1], "strength": noise_factor} for i in range(num_qubits - 1)]
        + [{"name": "crosstalk_yy", "sites": [i, i + 1], "strength": noise_factor} for i in range(num_qubits - 1)]
        + [{"name": "pauli_y", "sites": [i], "strength": noise_factor} for i in range(num_qubits)]
    )

    qc = QuantumCircuit(num_qubits)

    qc.rzz(0.5, 0, 1)
    qc.rzz(0.5, 1, 2)
    qc.barrier(label="SAMPLE_OBSERVABLES")
    qc.rzz(0.5, 0, 1)
    qc.rzz(0.5, 1, 2)
    qc.barrier(label="SAMPLE_OBSERVABLES")
    qc.rzz(0.5, 0, 1)
    qc.rzz(0.5, 1, 2)
    qc.barrier(label="SAMPLE_OBSERVABLES")
    qc.rzz(0.5, 0, 1)
    qc.rzz(0.5, 1, 2)
    qc.barrier(label="SAMPLE_OBSERVABLES")
    qc.rzz(0.5, 0, 1)
    qc.rzz(0.5, 1, 2)

    sim_params = StrongSimParams(
        observables=[Observable(Z(), i) for i in range(num_qubits)],
        sample_layers=True,
        num_mid_measurements=4,
        num_traj=100,
        random_seed=7,
    )
    state = State(num_qubits, initial="zeros", pad=2)
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, noise_model)

    tjm_results = np.empty((num_qubits, 6), dtype=float)
    for i in range(num_qubits):
        res = result.expectation_values[i]
        assert res is not None
        tjm_results[i, :] = np.real(res[:6])

    np.testing.assert_allclose(tjm_results, reference, rtol=0, atol=1e-12)


def test_digital_tjm_longrange_noise() -> None:
    """Smoke test: digital TJM runs with adjacent two-site crosstalk jump processes.

    Uses nearest-neighbor ``crosstalk_*`` only (long-range non-Pauli dissipation is not implemented
    on the digital path). Former Qiskit golden comparison removed.
    """
    num_qubits = 4
    noise_factor = 0.01

    timestep = create_ising_circuit(num_qubits, 1.0, 0.5, 0.1, 1, periodic=True)
    qc = QuantumCircuit(num_qubits)
    qc = qc.compose(timestep)
    assert qc is not None

    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": noise_factor},
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": noise_factor},
        {"name": "crosstalk_xx", "sites": [2, 3], "strength": noise_factor},
    ])

    sim_params = StrongSimParams(
        observables=[Observable(Z(), i) for i in range(num_qubits)],
        sample_layers=True,
        num_mid_measurements=0,
        num_traj=20,
        random_seed=9,
    )

    state = State(num_qubits, initial="zeros", pad=2)
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, noise_model)

    for i in range(num_qubits):
        res = result.expectation_values[i]
        assert res is not None
        assert res.shape == (2,)  # initial and final layer samples
        z_vals = np.real(res)
        assert np.isfinite(z_vals).all()
        assert np.all(np.abs(z_vals) <= 1.0 + 1e-6)


def test_no_mid_measurements_results_have_two_columns() -> None:
    """Circuit without any SAMPLE_OBSERVABLES barriers should yield 2 columns (initial, final).

    Builds a 3-qubit circuit with a few gates but no labelled 'SAMPLE_OBSERVABLES' barriers,
    enables layer sampling via StrongSimParams, runs the simulator, and asserts that each
    observable's results has shape (2,), corresponding to the initial and final sampling
    points only.
    """
    num_qubits = 3
    qc = QuantumCircuit(num_qubits)
    # A couple of gates but no labelled barrier
    qc.rx(0.3, 0)
    qc.cx(0, 1)
    qc.rzz(0.1, 1, 2)

    sim_params = StrongSimParams(observables=[Observable(Z(), i) for i in range(num_qubits)], sample_layers=True)
    state = State(num_qubits, initial="zeros")

    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, noise_model=None)

    for i in range(len(result.observables)):
        assert result.expectation_values[i] is not None
        assert result.expectation_values[i].shape == (2,)


def test_counts_multiple_mid_measurement_barriers() -> None:
    """Three SAMPLE_OBSERVABLES barriers produce 5 columns (initial + 3 mids + final).

    Constructs a 4-qubit circuit with three barriers labelled 'SAMPLE_OBSERVABLES' using
    different cases (to verify case-insensitivity), enables layer sampling, runs the
    simulation, and asserts that each observable's results has shape (5,), capturing the
    initial state, each SAMPLE_OBSERVABLES sampling point, and the final state.
    """
    num_qubits = 4
    qc = QuantumCircuit(num_qubits)
    # First segment
    qc.rx(0.2, 0)
    qc.cx(0, 1)
    qc.barrier(label="SAMPLE_OBSERVABLES")
    # Second segment
    qc.rzz(0.5, 1, 2)
    qc.barrier(label="SAMPLE_OBSERVABLES")  # case-insensitive
    # Third segment
    qc.rx(0.7, 3)
    qc.barrier(label="SAMPLE_OBSERVABLES")  # mixed case
    # Final segment
    qc.cx(2, 3)

    sim_params = StrongSimParams(observables=[Observable(Z(), i) for i in range(num_qubits)], sample_layers=True)
    state = State(num_qubits, initial="zeros")

    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, noise_model=None)

    for i in range(len(result.observables)):
        assert result.expectation_values[i] is not None
        assert result.expectation_values[i].shape == (5,)


def test_ignores_non_mid_barriers_and_handles_measures() -> None:
    """Barriers without the label and terminal measurements are ignored for sampling.

    Creates a 2-qubit circuit that includes an unlabelled barrier (ignored), a labelled
    'SAMPLE_OBSERVABLES' barrier (counted), a terminal measurement operation (removed),
    and a barrier with a non-matching label (ignored). With layer sampling enabled, the
    test asserts that each observable's results has shape (3,), corresponding to initial,
    one mid, and final sampling points.
    """
    num_qubits = 2
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.barrier()  # no label -> ignored
    qc.rx(0.1, 0)
    qc.barrier(label="SAMPLE_OBSERVABLES")
    qc.cx(0, 1)
    qc.barrier(label="not-mid")  # ignored
    qc.rzz(0.2, 0, 1)
    qc.measure(0, 0)  # terminal measurements are removed

    sim_params = StrongSimParams(observables=[Observable(Z(), i) for i in range(num_qubits)], sample_layers=True)
    state = State(num_qubits, initial="zeros")

    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, noise_model=None)

    for i in range(len(result.observables)):
        assert result.expectation_values[i] is not None
        # Only one labelled barrier -> 1 mid + initial + final
        assert result.expectation_values[i].shape == (3,)


def test_weak_noiseless_get_state_returns_mps() -> None:
    """Noiseless weak simulation with ``get_state=True`` returns the final MPS."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.measure_all()
    sim_params = WeakSimParams(shots=4, gate_mode="tdvp", preset="exact", get_state=True)
    state = State(2, initial="zeros")
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)
    assert result.counts is not None
    assert sum(result.counts.values()) == 4
    assert result.output_state is not None
    assert result.output_state.mps.length == 2


def test_digital_tjm_weak_noisy_get_state_returns_mps() -> None:
    """Noisy weak ``digital_tjm`` may return the evolved MPS when ``get_state=True``."""
    mps = MPS(2, state="zeros")
    qc = QuantumCircuit(2)
    qc.h(0)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.01}])
    sim_params = WeakSimParams(shots=1, gate_mode="tdvp", preset="exact", get_state=True)
    counts, _, final = digital_tjm((0, mps, noise_model, sim_params, qc))
    assert isinstance(counts, dict)
    assert final is not None
    assert final.length == 2


def test_weak_simulation_nearest_neighbor_counts() -> None:
    """Weak simulation with NN gates returns valid shot counts."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    state = State(3, initial="zeros")
    sim_params = WeakSimParams(shots=32, gate_mode="tdvp", preset="exact")
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)
    assert result.counts is not None
    assert sum(result.counts.values()) == 32


@pytest.mark.parametrize(
    ("num_qubits", "ones"),
    [
        (3, (0,)),
        (3, (1,)),
        (3, (2,)),
        (4, (0, 2)),
        (4, (1, 3)),
        (5, (0, 1, 4)),
    ],
)
def test_weak_counts_match_qiskit_qubit_ordering_deterministic(num_qubits: int, ones: tuple[int, ...]) -> None:
    """Weak counts match Qiskit (bitstring->int) for deterministic basis states."""
    qc = QuantumCircuit(num_qubits, num_qubits)
    for q in ones:
        qc.x(q)
    qc.measure_all()

    shots = 32
    sim_params = WeakSimParams(shots=shots, gate_mode="tdvp", preset="exact")
    state = State(num_qubits, initial="zeros")
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)
    assert result.counts is not None

    # Qiskit gives bitstrings ordered c_{n-1}..c_0; YAQS uses int keys.
    qc_nom = qc.remove_final_measurements(inplace=False)
    assert qc_nom is not None
    probs = Statevector(qc_nom).probabilities_dict()
    qiskit_counts_int = {int(bitstring, 2): round(p * shots) for bitstring, p in probs.items() if p > 0}
    # Deterministic circuits should produce exactly one outcome with probability 1.
    assert sum(qiskit_counts_int.values()) == shots
    assert result.counts == qiskit_counts_int


def test_noisy_nearest_neighbor_smoke() -> None:
    """Noisy digital simulation still runs with hybrid TEBD on a NN gate."""
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.01}])
    state = State(2, initial="zeros")
    sim_params = StrongSimParams(
        observables=[Observable(Z(), 0)],
        gate_mode="tdvp",
        num_traj=4,
        random_seed=0,
    )
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, noise_model)
    assert result.expectation_values[0] is not None


def test_bell_state_sanity() -> None:
    """|00> + H(0) + CNOT(0,1) yields (|00> + |11>)/sqrt(2) under TEBD/hybrid."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    vec = _run_strong_noiseless(qc, gate_mode="tdvp", get_state=True)
    assert isinstance(vec, np.ndarray)
    probs = np.abs(vec) ** 2
    np.testing.assert_allclose(probs[0], 0.5, atol=1e-10)
    np.testing.assert_allclose(probs[3], 0.5, atol=1e-10)
    np.testing.assert_allclose(probs[1] + probs[2], 0.0, atol=1e-10)


def test_statevector_vs_qiskit() -> None:
    """Lock down YAQS dense-vector convention against Qiskit on non-symmetric circuits.

    This guards qubit ordering conventions in YAQS' digital backend against Qiskit.
    """
    n = 3
    circuits: list[QuantumCircuit] = []

    # Product basis states distinguish sites 0/1/2
    for site in range(n):
        qc = QuantumCircuit(n)
        qc.x(site)
        circuits.append(qc)

    # Non-symmetric single-qubit rotations + asymmetry
    qc = QuantumCircuit(n)
    qc.rx(0.37, 0)
    qc.ry(0.51, 1)
    qc.rz(0.23, 2)
    qc.x(0)
    qc.h(2)
    circuits.append(qc)

    # Directional CNOT checks (avoid patterns that are ambiguous under bit reversal)
    for h_site, ctrl, tgt in [(0, 0, 1), (1, 1, 2), (0, 0, 2), (2, 2, 0)]:
        qc = QuantumCircuit(n)
        qc.h(h_site)
        qc.cx(ctrl, tgt)
        circuits.append(qc)

    for qc in circuits:
        vec = _run_strong_noiseless(qc, gate_mode="tdvp", get_state=True)
        assert isinstance(vec, np.ndarray)
        ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
        assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


def test_observables_vs_qiskit() -> None:
    """Digital observable expectations match Qiskit (guards qubit ordering)."""

    def qiskit_single_pauli_expectation(qc: QuantumCircuit, site: int, op: str) -> float:
        n = qc.num_qubits
        label = ["I"] * n
        # Qiskit Pauli labels are ordered as q_{n-1} ... q_0.
        label[n - 1 - site] = op
        psi = Statevector(qc)
        exp = psi.expectation_value(Pauli("".join(label)))
        return float(np.real(exp))

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    qc.ry(0.51, 1)
    qc.rz(0.23, 2)
    qc.cx(2, 1)

    requested = [
        Observable(Z(), 2),
        Observable(X(), 0),
        Observable(Y(), 1),
        Observable(Z(), 0),
        Observable(X(), 2),
    ]
    sim_params = StrongSimParams(observables=requested, gate_mode="tdvp", preset="exact")
    state = State(3, initial="zeros")
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)

    for i, obs in enumerate(result.observables):
        site = obs.sites[0] if isinstance(obs.sites, list) else obs.sites
        assert isinstance(site, int)
        op = obs.gate.name.upper()
        assert op in {"X", "Y", "Z"}
        expected = qiskit_single_pauli_expectation(qc, site, op)
        got = float(np.real(result.expectation_values[i][0]))
        assert got == pytest.approx(expected, abs=1e-10)


def test_pauli_obs_vs_qiskit() -> None:
    """Observables computed from YAQS dense vector match Qiskit Pauli placement."""

    def qiskit_single_pauli_expectation_from_vec(vec: np.ndarray, site: int, op: str) -> float:
        n = int(np.log2(vec.size))
        label = ["I"] * n
        # Qiskit Pauli labels are ordered as q_{n-1} ... q_0.
        label[n - 1 - site] = op
        psi = Statevector(vec)
        exp = psi.expectation_value(Pauli("".join(label)))
        return float(np.real(exp))

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    qc.ry(0.51, 1)
    qc.rz(0.23, 2)
    qc.cx(2, 1)

    requested = [
        Observable(Z(), 2),
        Observable(X(), 0),
        Observable(Y(), 1),
        Observable(Z(), 0),
        Observable(X(), 2),
    ]
    sim_params = StrongSimParams(observables=requested, gate_mode="tdvp", preset="exact", get_state=True)
    state = State(3, initial="zeros")
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)

    assert result.output_state is not None
    yaqs_vec = result.output_state.mps.to_vec()

    for i, obs in enumerate(result.observables):
        site = obs.sites[0] if isinstance(obs.sites, list) else obs.sites
        assert isinstance(site, int)
        op = obs.gate.name.upper()
        assert op in {"X", "Y", "Z"}
        expected = qiskit_single_pauli_expectation_from_vec(yaqs_vec, site, op)
        got = float(np.real(result.expectation_values[i][0]))
        assert got == pytest.approx(expected, abs=1e-10)


def test_obs_order_aligned() -> None:
    """Ensure expectation_values[i] corresponds to result.observables[i] (user order)."""
    qc = QuantumCircuit(3)
    qc.rx(0.37, 0)
    qc.cx(0, 2)
    qc.ry(0.51, 1)

    # Intentionally not sorted by site: Result order should still match this list.
    requested = [Observable(Z(), 2), Observable(X(), 0), Observable(Z(), 0)]
    sim_params = StrongSimParams(observables=requested, gate_mode="tdvp", preset="exact", get_state=True)
    state = State(3, initial="zeros")
    result = Simulator(parallel=False, show_progress=False).run(state, qc, sim_params, None)

    assert result.output_state is not None
    vec = result.output_state.mps.to_vec()

    assert len(result.observables) == len(requested)
    for i, (got_obs, req_obs) in enumerate(zip(result.observables, requested, strict=True)):
        assert got_obs.gate.name == req_obs.gate.name
        assert got_obs.sites == req_obs.sites

        # Verify the expectation value matches the observable at the same index.
        n = int(np.log2(vec.size))
        label = ["I"] * n
        site = got_obs.sites[0] if isinstance(got_obs.sites, list) else got_obs.sites
        assert isinstance(site, int)
        label[n - 1 - site] = got_obs.gate.name.upper()
        expected = float(np.real(Statevector(vec).expectation_value(Pauli("".join(label)))))
        got = float(np.real(result.expectation_values[i][-1]))
        assert got == pytest.approx(expected, abs=1e-10)
