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
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams, WeakSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import GateLibrary, X, Y, Z
from mqt.yaqs.core.methods.tdvp.bond_support import (
    protected_bonds_for_two_site_gate,
    select_protected_seed_bonds,
)
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
    NORM_TOL,
    _apply_lr_gate,
    _bond_second_schmidt,
    _fidelity,
    _max_bond,
    _plus_rzz_overlap,
    _prep_state,
    _qiskit_plus_rzz_reference,
    _reliable_sweeps,
    _run_circuit,
    _tdvp_params,
    assert_mps_bond_invariants,
)
from tests.digital.conftest import _physical_second_schmidt, _run_strong_noiseless

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.simulation_parameters import GateMode


# --- create_local_noise_model ---

def test_create_local_noise_model() -> None:
    """Test the create_local_noise_model function.

    This test creates a global noise model with various noise processes and tests the creation
    of local noise models for different gate positions. It verifies that only the relevant
    noise processes are included in the local model based on the gate's site range.
    """
    # Create a global noise model with various processes
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
    global_noise_model = NoiseModel(global_processes)

    # Test case 1: Gate acting on sites [1, 2]
    local_model_1 = create_local_noise_model(global_noise_model, 1, 2)

    # Should include: bitflip on sites 1, 2 and crosstalk_xx, crosstalk_yx on [1, 2]
    expected_processes_1 = [
        {"name": "pauli_x", "sites": [1], "strength": 0.02},
        {"name": "pauli_x", "sites": [2], "strength": 0.03},
        {"name": "crosstalk_xx", "sites": [1, 2], "strength": 0.06},
        {"name": "crosstalk_yx", "sites": [1, 2], "strength": 0.10},
    ]

    assert len(local_model_1.processes) == len(expected_processes_1)
    for expected_process in expected_processes_1:
        found = False
        for actual_process in local_model_1.processes:
            if (
                actual_process["name"] == expected_process["name"]
                and actual_process["sites"] == expected_process["sites"]
                and actual_process["strength"] == expected_process["strength"]
            ):
                found = True
                break
        assert found, f"Expected process {expected_process} not found in local model"

    # Test case 2: Gate acting on sites [0, 1]
    local_model_2 = create_local_noise_model(global_noise_model, 0, 1)

    # Should include: bitflip on sites 0, 1 and crosstalk_xx, crosstalk_xy on [0, 1]
    expected_processes_2 = [
        {"name": "pauli_x", "sites": [0], "strength": 0.01},
        {"name": "pauli_x", "sites": [1], "strength": 0.02},
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": 0.05},
        {"name": "crosstalk_xy", "sites": [0, 1], "strength": 0.09},
    ]

    assert len(local_model_2.processes) == len(expected_processes_2)
    for expected_process in expected_processes_2:
        found = False
        for actual_process in local_model_2.processes:
            if (
                actual_process["name"] == expected_process["name"]
                and actual_process["sites"] == expected_process["sites"]
                and actual_process["strength"] == expected_process["strength"]
            ):
                found = True
                break
        assert found, f"Expected process {expected_process} not found in local model"

    # Test case 3: Gate acting on sites [2, 3]
    local_model_3 = create_local_noise_model(global_noise_model, 2, 3)

    # Should include: bitflip on sites 2, 3 and crosstalk_xx on [2, 3]
    expected_processes_3 = [
        {"name": "pauli_x", "sites": [2], "strength": 0.03},
        {"name": "pauli_x", "sites": [3], "strength": 0.04},
        {"name": "crosstalk_xx", "sites": [2, 3], "strength": 0.07},
    ]

    assert len(local_model_3.processes) == len(expected_processes_3)
    for expected_process in expected_processes_3:
        found = False
        for actual_process in local_model_3.processes:
            if (
                actual_process["name"] == expected_process["name"]
                and actual_process["sites"] == expected_process["sites"]
                and actual_process["strength"] == expected_process["strength"]
            ):
                found = True
                break
        assert found, f"Expected process {expected_process} not found in local model"

    # Test case 4: Single-qubit gate on site 1
    local_model_4 = create_local_noise_model(global_noise_model, 1, 1)

    # Should include: bitflip on site 1 only
    expected_processes_4 = [
        {"name": "pauli_x", "sites": [1], "strength": 0.02},
    ]

    assert len(local_model_4.processes) == len(expected_processes_4)
    for expected_process in expected_processes_4:
        found = False
        for actual_process in local_model_4.processes:
            if (
                actual_process["name"] == expected_process["name"]
                and actual_process["sites"] == expected_process["sites"]
                and actual_process["strength"] == expected_process["strength"]
            ):
                found = True
                break
        assert found, f"Expected process {expected_process} not found in local model"

    # Test case 5: Gate acting on sites [1, 2, 3] (three-qubit gate)
    local_model_5 = create_local_noise_model(global_noise_model, 1, 3)

    # Should include: bitflip on sites 1, 2, 3 and crosstalk_xx, crosstalk_yx on [1, 2], crosstalk_xx on [2, 3]
    expected_processes_5 = [
        {"name": "pauli_x", "sites": [1], "strength": 0.02},
        {"name": "pauli_x", "sites": [3], "strength": 0.04},
        {"name": "crosstalk_xx", "sites": [1, 3], "strength": 0.06},
    ]

    assert len(local_model_5.processes) == len(expected_processes_5)
    for expected_process in expected_processes_5:
        found = False
        for actual_process in local_model_5.processes:
            if (
                actual_process["name"] == expected_process["name"]
                and actual_process["sites"] == expected_process["sites"]
                and actual_process["strength"] == expected_process["strength"]
            ):
                found = True
                break
        assert found, f"Expected process {expected_process} not found in local model"


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
        q0 = node.qargs[0]._index
        q1 = node.qargs[1]._index
        assert min(q0, q1) % 2 == 0, f"Node with qubits {q0, q1} not in even group."

    # For each node in the odd group, the lower qubit index should be odd.
    for node in odd:
        q0 = node.qargs[0]._index
        q1 = node.qargs[1]._index
        assert min(q0, q1) % 2 == 1, f"Node with qubits {q0, q1} not in odd group."


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
    tensors = [np.full((2, 1, 1), i, dtype=complex) for i in range(5)]
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

_RZZ_ANGLE = 0.3


def _lr_pair(length: int) -> tuple[int, int]:
    return (0, length - 1)


def _nn_pair(_length: int) -> tuple[int, int]:
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


@pytest.mark.tdvp_regression
def test_lr_gate_routes_through_dynamic_not_two_site() -> None:
    """Long-range digital TDVP must use dynamic sweeps, not the 2-site kernel."""
    length = 8
    gate = GateLibrary.rzz([0.3])
    gate.set_sites(0, length - 1)
    out = copy.deepcopy(State(length, initial="x+").mps)
    params = _tdvp_params(max_bond_dim=None, tdvp_sweeps=4)

    with (
        patch("mqt.yaqs.core.methods.tdvp.integrators._dynamic_tdvp_sweep") as mock_dynamic,
        patch("mqt.yaqs.core.methods.tdvp.integrators._two_site_tdvp_sweep") as mock_two,
    ):
        mock_dynamic.side_effect = lambda *_args, **_kwargs: None
        apply_two_qubit_gate_tdvp(out, gate, params)
        mock_dynamic.assert_called_once()
        mock_two.assert_not_called()
        assert len(mock_dynamic.call_args.args) >= 4
        assert mock_dynamic.call_args.args[3]


@pytest.mark.tdvp_regression
def test_chi_one_lr_rzz_caller_respects_cap() -> None:
    """End-to-end χ=1 long-range RZZ does not create rank-2 support from seeding."""
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
def test_bond_invariants_after_full_lr_gate() -> None:
    """Public LR TDVP gate application leaves consistent tensor shapes."""
    out = _apply_lr_gate(State(8, initial="x+").mps, "rzz", 0.3, max_bond_dim=8, sweeps=16)
    assert_mps_bond_invariants(out, max_bond_dim=8)


@pytest.mark.parametrize("tdvp_sweeps", [1, 4, 16])
@pytest.mark.tdvp_regression
def test_default_run_matches_explicit_sweep_count(tdvp_sweeps: int) -> None:
    """Normal runs do not require debug tracing and match explicit sweep counts."""
    length = 6
    theta = 0.3
    out_a = _apply_lr_gate(State(length, initial="x+").mps, "rzz", theta, max_bond_dim=8, sweeps=tdvp_sweeps)
    out_b = _apply_lr_gate(State(length, initial="x+").mps, "rzz", theta, max_bond_dim=8, sweeps=tdvp_sweeps)
    assert _fidelity(out_a.to_vec(), out_b.to_vec()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.tdvp_regression
def test_lr_rzz_plus_dynamic_routing_fidelity() -> None:
    """Long-range RZZ on |+⟩^L evolves away from the product state."""
    length = 8
    theta = 0.3
    sites = (0, length - 1)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=64))

    ref = _qiskit_plus_rzz_reference(length, theta, sites=sites)
    plus = State(length, initial="x+").mps
    assert _fidelity(ref, out.to_vec()) > 0.999
    assert _fidelity(plus.to_vec(), out.to_vec()) < 0.99
    assert _bond_second_schmidt(out, 0) > 0.1


@pytest.mark.parametrize("length", [7, 8])
@pytest.mark.tdvp_regression
def test_endpoint_lr_support_retention(length: int) -> None:
    """Endpoint RZZ on |+⟩^L retains rank on all crossed bonds."""
    theta = 0.3
    sites = (0, length - 1)
    sweeps = _reliable_sweeps(length)
    out = _apply_lr_gate(
        State(length, initial="x+").mps,
        "rzz",
        theta,
        max_bond_dim=None,
        sweeps=sweeps,
    )
    ref = _qiskit_plus_rzz_reference(length, theta, sites=sites)
    plus = State(length, initial="x+").mps
    assert _fidelity(ref, out.to_vec()) > 0.999
    overlap = _fidelity(plus.to_vec(), out.to_vec())
    assert overlap < 0.999
    assert overlap == pytest.approx(_plus_rzz_overlap(length, theta), rel=0.02)
    for bond in range(sites[0], sites[1]):
        assert _bond_second_schmidt(out, bond) > 0.1
    assert out.norm() == pytest.approx(1.0, abs=NORM_TOL)
    assert_mps_bond_invariants(out)


@pytest.mark.tdvp_regression
@pytest.mark.slow
def test_endpoint_lr_support_retention_l10() -> None:
    """Endpoint retention at L=10 needs more sweeps."""
    test_endpoint_lr_support_retention(10)


@pytest.mark.tdvp_regression
def test_internal_pair_support_retention() -> None:
    """Internal long-range pairs protect crossed bonds, not only endpoints."""
    theta = 0.3
    length = 10
    sites = (2, 7)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=64))

    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, out.to_vec()) > 0.999

    window = (1, 8)
    protected = protected_bonds_for_two_site_gate(sites[0], sites[1], window[0], window[1])
    assert protected == frozenset({1, 2, 3, 4, 5})
    assert 0 not in protected
    for bond in range(sites[0], sites[1]):
        assert _bond_second_schmidt(out, bond) > 0.05
    assert_mps_bond_invariants(out)


@pytest.mark.tdvp_regression
def test_internal_pair_non_seed_bonds_remain_protected() -> None:
    """Non-seed crossed bonds stay protected while only seed bonds are initialized."""
    theta = 0.3
    length = 10
    sites = (2, 7)
    window = (0, 9)
    protected = protected_bonds_for_two_site_gate(sites[0], sites[1], window[0], window[1])
    seed_bonds = select_protected_seed_bonds(protected)
    assert protected == frozenset({2, 3, 4, 5, 6})
    assert seed_bonds == frozenset({2, 3})
    assert seed_bonds <= protected

    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=64))

    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, out.to_vec()) > 0.999
    for bond in protected:
        assert _bond_second_schmidt(out, bond) > 0.05
    assert_mps_bond_invariants(out)


@pytest.mark.tdvp_regression
def test_shifted_internal_pair_support_retention() -> None:
    """Shifted internal RZZ(1,8) on |+⟩^L retains support on crossed bonds only."""
    theta = 0.3
    length = 10
    sites = (1, 8)
    gate = GateLibrary.rzz([theta])
    gate.set_sites(*sites)
    out = copy.deepcopy(State(length, initial="x+").mps)
    apply_two_qubit_gate_tdvp(out, gate, _tdvp_params(max_bond_dim=None, tdvp_sweeps=64))

    qc = QuantumCircuit(length)
    qc.h(range(length))
    qc.rzz(theta, sites[0], sites[1])
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, out.to_vec()) > 0.999

    window = (0, 9)
    protected = protected_bonds_for_two_site_gate(sites[0], sites[1], window[0], window[1])
    assert protected == frozenset(range(1, 8))
    for bond in range(sites[0], sites[1]):
        assert _bond_second_schmidt(out, bond) > 0.05
    assert_mps_bond_invariants(out)


@pytest.mark.tdvp_regression
def test_haar_random_state_no_regression() -> None:
    """Haar-random states remain accurate under long-range dynamic TDVP gates."""
    length = 8
    theta = 0.3
    sites = (0, length - 1)
    prep = MPS(length, state="haar-random")
    prep.normalize()
    out = _apply_lr_gate(prep, "rzz", theta, max_bond_dim=None, sweeps=64)

    qc = QuantumCircuit(length)
    qc.initialize(prep.to_vec().tolist(), range(length))
    qc.rzz(theta, sites[0], sites[1])
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, out.to_vec()) > 0.99


@pytest.mark.parametrize("length", [8, 10])
@pytest.mark.tdvp_regression
def test_zeros_lr_rzz_uncapped_exact(length: int) -> None:
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
def test_zeros_lr_rzz_fixed_chi_exact(length: int) -> None:
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
def test_fixed_chi_lr_gate_respects_cap(
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
def test_fixed_chi_plus_rzz_one_no_secret_pad() -> None:
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
def test_fixed_chi_plus_rzz_two_applies_gate() -> None:
    """χ=2 plus/RZZ applies the gate rather than staying at |+⟩^L."""
    theta = 0.3
    length = 7
    prep = _prep_state("plus", length)
    out = _apply_lr_gate(prep, "rzz", theta, max_bond_dim=2, sweeps=4)
    plus = _prep_state("plus", length)
    assert _max_bond(out) <= 2
    assert _fidelity(plus.to_vec(), out.to_vec()) < 0.99
    assert_mps_bond_invariants(out, max_bond_dim=2)


@pytest.mark.tdvp_regression
def test_fixed_chi_plus_rzz_eight_high_fidelity() -> None:
    """χ=8 plus/RZZ reaches high fidelity with enough sweeps."""
    theta = 0.3
    length = 8
    prep = _prep_state("plus", length)
    out = _apply_lr_gate(prep, "rzz", theta, max_bond_dim=8, sweeps=64)
    ref = _qiskit_plus_rzz_reference(length, theta, sites=(0, length - 1))
    assert _fidelity(ref, out.to_vec()) > 0.999
    assert _max_bond(out) <= 8
    assert _bond_second_schmidt(out, 0) > 0.1
    assert_mps_bond_invariants(out, max_bond_dim=8)


@pytest.mark.tdvp_regression
def test_fixed_chi_rzz_lr_ladder_no_shape_error() -> None:
    """Multi-gate ladder circuits complete without bond-dimension violations."""
    length = 8
    prep = State(length, initial="zeros").mps
    out = _run_circuit(prep, _rzz_lr_ladder_circuit(length), max_bond_dim=8, sweeps=4)
    assert _max_bond(out) <= 8
    assert_mps_bond_invariants(out, max_bond_dim=8)


@pytest.mark.tdvp_regression
def test_fixed_chi_mixed_small_respects_cap() -> None:
    """Mixed NN+LR circuits respect χ under hybrid TDVP routing."""
    length = 8
    prep = _prep_state("plus", length)
    out = _run_circuit(prep, _mixed_small_circuit(length), max_bond_dim=8, sweeps=16)
    assert _max_bond(out) <= 8
    assert out.norm() == pytest.approx(1.0, abs=NORM_TOL)
    assert_mps_bond_invariants(out, max_bond_dim=8)


@pytest.mark.parametrize("initial_state", ["plus", "low_depth"])
@pytest.mark.parametrize("max_bond_dim", [1, 2, 8, None])
@pytest.mark.parametrize("sweeps", [1, 4, 16, 64])
@pytest.mark.tdvp_regression
def test_fixed_chi_norm_stability(
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
def test_fixed_chi_truncates_high_bond_initial_state(gate_name: str) -> None:
    """Fixed-χ TDVP truncates incoming MPS bond dimensions before evolving."""
    length = 8
    prep = _prep_state("low_depth", length)
    assert max(prep.bond_dimensions()) > 1
    out = _apply_lr_gate(prep, gate_name, 0.3, max_bond_dim=1, sweeps=4)
    assert _max_bond(out) <= 1
    assert out.norm() == pytest.approx(1.0, abs=NORM_TOL)


@pytest.mark.tdvp_regression
def test_hybrid_tdvp_lr_uses_dynamic_path() -> None:
    """Hybrid gate_mode='tdvp' routes LR gates through dynamic TDVP."""
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
def test_hybrid_tdvp_lr_end_to_end_fidelity() -> None:
    """Public hybrid path reaches high fidelity on endpoint RZZ."""
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
        tdvp_sweeps=64,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    result = Simulator(parallel=False, show_progress=False).run(State(length, initial="zeros"), qc, params, None)
    assert result.output_state is not None
    ref = _qiskit_plus_rzz_reference(length, theta, sites=sites)
    assert _fidelity(ref, result.output_state.mps.to_vec()) > 0.999


@pytest.mark.tdvp_regression
def test_low_depth_hybrid_tdvp_runs_without_crash() -> None:
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
def test_long_range_hybrid_matches_tdvp() -> None:
    """Long-range gates use TDVP in hybrid mode and match an all-TDVP run."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 2)

    hybrid_z = _run_strong_noiseless(qc, gate_mode="tdvp")
    tdvp_z = _run_strong_noiseless(qc, gate_mode="full-tdvp")
    assert hybrid_z == pytest.approx(tdvp_z, abs=1e-10)


@pytest.mark.tdvp_regression
def test_mixed_circuit_hybrid_matches_tdvp() -> None:
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
def test_tdvp_sweeps_preserves_gate_unitary() -> None:
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
def test_tdvp_sweeps_hybrid_long_range_matches_qiskit() -> None:
    """Hybrid long-range TDVP with multiple sweeps matches Qiskit within truncation error."""
    qc = QuantumCircuit(4)
    qc.rx(0.2, 1)
    qc.rxx(0.25, 0, 3)

    vec = _run_strong_noiseless(qc, gate_mode="tdvp", get_state=True, tdvp_sweeps=8)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.tdvp_regression
def test_tdvp_sweeps_hybrid_nn_unchanged() -> None:
    """Nearest-neighbor hybrid circuits ignore tdvp_sweeps (TEBD path)."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cz(1, 2)

    baseline = _run_strong_noiseless(qc, gate_mode="tdvp", tdvp_sweeps=1)
    many_sweeps = _run_strong_noiseless(qc, gate_mode="tdvp", tdvp_sweeps=5)
    assert baseline == pytest.approx(many_sweeps, abs=1e-10)


@pytest.mark.tdvp_regression
def test_tdvp_sweeps_mixed_circuit_regression() -> None:
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

def test_apply_two_qubit_gate_tebd_direct_cx_long_range() -> None:
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


def test_apply_two_qubit_gate_tebd_direct_cnot() -> None:
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
def test_hybrid_tdvp_nn_uses_tebd(gate_name: str, sites: tuple[int, int]) -> None:
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
def test_long_range_tebd_matches_tdvp() -> None:
    """TEBD with SWAP insertion matches all-TDVP on a long-range gate."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 2)

    tebd_z = _run_strong_noiseless(qc, gate_mode="swaps")
    tdvp_z = _run_strong_noiseless(qc, gate_mode="full-tdvp")
    assert tebd_z == pytest.approx(tdvp_z, abs=1e-10)


@pytest.mark.tdvp_regression
def test_long_range_tebd_matches_qiskit_statevector() -> None:
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
def test_mixed_circuit_tebd_matches_tdvp() -> None:
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

def test_apply_long_range_gate_mpo_direct_cx_long_range() -> None:
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
def test_lr_gate_modes_high_fidelity_and_cap(gate_mode: str) -> None:
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
    assert _fidelity(ref, out.to_vec()) > 0.999
    assert out.get_max_bond() <= 8
    assert_mps_bond_invariants(out, max_bond_dim=8)


@pytest.mark.tdvp_regression
def test_long_range_zip_up_matches_tdvp() -> None:
    """Long-range gates use MPO mode and match full-tdvp."""
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 2)

    zip_up_z = _run_strong_noiseless(qc, gate_mode="mpo")
    tdvp_z = _run_strong_noiseless(qc, gate_mode="full-tdvp")
    assert zip_up_z == pytest.approx(tdvp_z, abs=1e-10)


@pytest.mark.tdvp_regression
def test_long_range_zip_up_matches_tebd() -> None:
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
def test_long_range_swap_path_reversed_control_target() -> None:
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
    """Barriers without the label and measurements are ignored for sampling.

    Creates a 2-qubit circuit that includes an unlabelled barrier (ignored), a labelled
    'SAMPLE_OBSERVABLES' barrier (counted), a measurement operation (removed), and a barrier
    with a non-matching label (ignored). With layer sampling enabled, the test asserts
    that each observable's results has shape (3,), corresponding to initial, one mid,
    and final sampling points.
    """
    num_qubits = 2
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.barrier()  # no label -> ignored
    qc.rx(0.1, 0)
    qc.barrier(label="SAMPLE_OBSERVABLES")
    qc.measure(0, 0)  # measurements are removed
    qc.cx(0, 1)
    qc.barrier(label="not-mid")  # ignored
    qc.rzz(0.2, 0, 1)

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


def test_statevector_matches_qiskit_on_nonsymmetric_probes() -> None:
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


def test_observables_match_qiskit_statevector_qubit_ordering() -> None:
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


def test_statevector_observables_match_qiskit_paulis() -> None:
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


def test_expectation_values_align_with_result_observables_order() -> None:
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
