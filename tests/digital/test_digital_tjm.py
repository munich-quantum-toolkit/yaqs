# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the digital tensor jump method implementation.

This module provides unit tests for the CircuitTJM functionality.
The tests verify that various components of the CircuitTJM implementation work correctly,
including:
  - Grouping and processing of DAG layers for single-qubit and two-qubit gates.
  - Application of single-qubit and two-qubit gates to a Matrix Product State (MPS).
  - Construction of generator MPOs from gate operations.
  - Extraction of local windows from MPS and MPO objects.
  - Execution of circuit-based simulations in both strong and weak simulation regimes.

These tests ensure that the implemented routines correctly simulate quantum circuits using
the Tensor Jump Method.
"""

# ignore non-lowercase variable names for physics notation

from __future__ import annotations

import copy

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams, WeakSimParams
from mqt.yaqs.core.libraries.gate_library import GateLibrary, X, Z
from mqt.yaqs.digital.digital_tjm import (
    apply_single_qubit_gate,
    apply_two_qubit_gate,
    apply_window,
    construct_generator_mpo,
    create_local_noise_model,
    digital_tjm,
    process_layer,
)


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
    qc.barrier(1)
    qc.x(qc.qubits[2])
    qc.cx(5, 4)
    qc.cx(7, 8)

    # Convert the circuit to a DAG.
    dag = circuit_to_dag(qc)

    # Call process_layer on the DAG.
    single, even, odd = process_layer(dag)

    # After processing, the measurement and barrier nodes should have been removed.
    for node in dag.op_nodes():
        assert node.op.name not in {"measure", "barrier"}, f"Unexpected node {node.op.name} in the DAG op nodes."

    # Verify that the single-qubit gate is in the single-qubit group.
    single_names = [node.op.name.lower() for node in single]
    assert any("x" in name for name in single_names), "X gate not found in single group."

    # Verify the grouping of two-qubit gates.
    # For each node in the even group, the lower qubit index should be even.
    for node in even:
        q0 = node.qargs[0]._index  # noqa: SLF001
        q1 = node.qargs[1]._index  # noqa: SLF001
        assert min(q0, q1) % 2 == 0, f"Node with qubits {q0, q1} not in even group."

    # For each node in the odd group, the lower qubit index should be odd.
    for node in odd:
        q0 = node.qargs[0]._index  # noqa: SLF001
        q1 = node.qargs[1]._index  # noqa: SLF001
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

    num_traj = 1
    max_bond_dim = 4
    threshold = 1e-12
    min_bond_dim = 2
    observable = Observable(Z(), 0)
    sim_params = StrongSimParams([observable], num_traj, max_bond_dim, min_bond_dim, threshold)
    copy.deepcopy(mps0.tensors)
    apply_two_qubit_gate(mps0, node, sim_params)
    mps0.normalize(decomposition="SVD")
    for i, element in enumerate(mps0.to_vec()):
        if i == 11:
            np.testing.assert_allclose(np.abs(element), 1, atol=1e-15)
        else:
            np.testing.assert_allclose(np.abs(element), 0, atol=1e-15)


def test_create_local_noise_model() -> None:
    """Test the create_local_noise_model function.

    This test creates a global noise model with various noise processes and tests the creation
    of local noise models for different gate positions. It verifies that only the relevant
    noise processes are included in the local model based on the gate's site range.
    """
    # Create a global noise model with various processes
    global_processes = [
        {"name": "paulix", "sites": [0], "strength": 0.01},
        {"name": "paulix", "sites": [1], "strength": 0.02},
        {"name": "paulix", "sites": [2], "strength": 0.03},
        {"name": "paulix", "sites": [3], "strength": 0.04},
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": 0.05},
        {"name": "crosstalk_xx", "sites": [1, 2], "strength": 0.06},
        {"name": "crosstalk_xx", "sites": [2, 3], "strength": 0.07},
        {"name": "crosstalk_yy", "sites": [3, 4], "strength": 0.08},
        {"name": "crosstalk_xy", "sites": [0, 1], "strength": 0.09},
        {"name": "crosstalk_yx", "sites": [1, 2], "strength": 0.10},
    ]
    global_noise_model = NoiseModel(global_processes)

    # Test case 1: Gate acting on sites [1, 2]
    local_model_1 = create_local_noise_model(global_noise_model, 1, 2)

    # Should include: bitflip on sites 1, 2 and crosstalk_xx, crosstalk_yx on [1, 2]
    expected_processes_1 = [
        {"name": "paulix", "sites": [1], "strength": 0.02},
        {"name": "paulix", "sites": [2], "strength": 0.03},
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
        {"name": "paulix", "sites": [0], "strength": 0.01},
        {"name": "paulix", "sites": [1], "strength": 0.02},
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
        {"name": "paulix", "sites": [2], "strength": 0.03},
        {"name": "paulix", "sites": [3], "strength": 0.04},
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
        {"name": "paulix", "sites": [1], "strength": 0.02},
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
        {"name": "paulix", "sites": [1], "strength": 0.02},
        {"name": "paulix", "sites": [2], "strength": 0.03},
        {"name": "paulix", "sites": [3], "strength": 0.04},
        {"name": "crosstalk_xx", "sites": [1, 2], "strength": 0.06},
        {"name": "crosstalk_yx", "sites": [1, 2], "strength": 0.10},
        {"name": "crosstalk_xx", "sites": [2, 3], "strength": 0.07},
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


def test_digital_tjm_strong() -> None:
    """Test the digital_tjm function for strong simulation.

    This test creates a random MPS and a circuit with a CX gate, sets up strong simulation parameters,
    and runs digital_tjm. The test verifies that the simulation completes without errors.
    """
    length = 4
    mps0 = MPS(length, state="random")
    mps0.normalize()

    qc = QuantumCircuit(length)
    qc.cx(1, 3)

    num_traj = 1
    max_bond_dim = 4
    min_bond_dim = 2
    threshold = 1e-12
    observable = Observable(Z(), 0)
    sim_params = StrongSimParams([observable], num_traj, max_bond_dim, min_bond_dim, threshold)
    args = 0, mps0, None, sim_params, qc
    digital_tjm(args)


def test_digital_tjm_weak() -> None:
    """Test the digital_tjm function for weak simulation.

    This test creates a random MPS and a circuit with a CX gate, sets up weak simulation parameters,
    and runs digital_tjm. The test verifies that the simulation completes and measurements are obtained.
    """
    length = 4
    mps0 = MPS(length, state="random")
    mps0.normalize()

    qc = QuantumCircuit(length)
    qc.cx(1, 3)

    max_bond_dim = 4
    min_bond_dim = 2
    threshold = 1e-12
    shots = 10
    sim_params = WeakSimParams(shots, max_bond_dim, min_bond_dim, threshold)
    args = 0, mps0, None, sim_params, qc
    digital_tjm(args)


def test_noisy_digital_tjm_matches_reference() -> None:
    """Noisy circuit TJM should match hardcoded Qiskit reference within tolerance.

    Circuit: for layer k, apply k repetitions of rzz(0.5) on (0,1) and (1,2) for a 3-qubit chain.
    Noise model: single-qubit bitflip on each qubit and crosstalk_xx on each neighboring pair,
    both with strength 0.01. We compare Z-expectations on sites 0,1,2 over layers 0..5.
    """
    num_qubits = 3
    num_layers = 5  # compare layers 0..5 (including initial)
    noise_factor = 0.01
    num_traj = 1000  # Monte Carlo trajectories

    # Hardcoded Qiskit reference results (rows: qubit 0,1,2)
    reference = np.array([
        [1.0, 0.9607894391523233, 0.9231163463866354, 0.8869204367171571, 0.8521437889662108, 0.8187307530779814],
        [1.0, 0.9231163463866359, 0.8521437889662113, 0.7866278610665535, 0.726149037073691, 0.6703200460356394],
        [1.0, 0.9607894391523233, 0.9231163463866354, 0.8869204367171571, 0.8521437889662108, 0.8187307530779814],
    ])

    # YAQS noise model: bitflip on each site and crosstalk_xx on neighbors
    processes = [{"name": "paulix", "sites": [i], "strength": noise_factor} for i in range(num_qubits)] + [
        {"name": "crosstalk_xx", "sites": [i, i + 1], "strength": noise_factor} for i in range(num_qubits - 1)
    ]
    noise_model = NoiseModel(processes)

    # Collect TJM results per qubit across layers
    tjm_results = np.zeros_like(reference)
    for k in range(num_layers + 1):
        qc = QuantumCircuit(num_qubits)
        for _ in range(k):
            qc.rzz(0.5, 0, 1)
            qc.rzz(0.5, 1, 2)

        observables = [Observable(Z(), i) for i in range(num_qubits)]
        sim_params = StrongSimParams(observables, num_traj=num_traj)
        state = MPS(num_qubits, state="zeros", pad=2)
        simulator.run(state, qc, sim_params, noise_model, parallel=False)

        for q in range(num_qubits):
            # results is shape (1,) for StrongSim without layer sampling
            res_arr = sim_params.observables[q].results
            assert res_arr is not None
            tjm_results[q, k] = float(np.real(res_arr[0]))

    # Compare within tolerance
    tol = 0.1
    diff = np.abs(tjm_results - reference)
    assert np.all(diff <= tol), f"Noisy circuit TJM mismatch. max|diff|={diff.max():.4f} > {tol}"
