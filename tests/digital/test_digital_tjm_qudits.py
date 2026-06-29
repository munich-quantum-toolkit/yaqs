# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the qudit Tensor Jump Method (mqt.yaqs.digital.qudit_tjm)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mqt.qudits")

from mqt.qudits.quantum_circuit import QuantumCircuit as QuditCircuit

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams, WeakSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.digital.qudit_tjm import (
    apply_single_qudit_gate,
    apply_two_qudit_gate,
    process_layer_qudit,
    qudit_tjm,
)
from mqt.yaqs.digital.utils.qudit_dag_utils import circuit_to_dag


def test_process_layer_qudit_groups_by_parity() -> None:
    """Single-qudit gates and two-qudit gates (grouped by parity) are correctly separated."""
    circuit = QuditCircuit(8, [2] * 8)
    circuit.h(0)
    circuit.cx([1, 2])
    circuit.cx([3, 4])
    circuit.cx([6, 7])
    circuit.h(5)

    dag = circuit_to_dag(circuit)
    single_nodes, even_nodes, odd_nodes = process_layer_qudit(dag)

    assert [n.target_qudits[0] for n in single_nodes] == [0, 5]
    assert [n.target_qudits for n in even_nodes] == [[6, 7]]
    assert [n.target_qudits for n in odd_nodes] == [[1, 2], [3, 4]]


def test_apply_single_qudit_gate_hadamard() -> None:
    """Applying H to |0> on a single qubit-dimension qudit yields the |+> state."""
    circuit = QuditCircuit(1, [2])
    circuit.h(0)
    node = circuit_to_dag(circuit).front_layer()[0]

    state = State(1, physical_dimensions=[2], initial="zeros")
    state.ensure_encoded("mps")
    mps = state.mps
    apply_single_qudit_gate(mps, node)

    expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(mps.tensors[0][:, 0, 0], expected)


def test_apply_two_qudit_gate_adjacent_mixed_dimension() -> None:
    """A CX gate from a qubit control to an adjacent qutrit target flips the qutrit level."""
    circuit = QuditCircuit(2, [2, 3])
    circuit.x(0)
    circuit.cx([0, 1])
    dag = circuit_to_dag(circuit)

    state = State(2, physical_dimensions=[2, 3], initial="zeros")
    state.ensure_encoded("mps")
    mps = state.mps
    sim_params = StrongSimParams(get_state=True)

    x_node = dag.front_layer()[0]
    apply_single_qudit_gate(mps, x_node)
    dag.remove_op_node(x_node)

    cx_node = dag.front_layer()[0]
    apply_two_qudit_gate(mps, cx_node, sim_params)

    np.testing.assert_allclose(np.abs(mps.tensors[0][:, 0, 0]), [0, 1], atol=1e-9)
    np.testing.assert_allclose(np.abs(mps.tensors[1][:, 0, 0]), [0, 1, 0], atol=1e-9)


def test_apply_two_qudit_gate_long_range_swap_network() -> None:
    """A CX gate spanning non-adjacent qudits is correctly applied via the SWAP network."""
    circuit = QuditCircuit(3, [2, 3, 2])
    circuit.x(0)
    circuit.cx([0, 2])
    dag = circuit_to_dag(circuit)

    state = State(3, physical_dimensions=[2, 3, 2], initial="zeros")
    state.ensure_encoded("mps")
    mps = state.mps
    sim_params = StrongSimParams(get_state=True)

    x_node = dag.front_layer()[0]
    apply_single_qudit_gate(mps, x_node)
    dag.remove_op_node(x_node)

    cx_node = dag.front_layer()[0]
    apply_two_qudit_gate(mps, cx_node, sim_params)

    np.testing.assert_allclose(np.abs(mps.tensors[0][:, 0, 0]), [0, 1], atol=1e-9)
    np.testing.assert_allclose(np.abs(mps.tensors[1][:, 0, 0]), [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(np.abs(mps.tensors[2][:, 0, 0]), [0, 1], atol=1e-9)


def test_qudit_tjm_end_to_end_no_noise() -> None:
    """qudit_tjm correctly simulates a small mixed-dimension circuit end to end."""
    circuit = QuditCircuit(2, [2, 3])
    circuit.x(0)
    circuit.cx([0, 1])

    state = State(2, physical_dimensions=[2, 3], initial="zeros")
    state.ensure_encoded("mps")
    mps = state.mps
    sim_params = StrongSimParams(get_state=True)

    _, _, final_mps = qudit_tjm((0, mps, None, sim_params, circuit))

    np.testing.assert_allclose(np.abs(final_mps.tensors[0][:, 0, 0]), [0, 1], atol=1e-9)
    np.testing.assert_allclose(np.abs(final_mps.tensors[1][:, 0, 0]), [0, 1, 0], atol=1e-9)


def test_qudit_tjm_with_noise_runs_without_error() -> None:
    """qudit_tjm applies dissipation/stochastic jumps without crashing when a noise model is given."""
    circuit = QuditCircuit(2, [2, 2])
    circuit.x(0)
    circuit.cx([0, 1])

    state = State(2, physical_dimensions=[2, 2], initial="zeros")
    state.ensure_encoded("mps")
    mps = state.mps
    sim_params = StrongSimParams(get_state=True, random_seed=0)

    lowering = np.array([[0, 0], [1, 0]], dtype=complex)
    noise_model = NoiseModel(processes=[{"name": "custom", "sites": [0], "strength": 0.1, "matrix": lowering}])

    _, _, final_mps = qudit_tjm((0, mps, noise_model, sim_params, circuit))

    assert final_mps is not None
    assert final_mps.norm() == pytest.approx(1.0, abs=1e-6)


def test_simulator_run_dispatches_to_qudit_path() -> None:
    """Simulator.run() correctly dispatches a qudit circuit to the qudit backend."""
    circuit = QuditCircuit(2, [2, 3])
    circuit.x(0)
    circuit.cx([0, 1])

    state = State(2, physical_dimensions=[2, 3], initial="zeros")
    sim_params = StrongSimParams(get_state=True)

    result = Simulator(parallel=False, show_progress=False).run(state, circuit, sim_params)

    assert result.output_state is not None
    final_mps = result.output_state.mps
    np.testing.assert_allclose(np.abs(final_mps.tensors[0][:, 0, 0]), [0, 1], atol=1e-9)
    np.testing.assert_allclose(np.abs(final_mps.tensors[1][:, 0, 0]), [0, 1, 0], atol=1e-9)


def test_qudit_tjm_weak_sim_no_noise_returns_shot_counts() -> None:
    """qudit_tjm with WeakSimParams and no noise returns a single-call shot-count histogram."""
    circuit = QuditCircuit(2, [2, 3])
    circuit.x(0)
    circuit.cx([0, 1])

    state = State(2, physical_dimensions=[2, 3], initial="zeros")
    state.ensure_encoded("mps")
    mps = state.mps
    sim_params = WeakSimParams(shots=20, random_seed=0)

    counts, diagnostics, final_mps = qudit_tjm((0, mps, None, sim_params, circuit))

    assert diagnostics is None
    assert final_mps is None
    # X(0) -> |1>|0>, CX flips the qutrit target -> |1>|1>: mixed-radix outcome 1*1 + 1*2 = 3.
    assert counts == {3: 20}


def test_qudit_tjm_weak_sim_with_noise_returns_single_shot() -> None:
    """qudit_tjm with WeakSimParams and noise returns exactly one shot per call."""
    circuit = QuditCircuit(2, [2, 2])
    circuit.x(0)
    circuit.cx([0, 1])

    state = State(2, physical_dimensions=[2, 2], initial="zeros")
    state.ensure_encoded("mps")
    mps = state.mps
    sim_params = WeakSimParams(shots=20, random_seed=0)

    lowering = np.array([[0, 0], [1, 0]], dtype=complex)
    noise_model = NoiseModel(processes=[{"name": "custom", "sites": [0], "strength": 0.1, "matrix": lowering}])

    counts, _, _ = qudit_tjm((0, mps, noise_model, sim_params, circuit))

    assert sum(counts.values()) == 1


def test_simulator_run_dispatches_weak_sim_qudit_path() -> None:
    """Simulator.run() correctly dispatches a qudit circuit with WeakSimParams end to end."""
    circuit = QuditCircuit(2, [2, 3])
    circuit.x(0)
    circuit.cx([0, 1])

    state = State(2, physical_dimensions=[2, 3], initial="zeros")
    sim_params = WeakSimParams(shots=20, random_seed=0)

    result = Simulator(parallel=False, show_progress=False).run(state, circuit, sim_params)

    assert sum(result.counts.values()) == 20


def test_simulator_run_dispatches_weak_sim_qudit_path_with_noise() -> None:
    """Simulator.run() with WeakSimParams and noise runs one trajectory per shot for qudits."""
    circuit = QuditCircuit(2, [2, 2])
    circuit.x(0)
    circuit.cx([0, 1])

    state = State(2, physical_dimensions=[2, 2], initial="zeros")
    sim_params = WeakSimParams(shots=15, random_seed=0)

    lowering = np.array([[0, 0], [1, 0]], dtype=complex)
    noise_model = NoiseModel(processes=[{"name": "custom", "sites": [0], "strength": 0.1, "matrix": lowering}])

    result = Simulator(parallel=False, show_progress=False).run(state, circuit, sim_params, noise_model)

    assert sum(result.counts.values()) == 15
