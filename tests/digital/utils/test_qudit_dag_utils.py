# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the QuditDAG utility functions.

Covers node creation, dependency wiring, level tracking, topological layer
iteration, ``check_longest_gate``, ``get_temporal_zone``, and the
``circuit_to_dag`` / ``dag_to_circuit`` round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest
from mqt.qudits.quantum_circuit import QuantumCircuit

from mqt.yaqs.digital.utils.qudit_dag_utils import (
    QuditDAG,
    QuditOpNode,
    check_longest_gate,
    circuit_to_dag,
    dag_to_circuit,
    get_temporal_zone,
)

@pytest.fixture
def simple_circuit() -> QuantumCircuit:
    """3-qudit circuit: CX(1,2), RZ(1), R(0) — one dependency."""
    qc = QuantumCircuit(3, [2, 5, 4])
    qc.cx([1, 2])
    qc.rz(1, [0, 3, np.pi / 2])
    qc.r(0, [0, 1, np.pi, np.pi / 2])
    return qc


@pytest.fixture
def linear_circuit() -> QuantumCircuit:
    """Single qudit, three sequential single-qudit gates — chain dependency."""
    qc = QuantumCircuit(1, [3])
    qc.rz(0, [0, 1, np.pi / 4])
    qc.rz(0, [1, 2, np.pi / 4])
    qc.rz(0, [0, 2, np.pi / 2])
    return qc


@pytest.fixture
def long_range_circuit() -> QuantumCircuit:
    """4-qudit circuit with a CX spanning qudits 0 and 3 (long-range)."""
    qc = QuantumCircuit(4, [2, 2, 2, 2])
    qc.cx([0, 3])
    return qc

class TestCircuitToDag:
    def test_node_count_matches_gates(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        assert len(dag.nodes) == 3

    def test_dimensions_preserved(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        assert dag.dimensions == [2, 5, 4]

    def test_num_qudits(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        assert dag.num_qudits == 3

    def test_nodes_are_op_nodes(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        assert all(isinstance(n, QuditOpNode) for n in dag.nodes)

    def test_op_names(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        names = [n.op_name for n in dag.nodes]
        assert names == ["cx", "rz", "rxy"]

    def test_gate_object_is_stored(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        for node in dag.nodes:
            assert node.gate is not None

    def test_to_matrix_accessible_via_node(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        for node in dag.nodes:
            m = node.gate.to_matrix(identities=0)
            assert m.ndim == 2

class TestDependencies:
    def test_independent_gates_have_no_deps(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        cx_node = dag.nodes[0]   
        rxy_node = dag.nodes[2]  
        assert len(cx_node.dependencies) == 0
        assert len(rxy_node.dependencies) == 0

    def test_sequential_gate_has_dep(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        rz_node = dag.nodes[1]   
        assert dag.nodes[0] in rz_node.dependencies

    def test_chain_on_single_qudit(self, linear_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(linear_circuit)
        assert dag.nodes[0] in dag.nodes[1].dependencies
        assert dag.nodes[1] in dag.nodes[2].dependencies
        assert dag.nodes[0] not in dag.nodes[2].dependencies  

class TestLevelTracking:
    def test_cx_levels(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        cx_node = dag.nodes[0]
        assert cx_node.levels[1] == [0, 1]
        assert cx_node.levels[2] == [0, 1]

    def test_rz_levels_with_high_transition(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        rz_node = dag.nodes[1]
        assert rz_node.levels[1] == [0, 3]

    def test_r_gate_levels(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        r_node = dag.nodes[2]
        assert r_node.levels[0] == [0, 1]

    def test_full_range_when_lev_a_equals_lev_b(self) -> None:
        qc = QuantumCircuit(1, [4])
        qc.h(0)
        dag = circuit_to_dag(qc)
        assert dag.nodes[0].levels[0] == [0, 1, 2, 3]

    def test_levels_keys_match_target_qudits(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        for node in dag.nodes:
            assert set(node.levels.keys()) == set(node.target_qudits)

class TestLayers:
    def test_independent_gates_in_same_layer(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        layers = list(dag.layers())
        layer0_names = {n.op_name for n in layers[0]["graph"]}
        assert "cx" in layer0_names
        assert "rxy" in layer0_names

    def test_dependent_gate_in_later_layer(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        layers = list(dag.layers())
        layer1_names = {n.op_name for n in layers[1]["graph"]}
        assert "rz" in layer1_names

    def test_chain_yields_one_gate_per_layer(self, linear_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(linear_circuit)
        layers = list(dag.layers())
        assert len(layers) == 3
        for layer in layers:
            assert len(layer["graph"]) == 1

    def test_layer_indices_are_sequential(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        indices = [layer["index"] for layer in dag.layers()]
        assert indices == list(range(len(indices)))

    def test_all_nodes_covered_by_layers(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        nodes_in_layers = [n for layer in dag.layers() for n in layer["graph"]]
        assert set(nodes_in_layers) == set(dag.nodes)

    def test_multigraph_layers_alias(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        from_layers = list(dag.layers())
        from_multi = list(dag.multigraph_layers())
        assert [l["index"] for l in from_layers] == [l["index"] for l in from_multi]

class TestNodeOperations:
    def test_op_nodes_returns_all(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        assert len(dag.op_nodes()) == 3

    def test_front_layer_initially_independent(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        front = dag.front_layer()
        front_names = {n.op_name for n in front}
        assert "cx" in front_names
        assert "rxy" in front_names
        assert "rz" not in front_names

    def test_remove_node_updates_successors(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        cx_node = dag.nodes[0]
        rz_node = dag.nodes[1]
        assert cx_node in rz_node.dependencies
        dag.remove_op_node(cx_node)
        assert cx_node not in rz_node.dependencies

    def test_remove_node_shrinks_op_nodes(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        dag.remove_op_node(dag.nodes[0])
        assert len(dag.op_nodes()) == 2

    def test_remove_nonexistent_node_is_noop(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        phantom = dag.nodes[0]
        dag.remove_op_node(phantom)
        dag.remove_op_node(phantom)  
        assert len(dag.op_nodes()) == 2

    def test_front_layer_after_removal(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        cx_node = dag.nodes[0]
        dag.remove_op_node(cx_node)
        front_names = {n.op_name for n in dag.front_layer()}
        assert "rz" in front_names 

class TestCheckLongestGate:
    def test_single_qudit_gate_span_is_one(self) -> None:
        qc = QuantumCircuit(2, [3, 3])
        qc.rz(0, [0, 1, np.pi / 2])
        dag = circuit_to_dag(qc)
        assert check_longest_gate(dag) == 1

    def test_nearest_neighbor_span_is_two(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        assert check_longest_gate(dag) == 2

    def test_long_range_gate_span(self, long_range_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(long_range_circuit)
        assert check_longest_gate(dag) == 4

    def test_empty_dag_returns_one(self) -> None:
        dag = QuditDAG(dimensions=[2, 2])
        assert check_longest_gate(dag) == 1

class TestGetTemporalZone:
    def test_extracts_gate_in_window(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        zone = get_temporal_zone(dag, [1, 2])
        zone_names = [n.op_name for n in zone.nodes]
        assert "cx" in zone_names

    def test_extracted_node_removed_from_dag(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        original_count = len(dag.nodes)
        get_temporal_zone(dag, [1, 2])
        assert len(dag.nodes) < original_count

    def test_gate_outside_window_not_extracted(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        zone = get_temporal_zone(dag, [1, 2])
        zone_names = [n.op_name for n in zone.nodes]
        assert "rxy" not in zone_names

    def test_zone_dimensions_match_original(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        zone = get_temporal_zone(dag, [0, 1])
        assert zone.dimensions == dag.dimensions

class TestDagToCircuit:
    def test_gate_count_preserved(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        qc2 = dag_to_circuit(dag)
        assert qc2.number_gates == simple_circuit.number_gates

    def test_dimensions_preserved(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        qc2 = dag_to_circuit(dag)
        assert list(qc2.dimensions) == list(simple_circuit.dimensions)

    def test_gate_objects_identical_after_roundtrip(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        qc2 = dag_to_circuit(dag)
        assert set(id(g) for g in qc2.instructions) == set(id(g) for g in simple_circuit.instructions)

class TestCopyEmptyLike:
    def test_empty_dag_has_no_nodes(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        empty = dag.copy_empty_like()
        assert len(empty.nodes) == 0

    def test_empty_dag_keeps_dimensions(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        empty = dag.copy_empty_like()
        assert empty.dimensions == dag.dimensions

class TestApplyOperationBack:
    def test_appended_node_appears_in_op_nodes(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        gate = simple_circuit.instructions[2]  
        initial_count = len(dag.nodes)
        dag.apply_operation_back(gate, [0])
        assert len(dag.nodes) == initial_count + 1

    def test_appended_node_has_correct_op_name(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        gate = simple_circuit.instructions[2]  
        node = dag.apply_operation_back(gate, [0])
        assert node.op_name == "rxy"

    def test_appended_node_levels_are_set(self, simple_circuit: QuantumCircuit) -> None:
        dag = circuit_to_dag(simple_circuit)
        gate = simple_circuit.instructions[2]
        node = dag.apply_operation_back(gate, [0])
        assert node.levels[0] == [0, 1]
