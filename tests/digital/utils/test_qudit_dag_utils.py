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
