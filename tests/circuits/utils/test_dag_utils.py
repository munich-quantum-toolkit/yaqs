# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

""" Tests for the DAG utility functions.

This module contains unit tests for the DAG utility functions used in the conversion and processing
of quantum circuits. It verifies that the functions correctly extract gate operations from a DAGCircuit,
group nodes into single-qubit and two-qubit (even/odd) categories, ignore unsupported operations (such as
measure and barrier), and compute properties like the longest gate distance and appropriate starting point
ranges for gate application.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

from mqt.yaqs.circuits.utils.dag_utils import (
    check_longest_gate,
    convert_dag_to_tensor_algorithm,
    get_temporal_zone,
    select_starting_point,
)


def test_convert_dag_to_tensor_algorithm_single_qubit_gate() -> None:
    """Test converting a DAGCircuit with a single-qubit X gate.

    This test creates a quantum circuit with one qubit and applies an X gate.
    It then converts the circuit to a DAG and uses convert_dag_to_tensor_algorithm
    to extract the gate. The test verifies that exactly one gate is returned,
    the gate's name matches 'x' (case-insensitive), and it acts on qubit 0.
    """
    qc = QuantumCircuit(1)
    qc.x(0)
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Should have one gate from a single X."
    gate = gates[0]
    assert gate.name.lower() == "x", "Gate name should match 'x'."
    assert gate.sites == [0], "Gate should act on qubit 0."


def test_convert_dag_to_tensor_algorithm_two_qubit_gate() -> None:
    """Test converting a DAGCircuit with a two-qubit CX gate.

    This test creates a two-qubit circuit with a controlled-NOT gate (CX) and converts it
    to a DAG. It then verifies that the conversion produces one two-qubit gate with a name
    matching 'cx' or 'cnot' (case-insensitive) and that the gate acts on qubits 0 and 1.
    """
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Should have one 2-qubit gate (CX)."
    gate = gates[0]
    assert gate.name.lower() in {"cx", "cnot"}, "Gate name should match CX/CNOT."
    assert gate.sites == [0, 1], "Gate should act on qubits 0 and 1."


def test_convert_dag_to_tensor_algorithm_two_qubit_gate_flipped() -> None:
    """Test converting a DAGCircuit with a two-qubit CX gate where control and target are flipped.

    This test creates a two-qubit circuit with a CX gate applied from qubit 1 to qubit 0,
    converts the circuit to a DAG, and extracts the gate. It verifies that the gate's name
    matches 'cx' or 'cnot' and that the gate correctly reflects the flipped qubit order.
    """
    qc = QuantumCircuit(2)
    qc.cx(1, 0)
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Should have one 2-qubit gate (CX)."
    gate = gates[0]
    assert gate.name.lower() in {"cx", "cnot"}, "Gate name should match CX/CNOT."
    assert gate.sites == [1, 0], "Gate should act on qubits 1 and 0."


def test_convert_dag_to_tensor_algorithm_single_dagopnode() -> None:
    """Test converting a single DAGOpNode representing a single-qubit RX gate.

    This test creates a circuit with a single RX gate, converts it to a DAG,
    and then extracts the first op node. The conversion function should return one gate
    with name 'rx', capture the rotation parameter (theta) correctly, and specify that it acts on qubit 0.
    """
    qc = QuantumCircuit(1)
    qc.rx(np.pi / 4, 0)
    dag = circuit_to_dag(qc)
    node = dag.op_nodes()[0]
    assert isinstance(node, DAGOpNode)

    gates = convert_dag_to_tensor_algorithm(node)
    assert len(gates) == 1, "Expected one gate from a single DAGOpNode."
    gate = gates[0]
    assert gate.name.lower() == "rx", "Gate name should match 'rx'."
    assert gate.theta == np.pi / 4, "Gate should capture the rotation parameter (pi/4)."
    assert gate.sites == [0], "Gate should act on qubit 0."


def test_convert_dag_to_tensor_algorithm_ignores_measure_barrier() -> None:
    """Test that convert_dag_to_tensor_algorithm ignores measure and barrier nodes.

    This test constructs a two-qubit circuit that includes an X gate, a barrier, and measurement operations.
    After converting the circuit to a DAG and extracting gates, the function should return only the X gate,
    ignoring measure and barrier operations.
    """
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.barrier()
    qc.measure_all()
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Only the X gate should be extracted."
    gate = gates[0]
    assert gate.name.lower() == "x", "The extracted gate should be an X gate."


def test_get_temporal_zone_simple() -> None:
    """Test extracting the temporal zone for a subset of qubits from a DAGCircuit.

    This test creates a three-qubit circuit with two X gates and one CX gate.
    It then extracts the temporal zone for qubits 0 and 1, expecting only the two single-qubit X gates
    to be present in the temporal zone.
    """
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.cx(1, 2)
    dag = circuit_to_dag(qc)

    new_dag = get_temporal_zone(dag, [0, 1])
    new_nodes = new_dag.op_nodes()
    assert len(new_nodes) == 2, "Should only have the 2 single-qubit gates in the temporal zone."


def test_check_longest_gate() -> None:
    """Test the computation of the longest gate distance in the first layer of a DAGCircuit.

    This test creates a three-qubit circuit with two CX gates: one between qubits 0 and 2 and one
    between qubits 0 and 1. The function check_longest_gate should return 3, indicating that the maximum distance
    between involved qubits is 3.
    """
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)

    dist = check_longest_gate(dag)
    assert dist == 3, f"Longest distance should be 3, got {dist}"


def test_select_starting_point_even_odd() -> None:
    """Test selecting starting points for gate application using a checkerboard pattern when the first gate
       is on an even qubit.

    This test creates a 4-qubit circuit with a CX gate starting at qubit 0.
    The function select_starting_point should return ranges corresponding to even-odd pairings:
    first_iterator = range(0, 3, 2) and second_iterator = range(1, 3, 2).
    """
    N = 4
    qc = QuantumCircuit(N)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)

    first_iter, second_iter = select_starting_point(N, dag)
    assert first_iter == range(0, 3, 2), "Expected the default even qubits first."
    assert second_iter == range(1, 3, 2), "Then the odd qubit pairs."


def test_select_starting_point_odd() -> None:
    """Test selecting starting points when the first two-qubit gate starts at an odd qubit.

    This test creates a 4-qubit circuit with a CX gate starting at qubit 1.
    The function select_starting_point should return ranges with odd qubits first:
    first_iterator = range(1, 3, 2) and second_iterator = range(0, 3, 2).
    """
    N = 4
    qc = QuantumCircuit(N)
    qc.cx(1, 2)

    dag = circuit_to_dag(qc)
    first_iter, second_iter = select_starting_point(N, dag)

    assert first_iter == range(1, 3, 2), "Expected odd qubits first."
    assert second_iter == range(0, 3, 2), "Then the even qubit pairs."
