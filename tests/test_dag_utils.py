import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGOpNode

from yaqs.circuits.dag.dag_utils import (
    convert_dag_to_tensor_algorithm,
    get_temporal_zone,
    check_longest_gate,
    select_starting_point
)


def test_convert_dag_to_tensor_algorithm_single_qubit_gate():
    qc = QuantumCircuit(1)
    qc.x(0)
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Should have one gate from a single X."
    gate = gates[0]
    assert gate.name.lower() == "x", "Gate name should match 'x'."
    assert gate.sites == [0], "Gate should act on qubit 0."


def test_convert_dag_to_tensor_algorithm_two_qubit_gate():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Should have one 2-qubit gate (CX)."
    gate = gates[0]
    assert gate.name.lower() in ["cx", "cnot"], "Gate name should match CX/CNOT."
    assert gate.sites == [0, 1], "Gate should act on qubits 0 and 1."


def test_convert_dag_to_tensor_algorithm_single_dagopnode():
    qc = QuantumCircuit(1)
    qc.rx(np.pi/4, 0)
    dag = circuit_to_dag(qc)
    node = dag.op_nodes()[0]
    assert isinstance(node, DAGOpNode)

    gates = convert_dag_to_tensor_algorithm(node)
    assert len(gates) == 1
    gate = gates[0]
    assert gate.name.lower() == "rx", "Gate name should match 'rx'."
    assert gate.params == [np.pi/4], "Check that gate captured the rotation parameter."
    assert gate.sites == [0], "Gate acts on qubit 0."


def test_convert_dag_to_tensor_algorithm_ignores_measure_barrier():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.barrier()
    qc.measure_all()
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1
    gate = gates[0]
    assert gate.name.lower() == "x"


def test_get_temporal_zone_simple():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.cx(1, 2)
    dag = circuit_to_dag(qc)

    new_dag = get_temporal_zone(dag, [0,1])
    new_nodes = new_dag.op_nodes()
    assert len(new_nodes) == 2, "Should only have the 2 single-qubit gates in the zone."


def test_check_longest_gate():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(0, 2)  # distance=3
    dag = circuit_to_dag(qc)

    dist = check_longest_gate(dag)
    assert dist == 3, f"Longest distance should be 3, got {dist}"


def test_select_starting_point_even_odd():
    N = 4
    qc = QuantumCircuit(N)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)

    first_iter, second_iter = select_starting_point(N, dag)
    assert list(first_iter) == [0, 2], "Expected the default even qubits first."
    assert list(second_iter) == [1], "Then the odd qubit pairs."


def test_select_starting_point_odd():
    N = 4
    qc = QuantumCircuit(N)
    qc.x(0)
    qc.x(1)
    qc.cx(1, 2)  # index=2 => "odd" scenario in your code

    dag = circuit_to_dag(qc)
    first_iter, second_iter = select_starting_point(N, dag)
    assert list(first_iter) == [1, 2], "Expected odd qubits first."
    assert list(second_iter) == [0], "Then the even qubit pairs."
