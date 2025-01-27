import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

def test_convert_dag_to_tensor_algorithm_single_qubit_gate():
    from yaqs.circuits.dag.dag_utils import convert_dag_to_tensor_algorithm

    qc = QuantumCircuit(1)
    qc.x(0)
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Should have one gate from a single X."
    gate = gates[0]
    assert gate.name.lower() == "x", "Gate name should match 'x'."
    assert gate.sites == [0], "Gate should act on qubit 0."


def test_convert_dag_to_tensor_algorithm_two_qubit_gate():
    from yaqs.circuits.dag.dag_utils import convert_dag_to_tensor_algorithm

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Should have one 2-qubit gate (CX)."
    gate = gates[0]
    assert gate.name.lower() in ["cx", "cnot"], "Gate name should match CX/CNOT."
    assert gate.sites == [0, 1], "Gate should act on qubits 0 and 1."


def test_convert_dag_to_tensor_algorithm_single_dagopnode():
    from qiskit.dagcircuit.dagnode import DAGOpNode
    from yaqs.circuits.dag.dag_utils import convert_dag_to_tensor_algorithm

    qc = QuantumCircuit(1)
    qc.rx(np.pi/4, 0)
    dag = circuit_to_dag(qc)
    node = dag.op_nodes()[0]
    assert isinstance(node, DAGOpNode)

    gates = convert_dag_to_tensor_algorithm(node)
    assert len(gates) == 1
    gate = gates[0]
    assert gate.name.lower() == "rx", "Gate name should match 'rx'."
    assert gate.theta == np.pi/4, "Check that gate captured the rotation parameter."
    assert gate.sites == [0], "Gate acts on qubit 0."


def test_convert_dag_to_tensor_algorithm_ignores_measure_barrier():
    from yaqs.circuits.dag.dag_utils import convert_dag_to_tensor_algorithm

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
    from yaqs.circuits.dag.dag_utils import get_temporal_zone

    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.cx(1, 2)
    dag = circuit_to_dag(qc)

    new_dag = get_temporal_zone(dag, [0, 1])
    new_nodes = new_dag.op_nodes()
    assert len(new_nodes) == 2, "Should only have the 2 single-qubit gates in the zone."


def test_check_longest_gate():
    from yaqs.circuits.dag.dag_utils import check_longest_gate

    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)

    dist = check_longest_gate(dag)
    assert dist == 3, f"Longest distance should be 3, got {dist}"


def test_select_starting_point_even_odd():
    from yaqs.circuits.dag.dag_utils import select_starting_point

    N = 4
    qc = QuantumCircuit(N)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)

    first_iter, second_iter = select_starting_point(N, dag)
    assert first_iter == range(0, 3, 2), "Expected the default even qubits first."
    assert second_iter == range(1, 3, 2), "Then the odd qubit pairs."


def test_select_starting_point_odd():
    from yaqs.circuits.dag.dag_utils import select_starting_point

    N = 4
    qc = QuantumCircuit(N)
    qc.cx(1, 2)

    dag = circuit_to_dag(qc)
    first_iter, second_iter = select_starting_point(N, dag)

    assert first_iter == range(1, 3, 2), "Expected odd qubits first."
    assert second_iter == range(0, 3, 2), "Then the even qubit pairs."
