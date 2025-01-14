import numpy as np

from qiskit._accelerate.circuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode

from yaqs.general.libraries.gate_library import GateLibrary


def convert_dag_to_tensor_algorithm(dag: DAGCircuit):
    """
    Converts a DAGCircuit into a list of gate objects from the TensorLibrary.

    Args:
        dag: The DAGCircuit (or a single DAGOpNode) representing a quantum operation.

    Returns:
        A list of gate objects, each containing attributes such as .tensor, .sites, etc.
    """
    algorithm = []

    if isinstance(dag, DAGOpNode):
        # Single node DAG
        gate = dag
        name = gate.op.name

        attr = getattr(GateLibrary, name)
        gate_object = attr()
        if gate.op.params:
            gate_object.set_params(gate.op.params)

        sites = [gate.qargs[0]._index]
        if len(gate.qargs) == 2:
            sites.append(gate.qargs[1]._index)
        if len(gate.qargs) == 3:
            sites.append(gate.qargs[1]._index)
            sites.append(gate.qargs[2]._index)

        gate_object.set_sites(*sites)
        algorithm.append(gate_object)
    else:
        # Multi-node DAG
        for gate in dag.op_nodes():
            name = gate.op.name
            if name in ['measure', 'barrier']:
                continue

            attr = getattr(GateLibrary, name)
            gate_object = attr()

            if gate.op.params:
                gate_object.set_params(gate.op.params)

            sites = [gate.qargs[0]._index]
            if len(gate.qargs) == 2:
                sites.append(gate.qargs[1]._index)
            if len(gate.qargs) == 3:
                sites.append(gate.qargs[1]._index)
                sites.append(gate.qargs[2]._index)

            gate_object.set_sites(*sites)
            algorithm.append(gate_object)

    return algorithm


def get_temporal_zone(dag: DAGCircuit, qubits: list[int]):
    """
    Extracts the temporal zone from a DAGCircuit for the specified qubits.
    The temporal zone is the subset of operations acting only on these qubits
    until they no longer participate in further gates.

    Args:
        dag: The input DAGCircuit.
        qubits: List of qubit indices for which to retrieve the temporal zone.

    Returns:
        A new DAGCircuit containing only the operations within the temporal zone
        of the specified qubits.
    """
    new_dag = dag.copy_empty_like()
    layers = list(dag.multigraph_layers())
    qubits_to_check = set()
    for qubit in range(min(qubits), max(qubits)+1):
        qubits_to_check.add(dag.qubits[qubit])

    for layer in layers:
        for node in layer:
            if isinstance(node, DAGOpNode):
                qubit_set = set(node.qargs)

                # Gate is entirely within cone
                if qubit_set <= qubits_to_check:
                    if node.op.name in ['measure', 'barrier']:
                        dag.remove_op_node(node)
                        continue
                    new_dag.apply_operation_back(node.op, node.qargs)
                    dag.remove_op_node(node)
                else:
                    # If there is partial overlap, remove those qubits from the cone
                    if node.op.name in ['measure', 'barrier']:
                        dag.remove_op_node(node)
                        continue
                    for item in qubit_set & qubits_to_check:
                        qubits_to_check.remove(item)

        # Once no qubits remain in the cone, stop
        if len(qubits_to_check) == 0:
            break

    return new_dag


def check_longest_gate(dag: DAGCircuit):
    """
    Checks the maximum 'distance' between qubits in any multi-qubit gate
    present in the first layer of the DAGCircuit.

    Args:
        dag: The DAGCircuit in question.

    Returns:
        The integer distance (in terms of qubit indices) of the largest gate
        encountered. A value of 1 or 2 indicates nearest-neighbor gates only.
    """
    largest_distance = 1
    for layer in dag.layers():
        first_layer = layer
        break

    if 'first_layer' in locals():
        from qiskit.converters import dag_to_circuit
        layer_circuit = dag_to_circuit(first_layer['graph'])
        for gate in layer_circuit.data:
            if gate.operation.num_qubits > 1:
                distance = abs(gate.qubits[0]._index - gate.qubits[-1]._index) + 1
                if distance > largest_distance:
                    largest_distance = distance

    return largest_distance


def select_starting_point(N: int, dag: DAGCircuit):
    """
    Determines which set of neighboring qubits (even-even or odd-odd) to start with
    based on the arrangement of gates in the first layer of the DAG.

    Args:
        N: Total number of qubits (or sites) in the system.
        dag: The DAGCircuit used to inspect the first set of gates.

    Returns:
        A tuple (first_iterator, second_iterator) containing ranges of qubit indices
        for a checkerboard-like application of gates.
    """
    assert N > 1

    for layer in dag.layers():
        first_layer = layer
        break

    first_iterator = range(0, N-1, 2)
    second_iterator = range(1, N-1, 2)
    odd = False

    if 'first_layer' in locals():
        from qiskit.converters import dag_to_circuit
        layer_circuit = dag_to_circuit(first_layer['graph'])
        for i, gate in enumerate(layer_circuit.data):
            # If there's a two-qubit gate at an odd position in the list, switch
            if gate.operation.num_qubits == 2:
                if i % 2 != 0:
                    odd = True
                break

        if odd:
            first_iterator = range(1, N-1, 2)
            second_iterator = range(0, N-1, 2)

    return first_iterator, second_iterator