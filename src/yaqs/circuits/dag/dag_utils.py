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


def same_gate(node1, node2):
    """
    Returns True if node1 and node2 are considered the same gate.
    Here we compare the operation name and the set of qubits.
    (You can adjust this if you need to consider parameters, etc.)
    """
    return (node1.op.name == node2.op.name) and (set(node1.qargs) == set(node2.qargs))



def get_restricted_temporal_zone(dag: DAGCircuit, qubits: list[int], pending_queue: list):
    """
    Constructs a temporal zone for the given adjacent qubits (provided as indices).

    For each layer in the DAG:
      • If an operation acts entirely on the zone (i.e. its qubits are a subset of the zone),
        the operation is applied immediately.
      • If an operation exactly covers the zone (a full two-qubit gate), it is applied and
        then processing of the zone is stopped.
      • If an operation touches only part of the zone (i.e. a long-range gate), then:
            – If that gate is not already in pending_queue, add it (and do not apply it yet)
            – Otherwise, remove it from the queue and apply it.
    
    Returns a tuple: (new_dag, lr_gate)
      • new_dag is the restricted (accumulated) DAG for the zone.
      • lr_gate is the long-range gate that was just released (if any), or None.
    """
    new_dag = dag.copy_empty_like()
    layers = list(dag.multigraph_layers())
    # Build the set of qubit objects for the zone.
    qubits_to_check = {dag.qubits[q] for q in range(min(qubits), max(qubits) + 1)}
    lr_gate = None
    stop_zone = False  # flag to indicate that we should stop processing further layers

    for layer in layers:
        for node in layer:
            if not isinstance(node, DAGOpNode):
                continue

            # Remove measure and barrier nodes immediately.
            if node.op.name in ['measure', 'barrier']:
                dag.remove_op_node(node)
                continue

            node_qubits = set(node.qargs)

            # If the gate is entirely contained in the zone, add it.
            if node_qubits < qubits_to_check:
                new_dag.apply_operation_back(node.op, node.qargs)
                dag.remove_op_node(node)
            # If the gate exactly covers the zone (i.e. a two-qubit gate that fully entangles the zone)
            elif node_qubits == qubits_to_check:
                new_dag.apply_operation_back(node.op, node.qargs)
                dag.remove_op_node(node)
                # Mark that we have reached a full-entangling gate and break out
                stop_zone = True
                break
            # If the gate touches part of the zone (long-range gate)
            elif node_qubits & qubits_to_check:
                if not any(same_gate(node, pending_node) for pending_node in pending_queue):
                    # First occurrence: add it to the pending queue and stop processing further.
                    pending_queue.append(node)
                    stop_zone = True
                    break
                else:
                    # Second occurrence: remove from pending and apply it.
                    for i, pending_node in enumerate(pending_queue):
                        if same_gate(node, pending_node):
                            pending_queue.pop(i)
                            break
                    new_dag.apply_operation_back(node.op, node.qargs)
                    dag.remove_op_node(node)
                    lr_gate = node
                    stop_zone = True
                    break
            # Otherwise, ignore operations that do not affect the zone.
        if stop_zone:
            # Once a stopping gate (full two-qubit or long-range gate) is encountered,
            # we do not process later layers.
            break

    return new_dag, lr_gate


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