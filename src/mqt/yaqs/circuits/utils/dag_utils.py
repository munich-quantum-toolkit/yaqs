# Copyright (c) 2025 Chair for Design Automation, TUM

# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module implements conversion and processing functions for quantum circuits using their DAG representations.
It provides utilities to:
  - Convert a DAGCircuit into a list of gate objects from the GateLibrary.
  - Extract a temporal zone from a DAGCircuit for specified qubits.
  - Determine the maximum distance (in terms of qubit indices) of multi-qubit gates.
  - Select starting points for gate application based on a checkerboard pattern.

These functions facilitate the manipulation and analysis of quantum circuit representations.
"""


from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGOpNode

from ...core.libraries.gate_library import GateLibrary

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from qiskit.circuit import Qubit
    from qiskit.dagcircuit import DAGCircuit


def convert_dag_to_tensor_algorithm(dag: DAGCircuit) -> list[NDArray[np.complex128]]:
    """Convert a DAGCircuit into a list of gate objects from the GateLibrary.

    This function traverses the input DAGCircuit (or a single DAGOpNode) and creates a list of gate objects.
    For each node, it retrieves the corresponding gate class from the GateLibrary, initializes it, sets any
    parameters if present, and assigns the qubit indices (sites) on which the gate acts.

    Args:
        dag (DAGCircuit or DAGOpNode): The DAGCircuit (or a single DAGOpNode) representing a quantum operation.

    Returns:
        list[NDArray[np.complex128]]: A list of gate objects, each with attributes such as .tensor and .sites.
    """
    algorithm = []

    if isinstance(dag, DAGOpNode):
        # Single node DAG.
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
            sites.extend((gate.qargs[1]._index, gate.qargs[2]._index))

        gate_object.set_sites(*sites)
        algorithm.append(gate_object)
    else:
        # Multi-node DAG.
        for gate in dag.op_nodes():
            name = gate.op.name
            if name in {"measure", "barrier"}:
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


def get_temporal_zone(dag: DAGCircuit, qubits: list[int]) -> DAGCircuit:
    """Extract the temporal zone from a DAGCircuit for the specified qubits.

    The temporal zone is defined as the subset of operations (layers) acting solely on the specified qubits,
    continuing until those qubits no longer participate in any further operations. The function builds a new
    DAGCircuit containing only these operations.

    Args:
        dag (DAGCircuit): The input DAGCircuit.
        qubits (list[int]): List of qubit indices for which to extract the temporal zone.

    Returns:
        DAGCircuit: A new DAGCircuit containing only the operations within the temporal zone for the specified qubits.
    """
    new_dag = dag.copy_empty_like()
    layers = list(dag.multigraph_layers())
    qubits_to_check: set[Qubit] = set()
    qubits_to_check.update(dag.qubits[qubit] for qubit in range(min(qubits), max(qubits) + 1))

    for layer in layers:
        for node in layer:
            if isinstance(node, DAGOpNode):
                qubit_set = set(node.qargs)

                # If the gate acts entirely within the current cone of qubits.
                if qubit_set <= qubits_to_check:
                    if node.op.name in {"measure", "barrier"}:
                        dag.remove_op_node(node)
                        continue
                    new_dag.apply_operation_back(node.op, node.qargs)
                    dag.remove_op_node(node)
                else:
                    # For partial overlap, remove the overlapping qubits from the cone.
                    if node.op.name in {"measure", "barrier"}:
                        dag.remove_op_node(node)
                        continue
                    for item in qubit_set & qubits_to_check:
                        qubits_to_check.remove(item)

        # Stop once no qubits remain in the cone.
        if len(qubits_to_check) == 0:
            break

    return new_dag


def check_longest_gate(dag: DAGCircuit) -> int:
    """Determine the maximum distance between qubits in any multi-qubit gate in the first layer.

    This function inspects the first layer of the DAGCircuit and computes the distance between the first
    and last qubits involved in each multi-qubit gate. The distance is defined in terms of qubit indices.
    A result of 1 or 2 indicates that only nearest-neighbor gates are present.

    Args:
        dag (DAGCircuit): The DAGCircuit to inspect.

    Returns:
        int: The largest distance (in terms of qubit indices) found among the multi-qubit gates in the first layer.
    """
    largest_distance = 1
    for layer in dag.layers():
        first_layer = layer
        break

    if "first_layer" in locals():
        from qiskit.converters import dag_to_circuit

        layer_circuit = dag_to_circuit(first_layer["graph"])
        for gate in layer_circuit.data:
            if gate.operation.num_qubits > 1:
                distance = abs(gate.qubits[0]._index - gate.qubits[-1]._index) + 1
                largest_distance = max(largest_distance, distance)

    return largest_distance


def select_starting_point(N: int, dag: DAGCircuit) -> tuple[range, range]:
    """Determine the starting set of neighboring qubits (even-even or odd-odd) for gate application.

    This function selects a checkerboard pattern for gate application based on the layout of gates in the first
    layer of the DAGCircuit. It returns two ranges of qubit indices that define the groups of neighboring qubits
    to be used as starting points.

    Args:
        N (int): Total number of qubits (or sites) in the system.
        dag (DAGCircuit): The DAGCircuit used to inspect the first set of gates.

    Returns:
        tuple[range, range]: A tuple containing two ranges:
            - The first range corresponds to the first group of qubits.
            - The second range corresponds to the complementary group.
    """
    assert N > 1

    for layer in dag.layers():
        first_layer = layer
        break

    first_iterator = range(0, N - 1, 2)
    second_iterator = range(1, N - 1, 2)
    odd = False

    if "first_layer" in locals():
        layer_circuit = dag_to_circuit(first_layer["graph"])
        for gate in layer_circuit.data:
            # If a two-qubit gate appears with an odd-indexed starting qubit, switch the ordering.
            if gate.operation.num_qubits == 2:
                if gate.qubits[0]._index % 2 != 0:
                    odd = True
                break

        if odd:
            first_iterator = range(1, N - 1, 2)
            second_iterator = range(0, N - 1, 2)

    return first_iterator, second_iterator
