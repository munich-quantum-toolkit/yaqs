# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for DAG circuits.

This module implements conversion and processing functions for quantum circuits
using their DAG representations. It provides utilities to:

- Convert a ``DAGCircuit`` into a list of gate objects from the ``GateLibrary``.
- Extract a temporal zone from a ``DAGCircuit`` for specified qubits.
- Determine the maximum distance (in terms of qubit indices) of multi-qubit gates.
- Select starting points for gate application based on a checkerboard pattern.

Qiskit instructions whose ``op.name`` matches one of the 28 entries in
:data:`SUPPORTED_QISKIT_GATE_NAMES` are translated via hardcoded ``GateLibrary``
classes. All other one- and two-qubit unitary gates fall back to matrix-backed
:class:`~mqt.yaqs.core.libraries.gate_library.BaseGate` instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import Operation, Parameter, ParameterExpression
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator

from ...core.libraries.gate_library import BaseGate, GateLibrary

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import Qubit
    from qiskit.dagcircuit import DAGCircuit

# ``barrier`` is ignored during DAG-to-gate conversion. ``measure`` is rejected
# here because conversion builds a unitary gate list; mid-circuit measurements are
# not representable in that form. Circuit simulation (`process_layer` in
# ``digital_tjm``) still drops ``measure`` nodes from the live DAG before gate
# application so terminal measurements in a Qiskit circuit do not block simulation.
_SKIP_INSTRUCTIONS = frozenset({"barrier"})
_ZONE_SKIP_INSTRUCTIONS = frozenset({"measure", "barrier"})
_REJECTED_INSTRUCTIONS = frozenset({"reset", "delay", "store", "measure"})
# Qiskit ``op.name`` values resolved through ``getattr(GateLibrary, name)``.
SUPPORTED_QISKIT_GATE_NAMES: tuple[str, ...] = (
    "cp",
    "cx",
    "cz",
    "h",
    "i",
    "id",
    "iden",
    "p",
    "rx",
    "ry",
    "rz",
    "rxx",
    "ryy",
    "rzz",
    "s",
    "sdg",
    "swap",
    "sx",
    "sxdg",
    "t",
    "tdg",
    "u",
    "u1",
    "u2",
    "u3",
    "x",
    "y",
    "z",
)
_CONTROL_FLOW_INSTRUCTIONS = frozenset({
    "for_loop",
    "while_loop",
    "if_else",
    "switch_case",
    "break_loop",
    "continue_loop",
})


def _convert_matrix_layout(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Convert a Qiskit two-qubit unitary matrix into YAQS gate storage order.

    Args:
        matrix: Dense ``4 x 4`` unitary in Qiskit little-endian ordering.

    Returns:
        Matrix that reshapes to the YAQS two-qubit gate tensor convention.
    """
    tensor = np.reshape(matrix, (2, 2, 2, 2)).transpose(1, 0, 3, 2)
    return np.asarray(tensor.reshape(4, 4), dtype=np.complex128)


def _get_qubit_indices(dag: DAGCircuit | None, node: DAGOpNode) -> list[int]:
    """Return MPS site indices for a DAG operation node.

    Args:
        dag: Parent DAG when available; used for ``find_bit`` lookup.
        node: DAG operation node.

    Returns:
        Qubit indices in circuit order.
    """
    indices: list[int] = []
    for qubit in node.qargs:
        if dag is not None:
            indices.append(dag.find_bit(qubit).index)
        else:
            indices.append(qubit._index)  # noqa: SLF001
    return indices


def _has_unbound_params(op: object) -> bool:
    """Return whether a Qiskit operation still has free symbolic parameters.

    Args:
        op: Qiskit instruction or gate.

    Returns:
        True if any entry in ``op.params`` is an unbound ``Parameter`` or
        ``ParameterExpression`` with remaining free symbols.
    """
    params = getattr(op, "params", ())
    return any(
        isinstance(param, Parameter) or (isinstance(param, ParameterExpression) and bool(param.parameters))
        for param in params
    )


def _is_unitary(matrix: NDArray[np.complex128], *, atol: float = 1e-10) -> bool:
    """Return whether ``matrix`` is unitary within tolerance.

    Args:
        matrix: Square complex operator matrix.
        atol: Absolute tolerance for ``matrix @ matrix.conj().T ≈ I``.

    Returns:
        True if the matrix is unitary within ``atol``.
    """
    dim = matrix.shape[0]
    product = matrix @ matrix.conj().T
    return bool(np.allclose(product, np.eye(dim, dtype=np.complex128), atol=atol))


def _extract_matrix(op: Operation, *, name: str) -> NDArray[np.complex128]:
    """Extract a unitary matrix from a Qiskit operation.

    Args:
        op: Qiskit instruction or gate.
        name: Operation name for error reporting.

    Returns:
        Complex unitary matrix.

    Raises:
        ValueError: If no matrix representation is available.
    """
    to_matrix = getattr(op, "to_matrix", None)
    if callable(to_matrix):
        try:
            matrix = to_matrix()
        except (TypeError, QiskitError) as exc:
            try:
                return np.asarray(Operator(op).data, dtype=np.complex128)
            except (TypeError, QiskitError, ValueError):
                msg = f"Cannot translate Qiskit instruction '{name}': failed to build a matrix representation."
                raise ValueError(msg) from exc
        if matrix is not None:
            return np.asarray(matrix, dtype=np.complex128)

    try:
        return np.asarray(Operator(op).data, dtype=np.complex128)
    except (TypeError, QiskitError, ValueError) as exc:
        msg = f"Cannot translate Qiskit instruction '{name}': no matrix representation available."
        raise ValueError(msg) from exc


def _reject_unsupported(node: DAGOpNode) -> None:
    """Raise for Qiskit instructions that YAQS cannot translate.

    Args:
        node: DAG operation node.

    Raises:
        ValueError: If the instruction is unsupported.
    """
    name = node.op.name
    if name in _REJECTED_INSTRUCTIONS:
        if name == "measure":
            msg = (
                f"Cannot translate Qiskit instruction '{name}': mid-circuit measurements are not supported; "
                "remove measure instructions before conversion."
            )
        else:
            msg = f"Cannot translate Qiskit instruction '{name}': {name} is not supported in YAQS circuit simulation."
        raise ValueError(msg)
    if name in _CONTROL_FLOW_INSTRUCTIONS:
        msg = f"Cannot translate Qiskit instruction '{name}': control-flow operations are not supported."
        raise ValueError(msg)
    if getattr(node.op, "condition", None) is not None:
        msg = f"Cannot translate Qiskit instruction '{name}': conditional operations are not supported."
        raise ValueError(msg)
    if node.cargs:
        msg = f"Cannot translate Qiskit instruction '{name}': classically controlled operations are not supported."
        raise ValueError(msg)


def _translate_matrix(op: Operation, *, name: str, sites: list[int]) -> BaseGate:
    """Build a YAQS gate from a Qiskit operation matrix.

    Args:
        op: Qiskit instruction or gate.
        name: Qiskit operation name to store on the YAQS gate.
        sites: Target qubit indices.

    Returns:
        Matrix-backed YAQS gate without an analytic generator.

    Raises:
        ValueError: If parameters are unbound or the operator is not unitary.
    """
    if _has_unbound_params(op):
        msg = f"Cannot translate Qiskit gate '{name}': unbound parameters; bind parameters before simulation."
        raise ValueError(msg)

    matrix = _extract_matrix(op, name=name)
    if not _is_unitary(matrix):
        msg = f"Cannot translate Qiskit gate '{name}': operator is not unitary."
        raise ValueError(msg)

    if len(sites) == 2:
        matrix = _convert_matrix_layout(matrix)

    gate = GateLibrary.custom(matrix)
    gate.name = name
    gate.set_sites(*sites)
    return gate


def _translate_node(dag: DAGCircuit | None, node: DAGOpNode) -> BaseGate:
    """Convert a single DAG operation node into a YAQS gate.

    Args:
        dag: Parent DAG when available.
        node: DAG operation node.

    Returns:
        Initialized YAQS gate with sites assigned.

    Raises:
        ValueError: If the instruction cannot be translated.
    """
    name = node.op.name
    _reject_unsupported(node)

    if _has_unbound_params(node.op):
        msg = f"Cannot translate Qiskit gate '{name}': unbound parameters; bind parameters before simulation."
        raise ValueError(msg)

    sites = _get_qubit_indices(dag, node)
    num_qubits = len(sites)
    if num_qubits > 2:
        msg = f"Cannot translate Qiskit instruction '{name}': {num_qubits}-qubit gates are not supported yet."
        raise ValueError(msg)

    if name not in SUPPORTED_QISKIT_GATE_NAMES:
        return _translate_matrix(node.op, name=name, sites=sites)

    try:
        gate_cls = getattr(GateLibrary, name)
    except AttributeError:
        return _translate_matrix(node.op, name=name, sites=sites)

    gate_object = gate_cls(node.op.params) if node.op.params else gate_cls()
    gate_object.set_sites(*sites)
    return gate_object


def convert_dag_to_tensor_algorithm(dag: DAGCircuit) -> list[BaseGate]:
    """Convert a DAGCircuit into a list of gate objects from the GateLibrary.

    This function traverses the input DAGCircuit (or a single DAGOpNode) and creates a list of gate objects.
    For each node, it retrieves the corresponding gate class from the GateLibrary, initializes it, sets any
    parameters if present, and assigns the qubit indices (sites) on which the gate acts.

    Unknown one- and two-qubit unitary Qiskit gates are converted from their matrix representation.
    Three-qubit and larger instructions (including Toffoli/CCX) raise ``ValueError``. ``barrier`` nodes
    are skipped. ``measure``, ``reset``, ``delay``, classically controlled ops, and control-flow
    instructions are rejected. See :data:`SUPPORTED_QISKIT_GATE_NAMES` for hardcoded gate names.

    Note:
        This conversion path rejects ``measure`` because it builds a unitary gate list. Circuit
        simulation removes ``measure`` nodes from the live DAG separately (see ``process_layer`` in
        ``digital_tjm``) so terminal Qiskit measurements do not block evolution.

    Args:
        dag: The DAGCircuit (or a single DAGOpNode) representing a quantum operation.

    Returns:
        A list of gate objects, each with attributes such as .tensor and .sites.
    """
    algorithm: list[BaseGate] = []

    if isinstance(dag, DAGOpNode):
        algorithm.append(_translate_node(None, dag))
    else:
        parent_dag = dag
        for gate in dag.op_nodes():
            name = gate.op.name
            if name in _SKIP_INSTRUCTIONS:
                continue
            algorithm.append(_translate_node(parent_dag, gate))

    return algorithm


def extract_temporal_zone(dag: DAGCircuit, qubits: list[int]) -> DAGCircuit:
    """Extract the temporal zone without modifying ``dag``.

    Args:
        dag: The input DAGCircuit (unchanged).
        qubits: Qubit indices defining the strip ``[min(qubits), max(qubits)]``.

    Returns:
        A new DAGCircuit containing only operations in the temporal zone.
    """
    return _build_temporal_zone(dag.copy(), qubits)


def get_temporal_zone(dag: DAGCircuit, qubits: list[int]) -> DAGCircuit:
    """Extract the temporal zone from a DAGCircuit for the specified qubits.

    The temporal zone is defined as the subset of operations (layers) acting solely on the specified qubits,
    continuing until those qubits no longer participate in any further operations. The function builds a new
    DAGCircuit containing only these operations and removes consumed nodes from ``dag``.

    Args:
        dag (DAGCircuit): The input DAGCircuit (mutated in place).
        qubits (list[int]): List of qubit indices for which to extract the temporal zone.

    Returns:
        DAGCircuit: A new DAGCircuit containing only the operations within the temporal zone for the specified qubits.
    """
    return _build_temporal_zone(dag, qubits)


def _build_temporal_zone(dag: DAGCircuit, qubits: list[int]) -> DAGCircuit:
    """Build a temporal-zone DAG and remove consumed nodes from ``dag``.

    Returns:
        A new DAGCircuit containing only operations in the temporal zone.
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
                    if node.op.name in _ZONE_SKIP_INSTRUCTIONS:
                        dag.remove_op_node(node)
                        continue
                    new_dag.apply_operation_back(node.op, node.qargs)
                    dag.remove_op_node(node)
                else:
                    # For partial overlap, remove the overlapping qubits from the cone.
                    if node.op.name in _ZONE_SKIP_INSTRUCTIONS:
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
    first_layer = next(dag.layers(), None)

    if first_layer is not None:
        layer_circuit = dag_to_circuit(first_layer["graph"])
        for gate in layer_circuit.data:
            if gate.operation.num_qubits > 1:
                distance = abs(gate.qubits[0]._index - gate.qubits[-1]._index) + 1  # noqa: SLF001
                largest_distance = max(largest_distance, distance)

    return largest_distance


def select_starting_point(num_qubits: int, dag: DAGCircuit) -> tuple[range, range]:
    """Determine the starting set of neighboring qubits (even-even or odd-odd) for gate application.

    This function selects a checkerboard pattern for gate application based on the layout of gates in the first
    layer of the DAGCircuit. It returns two ranges of qubit indices that define the groups of neighboring qubits
    to be used as starting points.

    Args:
        num_qubits (int): Total number of qubits (or sites) in the system.
        dag (DAGCircuit): The DAGCircuit used to inspect the first set of gates.

    Returns:
        tuple[range, range]: A tuple containing two ranges:
            - The first range corresponds to the first group of qubits.
            - The second range corresponds to the complementary group.
    """
    assert num_qubits > 1

    first_layer = next(dag.layers(), None)

    first_iterator = range(0, num_qubits - 1, 2)
    second_iterator = range(1, num_qubits - 1, 2)
    odd = False

    if first_layer is not None:
        layer_circuit = dag_to_circuit(first_layer["graph"])
        for gate in layer_circuit.data:
            # If a two-qubit gate appears with an odd-indexed starting qubit, switch the ordering.
            if gate.operation.num_qubits == 2:
                if gate.qubits[0]._index % 2 != 0:  # noqa: SLF001
                    odd = True
                break

        if odd:
            first_iterator = range(1, num_qubits - 1, 2)
            second_iterator = range(0, num_qubits - 1, 2)

    return first_iterator, second_iterator
