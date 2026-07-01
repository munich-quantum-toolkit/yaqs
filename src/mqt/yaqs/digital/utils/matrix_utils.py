# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dense tensorized utilities for matrix-based equivalence checking.

Builds the composed operator ``W = U2† U1`` as a tensor with shape ``(2,) * (2 * n_qubits)``
and applies local gate contractions. The final operator is checked for closeness to the
identity using the same trace-based criterion as the MPO backend.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.quantum_info import Operator

from .dag_utils import convert_dag_to_tensor_algorithm
from .scheduler_utils import partition_disjoint_gate_batches

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.dagcircuit import DAGCircuit

    from ...core.libraries.gate_library import BaseGate

_EINSUM_LETTERS = string.ascii_lowercase
_GATE_EINSUM_LETTERS = string.ascii_uppercase


def strip_final_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``circuit`` with final measurements removed.

    Args:
        circuit: Input circuit (may include terminal measurements).

    Returns:
        A copy with final measurements stripped.

    Raises:
        ValueError: If mid-circuit measurements remain after removing final measurements.
    """
    qc = circuit.copy()
    qc.remove_final_measurements(inplace=True)
    dag = circuit_to_dag(qc)
    if any(isinstance(node, DAGOpNode) and node.op.name == "measure" for node in dag.op_nodes()):
        msg = "Mid-circuit measurements are not supported by the equivalence checker."
        raise ValueError(msg)
    return qc


def make_identity_tensor(num_qubits: int) -> NDArray[np.complex128]:
    """Return the identity operator as a tensor with ``2 * num_qubits`` indices of size 2.

    Args:
        num_qubits: Number of qubits.

    Returns:
        Identity operator tensor of shape ``(2,) * (2 * num_qubits)``.
    """
    dim = 2**num_qubits
    return np.eye(dim, dtype=np.complex128).reshape((2,) * (2 * num_qubits))


def embed_unitary(local: NDArray[np.complex128], sites: list[int], num_qubits: int) -> NDArray[np.complex128]:
    """Embed a k-qubit unitary on the given sites (Qiskit little-endian ordering).

    Args:
        local: ``2**k`` by ``2**k`` unitary on ``k = len(sites)`` qubits.
        sites: Target qubit indices.
        num_qubits: Total number of qubits in the circuit.

    Returns:
        The ``2**num_qubits``-dimensional unitary matrix with ``local`` on ``sites``.
    """
    k = len(sites)
    if k == num_qubits and sites == list(range(num_qubits)):
        return local.copy()
    qc = QuantumCircuit(num_qubits)
    qc.append(UnitaryGate(local), sites)
    return Operator(qc).data


def apply_1q_left(
    op: NDArray[np.complex128],
    matrix: NDArray[np.complex128],
    qubit: int,
    num_qubits: int,
    *,
    dagger: bool = False,
) -> NDArray[np.complex128]:
    """Left-multiply ``op`` by a single-qubit gate on ``qubit``.

    Args:
        op: Operator tensor with ``2 * num_qubits`` indices of dimension 2.
        matrix: Single-qubit gate matrix.
        qubit: Target qubit index.
        num_qubits: Total number of qubits.
        dagger: If True, apply the conjugate transpose of ``matrix``.

    Returns:
        The updated operator tensor.
    """
    gate = matrix.conj().T if dagger else matrix
    if num_qubits > len(_EINSUM_LETTERS) // 2:
        op_mat = op.reshape(2**num_qubits, 2**num_qubits)
        full = embed_unitary(gate, [qubit], num_qubits)
        return (full @ op_mat).reshape((2,) * (2 * num_qubits))

    out_labels = list(_EINSUM_LETTERS[:num_qubits])
    in_labels = list(_EINSUM_LETTERS[num_qubits : 2 * num_qubits])
    gate_out, gate_prev_out = _GATE_EINSUM_LETTERS[0], _GATE_EINSUM_LETTERS[1]
    out_labels[qubit] = gate_prev_out
    eq_op = "".join(out_labels + in_labels)
    out_labels[qubit] = gate_out
    eq_result = "".join(out_labels + in_labels)
    eq = f"{gate_out}{gate_prev_out},{eq_op}->{eq_result}"
    return np.einsum(eq, gate, op, optimize=True)


def apply_2q_left(
    op: NDArray[np.complex128],
    gate_tensor: NDArray[np.complex128],
    site0: int,
    site1: int,
    num_qubits: int,
    *,
    dagger: bool = False,
) -> NDArray[np.complex128]:
    """Left-multiply ``op`` by a two-qubit gate on ``(site0, site1)``.

    Args:
        op: Operator tensor with ``2 * num_qubits`` indices of dimension 2.
        gate_tensor: Two-qubit gate in ``(out0, out1, in0, in1)`` layout.
        site0: First qubit index (need not be sorted).
        site1: Second qubit index.
        num_qubits: Total number of qubits.
        dagger: If True, apply the conjugate transpose of the gate.

    Returns:
        The updated operator tensor.
    """
    if site0 > site1:
        return apply_2q_left(
            op,
            np.transpose(gate_tensor, (1, 0, 3, 2)),
            site1,
            site0,
            num_qubits,
            dagger=dagger,
        )

    gate = gate_tensor
    if dagger:
        gate = np.conjugate(np.transpose(gate, (2, 3, 0, 1)))

    if num_qubits > len(_EINSUM_LETTERS) // 2:
        op_mat = op.reshape(2**num_qubits, 2**num_qubits)
        local = np.transpose(gate, (0, 2, 1, 3)).reshape(4, 4)
        full = embed_unitary(local, [site0, site1], num_qubits)
        return (full @ op_mat).reshape((2,) * (2 * num_qubits))

    out_labels = list(_EINSUM_LETTERS[:num_qubits])
    in_labels = list(_EINSUM_LETTERS[num_qubits : 2 * num_qubits])
    g_out0, g_out1, g_prev0, g_prev1 = _GATE_EINSUM_LETTERS[:4]
    out_labels[site0] = g_prev0
    out_labels[site1] = g_prev1
    eq_op = "".join(out_labels + in_labels)
    out_labels[site0] = g_out0
    out_labels[site1] = g_out1
    eq_result = "".join(out_labels + in_labels)
    eq = f"{g_out0}{g_out1}{g_prev0}{g_prev1},{eq_op}->{eq_result}"
    return np.einsum(eq, gate, op, optimize=True)


def apply_gate_left(
    op: NDArray[np.complex128],
    gate: BaseGate,
    num_qubits: int,
    *,
    dagger: bool = False,
) -> NDArray[np.complex128]:
    """Left-multiply ``op`` by ``gate`` embedded on its sites.

    Args:
        op: Operator tensor with ``2 * num_qubits`` indices of dimension 2.
        gate: Gate object with ``sites`` and ``matrix``/``tensor`` data.
        num_qubits: Total number of qubits.
        dagger: If True, apply each gate as its conjugate transpose.

    Returns:
        The updated operator tensor.
    """
    if gate.interaction == 1:
        return apply_1q_left(op, gate.matrix, gate.sites[0], num_qubits, dagger=dagger)
    if gate.interaction == 2:
        return apply_2q_left(op, gate.tensor, gate.sites[0], gate.sites[1], num_qubits, dagger=dagger)

    local = gate.matrix
    if dagger:
        local = local.conj().T
    op_mat = op.reshape(2**num_qubits, 2**num_qubits)
    full = embed_unitary(local, gate.sites, num_qubits)
    return (full @ op_mat).reshape((2,) * (2 * num_qubits))


def pop_front_gates(dag: DAGCircuit) -> list[BaseGate]:
    """Remove and return gates from the current DAG front layer.

    Args:
        dag: Circuit DAG being consumed layer by layer.

    Returns:
        Gate objects from the front layer, in extraction order.
    """
    gates: list[BaseGate] = []
    for node in list(dag.front_layer()):
        if not isinstance(node, DAGOpNode):
            continue
        if node.op.name in {"measure", "barrier"}:
            dag.remove_op_node(node)
            continue
        gates.extend(convert_dag_to_tensor_algorithm(node))
        dag.remove_op_node(node)
    return gates


def apply_gate_batch(
    op: NDArray[np.complex128],
    gates: list[BaseGate],
    num_qubits: int,
    *,
    dagger: bool,
) -> NDArray[np.complex128]:
    """Apply a commuting batch of gates in deterministic site order.

    Args:
        op: Operator tensor to update.
        gates: Gates with pairwise disjoint supports.
        num_qubits: Total number of qubits.
        dagger: If True, apply gates as conjugate-transpose.

    Returns:
        The operator tensor after applying all gates in the batch.
    """
    for gate in sorted(gates, key=lambda g: min(g.sites)):
        op = apply_gate_left(op, gate, num_qubits, dagger=dagger)
    return op


def apply_layer(
    op: NDArray[np.complex128],
    layer_gates: list[BaseGate],
    num_qubits: int,
    *,
    dagger: bool,
) -> NDArray[np.complex128]:
    """Apply one circuit layer as sequential disjoint gate batches.

    Args:
        op: Operator tensor to update.
        layer_gates: Gates in one DAG front layer.
        num_qubits: Total number of qubits.
        dagger: If True, apply gates as conjugate-transpose.

    Returns:
        The operator tensor after applying the full layer.
    """
    for batch in partition_disjoint_gate_batches(layer_gates):
        op = apply_gate_batch(op, batch, num_qubits, dagger=dagger)
    return op


def collect_layers(dag: DAGCircuit) -> list[list[BaseGate]]:
    """Consume ``dag`` layer by layer and return gate lists per layer.

    Args:
        dag: Circuit DAG to consume.

    Returns:
        One list of gates per consumed front layer.
    """
    layers: list[list[BaseGate]] = []
    while dag.op_nodes():
        layer_gates = pop_front_gates(dag)
        if layer_gates:
            layers.append(layer_gates)
    return layers


def compose_operator_tensor(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
) -> NDArray[np.complex128]:
    """Build ``W = U2† U1`` as a tensor with ``2 * num_qubits`` indices of dimension 2.

    Gates from ``circuit1`` are applied in DAG order. Gates from ``circuit2`` are applied in
    reverse DAG order with conjugation, implementing multiplication by ``U2†`` on the left.

    Args:
        circuit1: First circuit.
        circuit2: Second circuit (same number of qubits as ``circuit1``).

    Returns:
        Tensor of shape ``(2,) * (2 * num_qubits)`` representing the composed operator.

    Raises:
        ValueError: If ``circuit1`` and ``circuit2`` have different numbers of qubits.
    """
    if circuit1.num_qubits != circuit2.num_qubits:
        msg = "Circuits must have the same number of qubits."
        raise ValueError(msg)
    num_qubits = circuit1.num_qubits
    op = make_identity_tensor(num_qubits)

    dag1 = circuit_to_dag(strip_final_measurements(circuit1))
    dag2 = circuit_to_dag(strip_final_measurements(circuit2))

    for layer_gates in collect_layers(dag1):
        op = apply_layer(op, layer_gates, num_qubits, dagger=False)

    layers2 = collect_layers(dag2)
    for layer_gates in reversed(layers2):
        op = apply_layer(op, layer_gates, num_qubits, dagger=True)

    return op


def compute_identity_fidelity(operator_tensor: NDArray[np.complex128]) -> float:
    """Return normalized trace overlap of an operator tensor with the identity.

    Args:
        operator_tensor: Operator as a tensor with ``2 * n`` indices of size 2.

    Returns:
        ``|Tr(O)| / d`` where ``d`` is the Hilbert-space dimension.
    """
    num_qubits = operator_tensor.ndim // 2
    hilbert_dim = 2**num_qubits
    dense = operator_tensor.reshape(hilbert_dim, hilbert_dim)
    identity = np.eye(hilbert_dim, dtype=np.complex128)
    trace = np.vdot(dense.ravel(), identity.ravel())
    return float(np.abs(trace) / hilbert_dim)


def is_identity_tensor(
    operator_tensor: NDArray[np.complex128],
    fidelity: float,
) -> bool:
    """Return whether ``operator_tensor`` is identity-like within ``fidelity``.

    Uses the same trace-overlap criterion as :meth:`~mqt.yaqs.core.data_structures.mpo.MPO.check_if_identity`.

    Args:
        operator_tensor: Operator as a tensor with ``2 * n`` indices of size 2.
        fidelity: Minimum normalized overlap with the identity to count as equivalent.

    Returns:
        True if the operator is equivalent to the identity up to global phase within ``fidelity``.
    """
    return compute_identity_fidelity(operator_tensor) >= fidelity


def check_matrix_equivalence(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    *,
    fidelity: float,
) -> bool:
    """Check circuit equivalence using the tensorized dense operator backend.

    Args:
        circuit1: First circuit.
        circuit2: Second circuit.
        fidelity: Minimum overlap with the identity to count as equivalent.

    Returns:
        Whether the circuits are equivalent within ``fidelity``.
    """
    composed = compose_operator_tensor(circuit1, circuit2)
    return is_identity_tensor(composed, fidelity)
