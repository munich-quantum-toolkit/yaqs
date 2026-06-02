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

from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator

from .dag_utils import convert_dag_to_tensor_algorithm

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit

    from ...core.libraries.gate_library import BaseGate

_EINSUM_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_GATE_EINSUM_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _circuit_without_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``circuit`` with final measurements removed."""
    qc = circuit.copy()
    qc.remove_final_measurements(inplace=True)
    return qc


def _identity_operator_tensor(num_qubits: int) -> NDArray[np.complex128]:
    """Return the identity operator as a tensor with ``2 * num_qubits`` indices of size 2."""
    dim = 2**num_qubits
    return np.eye(dim, dtype=np.complex128).reshape((2,) * (2 * num_qubits))


def _embed_unitary(local: NDArray[np.complex128], sites: list[int], num_qubits: int) -> NDArray[np.complex128]:
    """Embed a k-qubit unitary on the given sites (Qiskit little-endian ordering)."""
    k = len(sites)
    if k == num_qubits and sites == list(range(num_qubits)):
        return local.copy()
    qc = QuantumCircuit(num_qubits)
    qc.append(UnitaryGate(local), sites)
    return Operator(qc).data


def _apply_single_qubit_left(
    op: NDArray[np.complex128],
    matrix: NDArray[np.complex128],
    qubit: int,
    num_qubits: int,
    *,
    dagger: bool = False,
) -> NDArray[np.complex128]:
    """Left-multiply ``op`` by a single-qubit gate on ``qubit``."""
    gate = matrix.conj().T if dagger else matrix
    if num_qubits > len(_EINSUM_LETTERS) // 2:
        op_mat = op.reshape(2**num_qubits, 2**num_qubits)
        full = _embed_unitary(gate, [qubit], num_qubits)
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


def _apply_two_qubit_left(
    op: NDArray[np.complex128],
    gate_tensor: NDArray[np.complex128],
    site0: int,
    site1: int,
    num_qubits: int,
    *,
    dagger: bool = False,
) -> NDArray[np.complex128]:
    """Left-multiply ``op`` by a two-qubit gate on ``(site0, site1)``."""
    if site0 > site1:
        return _apply_two_qubit_left(
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
        full = _embed_unitary(local, [site0, site1], num_qubits)
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


def _apply_gate_left(
    op: NDArray[np.complex128],
    gate: BaseGate,
    num_qubits: int,
    *,
    dagger: bool = False,
) -> NDArray[np.complex128]:
    """Left-multiply ``op`` by ``gate`` embedded on its sites."""
    if gate.interaction == 1:
        return _apply_single_qubit_left(op, gate.matrix, gate.sites[0], num_qubits, dagger=dagger)
    if gate.interaction == 2:
        return _apply_two_qubit_left(op, gate.tensor, gate.sites[0], gate.sites[1], num_qubits, dagger=dagger)

    local = gate.matrix
    if dagger:
        local = local.conj().T
    op_mat = op.reshape(2**num_qubits, 2**num_qubits)
    full = _embed_unitary(local, gate.sites, num_qubits)
    return (full @ op_mat).reshape((2,) * (2 * num_qubits))


def build_composed_operator_tensor(
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
    """
    num_qubits = circuit1.num_qubits
    op = _identity_operator_tensor(num_qubits)

    dag1 = circuit_to_dag(_circuit_without_measurements(circuit1))
    dag2 = circuit_to_dag(_circuit_without_measurements(circuit2))
    gates1 = convert_dag_to_tensor_algorithm(dag1)
    gates2 = convert_dag_to_tensor_algorithm(dag2)

    for gate in gates1:
        op = _apply_gate_left(op, gate, num_qubits, dagger=False)
    for gate in reversed(gates2):
        op = _apply_gate_left(op, gate, num_qubits, dagger=True)

    return op


def check_operator_is_identity(
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
    num_qubits = operator_tensor.ndim // 2
    hilbert_dim = 2**num_qubits
    dense = operator_tensor.reshape(hilbert_dim, hilbert_dim)
    identity = np.eye(hilbert_dim, dtype=np.complex128)
    trace = np.vdot(dense.ravel(), identity.ravel())
    return not np.round(np.abs(trace), 1) / hilbert_dim < fidelity


def check_equivalence_matrix(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    *,
    fidelity: float,
) -> bool:
    """Check circuit equivalence using the tensorized dense operator backend."""
    composed = build_composed_operator_tensor(circuit1, circuit2)
    return check_operator_is_identity(composed, fidelity)
