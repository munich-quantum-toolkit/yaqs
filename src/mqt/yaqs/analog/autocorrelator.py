# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Autocorrelator helper routines for analog MPS simulations."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe

if TYPE_CHECKING:
    from ..core.data_structures.networks import MPS
    from ..core.data_structures.simulation_parameters import Observable


def _swap_gate_4() -> np.ndarray:
    """Two-qubit SWAP in the lexicographic two-qubit basis ``|ab⟩`` (``a`` left, ``b`` right).

    Returns:
        ``4x4`` complex SWAP matrix in the lexicographic two-qubit basis.
    """
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)


def _permuted_periodic_wrap_gate(gate4: np.ndarray) -> np.ndarray:
    """Express a gate on ``|q_{L-1}, q_0⟩`` in the merged NN basis ``|q_0, q_{L-1}⟩`` at sites ``(L-2, L-1)``.

    After forwarding SWAPs, the tensor pair ``(L-2, L-1)`` carries ``(q_0, q_{L-1})`` while ``gate4`` is defined
    for ``(q_{L-1}, q_0)``.

    Returns:
        Permuted ``4x4`` matrix acting on the merged NN basis at ``(L-2, L-1)``.
    """
    p_perm = np.zeros((4, 4), dtype=np.complex128)
    for a in range(2):
        for b in range(2):
            idx_merged = 2 * a + b
            idx_bond = 2 * b + a
            p_perm[idx_bond, idx_merged] = 1.0
    g = np.asarray(gate4, dtype=np.complex128)
    return p_perm.conj().T @ g @ p_perm


def _apply_two_site_nn_matrix_inplace(state: MPS, site_left: int, mat4: np.ndarray) -> None:
    """Apply a ``4 x 4`` gate to adjacent sites ``(site_left, site_left+1)`` in-place."""
    i, j = site_left, site_left + 1
    a = state.tensors[i]
    b = state.tensors[j]
    d_i, left, _ = a.shape
    d_j, _, right = b.shape

    theta = np.tensordot(a, b, axes=(2, 1))
    theta = theta.transpose(1, 0, 2, 3)
    theta = theta.reshape(left, d_i * d_j, right)
    theta = oe.contract("ab, cbd->cad", mat4, theta)
    theta = theta.reshape(left, d_i, d_j, right)

    theta_mat = theta.reshape(left * d_i, d_j * right)
    u_mat, s_vec, v_mat = np.linalg.svd(theta_mat, full_matrices=False)
    chi_new = len(s_vec)

    u_tensor = u_mat.reshape(left, d_i, chi_new)
    a_new = u_tensor.transpose(1, 0, 2)

    v_tensor = (np.diag(s_vec) @ v_mat).reshape(chi_new, d_j, right)
    b_new = v_tensor.transpose(1, 0, 2)

    state.tensors[i] = a_new
    state.tensors[j] = b_new


def _bubble_swaps_forward_for_wrap(state: MPS) -> None:
    """Apply ``L-2`` adjacent SWAPs so logical ``q_0`` meets ``q_{L-1}`` on sites ``(L-2, L-1)``."""
    length = state.length
    if length <= 2:
        return
    sw = _swap_gate_4()
    for i in range(length - 2):
        _apply_two_site_nn_matrix_inplace(state, i, sw)


def _bubble_swaps_backward_for_wrap(state: MPS) -> None:
    """Undo :func:`_bubble_swaps_forward_for_wrap`."""
    length = state.length
    if length <= 2:
        return
    sw = _swap_gate_4()
    for i in reversed(range(length - 2)):
        _apply_two_site_nn_matrix_inplace(state, i, sw)


def apply_observable_inplace(state: MPS, observable: Observable) -> None:
    r"""Apply a one- or two-site observable to an MPS in-place.

    The implementation mirrors the operator-application part of :meth:`MPS.local_expect`
    but preserves the transformed ``state`` tensor train (for example, to build
    :math:`O|\psi\rangle` for mixed contractions).

    Two-site observables on the periodic wrap ``(L-1, 0)`` (``sites=[L-1, 0]``) are supported for
    ``L > 2`` using a reversible adjacent-SWAP circuit. For ``L == 2``, the same wrap convention
    applies when ``sites=[L-1, 0]``: the ``4 x 4`` matrix is taken in the ``|q_{L-1}, q_0⟩`` basis
    and mapped to the merged nearest-neighbor basis on ``(0, 1)`` via :func:`_permuted_periodic_wrap_gate`
    (no SWAP network is needed). Use ``sites=[0, 1]`` for a standard nearest-neighbor gate in the
    ``|q_0, q_1⟩`` basis.

    Args:
        state (MPS): The MPS to modify in-place.
        observable (Observable): One-site or nearest-neighbor two-site observable specification.

    Raises:
        ValueError: If the gate dimension is not ``2`` or ``4``, if a two-site operator is not
            on adjacent sites (except the supported periodic wrap), or if the observable type is
            unsupported.
    """
    sites = [observable.sites] if isinstance(observable.sites, int) else observable.sites

    if observable.gate.matrix.shape[0] == 2:
        site = sites[0]
        state.tensors[site] = oe.contract("ab, bcd->acd", observable.gate.matrix, state.tensors[site])
        return

    if observable.gate.matrix.shape[0] == 4:
        i, j = int(sites[0]), int(sites[1])
        length = state.length

        if length == 2:
            if i == length - 1 and j == 0:
                mat = np.asarray(observable.gate.matrix, dtype=np.complex128)
                g_merged = _permuted_periodic_wrap_gate(mat)
                _apply_two_site_nn_matrix_inplace(state, 0, g_merged)
                return
            i, j = min(i, j), max(i, j)
        elif (i == length - 1 and j == 0) or (i == 0 and j == length - 1):
            mat = np.asarray(observable.gate.matrix, dtype=np.complex128)
            _bubble_swaps_forward_for_wrap(state)
            g_merged = _permuted_periodic_wrap_gate(mat)
            _apply_two_site_nn_matrix_inplace(state, length - 2, g_merged)
            _bubble_swaps_backward_for_wrap(state)
            return

        if j != i + 1:
            msg = "Only nearest-neighbor two-site observables are currently implemented."
            raise ValueError(msg)

        _apply_two_site_nn_matrix_inplace(state, i, np.asarray(observable.gate.matrix, dtype=np.complex128))
        return

    msg = "Autocorrelator observable must be one-site or nearest-neighbor two-site."
    raise ValueError(msg)


def mixed_expectation(
    bra: MPS,
    ket: MPS,
    observable: Observable,
) -> np.complex128:
    r"""Compute the mixed matrix element :math:`\langle\mathrm{bra}|O|\mathrm{ket}\rangle`.

    This applies ``observable`` to a deep copy of ``ket`` and contracts with ``bra`` using
    :meth:`MPS.scalar_product`.

    Args:
        bra (MPS): Bra MPS (left vector).
        ket (MPS): Ket MPS; a copy is transformed by ``observable``.
        observable (Observable): Same observable conventions as :func:`apply_observable_inplace`.

    Returns:
        np.complex128: The scalar contraction :math:`\langle\mathrm{bra}|O|\mathrm{ket}\rangle`.
    """
    ket_with_op = copy.deepcopy(ket)
    apply_observable_inplace(ket_with_op, observable)
    return bra.scalar_product(ket_with_op)
