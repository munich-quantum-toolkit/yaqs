# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Low-level MPO/MPS tensor contractions and gate-MPO construction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from numpy.typing import NDArray

from .. import linalg
from ..libraries.gate_library import extend_gate

if TYPE_CHECKING:
    from ..libraries.gate_library import BaseGate

ComplexTensor = NDArray[np.complex128]


def contract_mpo_site_with_mps_site(
    mpo_tensor: NDArray[np.complex128],
    mps_tensor: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Contract one MPO site with one MPS site using library leg ordering.

    MPO layout: ``(phys_out, phys_in, chi_left, chi_right)``.
    MPS layout: ``(phys, chi_left, chi_right)``.

    Virtual bonds are fused with MPS indices first and MPO indices second on both sides.

    Args:
        mpo_tensor: Gate or operator MPO tensor at one site.
        mps_tensor: MPS site tensor.

    Returns:
        Updated MPS site tensor ``(phys_out, chi_left * mpo_left, chi_right * mpo_right)``.
    """
    operator = np.asarray(mpo_tensor, dtype=np.complex128)
    site = np.asarray(mps_tensor, dtype=np.complex128)
    theta = np.tensordot(operator, site, axes=([1], [0]))
    phys_out, mpo_left, mpo_right, mps_left, mps_right = theta.shape
    return np.asarray(
        theta.transpose(0, 3, 1, 4, 2).reshape(
            phys_out,
            mps_left * mpo_left,
            mps_right * mpo_right,
        ),
        dtype=np.complex128,
    )


def contract_mpo_site_with_mpo_site(
    left_mpo_tensor: NDArray[np.complex128],
    right_mpo_tensor: NDArray[np.complex128],
    *,
    conjugate: bool = False,
) -> NDArray[np.complex128]:
    """Contract two MPO site tensors (left factor times right factor).

    Uses the transposed layout ``(phys_out, chi_left, phys_in, chi_right)`` and the
    ``abcd,cefg`` contraction from equivalence-checking MPO updates.

    Args:
        left_mpo_tensor: Left MPO factor at one site, library order
            ``(phys_out, phys_in, chi_left, chi_right)``.
        right_mpo_tensor: Right MPO factor at the same site.
        conjugate: If True, use the conjugated contraction used when updating
            the right-hand MPO in equivalence checking.

    Returns:
        Product tensor in library order ``(phys_out, phys_in, chi_left, chi_right)``.
    """
    tensor1 = np.transpose(np.asarray(left_mpo_tensor, dtype=np.complex128), (0, 2, 1, 3))
    tensor2 = np.transpose(np.asarray(right_mpo_tensor, dtype=np.complex128), (0, 2, 1, 3))
    if conjugate:
        theta = oe.contract("abcd,cefg->febagd", tensor1, tensor2)
    else:
        theta = oe.contract("abcd,cefg->abefdg", tensor1, tensor2)
    dims = theta.shape
    fused = np.reshape(theta, (dims[0], dims[1] * dims[2], dims[3], dims[4] * dims[5]))
    return np.transpose(fused, (0, 2, 1, 3))


def make_identity_site(physical_dimension: int) -> NDArray[np.complex128]:
    """Single-site identity MPO tensor ``(phys_out, phys_in, 1, 1)``.

    Args:
        physical_dimension: Local Hilbert-space dimension per site.

    Returns:
        Identity MPO site tensor.
    """
    tensor = np.eye(physical_dimension, dtype=np.complex128)
    return np.expand_dims(np.expand_dims(tensor, axis=2), axis=3)


def convert_nn_matrix(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Map a ``4 x 4`` two-qubit unitary into ``U[out_l, out_r, in_l, in_r]`` for TEBD.

    The MPS merge uses ``ijkl,klab->ijab`` with ``k,l`` the left/right input physical
    indices and ``i,j`` the outputs. Qiskit little-endian state indices use
    ``index = q_left + 2 * q_right`` for the two-site support ``(left, right)``.

    Args:
        matrix: Unitary on the two-qubit computational basis in that index ordering.

    Returns:
        Rank-4 gate tensor for nearest-neighbor TEBD contraction.
    """
    mat = np.asarray(matrix, dtype=np.complex128)
    tensor = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    for col in range(4):
        in_left, in_right = col % 2, col // 2
        for row in range(4):
            out_left, out_right = row % 2, row // 2
            tensor[out_left, out_right, in_left, in_right] = mat[row, col]
    return tensor


def resolve_lr_tensor(
    gate: BaseGate,
    left_site: int | None = None,
    right_site: int | None = None,
) -> NDArray[np.complex128]:
    """Return ``gate.tensor`` as ``U[out_l, out_r, in_l, in_r]`` on ascending MPS sites.

    When ``gate.sites == (left_site, right_site)``, the gate matrix is reshaped directly.
    When the declared sites are reversed on the same nearest-neighbor pair, the matrix
    already encodes the operator for ``gate.sites`` and is mapped with
    :func:`convert_nn_matrix` instead of transposing a naive reshape.

    Args:
        gate: Two-qubit gate with ``sites`` and ``tensor`` set.
        left_site: Lower MPS site index; defaults to ``min(gate.sites)``.
        right_site: Higher MPS site index; defaults to ``max(gate.sites)``.

    Returns:
        Gate tensor with left-then-right site axis order.

    Raises:
        ValueError: If ``gate.sites`` does not match ``(left_site, right_site)``.
    """
    if left_site is None or right_site is None:
        site0, site1 = gate.sites[0], gate.sites[1]
        left_site = min(site0, site1)
        right_site = max(site0, site1)
    if gate.sites[0] == left_site and gate.sites[1] == right_site:
        return np.asarray(gate.tensor, dtype=np.complex128)
    if gate.sites[0] == right_site and gate.sites[1] == left_site:
        return convert_nn_matrix(gate.matrix)
    msg = f"Gate sites {gate.sites!r} are not consistent with MPS sites ({left_site}, {right_site})."
    raise ValueError(msg)


def get_support_mpo(
    gate: BaseGate,
    *,
    first_site: int,
    last_site: int,
) -> list[NDArray[np.complex128]]:
    """MPO tensors for the gate support ``[first_site, last_site]`` in library order.

    Args:
        gate: Two-qubit gate with optional cached ``mpo_tensors``.
        first_site: First site of the support interval (inclusive).
        last_site: Last site of the support interval (inclusive).

    Returns:
        Support MPO tensors from the gate cache or :func:`~mqt.yaqs.core.libraries.gate_library.extend_gate`.
    """
    support_len = last_site - first_site + 1
    try:
        cached = gate.mpo_tensors
    except AttributeError:
        cached = None
    if cached is not None and len(cached) == support_len:
        return list(cached)
    return extend_gate(
        resolve_lr_tensor(gate),
        [first_site, last_site],
    )


def decompose_theta(
    theta: NDArray[np.complex128], threshold: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """SVD-split a fused two-site MPO tensor back into two rank-4 site tensors.

    Args:
        theta: High-rank tensor from contracting two MPO sites (and optional gates).
        threshold: Singular-value cutoff for truncation.

    Returns:
        Left and right MPO site tensors after truncated SVD.
    """
    dims = theta.shape
    theta = np.transpose(theta, (0, 3, 2, 1, 4, 5))
    theta_matrix = np.reshape(theta, (dims[0] * dims[1] * dims[2], dims[3] * dims[4] * dims[5]))

    u_mat, s_list, v_mat = linalg.svd(theta_matrix, full_matrices=False)
    keep = linalg.truncate(s_list, mode="hard_cutoff", threshold=threshold, min_keep=1)
    s_list = s_list[:keep]
    u_mat = u_mat[:, :keep]
    v_mat = v_mat[:keep, :]

    u_tensor = np.reshape(u_mat, (dims[0], dims[1], dims[2], len(s_list)))

    m_mat = np.diag(s_list) @ v_mat
    m_tensor = np.reshape(m_mat, (len(s_list), dims[3], dims[4], dims[5]))
    m_tensor = np.transpose(m_tensor, (1, 2, 0, 3))

    return u_tensor, m_tensor
