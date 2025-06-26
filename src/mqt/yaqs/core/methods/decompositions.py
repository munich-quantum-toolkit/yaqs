# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tensor Network Decompositions.

This module implements left and right moving versions of the QR and SVD decompositions which are used throughout YAQS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def right_qr(mps_tensor: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Right QR.

    Performs the QR decomposition of an MPS tensor moving to the right.

    Args:
        mps_tensor: The tensor to be decomposed.

    Returns:
        q_tensor: The Q tensor with the left virtual leg and the physical
            leg (phys,left,new).
        r_mat: The R matrix with the right virtual leg (new,right).
    """
    old_shape = mps_tensor.shape
    qr_shape = (old_shape[0] * old_shape[1], old_shape[2])
    mps_tensor = mps_tensor.reshape(qr_shape)
    q_mat, r_mat = np.linalg.qr(mps_tensor)
    new_shape = (old_shape[0], old_shape[1], -1)
    q_tensor = q_mat.reshape(new_shape)
    return q_tensor, r_mat


def left_qr(mps_tensor: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Left QR.

    Performs the QR decomposition of an MPS tensor moving to the left.

    Args:
        mps_tensor: The tensor to be decomposed.

    Returns:
        q_tensor: The Q tensor with the physical leg and the right virtual
            leg (phys,new,right).
        r_mat: The R matrix with the left virtual leg (left,new).

    """
    old_shape = mps_tensor.shape
    mps_tensor = mps_tensor.transpose(0, 2, 1)
    qr_shape = (old_shape[0] * old_shape[2], old_shape[1])
    mps_tensor = mps_tensor.reshape(qr_shape)
    q_mat, r_mat = np.linalg.qr(mps_tensor)
    q_tensor = q_mat.reshape((old_shape[0], old_shape[2], -1))
    q_tensor = q_tensor.transpose(0, 2, 1)
    r_mat = r_mat.T
    return q_tensor, r_mat


def right_svd(
    mps_tensor: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Right SVD.

    Performs the singular value decomposition of an MPS tensor.

    Args:
        mps_tensor: The tensor to be decomposed.

    Returns:
        NDArray[np.complex128]: The U tensor with the left virtual leg and the physical
            leg (phys,left,new).
        NDArray[np.complex128]: The S vector with the singular values.
        NDArray[np.complex128]: The V matrix with the right virtual leg (new,right).

    """
    old_shape = mps_tensor.shape
    svd_shape = (old_shape[0] * old_shape[1], old_shape[2])
    mps_tensor = mps_tensor.reshape(svd_shape)
    u_mat, s_vec, v_mat = np.linalg.svd(mps_tensor, full_matrices=False)
    new_shape = (old_shape[0], old_shape[1], -1)
    u_tensor = u_mat.reshape(new_shape)
    return u_tensor, s_vec, v_mat


def truncated_right_svd(
    mps_tensor: NDArray[np.complex128],
    threshold: float,
    max_bond_dim: int | None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Truncated right SVD.

    Performs the truncated singular value decomposition of an MPS tensor.

    Args:
        mps_tensor: The tensor to be decomposed.
        threshold: SVD threshold
        max_bond_dim: Maximum bond dimension of MPS

    Returns:
        a_new: The U tensor with the left virtual leg and the physical
            leg (phys,left,new).
        b_new: The V matrix with the right virtual leg (new,right).

    """
    u_tensor, s_vec, v_mat = right_svd(mps_tensor)
    cut_sum = 0
    cut_index = 1
    for i, s_val in enumerate(np.flip(s_vec)):
        cut_sum += s_val**2
        if cut_sum >= threshold:
            cut_index = len(s_vec) - i
            break
    if max_bond_dim is not None:
        cut_index = min(cut_index, max_bond_dim)
    u_tensor = u_tensor[:, :, :cut_index]
    s_vec = s_vec[:cut_index]
    v_mat = v_mat[:cut_index, :]
    return u_tensor, s_vec, v_mat


def two_site_svd(
    a: NDArray[np.complex128],
    b: NDArray[np.complex128],
    threshold: float,
    max_bond_dim: int | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Two site SVD.

    Performs the truncated singular value decomposition of two MPS tensors.

    Args:
        a: The left tensor to be decomposed.
        b: The right tensor to be decomposed.
        threshold: SVD threshold
        max_bond_dim: Maximum bond dimension of MPS

    Returns:
        a_new: The U tensor with the left virtual leg and the physical
            leg (phys,left,new).
        b_new: The V matrix with the right virtual leg (new,right).

    """
    # 1) build the two-site tensor theta_{(phys_i,L),(phys_j,R)}
    theta = np.tensordot(a, b, axes=(2, 1))
    phys_i, left = a.shape[0], a.shape[1]
    phys_j, right = b.shape[0], b.shape[2]

    # 2) reshape to matrix M of shape (L*phys_i) x (phys_j*R)
    theta_mat = theta.reshape(left * phys_i, phys_j * right)

    # 3) full SVD
    u_mat, s_vec, v_mat = np.linalg.svd(theta_mat, full_matrices=False)

    # 4) decide how many singular values to keep:
    #    sum of squares of discarded values ≤ threshold
    discard = 0.0
    keep = len(s_vec)
    min_keep = 2  # Prevents pathological dimension-1 truncation
    for idx, s in enumerate(reversed(s_vec)):
        discard += s**2
        if discard >= threshold:
            keep = max(len(s_vec) - idx, min_keep)
            break
    if max_bond_dim is not None:
        keep = min(keep, max_bond_dim)

    # 5) build the truncated A' of shape (phys_i, L, keep)
    a_new = u_mat[:, :keep].reshape(phys_i, left, keep)

    # 6) absorb S into Vh and reshape to B' of shape (phys_j, keep, R)
    v_tensor = np.diag(s_vec[:keep]) @ v_mat[:keep, :]  # shape (keep, phys_j*R)
    v_tensor = v_tensor.reshape(keep, phys_j, right)  # (keep, phys_j, R)
    b_new = v_tensor.transpose(1, 0, 2)  # (phys_j, keep, R)

    return a_new, b_new
