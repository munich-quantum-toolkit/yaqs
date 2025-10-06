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
    mps_mat = mps_tensor.reshape(svd_shape)
    
    # Check for numerical issues and apply fallback if needed
    if not np.isfinite(mps_mat).all():
        mps_mat = np.nan_to_num(mps_mat, nan=0.0, posinf=1e10, neginf=-1e10)
    
    try:
        u_mat, s_vec, v_mat = np.linalg.svd(mps_mat, full_matrices=False)
    except np.linalg.LinAlgError:
        # SVD didn't converge - try fallback strategies
        try:
            # Strategy 1: Add small regularization to diagonal
            min_dim = min(mps_mat.shape[0], mps_mat.shape[1])
            mps_reg = mps_mat.copy()
            for i in range(min_dim):
                mps_reg[i, i] += 1e-14
            u_mat, s_vec, v_mat = np.linalg.svd(mps_reg, full_matrices=False)
        except np.linalg.LinAlgError:
            try:
                # Strategy 2: Use eigenvalue decomposition
                ata = mps_mat.conj().T @ mps_mat
                eigenvals, v_mat = np.linalg.eigh(ata)
                idx = np.argsort(eigenvals)[::-1]
                s_vec = np.sqrt(np.abs(eigenvals[idx]))
                v_mat = v_mat[:, idx].conj().T
                s_inv = np.where(s_vec > 1e-15, 1.0 / s_vec, 0.0)
                u_mat = mps_mat @ v_mat.conj().T @ np.diag(s_inv)
            except np.linalg.LinAlgError:
                # Strategy 3: QR-based fallback
                m, n = mps_mat.shape
                if m <= n:
                    Q, R = np.linalg.qr(mps_mat)
                    U_r, s_vec, Vh = np.linalg.svd(R, full_matrices=False)
                    u_mat = Q @ U_r
                    v_mat = Vh
                else:
                    Qh, Rh = np.linalg.qr(mps_mat.conj().T)
                    U_r, s_vec, Vh = np.linalg.svd(Rh.conj().T, full_matrices=False)
                    u_mat = U_r
                    v_mat = Vh @ Qh.conj().T
    
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

    # 3) Check for numerical issues before SVD
    if not np.isfinite(theta_mat).all():
        # If we have NaN or Inf, try to recover by sanitizing
        theta_mat = np.nan_to_num(theta_mat, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # 4) full SVD with fallback strategies
    try:
        u_mat, s_vec, v_mat = np.linalg.svd(theta_mat, full_matrices=False)
    except np.linalg.LinAlgError:
        # SVD didn't converge - try fallback strategies
        try:
            # Strategy 1: Add small regularization to diagonal
            min_dim = min(theta_mat.shape[0], theta_mat.shape[1])
            theta_reg = theta_mat.copy()
            for i in range(min_dim):
                theta_reg[i, i] += 1e-14
            u_mat, s_vec, v_mat = np.linalg.svd(theta_reg, full_matrices=False)
        except np.linalg.LinAlgError:
            try:
                # Strategy 2: Use divide-and-conquer algorithm via gesdd (default is gesvd)
                ata = theta_mat.conj().T @ theta_mat
                eigenvals, v_mat = np.linalg.eigh(ata)
                idx = np.argsort(eigenvals)[::-1]
                s_vec = np.sqrt(np.abs(eigenvals[idx]))
                v_mat = v_mat[:, idx].conj().T
                # Compute U from A @ V @ S^{-1}
                s_inv = np.where(s_vec > 1e-15, 1.0 / s_vec, 0.0)
                u_mat = theta_mat @ v_mat.conj().T @ np.diag(s_inv)
            except np.linalg.LinAlgError:
                # Strategy 3: Use QR decomposition instead of SVD
                m, n = theta_mat.shape
                if m <= n:
                    Q, R = np.linalg.qr(theta_mat)              # A = Q R
                    U_r, s_vec, Vh = np.linalg.svd(R, full_matrices=False)  # R = U_r Σ Vh
                    u_mat = Q @ U_r                              # U = Q U_r
                    v_mat = Vh                                   # Vh as usual
                else:
                    # LQ via QR on A^H
                    Qh, Rh = np.linalg.qr(theta_mat.conj().T)    # A^H = Qh Rh
                    # A = Rh^H Qh^H, do SVD of Rh^H (shape m×n but better conditioned)
                    U_r, s_vec, Vh = np.linalg.svd(Rh.conj().T, full_matrices=False)
                    u_mat = U_r
                    v_mat = Vh @ Qh.conj().T

    # 4) decide how many singular values to keep:
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
    a_new = u_mat[:, :keep].reshape(phys_i, left, keep).astype(np.complex128)

    # 6) absorb S into Vh and reshape to B' of shape (phys_j, keep, R)
    v_tensor = np.diag(s_vec[:keep]) @ v_mat[:keep, :]  # shape (keep, phys_j*R)
    v_tensor = v_tensor.reshape(keep, phys_j, right)  # (keep, phys_j, R)
    b_new = v_tensor.transpose(1, 0, 2).astype(np.complex128)  # (phys_j, keep, R)

    return a_new, b_new
