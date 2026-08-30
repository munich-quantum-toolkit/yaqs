# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tensor Network Decompositions.

This module implements left and right moving versions of the QR decomposition,
two-site MPS merge/split with SVD truncation, which are used throughout YAQS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import opt_einsum as oe

from .. import linalg
from ..linalg.svd_utils import TruncMode

if TYPE_CHECKING:
    from numpy.typing import NDArray

SvdDistribution = Literal["left", "right", "sqrt"]

__all__ = [
    "SvdDistribution",
    "TruncMode",
    "left_qr",
    "merge_two_site",
    "right_qr",
    "split_two_site",
]


def right_qr(mps_tensor: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Right QR.

    Performs the QR decomposition of an MPS tensor moving to the right.

    Args:
        mps_tensor: The tensor to be decomposed.

    Returns:
        Tuple ``(q_tensor, r_mat)`` where ``q_tensor`` is the Q tensor with the
        left virtual leg and the physical leg ``(phys, left, new)`` and
        ``r_mat`` is the R matrix with the right virtual leg ``(new, right)``.
    """
    phys, left, right = mps_tensor.shape
    mat = mps_tensor.reshape(phys * left, right)

    q_mat, r_mat = np.linalg.qr(mat)

    q_tensor = q_mat.reshape(phys, left, q_mat.shape[1])
    return q_tensor, r_mat


def left_qr(mps_tensor: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Left QR.

    Performs the QR decomposition of an MPS tensor moving to the left.

    Args:
        mps_tensor: The tensor to be decomposed.

    Returns:
        Tuple ``(q_tensor, r_mat)`` where ``q_tensor`` is the Q tensor with the
        physical leg and the right virtual leg ``(phys, new, right)`` and
        ``r_mat`` is the R matrix with the left virtual leg ``(left, new)``.
    """
    old_shape = mps_tensor.shape
    mps_tensor_t = mps_tensor.transpose(0, 2, 1)
    qr_shape = (old_shape[0] * old_shape[2], old_shape[1])
    mat = mps_tensor_t.reshape(qr_shape)

    q_mat, r_mat = np.linalg.qr(mat)

    q_tensor = q_mat.reshape((old_shape[0], old_shape[2], q_mat.shape[1])).transpose(0, 2, 1)
    r_mat = r_mat.T

    return q_tensor, r_mat


def merge_two_site(left_tensor: NDArray[np.complex128], right_tensor: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Merge two neighboring MPS site tensors into one two-site tensor.

    Contracts over the shared bond and reshapes so the two physical legs become
    one composite leg of dimension ``d_left * d_right``.

    Args:
        left_tensor: Left MPS tensor, shape ``(d_left, D0, D1)``.
        right_tensor: Right MPS tensor, shape ``(d_right, D1, D2)``.

    Returns:
        Merged tensor of shape ``(d_left * d_right, D0, D2)``.
    """
    merged_tensor = np.asarray(oe.contract("abc,dce->adbe", left_tensor, right_tensor), dtype=np.complex128)
    merged_shape = merged_tensor.shape
    return merged_tensor.reshape((merged_shape[0] * merged_shape[1], merged_shape[2], merged_shape[3]))


def split_two_site(
    merged: NDArray[np.complex128],
    physical_dimensions: list[int],
    *,
    svd_distribution: SvdDistribution,
    trunc_mode: TruncMode,
    threshold: float,
    max_bond_dim: int | None,
    min_keep: int = 1,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Split a merged two-site MPS tensor back into two sites via truncated SVD.

    The merged tensor must have shape ``(d_left * d_right, D0, D2)`` with
    ``physical_dimensions == [d_left, d_right]``.

    Args:
        merged: Two-site tensor ``(d_left * d_right, D0, D2)``.
        physical_dimensions: ``[d_left, d_right]`` physical dimensions.
        svd_distribution: How to absorb singular values: ``"left"``, ``"right"``, or ``"sqrt"``.
        trunc_mode: Truncation mode forwarded to :func:`mqt.yaqs.core.linalg.truncate`.
        threshold: Truncation threshold for the chosen mode.
        max_bond_dim: Optional hard cap on bond dimension passed to
            :func:`mqt.yaqs.core.linalg.truncate` (``None`` for no cap).
        min_keep: Minimum number of numerically nonzero singular values to retain (default ``1``).

    Returns:
        Left tensor ``(d_left, D0, keep)`` and right tensor ``(d_right, keep, D2)``.

    """
    return _split_two_site(
        merged,
        physical_dimensions,
        svd_distribution=svd_distribution,
        trunc_mode=trunc_mode,
        threshold=threshold,
        max_bond_dim=max_bond_dim,
        min_keep=min_keep,
        retain_null_workspace=False,
    )


def _split_two_site(
    merged: NDArray[np.complex128],
    physical_dimensions: list[int],
    *,
    svd_distribution: SvdDistribution,
    trunc_mode: TruncMode,
    threshold: float,
    max_bond_dim: int | None,
    min_keep: int,
    retain_null_workspace: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Split using either physical-rank or temporary TDVP-workspace semantics.

    Returns:
        Left and right MPS site tensors after the configured split.

    Raises:
        ValueError: If the physical dimensions or SVD policy are invalid.
    """
    if len(physical_dimensions) != 2:
        msg = f"physical_dimensions must have exactly 2 elements (d_left, d_right); got {len(physical_dimensions)}."
        raise ValueError(msg)
    d_left = physical_dimensions[0]
    d_right = physical_dimensions[1]
    if merged.shape[0] != d_left * d_right:
        msg = "The first dimension of the tensor must be a combination of the given physical dimensions."
        raise ValueError(msg)

    tensor_reshaped = merged.reshape(d_left, d_right, merged.shape[1], merged.shape[2])
    tensor_transposed = tensor_reshaped.transpose((0, 2, 1, 3))
    shape_transposed = tensor_transposed.shape

    theta_mat = tensor_transposed.reshape(
        shape_transposed[0] * shape_transposed[1],
        shape_transposed[2] * shape_transposed[3],
    )
    u_mat, s_vec, v_mat = linalg.svd(theta_mat, full_matrices=False)

    numerical_rank = 0
    if s_vec.size:
        null_tolerance = np.finfo(s_vec.dtype).eps * max(theta_mat.shape) * float(s_vec[0])
        numerical_rank = max(1, int(np.count_nonzero(s_vec > null_tolerance)))
    effective_min_keep = min_keep if retain_null_workspace else min(min_keep, numerical_rank)

    keep = linalg.truncate(
        s_vec,
        mode=trunc_mode,
        threshold=threshold,
        max_bond_dim=max_bond_dim,
        min_keep=effective_min_keep,
    )
    if not retain_null_workspace:
        keep = min(keep, numerical_rank)

    left_tensor = u_mat[:, :keep]
    s_vec = s_vec[:keep]
    right_tensor = v_mat[:keep, :]

    left_tensor = left_tensor.reshape((shape_transposed[0], shape_transposed[1], keep))
    right_tensor = right_tensor.reshape((keep, shape_transposed[2], shape_transposed[3]))

    if svd_distribution == "left":
        left_tensor *= s_vec
    elif svd_distribution == "right":
        right_tensor *= s_vec[:, None, None]
    elif svd_distribution == "sqrt":
        sqrt_sigma = np.sqrt(s_vec)
        left_tensor *= sqrt_sigma
        right_tensor *= sqrt_sigma[:, None, None]
    else:
        msg = "svd_distribution parameter must be left, right, or sqrt."
        raise ValueError(msg)

    right_tensor = right_tensor.transpose((1, 0, 2))
    return left_tensor, right_tensor
