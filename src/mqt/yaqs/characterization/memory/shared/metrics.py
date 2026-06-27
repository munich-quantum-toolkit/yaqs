# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Metrics shared by process-tensor tomography and surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .encoding import unpack_rho8

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _validate_square_matrix_pair(
    a_mat: NDArray[np.complex128],
    b_mat: NDArray[np.complex128],
    *,
    name_a: str,
    name_b: str,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Validate two inputs are identically shaped square matrices.

    Args:
        a_mat: First matrix.
        b_mat: Second matrix.
        name_a: Parameter name for error messages.
        name_b: Parameter name for error messages.

    Returns:
        The validated pair as complex128 arrays.

    Raises:
        ValueError: If shapes differ or are not square.
    """
    a = np.asarray(a_mat, dtype=np.complex128)
    b = np.asarray(b_mat, dtype=np.complex128)
    if a.shape != b.shape:
        msg = f"{name_a} and {name_b} must share the same shape, got {a.shape} vs {b.shape}."
        raise ValueError(msg)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        msg = f"{name_a} and {name_b} must be square matrices, got shape {a.shape}."
        raise ValueError(msg)
    return a, b


def compute_rel_fro_error(a_mat: NDArray[np.complex128], b_mat: NDArray[np.complex128]) -> float:
    """Compute relative Frobenius error.

    Args:
        a_mat: Predicted matrix.
        b_mat: Reference matrix.

    Returns:
        Relative Frobenius error: ||A-B||_F / max(||B||_F, eps).
    """
    a, b = _validate_square_matrix_pair(a_mat, b_mat, name_a="a_mat", name_b="b_mat")
    num = np.linalg.norm(a - b, "fro")
    den = np.linalg.norm(b, "fro")
    return float(num / max(den, 1e-15))


def compute_trace_distance(rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
    """Compute trace distance between two density matrices.

    Args:
        rho: Density matrix.
        sigma: Density matrix.

    Returns:
        Trace distance: 0.5 * ||rho - sigma||_1.
    """
    rho_h, sigma_h = _validate_square_matrix_pair(rho, sigma, name_a="rho", name_b="sigma")
    diff_mat = rho_h - sigma_h
    diff_mat = 0.5 * (diff_mat + diff_mat.conj().T)
    evals = np.linalg.eigvalsh(diff_mat)
    return float(0.5 * np.sum(np.abs(evals)))


def mean_trace_distance_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Compute mean trace distance over batches of rho8 encodings.

    Args:
        pred_rho8: Array of packed density matrices with shape (N, 8).
        tgt_rho8: Array of packed density matrices with shape (N, 8).

    Returns:
        Mean trace distance over the batch.

    Raises:
        ValueError: If ``pred_rho8`` and ``tgt_rho8`` do not share the same shape.
    """
    if pred_rho8.shape != tgt_rho8.shape:
        msg = f"pred_rho8 and tgt_rho8 must share shape, got {pred_rho8.shape} vs {tgt_rho8.shape}."
        raise ValueError(msg)
    if pred_rho8.shape[0] == 0:
        msg = "pred_rho8 and tgt_rho8 must have a non-zero batch dimension."
        raise ValueError(msg)
    tds: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        tds.append(compute_trace_distance(rp, rt))
    return float(np.mean(tds))


def mean_frobenius_mse_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Compute mean squared Frobenius error over batches of rho8 encodings.

    Args:
        pred_rho8: Array of packed density matrices with shape (N, 8).
        tgt_rho8: Array of packed density matrices with shape (N, 8).

    Returns:
        Mean squared Frobenius error (Hilbert-Schmidt squared norm) over the batch.

    Raises:
        ValueError: If ``pred_rho8`` and ``tgt_rho8`` do not share the same shape.
    """
    if pred_rho8.shape != tgt_rho8.shape:
        msg = f"pred_rho8 and tgt_rho8 must share shape, got {pred_rho8.shape} vs {tgt_rho8.shape}."
        raise ValueError(msg)
    if pred_rho8.shape[0] == 0:
        msg = "pred_rho8 and tgt_rho8 must have a non-zero batch dimension."
        raise ValueError(msg)
    diffs: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        d = rp - rt
        diffs.append(float(np.real(np.vdot(d, d))))
    return float(np.mean(diffs))
