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


def _rel_fro_error(a_mat: NDArray[np.complex128], b_mat: NDArray[np.complex128]) -> float:
    """Compute relative Frobenius error.

    Args:
        a_mat: Predicted matrix.
        b_mat: Reference matrix.

    Returns:
        Relative Frobenius error: ||A-B||_F / max(||B||_F, eps).
    """
    num = np.linalg.norm(a_mat - b_mat, "fro")
    den = np.linalg.norm(b_mat, "fro")
    return float(num / max(den, 1e-15))


def _trace_distance(rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
    """Compute trace distance between two density matrices.

    Args:
        rho: Density matrix.
        sigma: Density matrix.

    Returns:
        Trace distance: 0.5 * ||rho - sigma||_1.
    """
    diff_mat = rho - sigma
    diff_mat = 0.5 * (diff_mat + diff_mat.conj().T)
    evals = np.linalg.eigvalsh(diff_mat)
    return float(0.5 * np.sum(np.abs(evals)))


def _mean_trace_distance_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Compute mean trace distance over batches of rho8 encodings.

    Args:
        pred_rho8: Array of packed density matrices with shape (N, 8).
        tgt_rho8: Array of packed density matrices with shape (N, 8).

    Returns:
        Mean trace distance over the batch.
    """
    assert pred_rho8.shape == tgt_rho8.shape
    tds: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        tds.append(_trace_distance(rp, rt))
    return float(np.mean(tds))


def _mean_frobenius_mse_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Compute mean squared Frobenius error over batches of rho8 encodings.

    Args:
        pred_rho8: Array of packed density matrices with shape (N, 8).
        tgt_rho8: Array of packed density matrices with shape (N, 8).

    Returns:
        Mean squared Frobenius error (Hilbert-Schmidt squared norm) over the batch.
    """
    assert pred_rho8.shape == tgt_rho8.shape
    diffs: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        d = rp - rt
        diffs.append(float(np.real(np.vdot(d, d))))
    return float(np.mean(diffs))
