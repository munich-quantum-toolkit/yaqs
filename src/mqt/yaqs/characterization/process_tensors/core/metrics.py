# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Metrics shared by process-tensor tomography and surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _rel_fro_error(A: NDArray[np.complex128], B: NDArray[np.complex128]) -> float:
    """Relative Frobenius error between two matrices."""
    num = np.linalg.norm(A - B, "fro")
    den = np.linalg.norm(B, "fro")
    return float(num / max(den, 1e-15))


def _trace_distance(rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
    """Trace distance between two density matrices."""
    X = rho - sigma
    X = 0.5 * (X + X.conj().T)
    evals = np.linalg.eigvalsh(X)
    return float(0.5 * np.sum(np.abs(evals)))


def _mean_trace_distance_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Mean trace distance over batches of 8-float packed encodings."""
    from .encoding import unpack_rho8

    assert pred_rho8.shape == tgt_rho8.shape
    tds: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        tds.append(_trace_distance(rp, rt))
    return float(np.mean(tds))


def _mean_frobenius_mse_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Mean squared Frobenius error for 8-float encodings (Hilbert–Schmidt squared norm)."""
    from .encoding import unpack_rho8

    assert pred_rho8.shape == tgt_rho8.shape
    diffs: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        d = rp - rt
        diffs.append(float(np.real(np.vdot(d, d))))
    return float(np.mean(diffs))
