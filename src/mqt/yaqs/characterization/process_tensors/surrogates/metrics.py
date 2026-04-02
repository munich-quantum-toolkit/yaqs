# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Surrogate-specific batch metrics on packed `rho8` vectors."""

from __future__ import annotations

import numpy as np

from .encoding import unpack_rho8
from ..core.metrics import _trace_distance


def _mean_trace_distance_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Mean trace distance over batches of 8-float packed encodings."""
    assert pred_rho8.shape == tgt_rho8.shape
    tds: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        tds.append(_trace_distance(rp, rt))
    return float(np.mean(tds))


def _mean_frobenius_mse_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Mean squared Frobenius error for 8-float encodings (Hilbert–Schmidt squared norm)."""
    assert pred_rho8.shape == tgt_rho8.shape
    diffs: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        d = rp - rt
        diffs.append(float(np.real(np.vdot(d, d))))
    return float(np.mean(diffs))

