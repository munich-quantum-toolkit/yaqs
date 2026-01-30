# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Numba-accelerated Lanczos methods."""

from __future__ import annotations

import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=True)
def orthogonalize_step(
    v: np.ndarray,
    w: np.ndarray,
    j: int,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    """Perform the orthogonalization step of the Lanczos iteration.

    Computes alpha[j] = <v_j, w>, orthogonalizes w against v_j and v_{j-1},
    and computes the norm beta[j] = ||w||.

    Args:
        v: (N, m) matrix of Lanczos vectors. Columns should be contiguous (F-order) for best performance.
        w: (N,) candidate vector. Modified in-place.
        j: Current iteration index.
        alpha: (m,) diagonal elements.
        beta: (m-1,) off-diagonal elements.

    Returns:
        float: The norm of the orthogonalized vector (beta[j]), or 0.0 if not computed.
    """
    # Note: v[:, j] creates a view. If v is F-ordered, this view is contiguous.
    vj = v[:, j]

    # 1. alpha_j = <v_j, w>
    aj = np.vdot(vj, w).real
    alpha[j] = aj

    # 2. w <- w - aj * vj
    w_len = w.size
    for i in range(w_len):
        w[i] -= aj * vj[i]

    # 3. Orthogonalize against v_{j-1} if applicable
    if j > 0:
        v_prev = v[:, j - 1]
        b_prev = beta[j - 1]
        for i in range(w_len):
            w[i] -= b_prev * v_prev[i]

    # 4. Compute norm
    norm_sq = 0.0
    for i in range(w_len):
        val = w[i]
        norm_sq += val.real * val.real + val.imag * val.imag

    bj = np.sqrt(norm_sq)

    if j < len(beta):
        beta[j] = bj

    return bj


@jit(nopython=True, cache=True, fastmath=True)
def normalize_and_store(v: np.ndarray, w: np.ndarray, j: int, bj: float) -> None:
    """Normalize w by bj and store it in v[:, j+1]."""
    if bj > 0:
        inv_bj = 1.0 / bj

        # explicit loop
        w_len = w.size
        for i in range(w_len):
            v[i, j + 1] = w[i] * inv_bj
