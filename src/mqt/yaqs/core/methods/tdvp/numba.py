# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Numba kernels for dense TDVP effective operators.

Parallel loops respect ``numba.set_num_threads``. The contraction and
flattening conventions are defined in :mod:`.primitives`.
"""

from __future__ import annotations

import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def build_dense_heff_site_numba(left_env: np.ndarray, right_env: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Build the single-site effective operator with parallel Numba loops.

    Uses the contraction and C-order flattening of
    :func:`~mqt.yaqs.core.methods.tdvp.primitives.build_dense_heff_site`.

    Args:
        left_env: Left operator environment with shape ``(a, l, A)``.
        right_env: Right operator environment with shape ``(b, r, B)``.
        op: MPO tensor with shape ``(o, p, l, r)``.

    Returns:
        Complex128 matrix with shape ``(o*A*B, p*a*b)``.
    """
    o_dim, p_dim, mpo_l, mpo_r = op.shape
    a_in, _, a_out = left_env.shape
    b_in, _, b_out = right_env.shape

    # Precompute T1 via partial contraction over MPO left bond
    t1 = np.zeros((o_dim, p_dim, mpo_r, a_in, a_out), dtype=np.complex128)

    for opr in range(o_dim * p_dim * mpo_r):
        o = opr // (p_dim * mpo_r)
        rem = opr % (p_dim * mpo_r)
        p = rem // mpo_r
        r = rem % mpo_r
        for a in prange(a_in):
            for aa in range(a_out):
                sum_val = 0.0 + 0.0j
                for mpo_l_idx in range(mpo_l):
                    sum_val += op[o, p, mpo_l_idx, r] * left_env[a, mpo_l_idx, aa]
                t1[o, p, r, a, aa] = sum_val

    rows = o_dim * a_out * b_out
    cols = p_dim * a_in * b_in
    out = np.zeros((rows, cols), dtype=np.complex128)

    # Final contraction over MPO right bond and reshape to matrix
    for o_aa in prange(o_dim * a_out):
        o = o_aa // a_out
        aa = o_aa % a_out
        for bb in range(b_out):
            row_idx = o_aa * b_out + bb

            for p_a_b in range(p_dim * a_in * b_in):
                col_idx = p_a_b
                p = p_a_b // (a_in * b_in)
                rem_p = p_a_b % (a_in * b_in)
                a = rem_p // b_in
                b = rem_p % b_in

                sum_val = 0.0 + 0.0j
                for r in range(mpo_r):
                    sum_val += t1[o, p, r, a, aa] * right_env[b, r, bb]
                out[row_idx, col_idx] = sum_val

    return out


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def build_dense_heff_bond_numba(left_env: np.ndarray, right_env: np.ndarray) -> np.ndarray:
    """Build the bond effective operator with parallel Numba loops.

    Uses the contraction and C-order flattening of
    :func:`~mqt.yaqs.core.methods.tdvp.primitives.build_dense_heff_bond`.

    Args:
        left_env: Left operator environment with shape ``(u, a, p)``.
        right_env: Right operator environment with shape ``(v, a, w)``.

    Returns:
        Complex128 matrix with shape ``(p*w, u*v)``.
    """
    u_dim, a_dim, p_dim = left_env.shape
    v_dim, _, w_dim = right_env.shape

    rows = p_dim * w_dim
    cols = u_dim * v_dim
    out = np.zeros((rows, cols), dtype=np.complex128)

    for p in range(p_dim):
        for w in range(w_dim):
            row_idx = p * w_dim + w

            for u in prange(u_dim):
                for v in range(v_dim):
                    col_idx = u * v_dim + v

                    sum_val = 0.0 + 0.0j
                    for a in range(a_dim):
                        sum_val += left_env[u, a, p] * right_env[v, a, w]
                    out[row_idx, col_idx] = sum_val

    return out
