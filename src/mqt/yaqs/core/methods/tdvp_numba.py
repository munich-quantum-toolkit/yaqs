# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Numba-accelerated kernels for TDVP dense effective Hamiltonian construction.

This module provides JIT-compiled, parallelized implementations of the dense
effective Hamiltonian construction for both single-site and bond updates in
Time-Dependent Variational Principle (TDVP) simulations. These kernels replace
slower einsum-based implementations with explicit nested loops optimized by
Numba's LLVM backend, achieving 2-3x speedups for bond dimensions D >= 16.
"""

from __future__ import annotations

import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def build_dense_heff_site_numba(left_env: np.ndarray, right_env: np.ndarray, op: np.ndarray) -> np.ndarray:
    r"""Numba-optimized construction of the dense effective operator for single-site updates.

    This function is a JIT-compiled, parallelized implementation of the dense effective
    Hamiltonian construction for TDVP single-site updates. It replaces the reference
    ``einsum``-based implementation with explicit nested loops that are optimized by
    Numba's LLVM backend.

    The operator is defined by the contraction:

        H_eff[o,A,B,p,a,b] = sum_{l,r} op[o,p,l,r] * left_env[a,l,A] * right_env[b,r,B]

    This is computed in two stages to reduce memory overhead:
      1. Partial contraction: T1[o,p,r,a,A] = sum_l op[o,p,l,r] * left_env[a,l,A]
      2. Final contraction: H_eff = sum_r T1[o,p,r,a,A] * right_env[b,r,B]

    The result is flattened to a 2D matrix using row-major (C-order) index mapping:
      - Row index: (o * a_out + aa) * b_out + bb
      - Col index: (p * a_in + a) * b_in + b

    Args:
        left_env: Left operator block, shape ``(a, l, A)`` where lowercase/uppercase
            denote input/output virtual dimensions.
        right_env: Right operator block, shape ``(b, r, B)``.
        op: Local MPO tensor, shape ``(o, p, l, r)`` where ``o`` and ``p`` are
            physical dimensions and ``l``, ``r`` are MPO bond dimensions.

    Returns:
        Dense matrix ``H_eff`` of shape ``(o*A*B, p*a*b)`` representing the
        effective local Hamiltonian operator.

    Notes:
        - Uses ``parallel=True`` for automatic parallelization across outer loops.
        - Benchmarks show this is ~2-3x faster than ``einsum`` for bond dimensions
          D >= 16, with the speedup increasing for larger tensors.
        - The two-stage contraction pattern reduces peak memory usage compared to
          a single 6-index intermediate tensor.
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
        for a in prange(a_in):  # type: ignore[attr-defined]
            for aa in range(a_out):
                sum_val = 0.0 + 0.0j
                for mpo_l_idx in range(mpo_l):
                    sum_val += op[o, p, mpo_l_idx, r] * left_env[a, mpo_l_idx, aa]
                t1[o, p, r, a, aa] = sum_val

    rows = o_dim * a_out * b_out
    cols = p_dim * a_in * b_in
    out = np.zeros((rows, cols), dtype=np.complex128)

    # Final contraction over MPO right bond and reshape to matrix
    for o_aa in prange(o_dim * a_out):  # type: ignore[attr-defined]
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
    r"""Numba-optimized construction of the dense effective operator for bond updates.

    This function is a JIT-compiled, parallelized implementation of the dense effective
    Hamiltonian construction for zero-site (bond) updates in TDVP. It replaces the
    reference ``einsum``-based implementation with explicit nested loops optimized by
    Numba's LLVM backend.

    The operator is defined by the contraction:

        H_eff[p,w,u,v] = sum_a left_env[u,a,p] * right_env[v,a,w]

    The result is flattened to a 2D matrix using row-major (C-order) index mapping:
      - Row index: p * w_dim + w
      - Col index: u * v_dim + v

    Args:
        left_env: Left operator block, shape ``(u, a, p)``.
        right_env: Right operator block, shape ``(v, a, w)``.

    Returns:
        Dense matrix ``H_eff`` of shape ``(p*w, u*v)`` representing the
        effective bond Hamiltonian operator.

    Notes:
        - Uses ``parallel=True`` for automatic parallelization across ``u`` dimension.
        - Benchmarks show this is ~2-3x faster than ``einsum`` for bond dimensions
          D >= 16.
        - This function is called during the backward evolution steps in 1TDVP.
    """
    u_dim, a_dim, p_dim = left_env.shape
    v_dim, _, w_dim = right_env.shape

    rows = p_dim * w_dim
    cols = u_dim * v_dim
    out = np.zeros((rows, cols), dtype=np.complex128)

    for p in range(p_dim):
        for w in range(w_dim):
            row_idx = p * w_dim + w

            for u in prange(u_dim):  # type: ignore[attr-defined]
                for v in range(v_dim):
                    col_idx = u * v_dim + v

                    sum_val = 0.0 + 0.0j
                    for a in range(a_dim):
                        sum_val += left_env[u, a, p] * right_env[v, a, w]
                    out[row_idx, col_idx] = sum_val

    return out
