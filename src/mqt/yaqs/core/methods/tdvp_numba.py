
from __future__ import annotations

import numba
import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def build_dense_heff_site_numba(
    left_env: np.ndarray, right_env: np.ndarray, op: np.ndarray
) -> np.ndarray:
    # op: (o, p, l, r)
    # left_env: (a, l, A)
    # right_env: (b, r, B)
    # Out: (rows, cols) where rows = o*A*B, cols = p*a*b
    
    o_dim, p_dim, l_dim, r_dim = op.shape
    a_dim, _, A_dim = left_env.shape
    b_dim, _, B_dim = right_env.shape
    
    # Precompute T1: (o, p, r, a, A)
    # This contraction is (o, p, l, r) * (a, l, A) -> (o, p, r, a, A)
    # It sums over l.
    t1 = np.zeros((o_dim, p_dim, r_dim, a_dim, A_dim), dtype=np.complex128)
    
    for o in range(o_dim):
        for p in range(p_dim):
            for r in range(r_dim):
                for a in prange(a_dim):
                    for A in range(A_dim):
                        sum_val = 0.0 + 0.0j
                        for l in range(l_dim):
                            sum_val += op[o, p, l, r] * left_env[a, l, A]
                        t1[o, p, r, a, A] = sum_val

    rows = o_dim * A_dim * B_dim
    cols = p_dim * a_dim * b_dim
    out = np.zeros((rows, cols), dtype=np.complex128)
    
    # Final contraction: T1 * right_env -> Out
    # T1: (o, p, r, a, A)
    # right_env: (b, r, B)
    # Sum over r.
    # Out index mapping:
    # row = (o * A_dim + A) * B_dim + B
    # col = (p * a_dim + a) * b_dim + b
    
    for o in range(o_dim):
        for A in prange(A_dim):
            for B in range(B_dim):
                row_idx = (o * A_dim + A) * B_dim + B
                
                for p in range(p_dim):
                    for a in range(a_dim):
                        for b in range(b_dim):
                            col_idx = (p * a_dim + a) * b_dim + b
                            
                            sum_val = 0.0 + 0.0j
                            for r in range(r_dim):
                                sum_val += t1[o, p, r, a, A] * right_env[b, r, B]
                            out[row_idx, col_idx] = sum_val
                            
    return out


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def build_dense_heff_bond_numba(
    left_env: np.ndarray, right_env: np.ndarray
) -> np.ndarray:
    # left_env: (u, a, p)
    # right_env: (v, a, w)
    # Out: (rows, cols) where rows = p*w, cols = u*v
    # h[p,w,u,v] = sum_a left_env[u,a,p] * right_env[v,a,w]
    
    u_dim, a_dim, p_dim = left_env.shape
    v_dim, _, w_dim = right_env.shape
    
    rows = p_dim * w_dim
    cols = u_dim * v_dim
    out = np.zeros((rows, cols), dtype=np.complex128)
    
    # We want out[row, col]
    # row corresponds to (p, w) -> row = p * w_dim + w
    # col corresponds to (u, v) -> col = u * v_dim + v
    
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
