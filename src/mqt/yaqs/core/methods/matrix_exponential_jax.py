from __future__ import annotations

from functools import partial
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def _tridiag_to_dense(alpha: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """Build dense symmetric tridiagonal matrix from diag alpha (m,) and offdiag beta (m-1,)."""
    m = alpha.shape[0]
    T = jnp.diag(alpha)
    T = T + jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)
    return T


@partial(jax.jit, static_argnames=("lanczos_iterations",))
def expm_krylov_dense_jax(
    h_eff: jnp.ndarray,  # (n, n) complex
    vec: jnp.ndarray,    # (n,) complex
    dt: float,
    lanczos_iterations: int,
) -> jnp.ndarray:
    """
    Krylov/Lanczos approximation of exp(-1j*dt*h_eff) @ vec, dense operator path.

    Notes:
    - Fixed-shape (runs exactly `lanczos_iterations` iterations).
    - If Lanczos "breaks down" (beta_j ~ 0), we mask subsequent updates to keep shapes static.
    - Intended to be called with complex128 if you care about matching NumPy closely.
    """
    n = vec.shape[0]
    m = lanczos_iterations

    # Normalize starting vector
    v0_norm = jnp.linalg.norm(vec)
    # Avoid NaNs on zero vector; behavior matches "return zero" effectively
    v0 = jnp.where(v0_norm > 0, vec / v0_norm, vec)

    # Allocate
    alpha = jnp.zeros((m,), dtype=jnp.float64)
    beta = jnp.zeros((m - 1,), dtype=jnp.float64)
    V = jnp.zeros((n, m), dtype=h_eff.dtype)
    V = V.at[:, 0].set(v0)

    # Similar spirit to your eps_cut; static threshold
    eps_cut = 100.0 * n * jnp.finfo(jnp.float64).eps

    def step(j: int, carry):
        alpha, beta, V, broken = carry

        vj = V[:, j]
        w = h_eff @ vj

        aj = jnp.real(jnp.vdot(vj, w))
        w = w - aj * vj
        w = jnp.where(j > 0, w - beta[j - 1] * V[:, j - 1], w)

        bj = jnp.linalg.norm(w)
        new_broken = jnp.logical_or(broken, bj < eps_cut)

        # Write alpha_j if not already broken
        alpha = alpha.at[j].set(jnp.where(broken, alpha[j], aj))

        # Write beta_j and next vector if j < m-1 and not broken
        def write_next(_):
            # Avoid divide-by-zero: if bj==0 this branch won't be taken (since new_broken True)
            v_next = w / bj
            V2 = V.at[:, j + 1].set(v_next)
            b2 = beta.at[j].set(bj)
            return alpha, b2, V2, new_broken

        def skip_next(_):
            # Keep arrays unchanged (or keep existing beta/V entries)
            return alpha, beta, V, new_broken

        return jax.lax.cond((j < m - 1) & (~broken) & (~new_broken), write_next, skip_next, operand=None)

    # Run steps j=0..m-2
    alpha, beta, V, broken = jax.lax.fori_loop(0, m - 1, step, (alpha, beta, V, False))

    # Final alpha_{m-1} (if not broken)
    v_last = V[:, m - 1]
    w_last = h_eff @ v_last
    a_last = jnp.real(jnp.vdot(v_last, w_last))
    alpha = alpha.at[m - 1].set(jnp.where(broken, alpha[m - 1], a_last))

    # Krylov-space eigenproblem (m is small; dense is fine)
    T = _tridiag_to_dense(alpha, beta)
    w_hess, U = jnp.linalg.eigh(T)  # (m,), (m,m)

    # coeffs = ||v|| * exp(-i dt w) * U[0, :]
    coeffs = v0_norm * jnp.exp(-1j * dt * w_hess) * U[0, :]

    # y ≈ V @ (U @ coeffs)
    y = V @ (U @ coeffs)
    return y
