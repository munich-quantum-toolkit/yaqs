from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.tomography.estimate.basis import get_basis_states


def _q_alpha_local(rho_0: np.ndarray, basis_set) -> np.ndarray:
    # Local proposal used by discrete SIS:
    # sample m ∝ Tr(E_m rho_0), p uniform => q(alpha=(p,m)) = q_m[m] * 1/4.
    tr_rho = float(np.trace(rho_0).real)
    if tr_rho <= 0.0:
        q_m = np.full(4, 0.25, dtype=float)
    else:
        E = [basis_set[m][2] for m in range(4)]
        l = np.array([float(np.trace(E_m @ rho_0).real) for E_m in E], dtype=float)
        l = np.clip(l, 0.0, np.inf)
        s = float(np.sum(l))
        q_m = np.full(4, 0.25, dtype=float) if s <= 0.0 else (l / s)
    q = np.zeros(16, dtype=float)
    for p in range(4):
        for m in range(4):
            alpha = 4 * p + m
            q[alpha] = float(q_m[m]) * 0.25
    return q


def test_discrete_sis_uniform_local_mixture_proposals_normalize() -> None:
    rng = np.random.default_rng(123)
    z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = z @ z.conj().T
    rho = rho / np.trace(rho)

    basis_set = get_basis_states(basis="tetrahedral")
    q_uniform = np.full(16, 1.0 / 16.0, dtype=float)
    q_local = _q_alpha_local(rho, basis_set)

    np.testing.assert_allclose(float(np.sum(q_uniform)), 1.0, atol=1e-12)
    np.testing.assert_allclose(float(np.sum(q_local)), 1.0, atol=1e-12)
    assert np.all(q_uniform >= 0.0)
    assert np.all(q_local >= 0.0)

    eps = 0.1
    q_mix = eps * q_uniform + (1.0 - eps) * q_local
    np.testing.assert_allclose(float(np.sum(q_mix)), 1.0, atol=1e-12)
    assert np.all(q_mix >= 0.0)

