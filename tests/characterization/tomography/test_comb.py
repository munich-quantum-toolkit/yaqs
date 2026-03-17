"""Tests for DenseComb and MPOComb wrappers."""

from __future__ import annotations

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.characterization.tomography.comb import DenseComb, MPOComb
from mqt.yaqs.characterization.tomography.estimator import (
    predict_from_dense_upsilon,
)


def test_densecomb_predict_matches_helper() -> None:
    """DenseComb.predict must match the low-level helper."""
    # Simple 1-step identity-like comb on output ⊗ past
    ups = np.eye(2 * 4, dtype=np.complex128)
    timesteps = [0.1]

    def id_map(rho: np.ndarray) -> np.ndarray:
        return rho

    comb = DenseComb(ups, timesteps)
    rho1 = comb.predict([id_map])
    rho2 = predict_from_dense_upsilon(ups, [id_map])
    np.testing.assert_allclose(rho1, rho2, atol=1e-12)


def test_mpocomb_matrix_matches_dense() -> None:
    """MPOComb.to_matrix should match MPO.to_matrix()."""
    mpo = MPO.ising(length=1, J=1.0, g=0.5)
    timesteps: list[float] = [0.1]
    comb = MPOComb(mpo, timesteps)

    np.testing.assert_allclose(
        comb.to_matrix(),
        mpo.to_matrix(),
        atol=1e-12,
    )


def test_mpocomb_qmi_fallback_to_dense() -> None:
    """MPOComb.qmi should agree with DenseComb.qmi via dense fallback."""
    mpo = MPO.ising(length=1, J=1.0, g=0.5)
    timesteps: list[float] = [0.1]
    comb = MPOComb(mpo, timesteps)

    q1 = comb.qmi()
    q2 = comb.to_dense().qmi()
    assert abs(q1 - q2) < 1e-12

