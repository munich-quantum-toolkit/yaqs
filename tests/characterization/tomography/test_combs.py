"""Tests for DenseComb and MPOComb wrappers."""

from __future__ import annotations

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.characterization.tomography.exact.combs import DenseComb, MPOComb


def test_densecomb_predict_matches_helper() -> None:
    """DenseComb._predict_raw matches the Choi contraction; predict physicalizes."""
    ups = np.eye(2 * 4, dtype=np.complex128)
    timesteps = [0.1]

    def id_map(rho: np.ndarray) -> np.ndarray:
        return rho

    comb = DenseComb(ups, timesteps)
    # Identity map Choi has trace 2; contract U = I with it gives unnormalized rho = 2*I
    rho_raw = comb._predict_raw([id_map])
    np.testing.assert_allclose(rho_raw, 2.0 * np.eye(2, dtype=np.complex128), atol=1e-12)
    rho = comb.predict([id_map])
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)
    np.testing.assert_allclose(rho, rho_raw / np.trace(rho_raw), atol=1e-12)


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
