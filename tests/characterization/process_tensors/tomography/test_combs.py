"""Tests for DenseComb and MPOComb wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.characterization.process_tensors.tomography.combs import DenseComb, MPOComb
from mqt.yaqs.characterization.process_tensors.tomography.data import SequenceData


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


def test_mpocomb_predict_smoke_identity_map() -> None:
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    data = SequenceData(
        sequences=[(0,)],
        outputs=[rho],
        weights=[1.0],
        choi_basis=[np.eye(4, dtype=np.complex128)] * 16,
        choi_indices=[(0, 0)] * 16,
        choi_duals=[np.eye(4, dtype=np.complex128)] * 16,
        timesteps=[0.1],
    )
    comb = data.to_mpo_comb(compress_every=1)

    def id_map(x: np.ndarray) -> np.ndarray:
        return x

    rho_out = comb.predict([id_map])
    assert rho_out.shape == (2, 2)
    np.testing.assert_allclose(rho_out, rho_out.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho_out).real, 1.0, atol=1e-12)


def test_mpocomb_predict_raises_on_empty_interventions() -> None:
    data = SequenceData(
        sequences=[(0,)],
        outputs=[np.eye(2, dtype=np.complex128)],
        weights=[1.0],
        choi_basis=[np.eye(4, dtype=np.complex128)] * 16,
        choi_indices=[(0, 0)] * 16,
        choi_duals=[np.eye(4, dtype=np.complex128)] * 16,
        timesteps=[0.1],
    )
    comb = data.to_mpo_comb(compress_every=1)
    with pytest.raises(ValueError):
        comb.predict([])


def test_mpocomb_predict_raises_on_length_mismatch() -> None:
    data = SequenceData(
        sequences=[(0,)],
        outputs=[np.eye(2, dtype=np.complex128)],
        weights=[1.0],
        choi_basis=[np.eye(4, dtype=np.complex128)] * 16,
        choi_indices=[(0, 0)] * 16,
        choi_duals=[np.eye(4, dtype=np.complex128)] * 16,
        timesteps=[0.1],
    )
    comb = data.to_mpo_comb(compress_every=1)

    def id_map(x: np.ndarray) -> np.ndarray:
        return x

    with pytest.raises(ValueError):
        comb.predict([id_map, id_map])
