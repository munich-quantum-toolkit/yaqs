# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: SLF001 -- white-box tests exercise private comb prediction helpers

"""Tests for DenseComb and MPOComb wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.tomography.combs import DenseComb, MPOComb, compute_entropy_dense
from mqt.yaqs.characterization.memory.backends.tomography.data import SequenceData
from mqt.yaqs.core.data_structures.mpo import MPO


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


def test_densecomb_predict_raises_on_length_mismatch() -> None:
    """DenseComb.predict rejects intervention lists whose length mismatches k."""
    ups = np.eye(2 * 4, dtype=np.complex128)
    comb = DenseComb(ups, [0.1])

    def id_map(rho: np.ndarray) -> np.ndarray:
        return rho

    with pytest.raises(ValueError, match="DenseComb expects"):
        comb.predict([id_map, id_map])


def test_compute_entropy_dense_rejects_invalid_base() -> None:
    """Entropy helpers reject non-positive bases and base equal to 1."""
    rho = np.eye(2, dtype=np.complex128) * 0.5
    with pytest.raises(ValueError, match="entropy base"):
        compute_entropy_dense(rho, base=1)
    with pytest.raises(ValueError, match="entropy base"):
        compute_entropy_dense(rho, base=0)


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
    """MPOComb.predict returns a physical density matrix for a trivial intervention."""
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
    """Predict rejects an empty intervention list."""
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
    with pytest.raises(ValueError, match="interventions list must be non-empty"):
        comb.predict([])


def test_mpocomb_predict_raises_on_length_mismatch() -> None:
    """Predict rejects intervention lists whose length mismatches the comb."""
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

    with pytest.raises(ValueError, match="MPOComb length"):
        comb.predict([id_map, id_map])
