# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mqt.yaqs.characterization.tomography.estimator_class import TomographyEstimate
from mqt.yaqs.characterization.tomography.ml_dataset import (
    bloch_vector_from_rho,
    clip_bloch_to_unit_ball,
    density_matrix_from_bloch,
    summarize_grid_coverage,
    tomography_estimate_to_ml_dataset,
    trace_distance,
)


def test_bloch_round_trip_pure_z() -> None:
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    b = bloch_vector_from_rho(rho)
    assert b.shape == (3,)
    rho2 = density_matrix_from_bloch(b)
    assert np.allclose(rho, rho2, atol=1e-6)


def test_clip_bloch() -> None:
    v = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    c = clip_bloch_to_unit_ball(v)
    assert np.linalg.norm(c) <= 1.0 + 1e-6
    assert np.allclose(c, [1.0, 0.0, 0.0], atol=1e-5)


def test_tomography_estimate_to_ml_dataset_k1() -> None:
    k = 1
    tensor = np.zeros((4, 16), dtype=np.complex128)
    weights = np.zeros(16, dtype=np.float64)
    for a in (0, 3, 7):
        rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=np.complex128)
        tensor[(slice(None), a)] = rho.reshape(4)
        weights[a] = 0.5 if a == 0 else 0.25
    est = TomographyEstimate(
        tensor=tensor,
        weights=weights,
        timesteps=[0.0],
        choi_duals=None,
        choi_indices=None,
        choi_basis=None,
    )
    ds = tomography_estimate_to_ml_dataset(est, weight_tol=1e-12, use_estimator_weight=True)
    assert ds.k == 1
    assert ds.sequence_indices.shape == (3, 1)
    assert ds.bloch_target.shape == (3, 3)
    assert ds.sample_weight.shape == (3,)
    assert ds.estimator_weight.shape == (3,)
    assert float(ds.sample_weight[0]) == 0.5
    assert float(ds.estimator_weight[0]) == 0.5
    # exhaustive-style positive cells only
    ds_u = tomography_estimate_to_ml_dataset(est, use_estimator_weight=False)
    assert np.allclose(ds_u.sample_weight, 1.0)
    assert np.allclose(ds_u.estimator_weight, ds.estimator_weight)


def test_trace_distance_identical() -> None:
    rho = np.eye(2, dtype=np.complex128) * 0.5
    assert trace_distance(rho, rho) < 1e-9


def test_empty_dataset_raises() -> None:
    k = 1
    tensor = np.zeros((4, 16), dtype=np.complex128)
    weights = np.zeros(16, dtype=np.float64)
    est = TomographyEstimate(
        tensor=tensor,
        weights=weights,
        timesteps=[0.0],
        choi_duals=None,
        choi_indices=None,
        choi_basis=None,
    )
    with pytest.raises(ValueError, match="No sequences"):
        tomography_estimate_to_ml_dataset(est, weight_tol=1e-30)


def test_exhaustive_count_k2() -> None:
    k = 2
    tensor = np.zeros((4,) + (16,) * k, dtype=np.complex128)
    weights = np.ones((16,) * k, dtype=np.float64) / (16**k)
    for alphas in itertools.product(range(16), repeat=k):
        tensor[(slice(None), *alphas)] = np.eye(2, dtype=np.complex128).reshape(4) * 0.5
    est = TomographyEstimate(
        tensor=tensor,
        weights=weights,
        timesteps=[0.0, 0.0],
        choi_duals=None,
        choi_indices=None,
        choi_basis=None,
    )
    ds = tomography_estimate_to_ml_dataset(est)
    assert ds.sequence_indices.shape[0] == 16**k
    assert ds.weighted_rho8 is None
    cov = summarize_grid_coverage(est, k=k)
    assert cov.n_weight_positive == 16**k
    assert cov.n_weight_at_or_below_tol == 0

    ds8 = tomography_estimate_to_ml_dataset(est, include_weighted_rho8=True)
    assert ds8.weighted_rho8 is not None
    assert ds8.weighted_rho8.shape == (16**k, 8)
