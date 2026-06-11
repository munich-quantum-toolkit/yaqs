# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for TDVP sweep utilities and truncation policy."""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.methods.tdvp.sweep_utils import (
    _is_fixed_chi_digital,  # noqa: PLC2701
    _renorm_on_drift,  # noqa: PLC2701
    _renorm_on_trunc,  # noqa: PLC2701
    _split_two_site_tdvp,  # noqa: PLC2701
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

rng = np.random.default_rng()


def test_split_two_site_tdvp_left_right_sqrt() -> None:
    """Test splitting of an MPS tensor using different singular value distribution options."""
    A = rng.random(size=(4, 3, 5)).astype(np.complex128)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)], elapsed_time=0.2, dt=0.1, sample_timesteps=True, trunc_mode="relative"
    )
    physical_dimensions = [A.shape[0] // 2, A.shape[0] // 2]
    for distr in ["left", "right", "sqrt"]:
        A0, A1 = _split_two_site_tdvp(A, sim_params, physical_dimensions, distr, dynamic=False)
        assert A0.ndim == 3
        assert A1.ndim == 3
        r = A0.shape[2]
        assert A1.shape[1] == r
        A1_recon = A1.transpose((1, 0, 2))
        A_recon = np.tensordot(A0, A1_recon, axes=(2, 0))
        A_recon = A_recon.transpose((0, 2, 1, 3)).reshape(4, 3, 5)
        np.testing.assert_allclose(A, A_recon, atol=1e-6)


def _rand_unitary_like(m: int, n: int, *, seed: int) -> NDArray[np.complex128]:
    rng_local = np.random.default_rng(seed)
    A = rng_local.normal(size=(m, n)) + 1j * rng_local.normal(size=(m, n))
    Q, _ = np.linalg.qr(A)
    Q = np.asarray(Q, dtype=np.complex128)
    return cast("NDArray[np.complex128]", Q[:, :n])


def _theta_from_singulars(s: NDArray[np.float64], m: int, n: int, *, seed: int) -> NDArray[np.complex128]:
    r = min(len(s), m, n)
    U = _rand_unitary_like(m, r, seed=seed)
    V = _rand_unitary_like(n, r, seed=seed + 1)
    S = np.diag(s[:r].astype(np.complex128))
    theta = (U @ S @ V.conj().T).astype(np.complex128, copy=False)
    return cast("NDArray[np.complex128]", theta)


def _as_input_tensor(theta: NDArray[np.complex128], d0: int, d1: int, d2: int, d3: int) -> NDArray[np.complex128]:
    t = theta.reshape(d0, d2, d1, d3).transpose(0, 2, 1, 3)
    return cast("NDArray[np.complex128]", t.reshape(d0 * d1, d2, d3))


@pytest.mark.parametrize(
    ("svs", "threshold", "expected_keep"),
    [
        (np.array([1.0, 0.5, 0.1, 0.0100001]), 1e-4, 4),
        (np.array([1.0, 0.5, 0.01, 0.001]), 1e-4, 3),
        (np.array([1.0, 0.2, 0.2, 0.2]), 0.2**2 * 3, 1),
    ],
)
def test_split_truncation_discarded_weight_kept_count(
    svs: NDArray[np.float64], threshold: float, expected_keep: int
) -> None:
    """discarded_weight: keep count matches tail-power threshold; shapes consistent, robust at boundary."""
    d0, d1, D0, D2 = 2, 2, 3, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=11)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        svd_threshold=threshold,
        trunc_mode="discarded_weight",
        sample_timesteps=True,
    )

    A0, A1 = _split_two_site_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=True)
    keep = A0.shape[2]
    assert A1.shape[1] == keep

    total_power = float(np.sum(svs**2))
    tol = 64.0 * np.finfo(float).eps * max(1.0, total_power)

    tail_at_expected = svs[expected_keep:] if expected_keep < len(svs) else np.array([], dtype=svs.dtype)
    boundary_case = np.isclose(np.sum(tail_at_expected**2), threshold, rtol=0.0, atol=tol)

    if boundary_case:
        acceptable = {expected_keep}
        if expected_keep > 1:
            acceptable.add(expected_keep - 1)
        if expected_keep < len(svs):
            acceptable.add(expected_keep + 1)
        assert keep in acceptable
    else:
        assert keep == expected_keep

    tail = svs[keep:] if keep < len(svs) else np.array([], dtype=svs.dtype)
    tail_power = float(np.sum(tail**2))

    assert tail_power <= threshold + tol or keep == len(svs)

    if keep > 1:
        tail_prev = svs[keep - 1 :]
        tail_prev_power = float(np.sum(tail_prev**2))
        assert tail_prev_power > threshold - tol


@pytest.mark.parametrize(
    ("svs", "rel_the", "expected_keep"),
    [
        (np.array([1.0, 0.6, 0.4, 0.1]), 0.5, 2),
        (np.array([1.0, 0.99, 0.98]), 0.95, 3),
        (np.array([1.0, 0.55, 0.3]), 0.5, 2),
    ],
)
def test_split_truncation_relative_kept_count(svs: NDArray[np.float64], rel_the: float, expected_keep: int) -> None:
    """relative: keep count matches s_i/s_max >= threshold; shapes consistent."""
    d0, d1, D0, D2 = 2, 3, 2, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=12)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        svd_threshold=rel_the,
        trunc_mode="relative",
        sample_timesteps=True,
    )

    A0, A1 = _split_two_site_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=True)
    keep = A0.shape[2]
    assert keep == expected_keep
    assert A1.shape[1] == keep

    smax = float(np.max(svs))
    if expected_keep > 0:
        assert np.all(svs[:expected_keep] / smax > rel_the)
    if expected_keep < len(svs):
        assert not (svs[expected_keep] / smax > rel_the)


def test_split_truncation_max_bond_enforced() -> None:
    """max_bond_dim caps truncation in relative mode."""
    svs = np.array([1.0, 0.9, 0.8, 0.7])
    d0, d1, D0, D2 = 2, 2, 3, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=13)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        max_bond_dim=2,
        svd_threshold=0.5,
        trunc_mode="relative",
        sample_timesteps=True,
    )
    A0, A1 = _split_two_site_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=False)
    assert A0.shape[2] == 2
    assert A1.shape[1] == 2


def test_split_two_site_tdvp_min_keep() -> None:
    """``_split_two_site_tdvp`` enforces ``min_keep=2`` even when threshold would drop further."""
    svs = np.array([1.0, 1e-12, 1e-13, 1e-14], dtype=np.float64)
    d0, d1, D0, D2 = 2, 2, 2, 2
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=31)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        svd_threshold=1e-6,
        trunc_mode="relative",
        sample_timesteps=True,
    )
    A0, A1 = _split_two_site_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=True)
    assert A0.shape[2] == 2
    assert A1.shape[1] == 2


@pytest.mark.parametrize("distr", ["left", "right", "sqrt"])
def test_split_truncation_distribution_reconstructs_optimal_rank(distr: str) -> None:
    """All SVD distribution choices reconstruct the optimal rank-k approximation."""
    svs = np.array([1.0, 0.7, 0.3, 0.1])
    d0, d1, D0, D2 = 2, 2, 3, 3
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=14)
    A_in = _as_input_tensor(theta, d0, d1, D0, D2)

    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        max_bond_dim=2,
        svd_threshold=0.5,
        trunc_mode="relative",
        sample_timesteps=True,
    )

    A0, A1 = _split_two_site_tdvp(A_in, sim_params, [d0, d1], distr, dynamic=True)
    k = A0.shape[2]

    L = A0.reshape(d0 * D0, k)
    R = A1.transpose(1, 0, 2).reshape(k, d1 * D2)
    theta_recon = L @ R

    u, s, v = np.linalg.svd(theta, full_matrices=False)
    theta_opt_k = (u[:, :k] * s[:k]) @ v[:k, :]
    np.testing.assert_allclose(theta_recon, theta_opt_k, atol=1e-10, rtol=1e-8)


def test_dynamic_split_matches_uncapped_when_rank_below_cap() -> None:
    """Dynamic splits ignore ``max_bond_dim`` when the kept rank stays below the cap."""
    svs = np.array([1.0, 0.4, 0.2])
    d0, d1, D0, D2 = 2, 2, 2, 2
    theta = _theta_from_singulars(svs, d0 * D0, d1 * D2, seed=21)
    a_in = _as_input_tensor(theta, d0, d1, D0, D2)

    uncapped = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        max_bond_dim=None,
        svd_threshold=1e-14,
        trunc_mode="discarded_weight",
        sample_timesteps=True,
    )
    capped = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        max_bond_dim=64,
        svd_threshold=1e-14,
        trunc_mode="discarded_weight",
        sample_timesteps=True,
    )

    left_u, right_u = _split_two_site_tdvp(a_in, uncapped, [d0, d1], "sqrt", dynamic=True)
    left_c, right_c = _split_two_site_tdvp(a_in, capped, [d0, d1], "sqrt", dynamic=True)

    assert left_u.shape == left_c.shape
    assert right_u.shape == right_c.shape
    np.testing.assert_allclose(left_u, left_c, atol=1e-12)
    np.testing.assert_allclose(right_u, right_c, atol=1e-12)


def test_is_fixed_chi_digital() -> None:
    """Fixed-χ policy applies to digital params with a cap, not analog or uncapped digital."""
    assert not _is_fixed_chi_digital(AnalogSimParams())
    assert not _is_fixed_chi_digital(AnalogSimParams(max_bond_dim=4))
    assert not _is_fixed_chi_digital(StrongSimParams(preset="exact", get_state=True))
    assert _is_fixed_chi_digital(StrongSimParams(preset="exact", get_state=True, max_bond_dim=4))


def test_renorm_on_trunc_always_normalizes() -> None:
    """Truncation renorm always calls normalize when invoked."""
    state = MPS(2, state="zeros")
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=2)
    with patch.object(MPS, "normalize") as mock_normalize:
        _renorm_on_trunc(state, params)
        mock_normalize.assert_called_once()


def test_renorm_on_drift_skips_when_within_tolerance() -> None:
    """Drift renorm is a no-op when the global norm is already unit."""
    state = MPS(2, state="zeros")
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=2, svd_threshold=1e-10)
    with patch.object(MPS, "normalize") as mock_normalize:
        _renorm_on_drift(state, params)
        mock_normalize.assert_not_called()


def test_renorm_on_drift_normalizes_large_drift() -> None:
    """Drift renorm restores unit norm when truncation drifts far from unity."""
    state = MPS(2, state="zeros")
    state.tensors[0] *= 0.1
    state.tensors[1] *= 0.1
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=2, svd_threshold=1e-10)
    with patch.object(MPS, "normalize") as mock_normalize:
        _renorm_on_drift(state, params)
        mock_normalize.assert_called_once()
