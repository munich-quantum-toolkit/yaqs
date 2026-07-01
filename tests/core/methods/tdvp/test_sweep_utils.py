# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for TDVP sweep utilities and truncation policy."""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806, PLC2701

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.methods.tdvp.sweep_utils import (
    _align_bond,
    _cap_bonds,
    _get_bond_dim,
    _resize_bond,
    _scale_dt,
    _sync_bond_dim,
    get_min_keep,
    renorm_drift,
    renorm_trunc,
    split_tdvp,
    uses_fixed_chi,
)
from tests.conftest import YAQS_TEST_SEED

if TYPE_CHECKING:
    from numpy.typing import NDArray

rng = np.random.default_rng()


def _seeded_haar_random_mps(length: int, *, pad: int) -> MPS:
    """Build a reproducible Haar-random MPS for truncation tests.

    Args:
        length: Chain length.
        pad: Target maximum internal bond dimension.

    Returns:
        MPS initialized with ``state="haar-random"`` using ``YAQS_TEST_SEED``.
    """
    with patch("numpy.random.default_rng", return_value=np.random.default_rng(YAQS_TEST_SEED)):
        return MPS(length, state="haar-random", pad=pad)


def test_split_tdvp_left_right_sqrt() -> None:
    """Test splitting of an MPS tensor using different singular value distribution options."""
    A = rng.random(size=(4, 3, 5)).astype(np.complex128)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)], elapsed_time=0.2, dt=0.1, sample_timesteps=True, trunc_mode="relative"
    )
    physical_dimensions = [A.shape[0] // 2, A.shape[0] // 2]
    for distr in ["left", "right", "sqrt"]:
        A0, A1 = split_tdvp(A, sim_params, physical_dimensions, distr, dynamic=False)
        assert A0.ndim == 3
        assert A1.ndim == 3
        r = A0.shape[2]
        assert A1.shape[1] == r
        A1_recon = A1.transpose((1, 0, 2))
        A_recon = np.tensordot(A0, A1_recon, axes=(2, 0))
        A_recon = A_recon.transpose((0, 2, 1, 3)).reshape(4, 3, 5)
        np.testing.assert_allclose(A, A_recon, atol=1e-6)


def _rand_unitary_like(m: int, n: int, *, seed: int) -> NDArray[np.complex128]:
    """Return an ``m x n`` matrix with orthonormal columns.

    Args:
        m: Row count.
        n: Column count.
        seed: RNG seed for reproducibility.

    Returns:
        Complex matrix with orthonormal columns.

    """
    rng_local = np.random.default_rng(seed)
    A = rng_local.normal(size=(m, n)) + 1j * rng_local.normal(size=(m, n))
    Q, _ = np.linalg.qr(A)
    Q = np.asarray(Q, dtype=np.complex128)
    return cast("NDArray[np.complex128]", Q[:, :n])


def _theta_from_singulars(s: NDArray[np.float64], m: int, n: int, *, seed: int) -> NDArray[np.complex128]:
    """Build a rank-``r`` matrix with prescribed singular values.

    Args:
        s: Singular values to embed.
        m: Row count.
        n: Column count.
        seed: RNG seed for the random unitary factors.

    Returns:
        Complex matrix with the requested singular spectrum.

    """
    r = min(len(s), m, n)
    U = _rand_unitary_like(m, r, seed=seed)
    V = _rand_unitary_like(n, r, seed=seed + 1)
    S = np.diag(s[:r].astype(np.complex128))
    theta = (U @ S @ V.conj().T).astype(np.complex128, copy=False)
    return cast("NDArray[np.complex128]", theta)


def _as_input_tensor(theta: NDArray[np.complex128], d0: int, d1: int, d2: int, d3: int) -> NDArray[np.complex128]:
    """Reshape a matrix into the three-index MPS tensor layout used by split tests.

    Args:
        theta: Dense matrix to reshape.
        d0: Left physical dimension.
        d1: Right physical dimension.
        d2: Left virtual dimension.
        d3: Right virtual dimension.

    Returns:
        Three-index tensor with shape ``(d0 * d1, d2, d3)``.

    """
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
def test_split_discarded_weight(svs: NDArray[np.float64], threshold: float, expected_keep: int) -> None:
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

    A0, A1 = split_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=True)
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
def test_split_relative_kept(svs: NDArray[np.float64], rel_the: float, expected_keep: int) -> None:
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

    A0, A1 = split_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=True)
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
    A0, A1 = split_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=False)
    assert A0.shape[2] == 2
    assert A1.shape[1] == 2


def test_split_tdvp_min_keep() -> None:
    """``split_tdvp`` enforces ``min_keep=2`` even when threshold would drop further."""
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
    A0, A1 = split_tdvp(A_in, sim_params, [d0, d1], "sqrt", dynamic=True)
    assert A0.shape[2] == 2
    assert A1.shape[1] == 2


@pytest.mark.parametrize("distr", ["left", "right", "sqrt"])
def test_split_distribution_rank(distr: str) -> None:
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

    A0, A1 = split_tdvp(A_in, sim_params, [d0, d1], distr, dynamic=True)
    k = A0.shape[2]

    L = A0.reshape(d0 * D0, k)
    R = A1.transpose(1, 0, 2).reshape(k, d1 * D2)
    theta_recon = L @ R

    u, s, v = np.linalg.svd(theta, full_matrices=False)
    theta_opt_k = (u[:, :k] * s[:k]) @ v[:k, :]
    np.testing.assert_allclose(theta_recon, theta_opt_k, atol=1e-10, rtol=1e-8)


def test_split_dynamic_below_cap() -> None:
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

    left_u, right_u = split_tdvp(a_in, uncapped, [d0, d1], "sqrt", dynamic=True)
    left_c, right_c = split_tdvp(a_in, capped, [d0, d1], "sqrt", dynamic=True)

    assert left_u.shape == left_c.shape
    assert right_u.shape == right_c.shape
    np.testing.assert_allclose(left_u, left_c, atol=1e-12)
    np.testing.assert_allclose(right_u, right_c, atol=1e-12)


def test_uses_fixed_chi() -> None:
    """Fixed-χ policy applies to digital params with a cap, not analog or uncapped digital."""
    assert not uses_fixed_chi(AnalogSimParams())
    assert not uses_fixed_chi(AnalogSimParams(max_bond_dim=4))
    assert not uses_fixed_chi(StrongSimParams(preset="exact", get_state=True))
    assert uses_fixed_chi(StrongSimParams(preset="exact", get_state=True, max_bond_dim=4))


def test_renorm_trunc_always_normalizes() -> None:
    """Truncation renorm always calls normalize when invoked."""
    state = MPS(2, state="zeros")
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=2)
    with patch.object(MPS, "normalize") as mock_normalize:
        renorm_trunc(state, params)
        mock_normalize.assert_called_once()


def test_renorm_drift_skips_when_within_tolerance() -> None:
    """Drift renorm is a no-op when the global norm is already unit."""
    state = MPS(2, state="zeros")
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=2, svd_threshold=1e-10)
    with patch.object(MPS, "normalize") as mock_normalize:
        renorm_drift(state, params)
        mock_normalize.assert_not_called()


def test_renorm_drift_normalizes_large_drift() -> None:
    """Drift renorm restores unit norm when truncation drifts far from unity."""
    state = MPS(2, state="zeros")
    state.tensors[0] *= 0.1
    state.tensors[1] *= 0.1
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=2, svd_threshold=1e-10)
    with patch.object(MPS, "normalize") as mock_normalize:
        renorm_drift(state, params)
        mock_normalize.assert_called_once()


def test_sync_bond_dim_truncates_with_consistent_shapes() -> None:
    """SVD bond sync enforces a shared capped dimension on both adjacent tensors."""
    state = _seeded_haar_random_mps(4, pad=4)
    state.normalize()
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=2, svd_threshold=1e-12)
    reference = state.to_vec()
    _sync_bond_dim(state, 1, 2, params)
    assert state.tensors[1].shape[2] == 2
    assert state.tensors[2].shape[1] == 2
    overlap = abs(np.vdot(reference, state.to_vec())) ** 2
    assert overlap >= 0.5


def test_sync_bond_dim_preserves_low_rank_state() -> None:
    """Bond sync is exact when the target dimension keeps the full Schmidt rank."""
    state = MPS(2, state="x+")
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=2, svd_threshold=1e-12)
    reference = state.to_vec()
    _sync_bond_dim(state, 0, 1, params)
    assert state.tensors[0].shape[2] == 1
    assert state.tensors[1].shape[1] == 1
    assert abs(np.vdot(reference, state.to_vec())) ** 2 == pytest.approx(1.0, abs=1e-12)


def test_get_min_keep_with_and_without_cap() -> None:
    """Minimum retained bond rank respects an explicit max_bond_dim cap."""
    uncapped = StrongSimParams(preset="exact", get_state=True)
    capped = StrongSimParams(preset="exact", get_state=True, max_bond_dim=1)
    assert get_min_keep(uncapped) == 2
    assert get_min_keep(capped) == 1


def test_scale_dt_analog_vs_digital() -> None:
    """Analog sweeps scale by dt; digital gate sweeps use the substep fraction directly."""
    analog = AnalogSimParams(observables=[Observable(Z(), 0)], elapsed_time=0.2, dt=0.1, sample_timesteps=True)
    digital = StrongSimParams(preset="exact", get_state=True)
    assert _scale_dt(analog, 0.5) == pytest.approx(0.05)
    assert _scale_dt(digital, 0.5) == pytest.approx(0.5)


def test_get_bond_dim_caps_target() -> None:
    """Bond-dimension lookup respects max_bond_dim when tensors exceed the cap."""
    t0 = np.zeros((2, 1, 4), dtype=np.complex128)
    t1 = np.zeros((2, 4, 1), dtype=np.complex128)
    state = MPS(length=2, tensors=[t0, t1], physical_dimensions=[2, 2])
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=2)
    assert _get_bond_dim(state, 0, params) == 2


def test_align_bond_syncs_mismatched_shapes() -> None:
    """Bond alignment raises the smaller virtual index to the capped target."""
    t0 = np.zeros((2, 1, 2), dtype=np.complex128)
    t1 = np.zeros((2, 1, 1), dtype=np.complex128)
    state = MPS(length=2, tensors=[t0, t1], physical_dimensions=[2, 2])
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=2, svd_threshold=1e-12)
    _align_bond(state, 0, params)
    assert state.tensors[0].shape[2] == state.tensors[1].shape[1]
    assert state.tensors[0].shape[2] <= 2


def test_align_bond_noop_without_max_bond_dim() -> None:
    """Bond alignment is skipped when no fixed bond cap is configured."""
    state = MPS(2, state="x+")
    params = StrongSimParams(preset="exact", get_state=True)
    before = [tensor.copy() for tensor in state.tensors]
    _align_bond(state, 0, params)
    for original, updated in zip(before, state.tensors, strict=True):
        np.testing.assert_array_equal(original, updated)


def test_align_bond_noop_when_bonds_already_match() -> None:
    """Bond alignment is a no-op when adjacent virtual indices already agree."""
    state = MPS(2, state="x+")
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=4)
    before = [tensor.copy() for tensor in state.tensors]
    _align_bond(state, 0, params)
    for original, updated in zip(before, state.tensors, strict=True):
        np.testing.assert_array_equal(original, updated)


def test_cap_bonds_truncates_oversized_internal_bonds() -> None:
    """Global bond capping shrinks bonds above max_bond_dim before a sweep."""
    state = _seeded_haar_random_mps(4, pad=4)
    state.normalize()
    params = StrongSimParams(preset="exact", get_state=True, max_bond_dim=2, svd_threshold=1e-12)
    _cap_bonds(state, params)
    for bond in range(state.length - 1):
        assert state.tensors[bond].shape[2] <= 2
        assert state.tensors[bond + 1].shape[1] <= 2


def test_resize_bond_lead_trail_branches() -> None:
    """Bond transfer resize handles shrink, grow, and no-op paths."""
    bond = np.ones((3, 2), dtype=np.complex128)
    shrunk = _resize_bond(bond, lead=2, trail=1)
    assert shrunk.shape == (2, 1)
    grown = _resize_bond(np.ones((1, 1), dtype=np.complex128), lead=2, trail=3)
    assert grown.shape == (2, 3)
    assert np.isclose(grown[0, 0], 1.0)
    unchanged = _resize_bond(bond, lead=3, trail=2)
    assert unchanged.shape == (3, 2)


def test_sync_bond_dim_padding_path() -> None:
    """Bond sync uses padding when both sides are below the target dimension."""
    t0 = np.zeros((2, 1, 1), dtype=np.complex128)
    t1 = np.zeros((2, 1, 1), dtype=np.complex128)
    state = MPS(length=2, tensors=[t0, t1], physical_dimensions=[2, 2])
    _sync_bond_dim(state, 0, 2, StrongSimParams(preset="exact", get_state=True, max_bond_dim=2))
    assert state.tensors[0].shape[2] == 2
    assert state.tensors[1].shape[1] == 2


def test_sync_bond_dim_aligns_mismatched_bond_widths() -> None:
    """Bond sync pads mismatched virtual indices before targeting a shared dimension."""
    t0 = np.zeros((2, 1, 2), dtype=np.complex128)
    t1 = np.zeros((2, 1, 1), dtype=np.complex128)
    state = MPS(length=2, tensors=[t0, t1], physical_dimensions=[2, 2])
    _sync_bond_dim(state, 0, 2, StrongSimParams(preset="exact", get_state=True, max_bond_dim=2))
    assert state.tensors[0].shape[2] == 2
    assert state.tensors[1].shape[1] == 2
