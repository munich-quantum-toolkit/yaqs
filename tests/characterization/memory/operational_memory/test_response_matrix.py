# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for response-matrix construction and spectrum analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.exact import simulate_exact
from mqt.yaqs.characterization.memory.operational_memory.response_matrix import (
    assemble_response_matrix,
    compute_spectrum,
    sanitize_branch_weights,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import sample_probes
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


@pytest.mark.parametrize("shape", [(2, 3, 3), (2, 4), (2, 3, 5)])
def test_assemble_response_matrix_requires_ixyz_tomography(shape: tuple[int, ...]) -> None:
    """Ambiguous or malformed Pauli-channel inputs are rejected."""
    pauli = np.zeros(shape, dtype=np.float64)
    weights = np.ones((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="pauli_ij must have shape"):
        assemble_response_matrix(pauli, weights)


def test_assemble_response_matrix_requires_matching_weight_shape() -> None:
    """Weights must use the same history and future axes as tomography."""
    pauli = np.zeros((2, 3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="weights_ij must have shape"):
        assemble_response_matrix(pauli, np.ones((3, 2), dtype=np.float64))


def test_sanitize_branch_weights_clamps_negative_and_nan() -> None:
    """Negative and non-finite weights are clamped for matrix assembly."""
    w = np.array([[1.0, -0.5], [np.nan, np.inf]], dtype=np.float64)
    clean, meta = sanitize_branch_weights(w, log_warnings=False)
    assert meta["negative_count"] == 1
    assert meta["weight_data_invalid"] is True
    np.testing.assert_allclose(clean, [[1.0, 0.0], [0.0, 0.0]])


def test_assemble_response_matrix_uses_future_rows_and_history_columns() -> None:
    """A non-square sentinel fixes every response-matrix index and flattening convention."""
    pauli = np.arange(1.0, 25.0, dtype=np.float64).reshape(2, 3, 4)
    weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    response_matrix = assemble_response_matrix(pauli, weights)
    expected = np.empty((12, 2), dtype=np.float64)
    for i in range(2):
        for j in range(3):
            for alpha in range(4):
                expected[4 * j + alpha, i] = weights[i, j] * pauli[i, j, alpha]
    np.testing.assert_allclose(response_matrix, expected)
    assert response_matrix.shape == (12, 2)
    assert not np.allclose(response_matrix.mean(axis=1), 0.0)


def test_transpose_preserves_raw_xyz_block_singular_values() -> None:
    """The transposed XYZ block retains its pre-identity scalar diagnostics."""
    pauli = np.arange(1.0, 25.0, dtype=np.float64).reshape(2, 3, 4)
    weights = np.array([[1.0, 0.5, 0.25], [0.75, 0.4, 0.2]], dtype=np.float64)
    old_orientation = (pauli[..., 1:] * weights[..., np.newaxis]).reshape(2, 9)
    full_orientation = assemble_response_matrix(pauli, weights)
    new_orientation = full_orientation.reshape(3, 4, 2)[:, 1:, :].reshape(9, 2)
    np.testing.assert_allclose(new_orientation, old_orientation.T)
    np.testing.assert_allclose(
        np.linalg.svd(new_orientation, compute_uv=False),
        np.linalg.svd(old_orientation, compute_uv=False),
    )
    assert np.linalg.matrix_rank(new_orientation) == np.linalg.matrix_rank(old_orientation)
    assert np.linalg.norm(new_orientation) == pytest.approx(np.linalg.norm(old_orientation))
    assert compute_spectrum(new_orientation, discarded_weight_threshold=None)["entropy"] == pytest.approx(
        compute_spectrum(old_orientation, discarded_weight_threshold=None)["entropy"]
    )


def test_assemble_response_matrix_beta_scales_weights() -> None:
    """Beta exponent scales branch weights in the raw response matrix."""
    pauli = np.ones((2, 2, 4), dtype=np.float32)
    pauli[..., 0] = 1.0
    weights = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    m1 = assemble_response_matrix(pauli, weights, beta=1.0)
    m2 = assemble_response_matrix(pauli, weights, beta=2.0)
    assert m2[4, 0] == pytest.approx(2.0 * m1[4, 0], rel=1e-6)


def test_identity_rows_equal_branch_weights() -> None:
    """Normalized identity expectations expose branch weights in every future block."""
    pauli = np.zeros((2, 3, 4), dtype=np.float64)
    pauli[..., 0] = 1.0
    weights = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype=np.float64)
    response_matrix = assemble_response_matrix(pauli, weights)
    np.testing.assert_allclose(response_matrix[0::4], weights.T)


def test_maximally_mixed_memoryless_response_is_nonzero_rank_one() -> None:
    """Identity retains the deterministic normalization direction for mixed outputs."""
    pauli = np.zeros((3, 2, 4), dtype=np.float64)
    pauli[..., 0] = 1.0
    history_weights = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    weights = np.broadcast_to(history_weights[:, np.newaxis], (3, 2))
    response_matrix = assemble_response_matrix(pauli, weights)
    assert not np.allclose(response_matrix, 0.0)
    np.testing.assert_allclose(response_matrix.reshape(2, 4, 3)[:, 1:, :], 0.0)
    assert np.linalg.matrix_rank(response_matrix) == 1


def test_compute_spectrum_tail_truncation_reduces_entropy() -> None:
    """Aggressive tail truncation lowers reported entropy."""
    m = np.diag(np.array([10.0, 5.0, 1e-6, 1e-8], dtype=np.float64))
    full = compute_spectrum(m, discarded_weight_threshold=None)
    truncated = compute_spectrum(m, discarded_weight_threshold=1e-4)
    assert truncated["entropy"] <= full["entropy"]


def test_compute_spectrum_tail_truncation_keeps_threshold_mode() -> None:
    """Tail truncation retains modes up to the last one exceeding the weight threshold."""
    m = np.diag(np.array([10.0, 5.0, 1e-6, 1e-8], dtype=np.float64))
    out = compute_spectrum(m, discarded_weight_threshold=1e-4)
    assert out["singular_values"].size == 2
    np.testing.assert_allclose(out["singular_values"], np.array([10.0, 5.0]))


def test_compute_spectrum_tail_truncation_keeps_significant_mode_near_threshold() -> None:
    """Threshold breach keeps modes up to the first discarded tail without over-keeping."""
    m = np.diag(np.array([10.0, 5.0, 1e-6, 1e-8], dtype=np.float64))
    out = compute_spectrum(m, discarded_weight_threshold=0.21)
    np.testing.assert_allclose(out["singular_values"], np.array([10.0]))
    full = compute_spectrum(m, discarded_weight_threshold=None)
    assert out["entropy"] < full["entropy"]


def test_compute_spectrum_modes_equals_exp_entropy() -> None:
    """compute_spectrum reports R(c)=exp(S_V(c))."""
    m = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    out = compute_spectrum(m)
    assert out["modes"] == pytest.approx(math.exp(out["entropy"]), rel=1e-12, abs=1e-12)


def test_compute_spectrum_singular_values_full_matches_svd() -> None:
    """singular_values_full from compute_spectrum matches a direct SVD."""
    rng = np.random.default_rng(5)
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1)
    probe_set = sample_probes(cut=2, num_interventions=4, n_pasts=4, n_futures=3, rng=rng)
    psi0 = np.zeros(4, dtype=np.complex128)
    psi0[0] = 1.0 + 0.0j
    pauli, weights, _ = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=psi0,
        parallel=False,
    )
    response_matrix = assemble_response_matrix(pauli, weights, log_weight_warnings=False)
    s_direct = np.linalg.svd(response_matrix, compute_uv=False)
    ana = compute_spectrum(response_matrix)
    np.testing.assert_allclose(
        np.sort(s_direct)[::-1],
        np.sort(ana["singular_values_full"])[::-1],
        rtol=1e-10,
        atol=1e-10,
    )


def test_paper_convergence_larger_budget_raises_entropy_at_strong_coupling() -> None:
    """Smoke convergence benchmark: larger probe grids resolve stronger memory."""
    cut = 2
    m_values = (4, 16)
    m_max = max(m_values)
    op = MPO.ising(length=6, J=2.0, g=1.0)
    params = AnalogSimParams(dt=0.1)
    psi0 = np.zeros(2**6, dtype=np.complex128)
    psi0[0] = 1.0 + 0.0j
    draw_seed = 100_000 * cut + 10 * round(100 * 2.0)
    probe_set = sample_probes(
        cut=cut,
        num_interventions=20,
        n_pasts=m_max,
        n_futures=m_max,
        rng=np.random.default_rng(draw_seed),
        intervention_style="haar",
    )
    pauli, weights, _ = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=psi0,
        parallel=False,
    )
    entropies: list[float] = []
    for m in m_values:
        p_sub = np.asarray(pauli[:m, :m, ...])
        w = np.asarray(weights)
        w_sub = w[:m, :m, ...] if w.ndim >= 2 else w[:m, ...]
        response_matrix = assemble_response_matrix(
            p_sub,
            w_sub,
            log_weight_warnings=False,
        )
        entropies.append(float(compute_spectrum(response_matrix, discarded_weight_threshold=None)["entropy"]))
    assert entropies[-1] > entropies[0] * 1.05
