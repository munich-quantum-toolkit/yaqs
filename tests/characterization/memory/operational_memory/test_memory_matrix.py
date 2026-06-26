# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for memory-matrix construction and spectrum analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.exact import simulate_exact
from mqt.yaqs.characterization.memory.operational_memory.memory_matrix import (
    assemble_memory_matrix,
    center_rows,
    compute_spectrum,
    extract_xyz_channels,
    sanitize_branch_weights,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import sample_probes
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_four_component_memory_metric_matches_xyz_only() -> None:
    """S_V is unchanged when storing (I,X,Y,Z) but using X,Y,Z for the memory matrix."""
    rng = np.random.default_rng(11)
    op = MPO.ising(length=1, J=0.5, g=0.3)
    params = AnalogSimParams(dt=0.05, max_bond_dim=8, order=1)
    probe_set = sample_probes(cut=1, k=1, n_pasts=5, n_futures=4, rng=rng)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    pauli4, weights, _ = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=psi0,
        parallel=False,
    )
    pauli3 = extract_xyz_channels(pauli4)
    m4_raw, m4 = assemble_memory_matrix(pauli4, weights)
    m3_raw, m3 = assemble_memory_matrix(pauli3, weights)
    np.testing.assert_allclose(m4_raw, m3_raw, atol=1e-12)
    np.testing.assert_allclose(m4, m3, atol=1e-12)
    out4 = compute_spectrum(m4)
    out3 = compute_spectrum(m3)
    assert out4["entropy"] == pytest.approx(out3["entropy"])
    assert out4["rank"] == pytest.approx(out3["rank"])


def test_sanitize_branch_weights_clamps_negative_and_nan() -> None:
    """Negative and non-finite weights are clamped for matrix assembly."""
    w = np.array([[1.0, -0.5], [np.nan, np.inf]], dtype=np.float64)
    clean, meta = sanitize_branch_weights(w, log_warnings=False)
    assert meta["negative_count"] == 1
    assert meta["weight_data_invalid"] is True
    np.testing.assert_allclose(clean, [[1.0, 0.0], [0.0, 0.0]])


def test_center_rows_removes_past_mean() -> None:
    """Past-row centering subtracts the column mean along axis 0."""
    m = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    centered = center_rows(m)
    np.testing.assert_allclose(centered.mean(axis=0), [0.0, 0.0], atol=1e-14)


def test_assemble_memory_matrix_beta_scales_rows() -> None:
    """Beta exponent scales branch weights before centering."""
    pauli = np.ones((2, 2, 4), dtype=np.float32)
    pauli[..., 0] = 1.0
    weights = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)
    _raw1, m1 = assemble_memory_matrix(pauli, weights, beta=1.0, center=False)
    _raw2, m2 = assemble_memory_matrix(pauli, weights, beta=2.0, center=False)
    # beta=2 squares branch weights: entries with w=2 scale by 4/2=2 relative to beta=1.
    assert m2[0, 3] == pytest.approx(2.0 * m1[0, 3], rel=1e-6)


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
    """Do not drop a significant SV when only the full cumulative tail exceeds the budget."""
    m = np.diag(np.array([10.0, 5.0, 1e-6, 1e-8], dtype=np.float64))
    out = compute_spectrum(m, discarded_weight_threshold=0.21)
    np.testing.assert_allclose(out["singular_values"], np.array([10.0, 5.0]))
    full = compute_spectrum(m, discarded_weight_threshold=None)
    assert out["entropy"] == pytest.approx(full["entropy"], rel=1e-12, abs=1e-12)


def test_compute_spectrum_rank_equals_exp_entropy() -> None:
    """analyze_memory_matrix reports R(c)=exp(S_V(c))."""
    m = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    memory_matrix = m - m.mean(axis=0, keepdims=True)
    out = compute_spectrum(memory_matrix)
    assert out["rank"] == pytest.approx(math.exp(out["entropy"]), rel=1e-12, abs=1e-12)


def test_compute_spectrum_singular_values_full_matches_svd() -> None:
    """singular_values_full from compute_spectrum matches a direct SVD."""
    rng = np.random.default_rng(5)
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1)
    probe_set = sample_probes(cut=2, k=4, n_pasts=4, n_futures=3, rng=rng)
    psi0 = np.zeros(4, dtype=np.complex128)
    psi0[0] = 1.0 + 0.0j
    pauli, weights, _ = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=psi0,
        parallel=False,
    )
    _raw, memory_matrix = assemble_memory_matrix(pauli, weights, log_weight_warnings=False)
    s_direct = np.linalg.svd(memory_matrix, compute_uv=False)
    ana = compute_spectrum(memory_matrix)
    np.testing.assert_allclose(
        np.sort(s_direct)[::-1],
        np.sort(ana["singular_values_full"])[::-1],
        rtol=1e-10,
        atol=1e-10,
    )
