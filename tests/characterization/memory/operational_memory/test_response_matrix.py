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


@pytest.mark.parametrize("invalid_feature", [np.nan, np.inf])
def test_assemble_response_matrix_rejects_nonfinite_tomography(invalid_feature: float) -> None:
    """Response assembly rejects non-finite tomography coefficients."""
    pauli = np.zeros((1, 1, 4), dtype=np.float64)
    pauli[..., 0] = 1.0
    pauli[..., 1] = invalid_feature

    with pytest.raises(ValueError, match="pauli_ij must contain only finite values"):
        assemble_response_matrix(pauli, np.ones((1, 1), dtype=np.float64))


def test_assemble_response_matrix_uses_future_rows_and_history_columns() -> None:
    """A non-square sentinel fixes every response-matrix index and flattening convention."""
    pauli = np.arange(1.0, 25.0, dtype=np.float64).reshape(2, 3, 4)
    pauli[..., 0] = 1.0
    weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    response_matrix = assemble_response_matrix(pauli, weights)
    expected = np.empty((12, 2), dtype=np.float64)
    for i in range(2):
        for j in range(3):
            for alpha in range(4):
                expected[4 * j + alpha, i] = weights[i, j] * pauli[i, j, alpha]
    np.testing.assert_allclose(response_matrix, expected)
    assert response_matrix.shape == (12, 2)
    assert not np.allclose(response_matrix.mean(axis=1), 0.0)


def test_transpose_preserves_xyz_block_singular_values() -> None:
    """The transposed XYZ block retains its pre-identity scalar diagnostics."""
    pauli = np.arange(1.0, 25.0, dtype=np.float64).reshape(2, 3, 4)
    pauli[..., 0] = 1.0
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


def test_assemble_response_matrix_is_linear_in_probabilities() -> None:
    """The canonical response matrix uses probabilities linearly."""
    pauli = np.ones((2, 2, 4), dtype=np.float32)
    weights = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    response_matrix = assemble_response_matrix(pauli, weights)
    np.testing.assert_allclose(assemble_response_matrix(pauli, 2.0 * weights), 2.0 * response_matrix)


@pytest.mark.parametrize(
    "invalid_weight",
    [np.nextafter(0.0, -np.inf), -0.1, 1.1, np.nan, np.inf],
)
def test_assemble_response_matrix_rejects_invalid_probabilities(invalid_weight: float) -> None:
    """Canonical assembly never silently changes invalid backend probabilities."""
    pauli = np.zeros((1, 1, 4), dtype=np.float64)
    pauli[..., 0] = 1.0
    with pytest.raises(ValueError, match="probabilities in \\[0, 1\\]"):
        assemble_response_matrix(pauli, np.array([[invalid_weight]], dtype=np.float64))


def test_assemble_response_matrix_clips_probability_roundoff() -> None:
    """A probability one ulp above one is accepted and clipped to one."""
    pauli = np.array([[[1.0, 0.5, 0.0, -0.5]]], dtype=np.float64)
    weights = np.array([[np.nextafter(1.0, np.inf)]], dtype=np.float64)

    response_matrix = assemble_response_matrix(pauli, weights)

    np.testing.assert_array_equal(response_matrix[:, 0], pauli[0, 0])


def test_assemble_response_matrix_requires_normalized_identity_channel() -> None:
    """The I response must expose the supplied probability without rescaling."""
    pauli = np.zeros((1, 1, 4), dtype=np.float64)
    pauli[..., 0] = 0.75
    with pytest.raises(ValueError, match="identity expectations must equal 1"):
        assemble_response_matrix(pauli, np.ones((1, 1), dtype=np.float64))


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


def test_compute_spectrum_rejects_zero_response() -> None:
    """A zero response has no normalized modal distribution."""
    with pytest.raises(ValueError, match="nonzero Frobenius norm"):
        compute_spectrum(np.zeros((4, 3), dtype=np.float64))


def test_compute_spectrum_accepts_nonzero_response_below_squared_underflow_scale() -> None:
    """A nonzero response remains defined when direct singular-value squaring would underflow."""
    tiny = compute_spectrum(np.diag([1e-200, 5e-201]), discarded_weight_threshold=None)
    reference = compute_spectrum(np.diag([1.0, 0.5]), discarded_weight_threshold=None)
    assert tiny["entropy"] == pytest.approx(reference["entropy"])
    assert tiny["modes"] == pytest.approx(reference["modes"])


def test_compute_spectrum_returns_compact_svd_in_response_matrix_orientation() -> None:
    """Compact SVD columns represent future responses on the left and histories on the right."""
    response_matrix = np.array(
        [[3.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 0.5]],
        dtype=np.float64,
    )
    out = compute_spectrum(response_matrix, discarded_weight_threshold=None)
    left = out["left_singular_vectors"]
    singular_values = out["singular_values_full"]
    right = out["right_singular_vectors"]

    assert left.shape == (response_matrix.shape[0], 2)
    assert singular_values.shape == (2,)
    assert right.shape == (response_matrix.shape[1], 2)
    np.testing.assert_allclose((left * singular_values) @ right.conj().T, response_matrix)


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
    response_matrix = assemble_response_matrix(pauli, weights)
    s_direct = np.linalg.svd(response_matrix, compute_uv=False)
    ana = compute_spectrum(response_matrix)
    np.testing.assert_allclose(
        np.sort(s_direct)[::-1],
        np.sort(ana["singular_values_full"])[::-1],
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    ("depolarization", "expected_entropy"),
    [(0.0, 1.242453324894), (2.0 / 3.0, 0.43494420225825936), (1.0, 0.0)],
)
def test_paper_quantum_memory_plot_data_match_analytic_curve(
    depolarization: float,
    expected_entropy: float,
) -> None:
    """Selected noisy SWAP plot points match analytic entropy and witness values."""
    history_rows = np.array([1, 1, 2, 2, 3, 3])
    history_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    contraction = 1.0 - depolarization
    pauli = np.zeros((6, 1, 4), dtype=np.float64)
    pauli[..., 0] = 1.0
    pauli[np.arange(6), 0, history_rows] = contraction * history_signs

    response_matrix = assemble_response_matrix(pauli, np.ones((6, 1), dtype=np.float64))
    spectrum = compute_spectrum(response_matrix, discarded_weight_threshold=None)
    expected_singular_values = np.array([
        np.sqrt(6.0),
        np.sqrt(2.0) * contraction,
        np.sqrt(2.0) * contraction,
        np.sqrt(2.0) * contraction,
    ])

    assert response_matrix.shape == (4, 6)
    np.testing.assert_allclose(spectrum["singular_values_full"], expected_singular_values, atol=1e-12)
    assert spectrum["entropy"] == pytest.approx(expected_entropy, abs=1e-12)

    average_fidelity = (
        sum(
            response_matrix[0, column] + sign * response_matrix[row, column]
            for column, (row, sign) in enumerate(zip(history_rows, history_signs, strict=True))
        )
        / 12.0
    )
    witness = 2.0 / 3.0 - average_fidelity
    assert witness == pytest.approx(depolarization / 2.0 - 1.0 / 3.0, abs=1e-12)


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
        response_matrix = assemble_response_matrix(p_sub, w_sub)
        entropies.append(float(compute_spectrum(response_matrix, discarded_weight_threshold=None)["entropy"]))
    assert entropies[-1] > entropies[0] * 1.05
