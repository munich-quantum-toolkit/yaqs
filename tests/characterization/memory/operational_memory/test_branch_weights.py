# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for branch-weight parity (paper benchmark weighting)."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.exact import simulate_exact
from mqt.yaqs.characterization.memory.operational_memory.branch_weights import (
    compute_branch_weight,
    compute_trace_weights,
)
from mqt.yaqs.characterization.memory.shared.intervention_steps import compute_born_probability
from mqt.yaqs.characterization.memory.operational_memory.samples import sample_probes
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

_PSI0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
_Z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)


def _trace_final_weight(trace: dict[str, object]) -> float:
    """Read ``cumulative_weight_final`` from a traced rollout dict.

    Args:
        trace: Per-sequence trace payload from :func:`simulate_exact`.

    Returns:
        Final cumulative branch weight as a float.

    Raises:
        TypeError: If ``cumulative_weight_final`` is not numeric.
    """
    val = trace["cumulative_weight_final"]
    if not isinstance(val, (int, float)):
        msg = "cumulative_weight_final must be numeric"
        raise TypeError(msg)
    return float(val)


def _cumulative_weights_from_traces(
    traces: list[dict[str, object]],
    *,
    n_pasts: int,
    n_futures: int,
) -> np.ndarray:
    """Mirror experiments/_benchmark_memory.py cumulative_weight_final weighting.

    Args:
        traces: Flat list of per-(past, future) trace dicts from :func:`simulate_exact`.
        n_pasts: Number of past probe rows.
        n_futures: Number of future probe columns.

    Returns:
        Branch-weight matrix of shape ``(n_pasts, n_futures)``.
    """
    n_p, n_f = n_pasts, n_futures
    weights = np.zeros((n_p, n_f), dtype=np.float64)
    for ii in range(n_p):
        for jj in range(n_f):
            weights[ii, jj] = _trace_final_weight(traces[ii * n_f + jj])
    return weights


def test_compute_born_probability_identity_state() -> None:
    """Born probability for |0⟩ on |0⟩⟨0| is unity."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    assert compute_born_probability(rho, _Z) == pytest.approx(1.0)


def test_dict_step_branch_weight_cut_measurement() -> None:
    """Structured cut_measurement steps contribute Born probabilities."""
    steps = [
        {"type": "cut_measurement", "psi_meas": _Z},
        {"type": "cut_preparation", "psi_prep": _Z},
    ]
    assert compute_branch_weight(steps, cut=2) == pytest.approx(1.0)


def test_cut_measurement_without_reset_projects_onto_measurement() -> None:
    """Two pre-cut cut_measurement steps use the measured state, not |0>, by default."""
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    steps = [
        {"type": "cut_measurement", "psi_meas": plus},
        {"type": "cut_measurement", "psi_meas": _Z},
    ]
    assert compute_branch_weight(steps, cut=2) == pytest.approx(0.25)


_PSI0_L2 = np.zeros(4, dtype=np.complex128)
_PSI0_L2[0] = 1.0 + 0.0j


def test_trace_weights_match_cumulative_final_split_cut_unitary() -> None:
    """Paper metric path: cumulative_weight_final agrees with cut-truncated step_probs."""
    rng = np.random.default_rng(21)
    op = MPO.ising(length=2, J=0.5, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    probe_set = sample_probes(
        cut=3,
        num_interventions=5,
        n_pasts=4,
        n_futures=3,
        rng=rng,
        intervention_mode="split_cut_unitary",
        unitary_ensemble="haar",
    )
    _, weights_cut, traces = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=_PSI0_L2,
        parallel=False,
    )
    w_cumulative = _cumulative_weights_from_traces(
        traces,
        n_pasts=len(probe_set.past_pairs),
        n_futures=len(probe_set.future_pairs),
    )
    w_trace = compute_trace_weights(
        traces,
        n_pasts=len(probe_set.past_pairs),
        n_futures=len(probe_set.future_pairs),
        cut=probe_set.cut,
    )
    np.testing.assert_allclose(w_trace, weights_cut, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(w_cumulative, weights_cut, rtol=1e-10, atol=1e-12)


def test_exact_weights_positive_l2_quick_geometry() -> None:
    """L=2 paper quick geometry yields positive branch weights from exact rollouts."""
    rng = np.random.default_rng(44)
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    probe_set = sample_probes(
        cut=4,
        num_interventions=8,
        n_pasts=4,
        n_futures=3,
        rng=rng,
        intervention_mode="split_cut_unitary",
        unitary_ensemble="haar",
    )
    _, weights, traces = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=_PSI0_L2,
        parallel=False,
    )
    assert np.all(weights > 0.0)
    assert np.allclose(weights.std(axis=1), 0.0)
    assert all(float(t["cumulative_weight_final"]) > 0.0 for t in traces)
