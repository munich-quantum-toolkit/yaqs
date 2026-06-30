# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for probe sampling ensembles used in paper benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.operational_memory.samples import (
    enumerate_clifford_unitaries,
    resolve_unitary_sampler,
    sample_probes,
)


def test_enumerate_clifford_unitaries_returns_24_unique() -> None:
    """Single-qubit Clifford group has 24 elements."""
    cliffords = enumerate_clifford_unitaries()
    assert len(cliffords) == 24
    flat = [tuple(np.round(c.reshape(-1), 8)) for c in cliffords]
    assert len(set(flat)) == 24


def test_resolve_unitary_sampler_clifford_draws_from_clifford_group() -> None:
    """Clifford ensemble only samples enumerated Cliffords."""
    rng = np.random.default_rng(0)
    sampler = resolve_unitary_sampler("clifford")
    cliffords = enumerate_clifford_unitaries()
    flat_cliffords = {tuple(np.round(c.reshape(-1), 8)) for c in cliffords}
    for _ in range(20):
        u = sampler(rng)
        key = tuple(np.round(u.reshape(-1), 8))
        assert key in flat_cliffords


def test_resolve_unitary_sampler_rejects_unknown() -> None:
    """Unknown ensemble names raise ValueError."""
    with pytest.raises(ValueError, match="unitary_ensemble"):
        resolve_unitary_sampler("so3")


def test_sample_probes_measure_prepare_mode() -> None:
    """Measure-prepare mode produces MP tuple steps on legs."""
    rng = np.random.default_rng(1)
    probe_set = sample_probes(
        cut=2,
        num_interventions=3,
        n_pasts=3,
        n_futures=2,
        rng=rng,
        intervention_mode="measure_prepare",
    )
    seq = probe_set.past_pairs[0] + [(probe_set.past_cut_meas[0], probe_set.future_prep_cut[0])]
    assert all(isinstance(step, tuple) for step in seq)


def test_sample_probes_cut_validation() -> None:
    """Invalid cut indices are rejected."""
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="cut must satisfy"):
        sample_probes(cut=0, num_interventions=2, n_pasts=2, n_futures=2, rng=rng)


def test_sample_random_clifford_unitary_returns_copy() -> None:
    """Mutating a sampled Clifford matrix must not affect later draws."""
    clifford_sampler = resolve_unitary_sampler("clifford")
    a = clifford_sampler(np.random.default_rng(7))
    b = clifford_sampler(np.random.default_rng(7))
    np.testing.assert_allclose(a, b, atol=1e-12)
    a[0, 0] = 999.0 + 0.0j
    c = clifford_sampler(np.random.default_rng(7))
    np.testing.assert_allclose(b, c, atol=1e-12)


def test_sample_probes_measure_prepare_ignores_unitary_ensemble() -> None:
    """Measure-prepare mode does not validate unitary_ensemble."""
    rng = np.random.default_rng(3)
    probe_set = sample_probes(
        cut=1,
        num_interventions=2,
        n_pasts=2,
        n_futures=2,
        rng=rng,
        intervention_mode="measure_prepare",
        unitary_ensemble="not-a-real-ensemble",
    )
    assert probe_set.num_interventions == 2
