# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for split-cut probe grid assembly."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.operational_memory.grid import (
    assemble_delayed_probe_grid,
    assemble_delayed_probe_sequence,
    assemble_probe_grid,
    assemble_probe_sequence,
    compute_delayed_length,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet, sample_probes


def test_assemble_probe_num_interventions_and_cut_step() -> None:
    """Each grid entry has length k with the causal break at index cut-1."""
    rng = np.random.default_rng(7)
    cut, k = 3, 5
    probe_set = sample_probes(cut=cut, num_interventions=k, n_pasts=4, n_futures=3, rng=rng)
    seq = assemble_probe_sequence(probe_set, i=1, j=2)
    assert len(seq) == k
    assert seq[cut - 1] == (probe_set.past_cut_meas[1], probe_set.future_prep_cut[2])


def test_assemble_probe_grid_size() -> None:
    """Flat grid has n_pasts * n_futures sequences, each of length k."""
    rng = np.random.default_rng(8)
    n_pasts, n_futures, cut, k = 5, 4, 2, 4
    probe_set = sample_probes(cut=cut, num_interventions=k, n_pasts=n_pasts, n_futures=n_futures, rng=rng)
    all_pairs, n_p, n_f = assemble_probe_grid(probe_set)
    assert n_p == n_pasts
    assert n_f == n_futures
    assert len(all_pairs) == n_pasts * n_futures
    assert all(len(seq) == k for seq in all_pairs)


def test_assemble_probe_sequence_rejects_inconsistent_probe_set() -> None:
    """Direct callers get explicit branch-length validation."""
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    probe_set = ProbeSet(
        cut=1,
        num_interventions=3,
        past_features=np.zeros((1, 1, 32), dtype=np.float32),
        future_features=np.zeros((1, 3, 32), dtype=np.float32),
        past_pairs=[[]],
        past_cut_meas=[z],
        future_prep_cut=[z],
        future_pairs=[[{"type": "unitary", "U": np.eye(2, dtype=np.complex128)}]],
    )
    with pytest.raises(ValueError, match="future_pairs\\[0\\] length 1 != k-cut=2"):
        assemble_probe_sequence(probe_set, i=0, j=0)


def test_assemble_probe_sequence_rejects_short_past_branch() -> None:
    """Undersized past branch lists are rejected before indexed access."""
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    probe_set = ProbeSet(
        cut=3,
        num_interventions=3,
        past_features=np.zeros((1, 3, 32), dtype=np.float32),
        future_features=np.zeros((1, 1, 32), dtype=np.float32),
        past_pairs=[[{"type": "unitary", "U": np.eye(2, dtype=np.complex128)}]],
        past_cut_meas=[z],
        future_prep_cut=[z],
        future_pairs=[[]],
    )
    with pytest.raises(ValueError, match="past_pairs\\[0\\] length 1 != cut-1=2"):
        assemble_probe_sequence(probe_set, i=0, j=0)


def test_assemble_probe_sequence_rejects_mismatched_cut_arrays() -> None:
    """Malformed cut-branch arrays raise ValueError instead of IndexError."""
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    u = np.eye(2, dtype=np.complex128)
    probe_set = ProbeSet(
        cut=2,
        num_interventions=2,
        past_features=np.zeros((1, 1, 32), dtype=np.float32),
        future_features=np.zeros((1, 1, 32), dtype=np.float32),
        past_pairs=[[{"type": "unitary", "U": u}]],
        past_cut_meas=[z],
        future_prep_cut=[z, z],
        future_pairs=[[]],
    )
    with pytest.raises(ValueError, match="future_prep_cut length 2 != n_futures=1"):
        assemble_probe_sequence(probe_set, i=0, j=0)


def test_compute_delayed_length_rejects_negative() -> None:
    """Negative reset delay is rejected at assembly time."""
    with pytest.raises(ValueError, match="delay must be >= 0"):
        compute_delayed_length(num_interventions=5, delay=-1)


def test_assemble_delayed_probe_sequence_delay_zero_matches_standard() -> None:
    """delay=0 delegates to the standard split-cut assembler."""
    rng = np.random.default_rng(1)
    probe_set = sample_probes(cut=2, num_interventions=4, n_pasts=3, n_futures=2, rng=rng)
    assert assemble_delayed_probe_sequence(probe_set, 0, 1, delay=0) == assemble_probe_sequence(
        probe_set,
        0,
        1,
    )


def test_assemble_delayed_probe_grid_inserts_reset_slots() -> None:
    """delay>0 lengthens sequences by delay+1 and adds (|0>,|0>) bridge slots."""
    rng = np.random.default_rng(9)
    cut, k, delay = 3, 5, 2
    probe_set = sample_probes(cut=cut, num_interventions=k, n_pasts=2, n_futures=2, rng=rng)
    delayed_pairs, _, _ = assemble_delayed_probe_grid(probe_set, delay=delay)
    expected_len = compute_delayed_length(num_interventions=k, delay=delay)
    assert expected_len == k + delay + 1
    assert all(len(seq) == expected_len for seq in delayed_pairs)
    z0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    for seq in delayed_pairs:
        reset_pairs = sum(
            1 for step in seq if isinstance(step, tuple) and np.allclose(step[0], z0) and np.allclose(step[1], z0)
        )
        assert reset_pairs == delay


def test_assemble_delayed_probe_sequence_rejects_mismatched_cut_arrays() -> None:
    """Delayed assembly validates cut-branch arrays before indexed access."""
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    u = np.eye(2, dtype=np.complex128)
    probe_set = ProbeSet(
        cut=2,
        num_interventions=4,
        past_features=np.zeros((1, 1, 32), dtype=np.float32),
        future_features=np.zeros((1, 3, 32), dtype=np.float32),
        past_pairs=[[{"type": "unitary", "U": u}]],
        past_cut_meas=[z],
        future_prep_cut=[z, z],
        future_pairs=[[{"type": "unitary", "U": u}, {"type": "unitary", "U": u}]],
    )
    with pytest.raises(ValueError, match="future_prep_cut length 2 != n_futures=1"):
        assemble_delayed_probe_sequence(probe_set, i=0, j=0, delay=1)
