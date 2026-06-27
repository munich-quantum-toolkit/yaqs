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
    assemble_probe_grid,
    assemble_probe_sequence,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet, sample_probes


def test_assemble_probe_sequence_length_and_cut_step() -> None:
    """Each grid entry has length k with the causal break at index cut-1."""
    rng = np.random.default_rng(7)
    cut, k = 3, 5
    probe_set = sample_probes(cut=cut, k=k, n_pasts=4, n_futures=3, rng=rng)
    seq = assemble_probe_sequence(probe_set, i=1, j=2)
    assert len(seq) == k
    assert seq[cut - 1] == (probe_set.past_cut_meas[1], probe_set.future_prep_cut[2])


def test_assemble_probe_grid_size() -> None:
    """Flat grid has n_pasts * n_futures sequences, each of length k."""
    rng = np.random.default_rng(8)
    n_pasts, n_futures, cut, k = 5, 4, 2, 4
    probe_set = sample_probes(cut=cut, k=k, n_pasts=n_pasts, n_futures=n_futures, rng=rng)
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
        k=3,
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
        k=3,
        past_features=np.zeros((1, 3, 32), dtype=np.float32),
        future_features=np.zeros((1, 1, 32), dtype=np.float32),
        past_pairs=[[{"type": "unitary", "U": np.eye(2, dtype=np.complex128)}]],
        past_cut_meas=[z],
        future_prep_cut=[z],
        future_pairs=[[]],
    )
    with pytest.raises(ValueError, match="past_pairs\\[0\\] length 1 != cut-1=2"):
        assemble_probe_sequence(probe_set, i=0, j=0)
