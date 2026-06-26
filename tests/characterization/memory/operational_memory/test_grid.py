# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for split-cut probe grid assembly."""

from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.memory.operational_memory.grid import (
    assemble_probe_grid,
    assemble_probe_sequence,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import sample_probes


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
