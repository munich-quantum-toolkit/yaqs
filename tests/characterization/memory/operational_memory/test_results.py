# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for CharacterizationResult."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.operational_memory.results import (
    merge_cut_results,
    pack_result,
    parse_cut_result,
)


def test_rank_equals_exp_entropy() -> None:
    """``rank()`` equals ``exp(entropy())`` for a single-cut result."""
    ham = Hamiltonian.ising(length=1, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    result = MemoryCharacterizer(parallel=False, show_progress=False).characterize(
        ham,
        params,
        cut=1,
        k=1,
        n_pasts=6,
        n_futures=6,
    )
    sv = result.entropy()
    r = result.rank()
    assert r == pytest.approx(math.exp(sv), rel=1e-9, abs=1e-9)


def test_probes_export_arrays() -> None:
    """characterize() stores probe arrays retrievable via result.probes(cut)."""
    ham = Hamiltonian.ising(length=1, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    result = MemoryCharacterizer(parallel=False, show_progress=False).characterize(
        ham,
        params,
        cut=1,
        k=1,
        n_pasts=4,
        n_futures=3,
        rng=np.random.default_rng(0),
    )
    probes = result.probes()
    assert probes["cut"] == 1
    assert probes["k"] == 1
    assert probes["past_features"].shape[0] == 4
    assert probes["future_features"].shape[0] == 3


def test_parse_cut_result_requires_memory_matrix() -> None:
    """parse_cut_result rejects incomplete probe dicts."""
    with pytest.raises(ValueError, match="missing memory_matrix"):
        parse_cut_result({"entropy": 0.0}, cut=1)


def test_merge_cut_results_multi_cut_summary() -> None:
    """merge_cut_results builds a multi-cut CharacterizationResult."""
    parts = {
        1: pack_result({"entropy": 0.5, "rank": 1.6, "singular_values": np.array([1.0]), "memory_matrix": np.eye(2)}, cut=1),
        2: pack_result({"entropy": 0.8, "rank": 2.2, "singular_values": np.array([1.0, 0.5]), "memory_matrix": np.eye(2)}, cut=2),
    }
    merged = merge_cut_results(parts)
    assert merged.entropy(1) == pytest.approx(0.5)
    assert merged.entropy(2) == pytest.approx(0.8)
    summary = merged.summary()
    assert "cut  S_V" in summary
    assert "1" in summary and "2" in summary


def test_entropy_requires_cut_when_multiple_stored() -> None:
    """Accessors require an explicit cut for multi-cut results."""
    merged = merge_cut_results(
        {
            1: pack_result(
                {"entropy": 0.1, "rank": 1.1, "singular_values": np.array([1.0]), "memory_matrix": np.eye(2)},
                cut=1,
            ),
            2: pack_result(
                {"entropy": 0.2, "rank": 1.2, "singular_values": np.array([1.0]), "memory_matrix": np.eye(2)},
                cut=2,
            ),
        }
    )
    with pytest.raises(ValueError, match="cut is required"):
        merged.entropy()


def test_characterize_multiple_cuts_smoke() -> None:
    """MemoryCharacterizer supports cuts='all' for Hamiltonian characterize."""
    ham = Hamiltonian.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    result = MemoryCharacterizer(parallel=False, show_progress=False).characterize(
        ham,
        params,
        k=3,
        cuts=[1, 2, 3],
        n_pasts=4,
        n_futures=4,
        rng=np.random.default_rng(0),
    )
    for cut in (1, 2, 3):
        assert result.entropy(cut) >= 0.0
        assert result.rank(cut) >= 1.0

