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
from mqt.yaqs.characterization.memory.operational_memory.memory_matrix import compute_spectrum
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
        1: pack_result(
            {"entropy": 0.5, "rank": 1.6, "singular_values": np.array([1.0]), "memory_matrix": np.eye(2)}, cut=1
        ),
        2: pack_result(
            {"entropy": 0.8, "rank": 2.2, "singular_values": np.array([1.0, 0.5]), "memory_matrix": np.eye(2)}, cut=2
        ),
    }
    merged = merge_cut_results(parts)
    assert merged.entropy(1) == pytest.approx(0.5)
    assert merged.entropy(2) == pytest.approx(0.8)
    summary = merged.summary()
    assert "cut  S_V" in summary
    assert "1" in summary
    assert "2" in summary


def test_entropy_requires_cut_when_multiple_stored() -> None:
    """Accessors require an explicit cut for multi-cut results."""
    merged = merge_cut_results({
        1: pack_result(
            {"entropy": 0.1, "rank": 1.1, "singular_values": np.array([1.0]), "memory_matrix": np.eye(2)},
            cut=1,
        ),
        2: pack_result(
            {"entropy": 0.2, "rank": 1.2, "singular_values": np.array([1.0]), "memory_matrix": np.eye(2)},
            cut=2,
        ),
    })
    with pytest.raises(ValueError, match="cut is required"):
        merged.entropy()


def test_resolve_cut_missing_raises() -> None:
    """Explicit cut values missing from by_cut raise ValueError."""
    merged = merge_cut_results({
        1: pack_result(
            {"entropy": 0.1, "rank": 1.1, "singular_values": np.array([1.0]), "memory_matrix": np.eye(2)},
            cut=1,
        ),
    })
    with pytest.raises(ValueError, match="cut 2 is not stored"):
        merged.entropy(2)


def test_parse_cut_result_stores_truncated_singular_values() -> None:
    """parse_cut_result exposes the truncated spectrum used for entropy/rank."""
    full = np.array([10.0, 5.0, 1e-6, 1e-8], dtype=np.float64)
    m = np.diag(full)
    out = compute_spectrum(m, discarded_weight_threshold=1e-4)
    packed = parse_cut_result(
        {
            "entropy": out["entropy"],
            "rank": out["rank"],
            "singular_values": out["singular_values"],
            "singular_values_full": out["singular_values_full"],
            "memory_matrix": m,
        },
        cut=1,
    )
    assert packed.singular_values.size == out["singular_values"].size
    assert packed.singular_values.size < full.size
    np.testing.assert_allclose(packed.singular_values, out["singular_values"])


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


def test_probes_raises_when_not_recorded() -> None:
    """probes() requires probe_set data on the stored cut."""
    packed = pack_result(
        {"entropy": 0.0, "rank": 1.0, "singular_values": np.array([1.0]), "memory_matrix": np.eye(2)},
        cut=1,
    )
    with pytest.raises(ValueError, match="No probe data recorded for cut=1"):
        packed.probes()


def test_summary_single_cut_format() -> None:
    """Single-cut results use the compact one-line summary."""
    packed = pack_result(
        {"entropy": 0.5, "rank": 1.6, "singular_values": np.array([1.0]), "memory_matrix": np.eye(2)},
        cut=2,
    )
    assert packed.summary() == "cut=2: S_V=0.5000, R=1.600"


def test_merge_cut_results_rejects_multi_cut_parts() -> None:
    """merge_cut_results expects each partial result to hold one cut."""
    multi = merge_cut_results({
        1: pack_result(
            {"entropy": 0.1, "rank": 1.1, "singular_values": np.array([1.0]), "memory_matrix": np.eye(2)},
            cut=1,
        ),
        2: pack_result(
            {"entropy": 0.2, "rank": 1.2, "singular_values": np.array([1.0]), "memory_matrix": np.eye(2)},
            cut=2,
        ),
    })
    with pytest.raises(ValueError, match="exactly one cut"):
        merge_cut_results({1: multi})
