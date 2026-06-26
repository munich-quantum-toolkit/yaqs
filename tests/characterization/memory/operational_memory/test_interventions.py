# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for user-facing intervention style encoding."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.operational_memory.interventions import (
    DEFAULT_INTERVENTION_STYLE,
    encode_sequence,
    encode_slot,
    expand_sequence,
    map_probe_kwargs,
    normalize_style,
    sample_train_sequence,
)


def test_default_intervention_style_is_haar() -> None:
    """Paper V-matrix standard is the default intervention style."""
    assert DEFAULT_INTERVENTION_STYLE == "haar"


def test_normalize_style_accepts_paper_presets() -> None:
    """All documented intervention styles normalize cleanly."""
    assert normalize_style("haar") == "haar"
    assert normalize_style("  Clifford ") == "clifford"
    assert normalize_style("measure_prepare") == "measure_prepare"


def test_normalize_style_rejects_unknown() -> None:
    """Unsupported style strings raise ValueError."""
    with pytest.raises(ValueError, match="style must be"):
        normalize_style("random_unitary")


def test_map_probe_kwargs_matches_paper_experiments() -> None:
    """Haar style maps to unitary_break_mp + haar (experiments/ default)."""
    assert map_probe_kwargs("haar") == {
        "intervention_mode": "unitary_break_mp",
        "unitary_ensemble": "haar",
    }
    assert map_probe_kwargs("clifford") == {
        "intervention_mode": "unitary_break_mp",
        "unitary_ensemble": "clifford",
    }
    assert map_probe_kwargs("measure_prepare") == {
        "intervention_mode": "measure_prepare",
        "unitary_ensemble": "haar",
    }


def test_expand_sequence_broadcasts_scalar_style() -> None:
    """Scalar style expands to k identical slots."""
    rng = np.random.default_rng(0)
    slots = expand_sequence("haar", k=4, _rng=rng)
    assert slots == ["haar", "haar", "haar", "haar"]


def test_expand_sequence_length_mismatch_raises() -> None:
    """Explicit slot lists must match k."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="intervention sequence length"):
        expand_sequence(["haar", "clifford"], k=3, _rng=rng)


def test_encode_slot_unitary_dict() -> None:
    """Explicit unitary dict slots encode to Choi features."""
    rng = np.random.default_rng(1)
    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    step, feat = encode_slot({"unitary": u}, rng)
    assert step["type"] == "unitary"
    np.testing.assert_allclose(step["U"], u)
    assert feat.shape == (32,)


def test_encode_slot_rejects_non_unitary_dict() -> None:
    """Non-unitary 2x2 matrices in dict slots raise ValueError."""
    rng = np.random.default_rng(0)
    bad = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.complex128)
    with pytest.raises(ValueError, match="unitary"):
        encode_slot({"unitary": bad}, rng)


def test_encode_sequence_haar_shape() -> None:
    """Haar-encoded sequences return k Choi rows."""
    rng = np.random.default_rng(2)
    steps, choi = encode_sequence("haar", k=3, rng=rng)
    assert len(steps) == 3
    assert choi.shape == (3, 32)
    assert all(isinstance(s, dict) and s.get("type") == "unitary" for s in steps)


def test_sample_train_sequence_measure_prepare() -> None:
    """Measure-prepare style yields MP pairs and Choi rows."""
    rng = np.random.default_rng(3)
    steps, choi = sample_train_sequence(2, "measure_prepare", rng)
    assert len(steps) == 2
    assert choi.shape == (2, 32)
    for step in steps:
        assert isinstance(step, tuple)
        assert len(step) == 2
