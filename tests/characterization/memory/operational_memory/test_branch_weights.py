# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for analytic branch-weight computation."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.operational_memory.branch_weights import (
    _compute_branch_weight_for_sequence,  # ruff:ignore[import-private-name] -- white-box parity test for analytic branch weights; no public equivalent
    compute_branch_weights,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet

_PSI0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
_Z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)


def test_dict_step_branch_weight_cut_measurement() -> None:
    """Structured cut_measurement steps contribute Born probabilities."""
    steps = [
        {"type": "cut_measurement", "psi_meas": _Z},
        {"type": "cut_preparation", "psi_prep": _Z},
    ]
    assert _compute_branch_weight_for_sequence(steps) == pytest.approx(1.0)


def test_cut_measurement_without_reset_projects_onto_measurement() -> None:
    """Two pre-cut cut_measurement steps use the measured state, not |0>, by default."""
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    steps = [
        {"type": "cut_measurement", "psi_meas": plus},
        {"type": "cut_measurement", "psi_meas": _Z},
    ]
    assert _compute_branch_weight_for_sequence(steps) == pytest.approx(0.25)


def test_complete_weights_include_history_and_future_outcomes() -> None:
    """Selected future outcomes make complete weights depend on both probe indices."""
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    probe_set = ProbeSet(
        cut=2,
        num_interventions=3,
        past_features=np.zeros((2, 2, 32), dtype=np.float32),
        future_features=np.zeros((2, 2, 32), dtype=np.float32),
        past_pairs=[[(_Z, _Z)], [(plus, plus)]],
        past_cut_meas=[_Z.copy(), _Z.copy()],
        future_prep_cut=[_Z.copy(), _Z.copy()],
        future_pairs=[[(_Z, _Z)], [(plus, plus)]],
    )
    np.testing.assert_allclose(
        compute_branch_weights(probe_set),
        np.array([[1.0, 0.5], [0.25, 0.125]], dtype=np.float64),
    )
