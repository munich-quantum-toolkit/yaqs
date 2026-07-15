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
    _compute_branch_weight_for_sequence,  # noqa: PLC2701 -- white-box parity test for analytic branch weights; no public equivalent
)

_PSI0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
_Z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)


def test_dict_step_branch_weight_cut_measurement() -> None:
    """Structured cut_measurement steps contribute Born probabilities."""
    steps = [
        {"type": "cut_measurement", "psi_meas": _Z},
        {"type": "cut_preparation", "psi_prep": _Z},
    ]
    assert _compute_branch_weight_for_sequence(steps, cut=2) == pytest.approx(1.0)


def test_cut_measurement_without_reset_projects_onto_measurement() -> None:
    """Two pre-cut cut_measurement steps use the measured state, not |0>, by default."""
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    steps = [
        {"type": "cut_measurement", "psi_meas": plus},
        {"type": "cut_measurement", "psi_meas": _Z},
    ]
    assert _compute_branch_weight_for_sequence(steps, cut=2) == pytest.approx(0.25)
