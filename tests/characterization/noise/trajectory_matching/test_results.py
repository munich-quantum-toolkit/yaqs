# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for trajectory-matching result helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.noise.trajectory_matching.results import NoiseCharacterizationResult
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel


def _minimal_result(
    *,
    best_loss: float = 0.01,
    best_parameters: np.ndarray | None = None,
    loss_history: list[float] | None = None,
    ref_traj: np.ndarray | None = None,
    fit_traj: np.ndarray | None = None,
) -> NoiseCharacterizationResult:
    model = CompactNoiseModel([{"name": "pauli_y", "sites": [0], "strength": 0.1}])
    return NoiseCharacterizationResult(
        optimal_model=model,
        best_loss=best_loss,
        best_parameters=np.array([0.1]) if best_parameters is None else best_parameters,
        loss_history=[1.0, 0.01] if loss_history is None else loss_history,
        ref_traj=np.zeros((1, 3)) if ref_traj is None else ref_traj,
        fit_traj=np.zeros((1, 3)) if fit_traj is None else fit_traj,
    )


def test_sqrt_loss_helpers() -> None:
    """Result exposes square-root loss before and after optimization."""
    result = _minimal_result()
    assert result.sqrt_loss_before() == pytest.approx(1.0)
    assert result.sqrt_loss_after() == pytest.approx(0.1)


def test_trajectory_rmse_zero_for_identical_trajs() -> None:
    """RMSE vanishes when fitted and reference trajectories match."""
    traj = np.linspace(0.0, 1.0, 4)[None, :]
    result = _minimal_result(ref_traj=traj, fit_traj=traj.copy())
    assert result.trajectory_rmse() == pytest.approx(0.0)


def test_rate_table_with_reference() -> None:
    """Rate table reports relative errors when reference strengths are given."""
    result = _minimal_result(best_parameters=np.array([0.09]))
    rows = result.rate_table(reference_strengths=[0.1])
    assert rows[0]["learned"] == pytest.approx(0.09)
    assert rows[0]["rel_error"] == pytest.approx(0.1)
