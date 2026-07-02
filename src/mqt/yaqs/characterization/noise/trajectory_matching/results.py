# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Typed results for noise-parameter characterization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel


@dataclass(slots=True)
class NoiseCharacterizationResult:
    """Outcome of a gradient-free noise-parameter fit."""

    optimal_model: NoiseModel
    best_loss: float
    best_parameters: np.ndarray
    loss_history: list[float] = field(default_factory=list)
    ref_traj: np.ndarray | None = None
    fit_traj: np.ndarray | None = None
    times: np.ndarray | None = None

    def sqrt_loss_before(self) -> float:
        """Return ``sqrt(J)`` before optimization.

        Returns:
            Square root of the first loss value in ``loss_history``.

        Raises:
            ValueError: If ``loss_history`` is empty.
        """
        if not self.loss_history:
            msg = "loss_history is empty."
            raise ValueError(msg)
        return float(math.sqrt(self.loss_history[0]))

    def sqrt_loss_after(self) -> float:
        """Return ``sqrt(J)`` after optimization.

        Returns:
            Square root of ``best_loss``.
        """
        return float(math.sqrt(self.best_loss))

    def trajectory_rmse(self) -> float:
        """Root-mean-square mismatch between fitted and reference trajectories.

        Returns:
            RMSE over stored ``ref_traj`` and ``fit_traj``.

        Raises:
            ValueError: If trajectory arrays were not stored on the result or shapes
                do not match.
        """
        if self.ref_traj is None or self.fit_traj is None:
            msg = "ref_traj and fit_traj are required for trajectory_rmse()."
            raise ValueError(msg)
        ref = np.asarray(self.ref_traj, dtype=float)
        fit = np.asarray(self.fit_traj, dtype=float)
        if ref.shape != fit.shape:
            msg = f"ref_traj shape {ref.shape} does not match fit_traj shape {fit.shape}."
            raise ValueError(msg)
        residual = fit - ref
        return float(np.sqrt(np.mean(residual**2)))
