# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Typed results for noise-parameter characterization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel


@dataclass(slots=True)
class NoiseCharacterizationResult:
    """Outcome of a gradient-free noise-parameter fit."""

    optimal_model: CompactNoiseModel
    best_loss: float
    best_parameters: np.ndarray
    loss_history: list[float] = field(default_factory=list)
    parameter_history: list[np.ndarray] = field(default_factory=list)
