# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deprecated import path; use :mod:`mqt.yaqs.characterization.noise.trajectory_matching`."""

from mqt.yaqs.characterization.noise.trajectory_matching import (
    NoiseCharacterizationResult,
    run_trajectory_characterization,
)

__all__ = ["NoiseCharacterizationResult", "run_trajectory_characterization"]
