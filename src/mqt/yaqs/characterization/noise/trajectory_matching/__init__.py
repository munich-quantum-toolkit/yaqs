# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Trajectory-matching protocol and helpers for Markovian noise fitting.

Submodules:

- :mod:`.reference` — reference trajectories, loss assembly, simulation helpers
- :mod:`.run` — :func:`run_trajectory_characterization`
- :mod:`.results` — :class:`NoiseCharacterizationResult`

User code should use :class:`~mqt.yaqs.noise_characterizer.NoiseCharacterizer` only.
"""

from .results import NoiseCharacterizationResult
from .run import run_trajectory_characterization

__all__ = ["NoiseCharacterizationResult", "run_trajectory_characterization"]
