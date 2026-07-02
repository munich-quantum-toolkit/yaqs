# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deprecated import path; use :mod:`mqt.yaqs.characterization.noise.trajectory_matching.results`."""

import warnings

from mqt.yaqs.characterization.noise.trajectory_matching.results import NoiseCharacterizationResult

warnings.warn(
    "mqt.yaqs.characterization.noise.protocol.results is deprecated; "
    "use mqt.yaqs.characterization.noise.trajectory_matching.results instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["NoiseCharacterizationResult"]
