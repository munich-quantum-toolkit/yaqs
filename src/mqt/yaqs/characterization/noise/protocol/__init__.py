# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deprecated import path; use :mod:`mqt.yaqs.characterization.noise.trajectory_matching`."""

import warnings

from mqt.yaqs.characterization.noise.trajectory_matching import (
    NoiseCharacterizationResult,
    run_trajectory_characterization,
)

warnings.warn(  # noqa: RUF067 -- deprecated shim must warn at import time
    "mqt.yaqs.characterization.noise.protocol is deprecated; "
    "use mqt.yaqs.characterization.noise.trajectory_matching instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["NoiseCharacterizationResult", "run_trajectory_characterization"]
