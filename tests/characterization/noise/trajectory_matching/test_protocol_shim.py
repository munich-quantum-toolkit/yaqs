# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for deprecated ``protocol`` import shims."""

from __future__ import annotations

from mqt.yaqs.characterization.noise.protocol import NoiseCharacterizationResult
from mqt.yaqs.characterization.noise.protocol.results import NoiseCharacterizationResult as ResultAlias
from mqt.yaqs.characterization.noise.trajectory_matching import run_trajectory_characterization
from mqt.yaqs.characterization.noise.trajectory_matching.results import (
    NoiseCharacterizationResult as TrajectoryResult,
)


def test_protocol_package_reexports_result() -> None:
    """Deprecated protocol package re-exports trajectory-matching symbols."""
    assert NoiseCharacterizationResult is ResultAlias
    assert NoiseCharacterizationResult is TrajectoryResult
    assert run_trajectory_characterization.__name__ == "run_trajectory_characterization"
