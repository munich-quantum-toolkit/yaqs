# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for deprecated ``protocol`` import shims."""

from __future__ import annotations

import importlib

import pytest

from mqt.yaqs.characterization.noise.trajectory_matching import run_trajectory_characterization
from mqt.yaqs.characterization.noise.trajectory_matching.results import (
    NoiseCharacterizationResult as TrajectoryResult,
)


def test_protocol_package_reexports_result() -> None:
    """Deprecated protocol package re-exports trajectory-matching symbols."""
    with pytest.warns(DeprecationWarning, match="protocol is deprecated"):
        protocol = importlib.import_module("mqt.yaqs.characterization.noise.protocol")
    shim_result = protocol.NoiseCharacterizationResult

    with pytest.warns(DeprecationWarning, match="protocol.results is deprecated"):
        results = importlib.import_module("mqt.yaqs.characterization.noise.protocol.results")
    result_alias = results.NoiseCharacterizationResult

    assert shim_result is result_alias
    assert shim_result is TrajectoryResult
    assert run_trajectory_characterization.__name__ == "run_trajectory_characterization"
