# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for noise optimization package exports."""

from mqt.yaqs.characterization.noise import NoiseCharacterizationResult, optimization


def test_optimization_exports_result_without_runner_shortcut() -> None:
    """The result alias remains available without an execution helper export."""
    assert optimization.__all__ == ["NoiseCharacterizationResult"]
    assert optimization.NoiseCharacterizationResult is NoiseCharacterizationResult
    assert not hasattr(optimization, "run_optimization_characterization")
