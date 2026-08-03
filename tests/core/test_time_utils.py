# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared exact evolution time grids."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.time_utils import exact_time_grid


def test_exact_time_grid_ends_at_duration() -> None:
    """Floating-point steps produce an exact, non-overshooting endpoint."""
    grid = exact_time_grid(0.3, 0.1)

    np.testing.assert_allclose(grid, np.array([0.0, 0.1, 0.2, 0.3]))
    assert grid[-1] == pytest.approx(0.3)


def test_exact_time_grid_validates_duration_and_step() -> None:
    """Invalid or non-divisible grids fail before backend evolution."""
    with pytest.raises(ValueError, match="integer multiple"):
        exact_time_grid(0.25, 0.1)
    with pytest.raises(ValueError, match="dt must be a positive"):
        exact_time_grid(0.1, 0.0)
    with pytest.raises(ValueError, match="duration must be positive"):
        exact_time_grid(0.0, 0.1)
    np.testing.assert_array_equal(exact_time_grid(0.0, 0.1, allow_zero=True), np.array([0.0]))
