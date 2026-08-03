# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared construction of exact, non-overshooting evolution time grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def exact_time_grid(duration: float, dt: float, *, allow_zero: bool = False) -> NDArray[np.float64]:
    """Build a time grid ending exactly at ``duration``.

    Args:
        duration: Requested evolution duration.
        dt: Fixed integration step size.
        allow_zero: Whether a zero-duration grid ``[0.0]`` is valid.

    Returns:
        A one-dimensional grid from zero through ``duration``.

    Raises:
        ValueError: If ``dt`` is not positive, ``duration`` is invalid, or the
            duration is not an integer multiple of ``dt``.
    """
    dt_float = float(dt)
    duration_float = float(duration)
    if not np.isfinite(dt_float) or dt_float <= 0:
        msg = f"dt must be a positive finite value, got {dt!r}."
        raise ValueError(msg)
    if not np.isfinite(duration_float) or duration_float < 0:
        msg = f"duration must be a non-negative finite value, got {duration!r}."
        raise ValueError(msg)
    if abs(duration_float) < 1e-15:
        if allow_zero:
            return np.array([0.0], dtype=np.float64)
        msg = "duration must be positive."
        raise ValueError(msg)

    num_steps = round(duration_float / dt_float)
    tolerance = 1e-9 * max(1.0, duration_float)
    if num_steps < 1 or abs(num_steps * dt_float - duration_float) > tolerance:
        msg = f"duration={duration_float} must be a positive integer multiple of dt={dt_float}."
        raise ValueError(msg)
    return np.linspace(0.0, duration_float, num_steps + 1, dtype=np.float64)
