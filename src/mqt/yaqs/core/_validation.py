# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Private validation helpers shared by public YAQS entry points."""

from __future__ import annotations

from numbers import Integral


def validate_integer(value: object, *, name: str, minimum: int | None = None) -> int:
    """Validate and normalize an integer with a lower bound.

    Args:
        value: Candidate integer. Python and NumPy integer types are accepted.
        name: Parameter name used in error messages.
        minimum: Smallest accepted value, or ``None`` for no lower bound.

    Returns:
        The value normalized to a Python :class:`int`.

    Raises:
        TypeError: If ``value`` is a boolean or not an integer.
        ValueError: If ``minimum`` is set and ``value`` is smaller.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        msg = f"{name} must be an integer, got {type(value).__name__}."
        raise TypeError(msg)
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        msg = f"{name} must be >= {minimum}, got {normalized}."
        raise ValueError(msg)
    return normalized
