# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Private validation helpers shared by public YAQS entry points."""

from __future__ import annotations

import math
from numbers import Complex, Integral

_REAL_RTOL = 1e-10
_REAL_ATOL = 1e-12


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


def validate_real(
    value: object,
    *,
    name: str,
    rtol: float = _REAL_RTOL,
    atol: float = _REAL_ATOL,
) -> float:
    """Validate and normalize a finite scalar that must be numerically real.

    A complex value is accepted only when its imaginary residual satisfies
    ``abs(imag) <= atol + rtol * abs(real)``. The default tolerances match the
    Hermiticity policy for user-defined observables.

    Args:
        value: Candidate real or complex numeric scalar.
        name: Quantity name used in error messages.
        rtol: Relative tolerance for the imaginary residual.
        atol: Absolute tolerance for the imaginary residual.

    Returns:
        The finite real component as a Python :class:`float`.

    Raises:
        TypeError: If ``value`` is a boolean or not a numeric scalar.
        ValueError: If either tolerance is negative or non-finite, either value
            component is non-finite, or the imaginary residual exceeds the
            tolerance.
    """
    if not math.isfinite(rtol) or rtol < 0:
        msg = f"rtol must be finite and non-negative, got {rtol!r}."
        raise ValueError(msg)
    if not math.isfinite(atol) or atol < 0:
        msg = f"atol must be finite and non-negative, got {atol!r}."
        raise ValueError(msg)
    if isinstance(value, bool) or not isinstance(value, Complex):
        msg = f"{name} must be a real numeric scalar, got {type(value).__name__}."
        raise TypeError(msg)
    normalized = complex(value)
    real = float(normalized.real)
    imaginary = float(normalized.imag)
    if not math.isfinite(real) or not math.isfinite(imaginary):
        msg = f"{name} must be finite, got {normalized!r}."
        raise ValueError(msg)
    tolerance = atol + rtol * abs(real)
    if abs(imaginary) > tolerance:
        msg = f"{name} must be real within atol={atol} and rtol={rtol}; got real={real} and imag={imaginary}."
        raise ValueError(msg)
    return real
