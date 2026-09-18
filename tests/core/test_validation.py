# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core._validation import validate_real


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (np.float64(-2.5), -2.5),
        (complex(0.0, 5e-13), 0.0),
        (complex(0.0, -5e-13), 0.0),
        (complex(1e6, 5e-5), 1e6),
        (complex(1e6, -5e-5), 1e6),
    ],
)
def test_validate_real_accepts_finite_values_with_scale_dependent_roundoff(value: object, expected: float) -> None:
    """Real values and symmetric numerical imaginary residuals are accepted."""
    assert validate_real(value, name="quantity") == expected


@pytest.mark.parametrize("imaginary", [2e-12, -2e-12, 2e-4, -2e-4])
def test_validate_real_rejects_imaginary_residual_outside_tolerance(imaginary: float) -> None:
    """Both signs of an excessive imaginary residual are rejected."""
    real = 0.0 if abs(imaginary) < 1e-10 else 1e6
    with pytest.raises(ValueError, match="quantity must be real"):
        validate_real(complex(real, imaginary), name="quantity")


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        complex(1.0, float("nan")),
        complex(1.0, float("inf")),
    ],
)
def test_validate_real_rejects_non_finite_components(value: complex | float) -> None:
    """Non-finite real and imaginary components fail explicitly."""
    with pytest.raises(ValueError, match="quantity must be finite"):
        validate_real(value, name="quantity")


@pytest.mark.parametrize("value", [True, "1", None, object()])
def test_validate_real_rejects_non_numeric_scalars(value: object) -> None:
    """Boolean and non-numeric inputs are not interpreted as real values."""
    with pytest.raises(TypeError, match="quantity must be a real numeric scalar"):
        validate_real(value, name="quantity")
