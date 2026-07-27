# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for gate metadata and single-pass parameter resolution."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.optimization.parameterized_circuit import ParameterizedCircuit, ParameterizedGate


def test_gate_metadata_defaults_and_valid_values() -> None:
    """Gate-local noise metadata has compatible defaults and accepts stable identifiers."""
    default_gate = ParameterizedGate("h", (0,))
    custom_gate = ParameterizedGate(
        "x",
        (0,),
        logical_gate_id=3,
        native_gate_id="native:7",
        noise_enabled=False,
    )

    ParameterizedCircuit(1, [default_gate, custom_gate])

    assert default_gate.logical_gate_id is None
    assert default_gate.native_gate_id is None
    assert default_gate.noise_enabled is True
    assert custom_gate.logical_gate_id == 3
    assert custom_gate.native_gate_id == "native:7"
    assert custom_gate.noise_enabled is False


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("logical_gate_id", True, TypeError, "logical_gate_id"),
        ("logical_gate_id", -1, ValueError, "logical_gate_id must be nonnegative"),
        ("logical_gate_id", "", ValueError, "logical_gate_id must be a nonempty string"),
        ("logical_gate_id", "   ", ValueError, "logical_gate_id must be a nonempty string"),
        ("native_gate_id", 1.5, TypeError, "native_gate_id"),
        ("native_gate_id", -2, ValueError, "native_gate_id must be nonnegative"),
        ("native_gate_id", "", ValueError, "native_gate_id must be a nonempty string"),
        ("native_gate_id", " native ", ValueError, "without surrounding whitespace"),
        ("noise_enabled", 1, TypeError, "noise_enabled must be a bool"),
    ],
)
def test_gate_metadata_validation(
    field: str,
    value: object,
    exception: type[TypeError | ValueError],
    message: str,
) -> None:
    """Circuit construction rejects ambiguous or malformed gate metadata."""
    kwargs: dict[str, Any] = {field: value}
    gate = ParameterizedGate("h", (0,), **kwargs)

    with pytest.raises(exception, match=message):
        ParameterizedCircuit(1, [gate])


def test_gate_matrix_and_angle_resolves_affine_data_map_once() -> None:
    """The affine and data-dependent angle is evaluated once and reused for the matrix."""
    calls = 0

    def data_map(x: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return float(x[0])

    gate = ParameterizedGate(
        "rx",
        (0,),
        param_index=0,
        angle_scale=1.5,
        angle_offset=-0.2,
        data_map=data_map,
    )
    circuit = ParameterizedCircuit(1, [gate])
    theta = np.array([0.4], dtype=np.float64)
    x = np.array([0.3], dtype=np.float64)
    expected_angle = 1.5 * theta[0] - 0.2 + x[0]

    matrix, sites, resolved_angle = circuit.gate_matrix_and_angle(gate, theta, x)

    assert calls == 1
    assert resolved_angle == pytest.approx(expected_angle)
    assert sites == (0,)
    np.testing.assert_allclose(matrix, GateLibrary.rx([expected_angle]).matrix)


def test_gate_matrix_delegates_with_one_data_map_evaluation() -> None:
    """The backward-compatible matrix API delegates without resolving data twice."""
    calls = 0

    def data_map(x: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return float(x[0])

    gate = ParameterizedGate("ry", (0,), data_map=data_map)
    circuit = ParameterizedCircuit(1, [gate])

    matrix, sites = circuit.gate_matrix(gate, np.array([], dtype=np.float64), np.array([0.25]))

    assert calls == 1
    assert sites == (0,)
    np.testing.assert_allclose(matrix, GateLibrary.ry([0.25]).matrix)


def test_gate_matrix_and_angle_fixed_parametric_offset() -> None:
    """A fixed single-angle gate reports its stored angle offset."""
    gate = ParameterizedGate("rz", (0,), angle_offset=0.37)
    circuit = ParameterizedCircuit(1, [gate])

    matrix, sites, resolved_angle = circuit.gate_matrix_and_angle(gate, np.array([], dtype=np.float64))

    assert resolved_angle == pytest.approx(0.37)
    assert sites == (0,)
    np.testing.assert_allclose(matrix, GateLibrary.rz([0.37]).matrix)


def test_gate_matrix_and_angle_nonparametric_gate() -> None:
    """A nonparametric gate has no resolved angle."""
    gate = ParameterizedGate("h", (0,))
    circuit = ParameterizedCircuit(1, [gate])

    matrix, sites, resolved_angle = circuit.gate_matrix_and_angle(gate, np.array([], dtype=np.float64))

    assert resolved_angle is None
    assert sites == (0,)
    np.testing.assert_allclose(matrix, GateLibrary.h().matrix)


def test_gate_matrix_and_angle_preserves_reversed_site_convention() -> None:
    """Returning the angle must not alter reversed two-site matrix conversion."""
    gate = ParameterizedGate("cx", (1, 0))
    circuit = ParameterizedCircuit(2, [gate])
    expected = np.asarray(GateLibrary.cx().matrix, dtype=np.complex128)
    expected = expected.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)

    matrix, sites, resolved_angle = circuit.gate_matrix_and_angle(gate, np.array([], dtype=np.float64))
    delegated_matrix, delegated_sites = circuit.gate_matrix(gate, np.array([], dtype=np.float64))

    assert resolved_angle is None
    assert sites == delegated_sites == (0, 1)
    np.testing.assert_allclose(matrix, expected)
    np.testing.assert_allclose(delegated_matrix, expected)
