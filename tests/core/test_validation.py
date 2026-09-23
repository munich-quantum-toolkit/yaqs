# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared validation helpers and interpreter-independent validation."""

# ruff:file-ignore[import-private-name] -- white-box tests cover private shared validation helpers

from __future__ import annotations

import json
import subprocess
import sys
from typing import cast

import numpy as np
import pytest

from mqt.yaqs.core._validation import (
    validate_bool,
    validate_choice,
    validate_finite_real,
    validate_integer,
    validate_real,
)

_OPTIMIZED_VALIDATION_SCRIPT = r"""
import json

import numpy as np

from mqt.yaqs import Hamiltonian, Simulator, State
from mqt.yaqs.characterization.noise.backends.cma import cma_opt
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.observable import Observable
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, DigitalSimParams
from mqt.yaqs.core.libraries.circuit_library_utils import extract_u_parameters
from mqt.yaqs.core.libraries.gate_library import split_tensor
from mqt.yaqs.core.parallel_utils import ExecutionConfig


def invalid_expect_sites():
    state = MPS(2, state="zeros")
    observable = Observable("z", 0)
    observable.sites = "0"
    state.expect(observable)


def invalid_bond_dimensions():
    state = MPS(2, state="zeros")
    state.tensors[1] = np.zeros((2, 2, 1), dtype=np.complex128)
    state.check_if_valid_mps()


def mismatched_shift_right():
    state = MPS(3, state="zeros")
    state.set_center(1)
    state.shift_orthogonality_center_right(0)


def mismatched_shift_left():
    state = MPS(3, state="zeros")
    state.shift_orthogonality_center_left(1)


def invalid_custom_mpo():
    mpo = MPO()
    mpo.custom(
        [np.zeros((2, 2, 1, 2), dtype=np.complex128), np.zeros((2, 2, 3, 1), dtype=np.complex128)],
        transpose=False,
    )


def invalid_finite_state_machine_mpo():
    mpo = MPO()
    mpo.finite_state_machine(
        3,
        np.zeros((1, 2, 2, 2), dtype=np.complex128),
        np.zeros((3, 3, 2, 2), dtype=np.complex128),
        np.zeros((3, 1, 2, 2), dtype=np.complex128),
    )


def invalid_mutated_simulation_control():
    params = AnalogSimParams(observables=[ordinary], elapsed_time=0.0, dt=0.1)
    params.num_traj = 0
    Simulator(parallel=False, show_progress=False).run(
        State(1, initial="zeros", representation="vector"),
        Hamiltonian.ising(1, J=0.0, g=0.0),
        params,
    )


pvm = Observable("00")
ordinary = Observable("z", 0)
state = MPS(2, state="zeros")
tensor = np.zeros((2, 1, 1), dtype=np.complex128)
calls = {
    "length-type": lambda: MPS(1.5),
    "tensor-count": lambda: MPS(2, tensors=[tensor]),
    "basis-string": lambda: MPS(2, state="basis"),
    "bond-sites": lambda: state.get_entropy([1, 0]),
    "observable-sites": invalid_expect_sites,
    "bitstring": lambda: state.project_onto_bitstring("0x"),
    "bond-dimensions": invalid_bond_dimensions,
    "shift-right-center-mismatch": mismatched_shift_right,
    "shift-left-center-mismatch": mismatched_shift_left,
    "custom-mpo-bond-mismatch": invalid_custom_mpo,
    "finite-state-machine-mpo-bond-mismatch": invalid_finite_state_machine_mpo,
    "u-parameter-matrix-shape": lambda: extract_u_parameters(np.eye(3)),
    "gate-tensor-shape": lambda: split_tensor(np.zeros((2, 2, 2))),
    "analog-observable-mix": lambda: AnalogSimParams(observables=[pvm, ordinary]),
    "digital-observable-mix": lambda: DigitalSimParams(observables=[pvm, ordinary]),
    "execution-boolean": lambda: ExecutionConfig(parallel="false"),
    "mutated-simulation-control": invalid_mutated_simulation_control,
    "optimizer-population": lambda: cma_opt(lambda x: float(np.sum(x**2)), np.array([0.1, 0.2]), popsize=1),
}

observed = {}
for name, call in calls.items():
    try:
        call()
    except Exception as exc:
        observed[name] = [type(exc).__name__, str(exc)]
    else:
        observed[name] = None

print(json.dumps(observed, sort_keys=True))
"""


def _run_optimized_validation_script(*interpreter_args: str) -> dict[str, list[str]]:
    """Run the public-validation probe with the current Python interpreter.

    Returns:
        Mapping from each invalid call to its exception type and message.
    """
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - sys.executable is trusted.
        [sys.executable, *interpreter_args, "-c", _OPTIMIZED_VALIDATION_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return cast("dict[str, list[str]]", json.loads(completed.stdout))


def test_scalar_validators_accept_numpy_equivalents() -> None:
    """Strict scalar validation accepts equivalent NumPy scalar types."""
    assert validate_bool(np.bool_(1), name="enabled") is True
    assert validate_integer(np.int64(3), name="count", minimum=1) == 3
    assert validate_finite_real(np.float32(0.25), name="threshold") == pytest.approx(0.25)


@pytest.mark.parametrize("value", [1, "false", None])
def test_validate_bool_rejects_truthy_and_falsey_non_booleans(value: object) -> None:
    """Boolean settings do not use Python truth-value coercion."""
    with pytest.raises(TypeError, match="enabled must be a boolean"):
        validate_bool(value, name="enabled")


def test_validate_choice_requires_an_exact_supported_string() -> None:
    """Selectors reject non-strings and unsupported spellings."""
    allowed = ("auto", "vector")
    with pytest.raises(TypeError, match="representation must be a string"):
        validate_choice(1, name="representation", allowed=allowed)
    with pytest.raises(ValueError, match="representation must be one of"):
        validate_choice("AUTO", name="representation", allowed=allowed)


@pytest.mark.parametrize("value", [True, 1.5, np.ones((), dtype=np.bool_)[()]])
def test_validate_integer_rejects_non_integer_values(value: object) -> None:
    """Integer settings reject booleans and fractional values."""
    with pytest.raises(TypeError, match="count must be an integer"):
        validate_integer(value, name="count")


def test_validate_integer_enforces_minimum() -> None:
    """Integer settings apply their shared inclusive lower bound."""
    with pytest.raises(ValueError, match=r"count must be >= 1"):
        validate_integer(0, name="count", minimum=1)


@pytest.mark.parametrize("value", [True, 1 + 0j, np.nan, np.inf])
def test_validate_finite_real_rejects_non_real_or_non_finite_values(value: object) -> None:
    """Finite-real settings reject booleans, complex values, NaN, and infinity."""
    error = TypeError if isinstance(value, (bool, complex)) else ValueError
    with pytest.raises(error, match="threshold must be"):
        validate_finite_real(value, name="threshold")


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
def test_validate_real_rejects_non_finite_components(value: complex) -> None:
    """Non-finite real and imaginary components fail explicitly."""
    with pytest.raises(ValueError, match="quantity must be finite"):
        validate_real(value, name="quantity")


@pytest.mark.parametrize("value", [True, "1", None, object()])
def test_validate_real_rejects_non_numeric_scalars(value: object) -> None:
    """Boolean and non-numeric inputs are not interpreted as real values."""
    with pytest.raises(TypeError, match="quantity must be a real numeric scalar"):
        validate_real(value, name="quantity")


@pytest.mark.parametrize(
    ("rtol", "atol", "parameter"),
    [
        (float("nan"), 1e-12, "rtol"),
        (float("inf"), 1e-12, "rtol"),
        (-1.0, 1e-12, "rtol"),
        (1e-10, float("nan"), "atol"),
        (1e-10, float("inf"), "atol"),
        (1e-10, -1.0, "atol"),
    ],
)
def test_validate_real_rejects_invalid_tolerances(rtol: float, atol: float, parameter: str) -> None:
    """Relative and absolute tolerances must be finite and non-negative."""
    with pytest.raises(ValueError, match=rf"{parameter} must be finite and non-negative"):
        validate_real(1.0 + 1.0j, name="quantity", rtol=rtol, atol=atol)


def test_public_validation_matches_under_optimized_python() -> None:
    """Representative public validation errors must not depend on assertions."""
    expected = {
        "length-type": ["TypeError", "length must be an integer."],
        "tensor-count": ["ValueError", "Expected 2 MPS tensors, got 1."],
        "basis-string": ["ValueError", "basis_string must be provided for 'basis' state initialization."],
        "bond-sites": ["ValueError", "entropy sites must be ordered nearest neighbors, got [1, 0]."],
        "observable-sites": ["TypeError", "observable sites must be an integer or a list of integers."],
        "bitstring": ["ValueError", "bitstring character at site 1 must be numeric, got 'x'."],
        "bond-dimensions": ["ValueError", "MPS bond between sites 0 and 1 has dimensions 1 and 2."],
        "shift-right-center-mismatch": [
            "ValueError",
            "shift_orthogonality_center_right: tracked center is 1, but shift requested from site 0.",
        ],
        "shift-left-center-mismatch": [
            "ValueError",
            "shift_orthogonality_center_left: tracked center is 0, but shift requested from site 1.",
        ],
        "custom-mpo-bond-mismatch": [
            "ValueError",
            "MPO tensors must have matching adjacent bond dimensions.",
        ],
        "finite-state-machine-mpo-bond-mismatch": [
            "ValueError",
            "MPO tensors must have matching adjacent bond dimensions.",
        ],
        "u-parameter-matrix-shape": ["ValueError", "Input must be a 2x2 matrix."],
        "gate-tensor-shape": [
            "ValueError",
            "tensor must have an even number of axes for at least two sites, got 3.",
        ],
        "analog-observable-mix": [
            "ValueError",
            "Mixed observable and projective-measurement simulation is not supported.",
        ],
        "digital-observable-mix": [
            "ValueError",
            "Mixed observable and projective-measurement simulation is not supported.",
        ],
        "execution-boolean": ["TypeError", "parallel must be a boolean, got str."],
        "mutated-simulation-control": ["ValueError", "num_traj must be >= 1, got 0."],
        "optimizer-population": ["ValueError", "popsize must be >= 2, got 1."],
    }

    normal = _run_optimized_validation_script()
    optimized = _run_optimized_validation_script("-O")

    assert optimized == normal == expected
