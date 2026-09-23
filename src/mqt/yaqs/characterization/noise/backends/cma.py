# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""CMA-ES wrapper for noise-parameter optimization."""

from __future__ import annotations

from typing import Protocol

import cma
import numpy as np
from scipy.optimize import minimize_scalar

from mqt.yaqs.core._validation import validate_finite_real, validate_integer  # ruff: ignore[import-private-name] -- shared package-internal validators


class ScalarLoss(Protocol):
    """Callable that maps a parameter vector to a scalar objective."""

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the loss at ``x``."""
        ...


def _as_parameter_vector(value: object, *, name: str, allow_infinite: bool) -> np.ndarray:
    """Return a validated one-dimensional optimizer vector.

    Args:
        value: Candidate vector.
        name: Parameter name used in error messages.
        allow_infinite: Whether positive and negative infinity are valid entries.

    Returns:
        A floating-point one-dimensional array.

    Raises:
        TypeError: If ``value`` cannot be converted to a numeric array.
        ValueError: If the array is not one-dimensional, is empty, or contains
            unsupported non-finite values.
    """
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be a one-dimensional numeric array."
        raise TypeError(msg) from exc
    if raw.ndim != 1:
        msg = f"{name} must be one-dimensional, got shape {raw.shape}."
        raise ValueError(msg)
    if raw.size == 0:
        msg = f"{name} must not be empty."
        raise ValueError(msg)
    if (
        not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.bool_)
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        msg = f"{name} must be a one-dimensional real numeric array."
        raise TypeError(msg)
    array = raw.astype(float, copy=False)
    if np.isnan(array).any() or (not allow_infinite and not np.isfinite(array).all()):
        qualifier = "NaN" if allow_infinite else "non-finite"
        msg = f"{name} must not contain {qualifier} values."
        raise ValueError(msg)
    return array


def validate_cma_inputs(
    x0: object,
    x_low: object | None = None,
    x_up: object | None = None,
    sigma0: object = 0.01,
    popsize: object = 4,
    max_iter: object = 500,
    seed: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int, int | None]:
    """Validate and normalize CMA-ES inputs before selecting an optimizer.

    Args:
        x0: Initial parameter vector.
        x_low: Optional per-parameter lower bounds.
        x_up: Optional per-parameter upper bounds.
        sigma0: Initial CMA-ES step size.
        popsize: CMA-ES population size.
        max_iter: Maximum optimizer iteration count.
        seed: Optional CMA-ES random seed.

    Returns:
        Normalized initial values, bounds, step size, population size, iteration
        count, and seed.

    Raises:
        ValueError: If vector shapes, bounds, or scalar ranges are invalid.
    """
    initial = _as_parameter_vector(x0, name="x0", allow_infinite=False)
    lower = (
        np.full(initial.shape, -np.inf, dtype=float)
        if x_low is None
        else _as_parameter_vector(x_low, name="x_low", allow_infinite=True)
    )
    upper = (
        np.full(initial.shape, np.inf, dtype=float)
        if x_up is None
        else _as_parameter_vector(x_up, name="x_up", allow_infinite=True)
    )
    if lower.shape != initial.shape:
        msg = f"x_low shape {lower.shape} must match x0 shape {initial.shape}."
        raise ValueError(msg)
    if upper.shape != initial.shape:
        msg = f"x_up shape {upper.shape} must match x0 shape {initial.shape}."
        raise ValueError(msg)
    if np.any(lower >= upper):
        msg = "Each x_low entry must be strictly smaller than the corresponding x_up entry."
        raise ValueError(msg)
    if np.any(initial < lower) or np.any(initial > upper):
        msg = "Every x0 entry must lie within its corresponding [x_low, x_up] interval."
        raise ValueError(msg)

    sigma = validate_finite_real(sigma0, name="sigma0")
    if sigma <= 0.0:
        msg = f"sigma0 must be positive, got {sigma}."
        raise ValueError(msg)
    population = validate_integer(popsize, name="popsize", minimum=2)
    iterations = validate_integer(max_iter, name="max_iter", minimum=1)
    normalized_seed = None if seed is None else validate_integer(seed, name="seed", minimum=0)
    return initial, lower, upper, sigma, population, iterations, normalized_seed


def _optimize_scalar_bounded(
    loss: ScalarLoss,
    _x0: np.ndarray,
    x_low: np.ndarray,
    x_up: np.ndarray,
) -> tuple[np.ndarray, float, list[float], list[np.ndarray]]:
    """Minimize a one-dimensional bounded loss.

    CMA-ES does not reliably support ``d=1``; use bounded scalar search instead.

    Args:
        loss: Callable loss object.
        _x0: Initial parameter vector with length one (unused; search is global on bounds).
        x_low: Lower bound vector with length one.
        x_up: Upper bound vector with length one.

    Returns:
        Best parameter vector, best loss, per-evaluation loss history, and
        parameter history.
    """
    f_history: list[float] = []
    x_history: list[np.ndarray] = []

    def evaluate(value: float) -> float:
        loss_value = float(loss(np.array([value], dtype=float)))
        f_history.append(loss_value)
        x_history.append(np.array([value], dtype=float))
        return loss_value

    minimize_scalar(
        evaluate,
        bounds=(float(x_low[0]), float(x_up[0])),
        method="bounded",
        options={"xatol": 1e-8},
    )
    best_idx = int(np.argmin(f_history))
    return x_history[best_idx], f_history[best_idx], f_history, x_history


def cma_opt(
    loss: ScalarLoss,
    x0: np.ndarray,
    x_low: np.ndarray | None = None,
    x_up: np.ndarray | None = None,
    sigma0: float = 0.01,
    popsize: int = 4,
    max_iter: int = 500,
    seed: int | None = None,
) -> tuple[np.ndarray, float, list[float], list[np.ndarray]]:
    """Minimize a black-box loss with CMA-ES.

    Args:
        loss: Callable loss object.
        x0: Initial parameter vector.
        x_low: Optional per-dimension lower bounds.
        x_up: Optional per-dimension upper bounds.
        sigma0: Initial step size.
        popsize: Population size.
        max_iter: Maximum optimizer iterations.
        seed: Optional RNG seed forwarded to CMA-ES for reproducible runs.

    Returns:
        Best parameter vector, best loss, per-evaluation loss history, and
        parameter history.
    """
    x0, x_low, x_up, sigma0, popsize, max_iter, seed = validate_cma_inputs(
        x0,
        x_low,
        x_up,
        sigma0,
        popsize,
        max_iter,
        seed,
    )

    if x0.size == 1 and np.isfinite(x_low).all() and np.isfinite(x_up).all():
        return _optimize_scalar_bounded(loss, x0, x_low, x_up)

    f_history: list[float] = []
    x_history: list[np.ndarray] = []

    def evaluate(x: np.ndarray) -> float:
        loss_value = loss(x)
        f_history.append(loss_value)
        x_history.append(np.asarray(x, dtype=float).copy())
        return loss_value

    options: dict[str, object] = {
        "popsize": popsize,
        "verb_disp": 0,
        "bounds": [np.asarray(x_low, dtype=float).tolist(), np.asarray(x_up, dtype=float).tolist()],
    }
    if seed is not None:
        options["seed"] = seed

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        options,
    )

    for _ in range(max_iter):
        solutions = es.ask()
        values = [evaluate(x) for x in solutions]
        es.tell(solutions, values)
        if es.stop():
            break

    result = es.result
    return result.xbest, float(result.fbest), f_history, x_history
