# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""CMA-ES wrapper for noise-parameter optimization."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.optimize import minimize_scalar


class ScalarLoss(Protocol):
    """Callable that maps a parameter vector to a scalar objective."""

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the loss at ``x``."""
        ...


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
    # Lazy import: ``cma`` is an optional extra; keep this module importable without it.
    import cma  # noqa: PLC0415

    x0 = np.asarray(x0, dtype=float)
    if x_low is None:
        x_low = -np.inf * np.ones_like(x0)
    if x_up is None:
        x_up = np.inf * np.ones_like(x0)
    x_low = np.asarray(x_low, dtype=float)
    x_up = np.asarray(x_up, dtype=float)

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
