# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""CMA-ES wrapper for noise-parameter optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mqt.yaqs.characterization.noise.shared.loss import TrajectoryLoss


def cma_opt(
    loss: TrajectoryLoss,
    x0: np.ndarray,
    x_low: np.ndarray | None = None,
    x_up: np.ndarray | None = None,
    sigma0: float = 0.01,
    popsize: int = 4,
    max_iter: int = 500,
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

    Returns:
        Best parameter vector, best loss, per-evaluation loss history, and
        parameter history.
    """
    import cma  # noqa: PLC0415

    x0 = np.asarray(x0, dtype=float)
    if x_low is None:
        x_low = -np.inf * np.ones_like(x0)
    if x_up is None:
        x_up = np.inf * np.ones_like(x0)

    f_history: list[float] = []
    x_history: list[np.ndarray] = []

    def evaluate(x: np.ndarray) -> tuple[float, np.ndarray, float]:
        loss_value, grad, elapsed = loss(x)
        f_history.append(loss_value)
        x_history.append(np.asarray(x, dtype=float).copy())
        return loss_value, grad, elapsed

    es = cma.CMAEvolutionStrategy(  # type: ignore[possibly-missing-attribute]
        x0,
        sigma0,
        {
            "popsize": popsize,
            "verb_disp": 0,
            "bounds": [np.asarray(x_low, dtype=float).tolist(), np.asarray(x_up, dtype=float).tolist()],
        },
    )

    for _ in range(max_iter):
        solutions = es.ask()
        values = [evaluate(x)[0] for x in solutions]
        es.tell(solutions, values)
        if es.stop():
            break

    result = es.result
    return result.xbest, float(result.fbest), f_history, x_history
