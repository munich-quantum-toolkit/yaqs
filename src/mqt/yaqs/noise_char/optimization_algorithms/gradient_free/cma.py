# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Module for implementing the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization algorithm.

This module provides a function `cma_opt` that performs optimization using the CMA-ES algorithm,
which is a derivative-free optimization method suitable for non-linear and non-convex problems.

The `cma_opt` function allows for the specification of an objective function to minimize,
initial guesses, and optional bounds for the optimization variables. It also supports
custom convergence detection through the provided objective function.

The CMA-ES algorithm is particularly useful in scenarios where the objective function is expensive
to evaluate or when the gradient information is not available.
"""

# %%
from __future__ import annotations

from typing import TYPE_CHECKING

import cma
import numpy as np

if TYPE_CHECKING:
    from mqt.yaqs.noise_char.loss import LossClass


def cma_opt(
    f: LossClass,
    x0: np.ndarray,
    x_low: np.ndarray | None = None,
    x_up: np.ndarray | None = None,
    sigma0: float = 0.01,
    popsize: int = 4,
    max_iter: int = 500,
) -> tuple[np.ndarray, float]:
    """CMA-ES optimization with optional lower and upper bounds per dimension and a maximum number of iterations.

    Parameters
    ----------
    f : callable
        Objective function to minimize.
    x0 : array-like
        Initial guess.
    sigma0 : float
        Initial standard deviation (step size).
    popsize : int, optional
        Population size.
    x_low : float or array-like, optional
        Lower bounds for each dimension (default: -inf).
    x_up : float or array-like, optional
        Upper bounds for each dimension (default: +inf).
    max_iter : int, optional
        Maximum number of iterations (default: 500).

    Returns:
    -------
    fbest : float
        Best objective function value.
    xbest : ndarray
        Best solution found.
    """
    x0 = np.array(x0, dtype=float)

    # Handle flexible bounds
    if x_low is None:
        x_low = -np.inf * np.ones_like(x0)
    if x_up is None:
        x_up = np.inf * np.ones_like(x0)
    x_low = np.array(x_low, dtype=float)
    x_up = np.array(x_up, dtype=float)

    # CMA-ES configuration
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "popsize": popsize,
            "verb_disp": 0,
            "bounds": [x_low.tolist(), x_up.tolist()],
        },
    )

    # Run optimization loop
    for _i in range(max_iter):
        solutions = es.ask()
        values = [f(x)[0] for x in solutions]
        es.tell(solutions, values)

        # Optional custom convergence detection
        if hasattr(f, "converged") and getattr(f, "converged", False):
            break

        if es.stop():
            break

        if f.converged:
            break

    result = es.result
    return result.xbest, result.fbest
