# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Markov Chain Monte Carlo (MCMC) optimization algorithm for noise characterization.

This module implements MCMC-based optimization using simulated annealing for
gradient-free optimization of noise model parameters. It supports bounds,
early stopping based on patience, and uses modern NumPy random number generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mqt.yaqs.noise_char.loss import LossClass


def mcmc_opt(
    f: LossClass,
    x0: np.ndarray,
    x_low: np.ndarray | None = None,
    x_up: np.ndarray | None = None,
    max_iter: int = 500,
    step_size: float = 0.05,
    step_rate: float = 0.99,
    min_step_size: float = 0,
    temperature: float = 1.0,
    anneal_rate: float = 0.99,
    patience: int = 100,
) -> tuple[np.ndarray, float]:
    """MCMC-based optimization with simulated annealing and early stopping.

    This function performs Markov Chain Monte Carlo (MCMC) optimization using a Metropolis-Hastings
    algorithm with simulated annealing. It includes early stopping if no improvement occurs for a
    specified number of iterations (patience).

    Parameters
    ----------
    f : LossClass
        Loss function instance that takes a numpy array and returns a tuple (loss, grad, obs).
    x0 : np.ndarray
        Initial point for optimization, a 1D numpy array.
    x_low : np.ndarray or None, optional
        Lower bounds for x. If None, no lower bounds are applied. Must be same shape as x0.
    x_up : np.ndarray or None, optional
        Upper bounds for x. If None, no upper bounds are applied. Must be same shape as x0.
    max_iter : int, optional
        Maximum number of MCMC iterations. Default is 500.
    step_size : float, optional
        Initial standard deviation of the Gaussian proposal distribution. Default is 0.05.
    step_rate : float, optional
        Factor by which step_size is multiplied each iteration. Default is 0.99.
    min_step_size : float, optional
        Minimum allowed step_size. Default is 0 (not used in current implementation).
    temperature : float, optional
        Initial temperature for Metropolis-Hastings acceptance. Default is 1.0.
    anneal_rate : float, optional
        Cooling factor for temperature per iteration. Default is 0.99.
    patience : int, optional
        Number of iterations without improvement before early stopping. Default is 100.

    Returns:
    -------
    xbest : np.ndarray
        The best point found during optimization.
    fbest : float
        The best (lowest) loss value found.
    """
    x = np.array(x0, dtype=float)
    ndim = x.size

    fx, _, _ = f(x)
    xbest, fbest = x.copy(), fx

    if x_low is not None:
        x_low = np.array(x_low, dtype=float)
    if x_up is not None:
        x_up = np.array(x_up, dtype=float)

    rng = np.random.default_rng()

    no_improve_counter = 0

    for _i in range(max_iter):
        # Gaussian proposal
        x_new = x + rng.normal(scale=step_size, size=ndim)

        # Apply bounds
        if x_low is not None:
            x_new = np.maximum(x_new, x_low)
        if x_up is not None:
            x_new = np.minimum(x_new, x_up)

        f_new, _, _ = f(x_new)

        # Metropolis-Hastings acceptance
        delta = f_new - fx
        acceptance_prob = np.exp(-delta / temperature)

        if rng.random() < acceptance_prob:
            x, fx = x_new, f_new

        # Track global best
        if fx < fbest:
            xbest, fbest = x.copy(), fx
            no_improve_counter = 0  # reset
        else:
            no_improve_counter += 1

        # Early stopping condition
        if no_improve_counter >= patience:
            break

        # Annealing
        temperature *= anneal_rate

        step_size *= step_rate

        step_size = max(step_size, min_step_size)

    return xbest, fbest
