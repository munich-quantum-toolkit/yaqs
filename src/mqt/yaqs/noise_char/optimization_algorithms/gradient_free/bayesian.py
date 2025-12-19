# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bayesian optimization algorithms for noise characterization.

This module implements Bayesian optimization using Gaussian processes for optimizing
noise model parameters. It leverages BoTorch for acquisition functions and optimization,
and GPyTorch for Gaussian process modeling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

if TYPE_CHECKING:
    from mqt.yaqs.noise_char.loss import LossClass
# --------------------------------------------
# Select acquisition function
# --------------------------------------------


def get_acquisition_function(name: str, model: SingleTaskGP, best_f: float | None = None, beta: float = 2.0) -> object:
    """Get an acquisition function for Bayesian optimization.

    Parameters
    ----------
    name : str
        Name of the acquisition function. Supported options are:
        - "EI": Expected Improvement
        - "LEI": Log Expected Improvement
        - "PI": Probability of Improvement
        - "UCB": Upper Confidence Bound
    model : object
        The surrogate model used for acquisition function evaluation.
    best_f : float, optional
        The best function value found so far. Required for EI, LEI, and PI.
        Default is None.
    beta : float, optional
        Exploration-exploitation trade-off parameter for UCB.
        Default is 2.0.

    Returns:
    -------
    object
        An acquisition function object of the specified type.

    Raises:
        ValueError: If the acquisition function name is not recognized.

    Examples:
    --------
    >>> acq_ei = get_acquisition_function("EI", model, best_f=0.5)
    >>> acq_ucb = get_acquisition_function("UCB", model, beta=2.576)
    """
    name = name.upper()
    if name == "EI":
        return ExpectedImprovement(model=model, best_f=best_f, maximize=True)
    if name == "LEI":
        return LogExpectedImprovement(model=model, best_f=best_f, maximize=True)
    if name == "PI":
        return ProbabilityOfImprovement(model=model, best_f=best_f, maximize=True)
    if name == "UCB":
        return UpperConfidenceBound(model=model, beta=beta)
    msg = f"Unknown acquisition function: {name}"
    raise ValueError(msg)


# --------------------------------------------
# Bayesian Optimization Loop
# --------------------------------------------
def bayesian_opt(
    f: LossClass,
    x_low: np.ndarray | None = None,
    x_up: np.ndarray | None = None,
    n_init: int = 5,
    max_iter: int = 15,
    acq_name: str = "EI",
    std: float = 1e-6,
    beta: float = 2.0,
    dtype: torch.dtype = torch.double,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """Perform Bayesian Optimization to minimize a black-box function.

    This function uses a Gaussian Process surrogate model with an acquisition function
    to efficiently explore the search space and find the minimum of the objective function.

    Args:
        f: Callable objective function to minimize. Takes a 1D array and returns a scalar.
        x_low: Lower bounds for each dimension. Shape (d,).
        x_up: Upper bounds for each dimension. Shape (d,).
        n_init: Number of initial random samples. Defaults to 5.
        max_iter: Maximum number of optimization iterations. Defaults to 15.
        acq_name: Acquisition function name ('EI', 'UCB', etc.). Defaults to "EI".
        std: Observation noise standard deviation. Defaults to 1e-6.
        beta: Exploration parameter for acquisition function. Defaults to 2.0.
        dtype: PyTorch data type for computations. Defaults to torch.double.
        device: Compute device ('cpu' or 'cuda'). Defaults to "cpu".

    Returns:
        tuple: A tuple containing:
            - best_x (np.ndarray): Optimal point found. Shape (d,).
            - best_y (np.ndarray): Function value at optimal point.
            - x_train (torch.Tensor): All evaluated points in normalized space. Shape (n_evals, d).
            - y_train (torch.Tensor): Negated function values for all evaluations. Shape (n_evals, 1).
    """
    bounds = torch.tensor(np.array([x_low, x_up]), dtype=torch.double)

    d = bounds.shape[1]

    # Normalized [0,1]^d → real-space bounds
    def scale_to_bounds(x_unit: torch.Tensor) -> torch.Tensor:
        """Scale a unit interval value to a specified bounds range.

        Transforms a value from the unit interval [0, 1] to the range [bounds[0], bounds[1]]
        using linear scaling.

        Args:
            x_unit: A value in the unit interval [0, 1] to be scaled.

        Returns:
            The scaled value mapped to the bounds range [bounds[0], bounds[1]].
        """
        return bounds[0] + (bounds[1] - bounds[0]) * x_unit

    # -----------------------
    # Helper: evaluate f safely
    # -----------------------
    def eval_function(x: torch.Tensor) -> torch.Tensor:
        """Evaluates f safely.

        Args:
            x: torch.Tensor of shape (n, d).

        Returns:
            torch.Tensor of shape (n, 1).
        """
        x_np = x.detach().cpu().numpy()
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        y_np = np.array([f(xi)[0] for xi in x_np], dtype=np.float64)
        return torch.tensor(y_np, dtype=dtype, device=device).unsqueeze(-1)

    # -----------------------
    # Initial data
    # -----------------------
    x_train = torch.rand(n_init, d, dtype=dtype, device=device)
    y = eval_function(scale_to_bounds(x_train))
    y_train = -y  # Negate for minimization (BO maximizes internally)

    # Constant noise variance
    yvar_train = torch.full_like(y_train, std**2)

    # -----------------------
    # BO loop
    # -----------------------
    for _i in range(max_iter):
        model = SingleTaskGP(
            x_train,
            y_train,
            yvar_train,
            input_transform=Normalize(d),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        best_f = y_train.max()
        acq_func = get_acquisition_function(acq_name, model, best_f=best_f, beta=beta)

        new_x_unit, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([torch.zeros(d, device=device), torch.ones(d, device=device)]),
            q=1,
            num_restarts=5,
            raw_samples=50,
        )

        new_y = eval_function(scale_to_bounds(new_x_unit))
        new_y = -new_y  # negate for minimization

        # Append new data
        x_train = torch.cat([x_train, new_x_unit])
        y_train = torch.cat([y_train, new_y])
        yvar_train = torch.cat([yvar_train, torch.full_like(new_y, std**2)])

        best_idx = torch.argmax(y_train)
        best_x = scale_to_bounds(x_train[best_idx])
        best_y = -y_train[best_idx]

        if f.converged:
            break

    # -----------------------
    # Return best found point
    # -----------------------
    # flip back to minimization scale

    return best_x.numpy(), best_y.numpy()[0], x_train, -y_train  # flip Y back to original scale
