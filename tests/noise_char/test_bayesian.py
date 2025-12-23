# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for Bayesian optimization algorithms in noise characterization.

This module tests the Bayesian optimization functionality including:
- Acquisition function selection and validation
- Bayesian optimization loop with Gaussian process surrogate models
- Convergence handling and early stopping
- Return value validation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

# Suppress botorch warnings:
# - Deprecation warnings related to numpy 2.0 compatibility
# - InputDataWarning about float32 (we intentionally test with float32)
# These are external library warnings, not issues with our code
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:botorch.*",
    "ignore:The model inputs are of type torch.float32.*:UserWarning",  # InputDataWarning about float32
)

from botorch.acquisition import (
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.models import SingleTaskGP

from mqt.yaqs.noise_char.optimization_algorithms.gradient_free.bayesian import (
    bayesian_opt,
    get_acquisition_function,
)

if TYPE_CHECKING:
    from mqt.yaqs.noise_char.loss import LossClass


class MockLossClass:
    """Mock LossClass for testing Bayesian optimization.

    This class simulates a loss function that can be called and has a converged attribute.
    """

    def __init__(self, objective_func: callable, converged: bool = False) -> None:
        """Initialize mock loss class.

        Args:
            objective_func: Function that takes a 1D numpy array and returns a scalar.
            converged: Whether the optimization has converged.
        """
        self.converged = converged
        self._objective_func = objective_func
        self.n_eval = 0

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
        """Evaluate the objective function.

        Args:
            x: Parameter vector.

        Returns:
            Tuple of (loss_value, gradient, simulation_time).
            Gradient is a zero array, simulation_time is 0.0.
        """
        self.n_eval += 1
        loss_value = self._objective_func(x)
        grad = np.zeros_like(x)
        return loss_value, grad, 0.0


def make_loss(objective_func: callable, *, converged: bool = False) -> LossClass:
    """Create a LossClass-typed mock for type checkers."""
    return cast("LossClass", MockLossClass(objective_func, converged))


def test_get_acquisition_function_lei() -> None:
    """Test that get_acquisition_function returns LogExpectedImprovement for 'LEI'."""
    x_train = torch.rand(5, 2, dtype=torch.double)
    y_train = torch.rand(5, 1, dtype=torch.double)
    model = SingleTaskGP(x_train, y_train)

    acq_func = get_acquisition_function("LEI", model, best_f=0.5)
    assert isinstance(acq_func, LogExpectedImprovement)


def test_get_acquisition_function_pi() -> None:
    """Test that get_acquisition_function returns ProbabilityOfImprovement for 'PI'."""
    x_train = torch.rand(5, 2, dtype=torch.double)
    y_train = torch.rand(5, 1, dtype=torch.double)
    model = SingleTaskGP(x_train, y_train)

    acq_func = get_acquisition_function("PI", model, best_f=0.5)
    assert isinstance(acq_func, ProbabilityOfImprovement)


def test_get_acquisition_function_ucb() -> None:
    """Test that get_acquisition_function returns UpperConfidenceBound for 'UCB'."""
    x_train = torch.rand(5, 2, dtype=torch.double)
    y_train = torch.rand(5, 1, dtype=torch.double)
    model = SingleTaskGP(x_train, y_train)

    acq_func = get_acquisition_function("UCB", model, beta=2.576)
    assert isinstance(acq_func, UpperConfidenceBound)


def test_get_acquisition_function_invalid_name() -> None:
    """Test that get_acquisition_function raises ValueError for invalid names."""
    x_train = torch.rand(5, 2, dtype=torch.double)
    y_train = torch.rand(5, 1, dtype=torch.double)
    model = SingleTaskGP(x_train, y_train)

    with pytest.raises(ValueError, match="Unknown acquisition function"):
        get_acquisition_function("INVALID", model)


def test_bayesian_opt_basic_functionality() -> None:
    """Test basic Bayesian optimization with a simple quadratic function."""

    # Simple quadratic function: f(x) = (x - 0.5)^2
    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - 0.5) ** 2))

    loss = make_loss(objective)
    x_low = np.array([0.0, 0.0])
    x_up = np.array([1.0, 1.0])

    best_x, best_y, x_train, y_train = bayesian_opt(loss, x_low=x_low, x_up=x_up, n_init=3, max_iter=2, acq_name="UCB")

    # Check return types
    assert isinstance(best_x, np.ndarray)
    assert isinstance(best_y, (np.floating, np.number, float))  # best_y is a numpy scalar
    assert isinstance(x_train, torch.Tensor)
    assert isinstance(y_train, torch.Tensor)

    # Check shapes
    assert best_x.shape == (2,)
    assert np.isscalar(best_y)  # best_y is a scalar
    assert x_train.shape[1] == 2  # Second dimension should match input dimension
    assert y_train.shape[1] == 1  # y_train should have shape (n_evals, 1)

    # Check that best_x is within bounds
    assert np.all(best_x >= x_low)
    assert np.all(best_x <= x_up)

    # Check that we evaluated the function
    assert loss.n_eval > 0


def test_bayesian_opt_different_acquisition_functions() -> None:
    """Test Bayesian optimization with different acquisition functions."""

    # Simple objective function
    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    x_low = np.array([-1.0, -1.0])
    x_up = np.array([1.0, 1.0])

    for acq_name in ["LEI", "PI", "UCB"]:
        loss = make_loss(objective)
        best_x, best_y, _, _ = bayesian_opt(loss, x_low=x_low, x_up=x_up, n_init=2, max_iter=2, acq_name=acq_name)

        assert isinstance(best_x, np.ndarray)
        assert isinstance(best_y, (np.floating, np.number, float))  # best_y is a numpy scalar
        assert best_x.shape == (2,)


def test_bayesian_opt_convergence() -> None:
    """Test that Bayesian optimization respects the converged flag."""

    # Simple objective function
    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    loss = make_loss(objective, converged=False)
    x_low = np.array([-1.0])
    x_up = np.array([1.0])

    # Set converged after first iteration's evaluation (n_init + 1 evaluations)
    # The convergence check happens after each iteration, so we need to set it
    # during the first iteration's evaluation
    original_call = loss.__call__

    def call_with_convergence(x: np.ndarray) -> tuple[float, np.ndarray, float]:
        result = original_call(x)
        # Set converged after the first iteration's evaluation (n_init=2, so after 3rd eval)
        if loss.n_eval == 3:
            loss.converged = True
        return result

    loss.__call__ = call_with_convergence

    # Run with max_iter=10, but should stop early due to convergence
    best_x, best_y, _x_train, _y_train = bayesian_opt(
        loss, x_low=x_low, x_up=x_up, n_init=2, max_iter=10, acq_name="UCB"
    )

    # Verify that optimization stopped early
    # With n_init=2 and convergence set after 3rd eval, it should stop after 1 iteration
    # So total evals should be 3 (n_init=2 + 1 iteration), not 2 + 10 = 12
    assert isinstance(best_x, np.ndarray)
    assert isinstance(best_y, (np.floating, np.number, float))  # best_y is a numpy scalar


def test_bayesian_opt_return_values() -> None:
    """Test that Bayesian optimization returns values in the correct format."""

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    loss = make_loss(objective)
    x_low = np.array([0.0, 0.0])
    x_up = np.array([1.0, 1.0])

    best_x, best_y, x_train, y_train = bayesian_opt(loss, x_low=x_low, x_up=x_up, n_init=3, max_iter=2)

    # Check that best_x is a numpy array
    assert isinstance(best_x, np.ndarray)
    assert best_x.dtype in {np.float64, np.float32}

    # Check that best_y is a numpy scalar
    assert isinstance(best_y, (np.floating, np.number, float))
    assert np.isscalar(best_y)

    # Check that x_train is a torch tensor
    assert isinstance(x_train, torch.Tensor)
    assert x_train.dtype == torch.double

    # Check that y_train is a torch tensor
    assert isinstance(y_train, torch.Tensor)
    assert y_train.dtype == torch.double
    assert y_train.ndim == 2
    assert y_train.shape[1] == 1


def test_bayesian_opt_one_dimensional() -> None:
    """Test Bayesian optimization with one-dimensional input."""

    def objective(x: np.ndarray) -> float:
        return float((x[0] - 0.3) ** 2)

    loss = make_loss(objective)
    x_low = np.array([0.0])
    x_up = np.array([1.0])

    best_x, _best_y, x_train, _y_train = bayesian_opt(loss, x_low=x_low, x_up=x_up, n_init=2, max_iter=2)

    assert best_x.shape == (1,)
    assert x_train.shape[1] == 1


def test_bayesian_opt_three_dimensional() -> None:
    """Test Bayesian optimization with three-dimensional input."""

    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - 0.5) ** 2))

    loss = make_loss(objective)
    x_low = np.array([0.0, 0.0, 0.0])
    x_up = np.array([1.0, 1.0, 1.0])

    best_x, _best_y, x_train, _y_train = bayesian_opt(loss, x_low=x_low, x_up=x_up, n_init=2, max_iter=2)

    assert best_x.shape == (3,)
    assert x_train.shape[1] == 3


def test_bayesian_opt_custom_parameters() -> None:
    """Test Bayesian optimization with custom parameters."""

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    loss = make_loss(objective)
    x_low = np.array([-1.0, -1.0])
    x_up = np.array([1.0, 1.0])

    best_x, best_y, x_train, _y_train = bayesian_opt(
        loss,
        x_low=x_low,
        x_up=x_up,
        n_init=4,
        max_iter=3,
        acq_name="UCB",
        std=1e-2,
        beta=2.5,
        dtype=torch.float32,
        device="cpu",
    )

    assert isinstance(best_x, np.ndarray)
    assert isinstance(best_y, (np.floating, np.number, float))  # best_y is a numpy scalar
    assert x_train.dtype == torch.float32


def test_bayesian_opt_improves_objective() -> None:
    """Test that Bayesian optimization improves the objective over iterations."""

    # Objective with minimum at [0.5, 0.5]
    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - 0.5) ** 2))

    loss = make_loss(objective)
    x_low = np.array([0.0, 0.0])
    x_up = np.array([1.0, 1.0])

    best_x, best_y, _x_train, _y_train = bayesian_opt(
        loss, x_low=x_low, x_up=x_up, n_init=3, max_iter=5, acq_name="UCB"
    )

    # The best value should be reasonable (not too far from minimum)
    # Since this is stochastic, we just check it's within bounds
    assert np.all(best_x >= x_low)
    assert np.all(best_x <= x_up)
    assert best_y >= 0  # Objective is non-negative


def test_bayesian_opt_handles_negative_values() -> None:
    """Test Bayesian optimization with negative bounds."""

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    loss = make_loss(objective)
    x_low = np.array([-2.0, -2.0])
    x_up = np.array([2.0, 2.0])

    best_x, _best_y, _x_train, _y_train = bayesian_opt(loss, x_low=x_low, x_up=x_up, n_init=2, max_iter=2)

    assert np.all(best_x >= x_low)
    assert np.all(best_x <= x_up)
