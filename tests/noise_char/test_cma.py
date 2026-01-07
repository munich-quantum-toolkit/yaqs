# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the CMA-ES wrapper used in noise characterization."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from mqt.yaqs.noise_char.optimization_algorithms.gradient_free import cma

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.monkeypatch import MonkeyPatch

    from mqt.yaqs.noise_char.loss import LossClass


class DummyStrategy:
    """Lightweight stand-in for ``cma.CMAEvolutionStrategy``."""

    def __init__(
        self,
        x0: np.ndarray,
        sigma0: float,
        options: dict[str, Any],
        *,
        stop_after_first: bool = True,
    ) -> None:
        """Initialize the dummy strategy.

        Args:
            x0: Initial solution vector.
            sigma0: Initial step size.
            options: Strategy options dictionary.
            stop_after_first: Whether to stop after the first iteration.
        """
        self.x0 = np.array(x0, dtype=float)
        self.sigma0 = sigma0
        self.options = options
        self.stop_after_first = stop_after_first
        self.calls = 0
        self.tell_calls = 0
        self.result = types.SimpleNamespace(xbest=None, fbest=None)

    def ask(self) -> list[np.ndarray]:
        """Request candidate solutions from the strategy.

        Returns:
            List of candidate solution vectors.
        """
        self.calls += 1
        return [np.array([1.0, 2.0]), np.array([-1.0, 0.5])]

    def tell(self, solutions: list[np.ndarray], values: list[float]) -> None:
        """Provide objective values for candidate solutions.

        Args:
            solutions: List of candidate solution vectors.
            values: List of objective function values corresponding to each solution.
        """
        self.tell_calls += 1
        best_idx = int(np.argmin(values))
        self.result.xbest = np.array(solutions[best_idx])
        self.result.fbest = float(values[best_idx])

    def stop(self) -> bool:
        """Check if the strategy should stop.

        Returns:
            True if the strategy should stop, False otherwise.
        """
        return self.stop_after_first and self.calls >= 1


def _patch_strategy(monkeypatch: MonkeyPatch, factory: Callable[..., DummyStrategy]) -> list[DummyStrategy]:
    """Patch CMAEvolutionStrategy and capture created instances.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        factory: Factory function to create DummyStrategy instances.

    Returns:
        List of created DummyStrategy instances.
    """
    created: list[DummyStrategy] = []

    def _wrapper(x0: np.ndarray, sigma0: float, options: dict[str, Any]) -> DummyStrategy:
        inst = factory(x0, sigma0, options)
        created.append(inst)
        return inst

    monkeypatch.setattr("cma.CMAEvolutionStrategy", _wrapper)
    return created


def make_loss(obj: object) -> LossClass:
    """Treat a simple callable/object as a LossClass for static type checking.

    Args:
        obj: The object to cast to LossClass.

    Returns:
        The object cast to LossClass type.
    """
    return cast("LossClass", obj)


def test_cma_opt_returns_best_solution(monkeypatch: MonkeyPatch) -> None:
    """Best solution should come from the lowest objective value."""
    created = _patch_strategy(monkeypatch, DummyStrategy)

    class Objective:
        converged = False

        def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
            return float(np.sum(x**2)), np.zeros_like(x), 0.0

    xbest, fbest = cma.cma_opt(make_loss(Objective()), np.array([0.0, 0.0]), sigma0=0.1, max_iter=2)

    assert created[0].tell_calls == 1
    np.testing.assert_array_equal(xbest, np.array([-1.0, 0.5]))
    assert fbest == pytest.approx(1.25)


def test_cma_opt_stops_when_objective_converges(monkeypatch: MonkeyPatch) -> None:
    """Optimization loop stops early when the objective signals convergence."""
    created = _patch_strategy(monkeypatch, lambda x0, s0, opt: DummyStrategy(x0, s0, opt, stop_after_first=False))

    class Objective:
        def __init__(self) -> None:
            self.converged = False
            self.calls = 0

        def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
            self.calls += 1
            if self.calls >= 2:
                self.converged = True
            return float(np.sum(x**2)), np.zeros_like(x), 0.0

    objective = Objective()
    xbest, fbest = cma.cma_opt(make_loss(objective), np.array([0.0, 0.0]), sigma0=0.1, max_iter=5)

    assert objective.converged
    assert created[0].tell_calls == 1  # stopped by convergence, not by strategy
    np.testing.assert_array_equal(xbest, np.array([-1.0, 0.5]))
    assert fbest == pytest.approx(1.25)


def test_cma_opt_passes_bounds(monkeypatch: MonkeyPatch) -> None:
    """Bounds are forwarded to the CMA strategy in list form."""
    created = _patch_strategy(monkeypatch, DummyStrategy)
    x_low = np.array([-1.0, -2.0])
    x_up = np.array([1.0, 2.0])

    class Objective:
        converged = False

        def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
            return 0.0, np.zeros_like(x), 0.0

    cma.cma_opt(make_loss(Objective()), np.array([0.0, 0.0]), x_low=x_low, x_up=x_up, sigma0=0.1)

    bounds = created[0].options["bounds"]
    assert bounds == [x_low.tolist(), x_up.tolist()]
