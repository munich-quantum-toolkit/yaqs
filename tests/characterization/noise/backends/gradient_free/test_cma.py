# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the CMA-ES backend."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from mqt.yaqs.characterization.noise.backends.gradient_free import cma

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.monkeypatch import MonkeyPatch

    from mqt.yaqs.characterization.noise.shared.loss import TrajectoryLoss


class DummyStrategy:
    """Lightweight stand-in for ``cma.CMAEvolutionStrategy``."""

    def __init__(
        self,
        _x0: np.ndarray,
        _sigma0: float,
        options: dict[str, Any],
        *,
        stop_after_first: bool = True,
    ) -> None:
        """Record optimizer options and configure early-stop behavior."""
        self.options = options
        self.calls = 0
        self.tell_calls = 0
        self.stop_after_first = stop_after_first
        self.result = types.SimpleNamespace(xbest=None, fbest=None)

    def ask(self) -> list[np.ndarray]:
        """Return a fixed candidate population.

        Returns:
            Two candidate parameter vectors for the mocked optimizer.
        """
        self.calls += 1
        return [np.array([1.0, 2.0]), np.array([-1.0, 0.5])]

    def tell(self, solutions: list[np.ndarray], values: list[float]) -> None:
        """Track the best candidate from the latest population."""
        self.tell_calls += 1
        best_idx = int(np.argmin(values))
        self.result.xbest = np.array(solutions[best_idx])
        self.result.fbest = float(values[best_idx])

    def stop(self) -> bool:
        """Stop after the first iteration when configured for smoke tests.

        Returns:
            ``True`` once the first ask/tell cycle completed.
        """
        return self.stop_after_first and self.calls >= 1


def _patch_strategy(monkeypatch: MonkeyPatch, factory: Callable[..., DummyStrategy]) -> list[DummyStrategy]:
    created: list[DummyStrategy] = []

    def _wrapper(x0: np.ndarray, sigma0: float, options: dict[str, Any]) -> DummyStrategy:
        inst = factory(x0, sigma0, options)
        created.append(inst)
        return inst

    monkeypatch.setattr("cma.CMAEvolutionStrategy", _wrapper)
    return created


def make_loss(obj: object) -> TrajectoryLoss:
    """Cast a test double to :class:`~mqt.yaqs.characterization.noise.shared.loss.TrajectoryLoss`.

    Returns:
        The input object typed as a trajectory loss callable.
    """
    return cast("TrajectoryLoss", obj)


def test_cma_opt_returns_best_solution(monkeypatch: MonkeyPatch) -> None:
    """CMA-ES returns the lowest-loss candidate from the mocked population."""
    created = _patch_strategy(monkeypatch, DummyStrategy)

    class Objective:
        def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
            return float(np.sum(x**2)), np.zeros_like(x), 0.0

    xbest, fbest, loss_history, param_history = cma.cma_opt(
        make_loss(Objective()),
        np.array([0.0, 0.0]),
        sigma0=0.1,
        max_iter=2,
    )

    assert created[0].tell_calls == 1
    np.testing.assert_array_equal(xbest, np.array([-1.0, 0.5]))
    assert fbest == pytest.approx(1.25)
    assert len(loss_history) == 2
    assert len(param_history) == 2


def test_cma_opt_forwards_seed(monkeypatch: MonkeyPatch) -> None:
    """Optional ``seed`` values are forwarded to the CMA-ES options dict."""
    created = _patch_strategy(monkeypatch, DummyStrategy)

    class Objective:
        def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
            return float(np.sum(x**2)), np.zeros_like(x), 0.0

    cma.cma_opt(
        make_loss(Objective()),
        np.array([0.0, 0.0]),
        sigma0=0.1,
        max_iter=1,
        seed=42,
    )

    assert created[0].options["seed"] == 42
