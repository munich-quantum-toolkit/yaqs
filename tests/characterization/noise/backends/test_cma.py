# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the CMA-ES backend."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from mqt.yaqs.characterization.noise.backends import cma as cma_backend
from mqt.yaqs.characterization.noise.backends import cma_opt

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.monkeypatch import MonkeyPatch


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


def test_cma_opt_scalar_fallback() -> None:
    """Single-parameter fits use bounded scalar search instead of CMA-ES."""

    class Objective:
        def __call__(self, x: np.ndarray) -> float:
            return float((x[0] - 0.08) ** 2)

    xbest, fbest, loss_history, param_history = cma_opt(
        Objective(),
        np.array([0.3]),
        x_low=np.array([0.0]),
        x_up=np.array([0.5]),
    )

    assert xbest.shape == (1,)
    assert xbest[0] == pytest.approx(0.08, abs=1e-3)
    assert fbest == pytest.approx(0.0, abs=1e-6)
    assert len(loss_history) >= 1
    assert len(param_history) == len(loss_history)


@pytest.mark.filterwarnings("ignore:Initial solution argument x0.*:UserWarning")
def test_cma_opt_default_bounds() -> None:
    """Unbounded optimization uses infinite lower and upper limits."""
    pytest.importorskip("cma")

    class Objective:
        def __call__(self, x: np.ndarray) -> float:
            return float(np.sum(x**2))

    xbest, fbest, _, _ = cma_backend.cma_opt(
        Objective(),
        np.array([0.5, 0.5]),
        sigma0=0.1,
        max_iter=2,
        popsize=4,
    )
    assert xbest.shape == (2,)
    assert fbest >= 0.0


def test_backend_exports_cma_opt() -> None:
    """Backend package re-exports the CMA-ES entry point."""
    assert callable(cma_opt)


def test_cma_opt_integration_smoke() -> None:
    """Real CMA-ES backend minimizes a simple quadratic objective."""
    pytest.importorskip("cma")

    class Objective:
        def __call__(self, x: np.ndarray) -> float:
            return float(np.sum(x**2))

    xbest, fbest, loss_history, param_history = cma_backend.cma_opt(
        Objective(),
        np.array([1.0, 1.0]),
        sigma0=0.2,
        max_iter=3,
        popsize=4,
        seed=0,
    )

    assert fbest < 2.0
    assert len(loss_history) >= 4
    assert len(param_history) == len(loss_history)
    assert xbest.shape == (2,)


def test_cma_opt_returns_best_solution(monkeypatch: MonkeyPatch) -> None:
    """CMA-ES returns the lowest-loss candidate from the mocked population."""
    pytest.importorskip("cma")
    created = _patch_strategy(monkeypatch, DummyStrategy)

    class Objective:
        def __call__(self, x: np.ndarray) -> float:
            return float(np.sum(x**2))

    xbest, fbest, loss_history, param_history = cma_backend.cma_opt(
        Objective(),
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
    pytest.importorskip("cma")
    created = _patch_strategy(monkeypatch, DummyStrategy)

    class Objective:
        def __call__(self, x: np.ndarray) -> float:
            return float(np.sum(x**2))

    cma_backend.cma_opt(
        Objective(),
        np.array([0.0, 0.0]),
        sigma0=0.1,
        max_iter=1,
        seed=42,
    )

    assert created[0].options["seed"] == 42
