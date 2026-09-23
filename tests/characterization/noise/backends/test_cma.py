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


def test_cma_opt_default_bounds(monkeypatch: MonkeyPatch) -> None:
    """Unbounded optimization uses infinite lower and upper limits."""
    pytest.importorskip("cma")
    created = _patch_strategy(monkeypatch, DummyStrategy)

    class Objective:
        def __call__(self, x: np.ndarray) -> float:
            return float(np.sum(x**2))

    cma_backend.cma_opt(
        Objective(),
        np.array([0.5, 0.5]),
        sigma0=0.1,
        max_iter=1,
        popsize=4,
    )

    assert created[0].options["bounds"] == [[-np.inf, -np.inf], [np.inf, np.inf]]


def test_backend_exports_cma_opt() -> None:
    """Backend package re-exports the CMA-ES entry point."""
    assert callable(cma_opt)


def test_cma_opt_integration_smoke() -> None:
    """Real CMA-ES backend minimizes a simple quadratic objective."""
    pytest.importorskip("cma")

    class Objective:
        def __call__(self, x: np.ndarray) -> float:
            return float(np.sum(x**2))

    # Mild step size / short run avoids CMA sigma-clip advisories that vary by
    # cma/numpy version under pytest's warnings-as-errors policy.
    xbest, fbest, loss_history, param_history = cma_backend.cma_opt(
        Objective(),
        np.array([0.5, 0.5]),
        sigma0=0.05,
        max_iter=2,
        popsize=4,
        seed=42,
    )

    assert fbest < 1.0
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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"x0": np.array([])}, "x0 must not be empty"),
        ({"x0": np.zeros((1, 2))}, "x0 must be one-dimensional"),
        ({"x0": np.array([np.nan])}, "x0 must not contain non-finite"),
        ({"x0": np.array([0.1, 0.2]), "x_low": np.array([0.0])}, "x_low shape"),
        ({"x0": np.array([0.1]), "x_up": np.array([np.nan])}, "x_up must not contain NaN"),
        (
            {"x0": np.array([0.1]), "x_low": np.array([0.2]), "x_up": np.array([0.2])},
            "x_low entry must be strictly smaller",
        ),
        (
            {"x0": np.array([0.3]), "x_low": np.array([0.0]), "x_up": np.array([0.2])},
            "x0 entry must lie within",
        ),
    ],
)
def test_cma_opt_rejects_invalid_parameter_vectors(kwargs: dict[str, np.ndarray], match: str) -> None:
    """Malformed parameter vectors and bounds fail before optimizer construction."""

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    with pytest.raises(ValueError, match=match):
        cma_opt(objective, kwargs["x0"], x_low=kwargs.get("x_low"), x_up=kwargs.get("x_up"))


@pytest.mark.parametrize("x0", [["0.1"], [True], [1 + 0j]])
def test_cma_opt_rejects_coercive_parameter_vectors(x0: object) -> None:
    """Optimizer vectors must contain real numeric values before conversion."""

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    with pytest.raises(TypeError, match="x0 must be a one-dimensional real numeric array"):
        cma_opt(objective, cast("Any", x0))


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("sigma0", 0.0, ValueError),
        ("sigma0", np.inf, ValueError),
        ("sigma0", True, TypeError),
        ("popsize", 1, ValueError),
        ("popsize", 4.0, TypeError),
        ("max_iter", 0, ValueError),
        ("max_iter", 2.0, TypeError),
        ("seed", -1, ValueError),
        ("seed", 1.5, TypeError),
    ],
)
def test_cma_opt_rejects_invalid_scalar_controls(name: str, value: object, error: type[Exception]) -> None:
    """CMA-ES controls use explicit Boolean, integer, and finite-real contracts."""

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    kwargs: dict[str, Any] = {name: value}
    with pytest.raises(error, match=name):
        cma_opt(objective, np.array([0.1, 0.2]), **kwargs)


def test_cma_opt_accepts_numpy_scalar_controls(monkeypatch: MonkeyPatch) -> None:
    """Equivalent NumPy scalar controls are normalized before CMA-ES dispatch."""
    created = _patch_strategy(monkeypatch, DummyStrategy)

    def objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    cma_opt(
        objective,
        np.array([0.1, 0.2]),
        sigma0=np.float64(0.1),
        popsize=cast("Any", np.int64(4)),
        max_iter=cast("Any", np.int64(1)),
        seed=cast("Any", np.int64(3)),
    )

    assert created[0].options["popsize"] == 4
    assert created[0].options["seed"] == 3
