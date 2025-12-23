# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the MCMC-based gradient-free optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.noise_char.optimization_algorithms.gradient_free import mcmc

if TYPE_CHECKING:
    from collections.abc import Sequence


class DummyRng:
    """Deterministic RNG to control proposals and acceptance decisions."""

    def __init__(
        self, normals: Sequence[Sequence[float]] | None = None, randoms: Sequence[float] | None = None
    ) -> None:
        self.normals = list(normals or [])
        self.randoms = list(randoms or [])
        self.normal_calls = 0
        self.random_calls = 0

    def normal(self, *, scale: float, size: int) -> np.ndarray:
        idx = min(self.normal_calls, len(self.normals) - 1)
        self.normal_calls += 1
        vec = np.array(self.normals[idx], dtype=float)
        # Ensure correct dimensionality
        if vec.size != size:
            vec = np.resize(vec, size)
        return scale * vec

    def random(self) -> float:
        idx = min(self.random_calls, len(self.randoms) - 1)
        self.random_calls += 1
        return float(self.randoms[idx])


def _patch_rng(monkeypatch, rng: DummyRng) -> None:
    """Patch ``np.random.default_rng`` to return a deterministic RNG."""

    def _factory():
        return rng

    monkeypatch.setattr(mcmc.np.random, "default_rng", _factory)


def test_mcmc_opt_tracks_best(monkeypatch) -> None:
    """Accepts an improved proposal and updates the global best."""
    rng = DummyRng(normals=[[-1.0, -1.0]], randoms=[0.0])
    _patch_rng(monkeypatch, rng)

    class QuadObjective:
        def __init__(self) -> None:
            self.call_count = 0

        def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
            self.call_count += 1
            return float(np.sum(x**2)), np.zeros_like(x), 0.0

    objective = QuadObjective()

    xbest, fbest = mcmc.mcmc_opt(
        objective,
        x0=np.array([1.0, 1.0]),
        max_iter=1,
        step_size=1.0,
        step_rate=1.0,
        anneal_rate=1.0,
    )

    np.testing.assert_array_equal(xbest, np.array([0.0, 0.0]))
    assert fbest == pytest.approx(0.0)
    assert objective.call_count == 2  # initial + one proposal


def test_mcmc_opt_early_stops_on_patience(monkeypatch) -> None:
    """Loop stops when no improvements occur for ``patience`` iterations."""
    patience = 3
    rng = DummyRng(normals=[[0.0]], randoms=[0.0])
    _patch_rng(monkeypatch, rng)

    class FlatObjective:
        def __init__(self) -> None:
            self.call_count = 0

        def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
            self.call_count += 1
            return 1.0, np.zeros_like(x), 0.0

    objective = FlatObjective()

    _xbest, fbest = mcmc.mcmc_opt(
        objective,
        x0=np.array([0.0]),
        max_iter=10,
        step_size=0.0,
        step_rate=1.0,
        anneal_rate=1.0,
        patience=patience,
    )

    assert fbest == pytest.approx(1.0)
    # initial evaluation plus one per iteration until patience reached
    assert objective.call_count == patience + 1


def test_mcmc_opt_applies_bounds(monkeypatch) -> None:
    """Proposals are clipped to provided bounds before evaluation."""
    rng = DummyRng(normals=[[5.0, -5.0]], randoms=[0.0])
    _patch_rng(monkeypatch, rng)

    class CaptureObjective:
        def __init__(self) -> None:
            self.last_x = None

        def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
            self.last_x = np.array(x)
            return float(np.sum(x**2)), np.zeros_like(x), 0.0

    objective = CaptureObjective()

    mcmc.mcmc_opt(
        objective,
        x0=np.array([0.0, 0.0]),
        x_low=np.array([-1.0, -1.0]),
        x_up=np.array([1.0, 1.0]),
        max_iter=1,
        step_size=1.0,
        step_rate=1.0,
        anneal_rate=1.0,
    )

    np.testing.assert_array_equal(objective.last_x, np.array([1.0, -1.0]))
