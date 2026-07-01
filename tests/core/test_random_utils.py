# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for trajectory RNG helpers and reproducible simulation."""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

import copy

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.random_utils import make_trajectory_rng


def test_make_trajectory_rng_reproducible_per_index() -> None:
    """Same base seed and traj index yield identical streams."""
    assert make_trajectory_rng(3, base_seed=100).random() == make_trajectory_rng(3, base_seed=100).random()

    assert make_trajectory_rng(3, base_seed=100).random() != make_trajectory_rng(4, base_seed=100).random()


def test_make_trajectory_rng_none_returns_generator() -> None:
    """Unseeded mode returns a fresh NumPy Generator."""
    rng = make_trajectory_rng(0, base_seed=None)
    assert isinstance(rng, np.random.Generator)


@pytest.mark.parametrize("run_parallel", [False, True])
def test_analog_run_reproducible_with_random_seed(*, run_parallel: bool) -> None:
    """Two runs with the same random_seed produce identical aggregated observables."""
    length = 2
    state = State(length, initial="zeros", pad=4)
    H = Hamiltonian.ising(length, J=1.0, g=0.5)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.05}])
    observables = [Observable(Z(), site) for site in range(length)]

    def run_once() -> list[float]:
        st = copy.deepcopy(state)
        params = AnalogSimParams(
            observables=observables,
            elapsed_time=0.1,
            dt=0.1,
            num_traj=2,
            max_bond_dim=8,
            order=2,
            sample_timesteps=False,
            random_seed=2025,
        )
        run_result = Simulator(parallel=run_parallel, show_progress=False).run(
            st, H, params, copy.deepcopy(noise_model)
        )
        return [float(np.real(vals[0])) for vals in run_result.expectation_values]

    first = run_once()
    second = run_once()
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
