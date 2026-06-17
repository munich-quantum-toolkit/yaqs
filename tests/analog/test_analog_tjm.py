# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for analog simulation with the Tensor Jump Method.

This module provides unit tests for the analog simulation functions implemented in the
AnalogTJM module. It verifies that the initialization and time evolution routines for
the Tensor Jump Method (TJM) work as expected in various configurations, including both
first and second order evolution schemes, with and without timestep sampling.

The tests cover:
  - Initialization: Ensuring that a half time step of dissipation followed by a stochastic process
    is correctly applied to the initial state.
  - Step-through evolution: Verifying that dynamic_tdvp, apply_dissipation, and stochastic_process
    are called with the proper arguments during a single time step.
  - Analog simulation (order=2): Checking the shape of the results when running a second order evolution,
    with and without sampling timesteps.
  - Analog simulation (order=1): Checking the shape of the results when running a first order evolution,
    with and without sampling timesteps.

These tests ensure that the evolution functions correctly integrate the MPS state under the
specified Hamiltonian and noise model, and that observable measurements are properly aggregated.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from mqt.yaqs import (
    AnalogSimParams,
    Hamiltonian,
    NoiseModel,
    Observable,
    Simulator,
    State,
)
from mqt.yaqs.analog.analog_tjm import initialize, step_through
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.libraries.gate_library import X, Z
from tests.conftest import YAQS_TEST_SEED


def test_initialize() -> None:
    """Test that initialize applies a half-time dissipation and then a stochastic process to the MPS.

    This test creates an Ising MPO and an MPS of length 5, along with a minimal NoiseModel and AnalogSimParams.
    It patches the functions apply_dissipation and stochastic_process to ensure that initialize calls them with the
    correct arguments: apply_dissipation should be called with dt/2, and stochastic_process with dt.
    """
    L = 5
    J = 1
    g = 0.5
    MPO.ising(L, J, g)

    state = MPS(L)
    noise_model = NoiseModel([{"name": "lowering", "sites": [i], "strength": 0.1} for i in range(L)])
    sim_params = AnalogSimParams(
        observables=[Observable(X(), site) for site in range(L)],
        elapsed_time=0.2,
        dt=0.2,
        num_traj=1,
        max_bond_dim=2,
        sample_timesteps=False,
    )
    with (
        patch("mqt.yaqs.analog.analog_tjm.apply_dissipation") as mock_dissipation,
        patch("mqt.yaqs.analog.analog_tjm.stochastic_process") as mock_stochastic_process,
    ):
        initialize(state, noise_model, sim_params)
        mock_dissipation.assert_called_once_with(state, noise_model, sim_params.dt / 2, sim_params)
        mock_stochastic_process.assert_called_once_with(state, noise_model, sim_params.dt, sim_params, rng=None)


def test_step_through() -> None:
    """Test that step_through calls dynamic_tdvp, apply_dissipation, and stochastic_process with correct arguments.

    This test creates an Ising MPO and an MPS of length 5, along with a minimal NoiseModel and AnalogSimParams.
    It patches dynamic_tdvp, apply_dissipation, and stochastic_process to ensure that step_through calls each of them
    correctly: dynamic_tdvp should be called with the state, H, and sim_params, and both apply_dissipation and
    stochastic_process should be called with dt.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO.ising(L, J, g)

    state = MPS(L)
    noise_model = NoiseModel([{"name": "lowering", "sites": [i], "strength": 0.1} for i in range(L)])
    sim_params = AnalogSimParams(
        observables=[Observable(X(), site) for site in range(L)],
        elapsed_time=0.2,
        dt=0.2,
        num_traj=1,
        max_bond_dim=2,
        sample_timesteps=False,
    )
    with (
        patch("mqt.yaqs.analog.analog_tjm.tdvp") as mock_dynamic_tdvp,
        patch("mqt.yaqs.analog.analog_tjm.apply_dissipation") as mock_dissipation,
        patch("mqt.yaqs.analog.analog_tjm.stochastic_process") as mock_stochastic_process,
    ):
        step_through(state, H, noise_model, sim_params, current_time=0.2)
        mock_dynamic_tdvp.assert_called_once_with(state, H, sim_params)
        mock_dissipation.assert_called_once_with(state, noise_model, sim_params.dt, sim_params)
        mock_stochastic_process.assert_called_once_with(state, noise_model, sim_params.dt, sim_params, rng=None)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("sample_timesteps", [False, True])
def test_analog_tjm_shape_via_simulator(order: int, *, sample_timesteps: bool) -> None:
    """Simulator-driven analog TJM produces per-observable trajectories of the expected shape.

    Covers both one-site (order=1) and two-site (order=2) evolution, with and without
    intermediate time sampling. Per-trajectory rows on ``result.trajectories[i]`` must
    have one column when ``sample_timesteps=False`` and ``len(sim_params.times)``
    columns otherwise.
    """
    length = 5
    state = State(length, initial="zeros")
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
    observables = [Observable(Z(), site) for site in range(length)]
    sim_params = AnalogSimParams(
        observables=observables,
        elapsed_time=0.2,
        dt=0.2,
        num_traj=1,
        max_bond_dim=2,
        order=order,
        sample_timesteps=sample_timesteps,
    )

    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params)

    expected_cols = len(sim_params.times) if sample_timesteps else 1
    assert result.expectation_values is not None
    assert result.trajectories is not None
    for traj in result.trajectories:
        assert traj.shape == (sim_params.num_traj, expected_cols)


@pytest.mark.parametrize("two_site_process", ["crosstalk_xx", "lowering_two"])
def test_analog_two_site_jump_operators_smoke(two_site_process: str) -> None:
    """Smoke test: analog TJM runs with single-site plus one adjacent two-site jump process.

    Replaces former QuTiP golden-trajectory integration tests; keeps both crosstalk and
    lowering_two library names exercised at minimal cost.
    """
    length = 2
    hamiltonian = Hamiltonian.ising(length, 1.0, 0.5)
    state = State(length, initial="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=20,
        max_bond_dim=8,
        order=2,
        sample_timesteps=False,
        random_seed=YAQS_TEST_SEED,
    )
    noise = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.02},
        {"name": two_site_process, "sites": [0, 1], "strength": 0.01},
    ])
    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params, noise)

    results = result.expectation_values[0]
    assert results is not None
    z_mean = np.real(results)
    assert np.isfinite(z_mean).all()
    assert np.all(np.abs(z_mean) <= 1.0 + 1e-6)
