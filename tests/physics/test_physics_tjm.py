# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Hamiltonian simulation with the Tensor Jump Method.

This module provides unit tests for the physics simulation functions implemented in the
PhysicsTJM module. It verifies that the initialization and time evolution routines for
the Tensor Jump Method (TJM) work as expected in various configurations, including both
first and second order evolution schemes, with and without timestep sampling.

The tests cover:
  - Initialization: Ensuring that a half time step of dissipation followed by a stochastic process
    is correctly applied to the initial state.
  - Step-through evolution: Verifying that dynamic_tdvp, apply_dissipation, and stochastic_process
    are called with the proper arguments during a single time step.
  - Physics simulation (order=2): Checking the shape of the results when running a second order evolution,
    with and without sampling timesteps.
  - Physics simulation (order=1): Checking the shape of the results when running a first order evolution,
    with and without sampling timesteps.

These tests ensure that the evolution functions correctly integrate the MPS state under the
specified Hamiltonian and noise model, and that observable measurements are properly aggregated.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

from unittest.mock import patch

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs.core.libraries.gate_library import X, Z
from mqt.yaqs.physics.physics_tjm import initialize, physics_tjm_1, physics_tjm_2, step_through


def test_initialize() -> None:
    """Test that initialize applies a half-time dissipation and then a stochastic process to the MPS.

    This test creates an Ising MPO and an MPS of length 5, along with a minimal NoiseModel and PhysicsSimParams.
    It patches the functions apply_dissipation and stochastic_process to ensure that initialize calls them with the
    correct arguments: apply_dissipation should be called with dt/2, and stochastic_process with dt.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = NoiseModel(["relaxation"], [0.1])
    measurements = [Observable(X(), site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.2,
        sample_timesteps=False,
        num_traj=1,
        max_bond_dim=2,
        threshold=1e-6,
        order=1,
    )
    with (
        patch("mqt.yaqs.physics.physics_tjm.apply_dissipation") as mock_dissipation,
        patch("mqt.yaqs.physics.physics_tjm.stochastic_process") as mock_stochastic_process,
    ):
        initialize(state, noise_model, sim_params)
        mock_dissipation.assert_called_once_with(state, noise_model, sim_params.dt / 2)
        mock_stochastic_process.assert_called_once_with(state, noise_model, sim_params.dt)


def test_step_through() -> None:
    """Test that step_through calls dynamic_tdvp, apply_dissipation, and stochastic_process with correct arguments.

    This test creates an Ising MPO and an MPS of length 5, along with a minimal NoiseModel and PhysicsSimParams.
    It patches dynamic_tdvp, apply_dissipation, and stochastic_process to ensure that step_through calls each of them
    correctly: dynamic_tdvp should be called with the state, H, and sim_params, and both apply_dissipation and
    stochastic_process should be called with dt.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = NoiseModel(["relaxation"], [0.1])
    measurements = [Observable(X(), site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.2,
        sample_timesteps=False,
        num_traj=1,
        max_bond_dim=2,
        threshold=1e-6,
        order=1,
    )
    with (
        patch("mqt.yaqs.physics.physics_tjm.local_dynamic_tdvp") as mock_dynamic_tdvp,
        patch("mqt.yaqs.physics.physics_tjm.apply_dissipation") as mock_dissipation,
        patch("mqt.yaqs.physics.physics_tjm.stochastic_process") as mock_stochastic_process,
    ):
        step_through(state, H, noise_model, sim_params)
        mock_dynamic_tdvp(state, H, sim_params)
        mock_dissipation.assert_called_once_with(state, noise_model, sim_params.dt)
        mock_stochastic_process.assert_called_once_with(state, noise_model, sim_params.dt)


def test_physics_tjm_2() -> None:
    """Test the physics_tjm_2 function for a two-site evolution (order=2) without sampling timesteps.

    This test creates an Ising MPO and an MPS of length 5, with no noise model.
    It calls physics_tjm_2 with sim_params configured for order 2 and sample_timesteps False.
    The returned results array should have shape (num_observables, 1).
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=False,
        num_traj=1,
        max_bond_dim=4,
        threshold=1e-6,
        order=2,
    )
    args = (0, state, noise_model, sim_params, H)
    results = physics_tjm_2(args)
    assert results.shape == (len(measurements), 1), "Results incorrect shape"


def test_physics_tjm_2_sample_timesteps() -> None:
    """Test the physics_tjm_2 function for a two-site evolution (order=2) with sampling timesteps.

    This test creates an Ising MPO and an MPS of length 5, with no noise model,
    and sim_params with sample_timesteps True. The resulting results array should have shape
    (num_observables, len(sim_params.times)).
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=4,
        threshold=1e-6,
        order=2,
    )
    args = (0, state, noise_model, sim_params, H)
    results = physics_tjm_2(args)
    assert results.shape == (len(measurements), len(sim_params.times)), "Results incorrect shape"


def test_physics_tjm_1() -> None:
    """Test the physics_tjm_1 function for a one-site evolution (order=1) without sampling timesteps.

    This test creates an Ising MPO and an MPS of length 5, with no noise model,
    and sim_params with order 1 and sample_timesteps False.
    The resulting results array should have shape (num_observables, 1).
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=False,
        num_traj=1,
        max_bond_dim=4,
        threshold=1e-6,
        order=1,
    )
    args = (0, state, noise_model, sim_params, H)
    results = physics_tjm_1(args)
    assert results.shape == (len(measurements), 1), "Results incorrect shape"


def test_physics_tjm_1_sample_timesteps() -> None:
    """Test the physics_tjm_1 function for a one-site evolution (order=1) with sampling timesteps.

    This test creates an Ising MPO and an MPS of length 5, with no noise model,
    and sim_params with sample_timesteps True.
    The results array should have shape (num_observables, len(sim_params.times)).
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=4,
        threshold=1e-6,
        order=1,
    )
    args = (0, state, noise_model, sim_params, H)
    results = physics_tjm_1(args)
    assert results.shape == (len(measurements), len(sim_params.times)), "Results incorrect shape"
