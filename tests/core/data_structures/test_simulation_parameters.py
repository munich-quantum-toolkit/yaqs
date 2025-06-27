# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for simulation parameters classes.

This module contains unit tests for the Observable and PhysicsSimParams classes used in
quantum simulation. It verifies that:
  - An Observable is correctly initialized with valid parameters and that invalid parameters
    raise an appropriate error.
  - PhysicsSimParams instances are created with the correct attributes (such as elapsed_time, dt, times,
    sample_timesteps, and num_traj) both with explicit and default values.
  - The Observable.initialize method properly sets up the results and trajectories arrays
    depending on whether sample_timesteps is True or False.
"""

# ignore non-lowercase variable names for physics notation

from __future__ import annotations

import numpy as np

from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs.core.libraries.gate_library import X


def test_observable_creation_valid() -> None:
    """Test that an Observable is created correctly with valid parameters.

    This test constructs an Observable with the name "x" on site 0 and verifies that its attributes
    (name, site, results, and trajectories) are correctly initialized.
    """
    gate = X()
    site = 0
    obs = Observable(gate, site)

    assert np.array_equal(obs.gate.matrix, np.array([[0, 1], [1, 0]]))
    assert obs.sites == site
    assert obs.results is None
    assert obs.trajectories is None


def test_physics_simparams_basic() -> None:
    """Test that PhysicsSimParams is initialized with correct parameters.

    This test creates a PhysicsSimParams instance with a single observable, total time elapsed_time, time step dt,
    sample_timesteps flag set to True, and a specified number of trajectories num_traj. It then verifies that the
    observables, elapsed_time, dt, times array, sample_timesteps flag, and num_traj are set correctly.
    """
    obs_list = [Observable(X(), 0)]
    elapsed_time = 1.0
    dt = 0.2
    params = PhysicsSimParams(obs_list, elapsed_time, dt=dt, sample_timesteps=True, num_traj=50)

    assert params.observables == obs_list
    assert params.elapsed_time == elapsed_time
    assert params.dt == dt
    expected_times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(params.times, expected_times), "Times array should match numpy.arange(0, elapsed_time+dt, dt)."
    assert params.sample_timesteps is True
    assert params.num_traj == 50


def test_physics_simparams_defaults() -> None:
    """Test the default parameters for PhysicsSimParams.

    This test constructs a PhysicsSimParams instance with an empty observable list and total time elapsed_time,
    and verifies that default values for dt, sample_timesteps, number of trajectories (num_traj), max_bond_dim,
    threshold, and order are correctly assigned.
    """
    obs_list: list[Observable] = []
    elapsed_time = 2.0
    params = PhysicsSimParams(obs_list, elapsed_time)

    assert params.observables == obs_list
    assert params.elapsed_time == 2.0
    assert params.dt == 0.1
    assert params.sample_timesteps is True
    # times should be np.arange(0, elapsed_time+dt, dt)
    assert np.isclose(params.times[-1], 2.0)
    assert params.num_traj == 1000
    assert params.max_bond_dim == 4096
    assert params.threshold == 1e-9
    assert params.order == 1


def test_observable_initialize_with_sample_timesteps() -> None:
    """Test that Observable.initialize sets up results and trajectories correctly when sample_timesteps is True.

    This test creates an Observable on site 1 and a PhysicsSimParams instance with sample_timesteps=True.
    It verifies that the results array has shape equal to the length of the times array and that the
    trajectories array has shape (num_traj, len(times)).
    """
    obs = Observable(X(), 1)
    sim_params = PhysicsSimParams([obs], elapsed_time=1.0, dt=0.5, sample_timesteps=True, num_traj=10)
    # sim_params.times => [0.0, 0.5, 1.0]

    obs.initialize(sim_params)
    assert obs.results is not None
    assert obs.trajectories is not None
    assert obs.results.shape == (3,), "results should match len(sim_params.times)."
    assert obs.trajectories.shape == (sim_params.num_traj, 3), "trajectories should have shape (num_traj, len(times))."


def test_observable_initialize_without_sample_timesteps() -> None:
    """Test that Observable.initialize sets up results and trajectories correctly when sample_timesteps is False.

    This test creates an Observable on site 0 and a PhysicsSimParams instance with sample_timesteps=False.
    It verifies that the results array has shape equal to the length of the times array, the trajectories array
    has shape (num_traj, 1), and that the observable's times attribute is set to elapsed_time.
    """
    obs = Observable(X(), 0)
    sim_params = PhysicsSimParams([obs], elapsed_time=1.0, dt=0.25, sample_timesteps=False, num_traj=5)
    # times => [0.0, 0.25, 0.5, 0.75, 1.0]

    obs.initialize(sim_params)
    assert obs.results is not None
    assert obs.trajectories is not None
    assert obs.results.shape == (len(sim_params.times),)
    assert obs.trajectories.shape == (sim_params.num_traj, 1)
    assert obs.times == 1.0, "If sample_timesteps=False, obs.times should be equal to elapsed_time."
