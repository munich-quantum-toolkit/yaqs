# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

""" Tests for simulation parameters classes

This module contains unit tests for the Observable and PhysicsSimParams classes used in
quantum simulation. It verifies that:
  - An Observable is correctly initialized with valid parameters and that invalid parameters
    raise an appropriate error.
  - PhysicsSimParams instances are created with the correct attributes (such as T, dt, times,
    sample_timesteps, and N) both with explicit and default values.
  - The Observable.initialize method properly sets up the results and trajectories arrays
    depending on whether sample_timesteps is True or False.
"""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams


def test_observable_creation_valid() -> None:
    """Test that an Observable is created correctly with valid parameters.

    This test constructs an Observable with the name "x" on site 0 and verifies that its attributes
    (name, site, results, and trajectories) are correctly initialized.
    """
    name = "x"
    site = 0
    obs = Observable(name, site)

    assert obs.name == name
    assert obs.site == site
    assert obs.results is None
    assert obs.trajectories is None


def test_observable_creation_invalid() -> None:
    """Test that creating an Observable with an invalid name raises an AttributeError.

    This test attempts to create an Observable with a name not supported by the GateLibrary,
    expecting an AttributeError to be raised.
    """
    name = "FakeName"
    with pytest.raises(AttributeError) as exc_info:
        Observable(name, 0)
    # The default error message should indicate that the GateLibrary has no attribute 'FakeName'
    assert "has no attribute" in str(exc_info.value)


def test_physics_simparams_basic() -> None:
    """Test that PhysicsSimParams is initialized with correct parameters.

    This test creates a PhysicsSimParams instance with a single observable, total time T, time step dt,
    sample_timesteps flag set to True, and a specified number of trajectories N. It then verifies that the
    observables, T, dt, times array, sample_timesteps flag, and N are set correctly.
    """
    obs_list = [Observable("x", 0)]
    T = 1.0
    dt = 0.2
    params = PhysicsSimParams(obs_list, T, dt=dt, sample_timesteps=True, N=50)

    assert params.observables == obs_list
    assert params.T == T
    assert params.dt == dt
    expected_times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(params.times, expected_times), "Times array should match numpy.arange(0, T+dt, dt)."
    assert params.sample_timesteps is True
    assert params.N == 50


def test_physics_simparams_defaults() -> None:
    """Test the default parameters for PhysicsSimParams.

    This test constructs a PhysicsSimParams instance with an empty observable list and total time T,
    and verifies that default values for dt, sample_timesteps, number of trajectories (N), max_bond_dim,
    threshold, and order are correctly assigned.
    """
    obs_list: list[Observable] = []
    T = 2.0
    params = PhysicsSimParams(obs_list, T)

    assert params.observables == obs_list
    assert params.T == 2.0
    assert params.dt == 0.1
    assert params.sample_timesteps is True
    # times should be np.arange(0, T+dt, dt)
    assert np.isclose(params.times[-1], 2.0)
    assert params.N == 1000
    assert params.max_bond_dim == 2
    assert params.threshold == 1e-6
    assert params.order == 1


def test_observable_initialize_with_sample_timesteps() -> None:
    """Test that Observable.initialize sets up results and trajectories correctly when sample_timesteps is True.

    This test creates an Observable on site 1 and a PhysicsSimParams instance with sample_timesteps=True.
    It verifies that the results array has shape equal to the length of the times array and that the
    trajectories array has shape (N, len(times)).
    """
    obs = Observable("x", 1)
    sim_params = PhysicsSimParams([obs], T=1.0, dt=0.5, sample_timesteps=True, N=10)
    # sim_params.times => [0.0, 0.5, 1.0]

    obs.initialize(sim_params)
    assert obs.results is not None
    assert obs.trajectories is not None
    assert obs.results.shape == (3,), "results should match len(sim_params.times)."
    assert obs.trajectories.shape == (sim_params.N, 3), "trajectories should have shape (N, len(times))."


def test_observable_initialize_without_sample_timesteps() -> None:
    """Test that Observable.initialize sets up results and trajectories correctly when sample_timesteps is False.

    This test creates an Observable on site 0 and a PhysicsSimParams instance with sample_timesteps=False.
    It verifies that the results array has shape equal to the length of the times array, the trajectories array
    has shape (N, 1), and that the observable's times attribute is set to T.
    """
    obs = Observable("x", 0)
    sim_params = PhysicsSimParams([obs], T=1.0, dt=0.25, sample_timesteps=False, N=5)
    # times => [0.0, 0.25, 0.5, 0.75, 1.0]

    obs.initialize(sim_params)
    assert obs.results is not None
    assert obs.trajectories is not None
    assert obs.results.shape == (len(sim_params.times),)
    assert obs.trajectories.shape == (sim_params.N, 1)
    assert obs.times == 1.0, "If sample_timesteps=False, obs.times should be equal to T."
