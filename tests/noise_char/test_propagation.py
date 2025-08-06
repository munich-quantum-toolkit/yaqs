# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the propagation module's noise characterization functionality."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.noise_char import propagation


def test_simulationparameters_set_gammas_float_and_list() -> None:
    """Test the initialization of SimulationParameters with different types of gamma_rel and gamma_deph inputs.

    This test covers the following cases:
    1. When gamma_rel and gamma_deph are provided as floats, they should be expanded to lists of length `n`.
    2. When gamma_rel and gamma_deph are provided as lists, they should be assigned directly.
    3. When gamma_rel or gamma_deph lists have incorrect lengths, a ValueError should be raised.
    """
    # Test with float
    sim = propagation.SimulationParameters(3, 0.1, 0.2)
    assert sim.gamma_rel == [0.1, 0.1, 0.1]
    assert sim.gamma_deph == [0.2, 0.2, 0.2]
    # Test with list
    sim = propagation.SimulationParameters(3, [0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    assert sim.gamma_rel == [0.1, 0.2, 0.3]
    assert sim.gamma_deph == [0.4, 0.5, 0.6]
    # Test with wrong length
    with pytest.raises(ValueError, match=r"must be a list of length L"):
        propagation.SimulationParameters(3, [0.1, 0.2], [0.3, 0.4, 0.5])


def test_simulationparameters_set_gammas_types() -> None:
    """Test that SimulationParameters correctly sets gamma_rel and gamma_deph attributes.

    - When gamma_rel is a scalar and gamma_deph is a list, gamma_rel is broadcast to a list.
    - When gamma_rel is a list and gamma_deph is a scalar, gamma_deph is broadcast to a list.
    Asserts that the attributes are set to the expected list values.
    """
    sim = propagation.SimulationParameters(2, 0.1, [0.2, 0.3])
    assert sim.gamma_rel == [0.1, 0.1]
    assert sim.gamma_deph == [0.2, 0.3]
    sim = propagation.SimulationParameters(2, [0.1, 0.2], 0.3)
    assert sim.gamma_rel == [0.1, 0.2]
    assert sim.gamma_deph == [0.3, 0.3]


def test_tjm_traj_runs() -> None:
    """Test that `propagation.tjm_traj` executes correctly and returns expected output shapes.

    This test verifies that:
    - The function can be called with a valid `SimulationParameters` instance.
    - The returned values `t`, `original_exp_vals`, and `d_on_d_gk` are NumPy arrays.
    - The shapes of the outputs match the expected dimensions based on simulation parameters.
    - The average minimum and maximum trajectory time is returned as a list of None values.
    """
    # Prepare SimulationParameters
    sim_params = propagation.SimulationParameters(2, 0.1, 0.2)
    sim_params.N = 2
    sim_params.threshold = 1e-4
    sim_params.max_bond_dim = 4
    sim_params.order = 1
    sim_params.T = 1
    sim_params.dt = 0.5
    sim_params.L = 2
    t, original_exp_vals, d_on_d_gk, avg_min_max_traj_time = propagation.tjm_traj(sim_params)

    assert isinstance(t, np.ndarray)
    assert isinstance(original_exp_vals, np.ndarray)
    assert isinstance(d_on_d_gk, np.ndarray)

    n_obs_site = 3
    n_jump_sites = 2
    sites = sim_params.L
    n_t = int(sim_params.T / sim_params.dt) + 1

    assert t.shape == (n_t,)
    assert original_exp_vals.shape == (n_obs_site, sites, n_t)
    assert d_on_d_gk.shape == (n_jump_sites, n_obs_site, sites, n_t)
    assert avg_min_max_traj_time == [None, None, None]
