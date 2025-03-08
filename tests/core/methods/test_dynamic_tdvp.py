# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from unittest.mock import patch

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs.core.methods.dynamic_TDVP import dynamic_TDVP


def test_dynamic_tdvp_one_site() -> None:
    """Test that dynamic_TDVP calls single_site_TDVP exactly once when the current maximum bond dimension
    exceeds sim_params.max_bond_dim.

    In this test, sim_params.max_bond_dim is set to 0 so that the current maximum bond dimension of the MPS,
    computed by state.write_max_bond_dim(), is greater than 0. Therefore, the else branch of dynamic_TDVP should be
    taken, and single_site_TDVP should be called exactly once.
    """
    # Define the system Hamiltonian.
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)

    # Define the initial state.
    state = MPS(L, state="zeros")

    # Define simulation parameters with max_bond_dim set to 0.
    T = 0.2
    dt = 0.1
    sample_timesteps = False
    N = 1
    max_bond_dim = 0  # Force condition for single_site_TDVP.
    threshold = 1e-6
    order = 1
    measurements = [Observable("x", site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

    with patch("mqt.yaqs.core.methods.dynamic_TDVP.single_site_TDVP") as mock_single_site:
        dynamic_TDVP(state, H, sim_params)
        mock_single_site.assert_called_once_with(state, H, sim_params)


def test_dynamic_tdvp_two_site() -> None:
    """Test that dynamic_TDVP calls two_site_TDVP exactly once when the current maximum bond dimension
    is less than or equal to sim_params.max_bond_dim.

    In this test, sim_params.max_bond_dim is set to 2, so if the current maximum bond dimension is ≤ 2,
    the if branch of dynamic_TDVP is executed and two_site_TDVP is called exactly once.
    """
    # Define the system Hamiltonian.
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)

    # Define the initial state.
    state = MPS(L, state="zeros")

    # Define simulation parameters with max_bond_dim set to 2.
    T = 0.2
    dt = 0.1
    sample_timesteps = False
    N = 1
    max_bond_dim = 2  # Force condition for two_site_TDVP.
    threshold = 1e-6
    order = 1
    measurements = [Observable("x", site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

    with patch("mqt.yaqs.core.methods.dynamic_TDVP.two_site_TDVP") as mock_two_site:
        dynamic_TDVP(state, H, sim_params)
        mock_two_site.assert_called_once_with(state, H, sim_params)
