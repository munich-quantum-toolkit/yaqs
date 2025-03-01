import pytest
from unittest.mock import patch

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs.core.methods.dynamic_TDVP import dynamic_TDVP


def test_dynamic_tdvp_one_site():
    """
    If current_max_bond_dim <= sim_params.max_bond_dim,
    dynamic_TDVP should call two_site_TDVP exactly once.
    """
    # Define the system Hamiltonian
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, J, g)

    # Define the initial state
    state = MPS(L, state="zeros")

    # Define the simulation parameters
    T = 0.2
    dt = 0.1
    sample_timesteps = False
    N = 1
    max_bond_dim = 0
    threshold = 1e-6
    order = 1
    measurements = [Observable("x", site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

    # dynamic_TDVP(state, H_0, sim_params)

    with patch("mqt.yaqs.core.methods.dynamic_TDVP.single_site_TDVP") as mock_single_site:
        dynamic_TDVP(state, H, sim_params)
        mock_single_site.assert_called_once_with(state, H, sim_params)


def test_dynamic_tdvp_two_site():
    """
    If current_max_bond_dim <= sim_params.max_bond_dim,
    dynamic_TDVP should call two_site_TDVP exactly once.
    """
    # Define the system Hamiltonian
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, J, g)

    # Define the initial state
    state = MPS(L, state="zeros")

    # Define the simulation parameters
    T = 0.2
    dt = 0.1
    sample_timesteps = False
    N = 1
    max_bond_dim = 2
    threshold = 1e-6
    order = 1
    measurements = [Observable("x", site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

    # dynamic_TDVP(state, H_0, sim_params)

    with patch("mqt.yaqs.core.methods.dynamic_TDVP.two_site_TDVP") as mock_two_site:
        dynamic_TDVP(state, H, sim_params)
        mock_two_site.assert_called_once_with(state, H, sim_params)
