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
    with pytest.raises(ValueError):
        propagation.SimulationParameters(3, [0.1, 0.2], [0.3, 0.4, 0.5])


def test_simulationparameters_set_gammas_types() -> None:
    """Test that SimulationParameters correctly sets gamma_rel and gamma_deph attributes
    when provided with scalar or list inputs.
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


def test_tjm_traj_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the `tjm_traj` function from the `propagation` module to ensure it runs
    without errors and returns expected shapes. This test verifies that:
    - The function can be called with a SimulationParameters instance.
    - The returned time array `t`, original expectation values `original_exp_vals`,
      and derivatives `d_On_d_gk` have the expected shapes.
    - The average minimum and maximum trajectory time is returned as a list with None values.
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
    t, original_exp_vals, d_On_d_gk, avg_min_max_traj_time = propagation.tjm_traj(sim_params)

    assert isinstance(t, np.ndarray)
    assert isinstance(original_exp_vals, np.ndarray)
    assert isinstance(d_On_d_gk, np.ndarray)

    n_obs_site = 3
    n_jump_sites = 2
    L = sim_params.L
    n_t = int(sim_params.T / sim_params.dt) + 1

    assert t.shape == (n_t,)
    assert original_exp_vals.shape == (n_obs_site, L, n_t)
    assert d_On_d_gk.shape == (n_jump_sites, n_obs_site, L, n_t)
    assert avg_min_max_traj_time == [None, None, None]
