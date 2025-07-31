import pytest
import numpy as np

from mqt.yaqs.noise_char import propagation


def test_simulationparameters_set_gammas_float_and_list():
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

def test_simulationparameters_set_gammas_types():
    sim = propagation.SimulationParameters(2, 0.1, [0.2, 0.3])
    assert sim.gamma_rel == [0.1, 0.1]
    assert sim.gamma_deph == [0.2, 0.3]
    sim = propagation.SimulationParameters(2, [0.1, 0.2], 0.3)
    assert sim.gamma_rel == [0.1, 0.2]
    assert sim.gamma_deph == [0.3, 0.3]



def test_tjm_traj_runs(monkeypatch):
    
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