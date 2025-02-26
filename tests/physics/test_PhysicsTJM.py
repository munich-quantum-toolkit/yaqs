import pytest
from unittest.mock import patch

from yaqs.core.data_structures.networks import MPO, MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from yaqs.physics.PhysicsTJM import initialize, step_through, sample, PhysicsTJM_2, PhysicsTJM_1


def test_initialize():
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, J, g)
    state = MPS(L)
    noise_model = NoiseModel(['relaxation'], [0.1])
    measurements = [Observable('x', site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T=0.2, dt=0.2, sample_timesteps=False, N=1, max_bond_dim=2, threshold=1e-6, order=1)
    with patch('yaqs.physics.PhysicsTJM.apply_dissipation') as mock_dissipation, \
         patch('yaqs.physics.PhysicsTJM.stochastic_process') as mock_stochastic_process:
        initialize(state, noise_model, sim_params)
        mock_dissipation.assert_called_once_with(state, noise_model, sim_params.dt/2)
        mock_stochastic_process.assert_called_once_with(state, noise_model, sim_params.dt)

def test_step_through():
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, J, g)
    state = MPS(L)
    noise_model = NoiseModel(['relaxation'], [0.1])
    measurements = [Observable('x', site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T=0.2, dt=0.2, sample_timesteps=False, N=1, max_bond_dim=2, threshold=1e-6, order=1)
    with patch('yaqs.physics.PhysicsTJM.dynamic_TDVP') as mock_dynamic_TDVP, \
         patch('yaqs.physics.PhysicsTJM.apply_dissipation') as mock_dissipation, \
         patch('yaqs.physics.PhysicsTJM.stochastic_process') as mock_stochastic_process:
        step_through(state, H, noise_model, sim_params)
        mock_dynamic_TDVP(state, H, sim_params)
        mock_dissipation.assert_called_once_with(state, noise_model, sim_params.dt)
        mock_stochastic_process.assert_called_once_with(state, noise_model, sim_params.dt)

def test_PhysicsTJM_2():
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable('z', site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T=0.2, dt=0.1, sample_timesteps=False, N=1, max_bond_dim=4, threshold=1e-6, order=2)
    args = (0, state, noise_model, sim_params, H)
    results = PhysicsTJM_2(args)
    # When sample_timesteps is True, results should have shape (num_observables, len(times))
    assert results.shape == (len(measurements), 1), "Results incorrect shape"

def test_PhysicsTJM_2_sample_timesteps():
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable('z', site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T=0.2, dt=0.1, sample_timesteps=True, N=1, max_bond_dim=4, threshold=1e-6, order=2)
    args = (0, state, noise_model, sim_params, H)
    results = PhysicsTJM_2(args)
    # When sample_timesteps is True, results should have shape (num_observables, len(times))
    assert results.shape == (len(measurements), len(sim_params.times)), "Results incorrect shape"

def test_PhysicsTJM_1():
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable('z', site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T=0.2, dt=0.1, sample_timesteps=False, N=1, max_bond_dim=4, threshold=1e-6, order=1)
    args = (0, state, noise_model, sim_params, H)
    results = PhysicsTJM_1(args)
    # When sample_timesteps is True, results should have shape (num_observables, len(times))
    assert results.shape == (len(measurements), 1), "Results incorrect shape"

def test_PhysicsTJM_1_sample_timesteps():
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable('z', site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T=0.2, dt=0.1, sample_timesteps=True, N=1, max_bond_dim=4, threshold=1e-6, order=1)
    args = (0, state, noise_model, sim_params, H)
    results = PhysicsTJM_1(args)
    # When sample_timesteps is True, results should have shape (num_observables, len(times))
    assert results.shape == (len(measurements), len(sim_params.times)), "Results incorrect shape"
