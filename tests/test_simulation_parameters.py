import pytest
import numpy as np

from yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams

def test_observable_creation_valid():
    name = 'x'
    site = 0
    obs = Observable(name, site)

    assert obs.name == name
    assert obs.site == site
    assert obs.results is None
    assert obs.trajectories is None

def test_observable_creation_invalid():
    name = 'FakeName'
    with pytest.raises(AttributeError) as exc_info:
        Observable(name, 0)
    # The default message is typically "type object 'GateLibrary' has no attribute 'FakeName'"
    assert "has no attribute" in str(exc_info.value)

def test_custom_observable():
    matrix = np.random.rand(2,2)
    site = 0
    obs = Observable('custom', site, matrix)
    assert obs.name == 'custom'
    assert obs.site == site
    with pytest.raises(AssertionError):
        Observable('x', site, matrix)
    with pytest.raises(AssertionError):
        Observable('custom', site)



def test_physics_simparams_basic():
    obs_list = [Observable('x', 0)]
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

def test_physics_simparams_defaults():
    """
    Test default arguments and typical usage.
    """
    obs_list = []
    T = 2.0
    params = PhysicsSimParams(obs_list, T)

    assert params.observables == obs_list
    assert params.T == 2.0
    assert params.dt == 0.1
    assert params.sample_timesteps is True
    # times => arange(0,2.0+0.1,0.1) => 0, 0.1, 0.2, ..., 2.0
    assert np.isclose(params.times[-1], 2.0)
    assert params.N == 1000
    assert params.max_bond_dim == 2
    assert params.threshold == 1e-6
    assert params.order == 1

def test_observable_initialize_with_sample_timesteps():
    """
    Check shape of 'results' and 'trajectories' when sample_timesteps = True.
    """
    obs = Observable('x', 1)
    sim_params = PhysicsSimParams([obs], T=1.0, dt=0.5, sample_timesteps=True, N=10)
    # sim_params.times => [0.0, 0.5, 1.0]
    # length(sim_params.times) => 3

    obs.initialize(sim_params)
    assert obs.results.shape == (3,), "results should match len(sim_params.times)."
    assert obs.trajectories.shape == (sim_params.N, 3), "trajectories => (N, len(times))."

def test_observable_initialize_without_sample_timesteps():
    """
    Check shape of 'results' and 'trajectories' when sample_timesteps = False.
    """
    obs = Observable('x', 0)
    sim_params = PhysicsSimParams([obs], T=1.0, dt=0.25, sample_timesteps=False, N=5)
    # times => [0.0, 0.25, 0.5, 0.75, 1.0]
    # but if sample_timesteps=False, 'trajectories' => shape (N, 1)
    # and 'results' => shape(len(times))

    obs.initialize(sim_params)
    assert obs.results.shape == (len(sim_params.times),)
    # The code sets self.trajectories => shape (sim_params.N, 1) if not sampling all timesteps
    assert obs.trajectories.shape == (sim_params.N, 1)

    assert obs.times == 1.0, "If sample_timesteps=False, obs.times => float T"
