import numpy as np

from yaqs.library.tensor_library import TensorLibrary


class Observable:
    def __init__(self, name: str, site: int):
        assert getattr(TensorLibrary, name), "Selected observable to measure does not exist."
        self.name = name
        self.site = site
        self.results = None
        self.trajectories = None

    def initialize(self, sim_params: 'SimulationParams'):
        if sim_params.sample_timesteps:
            self.trajectories = np.empty((sim_params.N, len(sim_params.times)), dtype=float)
        else:
            self.trajectories = np.empty((sim_params.N, 1), dtype=float)
        self.results = np.empty(len(sim_params.times), dtype=float)
        if sim_params.sample_timesteps:
            self.times = sim_params.times
        else:
            self.times = sim_params.T

class SimulationParams:
    def __init__(self, observables: list[Observable], T: float, dt: float=0.1, sample_timesteps: bool=True, N: int=1000, max_bond_dim: int=2, threshold: float=1e-6, order: int=1):
        self.observables = observables
        self.T = T
        self.dt = dt
        self.times = np.arange(0, T+dt, dt)
        self.sample_timesteps = sample_timesteps
        self.N = N
        self.max_bond_dim = max_bond_dim
        self.threshold = threshold
        self.order = order
