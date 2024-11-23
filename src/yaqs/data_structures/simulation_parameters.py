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
        self.trajectories = np.empty((sim_params.N, len(sim_params.times)), dtype=float)
        self.results = np.empty(len(sim_params.times), dtype=float)

class SimulationParams:
    def __init__(self, observables: list[Observable], T: float, dt: float, N: int, max_bond_dim: int, threshold: float):
        self.observables = observables
        self.T = T
        self.dt = dt
        self.times = np.arange(0, T+dt, dt)
        self.N = N
        self.max_bond_dim = max_bond_dim
        self.threshold = threshold
