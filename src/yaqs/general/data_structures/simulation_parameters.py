import numpy as np

from yaqs.general.libraries.gate_library import GateLibrary


class Observable:
    def __init__(self, name: str, site: int):
        assert getattr(GateLibrary, name), "Selected observable to measure does not exist."
        self.name = name
        self.site = site
        self.results = None
        self.trajectories = None

    def initialize(self, sim_params):
        if type(sim_params) == PhysicsSimParams:
            if sim_params.sample_timesteps:
                self.trajectories = np.empty((sim_params.N, len(sim_params.times)), dtype=float)
                self.times = sim_params.times
            else:
                self.trajectories = np.empty((sim_params.N, 1), dtype=float)
                self.times = sim_params.T
            self.results = np.empty(len(sim_params.times), dtype=float)
        elif type(sim_params) == WeakSimParams:
            self.trajectories = np.empty((sim_params.N, 1), dtype=float)
            self.results = np.empty(1, dtype=float)
        elif type(sim_params) == StrongSimParams:
            self.trajectories = np.empty((sim_params.N, 1), dtype=float)
            self.results = np.empty(1, dtype=float)


class PhysicsSimParams:
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

    def aggregate_trajectories(self):
        for observable in self.observables:
            observable.results = np.mean(observable.trajectories, axis=0)


class WeakSimParams:
    def __init__(self, shots: int, max_bond_dim: int=2, threshold: float=1e-6, window_size: int=None):
        self.measurements = shots*[None]
        self.shots = shots
        self.max_bond_dim = max_bond_dim
        self.threshold = threshold
        self.window_size = window_size

    def aggregate_measurements(self):
        self.results = {}
        # Noise-free simulation stores shots in first element
        if None in self.measurements:
            self.results = self.measurements[0]
            self.results = dict(sorted(self.results.items()))

        else:
            for d in self.measurements:
                for key, value in d.items():
                    self.results[key] = self.results.get(key, 0) + value
            self.results = dict(sorted(self.results.items()))


class StrongSimParams:
    def __init__(self, observables: list[Observable], N: int=1000, max_bond_dim: int=2, threshold: float=1e-6, window_size: int=None):
        self.observables = observables
        self.N = N
        self.max_bond_dim = max_bond_dim
        self.threshold = threshold
        self.window_size = window_size

    def aggregate_trajectories(self):
            for observable in self.observables:
                observable.results = np.mean(observable.trajectories, axis=0)