# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from typing import cast, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

class Observable:
    def __init__(self, name: str, site: int) -> None:
        from ..libraries.gate_library import GateLibrary

        assert getattr(GateLibrary, name)
        self.name = name
        self.site = site
        self.results: Optional[NDArray[np.float64]] = None
        self.trajectories: Optional[NDArray[np.float64]] = None

    def initialize(self, sim_params: PhysicsSimParams | StrongSimParams | WeakSimParams) -> None:
        if type(sim_params) == PhysicsSimParams:
            if sim_params.sample_timesteps:
                self.trajectories = np.empty((sim_params.N, len(sim_params.times)), dtype=np.float64)
                self.times = sim_params.times
            else:
                self.trajectories = np.empty((sim_params.N, 1), dtype=np.float64)
                self.times = sim_params.T
            self.results = np.empty(len(sim_params.times), dtype=np.float64)
        elif type(sim_params) == WeakSimParams:
            self.trajectories = np.empty((sim_params.shots, 1), dtype=np.complex128)
            self.results = np.empty(1, dtype=np.float64)
        elif type(sim_params) == StrongSimParams:
            self.trajectories = np.empty((sim_params.N, 1), dtype=np.complex128)
            self.results = np.empty(1, dtype=np.float64)


class PhysicsSimParams:
    def __init__(
        self,
        observables: list[Observable],
        T: float,
        dt: float = 0.1,
        sample_timesteps: bool = True,
        N: int = 1000,
        max_bond_dim: int = 2,
        threshold: float = 1e-6,
        order: int = 1,
    ) -> None:

        self.observables = observables
        self.sorted_observables = sorted(observables, key=lambda obs: (obs.site, obs.name))
        self.T = T
        self.dt = dt
        self.times = np.arange(0, T + dt, dt)
        self.sample_timesteps = sample_timesteps
        self.N = N
        self.max_bond_dim = max_bond_dim
        self.threshold = threshold
        self.order = order

    def aggregate_trajectories(self) -> None:
        for observable in self.observables:
            observable.results = np.mean(observable.trajectories, axis=0)


class WeakSimParams:
    # Properties set as placeholders for code compatability
    dt = 1
    N = 0

    def __init__(
        self, shots: int, max_bond_dim: int = 2, threshold: float = 1e-6, window_size: int | None = None
    ) -> None:
        self.measurements: list[dict[int, int] | None] = cast("list[Optional[dict[int, int]]]", shots * [None])
        self.shots = shots
        self.max_bond_dim = max_bond_dim
        self.threshold = threshold
        self.window_size = window_size

    def aggregate_measurements(self) -> None:
        self.results: dict[int, int] = {}
        # Noise-free simulation stores shots in first element
        if None in self.measurements:
            assert self.measurements[0] is not None
            self.results = self.measurements[0]
            self.results = dict(sorted(self.results.items()))

        else:
            for d in filter(None, self.measurements):
                for key, value in d.items():
                    self.results[key] = self.results.get(key, 0) + value
            self.results = dict(sorted(self.results.items()))


class StrongSimParams:
    dt = 1

    def __init__(
        self,
        observables: list[Observable],
        N: int = 1000,
        max_bond_dim: int = 2,
        threshold: float = 1e-6,
        window_size: int | None = None,
    ) -> None:

        self.observables = observables
        self.sorted_observables = sorted(observables, key=lambda obs: (obs.site, obs.name))
        self.N = N
        self.max_bond_dim = max_bond_dim
        self.threshold = threshold
        self.window_size = window_size

    def aggregate_trajectories(self) -> None:
        for observable in self.observables:
            observable.results = np.mean(observable.trajectories, axis=0)
