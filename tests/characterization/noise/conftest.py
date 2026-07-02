# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared fixtures for noise characterization tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, State
from mqt.yaqs.characterization.noise.shared.propagation import Propagator
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.libraries.gate_library import X, Y, Z


@dataclass
class NoiseTestConfig:
    """Lightweight open-system configuration for noise characterization tests."""

    sites: int = 1
    sim_time: float = 0.6
    dt: float = 0.2
    order: int = 1
    threshold: float = 1e-4
    ntraj: int = 1
    max_bond_dim: int = 4
    j: float = 1.0
    g: float = 0.5
    gamma_x: float = 0.1
    gamma_y: float = 0.12
    gamma_z: float = 0.15

    @property
    def times(self) -> np.ndarray:
        """Simulation time grid used by the test configuration."""
        return np.arange(0, self.sim_time + self.dt, self.dt)

    @property
    def n_obs(self) -> int:
        """Number of tracked observables (three Pauli components per site)."""
        return self.sites * 3

    @property
    def n_t(self) -> int:
        """Number of sampled time points."""
        return len(self.times)


@pytest.fixture
def noise_test_config() -> NoiseTestConfig:
    """Default open-system geometry for noise characterization tests.

    Returns:
        Shared :class:`NoiseTestConfig` instance for parametrized smoke tests.
    """
    return NoiseTestConfig()


def build_propagator(
    test: NoiseTestConfig,
) -> tuple[
    Hamiltonian,
    State,
    list[Observable],
    AnalogSimParams,
    CompactNoiseModel,
    Propagator,
]:
    """Construct a configured propagator for the shared test geometry.

    Returns:
        Tuple of Hamiltonian, initial state, observables, simulation parameters,
        compact noise model, and propagator.
    """
    hamiltonian = Hamiltonian.ising(test.sites, J=test.j, g=test.g)
    init_state = State(test.sites, initial="zeros")
    observables = (
        [Observable(X(), site) for site in range(test.sites)]
        + [Observable(Y(), site) for site in range(test.sites)]
        + [Observable(Z(), site) for site in range(test.sites)]
    )
    sim_params = AnalogSimParams(
        observables=observables,
        elapsed_time=test.sim_time,
        dt=test.dt,
        num_traj=test.ntraj,
        max_bond_dim=test.max_bond_dim,
        svd_threshold=test.threshold,
        order=test.order,
        sample_timesteps=True,
    )
    noise_model = CompactNoiseModel([
        {"name": "pauli_x", "sites": list(range(test.sites)), "strength": test.gamma_x},
        {"name": "pauli_y", "sites": list(range(test.sites)), "strength": test.gamma_y},
        {"name": "pauli_z", "sites": list(range(test.sites)), "strength": test.gamma_z},
    ])
    propagator = Propagator(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        compact_noise_model=noise_model,
        init_state=init_state,
    )
    propagator.set_observable_list(observables)
    return hamiltonian, init_state, observables, sim_params, noise_model, propagator
