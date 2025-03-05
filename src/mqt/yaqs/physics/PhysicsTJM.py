# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from ..core.methods.dissipation import apply_dissipation
from ..core.methods.dynamic_TDVP import dynamic_TDVP
from ..core.methods.stochastic_process import stochastic_process

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.networks import MPO, MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import PhysicsSimParams


def initialize(state: MPS, noise_model: NoiseModel, sim_params: PhysicsSimParams) -> MPS:
    """Initialize the sampling MPS for second-order Trotterization.
    Corresponds to F0 in the TJM paper.

    Args:
        state (MPS): Initial state of the system.
        noise_model (NoiseModel): Noise model to apply to the system.
        sim_params (SimulationParams): Simulation parameters, including time step.

    Returns:
        MPS: The initialized sampling MPS Phi(0).
    """
    apply_dissipation(state, noise_model, sim_params.dt / 2)
    return stochastic_process(state, noise_model, sim_params.dt)


def step_through(state: MPS, H: MPO, noise_model: NoiseModel, sim_params: PhysicsSimParams) -> MPS:
    """Perform a single time step of the TJM of the system state.
    Corresponds to Fj in the TJM paper.

    Args:
        state (MPS): Current state of the system.
        H (MPO): Hamiltonian operator for the system.
        noise_model (NoiseModel): Noise model to apply to the system.
        sim_params (SimulationParams): Simulation parameters, including time step and max bond dimension.

    Returns:
        MPS: The updated state after performing the time step evolution.
    """
    dynamic_TDVP(state, H, sim_params)
    apply_dissipation(state, noise_model, sim_params.dt)
    return stochastic_process(state, noise_model, sim_params.dt)


def sample(
    phi: MPS, H: MPO, noise_model: NoiseModel, sim_params: PhysicsSimParams, results: NDArray[np.float64], j: int
) -> None:
    """Sample the quantum state and measure an observable from the sampling MPS.
    Corresponds to Fn in the TJM paper.

    Args:
        phi (MPS): State of the system before sampling.
        H (MPO): Hamiltonian operator for the system.
        noise_model (NoiseModel): Noise model to apply to the system.
        sim_params (SimulationParams): Simulation parameters, including time step and measurements.

    Returns:
        MPS: The measured observable value.
    """
    psi = copy.deepcopy(phi)
    dynamic_TDVP(psi, H, sim_params)
    apply_dissipation(psi, noise_model, sim_params.dt / 2)
    psi = stochastic_process(psi, noise_model, sim_params.dt)
    if sim_params.sample_timesteps:
        temp_state = copy.deepcopy(psi)
        last_site = 0
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            if observable.site > last_site:
                for site in range(last_site, observable.site):
                    temp_state.shift_orthogonality_center_right(site)
                last_site = observable.site
            results[obs_index, j] = temp_state.measure(observable)
    else:
        temp_state = copy.deepcopy(psi)
        last_site = 0
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            if observable.site > last_site:
                for site in range(last_site, observable.site):
                    temp_state.shift_orthogonality_center_right(site)
                last_site = observable.site
            results[obs_index, 0] = temp_state.measure(observable)


def PhysicsTJM_2(args):
    """Run a single trajectory of the TJM.

    Args:
        args (tuple): Tuple containing index, initial state, noise model, simulation parameters, observables, sites, and Hamiltonian.

    Returns:
        list: Expectation values for the trajectory over time.
    """
    _i, initial_state, noise_model, sim_params, H = args

    # Create deep copies of the shared inputs to avoid race conditions
    state = copy.deepcopy(initial_state)
    if sim_params.sample_timesteps:
        results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)))
    else:
        results = np.zeros((len(sim_params.sorted_observables), 1))

    if sim_params.sample_timesteps:
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            results[obs_index, 0] = copy.deepcopy(state).measure(observable)

    phi = initialize(state, noise_model, sim_params)
    if sim_params.sample_timesteps:
        sample(phi, H, noise_model, sim_params, results, j=1)

    for j, _ in enumerate(sim_params.times[2:], start=2):
        phi = step_through(phi, H, noise_model, sim_params)
        if sim_params.sample_timesteps or j == len(sim_params.times) - 1:
            sample(phi, H, noise_model, sim_params, results, j)

    return results


def PhysicsTJM_1(args):
    _i, initial_state, noise_model, sim_params, H = args
    state = copy.deepcopy(initial_state)

    if sim_params.sample_timesteps:
        results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)))
    else:
        results = np.zeros((len(sim_params.sorted_observables), 1))

    if sim_params.sample_timesteps:
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            results[obs_index, 0] = copy.deepcopy(state).measure(observable)

    for j, _ in enumerate(sim_params.times[1:], start=1):
        dynamic_TDVP(state, H, sim_params)
        if noise_model:
            apply_dissipation(state, noise_model, sim_params.dt)
            state = stochastic_process(state, noise_model, sim_params.dt)
        if sim_params.sample_timesteps:
            temp_state = copy.deepcopy(state)
            last_site = 0
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                if observable.site > last_site:
                    for site in range(last_site, observable.site):
                        temp_state.shift_orthogonality_center_right(site)
                    last_site = observable.site
                results[obs_index, j] = temp_state.measure(observable)
        elif j == len(sim_params.times) - 1:
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                results[obs_index, 0] = copy.deepcopy(state).measure(observable)

    return results
