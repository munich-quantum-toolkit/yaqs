# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Hamiltonian simulation of quantum many-body systems using the Tensor Jump Method (TJM).

This module implements the Tensor Jump Method (TJM) for simulating the dynamics of quantum many-body systems.
It provides functions for initializing the sampling state with noise (via dissipation and stochastic processes),
evolving the state through single-site and two-site TDVP updates, and sampling observable measurements over time.
The functions physics_tjm_2 and physics_tjm_1 correspond to second-order and first-order evolution schemes,
respectively, and return trajectories of expectation values for further analysis.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from ..core.data_structures.simulation_parameters import TensorEvolMode
from ..core.methods.bug import bug
from ..core.methods.dissipation import apply_dissipation
from ..core.methods.dynamic_tdvp import dynamic_tdvp
from ..core.methods.stochastic_process import stochastic_process

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.networks import MPO, MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import PhysicsSimParams


def initialize(state: MPS, noise_model: NoiseModel | None, sim_params: PhysicsSimParams) -> MPS:
    """Initialize the sampling MPS for second-order Trotterization.

    This function prepares the initial sampling MPS (denoted as Phi(0)) by applying a half time step of dissipation
    followed by a stochastic process. It corresponds to F0 in the TJM paper.

    Args:
        state (MPS): The initial state of the system.
        noise_model (NoiseModel | None): The noise model to apply to the system.
        sim_params (PhysicsSimParams): Simulation parameters including the time step (dt).

    Returns:
        MPS: The initialized sampling MPS Phi(0).
    """
    apply_dissipation(state, noise_model, sim_params.dt / 2)
    return stochastic_process(state, noise_model, sim_params.dt)


def step_through(state: MPS, hamiltonian: MPO, noise_model: NoiseModel | None, sim_params: PhysicsSimParams) -> MPS:
    """Perform a single time step evolution of the system state using the TJM.

    Corresponding to Fj in the TJM paper, this function evolves the state by applying dynamic TDVP,
    dissipation, and a stochastic process in sequence.

    Args:
        state (MPS): The current state of the system.
        hamiltonian (MPO): The Hamiltonian operator for the system.
        noise_model (NoiseModel | None): The noise model to apply to the system.
        sim_params (PhysicsSimParams): Simulation parameters including the time step and measurement settings.

    Returns:
        MPS: The updated state after one time step evolution.
    """
    if sim_params.tensorevol_mode == TensorEvolMode.TDVP:
        dynamic_tdvp(state, hamiltonian, sim_params)
    else:
        bug(state, hamiltonian, sim_params)
    apply_dissipation(state, noise_model, sim_params.dt)
    return stochastic_process(state, noise_model, sim_params.dt)


def sample(
    phi: MPS,
    hamiltonian: MPO,
    noise_model: NoiseModel | None,
    sim_params: PhysicsSimParams,
    results: NDArray[np.float64],
    j: int,
) -> None:
    """Sample the quantum state and record observable measurements from the sampling MPS.

    This function evolves a deep copy of the sampling MPS, applies dissipation and a stochastic process,
    and then measures the observables specified in sim_params. The measured values are stored in the provided
    results array at index j (or at index 0 if only one measurement is taken).

    Args:
        phi (MPS): The sampling MPS prior to measurement.
        hamiltonian (MPO): The Hamiltonian operator for the system.
        noise_model (NoiseModel | None): The noise model to apply during evolution.
        sim_params (PhysicsSimParams): Simulation parameters including time step and measurement settings.
        results (NDArray[np.float64]): An array to store the measured observable values.
        j (int): The time step or shot index at which the measurement is recorded.


    """
    psi = copy.deepcopy(phi)
    if sim_params.tensorevol_mode == TensorEvolMode.TDVP:
        dynamic_tdvp(psi, hamiltonian, sim_params)
    else:
        bug(psi, hamiltonian, sim_params)
    apply_dissipation(psi, noise_model, sim_params.dt / 2)
    psi = stochastic_process(psi, noise_model, sim_params.dt)
    if j == len(sim_params.times) - 1 and sim_params.get_state:
        sim_params.output_state = psi

    if sim_params.sample_timesteps:
        temp_state = copy.deepcopy(psi)
        last_site = 0
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            if observable.site > last_site:
                for site in range(last_site, observable.site):
                    temp_state.shift_orthogonality_center_right(site)
                last_site = observable.site
            results[obs_index, j] = temp_state.measure_expectation_value(observable)
    else:
        temp_state = copy.deepcopy(psi)
        last_site = 0
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            if observable.site > last_site:
                for site in range(last_site, observable.site):
                    temp_state.shift_orthogonality_center_right(site)
                last_site = observable.site
            results[obs_index, 0] = temp_state.measure_expectation_value(observable)


def physics_tjm_2(args: tuple[int, MPS, NoiseModel | None, PhysicsSimParams, MPO]) -> NDArray[np.float64]:
    """Run a single trajectory of the TJM using a two-site evolution scheme.

    This function executes a full trajectory by evolving the initial state,
    sampling observable measurements over time, and recording the results.
    It corresponds to the two-site evolution method presented in the TJM paper.

    Args:
        args (tuple): A tuple containing:
            - int: Trajectory identifier.
            - MPS: The initial state of the system.
            - NoiseModel | None: The noise model to be applied (if any).
            - PhysicsSimParams: Simulation parameters (including time step, SVD threshold, etc.).
            - MPO: The Hamiltonian operator represented as an MPO.

    Returns:
        NDArray[np.float64]: An array of expectation values for the trajectory, with dimensions
        determined by the number of observables and time steps.
    """
    _i, initial_state, noise_model, sim_params, hamiltonian = args

    state = copy.deepcopy(initial_state)
    if sim_params.sample_timesteps:
        results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)))
    else:
        results = np.zeros((len(sim_params.sorted_observables), 1))

    if sim_params.sample_timesteps:
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            results[obs_index, 0] = copy.deepcopy(state).measure_expectation_value(observable)

    phi = initialize(state, noise_model, sim_params)
    if sim_params.sample_timesteps:
        sample(phi, hamiltonian, noise_model, sim_params, results, j=1)

    for j, _ in enumerate(sim_params.times[2:], start=2):
        phi = step_through(phi, hamiltonian, noise_model, sim_params)
        if sim_params.sample_timesteps or j == len(sim_params.times) - 1:
            sample(phi, hamiltonian, noise_model, sim_params, results, j)

    return results


def physics_tjm_1(args: tuple[int, MPS, NoiseModel | None, PhysicsSimParams, MPO]) -> NDArray[np.float64]:
    """Run a single trajectory of the TJM using a one-site evolution scheme.

    This function evolves the state with a one-site TDVP update, applying noise (if provided)
    and taking observable measurements over time. It corresponds to the one-site evolution method in the TJM paper.

    Args:
        args (tuple): A tuple containing:
            - int: Trajectory identifier.
            - MPS: The initial state of the system.
            - NoiseModel | None: The noise model to be applied (if any).
            - PhysicsSimParams: Simulation parameters including the time step and measurement settings.
            - MPO: The Hamiltonian operator represented as an MPO.

    Returns:
        NDArray[np.float64]: An array of expectation values for the trajectory, with shape determined
        by the number of observables and time steps.
    """
    _i, initial_state, noise_model, sim_params, hamiltonian = args

    state = copy.deepcopy(initial_state)

    if sim_params.sample_timesteps:
        results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)))
    else:
        results = np.zeros((len(sim_params.sorted_observables), 1))

    if sim_params.sample_timesteps:
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            results[obs_index, 0] = copy.deepcopy(state).measure_expectation_value(observable)

    for j, _ in enumerate(sim_params.times[1:], start=1):
        dynamic_tdvp(state, hamiltonian, sim_params)
        if noise_model is not None:
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
                results[obs_index, j] = temp_state.measure_expectation_value(observable)
        elif j == len(sim_params.times) - 1:
            temp_state = copy.deepcopy(state)
            last_site = 0
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                if observable.site > last_site:
                    for site in range(last_site, observable.site):
                        temp_state.shift_orthogonality_center_right(site)
                    last_site = observable.site
                results[obs_index, 0] = temp_state.measure_expectation_value(observable)

    if sim_params.get_state:
        sim_params.output_state = state

    return results
