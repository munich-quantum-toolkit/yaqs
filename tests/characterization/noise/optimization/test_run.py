# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for trajectory-matching orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, State
from mqt.yaqs.characterization.noise.optimization.run import run_optimization_characterization
from mqt.yaqs.characterization.noise.optimization.trajectories import simulate_observable_trajectories
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.core.parallel_utils import ExecutionConfig


def _homogeneous_pauli_noise_model(sites: list[int], strength: float) -> NoiseModel:
    """Build a homogeneous Pauli noise model on the given sites.

    Args:
        sites: Qubit/site indices receiving independent Pauli channels.
        strength: Shared jump rate for every channel.

    Returns:
        Noise model with ``pauli_x``, ``pauli_y``, and ``pauli_z`` on each site.
    """
    return NoiseModel(
        [{"name": "pauli_x", "sites": [s], "strength": strength} for s in sites]
        + [{"name": "pauli_y", "sites": [s], "strength": strength} for s in sites]
        + [{"name": "pauli_z", "sites": [s], "strength": strength} for s in sites]
    )


def _digital_twin_setup() -> tuple[
    Hamiltonian,
    State,
    list[Observable],
    AnalogSimParams,
    NoiseModel,
    NoiseModel,
    np.ndarray,
]:
    """Build a three-site digital-twin benchmark with simulated experimental data.

    Returns:
        Tuple of Hamiltonian, initial state, fitting observables, simulation
        parameters, reference model, initial guess, and reference trajectories.
    """
    n_sites = 3
    gamma_true = 0.08
    sites = list(range(n_sites))
    hamiltonian = Hamiltonian.ising(n_sites, J=1.0, g=2.0)
    init_state = State(n_sites, initial="zeros")
    fitting_observables = [Observable(g(), s) for s in range(n_sites) for g in (X, Y, Z)]
    sim_params = AnalogSimParams(
        observables=fitting_observables,
        elapsed_time=0.8,
        dt=0.1,
        order=1,
        sample_timesteps=True,
    )
    reference_model = _homogeneous_pauli_noise_model(sites, gamma_true)
    init_guess = _homogeneous_pauli_noise_model(sites, 0.35)
    experimental_data, _, _ = simulate_observable_trajectories(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        noise_model=reference_model,
        observables=fitting_observables,
        representation="density_matrix",
    )
    return (
        hamiltonian,
        init_state,
        fitting_observables,
        sim_params,
        reference_model,
        init_guess,
        experimental_data,
    )


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_run_optimization_characterization_three_site_digital_twin() -> None:
    """3-site Lindblad fit from experimental trajectories recovers rates and dynamics."""
    (
        hamiltonian,
        init_state,
        fitting_observables,
        sim_params,
        _reference_model,
        init_guess,
        experimental_data,
    ) = _digital_twin_setup()

    n_params = len(init_guess.processes)
    result = run_optimization_characterization(
        hamiltonian=hamiltonian,
        sim_params=sim_params,
        init_state=init_state,
        init_guess=init_guess,
        observables=fitting_observables,
        ref_expectations=experimental_data,
        x_low=np.zeros(n_params),
        x_up=np.full(n_params, 0.5),
        execution=ExecutionConfig(parallel=False, show_progress=False),
        representation="density_matrix",
        sigma0=0.05,
        popsize=12,
        max_iter=100,
        seed=42,
    )

    assert result.trajectory_rmse() < 1e-2
    gamma_true = 0.08
    n_sites = 3
    for channel in range(3):
        learned_mean = float(result.best_parameters[channel * n_sites : (channel + 1) * n_sites].mean())
        rel_err = abs(learned_mean - gamma_true) / gamma_true
        assert rel_err < 0.05
