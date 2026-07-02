# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for trajectory-matching orchestration."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, State
from mqt.yaqs.characterization.noise.trajectory_matching.reference import simulate_observable_trajectories
from mqt.yaqs.characterization.noise.trajectory_matching.run import (
    run_trajectory_characterization,
    simulate_fit_trajectory,
)
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.core.parallel_utils import ExecutionConfig


def _digital_twin_setup() -> tuple[
    Hamiltonian,
    State,
    list[Observable],
    AnalogSimParams,
    CompactNoiseModel,
    CompactNoiseModel,
    np.ndarray,
]:
    n_sites = 3
    gamma_true = 0.08
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
    reference_model = CompactNoiseModel([
        {"name": "pauli_x", "sites": list(range(n_sites)), "strength": gamma_true},
        {"name": "pauli_y", "sites": list(range(n_sites)), "strength": gamma_true},
        {"name": "pauli_z", "sites": list(range(n_sites)), "strength": gamma_true},
    ])
    init_guess = CompactNoiseModel([
        {"name": "pauli_x", "sites": list(range(n_sites)), "strength": 0.35},
        {"name": "pauli_y", "sites": list(range(n_sites)), "strength": 0.35},
        {"name": "pauli_z", "sites": list(range(n_sites)), "strength": 0.35},
    ])
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
def test_run_trajectory_characterization_three_site_digital_twin() -> None:
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

    result = run_trajectory_characterization(
        hamiltonian=hamiltonian,
        sim_params=sim_params,
        init_state=init_state,
        init_guess=init_guess,
        observables=fitting_observables,
        ref_expectations=experimental_data,
        x_low=np.zeros(3),
        x_up=np.full(3, 0.5),
        execution=ExecutionConfig(parallel=False, show_progress=False),
        representation="density_matrix",
        sigma0=0.05,
        popsize=8,
        max_iter=40,
        seed=42,
    )

    assert result.resolved_representation == "density_matrix"
    assert result.trajectory_rmse() < 2e-3
    gamma_true = 0.08
    for learned in result.best_parameters:
        rel_err = abs(float(learned) - gamma_true) / gamma_true
        assert rel_err < 0.05


def test_run_rejects_unsupported_optimizer() -> None:
    """Only the CMA backend is wired in the orchestration layer."""
    hamiltonian, init_state, fitting_observables, sim_params, reference_model, init_guess, experimental = (
        _digital_twin_setup()
    )
    with pytest.raises(ValueError, match="optimizer must be 'cma'"):
        run_trajectory_characterization(
            hamiltonian=hamiltonian,
            sim_params=sim_params,
            init_state=init_state,
            init_guess=init_guess,
            observables=fitting_observables,
            ref_expectations=experimental,
            x_low=np.zeros(3),
            x_up=np.full(3, 0.5),
            execution=ExecutionConfig(parallel=False, show_progress=False),
            representation="density_matrix",
            optimizer=cast("Any", "adam"),
        )
    _ = reference_model


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_simulate_fit_trajectory_helper() -> None:
    """simulate_fit_trajectory delegates to the reference simulator helper."""
    hamiltonian, init_state, fitting_observables, sim_params, reference_model, init_guess, _ = _digital_twin_setup()
    expectations, times = simulate_fit_trajectory(
        hamiltonian=hamiltonian,
        sim_params=sim_params,
        init_state=init_state,
        noise_model=reference_model,
        observables=fitting_observables,
        execution=ExecutionConfig(parallel=False, show_progress=False),
        representation="density_matrix",
    )
    assert expectations.shape == (len(fitting_observables), len(times))
    _ = init_guess
