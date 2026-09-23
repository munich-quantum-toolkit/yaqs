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
from mqt.yaqs.characterization.noise.optimization import run as optimization_run
from mqt.yaqs.characterization.noise.optimization.run import run_optimization_characterization
from mqt.yaqs.characterization.noise.optimization.trajectories import simulate_observable_trajectories
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
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
    fitting_observables = [Observable(name, site) for site in range(n_sites) for name in ("x", "y", "z")]
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


def test_invalid_optimizer_bounds_fail_before_forward_model_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bound-count mismatch fails before simulation or optimizer dispatch."""
    hamiltonian = Hamiltonian.ising(1, J=0.0, g=0.0)
    init_state = State(1, initial="zeros")
    observables = [Observable("z", 0)]
    sim_params = AnalogSimParams(observables=observables, elapsed_time=0.1, dt=0.1)
    init_guess = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 0.1}])

    def fail_if_called(_execution: ExecutionConfig) -> None:
        pytest.fail("Forward-model setup ran before optimizer input validation.")

    monkeypatch.setattr(optimization_run, "build_simulator", fail_if_called)
    with pytest.raises(ValueError, match="x_low shape"):
        run_optimization_characterization(
            hamiltonian=hamiltonian,
            sim_params=sim_params,
            init_state=init_state,
            init_guess=init_guess,
            observables=observables,
            ref_expectations=np.zeros((1, 2)),
            x_low=np.zeros(2),
            x_up=np.ones(1),
            execution=ExecutionConfig(parallel=False, show_progress=False),
        )


def test_mutated_time_controls_are_rebuilt_before_reference_shape_validation() -> None:
    """Noise fitting validates reference data against the current analog time grid."""
    hamiltonian = Hamiltonian.ising(1, J=0.0, g=0.0)
    init_state = State(1, initial="zeros")
    observables = [Observable("z", 0)]
    sim_params = AnalogSimParams(observables=observables, elapsed_time=0.1, dt=0.1)
    sim_params.elapsed_time = 0.2
    init_guess = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 0.1}])

    with pytest.raises(ValueError, match="2 columns but sim_params defines 3"):
        run_optimization_characterization(
            hamiltonian=hamiltonian,
            sim_params=sim_params,
            init_state=init_state,
            init_guess=init_guess,
            observables=observables,
            ref_expectations=np.zeros((1, 2)),
            x_low=np.zeros(1),
            x_up=np.ones(1),
            execution=ExecutionConfig(parallel=False, show_progress=False),
        )

    np.testing.assert_allclose(sim_params.times, [0.0, 0.1, 0.2])


@pytest.mark.release
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


def test_run_optimization_characterization_two_site_crosstalk() -> None:
    """Adjacent two-site crosstalk processes fit via the optimization pipeline."""
    n_sites = 3
    gamma_true = 0.08
    hamiltonian = Hamiltonian.ising(n_sites, J=1.0, g=2.0)
    init_state = State(n_sites, initial="zeros")
    fitting_observables = [Observable("z", 0), Observable("z", 1)]
    sim_params = AnalogSimParams(
        observables=fitting_observables,
        elapsed_time=0.6,
        dt=0.1,
        order=1,
        sample_timesteps=True,
    )
    reference_model = NoiseModel([
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": gamma_true},
        {"name": "crosstalk_xx", "sites": [1, 2], "strength": gamma_true},
    ])
    init_guess = NoiseModel([
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": 0.25},
        {"name": "crosstalk_xx", "sites": [1, 2], "strength": 0.25},
    ])
    experimental_data, _, _ = simulate_observable_trajectories(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        noise_model=reference_model,
        observables=fitting_observables,
        representation="density_matrix",
    )

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
        max_iter=60,
        seed=7,
    )

    assert result.trajectory_rmse() < 5e-2
    np.testing.assert_allclose(result.best_parameters, gamma_true, rtol=0.15, atol=0.02)
