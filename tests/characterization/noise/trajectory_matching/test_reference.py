# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for trajectory-matching reference helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, State
from mqt.yaqs.characterization.noise.trajectory_matching.reference import (
    build_trajectory_loss,
    resolve_reference_expectations,
    simulate_observable_trajectories,
)
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.core.parallel_utils import ExecutionConfig


def _three_site_problem() -> tuple[
    Hamiltonian,
    State,
    list[Observable],
    AnalogSimParams,
    CompactNoiseModel,
]:
    n_sites = 3
    hamiltonian = Hamiltonian.ising(n_sites, J=1.0, g=2.0)
    init_state = State(n_sites, initial="zeros")
    observables = [Observable(g(), s) for s in range(n_sites) for g in (X, Y, Z)]
    sim_params = AnalogSimParams(
        observables=observables,
        elapsed_time=0.8,
        dt=0.1,
        order=1,
        sample_timesteps=True,
    )
    reference_model = CompactNoiseModel([
        {"name": "pauli_x", "sites": list(range(n_sites)), "strength": 0.08},
        {"name": "pauli_y", "sites": list(range(n_sites)), "strength": 0.08},
        {"name": "pauli_z", "sites": list(range(n_sites)), "strength": 0.08},
    ])
    return hamiltonian, init_state, observables, sim_params, reference_model


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_simulate_observable_trajectories_shape() -> None:
    """Simulation helper returns trajectories with the expected shape."""
    hamiltonian, init_state, observables, sim_params, reference_model = _three_site_problem()
    expectations, times, resolved = simulate_observable_trajectories(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        noise_model=reference_model,
        observables=observables,
        representation="density_matrix",
    )
    assert resolved == "density_matrix"
    assert expectations.shape == (len(observables), len(times))


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_ref_expectations_path_matches_simulation() -> None:
    """Precomputed expectations are accepted when shapes match the fitting set."""
    hamiltonian, init_state, observables, sim_params, reference_model = _three_site_problem()
    execution = ExecutionConfig(parallel=False, show_progress=False)
    from mqt.yaqs.characterization.noise.trajectory_matching.reference import build_simulator

    simulator = build_simulator(execution)
    simulated, times, _ = resolve_reference_expectations(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        observables=observables,
        reference_model=reference_model,
        ref_expectations=None,
        simulator=simulator,
        representation="density_matrix",
        lindblad_max_qubits=8,
        vector_max_qubits=10,
    )
    accepted, accepted_times, _ = resolve_reference_expectations(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        observables=observables,
        reference_model=None,
        ref_expectations=simulated,
        simulator=simulator,
        representation="density_matrix",
        lindblad_max_qubits=8,
        vector_max_qubits=10,
    )
    np.testing.assert_allclose(accepted, simulated)
    np.testing.assert_allclose(accepted_times, times)


def test_resolve_reference_requires_exactly_one_source() -> None:
    """Reference resolution rejects missing or duplicate sources."""
    hamiltonian, init_state, observables, sim_params, reference_model = _three_site_problem()
    execution = ExecutionConfig(parallel=False, show_progress=False)
    from mqt.yaqs.characterization.noise.trajectory_matching.reference import build_simulator

    simulator = build_simulator(execution)
    with pytest.raises(ValueError, match="exactly one"):
        resolve_reference_expectations(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            init_state=init_state,
            observables=observables,
            reference_model=reference_model,
            ref_expectations=np.zeros((1, 1)),
            simulator=simulator,
            representation="density_matrix",
            lindblad_max_qubits=8,
            vector_max_qubits=10,
        )


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_build_trajectory_loss_wires_propagator() -> None:
    """Loss assembly returns a propagator sharing the fitting topology."""
    hamiltonian, init_state, observables, sim_params, reference_model = _three_site_problem()
    init_guess = CompactNoiseModel([
        {"name": "pauli_x", "sites": list(range(3)), "strength": 0.35},
        {"name": "pauli_y", "sites": list(range(3)), "strength": 0.35},
        {"name": "pauli_z", "sites": list(range(3)), "strength": 0.35},
    ])
    ref, _, _ = simulate_observable_trajectories(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        noise_model=reference_model,
        observables=observables,
        representation="density_matrix",
    )
    from mqt.yaqs.characterization.noise.trajectory_matching.reference import build_simulator

    simulator = build_simulator(ExecutionConfig(parallel=False, show_progress=False))
    loss, _propagator, resolved = build_trajectory_loss(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        init_guess=init_guess,
        observables=observables,
        ref_expectations=ref,
        simulator=simulator,
        representation="density_matrix",
        lindblad_max_qubits=8,
        vector_max_qubits=10,
    )
    assert resolved == "density_matrix"
    np.testing.assert_allclose(loss.ref_traj_array, ref)
    loss_value, _, _ = loss(init_guess.strength_list)
    assert loss_value >= 0.0
