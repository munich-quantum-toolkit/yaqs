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
from mqt.yaqs.characterization.noise.optimization.trajectories import (
    build_simulator,
    build_trajectory_loss,
    resolve_prepared_state,
    resolve_reference_expectations,
    simulate_observable_trajectories,
)
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.parallel_utils import ExecutionConfig


def _three_site_problem() -> tuple[
    Hamiltonian,
    State,
    list[Observable],
    AnalogSimParams,
    NoiseModel,
]:
    n_sites = 3
    sites = list(range(n_sites))
    hamiltonian = Hamiltonian.ising(n_sites, J=1.0, g=2.0)
    init_state = State(n_sites, initial="zeros")
    observables = [Observable(name, site) for site in range(n_sites) for name in ("x", "y", "z")]
    sim_params = AnalogSimParams(
        observables=observables,
        elapsed_time=0.8,
        dt=0.1,
        order=1,
        sample_timesteps=True,
    )
    reference_model = NoiseModel(
        [{"name": "pauli_x", "sites": [s], "strength": 0.08} for s in sites]
        + [{"name": "pauli_y", "sites": [s], "strength": 0.08} for s in sites]
        + [{"name": "pauli_z", "sites": [s], "strength": 0.08} for s in sites]
    )
    return hamiltonian, init_state, observables, sim_params, reference_model


def test_simulate_observable_trajectories_shape() -> None:
    """Simulation helper returns trajectories with the expected shape."""
    hamiltonian, init_state, observables, sim_params, reference_model = _three_site_problem()
    expectations, times, prepared = simulate_observable_trajectories(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        noise_model=reference_model,
        observables=observables,
        representation="density_matrix",
    )
    assert prepared.representation == "density_matrix"
    assert expectations.shape == (len(observables), len(times))


def test_seeded_observable_trajectories_match_in_serial_and_parallel() -> None:
    """Noise-fitting forward trajectories agree with automatic process-pool execution."""
    hamiltonian, init_state, observables, _sim_params, noise = _three_site_problem()
    params = AnalogSimParams(
        observables=observables,
        elapsed_time=0.2,
        dt=0.1,
        num_traj=8,
        random_seed=42,
        order=1,
    )
    results = [
        simulate_observable_trajectories(
            sim_params=params,
            hamiltonian=hamiltonian,
            init_state=init_state,
            noise_model=noise,
            observables=observables,
            simulator=build_simulator(ExecutionConfig(parallel=parallel, max_workers=2, show_progress=False)),
            representation="mps",
        )
        for parallel in (False, True)
    ]
    serial, parallel = results
    np.testing.assert_array_equal(parallel[1], serial[1])
    np.testing.assert_allclose(parallel[0], serial[0], atol=1e-12)


def test_ref_expectations_path_matches_simulation() -> None:
    """Precomputed expectations are accepted when shapes match the fitting set."""
    hamiltonian, init_state, observables, sim_params, reference_model = _three_site_problem()
    execution = ExecutionConfig(parallel=False, show_progress=False)

    simulator = build_simulator(execution)
    simulated, times, _prepared = resolve_reference_expectations(
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


def test_resolve_reference_requires_a_source() -> None:
    """Reference resolution rejects when neither source is provided."""
    hamiltonian, init_state, observables, sim_params, _reference_model = _three_site_problem()
    execution = ExecutionConfig(parallel=False, show_progress=False)

    simulator = build_simulator(execution)
    with pytest.raises(ValueError, match="exactly one"):
        resolve_reference_expectations(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            init_state=init_state,
            observables=observables,
            reference_model=None,
            ref_expectations=None,
            simulator=simulator,
            representation="density_matrix",
            lindblad_max_qubits=8,
            vector_max_qubits=10,
        )


def test_resolve_reference_requires_exactly_one_source() -> None:
    """Reference resolution rejects missing or duplicate sources."""
    hamiltonian, init_state, observables, sim_params, reference_model = _three_site_problem()
    execution = ExecutionConfig(parallel=False, show_progress=False)

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


def test_ref_expectations_shape_validation() -> None:
    """Precomputed expectations must match observable and time dimensions."""
    hamiltonian, init_state, observables, sim_params, _reference_model = _three_site_problem()
    execution = ExecutionConfig(parallel=False, show_progress=False)

    simulator = build_simulator(execution)
    with pytest.raises(ValueError, match="2-D"):
        resolve_reference_expectations(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            init_state=init_state,
            observables=observables,
            reference_model=None,
            ref_expectations=np.zeros(3),
            simulator=simulator,
            representation="density_matrix",
            lindblad_max_qubits=8,
            vector_max_qubits=10,
        )
    with pytest.raises(ValueError, match="rows"):
        resolve_reference_expectations(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            init_state=init_state,
            observables=observables,
            reference_model=None,
            ref_expectations=np.zeros((1, len(sim_params.times))),
            simulator=simulator,
            representation="density_matrix",
            lindblad_max_qubits=8,
            vector_max_qubits=10,
        )
    with pytest.raises(ValueError, match="columns"):
        resolve_reference_expectations(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            init_state=init_state,
            observables=observables,
            reference_model=None,
            ref_expectations=np.zeros((len(observables), 1)),
            simulator=simulator,
            representation="density_matrix",
            lindblad_max_qubits=8,
            vector_max_qubits=10,
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_ref_expectations_rejects_non_finite_values(invalid: float) -> None:
    """Experimental reference trajectories must contain finite observations."""
    hamiltonian, init_state, observables, sim_params, _reference_model = _three_site_problem()
    execution = ExecutionConfig(parallel=False, show_progress=False)
    reference = np.zeros((len(observables), len(sim_params.times)))
    reference[0, 0] = invalid

    with pytest.raises(ValueError, match="only finite values"):
        resolve_reference_expectations(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            init_state=init_state,
            observables=observables,
            reference_model=None,
            ref_expectations=reference,
            simulator=build_simulator(execution),
            representation="density_matrix",
            lindblad_max_qubits=8,
            vector_max_qubits=10,
        )


def test_resolve_prepared_state_encodes_density_matrix() -> None:
    """resolve_prepared_state returns an encoded state for Lindblad."""
    hamiltonian, init_state, _, _, _ = _three_site_problem()
    prepared = resolve_prepared_state(
        hamiltonian,
        init_state,
        "density_matrix",
        lindblad_max_qubits=8,
        vector_max_qubits=10,
    )
    assert prepared.representation == "density_matrix"


def test_build_trajectory_loss_wires_propagator() -> None:
    """Loss assembly returns a propagator sharing the fitting topology."""
    hamiltonian, init_state, observables, sim_params, reference_model = _three_site_problem()
    sites = list(range(3))
    init_guess = NoiseModel(
        [{"name": "pauli_x", "sites": [s], "strength": 0.35} for s in sites]
        + [{"name": "pauli_y", "sites": [s], "strength": 0.35} for s in sites]
        + [{"name": "pauli_z", "sites": [s], "strength": 0.35} for s in sites]
    )
    ref, _, prepared = simulate_observable_trajectories(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        noise_model=reference_model,
        observables=observables,
        representation="density_matrix",
    )
    simulator = build_simulator(ExecutionConfig(parallel=False, show_progress=False))
    loss, _propagator = build_trajectory_loss(
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
        prepared_state=prepared,
    )
    assert prepared.representation == "density_matrix"
    np.testing.assert_allclose(loss.ref_traj_array, ref)
    loss_value = loss(np.array([proc["strength"] for proc in init_guess.processes], dtype=float))
    assert loss_value >= 0.0
