# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for deterministic unitary ensemble evolution in analog simulations."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z


def test_unitary_ensemble_observable_average() -> None:
    """List-of-state analog runs should aggregate observables over ensemble members."""
    length = 2
    hamiltonian = MPO.ising(length, J=0.6, g=0.2)
    initial_states = [MPS(length, state="zeros"), MPS(length, state="ones")]

    observable = Observable(Z(), 0)
    sim_params = AnalogSimParams(
        observables=[observable],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        show_progress=False,
    )

    simulator.run(initial_states, hamiltonian, sim_params, noise_model=None, parallel=False)

    assert observable.trajectories is not None
    assert observable.results is not None
    assert observable.trajectories.shape == (len(initial_states), len(sim_params.times))
    np.testing.assert_allclose(observable.results, np.mean(observable.trajectories, axis=0))


def test_unitary_ensemble_autocorrelator_outputs_mean_vector() -> None:
    """Autocorrelator mode should return a complex mean trajectory over state ensemble."""
    length = 2
    hamiltonian = MPO.ising(length, J=0.5, g=0.1)
    initial_states = [MPS(length, state="zeros"), MPS(length, state="ones")]
    correlator_op = Observable(Z(), 0)

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        show_progress=False,
        compute_autocorrelator=True,
        autocorrelator_observable=correlator_op,
    )

    simulator.run(initial_states, hamiltonian, sim_params, noise_model=None, parallel=False)

    assert sim_params.autocorrelator_times is not None
    assert sim_params.autocorrelator_results is not None
    assert sim_params.autocorrelator_results.shape == (len(sim_params.times),)
    assert np.iscomplexobj(sim_params.autocorrelator_results)
    np.testing.assert_allclose(sim_params.autocorrelator_results[0], 1.0 + 0.0j, atol=1e-10)


def test_unitary_ensemble_two_time_correlators_mean_matrix() -> None:
    """Two-time correlator pairs should aggregate to ensemble mean with shape (n_pairs, n_times)."""
    length = 2
    hamiltonian = MPO.ising(length, J=0.2, g=0.1)
    initial_states = [MPS(length, state="zeros"), MPS(length, state="ones")]
    z0 = Observable(Z(), 0)
    z1 = Observable(Z(), 1)
    pairs: list[tuple[Observable, Observable]] = [(z0, z1), (z1, z0)]

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        show_progress=False,
        two_time_correlators=pairs,
    )

    simulator.run(initial_states, hamiltonian, sim_params, noise_model=None, parallel=False)

    assert sim_params.two_time_correlator_times is not None
    assert sim_params.two_time_correlator_results is not None
    assert sim_params.two_time_correlator_results.shape == (len(pairs), len(sim_params.times))
    assert np.iscomplexobj(sim_params.two_time_correlator_results)


def test_list_initial_states_with_noise_raises() -> None:
    """Noisy analog simulation with list[MPS] should fail with explicit guidance."""
    length = 2
    hamiltonian = MPO.ising(length, J=1.0, g=0.2)
    initial_states = [MPS(length, state="zeros"), MPS(length, state="ones")]
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.1}])
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        show_progress=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"list\[MPS\] with noisy analog simulation is not supported yet\."
            r".*list\[MPS\] with no noise.*single MPS for noisy simulation"
        ),
    ):
        simulator.run(initial_states, hamiltonian, sim_params, noise_model=noise_model, parallel=False)
