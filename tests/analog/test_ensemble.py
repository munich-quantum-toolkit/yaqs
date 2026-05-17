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
from mqt.yaqs.analog.ensemble import ensemble_member_worker
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, EvolutionMode, Observable
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import Z


def test_unitary_ensemble_observable_average() -> None:
    """Aggregate observables over ensemble members in list-of-state analog runs."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.6, g=0.2)
    initial_states = [State(length, initial="zeros"), State(length, initial="ones")]

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


def test_unitary_ensemble_autocorrelator_outputs_mean_matrix_row() -> None:
    """Autocorrelation (O,O) pair yields a ``(1, n_times)`` ensemble-mean result."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.5, g=0.1)
    initial_states = [State(length, initial="zeros"), State(length, initial="ones")]
    correlator_op = Observable(Z(), 0)

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        show_progress=False,
        multi_time_observables=[(correlator_op, correlator_op)],
    )

    simulator.run(initial_states, hamiltonian, sim_params, noise_model=None, parallel=False)

    assert sim_params.multi_time_observables_times is not None
    assert sim_params.multi_time_observables_results is not None
    assert sim_params.multi_time_observables_results.shape == (1, len(sim_params.times))
    assert np.iscomplexobj(sim_params.multi_time_observables_results)
    np.testing.assert_allclose(sim_params.multi_time_observables_results[0, 0], 1.0 + 0.0j, atol=1e-10)


def test_unitary_ensemble_multi_time_observables_mean_matrix() -> None:
    """Aggregate multi_time_observables pairs to an ensemble-mean matrix."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    initial_states = [State(length, initial="zeros"), State(length, initial="ones")]
    z0 = Observable(Z(), 0)
    z1 = Observable(Z(), 1)
    pairs: list[tuple[Observable, Observable]] = [(z0, z1), (z1, z0)]

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        show_progress=False,
        multi_time_observables=pairs,
    )

    simulator.run(initial_states, hamiltonian, sim_params, noise_model=None, parallel=False)

    assert sim_params.multi_time_observables_times is not None
    assert sim_params.multi_time_observables_results is not None
    assert sim_params.multi_time_observables_results.shape == (len(pairs), len(sim_params.times))
    assert np.iscomplexobj(sim_params.multi_time_observables_results)


def test_unitary_ensemble_t0_only_records_when_not_sampling_timesteps() -> None:
    """When only ``t=0`` exists and sampling is off, observable/correlators are still recorded."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    initial_states = [State(length, initial="zeros")]
    z0 = Observable(Z(), 0)
    z1 = Observable(Z(), 1)

    sim_params = AnalogSimParams(
        observables=[z0],
        elapsed_time=0.0,
        dt=0.1,
        sample_timesteps=False,
        show_progress=False,
        multi_time_observables=[(z0, z0), (z0, z1)],
    )

    simulator.run(initial_states, hamiltonian, sim_params, noise_model=None, parallel=False)

    assert z0.results is not None
    assert z0.results.shape == (1,)
    np.testing.assert_allclose(z0.results[0], 1.0, atol=1e-10)

    assert sim_params.multi_time_observables_results is not None
    assert sim_params.multi_time_observables_results.shape == (2, 1)
    # (Z0, Z0) autocorrelator at t=0: <0|Z0^2|0> = 1
    np.testing.assert_allclose(sim_params.multi_time_observables_results[0, 0], 1.0 + 0.0j, atol=1e-10)


def test_unitary_ensemble_clears_multi_time_outputs_when_feature_disabled() -> None:
    """Reusing ``sim_params`` should clear prior multi_time_observables outputs when feature is off."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    initial_states = [State(length, initial="zeros"), State(length, initial="ones")]
    z0 = Observable(Z(), 0)
    z1 = Observable(Z(), 1)

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        show_progress=False,
        multi_time_observables=[(z0, z0), (z0, z1)],
    )

    simulator.run(initial_states, hamiltonian, sim_params, noise_model=None, parallel=False)
    assert sim_params.multi_time_observables_results is not None
    assert sim_params.multi_time_observables_times is not None

    sim_params_off = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        show_progress=False,
    )
    # Seed stale fields to verify they are cleared on next run.
    sim_params_off.multi_time_observables_times = np.array([0.0], dtype=np.float64)
    sim_params_off.multi_time_observables_results = np.array([[1.0 + 0.0j]], dtype=np.complex128)

    simulator.run(initial_states, hamiltonian, sim_params_off, noise_model=None, parallel=False)
    assert sim_params_off.multi_time_observables_results is None
    assert sim_params_off.multi_time_observables_times is None


def test_list_mps_analog_ensemble_rejects_non_mps_representation() -> None:
    """List-of-MPS analog ensemble only supports the mps representation path."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    states = [
        State(length, initial="zeros", representation="density_matrix"),
        State(length, initial="ones", representation="density_matrix"),
    ]
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        show_progress=False,
    )
    with pytest.raises(
        ValueError, match=r"list\[State\] analog ensemble currently supports only State\.representation='mps'\."
    ):
        simulator.run(states, hamiltonian, sim_params, noise_model=None, parallel=False)


def test_list_mps_analog_ensemble_rejects_empty_state_list() -> None:
    """Empty list[MPS] must fail before evolution."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        show_progress=False,
    )
    with pytest.raises(ValueError, match="initial_state list must not be empty"):
        simulator.run([], hamiltonian, sim_params, noise_model=None, parallel=False)


def test_list_mps_analog_ensemble_rejects_state_length_mismatch() -> None:
    """All ensemble MPS chain lengths must match the Hamiltonian MPO length."""
    hamiltonian = Hamiltonian.ising(2, J=0.2, g=0.1)
    states = [State(3, initial="zeros"), State(3, initial="ones")]
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        show_progress=False,
    )
    with pytest.raises(ValueError, match=r"State\.length=3 does not match Hamiltonian\.length=2"):
        simulator.run(states, hamiltonian, sim_params, noise_model=None, parallel=False)


def test_list_mps_analog_ensemble_rejects_get_state() -> None:
    """get_state is not supported together with list[MPS] analog ensemble mode."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    states = [State(length, initial="zeros"), State(length, initial="ones")]
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        show_progress=False,
        get_state=True,
    )
    with pytest.raises(ValueError, match="get_state=True is not supported for list\\[State\\] analog ensemble mode"):
        simulator.run(states, hamiltonian, sim_params, noise_model=None, parallel=False)


def test_list_mps_unitary_ensemble_parallel_worker_path() -> None:
    """parallel=True with multiple members exercises the process-pool ensemble worker."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    states = [State(length, initial="zeros"), State(length, initial="ones")]
    z0 = Observable(Z(), 0)
    z1 = Observable(Z(), 1)
    sim_params = AnalogSimParams(
        observables=[z0],
        elapsed_time=0.15,
        dt=0.05,
        show_progress=False,
        multi_time_observables=[(z0, z0), (z0, z1)],
    )
    simulator.run(states, hamiltonian, sim_params, noise_model=None, parallel=True)
    assert z0.results is not None
    assert sim_params.multi_time_observables_results is not None


def test_unitary_ensemble_member_worker_uses_bug_evolution_mode() -> None:
    """BUG tensor evolution should exercise the non-TDVP branch in ``_unitary_step``."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    mps = MPS(length, state="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.05,
        dt=0.05,
        show_progress=False,
        evolution_mode=EvolutionMode.BUG,
        max_bond_dim=64,
        threshold=1e-10,
    )
    obs_result, multi_time = ensemble_member_worker((0, mps, sim_params, hamiltonian.mpo))
    assert obs_result.shape == (1, len(sim_params.times))
    assert multi_time is None


def test_unitary_ensemble_member_worker_final_timestep_when_not_sampling() -> None:
    """With ``sample_timesteps=False`` and multiple time slices, record correlators on the last step."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    mps = MPS(length, state="zeros")
    z0 = Observable(Z(), 0)
    z1 = Observable(Z(), 1)
    sim_params = AnalogSimParams(
        observables=[z0],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=False,
        show_progress=False,
        multi_time_observables=[(z0, z0), (z0, z1)],
        max_bond_dim=64,
        threshold=1e-10,
    )
    assert len(sim_params.times) >= 3
    obs_result, multi_time = ensemble_member_worker((0, mps, sim_params, hamiltonian.mpo))
    assert obs_result.shape == (1, 1)
    assert multi_time is not None
    assert multi_time.shape == (2, 1)


def test_list_initial_states_with_noise_raises() -> None:
    """Raise an explicit error for noisy analog runs with list[MPS]."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.2)
    initial_states = [State(length, initial="zeros"), State(length, initial="ones")]
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
            r"(?s)list\[State\] with noisy analog simulation is not supported yet\."
            r".*list\[State\] with no noise.*single State for noisy simulation"
        ),
    ):
        simulator.run(initial_states, hamiltonian, sim_params, noise_model=noise_model, parallel=False)
