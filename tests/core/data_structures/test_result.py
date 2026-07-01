# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the :class:`~mqt.yaqs.core.data_structures.result.Result` container.

These tests verify that ``Result`` holds simulation outputs separately from the
read-only ``*SimParams`` configuration object and can be pickled.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

import pickle  # noqa: S403  # test-only: controlled Result round-trip; no untrusted input deserialized

import numpy as np

from mqt.yaqs import (
    AnalogSimParams,
    Hamiltonian,
    NoiseModel,
    Observable,
    Result,
    Simulator,
    State,
    StrongSimParams,
    WeakSimParams,
)
from mqt.yaqs.core.data_structures.result import aggregate_counts
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import Z


def test_result_holds_outputs_for_analog_run() -> None:
    """Result exposes observables, output_state, and noise_model from the run."""
    length = 2
    state = State(length, initial="zeros")
    H = Hamiltonian.ising(length, J=1.0, g=0.5)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=1,
        get_state=True,
        sample_timesteps=False,
    )

    result = Simulator(parallel=False, show_progress=False).run(state, H, sim_params)

    assert isinstance(result, Result)
    assert result.sim_params is sim_params
    assert result.observables is not sim_params.observables
    assert len(result.observables) == 1
    assert len(result.expectation_values) == 1
    assert len(result.trajectories) == 1
    assert result.output_state is not None
    assert result.noise_model is None
    assert result.counts is None
    assert result.multi_time_times is None
    assert result.multi_time_results is None
    assert result.runtime_cost is not None
    assert result.max_bond is not None
    assert result.total_bond is not None
    assert result.times is not None
    assert len(result.runtime_cost) == len(result.times)


def test_result_counts_only_set_for_weak_simulation() -> None:
    """Result.counts is populated for weak simulations and None otherwise."""
    num_qubits = 2
    state = State(num_qubits, initial="zeros")
    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=1)
    circuit.measure_all()

    weak_params = WeakSimParams(shots=16, max_bond_dim=4)
    weak_result = Simulator(parallel=False, show_progress=False).run(state, circuit, weak_params)

    assert weak_result.counts is not None
    assert sum(weak_result.counts.values()) == weak_params.shots
    assert weak_result.multi_time_times is None
    assert weak_result.multi_time_results is None
    assert weak_result.runtime_cost is None
    assert weak_result.max_bond is None
    assert weak_result.total_bond is None

    strong_state = State(num_qubits, initial="zeros")
    strong_params = StrongSimParams(observables=[Observable(Z(), 0)], num_traj=1, max_bond_dim=4)
    strong_result = Simulator(parallel=False, show_progress=False).run(strong_state, circuit, strong_params)

    assert strong_result.counts is None
    assert strong_result.runtime_cost is not None
    assert strong_result.max_bond is not None
    assert strong_result.total_bond is not None


def test_result_noise_model_reflects_sampled_noise() -> None:
    """Result.noise_model reflects the noise model that was sampled at run time."""
    num_qubits = 2
    state = State(num_qubits, initial="zeros")
    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=1)
    circuit.measure_all()

    noise_model = NoiseModel([{"name": "pauli_z", "sites": [i], "strength": 1e-3} for i in range(num_qubits)])
    weak_params = WeakSimParams(shots=4, max_bond_dim=4, random_seed=0)
    result = Simulator(parallel=False, show_progress=False).run(state, circuit, weak_params, noise_model)

    assert result.noise_model is not None
    assert not hasattr(weak_params, "noise_model")


def test_sim_params_not_mutated_after_analog_run() -> None:
    """User-supplied sim_params are unchanged after Simulator.run."""
    length = 2
    state = State(length, initial="zeros")
    H = Hamiltonian.ising(length, J=1.0, g=0.5)
    user_obs = Observable(Z(), 0)
    sim_params = AnalogSimParams(
        observables=[user_obs],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=1000,
        get_state=True,
        sample_timesteps=False,
    )
    original_num_traj = sim_params.num_traj

    result = Simulator(parallel=False, show_progress=False).run(state, H, sim_params)

    assert sim_params.num_traj == original_num_traj
    assert not hasattr(user_obs, "results")
    assert result.expectation_values[0] is not None
    assert result.observables[0] is not user_obs


def test_result_is_pickleable() -> None:
    """A populated :class:`Result` round-trips through pickle."""
    length = 2
    state = State(length, initial="zeros")
    H = Hamiltonian.ising(length, J=1.0, g=0.5)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=1,
        sample_timesteps=False,
    )
    result = Simulator(parallel=False, show_progress=False).run(state, H, sim_params)

    blob = pickle.dumps(result)
    restored = pickle.loads(blob)  # noqa: S301  # test-only: bytes are produced one line above (round-trip)

    assert isinstance(restored, Result)
    assert isinstance(restored.sim_params, AnalogSimParams)
    assert len(restored.observables) == 1
    restored_results = restored.expectation_values[0]
    original_results = result.expectation_values[0]
    assert restored_results is not None
    assert original_results is not None
    np.testing.assert_allclose(np.asarray(restored_results), np.asarray(original_results))


def test_aggregate_counts_skips_none_entries_and_sums_remainder() -> None:
    """aggregate_counts must sum every non-None measurement, even after a None entry.

    Regression: previously, the presence of any None short-circuited the aggregator
    to ``result.measurements[0]`` alone, silently dropping later valid per-shot dicts.
    """
    weak_params = WeakSimParams(shots=1, max_bond_dim=4)
    result = Result(sim_params=weak_params)
    result.measurements = [{0: 2, 1: 1}, None, {1: 3, 2: 4}]

    aggregate_counts(result)

    assert result.counts == {0: 2, 1: 4, 2: 4}


def test_aggregate_counts_handles_all_none() -> None:
    """An all-None measurements list yields an empty counts dict."""
    weak_params = WeakSimParams(shots=1, max_bond_dim=4)
    result = Result(sim_params=weak_params)
    result.measurements = [None, None, None]

    aggregate_counts(result)

    assert result.counts == {}
