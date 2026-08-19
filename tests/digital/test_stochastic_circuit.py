# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for explicit stochastic circuit sampling and execution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

import mqt.yaqs.simulator as simulator_module
from mqt.yaqs import DigitalSimParams, NoiseModel, Observable, Simulator, State
from mqt.yaqs.core.random_utils import make_trajectory_rng
from mqt.yaqs.digital import sample_stochastic_circuit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.yaqs.core.data_structures.mps import MPS


class _SequenceRNG:
    """Return a fixed sequence of random draws."""

    def __init__(self, draws: Sequence[float]) -> None:
        self.draws = iter(draws)

    def random(self) -> float:
        """Return the next configured draw."""
        return next(self.draws)


def _rng(*draws: float) -> np.random.Generator:
    return cast("np.random.Generator", _SequenceRNG(draws))


def _operation_names(circuit: QuantumCircuit) -> list[str]:
    return [instruction.operation.name for instruction in circuit.data]


def _operation_sites(circuit: QuantumCircuit) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (instruction.operation.name, tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits))
        for instruction in circuit.data
    ]


def _params(*, num_traj: int = 1, random_seed: int = 7) -> DigitalSimParams:
    return DigitalSimParams(
        observables=[Observable("z", 0)],
        num_traj=num_traj,
        random_seed=random_seed,
        preset="exact",
    )


@pytest.mark.parametrize(
    ("gate", "process_site", "expected_noise"),
    [
        ("rx", 0, ("x", (0,))),
        ("rzz", 0, ("x", (0,))),
        ("rzz", 1, ("x", (1,))),
    ],
)
def test_one_site_noise_opportunities(
    gate: str,
    process_site: int,
    expected_noise: tuple[str, tuple[int, ...]],
) -> None:
    """One- and two-qubit gates accept one-site noise on either gate site."""
    circuit = QuantumCircuit(1 if gate == "rx" else 2)
    if gate == "rx":
        circuit.rx(0.1, 0)
    else:
        circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [process_site], "strength": 1.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0))

    gate_sites = (0,) if gate == "rx" else (0, 1)
    assert _operation_sites(sampled) == [(gate, gate_sites), expected_noise]


def test_two_site_pauli_process_acts_on_both_gate_sites() -> None:
    """A two-site event appends its complete Pauli product."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([{"name": "crosstalk_xy", "sites": [0, 1], "strength": 1.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0))

    assert _operation_sites(sampled) == [("rzz", (0, 1)), ("x", (0,)), ("y", (1,))]


def test_processes_outside_the_gate_support_are_ignored() -> None:
    """Only processes whose complete support is contained in a gate participate."""
    circuit = QuantumCircuit(3)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "pauli_z", "sites": [2], "strength": 1e6},
        {"name": "crosstalk_xx", "sites": [0, 2], "strength": 1e6},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.9))

    assert _operation_sites(sampled) == [("rzz", (0, 1)), ("x", (0,))]


def test_at_most_one_categorical_process_is_selected_per_gate() -> None:
    """Gate sites do not receive independent Bernoulli events."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 10.0},
        {"name": "pauli_y", "sites": [0], "strength": 10.0},
        {"name": "pauli_z", "sites": [1], "strength": 10.0},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.5))

    assert _operation_names(sampled) == ["cx", "y"]


def test_non_gate_instructions_are_copied_without_noise() -> None:
    """Non-Gate instructions are not noise opportunities."""
    circuit = QuantumCircuit(3, 1)
    circuit.barrier(0, 1)
    circuit.reset(0)
    circuit.delay(5, 1)
    circuit.measure(2, 0)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1e6}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng())

    assert _operation_names(sampled) == _operation_names(circuit)


def test_zero_rate_inserts_no_noise_without_rng_draws() -> None:
    """A zero-rate opportunity consumes neither event nor categorical draws."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.0},
        {"name": "lowering", "sites": [1], "strength": 0.0},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng())

    assert _operation_names(sampled) == ["rzz"]


def test_event_probability_is_one_minus_exp_of_total_rate() -> None:
    """The event boundary uses ``1 - exp(-sum(gamma_i))``."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.2},
        {"name": "pauli_z", "sites": [1], "strength": 0.3},
    ])
    event_probability = -math.expm1(-0.5)

    event = sample_stochastic_circuit(circuit, noise_model, _rng(float(np.nextafter(event_probability, 0.0)), 0.0))
    no_event = sample_stochastic_circuit(circuit, noise_model, _rng(event_probability))

    assert _operation_names(event) == ["rzz", "x"]
    assert _operation_names(no_event) == ["rzz"]


@pytest.mark.parametrize(("draw", "expected"), [(0.05, "x"), (0.2, "y"), (0.8, "z")])
def test_categorical_probability_is_proportional_to_rate(draw: float, expected: str) -> None:
    """Conditional process probabilities are ``gamma_i / Gamma``."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "pauli_y", "sites": [0], "strength": 3.0},
        {"name": "pauli_z", "sites": [0], "strength": 6.0},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, draw))

    assert _operation_names(sampled) == ["rzz", expected]


def test_input_is_not_mutated_and_native_gates_are_preserved() -> None:
    """The copy retains native RX and RZZ instructions without mutating the input."""
    circuit = QuantumCircuit(2, name="ideal", metadata={"source": "test"})
    circuit.rx(0.1, 0)
    circuit.rzz(0.2, 0, 1)
    original = circuit.copy()
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0, 0.0, 0.0))

    assert circuit == original
    assert sampled.name == "ideal"
    assert sampled.metadata == {"source": "test"}
    assert _operation_names(sampled)[::2] == ["rx", "rzz"]


def test_seeded_trajectory_streams_are_reproducible_and_distinct() -> None:
    """A trajectory index identifies a stable, independent circuit realization."""
    circuit = QuantumCircuit(2)
    for _ in range(30):
        circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.1},
        {"name": "pauli_z", "sites": [1], "strength": 0.2},
    ])

    first = sample_stochastic_circuit(circuit, noise_model, make_trajectory_rng(0, base_seed=17))
    repeated = sample_stochastic_circuit(circuit, noise_model, make_trajectory_rng(0, base_seed=17))
    second = sample_stochastic_circuit(circuit, noise_model, make_trajectory_rng(1, base_seed=17))

    assert first == repeated
    assert first != second


def test_distribution_is_resolved_once_per_helper_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The direct helper resolves a distribution-valued model once."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    noise_model = NoiseModel([
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "normal", "mean": 0.0, "std": 0.0},
        }
    ])
    original_sample = NoiseModel.sample
    sample_calls = 0

    def recording_sample(self: NoiseModel, rng: np.random.Generator | int | None = None) -> NoiseModel:
        nonlocal sample_calls
        sample_calls += 1
        return original_sample(self, rng=rng)

    monkeypatch.setattr(NoiseModel, "sample", recording_sample)
    sampled = sample_stochastic_circuit(circuit, noise_model, np.random.default_rng(11))

    assert sample_calls == 1
    assert _operation_names(sampled) == ["x"]


@pytest.mark.parametrize(
    ("name", "sites"),
    [("raising", [0]), ("lowering", [0]), ("longrange_crosstalk_xx", [0, 2])],
)
def test_unsupported_process_is_rejected(name: str, sites: list[int]) -> None:
    """Explicit circuits never fall back to state-dependent TJM noise."""
    circuit = QuantumCircuit(3)
    circuit.x(0)
    noise_model = NoiseModel([{"name": name, "sites": sites, "strength": 0.1}])

    with pytest.raises(ValueError, match="supports recognized YAQS Pauli processes only"):
        sample_stochastic_circuit(circuit, noise_model, _rng())


def test_scheduled_jumps_are_rejected() -> None:
    """Explicit circuit sampling does not silently ignore scheduled jumps."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    noise_model = NoiseModel(scheduled_jumps=[{"time": 0.0, "sites": [0], "name": "x"}])

    with pytest.raises(ValueError, match="does not support scheduled jumps"):
        sample_stochastic_circuit(circuit, noise_model, _rng())


def test_trajectory_pipeline_samples_once_executes_once_and_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each raw trajectory is one sampled circuit execution with no second noise model."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    params = _params(num_traj=4, random_seed=21)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.7}])
    original_execute = simulator_module.digital_tjm
    sampled_circuits: list[QuantumCircuit] = []
    executions: list[tuple[int, NoiseModel | None, QuantumCircuit]] = []

    def recording_sampler(
        ideal: QuantumCircuit,
        concrete_model: NoiseModel,
        rng: np.random.Generator,
    ) -> QuantumCircuit:
        sampled = sample_stochastic_circuit(ideal, concrete_model, rng)
        sampled_circuits.append(sampled)
        return sampled

    def recording_execute(
        args: tuple[int, MPS, NoiseModel | None, DigitalSimParams, QuantumCircuit],
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[object | None, object | None, dict[int, int] | None, MPS | None]:
        executions.append((args[0], args[2], args[4]))
        return original_execute(args, rng=rng)

    monkeypatch.setattr(simulator_module, "sample_stochastic_circuit", recording_sampler)
    monkeypatch.setattr(simulator_module, "digital_tjm", recording_execute)
    result = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1, initial="zeros"), circuit, params, noise_model
    )

    assert len(sampled_circuits) == params.num_traj
    assert [index for index, _model, _circuit in executions] == list(range(params.num_traj))
    assert all(model is None for _index, model, _circuit in executions)
    assert all(executed is sampled_circuits[index] for index, _model, executed in executions)
    np.testing.assert_allclose(result.expectation_values[0], np.mean(result.trajectories[0], axis=0), atol=1e-12)


def test_standard_count_and_diagnostic_aggregation() -> None:
    """The existing result buffers aggregate combined outputs unchanged."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    params = DigitalSimParams(observables=[Observable("z", 0)], shots=5, num_traj=3, random_seed=27, preset="exact")
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.4}])

    result = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1, initial="zeros"), circuit, params, noise_model
    )

    assert result.trajectories[0].shape == (3, 1)
    assert result.counts is not None
    assert sum(result.counts.values()) == 5
    assert result.runtime_cost is not None
    assert result.runtime_cost.shape == (1,)


def test_seeded_parallel_and_serial_ensembles_match() -> None:
    """Serial and multiprocessing workers use the same trajectory-index streams."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    params = _params(num_traj=4, random_seed=22)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.7}])

    serial = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1), circuit, params, noise_model
    )
    parallel = Simulator(parallel=True, max_workers=2, show_progress=False).run_stochastic_circuit(
        State(1), circuit, params, noise_model
    )

    np.testing.assert_array_equal(parallel.trajectories[0], serial.trajectories[0])
    np.testing.assert_array_equal(parallel.expectation_values[0], serial.expectation_values[0])


def test_distribution_is_fixed_once_per_ensemble(monkeypatch: pytest.MonkeyPatch) -> None:
    """Static disorder is sampled once before trajectory-local circuit sampling."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    params = _params(num_traj=4, random_seed=41)
    noise_model = NoiseModel([
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "normal", "mean": 0.4, "std": 0.0},
        }
    ])
    original_sample = NoiseModel.sample
    original_sampler = simulator_module.sample_stochastic_circuit
    sample_calls = 0
    strengths: list[float] = []

    def recording_sample(self: NoiseModel, rng: np.random.Generator | int | None = None) -> NoiseModel:
        nonlocal sample_calls
        sample_calls += 1
        return original_sample(self, rng=rng)

    def recording_sampler(
        ideal: QuantumCircuit,
        concrete_model: NoiseModel,
        rng: np.random.Generator,
    ) -> QuantumCircuit:
        strengths.append(float(concrete_model.processes[0]["strength"]))
        return original_sampler(ideal, concrete_model, rng)

    monkeypatch.setattr(NoiseModel, "sample", recording_sample)
    monkeypatch.setattr(simulator_module, "sample_stochastic_circuit", recording_sampler)
    result = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1), circuit, params, noise_model
    )

    assert sample_calls == 1
    assert strengths == [0.4] * params.num_traj
    assert result.trajectories[0].shape == (params.num_traj, 1)
