# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for stochastic circuit construction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

import mqt.yaqs.digital.digital_tjm as digital_tjm_module
import mqt.yaqs.digital.stochastic_circuit as stochastic_circuit_module
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
        (
            instruction.operation.name,
            tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits),
        )
        for instruction in circuit.data
    ]


def test_non_gate_instructions_are_not_noise_locations() -> None:
    """Barriers, resets, delays, and measurements are copied without noise."""
    circuit = QuantumCircuit(3, 1)
    circuit.barrier(0, 1)
    circuit.reset(0)
    circuit.delay(5, 1)
    circuit.measure(2, 0)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1e6}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng())

    assert _operation_names(sampled) == _operation_names(circuit)


@pytest.mark.parametrize(
    ("gate_name", "gate_args", "pauli_name"),
    [("rx", (0.1, 0), "x"), ("h", (0,), "z")],
)
def test_one_qubit_gate_is_noise_location(
    gate_name: str,
    gate_args: tuple[float | int, ...],
    pauli_name: str,
) -> None:
    """A supported Pauli process is sampled after a one-qubit gate."""
    circuit = QuantumCircuit(1)
    getattr(circuit, gate_name)(*gate_args)
    noise_model = NoiseModel([{"name": f"pauli_{pauli_name}", "sites": [0], "strength": 1.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0))

    assert _operation_sites(sampled) == [(gate_name, (0,)), (pauli_name, (0,))]


def test_two_site_process_is_excluded_from_one_qubit_gate_support() -> None:
    """A one-qubit gate sees only processes fully supported on its target."""
    circuit = QuantumCircuit(2)
    circuit.rx(0.1, 0)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "crosstalk_xx", "sites": [0, 1], "strength": 1e6},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.9))

    assert _operation_sites(sampled) == [("rx", (0,)), ("x", (0,))]


@pytest.mark.parametrize(
    ("process", "expected"),
    [
        ({"name": "pauli_x", "sites": [0], "strength": 1.0}, ("x", (0,))),
        ({"name": "pauli_z", "sites": [1], "strength": 1.0}, ("z", (1,))),
    ],
)
def test_one_site_process_can_target_either_gate_qubit(
    process: dict[str, object], expected: tuple[str, tuple[int, ...]]
) -> None:
    """A relevant one-site process retains its declared gate-qubit support."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)

    sampled = sample_stochastic_circuit(circuit, NoiseModel([process]), _rng(0.0, 0.0))

    assert _operation_sites(sampled) == [("rzz", (0, 1)), expected]


@pytest.mark.parametrize(
    ("name", "pauli_name"),
    [("crosstalk_xx", "x"), ("crosstalk_yy", "y"), ("crosstalk_zz", "z")],
)
def test_two_site_pauli_process_acts_on_both_gate_qubits(name: str, pauli_name: str) -> None:
    """A selected two-site process is one event represented by one Pauli gate per support site."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([{"name": name, "sites": [0, 1], "strength": 1.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0))

    assert _operation_sites(sampled) == [("rzz", (0, 1)), (pauli_name, (0,)), (pauli_name, (1,))]


def test_spectator_process_is_excluded_from_gate_support() -> None:
    """A process on an unrelated qubit cannot enter the gate-level categorical draw."""
    circuit = QuantumCircuit(3)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "pauli_z", "sites": [2], "strength": 1e6},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.9))

    assert _operation_sites(sampled) == [("rzz", (0, 1)), ("x", (0,))]


def test_at_most_one_categorical_process_is_selected_per_gate() -> None:
    """Shared gate support produces one categorical event rather than independent site events."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 10.0},
        {"name": "pauli_y", "sites": [0], "strength": 10.0},
        {"name": "pauli_z", "sites": [1], "strength": 10.0},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.5))

    assert _operation_names(sampled) == ["cx", "y"]


def test_zero_total_gamma_inserts_no_noise_without_drawing() -> None:
    """A zero total rate skips event and categorical RNG draws."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.0},
        {"name": "pauli_z", "sites": [1], "strength": 0.0},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng())

    assert _operation_names(sampled) == ["rzz"]


def test_event_probability_is_one_minus_exp_of_total_gamma() -> None:
    """The event boundary uses ``1 - exp(-sum(gamma_i))`` without treating gamma as probability."""
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


@pytest.mark.parametrize(
    ("category_draw", "expected"),
    [(0.05, "x"), (0.2, "y"), (0.8, "z")],
)
def test_categorical_process_weighting_uses_gamma_ratios(category_draw: float, expected: str) -> None:
    """Categorical intervals are proportional to rates one, three, and six."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "pauli_y", "sites": [0], "strength": 3.0},
        {"name": "pauli_z", "sites": [0], "strength": 6.0},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, category_draw))

    assert _operation_names(sampled) == ["rzz", expected]


def test_seeded_trajectory_realizations_are_reproducible_and_independent() -> None:
    """Equal trajectory streams reproduce a circuit and trajectory seeds derive distinct streams."""
    circuit = QuantumCircuit(2)
    for _ in range(20):
        circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.1},
        {"name": "pauli_z", "sites": [1], "strength": 0.2},
    ])

    first = sample_stochastic_circuit(circuit, noise_model, make_trajectory_rng(0, base_seed=17))
    repeated = sample_stochastic_circuit(circuit, noise_model, make_trajectory_rng(0, base_seed=17))
    second_trajectory = sample_stochastic_circuit(circuit, noise_model, make_trajectory_rng(1, base_seed=17))

    assert first == repeated
    first_stream = make_trajectory_rng(0, base_seed=17).bit_generator.state
    repeated_stream = make_trajectory_rng(0, base_seed=17).bit_generator.state
    second_stream = make_trajectory_rng(1, base_seed=17).bit_generator.state
    assert first_stream == repeated_stream
    assert first_stream != second_stream
    assert second_trajectory == sample_stochastic_circuit(circuit, noise_model, make_trajectory_rng(1, base_seed=17))


def test_multiple_one_and_two_qubit_gates_are_independent_opportunities() -> None:
    """One RNG stream supplies separate event and category draws for every eligible gate."""
    circuit = QuantumCircuit(2)
    circuit.rx(0.1, 0)
    circuit.rzz(0.2, 0, 1)
    circuit.rx(0.3, 1)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 100.0},
        {"name": "pauli_z", "sites": [1], "strength": 100.0},
    ])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0, 0.0, 0.75, 0.0, 0.0))

    assert _operation_sites(sampled) == [
        ("rx", (0,)),
        ("x", (0,)),
        ("rzz", (0, 1)),
        ("z", (1,)),
        ("rx", (1,)),
        ("z", (1,)),
    ]


def test_gate_wider_than_two_qubits_is_not_a_noise_location() -> None:
    """A three-qubit gate is copied without consuming stochastic draws."""
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1e6}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng())

    assert _operation_sites(sampled) == [("ccx", (0, 1, 2))]


def test_input_is_not_mutated_and_native_rzz_is_preserved() -> None:
    """Construction copies circuit metadata and instructions without decomposing or mutating RZZ."""
    circuit = QuantumCircuit(2, name="ideal", metadata={"source": "test"})
    circuit.global_phase = 0.25
    circuit.rzz(0.2, 0, 1)
    original = circuit.copy()
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0))

    assert circuit == original
    assert sampled is not circuit
    assert sampled.name == "ideal"
    assert sampled.metadata == {"source": "test"}
    assert sampled.data[0].operation.name == "rzz"
    assert sampled.data[0].operation.params == [0.2]


def test_pauli_structure_is_resolved_from_custom_operator_matrix() -> None:
    """A custom process name is accepted when its resolved matrix has Pauli structure."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    matrix = -1j * NoiseModel.get_operator("pauli_y")
    noise_model = NoiseModel([{"name": "custom_jump", "sites": [1], "strength": 1.0, "matrix": matrix}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng(0.0, 0.0))

    assert _operation_sites(sampled) == [("cx", (0, 1)), ("y", (1,))]
    assert np.isclose(np.exp(1j * float(sampled.global_phase)), -1j)


def test_distribution_strengths_are_sampled_once_per_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static strength disorder is resolved once before all gate opportunities."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    circuit.rzz(0.3, 0, 1)
    noise_model = NoiseModel([
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "normal", "mean": 1.0, "std": 0.0},
        }
    ])
    original_sample = NoiseModel.sample
    sample_calls = 0

    def recording_sample(self: NoiseModel, rng: np.random.Generator | int | None = None) -> NoiseModel:
        nonlocal sample_calls
        sample_calls += 1
        return original_sample(self, rng=rng)

    monkeypatch.setattr(NoiseModel, "sample", recording_sample)

    sampled = sample_stochastic_circuit(circuit, noise_model, np.random.default_rng(2))

    assert sample_calls == 1
    assert _operation_names(sampled).count("rzz") == 2


def test_positive_rate_unsupported_process_is_rejected_before_sampling() -> None:
    """A relevant lowering jump fails deterministically before any random draw."""
    circuit = QuantumCircuit(2)
    circuit.rzz(0.2, 0, 1)
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 1.0}])

    with pytest.raises(ValueError, match="supports only Pauli jump processes"):
        sample_stochastic_circuit(circuit, noise_model, _rng())


def test_zero_rate_unsupported_process_is_ignored() -> None:
    """A zero-rate unsupported process cannot affect the circuit or consume RNG draws."""
    circuit = QuantumCircuit(1)
    circuit.rx(0.2, 0)
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.0}])

    sampled = sample_stochastic_circuit(circuit, noise_model, _rng())

    assert _operation_names(sampled) == ["rx"]


def _one_qubit_params(*, num_traj: int = 1, random_seed: int = 7) -> DigitalSimParams:
    """Create compact stochastic-circuit parameters for one Z observable.

    Returns:
        Digital parameters for focused one-qubit tests.
    """
    return DigitalSimParams(
        observables=[Observable("z", 0)],
        num_traj=num_traj,
        random_seed=random_seed,
        preset="exact",
    )


def test_pauli_noise_is_applied_once() -> None:
    """A materialized Pauli process is not applied again during execution."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1e6}])
    params = _one_qubit_params()

    result = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1, initial="zeros"), circuit, params, noise_model
    )

    np.testing.assert_allclose(result.expectation_values[0], [1.0], atol=1e-12)


def test_zero_noise_stochastic_execution_matches_ideal_execution() -> None:
    """A zero-rate stochastic run is exactly the standard ideal YAQS circuit result."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    params = _one_qubit_params()
    simulator = Simulator(parallel=False, show_progress=False)

    ideal = simulator.run(State(1, initial="zeros"), circuit, params)
    stochastic = simulator.run_stochastic_circuit(
        State(1, initial="zeros"),
        circuit,
        params,
        NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.0}]),
    )

    np.testing.assert_allclose(stochastic.trajectories[0], ideal.trajectories[0], atol=1e-12)
    np.testing.assert_allclose(stochastic.expectation_values[0], ideal.expectation_values[0], atol=1e-12)


def test_single_trajectory_run_stores_one_result() -> None:
    """A one-trajectory ensemble stores exactly one observable value."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    noise_model = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 0.5}])

    result = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1, initial="zeros"), circuit, _one_qubit_params(num_traj=1), noise_model
    )

    assert result.trajectories[0].shape == (1, 1)


def test_dissipative_one_qubit_opportunity_observes_post_gate_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One-qubit lowering is invoked after X, when the MPS already represents the excited state."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.4}])
    seen_states: list[np.ndarray] = []
    original_apply = digital_tjm_module.apply_dissipation

    def recording_apply(state: MPS, local_model: NoiseModel, dt: float, sim_params: DigitalSimParams) -> None:
        if local_model.processes:
            seen_states.append(state.to_vec().copy())
        original_apply(state, local_model, dt, sim_params)

    monkeypatch.setattr(digital_tjm_module, "apply_dissipation", recording_apply)

    result = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1, initial="zeros"),
        circuit,
        _one_qubit_params(random_seed=5),
        noise_model,
    )

    assert result.trajectories[0].shape == (1, 1)
    assert len(seen_states) == 1
    np.testing.assert_allclose(np.abs(seen_states[0]), [0.0, 1.0], atol=1e-12)


def test_dissipative_two_qubit_opportunity_observes_post_gate_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A two-site dissipative opportunity runs immediately after the eligible CX."""
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.cx(0, 1)
    noise_model = NoiseModel([{"name": "lowering_two", "sites": [0, 1], "strength": 0.3}])
    seen_states: list[np.ndarray] = []
    original_apply = digital_tjm_module.apply_dissipation

    def recording_apply(state: MPS, local_model: NoiseModel, dt: float, sim_params: DigitalSimParams) -> None:
        if local_model.processes:
            seen_states.append(state.to_vec().copy())
        original_apply(state, local_model, dt, sim_params)

    monkeypatch.setattr(digital_tjm_module, "apply_dissipation", recording_apply)
    params = DigitalSimParams(observables=[Observable("z", 0)], num_traj=1, random_seed=9, preset="exact")

    result = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(2, initial="zeros"),
        circuit,
        params,
        noise_model,
    )

    assert result.trajectories[0].shape == (1, 1)
    assert len(seen_states) == 1
    np.testing.assert_allclose(np.abs(seen_states[0]), [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_dissipative_execution_excludes_gates_wider_than_two_qubits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in post-gate executor does not add a noise opportunity after CCX."""
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.3}])
    opportunities = 0
    original_apply = digital_tjm_module.apply_dissipation

    def recording_apply(
        state: MPS,
        model: NoiseModel,
        dt: float,
        sim_params: DigitalSimParams,
    ) -> None:
        nonlocal opportunities
        opportunities += 1
        original_apply(state, model, dt, sim_params)

    monkeypatch.setattr(digital_tjm_module, "apply_dissipation", recording_apply)
    params = DigitalSimParams(observables=[Observable("z", 0)], num_traj=1, random_seed=10, preset="exact")

    Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(3, initial="zeros"),
        circuit,
        params,
        noise_model,
    )

    assert opportunities == 0


def test_dissipative_trajectory_is_reproducible_for_fixed_seed() -> None:
    """The same base seed reproduces state-dependent jump output."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.6}])
    params = _one_qubit_params(random_seed=13)

    def execute() -> np.ndarray:
        result = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
            State(1, initial="zeros"),
            circuit,
            params,
            noise_model,
        )
        return result.trajectories[0]

    np.testing.assert_array_equal(execute(), execute())


def test_ensemble_stores_requested_trajectories_and_uses_result_mean() -> None:
    """The ensemble stores N independent realizations before standard Result aggregation."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    params = _one_qubit_params(num_traj=7, random_seed=21)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.7}])

    result = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1, initial="zeros"), circuit, params, noise_model
    )

    assert result.trajectories[0].shape == (7, 1)
    np.testing.assert_allclose(
        result.expectation_values[0],
        np.mean(result.trajectories[0], axis=0),
        atol=1e-12,
    )


def test_ensemble_uses_standard_counts_and_diagnostic_aggregation() -> None:
    """Combined outputs use the requested trajectory and shot budgets."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    params = DigitalSimParams(
        observables=[Observable("z", 0)],
        shots=5,
        num_traj=3,
        random_seed=27,
        preset="exact",
    )
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.4}])

    result = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1, initial="zeros"), circuit, params, noise_model
    )

    assert result.trajectories[0].shape == (3, 1)
    assert len(result.measurements) == 3
    assert result.counts is not None
    assert sum(result.counts.values()) == 5
    assert result.runtime_cost is not None
    assert result.runtime_cost.shape == (1,)


def test_seeded_ensemble_parallel_and_serial_paths_match() -> None:
    """The stochastic-circuit worker preserves trajectory-index streams in a process pool."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    params = _one_qubit_params(num_traj=4, random_seed=22)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.7}])

    serial = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1, initial="zeros"), circuit, params, noise_model
    )
    parallel = Simulator(parallel=True, max_workers=2, show_progress=False).run_stochastic_circuit(
        State(1, initial="zeros"), circuit, params, noise_model
    )

    np.testing.assert_array_equal(parallel.trajectories[0], serial.trajectories[0])
    np.testing.assert_array_equal(parallel.expectation_values[0], serial.expectation_values[0])


def test_mixed_model_is_realized_once_by_existing_tjm_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed Pauli/dissipative noise remains one local TJM process set with no explicit duplicate."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    noise_model = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.2},
        {"name": "lowering", "sites": [0], "strength": 0.3},
    ])
    process_sets: list[list[str]] = []
    original_stochastic_process = digital_tjm_module.stochastic_process

    def recording_process(
        state: MPS,
        local_model: NoiseModel,
        dt: float,
        sim_params: DigitalSimParams,
        rng: np.random.Generator,
    ) -> MPS:
        process_sets.append([str(process["name"]) for process in local_model.processes])
        return original_stochastic_process(state, local_model, dt, sim_params, rng)

    monkeypatch.setattr(digital_tjm_module, "stochastic_process", recording_process)

    Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1, initial="zeros"),
        circuit,
        _one_qubit_params(random_seed=31),
        noise_model,
    )

    assert process_sets == [["pauli_x", "lowering"]]


def test_distribution_is_fixed_per_ensemble_while_trajectory_streams_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static disorder is sampled once and every schedule receives a distinct trajectory RNG."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    params = _one_qubit_params(num_traj=4, random_seed=41)
    noise_model = NoiseModel([
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "normal", "mean": 0.4, "std": 0.0},
        }
    ])
    original_sample = NoiseModel.sample
    original_sampler = stochastic_circuit_module.sample_stochastic_circuit
    sample_calls = 0
    strengths: list[float] = []
    rng_states: list[str] = []

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
        rng_states.append(repr(rng.bit_generator.state))
        return original_sampler(ideal, concrete_model, rng)

    monkeypatch.setattr(NoiseModel, "sample", recording_sample)
    monkeypatch.setattr(stochastic_circuit_module, "sample_stochastic_circuit", recording_sampler)

    result = Simulator(parallel=False, show_progress=False).run_stochastic_circuit(
        State(1, initial="zeros"), circuit, params, noise_model
    )

    assert sample_calls == 1
    assert strengths == [0.4] * 4
    assert len(set(rng_states)) == 4
    assert result.trajectories[0].shape == (4, 1)
