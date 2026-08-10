# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for analog and digital simulation programs."""

from __future__ import annotations

import contextlib
import copy
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

import mqt.yaqs.analog.analog_tjm as analog_module
import mqt.yaqs.digital.digital_tjm as digital_module
import mqt.yaqs.simulator as simulator_module
from mqt.yaqs import (
    AnalogSimParams,
    DigitalSimParams,
    Hamiltonian,
    NoiseModel,
    Observable,
    SimulationProgram,
    Simulator,
    State,
)
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_program import (
    _compile_program,  # ruff: ignore[import-private-name]  # validate private compiler invariant
)
from mqt.yaqs.core.random_utils import make_trajectory_rng

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGOpNode

    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.core.data_structures.simulation_parameters import GateMode
    from mqt.yaqs.core.libraries.gate_library import BaseGate
    from mqt.yaqs.digital.digital_tjm import _CompiledCircuit


def _zero_hamiltonian(length: int) -> Hamiltonian:
    """Return a static zero Ising Hamiltonian."""
    return Hamiltonian.ising(length, J=0.0, g=0.0)


def test_mixed_program_matches_manual_state_handoff() -> None:
    """Digital-analog-digital execution matches explicit standalone calls."""
    length = 2
    preparation = QuantumCircuit(length)
    preparation.x(0)
    intervention = QuantumCircuit(length)
    intervention.x(1)
    observables = [Observable("z", 0), Observable("z", 1)]
    digital_params = DigitalSimParams(get_state=True)
    analog_params = AnalogSimParams(
        elapsed_time=0.1,
        dt=0.1,
        sample_timesteps=False,
        get_state=True,
    )
    outputless_params = DigitalSimParams()
    hamiltonian = _zero_hamiltonian(length)
    initial_state = State(length, initial="zeros")
    initial_vector = initial_state.mps.to_vec().copy()
    program = SimulationProgram(
        [
            (preparation, digital_params),
            (hamiltonian, analog_params),
            (intervention, outputless_params),
        ],
        observables=observables,
        get_state=True,
    )

    result = Simulator(parallel=False, show_progress=False).run(initial_state, program)

    manual_simulator = Simulator(parallel=False, show_progress=False)
    manual_first = manual_simulator.run(
        State(length, initial="zeros"),
        preparation,
        DigitalSimParams(observables=observables, get_state=True),
    )
    assert manual_first.output_state is not None
    manual_second = manual_simulator.run(
        manual_first.output_state,
        hamiltonian,
        AnalogSimParams(
            observables=observables,
            elapsed_time=0.1,
            dt=0.1,
            sample_timesteps=False,
            get_state=True,
        ),
    )
    assert manual_second.output_state is not None
    manual_final = manual_simulator.run(
        manual_second.output_state,
        intervention,
        DigitalSimParams(observables=observables, get_state=True),
    )

    assert result.sim_params is None
    assert result.output_state is not None
    assert manual_final.output_state is not None
    np.testing.assert_allclose(result.output_state.mps.to_vec(), manual_final.output_state.mps.to_vec(), atol=1e-10)
    np.testing.assert_allclose(initial_state.mps.to_vec(), initial_vector, atol=1e-12)
    assert [segment.segment_type for segment in result.segment_results] == ["digital", "analog", "digital"]
    assert [segment.segment_index for segment in result.segment_results] == [0, 1, 2]
    assert [segment.time_offset for segment in result.segment_results] == [0.0, 0.0, 0.1]
    assert program.segments[0].sim_params is digital_params
    assert program.segments[1].sim_params is analog_params
    assert program.segments[2].sim_params is outputless_params
    assert isinstance(result.segment_results[0].sim_params, DigitalSimParams)
    assert isinstance(result.segment_results[1].sim_params, AnalogSimParams)
    assert isinstance(result.segment_results[2].sim_params, DigitalSimParams)
    assert result.segment_results[0].output_state is not None
    assert result.segment_results[1].output_state is not None
    assert result.segment_results[2].output_state is None
    analog_times = result.segment_results[1].times
    assert analog_times is not None
    np.testing.assert_allclose(analog_times, np.array([0.1]))
    assert result.times is not None
    assert len(result.expectation_values) == 2


def test_flattened_program_result_preserves_boundary_values_around_digital_pulse() -> None:
    """Equal boundary times retain analog and digital samples around an instantaneous gate."""
    observable = Observable("z", 0)
    params = AnalogSimParams(
        elapsed_time=0.1,
        dt=0.1,
        sample_timesteps=True,
    )
    pulse = QuantumCircuit(1)
    pulse.x(0)
    pulse_params = DigitalSimParams(sample_layers=True)
    program = SimulationProgram(
        [
            (_zero_hamiltonian(1), params),
            (pulse, pulse_params),
            (_zero_hamiltonian(1), params),
        ],
        observables=[observable],
    )

    result = Simulator(parallel=False, show_progress=False).run(State(1, initial="zeros"), program)

    assert result.times is not None
    np.testing.assert_allclose(result.times, np.array([0.0, 0.1, 0.1, 0.1, 0.1, 0.2]))
    np.testing.assert_allclose(result.expectation_values[0], np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]), atol=1e-10)


def test_flattened_program_result_preserves_boundary_values_between_adjacent_analog_segments() -> None:
    """Adjacent analog grids remain lossless even without an intervention."""
    observable = Observable("z", 0)
    first_params = AnalogSimParams(
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
    )
    second_params = AnalogSimParams(
        elapsed_time=0.2,
        dt=0.2,
        sample_timesteps=True,
    )
    program = SimulationProgram(
        [
            (_zero_hamiltonian(1), first_params),
            (_zero_hamiltonian(1), second_params),
        ],
        observables=[observable],
    )

    result = Simulator(parallel=False, show_progress=False).run(State(1, initial="zeros"), program)

    assert result.times is not None
    np.testing.assert_allclose(result.times, np.array([0.0, 0.1, 0.2, 0.2, 0.4]))
    np.testing.assert_allclose(result.expectation_values[0], np.ones(5), atol=1e-10)


def test_program_resolves_default_segment_parameters() -> None:
    """Ordinary parameter defaults support state-only internal propagation."""
    circuit = QuantumCircuit(2)
    circuit.x(0)
    program = SimulationProgram(
        [
            (_zero_hamiltonian(2), AnalogSimParams()),
            (circuit, DigitalSimParams()),
        ],
        get_state=True,
    )

    result = Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program)

    assert result.output_state is not None
    assert isinstance(result.segment_results[0].sim_params, AnalogSimParams)
    assert isinstance(result.segment_results[1].sim_params, DigitalSimParams)
    np.testing.assert_allclose(np.abs(result.output_state.mps.to_vec()), np.array([0.0, 1.0, 0.0, 0.0]))


def test_simulator_run_accepts_pair_list_kwargs() -> None:
    """``Simulator.run`` can wrap a segment pair list with program-level kwargs."""
    circuit = QuantumCircuit(2)
    circuit.x(0)
    result = Simulator(parallel=False, show_progress=False).run(
        State(2, initial="zeros"),
        [
            (_zero_hamiltonian(2), AnalogSimParams(elapsed_time=0.1, dt=0.1)),
            (circuit, DigitalSimParams()),
        ],
        observables=[Observable("z", 0)],
        get_state=True,
    )

    assert result.output_state is not None
    assert result.expectation_values[0][-1] == pytest.approx(-1.0)


def test_program_propagates_state_without_exposing_final_state() -> None:
    """Program and segment get_state flags affect output, not internal handoff."""
    preparation = QuantumCircuit(2)
    preparation.x(0)
    program = SimulationProgram(
        [
            (preparation, DigitalSimParams()),
            (_zero_hamiltonian(2), AnalogSimParams(elapsed_time=0.1, dt=0.1)),
            (QuantumCircuit(2), DigitalSimParams()),
        ],
        observables=[Observable("z", 0)],
    )

    result = Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program)

    assert result.output_state is None
    assert all(segment.output_state is None for segment in result.segment_results)
    assert result.segment_results[2].expectation_values[0][0] == pytest.approx(-1.0)
    assert result.expectation_values[0][-1] == pytest.approx(-1.0)


def test_digital_program_segment_returns_requested_shot_counts() -> None:
    """Digital shot output is available on the segment and outer program result."""
    shots = 8
    program = SimulationProgram([(QuantumCircuit(2), DigitalSimParams(shots=shots))])

    result = Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program)

    segment_result = result.segment_results[0]
    assert segment_result.counts is not None
    assert sum(segment_result.counts.values()) == shots
    assert result.counts is not None
    assert sum(result.counts.values()) == shots
    assert segment_result.output_state is None


def test_program_call_contract_is_distinct_from_standalone_run() -> None:
    """Program calls reject standalone parameters and state lists early."""
    program = SimulationProgram([(QuantumCircuit(2), DigitalSimParams())])
    simulator = Simulator(parallel=False, show_progress=False)
    state = State(2, initial="zeros")

    with pytest.raises(TypeError, match="sim_params must be None"):
        simulator.run(state, program, DigitalSimParams())
    with pytest.raises(TypeError, match="single State"):
        simulator.run([state], program)
    with pytest.raises(TypeError, match="Standalone simulation requires"):
        simulator.run(state, QuantumCircuit(2))


@pytest.mark.parametrize(
    ("state", "program", "message"),
    [
        (
            State(2, initial="zeros", representation="vector"),
            SimulationProgram([(QuantumCircuit(2), DigitalSimParams())], get_state=True),
            "representation='mps'",
        ),
        (
            State(2, initial="zeros", physical_dimensions=3),
            SimulationProgram([(Hamiltonian.ising(2, J=0.0, g=0.0), AnalogSimParams())], get_state=True),
            "Hamiltonian MPO site 0 has physical legs",
        ),
        (
            State(2, initial="zeros"),
            SimulationProgram([(QuantumCircuit(3), DigitalSimParams())], get_state=True),
            "circuit.num_qubits=3",
        ),
        (
            State(2, initial="zeros"),
            SimulationProgram([(_zero_hamiltonian(3), AnalogSimParams())], get_state=True),
            "Hamiltonian.length=3",
        ),
    ],
)
def test_program_compilation_rejects_unsupported_inputs(
    state: State,
    program: SimulationProgram,
    message: str,
) -> None:
    """Compilation reports unsupported states and segments before execution."""
    with pytest.raises(ValueError, match=message):
        Simulator(parallel=False, show_progress=False).run(state, program)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (AnalogSimParams(order=3), "order must be 1 or 2"),
        (
            AnalogSimParams(multi_time_observables=[(Observable("z", 0), Observable("z", 0))]),
            "multi_time_observables are not supported",
        ),
    ],
)
def test_program_compilation_rejects_unsupported_analog_parameters(
    params: AnalogSimParams,
    message: str,
) -> None:
    """Program compilation rejects standalone-only analog configuration."""
    program = SimulationProgram([(_zero_hamiltonian(2), params)], get_state=True)

    with pytest.raises(ValueError, match=message):
        Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program)


def test_program_executor_rejects_corrupted_private_instructions() -> None:
    """The private executor reports an instruction outside the compiler's closed union."""
    state = State(2, initial="zeros")
    program = SimulationProgram([(QuantumCircuit(2), DigitalSimParams())])
    compiled = _compile_program(program, state)
    corrupted = replace(compiled, instructions=(object(),))

    with pytest.raises(TypeError, match="Unknown instruction type object"):
        simulator_module._execute_program_trajectory(0, state.mps, corrupted, 1)  # ruff: ignore[private-member-access]

    invalid_segment = SimpleNamespace(sim_params=None, noise_model=None)
    object.__setattr__(  # ruff: ignore[unnecessary-dunder-call]  # deliberately corrupt frozen invariant
        program, "segments", (invalid_segment,)
    )
    with pytest.raises(TypeError, match=r"segments\[0\].*unsupported private segment type SimpleNamespace"):
        _compile_program(program, state)


def test_program_executor_requires_propagated_segment_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend that omits state handoff fails at the segment boundary."""
    state = State(2, initial="zeros")
    program = SimulationProgram([(_zero_hamiltonian(2), AnalogSimParams())])
    compiled = _compile_program(program, state)

    def omit_state(_args: object, **_kwargs: object) -> tuple[None, None, None]:
        return None, None, None

    monkeypatch.setattr(simulator_module, "analog_tjm_1", omit_state)

    with pytest.raises(RuntimeError, match="did not return its propagated state"):
        simulator_module._execute_program_trajectory(0, state.mps, compiled, 1)  # ruff: ignore[private-member-access]


def test_order_two_single_step_program_returns_propagated_state() -> None:
    """A one-step order-2 analog segment returns its state for program handoff."""
    params = AnalogSimParams(elapsed_time=0.1, dt=0.1, order=2)
    program = SimulationProgram([(_zero_hamiltonian(2), params)], get_state=True)

    result = Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program)

    assert result.output_state is not None
    np.testing.assert_allclose(np.abs(result.output_state.mps.to_vec()), np.array([1.0, 0.0, 0.0, 0.0]))


def test_program_noise_default_and_empty_segment_override() -> None:
    """Segments inherit one sampled run model while an empty override disables it."""
    params = AnalogSimParams(
        elapsed_time=0.2,
        dt=0.1,
    )
    disabled_noise = NoiseModel()
    distributed_noise = NoiseModel([
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "normal", "mean": 0.2, "std": 0.01},
        }
    ])
    program = SimulationProgram(
        [
            (_zero_hamiltonian(2), params),
            (QuantumCircuit(2), DigitalSimParams(), disabled_noise),
            (_zero_hamiltonian(2), params),
        ],
        observables=[Observable("z", 0)],
        num_traj=3,
        random_seed=17,
    )

    result = Simulator(parallel=False, show_progress=False).run(
        State(2, initial="zeros"),
        program,
        noise_model=distributed_noise,
    )

    assert result.noise_model is not None
    assert isinstance(result.noise_model.processes[0]["strength"], float)
    assert isinstance(distributed_noise.processes[0]["strength"], dict)
    assert result.segment_results[0].noise_model is result.noise_model
    assert result.segment_results[2].noise_model is result.noise_model
    assert result.segment_results[1].noise_model is not result.noise_model
    assert result.segment_results[1].noise_model is not None
    assert result.segment_results[1].noise_model.processes == []
    assert result.segment_results[0].trajectories[0].shape == (3, 3)
    times = result.segment_results[0].times
    assert times is not None
    np.testing.assert_allclose(times, np.array([0.0, 0.1, 0.2]))


def test_noisy_program_preserves_digital_shot_budget() -> None:
    """A noisy shots-only program executes one full trajectory per requested shot."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    params = DigitalSimParams(shots=3)
    program = SimulationProgram([(circuit, params)], num_traj=5, random_seed=23)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.1}])

    result = Simulator(parallel=False, show_progress=False).run(
        State(2, initial="zeros"),
        program,
        noise_model=noise_model,
    )

    segment_result = result.segment_results[0]
    assert len(segment_result.measurements) == 3
    assert segment_result.counts is not None
    assert sum(segment_result.counts.values()) == 3
    assert result.counts is not None
    assert sum(result.counts.values()) == 3


def test_noisy_program_uses_num_traj_when_observables_and_shots_are_combined() -> None:
    """Observable ensembles retain the program count and share the total shot budget."""
    params = DigitalSimParams(shots=3)
    program = SimulationProgram(
        [(QuantumCircuit(2), params)],
        observables=[Observable("z", 0)],
        num_traj=5,
        random_seed=23,
    )

    result = Simulator(parallel=False, show_progress=False).run(
        State(2, initial="zeros"),
        program,
        noise_model=NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.1}]),
    )

    segment_result = result.segment_results[0]
    assert len(segment_result.measurements) == 5
    assert segment_result.trajectories[0].shape[0] == 5
    assert segment_result.counts is not None
    assert sum(segment_result.counts.values()) == 3


def test_program_threads_one_rng_stream_across_analog_and_digital_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One trajectory RNG advances across every stochastic segment boundary."""
    samples: list[float] = []

    def record_rng(
        state: MPS,
        noise_model: NoiseModel | None,
        dt: float,
        sim_params: AnalogSimParams | DigitalSimParams,
        rng: np.random.Generator | None = None,
    ) -> MPS:
        del noise_model, dt, sim_params
        assert rng is not None
        samples.append(float(rng.random()))
        return state

    monkeypatch.setattr(analog_module, "stochastic_process", record_rng)
    monkeypatch.setattr(digital_module, "stochastic_process", record_rng)

    analog_params = AnalogSimParams(
        elapsed_time=0.1,
        dt=0.1,
        order=1,
    )
    digital_params = DigitalSimParams()
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    program = SimulationProgram(
        [
            (_zero_hamiltonian(2), analog_params),
            (circuit, digital_params),
            (_zero_hamiltonian(2), analog_params),
        ],
        observables=[Observable("z", 0)],
        num_traj=2,
        random_seed=31,
    )
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.1}])

    Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program, noise_model=noise_model)

    expected = [
        *make_trajectory_rng(0, base_seed=31).random(3),
        *make_trajectory_rng(1, base_seed=31).random(3),
    ]
    np.testing.assert_allclose(samples, expected, rtol=0, atol=0)


def test_order_two_program_handoff_uses_continuous_trajectory_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    """Order-2 final sampling advances the owned trajectory RNG at segment boundaries."""
    samples: list[float] = []

    def record_rng(
        state: MPS,
        noise_model: NoiseModel | None,
        dt: float,
        sim_params: AnalogSimParams,
        rng: np.random.Generator | None = None,
    ) -> MPS:
        del noise_model, dt, sim_params
        assert rng is not None
        samples.append(float(rng.random()))
        return state

    monkeypatch.setattr(analog_module, "stochastic_process", record_rng)
    params = AnalogSimParams(
        elapsed_time=0.1,
        dt=0.1,
        order=2,
    )
    program = SimulationProgram(
        [
            (_zero_hamiltonian(1), params),
            (_zero_hamiltonian(1), params),
        ],
        observables=[Observable("z", 0)],
        num_traj=1,
        random_seed=31,
    )
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.1}])

    Simulator(parallel=False, show_progress=False).run(State(1, initial="zeros"), program, noise_model=noise_model)

    expected = make_trajectory_rng(0, base_seed=31).random(4)
    np.testing.assert_allclose(samples, expected, rtol=0, atol=0)


def test_order_two_program_uses_distinct_segment_sample_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intermediate measurement copies include the program segment in their seed coordinates."""
    streams: list[tuple[int | None, int]] = []
    original_make_sample_rng = analog_module.make_sample_rng

    def record_sample_stream(
        traj_idx: int,
        *,
        base_seed: int | None,
        timestep: int,
        stream_id: int | None = None,
    ) -> np.random.Generator:
        streams.append((stream_id, timestep))
        return original_make_sample_rng(
            traj_idx,
            base_seed=base_seed,
            timestep=timestep,
            stream_id=stream_id,
        )

    monkeypatch.setattr(analog_module, "make_sample_rng", record_sample_stream)
    params = AnalogSimParams(
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        order=2,
    )
    program = SimulationProgram(
        [
            (_zero_hamiltonian(1), params),
            (_zero_hamiltonian(1), params),
        ],
        observables=[Observable("z", 0)],
        num_traj=1,
        random_seed=31,
    )

    Simulator(parallel=False, show_progress=False).run(
        State(1, initial="zeros"),
        program,
        noise_model=NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.1}]),
    )

    assert streams == [(0, 1), (1, 1)]


def _run_seeded_noisy_program(*, parallel: bool) -> list[list[np.ndarray]]:
    """Run a small noisy mixed program.

    Returns:
        Per-observable trajectory arrays grouped by segment.
    """
    analog_params = AnalogSimParams(
        elapsed_time=0.1,
        dt=0.1,
        order=1,
    )
    digital_params = DigitalSimParams()
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    program = SimulationProgram(
        [
            (_zero_hamiltonian(2), analog_params),
            (circuit, digital_params),
            (_zero_hamiltonian(2), analog_params),
        ],
        observables=[Observable("z", 0)],
        num_traj=4,
        random_seed=101,
    )
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.3}])
    result = Simulator(parallel=parallel, max_workers=2, show_progress=False).run(
        State(2, initial="zeros"),
        program,
        noise_model=noise_model,
    )
    return [[trajectory.copy() for trajectory in segment.trajectories] for segment in result.segment_results]


def test_seeded_noisy_program_matches_in_serial_and_parallel() -> None:
    """Trajectory indices determine streams independently of worker ordering."""
    serial = _run_seeded_noisy_program(parallel=False)
    parallel = _run_seeded_noisy_program(parallel=True)

    for serial_segment, parallel_segment in zip(serial, parallel, strict=True):
        for serial_trajectory, parallel_trajectory in zip(serial_segment, parallel_segment, strict=True):
            np.testing.assert_allclose(serial_trajectory, parallel_trajectory, rtol=0, atol=0)


def test_noisy_program_rejects_requested_trajectory_state() -> None:
    """A stochastic ensemble has no single representative output state."""
    params = AnalogSimParams(elapsed_time=0.1, dt=0.1)
    program = SimulationProgram([(_zero_hamiltonian(2), params)], num_traj=2, get_state=True)
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.1}])

    with pytest.raises(ValueError, match="Cannot return state from a noisy SimulationProgram"):
        Simulator(parallel=False, show_progress=False).run(
            State(2, initial="zeros"),
            program,
            noise_model=noise_model,
        )


def test_program_rejects_program_owned_fields_on_segment_params() -> None:
    """Segment params must leave observables empty and random_seed unset."""
    with pytest.raises(ValueError, match=r"sim_params.observables must be empty"):
        SimulationProgram([
            (_zero_hamiltonian(2), AnalogSimParams(observables=[Observable("z", 0)])),
            (QuantumCircuit(2), DigitalSimParams()),
        ])
    with pytest.raises(ValueError, match=r"sim_params.random_seed must be None"):
        SimulationProgram([
            (_zero_hamiltonian(2), AnalogSimParams(random_seed=1)),
            (QuantumCircuit(2), DigitalSimParams(random_seed=2)),
        ])


def test_program_num_traj_configures_the_ensemble() -> None:
    """The program count wins without mutating lower-level parameter objects."""
    analog_params = AnalogSimParams()
    digital_params = DigitalSimParams(shots=3, preset="fast")
    program = SimulationProgram(
        [
            (_zero_hamiltonian(2), analog_params),
            (QuantumCircuit(2), digital_params),
        ],
        observables=[Observable("z", 0)],
        num_traj=4,
    )

    result = Simulator(parallel=False, show_progress=False).run(
        State(2, initial="zeros"),
        program,
        noise_model=NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.1}]),
    )

    assert result.segment_results[0].trajectories[0].shape[0] == 4
    assert len(result.segment_results[1].measurements) == 4
    assert analog_params.num_traj == AnalogSimParams().num_traj
    assert digital_params.num_traj == DigitalSimParams(shots=3, preset="fast").num_traj


def test_program_rejects_scheduled_jumps_until_program_time_semantics_are_defined() -> None:
    """Segment-local clocks must not silently repeat run-level scheduled jumps."""
    scheduled = NoiseModel(scheduled_jumps=[{"time": 0.1, "sites": [0], "name": "pauli_x"}])
    program = SimulationProgram(
        [
            (_zero_hamiltonian(1), AnalogSimParams()),
            (_zero_hamiltonian(1), AnalogSimParams()),
        ],
        get_state=True,
    )

    with pytest.raises(ValueError, match=r"scheduled_jumps.*not supported"):
        Simulator(parallel=False, show_progress=False).run(
            State(1, initial="zeros"),
            program,
            noise_model=scheduled,
        )


def test_noiseless_program_ignores_trajectory_configuration_without_warning() -> None:
    """Trajectory precedence is irrelevant to noiseless execution."""
    params = AnalogSimParams()
    program = SimulationProgram(
        [(_zero_hamiltonian(2), params)],
        observables=[Observable("z", 0)],
        num_traj=4,
    )

    result = Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program)

    assert result.segment_results[0].trajectories[0].shape[0] == 1


def _heterogeneous_zero_hamiltonian(physical_dimensions: list[int]) -> Hamiltonian:
    """Return a zero MPO with physical legs matching a heterogeneous layout."""
    tensors = [np.zeros((dimension, dimension, 1, 1), dtype=np.complex128) for dimension in physical_dimensions]
    mpo = MPO()
    mpo.custom(tensors, transpose=False)
    return Hamiltonian.from_mpo(mpo)


def test_heterogeneous_program_preserves_idle_spectator() -> None:
    """Analog evolution and qubit gates preserve an idle non-qubit site."""
    dimensions = [2, 2, 3]
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.cx(0, 1)
    program = SimulationProgram(
        [
            (
                _heterogeneous_zero_hamiltonian(dimensions),
                AnalogSimParams(elapsed_time=0.1, dt=0.1),
            ),
            (circuit, DigitalSimParams()),
        ],
        get_state=True,
    )

    result = Simulator(parallel=False, show_progress=False).run(
        State(3, initial="zeros", physical_dimensions=dimensions),
        program,
    )

    assert result.output_state is not None
    assert result.output_state.mps.physical_dimensions == dimensions
    assert result.output_state.mps.project_onto_bitstring("110") == pytest.approx(1.0)


@pytest.mark.parametrize("gate_mode", ["mpo", "tdvp", "full-tdvp"])
def test_long_range_qubit_gate_crosses_non_qubit_spectator(gate_mode: GateMode) -> None:
    """Supported non-SWAP modes derive spectator identities from the MPS layout."""
    dimensions = [2, 3, 2]
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.cx(0, 2)
    params = DigitalSimParams(gate_mode=gate_mode, max_bond_dim=8, svd_threshold=1e-12)
    program = SimulationProgram([(circuit, params)], get_state=True)

    result = Simulator(parallel=False, show_progress=False).run(
        State(3, initial="zeros", physical_dimensions=dimensions),
        program,
    )

    assert result.output_state is not None
    assert result.output_state.mps.physical_dimensions == dimensions
    assert result.output_state.mps.project_onto_bitstring("101") == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("gate_mode", ["mpo", "swaps"])
def test_multi_qubit_gate_crosses_non_qubit_spectator(gate_mode: GateMode) -> None:
    """Native three-qubit gate MPOs retain every target around a non-qubit spectator."""
    dimensions = [2, 2, 3, 2]
    circuit = QuantumCircuit(4)
    circuit.x(0)
    circuit.x(1)
    circuit.ccx(0, 1, 3)
    params = DigitalSimParams(gate_mode=gate_mode, max_bond_dim=16, svd_threshold=1e-12)

    result = Simulator(parallel=False, show_progress=False).run(
        State(4, initial="zeros", physical_dimensions=dimensions),
        SimulationProgram([(circuit, params)], get_state=True),
    )

    assert result.output_state is not None
    assert result.output_state.mps.physical_dimensions == dimensions
    assert result.output_state.mps.project_onto_bitstring("1101") == pytest.approx(1.0, abs=1e-8)


def test_heterogeneous_program_rejects_incompatible_gate_target_and_swap_route() -> None:
    """Compilation reports incompatible targets and heterogeneous SWAP routing."""
    target_circuit = QuantumCircuit(3)
    target_circuit.x(2)
    with pytest.raises(ValueError, match="targets site 2 with physical dimension 3"):
        Simulator(parallel=False, show_progress=False).run(
            State(3, initial="zeros", physical_dimensions=[2, 2, 3]),
            SimulationProgram([(target_circuit, DigitalSimParams())], get_state=True),
        )

    routed_circuit = QuantumCircuit(3)
    routed_circuit.cx(0, 2)
    with pytest.raises(ValueError, match=r"cannot route.*non-qubit spectator"):
        Simulator(parallel=False, show_progress=False).run(
            State(3, initial="zeros", physical_dimensions=[2, 3, 2]),
            SimulationProgram([(routed_circuit, DigitalSimParams(gate_mode="swaps"))], get_state=True),
        )


def test_program_translates_digital_gates_once_for_all_trajectories(monkeypatch: pytest.MonkeyPatch) -> None:
    """Circuit-to-gate translation is shared rather than repeated per trajectory."""
    calls = 0
    original_convert = digital_module.convert_dag_to_tensor_algorithm

    def counted_convert(node: DAGOpNode) -> list[BaseGate]:
        nonlocal calls
        calls += 1
        return original_convert(node)

    monkeypatch.setattr(digital_module, "convert_dag_to_tensor_algorithm", counted_convert)
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    params = DigitalSimParams()
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.1}])

    Simulator(parallel=False, show_progress=False).run(
        State(2, initial="zeros"),
        SimulationProgram(
            [(circuit, params)],
            observables=[Observable("z", 0)],
            num_traj=4,
            random_seed=19,
        ),
        noise_model=noise_model,
    )

    assert calls == 1


def test_compiled_gates_are_executor_owned_and_never_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated trajectory execution leaves executor-owned compiled gates unchanged."""
    compiled_ids: list[int] = []
    original_digital_tjm = simulator_module.digital_tjm

    def fingerprint(compiled: _CompiledCircuit) -> tuple[object, ...]:
        """Return mutation-sensitive metadata for every gate in a compiled circuit."""

        def arrays(gate: BaseGate) -> tuple[bytes, ...]:
            values = [gate.matrix, gate.tensor]
            generator = getattr(gate, "generator", None)
            if isinstance(generator, np.ndarray):
                values.append(generator)
            elif generator is not None:
                values.extend(generator)
            with contextlib.suppress(AttributeError):
                values.extend(gate.mpo_tensors)
            return tuple(np.asarray(value).tobytes() for value in values)

        return tuple(
            (id(gate), tuple(gate.sites), arrays(gate))
            for layer in compiled.layers
            for gate in (*layer.single_qubit_gates, *layer.even_two_qubit_gates, *layer.odd_two_qubit_gates)
        )

    def recording_digital_tjm(
        args: tuple[int, MPS, NoiseModel | None, DigitalSimParams, QuantumCircuit],
        *,
        copy_initial_state: bool = True,
        rng: np.random.Generator | None = None,
        compiled_circuit: _CompiledCircuit | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[int, int] | None, MPS | None]:
        assert compiled_circuit is not None
        before = fingerprint(compiled_circuit)
        result = original_digital_tjm(
            args,
            copy_initial_state=copy_initial_state,
            rng=rng,
            compiled_circuit=compiled_circuit,
        )
        assert fingerprint(compiled_circuit) == before
        compiled_ids.append(id(compiled_circuit))
        return result

    monkeypatch.setattr(simulator_module, "digital_tjm", recording_digital_tjm)
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.cx(0, 2)
    params = DigitalSimParams(
        gate_mode="mpo",
        max_bond_dim=8,
        svd_threshold=1e-12,
    )

    Simulator(parallel=False, show_progress=False).run(
        State(3, initial="zeros"),
        SimulationProgram(
            [(circuit, params)],
            observables=[Observable("z", 0)],
            num_traj=2,
            random_seed=41,
        ),
        noise_model=NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 0.1}]),
    )

    assert len(compiled_ids) == 2
    assert len(set(compiled_ids)) == 1


def test_program_makes_one_owned_state_copy_not_per_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A noiseless program copies its MPS once, not at each segment boundary."""
    copied_mps = 0
    original_deepcopy = copy.deepcopy

    def counted_deepcopy(value: object) -> object:
        nonlocal copied_mps
        if isinstance(value, digital_module.MPS):
            copied_mps += 1
        return original_deepcopy(value)

    monkeypatch.setattr(simulator_module, "copy", SimpleNamespace(copy=copy.copy, deepcopy=counted_deepcopy))
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    program = SimulationProgram(
        [
            (circuit, DigitalSimParams()),
            (_zero_hamiltonian(2), AnalogSimParams(elapsed_time=0.1, dt=0.1)),
        ],
        observables=[Observable("z", 0)],
    )

    Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program)

    assert copied_mps == 1


def test_noisy_digital_then_analog_program_preserves_reverse_order() -> None:
    """Whole-trajectory execution also supports digital-to-analog ordering."""
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.cx(0, 1)
    params = AnalogSimParams(
        elapsed_time=0.1,
        dt=0.1,
    )
    program = SimulationProgram(
        [
            (circuit, DigitalSimParams()),
            (_zero_hamiltonian(2), params),
        ],
        observables=[Observable("z", 0)],
        num_traj=4,
        random_seed=13,
    )

    result = Simulator(parallel=False, show_progress=False).run(
        State(2, initial="zeros"),
        program,
        noise_model=NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 0.1}]),
    )

    assert [segment.segment_type for segment in result.segment_results] == ["digital", "analog"]
    assert result.segment_results[1].expectation_values[0][-1] == pytest.approx(-1.0)
    assert result.expectation_values[0][-1] == pytest.approx(-1.0)


def test_program_rejects_noise_operator_incompatible_with_non_qubit_site() -> None:
    """Noise layout errors are reported before a trajectory enters tensor code."""
    circuit = QuantumCircuit(3)
    noise_model = NoiseModel([{"name": "pauli_z", "sites": [2], "strength": 0.1}])

    with pytest.raises(ValueError, match=r"noise operator on sites \[2\].*expected \(3, 3\)"):
        Simulator(parallel=False, show_progress=False).run(
            State(3, initial="zeros", physical_dimensions=[2, 2, 3]),
            SimulationProgram([(circuit, DigitalSimParams())], get_state=True),
            noise_model=noise_model,
        )


def test_program_preserves_contextual_noise_validation() -> None:
    """Programs reject noise models unsupported by their analog or digital backend."""
    digital_noise = NoiseModel([
        {"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 0.1},
    ])
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    with pytest.raises(ValueError, match="Digital TJM does not support non-adjacent"):
        Simulator(parallel=False, show_progress=False).run(
            State(3, initial="zeros"),
            SimulationProgram([(circuit, DigitalSimParams())], get_state=True),
            noise_model=digital_noise,
        )

    analog_noise = NoiseModel([
        {
            "name": "custom",
            "sites": [0, 2],
            "strength": 0.1,
            "factors": (np.array([[0, 1], [0, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]])),
        }
    ])
    with pytest.raises(ValueError, match="Analog MPS TJM does not support non-Pauli long-range"):
        Simulator(parallel=False, show_progress=False).run(
            State(3, initial="zeros"),
            SimulationProgram([(_zero_hamiltonian(3), AnalogSimParams())], get_state=True),
            noise_model=analog_noise,
        )


def _run_spin_echo(*, coupling: float, include_pulse: bool) -> tuple[float, float]:
    """Run a two-spin echo.

    Returns:
        Final-state fidelity and transverse magnetization.
    """
    length = 2
    half_duration = 0.4
    hamiltonian = Hamiltonian.heisenberg(length, Jx=0.0, Jy=0.0, Jz=coupling, h=1.1)
    observables = [Observable("x", site) for site in range(length)]
    analog_params = AnalogSimParams(
        elapsed_time=half_duration,
        dt=0.05,
        max_bond_dim=8,
        svd_threshold=1e-12,
        order=2,
    )
    segments: list[tuple[object, object]] = [(hamiltonian, analog_params)]
    if include_pulse:
        pulse = QuantumCircuit(length)
        pulse.x(range(length))
        segments.append((pulse, DigitalSimParams()))
    segments.append((hamiltonian, analog_params))
    if include_pulse:
        segments.append((pulse, DigitalSimParams()))

    initial_state = State(length, initial="x+")
    initial_vector = initial_state.mps.to_vec().copy()
    result = Simulator(parallel=False, show_progress=False).run(
        initial_state,
        SimulationProgram(
            segments,  # ty: ignore[invalid-argument-type]
            observables=observables,
            get_state=True,
        ),
    )

    assert result.output_state is not None
    final_vector = result.output_state.mps.to_vec()
    fidelity = float(np.abs(np.vdot(initial_vector, final_vector)) ** 2)
    final_analog_result = next(
        segment for segment in reversed(result.segment_results) if segment.segment_type == "analog"
    )
    transverse_magnetization = float(np.mean([values[-1].real for values in final_analog_result.expectation_values]))
    return fidelity, transverse_magnetization


def test_spin_echo_refocuses_field_but_not_interactions() -> None:
    """A digital echo pulse refocuses fields while leaving interaction dynamics."""
    echo_fidelity, echo_magnetization = _run_spin_echo(coupling=0.0, include_pulse=True)
    no_pulse_fidelity, no_pulse_magnetization = _run_spin_echo(coupling=0.0, include_pulse=False)
    interacting_fidelity, interacting_magnetization = _run_spin_echo(coupling=0.7, include_pulse=True)

    assert echo_fidelity == pytest.approx(1.0, abs=1e-10)
    assert echo_magnetization == pytest.approx(1.0, abs=1e-10)
    assert no_pulse_fidelity < 0.2
    assert no_pulse_magnetization < 0.0
    assert interacting_fidelity < 0.8
    assert interacting_magnetization < 0.5


def _run_noisy_spin_echo(*, include_pulse: bool) -> float:
    """Return final transverse magnetization under existing Markovian dephasing."""
    length = 2
    hamiltonian = Hamiltonian.heisenberg(length, Jx=0.0, Jy=0.0, Jz=0.0, h=1.1)
    params = AnalogSimParams(
        elapsed_time=0.4,
        dt=0.05,
        order=1,
    )
    segments: list[tuple[object, object]] = [(hamiltonian, params)]
    if include_pulse:
        pulse = QuantumCircuit(length)
        pulse.x(range(length))
        segments.append((pulse, DigitalSimParams()))
    segments.append((hamiltonian, params))
    noise_model = NoiseModel([{"name": "pauli_z", "sites": [site], "strength": 0.35} for site in range(length)])

    result = Simulator(parallel=False, show_progress=False).run(
        State(length, initial="x+"),
        SimulationProgram(
            segments,  # ty: ignore[invalid-argument-type]
            observables=[Observable("x", site) for site in range(length)],
            num_traj=128,
            random_seed=2026,
        ),
        noise_model=noise_model,
    )
    final_analog = result.segment_results[-1]
    return float(np.mean([values[-1].real for values in final_analog.expectation_values]))


def test_hahn_echo_refocuses_detuning_but_not_markovian_dephasing() -> None:
    """The pulse restores phase while the existing Lindblad envelope remains."""
    echo_magnetization = _run_noisy_spin_echo(include_pulse=True)
    no_pulse_magnetization = _run_noisy_spin_echo(include_pulse=False)

    assert 0.3 < echo_magnetization < 0.9
    assert no_pulse_magnetization < 0.2
    assert echo_magnetization > no_pulse_magnetization + 0.3
