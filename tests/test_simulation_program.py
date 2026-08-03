# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for noiseless composable simulation programs."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import (
    AnalogSegment,
    AnalogSimParams,
    DigitalSegment,
    DigitalSimParams,
    Hamiltonian,
    NoiseModel,
    Observable,
    SimulationProgram,
    Simulator,
    State,
)


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
    digital_params = DigitalSimParams(observables=[Observable("z", 0)], get_state=True)
    analog_params = AnalogSimParams(
        observables=[Observable("z", 0), Observable("z", 1)],
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
            DigitalSegment(preparation, sim_params=digital_params),
            AnalogSegment(hamiltonian, sim_params=analog_params),
            DigitalSegment(intervention, sim_params=outputless_params),
        ],
        get_state=True,
    )

    result = Simulator(parallel=False, show_progress=False).run(initial_state, program)

    manual_simulator = Simulator(parallel=False, show_progress=False)
    manual_first = manual_simulator.run(State(length, initial="zeros"), preparation, digital_params)
    assert manual_first.output_state is not None
    manual_second = manual_simulator.run(manual_first.output_state, hamiltonian, analog_params)
    assert manual_second.output_state is not None
    manual_final = manual_simulator.run(
        manual_second.output_state,
        intervention,
        DigitalSimParams(get_state=True),
    )

    assert result.sim_params is None
    assert result.output_state is not None
    assert manual_final.output_state is not None
    np.testing.assert_allclose(result.output_state.mps.to_vec(), manual_final.output_state.mps.to_vec(), atol=1e-10)
    np.testing.assert_allclose(initial_state.mps.to_vec(), initial_vector, atol=1e-12)
    assert [segment.segment_type for segment in result.segment_results] == ["digital", "analog", "digital"]
    assert [segment.segment_index for segment in result.segment_results] == [0, 1, 2]
    assert [segment.time_offset for segment in result.segment_results] == [0.0, 0.0, 0.1]
    assert result.segment_results[0].sim_params is digital_params
    assert result.segment_results[1].sim_params is analog_params
    assert result.segment_results[2].sim_params is outputless_params
    assert result.segment_results[0].output_state is not None
    assert result.segment_results[1].output_state is not None
    assert result.segment_results[2].output_state is None
    analog_times = result.segment_results[1].times
    assert analog_times is not None
    np.testing.assert_allclose(analog_times, np.array([0.0, 0.1]))


def test_program_resolves_omitted_segment_parameters() -> None:
    """Ordinary parameter defaults support state-only internal propagation."""
    circuit = QuantumCircuit(2)
    circuit.x(0)
    program = SimulationProgram(
        [
            AnalogSegment(_zero_hamiltonian(2)),
            DigitalSegment(circuit),
        ],
        get_state=True,
    )

    result = Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program)

    assert result.output_state is not None
    assert isinstance(result.segment_results[0].sim_params, AnalogSimParams)
    assert isinstance(result.segment_results[1].sim_params, DigitalSimParams)
    np.testing.assert_allclose(np.abs(result.output_state.mps.to_vec()), np.array([0.0, 1.0, 0.0, 0.0]))


def test_program_propagates_state_without_exposing_final_state() -> None:
    """Program and segment get_state flags affect output, not internal handoff."""
    preparation = QuantumCircuit(2)
    preparation.x(0)
    observation_params = DigitalSimParams(observables=[Observable("z", 0)])
    program = SimulationProgram([
        DigitalSegment(preparation, sim_params=DigitalSimParams()),
        AnalogSegment(
            _zero_hamiltonian(2),
            sim_params=AnalogSimParams(elapsed_time=0.1, dt=0.1),
        ),
        DigitalSegment(QuantumCircuit(2), sim_params=observation_params),
    ])

    result = Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program)

    assert result.output_state is None
    assert all(segment.output_state is None for segment in result.segment_results)
    assert result.segment_results[2].expectation_values[0][0] == pytest.approx(-1.0)


def test_digital_program_segment_returns_requested_shot_counts() -> None:
    """Digital shot output remains scoped to its segment result."""
    shots = 8
    program = SimulationProgram([DigitalSegment(QuantumCircuit(2), sim_params=DigitalSimParams(shots=shots))])

    result = Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program)

    segment_result = result.segment_results[0]
    assert segment_result.counts is not None
    assert sum(segment_result.counts.values()) == shots
    assert segment_result.output_state is None


def test_program_call_contract_is_distinct_from_standalone_run() -> None:
    """Program calls reject standalone parameters, noise, and state lists early."""
    program = SimulationProgram([DigitalSegment(QuantumCircuit(2))])
    simulator = Simulator(parallel=False, show_progress=False)
    state = State(2, initial="zeros")

    with pytest.raises(TypeError, match="sim_params must be None"):
        simulator.run(state, program, DigitalSimParams())
    with pytest.raises(ValueError, match="noise_model must be None"):
        simulator.run(state, program, noise_model=NoiseModel())
    with pytest.raises(TypeError, match="single State"):
        simulator.run([state], program)
    with pytest.raises(TypeError, match="Standalone simulation requires"):
        simulator.run(state, QuantumCircuit(2))


@pytest.mark.parametrize(
    ("state", "program", "message"),
    [
        (
            State(2, initial="zeros", representation="vector"),
            SimulationProgram([DigitalSegment(QuantumCircuit(2))]),
            "representation='mps'",
        ),
        (
            State(2, initial="zeros", physical_dimensions=3),
            SimulationProgram([AnalogSegment(Hamiltonian.ising(2, J=0.0, g=0.0))]),
            "qubit physical dimensions only",
        ),
        (
            State(2, initial="zeros"),
            SimulationProgram([DigitalSegment(QuantumCircuit(3))]),
            "circuit.num_qubits=3",
        ),
        (
            State(2, initial="zeros"),
            SimulationProgram([AnalogSegment(_zero_hamiltonian(3))]),
            "Hamiltonian.length=3",
        ),
        (
            State(2, initial="zeros"),
            SimulationProgram([DigitalSegment(QuantumCircuit(2), noise_model=NoiseModel())]),
            "noise_model must be None",
        ),
        (
            State(2, initial="zeros"),
            SimulationProgram([
                AnalogSegment(
                    _zero_hamiltonian(2),
                    sim_params=AnalogSimParams(elapsed_time=0.15, dt=0.1, get_state=True),
                )
            ]),
            "integer multiple",
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


def test_order_two_single_step_program_returns_propagated_state() -> None:
    """A one-step order-2 analog segment returns its state for program handoff."""
    params = AnalogSimParams(elapsed_time=0.1, dt=0.1, order=2)
    program = SimulationProgram([AnalogSegment(_zero_hamiltonian(2), sim_params=params)], get_state=True)

    result = Simulator(parallel=False, show_progress=False).run(State(2, initial="zeros"), program)

    assert result.output_state is not None
    np.testing.assert_allclose(np.abs(result.output_state.mps.to_vec()), np.array([1.0, 0.0, 0.0, 0.0]))


def _run_spin_echo(*, coupling: float, include_pulse: bool) -> tuple[float, float]:
    """Run a two-spin echo.

    Returns:
        Final-state fidelity and transverse magnetization.
    """
    length = 2
    half_duration = 0.4
    hamiltonian = Hamiltonian.heisenberg(length, Jx=0.0, Jy=0.0, Jz=coupling, h=1.1)
    observables = [Observable("x", site) for site in range(length)]
    segments: list[AnalogSegment | DigitalSegment] = [
        AnalogSegment(
            hamiltonian,
            sim_params=AnalogSimParams(
                observables=observables,
                elapsed_time=half_duration,
                dt=0.05,
                max_bond_dim=8,
                svd_threshold=1e-12,
                order=2,
            ),
        )
    ]
    if include_pulse:
        pulse = QuantumCircuit(length)
        pulse.x(range(length))
        segments.append(DigitalSegment(pulse))
    segments.append(
        AnalogSegment(
            hamiltonian,
            sim_params=AnalogSimParams(
                observables=observables,
                elapsed_time=half_duration,
                dt=0.05,
                max_bond_dim=8,
                svd_threshold=1e-12,
                order=2,
            ),
        )
    )
    if include_pulse:
        segments.append(DigitalSegment(pulse))

    initial_state = State(length, initial="x+")
    initial_vector = initial_state.mps.to_vec().copy()
    result = Simulator(parallel=False, show_progress=False).run(
        initial_state,
        SimulationProgram(segments, get_state=True),
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
