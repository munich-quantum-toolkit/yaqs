# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for analog and digital program specifications."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]  # controlled test round-trips; no untrusted input
from dataclasses import FrozenInstanceError

import pytest
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import (
    AnalogSegment,
    AnalogSimParams,
    DigitalSegment,
    DigitalSimParams,
    Hamiltonian,
    NoiseModel,
    SimulationProgram,
)


def test_segments_store_typed_specifications() -> None:
    """Analog and digital segments retain their corresponding inputs."""
    hamiltonian = Hamiltonian.ising(2, J=1.0, g=0.5)
    analog_params = AnalogSimParams(elapsed_time=0.1, dt=0.1, get_state=True)
    circuit = QuantumCircuit(2)
    digital_params = DigitalSimParams(get_state=True)
    noise_model = NoiseModel()

    analog = AnalogSegment(hamiltonian, sim_params=analog_params, noise_model=noise_model)
    digital = DigitalSegment(circuit, sim_params=digital_params, noise_model=noise_model)

    assert analog.hamiltonian is hamiltonian
    assert analog.sim_params is analog_params
    assert analog.noise_model is noise_model
    assert digital.circuit is circuit
    assert digital.sim_params is digital_params
    assert digital.noise_model is noise_model


def test_segment_parameters_and_noise_are_optional() -> None:
    """Program execution may resolve omitted per-segment configuration later."""
    analog = AnalogSegment(Hamiltonian.ising(2, J=1.0, g=0.5))
    digital = DigitalSegment(QuantumCircuit(2))

    assert analog.sim_params is None
    assert analog.noise_model is None
    assert digital.sim_params is None
    assert digital.noise_model is None


def test_segments_reject_wrong_operator_types() -> None:
    """Each public segment accepts only its corresponding operator type."""
    with pytest.raises(TypeError, match="hamiltonian must be Hamiltonian"):
        AnalogSegment(object())  # ty: ignore[invalid-argument-type]  # exercise runtime validation
    with pytest.raises(TypeError, match="circuit must be QuantumCircuit"):
        DigitalSegment(object())  # ty: ignore[invalid-argument-type]  # exercise runtime validation


def test_segments_reject_mismatched_parameters() -> None:
    """Analog and digital simulation parameters cannot be interchanged."""
    analog_params = AnalogSimParams(elapsed_time=0.1, dt=0.1, get_state=True)
    digital_params = DigitalSimParams(get_state=True)

    with pytest.raises(TypeError, match="sim_params must be AnalogSimParams"):
        AnalogSegment(
            Hamiltonian.ising(2, J=1.0, g=0.5),
            sim_params=digital_params,  # ty: ignore[invalid-argument-type]  # exercise runtime validation
        )
    with pytest.raises(TypeError, match="sim_params must be DigitalSimParams"):
        DigitalSegment(
            QuantumCircuit(2),
            sim_params=analog_params,  # ty: ignore[invalid-argument-type]  # exercise runtime validation
        )


def test_segments_reject_wrong_noise_model_type() -> None:
    """The reserved noise slot has the existing YAQS NoiseModel type."""
    with pytest.raises(TypeError, match="noise_model must be NoiseModel"):
        AnalogSegment(
            Hamiltonian.ising(2, J=1.0, g=0.5),
            noise_model=object(),  # ty: ignore[invalid-argument-type]  # exercise runtime validation
        )


def test_program_preserves_order_and_defensively_copies_input() -> None:
    """A program stores a stable tuple independent of the caller's list."""
    analog = AnalogSegment(Hamiltonian.ising(2, J=1.0, g=0.5))
    digital = DigitalSegment(QuantumCircuit(2))
    source = [analog, digital]

    program = SimulationProgram(source, num_traj=17, get_state=True)
    source.reverse()

    assert program.segments == (analog, digital)
    assert list(program) == [analog, digital]
    assert len(program) == 2
    assert program.num_traj == 17
    assert program.get_state
    with pytest.raises(FrozenInstanceError):
        program.get_state = False  # ty: ignore[invalid-assignment]  # exercise frozen dataclass


def test_program_rejects_empty_or_invalid_segments() -> None:
    """Programs are non-empty and report an invalid item's exact index."""
    with pytest.raises(ValueError, match="at least one segment"):
        SimulationProgram([])
    with pytest.raises(TypeError, match=r"segments\[1\].*got object"):
        SimulationProgram(
            [DigitalSegment(QuantumCircuit(2)), object()]  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(TypeError, match="segments must be an iterable"):
        SimulationProgram(None)  # ty: ignore[invalid-argument-type]  # exercise runtime validation
    for invalid_segments in ("analog", b"digital"):
        with pytest.raises(TypeError, match="segments must be an iterable"):
            SimulationProgram(invalid_segments)  # ty: ignore[invalid-argument-type]  # exercise runtime validation


def test_program_rejects_non_boolean_get_state() -> None:
    """The program-level output switch does not silently accept integers."""
    with pytest.raises(TypeError, match="get_state must be bool"):
        SimulationProgram(
            [DigitalSegment(QuantumCircuit(2))],
            get_state=1,  # ty: ignore[invalid-argument-type]  # exercise runtime validation
        )


@pytest.mark.parametrize("num_traj", [True, 1.5, "2"])
def test_program_rejects_non_integer_num_traj(num_traj: object) -> None:
    """The program-wide trajectory count does not accept integer-like values."""
    with pytest.raises(TypeError, match="num_traj must be int or None"):
        SimulationProgram(
            [DigitalSegment(QuantumCircuit(2))],
            num_traj=num_traj,  # ty: ignore[invalid-argument-type]  # exercise runtime validation
        )


def test_program_rejects_non_positive_num_traj() -> None:
    """A stochastic ensemble must contain at least one trajectory."""
    with pytest.raises(ValueError, match="num_traj must be at least 1"):
        SimulationProgram([DigitalSegment(QuantumCircuit(2))], num_traj=0)


def test_program_specification_is_pickleable() -> None:
    """A mixed program specification round-trips for future worker execution."""
    program = SimulationProgram(
        [
            DigitalSegment(QuantumCircuit(2), sim_params=DigitalSimParams(get_state=True)),
            AnalogSegment(
                Hamiltonian.ising(2, J=1.0, g=0.5),
                sim_params=AnalogSimParams(elapsed_time=0.1, dt=0.1, get_state=True),
            ),
        ],
        num_traj=7,
        get_state=True,
    )

    restored = pickle.loads(pickle.dumps(program))  # ruff: ignore[suspicious-pickle-usage]  # controlled round-trip

    assert len(restored) == 2
    assert isinstance(restored.segments[0], DigitalSegment)
    assert isinstance(restored.segments[1], AnalogSegment)
    assert restored.num_traj == 7
    assert restored.get_state
