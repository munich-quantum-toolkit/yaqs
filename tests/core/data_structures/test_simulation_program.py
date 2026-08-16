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
from typing import TYPE_CHECKING

import pytest
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import (
    AnalogSimParams,
    DigitalSimParams,
    Hamiltonian,
    NoiseModel,
    Observable,
    SimulationProgram,
)
from mqt.yaqs.core.data_structures.simulation_program import (
    _AnalogSegment,  # ruff: ignore[import-private-name]  # private normalized segment type
    _DigitalSegment,  # ruff: ignore[import-private-name]  # private normalized segment type
)

if TYPE_CHECKING:
    from pathlib import Path

_QASM2_X = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
x q[0];
"""


def test_program_normalizes_pair_segments() -> None:
    """``(operator, params[, noise])`` pairs become typed private segments."""
    hamiltonian = Hamiltonian.ising(2, J=1.0, g=0.5)
    analog_params = AnalogSimParams(elapsed_time=0.1, dt=0.1)
    circuit = QuantumCircuit(2)
    digital_params = DigitalSimParams()
    noise_model = NoiseModel()

    program = SimulationProgram([
        (hamiltonian, analog_params, noise_model),
        (circuit, digital_params, noise_model),
    ])

    assert isinstance(program.segments[0], _AnalogSegment)
    assert isinstance(program.segments[1], _DigitalSegment)
    assert program.segments[0].hamiltonian is hamiltonian
    assert program.segments[0].sim_params is analog_params
    assert program.segments[0].noise_model is noise_model
    assert program.segments[1].circuit is circuit
    assert program.segments[1].sim_params is digital_params
    assert program.segments[1].noise_model is noise_model


def test_segment_noise_is_optional() -> None:
    """A two-tuple inherits the run-level noise model later."""
    program = SimulationProgram([
        (Hamiltonian.ising(2, J=1.0, g=0.5), AnalogSimParams()),
        (QuantumCircuit(2), DigitalSimParams()),
    ])

    assert program.segments[0].noise_model is None
    assert program.segments[1].noise_model is None


def test_program_rejects_wrong_operator_or_params_types() -> None:
    """Each pair must combine a matching operator and parameter type."""
    with pytest.raises(TypeError, match=r"segments\[0\] operator must be Hamiltonian, QuantumCircuit"):
        SimulationProgram([(object(), AnalogSimParams())])  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"segments\[0\].*expected AnalogSimParams"):
        SimulationProgram([  # ty: ignore[invalid-argument-type]
            (
                Hamiltonian.ising(2, J=1.0, g=0.5),
                DigitalSimParams(),
            )
        ])
    with pytest.raises(TypeError, match=r"segments\[0\].*expected DigitalSimParams"):
        SimulationProgram([  # ty: ignore[invalid-argument-type]
            (
                QuantumCircuit(2),
                AnalogSimParams(),
            )
        ])


def test_program_normalizes_qasm_string_and_path_operators(tmp_path: Path) -> None:
    """OpenQASM strings and paths become digital segments via ``load_circuit``."""
    qasm_path = tmp_path / "prep.qasm"
    qasm_path.write_text(_QASM2_X, encoding="utf-8")

    from_string = SimulationProgram([(_QASM2_X, DigitalSimParams())])
    from_path = SimulationProgram([(qasm_path, DigitalSimParams())])

    assert isinstance(from_string.segments[0], _DigitalSegment)
    assert isinstance(from_path.segments[0], _DigitalSegment)
    assert from_string.segments[0].circuit.num_qubits == 2
    assert from_path.segments[0].circuit.num_qubits == 2
    assert from_string.segments[0].circuit.data[0].operation.name == "x"
    assert from_path.segments[0].circuit.data[0].operation.name == "x"


def test_program_rejects_wrong_noise_model_type() -> None:
    """The optional third entry must be a NoiseModel when provided."""
    with pytest.raises(TypeError, match=r"segments\[0\] noise_model must be NoiseModel"):
        SimulationProgram([  # ty: ignore[invalid-argument-type]
            (
                Hamiltonian.ising(2, J=1.0, g=0.5),
                AnalogSimParams(),
                object(),
            )
        ])
    with pytest.raises(TypeError, match=r"segments\[0\] noise_model must be NoiseModel"):
        SimulationProgram([  # ty: ignore[invalid-argument-type]
            (
                QuantumCircuit(2),
                DigitalSimParams(),
                object(),
            )
        ])


def test_program_rejects_program_owned_fields_on_segment_params() -> None:
    """Observables, random_seed, and get_state belong on SimulationProgram."""
    with pytest.raises(ValueError, match=r"segments\[0\] sim_params.observables must be empty"):
        SimulationProgram([(Hamiltonian.ising(2, J=1.0, g=0.5), AnalogSimParams(observables=[Observable("z", 0)]))])
    with pytest.raises(ValueError, match=r"segments\[0\] sim_params.random_seed must be None"):
        SimulationProgram([(QuantumCircuit(2), DigitalSimParams(random_seed=1))])
    with pytest.raises(ValueError, match=r"segments\[0\] sim_params.get_state must be False"):
        SimulationProgram([(QuantumCircuit(2), DigitalSimParams(get_state=True))])


def test_program_preserves_order_and_defensively_copies_input() -> None:
    """A program stores a stable tuple independent of the caller's list."""
    hamiltonian = Hamiltonian.ising(2, J=1.0, g=0.5)
    circuit = QuantumCircuit(2)
    analog_params = AnalogSimParams()
    digital_params = DigitalSimParams()
    source: list[tuple[object, object]] = [
        (hamiltonian, analog_params),
        (circuit, digital_params),
    ]

    program = SimulationProgram(
        source,  # ty: ignore[invalid-argument-type]
        observables=[Observable("z", 0)],
        num_traj=17,
        random_seed=3,
        get_state=True,
    )
    source.reverse()

    assert len(program) == 2
    assert list(program) == list(program.segments)
    first = program.segments[0]
    second = program.segments[1]
    assert isinstance(first, _AnalogSegment)
    assert isinstance(second, _DigitalSegment)
    assert first.hamiltonian is hamiltonian
    assert second.circuit is circuit
    assert len(program.observables) == 1
    assert program.observables[0].gate.name == "z"
    assert program.num_traj == 17
    assert program.random_seed == 3
    assert program.get_state
    with pytest.raises(FrozenInstanceError):
        program.get_state = False  # ty: ignore[invalid-assignment]  # exercise frozen dataclass


def test_program_rejects_empty_or_invalid_segments() -> None:
    """Programs are non-empty and report an invalid item's exact index."""
    with pytest.raises(ValueError, match="at least one segment"):
        SimulationProgram([])
    with pytest.raises(TypeError, match=r"segments\[1\] must be a \(operator, params\)"):
        SimulationProgram(
            [(QuantumCircuit(2), DigitalSimParams()), object()]  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(TypeError, match="segments must be an iterable"):
        SimulationProgram(None)  # ty: ignore[invalid-argument-type]  # exercise runtime validation
    for invalid_segments in ("analog", b"digital"):
        with pytest.raises(TypeError, match="segments must be an iterable"):
            SimulationProgram(invalid_segments)  # ty: ignore[invalid-argument-type]  # exercise runtime validation


def test_program_rejects_invalid_observables_sequence() -> None:
    """Program-wide observables must be a sequence of Observable."""
    with pytest.raises(TypeError, match="observables must be a sequence"):
        SimulationProgram(
            [(QuantumCircuit(2), DigitalSimParams())],
            observables="z",  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(TypeError, match=r"observables\[0\] must be Observable"):
        SimulationProgram(
            [(QuantumCircuit(2), DigitalSimParams())],
            observables=[object()],  # ty: ignore[invalid-argument-type]
        )


def test_program_rejects_non_boolean_get_state() -> None:
    """The program-level output switch does not silently accept integers."""
    with pytest.raises(TypeError, match="get_state must be bool"):
        SimulationProgram(
            [(QuantumCircuit(2), DigitalSimParams())],
            get_state=1,  # ty: ignore[invalid-argument-type]  # exercise runtime validation
        )


@pytest.mark.parametrize("num_traj", [True, 1.5, "2"])
def test_program_rejects_non_integer_num_traj(num_traj: object) -> None:
    """The program-wide trajectory count does not accept integer-like values."""
    with pytest.raises(TypeError, match="num_traj must be int or None"):
        SimulationProgram(
            [(QuantumCircuit(2), DigitalSimParams())],
            num_traj=num_traj,  # ty: ignore[invalid-argument-type]  # exercise runtime validation
        )


def test_program_rejects_non_positive_num_traj() -> None:
    """A stochastic ensemble must contain at least one trajectory."""
    with pytest.raises(ValueError, match="num_traj must be at least 1"):
        SimulationProgram([(QuantumCircuit(2), DigitalSimParams())], num_traj=0)


@pytest.mark.parametrize("random_seed", [True, 1.5, "2"])
def test_program_rejects_non_integer_random_seed(random_seed: object) -> None:
    """The program-wide seed does not accept integer-like values."""
    with pytest.raises(TypeError, match="random_seed must be int or None"):
        SimulationProgram(
            [(QuantumCircuit(2), DigitalSimParams())],
            random_seed=random_seed,  # ty: ignore[invalid-argument-type]
        )


def test_program_rejects_negative_random_seed() -> None:
    """A program seed must be non-negative when provided."""
    with pytest.raises(ValueError, match="random_seed must be non-negative"):
        SimulationProgram([(QuantumCircuit(2), DigitalSimParams())], random_seed=-1)


def test_program_specification_is_pickleable() -> None:
    """A mixed program specification round-trips for future worker execution."""
    program = SimulationProgram(
        [
            (QuantumCircuit(2), DigitalSimParams()),
            (
                Hamiltonian.ising(2, J=1.0, g=0.5),
                AnalogSimParams(elapsed_time=0.1, dt=0.1),
            ),
        ],
        observables=[Observable("z", 0)],
        num_traj=7,
        random_seed=11,
        get_state=True,
    )

    restored = pickle.loads(pickle.dumps(program))  # ruff: ignore[suspicious-pickle-usage]  # controlled round-trip

    assert len(restored) == 2
    assert isinstance(restored.segments[0], _DigitalSegment)
    assert isinstance(restored.segments[1], _AnalogSegment)
    assert restored.num_traj == 7
    assert restored.random_seed == 11
    assert restored.get_state
    assert len(restored.observables) == 1
