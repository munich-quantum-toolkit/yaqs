# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Composable analog and digital simulation program specifications."""

from __future__ import annotations

import copy
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

from qiskit.circuit import QuantumCircuit

from ..time_utils import exact_time_grid
from .hamiltonian import Hamiltonian
from .noise_model import NoiseModel
from .simulation_parameters import AnalogSimParams, DigitalSimParams

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.random import Generator

    from .mpo import MPO
    from .mps import MPS
    from .result import Result
    from .state import State

__all__ = ["AnalogSegment", "DigitalSegment", "SimulationProgram"]


@dataclass(frozen=True)
class AnalogSegment:
    """Specification of one static analog part of a simulation program.

    Args:
        hamiltonian: Hamiltonian evolved during this segment.
        sim_params: Analog simulation parameters for the segment. If omitted,
            program execution supplies its documented internal defaults.
        noise_model: Optional segment noise model. Noisy program execution is
            not currently supported.

    Raises:
        TypeError: If an argument does not have the corresponding analog type.
    """

    hamiltonian: Hamiltonian
    _: KW_ONLY
    sim_params: AnalogSimParams | None = None
    noise_model: NoiseModel | None = None

    def __post_init__(self) -> None:
        """Validate the closed analog segment specification.

        Raises:
            TypeError: If any value has the wrong analog type.
        """
        if not isinstance(self.hamiltonian, Hamiltonian):
            msg = f"hamiltonian must be Hamiltonian, got {type(self.hamiltonian).__name__}."
            raise TypeError(msg)
        if self.sim_params is not None and not isinstance(self.sim_params, AnalogSimParams):
            msg = f"sim_params must be AnalogSimParams or None, got {type(self.sim_params).__name__}."
            raise TypeError(msg)
        if self.noise_model is not None and not isinstance(self.noise_model, NoiseModel):
            msg = f"noise_model must be NoiseModel or None, got {type(self.noise_model).__name__}."
            raise TypeError(msg)


@dataclass(frozen=True)
class DigitalSegment:
    """Specification of one digital part of a simulation program.

    Args:
        circuit: Qiskit circuit executed during this segment.
        sim_params: Digital simulation parameters for the segment. If omitted,
            program execution supplies its documented internal defaults.
        noise_model: Optional segment noise model. Noisy program execution is
            not currently supported.

    Raises:
        TypeError: If an argument does not have the corresponding digital type.
    """

    circuit: QuantumCircuit
    _: KW_ONLY
    sim_params: DigitalSimParams | None = None
    noise_model: NoiseModel | None = None

    def __post_init__(self) -> None:
        """Validate the closed digital segment specification.

        Raises:
            TypeError: If any value has the wrong digital type.
        """
        if not isinstance(self.circuit, QuantumCircuit):
            msg = f"circuit must be QuantumCircuit, got {type(self.circuit).__name__}."
            raise TypeError(msg)
        if self.sim_params is not None and not isinstance(self.sim_params, DigitalSimParams):
            msg = f"sim_params must be DigitalSimParams or None, got {type(self.sim_params).__name__}."
            raise TypeError(msg)
        if self.noise_model is not None and not isinstance(self.noise_model, NoiseModel):
            msg = f"noise_model must be NoiseModel or None, got {type(self.noise_model).__name__}."
            raise TypeError(msg)


@dataclass(frozen=True, init=False)
class SimulationProgram:
    """Ordered analog and digital segments that share one evolving state.

    The supplied iterable is copied into a tuple. Its order therefore cannot be
    changed through the caller's original collection after construction.

    Args:
        segments: Non-empty iterable of analog and digital segment specifications.
        get_state: Whether program execution should retain the final state in the
            outer :class:`~mqt.yaqs.Result`.

    Raises:
        TypeError: If ``segments`` is not iterable, contains an unsupported item,
            or ``get_state`` is not a Boolean.
        ValueError: If ``segments`` is empty.
    """

    segments: tuple[AnalogSegment | DigitalSegment, ...]
    get_state: bool

    def __init__(
        self,
        segments: Iterable[AnalogSegment | DigitalSegment],
        *,
        get_state: bool = False,
    ) -> None:
        """Initialize and validate an immutable ordered program.

        Raises:
            TypeError: If the segment collection, an item, or ``get_state`` has
                the wrong type.
            ValueError: If the program has no segments.
        """
        if isinstance(segments, (str, bytes)):
            msg = "segments must be an iterable of AnalogSegment or DigitalSegment."
            raise TypeError(msg)
        try:
            normalized_segments = tuple(segments)
        except TypeError as error:
            msg = "segments must be an iterable of AnalogSegment or DigitalSegment."
            raise TypeError(msg) from error

        if not normalized_segments:
            msg = "SimulationProgram requires at least one segment."
            raise ValueError(msg)
        for index, segment in enumerate(normalized_segments):
            if not isinstance(segment, (AnalogSegment, DigitalSegment)):
                msg = f"segments[{index}] must be AnalogSegment or DigitalSegment, got {type(segment).__name__}."
                raise TypeError(msg)
        if not isinstance(get_state, bool):
            msg = f"get_state must be bool, got {type(get_state).__name__}."
            raise TypeError(msg)

        object.__setattr__(self, "segments", normalized_segments)
        object.__setattr__(self, "get_state", get_state)

    def __iter__(self) -> Iterator[AnalogSegment | DigitalSegment]:
        """Iterate over segments in program order.

        Returns:
            An iterator over the immutable segment tuple.
        """
        return iter(self.segments)

    def __len__(self) -> int:
        """Return the number of segments in the program."""
        return len(self.segments)


@dataclass(frozen=True)
class _StateSignature:
    """State properties preserved by every compiled instruction."""

    representation: str
    length: int
    physical_dimensions: tuple[int, ...]


@dataclass(frozen=True)
class _CompiledAnalogInstruction:
    """Validated analog instruction used by the private program executor."""

    index: int
    hamiltonian: MPO
    sim_params: AnalogSimParams
    execution_params: AnalogSimParams
    time_offset: float


@dataclass(frozen=True)
class _CompiledDigitalInstruction:
    """Validated digital instruction used by the private program executor."""

    index: int
    circuit: QuantumCircuit
    sim_params: DigitalSimParams
    execution_params: DigitalSimParams
    time_offset: float


_CompiledInstruction = _CompiledAnalogInstruction | _CompiledDigitalInstruction


@dataclass(frozen=True)
class _CompiledProgram:
    """Private validated program representation."""

    instructions: tuple[_CompiledInstruction, ...]
    state_signature: _StateSignature
    get_state: bool


@dataclass
class _ProgramExecutionContext:
    """Mutable state shared by instructions during one program execution.

    The current state and signature are replaceable so a future validated
    state-space transition does not require changing the instruction loop.
    ``rng``, ``noise_model``, and ``artifacts`` reserve future noisy-execution
    and transition seams without exposing them in the public program specification.
    """

    current_state: MPS
    state_signature: _StateSignature
    absolute_time: float = 0.0
    segment_results: list[Result] = field(default_factory=list)
    rng: Generator | None = None
    noise_model: NoiseModel | None = None
    artifacts: dict[str, object] = field(default_factory=dict)


def _compile_program(program: SimulationProgram, initial_state: State) -> _CompiledProgram:
    """Validate and compile a program for noiseless MPS execution.

    Args:
        program: Public ordered program specification.
        initial_state: State used to validate the program-wide state signature.

    Returns:
        A private immutable sequence of executable instructions.

    Raises:
        TypeError: If a program contains an unknown segment type.
        ValueError: If the state, dimensions, segment noise, lengths, or analog
            time grids are not currently supported by program execution.
    """
    if initial_state.representation != "mps":
        msg = "SimulationProgram execution currently requires State.representation='mps'."
        raise ValueError(msg)
    physical_dimensions = tuple(initial_state.mps.physical_dimensions)
    if any(dimension != 2 for dimension in physical_dimensions):
        msg = "SimulationProgram execution currently supports qubit physical dimensions only."
        raise ValueError(msg)

    signature = _StateSignature("mps", initial_state.length, physical_dimensions)
    instructions: list[_CompiledInstruction] = []
    time_offset = 0.0

    for index, segment in enumerate(program.segments):
        if segment.noise_model is not None:
            msg = f"segments[{index}].noise_model must be None for noiseless program execution."
            raise ValueError(msg)

        if isinstance(segment, AnalogSegment):
            if segment.hamiltonian.length != signature.length:
                msg = (
                    f"segments[{index}] Hamiltonian.length={segment.hamiltonian.length} "
                    f"does not match State.length={signature.length}."
                )
                raise ValueError(msg)
            if segment.hamiltonian.physical_dimension != 2:
                msg = f"segments[{index}] Hamiltonian must currently use physical_dimension=2."
                raise ValueError(msg)
            sim_params = segment.sim_params if segment.sim_params is not None else AnalogSimParams()
            if sim_params.order not in {1, 2}:
                msg = f"segments[{index}] AnalogSimParams.order must be 1 or 2, got {sim_params.order}."
                raise ValueError(msg)
            if sim_params.multi_time_observables:
                msg = f"segments[{index}] multi_time_observables are not supported in program execution."
                raise ValueError(msg)
            execution_params = copy.deepcopy(sim_params)
            execution_params.times = exact_time_grid(sim_params.elapsed_time, sim_params.dt)
            execution_params.get_state = True
            segment.hamiltonian.ensure_mpo()
            instructions.append(
                _CompiledAnalogInstruction(
                    index=index,
                    hamiltonian=segment.hamiltonian.mpo,
                    sim_params=sim_params,
                    execution_params=execution_params,
                    time_offset=time_offset,
                )
            )
            time_offset += float(sim_params.elapsed_time)
            continue
        if isinstance(segment, DigitalSegment):
            if segment.circuit.num_qubits != signature.length:
                msg = (
                    f"segments[{index}] circuit.num_qubits={segment.circuit.num_qubits} "
                    f"does not match State.length={signature.length}."
                )
                raise ValueError(msg)
            sim_params = segment.sim_params if segment.sim_params is not None else DigitalSimParams()
            execution_params = copy.deepcopy(sim_params)
            execution_params.get_state = True
            instructions.append(
                _CompiledDigitalInstruction(
                    index=index,
                    circuit=segment.circuit,
                    sim_params=sim_params,
                    execution_params=execution_params,
                    time_offset=time_offset,
                )
            )
        else:
            msg = f"segments[{index}] must be AnalogSegment or DigitalSegment, got {type(segment).__name__}."
            raise TypeError(msg)

    return _CompiledProgram(tuple(instructions), signature, program.get_state)
