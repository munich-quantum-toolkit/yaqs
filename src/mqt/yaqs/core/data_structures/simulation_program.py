# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Analog and digital simulation program specifications."""

from __future__ import annotations

import copy
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit import QuantumCircuit

from ...digital.digital_tjm import _compile_circuit, _CompiledCircuit
from .hamiltonian import Hamiltonian
from .noise_model import NoiseModel
from .simulation_parameters import AnalogSimParams, DigitalSimParams

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from .mpo import MPO
    from .state import State

__all__ = ["AnalogSegment", "DigitalSegment", "SimulationProgram"]


@dataclass(frozen=True)
class AnalogSegment:
    """Specification of one static analog part of a simulation program.

    Args:
        hamiltonian: Hamiltonian evolved during this segment.
        sim_params: Analog simulation parameters for the segment. If omitted,
            program execution supplies its documented internal defaults.
        noise_model: Optional segment noise model. ``None`` inherits the
            program-wide model supplied to :meth:`~mqt.yaqs.Simulator.run`; an
            explicit empty model disables noise for this segment.

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
        noise_model: Optional segment noise model. ``None`` inherits the
            program-wide model supplied to :meth:`~mqt.yaqs.Simulator.run`; an
            explicit empty model disables noise for this segment.

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
        num_traj: Program-wide trajectory count for stochastic execution. If
            supplied, it overrides values resolved on segment simulation
            parameters. If omitted, execution uses the shared segment value;
            supplied segment values must agree. As in standalone digital
            simulation, a noisy program with shots but no observables instead
            executes one complete-program trajectory per requested shot.
        get_state: Whether program execution should retain the final state in the
            outer :class:`~mqt.yaqs.Result`.

    Raises:
        TypeError: If ``segments`` is not iterable, contains an unsupported item,
            ``num_traj`` is not an integer or ``None``, or ``get_state`` is not a
            Boolean.
        ValueError: If ``segments`` is empty or ``num_traj`` is less than one.
    """

    segments: tuple[AnalogSegment | DigitalSegment, ...]
    num_traj: int | None
    get_state: bool

    def __init__(
        self,
        segments: Iterable[AnalogSegment | DigitalSegment],
        *,
        num_traj: int | None = None,
        get_state: bool = False,
    ) -> None:
        """Initialize and validate an immutable ordered program.

        Raises:
            TypeError: If the segment collection, an item, ``num_traj``, or
                ``get_state`` has the wrong type.
            ValueError: If the program has no segments or ``num_traj`` is less
                than one.
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
        if num_traj is not None and (isinstance(num_traj, bool) or not isinstance(num_traj, int)):
            msg = f"num_traj must be int or None, got {type(num_traj).__name__}."
            raise TypeError(msg)
        if num_traj is not None and num_traj < 1:
            msg = f"num_traj must be at least 1, got {num_traj}."
            raise ValueError(msg)
        if not isinstance(get_state, bool):
            msg = f"get_state must be bool, got {type(get_state).__name__}."
            raise TypeError(msg)

        object.__setattr__(self, "segments", normalized_segments)
        object.__setattr__(self, "num_traj", num_traj)
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
    noise_model: NoiseModel | None
    time_offset: float


@dataclass(frozen=True)
class _CompiledDigitalInstruction:
    """Validated digital instruction used by the private program executor."""

    index: int
    circuit: QuantumCircuit
    compiled_circuit: _CompiledCircuit
    sim_params: DigitalSimParams
    execution_params: DigitalSimParams
    noise_model: NoiseModel | None
    time_offset: float


_CompiledInstruction = _CompiledAnalogInstruction | _CompiledDigitalInstruction


@dataclass(frozen=True)
class _CompiledProgram:
    """Private validated program representation."""

    instructions: tuple[_CompiledInstruction, ...]
    state_signature: _StateSignature
    get_state: bool
    num_traj: int | None
    random_seed: int | None
    num_traj_conflict: bool
    random_seed_conflict: bool
    default_noise_model: NoiseModel | None


def _validate_noise_layout(
    noise_model: NoiseModel | None,
    physical_dimensions: tuple[int, ...],
    *,
    segment_index: int,
) -> None:
    """Validate noise target indices and operator dimensions during compilation.

    Raises:
        ValueError: If a target is invalid or an operator does not match its local dimensions.
    """
    if noise_model is None:
        return
    entries = [*noise_model.processes, *noise_model.scheduled_jumps]
    for entry in entries:
        sites = tuple(entry["sites"])
        if not sites or any(
            not isinstance(site, int) or site < 0 or site >= len(physical_dimensions) for site in sites
        ):
            msg = f"segments[{segment_index}] noise process has invalid target sites {list(sites)}."
            raise ValueError(msg)
        factors = entry.get("factors")
        if factors is not None:
            if len(factors) != len(sites):
                msg = f"segments[{segment_index}] noise process has {len(factors)} factors for {len(sites)} sites."
                raise ValueError(msg)
            for site, factor in zip(sites, factors, strict=True):
                expected = physical_dimensions[site]
                if np.shape(factor) != (expected, expected):
                    msg = (
                        f"segments[{segment_index}] noise operator on site {site} has shape "
                        f"{np.shape(factor)}, expected ({expected}, {expected})."
                    )
                    raise ValueError(msg)
            continue
        matrix = entry.get("matrix")
        if matrix is None:
            continue
        expected = int(np.prod([physical_dimensions[site] for site in sites]))
        if np.shape(matrix) != (expected, expected):
            msg = (
                f"segments[{segment_index}] noise operator on sites {list(sites)} has shape "
                f"{np.shape(matrix)}, expected ({expected}, {expected})."
            )
            raise ValueError(msg)


def _compile_analog_segment(
    segment: AnalogSegment,
    *,
    index: int,
    signature: _StateSignature,
    noise_model: NoiseModel | None,
    time_offset: float,
    num_traj: int | None,
    random_seed: int | None,
    random_seed_conflict: bool,
) -> _CompiledAnalogInstruction:
    """Compile one analog segment after program-wide settings are resolved.

    Returns:
        The validated private analog instruction.

    Raises:
        ValueError: If the segment is incompatible with the state or program executor.
    """
    if segment.hamiltonian.length != signature.length:
        msg = (
            f"segments[{index}] Hamiltonian.length={segment.hamiltonian.length} "
            f"does not match State.length={signature.length}."
        )
        raise ValueError(msg)
    sim_params = segment.sim_params if segment.sim_params is not None else AnalogSimParams()
    if sim_params.order not in {1, 2}:
        msg = f"segments[{index}] AnalogSimParams.order must be 1 or 2, got {sim_params.order}."
        raise ValueError(msg)
    if sim_params.multi_time_observables:
        msg = f"segments[{index}] multi_time_observables are not supported in program execution."
        raise ValueError(msg)
    execution_params = copy.deepcopy(sim_params)
    execution_params.get_state = True
    if num_traj is not None:
        execution_params.num_traj = num_traj
    if not random_seed_conflict:
        execution_params.random_seed = random_seed
    execution_params.observables = [copy.deepcopy(observable) for observable in sim_params.sorted_observables]
    segment.hamiltonian.ensure_mpo()
    for site, (tensor, dimension) in enumerate(
        zip(segment.hamiltonian.mpo.tensors, signature.physical_dimensions, strict=False)
    ):
        if tensor.ndim != 4 or tensor.shape[0] != dimension or tensor.shape[1] != dimension:
            msg = (
                f"segments[{index}] Hamiltonian MPO site {site} has physical legs "
                f"{tensor.shape[:2]}, expected ({dimension}, {dimension})."
            )
            raise ValueError(msg)
    return _CompiledAnalogInstruction(
        index=index,
        hamiltonian=segment.hamiltonian.mpo,
        sim_params=sim_params,
        execution_params=execution_params,
        noise_model=noise_model,
        time_offset=time_offset,
    )


def _compile_digital_segment(
    segment: DigitalSegment,
    *,
    index: int,
    signature: _StateSignature,
    noise_model: NoiseModel | None,
    time_offset: float,
    num_traj: int | None,
    random_seed: int | None,
    random_seed_conflict: bool,
) -> _CompiledDigitalInstruction:
    """Compile one digital segment after program-wide settings are resolved.

    Returns:
        The validated private digital instruction.

    Raises:
        ValueError: If the circuit size or gate layout is incompatible with the state.
    """
    if segment.circuit.num_qubits != signature.length:
        msg = (
            f"segments[{index}] circuit.num_qubits={segment.circuit.num_qubits} "
            f"does not match State.length={signature.length}."
        )
        raise ValueError(msg)
    sim_params = segment.sim_params if segment.sim_params is not None else DigitalSimParams()
    execution_params = copy.deepcopy(sim_params)
    execution_params.get_state = True
    if num_traj is not None:
        execution_params.num_traj = num_traj
    if not random_seed_conflict:
        execution_params.random_seed = random_seed
    execution_params.observables = [copy.deepcopy(observable) for observable in sim_params.sorted_observables]
    compiled_circuit = _compile_circuit(
        segment.circuit,
        signature.physical_dimensions,
        gate_mode=execution_params.gate_mode,
    )
    if execution_params.sample_layers:
        execution_params.num_mid_measurements = compiled_circuit.num_mid_measurements
    return _CompiledDigitalInstruction(
        index=index,
        circuit=segment.circuit,
        compiled_circuit=compiled_circuit,
        sim_params=sim_params,
        execution_params=execution_params,
        noise_model=noise_model,
        time_offset=time_offset,
    )


def _compile_program(
    program: SimulationProgram,
    initial_state: State,
    default_noise_model: NoiseModel | None = None,
) -> _CompiledProgram:
    """Validate and compile a program for MPS execution.

    Args:
        program: Public ordered program specification.
        initial_state: State used to validate the program-wide state signature.
        default_noise_model: Run-level model inherited by segments whose model
            is ``None``.

    Returns:
        A private immutable sequence of executable instructions.

    Raises:
        TypeError: If a program contains an unknown segment type.
        ValueError: If the state, dimensions, lengths, trajectory settings, or
            analog time grids are not supported by program execution.
    """
    if initial_state.representation != "mps":
        msg = "SimulationProgram execution currently requires State.representation='mps'."
        raise ValueError(msg)
    physical_dimensions = tuple(initial_state.mps.physical_dimensions)
    signature = _StateSignature("mps", initial_state.length, physical_dimensions)
    instructions: list[_CompiledInstruction] = []
    time_offset = 0.0

    segment_num_traj = {segment.sim_params.num_traj for segment in program.segments if segment.sim_params is not None}
    if program.num_traj is not None:
        num_traj = program.num_traj
        num_traj_conflict = False
    elif len(segment_num_traj) == 1:
        num_traj = segment_num_traj.pop()
        num_traj_conflict = False
    elif segment_num_traj:
        num_traj = None
        num_traj_conflict = True
    else:
        num_traj = AnalogSimParams().num_traj
        num_traj_conflict = False

    explicit_seeds = {
        segment.sim_params.random_seed
        for segment in program.segments
        if segment.sim_params is not None and segment.sim_params.random_seed is not None
    }
    random_seed_conflict = len(explicit_seeds) > 1
    random_seed = explicit_seeds.pop() if len(explicit_seeds) == 1 else None

    for index, segment in enumerate(program.segments):
        resolved_noise_model = segment.noise_model if segment.noise_model is not None else default_noise_model
        if resolved_noise_model is not None and resolved_noise_model.scheduled_jumps:
            msg = (
                f"segments[{index}] uses scheduled_jumps, which are not supported in SimulationProgram execution; "
                "their timing semantics across segment-local clocks are not yet defined."
            )
            raise ValueError(msg)
        _validate_noise_layout(resolved_noise_model, signature.physical_dimensions, segment_index=index)

        if isinstance(segment, AnalogSegment):
            instruction = _compile_analog_segment(
                segment,
                index=index,
                signature=signature,
                noise_model=resolved_noise_model,
                time_offset=time_offset,
                num_traj=num_traj,
                random_seed=random_seed,
                random_seed_conflict=random_seed_conflict,
            )
            instructions.append(instruction)
            time_offset += float(instruction.sim_params.elapsed_time)
            continue
        if isinstance(segment, DigitalSegment):
            instructions.append(
                _compile_digital_segment(
                    segment,
                    index=index,
                    signature=signature,
                    noise_model=resolved_noise_model,
                    time_offset=time_offset,
                    num_traj=num_traj,
                    random_seed=random_seed,
                    random_seed_conflict=random_seed_conflict,
                )
            )
        else:
            msg = f"segments[{index}] must be AnalogSegment or DigitalSegment, got {type(segment).__name__}."
            raise TypeError(msg)

    return _CompiledProgram(
        tuple(instructions),
        signature,
        program.get_state,
        num_traj,
        random_seed,
        num_traj_conflict,
        random_seed_conflict,
        default_noise_model,
    )
