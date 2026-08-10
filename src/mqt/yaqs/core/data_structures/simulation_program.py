# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Analog and digital simulation program specifications."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.circuit import QuantumCircuit

from ...digital.digital_tjm import _compile_circuit, _CompiledCircuit
from .hamiltonian import Hamiltonian
from .noise_model import NoiseModel
from .simulation_parameters import AnalogSimParams, DigitalSimParams, Observable

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .mpo import MPO
    from .result import Result
    from .state import State

__all__ = ["SimulationProgram"]

SegmentInput = (
    tuple[Hamiltonian, AnalogSimParams]
    | tuple[Hamiltonian, AnalogSimParams, NoiseModel | None]
    | tuple[QuantumCircuit, DigitalSimParams]
    | tuple[QuantumCircuit, DigitalSimParams, NoiseModel | None]
)


@dataclass(frozen=True)
class _AnalogSegment:
    """Private analog segment after pair normalization."""

    hamiltonian: Hamiltonian
    _: KW_ONLY
    sim_params: AnalogSimParams
    noise_model: NoiseModel | None = None


@dataclass(frozen=True)
class _DigitalSegment:
    """Private digital segment after pair normalization."""

    circuit: QuantumCircuit
    _: KW_ONLY
    sim_params: DigitalSimParams
    noise_model: NoiseModel | None = None


_ProgramSegment = _AnalogSegment | _DigitalSegment


def _reject_program_owned_fields(sim_params: AnalogSimParams | DigitalSimParams, *, index: int) -> None:
    """Reject segment-level fields that belong on :class:`SimulationProgram`.

    Raises:
        ValueError: If observables or ``random_seed`` are set on the segment params.
    """
    if sim_params.observables:
        msg = (
            f"segments[{index}] sim_params.observables must be empty; "
            "pass observables=... on SimulationProgram instead."
        )
        raise ValueError(msg)
    if sim_params.random_seed is not None:
        msg = (
            f"segments[{index}] sim_params.random_seed must be None; pass random_seed=... on SimulationProgram instead."
        )
        raise ValueError(msg)


def _normalize_segment(item: object, *, index: int) -> _ProgramSegment:
    """Convert a public ``(operator, params[, noise])`` pair into a private segment.

    Returns:
        The private analog or digital segment.

    Raises:
        TypeError: If ``item`` is not a supported pair or types disagree.
    """
    if not isinstance(item, tuple) or len(item) not in {2, 3}:
        msg = (
            f"segments[{index}] must be a (operator, params) or "
            f"(operator, params, noise_model) tuple, got {type(item).__name__}."
        )
        raise TypeError(msg)

    operator = item[0]
    params = item[1]
    noise_model = item[2] if len(item) == 3 else None
    if noise_model is not None and not isinstance(noise_model, NoiseModel):
        msg = f"segments[{index}] noise_model must be NoiseModel or None, got {type(noise_model).__name__}."
        raise TypeError(msg)

    if isinstance(operator, Hamiltonian):
        if not isinstance(params, AnalogSimParams):
            msg = f"segments[{index}] pairs a Hamiltonian with {type(params).__name__}; expected AnalogSimParams."
            raise TypeError(msg)
        _reject_program_owned_fields(params, index=index)
        return _AnalogSegment(operator, sim_params=params, noise_model=noise_model)

    if isinstance(operator, QuantumCircuit):
        if not isinstance(params, DigitalSimParams):
            msg = f"segments[{index}] pairs a QuantumCircuit with {type(params).__name__}; expected DigitalSimParams."
            raise TypeError(msg)
        _reject_program_owned_fields(params, index=index)
        return _DigitalSegment(operator, sim_params=params, noise_model=noise_model)

    msg = f"segments[{index}] operator must be Hamiltonian or QuantumCircuit, got {type(operator).__name__}."
    raise TypeError(msg)


@dataclass(frozen=True, init=False)
class SimulationProgram:
    """Ordered analog and digital segments that share one evolving state.

    Construct from ``(operator, params)`` pairs. Mode is selected by the operator
    type: :class:`~mqt.yaqs.Hamiltonian` with :class:`~mqt.yaqs.AnalogSimParams`, or
    :class:`~qiskit.circuit.QuantumCircuit` with :class:`~mqt.yaqs.DigitalSimParams`.
    An optional third ``noise_model`` entry overrides the run-level model for that
    segment (``None`` inherits; an empty :class:`~mqt.yaqs.NoiseModel` disables noise).

    Observables, trajectory count, and RNG seed are program-wide. Segment
    ``sim_params`` must leave ``observables`` empty and ``random_seed`` unset;
    they only carry truncation, timing, gate-mode, and related backend settings.

    Args:
        segments: Non-empty iterable of ``(operator, params)`` pairs.
        observables: Shared observables recorded by every segment.
        num_traj: Trajectory count for stochastic execution. If omitted, the
            default from a fresh :class:`~mqt.yaqs.AnalogSimParams` is used for
            noisy runs. A noisy program with shots but no observables instead
            executes one complete-program trajectory per requested shot.
        random_seed: Program-wide base seed for reproducible stochastic runs.
        get_state: Whether to retain the final state on the outer result.

    Raises:
        TypeError: If ``segments`` or an item has the wrong type.
        ValueError: If ``segments`` is empty, a segment sets program-owned fields,
            or ``num_traj`` is less than one.
    """

    segments: tuple[_ProgramSegment, ...]
    observables: tuple[Observable, ...]
    num_traj: int | None
    random_seed: int | None
    get_state: bool

    def __init__(
        self,
        segments: Iterable[SegmentInput],
        *,
        observables: Sequence[Observable] | None = None,
        num_traj: int | None = None,
        random_seed: int | None = None,
        get_state: bool = False,
    ) -> None:
        """Initialize and validate an immutable ordered program.

        Raises:
            TypeError: If ``segments``, an item, or a keyword argument has the wrong type.
            ValueError: If the program is empty, ``num_traj`` is invalid, or a segment
                sets program-owned fields.
        """
        if isinstance(segments, (str, bytes)):
            msg = "segments must be an iterable of (operator, params) pairs."
            raise TypeError(msg)
        try:
            raw_segments = tuple(segments)
        except TypeError as error:
            msg = "segments must be an iterable of (operator, params) pairs."
            raise TypeError(msg) from error

        if not raw_segments:
            msg = "SimulationProgram requires at least one segment."
            raise ValueError(msg)

        normalized_segments = tuple(_normalize_segment(item, index=index) for index, item in enumerate(raw_segments))

        if observables is None:
            obs_tuple: tuple[Observable, ...] = ()
        else:
            if isinstance(observables, (str, bytes)) or not isinstance(observables, Sequence):
                msg = "observables must be a sequence of Observable."
                raise TypeError(msg)
            obs_list = list(observables)
            for index, observable in enumerate(obs_list):
                if not isinstance(observable, Observable):
                    msg = f"observables[{index}] must be Observable, got {type(observable).__name__}."
                    raise TypeError(msg)
            obs_tuple = tuple(obs_list)

        if num_traj is not None and (isinstance(num_traj, bool) or not isinstance(num_traj, int)):
            msg = f"num_traj must be int or None, got {type(num_traj).__name__}."
            raise TypeError(msg)
        if num_traj is not None and num_traj < 1:
            msg = f"num_traj must be at least 1, got {num_traj}."
            raise ValueError(msg)
        if random_seed is not None and (isinstance(random_seed, bool) or not isinstance(random_seed, int)):
            msg = f"random_seed must be int or None, got {type(random_seed).__name__}."
            raise TypeError(msg)
        if random_seed is not None and random_seed < 0:
            msg = f"random_seed must be non-negative, got {random_seed}."
            raise ValueError(msg)
        if not isinstance(get_state, bool):
            msg = f"get_state must be bool, got {type(get_state).__name__}."
            raise TypeError(msg)

        object.__setattr__(self, "segments", normalized_segments)
        object.__setattr__(self, "observables", obs_tuple)
        object.__setattr__(self, "num_traj", num_traj)
        object.__setattr__(self, "random_seed", random_seed)
        object.__setattr__(self, "get_state", get_state)

    def __iter__(self) -> Iterator[_ProgramSegment]:
        """Iterate over private segments in program order.

        Returns:
            An iterator over the immutable private segment tuple.
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
    observables: tuple[Observable, ...]
    get_state: bool
    num_traj: int | None
    random_seed: int | None
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


def _inject_program_settings(
    sim_params: AnalogSimParams | DigitalSimParams,
    *,
    observables: Sequence[Observable],
    num_traj: int | None,
    random_seed: int | None,
) -> AnalogSimParams | DigitalSimParams:
    """Deep-copy segment params and write program-owned ensemble fields.

    Returns:
        Execution params with program observables, trajectory count, and seed applied.
    """
    execution_params = copy.deepcopy(sim_params)
    execution_params.get_state = True
    execution_params.observables = [copy.deepcopy(observable) for observable in observables]
    if num_traj is not None:
        execution_params.num_traj = num_traj
    execution_params.random_seed = random_seed
    return execution_params


def _compile_analog_segment(
    segment: _AnalogSegment,
    *,
    index: int,
    signature: _StateSignature,
    noise_model: NoiseModel | None,
    time_offset: float,
    observables: Sequence[Observable],
    num_traj: int | None,
    random_seed: int | None,
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
    sim_params = segment.sim_params
    if sim_params.order not in {1, 2}:
        msg = f"segments[{index}] AnalogSimParams.order must be 1 or 2, got {sim_params.order}."
        raise ValueError(msg)
    if sim_params.multi_time_observables:
        msg = f"segments[{index}] multi_time_observables are not supported in program execution."
        raise ValueError(msg)
    execution_params = _inject_program_settings(
        sim_params,
        observables=observables,
        num_traj=num_traj,
        random_seed=random_seed,
    )
    assert isinstance(execution_params, AnalogSimParams)
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
    segment: _DigitalSegment,
    *,
    index: int,
    signature: _StateSignature,
    noise_model: NoiseModel | None,
    time_offset: float,
    observables: Sequence[Observable],
    num_traj: int | None,
    random_seed: int | None,
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
    sim_params = segment.sim_params
    execution_params = _inject_program_settings(
        sim_params,
        observables=observables,
        num_traj=num_traj,
        random_seed=random_seed,
    )
    assert isinstance(execution_params, DigitalSimParams)
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

    Returns:
        A private immutable sequence of executable instructions.

    Raises:
        TypeError: If a program contains an unknown segment type.
        ValueError: If the state, dimensions, lengths, or analog settings are unsupported.
    """
    if initial_state.representation != "mps":
        msg = "SimulationProgram execution currently requires State.representation='mps'."
        raise ValueError(msg)
    physical_dimensions = tuple(initial_state.mps.physical_dimensions)
    signature = _StateSignature("mps", initial_state.length, physical_dimensions)
    instructions: list[_CompiledInstruction] = []
    time_offset = 0.0

    num_traj = program.num_traj if program.num_traj is not None else AnalogSimParams().num_traj
    random_seed = program.random_seed
    observables = program.observables

    for index, segment in enumerate(program.segments):
        resolved_noise_model = segment.noise_model if segment.noise_model is not None else default_noise_model
        if resolved_noise_model is not None and resolved_noise_model.scheduled_jumps:
            msg = (
                f"segments[{index}] uses scheduled_jumps, which are not supported in SimulationProgram execution; "
                "their timing semantics across segment-local clocks are not yet defined."
            )
            raise ValueError(msg)
        _validate_noise_layout(resolved_noise_model, signature.physical_dimensions, segment_index=index)

        if isinstance(segment, _AnalogSegment):
            instruction = _compile_analog_segment(
                segment,
                index=index,
                signature=signature,
                noise_model=resolved_noise_model,
                time_offset=time_offset,
                observables=observables,
                num_traj=num_traj,
                random_seed=random_seed,
            )
            instructions.append(instruction)
            time_offset += float(instruction.sim_params.elapsed_time)
            continue
        if isinstance(segment, _DigitalSegment):
            instructions.append(
                _compile_digital_segment(
                    segment,
                    index=index,
                    signature=signature,
                    noise_model=resolved_noise_model,
                    time_offset=time_offset,
                    observables=observables,
                    num_traj=num_traj,
                    random_seed=random_seed,
                )
            )
            continue
        msg = f"segments[{index}] has unsupported private segment type {type(segment).__name__}."
        raise TypeError(msg)

    return _CompiledProgram(
        tuple(instructions),
        signature,
        observables,
        program.get_state,
        num_traj,
        random_seed,
        default_noise_model,
    )


def _segment_physical_times(segment: Result, value_count: int) -> NDArray[np.float64]:
    """Build one segment's physical-time coordinates on the program timeline.

    Returns:
        The segment coordinates on the physical program timeline.

    Raises:
        ValueError: If required time metadata is missing or malformed.
    """
    offset = 0.0 if segment.time_offset is None else float(segment.time_offset)
    if segment.segment_type == "digital":
        return np.full(value_count, offset, dtype=np.float64)
    if segment.times is None:
        msg = f"Analog segment {segment.segment_index} has no time data."
        raise ValueError(msg)
    times = np.asarray(segment.times, dtype=np.float64)
    if times.ndim != 1 or len(times) != value_count:
        msg = f"Observable in analog segment {segment.segment_index} is not aligned with times."
        raise ValueError(msg)
    return np.asarray(times + offset, dtype=np.float64)


def flatten_program_results(
    segment_results: Sequence[Result],
    observables: Sequence[Observable],
) -> tuple[list[NDArray[Any]], NDArray[np.float64] | None, dict[int, int] | None]:
    """Stitch per-segment outputs into top-level expectation values, times, and counts.

    Returns:
        ``(expectation_values, times, counts)`` for the outer program result.

    Raises:
        ValueError: If a segment's observable buffers are missing or misaligned.
    """
    if not observables:
        counts: dict[int, int] = {}
        for segment in segment_results:
            if segment.counts:
                for key, value in segment.counts.items():
                    counts[key] = counts.get(key, 0) + value
        return [], None, (dict(sorted(counts.items())) if counts else None)

    expectation_parts: list[list[NDArray[Any]]] = [[] for _ in observables]
    time_parts: list[NDArray[np.float64]] = []
    times_initialized = False

    for segment in segment_results:
        if len(segment.expectation_values) != len(observables):
            msg = (
                f"Segment {segment.segment_index} recorded {len(segment.expectation_values)} "
                f"observables, expected {len(observables)}."
            )
            raise ValueError(msg)
        first_values = np.asarray(segment.expectation_values[0])
        segment_times = _segment_physical_times(segment, len(first_values))
        if not times_initialized:
            time_parts.append(segment_times)
            times_initialized = True
        else:
            time_parts.append(segment_times)
        for index, values in enumerate(segment.expectation_values):
            arr = np.asarray(values)
            if arr.ndim != 1 or len(arr) != len(first_values):
                msg = f"Segment {segment.segment_index} observable {index} has inconsistent shape."
                raise ValueError(msg)
            expectation_parts[index].append(arr)

    expectation_values = [np.concatenate(parts) if parts else np.empty(0) for parts in expectation_parts]
    times = np.concatenate(time_parts) if time_parts else None

    counts = {}
    for segment in segment_results:
        if segment.counts:
            for key, value in segment.counts.items():
                counts[key] = counts.get(key, 0) + value

    return expectation_values, times, (dict(sorted(counts.items())) if counts else None)
