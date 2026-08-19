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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.circuit import QuantumCircuit

from ...digital.digital_tjm import _compile_circuit, _CompiledCircuit
from ...digital.utils.qasm_utils import load_circuit
from .hamiltonian import Hamiltonian
from .noise_model import NoiseModel
from .simulation_parameters import AnalogSimParams, DigitalSimParams, EvolutionMode, Observable

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
    | tuple[str, DigitalSimParams]
    | tuple[str, DigitalSimParams, NoiseModel | None]
    | tuple[Path, DigitalSimParams]
    | tuple[Path, DigitalSimParams, NoiseModel | None]
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


def _normalize_program_settings(
    *,
    observables: Sequence[Observable] | None,
    num_traj: int | None,
    random_seed: int | None,
    get_state: bool,
) -> tuple[tuple[Observable, ...], int | None, int | None, bool]:
    """Validate and normalize program-level keyword arguments.

    Args:
        observables: Optional program-wide observables.
        num_traj: Optional program-wide trajectory count.
        random_seed: Optional program-wide RNG seed.
        get_state: Whether to retain the final noiseless state.

    Returns:
        Normalized ``(observables, num_traj, random_seed, get_state)``.

    Raises:
        TypeError: If a keyword argument has the wrong type.
        ValueError: If ``num_traj`` or ``random_seed`` is out of range.
    """
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

    return obs_tuple, num_traj, random_seed, get_state


def _reject_program_owned_fields(sim_params: AnalogSimParams | DigitalSimParams, *, index: int) -> None:
    """Reject segment-level fields that belong on :class:`SimulationProgram`.

    Args:
        sim_params: Segment simulation parameters to validate.
        index: Segment index used in error messages.

    Raises:
        ValueError: If observables, ``random_seed``, or ``get_state`` are set on the segment.
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
    if sim_params.get_state:
        msg = f"segments[{index}] sim_params.get_state must be False; pass get_state=... on SimulationProgram instead."
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

    if isinstance(operator, (str, Path)):
        operator = load_circuit(operator)

    if isinstance(operator, QuantumCircuit):
        if not isinstance(params, DigitalSimParams):
            msg = f"segments[{index}] pairs a circuit operator with {type(params).__name__}; expected DigitalSimParams."
            raise TypeError(msg)
        _reject_program_owned_fields(params, index=index)
        return _DigitalSegment(operator, sim_params=params, noise_model=noise_model)

    msg = (
        f"segments[{index}] operator must be Hamiltonian, QuantumCircuit, or an OpenQASM str/Path, "
        f"got {type(operator).__name__}."
    )
    raise TypeError(msg)


@dataclass(frozen=True, init=False)
class SimulationProgram:
    """Ordered analog and digital segments that share one evolving state.

    Construct from ``(operator, params)`` pairs. Mode is selected by the operator
    type: :class:`~mqt.yaqs.Hamiltonian` with :class:`~mqt.yaqs.AnalogSimParams`, or
    a :class:`~qiskit.circuit.QuantumCircuit` / OpenQASM ``str`` / :class:`~pathlib.Path`
    with :class:`~mqt.yaqs.DigitalSimParams`. An optional third ``noise_model`` entry
    overrides the run-level model for that segment (``None`` inherits; an empty
    :class:`~mqt.yaqs.NoiseModel` disables noise).

    Observables, RNG seed, and ``get_state`` are program-wide. Segment
    ``sim_params`` must leave ``observables`` empty and keep ``random_seed`` /
    ``get_state`` unset; they carry truncation, timing, ``shots``, gate-mode, and
    related backend settings. Analog ``scheduled_jumps`` times are segment-local.

    Args:
        segments: Non-empty iterable of ``(operator, params)`` pairs.
        observables: Shared observables recorded by every segment.
        num_traj: Trajectory count for stochastic execution. When omitted, a
            unanimous segment ``num_traj`` is used; conflicting segment values
            require an explicit program-level ``num_traj``. A noisy program with
            shots but no observables executes one complete-program trajectory per
            requested shot.
        random_seed: Program-wide base seed for reproducible stochastic runs.
        get_state: Whether to retain the final noiseless state on the outer result.

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
        obs_tuple, num_traj, random_seed, get_state = _normalize_program_settings(
            observables=observables,
            num_traj=num_traj,
            random_seed=random_seed,
            get_state=get_state,
        )

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
    hamiltonian: MPO | tuple[MPO, ...]
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
    num_traj: int
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


def _resolve_program_num_traj(program: SimulationProgram) -> int:
    """Resolve the ensemble size for a program from program- or segment-level values.

    Args:
        program: Program whose trajectory count should be resolved.

    Returns:
        The trajectory count applied to every compiled instruction.

    Raises:
        ValueError: If ``program.num_traj`` is omitted and segment values disagree.
    """
    if program.num_traj is not None:
        return program.num_traj
    segment_counts = {segment.sim_params.num_traj for segment in program.segments}
    if len(segment_counts) == 1:
        return next(iter(segment_counts))
    msg = (
        "segments disagree on sim_params.num_traj; pass num_traj=... on SimulationProgram to select the ensemble size."
    )
    raise ValueError(msg)


def _apply_program_settings(
    sim_params: AnalogSimParams | DigitalSimParams,
    *,
    observables: Sequence[Observable],
    num_traj: int,
    random_seed: int | None,
) -> AnalogSimParams | DigitalSimParams:
    """Copy segment params and write program-owned ensemble fields for execution.

    Args:
        sim_params: User-facing segment parameters.
        observables: Program-wide observables injected into every segment.
        num_traj: Resolved trajectory count.
        random_seed: Program-wide RNG seed.

    Returns:
        Execution params with handoff ``get_state``, observables, trajectory count,
        and seed applied.
    """
    execution_params = copy.deepcopy(sim_params)
    execution_params.get_state = True
    execution_params.observables = [copy.deepcopy(observable) for observable in observables]
    execution_params.num_traj = num_traj
    execution_params.random_seed = random_seed
    return execution_params


def _duration_to_steps(duration: float, dt: float, *, label: str) -> int:
    """Convert a duration into a positive number of analog ``dt`` intervals.

    Args:
        duration: Requested duration.
        dt: Analog time step.
        label: Name used in the error message.

    Returns:
        A positive integer step count.

    Raises:
        ValueError: If ``duration`` is not a positive integer multiple of ``dt``.
    """
    n_float = duration / dt
    n_steps = round(n_float)
    evolved = n_steps * dt
    residual = abs(duration - evolved)
    roundoff_tol = max(
        np.spacing(duration),
        abs(n_steps) * np.spacing(dt),
        8 * np.finfo(np.float64).eps * max(duration, evolved, dt),
    )
    tol = min(roundoff_tol, 0.25 * dt)
    if n_steps <= 0 or residual > tol:
        msg = f"{label} ({duration}) must be an integer multiple of dt ({dt})."
        raise ValueError(msg)
    return n_steps


def _validate_mpo_physical_dimensions(
    mpo: MPO,
    physical_dimensions: Sequence[int],
    *,
    label: str,
) -> None:
    """Reject an MPO whose physical legs do not match the state.

    Raises:
        ValueError: If a site's physical legs disagree with ``physical_dimensions``.
    """
    for site, (tensor, dimension) in enumerate(zip(mpo.tensors, physical_dimensions, strict=True)):
        if tensor.ndim != 4 or tensor.shape[0] != dimension or tensor.shape[1] != dimension:
            msg = f"{label} MPO site {site} has physical legs {tensor.shape[:2]}, expected ({dimension}, {dimension})."
            raise ValueError(msg)


def _expand_analog_operator(
    hamiltonian: Hamiltonian,
    sim_params: AnalogSimParams,
    *,
    physical_dimensions: Sequence[int],
    label: str = "",
) -> MPO | tuple[MPO, ...]:
    """Materialize a static or piecewise Hamiltonian as analog interval MPOs.

    A fully static Hamiltonian returns one ``MPO``. A piecewise Hamiltonian
    returns one MPO reference per analog interval. Consecutive intervals that
    share the same MPO object collapse to a bare ``MPO``.

    Args:
        hamiltonian: Static or piecewise Hamiltonian.
        sim_params: Analog parameters whose ``dt`` grid defines the intervals.
        physical_dimensions: Per-site physical dimensions of the MPS.
        label: Optional prefix for validation error messages.

    Returns:
        A static MPO, or one MPO per analog interval.

    Raises:
        ValueError: If a duration is not a multiple of ``dt``, durations do not
            sum to ``elapsed_time``, or a piece has incompatible physical legs.
    """
    prefix = f"{label} " if label else ""
    if not hamiltonian.is_piecewise:
        hamiltonian.ensure_mpo()
        _validate_mpo_physical_dimensions(
            hamiltonian.mpo,
            physical_dimensions,
            label=f"{prefix}Hamiltonian",
        )
        return hamiltonian.mpo

    dt = float(sim_params.dt)
    n_intervals = max(len(sim_params.times) - 1, 0)
    interval_mpos: list[MPO] = []
    for index, (piece, duration) in enumerate(hamiltonian.pieces):
        n_steps = _duration_to_steps(duration, dt, label=f"{prefix}pieces[{index}] duration")
        piece.ensure_mpo()
        mpo = piece.mpo
        _validate_mpo_physical_dimensions(
            mpo,
            physical_dimensions,
            label=f"{prefix}pieces[{index}] Hamiltonian",
        )
        interval_mpos.extend([mpo] * n_steps)
    if len(interval_mpos) != n_intervals:
        msg = (
            f"{prefix}piecewise durations must sum to elapsed_time ({sim_params.elapsed_time}); "
            f"got {len(interval_mpos) * dt}, expected {sim_params.elapsed_time}."
        )
        raise ValueError(msg)
    if interval_mpos and all(mpo is interval_mpos[0] for mpo in interval_mpos):
        return interval_mpos[0]
    return tuple(interval_mpos)


def _compile_analog_segment(
    segment: _AnalogSegment,
    *,
    index: int,
    signature: _StateSignature,
    noise_model: NoiseModel | None,
    time_offset: float,
    observables: Sequence[Observable],
    num_traj: int,
    random_seed: int | None,
) -> _CompiledAnalogInstruction:
    """Compile one analog segment after program-wide settings are resolved.

    Args:
        segment: Private analog segment.
        index: Segment index in the program.
        signature: Shared MPS state signature.
        noise_model: Resolved noise model for this segment.
        time_offset: Physical start time on the program timeline.
        observables: Program-wide observables.
        num_traj: Resolved trajectory count.
        random_seed: Program-wide RNG seed.

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
    if segment.hamiltonian.is_piecewise and sim_params.evolution_mode != EvolutionMode.TDVP:
        msg = f"segments[{index}] piecewise Hamiltonians require evolution_mode=EvolutionMode.TDVP."
        raise ValueError(msg)
    execution_params = _apply_program_settings(
        sim_params,
        observables=observables,
        num_traj=num_traj,
        random_seed=random_seed,
    )
    assert isinstance(execution_params, AnalogSimParams)
    operator = _expand_analog_operator(
        segment.hamiltonian,
        execution_params,
        physical_dimensions=signature.physical_dimensions,
        label=f"segments[{index}]",
    )
    return _CompiledAnalogInstruction(
        index=index,
        hamiltonian=operator,
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
    num_traj: int,
    random_seed: int | None,
) -> _CompiledDigitalInstruction:
    """Compile one digital segment after program-wide settings are resolved.

    Args:
        segment: Private digital segment.
        index: Segment index in the program.
        signature: Shared MPS state signature.
        noise_model: Resolved noise model for this segment.
        time_offset: Physical start time on the program timeline.
        observables: Program-wide observables.
        num_traj: Resolved trajectory count.
        random_seed: Program-wide RNG seed.

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
    execution_params = _apply_program_settings(
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

    Args:
        program: Public program specification.
        initial_state: Initial MPS-backed state.
        default_noise_model: Run-level noise inherited by segments without an override.

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

    for index, segment in enumerate(program.segments):
        if not isinstance(segment, (_AnalogSegment, _DigitalSegment)):
            msg = f"segments[{index}] has unsupported private segment type {type(segment).__name__}."
            raise TypeError(msg)

    num_traj = _resolve_program_num_traj(program)
    random_seed = program.random_seed
    observables = program.observables

    for index, segment in enumerate(program.segments):
        resolved_noise_model = segment.noise_model if segment.noise_model is not None else default_noise_model
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
        assert isinstance(segment, _DigitalSegment)
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

    return _CompiledProgram(
        instructions=tuple(instructions),
        state_signature=signature,
        observables=observables,
        get_state=program.get_state,
        num_traj=num_traj,
        random_seed=random_seed,
        default_noise_model=default_noise_model,
    )


def _build_segment_timeline(segment: Result, value_count: int) -> NDArray[np.float64]:
    """Build one segment's physical-time coordinates on the program timeline.

    Args:
        segment: Per-segment result with local times and ``time_offset``.
        value_count: Number of samples recorded for each observable.

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


def _select_final_segment_counts(segment_results: Sequence[Result]) -> dict[int, int] | None:
    """Return shot counts from the last segment that recorded any.

    Args:
        segment_results: Per-segment results in program order.

    Returns:
        The final shot histogram, or ``None`` when no segment recorded counts.
    """
    for segment in reversed(segment_results):
        if segment.counts:
            return dict(sorted(segment.counts.items()))
    return None


def stitch_program_results(
    segment_results: Sequence[Result],
    observables: Sequence[Observable],
) -> tuple[list[NDArray[Any]], NDArray[np.float64] | None, dict[int, int] | None]:
    """Stitch per-segment outputs into top-level expectation values, times, and counts.

    Outer ``counts`` come from the last segment that recorded shots. Per-segment
    histograms remain available on ``result.segment_results[i].counts``.

    Args:
        segment_results: Per-segment results in program order.
        observables: Shared program observables.

    Returns:
        ``(expectation_values, times, counts)`` for the outer program result.

    Raises:
        ValueError: If a segment's observable buffers are missing or misaligned.
    """
    if not observables:
        return [], None, _select_final_segment_counts(segment_results)

    expectation_parts: list[list[NDArray[Any]]] = [[] for _ in observables]
    time_parts: list[NDArray[np.float64]] = []

    for segment in segment_results:
        if len(segment.expectation_values) != len(observables):
            msg = (
                f"Segment {segment.segment_index} recorded {len(segment.expectation_values)} "
                f"observables, expected {len(observables)}."
            )
            raise ValueError(msg)
        first_values = np.asarray(segment.expectation_values[0])
        time_parts.append(_build_segment_timeline(segment, len(first_values)))
        for index, values in enumerate(segment.expectation_values):
            arr = np.asarray(values)
            if arr.ndim != 1 or len(arr) != len(first_values):
                msg = f"Segment {segment.segment_index} observable {index} has inconsistent shape."
                raise ValueError(msg)
            expectation_parts[index].append(arr)

    expectation_values = [np.concatenate(parts) if parts else np.empty(0) for parts in expectation_parts]
    times = np.concatenate(time_parts) if time_parts else None
    return expectation_values, times, _select_final_segment_counts(segment_results)
