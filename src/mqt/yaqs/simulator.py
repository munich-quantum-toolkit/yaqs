# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""High-level :class:`Simulator` class for YAQS.

This module implements the common simulation routine for Hamiltonian (analog) and
circuit-based simulations as the public :class:`Simulator` class. Analog evolution is
the primary mode supported by YAQS; circuit (digital) simulation reuses the same
execution machinery.

The :class:`Simulator` class owns the execution-side configuration (parallel vs.
serial execution, worker count, progress reporting, multiprocessing context, and
retry policy), while the physics inputs are passed to :meth:`Simulator.run` as a
:class:`~mqt.yaqs.core.data_structures.state.State` and either a
:class:`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` (analog) or a
:class:`~qiskit.circuit.QuantumCircuit` (digital) together with a simulation
parameter object, or as an ordered :class:`~mqt.yaqs.SimulationProgram`.
Depending on the input type
(``AnalogSimParams`` or ``DigitalSimParams``), the simulation is
dispatched to the appropriate backend:

  - For analog simulations, a ``Hamiltonian`` is validated and materialized for the
    selected state representation, then processed via the analog dispatch. Passing a
    ``list[State]`` triggers the deterministic unitary ensemble path.
  - For circuit simulations, a ``QuantumCircuit`` is used and processed via the
    digital dispatch (observables and/or measurement counts).

The module supports analog (TJM / MCWF / Lindblad / unitary ensemble) and digital
simulation, including functionality for:

  - Initializing the state (``State``) to a canonical form (B normalized).
  - Running trajectories with noise (using a provided ``NoiseModel``) and aggregating results.
  - Parallel execution of trajectories using a ``ProcessPoolExecutor`` with progress reporting via tqdm.

:meth:`Simulator.run` returns a :class:`~mqt.yaqs.core.data_structures.result.Result`
holding every simulation output (aggregated expectation values, per-trajectory data,
shared time grid, optional output state, measurement counts, and the sampled noise
model). The ``*SimParams`` object passed in is never mutated; ``Result.sim_params``
references it unchanged.
"""

from __future__ import annotations

import copy

# ruff:file-ignore[module-import-not-at-top-of-file]
# ---------------------------------------------------------------------------
# 0) IMPORTS
# Thread caps are NOT set at module level to allow single-trajectory
# simulations to use multi-threading via threadpoolctl.
# Thread limits are enforced in worker processes via limit_worker_threads()
# and in backend calls via call_serial_capped() with threadpoolctl.
# ---------------------------------------------------------------------------
from collections.abc import Sequence
from concurrent.futures import CancelledError
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from .core.data_structures.mpo import MPO
    from .core.parallel_utils import MPContext

# Optional: extra control over threadpools inside worker processes.
# We keep references as optionals, set by a guarded import.
threadpool_limits: Callable[..., Any] | None
threadpool_info: Callable[[], Any] | None
try:
    from threadpoolctl import threadpool_info as _threadpool_info
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover - optional dependency
    threadpool_limits = None
    threadpool_info = None
else:
    threadpool_limits = _threadpool_limits
    threadpool_info = _threadpool_info

if TYPE_CHECKING:
    from numpy.typing import NDArray

from itertools import starmap
from pathlib import Path

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from tqdm import tqdm

from .analog.analog_tjm import analog_tjm_1, analog_tjm_2
from .analog.ensemble import ensemble_member_worker
from .analog.lindblad import lindblad_evolve, preprocess_lindblad
from .analog.mcwf import mcwf, preprocess_mcwf
from .core.data_structures.hamiltonian import Hamiltonian
from .core.data_structures.mps import MPS
from .core.data_structures.noise_model import NoiseModel, validate_noise_model_for_run
from .core.data_structures.result import (
    Result,
    aggregate_counts,
    aggregate_diagnostics,
    aggregate_trajectories,
    allocate_diagnostic_buffers,
    allocate_observable_buffers,
)
from .core.data_structures.simulation_parameters import (
    AnalogSimParams,
    DigitalSimParams,
    EvolutionMode,
    Observable,
    _prepare_observable_ordering,
)
from .core.data_structures.simulation_program import (
    SegmentInput,
    SimulationProgram,
    _compile_program,
    _CompiledAnalogInstruction,
    _CompiledDigitalInstruction,
    _CompiledProgram,
    _expand_analog_operator,
    stitch_program_results,
)
from .core.data_structures.state import State
from .core.parallel_utils import (
    WORKER_CTX,
    ExecutionConfig,
    MPContext,
    available_cpus,
    call_serial_capped,
    get_parallel_context,
    merge_execution_config,
    resolve_worker_ctx,
    run_backend_parallel,
)
from .core.random_utils import make_disorder_rng, make_trajectory_rng
from .digital.digital_tjm import digital_tjm
from .digital.utils.qasm_utils import load_circuit

__all__ = ["Simulator", "available_cpus"]


# ---------------------------------------------------------------------------
# 4) TYPE VARS FOR GENERIC PARALLEL RUNNERS
# ---------------------------------------------------------------------------
TArg = TypeVar("TArg")
TRes = TypeVar("TRes")

# Backward-compatible alias for tests and docs that import the private name.
_get_parallel_context = get_parallel_context


# ---------------------------------------------------------------------------
# 5) WORKER WRAPPERS
# These functions are pickled and sent to workers. They retrieve large objects
# from the global _WORKER_CTX instead of receiving them as arguments. Analog
# workers come first (primary simulation mode), followed by the digital
# digital workers, with the unitary ensemble worker last.
# ---------------------------------------------------------------------------
def _analog_worker(traj_idx: int) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, MPS | None]:
    """Execute a single analog simulation trajectory (TJM or Lindblad).

    Retrieves the appropriate backend function and simulation arguments from
    `WORKER_CTX` and executes the trajectory.

    Args:
        traj_idx: The integer index of the trajectory to execute.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64] | None, MPS | None]:
            Observable data, optional diagnostics, and optional final MPS.
    """
    # backend is chosen by Simulator._run_analog and stored in context
    backend = WORKER_CTX["backend"]
    return backend((
        traj_idx,
        WORKER_CTX["initial_state"],
        WORKER_CTX["noise_model"],
        WORKER_CTX["sim_params"],
        WORKER_CTX["operator"],
    ))


def _mcwf_worker(traj_idx: int) -> tuple[NDArray[np.float64], None, np.ndarray | None]:
    """Execute a single Monte Carlo Wavefunction (MCWF) trajectory.

    Retrieves the preprocessed MCWF context from `WORKER_CTX` and executes
    the trajectory.

    Args:
        traj_idx: The integer index of the trajectory to execute.

    Returns:
        NDArray[np.float64]: The result of the MCWF trajectory.
    """
    return mcwf((traj_idx, WORKER_CTX["ctx"]))


def _lindblad_ctx_worker(_traj_idx: int) -> tuple[NDArray[np.float64], None, NDArray[np.complex128] | None]:
    """Execute Lindblad evolution from a preprocessed context in `WORKER_CTX`.

    Args:
        _traj_idx: Trajectory index (unused; Lindblad evolution is deterministic in rho).

    Returns:
        tuple[NDArray[np.float64], None, NDArray[np.complex128] | None]:
            Observable expectation values over time and optional final density matrix.
    """
    return lindblad_evolve(WORKER_CTX["ctx"])


def _digital_worker(
    traj_idx: int,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, dict[int, int] | None, MPS | None]:
    """Execute a single digital simulation trajectory.

    Returns:
        Observable data, diagnostics, shot counts, and optional final MPS (any may be ``None``).
    """
    return digital_tjm((
        traj_idx,
        WORKER_CTX["initial_state"],
        WORKER_CTX["noise_model"],
        WORKER_CTX["sim_params"],
        WORKER_CTX["operator"],
    ))


_ProgramSegmentTrajectory = tuple[
    np.ndarray | None,
    np.ndarray | None,
    dict[int, int] | None,
    MPS | None,
]
_ProgramTrajectory = tuple[tuple[_ProgramSegmentTrajectory, ...], MPS | None]


def _noise_model_is_stochastic(noise_model: NoiseModel | None) -> bool:
    """Return whether a sampled model contains a nonzero stochastic process."""
    if noise_model is None:
        return False
    return any(float(process["strength"]) > 0 for process in noise_model.processes)


def _sample_program_noise_models(compiled: _CompiledProgram) -> _CompiledProgram:
    """Sample each distinct program noise model once for this run.

    Reusing the same model object as a program default or segment override also
    reuses the same sampled realization. This preserves standalone run-scoped
    distribution sampling without introducing trajectory-scoped noise types.

    Returns:
        A compiled program whose noise models contain concrete strengths.
    """
    rng = make_disorder_rng(base_seed=compiled.random_seed)
    sampled_by_id: dict[int, NoiseModel] = {}

    def sample(noise_model: NoiseModel | None) -> NoiseModel | None:
        if noise_model is None:
            return None
        key = id(noise_model)
        if key not in sampled_by_id:
            sampled_by_id[key] = noise_model.sample(rng=rng)
        return sampled_by_id[key]

    sampled_default = sample(compiled.default_noise_model)
    instructions = tuple(
        replace(instruction, noise_model=sample(instruction.noise_model)) for instruction in compiled.instructions
    )
    return replace(compiled, instructions=instructions, default_noise_model=sampled_default)


def _same_order2_operator(
    left: _CompiledAnalogInstruction,
    right: _CompiledAnalogInstruction,
) -> bool:
    """Return whether two analog instructions share Hamiltonian and noise content."""
    if left.noise_model is not right.noise_model:
        return False
    left_mpo = left.hamiltonian
    right_mpo = right.hamiltonian
    if isinstance(left_mpo, tuple) or isinstance(right_mpo, tuple):
        return False
    if left_mpo is right_mpo:
        return True
    if len(left_mpo.tensors) != len(right_mpo.tensors):
        return False
    return all(starmap(np.array_equal, zip(left_mpo.tensors, right_mpo.tensors, strict=True)))


def _order2_chain_continues(
    instructions: tuple[_CompiledAnalogInstruction | _CompiledDigitalInstruction, ...],
    current: _CompiledAnalogInstruction,
    index: int,
) -> bool:
    """Return whether the next instruction can continue this order-2 trajectory.

    Continuation requires an adjacent order-2 analog segment with the same
    Hamiltonian, resolved noise model, ``dt``, ``evolution_mode``, and
    ``sample_timesteps``, and a non-trivial current time grid so a mid-Trotter
    ``phi`` handoff is meaningful.
    """
    nxt = instructions[index + 1] if index + 1 < len(instructions) else None
    next_params = nxt.execution_params if isinstance(nxt, _CompiledAnalogInstruction) else None
    current_params = current.execution_params
    return (
        isinstance(nxt, _CompiledAnalogInstruction)
        and next_params is not None
        and next_params.order == 2
        and len(current_params.times) > 1
        and len(next_params.times) > 1
        and current_params.dt == next_params.dt
        and current_params.evolution_mode == next_params.evolution_mode
        and current_params.sample_timesteps == next_params.sample_timesteps
        and _same_order2_operator(current, nxt)
    )


def _order2_hand_off(
    instructions: tuple[_CompiledAnalogInstruction | _CompiledDigitalInstruction, ...],
    current: _CompiledAnalogInstruction,
) -> bool:
    """Return whether this analog segment should return mid-Trotter ``phi``.

    ``get_state`` on this or the next segment needs a physical checkpoint, so
    the handoff is suppressed and the next analog starts a new Trotter chain.
    """
    if not _order2_chain_continues(instructions, current, current.index):
        return False
    nxt = instructions[current.index + 1]
    assert isinstance(nxt, _CompiledAnalogInstruction)
    return not current.sim_params.get_state and not nxt.sim_params.get_state


def _execute_analog_instruction(
    traj_idx: int,
    current_state: MPS,
    instruction: _CompiledAnalogInstruction,
    instructions: tuple[_CompiledAnalogInstruction | _CompiledDigitalInstruction, ...],
    rng: np.random.Generator,
    *,
    sample_timestep_offset: int,
    continue_order2_trajectory: bool,
) -> tuple[
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    MPS | None,
    int,
    bool,
]:
    """Execute one analog program segment and update order-2 continuation state.

    Returns:
        Observable data, diagnostics, next MPS, and the updated
        ``(sample_timestep_offset, continue_order2_trajectory)`` pair.
    """
    backend = analog_tjm_1 if instruction.execution_params.order == 1 else analog_tjm_2
    if backend is analog_tjm_2:
        hand_off_trajectory = _order2_hand_off(instructions, instruction)
        traj_data, traj_diag, next_state = backend(
            (
                traj_idx,
                current_state,
                instruction.noise_model,
                instruction.execution_params,
                instruction.hamiltonian,
            ),
            copy_initial_state=False,
            rng=rng,
            sample_timestep_offset=sample_timestep_offset,
            use_trajectory_rng_for_final_sample=False,
            return_trajectory_state=hand_off_trajectory,
            continue_trajectory=continue_order2_trajectory,
        )
        if hand_off_trajectory:
            n_times = len(instruction.execution_params.times)
            # Match global sample indices of one continuous order-2 run: after a
            # segment with T grid points, the next sample timestep is T - 1 + local.
            sample_timestep_offset += max(n_times - 1, 0)
            continue_order2_trajectory = True
        else:
            sample_timestep_offset = 0
            continue_order2_trajectory = False
    else:
        traj_data, traj_diag, next_state = backend(
            (
                traj_idx,
                current_state,
                instruction.noise_model,
                instruction.execution_params,
                instruction.hamiltonian,
            ),
            copy_initial_state=False,
            rng=rng,
        )
        sample_timestep_offset = 0
        continue_order2_trajectory = False
    return traj_data, traj_diag, next_state, sample_timestep_offset, continue_order2_trajectory


def _analog_runs_can_merge(left: _CompiledAnalogInstruction, right: _CompiledAnalogInstruction) -> bool:
    """Return whether two adjacent analog segments share one ``analog_tjm`` schedule."""
    left_params = left.execution_params
    right_params = right.execution_params
    if left_params.evolution_mode != EvolutionMode.TDVP or right_params.evolution_mode != EvolutionMode.TDVP:
        return False
    if left.sim_params.get_state or right.sim_params.get_state:
        return False
    return (
        left_params.order == right_params.order
        and left_params.dt == right_params.dt
        and left_params.sample_timesteps == right_params.sample_timesteps
        and left.noise_model is right.noise_model
        and len(left_params.times) > 1
        and len(right_params.times) > 1
        and left_params.max_bond_dim == right_params.max_bond_dim
        and left_params.svd_threshold == right_params.svd_threshold
        and left_params.trunc_mode == right_params.trunc_mode
        and left_params.krylov_tol == right_params.krylov_tol
        and left_params.tdvp_sweeps == right_params.tdvp_sweeps
        and left_params.tdvp_mode == right_params.tdvp_mode
    )


def _analog_interval_mpos(instruction: _CompiledAnalogInstruction) -> tuple[MPO, ...]:
    """Return one MPO per analog interval for a compiled instruction."""
    n_intervals = max(len(instruction.execution_params.times) - 1, 0)
    operator = instruction.hamiltonian
    if isinstance(operator, tuple):
        assert len(operator) == n_intervals
        return operator
    return (operator,) * n_intervals


def _merged_analog_operator(run: Sequence[_CompiledAnalogInstruction]) -> MPO | tuple[MPO, ...]:
    """Return concatenated interval MPOs, collapsing a constant schedule to a bare MPO."""
    mpos = tuple(mpo for instruction in run for mpo in _analog_interval_mpos(instruction))
    if not mpos:
        return run[0].hamiltonian
    first = mpos[0]
    if all(mpo is first for mpo in mpos):
        return first
    return mpos


def _merged_analog_execution_params(run: Sequence[_CompiledAnalogInstruction]) -> AnalogSimParams:
    """Return analog parameters whose grid spans a merged analog run."""
    first = run[0].execution_params
    merged = copy.deepcopy(first)
    n_steps = sum(max(len(instruction.execution_params.times) - 1, 0) for instruction in run)
    elapsed = sum(float(instruction.execution_params.elapsed_time) for instruction in run)
    merged.elapsed_time = elapsed
    merged.times = merged.dt * np.arange(n_steps + 1, dtype=np.float64)
    if n_steps > 0:
        merged.times[-1] = elapsed
    merged.sample_timesteps = True
    return merged


def _merged_segment_boundary_indices(run: Sequence[_CompiledAnalogInstruction]) -> tuple[int, ...]:
    """Return time-grid indices at the end of each merged analog segment."""
    indices: list[int] = []
    cursor = 0
    for instruction in run:
        cursor += max(len(instruction.execution_params.times) - 1, 0)
        indices.append(cursor)
    return tuple(indices)


def _slice_trajectory_columns(
    data: NDArray[np.float64] | None,
    start: int,
    end: int,
) -> NDArray[np.float64] | None:
    """Return a column slice of a ``(n_obs, T)`` trajectory buffer, preserving ``None``."""
    if data is None:
        return None
    return data[:, start:end]


def _split_merged_analog_results(
    traj_data: NDArray[np.float64] | None,
    traj_diag: NDArray[np.float64] | None,
    run: Sequence[_CompiledAnalogInstruction],
    *,
    sample_timesteps: bool,
) -> list[tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]]:
    """Return per-segment slices of one merged analog trajectory."""
    slices: list[tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]] = []
    start = 0
    for instruction in run:
        n_times = len(instruction.execution_params.times)
        n_intervals = max(n_times - 1, 0)
        if sample_timesteps:
            end = start + n_times
            slices.append((
                _slice_trajectory_columns(traj_data, start, end),
                _slice_trajectory_columns(traj_diag, start, end),
            ))
        else:
            last = start + n_intervals
            slices.append((
                _slice_trajectory_columns(traj_data, last, last + 1),
                _slice_trajectory_columns(traj_diag, last, last + 1),
            ))
        start += n_intervals
    return slices


def _execute_merged_analog_run(
    traj_idx: int,
    current_state: MPS,
    run: Sequence[_CompiledAnalogInstruction],
    rng: np.random.Generator,
    *,
    return_trajectory_state: bool = False,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, MPS | None]:
    """Return the analog_tjm payload for one merged analog run.

    Merged runs always start a fresh Trotter chain. A pending mid-Trotter
    ``phi`` is consumed on the single-instruction path before any merge.
    ``return_trajectory_state`` hands ``phi`` to a later analog that cannot join.
    """
    merged_params = _merged_analog_execution_params(run)
    operator = _merged_analog_operator(run)
    user_sample = run[0].execution_params.sample_timesteps
    sample_at = None if user_sample else _merged_segment_boundary_indices(run)
    backend = analog_tjm_1 if merged_params.order == 1 else analog_tjm_2
    args = (
        traj_idx,
        current_state,
        run[0].noise_model,
        merged_params,
        operator,
    )
    if backend is analog_tjm_2:
        return backend(
            args,
            copy_initial_state=False,
            rng=rng,
            sample_timestep_offset=0,
            use_trajectory_rng_for_final_sample=False,
            return_trajectory_state=return_trajectory_state,
            continue_trajectory=False,
            sample_at=sample_at,
        )
    return backend(args, copy_initial_state=False, rng=rng, sample_at=sample_at)


def _execute_digital_instruction(
    traj_idx: int,
    current_state: MPS,
    instruction: _CompiledDigitalInstruction,
    rng: np.random.Generator,
    *,
    effective_num_traj: int,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, dict[int, int] | None, MPS | None]:
    """Execute one digital program segment.

    Returns:
        Observable data, diagnostics, optional shot counts, and the next MPS.
    """
    shots = instruction.execution_params.shots
    if shots is not None:
        WORKER_CTX["shot_distribution"] = (shots, effective_num_traj)
    try:
        return digital_tjm(
            (
                traj_idx,
                current_state,
                instruction.noise_model,
                instruction.execution_params,
                instruction.circuit,
            ),
            copy_initial_state=False,
            rng=rng,
            compiled_circuit=instruction.compiled_circuit,
        )
    finally:
        if shots is not None:
            WORKER_CTX.pop("shot_distribution", None)


def _execute_program_trajectory(
    traj_idx: int,
    initial_state: MPS,
    compiled: _CompiledProgram,
    effective_num_traj: int,
) -> _ProgramTrajectory:
    """Execute one complete compiled program with one owned state and RNG.

    Args:
        traj_idx: Trajectory index for seeded RNG streams.
        initial_state: Deep-copied into the worker-owned MPS.
        compiled: Validated program instructions.
        effective_num_traj: Ensemble size used for shot distribution.

    Returns:
        Per-segment trajectory payloads and the optional requested final MPS.

    Raises:
        RuntimeError: If a backend does not return the state required for handoff.
        TypeError: If the compiled program contains an unknown instruction.
    """
    current_state = copy.deepcopy(initial_state)
    rng = make_trajectory_rng(traj_idx, base_seed=compiled.random_seed)
    segment_payloads: list[_ProgramSegmentTrajectory] = []
    sample_timestep_offset = 0
    continue_order2_trajectory = False
    instructions = compiled.instructions
    index = 0

    while index < len(instructions):
        instruction = instructions[index]
        if isinstance(instruction, _CompiledDigitalInstruction):
            continue_order2_trajectory = False
            sample_timestep_offset = 0
            traj_data, traj_diag, shot_counts, next_state = _execute_digital_instruction(
                traj_idx,
                current_state,
                instruction,
                rng,
                effective_num_traj=effective_num_traj,
            )
            if next_state is None:
                msg = f"Program segment {instruction.index} did not return its propagated state."
                raise RuntimeError(msg)
            current_state = next_state
            checkpoint = (
                copy.deepcopy(current_state)
                if instruction.sim_params.get_state and not _noise_model_is_stochastic(instruction.noise_model)
                else None
            )
            segment_payloads.append((traj_data, traj_diag, shot_counts, checkpoint))
            index += 1
            continue
        if not isinstance(instruction, _CompiledAnalogInstruction):
            msg = f"Unknown instruction type {type(instruction).__name__} in compiled program."
            raise TypeError(msg)

        run: list[_CompiledAnalogInstruction] = [instruction]
        # A pending order-2 handoff supplies a mid-Trotter ``phi``; the merged path
        # cannot consume it (and analog_tjm_2 continuation is static-MPO only).
        while not continue_order2_trajectory and index + len(run) < len(instructions):
            nxt = instructions[index + len(run)]
            if not isinstance(nxt, _CompiledAnalogInstruction) or not _analog_runs_can_merge(run[-1], nxt):
                break
            run.append(nxt)

        if len(run) == 1:
            traj_data, traj_diag, next_state, sample_timestep_offset, continue_order2_trajectory = (
                _execute_analog_instruction(
                    traj_idx,
                    current_state,
                    instruction,
                    instructions,
                    rng,
                    sample_timestep_offset=sample_timestep_offset,
                    continue_order2_trajectory=continue_order2_trajectory,
                )
            )
            if next_state is None:
                msg = f"Program segment {instruction.index} did not return its propagated state."
                raise RuntimeError(msg)
            current_state = next_state
            checkpoint = (
                copy.deepcopy(current_state)
                if instruction.sim_params.get_state and not _noise_model_is_stochastic(instruction.noise_model)
                else None
            )
            segment_payloads.append((traj_data, traj_diag, None, checkpoint))
        else:
            hand_off_trajectory = _order2_hand_off(instructions, run[-1])
            traj_data, traj_diag, next_state = _execute_merged_analog_run(
                traj_idx,
                current_state,
                run,
                rng,
                return_trajectory_state=hand_off_trajectory,
            )
            if next_state is None:
                msg = f"Program segment {instruction.index} did not return its propagated state."
                raise RuntimeError(msg)
            current_state = next_state
            if hand_off_trajectory:
                n_steps = sum(max(len(analog.execution_params.times) - 1, 0) for analog in run)
                sample_timestep_offset = n_steps
                continue_order2_trajectory = True
            else:
                sample_timestep_offset = 0
                continue_order2_trajectory = False
            for analog_instruction, (seg_data, seg_diag) in zip(
                run,
                _split_merged_analog_results(
                    traj_data,
                    traj_diag,
                    run,
                    sample_timesteps=instruction.execution_params.sample_timesteps,
                ),
                strict=True,
            ):
                checkpoint = (
                    copy.deepcopy(current_state)
                    if analog_instruction.sim_params.get_state
                    and analog_instruction is run[-1]
                    and not _noise_model_is_stochastic(analog_instruction.noise_model)
                    else None
                )
                segment_payloads.append((seg_data, seg_diag, None, checkpoint))

        index += len(run)

    final_state = current_state if compiled.get_state else None
    return tuple(segment_payloads), final_state


def _validate_compiled_program(compiled: _CompiledProgram) -> tuple[bool, bool, bool]:
    """Validate program outputs and per-segment noise models.

    Args:
        compiled: Compiled program after pair normalization.

    Returns:
        ``(has_observables, has_shots, stochastic)`` flags used by the executor.

    Raises:
        ValueError: If no output is specified or a noisy program requests a state.
    """
    has_observables = bool(compiled.observables)
    has_shots = any(
        isinstance(instruction, _CompiledDigitalInstruction) and instruction.execution_params.shots is not None
        for instruction in compiled.instructions
    )
    if not has_observables and not has_shots and not compiled.get_state:
        msg = "No output specified: set observables=, shots on a digital segment, and/or get_state=."
        raise ValueError(msg)

    for instruction in compiled.instructions:
        if instruction.noise_model is None:
            continue
        is_digital = isinstance(instruction, _CompiledDigitalInstruction)
        validate_noise_model_for_run(
            instruction.noise_model,
            length=compiled.state_signature.length,
            physical_dimensions=list(compiled.state_signature.physical_dimensions),
            representation="mps",
            is_digital=is_digital,
            sim_params=None if is_digital else instruction.execution_params,
        )
    stochastic = any(_noise_model_is_stochastic(instruction.noise_model) for instruction in compiled.instructions)
    if stochastic and compiled.get_state:
        msg = "Cannot return state from a noisy SimulationProgram due to stochastic trajectories."
        raise ValueError(msg)
    for instruction in compiled.instructions:
        if instruction.sim_params.get_state and _noise_model_is_stochastic(instruction.noise_model):
            msg = "Cannot return state from a noisy SimulationProgram due to stochastic trajectories."
            raise ValueError(msg)
    return has_observables, has_shots, stochastic


def _program_worker(traj_idx: int, payload: dict[str, Any] | None = None) -> _ProgramTrajectory:
    """Execute one complete program trajectory from worker context.

    Returns:
        Per-segment trajectory payloads and the optional requested final MPS.
    """
    context = resolve_worker_ctx(payload)
    return _execute_program_trajectory(
        traj_idx,
        context["initial_state"],
        context["compiled_program"],
        context["effective_num_traj"],
    )


def _ensemble_worker(
    job_idx: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128] | None]:
    """Execute one deterministic unitary ensemble member trajectory.

    Uses :data:`WORKER_CTX` keys ``initial_states``, ``sim_params``, and ``operator``.

    Args:
        job_idx: Index of this member; selects ``WORKER_CTX["initial_states"][job_idx]``.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128] | None]:
            Observable trajectories, diagnostics, and optional multi-time block for one member.
    """
    return ensemble_member_worker((
        job_idx,
        WORKER_CTX["initial_states"][job_idx],
        WORKER_CTX["sim_params"],
        WORKER_CTX["operator"],
    ))


def _materialized_mps(state: State) -> MPS | None:
    """Return the encoded MPS if present, else ``None``."""
    try:
        return state.mps
    except RuntimeError:
        return None


def _validate_state_hamiltonian_pairing(state: State, hamiltonian: Hamiltonian) -> None:
    """Check ``State`` and ``Hamiltonian`` can be evolved together.

    Raises:
        ValueError: If lengths are incompatible.
    """
    if state.length != hamiltonian.length:
        msg = f"State.length={state.length} does not match Hamiltonian.length={hamiltonian.length}."
        raise ValueError(msg)


def _prepare_hamiltonian_for_run(
    hamiltonian: Hamiltonian,
    state_rep: str,
) -> tuple[MPO | None, Any]:
    """Ensure ``hamiltonian`` is converted for the backend matching ``state_rep``.

    Returns:
        ``(mpo, h_sparse)`` with one entry set for the active backend.
    """
    if state_rep in {"vector", "density_matrix"}:
        hamiltonian.ensure_sparse()
        return None, hamiltonian.sparse_matrix
    hamiltonian.ensure_mpo()
    return hamiltonian.mpo, None


def _prepare_result_observables(
    result: Result,
    sim_params: AnalogSimParams | DigitalSimParams,
    *,
    num_traj: int,
    num_mid_measurements: int | None = None,
    buffer_params: AnalogSimParams | DigitalSimParams | None = None,
) -> None:
    """Deep-copy user-ordered observables and allocate compatible output buffers."""
    result.observables = [copy.deepcopy(obs) for obs in sim_params.observables]
    trajectories, expectation_values, times = allocate_observable_buffers(
        sim_params if buffer_params is None else buffer_params,
        len(result.observables),
        num_traj=num_traj,
        num_mid_measurements=num_mid_measurements,
    )
    result.trajectories = trajectories
    result.expectation_values = expectation_values
    result.times = times


def _allocate_program_segment_results(
    compiled: _CompiledProgram,
    effective_num_traj: int,
) -> tuple[list[Result], list[NDArray[np.float64] | None]]:
    """Allocate result and diagnostic buffers for every compiled instruction.

    Returns:
        Ordered segment results and their matching optional diagnostic buffers.
    """
    segment_results: list[Result] = []
    diagnostic_buffers: list[NDArray[np.float64] | None] = []
    for instruction in compiled.instructions:
        execution_params = instruction.execution_params
        segment_result = Result(
            sim_params=execution_params,
            noise_model=instruction.noise_model,
            segment_index=instruction.index,
            segment_type="analog" if isinstance(instruction, _CompiledAnalogInstruction) else "digital",
            time_offset=instruction.time_offset,
        )
        if isinstance(instruction, _CompiledAnalogInstruction):
            _prepare_result_observables(
                segment_result,
                execution_params,
                num_traj=effective_num_traj,
                buffer_params=execution_params,
            )
            # Keep the sampling-aware time grid from ``_prepare_result_observables``
            # (full ``sim_params.times`` when ``sample_timesteps``, else final time only).
            diag_per_traj, _ = allocate_diagnostic_buffers(
                execution_params,
                num_traj=effective_num_traj,
            )
        else:
            digital_params = cast("DigitalSimParams", execution_params)
            wants_observables = bool(digital_params.observables)
            wants_shots = digital_params.shots is not None
            shots_only = wants_shots and not wants_observables
            if wants_observables:
                _prepare_result_observables(
                    segment_result,
                    digital_params,
                    num_traj=effective_num_traj,
                    num_mid_measurements=digital_params.num_mid_measurements,
                )
            if wants_shots:
                segment_result.measurements = [None] * effective_num_traj
            if shots_only:
                diag_per_traj = None
            else:
                diag_per_traj, _ = allocate_diagnostic_buffers(
                    digital_params,
                    num_traj=effective_num_traj,
                )
        segment_results.append(segment_result)
        diagnostic_buffers.append(diag_per_traj)
    return segment_results, diagnostic_buffers


def _worker_sim_params(
    sim_params: AnalogSimParams | DigitalSimParams,
) -> AnalogSimParams | DigitalSimParams:
    """Build worker-visible params that expose sorted observables for measurement.

    Returns:
        A deep copy of ``sim_params`` whose observable lists are ordered for worker evaluation.
    """
    worker_params = copy.deepcopy(sim_params)
    # Workers evaluate in sorted order for efficiency; Result retains user order.
    sorted_obs, _ = _prepare_observable_ordering(sim_params.observables)
    worker_params.observables = [copy.deepcopy(obs) for obs in sorted_obs]
    return worker_params


def _store_observable_trajectory(
    result: Result,
    sim_params: AnalogSimParams | DigitalSimParams,
    *,
    traj_index: int,
    sorted_traj_data: NDArray[np.float64] | NDArray[np.complex128],
) -> None:
    """Store one trajectory's observable data into result buffers in user order."""
    _, observable_sorted_indices = _prepare_observable_ordering(sim_params.observables)
    for user_i, sorted_i in enumerate(observable_sorted_indices):
        result.trajectories[user_i][traj_index] = sorted_traj_data[sorted_i]


def _store_final_mps(result: Result, final_mps: MPS | None) -> None:
    if final_mps is not None:
        result.output_state = State.from_mps(final_mps)


def _store_mcwf_final_state(
    result: Result,
    psi: np.ndarray | None,
    *,
    length: int | None = None,
    physical_dimensions: list[int] | int | None = None,
) -> None:
    """Store the final MCWF state vector on ``result.output_state``.

    If ``psi`` is not ``None``, this function stores a vector
    :class:`~mqt.yaqs.core.data_structures.state.State` on ``result.output_state``
    while preserving the original lattice length and local dimensions.

    Args:
        result: Output container for the simulation run.
        psi: Final state vector, or ``None`` when ``get_state`` is ``False``.
        length: Number of lattice sites from the initial state. Passed through so
            non-qubit vector states do not need to infer a qubit chain length.
        physical_dimensions: Per-site physical dimensions from the initial state.
    """
    if psi is not None:
        result.output_state = State(length=length, vector=psi, physical_dimensions=physical_dimensions)


def _store_lindblad_final_state(
    result: Result,
    rho: np.ndarray | None,
    *,
    length: int,
    physical_dimensions: list[int] | int | None,
) -> None:
    """Store the final Lindblad density matrix on ``result.output_state``.

    Args:
        result: Output container for the simulation run.
        rho: Final density matrix, or ``None`` when ``get_state`` is ``False``.
        length: Number of lattice sites from the initial state.
        physical_dimensions: Per-site physical dimensions from the initial state.
    """
    if rho is not None:
        result.output_state = State(
            density_matrix=rho,
            length=length,
            physical_dimensions=physical_dimensions,
        )


# Backward-compatible alias for in-module serial backend calls.
_call_backend = call_serial_capped


def _plan_digital_shots(
    sim_params: DigitalSimParams,
    *,
    noisy: bool,
) -> tuple[int, int | None, tuple[int, int] | None]:
    """Plan trajectory count and per-call shot allocation for a digital run.

    Args:
        sim_params: Digital simulation parameters (observables, shots, num_traj).
        noisy: Whether a non-trivial noise model is active.

    Returns:
        ``(effective_num_traj, per_call_shots, shot_distribution)`` where
        ``per_call_shots`` is a fixed per-trajectory shot count (or ``None``),
        and ``shot_distribution`` is ``(total_shots, n_traj)`` for combined
        noisy runs that divide the budget across trajectories.
    """
    wants_obs = bool(sim_params.observables)
    wants_shots = sim_params.shots is not None
    shots_only = wants_shots and not wants_obs

    per_call_shots: int | None = None
    # (total_shots, n_traj) for combined noisy runs; per-traj allocation may be 0.
    shot_distribution: tuple[int, int] | None = None

    if shots_only:
        assert sim_params.shots is not None
        if noisy:
            # One stochastic state per shot (ignore num_traj for the ensemble).
            effective_num_traj = sim_params.shots
            per_call_shots = 1
        else:
            # One noiseless trajectory; sample the full shot budget from the final state.
            effective_num_traj = 1
            per_call_shots = sim_params.shots
    elif wants_obs:
        # Observables (and diagnostics) always use num_traj when noisy, else one traj.
        effective_num_traj = sim_params.num_traj if noisy else 1
        if wants_shots:
            assert sim_params.shots is not None
            if noisy:
                # Distribute the total shot budget across the observable ensemble.
                shot_distribution = (sim_params.shots, effective_num_traj)
            else:
                per_call_shots = sim_params.shots
    else:
        # get_state only
        effective_num_traj = 1

    return effective_num_traj, per_call_shots, shot_distribution


# ---------------------------------------------------------------------------
# 8) SIMULATOR — public entry point
# Owns the execution-side configuration (parallel/serial, workers, progress,
# multiprocessing context, retry policy) and dispatches to circuit or analog
# engines based on the sim_params type.
# ---------------------------------------------------------------------------
class Simulator:
    """Public entry point for running YAQS simulations.

    A :class:`Simulator` owns the execution-side configuration: how trajectories
    are parallelized, how many workers to use, whether to display a progress bar,
    which multiprocessing context to use, and the retry policy for transient
    worker errors. The physics inputs (initial state, operator, simulation
    parameters, optional noise model) are passed per call to :meth:`run`.

    Multiple :meth:`run` calls share the same configuration. Each call constructs
    its own short-lived process pool when ``parallel=True``; the pool is not
    persisted across runs in the current implementation.

    Attributes:
        parallel: Whether to execute trajectories in parallel via a process pool.
        max_workers: Maximum number of worker processes when ``parallel=True``.
            Defaults to ``max(1, available_cpus() - 1)``.
        show_progress: Whether to display a tqdm progress bar.
        mp_context: Multiprocessing context: ``"auto"`` (default), ``"fork"``,
            or ``"spawn"``. ``"auto"`` selects ``"fork"`` on Linux and ``"spawn"`` elsewhere.
        max_retries: Maximum retry attempts for transient worker errors.
        retry_exceptions: Exception types that trigger a retry.
    """

    def __init__(
        self,
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        show_progress: bool = True,
        mp_context: MPContext = "auto",
        max_retries: int = 10,
        retry_exceptions: tuple[type[BaseException], ...] = (CancelledError, TimeoutError, OSError),
    ) -> None:
        """Initialize the simulator with execution-side configuration.

        Args:
            parallel: If ``True`` (default), use a process pool for multi-trajectory runs.
            max_workers: Maximum worker processes when running in parallel. ``None`` (default)
                resolves to ``max(1, available_cpus() - 1)``.
            show_progress: Show a tqdm progress bar during trajectory execution.
            mp_context: Multiprocessing start method (``"auto"``, ``"fork"``, or ``"spawn"``).
            max_retries: Maximum retries for transient worker errors.
            retry_exceptions: Exception types that trigger a retry.
        """
        self._execution = ExecutionConfig(
            parallel=parallel,
            max_workers=max_workers,
            show_progress=show_progress,
            mp_context=mp_context,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
        )

    @property
    def parallel(self) -> bool:
        """Whether parallel execution is enabled."""
        return self._execution.parallel

    @parallel.setter
    def parallel(self, value: bool) -> None:
        self._execution = merge_execution_config(self._execution, parallel=bool(value))

    @property
    def max_workers(self) -> int:
        """Effective worker count for parallel execution."""
        return self._execution.resolved_max_workers()

    @max_workers.setter
    def max_workers(self, value: int | None) -> None:
        self._execution = merge_execution_config(
            self._execution,
            max_workers=None if value is None else int(value),
        )

    @property
    def show_progress(self) -> bool:
        """Whether progress bars are shown during execution."""
        return self._execution.show_progress

    @show_progress.setter
    def show_progress(self, value: bool) -> None:
        self._execution = merge_execution_config(self._execution, show_progress=bool(value))

    @property
    def mp_context(self) -> MPContext:
        """Multiprocessing start-method context for worker processes."""
        return self._execution.mp_context

    @mp_context.setter
    def mp_context(self, value: MPContext) -> None:
        self._execution = merge_execution_config(self._execution, mp_context=value)

    @property
    def max_retries(self) -> int:
        """Maximum retries per job in parallel execution."""
        return self._execution.max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        self._execution = merge_execution_config(self._execution, max_retries=int(value))

    @property
    def retry_exceptions(self) -> tuple[type[BaseException], ...]:
        """Exception types that trigger a parallel job retry."""
        return self._execution.retry_exceptions

    @retry_exceptions.setter
    def retry_exceptions(self, value: tuple[type[BaseException], ...]) -> None:
        self._execution = replace(self._execution, retry_exceptions=value)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def run(
        self,
        initial_state: State | list[State],
        operator: Hamiltonian | QuantumCircuit | SimulationProgram | Sequence[SegmentInput] | str | Path,
        sim_params: AnalogSimParams | DigitalSimParams | None = None,
        noise_model: NoiseModel | None = None,
        *,
        observables: Sequence[Observable] | None = None,
        num_traj: int | None = None,
        random_seed: int | None = None,
        get_state: bool = False,
    ) -> Result:
        """Execute the common simulation routine for Hamiltonian (analog) and circuit simulations.

        Dispatches the simulation to the appropriate backend based on the type of simulation
        parameters provided. For analog simulations, the
        :class:`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` and ``State`` are
        materialized for the active representation as needed (``list[State]`` triggers the
        deterministic unitary ensemble path). For circuit-based simulations, the initial
        :class:`~mqt.yaqs.core.data_structures.state.State` must use ``representation="mps"``.

        Args:
            initial_state: The initial state as a :class:`~mqt.yaqs.core.data_structures.state.State`,
                or a list of states for deterministic analog unitary ensemble evolution
                (``AnalogSimParams`` only).
            operator: :class:`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` for analog
                simulations; a :class:`~qiskit.circuit.QuantumCircuit`, raw QASM ``str``, or
                ``Path`` for digital simulation; a :class:`~mqt.yaqs.SimulationProgram`; or a
                sequence of ``(operator, params)`` pairs that is wrapped as a program.
            sim_params: Simulation parameters specifying a standalone simulation's mode
                and settings. Must be omitted for a ``SimulationProgram`` or pair list.
            noise_model: The run-level noise model. For a standalone simulation
                or program, distribution-valued strengths are sampled once at
                the beginning of the run. Program segments inherit this model
                unless they provide an explicit override.
            observables: Program-wide observables when ``operator`` is a pair list.
            num_traj: Program-wide trajectory count when ``operator`` is a pair list.
            random_seed: Program-wide seed when ``operator`` is a pair list.
            get_state: Retain the final state when ``operator`` is a pair list.

        Returns:
            A :class:`~mqt.yaqs.core.data_structures.result.Result` holding all
            simulation outputs. The supplied ``sim_params`` is not mutated;
            ``Result.sim_params`` references the original configuration object.

        Raises:
            ValueError: If no output is specified (neither observables, shots, nor ``get_state``).
            TypeError: If ``sim_params`` is missing for a standalone run, supplied for a
                program, or the initial state is incompatible with the selected mode.
        """
        program_kwargs_set = observables is not None or num_traj is not None or random_seed is not None or get_state
        if (
            isinstance(operator, Sequence)
            and not isinstance(operator, (str, bytes))
            and (not operator or (isinstance(operator[0], tuple) and len(operator[0]) in {2, 3}))
        ):
            if sim_params is not None:
                msg = "sim_params must be None when operator is a segment pair list."
                raise TypeError(msg)
            operator = SimulationProgram(
                cast("Sequence[SegmentInput]", operator),
                observables=observables,
                num_traj=num_traj,
                random_seed=random_seed,
                get_state=get_state,
            )
        elif program_kwargs_set and not isinstance(operator, SimulationProgram):
            msg = (
                "observables=, num_traj=, random_seed=, and get_state= are only valid "
                "for a SimulationProgram or a segment pair list."
            )
            raise TypeError(msg)

        if isinstance(operator, SimulationProgram):
            if sim_params is not None:
                msg = "sim_params must be None when operator is a SimulationProgram."
                raise TypeError(msg)
            if not isinstance(initial_state, State):
                msg = "SimulationProgram execution requires a single State initial_state."
                raise TypeError(msg)
            return self._run_program(initial_state, operator, noise_model)

        if sim_params is None:
            msg = "Standalone simulation requires AnalogSimParams or DigitalSimParams."
            raise TypeError(msg)
        if not isinstance(sim_params, (AnalogSimParams, DigitalSimParams)):
            msg = f"sim_params must be AnalogSimParams or DigitalSimParams, got {type(sim_params).__name__}."
            raise TypeError(msg)

        if not isinstance(sim_params, AnalogSimParams) and isinstance(operator, (str, Path)):
            operator = load_circuit(operator)

        if isinstance(initial_state, list) and any(not isinstance(state, State) for state in initial_state):
            msg = "initial_state list must contain only State objects."
            raise TypeError(msg)

        if noise_model is not None:
            sample_seed = getattr(sim_params, "random_seed", None)
            noise_model = noise_model.sample(rng=make_disorder_rng(base_seed=sample_seed))

        result = Result(sim_params=sim_params, noise_model=noise_model)

        if (
            isinstance(sim_params, AnalogSimParams)
            and not sim_params.get_state
            and not sim_params.observables
            and not sim_params.multi_time_observables
        ):
            msg = "No output specified: either observables or get_state must be set."
            raise ValueError(msg)
        if (
            isinstance(sim_params, DigitalSimParams)
            and not sim_params.get_state
            and not sim_params.observables
            and sim_params.shots is None
        ):
            msg = "No output specified: set observables, shots, and/or get_state."
            raise ValueError(msg)

        if isinstance(sim_params, AnalogSimParams):
            if not isinstance(operator, Hamiltonian):
                msg = "Analog simulation requires a Hamiltonian operator."
                raise TypeError(msg)
            if not isinstance(initial_state, (State, list)):
                msg = "Analog simulation requires initial_state to be a list or State."
                raise TypeError(msg)
            self._run_analog(initial_state, operator, sim_params, noise_model, result)
        elif isinstance(sim_params, DigitalSimParams):
            if isinstance(initial_state, list):
                msg = "Circuit simulation requires a single State initial_state."
                raise TypeError(msg)
            if not isinstance(operator, QuantumCircuit):
                msg = "Circuit simulation requires a QuantumCircuit operator."
                raise TypeError(msg)
            if not isinstance(initial_state, State):
                msg = "Circuit simulation requires a State initial_state."
                raise TypeError(msg)
            self._run_circuit(initial_state, operator, sim_params, noise_model, result)

        return result

    # -----------------------------------------------------------------------
    # SimulationProgram execution
    # -----------------------------------------------------------------------
    def _run_program(
        self,
        initial_state: State,
        program: SimulationProgram,
        noise_model: NoiseModel | None,
    ) -> Result:
        """Compile and execute complete MPS trajectories through a program.

        Args:
            initial_state: User-owned initial MPS state, preserved by the executor.
            program: Ordered public program specification.
            noise_model: Run-level model inherited by segments without an override.

        Returns:
            An outer result containing one result per segment in program order.

        Raises:
            RuntimeError: If stochastic compilation does not resolve a trajectory count.
        """
        compiled = _compile_program(program, initial_state, noise_model)
        compiled = _sample_program_noise_models(compiled)
        has_observables, _has_shots, stochastic = _validate_compiled_program(compiled)
        if stochastic:
            if compiled.num_traj is None:  # pragma: no cover
                msg = "Noisy SimulationProgram has no resolved trajectory count."
                raise RuntimeError(msg)
            shot_budgets = [
                instruction.execution_params.shots
                for instruction in compiled.instructions
                if isinstance(instruction, _CompiledDigitalInstruction)
                and instruction.execution_params.shots is not None
            ]
            effective_num_traj = max(shot_budgets) if shot_budgets and not has_observables else compiled.num_traj
        else:
            effective_num_traj = 1

        segment_results, diagnostic_buffers = _allocate_program_segment_results(compiled, effective_num_traj)

        payload: dict[str, Any] = {
            "initial_state": initial_state.mps,
            "compiled_program": compiled,
            "effective_num_traj": effective_num_traj,
        }
        final_mps: MPS | None = None

        def consume(traj_index: int, trajectory: _ProgramTrajectory) -> None:
            nonlocal final_mps
            segment_payloads, trajectory_final = trajectory
            for instruction, segment_result, diag_per_traj, segment_payload in zip(
                compiled.instructions,
                segment_results,
                diagnostic_buffers,
                segment_payloads,
                strict=True,
            ):
                traj_data, traj_diag, shot_counts, checkpoint = segment_payload
                if traj_data is not None and segment_result.observables:
                    _store_observable_trajectory(
                        segment_result,
                        instruction.execution_params,
                        traj_index=traj_index,
                        sorted_traj_data=traj_data,
                    )
                if traj_diag is not None and diag_per_traj is not None:
                    diag_per_traj[:, traj_index, :] = traj_diag
                if shot_counts is not None:
                    segment_result.measurements[traj_index] = shot_counts
                if checkpoint is not None:
                    segment_result.output_state = State.from_mps(checkpoint)
            if trajectory_final is not None:
                final_mps = trajectory_final

        if self.parallel and effective_num_traj > 1:
            for traj_index, trajectory in run_backend_parallel(
                worker_fn=_program_worker,
                payload=payload,
                n_jobs=effective_num_traj,
                max_workers=self.max_workers,
                show_progress=self.show_progress,
                desc="Running program trajectories",
                max_retries=self.max_retries,
                retry_exceptions=self.retry_exceptions,
                mp_context=self.mp_context,
            ):
                consume(traj_index, trajectory)
        else:
            iterator = tqdm(
                range(effective_num_traj),
                desc="Running program trajectories",
                ncols=80,
                disable=not self.show_progress,
            )
            for traj_index in iterator:
                trajectory = call_serial_capped(
                    _program_worker,
                    traj_index,
                    payload,
                    n_threads=available_cpus(),
                )
                consume(traj_index, trajectory)

        for segment_result, diag_per_traj in zip(segment_results, diagnostic_buffers, strict=True):
            if segment_result.observables:
                aggregate_trajectories(segment_result)
            if diag_per_traj is not None:
                segment_result.runtime_cost, segment_result.max_bond, segment_result.total_bond = aggregate_diagnostics(
                    diag_per_traj
                )
            if segment_result.measurements:
                aggregate_counts(segment_result)

        expectation_values, times, counts = stitch_program_results(segment_results, compiled.observables)
        result = Result(
            observables=[copy.deepcopy(obs) for obs in compiled.observables],
            expectation_values=expectation_values,
            times=times,
            counts=counts,
            noise_model=compiled.default_noise_model,
            segment_results=segment_results,
        )
        if final_mps is not None:
            result.output_state = State.from_mps(final_mps)
        return result

    # -----------------------------------------------------------------------
    # Analog (Hamiltonian) simulation -- primary YAQS simulation mode
    # -----------------------------------------------------------------------
    def _run_analog(
        self,
        initial_state: State | list[State],
        operator: Hamiltonian,
        sim_params: AnalogSimParams,
        noise_model: NoiseModel | None,
        result: Result,
    ) -> None:
        """Run analog simulation trajectories for Hamiltonian evolution.

        Selects the appropriate analog simulation backend based on ``sim_params.order``
        (either one-site or two-site evolution) and runs the simulation trajectories for the given
        Hamiltonian. The trajectories are executed and the results are aggregated.

        A ``list[State]`` ``initial_state`` triggers the deterministic unitary ensemble
        path via :meth:`_run_ensemble`.

        Args:
            initial_state: The initial system state as a :class:`State`, or a list of
                :class:`State` objects for deterministic unitary analog ensemble evolution.
            operator: The Hamiltonian specification (materialized once per ``run`` call).
            sim_params: Simulation parameters for analog simulation.
            noise_model: The noise model applied during simulation.
            result: Output container populated during this run.

        Raises:
            ValueError: If ``get_state=True`` is combined with a non-trivial noise model
                on ``mps`` or ``vector`` representations (the trajectory ensemble has no
                single representative state). Lindblad ``density_matrix`` evolution always
                returns the exact ensemble-averaged state when ``get_state=True``.
        """
        if isinstance(initial_state, list):
            initial_state_list = cast("list[State]", initial_state)
            if operator.is_piecewise:
                msg = "Piecewise Hamiltonians do not support list[State] ensemble execution."
                raise ValueError(msg)
            if any(spec.representation != "mps" for spec in initial_state_list):
                msg = "list[State] analog ensemble currently supports only State.representation='mps'."
                raise ValueError(msg)
            operator.ensure_mpo()
            for spec in initial_state_list:
                spec.ensure_encoded("mps")
                _validate_state_hamiltonian_pairing(spec, operator)
            if noise_model is not None:
                validate_noise_model_for_run(
                    noise_model,
                    length=operator.length,
                    physical_dimensions=(initial_state_list[0].physical_dimensions if initial_state_list else None),
                    representation="mps",
                    is_ensemble=True,
                    sim_params=sim_params,
                )
            self._run_ensemble(
                [spec.mps for spec in initial_state_list],
                operator.mpo,
                sim_params,
                noise_model,
                result,
            )
            return

        initial_state.ensure_encoded(initial_state.representation)
        mps = _materialized_mps(initial_state)
        state_rep = initial_state.representation
        _validate_state_hamiltonian_pairing(initial_state, operator)
        if noise_model is not None:
            validate_noise_model_for_run(
                noise_model,
                length=initial_state.length,
                physical_dimensions=initial_state.physical_dimensions,
                representation=state_rep,
                sim_params=sim_params,
            )
        if operator.is_piecewise:
            if state_rep != "mps":
                msg = "Piecewise Hamiltonians currently require State.representation='mps'."
                raise ValueError(msg)
            if sim_params.evolution_mode != EvolutionMode.TDVP:
                msg = "Piecewise Hamiltonians require evolution_mode=EvolutionMode.TDVP."
                raise ValueError(msg)
            if sim_params.multi_time_observables:
                msg = "Piecewise Hamiltonians do not support multi_time_observables."
                raise ValueError(msg)
        mpo_op: MPO | tuple[MPO, ...] | None
        if state_rep == "mps":
            assert mps is not None, "MPS representation requires a materialized MPS."
            mpo_op = _expand_analog_operator(
                operator,
                sim_params,
                physical_dimensions=mps.physical_dimensions,
            )
            h_sparse = None
        else:
            mpo_op, h_sparse = _prepare_hamiltonian_for_run(operator, state_rep)

        backend: Callable[..., tuple[NDArray[np.float64], Any, Any]]
        if state_rep == "density_matrix":
            backend = lindblad_evolve
        elif state_rep == "vector":
            backend = mcwf
        elif sim_params.order == 1:
            backend = analog_tjm_1
        else:
            backend = analog_tjm_2

        if (
            noise_model is None
            or all(proc["strength"] == 0 for proc in noise_model.processes)
            or state_rep == "density_matrix"
        ):
            effective_num_traj = 1
        else:
            if sim_params.get_state:
                msg = "Cannot return state in noisy analog simulation due to stochastics."
                raise ValueError(msg)
            effective_num_traj = sim_params.num_traj

        _prepare_result_observables(result, sim_params, num_traj=effective_num_traj)
        worker_params = cast("AnalogSimParams", _worker_sim_params(sim_params))

        diag_per_traj: NDArray[np.float64] | None = None
        if state_rep == "mps":
            diag_per_traj, _ = allocate_diagnostic_buffers(sim_params, num_traj=effective_num_traj)

        payload: dict[str, Any]
        worker_fn: Callable[[int], Any]

        if state_rep == "vector":
            assert h_sparse is not None
            ctx = preprocess_mcwf(
                psi_initial=initial_state.vector,
                h_sparse=h_sparse,
                noise_model=noise_model,
                sim_params=worker_params,
                num_sites=initial_state.length,
                physical_dimensions=initial_state.physical_dimensions,
            )
            payload = {"ctx": ctx}
            worker_fn = _mcwf_worker
        elif state_rep == "density_matrix":
            assert h_sparse is not None
            lindblad_ctx = preprocess_lindblad(
                rho_initial=initial_state.density_matrix,
                h_sparse=h_sparse,
                noise_model=noise_model,
                sim_params=worker_params,
                num_sites=initial_state.length,
                physical_dimensions=initial_state.physical_dimensions,
            )
            payload = {"ctx": lindblad_ctx}
            worker_fn = _lindblad_ctx_worker
        else:
            assert mps is not None, "MPS representation requires a materialized MPS."
            assert mpo_op is not None
            payload = {
                "initial_state": mps,
                "noise_model": noise_model,
                "sim_params": worker_params,
                "operator": mpo_op,
                "backend": backend,
            }
            worker_fn = _analog_worker

        final_mps: MPS | None = None
        final_psi: np.ndarray | None = None
        final_rho: np.ndarray | None = None

        if self.parallel and effective_num_traj > 1:
            for i, traj_payload in run_backend_parallel(
                worker_fn=worker_fn,
                payload=payload,
                n_jobs=effective_num_traj,
                max_workers=self.max_workers,
                show_progress=self.show_progress,
                desc="Running trajectories",
                max_retries=self.max_retries,
                retry_exceptions=self.retry_exceptions,
                mp_context=self.mp_context,
            ):
                traj_data, traj_diag, traj_final = traj_payload
                _store_observable_trajectory(result, sim_params, traj_index=i, sorted_traj_data=traj_data)
                if traj_diag is not None and diag_per_traj is not None:
                    diag_per_traj[:, i, :] = traj_diag
                if traj_final is not None:
                    if state_rep == "vector":
                        final_psi = cast("np.ndarray", traj_final)
                    elif state_rep == "density_matrix":
                        final_rho = cast("np.ndarray", traj_final)
                    else:
                        final_mps = cast("MPS", traj_final)
        else:
            n_threads = available_cpus()

            args: list[Any]
            if state_rep == "vector":
                args = [(i, copy.copy(ctx)) for i in range(effective_num_traj)]
            elif state_rep == "density_matrix":
                args = [lindblad_ctx for _ in range(effective_num_traj)]
            else:
                args = [(i, mps, noise_model, worker_params, mpo_op) for i in range(effective_num_traj)]

            iterator = tqdm(args, desc="Running trajectories", ncols=80, disable=not self.show_progress)

            for i, arg in enumerate(iterator):
                traj_data, traj_diag, traj_final = call_serial_capped(backend, arg, n_threads=n_threads)
                _store_observable_trajectory(result, sim_params, traj_index=i, sorted_traj_data=traj_data)
                if traj_diag is not None and diag_per_traj is not None:
                    diag_per_traj[:, i, :] = traj_diag
                if traj_final is not None:
                    if state_rep == "vector":
                        final_psi = cast("np.ndarray", traj_final)
                    elif state_rep == "density_matrix":
                        final_rho = cast("np.ndarray", traj_final)
                    else:
                        final_mps = cast("MPS", traj_final)

        if state_rep == "vector":
            _store_mcwf_final_state(
                result,
                final_psi,
                length=initial_state.length,
                physical_dimensions=initial_state.physical_dimensions,
            )
        elif state_rep == "density_matrix":
            _store_lindblad_final_state(
                result,
                final_rho,
                length=initial_state.length,
                physical_dimensions=initial_state.physical_dimensions,
            )
        else:
            _store_final_mps(result, final_mps)

        if diag_per_traj is not None:
            result.runtime_cost, result.max_bond, result.total_bond = aggregate_diagnostics(diag_per_traj)
        aggregate_trajectories(result)

    # -----------------------------------------------------------------------
    # Digital (circuit) simulation
    # -----------------------------------------------------------------------
    def _run_digital_sim(
        self,
        initial_state: MPS,
        operator: QuantumCircuit,
        sim_params: DigitalSimParams,
        noise_model: NoiseModel | None,
        result: Result,
    ) -> None:
        """Run digital circuit simulation trajectories.

        ``num_traj`` and ``shots`` are independent:

        - ``num_traj`` sizes the noisy stochastic ensemble for observables and diagnostics.
        - ``shots`` is the total bitstring-sample budget.
        - Shots-only noisy runs use ``shots`` one-shot trajectories; noiseless shots-only
          and noiseless combined runs use one trajectory and sample all ``shots`` from it.
        - Combined noisy runs (observables and shots) run ``num_traj`` trajectories and
          distribute the total ``shots`` across them. ``shots < num_traj`` is valid: some
          trajectories still contribute observables but receive zero measurement samples.

        Raises:
            ValueError: If ``get_state`` is ``True`` with a non-trivial noise model.
        """
        wants_obs = bool(sim_params.observables)
        wants_shots = sim_params.shots is not None
        shots_only = wants_shots and not wants_obs
        noisy = not (noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes))

        if noisy and sim_params.get_state:
            msg = "Cannot return state in noisy circuit simulation due to stochastics."
            raise ValueError(msg)

        effective_num_traj, per_call_shots, shot_distribution = _plan_digital_shots(sim_params, noisy=noisy)

        effective_num_mid_measurements = sim_params.num_mid_measurements
        if sim_params.sample_layers:
            dag = circuit_to_dag(operator)
            effective_num_mid_measurements = sum(
                1
                for n in dag.op_nodes()
                if n.op.name == "barrier" and str(getattr(n.op, "label", "")).strip().upper() == "SAMPLE_OBSERVABLES"
            )

        if wants_obs:
            _prepare_result_observables(
                result,
                sim_params,
                num_traj=effective_num_traj,
                num_mid_measurements=effective_num_mid_measurements,
            )
            worker_params = cast("DigitalSimParams", _worker_sim_params(sim_params))
            if sim_params.sample_layers:
                worker_params.num_mid_measurements = effective_num_mid_measurements
        else:
            worker_params = copy.deepcopy(sim_params)

        diag_per_traj: NDArray[np.float64] | None = None
        if not shots_only:
            diag_per_traj, _ = allocate_diagnostic_buffers(
                sim_params,
                num_traj=effective_num_traj,
                num_mid_measurements=effective_num_mid_measurements,
            )

        if wants_shots:
            if noisy and shots_only:
                result.measurements = [None] * effective_num_traj
            else:
                result.measurements = [None] * (effective_num_traj if noisy else 1)

        payload: dict[str, Any] = {
            "initial_state": initial_state,
            "noise_model": noise_model,
            "sim_params": worker_params,
            "operator": operator,
        }
        if per_call_shots is not None:
            payload["per_call_shots"] = per_call_shots
            WORKER_CTX["per_call_shots"] = per_call_shots
        if shot_distribution is not None:
            payload["shot_distribution"] = shot_distribution
            WORKER_CTX["shot_distribution"] = shot_distribution

        final_mps: MPS | None = None

        def _consume(
            i: int,
            traj_data: NDArray[np.float64] | None,
            traj_diag: NDArray[np.float64] | None,
            shot_counts: dict[int, int] | None,
            traj_final: MPS | None,
        ) -> None:
            nonlocal final_mps
            if traj_data is not None and wants_obs:
                _store_observable_trajectory(
                    result,
                    sim_params,
                    traj_index=i,
                    sorted_traj_data=cast("NDArray[np.float64] | NDArray[np.complex128]", traj_data),
                )
            if traj_diag is not None and diag_per_traj is not None:
                diag_per_traj[:, i, :] = traj_diag
            if shot_counts is not None:
                if noisy:
                    result.measurements[i] = shot_counts
                else:
                    result.measurements[0] = shot_counts
            if traj_final is not None:
                final_mps = traj_final

        try:
            if self.parallel and effective_num_traj > 1:
                for i, traj_payload in run_backend_parallel(
                    worker_fn=_digital_worker,
                    payload=payload,
                    n_jobs=effective_num_traj,
                    max_workers=self.max_workers,
                    show_progress=self.show_progress,
                    desc="Running trajectories",
                    max_retries=self.max_retries,
                    retry_exceptions=self.retry_exceptions,
                    mp_context=self.mp_context,
                ):
                    traj_data, traj_diag, shot_counts, traj_final = traj_payload
                    _consume(i, traj_data, traj_diag, shot_counts, traj_final)
            else:
                n_threads = available_cpus()
                args: list[tuple[int, MPS, NoiseModel | None, DigitalSimParams, QuantumCircuit]] = [
                    (i, initial_state, noise_model, worker_params, operator) for i in range(effective_num_traj)
                ]
                iterator = tqdm(args, desc="Running trajectories", ncols=80, disable=not self.show_progress)
                for i, arg in enumerate(iterator):
                    traj_data, traj_diag, shot_counts, traj_final = call_serial_capped(
                        digital_tjm, arg, n_threads=n_threads
                    )
                    _consume(i, traj_data, traj_diag, shot_counts, traj_final)
        finally:
            WORKER_CTX.pop("per_call_shots", None)
            WORKER_CTX.pop("shot_distribution", None)

        _store_final_mps(result, final_mps)
        if diag_per_traj is not None:
            result.runtime_cost, result.max_bond, result.total_bond = aggregate_diagnostics(diag_per_traj)
        if wants_obs:
            aggregate_trajectories(result)
        if wants_shots:
            aggregate_counts(result)

    def _run_circuit(
        self,
        initial_state: State,
        operator: QuantumCircuit,
        sim_params: DigitalSimParams,
        noise_model: NoiseModel | None,
        result: Result,
    ) -> None:
        """Run circuit-based simulation trajectories.

        Requires :attr:`~mqt.yaqs.core.data_structures.state.State.representation` ``"mps"``,
        materializes the state, validates qubit counts, and runs digital simulation.

        Raises:
            ValueError: If ``initial_state.representation`` is not ``"mps"``.
        """
        if initial_state.representation != "mps":
            msg = (
                "Circuit simulation requires State.representation='mps'. "
                "Use representation='vector' or 'density_matrix' only for analog Hamiltonian runs."
            )
            raise ValueError(msg)
        initial_state.ensure_encoded("mps")
        mps = initial_state.mps

        if mps.length != operator.num_qubits:
            msg = "State and circuit qubit counts do not match."
            raise ValueError(msg)

        if noise_model is not None:
            validate_noise_model_for_run(
                noise_model,
                length=mps.length,
                physical_dimensions=mps.physical_dimensions,
                representation="mps",
                is_digital=True,
            )

        self._run_digital_sim(
            mps,
            operator,
            sim_params,
            noise_model,
            result,
        )

    # -----------------------------------------------------------------------
    # Unitary ensemble (deterministic, no noise)
    # -----------------------------------------------------------------------
    def _run_ensemble(
        self,
        initial_states: list[MPS],
        operator: MPO,
        sim_params: AnalogSimParams,
        noise_model: NoiseModel | None,
        result: Result,
    ) -> None:
        """Run deterministic unitary evolution for an ensemble of initial MPS states.

        This mode is intentionally separate from stochastic-noise trajectories: users may either
        provide a list of initial states with no noise (this mode), or a single initial state with
        noise (standard TJM / Lindblad / MCWF paths).

        Args:
            initial_states: One MPS per ensemble member; lengths must match ``operator.length``.
            operator: Hamiltonian as an MPO shared by all members.
            sim_params: Analog parameters; ``num_traj`` is set to ``len(initial_states)``.
            noise_model: Must be absent or contain only zero-strength processes.
            result: Output container populated during this run.

        Raises:
            ValueError: If noisy simulation is requested with a list of states, if
                ``representation`` is unsupported in list mode, if the list is empty, or if state
                lengths do not match the MPO.
        """
        if noise_model is not None and any(proc["strength"] > 0 for proc in noise_model.processes):
            msg = (
                "list[State] with noisy analog simulation is not supported yet. "
                "Use list[State] with no noise for unitary ensembles, or use a single State for noisy simulation."
            )
            raise ValueError(msg)
        if not initial_states:
            msg = "initial_state list must not be empty."
            raise ValueError(msg)

        if any(state.length != operator.length for state in initial_states):
            msg = "All initial states in the list must match the MPO length."
            raise ValueError(msg)
        if sim_params.get_state:
            msg = "get_state=True is not supported for list[State] analog ensemble mode."
            raise ValueError(msg)

        effective_num_traj = len(initial_states)

        _prepare_result_observables(result, sim_params, num_traj=effective_num_traj)
        worker_params = cast("AnalogSimParams", _worker_sim_params(sim_params))
        diag_per_traj, _ = allocate_diagnostic_buffers(sim_params, num_traj=effective_num_traj)

        n_pairs = len(sim_params.multi_time_observables)
        n_cols = len(sim_params.times) if sim_params.sample_timesteps else 1
        multi_time_matrix: NDArray[np.complex128] | None = None
        if n_pairs > 0:
            multi_time_matrix = np.zeros((len(initial_states), n_pairs, n_cols), dtype=np.complex128)
            result.multi_time_times = np.asarray(
                sim_params.times if sim_params.sample_timesteps else [sim_params.elapsed_time],
                dtype=np.float64,
            )

        payload: dict[str, Any] = {
            "initial_states": initial_states,
            "sim_params": worker_params,
            "operator": operator,
        }

        if self.parallel and len(initial_states) > 1:
            for i, (obs_result, traj_diag, multi_time_result) in run_backend_parallel(
                worker_fn=_ensemble_worker,
                payload=payload,
                n_jobs=len(initial_states),
                max_workers=self.max_workers,
                show_progress=self.show_progress,
                desc="Running unitary ensemble",
                max_retries=self.max_retries,
                retry_exceptions=self.retry_exceptions,
                mp_context=self.mp_context,
            ):
                _store_observable_trajectory(result, sim_params, traj_index=i, sorted_traj_data=obs_result)
                diag_per_traj[:, i, :] = traj_diag
                if multi_time_matrix is not None:
                    assert multi_time_result is not None
                    multi_time_matrix[i] = multi_time_result
        else:
            n_threads = available_cpus()
            args = [(i, initial_states[i], worker_params, operator) for i in range(len(initial_states))]
            iterator = tqdm(args, desc="Running unitary ensemble", ncols=80, disable=not self.show_progress)
            for i, arg in enumerate(iterator):
                obs_result, traj_diag, multi_time_result = call_serial_capped(
                    ensemble_member_worker, arg, n_threads=n_threads
                )
                _store_observable_trajectory(result, sim_params, traj_index=i, sorted_traj_data=obs_result)
                diag_per_traj[:, i, :] = traj_diag
                if multi_time_matrix is not None:
                    assert multi_time_result is not None
                    multi_time_matrix[i] = multi_time_result

        result.runtime_cost, result.max_bond, result.total_bond = aggregate_diagnostics(diag_per_traj)
        aggregate_trajectories(result)
        if multi_time_matrix is not None:
            result.multi_time_results = np.mean(multi_time_matrix, axis=0)
