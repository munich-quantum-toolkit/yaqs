# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Parallel pool workers for process-tensor schedule sequence simulation.

Workers follow the standard :mod:`mqt.yaqs.core.parallel_utils` pattern:
``(job_idx, payload=None)`` with flat indexing ``sequence_index * num_trajectories + trajectory_index``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.yaqs.core.parallel_utils import resolve_worker_ctx, unpack_flat_job

from ...shared.encoding import normalize_backend_rho, pack_rho8
from ...shared.intervention_steps import apply_intervention_to_backend
from ...shared.utils import (
    _evolve_backend_state,
    extract_site0_rho,
    resolve_stochastic_solver,
)

if TYPE_CHECKING:
    from mqt.yaqs.analog.mcwf import MCWFContext
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.mps import MPS


def _get_times_cached(times_cache: dict[tuple[float, float], np.ndarray], *, dt: float, duration: float) -> np.ndarray:
    """Return a cached time grid for a step.

    Args:
        times_cache: Cache mapping ``(dt, duration)`` to time grids.
        dt: Integration step size.
        duration: Desired evolution duration.

    Returns:
        A 1D float array suitable for ``AnalogSimParams.times``.

    Raises:
        ValueError: If ``duration`` is not a positive integer multiple of ``dt``.
    """
    dt_f = float(dt)
    dur_f = float(duration)
    if abs(dur_f) < 1e-15:
        key = (dt_f, 0.0)
        out = times_cache.get(key)
        if out is None:
            out = np.array([0.0], dtype=np.float64)
            times_cache[key] = out
        return out
    n_steps = round(dur_f / dt_f)
    if n_steps < 1 or abs(n_steps * dt_f - dur_f) > 1e-9 * max(1.0, dur_f):
        msg = f"duration={dur_f} must be a positive integer multiple of dt={dt_f}."
        raise ValueError(msg)
    key = (dt_f, dur_f)
    out = times_cache.get(key)
    if out is None:
        out = np.linspace(0.0, dur_f, n_steps + 1)
        times_cache[key] = out
    return out


# ---------------------------------------------------------------------------
# Parallel job payload (pickle-stable keys for WORKER_CTX)
# ---------------------------------------------------------------------------
# ``simulate_sequences`` in :mod:`mqt.yaqs.characterization.memory.backends.sequences.workflow` passes this dict to
# :func:`~mqt.yaqs.core.parallel_utils.run_indexed_jobs` (initializer →
# :data:`~mqt.yaqs.core.parallel_utils.WORKER_CTX`) or directly to workers on the
# serial path. Workers use :func:`~mqt.yaqs.core.parallel_utils.resolve_worker_ctx`
# and :func:`~mqt.yaqs.core.parallel_utils.unpack_flat_job`.
#
#   intervention_steps               list[list[step]] per sequence — MP tuple or unitary dict
#   initial_psi             list of initial backend states (dense or MPS; one per sequence)
#   num_trajectories        flat-index stride (1 when noise_model is None)
#   operator, sim_params    Hamiltonian MPO and analog parameters
#   timesteps               process-tensor schedule: ``num_interventions+1`` evolution segments
#   timesteps_rows          optional per-sequence durations, each length ``num_interventions+1``
#   operators_list          optional per-sequence MPOs, length ``num_interventions+1`` per sequence
#   noise_model             None for deterministic surrogate sequences
#   mcwf_static_ctx         static MCWF context for the whole sequence
#   mcwf_static_ctx_list    optional per-evolution-slot context (length ``num_interventions+1``)
#   e_features_rows         per-sequence Choi rows ``(num_interventions, d_e)`` — required for trace workers


# ---------------------------------------------------------------------------
# Process-tensor schedule — ``num_interventions`` interventions, ``num_interventions+1`` evolutions
# ---------------------------------------------------------------------------
def _validate_timesteps_rows_schedule(
    intervention_steps_list: list[list[Any]],
    timesteps_rows: list[list[float]],
) -> None:
    """Require per-sequence duration rows of length ``num_interventions + 1``.

    Args:
        intervention_steps_list: Per-sequence intervention step lists.
        timesteps_rows: Per-sequence evolution durations.

    Raises:
        ValueError: If row counts do not match the number of sequences or intervention counts.
    """
    num_sequences = len(intervention_steps_list)
    if len(timesteps_rows) != num_sequences:
        msg = "`timesteps_rows` length must match number of sequences."
        raise ValueError(msg)
    for i, pairs in enumerate(intervention_steps_list):
        n_interventions = len(pairs)
        if len(timesteps_rows[i]) != n_interventions + 1:
            msg = (
                f"Sequence {i}: `timesteps_rows[{i}]` must have length "
                f"num_interventions+1={n_interventions + 1}, got {len(timesteps_rows[i])}."
            )
            raise ValueError(msg)


def _validate_operators_list_schedule(
    intervention_steps_list: list[list[Any]],
    operators_list: list[list[MPO]],
) -> None:
    """Require per-sequence operator lists of length ``num_interventions + 1``.

    Args:
        intervention_steps_list: Per-sequence intervention step lists.
        operators_list: Per-sequence Hamiltonian MPO lists.

    Raises:
        ValueError: If operator list counts do not match the number of sequences or intervention counts.
    """
    num_sequences = len(intervention_steps_list)
    if len(operators_list) != num_sequences:
        msg = "`operators_list` length must match number of sequences."
        raise ValueError(msg)
    for i, pairs in enumerate(intervention_steps_list):
        n_interventions = len(pairs)
        if len(operators_list[i]) != n_interventions + 1:
            msg = (
                f"Sequence {i}: `operators_list[{i}]` must have length "
                f"num_interventions+1={n_interventions + 1}, got {len(operators_list[i])}."
            )
            raise ValueError(msg)


def _validate_static_ctx_list_schedule(
    intervention_steps_list: list[list[Any]],
    static_ctx_list: list[list[MCWFContext | None]],
) -> None:
    """Require per-sequence MCWF context lists of length ``num_interventions + 1``.

    Args:
        intervention_steps_list: Per-sequence intervention step lists.
        static_ctx_list: Per-sequence MCWF context lists.

    Raises:
        ValueError: If context list counts do not match the number of sequences or intervention counts.
    """
    num_sequences = len(intervention_steps_list)
    if len(static_ctx_list) != num_sequences:
        msg = "`static_ctx_list` length must match number of sequences."
        raise ValueError(msg)
    for i, pairs in enumerate(intervention_steps_list):
        n_interventions = len(pairs)
        if len(static_ctx_list[i]) != n_interventions + 1:
            msg = (
                f"Sequence {i}: `static_ctx_list[{i}]` must have length "
                f"num_interventions+1={n_interventions + 1}, got {len(static_ctx_list[i])}."
            )
            raise ValueError(msg)


def _validate_process_tensor_schedule_inputs(
    *,
    intervention_steps_list: list[list[Any]],
    timesteps: list[float],
    timesteps_rows: list[list[float]] | None,
    operators_list: list[list[MPO]] | None,
    static_ctx_list: list[list[MCWFContext | None]] | None,
) -> None:
    """Require compatible lengths for the process-tensor schedule convention.

    Args:
        intervention_steps_list: Per-sequence intervention step lists.
        timesteps: Shared evolution durations when ``timesteps_rows`` is omitted.
        timesteps_rows: Optional per-sequence duration rows.
        operators_list: Optional per-sequence Hamiltonian MPO lists.
        static_ctx_list: Optional per-sequence MCWF context lists.

    Raises:
        ValueError: If sequence lengths or optional per-sequence schedules are inconsistent.
    """
    num_sequences = len(intervention_steps_list)
    if num_sequences == 0:
        return
    if timesteps_rows is None:
        intervention_counts = [len(p) for p in intervention_steps_list]
        if len(set(intervention_counts)) != 1:
            msg = "All sequences must share the same num_interventions when `timesteps_rows` is omitted."
            raise ValueError(msg)
        n_interventions = intervention_counts[0]
        if len(timesteps) != n_interventions + 1:
            msg = (
                "Process-tensor schedule: `timesteps` must have length num_interventions+1 "
                f"({n_interventions + 1} for num_interventions={n_interventions} intervention steps), "
                f"got {len(timesteps)}."
            )
            raise ValueError(msg)
    else:
        _validate_timesteps_rows_schedule(intervention_steps_list, timesteps_rows)
    if operators_list is not None:
        _validate_operators_list_schedule(intervention_steps_list, operators_list)
    if static_ctx_list is not None:
        _validate_static_ctx_list_schedule(intervention_steps_list, static_ctx_list)


def _copy_initial_backend_state(state: np.ndarray | MPS) -> np.ndarray | MPS:
    """Return a backend-safe copy of an initial state (dense vector or MPS).

    Args:
        state: Initial MCWF state vector or TJM MPS.

    Returns:
        A copied dense array or deep-copied MPS suitable for in-place evolution.
    """
    if isinstance(state, np.ndarray):
        return np.asarray(state, dtype=np.complex128).copy()
    return copy.deepcopy(state)


def _schedule_slots_for_sequence(
    *,
    sequence_idx: int,
    num_interventions: int,
    timesteps: list[float],
    timesteps_rows: list[list[float]] | None,
    hamiltonian: MPO,
    operators_list: list[list[MPO]] | None,
    mcwf_static_ctx: MCWFContext | None,
    mcwf_static_ctx_list: list[list[MCWFContext | None]] | None,
) -> tuple[list[float], list[MPO], list[MCWFContext | None]]:
    """Resolve per-sequence process-tensor schedule durations, Hamiltonians, and MCWF contexts.

    Args:
        sequence_idx: Index of the sequence being simulated.
        num_interventions: Number of intervention steps.
        timesteps: Default process-tensor schedule of length ``num_interventions+1``.
        timesteps_rows: Optional per-sequence durations, each length ``num_interventions+1``.
        hamiltonian: Default Hamiltonian MPO for every evolution slot.
        operators_list: Optional per-sequence MPO list of length ``num_interventions+1``.
        mcwf_static_ctx: Shared static MCWF context when no per-slot list is given.
        mcwf_static_ctx_list: Optional per-sequence MCWF contexts, each length ``num_interventions+1``.

    Returns:
        Tuple ``(durations, operators, mcwf_contexts)`` with one entry per evolution slot.
    """
    if timesteps_rows is not None:
        durs = [float(timesteps_rows[sequence_idx][i]) for i in range(num_interventions + 1)]
    else:
        durs = [float(timesteps[i]) for i in range(num_interventions + 1)]
    ops: list[MPO] = []
    ctxs: list[MCWFContext | None] = []
    for i in range(num_interventions + 1):
        op = hamiltonian if operators_list is None else operators_list[sequence_idx][i]
        ctx = mcwf_static_ctx if mcwf_static_ctx_list is None else mcwf_static_ctx_list[sequence_idx][i]
        ops.append(op)
        ctxs.append(ctx)
    return durs, ops, ctxs


def _reshape_choi_feature_rows(raw_rows: np.ndarray, *, num_steps: int) -> np.ndarray:
    """Validate and reshape per-sequence Choi feature rows.

    Args:
        raw_rows: Flat or matrix Choi feature storage for one sequence.
        num_steps: Expected number of intervention steps.

    Returns:
        Float32 array of shape ``(num_steps, d_e)``.

    Raises:
        ValueError: If the row count does not match ``num_steps``.
    """
    rows = np.asarray(raw_rows, dtype=np.float32)
    if rows.ndim == 1:
        if rows.size % num_steps != 0:
            msg = f"Choi feature rows length {rows.size} is not divisible by num_steps={num_steps}."
            raise ValueError(msg)
        rows = rows.reshape(num_steps, -1)
    elif rows.ndim == 2:
        if rows.shape[0] != num_steps:
            msg = f"Choi feature rows must have length num_steps={num_steps}, got {rows.shape[0]}."
            raise ValueError(msg)
    else:
        msg = f"Choi feature rows must be 1D or 2D, got ndim={rows.ndim}."
        raise ValueError(msg)
    return rows


# ---------------------------------------------------------------------------
# Sequence simulation core
# ---------------------------------------------------------------------------
def _simulate_seq_core(
    *,
    sequence_idx: int,
    trajectory_idx: int,
    worker_ctx: dict[str, Any],
    collect_trace: bool,
) -> tuple[np.ndarray, float, dict[str, Any] | None]:
    """Shared process-tensor schedule: ``U_1`` then ``num_interventions`` times (reprepare → ``U``).

    Optionally collect a per-sequence trace dict when ``collect_trace`` is set.

    Returns:
        Tuple ``(rho_final, cumulative_weight, trace)`` where ``trace`` is ``None`` when
        ``collect_trace`` is ``False``.
    """
    intervention_steps = worker_ctx["intervention_steps"][sequence_idx]
    hamiltonian = worker_ctx["operator"]
    sim_params = worker_ctx["sim_params"]
    timesteps: list[float] = worker_ctx["timesteps"]
    timesteps_rows: list[list[float]] | None = worker_ctx.get("timesteps_rows")
    operators_list: list[list[MPO]] | None = worker_ctx.get("operators_list")
    mcwf_ctx_list: list[list[MCWFContext | None]] | None = worker_ctx.get("mcwf_static_ctx_list")
    noise_model = worker_ctx["noise_model"]
    initial_states: list[np.ndarray | MPS] = worker_ctx["initial_psi"]

    if noise_model is None:
        assert int(worker_ctx["num_trajectories"]) == 1, "num_trajectories must be 1 when noise_model is None."

    solver = resolve_stochastic_solver(sim_params, solver=worker_ctx.get("solver"))
    state = _copy_initial_backend_state(initial_states[sequence_idx])
    times_cache: dict[tuple[float, float], np.ndarray] = worker_ctx.setdefault("_times_cache", {})
    step_params = copy.copy(sim_params)
    step_params.num_traj = 1
    step_params.get_state = True

    num_interventions = len(intervention_steps)
    durs, ops, mcwf_ctxs = _schedule_slots_for_sequence(
        sequence_idx=sequence_idx,
        num_interventions=num_interventions,
        timesteps=timesteps,
        timesteps_rows=timesteps_rows,
        hamiltonian=hamiltonian,
        operators_list=operators_list,
        mcwf_static_ctx=worker_ctx.get("mcwf_static_ctx"),
        mcwf_static_ctx_list=mcwf_ctx_list,
    )

    step_probs: list[float] = []
    prob_skipped_renormalize: list[bool] = []

    cumulative_weight = 1.0
    duration = float(durs[0])
    step_params.elapsed_time = duration
    step_params.times = _get_times_cached(times_cache, dt=float(step_params.dt), duration=duration)
    state = _evolve_backend_state(
        state,
        ops[0],
        noise_model,
        step_params,
        solver,
        traj_idx=trajectory_idx,
        static_ctx=mcwf_ctxs[0],
    )

    break_step: int | None = None
    num_evolutions_in_loop = 0

    for step_idx, step in enumerate(intervention_steps):
        state, sp = apply_intervention_to_backend(
            state,
            step,
            solver=solver,
            chain_length=int(hamiltonian.length),
        )
        step_probs.append(sp)
        prob_skipped_renormalize.append(sp <= 1e-15)
        cumulative_weight *= sp
        if cumulative_weight < 1e-15:
            break_step = step_idx
            break

        duration = float(durs[step_idx + 1])
        step_params.elapsed_time = duration
        step_params.times = _get_times_cached(times_cache, dt=float(step_params.dt), duration=duration)
        state = _evolve_backend_state(
            state,
            ops[step_idx + 1],
            noise_model,
            step_params,
            solver,
            traj_idx=trajectory_idx,
            static_ctx=mcwf_ctxs[step_idx + 1],
        )
        num_evolutions_in_loop += 1

    rho_final = extract_site0_rho(state)
    wfin = float(cumulative_weight)

    trace: dict[str, Any] | None = None
    if collect_trace:
        terminated_early = break_step is not None or num_evolutions_in_loop < num_interventions
        mins = min(step_probs) if step_probs else 0.0
        maxs = max(step_probs) if step_probs else 0.0
        means = float(np.mean(step_probs)) if step_probs else 0.0
        trace = {
            "terminated_early": bool(terminated_early),
            "break_step": break_step,
            "cumulative_weight_final": wfin,
            "step_probs": step_probs,
            "min_step_prob": float(mins),
            "max_step_prob": float(maxs),
            "mean_step_prob": float(means),
            "num_steps_completed": int(num_evolutions_in_loop),
            "num_reprepare_steps_recorded": len(step_probs),
            "prob_skipped_renormalize": prob_skipped_renormalize,
            "any_prob_skipped_renormalize": bool(any(prob_skipped_renormalize)),
        }

    return rho_final, wfin, trace


# ---------------------------------------------------------------------------
# Pool workers — signature (job_idx, payload=None)
# ---------------------------------------------------------------------------
def _seq_final_worker(
    job_idx: int,
    job_payload: dict[str, Any] | None = None,
) -> tuple[int, int, np.ndarray, float]:
    """Simulate one intervention sequence and return the final reduced state.

    Does not record per-step states (cheaper than :func:`_seq_trace_worker`).

    Args:
        job_idx: Flat index ``sequence_index * num_trajectories + trajectory_index``.
        job_payload: Per-pool shared context; defaults to :data:`~mqt.yaqs.core.parallel_utils.WORKER_CTX`.

    Returns:
        ``(sequence_index, trajectory_index, rho_final_site0, cumulative_weight)`` where
        ``rho_final_site0`` is the reduced state on site 0 after the last evolution segment
        (process-tensor schedule: ``num_interventions`` interventions,
        ``num_interventions+1`` evolutions).
    """
    worker_ctx = resolve_worker_ctx(job_payload)
    sequence_idx, trajectory_idx = unpack_flat_job(job_idx, int(worker_ctx["num_trajectories"]))

    rho_final, cum_w, _trace = _simulate_seq_core(
        sequence_idx=sequence_idx,
        trajectory_idx=trajectory_idx,
        worker_ctx=worker_ctx,
        collect_trace=False,
    )
    return (sequence_idx, trajectory_idx, rho_final, float(cum_w))


def _seq_final_worker_diagnostics(
    job_idx: int,
    job_payload: dict[str, Any] | None = None,
) -> tuple[int, int, np.ndarray, float, dict[str, Any]]:
    """Same as :func:`_seq_final_worker` but includes per-sequence trace diagnostics.

    Returns:
        Tuple ``(sequence_index, trajectory_index, rho_final, cumulative_weight, trace)``.

    Raises:
        RuntimeError: If the diagnostics trace is missing.
    """
    worker_ctx = resolve_worker_ctx(job_payload)
    sequence_idx, trajectory_idx = unpack_flat_job(job_idx, int(worker_ctx["num_trajectories"]))

    rho_final, cum_w, trace = _simulate_seq_core(
        sequence_idx=sequence_idx,
        trajectory_idx=trajectory_idx,
        worker_ctx=worker_ctx,
        collect_trace=True,
    )
    if trace is None:
        msg = "internal: diagnostics trace missing"
        raise RuntimeError(msg)
    return (sequence_idx, trajectory_idx, rho_final, float(cum_w), trace)


def _seq_trace_worker(
    job_idx: int,
    job_payload: dict[str, Any] | None = None,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, float]:
    """Simulate one sequence and record per-step reduced states (packed).

    Choi feature rows (``e_features_rows``) are passed through unchanged into the sample; the worker
    only ensures shape ``(num_steps, d_e)`` matches the simulation.

    Args:
        job_idx: Flat index ``sequence_index * num_trajectories + trajectory_index``.
        job_payload: Shared pool context; defaults to :data:`~mqt.yaqs.core.parallel_utils.WORKER_CTX`.

    Returns:
        ``(sequence_index, trajectory_index, rho0_packed, choi_features_matrix, rho_seq_packed, weight)`` where
        ``choi_features_matrix`` is ``(num_steps, d_e)`` and ``rho_seq_packed`` is ``(num_steps, 8)``.
        Here ``rho0_packed`` is the reduced state on site 0 **after** the first free evolution ``U_1`` and
        **before** the first intervention (process-tensor schedule boundary), matching the
        process-tensor slicing convention.

    Raises:
        ValueError: If required feature rows are missing or shapes are inconsistent.
    """
    worker_ctx = resolve_worker_ctx(job_payload)
    sequence_idx, trajectory_idx = unpack_flat_job(job_idx, int(worker_ctx["num_trajectories"]))

    intervention_steps = worker_ctx["intervention_steps"][sequence_idx]
    per_sequence_choi_rows: list[np.ndarray] | None = worker_ctx.get("e_features_rows")
    hamiltonian = worker_ctx["operator"]
    sim_params = worker_ctx["sim_params"]
    timesteps: list[float] = worker_ctx["timesteps"]
    timesteps_per_sequence = worker_ctx.get("timesteps_rows")
    hamiltonians_per_step = worker_ctx.get("operators_list")
    mcwf_ctx_per_step = worker_ctx.get("mcwf_static_ctx_list")
    noise_model = worker_ctx["noise_model"]
    initial_states: list[np.ndarray | MPS] = worker_ctx["initial_psi"]

    if noise_model is None:
        assert int(worker_ctx["num_trajectories"]) == 1, "num_trajectories must be 1 when noise_model is None."

    num_steps = len(intervention_steps)
    if num_steps == 0:
        msg = "Trace worker requires at least one intervention step."
        raise ValueError(msg)
    solver = resolve_stochastic_solver(sim_params, solver=worker_ctx.get("solver"))
    state = _copy_initial_backend_state(initial_states[sequence_idx])
    times_cache: dict[tuple[float, float], np.ndarray] = worker_ctx.setdefault("_times_cache", {})
    step_params = copy.copy(sim_params)
    step_params.num_traj = 1
    step_params.get_state = True

    if per_sequence_choi_rows is None:
        msg = "Trace worker requires `e_features_rows`: per-sequence Choi feature rows."
        raise ValueError(msg)
    choi_features_matrix = _reshape_choi_feature_rows(
        per_sequence_choi_rows[sequence_idx],
        num_steps=num_steps,
    )

    durs, ops, mcwf_ctxs = _schedule_slots_for_sequence(
        sequence_idx=sequence_idx,
        num_interventions=num_steps,
        timesteps=timesteps,
        timesteps_rows=timesteps_per_sequence,
        hamiltonian=hamiltonian,
        operators_list=hamiltonians_per_step,
        mcwf_static_ctx=worker_ctx.get("mcwf_static_ctx"),
        mcwf_static_ctx_list=mcwf_ctx_per_step,
    )

    # U_1: reduced state immediately before the first intervention (schedule boundary).
    duration = float(durs[0])
    step_params.elapsed_time = duration
    step_params.times = _get_times_cached(times_cache, dt=float(step_params.dt), duration=duration)
    state = _evolve_backend_state(
        state,
        ops[0],
        noise_model,
        step_params,
        solver,
        traj_idx=trajectory_idx,
        static_ctx=mcwf_ctxs[0],
    )

    rho0_raw = extract_site0_rho(state)
    rho0_packed = pack_rho8(normalize_backend_rho(rho0_raw)).astype(np.float32)

    cumulative_weight = 1.0
    last_rho_packed = rho0_packed.copy()
    rho_sequence_packed = np.empty((num_steps, 8), dtype=np.float32)
    out_i = 0

    for step_idx, step in enumerate(intervention_steps):
        state, step_prob = apply_intervention_to_backend(
            state,
            step,
            solver=solver,
            chain_length=int(hamiltonian.length),
        )
        cumulative_weight *= float(step_prob)
        if cumulative_weight < 1e-15:
            if out_i < num_steps:
                rho_sequence_packed[out_i:, :] = last_rho_packed[None, :]
            out_i = num_steps
            break

        duration = float(durs[step_idx + 1])
        step_params.elapsed_time = duration
        step_params.times = _get_times_cached(times_cache, dt=float(step_params.dt), duration=duration)
        state = _evolve_backend_state(
            state,
            ops[step_idx + 1],
            noise_model,
            step_params,
            solver,
            traj_idx=trajectory_idx,
            static_ctx=mcwf_ctxs[step_idx + 1],
        )

        rho_step = extract_site0_rho(state)
        rho_normalized = normalize_backend_rho(rho_step)
        last_rho_packed = pack_rho8(rho_normalized).astype(np.float32)
        rho_sequence_packed[out_i, :] = last_rho_packed
        out_i += 1

    if out_i < num_steps:
        rho_sequence_packed[out_i:, :] = last_rho_packed[None, :]
    return (
        sequence_idx,
        trajectory_idx,
        rho0_packed,
        choi_features_matrix,
        rho_sequence_packed,
        float(cumulative_weight),
    )
