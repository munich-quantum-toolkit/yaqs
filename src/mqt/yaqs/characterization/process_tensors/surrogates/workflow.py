# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Surrogate workflow: data generation and end-to-end training entry points.

**Public** (see ``__all__``): :func:`generate_data`, :func:`create_surrogate`.

:func:`generate_data` returns a :class:`~torch.utils.data.TensorDataset` for
:meth:`~mqt.yaqs.characterization.process_tensors.surrogates.model.TransformerComb.fit`.

**Internals** — Same execution pattern as :mod:`mqt.yaqs.simulator`: :func:`_simulate_sequences` builds a
process-pool payload (or uses :data:`~mqt.yaqs.simulator.WORKER_CTX`), then dispatches to
:func:`~mqt.yaqs.simulator.run_backend_parallel` or runs workers serially. Rollout types live in
:mod:`mqt.yaqs.characterization.process_tensors.surrogates.data`; the model is
:class:`~mqt.yaqs.characterization.process_tensors.surrogates.model.TransformerComb`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1) STANDARD LIBRARY
# ---------------------------------------------------------------------------
import copy
from typing import TYPE_CHECKING, Any, cast

# ---------------------------------------------------------------------------
# 2) THIRD PARTY
# ---------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 3) LOCAL — core / simulator
# ---------------------------------------------------------------------------
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.simulator import WORKER_CTX, available_cpus, run_backend_parallel

if TYPE_CHECKING:
    from collections.abc import Callable

    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

# ---------------------------------------------------------------------------
# 4) LOCAL — tomography surrogate stack
# ---------------------------------------------------------------------------
from ..core.encoding import normalize_rho_from_backend_output, pack_rho8
from ..core.utils import (
    _apply_backend_unitary_site_zero,
    _evolve_backend_state,
    _get_rho_site_zero,
    _reprepare_backend_state_forced,
)
from .data import SequenceRolloutSample, stack_rollouts
from .model import TransformerComb
from .utils import _random_density_matrix, _sample_random_intervention_sequence, build_initial_psi


# ---------------------------------------------------------------------------
# 4b) Performance helpers (hot path)
# ---------------------------------------------------------------------------
def _get_times_cached(times_cache: dict[tuple[float, int], np.ndarray], *, dt: float, duration: float) -> np.ndarray:
    """Return a cached time grid for a step.

    Args:
        times_cache: Cache mapping ``(dt, n_steps)`` to time grids.
        dt: Integration step size.
        duration: Desired evolution duration.

    Returns:
        A 1D float array suitable for ``AnalogSimParams.times``.
    """
    n_steps = max(1, int(np.round(float(duration) / float(dt))))
    key = (float(dt), int(n_steps))
    out = times_cache.get(key)
    if out is None:
        out = np.linspace(0.0, float(n_steps) * float(dt), int(n_steps) + 1)
        times_cache[key] = out
    return out


# ---------------------------------------------------------------------------
# 5) PARALLEL JOB PAYLOAD
# ---------------------------------------------------------------------------
# ``_simulate_sequences`` passes the following dict to ``run_backend_parallel`` (initializer →
# ``WORKER_CTX``) or directly to workers in the serial path. Keys must stay stable for pickling.
#
#   psi_pairs               list[list[step]] per sequence where step is either
#                           (meas, prep) or {"type": "unitary", "U": (2x2)}
#   initial_psi             list of MPS initial states (one per sequence)
#   num_trajectories        MCWF trajectory index split (1 when noise_model is None)
#   operator, sim_params    Hamiltonian MPO and analog parameters
#   timesteps               comb: ``k+1`` evolution segments per sequence (``U_1`` … ``U_{k+1}``)
#   timesteps_rows          optional per-sequence durations, each of length ``k+1``
#   operators_list          optional per-sequence MPOs, length ``k+1`` (one per evolution slot)
#   noise_model             None here (deterministic surrogate rollouts)
#   mcwf_static_ctx         static context forwarded to the backend for the whole sequence
#   mcwf_static_ctx_list    optional per-evolution-slot static context (length ``k+1``)
#   e_features_rows         list of (k, d_e) float32 Choi feature rows (rollout path only)


# ---------------------------------------------------------------------------
# 5b) Comb schedule — ``k`` instruments, ``k+1`` free evolutions (``U_1 … U_{k+1}``)
# ---------------------------------------------------------------------------
def _validate_comb_sequence_inputs(
    *,
    psi_pairs_list: list[list[Any]],
    timesteps: list[float],
    timesteps_rows: list[list[float]] | None,
    operators_list: list[list[MPO]] | None,
    static_ctx_list: list[list[Any]] | None,
) -> None:
    """Require compatible lengths for the process-tensor / comb convention."""
    num_sequences = len(psi_pairs_list)
    if timesteps_rows is None:
        ks = [len(p) for p in psi_pairs_list]
        if len(set(ks)) != 1:
            msg = "All sequences must share the same k when `timesteps_rows` is omitted."
            raise ValueError(msg)
        k0 = ks[0]
        if len(timesteps) != k0 + 1:
            msg = (
                "Comb schedule: `timesteps` must have length k+1 "
                f"({k0 + 1} for k={k0} intervention steps), got {len(timesteps)}."
            )
            raise ValueError(msg)
    else:
        if len(timesteps_rows) != num_sequences:
            msg = "`timesteps_rows` length must match number of sequences."
            raise ValueError(msg)
        for i, pairs in enumerate(psi_pairs_list):
            k = len(pairs)
            if len(timesteps_rows[i]) != k + 1:
                msg = (
                    f"Sequence {i}: `timesteps_rows[{i}]` must have length k+1={k + 1}, "
                    f"got {len(timesteps_rows[i])}."
                )
                raise ValueError(msg)
    if operators_list is not None:
        if len(operators_list) != num_sequences:
            msg = "`operators_list` length must match number of sequences."
            raise ValueError(msg)
        for i, pairs in enumerate(psi_pairs_list):
            k = len(pairs)
            if len(operators_list[i]) != k + 1:
                msg = (
                    f"Sequence {i}: `operators_list[{i}]` must have length k+1={k + 1}, "
                    f"got {len(operators_list[i])}."
                )
                raise ValueError(msg)
    if static_ctx_list is not None:
        if len(static_ctx_list) != num_sequences:
            msg = "`static_ctx_list` length must match number of sequences."
            raise ValueError(msg)
        for i, pairs in enumerate(psi_pairs_list):
            k = len(pairs)
            if len(static_ctx_list[i]) != k + 1:
                msg = (
                    f"Sequence {i}: `static_ctx_list[{i}]` must have length k+1={k + 1}, "
                    f"got {len(static_ctx_list[i])}."
                )
                raise ValueError(msg)


def _comb_durations_ops_ctx(
    *,
    sequence_idx: int,
    k: int,
    timesteps: list[float],
    timesteps_rows: list[list[float]] | None,
    hamiltonian: MPO,
    operators_list: list[list[MPO]] | None,
    mcwf_static_ctx: Any,
    mcwf_static_ctx_list: list[list[Any]] | None,
) -> tuple[list[float], list[MPO], list[Any]]:
    if timesteps_rows is not None:
        durs = [float(timesteps_rows[sequence_idx][i]) for i in range(k + 1)]
    else:
        durs = [float(timesteps[i]) for i in range(k + 1)]
    ops: list[MPO] = []
    ctxs: list[Any] = []
    for i in range(k + 1):
        op = hamiltonian if operators_list is None else operators_list[sequence_idx][i]
        ctx = mcwf_static_ctx if mcwf_static_ctx_list is None else mcwf_static_ctx_list[sequence_idx][i]
        ops.append(op)
        ctxs.append(ctx)
    return durs, ops, ctxs


# ---------------------------------------------------------------------------
# 6) WORKERS — one job index → one sequence (see ``num_trajectories`` for MCWF)
# ---------------------------------------------------------------------------
def _final_state_rollout_core(
    *,
    sequence_idx: int,
    trajectory_idx: int,
    worker_ctx: dict[str, Any],
    collect_trace: bool,
) -> tuple[np.ndarray, float, dict[str, Any] | None]:
    """Shared comb rollout: ``U_1`` then ``k`` times (reprepare → ``U``). Optionally return diagnostic trace."""
    psi_pairs = worker_ctx["psi_pairs"][sequence_idx]
    hamiltonian = worker_ctx["operator"]
    sim_params = worker_ctx["sim_params"]
    timesteps: list[float] = worker_ctx["timesteps"]
    timesteps_rows: list[list[float]] | None = worker_ctx.get("timesteps_rows")
    operators_list: list[list[MPO]] | None = worker_ctx.get("operators_list")
    mcwf_ctx_list: list[list[Any]] | None = worker_ctx.get("mcwf_static_ctx_list")
    noise_model = worker_ctx["noise_model"]
    initial_states: list[np.ndarray] = worker_ctx["initial_psi"]

    if noise_model is None:
        assert int(worker_ctx["num_trajectories"]) == 1, "num_trajectories must be 1 when noise_model is None."

    solver = sim_params.solver
    state = np.asarray(initial_states[sequence_idx], dtype=np.complex128).copy()
    times_cache: dict[tuple[float, int], np.ndarray] = worker_ctx.setdefault("_times_cache", {})
    step_params = copy.copy(sim_params)
    step_params.num_traj = 1
    step_params.show_progress = False
    step_params.get_state = True

    k = len(psi_pairs)
    durs, ops, mcwf_ctxs = _comb_durations_ops_ctx(
        sequence_idx=sequence_idx,
        k=k,
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
    if solver != "MCWF":
        step_params.output_state = None
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

    for step_idx, step in enumerate(psi_pairs):
        if isinstance(step, dict) and str(step.get("type", "")).lower() == "unitary":
            u = np.asarray(step["U"], dtype=np.complex128).reshape(2, 2)
            state = _apply_backend_unitary_site_zero(state, u, solver)
            sp = 1.0
        else:
            psi_meas, psi_prep = step
            state, step_prob = _reprepare_backend_state_forced(state, psi_meas, psi_prep, solver)
            sp = float(step_prob)
        step_probs.append(sp)
        prob_skipped_renormalize.append(sp <= 1e-15)
        cumulative_weight *= sp
        if cumulative_weight < 1e-15:
            break_step = step_idx
            break

        duration = float(durs[step_idx + 1])
        step_params.elapsed_time = duration
        step_params.times = _get_times_cached(times_cache, dt=float(step_params.dt), duration=duration)
        if solver != "MCWF":
            step_params.output_state = None

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

    rho_final = _get_rho_site_zero(state)
    wfin = float(cumulative_weight)

    trace: dict[str, Any] | None = None
    if collect_trace:
        terminated_early = break_step is not None or num_evolutions_in_loop < k
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


def _surrogate_final_state_worker(
    job_idx: int,
    job_payload: dict[str, Any] | None = None,
) -> tuple[int, int, np.ndarray, float]:
    """Simulate one intervention sequence and return the final reduced state.

    Does not record per-step states (cheaper than :func:`_surrogate_rollout_worker`).

    Args:
        job_idx: Flat index ``sequence_index * num_trajectories + trajectory_index``.
        job_payload: Per-pool shared context; defaults to :data:`~mqt.yaqs.simulator.WORKER_CTX`.

    Returns:
        ``(sequence_index, trajectory_index, rho_final_site0, cumulative_weight)`` where
        ``rho_final_site0`` is the reduced state on site 0 after the last evolution segment ``U_{k+1}``
        (comb schedule: ``k`` instruments, ``k+1`` evolutions).
    """
    worker_ctx = job_payload if job_payload is not None else WORKER_CTX
    num_trajectories: int = int(worker_ctx["num_trajectories"])
    sequence_idx = int(job_idx // num_trajectories)
    trajectory_idx = int(job_idx % num_trajectories)

    rho_final, cum_w, _trace = _final_state_rollout_core(
        sequence_idx=sequence_idx,
        trajectory_idx=trajectory_idx,
        worker_ctx=worker_ctx,
        collect_trace=False,
    )
    rho_final = cast("np.ndarray", rho_final)
    return (sequence_idx, trajectory_idx, rho_final, float(cum_w))


def _surrogate_final_state_worker_diagnostics(
    job_idx: int,
    job_payload: dict[str, Any] | None = None,
) -> tuple[int, int, np.ndarray, float, dict[str, Any]]:
    """Same as :func:`_surrogate_final_state_worker` but includes per-sequence rollout diagnostics."""
    worker_ctx = job_payload if job_payload is not None else WORKER_CTX
    num_trajectories: int = int(worker_ctx["num_trajectories"])
    sequence_idx = int(job_idx // num_trajectories)
    trajectory_idx = int(job_idx % num_trajectories)

    rho_final, cum_w, trace = _final_state_rollout_core(
        sequence_idx=sequence_idx,
        trajectory_idx=trajectory_idx,
        worker_ctx=worker_ctx,
        collect_trace=True,
    )
    if trace is None:
        raise RuntimeError("internal: diagnostics trace missing")
    rho_final = cast("np.ndarray", rho_final)
    return (sequence_idx, trajectory_idx, rho_final, float(cum_w), trace)


def _surrogate_rollout_worker(
    job_idx: int,
    job_payload: dict[str, Any] | None = None,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, float]:
    """Simulate one sequence and record per-step reduced states (packed).

    Choi feature rows (``e_features_rows``) are passed through unchanged into the sample; the worker
    only ensures shape ``(num_steps, d_e)`` matches the simulation.

    Args:
        job_idx: Flat index ``sequence_index * num_trajectories + trajectory_index``.
        job_payload: Shared pool context; defaults to :data:`~mqt.yaqs.simulator.WORKER_CTX`.

    Returns:
        ``(sequence_index, trajectory_index, rho0_packed, choi_features_matrix, rho_seq_packed, weight)`` where
        ``choi_features_matrix`` is ``(num_steps, d_e)`` and ``rho_seq_packed`` is ``(num_steps, 8)``.
        Here ``rho0_packed`` is the reduced state on site 0 **after** the first free evolution ``U_1`` and
        **before** the first instrument (comb boundary), matching the process-tensor slicing convention.
    """
    worker_ctx = job_payload if job_payload is not None else WORKER_CTX
    num_trajectories: int = int(worker_ctx["num_trajectories"])
    sequence_idx = int(job_idx // num_trajectories)
    trajectory_idx = int(job_idx % num_trajectories)

    psi_pairs = worker_ctx["psi_pairs"][sequence_idx]
    per_sequence_choi_rows: list[np.ndarray] | None = worker_ctx.get("e_features_rows")
    hamiltonian = worker_ctx["operator"]
    sim_params = worker_ctx["sim_params"]
    timesteps: list[float] = worker_ctx["timesteps"]
    timesteps_per_sequence = worker_ctx.get("timesteps_rows")
    hamiltonians_per_step = worker_ctx.get("operators_list")
    mcwf_ctx_per_step = worker_ctx.get("mcwf_static_ctx_list")
    noise_model = worker_ctx["noise_model"]
    initial_states: list[np.ndarray] = worker_ctx["initial_psi"]

    if noise_model is None:
        assert num_trajectories == 1, "num_trajectories must be 1 when noise_model is None."

    num_steps = len(psi_pairs)
    solver = sim_params.solver
    state = np.asarray(initial_states[sequence_idx], dtype=np.complex128).copy()
    times_cache: dict[tuple[float, int], np.ndarray] = worker_ctx.setdefault("_times_cache", {})
    step_params = copy.copy(sim_params)
    step_params.num_traj = 1
    step_params.show_progress = False
    step_params.get_state = True

    if per_sequence_choi_rows is None:
        msg = "Rollout worker requires `e_features_rows`: per-sequence Choi feature rows."
        raise ValueError(msg)
    choi_features_matrix = np.asarray(per_sequence_choi_rows[sequence_idx], dtype=np.float32).reshape(num_steps, -1)

    durs, ops, mcwf_ctxs = _comb_durations_ops_ctx(
        sequence_idx=sequence_idx,
        k=num_steps,
        timesteps=timesteps,
        timesteps_rows=timesteps_per_sequence,
        hamiltonian=hamiltonian,
        operators_list=hamiltonians_per_step,
        mcwf_static_ctx=worker_ctx.get("mcwf_static_ctx"),
        mcwf_static_ctx_list=mcwf_ctx_per_step,
    )

    # U_1: reduced state immediately before the first instrument (comb boundary).
    duration = float(durs[0])
    step_params.elapsed_time = duration
    step_params.times = _get_times_cached(times_cache, dt=float(step_params.dt), duration=duration)
    if solver != "MCWF":
        step_params.output_state = None
    state = _evolve_backend_state(
        state,
        ops[0],
        noise_model,
        step_params,
        solver,
        traj_idx=trajectory_idx,
        static_ctx=mcwf_ctxs[0],
    )

    rho0_raw = _get_rho_site_zero(state)
    rho0_packed = pack_rho8(normalize_rho_from_backend_output(rho0_raw)).astype(np.float32)

    cumulative_weight = 1.0
    last_rho_packed = rho0_packed.copy()
    rho_sequence_packed = np.empty((num_steps, 8), dtype=np.float32)
    out_i = 0

    for step_idx, step in enumerate(psi_pairs):
        if isinstance(step, dict) and str(step.get("type", "")).lower() == "unitary":
            u = np.asarray(step["U"], dtype=np.complex128).reshape(2, 2)
            state = _apply_backend_unitary_site_zero(state, u, solver)
            step_prob = 1.0
        else:
            psi_meas, psi_prep = step
            state, step_prob = _reprepare_backend_state_forced(state, psi_meas, psi_prep, solver)
        cumulative_weight *= float(step_prob)
        if cumulative_weight < 1e-15:
            if out_i < num_steps:
                rho_sequence_packed[out_i:, :] = last_rho_packed[None, :]
            out_i = num_steps
            break

        duration = float(durs[step_idx + 1])
        step_params.elapsed_time = duration
        step_params.times = _get_times_cached(times_cache, dt=float(step_params.dt), duration=duration)
        if solver != "MCWF":
            step_params.output_state = None

        state = _evolve_backend_state(
            state,
            ops[step_idx + 1],
            noise_model,
            step_params,
            solver,
            traj_idx=trajectory_idx,
            static_ctx=mcwf_ctxs[step_idx + 1],
        )

        rho_step = _get_rho_site_zero(state)
        rho_normalized = normalize_rho_from_backend_output(rho_step)
        last_rho_packed = pack_rho8(rho_normalized).astype(np.float32)
        rho_sequence_packed[out_i, :] = last_rho_packed
        out_i += 1

    if out_i < num_steps:
        rho_sequence_packed[out_i:, :] = last_rho_packed[None, :]
    if choi_features_matrix.shape[0] != num_steps:
        msg = "Choi feature rows must have length num_steps matching intervention steps."
        raise ValueError(msg)
    return (
        sequence_idx,
        trajectory_idx,
        rho0_packed,
        choi_features_matrix,
        rho_sequence_packed,
        float(cumulative_weight),
    )


# ---------------------------------------------------------------------------
# 7) SERIAL BACKEND WRAPPER — same thread cap pattern as :func:`mqt.yaqs.simulator._call_backend`
# ---------------------------------------------------------------------------
def _call_worker_serial(worker_fn: Callable[..., Any], *args: Any) -> Any:
    """Run one worker call under BLAS thread limits if available.

    Args:
        worker_fn: Worker function to call.
        *args: Positional arguments forwarded to ``worker_fn``.

    Returns:
        The return value of ``worker_fn(*args)``.
    """
    import contextlib  # noqa: PLC0415

    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        return worker_fn(*args)
    with contextlib.suppress(Exception), threadpool_limits(limits=1):
        return worker_fn(*args)


# ---------------------------------------------------------------------------
# 8) _simulate_sequences — internal primitive (feeds :func:`generate_data`)
# ---------------------------------------------------------------------------
def _simulate_sequences(
    *,
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    psi_pairs_list: list[list[Any]],
    initial_psis: list[np.ndarray],
    static_ctx: Any,
    parallel: bool = True,
    show_progress: bool = True,
    record_step_states: bool = True,
    e_features_rows: list[np.ndarray] | None = None,
    timesteps_rows: list[list[float]] | None = None,
    operators_list: list[list[MPO]] | None = None,
    static_ctx_list: list[list[Any]] | None = None,
    context_vec: np.ndarray | None = None,
) -> list[SequenceRolloutSample] | np.ndarray:
    """Simulate many intervention sequences.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        timesteps: Comb schedule: ``k+1`` evolution durations per sequence (``U_1`` … ``U_{k+1}``)
            when ``timesteps_rows`` is omitted (shared across sequences; all must have the same ``k``).
        psi_pairs_list: One list of ``k`` intervention steps per sequence (MP tuples or unitary-step dicts).
        initial_psis: One initial state vector per sequence.
        static_ctx: Optional static backend context (used for MCWF preprocessing).
        parallel: Whether to use process-based parallelism over sequences.
        show_progress: Whether to show a progress bar.
        record_step_states: If ``True``, record and return per-step packed reduced states.
        e_features_rows: Per-sequence Choi feature rows (required when ``record_step_states=True``).
        timesteps_rows: Optional per-sequence durations, each of length ``k+1``.
        operators_list: Optional per-sequence Hamiltonians, length ``k+1`` per sequence.
        static_ctx_list: Optional per-sequence MCWF contexts, length ``k+1`` per sequence.
        context_vec: Optional static context vector to attach to each sample.

    Returns:
        If ``record_step_states=True``: list of :class:`~mqt.yaqs.characterization.process_tensors.surrogates.data.SequenceRolloutSample`.
        Otherwise: float32 array of shape ``(N, 8)`` with final packed reduced states.

    Raises:
        ValueError: If input lengths are inconsistent or if required feature rows are missing.
        RuntimeError: If parallel execution returns incomplete results.
    """
    num_sequences = len(initial_psis)
    if len(psi_pairs_list) != num_sequences:
        msg = "psi_pairs_list and initial_psis must have equal length."
        raise ValueError(msg)

    if record_step_states:
        if e_features_rows is None:
            msg = "record_step_states=True requires e_features_rows (per-sequence Choi feature rows)."
            raise ValueError(msg)
        if len(e_features_rows) != num_sequences:
            msg = "e_features_rows length must match initial_psis."
            raise ValueError(msg)
    elif e_features_rows is not None:
        msg = "e_features_rows is only used when record_step_states=True."
        raise ValueError(msg)

    _validate_comb_sequence_inputs(
        psi_pairs_list=psi_pairs_list,
        timesteps=timesteps,
        timesteps_rows=timesteps_rows,
        operators_list=operators_list,
        static_ctx_list=static_ctx_list,
    )

    job_payload: dict[str, Any] = {
        "psi_pairs": psi_pairs_list,
        "initial_psi": initial_psis,
        "num_trajectories": 1,
        "operator": operator,
        "sim_params": sim_params,
        "timesteps": timesteps,
        "timesteps_rows": timesteps_rows,
        "operators_list": operators_list,
        "noise_model": None,
        "mcwf_static_ctx": static_ctx,
        "mcwf_static_ctx_list": static_ctx_list,
        "_times_cache": {},
    }
    if record_step_states:
        job_payload["e_features_rows"] = e_features_rows

    if not record_step_states:
        if parallel and num_sequences > 1:
            max_workers = max(1, available_cpus() - 1)
            parallel_results = run_backend_parallel(
                worker_fn=_surrogate_final_state_worker,
                payload=job_payload,
                n_jobs=num_sequences,
                max_workers=max_workers,
                show_progress=bool(show_progress),
                desc="Simulating sequences (final states)",
            )
            final_packed_by_index: list[np.ndarray | None] = [None] * num_sequences
            for _job_idx, worker_out in parallel_results:
                sequence_idx, _traj_idx, rho_final, _weight = worker_out
                rho_norm = normalize_rho_from_backend_output(rho_final)
                final_packed_by_index[sequence_idx] = pack_rho8(rho_norm)
            if any(x is None for x in final_packed_by_index):
                msg = "Parallel sequence simulation incomplete."
                raise RuntimeError(msg)
            stacked_final = [cast("np.ndarray", x) for x in final_packed_by_index]
            return np.stack(stacked_final, axis=0).astype(np.float32)

        final_packed_serial: list[np.ndarray] = []
        for seq_idx in tqdm(
            range(num_sequences),
            desc="Simulating sequences (final states)",
            disable=(not bool(show_progress)),
            ncols=80,
        ):
            _s, _t, rho_final, _w = _call_worker_serial(_surrogate_final_state_worker, seq_idx, job_payload)
            rho_norm = normalize_rho_from_backend_output(rho_final)
            final_packed_serial.append(pack_rho8(rho_norm))
        return np.stack(final_packed_serial, axis=0).astype(np.float32)

    optional_context_vec = None if context_vec is None else np.asarray(context_vec, dtype=np.float32).reshape(-1)

    def rollout_one_sequence(sequence_idx: int) -> SequenceRolloutSample:
        _s, _t, rho0, choi_mat, rho_seq, weight = _call_worker_serial(
            _surrogate_rollout_worker, sequence_idx, job_payload
        )
        return SequenceRolloutSample(
            rho_0=rho0,
            E_features=choi_mat,
            rho_seq=rho_seq,
            context=None if optional_context_vec is None else optional_context_vec.copy(),
            weight=float(weight),
        )

    if parallel and num_sequences > 1:
        max_workers = max(1, available_cpus() - 1)
        parallel_results = run_backend_parallel(
            worker_fn=_surrogate_rollout_worker,
            payload=job_payload,
            n_jobs=num_sequences,
            max_workers=max_workers,
            show_progress=bool(show_progress),
            desc="Simulating sequences (rollouts)",
        )
        samples_by_index: list[SequenceRolloutSample | None] = [None] * num_sequences
        for _job_idx, worker_out in parallel_results:
            sequence_idx, _t, rho0, choi_mat, rho_seq, weight = worker_out
            samples_by_index[sequence_idx] = SequenceRolloutSample(
                rho_0=rho0,
                E_features=choi_mat,
                rho_seq=rho_seq,
                context=None if optional_context_vec is None else optional_context_vec.copy(),
                weight=float(weight),
            )
        if any(s is None for s in samples_by_index):
            msg = "Parallel sequence rollout simulation incomplete."
            raise RuntimeError(msg)
        return [cast("SequenceRolloutSample", s) for s in samples_by_index]

    return [rollout_one_sequence(j) for j in range(num_sequences)]


def simulate_final_states_with_diagnostics(
    *,
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    psi_pairs_list: list[list[Any]],
    initial_psis: list[np.ndarray],
    static_ctx: Any,
    parallel: bool = True,
    show_progress: bool = True,
    timesteps_rows: list[list[float]] | None = None,
    operators_list: list[list[MPO]] | None = None,
    static_ctx_list: list[list[Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Like final-state :func:`_simulate_sequences` (``record_step_states=False``) but returns per-sequence diagnostics.

    Does not change physics; only threads rollout metadata from :func:`_surrogate_final_state_worker_diagnostics`.

    Returns:
        ``(final_packed, traces)`` where ``final_packed`` is ``(N, 8)`` float32 and ``traces[i]`` matches
        sequence ``i`` (same order as ``psi_pairs_list``).
    """
    num_sequences = len(initial_psis)
    if len(psi_pairs_list) != num_sequences:
        msg = "psi_pairs_list and initial_psis must have equal length."
        raise ValueError(msg)

    _validate_comb_sequence_inputs(
        psi_pairs_list=psi_pairs_list,
        timesteps=timesteps,
        timesteps_rows=timesteps_rows,
        operators_list=operators_list,
        static_ctx_list=static_ctx_list,
    )

    job_payload: dict[str, Any] = {
        "psi_pairs": psi_pairs_list,
        "initial_psi": initial_psis,
        "num_trajectories": 1,
        "operator": operator,
        "sim_params": sim_params,
        "timesteps": timesteps,
        "timesteps_rows": timesteps_rows,
        "operators_list": operators_list,
        "noise_model": None,
        "mcwf_static_ctx": static_ctx,
        "mcwf_static_ctx_list": static_ctx_list,
        "_times_cache": {},
    }

    traces_ordered: list[dict[str, Any] | None] = [None] * num_sequences

    if parallel and num_sequences > 1:
        max_workers = max(1, available_cpus() - 1)
        parallel_results = run_backend_parallel(
            worker_fn=_surrogate_final_state_worker_diagnostics,
            payload=job_payload,
            n_jobs=num_sequences,
            max_workers=max_workers,
            show_progress=bool(show_progress),
            desc="Simulating sequences (final states + diagnostics)",
        )
        final_packed_by_index: list[np.ndarray | None] = [None] * num_sequences
        for _job_idx, worker_out in parallel_results:
            sequence_idx, _traj_idx, rho_final, _weight, trace = worker_out
            rho_norm = normalize_rho_from_backend_output(rho_final)
            final_packed_by_index[sequence_idx] = pack_rho8(rho_norm)
            traces_ordered[sequence_idx] = trace
        if any(x is None for x in final_packed_by_index) or any(t is None for t in traces_ordered):
            msg = "Parallel sequence simulation incomplete."
            raise RuntimeError(msg)
        stacked_final = [cast("np.ndarray", x) for x in final_packed_by_index]
        return np.stack(stacked_final, axis=0).astype(np.float32), cast("list[dict[str, Any]]", traces_ordered)

    final_packed_serial: list[np.ndarray] = []
    for seq_idx in tqdm(
        range(num_sequences),
        desc="Simulating sequences (final states + diagnostics)",
        disable=(not bool(show_progress)),
        ncols=80,
    ):
        _s, _t, rho_final, _w, trace = _call_worker_serial(
            _surrogate_final_state_worker_diagnostics, seq_idx, job_payload
        )
        rho_norm = normalize_rho_from_backend_output(rho_final)
        final_packed_serial.append(pack_rho8(rho_norm))
        traces_ordered[seq_idx] = trace
    return np.stack(final_packed_serial, axis=0).astype(np.float32), cast("list[dict[str, Any]]", traces_ordered)


# ---------------------------------------------------------------------------
# 9) Helpers — used only by :func:`generate_data`
# ---------------------------------------------------------------------------
def _psi_from_rank1_projector(projector: np.ndarray) -> np.ndarray:
    """Extract a unit ket from a 2x2 rank-1 projector.

    Args:
        projector: 2x2 Hermitian projector/density matrix.

    Returns:
        Normalized eigenvector corresponding to the largest eigenvalue.
    """
    eigvals, eigvecs = np.linalg.eigh(np.asarray(projector, dtype=np.complex128).reshape(2, 2))
    idx = int(np.argmax(eigvals.real))
    psi = eigvecs[:, idx]
    norm = float(np.linalg.norm(psi))
    if norm < 1e-15:
        return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return (psi / norm).astype(np.complex128)


# ---------------------------------------------------------------------------
# 10) Public — :func:`generate_data`
# ---------------------------------------------------------------------------
def _rollout_arrays_to_tensor_dataset(
    rho0: np.ndarray,
    E_features: np.ndarray,
    rho_seq: np.ndarray,
):
    """Convert rollout arrays into a PyTorch TensorDataset.

    Args:
        rho0: Array of shape ``(N, 8)``.
        E_features: Array of shape ``(N, K, d_e)``.
        rho_seq: Array of shape ``(N, K, 8)``.

    Returns:
        TensorDataset with tensors ``(E_features, rho0, rho_seq)`` in that order.
    """
    import torch  # noqa: PLC0415
    from torch.utils.data import TensorDataset  # noqa: PLC0415

    return TensorDataset(
        torch.as_tensor(E_features, dtype=torch.float32),
        torch.as_tensor(rho0, dtype=torch.float32),
        torch.as_tensor(rho_seq, dtype=torch.float32),
    )


def generate_data(
    operator: MPO,
    sim_params: AnalogSimParams,
    *,
    k: int,
    n: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    parallel: bool = True,
    show_progress: bool = True,
    timesteps: list[float] | None = None,
    init_mode: str = "eigenstate",
):
    """Generate surrogate training data by sampling interventions and simulating rollouts.

    Args:
        operator: Hamiltonian MPO. The chain length is inferred from ``operator.length``.
        sim_params: Analog simulation parameters.
        k: Number of intervention steps.
        n: Number of sampled sequences.
        rng: Optional RNG (overrides ``seed`` if provided).
        seed: Optional seed used to create a default RNG.
        parallel: Whether to parallelize over sequences.
        show_progress: Whether to show progress bars.
        timesteps: Optional comb evolution durations (defaults to ``[sim_params.dt] * (k+1)``).
        init_mode: Initial-state sampling mode (see :func:`build_initial_psi`).

    Returns:
        A :class:`~torch.utils.data.TensorDataset` with tensors ``(E_features, rho0, rho_seq)``.

    Raises:
        ValueError: If ``timesteps`` has the wrong length (must be ``k+1``).
    """
    chain_length = int(operator.length)

    static_ctx: Any | None = None
    if getattr(sim_params, "solver", None) == "MCWF":
        # Convenience for users: MCWF backend benefits from preprocessing.
        # When no noise model is provided, default to noiseless.
        from ..core.utils import make_mcwf_static_context  # noqa: PLC0415

        static_ctx = make_mcwf_static_context(operator, sim_params, noise_model=None)

    if rng is None:
        rng = np.random.default_rng(0 if seed is None else int(seed))
    if timesteps is None:
        timesteps = [float(sim_params.dt)] * (int(k) + 1)
    if len(timesteps) != int(k) + 1:
        msg = f"Comb schedule: timesteps length must be k+1={int(k) + 1}, got {len(timesteps)}."
        raise ValueError(msg)

    psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]] = []
    initial_psis: list[np.ndarray] = []
    choi_feature_rows_per_sequence: list[np.ndarray] = []

    for _ in range(int(n)):
        rho_in = _random_density_matrix(rng)
        intervention_maps, choi_rows = _sample_random_intervention_sequence(int(k), rng)
        step_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for emap in intervention_maps:
            rho_prep = np.asarray(emap.rho_prep, dtype=np.complex128)
            effect_dm = np.asarray(emap.effect, dtype=np.complex128)
            psi_meas = _psi_from_rank1_projector(effect_dm)
            psi_prep = _psi_from_rank1_projector(rho_prep)
            step_pairs.append((psi_meas, psi_prep))
        psi_pairs_list.append(step_pairs)
        choi_feature_rows_per_sequence.append(choi_rows.astype(np.float32))
        initial_psis.append(build_initial_psi(rho_in, length=int(chain_length), rng=rng, init_mode=init_mode))

    samples = _simulate_sequences(
        operator=operator,
        sim_params=sim_params,
        timesteps=timesteps,
        psi_pairs_list=psi_pairs_list,
        initial_psis=initial_psis,
        e_features_rows=choi_feature_rows_per_sequence,
        parallel=bool(parallel),
        show_progress=bool(show_progress),
        record_step_states=True,
        static_ctx=static_ctx,
        context_vec=None,
    )
    assert isinstance(samples, list)
    rho0_batch, features_batch, rho_seq_batch, _ctx = stack_rollouts(samples)
    return _rollout_arrays_to_tensor_dataset(rho0_batch, features_batch, rho_seq_batch)


# ---------------------------------------------------------------------------
# 11) Public — :func:`create_surrogate`
# ---------------------------------------------------------------------------
def create_surrogate(
    operator: MPO,
    sim_params: AnalogSimParams,
    *,
    k: int,
    n: int,
    seed: int | None = None,
    parallel: bool = True,
    show_progress: bool = True,
    timesteps: list[float] | None = None,
    init_mode: str = "eigenstate",
    model_kwargs: dict[str, Any] | None = None,
    train_kwargs: dict[str, Any] | None = None,
) -> TransformerComb:
    """Train a surrogate model end-to-end on sampled rollout data.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        k: Number of intervention steps.
        n: Number of sampled sequences.
        seed: Seed used for data generation RNG.
        parallel: Whether to parallelize data generation.
        show_progress: Whether to show progress bars.
        timesteps: Optional per-step durations passed to :func:`generate_data`.
        init_mode: Initial-state sampling mode passed to :func:`generate_data`.
        model_kwargs: Optional keyword arguments forwarded to :class:`TransformerComb`.
        train_kwargs: Optional keyword arguments forwarded to :meth:`TransformerComb.fit`.

    Returns:
        Trained :class:`TransformerComb`.
    """
    import torch  # noqa: PLC0415

    rng = np.random.default_rng(0 if seed is None else int(seed))
    train_data = generate_data(
        operator,
        sim_params,
        k=int(k),
        n=int(n),
        rng=rng,
        parallel=bool(parallel),
        show_progress=bool(show_progress),
        timesteps=timesteps,
        init_mode=init_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    d_e = int(train_data.tensors[0].shape[-1])
    model = TransformerComb(d_e=d_e, d_rho=8, **resolved_model_kwargs).to(device)

    resolved_train_kwargs = {} if train_kwargs is None else dict(train_kwargs)
    model.fit(train_data, device=device, **resolved_train_kwargs)
    return model


__all__ = ["create_surrogate", "generate_data"]
