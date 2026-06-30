# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Parallel process-tensor schedule sequence simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mqt.yaqs.core.parallel_utils import (
    ExecutionConfig,
    merge_execution_config,
    run_indexed_jobs,
)

from ...shared.encoding import normalize_backend_rho, pack_rho8
from ...shared.utils import StochasticSolver, resolve_stochastic_solver
from ..surrogates.data import SeqTrace
from .workers import (
    _seq_final_worker,
    _seq_final_worker_diagnostics,
    _seq_trace_worker,
    _validate_process_tensor_schedule_inputs,
)

if TYPE_CHECKING:
    from mqt.yaqs.analog.mcwf import MCWFContext
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


# ---------------------------------------------------------------------------
# simulate_sequences — parallel dispatch via run_indexed_jobs
# ---------------------------------------------------------------------------
def simulate_sequences(
    *,
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    intervention_steps_list: list[list[Any]],
    initial_psis: list[np.ndarray],
    static_ctx: MCWFContext | None,
    parallel: bool = True,
    show_progress: bool = True,
    record_step_states: bool = True,
    traced: bool = False,
    e_features_rows: list[np.ndarray] | None = None,
    timesteps_rows: list[list[float]] | None = None,
    operators_list: list[list[MPO]] | None = None,
    static_ctx_list: list[list[MCWFContext | None]] | None = None,
    context_vec: np.ndarray | None = None,
    solver: StochasticSolver | None = None,
    _execution: ExecutionConfig | None = None,
) -> list[SeqTrace] | np.ndarray | tuple[np.ndarray, list[dict[str, Any]]]:
    """Simulate many intervention sequences in parallel.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        timesteps: Process-tensor schedule: ``num_interventions+1`` evolution durations per
            sequence when ``timesteps_rows`` is omitted.
        intervention_steps_list: One list of ``num_interventions`` intervention steps per sequence.
        initial_psis: One initial state vector per sequence.
        static_ctx: Optional static backend context (MCWF preprocessing).
        parallel: Whether to use process-based parallelism over sequences.
        show_progress: Whether to show a progress bar.
        record_step_states: If ``True``, return per-step :class:`SeqTrace` records.
        traced: If ``True``, return final packed states and per-sequence diagnostics
            (incompatible with ``record_step_states=True``).
        e_features_rows: Per-sequence Choi feature rows (required when ``record_step_states=True``).
        timesteps_rows: Optional per-sequence durations, each of length ``num_interventions+1``.
        operators_list: Optional per-sequence Hamiltonians, length ``num_interventions+1`` per sequence.
        static_ctx_list: Optional per-sequence MCWF contexts, length ``num_interventions+1`` per sequence.
        context_vec: Optional static context vector attached to each trace when
            ``record_step_states=True``. Raises :class:`ValueError` when set while
            ``record_step_states=False``.
        solver: Optional stochastic solver override (``"MCWF"`` or ``"TJM"``).

    Returns:
        - ``record_step_states=True``: list of :class:`SeqTrace`
        - ``traced=True``: ``(final_packed, traces)`` with ``final_packed`` of shape ``(N, 8)``
        - otherwise: float32 array of shape ``(N, 8)`` with final packed reduced states

    Raises:
        ValueError: If input lengths are inconsistent or modes are incompatible.
        RuntimeError: If parallel execution returns incomplete results.
    """
    if traced and record_step_states:
        msg = "traced=True is incompatible with record_step_states=True."
        raise ValueError(msg)

    num_sequences = len(initial_psis)
    if len(intervention_steps_list) != num_sequences:
        msg = "intervention_steps_list and initial_psis must have equal length."
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

    if context_vec is not None and not record_step_states:
        msg = "context_vec is only used when record_step_states=True."
        raise ValueError(msg)

    _validate_process_tensor_schedule_inputs(
        intervention_steps_list=intervention_steps_list,
        timesteps=timesteps,
        timesteps_rows=timesteps_rows,
        operators_list=operators_list,
        static_ctx_list=static_ctx_list,
    )

    if num_sequences == 0:
        if traced:
            return np.zeros((0, 8), dtype=np.float32), []
        if record_step_states:
            return []
        return np.zeros((0, 8), dtype=np.float32)

    # Pickle-stable payload — schema documented in :mod:`.workers`.
    job_payload: dict[str, Any] = {
        "intervention_steps": intervention_steps_list,
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
        "solver": resolve_stochastic_solver(sim_params, solver=solver),
    }
    if record_step_states:
        job_payload["e_features_rows"] = e_features_rows

    exec_cfg = merge_execution_config(_execution, parallel=parallel, show_progress=show_progress)

    if traced:
        job_results = run_indexed_jobs(
            _seq_final_worker_diagnostics,
            payload=job_payload,
            n_jobs=num_sequences,
            config=exec_cfg,
            desc="Simulating sequences (final states + diagnostics)",
        )
        final_packed_by_index: list[np.ndarray | None] = [None] * num_sequences
        traces_ordered: list[dict[str, Any] | None] = [None] * num_sequences
        for worker_out in job_results.values():
            sequence_idx, _traj_idx, rho_final, _weight, trace = worker_out
            rho_norm = normalize_backend_rho(rho_final)
            final_packed_by_index[sequence_idx] = pack_rho8(rho_norm)
            traces_ordered[sequence_idx] = trace
        if any(x is None for x in final_packed_by_index) or any(t is None for t in traces_ordered):
            msg = "Parallel sequence simulation incomplete."
            raise RuntimeError(msg)
        stacked_final = [cast("np.ndarray", x) for x in final_packed_by_index]
        return np.stack(stacked_final, axis=0).astype(np.float32), cast("list[dict[str, Any]]", traces_ordered)

    if not record_step_states:
        job_results = run_indexed_jobs(
            _seq_final_worker,
            payload=job_payload,
            n_jobs=num_sequences,
            config=exec_cfg,
            desc="Simulating sequences (final states)",
        )
        final_packed_by_index: list[np.ndarray | None] = [None] * num_sequences
        for worker_out in job_results.values():
            sequence_idx, _traj_idx, rho_final, _weight = worker_out
            rho_norm = normalize_backend_rho(rho_final)
            final_packed_by_index[sequence_idx] = pack_rho8(rho_norm)
        if any(x is None for x in final_packed_by_index):
            msg = "Parallel sequence simulation incomplete."
            raise RuntimeError(msg)
        stacked_final = [cast("np.ndarray", x) for x in final_packed_by_index]
        return np.stack(stacked_final, axis=0).astype(np.float32)

    optional_context_vec = None if context_vec is None else np.asarray(context_vec, dtype=np.float32).reshape(-1)

    job_results = run_indexed_jobs(
        _seq_trace_worker,
        payload=job_payload,
        n_jobs=num_sequences,
        config=exec_cfg,
        desc="Simulating sequences (traces)",
    )
    samples_by_index: list[SeqTrace | None] = [None] * num_sequences
    for worker_out in job_results.values():
        sequence_idx, _t, rho0, choi_mat, rho_seq, weight = worker_out
        samples_by_index[sequence_idx] = SeqTrace(
            rho_0=rho0,
            E_features=choi_mat,
            rho_seq=rho_seq,
            context=None if optional_context_vec is None else optional_context_vec.copy(),
            weight=float(weight),
        )
    if any(s is None for s in samples_by_index):
        msg = "Parallel sequence trace simulation incomplete."
        raise RuntimeError(msg)
    return [cast("SeqTrace", s) for s in samples_by_index]
