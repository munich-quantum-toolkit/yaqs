# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Surrogate workflow: sample training data and train models.

**Public API** (see ``__all__``): :func:`sample_train_dataset`, :func:`train_surrogate_model`.

:func:`sample_train_dataset` returns a :class:`~torch.utils.data.TensorDataset` for
:meth:`~mqt.yaqs.characterization.memory.backends.surrogates.model.TransformerComb.fit`.

**Internals** — :func:`simulate_sequences` builds a process-pool payload (or uses
:data:`~mqt.yaqs.core.parallel_utils.WORKER_CTX`), then dispatches via
:func:`~mqt.yaqs.core.parallel_utils.run_indexed_jobs`. Pool workers live in
:mod:`.workers`; trace records are :class:`~mqt.yaqs.characterization.memory.backends.surrogates.data.SeqTrace`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mqt.yaqs.core.parallel_utils import (
    ExecutionConfig,
    merge_execution_config,
    run_indexed_jobs,
)

if TYPE_CHECKING:
    from torch.utils.data import TensorDataset

    from mqt.yaqs.analog.mcwf import MCWFContext
    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

# ---------------------------------------------------------------------------
# Imports — local surrogate stack
# ---------------------------------------------------------------------------
from ...shared.encoding import normalize_backend_rho, pack_rho8
from ...shared.utils import (
    StochasticSolver,
    make_mcwf_static_context,
    resolve_stochastic_solver,
)
from .data import SeqTrace, stack_traces
from .utils import sample_density_matrix, sample_initial_psi
from .workers import (
    _seq_final_worker,
    _seq_final_worker_diagnostics,
    _seq_trace_worker,
    _validate_comb_sequence_inputs,
)


# ---------------------------------------------------------------------------
# simulate_sequences — parallel dispatch via run_indexed_jobs
# ---------------------------------------------------------------------------
def simulate_sequences(
    *,
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    psi_pairs_list: list[list[Any]],
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
        timesteps: Comb schedule: ``k+1`` evolution durations per sequence when
            ``timesteps_rows`` is omitted.
        psi_pairs_list: One list of ``k`` intervention steps per sequence.
        initial_psis: One initial state vector per sequence.
        static_ctx: Optional static backend context (MCWF preprocessing).
        parallel: Whether to use process-based parallelism over sequences.
        show_progress: Whether to show a progress bar.
        record_step_states: If ``True``, return per-step :class:`SeqTrace` records.
        traced: If ``True``, return final packed states and per-sequence diagnostics
            (incompatible with ``record_step_states=True``).
        e_features_rows: Per-sequence Choi feature rows (required when ``record_step_states=True``).
        timesteps_rows: Optional per-sequence durations, each of length ``k+1``.
        operators_list: Optional per-sequence Hamiltonians, length ``k+1`` per sequence.
        static_ctx_list: Optional per-sequence MCWF contexts, length ``k+1`` per sequence.
        context_vec: Optional static context vector attached to each trace.
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

    # Pickle-stable payload — schema documented in :mod:`.workers`.
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


def simulate_traced(
    *,
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    psi_pairs_list: list[list[Any]],
    initial_psis: list[np.ndarray],
    static_ctx: MCWFContext | None,
    parallel: bool = True,
    show_progress: bool = True,
    timesteps_rows: list[list[float]] | None = None,
    operators_list: list[list[MPO]] | None = None,
    static_ctx_list: list[list[MCWFContext | None]] | None = None,
    solver: StochasticSolver | None = None,
    _execution: ExecutionConfig | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Simulate sequences and return final states plus per-sequence traces.

    Thin wrapper around :func:`simulate_sequences` with ``record_step_states=False, traced=True``.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        timesteps: Comb schedule evolution durations.
        psi_pairs_list: Intervention steps per sequence.
        initial_psis: Initial state vectors per sequence.
        static_ctx: Optional static MCWF context.
        parallel: Whether to parallelize over sequences.
        show_progress: Whether to show a progress bar.
        timesteps_rows: Optional per-sequence durations.
        operators_list: Optional per-sequence Hamiltonians.
        static_ctx_list: Optional per-sequence MCWF contexts.
        solver: Optional stochastic solver override.

    Returns:
        Tuple ``(final_packed, traces)`` where ``final_packed`` is ``(N, 8)`` float32.

    Raises:
        TypeError: If the traced simulation output has an unexpected type.
    """
    result = simulate_sequences(
        operator=operator,
        sim_params=sim_params,
        timesteps=timesteps,
        psi_pairs_list=psi_pairs_list,
        initial_psis=initial_psis,
        static_ctx=static_ctx,
        parallel=parallel,
        show_progress=show_progress,
        record_step_states=False,
        traced=True,
        timesteps_rows=timesteps_rows,
        operators_list=operators_list,
        static_ctx_list=static_ctx_list,
        solver=solver,
        _execution=_execution,
    )
    if not isinstance(result, tuple):
        msg = "Expected traced simulation output."
        raise TypeError(msg)
    return result


# ---------------------------------------------------------------------------
# Public API — training dataset and surrogate model
# ---------------------------------------------------------------------------
def pack_dataset(
    rho0: np.ndarray,
    e_features: np.ndarray,
    rho_seq: np.ndarray,
) -> TensorDataset:
    """Pack sequence trace arrays into a PyTorch :class:`~torch.utils.data.TensorDataset`.

    Args:
        rho0: Array of shape ``(N, 8)``.
        e_features: Array of shape ``(N, K, d_e)``.
        rho_seq: Array of shape ``(N, K, 8)``.

    Returns:
        TensorDataset with tensors ``(e_features, rho0, rho_seq)`` in that order.
    """
    import torch  # noqa: PLC0415
    from torch.utils.data import TensorDataset  # noqa: PLC0415

    return TensorDataset(
        torch.as_tensor(e_features, dtype=torch.float32),
        torch.as_tensor(rho0, dtype=torch.float32),
        torch.as_tensor(rho_seq, dtype=torch.float32),
    )


def sample_train_dataset(
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
    solver: StochasticSolver | None = None,
    interventions: str = "measure_prepare",
    _execution: ExecutionConfig | None = None,
) -> TensorDataset:
    """Sample intervention sequences and pack a surrogate training dataset.

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
        init_mode: Initial-state sampling mode (see :func:`sample_initial_psi`).
        solver: Optional stochastic solver override (``"MCWF"`` or ``"TJM"``).
        interventions: Training intervention kind (``"haar"``, ``"clifford"``, or ``"measure_prepare"``).

    Returns:
        A :class:`~torch.utils.data.TensorDataset` with tensors ``(E_features, rho0, rho_seq)``.

    Raises:
        ValueError: If ``timesteps`` has the wrong length (must be ``k+1``).
    """
    from mqt.yaqs.characterization.memory.operational_memory.interventions import (
        normalize_kind,
        sample_train_sequence,
    )

    chain_length = int(operator.length)
    stochastic_solver = resolve_stochastic_solver(sim_params, solver=solver)

    static_ctx: MCWFContext | None = None
    if stochastic_solver == "MCWF":
        static_ctx = make_mcwf_static_context(operator, sim_params, noise_model=None)

    if rng is None:
        rng = np.random.default_rng(0 if seed is None else int(seed))
    if timesteps is None:
        timesteps = [float(sim_params.dt)] * (int(k) + 1)
    if len(timesteps) != int(k) + 1:
        msg = f"Comb schedule: timesteps length must be k+1={int(k) + 1}, got {len(timesteps)}."
        raise ValueError(msg)

    psi_pairs_list: list[list[Any]] = []
    initial_psis: list[np.ndarray] = []
    choi_feature_rows_per_sequence: list[np.ndarray] = []

    for _ in range(int(n)):
        rho_in = sample_density_matrix(rng)
        step_pairs, choi_rows = sample_train_sequence(
            int(k),
            normalize_kind(str(interventions)),
            rng,
        )
        psi_pairs_list.append(step_pairs)
        choi_feature_rows_per_sequence.append(choi_rows.astype(np.float32))
        initial_psi = sample_initial_psi(rho_in, length=int(chain_length), rng=rng, init_mode=init_mode)
        if isinstance(initial_psi, tuple):
            initial_psi = initial_psi[0]
        initial_psis.append(initial_psi)

    samples = cast(
        "list[SeqTrace]",
        simulate_sequences(
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
            solver=stochastic_solver,
            _execution=_execution,
        ),
    )
    rho0_batch, features_batch, rho_seq_batch, _ctx = stack_traces(samples)
    return pack_dataset(rho0_batch, features_batch, rho_seq_batch)


def train_surrogate_model(
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
    solver: StochasticSolver | None = None,
    interventions: str = "measure_prepare",
    _execution: ExecutionConfig | None = None,
) -> TransformerComb:
    """Train a surrogate model end-to-end on simulated sequence traces.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        k: Number of intervention steps.
        n: Number of sampled sequences.
        seed: Seed used for data generation RNG.
        parallel: Whether to parallelize data generation.
        show_progress: Whether to show progress bars.
        timesteps: Optional per-step durations passed to :func:`sample_train_dataset`.
        init_mode: Initial-state sampling mode passed to :func:`sample_train_dataset`.
        solver: Optional stochastic solver override passed to :func:`sample_train_dataset`.
        interventions: Training intervention kind passed to :func:`sample_train_dataset`.
        model_kwargs: Optional keyword arguments forwarded to :class:`TransformerComb`.
        train_kwargs: Optional keyword arguments forwarded to :meth:`TransformerComb.fit`.

    Returns:
        Trained :class:`TransformerComb`.
    """
    import torch  # noqa: PLC0415

    from .model import TransformerComb

    rng = np.random.default_rng(0 if seed is None else int(seed))
    train_data = sample_train_dataset(
        operator,
        sim_params,
        k=int(k),
        n=int(n),
        rng=rng,
        parallel=bool(parallel),
        show_progress=bool(show_progress),
        timesteps=timesteps,
        init_mode=init_mode,
        solver=solver,
        interventions=interventions,
        _execution=_execution,
    )

    resolved_model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    resolved_train_kwargs = {} if train_kwargs is None else dict(train_kwargs)
    device_arg = resolved_train_kwargs.pop("device", None)
    if device_arg is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg) if isinstance(device_arg, str) else device_arg
    d_e = int(train_data.tensors[0].shape[-1])
    model = TransformerComb(d_e=d_e, d_rho=8, **resolved_model_kwargs).to(device)

    model.fit(train_data, device=device, **resolved_train_kwargs)
    return model


__all__ = ["sample_train_dataset", "train_surrogate_model"]
