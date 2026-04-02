# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Surrogate workflow: data generation and end-to-end training entry points.

**Public** (see ``__all__``): :func:`generate_data`, :func:`create_surrogate`.

:func:`generate_data` returns a :class:`~torch.utils.data.TensorDataset` for :meth:`~mqt.yaqs.characterization.tomography.surrogate.model.TransformerComb.fit`.

**Internals** — Same execution pattern as :mod:`mqt.yaqs.simulator`: :func:`simulate_sequences` builds a
process-pool payload (or uses :data:`~mqt.yaqs.simulator.WORKER_CTX`), then dispatches to
:func:`~mqt.yaqs.simulator.run_backend_parallel` or runs workers serially. Rollout types live in
:mod:`mqt.yaqs.characterization.tomography.surrogate.data`; the model is
:class:`~mqt.yaqs.characterization.tomography.surrogate.model.TransformerComb`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1) STANDARD LIBRARY
# ---------------------------------------------------------------------------
import copy
from collections.abc import Callable
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
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

# ---------------------------------------------------------------------------
# 4) LOCAL — tomography surrogate stack
# ---------------------------------------------------------------------------
from ..core.predictor_encoding import (
    normalize_rho_from_backend_output,
    pack_rho8,
    random_density_matrix,
    sample_random_intervention_sequence,
)
from ..core.tomography_utils import (
    _evolve_backend_state,
    _get_rho_site_zero,
    _reprepare_backend_state_forced,
    build_initial_psi,
)
from .data import SequenceRolloutSample, stack_rollouts
from .model import TransformerComb


# ---------------------------------------------------------------------------
# 5) PARALLEL JOB PAYLOAD
# ---------------------------------------------------------------------------
# ``simulate_sequences`` passes the following dict to ``run_backend_parallel`` (initializer →
# ``WORKER_CTX``) or directly to workers in the serial path. Keys must stay stable for pickling.
#
#   psi_pairs               list[list[(meas, prep)]] per sequence — Kraus-style intervention steps
#   initial_psi             list of MPS initial states (one per sequence)
#   num_trajectories        MCWF trajectory index split (1 when noise_model is None)
#   operator, sim_params    Hamiltonian MPO and analog parameters
#   timesteps               default duration per step (length k) unless timesteps_rows is set
#   timesteps_rows          optional per-sequence per-step durations
#   operators_list          optional per-sequence per-step MPO (piecewise Hamiltonian)
#   noise_model             None here (deterministic surrogate rollouts)
#   mcwf_static_ctx         static context forwarded to the backend for the whole sequence
#   mcwf_static_ctx_list    optional per-step static context overrides
#   e_features_rows         list of (k, d_e) float32 Choi feature rows (rollout path only)


# ---------------------------------------------------------------------------
# 6) WORKERS — one job index → one sequence (see ``num_trajectories`` for MCWF)
# ---------------------------------------------------------------------------
def _surrogate_final_state_worker(
    job_idx: int,
    job_payload: dict[str, Any] | None = None,
) -> tuple[int, int, np.ndarray, float]:
    """Run one intervention sequence and return the **final** reduced state on site 0 plus importance weight.

    Does not record per-step states (cheaper than :func:`_surrogate_rollout_worker`).

    Args:
        job_idx: Flat index ``sequence_index * num_trajectories + trajectory_index``.
        job_payload: Per-pool shared context; defaults to :data:`~mqt.yaqs.simulator.WORKER_CTX`.

    Returns:
        ``(sequence_index, trajectory_index, rho_final_site0, cumulative_weight)``.
    """
    worker_ctx = job_payload if job_payload is not None else WORKER_CTX
    num_trajectories: int = int(worker_ctx["num_trajectories"])
    sequence_idx = int(job_idx // num_trajectories)
    trajectory_idx = int(job_idx % num_trajectories)

    psi_pairs = worker_ctx["psi_pairs"][sequence_idx]
    hamiltonian = worker_ctx["operator"]
    sim_params = worker_ctx["sim_params"]
    timesteps: list[float] = worker_ctx["timesteps"]
    noise_model = worker_ctx["noise_model"]
    initial_states: list[np.ndarray] = worker_ctx["initial_psi"]

    if noise_model is None:
        assert num_trajectories == 1, "num_trajectories must be 1 when noise_model is None."

    solver = sim_params.solver
    state = np.asarray(initial_states[sequence_idx], dtype=np.complex128).copy()

    cumulative_weight = 1.0
    for step_idx, (psi_meas, psi_prep) in enumerate(psi_pairs):
        state, step_prob = _reprepare_backend_state_forced(state, psi_meas, psi_prep, solver)
        cumulative_weight *= float(step_prob)
        if cumulative_weight < 1e-15:
            break

        duration = float(timesteps[step_idx])
        step_params = copy.deepcopy(sim_params)
        step_params.elapsed_time = duration
        step_params.num_traj = 1
        step_params.show_progress = False
        step_params.get_state = True
        n_integration_steps = max(1, int(np.round(duration / step_params.dt)))
        step_params.times = np.linspace(0, n_integration_steps * step_params.dt, n_integration_steps + 1)

        state = _evolve_backend_state(
            state,
            hamiltonian,
            noise_model,
            step_params,
            solver,
            traj_idx=trajectory_idx,
            static_ctx=worker_ctx.get("mcwf_static_ctx"),
        )

    rho_final = _get_rho_site_zero(state)
    return (sequence_idx, trajectory_idx, rho_final, float(cumulative_weight))


def _surrogate_rollout_worker(
    job_idx: int,
    job_payload: dict[str, Any] | None = None,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run one sequence and record **packed** site-0 states after each step (for supervised training).

    Choi feature rows (``e_features_rows``) are passed through unchanged into the sample; the worker
    only ensures shape ``(num_steps, d_e)`` matches the simulation.

    Args:
        job_idx: Flat index ``sequence_index * num_trajectories + trajectory_index``.
        job_payload: Shared pool context; defaults to :data:`~mqt.yaqs.simulator.WORKER_CTX`.

    Returns:
        ``(sequence_index, trajectory_index, rho0_packed, choi_features_matrix, rho_seq_packed, weight)`` where
        ``choi_features_matrix`` is ``(num_steps, d_e)`` and ``rho_seq_packed`` is ``(num_steps, 8)``.
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

    rho0_raw = _get_rho_site_zero(state)
    rho0_packed = pack_rho8(normalize_rho_from_backend_output(rho0_raw)).astype(np.float32)

    if per_sequence_choi_rows is None:
        msg = "Rollout worker requires `e_features_rows`: per-sequence Choi feature rows."
        raise ValueError(msg)
    choi_features_matrix = np.asarray(per_sequence_choi_rows[sequence_idx], dtype=np.float32).reshape(num_steps, -1)

    if timesteps_per_sequence is not None:
        step_durations = [float(timesteps_per_sequence[sequence_idx][i]) for i in range(num_steps)]
    else:
        step_durations = [float(timesteps[i]) for i in range(num_steps)]

    if hamiltonians_per_step is not None:
        hamiltonian_this_step = [hamiltonians_per_step[sequence_idx][i] for i in range(num_steps)]
    else:
        hamiltonian_this_step = None

    if mcwf_ctx_per_step is not None:
        mcwf_ctx_for_step = [mcwf_ctx_per_step[sequence_idx][i] for i in range(num_steps)]
    else:
        mcwf_ctx_for_step = None

    cumulative_weight = 1.0
    rho_packed_after_step: list[np.ndarray] = []
    last_rho_packed = rho0_packed.copy()

    for step_idx, (psi_meas, psi_prep) in enumerate(psi_pairs):
        state, step_prob = _reprepare_backend_state_forced(state, psi_meas, psi_prep, solver)
        cumulative_weight *= float(step_prob)
        if cumulative_weight < 1e-15:
            while len(rho_packed_after_step) < num_steps:
                rho_packed_after_step.append(last_rho_packed.copy())
            break

        duration = float(step_durations[step_idx])
        op = hamiltonian if hamiltonian_this_step is None else hamiltonian_this_step[step_idx]
        step_mcwf_ctx = worker_ctx.get("mcwf_static_ctx") if mcwf_ctx_for_step is None else mcwf_ctx_for_step[step_idx]

        step_params = copy.deepcopy(sim_params)
        step_params.elapsed_time = duration
        step_params.num_traj = 1
        step_params.show_progress = False
        step_params.get_state = True
        n_integration_steps = max(1, int(np.round(duration / step_params.dt)))
        step_params.times = np.linspace(0, n_integration_steps * step_params.dt, n_integration_steps + 1)

        state = _evolve_backend_state(
            state,
            op,
            noise_model,
            step_params,
            solver,
            traj_idx=trajectory_idx,
            static_ctx=step_mcwf_ctx,
        )

        rho_step = _get_rho_site_zero(state)
        rho_normalized = normalize_rho_from_backend_output(rho_step)
        last_rho_packed = pack_rho8(rho_normalized).astype(np.float32)
        rho_packed_after_step.append(last_rho_packed)

    while len(rho_packed_after_step) < num_steps:
        rho_packed_after_step.append(last_rho_packed.copy())

    rho_sequence_packed = np.stack(rho_packed_after_step, axis=0).astype(np.float32)
    if choi_features_matrix.shape[0] != num_steps:
        msg = "Choi feature rows must have length num_steps matching intervention steps."
        raise ValueError(msg)
    return (sequence_idx, trajectory_idx, rho0_packed, choi_features_matrix, rho_sequence_packed, float(cumulative_weight))


# ---------------------------------------------------------------------------
# 7) SERIAL BACKEND WRAPPER — same thread cap pattern as :func:`mqt.yaqs.simulator._call_backend`
# ---------------------------------------------------------------------------
def _call_worker_serial(worker_fn: Callable[..., Any], *args: Any) -> Any:
    """Run a single worker call under ``threadpoolctl`` limits (if installed); else direct call."""
    import contextlib  # noqa: PLC0415

    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        return worker_fn(*args)
    with contextlib.suppress(Exception), threadpool_limits(limits=1):
        return worker_fn(*args)


# ---------------------------------------------------------------------------
# 8) simulate_sequences — internal primitive (feeds :func:`generate_data`)
# ---------------------------------------------------------------------------
def simulate_sequences(
    *,
    operator: MPO,
    sim_params: "AnalogSimParams",
    timesteps: list[float],
    psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]],
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
    """Simulate many intervention **sequences** and return rollouts or final states.

    **Modes**

    - ``record_step_states=True`` (default): returns ``list[SequenceRolloutSample]`` with per-step
      packed reduced states and Choi features (``d_e``-dim rows) aligned with ``psi_pairs_list``.
    - ``record_step_states=False``: returns a single ``(N, 8)`` float32 tensor of **final** packed
      states only (no Choi tensor needed).

    **Choi features**

    When recording rollouts, pass ``e_features_rows``: one ``(k, d_e)`` float32 array per sequence
    (fixed-basis indices map to rows of a precomputed table; continuous sampling uses
    :func:`~..core.predictor_encoding.sample_random_intervention_sequence` features).

    **Parallelism**

    Uses :func:`~mqt.yaqs.simulator.run_backend_parallel` when ``parallel`` and ``N > 1``; otherwise
    calls :func:`_call_worker_serial` per sequence index.
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
                raise RuntimeError("Parallel sequence simulation incomplete.")
            stacked_final = [cast(np.ndarray, x) for x in final_packed_by_index]
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

    optional_context_vec = (
        None if context_vec is None else np.asarray(context_vec, dtype=np.float32).reshape(-1)
    )

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
            raise RuntimeError("Parallel sequence rollout simulation incomplete.")
        return [cast(SequenceRolloutSample, s) for s in samples_by_index]

    return [rollout_one_sequence(j) for j in range(num_sequences)]


# ---------------------------------------------------------------------------
# 9) Helpers — used only by :func:`generate_data`
# ---------------------------------------------------------------------------
def _psi_from_rank1_projector(projector: np.ndarray) -> np.ndarray:
    """Unit ket from a 2×2 rank-1 projector (measurement / prep direction)."""
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
    """Stack NumPy rollout batches into a :class:`~torch.utils.data.TensorDataset` (internal; used by :func:`generate_data`)."""

    import torch  # noqa: PLC0415
    from torch.utils.data import TensorDataset  # noqa: PLC0415

    return TensorDataset(
        torch.as_tensor(E_features, dtype=torch.float32),
        torch.as_tensor(rho0, dtype=torch.float32),
        torch.as_tensor(rho_seq, dtype=torch.float32),
    )


def generate_data(
    operator: MPO,
    sim_params: "AnalogSimParams",
    static_ctx: Any,
    *,
    k: int,
    n: int,
    chain_length: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    parallel: bool = True,
    show_progress: bool = True,
    timesteps: list[float] | None = None,
    init_mode: str = "eigenstate",
):
    """Sample **continuous** random rank-1 interventions, simulate, return a :class:`~torch.utils.data.TensorDataset`.

    Requires PyTorch. Underlying simulation uses NumPy; the returned dataset packs
    ``(E_features, rho0, rho_seq)`` with shapes ``(n, k, d_e)``, ``(n, 8)``, ``(n, k, 8)`` (see :func:`stack_rollouts`).
    """
    if rng is None:
        rng = np.random.default_rng(0 if seed is None else int(seed))
    if timesteps is None:
        timesteps = [float(sim_params.dt)] * int(k)
    if len(timesteps) != int(k):
        msg = f"timesteps length {len(timesteps)} must equal k={k}."
        raise ValueError(msg)

    psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]] = []
    initial_psis: list[np.ndarray] = []
    choi_feature_rows_per_sequence: list[np.ndarray] = []

    for _ in range(int(n)):
        rho_in = random_density_matrix(rng)
        intervention_maps, choi_rows = sample_random_intervention_sequence(int(k), rng)
        step_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for emap in intervention_maps:
            rho_prep = np.asarray(getattr(emap, "rho_prep"), dtype=np.complex128)
            effect_dm = np.asarray(getattr(emap, "effect"), dtype=np.complex128)
            psi_meas = _psi_from_rank1_projector(effect_dm)
            psi_prep = _psi_from_rank1_projector(rho_prep)
            step_pairs.append((psi_meas, psi_prep))
        psi_pairs_list.append(step_pairs)
        choi_feature_rows_per_sequence.append(choi_rows.astype(np.float32))
        initial_psis.append(
            build_initial_psi(rho_in, length=int(chain_length), rng=rng, init_mode=init_mode)
        )

    samples = simulate_sequences(
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
    sim_params: "AnalogSimParams",
    static_ctx: Any,
    *,
    k: int,
    n: int,
    chain_length: int,
    seed: int | None = None,
    parallel: bool = True,
    show_progress: bool = True,
    timesteps: list[float] | None = None,
    init_mode: str = "eigenstate",
    model_kwargs: dict[str, Any] | None = None,
    train_kwargs: dict[str, Any] | None = None,
) -> TransformerComb:
    """End-to-end: :func:`generate_data` → :class:`~mqt.yaqs.characterization.tomography.surrogate.model.TransformerComb` → :meth:`~mqt.yaqs.characterization.tomography.surrogate.model.TransformerComb.fit`."""
    import torch  # noqa: PLC0415

    rng = np.random.default_rng(0 if seed is None else int(seed))
    train_data = generate_data(
        operator,
        sim_params,
        static_ctx,
        k=int(k),
        n=int(n),
        chain_length=int(chain_length),
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
