# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Tomography estimation pipelines (sampled subset or SIS) over discrete basis sequences."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from tqdm import tqdm

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.simulator import WORKER_CTX, available_cpus, run_backend_parallel

from .basis import TomographyBasis, _finalize_sequence_averages, calculate_dual_choi_basis, get_basis_states, get_choi_basis
from .sampling import SequenceData, _enumerate_sequences
from ..core.tomography_utils import (
    _evolve_backend_state,
    _get_rho_site_zero,
    _initialize_backend_state,
    _reprepare_backend_state_forced,
    make_mcwf_static_context,
)
from ..exact.formatters import _to_dense, _to_mpo

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from ..exact.combs import DenseComb, MPOComb


def _estimate_sequence_worker(
    job_idx: int,
    payload: dict[str, Any] | None = None,
) -> tuple[int, int, "np.ndarray", float]:
    """Estimate worker: simulate one discrete-basis sequence (one trajectory)."""
    ctx = payload if payload is not None else WORKER_CTX
    num_trajectories: int = int(ctx["num_trajectories"])
    s_idx: int = int(job_idx // num_trajectories)
    traj_idx: int = int(job_idx % num_trajectories)

    psi_pairs = ctx["psi_pairs"][s_idx]
    operator = ctx["operator"]
    sim_params = ctx["sim_params"]
    timesteps: list[float] = ctx["timesteps"]
    noise_model = ctx["noise_model"]

    if noise_model is None:
        assert num_trajectories == 1, "num_trajectories must be 1 when noise_model is None."

    solver = sim_params.solver
    current_state = _initialize_backend_state(operator, solver)

    weight = 1.0
    for step_i, (psi_meas, psi_prep) in enumerate(psi_pairs):
        current_state, step_prob = _reprepare_backend_state_forced(
            current_state, psi_meas, psi_prep, solver
        )
        weight *= float(step_prob)
        if weight < 1e-15:
            break

        duration = float(timesteps[step_i])
        step_params = copy.deepcopy(sim_params)
        step_params.elapsed_time = duration
        step_params.num_traj = 1
        step_params.show_progress = False
        step_params.get_state = True
        n_steps = max(1, int(np.round(duration / step_params.dt)))
        step_params.times = np.linspace(0, n_steps * step_params.dt, n_steps + 1)

        current_state = _evolve_backend_state(
            current_state,
            operator,
            noise_model,
            step_params,
            solver,
            traj_idx=traj_idx,
            static_ctx=ctx.get("mcwf_static_ctx"),
        )

    rho_final = _get_rho_site_zero(current_state)
    return (s_idx, traj_idx, rho_final, float(weight))


def _call_backend_serial(backend: Callable[..., Any], *args: Any) -> Any:
    import contextlib  # noqa: PLC0415

    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        return backend(*args)
    with contextlib.suppress(Exception), threadpool_limits(limits=1):
        return backend(*args)


def _estimate_uniform_subset(
    operator: MPO,
    sim_params: "AnalogSimParams",
    timesteps: list[float],
    *,
    parallel: bool,
    num_samples: int,
    num_trajectories: int,
    noise_model: "NoiseModel | None",
    seed: int | None,
    basis: TomographyBasis,
    basis_seed: int | None,
) -> SequenceData:
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True

    basis_set = get_basis_states(basis=basis, seed=basis_seed)
    choi_basis, choi_indices = get_choi_basis(basis=basis, seed=basis_seed)
    choi_duals = calculate_dual_choi_basis(choi_basis)

    k = len(timesteps)
    rng = np.random.default_rng(seed)
    all_seqs = _enumerate_sequences(k)
    if len(all_seqs) == 0:
        raise ValueError("No sequences available for k=0.")

    num_seqs = len(all_seqs)
    n_pick = min(int(num_samples), num_seqs)
    if n_pick <= 0:
        raise ValueError("num_samples must be positive.")

    if n_pick == num_seqs:
        picked_seqs = all_seqs
    else:
        pick_idx = rng.choice(num_seqs, size=n_pick, replace=False)
        picked_seqs = [all_seqs[i] for i in pick_idx]

    samples_psi_pairs = [
        [(basis_set[choi_indices[a][1]][1], basis_set[choi_indices[a][0]][1]) for a in seq]
        for seq in picked_seqs
    ]

    if noise_model is None:
        noise_model = local_params.noise_model
    if noise_model is None:
        num_trajectories = 1

    mcwf_static_ctx = None
    if local_params.solver == "MCWF":
        mcwf_static_ctx = make_mcwf_static_context(operator, local_params, noise_model=noise_model)
    elif local_params.solver != "TJM":
        raise ValueError(f"Basis sampler does not support solver {local_params.solver!r}.")

    total_jobs = n_pick * num_trajectories
    payload = {
        "psi_pairs": samples_psi_pairs,
        "num_trajectories": num_trajectories,
        "operator": operator,
        "sim_params": local_params,
        "timesteps": timesteps,
        "noise_model": noise_model,
        "mcwf_static_ctx": mcwf_static_ctx,
    }

    aggregated_outputs = [np.zeros((2, 2), dtype=np.complex128) for _ in range(n_pick)]
    aggregated_weights = np.zeros(n_pick, dtype=np.float64)

    if parallel and total_jobs > 1:
        max_workers = max(1, available_cpus() - 1)
        results_iterator = run_backend_parallel(
            worker_fn=_estimate_sequence_worker,
            payload=payload,
            n_jobs=total_jobs,
            max_workers=max_workers,
            show_progress=local_params.show_progress,
            desc=f"Simulating {n_pick} basis sequences (Parallel)",
        )
        for _, (s_idx, _traj_idx, rho_final, weight) in results_iterator:
            aggregated_outputs[s_idx] += rho_final * weight
            aggregated_weights[s_idx] += weight
    else:
        disable_tqdm = not local_params.show_progress
        for job_idx in tqdm(
            range(total_jobs),
            desc=f"Simulating {n_pick} basis sequences (Serial)",
            disable=disable_tqdm,
        ):
            (s_idx, _traj_idx, rho_final, weight) = _call_backend_serial(
                _estimate_sequence_worker, job_idx, payload
            )
            aggregated_outputs[s_idx] += rho_final * weight
            aggregated_weights[s_idx] += weight

    acc: dict[tuple[int, ...], list[Any]] = {}
    for i in range(n_pick):
        acc[picked_seqs[i]] = [aggregated_outputs[i], aggregated_weights[i], num_trajectories]

    final_seqs, final_outputs, final_weights = _finalize_sequence_averages(acc, float(num_trajectories))

    if n_pick < num_seqs:
        inclusion_correction = float(num_seqs) / float(n_pick)
        final_weights = [w * inclusion_correction for w in final_weights]

    return SequenceData(
        sequences=final_seqs,
        outputs=final_outputs,
        weights=final_weights,
        choi_basis=choi_basis,
        choi_indices=choi_indices,
        choi_duals=choi_duals,
        timesteps=timesteps,
    )


def _estimate_sis(
    operator: MPO,
    sim_params: "AnalogSimParams",
    timesteps: list[float],
    *,
    parallel: bool,
    num_samples: int,
    noise_model: "NoiseModel | None",
    seed: int | None,
    proposal: Literal["uniform", "local", "mixture"],
    prep_mixture_eps: float,
    basis: TomographyBasis,
    basis_seed: int | None,
) -> SequenceData:
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True

    basis_set = get_basis_states(basis=basis, seed=basis_seed)
    choi_basis, choi_indices = get_choi_basis(basis=basis, seed=basis_seed)
    choi_duals = calculate_dual_choi_basis(choi_basis)

    k = len(timesteps)
    rng = np.random.default_rng(seed)
    solver = local_params.solver

    if noise_model is None:
        noise_model = local_params.noise_model

    static_ctx = None
    if solver == "MCWF":
        static_ctx = make_mcwf_static_context(operator, local_params, noise_model=noise_model)
    elif solver != "TJM":
        raise ValueError(f"Discrete SIS does not support solver {solver!r}.")

    num_seqs = 16**k
    if num_samples >= num_seqs:
        return _estimate_uniform_subset(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=num_seqs,
            num_trajectories=1,
            noise_model=noise_model,
            seed=0,
            basis=basis,
            basis_seed=basis_seed,
        )

    meas_projectors = [basis_set[m][2] for m in range(4)]

    seen: set[tuple[int, ...]] = set()
    acc: dict[tuple[int, ...], list[Any]] = {}

    max_tries = 10_000
    for _sample_idx in range(int(num_samples)):
        tries = 0
        while True:
            tries += 1
            if tries > max_tries:
                raise RuntimeError("Too many duplicate sequences while sampling without replacement.")

            state = _initialize_backend_state(operator, solver)
            log_w = 0.0
            seq: list[int] = []

            alive = True
            for step_idx, duration in enumerate(timesteps):
                rho_0 = _get_rho_site_zero(state)
                tr_rho = float(np.trace(rho_0).real)
                if tr_rho < 1e-300:
                    alive = False
                    break

                l = np.array([float(np.trace(E @ rho_0).real) for E in meas_projectors], dtype=np.float64)
                l = np.clip(l, 0.0, np.inf)
                l_sum = float(np.sum(l))
                q_m = (
                    np.full(4, 0.25, dtype=np.float64)
                    if (l_sum <= 0.0 or not np.isfinite(l_sum))
                    else (l / l_sum)
                )

                if proposal == "uniform":
                    mode = "uniform"
                elif proposal == "local":
                    mode = "local"
                else:
                    mode = "uniform" if rng.random() < prep_mixture_eps else "local"

                if mode == "uniform":
                    alpha = int(rng.integers(0, 16))
                    p, m = choi_indices[alpha]
                    q_alpha = 1.0 / 16.0
                else:
                    m = int(rng.choice(4, p=q_m))
                    p = int(rng.integers(0, 4))
                    alpha = 4 * p + m
                    q_struct_alpha = float(q_m[m]) * 0.25
                    q_alpha = (
                        float(prep_mixture_eps) * (1.0 / 16.0) + float(1.0 - prep_mixture_eps) * q_struct_alpha
                        if proposal == "mixture"
                        else q_struct_alpha
                    )

                psi_meas = basis_set[m][1]
                psi_prep = basis_set[p][1]
                state, step_prob = _reprepare_backend_state_forced(state, psi_meas, psi_prep, solver)
                if step_prob < 1e-300 or q_alpha <= 0.0:
                    alive = False
                    break

                log_w += float(np.log(step_prob) - np.log(q_alpha))
                seq.append(alpha)

                sp = copy.deepcopy(local_params)
                sp.elapsed_time = duration
                sp.num_traj = 1
                sp.get_state = True
                sp.show_progress = False
                n_steps = max(1, int(np.round(duration / sp.dt)))
                sp.times = np.linspace(0, n_steps * sp.dt, n_steps + 1)
                state = _evolve_backend_state(
                    state,
                    operator,
                    noise_model,
                    sp,
                    solver,
                    traj_idx=0,
                    static_ctx=static_ctx,
                )

            if not alive:
                continue

            seq_t = tuple(seq)
            if seq_t in seen:
                continue
            seen.add(seq_t)

            w = float(np.exp(log_w))
            rho_final = _get_rho_site_zero(state)
            acc[seq_t] = [w * rho_final, w, 1]
            break

    final_seqs, final_outputs, final_weights = _finalize_sequence_averages(acc, float(num_samples))
    return SequenceData(
        sequences=final_seqs,
        outputs=final_outputs,
        weights=final_weights,
        choi_basis=choi_basis,
        choi_indices=choi_indices,
        choi_duals=choi_duals,
        timesteps=timesteps,
    )


def run_estimate(
    operator: MPO,
    sim_params: "AnalogSimParams",
    timesteps: list[float] | None = None,
    *,
    mode: Literal["estimate", "exhaustive", "sis"] = "estimate",
    output: Literal["dense", "mpo"] = "dense",
    noise_model: "NoiseModel | None" = None,
    parallel: bool = True,
    num_samples: int | None = 1000,
    seed: int | None = None,
    num_trajectories: int = 100,
    compress_every: int = 100,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 2,
    basis: TomographyBasis = "tetrahedral",
    basis_seed: int | None = None,
    proposal: Literal["uniform", "local", "mixture"] = "mixture",
    prep_mixture_eps: float = 0.1,
) -> "DenseComb | MPOComb":
    """Estimate a comb from discrete basis sequences (sampled subset or SIS).

    - ``mode="estimate"``: uniform subset of sequences without replacement (MC).
    - ``mode="sis"``: discrete SIS over the 16-element basis (kept for optionality).
    - ``mode="exhaustive"``: evaluate all sequences (used by :func:`exact.exhaustive.run_exhaustive`).
    """
    if timesteps is None:
        timesteps = [sim_params.elapsed_time]

    valid_solvers = {"MCWF", "TJM"}
    if sim_params.solver not in valid_solvers:
        raise ValueError(f"Tomography currently only supports solvers {valid_solvers}, got {sim_params.solver!r}.")

    if mode == "sis" and num_trajectories != 1:
        raise ValueError("SIS currently only supports num_trajectories=1 (one realization per particle).")

    if mode == "exhaustive":
        k = len(timesteps)
        num_samples_eff = 16**k
        data = _estimate_uniform_subset(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=num_samples_eff,
            num_trajectories=num_trajectories,
            noise_model=noise_model,
            seed=0,
            basis=basis,
            basis_seed=basis_seed,
        )
    elif mode == "estimate":
        if num_samples is None:
            raise ValueError("num_samples must be set for mode='estimate'.")
        data = _estimate_uniform_subset(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=int(num_samples),
            num_trajectories=num_trajectories,
            noise_model=noise_model,
            seed=seed,
            basis=basis,
            basis_seed=basis_seed,
        )
    elif mode == "sis":
        if num_samples is None:
            raise ValueError("num_samples must be set for mode='sis'.")
        data = _estimate_sis(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=int(num_samples),
            noise_model=noise_model,
            seed=seed,
            proposal=proposal,
            prep_mixture_eps=prep_mixture_eps,
            basis=basis,
            basis_seed=basis_seed,
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}.")

    if output == "dense":
        estimate = _to_dense(data)
        return estimate.to_dense_comb()
    if output == "mpo":
        return _to_mpo(
            data,
            compress_every=compress_every,
            tol=tol,
            max_bond_dim=max_bond_dim,
            n_sweeps=n_sweeps,
        )
    raise ValueError(f"Unknown output format {output!r}.")

