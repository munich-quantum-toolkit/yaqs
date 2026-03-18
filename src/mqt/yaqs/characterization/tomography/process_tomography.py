# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Automated Process Tomography Module.

Provides functions for process tomography on a quantum system modeled by MPO
evolution. The primary entry point is :func:`run`.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from tqdm import tqdm

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.simulator import available_cpus, run_backend_parallel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from .combs import DenseComb, MPOComb

from .basis import (
    _finalize_sequence_averages,
    calculate_dual_choi_basis,
    get_basis_states,
    get_choi_basis,
)
from .estimator_class import TomographyEstimate
from .formatters import _to_dense, _to_mpo, rank1_upsilon_mpo_term
from .sampling import (
    SamplingData,
    SequenceData,
    _continuous_dual_step,
    _enumerate_sequences,
    _normalize_log_weights,
    _sample_haar_pure_state,
    _sample_local_measurement_exact,
    _sample_random_intervention_sequence,
)
from .tomography_utils import (
    _evolve_backend_state,
    _get_rho_site_zero,
    _initialize_backend_state,
    _reprepare_backend_state_forced,
    _sis_evolve_worker,
    _tomography_sequence_worker,
)

# Aliases used by tests
_sequence_data_to_mpo = _to_mpo
_sequence_data_to_dense = _to_dense

# Re-exports for tests and external callers
__all__ = [
    "run",
    "get_basis_states",
    "get_choi_basis",
    "calculate_dual_choi_basis",
    "rank1_upsilon_mpo_term",
    "_run_exhaustive_sequence_data",
    "_to_dense",
    "_to_mpo",
    "_estimate_sis_sequence_data",
    "_sequence_data_to_mpo",
    "_sequence_data_to_dense",
]


def _call_backend_serial(backend: Callable[..., Any], *args: Any) -> Any:
    """Invoke a backend function serially with thread capping (if available)."""
    import contextlib  # noqa: PLC0415
    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        return backend(*args)
    with contextlib.suppress(Exception), threadpool_limits(limits=1):
        return backend(*args)


def _run_mc_sampling(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_samples: int = 1000,
    num_trajectories: int = 1,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
) -> SamplingData:
    """True continuous Monte Carlo sampling estimator."""
    from mqt.yaqs.analog.mcwf import preprocess_mcwf

    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True
    k = len(timesteps)
    rng = np.random.default_rng(seed)

    samples_psi_pairs = []
    samples_dual_ops = []
    for _ in range(num_samples):
        psi_pairs, dual_ops = _sample_random_intervention_sequence(k, rng)
        samples_psi_pairs.append(psi_pairs)
        samples_dual_ops.append(dual_ops)

    if noise_model is None:
        noise_model = local_params.noise_model
    if noise_model is None:
        num_trajectories = 1

    mcwf_static_ctx = None
    if local_params.solver == "MCWF":
        dummy_mps = MPS(length=operator.length, state="zeros")
        mcwf_static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, local_params)
    elif local_params.solver != "TJM":
        msg = f"MC sampler does not support solver {local_params.solver!r}."
        raise ValueError(msg)

    total_jobs = num_samples * num_trajectories
    payload = {
        "psi_pairs": samples_psi_pairs,
        "num_trajectories": num_trajectories,
        "operator": operator,
        "sim_params": local_params,
        "timesteps": timesteps,
        "noise_model": noise_model,
        "mcwf_static_ctx": mcwf_static_ctx,
    }

    sample_outputs = [np.zeros((2, 2), dtype=np.complex128) for _ in range(num_samples)]

    if parallel and total_jobs > 1:
        max_workers = max(1, available_cpus() - 1)
        results_iterator = run_backend_parallel(
            worker_fn=_tomography_sequence_worker,
            payload=payload,
            n_jobs=total_jobs,
            max_workers=max_workers,
            show_progress=local_params.show_progress,
            desc=f"Simulating {num_samples} MC Sampling trajectories (Parallel)",
        )
        for _, (s_idx, _traj_idx, rho_final, weight) in results_iterator:
            sample_outputs[s_idx] += rho_final * weight
    else:
        disable_tqdm = not local_params.show_progress
        for job_idx in tqdm(
            range(total_jobs),
            desc=f"Simulating {num_samples} MC Sampling trajectories (Serial)",
            disable=disable_tqdm,
        ):
            (s_idx, _traj_idx, rho_final, weight) = _call_backend_serial(
                _tomography_sequence_worker, job_idx, payload
            )
            sample_outputs[s_idx] += rho_final * weight

    final_outputs = [out / num_trajectories for out in sample_outputs]
    final_weights = [1.0 / num_samples] * num_samples
    return SamplingData(
        outputs=final_outputs,
        dual_ops=samples_dual_ops,
        weights=final_weights,
        timesteps=timesteps,
    )


def _run_basis_sampling(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_samples: int = 1000,
    num_trajectories: int = 1,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
    basis: Literal["standard", "random_low_overlap", "tetrahedral", "random_unitary"] = "standard",
    basis_seed: int | None = None,
) -> SequenceData:
    """Exhaustive-style evaluation on a uniformly sampled subset of the 16^k sequences.

    Sequences are sampled uniformly **without replacement**. The per-sequence outputs
    and weights are computed exactly as in the exhaustive estimator; only the set of
    evaluated sequences is reduced.

    If num_samples >= 16^k, this evaluates all sequences (i.e., becomes exhaustive).
    """
    from mqt.yaqs.analog.mcwf import preprocess_mcwf

    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True

    basis_set = get_basis_states(basis=basis, seed=basis_seed)
    choi_basis, choi_indices = get_choi_basis(basis=basis, seed=basis_seed)
    choi_duals = calculate_dual_choi_basis(choi_basis)

    k = len(timesteps)
    rng = np.random.default_rng(seed)
    all_seqs = _enumerate_sequences(k)
    if len(all_seqs) == 0:
        msg = "No sequences available for k=0."
        raise ValueError(msg)

    num_seqs = len(all_seqs)
    n_pick = min(int(num_samples), num_seqs)
    if n_pick <= 0:
        msg = "num_samples must be positive."
        raise ValueError(msg)

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
        dummy_mps = MPS(length=operator.length, state="zeros")
        mcwf_static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, local_params)
    elif local_params.solver != "TJM":
        msg = f"Basis sampler does not support solver {local_params.solver!r}."
        raise ValueError(msg)

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
            worker_fn=_tomography_sequence_worker,
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
                _tomography_sequence_worker, job_idx, payload
            )
            aggregated_outputs[s_idx] += rho_final * weight
            aggregated_weights[s_idx] += weight

    acc: dict[tuple[int, ...], list[Any]] = {}
    for i in range(n_pick):
        acc[picked_seqs[i]] = [aggregated_outputs[i], aggregated_weights[i], num_trajectories]

    final_seqs, final_outputs, final_weights = _finalize_sequence_averages(
        acc, float(num_trajectories)
    )

    # Uniform inclusion correction (Horvitz–Thompson style):
    # pi = n_pick / num_seqs for each selected sequence under uniform sampling without replacement.
    # Reweight sampled contributions by 1/pi so the expected sum matches the exhaustive estimator.
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


def _run_sis_sampling(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_samples: int = 1000,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
    ess_threshold: float = 0.5,
    proposal: str = "uniform",
    prep_mixture_eps: float = 0.1,
    prep_corr_prob: float = 0.0,
    resample: bool = True,
) -> SamplingData:
    """SIS estimator using log-weights and optional local proposal."""
    from mqt.yaqs.analog.mcwf import preprocess_mcwf

    if proposal not in ["uniform", "local", "mixture"]:
        msg = (
            f"Proposal {proposal!r} is not currently supported. "
            "Use 'uniform', 'local', or 'mixture'."
        )
        raise ValueError(msg)

    k = len(timesteps)
    rng = np.random.default_rng(seed)
    solver = sim_params.solver
    LOG_ZERO = -np.inf

    if noise_model is None:
        noise_model = sim_params.noise_model

    particles = [_initialize_backend_state(operator, solver) for _ in range(num_samples)]
    log_weights = np.zeros(num_samples, dtype=np.float64)
    dead = np.zeros(num_samples, dtype=bool)
    particles_dual_ops: list[list[NDArray[np.complex128]]] = [[] for _ in range(num_samples)]
    corr_prob_cap = 0.1
    confidence_threshold = 0.75

    static_ctx = None
    if solver == "MCWF":
        dummy_mps = MPS(length=operator.length, state="zeros")
        static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, sim_params)

    use_parallel = parallel and num_samples > 32
    max_workers = available_cpus()

    for step_idx, duration in enumerate(timesteps):
        prep_states_alive = []
        alive_indices = []

        for i in range(num_samples):
            if dead[i] or not np.isfinite(log_weights[i]):
                continue
            rho_0 = _get_rho_site_zero(particles[i])
            tr_rho = float(np.trace(rho_0).real)
            if tr_rho < 1e-300:
                log_weights[i] = LOG_ZERO
                dead[i] = True
                continue

            # Choose effective proposal mode (uniform / local) for this step.
            if proposal == "uniform":
                mode = "uniform"
            elif proposal == "local":
                mode = "local"
            elif proposal == "mixture":
                mode = "uniform" if rng.random() < prep_mixture_eps else "local"
            else:  # pragma: no cover - guarded by earlier validation
                mode = "uniform"

            if mode == "uniform":
                pm = _sample_haar_pure_state(rng)
                pp = _sample_haar_pure_state(rng)
                psi_new, step_prob = _reprepare_backend_state_forced(particles[i], pm, pp, solver)
                if step_prob < 1e-300:
                    log_weights[i] = LOG_ZERO
                    dead[i] = True
                    continue
                log_weights[i] += np.log(step_prob)
            else:  # mode == "local"
                pm = _sample_local_measurement_exact(rho_0, rng)
                pp = _sample_haar_pure_state(rng)
                if proposal == "mixture" and prep_corr_prob > 0.0:
                    # Confidence proxy for informativeness of the local draw pm:
                    # q(pm) ∝ <pm|rho_0|pm>. We normalize by tr(rho_0) to get a [0,1] score.
                    tr_rho_safe = max(float(np.trace(rho_0).real), 1e-300)
                    conf = float(np.real(np.vdot(pm, rho_0 @ pm)) / tr_rho_safe)
                    corr_p = min(float(prep_corr_prob), corr_prob_cap)
                    if conf > confidence_threshold and rng.random() < corr_p:
                        pp = pm
                psi_new, _ = _reprepare_backend_state_forced(particles[i], pm, pp, solver)
                log_weights[i] += np.log(max(tr_rho / 2.0, 1e-300))

            if np.isfinite(log_weights[i]):
                prep_states_alive.append(psi_new)
                alive_indices.append(i)
                particles_dual_ops[i].append(_continuous_dual_step(pm, pp))
            else:
                dead[i] = True

        if not alive_indices:
            log_weights[:] = LOG_ZERO
            dead[:] = True
            break

        if use_parallel:
            payload = {
                "prep_states": prep_states_alive,
                "duration": duration,
                "static_ctx": static_ctx,
                "sim_params": sim_params,
                "operator": operator,
                "noise_model": noise_model,
            }
            results = run_backend_parallel(
                worker_fn=_sis_evolve_worker,
                payload=payload,
                n_jobs=len(alive_indices),
                max_workers=max_workers,
                show_progress=sim_params.show_progress,
                desc=f"SIS Step {step_idx+1}/{k}",
            )
            for job_idx, evolved_state in results:
                particles[alive_indices[job_idx]] = evolved_state
        else:
            for j, idx in enumerate(alive_indices):
                sp = copy.deepcopy(sim_params)
                sp.elapsed_time = duration
                sp.num_traj = 1
                sp.get_state = True
                sp.show_progress = False
                n_steps = max(1, int(np.round(duration / sp.dt)))
                sp.times = np.linspace(0, n_steps * sp.dt, n_steps + 1)
                particles[idx] = _evolve_backend_state(
                    prep_states_alive[j], operator, noise_model, sp, solver,
                    traj_idx=0, static_ctx=static_ctx
                )

        alive_idx = np.flatnonzero(np.isfinite(log_weights))
        if len(alive_idx) == 0:
            log_weights[:] = LOG_ZERO
            dead[:] = True
            break

        w_norm_alive, log_w_sum = _normalize_log_weights(log_weights[alive_idx])
        ess = 1.0 / float(np.sum(w_norm_alive**2))

        if resample and ess < ess_threshold * num_samples:
            positions = (rng.random() + np.arange(num_samples)) / num_samples
            cumsum = np.cumsum(w_norm_alive)
            sel_local = np.searchsorted(cumsum, positions)
            idxs = alive_idx[sel_local]
            if solver == "MCWF":
                particles = [particles[j].copy() for j in idxs]
            else:
                particles = [copy.deepcopy(particles[j]) for j in idxs]
            particles_dual_ops = [copy.deepcopy(particles_dual_ops[j]) for j in idxs]
            new_log_weight = log_w_sum - np.log(num_samples)
            log_weights = np.full(num_samples, new_log_weight, dtype=np.float64)
            dead[:] = False

    final_outputs = []
    w_list: list[float] = []
    obs_list: list[float] = []
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    for i in range(num_samples):
        if not np.isfinite(log_weights[i]):
            final_outputs.append(np.zeros((2, 2), dtype=np.complex128))
            continue
        w = float(np.exp(log_weights[i]))
        rho_final = _get_rho_site_zero(particles[i])
        final_outputs.append(w * rho_final)
        w_list.append(w)
        obs_list.append(float(np.real(np.trace(z @ rho_final))))

    final_weights = [1.0 / num_samples] * num_samples

    weight_signal_corr = float("nan")
    if len(w_list) >= 2:
        w_arr = np.asarray(w_list, dtype=float)
        o_arr = np.asarray(obs_list, dtype=float)
        w_std = float(np.std(w_arr))
        o_std = float(np.std(o_arr))
        if w_std > 0.0 and o_std > 0.0:
            weight_signal_corr = float(np.corrcoef(w_arr, o_arr)[0, 1])

    # Diagnostics from final log_weights
    mask = np.isfinite(log_weights)
    if np.any(mask):
        w_norm, _ = _normalize_log_weights(log_weights[mask])
        ess = float(1.0 / np.sum(w_norm**2))
        ess_over_N = ess / num_samples
        log_weight_var = float(np.var(log_weights[mask]))
        max_weight = float(np.max(w_norm))
    else:
        ess = ess_over_N = log_weight_var = max_weight = float("nan")

    return SamplingData(
        outputs=final_outputs,
        dual_ops=particles_dual_ops,
        weights=final_weights,
        timesteps=timesteps,
        ess=ess,
        ess_over_N=ess_over_N,
        log_weight_var=log_weight_var,
        max_weight=max_weight,
        weight_signal_corr=weight_signal_corr,
    )


def _run_sis_basis_sampling(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    *,
    parallel: bool = True,
    num_samples: int = 1000,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
    ess_threshold: float = 0.5,
    proposal: str = "uniform",
    prep_mixture_eps: float = 0.1,
    resample: bool = True,
    basis: Literal["standard", "random_low_overlap", "tetrahedral", "random_unitary"] = "tetrahedral",
    basis_seed: int | None = None,
) -> SequenceData:
    """Discrete SIS over the 16-element Choi-map basis at each step.

    This samples sequences of discrete basis indices alpha_t ∈ {0..15} and returns a sparse
    `SequenceData` compatible with the standard formatting pipeline.
    """
    from mqt.yaqs.analog.mcwf import preprocess_mcwf

    if proposal not in ["uniform", "local", "mixture"]:
        msg = (
            f"Proposal {proposal!r} is not currently supported. "
            "Use 'uniform', 'local', or 'mixture'."
        )
        raise ValueError(msg)

    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True

    basis_set = get_basis_states(basis=basis, seed=basis_seed)
    choi_basis, choi_indices = get_choi_basis(basis=basis, seed=basis_seed)
    choi_duals = calculate_dual_choi_basis(choi_basis)

    k = len(timesteps)
    rng = np.random.default_rng(seed)
    solver = local_params.solver
    LOG_ZERO = -np.inf

    if noise_model is None:
        noise_model = local_params.noise_model

    static_ctx = None
    if solver == "MCWF":
        dummy_mps = MPS(length=operator.length, state="zeros")
        static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, local_params)
    elif solver != "TJM":
        msg = f"Discrete SIS does not support solver {solver!r}."
        raise ValueError(msg)

    if resample:
        raise ValueError("Discrete SIS does not support resampling (would repeat sequences).")

    num_seqs = 16**k
    if num_samples >= num_seqs:
        # Evaluate all sequences exactly (without replacement): identical to exhaustive.
        return _run_exhaustive_sequence_data(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_trajectories=1,
            noise_model=noise_model,
            basis=basis,
            basis_seed=basis_seed,
        )

    # Precompute measurement projectors for the 4 basis states.
    meas_projectors = [basis_set[m][2] for m in range(4)]

    seen: set[tuple[int, ...]] = set()
    acc: dict[tuple[int, ...], list[Any]] = {}

    max_tries = 10_000
    for _sample_idx in range(num_samples):
        tries = 0
        while True:
            tries += 1
            if tries > max_tries:
                raise RuntimeError("Too many duplicate sequences while sampling without replacement.")

            # Fresh particle
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

                l = np.array(
                    [float(np.trace(E @ rho_0).real) for E in meas_projectors], dtype=np.float64
                )
                l = np.clip(l, 0.0, np.inf)
                l_sum = float(np.sum(l))
                q_m = np.full(4, 0.25, dtype=np.float64) if (l_sum <= 0.0 or not np.isfinite(l_sum)) else (l / l_sum)

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
                    q_alpha = float(q_m[m]) * 0.25

                if proposal == "mixture":
                    q_alpha = float(prep_mixture_eps) * (1.0 / 16.0) + float(1.0 - prep_mixture_eps) * float(q_alpha)

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


def _estimate_sis_sequence_data(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    *,
    parallel: bool = True,
    num_samples: int = 1000,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
    ess_threshold: float = 0.5,
    proposal: str = "uniform",
    prep_mixture_eps: float = 0.1,
    prep_corr_prob: float = 0.0,
    resample: bool = False,
    basis: Literal["standard", "random_low_overlap", "tetrahedral", "random_unitary"] = "tetrahedral",
    basis_seed: int | None = None,
) -> SequenceData:
    """Internal helper used in tests: run discrete SIS and return SequenceData."""
    return _run_sis_basis_sampling(
        operator,
        sim_params,
        timesteps,
        parallel=parallel,
        num_samples=num_samples,
        noise_model=noise_model,
        seed=seed,
        ess_threshold=ess_threshold,
        proposal=proposal,
        prep_mixture_eps=prep_mixture_eps,
        resample=resample,
        basis=basis,
        basis_seed=basis_seed,
    )


def _run_exhaustive_sequence_data(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_trajectories: int = 100,
    noise_model: NoiseModel | None = None,
    basis: Literal["standard", "random_low_overlap", "tetrahedral", "random_unitary"] = "standard",
    basis_seed: int | None = None,
) -> SequenceData:
    """Runs the core simulation for exhaustive basis-sequence tomography."""
    from mqt.yaqs.analog.mcwf import preprocess_mcwf

    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True

    basis_set = get_basis_states(basis=basis, seed=basis_seed)
    choi_basis, choi_indices = get_choi_basis(basis=basis, seed=basis_seed)
    choi_duals = calculate_dual_choi_basis(choi_basis)

    k = len(timesteps)
    all_seqs = _enumerate_sequences(k)
    num_seqs = len(all_seqs)

    psi_pairs = [
        [(basis_set[choi_indices[a][1]][1], basis_set[choi_indices[a][0]][1]) for a in seq]
        for seq in all_seqs
    ]

    if noise_model is None:
        noise_model = local_params.noise_model
    if noise_model is None:
        num_trajectories = 1

    mcwf_static_ctx = None
    if local_params.solver == "MCWF":
        dummy_mps = MPS(length=operator.length, state="zeros")
        mcwf_static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, local_params)
    elif local_params.solver != "TJM":
        msg = f"Exhaustive estimator does not support solver {local_params.solver!r}."
        raise ValueError(msg)

    payload = {
        "psi_pairs": psi_pairs,
        "num_trajectories": num_trajectories,
        "operator": operator,
        "sim_params": local_params,
        "timesteps": timesteps,
        "noise_model": noise_model,
        "mcwf_static_ctx": mcwf_static_ctx,
    }

    total_jobs = num_seqs * num_trajectories
    aggregated_outputs = [np.zeros((2, 2), dtype=np.complex128) for _ in range(num_seqs)]
    aggregated_weights = np.zeros(num_seqs, dtype=np.float64)

    if parallel and total_jobs > 1:
        max_workers = max(1, available_cpus() - 1)
        for _, (s_idx, _traj_idx, rho_final, sequence_weight) in run_backend_parallel(
            worker_fn=_tomography_sequence_worker,
            payload=payload,
            n_jobs=total_jobs,
            max_workers=max_workers,
            show_progress=local_params.show_progress,
            desc="Simulating Tomography Sequences (Parallel)",
        ):
            aggregated_outputs[s_idx] += rho_final * sequence_weight
            aggregated_weights[s_idx] += sequence_weight
    else:
        disable_tqdm = not local_params.show_progress
        for j_idx in tqdm(
            range(total_jobs),
            desc="Simulating Tomography Sequences (Serial)",
            disable=disable_tqdm,
        ):
            (s_idx, _traj_idx, rho_final, sequence_weight) = _call_backend_serial(
                _tomography_sequence_worker, j_idx, payload
            )
            aggregated_outputs[s_idx] += rho_final * sequence_weight
            aggregated_weights[s_idx] += sequence_weight

    acc: dict[tuple[int, ...], list[Any]] = {}
    for i in range(num_seqs):
        acc[all_seqs[i]] = [aggregated_outputs[i], aggregated_weights[i], num_trajectories]

    final_seqs, final_outputs, final_weights = _finalize_sequence_averages(
        acc, float(num_trajectories)
    )

    return SequenceData(
        sequences=final_seqs,
        outputs=final_outputs,
        weights=final_weights,
        choi_basis=choi_basis,
        choi_indices=choi_indices,
        choi_duals=choi_duals,
        timesteps=timesteps,
    )


def run(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    *,
    method: Literal["exhaustive", "mc", "sis", "basis_sample"] = "exhaustive",
    output: Literal["dense", "mpo"] = "dense",
    noise_model: NoiseModel | None = None,
    parallel: bool = True,
    num_samples: int = 1000,
    num_trajectories: int = 100,
    seed: int | None = None,
    compress_every: int = 100,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 2,
    basis: Literal["standard", "random_low_overlap", "tetrahedral", "random_unitary"] = "tetrahedral",
    basis_seed: int | None = None,
    proposal: Literal["uniform", "local", "mixture"] = "mixture",
    ess_threshold: float = 0.5,
    prep_mixture_eps: float = 0.1,
    prep_corr_prob: float = 0.0,
    resample: bool = False,
) -> "DenseComb | MPOComb":
    """Main entry point for Process Tomography.

    Args:
        operator: The Hamiltonian (MPO) representing the system.
        sim_params: Simulation parameters (solver, dt, etc.).
        timesteps: List of durations for each evolution segment.
            If None, [sim_params.elapsed_time] is used.
        method: "exhaustive", "mc", "sis", or "basis_sample".
        output: "dense" (DenseComb) or "mpo" (MPOComb).
        noise_model: Optional noise model.
        parallel: Whether to use multi-processing.
        num_samples: Number of MC/SIS samples.
        num_trajectories: Trajectories per sequence (must be 1 for SIS).
        seed: Random seed.
        compress_every, tol, max_bond_dim, n_sweeps: MPO compression options.
        proposal: (SIS) "uniform" or "local".
        ess_threshold: (SIS) Resample when ESS < ess_threshold * N.

    Returns:
        DenseComb (output="dense") or MPOComb (output="mpo").
    """
    if timesteps is None:
        timesteps = [sim_params.elapsed_time]

    valid_solvers = {"MCWF", "TJM"}
    if sim_params.solver not in valid_solvers:
        msg = f"Tomography currently only supports solvers {valid_solvers}, got {sim_params.solver!r}."
        raise ValueError(msg)

    if method == "sis":
        if num_trajectories > 1:
            msg = "SIS currently only supports num_trajectories=1 (one realization per particle)."
            raise ValueError(msg)
        if proposal not in ["uniform", "local", "mixture"]:
            msg = (
                f"Proposal {proposal!r} is not currently supported for SIS. "
                "Use 'uniform', 'local', or 'mixture'."
            )
            raise ValueError(msg)

    if method == "exhaustive":
        data = _run_exhaustive_sequence_data(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_trajectories=num_trajectories,
            noise_model=noise_model,
            basis=basis,
            basis_seed=basis_seed,
        )
    elif method == "mc":
        # Discrete MC baseline: uniformly sample basis sequences (without replacement)
        # and apply inclusion correction. This replaces the continuous Haar-based MC sampler.
        data = _run_basis_sampling(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=num_samples,
            num_trajectories=num_trajectories,
            noise_model=noise_model,
            seed=seed,
            basis=basis,
            basis_seed=basis_seed,
        )
    elif method == "basis_sample":
        data = _run_basis_sampling(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=num_samples,
            num_trajectories=num_trajectories,
            noise_model=noise_model,
            seed=seed,
            basis=basis,
            basis_seed=basis_seed,
        )
    elif method == "sis":
        data = _run_sis_basis_sampling(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=num_samples,
            noise_model=noise_model,
            seed=seed,
            ess_threshold=ess_threshold,
            proposal=proposal,
            prep_mixture_eps=prep_mixture_eps,
            resample=resample,
            basis=basis,
            basis_seed=basis_seed,
        )
    else:
        msg = f"Unknown estimation method {method!r}."
        raise ValueError(msg)

    # Format into desired comb representation.
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

    msg = f"Unknown output format {output!r}."
    raise ValueError(msg)
