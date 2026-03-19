# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Automated Process Tomography Module.

Provides functions for process tomography on a quantum system modeled by MPO
evolution. The primary entry point is :func:`run`.

**Public API** (stable): :func:`run` plus re-exports ``TomographyBasis``,
``get_basis_states``, ``get_choi_basis``, ``calculate_dual_choi_basis``, and
``rank1_upsilon_mpo_term``.

**Private** (leading ``_``): implementation helpers such as :func:`_estimate`,
:func:`_estimate_sis`, :func:`_build_tomography_data`, :func:`_call_backend_serial`.
They may change without notice; tests and benchmarks import them explicitly when needed.
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
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from .combs import DenseComb, MPOComb

from .basis import (
    TomographyBasis,
    _finalize_sequence_averages,
    calculate_dual_choi_basis,
    get_basis_states,
    get_choi_basis,
)
from .estimator_class import TomographyEstimate
from .formatters import _to_dense, _to_mpo, rank1_upsilon_mpo_term
from .sampling import (
    SequenceData,
    _enumerate_sequences,
)
from .tomography_utils import (
    _evolve_backend_state,
    _get_rho_site_zero,
    _initialize_backend_state,
    _reprepare_backend_state_forced,
    _tomography_sequence_worker,
)

# Aliases used by tests
_sequence_data_to_mpo = _to_mpo
_sequence_data_to_dense = _to_dense

# Stable surface for `from module import *` and documentation.
__all__ = [
    "run",
    "TomographyBasis",
    "get_basis_states",
    "get_choi_basis",
    "calculate_dual_choi_basis",
    "rank1_upsilon_mpo_term",
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


def _estimate(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_samples: int = 1000,
    num_trajectories: int = 1,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
    basis: TomographyBasis = "standard",
    basis_seed: int | None = None,
) -> SequenceData:
    """Discrete Monte Carlo over basis-map sequences (uniform subset of ``16^k`` without replacement).

    Each selected sequence is simulated like exhaustive tomography; Horvitz–Thompson-style
    weights correct for partial sampling. If ``num_samples`` equals ``16^k``, all sequences
    are evaluated (deterministic subset; ``seed`` unused for picking).
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


def _estimate_sis(
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
    resample: bool = False,
    basis: TomographyBasis = "tetrahedral",
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
        # Identical to exhaustive: all sequences via :func:`_estimate`.
        return _estimate(
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
                else:  # mixture
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
                    if proposal == "mixture":
                        q_alpha = float(prep_mixture_eps) * (1.0 / 16.0) + float(1.0 - prep_mixture_eps) * float(
                            q_struct_alpha
                        )
                    else:
                        q_alpha = float(q_struct_alpha)

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


def _build_tomography_data(
    method: Literal["exhaustive", "estimate", "sis"],
    *,
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool,
    num_samples: int,
    num_trajectories: int,
    noise_model: NoiseModel | None,
    seed: int | None,
    basis: TomographyBasis,
    basis_seed: int | None,
    proposal: str,
    ess_threshold: float,
    prep_mixture_eps: float,
    resample: bool,
) -> SequenceData:
    """Build Stage-A tomography data (``SequenceData``) for discrete exhaustive / estimate / SIS.

    ``exhaustive`` and ``estimate`` share :func:`_estimate` (full support vs random subset of
    ``16^k`` sequences, simulated in batch via ``_tomography_sequence_worker``).

    ``sis`` uses :func:`_estimate_sis`: sequential particles with state-dependent proposals and
    importance weights; it only delegates to :func:`_estimate` when ``num_samples >= 16^k``.
    """
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
        return _estimate_sis(
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

    k = len(timesteps)
    n_all = len(_enumerate_sequences(k))
    if method == "exhaustive":
        return _estimate(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=n_all,
            num_trajectories=num_trajectories,
            noise_model=noise_model,
            seed=0,
            basis=basis,
            basis_seed=basis_seed,
        )
    if method == "estimate":
        return _estimate(
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
    msg = f"Unknown estimation method {method!r}."
    raise ValueError(msg)


def run(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    *,
    method: Literal["exhaustive", "estimate", "sis"] = "exhaustive",
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
    basis: TomographyBasis = "tetrahedral",
    basis_seed: int | None = None,
    proposal: Literal["uniform", "local", "mixture"] = "mixture",
    ess_threshold: float = 0.5,
    prep_mixture_eps: float = 0.1,
    resample: bool = False,
) -> "DenseComb | MPOComb":
    """Main entry point for Process Tomography.

    Args:
        operator: The Hamiltonian (MPO) representing the system.
        sim_params: Simulation parameters (solver, dt, etc.).
        timesteps: List of durations for each evolution segment.
            If None, [sim_params.elapsed_time] is used.
        method: ``exhaustive`` (all ``16^k`` sequences), ``estimate`` (discrete MC subset of sequences),
            or ``sis`` (sequential importance over sequences).
        output: "dense" (DenseComb) or "mpo" (MPOComb).
        noise_model: Optional noise model.
        parallel: Whether to use multi-processing.
        num_samples: Number of sequences/particles for ``estimate``/``sis`` (ignored for ``exhaustive``).
        num_trajectories: Trajectories per sequence (must be 1 for SIS).
        seed: RNG seed for subsampling sequences in ``estimate``.
        basis, basis_seed: Choi basis for ``exhaustive``, ``estimate``, and ``sis``.
        compress_every, tol, max_bond_dim, n_sweeps: MPO compression options.
        proposal: (SIS) ``uniform``, ``local``, or ``mixture``.
        ess_threshold: SIS-only parameter (discrete SIS does not resample).

    Returns:
        DenseComb (output="dense") or MPOComb (output="mpo").
    """
    if timesteps is None:
        timesteps = [sim_params.elapsed_time]

    valid_solvers = {"MCWF", "TJM"}
    if sim_params.solver not in valid_solvers:
        msg = f"Tomography currently only supports solvers {valid_solvers}, got {sim_params.solver!r}."
        raise ValueError(msg)

    data = _build_tomography_data(
        method,
        operator=operator,
        sim_params=sim_params,
        timesteps=timesteps,
        parallel=parallel,
        num_samples=num_samples,
        num_trajectories=num_trajectories,
        noise_model=noise_model,
        seed=seed,
        basis=basis,
        basis_seed=basis_seed,
        proposal=proposal,
        ess_threshold=ess_threshold,
        prep_mixture_eps=prep_mixture_eps,
        resample=resample,
    )

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
