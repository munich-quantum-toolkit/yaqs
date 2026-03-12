# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Automated Process Tomography Module.

Provides functions for process tomography on a quantum system modeled by MPO
evolution. The primary entry points are:
 
* :func:`run_exact` – Full deterministic enumeration for exact ProcessTensor outcomes.
* :func:`estimate`  – Approximate estimation (SIS or MC) returning an MPO representation.
 
The module supports both exact ProcessTensor reconstruction and approximate Upsilon (MPO) estimation.

"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from tqdm import tqdm

from mqt.yaqs.analog.analog_tjm import analog_tjm_1, analog_tjm_2
from mqt.yaqs.analog.mcwf import mcwf, preprocess_mcwf
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.simulator import WORKER_CTX, available_cpus, run_backend_parallel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

from .process_tensor import ProcessTensor, rank1_upsilon_mpo_term

# ═══ Sampling & Sequence Utilities ═══════════════════════════════════════════




def _enumerate_sequences(k: int) -> list[tuple[int, ...]]:
    """Iterate over all 16^k basis sequences as a list of ``tuple[int, ...]`` (deterministic)."""
    return list(itertools.product(range(16), repeat=k))


@dataclass
class SequenceData:
    """Internal container for Stage-A tomography estimation results.

    Stage A (estimation) fills this object. Stage B (formatting) converts it
    to the desired public representation (ProcessTensor or MPO).
    """

    sequences: list[tuple[int, ...]]
    outputs: list[np.ndarray]  # (2, 2) density matrices
    weights: list[float]
    choi_basis: list[np.ndarray]
    choi_indices: list[tuple[int, int]]
    choi_duals: list[np.ndarray]
    timesteps: list[float]


# ═══ Shared Computation Helpers ═══════════════════════════════════════════════


def _call_backend_serial(backend: Callable[..., Any], *args: Any) -> Any:
    """Invoke a backend function serially with thread capping (if available)."""
    import contextlib  # noqa: PLC0415

    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        return backend(*args)

    with contextlib.suppress(Exception), threadpool_limits(limits=1):
        return backend(*args)


def _probs_for_all_particles_vec(psis, projs_batch):
    """Vectorized (N, 16) step_probs. psis: (N, n_hilbert), projs_batch: (16, 2)."""
    psi_r = psis.reshape(-1, 2, psis.shape[1] // 2)
    env = np.einsum("as,nse->nae", projs_batch.conj(), psi_r)
    return (np.abs(env) ** 2).sum(axis=-1)


def _apply_floor_prob(q, floor_eps):
    """Enforce q(a) >= floor_eps/16 then renormalize. Returns new array."""
    if floor_eps > 0:
        q = np.maximum(q, floor_eps / 16.0)
        q = q / q.sum()
    return q


def _safe_normalize_prob(q):
    """Normalize q; return None if degenerate (all-zero, non-finite)."""
    s = float(q.sum())
    if not np.isfinite(s) or s < 1e-300:
        return None
    qn = q / s
    return qn if np.all(np.isfinite(qn)) else None




def _eps_for_step(t, mixture_eps, eps_schedule):
    """Return epsilon for step t in mixture proposal."""
    if eps_schedule:
        return float(eps_schedule[min(t, len(eps_schedule) - 1)])
    return mixture_eps


def _accumulate_rank1_terms(terms, compress_every, tol, max_bond_dim, n_sweeps):
    """Accumulate an iterable of rank-1 MPO terms with periodic SVD compression.

    Args:
        terms: Iterable of rank-1 MPO objects (e.g. from rank1_upsilon_mpo_term).
        compress_every: Flush and compress after this many pending terms.
        tol: SVD truncation threshold.
        max_bond_dim: Hard cap on bond dimension (None = unlimited).
        n_sweeps: Number of SVD sweeps per compression step.

    Returns:
        Accumulated MPO.
    """
    pending: list[MPO] = []
    running: MPO | None = None

    def _flush() -> None:
        nonlocal running, pending
        if not pending:
            return
        chunk = MPO.mpo_sum(pending)
        pending.clear()
        running = chunk if running is None else running + chunk
        running.compress(tol=tol, max_bond_dim=max_bond_dim, n_sweeps=n_sweeps)

    for term in terms:
        pending.append(term)
        if len(pending) >= compress_every:
            _flush()
    _flush()
    return running  # type: ignore[return-value]


# ═══ Basis Utilities ══════════════════════════════════════════════════════════


def get_basis_states() -> list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]]:
    """Return the 4 minimal single-qubit basis states for tomography."""
    psi_0 = np.array([1, 0], dtype=complex)
    psi_1 = np.array([0, 1], dtype=complex)
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    states = [("zeros", psi_0), ("ones", psi_1), ("x+", psi_plus), ("y+", psi_i_plus)]
    return [(name, psi, np.outer(psi, psi.conj())) for name, psi in states]


def get_choi_basis() -> tuple[list[NDArray[np.complex128]], list[tuple[int, int]]]:
    r"""Generate the 16 basis CP-map Choi matrices from the 4 basis states.

    A basis CP map is A_{p,m}(rho) = Tr(E_m rho) rho_p.
    Its Choi matrix is B_{p,m} = rho_p \otimes E_m^T.
    """
    basis_set = get_basis_states()
    choi_matrices, indices = [], []
    for p, (_, _, rho_p) in enumerate(basis_set):
        for m, (_, _, e_m) in enumerate(basis_set):
            choi_matrices.append(np.kron(rho_p, e_m.T))
            indices.append((p, m))
    return choi_matrices, indices


def _finalize_sequence_averages(
    acc: dict[tuple[int, ...], list[Any]],
    weight_scale: float,
) -> tuple[list[tuple[int, ...]], list[NDArray[np.complex128]], list[float]]:
    """Consolidated logic for result collection, normalization, and weight assignment."""
    final_seqs = []
    final_outputs = []
    final_weights = []

    for seq, (rho_weighted_sum, weight_sum, count) in acc.items():
        if weight_sum > 1e-30:
            rho_avg = (rho_weighted_sum / count) / (weight_sum / count)
        else:
            rho_avg = np.zeros((2, 2), dtype=np.complex128)

        final_seqs.append(seq)
        final_outputs.append(rho_avg)

        # Standard reported sequence weights (relative to sample volume/count)
        final_weights.append(weight_sum / weight_scale)

    return final_seqs, final_outputs, final_weights


def calculate_dual_choi_basis(
    basis_matrices: list[NDArray[np.complex128]],
) -> list[NDArray[np.complex128]]:
    """Calculate the dual frame for the given Choi basis matrices."""
    frame_matrix = np.column_stack([m.reshape(-1) for m in basis_matrices])
    dual_frame = np.linalg.pinv(frame_matrix).conj().T
    dim = basis_matrices[0].shape[0]
    return [dual_frame[:, k].reshape(dim, dim) for k in range(dual_frame.shape[1])]


def _reprepare_site_zero_forced(
    mps: MPS,
    proj_state: NDArray[np.complex128],
    new_state: NDArray[np.complex128],
) -> float:
    """Project site 0 onto proj_state and reprepare new_state (in-place). Returns prob."""
    mps.set_canonical_form(orthogonality_center=0)
    t_mps = mps.tensors[0]
    env_vec = np.einsum("s c, s -> c", t_mps[:, 0, :], proj_state.conj())
    prob = float(np.linalg.norm(env_vec) ** 2)
    if prob > 1e-15:
        env_vec /= np.sqrt(prob)
    d, chi = new_state.shape[0], env_vec.shape[0]
    new_tensor = np.zeros((d, 1, chi), dtype=np.complex128)
    for s in range(d):
        new_tensor[s, 0, :] = new_state[s] * env_vec
    mps.tensors[0] = new_tensor
    final_norm = mps.norm()
    if abs(final_norm) > 1e-15:
        mps.tensors[0] /= final_norm
    return prob


def _reprepare_site_zero_vector_forced(
    state_vec: NDArray[np.complex128],
    proj_state: NDArray[np.complex128],
    new_state: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], float]:
    """Reprepare site 0 for a dense vector state. Returns (new_state_vec, prob)."""
    psi_reshaped = state_vec.reshape(2, state_vec.shape[0] // 2)
    env_vec = proj_state.conj() @ psi_reshaped
    prob = float(np.linalg.norm(env_vec) ** 2)
    if prob > 1e-15:
        env_vec /= np.sqrt(prob)
    return np.outer(new_state, env_vec).flatten(), prob


def _reconstruct_state(expectations: dict[str, float]) -> NDArray[np.complex128]:
    """Reconstruct single-qubit density matrix from Pauli expectations."""
    eye = np.eye(2, dtype=complex)
    return 0.5 * (
        eye
        + expectations["x"] * X().matrix
        + expectations["y"] * Y().matrix
        + expectations["z"] * Z().matrix
    )


def _get_rho_site_zero(state: MPS | NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Extract single-qubit (site 0) density matrix from MPS or dense vector."""
    if isinstance(state, np.ndarray):
        rho = np.reshape(state, (2, -1))
        return rho @ rho.conj().T
    assert isinstance(state, MPS)
    trace = float(state.norm() ** 2)
    if trace < 1e-15:
        return np.zeros((2, 2), dtype=np.complex128)
    rx = state.expect(Observable(X(), sites=[0]))
    ry = state.expect(Observable(Y(), sites=[0]))
    rz = state.expect(Observable(Z(), sites=[0]))
    return trace * _reconstruct_state({"x": rx / trace, "y": ry / trace, "z": rz / trace})


def _measurement_probs_site0_mps(
    state: MPS,
    basis_states: list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]],
) -> NDArray[np.float64]:
    """Efficiently compute measurement probabilities for site 0 using MPS tensors."""
    state.set_canonical_form(orthogonality_center=0)
    t0 = state.tensors[0]  # (d, 1, chi) if left edge
    # We only need the first 4 basis states (measurement states)
    amps = np.array(
        [np.einsum("sc,s->c", t0[:, 0, :], basis_states[m][1].conj()) for m in range(4)]
    )
    probs4 = np.sum(np.abs(amps) ** 2, axis=1)
    # The A_{p,m} branch prob is identical for all 4 preps p for fixed m.
    # In get_choi_basis, m is the inner loop, so we tile probs4 over p indices.
    return np.tile(probs4, 4)


def _initialize_backend_state(
    operator: MPO, solver: str
) -> MPS | NDArray[np.complex128]:
    """Initialise |0...0> state for the given solver."""
    if solver == "MCWF":
        psi = np.zeros(2**operator.length, dtype=np.complex128)
        psi[0] = 1.0
        return psi
    return MPS(length=operator.length, state="zeros")


def _reprepare_backend_state_forced(
    state: MPS | NDArray[np.complex128],
    proj_state: NDArray[np.complex128],
    new_state: NDArray[np.complex128],
    solver: str,
) -> tuple[MPS | NDArray[np.complex128], float]:
    """Reprepare site 0 for the given solver. Returns (new_state, prob)."""
    if solver == "MCWF":
        assert isinstance(state, np.ndarray)
        return _reprepare_site_zero_vector_forced(state, proj_state, new_state)
    assert isinstance(state, MPS)
    # MPS helper is in-place normally, but for SIS we often need a copy or
    # explicit return.
    new_mps = copy.deepcopy(state)
    prob = _reprepare_site_zero_forced(new_mps, proj_state, new_state)
    return new_mps, prob


def _evolve_backend_state(
    state: MPS | NDArray[np.complex128],
    operator: MPO,
    noise_model: NoiseModel | None,
    step_params: AnalogSimParams,
    solver: str,
    traj_idx: int = 0,
    static_ctx: Any = None,
) -> MPS | NDArray[np.complex128]:
    """Evolve state for one step using the given solver.

    For MCWF, static_ctx should be pre-computed via preprocess_mcwf.
    """
    if solver == "MCWF":
        if not isinstance(state, np.ndarray):
            msg = f"MCWF solver requires dense NDArray state, got {type(state)}."
            raise TypeError(msg)

        if static_ctx is None:
            # Fallback but warn/notify: local build is inefficient for particles
            dummy_mps = MPS(length=operator.length, state="zeros")
            static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, step_params)

        dynamic_ctx = copy.copy(static_ctx)
        dynamic_ctx.psi_initial = state
        dynamic_ctx.sim_params = step_params
        dynamic_ctx.output_state = None  # Clear stale state
        mcwf((traj_idx, dynamic_ctx))

        out = dynamic_ctx.output_state
        if out is None:
            msg = "MCWF backend returned None state."
            raise RuntimeError(msg)
        return cast("NDArray[np.complex128]", out)

    if not isinstance(state, MPS):
        msg = f"TJM solver requires MPS state, got {type(state)}."
        raise TypeError(msg)

    backend = analog_tjm_1 if step_params.order == 1 else analog_tjm_2
    step_params.output_state = None  # Clear stale state
    backend((traj_idx, state, noise_model, step_params, operator))

    out = step_params.output_state
    if out is None:
        msg = "TJM backend returned None state."
        raise RuntimeError(msg)
    return cast("MPS", out)


def _get_branch_probs_for_state(
    state: MPS | NDArray[np.complex128],
    basis_states: list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]],
    choi_indices: list[tuple[int, int]],
    solver: str,
) -> NDArray[np.float64]:
    """Compute branch probabilities for all 16 alphas for a single particle.

    NOTE: This is a legacy fallback path. Specialized solvers (MCWF, TJM) use
    more efficient vectorized or tensor-based extraction routines.
    """
    probs = np.zeros(16, dtype=np.float64)
    for a in range(16):
        p_idx, m_idx = choi_indices[a]
        # We only care about the probability, not the reprepared state
        _, prob = _reprepare_backend_state_forced(
            state, basis_states[m_idx][1], basis_states[p_idx][1], solver
        )
        probs[a] = prob
    return probs


# ═══ Simulation Workers ═══════════════════════════════════════════════════════


def _tomography_sequence_worker(
    job_idx: int,
    payload: dict[str, Any] | None = None,
) -> tuple[int, int, NDArray[np.complex128], float]:
    """Execute one tomography trajectory given pre-specified (psi_meas, psi_prep) pairs."""
    ctx = payload if payload is not None else WORKER_CTX
 
    num_trajectories: int = ctx["num_trajectories"]
    s_idx: int = job_idx // num_trajectories
    traj_idx: int = job_idx % num_trajectories
 
    psi_pairs = ctx["psi_pairs"][s_idx]
    operator = ctx["operator"]
    sim_params = ctx["sim_params"]
    timesteps: list[float] = ctx["timesteps"]
    noise_model = ctx["noise_model"]

    if noise_model is None:
        assert num_trajectories == 1, (
            "num_trajectories must be 1 when noise_model is None "
            "(evolution is deterministic)."
        )

    solver = sim_params.solver
    current_state = _initialize_backend_state(operator, solver)

    weight = 1.0

    for step_i, (psi_meas, psi_prep) in enumerate(psi_pairs):
        # ── Reprepare site 0 ─────────────────────────────────────────────────
        current_state, step_prob = _reprepare_backend_state_forced(
            current_state, psi_meas, psi_prep, solver
        )

        weight *= step_prob
        if weight < 1e-15:
            break

        # ── Evolve ───────────────────────────────────────────────────────────
        duration = timesteps[step_i]
        step_params = copy.deepcopy(sim_params)
        step_params.elapsed_time = duration
        step_params.num_traj = 1
        step_params.show_progress = False
        step_params.get_state = True
        n_steps = max(1, int(np.round(duration / step_params.dt)))
        step_params.times = np.linspace(0, n_steps * step_params.dt, n_steps + 1)

        static_ctx = ctx.get("mcwf_static_ctx")
        current_state = _evolve_backend_state(
            current_state,
            operator,
            noise_model,
            step_params,
            solver,
            traj_idx=traj_idx,
            static_ctx=static_ctx,
        )

    rho_final = _get_rho_site_zero(current_state)
    return (s_idx, traj_idx, rho_final, weight)


def _sis_mcwf_evolve_worker(
    job_idx: int,
    payload: dict[str, Any] | None = None,
) -> tuple[int, NDArray[np.complex128]]:
    """Parallel worker: evolve one MCWF particle for one SIS step.

    WORKER_CTX keys: mcwf_static_ctx, step_params, particle_states.
    """
    ctx = payload if payload is not None else WORKER_CTX

    particle_idx, psi_initial = ctx["particle_states"][job_idx]
    static_ctx = ctx["mcwf_static_ctx"]
    step_params = ctx["step_params"]
    dynamic_ctx = copy.copy(static_ctx)
    dynamic_ctx.psi_initial = psi_initial
    dynamic_ctx.sim_params = step_params
    dynamic_ctx.output_state = None  # Clear stale state
    mcwf((0, dynamic_ctx))
    assert dynamic_ctx.output_state is not None
    return particle_idx, dynamic_ctx.output_state  # type: ignore[return-value]


# ═══ Approximate Estimators (all return MPO) ═════════════════════════════════



def _estimate_mc_sequence_data(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_samples: int = 1000,
    num_trajectories: int = 1,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
) -> SequenceData:
    """Monte Carlo estimator (discrete ensemble)."""
    # Local copy to avoid mutation
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True

    k = len(timesteps)
    rng = np.random.default_rng(seed)

    basis_set = get_basis_states()
    choi_basis, choi_indices = get_choi_basis()
    choi_duals = calculate_dual_choi_basis(choi_basis)

    # ── Generate discrete sequences ──────────────────────────────────────────
    n_total = 16**k
    if num_samples >= n_total:
        # If exhaustive, use exact enumeration for zero variance
        alpha_seqs = _enumerate_sequences(k)
        num_samples = n_total
    else:
        # Random sampling with replacement
        alpha_seqs = [tuple(rng.integers(0, 16, size=k).tolist()) for _ in range(num_samples)]

    psi_pair_seqs = [
        [(basis_set[choi_indices[a][1]][1], basis_set[choi_indices[a][0]][1]) for a in seq]
        for seq in alpha_seqs
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
        msg = f"MC estimator does not support solver {local_params.solver!r}."
        raise ValueError(msg)

    total_jobs = num_samples * num_trajectories
    payload = {
        "psi_pairs": psi_pair_seqs,
        "num_trajectories": num_trajectories,
        "operator": operator,
        "sim_params": local_params,
        "timesteps": timesteps,
        "noise_model": noise_model,
        "mcwf_static_ctx": mcwf_static_ctx,
    }

    # acc: dict[tuple[int,...], [rho_sum, weight_sum, count]]
    acc: dict[tuple[int, ...], list[Any]] = {}

    if parallel and total_jobs > 1:
        max_workers = max(1, available_cpus() - 1)
        results_iterator = run_backend_parallel(
            worker_fn=_tomography_sequence_worker,
            payload=payload,
            n_jobs=total_jobs,
            max_workers=max_workers,
            show_progress=local_params.show_progress,
            desc=f"Simulating {num_samples} MC sequences (Parallel)",
        )
        for _, (s_idx, _traj_idx, rho_final, weight) in results_iterator:
            seq = alpha_seqs[s_idx]
            if seq not in acc:
                acc[seq] = [np.zeros((2, 2), dtype=np.complex128), 0.0, 0]
            acc[seq][0] += rho_final * weight
            acc[seq][1] += weight
            acc[seq][2] += 1
    else:
        # Serial path
        disable_tqdm = not local_params.show_progress
        for job_idx in tqdm(
            range(total_jobs), desc=f"Simulating {num_samples} MC sequences (Serial)", disable=disable_tqdm
        ):
            (s_idx, _traj_idx, rho_final, weight) = _call_backend_serial(
                _tomography_sequence_worker, job_idx, payload
            )
            seq = alpha_seqs[s_idx]
            if seq not in acc:
                acc[seq] = [np.zeros((2, 2), dtype=np.complex128), 0.0, 0]
            acc[seq][0] += rho_final * weight
            acc[seq][1] += weight
            acc[seq][2] += 1

    # For MC sampling from 16^k branches, normalizer is n/16^k to sum to 1.0. 
    # This works for exhaustive too (n=16^k -> normalizer=1.0).
    final_seqs, final_outputs, final_weights = _finalize_sequence_averages(
        acc, num_samples / (16.0**k)
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


def _estimate_sis_sequence_data(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_samples: int = 1000,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
) -> SequenceData:
    """Sequential Importance Sampling (SMC) estimator — returns SequenceData."""
    k = len(timesteps)
    # Internal defaults for advanced parameters
    proposal = "mixture"
    ess_threshold = 0.5
    resample = True
    mixture_eps = 0.1
    eps_schedule = None
    floor_eps = 0.0
    stratify_step1 = True

    # Local copy to avoid mutation
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True

    rng = np.random.default_rng(seed)

    if noise_model is None:
        noise_model = local_params.noise_model

    if local_params.solver not in {"MCWF", "TJM"}:
        msg = f"SIS requires solver in {{'MCWF', 'TJM'}}, got {local_params.solver!r}."
        raise ValueError(msg)

    # ── Basis ─────────────────────────────────────────────────────────────────
    basis_set = get_basis_states()
    choi_basis, choi_indices = get_choi_basis()
    choi_duals = calculate_dual_choi_basis(choi_basis)

    static_ctx = None
    if local_params.solver == "MCWF":
        dummy_mps = MPS(length=operator.length, state="zeros")
        static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, local_params)

    # ── Initialise particles ───────────────────────────────────────────────────
    # states: list[MPS | NDArray[np.complex128]]
    states = [
        _initialize_backend_state(operator, local_params.solver)
        for _ in range(num_samples)
    ]
    weights = np.ones(num_samples, dtype=np.float64)
    past_alpha_indices: list[list[int]] = [[] for _ in range(num_samples)]

    # ── Pre-compute projectors (16, 2) for vectorised step-prob (MCWF only) ──────
    _projs_batch = np.array(
        [basis_set[choi_indices[a][1]][1] for a in range(16)], dtype=np.complex128
    )
    max_workers = max(1, available_cpus() - 1)

    # ── SMC loop ──────────────────────────────────────────────────────────────
    for step_i, duration in enumerate(timesteps):
        eps_t = _eps_for_step(step_i, mixture_eps, eps_schedule)

        # Stratified initialisation at step 0
        if step_i == 0 and stratify_step1:
            base_count = num_samples // 16
            remainder = num_samples - 16 * base_count
            stratified_alphas: list[int] = []
            for a_val in range(16):
                cnt = base_count + (1 if a_val < remainder else 0)
                stratified_alphas.extend([a_val] * cnt)
            rng.shuffle(stratified_alphas)  # type: ignore[arg-type]
            forced_alpha0: list[int] | None = stratified_alphas
        else:
            forced_alpha0 = None

        reprepared: list[MPS | NDArray[np.complex128]] = []

        # Probability extraction
        if local_params.solver == "MCWF":
            # Optimized vectorized path for MCWF
            psis_dense = np.array(states)
            p_all_batch = _probs_for_all_particles_vec(psis_dense, _projs_batch)
        elif local_params.solver == "TJM":
            # Efficient MPS-based probability extraction for TJM
            p_all_batch = np.array(
                [_measurement_probs_site0_mps(s, basis_set) for s in states]
            )
        else:
            # Generic particle-by-particle path
            p_all_batch = np.array(
                [
                    _get_branch_probs_for_state(
                        s, basis_set, choi_indices, local_params.solver
                    )
                    for s in states
                ]
            )

        for i in range(num_samples):
            state = states[i]
            p_all = p_all_batch[i]
            z_t = float(p_all.sum())

            if forced_alpha0 is not None:
                alpha_t = forced_alpha0[i]
                p_idx, m_idx = choi_indices[alpha_t]
                state_new, step_prob = _reprepare_backend_state_forced(
                    state, basis_set[m_idx][1], basis_set[p_idx][1], local_params.solver
                )
                weights[i] *= step_prob / max(1.0 / 16.0, 1e-30)

            elif proposal == "uniform":
                alpha_t = int(rng.integers(0, 16))
                p_idx, m_idx = choi_indices[alpha_t]
                state_new, step_prob = _reprepare_backend_state_forced(
                    state, basis_set[m_idx][1], basis_set[p_idx][1], local_params.solver
                )
                weights[i] *= step_prob * 16.0

            elif proposal == "local":
                if floor_eps > 0:
                    qn = _safe_normalize_prob(
                        _apply_floor_prob(
                            p_all / z_t if z_t > 1e-300 else np.full(16, 1.0 / 16.0),
                            floor_eps,
                        )
                    )
                else:
                    qn = _safe_normalize_prob(p_all)

                if qn is None:
                    alpha_t = int(rng.integers(0, 16))
                    weights[i] *= p_all[alpha_t] / max(1.0 / 16.0, 1e-30)
                elif floor_eps > 0:
                    alpha_t = int(rng.choice(16, p=qn))
                    weights[i] *= p_all[alpha_t] / max(float(qn[alpha_t]), 1e-30)
                else:
                    alpha_t = int(rng.choice(16, p=qn))
                    weights[i] *= z_t

                p_idx, m_idx = choi_indices[alpha_t]
                state_new, _ = _reprepare_backend_state_forced(
                    state, basis_set[m_idx][1], basis_set[p_idx][1], local_params.solver
                )

            else:  # mixture
                if z_t > 1e-300:
                    q_mix = _apply_floor_prob(
                        (1.0 - eps_t) * (p_all / z_t) + eps_t / 16.0, floor_eps
                    )
                    qn = _safe_normalize_prob(q_mix)
                else:
                    qn = None

                if qn is None:
                    alpha_t = int(rng.integers(0, 16))
                    q_alpha = 1.0 / 16.0
                else:
                    alpha_t = int(rng.choice(16, p=qn))
                    q_alpha = float(qn[alpha_t])

                p_idx, m_idx = choi_indices[alpha_t]
                state_new, step_prob = _reprepare_backend_state_forced(
                    state, basis_set[m_idx][1], basis_set[p_idx][1], local_params.solver
                )
                weights[i] *= step_prob / max(q_alpha, 1e-30)

            past_alpha_indices[i].append(alpha_t)
            reprepared.append(state_new)

        # ── Evolve particles ──────────────────────────────────────────────────
        n_steps_seg = max(1, int(np.round(duration / local_params.dt)))
        step_params_seg = copy.deepcopy(local_params)
        step_params_seg.elapsed_time = duration
        step_params_seg.num_traj = 1
        step_params_seg.show_progress = False
        step_params_seg.get_state = True
        step_params_seg.times = np.linspace(
            0, n_steps_seg * local_params.dt, n_steps_seg + 1
        )

        def _is_alive(s):
            if isinstance(s, np.ndarray):
                return bool(np.linalg.norm(s) > 1e-30)
            return bool(s.norm() > 1e-30)

        alive_idxs = [i for i, s in enumerate(reprepared) if _is_alive(s)]

        if local_params.solver == "MCWF" and parallel and alive_idxs and len(alive_idxs) > 1:
            # Parallel MCWF Path
            alive_jobs = [(i, reprepared[i]) for i in alive_idxs]
            evo_payload = {
                "mcwf_static_ctx": static_ctx,
                "step_params": step_params_seg,
                "particle_states": alive_jobs,
            }
            evolved_states = [None] * num_samples
            for _, (particle_idx, evolved_psi) in run_backend_parallel(
                worker_fn=_sis_mcwf_evolve_worker,
                payload=evo_payload,
                n_jobs=len(alive_jobs),
                max_workers=max_workers,
                show_progress=False,
                desc="",
            ):
                evolved_states[particle_idx] = evolved_psi
            # Fill in newly evolved ones and preserve dead ones
            for i in range(num_samples):
                if evolved_states[i] is not None:
                    states[i] = evolved_states[i]  # type: ignore[assignment]
                else:
                    states[i] = reprepared[i]
        else:
            # Serial Path (TJM or single-sample MCWF)
            alive_set = set(alive_idxs)
            for i in range(num_samples):
                if i in alive_set:
                    states[i] = _evolve_backend_state(
                        reprepared[i],
                        operator,
                        noise_model,
                        step_params_seg,
                        local_params.solver,
                        static_ctx=static_ctx,
                    )
                else:
                    states[i] = reprepared[i]

        # ── Weight normalization & ESS check ──────────────────────────────────
        w_sum = float(weights.sum())
        if w_sum < 1e-30:
            weights[:] = 1.0
            states = [
                _initialize_backend_state(operator, local_params.solver)
                for _ in range(num_samples)
            ]
            past_alpha_indices = [[] for _ in range(num_samples)]
            continue

        w_norm = weights / w_sum
        ess = float(1.0 / np.sum(w_norm**2))

        # ── Resampling ────────────────────────────────────────────────────────
        if resample and ess < ess_threshold * num_samples:
            positions = (rng.random() + np.arange(num_samples)) / num_samples
            cumsum = np.cumsum(w_norm)
            idxs = np.searchsorted(cumsum, positions)
            states = [copy.deepcopy(states[j]) for j in idxs]
            past_alpha_indices = [past_alpha_indices[j][:] for j in idxs]
            weights[:] = w_sum / num_samples

    # acc: dict[tuple[int,...], [rho_sum, weight_sum, count]]
    acc: dict[tuple[int, ...], list[Any]] = {}
    for i in range(num_samples):
        rho_i = _get_rho_site_zero(states[i])
        seq = tuple(past_alpha_indices[i])
        wi = weights[i]

        if seq not in acc:
            acc[seq] = [np.zeros((2, 2), dtype=np.complex128), 0.0, 0]
        acc[seq][0] += rho_i * wi
        acc[seq][1] += wi
        acc[seq][2] += 1

    final_seqs, final_outputs, final_weights = _finalize_sequence_averages(
        acc, float(num_samples)
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
# ═══ Stage-B Formatters (Conversion) ══════════════════════════════════════════


def _sequence_data_to_dense(data: SequenceData) -> ProcessTensor:
    """Format SequenceData into a ProcessTensor."""
    k = len(data.timesteps)
    tensor_shape = [4] + [16] * k
    dense_data = np.zeros(tensor_shape, dtype=np.complex128)
    dense_weights = np.zeros([16] * k, dtype=np.float64)

    for i, seq in enumerate(data.sequences):
        rho = data.outputs[i]
        w = data.weights[i]
        # Store normalized density matrix; ProcessTensor methods now apply weights separately
        rho_vec = rho.reshape(-1)
        # Using slice(None) for the first dimension (4)
        dense_data[(slice(None), *seq)] = rho_vec
        dense_weights[seq] = w

    return ProcessTensor(
        tensor=dense_data,
        weights=dense_weights,
        timesteps=data.timesteps,
        choi_duals=data.choi_duals,
        choi_indices=data.choi_indices,
        choi_basis=data.choi_basis,
    )


def _sequence_data_to_mpo(
    data: SequenceData,
    compress_every: int = 100,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 2,
) -> MPO:
    """Format SequenceData into an MPO (Upsilon) representation."""
    k = len(data.timesteps)

    def _terms():
        for i, seq in enumerate(data.sequences):
            rho = data.outputs[i]
            w = data.weights[i]
            # Use elementwise conjugate here because upsilon_mpo_to_dense / prediction
            # contracts with the transpose of the inserted Choi operator.
            dual_ops = [data.choi_duals[a].conj() for a in seq]
            # Expansion in dual basis.
            yield rank1_upsilon_mpo_term(rho, dual_ops, weight=w)

    return _accumulate_rank1_terms(_terms(), compress_every, tol, max_bond_dim, n_sweeps)


# ═══ Exact Simulation & Process Tensor ════════════════════════════════════════


def _run_exact_sequence_data(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_trajectories: int = 100,
    noise_model: NoiseModel | None = None,
) -> SequenceData:
    """Runs the core simulation for exact process tensor tomography."""
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True

    basis_set = get_basis_states()
    choi_basis, choi_indices = get_choi_basis()
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
        msg = f"Exact estimator does not support solver {local_params.solver!r}."
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
        # Serial path
        disable_tqdm = not local_params.show_progress
        for j_idx in tqdm(range(total_jobs), desc="Simulating Tomography Sequences (Serial)", disable=disable_tqdm):
            (s_idx, _traj_idx, rho_final, sequence_weight) = _call_backend_serial(
                _tomography_sequence_worker, j_idx, payload
            )
            aggregated_outputs[s_idx] += rho_final * sequence_weight
            aggregated_weights[s_idx] += sequence_weight

    # acc: dict[tuple[int,...], [rho_sum, weight_sum, count]]
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


# ═══ Public API ═══════════════════════════════════════════════════════════════


def run_exact(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    *,
    output: Literal["dense", "mpo"] = "dense",
    noise_model: NoiseModel | None = None,
    parallel: bool = True,
    num_trajectories: int = 100,
) -> ProcessTensor | MPO:
    """Run exact basis-sequence tomography.

    This estimator enumerates all 16^k sequences and returns a ProcessTensor or MPO.
    """
    if timesteps is None:
        timesteps = [sim_params.elapsed_time]

    valid_solvers = {"MCWF", "TJM"}
    if sim_params.solver not in valid_solvers:
        msg = f"Tomography currently only supports solvers {valid_solvers}, got {sim_params.solver!r}."
        raise ValueError(msg)

    data = _run_exact_sequence_data(
        operator,
        sim_params,
        timesteps,
        parallel=parallel,
        num_trajectories=num_trajectories,
        noise_model=noise_model,
    )

    if output == "dense":
        return _sequence_data_to_dense(data)
    if output == "mpo":
        return _sequence_data_to_mpo(data)
    msg = f"Unknown output format {output!r}."
    raise ValueError(msg)


def estimate(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    *,
    method: Literal["sis", "mc"] = "sis",
    output: Literal["dense", "mpo"] = "mpo",
    noise_model: NoiseModel | None = None,
    parallel: bool = True,
    num_samples: int = 1000,
    num_trajectories: int = 1,
    seed: int | None = None,
) -> ProcessTensor | MPO:
    """Estimate the process using SIS or Monte Carlo sampling."""
    if timesteps is None:
        timesteps = [sim_params.elapsed_time]

    valid_solvers = {"MCWF", "TJM"}
    if sim_params.solver not in valid_solvers:
        msg = f"Solver {sim_params.solver!r} not supported for approximate estimators."
        raise ValueError(msg)

    if method == "sis":
        data = _estimate_sis_sequence_data(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=num_samples,
            noise_model=noise_model,
            seed=seed,
        )
    elif method == "mc":
        data = _estimate_mc_sequence_data(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=num_samples,
            num_trajectories=num_trajectories,
            noise_model=noise_model,
            seed=seed,
        )
    else:
        msg = f"Unknown estimation method {method!r}."
        raise ValueError(msg)

    if output == "dense":
        return _sequence_data_to_dense(data)
    if output == "mpo":
        return _sequence_data_to_mpo(data)
    msg = f"Unknown output format {output!r}."
    raise ValueError(msg)
