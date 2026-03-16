# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Automated Process Tomography Module.

Provides functions for process tomography on a quantum system modeled by MPO
evolution. The module supports exhaustive sequence enumeration as well as approximate SIS or Monte Carlo sampling.
The primary entry point is :func:`run`.

"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterable
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


def _sample_haar_pure_state(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Sample a 2-dim complex vector from the Haar measure (uniform on sphere)."""
    z = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    return z / np.linalg.norm(z)


def _sample_local_measurement_exact(
    rho_0: NDArray[np.complex128],
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Sample a qubit state from the exact local proposal density q(psi) propto <psi|rho_0|psi>.

    This uses the analytic qubit sampler derived from the eigendecomposition of rho_0.
    1. Diagonalize rho_0 = V D V^dagger.
    2. Sample t = |alpha|^2 from f(t) propto lambda_2 + (lambda_1 - lambda_2)t.
    3. Sample a uniform phase for the first coefficient.
    4. Rotate back to the computational basis.
    """
    tr_rho = float(np.trace(rho_0).real)
    if tr_rho < 1e-18:
        # Near zero state: fall back to Haar uniform
        return _sample_haar_pure_state(rng)

    rho = rho_0 / tr_rho
    evals, evecs = np.linalg.eigh(rho)
    # Ensure descending order
    idx = np.argsort(evals.real)[::-1]
    lam1, lam2 = evals.real[idx]
    v1, v2 = evecs[:, idx[0]], evecs[:, idx[1]]

    # Quadratic coefficients for sampling t = |alpha|^2
    diff = lam1 - lam2
    sum_l = lam1 + lam2

    u = rng.random()
    if abs(diff) < 1e-12:
        # Near-degenerate case: t is uniform
        t = u
    else:
        # Solve u * (lam1 + lam2)/2 = lam2*t + (diff/2)*t^2
        # (diff/2)*t^2 + (lam2)*t - u*(sum_l/2) = 0
        # a*t^2 + b*t + c = 0
        a = 0.5 * diff
        b = lam2
        c = -0.5 * u * sum_l
        # Discriminant: b^2 - 4ac
        # b^2 - 4 * (diff/2) * (-0.5 * u * sum_l) = lam2^2 + diff * u * sum_l
        disc = b**2 - 4.0 * a * c
        t = (-b + np.sqrt(max(0.0, disc))) / (2.0 * a)

    t = np.clip(t, 0.0, 1.0)
    phi = 2.0 * np.pi * rng.random()
    alpha = np.sqrt(t) * np.exp(1j * phi)
    beta = np.sqrt(1.0 - t)

    return alpha * v1 + beta * v2


def _get_haar_rho_dual(psi: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return the dual frame operator D = 2*(3|psi><psi| - I) for a Haar pure state.

    The factor of 2 is the qubit-dimension normalisation required so that
    E_{psi~Haar}[D(psi)] = I, giving an unbiased frame for the Choi estimator.
    """
    rho = np.outer(psi, psi.conj())
    return 2.0 * (3.0 * rho - np.eye(2, dtype=np.complex128))


@dataclass
class SequenceData:
    """Internal container for Stage-A tomography estimation results (Discrete)."""

    sequences: list[tuple[int, ...]]
    outputs: list[np.ndarray]  # (2, 2) density matrices
    weights: list[float]
    choi_basis: list[np.ndarray]
    choi_indices: list[tuple[int, int]]
    choi_duals: list[np.ndarray]
    timesteps: list[float]


@dataclass
class SamplingData:
    """Internal container for Stage-A tomography estimation results (Continuous)."""

    outputs: list[np.ndarray]  # (2, 2) final density matrices
    dual_ops: list[list[np.ndarray]]  # [N][k] list of 4x4 dual frame operators
    weights: list[float]
    timesteps: list[float]


def _sample_random_intervention_sequence(
    k: int, rng: np.random.Generator
) -> tuple[list[tuple[NDArray[np.complex128], NDArray[np.complex128]]], list[NDArray[np.complex128]]]:
    """Sample k pairs of (psi_meas, psi_prep) and calculate their 4x4 dual operators.

    Uses the exact convention from the original ``run_mc_upsilon``:
        D_Q  = 2*(3*Q - I)      (Q = |pp><pp|)
        D_PT = 2*(3*P.T - I)    (P = |pm><pm|)
        dual = kron(D_Q, D_PT).T
    """
    psi_pairs = []
    dual_ops = []
    for _ in range(k):
        pm = _sample_haar_pure_state(rng)
        pp = _sample_haar_pure_state(rng)
        psi_pairs.append((pm, pp))

        d_q = _get_haar_rho_dual(pp)   # 2*(3*Q - I)
        p_mat = np.outer(pm, pm.conj())
        d_pt = 2.0 * (3.0 * p_mat.T - np.eye(2, dtype=np.complex128))
        dual_ops.append(np.kron(d_q, d_pt).T)

    return psi_pairs, dual_ops


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


def _logsumexp(x: NDArray[np.float64]) -> float:
    """Numerically stable log(sum(exp(x)))."""
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return -np.inf
    return float(x_max + np.log(np.sum(np.exp(x - x_max))))


def _normalize_log_weights(log_weights: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Return normalized linear weights and log(sum w)."""
    log_w_sum = _logsumexp(log_weights)
    if not np.isfinite(log_w_sum):
        n = log_weights.shape[0]
        return np.full(n, 1.0 / n, dtype=np.float64), -np.inf
    w_norm = np.exp(log_weights - log_w_sum)
    return w_norm.astype(np.float64), log_w_sum


def _accumulate_rank1_terms(
    terms: Iterable[MPO],
    k: int,
    dims: tuple[int, int] = (2, 2),
    compress_every: int = 100,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 4,
) -> MPO:
    """Accumulate an iterable of rank-1 MPO terms with periodic SVD compression.

    Args:
        terms: Iterable of rank-1 MPO objects (e.g. from rank1_upsilon_mpo_term).
        k: Number of past legs (for zero fallback).
        dims: (output_dim, past_dim) per leg.
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
    if running is None:
        # Fallback: if all samples were skipped, return a zero MPO of the appropriate structure.
        return rank1_upsilon_mpo_term(
            np.zeros(dims, dtype=np.complex128), [np.eye(4, dtype=np.complex128)] * k, weight=0.0
        )
    return running


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


# ═══ Simulation Workers ═══════════════════════════════════════════════════════


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


# ═══ Approximate Estimators (Sampling) ════════════════════════
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
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True
    k = len(timesteps)
    rng = np.random.default_rng(seed)

    # 1. Pre-sample N sequences of k interventions + duals
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

    # Aggregate trajectories for each sample index
    # We want to keep samples separate for the ProcessTensor build
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

    # Final outputs are averaged over trajectories
    # Weight per sample is 1/N for iid sampling
    final_outputs = [out / num_trajectories for out in sample_outputs]
    final_weights = [1.0 / num_samples] * num_samples

    return SamplingData(
        outputs=final_outputs,
        dual_ops=samples_dual_ops,
        weights=final_weights,
        timesteps=timesteps,
    )




def _continuous_dual_step(
    psi_meas: NDArray[np.complex128], psi_prep: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Compute the 4×4 dual operator for one continuous intervention step.

    Exact formula from the validated MC estimator (commit b165cae):
        D_Q  = 2*(3*Q - I)      where Q = |psi_prep><psi_prep|
        D_PT = 2*(3*P.T - I)    where P = |psi_meas><psi_meas|
        return kron(D_Q, D_PT).T
    """
    d_q = _get_haar_rho_dual(psi_prep)
    p_mat = np.outer(psi_meas, psi_meas.conj())
    d_pt = 2.0 * (3.0 * p_mat.T - np.eye(2, dtype=np.complex128))
    return np.kron(d_q, d_pt).T




def _sis_evolve_worker(job_idx: int) -> MPS | NDArray[np.complex128]:
    """Worker function for parallel SIS particle evolution."""
    from ...simulator import WORKER_CTX
    
    state = WORKER_CTX["prep_states"][job_idx]
    duration = WORKER_CTX["duration"]
    static_ctx = WORKER_CTX["static_ctx"]
    sim_params = WORKER_CTX["sim_params"]
    operator = WORKER_CTX["operator"]
    noise_model = WORKER_CTX["noise_model"]
    solver = sim_params.solver
    
    # Configure simulated segment
    sp = copy.deepcopy(sim_params)
    sp.elapsed_time = duration
    sp.num_traj = 1
    sp.get_state = True
    sp.show_progress = False
    n_steps = max(1, int(np.round(duration / sp.dt)))
    sp.times = np.linspace(0, n_steps * sp.dt, n_steps + 1)
    
    return _evolve_backend_state(
        state,
        operator,
        noise_model,
        sp,
        solver,
        traj_idx=0,
        static_ctx=static_ctx,
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
) -> SamplingData:
    """SIS estimator using log-weights for stability and robust proposals.

    Args:
        operator: System Hamiltonian MPO.
        sim_params: Simulation parameters.
        timesteps: k time-step durations.
        parallel: Whether to parallelize particle evolution.
        num_samples: Number of SIS particles.
        noise_model: Optional noise model.
        seed: Random seed.
        ess_threshold: Threshold for ESS-based resampling.
        proposal: Sequence proposal method:
            - "uniform": Haar-random interventions.
            - "local": Exact qubit state-matched proposal (unbiased).
    """
    if proposal not in ["uniform", "local"]:
        msg = f"Proposal {proposal!r} is not currently supported. Use 'uniform' or 'local'."
        raise ValueError(msg)

    k = len(timesteps)
    rng = np.random.default_rng(seed)
    solver = sim_params.solver
    LOG_ZERO = -np.inf

    if noise_model is None:
        noise_model = sim_params.noise_model

    # 1. Initialize Particles & Log-Weights
    particles = [_initialize_backend_state(operator, solver) for _ in range(num_samples)]
    log_weights = np.zeros(num_samples, dtype=np.float64)
    dead = np.zeros(num_samples, dtype=bool)
    particles_dual_ops: list[list[NDArray[np.complex128]]] = [[] for _ in range(num_samples)]

    # Diagnostics
    ess_history: list[float] = []
    alive_history: list[int] = []
    resample_count = 0

    # Preprocess static context for solver efficiency
    static_ctx = None
    if solver == "MCWF":
        dummy_mps = MPS(length=operator.length, state="zeros")
        static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, sim_params)

    # Parallel settings
    use_parallel = parallel and num_samples > 32
    max_workers = available_cpus()

    for step_idx, duration in enumerate(timesteps):
        # ── 2. Intervene & Weight Update ─────────────────────────────────────
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

            if proposal == "uniform":
                pm = _sample_haar_pure_state(rng)
                pp = _sample_haar_pure_state(rng)
                psi_new, step_prob = _reprepare_backend_state_forced(particles[i], pm, pp, solver)
                if step_prob < 1e-300:
                    log_weights[i] = LOG_ZERO
                    dead[i] = True
                    continue
                log_weights[i] += np.log(step_prob)

            elif proposal == "local":
                pm = _sample_local_measurement_exact(rho_0, rng)
                pp = _sample_haar_pure_state(rng)
                psi_new, _ = _reprepare_backend_state_forced(particles[i], pm, pp, solver)
                log_weights[i] += np.log(max(tr_rho / 2.0, 1e-300))

            if np.isfinite(log_weights[i]):
                prep_states_alive.append(psi_new)
                alive_indices.append(i)
                particles_dual_ops[i].append(_continuous_dual_step(pm, pp))
            else:
                dead[i] = True

        # ── 3. Evolve Alive Particles (Parallel or Serial) ───────────────────
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
                    prep_states_alive[j], operator, noise_model, sp, solver, traj_idx=0, static_ctx=static_ctx
                )

        # ── 4. ESS-triggered Systematic Resampling ─────────────────────────────
        alive_idx = np.flatnonzero(np.isfinite(log_weights))
        if len(alive_idx) == 0:
            log_weights[:] = LOG_ZERO
            dead[:] = True
            break

        w_norm_alive, log_w_sum = _normalize_log_weights(log_weights[alive_idx])
        ess = 1.0 / float(np.sum(w_norm_alive**2))
        
        ess_history.append(ess)
        alive_history.append(len(alive_idx))

        if ess < ess_threshold * num_samples:
            resample_count += 1
            positions = (rng.random() + np.arange(num_samples)) / num_samples
            cumsum = np.cumsum(w_norm_alive)
            sel_local = np.searchsorted(cumsum, positions)
            idxs = alive_idx[sel_local]

            if solver == "MCWF":
                particles = [particles[j].copy() for j in idxs]
            else:
                particles = [copy.deepcopy(particles[j]) for j in idxs]

            particles_dual_ops = [copy.deepcopy(particles_dual_ops[j]) for j in idxs]
            
            # Reset log-weights to log(sum w) - log(N)
            new_log_weight = log_w_sum - np.log(num_samples)
            log_weights = np.full(num_samples, new_log_weight, dtype=np.float64)
            dead[:] = False

    # ── 5. Finalize Outputs ──────────────────────────────────────────────────
    # Match MC convention: outputs[i] includes particle weight, global weight = 1/N
    final_outputs = []
    for i in range(num_samples):
        if not np.isfinite(log_weights[i]):
            final_outputs.append(np.zeros((2, 2), dtype=np.complex128))
            continue
        rho_final = _get_rho_site_zero(particles[i])
        final_outputs.append(np.exp(log_weights[i]) * rho_final)

    final_weights = [1.0 / num_samples] * num_samples

    return SamplingData(
        outputs=final_outputs,
        dual_ops=particles_dual_ops,
        weights=final_weights,
        timesteps=timesteps,
    )

# ═══ Formatters ══════════════════════════════════════════
def _to_dense(data: SequenceData | SamplingData) -> ProcessTensor:
    """Format estimation results into a dense ProcessTensor.

    For SamplingData (continuous MC/SIS), reconstructs Υ = (1/N) Σ_i kron(rho_i, past_i).
    The input rho_i already incorporates the per-trajectory weight (e.g. survival prob)
    and any averaging over unravelled trajectories (1/num_traj).
    The global weight w=1/N completes the i.i.d. or importance-sampled average.
    """
    if isinstance(data, SamplingData):
        k = len(data.timesteps)
        dim = 2 * (4**k)
        ups = np.zeros((dim, dim), dtype=np.complex128)
        n = len(data.outputs)
        for rho, duals, w in zip(data.outputs, data.dual_ops, data.weights):
            if abs(w) < 1e-30 or np.linalg.norm(rho) < 1e-30:
                continue
            if len(duals) != k:
                msg = f"Nonzero sample has incomplete dual history: expected {k}, got {len(duals)}."
                raise ValueError(msg)
            past = duals[0]
            for d_step in duals[1:]:
                past = np.kron(past, d_step)
            # rho already contains the accumulated sequence/particle weight.
            # w = 1.0 / num_samples
            ups += w * np.kron(rho, past)
        return ProcessTensor.from_dense_choi(ups, data.timesteps)

    # SequenceData Path (Discrete)
    k = len(data.timesteps)
    tensor_shape = [4] + [16] * k
    dense_data = np.zeros(tensor_shape, dtype=np.complex128)
    dense_weights = np.zeros([16] * k, dtype=np.float64)

    for i, seq in enumerate(data.sequences):
        rho = data.outputs[i]
        w = data.weights[i]
        dense_data[(slice(None), *seq)] = rho.reshape(-1)
        dense_weights[seq] = w

    return ProcessTensor(
        tensor=dense_data,
        weights=dense_weights,
        timesteps=data.timesteps,
        choi_duals=data.choi_duals,
        choi_indices=data.choi_indices,
        choi_basis=data.choi_basis,
    )


def _to_mpo(
    data: SequenceData | SamplingData,
    compress_every: int = 100,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 2,
) -> MPO:
    """Format estimation results into an MPO (Upsilon) representation."""
    k = len(data.timesteps)
    if isinstance(data, SamplingData):
        def _sampling_terms():
            for i in range(len(data.outputs)):
                rho = data.outputs[i]
                w = data.weights[i]
                if abs(w) < 1e-30 or np.linalg.norm(rho) < 1e-30:
                    continue
                dual_ops = data.dual_ops[i]
                if len(dual_ops) != k:
                    msg = f"Nonzero sample has incomplete dual history: expected {k}, got {len(dual_ops)}."
                    raise ValueError(msg)
                yield rank1_upsilon_mpo_term(rho, dual_ops, weight=w)
        
        mpo = _accumulate_rank1_terms(_sampling_terms(), k=k, dims=(2, 2), compress_every=compress_every, tol=tol, max_bond_dim=max_bond_dim, n_sweeps=n_sweeps)
        return mpo

    # SequenceData Path
    def _sequence_terms():
        for i, seq in enumerate(data.sequences):
            rho = data.outputs[i]
            w = data.weights[i]
            dual_ops = [data.choi_duals[a] for a in seq]
            yield rank1_upsilon_mpo_term(rho, dual_ops, weight=w)

    return _accumulate_rank1_terms(_sequence_terms(), k=k, dims=(2, 2), compress_every=compress_every, tol=tol, max_bond_dim=max_bond_dim, n_sweeps=n_sweeps)


# ═══ Exhaustive Simulation ═══════════════════════════════════════════════════
def _run_exhaustive_sequence_data(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_trajectories: int = 100,
    noise_model: NoiseModel | None = None,
) -> SequenceData:
    """Runs the core simulation for exhaustive basis-sequence tomography."""
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
def run(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    *,
    method: Literal["exhaustive", "mc", "sis"] = "mc",
    output: Literal["dense", "mpo"] = "mpo",
    noise_model: NoiseModel | None = None,
    parallel: bool = True,
    num_samples: int = 1000,
    num_trajectories: int = 100,
    seed: int | None = None,
    compress_every: int = 100,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 2,
    proposal: Literal["uniform", "local"] = "local",
    ess_threshold: float = 0.5,
) -> ProcessTensor | MPO:
    """Main entry point for Process Tomography.

    Args:
        operator: The Hamiltonian (MPO) representing the system.
        sim_params: Simulation parameters (solver, dt, etc.).
        timesteps: List of durations for each evolution segment.
            If None, [sim_params.elapsed_time] is used.
        method: The estimation method to use:
            - "exhaustive": Enumerates all 16^k discrete sequences.
            - "mc": Continuous Monte Carlo sampling of Haar-random interventions.
            - "sis": Continuous Sequential Importance Sampling.
        output: Output format, either "dense" (ProcessTensor) or "mpo" (Upsilon MPO).
        noise_model: Optional noise model. If None, uses sim_params.noise_model.
        parallel: Whether to use multi-processing for simulation.
        num_samples: Number of MC/SIS samples (particles).
        num_trajectories: Number of trajectories per sequence (for noise unravelling).
            NOTE: For method="sis", num_trajectories MUST be 1. Each particle
            carries exactly one noisy trajectory realization, matching the 
            established particle-based sequential Monte Carlo convention.
        seed: Random seed for reproducibility.
        compress_every: (MPO only) Frequency of intermediate MPO compression.
        tol: (MPO only) Relative truncation tolerance for MPO compression.
        max_bond_dim: (MPO only) Hard bond dimension limit for MPO compression.
        n_sweeps: (MPO only) Number of sweeps for MPO accumulation.
        proposal: (SIS only) Continuous proposal distribution for interventions.
             - "uniform": Haar-random interventions (baseline).
             - "local": Exact qubit state-matched proposal (unbiased).
        ess_threshold: (SIS only) Resample when ESS < ess_threshold * N.

    Returns:
        A ProcessTensor (if output="dense") or an Upsilon-MPO (if output="mpo").
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
        if proposal not in ["uniform", "local"]:
            msg = f"Proposal {proposal!r} is not currently supported for SIS. Use 'uniform' or 'local'."
            raise ValueError(msg)

    if method == "exhaustive":
        data = _run_exhaustive_sequence_data(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_trajectories=num_trajectories,
            noise_model=noise_model,
        )
    elif method == "mc":
        data = _run_mc_sampling(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=num_samples,
            num_trajectories=num_trajectories,
            noise_model=noise_model,
            seed=seed,
        )
    elif method == "sis":
        data = _run_sis_sampling(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=num_samples,
            noise_model=noise_model,
            seed=seed,
            ess_threshold=ess_threshold,
            proposal=proposal,
        )
    else:
        msg = f"Unknown estimation method {method!r}."
        raise ValueError(msg)

    if output == "dense":
        return _to_dense(data)
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

