# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Automated Process Tomography Module.

Provides functions for process tomography on a quantum system modeled by MPO
evolution. The primary entry point is the :func:`run` function, which
dispatches to three internal methods:

* ``"sis"``   – Sequential Importance Sampling (most efficient).
* ``"mc"``    – Plain Monte Carlo (discrete or continuous ensemble baseline).
* ``"exact"`` – Full deterministic enumeration (default).

The :func:`run` function supports multiple output formats including
:class:`ProcessTensor`, ``MPO``, or dense matrices.

"""

from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

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

from .process_tensor import ProcessTensor, rank1_upsilon_mpo_term, upsilon_mpo_to_dense

# ═══ Sampling & Sequence Utilities ═══════════════════════════════════════════


def _random_pure_state(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Generate a random pure state vector (continuous-ensemble tomography)."""
    z = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    z /= np.linalg.norm(z)
    return z


def _continuous_dual(
    psi_meas: NDArray[np.complex128],
    psi_prep: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Return the 4x4 dual operator for a continuous (psi_meas, psi_prep) pair."""
    P = np.outer(psi_meas, psi_meas.conj())
    Q = np.outer(psi_prep, psi_prep.conj())
    D_Q = 2.0 * (3.0 * Q - np.eye(2, dtype=np.complex128))
    D_PT = 2.0 * (3.0 * P.T - np.eye(2, dtype=np.complex128))
    return np.kron(D_Q, D_PT).T


def _enumerate_sequences(k: int):
    """Iterate over all 16^k basis sequences as ``tuple[int, ...]`` (deterministic)."""
    return itertools.product(range(16), repeat=k)


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


def _apply_dual_transform(D: NDArray[np.complex128], transform: str) -> NDArray[np.complex128]:
    """Apply a linear transformation to dual matrix D."""
    if transform == "id":
        return D
    if transform == "T":
        return D.T
    if transform == "conj":
        return D.conj()
    if transform == "dag":
        return D.conj().T
    msg = f"Invalid dual_transform: {transform!r}. Must be one of {{'id', 'T', 'conj', 'dag'}}."
    raise ValueError(msg)


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

    is_mcwf: bool = sim_params.solver == "MCWF"
    current_state: MPS | NDArray[np.complex128]
 
    if is_mcwf:
        current_state = np.zeros(2**operator.length, dtype=np.complex128)
        current_state[0] = 1.0
    else:
        current_state = MPS(length=operator.length, state="zeros")

    weight = 1.0

    for step_i, (psi_meas, psi_prep) in enumerate(psi_pairs):
        # ── Reprepare site 0 ─────────────────────────────────────────────────
        if is_mcwf:
            assert isinstance(current_state, np.ndarray)
            current_state, step_prob = _reprepare_site_zero_vector_forced(
                current_state, psi_meas, psi_prep
            )
        else:
            assert isinstance(current_state, MPS)
            step_prob = _reprepare_site_zero_forced(current_state, psi_meas, psi_prep)

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

        if is_mcwf:
            static_ctx = ctx.get("mcwf_static_ctx")
            if static_ctx is None:
                # Fallback if no static ctx (serial path usually prepares it locally)
                dummy_mps = MPS(length=operator.length, state="zeros")
                static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, step_params)
 
            dynamic_ctx = copy.copy(static_ctx)
            dynamic_ctx.psi_initial = current_state
            dynamic_ctx.sim_params = step_params
            mcwf((traj_idx, dynamic_ctx))
            assert dynamic_ctx.output_state is not None
            current_state = cast("NDArray[np.complex128]", dynamic_ctx.output_state)
        else:
            backend = analog_tjm_1 if step_params.order == 1 else analog_tjm_2
            assert isinstance(current_state, MPS)
            backend((traj_idx, current_state, noise_model, step_params, operator))
            assert step_params.output_state is not None
            current_state = cast("MPS", step_params.output_state)

    rho_final = _get_rho_site_zero(current_state)
    return (s_idx, traj_idx, rho_final, weight)


def _sis_evolve_worker(
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
    mcwf((0, dynamic_ctx))
    assert dynamic_ctx.output_state is not None
    return particle_idx, dynamic_ctx.output_state  # type: ignore[return-value]


# ═══ Core Estimators (all return MPO) ════════════════════════════════════════


def _run_exact_mpo(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_trajectories: int = 1,
    noise_model: NoiseModel | None = None,
) -> MPO:
    """Deterministic exact estimator: enumerate all 16^k sequences.
    
    If noise_model is provided and num_trajectories > 1, this evaluates the
    exact sum over intervention sequences, but computes a Monte Carlo average
    over the stochastic MCWF trajectories for each sequence.
    """
    # Internal defaults for advanced parameters
    compress_every = 100
    tol = 1e-12
    max_bond_dim = None
    n_sweeps = 2
 
    # Local copy to avoid mutation
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True
 
    k = len(timesteps)
    basis_set = get_basis_states()
    _cb, choi_indices = get_choi_basis()
    duals = calculate_dual_choi_basis(_cb)
 
    all_seqs = list(_enumerate_sequences(k))
    num_seqs = len(all_seqs)  # 16^k
 
    # Convert alpha sequences to (psi_meas, psi_prep) pairs
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
 
    total_jobs = num_seqs * num_trajectories
    payload = {
        "psi_pairs": psi_pairs,
        "num_trajectories": num_trajectories,
        "operator": operator,
        "sim_params": local_params,
        "timesteps": timesteps,
        "noise_model": noise_model,
        "mcwf_static_ctx": mcwf_static_ctx,
    }
 
    raw_sum = np.zeros((num_seqs, 2, 2), dtype=np.complex128)
 
    if parallel and total_jobs > 1:
        max_workers = max(1, available_cpus() - 1)
        results_iterator = run_backend_parallel(
            worker_fn=_tomography_sequence_worker,
            payload=payload,
            n_jobs=total_jobs,
            max_workers=max_workers,
            show_progress=local_params.show_progress,
            desc=f"Enumerating {num_seqs} sequences (Parallel)",
        )
        for _, (s_idx, _traj_idx, rho_final, weight) in results_iterator:
            raw_sum[s_idx] += rho_final * weight
    else:
        # Serial path
        disable_tqdm = not local_params.show_progress
        for j_idx in tqdm(range(total_jobs), desc=f"Enumerating {num_seqs} sequences (Serial)", disable=disable_tqdm):
            (s_idx, _traj_idx, rho_final, weight) = _call_backend_serial(
                _tomography_sequence_worker, j_idx, payload
            )
            raw_sum[s_idx] += rho_final * weight

    raw_sum /= num_trajectories
 
    def _terms():
        for s_idx, alpha_seq in enumerate(all_seqs):
            dual_ops = [_apply_dual_transform(duals[a], "conj") for a in alpha_seq]
            yield rank1_upsilon_mpo_term(raw_sum[s_idx], dual_ops, weight=1.0)
 
    return _accumulate_rank1_terms(_terms(), compress_every, tol, max_bond_dim, n_sweeps)


def _run_mc_mpo(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_samples: int = 1000,
    num_trajectories: int = 1,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
    sampling: str = "discrete",
    replace: bool = True,
) -> MPO:
    """Monte Carlo estimator (discrete or continuous ensemble)."""
    # Internal defaults for advanced parameters
    compress_every = 100
    tol = 1e-12
    max_bond_dim = None
    n_sweeps = 2
 
    # Local copy to avoid mutation
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True
 
    k = len(timesteps)
    rng = np.random.default_rng(seed)
 
    basis_set = get_basis_states()
    _cb, choi_indices = get_choi_basis()
    duals = calculate_dual_choi_basis(_cb)
 
    if sampling not in {"discrete", "continuous"}:
        msg = f"sampling must be 'discrete' or 'continuous', got {sampling!r}."
        raise ValueError(msg)
 
    # ── Generate sequences ────────────────────────────────────────────────────
    if sampling == "discrete":
        if replace:
            alpha_seqs: list[tuple[int, ...]] = [
                tuple(rng.integers(0, 16, size=k).tolist()) for _ in range(num_samples)
            ]
        else:
            total = 16**k
            num_samples = min(num_samples, total)
            sampled_indices = rng.choice(total, size=num_samples, replace=False)
            alpha_seqs = []
            for idx in sampled_indices:
                seq, temp = [], int(idx)
                for _ in range(k):
                    seq.append(temp % 16)
                    temp //= 16
                alpha_seqs.append(tuple(reversed(seq)))
 
        psi_pair_seqs = [
            [(basis_set[choi_indices[a][1]][1], basis_set[choi_indices[a][0]][1]) for a in seq]
            for seq in alpha_seqs
        ]
        inv_q = float(16**k)
    else:  # continuous
        psi_pair_seqs = [
            [(_random_pure_state(rng), _random_pure_state(rng)) for _ in range(k)]
            for _ in range(num_samples)
        ]
        alpha_seqs = None  # type: ignore[assignment]
        inv_q = 1.0  # Haar duals self-normalize
 
    if noise_model is None:
        noise_model = local_params.noise_model
    if noise_model is None:
        num_trajectories = 1
 
    mcwf_static_ctx = None
    if local_params.solver == "MCWF":
        dummy_mps = MPS(length=operator.length, state="zeros")
        mcwf_static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, local_params)
 
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
 
    raw_sum = np.zeros((num_samples, 2, 2), dtype=np.complex128)
 
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
            raw_sum[s_idx] += rho_final * weight
    else:
        # Serial path
        disable_tqdm = not local_params.show_progress
        for job_idx in tqdm(range(total_jobs), desc=f"Simulating {num_samples} MC sequences (Serial)", disable=disable_tqdm):
            (s_idx, _traj_idx, rho_final, weight) = _call_backend_serial(
                _tomography_sequence_worker, job_idx, payload
            )
            raw_sum[s_idx] += rho_final * weight
 
    raw_sum /= num_trajectories  # average over trajectories
 
    def _terms():
        for s_idx in range(num_samples):
            if sampling == "discrete":
                dual_ops = [
                    _apply_dual_transform(duals[a], "conj")
                    for a in alpha_seqs[s_idx]
                ]
                w = inv_q / num_samples
            else:
                dual_ops = [
                    _apply_dual_transform(_continuous_dual(pm, pp), "conj")
                    for pm, pp in psi_pair_seqs[s_idx]
                ]
                w = inv_q / num_samples
            yield rank1_upsilon_mpo_term(raw_sum[s_idx], dual_ops, weight=w)
 
    return _accumulate_rank1_terms(_terms(), compress_every, tol, max_bond_dim, n_sweeps)


def _run_sis_mpo(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_particles: int = 1000,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
) -> MPO:
    """Sequential Importance Sampling (SMC) estimator — returns MPO."""
    # Internal defaults for advanced parameters
    proposal = "mixture"
    ess_threshold = 0.5
    resample = True
    mixture_eps = 0.1
    eps_schedule = None
    floor_eps = 0.0
    stratify_step1 = True
    compress_every = 100
    tol = 1e-12
    max_bond_dim = None
    n_sweeps = 2
 
    # Local copy to avoid mutation
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True
 
    rng = np.random.default_rng(seed)
 
    if noise_model is None:
        noise_model = local_params.noise_model
 
    if local_params.solver != "MCWF":
        raise ValueError(f"SIS requires solver='MCWF', got {local_params.solver!r}.")
 
    # ── Basis ─────────────────────────────────────────────────────────────────
    basis_set = get_basis_states()
    _cb, choi_indices = get_choi_basis()
    duals = calculate_dual_choi_basis(_cb)
    dual_mats = [_apply_dual_transform(duals[a], "conj") for a in range(16)]
 
    dummy_mps = MPS(length=operator.length, state="zeros")
    static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, local_params)
 
    n_hilbert = 2**operator.length
 
    # ── Initialise particles ───────────────────────────────────────────────────
    psi_zero = np.zeros(n_hilbert, dtype=np.complex128)
    psi_zero[0] = 1.0
    states: NDArray[np.complex128] = np.tile(psi_zero, (num_particles, 1))
    weights = np.ones(num_particles, dtype=np.float64)
    past_local_ops: list[list[NDArray[np.complex128]]] = [[] for _ in range(num_particles)]
 
    # ── Pre-compute projectors (16, 2) for vectorised step-prob ───────────────
    _projs_batch = np.array(
        [basis_set[choi_indices[a][1]][1] for a in range(16)], dtype=np.complex128
    )
    max_workers = max(1, available_cpus() - 1)
 
    # ── SMC loop ──────────────────────────────────────────────────────────────
    for step_i, duration in enumerate(timesteps):
        eps_t = _eps_for_step(step_i, mixture_eps, eps_schedule)
 
        # Stratified initialisation at step 0
        if step_i == 0 and stratify_step1:
            base_count = num_particles // 16
            remainder = num_particles - 16 * base_count
            stratified_alphas: list[int] = []
            for a_val in range(16):
                cnt = base_count + (1 if a_val < remainder else 0)
                stratified_alphas.extend([a_val] * cnt)
            rng.shuffle(stratified_alphas)  # type: ignore[arg-type]
            forced_alpha0: list[int] | None = stratified_alphas
        else:
            forced_alpha0 = None
 
        reprepared: list[NDArray[np.complex128]] = []
        p_all_vec = _probs_for_all_particles_vec(states, _projs_batch)  # (N, 16)
 
        for i in range(num_particles):
            psi = states[i]
            p_all = p_all_vec[i]
            z_t = float(p_all.sum())
 
            if forced_alpha0 is not None:
                alpha_t = forced_alpha0[i]
                p_idx, m_idx = choi_indices[alpha_t]
                psi_new, step_prob = _reprepare_site_zero_vector_forced(
                    psi, basis_set[m_idx][1], basis_set[p_idx][1]
                )
                weights[i] *= step_prob / max(1.0 / 16.0, 1e-30)
 
            elif proposal == "uniform":
                alpha_t = int(rng.integers(0, 16))
                p_idx, m_idx = choi_indices[alpha_t]
                psi_new, step_prob = _reprepare_site_zero_vector_forced(
                    psi, basis_set[m_idx][1], basis_set[p_idx][1]
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
                psi_new, _ = _reprepare_site_zero_vector_forced(
                    psi, basis_set[m_idx][1], basis_set[p_idx][1]
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
                psi_new, step_prob = _reprepare_site_zero_vector_forced(
                    psi, basis_set[m_idx][1], basis_set[p_idx][1]
                )
                weights[i] *= step_prob / max(q_alpha, 1e-30)
 
            past_local_ops[i].append(dual_mats[alpha_t].copy())
            reprepared.append(psi_new)
 
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
 
        alive_jobs = [
            (i, reprepared[i])
            for i in range(num_particles)
            if np.linalg.norm(reprepared[i]) > 1e-30
        ]
 
        new_states = np.zeros_like(states)
        if parallel and alive_jobs and len(alive_jobs) > 1:
            evo_payload = {
                "mcwf_static_ctx": static_ctx,
                "step_params": step_params_seg,
                "particle_states": alive_jobs,
            }
            for _, (particle_idx, evolved_psi) in run_backend_parallel(
                worker_fn=_sis_evolve_worker,
                payload=evo_payload,
                n_jobs=len(alive_jobs),
                max_workers=max_workers,
                show_progress=False,
                desc="",
            ):
                new_states[particle_idx] = evolved_psi
        else:
            # Serial path
            for i, psi_new in enumerate(reprepared):
                if np.linalg.norm(psi_new) > 1e-30:
                    # Reuse existing _sis_evolve_worker for consistency
                    local_evo_payload = {
                        "mcwf_static_ctx": static_ctx,
                        "step_params": step_params_seg,
                        "particle_states": [(i, psi_new)],
                    }
                    (particle_idx, evolved_psi) = _call_backend_serial(
                        _sis_evolve_worker, 0, local_evo_payload
                    )
                    new_states[i] = evolved_psi
 
        states = new_states
 
        # ── Weight normalization & ESS check ──────────────────────────────────
        w_sum = float(weights.sum())
        if w_sum < 1e-30:
            weights[:] = 1.0
            states = np.tile(psi_zero, (num_particles, 1))
            past_local_ops = [[] for _ in range(num_particles)]
            continue
 
        w_norm = weights / w_sum
        ess = float(1.0 / np.sum(w_norm**2))
 
        # ── Resampling ────────────────────────────────────────────────────────
        if resample and ess < ess_threshold * num_particles:
            positions = (rng.random() + np.arange(num_particles)) / num_particles
            cumsum = np.cumsum(w_norm)
            idxs = np.searchsorted(cumsum, positions)
            states = states[idxs]
            past_local_ops = [past_local_ops[j][:] for j in idxs]
            weights[:] = w_sum / num_particles
 
    # ── Build Upsilon MPO ─────────────────────────────────────────────────────
    def _terms():
        for i in range(num_particles):
            rho_mat = np.reshape(states[i], (2, -1))
            rho_i = rho_mat @ rho_mat.conj().T
            wi = weights[i] / float(num_particles)
            yield rank1_upsilon_mpo_term(rho_i, past_local_ops[i], weight=wi)
 
    return _accumulate_rank1_terms(_terms(), compress_every, tol, max_bond_dim, n_sweeps)


# ═══ Public API ═══════════════════════════════════════════════════════════════


def _run_exact_process_tensor(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    parallel: bool = True,
    num_trajectories: int = 100,
    noise_model: NoiseModel | None = None,
) -> ProcessTensor:
    """Internal implementation of exact basis-injection Process Tensor tomography."""
    # Local copy to avoid mutating user sim_params
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True

    basis_set = get_basis_states()
    choi_basis, choi_indices = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    k = len(timesteps)
    all_seqs = list(itertools.product(range(16), repeat=k))
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

    for i in range(num_seqs):
        aggregated_outputs[i] /= num_trajectories
        aggregated_weights[i] /= num_trajectories

    tensor_shape = [4] + [16] * k
    process_tensor_data = np.zeros(tensor_shape, dtype=np.complex128)
    weights_shape = [16] * k
    process_tensor_weights = np.zeros(weights_shape, dtype=np.float64)

    for s_idx, avg_rho in enumerate(aggregated_outputs):
        seq_tuple = all_seqs[s_idx]
        rho_vec = avg_rho.reshape(-1)
        idx = (slice(None), *seq_tuple)
        process_tensor_data[idx] = rho_vec
        process_tensor_weights[seq_tuple] = aggregated_weights[s_idx]

    return ProcessTensor(
        tensor=process_tensor_data,
        weights=process_tensor_weights,
        timesteps=timesteps,
        choi_duals=duals,
        choi_indices=choi_indices,
        choi_basis=choi_basis,
    )



 
 
def run(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    *,
    noise_model: NoiseModel | None = None,
    method: Literal["exact", "mc", "sis"] = "exact",
    output: Literal["process_tensor", "mpo", "dense"] = "process_tensor",
    parallel: bool = True,
    num_samples: int = 1000,
    num_trajectories: int = 100,
    seed: int | None = None,
    sampling: str = "discrete",
    replace: bool = True,
) -> ProcessTensor | MPO | NDArray[np.complex128]:
    """Run process tomography on a quantum system.
 
    This is the primary entry-point for the tomography module, supporting exact
    basis-injection as well as sampling-based (MC, SIS) estimation.
    """
    if timesteps is None:
        timesteps = [sim_params.elapsed_time]
        
    valid_solvers = {"MCWF", "TJM"}
    if sim_params.solver not in valid_solvers:
        msg = f"Tomography currently only supports solvers {valid_solvers}, got {sim_params.solver!r}."
        raise ValueError(msg)
 
    if output == "process_tensor" and method != "exact":
        msg = f"output='process_tensor' is only supported for method='exact', got method={method!r}."
        raise ValueError(msg)
 
    if method == "exact":
        if output == "process_tensor":
            res = _run_exact_process_tensor(
                operator,
                sim_params,
                timesteps,
                parallel=parallel,
                num_trajectories=num_trajectories,
                noise_model=noise_model,
            )
        else:
            res = _run_exact_mpo(
                operator,
                sim_params,
                timesteps,
                parallel=parallel,
                num_trajectories=num_trajectories,
                noise_model=noise_model,
            )
    elif method == "mc":
        res = _run_mc_mpo(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_samples=num_samples,
            num_trajectories=num_trajectories,
            noise_model=noise_model,
            seed=seed,
            sampling=sampling,
            replace=replace,
        )
    elif method == "sis":
        res = _run_sis_mpo(
            operator,
            sim_params,
            timesteps,
            parallel=parallel,
            num_particles=num_samples,
            noise_model=noise_model,
            seed=seed,
        )
    else:
        msg = f"method must be 'sis', 'mc', or 'exact', got {method!r}."
        raise ValueError(msg)
 
    if output == "dense":
        return upsilon_mpo_to_dense(res)
 
    return res
