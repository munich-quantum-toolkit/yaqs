# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""
Shared helpers for the channel-predictor NN benchmarks.

Goal: learn (vec(rho_in), Choi(E1), ..., Choi(Ek)) -> vec(rho_out)
from backend-simulated labels.

This intentionally contains only small, testable utilities (no main entrypoint).
"""

from __future__ import annotations

import copy
from typing import Any, cast

import numpy as np

from mqt.yaqs.analog.mcwf import preprocess_mcwf
from mqt.yaqs.characterization.tomography.basis import (
    TomographyBasis,
    get_basis_states,
    get_choi_basis,
)
from mqt.yaqs.characterization.tomography.process_tomography import (
    _call_backend_serial,
)
from mqt.yaqs.characterization.tomography.tomography_utils import (
    _evolve_backend_state,
    _get_rho_site_zero,
    _reprepare_backend_state_forced,
)
from mqt.yaqs.core.data_structures.networks import MPS, MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.characterization.tomography.ml_dataset import trace_distance
from mqt.yaqs.simulator import available_cpus, run_backend_parallel


CHOI_FLAT_DIM = 32  # 4x4 complex -> 16 complex -> 32 reals (Re/Im, row-major)


def pack_rho8(rho: np.ndarray) -> np.ndarray:
    """Unweighted 2x2 density matrix as 8 floats (Re/Im, row-major)."""
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    return np.array(
        [
            r[0, 0].real,
            r[0, 0].imag,
            r[0, 1].real,
            r[0, 1].imag,
            r[1, 0].real,
            r[1, 0].imag,
            r[1, 1].real,
            r[1, 1].imag,
        ],
        dtype=np.float32,
    )


def unpack_rho8(y: np.ndarray) -> np.ndarray:
    """Convert 8 reals back to a Hermitian 2x2 density matrix."""
    t = np.asarray(y, dtype=np.float64).reshape(8)
    rho = np.array(
        [
            [t[0] + 1j * t[1], t[2] + 1j * t[3]],
            [t[4] + 1j * t[5], t[6] + 1j * t[7]],
        ],
        dtype=np.complex128,
    )
    return 0.5 * (rho + rho.conj().T)


def normalize_rho_like_densecomb(rho: np.ndarray) -> np.ndarray:
    """Match DenseComb.predict convention: Hermitize + trace/PSD projection."""
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 1e-12:
        rho = rho / tr

    w, V = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    rho = (V * w) @ V.conj().T

    tr2 = np.trace(rho)
    if abs(tr2) > 1e-15:
        rho = rho / tr2
    return rho


def random_density_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a random valid 2x2 density matrix."""
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = a @ a.conj().T
    tr = float(np.trace(rho).real)
    rho = rho / max(tr, 1e-15)
    return 0.5 * (rho + rho.conj().T)


def flatten_choi4_to_real32(j: np.ndarray) -> np.ndarray:
    """Vectorize 4x4 complex Choi matrix to 32 floats (Re/Im, row-major)."""
    m = np.asarray(j, dtype=np.complex128).reshape(4, 4)
    flat = m.reshape(-1)
    interleaved = np.stack([flat.real, flat.imag], axis=-1).astype(np.float32)
    return interleaved.reshape(-1)


def build_choi_feature_table(choi_matrices: list[np.ndarray]) -> np.ndarray:
    """Return array of shape (16, 32): one feature row per fixed-basis index."""
    rows = [flatten_choi4_to_real32(c) for c in choi_matrices]
    return np.stack(rows, axis=0)


def concat_choi_features(alphas: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Concatenate `table[alpha_t]` for each step; table is (16, 32)."""
    a = np.asarray(alphas, dtype=np.int64).reshape(-1)
    parts = [table[int(ai)] for ai in a]
    return np.concatenate(parts, axis=0).astype(np.float32)


def intervention_from_alpha(
    alpha: int,
    basis_set: list[tuple[str, np.ndarray, np.ndarray]],
    choi_pm_pairs: list[tuple[int, int]],
) -> Any:
    """Build map E_alpha(rho) = Tr(E_m rho) * rho_p from fixed basis tables."""
    p, m = choi_pm_pairs[int(alpha)]
    rho_p = np.asarray(basis_set[p][2], dtype=np.complex128)
    e_m = np.asarray(basis_set[m][2], dtype=np.complex128)

    def emap(rho: np.ndarray) -> np.ndarray:
        r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
        coeff = np.trace(e_m @ r)
        return coeff * rho_p

    return emap


def mean_trace_distance_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Mean trace distance on batches of rho8 vectors."""
    assert pred_rho8.shape == tgt_rho8.shape
    tds: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        tds.append(trace_distance(rp, rt))
    return float(np.mean(tds))


def mean_frobenius_mse_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Mean squared Frobenius error: E[||rho_pred - rho_tgt||_F^2]."""
    assert pred_rho8.shape == tgt_rho8.shape
    diffs: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        d = rp - rt
        diffs.append(float(np.real(np.vdot(d, d))))
    return float(np.mean(diffs))


def normalize_rho_from_backend_output(rho_final: Any) -> np.ndarray:
    """DenseComb-like normalization for backend outputs."""
    rho_h = np.asarray(rho_final, dtype=np.complex128).reshape(2, 2)
    rho_h = 0.5 * (rho_h + rho_h.conj().T)
    return normalize_rho_like_densecomb(rho_h)


def worker_initial_psi(job_idx: int, payload: dict[str, Any] | None = None) -> tuple[int, int, np.ndarray, float]:
    """Backend worker: starts from `initial_psi[s_idx]` and applies psi-pairs."""
    ctx = payload if payload is not None else {}
    num_trajectories: int = int(ctx["num_trajectories"])
    s_idx = job_idx // num_trajectories
    traj_idx = job_idx % num_trajectories

    psi_pairs = ctx["psi_pairs"][s_idx]
    operator = ctx["operator"]
    sim_params: AnalogSimParams = ctx["sim_params"]
    timesteps: list[float] = ctx["timesteps"]
    noise_model = ctx["noise_model"]
    initial_list: list[np.ndarray] = ctx["initial_psi"]

    if noise_model is None:
        assert num_trajectories == 1, "num_trajectories must be 1 when noise_model is None."

    solver = sim_params.solver
    current_state = np.asarray(initial_list[s_idx], dtype=np.complex128).copy()
    weight = 1.0

    for step_i, (psi_meas, psi_prep) in enumerate(psi_pairs):
        current_state, step_prob = _reprepare_backend_state_forced(
            current_state, psi_meas, psi_prep, solver
        )
        weight *= step_prob
        if weight < 1e-15:
            break

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


def build_initial_psi(
    rho_in: np.ndarray,
    *,
    length: int,
    rng: np.random.Generator,
    init_mode: str,
    return_eig_sample: bool = False,
) -> Any:
    """Backend pure-state initialization matching the benchmark conventions."""
    return _initial_mcwf_state_from_rho0(
        rho_in,
        length,
        rng=rng,
        init_mode=init_mode,
        return_eig_sample=return_eig_sample,
    )


def _initial_mcwf_state_from_rho0(
    rho: np.ndarray,
    length: int,
    *,
    rng: np.random.Generator | None = None,
    init_mode: str = "eigenstate",
    return_eig_sample: bool = False,
) -> Any:
    """Initialize a *pure* MCWF state on ``length`` qubits matching the requested input.

    init_mode:
      - ``"eigenstate"`` (default): sample one eigenvector |v_i> of rho_in with probability p_i and set
        |psi_in> = |v_i> ⊗ |0...0> (no entanglement between site 0 and site 1).
      - ``"purified"``: old purification trick (entangles site 0 with an ancilla on site 1).
    """
    rho = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    rho = 0.5 * (rho + rho.conj().T)
    w, v = np.linalg.eigh(rho)
    w = np.maximum(w.real, 0.0)
    s = float(w.sum())
    if s > 1e-15:
        w = w / s
    else:
        w = np.array([1.0, 0.0], dtype=np.float64)

    if init_mode not in {"eigenstate", "purified"}:
        raise ValueError(f"init_mode must be 'eigenstate' or 'purified', got {init_mode!r}")

    if init_mode == "eigenstate":
        if rng is None:
            rng = np.random.default_rng()
        idx = int(rng.choice(2, p=w))
        p = float(w[idx])
        v_idx = v[:, idx].astype(np.complex128)
        if length <= 1:
            psi = v_idx
        else:
            env0 = np.array([1.0, 0.0], dtype=np.complex128)
            env_state = env0
            for _ in range(length - 2):
                env_state = np.kron(env_state, env0)
            psi = np.kron(v_idx, env_state)
        if return_eig_sample:
            return psi, idx, p
        return psi

    # init_mode == "purified" (previous behavior)
    if length <= 1:
        psi = np.zeros(2, dtype=np.complex128)
        for i in range(2):
            if w[i] > 1e-15:
                psi += np.sqrt(w[i]) * v[:, i].astype(np.complex128)
        nrm = float(np.linalg.norm(psi))
        psi = psi / max(nrm, 1e-15)
        return (psi, 0, float(w[0])) if return_eig_sample else psi

    psi_2 = np.zeros(4, dtype=np.complex128)
    for i in range(2):
        if w[i] < 1e-15:
            continue
        anc = np.zeros(2, dtype=np.complex128)
        anc[i] = 1.0
        psi_2 += np.sqrt(w[i]) * np.kron(v[:, i].astype(np.complex128), anc)
    nrm = float(np.linalg.norm(psi_2))
    if nrm < 1e-15:
        psi_2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    else:
        psi_2 /= nrm
    psi = psi_2
    for _ in range(length - 2):
        psi = np.kron(psi, np.array([1.0, 0.0], dtype=np.complex128))
    return (psi, 0, float(w[0])) if return_eig_sample else psi


def state_prep_map_from_rho(rho_in: np.ndarray) -> Any:
    """Replacement preparation CP map: sigma -> Tr(sigma) * rho_in."""
    rho0 = 0.5 * (rho_in + rho_in.conj().T)
    tr = np.trace(rho0)
    if abs(tr) > 1e-12:
        rho0 = rho0 / tr
    return lambda rho: np.trace(rho) * rho0


def build_basis_for_fixed_alphabet(
    *,
    basis: str,
    basis_seed: int | None,
):
    """Return (basis_set, choi_matrices, choi_pm_pairs, choi_feat_table) for fixed basis."""
    basis_t = cast(TomographyBasis, basis)
    seed_for_basis = int(basis_seed) if basis_seed is not None else None
    # `random` basis needs a stable seed; the underlying helpers already support it.
    basis_set = get_basis_states(basis=basis_t, seed=seed_for_basis if basis == "random" else None)
    choi_matrices, choi_pm_pairs = get_choi_basis(basis=basis_t, seed=seed_for_basis if basis == "random" else None)
    choi_feat_table = build_choi_feature_table(choi_matrices)
    return basis_set, choi_matrices, choi_pm_pairs, choi_feat_table


def make_static_ctx(op: MPO, params: AnalogSimParams) -> Any:
    """Prepare the MCWF preprocessing context once."""
    dummy_mps = MPS(length=op.length, state="zeros")
    return preprocess_mcwf(dummy_mps, op, None, params)


def simulate_backend_labels(
    *,
    op: MPO,
    params: AnalogSimParams,
    timesteps: list[float],
    psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]],
    initial_psis: list[np.ndarray],
    parallel: bool,
    static_ctx: Any,
) -> np.ndarray:
    """Simulate backend labels for a dataset."""
    n = len(initial_psis)
    payload: dict[str, Any] = {
        "psi_pairs": psi_pairs_list,
        "initial_psi": initial_psis,
        "num_trajectories": 1,
        "operator": op,
        "sim_params": params,
        "timesteps": timesteps,
        "noise_model": None,
        "mcwf_static_ctx": static_ctx,
    }

    if parallel and n > 1:
        max_workers = max(1, available_cpus() - 1)
        it = run_backend_parallel(
            worker_fn=worker_initial_psi,
            payload=payload,
            n_jobs=n,
            max_workers=max_workers,
            show_progress=False,
            desc="Channel predictor backend sim",
        )
        tmp: list[np.ndarray | None] = [None] * n
        for _job, out in it:
            s_idx, _tr, rho_final, _w = out
            rho_norm = normalize_rho_from_backend_output(rho_final)
            tmp[s_idx] = pack_rho8(rho_norm)
        if any(t is None for t in tmp):
            raise RuntimeError("Parallel backend simulation incomplete.")
        rows = [cast(np.ndarray, t) for t in tmp]
        return np.stack(rows, axis=0).astype(np.float32)

    rows: list[np.ndarray] = []
    for j in range(n):
        _s, _tr, rho_final, _w = _call_backend_serial(worker_initial_psi, j, payload)
        rho_norm = normalize_rho_from_backend_output(rho_final)
        rows.append(pack_rho8(rho_norm))
    return np.stack(rows, axis=0).astype(np.float32)

