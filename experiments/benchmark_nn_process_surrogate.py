# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Channel predictor benchmark (default): ``(rho_in, E_1..E_k) -> rho_out``.

The **main** entry trains ``DenseMLP(8 + 32*k, hidden, 8)`` on backend-generated samples:
``x = [vec(rho_in), flat_Choi(E_1), ..., flat_Choi(E_k)]`` where each Choi is the fixed-basis
``4×4`` complex matrix flattened to **32** reals (Re/Im row-major). Targets are
``vec(rho_out)`` (8 reals); metrics use **Hermitianized** predictions and mean trace distance
on a held-out test split. Sweeps train size and seeds; saves **test TD vs train size** curves
(full model vs last-step and last-two-step baselines).

Optional **exhaustive grid** mode (``--exhaustive_grid``): sample-efficiency vs finite-memory
baselines on :class:`TomographyEstimate` labels (Bloch + weighted ``rho``).

Examples:
  python -m experiments.benchmark_nn_process_surrogate --ks 2 3 --n_samples 4096
  python -m experiments.benchmark_nn_process_surrogate --exhaustive_grid --ks 2 3 --parallel_tomo
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("This benchmark needs matplotlib. pip install matplotlib") from e

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise ImportError("This benchmark needs PyTorch. pip install torch") from e

from mqt.yaqs.analog.mcwf import preprocess_mcwf
from mqt.yaqs.characterization.tomography.basis import TomographyBasis, get_basis_states, get_choi_basis
from mqt.yaqs.characterization.tomography.formatters import _to_dense
from mqt.yaqs.characterization.tomography.combs import NNComb
from mqt.yaqs.characterization.tomography.ml_dataset import (
    TomographyMLDataset,
    bloch_vector_from_rho,
    clip_bloch_to_unit_ball,
    density_matrix_from_bloch,
    summarize_grid_coverage,
    tomography_estimate_to_ml_dataset,
    trace_distance,
)
from mqt.yaqs.characterization.tomography.process_tomography import (
    _build_tomography_data,
    _call_backend_serial,
    run as tomography_run,
)
from mqt.yaqs.characterization.tomography.tomography_utils import (
    _evolve_backend_state,
    _get_rho_site_zero,
    _initialize_backend_state,
    _reprepare_backend_state_forced,
    _tomography_sequence_worker,
)
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.simulator import available_cpus, run_backend_parallel


def _pack_rho8(rho: np.ndarray) -> np.ndarray:
    """Unweighted 2x2 density matrix as 8 floats (Re/Im row-major)."""
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


def _unpack_rho8(y: np.ndarray) -> np.ndarray:
    t = np.asarray(y, dtype=np.float64).reshape(8)
    rho = np.array(
        [
            [t[0] + 1j * t[1], t[2] + 1j * t[3]],
            [t[4] + 1j * t[5], t[6] + 1j * t[7]],
        ],
        dtype=np.complex128,
    )
    return 0.5 * (rho + rho.conj().T)


def _pack_wrho8(rho: np.ndarray, w: float) -> np.ndarray:
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2) * float(w)
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


def _max_within_suffix_msq(bloch: np.ndarray, seq: np.ndarray, n_suffix: int) -> float:
    """Max over keys of mean squared deviation of Bloch rows sharing the last ``n_suffix`` tokens."""
    k = seq.shape[1]
    if n_suffix <= 0 or n_suffix > k:
        return 0.0
    mx = 0.0
    # keys from columns k-n_suffix .. k-1
    keys = seq[:, k - n_suffix :].astype(np.int64, copy=False)
    # group by row key via dict
    buck: dict[tuple[int, ...], list[np.ndarray]] = defaultdict(list)
    for i in range(seq.shape[0]):
        key = tuple(int(x) for x in keys[i])
        buck[key].append(bloch[i])
    for vs in buck.values():
        if len(vs) < 2:
            continue
        g = np.stack(vs, axis=0)
        mu = g.mean(axis=0)
        se = float(np.mean(np.sum((g - mu) ** 2, axis=1)))
        mx = max(mx, se)
    return mx


class FlatMLP(nn.Module):
    """Embedding table + MLP on flattened token embeddings."""

    def __init__(self, k: int, d_model: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.emb = nn.Embedding(16, d_model)
        in_dim = k * d_model
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        e = self.emb(seq)
        flat = e.reshape(seq.shape[0], self.k * self.d_model)
        return self.mlp(flat)


class DenseMLP(nn.Module):
    """Plain MLP on a flat real feature vector (sampled rho + intervention encodings)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def weighted_mse(pred: torch.Tensor, tgt: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    err = ((pred - tgt) ** 2).sum(dim=-1)
    return (err * w).sum() / w.sum().clamp_min(1e-12)


def _train_mlp(
    model: nn.Module,
    seq: torch.Tensor,
    tgt: torch.Tensor,
    w: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> None:
    n = seq.shape[0]
    loader = DataLoader(
        TensorDataset(seq, tgt, w),
        batch_size=min(batch_size, max(1, n)),
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for sb, tb, wb in loader:
            sb, tb, wb = sb.to(device), tb.to(device), wb.to(device)
            opt.zero_grad()
            loss = weighted_mse(model(sb), tb, wb)
            loss.backward()
            opt.step()


def _train_mlp_dense(
    model: nn.Module,
    x: torch.Tensor,
    tgt: torch.Tensor,
    w: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> None:
    n = x.shape[0]
    loader = DataLoader(
        TensorDataset(x, tgt, w),
        batch_size=min(batch_size, max(1, n)),
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, tb, wb in loader:
            xb, tb, wb = xb.to(device), tb.to(device), wb.to(device)
            opt.zero_grad()
            loss = weighted_mse(model(xb), tb, wb)
            loss.backward()
            opt.step()


def _random_density_matrix(rng: np.Generator) -> np.ndarray:
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = a @ a.conj().T
    tr = float(np.trace(rho).real)
    rho = rho / max(tr, 1e-15)
    return 0.5 * (rho + rho.conj().T)


def _initial_mcwf_state_from_rho0(
    rho: np.ndarray,
    length: int,
    *,
    rng: np.random.Generator | None = None,
    init_mode: str = "eigenstate",
    return_eig_sample: bool = False,
) -> np.ndarray | tuple[np.ndarray, int, float]:
    """Initialize a *pure* MCWF state on ``length`` qubits matching the requested input.

    init_mode:
      - ``"eigenstate"``: sample one eigenvector |v_i> of rho_in with probability p_i and set
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
        # Build |v_idx> ⊗ |0...0> with site-1..(L-1) in the computational |0>.
        v_idx = v[:, idx].astype(np.complex128)
        if length <= 1:
            psi = v_idx
        else:
            env0 = np.array([1.0, 0.0], dtype=np.complex128)
            env_state = env0
            for _ in range(length - 2):
                env_state = np.kron(env_state, env0)
            # length-1 environment qubits: sites 1..L-1
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


def _worker_initial_psi(job_idx: int, payload: dict[str, Any] | None = None) -> tuple[int, int, np.ndarray, float]:
    """Like :func:`_tomography_sequence_worker` but starts from ``payload['initial_psi'][s_idx]``."""
    ctx = payload if payload is not None else {}
    num_trajectories: int = int(ctx["num_trajectories"])
    s_idx = job_idx // num_trajectories
    traj_idx = job_idx % num_trajectories
    psi_pairs = ctx["psi_pairs"][s_idx]
    operator = ctx["operator"]
    sim_params = ctx["sim_params"]
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


CHOI_FLAT_DIM = 32  # 4x4 complex -> 16 complex -> 32 real (Re, Im row-major)


def _flatten_choi4_to_real32(j: np.ndarray) -> np.ndarray:
    """Vectorize ``4×4`` complex Choi matrix to 32 floats (Re/Im, row-major)."""
    m = np.asarray(j, dtype=np.complex128).reshape(4, 4)
    flat = m.reshape(-1)
    interleaved = np.stack([flat.real, flat.imag], axis=-1).astype(np.float32).reshape(-1)
    return interleaved


def _build_choi_feature_table(choi_matrices: list[np.ndarray]) -> np.ndarray:
    """Shape ``(16, 32)``: one real feature row per fixed-basis CP map index."""
    rows = [_flatten_choi4_to_real32(c) for c in choi_matrices]
    return np.stack(rows, axis=0)


def _concat_choi_features(alphas: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Concatenate ``table[alpha_t]`` for each step; ``table`` is ``(16, 32)``."""
    a = np.asarray(alphas, dtype=np.int64).reshape(-1)
    parts = [table[int(ai)] for ai in a]
    return np.concatenate(parts, axis=0).astype(np.float32)


def _intervention_from_alpha(
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


def _mean_td_rho8(pred: np.ndarray, tgt: np.ndarray) -> float:
    """Mean trace distance; densities from 8-vectors are Hermitianized in ``_unpack_rho8``."""
    tds: list[float] = []
    for i in range(pred.shape[0]):
        rp = _unpack_rho8(pred[i])
        rt = _unpack_rho8(tgt[i])
        tds.append(trace_distance(rp, rt))
    return float(np.mean(tds))


def run_channel_predictor_benchmark(
    *,
    ks: list[int],
    n_pool: int,
    test_frac: float,
    train_fracs: list[float],
    train_sizes: list[int] | None,
    seeds: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
    hidden: int,
    L: int,
    J: float,
    g: float,
    timesteps_all: list[float],
    basis: str,
    basis_seed: int | None,
    max_bond_dim: int,
    parallel_sim: bool,
    backend_init_mode: str,
    out_dir: Path,
) -> None:
    """Main benchmark: learn ``(rho_in, E_1..E_k) -> rho_out`` with fixed basis and train-size sweeps."""
    valid_bases = ("standard", "tetrahedral", "random")
    if basis not in valid_bases:
        msg = f"basis must be one of {valid_bases}, got {basis!r}"
        raise ValueError(msg)
    basis_t = cast(TomographyBasis, basis)
    seed_fix = int(basis_seed) if basis_seed is not None else 12345
    # One fixed alphabet for the whole run; random basis requires a stable seed.
    seed_for_basis = seed_fix if basis == "random" else None
    basis_set = get_basis_states(basis=basis_t, seed=seed_for_basis)
    choi_matrices, choi_pm_pairs = get_choi_basis(basis=basis_t, seed=seed_for_basis)
    choi_feat_table = _build_choi_feature_table(choi_matrices)

    k_max = max(ks)
    if len(timesteps_all) < k_max:
        msg = f"Need len(timesteps) >= max(ks)={k_max}, got {len(timesteps_all)}"
        raise ValueError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Channel predictor: (rho_in, E_1..E_k) -> rho_out ===")
    print(f"  Fixed basis={basis!r}  basis_seed={seed_fix} (used for random basis; ignored for tetrahedral/standard)")
    print(f"  L={L} J={J} g={g}  pool={n_pool}  test_frac={test_frac}  seeds={seeds}")
    print(
        f"  x: [vec(rho_in) (8), flat Choi(E_t) ({CHOI_FLAT_DIM} reals) per step]  "
        f"=> in_dim = 8 + {CHOI_FLAT_DIM}*k"
    )
    print("  y: vec(rho_out) (8 reals); test metric: mean TD (Hermitianized preds)")

    op = MPO.ising(length=L, J=J, g=g)

    for k in sorted(set(ks)):
        timesteps = timesteps_all[:k]
        solver_dt = float(min(timesteps))
        params = AnalogSimParams(
            dt=solver_dt,
            solver="MCWF",
            show_progress=False,
            max_bond_dim=max_bond_dim,
        )
        dummy_mps = MPS(length=op.length, state="zeros")
        static_ctx = preprocess_mcwf(dummy_mps, op, None, params)

        n_test = max(1, int(round(test_frac * n_pool)))
        n_tr_pool = n_pool - n_test
        if n_tr_pool < 1:
            raise ValueError("test_frac too large: no training pool left.")

        if train_sizes is not None:
            sizes_sorted = sorted({int(s) for s in train_sizes if 1 <= int(s) <= n_tr_pool})
        else:
            sizes_sorted = sorted(
                {max(1, min(n_tr_pool, int(round(f * n_tr_pool)))) for f in train_fracs}
            )
        if not sizes_sorted:
            print(f"  k={k}: no valid train sizes; skip.")
            continue

        # Accumulate [seed][n_train] -> (td_full, td_b1, td_b2 or nan)
        by_size_full: dict[int, list[float]] = {s: [] for s in sizes_sorted}
        by_size_b1: dict[int, list[float]] = {s: [] for s in sizes_sorted}
        by_size_b2: dict[int, list[float]] = {s: [] for s in sizes_sorted}

        for seed in seeds:
            rng = np.random.default_rng(seed + k * 100_003)
            rho_ins: list[np.ndarray] = []
            alpha_rows: list[np.ndarray] = []
            psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]] = []
            initial_psis: list[np.ndarray] = []

            for _ in range(n_pool):
                rho_in = _random_density_matrix(rng)
                alphas = rng.integers(0, 16, size=k, dtype=np.int64)
                pairs = [
                    (basis_set[choi_pm_pairs[int(a)][1]][1], basis_set[choi_pm_pairs[int(a)][0]][1])
                    for a in alphas
                ]
                psi0 = _initial_mcwf_state_from_rho0(
                    rho_in, L, rng=rng, init_mode=backend_init_mode
                )
                rho_ins.append(rho_in)
                alpha_rows.append(alphas)
                psi_pairs_list.append(pairs)
                initial_psis.append(psi0)

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

            rho_out_rows: list[np.ndarray] = []
            if parallel_sim and n_pool > 1:
                max_workers = max(1, available_cpus() - 1)
                it = run_backend_parallel(
                    worker_fn=_worker_initial_psi,
                    payload=payload,
                    n_jobs=n_pool,
                    max_workers=max_workers,
                    show_progress=False,
                    desc=f"k={k} channel sim",
                )
                tmp: list[np.ndarray | None] = [None] * n_pool
                for _job, out in it:
                    s_idx, _tr, rho_final, _w = out
                    rho_h = np.asarray(rho_final, dtype=np.complex128).reshape(2, 2)
                    rho_h = 0.5 * (rho_h + rho_h.conj().T)
                    tmp[s_idx] = _pack_rho8(rho_h)
                if any(t is None for t in tmp):
                    raise RuntimeError("Parallel channel simulation incomplete.")
                rho_out_rows = [cast(np.ndarray, t) for t in tmp]
            else:
                for j in range(n_pool):
                    _s, _tr, rho_final, _w = _call_backend_serial(_worker_initial_psi, j, payload)
                    rho_h = np.asarray(rho_final, dtype=np.complex128).reshape(2, 2)
                    rho_h = 0.5 * (rho_h + rho_h.conj().T)
                    rho_out_rows.append(_pack_rho8(rho_h))

            y = np.stack(rho_out_rows, axis=0).astype(np.float32)
            rho8_in = np.stack([_pack_rho8(r) for r in rho_ins], axis=0).astype(np.float32)
            alphas_arr = np.stack(alpha_rows, axis=0)

            x_full = np.stack(
                [
                    np.concatenate(
                        [rho8_in[i], _concat_choi_features(alphas_arr[i], choi_feat_table)],
                        axis=0,
                    )
                    for i in range(n_pool)
                ],
                axis=0,
            ).astype(np.float32)
            x_last = np.stack(
                [
                    np.concatenate(
                        [
                            rho8_in[i],
                            _concat_choi_features(alphas_arr[i, k - 1 : k], choi_feat_table),
                        ],
                        axis=0,
                    )
                    for i in range(n_pool)
                ],
                axis=0,
            ).astype(np.float32)
            x_penult_last: np.ndarray | None = None
            if k >= 2:
                x_penult_last = np.stack(
                    [
                        np.concatenate(
                            [
                                rho8_in[i],
                                _concat_choi_features(alphas_arr[i, k - 2 : k], choi_feat_table),
                            ],
                            axis=0,
                        )
                        for i in range(n_pool)
                    ],
                    axis=0,
                ).astype(np.float32)

            perm = rng.permutation(n_pool)
            te_idx = perm[:n_test]
            tr_perm = perm[n_test:]
            y_te = y[te_idx]
            device = torch.device("cpu")

            def fit_eval(x_all: np.ndarray, tr_idx: np.ndarray) -> float:
                nn = NNComb(in_dim=int(x_all.shape[1]), hidden=hidden, out_dim=8)
                nn.fit_features(
                    x_all[tr_idx],
                    y[tr_idx],
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    device=device,
                    w_train=np.ones(tr_idx.shape[0], dtype=np.float32),
                )
                pred = nn.predict_features(x_all[te_idx], device=device)
                return _mean_td_rho8(pred, y_te)

            print(f"\n--- k={k}  seed={seed}  pool={n_pool}  test={n_test}  train_pool={n_tr_pool} ---")
            for n_tr in sizes_sorted:
                tr_idx = tr_perm[:n_tr]
                td_f = fit_eval(x_full, tr_idx)
                td_1 = fit_eval(x_last, tr_idx)
                td_2 = float("nan")
                if x_penult_last is not None:
                    td_2 = fit_eval(x_penult_last, tr_idx)
                by_size_full[n_tr].append(td_f)
                by_size_b1[n_tr].append(td_1)
                by_size_b2[n_tr].append(td_2)
                b2s = f"{td_2:.6f}" if k >= 2 and np.isfinite(td_2) else "n/a"
                print(f"  n_train={n_tr:5d}  full TD={td_f:.6f}  [rho,E_k]={td_1:.6f}  [rho,E_k-1,E_k]={b2s}")

        print(f"\n>>> k={k}  aggregate over {len(seeds)} seeds (mean +- std, test TD)")
        xs_plot: list[int] = []
        mf, sf = [], []
        m1, s1 = [], []
        m2, s2 = [], []
        for n_tr in sizes_sorted:
            xs_plot.append(n_tr)
            af, bf = _mean_std(by_size_full[n_tr])
            a1, b1 = _mean_std(by_size_b1[n_tr])
            a2, b2 = _mean_std([v for v in by_size_b2[n_tr] if np.isfinite(v)])
            print(
                f"  n_train={n_tr:5d}  full {af:.6f}+-{bf:.6f}  "
                f"b1 {a1:.6f}+-{b1:.6f}  "
                f"b2 {a2:.6f}+-{b2:.6f}"
            )
            mf.append(af)
            sf.append(bf)
            m1.append(a1)
            s1.append(b1)
            m2.append(a2)
            s2.append(b2)

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.errorbar(xs_plot, mf, yerr=sf, fmt="o-", capsize=4, label=f"full (8+{CHOI_FLAT_DIM}*k)", lw=2)
        ax.errorbar(xs_plot, m1, yerr=s1, fmt="s--", capsize=4, label="baseline [rho_in, E_k]", lw=1.5)
        if k >= 2 and any(np.isfinite(v) for v in m2):
            ax.errorbar(xs_plot, m2, yerr=s2, fmt="^:", capsize=4, label="baseline [rho_in, E_k-1, E_k]", lw=1.5)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Mean trace distance (test)")
        ax.set_title(f"Channel predictor k={k}  basis={basis}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig_path = out_dir / f"channel_predictor_k{k}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved figure: {fig_path}")


def run_channel_predictor_vs_exhaustive_comb(
    *,
    ks: list[int],
    n_samples: int,
    seeds: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
    hidden: int,
    L: int,
    J: float,
    g: float,
    timesteps_all: list[float],
    basis: str,
    basis_seed: int | None,
    max_bond_dim: int,
    parallel_sim: bool,
    backend_init_mode: str,
    out_dir: Path,
    debug_compare: int,
) -> None:
    """Minimal benchmark: train on backend labels, evaluate against exhaustive comb labels."""
    valid_bases = ("standard", "tetrahedral", "random")
    if basis not in valid_bases:
        raise ValueError(f"basis must be one of {valid_bases}, got {basis!r}")
    basis_t = cast(TomographyBasis, basis)
    seed_fix = int(basis_seed) if basis_seed is not None else 12345
    seed_for_basis = seed_fix if basis == "random" else None

    out_dir.mkdir(parents=True, exist_ok=True)
    op = MPO.ising(length=L, J=J, g=g)
    basis_set = get_basis_states(basis=basis_t, seed=seed_for_basis)
    choi_matrices, choi_pm_pairs = get_choi_basis(basis=basis_t, seed=seed_for_basis)
    choi_feat_table = _build_choi_feature_table(choi_matrices)

    def _normalize_rho_like_densecomb(rho: np.ndarray) -> np.ndarray:
        """Match DenseComb.predict convention: Hermitize, normalize trace, PSD-project, renormalize trace."""
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

    def _cptp_to_choi_local(emap: Any) -> np.ndarray:
        """Local copy of mqt.yaqs.characterization.tomography.combs._cptp_to_choi."""
        j_choi = np.zeros((4, 4), dtype=complex)
        for i in range(2):
            for j in range(2):
                e_in = np.zeros((2, 2), dtype=complex)
                e_in[i, j] = 1.0
                j_choi += np.kron(emap(e_in), e_in)
        return j_choi

    def _state_prep_map_from_rho(rho_in: np.ndarray) -> Any:
        """Linear preparation-only CP map: rho -> Tr(rho) * rho_in (replacement channel)."""
        rho0 = 0.5 * (rho_in + rho_in.conj().T)
        tr = np.trace(rho0)
        if abs(tr) > 1e-12:
            rho0 = rho0 / tr
        # Ensure Hermiticity + trace-1 numerically; DenseComb.predict already projects PSD.
        return lambda rho: np.trace(rho) * rho0

    def _debug_floor_compare_for_one_k(k: int, seed_for_k: int) -> None:
        """Strict debug compare: backend vs exhaustive for tiny set, abort if inconsistent."""
        if debug_compare <= 0:
            return

        timesteps = timesteps_all[:k]
        solver_dt = float(min(timesteps))
        params = AnalogSimParams(
            dt=solver_dt,
            solver="MCWF",
            show_progress=False,
            max_bond_dim=max_bond_dim,
        )
        print(f"\n[debug_compare={debug_compare}] k={k}: build backend+exhaustive reference consistency check")

        dummy_mps = MPS(length=op.length, state="zeros")
        static_ctx = preprocess_mcwf(dummy_mps, op, None, params)

        comb_ex_no_prep = tomography_run(
            operator=op,
            sim_params=params,
            timesteps=timesteps,
            method="exhaustive",
            output="dense",
            noise_model=None,
            parallel=False,
            num_samples=16**k,
            num_trajectories=1,
            seed=0,
            basis=basis_t,
            basis_seed=seed_for_basis,
        )
        if not hasattr(comb_ex_no_prep, "predict"):
            raise RuntimeError("Expected DenseComb output from exhaustive run.")

        tds: list[float] = []
        tds_old: list[float] = []
        tds_new: list[float] = []
        for j in range(debug_compare):
            rng = np.random.default_rng(seed_for_k + j * 10_001)
            rho_in = _random_density_matrix(rng)
            alphas = rng.integers(0, 16, size=k, dtype=np.int64)
            intervention_fns = [
                _intervention_from_alpha(int(a), basis_set, choi_pm_pairs) for a in alphas
            ]

            # Verify intervention Choi objects match fixed get_choi_basis elements.
            for t, a in enumerate(alphas):
                a_i = int(a)
                expected = choi_matrices[a_i]
                got = _cptp_to_choi_local(intervention_fns[t])
                diff = float(np.max(np.abs(expected - got)))
                if diff > 1e-8:
                    print(f"  [debug] alpha[{t}]={a_i} choi mismatch max|diff|={diff:.3e}")

            pairs = [
                (basis_set[choi_pm_pairs[int(a)][1]][1], basis_set[choi_pm_pairs[int(a)][0]][1])
                for a in alphas
            ]
            psi0, eig_idx, eig_p = _initial_mcwf_state_from_rho0(
                rho_in,
                L,
                rng=rng,
                init_mode=backend_init_mode,
                return_eig_sample=True,
            )
            payload = {
                "psi_pairs": [pairs],
                "initial_psi": [psi0],
                "num_trajectories": 1,
                "operator": op,
                "sim_params": params,
                "timesteps": timesteps,
                "noise_model": None,
                "mcwf_static_ctx": static_ctx,
            }
            s_idx, _tr, rho_final, _w = _call_backend_serial(_worker_initial_psi, 0, payload)
            assert s_idx == 0
            rho_backend = _normalize_rho_like_densecomb(
                np.asarray(rho_final, dtype=np.complex128).reshape(2, 2)
            )

            rho_ex_old = comb_ex_no_prep.predict(intervention_fns)
            rho_ex_old = _normalize_rho_like_densecomb(np.asarray(rho_ex_old, dtype=np.complex128).reshape(2, 2))

            # Minimal convention: inject rho_in into the *first* intervention slot by composing
            # the alpha_0 map with a replacement/preparation CP map.
            prep_map = _state_prep_map_from_rho(rho_in)
            alpha0 = intervention_fns[0]
            def first_map(sigma: np.ndarray, alpha0: Any = alpha0, prep_map: Any = prep_map) -> np.ndarray:
                return alpha0(prep_map(sigma))

            rho_ex_new = comb_ex_no_prep.predict([first_map] + intervention_fns[1:])
            rho_ex_new = _normalize_rho_like_densecomb(np.asarray(rho_ex_new, dtype=np.complex128).reshape(2, 2))

            td_old = trace_distance(rho_backend, rho_ex_old)
            td_new = trace_distance(rho_backend, rho_ex_new)
            tds_old.append(float(td_old))
            tds_new.append(float(td_new))

            print(f"\n  [debug] sample {j}")
            print(f"    sequence(alphas) = {alphas.tolist()}")
            print(f"    rho_in =\n{rho_in}")
            print(
                f"    init_mode={backend_init_mode}: sampled eig_idx={eig_idx} with eigenvalue p={eig_p:.6f}"
            )
            print(f"    backend rho_out =\n{rho_backend}")
            print(f"    exhaustive (no rho_in prep) rho_out =\n{rho_ex_old}")
            print(f"    exhaustive (with rho_in prep) rho_out =\n{rho_ex_new}")
            print(f"    TD old (backend vs no-prep) = {td_old:.6e}")
            print(f"    TD new (backend vs with-prep) = {td_new:.6e}")

        mean_old = float(np.mean(tds_old)) if tds_old else float("nan")
        mean_new = float(np.mean(tds_new)) if tds_new else float("nan")
        print(
            f"\n[debug_compare] k={k}: mean TD backend vs exhaustive\n"
            f"  old (no rho_in prep):  {mean_old:.6e}\n"
            f"  new (with rho_in prep): {mean_new:.6e}"
        )
        if not (mean_new < 1e-2):
            print("benchmark inconsistent")
            raise RuntimeError("benchmark inconsistent: exhaustive reference still inconsistent after rho_in prep injection.")

    default_sizes = {
        2: [8, 16, 32, 64, 128],
        3: [16, 32, 64, 128, 256, 512],
        4: [32, 64, 128, 256, 512],
    }

    print("\n=== Minimal Channel Predictor vs Exhaustive Comb ===")
    print("  Goal: learn (vec(rho_in), Choi(E_1..E_k)) -> vec(rho_out)")
    print(f"  basis={basis!r} basis_seed={seed_fix}  N={n_samples}  seeds={seeds}")

    for k in sorted(set(ks)):
        if len(timesteps_all) < k:
            raise ValueError(f"Need len(timesteps) >= {k}")
        timesteps = timesteps_all[:k]
        solver_dt = float(min(timesteps))
        params = AnalogSimParams(
            dt=solver_dt,
            solver="MCWF",
            show_progress=False,
            max_bond_dim=max_bond_dim,
        )
        dummy_mps = MPS(length=op.length, state="zeros")
        static_ctx = preprocess_mcwf(dummy_mps, op, None, params)

        print(f"\n--- k={k}: building exhaustive DenseComb reference once ---")
        seed_dbg = int(seeds[0]) if seeds else 0
        _debug_floor_compare_for_one_k(k, seed_dbg + k * 12345)
        comb_ex_no_prep = tomography_run(
            operator=op,
            sim_params=params,
            timesteps=timesteps,
            method="exhaustive",
            output="dense",
            noise_model=None,
            parallel=False,
            num_samples=16**k,
            num_trajectories=1,
            seed=0,
            basis=basis_t,
            basis_seed=seed_for_basis,
        )
        if not hasattr(comb_ex_no_prep, "predict"):
            raise RuntimeError("Expected DenseComb output from exhaustive run.")

        td_nn_by_size: dict[int, list[float]] = defaultdict(list)
        td_floor_by_size: dict[int, list[float]] = defaultdict(list)
        td_old_floor_seed: list[float] = []
        td_new_floor_seed: list[float] = []

        for seed in seeds:
            rng = np.random.default_rng(seed + 97_531 * k)
            rho_ins: list[np.ndarray] = []
            alpha_rows: list[np.ndarray] = []
            psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]] = []
            initial_psis: list[np.ndarray] = []
            intervention_fns: list[list[Any]] = []

            for _ in range(n_samples):
                rho_in = _random_density_matrix(rng)
                alphas = rng.integers(0, 16, size=k, dtype=np.int64)
                pairs = [
                    (basis_set[choi_pm_pairs[int(a)][1]][1], basis_set[choi_pm_pairs[int(a)][0]][1])
                    for a in alphas
                ]
                fns = [_intervention_from_alpha(int(a), basis_set, choi_pm_pairs) for a in alphas]
                psi0 = _initial_mcwf_state_from_rho0(
                    rho_in, L, rng=rng, init_mode=backend_init_mode
                )
                rho_ins.append(rho_in)
                alpha_rows.append(alphas)
                psi_pairs_list.append(pairs)
                initial_psis.append(psi0)
                intervention_fns.append(fns)

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

            y_backend_rows: list[np.ndarray] = []
            if parallel_sim and n_samples > 1:
                max_workers = max(1, available_cpus() - 1)
                it = run_backend_parallel(
                    worker_fn=_worker_initial_psi,
                    payload=payload,
                    n_jobs=n_samples,
                    max_workers=max_workers,
                    show_progress=False,
                    desc=f"k={k} backend sampled",
                )
                tmp: list[np.ndarray | None] = [None] * n_samples
                for _job, out in it:
                    s_idx, _tr, rho_final, _w = out
                    rho_h = _normalize_rho_like_densecomb(
                        np.asarray(rho_final, dtype=np.complex128).reshape(2, 2)
                    )
                    tmp[s_idx] = _pack_rho8(rho_h)
                if any(t is None for t in tmp):
                    raise RuntimeError("Parallel sampled simulation incomplete.")
                y_backend_rows = [cast(np.ndarray, t) for t in tmp]
            else:
                for j in range(n_samples):
                    _s, _tr, rho_final, _w = _call_backend_serial(_worker_initial_psi, j, payload)
                    rho_h = _normalize_rho_like_densecomb(
                        np.asarray(rho_final, dtype=np.complex128).reshape(2, 2)
                    )
                    y_backend_rows.append(_pack_rho8(rho_h))

            y_ex_old_rows: list[np.ndarray] = []
            y_ex_new_rows: list[np.ndarray] = []
            for i in range(n_samples):
                rho_ex_old = comb_ex_no_prep.predict(intervention_fns[i])
                prep_map = _state_prep_map_from_rho(rho_ins[i])
                alpha0 = intervention_fns[i][0]
                def first_map(sigma: np.ndarray, alpha0: Any = alpha0, prep_map: Any = prep_map) -> np.ndarray:
                    return alpha0(prep_map(sigma))

                rho_ex_new = comb_ex_no_prep.predict([first_map] + intervention_fns[i][1:])
                y_ex_old_rows.append(_pack_rho8(rho_ex_old))
                y_ex_new_rows.append(_pack_rho8(rho_ex_new))

            y_backend = np.stack(y_backend_rows, axis=0).astype(np.float32)
            y_ex_old = np.stack(y_ex_old_rows, axis=0).astype(np.float32)
            y_ex = np.stack(y_ex_new_rows, axis=0).astype(np.float32)
            rho8_in = np.stack([_pack_rho8(r) for r in rho_ins], axis=0).astype(np.float32)
            alphas_arr = np.stack(alpha_rows, axis=0)
            x_full = np.stack(
                [
                    np.concatenate([rho8_in[i], _concat_choi_features(alphas_arr[i], choi_feat_table)], axis=0)
                    for i in range(n_samples)
                ],
                axis=0,
            ).astype(np.float32)

            # Sanity: backend vs exhaustive reference should be close.
            sanity_n = min(16, n_samples)
            sanity_idx = rng.choice(n_samples, size=sanity_n, replace=False)
            sanity_td_old = _mean_td_rho8(y_backend[sanity_idx], y_ex_old[sanity_idx])
            sanity_td_new = _mean_td_rho8(y_backend[sanity_idx], y_ex[sanity_idx])
            print(
                f"  k={k} seed={seed} sanity TD backend vs exhaustive: "
                f"old(no rho_in prep)={sanity_td_old:.3e}  new(with rho_in prep)={sanity_td_new:.3e}"
            )

            n_test = max(1, int(round(0.2 * n_samples)))
            perm = rng.permutation(n_samples)
            te_idx = perm[:n_test]
            tr_idx_all = perm[n_test:]
            n_train_pool = tr_idx_all.shape[0]
            sweep = default_sizes.get(k, default_sizes[4])
            sweep = sorted([s for s in sweep if 1 <= s <= n_train_pool])
            if not sweep:
                sweep = [max(1, min(n_train_pool, n_train_pool // 2))]

            floor_td = _mean_td_rho8(y_backend[te_idx], y_ex[te_idx])
            floor_td_old = _mean_td_rho8(y_backend[te_idx], y_ex_old[te_idx])
            td_old_floor_seed.append(float(floor_td_old))
            td_new_floor_seed.append(float(floor_td))
            device = torch.device("cpu")

            for n_train in sweep:
                tr_idx = tr_idx_all[:n_train]
                nn = NNComb(in_dim=int(x_full.shape[1]), hidden=hidden, out_dim=8)
                nn.fit_features(
                    x_full[tr_idx],
                    y_backend[tr_idx],
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    device=device,
                    w_train=np.ones(n_train, dtype=np.float32),
                )
                pred = nn.predict_features(x_full[te_idx], device=device)
                td_nn = _mean_td_rho8(pred, y_ex[te_idx])
                td_nn_by_size[n_train].append(td_nn)
                td_floor_by_size[n_train].append(floor_td)

        # Print oracle mismatch floors (backend vs exhaustive) with/without rho_in injection.
        if td_old_floor_seed:
            m_old, s_old = _mean_std(td_old_floor_seed)
            m_new, s_new = _mean_std(td_new_floor_seed)
            print(
                f"\n>>> k={k} oracle floor (mean over seeds, test split)\n"
                f"    old floor: mean TD={m_old:.6f} +- {s_old:.6f}  (no rho_in prep injection)\n"
                f"    new floor: mean TD={m_new:.6f} +- {s_new:.6f}  (with rho_in prep injection)"
            )

        xs = sorted(td_nn_by_size.keys())
        print(f"\n>>> k={k} summary (vs exhaustive reference)")
        for n_train in xs:
            m_nn, s_nn = _mean_std(td_nn_by_size[n_train])
            m_floor, s_floor = _mean_std(td_floor_by_size[n_train])
            print(
                f"  full_grid=16^{k}={16**k:6d}  train={n_train:4d}  "
                f"NN TD={m_nn:.6f}+-{s_nn:.6f}  floor TD={m_floor:.6f}+-{s_floor:.6f}"
            )

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        mean_nn = [_mean_std(td_nn_by_size[n])[0] for n in xs]
        std_nn = [_mean_std(td_nn_by_size[n])[1] for n in xs]
        mean_floor = [_mean_std(td_floor_by_size[n])[0] for n in xs]
        std_floor = [_mean_std(td_floor_by_size[n])[1] for n in xs]
        ax.errorbar(xs, mean_nn, yerr=std_nn, fmt="o-", capsize=4, lw=2, label="NN vs exhaustive")
        ax.errorbar(xs, mean_floor, yerr=std_floor, fmt="s--", capsize=4, lw=1.5, label="backend vs exhaustive floor")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Mean trace distance on test")
        ax.set_title(f"Minimal channel predictor vs exhaustive comb (k={k})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig_path = out_dir / f"channel_predictor_vs_exhaustive_k{k}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved figure: {fig_path}")


def _bloch_td_stats(pred: np.ndarray, tgt: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    td_uw_list: list[float] = []
    td_w_num = 0.0
    w_sum = 0.0
    for i in range(pred.shape[0]):
        pb = clip_bloch_to_unit_ball(pred[i])
        rho_p = density_matrix_from_bloch(pb)
        rho_t = density_matrix_from_bloch(tgt[i])
        rho_t = np.asarray(rho_t, dtype=np.complex128)
        rho_t = 0.5 * (rho_t + rho_t.conj().T)
        td = trace_distance(rho_p, rho_t)
        td_uw_list.append(td)
        td_w_num += float(w[i]) * td
        w_sum += float(w[i])
    return float(np.mean(td_uw_list)), td_w_num / max(w_sum, 1e-12)


def _wrho_rmse_stats(pred: np.ndarray, tgt: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    diff = pred - tgt
    se = (diff**2).sum(axis=-1)
    rmse_uw = float(np.sqrt(np.mean(se)))
    rmse_w = float(np.sqrt(np.sum(w * se) / max(float(np.sum(w)), 1e-12)))
    return rmse_uw, rmse_w


def _mem_table(
    train_seq: np.ndarray,
    train_y: np.ndarray,
    k: int,
    m: int,
) -> dict[tuple[int, ...], np.ndarray]:
    buck: dict[tuple[int, ...], list[np.ndarray]] = defaultdict(list)
    for i in range(train_seq.shape[0]):
        key = tuple(int(train_seq[i, j]) for j in range(k - m, k))
        buck[key].append(train_y[i])
    return {key: np.mean(np.stack(vs, axis=0), axis=0) for key, vs in buck.items()}


def _predict_mem(
    test_seq: np.ndarray,
    table: dict[tuple[int, ...], np.ndarray],
    gmean: np.ndarray,
    k: int,
    m: int,
) -> np.ndarray:
    out = np.zeros((test_seq.shape[0], gmean.shape[0]), dtype=np.float32)
    gm = gmean.astype(np.float32)
    for i in range(test_seq.shape[0]):
        key = tuple(int(test_seq[i, j]) for j in range(k - m, k))
        out[i] = table.get(key, gm)
    return out


def _mem_metrics_full_and_cov(
    pred: np.ndarray,
    tgt: np.ndarray,
    w: np.ndarray,
    test_seq: np.ndarray,
    table: dict[tuple[int, ...], np.ndarray],
    k: int,
    m: int,
    *,
    y_dim: int,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Returns (td_uw, td_w) or (rmse_uw, rmse_w) for full and covered; covered fraction."""
    n = test_seq.shape[0]
    mask = np.array(
        [tuple(int(test_seq[i, j]) for j in range(k - m, k)) in table for i in range(n)],
        dtype=bool,
    )
    frac = float(mask.sum()) / float(max(n, 1))
    if y_dim == 3:
        f_td_uw, f_td_w = _bloch_td_stats(pred, tgt, w)
        if not mask.any():
            return (f_td_uw, f_td_w), (float("nan"), float("nan")), frac
        c_td_uw, c_td_w = _bloch_td_stats(pred[mask], tgt[mask], w[mask])
        return (f_td_uw, f_td_w), (c_td_uw, c_td_w), frac
    f_r_uw, f_r_w = _wrho_rmse_stats(pred, tgt, w)
    if not mask.any():
        return (f_r_uw, f_r_w), (float("nan"), float("nan")), frac
    c_r_uw, c_r_w = _wrho_rmse_stats(pred[mask], tgt[mask], w[mask])
    return (f_r_uw, f_r_w), (c_r_uw, c_r_w), frac


def _simulate_intervention_set(
    seqs: list[tuple[int, ...]],
    *,
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    basis: TomographyBasis,
    basis_seed: int | None,
    parallel: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backend gold labels for arbitrary basis sequences (same pipeline as exhaustive)."""
    k = len(timesteps)
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True
    basis_set = get_basis_states(basis=basis, seed=basis_seed)
    _choi_basis, choi_indices = get_choi_basis(basis=basis, seed=basis_seed)
    psi_pairs = [
        [(basis_set[choi_indices[a][1]][1], basis_set[choi_indices[a][0]][1]) for a in seq]
        for seq in seqs
    ]
    noise_model = None
    static_ctx = None
    if local_params.solver == "MCWF":
        dummy_mps = MPS(length=operator.length, state="zeros")
        static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, local_params)
    payload: dict[str, Any] = {
        "psi_pairs": psi_pairs,
        "num_trajectories": 1,
        "operator": operator,
        "sim_params": local_params,
        "timesteps": timesteps,
        "noise_model": noise_model,
        "mcwf_static_ctx": static_ctx,
    }
    n_seq = len(seqs)
    bloch_rows: list[np.ndarray] = []
    wr8_rows: list[np.ndarray] = []
    w_rows: list[float] = []
    seq_arr = np.array(seqs, dtype=np.int64)

    if parallel and n_seq > 1:
        max_workers = max(1, available_cpus() - 1)
        it = run_backend_parallel(
            worker_fn=_tomography_sequence_worker,
            payload=payload,
            n_jobs=n_seq,
            max_workers=max_workers,
            show_progress=False,
            desc="Intervention sim",
        )
        results: list[tuple[Any, float] | None] = [None] * n_seq
        for _job, out in it:
            s_idx, _t, rho_final, weight = out
            results[s_idx] = (rho_final, weight)
        ordered = [t for t in results if t is not None]
        if len(ordered) != n_seq:
            msg = "Parallel intervention simulation dropped jobs."
            raise RuntimeError(msg)
    else:
        ordered = []
        for j in range(n_seq):
            _s, _t, rho_final, weight = _call_backend_serial(_tomography_sequence_worker, j, payload)
            ordered.append((rho_final, weight))

    for rho_final, weight in ordered:
        rho_h = np.asarray(rho_final, dtype=np.complex128).reshape(2, 2)
        rho_h = 0.5 * (rho_h + rho_h.conj().T)
        w = float(weight)
        bloch_rows.append(bloch_vector_from_rho(rho_h))
        wr8_rows.append(_pack_wrho8(rho_h, w))
        w_rows.append(w)

    return (
        seq_arr,
        np.stack(bloch_rows, axis=0).astype(np.float32),
        np.stack(wr8_rows, axis=0).astype(np.float32),
        np.array(w_rows, dtype=np.float32),
    )


def _sample_intervention_sequences(
    rng: np.Generator,
    k: int,
    n: int,
    forbidden: set[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        seq = tuple(int(x) for x in rng.integers(0, 16, size=k))
        if seq in forbidden:
            continue
        out.append(seq)
    if len(out) < n:
        # relax: allow overlap if space exhausted (tiny k and huge forbidden)
        while len(out) < n:
            out.append(tuple(int(x) for x in rng.integers(0, 16, size=k)))
    return out


def run_one_split(
    *,
    ds: TomographyMLDataset,
    train_idx: np.ndarray,
    grid_test_idx: np.ndarray,
    int_seq: np.ndarray,
    int_bloch: np.ndarray,
    int_wrho8: np.ndarray,
    int_w: np.ndarray,
    k: int,
    epochs: int,
    lr: float,
    batch_size: int,
    d_model: int,
    hidden: int,
    device: torch.device,
) -> dict[str, float]:
    tr_seq = ds.sequence_indices[train_idx]
    tr_b = ds.bloch_target[train_idx]
    tr_w = ds.estimator_weight[train_idx]
    tr_wr = ds.weighted_rho8
    assert tr_wr is not None
    tr_wr = tr_wr[train_idx]

    g_seq = ds.sequence_indices[grid_test_idx]
    g_b = ds.bloch_target[grid_test_idx]
    g_w = ds.estimator_weight[grid_test_idx]
    g_wr = cast(np.ndarray, ds.weighted_rho8)[grid_test_idx]

    gmean_b = np.mean(tr_b, axis=0)
    gmean_wr = np.mean(tr_wr, axis=0)

    device_t = device
    # --- Train NNs (Bloch: unweighted MSE; w·ρ: weighted MSE) ---
    m_b = FlatMLP(k, d_model, hidden, 3).to(device_t)
    _train_mlp(
        m_b,
        torch.as_tensor(tr_seq, dtype=torch.long),
        torch.as_tensor(tr_b, dtype=torch.float32),
        torch.ones(tr_seq.shape[0], dtype=torch.float32),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device_t,
    )
    m_w = FlatMLP(k, d_model, hidden, 8).to(device_t)
    _train_mlp(
        m_w,
        torch.as_tensor(tr_seq, dtype=torch.long),
        torch.as_tensor(tr_wr, dtype=torch.float32),
        torch.as_tensor(tr_w, dtype=torch.float32),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device_t,
    )
    m_b.eval()
    m_w.eval()
    with torch.no_grad():
        nn_g_b = m_b(torch.as_tensor(g_seq, dtype=torch.long, device=device_t)).cpu().numpy()
        nn_g_wr = m_w(torch.as_tensor(g_seq, dtype=torch.long, device=device_t)).cpu().numpy()
        nn_i_b = m_b(torch.as_tensor(int_seq, dtype=torch.long, device=device_t)).cpu().numpy()
        nn_i_wr = m_w(torch.as_tensor(int_seq, dtype=torch.long, device=device_t)).cpu().numpy()

    row: dict[str, float] = {}

    def add_mem_sweep(prefix: str, te_seq: np.ndarray, te_b: np.ndarray, te_wr: np.ndarray, te_w: np.ndarray) -> None:
        # m=0 global
        pred_b = np.broadcast_to(gmean_b.astype(np.float32), (te_seq.shape[0], 3)).copy()
        pred_wr = np.broadcast_to(gmean_wr.astype(np.float32), (te_seq.shape[0], 8)).copy()
        tb_uw, tb_w = _bloch_td_stats(pred_b, te_b, te_w)
        tr_uw, tr_w = _wrho_rmse_stats(pred_wr, te_wr, te_w)
        row[f"{prefix}_mem0_bloch_td_uw"] = tb_uw
        row[f"{prefix}_mem0_bloch_td_w"] = tb_w
        row[f"{prefix}_mem0_wrho_rmse_uw"] = tr_uw
        row[f"{prefix}_mem0_wrho_rmse_w"] = tr_w
        best_b_td = tb_uw
        best_wr_w = tr_w
        best_m_b = 0
        best_m_wr = 0
        for m in range(1, k + 1):
            tab_b = _mem_table(tr_seq, tr_b, k, m)
            tab_wr = _mem_table(tr_seq, tr_wr, k, m)
            pb = _predict_mem(te_seq, tab_b, gmean_b, k, m)
            pw = _predict_mem(te_seq, tab_wr, gmean_wr, k, m)
            (fb_uw, fb_w), (cb_uw, cb_w), _frac = _mem_metrics_full_and_cov(
                pb, te_b, te_w, te_seq, tab_b, k, m, y_dim=3
            )
            (fw_uw, fw_w), (cw_uw, cw_w), frac = _mem_metrics_full_and_cov(
                pw, te_wr, te_w, te_seq, tab_wr, k, m, y_dim=8
            )
            row[f"{prefix}_mem{m}_bloch_td_uw_full"] = fb_uw
            row[f"{prefix}_mem{m}_bloch_td_uw_cov"] = cb_uw
            row[f"{prefix}_mem{m}_wrho_rmse_w_full"] = fw_w
            row[f"{prefix}_mem{m}_wrho_rmse_w_cov"] = cw_w
            row[f"{prefix}_mem{m}_cov_frac"] = frac
            if fb_uw < best_b_td:
                best_b_td = fb_uw
                best_m_b = m
            if fw_w < best_wr_w:
                best_wr_w = fw_w
                best_m_wr = m
        row[f"{prefix}_best_mem_bloch_m"] = float(best_m_b)
        row[f"{prefix}_best_mem_wrho_m"] = float(best_m_wr)

    # Grid test baselines + NN metrics
    gb_uw, gb_w = _bloch_td_stats(nn_g_b, g_b, g_w)
    gr_uw, gr_w = _wrho_rmse_stats(nn_g_wr, g_wr, g_w)
    row["grid_nn_bloch_td_uw"] = gb_uw
    row["grid_nn_bloch_td_w"] = gb_w
    row["grid_nn_wrho_rmse_uw"] = gr_uw
    row["grid_nn_wrho_rmse_w"] = gr_w
    add_mem_sweep("grid", g_seq, g_b, g_wr, g_w)

    # Intervention
    ib_uw, ib_w = _bloch_td_stats(nn_i_b, int_bloch, int_w)
    ir_uw, ir_w = _wrho_rmse_stats(nn_i_wr, int_wrho8, int_w)
    row["int_nn_bloch_td_uw"] = ib_uw
    row["int_nn_bloch_td_w"] = ib_w
    row["int_nn_wrho_rmse_uw"] = ir_uw
    row["int_nn_wrho_rmse_w"] = ir_w
    add_mem_sweep("int", int_seq, int_bloch, int_wrho8, int_w)

    def _min_mem_bloch(prefix: str) -> float:
        vals = [row[f"{prefix}_mem0_bloch_td_uw"]]
        vals += [row[f"{prefix}_mem{m}_bloch_td_uw_full"] for m in range(1, k + 1)]
        return float(min(vals))

    def _min_mem_wrho(prefix: str) -> float:
        vals = [row[f"{prefix}_mem0_wrho_rmse_w"]]
        vals += [row[f"{prefix}_mem{m}_wrho_rmse_w_full"] for m in range(1, k + 1)]
        return float(min(vals))

    # Beat best full-memory baseline (same train budget)?
    row["grid_nn_beats_best_mem_bloch"] = float(gb_uw < _min_mem_bloch("grid"))
    row["grid_nn_beats_best_mem_wrho"] = float(gr_w < _min_mem_wrho("grid"))
    row["int_nn_beats_best_mem_bloch"] = float(ib_uw < _min_mem_bloch("int"))
    row["int_nn_beats_best_mem_wrho"] = float(ir_w < _min_mem_wrho("int"))
    return row


def _mean_std(a: list[float]) -> tuple[float, float]:
    x = np.asarray(a, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=0))


def run_benchmark(
    *,
    ks: list[int],
    train_fracs: list[float],
    seeds: list[int],
    n_intervention: int,
    grid_test_frac: float,
    epochs: int,
    lr: float,
    batch_size: int,
    d_model: int,
    hidden: int,
    weight_tol: float,
    verbose: bool,
    L: int,
    J: float,
    g: float,
    timesteps_all: list[float],
    basis: str,
    basis_seed: int | None,
    max_bond_dim: int,
    parallel_tomo: bool,
    parallel_intervention: bool,
    out_dir: Path,
    bloch_td_threshold: float | None,
    wrho_rmse_threshold: float | None,
) -> None:
    k_max = max(ks)
    if len(timesteps_all) < k_max:
        msg = f"Need len(timesteps) >= max(ks)={k_max}, got {len(timesteps_all)}"
        raise ValueError(msg)
    if any(t <= 0 for t in timesteps_all):
        raise ValueError("all timesteps must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Gold-standard surrogate benchmark ===")
    print("  Labels: exhaustive TomographyEstimate (dense); models never see full grid at train time.")
    print(f"  Physics: L={L} J={J} g={g}  basis={basis!r}  max_bond_dim={max_bond_dim}")
    print(f"  ks={ks}  train_fracs={train_fracs}  seeds={seeds}")
    print(f"  intervention test size={n_intervention}  grid_test_frac={grid_test_frac}")

    op = MPO.ising(length=L, J=J, g=g)
    basis_t = cast(TomographyBasis, basis)

    for k in ks:
        timesteps = timesteps_all[:k]
        solver_dt = float(min(timesteps))
        params = AnalogSimParams(
            dt=solver_dt,
            solver="MCWF",
            show_progress=False,
            max_bond_dim=max_bond_dim,
        )
        n_grid = 16**k
        print(f"\n{'=' * 72}\n=== k={k}  build exhaustive gold  (grid {n_grid}) ===")
        print(f"  timesteps={timesteps}  AnalogSimParams.dt={solver_dt}")

        data = _build_tomography_data(
            "exhaustive",
            operator=op,
            sim_params=params,
            timesteps=timesteps,
            parallel=parallel_tomo,
            num_samples=n_grid,
            num_trajectories=1,
            noise_model=None,
            seed=0,
            basis=basis_t,
            basis_seed=basis_seed,
            proposal="uniform",
            ess_threshold=0.5,
            prep_mixture_eps=0.1,
            resample=False,
        )
        estimate = _to_dense(data)
        cov = summarize_grid_coverage(estimate, k=k, weight_tol=weight_tol)
        print(
            f"  positive-weight cells: {cov.n_weight_positive} / {cov.n_grid}  "
            f"(tol={cov.weight_tol})"
        )

        ds = tomography_estimate_to_ml_dataset(
            estimate,
            weight_tol=weight_tol,
            use_estimator_weight=True,
            include_weighted_rho8=True,
        )
        assert ds.weighted_rho8 is not None
        N = ds.sequence_indices.shape[0]
        seq_all = ds.sequence_indices
        bloch_all = ds.bloch_target

        v_last1 = _max_within_suffix_msq(bloch_all, seq_all, 1)
        v_last2 = _max_within_suffix_msq(bloch_all, seq_all, min(2, k))
        print("\n--- Target diagnostics (normalized Bloch, full positive-weight set) ---")
        print(f"  max within-last-token mean sq deviation:      {v_last1:.4e}")
        print(f"  max within-last-2-token mean sq deviation:    {v_last2:.4e}")
        if v_last1 < 1e-10:
            print(
                "  WARNING: Bloch nearly determined by last token alone - "
                "do not use Bloch error as evidence vs 16^k; rely on w*rho metrics."
            )
        elif v_last2 < 1e-10 and k >= 2:
            print(
                "  WARNING: Bloch nearly determined by last two tokens - "
                "order-sensitive signal in Bloch is weak."
            )

        # Intervention sequences (fixed for this k; disjoint from a placeholder empty set)
        rng_i = np.random.default_rng((basis_seed or 0) + 1000 + k)
        forbidden: set[tuple[int, ...]] = set()
        int_list = _sample_intervention_sequences(rng_i, k, n_intervention, forbidden)
        print(f"\n--- Simulating {len(int_list)} intervention sequences (backend) ---")
        int_seq, int_bloch, int_wrho8, int_w = _simulate_intervention_set(
            int_list,
            operator=op,
            sim_params=params,
            timesteps=timesteps,
            basis=basis_t,
            basis_seed=basis_seed,
            parallel=parallel_intervention,
        )

        n_test = max(64, int(grid_test_frac * N))
        n_test = min(n_test, N - 16)  # leave room for smallest train
        if n_test < 32:
            n_test = max(1, N // 5)

        frac_results: list[dict[str, Any]] = []

        for frac in sorted(set(train_fracs)):
            n_train_req = max(1, int(round(frac * n_grid)))
            rows: list[dict[str, float]] = []
            for seed in seeds:
                rng = np.random.default_rng(seed + k * 100_000)
                perm = rng.permutation(N)
                grid_test_idx = perm[:n_test]
                train_pool = perm[n_test:]
                n_tr = min(n_train_req, len(train_pool))
                if n_tr < 1:
                    continue
                train_idx = rng.choice(train_pool, size=n_tr, replace=False)
                if verbose and seed == seeds[0]:
                    print(
                        f"  frac={frac:.4f}  n_train={n_tr}  grid_test={n_test}  "
                        f"seed={seed}"
                    )
                row = run_one_split(
                    ds=ds,
                    train_idx=train_idx,
                    grid_test_idx=grid_test_idx,
                    int_seq=int_seq,
                    int_bloch=int_bloch,
                    int_wrho8=int_wrho8,
                    int_w=int_w,
                    k=k,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    d_model=d_model,
                    hidden=hidden,
                    device=torch.device("cpu"),
                )
                row["n_train"] = float(n_tr)
                row["frac"] = float(frac)
                row["seed"] = float(seed)
                rows.append(row)

            if not rows:
                continue

            def col(key: str) -> list[float]:
                return [float(r[key]) for r in rows]

            n_tr_mean, _ = _mean_std(col("n_train"))
            n_tr_i = int(round(n_tr_mean))

            m_nn_gb, s_nn_gb = _mean_std(col("grid_nn_bloch_td_uw"))
            m_nn_gw, s_nn_gw = _mean_std(col("grid_nn_wrho_rmse_w"))
            m_nn_ib, s_nn_ib = _mean_std(col("int_nn_bloch_td_uw"))
            m_nn_iw, s_nn_iw = _mean_std(col("int_nn_wrho_rmse_w"))

            best_b_vals = [
                min(
                    float(r["grid_mem0_bloch_td_uw"]),
                    *[float(r[f"grid_mem{m}_bloch_td_uw_full"]) for m in range(1, k + 1)],
                )
                for r in rows
            ]
            best_w_vals = [
                min(
                    float(r["grid_mem0_wrho_rmse_w"]),
                    *[float(r[f"grid_mem{m}_wrho_rmse_w_full"]) for m in range(1, k + 1)],
                )
                for r in rows
            ]
            m_best_b, s_best_b = _mean_std(best_b_vals)
            m_best_w, s_best_w = _mean_std(best_w_vals)

            beats_b = _mean_std(col("grid_nn_beats_best_mem_bloch"))[0]
            beats_w = _mean_std(col("grid_nn_beats_best_mem_wrho"))[0]

            row_agg: dict[str, Any] = {
                "k": k,
                "frac": frac,
                "n_train": n_tr_i,
                "grid_nn_bloch_td_uw": m_nn_gb,
                "grid_nn_wrho_rmse_w": m_nn_gw,
                "int_nn_bloch_td_uw": m_nn_ib,
                "int_nn_wrho_rmse_w": m_nn_iw,
                "grid_best_mem_bloch_td": m_best_b,
                "grid_best_mem_wrho_rmse": m_best_w,
                "beats_mem_bloch_frac": beats_b,
                "beats_mem_wrho_frac": beats_w,
            }
            for m in range(0, k + 1):
                kb_g = "grid_mem0_bloch_td_uw" if m == 0 else f"grid_mem{m}_bloch_td_uw_full"
                kw_g = "grid_mem0_wrho_rmse_w" if m == 0 else f"grid_mem{m}_wrho_rmse_w_full"
                kb_i = "int_mem0_bloch_td_uw" if m == 0 else f"int_mem{m}_bloch_td_uw_full"
                kw_i = "int_mem0_wrho_rmse_w" if m == 0 else f"int_mem{m}_wrho_rmse_w_full"
                row_agg[f"bloch_grid_m{m}"] = _mean_std(col(kb_g))[0]
                row_agg[f"wrho_grid_m{m}"] = _mean_std(col(kw_g))[0]
                row_agg[f"bloch_int_m{m}"] = _mean_std(col(kb_i))[0]
                row_agg[f"wrho_int_m{m}"] = _mean_std(col(kw_i))[0]
            frac_results.append(row_agg)

            cov_frac_k = _mean_std([float(r[f"grid_mem{k}_cov_frac"]) for r in rows])[0]
            extra_cov = ""
            if cov_frac_k > 1e-5:
                vals_b = np.array(
                    [float(r[f"grid_mem{k}_bloch_td_uw_cov"]) for r in rows],
                    dtype=np.float64,
                )
                vals_w = np.array(
                    [float(r[f"grid_mem{k}_wrho_rmse_w_cov"]) for r in rows],
                    dtype=np.float64,
                )
                vals_b = vals_b[np.isfinite(vals_b)]
                vals_w = vals_w[np.isfinite(vals_w)]
                if vals_b.size and vals_w.size:
                    extra_cov = (
                        f"\n    grid  mem-(m=k) covered-only vs full: "
                        f"Bloch TD_uw cov={float(vals_b.mean()):.5f}  "
                        f"w*rho RMSE_w cov={float(vals_w.mean()):.5f}  "
                        f"test cov frac={cov_frac_k:.3f}"
                    )
            print(
                f"\n>>> k={k}  train_frac={frac:.3%}  n_train~{n_tr_i}  seeds={len(rows)}\n"
                f"    grid  NN Bloch TD_uw={m_nn_gb:.5f}+-{s_nn_gb:.5f}   "
                f"w*rho RMSE_w={m_nn_gw:.5f}+-{s_nn_gw:.5f}\n"
                f"    int   NN Bloch TD_uw={m_nn_ib:.5f}+-{s_nn_ib:.5f}   "
                f"w*rho RMSE_w={m_nn_iw:.5f}+-{s_nn_iw:.5f}\n"
                f"    grid  best mem-(m) Bloch TD_uw (full) min_m mean={m_best_b:.5f}  "
                f"w*rho RMSE_w min_m mean={m_best_w:.5f}\n"
                f"    grid  frac seeds NN beats best mem (full): Bloch={beats_b:.2f}  w*rho={beats_w:.2f}"
                f"{extra_cov}"
            )

        # ----- Figure: test error vs n_train -----
        if frac_results:
            fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            rs = sorted(frac_results, key=lambda z: int(z["n_train"]))
            xv = [int(r["n_train"]) for r in rs]

            def style_plot(
                ax: Any,
                nn_key: str,
                ylabel: str,
                title: str,
                prefix: str,
            ) -> None:
                ax.set_title(title)
                ax.set_xlabel("Training sequences")
                ax.set_ylabel(ylabel)
                ax.plot(xv, [r[nn_key] for r in rs], "o-", label="NN", lw=2, color="C0")
                for m in range(0, k + 1):
                    mk = f"{prefix}_m{m}"
                    ax.plot(
                        xv,
                        [r[mk] for r in rs],
                        "--",
                        label=f"mem-{m}",
                        alpha=0.75,
                    )
                ax.legend(fontsize=7, ncol=3)
                ax.grid(True, alpha=0.3)

            style_plot(
                axs[0, 0],
                "grid_nn_bloch_td_uw",
                "mean TD",
                f"k={k} grid - Bloch (uw)",
                "bloch_grid",
            )
            style_plot(
                axs[0, 1],
                "grid_nn_wrho_rmse_w",
                "weighted RMSE",
                f"k={k} grid - w*rho (8D, weighted)",
                "wrho_grid",
            )
            style_plot(
                axs[1, 0],
                "int_nn_bloch_td_uw",
                "mean TD",
                f"k={k} intervention - Bloch (uw)",
                "bloch_int",
            )
            style_plot(
                axs[1, 1],
                "int_nn_wrho_rmse_w",
                "weighted RMSE",
                f"k={k} intervention - w*rho (8D, weighted)",
                "wrho_int",
            )
            fig_path = out_dir / f"surrogate_efficiency_k{k}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"\n  Saved figure: {fig_path}")

        # ----- Smallest n_train hitting accuracy (grid, mean over seeds) -----
        print("\n--- Accuracy threshold report (grid test, mean over seeds) ---")
        rs_sorted = sorted(frac_results, key=lambda z: int(z["n_train"]))
        n_full = 16**k

        def _report_thr(
            name: str,
            thr: float | None,
            nn_key: str,
            best_key: str,
        ) -> None:
            if thr is None:
                print(f"  {name}: (no threshold set)")
                return
            hit = next((r for r in rs_sorted if float(r[nn_key]) < thr), None)
            if hit is None:
                print(f"  {name}: threshold {thr} not reached in this sweep.")
                return
            nnv = float(hit[nn_key])
            bestv = float(hit[best_key])
            frac_tr = int(hit["n_train"]) / n_full
            beats = nnv < bestv
            print(
                f"  {name}: first hit at n_train={hit['n_train']} "
                f"({frac_tr:.3%} of 16^{k})  NN={nnv:.5f}  best_mem={bestv:.5f}  "
                f"NN beats best mem-m? {beats}"
            )

        _report_thr("Bloch TD_uw", bloch_td_threshold, "grid_nn_bloch_td_uw", "grid_best_mem_bloch_td")
        _report_thr("w*rho RMSE_w", wrho_rmse_threshold, "grid_nn_wrho_rmse_w", "grid_best_mem_wrho_rmse")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Default: channel predictor (rho_in, E_1..E_k)->rho_out. "
        "Use --exhaustive_grid for tomography grid benchmark."
    )
    p.add_argument(
        "--exhaustive_grid",
        action="store_true",
        help="Run exhaustive-label grid benchmark instead of the channel predictor.",
    )
    p.add_argument(
        "--comb_reference_minimal",
        action="store_true",
        help="Run minimal channel benchmark against exhaustive DenseComb reference.",
    )
    p.add_argument(
        "--debug_compare",
        type=int,
        default=0,
        help="Minimal-mode only: strict backend vs exhaustive debug compare count (aborts if inconsistent).",
    )
    p.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument(
        "--train_fracs",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05, 0.10],
        help="Grid: fractions of 16^k. Channel: fractions of train pool unless --train_sizes.",
    )
    p.add_argument(
        "--train_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Channel only: absolute train sizes (capped to pool). Overrides train_fracs.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_intervention", type=int, default=256, help="Backend-simulated held-out sequences.")
    p.add_argument("--grid_test_frac", type=float, default=0.2, help="Hold out fraction of positive-weight grid.")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--weight_tol", type=float, default=1e-30)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--g", type=float, default=0.75)
    p.add_argument(
        "--timesteps",
        type=float,
        nargs="+",
        default=[0.15, 0.12, 0.18, 0.14],
        help="Segment lengths; len must be >= max(k). First k used per run.",
    )
    p.add_argument(
        "--basis",
        type=str,
        default="tetrahedral",
        choices=["standard", "tetrahedral", "random"],
        help="Fixed tomography basis for the whole run (channel default: tetrahedral).",
    )
    p.add_argument("--basis_seed", type=int, default=12345)
    p.add_argument(
        "--backend_init_mode",
        type=str,
        default="eigenstate",
        choices=["eigenstate", "purified"],
        help="How to turn rho_in into an initial MCWF pure state on the backend.",
    )
    p.add_argument("--max_bond_dim", type=int, default=16)
    p.add_argument(
        "--parallel_tomo",
        action="store_true",
        help="Parallel exhaustive tomography build (faster, more RAM).",
    )
    p.add_argument(
        "--parallel_intervention",
        action="store_true",
        help="Parallel backend sim for intervention set (can fail on some Windows setups).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="benchmark_results",
        help="Directory for PNG figures.",
    )
    p.add_argument(
        "--bloch_td_threshold",
        type=float,
        default=None,
        help="Report first n_train where mean grid Bloch TD_uw drops below this.",
    )
    p.add_argument(
        "--wrho_rmse_threshold",
        type=float,
        default=None,
        help="First n_train where mean grid w*rho RMSE_w drops below this (primary sample-efficiency target).",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=2048,
        help="Channel mode: total labeled pool (train+test). Grid mode: unused.",
    )
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.2,
        help="Channel mode: held-out test fraction of pool. Grid mode: unused (see --grid_test_frac).",
    )
    p.add_argument(
        "--parallel_channel_sim",
        action="store_true",
        help="Parallel backend jobs when generating channel training pool.",
    )
    args = p.parse_args()
    if args.comb_reference_minimal:
        run_channel_predictor_vs_exhaustive_comb(
            ks=list(args.ks),
            n_samples=int(args.n_samples),
            seeds=list(args.seeds),
            epochs=int(args.epochs),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            hidden=int(args.hidden),
            L=int(args.L),
            J=float(args.J),
            g=float(args.g),
            timesteps_all=list(args.timesteps),
            basis=str(args.basis),
            basis_seed=int(args.basis_seed) if args.basis_seed is not None else None,
            max_bond_dim=int(args.max_bond_dim),
            parallel_sim=bool(args.parallel_channel_sim),
            backend_init_mode=str(args.backend_init_mode),
            out_dir=Path(args.out_dir),
            debug_compare=int(args.debug_compare),
        )
        return
    if not args.exhaustive_grid:
        run_channel_predictor_benchmark(
            ks=list(args.ks),
            n_pool=int(args.n_samples),
            test_frac=float(args.test_frac),
            train_fracs=list(args.train_fracs),
            train_sizes=list(args.train_sizes) if args.train_sizes is not None else None,
            seeds=list(args.seeds),
            epochs=int(args.epochs),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            hidden=int(args.hidden),
            L=int(args.L),
            J=float(args.J),
            g=float(args.g),
            timesteps_all=list(args.timesteps),
            basis=str(args.basis),
            basis_seed=int(args.basis_seed) if args.basis_seed is not None else None,
            max_bond_dim=int(args.max_bond_dim),
            parallel_sim=bool(args.parallel_channel_sim),
            backend_init_mode=str(args.backend_init_mode),
            out_dir=Path(args.out_dir),
        )
        return

    run_benchmark(
        ks=list(args.ks),
        train_fracs=list(args.train_fracs),
        seeds=list(args.seeds),
        n_intervention=int(args.n_intervention),
        grid_test_frac=float(args.grid_test_frac),
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        d_model=int(args.d_model),
        hidden=int(args.hidden),
        weight_tol=float(args.weight_tol),
        verbose=bool(args.verbose),
        L=int(args.L),
        J=float(args.J),
        g=float(args.g),
        timesteps_all=list(args.timesteps),
        basis=str(args.basis),
        basis_seed=int(args.basis_seed) if args.basis_seed is not None else None,
        max_bond_dim=int(args.max_bond_dim),
        parallel_tomo=bool(args.parallel_tomo),
        parallel_intervention=bool(args.parallel_intervention),
        out_dir=Path(args.out_dir),
        bloch_td_threshold=args.bloch_td_threshold,
        wrho_rmse_threshold=args.wrho_rmse_threshold,
    )


if __name__ == "__main__":
    main()
