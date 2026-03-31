# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Fixed-k training and generalization to k' ≠ k (Transformer rho0, single Ising regime).

**Data (same as the continuous branch of ``benchmark_basis_training_convergence``):** every
trajectory is built with ``sample_random_intervention_sequence``: rank-1 prep/effect CP maps, backend
via ``simulate_backend_trajectory_batch`` with ``e_features_rows`` = flattened Choi rows (shape
``k × 32``). There is no discrete 16-letter alphabet in the features seen by the network.

**Training:** ``N_train`` trajectories of length ``k_train``; optional train/val split from that
pool; MSE on packed ``rho_seq`` targets vs. :class:`~.sequence_models.TransformerComb` in
``input_mode="rho0"`` (initial reduced state broadcast to every step).

**Evaluation:** for each ``k_test``, a **fresh** batch of ``N_test`` trajectories of length
``k_test`` (independent RNG). Ground-truth final state is ``rho_seq[:, -1, :]`` from the backend.
Model prediction uses a single ``forward`` on the full ``E`` tensor (causal transformer).

Reports Frobenius MSE and Pauli X/Z expectation errors on the final reduced state.

**Plots:** one series per metric vs ``k_test``. Passing a single ``--k_tests 8`` gives one x-value;
error bars then come only from seed spread. For a generalization **curve**, pass several values
(e.g. ``--k_tests 2 4 6 8 10``) or ``--k_test_sweep 1 12 1``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mqt.yaqs.characterization.tomography.ml_dataset import (
    build_rho_prev_rho_target,
    mean_frobenius_mse_rho8,
    trajectory_batch_to_tensors,
)
from mqt.yaqs.characterization.tomography.predictor_encoding import (
    build_choi_feature_table,
    random_density_matrix,
    sample_random_intervention_sequence,
    unpack_rho8,
)
from mqt.yaqs.characterization.tomography.process_tomography import simulate_backend_trajectory_batch
from mqt.yaqs.characterization.tomography.sequence_models import TransformerComb
from mqt.yaqs.characterization.tomography.tomography_utils import build_initial_psi, make_mcwf_static_context
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed k_train, generalize to k_test (Transformer rho0).")
    p.add_argument("--out_dir", type=str, default="benchmark_results")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--k_train", type=int, required=True)
    p.add_argument(
        "--k_tests",
        type=int,
        nargs="*",
        default=[],
        help="k_test horizons to evaluate (after training). Pass several for a curve, e.g. 2 4 6 8.",
    )
    p.add_argument(
        "--k_test_sweep",
        type=int,
        nargs=3,
        metavar=("MIN", "MAX", "STEP"),
        default=None,
        help="Optional sequence MIN, MIN+STEP, … MAX (inclusive), merged with --k_tests. Example: 1 12 1",
    )
    p.add_argument("--N_train", type=int, default=512)
    p.add_argument("--N_test", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--val_frac_of_train", type=float, default=0.2)
    p.add_argument("--parallel_channel_sim", action="store_true")
    p.add_argument("--max_bond_dim", type=int, default=None)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--layernorm_in", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dim_ff", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--prefix_loss",
        type=str,
        default="full",
        choices=["full", "random", "all"],
        help="full = full-sequence MSE at k_train (matches evaluating horizon k'); "
        "all/random = prefix losses (see training loop).",
    )
    p.add_argument("--no_plots", action="store_true")
    ns = p.parse_args()
    k_list = list(ns.k_tests)
    if ns.k_test_sweep is not None:
        lo, hi, step = (int(ns.k_test_sweep[0]), int(ns.k_test_sweep[1]), int(ns.k_test_sweep[2]))
        if step <= 0:
            p.error("--k_test_sweep STEP must be positive.")
        if lo > hi:
            p.error("--k_test_sweep MIN must be <= MAX.")
        k_list.extend(range(lo, hi + 1, step))
    k_tests = sorted({int(k) for k in k_list if int(k) > 0})
    if not k_tests:
        p.error("Set at least one positive k_test via --k_tests and/or --k_test_sweep.")
    setattr(ns, "_k_tests_resolved", k_tests)
    return ns


def _state_from_rank1_projector(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(np.asarray(P, dtype=np.complex128).reshape(2, 2))
    idx = int(np.argmax(w.real))
    psi = v[:, idx]
    nrm = float(np.linalg.norm(psi))
    if nrm < 1e-15:
        return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return (psi / nrm).astype(np.complex128)


def _backend_placeholder_choi_table() -> np.ndarray:
    """Shape (16, 32) required by :func:`simulate_backend_trajectory_batch`.

    For continuous trajectories we always pass ``e_features_rows``; the worker then sets
    ``E_rows`` from those tensors and **does not** index this table (see
    ``worker_initial_psi_trajectory``).  Entries are arbitrary placeholders.
    """
    z = np.zeros((4, 4), dtype=np.complex128)
    return build_choi_feature_table([z] * 16)


def _make_continuous_dataset(
    *,
    k: int,
    n: int,
    rng: np.random.Generator,
    L: int,
    op: MPO,
    params: AnalogSimParams,
    static_ctx: Any,
    choi_feat_table: np.ndarray,
    parallel: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas_rows: list[np.ndarray] = []
    psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]] = []
    initial_psis: list[np.ndarray] = []
    e_features_rows: list[np.ndarray] = []
    for _ in range(int(n)):
        rho_in = random_density_matrix(rng)
        maps, rows_feat = sample_random_intervention_sequence(int(k), rng)
        pairs = []
        for emap in maps:
            rho_prep = np.asarray(getattr(emap, "rho_prep"), dtype=np.complex128)
            E = np.asarray(getattr(emap, "effect"), dtype=np.complex128)
            psi_meas = _state_from_rank1_projector(E)
            psi_prep = _state_from_rank1_projector(rho_prep)
            pairs.append((psi_meas, psi_prep))
        alphas_rows.append(np.zeros(int(k), dtype=np.int64))
        psi_pairs_list.append(pairs)
        e_features_rows.append(rows_feat.astype(np.float32))
        initial_psis.append(build_initial_psi(rho_in, length=int(L), rng=rng, init_mode="eigenstate"))
    timesteps = [float(params.dt)] * int(k)
    samples = simulate_backend_trajectory_batch(
        operator=op,
        sim_params=params,
        timesteps=timesteps,
        psi_pairs_list=psi_pairs_list,
        alphas_rows=alphas_rows,
        initial_psis=initial_psis,
        choi_feat_table=choi_feat_table,
        e_features_rows=e_features_rows,
        parallel=bool(parallel),
        static_ctx=static_ctx,
        context_vec=None,
    )
    rho0_np, E_np, rho_seq_np, _ctx = trajectory_batch_to_tensors(samples)
    return rho0_np, E_np, rho_seq_np


def _train_transformer(
    *,
    model: Any,
    E_tr: Any,
    rho0_tr: Any,
    tgt_tr: Any,
    E_va: Any,
    rho0_va: Any,
    tgt_va: Any,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_clip: float,
    prefix_loss: str,
) -> Any:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(E_tr, rho0_tr, tgt_tr),
        batch_size=min(int(batch_size), max(1, int(E_tr.shape[0]))),
        shuffle=True,
    )
    k_max = int(tgt_tr.shape[1])
    best = float("inf")
    best_state = None
    for _ep in range(int(epochs)):
        model.train()
        for E_b, rho0_b, tgt_b in loader:
            opt.zero_grad(set_to_none=True)
            if prefix_loss == "full" or k_max <= 1:
                pred = model(E_b, rho0_b)
                loss = loss_fn(pred, tgt_b)
            elif prefix_loss == "random":
                Ls = int(torch.randint(low=1, high=k_max + 1, size=(1,), device=E_b.device).item())
                pred = model(E_b[:, :Ls, :], rho0_b)
                loss = loss_fn(pred, tgt_b[:, :Ls, :])
            elif prefix_loss == "all":
                losses = []
                for Ls in range(1, k_max + 1):
                    pred_L = model(E_b[:, :Ls, :], rho0_b)
                    losses.append(loss_fn(pred_L, tgt_b[:, :Ls, :]))
                loss = torch.stack(losses, dim=0).mean()
            else:
                raise ValueError(prefix_loss)
            loss.backward()
            if grad_clip and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_va = model(E_va, rho0_va)
            val = float(loss_fn(pred_va, tgt_va).detach().cpu().item())
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _mean_abs_obs_err(pred_rho8: np.ndarray, tgt_rho8: np.ndarray, *, obs: np.ndarray) -> float:
    p = np.asarray(pred_rho8, dtype=np.float64)
    t = np.asarray(tgt_rho8, dtype=np.float64)
    O = np.asarray(obs, dtype=np.complex128).reshape(2, 2)
    errs = []
    for i in range(int(p.shape[0])):
        rho_p = unpack_rho8(p[i])
        rho_t = unpack_rho8(t[i])
        errs.append(abs(float(np.trace(O @ rho_p).real) - float(np.trace(O @ rho_t).real)))
    return float(np.mean(errs)) if errs else 0.0


@dataclass(frozen=True)
class Row:
    k_train: int
    k_test: int
    N_train: int
    N_test: int
    seed: int
    final_frob: float
    obs_x_mae: float
    obs_z_mae: float


def _plot_metric(rows: list[Row], out_png: Path, title: str, metric: str) -> None:
    import matplotlib.pyplot as plt

    key = "final_frob" if metric == "frob" else ("obs_x_mae" if metric == "x" else "obs_z_mae")
    by_k: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        by_k[int(r.k_test)].append(float(getattr(r, key)))
    k_sorted = sorted(by_k.keys())
    from statistics import mean, stdev

    m = [mean(by_k[k]) for k in k_sorted]
    s = [stdev(by_k[k]) if len(by_k[k]) > 1 else 0.0 for k in k_sorted]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(k_sorted) == 1:
        ax.errorbar(k_sorted, m, yerr=s, fmt="o", capsize=4, markersize=8)
    else:
        ax.errorbar(k_sorted, m, yerr=s, fmt="o-", capsize=3)
    ax.set_xlabel("k_test")
    ylab = {"frob": "Frobenius MSE", "x": "mean |d<X>|", "z": "mean |d<Z>|"}[metric]
    ax.set_ylabel(ylab)
    if metric == "frob":
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    J = 1.0
    g = 1.0
    dt = 0.1
    args = parse_args()
    k_tests_plan = list(getattr(args, "_k_tests_resolved"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"k_test grid ({len(k_tests_plan)} points): {k_tests_plan}")
    if len(k_tests_plan) == 1:
        print("Note: only one k_test — plots show a single x with error bars from --seeds; add more k' for a curve.")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    choi_feat_table = _backend_placeholder_choi_table()
    op = MPO.ising(length=int(args.L), J=J, g=g)
    params = AnalogSimParams(dt=float(dt), solver="MCWF", show_progress=False, max_bond_dim=args.max_bond_dim)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    obs_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    obs_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    rows: list[Row] = []
    for seed in args.seeds:
        rng = np.random.default_rng(int(seed) + 12345 * int(args.k_train))
        rho0_tr, E_tr, rho_seq_tr = _make_continuous_dataset(
            k=int(args.k_train),
            n=int(args.N_train),
            rng=rng,
            L=int(args.L),
            op=op,
            params=params,
            static_ctx=static_ctx,
            choi_feat_table=choi_feat_table,
            parallel=bool(args.parallel_channel_sim),
        )
        _, rho_tgt = build_rho_prev_rho_target(rho0_tr, rho_seq_tr)
        idx = np.arange(int(args.N_train), dtype=np.int64)
        rng.shuffle(idx)
        n_val = max(1, min(int(round(float(args.val_frac_of_train) * len(idx))), len(idx) - 1))
        va_idx, tr_idx = idx[:n_val], idx[n_val:]
        E_tr_t = torch.as_tensor(E_tr[tr_idx], dtype=torch.float32, device=device)
        r0_tr_t = torch.as_tensor(rho0_tr[tr_idx], dtype=torch.float32, device=device)
        tg_tr_t = torch.as_tensor(rho_tgt[tr_idx], dtype=torch.float32, device=device)
        E_va_t = torch.as_tensor(E_tr[va_idx], dtype=torch.float32, device=device)
        r0_va_t = torch.as_tensor(rho0_tr[va_idx], dtype=torch.float32, device=device)
        tg_va_t = torch.as_tensor(rho_tgt[va_idx], dtype=torch.float32, device=device)
        tfm = TransformerComb(
            d_e=int(E_tr.shape[-1]),
            d_rho=8,
            d_model=int(args.d_model),
            nhead=int(args.nhead),
            num_layers=int(args.layers),
            dim_ff=int(args.dim_ff),
            dropout=float(args.dropout),
            layernorm_in=bool(args.layernorm_in),
            memory_window=None,
        ).to(device)
        tfm = _train_transformer(
            model=tfm,
            E_tr=E_tr_t,
            rho0_tr=r0_tr_t,
            tgt_tr=tg_tr_t,
            E_va=E_va_t,
            rho0_va=r0_va_t,
            tgt_va=tg_va_t,
            epochs=int(args.epochs),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            grad_clip=float(args.grad_clip),
            prefix_loss=str(args.prefix_loss),
        )
        tfm.eval()
        for k_test in k_tests_plan:
            rng_te = np.random.default_rng(int(seed) + 999_983 * int(k_test) + 77_777 * int(args.k_train))
            r0_te, E_te, rho_seq_te = _make_continuous_dataset(
                k=int(k_test),
                n=int(args.N_test),
                rng=rng_te,
                L=int(args.L),
                op=op,
                params=params,
                static_ctx=static_ctx,
                choi_feat_table=choi_feat_table,
                parallel=bool(args.parallel_channel_sim),
            )
            tgt_final = rho_seq_te[:, -1, :].astype(np.float32)
            with torch.no_grad():
                E_te_t = torch.as_tensor(E_te, dtype=torch.float32, device=device)
                r0_te_t = torch.as_tensor(r0_te, dtype=torch.float32, device=device)
                pred_seq = tfm(E_te_t, r0_te_t).cpu().numpy().astype(np.float32)
                pred_final = pred_seq[:, -1, :]
            fr = mean_frobenius_mse_rho8(pred_final, tgt_final)
            ox = _mean_abs_obs_err(pred_final, tgt_final, obs=obs_x)
            oz = _mean_abs_obs_err(pred_final, tgt_final, obs=obs_z)
            rows.append(
                Row(
                    k_train=int(args.k_train),
                    k_test=int(k_test),
                    N_train=int(args.N_train),
                    N_test=int(args.N_test),
                    seed=int(seed),
                    final_frob=float(fr),
                    obs_x_mae=float(ox),
                    obs_z_mae=float(oz),
                )
            )
            print(f"seed={seed} k_train={args.k_train} k_test={k_test} frob={fr:.3e} |dX|={ox:.3e} |dZ|={oz:.3e}")

    per_csv = out_dir / "benchmark_fixed_k_generalization.csv"
    if rows:
        cols = list(asdict(rows[0]).keys())
        per_csv.write_text("\n".join([",".join(cols), *[",".join(str(asdict(r)[c]) for c in cols) for r in rows]]) + "\n", encoding="utf-8")
        print(f"\nWrote {len(rows)} rows to {per_csv}")

    agg: dict[tuple[int, int], list[tuple[float, float, float]]] = defaultdict(list)
    for r in rows:
        agg[(r.k_train, r.k_test)].append((r.final_frob, r.obs_x_mae, r.obs_z_mae))
    agg_lines = ["k_train,k_test,mean_frob,std_frob,mean_dX,std_dX,mean_dZ,std_dZ,n"]
    from statistics import mean, stdev

    for (kt, ks), vals in sorted(agg.items()):
        f = [v[0] for v in vals]
        x = [v[1] for v in vals]
        z = [v[2] for v in vals]
        agg_lines.append(
            f"{kt},{ks},{mean(f)},{stdev(f) if len(f) > 1 else 0},{mean(x)},{stdev(x) if len(x) > 1 else 0},{mean(z)},{stdev(z) if len(z) > 1 else 0},{len(vals)}"
        )
    (out_dir / "benchmark_fixed_k_generalization_aggregate.csv").write_text("\n".join(agg_lines) + "\n", encoding="utf-8")

    if rows and not args.no_plots:
        kt = int(args.k_train)
        sub = f"continuous_ktrain{kt}"
        _plot_metric(
            rows,
            out_dir / f"benchmark_fixed_k_generalization_frob_{sub}.png",
            f"Frobenius MSE (continuous random, k_train={kt})",
            "frob",
        )
        _plot_metric(
            rows,
            out_dir / f"benchmark_fixed_k_generalization_obs_x_{sub}.png",
            f"|d<X>| (continuous random, k_train={kt})",
            "x",
        )
        _plot_metric(
            rows,
            out_dir / f"benchmark_fixed_k_generalization_obs_z_{sub}.png",
            f"|d<Z>| (continuous random, k_train={kt})",
            "z",
        )


if __name__ == "__main__":
    main()
