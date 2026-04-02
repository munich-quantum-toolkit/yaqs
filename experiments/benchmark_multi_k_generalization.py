# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Multi-k_train generalization benchmark (continuous random interventions).

Extends ``benchmark_fixed_k_generalization.py`` by training separate models at several
``k_train`` values and evaluating each model on a grid of ``k_test`` values.

Outputs:
  - Per-run CSV with one row per (k_train, k_test, seed)
  - Aggregate CSV with mean/std over seeds per (k_train, k_test)
  - Plots of error vs k_test and vs gap (k_test - k_train)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mqt.yaqs.characterization.process_tensors.core.utils import make_mcwf_static_context
from mqt.yaqs.characterization.process_tensors.surrogates.data import stack_rollouts
from mqt.yaqs.characterization.process_tensors.surrogates.encoding import unpack_rho8
from mqt.yaqs.characterization.process_tensors.surrogates.metrics import _mean_frobenius_mse_rho8 as mean_frobenius_mse_rho8
from mqt.yaqs.characterization.process_tensors.surrogates.model import TransformerComb
from mqt.yaqs.characterization.process_tensors.surrogates.utils import build_initial_psi, _random_density_matrix as random_density_matrix, _sample_random_intervention_sequence as sample_random_intervention_sequence
from mqt.yaqs.characterization.process_tensors.surrogates.workflow import _simulate_sequences as simulate_sequences
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multiple k_train and evaluate on k_test grid (continuous random).")
    p.add_argument("--out_dir", type=str, default="benchmark_results")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--k_trains", type=int, nargs="+", required=True, help="Training horizons, e.g. 4 8 12 16.")
    p.add_argument(
        "--k_tests",
        type=int,
        nargs="*",
        default=[],
        help="Explicit k_test horizons. Merged with --k_test_sweep.",
    )
    p.add_argument(
        "--k_test_sweep",
        type=int,
        nargs=3,
        metavar=("MIN", "MAX", "STEP"),
        default=None,
        help="Optional k_test sequence MIN, MIN+STEP, … MAX (inclusive). Example: 2 20 2",
    )
    p.add_argument("--N_train", type=int, default=128)
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
        help="full = full-sequence MSE at k_train; all/random = prefix losses.",
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
    k_trains = sorted({int(k) for k in ns.k_trains if int(k) > 0})
    if not k_trains:
        p.error("Set at least one positive k_train via --k_trains.")
    setattr(ns, "_k_tests_resolved", k_tests)
    setattr(ns, "_k_trains_resolved", k_trains)
    return ns


def _state_from_rank1_projector(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(np.asarray(P, dtype=np.complex128).reshape(2, 2))
    idx = int(np.argmax(w.real))
    psi = v[:, idx]
    nrm = float(np.linalg.norm(psi))
    if nrm < 1e-15:
        return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return (psi / nrm).astype(np.complex128)


def _make_continuous_dataset(
    *,
    k: int,
    n: int,
    rng: np.random.Generator,
    L: int,
    op: MPO,
    params: AnalogSimParams,
    static_ctx: Any,
    parallel: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        psi_pairs_list.append(pairs)
        e_features_rows.append(rows_feat.astype(np.float32))
        initial_psis.append(build_initial_psi(rho_in, length=int(L), rng=rng, init_mode="eigenstate"))
    timesteps = [float(params.dt)] * int(k)
    samples = simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=timesteps,
        psi_pairs_list=psi_pairs_list,
        initial_psis=initial_psis,
        e_features_rows=e_features_rows,
        parallel=bool(parallel),
        show_progress=False,
        record_step_states=True,
        static_ctx=static_ctx,
        context_vec=None,
    )
    rho0_np, E_np, rho_seq_np, _ctx = stack_rollouts(samples)
    return rho0_np, E_np, rho_seq_np


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
    gap: int
    N_train: int
    N_test: int
    seed: int
    final_frob: float
    obs_x_mae: float
    obs_z_mae: float


def _plot_vs_k_test(rows: list[Row], out_png: Path, *, metric: str, title: str) -> None:
    import matplotlib.pyplot as plt

    key = "final_frob" if metric == "frob" else ("obs_x_mae" if metric == "x" else "obs_z_mae")
    by_ktrain: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_ktrain[int(r.k_train)][int(r.k_test)].append(float(getattr(r, key)))

    from statistics import mean, stdev

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cmap = plt.get_cmap("tab10")
    ktr_sorted = sorted(by_ktrain.keys())
    for i, kt in enumerate(ktr_sorted):
        ks_sorted = sorted(by_ktrain[kt].keys())
        m = [mean(by_ktrain[kt][k]) for k in ks_sorted]
        s = [stdev(by_ktrain[kt][k]) if len(by_ktrain[kt][k]) > 1 else 0.0 for k in ks_sorted]
        ax.errorbar(ks_sorted, m, yerr=s, fmt="o-", capsize=3, color=cmap(i % 10), label=f"k_train={kt}")
    ax.set_xlabel("k_test")
    ax.set_ylabel({"frob": "Frobenius MSE", "x": "mean |d<X>|", "z": "mean |d<Z>|"}[metric])
    if metric == "frob":
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _plot_vs_gap(rows: list[Row], out_png: Path, *, metric: str, title: str) -> None:
    import matplotlib.pyplot as plt

    key = "final_frob" if metric == "frob" else ("obs_x_mae" if metric == "x" else "obs_z_mae")
    by_ktrain: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_ktrain[int(r.k_train)][int(r.gap)].append(float(getattr(r, key)))

    from statistics import mean, stdev

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cmap = plt.get_cmap("tab10")
    ktr_sorted = sorted(by_ktrain.keys())
    for i, kt in enumerate(ktr_sorted):
        gaps_sorted = sorted(by_ktrain[kt].keys())
        m = [mean(by_ktrain[kt][g]) for g in gaps_sorted]
        s = [stdev(by_ktrain[kt][g]) if len(by_ktrain[kt][g]) > 1 else 0.0 for g in gaps_sorted]
        ax.errorbar(gaps_sorted, m, yerr=s, fmt="o-", capsize=3, color=cmap(i % 10), label=f"k_train={kt}")
    ax.set_xlabel("gap = k_test - k_train")
    ax.set_ylabel({"frob": "Frobenius MSE", "x": "mean |d<X>|", "z": "mean |d<Z>|"}[metric])
    if metric == "frob":
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    J = 1.0
    g = 1.0
    dt = 0.1
    args = parse_args()
    k_trains = list(getattr(args, "_k_trains_resolved"))
    k_tests = list(getattr(args, "_k_tests_resolved"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"k_train grid ({len(k_trains)}): {k_trains}")
    print(f"k_test grid ({len(k_tests)}): {k_tests}")

    import torch
    from torch.utils.data import TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    op = MPO.ising(length=int(args.L), J=J, g=g)
    params = AnalogSimParams(dt=float(dt), solver="MCWF", show_progress=False, max_bond_dim=args.max_bond_dim)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    obs_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    obs_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    rows: list[Row] = []

    for k_train in k_trains:
        for seed in args.seeds:
            rng = np.random.default_rng(int(seed) + 12345 * int(k_train))
            rho0_tr, E_tr, rho_seq_tr = _make_continuous_dataset(
                k=int(k_train),
                n=int(args.N_train),
                rng=rng,
                L=int(args.L),
                op=op,
                params=params,
                static_ctx=static_ctx,
                parallel=bool(args.parallel_channel_sim),
            )
            rho_tgt = rho_seq_tr
            idx = np.arange(int(args.N_train), dtype=np.int64)
            rng.shuffle(idx)
            n_val = max(1, min(int(round(float(args.val_frac_of_train) * len(idx))), len(idx) - 1))
            va_idx, tr_idx = idx[:n_val], idx[n_val:]

            E_tr_t = torch.as_tensor(E_tr[tr_idx], dtype=torch.float32)
            r0_tr_t = torch.as_tensor(rho0_tr[tr_idx], dtype=torch.float32)
            tg_tr_t = torch.as_tensor(rho_tgt[tr_idx], dtype=torch.float32)
            E_va_t = torch.as_tensor(E_tr[va_idx], dtype=torch.float32)
            r0_va_t = torch.as_tensor(rho0_tr[va_idx], dtype=torch.float32)
            tg_va_t = torch.as_tensor(rho_tgt[va_idx], dtype=torch.float32)
            train_ds = TensorDataset(E_tr_t, r0_tr_t, tg_tr_t)
            val_ds = TensorDataset(E_va_t, r0_va_t, tg_va_t)

            tfm = TransformerComb(
                d_e=int(E_tr.shape[-1]),
                d_rho=8,
                d_model=int(args.d_model),
                nhead=int(args.nhead),
                num_layers=int(args.layers),
                dim_ff=int(args.dim_ff),
                dropout=float(args.dropout),
                layernorm_in=bool(args.layernorm_in),
            ).to(device)

            tfm.fit(
                train_ds,
                val_dataset=val_ds,
                epochs=int(args.epochs),
                lr=float(args.lr),
                batch_size=int(args.batch_size),
                grad_clip=float(args.grad_clip),
                prefix_loss=str(args.prefix_loss),
                device=device,
            )
            tfm.eval()

            for k_test in k_tests:
                rng_te = np.random.default_rng(int(seed) + 999_983 * int(k_test) + 77_777 * int(k_train))
                r0_te, E_te, rho_seq_te = _make_continuous_dataset(
                    k=int(k_test),
                    n=int(args.N_test),
                    rng=rng_te,
                    L=int(args.L),
                    op=op,
                    params=params,
                    static_ctx=static_ctx,
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
                        k_train=int(k_train),
                        k_test=int(k_test),
                        gap=int(k_test) - int(k_train),
                        N_train=int(args.N_train),
                        N_test=int(args.N_test),
                        seed=int(seed),
                        final_frob=float(fr),
                        obs_x_mae=float(ox),
                        obs_z_mae=float(oz),
                    )
                )
                print(
                    f"seed={seed} k_train={k_train} k_test={k_test} gap={int(k_test)-int(k_train)} "
                    f"frob={fr:.3e} |dX|={ox:.3e} |dZ|={oz:.3e}"
                )

    per_csv = out_dir / "benchmark_multi_k_generalization.csv"
    if rows:
        cols = list(asdict(rows[0]).keys())
        per_csv.write_text(
            "\n".join([",".join(cols), *[",".join(str(asdict(r)[c]) for c in cols) for r in rows]]) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {len(rows)} rows to {per_csv}")

    agg: dict[tuple[int, int], list[tuple[float, float, float]]] = defaultdict(list)
    for r in rows:
        agg[(r.k_train, r.k_test)].append((r.final_frob, r.obs_x_mae, r.obs_z_mae))
    agg_lines = ["k_train,k_test,gap,mean_frob,std_frob,mean_dX,std_dX,mean_dZ,std_dZ,n"]
    from statistics import mean, stdev

    for (kt, ks), vals in sorted(agg.items()):
        f = [v[0] for v in vals]
        x = [v[1] for v in vals]
        z = [v[2] for v in vals]
        agg_lines.append(
            f"{kt},{ks},{ks-kt},{mean(f)},{stdev(f) if len(f) > 1 else 0},"
            f"{mean(x)},{stdev(x) if len(x) > 1 else 0},{mean(z)},{stdev(z) if len(z) > 1 else 0},{len(vals)}"
        )
    (out_dir / "benchmark_multi_k_generalization_aggregate.csv").write_text("\n".join(agg_lines) + "\n", encoding="utf-8")

    if rows and not args.no_plots:
        _plot_vs_k_test(
            rows,
            out_dir / "benchmark_multi_k_generalization_frob_vs_ktest.png",
            metric="frob",
            title="Frobenius MSE vs k_test (continuous random)",
        )
        _plot_vs_k_test(
            rows,
            out_dir / "benchmark_multi_k_generalization_obs_x_vs_ktest.png",
            metric="x",
            title="|d<X>| vs k_test (continuous random)",
        )
        _plot_vs_k_test(
            rows,
            out_dir / "benchmark_multi_k_generalization_obs_z_vs_ktest.png",
            metric="z",
            title="|d<Z>| vs k_test (continuous random)",
        )
        _plot_vs_gap(
            rows,
            out_dir / "benchmark_multi_k_generalization_frob_vs_gap.png",
            metric="frob",
            title="Frobenius MSE vs gap (k_test - k_train; continuous random)",
        )
        _plot_vs_gap(
            rows,
            out_dir / "benchmark_multi_k_generalization_obs_x_vs_gap.png",
            metric="x",
            title="|d<X>| vs gap (k_test - k_train; continuous random)",
        )
        _plot_vs_gap(
            rows,
            out_dir / "benchmark_multi_k_generalization_obs_z_vs_gap.png",
            metric="z",
            title="|d<Z>| vs gap (k_test - k_train; continuous random)",
        )


if __name__ == "__main__":
    main()

