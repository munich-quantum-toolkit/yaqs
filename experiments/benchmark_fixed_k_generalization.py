# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Fixed-k training and generalization to k' ≠ k (Transformer rho0, single Ising regime).

**Data (same as the continuous branch of ``benchmark_basis_training_convergence``):** every
intervention **sequence** is built with ``sample_random_intervention_sequence``: rank-1 prep/effect CP maps, backend
via ``simulate_sequences`` with ``e_features_rows`` = flattened Choi rows (shape
``k × 32``). There is no discrete 16-letter alphabet in the features seen by the network.

**Training:** ``N_train`` trajectories of length ``k_train``; optional train/val split from that
pool; MSE on packed ``rho_seq`` targets vs.
:class:`~mqt.yaqs.characterization.process_tensors.surrogates.model.TransformerComb` with initial reduced
state broadcast to every step.

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
import numpy as np

from mqt.yaqs.characterization.process_tensors.core.encoding import unpack_rho8
from mqt.yaqs.characterization.process_tensors.core.metrics import _mean_frobenius_mse_rho8 as mean_frobenius_mse_rho8
from mqt.yaqs.characterization.process_tensors import TransformerComb, generate_data
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
    from torch.utils.data import TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    op = MPO.ising(length=int(args.L), J=J, g=g)
    params = AnalogSimParams(dt=float(dt), solver="MCWF", show_progress=False, max_bond_dim=args.max_bond_dim)
    obs_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    obs_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    rows: list[Row] = []
    for seed in args.seeds:
        ds = generate_data(
            op,
            params,
            k=int(args.k_train),
            n=int(args.N_train),
            seed=int(seed) + 12345 * int(args.k_train),
            parallel=bool(args.parallel_channel_sim),
            show_progress=False,
        )
        E_all, r0_all, tgt_all = ds.tensors
        idx = np.arange(int(E_all.shape[0]), dtype=np.int64)
        np.random.default_rng(int(seed) + 54321).shuffle(idx)
        n_val = max(1, min(int(round(float(args.val_frac_of_train) * len(idx))), len(idx) - 1))
        va_idx, tr_idx = idx[:n_val], idx[n_val:]
        train_ds = TensorDataset(E_all[tr_idx], r0_all[tr_idx], tgt_all[tr_idx])
        val_ds = TensorDataset(E_all[va_idx], r0_all[va_idx], tgt_all[va_idx])
        tfm = TransformerComb(
            d_e=int(E_all.shape[-1]),
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
        for k_test in k_tests_plan:
            ds_te = generate_data(
                op,
                params,
                k=int(k_test),
                n=int(args.N_test),
                seed=int(seed) + 999_983 * int(k_test) + 77_777 * int(args.k_train),
                parallel=bool(args.parallel_channel_sim),
                show_progress=False,
            )
            E_te, r0_te, tgt_te = ds_te.tensors
            with torch.no_grad():
                pred_seq = tfm(E_te.to(device), r0_te.to(device)).cpu().numpy().astype(np.float32)
            pred_final = pred_seq[:, -1, :]
            tgt_final = tgt_te.numpy()[:, -1, :].astype(np.float32)
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
        per_csv.write_text(
            "\n".join([",".join(cols), *[",".join(str(asdict(r)[c]) for c in cols) for r in rows]]) + "\n",
            encoding="utf-8",
        )
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
    (out_dir / "benchmark_fixed_k_generalization_aggregate.csv").write_text(
        "\n".join(agg_lines) + "\n", encoding="utf-8"
    )

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
