# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""
Backend-only NNComb training-history (Frobenius MSE objective).

This script samples training labels from the MCWF backend, trains `NNComb`
on `(x, y_backend)` and plots the epoch-by-epoch training loss history.

No exhaustive `DenseComb` reference is constructed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from typing import cast

from mqt.yaqs.characterization.tomography.combs import NNComb
from mqt.yaqs.characterization.tomography.basis import TomographyBasis
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

from experiments.nn_channel_predictor_utils import (
    build_basis_for_fixed_alphabet,
    build_initial_psi,
    concat_choi_features,
    make_static_ctx,
    mean_frobenius_mse_rho8,
    pack_rho8,
    random_density_matrix,
    simulate_backend_labels,
    CHOI_FLAT_DIM,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NNComb backend-only training history (Frobenius MSE).")
    p.add_argument("--ks", type=int, nargs="+", default=[4])
    p.add_argument("--n_samples", type=int, default=2048)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--train_fracs", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1])
    p.add_argument("--train_sizes", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--g", type=float, default=0.75)
    p.add_argument("--timesteps", type=float, nargs="+", default=[0.1]*100)
    p.add_argument("--basis", type=str, default="tetrahedral", choices=["standard", "tetrahedral", "random"])
    p.add_argument("--basis_seed", type=int, default=12345)
    p.add_argument("--backend_init_mode", type=str, default="eigenstate", choices=["eigenstate", "purified"])
    p.add_argument("--max_bond_dim", type=int, default=16)
    p.add_argument("--parallel_channel_sim", action="store_true")
    p.add_argument("--out_dir", type=str, default="benchmark_results")
    p.add_argument(
        "--no_plot_train_loss",
        action="store_true",
        help="Disable the training-loss curve plot for the first seed/smallest n_train.",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    op = MPO.ising(length=args.L, J=args.J, g=args.g)

    basis_t = cast(TomographyBasis, args.basis)

    for k in sorted(set(args.ks)):
        if len(args.timesteps) < k:
            raise ValueError(f"Need len(timesteps) >= k={k}, got {len(args.timesteps)}")
        timesteps = args.timesteps[:k]
        solver_dt = float(min(timesteps))

        params = AnalogSimParams(
            dt=solver_dt,
            solver="MCWF",
            show_progress=False,
            max_bond_dim=args.max_bond_dim,
        )

        static_ctx = make_static_ctx(op, params)

        (
            basis_set,
            _choi_matrices,
            choi_pm_pairs,
            choi_feat_table,
        ) = build_basis_for_fixed_alphabet(basis=args.basis, basis_seed=args.basis_seed)

        in_dim = 8 + CHOI_FLAT_DIM * k

        n_test = max(1, int(round(args.test_frac * args.n_samples)))
        n_tr_pool = args.n_samples - n_test
        if n_tr_pool < 1:
            raise ValueError("test_frac too large: no training pool left.")

        if args.train_sizes is not None:
            sizes = sorted({int(s) for s in args.train_sizes if 1 <= int(s) <= n_tr_pool})
        else:
            sizes = sorted({max(1, min(n_tr_pool, int(round(f * n_tr_pool)))) for f in args.train_fracs})
        if not sizes:
            raise ValueError("No valid train sizes for this configuration.")

        seeds_list = list(args.seeds)
        first_seed = int(seeds_list[0]) if seeds_list else 0
        first_n_tr = int(sizes[0]) if sizes else 0

        for seed in seeds_list:
            rng = np.random.default_rng(seed + 97_531 * k)

            rho_ins: list[np.ndarray] = []
            alphas_rows: list[np.ndarray] = []
            psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]] = []
            initial_psis: list[np.ndarray] = []

            for _ in range(args.n_samples):
                rho_in = random_density_matrix(rng)
                alphas = rng.integers(0, 16, size=k, dtype=np.int64)

                pairs = [
                    (
                        basis_set[choi_pm_pairs[int(a)][1]][1],
                        basis_set[choi_pm_pairs[int(a)][0]][1],
                    )
                    for a in alphas
                ]
                psi0 = build_initial_psi(
                    rho_in,
                    length=op.length,
                    rng=rng,
                    init_mode=args.backend_init_mode,
                )

                rho_ins.append(rho_in)
                alphas_rows.append(alphas)
                psi_pairs_list.append(pairs)
                initial_psis.append(psi0)

            y_backend = simulate_backend_labels(
                op=op,
                params=params,
                timesteps=timesteps,
                psi_pairs_list=psi_pairs_list,
                initial_psis=initial_psis,
                parallel=args.parallel_channel_sim,
                static_ctx=static_ctx,
            )

            rho8_in = np.stack([pack_rho8(r) for r in rho_ins], axis=0).astype(np.float32)
            alphas_arr = np.stack(alphas_rows, axis=0)
            x = np.stack(
                [
                    np.concatenate(
                        [rho8_in[i], concat_choi_features(alphas_arr[i], choi_feat_table)],
                        axis=0,
                    )
                    for i in range(args.n_samples)
                ],
                axis=0,
            ).astype(np.float32)

            perm = rng.permutation(args.n_samples)
            te_idx = perm[:n_test]
            tr_perm = perm[n_test:]

            sizes = [int(sizes[0])] if sizes else []

            device = torch.device("cpu")
            for n_tr in sizes:
                tr_idx = tr_perm[:n_tr]
                nn = NNComb(in_dim=in_dim, hidden=args.hidden, out_dim=8)
                show_loss = bool((not args.no_plot_train_loss) and seed == first_seed and n_tr == first_n_tr)
                hist = nn.fit_features(
                    x[tr_idx],
                    y_backend[tr_idx],
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    device=device,
                    w_train=np.ones(n_tr, dtype=np.float32),
                    return_history=show_loss,
                    verbose=False,
                )
                if show_loss and hist is not None:
                    # Plot the actual training objective used inside NNComb.fit_features.
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    xs_epoch = list(range(1, len(hist) + 1))
                    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
                    ax.loglog(xs_epoch, hist, "o-", lw=1.5, ms=3)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Train loss (weighted MSE objective)")
                    ax.set_title(f"NNComb training loss (k={k}, seed={seed}, n_train={n_tr})")
                    ax.grid(True, alpha=0.3)
                    fig_path = out_dir / f"nn_train_loss_mse_k{k}_seed{seed}_ntr{n_tr}.png"
                    fig.savefig(fig_path, dpi=150)
                    plt.close(fig)
                    print(
                        f"[k={k} seed={seed} n_train={n_tr}] train-loss: start={hist[0]:.3e} end={hist[-1]:.3e} (saved {fig_path})"
                    )
                pred = nn.predict_features(x[te_idx], device=device)
                mse_nn = mean_frobenius_mse_rho8(pred, y_backend[te_idx])
                print(f"[k={k} seed={seed}] test MSE vs backend labels: {mse_nn:.6e}")


if __name__ == "__main__":
    main()

