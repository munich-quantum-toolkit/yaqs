# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""
Channel predictor NN vs exhaustive comb reference.

For each k, samples a dataset:
  x = [vec(rho_in), flat_Choi(E1), ..., flat_Choi(Ek)]  (8 + 32*k reals)
  y_backend = vec(rho_out) from backend simulation
  y_ex = exhaustive DenseComb prediction for the same inputs (with rho_in injected)

Then sweeps train size and trains `NNComb` on (x, y_backend), reporting:
  mean test trace distance (Hermitianized) between NN predictions and y_ex
  mean test trace distance between backend labels and y_ex (oracle mismatch floor)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mqt.yaqs.characterization.tomography.combs import NNComb
from mqt.yaqs.characterization.tomography.process_tomography import run as tomography_run
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from typing import cast
from mqt.yaqs.characterization.tomography.basis import TomographyBasis

from experiments.nn_channel_predictor_utils import (
    build_basis_for_fixed_alphabet,
    build_initial_psi,
    concat_choi_features,
    intervention_from_alpha,
    make_static_ctx,
    mean_trace_distance_rho8,
    normalize_rho_like_densecomb,
    pack_rho8,
    random_density_matrix,
    simulate_backend_labels,
    state_prep_map_from_rho,
    CHOI_FLAT_DIM,
    unpack_rho8,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NNComb vs exhaustive comb (trace distance).")
    p.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
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
    p.add_argument("--timesteps", type=float, nargs="+", default=[0.1, 0.1, 0.1, 0.1])
    p.add_argument("--basis", type=str, default="tetrahedral", choices=["standard", "tetrahedral", "random"])
    p.add_argument("--basis_seed", type=int, default=12345)
    p.add_argument("--backend_init_mode", type=str, default="eigenstate", choices=["eigenstate", "purified"])
    p.add_argument("--max_bond_dim", type=int, default=16)
    p.add_argument("--parallel_channel_sim", action="store_true")
    p.add_argument("--out_dir", type=str, default="benchmark_results")
    p.add_argument("--oracle_floor_td_threshold", type=float, default=1e-2)
    return p.parse_args()


def _build_first_map(alpha0: Any, prep_map: Any) -> Any:
    def first_map(sigma: np.ndarray, alpha0: Any = alpha0, prep_map: Any = prep_map) -> np.ndarray:
        return alpha0(prep_map(sigma))

    return first_map


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    op = MPO.ising(length=args.L, J=args.J, g=args.g)

    # Fixed basis alphabet for all samples within each run.
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
            choi_matrices,
            choi_pm_pairs,
            choi_feat_table,
        ) = build_basis_for_fixed_alphabet(basis=args.basis, basis_seed=args.basis_seed)

        # Build exhaustive comb reference once.
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
            basis_seed=args.basis_seed,
        )
        if not hasattr(comb_ex_no_prep, "predict"):
            raise RuntimeError("Expected DenseComb output from exhaustive run.")

        in_dim = 8 + CHOI_FLAT_DIM * k

        td_nn_by_size: dict[int, list[float]] = defaultdict(list)
        td_floor_by_size: dict[int, list[float]] = defaultdict(list)

        # Determine train sizes.
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

        for seed in args.seeds:
            rng = np.random.default_rng(seed + 97_531 * k)

            rho_ins: list[np.ndarray] = []
            alphas_rows: list[np.ndarray] = []
            psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]] = []
            initial_psis: list[np.ndarray] = []
            intervention_fns: list[list[Any]] = []

            # Sample dataset
            for _ in range(args.n_samples):
                rho_in = random_density_matrix(rng)
                alphas = rng.integers(0, 16, size=k, dtype=np.int64)

                # Backend psi-pairs
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

                # Intervention CP maps for exhaustive reference
                fns = [intervention_from_alpha(int(a), basis_set, choi_pm_pairs) for a in alphas]

                rho_ins.append(rho_in)
                alphas_rows.append(alphas)
                psi_pairs_list.append(pairs)
                initial_psis.append(psi0)
                intervention_fns.append(fns)

            # Backend labels
            y_backend = simulate_backend_labels(
                op=op,
                params=params,
                timesteps=timesteps,
                psi_pairs_list=psi_pairs_list,
                initial_psis=initial_psis,
                parallel=args.parallel_channel_sim,
                static_ctx=static_ctx,
            )

            # Features
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

            # Exhaustive reference with rho_in injected into first intervention slot.
            y_ex_rows: list[np.ndarray] = []
            y_ex_old_rows: list[np.ndarray] = []
            for i in range(args.n_samples):
                fns = intervention_fns[i]
                rho_in = rho_ins[i]

                rho_ex_old = comb_ex_no_prep.predict(fns)
                rho_ex_old = normalize_rho_like_densecomb(np.asarray(rho_ex_old, dtype=np.complex128).reshape(2, 2))
                y_ex_old_rows.append(pack_rho8(rho_ex_old))

                prep_map = state_prep_map_from_rho(rho_in)
                first_map = _build_first_map(alpha0=fns[0], prep_map=prep_map)
                rho_ex_new = comb_ex_no_prep.predict([first_map] + fns[1:])
                rho_ex_new = normalize_rho_like_densecomb(np.asarray(rho_ex_new, dtype=np.complex128).reshape(2, 2))
                y_ex_rows.append(pack_rho8(rho_ex_new))

            y_ex = np.stack(y_ex_rows, axis=0).astype(np.float32)
            y_ex_old = np.stack(y_ex_old_rows, axis=0).astype(np.float32)

            # Sanity check oracle mismatch floor.
            sanity_n = min(16, args.n_samples)
            sanity_idx = rng.choice(args.n_samples, size=sanity_n, replace=False)
            td_floor_old = mean_trace_distance_rho8(y_backend[sanity_idx], y_ex_old[sanity_idx])
            td_floor_new = mean_trace_distance_rho8(y_backend[sanity_idx], y_ex[sanity_idx])
            print(
                f"[k={k} seed={seed}] oracle floor TD: old(no prep)={td_floor_old:.3e}  new(with rho_in prep)={td_floor_new:.3e}"
            )
            if td_floor_new > args.oracle_floor_td_threshold:
                raise RuntimeError(
                    "oracle mismatch floor too large; backend and exhaustive reference likely disagree."
                )

            perm = rng.permutation(args.n_samples)
            te_idx = perm[:n_test]
            tr_perm = perm[n_test:]

            floor_td = mean_trace_distance_rho8(y_backend[te_idx], y_ex[te_idx])
            for n_tr in sizes:
                td_floor_by_size[n_tr].append(floor_td)

            device = torch.device("cpu")
            for n_tr in sizes:
                tr_idx = tr_perm[:n_tr]
                nn = NNComb(in_dim=in_dim, hidden=args.hidden, out_dim=8)
                nn.fit_features(
                    x[tr_idx],
                    y_backend[tr_idx],
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    device=device,
                    w_train=np.ones(n_tr, dtype=np.float32),
                )
                pred = nn.predict_features(x[te_idx], device=device)
                td = mean_trace_distance_rho8(pred, y_ex[te_idx])
                td_nn_by_size[n_tr].append(td)

        # Plot + save
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = sorted(sizes)
        mean_nn, std_nn = [], []
        mean_floor, std_floor = [], []
        for n_tr in xs:
            a = np.asarray(td_nn_by_size[n_tr], dtype=np.float64)
            a = a[np.isfinite(a)]
            mean_nn.append(float(a.mean()))
            std_nn.append(float(a.std(ddof=0)) if a.size else float("nan"))
            b = np.asarray(td_floor_by_size[n_tr], dtype=np.float64)
            b = b[np.isfinite(b)]
            mean_floor.append(float(b.mean()))
            std_floor.append(float(b.std(ddof=0)) if b.size else float("nan"))

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.errorbar(xs, mean_nn, yerr=std_nn, fmt="o-", capsize=4, lw=2, label="NN vs exhaustive")
        ax.errorbar(xs, mean_floor, yerr=std_floor, fmt="s--", capsize=4, lw=1.5, label="backend vs exhaustive floor")
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Mean trace distance (test)")
        ax.set_title(f"NNComb vs exhaustive (trace TD), k={k}, basis={args.basis}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig_path = out_dir / f"nn_predictor_vs_exhaustive_td_k{k}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[k={k}] Saved figure: {fig_path}")


if __name__ == "__main__":
    main()

