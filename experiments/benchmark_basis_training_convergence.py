# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Training convergence at fixed sequence length k for different intervention/basis regimes.

Trains TransformerComb (rho0 mode) on Ising trajectories of fixed ``k`` for:
  - discrete random / standard / tetrahedral bases (16-symbol alphabet from that basis)
  - continuous rank-1 prep/effect interventions

Logs train MSE on the regime-specific data and validation MSE on a **shared continuous
random-intervention** holdout (same validation trajectories for every regime, for comparable curves).
CSV + plot.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from mqt.yaqs.characterization.process_tensors.core.utils import make_mcwf_static_context
from mqt.yaqs.characterization.process_tensors.surrogates.data import stack_rollouts
from mqt.yaqs.characterization.process_tensors.surrogates.model import TransformerComb
from mqt.yaqs.characterization.process_tensors.surrogates.utils import (
    build_initial_psi,
    _random_density_matrix as random_density_matrix,
    _sample_random_intervention_sequence as sample_random_intervention_sequence,
)
from mqt.yaqs.characterization.process_tensors.surrogates.workflow import _simulate_sequences as simulate_sequences
from mqt.yaqs.characterization.process_tensors.tomography.basis import TomographyBasis, build_basis_for_fixed_alphabet
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training convergence vs basis/intervention regime at fixed k.")
    p.add_argument("--out_dir", type=str, default="benchmark_results")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--k", type=int, required=True, help="Fixed intervention sequence length for training.")
    p.add_argument("--N_train", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument(
        "--val_frac_of_train",
        type=float,
        default=0.2,
        help="Sets validation set size as round(fraction * N_train) continuous random trajectories "
        "(not drawn from the training batch).",
    )
    p.add_argument("--basis_seed", type=int, default=0)
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
    p.add_argument("--prefix_loss", type=str, default="all", choices=["full", "random", "all"])
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def _make_backend_dataset(
    *,
    k: int,
    n: int,
    rng: np.random.Generator,
    L: int,
    op: MPO,
    params: AnalogSimParams,
    static_ctx: Any,
    intervention_mode: str,
    basis_set: Any,
    choi_pm_pairs: Any,
    choi_feat_table: np.ndarray,
    parallel: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]] = []
    initial_psis: list[np.ndarray] = []
    e_features_rows: list[np.ndarray] = []

    def _state_from_rank1_projector(P: np.ndarray) -> np.ndarray:
        w, v = np.linalg.eigh(np.asarray(P, dtype=np.complex128).reshape(2, 2))
        idx = int(np.argmax(w.real))
        psi = v[:, idx]
        nrm = float(np.linalg.norm(psi))
        if nrm < 1e-15:
            return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        return (psi / nrm).astype(np.complex128)

    for _ in range(int(n)):
        rho_in = random_density_matrix(rng)
        if intervention_mode == "discrete":
            alphas = rng.integers(0, 16, size=int(k), dtype=np.int64)
            pm = [choi_pm_pairs[int(a)] for a in alphas]
            pairs = [(basis_set[m][1], basis_set[p][1]) for (p, m) in pm]
            e_features_rows.append(np.asarray(choi_feat_table[alphas], dtype=np.float32))
            psi_pairs_list.append(pairs)
        else:
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


def _train_with_history(
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
) -> tuple[Any, list[tuple[int, float, float]]]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    # Tensor order (E, rho0, target) matches :meth:`TransformerComb.fit` / :func:`generate_data`.
    loader = DataLoader(
        TensorDataset(E_tr, rho0_tr, tgt_tr),
        batch_size=min(int(batch_size), max(1, int(E_tr.shape[0]))),
        shuffle=True,
    )
    k_max = int(tgt_tr.shape[1])
    history: list[tuple[int, float, float]] = []
    best = float("inf")
    best_state = None

    for ep in range(int(epochs)):
        model.train()
        sum_loss = 0.0
        n_batches = 0
        for E_b, rho0_b, tgt_b in loader:
            opt.zero_grad(set_to_none=True)
            if prefix_loss == "full" or k_max <= 1:
                pred = model(E_b, rho0_b)
                loss = loss_fn(pred, tgt_b)
            elif prefix_loss == "random":
                L = int(torch.randint(low=1, high=k_max + 1, size=(1,), device=E_b.device).item())
                pred = model(E_b[:, :L, :], rho0_b)
                loss = loss_fn(pred, tgt_b[:, :L, :])
            elif prefix_loss == "all":
                losses = []
                for L in range(1, k_max + 1):
                    pred_L = model(E_b[:, :L, :], rho0_b)
                    losses.append(loss_fn(pred_L, tgt_b[:, :L, :]))
                loss = torch.stack(losses, dim=0).mean()
            else:
                raise ValueError(f"Unknown prefix_loss={prefix_loss!r}")
            loss.backward()
            if grad_clip and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            sum_loss += float(loss.detach().cpu().item())
            n_batches += 1
        train_mse = sum_loss / max(1, n_batches)
        model.eval()
        with torch.no_grad():
            val_mse = float(loss_fn(model(E_va, rho0_va), tgt_va).detach().cpu().item())
        history.append((ep + 1, train_mse, val_mse))
        if val_mse < best:
            best = val_mse
            best_state = {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@dataclass(frozen=True)
class HistoryRow:
    regime: str
    seed: int
    epoch: int
    train_mse: float
    val_mse: float


def main() -> None:
    J = 1.0
    g = 1.0
    dt = 0.1
    args = parse_args()
    L = int(args.L)
    k_fixed = int(args.k)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("Requires PyTorch.") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    op = MPO.ising(length=L, J=J, g=g)
    params = AnalogSimParams(dt=float(dt), solver="MCWF", show_progress=False, max_bond_dim=args.max_bond_dim)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)

    run_configs: list[tuple[str, str, str]] = [
        ("random_discrete", "discrete", "random"),
        ("standard_discrete", "discrete", "standard"),
        ("tetrahedral_discrete", "discrete", "tetrahedral"),
        ("continuous", "continuous", "random"),
    ]

    basis_v, _cv, choi_pm_v, choi_feat_v = build_basis_for_fixed_alphabet(
        basis="random", basis_seed=int(args.basis_seed)
    )

    all_rows: list[HistoryRow] = []

    for seed in args.seeds:
        n_val = int(round(float(args.val_frac_of_train) * float(args.N_train)))
        n_val = max(1, n_val)
        rng_val = np.random.default_rng(int(seed) + 99_003 * k_fixed + 7)
        rho0_va_np, E_va_np, rho_seq_va_np = _make_backend_dataset(
            k=k_fixed,
            n=n_val,
            rng=rng_val,
            L=L,
            op=op,
            params=params,
            static_ctx=static_ctx,
            intervention_mode="continuous",
            basis_set=basis_v,
            choi_pm_pairs=choi_pm_v,
            choi_feat_table=choi_feat_v,
            parallel=bool(args.parallel_channel_sim),
        )
        rho_tgt_va_np = rho_seq_va_np
        E_va = torch.as_tensor(E_va_np, dtype=torch.float32, device=device)
        rho0_va = torch.as_tensor(rho0_va_np, dtype=torch.float32, device=device)
        tgt_va = torch.as_tensor(rho_tgt_va_np, dtype=torch.float32, device=device)

        for regime_label, imode, basis_build in run_configs:
            basis_t = cast(TomographyBasis, basis_build)
            basis_set, _c, choi_pm_pairs, choi_feat_table = build_basis_for_fixed_alphabet(
                basis=basis_t, basis_seed=int(args.basis_seed)
            )
            print(f"\n=== regime={regime_label} seed={seed} ===")
            rng = np.random.default_rng(int(seed) + 42_001 * k_fixed + (19 if imode == "continuous" else 0))
            rho0_tr_np, E_tr_np, rho_seq_tr_np = _make_backend_dataset(
                k=k_fixed,
                n=int(args.N_train),
                rng=rng,
                L=L,
                op=op,
                params=params,
                static_ctx=static_ctx,
                intervention_mode=str(imode),
                basis_set=basis_set,
                choi_pm_pairs=choi_pm_pairs,
                choi_feat_table=choi_feat_table,
                parallel=bool(args.parallel_channel_sim),
            )
            rho_tgt_np = rho_seq_tr_np
            if int(E_tr_np.shape[-1]) != int(E_va_np.shape[-1]):
                msg = f"Train E dim {E_tr_np.shape[-1]} != val E dim {E_va_np.shape[-1]}."
                raise ValueError(msg)

            E_tr = torch.as_tensor(E_tr_np, dtype=torch.float32, device=device)
            rho0_tr = torch.as_tensor(rho0_tr_np, dtype=torch.float32, device=device)
            tgt_tr = torch.as_tensor(rho_tgt_np, dtype=torch.float32, device=device)

            tfm = TransformerComb(
                d_e=int(E_tr_np.shape[-1]),
                d_rho=8,
                d_model=int(args.d_model),
                nhead=int(args.nhead),
                num_layers=int(args.layers),
                dim_ff=int(args.dim_ff),
                dropout=float(args.dropout),
                layernorm_in=bool(args.layernorm_in),
            ).to(device)

            _tfm, history = _train_with_history(
                model=tfm,
                E_tr=E_tr,
                rho0_tr=rho0_tr,
                tgt_tr=tgt_tr,
                E_va=E_va,
                rho0_va=rho0_va,
                tgt_va=tgt_va,
                epochs=int(args.epochs),
                lr=float(args.lr),
                batch_size=int(args.batch_size),
                grad_clip=float(args.grad_clip),
                prefix_loss=str(args.prefix_loss),
            )
            for ep, tr_m, va_m in history:
                all_rows.append(
                    HistoryRow(
                        regime=str(regime_label),
                        seed=int(seed),
                        epoch=int(ep),
                        train_mse=float(tr_m),
                        val_mse=float(va_m),
                    )
                )

    csv_path = out_dir / "benchmark_basis_training_convergence.csv"
    if all_rows:
        cols = list(asdict(all_rows[0]).keys())
        lines = [",".join(cols), *[",".join(str(asdict(r)[c]) for c in cols) for r in all_rows]]
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote {len(all_rows)} history rows to {csv_path}")

    if all_rows and not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError("matplotlib required for plots.") from e

        by_regime: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        for r in all_rows:
            by_regime[r.regime].append((r.epoch, r.train_mse, r.val_mse))

        from statistics import mean

        fig, ax = plt.subplots(figsize=(8, 5))
        labels_sorted = sorted(by_regime.keys())
        cmap = plt.get_cmap("tab10")
        for i, reg in enumerate(labels_sorted):
            c = cmap(i % 10)
            ep_to_val: dict[int, list[float]] = defaultdict(list)
            for e, _tr_m, va_m in by_regime[reg]:
                ep_to_val[int(e)].append(float(va_m))
            ep_sorted = sorted(ep_to_val.keys())
            v_mean = [mean(ep_to_val[e]) for e in ep_sorted]
            ax.plot(ep_sorted, v_mean, "-", label=f"{reg} val", color=c, linewidth=2)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation MSE (continuous random)")
        ax.set_yscale("log")
        ax.set_title(f"Training convergence (fixed k={k_fixed}, L={L}; val = continuous random)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png = out_dir / "benchmark_basis_training_convergence.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()
