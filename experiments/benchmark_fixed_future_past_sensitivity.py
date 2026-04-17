#!/usr/bin/env python3
"""Minimal fixed-future past-sensitivity benchmark for TransformerComb.

Question:
Does changing only the past interventions change the final predicted state
when the future interventions are fixed?
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset

from mqt.yaqs.characterization.process_tensors import TransformerComb, generate_data
from mqt.yaqs.characterization.process_tensors.core.encoding import normalize_rho_from_backend_output, pack_rho8, unpack_rho8
from mqt.yaqs.characterization.process_tensors.surrogates.utils import _sample_random_intervention_parts
from mqt.yaqs.characterization.process_tensors.surrogates.workflow import _psi_from_rank1_projector, _simulate_sequences
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_entropy_results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--J", type=float, default=0.0)
    p.add_argument("--g", type=float, default=0.0)
    p.add_argument("--T", type=float, default=6.0)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--prefix-loss", type=str, default="full")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dim-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--cut", type=int, default=3, help="Past/future split: past=steps [0:cut], future=[cut:k].")
    p.add_argument("--n-past-samples", type=int, default=12)
    p.add_argument("--include-truth", action="store_true", default=True)
    p.add_argument("--no-truth", dest="include_truth", action="store_false")
    return p.parse_args()


def _rho8_to_phys(y: np.ndarray) -> np.ndarray:
    return normalize_rho_from_backend_output(unpack_rho8(y))


def _z_expectation(rho: np.ndarray) -> float:
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return float(np.trace(np.asarray(rho, dtype=np.complex128) @ z).real)


def _pairwise_frobenius(rhos: np.ndarray) -> np.ndarray:
    n = int(rhos.shape[0])
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d[i, j] = float(np.linalg.norm(rhos[i] - rhos[j], ord="fro"))
    return d


def _pairwise_absdiff(vals: np.ndarray) -> np.ndarray:
    v = np.asarray(vals, dtype=np.float64).reshape(-1)
    return np.abs(v[:, None] - v[None, :]).astype(np.float64)


def _sample_step(rng: np.random.Generator) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    rho_prep, effect, feat = _sample_random_intervention_parts(rng)
    psi_meas = _psi_from_rank1_projector(effect)
    psi_prep = _psi_from_rank1_projector(rho_prep)
    return feat.astype(np.float32), (psi_meas, psi_prep)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = _parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.k < 3:
        raise ValueError("k must be >= 3")
    cut = int(args.cut)
    if not (1 <= cut <= int(args.k) - 1):
        raise ValueError(f"cut must satisfy 1 <= cut <= k-1, got cut={cut}, k={args.k}")

    dt = float(args.T) / float(args.k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(int(args.seed))

    print("=== benchmark_fixed_future_past_sensitivity ===", flush=True)
    print(f"device={device}", flush=True)
    print(f"setup: L={args.L}, J={args.J}, g={args.g}, T={args.T}, k={args.k}, dt={dt:.6f}", flush=True)
    print(f"fixed-future test: cut={cut}, n_past_samples={args.n_past_samples}", flush=True)

    op = MPO.ising(length=int(args.L), J=float(args.J), g=float(args.g))
    sim_params = AnalogSimParams(dt=float(dt), solver="MCWF", show_progress=False)

    # 1) Train surrogate quickly on random rollout data.
    train_seed = int(args.seed) + 12345
    ds = generate_data(
        op,
        sim_params,
        k=int(args.k),
        n=int(args.n_train),
        seed=train_seed,
        parallel=True,
        show_progress=True,
    )
    E_all, r0_all, tgt_all = ds.tensors
    idx = np.arange(int(E_all.shape[0]), dtype=np.int64)
    np.random.default_rng(int(args.seed) + 54321).shuffle(idx)
    n_val = max(1, min(int(round(float(args.val_frac) * len(idx))), len(idx) - 1))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    train_ds = TensorDataset(E_all[tr_idx], r0_all[tr_idx], tgt_all[tr_idx])
    val_ds = TensorDataset(E_all[va_idx], r0_all[va_idx], tgt_all[va_idx])

    model = TransformerComb(
        d_e=int(E_all.shape[-1]),
        d_rho=8,
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        dim_ff=int(args.dim_ff),
        dropout=float(args.dropout),
    ).to(device)
    model.fit(
        train_ds,
        val_dataset=val_ds,
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        prefix_loss=str(args.prefix_loss),
        device=device,
    )
    model.eval()

    # 2) Held-out sanity error check (same setup).
    with torch.no_grad():
        pred_val = model(E_all[va_idx].to(device), r0_all[va_idx].to(device)).detach().cpu().numpy()
    true_val = tgt_all[va_idx].detach().cpu().numpy()
    pred_val_last = pred_val[:, -1, :]
    true_val_last = true_val[:, -1, :]

    pred_val_rhos = np.stack([_rho8_to_phys(y) for y in pred_val_last], axis=0)
    true_val_rhos = np.stack([_rho8_to_phys(y) for y in true_val_last], axis=0)
    val_frob = np.asarray(
        [float(np.linalg.norm(pred_val_rhos[i] - true_val_rhos[i], ord="fro")) for i in range(pred_val_rhos.shape[0])],
        dtype=np.float64,
    )
    pred_val_z = np.asarray([_z_expectation(r) for r in pred_val_rhos], dtype=np.float64)
    true_val_z = np.asarray([_z_expectation(r) for r in true_val_rhos], dtype=np.float64)
    val_z_abs = np.abs(pred_val_z - true_val_z)
    mean_val_frob = float(np.mean(val_frob))
    mean_val_z_abs = float(np.mean(val_z_abs))

    print(f"sanity held-out mean final-state Frobenius error: {mean_val_frob:.6e}", flush=True)
    print(f"sanity held-out mean absolute <Z> error:       {mean_val_z_abs:.6e}", flush=True)

    # 3) Build one fixed future and many different pasts.
    past_len = cut
    future_len = int(args.k) - cut

    fixed_future_features: list[np.ndarray] = []
    fixed_future_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(future_len):
        feat, pair = _sample_step(rng)
        fixed_future_features.append(feat)
        fixed_future_pairs.append(pair)
    fixed_future_features_arr = np.stack(fixed_future_features, axis=0).astype(np.float32)

    full_features: list[np.ndarray] = []
    past_features_only: list[np.ndarray] = []
    psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]] = []
    initial_psis: list[np.ndarray] = []

    # Build full-chain computational basis |00...0> consistent with system size L.
    psi0 = np.zeros(2 ** int(args.L), dtype=np.complex128)
    psi0[0] = 1.0 + 0.0j
    for _ in range(int(args.n_past_samples)):
        p_feats: list[np.ndarray] = []
        p_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for _step in range(past_len):
            feat, pair = _sample_step(rng)
            p_feats.append(feat)
            p_pairs.append(pair)

        full = np.concatenate([np.stack(p_feats, axis=0), fixed_future_features_arr], axis=0).astype(np.float32)
        full_features.append(full)
        past_features_only.append(np.stack(p_feats, axis=0).astype(np.float32))
        psi_pairs_list.append([*p_pairs, *fixed_future_pairs])
        initial_psis.append(psi0.copy())

    full_features_arr = np.stack(full_features, axis=0).astype(np.float32)  # (N, k, d_e)
    past_features_arr = np.stack(past_features_only, axis=0).astype(np.float32)  # (N, cut, d_e)

    # 4) Model predictions for fixed-future batch with varying past.
    rho0_default = pack_rho8(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)).astype(np.float32)
    E_batch = torch.from_numpy(full_features_arr).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pred_last = model.predict_final_state_batch(torch.from_numpy(rho0_default).to(device), E_batch).detach().cpu().numpy()
    pred_rhos = np.stack([_rho8_to_phys(y) for y in pred_last], axis=0)
    pred_z = np.asarray([_z_expectation(r) for r in pred_rhos], dtype=np.float64)
    pairwise_pred_frob = _pairwise_frobenius(pred_rhos)
    pairwise_pred_z = _pairwise_absdiff(pred_z)

    # 5) Optional exact backend truth for the exact same sequences.
    true_rhos: np.ndarray | None = None
    true_z: np.ndarray | None = None
    pairwise_true_frob: np.ndarray | None = None
    pairwise_true_z: np.ndarray | None = None
    pred_true_frob_err: np.ndarray | None = None
    pred_true_z_abs_err: np.ndarray | None = None

    if bool(args.include_truth):
        print("computing backend truth for exactly the same sequences...", flush=True)
        final_true_packed = _simulate_sequences(
            operator=op,
            sim_params=sim_params,
            timesteps=[float(dt)] * (int(args.k) + 1),
            psi_pairs_list=psi_pairs_list,
            initial_psis=initial_psis,
            static_ctx=None,
            parallel=True,
            show_progress=True,
            record_step_states=False,
        )
        assert isinstance(final_true_packed, np.ndarray)
        true_rhos = np.stack([_rho8_to_phys(y) for y in final_true_packed], axis=0)
        true_z = np.asarray([_z_expectation(r) for r in true_rhos], dtype=np.float64)
        pairwise_true_frob = _pairwise_frobenius(true_rhos)
        pairwise_true_z = _pairwise_absdiff(true_z)
        pred_true_frob_err = np.asarray(
            [float(np.linalg.norm(pred_rhos[i] - true_rhos[i], ord="fro")) for i in range(pred_rhos.shape[0])],
            dtype=np.float64,
        )
        pred_true_z_abs_err = np.abs(pred_z - true_z)

    # 6) Save raw outputs.
    np.save(out_dir / "past_sequences.npy", past_features_arr)
    np.save(out_dir / "full_sequences.npy", full_features_arr)
    np.save(out_dir / "fixed_future_sequence.npy", fixed_future_features_arr)
    np.save(out_dir / "pred_rhos.npy", pred_rhos)
    np.save(out_dir / "pred_z.npy", pred_z)
    np.save(out_dir / "pairwise_pred_rho_fro.npy", pairwise_pred_frob)
    np.save(out_dir / "pairwise_pred_z_absdiff.npy", pairwise_pred_z)
    if true_rhos is not None and true_z is not None:
        np.save(out_dir / "true_rhos.npy", true_rhos)
        np.save(out_dir / "true_z.npy", true_z)
    if pairwise_true_frob is not None and pairwise_true_z is not None:
        np.save(out_dir / "pairwise_true_rho_fro.npy", pairwise_true_frob)
        np.save(out_dir / "pairwise_true_z_absdiff.npy", pairwise_true_z)
    if pred_true_frob_err is not None and pred_true_z_abs_err is not None:
        np.save(out_dir / "pred_true_rho_fro_error.npy", pred_true_frob_err)
        np.save(out_dir / "pred_true_z_abs_error.npy", pred_true_z_abs_err)

    meta = {
        "L": int(args.L),
        "J": float(args.J),
        "g": float(args.g),
        "T": float(args.T),
        "k": int(args.k),
        "dt": float(dt),
        "cut": int(cut),
        "n_past_samples": int(args.n_past_samples),
        "mean_val_frob_error": float(mean_val_frob),
        "mean_val_abs_z_error": float(mean_val_z_abs),
        "include_truth": bool(args.include_truth),
    }
    (out_dir / "benchmark_fixed_future_past_sensitivity_meta.json").write_text(json.dumps(meta, indent=2))

    # 7) Print full requested report.
    np.set_printoptions(precision=5, suppress=True, linewidth=120)
    print("\n=== Fixed Future / Vary Past Report ===", flush=True)
    print("\nPast sequences used (feature rows):", flush=True)
    print(past_features_arr, flush=True)
    print("\nFinal predicted density matrices:", flush=True)
    print(pred_rhos, flush=True)
    print("\nFinal predicted <Z> values:", flush=True)
    print(pred_z, flush=True)
    print("\nPairwise Frobenius distances between predicted final density matrices:", flush=True)
    print(pairwise_pred_frob, flush=True)
    print("\nPairwise absolute differences between predicted final <Z> values:", flush=True)
    print(pairwise_pred_z, flush=True)

    if true_rhos is not None and true_z is not None:
        print("\nFinal true density matrices:", flush=True)
        print(true_rhos, flush=True)
        print("\nFinal true <Z> values:", flush=True)
        print(true_z, flush=True)
        print("\nPairwise true Frobenius distances:", flush=True)
        print(pairwise_true_frob, flush=True)
        print("\nPairwise true <Z> absolute differences:", flush=True)
        print(pairwise_true_z, flush=True)
        print("\nPer-sequence prediction error (Frobenius rho / abs <Z>):", flush=True)
        print(np.stack([pred_true_frob_err, pred_true_z_abs_err], axis=1), flush=True)

    # Final compact comparison: model error scale vs past-induced variation scale.
    pred_var_rho_med = float(np.median(pairwise_pred_frob[np.triu_indices(pairwise_pred_frob.shape[0], k=1)]))
    pred_var_z_med = float(np.median(pairwise_pred_z[np.triu_indices(pairwise_pred_z.shape[0], k=1)]))
    print("\n=== Compact Summary ===", flush=True)
    print(f"Typical model error scale (held-out): rho_frob_mean={mean_val_frob:.6e}, |Z|_mean={mean_val_z_abs:.6e}", flush=True)
    print(
        f"Typical past-induced variation scale (pred): rho_frob_pairwise_median={pred_var_rho_med:.6e}, "
        f"|dZ|_pairwise_median={pred_var_z_med:.6e}",
        flush=True,
    )
    if pred_true_frob_err is not None and pred_true_z_abs_err is not None:
        print(
            f"Prediction-vs-truth on fixed-future set: rho_frob_mean={float(np.mean(pred_true_frob_err)):.6e}, "
            f"|Z|_mean={float(np.mean(pred_true_z_abs_err)):.6e}",
            flush=True,
        )
    print(f"\nSaved raw outputs under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
