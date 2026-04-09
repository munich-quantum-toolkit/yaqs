# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Smoke benchmark: train a small :class:`~mqt.yaqs.characterization.process_tensors.TransformerComb`,
then sweep interior cuts and :meth:`~mqt.yaqs.characterization.process_tensors.surrogates.model.TransformerComb.entropy`.

Uses the same Ising + :func:`~mqt.yaqs.characterization.process_tensors.generate_data` path as
``benchmark_fixed_k_generalization.py``. Outputs CSV + entropy-vs-cut plot under ``OUT_DIR``.

Run from repo root::

    uv run python experiments/benchmark_entropy.py
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config (edit here)
# ---------------------------------------------------------------------------
K_TRAIN = 10
N_TRAIN = 512
SEED = 0
OUT_DIR = Path("benchmark_entropy_results")

PAST_SAMPLES = 64
FUTURE_SAMPLES = 16
N_REPEATS = 5

L = 1
J = 0.0
G = 0.0
DT = 0.1

EPOCHS = 80
LR = 2e-3
BATCH_SIZE = 64
VAL_FRAC = 0.2
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 256
DROPOUT = 0.0
PREFIX_LOSS = "full"


def _entropy_from_v(v: "np.ndarray") -> float:
    """Column-center V, compute svdvals, and return Shannon entropy of s^2 weights (nats)."""
    vv = np.asarray(v, dtype=np.float64)
    vv = vv - vv.mean(axis=0, keepdims=True)
    s = np.linalg.svd(vv, compute_uv=False)
    sq = s * s
    denom = float(np.sum(sq))
    if denom <= 0.0:
        return 0.0
    w = sq / denom
    w = np.clip(w, 1e-30, 1.0)
    return float(-np.sum(w * np.log(w)))


def _build_v_exact_trivial(
    *,
    k_total: int,
    cut_t: int,
    n_past: int,
    n_future: int,
    seed: int,
) -> np.ndarray:
    """Exact inside-timestep cut matrix V for the trivial system L=1,J=g=0.

    For L=1, J=g=0 there is no evolution; after the final intervention the normalized reduced state
    equals the **preparation state of the last timestep** (independent of history/effects).

    We still construct V with the same indexing as TransformerComb.entropy():
      - rows: past steps 0..t-1 (fused) + measurement effect at step t
      - cols: preparation at step t + future suffix steps t+1..k-1 (fused)
    and fill outputs using the exact normalized final rho8 from the intervention chain.
    """
    from mqt.yaqs.characterization.process_tensors.core.encoding import pack_rho8
    from mqt.yaqs.characterization.process_tensors.surrogates.utils import (  # noqa: PLC0415
        _sample_random_intervention_parts,
        _sample_random_intervention_sequence,
    )

    rng = np.random.default_rng(int(seed))
    d_y = 8

    suffix_len = int(k_total) - (int(cut_t) + 1)
    v = np.empty((int(n_past), int(n_future) * d_y), dtype=np.float64)

    # Sample one fixed future probe set (preparation at cut + future suffix).
    prep_at_cut = np.empty((int(n_future), 2, 2), dtype=np.complex128)
    last_prep = np.empty((int(n_future), 2, 2), dtype=np.complex128)
    for b in range(int(n_future)):
        rho_prep, _eff, _feat = _sample_random_intervention_parts(rng)
        prep_at_cut[b] = rho_prep
        if suffix_len > 0:
            maps_suf, _rows = _sample_random_intervention_sequence(suffix_len, rng)
            last_prep[b] = np.asarray(maps_suf[-1].rho_prep, dtype=np.complex128).reshape(2, 2)
        else:
            last_prep[b] = rho_prep

    # Rows depend on history/effect at cut, but for the trivial system the **final normalized state**
    # depends only on the last preparation, so every row should be identical (up to floating noise).
    for a in range(int(n_past)):
        # history 0..t-1 (unused analytically here) + measurement at cut (unused analytically)
        _maps_pre, _rows_pre = _sample_random_intervention_sequence(int(cut_t), rng)
        _rho_unused, _eff_unused, _feat_unused = _sample_random_intervention_parts(rng)

        out_blocks = []
        for b in range(int(n_future)):
            rho_final = last_prep[b]
            out_blocks.append(pack_rho8(rho_final).astype(np.float64))
        v[a, :] = np.concatenate(out_blocks, axis=0)

    return v


def main() -> None:
    # Ensure headless plotting even on Windows machines without a display backend.
    os.environ.setdefault("MPLBACKEND", "Agg")

    print("benchmark_entropy: imports...", flush=True)
    import torch
    from torch.utils.data import TensorDataset

    from mqt.yaqs.characterization.process_tensors import TransformerComb, generate_data
    from mqt.yaqs.core.data_structures.networks import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, out_dir={out_dir}", flush=True)

    op = MPO.ising(length=int(L), J=J, g=G)
    params = AnalogSimParams(dt=float(DT), solver="MCWF", show_progress=False)

    print(f"generating dataset (k={K_TRAIN}, n={N_TRAIN})...", flush=True)
    ds = generate_data(
        op,
        params,
        k=int(K_TRAIN),
        n=int(N_TRAIN),
        seed=int(SEED) + 12345 * int(K_TRAIN),
        parallel=True,
        show_progress=True,
    )
    E_all, r0_all, tgt_all = ds.tensors
    idx = np.arange(int(E_all.shape[0]), dtype=np.int64)
    np.random.default_rng(int(SEED) + 54321).shuffle(idx)
    n_val = max(1, min(round(float(VAL_FRAC) * len(idx)), len(idx) - 1))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    train_ds = TensorDataset(E_all[tr_idx], r0_all[tr_idx], tgt_all[tr_idx])
    val_ds = TensorDataset(E_all[va_idx], r0_all[va_idx], tgt_all[va_idx])

    model = TransformerComb(
        d_e=int(E_all.shape[-1]),
        d_rho=8,
        d_model=int(D_MODEL),
        nhead=int(NHEAD),
        num_layers=int(NUM_LAYERS),
        dim_ff=int(DIM_FF),
        dropout=float(DROPOUT),
    ).to(device)
    print("training model...", flush=True)
    model.fit(
        train_ds,
        val_dataset=val_ds,
        epochs=int(EPOCHS),
        lr=float(LR),
        batch_size=int(BATCH_SIZE),
        prefix_loss=str(PREFIX_LOSS),
        device=device,
    )
    model.eval()
    assert model.sequence_length == int(K_TRAIN)

    cuts = list(range(1, int(K_TRAIN)))
    rows: list[dict[str, float | int | str]] = []
    raw_lines: list[str] = []

    for t in cuts:
        print(f"cut t={t}...", flush=True)
        repeats: list[float] = []
        for _ in range(int(N_REPEATS)):
            e = model.entropy(timestep=int(t), past_samples=int(PAST_SAMPLES), future_samples=int(FUTURE_SAMPLES))
            if not math.isfinite(e):
                raise RuntimeError(f"non-finite entropy at cut={t}")
            repeats.append(float(e))
        mean_e = float(np.mean(repeats))
        std_e = float(np.std(repeats)) if len(repeats) > 1 else 0.0
        rows.append({"cut": int(t), "entropy_mean": mean_e, "entropy_std": std_e})
        raw_lines.append(f"cut={t}: " + ", ".join(f"{x:.8f}" for x in repeats))

    print("\ncut, entropy_mean, entropy_std")
    for r in rows:
        print(f"{r['cut']}, {r['entropy_mean']:.8f}, {r['entropy_std']:.8f}")

    print("\nraw repeats (debug):")
    for line in raw_lines:
        print(line)

    csv_path = out_dir / "results_entropy.csv"
    try:
        import pandas as pd

        df = pd.DataFrame(rows, columns=["cut", "entropy_mean", "entropy_std"])
        df.to_csv(csv_path, index=False)
    except Exception:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cut", "entropy_mean", "entropy_std"])
            w.writeheader()
            w.writerows(rows)
    print(f"\nwrote {csv_path}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [int(r["cut"]) for r in rows]
    y = [float(r["entropy_mean"]) for r in rows]
    yerr = [float(r["entropy_std"]) for r in rows]
    ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, markersize=6)
    ax.set_xlabel("cut timestep t")
    ax.set_ylabel("entropy (nats)")
    ax.set_title(f"TransformerComb.entropy (k={K_TRAIN}, N_train={N_TRAIN}, repeats={N_REPEATS})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / "entropy_vs_cut.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"wrote {png_path}")

    # -----------------------------------------------------------------------
    # Exact-vs-model sanity check on the trivial system (L=1, J=g=0).
    # -----------------------------------------------------------------------
    print("\ntrivial exact-vs-model check (L=1,J=0,g=0):", flush=True)
    for t in cuts:
        exact_reps: list[float] = []
        model_reps: list[float] = []
        for rep in range(int(N_REPEATS)):
            # Use a deterministic seed per (t, rep) so exact/model use the same probe distribution.
            probe_seed = int(SEED) + 1_000_003 * int(t) + 10_007 * int(rep)
            v_exact = _build_v_exact_trivial(
                k_total=int(K_TRAIN),
                cut_t=int(t),
                n_past=int(PAST_SAMPLES),
                n_future=int(FUTURE_SAMPLES),
                seed=probe_seed,
            )
            e_exact = _entropy_from_v(v_exact)
            e_model = model.entropy(timestep=int(t), past_samples=int(PAST_SAMPLES), future_samples=int(FUTURE_SAMPLES))
            exact_reps.append(float(e_exact))
            model_reps.append(float(e_model))

        print(
            f"t={t}: exact_mean={float(np.mean(exact_reps)):.6f} "
            f"model_mean={float(np.mean(model_reps)):.6f} "
            f"(exact reps: {', '.join(f'{x:.4f}' for x in exact_reps)}) "
            f"(model reps: {', '.join(f'{x:.4f}' for x in model_reps)})",
            flush=True,
        )


if __name__ == "__main__":
    main()
