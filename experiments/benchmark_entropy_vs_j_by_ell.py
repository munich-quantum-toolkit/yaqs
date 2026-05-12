#!/usr/bin/env python3
"""Exact benchmark: fixed-window delayed break sweep ``S_V(ell, J)``.

This benchmark fixes the window geometry and sweeps only the delay length ``ell``:

``past(15) + [measure, prepare |0>] + [(|0><0|, |0>) ]^ell + [prepare_only] + future(5)``.

The past/future random probe ensemble is sampled **once** (fixed lengths); each ``ell`` only
extends the intermediate reset measure-prepare bridge.

Outputs a single paper-style plot of representative ``S_V`` vs ``ell`` curves for multiple ``J``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_entropy_vs_j_by_cut import (
    BRANCH_WEIGHT_BETA,
    DT_FIXED,
    G_FIXED,
    HEATMAP_COLOR_VMIN,
    J_SWEEP_DEFAULT,
    L_FIXED,
    _configure_matplotlib_prl_figure,
    _linear_weighted_metrics,
    _list_initial_states_sys_env0,
    _load_summary_csv,
)
from mqt.yaqs.characterization.process_tensors.diagnostics.probe import (
    ProbeSet,
    _sample_cut_measurement_only,
    _sample_cut_preparation_only,
    _sample_random_clifford_unitary,
    _sample_random_unitary,
    _sample_step,
)
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

J_SWEEP_DEFAULT = [0.5, 1.0, 1.5, 2.0]
PAST_LEN_FIXED = 15
FUTURE_LEN_FIXED = 5
ELL_MAX_FIXED = 15
ELL_DEFAULT = tuple(range(0, ELL_MAX_FIXED + 1))
PANEL_TARGET_JS = (0.5, 1.0, 1.5, 2.0)


def _sample_base_past_future_ensemble(
    *,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
) -> tuple[list[list[Any]], list[np.ndarray], list[np.ndarray], list[list[Any]]]:
    """Sample **one** past/future probe ensemble (fixed lengths); reused for every ``ell``."""
    mode = str(intervention_mode).strip().lower()
    if mode not in {"unitary_break_mp", "measure_prepare"}:
        raise ValueError(f"intervention_mode must be 'unitary_break_mp' or 'measure_prepare', got {intervention_mode!r}")
    ensemble = str(unitary_ensemble).strip().lower()
    if ensemble not in {"haar", "clifford"}:
        raise ValueError(f"unitary_ensemble must be 'haar' or 'clifford', got {unitary_ensemble!r}")
    unitary_sampler = _sample_random_unitary if ensemble == "haar" else _sample_random_clifford_unitary

    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for _i in range(n_pasts):
        pairs_i: list[Any] = []
        for _t in range(PAST_LEN_FIXED):
            if mode == "measure_prepare":
                _feat, pair = _sample_step(rng)
                pairs_i.append(pair)
            else:
                u = unitary_sampler(rng)
                pairs_i.append({"type": "unitary", "U": u})
        _feat_m, psi_m = _sample_cut_measurement_only(rng)
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for _j in range(n_futures):
        _feat_p, psi_p = _sample_cut_preparation_only(rng)
        future_prep_cut.append(psi_p)
        pairs_j: list[Any] = []
        for _t in range(FUTURE_LEN_FIXED):
            if mode == "measure_prepare":
                _feat, pair = _sample_step(rng)
                pairs_j.append(pair)
            else:
                u = unitary_sampler(rng)
                pairs_j.append({"type": "unitary", "U": u})
        future_pairs.append(pairs_j)

    return past_pairs, past_cut_meas, future_prep_cut, future_pairs


def _probe_set_for_ell(
    *,
    past_pairs: list[list[Any]],
    past_cut_meas: list[np.ndarray],
    future_prep_cut: list[np.ndarray],
    future_pairs: list[list[Any]],
    ell: int,
) -> ProbeSet:
    """Assemble full ``all_pairs_grid``: past + left MP break + ``ell`` reset MP + prepare + future."""
    left_cut = int(PAST_LEN_FIXED + 1)
    ell_i = int(ell)
    k_this = int(PAST_LEN_FIXED + 1 + ell_i + 1 + FUTURE_LEN_FIXED)
    z0 = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    n_p = len(past_pairs)
    n_f = len(future_pairs)
    all_pairs: list[list[Any]] = []
    for i in range(n_p):
        for j in range(n_f):
            full: list[Any] = []
            full.extend(copy.deepcopy(past_pairs[i]))
            psi_m = np.asarray(past_cut_meas[i], dtype=np.complex128)
            full.append((psi_m, z0))
            for _ in range(ell_i):
                full.append((z0, z0))
            full.append(
                {"type": "prepare_only", "psi_prep": np.asarray(future_prep_cut[j], dtype=np.complex128)}
            )
            full.extend(copy.deepcopy(future_pairs[j]))
            if len(full) != k_this:
                raise RuntimeError(f"internal: sequence length mismatch, got {len(full)} expected {k_this}")
            all_pairs.append(full)

    past_features = np.zeros((n_p, max(1, PAST_LEN_FIXED + 1), 32), dtype=np.float32)
    future_features = np.zeros((n_f, max(1, 1 + ell_i + FUTURE_LEN_FIXED), 32), dtype=np.float32)
    return ProbeSet(
        cut=left_cut,
        k=k_this,
        past_features=past_features,
        future_features=future_features,
        past_pairs=copy.deepcopy(past_pairs),
        past_cut_meas=[np.asarray(x, dtype=np.complex128) for x in past_cut_meas],
        future_prep_cut=[np.asarray(x, dtype=np.complex128) for x in future_prep_cut],
        future_pairs=copy.deepcopy(future_pairs),
        all_pairs_grid=all_pairs,
        n_pasts_grid=int(n_p),
        n_futures_grid=int(n_f),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pasts", type=int, default=64)
    p.add_argument("--n-futures", type=int, default=64)
    p.add_argument("--ells", type=str, default=",".join(str(g) for g in ELL_DEFAULT))
    p.add_argument("--taus", type=str, default="", help="Deprecated alias for --ells.")
    p.add_argument("--gaps", type=str, default="", help="Deprecated alias for --ells.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_entropy_vs_j_by_ell_results"))
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--unitary-ensemble", type=str, default="haar", choices=("haar", "clifford"))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--summary-csv", type=Path, default=None)
    return p.parse_args()


def _parse_int_list(spec: str) -> list[int]:
    vals = sorted({int(tok.strip()) for tok in spec.split(",") if tok.strip()})
    if not vals:
        raise ValueError("expected at least one ell")
    for g in vals:
        if g < 0 or g > ELL_MAX_FIXED:
            raise ValueError(f"ell must satisfy 0 <= ell <= {ELL_MAX_FIXED}, got {g}")
    return vals


def _write_summary_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_entropy_vs_ell(rows: list[dict[str, str | float | int]], out_stem: Path) -> None:
    """Plot representative ``S_V`` vs ``ell`` curves at fixed couplings ``J``."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    if not rows:
        return
    _configure_matplotlib_prl_figure()
    ell_key = "ell" if ("ell" in rows[0]) else "tau"
    ells = sorted({int(float(r[ell_key])) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    ell_arr = np.asarray(ells, dtype=np.float64)
    j_arr = np.asarray(j_vals, dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2), constrained_layout=True)
    ax.set_facecolor("white")
    target_js = [float(j_arr[int(np.argmin(np.abs(j_arr - t)))]) for t in PANEL_TARGET_JS]
    target_js = list(dict.fromkeys(target_js))
    cmap = plt.get_cmap("Reds")
    norm = Normalize(vmin=0.0, vmax=2.0)
    for jv in target_js:
        sub = sorted((r for r in rows if float(r["J"]) == float(jv)), key=lambda r: int(float(r[ell_key])))
        if not sub:
            continue
        ax.semilogy(
            [int(float(r[ell_key])) for r in sub],
            [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in sub],
            lw=1.9,
            marker="o",
            ms=3.8,
            markeredgewidth=0.0,
            color=cmap(norm(jv)),
            alpha=0.94,
            label=rf"$J={jv:g}$",
        )

    ax.set_xlabel(r"Delay $\ell$")
    ax.set_ylabel(r"$S_V$")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.grid(True, which="major", axis="y", alpha=0.10, linewidth=0.35)
    if len(ell_arr) > 1:
        ax.set_xlim(ell_arr[0] - 0.4, ell_arr[-1] + 0.4)
    y_all = [max(float(r["entropy"]), HEATMAP_COLOR_VMIN) for r in rows]
    y_hi = min(1.0, max(HEATMAP_COLOR_VMIN * 1.2, (float(np.nanmax(y_all)) * 1.25 if y_all else 1.0)))
    ax.set_ylim(HEATMAP_COLOR_VMIN, y_hi)
    ax.legend(frameon=False, fontsize=7.0, handlelength=1.4, borderaxespad=0.2, loc="upper right")

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    prl_stem = out_stem.with_name(f"{out_stem.name}_prl")
    fig.savefig(prl_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(prl_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot_only):
        csv_path = Path(args.summary_csv) if args.summary_csv is not None else out_dir / "summary.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"summary CSV not found: {csv_path}")
        rows_raw = _load_summary_csv(csv_path)
        plot_entropy_vs_ell(rows_raw, out_dir / "fig_entropy_vs_ell")
        print(f"Wrote figure: {out_dir / 'fig_entropy_vs_ell.pdf'}", flush=True)
        return

    ell_spec = str(args.ells).strip()
    if not ell_spec:
        ell_spec = str(args.taus).strip() or str(args.gaps).strip()
    if (str(args.taus).strip() or str(args.gaps).strip()) and not str(args.ells).strip():
        print("Using deprecated --taus/--gaps alias; please use --ells.", flush=True)
    ells = _parse_int_list(ell_spec)

    n_seeds = int(args.n_seeds)
    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = _list_initial_states_sys_env0(n_seeds=n_seeds, rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    probe_rng = np.random.default_rng(int(args.seed) + 999_991)
    past_pairs, past_cut_meas, future_prep_cut, future_pairs = _sample_base_past_future_ensemble(
        n_pasts=int(args.n_pasts),
        n_futures=int(args.n_futures),
        rng=probe_rng,
        unitary_ensemble=str(args.unitary_ensemble),
    )

    rows: list[dict[str, float | int]] = []
    for ell in ells:
        left_cut = int(PAST_LEN_FIXED + 1)
        right_cut = int(left_cut + ell + 1)
        k_this = int(PAST_LEN_FIXED + 1 + ell + 1 + FUTURE_LEN_FIXED)
        print(
            f"ell={ell:2d}, left_cut={left_cut:2d}, right_cut={right_cut:2d}, k={k_this:2d}, "
            f"past={PAST_LEN_FIXED}, future={FUTURE_LEN_FIXED}, bridge=ell measure+prepare(|0>), "
            "shared ensemble",
            flush=True,
        )
        probe_set = _probe_set_for_ell(
            past_pairs=past_pairs,
            past_cut_meas=past_cut_meas,
            future_prep_cut=future_prep_cut,
            future_pairs=future_pairs,
            ell=ell,
        )
        for jv in J_SWEEP_DEFAULT:
            op = MPO.ising(length=L_FIXED, J=float(jv), g=G_FIXED)
            sim_params = AnalogSimParams(dt=DT_FIXED, solver="MCWF", show_progress=False)
            entropies: list[float] = []
            delta_norms: list[float] = []
            ranks: list[int] = []
            for psi0 in initial_list:
                m = _linear_weighted_metrics(
                    probe_set=probe_set,
                    op=op,
                    sim_params=sim_params,
                    psi0=psi0,
                    parallel=bool(args.parallel),
                )
                entropies.append(float(m["entropy"]))
                delta_norms.append(float(m["delta_norm"]))
                ranks.append(int(m["rank"]))
            rows.append(
                {
                    "L": L_FIXED,
                    "k": int(k_this),
                    "dt": DT_FIXED,
                    "g": G_FIXED,
                    "left_cut": int(left_cut),
                    "tau": int(ell),
                    "ell": int(ell),
                    "right_cut": int(right_cut),
                    "past_len_fixed": int(PAST_LEN_FIXED),
                    "future_len_fixed": int(FUTURE_LEN_FIXED),
                    "J": float(jv),
                    "n_pasts": int(args.n_pasts),
                    "n_futures": int(args.n_futures),
                    "n_seeds": n_seeds,
                    "branch_weight_beta": BRANCH_WEIGHT_BETA,
                    "entropy": float(np.mean(entropies)),
                    "entropy_std": float(np.std(entropies, ddof=1)) if len(entropies) > 1 else 0.0,
                    "delta_norm": float(np.mean(delta_norms)),
                    "rank": int(round(float(np.mean(ranks)))),
                }
            )
            print(f"ell={ell:2d}, J={jv:>4.2f}, S_mean={rows[-1]['entropy']:.6e}", flush=True)

    _write_summary_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    plot_entropy_vs_ell(rows, out_dir / "fig_entropy_vs_ell")
    print(f"Wrote results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

