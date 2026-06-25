#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact benchmark: bond entropy S_V vs coupling J for multiple causal cuts.

Paper-style outputs:
- ``fig_entropy_vs_j_by_cut`` — colored curves per cut
- ``fig_entropy_heatmap_cut_vs_J`` — three-panel heatmap + cross-sections

Use ``--quick`` for a small smoke run (L=2, k=8). Full defaults match the
published benchmark geometry (L=6, k=20, dense J sweep).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _benchmark_common import (
    BRANCH_WEIGHT_BETA,
    DT_DEFAULT,
    G_DEFAULT,
    linear_weighted_metrics,
    list_initial_states_sys_env0,
    parse_int_list,
    sample_probe_set,
    write_summary_csv,
    write_summary_json,
)
from _benchmark_plotting import plot_entropy_heatmap_cut_vs_j, plot_entropy_vs_j
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

L_PAPER = 6
K_PAPER = 20
J_SWEEP_PAPER = [0.05 * i for i in range(41)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Small geometry for visual smoke tests.")
    p.add_argument("--length", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--n-pasts", type=int, default=None)
    p.add_argument("--n-futures", type=int, default=None)
    p.add_argument("--cuts", type=str, default=None)
    p.add_argument("--js", type=str, default=None, help="Comma-separated J values (overrides default sweep).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--unitary-ensemble", type=str, default="haar", choices=("haar", "clifford"))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--summary-csv", type=Path, default=None)
    return p.parse_args()


def _resolve_config(args: argparse.Namespace) -> dict[str, object]:
    if args.quick:
        length = 2 if args.length is None else int(args.length)
        k = 8 if args.k is None else int(args.k)
        cuts = f"1,2,3,4,5,6,7,8" if args.cuts is None else str(args.cuts)
        js = "0.0,0.4,0.8,1.2,1.6,2.0" if args.js is None else str(args.js)
        n_pasts = 12 if args.n_pasts is None else int(args.n_pasts)
        n_futures = 12 if args.n_futures is None else int(args.n_futures)
        out_dir = Path("benchmark_entropy_vs_j_by_cut_quick_results") if args.out_dir is None else args.out_dir
    else:
        length = L_PAPER if args.length is None else int(args.length)
        k = K_PAPER if args.k is None else int(args.k)
        cuts = ",".join(str(i) for i in range(1, k + 1)) if args.cuts is None else str(args.cuts)
        js = None if args.js is None else str(args.js)
        n_pasts = 32 if args.n_pasts is None else int(args.n_pasts)
        n_futures = 32 if args.n_futures is None else int(args.n_futures)
        out_dir = Path("benchmark_entropy_vs_j_by_cut_results") if args.out_dir is None else args.out_dir
    j_list = [float(x.strip()) for x in js.split(",")] if js else list(J_SWEEP_PAPER)
    return {
        "length": length,
        "k": k,
        "cuts": parse_int_list(cuts),
        "js": j_list,
        "n_pasts": n_pasts,
        "n_futures": n_futures,
        "out_dir": Path(out_dir),
    }


def _load_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = _parse_args()
    cfg = _resolve_config(args)
    out_dir = cfg["out_dir"].resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot_only):
        csv_path = Path(args.summary_csv) if args.summary_csv is not None else out_dir / "summary.csv"
        rows_raw = _load_summary_csv(csv_path)
        k = int(cfg["k"])
        plot_entropy_vs_j([{**r, "entropy": float(r["entropy"]), "cut": int(float(r["cut"])), "J": float(r["J"])} for r in rows_raw], out_dir / "fig_entropy_vs_j_by_cut", k=k)
        plot_entropy_heatmap_cut_vs_j(rows_raw, out_dir / "fig_entropy_heatmap_cut_vs_J")
        print(f"Wrote figures under: {out_dir}", flush=True)
        return

    length = int(cfg["length"])
    k = int(cfg["k"])
    cuts = [c for c in cfg["cuts"] if 1 <= c <= k]
    js = cfg["js"]

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = list_initial_states_sys_env0(length=length, n_seeds=int(args.n_seeds), rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    print(f"=== S_V vs J by cut (L={length}, k={k}, beta={BRANCH_WEIGHT_BETA}) ===", flush=True)
    rows: list[dict[str, float | int]] = []

    for cut in cuts:
        probe_set = sample_probe_set(
            cut=cut,
            k=k,
            n_pasts=int(cfg["n_pasts"]),
            n_futures=int(cfg["n_futures"]),
            seed=int(args.seed),
            unitary_ensemble=str(args.unitary_ensemble),
        )
        for jv in js:
            op = MPO.ising(length=length, J=float(jv), g=G_DEFAULT)
            sim_params = AnalogSimParams(dt=DT_DEFAULT)
            entropies: list[float] = []
            ranks: list[int] = []
            for psi0 in initial_list:
                m = linear_weighted_metrics(
                    probe_set=probe_set,
                    op=op,
                    sim_params=sim_params,
                    psi0=psi0,
                    parallel=bool(args.parallel),
                )
                entropies.append(float(m["entropy"]))
                ranks.append(int(m["rank"]))
            row = {
                "L": length,
                "k": k,
                "dt": DT_DEFAULT,
                "g": G_DEFAULT,
                "cut": int(cut),
                "J": float(jv),
                "n_pasts": int(cfg["n_pasts"]),
                "n_futures": int(cfg["n_futures"]),
                "branch_weight_beta": BRANCH_WEIGHT_BETA,
                "entropy": float(np.mean(entropies)),
                "rank": int(round(float(np.mean(ranks)))),
            }
            rows.append(row)
            print(f"cut={cut:2d}, J={jv:4.1f}, S_V={row['entropy']:.6e}", flush=True)

    write_summary_csv(out_dir / "summary.csv", rows)
    write_summary_json(out_dir / "summary.json", rows)
    plot_entropy_vs_j(rows, out_dir / "fig_entropy_vs_j_by_cut", k=k)
    plot_entropy_heatmap_cut_vs_j(rows, out_dir / "fig_entropy_heatmap_cut_vs_J")
    print(f"Wrote results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
