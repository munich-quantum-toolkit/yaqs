#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact benchmark: operational memory vs delay length ell at fixed couplings J.

Uses :func:`sample_split_delayed_break_probes` — past + left break + ell identity
slots + prepare + future. Output: ``fig_entropy_vs_ell``.
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
    parse_float_list,
    parse_int_list,
    write_summary_csv,
    write_summary_json,
)
from _benchmark_plotting import plot_entropy_vs_ell
from mqt.yaqs.characterization.memory.diagnostics.probe import sample_split_delayed_break_probes
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

J_SWEEP_DEFAULT = [0.5, 1.0, 1.5, 2.0]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--length", type=int, default=None)
    p.add_argument("--left-cut", type=int, default=None, help="Past length + 1 (causal cut before delay bridge).")
    p.add_argument("--future-tail", type=int, default=None, help="Future steps after right prepare.")
    p.add_argument("--ells", type=str, default=None)
    p.add_argument("--js", type=str, default=None)
    p.add_argument("--n-pasts", type=int, default=None)
    p.add_argument("--n-futures", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
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
        left_cut = 4 if args.left_cut is None else int(args.left_cut)
        future_tail = 2 if args.future_tail is None else int(args.future_tail)
        ells = "0,1,2,3,4,5" if args.ells is None else str(args.ells)
        js = "0.5,1.0,1.5,2.0" if args.js is None else str(args.js)
        n_pasts = 8 if args.n_pasts is None else int(args.n_pasts)
        n_futures = 8 if args.n_futures is None else int(args.n_futures)
        out_dir = Path("benchmark_entropy_vs_j_by_ell_quick_results") if args.out_dir is None else args.out_dir
    else:
        length = 6 if args.length is None else int(args.length)
        left_cut = 16 if args.left_cut is None else int(args.left_cut)
        future_tail = 5 if args.future_tail is None else int(args.future_tail)
        ells = ",".join(str(i) for i in range(16)) if args.ells is None else str(args.ells)
        js = ",".join(str(v) for v in J_SWEEP_DEFAULT) if args.js is None else str(args.js)
        n_pasts = 64 if args.n_pasts is None else int(args.n_pasts)
        n_futures = 64 if args.n_futures is None else int(args.n_futures)
        out_dir = Path("benchmark_entropy_vs_j_by_ell_results") if args.out_dir is None else args.out_dir
    return {
        "length": length,
        "left_cut": left_cut,
        "future_tail": future_tail,
        "ells": parse_int_list(ells),
        "js": parse_float_list(js),
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
        plot_entropy_vs_ell(_load_summary_csv(csv_path), out_dir / "fig_entropy_vs_ell")
        print(f"Wrote figure: {out_dir / 'fig_entropy_vs_ell.png'}", flush=True)
        return

    length = int(cfg["length"])
    left_cut = int(cfg["left_cut"])
    future_tail = int(cfg["future_tail"])
    ells = cfg["ells"]
    js = cfg["js"]

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = list_initial_states_sys_env0(length=length, n_seeds=int(args.n_seeds), rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    probe_rng = np.random.default_rng(int(args.seed) + 999_991)
    base_past_future_rng = probe_rng

    print(f"=== S_V vs delay ell (L={length}, left_cut={left_cut}, future_tail={future_tail}) ===", flush=True)
    rows: list[dict[str, float | int]] = []

    for ell in ells:
        k_this = left_cut + int(ell) + 1 + future_tail
        probe_set = sample_split_delayed_break_probes(
            left_cut=left_cut,
            tau=int(ell),
            k=k_this,
            n_pasts=int(cfg["n_pasts"]),
            n_futures=int(cfg["n_futures"]),
            rng=np.random.default_rng(int(base_past_future_rng.integers(0, 2**31 - 1))),
            unitary_ensemble=str(args.unitary_ensemble),
        )
        for jv in js:
            op = MPO.ising(length=length, J=float(jv), g=G_DEFAULT)
            sim_params = AnalogSimParams(dt=DT_DEFAULT)
            entropies: list[float] = []
            for psi0 in initial_list:
                m = linear_weighted_metrics(
                    probe_set=probe_set,
                    op=op,
                    sim_params=sim_params,
                    psi0=psi0,
                    parallel=bool(args.parallel),
                )
                entropies.append(float(m["entropy"]))
            rows.append(
                {
                    "L": length,
                    "k": k_this,
                    "ell": int(ell),
                    "left_cut": left_cut,
                    "J": float(jv),
                    "entropy": float(np.mean(entropies)),
                    "branch_weight_beta": BRANCH_WEIGHT_BETA,
                }
            )
            print(f"ell={ell:2d}, J={jv:4.2f}, S_V={rows[-1]['entropy']:.6e}", flush=True)

    write_summary_csv(out_dir / "summary.csv", rows)
    write_summary_json(out_dir / "summary.json", rows)
    plot_entropy_vs_ell(rows, out_dir / "fig_entropy_vs_ell")
    print(f"Wrote results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
