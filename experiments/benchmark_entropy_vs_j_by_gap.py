#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact benchmark: fixed-window delayed break sweep S_V(ell, J) — paper gap figure.

Construction per sequence:
``past(15) + [measure, prepare |0>] + [identity]^ell + [prepare_only] + future(5)``.

Outputs (mc-process parity):
- ``fig_entropy_vs_ell`` — representative S_V vs delay curves
- ``fig_entropy_heatmap_gap_vs_J`` — three-panel heatmap + cross-sections
- ``fig_entropy_heatmap_tau_j_pair`` — operational memory horizon panel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _benchmark_common import (
    BRANCH_WEIGHT_BETA,
    DT_DEFAULT,
    ELL_MAX_GAP,
    FUTURE_LEN_FIXED,
    G_DEFAULT,
    J_SWEEP_PAPER,
    L_PAPER,
    PAST_LEN_FIXED,
    SV_THRESHOLD_DEFAULT,
    linear_weighted_metrics,
    list_initial_states_sys_env0,
    load_summary_csv,
    parse_int_list,
    write_summary_csv,
)
from _benchmark_plotting import (
    plot_entropy_heatmap_gap_vs_j,
    plot_entropy_heatmap_tau_j_pair,
    plot_entropy_vs_ell,
)

from mqt.yaqs.characterization.memory.diagnostics.probe import sample_split_delayed_break_probes
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

SIGMA_REF = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Small ell/J grid for smoke tests.")
    p.add_argument("--n-pasts", type=int, default=None)
    p.add_argument("--n-futures", type=int, default=None)
    p.add_argument("--ells", type=str, default=None)
    p.add_argument("--js", type=str, default=None, help="Comma-separated J values (default: dense paper sweep).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--unitary-ensemble", type=str, default="haar", choices=("haar", "clifford"))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--sv-threshold", type=float, default=SV_THRESHOLD_DEFAULT)
    p.add_argument("--taus", type=str, default="", help="Deprecated alias for --ells.")
    p.add_argument("--gaps", type=str, default="", help="Deprecated alias for --ells.")
    return p.parse_args()


def _resolve_config(args: argparse.Namespace) -> dict[str, object]:
    if args.quick:
        ells = "0,1,2,3,4,5" if args.ells is None else str(args.ells)
        js = "0.0,0.5,1.0,1.5,2.0" if args.js is None else str(args.js)
        n_pasts = 12 if args.n_pasts is None else int(args.n_pasts)
        n_futures = 12 if args.n_futures is None else int(args.n_futures)
        out_dir = Path("benchmark_entropy_vs_j_by_gap_quick_results") if args.out_dir is None else args.out_dir
        j_list = [float(x.strip()) for x in js.split(",") if x.strip()]
    else:
        ells = ",".join(str(i) for i in range(ELL_MAX_GAP + 1)) if args.ells is None else str(args.ells)
        j_list = (
            list(J_SWEEP_PAPER) if args.js is None else [float(x.strip()) for x in str(args.js).split(",") if x.strip()]
        )
        n_pasts = 32 if args.n_pasts is None else int(args.n_pasts)
        n_futures = 32 if args.n_futures is None else int(args.n_futures)
        out_dir = Path("benchmark_entropy_vs_j_by_gap_results") if args.out_dir is None else args.out_dir
    return {
        "ells": parse_int_list(ells),
        "js": j_list,
        "n_pasts": n_pasts,
        "n_futures": n_futures,
        "out_dir": Path(out_dir),
    }


def _plot_all(rows: list[dict[str, str | float | int]], out_dir: Path, *, sv_threshold: float) -> None:
    plot_entropy_vs_ell(rows, out_dir / "fig_entropy_vs_ell")
    plot_entropy_heatmap_gap_vs_j(rows, out_dir / "fig_entropy_heatmap_gap_vs_J")
    plot_entropy_heatmap_tau_j_pair(rows, out_dir / "fig_entropy_heatmap_tau_j_pair", sv_threshold=sv_threshold)


def main() -> None:
    args = _parse_args()
    cfg = _resolve_config(args)
    out_dir = cfg["out_dir"].resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot_only):
        csv_path = Path(args.summary_csv) if args.summary_csv is not None else out_dir / "summary.csv"
        if not csv_path.is_file():
            msg = f"summary CSV not found: {csv_path}"
            raise FileNotFoundError(msg)
        rows_raw = load_summary_csv(csv_path)
        _plot_all(rows_raw, out_dir, sv_threshold=float(args.sv_threshold))
        return

    ell_spec = str(args.ells).strip() if args.ells is not None else ""
    if not ell_spec:
        ell_spec = str(args.taus).strip() or str(args.gaps).strip()
    (str(args.taus).strip() or str(args.gaps).strip()) and args.ells is None

    ells = cfg["ells"]
    js = cfg["js"]
    n_seeds = int(args.n_seeds)

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = list_initial_states_sys_env0(length=L_PAPER, n_seeds=n_seeds, rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    rows: list[dict[str, float | int]] = []

    for ell in ells:
        if ell < 0 or ell > ELL_MAX_GAP:
            msg = f"ell must satisfy 0 <= ell <= {ELL_MAX_GAP}, got {ell}"
            raise ValueError(msg)
        left_cut = int(PAST_LEN_FIXED + 1)
        right_cut = int(left_cut + ell + 1)
        k_this = int(PAST_LEN_FIXED + 1 + ell + 1 + FUTURE_LEN_FIXED)
        probe_rng = np.random.default_rng(int(args.seed) + 10_000 * int(ell))
        probe_set = sample_split_delayed_break_probes(
            left_cut=left_cut,
            tau=int(ell),
            k=k_this,
            n_pasts=int(cfg["n_pasts"]),
            n_futures=int(cfg["n_futures"]),
            rng=probe_rng,
            sigma_ref=SIGMA_REF,
            unitary_ensemble=str(args.unitary_ensemble),
        )
        sim_params = AnalogSimParams(dt=DT_DEFAULT)
        for jv in js:
            op = MPO.ising(length=L_PAPER, J=float(jv), g=G_DEFAULT)
            entropies: list[float] = []
            delta_norms: list[float] = []
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
                delta_norms.append(float(m["delta_norm"]))
                ranks.append(int(m["rank"]))
            rows.append({
                "L": L_PAPER,
                "k": k_this,
                "dt": DT_DEFAULT,
                "g": G_DEFAULT,
                "left_cut": left_cut,
                "tau": int(ell),
                "ell": int(ell),
                "right_cut": right_cut,
                "past_len_fixed": PAST_LEN_FIXED,
                "future_len_fixed": FUTURE_LEN_FIXED,
                "J": float(jv),
                "n_pasts": int(cfg["n_pasts"]),
                "n_futures": int(cfg["n_futures"]),
                "n_seeds": n_seeds,
                "branch_weight_beta": BRANCH_WEIGHT_BETA,
                "entropy": float(np.mean(entropies)),
                "entropy_std": float(np.std(entropies, ddof=1)) if len(entropies) > 1 else 0.0,
                "delta_norm": float(np.mean(delta_norms)),
                "rank": round(float(np.mean(ranks))),
            })

    write_summary_csv(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _plot_all([{**r} for r in rows], out_dir, sv_threshold=float(args.sv_threshold))


if __name__ == "__main__":
    main()
