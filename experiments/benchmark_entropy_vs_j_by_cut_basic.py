#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Basic entropy-vs-J-by-cut benchmark (β=1, unitary_break_mp, causal cut weights)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _benchmark_common import (
    BRANCH_WEIGHT_BETA,
    DT_DEFAULT,
    G_DEFAULT,
    K_DEFAULT,
    L_DEFAULT,
    linear_weighted_metrics,
    list_initial_states_sys_env0,
    parse_float_list,
    parse_int_list,
    sample_probe_set,
    write_summary_csv,
    write_summary_json,
)

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=int, default=L_DEFAULT)
    p.add_argument("--k", type=int, default=K_DEFAULT)
    p.add_argument("--dt", type=float, default=DT_DEFAULT)
    p.add_argument("--g", type=float, default=G_DEFAULT)
    p.add_argument("--n-pasts", type=int, default=16)
    p.add_argument("--n-futures", type=int, default=16)
    p.add_argument("--cuts", type=str, default="3,5,8,10")
    p.add_argument("--js", type=str, default="0.0,0.4,0.8,1.2,1.6,2.0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_entropy_vs_j_by_cut_basic_results"))
    p.add_argument("--parallel", action="store_true", default=False)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--unitary-ensemble", type=str, default="haar", choices=("haar", "clifford"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    length = int(args.length)
    k = int(args.k)
    cuts = parse_int_list(str(args.cuts))
    js = parse_float_list(str(args.js))

    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = list_initial_states_sys_env0(length=length, n_seeds=int(args.n_seeds), rng=init_rng)

    rows: list[dict[str, float | int]] = []

    for cut in cuts:
        probe_set = sample_probe_set(
            cut=cut,
            k=k,
            n_pasts=int(args.n_pasts),
            n_futures=int(args.n_futures),
            seed=int(args.seed),
            unitary_ensemble=str(args.unitary_ensemble),
        )
        for jv in js:
            op = MPO.ising(length=length, J=float(jv), g=float(args.g))
            sim_params = AnalogSimParams(dt=float(args.dt))
            ent_list: list[float] = []
            rank_list: list[int] = []
            for psi0 in initial_list:
                m = linear_weighted_metrics(
                    probe_set=probe_set,
                    op=op,
                    sim_params=sim_params,
                    psi0=psi0,
                    parallel=bool(args.parallel),
                )
                ent_list.append(float(m["entropy"]))
                rank_list.append(int(m["rank"]))
            row = {
                "L": length,
                "k": k,
                "dt": float(args.dt),
                "g": float(args.g),
                "cut": int(cut),
                "J": float(jv),
                "branch_weight_beta": BRANCH_WEIGHT_BETA,
                "entropy": float(np.mean(ent_list)),
                "rank": float(np.mean(rank_list)),
            }
            rows.append(row)

    write_summary_csv(out_dir / "summary.csv", rows)
    write_summary_json(out_dir / "summary.json", rows)


if __name__ == "__main__":
    main()
