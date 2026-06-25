#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact benchmark: S_V vs number of zero-reset slots ell.

Fixed geometry (shared past/future probe pools):
``past(15) + [measure, prepare |0⟩] + [(|0⟩⟨0|, |0⟩)]^ell + [prepare_only] + future(5)``.

Here ``ell`` is the count of identity measure-prepare resets (not the gap/τ heatmap sweep).
Output: ``fig_entropy_vs_ell`` with representative S_V(ell) curves at fixed J.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from _benchmark_common import (
    BRANCH_WEIGHT_BETA,
    DT_DEFAULT,
    ELL_MAX_ELL,
    FUTURE_LEN_FIXED,
    G_DEFAULT,
    L_PAPER,
    PAST_LEN_FIXED,
    linear_weighted_metrics,
    list_initial_states_sys_env0,
    load_summary_csv,
    parse_int_list,
    write_summary_csv,
)
from _benchmark_plotting import plot_entropy_vs_ell

from mqt.yaqs.characterization.memory.diagnostics.probe import (
    ProbeSet,
    _sample_cut_measurement_only,
    _sample_cut_preparation_only,
    _sample_random_clifford_unitary,
    _sample_random_unitary,
    _sample_step,
)
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

J_SWEEP_ELL = [0.5, 1.0, 1.5, 2.0]


def _sample_base_past_future_ensemble(
    *,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
) -> tuple[list[list[Any]], list[np.ndarray], list[np.ndarray], list[list[Any]]]:
    mode = str(intervention_mode).strip().lower()
    if mode not in {"unitary_break_mp", "measure_prepare"}:
        msg = f"intervention_mode must be 'unitary_break_mp' or 'measure_prepare', got {intervention_mode!r}"
        raise ValueError(msg)
    ensemble = str(unitary_ensemble).strip().lower()
    if ensemble not in {"haar", "clifford"}:
        msg = f"unitary_ensemble must be 'haar' or 'clifford', got {unitary_ensemble!r}"
        raise ValueError(msg)
    unitary_sampler = _sample_random_unitary if ensemble == "haar" else _sample_random_clifford_unitary

    past_pairs: list[list[Any]] = []
    past_cut_meas: list[np.ndarray] = []
    for _ in range(n_pasts):
        pairs_i: list[Any] = []
        for _ in range(PAST_LEN_FIXED):
            if mode == "measure_prepare":
                _feat, pair = _sample_step(rng)
                pairs_i.append(pair)
            else:
                pairs_i.append({"type": "unitary", "U": unitary_sampler(rng)})
        _feat_m, psi_m = _sample_cut_measurement_only(rng)
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[Any]] = []
    for _ in range(n_futures):
        _feat_p, psi_p = _sample_cut_preparation_only(rng)
        future_prep_cut.append(psi_p)
        pairs_j: list[Any] = []
        for _ in range(FUTURE_LEN_FIXED):
            if mode == "measure_prepare":
                _feat, pair = _sample_step(rng)
                pairs_j.append(pair)
            else:
                pairs_j.append({"type": "unitary", "U": unitary_sampler(rng)})
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
            full.extend((z0, z0) for _ in range(ell_i))
            full.append({"type": "prepare_only", "psi_prep": np.asarray(future_prep_cut[j], dtype=np.complex128)})
            full.extend(copy.deepcopy(future_pairs[j]))
            if len(full) != k_this:
                msg = f"internal: sequence length mismatch, got {len(full)} expected {k_this}"
                raise RuntimeError(msg)
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
        n_pasts_grid=n_p,
        n_futures_grid=n_f,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-pasts", type=int, default=None)
    p.add_argument("--n-futures", type=int, default=None)
    p.add_argument("--ells", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--unitary-ensemble", type=str, default="haar", choices=("haar", "clifford"))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--summary-csv", type=Path, default=None)
    return p.parse_args()


def _resolve_config(args: argparse.Namespace) -> dict[str, object]:
    if args.quick:
        ells = "0,1,2,3,4,5" if args.ells is None else str(args.ells)
        n_pasts = 8 if args.n_pasts is None else int(args.n_pasts)
        n_futures = 8 if args.n_futures is None else int(args.n_futures)
        n_seeds = 1 if args.n_seeds is None else int(args.n_seeds)
        out_dir = Path("benchmark_entropy_vs_j_by_ell_quick_results") if args.out_dir is None else args.out_dir
    else:
        ells = ",".join(str(i) for i in range(ELL_MAX_ELL + 1)) if args.ells is None else str(args.ells)
        n_pasts = 64 if args.n_pasts is None else int(args.n_pasts)
        n_futures = 64 if args.n_futures is None else int(args.n_futures)
        n_seeds = 5 if args.n_seeds is None else int(args.n_seeds)
        out_dir = Path("benchmark_entropy_vs_j_by_ell_results") if args.out_dir is None else args.out_dir
    return {
        "ells": parse_int_list(ells),
        "n_pasts": n_pasts,
        "n_futures": n_futures,
        "n_seeds": n_seeds,
        "out_dir": Path(out_dir),
    }


def main() -> None:
    args = _parse_args()
    cfg = _resolve_config(args)
    out_dir = cfg["out_dir"].resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot_only):
        csv_path = Path(args.summary_csv) if args.summary_csv is not None else out_dir / "summary.csv"
        plot_entropy_vs_ell(load_summary_csv(csv_path), out_dir / "fig_entropy_vs_ell")
        return

    ells = cfg["ells"]
    n_seeds = int(cfg["n_seeds"])
    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = list_initial_states_sys_env0(length=L_PAPER, n_seeds=n_seeds, rng=init_rng)
    np.save(out_dir / "initial_states.npy", np.stack(initial_list, axis=0))

    probe_rng = np.random.default_rng(int(args.seed) + 999_991)
    past_pairs, past_cut_meas, future_prep_cut, future_pairs = _sample_base_past_future_ensemble(
        n_pasts=int(cfg["n_pasts"]),
        n_futures=int(cfg["n_futures"]),
        rng=probe_rng,
        unitary_ensemble=str(args.unitary_ensemble),
    )

    rows: list[dict[str, float | int]] = []
    sim_params = AnalogSimParams(dt=DT_DEFAULT)

    for ell in ells:
        if ell < 0 or ell > ELL_MAX_ELL:
            msg = f"ell must satisfy 0 <= ell <= {ELL_MAX_ELL}, got {ell}"
            raise ValueError(msg)
        left_cut = int(PAST_LEN_FIXED + 1)
        k_this = int(PAST_LEN_FIXED + 1 + ell + 1 + FUTURE_LEN_FIXED)
        probe_set = _probe_set_for_ell(
            past_pairs=past_pairs,
            past_cut_meas=past_cut_meas,
            future_prep_cut=future_prep_cut,
            future_pairs=future_pairs,
            ell=int(ell),
        )
        for jv in J_SWEEP_ELL:
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
                "right_cut": int(left_cut + ell + 1),
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
    plot_entropy_vs_ell(rows, out_dir / "fig_entropy_vs_ell")


if __name__ == "__main__":
    main()
