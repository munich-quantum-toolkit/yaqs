# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Variational MPO circuit control (TFIM χ=32 traj + Heisenberg first step)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from config import (
    CHI_HEISENBERG,
    CHI_MAIN,
    CORRECTED_OUTPUT_DIR,
    DT,
    RELIABILITY_THRESHOLD,
    TMAX_INITIAL,
    timesteps_for_tmax,
)
from generate_corrected import CSV_FIELDS, _write_csv, precompute_exact, run_trajectory
from horizons import reliable_horizon


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=CORRECTED_OUTPUT_DIR)
    parser.add_argument("--tfim-stop-after-crossing", action="store_true", default=True)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    n = int(cfg["tdvp_substeps"])
    tmax_steps = timesteps_for_tmax(TMAX_INITIAL)
    exact_ising = np.load(out / f"exact_ising_t{tmax_steps}.npy")
    exact_heis = np.load(out / f"exact_heisenberg_t{tmax_steps}.npy")

    rows = []
    print(f"TFIM variational χ={CHI_MAIN} n={n}", flush=True)
    rows.extend(
        run_trajectory(
            model="ising",
            method="variational_mpo",
            chi=CHI_MAIN,
            timesteps=tmax_steps,
            exact=exact_ising,
            tdvp_substeps=n,
            stop_after_crossing=bool(args.tfim_stop_after_crossing),
        )
    )
    for chi in CHI_HEISENBERG:
        print(f"Heisenberg variational χ={chi} step 1", flush=True)
        rows.extend(
            run_trajectory(
                model="heisenberg",
                method="variational_mpo",
                chi=chi,
                timesteps=1,
                exact=exact_heis,
                tdvp_substeps=n,
                stop_after_crossing=True,
            )
        )
    _write_csv(out / "variational_circuit_control.csv", rows, CSV_FIELDS)

    # Compare TFIM horizon vs zip-up
    zip_traj = [
        r
        for r in csv.DictReader((out / "circuit_results_corrected.csv").open(encoding="utf-8"))
        if r["model"] == "ising" and r["method"] == "mpo_zipup" and int(float(r["chi_max"])) == CHI_MAIN
    ]
    var_traj = [r for r in rows if r["model"] == "ising"]
    hz = reliable_horizon(zip_traj, epsilon=RELIABILITY_THRESHOLD, dt=DT)
    hv = reliable_horizon(var_traj, epsilon=RELIABILITY_THRESHOLD, dt=DT)
    summary = {
        "tfim_zipup_T_eps": hz["T_eps"],
        "tfim_variational_T_eps": hv["T_eps"],
        "horizon_delta_steps": int(hv["n_eps"]) - int(hz["n_eps"]),
        "tracks_zipup": abs(int(hv["n_eps"]) - int(hz["n_eps"])) <= 1,
        "include_in_main": int(hv["n_eps"]) > int(hz["n_eps"]) + 1,
    }
    (out / "variational_control_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
