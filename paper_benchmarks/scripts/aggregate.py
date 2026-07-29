# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Stage 4 (aggregate): build tidy processed CSVs from raw + raw_new inputs.

Outputs (processed/):
  single_gate_angle_sweep.csv   merged corrected campaign (RZZ seed 11) +
                                paper_benchmarks extension (all gates/seeds)
  single_gate_theta_zero.csv    theta = 0 identity rows
  single_gate_substeps_x025.csv exact-limit substep study (x=1/4, chi=32)
  single_gate_substeps_x001.csv corrected substep grid (RZZ seed 11, x=0.01
                                and representative angles, chi in {8,12,16})
  circuit_trajectories.csv      chi=32 trajectories: 4x4 TFIM/Heisenberg and
                                1D TFIM/XXX chains; TDVP method is full_tdvp
                                (gate-local TDVP on ALL two-qubit gates,
                                including nearest-neighbour). Hybrid TDVP
                                (2D only) retained for hybrid-vs-full comparison.
  circuit_horizons.csv          corrected reliability horizons (supplement)

Also runs cross-set consistency assertions and writes logs/aggregate_report.json.

Usage:
    uv run --with pandas python paper_benchmarks/scripts/aggregate.py
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from pb_common import (
    LOGS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RAW_NEW_DIR,
    save_json,
)

TIDY_COLS = [
    "task_type", "gate_type", "seed", "q0", "q1", "separation", "method",
    "chi_max", "chi0", "theta", "x_fraction", "special_angle", "substeps",
    "infidelity", "phase_aligned_error_exact", "phase_aligned_error_selfref",
    "norm_loss", "norm_drift", "max_bond", "bond_profile", "param_count",
    "runtime_s", "discarded_weight", "variational_converged",
    "variational_failed", "failure_message", "source",
]


def load_corrected_single_gate() -> pd.DataFrame:
    db = RAW_DIR / "single_gate_corrected" / "results.sqlite"
    con = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
    df = pd.read_sql("SELECT * FROM results", con)
    con.close()
    df["gate_type"] = "rzz"
    df["seed"] = 11
    df["q0"], df["q1"], df["separation"], df["chi0"] = 2, 9, 7, 8
    df["infidelity"] = df["infidelity_normalized"]
    df["norm_drift"] = (1.0 - df["norm_after"]).abs()
    df["phase_aligned_error_exact"] = np.nan
    df["phase_aligned_error_selfref"] = np.nan
    df["source"] = "corrected_campaign"
    return df


def load_extension() -> pd.DataFrame:
    df = pd.read_csv(RAW_NEW_DIR / "single_gate_ext.csv")
    df["source"] = "paper_benchmarks_ext"
    return df


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"checks": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    corr = load_corrected_single_gate()
    ext = load_extension()

    # --- single-gate angle sweep -----------------------------------------
    angle = pd.concat(
        [
            corr[corr.task_type == "angle_sweep"].reindex(columns=TIDY_COLS),
            ext[ext.task_type == "angle_sweep"].reindex(columns=TIDY_COLS),
        ],
        ignore_index=True,
    )
    # no duplicated cells (corrected campaign supplies rzz/seed 11 only)
    keys = ["gate_type", "seed", "method", "chi_max", "x_fraction", "substeps"]
    dups = angle.duplicated(subset=keys).sum()
    check("angle_sweep_no_duplicate_cells", dups == 0, f"({dups} duplicates)")
    # full grid: 3 gates x 3 seeds x 3 chi x 27 x-values x 6 methods
    expected = 3 * 3 * 3 * 27 * 6
    check("angle_sweep_complete_grid", len(angle) == expected,
          f"({len(angle)} rows, expected {expected})")
    # chi=16 is effectively exact for the direct MPO methods, all gates/seeds
    full = angle[(angle.chi_max == 16) & angle.method.isin(["mpo_zipup", "variational_mpo"])]
    worst_full = float(full.infidelity.max())
    check("chi16_mpo_methods_near_exact", worst_full < 1e-10, f"(max {worst_full:.2e})")
    # zero errors must be stored as true zeros (no display flooring in data)
    check("stored_zeros_preserved", float(angle.infidelity.min()) == 0.0)
    angle.to_csv(PROCESSED_DIR / "single_gate_angle_sweep.csv", index=False)

    theta0 = ext[ext.task_type == "theta_zero"].reindex(columns=TIDY_COLS)
    t0_corr = corr[corr.task_type == "theta_zero"]
    if len(t0_corr):
        theta0 = pd.concat([t0_corr.reindex(columns=TIDY_COLS), theta0], ignore_index=True)
    # theta=0 must be identity for non-routing methods
    t0 = theta0[~theta0.method.isin(["tebd_swap"])]
    worst_t0 = float(t0.infidelity.max())
    check("theta_zero_identity_nonrouting", worst_t0 < 1e-12, f"(max {worst_t0:.2e})")
    theta0.to_csv(PROCESSED_DIR / "single_gate_theta_zero.csv", index=False)

    # --- substep studies ---------------------------------------------------
    ss_new = ext[ext.task_type == "substep_study"].reindex(columns=TIDY_COLS)
    nonbinding = (ss_new.max_bond < ss_new.chi_max).all() and (
        ss_new.discarded_weight.max() < 1e-12
    )
    check("substep_study_cap_nonbinding", bool(nonbinding),
          f"(peak bond {int(ss_new.max_bond.max())} < cap {int(ss_new.chi_max.max())}, "
          f"max discarded {ss_new.discarded_weight.max():.1e})")
    ss_new.to_csv(PROCESSED_DIR / "single_gate_substeps_x025.csv", index=False)

    ss_old = corr[corr.task_type.isin(["substep_sweep", "tdvp_substep_angle"])]
    ss_old.reindex(columns=TIDY_COLS).to_csv(
        PROCESSED_DIR / "single_gate_substeps_x001.csv", index=False,
    )

    # --- circuits ----------------------------------------------------------
    circ = pd.read_csv(RAW_DIR / "circuits_corrected" / "circuit_results_corrected.csv")
    # Comparators (TEBD, zip-up) and hybrid TDVP from corrected / full re-run.
    tfim_cmp = circ[
        (circ.model == "ising") & (circ.chi_max == 32)
        & circ.method.isin(["hybrid_tdvp", "tebd_swap", "mpo_zipup"])
    ].copy()
    tfim_cmp["source"] = "corrected_campaign"
    heis_parts = []
    for m in ("hybrid_tdvp", "tebd_swap", "mpo_zipup"):
        d = pd.read_csv(RAW_NEW_DIR / "heisenberg_chi32_full" / f"heisenberg_chi32_{m}.csv")
        d["source"] = "paper_benchmarks_full_run"
        heis_parts.append(d)
    heis_cmp = pd.concat(heis_parts, ignore_index=True)

    # consistency: overlapping early Heisenberg steps agree with corrected campaign
    old_h = circ[(circ.model == "heisenberg") & (circ.chi_max == 32)]
    max_diff = 0.0
    for m in ("hybrid_tdvp", "tebd_swap", "mpo_zipup"):
        o = old_h[old_h.method == m].set_index("trotter_step").infidelity
        n = heis_cmp[heis_cmp.method == m].set_index("trotter_step").infidelity
        common = o.index.intersection(n.index)
        max_diff = max(max_diff, float((o.loc[common] - n.loc[common]).abs().max()))
    check(
        "heisenberg_rerun_consistency",
        max_diff < 5e-3,
        f"(max |diff| {max_diff:.2e}; TDVP/zip-up ~1e-13, TEBD 1.8e-3 roundoff "
        "under heavy step-1 truncation, flagged in validation report)",
    )

    # 2D full_tdvp: TDVP window update on every two-qubit gate (incl. NN).
    twod_full_parts = []
    for model in ("ising", "heisenberg"):
        d = pd.read_csv(
            RAW_NEW_DIR / "circuits_2d_full_tdvp" / f"{model}_chi32_full_tdvp.csv"
        )
        d["source"] = "paper_benchmarks_2d_full_tdvp"
        twod_full_parts.append(d)
    twod_full = pd.concat(twod_full_parts, ignore_index=True)
    twod_full_len = (twod_full.groupby("model").trotter_step.max() == 30).all()
    check("circuit_2d_full_tdvp_full_length", bool(twod_full_len))

    # hybrid vs full must differ on 2D (NN gates take different code paths).
    hyb_vs_full = True
    for model, hyb_df in (("ising", tfim_cmp), ("heisenberg", heis_cmp)):
        a = hyb_df[hyb_df.method == "hybrid_tdvp"].set_index("trotter_step").infidelity
        b = twod_full[twod_full.model == model].set_index("trotter_step").infidelity
        if float((a - b).abs().max()) == 0.0:
            hyb_vs_full = False
    check("circuit_2d_full_tdvp_distinct_from_hybrid", hyb_vs_full)

    # 1D chains (TFIM J=g=1, XXX Heisenberg J=h=1); TDVP applied to ALL
    # two-qubit gates ("full_tdvp"), plus TEBD and MPO zip-up comparators.
    oned_parts = []
    for model in ("ising_1d", "heisenberg_1d"):
        for m in ("full_tdvp", "tebd_swap", "mpo_zipup"):
            d = pd.read_csv(RAW_NEW_DIR / "circuits_1d" / f"{model}_chi32_{m}.csv")
            d["source"] = "paper_benchmarks_1d"
            oned_parts.append(d)
    oned = pd.concat(oned_parts, ignore_index=True)
    oned_full_len = (oned.groupby(["model", "method"]).trotter_step.max() == 30).all()
    check("circuit_1d_full_length", bool(oned_full_len))
    step0_max = float(oned[oned.trotter_step == 0].infidelity.abs().max())
    check("circuit_1d_step0_identity", step0_max < 1e-12, f"(max {step0_max:.1e})")
    # full_tdvp must be a genuinely different update than TEBD (all gates NN):
    # identical trajectories would mean the TDVP path was silently bypassed.
    tdvp_used = True
    for model in ("ising_1d", "heisenberg_1d"):
        a = oned[(oned.model == model) & (oned.method == "full_tdvp")].set_index(
            "trotter_step").infidelity
        b = oned[(oned.model == model) & (oned.method == "tebd_swap")].set_index(
            "trotter_step").infidelity
        if float((a - b).abs().max()) == 0.0:
            tdvp_used = False
    check("circuit_1d_full_tdvp_distinct_from_tebd", tdvp_used)

    traj = pd.concat([tfim_cmp, heis_cmp, twod_full, oned], ignore_index=True)
    traj.to_csv(PROCESSED_DIR / "circuit_trajectories.csv", index=False)

    horizons = pd.read_csv(RAW_DIR / "circuits_corrected" / "circuit_horizons_corrected.csv")
    horizons.to_csv(PROCESSED_DIR / "circuit_horizons.csv", index=False)

    report["rows"] = {
        "angle_sweep": len(angle),
        "theta_zero": len(theta0),
        "substeps_x025": len(ss_new),
        "substeps_x001": len(ss_old),
        "circuit_trajectories": len(traj),
        "circuit_horizons": len(horizons),
    }
    report["all_pass"] = all(c["pass"] for c in report["checks"])
    save_json(LOGS_DIR / "aggregate_report.json", report)
    print(f"\n{'ALL PASS' if report['all_pass'] else 'FAILURES PRESENT'}; "
          f"rows: {report['rows']}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
