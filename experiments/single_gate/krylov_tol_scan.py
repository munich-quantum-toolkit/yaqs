# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Krylov-tolerance scan for TDVP at χ_max = 16 (panel c)."""

from __future__ import annotations

import csv
import copy
import time
from pathlib import Path

import numpy as np
from config import ANGLE_TDVP_SUBSTEPS, GATE_TYPE, OUTPUT_DIR, Q0, Q1, SEED
from core import exact_reference
from gate_runtime import (
    DiscardedWeightTracker,
    KRYLOV_TOL,
    apply_method,
    make_dag_node,
    normalized_state_fidelity,
    prepare_initial_state,
    save_json,
)

CHI_FULL = 16
# Default production tol is 1e-12; probe coarser and tighter.
KRYLOV_TOLS = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16)
# Cover the weak-angle floor visible in panel (c) plus a mid/strong angle.
X_VALUES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def main() -> int:
    initial = prepare_initial_state(SEED)
    node_factory = lambda theta: make_dag_node(GATE_TYPE, theta, Q0, Q1, initial["mps"].length)
    rows: list[dict] = []
    print(f"Krylov tol scan: χ={CHI_FULL}, n_substeps={ANGLE_TDVP_SUBSTEPS}, default={KRYLOV_TOL:g}")
    for x in X_VALUES:
        theta = float(2.0 * np.pi * x)
        exact = exact_reference(initial["vec"], theta)
        node = node_factory(theta)
        for tol in KRYLOV_TOLS:
            tracker = DiscardedWeightTracker()
            t0 = time.perf_counter()
            state, runtime, discarded = apply_method(
                copy.deepcopy(initial["mps"]),
                node,
                method="hybrid_tdvp",
                chi=CHI_FULL,
                substeps=ANGLE_TDVP_SUBSTEPS,
                tracker=tracker,
                krylov_tol=tol,
            )
            wall = time.perf_counter() - t0
            fid = normalized_state_fidelity(exact, state.to_vec())
            row = {
                "chi_max": CHI_FULL,
                "x_fraction": x,
                "theta": theta,
                "substeps": ANGLE_TDVP_SUBSTEPS,
                "krylov_tol": tol,
                "infidelity": fid["infidelity_normalized"],
                "fidelity": fid["fidelity_normalized"],
                "runtime_s": runtime,
                "wall_s": wall,
                "discarded_weight": discarded,
                "max_bond": max(int(t.shape[2]) for t in state.tensors[:-1]),
            }
            rows.append(row)
            print(
                f"  x={x:<8g} tol={tol:<8.0e} 1-F={row['infidelity']:.6e}  "
                f"t={runtime:.3f}s  disc={discarded:.3e}"
            )

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "krylov_tol_scan_chi16.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summarize whether tightening tol moves the weak-angle floor.
    summary: dict = {"default_krylov_tol": KRYLOV_TOL, "chi_max": CHI_FULL, "by_x": {}}
    for x in X_VALUES:
        subset = [r for r in rows if r["x_fraction"] == x]
        infs = [r["infidelity"] for r in subset]
        summary["by_x"][repr(x)] = {
            "min_infidelity": min(infs),
            "max_infidelity": max(infs),
            "at_default": next(r["infidelity"] for r in subset if r["krylov_tol"] == KRYLOV_TOL),
            "at_tightest": next(r["infidelity"] for r in subset if r["krylov_tol"] == KRYLOV_TOLS[-1]),
            "at_coarsest": next(r["infidelity"] for r in subset if r["krylov_tol"] == KRYLOV_TOLS[0]),
            "ratio_tight_over_default": (
                next(r["infidelity"] for r in subset if r["krylov_tol"] == KRYLOV_TOLS[-1])
                / max(next(r["infidelity"] for r in subset if r["krylov_tol"] == KRYLOV_TOL), 1e-30)
            ),
        }
    # Floor change criterion: weak-angle (x<=1e-3) span across tols.
    weak = [r for r in rows if r["x_fraction"] <= 1e-3]
    weak_span = max(r["infidelity"] for r in weak) - min(r["infidelity"] for r in weak)
    summary["weak_angle_infidelity_span"] = weak_span
    summary["plateau_sensitive_to_krylov_tol"] = bool(weak_span > 1e-13 and max(r["infidelity"] for r in weak) > 1e-14)
    # More precise: does tightest differ from default by > factor 2 or absolute 1e-13?
    changes = []
    for x in (1e-4, 1e-3):
        d = next(r["infidelity"] for r in rows if r["x_fraction"] == x and r["krylov_tol"] == KRYLOV_TOL)
        t = next(r["infidelity"] for r in rows if r["x_fraction"] == x and r["krylov_tol"] == KRYLOV_TOLS[-1])
        changes.append(abs(d - t) > max(1e-13, 0.5 * max(d, t, 1e-30)))
    summary["tightening_changes_weak_floor"] = any(changes)
    save_json(out / "krylov_tol_scan_chi16_summary.json", summary)
    print(f"\nWrote {csv_path}")
    print(f"weak_angle_span={weak_span:.3e} tightening_changes_floor={summary['tightening_changes_weak_floor']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
