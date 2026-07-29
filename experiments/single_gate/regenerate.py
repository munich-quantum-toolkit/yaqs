# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Regenerate single-gate benchmark from repaired zip-up / variational / TDVP n=1."""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
from config import (
    ANGLE_TDVP_SUBSTEPS,
    BENCHMARK_ID,
    GATE_TYPE,
    METHODS,
    OUTPUT_DIR,
    Q0,
    Q1,
    SEED,
    SUBSTEP_ANGLE_X,
    SUBSTEP_VALUES,
    build_generic_angle_grid,
    build_special_angles,
    production_config,
)
from core import exact_reference, result_row, run_method
from gate_runtime import (
    L_DEFAULT,
    normalized_state_fidelity,
    prepare_initial_state,
    save_json,
)
from run import MainTextStore, export_csv, task_id
from variational import tt_svd_from_vec

from mqt.yaqs.core.libraries.gate_library import Z
from gate_runtime import apply_gate_to_dense_state


def zz_expectation(vec: np.ndarray) -> float:
    z2 = np.kron(np.asarray(Z().matrix), np.asarray(Z().matrix))
    return float(np.real(np.vdot(vec, apply_gate_to_dense_state(vec, z2, Q0, Q1, L_DEFAULT))))


def main() -> int:
    output = OUTPUT_DIR
    # Fresh DB from archive template meta only — do not reuse old method rows.
    db_path = output / "results.sqlite"
    if db_path.exists():
        db_path.unlink()
    for name in (
        "single_gate_angle_sweep.csv",
        "single_gate_substeps.csv",
        "single_gate_mpo_diagnostics.csv",
    ):
        p = output / name
        if p.exists():
            p.unlink()

    store = MainTextStore(db_path)
    # Restore χ selection from archive if present
    arch_meta = None
    for arch in sorted((Path(__file__).parent / "archive").glob("pre_repair_*/output/chi_selection.json")):
        arch_meta = json.loads(arch.read_text(encoding="utf-8"))
    if arch_meta:
        chi_low = int(arch_meta["chi0"])
        chi_mid = int(arch_meta["chi_intermediate"])
        chi_full = int(arch_meta["chi_full"])
    else:
        chi_low, chi_mid, chi_full = 8, 12, 16
    store.set_meta("chi_low", str(chi_low))
    store.set_meta("chi_intermediate", str(chi_mid))
    store.set_meta("chi_full", str(chi_full))
    store.set_meta("fidelity_definition", "normalized_state_fidelity_v2")
    store.set_meta("tdvp_angle_substeps", str(ANGLE_TDVP_SUBSTEPS))
    store.set_meta(
        "repair_protocol",
        "compress_rightcanon_ltr+var_multistart+tdvp_n1_v1",
    )

    initial = prepare_initial_state(SEED)
    zz = zz_expectation(initial["vec"])
    store.set_meta("zz_expectation", repr(zz))

    x_gen, _ = build_generic_angle_grid()
    x_spec, _ = build_special_angles()
    chis = (chi_low, chi_mid, chi_full)

    print(f"Angle sweep with TDVP substeps={ANGLE_TDVP_SUBSTEPS}")
    angle_tasks = []
    for chi in chis:
        for x in list(x_gen) + list(x_spec):
            special = any(abs(float(x) - float(s)) < 1e-12 for s in x_spec)
            angle_tasks.append((float(x), float(2 * np.pi * x), special, int(chi)))

    for i, (x, theta, special, chi) in enumerate(angle_tasks, 1):
        # no-update baseline
        exact = exact_reference(initial["vec"], theta)
        no_up = normalized_state_fidelity(exact, initial["vec"])
        analytic = float(np.sin(theta / 2.0) ** 2 * (1.0 - zz**2))
        payload_nu = {
            "benchmark_id": BENCHMARK_ID,
            "task_type": "angle_sweep",
            "method": "no_update",
            "chi_max": chi,
            "theta": theta,
            "substeps": 0,
            "special_angle": int(special),
        }
        row_nu = {
            "task_type": "angle_sweep",
            "method": "no_update",
            "chi_max": chi,
            "theta": theta,
            "x_fraction": x,
            "special_angle": int(special),
            "substeps": 0,
            "infidelity": no_up["infidelity_normalized"],
            "fidelity": no_up["fidelity_normalized"],
            "overlap_squared_raw": no_up["overlap_squared_raw"],
            "norm_squared_exact": no_up["norm_squared_exact"],
            "norm_squared_approx": no_up["norm_squared_approx"],
            "fidelity_normalized": no_up["fidelity_normalized"],
            "infidelity_normalized": no_up["infidelity_normalized"],
            "norm_loss": no_up["norm_loss"],
            "fidelity_definition": "normalized_state_fidelity_v2",
            "max_bond": max(initial["bond_profile"]),
            "bond_profile": json.dumps(initial["bond_profile"]),
            "param_count": 0,
            "runtime_s": 0.0,
            "peak_memory_mb": 0.0,
            "norm_before": 1.0,
            "norm_after": 1.0,
            "discarded_weight": 0.0,
            "variational_converged": None,
            "variational_failed": None,
            "failure_message": f"analytic={analytic:.16e}",
        }
        _insert(store, payload_nu, row_nu)

        # independent TT-SVD candidate (diagnostic; not a main-figure method)
        tt = tt_svd_from_vec(exact, L_DEFAULT, chi)
        tt_met = normalized_state_fidelity(exact, tt.to_vec())
        payload_tt = {
            "benchmark_id": BENCHMARK_ID,
            "task_type": "angle_sweep",
            "method": "ttsvd_candidate",
            "chi_max": chi,
            "theta": theta,
            "substeps": 0,
            "special_angle": int(special),
        }
        row_tt = dict(row_nu)
        row_tt.update(
            {
                "method": "ttsvd_candidate",
                "infidelity": tt_met["infidelity_normalized"],
                "fidelity": tt_met["fidelity_normalized"],
                "overlap_squared_raw": tt_met["overlap_squared_raw"],
                "norm_squared_approx": tt_met["norm_squared_approx"],
                "fidelity_normalized": tt_met["fidelity_normalized"],
                "infidelity_normalized": tt_met["infidelity_normalized"],
                "norm_loss": tt_met["norm_loss"],
                "norm_after": float(np.linalg.norm(tt.to_vec())),
                "failure_message": "independent_ttsvd_candidate",
            }
        )
        _insert(store, payload_tt, row_tt)

        for method in METHODS:
            payload = {
                "benchmark_id": BENCHMARK_ID,
                "task_type": "angle_sweep",
                "method": method,
                "chi_max": chi,
                "theta": theta,
                "substeps": ANGLE_TDVP_SUBSTEPS if method == "hybrid_tdvp" else 1,
                "special_angle": int(special),
            }
            result = run_method(
                initial["mps"],
                initial["vec"],
                theta=theta,
                method=method,
                chi=chi,
                substeps=ANGLE_TDVP_SUBSTEPS if method == "hybrid_tdvp" else 1,
            )
            row = result_row(
                task_type="angle_sweep",
                method=method,
                chi_max=chi,
                theta=theta,
                x_fraction=x,
                special_angle=special,
                substeps=ANGLE_TDVP_SUBSTEPS if method == "hybrid_tdvp" else 1,
                result=result,
            )
            _insert(store, payload, row)
        if i % 10 == 0 or i == len(angle_tasks):
            print(f"  angle {i}/{len(angle_tasks)}")

    print("TDVP subdivision study (full angle grid × n, χ set)")
    # Full substep study at all angles would be huge; do x grid × n for all χ,
    # plus denser n at x=0.1. User asked: recalculate TDVP for n in set over
    # full angle grid and all χ.
    for chi in chis:
        for x in list(x_gen) + list(x_spec):
            theta = float(2 * np.pi * x)
            special = any(abs(float(x) - float(s)) < 1e-12 for s in x_spec)
            for n in SUBSTEP_VALUES:
                payload = {
                    "benchmark_id": BENCHMARK_ID,
                    "task_type": "tdvp_substep_angle",
                    "method": "hybrid_tdvp",
                    "chi_max": chi,
                    "theta": theta,
                    "substeps": n,
                    "special_angle": int(special),
                }
                result = run_method(
                    initial["mps"],
                    initial["vec"],
                    theta=theta,
                    method="hybrid_tdvp",
                    chi=chi,
                    substeps=n,
                )
                row = result_row(
                    task_type="tdvp_substep_angle",
                    method="hybrid_tdvp",
                    chi_max=chi,
                    theta=theta,
                    x_fraction=float(x),
                    special_angle=special,
                    substeps=n,
                    result=result,
                )
                _insert(store, payload, row)
        print(f"  finished χ={chi} substep×angle grid")

    # Dedicated panel-(d) table at SUBSTEP_ANGLE_X (also stored as substep_sweep)
    theta = float(2 * np.pi * SUBSTEP_ANGLE_X)
    for chi in chis:
        for n in SUBSTEP_VALUES:
            payload = {
                "benchmark_id": BENCHMARK_ID,
                "task_type": "substep_sweep",
                "method": "hybrid_tdvp",
                "chi_max": chi,
                "theta": theta,
                "substeps": n,
            }
            # Reuse tdvp_substep_angle row if present
            tid = task_id(
                {
                    "benchmark_id": BENCHMARK_ID,
                    "task_type": "tdvp_substep_angle",
                    "method": "hybrid_tdvp",
                    "chi_max": chi,
                    "theta": theta,
                    "substeps": n,
                    "special_angle": 0,
                }
            )
            existing = store._conn.execute(
                "SELECT * FROM results WHERE task_id=?", (tid,)
            ).fetchone()
            if existing:
                cols = [d[0] for d in store._conn.execute("SELECT * FROM results LIMIT 0").description]
                row = dict(zip(cols, existing, strict=True))
                row["task_type"] = "substep_sweep"
                row["x_fraction"] = SUBSTEP_ANGLE_X
                payload_ss = {
                    "benchmark_id": BENCHMARK_ID,
                    "task_type": "substep_sweep",
                    "method": "hybrid_tdvp",
                    "chi_max": chi,
                    "theta": theta,
                    "substeps": n,
                }
                _insert(store, payload_ss, row)
            else:
                result = run_method(
                    initial["mps"], initial["vec"], theta=theta, method="hybrid_tdvp", chi=chi, substeps=n
                )
                row = result_row(
                    task_type="substep_sweep",
                    method="hybrid_tdvp",
                    chi_max=chi,
                    theta=theta,
                    x_fraction=SUBSTEP_ANGLE_X,
                    special_angle=False,
                    substeps=n,
                    result=result,
                )
                _insert(store, payload, row)

    export_csv(output / "single_gate_angle_sweep.csv", store.fetch_rows("angle_sweep"))
    export_csv(output / "single_gate_substeps.csv", store.fetch_rows("substep_sweep"))
    export_csv(output / "single_gate_tdvp_substep_angle.csv", store.fetch_rows("tdvp_substep_angle"))
    save_json(output / "config.json", production_config(chi_low=chi_low, chi_mid=chi_mid, chi_full=chi_full))
    save_json(
        output / "chi_selection.json",
        {"chi0": chi_low, "chi_intermediate": chi_mid, "chi_full": chi_full, "rule": "preserved_from_archive"},
    )
    store.close()
    print("Regeneration complete.")
    return 0


def _insert(store: MainTextStore, payload: dict, row: dict) -> None:
    tid = task_id(payload)
    if store.has_task(tid):
        return
    row = dict(row)
    row["task_id"] = tid
    if isinstance(row.get("bond_profile"), list):
        row["bond_profile"] = json.dumps(row["bond_profile"])
    for key in ("variational_converged", "variational_failed"):
        if row.get(key) is not None:
            row[key] = int(bool(row[key]))
        else:
            row[key] = None
    # Ensure all columns exist
    cols = [
        c
        for c in (
            "task_id",
            "task_type",
            "method",
            "chi_max",
            "theta",
            "x_fraction",
            "special_angle",
            "substeps",
            "infidelity",
            "fidelity",
            "overlap_squared_raw",
            "norm_squared_exact",
            "norm_squared_approx",
            "fidelity_normalized",
            "infidelity_normalized",
            "norm_loss",
            "fidelity_definition",
            "max_bond",
            "bond_profile",
            "param_count",
            "runtime_s",
            "peak_memory_mb",
            "norm_before",
            "norm_after",
            "discarded_weight",
            "variational_converged",
            "variational_failed",
            "failure_message",
        )
        if c in row
    ]
    store._conn.execute(
        f"INSERT INTO results ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [row[c] for c in cols],
    )


if __name__ == "__main__":
    raise SystemExit(main())
