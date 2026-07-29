# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Rebuild panel-(d) ``substep_sweep`` rows at ``SUBSTEP_ANGLE_X``."""

from __future__ import annotations

from config import BENCHMARK_ID, OUTPUT_DIR, SUBSTEP_ANGLE_X, SUBSTEP_VALUES, production_config
from gate_runtime import save_json
from run import MainTextStore, export_csv, task_id


def main() -> int:
    db_path = OUTPUT_DIR / "results.sqlite"
    store = MainTextStore(db_path)
    store._conn.execute("DELETE FROM results WHERE task_type='substep_sweep'")

    chi_low = int(store.get_meta("chi_low") or "8")
    chi_mid = int(store.get_meta("chi_intermediate") or "12")
    chi_full = int(store.get_meta("chi_full") or "16")
    chis = (chi_low, chi_mid, chi_full)
    theta = float(2.0 * 3.141592653589793 * SUBSTEP_ANGLE_X)

    n_inserted = 0
    for chi in chis:
        for n in SUBSTEP_VALUES:
            src_id = task_id(
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
            existing = store._conn.execute("SELECT * FROM results WHERE task_id=?", (src_id,)).fetchone()
            if existing is None:
                msg = f"Missing tdvp_substep_angle row for χ={chi} n={n} x={SUBSTEP_ANGLE_X}"
                raise SystemExit(msg)
            cols = [d[0] for d in store._conn.execute("SELECT * FROM results LIMIT 0").description]
            row = dict(zip(cols, existing, strict=True))
            row["task_type"] = "substep_sweep"
            row["x_fraction"] = float(SUBSTEP_ANGLE_X)
            payload = {
                "benchmark_id": BENCHMARK_ID,
                "task_type": "substep_sweep",
                "method": "hybrid_tdvp",
                "chi_max": chi,
                "theta": theta,
                "substeps": n,
            }
            row["task_id"] = task_id(payload)
            store.insert_row(row)
            n_inserted += 1

    export_csv(OUTPUT_DIR / "single_gate_substeps.csv", store.fetch_rows("substep_sweep"))
    save_json(OUTPUT_DIR / "config.json", production_config(chi_low=chi_low, chi_mid=chi_mid, chi_full=chi_full))
    store.set_meta("substep_angle_x", repr(SUBSTEP_ANGLE_X))
    store.close()
    print(f"Rebuilt {n_inserted} substep_sweep rows at θ/(2π)={SUBSTEP_ANGLE_X:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
