# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Summarize method runtimes from the corrected single-gate angle sweep."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from config import OUTPUT_DIR
from run import MainTextStore

METHODS = ("hybrid_tdvp", "tebd_swap", "mpo_zipup", "variational_mpo")
LABELS = {
    "hybrid_tdvp": "TDVP (n=1)",
    "tebd_swap": "TEBD+SWAP",
    "mpo_zipup": "MPO zip-up",
    "variational_mpo": "Variational MPO",
}
KEY_X = (1e-4, 1e-3, 1e-2, 0.1, 1.0)


def main() -> int:
    store = MainTextStore(OUTPUT_DIR / "results.sqlite")
    rows = store.fetch_rows("angle_sweep")
    store.close()
    by = {}
    for r in rows:
        if r["method"] not in METHODS:
            continue
        key = (int(r["chi_max"]), float(r["x_fraction"]), r["method"])
        by[key] = float(r["runtime_s"])

    chis = sorted({k[0] for k in by})
    xs = sorted({k[1] for k in by})

    out_rows = []
    for chi in chis:
        for x in xs:
            times = {m: by.get((chi, x, m)) for m in METHODS}
            if any(v is None for v in times.values()):
                continue
            fastest = min(times, key=times.get)  # type: ignore[arg-type]
            out_rows.append(
                {
                    "chi_max": chi,
                    "x_fraction": x,
                    **{f"runtime_{m}": times[m] for m in METHODS},
                    "fastest": fastest,
                    "tdvp_vs_zipup": times["hybrid_tdvp"] / times["mpo_zipup"],
                    "tdvp_vs_tebd": times["hybrid_tdvp"] / times["tebd_swap"],
                    "tdvp_vs_variational": times["hybrid_tdvp"] / times["variational_mpo"],
                }
            )

    csv_path = Path(OUTPUT_DIR) / "runtime_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print("Median runtime_s over full angle grid:")
    for chi in chis:
        print(f"  χ={chi}")
        for m in METHODS:
            ts = [r[f"runtime_{m}"] for r in out_rows if r["chi_max"] == chi]
            print(f"    {LABELS[m]:18s} med={statistics.median(ts):7.4f}s  min={min(ts):7.4f} max={max(ts):7.4f}")

    wins = {m: 0 for m in METHODS}
    for r in out_rows:
        wins[r["fastest"]] += 1
    print(f"\nFastest among production methods ({len(out_rows)} points):")
    for m in METHODS:
        print(f"  {LABELS[m]:18s} {wins[m]}")

    tdvp_fastest = wins["hybrid_tdvp"]
    print(f"\nTDVP ever fastest? {'yes' if tdvp_fastest else 'no'} ({tdvp_fastest}/{len(out_rows)})")
    print("TDVP slower than zip-up at every recorded (χ, x) point." if all(r["tdvp_vs_zipup"] > 1 for r in out_rows) else "TDVP sometimes beats zip-up.")
    print("TDVP slower than TEBD+SWAP at every recorded (χ, x) point." if all(r["tdvp_vs_tebd"] > 1 for r in out_rows) else "TDVP sometimes beats TEBD.")
    print(
        f"TDVP faster than variational at "
        f"{sum(1 for r in out_rows if r['tdvp_vs_variational'] < 1)}/{len(out_rows)} points."
    )

    print("\nKey-point snapshot (runtime_s):")
    print(f"{'χ':>3} {'x':>8} {'TDVP':>8} {'TEBD':>8} {'zip-up':>8} {'varMPO':>8} {'fastest':>12}")
    for chi in chis:
        for x in KEY_X:
            r = next((row for row in out_rows if row["chi_max"] == chi and abs(row["x_fraction"] - x) < 1e-15), None)
            if r is None:
                continue
            print(
                f"{chi:3d} {x:8.1e} "
                f"{r['runtime_hybrid_tdvp']:8.4f} "
                f"{r['runtime_tebd_swap']:8.4f} "
                f"{r['runtime_mpo_zipup']:8.4f} "
                f"{r['runtime_variational_mpo']:8.4f} "
                f"{LABELS[r['fastest']]:>12}"
            )
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
