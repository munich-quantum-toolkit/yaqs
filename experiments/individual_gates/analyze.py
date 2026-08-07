# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Matched summaries for the individual-gates campaign (no pooled method table)."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import conventional_median  # noqa: E402
from config import (  # noqa: E402
    CNOT_RANK_CHI_VALUES,
    CNOT_RANK_CONTROL,
    CNOT_RANK_DISPLAY_N_SUB,
    CNOT_RANK_TARGET,
    METHODS,
    OUTPUT_DIR,
    PAULI_GATES,
    SEEDS,
)

# chi_max used in main-text Pauli panels
PAULI_PANEL_CHI = 8


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _range_stats(vals: list[float]) -> dict[str, float]:
    arr = [float(v) for v in vals]
    return {
        "n": len(arr),
        "min": float(min(arr)) if arr else float("nan"),
        "median": conventional_median(arr),
        "max": float(max(arr)) if arr else float("nan"),
        "mean": float(sum(arr) / len(arr)) if arr else float("nan"),
    }


def write_pauli_finite_angle_summary(rows: list[dict[str, str]], path: Path) -> dict[str, Any]:
    """Matched Pauli summary at χ=8, n_sub=1, excluding identity (x=0) pooling abuse."""
    out_rows: list[dict[str, Any]] = []
    for gate in PAULI_GATES:
        for method in METHODS:
            by_x: dict[float, list[float]] = defaultdict(list)
            for row in rows:
                if row["family"] != "pauli":
                    continue
                if row["gate"] != gate or row["method"] != method:
                    continue
                if int(row["chi_max"]) != PAULI_PANEL_CHI:
                    continue
                if int(float(row["n_sub"])) != 1:
                    continue
                x = float(row["x"])
                by_x[x].append(float(row["infidelity_normalized"]))
            for x in sorted(by_x):
                st = _range_stats(by_x[x])
                out_rows.append(
                    {
                        "gate": gate,
                        "method": method,
                        "chi_max": PAULI_PANEL_CHI,
                        "n_sub": 1,
                        "x": x,
                        "theta": 2.0 * math.pi * x,
                        "n_states": st["n"],
                        "infidelity_min": st["min"],
                        "infidelity_median": st["median"],
                        "infidelity_max": st["max"],
                        "is_identity_angle": abs(x) < 1e-30,
                        "seeds": list(SEEDS),
                    }
                )

    fields = [
        "gate",
        "method",
        "chi_max",
        "n_sub",
        "x",
        "theta",
        "n_states",
        "infidelity_min",
        "infidelity_median",
        "infidelity_max",
        "is_identity_angle",
        "seeds",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({**row, "seeds": json.dumps(row["seeds"])})

    # Finite-angle only aggregate (x>0) for quick inspection — matched per gate/method.
    finite = [r for r in out_rows if not r["is_identity_angle"]]
    return {
        "csv": str(path),
        "n_matched_cells": len(out_rows),
        "n_finite_angle_cells": len(finite),
        "chi_max": PAULI_PANEL_CHI,
        "note": (
            "Matched cells: same gate, method, χ_max, n_sub, and angle; "
            "median/min/max over three states. Do not pool across unmatched "
            "caps/angles/gates. Discarded weights are not compared across methods "
            "here (methods perform different SVD sequences)."
        ),
        "tebd_x0": [r for r in out_rows if r["method"] == "tebd_swap" and r["is_identity_angle"]],
    }


def write_cnot_rank_summary_csv(rows: list[dict[str, str]], path: Path) -> dict[str, Any]:
    """Matched CNOT-rank summary: median/min/max over seeds at each (χ, method, n_sub)."""
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["chi_max"]), row["method"], int(float(row["n_sub"])))
        buckets[key].append(float(row["infidelity_normalized"]))

    out_rows: list[dict[str, Any]] = []
    for chi in CNOT_RANK_CHI_VALUES:
        for method in ("mpo_zipup", "tebd_swap"):
            key = (chi, method, 1)
            st = _range_stats(buckets.get(key, []))
            out_rows.append(
                {
                    "chi_max": chi,
                    "method": method,
                    "n_sub": 1,
                    "display": True,
                    "resolution_label": "direct",
                    **{f"infidelity_{k}": st[k] for k in ("min", "median", "max")},
                    "n_states": st["n"],
                }
            )
        for n_sub in (1, 16, 128, 256):
            key = (chi, "gate_local_2tdvp", n_sub)
            st = _range_stats(buckets.get(key, []))
            label = "fine_resolution" if n_sub == 256 else f"n_sub_{n_sub}"
            out_rows.append(
                {
                    "chi_max": chi,
                    "method": "gate_local_2tdvp",
                    "n_sub": n_sub,
                    "display": n_sub in CNOT_RANK_DISPLAY_N_SUB,
                    "resolution_label": label,
                    **{f"infidelity_{k}": st[k] for k in ("min", "median", "max")},
                    "n_states": st["n"],
                }
            )

    fields = [
        "chi_max",
        "method",
        "n_sub",
        "display",
        "resolution_label",
        "infidelity_min",
        "infidelity_median",
        "infidelity_max",
        "n_states",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    return {
        "csv": str(path),
        "n_rows": len(out_rows),
        "control": CNOT_RANK_CONTROL,
        "target": CNOT_RANK_TARGET,
        "sites_paper_one_based": [CNOT_RANK_CONTROL + 1, CNOT_RANK_TARGET + 1],
        "display_tdvp_n_sub": list(CNOT_RANK_DISPLAY_N_SUB),
        "note": (
            "n_sub=128 retained as refinement-control; display uses {1,16,256}. "
            "n_sub=256 is fine resolution, not labeled converged."
        ),
    }


def summarize_refinement(rows: list[dict[str, str]]) -> dict[str, Any]:
    parsed = []
    for row in rows:
        cap_key = "cap_reached" if "cap_reached" in row else "hard_cap_binding"
        bond_key = "final_max_bond" if "final_max_bond" in row else "peak_bond"
        parsed.append(
            {
                "seed": int(float(row["seed"])),
                "n_sub": int(float(row["n_sub"])),
                "infidelity_vs_exact": float(row["infidelity_vs_exact"]),
                "distance_to_finest": float(row["distance_to_finest"]) if row["distance_to_finest"] != "" else math.nan,
                "adjacent_refinement_distance": (
                    float(row["adjacent_refinement_distance"])
                    if row.get("adjacent_refinement_distance", "") not in {"", None}
                    else math.nan
                ),
                "final_max_bond": int(float(row[bond_key])),
                "discarded_weight": float(row["discarded_weight"]),
                "norm_drift": float(row["norm_drift"]),
                "cap_reached": row.get(cap_key, "") in {"True", "true", True},
            }
        )
    parsed.sort(key=lambda r: (r["seed"], r["n_sub"]))
    return {
        "n_rows": len(parsed),
        "rows": parsed,
        "note": (
            "Rows are seed resolved; plotted curves use three-seed medians and full ranges. "
            "adjacent_refinement_distance is min_φ ||Ψ_n - e^{iφ} Ψ_{2n}||. "
            "n=1024 is not a converged reference."
        ),
    }


def main() -> int:
    campaign_csv = OUTPUT_DIR / "campaign_rows.csv"
    if not campaign_csv.is_file():
        print(f"Missing {campaign_csv}; run campaign stage first.")
        return 1

    # Remove obsolete pooled table if present.
    obsolete = OUTPUT_DIR / "table_methods.tex"
    if obsolete.is_file():
        obsolete.unlink()
        print(f"Removed obsolete {obsolete}")

    rows = _read_csv(campaign_csv)
    pauli_path = OUTPUT_DIR / "pauli_finite_angle_summary.csv"
    pauli_meta = write_pauli_finite_angle_summary(rows, pauli_path)
    print(f"Wrote {pauli_path}")

    analysis = {
        "n_campaign_rows": len(rows),
        "pauli_finite_angle_summary": pauli_meta,
        "methods": list(METHODS),
        "pauli_gates": list(PAULI_GATES),
        "seeds": list(SEEDS),
        "discarded_weight_note": (
            "Do not compare cumulative discarded weights quantitatively across "
            "methods without noting that methods perform different numbers and "
            "sequences of SVDs."
        ),
    }

    cnot_csv = OUTPUT_DIR / "cnot_rank_rows.csv"
    if cnot_csv.is_file():
        cnot_rows = _read_csv(cnot_csv)
        cnot_path = OUTPUT_DIR / "cnot_rank_summary.csv"
        cnot_meta = write_cnot_rank_summary_csv(cnot_rows, cnot_path)
        analysis["cnot_rank_summary"] = cnot_meta
        print(f"Wrote {cnot_path}")
        # Merge distances from runner summary if present.
        rank_json = OUTPUT_DIR / "cnot_rank_summary.json"
        if rank_json.is_file():
            existing = json.loads(rank_json.read_text(encoding="utf-8"))
            existing["matched_table"] = cnot_meta
            rank_json.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Updated {rank_json}")

    refinement_csv = OUTPUT_DIR / "refinement_rows.csv"
    if refinement_csv.is_file():
        ref = summarize_refinement(_read_csv(refinement_csv))
        (OUTPUT_DIR / "refinement_analysis.json").write_text(
            json.dumps(ref, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {OUTPUT_DIR / 'refinement_analysis.json'}")

    out = OUTPUT_DIR / "analysis_summary.json"
    out.write_text(json.dumps(analysis, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
