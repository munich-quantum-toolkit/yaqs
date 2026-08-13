#!/usr/bin/env python3
"""Create compact CSV and Markdown summaries from the L=16 benchmark JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def scientific(value: float | None) -> str:
    return "-" if value is None else f"{value:.4e}"


def main() -> None:
    payload = json.loads((HERE / "raw_results.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for model, model_result in payload["models"].items():
        for dt, run in model_result["runs"].items():
            for method, result in run["methods"].items():
                rows.append(
                    {
                        "model": model,
                        "dt": dt,
                        "method": method,
                        "runtime_median_seconds": result["runtime_median_seconds"],
                        "runtime_min_seconds": result["runtime_min_seconds"],
                        "runtime_max_seconds": result["runtime_max_seconds"],
                        "tdvp_over_bug_runtime": run["tdvp_over_bug_runtime"],
                        "phase_aligned_state_error": result.get("phase_aligned_state_error"),
                        "infidelity": result.get("infidelity"),
                        "max_abs_z_error": result.get("max_abs_z_error"),
                        "energy_abs_error": result.get("energy_abs_error"),
                        "norm": result.get("norm"),
                        "max_chi": result.get("max_chi"),
                        "krylov_calls": result.get("krylov_calls"),
                        "krylov_operator_applications": result.get("krylov_operator_applications"),
                        "final_bond_profile": ";".join(map(str, result.get("final_bond_profile", []))),
                    }
                )
    with (HERE / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Corrected matched L=16 BUG versus 2TDVP benchmark",
        "",
        "All rows use `min_keep=2`, relative discarded weight `1e-12`, "
        "`chi_max=512`, a shared deterministically padded `chi=4` initial MPS, "
        "and the shared pure-NumPy matrix-free adaptive Lanczos implementation "
        "(maximum dimension 25, tolerance `1e-12`). Timings are medians of three "
        "warmed runs and exclude setup, diagnostics, and exact references.",
        "",
    ]
    for model in payload["models"]:
        title = "TFIM" if model == "tfim" else "periodic Haldane-Shastry"
        lines.extend(
            [
                f"## {title}",
                "",
                "| dt | BUG time | 2TDVP time | TDVP/BUG | BUG infid. | 2TDVP infid. | BUG max chi | 2TDVP max chi |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for dt, run in payload["models"][model]["runs"].items():
            bug = run["methods"]["bug"]
            tdvp = run["methods"]["2tdvp"]
            lines.append(
                f"| {dt} | {bug['runtime_median_seconds']:.3f} s | "
                f"{tdvp['runtime_median_seconds']:.3f} s | {run['tdvp_over_bug_runtime']:.3f} | "
                f"{scientific(bug.get('infidelity'))} | {scientific(tdvp.get('infidelity'))} | "
                f"{bug.get('max_chi', '-')} | {tdvp.get('max_chi', '-')} |"
            )
        lines.append("")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
