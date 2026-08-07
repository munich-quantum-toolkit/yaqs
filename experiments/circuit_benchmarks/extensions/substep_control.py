# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Run the selected-cap TDVP substep control quoted in the manuscript."""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.circuit_benchmarks import run as benchmark_run
from experiments.circuit_benchmarks.config import (
    CASES,
    FRONTIER_CASE_KEY,
    FRONTIER_STEPS,
    KRYLOV_TOL,
    OUTPUT_DIR,
    SVD_THRESHOLD,
)

CHI_MAX = 28
SUBSTEPS = (2, 4)
OUTPUT_PATH = OUTPUT_DIR / "fixed_horizon_refinement" / "substep_control.csv"
FIELDS = (
    "case",
    "method",
    "chi_max",
    "n_sub",
    "target_step",
    "max_infidelity_through",
    "endpoint_infidelity",
    "peak_parameter_count",
    "peak_bond_dim",
    "svd_threshold",
    "krylov_tol",
)


def summarize_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce one complete task to the quantities compared in the text."""
    if task.get("status") != "success":
        msg = f"Substep-control task did not succeed: {task.get('task_id', 'unknown')}."
        raise RuntimeError(msg)
    payload = task.get("payload")
    if not isinstance(payload, Mapping):
        msg = "Substep-control task has no payload."
        raise TypeError(msg)
    spec = payload.get("spec")
    if not isinstance(spec, Mapping):
        msg = "Substep-control task has no trajectory specification."
        raise TypeError(msg)
    raw_rows = task.get("rows")
    if not isinstance(raw_rows, Sequence):
        msg = "Substep-control task has no trajectory rows."
        raise TypeError(msg)
    rows = [row for row in raw_rows if isinstance(row, Mapping) and int(row["step"]) <= FRONTIER_STEPS]
    if [int(row["step"]) for row in rows] != list(range(FRONTIER_STEPS + 1)):
        msg = "Substep-control task does not contain every step through the target."
        raise RuntimeError(msg)
    return {
        "case": spec["case"],
        "method": spec["method"],
        "chi_max": int(spec["chi_max"]),
        "n_sub": int(spec["n_sub"]),
        "target_step": FRONTIER_STEPS,
        "max_infidelity_through": max(float(row["infidelity_normalized"]) for row in rows),
        "endpoint_infidelity": float(rows[-1]["infidelity_normalized"]),
        "peak_parameter_count": max(int(row["peak_parameter_count"]) for row in rows),
        "peak_bond_dim": max(int(row["peak_bond_dim"]) for row in rows),
        "svd_threshold": SVD_THRESHOLD,
        "krylov_tol": KRYLOV_TOL,
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write the compact control table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(*, resume: bool = True, retry_failed: bool = False) -> list[dict[str, Any]]:
    """Run both matched TDVP trajectories and write their compact comparison."""
    benchmark_run.ensure_exact_reference(
        CASES[FRONTIER_CASE_KEY],
        resume=True,
        retry_failed=retry_failed,
    )
    rows = []
    for n_sub in SUBSTEPS:
        spec = benchmark_run.TrajectorySpec(
            run_family="frontier",
            case=FRONTIER_CASE_KEY,
            method="gate_local_2tdvp",
            chi_max=CHI_MAX,
            n_sub=n_sub,
            steps=FRONTIER_STEPS,
        )
        task = benchmark_run.ensure_trajectory_task(
            spec,
            resume=resume,
            retry_failed=retry_failed,
        )
        rows.append(summarize_task(task))
    write_rows(OUTPUT_PATH, rows)
    print(f"Wrote {OUTPUT_PATH}")
    return rows


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    run(resume=not args.no_resume, retry_failed=args.retry_failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
