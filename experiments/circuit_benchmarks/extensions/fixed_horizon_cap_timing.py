# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Time every tested bond cap in the isolated fixed-horizon sweep.

The accuracy sweep in ``fixed_horizon_refinement`` combines the frozen coarse
grid with targeted points around each tolerance crossing.  This extension
uses the same production timing kernel to measure every one of those caps
through ``n=15``.  Timing tasks are content addressed, so compatible results
from earlier campaigns are reused without changing the frozen aggregates.

Example:
    uv run python -m experiments.circuit_benchmarks.extensions.fixed_horizon_cap_timing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from experiments.circuit_benchmarks import run as benchmark_run
from experiments.circuit_benchmarks.config import (
    FRONTIER_STEPS,
    METHODS,
    TIMING_REPEATS,
    TIMING_WARMUPS,
    Method,
)
from experiments.circuit_benchmarks.extensions.fixed_horizon_refinement import (
    COMBINED_PATH,
    REFINEMENT_DIR,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

TARGET_STEP = FRONTIER_STEPS
TIMING_ROWS_PATH = REFINEMENT_DIR / "cap_timing_rows.csv"
TIMING_SUMMARY_PATH = REFINEMENT_DIR / "cap_timing_summary.csv"
MANIFEST_PATH = REFINEMENT_DIR / "cap_timing_manifest.json"

TIMING_ROW_FIELDS = (
    "method",
    "chi_max",
    "target_step",
    "repeat",
    "runtime_s",
    "endpoint_infidelity",
    "task_id",
)
TIMING_SUMMARY_FIELDS = (
    "method",
    "chi_max",
    "target_step",
    "median_s",
    "min_s",
    "max_s",
    "repeats",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one required CSV table."""
    if not path.is_file():
        msg = f"Missing required input {path}."
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    """Write a CSV atomically with a fixed schema."""
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a JSON object atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def cap_specs_from_sweep(rows: Sequence[Mapping[str, str]]) -> list[tuple[Method, int, float]]:
    """Return unique method, cap, and endpoint tuples in manuscript order."""
    required = {"method", "chi_max", "target_step", "endpoint_infidelity"}
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required.difference(rows[0] if rows else {}))
        msg = f"combined_cap_sweep.csv is empty or missing fields: {', '.join(missing)}."
        raise ValueError(msg)

    specs: list[tuple[Method, int, float]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if int(row["target_step"]) != TARGET_STEP:
            continue
        method_text = str(row["method"])
        if method_text not in METHODS:
            msg = f"Unknown method {method_text!r}."
            raise ValueError(msg)
        cap = int(row["chi_max"])
        endpoint = float(row["endpoint_infidelity"])
        if cap < 1 or not math.isfinite(endpoint) or endpoint < 0.0:
            msg = f"Invalid cap-sweep row for {method_text}/chi{cap}."
            raise ValueError(msg)
        key = (method_text, cap)
        if key in seen:
            msg = f"Duplicate cap-sweep row for {method_text}/chi{cap}."
            raise RuntimeError(msg)
        seen.add(key)
        method: Method = method_text  # type: ignore[assignment]
        specs.append((method, cap, endpoint))

    order = {method: index for index, method in enumerate(METHODS)}
    return sorted(specs, key=lambda spec: (order[spec[0]], spec[1]))


def summarize_timing_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate measured repeats into min, median, and max runtimes."""
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        method = str(row["method"])
        cap = int(row["chi_max"])
        runtime = float(row["runtime_s"])
        if method not in METHODS or not math.isfinite(runtime) or runtime <= 0.0:
            msg = f"Invalid timing row for {method}/chi{cap}."
            raise ValueError(msg)
        grouped.setdefault((method, cap), []).append(runtime)

    order = {method: index for index, method in enumerate(METHODS)}
    summaries: list[dict[str, Any]] = []
    for (method, cap), values in sorted(
        grouped.items(), key=lambda item: (order[item[0][0]], item[0][1])
    ):
        if len(values) != TIMING_REPEATS:
            msg = f"Expected {TIMING_REPEATS} measured timings for {method}/chi{cap}."
            raise RuntimeError(msg)
        summaries.append(
            {
                "method": method,
                "chi_max": cap,
                "target_step": TARGET_STEP,
                "median_s": statistics.median(values),
                "min_s": min(values),
                "max_s": max(values),
                "repeats": len(values),
            }
        )
    return summaries


def _method_n_sub(method: Method) -> int:
    """Return the frozen production substep count for one method."""
    return benchmark_run.selected_tdvp_substeps() if method == "gate_local_2tdvp" else 1


def run_cap_timing_sweep(
    *,
    resume: bool = True,
    retry_failed: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Time every cap through the fixed horizon and write raw and summary tables."""
    specs = cap_specs_from_sweep(_read_csv(COMBINED_PATH))
    timing_rows: list[dict[str, Any]] = []
    task_ids: list[str] = []
    repeat_ids = [-(index + 1) for index in range(TIMING_WARMUPS)] + list(
        range(TIMING_REPEATS)
    )
    for method, cap, expected_endpoint in specs:
        for repeat in repeat_ids:
            task = benchmark_run._ensure_timing_task(  # noqa: SLF001
                method=method,
                chi=cap,
                n_sub=_method_n_sub(method),
                target_step=TARGET_STEP,
                repeat=repeat,
                resume=resume,
                retry_failed=retry_failed,
            )
            task_ids.append(str(task.get("task_id", "")))
            if task.get("status") != "success":
                msg = f"Timing task failed for {method}/chi{cap}/repeat{repeat}."
                raise RuntimeError(msg)
            endpoint = float(task["endpoint_metrics"]["infidelity_normalized"])
            if not math.isclose(endpoint, expected_endpoint, rel_tol=0.0, abs_tol=1e-10):
                msg = (
                    f"Timing endpoint mismatch for {method}/chi{cap}: "
                    f"{endpoint:.16g} != {expected_endpoint:.16g}."
                )
                raise RuntimeError(msg)
            if repeat >= 0:
                timing_rows.append(
                    {
                        "method": method,
                        "chi_max": cap,
                        "target_step": TARGET_STEP,
                        "repeat": repeat,
                        "runtime_s": float(task["runtime_s"]),
                        "endpoint_infidelity": endpoint,
                        "task_id": task["task_id"],
                    }
                )
        measured = [
            float(row["runtime_s"])
            for row in timing_rows
            if row["method"] == method and int(row["chi_max"]) == cap
        ]
        print(
            f"{method} chi={cap}: median={statistics.median(measured):.6g} s "
            f"[{min(measured):.6g}, {max(measured):.6g}]"
        )

    summaries = summarize_timing_rows(timing_rows)
    _atomic_csv(TIMING_ROWS_PATH, timing_rows, TIMING_ROW_FIELDS)
    _atomic_csv(TIMING_SUMMARY_PATH, summaries, TIMING_SUMMARY_FIELDS)
    _write_manifest(specs=specs, task_ids=task_ids)
    print(f"Wrote {TIMING_ROWS_PATH}")
    print(f"Wrote {TIMING_SUMMARY_PATH}")
    return timing_rows, summaries


def _write_manifest(
    *,
    specs: Sequence[tuple[Method, int, float]],
    task_ids: Sequence[str],
) -> None:
    """Write the timing-sweep protocol and provenance."""
    digest = hashlib.sha256(COMBINED_PATH.read_bytes()).hexdigest()
    _atomic_json(
        MANIFEST_PATH,
        {
            "campaign_id": "circuit_fixed_horizon_cap_timing_v1",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "target_step": TARGET_STEP,
            "timing_scope": "MPS gate application through the complete fixed-horizon circuit",
            "timing_warmups": TIMING_WARMUPS,
            "timing_repeats": TIMING_REPEATS,
            "threads": 1,
            "caps": {
                method: [cap for candidate, cap, _endpoint in specs if candidate == method]
                for method in METHODS
            },
            "combined_cap_sweep": str(COMBINED_PATH),
            "combined_cap_sweep_sha256": digest,
            "timing_task_ids": sorted(task_id for task_id in task_ids if task_id),
            "hardware": {
                "cpu_model": benchmark_run._cpu_model(),  # noqa: SLF001
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
            "artifacts": {
                "timing_rows": str(TIMING_ROWS_PATH),
                "timing_summary": str(TIMING_SUMMARY_PATH),
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    """Run the resumable full-cap timing sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    run_cap_timing_sweep(
        resume=not args.no_resume,
        retry_failed=args.retry_failed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
