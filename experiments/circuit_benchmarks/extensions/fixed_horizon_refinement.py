# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Refine the fixed-horizon accuracy--resource comparison near its crossings.

The frozen publication campaign uses a deliberately coarse common cap grid.
This isolated extension adds a small method-specific grid only inside each
observed fail/pass bracket at ``n=15``. It reuses the production trajectory
and timing kernels without changing the frozen global grid or its aggregates.

Examples:
    uv run python -m experiments.circuit_benchmarks.extensions.fixed_horizon_refinement --stage accuracy
    uv run python -m experiments.circuit_benchmarks.extensions.fixed_horizon_refinement --stage select
    uv run python -m experiments.circuit_benchmarks.extensions.fixed_horizon_refinement --stage timing
    uv run python -m experiments.circuit_benchmarks.extensions.fixed_horizon_refinement --stage all
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
    CASES,
    FRONTIER_CASE_KEY,
    FRONTIER_STEPS,
    METHODS,
    OUTPUT_DIR,
    RELIABILITY_THRESHOLD,
    TIMING_REPEATS,
    TIMING_WARMUPS,
    Method,
    time_for_step,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

CAMPAIGN_ID = "circuit_fixed_horizon_refinement_v1"
TARGET_STEP = FRONTIER_STEPS
MPS_ENTRY_BYTES = 16
REFINEMENT_CAPS: dict[Method, tuple[int, ...]] = {
    "gate_local_2tdvp": (26, 28, 30),
    "mpo_contract_compress": (80, 96, 112),
    "tebd_swap": (144, 160, 176),
}

REFINEMENT_DIR = OUTPUT_DIR / "fixed_horizon_refinement"
ACCURACY_PATH = REFINEMENT_DIR / "refined_accuracy.csv"
COMBINED_PATH = REFINEMENT_DIR / "combined_cap_sweep.csv"
SELECTED_PATH = REFINEMENT_DIR / "selected.csv"
TIMING_ROWS_PATH = REFINEMENT_DIR / "timing_rows.csv"
TIMING_SUMMARY_PATH = REFINEMENT_DIR / "timing_summary.csv"
MANIFEST_PATH = REFINEMENT_DIR / "manifest.json"
COARSE_PATH = OUTPUT_DIR / "frontier_candidates.csv"

ACCURACY_FIELDS = (
    "method",
    "chi_max",
    "source",
    "task_id",
    "target_step",
    "target_time",
    "max_infidelity_through",
    "endpoint_infidelity",
    "reliable",
    "peak_parameter_count",
    "peak_mps_bytes",
    "peak_bond_dim",
)
COMBINED_FIELDS = (*ACCURACY_FIELDS, "selected", "last_fail")
SELECTED_FIELDS = (
    "method",
    "selected_chi_max",
    "task_id",
    "target_step",
    "target_time",
    "max_infidelity_through",
    "endpoint_infidelity",
    "peak_parameter_count",
    "peak_mps_bytes",
    "peak_bond_dim",
    "last_fail_chi_max",
    "last_fail_max_infidelity_through",
    "last_fail_peak_parameter_count",
    "last_fail_peak_mps_bytes",
)
TIMING_ROW_FIELDS = (
    "method",
    "selected_chi_max",
    "target_step",
    "repeat",
    "runtime_s",
    "endpoint_infidelity",
    "task_id",
)
TIMING_SUMMARY_FIELDS = (
    "method",
    "selected_chi_max",
    "target_step",
    "median_s",
    "min_s",
    "max_s",
    "repeats",
)


def utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _true(value: object) -> bool:
    """Return whether a serialized flag is true."""
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one required CSV table."""
    if not path.is_file():
        msg = f"Missing required input {path}."
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    """Atomically write rows using a fixed schema."""
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
    """Atomically write one JSON object."""
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


def _file_digest(path: Path) -> str:
    """Return a SHA-256 file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _method_n_sub(method: Method) -> int:
    """Return the frozen production substep count for one method."""
    selected = benchmark_run.selected_tdvp_substeps()
    return selected if method == "gate_local_2tdvp" else 1


def _accuracy_row(task: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce one successful traced task to its fixed-horizon resources."""
    if task.get("status") != "success":
        msg = f"Accuracy task {task.get('task_id', '<unknown>')} failed."
        raise RuntimeError(msg)
    payload = task.get("payload", {})
    spec = payload.get("spec", {})
    method = str(spec.get("method", ""))
    if method not in METHODS:
        msg = f"Unknown accuracy-task method {method!r}."
        raise RuntimeError(msg)
    rows = task.get("rows", [])
    if not isinstance(rows, list):
        msg = f"Accuracy task {task.get('task_id', '<unknown>')} has invalid rows."
        raise TypeError(msg)
    indexed = {int(row["step"]): row for row in rows}
    if sorted(indexed) != list(range(TARGET_STEP + 1)):
        msg = f"Accuracy task {task.get('task_id', '<unknown>')} is incomplete through n={TARGET_STEP}."
        raise RuntimeError(msg)
    through = [indexed[step] for step in range(TARGET_STEP + 1)]
    infidelities = [float(row["infidelity_normalized"]) for row in through]
    parameters = [int(row["peak_parameter_count"]) for row in through]
    peak_bonds = [int(row["peak_bond_dim"]) for row in through]
    if not all(math.isfinite(value) for value in (*infidelities, *parameters, *peak_bonds)):
        msg = f"Nonfinite accuracy data in task {task.get('task_id', '<unknown>')}."
        raise RuntimeError(msg)
    peak_parameters = max(parameters)
    maximum = max(infidelities)
    return {
        "method": method,
        "chi_max": int(spec["chi_max"]),
        "source": "refined",
        "task_id": str(task["task_id"]),
        "target_step": TARGET_STEP,
        "target_time": time_for_step(TARGET_STEP),
        "max_infidelity_through": maximum,
        "endpoint_infidelity": infidelities[-1],
        "reliable": maximum <= RELIABILITY_THRESHOLD,
        "peak_parameter_count": peak_parameters,
        "peak_mps_bytes": MPS_ENTRY_BYTES * peak_parameters,
        "peak_bond_dim": max(peak_bonds),
    }


def run_accuracy_stage(
    *,
    resume: bool = True,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """Run or reuse the nine targeted accuracy trajectories."""
    benchmark_run.ensure_exact_reference(
        CASES[FRONTIER_CASE_KEY],
        resume=True,
        retry_failed=retry_failed,
    )
    tasks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for cap in REFINEMENT_CAPS[method]:
            spec = benchmark_run.TrajectorySpec(
                run_family="frontier",
                case=FRONTIER_CASE_KEY,
                method=method,
                chi_max=cap,
                n_sub=_method_n_sub(method),
                steps=TARGET_STEP,
            )
            task = benchmark_run.ensure_trajectory_task(
                spec,
                resume=resume,
                retry_failed=retry_failed,
            )
            tasks.append(task)
            rows.append(_accuracy_row(task))
            print(
                f"{method} chi={cap}: E*={rows[-1]['max_infidelity_through']:.6g}, "
                f"storage={rows[-1]['peak_mps_bytes']} B"
            )
    _atomic_csv(ACCURACY_PATH, rows, ACCURACY_FIELDS)
    _write_manifest(stage="accuracy", accuracy_tasks=tasks)
    print(f"Wrote {ACCURACY_PATH}")
    return rows


def _coarse_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Return the complete frozen candidates at the fixed horizon."""
    converted: list[dict[str, Any]] = []
    for row in rows:
        if int(row["target_step"]) != TARGET_STEP or not _true(row["complete"]):
            continue
        method = str(row["method"])
        if method not in METHODS:
            continue
        parameters = int(row["peak_parameter_count"])
        converted.append(
            {
                "method": method,
                "chi_max": int(row["chi_max"]),
                "source": "coarse",
                "task_id": row["task_id"],
                "target_step": TARGET_STEP,
                "target_time": float(row["target_time"]),
                "max_infidelity_through": float(row["max_infidelity_through"]),
                "endpoint_infidelity": float(row["achieved_infidelity"]),
                "reliable": _true(row["reliable"]),
                "peak_parameter_count": parameters,
                "peak_mps_bytes": MPS_ENTRY_BYTES * parameters,
                "peak_bond_dim": int(row["peak_bond_dim"]),
            }
        )
    return converted


def combine_and_select(
    coarse_rows: Sequence[Mapping[str, str]],
    refined_rows: Sequence[Mapping[str, str] | Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge both grids and select the minimum-storage reliable point."""
    combined = _coarse_rows(coarse_rows)
    for source in refined_rows:
        parameters = int(source["peak_parameter_count"])
        maximum = float(source["max_infidelity_through"])
        combined.append(
            {
                "method": str(source["method"]),
                "chi_max": int(source["chi_max"]),
                "source": "refined",
                "task_id": str(source["task_id"]),
                "target_step": TARGET_STEP,
                "target_time": float(source["target_time"]),
                "max_infidelity_through": maximum,
                "endpoint_infidelity": float(source["endpoint_infidelity"]),
                "reliable": _true(source["reliable"]),
                "peak_parameter_count": parameters,
                "peak_mps_bytes": int(source.get("peak_mps_bytes", MPS_ENTRY_BYTES * parameters)),
                "peak_bond_dim": int(source["peak_bond_dim"]),
            }
        )

    keys: set[tuple[str, int]] = set()
    for row in combined:
        key = (str(row["method"]), int(row["chi_max"]))
        if key in keys:
            msg = f"Duplicate combined cap point {key}."
            raise RuntimeError(msg)
        keys.add(key)
        if bool(row["reliable"]) != (
            float(row["max_infidelity_through"]) <= RELIABILITY_THRESHOLD
        ):
            msg = f"Inconsistent reliability classification for {key}."
            raise RuntimeError(msg)

    selected_rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = sorted(
            (row for row in combined if row["method"] == method),
            key=lambda row: int(row["chi_max"]),
        )
        reliable = [row for row in method_rows if bool(row["reliable"])]
        if not reliable:
            msg = f"No reliable combined point for {method}."
            raise RuntimeError(msg)
        selected = min(
            reliable,
            key=lambda row: (int(row["peak_mps_bytes"]), int(row["chi_max"])),
        )
        failing = [
            row
            for row in method_rows
            if int(row["chi_max"]) < int(selected["chi_max"]) and not bool(row["reliable"])
        ]
        if not failing:
            msg = f"No failing point below the selected cap for {method}."
            raise RuntimeError(msg)
        last_fail = max(failing, key=lambda row: int(row["chi_max"]))
        selected_rows.append(
            {
                "method": method,
                "selected_chi_max": int(selected["chi_max"]),
                "task_id": selected["task_id"],
                "target_step": TARGET_STEP,
                "target_time": selected["target_time"],
                "max_infidelity_through": selected["max_infidelity_through"],
                "endpoint_infidelity": selected["endpoint_infidelity"],
                "peak_parameter_count": selected["peak_parameter_count"],
                "peak_mps_bytes": selected["peak_mps_bytes"],
                "peak_bond_dim": selected["peak_bond_dim"],
                "last_fail_chi_max": last_fail["chi_max"],
                "last_fail_max_infidelity_through": last_fail["max_infidelity_through"],
                "last_fail_peak_parameter_count": last_fail["peak_parameter_count"],
                "last_fail_peak_mps_bytes": last_fail["peak_mps_bytes"],
            }
        )
        for row in method_rows:
            row["selected"] = row is selected
            row["last_fail"] = row is last_fail

    method_order = {method: index for index, method in enumerate(METHODS)}
    combined.sort(key=lambda row: (method_order[str(row["method"])], int(row["chi_max"])))
    return combined, selected_rows


def run_select_stage() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge the frozen and refined cap grids and write their selections."""
    coarse = _read_csv(COARSE_PATH)
    refined = _read_csv(ACCURACY_PATH)
    combined, selected = combine_and_select(coarse, refined)
    _atomic_csv(COMBINED_PATH, combined, COMBINED_FIELDS)
    _atomic_csv(SELECTED_PATH, selected, SELECTED_FIELDS)
    _write_manifest(stage="select", selected=selected)
    print(f"Wrote {COMBINED_PATH}")
    print(f"Wrote {SELECTED_PATH}")
    for row in selected:
        print(
            f"selected {row['method']}: chi={row['selected_chi_max']}, "
            f"last fail={row['last_fail_chi_max']}, E*={float(row['max_infidelity_through']):.6g}"
        )
    return combined, selected


def run_timing_stage(
    *,
    resume: bool = True,
    retry_failed: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Time the three selected refined-grid configurations."""
    selected = _read_csv(SELECTED_PATH)
    timing_rows: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    for row in selected:
        method_text = str(row["method"])
        if method_text not in METHODS:
            msg = f"Unknown selected method {method_text!r}."
            raise RuntimeError(msg)
        method: Method = method_text  # type: ignore[assignment]
        cap = int(row["selected_chi_max"])
        expected_endpoint = float(row["endpoint_infidelity"])
        n_sub = _method_n_sub(method)
        repeat_ids = [-(index + 1) for index in range(TIMING_WARMUPS)] + list(
            range(TIMING_REPEATS)
        )
        for repeat in repeat_ids:
            task = benchmark_run._ensure_timing_task(  # noqa: SLF001
                method=method,
                chi=cap,
                n_sub=n_sub,
                target_step=TARGET_STEP,
                repeat=repeat,
                resume=resume,
                retry_failed=retry_failed,
            )
            all_tasks.append(task)
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
                        "selected_chi_max": cap,
                        "target_step": TARGET_STEP,
                        "repeat": repeat,
                        "runtime_s": float(task["runtime_s"]),
                        "endpoint_infidelity": endpoint,
                        "task_id": task["task_id"],
                    }
                )

    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in timing_rows if row["method"] == method]
        if len(method_rows) != TIMING_REPEATS:
            msg = f"Expected {TIMING_REPEATS} measured timings for {method}."
            raise RuntimeError(msg)
        values = [float(row["runtime_s"]) for row in method_rows]
        summaries.append(
            {
                "method": method,
                "selected_chi_max": int(method_rows[0]["selected_chi_max"]),
                "target_step": TARGET_STEP,
                "median_s": statistics.median(values),
                "min_s": min(values),
                "max_s": max(values),
                "repeats": len(values),
            }
        )
    _atomic_csv(TIMING_ROWS_PATH, timing_rows, TIMING_ROW_FIELDS)
    _atomic_csv(TIMING_SUMMARY_PATH, summaries, TIMING_SUMMARY_FIELDS)
    _write_manifest(stage="timing", timing_tasks=all_tasks, selected=selected)
    print(f"Wrote {TIMING_ROWS_PATH}")
    print(f"Wrote {TIMING_SUMMARY_PATH}")
    return timing_rows, summaries


def _write_manifest(
    *,
    stage: str,
    accuracy_tasks: Sequence[Mapping[str, Any]] = (),
    timing_tasks: Sequence[Mapping[str, Any]] = (),
    selected: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Write concise provenance for this isolated refinement."""
    existing: dict[str, Any] = {}
    if MANIFEST_PATH.is_file():
        with MANIFEST_PATH.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            existing = loaded
    accuracy_ids = {
        *existing.get("accuracy_task_ids", []),
        *(str(task.get("task_id", "")) for task in accuracy_tasks),
    }
    timing_ids = {
        *existing.get("timing_task_ids", []),
        *(str(task.get("task_id", "")) for task in timing_tasks),
    }
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "updated_utc": utc_now(),
        "last_stage": stage,
        "target_step": TARGET_STEP,
        "target_time": time_for_step(TARGET_STEP),
        "case": FRONTIER_CASE_KEY,
        "threshold": RELIABILITY_THRESHOLD,
        "refinement_caps": REFINEMENT_CAPS,
        "selection_rule": (
            "Choose the tested reliable trajectory with the smallest observed peak MPS tensor "
            "storage; report the largest tested failing cap below it as the resolution bracket."
        ),
        "coarse_input_sha256": _file_digest(COARSE_PATH),
        "mps_entry_bytes": MPS_ENTRY_BYTES,
        "accuracy_task_ids": sorted(task_id for task_id in accuracy_ids if task_id),
        "timing_task_ids": sorted(task_id for task_id in timing_ids if task_id),
        "selected": list(selected) if selected else existing.get("selected", []),
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
            "combined_cap_sweep": str(COMBINED_PATH.relative_to(benchmark_run.REPO_ROOT)),
            "selected": str(SELECTED_PATH.relative_to(benchmark_run.REPO_ROOT)),
        },
    }
    _atomic_json(MANIFEST_PATH, manifest)


def main(argv: list[str] | None = None) -> int:
    """Run one or all isolated refinement stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("accuracy", "select", "timing", "all"),
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    resume = not args.no_resume

    if args.stage in {"accuracy", "all"}:
        run_accuracy_stage(resume=resume, retry_failed=args.retry_failed)
    if args.stage in {"select", "all"}:
        run_select_stage()
    if args.stage in {"timing", "all"}:
        run_timing_stage(resume=resume, retry_failed=args.retry_failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
