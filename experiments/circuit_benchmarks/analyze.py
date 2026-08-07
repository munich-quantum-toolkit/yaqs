# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Aggregate the circuit campaign into auditable tables for plotting.

The resource frontier is selected only from the fixed, preregistered bond-cap
grid.  A candidate is reliable at a target time when every sampled step from
zero through that target exists and has normalized infidelity no larger than
the stated threshold.  Among reliable candidates, the table reports the
smallest *observed peak retained parameter count*, with bond cap used only as a
deterministic tie-breaker.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import operator
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments.circuit_benchmarks.circuits import build_schedule
from experiments.circuit_benchmarks.common import protocol_metadata
from experiments.circuit_benchmarks.config import (
    CAMPAIGN_ID,
    CASE_KEYS,
    CASES,
    CHI_GRID,
    CHI_MAIN,
    FRONTIER_CASE_KEY,
    FRONTIER_STEPS,
    FRONTIER_TARGET_STEPS,
    METHODS,
    N_STEPS,
    OUTPUT_DIR,
    RELIABILITY_THRESHOLD,
    THRESHOLD_SENSITIVITY,
    TIMING_REPEATS,
    time_for_step,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

TASKS_DIR = OUTPUT_DIR / "tasks"
EXACT_DIR = OUTPUT_DIR / "exact"

TRAJECTORY_CSV = OUTPUT_DIR / "trajectory_rows.csv"
FRONTIER_CANDIDATES_CSV = OUTPUT_DIR / "frontier_candidates.csv"
FRONTIER_SELECTED_CSV = OUTPUT_DIR / "frontier_selected.csv"
FRONTIER_SENSITIVITY_CSV = OUTPUT_DIR / "frontier_threshold_sensitivity.csv"
RUNTIME_REPEATS_CSV = OUTPUT_DIR / "runtime_repeats.csv"
RUNTIME_SUMMARY_CSV = OUTPUT_DIR / "runtime_summary.csv"
SCHMIDT_TAILS_CSV = OUTPUT_DIR / "exact_schmidt_tails.csv"
CHECKPOINTS_CSV_GZ = OUTPUT_DIR / "refactorization_checkpoints.csv.gz"
VALIDATION_REPORT = OUTPUT_DIR / "validation_report.json"

TRAJECTORY_FIELDS = [
    "run_family",
    "case",
    "method",
    "chi_max",
    "n_sub",
    "step",
    "time",
    "infidelity_normalized",
    "peak_parameter_count",
    "peak_bond_dim",
    "failed",
    "task_id",
    "task_status",
    "fidelity_normalized",
    "norm_exact",
    "norm_approx",
    "norm_drift",
    "current_parameter_count",
    "current_peak_bond_dim",
    "step_peak_parameter_count",
    "step_peak_bond_dim",
    "step_runtime_instrumented_s",
    "trace_resources",
]

FRONTIER_FIELDS = [
    "target_step",
    "target_time",
    "method",
    "selected_chi_max",
    "achieved_infidelity",
    "max_infidelity_through",
    "peak_parameter_count",
    "peak_bond_dim",
    "missing",
]

RUNTIME_SUMMARY_FIELDS = [
    "target_step",
    "target_time",
    "method",
    "selected_chi_max",
    "median_s",
    "min_s",
    "max_s",
    "missing",
]


def _utc_now() -> str:
    """Return a UTC timestamp for derived artifacts."""
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    """Convert common scientific scalar types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    msg = f"Object of type {type(value).__name__} is not JSON serializable."
    raise TypeError(msg)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write an analysis document atomically."""
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
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    """Write one deterministic CSV table atomically."""
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
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    """Encode structured cells reproducibly."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON task object."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        msg = f"Expected a JSON object in {path}."
        raise TypeError(msg)
    return value


def _tasks(family: str) -> list[dict[str, Any]]:
    """Load all terminal task records in one family."""
    directory = TASKS_DIR / family
    if not directory.is_dir():
        return []
    source_hash = _source_hash()
    return [
        task
        for path in sorted(directory.glob("*.json"))
        if (task := _load_json(path)).get("payload", {}).get("campaign_id") == CAMPAIGN_ID
        and task.get("payload", {}).get("source_hash") == source_hash
    ]


def _source_hash() -> str:
    """Hash the files that determine raw-task scientific semantics."""
    names = ("config.py", "circuits.py", "common.py", "tracing.py", "run.py", "analyze.py")
    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    for name in names:
        path = directory / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _trajectory_tasks() -> list[dict[str, Any]]:
    """Load resolution, main, and frontier trajectory tasks."""
    return [
        task
        for family in ("resolution", "trajectories", "frontier")
        for task in _tasks(family)
    ]


def _spec(task: Mapping[str, Any]) -> dict[str, Any]:
    """Return a trajectory task's frozen specification."""
    value = task.get("payload", {}).get("spec", {})
    return dict(value) if isinstance(value, dict) else {}


def aggregate_trajectory_rows(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten trajectory task samples, retaining explicit failed-task rows."""
    rows: list[dict[str, Any]] = []
    for task in tasks:
        spec = _spec(task)
        task_rows = task.get("rows", [])
        if task.get("status") == "success" and isinstance(task_rows, list):
            for sample in task_rows:
                row = dict(sample)
                row.update({
                    "task_id": task.get("task_id"),
                    "task_status": task.get("status"),
                    "failed": bool(row.get("failed")),
                    "trace_resources": bool(spec.get("trace_resources", False)),
                })
                rows.append(row)
        else:
            rows.append({
                "run_family": spec.get("run_family"),
                "case": spec.get("case"),
                "method": spec.get("method"),
                "chi_max": spec.get("chi_max"),
                "n_sub": spec.get("n_sub"),
                "step": -1,
                "time": "",
                "infidelity_normalized": "",
                "peak_parameter_count": "",
                "peak_bond_dim": "",
                "failed": True,
                "task_id": task.get("task_id"),
                "task_status": task.get("status"),
                "trace_resources": bool(spec.get("trace_resources", False)),
            })
    rows.sort(
        key=lambda row: (
            str(row.get("run_family")),
            str(row.get("case")),
            str(row.get("method")),
            int(row.get("chi_max") or -1),
            int(row.get("n_sub") or -1),
            int(row.get("step") or 0),
            str(row.get("task_id")),
        )
    )
    _atomic_csv(TRAJECTORY_CSV, rows, TRAJECTORY_FIELDS)
    return rows


def _candidate_tasks(tasks: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """Choose the longest successful Ising-2D trajectory for each method/cap."""
    candidates: dict[tuple[str, int], dict[str, Any]] = {}
    for task in tasks:
        spec = _spec(task)
        if (
            task.get("status") != "success"
            or spec.get("case") != FRONTIER_CASE_KEY
            or spec.get("method") not in METHODS
        ):
            continue
        chi = int(spec.get("chi_max", -1))
        if chi not in CHI_GRID:
            continue
        steps = int(spec.get("steps", -1))
        if steps < FRONTIER_STEPS:
            continue
        key = (str(spec["method"]), chi)
        current = candidates.get(key)
        if current is None or int(_spec(current).get("steps", -1)) < steps:
            candidates[key] = dict(task)
    return candidates


def _samples_through(task: Mapping[str, Any], target_step: int) -> list[dict[str, Any]] | None:
    """Return a complete, unique sample sequence through a target."""
    by_step: dict[int, dict[str, Any]] = {}
    for sample in task.get("rows", []):
        step = int(sample.get("step", -1))
        if 0 <= step <= target_step:
            by_step[step] = dict(sample)
    if set(by_step) != set(range(target_step + 1)):
        return None
    return [by_step[step] for step in range(target_step + 1)]


def _frontier_for_threshold(
    tasks: Sequence[Mapping[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return every candidate diagnostic and the minimum-observed-P selections."""
    available = _candidate_tasks(tasks)
    diagnostics: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for target_step in FRONTIER_TARGET_STEPS:
        for method in METHODS:
            reliable: list[dict[str, Any]] = []
            for chi in CHI_GRID:
                task = available.get((method, chi))
                samples = _samples_through(task, target_step) if task is not None else None
                complete = samples is not None
                infidelities = (
                    [float(row["infidelity_normalized"]) for row in samples] if samples else []
                )
                max_infidelity = max(infidelities, default=math.nan)
                endpoint_infidelity = infidelities[-1] if infidelities else math.nan
                peak_p = (
                    max(int(row["peak_parameter_count"]) for row in samples) if samples else None
                )
                peak_chi = max(int(row["peak_bond_dim"]) for row in samples) if samples else None
                is_reliable = bool(complete and max_infidelity <= threshold)
                row = {
                    "threshold": threshold,
                    "target_step": target_step,
                    "target_time": time_for_step(target_step),
                    "method": method,
                    "chi_max": chi,
                    "complete": complete,
                    "reliable": is_reliable,
                    "achieved_infidelity": endpoint_infidelity,
                    "max_infidelity_through": max_infidelity,
                    "peak_parameter_count": peak_p,
                    "peak_bond_dim": peak_chi,
                    "task_id": task.get("task_id") if task else None,
                }
                diagnostics.append(row)
                if is_reliable:
                    reliable.append(row)

            if reliable:
                winner = min(
                    reliable,
                    key=lambda row: (int(row["peak_parameter_count"]), int(row["chi_max"])),
                )
                selected_rows.append({
                    "target_step": target_step,
                    "target_time": time_for_step(target_step),
                    "method": method,
                    "selected_chi_max": winner["chi_max"],
                    "achieved_infidelity": winner["achieved_infidelity"],
                    "max_infidelity_through": winner["max_infidelity_through"],
                    "peak_parameter_count": winner["peak_parameter_count"],
                    "peak_bond_dim": winner["peak_bond_dim"],
                    "missing": False,
                })
            else:
                selected_rows.append({
                    "target_step": target_step,
                    "target_time": time_for_step(target_step),
                    "method": method,
                    "selected_chi_max": "",
                    "achieved_infidelity": "",
                    "max_infidelity_through": "",
                    "peak_parameter_count": "",
                    "peak_bond_dim": "",
                    "missing": True,
                })
    return diagnostics, selected_rows


def aggregate_frontier(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Write the primary frontier and preregistered threshold sensitivity."""
    candidates, selected = _frontier_for_threshold(tasks, RELIABILITY_THRESHOLD)
    candidate_fields = [
        "threshold",
        "target_step",
        "target_time",
        "method",
        "chi_max",
        "complete",
        "reliable",
        "achieved_infidelity",
        "max_infidelity_through",
        "peak_parameter_count",
        "peak_bond_dim",
        "task_id",
    ]
    _atomic_csv(FRONTIER_CANDIDATES_CSV, candidates, candidate_fields)
    _atomic_csv(FRONTIER_SELECTED_CSV, selected, FRONTIER_FIELDS)

    sensitivity: list[dict[str, Any]] = []
    for threshold in THRESHOLD_SENSITIVITY:
        _, threshold_rows = _frontier_for_threshold(tasks, threshold)
        sensitivity.extend({"threshold": threshold, **row} for row in threshold_rows)
    _atomic_csv(
        FRONTIER_SENSITIVITY_CSV,
        sensitivity,
        ["threshold", *FRONTIER_FIELDS],
    )
    return selected


def _timing_key(task: Mapping[str, Any]) -> tuple[int, str, int] | None:
    """Return the selected-configuration key of a timing task."""
    payload = task.get("payload", {})
    try:
        return (
            int(payload["target_step"]),
            str(payload["method"]),
            int(payload["chi_max"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def aggregate_timing(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate three uninstrumented timing repeats for each frontier point."""
    tasks = _tasks("timing")
    repeat_rows: list[dict[str, Any]] = []
    for task in tasks:
        payload = task.get("payload", {})
        repeat = int(payload.get("repeat", -999))
        repeat_rows.append({
            "target_step": payload.get("target_step"),
            "target_time": time_for_step(int(payload["target_step"]))
            if "target_step" in payload
            else "",
            "method": payload.get("method"),
            "selected_chi_max": payload.get("chi_max"),
            "n_sub": payload.get("n_sub"),
            "repeat": repeat,
            "is_warmup": bool(task.get("is_warmup", repeat < 0)),
            "runtime_s": task.get("runtime_s", ""),
            "endpoint_infidelity": task.get("endpoint_metrics", {}).get(
                "infidelity_normalized", ""
            ),
            "status": task.get("status"),
            "task_id": task.get("task_id"),
        })
    repeat_rows.sort(
        key=lambda row: (
            int(row.get("target_step") or -1),
            str(row.get("method")),
            int(row.get("selected_chi_max") or -1),
            int(row.get("repeat") or 0),
        )
    )
    _atomic_csv(
        RUNTIME_REPEATS_CSV,
        repeat_rows,
        [
            "target_step",
            "target_time",
            "method",
            "selected_chi_max",
            "n_sub",
            "repeat",
            "is_warmup",
            "runtime_s",
            "endpoint_infidelity",
            "status",
            "task_id",
        ],
    )

    by_key: dict[tuple[int, str, int], list[float]] = {}
    for task in tasks:
        key = _timing_key(task)
        repeat = int(task.get("payload", {}).get("repeat", -1))
        if key is not None and repeat >= 0 and task.get("status") == "success":
            by_key.setdefault(key, []).append(float(task["runtime_s"]))

    summary: list[dict[str, Any]] = []
    for selection in selected:
        target_step = int(selection["target_step"])
        method = str(selection["method"])
        if bool(selection.get("missing")):
            summary.append({
                "target_step": target_step,
                "target_time": time_for_step(target_step),
                "method": method,
                "selected_chi_max": "",
                "median_s": "",
                "min_s": "",
                "max_s": "",
                "missing": True,
            })
            continue
        chi = int(selection["selected_chi_max"])
        values = sorted(by_key.get((target_step, method, chi), []))
        complete = len(values) == TIMING_REPEATS
        summary.append({
            "target_step": target_step,
            "target_time": time_for_step(target_step),
            "method": method,
            "selected_chi_max": chi,
            "median_s": float(np.median(values)) if complete else "",
            "min_s": min(values) if complete else "",
            "max_s": max(values) if complete else "",
            "missing": not complete,
        })
    _atomic_csv(RUNTIME_SUMMARY_CSV, summary, RUNTIME_SUMMARY_FIELDS)
    return summary


def _schmidt_tail_bound(vector: np.ndarray, n_qubits: int, chi: int) -> tuple[float, int, int]:
    """Return the strongest contiguous-cut infidelity lower bound for rank ``chi``.

    For every MPS cut, the squared Schmidt coefficients discarded above rank
    ``chi`` lower-bound the infidelity of any rank-``chi`` state.  Taking the
    largest such tail over cuts gives the strongest bound available from these
    individual bipartitions.
    """
    state = np.asarray(vector, dtype=np.complex128).reshape(-1)
    norm_sq = float(np.real(np.vdot(state, state)))
    if norm_sq <= 0.0:
        msg = "Cannot compute Schmidt weights for a zero state."
        raise ValueError(msg)
    best_tail = 0.0
    limiting_cut = 1
    rank_at_cut = 1
    for cut in range(1, n_qubits):
        matrix = state.reshape(2 ** (n_qubits - cut), 2**cut)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        weights = np.square(np.abs(singular_values)) / norm_sq
        tail = float(np.sum(weights[chi:])) if chi < len(weights) else 0.0
        tail = min(1.0, max(0.0, tail))
        if tail > best_tail:
            best_tail = tail
            limiting_cut = cut
            rank_at_cut = int(np.count_nonzero(weights > 1e-14))
    return best_tail, limiting_cut, rank_at_cut


def aggregate_schmidt_tails() -> list[dict[str, Any]]:
    """Write exact Schmidt-tail lower bounds for every plotted cap and time."""
    rows: list[dict[str, Any]] = []
    caps = tuple(sorted({*CHI_GRID, CHI_MAIN}))
    for case_key in CASE_KEYS:
        case = CASES[case_key]
        path = EXACT_DIR / f"{case_key}.npy"
        if not path.is_file():
            continue
        exact = np.load(path, mmap_mode="r", allow_pickle=False)
        for step in range(min(N_STEPS, exact.shape[0] - 1) + 1):
            # Compute one SVD per cut, then reuse its weights for every cap.
            state = np.asarray(exact[step], dtype=np.complex128)
            norm_sq = float(np.real(np.vdot(state, state)))
            weights_by_cut: list[tuple[int, np.ndarray]] = []
            for cut in range(1, case.n_qubits):
                matrix = state.reshape(2 ** (case.n_qubits - cut), 2**cut)
                singular_values = np.linalg.svd(matrix, compute_uv=False)
                weights_by_cut.append((cut, np.square(np.abs(singular_values)) / norm_sq))
            for chi in caps:
                tails = [
                    (float(np.sum(weights[chi:])) if chi < len(weights) else 0.0, cut, weights)
                    for cut, weights in weights_by_cut
                ]
                tail, limiting_cut, limiting_weights = max(tails, key=operator.itemgetter(0))
                rows.append({
                    "case": case_key,
                    "step": step,
                    "time": time_for_step(step),
                    "chi_max": chi,
                    "schmidt_infidelity_lower_bound": min(1.0, max(0.0, tail)),
                    "limiting_cut": limiting_cut,
                    "exact_schmidt_rank_at_cut": int(
                        np.count_nonzero(limiting_weights > 1e-14)
                    ),
                })
    _atomic_csv(
        SCHMIDT_TAILS_CSV,
        rows,
        [
            "case",
            "step",
            "time",
            "chi_max",
            "schmidt_infidelity_lower_bound",
            "limiting_cut",
            "exact_schmidt_rank_at_cut",
        ],
    )
    return rows


def aggregate_checkpoints(tasks: Sequence[Mapping[str, Any]]) -> int:
    """Combine compressed raw tracer records without loading them all at once."""
    fields = [
        "task_id",
        "run_family",
        "case",
        "method",
        "chi_max",
        "n_sub",
        "step",
        "gate_index",
        "gate_name",
        "sites",
        "checkpoint",
        "checkpoint_index",
        "checkpoint_in_gate",
        "parameter_count",
        "peak_bond_dim",
        "bond_dimensions",
        "updated_sites",
        "local_gate_name",
    ]
    CHECKPOINTS_CSV_GZ.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=CHECKPOINTS_CSV_GZ.parent,
            prefix=f".{CHECKPOINTS_CSV_GZ.name}.",
            suffix=".tmp",
            delete=False,
        ) as raw_output:
            temporary = Path(raw_output.name)
            with gzip.open(raw_output, mode="wt", encoding="utf-8", newline="") as zipped_output:
                writer = csv.DictWriter(zipped_output, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for task in tasks:
                    path_value = task.get("checkpoint_path")
                    if not path_value:
                        continue
                    path = Path(str(path_value))
                    if not path.is_file():
                        continue
                    with gzip.open(path, mode="rt", encoding="utf-8") as source:
                        for line in source:
                            row = json.loads(line)
                            row["task_id"] = task.get("task_id")
                            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})
                            count += 1
            raw_output.flush()
            os.fsync(raw_output.fileno())
        temporary.replace(CHECKPOINTS_CSV_GZ)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return count


def _complete_main_tasks(tasks: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """Return case/method pairs with complete 30-step main trajectories."""
    complete: set[tuple[str, str]] = set()
    for task in tasks:
        spec = _spec(task)
        if (
            task.get("status") == "success"
            and spec.get("run_family") == "trajectories"
            and int(spec.get("chi_max", -1)) == CHI_MAIN
            and int(spec.get("steps", -1)) == N_STEPS
            and _samples_through(task, N_STEPS) is not None
        ):
            complete.add((str(spec.get("case")), str(spec.get("method"))))
    return complete


def _validation_report(
    tasks: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    timing: Sequence[Mapping[str, Any]],
    *,
    checkpoint_count: int,
) -> dict[str, Any]:
    """Summarize completeness without converting missing data into a pass."""
    exact_checks: dict[str, Any] = {}
    for case_key in CASE_KEYS:
        path = EXACT_DIR / f"{case_key}.npy"
        if path.is_file():
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            exact_checks[case_key] = {
                "present": True,
                "shape": list(array.shape),
                "max_norm_error": float(np.max(np.abs(np.linalg.norm(array, axis=1) - 1.0))),
                "expected_protocol": protocol_metadata(CASES[case_key], build_schedule(CASES[case_key])),
            }
        else:
            exact_checks[case_key] = {"present": False}

    main_complete = _complete_main_tasks(tasks)
    expected_main = {(case, method) for case in CASE_KEYS for method in METHODS}
    frontier_missing = [
        {"target_step": row["target_step"], "method": row["method"]}
        for row in selected
        if bool(row.get("missing"))
    ]
    timing_missing = [
        {"target_step": row["target_step"], "method": row["method"]}
        for row in timing
        if bool(row.get("missing"))
    ]
    failed_tasks = [
        {
            "task_id": task.get("task_id"),
            "family": task.get("payload", {}).get("family"),
            "spec": _spec(task),
            "error_type": task.get("error_type"),
            "error_message": task.get("error_message"),
        }
        for task in tasks
        if task.get("status") != "success"
    ]
    resolution_path = OUTPUT_DIR / "resolution_summary.json"
    resolution = _load_json(resolution_path) if resolution_path.is_file() else {"confirmed": False}
    complete = bool(
        all(item.get("present") for item in exact_checks.values())
        and main_complete == expected_main
        and resolution.get("confirmed")
        and not frontier_missing
        and not timing_missing
        and not failed_tasks
    )
    return {
        "created_utc": _utc_now(),
        "complete": complete,
        "publication_ready": complete,
        "publication_ready_note": (
            "Completeness is necessary, not sufficient: inspect the trajectories, threshold "
            "sensitivity, numerical-resolution report, and figure before publication."
        ),
        "exact": exact_checks,
        "resolution_confirmed": bool(resolution.get("confirmed")),
        "main_complete_count": len(main_complete),
        "main_expected_count": len(expected_main),
        "main_missing": sorted([list(item) for item in expected_main - main_complete]),
        "frontier_missing": frontier_missing,
        "timing_missing": timing_missing,
        "failed_tasks": failed_tasks,
        "raw_checkpoint_count": checkpoint_count,
    }


def aggregate_results(*, compute_schmidt: bool = True) -> dict[str, Any]:
    """Regenerate every derived table from immutable raw task records."""
    tasks = _trajectory_tasks()
    trajectory_rows = aggregate_trajectory_rows(tasks)
    selected = aggregate_frontier(tasks)
    timing = aggregate_timing(selected)
    schmidt_rows = aggregate_schmidt_tails() if compute_schmidt else []
    checkpoint_count = aggregate_checkpoints(tasks)
    report = _validation_report(tasks, selected, timing, checkpoint_count=checkpoint_count)
    _atomic_json(VALIDATION_REPORT, report)
    summary = {
        "created_utc": _utc_now(),
        "trajectory_rows": len(trajectory_rows),
        "frontier_selections": len(selected),
        "timing_rows": len(timing),
        "schmidt_tail_rows": len(schmidt_rows),
        "checkpoint_rows": checkpoint_count,
        "validation_complete": report["complete"],
    }
    _atomic_json(OUTPUT_DIR / "analysis_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Aggregate existing raw tasks; no simulation is performed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-schmidt",
        action="store_true",
        help="Skip the exact Schmidt-tail table during a quick intermediate aggregation.",
    )
    args = parser.parse_args(argv)
    aggregate_results(compute_schmidt=not args.skip_schmidt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
