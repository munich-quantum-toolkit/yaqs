# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Measure cumulative fixed-cap update time to the frozen adaptive endpoints.

Each circuit/method pair receives one complete, unrecorded warm-up followed by
three isolated measured trajectories. Schedule construction, MPS
initialization, dense references, endpoint validation, and all diagnostics are
outside the timer. Only production MPS gate application is timed.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import os
import tempfile

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "mqt-yaqs-numba"))

import argparse
import csv
import functools
import gc
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_limits

from experiments.circuit_benchmarks.circuits import build_schedule
from experiments.circuit_benchmarks.common import (
    apply_dense_step,
    apply_mps_step,
    compile_schedule,
    digital_params,
    initial_mps,
    initial_vector,
    normalized_state_fidelity,
)
from experiments.circuit_benchmarks.config import CASES, METHODS
from experiments.circuit_benchmarks.run import _cpu_model, _git_metadata, _package_versions

from .config import CASE_ORDER, CHI_CAP, OUTPUT_DIR
from .run import _atomic_csv, _atomic_json, _n_sub, _utc_now

CAMPAIGN_ID = "circuit-fixed-endpoint-timing-v1"
REPEATS = 3
ENDPOINT_TOLERANCE = 1e-10

TIMING_DIR = OUTPUT_DIR / "timing"
TASK_DIR = TIMING_DIR / "tasks"
WARMUP_DIR = TIMING_DIR / "warmups"
ROWS_PATH = TIMING_DIR / "timing_rows.csv"
SUMMARY_PATH = TIMING_DIR / "timing_summary.csv"
MANIFEST_PATH = TIMING_DIR / "manifest.json"
ADAPTIVE_ROWS_PATH = OUTPUT_DIR / "trajectory_rows.csv"
ADAPTIVE_MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        msg = f"Expected a JSON object in {path}."
        raise TypeError(msg)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV table."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@functools.cache
def _source_hash() -> str:
    """Fingerprint every source file that can affect measured evolution."""
    repository = Path(__file__).resolve().parents[3]
    selected = [
        Path(__file__),
        Path(__file__).with_name("config.py"),
        Path(__file__).with_name("run.py"),
        repository / "experiments" / "circuit_benchmarks" / "circuits.py",
        repository / "experiments" / "circuit_benchmarks" / "common.py",
        repository / "experiments" / "circuit_benchmarks" / "config.py",
    ]
    selected.extend(sorted((repository / "src" / "mqt" / "yaqs").rglob("*.py")))
    digest = hashlib.sha256()
    for path in selected:
        digest.update(path.relative_to(repository).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@functools.cache
def _environment_identity() -> dict[str, Any]:
    """Return timing-relevant host and software metadata."""
    return {
        "python": sys.version,
        "packages": _package_versions(),
        "cpu_model": _cpu_model(),
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
    }


def _adaptive_endpoints() -> tuple[dict[str, int], dict[tuple[str, str], float], dict[str, Any]]:
    """Load frozen common endpoints and their method-specific infidelities."""
    manifest = _load_json(ADAPTIVE_MANIFEST_PATH)
    cases = manifest.get("cases")
    if not isinstance(cases, dict):
        msg = "The adaptive manifest has no case records."
        raise RuntimeError(msg)

    endpoints: dict[str, int] = {}
    for case_key in CASE_ORDER:
        record = cases.get(case_key)
        if (
            not isinstance(record, dict)
            or record.get("status") != "success"
            or record.get("criterion_met") is not True
            or record.get("right_censored") is not False
        ):
            msg = f"Adaptive endpoint is incomplete or censored for {case_key}."
            raise RuntimeError(msg)
        endpoints[case_key] = int(record["stop_step"])

    expected_errors: dict[tuple[str, str], float] = {}
    for row in _read_csv(ADAPTIVE_ROWS_PATH):
        case_key = row.get("case", "")
        method = row.get("method", "")
        if case_key in endpoints and method in METHODS and int(row["step"]) == endpoints[case_key]:
            key = (case_key, method)
            if key in expected_errors:
                msg = f"Duplicate adaptive endpoint row for {key}."
                raise RuntimeError(msg)
            expected_errors[key] = float(row["infidelity_normalized"])

    expected = {(case_key, method) for case_key in CASE_ORDER for method in METHODS}
    if set(expected_errors) != expected:
        msg = f"Missing adaptive endpoint errors: {sorted(expected - set(expected_errors))}."
        raise RuntimeError(msg)
    return endpoints, expected_errors, manifest


def _task_payload(
    *,
    case_key: str,
    method: str,
    stop_step: int,
    expected_infidelity: float,
    adaptive_manifest: dict[str, Any],
    repeat: int,
) -> dict[str, Any]:
    """Return the complete identity of one measured timing trajectory."""
    return {
        "campaign_id": CAMPAIGN_ID,
        "source_hash": _source_hash(),
        "environment": _environment_identity(),
        "adaptive_campaign_id": adaptive_manifest.get("campaign_id"),
        "adaptive_source_hash": adaptive_manifest.get("source_hash"),
        "case": case_key,
        "method": method,
        "chi_cap": CHI_CAP,
        "n_sub": _n_sub(method),
        "stop_step": stop_step,
        "expected_endpoint_infidelity": expected_infidelity,
        "repeat": repeat,
        "timing_scope": "apply_mps_step_only",
        "threads": 1,
    }


def _task_path(case_key: str, method: str, repeat: int) -> Path:
    """Return the task path for one measured repeat."""
    return TASK_DIR / f"{case_key}__{method}__repeat{repeat}.json"


def _load_reusable_task(path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return one current successful task, if present."""
    if not path.is_file():
        return None
    task = _load_json(path)
    if task.get("status") == "success" and task.get("payload") == payload:
        return task
    return None


def _dense_endpoint(case_key: str, stop_step: int) -> Any:
    """Construct the untimed dense endpoint for validation."""
    case = CASES[case_key]
    dense = initial_vector(case)
    for physical_step in build_schedule(case, steps=stop_step):
        dense = apply_dense_step(dense, physical_step, case.n_qubits)
    return dense


def _run_once(
    *,
    case_key: str,
    method: str,
    stop_step: int,
    dense_endpoint: Any,
    expected_infidelity: float,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, float], float]:
    """Run one isolated trajectory and time only production MPS updates."""
    case = CASES[case_key]
    schedule = build_schedule(case, steps=stop_step)
    compiled = compile_schedule(schedule, case.n_qubits)
    state = initial_mps(case)
    params = digital_params(method, CHI_CAP, n_sub=_n_sub(method))
    rows: list[dict[str, Any]] = [
        {
            "campaign_id": CAMPAIGN_ID,
            "case": case_key,
            "method": method,
            "step": 0,
            "step_runtime_s": 0.0,
            "cumulative_runtime_s": 0.0,
        }
    ]
    cumulative = 0.0
    wall_started = time.perf_counter()
    gc.collect()

    with threadpool_limits(limits=1):
        for step, compiled_step in enumerate(compiled, start=1):
            started = time.perf_counter()
            apply_mps_step(state, compiled_step, params)
            step_runtime = time.perf_counter() - started
            cumulative += step_runtime
            state.assert_bond_shapes_consistent(max_bond_dim=CHI_CAP)
            rows.append(
                {
                    "campaign_id": CAMPAIGN_ID,
                    "case": case_key,
                    "method": method,
                    "step": step,
                    "step_runtime_s": step_runtime,
                    "cumulative_runtime_s": cumulative,
                }
            )
            if step == 1 or step % 10 == 0 or step == stop_step:
                print(
                    f"{case_key}/{method}/{label}: step={step}/{stop_step} "
                    f"cumulative={cumulative:.3f}s",
                    flush=True,
                )

    metrics = normalized_state_fidelity(dense_endpoint, state.to_vec())
    observed = float(metrics["infidelity_normalized"])
    if not math.isclose(observed, expected_infidelity, rel_tol=0.0, abs_tol=ENDPOINT_TOLERANCE):
        msg = (
            f"Endpoint mismatch for {case_key}/{method}: observed {observed:.16g}, "
            f"expected {expected_infidelity:.16g}."
        )
        raise RuntimeError(msg)
    return rows, metrics, time.perf_counter() - wall_started


def _run_pair(
    *,
    case_key: str,
    method: str,
    stop_step: int,
    expected_infidelity: float,
    adaptive_manifest: dict[str, Any],
    resume: bool,
) -> list[dict[str, Any]]:
    """Warm one pair, then generate every missing measured repeat."""
    payloads = [
        _task_payload(
            case_key=case_key,
            method=method,
            stop_step=stop_step,
            expected_infidelity=expected_infidelity,
            adaptive_manifest=adaptive_manifest,
            repeat=repeat,
        )
        for repeat in range(REPEATS)
    ]
    current = [
        _load_reusable_task(_task_path(case_key, method, repeat), payload)
        if resume
        else None
        for repeat, payload in enumerate(payloads)
    ]
    if all(task is not None for task in current):
        return [task for task in current if task is not None]

    dense_endpoint = _dense_endpoint(case_key, stop_step)
    warmup_path = WARMUP_DIR / f"{case_key}__{method}.json"
    warmup_started = time.perf_counter()
    try:
        _, metrics, elapsed = _run_once(
            case_key=case_key,
            method=method,
            stop_step=stop_step,
            dense_endpoint=dense_endpoint,
            expected_infidelity=expected_infidelity,
            label="warmup",
        )
        _atomic_json(
            warmup_path,
            {
                "status": "success",
                "completed_utc": _utc_now(),
                "case": case_key,
                "method": method,
                "stop_step": stop_step,
                "endpoint_metrics": metrics,
                "elapsed_wall_s": elapsed,
                "source_hash": _source_hash(),
                "environment": _environment_identity(),
            },
        )
    except Exception as error:  # ruff: ignore[blind-except] - retain timing failures
        _atomic_json(
            warmup_path,
            {
                "status": "failed",
                "completed_utc": _utc_now(),
                "case": case_key,
                "method": method,
                "stop_step": stop_step,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_wall_s": time.perf_counter() - warmup_started,
            },
        )
        raise

    tasks: list[dict[str, Any]] = []
    for repeat, (payload, reusable) in enumerate(zip(payloads, current, strict=True)):
        if reusable is not None:
            tasks.append(reusable)
            continue
        path = _task_path(case_key, method, repeat)
        started = time.perf_counter()
        try:
            rows, metrics, elapsed = _run_once(
                case_key=case_key,
                method=method,
                stop_step=stop_step,
                dense_endpoint=dense_endpoint,
                expected_infidelity=expected_infidelity,
                label=f"repeat{repeat}",
            )
            for row in rows:
                row["repeat"] = repeat
            task = {
                "status": "success",
                "payload": payload,
                "completed_utc": _utc_now(),
                "endpoint_metrics": metrics,
                "elapsed_wall_s": elapsed,
                "rows": rows,
            }
        except Exception as error:  # ruff: ignore[blind-except] - retain timing failures
            task = {
                "status": "failed",
                "payload": payload,
                "completed_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_wall_s": time.perf_counter() - started,
                "rows": [],
            }
        _atomic_json(path, task)
        tasks.append(task)
        if task["status"] != "success":
            return tasks
    return tasks


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate measured repeats pointwise as median and min--max."""
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        key = (str(row["case"]), str(row["method"]), int(row["step"]))
        grouped.setdefault(key, []).append(float(row["cumulative_runtime_s"]))

    summary: list[dict[str, Any]] = []
    for case_key in CASE_ORDER:
        for method in METHODS:
            matching = sorted(
                (key, values)
                for key, values in grouped.items()
                if key[:2] == (case_key, method)
            )
            for (_, _, step), values in matching:
                if len(values) != REPEATS:
                    msg = f"Expected {REPEATS} timing repeats for {case_key}/{method}/step{step}."
                    raise RuntimeError(msg)
                summary.append(
                    {
                        "campaign_id": CAMPAIGN_ID,
                        "case": case_key,
                        "method": method,
                        "step": step,
                        "median_cumulative_runtime_s": statistics.median(values),
                        "min_cumulative_runtime_s": min(values),
                        "max_cumulative_runtime_s": max(values),
                        "repeats": REPEATS,
                    }
                )
    return summary


def _write_aggregate(
    tasks: list[dict[str, Any]],
    *,
    endpoints: dict[str, int],
    adaptive_manifest: dict[str, Any],
) -> None:
    """Write complete measured rows, summary, and provenance manifest."""
    rows = [
        row
        for task in tasks
        if task.get("status") == "success"
        for row in task.get("rows", [])
    ]
    if not rows:
        msg = "No successful timing rows are available."
        raise RuntimeError(msg)
    rows.sort(
        key=lambda row: (
            CASE_ORDER.index(str(row["case"])),
            METHODS.index(str(row["method"])),
            int(row["repeat"]),
            int(row["step"]),
        )
    )
    _atomic_csv(ROWS_PATH, rows)
    summary = _summary_rows(rows)
    _atomic_csv(SUMMARY_PATH, summary)

    git = _git_metadata()
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "created_utc": _utc_now(),
        "source_hash": _source_hash(),
        "git": git,
        "environment": _environment_identity(),
        "adaptive_campaign_id": adaptive_manifest.get("campaign_id"),
        "adaptive_source_hash": adaptive_manifest.get("source_hash"),
        "endpoints": endpoints,
        "repeats": REPEATS,
        "warmup_trajectories_per_pair": 1,
        "timing_scope": {
            "included": "apply_mps_step for every gate in each complete Trotter step",
            "excluded": (
                "schedule compilation, MPS initialization, dense evolution, endpoint fidelity, "
                "parameter/resource diagnostics, and plotting"
            ),
            "threads": 1,
        },
        "tasks": {
            f"{task['payload']['case']}/{task['payload']['method']}/repeat{task['payload']['repeat']}": {
                "status": task["status"],
                "elapsed_wall_s": task.get("elapsed_wall_s"),
                "endpoint_infidelity": task.get("endpoint_metrics", {}).get(
                    "infidelity_normalized"
                ),
            }
            for task in tasks
        },
        "artifacts": {
            "timing_rows": str(ROWS_PATH),
            "timing_summary": str(SUMMARY_PATH),
        },
    }
    _atomic_json(MANIFEST_PATH, manifest)


def main(argv: list[str] | None = None) -> int:
    """Run missing timing repeats serially and aggregate a complete campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASE_ORDER, dest="cases")
    parser.add_argument("--method", action="append", choices=METHODS, dest="methods")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    endpoints, expected_errors, adaptive_manifest = _adaptive_endpoints()
    selected_cases = tuple(args.cases) if args.cases else CASE_ORDER
    selected_methods = tuple(args.methods) if args.methods else METHODS
    failures: list[dict[str, Any]] = []
    for case_key in selected_cases:
        for method in selected_methods:
            tasks = _run_pair(
                case_key=case_key,
                method=method,
                stop_step=endpoints[case_key],
                expected_infidelity=expected_errors[(case_key, method)],
                adaptive_manifest=adaptive_manifest,
                resume=not args.no_resume,
            )
            failures.extend(task for task in tasks if task.get("status") != "success")
            if failures:
                break
        if failures:
            break

    current_tasks: list[dict[str, Any]] = []
    for case_key in CASE_ORDER:
        for method in METHODS:
            for repeat in range(REPEATS):
                payload = _task_payload(
                    case_key=case_key,
                    method=method,
                    stop_step=endpoints[case_key],
                    expected_infidelity=expected_errors[(case_key, method)],
                    adaptive_manifest=adaptive_manifest,
                    repeat=repeat,
                )
                task = _load_reusable_task(_task_path(case_key, method, repeat), payload)
                if task is not None:
                    current_tasks.append(task)

    expected_task_count = len(CASE_ORDER) * len(METHODS) * REPEATS
    if len(current_tasks) == expected_task_count:
        _write_aggregate(
            current_tasks,
            endpoints=endpoints,
            adaptive_manifest=adaptive_manifest,
        )
    else:
        print(
            f"Timing campaign incomplete: {len(current_tasks)}/{expected_task_count} measured tasks.",
            flush=True,
        )

    if failures:
        for task in failures:
            print(
                f"{task['payload']['case']}/{task['payload']['method']}: "
                f"{task.get('error_type')}: {task.get('error_message')}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
