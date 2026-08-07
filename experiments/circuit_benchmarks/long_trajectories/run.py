# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Run fixed-cap circuit trajectories until every method is saturated.

The campaign is intentionally isolated from the frozen 30-step resource
campaign. For each physical circuit, the dense state and all three MPS updates
advance in lockstep until every method is outside the reliable regime and its
infidelity has varied by at most ``SATURATION_LOG_RANGE_DECADES`` over the
trailing ``SATURATION_WINDOW_STEPS`` Trotter steps. ``MAX_STEPS`` is only a
safety bound; reaching it is recorded as right-censoring rather than successful
saturation.
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
import hashlib
import json
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.circuit_benchmarks.circuits import build_schedule, circuit_fingerprint
from experiments.circuit_benchmarks.common import (
    apply_dense_step,
    apply_mps_step,
    bond_profile,
    compile_schedule,
    digital_params,
    initial_mps,
    initial_vector,
    normalized_state_fidelity,
    parameter_count,
)
from experiments.circuit_benchmarks.config import (
    CASES,
    KRYLOV_TOL,
    METHODS,
    RELIABILITY_THRESHOLD,
    SVD_THRESHOLD,
    TDVP_PRODUCTION_SUBSTEPS,
    TRUNC_MODE,
)
from experiments.circuit_benchmarks.run import _cpu_model, _git_metadata, _package_versions

from .config import (
    CAMPAIGN_ID,
    CASE_ORDER,
    CHI_CAP,
    DT,
    MAX_STEPS,
    OUTPUT_DIR,
    SATURATION_LOG_RANGE_DECADES,
    SATURATION_WINDOW_STEPS,
)

TASK_DIR = OUTPUT_DIR / "tasks"
ROWS_PATH = OUTPUT_DIR / "trajectory_rows.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def _utc_now() -> str:
    """Return an ISO-formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON document atomically."""
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
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write homogeneous records as CSV atomically."""
    if not rows:
        msg = "Cannot write an empty trajectory table."
        raise ValueError(msg)
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _source_hash() -> str:
    """Fingerprint the numerical code used by this isolated campaign."""
    repository = Path(__file__).resolve().parents[3]
    selected = [
        Path(__file__),
        Path(__file__).with_name("config.py"),
        repository / "experiments" / "circuit_benchmarks" / "circuits.py",
        repository / "experiments" / "circuit_benchmarks" / "common.py",
    ]
    selected.extend(sorted((repository / "src" / "mqt" / "yaqs").rglob("*.py")))
    digest = hashlib.sha256()
    for path in selected:
        digest.update(path.relative_to(repository).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _n_sub(method: str) -> int:
    """Return the frozen subdivision setting for one comparison method."""
    return TDVP_PRODUCTION_SUBSTEPS if method == "gate_local_2tdvp" else 1


def _window_is_saturated(errors: list[float] | deque[float]) -> bool:
    """Return whether one trailing error window is unreliable and flat."""
    if len(errors) < SATURATION_WINDOW_STEPS:
        return False
    values = np.asarray(errors, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= RELIABILITY_THRESHOLD):
        return False
    log_range = float(np.ptp(np.log10(values)))
    return log_range <= SATURATION_LOG_RANGE_DECADES


def _criterion_met(saturated: dict[str, bool]) -> bool:
    """Return whether every method satisfies the persistence requirement."""
    return bool(saturated) and all(saturated.values())


def _task_payload(case_key: str) -> dict[str, Any]:
    """Return the complete identity of one case calculation."""
    case = CASES[case_key]
    schedule = build_schedule(case, steps=MAX_STEPS)
    return {
        "campaign_id": CAMPAIGN_ID,
        "source_hash": _source_hash(),
        "case": case_key,
        "circuit_fingerprint": circuit_fingerprint(case, schedule),
        "dt": DT,
        "chi_cap": CHI_CAP,
        "n_sub": {method: _n_sub(method) for method in METHODS},
        "saturation_log_range_decades": SATURATION_LOG_RANGE_DECADES,
        "saturation_window_steps": SATURATION_WINDOW_STEPS,
        "max_steps": MAX_STEPS,
        "svd_threshold": SVD_THRESHOLD,
        "krylov_tolerance": KRYLOV_TOL,
        "truncation_mode": TRUNC_MODE,
    }


def _task_path(case_key: str) -> Path:
    return TASK_DIR / f"{case_key}.json"


def _load_reusable_task(case_key: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    path = _task_path(case_key)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        task = json.load(handle)
    if task.get("status") == "success" and task.get("payload") == payload:
        return task
    return None


def _row(
    *,
    case_key: str,
    method: str,
    step: int,
    metrics: dict[str, float],
    state: Any,
) -> dict[str, Any]:
    """Create one plotted trajectory record."""
    profile = bond_profile(state)
    return {
        "campaign_id": CAMPAIGN_ID,
        "case": case_key,
        "method": method,
        "chi_cap": CHI_CAP,
        "n_sub": _n_sub(method),
        "step": step,
        "time": step * DT,
        **metrics,
        "current_parameter_count": parameter_count(state),
        "current_peak_bond_dim": max(profile[1:-1], default=1),
    }


def run_case(case_key: str, *, resume: bool) -> dict[str, Any]:
    """Run one dense reference and all MPS updates to their common endpoint."""
    payload = _task_payload(case_key)
    if resume:
        reusable = _load_reusable_task(case_key, payload)
        if reusable is not None:
            return reusable

    case = CASES[case_key]
    schedule = build_schedule(case, steps=MAX_STEPS)
    compiled = compile_schedule(schedule, case.n_qubits)
    dense = initial_vector(case)
    states = {method: initial_mps(case) for method in METHODS}
    params = {
        method: digital_params(method, CHI_CAP, n_sub=_n_sub(method))
        for method in METHODS
    }
    error_windows = {
        method: deque(maxlen=SATURATION_WINDOW_STEPS) for method in METHODS
    }
    individual_saturation_steps: dict[str, int | None] = {method: None for method in METHODS}
    first_reliability_crossings: dict[str, int | None] = {method: None for method in METHODS}
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        with threadpool_limits(limits=1):
            for method, state in states.items():
                metrics = normalized_state_fidelity(dense, np.asarray(state.to_vec()))
                rows.append(
                    _row(
                        case_key=case_key,
                        method=method,
                        step=0,
                        metrics=metrics,
                        state=state,
                    )
                )

            stop_step = MAX_STEPS
            criterion_met = False
            for step, (physical_step, compiled_step) in enumerate(
                zip(schedule, compiled, strict=True),
                start=1,
            ):
                dense = apply_dense_step(dense, physical_step, case.n_qubits)
                step_errors: dict[str, float] = {}
                for method, state in states.items():
                    apply_mps_step(state, compiled_step, params[method])
                    state.assert_bond_shapes_consistent(max_bond_dim=CHI_CAP)
                    metrics = normalized_state_fidelity(dense, np.asarray(state.to_vec()))
                    error = metrics["infidelity_normalized"]
                    step_errors[method] = error
                    rows.append(
                        _row(
                            case_key=case_key,
                            method=method,
                            step=step,
                            metrics=metrics,
                            state=state,
                        )
                    )
                    if (
                        first_reliability_crossings[method] is None
                        and error > RELIABILITY_THRESHOLD
                    ):
                        first_reliability_crossings[method] = step
                    error_windows[method].append(error)
                    if (
                        individual_saturation_steps[method] is None
                        and _window_is_saturated(error_windows[method])
                    ):
                        individual_saturation_steps[method] = step

                if step == 1 or step % 10 == 0:
                    error_summary = ", ".join(
                        f"{method}={step_errors[method]:.3g}" for method in METHODS
                    )
                    print(
                        f"{case_key}: step={step}/{MAX_STEPS} t={step * DT:g}; "
                        f"infidelity: {error_summary}",
                        flush=True,
                    )

                saturated = {
                    method: _window_is_saturated(error_windows[method])
                    for method in METHODS
                }
                if _criterion_met(saturated):
                    stop_step = step
                    criterion_met = True
                    break

        task = {
            "status": "success",
            "payload": payload,
            "completed_utc": _utc_now(),
            "criterion_met": criterion_met,
            "right_censored": not criterion_met,
            "stop_reason": "saturation" if criterion_met else "max_steps",
            "stop_step": stop_step,
            "stop_time": stop_step * DT,
            "first_reliability_crossing_steps": first_reliability_crossings,
            "individual_saturation_steps": individual_saturation_steps,
            "elapsed_s": time.perf_counter() - started,
            "rows": rows,
        }
    except Exception as error:  # ruff: ignore[blind-except] - retain scientific failures
        task = {
            "status": "failed",
            "payload": payload,
            "completed_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_s": time.perf_counter() - started,
            "rows": rows,
        }
    _atomic_json(_task_path(case_key), task)
    return task


def _write_aggregate(tasks: list[dict[str, Any]]) -> None:
    rows = [row for task in tasks if task.get("status") == "success" for row in task["rows"]]
    _atomic_csv(ROWS_PATH, rows)
    git = _git_metadata()
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "created_utc": _utc_now(),
        "source_hash": _source_hash(),
        "git": git,
        "environment": {
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
        },
        "criterion": {
            "definition": (
                "Every method remains outside the reliable regime while the "
                "range of log10 normalized infidelity over its trailing window "
                "does not exceed the stated tolerance."
            ),
            "reliability_threshold": RELIABILITY_THRESHOLD,
            "log_range_decades": SATURATION_LOG_RANGE_DECADES,
            "window_steps": SATURATION_WINDOW_STEPS,
            "window_time": SATURATION_WINDOW_STEPS * DT,
            "max_steps": MAX_STEPS,
        },
        "cases": {
            str(task["payload"]["case"]): {
                "status": task["status"],
                "criterion_met": task.get("criterion_met"),
                "right_censored": task.get("right_censored"),
                "stop_step": task.get("stop_step"),
                "stop_time": task.get("stop_time"),
                "first_reliability_crossing_steps": task.get(
                    "first_reliability_crossing_steps"
                ),
                "individual_saturation_steps": task.get("individual_saturation_steps"),
                "elapsed_s": task.get("elapsed_s"),
            }
            for task in tasks
        },
        "artifacts": {
            "trajectory_rows": str(ROWS_PATH),
        },
    }
    _atomic_json(MANIFEST_PATH, manifest)


def main(argv: list[str] | None = None) -> int:
    """Run selected cases and aggregate every completed current task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=CASE_ORDER,
        dest="cases",
        help="Run only this case; repeat to select multiple cases.",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    selected = tuple(args.cases) if args.cases else CASE_ORDER

    selected_tasks = [run_case(case_key, resume=not args.no_resume) for case_key in selected]
    failures = [task for task in selected_tasks if task.get("status") != "success"]

    current_tasks: list[dict[str, Any]] = []
    for case_key in CASE_ORDER:
        payload = _task_payload(case_key)
        task = _load_reusable_task(case_key, payload)
        if task is not None:
            current_tasks.append(task)
    if current_tasks:
        _write_aggregate(current_tasks)

    if failures:
        for task in failures:
            print(
                f"{task['payload']['case']}: {task.get('error_type')}: "
                f"{task.get('error_message')}",
                file=sys.stderr,
            )
        return 1
    for task in selected_tasks:
        print(
            f"{task['payload']['case']}: stop={task.get('stop_step')} "
            f"criterion_met={task.get('criterion_met')} elapsed={task.get('elapsed_s'):.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
