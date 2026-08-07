# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Run the bounded variational-MPO control used in the long-trajectory figure.

The three primary methods retain their frozen common endpoints.  This isolated
control advances one variational-MPO trajectory per circuit until the first
*completed* Trotter step whose cumulative update time reaches the configured
budget, or until the primary panel endpoint when that occurs first.  The
runtime budget therefore censors only this optional comparison curve; it does
not participate in the plateau criterion used to choose the primary endpoint.

Run from the repository root with::

    uv run python -m experiments.circuit_benchmarks.long_trajectories.variational_control
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
import gzip
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.circuit_benchmarks import circuits as benchmark_circuits
from experiments.circuit_benchmarks import common as benchmark_common
from experiments.circuit_benchmarks import config as benchmark_config
from experiments.circuit_benchmarks.circuits import build_schedule, circuit_fingerprint
from experiments.circuit_benchmarks.config import (
    CASES,
    KRYLOV_TOL,
    SVD_THRESHOLD,
    TRUNC_MODE,
)
from experiments.circuit_benchmarks.run import _cpu_model, _git_metadata, _package_versions
from experiments.variational_mpo import apply_variational_mpo_node
from mqt.yaqs.core.data_structures import mpo as mpo_module
from mqt.yaqs.core.data_structures import mps as mps_module
from mqt.yaqs.core.methods import decompositions as decompositions_module
from mqt.yaqs.digital import digital_tjm
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate, apply_two_qubit_gate

from .config import CASE_ORDER, CHI_CAP, DT, OUTPUT_DIR

CAMPAIGN_ID = "circuit-long-trajectory-variational-mpo-v1"
METHOD = "variational_mpo"
RUNTIME_BUDGET_S = 1.0e2
MAX_SWEEPS = 32
RETRY_MAX_SWEEPS = 128
MONOTONICITY_TOLERANCE = 2e-12

CONTROL_DIR = OUTPUT_DIR / "variational_mpo_control"
TASK_DIR = CONTROL_DIR / "tasks"
ROWS_PATH = CONTROL_DIR / "trajectory_rows.csv"
DIAGNOSTICS_PATH = CONTROL_DIR / "fit_diagnostics.csv.gz"
MANIFEST_PATH = CONTROL_DIR / "manifest.json"
PRIMARY_MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

ROW_FIELDS = (
    "campaign_id",
    "case",
    "method",
    "chi_cap",
    "step",
    "time",
    "fidelity_normalized",
    "infidelity_normalized",
    "norm_exact",
    "norm_approx",
    "norm_drift",
    "current_parameter_count",
    "current_peak_bond_dim",
    "cumulative_runtime_s",
)

DIAGNOSTIC_FIELDS = (
    "case",
    "step",
    "gate_index",
    "gate",
    "sites",
    "sweeps",
    "retried_with_128_sweeps",
    "converged",
    "objective_initial",
    "objective_final",
    "mpo_initializer_objective",
    "input_initializer_objective",
    "best_initializer",
    "rejected_nonimproving_updates",
    "target_max_bond",
    "target_parameter_count",
    "update_runtime_s",
    "fit_runtime_s",
    "fidelity_to_target",
    "objective_trace",
)


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


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    """Write homogeneous CSV records atomically."""
    if not rows:
        msg = f"Cannot write empty table {path}."
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_gzip_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    """Write gzip-compressed CSV records atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    if not path.is_file():
        msg = f"Missing {path}."
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"Expected a JSON object in {path}."
        raise TypeError(msg)
    return payload


@cache
def _source_hash() -> str:
    """Fingerprint every source file used by this control."""
    repository = Path(__file__).resolve().parents[3]
    modules = (
        benchmark_common,
        benchmark_circuits,
        benchmark_config,
        mpo_module,
        mps_module,
        decompositions_module,
        digital_tjm,
    )
    selected = [Path(__file__), repository / "experiments" / "variational_mpo.py"]
    selected.extend(Path(module.__file__).resolve() for module in modules)
    digest = hashlib.sha256()
    for path in sorted(set(selected)):
        digest.update(path.relative_to(repository).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _primary_endpoints() -> tuple[dict[str, int], dict[str, Any]]:
    """Return the frozen three-method panel endpoints."""
    manifest = _load_json(PRIMARY_MANIFEST_PATH)
    cases = manifest.get("cases")
    if not isinstance(cases, dict):
        msg = "The primary long-trajectory manifest has no case records."
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
            msg = f"The primary endpoint is incomplete for {case_key}."
            raise RuntimeError(msg)
        endpoint = int(record["stop_step"])
        if endpoint < 1:
            msg = f"The primary endpoint for {case_key} contains no evolved step."
            raise RuntimeError(msg)
        endpoints[case_key] = endpoint
    return endpoints, manifest


def _stop_reason(*, cumulative_runtime_s: float, step: int, primary_endpoint: int) -> str | None:
    """Choose a censoring reason only at a completed Trotter step."""
    if cumulative_runtime_s >= RUNTIME_BUDGET_S:
        return "runtime_budget_reached_at_completed_step"
    if step >= primary_endpoint:
        return "primary_panel_endpoint"
    return None


def _task_payload(
    case_key: str,
    *,
    primary_endpoint: int,
    primary_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return the full identity of one case calculation."""
    case = CASES[case_key]
    schedule = build_schedule(case, steps=primary_endpoint)
    return {
        "campaign_id": CAMPAIGN_ID,
        "source_hash": _source_hash(),
        "primary_campaign_id": primary_manifest.get("campaign_id"),
        "primary_source_hash": primary_manifest.get("source_hash"),
        "case": case_key,
        "circuit_fingerprint": circuit_fingerprint(case, schedule),
        "dt": DT,
        "chi_cap": CHI_CAP,
        "primary_endpoint": primary_endpoint,
        "runtime_budget_s": RUNTIME_BUDGET_S,
        "runtime_scope": "variational MPS updates only",
        "threads": 1,
        "max_sweeps": MAX_SWEEPS,
        "retry_max_sweeps": RETRY_MAX_SWEEPS,
        "svd_threshold": SVD_THRESHOLD,
        "krylov_tolerance": KRYLOV_TOL,
        "truncation_mode": TRUNC_MODE,
    }


def _task_path(case_key: str) -> Path:
    return TASK_DIR / f"{case_key}.json"


def _load_reusable_task(
    case_key: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    path = _task_path(case_key)
    if not path.is_file():
        return None
    task = _load_json(path)
    if task.get("status") == "success" and task.get("payload") == payload:
        return task
    return None


def _state_row(
    *,
    case_key: str,
    step: int,
    state: Any,
    dense: np.ndarray,
    cumulative_runtime_s: float,
) -> dict[str, Any]:
    """Create one step-end record for Figure 2."""
    metrics = benchmark_common.normalized_state_fidelity(dense, state.to_vec())
    profile = benchmark_common.bond_profile(state)
    return {
        "campaign_id": CAMPAIGN_ID,
        "case": case_key,
        "method": METHOD,
        "chi_cap": CHI_CAP,
        "step": step,
        "time": step * DT,
        **metrics,
        "current_parameter_count": benchmark_common.parameter_count(state),
        "current_peak_bond_dim": max(profile[1:-1], default=1),
        "cumulative_runtime_s": cumulative_runtime_s,
    }


def _apply_variational_step(
    state: Any,
    compiled_step: Any,
    compression_params: Any,
    *,
    case_key: str,
    step_number: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Apply one full step and return timed update cost and fit diagnostics."""
    step_runtime_s = 0.0
    diagnostics: list[dict[str, Any]] = []
    for gate_index, compiled_gate in enumerate(compiled_step.gates):
        started = time.perf_counter()
        if len(compiled_gate.gate.qubits) == 1:
            apply_single_qubit_gate(state, compiled_gate.node)
            step_runtime_s += time.perf_counter() - started
            continue

        q0, q1 = compiled_gate.gate.qubits
        if abs(q0 - q1) == 1:
            apply_two_qubit_gate(state, compiled_gate.node, compression_params)
            state.normalize(form="B", decomposition="QR")
            step_runtime_s += time.perf_counter() - started
            continue

        result = apply_variational_mpo_node(
            state,
            compiled_gate.node,
            compression_params=compression_params,
            max_sweeps=MAX_SWEEPS,
        )
        retried = False
        if not result.converged:
            retried = True
            result = apply_variational_mpo_node(
                state,
                compiled_gate.node,
                compression_params=compression_params,
                max_sweeps=RETRY_MAX_SWEEPS,
            )
        update_runtime_s = time.perf_counter() - started
        step_runtime_s += update_runtime_s
        if not result.converged:
            msg = f"Variational fit failed for {case_key} at step={step_number}, gate={gate_index}, sites=({q0}, {q1})."
            raise RuntimeError(msg)
        if any(np.diff(result.objective_trace) > MONOTONICITY_TOLERANCE) or any(
            np.diff(result.update_trace) > MONOTONICITY_TOLERANCE
        ):
            msg = f"Nonmonotone variational objective for {case_key} at step={step_number}."
            raise RuntimeError(msg)
        state.tensors = result.state.tensors
        state.set_center(result.state.orthogonality_center)
        diagnostics.append(
            {
                "case": case_key,
                "step": step_number,
                "gate_index": gate_index,
                "gate": compiled_gate.gate.name,
                "sites": json.dumps([q0, q1]),
                "sweeps": result.sweeps,
                "retried_with_128_sweeps": retried,
                "converged": result.converged,
                "objective_initial": result.objective_initial,
                "objective_final": result.objective_final,
                "mpo_initializer_objective": result.initializer_objectives["mpo_contract_compress"],
                "input_initializer_objective": result.initializer_objectives["input"],
                "best_initializer": result.best_initializer,
                "rejected_nonimproving_updates": result.rejected_nonimproving_updates,
                "target_max_bond": result.target_max_bond,
                "target_parameter_count": result.target_parameter_count,
                "update_runtime_s": update_runtime_s,
                "fit_runtime_s": result.runtime_s,
                "fidelity_to_target": result.fidelity_to_target,
                "objective_trace": json.dumps(result.objective_trace),
            }
        )
    return step_runtime_s, diagnostics


def run_case(
    case_key: str,
    *,
    primary_endpoint: int,
    primary_manifest: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    """Run one deterministic, runtime-censored variational trajectory."""
    payload = _task_payload(
        case_key,
        primary_endpoint=primary_endpoint,
        primary_manifest=primary_manifest,
    )
    if resume:
        reusable = _load_reusable_task(case_key, payload)
        if reusable is not None:
            return reusable

    case = CASES[case_key]
    schedule = build_schedule(case, steps=primary_endpoint)
    compiled = benchmark_common.compile_schedule(schedule, case.n_qubits)
    dense = benchmark_common.initial_vector(case)
    state = benchmark_common.initial_mps(case)
    compression_params = benchmark_common.digital_params(
        "mpo_contract_compress",
        CHI_CAP,
        n_sub=1,
    )
    rows = [
        _state_row(
            case_key=case_key,
            step=0,
            state=state,
            dense=dense,
            cumulative_runtime_s=0.0,
        )
    ]
    diagnostics: list[dict[str, Any]] = []
    cumulative_runtime_s = 0.0
    started = time.perf_counter()

    try:
        stop_reason: str | None = None
        stop_step = 0
        with threadpool_limits(limits=1):
            for step_number, (physical_step, compiled_step) in enumerate(
                zip(schedule, compiled, strict=True),
                start=1,
            ):
                dense = benchmark_common.apply_dense_step(dense, physical_step, case.n_qubits)
                step_runtime_s, step_diagnostics = _apply_variational_step(
                    state,
                    compiled_step,
                    compression_params,
                    case_key=case_key,
                    step_number=step_number,
                )
                cumulative_runtime_s += step_runtime_s
                diagnostics.extend(step_diagnostics)
                state.assert_bond_shapes_consistent(max_bond_dim=CHI_CAP)
                rows.append(
                    _state_row(
                        case_key=case_key,
                        step=step_number,
                        state=state,
                        dense=dense,
                        cumulative_runtime_s=cumulative_runtime_s,
                    )
                )
                stop_step = step_number
                print(
                    f"{case_key}: step={step_number}/{primary_endpoint} "
                    f"cumulative={cumulative_runtime_s:.3f}s fits={len(diagnostics)}",
                    flush=True,
                )
                stop_reason = _stop_reason(
                    cumulative_runtime_s=cumulative_runtime_s,
                    step=step_number,
                    primary_endpoint=primary_endpoint,
                )
                if stop_reason is not None:
                    break

        if stop_reason is None:
            msg = f"No bounded stopping condition was reached for {case_key}."
            raise RuntimeError(msg)
        task = {
            "status": "success",
            "payload": payload,
            "completed_utc": _utc_now(),
            "stop_reason": stop_reason,
            "stop_step": stop_step,
            "stop_time": stop_step * DT,
            "cumulative_runtime_s": cumulative_runtime_s,
            "runtime_budget_overshoot_s": max(0.0, cumulative_runtime_s - RUNTIME_BUDGET_S),
            "variational_fits": len(diagnostics),
            "retried_fits": sum(bool(row["retried_with_128_sweeps"]) for row in diagnostics),
            "maximum_sweeps": max((int(row["sweeps"]) for row in diagnostics), default=0),
            "all_selected_fits_converged": all(bool(row["converged"]) for row in diagnostics),
            "elapsed_wall_s": time.perf_counter() - started,
            "rows": rows,
            "diagnostics": diagnostics,
        }
    except Exception as error:  # ruff: ignore[blind-except] - retain scientific failures
        task = {
            "status": "failed",
            "payload": payload,
            "completed_utc": _utc_now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_wall_s": time.perf_counter() - started,
            "rows": rows,
            "diagnostics": diagnostics,
        }
    _atomic_json(_task_path(case_key), task)
    return task


def _write_aggregate(
    tasks: list[dict[str, Any]],
    *,
    primary_manifest: dict[str, Any],
) -> None:
    """Write current successful checkpoints and their provenance."""
    successful = [task for task in tasks if task.get("status") == "success"]
    rows = [row for task in successful for row in task["rows"]]
    diagnostics = [row for task in successful for row in task["diagnostics"]]
    if rows:
        _atomic_csv(ROWS_PATH, rows, ROW_FIELDS)
        _atomic_gzip_csv(DIAGNOSTICS_PATH, diagnostics, DIAGNOSTIC_FIELDS)

    git = _git_metadata()
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "created_utc": _utc_now(),
        "source_hash": _source_hash(),
        "primary_campaign_id": primary_manifest.get("campaign_id"),
        "primary_source_hash": primary_manifest.get("source_hash"),
        "runtime_budget_s": RUNTIME_BUDGET_S,
        "stop_definition": (
            "First completed Trotter step with cumulative variational update runtime "
            "at least the budget, or the primary panel endpoint if it occurs first."
        ),
        "runtime_scope": {
            "included": (
                "variational-state gate updates, including exact MPO target construction, "
                "both initializers, and alternating fits for separated gates"
            ),
            "excluded": (
                "schedule compilation, state initialization, dense evolution, fidelity and "
                "resource diagnostics, checkpointing, and plotting"
            ),
            "threads": 1,
            "repeats": 1,
            "warmups": 0,
        },
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
        "git": git,
        "cases": {
            str(task["payload"]["case"]): {
                "status": task["status"],
                "primary_endpoint": task["payload"]["primary_endpoint"],
                "stop_reason": task.get("stop_reason"),
                "stop_step": task.get("stop_step"),
                "stop_time": task.get("stop_time"),
                "cumulative_runtime_s": task.get("cumulative_runtime_s"),
                "runtime_budget_overshoot_s": task.get("runtime_budget_overshoot_s"),
                "variational_fits": task.get("variational_fits"),
                "retried_fits": task.get("retried_fits"),
                "maximum_sweeps": task.get("maximum_sweeps"),
                "all_selected_fits_converged": task.get("all_selected_fits_converged"),
                "elapsed_wall_s": task.get("elapsed_wall_s"),
            }
            for task in tasks
        },
        "artifacts": {
            "trajectory_rows": str(ROWS_PATH),
            "fit_diagnostics": str(DIAGNOSTICS_PATH),
        },
    }
    _atomic_json(MANIFEST_PATH, manifest)


def main(argv: list[str] | None = None) -> int:
    """Run selected controls and aggregate every current checkpoint."""
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
    endpoints, primary_manifest = _primary_endpoints()

    selected_tasks = [
        run_case(
            case_key,
            primary_endpoint=endpoints[case_key],
            primary_manifest=primary_manifest,
            resume=not args.no_resume,
        )
        for case_key in selected
    ]
    failures = [task for task in selected_tasks if task.get("status") != "success"]

    current_tasks: list[dict[str, Any]] = []
    for case_key in CASE_ORDER:
        payload = _task_payload(
            case_key,
            primary_endpoint=endpoints[case_key],
            primary_manifest=primary_manifest,
        )
        task = _load_reusable_task(case_key, payload)
        if task is not None:
            current_tasks.append(task)
    if current_tasks:
        _write_aggregate(current_tasks, primary_manifest=primary_manifest)

    if failures:
        for task in failures:
            print(
                f"{task['payload']['case']}: {task.get('error_type')}: {task.get('error_message')}",
                file=sys.stderr,
            )
        return 1
    for task in selected_tasks:
        runtime = float(task["cumulative_runtime_s"])
        if not math.isfinite(runtime):
            msg = f"Nonfinite terminal runtime for {task['payload']['case']}."
            raise RuntimeError(msg)
        print(
            f"{task['payload']['case']}: stop={task['stop_step']} reason={task['stop_reason']} runtime={runtime:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
