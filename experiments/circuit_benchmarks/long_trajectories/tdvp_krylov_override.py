# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Rerun only TDVP with a looser Krylov tolerance at frozen endpoints.

This campaign is deliberately additive: it reads the endpoints selected by the
published long-trajectory campaign, but never rewrites that campaign or runs
either direct gate-application baseline.  Accuracy and bond profiles come from
one TDVP trajectory per case.  Timing uses a separate uninstrumented warm-up
and three measured TDVP trajectories per case.

Examples:
    uv run python -m experiments.circuit_benchmarks.long_trajectories.tdvp_krylov_override \
        --stage accuracy
    uv run python -m experiments.circuit_benchmarks.long_trajectories.tdvp_krylov_override \
        --stage timing --case ising_2d
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
import functools
import gc
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    DT,
    SVD_THRESHOLD,
    TRUNC_MODE,
)
from experiments.circuit_benchmarks.run import _cpu_model, _git_metadata, _package_versions

from .config import CASE_ORDER, OUTPUT_DIR as BASE_OUTPUT_DIR
from .run import _atomic_csv, _atomic_json

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mps import MPS

CAMPAIGN_ID = "circuit-long-trajectory-tdvp-krylov-override-v1"
BASE_CAMPAIGN_ID = "circuit-infidelity-until-saturation-v2"
METHOD = "gate_local_2tdvp"
CHI_CAP = 32
N_SUB = 2
KRYLOV_TOLERANCE = 1e-5
PROFILE_MAX_STEP = 30
REPEATS = 3
ENDPOINT_TOLERANCE = 1e-10
# The first four accuracy tasks completed with this runner hash.  A subsequent
# change only repaired JSON-to-CSV field ordering; accepting that exact hash
# avoids repeating the numerical work while every scientific payload field is
# still required to match.
COMPLETED_ACCURACY_SOURCE_HASHES = frozenset({"81be240cd543b7b55c4bddb9811f7995f91237a6e41388754d27d8241951456a"})

OUTPUT_DIR = BASE_OUTPUT_DIR / "tdvp_krylov_1e-5"
ACCURACY_TASK_DIR = OUTPUT_DIR / "tasks" / "accuracy"
TIMING_TASK_DIR = OUTPUT_DIR / "tasks" / "timing"
WARMUP_DIR = OUTPUT_DIR / "warmups"
TRAJECTORY_PATH = OUTPUT_DIR / "trajectory_rows.csv"
PROFILE_PATH = OUTPUT_DIR / "bond_profiles.csv"
TIMING_PATH = OUTPUT_DIR / "timing_rows.csv"
TIMING_SUMMARY_PATH = OUTPUT_DIR / "timing_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

BASE_MANIFEST_PATH = BASE_OUTPUT_DIR / "manifest.json"
BASE_TRAJECTORY_PATH = BASE_OUTPUT_DIR / "trajectory_rows.csv"

TRAJECTORY_FIELDS = (
    "campaign_id",
    "case",
    "chi_cap",
    "current_parameter_count",
    "current_peak_bond_dim",
    "fidelity_normalized",
    "infidelity_normalized",
    "method",
    "n_sub",
    "norm_approx",
    "norm_drift",
    "norm_exact",
    "step",
    "time",
)
PROFILE_FIELDS = ("case", "method", "step", "bond", "bond_dimension")
TIMING_FIELDS = (
    "campaign_id",
    "case",
    "method",
    "step",
    "step_runtime_s",
    "cumulative_runtime_s",
    "repeat",
)
TIMING_SUMMARY_FIELDS = (
    "campaign_id",
    "case",
    "method",
    "step",
    "median_cumulative_runtime_s",
    "min_cumulative_runtime_s",
    "max_cumulative_runtime_s",
    "repeats",
)


def _utc_now() -> str:
    """Return an ISO-formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    """Load and validate one JSON object."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        msg = f"Expected a JSON object in {path}."
        raise TypeError(msg)
    return value


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one provenance input or artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.cache
def _source_hash() -> str:
    """Fingerprint numerical sources that can affect this control."""
    repository = Path(__file__).resolve().parents[3]
    selected = [
        Path(__file__),
        Path(__file__).with_name("config.py"),
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
    """Return host and software metadata relevant to measured timings."""
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


def _extract_frozen_endpoints(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Validate and extract the four common endpoints from the base campaign."""
    if manifest.get("campaign_id") != BASE_CAMPAIGN_ID:
        msg = f"Expected base campaign {BASE_CAMPAIGN_ID!r}."
        raise RuntimeError(msg)
    records = manifest.get("cases")
    if not isinstance(records, Mapping):
        msg = "The base long-trajectory manifest has no case records."
        raise RuntimeError(msg)

    endpoints: dict[str, int] = {}
    for case_key in CASE_ORDER:
        record = records.get(case_key)
        if (
            not isinstance(record, Mapping)
            or record.get("status") != "success"
            or record.get("criterion_met") is not True
            or record.get("right_censored") is not False
        ):
            msg = f"The frozen endpoint for {case_key} is incomplete or censored."
            raise RuntimeError(msg)
        endpoint = int(record.get("stop_step", -1))
        if endpoint < PROFILE_MAX_STEP:
            msg = (
                f"The frozen endpoint for {case_key} is {endpoint}, below the required "
                f"bond-profile horizon {PROFILE_MAX_STEP}."
            )
            raise RuntimeError(msg)
        endpoints[case_key] = endpoint
    return endpoints


@functools.cache
def _base_provenance() -> tuple[dict[str, int], dict[str, Any]]:
    """Return frozen endpoints and content-addressed base provenance."""
    if not BASE_MANIFEST_PATH.is_file() or not BASE_TRAJECTORY_PATH.is_file():
        msg = "The frozen long-trajectory outputs are required before this control can run."
        raise FileNotFoundError(msg)
    manifest = _load_json(BASE_MANIFEST_PATH)
    endpoints = _extract_frozen_endpoints(manifest)
    provenance = {
        "campaign_id": manifest.get("campaign_id"),
        "source_hash": manifest.get("source_hash"),
        "manifest_path": str(BASE_MANIFEST_PATH),
        "manifest_sha256": _sha256(BASE_MANIFEST_PATH),
        "trajectory_path": str(BASE_TRAJECTORY_PATH),
        "trajectory_sha256": _sha256(BASE_TRAJECTORY_PATH),
    }
    return endpoints, provenance


def _params() -> Any:
    """Construct TDVP settings while overriding only the Krylov tolerance."""
    params = digital_params(METHOD, CHI_CAP, n_sub=N_SUB)
    params.krylov_tol = KRYLOV_TOLERANCE
    return params


def _accuracy_payload(case_key: str, endpoint: int, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete identity of one accuracy trajectory."""
    case = CASES[case_key]
    schedule = build_schedule(case, steps=endpoint)
    return {
        "campaign_id": CAMPAIGN_ID,
        "source_hash": _source_hash(),
        "base_manifest_sha256": provenance["manifest_sha256"],
        "base_trajectory_sha256": provenance["trajectory_sha256"],
        "case": case_key,
        "circuit_fingerprint": circuit_fingerprint(case, schedule),
        "stop_step": endpoint,
        "chi_cap": CHI_CAP,
        "n_sub": N_SUB,
        "krylov_tolerance": KRYLOV_TOLERANCE,
        "svd_threshold": SVD_THRESHOLD,
        "truncation_mode": TRUNC_MODE,
        "bond_profile_max_step": PROFILE_MAX_STEP,
        "threads": 1,
    }


def _accuracy_task_path(case_key: str) -> Path:
    return ACCURACY_TASK_DIR / f"{case_key}.json"


def _timing_task_path(case_key: str, repeat: int) -> Path:
    return TIMING_TASK_DIR / f"{case_key}__repeat{repeat}.json"


def _load_reusable_task(path: Path, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a successful task only when its complete payload still matches."""
    if not path.is_file():
        return None
    task = _load_json(path)
    if task.get("status") == "success" and task.get("payload") == payload:
        return task
    return None


def _accuracy_payload_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Match current or completed pre-aggregation-fix accuracy payloads."""
    if actual == expected:
        return True
    if actual.get("source_hash") not in COMPLETED_ACCURACY_SOURCE_HASHES:
        return False
    actual_without_source = {key: value for key, value in actual.items() if key != "source_hash"}
    expected_without_source = {key: value for key, value in expected.items() if key != "source_hash"}
    return actual_without_source == expected_without_source


def _load_reusable_accuracy_task(path: Path, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a matching accuracy task, including the aggregation-only hash exception."""
    if not path.is_file():
        return None
    task = _load_json(path)
    actual = task.get("payload")
    if task.get("status") == "success" and isinstance(actual, Mapping) and _accuracy_payload_matches(actual, payload):
        return task
    return None


def _trajectory_row(*, case_key: str, step: int, state: MPS, metrics: Mapping[str, float]) -> dict[str, Any]:
    """Return one row in the base long-trajectory schema."""
    profile = bond_profile(state)
    return {
        "campaign_id": CAMPAIGN_ID,
        "case": case_key,
        "chi_cap": CHI_CAP,
        "current_parameter_count": parameter_count(state),
        "current_peak_bond_dim": max(profile[1:-1], default=1),
        "fidelity_normalized": metrics["fidelity_normalized"],
        "infidelity_normalized": metrics["infidelity_normalized"],
        "method": METHOD,
        "n_sub": N_SUB,
        "norm_approx": metrics["norm_approx"],
        "norm_drift": metrics["norm_drift"],
        "norm_exact": metrics["norm_exact"],
        "step": step,
        "time": step * DT,
    }


def _profile_rows(*, case_key: str, step: int, state: MPS) -> list[dict[str, Any]]:
    """Return portable internal-bond rows for one step-end state."""
    return [
        {
            "case": case_key,
            "method": METHOD,
            "step": step,
            "bond": bond,
            "bond_dimension": dimension,
        }
        for bond, dimension in enumerate(bond_profile(state)[1:-1], start=1)
    ]


def _run_accuracy_case(
    case_key: str,
    *,
    endpoint: int,
    provenance: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    """Run one dense reference and one TDVP trajectory through a frozen endpoint."""
    payload = _accuracy_payload(case_key, endpoint, provenance)
    path = _accuracy_task_path(case_key)
    if resume:
        reusable = _load_reusable_accuracy_task(path, payload)
        if reusable is not None:
            return reusable

    case = CASES[case_key]
    schedule = build_schedule(case, steps=endpoint)
    compiled = compile_schedule(schedule, case.n_qubits)
    dense = initial_vector(case)
    state = initial_mps(case)
    params = _params()
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        with threadpool_limits(limits=1):
            metrics = normalized_state_fidelity(dense, state.to_vec())
            rows.append(_trajectory_row(case_key=case_key, step=0, state=state, metrics=metrics))
            profiles.extend(_profile_rows(case_key=case_key, step=0, state=state))
            for step, (physical_step, compiled_step) in enumerate(zip(schedule, compiled, strict=True), start=1):
                dense = apply_dense_step(dense, physical_step, case.n_qubits)
                apply_mps_step(state, compiled_step, params)
                state.assert_bond_shapes_consistent(max_bond_dim=CHI_CAP)
                metrics = normalized_state_fidelity(dense, state.to_vec())
                rows.append(_trajectory_row(case_key=case_key, step=step, state=state, metrics=metrics))
                if step <= PROFILE_MAX_STEP:
                    profiles.extend(_profile_rows(case_key=case_key, step=step, state=state))
                if step == 1 or step % 10 == 0 or step == endpoint:
                    print(
                        f"accuracy {case_key}: step={step}/{endpoint}; "
                        f"infidelity={metrics['infidelity_normalized']:.6g}",
                        flush=True,
                    )
        task = {
            "status": "success",
            "payload": payload,
            "completed_utc": _utc_now(),
            "elapsed_wall_s": time.perf_counter() - started,
            "endpoint_metrics": metrics,
            "rows": rows,
            "bond_profiles": profiles,
        }
    except Exception as error:  # ruff: ignore[blind-except] - preserve numerical failures
        task = {
            "status": "failed",
            "payload": payload,
            "completed_utc": _utc_now(),
            "elapsed_wall_s": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "rows": rows,
            "bond_profiles": profiles,
        }
    _atomic_json(path, task)
    return task


def _current_accuracy_tasks(endpoints: Mapping[str, int], provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load every current successful accuracy task."""
    tasks: list[dict[str, Any]] = []
    for case_key in CASE_ORDER:
        payload = _accuracy_payload(case_key, endpoints[case_key], provenance)
        task = _load_reusable_accuracy_task(_accuracy_task_path(case_key), payload)
        if task is not None:
            tasks.append(task)
    return tasks


def _rows_in_schema_order(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    """Validate exact field sets and restore the declared CSV column order."""
    ordered_fields = tuple(fields)
    expected = set(ordered_fields)
    ordered: list[dict[str, Any]] = []
    for row in rows:
        actual = set(row)
        if actual != expected:
            msg = f"Row fields {sorted(actual)!r} do not match schema {sorted(expected)!r}."
            raise RuntimeError(msg)
        ordered.append({field: row[field] for field in ordered_fields})
    return ordered


def _write_accuracy_aggregate(tasks: Sequence[Mapping[str, Any]]) -> None:
    """Write portable accuracy and bond-profile tables from current tasks."""
    rows = [dict(row) for task in tasks for row in task["rows"]]
    profiles = [dict(row) for task in tasks for row in task["bond_profiles"]]
    rows.sort(key=lambda row: (CASE_ORDER.index(str(row["case"])), int(row["step"])))
    profiles.sort(
        key=lambda row: (
            CASE_ORDER.index(str(row["case"])),
            int(row["step"]),
            int(row["bond"]),
        )
    )
    rows = _rows_in_schema_order(rows, TRAJECTORY_FIELDS)
    profiles = _rows_in_schema_order(profiles, PROFILE_FIELDS)
    if rows:
        _atomic_csv(TRAJECTORY_PATH, rows)
    if profiles:
        _atomic_csv(PROFILE_PATH, profiles)


def _dense_endpoint(case_key: str, endpoint: int) -> Any:
    """Build the dense endpoint outside the timing region."""
    case = CASES[case_key]
    dense = initial_vector(case)
    for physical_step in build_schedule(case, steps=endpoint):
        dense = apply_dense_step(dense, physical_step, case.n_qubits)
    return dense


def _run_timed_once(
    *,
    case_key: str,
    endpoint: int,
    dense_endpoint: Any,
    expected_infidelity: float,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, float], float]:
    """Time only TDVP gate application for one complete trajectory."""
    case = CASES[case_key]
    compiled = compile_schedule(build_schedule(case, steps=endpoint), case.n_qubits)
    state = initial_mps(case)
    params = _params()
    rows = [
        {
            "campaign_id": CAMPAIGN_ID,
            "case": case_key,
            "method": METHOD,
            "step": 0,
            "step_runtime_s": 0.0,
            "cumulative_runtime_s": 0.0,
            "repeat": -1,
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
                    "method": METHOD,
                    "step": step,
                    "step_runtime_s": step_runtime,
                    "cumulative_runtime_s": cumulative,
                    "repeat": -1,
                }
            )
            if step == 1 or step % 10 == 0 or step == endpoint:
                print(
                    f"timing {case_key}/{label}: step={step}/{endpoint}; cumulative={cumulative:.3f}s",
                    flush=True,
                )

    metrics = normalized_state_fidelity(dense_endpoint, state.to_vec())
    observed = float(metrics["infidelity_normalized"])
    if not math.isclose(observed, expected_infidelity, rel_tol=0.0, abs_tol=ENDPOINT_TOLERANCE):
        msg = f"Endpoint mismatch for {case_key}: observed {observed:.16g}, expected {expected_infidelity:.16g}."
        raise RuntimeError(msg)
    return rows, metrics, time.perf_counter() - wall_started


def _timing_payload(
    *,
    case_key: str,
    endpoint: int,
    expected_infidelity: float,
    provenance: Mapping[str, Any],
    repeat: int,
) -> dict[str, Any]:
    """Return the complete identity of one measured timing trajectory."""
    return {
        "campaign_id": CAMPAIGN_ID,
        "source_hash": _source_hash(),
        "base_manifest_sha256": provenance["manifest_sha256"],
        "base_trajectory_sha256": provenance["trajectory_sha256"],
        "environment": _environment_identity(),
        "case": case_key,
        "method": METHOD,
        "stop_step": endpoint,
        "expected_endpoint_infidelity": expected_infidelity,
        "chi_cap": CHI_CAP,
        "n_sub": N_SUB,
        "krylov_tolerance": KRYLOV_TOLERANCE,
        "svd_threshold": SVD_THRESHOLD,
        "repeat": repeat,
        "timing_scope": "apply_mps_step_only",
        "threads": 1,
    }


def _run_timing_case(
    case_key: str,
    *,
    endpoint: int,
    expected_infidelity: float,
    provenance: Mapping[str, Any],
    resume: bool,
) -> list[dict[str, Any]]:
    """Run one warm-up and all missing measured TDVP timing repeats."""
    payloads = [
        _timing_payload(
            case_key=case_key,
            endpoint=endpoint,
            expected_infidelity=expected_infidelity,
            provenance=provenance,
            repeat=repeat,
        )
        for repeat in range(REPEATS)
    ]
    current = [
        _load_reusable_task(_timing_task_path(case_key, repeat), payload) if resume else None
        for repeat, payload in enumerate(payloads)
    ]
    if all(task is not None for task in current):
        return [task for task in current if task is not None]

    dense = _dense_endpoint(case_key, endpoint)
    warmup_path = WARMUP_DIR / f"{case_key}.json"
    try:
        _, metrics, elapsed = _run_timed_once(
            case_key=case_key,
            endpoint=endpoint,
            dense_endpoint=dense,
            expected_infidelity=expected_infidelity,
            label="warmup",
        )
        _atomic_json(
            warmup_path,
            {
                "status": "success",
                "completed_utc": _utc_now(),
                "case": case_key,
                "stop_step": endpoint,
                "endpoint_metrics": metrics,
                "elapsed_wall_s": elapsed,
                "source_hash": _source_hash(),
                "environment": _environment_identity(),
            },
        )
    except Exception as error:  # ruff: ignore[blind-except] - preserve timing failures
        _atomic_json(
            warmup_path,
            {
                "status": "failed",
                "completed_utc": _utc_now(),
                "case": case_key,
                "stop_step": endpoint,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise

    tasks: list[dict[str, Any]] = []
    for repeat, (payload, reusable) in enumerate(zip(payloads, current, strict=True)):
        if reusable is not None:
            tasks.append(reusable)
            continue
        path = _timing_task_path(case_key, repeat)
        started = time.perf_counter()
        try:
            rows, metrics, elapsed = _run_timed_once(
                case_key=case_key,
                endpoint=endpoint,
                dense_endpoint=dense,
                expected_infidelity=expected_infidelity,
                label=f"repeat{repeat}",
            )
            for row in rows:
                row["repeat"] = repeat
            task = {
                "status": "success",
                "payload": payload,
                "completed_utc": _utc_now(),
                "elapsed_wall_s": elapsed,
                "endpoint_metrics": metrics,
                "rows": rows,
            }
        except Exception as error:  # ruff: ignore[blind-except] - preserve timing failures
            task = {
                "status": "failed",
                "payload": payload,
                "completed_utc": _utc_now(),
                "elapsed_wall_s": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "rows": [],
            }
        _atomic_json(path, task)
        tasks.append(task)
        if task["status"] != "success":
            break
    return tasks


def _accuracy_endpoint(task: Mapping[str, Any]) -> float:
    """Return the validated final infidelity from an accuracy task."""
    rows = task.get("rows")
    if not isinstance(rows, list) or not rows:
        msg = "Accuracy task has no trajectory rows."
        raise RuntimeError(msg)
    endpoint = int(task["payload"]["stop_step"])
    final = rows[-1]
    if int(final["step"]) != endpoint:
        msg = "Accuracy task does not reach its frozen endpoint."
        raise RuntimeError(msg)
    return float(final["infidelity_normalized"])


def _current_timing_tasks(
    accuracy_tasks: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Load all current measured tasks associated with completed accuracy tasks."""
    tasks: list[dict[str, Any]] = []
    by_case = {str(task["payload"]["case"]): task for task in accuracy_tasks}
    for case_key in CASE_ORDER:
        accuracy = by_case.get(case_key)
        if accuracy is None:
            continue
        endpoint = int(accuracy["payload"]["stop_step"])
        expected = _accuracy_endpoint(accuracy)
        for repeat in range(REPEATS):
            payload = _timing_payload(
                case_key=case_key,
                endpoint=endpoint,
                expected_infidelity=expected,
                provenance=provenance,
                repeat=repeat,
            )
            task = _load_reusable_task(_timing_task_path(case_key, repeat), payload)
            if task is not None:
                tasks.append(task)
    return tasks


def summarize_timing_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate complete measured repeats pointwise as median and min--max."""
    grouped: dict[tuple[str, int], dict[int, float]] = {}
    for row in rows:
        key = (str(row["case"]), int(row["step"]))
        repeat = int(row["repeat"])
        runtime = float(row["cumulative_runtime_s"])
        if repeat in grouped.setdefault(key, {}):
            msg = f"Duplicate timing row for {key}, repeat {repeat}."
            raise RuntimeError(msg)
        grouped[key][repeat] = runtime

    summary: list[dict[str, Any]] = []
    for case_key in CASE_ORDER:
        matching = sorted((key, values) for key, values in grouped.items() if key[0] == case_key)
        for (_, step), values in matching:
            if set(values) != set(range(REPEATS)):
                continue
            runtimes = [values[repeat] for repeat in range(REPEATS)]
            summary.append(
                {
                    "campaign_id": CAMPAIGN_ID,
                    "case": case_key,
                    "method": METHOD,
                    "step": step,
                    "median_cumulative_runtime_s": statistics.median(runtimes),
                    "min_cumulative_runtime_s": min(runtimes),
                    "max_cumulative_runtime_s": max(runtimes),
                    "repeats": REPEATS,
                }
            )
    return summary


def _write_timing_aggregate(tasks: Sequence[Mapping[str, Any]]) -> None:
    """Write current measured timing rows and complete pointwise summaries."""
    rows = [dict(row) for task in tasks for row in task["rows"]]
    rows.sort(
        key=lambda row: (
            CASE_ORDER.index(str(row["case"])),
            int(row["repeat"]),
            int(row["step"]),
        )
    )
    rows = _rows_in_schema_order(rows, TIMING_FIELDS)
    if rows:
        _atomic_csv(TIMING_PATH, rows)
    summary = summarize_timing_rows(rows)
    summary = _rows_in_schema_order(summary, TIMING_SUMMARY_FIELDS)
    if summary:
        _atomic_csv(TIMING_SUMMARY_PATH, summary)


def _artifact_record(path: Path) -> dict[str, Any] | None:
    """Return path, digest, and byte count for an existing artifact."""
    if not path.is_file():
        return None
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _write_manifest(
    *,
    endpoints: Mapping[str, int],
    provenance: Mapping[str, Any],
    accuracy_tasks: Sequence[Mapping[str, Any]],
    timing_tasks: Sequence[Mapping[str, Any]],
) -> None:
    """Write a content-addressed manifest for every currently complete task."""
    accuracy_by_case = {str(task["payload"]["case"]): task for task in accuracy_tasks}
    timing_counts = {
        case_key: sum(1 for task in timing_tasks if task["payload"]["case"] == case_key) for case_key in CASE_ORDER
    }
    artifacts = {
        name: record
        for name, path in (
            ("trajectory_rows", TRAJECTORY_PATH),
            ("bond_profiles", PROFILE_PATH),
            ("timing_rows", TIMING_PATH),
            ("timing_summary", TIMING_SUMMARY_PATH),
        )
        if (record := _artifact_record(path)) is not None
    }
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "created_utc": _utc_now(),
        "source_hash": _source_hash(),
        "accuracy_task_source_hashes": sorted({str(task["payload"]["source_hash"]) for task in accuracy_tasks}),
        "git": _git_metadata(),
        "environment": _environment_identity(),
        "base_provenance": dict(provenance),
        "protocol": {
            "method": METHOD,
            "chi_cap": CHI_CAP,
            "n_sub": N_SUB,
            "krylov_tolerance": KRYLOV_TOLERANCE,
            "svd_threshold": SVD_THRESHOLD,
            "truncation_mode": TRUNC_MODE,
            "frozen_endpoints": dict(endpoints),
            "bond_profile_max_step": PROFILE_MAX_STEP,
            "threads": 1,
        },
        "timing_scope": {
            "warmup_trajectories_per_case": 1,
            "measured_repeats": REPEATS,
            "included": "apply_mps_step for every gate in each complete Trotter step",
            "excluded": (
                "schedule compilation, MPS initialization, dense evolution, endpoint fidelity, "
                "resource diagnostics, serialization, and plotting"
            ),
        },
        "cases": {
            case_key: {
                "stop_step": endpoints[case_key],
                "accuracy_status": (
                    accuracy_by_case[case_key]["status"] if case_key in accuracy_by_case else "missing"
                ),
                "endpoint_infidelity": (
                    _accuracy_endpoint(accuracy_by_case[case_key]) if case_key in accuracy_by_case else None
                ),
                "timing_repeats_complete": timing_counts[case_key],
            }
            for case_key in CASE_ORDER
        },
        "artifacts": artifacts,
    }
    _atomic_json(MANIFEST_PATH, manifest)


def _run_accuracy_stage(
    selected: Sequence[str],
    *,
    endpoints: Mapping[str, int],
    provenance: Mapping[str, Any],
    resume: bool,
) -> bool:
    """Run selected accuracy tasks and refresh all current aggregates."""
    failed = False
    for case_key in selected:
        task = _run_accuracy_case(
            case_key,
            endpoint=endpoints[case_key],
            provenance=provenance,
            resume=resume,
        )
        if task["status"] != "success":
            print(
                f"accuracy {case_key}: {task.get('error_type')}: {task.get('error_message')}",
                file=sys.stderr,
            )
            failed = True
            break
    current = _current_accuracy_tasks(endpoints, provenance)
    _write_accuracy_aggregate(current)
    return not failed


def _run_timing_stage(
    selected: Sequence[str],
    *,
    endpoints: Mapping[str, int],
    provenance: Mapping[str, Any],
    resume: bool,
) -> bool:
    """Run selected timing tasks using their completed accuracy endpoints."""
    current_accuracy = _current_accuracy_tasks(endpoints, provenance)
    by_case = {str(task["payload"]["case"]): task for task in current_accuracy}
    missing = [case_key for case_key in selected if case_key not in by_case]
    if missing:
        msg = f"Run the accuracy stage first for: {', '.join(missing)}."
        raise RuntimeError(msg)

    failed = False
    for case_key in selected:
        accuracy = by_case[case_key]
        tasks = _run_timing_case(
            case_key,
            endpoint=endpoints[case_key],
            expected_infidelity=_accuracy_endpoint(accuracy),
            provenance=provenance,
            resume=resume,
        )
        failure = next((task for task in tasks if task["status"] != "success"), None)
        if failure is not None:
            print(
                f"timing {case_key}: {failure.get('error_type')}: {failure.get('error_message')}",
                file=sys.stderr,
            )
            failed = True
            break
    current_timing = _current_timing_tasks(current_accuracy, provenance)
    _write_timing_aggregate(current_timing)
    return not failed


def main(argv: list[str] | None = None) -> int:
    """Run resumable accuracy, timing, or both for the isolated control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("accuracy", "timing", "all"), default="all")
    parser.add_argument("--case", action="append", choices=CASE_ORDER, dest="cases")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    selected = tuple(dict.fromkeys(args.cases)) if args.cases else CASE_ORDER
    endpoints, provenance = _base_provenance()
    success = True
    if args.stage in {"accuracy", "all"}:
        success = _run_accuracy_stage(
            selected,
            endpoints=endpoints,
            provenance=provenance,
            resume=not args.no_resume,
        )
    if success and args.stage in {"timing", "all"}:
        success = _run_timing_stage(
            selected,
            endpoints=endpoints,
            provenance=provenance,
            resume=not args.no_resume,
        )

    accuracy_tasks = _current_accuracy_tasks(endpoints, provenance)
    timing_tasks = _current_timing_tasks(accuracy_tasks, provenance)
    _write_manifest(
        endpoints=endpoints,
        provenance=provenance,
        accuracy_tasks=accuracy_tasks,
        timing_tasks=timing_tasks,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
