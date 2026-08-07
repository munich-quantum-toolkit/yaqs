# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Run an isolated TDVP-only Krylov-tolerance control at the fixed horizon.

The control holds the Figure 4 protocol fixed at the 4x4 Ising circuit through
``n=15`` and varies only the TDVP bond cap and Krylov stopping tolerance.  It
never invokes either direct gate-application baseline and writes only to its
own output directory.

Examples:
    uv run python -m experiments.circuit_benchmarks.extensions.krylov_tolerance_control
    uv run python -m experiments.circuit_benchmarks.extensions.krylov_tolerance_control \
        --tolerance 1e-12 --tolerance 1e-10 --cap 26 --cap 28 --timing-repeats 3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from experiments.circuit_benchmarks import circuits as benchmark_circuits
from experiments.circuit_benchmarks import common as benchmark_common
from experiments.circuit_benchmarks import config as benchmark_config
from experiments.circuit_benchmarks import run as benchmark_run
from experiments.circuit_benchmarks import tracing as benchmark_tracing
from experiments.circuit_benchmarks.circuits import build_schedule
from experiments.circuit_benchmarks.config import CASES, OUTPUT_DIR
from experiments.circuit_benchmarks.tracing import ResourceTracer
from mqt.yaqs.core.data_structures import mpo as mpo_module
from mqt.yaqs.core.data_structures import mps as mps_module
from mqt.yaqs.core.data_structures.simulation_parameters import DigitalSimParams
from mqt.yaqs.core.libraries import gate_library as gate_library_module
from mqt.yaqs.core.linalg import svd_utils as svd_utils_module
from mqt.yaqs.core.methods import decompositions as decompositions_module
from mqt.yaqs.core.methods.tdvp import sweep_utils as sweep_utils_module
from mqt.yaqs.digital import digital_tjm
from threadpoolctl import threadpool_info, threadpool_limits

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from experiments.circuit_benchmarks.common import CompiledStep

CAMPAIGN_ID = "circuit_tdvp_krylov_tolerance_control_v1"
SCHEMA_VERSION = 1
CASE_KEY = "ising_2d"
METHOD = "gate_local_2tdvp"
TARGET_STEP = 15
N_SUB = 2
THREADS = 1
DEFAULT_TOLERANCES = (benchmark_config.KRYLOV_TOL, 1e-10, 1e-8, 1e-6, 1e-4)
DEFAULT_CAPS = (24, 26, 28, 30, 32)
DEFAULT_TIMING_REPEATS = benchmark_config.TIMING_REPEATS
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

CONTROL_DIR = OUTPUT_DIR / "krylov_tolerance_control"
ACCURACY_PATH = CONTROL_DIR / "accuracy.csv"
TIMING_PATH = CONTROL_DIR / "timing_rows.csv"
SUMMARY_PATH = CONTROL_DIR / "summary.csv"
MANIFEST_PATH = CONTROL_DIR / "manifest.json"

ACCURACY_FIELDS = (
    "campaign_id",
    "case",
    "method",
    "chi_max",
    "n_sub",
    "target_step",
    "krylov_tolerance",
    "svd_threshold",
    "max_infidelity_through",
    "endpoint_infidelity",
    "infidelity_by_step_json",
    "parameter_count_by_step_json",
    "peak_parameter_count",
    "peak_bond_dim",
)
TIMING_FIELDS = (
    "campaign_id",
    "case",
    "method",
    "chi_max",
    "n_sub",
    "target_step",
    "krylov_tolerance",
    "repeat",
    "runtime_s",
    "endpoint_infidelity",
)
SUMMARY_FIELDS = (
    *ACCURACY_FIELDS,
    "median_runtime_s",
    "min_runtime_s",
    "max_runtime_s",
    "timing_repeats",
)


@dataclass(frozen=True, order=True)
class ControlPoint:
    """One Krylov-tolerance and bond-cap pair."""

    krylov_tolerance: float
    chi_max: int


def normalize_grid(
    tolerances: Sequence[float],
    caps: Sequence[int],
    timing_repeats: int,
) -> tuple[ControlPoint, ...]:
    """Validate and return the unique Cartesian control grid.

    Args:
        tolerances: Positive finite Krylov stopping tolerances.
        caps: Positive TDVP bond-dimension caps.
        timing_repeats: Number of measured timing trajectories per point.

    Returns:
        Unique points sorted first by tolerance and then by bond cap.

    Raises:
        ValueError: If either grid is empty, a value is invalid, or the repeat
            count is not positive.
    """
    if timing_repeats < 1:
        msg = "timing_repeats must be at least one."
        raise ValueError(msg)
    if not tolerances:
        msg = "At least one Krylov tolerance is required."
        raise ValueError(msg)
    if not caps:
        msg = "At least one bond cap is required."
        raise ValueError(msg)

    normalized_tolerances: set[float] = set()
    for tolerance in tolerances:
        value = float(tolerance)
        if not math.isfinite(value) or value <= 0.0:
            msg = f"Krylov tolerances must be finite and positive, got {tolerance!r}."
            raise ValueError(msg)
        normalized_tolerances.add(value)

    normalized_caps: set[int] = set()
    for cap in caps:
        value = int(cap)
        if value != cap or value < 1:
            msg = f"Bond caps must be positive integers, got {cap!r}."
            raise ValueError(msg)
        normalized_caps.add(value)

    return tuple(
        ControlPoint(tolerance, cap) for tolerance in sorted(normalized_tolerances) for cap in sorted(normalized_caps)
    )


def _point_key(row: Mapping[str, Any]) -> ControlPoint:
    """Return the control point represented by one serialized row."""
    return ControlPoint(float(row["krylov_tolerance"]), int(row["chi_max"]))


def summarize_complete_points(
    accuracy_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    *,
    timing_repeats: int,
) -> list[dict[str, Any]]:
    """Validate the compact raw rows and summarize every complete point.

    Incomplete timing groups are omitted so an interrupted campaign can be
    resumed from its atomically written measured repeats.  Malformed or
    duplicate rows always fail instead of being silently discarded.

    Args:
        accuracy_rows: One accuracy/resource record per completed point.
        timing_rows: Uninstrumented measured timing records.
        timing_repeats: Required repeat count for a complete point.

    Returns:
        Summary rows for points having one accuracy row and all timing repeats.

    Raises:
        RuntimeError: If a point or repeat occurs more than once.
        ValueError: If a row contains an invalid method, protocol, metric, or
            timing repeat.
    """
    if timing_repeats < 1:
        msg = "timing_repeats must be at least one."
        raise ValueError(msg)

    accuracy_by_point: dict[ControlPoint, Mapping[str, Any]] = {}
    for row in accuracy_rows:
        _validate_common_row(row)
        point = _point_key(row)
        if point in accuracy_by_point:
            msg = f"Duplicate accuracy row for tau={point.krylov_tolerance:g}, chi={point.chi_max}."
            raise RuntimeError(msg)
        maximum = float(row["max_infidelity_through"])
        endpoint = float(row["endpoint_infidelity"])
        parameters = int(row["peak_parameter_count"])
        peak_bond = int(row["peak_bond_dim"])
        try:
            errors = [float(value) for value in json.loads(str(row["infidelity_by_step_json"]))]
            parameters_by_step = [int(value) for value in json.loads(str(row["parameter_count_by_step_json"]))]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            msg = f"Invalid trajectory record for tau={point.krylov_tolerance:g}, chi={point.chi_max}."
            raise ValueError(msg) from error
        if (
            not math.isfinite(maximum)
            or not math.isfinite(endpoint)
            or maximum < 0.0
            or endpoint < 0.0
            or endpoint > maximum + 1e-12
            or parameters < 1
            or peak_bond < 1
            or peak_bond > point.chi_max
            or len(errors) != TARGET_STEP + 1
            or len(parameters_by_step) != TARGET_STEP + 1
            or any(not math.isfinite(value) or value < 0.0 for value in errors)
            or any(value < 1 for value in parameters_by_step)
            or not math.isclose(errors[-1], endpoint, rel_tol=0.0, abs_tol=1e-14)
            or not math.isclose(max(errors), maximum, rel_tol=0.0, abs_tol=1e-14)
        ):
            msg = f"Invalid accuracy row for tau={point.krylov_tolerance:g}, chi={point.chi_max}."
            raise ValueError(msg)
        accuracy_by_point[point] = row

    timings_by_point: dict[ControlPoint, dict[int, Mapping[str, Any]]] = {}
    for row in timing_rows:
        _validate_common_row(row)
        point = _point_key(row)
        repeat = int(row["repeat"])
        runtime = float(row["runtime_s"])
        endpoint = float(row["endpoint_infidelity"])
        if repeat < 0 or repeat >= timing_repeats or not math.isfinite(runtime) or runtime <= 0.0:
            msg = f"Invalid timing row for tau={point.krylov_tolerance:g}, chi={point.chi_max}."
            raise ValueError(msg)
        if not math.isfinite(endpoint) or endpoint < 0.0:
            msg = f"Invalid timing endpoint for tau={point.krylov_tolerance:g}, chi={point.chi_max}."
            raise ValueError(msg)
        repeats = timings_by_point.setdefault(point, {})
        if repeat in repeats:
            msg = f"Duplicate timing repeat {repeat} for tau={point.krylov_tolerance:g}, chi={point.chi_max}."
            raise RuntimeError(msg)
        repeats[repeat] = row

    orphaned = set(timings_by_point).difference(accuracy_by_point)
    if orphaned:
        point = min(orphaned)
        msg = f"Timing rows have no accuracy row for tau={point.krylov_tolerance:g}, chi={point.chi_max}."
        raise RuntimeError(msg)

    summaries: list[dict[str, Any]] = []
    for point, accuracy in sorted(accuracy_by_point.items()):
        repeats = timings_by_point.get(point, {})
        if set(repeats) != set(range(timing_repeats)):
            continue
        endpoints = [float(repeats[index]["endpoint_infidelity"]) for index in range(timing_repeats)]
        if any(
            not math.isclose(
                endpoint,
                float(accuracy["endpoint_infidelity"]),
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            for endpoint in endpoints
        ):
            msg = f"Timing and accuracy endpoints disagree for tau={point.krylov_tolerance:g}, chi={point.chi_max}."
            raise RuntimeError(msg)
        runtimes = [float(repeats[index]["runtime_s"]) for index in range(timing_repeats)]
        summaries.append(
            {
                **dict(accuracy),
                "median_runtime_s": statistics.median(runtimes),
                "min_runtime_s": min(runtimes),
                "max_runtime_s": max(runtimes),
                "timing_repeats": timing_repeats,
            }
        )
    return summaries


def _validate_common_row(row: Mapping[str, Any]) -> None:
    """Validate the fixed method and protocol fields shared by raw rows."""
    if (
        str(row.get("campaign_id")) != CAMPAIGN_ID
        or str(row.get("case")) != CASE_KEY
        or str(row.get("method")) != METHOD
        or int(row.get("n_sub", -1)) != N_SUB
        or int(row.get("target_step", -1)) != TARGET_STEP
    ):
        msg = "Control row does not match the fixed TDVP-only protocol."
        raise ValueError(msg)
    point = _point_key(row)
    normalize_grid([point.krylov_tolerance], [point.chi_max], 1)


def validate_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol_sha256: str,
    timing_repeats: int,
) -> tuple[ControlPoint, ...]:
    """Validate persisted provenance before reusing prior rows.

    Args:
        manifest: Previously written campaign manifest.
        protocol_sha256: Digest of the current source, implementation, circuit,
            dense reference, and fixed numerical protocol.
        timing_repeats: Requested measured repeat count.

    Returns:
        The explicitly persisted control points.

    Raises:
        RuntimeError: If the old artifacts are incompatible with this run.
    """
    expected = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "protocol_sha256": protocol_sha256,
        "timing_repeats": timing_repeats,
    }
    mismatches = [name for name, value in expected.items() if manifest.get(name) != value]
    if mismatches:
        msg = f"Cannot safely resume incompatible control data ({', '.join(mismatches)}); use --no-resume."
        raise RuntimeError(msg)
    raw_points = manifest.get("requested_points")
    if not isinstance(raw_points, list):
        msg = "Cannot safely resume a manifest without requested_points; use --no-resume."
        raise RuntimeError(msg)
    try:
        points = tuple(ControlPoint(float(point["krylov_tolerance"]), int(point["chi_max"])) for point in raw_points)
    except (KeyError, TypeError, ValueError) as error:
        msg = "Cannot safely resume malformed requested_points; use --no-resume."
        raise RuntimeError(msg) from error
    if len(set(points)) != len(points):
        msg = "Cannot safely resume duplicate requested_points; use --no-resume."
        raise RuntimeError(msg)
    for point in points:
        normalize_grid([point.krylov_tolerance], [point.chi_max], timing_repeats)
    return tuple(sorted(points))


def _params(point: ControlPoint) -> DigitalSimParams:
    """Construct the fixed TDVP settings for one control point."""
    return DigitalSimParams(
        observables=[],
        get_state=True,
        preset="exact",
        max_bond_dim=point.chi_max,
        trunc_mode=benchmark_config.TRUNC_MODE,
        svd_threshold=benchmark_config.SVD_THRESHOLD,
        krylov_tol=point.krylov_tolerance,
        gate_mode=benchmark_config.METHOD_TO_GATE_MODE[METHOD],
        tdvp_sweeps=N_SUB,
        tdvp_mode=benchmark_config.TDVP_MODE,
    )


def _spec(point: ControlPoint, *, trace_resources: bool) -> benchmark_run.TrajectorySpec:
    """Return tracer metadata compatible with the fixed-horizon campaign."""
    return benchmark_run.TrajectorySpec(
        run_family="frontier",
        case=CASE_KEY,
        method=METHOD,
        chi_max=point.chi_max,
        n_sub=N_SUB,
        steps=TARGET_STEP,
        trace_resources=trace_resources,
    )


def _apply_schedule(
    state: Any,
    compiled: Sequence[CompiledStep],
    params: DigitalSimParams,
    point: ControlPoint,
    *,
    tracer: ResourceTracer | None,
) -> None:
    """Apply the fixed schedule using the same kernel as Figure 4."""
    spec = _spec(point, trace_resources=tracer is not None)
    for step in compiled:
        benchmark_run._apply_compiled_step(  # noqa: SLF001
            state,
            step,
            params,
            tracer=tracer,
            spec=spec,
        )


def _run_accuracy(
    point: ControlPoint,
    compiled: Sequence[CompiledStep],
    exact: np.ndarray,
) -> dict[str, Any]:
    """Measure full-state error and transient retained MPS resources."""
    case = CASES[CASE_KEY]
    state = benchmark_common.initial_mps(case)
    params = _params(point)
    errors = [benchmark_common.normalized_state_fidelity(exact[0], state.to_vec())["infidelity_normalized"]]
    parameters_by_step = [benchmark_common.parameter_count(state)]
    with ResourceTracer() as tracer:
        tracer.checkpoint(
            state,
            "initial",
            run_family="frontier",
            case=CASE_KEY,
            method=METHOD,
            chi_max=point.chi_max,
            n_sub=N_SUB,
            step=0,
        )
        spec = _spec(point, trace_resources=True)
        for step_number, step in enumerate(compiled, start=1):
            benchmark_run._apply_compiled_step(  # noqa: SLF001
                state,
                step,
                params,
                tracer=tracer,
                spec=spec,
            )
            state.assert_bond_shapes_consistent(max_bond_dim=point.chi_max)
            tracer.checkpoint(
                state,
                "step_end",
                run_family="frontier",
                case=CASE_KEY,
                method=METHOD,
                chi_max=point.chi_max,
                n_sub=N_SUB,
                step=step_number,
            )
            errors.append(
                benchmark_common.normalized_state_fidelity(
                    exact[step_number],
                    state.to_vec(),
                )["infidelity_normalized"]
            )
            parameters_by_step.append(benchmark_common.parameter_count(state))

    return {
        "campaign_id": CAMPAIGN_ID,
        "case": CASE_KEY,
        "method": METHOD,
        "chi_max": point.chi_max,
        "n_sub": N_SUB,
        "target_step": TARGET_STEP,
        "krylov_tolerance": point.krylov_tolerance,
        "svd_threshold": benchmark_config.SVD_THRESHOLD,
        "max_infidelity_through": max(errors),
        "endpoint_infidelity": errors[-1],
        "infidelity_by_step_json": json.dumps(errors, separators=(",", ":")),
        "parameter_count_by_step_json": json.dumps(parameters_by_step, separators=(",", ":")),
        "peak_parameter_count": tracer.peak_parameter_count,
        "peak_bond_dim": tracer.peak_bond_dim,
    }


def _run_timing(
    point: ControlPoint,
    compiled: Sequence[CompiledStep],
    exact_endpoint: np.ndarray,
) -> tuple[float, float]:
    """Time only MPS gate application and return its endpoint infidelity."""
    case = CASES[CASE_KEY]
    state = benchmark_common.initial_mps(case)
    params = _params(point)
    started = time.perf_counter()
    _apply_schedule(state, compiled, params, point, tracer=None)
    runtime = time.perf_counter() - started
    state.assert_bond_shapes_consistent(max_bond_dim=point.chi_max)
    endpoint = benchmark_common.normalized_state_fidelity(
        exact_endpoint,
        state.to_vec(),
    )["infidelity_normalized"]
    return runtime, endpoint


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a compact CSV, returning an empty table when it is absent."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    """Atomically write a compact CSV using a fixed schema."""
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON object."""
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
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Return a SHA-256 file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def _implementation_hash() -> str:
    """Hash the experiment and YAQS modules that determine a trajectory."""
    modules = (
        benchmark_common,
        benchmark_circuits,
        benchmark_config,
        benchmark_run,
        benchmark_tracing,
        digital_tjm,
        sweep_utils_module,
        decompositions_module,
        svd_utils_module,
        gate_library_module,
        mpo_module,
        mps_module,
    )
    digest = hashlib.sha256()
    for module in modules:
        path = Path(module.__file__).resolve()
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _protocol_payload(exact: np.ndarray, schedule: Sequence[Any]) -> dict[str, Any]:
    """Return all fixed scientific and implementation provenance."""
    case = CASES[CASE_KEY]
    return {
        "campaign_id": CAMPAIGN_ID,
        "schema_version": SCHEMA_VERSION,
        "case": CASE_KEY,
        "method": METHOD,
        "target_step": TARGET_STEP,
        "n_sub": N_SUB,
        "svd_threshold": benchmark_config.SVD_THRESHOLD,
        "truncation_mode": benchmark_config.TRUNC_MODE,
        "tdvp_mode": benchmark_config.TDVP_MODE,
        "numerical_precision": str(exact.dtype),
        "dense_reference_shape": list(exact.shape),
        "dense_reference_sha256": hashlib.sha256(np.ascontiguousarray(exact).view(np.uint8)).hexdigest(),
        "circuit_fingerprint": benchmark_circuits.circuit_fingerprint(case, schedule),
        "control_source_sha256": _sha256(Path(__file__)),
        "implementation_sha256": _implementation_hash(),
    }


def _protocol_digest(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible protocol payload canonically."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_metadata() -> dict[str, Any]:
    """Return the repository revision and relevant dirty-state digest."""
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "--", "src/mqt/yaqs", "experiments/circuit_benchmarks"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "git_commit": commit,
        "git_dirty_for_relevant_paths": bool(diff),
        "relevant_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def _thread_metadata() -> dict[str, Any]:
    """Validate and describe the one-thread numerical runtime scope."""
    pools = threadpool_info()
    invalid = [
        pool
        for pool in pools
        if pool.get("user_api") in {"blas", "openmp"} and int(pool.get("num_threads", -1)) != THREADS
    ]
    if invalid:
        detail = ", ".join(f"{pool.get('internal_api')}={pool.get('num_threads')}" for pool in invalid)
        msg = f"Krylov control requires one numerical thread; found {detail}."
        raise RuntimeError(msg)
    return {
        "threads": THREADS,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "threadpools": pools,
    }


def _cpu_model() -> str:
    """Return the host CPU model using the benchmark helper."""
    return benchmark_run._cpu_model()  # noqa: SLF001


def _manifest(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    points: Sequence[ControlPoint],
    timing_repeats: int,
    accuracy_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    complete: bool,
    thread_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete manifest for the current resumable state."""
    return {
        **protocol,
        "protocol_sha256": protocol_sha256,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "isolated TDVP-only Krylov-tolerance control; no MPO or TEBD tasks",
        "requested_points": [asdict(point) for point in sorted(points)],
        "timing_scope": (
            "MPS gate application through the complete n=15 circuit; initialization, "
            "compilation, fidelity, resource tracing, and serialization excluded"
        ),
        "timing_warmups_per_point": 1,
        "timing_repeats": timing_repeats,
        "resource_scope": (
            "peak retained full-chain MPS parameters sampled after every state-changing "
            "factorization and at step endpoints"
        ),
        "complete": complete,
        "completed_points": len(summary_rows),
        "hardware": {"cpu_model": _cpu_model(), **dict(thread_metadata)},
        "artifacts": {
            "accuracy": str(ACCURACY_PATH),
            "timing_rows": str(TIMING_PATH),
            "summary": str(SUMMARY_PATH),
        },
        "row_counts": {
            "accuracy": len(accuracy_rows),
            "timing_rows": len(timing_rows),
            "summary": len(summary_rows),
        },
        "output_sha256": {
            "accuracy": _sha256(ACCURACY_PATH),
            "timing_rows": _sha256(TIMING_PATH),
            "summary": _sha256(SUMMARY_PATH),
        },
        **_git_metadata(),
    }


def _persist(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    points: Sequence[ControlPoint],
    timing_repeats: int,
    accuracy_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    complete: bool,
    thread_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Atomically replace the compact tables and then their manifest."""
    summary_rows = summarize_complete_points(
        accuracy_rows,
        timing_rows,
        timing_repeats=timing_repeats,
    )
    _atomic_csv(ACCURACY_PATH, sorted(accuracy_rows, key=_point_key), ACCURACY_FIELDS)
    _atomic_csv(
        TIMING_PATH,
        sorted(
            timing_rows,
            key=lambda row: (*asdict(_point_key(row)).values(), int(row["repeat"])),
        ),
        TIMING_FIELDS,
    )
    _atomic_csv(SUMMARY_PATH, summary_rows, SUMMARY_FIELDS)
    _atomic_json(
        MANIFEST_PATH,
        _manifest(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            points=points,
            timing_repeats=timing_repeats,
            accuracy_rows=accuracy_rows,
            timing_rows=timing_rows,
            summary_rows=summary_rows,
            complete=complete,
            thread_metadata=thread_metadata,
        ),
    )
    return summary_rows


def run_control(
    *,
    tolerances: Sequence[float],
    caps: Sequence[int],
    timing_repeats: int,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Run or resume the requested TDVP-only Krylov control grid."""
    requested = normalize_grid(tolerances, caps, timing_repeats)
    case = CASES[CASE_KEY]
    schedule = build_schedule(case, steps=TARGET_STEP)
    compiled = benchmark_common.compile_schedule(schedule, case.n_qubits)
    # Build only the required dense reference in memory.  Reference generation
    # is outside all timing scopes and does not create or invalidate shared
    # campaign tasks for any comparison method.
    exact = benchmark_common.dense_reference_trajectory(case, schedule)
    expected_shape = (TARGET_STEP + 1, 2**case.n_qubits)
    if exact.ndim != 2 or exact.shape[0] < expected_shape[0] or exact.shape[1] != expected_shape[1]:
        msg = f"Unexpected dense-reference shape {exact.shape}; expected at least {expected_shape}."
        raise RuntimeError(msg)

    protocol = _protocol_payload(exact, schedule)
    protocol_sha256 = _protocol_digest(protocol)
    accuracy_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    persisted_points: tuple[ControlPoint, ...] = ()
    if resume and MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        persisted_points = validate_resume_manifest(
            manifest,
            protocol_sha256=protocol_sha256,
            timing_repeats=timing_repeats,
        )
        accuracy_rows = [dict(row) for row in _read_csv(ACCURACY_PATH)]
        timing_rows = [dict(row) for row in _read_csv(TIMING_PATH)]
        summarize_complete_points(
            accuracy_rows,
            timing_rows,
            timing_repeats=timing_repeats,
        )
    elif resume and any(path.exists() for path in (ACCURACY_PATH, TIMING_PATH, SUMMARY_PATH)):
        msg = "Control tables exist without a compatible manifest; use --no-resume."
        raise RuntimeError(msg)

    points = tuple(sorted(set(persisted_points).union(requested)))
    existing_accuracy = {_point_key(row) for row in accuracy_rows}
    existing_timings = {(_point_key(row), int(row["repeat"])) for row in timing_rows}

    with threadpool_limits(limits=THREADS):
        thread_metadata = _thread_metadata()
        _persist(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            points=points,
            timing_repeats=timing_repeats,
            accuracy_rows=accuracy_rows,
            timing_rows=timing_rows,
            complete=False,
            thread_metadata=thread_metadata,
        )
        for point in points:
            if point not in existing_accuracy:
                print(
                    f"TDVP accuracy: tau={point.krylov_tolerance:g}, chi={point.chi_max}",
                    flush=True,
                )
                accuracy_rows.append(_run_accuracy(point, compiled, exact))
                existing_accuracy.add(point)
                _persist(
                    protocol=protocol,
                    protocol_sha256=protocol_sha256,
                    points=points,
                    timing_repeats=timing_repeats,
                    accuracy_rows=accuracy_rows,
                    timing_rows=timing_rows,
                    complete=False,
                    thread_metadata=thread_metadata,
                )

            missing_repeats = [repeat for repeat in range(timing_repeats) if (point, repeat) not in existing_timings]
            if missing_repeats:
                print(
                    f"TDVP timing warmup: tau={point.krylov_tolerance:g}, chi={point.chi_max}",
                    flush=True,
                )
                _run_timing(point, compiled, exact[TARGET_STEP])
            for repeat in missing_repeats:
                runtime, endpoint = _run_timing(point, compiled, exact[TARGET_STEP])
                timing_rows.append(
                    {
                        "campaign_id": CAMPAIGN_ID,
                        "case": CASE_KEY,
                        "method": METHOD,
                        "chi_max": point.chi_max,
                        "n_sub": N_SUB,
                        "target_step": TARGET_STEP,
                        "krylov_tolerance": point.krylov_tolerance,
                        "repeat": repeat,
                        "runtime_s": runtime,
                        "endpoint_infidelity": endpoint,
                    }
                )
                existing_timings.add((point, repeat))
                _persist(
                    protocol=protocol,
                    protocol_sha256=protocol_sha256,
                    points=points,
                    timing_repeats=timing_repeats,
                    accuracy_rows=accuracy_rows,
                    timing_rows=timing_rows,
                    complete=False,
                    thread_metadata=thread_metadata,
                )
                print(f"  repeat {repeat}: {runtime:.6g} s", flush=True)

        summaries = summarize_complete_points(
            accuracy_rows,
            timing_rows,
            timing_repeats=timing_repeats,
        )
        if {_point_key(row) for row in summaries} != set(points):
            msg = "The Krylov control ended without a complete summary for every requested point."
            raise RuntimeError(msg)
        _persist(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            points=points,
            timing_repeats=timing_repeats,
            accuracy_rows=accuracy_rows,
            timing_rows=timing_rows,
            complete=True,
            thread_metadata=thread_metadata,
        )

    print(f"Wrote {SUMMARY_PATH}")
    return summaries


def main(argv: list[str] | None = None) -> int:
    """Run the isolated, resumable TDVP Krylov-tolerance control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerance",
        action="append",
        type=float,
        dest="tolerances",
        help="Krylov tolerance to test; repeat for multiple values.",
    )
    parser.add_argument(
        "--cap",
        action="append",
        type=int,
        dest="caps",
        help="TDVP bond cap to test; repeat for multiple values.",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=DEFAULT_TIMING_REPEATS,
        help="Measured timings per point after one unrecorded warmup.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard prior control rows instead of safely resuming them.",
    )
    args = parser.parse_args(argv)
    run_control(
        tolerances=DEFAULT_TOLERANCES if args.tolerances is None else args.tolerances,
        caps=DEFAULT_CAPS if args.caps is None else args.caps,
        timing_repeats=args.timing_repeats,
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
