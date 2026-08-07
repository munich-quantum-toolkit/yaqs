# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Run the publication circuit and retained-resource benchmark campaign.

Examples:
    uv run python -m experiments.circuit_benchmarks.run --stage exact
    uv run python -m experiments.circuit_benchmarks.run --stage resolution
    uv run python -m experiments.circuit_benchmarks.run --stage trajectories
    uv run python -m experiments.circuit_benchmarks.run --stage frontier
    uv run python -m experiments.circuit_benchmarks.run --stage timing
    uv run python -m experiments.circuit_benchmarks.run --stage aggregate

Every expensive calculation is an atomic, content-addressed task.  A failed
task is retained rather than silently dropped; pass ``--retry-failed`` to run
such tasks again.  Accuracy runs trace retained MPS storage after every
state-changing SVD (and immediately after an MPO--MPS contraction), whereas
publication timing runs are deliberately uninstrumented.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import os
import tempfile

# Freeze numerical thread pools before importing NumPy/SciPy. Accuracy tasks
# and publication timings both use one numerical thread.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "mqt-yaqs-numba"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mqt-yaqs-matplotlib"))

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.circuit_benchmarks.circuits import build_schedule, circuit_fingerprint
from experiments.circuit_benchmarks.common import (
    CompiledStep,
    compile_schedule,
    dense_reference_trajectory,
    digital_params,
    initial_mps,
    normalized_state_fidelity,
    parameter_count,
    phase_aligned_distance,
    protocol_metadata,
)
from experiments.circuit_benchmarks.config import (
    CAMPAIGN_ID,
    CASE_KEYS,
    CASES,
    CHI_GRID,
    CHI_MAIN,
    DT,
    FRONTIER_CASE_KEY,
    FRONTIER_STEPS,
    FRONTIER_TARGET_STEPS,
    KRYLOV_TOL,
    METHODS,
    N_STEPS,
    OUTPUT_DIR,
    RELIABILITY_THRESHOLD,
    SVD_THRESHOLD,
    TDVP_PRODUCTION_SUBSTEPS,
    TDVP_RESOLUTION_CASE_STEPS,
    TDVP_SUBSTEP_CANDIDATES,
    THRESHOLD_SENSITIVITY,
    TIMING_REPEATS,
    TIMING_WARMUPS,
    TROTTER_ORDER,
    TRUNC_MODE,
    BenchmarkCase,
    Method,
    time_for_step,
)
from experiments.circuit_benchmarks.tracing import ResourceTracer
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate, apply_two_qubit_gate


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.core.data_structures.simulation_parameters import DigitalSimParams

RunFamily = Literal["resolution", "trajectories", "frontier"]

TASKS_DIR = OUTPUT_DIR / "tasks"
EXACT_DIR = OUTPUT_DIR / "exact"
STATE_DIR = OUTPUT_DIR / "states"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Finer subdivision also changes the finite-rank truncation sequence, so this
# confirms a fixed production setting rather than claiming monotonic convergence.
RESOLUTION_INFIDELITY_TOL = RELIABILITY_THRESHOLD * 5e-2


@dataclass(frozen=True)
class TrajectorySpec:
    """Complete identity of one MPS trajectory task."""

    run_family: RunFamily
    case: str
    method: Method
    chi_max: int
    n_sub: int
    steps: int
    trace_resources: bool = True
    save_final_state: bool = False


def utc_now() -> str:
    """Return a compact UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars and paths in publication records."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    msg = f"Object of type {type(value).__name__} is not JSON serializable."
    raise TypeError(msg)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON document atomically."""
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


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    """Write a NumPy array atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write compressed line-delimited checkpoint records atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as raw_handle:
            temporary = Path(raw_handle.name)
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as zipped:
                for row in rows:
                    line = json.dumps(row, sort_keys=True, default=_json_default) + "\n"
                    zipped.write(line.encode("utf-8"))
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"Expected a JSON object in {path}."
        raise TypeError(msg)
    return payload


def _canonical_hash(payload: Mapping[str, Any], *, length: int = 20) -> str:
    """Hash a JSON-compatible task payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def _source_hash() -> str:
    """Hash the experiment implementation that determines task semantics."""
    names = ("config.py", "circuits.py", "common.py", "tracing.py", "run.py", "analyze.py")
    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    for name in names:
        path = directory / name
        digest.update(name.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _task_payload(family: str, **fields: Any) -> dict[str, Any]:
    """Create the content-addressed portion of a task record."""
    return {
        "campaign_id": CAMPAIGN_ID,
        "family": family,
        "source_hash": _source_hash(),
        **fields,
    }


def _task_path(payload: Mapping[str, Any]) -> Path:
    """Return the output path for one task payload."""
    family = str(payload["family"])
    return TASKS_DIR / family / f"{_canonical_hash(payload)}.json"


def _existing_task(
    payload: Mapping[str, Any],
    *,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any] | None:
    """Return a reusable terminal task, if one exists."""
    if not resume:
        return None
    path = _task_path(payload)
    if not path.is_file():
        return None
    task = _load_json(path)
    if task.get("payload") != dict(payload):
        return None
    if task.get("status") == "success":
        return task
    if task.get("status") == "failed" and not retry_failed:
        return task
    return None


def _write_task(payload: Mapping[str, Any], **result: Any) -> dict[str, Any]:
    """Write and return one complete task record."""
    task = {
        "task_id": _canonical_hash(payload),
        "payload": dict(payload),
        "completed_utc": utc_now(),
        **result,
    }
    _atomic_json(_task_path(payload), task)
    return task


def _git_metadata() -> dict[str, Any]:
    """Return commit and dirty-tree provenance without changing the repository."""
    repository = Path(__file__).resolve().parents[2]
    result: dict[str, Any] = {"commit": "unknown", "dirty": None, "diff_hash": "unavailable"}
    try:
        result["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository, text=True, stderr=subprocess.DEVNULL
        )
        result["dirty"] = bool(status.strip())
        binary_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=repository, stderr=subprocess.DEVNULL
        )
        result["diff_hash"] = hashlib.sha256(binary_diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def _package_versions() -> dict[str, str]:
    """Return numerical package versions used by the campaign."""
    versions: dict[str, str] = {}
    for package in ("mqt.yaqs", "numpy", "scipy", "qiskit", "threadpoolctl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _cpu_model() -> str:
    """Return the processor model recorded for timing provenance."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", maxsplit=1)[-1].strip()
    return platform.processor() or "unknown"


def write_manifest(*, last_stage: str) -> dict[str, Any]:
    """Write the campaign protocol and machine provenance."""
    git = _git_metadata()
    schedules = {key: build_schedule(case) for key, case in CASES.items()}
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "updated_utc": utc_now(),
        "last_stage": last_stage,
        "source_hash": _source_hash(),
        "git": git,
        "software": {
            "python": sys.version.split()[0],
            "packages": _package_versions(),
        },
        "hardware": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "logical_cpus": os.cpu_count(),
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
            "threadpools": threadpool_info(),
        },
        "protocol": {
            "dt": DT,
            "n_steps": N_STEPS,
            "trotter_order": TROTTER_ORDER,
            "chi_main": CHI_MAIN,
            "chi_grid": list(CHI_GRID),
            "frontier_case": FRONTIER_CASE_KEY,
            "frontier_steps": FRONTIER_STEPS,
            "frontier_target_steps": list(FRONTIER_TARGET_STEPS),
            "reliability_threshold": RELIABILITY_THRESHOLD,
            "threshold_sensitivity": list(THRESHOLD_SENSITIVITY),
            "svd_threshold": SVD_THRESHOLD,
            "krylov_tol": KRYLOV_TOL,
            "trunc_mode": TRUNC_MODE,
            "timing_warmups": TIMING_WARMUPS,
            "timing_repeats": TIMING_REPEATS,
            "method_semantics": {
                "gate_local_2tdvp": (
                    "Generator-backed two-qubit gates use gate-local two-site TDVP, "
                    "including adjacent gates."
                ),
                "mpo_contract_compress": (
                    "Separated gates use full MPO-MPS contraction followed by global compression; "
                    "adjacent gates use the production TEBD kernel."
                ),
                "tebd_swap": "Separated gates use forward and reverse nearest-neighbor SWAP routing.",
            },
            "cases": {
                key: {
                    **asdict(case),
                    "circuit_fingerprint": circuit_fingerprint(case, schedules[key]),
                }
                for key, case in CASES.items()
            },
        },
        "artifacts": {
            "exact_directory": str(EXACT_DIR),
            "tasks_directory": str(TASKS_DIR),
            "checkpoint_directory": str(CHECKPOINT_DIR),
            "resolution_summary": str(OUTPUT_DIR / "resolution_summary.json"),
            "trajectory_rows": str(OUTPUT_DIR / "trajectory_rows.csv"),
            "frontier_selected": str(OUTPUT_DIR / "frontier_selected.csv"),
            "runtime_summary": str(OUTPUT_DIR / "runtime_summary.csv"),
            "exact_schmidt_tails": str(OUTPUT_DIR / "exact_schmidt_tails.csv"),
        },
    }
    _atomic_json(OUTPUT_DIR / "manifest.json", manifest)
    return manifest


def _exact_payload(case: BenchmarkCase) -> dict[str, Any]:
    """Return the task identity for one dense trajectory."""
    schedule = build_schedule(case)
    return _task_payload(
        "exact",
        case=case.key,
        steps=N_STEPS,
        protocol=protocol_metadata(case, schedule),
    )


def _exact_path(case: BenchmarkCase) -> Path:
    """Return the stable dense-reference path for a case."""
    return EXACT_DIR / f"{case.key}.npy"


def _valid_exact_artifact(task: Mapping[str, Any], case: BenchmarkCase) -> bool:
    """Check shape and hash of a resumed dense reference."""
    path = _exact_path(case)
    if task.get("status") != "success" or not path.is_file():
        return False
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_shape = (N_STEPS + 1, 2**case.n_qubits)
    if array.shape != expected_shape or array.dtype != np.dtype(np.complex128):
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == task.get("artifact_sha256")


def ensure_exact_reference(
    case: BenchmarkCase,
    *,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any]:
    """Generate or reuse the identical-circuit dense reference for one case."""
    payload = _exact_payload(case)
    existing = _existing_task(payload, resume=resume, retry_failed=retry_failed)
    if existing is not None and _valid_exact_artifact(existing, case):
        return existing

    started = time.perf_counter()
    try:
        schedule = build_schedule(case)
        trajectory = dense_reference_trajectory(case, schedule)
        _atomic_npy(_exact_path(case), trajectory.astype(np.complex128, copy=False))
        digest = hashlib.sha256(_exact_path(case).read_bytes()).hexdigest()
        return _write_task(
            payload,
            status="success",
            artifact=str(_exact_path(case)),
            artifact_sha256=digest,
            shape=list(trajectory.shape),
            max_norm_error=float(np.max(np.abs(np.linalg.norm(trajectory, axis=1) - 1.0))),
            elapsed_s=time.perf_counter() - started,
        )
    except Exception as error:  # ruff: ignore[blind-except] - scientific failures must remain in the record
        return _write_task(
            payload,
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            elapsed_s=time.perf_counter() - started,
        )


def run_exact_stage(*, resume: bool = True, retry_failed: bool = False) -> list[dict[str, Any]]:
    """Generate all four dense identical-circuit references."""
    tasks: list[dict[str, Any]] = []
    for key in CASE_KEYS:
        task = ensure_exact_reference(CASES[key], resume=resume, retry_failed=retry_failed)
        tasks.append(task)
    return tasks


def _load_exact(case: BenchmarkCase) -> np.ndarray:
    """Load a validated dense trajectory or raise a dependency error."""
    payload = _exact_payload(case)
    task_path = _task_path(payload)
    if not task_path.is_file():
        msg = f"Dense reference for {case.key} is missing; run --stage exact first."
        raise FileNotFoundError(msg)
    task = _load_json(task_path)
    if not _valid_exact_artifact(task, case):
        msg = f"Dense reference for {case.key} is stale or invalid; rerun --stage exact."
        raise RuntimeError(msg)
    return np.load(_exact_path(case), mmap_mode="r", allow_pickle=False)


def _trajectory_payload(spec: TrajectorySpec) -> dict[str, Any]:
    """Return the content identity of one MPS trajectory."""
    case = CASES[spec.case]
    schedule = build_schedule(case, steps=spec.steps)
    return _task_payload(
        spec.run_family,
        spec=asdict(spec),
        protocol=protocol_metadata(case, schedule),
        numerical={
            "svd_threshold": SVD_THRESHOLD,
            "krylov_tol": KRYLOV_TOL,
            "trunc_mode": TRUNC_MODE,
        },
    )


def _apply_compiled_step(
    state: MPS,
    step: CompiledStep,
    sim_params: DigitalSimParams,
    *,
    tracer: ResourceTracer | None,
    spec: TrajectorySpec,
) -> None:
    """Apply a compiled step, opening a resource scope around each gate."""
    for gate_index, compiled in enumerate(step.gates):
        scope: Any
        if tracer is None:
            scope = _null_scope()
        else:
            scope = tracer.gate_scope(
                run_family=spec.run_family,
                case=spec.case,
                method=spec.method,
                chi_max=spec.chi_max,
                n_sub=spec.n_sub,
                step=step.index + 1,
                gate_index=gate_index,
                gate_name=compiled.gate.name,
                sites=list(compiled.gate.qubits),
            )
        with scope:
            if len(compiled.gate.qubits) == 1:
                apply_single_qubit_gate(state, compiled.node)
            else:
                apply_two_qubit_gate(state, compiled.node, sim_params)
                # Match the noiseless production ``digital_tjm`` circuit path.
                state.normalize(form="B", decomposition="QR")


class _null_scope:
    """Minimal no-op context manager kept allocation-free in timing loops."""

    def __enter__(self) -> None:
        """Enter the no-op scope."""

    def __exit__(self, *args: object) -> None:
        """Exit the no-op scope."""


def _trajectory_row(
    *,
    spec: TrajectorySpec,
    step: int,
    exact: np.ndarray,
    state: MPS,
    step_runtime_s: float,
    step_peak_parameter_count: int,
    step_peak_bond_dim: int,
    running_peak_parameter_count: int,
    running_peak_bond_dim: int,
) -> dict[str, Any]:
    """Evaluate one trajectory sample against the normalized dense reference."""
    approximate = np.asarray(state.to_vec(), dtype=np.complex128)
    metrics = normalized_state_fidelity(exact[step], approximate)
    return {
        "run_family": spec.run_family,
        "case": spec.case,
        "method": spec.method,
        "chi_max": spec.chi_max,
        "n_sub": spec.n_sub,
        "step": step,
        "time": time_for_step(step),
        **metrics,
        "current_parameter_count": parameter_count(state),
        "current_peak_bond_dim": max(int(tensor.shape[2]) for tensor in state.tensors[:-1])
        if state.length > 1
        else 1,
        "step_peak_parameter_count": step_peak_parameter_count,
        "step_peak_bond_dim": step_peak_bond_dim,
        "peak_parameter_count": running_peak_parameter_count,
        "peak_bond_dim": running_peak_bond_dim,
        "step_runtime_instrumented_s": step_runtime_s,
        "failed": False,
    }


def _run_trajectory_calculation(
    spec: TrajectorySpec,
    exact: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray | None, float]:
    """Run one MPS task and return samples, checkpoints, and optional endpoint."""
    case = CASES[spec.case]
    schedule = build_schedule(case, steps=spec.steps)
    compiled = compile_schedule(schedule, case.n_qubits)
    state = initial_mps(case)
    sim_params = digital_params(spec.method, spec.chi_max, n_sub=spec.n_sub)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    tracer_context: Any = ResourceTracer() if spec.trace_resources else _null_scope()
    with threadpool_limits(limits=1), tracer_context as active:
        tracer = active if isinstance(active, ResourceTracer) else None
        initial_p = parameter_count(state)
        initial_chi = 1
        if tracer is not None:
            tracer.checkpoint(
                state,
                "initial",
                run_family=spec.run_family,
                case=spec.case,
                method=spec.method,
                chi_max=spec.chi_max,
                n_sub=spec.n_sub,
                step=0,
            )
        rows.append(
            _trajectory_row(
                spec=spec,
                step=0,
                exact=exact,
                state=state,
                step_runtime_s=0.0,
                step_peak_parameter_count=initial_p,
                step_peak_bond_dim=initial_chi,
                running_peak_parameter_count=initial_p,
                running_peak_bond_dim=initial_chi,
            )
        )
        running_peak_p = initial_p
        running_peak_chi = initial_chi
        for step_number, step in enumerate(compiled, start=1):
            checkpoint_start = len(tracer.rows) if tracer is not None else 0
            step_started = time.perf_counter()
            _apply_compiled_step(state, step, sim_params, tracer=tracer, spec=spec)
            state.assert_bond_shapes_consistent(max_bond_dim=spec.chi_max)
            step_runtime = time.perf_counter() - step_started
            if tracer is not None:
                tracer.checkpoint(
                    state,
                    "step_end",
                    run_family=spec.run_family,
                    case=spec.case,
                    method=spec.method,
                    chi_max=spec.chi_max,
                    n_sub=spec.n_sub,
                    step=step_number,
                )
                step_checkpoints = tracer.rows[checkpoint_start:]
                step_peak_p = max(int(row["parameter_count"]) for row in step_checkpoints)
                step_peak_chi = max(int(row["peak_bond_dim"]) for row in step_checkpoints)
            else:
                step_peak_p = parameter_count(state)
                step_peak_chi = max(
                    (int(tensor.shape[2]) for tensor in state.tensors[:-1]), default=1
                )
            running_peak_p = max(running_peak_p, step_peak_p)
            running_peak_chi = max(running_peak_chi, step_peak_chi)
            rows.append(
                _trajectory_row(
                    spec=spec,
                    step=step_number,
                    exact=exact,
                    state=state,
                    step_runtime_s=step_runtime,
                    step_peak_parameter_count=step_peak_p,
                    step_peak_bond_dim=step_peak_chi,
                    running_peak_parameter_count=running_peak_p,
                    running_peak_bond_dim=running_peak_chi,
                )
            )

        checkpoints = tracer.checkpoint_rows if tracer is not None else []
        endpoint = (
            np.asarray(state.to_vec(), dtype=np.complex128).copy() if spec.save_final_state else None
        )
    return rows, checkpoints, endpoint, time.perf_counter() - started


def ensure_trajectory_task(
    spec: TrajectorySpec,
    *,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any]:
    """Generate or reuse one complete accuracy trajectory task."""
    payload = _trajectory_payload(spec)
    existing = _existing_task(payload, resume=resume, retry_failed=retry_failed)
    if existing is not None:
        return existing

    task_id = _canonical_hash(payload)
    checkpoint_path = CHECKPOINT_DIR / f"{task_id}.jsonl.gz"
    state_path = STATE_DIR / f"{task_id}.npy"
    checkpoints: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        exact = _load_exact(CASES[spec.case])
        rows, checkpoints, endpoint, elapsed = _run_trajectory_calculation(spec, exact)
        if spec.trace_resources:
            _atomic_jsonl_gz(checkpoint_path, checkpoints)
        if endpoint is not None:
            _atomic_npy(state_path, endpoint)
        return _write_task(
            payload,
            status="success",
            rows=rows,
            checkpoint_path=str(checkpoint_path) if spec.trace_resources else None,
            checkpoint_count=len(checkpoints),
            final_state_path=str(state_path) if endpoint is not None else None,
            elapsed_s=elapsed,
        )
    except Exception as error:  # ruff: ignore[blind-except] - retain OOM/nonconvergence/numerical failures
        if checkpoints:
            _atomic_jsonl_gz(checkpoint_path, checkpoints)
        return _write_task(
            payload,
            status="failed",
            rows=[],
            checkpoint_path=str(checkpoint_path) if checkpoints else None,
            checkpoint_count=len(checkpoints),
            final_state_path=None,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            elapsed_s=time.perf_counter() - started,
        )


def _resolution_specs(case_key: str, n_sub_values: Iterable[int]) -> list[TrajectorySpec]:
    """Construct the preflight tasks for one discriminating 2D case."""
    return [
        TrajectorySpec(
            run_family="resolution",
            case=case_key,
            method="gate_local_2tdvp",
            chi_max=CHI_MAIN,
            n_sub=int(n_sub),
            steps=TDVP_RESOLUTION_CASE_STEPS[case_key],
            trace_resources=False,
            save_final_state=True,
        )
        for n_sub in n_sub_values
    ]


def _endpoint_from_task(task: Mapping[str, Any]) -> np.ndarray:
    """Load one saved resolution endpoint."""
    path_value = task.get("final_state_path")
    if task.get("status") != "success" or not path_value:
        msg = f"Resolution task {task.get('task_id', '<unknown>')} did not produce an endpoint."
        raise RuntimeError(msg)
    path = Path(str(path_value))
    if not path.is_file():
        msg = f"Resolution endpoint is missing: {path}."
        raise FileNotFoundError(msg)
    return np.load(path, allow_pickle=False)


def run_resolution_stage(
    *,
    resume: bool = True,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Select a TDVP gate-path resolution before any main benchmark is run."""
    for case_key in TDVP_RESOLUTION_CASE_STEPS:
        ensure_exact_reference(CASES[case_key], resume=resume, retry_failed=retry_failed)

    per_case: dict[str, Any] = {}
    for case_key in TDVP_RESOLUTION_CASE_STEPS:
        values = list(TDVP_SUBSTEP_CANDIDATES)
        tasks: dict[int, dict[str, Any]] = {}
        for spec in _resolution_specs(case_key, values):
            tasks[spec.n_sub] = ensure_trajectory_task(
                spec, resume=resume, retry_failed=retry_failed
            )
        comparisons: list[dict[str, Any]] = []
        for coarse, fine in itertools.pairwise(values):
            if tasks[coarse].get("status") != "success" or tasks[fine].get("status") != "success":
                continue
            coarse_state = _endpoint_from_task(tasks[coarse])
            fine_state = _endpoint_from_task(tasks[fine])
            metric = normalized_state_fidelity(fine_state, coarse_state)
            comparison = {
                "coarse_n_sub": coarse,
                "fine_n_sub": fine,
                "refinement_infidelity": metric["infidelity_normalized"],
                "phase_aligned_distance": phase_aligned_distance(fine_state, coarse_state),
                "passes": metric["infidelity_normalized"] <= RESOLUTION_INFIDELITY_TOL,
            }
            comparisons.append(comparison)

        production_comparison = next(
            (
                item
                for item in comparisons
                if item["coarse_n_sub"] == TDVP_PRODUCTION_SUBSTEPS
                and item["fine_n_sub"] == 2 * TDVP_PRODUCTION_SUBSTEPS
            ),
            None,
        )
        production_rows = tasks[TDVP_PRODUCTION_SUBSTEPS].get("rows", [])
        fine_rows = tasks[2 * TDVP_PRODUCTION_SUBSTEPS].get("rows", [])
        production_reliable = bool(
            production_rows
            and max(float(row["infidelity_normalized"]) for row in production_rows)
            <= RELIABILITY_THRESHOLD
        )
        fine_reliable = bool(
            fine_rows
            and max(float(row["infidelity_normalized"]) for row in fine_rows)
            <= RELIABILITY_THRESHOLD
        )
        classification_stable = production_reliable == fine_reliable
        endpoint_resources_stable = bool(
            production_rows
            and fine_rows
            and production_rows[-1]["current_parameter_count"]
            == fine_rows[-1]["current_parameter_count"]
            and production_rows[-1]["current_peak_bond_dim"]
            == fine_rows[-1]["current_peak_bond_dim"]
        )
        confirmed_case = bool(
            production_comparison
            and production_comparison["passes"]
            and classification_stable
            and endpoint_resources_stable
        )
        selected = TDVP_PRODUCTION_SUBSTEPS if confirmed_case else None

        per_case[case_key] = {
            "steps": TDVP_RESOLUTION_CASE_STEPS[case_key],
            "time": time_for_step(TDVP_RESOLUTION_CASE_STEPS[case_key]),
            "selected_n_sub": selected,
            "confirmed": confirmed_case,
            "production_reliable": production_reliable,
            "fine_reliable": fine_reliable,
            "threshold_classification_stable": classification_stable,
            "endpoint_resources_stable": endpoint_resources_stable,
            "comparisons": comparisons,
            "task_ids": {str(key): task.get("task_id") for key, task in tasks.items()},
            "task_status": {str(key): task.get("status") for key, task in tasks.items()},
        }

    selected_values = [
        int(item["selected_n_sub"])
        for item in per_case.values()
        if item.get("selected_n_sub") is not None
    ]
    confirmed = len(selected_values) == len(per_case)
    summary = {
        "criterion": (
            f"Confirm the fixed production choice n_sub={TDVP_PRODUCTION_SUBSTEPS} when doubling it changes "
            f"the endpoint by normalized infidelity at most {RESOLUTION_INFIDELITY_TOL:.3g}, preserves the "
            "reliability classification, and preserves endpoint rank and stored parameter count. This is a "
            "fixed-cap stability criterion, not a monotonic-convergence claim."
        ),
        "tolerance": RESOLUTION_INFIDELITY_TOL,
        "per_case": per_case,
        "global_n_sub": max(selected_values) if confirmed else None,
        "confirmed": confirmed,
        "created_utc": utc_now(),
    }
    _atomic_json(OUTPUT_DIR / "resolution_summary.json", summary)
    return summary


def selected_tdvp_substeps() -> int:
    """Load the confirmed global TDVP substep count."""
    path = OUTPUT_DIR / "resolution_summary.json"
    if not path.is_file():
        msg = "TDVP resolution has not been selected; run --stage resolution first."
        raise FileNotFoundError(msg)
    summary = _load_json(path)
    if not summary.get("confirmed") or summary.get("global_n_sub") is None:
        msg = "TDVP resolution preflight is unresolved; inspect resolution_summary.json."
        raise RuntimeError(msg)
    return int(summary["global_n_sub"])


def _method_n_sub(method: Method, selected: int) -> int:
    """Return the meaningful substep setting for a method."""
    return selected if method == "gate_local_2tdvp" else 1


def run_trajectory_stage(
    *,
    resume: bool = True,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """Run the four fixed-cap circuit comparisons at ``chi_max=32``."""
    selected = selected_tdvp_substeps()
    for case in CASES.values():
        ensure_exact_reference(case, resume=resume, retry_failed=retry_failed)
    tasks: list[dict[str, Any]] = []
    for case_key in CASE_KEYS:
        for method in METHODS:
            spec = TrajectorySpec(
                run_family="trajectories",
                case=case_key,
                method=method,
                chi_max=CHI_MAIN,
                n_sub=_method_n_sub(method, selected),
                steps=N_STEPS,
            )
            task = ensure_trajectory_task(spec, resume=resume, retry_failed=retry_failed)
            tasks.append(task)
    return tasks


def _successful_covering_task(spec: TrajectorySpec) -> dict[str, Any] | None:
    """Find a successful longer trajectory that can serve this frontier task."""
    for path in (TASKS_DIR / "trajectories").glob("*.json"):
        task = _load_json(path)
        task_spec = task.get("payload", {}).get("spec", {})
        if (
            task.get("status") == "success"
            and task_spec.get("case") == spec.case
            and task_spec.get("method") == spec.method
            and int(task_spec.get("chi_max", -1)) == spec.chi_max
            and int(task_spec.get("n_sub", -1)) == spec.n_sub
            and int(task_spec.get("steps", -1)) >= spec.steps
        ):
            return task
    return None


def run_frontier_stage(
    *,
    resume: bool = True,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """Run the common-grid 2D-Ising resource frontier through ``t=1.5``."""
    selected = selected_tdvp_substeps()
    case = CASES[FRONTIER_CASE_KEY]
    ensure_exact_reference(case, resume=resume, retry_failed=retry_failed)
    tasks: list[dict[str, Any]] = []
    for method in METHODS:
        for chi in CHI_GRID:
            spec = TrajectorySpec(
                run_family="frontier",
                case=FRONTIER_CASE_KEY,
                method=method,
                chi_max=chi,
                n_sub=_method_n_sub(method, selected),
                steps=FRONTIER_STEPS,
            )
            covering = _successful_covering_task(spec)
            if covering is not None:
                tasks.append(covering)
                continue
            task = ensure_trajectory_task(spec, resume=resume, retry_failed=retry_failed)
            tasks.append(task)
    return tasks


def _timing_payload(*, method: Method, chi: int, n_sub: int, target_step: int, repeat: int) -> dict[str, Any]:
    """Return the identity of one uninstrumented timing task."""
    case = CASES[FRONTIER_CASE_KEY]
    schedule = build_schedule(case, steps=target_step)
    return _task_payload(
        "timing",
        case=case.key,
        method=method,
        chi_max=chi,
        n_sub=n_sub,
        target_step=target_step,
        repeat=repeat,
        protocol=protocol_metadata(case, schedule),
        instrumentation=False,
        numerical={
            "svd_threshold": SVD_THRESHOLD,
            "krylov_tol": KRYLOV_TOL,
            "trunc_mode": TRUNC_MODE,
            "threads": 1,
        },
    )


def _run_uninstrumented_timing(
    *,
    method: Method,
    chi: int,
    n_sub: int,
    target_step: int,
) -> tuple[float, dict[str, float]]:
    """Time only MPS gate application and validate the resulting endpoint."""
    case = CASES[FRONTIER_CASE_KEY]
    schedule = build_schedule(case, steps=target_step)
    compiled = compile_schedule(schedule, case.n_qubits)
    state = initial_mps(case)
    sim_params = digital_params(method, chi, n_sub=n_sub)
    spec = TrajectorySpec("frontier", case.key, method, chi, n_sub, target_step, False)
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        for step in compiled:
            _apply_compiled_step(state, step, sim_params, tracer=None, spec=spec)
        elapsed = time.perf_counter() - started
    exact = _load_exact(case)
    metrics = normalized_state_fidelity(exact[target_step], state.to_vec())
    return elapsed, metrics


def _ensure_timing_task(
    *,
    method: Method,
    chi: int,
    n_sub: int,
    target_step: int,
    repeat: int,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any]:
    """Run or reuse one warm-up/measured timing repeat."""
    payload = _timing_payload(
        method=method,
        chi=chi,
        n_sub=n_sub,
        target_step=target_step,
        repeat=repeat,
    )
    existing = _existing_task(payload, resume=resume, retry_failed=retry_failed)
    if existing is not None:
        return existing
    started = time.perf_counter()
    try:
        elapsed, metrics = _run_uninstrumented_timing(
            method=method,
            chi=chi,
            n_sub=n_sub,
            target_step=target_step,
        )
        return _write_task(
            payload,
            status="success",
            runtime_s=elapsed,
            is_warmup=repeat < 0,
            endpoint_metrics=metrics,
            elapsed_wall_s=time.perf_counter() - started,
        )
    except Exception as error:  # ruff: ignore[blind-except] - retain timing failures
        return _write_task(
            payload,
            status="failed",
            is_warmup=repeat < 0,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            elapsed_wall_s=time.perf_counter() - started,
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_timing_stage(
    *,
    resume: bool = True,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """Time the observed minimum-resource reliable frontier selections."""
    from experiments.circuit_benchmarks.analyze import aggregate_results

    aggregate_results(compute_schmidt=False)
    selected_path = OUTPUT_DIR / "frontier_selected.csv"
    selections = _read_csv(selected_path)
    selected_n_sub = selected_tdvp_substeps()
    tasks: list[dict[str, Any]] = []
    for row in selections:
        if str(row.get("missing", "")).lower() == "true":
            continue
        method = str(row["method"])
        if method not in METHODS:
            msg = f"Unknown selected method {method!r}."
            raise ValueError(msg)
        typed_method: Method = method  # type: ignore[assignment]
        chi = int(row["selected_chi_max"])
        target_step = int(row["target_step"])
        n_sub = _method_n_sub(typed_method, selected_n_sub)
        for warmup_index in range(TIMING_WARMUPS):
            repeat = -(warmup_index + 1)
            tasks.append(
                _ensure_timing_task(
                    method=typed_method,
                    chi=chi,
                    n_sub=n_sub,
                    target_step=target_step,
                    repeat=repeat,
                    resume=resume,
                    retry_failed=retry_failed,
                )
            )
        tasks.extend(_ensure_timing_task(
                    method=typed_method,
                    chi=chi,
                    n_sub=n_sub,
                    target_step=target_step,
                    repeat=repeat,
                    resume=resume,
                    retry_failed=retry_failed,
                ) for repeat in range(TIMING_REPEATS))
    aggregate_results(compute_schmidt=False)
    return tasks


def _report_failures(tasks: Sequence[Mapping[str, Any]], *, fail_fast: bool) -> None:
    """Print retained failures and optionally make the stage fail."""
    failures = [task for task in tasks if task.get("status") != "success"]
    for task in failures:
        task.get("payload", {})
    if failures and fail_fast:
        msg = f"{len(failures)} task(s) failed; records were retained in {TASKS_DIR}."
        raise RuntimeError(msg)


def main(argv: list[str] | None = None) -> int:
    """Run one or all resumable campaign stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "exact",
            "resolution",
            "trajectories",
            "frontier",
            "timing",
            "aggregate",
            "all",
        ),
    )
    parser.add_argument("--no-resume", action="store_true", help="Recompute matching tasks.")
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry rather than reuse retained failed tasks."
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Return an error after a stage containing retained failures.",
    )
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume
    tasks: list[dict[str, Any]] = []
    if args.stage in {"exact", "all"}:
        tasks.extend(run_exact_stage(resume=resume, retry_failed=args.retry_failed))
        _report_failures(tasks, fail_fast=args.fail_fast)
    if args.stage in {"resolution", "all"}:
        summary = run_resolution_stage(resume=resume, retry_failed=args.retry_failed)
        if not summary["confirmed"]:
            write_manifest(last_stage="resolution")
            return 1
    if args.stage in {"trajectories", "all"}:
        stage_tasks = run_trajectory_stage(resume=resume, retry_failed=args.retry_failed)
        tasks.extend(stage_tasks)
        _report_failures(stage_tasks, fail_fast=args.fail_fast)
    if args.stage in {"frontier", "all"}:
        stage_tasks = run_frontier_stage(resume=resume, retry_failed=args.retry_failed)
        tasks.extend(stage_tasks)
        _report_failures(stage_tasks, fail_fast=args.fail_fast)
    if args.stage in {"timing", "all"}:
        stage_tasks = run_timing_stage(resume=resume, retry_failed=args.retry_failed)
        tasks.extend(stage_tasks)
        _report_failures(stage_tasks, fail_fast=args.fail_fast)
    if args.stage in {"aggregate", "all"}:
        from experiments.circuit_benchmarks.analyze import aggregate_results

        aggregate_results(compute_schmidt=True)
    write_manifest(last_stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
