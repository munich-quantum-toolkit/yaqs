# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Narrow SVD-threshold control for the final circuit-benchmark protocol.

This is deliberately a bounded validation rather than a new benchmark
campaign: one uninstrumented accuracy trajectory is run for each method at the
cap selected for the fixed-horizon comparison, while only the discarded-weight
threshold is varied.  No timing repetitions are performed.

Run from the repository root with::

    uv run python -m experiments.circuit_benchmarks.extensions.svd_threshold_control
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
THREADS = 1
for _thread_variable in THREAD_VARIABLES:
    os.environ[_thread_variable] = str(THREADS)

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.circuit_benchmarks import circuits as benchmark_circuits
from experiments.circuit_benchmarks import common as benchmark_common
from experiments.circuit_benchmarks import config as benchmark_config
from experiments.circuit_benchmarks.circuits import build_schedule
from experiments.circuit_benchmarks.config import CASES
from experiments.circuit_benchmarks.config import OUTPUT_DIR as BENCHMARK_OUTPUT_DIR
from mqt.yaqs.core.data_structures import mpo as mpo_module
from mqt.yaqs.core.data_structures import mps as mps_module
from mqt.yaqs.core.data_structures.simulation_parameters import DigitalSimParams
from mqt.yaqs.core.libraries import gate_library as gate_library_module
from mqt.yaqs.core.linalg import svd_utils as svd_utils_module
from mqt.yaqs.core.methods import decompositions as decompositions_module
from mqt.yaqs.core.methods.tdvp import sweep_utils as sweep_utils_module
from mqt.yaqs.digital import digital_tjm

CAMPAIGN_ID = "svd_threshold_control_final_protocol_v1"
CASE_KEY = "ising_2d"
TARGET_STEP = 15
THRESHOLDS = (1e-14, 1e-13, 1e-12, 1e-9)
METHOD_CAPS = {
    "gate_local_2tdvp": 28,
    "mpo_contract_compress": 26,
    "tebd_swap": 32,
}
METHOD_SUBSTEPS = {
    "gate_local_2tdvp": benchmark_config.TDVP_PRODUCTION_SUBSTEPS,
    "mpo_contract_compress": 1,
    "tebd_swap": 1,
}

OUTPUT_DIR = BENCHMARK_OUTPUT_DIR / "svd_threshold_control"
ROWS_PATH = OUTPUT_DIR / "trajectory_rows.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary_rows.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

ROW_FIELDS = (
    "campaign_id",
    "case",
    "method",
    "chi_max",
    "n_sub",
    "svd_threshold",
    "step",
    "infidelity_normalized",
    "fidelity_normalized",
    "norm_approx",
    "norm_drift",
    "parameter_count",
    "max_bond_dim",
    "cumulative_runtime_s",
)

SUMMARY_FIELDS = (
    "campaign_id",
    "case",
    "method",
    "chi_max",
    "n_sub",
    "svd_threshold",
    "target_step",
    "endpoint_infidelity",
    "max_infidelity_through_target",
    "endpoint_parameter_count",
    "max_completed_step_parameter_count",
    "endpoint_max_bond_dim",
    "max_completed_step_bond_dim",
    "single_trajectory_runtime_s",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def _implementation_hash() -> str:
    modules = (
        benchmark_common,
        benchmark_circuits,
        benchmark_config,
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


def _git_metadata() -> dict[str, Any]:
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
    ).stdout
    return {
        "git_commit": commit,
        "git_dirty_for_relevant_paths": bool(diff),
        "relevant_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _thread_metadata() -> dict[str, Any]:
    pools = threadpool_info()
    invalid = [
        pool
        for pool in pools
        if pool.get("user_api") in {"blas", "openmp"} and int(pool.get("num_threads", -1)) != THREADS
    ]
    if invalid:
        detail = ", ".join(f"{pool.get('internal_api')}={pool.get('num_threads')}" for pool in invalid)
        raise RuntimeError(f"Threshold control requires one numerical thread; found {detail}.")
    return {
        "threads": THREADS,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "threadpools": pools,
    }


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _params(method: str, chi: int, n_sub: int, threshold: float) -> DigitalSimParams:
    return DigitalSimParams(
        observables=[],
        get_state=True,
        preset="exact",
        max_bond_dim=chi,
        trunc_mode=benchmark_config.TRUNC_MODE,
        svd_threshold=threshold,
        krylov_tol=benchmark_config.KRYLOV_TOL,
        gate_mode=benchmark_config.METHOD_TO_GATE_MODE[method],
        tdvp_sweeps=n_sub,
        tdvp_mode=benchmark_config.TDVP_MODE,
    )


def _run_one(
    method: str,
    chi: int,
    threshold: float,
    exact: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case = CASES[CASE_KEY]
    n_sub = METHOD_SUBSTEPS[method]
    schedule = build_schedule(case, steps=TARGET_STEP)
    compiled = benchmark_common.compile_schedule(schedule, case.n_qubits)
    state = benchmark_common.initial_mps(case)
    params = _params(method, chi, n_sub, threshold)
    cumulative_runtime = 0.0
    rows: list[dict[str, Any]] = []

    for step in range(TARGET_STEP + 1):
        if step:
            started = time.perf_counter()
            benchmark_common.apply_mps_step(state, compiled[step - 1], params)
            cumulative_runtime += time.perf_counter() - started
            state.assert_bond_shapes_consistent(max_bond_dim=chi)
        metrics = benchmark_common.normalized_state_fidelity(exact[step], state.to_vec())
        profile = benchmark_common.bond_profile(state)
        rows.append(
            {
                "campaign_id": CAMPAIGN_ID,
                "case": CASE_KEY,
                "method": method,
                "chi_max": chi,
                "n_sub": n_sub,
                "svd_threshold": threshold,
                "step": step,
                "infidelity_normalized": metrics["infidelity_normalized"],
                "fidelity_normalized": metrics["fidelity_normalized"],
                "norm_approx": metrics["norm_approx"],
                "norm_drift": metrics["norm_drift"],
                "parameter_count": benchmark_common.parameter_count(state),
                "max_bond_dim": max(profile),
                "cumulative_runtime_s": cumulative_runtime,
            }
        )

    summary = {
        "campaign_id": CAMPAIGN_ID,
        "case": CASE_KEY,
        "method": method,
        "chi_max": chi,
        "n_sub": n_sub,
        "svd_threshold": threshold,
        "target_step": TARGET_STEP,
        "endpoint_infidelity": rows[-1]["infidelity_normalized"],
        "max_infidelity_through_target": max(float(row["infidelity_normalized"]) for row in rows),
        "endpoint_parameter_count": rows[-1]["parameter_count"],
        "max_completed_step_parameter_count": max(int(row["parameter_count"]) for row in rows),
        "endpoint_max_bond_dim": rows[-1]["max_bond_dim"],
        "max_completed_step_bond_dim": max(int(row["max_bond_dim"]) for row in rows),
        "single_trajectory_runtime_s": cumulative_runtime,
    }
    return rows, summary


def main() -> int:
    case = CASES[CASE_KEY]
    schedule = build_schedule(case, steps=TARGET_STEP)
    exact_path = BENCHMARK_OUTPUT_DIR / "exact" / f"{CASE_KEY}.npy"
    exact = np.load(exact_path, allow_pickle=False)
    if exact.ndim != 2 or exact.shape[0] < TARGET_STEP + 1 or exact.shape[1] != 2**case.n_qubits:
        raise RuntimeError(f"Unexpected dense-reference shape {exact.shape}.")

    trajectory_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    thread_metadata: dict[str, Any]
    with threadpool_limits(limits=THREADS):
        thread_metadata = _thread_metadata()
        for method, chi in METHOD_CAPS.items():
            for threshold in THRESHOLDS:
                print(
                    f"{method}: chi={chi}, tau={threshold:g}",
                    flush=True,
                )
                rows, summary = _run_one(method, chi, threshold, exact)
                trajectory_rows.extend(rows)
                summary_rows.append(summary)
                _atomic_csv(ROWS_PATH, trajectory_rows, ROW_FIELDS)
                _atomic_csv(SUMMARY_PATH, summary_rows, SUMMARY_FIELDS)
                print(
                    f"  E*={summary['max_infidelity_through_target']:.8g}, "
                    f"P(step)max={summary['max_completed_step_parameter_count']}, "
                    f"runtime={summary['single_trajectory_runtime_s']:.3f}s",
                    flush=True,
                )

    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded current-code SVD-threshold validation; not a timing campaign",
        "case": CASE_KEY,
        "target_step": TARGET_STEP,
        "method_caps": METHOD_CAPS,
        "method_substeps": METHOD_SUBSTEPS,
        "svd_thresholds": list(THRESHOLDS),
        "truncation": {
            "mode": benchmark_config.TRUNC_MODE,
            "threshold_meaning": "unnormalized cumulative discarded squared singular-value weight",
            "cap_applied_after_cutoff_rank": True,
            "minimum_retained_rank": 1,
            "exact_zero_padding": False,
            "gate_mpo_hard_split_cutoff": 1e-14,
        },
        "krylov_tolerance": benchmark_config.KRYLOV_TOL,
        "numerical_precision": "complex128",
        "trajectory_repeats": 1,
        "timing_repeats": 0,
        "timings_for_publication_comparison": False,
        "resource_sampling": (
            "Step 0 and completed Trotter-step endpoints only; these values are not "
            "the transient P_max traced after every state-changing factorization in Figure 4."
        ),
        "circuit_fingerprint": benchmark_circuits.circuit_fingerprint(case, schedule),
        "dense_reference_sha256": _sha256(exact_path),
        "control_source_sha256": _sha256(Path(__file__)),
        "implementation_sha256": _implementation_hash(),
        "outputs": {
            "trajectory_rows": str(ROWS_PATH.relative_to(Path(__file__).resolve().parents[3])),
            "summary_rows": str(SUMMARY_PATH.relative_to(Path(__file__).resolve().parents[3])),
        },
        "output_sha256": {
            "trajectory_rows": _sha256(ROWS_PATH),
            "summary_rows": _sha256(SUMMARY_PATH),
        },
        **thread_metadata,
        **_git_metadata(),
    }
    _atomic_json(MANIFEST_PATH, manifest)
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
