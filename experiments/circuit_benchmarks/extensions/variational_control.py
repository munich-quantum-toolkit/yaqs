# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Variational-MPO accuracy control for the fixed-horizon 4x4 Ising circuit.

The control is deliberately separate from the three-method plotting and timing
pipelines.  Adjacent gates use the current MPO two-site update, which already
performs the corresponding local optimum; separated gates use the global
variational endpoint fit.

Run from the repository root with::

    uv run python -m experiments.circuit_benchmarks.extensions.variational_control
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import time
from functools import cache
from pathlib import Path
from typing import Any

# Freeze numerical thread pools before importing NumPy/SciPy.  The primary
# Figure 4 timings use one numerical thread, so every variational control must
# use and record the same constraint.
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
from experiments.circuit_benchmarks import run as benchmark_run
from experiments.circuit_benchmarks.circuits import build_schedule
from experiments.circuit_benchmarks.config import CASES
from experiments.circuit_benchmarks.config import OUTPUT_DIR as BENCHMARK_OUTPUT_DIR
from experiments.variational_mpo import apply_variational_mpo_node
from mqt.yaqs.core.data_structures import mpo as mpo_module
from mqt.yaqs.core.data_structures import mps as mps_module
from mqt.yaqs.core.methods import decompositions as decompositions_module
from mqt.yaqs.digital import digital_tjm
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate, apply_two_qubit_gate

OUTPUT_DIR = BENCHMARK_OUTPUT_DIR / "variational_mpo_control"
ROWS_PATH = OUTPUT_DIR / "circuit_rows.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "circuit_fit_diagnostics.csv.gz"
SUMMARY_PATH = OUTPUT_DIR / "comparison_summary.json"
SUMMARY_MD_PATH = OUTPUT_DIR / "comparison_summary.md"
CAMPAIGN_ID = "variational_mpo_ising_fixed_horizon_v3"
CASE_KEY = "ising_2d"
TARGET_STEP = 15
CAPS = (4, 8, 16)
MAX_SWEEPS = 32
RETRY_MAX_SWEEPS = 128
MONOTONICITY_TOLERANCE = 2e-12

ROW_FIELDS = (
    "campaign_id",
    "case",
    "chi_max",
    "step",
    "method",
    "infidelity_normalized",
    "fidelity_normalized",
    "norm_approx",
    "norm_drift",
    "final_max_bond",
    "final_parameter_count",
    "peak_parameter_count",
    "cumulative_runtime_s",
    "phase_distance_variational_to_mpo",
    "core_source_sha256",
    "control_source_sha256",
    "benchmark_source_sha256",
    "exact_reference_sha256",
)

DIAGNOSTIC_FIELDS = (
    "chi_max",
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
    "initializer_runtimes_s",
    "initializer_converged",
    "best_initializer",
    "rejected_nonimproving_updates",
    "target_max_bond",
    "target_parameter_count",
    "fit_runtime_s",
    "fidelity_to_target",
    "objective_trace",
)


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _core_source_hash() -> str:
    from experiments import variational_mpo

    return _source_hash(Path(variational_mpo.__file__))


def _control_source_hash() -> str:
    return _source_hash(Path(__file__))


@cache
def _benchmark_source_hash() -> str:
    """Fingerprint shared protocol and production update sources used here."""
    modules = (
        benchmark_common,
        benchmark_circuits,
        benchmark_config,
        benchmark_run,
        mpo_module,
        mps_module,
        decompositions_module,
        digital_tjm,
    )
    digest = hashlib.sha256()
    for module in modules:
        path = Path(module.__file__).resolve()
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@cache
def _exact_reference_hash() -> str:
    path = BENCHMARK_OUTPUT_DIR / "exact" / f"{CASE_KEY}.npy"
    return _source_hash(path)


def _checkpoint_identity() -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "case": CASE_KEY,
        "target_step": TARGET_STEP,
        "threads": THREADS,
        "core_source_sha256": _core_source_hash(),
        "control_source_sha256": _control_source_hash(),
        "benchmark_source_sha256": _benchmark_source_hash(),
        "exact_reference_sha256": _exact_reference_hash(),
    }


def _thread_metadata() -> dict[str, Any]:
    """Return and validate the numerical thread configuration used for timing."""
    pools = threadpool_info()
    invalid = [
        pool
        for pool in pools
        if pool.get("user_api") in {"blas", "openmp"} and int(pool.get("num_threads", -1)) != THREADS
    ]
    if invalid:
        details = ", ".join(
            f"{pool.get('internal_api', 'unknown')}={pool.get('num_threads', 'unknown')}" for pool in invalid
        )
        msg = f"Variational controls require one numerical thread; found {details}."
        raise RuntimeError(msg)
    return {
        "threads": THREADS,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "threadpools": pools,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_gzip_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normalized_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_vector = np.asarray(first, dtype=np.complex128).reshape(-1)
    second_vector = np.asarray(second, dtype=np.complex128).reshape(-1)
    first_vector = first_vector / np.linalg.norm(first_vector)
    second_vector = second_vector / np.linalg.norm(second_vector)
    return benchmark_common.phase_aligned_distance(first_vector, second_vector)


def _state_row(
    *,
    chi: int,
    step: int,
    method: str,
    state,
    exact: np.ndarray,
    distance: float,
    peak_parameter_count: int,
    cumulative_runtime_s: float,
) -> dict[str, Any]:
    metrics = benchmark_common.normalized_state_fidelity(exact, state.to_vec())
    profile = benchmark_common.bond_profile(state)
    return {
        "campaign_id": CAMPAIGN_ID,
        "case": CASE_KEY,
        "chi_max": chi,
        "step": step,
        "method": method,
        "infidelity_normalized": metrics["infidelity_normalized"],
        "fidelity_normalized": metrics["fidelity_normalized"],
        "norm_approx": metrics["norm_approx"],
        "norm_drift": metrics["norm_drift"],
        "final_max_bond": max(profile),
        "final_parameter_count": benchmark_common.parameter_count(state),
        "peak_parameter_count": peak_parameter_count,
        "cumulative_runtime_s": cumulative_runtime_s,
        "phase_distance_variational_to_mpo": distance,
        "core_source_sha256": _core_source_hash(),
        "control_source_sha256": _control_source_hash(),
        "benchmark_source_sha256": _benchmark_source_hash(),
        "exact_reference_sha256": _exact_reference_hash(),
    }


def _run_cap(chi: int, exact_trajectory: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case = CASES[CASE_KEY]
    schedule = build_schedule(case)[:TARGET_STEP]
    compiled = benchmark_common.compile_schedule(schedule, case.n_qubits)
    compression_params = benchmark_common.digital_params("mpo_contract_compress", chi, n_sub=1)
    mpo_state = benchmark_common.initial_mps(case)
    variational_state = benchmark_common.initial_mps(case)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    variational_runtime_s = 0.0
    variational_peak_parameters = benchmark_common.parameter_count(variational_state)

    rows.append(
        _state_row(
            chi=chi,
            step=0,
            method="mpo_contract_compress",
            state=mpo_state,
            exact=exact_trajectory[0],
            distance=0.0,
            peak_parameter_count=benchmark_common.parameter_count(mpo_state),
            cumulative_runtime_s=float("nan"),
        )
    )
    rows.append(
        _state_row(
            chi=chi,
            step=0,
            method="variational_mpo",
            state=variational_state,
            exact=exact_trajectory[0],
            distance=0.0,
            peak_parameter_count=variational_peak_parameters,
            cumulative_runtime_s=variational_runtime_s,
        )
    )

    for step_number, step in enumerate(compiled, start=1):
        for gate_index, compiled_gate in enumerate(step.gates):
            if len(compiled_gate.gate.qubits) == 1:
                apply_single_qubit_gate(mpo_state, compiled_gate.node)
                started = time.perf_counter()
                apply_single_qubit_gate(variational_state, compiled_gate.node)
                variational_runtime_s += time.perf_counter() - started
                variational_peak_parameters = max(
                    variational_peak_parameters,
                    benchmark_common.parameter_count(variational_state),
                )
                continue

            apply_two_qubit_gate(mpo_state, compiled_gate.node, compression_params)
            mpo_state.normalize(form="B", decomposition="QR")

            q0, q1 = compiled_gate.gate.qubits
            if abs(q0 - q1) == 1:
                started = time.perf_counter()
                apply_two_qubit_gate(variational_state, compiled_gate.node, compression_params)
                variational_state.normalize(form="B", decomposition="QR")
                variational_runtime_s += time.perf_counter() - started
                variational_peak_parameters = max(
                    variational_peak_parameters,
                    benchmark_common.parameter_count(variational_state),
                )
                continue

            started = time.perf_counter()
            result = apply_variational_mpo_node(
                variational_state,
                compiled_gate.node,
                compression_params=compression_params,
                max_sweeps=MAX_SWEEPS,
            )
            retried = False
            if not result.converged:
                retried = True
                result = apply_variational_mpo_node(
                    variational_state,
                    compiled_gate.node,
                    compression_params=compression_params,
                    max_sweeps=RETRY_MAX_SWEEPS,
                )
            variational_runtime_s += time.perf_counter() - started
            if not result.converged:
                msg = (
                    f"Variational fit failed to converge at chi={chi}, step={step_number}, "
                    f"gate={gate_index}, sites=({q0}, {q1})."
                )
                raise RuntimeError(msg)
            if any(np.diff(result.objective_trace) > MONOTONICITY_TOLERANCE) or any(
                np.diff(result.update_trace) > MONOTONICITY_TOLERANCE
            ):
                msg = f"Nonmonotone variational objective at chi={chi}, step={step_number}."
                raise RuntimeError(msg)
            variational_state = result.state
            variational_peak_parameters = max(
                variational_peak_parameters,
                benchmark_common.parameter_count(variational_state),
                result.target_parameter_count,
            )
            diagnostics.append(
                {
                    "chi_max": chi,
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
                    "initializer_runtimes_s": json.dumps(result.initializer_runtimes_s, sort_keys=True),
                    "initializer_converged": json.dumps(result.initializer_converged, sort_keys=True),
                    "best_initializer": result.best_initializer,
                    "rejected_nonimproving_updates": result.rejected_nonimproving_updates,
                    "target_max_bond": result.target_max_bond,
                    "target_parameter_count": result.target_parameter_count,
                    "fit_runtime_s": result.runtime_s,
                    "fidelity_to_target": result.fidelity_to_target,
                    "objective_trace": json.dumps(result.objective_trace),
                }
            )

        distance = _normalized_distance(mpo_state.to_vec(), variational_state.to_vec())
        rows.append(
            _state_row(
                chi=chi,
                step=step_number,
                method="mpo_contract_compress",
                state=mpo_state,
                exact=exact_trajectory[step_number],
                distance=distance,
                peak_parameter_count=benchmark_common.parameter_count(mpo_state),
                cumulative_runtime_s=float("nan"),
            )
        )
        rows.append(
            _state_row(
                chi=chi,
                step=step_number,
                method="variational_mpo",
                state=variational_state,
                exact=exact_trajectory[step_number],
                distance=distance,
                peak_parameter_count=variational_peak_parameters,
                cumulative_runtime_s=variational_runtime_s,
            )
        )
        print(
            f"chi={chi} step={step_number}/{TARGET_STEP} fits={len(diagnostics)} distance={distance:.3e}",
            flush=True,
        )
    return rows, diagnostics


def _read_existing_cap(chi: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    cap_path = OUTPUT_DIR / f"chi_{chi}.json"
    if not cap_path.is_file():
        return None
    payload = json.loads(cap_path.read_text(encoding="utf-8"))
    for key, value in _checkpoint_identity().items():
        if payload.get(key) != value:
            return None
    if int(payload.get("chi_max", -1)) != chi:
        return None
    return payload["rows"], payload["diagnostics"]


def run(caps: tuple[int, ...], *, resume: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run selected caps, checkpointing after each completed trajectory."""
    timing_metadata = _thread_metadata()
    case = CASES[CASE_KEY]
    exact_path = BENCHMARK_OUTPUT_DIR / "exact" / f"{CASE_KEY}.npy"
    exact_trajectory = benchmark_run._load_exact(case)
    if exact_trajectory.shape[0] <= TARGET_STEP:
        msg = f"Dense reference {exact_path} does not reach step {TARGET_STEP}."
        raise RuntimeError(msg)

    for chi in caps:
        existing = _read_existing_cap(chi) if resume else None
        if existing is None:
            rows, diagnostics = _run_cap(chi, exact_trajectory)
            payload = {
                **_checkpoint_identity(),
                "chi_max": chi,
                "timing_metadata": timing_metadata,
                "rows": rows,
                "diagnostics": diagnostics,
            }
            _write_json(OUTPUT_DIR / f"chi_{chi}.json", payload)

    all_rows: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    for chi in sorted(set(CAPS) | set(caps)):
        existing = _read_existing_cap(chi)
        if existing is None:
            continue
        rows, diagnostics = existing
        all_rows.extend(rows)
        all_diagnostics.extend(diagnostics)
    _write_csv(ROWS_PATH, all_rows, ROW_FIELDS)
    _write_gzip_csv(DIAGNOSTICS_PATH, all_diagnostics, DIAGNOSTIC_FIELDS)
    return all_rows, all_diagnostics


def summarize(rows: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize worst-prefix and endpoint infidelity for each requested cap."""
    cap_summaries: dict[str, Any] = {}
    for chi in sorted({int(row["chi_max"]) for row in rows}):
        cap_rows = [row for row in rows if int(row["chi_max"]) == chi]
        method_summary: dict[str, Any] = {}
        for method in ("mpo_contract_compress", "variational_mpo"):
            method_rows = [row for row in cap_rows if row["method"] == method]
            method_summary[method] = {
                "worst_prefix_infidelity": max(float(row["infidelity_normalized"]) for row in method_rows),
                "endpoint_infidelity": float(
                    next(row for row in method_rows if int(row["step"]) == TARGET_STEP)["infidelity_normalized"]
                ),
                "endpoint_parameter_count": int(
                    next(row for row in method_rows if int(row["step"]) == TARGET_STEP)["final_parameter_count"]
                ),
                "peak_parameter_count": max(int(row["peak_parameter_count"]) for row in method_rows),
            }
            if method == "variational_mpo":
                method_summary[method]["runtime_s"] = float(
                    next(row for row in method_rows if int(row["step"]) == TARGET_STEP)["cumulative_runtime_s"]
                )
        fit_rows = [row for row in diagnostics if int(row["chi_max"]) == chi]
        cap_summaries[str(chi)] = {
            **method_summary,
            "variational_fits": len(fit_rows),
            "all_selected_fits_converged": all(str(row["converged"]).lower() == "true" for row in fit_rows),
            "maximum_sweeps": max(int(row["sweeps"]) for row in fit_rows),
            "retried_fits": sum(str(row["retried_with_128_sweeps"]).lower() == "true" for row in fit_rows),
            "maximum_phase_distance_to_mpo": max(float(row["phase_distance_variational_to_mpo"]) for row in cap_rows),
        }

    summary = {
        **_checkpoint_identity(),
        "caps": cap_summaries,
        "source_data": str(ROWS_PATH),
        "fit_diagnostics": str(DIAGNOSTICS_PATH),
        "timing_repeats_per_cap": 1,
        "timing_scope": (
            "one complete one-thread observation per cap for the current two-initializer implementation; "
            "no repeated timing baseline and no fitted scaling exponent"
        ),
        "thread_metadata": _thread_metadata(),
    }
    _write_json(SUMMARY_PATH, summary)
    lines = [
        "# Variational MPO circuit accuracy control",
        "",
        "Each runtime is one complete one-thread observation for the current two-initializer implementation. ",
        "The cap dependence is shown without repeated timing bars or a fitted scaling exponent.",
        "",
        "| chi_max | MPO E_star | Variational E_star | MPO endpoint | Variational endpoint | "
        "Runtime (s) | Fits | Max sweeps |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for chi, cap in cap_summaries.items():
        mpo = cap["mpo_contract_compress"]
        variational = cap["variational_mpo"]
        lines.append(
            f"| {chi} | {mpo['worst_prefix_infidelity']:.8e} | "
            f"{variational['worst_prefix_infidelity']:.8e} | {mpo['endpoint_infidelity']:.8e} | "
            f"{variational['endpoint_infidelity']:.8e} | {variational['runtime_s']:.6g} | "
            f"{cap['variational_fits']} | "
            f"{cap['maximum_sweeps']} |"
        )
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", nargs="+", type=int, default=list(CAPS))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    caps = tuple(dict.fromkeys(int(cap) for cap in args.caps))
    with threadpool_limits(limits=THREADS):
        rows, diagnostics = run(caps, resume=not args.no_resume)
        summary = summarize(rows, diagnostics)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
