#!/usr/bin/env python3
"""Reproducible L=16 accuracy/runtime and active-bond-cap experiments.

The script reuses the validated matched-parameter rows at epsilon=1e-12,
checkpoints every new method/configuration independently, and stores raw timing
samples as well as all state-derived diagnostics needed by the manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The machine has eight physical/logical cores.  Accelerate backs NumPy's BLAS
# and LAPACK here; these limits make all of them available to contractions and
# SVDs while the timed configurations themselves remain isolated.
CPU_THREADS = os.cpu_count() or 1
for variable in (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = str(CPU_THREADS)

import numpy as np
from scipy.sparse.linalg import expm_multiply

from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.methods import matrix_exponential
from mqt.yaqs.core.methods.bug import bug as bug_evolve
from mqt.yaqs.core.methods.tdvp import primitives
from mqt.yaqs.core.methods.tdvp import tdvp as tdvp_evolve


HERE = Path(__file__).resolve().parent
YAQS_ROOT = HERE.parents[2]
PROJECT_ROOT = YAQS_ROOT
MATCHED_DIR = HERE.parent / "l16_matched_optimized_2026-08-12"
MATCHED_JSON = MATCHED_DIR / "raw_results.json"
BASE_RUNNER = MATCHED_DIR / "run_benchmark.py"

MODELS = ("tfim", "hs")
METHODS = ("bug", "2tdvp")
DT_GRID = (0.01, 0.005, 0.0025)
EPSILON_GRID = (1e-8, 1e-10, 1e-12, 1e-14)
CAP_GRID = (32, 64, 96)
BASELINE_CAP = 512
CAP_DT = 0.005
CAP_EPSILON = 1e-12
TOTAL_TIME = 1.0
KRYLOV_TOL = 1e-12
KRYLOV_MAX_DIM = 25
SCHEMA_VERSION = 1


def utc_now() -> str:
    """Return a stable ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return sha256_bytes(path.read_bytes())


def command_output(command: list[str], *, cwd: Path | None = None) -> str:
    """Run a provenance command and return combined text without failing the run."""
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"ERROR: {exc}\n"
    return completed.stdout + completed.stderr


def load_base_runner() -> Any:
    """Load the already validated common Hamiltonian/state benchmark helpers."""
    spec = importlib.util.spec_from_file_location("l16_matched_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import benchmark helpers from {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_runner()


def parse_csv_floats(value: str) -> tuple[float, ...]:
    """Parse a comma-separated float list."""
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list."""
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    """Parse the benchmark command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "repeat", "caps", "all"), default="all")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--dts", default=",".join(map(str, DT_GRID)))
    parser.add_argument("--epsilons", default=",".join(map(str, EPSILON_GRID)))
    parser.add_argument("--caps", default=",".join(map(str, CAP_GRID)))
    parser.add_argument("--total-time", type=float, default=TOTAL_TIME)
    parser.add_argument("--pilot-samples", type=int, default=1)
    parser.add_argument("--target-samples", type=int, default=3)
    parser.add_argument("--output", type=Path, default=HERE / "raw_results.json")
    parser.add_argument("--no-baseline-import", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically write a compressed NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def safe_float(value: float) -> str:
    """Create a path/key-safe scientific float representation."""
    return format(value, ".8g").replace("+", "p").replace("-", "m").replace(".", "p")


def record_key(model: str, dt: float, epsilon: float, cap: int, method: str) -> str:
    """Return the canonical key for one method/configuration."""
    return f"{model}|dt={dt:.8g}|eps={epsilon:.8g}|cap={cap}|{method}"


def state_filename(model: str, dt: float, epsilon: float, cap: int, method: str) -> str:
    """Return a deterministic final-state archive name."""
    return f"{model}_dt-{safe_float(dt)}_eps-{safe_float(epsilon)}_cap-{cap}_{method}.npz"


def initial_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Create the immutable protocol and empty result container."""
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "protocol": {
            "length": BASE.LENGTH,
            "total_time": args.total_time,
            "models": list(MODELS),
            "methods": list(METHODS),
            "tradeoff_dt_grid": list(parse_csv_floats(args.dts)),
            "tradeoff_epsilon_grid": list(parse_csv_floats(args.epsilons)),
            "tradeoff_cap": BASELINE_CAP,
            "cap_model": "hs",
            "cap_dt": CAP_DT,
            "cap_epsilon": CAP_EPSILON,
            "cap_grid": [*parse_csv_ints(args.caps), BASELINE_CAP],
            "trunc_mode": "relative_discarded_weight",
            "min_keep": 2,
            "initial_chi": BASE.INITIAL_CHI,
            "initial_noise_scale": BASE.NOISE_SCALE,
            "initial_seed": BASE.SEED,
            "max_bond_dimension": BASELINE_CAP,
            "krylov_backend": "shared pure-NumPy matrix-free adaptive Lanczos",
            "krylov_tolerance": KRYLOV_TOL,
            "krylov_max_dimension": KRYLOV_MAX_DIM,
            "cpu_threads_available_to_numeric_kernels": CPU_THREADS,
            "configuration_parallelism_during_timing": 1,
            "pilot_timing_samples": args.pilot_samples,
            "pareto_target_timing_samples": args.target_samples,
            "cap_timing_samples": args.target_samples,
            "timed_region_includes": ["time-evolution kernel calls from t=0 to t=1"],
            "timed_region_excludes": [
                "state and MPO construction",
                "deterministic initial-state padding",
                "exact reference construction/evolution",
                "one-step warm-up",
                "garbage collection",
                "Krylov counters and diagnostics replay",
                "state-vector conversion and metric calculation",
                "file output",
            ],
            "pareto_definition": (
                "Within each model and method, a point is retained if no other tested point "
                "has both no larger pilot/median runtime and no larger infidelity, with at "
                "least one strict inequality."
            ),
        },
        "fixtures": {},
        "records": {},
        "pareto_selection": {},
        "events": [],
    }


def validate_resume_protocol(payload: dict[str, Any], args: argparse.Namespace) -> None:
    """Refuse to mix measurements from incompatible command lines."""
    protocol = payload["protocol"]
    expected = initial_payload(args)["protocol"]
    immutable_fields = (
        "length",
        "total_time",
        "tradeoff_dt_grid",
        "tradeoff_epsilon_grid",
        "tradeoff_cap",
        "cap_dt",
        "cap_epsilon",
        "cap_grid",
        "trunc_mode",
        "min_keep",
        "initial_chi",
        "initial_seed",
        "krylov_tolerance",
        "krylov_max_dimension",
    )
    mismatches = [field for field in immutable_fields if protocol.get(field) != expected.get(field)]
    if mismatches:
        raise ValueError(f"Cannot resume with changed protocol fields: {mismatches}")


def save_provenance(payload: dict[str, Any], output: Path) -> None:
    """Save source/environment snapshots that identify the exact implementation."""
    provenance = output.parent / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    files = {
        "yaqs_git_status.txt": command_output(["git", "status", "--short"], cwd=YAQS_ROOT),
        "yaqs_git_head.txt": command_output(["git", "rev-parse", "HEAD"], cwd=YAQS_ROOT),
        "yaqs_git_branch.txt": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=YAQS_ROOT),
        "yaqs_worktree.patch": command_output(["git", "diff", "--binary"], cwd=YAQS_ROOT),
        "pip_freeze.txt": command_output([sys.executable, "-m", "pip", "freeze"]),
    }
    for name, content in files.items():
        (provenance / name).write_text(content, encoding="utf-8")
    if MATCHED_JSON.exists():
        shutil.copy2(MATCHED_JSON, provenance / "matched_raw_results_source.json")
    info = {
        "captured_utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numeric_kernel_thread_environment": {
            variable: os.environ.get(variable)
            for variable in (
                "VECLIB_MAXIMUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "runner_sha256": sha256_file(Path(__file__)),
        "base_runner_sha256": sha256_file(BASE_RUNNER),
        "matched_json_sha256": sha256_file(MATCHED_JSON) if MATCHED_JSON.exists() else None,
        "provenance_files": {
            name: sha256_file(provenance / name) for name in files
        },
    }
    atomic_write_json(provenance / "environment.json", info)
    payload["provenance"] = info


def make_parameters(dt: float, epsilon: float, cap: int) -> AnalogSimParams:
    """Construct the exact shared BUG/2TDVP numerical policy."""
    return AnalogSimParams(
        elapsed_time=dt,
        dt=dt,
        max_bond_dim=cap,
        trunc_mode="relative_discarded_weight",
        svd_threshold=epsilon,
        krylov_tol=KRYLOV_TOL,
        tdvp_mode="2site",
        get_state=True,
    )


def build_fixture(model: str, total_time: float, payload: dict[str, Any], output: Path) -> dict[str, Any]:
    """Build, save, and verify the shared initial state, MPO, and exact reference."""
    initial = BASE.padded_initial_state(model)
    mpo = BASE.direct_ising_mpo() if model == "tfim" else BASE.direct_haldane_shastry_mpo()
    initial_vector = initial.to_vec().copy()
    hamiltonian = BASE.exact_sparse_hamiltonian(model)
    reference = expm_multiply((-1j * total_time) * hamiltonian, initial_vector)
    reference_z = BASE.z_expectations(reference)
    reference_energy = float(np.vdot(reference, hamiltonian @ reference).real / np.vdot(reference, reference).real)

    arrays: dict[str, np.ndarray] = {
        "initial_state_vector": initial_vector,
        "reference_state_vector": reference,
        "reference_z_expectations": reference_z,
        "reference_energy": np.asarray([reference_energy]),
    }
    for index, tensor in enumerate(initial.tensors):
        arrays[f"initial_mps_tensor_{index:02d}"] = tensor
    for index, tensor in enumerate(mpo.tensors):
        arrays[f"hamiltonian_mpo_tensor_{index:02d}"] = tensor

    fixture_path = output.parent / "fixtures" / f"{model}_fixture.npz"
    if fixture_path.exists():
        with np.load(fixture_path) as saved:
            for name in ("initial_state_vector", "reference_state_vector", "reference_z_expectations"):
                if not np.array_equal(saved[name], arrays[name]):
                    difference = float(np.max(np.abs(saved[name] - arrays[name])))
                    raise ValueError(f"Fixture drift for {model} {name}: max difference {difference:.3e}")
    else:
        atomic_save_npz(fixture_path, **arrays)

    mpo_sparse = mpo.to_sparse_matrix()
    exact_difference = mpo_sparse - hamiltonian
    difference_norm = float(np.sqrt(np.sum(np.abs(exact_difference.data) ** 2))) if exact_difference.nnz else 0.0
    exact_norm = float(np.sqrt(np.sum(np.abs(hamiltonian.data) ** 2)))
    metadata = {
        "path": str(fixture_path.relative_to(output.parent)),
        "sha256": sha256_file(fixture_path),
        "initial_state_vector_sha256": sha256_bytes(initial_vector.tobytes()),
        "reference_state_vector_sha256": sha256_bytes(reference.tobytes()),
        "initial_bond_profile": BASE.bond_profile(initial),
        "mpo_bond_profile": [int(tensor.shape[3]) for tensor in mpo.tensors[:-1]],
        "initial_norm": float(np.vdot(initial_vector, initial_vector).real),
        "reference_norm": float(np.vdot(reference, reference).real),
        "mpo_vs_analytic_relative_frobenius_error": difference_norm / exact_norm,
    }
    existing = payload["fixtures"].get(model)
    if existing is not None and existing != metadata:
        raise ValueError(f"Fixture metadata changed for {model}")
    payload["fixtures"][model] = metadata
    return {
        "initial": initial,
        "mpo": mpo,
        "hamiltonian": hamiltonian,
        "reference": reference,
        "reference_z": reference_z,
        "reference_energy": reference_energy,
    }


def empty_record(model: str, dt: float, epsilon: float, cap: int, method: str, steps: int) -> dict[str, Any]:
    """Create one method/configuration record."""
    return {
        "configuration": {
            "model": model,
            "method": method,
            "dt": dt,
            "steps": steps,
            "epsilon": epsilon,
            "max_bond_dim": cap,
            "trunc_mode": "relative_discarded_weight",
            "min_keep": 2,
        },
        "studies": [],
        "source": "measured",
        "diagnostics": None,
        "timing": {"samples_seconds": [], "events": []},
    }


def get_record(
    payload: dict[str, Any],
    model: str,
    dt: float,
    epsilon: float,
    cap: int,
    method: str,
    steps: int,
    study: str,
) -> dict[str, Any]:
    """Get or create a record and attach it to a named study."""
    key = record_key(model, dt, epsilon, cap, method)
    record = payload["records"].setdefault(key, empty_record(model, dt, epsilon, cap, method, steps))
    if study not in record["studies"]:
        record["studies"].append(study)
        record["studies"].sort()
    return record


def refresh_timing_summary(record: dict[str, Any]) -> None:
    """Update descriptive statistics without discarding raw samples."""
    samples = record["timing"]["samples_seconds"]
    record["timing"]["sample_count"] = len(samples)
    if samples:
        record["timing"].update(
            {
                "median_seconds": statistics.median(samples),
                "minimum_seconds": min(samples),
                "maximum_seconds": max(samples),
                "mean_seconds": statistics.fmean(samples),
                "sample_standard_deviation_seconds": statistics.stdev(samples) if len(samples) > 1 else None,
            }
        )


def import_matched_baseline(payload: dict[str, Any], args: argparse.Namespace) -> None:
    """Import exactly matching epsilon=1e-12 rows from the validated benchmark."""
    if args.no_baseline_import:
        return
    source = json.loads(MATCHED_JSON.read_text(encoding="utf-8"))
    requested_models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    dts = parse_csv_floats(args.dts)
    epsilons = parse_csv_floats(args.epsilons)
    if CAP_EPSILON not in epsilons:
        return
    for model in requested_models:
        for dt in dts:
            dt_key = format(dt, ".8g")
            if dt_key not in source["models"][model]["runs"]:
                continue
            source_run = source["models"][model]["runs"][dt_key]
            for method in METHODS:
                source_method = source_run["methods"][method]
                record = get_record(
                    payload,
                    model,
                    dt,
                    CAP_EPSILON,
                    BASELINE_CAP,
                    method,
                    source_run["steps"],
                    "tradeoff",
                )
                record["source"] = "imported_validated_matched_benchmark"
                record["source_path"] = str(MATCHED_JSON.relative_to(PROJECT_ROOT))
                record["source_sha256"] = sha256_file(MATCHED_JSON)
                record["diagnostics"] = {
                    name: source_method.get(name)
                    for name in (
                        "krylov_calls",
                        "krylov_operator_applications",
                        "max_chi",
                        "final_bond_profile",
                        "first_step_bug_checkpoints",
                        "norm",
                        "phase_aligned_state_error",
                        "infidelity",
                        "max_abs_z_error",
                        "rms_z_error",
                        "energy_abs_error",
                    )
                }
                record["timing"]["samples_seconds"] = list(source_method["runtime_samples_seconds"])
                record["timing"]["events"] = [
                    {
                        "sample_index": index,
                        "duration_seconds": duration,
                        "stage": "matched_baseline_import",
                        "order_position": None,
                    }
                    for index, duration in enumerate(source_method["runtime_samples_seconds"], start=1)
                ]
                refresh_timing_summary(record)

    if "hs" in requested_models:
        source_run = source["models"]["hs"]["runs"][format(CAP_DT, ".8g")]
        for method in METHODS:
            record = get_record(
                payload,
                "hs",
                CAP_DT,
                CAP_EPSILON,
                BASELINE_CAP,
                method,
                source_run["steps"],
                "cap",
            )
            # The record was already populated above when dt=0.005 is in the tradeoff grid.
            if record["diagnostics"] is None:
                source_method = source_run["methods"][method]
                record["source"] = "imported_validated_matched_benchmark"
                record["source_path"] = str(MATCHED_JSON.relative_to(PROJECT_ROOT))
                record["source_sha256"] = sha256_file(MATCHED_JSON)
                record["diagnostics"] = {
                    name: source_method.get(name)
                    for name in (
                        "krylov_calls",
                        "krylov_operator_applications",
                        "max_chi",
                        "final_bond_profile",
                        "first_step_bug_checkpoints",
                        "norm",
                        "phase_aligned_state_error",
                        "infidelity",
                        "max_abs_z_error",
                        "rms_z_error",
                        "energy_abs_error",
                    )
                }
                record["timing"]["samples_seconds"] = list(source_method["runtime_samples_seconds"])
                refresh_timing_summary(record)


def calculate_diagnostics(
    model: str,
    method: str,
    dt: float,
    epsilon: float,
    cap: int,
    steps: int,
    fixture: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Run one untimed diagnostic replay and save its final dense state."""
    params = make_parameters(dt, epsilon, cap)
    state, stats, max_chi, checkpoints = BASE.run_to_final(
        method,
        fixture["initial"],
        fixture["mpo"],
        params,
        steps,
        diagnostics=True,
    )
    vector = state.to_vec().copy()
    z_values = BASE.z_expectations(vector)
    energy = float(
        np.vdot(vector, fixture["hamiltonian"] @ vector).real / np.vdot(vector, vector).real
    )
    overlap = np.vdot(fixture["reference"], vector)
    state_path = output.parent / "states" / state_filename(model, dt, epsilon, cap, method)
    atomic_save_npz(state_path, final_state_vector=vector, z_expectations=z_values)
    diagnostics = {
        "krylov_calls": stats["calls"],
        "krylov_operator_applications": stats["operator_applications"],
        "max_chi": max_chi,
        "final_bond_profile": BASE.bond_profile(state),
        "first_step_bug_checkpoints": checkpoints,
        "norm": float(np.vdot(vector, vector).real),
        "reference_overlap_real": float(overlap.real),
        "reference_overlap_imag": float(overlap.imag),
        "phase_aligned_state_error": BASE.phase_error(fixture["reference"], vector),
        "infidelity": BASE.infidelity(fixture["reference"], vector),
        "max_abs_z_error": float(np.max(np.abs(z_values - fixture["reference_z"]))),
        "rms_z_error": float(np.sqrt(np.mean((z_values - fixture["reference_z"]) ** 2))),
        "energy": energy,
        "energy_abs_error": abs(energy - fixture["reference_energy"]),
        "state_vector_path": str(state_path.relative_to(output.parent)),
        "state_vector_sha256": sha256_file(state_path),
        "completed_utc": utc_now(),
    }
    expected_calls = (32 if method == "bug" else 57) * steps
    if diagnostics["krylov_calls"] != expected_calls:
        raise AssertionError(
            f"Unexpected Krylov calls for {model}/{method}: "
            f"{diagnostics['krylov_calls']} != {expected_calls}"
        )
    if max_chi > cap:
        raise AssertionError(f"Bond cap violated for {model}/{method}: {max_chi} > {cap}")
    if method == "bug":
        expected_stages = {
            "first_half_sweep",
            "first_compression",
            "second_half_sweep",
            "second_compression",
        }
        if set(checkpoints) != expected_stages:
            raise AssertionError(f"Missing BUG checkpoint stages: {set(checkpoints)}")
        retained = checkpoints["first_compression"] + checkpoints["second_compression"]
        if min(retained) < 2:
            raise AssertionError(f"BUG min_keep violated: minimum retained rank {min(retained)}")
    return diagnostics


def warm_method(method: str, fixture: dict[str, Any], params: AnalogSimParams) -> None:
    """Perform an excluded one-step warm-up."""
    evolve = bug_evolve if method == "bug" else tdvp_evolve
    evolve(deepcopy(fixture["initial"]), deepcopy(fixture["mpo"]), params)


def run_configuration_pair(
    payload: dict[str, Any],
    output: Path,
    fixture: dict[str, Any],
    *,
    model: str,
    dt: float,
    epsilon: float,
    cap: int,
    study: str,
    timing_targets: dict[str, int],
    stage: str,
) -> None:
    """Complete diagnostics and requested timing counts for one BUG/2TDVP pair."""
    steps = round(payload["protocol"]["total_time"] / dt)
    if not math.isclose(steps * dt, payload["protocol"]["total_time"], rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Total time is not divisible by dt={dt}")
    records = {
        method: get_record(payload, model, dt, epsilon, cap, method, steps, study)
        for method in METHODS
    }
    params = make_parameters(dt, epsilon, cap)

    for method in METHODS:
        record = records[method]
        if record["diagnostics"] is None:
            print(
                f"[{stage} {model} dt={dt:g} eps={epsilon:.0e} cap={cap}] "
                f"diagnostics {method}",
                flush=True,
            )
            record["diagnostics"] = calculate_diagnostics(
                model, method, dt, epsilon, cap, steps, fixture, output
            )
            payload["updated_utc"] = utc_now()
            atomic_write_json(output, payload)

    needed = {
        method: max(0, timing_targets.get(method, 0) - len(records[method]["timing"]["samples_seconds"]))
        for method in METHODS
    }
    active = [method for method in METHODS if needed[method] > 0]
    if not active:
        return
    for method in active:
        warm_method(method, fixture, params)

    round_index = min(len(records[method]["timing"]["samples_seconds"]) for method in active)
    while any(value > 0 for value in needed.values()):
        order = list(METHODS if round_index % 2 == 0 else reversed(METHODS))
        for order_position, method in enumerate(order, start=1):
            if needed[method] <= 0:
                continue
            record = records[method]
            sample_index = len(record["timing"]["samples_seconds"]) + 1
            print(
                f"[{stage} {model} dt={dt:g} eps={epsilon:.0e} cap={cap}] "
                f"timing {method} sample {sample_index}/{timing_targets[method]}",
                flush=True,
            )
            duration = BASE.time_method(method, fixture["initial"], fixture["mpo"], params, steps)
            record["timing"]["samples_seconds"].append(duration)
            record["timing"]["events"].append(
                {
                    "sample_index": sample_index,
                    "duration_seconds": duration,
                    "stage": stage,
                    "order_position": order_position,
                    "completed_utc": utc_now(),
                }
            )
            refresh_timing_summary(record)
            needed[method] -= 1
            payload["updated_utc"] = utc_now()
            atomic_write_json(output, payload)
        round_index += 1


def tradeoff_records(payload: dict[str, Any], model: str, method: str) -> list[tuple[str, dict[str, Any]]]:
    """Return complete tradeoff records for one model/method."""
    result = []
    for key, record in payload["records"].items():
        config = record["configuration"]
        if (
            "tradeoff" in record["studies"]
            and config["model"] == model
            and config["method"] == method
            and record["diagnostics"] is not None
            and record["timing"]["samples_seconds"]
        ):
            result.append((key, record))
    return result


def pareto_keys(records: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Return non-dominated runtime/infidelity record keys."""
    selected: list[str] = []
    for key, record in records:
        runtime = record["timing"]["median_seconds"]
        error = record["diagnostics"]["infidelity"]
        dominated = False
        for other_key, other in records:
            if other_key == key:
                continue
            other_runtime = other["timing"]["median_seconds"]
            other_error = other["diagnostics"]["infidelity"]
            if (
                other_runtime <= runtime
                and other_error <= error
                and (other_runtime < runtime or other_error < error)
            ):
                dominated = True
                break
        if not dominated:
            selected.append(key)
    return sorted(
        selected,
        key=lambda item: payload_sort_key(records, item),
    )


def payload_sort_key(records: list[tuple[str, dict[str, Any]]], key: str) -> tuple[float, float]:
    """Sort selected keys by runtime, then error."""
    lookup = dict(records)
    record = lookup[key]
    return record["timing"]["median_seconds"], record["diagnostics"]["infidelity"]


def pilot_complete(payload: dict[str, Any], models: tuple[str, ...], dts: tuple[float, ...], eps: tuple[float, ...]) -> bool:
    """Check that the entire requested tradeoff pilot grid is present."""
    for model in models:
        for dt in dts:
            for epsilon in eps:
                for method in METHODS:
                    key = record_key(model, dt, epsilon, BASELINE_CAP, method)
                    record = payload["records"].get(key)
                    if (
                        record is None
                        or record["diagnostics"] is None
                        or len(record["timing"]["samples_seconds"]) < payload["protocol"]["pilot_timing_samples"]
                    ):
                        return False
    return True


def current_pareto_groups(payload: dict[str, Any], models: tuple[str, ...]) -> dict[str, Any]:
    """Compute the current Pareto groups from the available timing medians."""
    groups: dict[str, Any] = {}
    for model in models:
        for method in METHODS:
            records = tradeoff_records(payload, model, method)
            key = f"{model}|{method}"
            selected = pareto_keys(records)
            groups[key] = {
                "candidate_count": len(records),
                "selected_count": len(selected),
                "record_keys": selected,
            }
    return groups


def run_pilot(
    payload: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
    fixtures: dict[str, dict[str, Any]],
) -> None:
    """Run one timing sample and one diagnostic replay over the full grid."""
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    dts = parse_csv_floats(args.dts)
    epsilons = parse_csv_floats(args.epsilons)
    for model in models:
        for epsilon in epsilons:
            for dt in dts:
                run_configuration_pair(
                    payload,
                    output,
                    fixtures[model],
                    model=model,
                    dt=dt,
                    epsilon=epsilon,
                    cap=BASELINE_CAP,
                    study="tradeoff",
                    timing_targets={method: args.pilot_samples for method in METHODS},
                    stage="pilot",
                )
    if not pilot_complete(payload, models, dts, epsilons):
        raise RuntimeError("Pilot grid is incomplete after pilot stage")
    pilot_groups = current_pareto_groups(payload, models)
    payload["pareto_selection"] = {
        "pilot_selected_utc": utc_now(),
        "pilot_groups": pilot_groups,
        "repeat_history": [],
        "final_groups": pilot_groups,
    }
    payload["events"].append({"event": "pilot_complete_and_pareto_selected", "utc": utc_now()})
    atomic_write_json(output, payload)


def run_pareto_repeats(
    payload: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
    fixtures: dict[str, dict[str, Any]],
) -> None:
    """Increase timing replication only for pilot-selected Pareto points."""
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    dts = parse_csv_floats(args.dts)
    epsilons = parse_csv_floats(args.epsilons)
    if not pilot_complete(payload, models, dts, epsilons):
        raise RuntimeError("Run the complete pilot stage before Pareto repeats")
    if not payload.get("pareto_selection", {}).get("pilot_groups"):
        groups = current_pareto_groups(payload, models)
        payload["pareto_selection"] = {
            "pilot_selected_utc": utc_now(),
            "pilot_groups": groups,
            "repeat_history": [],
            "final_groups": groups,
        }
        atomic_write_json(output, payload)

    # Recompute after each timing pass.  A pilot point can enter the frontier
    # when the medians of initially selected points move; if so, it also gets
    # the full replication count before the final frontier is frozen.
    for iteration in range(1, 9):
        groups = current_pareto_groups(payload, models)
        selected_keys = {
            key
            for group in groups.values()
            for key in group["record_keys"]
        }
        needing_repeats = {
            key
            for key in selected_keys
            if len(payload["records"][key]["timing"]["samples_seconds"]) < args.target_samples
        }
        payload["pareto_selection"]["repeat_history"].append(
            {
                "iteration": iteration,
                "evaluated_utc": utc_now(),
                "groups": groups,
                "records_needing_repeats": sorted(needing_repeats),
            }
        )
        payload["pareto_selection"]["final_groups"] = groups
        atomic_write_json(output, payload)
        if not needing_repeats:
            break

        by_config: dict[tuple[str, float, float, int], dict[str, int]] = {}
        for key in needing_repeats:
            config = payload["records"][key]["configuration"]
            config_key = (config["model"], config["dt"], config["epsilon"], config["max_bond_dim"])
            by_config.setdefault(config_key, {})[config["method"]] = args.target_samples
        for (model, dt, epsilon, cap), targets in sorted(by_config.items()):
            run_configuration_pair(
                payload,
                output,
                fixtures[model],
                model=model,
                dt=dt,
                epsilon=epsilon,
                cap=cap,
                study="tradeoff",
                timing_targets=targets,
                stage=f"pareto-repeat-{iteration}",
            )
    else:
        raise RuntimeError("Pareto repeat selection did not stabilize after eight passes")
    payload["events"].append({"event": "pareto_repeats_complete", "utc": utc_now()})
    atomic_write_json(output, payload)


def run_caps(
    payload: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
    fixtures: dict[str, dict[str, Any]],
) -> None:
    """Run the active-cap Haldane-Shastry comparison."""
    for cap in parse_csv_ints(args.caps):
        run_configuration_pair(
            payload,
            output,
            fixtures["hs"],
            model="hs",
            dt=CAP_DT,
            epsilon=CAP_EPSILON,
            cap=cap,
            study="cap",
            timing_targets={method: args.target_samples for method in METHODS},
            stage="cap",
        )
    payload["events"].append({"event": "cap_study_complete", "utc": utc_now()})
    atomic_write_json(output, payload)


def main() -> None:
    """Run the requested resumable benchmark stages."""
    args = parse_args()
    if args.pilot_samples < 1 or args.target_samples < args.pilot_samples:
        raise ValueError("Require target_samples >= pilot_samples >= 1")
    requested_models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    if not requested_models or not set(requested_models) <= set(MODELS):
        raise ValueError(f"Unsupported model selection: {requested_models}")

    # Pin every local exponential to the same matrix-free pure-NumPy path.
    primitives.DENSE_THRESHOLD = -1
    matrix_exponential.NUMBA_THRESHOLD = sys.maxsize

    output = args.output.resolve()
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        validate_resume_protocol(payload, args)
    else:
        payload = initial_payload(args)
        save_provenance(payload, output)
        import_matched_baseline(payload, args)
        atomic_write_json(output, payload)

    required_models = set(requested_models)
    if args.stage in {"caps", "all"}:
        required_models.add("hs")
    fixtures = {
        model: build_fixture(model, args.total_time, payload, output)
        for model in MODELS
        if model in required_models
    }
    payload["updated_utc"] = utc_now()
    atomic_write_json(output, payload)

    stages = ("pilot", "repeat", "caps") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "pilot":
            run_pilot(payload, output, args, fixtures)
        elif stage == "repeat":
            run_pareto_repeats(payload, output, args, fixtures)
        elif stage == "caps":
            run_caps(payload, output, args, fixtures)
    print(f"Completed stage(s): {', '.join(stages)}", flush=True)
    print(f"Raw results: {output}", flush=True)


if __name__ == "__main__":
    main()
