#!/usr/bin/env python3
"""Validate raw benchmark data and export flat manuscript/source tables."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
RAW_JSON = HERE / "raw_results.json"
TRADEOFF_CSV = HERE / "tradeoff_all_points.csv"
PARETO_CSV = HERE / "tradeoff_pareto_points.csv"
CAP_CSV = HERE / "cap_study.csv"
TIMINGS_CSV = HERE / "timing_samples.csv"
BUG_CHECKPOINTS_CSV = HERE / "bug_first_step_checkpoints.csv"
VALIDATION_MD = HERE / "VALIDATION.md"
MANIFEST_JSON = HERE / "MANIFEST.json"

METHODS = ("bug", "2tdvp")
MODELS = ("tfim", "hs")


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_key(model: str, dt: float, epsilon: float, cap: int, method: str) -> str:
    """Reproduce the runner's canonical record key."""
    return f"{model}|dt={dt:.8g}|eps={epsilon:.8g}|cap={cap}|{method}"


def final_pareto_keys(payload: dict[str, Any]) -> set[str]:
    """Return the saved stabilized Pareto selection."""
    return {
        key
        for group in payload["pareto_selection"]["final_groups"].values()
        for key in group["record_keys"]
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write a deterministic flat CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tradeoff_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten all runtime/accuracy configurations."""
    pareto = final_pareto_keys(payload)
    rows: list[dict[str, Any]] = []
    for key, record in payload["records"].items():
        if "tradeoff" not in record["studies"]:
            continue
        config = record["configuration"]
        diagnostics = record["diagnostics"]
        timing = record["timing"]
        rows.append(
            {
                "record_key": key,
                "model": config["model"],
                "method": config["method"],
                "dt": config["dt"],
                "steps": config["steps"],
                "epsilon": config["epsilon"],
                "max_bond_dim": config["max_bond_dim"],
                "min_keep": config["min_keep"],
                "timing_sample_count": timing["sample_count"],
                "runtime_median_seconds": timing["median_seconds"],
                "runtime_mean_seconds": timing["mean_seconds"],
                "runtime_sample_sd_seconds": timing["sample_standard_deviation_seconds"],
                "runtime_minimum_seconds": timing["minimum_seconds"],
                "runtime_maximum_seconds": timing["maximum_seconds"],
                "phase_aligned_state_error": diagnostics["phase_aligned_state_error"],
                "infidelity": diagnostics["infidelity"],
                "max_abs_z_error": diagnostics["max_abs_z_error"],
                "rms_z_error": diagnostics["rms_z_error"],
                "energy_abs_error": diagnostics["energy_abs_error"],
                "norm": diagnostics["norm"],
                "max_chi": diagnostics["max_chi"],
                "krylov_calls": diagnostics["krylov_calls"],
                "krylov_operator_applications": diagnostics["krylov_operator_applications"],
                "is_final_pareto": key in pareto,
                "source": record["source"],
                "final_bond_profile": ";".join(map(str, diagnostics["final_bond_profile"])),
                "state_vector_path": diagnostics.get("state_vector_path"),
                "state_vector_sha256": diagnostics.get("state_vector_sha256"),
            }
        )
    return sorted(rows, key=lambda row: (row["model"], row["method"], row["dt"], row["epsilon"]))


def cap_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the Haldane-Shastry active-cap study."""
    rows: list[dict[str, Any]] = []
    for key, record in payload["records"].items():
        if "cap" not in record["studies"]:
            continue
        config = record["configuration"]
        diagnostics = record["diagnostics"]
        timing = record["timing"]
        rows.append(
            {
                "record_key": key,
                "model": config["model"],
                "method": config["method"],
                "dt": config["dt"],
                "epsilon": config["epsilon"],
                "max_bond_dim": config["max_bond_dim"],
                "timing_sample_count": timing["sample_count"],
                "runtime_median_seconds": timing["median_seconds"],
                "runtime_mean_seconds": timing["mean_seconds"],
                "runtime_sample_sd_seconds": timing["sample_standard_deviation_seconds"],
                "phase_aligned_state_error": diagnostics["phase_aligned_state_error"],
                "infidelity": diagnostics["infidelity"],
                "max_abs_z_error": diagnostics["max_abs_z_error"],
                "energy_abs_error": diagnostics["energy_abs_error"],
                "norm": diagnostics["norm"],
                "attained_max_chi": diagnostics["max_chi"],
                "cap_was_active": diagnostics["max_chi"] >= config["max_bond_dim"],
                "krylov_calls": diagnostics["krylov_calls"],
                "krylov_operator_applications": diagnostics["krylov_operator_applications"],
                "source": record["source"],
            }
        )
    return sorted(rows, key=lambda row: (row["max_bond_dim"], row["method"]))


def timing_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every individual wall-time observation."""
    rows: list[dict[str, Any]] = []
    for key, record in payload["records"].items():
        config = record["configuration"]
        events = record["timing"].get("events", [])
        if events:
            for event in events:
                rows.append(
                    {
                        "record_key": key,
                        "model": config["model"],
                        "method": config["method"],
                        "dt": config["dt"],
                        "epsilon": config["epsilon"],
                        "max_bond_dim": config["max_bond_dim"],
                        "sample_index": event["sample_index"],
                        "duration_seconds": event["duration_seconds"],
                        "stage": event["stage"],
                        "order_position": event.get("order_position"),
                        "completed_utc": event.get("completed_utc"),
                    }
                )
        else:
            for index, duration in enumerate(record["timing"]["samples_seconds"], start=1):
                rows.append(
                    {
                        "record_key": key,
                        "model": config["model"],
                        "method": config["method"],
                        "dt": config["dt"],
                        "epsilon": config["epsilon"],
                        "max_bond_dim": config["max_bond_dim"],
                        "sample_index": index,
                        "duration_seconds": duration,
                        "stage": "imported",
                        "order_position": None,
                        "completed_utc": None,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["model"],
            row["dt"],
            row["epsilon"],
            row["max_bond_dim"],
            row["sample_index"],
            row["method"],
        ),
    )


def checkpoint_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every stored first-step BUG bond profile."""
    rows: list[dict[str, Any]] = []
    for key, record in payload["records"].items():
        if record["configuration"]["method"] != "bug":
            continue
        checkpoints = record["diagnostics"].get("first_step_bug_checkpoints") or {}
        for stage, profile in checkpoints.items():
            for bond, dimension in enumerate(profile):
                rows.append(
                    {
                        "record_key": key,
                        "model": record["configuration"]["model"],
                        "dt": record["configuration"]["dt"],
                        "epsilon": record["configuration"]["epsilon"],
                        "max_bond_dim": record["configuration"]["max_bond_dim"],
                        "checkpoint": stage,
                        "bond_index": bond,
                        "bond_dimension": dimension,
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["model"],
            row["dt"],
            row["epsilon"],
            row["max_bond_dim"],
            row["checkpoint"],
            row["bond_index"],
        ),
    )


def main() -> None:
    """Validate completeness/integrity and write all derived outputs."""
    payload = json.loads(RAW_JSON.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    checks: list[tuple[str, bool, str]] = []

    expected_records = set()
    for model in MODELS:
        for dt in protocol["tradeoff_dt_grid"]:
            for epsilon in protocol["tradeoff_epsilon_grid"]:
                for method in METHODS:
                    expected_records.add(record_key(model, dt, epsilon, protocol["tradeoff_cap"], method))
    for cap in protocol["cap_grid"]:
        for method in METHODS:
            expected_records.add(record_key("hs", protocol["cap_dt"], protocol["cap_epsilon"], cap, method))
    checks.append(
        (
            "all requested configuration/method records exist",
            expected_records <= set(payload["records"]),
            f"{len(expected_records & set(payload['records']))}/{len(expected_records)}",
        )
    )

    fixture_arrays: dict[str, dict[str, np.ndarray]] = {}
    for model, metadata in payload["fixtures"].items():
        path = HERE / metadata["path"]
        checks.append((f"{model} fixture exists", path.exists(), str(path)))
        checks.append((f"{model} fixture checksum", sha256_file(path) == metadata["sha256"], metadata["sha256"]))
        with np.load(path) as archive:
            fixture_arrays[model] = {name: archive[name].copy() for name in archive.files}
        checks.append(
            (
                f"{model} initial norm",
                abs(metadata["initial_norm"] - 1) < 1e-12,
                f"{metadata['initial_norm']:.16g}",
            )
        )
        checks.append(
            (
                f"{model} reference norm",
                abs(metadata["reference_norm"] - 1) < 1e-12,
                f"{metadata['reference_norm']:.16g}",
            )
        )
        checks.append(
            (
                f"{model} MPO matches analytic Hamiltonian",
                metadata["mpo_vs_analytic_relative_frobenius_error"] < 1e-14,
                f"{metadata['mpo_vs_analytic_relative_frobenius_error']:.3e}",
            )
        )

    pareto = final_pareto_keys(payload)
    for key in sorted(expected_records):
        record = payload["records"].get(key)
        if record is None:
            continue
        config = record["configuration"]
        diagnostics = record["diagnostics"]
        timing = record["timing"]
        checks.append((f"{key}: diagnostics present", diagnostics is not None, str(diagnostics is not None)))
        checks.append((f"{key}: timing present", bool(timing["samples_seconds"]), str(len(timing["samples_seconds"]))))
        if diagnostics is None or not timing["samples_seconds"]:
            continue
        checks.append(
            (
                f"{key}: timing median is reproducible",
                math.isclose(timing["median_seconds"], statistics.median(timing["samples_seconds"]), rel_tol=0, abs_tol=1e-15),
                f"median {timing['median_seconds']:.9g}",
            )
        )
        expected_samples = (
            protocol["pareto_target_timing_samples"]
            if key in pareto
            else protocol["pilot_timing_samples"]
        )
        if "cap" in record["studies"]:
            expected_samples = max(expected_samples, protocol["cap_timing_samples"])
        checks.append(
            (
                f"{key}: sufficient timing samples",
                len(timing["samples_seconds"]) >= expected_samples,
                f"{len(timing['samples_seconds'])} >= {expected_samples}",
            )
        )
        checks.append(
            (
                f"{key}: bond cap respected",
                diagnostics["max_chi"] <= config["max_bond_dim"],
                f"{diagnostics['max_chi']} <= {config['max_bond_dim']}",
            )
        )
        norm_deviation = abs(diagnostics["norm"] - 1)
        if config["method"] == "bug":
            # BUG explicitly rescales its canonical center after each half-sweep.
            norm_bound = 5e-7
            norm_reason = "explicit half-sweep normalization"
        elif diagnostics["max_chi"] >= config["max_bond_dim"]:
            # An active hard cap can discard more than the tolerance-controlled
            # weight, so use a conservative guard against catastrophic drift.
            norm_bound = 5e-3
            norm_reason = "active hard cap"
        else:
            # 2TDVP does not renormalize after every truncating split.  A full
            # symmetric step has at most two truncations per bond, making this
            # conservative accumulated-discarded-weight bound appropriate.
            truncations = 2 * (protocol["length"] - 1) * config["steps"]
            norm_bound = max(5e-7, 1.1 * truncations * config["epsilon"])
            norm_reason = "cumulative tolerance-controlled 2TDVP truncations"
        checks.append(
            (
                f"{key}: norm within method-aware stability bound",
                math.isfinite(diagnostics["norm"]) and norm_deviation <= norm_bound,
                (
                    f"norm {diagnostics['norm']:.16g}; deviation {norm_deviation:.3e} "
                    f"<= {norm_bound:.3e} ({norm_reason})"
                ),
            )
        )
        expected_calls = (32 if config["method"] == "bug" else 57) * config["steps"]
        checks.append(
            (
                f"{key}: Krylov call count",
                diagnostics["krylov_calls"] == expected_calls,
                f"{diagnostics['krylov_calls']} == {expected_calls}",
            )
        )

        state_path_text = diagnostics.get("state_vector_path")
        if state_path_text:
            state_path = HERE / state_path_text
            checks.append((f"{key}: final state exists", state_path.exists(), state_path_text))
            checks.append(
                (
                    f"{key}: final state checksum",
                    sha256_file(state_path) == diagnostics["state_vector_sha256"],
                    diagnostics["state_vector_sha256"],
                )
            )
            with np.load(state_path) as archive:
                vector = archive["final_state_vector"]
                reference = fixture_arrays[config["model"]]["reference_state_vector"]
                denominator = float(np.vdot(reference, reference).real * np.vdot(vector, vector).real)
                recalculated_infid = max(0.0, 1.0 - float(abs(np.vdot(reference, vector)) ** 2 / denominator))
            checks.append(
                (
                    f"{key}: saved state reproduces infidelity",
                    math.isclose(recalculated_infid, diagnostics["infidelity"], rel_tol=0, abs_tol=5e-15),
                    f"{recalculated_infid:.8e}",
                )
            )
        if config["method"] == "bug":
            checkpoints = diagnostics.get("first_step_bug_checkpoints") or {}
            expected_stages = {
                "first_half_sweep",
                "first_compression",
                "second_half_sweep",
                "second_compression",
            }
            checks.append((f"{key}: four BUG checkpoints", set(checkpoints) == expected_stages, str(sorted(checkpoints))))
            if set(checkpoints) == expected_stages:
                compressed = checkpoints["first_compression"] + checkpoints["second_compression"]
                checks.append((f"{key}: min_keep=2", min(compressed) >= 2, f"minimum {min(compressed)}"))

    cap_records = [
        record
        for record in payload["records"].values()
        if "cap" in record["studies"] and record["configuration"]["max_bond_dim"] < protocol["tradeoff_cap"]
    ]
    active_methods = {
        (record["configuration"]["max_bond_dim"], record["configuration"]["method"])
        for record in cap_records
        if record["diagnostics"]["max_chi"] >= record["configuration"]["max_bond_dim"]
    }
    checks.append(
        (
            "each tested finite cap is active for both methods",
            all((cap, method) in active_methods for cap in protocol["cap_grid"] if cap < protocol["tradeoff_cap"] for method in METHODS),
            str(sorted(active_methods)),
        )
    )

    tradeoff = tradeoff_rows(payload)
    caps = cap_rows(payload)
    timings = timing_rows(payload)
    checkpoints = checkpoint_rows(payload)
    write_csv(TRADEOFF_CSV, tradeoff, list(tradeoff[0]))
    write_csv(PARETO_CSV, [row for row in tradeoff if row["is_final_pareto"]], list(tradeoff[0]))
    write_csv(CAP_CSV, caps, list(caps[0]))
    write_csv(TIMINGS_CSV, timings, list(timings[0]))
    write_csv(BUG_CHECKPOINTS_CSV, checkpoints, list(checkpoints[0]))

    failures = [check for check in checks if not check[1]]
    lines = [
        "# Validation report",
        "",
        f"Passed {len(checks) - len(failures)} of {len(checks)} checks.",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'}: {name} ({detail})" for name, passed, detail in checks)
    VALIDATION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_paths = [
        RAW_JSON,
        TRADEOFF_CSV,
        PARETO_CSV,
        CAP_CSV,
        TIMINGS_CSV,
        BUG_CHECKPOINTS_CSV,
        VALIDATION_MD,
        *sorted((HERE / "fixtures").glob("*.npz")),
        *sorted((HERE / "states").glob("*.npz")),
        *sorted((HERE / "provenance").glob("*")),
    ]
    manifest = {
        "raw_schema_version": payload["schema_version"],
        "validation_checks_passed": len(checks) - len(failures),
        "validation_checks_total": len(checks),
        "files": {
            str(path.relative_to(HERE)): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_paths
            if path.is_file()
        },
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} validation checks failed; see {VALIDATION_MD}")
    print(f"Validated {len(checks)} checks; all passed.")


if __name__ == "__main__":
    main()
