# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Summarize fixed-horizon tolerance crossings from retained CSV data.

This module is deliberately analysis-only: it validates and reads the frozen
cap-sweep and timing summaries, then writes compact CSV and Markdown tables. It
does not import or invoke any circuit-simulation code.

Example:
    uv run python -m experiments.circuit_benchmarks.extensions.fixed_horizon_tolerance_summary
"""

from __future__ import annotations

import csv
import hashlib
import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "fixed_horizon_refinement"
CAP_SWEEP_PATH = OUTPUT_DIR / "combined_cap_sweep.csv"
TIMING_PATH = OUTPUT_DIR / "cap_timing_summary.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "tolerance_summary.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "tolerance_summary.md"

CAP_SWEEP_SHA256 = "a3510a5d152da66791a2dfb2d56b1f3c77ac29c781c0e26d3778574fa583990c"
TIMING_SHA256 = "43e926a5f6e862f831b81bf0c1c54d93abd17b3105c64ce663fc180c49760150"
TARGET_STEP = 15
TOLERANCES = (5e-3, 1e-2, 2e-2)
METHODS = ("gate_local_2tdvp", "mpo_contract_compress", "tebd_swap")
METHOD_LABELS = {
    "gate_local_2tdvp": "TDVP",
    "mpo_contract_compress": "MPO",
    "tebd_swap": "TEBD+SWAP",
}

CAP_SWEEP_FIELDS = (
    "method",
    "chi_max",
    "source",
    "task_id",
    "target_step",
    "target_time",
    "max_infidelity_through",
    "endpoint_infidelity",
    "reliable",
    "peak_parameter_count",
    "peak_mps_bytes",
    "peak_bond_dim",
    "selected",
    "last_fail",
)
TIMING_FIELDS = (
    "method",
    "chi_max",
    "target_step",
    "median_s",
    "min_s",
    "max_s",
    "repeats",
)
SUMMARY_FIELDS = (
    "epsilon",
    "method",
    "selection_status",
    "first_passing_tested_chi_max",
    "worst_prefix_infidelity",
    "peak_parameter_count",
    "median_runtime_s",
    "timing_repeats",
    "preceding_failing_chi_max",
    "preceding_failing_infidelity",
    "best_tested_chi_max",
    "best_tested_infidelity",
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_validated_csv(
    path: Path,
    *,
    expected_sha256: str,
    expected_fields: Sequence[str],
) -> list[dict[str, str]]:
    """Read a nonempty CSV after validating its pinned digest and exact schema."""
    if not path.is_file():
        msg = f"Missing required source {path}."
        raise FileNotFoundError(msg)
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        msg = f"Source hash mismatch for {path}: expected {expected_sha256}, found {actual_sha256}."
        raise RuntimeError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_fields = tuple(reader.fieldnames or ())
        if actual_fields != tuple(expected_fields):
            msg = f"Schema mismatch for {path}: expected {tuple(expected_fields)!r}, found {actual_fields!r}."
            raise ValueError(msg)
        rows = list(reader)
    if not rows:
        msg = f"Required source {path} is empty."
        raise ValueError(msg)
    return rows


def _validated_source_rows(
    cap_rows: Sequence[Mapping[str, str]],
    timing_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, int], dict[str, Any]]]:
    """Parse rows and enforce the fixed-horizon method/cap correspondence."""
    caps_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    seen_caps: set[tuple[str, int]] = set()
    for row in cap_rows:
        method = str(row["method"])
        if method not in caps_by_method:
            msg = f"Unknown cap-sweep method {method!r}."
            raise ValueError(msg)
        cap = int(row["chi_max"])
        target_step = int(row["target_step"])
        infidelity = float(row["max_infidelity_through"])
        parameters = int(row["peak_parameter_count"])
        key = (method, cap)
        if key in seen_caps:
            msg = f"Duplicate cap-sweep point for {method}/chi{cap}."
            raise RuntimeError(msg)
        if target_step != TARGET_STEP or cap < 1 or parameters < 1 or not math.isfinite(infidelity) or infidelity < 0.0:
            msg = f"Invalid cap-sweep point for {method}/chi{cap}."
            raise ValueError(msg)
        seen_caps.add(key)
        caps_by_method[method].append(
            {
                "method": method,
                "chi_max": cap,
                "max_infidelity_through": infidelity,
                "peak_parameter_count": parameters,
            }
        )

    timing_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in timing_rows:
        method = str(row["method"])
        cap = int(row["chi_max"])
        target_step = int(row["target_step"])
        median = float(row["median_s"])
        minimum = float(row["min_s"])
        maximum = float(row["max_s"])
        repeats = int(row["repeats"])
        key = (method, cap)
        if method not in caps_by_method:
            msg = f"Unknown timing method {method!r}."
            raise ValueError(msg)
        if key in timing_by_key:
            msg = f"Duplicate timing point for {method}/chi{cap}."
            raise RuntimeError(msg)
        if (
            target_step != TARGET_STEP
            or repeats < 1
            or not all(math.isfinite(value) and value > 0.0 for value in (minimum, median, maximum))
            or not minimum <= median <= maximum
        ):
            msg = f"Invalid timing point for {method}/chi{cap}."
            raise ValueError(msg)
        timing_by_key[key] = {"median_s": median, "repeats": repeats}

    if set(timing_by_key) != seen_caps:
        missing_timing = sorted(seen_caps.difference(timing_by_key))
        unexpected_timing = sorted(set(timing_by_key).difference(seen_caps))
        msg = f"Cap/timing key mismatch: missing timing={missing_timing!r}, unexpected timing={unexpected_timing!r}."
        raise RuntimeError(msg)
    for points in caps_by_method.values():
        if not points:
            msg = "Every expected method must have at least one tested cap."
            raise RuntimeError(msg)
        points.sort(key=lambda point: int(point["chi_max"]))
    return caps_by_method, timing_by_key


def summarize_tolerances(
    cap_rows: Sequence[Mapping[str, str]],
    timing_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Return first-passing tested points for the three fixed tolerances."""
    caps_by_method, timing_by_key = _validated_source_rows(cap_rows, timing_rows)
    summaries: list[dict[str, Any]] = []
    for epsilon in TOLERANCES:
        for method in METHODS:
            points = caps_by_method[method]
            passing_index = next(
                (index for index, point in enumerate(points) if float(point["max_infidelity_through"]) <= epsilon),
                None,
            )
            if passing_index is None:
                best = min(points, key=lambda point: float(point["max_infidelity_through"]))
                summaries.append(
                    {
                        "epsilon": epsilon,
                        "method": method,
                        "selection_status": "no pass on tested grid",
                        "first_passing_tested_chi_max": "",
                        "worst_prefix_infidelity": "",
                        "peak_parameter_count": "",
                        "median_runtime_s": "",
                        "timing_repeats": "",
                        "preceding_failing_chi_max": "",
                        "preceding_failing_infidelity": "",
                        "best_tested_chi_max": best["chi_max"],
                        "best_tested_infidelity": best["max_infidelity_through"],
                    }
                )
                continue

            selected = points[passing_index]
            timing = timing_by_key[(method, int(selected["chi_max"]))]
            preceding = points[passing_index - 1] if passing_index > 0 else None
            if preceding is not None and float(preceding["max_infidelity_through"]) <= epsilon:
                msg = f"Internal first-passing selection error for {method} at epsilon={epsilon:g}."
                raise RuntimeError(msg)
            summaries.append(
                {
                    "epsilon": epsilon,
                    "method": method,
                    "selection_status": "first-passing tested",
                    "first_passing_tested_chi_max": selected["chi_max"],
                    "worst_prefix_infidelity": selected["max_infidelity_through"],
                    "peak_parameter_count": selected["peak_parameter_count"],
                    "median_runtime_s": timing["median_s"],
                    "timing_repeats": timing["repeats"],
                    "preceding_failing_chi_max": "" if preceding is None else preceding["chi_max"],
                    "preceding_failing_infidelity": ("" if preceding is None else preceding["max_infidelity_through"]),
                    "best_tested_chi_max": "",
                    "best_tested_infidelity": "",
                }
            )
    return summaries


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the fixed-schema CSV atomically."""
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
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _epsilon_text(epsilon: float) -> str:
    """Format one tolerance compactly and exactly for the summary table."""
    return {5e-3: "5e-3", 1e-2: "1e-2", 2e-2: "2e-2"}[epsilon]


def _markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the compact human-readable summary and pinned provenance."""
    lines = [
        "# Fixed-horizon tolerance summary",
        "",
        "This is a pure reanalysis of the retained `n=15` cap sweep. No simulations were run.",
        "Each reported point is the **first-passing tested** cap, not a minimum over untested caps",
        "or a globally optimal accuracy--cost point.",
        "",
        "| $\\epsilon$ | Method | Selection | $E_\\star$ | $P_{\\max}$ | "
        "Median runtime (s) | Previous tested failure |",
        "|---:|:---|:---|---:|---:|---:|:---|",
    ]
    for row in rows:
        epsilon = _epsilon_text(float(row["epsilon"]))
        label = METHOD_LABELS[str(row["method"])]
        if row["selection_status"] == "first-passing tested":
            previous = (
                "none"
                if row["preceding_failing_chi_max"] == ""
                else (
                    f"$\\chi={row['preceding_failing_chi_max']}$, "
                    f"$E_\\star={float(row['preceding_failing_infidelity']):.6g}$"
                )
            )
            lines.append(
                f"| {epsilon} | {label} | $\\chi={row['first_passing_tested_chi_max']}$ "
                f"(first-passing tested) | {float(row['worst_prefix_infidelity']):.6g} | "
                f"{int(row['peak_parameter_count']):,} | {float(row['median_runtime_s']):.6g} | "
                f"{previous} |"
            )
        else:
            lines.append(
                f"| {epsilon} | {label} | **No pass on tested grid** "
                f"(best: $\\chi={row['best_tested_chi_max']}$) | "
                f"{float(row['best_tested_infidelity']):.6g} (best) | -- | -- | -- |"
            )
    lines.extend(
        [
            "",
            "`E_star` is the maximum normalized infidelity through step 15. `P_max` is the",
            "peak recorded MPS tensor-entry count. Runtimes are medians over the retained",
            "repeat count recorded in the CSV.",
            "",
            "## Source provenance",
            "",
            f"- `{CAP_SWEEP_PATH.name}` SHA-256: `{CAP_SWEEP_SHA256}`",
            f"- `{TIMING_PATH.name}` SHA-256: `{TIMING_SHA256}`",
            f"- Fixed tolerances: `{', '.join(_epsilon_text(value) for value in TOLERANCES)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, content: str) -> None:
    """Write text atomically."""
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
        handle.write(content)
    temporary.replace(path)


def run_analysis() -> list[dict[str, Any]]:
    """Validate frozen inputs and write both analysis-only artifacts."""
    cap_rows = read_validated_csv(
        CAP_SWEEP_PATH,
        expected_sha256=CAP_SWEEP_SHA256,
        expected_fields=CAP_SWEEP_FIELDS,
    )
    timing_rows = read_validated_csv(
        TIMING_PATH,
        expected_sha256=TIMING_SHA256,
        expected_fields=TIMING_FIELDS,
    )
    summaries = summarize_tolerances(cap_rows, timing_rows)
    _atomic_csv(SUMMARY_CSV_PATH, summaries)
    _atomic_text(SUMMARY_MD_PATH, _markdown(summaries))
    print(f"Wrote {SUMMARY_CSV_PATH}")
    print(f"Wrote {SUMMARY_MD_PATH}")
    return summaries


def main() -> int:
    """Run the pure retained-data analysis."""
    run_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
