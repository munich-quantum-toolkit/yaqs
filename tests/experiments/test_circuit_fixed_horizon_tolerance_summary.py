# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused tests for the retained fixed-horizon tolerance summary."""

from __future__ import annotations

import csv
import hashlib
from typing import TYPE_CHECKING

import pytest

from experiments.circuit_benchmarks.extensions.fixed_horizon_tolerance_summary import (
    CAP_SWEEP_FIELDS,
    METHODS,
    TIMING_FIELDS,
    TOLERANCES,
    read_validated_csv,
    summarize_tolerances,
)

if TYPE_CHECKING:
    from pathlib import Path


def _source_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    cap_rows: list[dict[str, str]] = []
    timing_rows: list[dict[str, str]] = []
    errors = {
        METHODS[0]: ((4, 0.03), (8, 0.009), (12, 0.004)),
        METHODS[1]: ((4, 0.04), (8, 0.019), (12, 0.008)),
        METHODS[2]: ((4, 0.04), (8, 0.03), (12, 0.025)),
    }
    for method in METHODS:
        for cap, error in errors[method]:
            cap_rows.append({
                "method": method,
                "chi_max": str(cap),
                "target_step": "15",
                "max_infidelity_through": str(error),
                "peak_parameter_count": str(100 * cap),
            })
            timing_rows.append({
                "method": method,
                "chi_max": str(cap),
                "target_step": "15",
                "median_s": str(cap / 2),
                "min_s": str(cap / 2 - 0.1),
                "max_s": str(cap / 2 + 0.1),
                "repeats": "3",
            })
    return cap_rows, timing_rows


def test_summary_uses_only_fixed_tolerances_and_marks_no_pass() -> None:
    """Every method/tolerance pair appears and unavailable crossings remain explicit."""
    cap_rows, timing_rows = _source_rows()

    rows = summarize_tolerances(cap_rows, timing_rows)

    assert len(rows) == len(TOLERANCES) * len(METHODS)
    assert {float(row["epsilon"]) for row in rows} == set(TOLERANCES)
    no_pass = [row for row in rows if row["selection_status"] == "no pass on tested grid"]
    assert len(no_pass) == 4
    assert all(not row["first_passing_tested_chi_max"] for row in no_pass)
    assert all(row["best_tested_chi_max"] for row in no_pass)


def test_summary_selects_lowest_tested_passing_cap_and_joins_timing() -> None:
    """Selection is by tested cap and carries its recorded timing, not an interpolation."""
    cap_rows, timing_rows = _source_rows()

    rows = summarize_tolerances(cap_rows, timing_rows)
    selected = next(row for row in rows if row["method"] == METHODS[0] and float(row["epsilon"]) == pytest.approx(1e-2))

    assert selected["selection_status"] == "first-passing tested"
    assert selected["first_passing_tested_chi_max"] == 8
    assert selected["preceding_failing_chi_max"] == 4
    assert selected["peak_parameter_count"] == 800
    assert selected["median_runtime_s"] == pytest.approx(4.0)
    assert selected["timing_repeats"] == 3


def test_validated_csv_rejects_hash_and_schema_drift(tmp_path: Path) -> None:
    """Pinned inputs must fail closed if their content or header changes."""
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAP_SWEEP_FIELDS)
        writer.writeheader()
        writer.writerow(dict.fromkeys(CAP_SWEEP_FIELDS, "value"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    rows = read_validated_csv(
        source,
        expected_sha256=digest,
        expected_fields=CAP_SWEEP_FIELDS,
    )
    assert len(rows) == 1

    with pytest.raises(RuntimeError, match="Source hash mismatch"):
        read_validated_csv(
            source,
            expected_sha256="0" * 64,
            expected_fields=CAP_SWEEP_FIELDS,
        )
    with pytest.raises(ValueError, match="Schema mismatch"):
        read_validated_csv(
            source,
            expected_sha256=digest,
            expected_fields=TIMING_FIELDS,
        )
