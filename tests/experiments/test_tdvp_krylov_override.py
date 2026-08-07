# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused tests for the isolated long-trajectory TDVP override."""
# ruff: file-ignore[private-member-access]

from __future__ import annotations

import pytest

from experiments.circuit_benchmarks.long_trajectories import (
    tdvp_krylov_override as control,
)


def _base_manifest(*, endpoint: int = 30) -> dict[str, object]:
    return {
        "campaign_id": control.BASE_CAMPAIGN_ID,
        "cases": {
            case_key: {
                "status": "success",
                "criterion_met": True,
                "right_censored": False,
                "stop_step": endpoint,
            }
            for case_key in control.CASE_ORDER
        },
    }


def test_frozen_endpoint_validation_requires_complete_uncensored_cases() -> None:
    """Only the four successful adaptive endpoints may define the control."""
    assert control._extract_frozen_endpoints(_base_manifest(endpoint=37)) == dict.fromkeys(control.CASE_ORDER, 37)

    manifest = _base_manifest()
    manifest["cases"][control.CASE_ORDER[0]]["right_censored"] = True  # type: ignore[index]
    with pytest.raises(RuntimeError, match="incomplete or censored"):
        control._extract_frozen_endpoints(manifest)


def test_frozen_endpoint_must_cover_bond_profile_horizon() -> None:
    """The same accuracy state stream must supply every requested profile."""
    with pytest.raises(RuntimeError, match="bond-profile horizon"):
        control._extract_frozen_endpoints(_base_manifest(endpoint=29))


def test_override_changes_only_krylov_tolerance() -> None:
    """The control keeps the production TDVP cap, subdivision, and SVD cutoff."""
    params = control._params()
    assert params.max_bond_dim == control.CHI_CAP
    assert params.tdvp_sweeps == control.N_SUB
    assert params.svd_threshold == control.SVD_THRESHOLD
    assert params.krylov_tol == control.KRYLOV_TOLERANCE


def test_timing_summary_requires_three_distinct_repeats() -> None:
    """Only complete pointwise warm-cache timing groups enter the summary."""
    rows = [
        {
            "case": "ising_1d",
            "step": 1,
            "repeat": repeat,
            "cumulative_runtime_s": runtime,
        }
        for repeat, runtime in enumerate((3.0, 1.0, 2.0))
    ]
    [summary] = control.summarize_timing_rows(rows)
    assert summary["median_cumulative_runtime_s"] == pytest.approx(2.0)
    assert summary["min_cumulative_runtime_s"] == pytest.approx(1.0)
    assert summary["max_cumulative_runtime_s"] == pytest.approx(3.0)

    duplicate = [*rows, dict(rows[0])]
    with pytest.raises(RuntimeError, match="Duplicate timing row"):
        control.summarize_timing_rows(duplicate)


def test_declared_table_schemas_reject_field_drift() -> None:
    """Aggregates restore schema order after sorted-key JSON serialization."""
    scrambled = {field: index for index, field in enumerate(reversed(control.TRAJECTORY_FIELDS))}
    [ordered] = control._rows_in_schema_order([scrambled], control.TRAJECTORY_FIELDS)
    assert tuple(ordered) == control.TRAJECTORY_FIELDS
    with pytest.raises(RuntimeError, match="do not match schema"):
        control._rows_in_schema_order([{"case": "ising_1d"}], control.TRAJECTORY_FIELDS)


def test_completed_accuracy_hash_exception_changes_no_scientific_field() -> None:
    """The aggregation repair reuses only the exact completed numerical payload."""
    expected = {"source_hash": "new-runner", "case": "ising_1d", "chi_cap": 32}
    actual = {
        **expected,
        "source_hash": next(iter(control.COMPLETED_ACCURACY_SOURCE_HASHES)),
    }
    assert control._accuracy_payload_matches(actual, expected)
    assert not control._accuracy_payload_matches({**actual, "chi_cap": 16}, expected)
    assert not control._accuracy_payload_matches({**actual, "source_hash": "unknown"}, expected)
