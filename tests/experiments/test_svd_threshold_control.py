# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Validation tests for the bounded final-protocol SVD-threshold control."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from experiments.circuit_benchmarks.extensions import svd_threshold_control as control
from mqt.yaqs.core import linalg


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_threshold_control_uses_the_final_bounded_protocol() -> None:
    """The control varies only tau at the three selected final configurations."""
    assert control.CASE_KEY == "ising_2d"
    assert control.TARGET_STEP == 15
    assert control.THRESHOLDS == (1e-14, 1e-13, 1e-12, 1e-9)
    assert control.METHOD_CAPS == {
        "gate_local_2tdvp": 28,
        "mpo_contract_compress": 26,
        "tebd_swap": 32,
    }
    assert control.METHOD_SUBSTEPS == {
        "gate_local_2tdvp": 2,
        "mpo_contract_compress": 1,
        "tebd_swap": 1,
    }

    for method, cap in control.METHOD_CAPS.items():
        params = control._params(  # ruff: ignore[private-member-access]
            method, cap, control.METHOD_SUBSTEPS[method], 1e-13
        )
        assert params.max_bond_dim == cap
        assert params.svd_threshold == pytest.approx(1e-13)
        assert params.krylov_tol == pytest.approx(1e-12)
        assert params.trunc_mode == "discarded_weight"

    # The final convention does not retain a zero-weight direction merely to
    # pad a product-state bond to rank two.
    singular_values = np.array([1.0, 0.0])
    assert (
        linalg.truncate(
            singular_values,
            mode="discarded_weight",
            threshold=1e-13,
            max_bond_dim=28,
            min_keep=1,
        )
        == 1
    )


def test_threshold_control_artifacts_are_complete_and_self_consistent() -> None:
    """Every threshold/method pair has a complete, provenance-matched trajectory."""
    manifest = json.loads(control.MANIFEST_PATH.read_text(encoding="utf-8"))
    trajectories = _read_csv(control.ROWS_PATH)
    summaries = _read_csv(control.SUMMARY_PATH)

    assert manifest["campaign_id"] == control.CAMPAIGN_ID
    assert manifest["case"] == control.CASE_KEY
    assert manifest["target_step"] == control.TARGET_STEP
    assert manifest["method_caps"] == control.METHOD_CAPS
    assert manifest["method_substeps"] == control.METHOD_SUBSTEPS
    assert manifest["svd_thresholds"] == list(control.THRESHOLDS)
    assert manifest["trajectory_repeats"] == 1
    assert manifest["timing_repeats"] == 0
    assert manifest["timings_for_publication_comparison"] is False
    assert manifest["threads"] == 1
    assert all(
        int(pool["num_threads"]) == 1 for pool in manifest["threadpools"] if pool["user_api"] in {"blas", "openmp"}
    )
    assert manifest["truncation"] == {
        "cap_applied_after_cutoff_rank": True,
        "exact_zero_padding": False,
        "gate_mpo_hard_split_cutoff": 1e-14,
        "minimum_retained_rank": 1,
        "mode": "discarded_weight",
        "threshold_meaning": "unnormalized cumulative discarded squared singular-value weight",
    }
    assert manifest["control_source_sha256"] == hashlib.sha256(Path(control.__file__).read_bytes()).hexdigest()
    assert manifest["implementation_sha256"] == control._implementation_hash()  # ruff: ignore[private-member-access]
    exact_path = control.BENCHMARK_OUTPUT_DIR / "exact" / f"{control.CASE_KEY}.npy"
    reference_hash = manifest["dense_reference_sha256"]
    assert len(reference_hash) == 64
    assert all(character in "0123456789abcdef" for character in reference_hash)
    if exact_path.is_file():
        assert reference_hash == hashlib.sha256(exact_path.read_bytes()).hexdigest()
    assert manifest["output_sha256"] == {
        "trajectory_rows": hashlib.sha256(control.ROWS_PATH.read_bytes()).hexdigest(),
        "summary_rows": hashlib.sha256(control.SUMMARY_PATH.read_bytes()).hexdigest(),
    }

    expected_pairs = {(method, float(threshold)) for method in control.METHOD_CAPS for threshold in control.THRESHOLDS}
    assert len(summaries) == len(expected_pairs) == 12
    assert len(trajectories) == len(expected_pairs) * (control.TARGET_STEP + 1)

    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in trajectories:
        key = (row["method"], float(row["svd_threshold"]))
        assert key in expected_pairs
        grouped[key].append(row)

    summaries_by_key = {(row["method"], float(row["svd_threshold"])): row for row in summaries}
    assert set(grouped) == set(summaries_by_key) == expected_pairs
    for (method, threshold), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        assert [int(row["step"]) for row in ordered] == list(range(control.TARGET_STEP + 1))
        cap = control.METHOD_CAPS[method]
        assert all(int(row["chi_max"]) == cap for row in ordered)
        assert all(int(row["max_bond_dim"]) <= cap for row in ordered)
        assert all(np.isfinite(float(row["infidelity_normalized"])) for row in ordered)

        summary = summaries_by_key[method, threshold]
        endpoint_infidelity = float(ordered[-1]["infidelity_normalized"])
        maximum_infidelity = max(float(row["infidelity_normalized"]) for row in ordered)
        maximum_parameters = max(int(row["parameter_count"]) for row in ordered)
        maximum_bond = max(int(row["max_bond_dim"]) for row in ordered)
        assert float(summary["endpoint_infidelity"]) == pytest.approx(endpoint_infidelity)
        assert float(summary["max_infidelity_through_target"]) == pytest.approx(maximum_infidelity)
        assert int(summary["max_completed_step_parameter_count"]) == maximum_parameters
        assert int(summary["max_completed_step_bond_dim"]) == maximum_bond


def test_production_threshold_is_inside_the_observed_stable_classification_interval() -> None:
    """The bounded claim holds through 1e-12, but deliberately not through 1e-9."""
    summaries = _read_csv(control.SUMMARY_PATH)
    by_threshold: dict[float, dict[str, float]] = defaultdict(dict)
    for row in summaries:
        by_threshold[float(row["svd_threshold"])][row["method"]] = float(row["max_infidelity_through_target"])

    for threshold in (1e-14, 1e-13, 1e-12):
        assert set(by_threshold[threshold]) == set(control.METHOD_CAPS)
        assert all(error <= 1e-2 for error in by_threshold[threshold].values())
    assert by_threshold[1e-9]["gate_local_2tdvp"] > 1e-2

    for method in control.METHOD_CAPS:
        method_rows = [row for row in summaries if row["method"] == method]
        for field in (
            "endpoint_parameter_count",
            "max_completed_step_parameter_count",
            "endpoint_max_bond_dim",
            "max_completed_step_bond_dim",
        ):
            assert len({int(row[field]) for row in method_rows}) == 1
