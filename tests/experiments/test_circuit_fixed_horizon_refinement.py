# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused tests for the fixed-horizon cap refinement."""

from __future__ import annotations

import pytest

from experiments.circuit_benchmarks.config import METHODS
from experiments.circuit_benchmarks.extensions.fixed_horizon_refinement import (
    MPS_ENTRY_BYTES,
    combine_and_select,
)


def _coarse_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method_index, method in enumerate(METHODS):
        for cap, error in ((4, 0.02), (8, 0.005)):
            parameters = (method_index + 1) * cap * 10
            rows.append(
                {
                    "target_step": "15",
                    "target_time": "1.5",
                    "method": method,
                    "chi_max": str(cap),
                    "complete": "True",
                    "reliable": str(error <= 0.01),
                    "achieved_infidelity": str(error),
                    "max_infidelity_through": str(error),
                    "peak_parameter_count": str(parameters),
                    "peak_bond_dim": str(cap),
                    "task_id": f"coarse-{method}-{cap}",
                }
            )
    return rows


def _refined_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(METHODS):
        cap = 6
        parameters = (method_index + 1) * cap * 10
        rows.append(
            {
                "method": method,
                "chi_max": cap,
                "source": "refined",
                "task_id": f"refined-{method}-{cap}",
                "target_step": 15,
                "target_time": 1.5,
                "max_infidelity_through": 0.008,
                "endpoint_infidelity": 0.008,
                "reliable": True,
                "peak_parameter_count": parameters,
                "peak_mps_bytes": MPS_ENTRY_BYTES * parameters,
                "peak_bond_dim": cap,
            }
        )
    return rows


def test_refinement_selects_lower_storage_passing_caps_and_brackets_them() -> None:
    """The refined passing point should replace the coarse one and retain its last failure."""
    combined, selected = combine_and_select(_coarse_rows(), _refined_rows())

    assert len(selected) == len(METHODS)
    for method, row in zip(METHODS, selected, strict=True):
        assert row["method"] == method
        assert row["selected_chi_max"] == 6
        assert row["last_fail_chi_max"] == 4
        method_rows = [candidate for candidate in combined if candidate["method"] == method]
        assert sum(bool(candidate["selected"]) for candidate in method_rows) == 1
        assert sum(bool(candidate["last_fail"]) for candidate in method_rows) == 1


def test_refinement_rejects_duplicate_method_cap_points() -> None:
    """A refined task must not silently replace a frozen point at the same cap."""
    refined = _refined_rows()
    refined[0]["chi_max"] = 4

    with pytest.raises(RuntimeError, match="Duplicate combined cap point"):
        combine_and_select(_coarse_rows(), refined)


def test_refinement_rejects_an_inconsistent_reliability_flag() -> None:
    """Reliability must be recomputable from the recorded worst infidelity."""
    refined = _refined_rows()
    refined[0]["reliable"] = False

    with pytest.raises(RuntimeError, match="Inconsistent reliability"):
        combine_and_select(_coarse_rows(), refined)
