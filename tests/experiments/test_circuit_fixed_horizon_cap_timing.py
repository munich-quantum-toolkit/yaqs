# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused tests for the full-cap fixed-horizon timing sweep."""

from __future__ import annotations

import pytest

from experiments.circuit_benchmarks.config import METHODS, TIMING_REPEATS
from experiments.circuit_benchmarks.extensions.fixed_horizon_cap_timing import (
    cap_specs_from_sweep,
    summarize_timing_rows,
)


def _sweep_rows() -> list[dict[str, str]]:
    return [
        {
            "method": method,
            "chi_max": str(cap),
            "target_step": "15",
            "endpoint_infidelity": str(0.1 / cap),
        }
        for method in reversed(METHODS)
        for cap in (8, 4)
    ]


def test_cap_specs_are_unique_and_follow_manuscript_order() -> None:
    """The campaign should time every tested cap exactly once."""
    specs = cap_specs_from_sweep(_sweep_rows())

    assert [(method, cap) for method, cap, _endpoint in specs] == [
        (method, cap) for method in METHODS for cap in (4, 8)
    ]


def test_cap_specs_reject_duplicate_points() -> None:
    """Duplicate cap rows must not silently trigger repeated timings."""
    rows = _sweep_rows()
    rows.append(dict(rows[0]))

    with pytest.raises(RuntimeError, match="Duplicate cap-sweep row"):
        cap_specs_from_sweep(rows)


def test_timing_summary_retains_three_repeat_range() -> None:
    """Each cap should be summarized by its median and full measured range."""
    rows = [
        {
            "method": method,
            "chi_max": cap,
            "runtime_s": runtime,
        }
        for method in METHODS
        for cap in (4, 8)
        for runtime in (3.0, 1.0, 2.0)
    ]
    summaries = summarize_timing_rows(rows)

    assert len(summaries) == 2 * len(METHODS)
    assert [row["median_s"] for row in summaries] == pytest.approx([2.0] * len(summaries))
    assert [row["min_s"] for row in summaries] == pytest.approx([1.0] * len(summaries))
    assert [row["max_s"] for row in summaries] == pytest.approx([3.0] * len(summaries))
    assert all(row["repeats"] == TIMING_REPEATS for row in summaries)


def test_timing_summary_requires_the_full_repeat_count() -> None:
    """Incomplete timing groups should fail before plotting."""
    rows = [
        {"method": METHODS[0], "chi_max": 4, "runtime_s": runtime}
        for runtime in (1.0, 2.0)
    ]

    with pytest.raises(RuntimeError, match=f"Expected {TIMING_REPEATS}"):
        summarize_timing_rows(rows)
