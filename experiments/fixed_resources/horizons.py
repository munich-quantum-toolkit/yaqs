# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Horizon helpers for the fixed-χ circuit benchmark."""

from __future__ import annotations

from typing import Any


def reliable_horizon(
    rows: list[dict[str, Any]],
    *,
    epsilon: float,
    dt: float,
) -> dict[str, Any]:
    """Compute ``T_ε`` from a trajectory ordered by trotter step.

    ``T_ε`` is the largest sampled time ``t_m`` such that ``1-F(t_j) < ε`` for every
    ``j ≤ m`` with ``j ≥ 1``. Step 0 (t=0) is ignored. If the first circuit step
    fails the threshold, ``T_ε = 0``. If the trajectory never crosses, the result
    is right-censored at the last simulated positive time.
    """
    by_step = {int(float(r["trotter_step"])): r for r in rows}
    steps = sorted(k for k in by_step if k >= 1)
    max_sim = max((float(by_step[k]["time"]) for k in by_step), default=0.0)
    last_reliable_step = 0
    first_crossing_step: int | None = None
    for k in steps:
        row = by_step[k]
        if int(float(row.get("failed", 0) or 0)):
            first_crossing_step = k
            break
        if float(row["infidelity"]) >= epsilon:
            first_crossing_step = k
            break
        last_reliable_step = k
    t_eps = float(last_reliable_step) * float(dt)
    crossed = first_crossing_step is not None
    right_censored = (not crossed) and max_sim > 0.0
    first_crossing_time = (
        float(first_crossing_step) * float(dt) if first_crossing_step is not None else ""
    )
    return {
        "T_eps": t_eps,
        "n_eps": last_reliable_step,
        "last_reliable_time": t_eps,
        "first_crossing_time": first_crossing_time,
        "first_crossing_step": first_crossing_step if first_crossing_step is not None else "",
        "crossed": int(crossed),
        "right_censored": int(right_censored),
        "max_simulated_time": max_sim,
        "epsilon": epsilon,
    }
