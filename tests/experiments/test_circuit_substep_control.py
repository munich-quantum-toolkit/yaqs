# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused tests for the selected-cap TDVP substep control."""

from experiments.circuit_benchmarks.extensions.substep_control import summarize_task


def test_substep_summary_uses_the_complete_prefix() -> None:
    """The compact row must retain the worst prefix error and peak storage."""
    rows = [
        {
            "step": step,
            "infidelity_normalized": 0.001 * step,
            "peak_parameter_count": 100 + step,
            "peak_bond_dim": min(28, step + 1),
        }
        for step in range(16)
    ]
    task = {
        "status": "success",
        "payload": {
            "spec": {
                "case": "ising_2d",
                "method": "gate_local_2tdvp",
                "chi_max": 28,
                "n_sub": 4,
            }
        },
        "rows": rows,
    }
    summary = summarize_task(task)
    assert summary["target_step"] == 15
    assert summary["max_infidelity_through"] == 0.015
    assert summary["endpoint_infidelity"] == 0.015
    assert summary["peak_parameter_count"] == 115
    assert summary["peak_bond_dim"] == 16
