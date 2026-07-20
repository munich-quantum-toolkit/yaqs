# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Tests for θ=0 identity-limit diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SINGLE_GATE_DIR = Path(__file__).resolve().parents[2] / "experiments" / "single_gate"
sys.path.insert(0, str(SINGLE_GATE_DIR))

from theta_zero_diagnostics import (  # noqa: E402
    THETA_ZERO_INF_TOL,
    run_theta_zero_diagnostics,
    verify_gate_construction,
)


@pytest.fixture(scope="module")
def diagnostic_report():
    return run_theta_zero_diagnostics()


def test_gate_matrix_is_identity_at_theta_zero() -> None:
    rows = verify_gate_construction()
    dense = [r for r in rows if r.representation.startswith("dense")]
    assert dense
    assert all(r.u_minus_i_l2 <= 1e-10 for r in dense)
    mpo = next(r for r in rows if r.representation == "mpo_from_gate_theta0")
    assert mpo.mpo_vs_identity_fro is not None
    assert mpo.mpo_vs_identity_fro <= 1e-12


def test_theta_zero_requirements(diagnostic_report) -> None:
    theta0 = [r for r in diagnostic_report.algorithm_runs if r.section == "theta_zero"]
    for method in ("hybrid_tdvp", "mpo_zipup", "variational_mpo"):
        for chi in (8, 12, 16):
            row = next(r for r in theta0 if r.method == method and r.chi_max == chi)
            assert row.exact_infidelity <= THETA_ZERO_INF_TOL, (method, chi, row.exact_infidelity)
            assert row.variational_worse_than_input is not True
    tebd16 = next(r for r in theta0 if r.method == "tebd_swap" and r.chi_max == 16)
    assert tebd16.exact_infidelity <= THETA_ZERO_INF_TOL
    assert diagnostic_report.summary["implementation_bug"] is False
