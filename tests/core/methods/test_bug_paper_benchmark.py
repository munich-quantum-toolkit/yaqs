# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regression test for the manuscript's uncompressed BUG benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = (
    PROJECT_ROOT
    / "paper"
    / "bug-mps-benchmarks"
    / "six_site_dense_reference_2026-08-17"
    / "run_benchmark.py"
)


def test_six_site_dense_reference_published_row(tmp_path: Path) -> None:
    """The public runner reproduces the first row and its structural guards."""
    output = tmp_path / "six_site.json"
    subprocess.run(  # noqa: S603
        [sys.executable, str(RUNNER), "--dts", "0.1", "--output", str(output)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload: dict[str, Any] = json.loads(output.read_text(encoding="utf-8"))
    structural = payload["structural_checks"]
    assert structural["mpo_dense_relative_frobenius_error"] < 2e-15
    assert structural["reflected_mpo_dense_relative_frobenius_error"] < 2e-15
    assert structural["site_ordering_gap"] == pytest.approx(0.4338667962246875, abs=1e-12)
    assert structural["reflection_asymmetry_residual"] == pytest.approx(0.4338667962246875, abs=1e-12)
    assert structural["all_endpoints_restored"]
    assert structural["initial_input_tensors_preserved"]

    run = payload["runs"]["0.1"]
    variants = run["variants"]
    assert variants["one_sweep_center"]["phase_aligned_state_error"] == pytest.approx(0.1353, rel=5e-4)
    assert variants["two_sweeps_center"]["phase_aligned_state_error"] == pytest.approx(0.06717, rel=5e-4)
    assert variants["two_sweeps_previous_basis"]["phase_aligned_state_error"] == pytest.approx(
        0.06717,
        rel=5e-4,
    )
    assert run["two_sweep_variant_phase_aligned_difference"] < 5e-7
