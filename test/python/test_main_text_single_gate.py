# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Tests for the main-text single RZZ gate benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SINGLE_GATE_DIR = Path(__file__).resolve().parents[2] / "experiments" / "single_gate"
sys.path.insert(0, str(SINGLE_GATE_DIR))

from config import (  # noqa: E402
    CHI0,
    FIGURES_DIR,
    FIGURE_STEM,
    OUTPUT_DIR,
    build_generic_angle_grid,
    build_special_angles,
    pick_intermediate_chi,
)
from gate_convention_checks import check_gate_convention  # noqa: E402


def test_chi0_is_eight() -> None:
    assert CHI0 == 8


def test_intermediate_chi_rule() -> None:
    assert pick_intermediate_chi(8, 16) == 12
    assert pick_intermediate_chi(8, 64) == 24


def test_generic_angle_grid_excludes_special() -> None:
    x_gen, _ = build_generic_angle_grid()
    x_spec, _ = build_special_angles()
    for x in x_gen:
        assert not any(abs(float(x) - s) < 1e-12 for s in x_spec)
    assert len(x_gen) >= 22


def test_rzz_gate_convention() -> None:
    assert check_gate_convention() == []


def test_main_text_outputs_exist() -> None:
    out = OUTPUT_DIR
    if not (out / "results.sqlite").exists():
        pytest.skip("main-text benchmark not run")
    for name in (
        "single_gate_angle_sweep.csv",
        "single_gate_substeps.csv",
        "single_gate_chi_scan.csv",
        "single_gate_mpo_diagnostics.csv",
        "single_gate_validation.md",
        "theta_zero_diagnostics.csv",
        "theta_zero_diagnostics.md",
    ):
        assert (out / name).is_file(), name
    for name in (f"{FIGURE_STEM}.pdf", f"{FIGURE_STEM}.png"):
        assert (FIGURES_DIR / name).is_file(), name
