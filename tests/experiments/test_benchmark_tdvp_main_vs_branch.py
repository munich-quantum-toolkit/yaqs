# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for TDVP main-vs-branch benchmark helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
_MODULE_PATH = _EXPERIMENTS / "benchmark_tdvp_main_vs_branch.py"


@pytest.fixture(scope="module")
def bm() -> ModuleType:
    """Load the main-vs-branch benchmark module once per worker.

    Returns:
        Imported benchmark module.
    """
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("benchmark_tdvp_main_vs_branch", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(
    bm: ModuleType,
    *,
    checkout: str,
    circuit: str,
    length: int,
    init: str,
    sweeps: int | None,
    fidelity: float,
    pauli: float,
    runtime: float,
) -> bm.CompareRow:
    return bm.CompareRow(
        checkout=checkout,
        circuit_name=circuit,
        circuit_family="family",
        length=length,
        initial_state=init,
        tdvp_sweeps=sweeps,
        runtime_s=runtime,
        fidelity=fidelity,
        infidelity=1.0 - fidelity,
        final_norm=1.0,
        norm_deviation=0.0,
        norm_fail=False,
        mean_abs_pauli_error=pauli,
        max_abs_pauli_error=pauli,
        global_fail=False,
        observable_fail=False,
    )


def test_build_head_to_head_picks_best_branch_sweep(bm: ModuleType) -> None:
    """Head-to-head rows compare main against the best branch sweep."""
    main_row = _row(
        bm,
        checkout="main",
        circuit="single_rzz_lr",
        length=8,
        init="zeros",
        sweeps=None,
        fidelity=0.978,
        pauli=0.001,
        runtime=0.004,
    )
    branch_n1 = _row(
        bm,
        checkout="branch",
        circuit="single_rzz_lr",
        length=8,
        init="zeros",
        sweeps=1,
        fidelity=0.970,
        pauli=0.120,
        runtime=0.007,
    )
    branch_n4 = _row(
        bm,
        checkout="branch",
        circuit="single_rzz_lr",
        length=8,
        init="zeros",
        sweeps=4,
        fidelity=0.987,
        pauli=0.050,
        runtime=0.020,
    )
    summary = bm.build_head_to_head([main_row], [branch_n1, branch_n4])
    assert summary[0]["best_branch_sweeps"] == 4
    assert summary[0]["branch_wins_fidelity"] is True
    assert summary[0]["fidelity_delta_best_branch_minus_main"] == pytest.approx(0.009)


def test_task_id_labels_default_sweeps(bm: ModuleType) -> None:
    """Checkpoint keys distinguish default main runs from branch sweep counts."""
    assert bm.task_id("main", "single_rzz_lr", 8, "zeros", None) == "main|single_rzz_lr|L8|zeros|Ndefault"
    assert bm.task_id("tdvp-sweeps", "single_rzz_lr", 8, "zeros", 4) == "tdvp-sweeps|single_rzz_lr|L8|zeros|N4"


def test_build_aggregate_summary_counts_main_wins(bm: ModuleType) -> None:
    """Aggregate summary tracks fidelity win rates."""
    main_row = _row(
        bm,
        checkout="main",
        circuit="single_rzz_lr",
        length=10,
        init="plus",
        sweeps=None,
        fidelity=0.980,
        pauli=0.001,
        runtime=0.005,
    )
    branch_row = _row(
        bm,
        checkout="branch",
        circuit="single_rzz_lr",
        length=10,
        init="plus",
        sweeps=8,
        fidelity=0.960,
        pauli=0.100,
        runtime=0.050,
    )
    head_to_head = bm.build_head_to_head([main_row], [branch_row])
    aggregate = bm.build_aggregate_summary([main_row], [branch_row], head_to_head)
    assert aggregate["main_fidelity_wins"] == 1
    assert aggregate["branch_best_fidelity_wins"] == 0
