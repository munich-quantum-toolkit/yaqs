# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for RZZ ladder gap diagnostic helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
_MODULE_PATH = _EXPERIMENTS / "diagnostic_rzz_lr_ladder_gap.py"


@pytest.fixture(scope="module")
def ladder() -> ModuleType:
    """Load the ladder gap diagnostic module once per worker.

    Returns:
        Imported ladder gap module.
    """
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("diagnostic_rzz_lr_ladder_gap", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ladder_gate_pairs(ladder: ModuleType) -> None:
    """Ladder pairs mirror the benchmark circuit construction."""
    assert ladder.ladder_gate_pairs(6) == [(0, 5), (1, 4), (2, 3)]
    assert ladder.ladder_gate_pairs(8) == [(0, 7), (1, 6), (2, 5), (3, 4)]


def test_task_id_encoding(ladder: ModuleType) -> None:
    """Checkpoint keys encode chi=None and main sweeps correctly."""
    assert ladder.task_id("main_legacy_2site", 10, "plus", None, None) == "main_legacy_2site|L10|plus|chinone|Nna"
    assert ladder.task_id("branch_production", 8, "zeros", 32, 2) == "branch_production|L8|zeros|chi32|N2"


def test_build_analysis_detects_uncapped_gap(ladder: ModuleType) -> None:
    """Analysis notes flag a persistent chi=None fidelity gap."""
    main = ladder.RunRow(
        variant="main_legacy_2site",
        length=10,
        initial_state="plus",
        chi=None,
        tdvp_sweeps=None,
        num_gates=5,
        fidelity=0.99,
        infidelity=0.01,
        norm_deviation=0.0,
        mean_abs_pauli_error=0.001,
        max_abs_pauli_error=0.002,
        mean_abs_x=0.0,
        mean_abs_y=0.0,
        mean_abs_z=0.0,
        endpoint_z0_err=0.0,
        endpoint_zL_err=0.0,
        max_spectator_z_err=0.0,
        max_two_site_err=0.0,
        two_site_err_json="{}",
        max_bond=8,
        mean_bond=4.0,
        bond_profile_final_json="[]",
        total_runtime_s=0.1,
        gate_runtimes_json="[]",
    )
    branch = ladder.RunRow(
        variant="branch_production",
        length=10,
        initial_state="plus",
        chi=None,
        tdvp_sweeps=1,
        num_gates=5,
        fidelity=0.90,
        infidelity=0.10,
        norm_deviation=0.0,
        mean_abs_pauli_error=0.05,
        max_abs_pauli_error=0.1,
        mean_abs_x=0.0,
        mean_abs_y=0.0,
        mean_abs_z=0.0,
        endpoint_z0_err=0.0,
        endpoint_zL_err=0.0,
        max_spectator_z_err=0.0,
        max_two_site_err=0.0,
        two_site_err_json="{}",
        max_bond=8,
        mean_bond=4.0,
        bond_profile_final_json="[]",
        total_runtime_s=0.2,
        gate_runtimes_json="[]",
    )
    analysis = ladder.build_analysis(
        [ladder.CheckpointEntry(run=main, gate_steps=[]), ladder.CheckpointEntry(run=branch, gate_steps=[])],
        lengths=(10,),
    )
    assert any("Q1" in note and "chi=None" in note for note in analysis["notes"])


def test_branch_ladder_smoke(ladder: ModuleType) -> None:
    """Branch production ladder run completes on a tiny grid point."""
    entry = ladder.run_branch_ladder("branch_production", 6, "plus", 64, 1)
    assert entry.run.error is None
    assert entry.run.fidelity is not None
    assert entry.run.fidelity > 0.9
    assert len(entry.gate_steps) == 3
    profiles = [json.loads(step.bond_profile_json) for step in entry.gate_steps]
    assert all(isinstance(p, list) for p in profiles)
    assert np.all(np.array(profiles[-1]) >= 1)
