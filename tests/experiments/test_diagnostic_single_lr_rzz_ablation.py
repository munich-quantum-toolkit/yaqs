# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for single LR RZZ ablation diagnostic helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
_MODULE_PATH = _EXPERIMENTS / "diagnostic_single_lr_rzz_ablation.py"


@pytest.fixture(scope="module")
def diag() -> ModuleType:
    """Load the ablation diagnostic module once per worker.

    Returns:
        Imported diagnostic module.
    """
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("diagnostic_single_lr_rzz_ablation", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_reference_circuit_shape(diag: ModuleType) -> None:
    """Reference circuit uses the requested number of qubits."""
    qc = diag.build_reference_circuit(8)
    assert qc.num_qubits == 8


def test_task_id_stable(diag: ModuleType) -> None:
    """Checkpoint keys encode variant, length, chi, and sweeps."""
    assert diag.task_id("current", 10, 16, 4) == "current|L10|chi16|N4"
    assert diag.task_id("main_legacy_2site", 6, None, None) == "main_legacy_2site|L6|chinone|Nna"


def test_exact_reference_normalized(diag: ModuleType) -> None:
    """Exact reference statevector has unit norm."""
    ref = diag.exact_reference(6)
    assert np.isclose(np.linalg.norm(ref), 1.0)


def test_iter_tasks_quick_includes_main(diag: ModuleType) -> None:
    """Quick grid includes branch variants and main legacy baseline."""
    tasks = diag.iter_tasks(quick=True, variant_filter=None)
    variants = {t[0] for t in tasks}
    assert "current" in variants
    assert "main_legacy_2site" in variants


def test_build_hypothesis_summary_notes_main_win(diag: ModuleType) -> None:
    """Hypothesis summary flags when main beats branch current at high chi."""
    main_run = diag.RunRow(
        variant="main_legacy_2site",
        length=10,
        chi=64,
        tdvp_sweeps=None,
        fidelity=0.99,
        infidelity=0.01,
        norm_deviation=0.0,
        mean_abs_pauli_error=0.001,
        max_abs_pauli_error=0.001,
        mean_abs_x=0.0,
        mean_abs_y=0.0,
        mean_abs_z=0.0,
        endpoint_z0_err=0.0,
        endpoint_zL_err=0.0,
        max_spectator_z_err=0.0,
        xx_err=0.0,
        yy_err=0.0,
        zz_err=0.0,
        xy_err=0.0,
        yx_err=0.0,
        max_bond=2,
        mean_bond=2.0,
        max_bond_crossed=2,
        bond_profile="[2,2,2,2,2,2,2,2,2]",
        runtime_s=0.01,
    )
    branch_run = diag.RunRow(
        variant="current",
        length=10,
        chi=64,
        tdvp_sweeps=1,
        fidelity=0.95,
        infidelity=0.05,
        norm_deviation=0.0,
        mean_abs_pauli_error=0.05,
        max_abs_pauli_error=0.05,
        mean_abs_x=0.0,
        mean_abs_y=0.0,
        mean_abs_z=0.05,
        endpoint_z0_err=0.05,
        endpoint_zL_err=0.05,
        max_spectator_z_err=0.05,
        xx_err=0.0,
        yy_err=0.0,
        zz_err=0.0,
        xy_err=0.0,
        yx_err=0.0,
        max_bond=2,
        mean_bond=2.0,
        max_bond_crossed=2,
        bond_profile="[2,2,2,2,2,2,2,2,2]",
        runtime_s=0.02,
    )
    summary = diag.build_hypothesis_summary([main_run, branch_run], [])
    assert any("Main legacy beats branch current" in note for note in summary["notes"])
