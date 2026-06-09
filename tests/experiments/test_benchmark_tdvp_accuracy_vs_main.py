# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for TDVP accuracy vs main benchmark helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
_MODULE_PATH = _EXPERIMENTS / "benchmark_tdvp_accuracy_vs_main.py"


@pytest.fixture(scope="module")
def acc_bm() -> ModuleType:
    """Load accuracy benchmark module once per worker.

    Returns:
        Imported benchmark module.
    """
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("benchmark_tdvp_accuracy_vs_main", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_row(
    acc_bm: ModuleType,
    *,
    variant: str,
    fidelity: float,
    pauli: float,
    max_pauli: float,
    runtime: float = 0.01,
) -> acc_bm.RunRow:
    return acc_bm.RunRow(
        run_id=f"{variant}|test|L6|zeros|seedna|chi16|exact|th1e-10",
        variant=variant,  # type: ignore[arg-type]
        checkout="main" if variant == "main_default" else "branch",
        circuit_name="rzz_lr_ladder",
        circuit_family="ladder",
        length=6,
        initial_state="zeros",
        random_seed=None,
        chi=16,
        preset="exact",
        svd_threshold=1e-10,
        tdvp_sweeps=None,
        seed_prep=None,
        runtime_s=runtime,
        runtime_per_gate_s=runtime,
        num_gates=3,
        fidelity=fidelity,
        infidelity=1.0 - fidelity,
        mean_abs_pauli_error=pauli,
        max_abs_pauli_error=max_pauli,
        mean_abs_pauli_x=pauli,
        mean_abs_pauli_y=pauli,
        mean_abs_pauli_z=pauli,
        endpoint_mean_pauli_error=pauli,
        spectator_z_error=pauli,
        two_site_xx_error=pauli,
        two_site_yy_error=pauli,
        two_site_zz_error=pauli,
        norm_deviation=0.0,
        max_bond=4,
        bond_profile="[2,2,2,2,2]",
        renorm_count=0,
        enforce_cap_count=0,
    )


def test_classify_branch_win(acc_bm: ModuleType) -> None:
    """Branch win when Pauli error drops by >=20% without max regression."""
    main = _run_row(acc_bm, variant="main_default", fidelity=0.95, pauli=0.10, max_pauli=0.12)
    branch = _run_row(acc_bm, variant="branch_production", fidelity=0.97, pauli=0.07, max_pauli=0.11)
    assert acc_bm.classify_pair(main, branch) == "branch_win"


def test_classify_tie_near_equal(acc_bm: ModuleType) -> None:
    """Near-equal fidelity and observables classify as tie."""
    main = _run_row(acc_bm, variant="main_default", fidelity=0.977668, pauli=1e-8, max_pauli=1e-8)
    branch = _run_row(acc_bm, variant="branch_production", fidelity=0.977668, pauli=1e-8, max_pauli=1e-8)
    assert acc_bm.classify_pair(main, branch) == "tie"


def test_build_extended_circuit_mixed_axis(acc_bm: ModuleType) -> None:
    """Extended ladder builder produces gates."""
    spec = acc_bm.build_extended_circuit("mixed_axis_ladder", 8)
    assert spec.depth >= 4
    assert spec.family == "ladder"
