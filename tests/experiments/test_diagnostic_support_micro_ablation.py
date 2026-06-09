# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for support micro ablation diagnostic helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
_MODULE_PATH = _EXPERIMENTS / "diagnostic_support_micro_ablation.py"


@pytest.fixture(scope="module")
def micro() -> ModuleType:
    """Load the support micro ablation module once per worker.

    Returns:
        Imported micro ablation module.
    """
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("diagnostic_support_micro_ablation", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_task_id_encoding(micro: ModuleType) -> None:
    """Checkpoint keys encode chi=None correctly."""
    assert micro.task_id("support_seed_only", 10, None, 4) == "support_seed_only|L10|chinone|N4"


def test_crossed_rank_status(micro: ModuleType) -> None:
    """Rank adequacy flags bonds below minimum support rank."""
    status = micro.crossed_rank_status([1, 2, 1], frozenset({0, 1}), min_rank=2)
    assert status[0] is False
    assert status[1] is True


def test_build_interpretation_detects_support_harm(micro: ModuleType) -> None:
    """Interpretation notes flag harmful support when spectator Z blows up."""
    bad = micro.MicroRunRow(
        variant="support_all_crossed_no_prepad",
        length=6,
        chi=64,
        tdvp_sweeps=1,
        fidelity=0.98,
        norm_deviation=0.0,
        signed_z_err_json=json.dumps({str(i): 0.5 for i in range(6)}),
        x_err_json="{}",
        y_err_json="{}",
        dense_z_json="{}",
        mps_z_json="{}",
        dense_x_json="{}",
        mps_x_json="{}",
        dense_y_json="{}",
        mps_y_json="{}",
        max_dense_mps_z_gap=0.0,
        max_dense_mps_x_gap=0.0,
        max_dense_mps_y_gap=0.0,
        observable_path_ok=True,
        bond_before_json="[]",
        bond_after_json="[]",
        crossed_bonds_json="[]",
        crossed_rank_before_json="{}",
        crossed_rank_after_json="{}",
        all_crossed_rank_ok_after=True,
        runtime_s=0.01,
    )
    good = micro.MicroRunRow(
        variant="dynamic_no_support_baseline",
        length=6,
        chi=64,
        tdvp_sweeps=1,
        fidelity=0.98,
        norm_deviation=0.0,
        signed_z_err_json=json.dumps({str(i): 0.0 for i in range(6)}),
        x_err_json="{}",
        y_err_json="{}",
        dense_z_json="{}",
        mps_z_json="{}",
        dense_x_json="{}",
        mps_x_json="{}",
        dense_y_json="{}",
        mps_y_json="{}",
        max_dense_mps_z_gap=0.0,
        max_dense_mps_x_gap=0.0,
        max_dense_mps_y_gap=0.0,
        observable_path_ok=True,
        bond_before_json="[]",
        bond_after_json="[]",
        crossed_bonds_json="[]",
        crossed_rank_before_json="{}",
        crossed_rank_after_json="{}",
        all_crossed_rank_ok_after=True,
        runtime_s=0.01,
    )
    notes = micro.build_interpretation_notes([bad, good])
    assert any("support_all_crossed_no_prepad" in n for n in notes)
