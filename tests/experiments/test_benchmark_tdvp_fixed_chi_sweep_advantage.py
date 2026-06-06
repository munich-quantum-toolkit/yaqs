# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for fixed-χ sweep-advantage benchmark helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
_MODULE_PATH = _EXPERIMENTS / "benchmark_tdvp_fixed_chi_sweep_advantage.py"


@pytest.fixture(scope="module")
def bm() -> ModuleType:
    """Load the benchmark module once per worker.

    Returns:
        Imported sweep-advantage benchmark module.
    """
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("benchmark_tdvp_fixed_chi_sweep_advantage", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_beats_baseline_when_factor_large_enough(bm: ModuleType) -> None:
    """TDVP wins when infidelity is at least a factor below baseline."""
    assert bm.beats_baseline(1e-4, 5e-4)


def test_beats_baseline_rejects_marginal_improvement(bm: ModuleType) -> None:
    """Marginal TDVP improvements below the win factor do not count."""
    assert not bm.beats_baseline(3e-4, 5e-4)


def test_beats_baseline_rejects_machine_precision_ties(bm: ModuleType) -> None:
    """Near-exact baselines cannot be beaten at machine precision."""
    assert not bm.beats_baseline(1e-13, 1e-13)


def test_classify_baseline_exact_when_baseline_near_machine_precision(bm: ModuleType) -> None:
    """Near-exact baselines are labeled baseline_exact even if TDVP is only good."""
    label = bm.classify_case(
        swaps_infidelity=1e-14,
        mpo_infidelity=1e-14,
        tdvp_by_sweep={1: 1e-8, 64: 1e-9},
        tdvp_improves=True,
        tdvp_beats_best=False,
        tdvp_n1_worse_than_baseline=True,
    )
    assert label == "baseline_exact"


def test_classify_sweep_crossing_win(bm: ModuleType) -> None:
    """Classification detects worse-at-N1 but better-at-larger-N cases."""
    label = bm.classify_case(
        swaps_infidelity=8e-3,
        mpo_infidelity=1e-2,
        tdvp_by_sweep={1: 3e-2, 64: 4e-4},
        tdvp_improves=True,
        tdvp_beats_best=True,
        tdvp_n1_worse_than_baseline=True,
    )
    assert label == "tdvp_sweep_crossing_win"


def test_build_gate_circuit_lr_and_nn(bm: ModuleType) -> None:
    """Gate builders set long-range vs nearest-neighbor metadata."""
    lr = bm.build_gate_circuit("rzz_lr", 8)
    nn = bm.build_gate_circuit("rzz_nn", 8)
    assert lr.gate_range == "lr"
    assert lr.sites == (0, 7)
    assert nn.gate_range == "nn"
    assert nn.sites == (0, 1)
    assert lr.qc.size() == 1


def test_internal_max_bond_ignores_physical_dim(bm: ModuleType) -> None:
    """Bond cap checks use internal bonds, not physical dimension 2."""
    mps = bm.prep_initial_state("plus", 8, chi=4, seed=0)
    assert bm.internal_max_bond(mps) == 1


def test_build_quick_grid_defaults(bm: ModuleType) -> None:
    """Quick profile includes the documented gate and init subsets."""
    grid = bm.build_grid(
        quick=True,
        full=False,
        length_list=None,
        chi_list=None,
        sweeps_list=None,
        seeds=None,
        initial_states=None,
        gates=None,
    )
    assert grid.L_values == (8, 10)
    assert "haar_mps_chi" in grid.initial_states
    assert "rzz_lr" in grid.gates


@pytest.mark.parametrize("name", ["rzz_lr_inner"])
def test_inner_gate_requires_minimum_length(bm: ModuleType, name: str) -> None:
    """Inner long-range gates are skipped on very short chains."""
    with pytest.raises(ValueError, match="requires L >= 4"):
        bm.build_gate_circuit(name, 3)
