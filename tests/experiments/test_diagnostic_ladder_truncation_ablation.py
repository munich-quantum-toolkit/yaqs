# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for ladder truncation ablation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
_MODULE_PATH = _EXPERIMENTS / "diagnostic_ladder_truncation_ablation.py"


@pytest.fixture(scope="module")
def trunc_ablation() -> ModuleType:
    """Load truncation ablation module once per worker.

    Returns:
        Imported truncation ablation module.
    """
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("diagnostic_ladder_truncation_ablation", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_truncation_invariant_passes_when_fidelities_match(trunc_ablation: ModuleType) -> None:
    """χ=64 with low threshold should match uncapped when observed χ stays below cap."""
    ok, msg = trunc_ablation.check_truncation_invariant(0.913, 0.913, max_observed_bond=4, svd_threshold=0.0)
    assert ok
    assert msg == "ok"


def test_truncation_invariant_fails_on_large_gap(trunc_ablation: ModuleType) -> None:
    """Invariant flags mismatch even when observed χ is small."""
    ok, _msg = trunc_ablation.check_truncation_invariant(0.913, 0.234, max_observed_bond=4, svd_threshold=0.0)
    assert not ok


def test_ladder_smoke_uncapped_vs_chi64(trunc_ablation: ModuleType) -> None:
    """L=10 plus: χ=64 matches uncapped when observed bonds stay below the cap."""
    _steps_u, sum_u = trunc_ablation.run_policy_case(
        policy="uncapped_current",
        max_bond_dim=None,
        svd_threshold=1e-14,
        length=10,
        seed_prep=True,
    )
    _steps_c, sum_c = trunc_ablation.run_policy_case(
        policy="chi64_current",
        max_bond_dim=64,
        svd_threshold=1e-14,
        length=10,
        seed_prep=True,
    )
    assert sum_u.final_fidelity == pytest.approx(sum_c.final_fidelity, abs=1e-10)
    assert sum_u.max_observed_bond <= 64
    assert sum_c.max_observed_bond <= 64
    ok, _msg = trunc_ablation.check_truncation_invariant(
        sum_u.final_fidelity,
        sum_c.final_fidelity,
        max_observed_bond=sum_c.max_observed_bond,
        svd_threshold=1e-14,
    )
    assert ok
