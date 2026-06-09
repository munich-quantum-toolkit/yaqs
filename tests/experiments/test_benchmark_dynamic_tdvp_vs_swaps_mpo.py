# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for dynamic TDVP vs swaps/MPO benchmark helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
_MODULE_PATH = _EXPERIMENTS / "benchmark_dynamic_tdvp_vs_swaps_mpo.py"


@pytest.fixture(scope="module")
def bm() -> ModuleType:
    """Load the circuit-method benchmark module once per worker.

    Returns:
        Imported benchmark module.
    """
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("benchmark_dynamic_tdvp_vs_swaps_mpo", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_is_good_enough_accepts_practical_targets(bm: ModuleType) -> None:
    """Runs within 1e-2 infidelity and observable error count as good enough."""
    row = bm.RunRow(
        row_key="k",
        regime="fixed_chi",
        method="dynamic_tdvp",
        L=8,
        circuit_family="family1_diagonal_lr",
        circuit_name="single_rzz_lr",
        depth=1,
        num_lr_gates=1,
        num_nn_gates=0,
        geometry_mismatch=False,
        initial_state="plus",
        max_bond_dim=16,
        svd_threshold=1e-14,
        tdvp_sweeps=4,
        runtime_s=0.1,
        max_bond_seen=16,
        mean_bond_seen=8.0,
        final_norm=1.0,
        fidelity=0.995,
        infidelity=0.005,
        mean_abs_pauli_error=0.004,
        max_abs_pauli_error=0.01,
        mean_abs_pauli_x=0.004,
        mean_abs_pauli_y=0.004,
        mean_abs_pauli_z=0.004,
        mean_magnetization_z=0.0,
        max_entanglement_entropy=0.5,
        reference_type="exact",
        global_fail=False,
        observable_fail=False,
        large_global_fail=False,
        large_observable_fail=False,
        norm_fail=False,
        high_fidelity_high_observable_error=False,
        tdvp_best_over_sweeps=True,
        seed=0,
    )
    assert bm.is_good_enough(row)


def test_build_grid_medium_uses_all_circuits(bm: ModuleType) -> None:
    """Medium profile spans every named circuit family."""
    grid = bm.build_grid(
        regime="fixed_chi",
        quick=False,
        medium=True,
        full=False,
        targeted_rerun=False,
        max_exact_L=10,
        families=None,
        methods=("dynamic_tdvp", "swaps", "mpo"),
        chi_list=None,
        sweeps_list=None,
        seed=0,
    )
    assert set(grid.circuits) == set(bm.ALL_CIRCUITS)
    assert grid.L_values == (8, 10)
    assert grid.initial_states == bm.STRUCTURED_INITS


def test_build_grid_rejects_medium_with_quick(bm: ModuleType) -> None:
    """Medium and quick profiles are mutually exclusive."""
    with pytest.raises(ValueError, match="only one"):
        bm.build_grid(
            regime="fixed_chi",
            quick=True,
            medium=True,
            full=False,
            targeted_rerun=False,
            max_exact_L=10,
            families=None,
            methods=("swaps",),
            chi_list=None,
            sweeps_list=None,
            seed=0,
        )
