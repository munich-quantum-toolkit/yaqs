# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for TDVP families vs main benchmark helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
_MODULE_PATH = _EXPERIMENTS / "benchmark_tdvp_families_vs_main.py"


@pytest.fixture(scope="module")
def fam_bm() -> ModuleType:
    """Load families benchmark module once per worker.

    Returns:
        Imported benchmark module.
    """
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("benchmark_tdvp_families_vs_main", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_row(
    fam_bm: ModuleType,
    *,
    variant: str,
    fidelity: float,
    pauli: float,
    max_pauli: float,
    family: str = "hamiltonian",
) -> fam_bm.RunRow:
    return fam_bm.RunRow(
        run_id=f"{variant}|test|L6|zeros|seedna|chi16|exact|th1e-10|{family}",
        variant=variant,  # type: ignore[arg-type]
        checkout="main" if variant == "main_tdvp" else "branch",
        circuit_name="rzz_lr_ladder",
        circuit_family=family,  # type: ignore[arg-type]
        length=6,
        initial_state="zeros",
        random_seed=None,
        chi=16,
        preset="exact",
        svd_threshold=1e-10,
        tdvp_sweeps=None,
        seed_prep=None,
        runtime_s=0.01,
        runtime_per_gate_s=0.01,
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
        energy_density_error=pauli,
        norm_deviation=0.0,
        max_bond=4,
        bond_profile="[2,2,2,2,2]",
        renorm_count=0,
        enforce_cap_count=0,
    )


def test_classify_fidelity_branch_win(fam_bm: ModuleType) -> None:
    """Branch win when infidelity drops materially."""
    main = _run_row(fam_bm, variant="main_tdvp", fidelity=0.95, pauli=0.10, max_pauli=0.12)
    branch = _run_row(fam_bm, variant="branch_production", fidelity=0.97, pauli=0.07, max_pauli=0.11)
    assert fam_bm.classify_fidelity(main, branch) == "branch_win"


def test_classify_fidelity_branch_loss(fam_bm: ModuleType) -> None:
    """Branch loss when fidelity drops >1e-3 vs main."""
    main = _run_row(fam_bm, variant="main_tdvp", fidelity=1.0, pauli=0.0, max_pauli=0.0)
    branch = _run_row(fam_bm, variant="branch_production", fidelity=0.984, pauli=0.002, max_pauli=0.03)
    assert fam_bm.classify_fidelity(main, branch) == "branch_loss"


def test_build_family_circuit_discrete(fam_bm: ModuleType) -> None:
    """Discrete extensions build LR CZ gate."""
    spec = fam_bm.build_family_circuit("single_cz_lr", 8)
    assert spec.depth == 1
    assert spec.num_lr_gates == 1


def test_quick_grid_non_empty(fam_bm: ModuleType) -> None:
    """Quick grid produces headline tasks for all four families."""
    import argparse

    args = argparse.Namespace(quick=True, family=None, with_sweeps=False)
    headline, opt_in = fam_bm.grid_from_args(args)
    assert len(headline) > 0
    assert opt_in == []
    families = {t.family for t in headline}
    assert families == {"hamiltonian", "discrete", "observables", "fidelity"}
