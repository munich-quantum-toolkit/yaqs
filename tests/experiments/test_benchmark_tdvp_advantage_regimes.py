# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for TDVP advantage regimes benchmark helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

if TYPE_CHECKING:
    from types import ModuleType

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
_MODULE_PATH = _EXPERIMENTS / "benchmark_tdvp_advantage_regimes.py"


@pytest.fixture(scope="module")
def bm() -> ModuleType:
    """Load the advantage-regimes benchmark module once per worker.

    Returns:
        Imported benchmark module.
    """
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("benchmark_tdvp_advantage_regimes", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_beats_baseline_requires_win_factor(bm: ModuleType) -> None:
    """Accuracy wins require at least 2x error reduction."""
    assert bm.beats_baseline(1e-4, 5e-4)
    assert not bm.beats_baseline(3e-4, 5e-4)


def test_bucket_all_exact(bm: ModuleType) -> None:
    """All-exact cases are labeled without mixing in failure buckets."""
    labels = bm.bucket_same_chi(
        swaps_infidelity=1e-12,
        mpo_infidelity=1e-12,
        swaps_obs_score=1e-12,
        mpo_obs_score=1e-12,
        best_tdvp_infidelity=1e-12,
        best_tdvp_obs_score=1e-12,
        tdvp_n1_obs_score=1e-12,
        tdvp_beats_obs_score=False,
        tdvp_beats_fid=False,
        tdvp_beats_task_energy=False,
        tdvp_beats_active_corr=False,
    )
    assert labels == ["all_exact"]


def test_bucket_all_fail_when_no_loose_target(bm: ModuleType) -> None:
    """Cases where no method reaches the loose obs_score target are all_fail."""
    labels = bm.bucket_same_chi(
        swaps_infidelity=5e-2,
        mpo_infidelity=6e-2,
        swaps_obs_score=5e-2,
        mpo_obs_score=6e-2,
        best_tdvp_infidelity=7e-2,
        best_tdvp_obs_score=7e-2,
        tdvp_n1_obs_score=8e-2,
        tdvp_beats_obs_score=False,
        tdvp_beats_fid=False,
        tdvp_beats_task_energy=False,
        tdvp_beats_active_corr=False,
    )
    assert labels == ["all_fail"]


def test_obs_score_uses_max_component(bm: ModuleType) -> None:
    """Composite obs_score is the max of Pauli, active corr, and task energy errors."""
    score = bm.composite_obs_score(
        mean_abs_pauli_error=1e-4,
        mean_abs_active_corr_error=2e-3,
        abs_task_energy_error=5e-4,
    )
    assert score == pytest.approx(2e-3)


def test_active_correlation_errors_long_range(bm: ModuleType) -> None:
    """Active-edge correlations include long-range pairs, not only NN."""
    qc = QuantumCircuit(4)
    qc.h(range(4))
    qc.rzz(0.3, 0, 3)
    vec = np.asarray(Statevector(qc).data, dtype=np.complex128)
    perturbed = vec.copy()
    perturbed[0] += 0.01
    perturbed /= np.linalg.norm(perturbed)
    err = bm.active_correlation_errors(perturbed, vec, active_pairs=((0, 3),))
    assert err.mean_abs > 0.0


def test_build_quick_grid(bm: ModuleType) -> None:
    """Quick profile matches documented use-case search grid."""
    grid = bm.build_grid(quick=True, focused=False)
    assert grid.L_values == (8, 10)
    assert grid.chi_values == (1, 2, 4, 8, 16, 32)
    assert grid.seeds == (0,)
    assert "low_depth" in grid.initial_states
    assert "grid_2d_rzz" in grid.circuits


def _raw_run_base() -> dict[str, object]:
    return {
        "L": 8,
        "chi": 8,
        "seed": 0,
        "theta": 0.3,
        "initial_state": "low_depth",
        "circuit_family": "family_c_commuting",
        "circuit": "grid_2d_rzz",
        "depth": 1,
        "num_lr_gates": 2,
        "num_nn_gates": 0,
        "geometry_mismatch": True,
        "reference_type": "exact",
        "max_abs_pauli_error": 0.0,
        "mean_abs_x_error": 0.0,
        "mean_abs_y_error": 0.0,
        "mean_abs_z_error": 0.0,
        "mean_abs_active_corr_error": 0.0,
        "max_abs_active_corr_error": 0.0,
        "mean_abs_active_xx_error": 0.0,
        "mean_abs_active_yy_error": 0.0,
        "mean_abs_active_zz_error": 0.0,
        "mean_nn_corr_error": 0.0,
        "mean_magnetization_z": 0.0,
        "mean_magnetization_x": 0.0,
        "task_energy_ref": 0.0,
        "task_energy_method": 0.0,
        "abs_task_energy_error": 0.0,
        "obs_score": 0.0,
        "initial_state_cap_respected": True,
        "max_entanglement_entropy": 0.0,
        "max_bond_seen": 2,
        "mean_bond_seen": 1.0,
        "bond_dims_final": "[]",
        "norm": 1.0,
        "cap_respected": True,
        "crashed": False,
        "exception_message": "",
    }


def test_analyze_same_chi_uses_obs_score(bm: ModuleType) -> None:
    """Fixed-χ analysis picks best TDVP over sweeps by obs_score."""
    base = _raw_run_base()
    rows = [
        bm.RawRun(
            row_key="swaps",
            method="swaps",
            tdvp_sweeps=1,
            runtime_s=0.01,
            fidelity=0.99,
            infidelity=0.01,
            mean_abs_pauli_error=0.02,
            **{**base, "obs_score": 0.02},
        ),
        bm.RawRun(
            row_key="mpo",
            method="mpo",
            tdvp_sweeps=1,
            runtime_s=0.01,
            fidelity=0.995,
            infidelity=0.005,
            mean_abs_pauli_error=0.015,
            **{**base, "obs_score": 0.015},
        ),
        bm.RawRun(
            row_key="tdvp1",
            method="dynamic_tdvp",
            tdvp_sweeps=1,
            runtime_s=0.02,
            fidelity=0.98,
            infidelity=0.02,
            mean_abs_pauli_error=0.04,
            **{**base, "obs_score": 0.04},
        ),
        bm.RawRun(
            row_key="tdvp64",
            method="dynamic_tdvp",
            tdvp_sweeps=64,
            runtime_s=0.2,
            fidelity=0.999,
            infidelity=0.001,
            mean_abs_pauli_error=0.002,
            **{**base, "obs_score": 0.002},
        ),
    ]
    matched = bm.analyze_same_chi(rows)
    assert len(matched) == 1
    assert matched[0].best_tdvp_sweeps == 64
    assert matched[0].best_tdvp_obs_score == pytest.approx(0.002)


def test_write_csv_unions_mixed_advantage_rows(bm: ModuleType, tmp_path: Path) -> None:
    """Advantage rows from different win kinds share one CSV schema."""
    rows = [
        {"kind": "fixed_chi_obs_score", "case_key": "a", "chi": 8, "best_tdvp_sweeps": 64},
        {
            "kind": "chi_to_obs_target",
            "L": 8,
            "target_type": "obs",
            "target": 1e-3,
            "chi_tdvp": 4,
            "tdvp_chi_win": True,
        },
    ]
    out = tmp_path / "advantage_cases.csv"
    bm.write_csv(out, None, rows)
    text = out.read_text(encoding="utf-8")
    assert "target_type" in text.splitlines()[0]
    assert "case_key" in text.splitlines()[0]
    assert len(text.strip().splitlines()) == 3


def test_run_single_smoke(bm: ModuleType) -> None:
    """One end-to-end method run completes without crashing."""
    spec = bm.build_circuit("single_rzz_nn", 8, theta=0.3)
    prep = bm.prep_initial_state("plus", 8, chi=8, seed=0)
    ref = bm.exact_reference(prep, spec.qc)
    row = bm.run_single(
        prep,
        spec,
        "swaps",
        chi=8,
        tdvp_sweeps=1,
        seed=0,
        initial_state="plus",
        theta=0.3,
        ref_vec=ref,
        reference_type="exact",
        initial_state_cap_respected=True,
    )
    assert not row.crashed
    assert row.cap_respected
    assert row.obs_score >= 0.0
    assert 0.0 <= row.fidelity <= 1.0
