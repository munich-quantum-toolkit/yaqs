# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for TDVP branch scaling benchmark helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
_MODULE_PATH = _EXPERIMENTS / "benchmark_tdvp_branch_scaling.py"


@pytest.fixture(scope="module")
def bm() -> ModuleType:
    """Load the branch scaling benchmark module once per worker.

    Returns:
        Imported benchmark module.
    """
    if str(_EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(_EXPERIMENTS))
    spec = importlib.util.spec_from_file_location("benchmark_tdvp_branch_scaling", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_lr_circuit_qubit_count(bm: ModuleType) -> None:
    """Long-range benchmark circuit uses the requested number of qubits."""
    qc = bm.build_lr_circuit(10)
    assert qc.num_qubits == 10


def test_process_budget_single_trajectory(bm: ModuleType) -> None:
    """Noiseless runs should allow only the worker process tree."""
    assert bm.process_budget(15, parallel_eligible=False) == 1 + bm.PROCESS_SLACK


def test_process_budget_parallel_pool(bm: ModuleType) -> None:
    """Parallel runs budget pool workers plus runner overhead."""
    assert bm.process_budget(7, parallel_eligible=True) == 7 + 1 + bm.PROCESS_SLACK


def test_filter_kwargs_drops_none(bm: ModuleType) -> None:
    """Optional parameters omitted on main are not forwarded as ``None``."""
    filtered = bm._filter_kwargs({"tdvp_sweeps", "gate_mode"}, {"gate_mode": "tdvp", "tdvp_sweeps": None})
    assert filtered == {"gate_mode": "tdvp"}


def test_compare_rows_flags_control_regression(bm: ModuleType) -> None:
    """Control cases fail when branch is much slower than main."""
    branch_row = bm.RunResult(
        branch="tdvp-sweeps",
        case="swaps_lr_noiseless",
        length=8,
        runtime_s=2.0,
        peak_processes=2,
        simulator_max_workers=7,
        available_cpus=8,
        parallel_eligible=False,
        effective_num_traj=1,
        process_budget=5,
        process_ok=True,
    )
    main_row = bm.RunResult(
        branch="main",
        case="swaps_lr_noiseless",
        length=8,
        runtime_s=1.0,
        peak_processes=1,
        simulator_max_workers=7,
        available_cpus=8,
        parallel_eligible=False,
        effective_num_traj=1,
        process_budget=5,
        process_ok=True,
    )
    summary = bm.compare_rows([branch_row], [main_row])
    assert summary[0]["slowdown_ratio"] == pytest.approx(2.0)
    assert summary[0]["pass"] is False


def test_build_sweep_comparison_joins_main_baseline(bm: ModuleType) -> None:
    """Sweep comparison rows include runtime ratio against main defaults."""
    main_row = bm.RunResult(
        branch="main",
        case="tdvp_lr_noiseless",
        length=8,
        tdvp_sweeps=None,
        runtime_s=0.004,
        peak_processes=1,
        simulator_max_workers=7,
        available_cpus=8,
        parallel_eligible=False,
        effective_num_traj=1,
        process_budget=5,
        process_ok=True,
        fidelity=0.978,
        infidelity=0.022,
        z_error=0.01,
    )
    branch_row = bm.RunResult(
        branch="tdvp-sweeps",
        case="tdvp_lr_noiseless",
        length=8,
        tdvp_sweeps=4,
        runtime_s=0.020,
        peak_processes=2,
        simulator_max_workers=7,
        available_cpus=8,
        parallel_eligible=False,
        effective_num_traj=1,
        process_budget=5,
        process_ok=True,
        fidelity=0.993,
        infidelity=0.007,
        z_error=0.002,
    )
    summary = bm.build_sweep_comparison([branch_row], [main_row])
    assert summary[0]["runtime_ratio_vs_main"] == pytest.approx(5.0)
    assert summary[0]["branch_fidelity"] == pytest.approx(0.993)


def test_compare_rows_accepts_healthy_tdvp_case(bm: ModuleType) -> None:
    """TDVP cases pass when slowdown and process budgets are within limits."""
    branch_row = bm.RunResult(
        branch="tdvp-sweeps",
        case="tdvp_lr_noiseless",
        length=8,
        runtime_s=1.1,
        peak_processes=3,
        simulator_max_workers=7,
        available_cpus=8,
        parallel_eligible=False,
        effective_num_traj=1,
        process_budget=5,
        process_ok=True,
    )
    main_row = bm.RunResult(
        branch="main",
        case="tdvp_lr_noiseless",
        length=8,
        runtime_s=1.0,
        peak_processes=1,
        simulator_max_workers=7,
        available_cpus=8,
        parallel_eligible=False,
        effective_num_traj=1,
        process_budget=5,
        process_ok=True,
    )
    summary = bm.compare_rows([branch_row], [main_row])
    assert summary[0]["pass"] is True
