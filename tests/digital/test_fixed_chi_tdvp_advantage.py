# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Smoke tests for fixed-χ TDVP advantage benchmark."""

from __future__ import annotations

from scripts.benchmark_fixed_chi_tdvp_advantage import (
    LR_PAULI_ROUTE,
    _active_config,
    _build_cases,
    _make_power_law_case,
    _run_circuit,
    _resolve_reference,
    STAGE_CONFIGS,
)


def test_lr_pauli_route_is_tdvp_only() -> None:
    assert LR_PAULI_ROUTE == "tdvp_only"


def test_local_generator_never_enriches() -> None:
    case = _make_power_law_case(
        stage=0,
        n=8,
        alpha=2.0,
        max_range=4,
        h=0.5,
        dt=0.0025,
        layers=2,
        initial_state="plus",
    )
    mps, _, _, _, routes, status = _run_circuit(case, method="local_generator_tdvp", max_bond_dim=16)
    assert status == "ok"
    assert mps is not None
    assert routes["enriched_lr"] == 0
    assert routes["tdvp_lr"] > 0


def test_stage0_build_and_reference() -> None:
    cases = _build_cases(0, _active_config(0))
    assert len(cases) > 0
    case = cases[0]
    ref, ref_method = _resolve_reference(case, stage_cfg=STAGE_CONFIGS[0])
    assert ref is not None
    assert ref_method == "qiskit_statevector"
