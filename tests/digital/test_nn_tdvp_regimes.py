# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Smoke tests for NN all-TDVP benchmark."""

from __future__ import annotations

from scripts.benchmark_long_range_tdvp_regimes import make_circuit_family
from scripts.benchmark_nn_tdvp_regimes import (
    NN_STAGE_CONFIGS,
    build_nn_specs,
    compute_reference,
    run_nn_method,
)


def test_build_nn_specs_only_brickwork() -> None:
    specs = build_nn_specs(NN_STAGE_CONFIGS[0])
    assert len(specs) > 0
    assert all(s.family == "nn_brickwork" for s in specs)
    assert all(len(s.lr_pairs) == 0 for s in specs)


def test_tebd_vs_tdvp_all_nn_smoke() -> None:
    spec = make_circuit_family("nn_brickwork", n=8, depth=2, seed=0, angle_regime="small")
    ref, ref_type, _, _ = compute_reference(spec, exact_n_max=16)
    assert ref is not None

    tebd = run_nn_method(
        spec,
        method="tebd_nn",
        chi=16,
        tdvp_sweeps=1,
        sweep_scan_enabled=False,
        ref=ref,
        ref_type=ref_type,
        ref_chi=None,
        ref_sweeps=None,
    )
    tdvp = run_nn_method(
        spec,
        method="tdvp_all",
        chi=16,
        tdvp_sweeps=1,
        sweep_scan_enabled=False,
        ref=ref,
        ref_type=ref_type,
        ref_chi=None,
        ref_sweeps=None,
    )
    assert tebd.status == "ok"
    assert tdvp.status == "ok"
    assert tebd.lr_gate_count == 0
    assert tdvp.lr_gate_count == 0
    assert tebd.tebd_direct_gate_count == tebd.nn_gate_count
    assert tdvp.tdvp_lr_gate_count == tdvp.nn_gate_count
    assert tdvp.tebd_direct_gate_count == 0
    assert tdvp.enriched_lr_count == 0
    assert tebd.mean_abs_observable_error is not None
