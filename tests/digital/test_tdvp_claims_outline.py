# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Smoke tests for TDVP claims outline benchmark."""

from __future__ import annotations

from scripts.benchmark_long_range_tdvp_regimes import compute_reference, run_method
from scripts.benchmark_tdvp_claims_outline import (
    STAGE_CONFIGS,
    build_claims_specs,
    make_claims_spec,
    make_heisenberg_floquet_spec,
    make_sparse_long_range_claims,
    validate_claims_benchmark,
)


def test_build_claims_specs_stage0() -> None:
    specs = build_claims_specs(STAGE_CONFIGS[0])
    assert len(specs) > 0
    families = {s.family for s in specs}
    assert "nn_brickwork" in families
    assert "periodic_1d" in families
    assert "sparse_long_range" in families
    sparse = [s for s in specs if s.family == "sparse_long_range"]
    assert all(len(s.lr_pairs) > 0 for s in sparse)


def test_sparse_long_range_has_minimum_separation() -> None:
    spec = make_sparse_long_range_claims(n=12, depth=2, seed=0, angle_regime="small")
    min_range = max(2, 12 // 3)
    for i, j in spec.lr_pairs:
        assert j - i >= min_range


def test_floquet_spec_snake_mapping() -> None:
    spec = make_heisenberg_floquet_spec(lx=3, ly=4, cycles=1, seed=0, J_over_pi=0.01)
    assert spec.n_qubits == 12
    assert spec.initial_state == "neel"
    assert len(spec.lr_pairs) > 0
    assert len(spec.floquet_patches) >= 2


def test_validate_claims_benchmark_smoke() -> None:
    validate_claims_benchmark(chi=16)


def test_hybrid_vs_tebd_sparse_smoke() -> None:
    spec = make_claims_spec("sparse_long_range", n=8, depth=2, seed=0, angle_regime="small")
    cs = spec.to_circuit_spec()
    ref, ref_type, _, _ = compute_reference(cs, exact_n_max=16)
    assert ref is not None

    tebd = run_method(
        cs,
        method="tebd_swap",
        chi=16,
        tdvp_sweeps=1,
        ref=ref,
        ref_type=ref_type,
        ref_chi=None,
        ref_sweeps=None,
        bond_history=[],
    )
    hybrid = run_method(
        cs,
        method="hybrid_tdvp_padded4",
        chi=16,
        tdvp_sweeps=1,
        ref=ref,
        ref_type=ref_type,
        ref_chi=None,
        ref_sweeps=None,
        bond_history=[],
    )
    assert tebd.status == "ok"
    assert hybrid.status == "ok"
    assert tebd.lr_gate_count > 0
    assert hybrid.tdvp_lr_gate_count > 0
