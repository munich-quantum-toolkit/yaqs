# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Smoke tests for long-range TDVP regime benchmark."""

from __future__ import annotations

import json

from scripts.benchmark_long_range_tdvp_regimes import (
    STAGE_CONFIGS,
    build_all_specs,
    build_benchmark_runs,
    compute_reference,
    make_circuit_family,
    run_method,
    validate_tdvp_padding,
)


def test_make_periodic_circuit() -> None:
    spec = make_circuit_family("periodic_1d", n=8, depth=2, seed=0, angle_regime="small")
    assert spec.n_qubits == 8
    assert (0, 7) in spec.lr_pairs


def test_build_benchmark_runs_stage0_default() -> None:
    plans = build_benchmark_runs(0, STAGE_CONFIGS[0])
    methods = {p.method for p in plans}
    assert methods == {"tebd_swap", "hybrid_tdvp_unpadded", "hybrid_tdvp_padded4"}
    assert all(p.tdvp_sweeps == 1 for p in plans)


def test_run_hybrid_and_tebd_smoke() -> None:
    spec = make_circuit_family("sparse_long_range", n=8, depth=2, seed=0, angle_regime="small")
    ref, ref_type, _, _ = compute_reference(spec, exact_n_max=16)
    assert ref is not None
    assert ref_type == "exact_statevector"
    assert len(spec.lr_pairs) >= 1

    tdvp = run_method(
        spec,
        method="hybrid_tdvp_unpadded",
        chi=16,
        tdvp_sweeps=1,
        ref=ref,
        ref_type=ref_type,
        ref_chi=None,
        ref_sweeps=None,
        bond_history=[],
    )
    tebd = run_method(
        spec,
        method="tebd_swap",
        chi=16,
        tdvp_sweeps=1,
        ref=ref,
        ref_type=ref_type,
        ref_chi=None,
        ref_sweeps=None,
        bond_history=[],
    )
    assert tdvp.status == "ok"
    assert tebd.status == "ok"
    assert tdvp.lr_gate_count >= 1
    assert tdvp.tdvp_lr_count == tdvp.tdvp_lr_gate_count
    assert tdvp.tdvp_lr_count == tdvp.lr_gate_count
    assert tdvp.enriched_lr_count == 0
    assert tebd.tdvp_lr_count == 0
    assert tebd.tebd_swap_gate_count == tebd.lr_gate_count
    assert tdvp.mean_abs_observable_error is not None
    assert tdvp.mean_abs_error_z_single is not None


def test_hybrid_never_enriches_long_range_gates() -> None:
    """Standard TDVP benchmark path uses TDVP only (no Pauli enrichment)."""
    spec = make_circuit_family("sparse_long_range", n=8, depth=2, seed=0, angle_regime="small")
    ref, ref_type, _, _ = compute_reference(spec, exact_n_max=16)
    assert ref is not None
    run = run_method(
        spec,
        method="hybrid_tdvp_padded4",
        chi=32,
        tdvp_sweeps=1,
        ref=ref,
        ref_type=ref_type,
        ref_chi=None,
        ref_sweeps=None,
        bond_history=[],
    )
    assert run.status == "ok"
    assert run.enriched_lr_count == 0
    assert run.tdvp_lr_count == run.lr_gate_count
    assert run.tebd_swap_gate_count == 0
    assert run.padding_applied is True
    assert run.mean_abs_observable_error is not None


def test_periodic_1d_hybrid_dispatch_counts() -> None:
    spec = make_circuit_family("periodic_1d", n=8, depth=2, seed=0, angle_regime="small")
    ref, ref_type, _, _ = compute_reference(spec, exact_n_max=16)
    assert ref is not None
    run = run_method(
        spec,
        method="hybrid_tdvp_unpadded",
        chi=32,
        tdvp_sweeps=1,
        ref=ref,
        ref_type=ref_type,
        ref_chi=None,
        ref_sweeps=None,
        bond_history=[],
    )
    assert run.status == "ok"
    assert run.lr_gate_count == spec.depth
    assert run.tdvp_lr_count == run.lr_gate_count
    assert run.enriched_lr_count == 0
    assert run.tebd_direct_gate_count == run.nn_gate_count
    assert run.tebd_swap_gate_count == 0
    assert [0, 7] in json.loads(run.lr_gate_pairs)


def test_stage0_build_specs() -> None:
    specs = build_all_specs(STAGE_CONFIGS[0])
    assert len(specs) > 0
    families = {s.family for s in specs}
    assert "nn_brickwork" in families


def test_stage1_build_all_specs() -> None:
    specs = build_all_specs(STAGE_CONFIGS[1])
    assert len(specs) > 0
    for spec in specs:
        for inst in spec.qc.data:
            if inst.operation.name == "barrier":
                continue
            if len(inst.qubits) == 2:
                qa = spec.qc.find_bit(inst.qubits[0]).index
                qb = spec.qc.find_bit(inst.qubits[1]).index
                assert qa != qb, f"{spec.case_id} has duplicate qubits on a 2q gate"


def test_hybrid_padded4_increases_bonds_before_first_lr() -> None:
    spec = make_circuit_family("periodic_1d", n=8, depth=2, seed=0, angle_regime="small")
    ref, ref_type, _, _ = compute_reference(spec, exact_n_max=16)
    assert ref is not None

    unpadded = run_method(
        spec,
        method="hybrid_tdvp_unpadded",
        chi=16,
        tdvp_sweeps=1,
        ref=ref,
        ref_type=ref_type,
        ref_chi=None,
        ref_sweeps=None,
        bond_history=[],
    )
    padded = run_method(
        spec,
        method="hybrid_tdvp_padded4",
        chi=16,
        tdvp_sweeps=1,
        ref=ref,
        ref_type=ref_type,
        ref_chi=None,
        ref_sweeps=None,
        bond_history=[],
    )
    assert unpadded.status == "ok"
    assert padded.status == "ok"
    assert unpadded.padding_applied is False
    assert padded.padding_applied is True
    assert padded.padded_bonds_count >= 1
    assert padded.max_bond_dim_before_padding is not None
    assert padded.max_bond_dim_after_padding is not None
    assert padded.max_bond_dim_after_padding >= padded.max_bond_dim_before_padding
    assert unpadded.run_key() != padded.run_key()


def test_run_key_includes_method_and_padding() -> None:
    spec = make_circuit_family("sparse_long_range", n=8, depth=2, seed=0, angle_regime="small")
    ref, ref_type, _, _ = compute_reference(spec, exact_n_max=16)
    assert ref is not None
    run = run_method(
        spec,
        method="hybrid_tdvp_padded4",
        chi=16,
        tdvp_sweeps=1,
        ref=ref,
        ref_type=ref_type,
        ref_chi=None,
        ref_sweeps=None,
        bond_history=[],
    )
    assert "hybrid_tdvp_padded4" in run.run_key()
    assert "|padding4|" in run.run_key()
    assert "|noise0" in run.run_key()


def test_validate_tdvp_padding_smoke() -> None:
    validate_tdvp_padding(chi=32)
