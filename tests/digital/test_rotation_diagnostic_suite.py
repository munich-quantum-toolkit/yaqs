# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Targeted assertions for the rotation diagnostic benchmark probes."""

from __future__ import annotations

from benchmarks.hybrid_benchmark_lib import (
    PASS_TOL,
    build_rotation_diagnostic_probes,
    run_direct_hybrid,
    run_simulator_hybrid,
)


def test_tebd_matches_qiskit_for_all_rotation_diagnostic_probes() -> None:
    probes = build_rotation_diagnostic_probes()
    for probe_id, qc in probes.items():
        vec, _, _, _ = run_simulator_hybrid(qc, gate_mode="tebd", tdvp_sweeps=4)
        # fidelity_error = 1 - overlap^2
        from benchmarks.hybrid_benchmark_lib import fidelity_vs_qiskit

        err = abs(1.0 - fidelity_vs_qiskit(qc, vec))
        assert err < PASS_TOL, f"TEBD mismatch vs Qiskit on {probe_id}: {err}"


def test_isolated_lr_rotations_pass_with_simulator_auto_padding() -> None:
    probes = build_rotation_diagnostic_probes()
    for probe_id in (
        "rzz_one_active_lr_6q",
        "rzz_both_active_lr_6q",
        "rxx_one_active_lr_6q",
        "ryy_one_active_lr_6q",
    ):
        qc = probes[probe_id]
        vec, _, _, _ = run_simulator_hybrid(qc, gate_mode="hybrid", tdvp_sweeps=4)
        from benchmarks.hybrid_benchmark_lib import fidelity_vs_qiskit

        err = abs(1.0 - fidelity_vs_qiskit(qc, vec))
        assert err < 1e-9, f"Hybrid auto-pad failed on {probe_id}: {err}"


def test_pair_creation_can_fail_at_chi1_but_should_pass_with_pad2_for_minimal_cases() -> None:
    probes = build_rotation_diagnostic_probes()
    for probe_id in ("rxx_vacuum_lr_6q", "ryy_vacuum_lr_6q"):
        qc = probes[probe_id]

        vec2, _, _ = run_direct_hybrid(qc, initial_pad=2, tdvp_sweeps=4)
        from benchmarks.hybrid_benchmark_lib import fidelity_vs_qiskit

        err2 = abs(1.0 - fidelity_vs_qiskit(qc, vec2))
        assert err2 < 1e-9, f"pad=2 did not fix {probe_id}: {err2}"


def test_lr_stack_mixed_is_known_limitation_for_now() -> None:
    probes = build_rotation_diagnostic_probes()
    qc = probes["lr_stack_mixed_12q"]
    vec, _, _, _ = run_simulator_hybrid(qc, gate_mode="hybrid", tdvp_sweeps=4)
    from benchmarks.hybrid_benchmark_lib import fidelity_vs_qiskit

    err = abs(1.0 - fidelity_vs_qiskit(qc, vec))
    # Expect not to be machine precision (regression case).
    assert err > 1e-6


def test_minimal_mixed_axes_disjoint_pairs_is_a_known_regression_case() -> None:
    probes = build_rotation_diagnostic_probes()
    qc = probes["mixed_axes_disjoint_pairs_8q"]
    vec, _, _, _ = run_simulator_hybrid(qc, gate_mode="hybrid", tdvp_sweeps=4)
    from benchmarks.hybrid_benchmark_lib import fidelity_vs_qiskit

    err = abs(1.0 - fidelity_vs_qiskit(qc, vec))
    assert err > 1e-6

