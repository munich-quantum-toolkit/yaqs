# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Automated checks for the paper_benchmarks pipeline outputs.

Fast tests only: they validate the stored validation JSONs, processed CSV
invariants, and re-execute the cheap dense/locality validators in-process.

Run with:
    uv run pytest paper_benchmarks/tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PB_DIR / "scripts"))


def _load(rel: str) -> dict:
    return json.loads((PB_DIR / rel).read_text(encoding="utf-8"))


def test_locality_validation_passes() -> None:
    """Fixed-rank gate-window locality residual below 1e-10 (spec 3.2)."""
    payload = _load("logs/validation_locality.json")
    assert payload["all_pass"]
    assert payload["r_loc_max"] < 1e-10


def test_locality_recomputed_single_case() -> None:
    """Re-derive one locality case from scratch (independent of stored JSON)."""
    import numpy as np
    from validate_locality import (
        CHI,
        L,
        apply_generator,
        random_dense_mps_state,
        schmidt_subspaces,
        tangent_projection,
    )

    rng = np.random.default_rng(2026)
    psi = random_dense_mps_state(L, CHI, rng)
    lefts, rights, _, _ = schmidt_subspaces(psi, L, CHI)
    q0, q1 = 2, 7
    w = apply_generator(psi, "z", 0.7, q0, q1, L)
    full = tangent_projection(w, lefts, rights, L, sites=range(L), cuts=range(1, L))
    win = tangent_projection(
        w, lefts, rights, L, sites=range(q0, q1 + 1), cuts=range(q0 + 1, q1 + 1)
    )
    assert np.linalg.norm(full - win) / np.linalg.norm(full) < 1e-12


def test_dense_validation_passes() -> None:
    """Dense-reference/ordering checks and comparator exact limits (spec 3.1)."""
    payload = _load("logs/validation_dense.json")
    assert payload["all_pass"]
    exact_limits = {
        c["check"]: c["value"] for c in payload["checks"]
        if c["check"].startswith("exact_limit_")
    }
    assert len(exact_limits) == 9  # 3 methods x 3 gates
    for name, value in exact_limits.items():
        tol = 1e-7 if "tebd_swap" in name else 1e-12
        assert value < tol, name


def test_gate_matrix_convention() -> None:
    """YAQS gate matrices equal exp(-i theta PP/2) built independently."""
    import numpy as np
    from validate_dense import independent_gate_matrix

    sys.path.insert(0, str(PB_DIR.parent / "experiments" / "single_gate"))
    from gate_runtime import make_gate

    for gate_type in ("rxx", "ryy", "rzz"):
        for theta in (0.0, 0.9, 4.2):
            expected = independent_gate_matrix(gate_type, theta)
            got = np.asarray(make_gate(gate_type, theta, 0, 1).matrix)
            assert np.max(np.abs(got - expected)) < 1e-12


def test_processed_angle_sweep_invariants() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(PB_DIR / "processed" / "single_gate_angle_sweep.csv")
    # complete grid: 3 gates x 3 seeds x 3 chi x 27 angles x 6 methods
    assert len(df) == 3 * 3 * 3 * 27 * 6
    keys = ["gate_type", "seed", "method", "chi_max", "x_fraction", "substeps"]
    assert not df.duplicated(subset=keys).any()
    assert set(df.gate_type) == {"rxx", "ryy", "rzz"}
    assert set(df.seed) == {11, 22, 33}
    # direct MPO application is essentially exact at chi=16 (uncapped rank 16)
    full = df[(df.chi_max == 16) & (df.method == "mpo_zipup")]
    assert full.infidelity.max() < 1e-10


def test_processed_substep_study_nonbinding() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(PB_DIR / "processed" / "single_gate_substeps_x025.csv")
    assert (df.max_bond < df.chi_max).all()
    assert df.discarded_weight.max() < 1e-12
    assert df.infidelity.max() < 1e-9  # numerically exact at nonbinding cap


def test_circuit_trajectories_complete() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(PB_DIR / "processed" / "circuit_trajectories.csv")
    cases = [
        (model, method)
        for model in ("ising", "heisenberg")
        for method in ("hybrid_tdvp", "full_tdvp", "tebd_swap", "mpo_zipup")
    ] + [
        (model, method)
        for model in ("ising_1d", "heisenberg_1d")
        for method in ("full_tdvp", "tebd_swap", "mpo_zipup")
    ]
    for model, method in cases:
        d = df[(df.model == model) & (df.method == method)]
        assert d.trotter_step.max() == 30, (model, method)
        assert len(d) == 31, (model, method)


def test_circuit_2d_full_tdvp_distinct_from_hybrid() -> None:
    """2D full_tdvp must differ from hybrid (NN gates take different paths)."""
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(PB_DIR / "processed" / "circuit_trajectories.csv")
    for model in ("ising", "heisenberg"):
        a = df[(df.model == model) & (df.method == "hybrid_tdvp")].set_index(
            "trotter_step").infidelity
        b = df[(df.model == model) & (df.method == "full_tdvp")].set_index(
            "trotter_step").infidelity
        assert float((a - b).abs().max()) > 0.0, model


def test_circuit_1d_full_tdvp_uses_tdvp_for_nn_gates() -> None:
    """full_tdvp maps to gate_mode="full-tdvp" and differs from TEBD in 1D.

    In the 1D chains every two-qubit gate is nearest-neighbour; identical
    full_tdvp and TEBD trajectories would mean the TDVP window path was
    silently bypassed.
    """
    pd = pytest.importorskip("pandas")
    sys.path.insert(0, str(PB_DIR.parent / "experiments" / "fixed_resources"))
    from trajectory import _gate_params

    params = _gate_params(32, "full_tdvp", tdvp_substeps=2)
    assert params.gate_mode == "full-tdvp"

    df = pd.read_csv(PB_DIR / "processed" / "circuit_trajectories.csv")
    for model in ("ising_1d", "heisenberg_1d"):
        a = df[(df.model == model) & (df.method == "full_tdvp")].set_index(
            "trotter_step").infidelity
        b = df[(df.model == model) & (df.method == "tebd_swap")].set_index(
            "trotter_step").infidelity
        assert float((a - b).abs().max()) > 0.0, model


def test_final_report_all_pass() -> None:
    payload = _load("validation_report.json")
    assert payload["all_pass"]
    for method, entry in payload["heisenberg_deterministic_repeat"].items():
        assert entry["reproducible"], method


def test_manifest_covers_raw_inputs() -> None:
    payload = _load("data_manifest.json")
    groups = {f["group"] for f in payload["files"]}
    assert {"single_gate_corrected", "circuits_corrected", "raw_new"} <= groups
    assert all(len(f["sha256"]) == 64 for f in payload["files"])
