# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused tests for the individual-gates campaign helpers and validation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import (  # noqa: E402
    apply_cx_dense_qiskit,
    apply_gate_dense_yaqs,
    conventional_median,
    cx_matrix,
    git_revision_for_hash,
    independent_r_pp_matrix,
    make_cx_gate,
    make_pauli_gate,
    normalized_state_fidelity,
    prepare_initial_state,
    state_distance,
    task_id_from_payload,
    two_site_h_cx,
)
from config import (  # noqa: E402
    EXPECTED_CAMPAIGN_ROWS,
    EXPECTED_CNOT_RANK_ROWS,
    EXPECTED_CNOT_ROWS,
    EXPECTED_PAULI_ROWS,
    Q0,
    Q1,
    X_VALUES,
    N,
    theta_from_x,
)
from run import iter_campaign_specs, iter_cnot_rank_specs  # noqa: E402
from validate import run_validation  # noqa: E402


def test_expected_row_counts() -> None:
    assert EXPECTED_PAULI_ROWS == 432
    assert EXPECTED_CNOT_ROWS == 36
    assert EXPECTED_CAMPAIGN_ROWS == 468
    assert EXPECTED_CNOT_RANK_ROWS == 90
    assert len(iter_campaign_specs()) == 468
    assert len(iter_cnot_rank_specs()) == 90


def test_cnot_rank_spec_composition() -> None:
    specs = iter_cnot_rank_specs()
    direct = [s for s in specs if s["method"] in {"mpo_zipup", "tebd_swap"}]
    tdvp = [s for s in specs if s["method"] == "gate_local_2tdvp"]
    assert len(direct) == 30
    assert len(tdvp) == 60
    assert {s["n_sub"] for s in tdvp} == {1, 16, 128, 256}
    assert all(s["svd_threshold"] == 1e-300 for s in specs)


def test_theta_from_x_no_wrapping() -> None:
    assert theta_from_x(0.0) == 0.0
    assert abs(theta_from_x(0.25) - 0.5 * np.pi) < 1e-15
    assert abs(theta_from_x(1e-4) - 2 * np.pi * 1e-4) < 1e-18


def test_independent_r_pp_matches_yaqs() -> None:
    for gate in ("rxx", "ryy", "rzz"):
        for x in (0.0, 1e-3, 0.25):
            theta = theta_from_x(x)
            indep = independent_r_pp_matrix(gate, theta)
            yaqs = np.asarray(make_pauli_gate(gate, theta, 0, 1).matrix)
            assert np.max(np.abs(indep - yaqs)) < 1e-13


def test_cx_expm_identity() -> None:
    h = two_site_h_cx()
    assert np.max(np.abs(expm(-1j * h) - cx_matrix())) < 1e-13


def test_cx_qiskit_endpoint() -> None:
    init = prepare_initial_state(11)
    for control, target in ((Q0, Q1), (Q1, Q0)):
        gate = make_cx_gate(control, target)
        yaqs = apply_gate_dense_yaqs(init["vec"], N, control, target, gate)
        qiskit = apply_cx_dense_qiskit(init["vec"], control, target, N)
        assert np.linalg.norm(yaqs - qiskit) < 1e-12


def test_initial_states_exact_rank_profile() -> None:
    from config import BOND_PROFILE

    for seed in (11, 22, 33):
        init = prepare_initial_state(seed)
        assert init["bond_profile"] == list(BOND_PROFILE)
        assert abs(init["norm"] - 1.0) < 1e-12


def test_task_id_stable() -> None:
    payload = {"a": 1, "b": [1, 2], "c": {"x": 0.1}}
    assert task_id_from_payload(payload) == task_id_from_payload(payload)


def test_git_hash_excludes_diff_hash() -> None:
    g = git_revision_for_hash()
    assert set(g) == {"git_commit", "git_dirty"}


def test_conventional_median_even() -> None:
    # numpy.median averages the two middle values for even length.
    assert conventional_median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_phase_aligned_distance_invariant() -> None:
    v = prepare_initial_state(11)["vec"]
    w = v * np.exp(1j * 0.37)
    assert state_distance(v, w) == pytest.approx(0.0, abs=1e-12)


def test_fidelity_bounds() -> None:
    v = prepare_initial_state(11)["vec"]
    m = normalized_state_fidelity(v, v)
    assert m["fidelity_normalized"] == pytest.approx(1.0, abs=1e-14)
    assert 0.0 <= m["infidelity_normalized"] <= 1.0


def test_x_grid_contains_zero() -> None:
    assert 0.0 in X_VALUES


def test_validation_stage() -> None:
    # Skip checks that require cnot_rank_rows if not yet generated.
    from config import OUTPUT_DIR

    cnot_csv = OUTPUT_DIR / "cnot_rank_rows.csv"
    if not cnot_csv.is_file():
        pytest.skip("cnot_rank_rows.csv not yet generated")
    report = run_validation()
    assert report["pass"] is True
    assert report["n_failures"] == 0
    assert "residuals" in report
    assert "expm_minus_i_H_CX_vs_CX" in report["residuals"]


def test_main_figure_plot_smoke(tmp_path: Path) -> None:
    """Regenerate the six-panel main figure from existing CSVs."""
    from config import OUTPUT_DIR
    from plot import FIGURE_STEM
    from plot import main as plot_main

    if not (OUTPUT_DIR / "cnot_rank_rows.csv").is_file():
        pytest.skip("cnot_rank_rows.csv not yet generated")
    if not (OUTPUT_DIR / "refinement_rows.csv").is_file():
        pytest.skip("refinement_rows.csv not yet generated")

    fig_dir = tmp_path / "figures"
    rc = plot_main(["--output-dir", str(OUTPUT_DIR), "--figures-dir", str(fig_dir)])
    assert rc == 0
    assert (fig_dir / f"{FIGURE_STEM}.pdf").is_file()
    assert (fig_dir / f"{FIGURE_STEM}.png").is_file()
    assert (OUTPUT_DIR / f"{FIGURE_STEM}_caption.md").is_file()
    # Must not regenerate the obsolete supplementary figure.
    assert not (fig_dir / "figure_individual_gates_cnot_refinement_supp.pdf").is_file()
