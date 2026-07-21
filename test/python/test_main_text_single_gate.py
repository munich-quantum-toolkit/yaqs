# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Tests for the main-text single RZZ gate benchmark."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SINGLE_GATE_DIR = Path(__file__).resolve().parents[2] / "experiments" / "single_gate"
sys.path.insert(0, str(SINGLE_GATE_DIR))

from config import (  # noqa: E402
    CHI0,
    FIGURES_DIR,
    FIGURE_STEM,
    OUTPUT_DIR,
    build_generic_angle_grid,
    build_special_angles,
    pick_intermediate_chi,
)
from gate_convention_checks import check_gate_convention  # noqa: E402
from gate_runtime import normalized_state_fidelity  # noqa: E402
from plot import ANGLE_XLABEL, INFIDELITY_YLABEL, plot_figure  # noqa: E402


def _synthetic_angle_row(
    *,
    method: str,
    chi_max: int,
    x_fraction: float,
    infidelity: float,
    special_angle: bool = False,
) -> dict:
    return {
        "method": method,
        "chi_max": chi_max,
        "x_fraction": x_fraction,
        "theta": 2.0 * math.pi * x_fraction,
        "infidelity": infidelity,
        "special_angle": special_angle,
        "substeps": 64,
    }


def test_chi0_is_eight() -> None:
    assert CHI0 == 8


def test_intermediate_chi_rule() -> None:
    assert pick_intermediate_chi(8, 16) == 12
    assert pick_intermediate_chi(8, 64) == 24


def test_generic_angle_grid_excludes_special() -> None:
    x_gen, _ = build_generic_angle_grid()
    x_spec, _ = build_special_angles()
    for x in x_gen:
        assert not any(abs(float(x) - s) < 1e-12 for s in x_spec)
    assert len(x_gen) >= 22


def test_rzz_gate_convention() -> None:
    assert check_gate_convention() == []


def test_main_text_outputs_exist() -> None:
    out = OUTPUT_DIR
    if not (out / "results.sqlite").exists():
        pytest.skip("main-text benchmark not run")
    for name in (
        "single_gate_angle_sweep.csv",
        "single_gate_substeps.csv",
        "single_gate_chi_scan.csv",
        "single_gate_mpo_diagnostics.csv",
        "single_gate_validation.md",
        "theta_zero_diagnostics.csv",
        "theta_zero_diagnostics.md",
    ):
        assert (out / name).is_file(), name
    for name in (f"{FIGURE_STEM}.pdf", f"{FIGURE_STEM}.png"):
        assert (FIGURES_DIR / name).is_file(), name


def test_plot_figure_axis_labels() -> None:
    """Panels (a)-(c) share theta/(2pi); (d) uses TDVP substeps; ylabel is set once."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    methods = ("hybrid_tdvp", "tebd_swap", "mpo_zipup", "variational_mpo")
    angle_rows = [
        _synthetic_angle_row(method=m, chi_max=chi, x_fraction=x, infidelity=1e-6 * (i + 1))
        for i, (m, chi, x) in enumerate(
            (m, chi, x) for chi in (8, 12, 16) for m in methods for x in (1e-4, 1e-2, 1.0)
        )
    ]
    substep_rows = [
        {
            "method": "hybrid_tdvp",
            "chi_max": chi,
            "x_fraction": 0.1,
            "theta": 0.2 * math.pi,
            "infidelity": 1e-4 / n,
            "special_angle": False,
            "substeps": n,
        }
        for chi in (8, 12, 16)
        for n in (1, 4, 16, 64)
    ]
    fig = plot_figure(angle_rows, substep_rows, chi_low=8, chi_mid=12, chi_full=16)
    axes = fig.axes
    assert len(axes) >= 4
    for ax in axes[:3]:
        assert ax.get_xlabel() == ANGLE_XLABEL
    assert axes[0].get_ylabel() == INFIDELITY_YLABEL
    assert axes[3].get_xlabel() == "TDVP substeps"
    matplotlib.pyplot.close(fig)


def test_normalized_fidelity_scaled_copy_is_one() -> None:
    rng = np.random.default_rng(0)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    metrics = normalized_state_fidelity(psi, 0.37 * psi)
    assert metrics["fidelity_normalized"] == pytest.approx(1.0, abs=1e-14)
    assert metrics["infidelity_normalized"] == pytest.approx(0.0, abs=1e-14)


def test_normalized_fidelity_global_rescaling_invariant() -> None:
    rng = np.random.default_rng(1)
    exact = rng.normal(size=32) + 1j * rng.normal(size=32)
    approx = rng.normal(size=32) + 1j * rng.normal(size=32)
    base = normalized_state_fidelity(exact, approx)["fidelity_normalized"]
    for c_e, c_a in ((2.0, 0.5), (1j, -3.0), (0.1 - 0.2j, 7.0 + 1j)):
        scaled = normalized_state_fidelity(c_e * exact, c_a * approx)["fidelity_normalized"]
        assert scaled == pytest.approx(base, abs=1e-12)


def test_normalized_fidelity_orthogonal_is_zero() -> None:
    a = np.array([1.0, 0.0], dtype=np.complex128)
    b = np.array([0.0, 1.0], dtype=np.complex128)
    metrics = normalized_state_fidelity(a, b)
    assert metrics["fidelity_normalized"] == pytest.approx(0.0, abs=1e-15)


def test_normalized_fidelity_range_and_clip() -> None:
    rng = np.random.default_rng(2)
    for _ in range(20):
        e = rng.normal(size=8) + 1j * rng.normal(size=8)
        a = rng.normal(size=8) + 1j * rng.normal(size=8)
        f = normalized_state_fidelity(e, a)["fidelity_normalized"]
        assert 0.0 <= f <= 1.0


def test_normalized_fidelity_raises_far_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="nonzero norms"):
        normalized_state_fidelity(np.zeros(2), np.array([1.0, 0.0], dtype=np.complex128))


def test_normalized_fidelity_clip_small_excursion(monkeypatch: pytest.MonkeyPatch) -> None:
    e = np.array([1.0, 0.0], dtype=np.complex128)
    a = np.array([0.0, 1.0], dtype=np.complex128)
    real_vdot = np.vdot
    calls = {"n": 0}

    def fake_vdot_clip(x, y):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            return complex(math.sqrt(1.0 + 5e-13))
        return real_vdot(x, y)

    monkeypatch.setattr(np, "vdot", fake_vdot_clip)
    metrics = normalized_state_fidelity(e, a, clip_tol=1e-12)
    assert metrics["fidelity_normalized"] == 1.0


def test_normalized_fidelity_raises_large_excursion(monkeypatch: pytest.MonkeyPatch) -> None:
    e = np.array([1.0, 0.0], dtype=np.complex128)
    a = np.array([0.0, 1.0], dtype=np.complex128)
    real_vdot = np.vdot
    calls = {"n": 0}

    def fake_vdot(x, y):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            return complex(math.sqrt(1.0 + 1e-6))
        return real_vdot(x, y)

    monkeypatch.setattr(np, "vdot", fake_vdot)
    with pytest.raises(ValueError, match="outside"):
        normalized_state_fidelity(e, a, clip_tol=1e-12)


def test_chi16_non_tdvp_at_numerical_precision() -> None:
    csv_path = OUTPUT_DIR / "single_gate_angle_sweep.csv"
    if not csv_path.is_file():
        pytest.skip("angle-sweep CSV not present")
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    non_tdvp = [
        r
        for r in rows
        if int(float(r["chi_max"])) == 16 and r["method"] != "hybrid_tdvp"
    ]
    assert non_tdvp
    for r in non_tdvp:
        assert float(r["infidelity"]) <= 1e-12, (
            f"{r['method']} x={r['x_fraction']}: I={r['infidelity']}"
        )
        if "fidelity_definition" in r and r["fidelity_definition"]:
            assert r["fidelity_definition"] == "normalized_state_fidelity_v2"
