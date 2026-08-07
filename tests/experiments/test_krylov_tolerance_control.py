# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Pure validation tests for the isolated TDVP Krylov control."""

from __future__ import annotations

import pytest

from experiments.circuit_benchmarks.extensions import krylov_tolerance_control as control


def _accuracy_row(*, tolerance: float = 1e-10, cap: int = 28) -> dict[str, object]:
    return {
        "campaign_id": control.CAMPAIGN_ID,
        "case": control.CASE_KEY,
        "method": control.METHOD,
        "chi_max": cap,
        "n_sub": control.N_SUB,
        "target_step": control.TARGET_STEP,
        "krylov_tolerance": tolerance,
        "svd_threshold": 1e-13,
        "max_infidelity_through": 0.009,
        "endpoint_infidelity": 0.008,
        "infidelity_by_step_json": "[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.009,0.008]",
        "parameter_count_by_step_json": "[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]",
        "peak_parameter_count": 10_000,
        "peak_bond_dim": cap,
    }


def _timing_rows(*, tolerance: float = 1e-10, cap: int = 28) -> list[dict[str, object]]:
    return [
        {
            "campaign_id": control.CAMPAIGN_ID,
            "case": control.CASE_KEY,
            "method": control.METHOD,
            "chi_max": cap,
            "n_sub": control.N_SUB,
            "target_step": control.TARGET_STEP,
            "krylov_tolerance": tolerance,
            "repeat": repeat,
            "runtime_s": runtime,
            "endpoint_infidelity": 0.008,
        }
        for repeat, runtime in enumerate((3.0, 1.0, 2.0))
    ]


def test_grid_is_deduplicated_and_tdvp_protocol_is_fixed() -> None:
    """Repeated CLI values should form one sorted TDVP-only Cartesian grid."""
    points = control.normalize_grid([1e-8, 1e-10, 1e-8], [28, 24, 28], 3)

    assert points == (
        control.ControlPoint(1e-10, 24),
        control.ControlPoint(1e-10, 28),
        control.ControlPoint(1e-8, 24),
        control.ControlPoint(1e-8, 28),
    )
    assert control.CASE_KEY == "ising_2d"
    assert control.METHOD == "gate_local_2tdvp"
    assert control.TARGET_STEP == 15
    assert control.N_SUB == 2


@pytest.mark.parametrize(
    ("tolerances", "caps", "repeats"),
    [([], [28], 3), ([1e-10], [], 3), ([0.0], [28], 3), ([1e-10], [0], 3), ([1e-10], [28], 0)],
)
def test_grid_rejects_invalid_values(
    tolerances: list[float],
    caps: list[int],
    repeats: int,
) -> None:
    """Invalid scientific or timing grids should fail before any simulation."""
    with pytest.raises(ValueError, match=r"required|positive|at least"):
        control.normalize_grid(tolerances, caps, repeats)


def test_summary_combines_accuracy_with_three_timing_repeats() -> None:
    """A complete point should retain resources and summarize timing robustly."""
    summaries = control.summarize_complete_points(
        [_accuracy_row()],
        _timing_rows(),
        timing_repeats=3,
    )

    assert len(summaries) == 1
    assert summaries[0]["method"] == "gate_local_2tdvp"
    assert summaries[0]["peak_parameter_count"] == 10_000
    assert summaries[0]["median_runtime_s"] == pytest.approx(2.0)
    assert summaries[0]["min_runtime_s"] == pytest.approx(1.0)
    assert summaries[0]["max_runtime_s"] == pytest.approx(3.0)


def test_summary_omits_an_interrupted_timing_group() -> None:
    """Persisted partial repeats should remain resumable without being summarized."""
    summaries = control.summarize_complete_points(
        [_accuracy_row()],
        _timing_rows()[:2],
        timing_repeats=3,
    )

    assert summaries == []


def test_summary_rejects_non_tdvp_or_duplicate_rows() -> None:
    """The isolated control must reject baseline data and duplicate repeats."""
    wrong_method = _accuracy_row()
    wrong_method["method"] = "tebd_swap"
    with pytest.raises(ValueError, match="TDVP-only protocol"):
        control.summarize_complete_points([wrong_method], [], timing_repeats=3)

    timings = _timing_rows()
    timings.append(dict(timings[0]))
    with pytest.raises(RuntimeError, match="Duplicate timing repeat"):
        control.summarize_complete_points([_accuracy_row()], timings, timing_repeats=3)


def test_resume_requires_identical_provenance_and_repeat_count() -> None:
    """Rows may only be resumed under the exact same fixed protocol."""
    manifest = {
        "schema_version": control.SCHEMA_VERSION,
        "campaign_id": control.CAMPAIGN_ID,
        "protocol_sha256": "abc",
        "timing_repeats": 3,
        "requested_points": [{"krylov_tolerance": 1e-10, "chi_max": 28}],
    }

    assert control.validate_resume_manifest(
        manifest,
        protocol_sha256="abc",
        timing_repeats=3,
    ) == (control.ControlPoint(1e-10, 28),)
    with pytest.raises(RuntimeError, match="protocol_sha256"):
        control.validate_resume_manifest(
            manifest,
            protocol_sha256="changed",
            timing_repeats=3,
        )
    with pytest.raises(RuntimeError, match="timing_repeats"):
        control.validate_resume_manifest(
            manifest,
            protocol_sha256="abc",
            timing_repeats=2,
        )
