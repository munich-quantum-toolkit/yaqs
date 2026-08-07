# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused tests for variable-length accuracy and all-model bond profiles."""
# ruff: file-ignore[import-private-name] - this focused test intentionally exercises private helpers

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from matplotlib import pyplot as plt

from experiments.circuit_benchmarks.config import METHODS, N_STEPS, N
from experiments.circuit_benchmarks.figures.bond_profiles import (
    _cropped_profile,
    _step_edges,
)
from experiments.circuit_benchmarks.long_trajectories import variational_control
from experiments.circuit_benchmarks.long_trajectories.config import (
    SATURATION_WINDOW_STEPS,
)
from experiments.circuit_benchmarks.long_trajectories.plot import (
    TDVP_OVERRIDE_CAMPAIGN_ID,
    TDVP_OVERRIDE_METHOD,
    TDVP_OVERRIDE_TOLERANCE,
    VARIATIONAL_CAMPAIGN_ID,
    VARIATIONAL_CENSOR_RECORD_TYPE,
    VARIATIONAL_CENSOR_SCHEMA_VERSION,
    VARIATIONAL_RUNTIME_BUDGET_S,
    _marker_indices,
    _parameter_transient_stop,
    _plateau_window,
    _plot_variational_runtime_censor,
    _validate_case_rows,
    _validate_runtime_rows,
    _validate_variational_rows,
    _validate_variational_runtime_censor,
    apply_tdvp_row_override,
    caption,
    load_validated_tdvp_override_manifest,
)
from experiments.circuit_benchmarks.long_trajectories.run import (
    _criterion_met,
    _window_is_saturated,
)
from experiments.circuit_benchmarks.long_trajectories.timing import (
    _summary_rows,
)
from experiments.circuit_benchmarks.long_trajectories.variational_control import (
    _apply_variational_step,
    _stop_reason,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_saturation_window_must_be_unreliable_and_flat() -> None:
    """Reject accurate plateaus and substantial changes in unreliable errors."""
    assert not _window_is_saturated([1e-6] * SATURATION_WINDOW_STEPS)
    assert not _window_is_saturated(np.geomspace(0.1, 0.2, SATURATION_WINDOW_STEPS).tolist())
    assert _window_is_saturated([0.2] * SATURATION_WINDOW_STEPS)


def test_common_saturation_requires_every_method() -> None:
    """The case endpoint cannot be selected while one method is unsaturated."""
    saturated = dict.fromkeys(METHODS, True)
    saturated[METHODS[-1]] = False
    assert not _criterion_met(saturated)
    saturated[METHODS[-1]] = True
    assert _criterion_met(saturated)


def test_variable_rows_require_one_common_contiguous_endpoint() -> None:
    """Every plotted method must cover every step through the shared stop."""
    rows = [{"case": "ising_1d", "method": method, "step": str(step)} for method in METHODS for step in range(4)]
    assert _validate_case_rows(rows, "ising_1d") == 3
    rows.pop()
    with pytest.raises(RuntimeError, match="share one endpoint"):
        _validate_case_rows(rows, "ising_1d")


def test_markers_include_final_variable_length_sample() -> None:
    """Every curve should visibly end at its case-specific stopping time."""
    indices = _marker_indices(37, method_index=1)
    assert indices[-1] == 36
    assert np.all(np.diff(indices) > 0)


def test_plateau_inset_uses_exact_trailing_window() -> None:
    """The inset must contain exactly the samples defining the endpoint."""
    points = [{"step": str(step), "infidelity_normalized": str(0.2 + step / 1000)} for step in range(15)]
    steps, errors = _plateau_window(points, stop_step=14)
    assert np.array_equal(steps, np.arange(5, 15))
    assert errors.shape == (SATURATION_WINDOW_STEPS,)


def test_parameter_inset_covers_the_last_growth_step() -> None:
    """The early-step crop must include the slowest retained-size transient."""
    trajectories = {
        METHODS[0]: (32, 100, 100, 100, 100),
        METHODS[1]: (32, 50, 100, 100, 100),
        METHODS[2]: (32, 40, 70, 100, 100),
    }
    rows = [
        {
            "case": "ising_1d",
            "method": method,
            "step": str(step),
            "current_parameter_count": str(value),
        }
        for method, values in trajectories.items()
        for step, value in enumerate(values)
    ]
    assert _parameter_transient_stop(rows, "ising_1d", stop_step=4) == 3


def test_parameter_inset_keeps_a_confirmation_step_after_immediate_growth() -> None:
    """A one-step jump should still be visually distinguishable from a single point."""
    rows = [
        {
            "case": "heisenberg_2d",
            "method": method,
            "step": str(step),
            "current_parameter_count": str(value),
        }
        for method in METHODS
        for step, value in enumerate((32, 100, 100, 100))
    ]
    assert _parameter_transient_stop(rows, "heisenberg_2d", stop_step=3) == 2


@pytest.mark.parametrize(
    ("case_key", "last_step"),
    [
        ("ising_1d", 27),
        ("heisenberg_1d", 6),
        ("ising_2d", 6),
        ("heisenberg_2d", 1),
    ],
)
def test_profile_crops_use_declared_step_transients(
    case_key: str,
    last_step: int,
) -> None:
    """Each heatmap should include both endpoints of its declared transient."""
    matrix = np.ones((N_STEPS + 1, N - 1), dtype=int)
    cropped = _cropped_profile(matrix, case_key)
    assert cropped.shape == (last_step + 1, N - 1)
    edges = _step_edges(last_step)
    assert edges.shape == (last_step + 2,)
    assert edges[[0, -1]] == pytest.approx([-0.5, last_step + 0.5])


def test_timing_summary_uses_pointwise_median_and_range() -> None:
    """Runtime bands must summarize independent cumulative-time repeats."""
    rows = [
        {
            "case": case_key,
            "method": method,
            "step": step,
            "cumulative_runtime_s": float(repeat + step),
        }
        for case_key in ("ising_1d", "heisenberg_1d", "ising_2d", "heisenberg_2d")
        for method in METHODS
        for repeat in range(3)
        for step in range(2)
    ]
    summary = _summary_rows(rows)
    first_evolved = next(
        row for row in summary if row["case"] == "ising_1d" and row["method"] == METHODS[0] and row["step"] == 1
    )
    assert first_evolved["median_cumulative_runtime_s"] == pytest.approx(2.0)
    assert first_evolved["min_cumulative_runtime_s"] == pytest.approx(1.0)
    assert first_evolved["max_cumulative_runtime_s"] == pytest.approx(3.0)


def test_timing_summary_rejects_missing_repeat() -> None:
    """A runtime curve cannot silently use fewer than three repeats."""
    rows = [
        {
            "case": "ising_1d",
            "method": METHODS[0],
            "step": 1,
            "cumulative_runtime_s": float(repeat),
        }
        for repeat in range(2)
    ]
    with pytest.raises(RuntimeError, match="Expected 3 timing repeats"):
        _summary_rows(rows)


def _variational_plot_fixture() -> tuple[
    list[dict[str, str]],
    dict[str, object],
    dict[str, object],
]:
    primary = {
        "campaign_id": "primary-campaign",
        "source_hash": "primary-source",
        "cases": {
            case_key: {"stop_step": 3} for case_key in ("ising_1d", "heisenberg_1d", "ising_2d", "heisenberg_2d")
        },
    }
    control_cases = {
        case_key: {
            "status": "success",
            "primary_endpoint": 3,
            "stop_reason": "runtime_budget_reached_at_completed_step",
            "stop_step": 2,
            "cumulative_runtime_s": 120.0,
            "all_selected_fits_converged": True,
        }
        for case_key in ("ising_1d", "heisenberg_1d", "ising_2d", "heisenberg_2d")
    }
    control = {
        "campaign_id": VARIATIONAL_CAMPAIGN_ID,
        "source_hash": "control-source",
        "primary_campaign_id": "primary-campaign",
        "primary_source_hash": "primary-source",
        "runtime_budget_s": VARIATIONAL_RUNTIME_BUDGET_S,
        "runtime_scope": {"threads": 1, "repeats": 1},
        "cases": control_cases,
    }
    rows = [
        {
            "case": case_key,
            "method": "variational_mpo",
            "chi_cap": "32",
            "step": str(step),
            "cumulative_runtime_s": str((0.0, 40.0, 120.0)[step]),
            "infidelity_normalized": str(step * 1e-3),
            "current_parameter_count": str(32 + step),
        }
        for case_key in ("ising_1d", "heisenberg_1d", "ising_2d", "heisenberg_2d")
        for step in range(3)
    ]
    return rows, control, primary


def _variational_runtime_censor_fixture() -> dict[str, object]:
    """Return a state-free lower-bound record matching the plotting schema."""
    return {
        "schema_version": VARIATIONAL_CENSOR_SCHEMA_VERSION,
        "record_type": VARIATIONAL_CENSOR_RECORD_TYPE,
        "campaign_id": VARIATIONAL_CAMPAIGN_ID,
        "control_source_hash": "control-source",
        "primary_campaign_id": "primary-campaign",
        "primary_source_hash": "primary-source",
        "case": "heisenberg_2d",
        "chi_cap": 32,
        "attempted_step": 1,
        "last_completed_step": 0,
        "plot_step": 1,
        "status": "runtime_censored",
        "attempted_step_completed": False,
        "state_metrics_available": False,
        "runtime_lower_bound_s": 100.0,
        "bound_relation": "greater_than",
        "runtime_quantity": "single_thread_wall_time_since_attempted_step_started",
        "threads": 1,
        "repeats": 1,
        "warmups": 0,
        "interruption_reason": "Interrupted after the configured runtime-censoring budget.",
    }


def test_variational_control_stops_only_after_complete_step() -> None:
    """Apply the runtime budget only at a state comparable to the dense step."""
    assert _stop_reason(cumulative_runtime_s=99.9, step=2, primary_endpoint=3) is None
    assert _stop_reason(cumulative_runtime_s=100.0, step=2, primary_endpoint=3) == (
        "runtime_budget_reached_at_completed_step"
    )
    assert _stop_reason(cumulative_runtime_s=99.9, step=3, primary_endpoint=3) == ("primary_panel_endpoint")


def test_variational_rows_are_contiguous_and_runtime_censored() -> None:
    """Require one uninterrupted curve and the first budget-crossing endpoint."""
    rows, control, primary = _variational_plot_fixture()
    _validate_variational_rows(rows, control, primary)

    bad_rows = [dict(row) for row in rows]
    bad_rows[1]["cumulative_runtime_s"] = "101"
    with pytest.raises(RuntimeError, match="runtime censoring"):
        _validate_variational_rows(bad_rows, control, primary)


def test_variational_caption_distinguishes_single_censored_observation() -> None:
    """Do not present the bounded control as a repeated or saturated curve."""
    _, control, primary = _variational_plot_fixture()
    text = caption(primary, {"repeats": 3}, control)
    assert "one complete one-thread observation" in text
    assert "computational censoring, not accuracy saturation" in text
    assert "reduces to and overlaps" in text


def test_incomplete_variational_step_is_runtime_only() -> None:
    """Accept the limit record only when no completed state row masquerades as data."""
    rows, control, primary = _variational_plot_fixture()
    censor = _variational_runtime_censor_fixture()
    control_cases = dict(control["cases"])
    control_cases.pop("heisenberg_2d")
    control["cases"] = control_cases
    rows = [row for row in rows if row["case"] != "heisenberg_2d"]
    _validate_variational_rows(rows, control, primary, censor)
    assert _validate_variational_runtime_censor(censor, control, primary) == "heisenberg_2d"

    fake_state_row = {
        "case": "heisenberg_2d",
        "method": "variational_mpo",
        "step": "1",
    }
    with pytest.raises(RuntimeError, match="cannot provide state rows"):
        _validate_variational_rows([*rows, fake_state_row], control, primary, censor)

    metric_censor = dict(censor)
    metric_censor["infidelity_normalized"] = 0.5
    with pytest.raises(RuntimeError, match="cannot carry state metrics"):
        _validate_variational_runtime_censor(metric_censor, control, primary)


def test_runtime_censor_uses_a_distinct_upward_limit_glyph() -> None:
    """Render the incomplete step as a lower bound rather than a data marker."""
    censor = _variational_runtime_censor_fixture()
    figure, axis = plt.subplots()
    _plot_variational_runtime_censor(axis, censor, "heisenberg_2d")
    assert len(axis.lines) == 1
    assert axis.lines[0].get_marker() == r"$\uparrow$"
    assert axis.lines[0].get_xdata().tolist() == [1]
    assert axis.lines[0].get_ydata().tolist() == [100.0]
    plt.close(figure)


def test_runtime_censor_caption_disclaims_missing_state_metrics() -> None:
    """State that the lower bound has no corresponding accuracy or size point."""
    _, control, primary = _variational_plot_fixture()
    censor = _variational_runtime_censor_fixture()
    text = caption(primary, {"repeats": 3}, control, censor)
    assert "upward purple caret" in text
    assert "lower bound $>100$ s" in text
    assert "runtime-censored, incomplete first step" in text
    assert "No corresponding infidelity or parameter datum exists" in text


def test_sequential_variational_gates_update_outer_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fitted state must be the input seen by the following separated gate."""

    class DummyState:
        def __init__(self, value: float) -> None:
            self.tensors = [np.asarray([value])]
            self.orthogonality_center: int | None = 0

        def set_center(self, center: int | None) -> None:
            self.orthogonality_center = center

    observed_inputs: list[float] = []

    def fake_apply(initial: DummyState, *_args: object, **_kwargs: object) -> SimpleNamespace:
        value = float(initial.tensors[0][0])
        observed_inputs.append(value)
        return SimpleNamespace(
            state=DummyState(value + 1.0),
            converged=True,
            objective_trace=[1.0, 0.5],
            update_trace=[1.0, 0.5],
            sweeps=1,
            initializer_objectives={"mpo_contract_compress": 1.0, "input": 1.1},
            best_initializer="mpo_contract_compress",
            rejected_nonimproving_updates=0,
            target_max_bond=2,
            target_parameter_count=8,
            runtime_s=0.01,
            fidelity_to_target=0.9,
            objective_initial=1.0,
            objective_final=0.5,
        )

    monkeypatch.setattr(
        variational_control,
        "apply_variational_mpo_node",
        fake_apply,
    )
    gate = SimpleNamespace(
        gate=SimpleNamespace(qubits=(0, 2), name="rzz"),
        node=object(),
    )
    compiled_step = SimpleNamespace(gates=(gate, gate))
    state = DummyState(0.0)
    _, diagnostics = _apply_variational_step(
        state,
        compiled_step,
        object(),
        case_key="ising_2d",
        step_number=1,
    )
    assert observed_inputs == [0.0, 1.0]
    assert float(state.tensors[0][0]) == pytest.approx(2.0)
    assert len(diagnostics) == 2


def test_runtime_plot_requires_common_contiguous_endpoint() -> None:
    """Every runtime method must cover every step through the shared stop."""
    rows = [
        {
            "case": "ising_1d",
            "method": method,
            "step": str(step),
            "repeats": "3",
            "median_cumulative_runtime_s": str(float(step)),
            "min_cumulative_runtime_s": str(float(step)),
            "max_cumulative_runtime_s": str(float(step)),
        }
        for method in METHODS
        for step in range(4)
    ]
    _validate_runtime_rows(rows, "ising_1d", stop_step=3)
    rows.pop()
    with pytest.raises(RuntimeError, match="Incomplete runtime trajectory"):
        _validate_runtime_rows(rows, "ising_1d", stop_step=3)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("median_cumulative_runtime_s", "nan"),
        ("min_cumulative_runtime_s", "0.001"),
        ("max_cumulative_runtime_s", "501.0"),
    ],
)
def test_runtime_plot_rejects_nonfinite_or_clipped_values(
    field: str,
    value: str,
) -> None:
    """Runtime curves must be finite and fully contained by the plotted limits."""
    rows = [
        {
            "case": "ising_1d",
            "method": method,
            "step": str(step),
            "repeats": "3",
            "median_cumulative_runtime_s": str(float(step)),
            "min_cumulative_runtime_s": str(float(step)),
            "max_cumulative_runtime_s": str(float(step)),
        }
        for method in METHODS
        for step in range(4)
    ]
    target = next(row for row in rows if row["method"] == METHODS[0] and row["step"] == "1")
    target[field] = value
    with pytest.raises(RuntimeError, match="Invalid cumulative runtime summary"):
        _validate_runtime_rows(rows, "ising_1d", stop_step=3)


def _write_tdvp_override_manifest_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write a compact content-addressed manifest fixture.

    Returns:
        The override and base campaign directories.
    """
    base_dir = tmp_path / "base"
    override_dir = base_dir / "tdvp-control"
    override_dir.mkdir(parents=True)
    base_trajectory = base_dir / "trajectory_rows.csv"
    base_trajectory.write_text("base trajectory\n", encoding="utf-8")
    base_manifest = {
        "campaign_id": "circuit-infidelity-until-saturation-v2",
        "source_hash": "strict-source",
        "cases": {
            case_key: {
                "status": "success",
                "criterion_met": True,
                "right_censored": False,
                "stop_step": 30,
            }
            for case_key in ("ising_1d", "heisenberg_1d", "ising_2d", "heisenberg_2d")
        },
    }
    base_manifest_path = base_dir / "manifest.json"
    base_manifest_path.write_text(json.dumps(base_manifest), encoding="utf-8")

    artifacts = {}
    for name, filename in {
        "trajectory_rows": "trajectory_rows.csv",
        "bond_profiles": "bond_profiles.csv",
        "timing_rows": "timing_rows.csv",
        "timing_summary": "timing_summary.csv",
    }.items():
        path = override_dir / filename
        path.write_text(f"{name}\n", encoding="utf-8")
        artifacts[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "campaign_id": TDVP_OVERRIDE_CAMPAIGN_ID,
        "source_hash": "control-source",
        "environment": {
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        },
        "base_provenance": {
            "campaign_id": base_manifest["campaign_id"],
            "source_hash": base_manifest["source_hash"],
            "manifest_sha256": hashlib.sha256(base_manifest_path.read_bytes()).hexdigest(),
            "trajectory_sha256": hashlib.sha256(base_trajectory.read_bytes()).hexdigest(),
        },
        "protocol": {
            "method": TDVP_OVERRIDE_METHOD,
            "chi_cap": 32,
            "n_sub": 2,
            "krylov_tolerance": TDVP_OVERRIDE_TOLERANCE,
            "svd_threshold": 1e-13,
            "truncation_mode": "discarded_weight",
            "frozen_endpoints": dict.fromkeys(
                ("ising_1d", "heisenberg_1d", "ising_2d", "heisenberg_2d"),
                30,
            ),
            "bond_profile_max_step": 30,
            "threads": 1,
        },
        "timing_scope": {
            "warmup_trajectories_per_case": 1,
            "measured_repeats": 3,
            "included": "apply_mps_step for every gate in each complete Trotter step",
        },
        "cases": {
            case_key: {
                "stop_step": 30,
                "accuracy_status": "success",
                "endpoint_infidelity": 0.1,
                "timing_repeats_complete": 3,
            }
            for case_key in ("ising_1d", "heisenberg_1d", "ising_2d", "heisenberg_2d")
        },
        "artifacts": artifacts,
    }
    (override_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return override_dir, base_dir


def test_tdvp_override_manifest_authenticates_protocol_and_artifacts(tmp_path: Path) -> None:
    """Accept only the complete tau=1e-5 control tied to the strict inputs."""
    override_dir, base_dir = _write_tdvp_override_manifest_fixture(tmp_path)
    manifest = load_validated_tdvp_override_manifest(override_dir, base_dir)
    assert manifest["protocol"]["krylov_tolerance"] == TDVP_OVERRIDE_TOLERANCE

    (override_dir / "timing_summary.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest digest"):
        load_validated_tdvp_override_manifest(override_dir, base_dir)


def test_tdvp_override_manifest_rejects_a_loose_or_incomplete_protocol(tmp_path: Path) -> None:
    """Do not silently plot exploratory tolerances or partial timing campaigns."""
    override_dir, base_dir = _write_tdvp_override_manifest_fixture(tmp_path)
    manifest_path = override_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"]["krylov_tolerance"] = 1e-3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="publication control"):
        load_validated_tdvp_override_manifest(override_dir, base_dir)


def test_tdvp_row_override_preserves_every_comparator_row() -> None:
    """Only TDVP data may change when the isolated control is overlaid."""
    base = [
        {"case": "ising_1d", "method": method, "step": str(step), "value": f"base-{method}-{step}"}
        for method in METHODS
        for step in range(2)
    ]
    override = [
        {
            "case": "ising_1d",
            "method": TDVP_OVERRIDE_METHOD,
            "step": str(step),
            "value": f"control-{step}",
        }
        for step in range(2)
    ]
    combined = apply_tdvp_row_override(base, override, table="accuracy")
    assert [row for row in combined if row["method"] == TDVP_OVERRIDE_METHOD] == override
    assert [row for row in combined if row["method"] != TDVP_OVERRIDE_METHOD] == [
        row for row in base if row["method"] != TDVP_OVERRIDE_METHOD
    ]


def test_tdvp_override_caption_does_not_reassert_the_flatness_criterion() -> None:
    """Frozen control windows must not be described as newly satisfying the stop rule."""
    _, control, primary = _variational_plot_fixture()
    text = caption(
        primary,
        {"repeats": 3},
        control,
        tdvp_override_manifest={"protocol": {"krylov_tolerance": TDVP_OVERRIDE_TOLERANCE}},
    )
    assert "selected and frozen by the original strict-Krylov campaign" in text
    assert "do not assert that the control itself re-satisfies" in text
    assert "control at tolerance $10^{-5}$" in text
