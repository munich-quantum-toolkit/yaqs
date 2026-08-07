# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Plot fixed-cap circuit accuracy, retained parameters, and runtime."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from experiments.circuit_benchmarks.config import METHODS, RELIABILITY_THRESHOLD, REPO_ROOT
from experiments.circuit_benchmarks.plotting import (
    CASE_LABELS,
    METHOD_STYLES,
)
from experiments.circuit_benchmarks.plotting import (
    apply_style as _apply_style,
)
from experiments.circuit_benchmarks.plotting import (
    legend_handles as _legend_handles,
)
from experiments.circuit_benchmarks.plotting import (
    style_axis as _style_axis,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

from .config import (
    CASE_ORDER,
    DISPLAY_FLOOR,
    DPI,
    FIGURE_WIDTH_MM,
    OUTPUT_DIR,
    SATURATION_LOG_RANGE_DECADES,
    SATURATION_WINDOW_STEPS,
)

FIGURE_STEM = "figure_circuit_infidelities"
MM_TO_IN = 1.0 / 25.4
FIGURE_HEIGHT_MM = 150.0
TIMING_DIRNAME = "timing"
VARIATIONAL_DIRNAME = "variational_mpo_control"
VARIATIONAL_CENSOR_FILENAME = "runtime_censor.json"
FULL_PROFILE_PARAMETERS = 15016
PARAMETER_Y_LIMITS = (25, 2.2e4)
PARAMETER_INSET_Y_LIMITS = (25, 1.8e4)
PARAMETER_INSET_Y_TICKS = (1e2, 1e3, 1e4)
RUNTIME_Y_LIMITS = (1e-2, 1e3)
TIMING_CAMPAIGN_ID = "circuit-fixed-endpoint-timing-v1"
VARIATIONAL_CAMPAIGN_ID = "circuit-long-trajectory-variational-mpo-v1"
VARIATIONAL_METHOD = "variational_mpo"
VARIATIONAL_RUNTIME_BUDGET_S = 1.0e2
VARIATIONAL_CENSOR_SCHEMA_VERSION = 1
VARIATIONAL_CENSOR_RECORD_TYPE = "incomplete_variational_step_runtime_lower_bound"
VARIATIONAL_STYLE = {
    "color": "#CC79A7",
    "marker": "D",
    "linestyle": ":",
}

PLATEAU_INSET_Y = {
    "ising_1d": ((0.135, 0.16), (0.14, 0.15, 0.16)),
    "heisenberg_1d": ((0.50, 0.61), (0.52, 0.56, 0.60)),
    "ising_2d": ((0.055, 0.10), (0.06, 0.08, 0.10)),
    "heisenberg_2d": ((0.50, 0.69), (0.52, 0.60, 0.68)),
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        msg = f"Missing {path}; run the variable-length trajectory campaign first."
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        msg = f"Missing {path}; run the variable-length trajectory campaign first."
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        msg = f"Expected a JSON object in {path}."
        raise TypeError(msg)
    return value


def _case_method_rows(
    rows: list[dict[str, str]],
    case_key: str,
    method: str,
) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("case") == case_key and row.get("method") == method]
    return sorted(selected, key=lambda row: int(row["step"]))


def _validate_case_rows(rows: list[dict[str, str]], case_key: str) -> int:
    """Require every method to cover the same contiguous case trajectory."""
    steps_by_method: dict[str, list[int]] = {}
    for method in METHODS:
        points = _case_method_rows(rows, case_key, method)
        if not points:
            msg = f"No variable-length trajectory for {case_key}/{method}."
            raise RuntimeError(msg)
        steps_by_method[method] = [int(row["step"]) for row in points]
    first = steps_by_method[METHODS[0]]
    if first != list(range(first[-1] + 1)):
        msg = f"Noncontiguous trajectory for {case_key}/{METHODS[0]}."
        raise RuntimeError(msg)
    for method, steps in steps_by_method.items():
        if steps != first:
            msg = f"Methods do not share one endpoint in {case_key}: {method}."
            raise RuntimeError(msg)
    return first[-1]


def _runtime_method_rows(
    rows: list[dict[str, str]],
    case_key: str,
    method: str,
) -> list[dict[str, str]]:
    """Return sorted timing-summary rows for one curve."""
    selected = [row for row in rows if row.get("case") == case_key and row.get("method") == method]
    return sorted(selected, key=lambda row: int(row["step"]))


def _variational_case_rows(
    rows: list[dict[str, str]],
    case_key: str,
) -> list[dict[str, str]]:
    """Return the optional runtime-censored control for one circuit."""
    selected = [row for row in rows if row.get("case") == case_key and row.get("method") == VARIATIONAL_METHOD]
    return sorted(selected, key=lambda row: int(row["step"]))


def _validate_variational_rows(
    rows: list[dict[str, str]],
    manifest: dict[str, object],
    primary_manifest: dict[str, object],
    runtime_censor: dict[str, object] | None = None,
) -> None:
    """Validate optional controls without changing primary panel endpoints."""
    if manifest.get("campaign_id") != VARIATIONAL_CAMPAIGN_ID:
        msg = "Unexpected long-trajectory variational-control campaign."
        raise RuntimeError(msg)
    if manifest.get("primary_campaign_id") != primary_manifest.get("campaign_id") or manifest.get(
        "primary_source_hash"
    ) != primary_manifest.get("source_hash"):
        msg = "Variational-control provenance does not match the primary campaign."
        raise RuntimeError(msg)
    if float(manifest.get("runtime_budget_s", float("nan"))) != VARIATIONAL_RUNTIME_BUDGET_S:
        msg = "Unexpected variational-control runtime budget."
        raise RuntimeError(msg)
    runtime_scope = manifest.get("runtime_scope")
    if not isinstance(runtime_scope, dict) or runtime_scope.get("threads") != 1 or runtime_scope.get("repeats") != 1:
        msg = "Variational control must be one complete one-thread observation."
        raise RuntimeError(msg)

    primary_cases = primary_manifest.get("cases")
    control_cases = manifest.get("cases")
    if not isinstance(primary_cases, dict) or not isinstance(control_cases, dict):
        msg = "Missing primary or variational case records."
        raise RuntimeError(msg)
    censored_case = None
    if runtime_censor is not None:
        censored_case = _validate_variational_runtime_censor(
            runtime_censor,
            manifest,
            primary_manifest,
        )
    for case_key in CASE_ORDER:
        primary_record = primary_cases.get(case_key)
        control_record = control_cases.get(case_key)
        if not isinstance(primary_record, dict):
            msg = f"Missing primary case record for {case_key}."
            raise RuntimeError(msg)
        if not isinstance(control_record, dict):
            if case_key != censored_case:
                msg = f"Missing variational case record for {case_key}."
                raise RuntimeError(msg)
            if _variational_case_rows(rows, case_key):
                msg = f"An incomplete variational step cannot provide state rows for {case_key}."
                raise RuntimeError(msg)
            continue
        if control_record.get("status") != "success" or control_record.get("all_selected_fits_converged") is not True:
            msg = f"Incomplete variational control for {case_key}."
            raise RuntimeError(msg)
        primary_stop = int(primary_record["stop_step"])
        stop_step = int(control_record["stop_step"])
        if int(control_record["primary_endpoint"]) != primary_stop or not 1 <= stop_step <= primary_stop:
            msg = f"Invalid variational endpoint for {case_key}."
            raise RuntimeError(msg)

        points = _variational_case_rows(rows, case_key)
        steps = [int(point["step"]) for point in points]
        if steps != list(range(stop_step + 1)):
            msg = f"Noncontiguous variational trajectory for {case_key}."
            raise RuntimeError(msg)
        if any(int(point["chi_cap"]) != 32 for point in points):
            msg = f"Unexpected variational bond cap for {case_key}."
            raise RuntimeError(msg)
        runtimes = np.asarray([float(point["cumulative_runtime_s"]) for point in points])
        infidelities = np.asarray([float(point["infidelity_normalized"]) for point in points])
        parameters = np.asarray([float(point["current_parameter_count"]) for point in points])
        if (
            not np.all(np.isfinite(runtimes))
            or not np.all(np.isfinite(infidelities))
            or not np.all(np.isfinite(parameters))
            or runtimes[0] != 0.0
            or np.any(runtimes[1:] <= 0.0)
            or np.any(np.diff(runtimes) < 0.0)
            or np.any(infidelities < -1e-12)
            or np.any(infidelities > 1.0 + 1e-12)
            or np.any(parameters <= 0.0)
        ):
            msg = f"Invalid variational trajectory values for {case_key}."
            raise RuntimeError(msg)
        reported_runtime = float(control_record.get("cumulative_runtime_s", float("nan")))
        if not np.isfinite(reported_runtime) or not np.isclose(
            runtimes[-1],
            reported_runtime,
            rtol=0.0,
            atol=1e-9,
        ):
            msg = f"Variational terminal runtime disagrees with the manifest for {case_key}."
            raise RuntimeError(msg)

        reason = control_record.get("stop_reason")
        if reason == "runtime_budget_reached_at_completed_step":
            if runtimes[-1] < VARIATIONAL_RUNTIME_BUDGET_S or np.any(runtimes[1:-1] >= VARIATIONAL_RUNTIME_BUDGET_S):
                msg = f"Variational runtime censoring is inconsistent for {case_key}."
                raise RuntimeError(msg)
        elif reason == "primary_panel_endpoint":
            if stop_step != primary_stop or runtimes[-1] >= VARIATIONAL_RUNTIME_BUDGET_S:
                msg = f"Variational panel-endpoint stop is inconsistent for {case_key}."
                raise RuntimeError(msg)
        else:
            msg = f"Unknown variational stop reason for {case_key}: {reason!r}."
            raise RuntimeError(msg)

    if runtime_censor is not None and censored_case in control_cases:
        msg = f"A case cannot be both completed and runtime-censored: {censored_case}."
        raise RuntimeError(msg)


def _validate_variational_runtime_censor(
    record: dict[str, object],
    manifest: dict[str, object],
    primary_manifest: dict[str, object],
) -> str:
    """Validate a runtime lower bound that contains no completed state datum."""
    expected_exact = {
        "schema_version": VARIATIONAL_CENSOR_SCHEMA_VERSION,
        "record_type": VARIATIONAL_CENSOR_RECORD_TYPE,
        "campaign_id": manifest.get("campaign_id"),
        "control_source_hash": manifest.get("source_hash"),
        "primary_campaign_id": primary_manifest.get("campaign_id"),
        "primary_source_hash": primary_manifest.get("source_hash"),
        "chi_cap": 32,
        "status": "runtime_censored",
        "bound_relation": "greater_than",
        "runtime_quantity": "single_thread_wall_time_since_attempted_step_started",
        "threads": 1,
        "repeats": 1,
        "warmups": 0,
        "state_metrics_available": False,
        "attempted_step_completed": False,
    }
    mismatches = {
        key: (record.get(key), expected)
        for key, expected in expected_exact.items()
        if record.get(key) != expected
    }
    if mismatches:
        msg = f"Invalid variational runtime-censor identity: {mismatches}."
        raise RuntimeError(msg)
    case_key = record.get("case")
    if case_key not in CASE_ORDER:
        msg = f"Unknown variational runtime-censor case {case_key!r}."
        raise RuntimeError(msg)
    attempted_step = int(record.get("attempted_step", -1))
    completed_step = int(record.get("last_completed_step", -1))
    plot_step = int(record.get("plot_step", -1))
    if attempted_step != completed_step + 1 or plot_step != attempted_step or completed_step < 0:
        msg = "The runtime censor must identify the first incomplete step after the last completed state."
        raise RuntimeError(msg)
    lower_bound = float(record.get("runtime_lower_bound_s", float("nan")))
    if not np.isfinite(lower_bound) or lower_bound < VARIATIONAL_RUNTIME_BUDGET_S:
        msg = "The variational runtime lower bound is missing or below the declared compute budget."
        raise RuntimeError(msg)
    forbidden_state_fields = {
        "fidelity_normalized",
        "infidelity_normalized",
        "current_parameter_count",
        "current_peak_bond_dim",
        "state",
    }
    present = sorted(forbidden_state_fields.intersection(record))
    if present:
        msg = f"An incomplete step cannot carry state metrics: {present}."
        raise RuntimeError(msg)
    reason = record.get("interruption_reason")
    if not isinstance(reason, str) or not reason.strip():
        msg = "The variational runtime censor requires a nonempty interruption reason."
        raise RuntimeError(msg)
    return str(case_key)


def _validate_runtime_rows(
    rows: list[dict[str, str]],
    case_key: str,
    stop_step: int,
) -> None:
    """Require every timing curve to cover the frozen common endpoint."""
    expected = list(range(stop_step + 1))
    for method in METHODS:
        points = _runtime_method_rows(rows, case_key, method)
        steps = [int(row["step"]) for row in points]
        if steps != expected:
            msg = f"Incomplete runtime trajectory for {case_key}/{method}."
            raise RuntimeError(msg)
        if any(int(row["repeats"]) != 3 for row in points):
            msg = f"Runtime summary for {case_key}/{method} does not contain three repeats."
            raise RuntimeError(msg)
        medians = np.asarray([float(row["median_cumulative_runtime_s"]) for row in points])
        lows = np.asarray([float(row["min_cumulative_runtime_s"]) for row in points])
        highs = np.asarray([float(row["max_cumulative_runtime_s"]) for row in points])
        if (
            not np.all(np.isfinite(medians))
            or not np.all(np.isfinite(lows))
            or not np.all(np.isfinite(highs))
            or medians[0] != 0.0
            or lows[0] != 0.0
            or highs[0] != 0.0
            or np.any(medians[1:] <= 0.0)
            or np.any(lows > medians)
            or np.any(medians > highs)
            or np.any(np.diff(medians) < 0.0)
            or np.any(np.diff(lows) < 0.0)
            or np.any(np.diff(highs) < 0.0)
            or np.any(lows[1:] < RUNTIME_Y_LIMITS[0])
            or np.any(highs[1:] > RUNTIME_Y_LIMITS[1])
        ):
            msg = f"Invalid cumulative runtime summary for {case_key}/{method}."
            raise RuntimeError(msg)


def _marker_indices(length: int, method_index: int) -> np.ndarray:
    """Return sparse, staggered marker locations for a variable-length curve."""
    if length <= 1:
        return np.asarray([0], dtype=int)
    stride = max(2, int(np.ceil(length / 10)))
    values = np.arange(method_index, length, stride, dtype=int)
    if not len(values) or values[-1] != length - 1:
        values = np.unique(np.append(values, length - 1))
    return values


def _plot_variational_series(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    case_key: str,
    field: str,
    *,
    display_floor: float | None = None,
    maximum_step: int | None = None,
    omit_initial: bool = False,
) -> None:
    """Plot the optional one-observation control without changing any endpoint."""
    points = _variational_case_rows(rows, case_key)
    if maximum_step is not None:
        points = [point for point in points if int(point["step"]) <= maximum_step]
    if omit_initial:
        points = [point for point in points if int(point["step"]) > 0]
    if not points:
        return
    steps = np.asarray([int(point["step"]) for point in points], dtype=int)
    values = np.asarray([float(point[field]) for point in points], dtype=float)
    if display_floor is not None:
        values = np.maximum(values, display_floor)
    axis.plot(
        steps,
        values,
        color=VARIATIONAL_STYLE["color"],
        linestyle=VARIATIONAL_STYLE["linestyle"],
        linewidth=1.55,
        zorder=5,
    )
    markers = _marker_indices(len(steps), method_index=len(METHODS))
    axis.plot(
        steps[markers],
        values[markers],
        linestyle="none",
        color=VARIATIONAL_STYLE["color"],
        marker=VARIATIONAL_STYLE["marker"],
        markersize=4.0,
        markeredgewidth=0.8,
        markeredgecolor=VARIATIONAL_STYLE["color"],
        markerfacecolor="white",
        zorder=10,
    )


def _plot_variational_runtime_censor(
    axis: plt.Axes,
    record: dict[str, object] | None,
    case_key: str,
) -> None:
    """Plot an upward caret for an interrupted step with no endpoint state."""
    if record is None or record.get("case") != case_key:
        return
    axis.plot(
        [int(record["plot_step"])],
        [float(record["runtime_lower_bound_s"])],
        linestyle="none",
        color=VARIATIONAL_STYLE["color"],
        marker=r"$\uparrow$",
        markersize=9.0,
        markeredgewidth=1.2,
        markeredgecolor=VARIATIONAL_STYLE["color"],
        zorder=12,
    )


def _plateau_window(
    points: list[dict[str, str]],
    stop_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exactly the trailing samples used by the plateau criterion."""
    start_step = stop_step - SATURATION_WINDOW_STEPS + 1
    selected = [point for point in points if start_step <= int(point["step"]) <= stop_step]
    steps = np.asarray([int(point["step"]) for point in selected], dtype=int)
    expected = np.arange(start_step, stop_step + 1, dtype=int)
    if not np.array_equal(steps, expected):
        msg = f"Expected plateau steps {expected.tolist()}, received {steps.tolist()}."
        raise RuntimeError(msg)
    errors = np.asarray([max(float(point["infidelity_normalized"]), DISPLAY_FLOOR) for point in selected])
    return steps, errors


def _plot_plateau_inset(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    case_key: str,
    stop_step: int,
) -> plt.Axes:
    """Add a linear-scale view of the final plateau window."""
    inset = axis.inset_axes((0.56, 0.14, 0.40, 0.34), zorder=10)
    for method_index, method in enumerate(METHODS):
        points = _case_method_rows(rows, case_key, method)
        steps, errors = _plateau_window(points, stop_step)
        style = METHOD_STYLES[method]
        inset.plot(
            steps,
            errors,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.35,
            marker=style["marker"],
            markersize=3.3,
            markeredgewidth=0.75,
            markeredgecolor=style["color"],
            markerfacecolor="white" if method == "tebd_swap" else style["color"],
            markevery=(method_index, 3),
            zorder=2 + method_index,
        )

    start_step = stop_step - SATURATION_WINDOW_STEPS + 1
    inset.set_xlim(start_step - 0.5, stop_step + 0.5)
    inset.set_xticks((stop_step - 8, stop_step - 4, stop_step))
    y_limits, y_ticks = PLATEAU_INSET_Y[case_key]
    inset.set_ylim(*y_limits)
    inset.set_yticks(y_ticks)
    inset.set_xlabel(r"$n$", fontsize=5.8, labelpad=0.5)
    if case_key != "ising_1d":
        inset.set_ylabel(r"$1-F$", fontsize=5.8, labelpad=0.8)
    inset.tick_params(
        which="both",
        direction="out",
        width=0.55,
        length=1.7,
        labelsize=5.8,
        pad=1.0,
    )
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_linewidth(0.55)
    return inset


def _plot_case(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    case_key: str,
    panel_label: str,
    variational_rows: list[dict[str, str]] | None = None,
) -> None:
    stop_step = _validate_case_rows(rows, case_key)
    for method_index, method in enumerate(METHODS):
        points = _case_method_rows(rows, case_key, method)
        steps = np.asarray([int(row["step"]) for row in points], dtype=int)
        errors = np.asarray([max(float(row["infidelity_normalized"]), DISPLAY_FLOOR) for row in points])
        style = METHOD_STYLES[method]
        axis.plot(
            steps,
            errors,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            zorder=2 + method_index,
        )
        markers = _marker_indices(len(steps), method_index)
        axis.plot(
            steps[markers],
            errors[markers],
            linestyle="none",
            color=style["color"],
            marker=style["marker"],
            markersize=4.1,
            markeredgewidth=0.8,
            markeredgecolor=style["color"],
            markerfacecolor="white" if method == "tebd_swap" else style["color"],
            zorder=6 + method_index,
        )

    if variational_rows:
        _plot_variational_series(
            axis,
            variational_rows,
            case_key,
            "infidelity_normalized",
            display_floor=DISPLAY_FLOOR,
        )

    if stop_step <= 0:
        msg = f"Trajectory for {case_key} contains no evolved step."
        raise RuntimeError(msg)
    axis.axhline(RELIABILITY_THRESHOLD, color="0.35", linewidth=0.85, linestyle=":")
    axis.set_yscale("log")
    plateau_start = stop_step - SATURATION_WINDOW_STEPS + 1
    axis.axvspan(
        plateau_start - 0.5,
        stop_step + 0.5,
        color="0.6",
        alpha=0.08,
        linewidth=0,
        zorder=0,
    )
    axis.set_xlim(-0.5, stop_step + 0.5)
    axis.set_ylim(DISPLAY_FLOOR, 1.2)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=3))
    axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 5)))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_title(CASE_LABELS[case_key], pad=3.0)
    axis.text(
        0.025,
        0.965,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        weight="bold",
        fontsize=8.8,
        zorder=20,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "pad": 0.4,
            "alpha": 0.85,
        },
    )
    _style_axis(axis)
    _plot_plateau_inset(axis, rows, case_key, stop_step)


def _plot_parameters(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    case_key: str,
    stop_step: int,
    panel_label: str,
    variational_rows: list[dict[str, str]] | None = None,
) -> None:
    """Plot retained step-end MPS tensor entries for one circuit."""
    for method_index, method in enumerate(METHODS):
        points = _case_method_rows(rows, case_key, method)
        steps = np.asarray([int(row["step"]) for row in points], dtype=int)
        parameters = np.asarray(
            [int(row["current_parameter_count"]) for row in points],
            dtype=int,
        )
        style = METHOD_STYLES[method]
        axis.plot(
            steps,
            parameters,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            zorder=2 + method_index,
        )
        markers = _marker_indices(len(steps), method_index)
        axis.plot(
            steps[markers],
            parameters[markers],
            linestyle="none",
            color=style["color"],
            marker=style["marker"],
            markersize=4.1,
            markeredgewidth=0.8,
            markeredgecolor=style["color"],
            markerfacecolor="white" if method == "tebd_swap" else style["color"],
            zorder=6 + method_index,
        )
    if variational_rows:
        _plot_variational_series(
            axis,
            variational_rows,
            case_key,
            "current_parameter_count",
        )
    axis.axhline(
        FULL_PROFILE_PARAMETERS,
        color="0.4",
        linewidth=0.85,
        linestyle=":",
        zorder=1,
    )
    axis.set_yscale("log")
    axis.set_ylim(*PARAMETER_Y_LIMITS)
    axis.set_xlim(-0.5, stop_step + 0.5)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=3))
    axis.text(
        0.025,
        0.86,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        weight="bold",
        fontsize=8.8,
    )
    _style_axis(axis)
    _plot_parameter_inset(
        axis,
        rows,
        case_key,
        stop_step,
        variational_rows=variational_rows,
    )


def _parameter_transient_stop(
    rows: list[dict[str, str]],
    case_key: str,
    stop_step: int,
) -> int:
    """Return the last step needed to show every parameter-growth transient."""
    stable_steps: list[int] = []
    for method in METHODS:
        points = _case_method_rows(rows, case_key, method)
        steps = np.asarray([int(row["step"]) for row in points], dtype=int)
        parameters = np.asarray(
            [int(row["current_parameter_count"]) for row in points],
            dtype=int,
        )
        differing = np.flatnonzero(parameters != parameters[-1])
        stable_index = int(differing[-1] + 1) if differing.size else 0
        stable_steps.append(int(steps[stable_index]))
    return min(stop_step, max(2, max(stable_steps)))


def _plot_parameter_inset(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    case_key: str,
    stop_step: int,
    variational_rows: list[dict[str, str]] | None = None,
) -> plt.Axes:
    """Expand the early steps containing the retained-size growth."""
    transient_stop = _parameter_transient_stop(rows, case_key, stop_step)
    inset = axis.inset_axes((0.50, 0.14, 0.46, 0.50), zorder=10)
    for method_index, method in enumerate(METHODS):
        points = [point for point in _case_method_rows(rows, case_key, method) if int(point["step"]) <= transient_stop]
        steps = np.asarray([int(point["step"]) for point in points], dtype=int)
        parameters = np.asarray(
            [int(point["current_parameter_count"]) for point in points],
            dtype=int,
        )
        style = METHOD_STYLES[method]
        inset.plot(
            steps,
            parameters,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.35,
            marker=style["marker"],
            markersize=3.3,
            markeredgewidth=0.75,
            markeredgecolor=style["color"],
            markerfacecolor="white" if method == "tebd_swap" else style["color"],
            markevery=_marker_indices(len(steps), method_index),
            zorder=2 + method_index,
        )

    if variational_rows:
        _plot_variational_series(
            inset,
            variational_rows,
            case_key,
            "current_parameter_count",
            maximum_step=transient_stop,
        )

    inset.axhline(
        FULL_PROFILE_PARAMETERS,
        color="0.4",
        linewidth=0.7,
        linestyle=":",
        zorder=1,
    )
    inset.set_xlim(-0.5, transient_stop + 0.5)
    inset.set_xticks((0, int(np.ceil(transient_stop / 2)), transient_stop))
    inset.set_yscale("log")
    inset.set_ylim(*PARAMETER_INSET_Y_LIMITS)
    inset.set_yticks(PARAMETER_INSET_Y_TICKS)
    inset.set_xlabel(r"$n$", fontsize=5.8, labelpad=0.5)
    inset.set_ylabel(r"$P$", fontsize=5.8, labelpad=0.8)
    inset.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 5)))
    inset.yaxis.set_minor_formatter(NullFormatter())
    inset.tick_params(
        which="both",
        direction="out",
        width=0.55,
        length=1.7,
        labelsize=5.8,
        pad=1.0,
        labelleft=True,
    )
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_linewidth(0.55)
    return inset


def _plot_runtime(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    case_key: str,
    stop_step: int,
    panel_label: str,
    variational_rows: list[dict[str, str]] | None = None,
    variational_runtime_censor: dict[str, object] | None = None,
) -> None:
    """Plot median cumulative update time with min--max repeat bands."""
    _validate_runtime_rows(rows, case_key, stop_step)
    for method_index, method in enumerate(METHODS):
        points = _runtime_method_rows(rows, case_key, method)
        steps = np.asarray([int(row["step"]) for row in points], dtype=int)
        median = np.asarray([float(row["median_cumulative_runtime_s"]) for row in points])
        low = np.asarray([float(row["min_cumulative_runtime_s"]) for row in points])
        high = np.asarray([float(row["max_cumulative_runtime_s"]) for row in points])
        positive = steps > 0
        steps = steps[positive]
        median = median[positive]
        low = low[positive]
        high = high[positive]
        style = METHOD_STYLES[method]
        axis.fill_between(
            steps,
            low,
            high,
            color=style["color"],
            alpha=0.16,
            linewidth=0,
            zorder=1 + method_index,
        )
        axis.plot(
            steps,
            median,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            zorder=3 + method_index,
        )
        markers = _marker_indices(len(steps), method_index)
        axis.plot(
            steps[markers],
            median[markers],
            linestyle="none",
            color=style["color"],
            marker=style["marker"],
            markersize=4.1,
            markeredgewidth=0.8,
            markeredgecolor=style["color"],
            markerfacecolor="white" if method == "tebd_swap" else style["color"],
            zorder=7 + method_index,
        )
    if variational_rows:
        _plot_variational_series(
            axis,
            variational_rows,
            case_key,
            "cumulative_runtime_s",
            omit_initial=True,
        )
    _plot_variational_runtime_censor(axis, variational_runtime_censor, case_key)
    axis.set_yscale("log")
    axis.set_ylim(*RUNTIME_Y_LIMITS)
    axis.set_xlim(-0.5, stop_step + 0.5)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=3))
    panel_label_x = (
        0.14
        if variational_runtime_censor is not None
        and variational_runtime_censor.get("case") == case_key
        else 0.025
    )
    axis.text(
        panel_label_x,
        0.95,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        weight="bold",
        fontsize=8.8,
    )
    _style_axis(axis)


def create_figure(
    rows: list[dict[str, str]],
    manifest: dict[str, object],
    runtime_rows: list[dict[str, str]],
    runtime_manifest: dict[str, object],
    variational_rows: list[dict[str, str]] | None = None,
    variational_manifest: dict[str, object] | None = None,
    variational_runtime_censor: dict[str, object] | None = None,
) -> plt.Figure:
    """Build the three-metric by four-circuit trajectory figure."""
    if (variational_rows is None) != (variational_manifest is None):
        msg = "Variational rows and manifest must be supplied together."
        raise RuntimeError(msg)
    if variational_runtime_censor is not None and variational_manifest is None:
        msg = "A variational runtime censor requires its control manifest."
        raise RuntimeError(msg)
    case_records = manifest.get("cases")
    if not isinstance(case_records, dict):
        msg = "The long-trajectory manifest has no case records."
        raise RuntimeError(msg)
    incomplete = [
        case_key
        for case_key in CASE_ORDER
        if not isinstance(case_records.get(case_key), dict) or not case_records[case_key].get("criterion_met")
    ]
    if incomplete:
        msg = f"Saturation criterion was not met for cases: {incomplete}."
        raise RuntimeError(msg)
    if runtime_manifest.get("endpoints") != {
        case_key: int(case_records[case_key]["stop_step"]) for case_key in CASE_ORDER
    }:
        msg = "Timing endpoints do not match the frozen adaptive endpoints."
        raise RuntimeError(msg)
    if int(runtime_manifest.get("repeats", 0)) != 3:
        msg = "The runtime row requires three measured repeats."
        raise RuntimeError(msg)
    if runtime_manifest.get("campaign_id") != TIMING_CAMPAIGN_ID:
        msg = "Unexpected fixed-endpoint timing campaign."
        raise RuntimeError(msg)
    timing_scope = runtime_manifest.get("timing_scope")
    if (
        not isinstance(timing_scope, dict)
        or timing_scope.get("threads") != 1
        or timing_scope.get("included") != "apply_mps_step for every gate in each complete Trotter step"
    ):
        msg = "Unexpected runtime timing scope."
        raise RuntimeError(msg)
    if runtime_manifest.get("adaptive_campaign_id") != manifest.get("campaign_id") or runtime_manifest.get(
        "adaptive_source_hash"
    ) != manifest.get("source_hash"):
        msg = "Timing provenance does not match the plotted adaptive campaign."
        raise RuntimeError(msg)
    if variational_rows is not None and variational_manifest is not None:
        _validate_variational_rows(
            variational_rows,
            variational_manifest,
            manifest,
            variational_runtime_censor,
        )

    _apply_style()
    mpl.rcParams["axes.titlesize"] = 8.5
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
    )
    grid = figure.add_gridspec(
        3,
        len(CASE_ORDER),
        height_ratios=(1.25, 1.0, 1.0),
        wspace=0.16,
        hspace=0.11,
    )
    panel_labels = (
        ("(a)", "(b)", "(c)", "(d)"),
        ("(e)", "(f)", "(g)", "(h)"),
        ("(i)", "(j)", "(k)", "(l)"),
    )
    infidelity_axes: list[plt.Axes] = []
    parameter_axes: list[plt.Axes] = []
    runtime_axes: list[plt.Axes] = []
    for column, case_key in enumerate(CASE_ORDER):
        stop_step = int(case_records[case_key]["stop_step"])
        infidelity_axis = figure.add_subplot(
            grid[0, column],
            sharey=infidelity_axes[0] if infidelity_axes else None,
        )
        parameter_axis = figure.add_subplot(
            grid[1, column],
            sharex=infidelity_axis,
            sharey=parameter_axes[0] if parameter_axes else None,
        )
        runtime_axis = figure.add_subplot(
            grid[2, column],
            sharex=infidelity_axis,
            sharey=runtime_axes[0] if runtime_axes else None,
        )
        _plot_case(
            infidelity_axis,
            rows,
            case_key,
            panel_labels[0][column],
            variational_rows=variational_rows,
        )
        _plot_parameters(
            parameter_axis,
            rows,
            case_key,
            stop_step,
            panel_labels[1][column],
            variational_rows=variational_rows,
        )
        _plot_runtime(
            runtime_axis,
            runtime_rows,
            case_key,
            stop_step,
            panel_labels[2][column],
            variational_rows=variational_rows,
            variational_runtime_censor=variational_runtime_censor,
        )
        infidelity_axis.tick_params(labelbottom=False, labelleft=column == 0)
        parameter_axis.tick_params(labelbottom=False, labelleft=column == 0)
        runtime_axis.tick_params(labelleft=column == 0)
        infidelity_axes.append(infidelity_axis)
        parameter_axes.append(parameter_axis)
        runtime_axes.append(runtime_axis)

    infidelity_axes[0].set_ylabel(r"Infidelity $1-F$")
    parameter_axes[0].set_ylabel(r"Total parameters $P$")
    runtime_axes[0].set_ylabel("Runtime (s)")
    figure.supxlabel(r"Trotter steps $n$", y=0.025)
    legend_handles = list(_legend_handles())
    legend_labels = ["TDVP", "MPO", "TEBD+SWAP"]
    if variational_rows is not None:
        variational_handle = Line2D(
            [0],
            [0],
            color=VARIATIONAL_STYLE["color"],
            linestyle=VARIATIONAL_STYLE["linestyle"],
            marker=VARIATIONAL_STYLE["marker"],
            markerfacecolor="white",
            markeredgecolor=VARIATIONAL_STYLE["color"],
            linewidth=1.55,
            markersize=4.0,
        )
        legend_handles.insert(1, variational_handle)
        legend_labels.insert(1, "Variational MPO")
        maximum_runtime = max(float(row["cumulative_runtime_s"]) for row in variational_rows if int(row["step"]) > 0)
        if variational_runtime_censor is not None:
            maximum_runtime = max(
                maximum_runtime,
                float(variational_runtime_censor["runtime_lower_bound_s"]),
            )
        if maximum_runtime > RUNTIME_Y_LIMITS[1]:
            runtime_axes[0].set_ylim(RUNTIME_Y_LIMITS[0], maximum_runtime * 2.0)
    runtime_axes[-1].legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower right",
        ncol=1,
        frameon=False,
        fontsize=7.3,
        handlelength=1.8,
        handletextpad=0.6,
        labelspacing=0.35,
        borderaxespad=0.55,
    )
    figure.subplots_adjust(top=0.96, bottom=0.08, left=0.075, right=0.99)
    return figure


def caption(
    manifest: dict[str, object],
    runtime_manifest: dict[str, object],
    variational_manifest: dict[str, object] | None = None,
    variational_runtime_censor: dict[str, object] | None = None,
) -> str:
    """Return a manuscript-ready explanation of all three metric rows."""
    cases = manifest["cases"]
    endpoints = ", ".join(f"{CASE_LABELS[key]}: $n={int(cases[key]['stop_step'])}$" for key in CASE_ORDER)
    variational_note = ""
    if variational_manifest is not None:
        variational_note = (
            "Variational MPO is one complete one-thread observation without a warm-up rather than a repeated timing "
            "baseline. Its curves stop after the first completed step at which cumulative update time "
            "reaches $10^2$ s, or at the primary panel endpoint if that occurs first; their endpoints are "
            "computational censoring, not accuracy saturation. In one dimension every two-site gate is "
            "adjacent, so variational MPO reduces to and overlaps the common direct two-site update."
        )
    if variational_runtime_censor is not None:
        lower_bound = float(variational_runtime_censor["runtime_lower_bound_s"])
        step = int(variational_runtime_censor["plot_step"])
        variational_note += (
            f" The upward purple caret at $n={step}$ in panel (l) records only the conservative wall-time "
            f"lower bound $>{lower_bound:g}$ s for the runtime-censored, incomplete first step. "
            "No corresponding infidelity or parameter datum exists."
        )
    return (
        "Fixed-cap circuit accuracy, retained MPS size, and cumulative update runtime for four 16-site open "
        "systems at $\\chi_{\\max}=32$ and physical step size $\\Delta t=0.1$. Here $n=0$ denotes the initial "
        "MPS before any Trotter step. TDVP denotes gate-local two-site TDVP, and MPO denotes the routing-free "
        "MPO update. Rows (a)--(d) show normalized infidelity "
        "relative to dense execution of the identical ordered second-order Trotter circuit. Each column ends "
        "at the first Trotter step "
        "for which each of the three primary methods has maintained "
        f"$1-F>10^{{-2}}$ and varied by at most {SATURATION_LOG_RANGE_DECADES:g} decades over the trailing "
        f"{SATURATION_WINDOW_STEPS} Trotter steps; this "
        "local-flatness rule sets only the displayed time range: it neither alters the first-crossing reliability "
        "horizon nor establishes asymptotic long-time saturation. In the 1D panels, MPO and TEBD+SWAP "
        "coincide because every gate is nearest-neighbor and both baselines use the same adjacent-gate update. "
        "The shaded region and linear-scale inset show the final ten-sample window satisfying the local-flatness "
        "criterion; the inset vertical range is chosen separately for each circuit. "
        "The dotted line marks the reliability tolerance $\\epsilon=10^{-2}$, and values at or below $10^{-13}$ are "
        "shown at that plotting floor. Rows (e)--(h) show the total number of retained MPS tensor entries "
        "$P$ after each complete "
        "step; these are not transient working-storage peaks. The dotted line marks the maximal open-boundary "
        "rank profile allowed by $\\chi_{\\max}=32$ for $N=16$, with $P=15\\,016$. The logarithmic insets "
        "expand the complete early-time growth transient and share the same vertical scale. The main curves "
        "continue through "
        "the common accuracy endpoint. Rows (i)--(l) show median "
        "cumulative MPS-update time over "
        f"{int(runtime_manifest['repeats'])} isolated repeats after one full warm-up; bands span the minimum "
        "and maximum. Measurements use one thread on an Intel Core i5-13600KF. Timings include only MPS "
        "evolution under the ordered circuit gates and exclude schedule "
        "construction, initialization, "
        "dense evolution, fidelity and resource diagnostics, and plotting. They are fixed-cap costs, not a "
        "matched-accuracy efficiency comparison. Runtime curves should therefore be compared within a column; "
        "the columns contain different gate counts and endpoint step counts. TDVP uses "
        "$n_{\\mathrm{sub}}=2$; the direct methods use $n_{\\mathrm{sub}}=1$. "
        f"{variational_note} "
        f"Panel endpoints are {endpoints}."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--timing-dir", type=Path)
    parser.add_argument("--variational-dir", type=Path)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "figures",
    )
    args = parser.parse_args(argv)
    timing_dir = args.timing_dir or args.output_dir / TIMING_DIRNAME
    rows = _read_rows(args.output_dir / "trajectory_rows.csv")
    manifest = _read_manifest(args.output_dir / "manifest.json")
    runtime_rows = _read_rows(timing_dir / "timing_summary.csv")
    runtime_manifest = _read_manifest(timing_dir / "manifest.json")
    variational_dir = args.variational_dir or args.output_dir / VARIATIONAL_DIRNAME
    variational_rows_path = variational_dir / "trajectory_rows.csv"
    variational_manifest_path = variational_dir / "manifest.json"
    variational_censor_path = variational_dir / VARIATIONAL_CENSOR_FILENAME
    if variational_rows_path.is_file() != variational_manifest_path.is_file():
        msg = "Variational control rows and manifest must either both exist or both be absent."
        raise RuntimeError(msg)
    variational_rows = _read_rows(variational_rows_path) if variational_rows_path.is_file() else None
    variational_manifest = _read_manifest(variational_manifest_path) if variational_manifest_path.is_file() else None
    variational_runtime_censor = (
        _read_manifest(variational_censor_path)
        if variational_censor_path.is_file()
        else None
    )
    figure = create_figure(
        rows,
        manifest,
        runtime_rows,
        runtime_manifest,
        variational_rows,
        variational_manifest,
        variational_runtime_censor,
    )
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figures_dir / f"{FIGURE_STEM}.pdf", dpi=DPI)
    figure.savefig(args.figures_dir / f"{FIGURE_STEM}.png", dpi=DPI)
    plt.close(figure)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{FIGURE_STEM}_caption.md").write_text(
        caption(
            manifest,
            runtime_manifest,
            variational_manifest,
            variational_runtime_censor,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
