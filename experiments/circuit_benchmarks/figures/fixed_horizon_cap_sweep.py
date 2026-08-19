# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Plot fixed-horizon accuracy, MPS size, and runtime against bond cap.

This pure plotting step reads
``output/fixed_horizon_refinement/combined_cap_sweep.csv``,
``cap_timing_summary.csv``, and the variational-MPO cap-control summary.
By default, the validated TDVP-only Krylov-tolerance control replaces the
original TDVP values in memory; the frozen sweep tables are never modified.
The three primary methods use controlled timing subsets with three repetitions;
the variational method uses one complete observed trajectory per cap.  No
interpolation, fitting, or simulation is performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from experiments.circuit_benchmarks.config import (
    KRYLOV_TOL,
    METHODS,
    OUTPUT_DIR,
    RELIABILITY_THRESHOLD,
    REPO_ROOT,
)
from experiments.circuit_benchmarks.plotting import METHOD_STYLES, apply_style
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
from matplotlib.transforms import ScaledTranslation

if TYPE_CHECKING:
    from collections.abc import Sequence

TARGET_STEP = 15
REFINEMENT_DIR_NAME = "fixed_horizon_refinement"
SWEEP_FILENAME = "combined_cap_sweep.csv"
TIMING_FILENAME = "cap_timing_summary.csv"
VARIATIONAL_SUMMARY_FILENAME = "comparison_summary.json"
VARIATIONAL_CONTROL_DIR_NAME = "variational_mpo_control"
KRYLOV_CONTROL_DIR_NAME = "krylov_tolerance_control"
KRYLOV_SUMMARY_FILENAME = "summary.csv"
KRYLOV_MANIFEST_FILENAME = "manifest.json"
KRYLOV_CAMPAIGN_ID = "circuit_tdvp_krylov_tolerance_control_v1"
KRYLOV_OVERLAY_TOLERANCE = 1e-5
KRYLOV_OVERLAY_CAPS = (4, 8, 16, 24, 26, 28, 30, 32)
TDVP_METHOD = "gate_local_2tdvp"
CONTROL_CASE = "ising_2d"
CONTROL_N_SUB = 2
CONTROL_SVD_THRESHOLD = 1e-13
CONTROL_GATE_MODE = "tdvp"
VARIATIONAL_CAPS = (4, 8, 16)
VARIATIONAL_TIMING_REPEATS = 1
EXPECTED_VARIATIONAL_FITS = 270
FIGURE_STEM = "figure_circuit_fixed_horizon_cap_sweep"
FIGURE_WIDTH_MM = 86.0
FIGURE_HEIGHT_MM = 110.0
MM_TO_IN = 1.0 / 25.4
DPI = 600
TIMING_REPEATS = 3
RING_GID = "first-passing-cap"
PARAMETER_CURVE_GID = "exact-parameter-guide"
SHARED_MARKER_GID = "shared-cap-offset-marker"
VARIATIONAL_POINT_GID = "variational-control-point"
SHARED_MARKER_OFFSET_PT = 2.0

METHOD_LABELS = {
    "gate_local_2tdvp": "Projection",
    "mpo_contract_compress": "Direct MPO",
    "tebd_swap": "TEBD+SWAP",
}
VARIATIONAL_LABEL = "Variational MPO"
VARIATIONAL_STYLE = {
    "color": "#CC79A7",
    "marker": "D",
    "linestyle": ":",
}


@dataclass(frozen=True)
class CapSweepPoint:
    """Accuracy, peak MPS size, and repeated timing at one cap."""

    method: str
    chi_max: int
    max_infidelity: float
    peak_parameters: int
    runtime_median_s: float | None
    runtime_min_s: float | None
    runtime_max_s: float | None

    @property
    def reliable(self) -> bool:
        """Return whether the complete prefix meets the accuracy tolerance."""
        return self.max_infidelity <= RELIABILITY_THRESHOLD


@dataclass(frozen=True)
class VariationalControlPoint:
    """One complete-circuit variational-MPO cap-control point."""

    chi_max: int
    max_infidelity: float
    peak_parameters: int
    runtime_s: float
    fits: int
    maximum_sweeps: int


@dataclass(frozen=True)
class TdvpKrylovOverlay:
    """Validated TDVP cap sweep at one Krylov stopping tolerance."""

    points: tuple[CapSweepPoint, ...]
    tolerance: float


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read one required CSV table."""
    if not path.is_file():
        msg = f"Missing required cap-sweep input {path}."
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_fields(rows: Sequence[Mapping[str, str]], fields: Sequence[str], *, table: str) -> None:
    """Fail clearly if a required table is empty or incomplete."""
    if not rows:
        msg = f"{table} is empty."
        raise RuntimeError(msg)
    missing = [field for field in fields if field not in rows[0]]
    if missing:
        msg = f"{table} is missing required fields: {', '.join(missing)}."
        raise ValueError(msg)


def _number(row: Mapping[str, str], field: str, *, context: str) -> float:
    """Return one finite numeric CSV value."""
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        msg = f"Invalid {field!r} in {context}."
        raise ValueError(msg) from error
    if not math.isfinite(value):
        msg = f"Nonfinite {field!r} in {context}."
        raise ValueError(msg)
    return value


def _integer(row: Mapping[str, str], field: str, *, context: str) -> int:
    """Return one integer-valued CSV field without rounding."""
    value = _number(row, field, context=context)
    integer = int(value)
    if value != integer:
        msg = f"Noninteger {field!r} in {context}."
        raise ValueError(msg)
    return integer


def _serialized_true(value: object) -> bool:
    """Return whether an optional serialized flag is true."""
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_tdvp_krylov_overlay(
    summary_rows: Sequence[Mapping[str, str]],
    manifest: Mapping[str, object],
    *,
    expected_caps: Sequence[int] = KRYLOV_OVERLAY_CAPS,
    expected_tolerance: float = KRYLOV_OVERLAY_TOLERANCE,
) -> TdvpKrylovOverlay:
    """Validate the production TDVP-only Krylov control for Figure 4.

    The control directory may contain exploratory tolerances.  Only the full
    eight-cap production slice is selected, while every row is required to
    belong to the same TDVP-only campaign and fixed numerical protocol.
    """
    _require_fields(
        summary_rows,
        (
            "campaign_id",
            "case",
            "method",
            "chi_max",
            "n_sub",
            "target_step",
            "krylov_tolerance",
            "svd_threshold",
            "max_infidelity_through",
            "peak_parameter_count",
            "median_runtime_s",
            "min_runtime_s",
            "max_runtime_s",
            "timing_repeats",
        ),
        table=KRYLOV_SUMMARY_FILENAME,
    )

    if not _serialized_true(manifest.get("complete")):
        msg = "The Krylov-tolerance control manifest is not complete."
        raise RuntimeError(msg)
    expected_manifest = {
        "campaign_id": KRYLOV_CAMPAIGN_ID,
        "case": CONTROL_CASE,
        "method": TDVP_METHOD,
        "gate_mode": CONTROL_GATE_MODE,
        "n_sub": CONTROL_N_SUB,
        "target_step": TARGET_STEP,
        "timing_repeats": TIMING_REPEATS,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            msg = f"Krylov control manifest {field!r} is {manifest.get(field)!r}, expected {expected!r}."
            raise ValueError(msg)
    try:
        manifest_svd_threshold = float(manifest["svd_threshold"])
        summary_row_count = int(manifest["row_counts"]["summary"])  # type: ignore[index]
        thread_count = int(manifest["hardware"]["threads"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as error:
        msg = "The Krylov control manifest is missing fixed-protocol metadata."
        raise ValueError(msg) from error
    if manifest_svd_threshold != CONTROL_SVD_THRESHOLD:
        msg = f"Krylov control SVD threshold is {manifest_svd_threshold:g}, expected {CONTROL_SVD_THRESHOLD:g}."
        raise ValueError(msg)
    if summary_row_count != len(summary_rows):
        msg = f"Krylov manifest records {summary_row_count} summary rows, found {len(summary_rows)}."
        raise RuntimeError(msg)
    if thread_count != 1:
        msg = f"Krylov control must use one numerical thread, found {thread_count}."
        raise RuntimeError(msg)

    requested = manifest.get("requested_points")
    if not isinstance(requested, list):
        msg = "The Krylov control manifest has no requested-point grid."
        raise ValueError(msg)
    requested_caps: list[int] = []
    for item in requested:
        if not isinstance(item, Mapping):
            msg = "The Krylov control manifest contains an invalid requested point."
            raise ValueError(msg)
        try:
            tolerance = float(item["krylov_tolerance"])
            cap = int(item["chi_max"])
        except (KeyError, TypeError, ValueError) as error:
            msg = "The Krylov control manifest contains an invalid requested point."
            raise ValueError(msg) from error
        if tolerance == expected_tolerance:
            requested_caps.append(cap)

    normalized_caps = tuple(int(cap) for cap in expected_caps)
    if tuple(sorted(requested_caps)) != tuple(sorted(normalized_caps)):
        msg = (
            f"Krylov manifest must request the complete tau={expected_tolerance:g} cap grid "
            f"{list(normalized_caps)}, found {sorted(requested_caps)}."
        )
        raise RuntimeError(msg)

    selected: dict[int, CapSweepPoint] = {}
    for row in summary_rows:
        context = f"{KRYLOV_SUMMARY_FILENAME}/row"
        if row["campaign_id"] != KRYLOV_CAMPAIGN_ID:
            msg = f"Unexpected campaign {row['campaign_id']!r} in {KRYLOV_SUMMARY_FILENAME}."
            raise ValueError(msg)
        if row["case"] != CONTROL_CASE or row["method"] != TDVP_METHOD:
            msg = "The Krylov control summary must contain only the 4x4 Ising TDVP method."
            raise ValueError(msg)
        cap = _integer(row, "chi_max", context=context)
        n_sub = _integer(row, "n_sub", context=f"{context}/chi{cap}")
        target_step = _integer(row, "target_step", context=f"{context}/chi{cap}")
        repeats = _integer(row, "timing_repeats", context=f"{context}/chi{cap}")
        tolerance = _number(row, "krylov_tolerance", context=f"{context}/chi{cap}")
        svd_threshold = _number(row, "svd_threshold", context=f"{context}/chi{cap}")
        if (
            n_sub != CONTROL_N_SUB
            or target_step != TARGET_STEP
            or repeats != TIMING_REPEATS
            or svd_threshold != CONTROL_SVD_THRESHOLD
        ):
            msg = f"Krylov control row chi{cap} does not match the fixed Figure 4 protocol."
            raise ValueError(msg)
        if tolerance != expected_tolerance:
            continue

        error = _number(row, "max_infidelity_through", context=f"{context}/chi{cap}")
        parameters = _integer(row, "peak_parameter_count", context=f"{context}/chi{cap}")
        median = _number(row, "median_runtime_s", context=f"{context}/chi{cap}")
        low = _number(row, "min_runtime_s", context=f"{context}/chi{cap}")
        high = _number(row, "max_runtime_s", context=f"{context}/chi{cap}")
        if cap < 1 or error <= 0.0 or parameters < 1 or not 0.0 < low <= median <= high:
            msg = f"Invalid Krylov overlay point at chi{cap}."
            raise ValueError(msg)
        if cap in selected:
            msg = f"Duplicate Krylov overlay point at chi{cap}."
            raise RuntimeError(msg)
        selected[cap] = CapSweepPoint(TDVP_METHOD, cap, error, parameters, median, low, high)

    if tuple(sorted(selected)) != tuple(sorted(normalized_caps)):
        msg = (
            f"Krylov summary must contain the complete tau={expected_tolerance:g} cap grid "
            f"{list(normalized_caps)}, found {sorted(selected)}."
        )
        raise RuntimeError(msg)
    overlay_points = tuple(selected[cap] for cap in sorted(selected))
    if not any(point.reliable for point in overlay_points):
        msg = f"No tau={expected_tolerance:g} TDVP cap meets the accuracy threshold."
        raise RuntimeError(msg)
    return TdvpKrylovOverlay(overlay_points, expected_tolerance)


def load_tdvp_krylov_overlay(summary_path: Path, manifest_path: Path) -> TdvpKrylovOverlay:
    """Load and authenticate the compact Krylov control artifacts."""
    if not manifest_path.is_file():
        msg = f"Missing required Krylov control manifest {manifest_path}."
        raise FileNotFoundError(msg)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        msg = "The Krylov control manifest must contain a JSON object."
        raise ValueError(msg)
    try:
        expected_digest = str(payload["output_sha256"]["summary"])  # type: ignore[index]
    except (KeyError, TypeError) as error:
        msg = "The Krylov control manifest has no summary digest."
        raise ValueError(msg) from error
    actual_digest = _sha256(summary_path)
    if actual_digest != expected_digest:
        msg = "The Krylov control summary does not match its manifest digest."
        raise RuntimeError(msg)
    return prepare_tdvp_krylov_overlay(_read_rows(summary_path), payload)


def prepare_cap_sweep_data(
    sweep_rows: Sequence[Mapping[str, str]],
    timing_rows: Sequence[Mapping[str, str]],
    *,
    target_step: int = TARGET_STEP,
) -> list[CapSweepPoint]:
    """Validate and join fixed-horizon accuracy and cap timing tables."""
    _require_fields(
        sweep_rows,
        (
            "method",
            "chi_max",
            "target_step",
            "max_infidelity_through",
            "peak_parameter_count",
        ),
        table=SWEEP_FILENAME,
    )
    _require_fields(
        timing_rows,
        ("method", "chi_max", "target_step", "median_s", "min_s", "max_s", "repeats"),
        table=TIMING_FILENAME,
    )

    accuracy: dict[tuple[str, int], tuple[float, int, bool | None]] = {}
    for row in sweep_rows:
        if _integer(row, "target_step", context=SWEEP_FILENAME) != target_step:
            continue
        method = row["method"]
        if method not in METHODS:
            msg = f"Unknown method {method!r} in {SWEEP_FILENAME}."
            raise ValueError(msg)
        cap = _integer(row, "chi_max", context=f"{SWEEP_FILENAME}/{method}")
        error = _number(row, "max_infidelity_through", context=f"{SWEEP_FILENAME}/{method}/chi{cap}")
        parameters = _integer(
            row,
            "peak_parameter_count",
            context=f"{SWEEP_FILENAME}/{method}/chi{cap}",
        )
        if cap < 1 or error <= 0.0 or parameters < 1:
            msg = f"Invalid accuracy point for {method}/chi{cap}."
            raise ValueError(msg)
        key = (method, cap)
        if key in accuracy:
            msg = f"Duplicate accuracy point for {method}/chi{cap}."
            raise RuntimeError(msg)
        selected = _serialized_true(row["selected"]) if "selected" in row else None
        accuracy[key] = (error, parameters, selected)

    timings: dict[tuple[str, int], tuple[float, float, float]] = {}
    for row in timing_rows:
        if _integer(row, "target_step", context=TIMING_FILENAME) != target_step:
            continue
        method = row["method"]
        if method not in METHODS:
            msg = f"Unknown method {method!r} in {TIMING_FILENAME}."
            raise ValueError(msg)
        cap = _integer(row, "chi_max", context=f"{TIMING_FILENAME}/{method}")
        repeats = _integer(row, "repeats", context=f"{TIMING_FILENAME}/{method}/chi{cap}")
        median = _number(row, "median_s", context=f"{TIMING_FILENAME}/{method}/chi{cap}")
        low = _number(row, "min_s", context=f"{TIMING_FILENAME}/{method}/chi{cap}")
        high = _number(row, "max_s", context=f"{TIMING_FILENAME}/{method}/chi{cap}")
        if repeats != TIMING_REPEATS:
            msg = f"Expected {TIMING_REPEATS} timing repeats for {method}/chi{cap}, found {repeats}."
            raise RuntimeError(msg)
        if not 0.0 < low <= median <= high:
            msg = f"Invalid timing range for {method}/chi{cap}."
            raise RuntimeError(msg)
        key = (method, cap)
        if key in timings:
            msg = f"Duplicate timing point for {method}/chi{cap}."
            raise RuntimeError(msg)
        timings[key] = (median, low, high)

    if not accuracy:
        msg = f"No n={target_step} rows in {SWEEP_FILENAME}."
        raise RuntimeError(msg)
    extra_timings = sorted(set(timings) - set(accuracy))
    if extra_timings:
        msg = f"Timing caps are absent from the accuracy sweep: {extra_timings}."
        raise RuntimeError(msg)

    points = [
        CapSweepPoint(
            method,
            cap,
            error,
            parameters,
            *(timings[(method, cap)] if (method, cap) in timings else (None, None, None)),
        )
        for (method, cap), (error, parameters, _selected) in accuracy.items()
    ]
    points.sort(key=lambda point: (METHODS.index(point.method), point.chi_max))

    for method in METHODS:
        method_points = [point for point in points if point.method == method]
        if len(method_points) < 2:
            msg = f"Need at least two cap points for {method}."
            raise RuntimeError(msg)
        first_passing = next((point for point in method_points if point.reliable), None)
        if first_passing is None:
            msg = f"No tested cap meets the accuracy tolerance for {method}."
            raise RuntimeError(msg)
        timed_points = [point for point in method_points if point.runtime_median_s is not None]
        if first_passing.runtime_median_s is None:
            msg = f"First passing cap chi{first_passing.chi_max} is not timed for {method}."
            raise RuntimeError(msg)
        if len(timed_points) < 2:
            msg = f"Need at least two controlled timing points for {method}."
            raise RuntimeError(msg)
        selected_flags = [
            cap
            for (flag_method, cap), (_error, _parameters, selected) in accuracy.items()
            if flag_method == method and selected is True
        ]
        if selected_flags and selected_flags != [first_passing.chi_max]:
            msg = (
                f"Selected flag for {method} identifies {selected_flags}; first passing cap is {first_passing.chi_max}."
            )
            raise RuntimeError(msg)
    return points


def prepare_variational_controls(
    payload: Mapping[str, object],
    *,
    target_step: int = TARGET_STEP,
    caps: Sequence[int] = VARIATIONAL_CAPS,
) -> list[VariationalControlPoint]:
    """Validate the fallback-free variational circuit cap controls."""
    if str(payload.get("case", "")) != "ising_2d":
        msg = "The variational control must use the 4x4 Ising case."
        raise ValueError(msg)
    try:
        payload_target = int(payload["target_step"])
    except (KeyError, TypeError, ValueError) as error:
        msg = "The variational control has no valid target_step."
        raise ValueError(msg) from error
    if payload_target != target_step:
        msg = f"Variational target step {payload_target} does not match {target_step}."
        raise ValueError(msg)

    requested_caps = tuple(int(cap) for cap in caps)
    if len(set(requested_caps)) != len(requested_caps) or any(cap < 1 for cap in requested_caps):
        msg = "Variational caps must be distinct positive integers."
        raise ValueError(msg)
    try:
        timing_repeats = int(payload["timing_repeats_per_cap"])
    except (KeyError, TypeError, ValueError) as exception:
        msg = "The variational control has no valid timing-repeat count."
        raise ValueError(msg) from exception
    if timing_repeats != VARIATIONAL_TIMING_REPEATS:
        msg = f"Expected one variational timing observation per cap, found {timing_repeats}."
        raise RuntimeError(msg)
    thread_metadata = payload.get("thread_metadata")
    if not isinstance(thread_metadata, Mapping) or int(thread_metadata.get("threads", -1)) != 1:
        msg = "The variational control does not record one-thread execution."
        raise RuntimeError(msg)

    cap_results = payload.get("caps")
    if not isinstance(cap_results, Mapping):
        msg = "The variational summary has no cap controls."
        raise ValueError(msg)

    points: list[VariationalControlPoint] = []
    for cap in requested_caps:
        cap_payload = cap_results.get(str(cap))
        if not isinstance(cap_payload, Mapping):
            msg = f"The variational summary has no chi_max={cap} control."
            raise ValueError(msg)
        if not _serialized_true(cap_payload.get("all_selected_fits_converged")):
            msg = f"The plotted variational control at chi_max={cap} contains a nonconverged fit."
            raise RuntimeError(msg)

        method_payload = cap_payload.get("variational_mpo")
        if not isinstance(method_payload, Mapping):
            msg = f"The chi_max={cap} variational control has no variational_mpo result."
            raise ValueError(msg)
        try:
            error = float(method_payload["worst_prefix_infidelity"])
            parameters_float = float(method_payload["peak_parameter_count"])
            runtime = float(method_payload["runtime_s"])
            fits_float = float(cap_payload["variational_fits"])
            sweeps_float = float(cap_payload["maximum_sweeps"])
        except (KeyError, TypeError, ValueError) as exception:
            msg = f"The chi_max={cap} variational control contains an invalid plotted value."
            raise ValueError(msg) from exception
        parameters = int(parameters_float)
        fits = int(fits_float)
        sweeps = int(sweeps_float)
        if (
            not all(math.isfinite(value) for value in (error, parameters_float, runtime, fits_float, sweeps_float))
            or error <= 0.0
            or parameters_float != parameters
            or parameters < 1
            or runtime <= 0.0
            or fits_float != fits
            or fits != EXPECTED_VARIATIONAL_FITS
            or sweeps_float != sweeps
            or sweeps < 1
        ):
            msg = f"The chi_max={cap} variational control contains an invalid plotted value."
            raise ValueError(msg)
        points.append(VariationalControlPoint(cap, error, parameters, runtime, fits, sweeps))
    return points


def first_passing_caps(points: Sequence[CapSweepPoint]) -> dict[str, int]:
    """Return the smallest tested reliable cap for every method."""
    result: dict[str, int] = {}
    for method in METHODS:
        method_points = sorted(
            (point for point in points if point.method == method),
            key=lambda point: point.chi_max,
        )
        passing = next((point for point in method_points if point.reliable), None)
        if passing is None:
            msg = f"No first-passing point for {method}."
            raise RuntimeError(msg)
        result[method] = passing.chi_max
    return result


def apply_tdvp_krylov_overlay(
    points: Sequence[CapSweepPoint],
    overlay: TdvpKrylovOverlay,
) -> list[CapSweepPoint]:
    """Replace only TDVP values while preserving the frozen comparator rows."""
    base_tdvp = sorted(
        (point for point in points if point.method == TDVP_METHOD),
        key=lambda point: point.chi_max,
    )
    replacement = {point.chi_max: point for point in overlay.points}
    base_caps = [point.chi_max for point in base_tdvp]
    if base_caps != sorted(replacement):
        msg = f"TDVP overlay caps {sorted(replacement)} do not match frozen TDVP caps {base_caps}."
        raise RuntimeError(msg)
    old_selected = next((point.chi_max for point in base_tdvp if point.reliable), None)
    new_selected = next((point.chi_max for point in overlay.points if point.reliable), None)
    if old_selected is None or new_selected != old_selected:
        msg = f"Krylov overlay changes the TDVP first-passing cap from {old_selected} to {new_selected}."
        raise RuntimeError(msg)

    combined = [replacement[point.chi_max] if point.method == TDVP_METHOD else point for point in points]
    if [point for point in combined if point.method != TDVP_METHOD] != [
        point for point in points if point.method != TDVP_METHOD
    ]:
        msg = "Krylov overlay modified a frozen comparator point."
        raise RuntimeError(msg)
    return combined


def _configure_style() -> None:
    """Apply compact, colorblind-safe manuscript styling."""
    apply_style()
    mpl.rcParams.update(
        {
            "font.size": 7.8,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.4,
            "lines.linewidth": 1.65,
            "lines.markersize": 4.6,
        }
    )


def _style_axis(axis: plt.Axes) -> None:
    """Apply the common Physical Review axis treatment."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(which="both", direction="out", width=0.7)
    axis.grid(axis="y", which="major", color="#E6E8EB", linewidth=0.45, zorder=0)


def _legend_handles(*, include_variational: bool = False) -> list[Line2D]:
    """Return method handles in manuscript order."""
    handles = [
        Line2D(
            [0],
            [0],
            label=METHOD_LABELS[method],
            color=METHOD_STYLES[method]["color"],
            linestyle=METHOD_STYLES[method]["linestyle"],
            marker=METHOD_STYLES[method]["marker"],
            markerfacecolor=("white" if method == "tebd_swap" else METHOD_STYLES[method]["color"]),
            markeredgecolor=METHOD_STYLES[method]["color"],
            markeredgewidth=0.9,
        )
        for method in METHODS
    ]
    if include_variational:
        handles.insert(
            1,
            Line2D(
                [0],
                [0],
                label=VARIATIONAL_LABEL,
                color=VARIATIONAL_STYLE["color"],
                linestyle=VARIATIONAL_STYLE["linestyle"],
                marker=VARIATIONAL_STYLE["marker"],
                markerfacecolor=VARIATIONAL_STYLE["color"],
                markeredgecolor=VARIATIONAL_STYLE["color"],
                markeredgewidth=0.9,
            ),
        )
    return handles


def _add_selection_ring(axis: plt.Axes, x: float, y: float) -> None:
    """Ring one first-passing raw marker without obscuring its method style."""
    ring = axis.scatter(
        [x],
        [y],
        s=42,
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        zorder=9,
    )
    ring.set_gid(RING_GID)


def create_figure(
    points: Sequence[CapSweepPoint],
    variational: Sequence[VariationalControlPoint] = (),
) -> plt.Figure:
    """Create the single-column accuracy, MPS-size, and runtime cap sweep."""
    _configure_style()
    figure, (accuracy_axis, parameter_axis, runtime_axis) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(FIGURE_WIDTH_MM * MM_TO_IN, FIGURE_HEIGHT_MM * MM_TO_IN),
        gridspec_kw={"height_ratios": (1.05, 1.0, 0.95)},
    )
    selected_caps = first_passing_caps(points)
    tdvp_parameters = {point.chi_max: point.peak_parameters for point in points if point.method == "gate_local_2tdvp"}
    tebd_parameters = {point.chi_max: point.peak_parameters for point in points if point.method == "tebd_swap"}
    shared_parameter_caps = sorted(
        cap for cap in set(tdvp_parameters) & set(tebd_parameters) if tdvp_parameters[cap] == tebd_parameters[cap]
    )
    if not shared_parameter_caps:
        msg = "Panel (b) requires at least one coincident TDVP/TEBD+SWAP cap."
        raise RuntimeError(msg)

    for method_index, method in enumerate(METHODS):
        method_points = sorted(
            (point for point in points if point.method == method),
            key=lambda point: point.chi_max,
        )
        timed_points = [point for point in method_points if point.runtime_median_s is not None]
        caps = np.asarray([point.chi_max for point in method_points], dtype=float)
        errors = np.asarray([point.max_infidelity for point in method_points])
        parameters = np.asarray([point.peak_parameters for point in method_points], dtype=float)
        timed_caps = np.asarray([point.chi_max for point in timed_points], dtype=float)
        medians = np.asarray([float(point.runtime_median_s) for point in timed_points])
        lows = np.asarray([float(point.runtime_min_s) for point in timed_points])
        highs = np.asarray([float(point.runtime_max_s) for point in timed_points])
        style = METHOD_STYLES[method]
        marker_face = "white" if method == "tebd_swap" else style["color"]
        parameter_marker_size = 5.0 if method == "tebd_swap" else 4.6

        accuracy_axis.plot(
            caps,
            errors,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor=marker_face,
            markeredgecolor=style["color"],
            markeredgewidth=0.9,
            linewidth=1.65,
            alpha=1.0,
            zorder=3 + method_index,
        )
        parameter_curve = parameter_axis.plot(
            caps,
            parameters,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.65,
            alpha=1.0,
            zorder=3 + method_index,
        )[0]
        parameter_curve.set_gid(PARAMETER_CURVE_GID)

        shared_mask = (
            np.isin(caps, shared_parameter_caps)
            if method
            in {
                "gate_local_2tdvp",
                "tebd_swap",
            }
            else np.zeros(caps.shape, dtype=bool)
        )
        if np.any(~shared_mask):
            parameter_axis.plot(
                caps[~shared_mask],
                parameters[~shared_mask],
                linestyle="none",
                marker=style["marker"],
                markersize=parameter_marker_size,
                markerfacecolor=marker_face,
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
                zorder=6 + method_index,
            )
        if np.any(shared_mask):
            direction = -1.0 if method == "gate_local_2tdvp" else 1.0
            display_offset = ScaledTranslation(
                direction * SHARED_MARKER_OFFSET_PT / 72.0,
                0.0,
                figure.dpi_scale_trans,
            )
            shared_markers = parameter_axis.plot(
                caps[shared_mask],
                parameters[shared_mask],
                linestyle="none",
                marker=style["marker"],
                markersize=parameter_marker_size,
                markerfacecolor=marker_face,
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
                transform=parameter_axis.transData + display_offset,
                zorder=6 + method_index,
            )[0]
            shared_markers.set_gid(SHARED_MARKER_GID)
        runtime_axis.errorbar(
            timed_caps,
            medians,
            yerr=np.vstack((medians - lows, highs - medians)),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor=marker_face,
            markeredgecolor=style["color"],
            markeredgewidth=0.9,
            linewidth=1.65,
            alpha=1.0,
            elinewidth=1.0,
            capsize=1.8,
            capthick=1.0,
            zorder=3 + method_index,
        )

        selected = next(point for point in method_points if point.chi_max == selected_caps[method])
        _add_selection_ring(accuracy_axis, selected.chi_max, selected.max_infidelity)
        _add_selection_ring(parameter_axis, selected.chi_max, selected.peak_parameters)
        if selected.runtime_median_s is None:
            msg = f"First passing cap is not timed for {method}."
            raise RuntimeError(msg)
        _add_selection_ring(runtime_axis, selected.chi_max, selected.runtime_median_s)

    variational_points = tuple(sorted(variational, key=lambda point: point.chi_max))
    if variational_points:
        variational_caps = np.asarray([point.chi_max for point in variational_points], dtype=float)
        for axis, values in (
            (accuracy_axis, [point.max_infidelity for point in variational_points]),
            (parameter_axis, [point.peak_parameters for point in variational_points]),
            (runtime_axis, [point.runtime_s for point in variational_points]),
        ):
            control = axis.plot(
                variational_caps,
                values,
                color=VARIATIONAL_STYLE["color"],
                linestyle=VARIATIONAL_STYLE["linestyle"],
                marker=VARIATIONAL_STYLE["marker"],
                markerfacecolor=VARIATIONAL_STYLE["color"],
                markeredgecolor=VARIATIONAL_STYLE["color"],
                markeredgewidth=0.9,
                markersize=5.0,
                linewidth=1.65,
                zorder=8,
            )[0]
            control.set_gid(VARIATIONAL_POINT_GID)

    accuracy_axis.axhline(
        RELIABILITY_THRESHOLD,
        color="0.35",
        linestyle=":",
        linewidth=0.9,
        zorder=1,
    )
    accuracy_axis.text(
        0.02,
        RELIABILITY_THRESHOLD,
        r"$\epsilon=10^{-2}$",
        transform=accuracy_axis.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="0.3",
    )

    all_caps = np.asarray([point.chi_max for point in points], dtype=float)
    all_errors = np.asarray([point.max_infidelity for point in points])
    all_parameters = np.asarray([point.peak_parameters for point in points], dtype=float)
    all_lows = np.asarray([float(point.runtime_min_s) for point in points if point.runtime_min_s is not None])
    all_highs = np.asarray([float(point.runtime_max_s) for point in points if point.runtime_max_s is not None])
    if variational_points:
        all_caps = np.append(all_caps, [point.chi_max for point in variational_points])
        all_errors = np.append(all_errors, [point.max_infidelity for point in variational_points])
        all_parameters = np.append(all_parameters, [point.peak_parameters for point in variational_points])
        all_lows = np.append(all_lows, [point.runtime_s for point in variational_points])
        all_highs = np.append(all_highs, [point.runtime_s for point in variational_points])
    for axis in (accuracy_axis, parameter_axis, runtime_axis):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        _style_axis(axis)
    accuracy_axis.set_ylim(all_errors.min() / 1.7, min(1.25, all_errors.max() * 1.7))
    parameter_axis.set_ylim(all_parameters.min() / 1.45, all_parameters.max() * 1.45)
    runtime_axis.set_ylim(all_lows.min() / 1.5, all_highs.max() * 1.55)
    runtime_axis.set_xlim(all_caps.min() / 1.18, all_caps.max() * 1.18)

    maximum_power = int(math.ceil(math.log2(all_caps.max())))
    minimum_power = int(math.floor(math.log2(all_caps.min())))
    power_ticks = [2**power for power in range(minimum_power, maximum_power + 1)]
    largest_tested_cap = int(all_caps.max())
    if largest_tested_cap not in power_ticks:
        power_ticks.append(largest_tested_cap)
        power_ticks.sort()
    runtime_axis.xaxis.set_major_locator(FixedLocator(power_ticks))
    runtime_axis.xaxis.set_major_formatter(ScalarFormatter())
    runtime_axis.xaxis.set_minor_formatter(NullFormatter())
    accuracy_axis.set_ylabel("Infidelity")
    parameter_axis.set_ylabel(r"Peak MPS coefficients $P_{\max}$")
    runtime_axis.set_ylabel("Runtime (s)")
    runtime_axis.set_xlabel(r"$\chi_{\max}$")
    parameter_axis.legend(
        handles=_legend_handles(include_variational=bool(variational_points)),
        loc="lower right",
        ncols=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.90,
        handlelength=1.45,
        handletextpad=0.45,
        labelspacing=0.20,
        borderaxespad=0.45,
    )

    for axis, label in (
        (accuracy_axis, "(a)"),
        (parameter_axis, "(b)"),
        (runtime_axis, "(c)"),
    ):
        axis.text(
            0.035,
            1.035,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=8.8,
        )
    figure.subplots_adjust(top=0.985, bottom=0.10, left=0.225, right=0.985, hspace=0.23)
    return figure


def _latex_power_of_ten(value: float) -> str:
    """Format one validated power of ten for a LaTeX caption."""
    exponent = int(round(math.log10(value)))
    if not math.isclose(value, 10.0**exponent, rel_tol=1e-12, abs_tol=0.0):
        msg = f"Expected a power-of-ten tolerance, found {value:g}."
        raise ValueError(msg)
    return f"10^{{{exponent}}}"


def caption(
    points: Sequence[CapSweepPoint],
    *,
    tdvp_krylov_tolerance: float,
) -> str:
    """Return the concise manuscript-ready caption from the plotted data."""
    selected = first_passing_caps(points)
    tolerance = _latex_power_of_ten(tdvp_krylov_tolerance)
    return (
        "\\textbf{Fixed-horizon cap sweep.} For the $4\\times4$ Ising circuit through "
        "$n_\\star=15$, (a) the worst prefix infidelity "
        "$E_\\star=\\max_{k\\leq n_\\star}(1-F_k)$, (b) the peak MPS coefficient "
        "count $P_{\\max}$ through $n_\\star=15$, and (c) runtime are shown "
        "against the configured bond-dimension cap. For Direct MPO, $P_{\\max}$ includes the "
        "uncompressed MPO--MPS target; temporary working arrays are excluded. Runtime "
        "markers for Projection, Direct MPO, and TEBD+SWAP are medians of three one-thread "
        "repetitions at every timed cap, with bars spanning their full range. Variational-MPO "
        "diamonds at $\\chi_{\\max}=4,8,$ and $16$ are one complete one-thread run per cap, "
        "without timing repeats; their $P_{\\max}$ values include the largest uncompressed target MPS. "
        "The dotted curve shows only the observed cap dependence and is not a fitted scaling law. "
        f"Projection applies adjacent gates directly and uses gate-local two-site TDVP at tolerance ${tolerance}$ "
        "only for separated gates; the direct-method "
        "series retain the original cap-sweep data. Black rings mark "
        f"the first caps satisfying $E_\\star\\leq10^{{-2}}$: "
        f"$\\chi_{{\\max}}={selected[TDVP_METHOD]}$, "
        f"{selected['mpo_contract_compress']}, and {selected['tebd_swap']} for "
        "Projection, Direct MPO, and TEBD+SWAP, respectively. Coincident Projection and TEBD+SWAP markers at shared "
        "caps in (b) are offset horizontally in display space only for visibility; their guide "
        "curves remain at identical data coordinates. Thin lines only guide the eye between raw "
        "tested caps and are not fits."
    )


def _save_outputs(
    figure: plt.Figure,
    figures_dir: Path,
    input_dir: Path,
    points: Sequence[CapSweepPoint],
    *,
    tdvp_krylov_tolerance: float,
) -> None:
    """Write one canonical figure and keep its caption beside the source data."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures_dir / f"{FIGURE_STEM}.pdf")
    figure.savefig(figures_dir / f"{FIGURE_STEM}.png", dpi=DPI)
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / f"{FIGURE_STEM}_caption.md").write_text(
        caption(points, tdvp_krylov_tolerance=tdvp_krylov_tolerance) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    """Load the fixed-horizon cap tables and render the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=OUTPUT_DIR / REFINEMENT_DIR_NAME,
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "figures",
    )
    parser.add_argument(
        "--variational-summary",
        type=Path,
        default=OUTPUT_DIR / VARIATIONAL_CONTROL_DIR_NAME / VARIATIONAL_SUMMARY_FILENAME,
    )
    parser.add_argument(
        "--krylov-summary",
        type=Path,
        default=OUTPUT_DIR / KRYLOV_CONTROL_DIR_NAME / KRYLOV_SUMMARY_FILENAME,
    )
    parser.add_argument(
        "--krylov-manifest",
        type=Path,
        default=OUTPUT_DIR / KRYLOV_CONTROL_DIR_NAME / KRYLOV_MANIFEST_FILENAME,
    )
    parser.add_argument(
        "--no-krylov-overlay",
        action="store_true",
        help="Render the frozen TDVP cap sweep instead of the validated production control.",
    )
    args = parser.parse_args(argv)

    points = prepare_cap_sweep_data(
        _read_rows(args.input_dir / SWEEP_FILENAME),
        _read_rows(args.input_dir / TIMING_FILENAME),
    )
    tdvp_krylov_tolerance = KRYLOV_TOL
    if not args.no_krylov_overlay:
        overlay = load_tdvp_krylov_overlay(args.krylov_summary, args.krylov_manifest)
        points = apply_tdvp_krylov_overlay(points, overlay)
        tdvp_krylov_tolerance = overlay.tolerance
    variational = prepare_variational_controls(json.loads(args.variational_summary.read_text(encoding="utf-8")))
    figure = create_figure(points, variational)
    _save_outputs(
        figure,
        args.figures_dir,
        args.input_dir,
        points,
        tdvp_krylov_tolerance=tdvp_krylov_tolerance,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
