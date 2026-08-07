# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused tests for the fixed-horizon cap-sweep figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from experiments.circuit_benchmarks.config import METHODS
from experiments.circuit_benchmarks.figures.fixed_horizon_cap_sweep import (
    FIGURE_HEIGHT_MM,
    FIGURE_WIDTH_MM,
    PARAMETER_CURVE_GID,
    RING_GID,
    SHARED_MARKER_GID,
    VARIATIONAL_POINT_GID,
    caption,
    create_figure,
    first_passing_caps,
    prepare_cap_sweep_data,
    prepare_variational_controls,
)


def _inputs() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    sweep: list[dict[str, str]] = []
    timing: list[dict[str, str]] = []
    caps = (4, 8, 16, 24, 26, 28, 30, 32)
    errors = {
        METHODS[0]: (0.25, 0.10, 0.03, 0.012, 0.011, 0.0095, 0.0083, 0.0074),
        METHODS[1]: (0.16, 0.089, 0.027, 0.0104, 0.0086, 0.0073, 0.0063, 0.0055),
        METHODS[2]: (0.20, 0.10, 0.037, 0.017, 0.0145, 0.0123, 0.0108, 0.0094),
    }
    selected_caps = dict(zip(METHODS, (28, 26, 32), strict=True))
    for method_index, method in enumerate(METHODS):
        for cap_index, (cap, error) in enumerate(zip(caps, errors[method], strict=True)):
            sweep.append({
                "method": method,
                "chi_max": str(cap),
                "target_step": "15",
                "max_infidelity_through": str(error),
                "peak_parameter_count": str(
                    (cap_index + 1) * 1000 * (2 if method == METHODS[1] else 1)
                ),
                "selected": str(cap == selected_caps[method]),
            })
            median = float((method_index + 1) * (cap_index + 1))
            timing.append({
                "method": method,
                "chi_max": str(cap),
                "target_step": "15",
                "median_s": str(median),
                "min_s": str(median * 0.95),
                "max_s": str(median * 1.05),
                "repeats": "3",
            })
    return sweep, timing


def _variational_payload() -> dict[str, object]:
    errors = {4: 0.60, 8: 0.35, 16: 0.18}
    parameters = {4: 1200, 8: 4200, 16: 14500}
    runtimes = {4: 4.0, 8: 35.0, 16: 420.0}
    return {
        "case": "ising_2d",
        "target_step": 15,
        "timing_repeats_per_cap": 1,
        "thread_metadata": {"threads": 1},
        "caps": {
            str(cap): {
                "all_selected_fits_converged": True,
                "variational_fits": 270,
                "maximum_sweeps": 8,
                "variational_mpo": {
                    "worst_prefix_infidelity": errors[cap],
                    "peak_parameter_count": parameters[cap],
                    "runtime_s": runtimes[cap],
                },
            }
            for cap in (4, 8, 16)
        },
    }


def test_prepare_cap_sweep_data_joins_complete_method_cap_grid() -> None:
    """Accuracy and timings should join one-to-one in manuscript order."""
    points = prepare_cap_sweep_data(*_inputs())

    assert len(points) == 8 * len(METHODS)
    assert [point.method for point in points[::8]] == list(METHODS)
    assert first_passing_caps(points) == dict(zip(METHODS, (28, 26, 32), strict=True))
    assert [point.peak_parameters for point in points[:3]] == [1000, 2000, 3000]
    assert sum(point.runtime_median_s is not None for point in points) == 8 * len(METHODS)
    assert all(
        point.runtime_min_s <= point.runtime_median_s <= point.runtime_max_s
        for point in points
        if point.runtime_median_s is not None and point.runtime_min_s is not None and point.runtime_max_s is not None
    )


def test_prepare_cap_sweep_data_rejects_untimed_selected_cap() -> None:
    """The first passing accuracy point must also appear in the runtime panel."""
    sweep, timing = _inputs()
    timing[:] = [row for row in timing if not (row["method"] == METHODS[-1] and row["chi_max"] == "32")]

    with pytest.raises(RuntimeError, match="First passing cap chi32 is not timed"):
        prepare_cap_sweep_data(sweep, timing)


def test_prepare_cap_sweep_data_requires_three_repeats() -> None:
    """Every min--max bar should summarize the declared three-run protocol."""
    sweep, timing = _inputs()
    timing[0]["repeats"] = "2"

    with pytest.raises(RuntimeError, match="Expected 3 timing repeats"):
        prepare_cap_sweep_data(sweep, timing)


def test_prepare_cap_sweep_data_requires_positive_integer_parameters() -> None:
    """Peak MPS tensor counts must be positive integers."""
    sweep, timing = _inputs()
    sweep[0]["peak_parameter_count"] = "0"

    with pytest.raises(ValueError, match="Invalid accuracy point"):
        prepare_cap_sweep_data(sweep, timing)

    sweep, timing = _inputs()
    sweep[0]["peak_parameter_count"] = "10.5"
    with pytest.raises(ValueError, match="Noninteger 'peak_parameter_count'"):
        prepare_cap_sweep_data(sweep, timing)


def test_prepare_cap_sweep_data_validates_selected_flag() -> None:
    """Campaign selection metadata must mark the first passing cap."""
    sweep, timing = _inputs()
    sweep[4]["selected"] = "True"
    sweep[5]["selected"] = "False"

    with pytest.raises(RuntimeError, match="first passing cap is 28"):
        prepare_cap_sweep_data(sweep, timing)


def test_prepare_variational_controls_require_complete_converged_cap_grid() -> None:
    """Only the complete, converged, one-thread cap grid may enter the figure."""
    payload = _variational_payload()
    points = prepare_variational_controls(payload)
    assert [point.chi_max for point in points] == [4, 8, 16]
    assert points[-1].max_infidelity == pytest.approx(0.18)
    assert points[-1].peak_parameters == 14500
    assert points[-1].runtime_s == pytest.approx(420.0)
    assert all(point.fits == 270 for point in points)
    assert all(point.maximum_sweeps == 8 for point in points)

    payload["caps"]["16"]["all_selected_fits_converged"] = False
    with pytest.raises(RuntimeError, match=r"chi_max=16.*nonconverged fit"):
        prepare_variational_controls(payload)

    payload = _variational_payload()
    del payload["caps"]["8"]
    with pytest.raises(ValueError, match="no chi_max=8 control"):
        prepare_variational_controls(payload)

    payload = _variational_payload()
    payload["thread_metadata"]["threads"] = 20
    with pytest.raises(RuntimeError, match="one-thread execution"):
        prepare_variational_controls(payload)


def test_cap_sweep_figure_dimensions_scales_labels_and_rings() -> None:
    """The output should be a compact shared-x, log-log three-panel figure."""
    points = prepare_cap_sweep_data(*_inputs())
    figure = create_figure(points, prepare_variational_controls(_variational_payload()))
    axes = figure.axes
    size_mm = tuple(value * 25.4 for value in figure.get_size_inches())
    ring_counts = [len(axis.findobj(lambda artist: artist.get_gid() == RING_GID)) for axis in axes]
    legend = axes[1].get_legend()
    legend_labels = [] if legend is None else [text.get_text() for text in legend.get_texts()]
    exact_parameter_curves = axes[1].findobj(lambda artist: artist.get_gid() == PARAMETER_CURVE_GID)
    offset_markers = axes[1].findobj(lambda artist: artist.get_gid() == SHARED_MARKER_GID)
    parameter_ring_x = [float(ring.get_offsets()[0, 0]) for ring in axes[1].collections if ring.get_gid() == RING_GID]
    offset_marker_sizes = [marker.get_markersize() for marker in offset_markers]
    tebd_parameter_face = offset_markers[-1].get_markerfacecolor()
    variational_curves = [axis.findobj(lambda artist: artist.get_gid() == VARIATIONAL_POINT_GID) for axis in axes]
    variational_counts = [len(curves) for curves in variational_curves]
    variational_x = [list(curves[0].get_xdata()) for curves in variational_curves]
    variational_linestyles = [curves[0].get_linestyle() for curves in variational_curves]
    plt.close(figure)

    assert size_mm == pytest.approx((FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM))
    assert len(axes) == 3
    assert all(axis.get_xscale() == "log" for axis in axes)
    assert all(axis.get_yscale() == "log" for axis in axes)
    assert not axes[0].get_xlabel()
    assert not axes[1].get_xlabel()
    assert axes[2].get_xlabel() == r"$\chi_{\max}$"
    assert axes[0].get_ylabel() == "Infidelity"
    assert axes[1].get_ylabel() == r"Peak MPS parameters $P_{\max}$"
    assert axes[2].get_ylabel() == "Runtime (s)"
    assert axes[0].get_legend() is None
    assert axes[2].get_legend() is None
    assert ring_counts == [len(METHODS), len(METHODS), len(METHODS)]
    assert len(exact_parameter_curves) == len(METHODS)
    expected_caps = [4.0, 8.0, 16.0, 24.0, 26.0, 28.0, 30.0, 32.0]
    assert all(list(curve.get_xdata()) == expected_caps for curve in exact_parameter_curves)
    assert len(offset_markers) == 2
    assert all(list(marker.get_xdata()) == expected_caps for marker in offset_markers)
    assert offset_marker_sizes[-1] > offset_marker_sizes[0]
    assert tebd_parameter_face == "white"
    assert parameter_ring_x == [28.0, 26.0, 32.0]
    assert variational_counts == [1, 1, 1]
    assert variational_x == [[4.0, 8.0, 16.0]] * 3
    assert variational_linestyles == [":", ":", ":"]
    assert legend_labels == ["TDVP", "Variational MPO", "MPO", "TEBD+SWAP"]


def test_cap_sweep_caption_is_concise_and_explicit() -> None:
    """The caption should identify raw guides, selection, and timing scope."""
    text = caption()
    assert text.startswith("\\textbf{Fixed-horizon cap sweep.}")
    assert "three one-thread" in text
    assert "at every timed cap" in text
    assert "one complete one-thread run per cap" in text
    assert "without timing repeats" in text
    assert "not a fitted scaling law" in text
    assert "largest uncompressed target MPS" in text
    assert "maximum observed MPS tensor" in text
    assert "precompression MPO--MPS intermediate" in text
    assert "temporary working arrays are excluded" in text
    assert "offset horizontally in display space only for visibility" in text
    assert "guide curves remain at identical data coordinates" in text
    assert "$\\chi_{\\max}=28$, 26, and 32" in text
    assert "Black rings" in text
    assert "not fits" in text
