# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regenerate paper Figure 4 with the updated response-matrix definition."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import mqt.yaqs as yaqs_package

from .constants import (
    CUTS,
    DEFAULT_DATA_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPECTRA_NPZ,
    DT_DEFAULT,
    G_DEFAULT,
    K_DEFAULT,
    L_DEFAULT,
    SPECTRUM_CUT,
    SPECTRUM_DISCARDED_WEIGHT_THRESHOLD,
    SPECTRUM_JS,
)
from .data import (
    EntropyPoint,
    SpectrumCurve,
    build_full_probe_set,
    build_process_tensor,
    load_entropy_table,
    load_spectrum_curves,
    process_tensor_spectrum,
    response_spectrum,
    save_spectrum_curves,
    spectrum_curve,
    write_entropy_table,
)
from .layouts import PRX_SINGLE_COLUMN_LAYOUT
from .plot import _panel_a_ylim, build_figure, save_figure

LOGGER = logging.getLogger(__name__)


def _j_grid(*, maximum: float, step: float) -> list[float]:
    if maximum < 0.0 or step <= 0.0:
        message = f"Expected maximum >= 0 and step > 0, got maximum={maximum}, step={step}."
        raise ValueError(message)
    count = round(maximum / step)
    if not np.isclose(count * step, maximum, rtol=0.0, atol=1e-12):
        message = f"maximum={maximum} must be an integer multiple of step={step}."
        raise ValueError(message)
    return [round(index * step, 12) for index in range(count + 1)]


def _normalized_j(j_value: float) -> float:
    return round(float(j_value), 12)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        return subprocess.check_output(
            [git, "-C", str(path), "rev-parse", "HEAD"],  # noqa: S603
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _generate_campaign(
    panel_a_js: list[float],
) -> tuple[list[EntropyPoint], list[SpectrumCurve], dict[str, Any]]:
    probe_sets = {cut: build_full_probe_set(cut) for cut in CUTS}
    panel_a_keys = {_normalized_j(j_value) for j_value in panel_a_js}
    spectrum_keys = {_normalized_j(j_value) for j_value in SPECTRUM_JS}
    all_js = sorted(panel_a_keys | spectrum_keys)
    points: list[EntropyPoint] = []
    selected: dict[tuple[float, str], np.ndarray] = {}
    timings: dict[str, float] = {}

    for j_value in all_js:
        started = time.perf_counter()
        LOGGER.info("J=%g: reconstructing and evaluating", j_value)
        process = build_process_tensor(j_value)
        response_cache: dict[int, tuple[np.ndarray, float]] = {}
        mpo_cache: dict[int, tuple[np.ndarray, float]] = {}

        cuts = CUTS if j_value in panel_a_keys else (SPECTRUM_CUT,)
        for cut in cuts:
            response_cache[cut] = response_spectrum(process, probe_sets[cut])
            mpo_cache[cut] = process_tensor_spectrum(process, cut=cut)
            if j_value in panel_a_keys:
                points.append(
                    EntropyPoint(
                        cut=cut,
                        j=j_value,
                        s_v=response_cache[cut][1],
                        s_mpo=mpo_cache[cut][1],
                    )
                )

        if j_value in spectrum_keys:
            if SPECTRUM_CUT not in response_cache:
                response_cache[SPECTRUM_CUT] = response_spectrum(process, probe_sets[SPECTRUM_CUT])
                mpo_cache[SPECTRUM_CUT] = process_tensor_spectrum(process, cut=SPECTRUM_CUT)
            selected[j_value, "S_V"] = response_cache[SPECTRUM_CUT][0]
            selected[j_value, "S_MPO"] = mpo_cache[SPECTRUM_CUT][0]

        elapsed = time.perf_counter() - started
        timings[f"J={j_value:g}"] = elapsed
        LOGGER.info("J=%g complete in %.2f s", j_value, elapsed)

    curves = [
        spectrum_curve(
            quantity=quantity,
            cut=SPECTRUM_CUT,
            j_value=j_value,
            singular_values=selected[_normalized_j(j_value), quantity],
        )
        for j_value in SPECTRUM_JS
        for quantity in ("S_V", "S_MPO")
    ]
    return points, curves, {"per_coupling_seconds": timings, "total_seconds": float(sum(timings.values()))}


def _point_lookup(points: list[EntropyPoint], *, cut: int, j_value: float) -> EntropyPoint:
    for point in points:
        if point.cut == cut and np.isclose(point.j, j_value, rtol=0.0, atol=1e-12):
            return point
    message = f"Missing entropy point at cut={cut}, J={j_value}."
    raise KeyError(message)


def _qualitative_checks(
    points: list[EntropyPoint],
    curves: list[SpectrumCurve],
) -> dict[str, Any]:
    finite = all(np.isfinite((point.s_v, point.s_mpo)).all() for point in points)
    nonnegative = all(point.s_v >= 0.0 and point.s_mpo >= 0.0 for point in points)
    baseline = max(
        max(_point_lookup(points, cut=cut, j_value=0.0).s_v, _point_lookup(points, cut=cut, j_value=0.0).s_mpo)
        for cut in CUTS
    )
    moderate_growth = {
        str(cut): bool(
            _point_lookup(points, cut=cut, j_value=2.0).s_v > _point_lookup(points, cut=cut, j_value=0.2).s_v
        )
        for cut in CUTS
    }
    peak_summary: dict[str, dict[str, float]] = {}
    for cut in CUTS:
        subset = [point for point in points if point.cut == cut]
        response_peak = max(subset, key=lambda point: point.s_v)
        mpo_peak = max(subset, key=lambda point: point.s_mpo)
        peak_summary[str(cut)] = {
            "response_J": response_peak.j,
            "response_entropy": response_peak.s_v,
            "mpo_J": mpo_peak.j,
            "mpo_entropy": mpo_peak.s_mpo,
        }

    p1: dict[str, dict[str, float]] = {"S_V": {}, "S_MPO": {}}
    for curve in curves:
        p1[curve.quantity][f"{curve.j:g}"] = float(curve.weights[0])
    response_tail_growth = (1.0 - p1["S_V"]["4"]) - (1.0 - p1["S_V"]["0.1"])
    mpo_tail_growth = (1.0 - p1["S_MPO"]["4"]) - (1.0 - p1["S_MPO"]["0.1"])

    return {
        "finite": finite,
        "nonnegative": nonnegative,
        "J0_max_entropy": baseline,
        "J0_rank_one_baseline": bool(baseline < 1e-10),
        "response_grows_from_J0p2_to_J2": moderate_growth,
        "peak_by_cut": peak_summary,
        "selected_spectrum_leading_weights": p1,
        "response_tail_growth_J0p1_to_J4": response_tail_growth,
        "mpo_tail_growth_J0p1_to_J4": mpo_tail_growth,
        "response_redistribution_stronger_than_mpo": bool(response_tail_growth > mpo_tail_growth),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifest(
    output_dir: Path,
    *,
    panel_a_js: list[float],
    timing: dict[str, Any],
    checks: dict[str, Any],
    artifacts: list[Path],
) -> None:
    package_file = Path(yaqs_package.__file__).resolve()
    implementation_root = package_file.parents[3]
    experiment_root = Path(__file__).resolve().parents[3]
    manifest = {
        "figure": 4,
        "description": "Updated uncentered IXYZ response entropy versus causal-block PT-MPO entropy",
        "configuration": {
            "L": L_DEFAULT,
            "k": K_DEFAULT,
            "dt": DT_DEFAULT,
            "g": G_DEFAULT,
            "cuts": list(CUTS),
            "panel_a_J": panel_a_js,
            "spectrum_cut": SPECTRUM_CUT,
            "spectrum_J": list(SPECTRUM_JS),
            "probe_basis": "full tetrahedral",
            "response_orientation": "future_rows_history_columns",
            "response_channels": "IXYZ",
            "centered": False,
            "weight_scope": "complete_retained_record",
            "spectrum_discarded_weight_threshold": SPECTRUM_DISCARDED_WEIGHT_THRESHOLD,
        },
        "implementation": {
            "package_file": str(package_file.relative_to(implementation_root)),
            "git_commit": _git_commit(implementation_root),
        },
        "experiment_git_commit_before_results": _git_commit(experiment_root),
        "timing": timing,
        "qualitative_checks": checks,
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in artifacts},
    }
    _write_json(output_dir / "run_manifest.json", manifest)


def main() -> None:
    """Run the full numerical campaign or redraw its stored results."""
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-csv", type=Path, default=None)
    parser.add_argument("--spectra-npz", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--j-max", type=float, default=6.0)
    parser.add_argument("--j-step", type=float, default=0.2)
    parser.add_argument("--png-dpi", type=int, default=PRX_SINGLE_COLUMN_LAYOUT.png_dpi)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_csv = (args.data_csv or output_dir / DEFAULT_DATA_CSV.name).resolve()
    spectra_npz = (args.spectra_npz or output_dir / DEFAULT_SPECTRA_NPZ.name).resolve()

    if args.plot_only:
        points = load_entropy_table(data_csv)
        curves = load_spectrum_curves(spectra_npz)
        timing: dict[str, Any] = {"plot_only": True}
    else:
        panel_a_js = _j_grid(maximum=float(args.j_max), step=float(args.j_step))
        points, curves, timing = _generate_campaign(panel_a_js)
        write_entropy_table(data_csv, points)
        save_spectrum_curves(spectra_npz, curves)
        checks = _qualitative_checks(points, curves)
        _write_json(output_dir / "qualitative_checks.json", checks)

    figure = build_figure(points, curves, layout=PRX_SINGLE_COLUMN_LAYOUT)
    pdf_path, png_path = save_figure(
        figure,
        output_dir,
        stem=PRX_SINGLE_COLUMN_LAYOUT.output_stem,
        dpi=int(args.png_dpi),
        pad_inches=PRX_SINGLE_COLUMN_LAYOUT.savefig_pad_inches,
    )
    plt.close(figure)
    y_low, y_high = _panel_a_ylim(points)
    LOGGER.info("Panel (a) y-limits: [%.3e, %.3e] nats", y_low, y_high)
    LOGGER.info("Wrote %s", pdf_path)
    LOGGER.info("Wrote %s", png_path)

    if not args.plot_only:
        checks = _qualitative_checks(points, curves)
        artifacts = [data_csv, spectra_npz, output_dir / "qualitative_checks.json", pdf_path, png_path]
        _write_manifest(
            output_dir,
            panel_a_js=_j_grid(maximum=float(args.j_max), step=float(args.j_step)),
            timing=timing,
            checks=checks,
            artifacts=artifacts,
        )


if __name__ == "__main__":
    main()
