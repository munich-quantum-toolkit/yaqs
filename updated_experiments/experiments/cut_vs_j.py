#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

r"""Raw IXYZ response entropy :math:`S_V` vs coupling ``J`` and causal cut.

Fixed setup:
- L=6 sites (Ising chain)
- k=20 instrument slots
- dt=0.1
- g=1.0
- J sweep: 0.0, 0.05, ..., 2.0

For each cut, the public YAQS characterizer constructs the uncentered matrix
:math:`V_{(j,\\alpha),i}=p_{ij}f_{ij,\\alpha}` in ``(I,X,Y,Z)`` order, with complete
retained-record probabilities. The default single initial state is :math:`|0\\rangle^{\\otimes L}`.
For comparison, this script also reports entropies for the historical raw-XYZ and centered-XYZ
matrices reconstructed from the same run and evaluated with the same numerical threshold.

Outputs include a **three-panel PRL-style figure**: (1) heatmap of :math:`S_V` (:math:`c` vs :math:`J`, log colors
on a logarithmic scale with values below :math:`10^{-3}` clipped to the scale floor); (2) :math:`S_V` vs :math:`J`
for **representative cuts** ``PANEL2_FIXED_CUTS``; (3) :math:`S_V` vs :math:`c` for **representative couplings**
``PANEL3_TARGET_JS`` with nearest available :math:`J` from the sweep. Regenerate from ``summary.csv`` via
``--plot-heatmap-only`` (optional ``--summary-csv PATH``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from common import (
    DT_DEFAULT,
    G_DEFAULT,
    HEATMAP_VMAX,
    HEATMAP_VMIN,
    J_SWEEP,
    K_DEFAULT,
    L_DEFAULT,
    PANEL3_JS,
    characterize,
    characterizer,
    configure_matplotlib,
    configure_matplotlib_prl,
    initial_states_sys_env0,
    ising_chain,
    load_csv,
    sample_cut_probes,
    save_figure,
    sim_params,
    write_csv,
)

from mqt.yaqs.characterization.memory.operational_memory.response_matrix import compute_spectrum

if TYPE_CHECKING:
    from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet

SPECTRUM_THRESHOLD = 1e-12
BASIS = "IXYZ"
ORIENTATION = "future_rows_history_columns"
WEIGHT_SCOPE = "complete_retained_record"
LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pasts", type=int, default=32)
    p.add_argument("--n-futures", type=int, default=32)
    p.add_argument(
        "--cuts",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
        help='Comma-separated cuts c to run (1..20), e.g. "1,4,7,10,13,16,20".',
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        dest="n_seeds",
        help=(
            "Number of initial states: env fixed to |0>, system (site 0) random for n_seeds>1; "
            "n_seeds=1 uses |0...0>. Scalar metrics are averaged across seeds."
        ),
    )
    p.add_argument(
        "--j-values",
        type=str,
        default="",
        help="Optional comma-separated J sweep (default: full 0..2 step 0.05).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_entropy_vs_j_by_cut_results"))
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument("--max-workers", type=int, default=8, help="Maximum worker processes when parallel.")
    p.add_argument("--save-raw", action="store_true", help="Save per-point matrices and per-cut probe arrays.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible partial run from <out-dir>/summary.csv.",
    )
    p.add_argument(
        "--unitary-ensemble",
        type=str,
        default="haar",
        choices=("haar", "clifford"),
        help="Non-break random unitary ensemble for unitary_break_mp probes.",
    )
    p.add_argument(
        "--plot-heatmap-only",
        action="store_true",
        dest="plot_heatmap_only",
        help="Only build the cut×J entropy heatmap from --summary-csv (no simulations).",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Input summary for --plot-heatmap-only (default: <out-dir>/summary.csv).",
    )
    return p.parse_args()


def _parse_int_list(spec: str) -> list[int]:
    vals = [int(tok.strip()) for tok in spec.split(",") if tok.strip()]
    if not vals:
        msg = "cuts must contain at least one integer."
        raise ValueError(msg)
    uniq = sorted(set(vals))
    for c in uniq:
        if not (1 <= c <= K_DEFAULT):
            msg = f"cut must satisfy 1 <= cut <= {K_DEFAULT}, got {c}."
            raise ValueError(msg)
    return uniq


def _git_output(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["/usr/bin/git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        shell=False,
        text=True,
    )
    return proc.stdout.strip()


def _write_json_atomic(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_checkpoint(out_dir: Path, rows: list[dict[str, float | int | str]]) -> None:
    ordered = sorted(rows, key=lambda row: (int(row["cut"]), float(row["J"])))
    csv_tmp = out_dir / ".summary.csv.tmp"
    write_csv(csv_tmp, ordered)
    csv_tmp.replace(out_dir / "summary.csv")
    _write_json_atomic(out_dir / "summary.json", ordered)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _point_key(cut: int, j_value: float) -> tuple[int, float]:
    return int(cut), round(float(j_value), 12)


def _raw_path(out_dir: Path, cut: int, j_value: float) -> Path:
    j_tag = f"{float(j_value):.12g}".replace("-", "m").replace(".", "p")
    return out_dir / "raw" / f"cut_{int(cut):02d}_J_{j_tag}.npz"


def _reconstruct_and_compare(
    response_matrix: np.ndarray,
    *,
    n_pasts: int,
    n_futures: int,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    """Validate raw IXYZ layout and compute matched historical XYZ diagnostics."""
    response = np.asarray(response_matrix, dtype=np.float64)
    expected_shape = (4 * int(n_futures), int(n_pasts))
    if response.shape != expected_shape or not np.all(np.isfinite(response)):
        msg = f"invalid response matrix: expected finite {expected_shape}, got {response.shape}"
        raise AssertionError(msg)

    weighted_ixyz = response.reshape(n_futures, 4, n_pasts).transpose(2, 0, 1)
    weights = weighted_ixyz[:, :, 0]
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0) or np.any(weights > 1.0 + 1e-8):
        msg = "complete retained-record weights must be finite probabilities in (0, 1]"
        raise AssertionError(msg)
    if not np.allclose(weights, weights[:, :1], rtol=1e-10, atol=1e-12):
        msg = "unitary future controls must leave each history weight constant across futures"
        raise AssertionError(msg)

    pauli_ixyz = weighted_ixyz / weights[:, :, np.newaxis]
    if not np.allclose(pauli_ixyz[:, :, 0], 1.0, rtol=1e-10, atol=1e-12):
        msg = "reconstructed identity coefficients must equal one"
        raise AssertionError(msg)
    rebuilt = (pauli_ixyz * weights[:, :, np.newaxis]).transpose(1, 2, 0).reshape(expected_shape)
    if not np.allclose(rebuilt, response, rtol=1e-12, atol=1e-14):
        msg = "IXYZ response reconstruction does not match the public result"
        raise AssertionError(msg)

    singular_values = np.linalg.svd(response, compute_uv=False)
    singular_values_t = np.linalg.svd(response.T, compute_uv=False)
    if not np.allclose(singular_values, singular_values_t, rtol=1e-11, atol=1e-13):
        msg = "response spectrum changed under transpose"
        raise AssertionError(msg)

    raw_xyz = (pauli_ixyz[:, :, 1:] * weights[:, :, np.newaxis]).reshape(n_pasts, 3 * n_futures)
    centered_xyz = raw_xyz - raw_xyz.mean(axis=0, keepdims=True)
    raw_xyz_analysis = compute_spectrum(raw_xyz, discarded_weight_threshold=SPECTRUM_THRESHOLD)
    centered_xyz_analysis = compute_spectrum(centered_xyz, discarded_weight_threshold=SPECTRUM_THRESHOLD)
    current_analysis = compute_spectrum(response, discarded_weight_threshold=SPECTRUM_THRESHOLD)
    diagnostics: dict[str, float | int] = {
        "entropy": float(current_analysis["entropy"]),
        "entropy_old_raw_xyz": float(raw_xyz_analysis["entropy"]),
        "entropy_old_centered_xyz": float(centered_xyz_analysis["entropy"]),
        "rank": int(np.sum(singular_values > SPECTRUM_THRESHOLD)),
        "response_norm": float(np.linalg.norm(response)),
    }
    return diagnostics, pauli_ixyz, weights, singular_values


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def _save_probe_arrays(out_dir: Path, probe_set: ProbeSet, cut: int) -> Path:
    path = out_dir / "raw" / f"probes_cut_{int(cut):02d}.npz"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            cut=np.asarray(int(cut)),
            num_interventions=np.asarray(int(probe_set.num_interventions)),
            past_features=np.asarray(probe_set.past_features),
            future_features=np.asarray(probe_set.future_features),
            past_cut_meas=np.stack(probe_set.past_cut_meas),
            future_prep_cut=np.stack(probe_set.future_prep_cut),
        )
    temporary.replace(path)
    return path


def _save_raw_point(
    out_dir: Path,
    cut: int,
    j_value: float,
    responses: list[np.ndarray],
    pauli: list[np.ndarray],
    weights: list[np.ndarray],
    singular_values: list[np.ndarray],
) -> tuple[str, str]:
    path = _raw_path(out_dir, cut, j_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            response_matrix=np.stack(responses),
            pauli_ixyz_ij=np.stack(pauli),
            weights_ij=np.stack(weights),
            singular_values_full=np.stack(singular_values),
        )
    temporary.replace(path)
    return str(path.relative_to(out_dir)), _sha256(path)


def _validate_resumed_rows(
    rows: list[dict[str, str]],
    *,
    n_pasts: int,
    n_futures: int,
    n_seeds: int,
    seed: int,
    ensemble: str,
) -> None:
    for row in rows:
        compatible = (
            int(float(row["n_pasts"])) == n_pasts
            and int(float(row["n_futures"])) == n_futures
            and int(float(row["n_seeds"])) == n_seeds
            and int(float(row["seed"])) == seed
            and row["unitary_ensemble"] == ensemble
            and row["basis"] == BASIS
            and row["orientation"] == ORIENTATION
            and row["weight_scope"] == WEIGHT_SCOPE
        )
        if not compatible:
            msg = "Existing summary.csv is incompatible with the requested resume settings."
            raise ValueError(msg)


def _plot_entropy_vs_j(rows: list[dict[str, float | int | str]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if not rows:
        return
    cuts = sorted({int(r["cut"]) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    c_mid = K_DEFAULT // 2

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(cuts), vmax=max(cuts))

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.0), constrained_layout=True)

    for c in cuts:
        sub = sorted((r for r in rows if int(r["cut"]) == c), key=lambda r: float(r["J"]))
        xs = [float(r["J"]) for r in sub]
        ys = [float(r["entropy"]) for r in sub]
        color = cmap(norm(c))
        lw = 2.8 if c == c_mid else 1.4
        alpha = 1.0 if c == c_mid else 0.85
        label = f"c={c} (k/2 baseline)" if c == c_mid else None
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, label=label, zorder=3 if c == c_mid else 2)

    ax.set_xlabel(r"$J$")
    ax.set_ylabel(r"$S_V$ (linear $w_{ij}$)")
    ax.set_xticks(j_vals)
    ax.grid(True, axis="y")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.95)
    cbar.ax.set_ylabel("cut c")
    if any(c == c_mid for c in cuts):
        ax.legend(frameon=False, loc="best")

    save_figure(fig, out_dir / "fig_entropy_vs_j_by_cut")


def plot_entropy_heatmap_cut_vs_j(
    rows: list[dict[str, str | float | int]],
    out_stem: Path,
) -> None:
    """Hierarchy layout: dominant heatmap with two supporting cross-sections."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

    if not rows:
        return

    def _truncate_cmap(name: str, lo: float = 0.10, hi: float = 0.85, n: int = 256) -> LinearSegmentedColormap:
        base = plt.get_cmap(name)
        return LinearSegmentedColormap.from_list(
            f"{name}_trunc_{lo:.2f}_{hi:.2f}",
            base(np.linspace(lo, hi, n)),
        )

    cuts = sorted({int(float(r["cut"])) for r in rows})
    j_vals = sorted({float(r["J"]) for r in rows})
    z = np.full((len(cuts), len(j_vals)), np.nan, dtype=np.float64)
    for r in rows:
        ci = cuts.index(int(float(r["cut"])))
        ji = j_vals.index(float(r["J"]))
        z[ci, ji] = float(r["entropy"])

    j_arr = np.asarray(j_vals, dtype=np.float64)
    if j_arr.size >= 2:
        dj = float(np.median(np.diff(j_arr)))
        j_edges = np.concatenate([[j_arr[0] - dj / 2], (j_arr[:-1] + j_arr[1:]) / 2, [j_arr[-1] + dj / 2]])
    else:
        j_edges = np.array([j_arr[0] - 0.1, j_arr[0] + 0.1])

    c_arr = np.asarray(cuts, dtype=np.float64)
    if c_arr.size >= 2:
        dc = float(np.median(np.diff(c_arr)))
        c_edges = np.concatenate([[c_arr[0] - dc / 2], (c_arr[:-1] + c_arr[1:]) / 2, [c_arr[-1] + dc / 2]])
    else:
        c_edges = np.array([c_arr[0] - 0.5, c_arr[0] + 0.5])

    def nearest_j(target: float) -> float:
        return float(j_arr[int(np.argmin(np.abs(j_arr - float(target))))])

    configure_matplotlib_prl()
    fig = plt.figure(figsize=(8.2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.06, hspace=0.10)
    ax0 = fig.add_subplot(gs[:, 0])  # dominant heatmap
    ax1 = fig.add_subplot(gs[0, 1])  # S vs J
    ax2 = fig.add_subplot(gs[1, 1])  # S vs c
    ax0.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    # Show exact zeros explicitly in black via colormap "under" color.
    # Positive values remain on the log color scale [HEATMAP_COLOR_VMIN, HEATMAP_COLOR_VMAX].
    z_plot = np.where(
        np.isfinite(z),
        np.where(z <= 0.0, HEATMAP_VMIN * 0.1, np.maximum(z, HEATMAP_VMIN)),
        np.nan,
    )
    z_mesh = np.ma.masked_invalid(np.transpose(z_plot))

    norm = LogNorm(vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
    cmap = _truncate_cmap("magma", 0.08, 0.78).copy()
    cmap.set_under(color="black")
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))

    im = ax0.pcolormesh(
        c_edges,
        j_edges,
        z_mesh,
        cmap=cmap,
        norm=norm,
        shading="auto",
        linewidth=0,
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )
    ax0.set_xlabel(r"Causal cut $c$")
    ax0.set_ylabel(r"Coupling $J$")

    cbar = fig.colorbar(im, ax=ax0, shrink=0.92, pad=0.012, aspect=18)
    cbar.ax.set_title(r"$S_V$", fontsize=16, pad=3)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    cbar.ax.tick_params(length=3.0, width=0.7, labelsize=13)
    cbar.outline.set_linewidth(0.8)

    ax0.set_xlim(c_edges[0], c_edges[-1])
    ax0.set_ylim(j_edges[0], j_edges[-1])

    ax0.set_xticks([1, 5, 10, 15, 20])
    ax0.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax0.grid(False)

    for spine in ax0.spines.values():
        spine.set_linewidth(0.9)
    ax0.tick_params(which="major", direction="in", top=True, right=True, length=4.2, width=0.8, labelsize=15)
    ax0.xaxis.label.set_size(20)
    ax0.yaxis.label.set_size(20)

    # Representative slices.
    panel2_pref = [1, 5, 10, 15, 20]
    panel2_cuts = [c for c in panel2_pref if c in cuts]
    if len(panel2_cuts) < 4:
        panel2_all = [int(c) for c in cuts if 1 <= int(c) <= 20]
        if panel2_all:
            idx = np.linspace(0, len(panel2_all) - 1, min(5, len(panel2_all))).round().astype(int)
            panel2_cuts = [panel2_all[i] for i in idx]
    panel3_js = [nearest_j(v) for v in PANEL3_JS]

    # Heatmap slice guides.
    for c_sel in panel2_cuts:
        ax0.axvline(float(c_sel), color="white", lw=0.55, ls="--", alpha=0.10, zorder=3)
    guide_cmap = plt.get_cmap("Reds")
    guide_norm = Normalize(vmin=0.0, vmax=2.0)
    for j_sel in panel3_js:
        ax0.axhline(float(j_sel), color=guide_cmap(guide_norm(j_sel)), lw=0.55, ls="--", alpha=0.12, zorder=3)

    # Panel (b): S_V vs J for all cuts (continuous colormap).
    panel2_vals = sorted(panel2_cuts)
    panel2_cmap = _truncate_cmap("Blues", 0.35, 0.95)
    panel2_norm = Normalize(vmin=1.0, vmax=20.0)
    panel2_color_map = {c: panel2_cmap(panel2_norm(float(c))) for c in panel2_vals}
    for c_sel in panel2_vals:
        sub = sorted((r for r in rows if int(float(r["cut"])) == c_sel), key=lambda r: float(r["J"]))
        if not sub:
            continue
        xs = [float(r["J"]) for r in sub]
        ys = [max(float(r["entropy"]), HEATMAP_VMIN) for r in sub]
        ax1.semilogy(
            xs,
            ys,
            color=panel2_color_map[c_sel],
            lw=2.0,
            marker="o",
            ms=3.8,
            markeredgewidth=0.0,
            alpha=0.95,
            label=rf"$c={c_sel}$",
        )
    ax1.set_xlabel(r"Coupling $J$")
    ax1.set_ylabel(r"$S_V$")
    ax1.tick_params(which="major", direction="in", top=True, right=True, length=3.8, width=0.8, labelsize=12)
    ax1.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax1.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.grid(True, which="major", axis="y", alpha=0.12, linewidth=0.45)
    for s in ax1.spines.values():
        s.set_linewidth(0.9)
    ax1.legend(frameon=False, loc="lower right", fontsize=7.5, handlelength=1.5, borderaxespad=0.2, labelspacing=0.2)
    ax1.xaxis.label.set_size(17)
    ax1.yaxis.label.set_size(17)

    # Panel (c): S_V vs c for all couplings (raw discrete points only).
    xs_c = list(cuts)
    panel3_vals = sorted(panel3_js)
    panel3_cmap = _truncate_cmap("Reds", 0.30, 0.95)
    panel3_norm = Normalize(vmin=0.0, vmax=2.0)

    def _marker_for_j(jv: float) -> str:
        if abs(jv - 0.4) < 1e-9:
            return "o"
        if abs(jv - 1.0) < 1e-9:
            return "s"
        if abs(jv - 2.0) < 1e-9:
            return "^"
        return "o"

    for j_use in panel3_vals:
        ji = j_vals.index(j_use)
        ys_s = np.asarray([max(float(z[ci, ji]), HEATMAP_VMIN) for ci in range(len(cuts))], dtype=np.float64)
        col = panel3_cmap(panel3_norm(j_use))
        ax2.semilogy(
            xs_c,
            ys_s,
            ls="-",
            lw=1.9,
            marker=_marker_for_j(j_use),
            ms=5.0,
            markeredgewidth=0.0,
            alpha=0.92,
            color=col,
            label=rf"$J={j_use:g}$",
        )
    ax2.set_xlabel(r"Causal cut $c$")
    ax2.set_ylabel(r"$S_V$")
    ax2.tick_params(which="major", direction="in", top=True, right=True, length=3.8, width=0.8, labelsize=12)
    ax2.set_xticks([4, 8, 12, 16, 20])
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax2.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.grid(True, which="major", axis="y", alpha=0.12, linewidth=0.45)
    for s in ax2.spines.values():
        s.set_linewidth(0.9)
    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=1.25,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    ax2.xaxis.label.set_size(17)
    ax2.yaxis.label.set_size(17)

    # Match y-limits across side panels for fair comparison.
    y_floor = HEATMAP_VMIN
    y_vals_side: list[float] = []
    for c_sel in panel2_cuts:
        y_vals_side.extend(max(float(r["entropy"]), HEATMAP_VMIN) for r in rows if int(float(r["cut"])) == c_sel)
    for j_use in panel3_js:
        ji = j_vals.index(j_use)
        y_vals_side.extend([max(float(z[ci, ji]), HEATMAP_VMIN) for ci in range(len(cuts))])
    y_hi = min(1.0, max(y_floor * 1.2, float(np.nanmax(y_vals_side)) * 1.25)) if y_vals_side else 1.0
    ax1.set_ylim(y_floor, y_hi)
    ax2.set_ylim(y_floor, y_hi)

    for ax, tag in ((ax0, "(a)"), (ax1, "(b)"), (ax2, "(c)")):
        ax.text(
            0.04,
            0.955,
            tag,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            fontweight="bold",
        )

    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    prl_stem = out_stem.with_name(f"{out_stem.name}_prl")
    fig.savefig(prl_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(prl_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def main() -> None:
    """Run or resume the cut-resolved Figure 2 campaign."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.plot_heatmap_only):
        csv_path = Path(args.summary_csv) if args.summary_csv is not None else out_dir / "summary.csv"
        if not csv_path.is_file():
            msg = f"summary CSV not found: {csv_path}"
            raise FileNotFoundError(msg)
        rows_raw = load_csv(csv_path)
        plot_entropy_heatmap_cut_vs_j(rows_raw, out_dir / "fig_entropy_heatmap_cut_vs_J")
        return

    cuts = _parse_int_list(str(args.cuts))
    n_seeds = int(args.n_seeds)
    j_values = (
        [float(tok.strip()) for tok in str(args.j_values).split(",") if tok.strip()]
        if str(args.j_values).strip()
        else list(J_SWEEP)
    )
    if n_seeds < 1:
        msg = "n-seeds must be >= 1."
        raise ValueError(msg)
    if int(args.max_workers) < 1:
        msg = "max-workers must be >= 1."
        raise ValueError(msg)
    init_rng = np.random.default_rng(int(args.seed) + 77_777)
    initial_list = initial_states_sys_env0(length=L_DEFAULT, n_seeds=n_seeds, rng=init_rng)
    initial_states_path = out_dir / "initial_states.npy"
    np.save(initial_states_path, np.stack(initial_list, axis=0))

    mc = characterizer(parallel=bool(args.parallel), max_workers=int(args.max_workers))
    params = sim_params(dt=DT_DEFAULT)
    LOGGER.info(
        "Figure 2: L=%d, k=%d, dt=%g, g=%g, Np=%d, Nf=%d, seeds=%d, "
        "basis=%s, orientation=%s, weights=%s, centered=False, threshold=%g, workers=%d",
        L_DEFAULT,
        K_DEFAULT,
        DT_DEFAULT,
        G_DEFAULT,
        int(args.n_pasts),
        int(args.n_futures),
        n_seeds,
        BASIS,
        ORIENTATION,
        WEIGHT_SCOPE,
        SPECTRUM_THRESHOLD,
        int(args.max_workers),
    )

    summary_path = out_dir / "summary.csv"
    rows: list[dict[str, float | int | str]] = []
    if bool(args.resume) and summary_path.is_file():
        resumed = load_csv(summary_path)
        _validate_resumed_rows(
            resumed,
            n_pasts=int(args.n_pasts),
            n_futures=int(args.n_futures),
            n_seeds=n_seeds,
            seed=int(args.seed),
            ensemble=str(args.unitary_ensemble),
        )
        rows.extend(resumed)
        LOGGER.info("Resuming from %d completed parameter points", len(rows))
    completed = {_point_key(int(float(row["cut"])), float(row["J"])) for row in rows}

    repo = Path(__file__).resolve().parents[1]
    manifest: dict[str, object] = {
        "integration_commit": _git_output(repo, "rev-parse", "HEAD"),
        "response_matrix_update_commit": _git_output(repo, "rev-parse", "response-matrix-update"),
        "response_matrix_tests_commit": _git_output(repo, "rev-parse", "response-matrix-tests"),
        "git_status": _git_output(repo, "status", "--short"),
        "python": sys.version,
        "settings": {
            "L": L_DEFAULT,
            "k": K_DEFAULT,
            "dt": DT_DEFAULT,
            "g": G_DEFAULT,
            "cuts": cuts,
            "J_values": j_values,
            "n_pasts": int(args.n_pasts),
            "n_futures": int(args.n_futures),
            "n_seeds": n_seeds,
            "seed": int(args.seed),
            "unitary_ensemble": str(args.unitary_ensemble),
            "parallel": bool(args.parallel),
            "max_workers": int(args.max_workers),
            "save_raw": bool(args.save_raw),
        },
        "response_matrix_contract": {
            "basis": BASIS,
            "orientation": ORIENTATION,
            "weight_scope": WEIGHT_SCOPE,
            "centered": False,
            "probability_power": 1,
            "spectrum_discarded_weight_threshold": SPECTRUM_THRESHOLD,
        },
        "initial_states_sha256": _sha256(initial_states_path),
        "completed_points": len(rows),
    }
    _write_json_atomic(out_dir / "run_manifest.json", manifest)

    for cut in cuts:
        probe_set = sample_cut_probes(
            cut=int(cut),
            k=K_DEFAULT,
            n_pasts=int(args.n_pasts),
            n_futures=int(args.n_futures),
            seed=int(args.seed),
            style=str(args.unitary_ensemble),
        )
        probe_path = _save_probe_arrays(out_dir, probe_set, int(cut)) if bool(args.save_raw) else None
        for jv in j_values:
            if _point_key(int(cut), float(jv)) in completed:
                continue
            ham = ising_chain(length=L_DEFAULT, j=float(jv), g=G_DEFAULT)
            per_seed: list[dict[str, float | int]] = []
            responses: list[np.ndarray] = []
            pauli_arrays: list[np.ndarray] = []
            weight_arrays: list[np.ndarray] = []
            singular_arrays: list[np.ndarray] = []
            for psi0 in initial_list:
                result = characterize(
                    mc,
                    ham,
                    params,
                    k=K_DEFAULT,
                    cut=int(cut),
                    n_pasts=int(args.n_pasts),
                    n_futures=int(args.n_futures),
                    probe_set=probe_set,
                    initial_psi=psi0,
                    style=str(args.unitary_ensemble),
                )
                response = result.response_matrix(int(cut))
                diagnostics, pauli_ixyz, weights_ij, singular_values = _reconstruct_and_compare(
                    response,
                    n_pasts=int(args.n_pasts),
                    n_futures=int(args.n_futures),
                )
                if not np.isclose(
                    float(result.entropy(int(cut))),
                    float(diagnostics["entropy"]),
                    rtol=1e-12,
                    atol=1e-14,
                ):
                    msg = "public and independently recomputed response entropies disagree"
                    raise AssertionError(msg)
                per_seed.append(diagnostics)
                responses.append(response)
                pauli_arrays.append(pauli_ixyz)
                weight_arrays.append(weights_ij)
                singular_arrays.append(singular_values)

            entropy, entropy_std = _mean_and_std([float(item["entropy"]) for item in per_seed])
            entropy_raw_xyz, entropy_raw_xyz_std = _mean_and_std([
                float(item["entropy_old_raw_xyz"]) for item in per_seed
            ])
            entropy_centered_xyz, entropy_centered_xyz_std = _mean_and_std([
                float(item["entropy_old_centered_xyz"]) for item in per_seed
            ])
            raw_relpath = ""
            raw_sha256 = ""
            if bool(args.save_raw):
                raw_relpath, raw_sha256 = _save_raw_point(
                    out_dir,
                    int(cut),
                    float(jv),
                    responses,
                    pauli_arrays,
                    weight_arrays,
                    singular_arrays,
                )
            row: dict[str, float | int | str] = {
                "L": L_DEFAULT,
                "k": K_DEFAULT,
                "dt": DT_DEFAULT,
                "g": G_DEFAULT,
                "cut": int(cut),
                "J": float(jv),
                "n_pasts": int(args.n_pasts),
                "n_futures": int(args.n_futures),
                "n_seeds": n_seeds,
                "seed": int(args.seed),
                "unitary_ensemble": str(args.unitary_ensemble),
                "basis": BASIS,
                "orientation": ORIENTATION,
                "weight_scope": WEIGHT_SCOPE,
                "probability_power": 1,
                "centered": 0,
                "spectrum_discarded_weight_threshold": SPECTRUM_THRESHOLD,
                "entropy": entropy,
                "entropy_std": entropy_std,
                "entropy_old_raw_xyz": entropy_raw_xyz,
                "entropy_old_raw_xyz_std": entropy_raw_xyz_std,
                "entropy_old_centered_xyz": entropy_centered_xyz,
                "entropy_old_centered_xyz_std": entropy_centered_xyz_std,
                "entropy_change_vs_raw_xyz": entropy - entropy_raw_xyz,
                "entropy_change_vs_centered_xyz": entropy - entropy_centered_xyz,
                "response_norm": float(np.mean([float(item["response_norm"]) for item in per_seed])),
                "rank": round(float(np.mean([int(item["rank"]) for item in per_seed]))),
                "response_shape": f"{4 * int(args.n_futures)}x{int(args.n_pasts)}",
                "probe_file": str(probe_path.relative_to(out_dir)) if probe_path is not None else "",
                "probe_sha256": _sha256(probe_path) if probe_path is not None else "",
                "raw_file": raw_relpath,
                "raw_sha256": raw_sha256,
            }
            rows.append(row)
            completed.add(_point_key(int(cut), float(jv)))
            _write_checkpoint(out_dir, rows)
            manifest["completed_points"] = len(rows)
            _write_json_atomic(out_dir / "run_manifest.json", manifest)
            LOGGER.info(
                "cut=%2d J=%4.2f S_IXYZ=%.6e S_raw_XYZ=%.6e S_centered_XYZ=%.6e",
                int(cut),
                float(jv),
                entropy,
                entropy_raw_xyz,
                entropy_centered_xyz,
            )

    _write_checkpoint(out_dir, rows)
    configure_matplotlib()
    _plot_entropy_vs_j(rows, out_dir)
    plot_entropy_heatmap_cut_vs_j(rows, out_dir / "fig_entropy_heatmap_cut_vs_J")
    LOGGER.info("Wrote results to %s", out_dir)


if __name__ == "__main__":
    main()
