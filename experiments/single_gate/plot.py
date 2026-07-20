# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Publication 1×4 figure for the main-text single RZZ gate benchmark."""

from __future__ import annotations

import argparse
import json
import operator
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from config import FIGURES_DIR, FIGURE_STEM, FIT_X_MAX, FIT_X_MIN, OUTPUT_DIR, PLOT_FLOOR
from matplotlib.lines import Line2D

FIG_WIDTH_MM = 180.0
FIG_HEIGHT_MM = 55.0
MM_TO_IN = 1.0 / 25.4
DPI = 600

METHODS = ("hybrid_tdvp", "tebd_swap", "mpo_zipup", "variational_mpo")
METHOD_LABELS = {
    "hybrid_tdvp": "TDVP",
    "tebd_swap": "TEBD+SWAP",
    "mpo_zipup": "MPO zip-up",
    "variational_mpo": "Variational MPO",
}
METHOD_STYLES = {
    "hybrid_tdvp": {"color": "#0072B2", "marker": "o", "linestyle": "-", "fillstyle": "full"},
    "tebd_swap": {"color": "#D55E00", "marker": "^", "linestyle": "-", "fillstyle": "full"},
    "mpo_zipup": {"color": "#009E73", "marker": "s", "linestyle": "-", "fillstyle": "full"},
    "variational_mpo": {"color": "#CC79A7", "marker": "D", "linestyle": "--", "fillstyle": "none"},
}
PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)")
Y_FLOOR_DISPLAY = PLOT_FLOOR / 3.0  # slightly below 10^-12 for visibility


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _display_y(val: float) -> float:
    return max(val, PLOT_FLOOR)


def load_rows(db_path: Path, task_type: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM results WHERE task_type=? ORDER BY chi_max, x_fraction, method",
        (task_type,),
    ).fetchall()
    conn.close()
    out = []
    for row in rows:
        item = dict(row)
        item["chi_max"] = int(item["chi_max"])
        item["x_fraction"] = float(item["x_fraction"])
        item["theta"] = float(item["theta"])
        item["infidelity"] = float(item["infidelity"])
        item["special_angle"] = bool(int(item["special_angle"]))
        if "substeps" in item and item["substeps"] not in {"", None}:
            item["substeps"] = int(item["substeps"])
        out.append(item)
    return out


def load_chi_values(output_dir: Path) -> tuple[int, int, int]:
    path = output_dir / "chi_selection.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data["chi0"]), int(data["chi_intermediate"]), int(data["chi_full"])
    conn = sqlite3.connect(output_dir / "results.sqlite")
    low = int(conn.execute("SELECT value FROM meta WHERE key='chi_low'").fetchone()[0])
    mid = int(conn.execute("SELECT value FROM meta WHERE key='chi_intermediate'").fetchone()[0])
    full = int(conn.execute("SELECT value FROM meta WHERE key='chi_full'").fetchone()[0])
    conn.close()
    return low, mid, full


def _plot_method_curves(ax: plt.Axes, rows: list[dict[str, Any]], *, show_theta2: bool = False) -> None:
    zip_lookup: dict[tuple[float, str], float] = {}
    for row in rows:
        if row["method"] == "mpo_zipup":
            zip_lookup[row["x_fraction"], "mpo"] = row["infidelity"]

    for method in METHODS:
        style = METHOD_STYLES[method]
        generic = sorted(
            [r for r in rows if r["method"] == method and not r["special_angle"]],
            key=operator.itemgetter("x_fraction"),
        )
        special = sorted(
            [r for r in rows if r["method"] == method and r["special_angle"]],
            key=operator.itemgetter("x_fraction"),
        )
        if generic:
            xs = np.array([r["x_fraction"] for r in generic])
            ys = np.array([_display_y(r["infidelity"]) for r in generic])
            below = np.array([r["infidelity"] < PLOT_FLOOR for r in generic])
            linestyle = style["linestyle"]
            if method == "variational_mpo":
                coincident = all(abs(r["infidelity"] - zip_lookup.get((r["x_fraction"], "mpo"), -1.0)) < 1e-14 for r in generic)
                if coincident:
                    linestyle = "--"
            ax.plot(xs, ys, color=style["color"], linestyle=linestyle, zorder=2)
            ax.plot(
                xs[~below],
                ys[~below],
                linestyle="none",
                marker=style["marker"],
                color=style["color"],
                fillstyle=style["fillstyle"],
                markeredgecolor=style["color"],
                markeredgewidth=0.5,
                markersize=3.0 if method == "variational_mpo" else 2.8,
                zorder=3,
            )
            if np.any(below):
                ax.plot(
                    xs[below],
                    ys[below],
                    linestyle="none",
                    marker="v",
                    color=style["color"],
                    markersize=2.5,
                    zorder=3,
                )
        if special:
            xs = np.array([r["x_fraction"] for r in special])
            ys = np.array([_display_y(r["infidelity"]) for r in special])
            ax.plot(
                xs,
                ys,
                linestyle="none",
                marker=style["marker"],
                color=style["color"],
                fillstyle=style["fillstyle"],
                markeredgecolor=style["color"],
                markeredgewidth=0.5,
                markersize=4.0 if method == "variational_mpo" else 3.5,
                zorder=4,
            )

    if show_theta2:
        tdvp = sorted([r for r in rows if r["method"] == "hybrid_tdvp" and not r["special_angle"]], key=operator.itemgetter("x_fraction"))
        fit_pts = [r for r in tdvp if FIT_X_MIN <= r["x_fraction"] <= FIT_X_MAX]
        if fit_pts:
            ref = fit_pts[len(fit_pts) // 2]
            guide_x = np.logspace(np.log10(FIT_X_MIN), np.log10(FIT_X_MAX), 20)
            guide_y = ref["infidelity"] * (guide_x / ref["x_fraction"]) ** 2 * 0.55
            disp_y = [_display_y(y) for y in guide_y]
            ax.plot(guide_x, disp_y, linestyle="--", color="0.55", linewidth=0.6, zorder=1)
            ann_x = FIT_X_MAX
            ann_y = _display_y(ref["infidelity"] * (ann_x / ref["x_fraction"]) ** 2 * 0.55)
            ax.text(
                ann_x * 1.45,
                ann_y,
                r"$\propto\theta^2$",
                fontsize=7.0,
                color="0.15",
                ha="left",
                va="center",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
                zorder=5,
            )


def plot_figure(
    angle_rows: list[dict[str, Any]],
    substep_rows: list[dict[str, Any]],
    *,
    chi_low: int,
    chi_mid: int,
    chi_full: int,
) -> plt.Figure:
    _apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN))
    gs = fig.add_gridspec(1, 4, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)
    ax3 = fig.add_subplot(gs[0, 3])
    axes = [ax0, ax1, ax2, ax3]
    chi_panels = [chi_low, chi_mid, chi_full]

    y_vals = [r["infidelity"] for r in angle_rows if r["infidelity"] > 0]
    y_lo = Y_FLOOR_DISPLAY
    y_hi = 1.0
    if y_vals:
        y_hi = min(1.0, max(y_vals) * 1.5)

    method_handles = [
        Line2D(
            [0], [0],
            color=METHOD_STYLES[m]["color"],
            marker=METHOD_STYLES[m]["marker"],
            linestyle=METHOD_STYLES[m]["linestyle"],
            fillstyle=METHOD_STYLES[m]["fillstyle"],
            markeredgecolor=METHOD_STYLES[m]["color"],
            markeredgewidth=0.5,
            markersize=3.5,
            label=METHOD_LABELS[m],
        )
        for m in METHODS
    ]

    for idx, chi in enumerate(chi_panels):
        ax = axes[idx]
        subset = [r for r in angle_rows if r["chi_max"] == chi]
        _plot_method_curves(ax, subset, show_theta2=(idx == 0))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1.0)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(rf"$\chi_{{\max}}={chi}$")
        ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
        ax.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
        ax.grid(True, which="major", axis="y", color="0.92", linewidth=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.03, 0.97, PANEL_LABELS[idx], transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="top")
        if idx == 0:
            ax.set_ylabel("Infidelity, 1 − F")
            ax.legend(
                handles=method_handles,
                loc="lower right",
                frameon=False,
                fontsize=6.0,
                handlelength=1.5,
            )
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

    ax = axes[3]
    colors = {8: "#0072B2", 12: "#56B4E9", 16: "#999999"}
    for chi in chi_panels:
        subset = sorted(
            [r for r in substep_rows if r["chi_max"] == chi],
            key=lambda r: int(r["substeps"]),
        )
        if not subset:
            continue
        ns = np.array([int(r["substeps"]) for r in subset])
        raw = np.array([float(r["infidelity"]) for r in subset])
        below = raw <= 0.0
        below |= raw < PLOT_FLOOR
        ys = np.array([PLOT_FLOOR if b else max(v, PLOT_FLOOR) for v, b in zip(raw, below, strict=True)])
        color = colors.get(chi, "#0072B2")
        ax.plot(ns, ys, color=color, linewidth=0.9, zorder=2)
        if np.any(~below):
            ax.plot(
                ns[~below],
                ys[~below],
                linestyle="none",
                marker="o",
                color=color,
                markersize=3.0,
                zorder=3,
            )
        if np.any(below):
            ax.plot(
                ns[below],
                np.full(np.sum(below), PLOT_FLOOR),
                linestyle="none",
                marker="v",
                color=color,
                markersize=2.8,
                zorder=3,
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(0.8, 80)
    ax.set_ylim(y_lo, y_hi)
    substep_ticks = (1, 2, 4, 8, 16, 32, 64)
    ax.set_xticks(list(substep_ticks))
    ax.set_xticklabels([str(v) for v in substep_ticks])
    ax.set_xlabel("TDVP substeps")
    ax.grid(True, which="major", axis="y", color="0.92", linewidth=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.03, 0.97, PANEL_LABELS[3], transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="top")
    chi_handles = [
        Line2D([0], [0], color=colors[chi], marker="o", linestyle="-", markersize=3.0, label=rf"$\chi_{{\max}}={chi}$")
        for chi in chi_panels
    ]
    ax.legend(
        handles=chi_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.34),
        frameon=False,
        fontsize=6.0,
    )

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.90)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot main-text 1×4 single-gate figure.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Benchmark data directory")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Publication figure output directory")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    db_path = output_dir / "results.sqlite"
    if not db_path.exists():
        msg = f"Missing benchmark database: {db_path}"
        raise SystemExit(msg)

    chi_low, chi_mid, chi_full = load_chi_values(output_dir)
    angle_rows = load_rows(db_path, "angle_sweep")
    substep_rows = load_rows(db_path, "substep_sweep")
    if not angle_rows or not substep_rows:
        msg = "Benchmark data incomplete; run run.py first."
        raise SystemExit(msg)

    fig = plot_figure(angle_rows, substep_rows, chi_low=chi_low, chi_mid=chi_mid, chi_full=chi_full)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figures_dir / f"{FIGURE_STEM}.pdf"
    png_path = figures_dir / f"{FIGURE_STEM}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
