# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Diagnostic 1×3 TFIM TDVP substep convergence figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from config import CHI_VALUES, OUTPUT_DIR, RELIABILITY_THRESHOLD

FIG_WIDTH_MM = 180.0
FIG_HEIGHT_MM = 55.0
MM_TO_IN = 1.0 / 25.4
DPI = 600

SUBSTEP_STYLES = {
    1: {"color": "#0072B2", "marker": "o", "linestyle": "-", "label": "n=1"},
    2: {"color": "#D55E00", "marker": "^", "linestyle": "-", "label": "n=2"},
    4: {"color": "#009E73", "marker": "s", "linestyle": "-", "label": "n=4"},
    8: {"color": "#CC79A7", "marker": "D", "linestyle": "--", "label": "n=8"},
    16: {"color": "#56B4E9", "marker": "v", "linestyle": "--", "label": "n=16"},
    32: {"color": "#E69F00", "marker": "P", "linestyle": ":", "label": "n=32"},
}


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
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_figure(rows: list[dict[str, Any]]) -> plt.Figure:
    _apply_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN),
        sharey=True,
    )
    present_n = sorted({int(float(r["tdvp_substeps"])) for r in rows})

    for ax, chi, lab in zip(axes, CHI_VALUES, ("(a)", "(b)", "(c)"), strict=True):
        for n in present_n:
            style = SUBSTEP_STYLES.get(
                n, {"color": "0.3", "marker": "o", "linestyle": "-", "label": f"n={n}"}
            )
            pts = [
                r
                for r in rows
                if int(float(r["chi_max"])) == chi
                and int(float(r["tdvp_substeps"])) == n
                and int(float(r["trotter_step"])) > 0
            ]
            pts = sorted(pts, key=lambda r: int(float(r["trotter_step"])))
            if not pts:
                continue
            xs = np.array([int(float(p["trotter_step"])) for p in pts], dtype=float)
            ys = np.array([max(float(p["infidelity"]), 1e-16) for p in pts])
            ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"], zorder=2)
            ax.plot(
                xs,
                ys,
                linestyle="none",
                marker=style["marker"],
                color=style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=0.5,
                markersize=2.8,
                zorder=3,
            )
        ax.axhline(RELIABILITY_THRESHOLD, color="0.45", linestyle="--", linewidth=0.6, zorder=1)
        ax.set_yscale("log")
        ax.set_xlabel("Trotter step")
        ax.set_title(rf"$\chi_{{\max}}={chi}$")
        ax.set_xlim(0.5, 15.5)
        ax.set_ylim(1e-6, 2.0)
        ax.grid(True, which="major", axis="y", color="0.92", linewidth=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="top")

    axes[0].set_ylabel(r"Infidelity ($1{-}F$)")
    handles = [
        Line2D(
            [0],
            [0],
            color=SUBSTEP_STYLES[n]["color"],
            marker=SUBSTEP_STYLES[n]["marker"],
            linestyle=SUBSTEP_STYLES[n]["linestyle"],
            label=SUBSTEP_STYLES[n]["label"],
            markersize=3.0,
        )
        for n in present_n
        if n in SUBSTEP_STYLES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        fontsize=6.0,
        handlelength=1.3,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.16, top=0.82, wspace=0.18)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot TDVP substep convergence figure.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    rows = _load(args.output_dir / "tfim_tdvp_substeps.csv")
    fig = plot_figure(rows)
    pdf = args.output_dir / "tfim_tdvp_substeps.pdf"
    png = args.output_dir / "tfim_tdvp_substeps.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
