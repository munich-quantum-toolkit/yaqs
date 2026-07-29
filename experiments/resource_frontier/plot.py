# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Publication 1×2 resource-frontier figure (MPS representation + runtime)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from config import METHODS, OUTPUT_DIR

FIG_WIDTH_MM = 89.0  # Nature Communications single-column width
FIG_HEIGHT_MM = 110.0
MM_TO_IN = 1.0 / 25.4
DPI = 600

METHOD_LABELS = {
    "hybrid_tdvp": "TDVP",
    "tebd_swap": "TEBD+SWAP",
    "mpo_zipup": "MPO zip-up",
}
METHOD_STYLES = {
    "hybrid_tdvp": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "tebd_swap": {"color": "#D55E00", "marker": "^", "linestyle": "-"},
    "mpo_zipup": {"color": "#009E73", "marker": "s", "linestyle": "-"},
}

FIGURE_CAPTION = (
    "Resource requirements for reliable 4×4 TFIM circuit simulation. "
    "For each target time, the plotted value is minimized over all tested "
    "bond-dimension caps satisfying 1-F<10⁻² at every preceding Trotter step. "
    "(a) Minimum peak retained MPS parameter count. "
    "(b) Measured runtime trade-off, using median wall-clock timings from three "
    "isolated repetitions on fixed hardware. "
    "TDVP combines a smaller retained representation with competitive runtime at "
    "intermediate times, while its measured cost rises sharply at later times when "
    "the runtime-minimizing reliable χmax increases (32→48 at t=1.3; 48→64 at t=1.5)."
)


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
            "savefig.facecolor": "white",
        }
    )


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _method_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLES[m]["color"],
            marker=METHOD_STYLES[m]["marker"],
            linestyle=METHOD_STYLES[m]["linestyle"],
            markeredgecolor=METHOD_STYLES[m]["color"],
            markeredgewidth=0.5,
            markersize=3.5,
            label=METHOD_LABELS[m],
        )
        for m in METHODS
    ]


def plot_figure(mem: list[dict[str, Any]], runtime: list[dict[str, Any]]) -> plt.Figure:
    _apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        2,
        1,
        figsize=(FIG_WIDTH_MM * MM_TO_IN, FIG_HEIGHT_MM * MM_TO_IN),
        sharex=True,
    )

    for method in METHODS:
        style = METHOD_STYLES[method]
        pts = [
            r
            for r in mem
            if r["method"] == method
            and str(r.get("missing", "1")) in {"0", "0.0"}
            and r.get("P_star", "") != ""
        ]
        pts = sorted(pts, key=lambda r: float(r["target_time"]))
        if not pts:
            continue
        xs = np.array([float(p["target_time"]) for p in pts])
        ys = np.array([float(p["P_star"]) for p in pts])
        ax_a.plot(xs, ys, color=style["color"], linestyle=style["linestyle"], zorder=2)
        ax_a.plot(
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

    ax_a.set_yscale("log")
    ax_a.set_ylim(80.0, 1.2e6)
    ax_a.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_a.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax_a.yaxis.set_minor_formatter(NullFormatter())
    ax_a.set_ylabel(r"$P_{\max}$")
    ax_a.set_xlim(0.05, 1.55)
    ax_a.grid(True, which="major", axis="y", color="0.92", linewidth=0.35)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.legend(
        handles=_method_handles(),
        loc="lower right",
        frameon=False,
        fontsize=5.5,
        handlelength=1.2,
        labelspacing=0.2,
        borderaxespad=0.3,
    )

    for method in METHODS:
        style = METHOD_STYLES[method]
        pts = [
            r
            for r in runtime
            if r["method"] == method
            and str(r.get("missing", "1")) in {"0", "0.0"}
            and r.get("R_star_s", "") != ""
        ]
        pts = sorted(pts, key=lambda r: float(r["target_time"]))
        if not pts:
            continue
        xs = np.array([float(p["target_time"]) for p in pts])
        ys = np.array([float(p["R_star_s"]) for p in pts])
        lo = np.array(
            [
                float(p["R_iqr_low_s"]) if p.get("R_iqr_low_s", "") != "" else float(p["R_star_s"])
                for p in pts
            ]
        )
        hi = np.array(
            [
                float(p["R_iqr_high_s"]) if p.get("R_iqr_high_s", "") != "" else float(p["R_star_s"])
                for p in pts
            ]
        )
        ax_b.fill_between(xs, lo, hi, color=style["color"], alpha=0.15, linewidth=0, zorder=1)
        ax_b.plot(xs, ys, color=style["color"], linestyle=style["linestyle"], zorder=2)
        ax_b.plot(
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

    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"$t$")
    ax_b.set_ylabel("Measured runtime (s)")
    ax_b.set_xlim(0.05, 1.55)
    ax_b.grid(True, which="major", axis="y", color="0.92", linewidth=0.35)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    for ax, lab in zip((ax_a, ax_b), ("(a)", "(b)"), strict=True):
        ax.text(
            0.03,
            0.97,
            lab,
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            va="top",
            ha="left",
            zorder=5,
        )

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.08, top=0.97, hspace=0.22)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot resource frontier figure.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    out = args.output_dir
    mem = _load(out / "memory_frontier.csv")
    runtime = _load(out / "runtime_frontier.csv")
    fig = plot_figure(mem, runtime)
    pdf = out / "resource_frontier.pdf"
    png = out / "resource_frontier.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    caption_path = out / "resource_frontier.md"
    caption_path.write_text(
        "\n".join(
            [
                "# Resource frontier figure",
                "",
                "## Caption",
                "",
                FIGURE_CAPTION,
                "",
                "## Layout",
                "",
                "- Nature Communications single-column width (89 mm), stacked (a)/(b), no panel titles.",
                "- (a) MPS representation frontier (Pmax); (b) measured runtime trade-off (median ± IQR).",
                "- Shared method legend in the lower-right of panel (a).",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(f"Wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
