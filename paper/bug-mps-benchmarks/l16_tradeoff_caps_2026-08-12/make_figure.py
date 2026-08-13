#!/usr/bin/env python3
"""Create the manuscript runtime-versus-infidelity figure from validated CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
INPUT = HERE / "tradeoff_all_points.csv"
PDF = HERE / "runtime_accuracy_tradeoff.pdf"
PNG = HERE / "runtime_accuracy_tradeoff.png"


def load_rows() -> list[dict[str, object]]:
    """Read plot inputs as typed records."""
    rows: list[dict[str, object]] = []
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "model": row["model"],
                    "method": row["method"],
                    "dt": float(row["dt"]),
                    "epsilon": float(row["epsilon"]),
                    "runtime": float(row["runtime_median_seconds"]),
                    "runtime_minimum": float(row["runtime_minimum_seconds"]),
                    "runtime_maximum": float(row["runtime_maximum_seconds"]),
                    "infidelity": float(row["infidelity"]),
                    "is_pareto": row["is_final_pareto"].lower() == "true",
                }
            )
    return rows


def main() -> None:
    """Render publication-ready vector and raster versions."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    rows = load_rows()
    styles = {
        "bug": {"label": "BUG", "color": "#0072B2", "marker": "o"},
        "2tdvp": {"label": "2TDVP", "color": "#D55E00", "marker": "s"},
    }
    titles = {"tfim": "(a) Transverse-field Ising model", "hs": "(b) Haldane-Shastry model"}

    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.72), constrained_layout=True)
    for axis, model in zip(axes, ("tfim", "hs"), strict=True):
        model_rows = [row for row in rows if row["model"] == model]
        for method, style in styles.items():
            candidates = [row for row in model_rows if row["method"] == method]
            dominated = [row for row in candidates if not row["is_pareto"]]
            pareto = sorted(
                [row for row in candidates if row["is_pareto"]],
                key=lambda row: row["runtime"],
            )
            axis.scatter(
                [row["runtime"] for row in dominated],
                [row["infidelity"] for row in dominated],
                marker=style["marker"],
                s=24,
                facecolors="none",
                edgecolors=style["color"],
                alpha=0.35,
                linewidths=0.8,
                zorder=1,
            )
            runtimes = [row["runtime"] for row in pareto]
            axis.errorbar(
                runtimes,
                [row["infidelity"] for row in pareto],
                xerr=[
                    [row["runtime"] - row["runtime_minimum"] for row in pareto],
                    [row["runtime_maximum"] - row["runtime"] for row in pareto],
                ],
                color=style["color"],
                marker=style["marker"],
                markersize=4.4,
                markeredgewidth=0.6,
                markeredgecolor="white",
                capsize=1.8,
                elinewidth=0.7,
                label=style["label"],
                zorder=3,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(titles[model], loc="left", pad=5)
        axis.set_xlabel("Runtime (s)")
        axis.grid(True, which="major", color="#D8D8D8", linewidth=0.55)
        axis.grid(True, which="minor", color="#EEEEEE", linewidth=0.35)
        axis.set_axisbelow(True)
        axis.tick_params(which="both", direction="in", top=True, right=True)
    axes[0].set_ylabel("Infidelity")

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            markersize=4.4,
            label=style["label"],
        )
        for style in styles.values()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#777777",
            marker="o",
            markerfacecolor="none",
            linestyle="none",
            alpha=0.55,
            label="dominated grid point",
        )
    )
    figure.legend(
        handles=handles,
        loc="outside upper center",
        ncols=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
    )
    figure.savefig(PDF)
    figure.savefig(PNG, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
