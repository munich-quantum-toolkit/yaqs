#!/usr/bin/env python3
"""Create the manuscript runtime-versus-infidelity figure from validated CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter


HERE = Path(__file__).resolve().parent
INPUT = HERE / "tradeoff_all_points.csv"
PDF = HERE / "runtime_accuracy_tradeoff.pdf"
PNG = HERE / "runtime_accuracy_tradeoff.png"

DT_STYLES = {
    0.01: {"marker": "o", "label": r"$h=0.01$"},
    0.005: {"marker": "s", "label": r"$h=0.005$"},
    0.0025: {"marker": "^", "label": r"$h=0.0025$"},
}
EPSILON_SIZES = {
    1e-8: 18,
    1e-10: 31,
    1e-12: 48,
    1e-14: 69,
}


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
        "bug": {"label": "BUG", "color": "#0072B2", "linestyle": "-"},
        "2tdvp": {"label": "2TDVP", "color": "#D55E00", "linestyle": "--"},
    }
    titles = {"tfim": "(a) Transverse-field Ising model", "hs": "(b) Haldane-Shastry model"}

    figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.72))
    figure.subplots_adjust(left=0.085, right=0.995, bottom=0.16, top=0.70, wspace=0.13)
    for axis, model in zip(axes, ("tfim", "hs"), strict=True):
        model_rows = [row for row in rows if row["model"] == model]
        for method, style in styles.items():
            candidates = [row for row in model_rows if row["method"] == method]
            pareto = sorted(
                [row for row in candidates if row["is_pareto"]],
                key=lambda row: row["runtime"],
            )
            axis.plot(
                [row["runtime"] for row in pareto],
                [row["infidelity"] for row in pareto],
                color=style["color"],
                linestyle=style["linestyle"],
                zorder=2,
            )
            for row in candidates:
                pareto_point = bool(row["is_pareto"])
                marker = DT_STYLES[float(row["dt"])]["marker"]
                size = EPSILON_SIZES[float(row["epsilon"])]
                axis.scatter(
                    [row["runtime"]],
                    [row["infidelity"]],
                    marker=marker,
                    s=size,
                    facecolors=style["color"] if pareto_point else "none",
                    edgecolors=style["color"],
                    alpha=1.0 if pareto_point else 0.52,
                    linewidths=0.75,
                    zorder=4 if pareto_point else 1,
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.xaxis.set_minor_formatter(NullFormatter())
        if model == "tfim":
            axis.set_xticks((1, 2, 4, 8), labels=("1", "2", "4", "8"))
        axis.set_title(titles[model], loc="left", pad=5)
        axis.set_xlabel("Runtime (s)")
        axis.grid(True, which="major", color="#D8D8D8", linewidth=0.55)
        axis.grid(True, which="minor", color="#EEEEEE", linewidth=0.35)
        axis.set_axisbelow(True)
        axis.tick_params(which="both", direction="in", top=True, right=True)
    axes[0].set_ylabel("Infidelity")

    method_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
        )
        for style in styles.values()
    ]
    method_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="o",
                markerfacecolor="#555555",
                linestyle="none",
                markersize=4.8,
                label="Pareto",
            ),
            Line2D(
                [0],
                [0],
                color="#777777",
                marker="o",
                markerfacecolor="none",
                linestyle="none",
                markersize=4.8,
                label="dominated",
            ),
        ]
    )
    figure.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncols=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
    )

    dt_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker=style["marker"],
            markerfacecolor="white",
            linestyle="none",
            markersize=4.8,
            label=style["label"],
        )
        for style in DT_STYLES.values()
    ]
    epsilon_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            markerfacecolor="#777777",
            linestyle="none",
            markersize=EPSILON_SIZES[epsilon] ** 0.5,
            label=label,
        )
        for epsilon, label in (
            (1e-8, r"$\epsilon=10^{-8}$"),
            (1e-10, r"$10^{-10}$"),
            (1e-12, r"$10^{-12}$"),
            (1e-14, r"$10^{-14}$"),
        )
    ]
    figure.legend(
        handles=[*dt_handles, *epsilon_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncols=7,
        frameon=False,
        handlelength=0.8,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    figure.savefig(PDF)
    figure.savefig(PNG, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
