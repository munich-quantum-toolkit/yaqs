# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Shared journal styling for the circuit-benchmark figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CASE_ORDER = ("ising_1d", "heisenberg_1d", "ising_2d", "heisenberg_2d")
CASE_LABELS = {
    "ising_1d": "1D Ising",
    "heisenberg_1d": "1D Heisenberg",
    "ising_2d": r"$4\times4$ Ising",
    "heisenberg_2d": r"$4\times4$ Heisenberg",
}
METHOD_ORDER = ("gate_local_2tdvp", "mpo_contract_compress", "tebd_swap")
METHOD_LABELS = {
    "gate_local_2tdvp": "TDVP",
    "mpo_contract_compress": "MPO",
    "tebd_swap": "TEBD+SWAP",
}
METHOD_STYLES = {
    "gate_local_2tdvp": {"color": "#E64B35", "marker": "o", "linestyle": "-"},
    "mpo_contract_compress": {"color": "#00A087", "marker": "s", "linestyle": "-."},
    "tebd_swap": {"color": "#3C5488", "marker": "^", "linestyle": "--"},
}


def apply_style() -> None:
    """Use compact, colorblind-safe journal figure settings."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 7.8,
            "axes.labelsize": 9.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.72,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#202020",
            "text.color": "#202020",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "lines.linewidth": 1.6,
            "lines.markersize": 4.4,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes) -> None:
    """Apply the shared axis treatment."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(which="both", direction="out", width=0.7)
    axis.grid(axis="y", which="major", color="#E6E8EB", linewidth=0.45, zorder=0)


def legend_handles() -> list[Line2D]:
    """Return method handles in the shared display order."""
    return [
        Line2D(
            [0],
            [0],
            label=METHOD_LABELS[method],
            color=METHOD_STYLES[method]["color"],
            linestyle=METHOD_STYLES[method]["linestyle"],
            marker=METHOD_STYLES[method]["marker"],
            markerfacecolor="white" if method == "tebd_swap" else METHOD_STYLES[method]["color"],
            markeredgewidth=0.85,
        )
        for method in METHOD_ORDER
    ]
