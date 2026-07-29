# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Diagnostic: hybrid_tdvp vs full_tdvp on the 2D circuits (not a main figure).

Shows the quantitative effect of routing nearest-neighbour gates through the
gate-local TDVP window update instead of direct TEBD contraction.

Usage:
    uv run --with pandas python paper_benchmarks/scripts/plot_hybrid_vs_full.py
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pb_common import (
    CIRCUIT_CHI_MAIN,
    DOUBLE_COL_IN,
    FIGURES_DIR,
    METHOD_STYLES,
    PROCESSED_DIR,
    REFERENCE_COLOR,
    apply_pra_style,
    panel_label,
    save_figure,
)

FLOOR = 1e-13
EPSILON = 1e-2


def main() -> int:
    apply_pra_style()
    traj = pd.read_csv(PROCESSED_DIR / "circuit_trajectories.csv")
    traj = traj[traj.chi_max == CIRCUIT_CHI_MAIN]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 2.55), sharey=True)
    panels = (
        ("ising", "2D TFIM"),
        ("heisenberg", "2D Heisenberg"),
    )
    styles = {
        "hybrid_tdvp": {
            "label": "hybrid (TEBD on NN)",
            "color": "#E69F00",
            "linestyle": "--",
            "marker": "^",
        },
        "full_tdvp": {
            "label": "full (TDVP on all 2q)",
            "color": METHOD_STYLES["full_tdvp"]["color"],
            "linestyle": "-",
            "marker": "o",
        },
        "mpo_zipup": {
            "label": "MPO zip-up",
            "color": METHOD_STYLES["mpo_zipup"]["color"],
            "linestyle": "-.",
            "marker": "s",
        },
    }
    for ax, (model, title) in zip(axes, panels, strict=True):
        d = traj[(traj.model == model) & (traj.trotter_step > 0)]
        for method, st in styles.items():
            g = d[d.method == method].sort_values("time")
            y = np.maximum(g.infidelity.to_numpy(dtype=float), FLOOR * 1.6)
            ax.plot(g.time, y, color=st["color"], linestyle=st["linestyle"],
                    marker=st["marker"], markersize=3.0, label=st["label"],
                    markevery=2)
        ax.axhline(EPSILON, color=REFERENCE_COLOR, linewidth=0.7,
                   linestyle=(0, (4, 2)))
        ax.set_yscale("log")
        ax.set_xlim(0.0, 3.05)
        ax.set_ylim(FLOOR, 3.0)
        ax.set_xlabel(r"physical time $t$")
        ax.text(0.5, 0.03, title, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel(r"$1-F$")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.04), columnspacing=1.5, handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.2)
    pdf, png = save_figure(fig, "fig_hybrid_vs_full_tdvp")
    print(f"wrote {pdf} and {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
