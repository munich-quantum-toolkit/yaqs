# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Plot the dt=0.01 circuit suite (same layout as Fig 3 / Fig S3).

Usage:
    uv run --with pandas python paper_benchmarks/scripts/plot_dt001.py
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

EPSILON = 1e-2
FLOOR = 1e-13
PANELS = (
    ("ising_1d", "1D TFIM"),
    ("heisenberg_1d", "1D XXX Heisenberg"),
    ("ising", r"2D TFIM ($4\times4$)"),
    ("heisenberg", r"2D Heisenberg ($4\times4$)"),
)
METHODS = ("full_tdvp", "tebd_swap", "mpo_zipup")


def _plot_infidelity(ax, traj: pd.DataFrame, model: str, title: str, *, label_eps: bool) -> None:
    d = traj[(traj.model == model) & (traj.trotter_step > 0)]
    # Subsample markers: 300 points is too dense.
    for im, method in enumerate(METHODS):
        st = METHOD_STYLES[method]
        g = d[d.method == method].sort_values("time")
        x = g.time.to_numpy(dtype=float)
        y = np.maximum(g.infidelity.to_numpy(dtype=float), FLOOR * 1.6)
        ax.plot(x, y, color=st["color"], linestyle=st["linestyle"], zorder=2, linewidth=1.1)
        markevery = max(1, len(x) // 15)
        ax.plot(x[im::markevery * 3], y[im::markevery * 3], color=st["color"],
                marker=st["marker"], linestyle="none", markersize=2.8, zorder=3)
    ax.axhline(EPSILON, color=REFERENCE_COLOR, linewidth=0.7, linestyle=(0, (4, 2)), zorder=1)
    if label_eps:
        ax.text(2.97, EPSILON * 1.35, r"$1-F=10^{-2}$", color=REFERENCE_COLOR,
                fontsize=6.8, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 3.05)
    ax.set_ylim(FLOOR, 3.0)
    ax.text(0.5, 0.03, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)


def fig_infidelity(traj: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 4.8), sharex=True, sharey=True)
    for ax, lab, (model, title) in zip(
        axes.flat, ("(a)", "(b)", "(c)", "(d)"), PANELS, strict=True
    ):
        _plot_infidelity(ax, traj, model, title, label_eps=(lab == "(b)"))
        panel_label(ax, lab)
    for r in range(2):
        axes[r, 0].set_ylabel(r"$1-F$")
    for c in range(2):
        axes[1, c].set_xlabel(r"physical time $t$")
    handles = [
        mpl.lines.Line2D([], [], color=METHOD_STYLES[m]["color"],
                         linestyle=METHOD_STYLES[m]["linestyle"],
                         marker=METHOD_STYLES[m]["marker"], markersize=3.0,
                         label=METHOD_STYLES[m]["label_single"] if m == "full_tdvp"
                         else METHOD_STYLES[m]["label"])
        for m in METHODS
    ]
    # Fix labels
    handles[0].set_label("gate-local TDVP")
    handles[1].set_label("TEBD+SWAP")
    handles[2].set_label("MPO zip-up")
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.015), columnspacing=1.8, handlelength=2.3)
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=1.2)
    pdf, png = save_figure(fig, "fig3_circuits_dt001")
    print(f"wrote {pdf} and {png}")


def fig_resources(traj: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_IN, 4.0),
                             sharex=True, sharey="row")
    for c, (model, title) in enumerate(PANELS):
        d = traj[(traj.model == model) & (traj.chi_max == CIRCUIT_CHI_MAIN)]
        for im, method in enumerate(METHODS):
            st = METHOD_STYLES[method]
            g = d[d.method == method].sort_values("time")
            markevery = (im, max(3, len(g) // 20))
            axes[0, c].plot(g.time, g.peak_max_bond, color=st["color"],
                            marker=st["marker"], linestyle=st["linestyle"],
                            markersize=2.4, markevery=markevery)
            axes[1, c].plot(g.time, g.peak_param_count, color=st["color"],
                            marker=st["marker"], linestyle=st["linestyle"],
                            markersize=2.4, markevery=markevery)
        axes[0, c].axhline(CIRCUIT_CHI_MAIN, color=REFERENCE_COLOR,
                           linewidth=0.7, linestyle=(0, (4, 2)))
        axes[0, c].set_ylim(0, 38.5)
        axes[0, c].set_title(title, fontsize=7.5)
        axes[1, c].set_xlabel(r"physical time $t$")
        axes[1, c].set_yscale("log")
    axes[0, 3].text(2.9, CIRCUIT_CHI_MAIN + 1.2, r"$\chi_{\max}$",
                    color=REFERENCE_COLOR, fontsize=6.5, va="bottom", ha="right")
    axes[0, 0].set_ylabel(r"peak bond dimension")
    axes[1, 0].set_ylabel(r"peak parameter count $P(\psi)$")
    for ax, lab in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)",
                                   "(e)", "(f)", "(g)", "(h)"), strict=True):
        panel_label(ax, lab)
    handles = [
        mpl.lines.Line2D([], [], color=METHOD_STYLES[m]["color"],
                         linestyle=METHOD_STYLES[m]["linestyle"],
                         marker=METHOD_STYLES[m]["marker"], markersize=3.0,
                         label=lab)
        for m, lab in (("full_tdvp", "gate-local TDVP"),
                       ("tebd_swap", "TEBD+SWAP"),
                       ("mpo_zipup", "MPO zip-up"))
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.8, handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    pdf, png = save_figure(fig, "figS3_circuit_resources_dt001")
    print(f"wrote {pdf} and {png}")


def fig_dt_compare(traj01: pd.DataFrame, traj001: pd.DataFrame) -> None:
    """Overlay dt=0.1 vs dt=0.01 for full_tdvp and mpo_zipup on each model."""
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 4.8), sharex=True, sharey=True)
    styles = {
        ("full_tdvp", 0.1): {"color": "#D55E00", "ls": "-", "label": r"TDVP $\Delta t=0.1$"},
        ("full_tdvp", 0.01): {"color": "#D55E00", "ls": "--", "label": r"TDVP $\Delta t=0.01$"},
        ("mpo_zipup", 0.1): {"color": "#009E73", "ls": "-", "label": r"zip-up $\Delta t=0.1$"},
        ("mpo_zipup", 0.01): {"color": "#009E73", "ls": "--", "label": r"zip-up $\Delta t=0.01$"},
        ("tebd_swap", 0.1): {"color": "#0072B2", "ls": "-", "label": r"TEBD $\Delta t=0.1$"},
        ("tebd_swap", 0.01): {"color": "#0072B2", "ls": "--", "label": r"TEBD $\Delta t=0.01$"},
    }
    for ax, lab, (model, title) in zip(
        axes.flat, ("(a)", "(b)", "(c)", "(d)"), PANELS, strict=True
    ):
        for method in METHODS:
            for dt, df in ((0.1, traj01), (0.01, traj001)):
                g = df[(df.model == model) & (df.method == method)
                       & (df.trotter_step > 0)].sort_values("time")
                st = styles[(method, dt)]
                y = np.maximum(g.infidelity.to_numpy(dtype=float), FLOOR * 1.6)
                ax.plot(g.time, y, color=st["color"], linestyle=st["ls"],
                        linewidth=1.05, label=st["label"] if lab == "(a)" else None)
        ax.axhline(EPSILON, color=REFERENCE_COLOR, linewidth=0.7, linestyle=(0, (4, 2)))
        ax.set_yscale("log")
        ax.set_xlim(0.0, 3.05)
        ax.set_ylim(FLOOR, 3.0)
        ax.text(0.5, 0.03, title, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8)
        panel_label(ax, lab)
    for r in range(2):
        axes[r, 0].set_ylabel(r"$1-F$")
    for c in range(2):
        axes[1, c].set_xlabel(r"physical time $t$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.2, handlelength=2.0,
               fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.2)
    pdf, png = save_figure(fig, "fig_dt01_vs_dt001")
    print(f"wrote {pdf} and {png}")


def main() -> int:
    apply_pra_style()
    traj001 = pd.read_csv(PROCESSED_DIR / "circuit_trajectories_dt001.csv")
    traj01 = pd.read_csv(PROCESSED_DIR / "circuit_trajectories.csv")
    # Prefer full_tdvp over hybrid for the dt=0.1 overlay.
    traj01 = traj01[traj01.method.isin(METHODS)]
    fig_infidelity(traj001)
    fig_resources(traj001)
    fig_dt_compare(traj01, traj001)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
