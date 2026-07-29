# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Stage 5: supplemental figures.

  figS1_angle_chi_grid   full angle-vs-chi_max grid (3 gates x chi {8,12,16})
  figS2_substeps         complete substep convergence: corrected x=0.01 grid
                         (RZZ, chi in {8,12,16}) and phase-aligned
                         self-convergence at x=1/4 with nonbinding cap
  figS3_circuit_resources peak bond dimension and peak parameter count vs
                         time at chi_max=32 for all four circuits: 1D TFIM
                         and 1D XXX Heisenberg chains (TDVP on all two-qubit
                         gates) plus the two 4x4 lattices (hybrid routing)

Usage:
    uv run --with pandas python paper_benchmarks/scripts/plot_supp.py
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
    GATE_LABELS,
    METHOD_STYLES,
    PROCESSED_DIR,
    REFERENCE_COLOR,
    apply_pra_style,
    panel_label,
    save_figure,
)

FLOOR = 1e-16
METHODS = ("hybrid_tdvp", "tebd_swap", "mpo_zipup", "variational_mpo", "no_update")
GATE_PANEL_STYLES = {
    "rxx": {"color": "#E69F00", "marker": "o", "linestyle": "-"},
    "ryy": {"color": "#56B4E9", "marker": "s", "linestyle": "--"},
    "rzz": {"color": "#000000", "marker": "^", "linestyle": "-."},
}
CHI_STYLES = {8: "-", 12: "--", 16: ":"}
CHI_COLORS = {8: "#D55E00", 12: "#0072B2", 16: "#009E73"}


def fig_s1(angle: pd.DataFrame) -> None:
    gates = ("rxx", "ryy", "rzz")
    chis = (8, 12, 16)
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL_IN, 6.6),
                             sharex=True, sharey=True)
    for r, gate in enumerate(gates):
        for c, chi in enumerate(chis):
            ax = axes[r, c]
            d = angle[(angle.gate_type == gate) & (angle.chi_max == chi)
                      & (angle.x_fraction > 0)]
            for method in METHODS:
                st = METHOD_STYLES[method]
                gen = d[(d.method == method) & (d.special_angle == 0)]
                med = gen.groupby("x_fraction").infidelity.median()
                x = np.asarray(med.index, dtype=float)
                ax.plot(x, np.maximum(med.to_numpy(), FLOOR), color=st["color"],
                        marker=st["marker"], linestyle=st["linestyle"],
                        label=st["label"], markersize=2.4, linewidth=1.0)
                spec = d[(d.method == method) & (d.special_angle == 1)]
                ms = spec.groupby("x_fraction").infidelity.median()
                if len(ms):
                    ax.plot(np.asarray(ms.index, dtype=float),
                            np.maximum(ms.to_numpy(), 2.2e-16),
                            color=st["color"], marker=st["marker"],
                            linestyle="none", markersize=2.8,
                            markerfacecolor="white", markeredgewidth=0.7)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-16, 5)
            ax.set_xlim(6e-5, 1.6)
            ax.text(0.03, 0.97, f"{GATE_LABELS[gate]}, "
                    rf"$\chi_{{\max}}={chi}$", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none",
                          "pad": 1.2})
            if r == 2:
                ax.set_xlabel(r"$\theta/(2\pi)$")
            if c == 0:
                ax.set_ylabel(r"$1-F$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.005), columnspacing=1.3, handlelength=1.9)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    pdf, png = save_figure(fig, "figS1_angle_chi_grid")
    print(f"wrote {pdf} and {png}")


def fig_s2(ss001: pd.DataFrame, ss025: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 2.35))
    # (a) corrected x=0.01 substep sweep, RZZ, chi grid
    ax = axes[0]
    d = ss001[ss001.task_type == "substep_sweep"]
    for chi in (8, 12, 16):
        g = d[d.chi_max == chi].sort_values("substeps")
        ax.plot(g.substeps, np.maximum(g.infidelity, FLOOR),
                color=CHI_COLORS[chi], linestyle=CHI_STYLES[chi], marker="o",
                markersize=3.0, label=rf"$\chi_{{\max}}={chi}$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"TDVP substeps $n$")
    ax.set_ylabel(r"$1-F$ (vs exact gate)")
    ax.text(0.97, 0.97, r"$R_{ZZ}$, $\theta/(2\pi)=10^{-2}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.7)
    panel_label(ax, "(a)")
    # (b) exact-limit infidelity at x=1/4 (all gates)
    ax = axes[1]
    for gate in ("rxx", "ryy", "rzz"):
        st = GATE_PANEL_STYLES[gate]
        g = ss025[ss025.gate_type == gate].sort_values("substeps")
        y = np.maximum(g.infidelity.to_numpy(), FLOOR)
        ax.plot(g.substeps, y, color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], markersize=3.0, label=GATE_LABELS[gate])
    ax.axhline(FLOOR, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(2e-17, 1e-7)
    ax.set_xlabel(r"TDVP substeps $n$")
    ax.set_ylabel(r"$1-F$ (vs exact gate)")
    ax.text(0.55, 0.86, r"$\theta/(2\pi)=1/4$," + "\n" + r"$\chi_{\max}=32$ (nonbinding)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7)
    ax.legend(loc="center left", fontsize=6.5, handlelength=1.7)
    panel_label(ax, "(b)")
    # (c) phase-aligned self-convergence vs n_ref = 256 (reference point excluded)
    ax = axes[2]
    for gate in ("rxx", "ryy", "rzz"):
        st = GATE_PANEL_STYLES[gate]
        g = ss025[(ss025.gate_type == gate) & (ss025.substeps < 256)].sort_values("substeps")
        y = np.maximum(g.phase_aligned_error_selfref.to_numpy(), 1e-9)
        ax.plot(g.substeps, y, color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], markersize=3.0, label=GATE_LABELS[gate])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"TDVP substeps $n$")
    ax.set_ylabel(r"$\||\psi_n\rangle - |\psi_{256}\rangle\|$ (phase-aligned)")
    panel_label(ax, "(c)")
    fig.tight_layout(w_pad=1.2)
    pdf, png = save_figure(fig, "figS2_substeps")
    print(f"wrote {pdf} and {png}")


def fig_s3(traj: pd.DataFrame) -> None:
    # All panels use full_tdvp (TDVP on every two-qubit gate, including NN).
    panels = (
        ("ising_1d", "1D TFIM", "full_tdvp"),
        ("heisenberg_1d", "1D XXX Heisenberg", "full_tdvp"),
        ("ising", r"2D TFIM ($4\times4$)", "full_tdvp"),
        ("heisenberg", r"2D Heisenberg ($4\times4$)", "full_tdvp"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL_IN, 4.0),
                             sharex=True, sharey="row")
    for c, (model, title, tdvp_method) in enumerate(panels):
        d = traj[(traj.model == model) & (traj.chi_max == CIRCUIT_CHI_MAIN)]
        for im, method in enumerate((tdvp_method, "tebd_swap", "mpo_zipup")):
            st = METHOD_STYLES[method]
            g = d[d.method == method].sort_values("time")
            # interleaved markers keep coincident curves individually visible
            axes[0, c].plot(g.time, g.peak_max_bond, color=st["color"],
                            marker=st["marker"], linestyle=st["linestyle"],
                            markersize=2.8, markevery=(im, 3))
            axes[1, c].plot(g.time, g.peak_param_count, color=st["color"],
                            marker=st["marker"], linestyle=st["linestyle"],
                            markersize=2.8, markevery=(im, 3))
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
    pdf, png = save_figure(fig, "figS3_circuit_resources")
    print(f"wrote {pdf} and {png}")


def main() -> int:
    apply_pra_style()
    angle = pd.read_csv(PROCESSED_DIR / "single_gate_angle_sweep.csv")
    ss001 = pd.read_csv(PROCESSED_DIR / "single_gate_substeps_x001.csv")
    ss025 = pd.read_csv(PROCESSED_DIR / "single_gate_substeps_x025.csv")
    traj = pd.read_csv(PROCESSED_DIR / "circuit_trajectories.csv")
    fig_s1(angle)
    fig_s2(ss001, ss025)
    fig_s3(traj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
