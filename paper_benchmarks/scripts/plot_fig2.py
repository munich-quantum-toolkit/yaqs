# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Stage 5: Figure 2 - individual gates and numerical control (2x2, double column).

(a) RXX, (b) RYY, (c) RZZ: normalized infidelity vs theta/(2pi) at the
corrected chi_max = 8 slice; median over seeds {11,22,33} with min-max band;
all four comparator methods; empirical theta^2 guide in the small-angle region.
(d) TDVP substep convergence at theta/(2pi) = 1/4 with a nonbinding cap
(chi_max = 32; peak chi = 16): exact-gate infidelity vs substeps for all
three gates. Stored zeros are shown at the display floor with open markers.

Usage:
    uv run --with pandas python paper_benchmarks/scripts/plot_fig2.py
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pb_common import (
    DOUBLE_COL_IN,
    FIGURES_DIR,
    GATE_LABELS,
    METHOD_STYLES,
    PROCESSED_DIR,
    REFERENCE_COLOR,
    apply_pra_style,
    panel_label,
    save_figure,
)

CHI_SLICE = 8
FLOOR = 1e-16
METHODS = ("hybrid_tdvp", "tebd_swap", "mpo_zipup", "variational_mpo")
GATE_PANEL_STYLES = {
    "rxx": {"color": "#E69F00", "marker": "o", "linestyle": "-"},
    "ryy": {"color": "#56B4E9", "marker": "s", "linestyle": "--"},
    "rzz": {"color": "#000000", "marker": "^", "linestyle": "-."},
}
YLIM_ABC = (1e-9, 3.0)


def plot_angle_panel(ax, angle: pd.DataFrame, gate: str) -> None:
    d = angle[(angle.gate_type == gate) & (angle.chi_max == CHI_SLICE)]
    d = d[d.x_fraction > 0]
    for method in METHODS:
        st = METHOD_STYLES[method]
        gen = d[(d.method == method) & (d.special_angle == 0)]
        g = gen.groupby("x_fraction").infidelity
        med, lo, hi = g.median(), g.min(), g.max()
        x = np.asarray(med.index, dtype=float)
        ax.fill_between(
            x, np.maximum(lo.to_numpy(), FLOOR), np.maximum(hi.to_numpy(), FLOOR),
            color=st["color"], alpha=0.16, linewidth=0.0, zorder=1,
        )
        ax.plot(
            x, np.maximum(med.to_numpy(), FLOOR),
            color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
            label=st["label"], markersize=3.0, zorder=3,
        )
        # special angles (theta/2pi in {1/4, 1/2, 1}: product gate / identity):
        # isolated open markers; below-range values pinned to the axis floor
        spec = d[(d.method == method) & (d.special_angle == 1)]
        gs = spec.groupby("x_fraction").infidelity.median()
        if len(gs):
            xs = np.asarray(gs.index, dtype=float)
            ys = np.maximum(gs.to_numpy(), YLIM_ABC[0] * 1.35)
            ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                    linestyle="none", markersize=3.4, markerfacecolor="white",
                    markeredgewidth=0.8, zorder=4)
    # empirical small-angle theta^2 guide anchored on the TDVP median
    td = d[d.method == "hybrid_tdvp"].groupby("x_fraction").infidelity.median()
    xs = np.asarray(td.index, dtype=float)
    small = (xs >= 1e-4) & (xs <= 1e-2)
    coef = float(np.median(td.to_numpy()[small] / xs[small] ** 2))
    xg = np.array([1e-4, 3e-2])
    ax.plot(xg, 1.6 * coef * xg**2, color=REFERENCE_COLOR,
            linestyle=(0, (2.5, 1.5)), linewidth=0.9, zorder=2)
    ax.text(2.2e-4, 3.0 * coef * (2.2e-4) ** 2, r"$\propto\theta^2$",
            color=REFERENCE_COLOR, fontsize=7.5, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6e-5, 1.6)
    ax.set_ylim(*YLIM_ABC)
    ax.set_xlabel(r"$\theta/(2\pi)$")
    ax.text(0.97, 0.05, GATE_LABELS[gate], transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5)


def plot_substep_panel(ax, ss: pd.DataFrame) -> None:
    for gate in ("rxx", "ryy", "rzz"):
        st = GATE_PANEL_STYLES[gate]
        d = ss[ss.gate_type == gate].sort_values("substeps")
        n = d.substeps.to_numpy(dtype=float)
        y = d.infidelity.to_numpy(dtype=float)
        below = y < FLOOR
        yy = np.maximum(y, FLOOR)
        ax.plot(n, yy, color=st["color"], linestyle=st["linestyle"],
                linewidth=1.1, zorder=2)
        ax.plot(n[~below], yy[~below], color=st["color"], marker=st["marker"],
                linestyle="none", markersize=3.4, label=GATE_LABELS[gate], zorder=3)
        if below.any():  # stored zeros: open markers at the display floor
            ax.plot(n[below], yy[below], color=st["color"], marker=st["marker"],
                    linestyle="none", markersize=3.4, markerfacecolor="white", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"TDVP substeps $n$")
    ax.set_ylim(2e-17, 1e-7)
    ax.axhline(FLOOR, color=REFERENCE_COLOR, linewidth=0.6, linestyle=":")
    ax.text(0.985, 0.035, "open symbols: stored zeros at display floor",
            color=REFERENCE_COLOR, fontsize=6.3, ha="right", va="bottom",
            transform=ax.transAxes)
    ax.text(
        0.03, 0.76,
        r"$\theta/(2\pi)=1/4$, $\chi_{\max}=32$" + "\n"
        r"(nonbinding; peak $\chi=16$)",
        transform=ax.transAxes, fontsize=7, va="top",
    )
    ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.4)


def main() -> int:
    apply_pra_style()
    angle = pd.read_csv(PROCESSED_DIR / "single_gate_angle_sweep.csv")
    ss = pd.read_csv(PROCESSED_DIR / "single_gate_substeps_x025.csv")

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 4.9))
    for ax, gate, lab in zip(axes.flat[:3], ("rxx", "ryy", "rzz"), "abc", strict=False):
        plot_angle_panel(ax, angle, gate)
        panel_label(ax, f"({lab})")
    axes[0, 0].set_ylabel(r"$1-F$")
    axes[1, 0].set_ylabel(r"$1-F$")
    axes[0, 1].set_yticklabels([])
    plot_substep_panel(axes[1, 1], ss)
    axes[1, 1].set_ylabel(r"$1-F$ (vs exact gate)")
    panel_label(axes[1, 1], "(d)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.015), columnspacing=1.6, handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.4, w_pad=1.0)

    # export the exact plotted data
    plotted = angle[(angle.chi_max == CHI_SLICE) & angle.method.isin(METHODS)
                    & (angle.x_fraction > 0)]
    plotted.to_csv(PROCESSED_DIR / "fig2_panels_abc_source.csv", index=False)
    ss.to_csv(PROCESSED_DIR / "fig2_panel_d_source.csv", index=False)

    pdf, png = save_figure(fig, "fig2_single_gate")
    print(f"wrote {pdf} and {png}")

    caption = r"""% Auto-generated by paper_benchmarks/scripts/plot_fig2.py
\newcommand{\figtwocaption}{%
Individual long-range gate benchmark on $L=12$ qubits (random normalized
complex MPS, initial bond dimension $\chi_0=8$, gate acting on sites $2$ and
$9$, i.e.\ seven MPS bonds apart).
(a)--(c) Normalized infidelity $1-F$ after a single
$R_{XX}(\theta)$, $R_{YY}(\theta)$, $R_{ZZ}(\theta)$ gate at fixed cap
$\chi_{\max}=8$ versus $\theta/(2\pi)$: median over seeds
$\{11,22,33\}$ (bands show the min--max range) for hybrid gate-local TDVP
($n=1$ substep), TEBD+SWAP, MPO zip-up, and multi-start variational MPO.
The gray guide indicates the empirical small-angle $\propto\theta^2$ scaling
of the gate-local TDVP error; it is an observed scaling and is not a
numerical verification of first-order apply-and-project equivalence.
Open symbols mark the special angles $\theta/(2\pi)\in\{1/4,1/2,1\}$, at
which the gate degenerates to a product operator or the identity and the
direct MPO methods become exact; values below the axis range are pinned to
the lower axis edge.
(d) Exact-gate infidelity of gate-local TDVP versus the number of
fractional-time substeps $n$ at $\theta/(2\pi)=1/4$ with a nonbinding cap
$\chi_{\max}=32$ (the exact update has peak bond dimension $16$; cumulative
discarded weight $<10^{-13}$): the update is numerically exact already at
$n=1$, and larger $n$ only accumulates Krylov/projector roundoff at or below
the $10^{-10}$ level. Stored zero errors are drawn as open symbols at the
$10^{-16}$ display floor.}%
"""
    (FIGURES_DIR / "fig2_single_gate_caption.tex").write_text(caption, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
