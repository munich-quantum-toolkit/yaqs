# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Stage 5: Figure 1 - gate-local projected dynamics schematic (single column).

Visualizes the locality statement and the theory-to-algorithm bridge: an MPS
chain, a long-range gate acting on sites i and j, the fixed-rank gate window
[i, j] on which the TDVP vector field is exactly supported, the one-site halo
used by the adaptive two-site variation space, grayed-out tensors outside the
active region, and a note that single-qubit and nearest-neighbour gates are
applied directly in the hybrid implementation.

Usage:
    uv run --with matplotlib python paper_benchmarks/scripts/plot_fig1.py
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pb_common import FIGURES_DIR, SINGLE_COL_IN, apply_pra_style, save_figure

L = 12
I_SITE, J_SITE = 4, 9
CHAIN_Y = 1.05
R = 0.21

C_ACTIVE = "#D55E00"      # gate sites (vermillion, matches TDVP identity)
C_WINDOW = "#F4B183"      # window tensors
C_HALO = "#FBE3D0"        # halo tensors
C_INACTIVE = "#CFCFCF"    # frozen tensors
C_EDGE = "#5A5A5A"
C_NN = "#0072B2"


def main() -> int:
    apply_pra_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 2.15))
    ax.set_xlim(-0.75, L - 0.25)
    ax.set_ylim(-1.05, 2.35)
    ax.axis("off")

    # halo shading (window +- one site) and window shading
    ax.add_patch(Rectangle(
        (I_SITE - 1 - 0.38, CHAIN_Y - 0.46), (J_SITE + 1) - (I_SITE - 1) + 0.76, 0.92,
        facecolor=C_HALO, edgecolor="#D9A47C", linewidth=0.7, linestyle=(0, (2, 2)),
        zorder=0))
    ax.add_patch(Rectangle(
        (I_SITE - 0.38, CHAIN_Y - 0.38), J_SITE - I_SITE + 0.76, 0.76,
        facecolor="#FAD7BC", edgecolor="#C86A1F", linewidth=0.8, zorder=0.5,
    ))

    # bonds
    for k in range(L - 1):
        ax.plot([k + R, k + 1 - R], [CHAIN_Y, CHAIN_Y], color=C_EDGE,
                linewidth=1.0, zorder=1)
    # physical legs
    for k in range(L):
        ax.plot([k, k], [CHAIN_Y - R, CHAIN_Y - R - 0.17], color=C_EDGE,
                linewidth=0.8, zorder=1)

    # tensors (scatter markers stay round regardless of axis aspect)
    for k in range(L):
        if k in {I_SITE, J_SITE}:
            fc = C_ACTIVE
        elif I_SITE < k < J_SITE:
            fc = C_WINDOW
        elif k in {I_SITE - 1, J_SITE + 1}:
            fc = C_HALO
        else:
            fc = C_INACTIVE
        ec = C_EDGE if fc != C_INACTIVE else "#9A9A9A"
        ax.scatter([k], [CHAIN_Y], s=165, facecolor=fc, edgecolor=ec,
                   linewidth=0.9, zorder=2)

    # long-range gate: legs from sites i and j up to a gate box
    gate_y = 1.98
    for k in (I_SITE, J_SITE):
        ax.plot([k, k], [CHAIN_Y + R, gate_y - 0.14], color=C_ACTIVE,
                linewidth=1.2, zorder=1.5)
    xc = 0.5 * (I_SITE + J_SITE)
    ax.plot([I_SITE, J_SITE], [gate_y - 0.14, gate_y - 0.14], color=C_ACTIVE,
            linewidth=1.2, zorder=1.5)
    ax.add_patch(FancyBboxPatch(
        (xc - 1.55, gate_y - 0.16), 3.1, 0.44,
        boxstyle="round,pad=0.06,rounding_size=0.1",
        facecolor="white", edgecolor=C_ACTIVE, linewidth=1.0, zorder=3,
    ))
    ax.text(xc, gate_y + 0.06, r"$\exp[-i\theta\, P_i P_j/2]$", ha="center",
            va="center", fontsize=8, color="#8A3C00", zorder=4)
    ax.text(I_SITE - 0.34, CHAIN_Y + R + 0.16, r"$i$", ha="right", va="bottom",
            fontsize=8, color="#8A3C00")
    ax.text(J_SITE + 0.34, CHAIN_Y + R + 0.16, r"$j$", ha="left", va="bottom",
            fontsize=8, color="#8A3C00")

    # labels below the chain
    ax.annotate(
        r"gate window $[i,j]$: exact support of the"
        "\n" + r"fixed-rank TDVP vector field $P_{\mathcal{T}}(-i\,G|\psi\rangle)$",
        xy=(xc + 0.5, CHAIN_Y - 0.46), xytext=(xc + 0.35, -0.62),
        ha="center", va="center", fontsize=7,
        arrowprops={"arrowstyle": "-", "color": "#C86A1F", "linewidth": 0.7},
    )
    ax.annotate(
        "one-site halo\n(two-site updates)",
        xy=(I_SITE - 1, CHAIN_Y - 0.55), xytext=(1.15, -0.02),
        ha="center", va="center", fontsize=7,
        arrowprops={"arrowstyle": "-", "color": "#D9A47C", "linewidth": 0.7},
    )
    ax.text(10.9, 0.30, "frozen\ntensors", ha="center", va="center",
            fontsize=7, color="#7A7A7A")
    ax.annotate("", xy=(11, CHAIN_Y - 0.28), xytext=(10.9, 0.52),
                arrowprops={"arrowstyle": "-", "color": "#9A9A9A", "linewidth": 0.7})

    # direct nearest-neighbour gate indication (hybrid routing)
    nn_y = 1.72
    ax.add_patch(FancyBboxPatch(
        (0 - 0.48, nn_y - 0.13), 1.96, 0.30,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        facecolor="white", edgecolor=C_NN, linewidth=0.9, zorder=3,
    ))
    for k in (0, 1):
        ax.plot([k, k], [CHAIN_Y + R, nn_y - 0.12], color=C_NN, linewidth=1.0,
                zorder=1.5)
    ax.text(0.5, nn_y + 0.035, "NN gate", ha="center", va="center", fontsize=6.5,
            color=C_NN, zorder=4)
    ax.text(0.5, nn_y + 0.36, "applied directly\n(also 1-qubit gates)",
            ha="center", va="bottom", fontsize=6.3, color=C_NN)

    pdf, png = save_figure(fig, "fig1_gate_locality")
    print(f"wrote {pdf} and {png}")

    caption = r"""% Auto-generated by paper_benchmarks/scripts/plot_fig1.py
\newcommand{\figonecaption}{%
Gate-local projected dynamics. A long-range two-qubit rotation
$\exp[-i\theta P_iP_j/2]$ with product generator $G=(\theta/2)P_iP_j$
(generator-MPO bond dimension $D_g=1$) acts on sites $i$ and $j$ of an MPS.
The fixed-rank TDVP vector field $P_{\mathcal{T}}(-i\,G|\psi\rangle)$ is
exactly supported on the gate-spanning window $[i,j]$ (dark shading);
tensors outside the active region remain frozen. The adaptive two-site
integrator acts on the window extended by a one-site halo (light shading).
In the hybrid implementation, single-qubit and nearest-neighbour circuit
gates are applied directly, and only long-range gates are routed through the
gate-local TDVP update.}%
"""
    (FIGURES_DIR / "fig1_gate_locality_caption.tex").write_text(caption, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
