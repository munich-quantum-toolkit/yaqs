# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def plot_1000L_test() -> None:
    # Overall Plot
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
        "lines.linewidth": 3,
    })
    fig = plt.figure(figsize=(7.2, 4.2))  # a size often acceptable for Nature

    gs = GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    # ax4 = fig.add_subplot(gs[1, 2])

    axes = [ax1, ax2, ax3]
    axes[2].set_xlabel("Time $(Jt)$", fontsize=12)

    axes[0].set_ylabel("Site", fontsize=12)
    axes[1].set_ylabel("Site", fontsize=12)
    axes[2].set_ylabel("Site", fontsize=12)

    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[2].set_xticks(list(range(0, 101, 10)), range(0, 11, 1))

    axes[0].set_yticks([x - 0.5 for x in [1, 25, 50, 75, 100]], [1, 250, 500, 750, 1000])
    axes[1].set_yticks([x - 0.5 for x in [1, 25, 50, 75, 100]], [1, 250, 500, 750, 1000])
    axes[2].set_yticks([x - 0.5 for x in [1, 25, 50, 75, 100]], [1, 250, 500, 750, 1000])

    axes[0].tick_params(labelsize=10)

    #########################################################################################

    data = pickle.load(open("TJM_1000L_Exact.pickle", "rb"))
    heatmap_exact = [observable.results for observable in data["sim_params"].observables]
    heatmap_exact = np.array(heatmap_exact)
    im_exact = axes[0].imshow(heatmap_exact, aspect="auto", extent=(0, 100, 100, 0), vmin=-1, vmax=1)

    data = pickle.load(open("TJM_1000L_Gamma01.pickle", "rb"))
    heatmap = [observable.results for observable in data["sim_params"].observables]
    heatmap = np.array(heatmap)
    im = axes[1].imshow(heatmap, aspect="auto", extent=(0, 100, 100, 0), vmin=-1, vmax=1)

    im = axes[2].imshow(
        heatmap - heatmap_exact, cmap="coolwarm", aspect="auto", extent=(0, 100, 100, 0), vmin=-0.1, vmax=0.1
    )

    axes[0].text(
        -0.125, 0.5, "$\\gamma = 0$", fontsize=12, transform=axes[0].transAxes, va="center", ha="center", rotation=90
    )
    axes[1].text(
        -0.125, 0.5, "$\\gamma = 0.1$", fontsize=12, transform=axes[1].transAxes, va="center", ha="center", rotation=90
    )

    fig.subplots_adjust(top=0.95, right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.42, 0.025, 0.48])
    cbar = fig.colorbar(im_exact, cax=cbar_ax)
    cbar.ax.set_title("$\\langle Z \\rangle$")

    cbar_ax = fig.add_axes([0.9, 0.12, 0.025, 0.225])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title("$\\Delta$")

    plt.savefig("results.pdf", dpi=300, format="pdf")
    plt.show()
