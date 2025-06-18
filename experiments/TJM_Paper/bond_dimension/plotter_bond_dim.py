# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm

# right after your imports
os.chdir(os.path.dirname(__file__))


def plot_bonddimension_data() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
        "lines.linewidth": 3,
    })

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 4.3))
    # gs = GridSpec(3, 2, figure=fig)
    # ax1 = fig.add_subplot(gs[:, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, 1])
    # ax4 = fig.add_subplot(gs[2, 1])

    L = 10

    axes[0, 0].set_title("$\\chi=4$", fontsize=12)
    axes[0, 1].set_title("$\\chi=8$", fontsize=12)
    axes[0, 2].set_title("$\\chi=16$", fontsize=12)
    axes[0, 0].set_ylabel("Site", fontsize=12)
    axes[1, 0].set_ylabel("Site", fontsize=12)
    axes[2, 0].set_ylabel("Site", fontsize=12)
    axes[0, 0].set_yticks([x - 0.5 for x in list(range(2, L + 2, 2))], range(2, L + 2, 2))
    axes[1, 0].set_yticks([x - 0.5 for x in list(range(2, L + 2, 2))], range(2, L + 2, 2))
    axes[2, 0].set_yticks([x - 0.5 for x in list(range(2, L + 2, 2))], range(2, L + 2, 2))
    axes[2, 0].set_xlabel("Time ($Jt$)", fontsize=12)
    axes[2, 1].set_xlabel("Time ($Jt$)", fontsize=12)
    axes[2, 2].set_xlabel("Time ($Jt$)", fontsize=12)

    axes[0, 0].set_xticks([])
    axes[0, 1].set_xticks([])
    axes[0, 2].set_xticks([])
    axes[1, 0].set_xticks([])
    axes[1, 1].set_xticks([])
    axes[1, 2].set_xticks([])
    axes[0, 1].set_yticks([])
    axes[0, 2].set_yticks([])
    axes[1, 1].set_yticks([])
    axes[1, 2].set_yticks([])
    axes[2, 1].set_yticks([])
    axes[2, 2].set_yticks([])

    ##########################################################################################################
    L = 10
    data = pickle.load(open("QuTip_exact_convergence.pickle", "rb"))
    heatmap_exact = [data["observables"][site] for site in range(L)]

    num_samples = 1000
    norm = LogNorm(vmin=1e-3, vmax=1e-1)

    cmap = cm.coolwarm

    data = pickle.load(open("TJM_convergence_Bond4.pickle", "rb"))
    heatmap1000 = [observable.results for observable in data["sim_params"].observables]

    heatmap1000 = np.array(heatmap1000)
    heatmap_exact = np.array(heatmap_exact)

    trajectories = 100
    error_heatmap = []
    for _ in range(num_samples):
        indices = np.random.choice(data["sim_params"].observables[0].trajectories.shape[0], trajectories, replace=False)
        heatmap = []
        for site in range(L):
            samples = data["sim_params"].observables[site].trajectories[indices]
            heatmap.append(np.mean(samples, axis=0))
        heatmap = np.array(heatmap)
        error_heatmap.append(np.abs(heatmap_exact - heatmap))
    error_heatmap = np.array(error_heatmap)
    error_heatmap = np.mean(error_heatmap, axis=0)
    im = axes[0, 0].imshow(
        error_heatmap, cmap=cmap, aspect="auto", extent=[0, data["sim_params"].elapsed_time, L, 0], norm=norm
    )

    trajectories = 1000
    error_heatmap = []
    for _ in range(num_samples):
        indices = np.random.choice(data["sim_params"].observables[0].trajectories.shape[0], trajectories, replace=False)
        heatmap = []
        for site in range(L):
            samples = data["sim_params"].observables[site].trajectories[indices]
            heatmap.append(np.mean(samples, axis=0))
        heatmap = np.array(heatmap)
        error_heatmap.append(np.abs(heatmap_exact - heatmap))
    error_heatmap = np.array(error_heatmap)
    error_heatmap = np.mean(error_heatmap, axis=0)
    im = axes[1, 0].imshow(
        error_heatmap, cmap=cmap, aspect="auto", extent=[0, data["sim_params"].elapsed_time, L, 0], norm=norm
    )

    trajectories = 9999
    error_heatmap = []
    for _ in range(num_samples):
        indices = np.random.choice(data["sim_params"].observables[0].trajectories.shape[0], trajectories, replace=False)
        heatmap = []
        for site in range(L):
            samples = data["sim_params"].observables[site].trajectories[indices]
            heatmap.append(np.mean(samples, axis=0))
        heatmap = np.array(heatmap)
        error_heatmap.append(np.abs(heatmap_exact - heatmap))
    error_heatmap = np.array(error_heatmap)
    error_heatmap = np.mean(error_heatmap, axis=0)
    im = axes[2, 0].imshow(
        error_heatmap, cmap=cmap, aspect="auto", extent=[0, data["sim_params"].elapsed_time, L, 0], norm=norm
    )

    data = pickle.load(open("TJM_convergence_Bond8.pickle", "rb"))
    heatmap1000 = []
    for observable in data["sim_params"].observables:
        heatmap1000.append(observable.results)

    heatmap1000 = np.array(heatmap1000)
    heatmap_exact = np.array(heatmap_exact)

    trajectories = 100
    error_heatmap = []
    for _ in range(num_samples):
        indices = np.random.choice(data["sim_params"].observables[0].trajectories.shape[0], trajectories, replace=False)
        heatmap = []
        for site in range(L):
            samples = data["sim_params"].observables[site].trajectories[indices]
            heatmap.append(np.mean(samples, axis=0))
        heatmap = np.array(heatmap)
        error_heatmap.append(np.abs(heatmap_exact - heatmap))
    error_heatmap = np.array(error_heatmap)
    error_heatmap = np.mean(error_heatmap, axis=0)
    im = axes[0, 1].imshow(
        error_heatmap, cmap=cmap, aspect="auto", extent=[0, data["sim_params"].elapsed_time, L, 0], norm=norm
    )

    trajectories = 1000
    error_heatmap = []
    for _ in range(num_samples):
        indices = np.random.choice(data["sim_params"].observables[0].trajectories.shape[0], trajectories, replace=False)
        heatmap = []
        for site in range(L):
            samples = data["sim_params"].observables[site].trajectories[indices]
            heatmap.append(np.mean(samples, axis=0))
        heatmap = np.array(heatmap)
        error_heatmap.append(np.abs(heatmap_exact - heatmap))
    error_heatmap = np.array(error_heatmap)
    error_heatmap = np.mean(error_heatmap, axis=0)
    im = axes[1, 1].imshow(
        error_heatmap, cmap=cmap, aspect="auto", extent=[0, data["sim_params"].elapsed_time, L, 0], norm=norm
    )

    trajectories = 9999
    error_heatmap = []
    for _ in range(num_samples):
        indices = np.random.choice(data["sim_params"].observables[0].trajectories.shape[0], trajectories, replace=False)
        heatmap = []
        for site in range(L):
            samples = data["sim_params"].observables[site].trajectories[indices]
            heatmap.append(np.mean(samples, axis=0))
        heatmap = np.array(heatmap)
        error_heatmap.append(np.abs(heatmap_exact - heatmap))
    error_heatmap = np.array(error_heatmap)
    error_heatmap = np.mean(error_heatmap, axis=0)
    im = axes[2, 1].imshow(
        error_heatmap, cmap=cmap, aspect="auto", extent=[0, data["sim_params"].elapsed_time, L, 0], norm=norm
    )

    data = pickle.load(open("TJM_convergence_Bond16.pickle", "rb"))
    heatmap1000 = []
    for observable in data["sim_params"].observables:
        heatmap1000.append(observable.results)

    heatmap1000 = np.array(heatmap1000)
    heatmap_exact = np.array(heatmap_exact)
    # im = axes[2, 2].imshow(np.abs(heatmap_exact-heatmap1000), cmap=cmap, aspect='auto', extent=[0, data['sim_params'].elapsed_time, L, 0], norm=norm)

    trajectories = 100
    error_heatmap = []
    for _ in range(num_samples):
        indices = np.random.choice(data["sim_params"].observables[0].trajectories.shape[0], trajectories, replace=False)
        heatmap = []
        for site in range(L):
            samples = data["sim_params"].observables[site].trajectories[indices]
            heatmap.append(np.mean(samples, axis=0))
        heatmap = np.array(heatmap)
        error_heatmap.append(np.abs(heatmap_exact - heatmap))
    error_heatmap = np.array(error_heatmap)
    error_heatmap = np.mean(error_heatmap, axis=0)
    im = axes[0, 2].imshow(
        error_heatmap, cmap=cmap, aspect="auto", extent=[0, data["sim_params"].elapsed_time, L, 0], norm=norm
    )

    trajectories = 1000
    error_heatmap = []
    for _ in range(num_samples):
        indices = np.random.choice(data["sim_params"].observables[0].trajectories.shape[0], trajectories, replace=False)
        heatmap = []
        for site in range(L):
            samples = data["sim_params"].observables[site].trajectories[indices]
            heatmap.append(np.mean(samples, axis=0))
        heatmap = np.array(heatmap)
        error_heatmap.append(np.abs(heatmap_exact - heatmap))
    error_heatmap = np.array(error_heatmap)
    error_heatmap = np.mean(error_heatmap, axis=0)
    im = axes[1, 2].imshow(
        error_heatmap, cmap=cmap, aspect="auto", extent=[0, data["sim_params"].elapsed_time, L, 0], norm=norm
    )

    trajectories = 9999
    error_heatmap = []
    for _ in range(num_samples):
        indices = np.random.choice(data["sim_params"].observables[0].trajectories.shape[0], trajectories, replace=False)
        heatmap = []
        for site in range(L):
            samples = data["sim_params"].observables[site].trajectories[indices]
            heatmap.append(np.mean(samples, axis=0))
        heatmap = np.array(heatmap)
        error_heatmap.append(np.abs(heatmap_exact - heatmap))
    error_heatmap = np.array(error_heatmap)
    error_heatmap = np.mean(error_heatmap, axis=0)
    im = axes[2, 2].imshow(
        error_heatmap, cmap=cmap, aspect="auto", extent=[0, data["sim_params"].elapsed_time, L, 0], norm=norm
    )

    # Adjust the layout to make room for vertically rotated labels
    fig.subplots_adjust(left=0.1, right=0.875, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)

    # Add vertical annotations on the left side above the "Site" label
    axes[0, 0].text(
        -0.4, 0.5, "$N=100$", fontsize=12, transform=axes[0, 0].transAxes, va="center", ha="center", rotation=90
    )
    axes[1, 0].text(
        -0.4, 0.5, "$N=1000$", fontsize=12, transform=axes[1, 0].transAxes, va="center", ha="center", rotation=90
    )
    axes[2, 0].text(
        -0.4, 0.5, "$N=10000$", fontsize=12, transform=axes[2, 0].transAxes, va="center", ha="center", rotation=90
    )

    cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.75])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title("$\\epsilon$")

    plt.savefig("results.pdf", dpi=300, format="pdf")
    plt.show()


if __name__ == "__main__":
    plot_bonddimension_data()
