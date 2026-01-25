from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext


# -------------------------
# Data
# -------------------------
L_list = [5, 10, 15, 20, 25, 30, 40, 60, 80]
dp_list = np.logspace(-3, 0, 20)
data_dir = Path(".")


def bin_edges_from_centers(x: np.ndarray) -> np.ndarray:
    """Edges for pcolormesh from strictly increasing centers."""
    x = np.asarray(x, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("Centers must be strictly increasing.")

    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] * (x[0] / x[1])
    edges[-1] = x[-1] * (x[-1] / x[-2])
    return edges


def load_heatmap(*, u: int, L_list: Sequence[int], dp_list: np.ndarray, prefix: str = "") -> np.ndarray:
    """
    grid[i, j] = mean chi_max over trajectories for L=L_list[i], dp=dp_list[j].

    Expects: f"{prefix}u{u}_{L}.pickle"
    Each results[j] is something like [entropy_obs, max_bond_obs] or similar.
    !!! IMPORTANT: set `CHI_INDEX` correctly for your stored structure.
    """
    CHI_INDEX = 0  # <-- change to 1 if your max-bond observable is stored at index 1

    grid = np.zeros((len(L_list), len(dp_list)), dtype=float)
    for i, L in enumerate(L_list):
        fname = data_dir / f"{prefix}u{u}_{L}.pickle"
        with open(fname, "rb") as f:
            results = pickle.load(f)

        if len(results) != len(dp_list):
            raise ValueError(f"{fname} has {len(results)} entries, expected {len(dp_list)}")

        for j, entry in enumerate(results):
            obs = entry[CHI_INDEX]
            vals = np.asarray(obs.results, dtype=float)
            grid[i, j] = float(np.mean(vals))

    return grid


def style_prx(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3.0, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_contours(ax: plt.Axes, DPc: np.ndarray, Lc: np.ndarray, Z: np.ndarray) -> None:
    levels = [16, 32, 48, 64, 80]
    cs = ax.contour(DPc, Lc, Z, levels=levels, colors="white", linewidths=1.0, alpha=0.95)
    ax.clabel(
        cs,
        fmt=lambda v: rf"$\chi={int(v)}$",
        inline=True,
        fontsize=8,
        colors="white",
    )


if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
    })

    # Load
    Z_A = load_heatmap(u=1, L_list=L_list, dp_list=dp_list)  # Unraveling A
    Z_B = load_heatmap(u=2, L_list=L_list, dp_list=dp_list)  # Unraveling B

    # Shared color scale
    vmin = 2.0
    vmax = float(max(Z_A.max(), Z_B.max()))

    # Geometry for pcolormesh
    dp_edges = bin_edges_from_centers(dp_list)
    L_arr = np.asarray(L_list, dtype=float)
    L_edges = bin_edges_from_centers(L_arr)

    # Centers for contour plotting
    DPc, Lc = np.meshgrid(dp_list, L_arr)

    # -------------------------
    # FIGURE* LAYOUT (side-by-side)
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    axA, axB = axes

    for ax, Z, title, panel in [
        (axA, Z_A, "Unraveling A", "(a)"),
        (axB, Z_B, "Unraveling B", "(b)"),
    ]:
        pcm = ax.pcolormesh(
            dp_edges,
            L_edges,
            Z,
            shading="auto",
            cmap="magma_r",
            vmin=vmin,
            vmax=vmax,
        )

        # x is log, y is LINEAR (as you requested)
        ax.set_xscale("log")
        ax.set_ylim(min(L_list) - 2, max(L_list) + 5)

        style_prx(ax)
        ax.set_title(title, pad=4)

        # Panel label
        ax.text(
            0.02, 0.98, panel,
            transform=ax.transAxes,
            ha="left", va="top",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        )

        # Clean y ticks at your L samples
        ax.set_yticks(L_arr)

        # Log x ticks: only decades as majors
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.tick_params(which="minor", length=2.0)

        # Contours + labels
        add_contours(ax, DPc, Lc, Z)
    # --- after plotting both panels ---
    axA.set_ylabel(r"$L$")
    axA.set_xlabel(r"$\delta p=\gamma\,\delta t$")
    axB.set_xlabel(r"$\delta p=\gamma\,\delta t$")

    # -------------------------
    # Layout tuning (reduce left whitespace, reserve space for cbar)
    # -------------------------
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.18, top=0.86, wspace=0.10)

    # Dedicated colorbar axis (outside the panels)
    cbar_ax = fig.add_axes([0.92, 0.22, 0.02, 0.58])
    cbar = fig.colorbar(pcm, cax=cbar_ax)

    # Put chi as the colorbar title instead of a side label
    cbar.set_label("")  # ensure no side label
    cbar_ax.set_title(r"$\overline{\chi}_{peak}$", pad=6)

    fig.savefig("large_scale.pdf")
    plt.show()

