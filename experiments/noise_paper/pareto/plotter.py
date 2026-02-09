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
    # FIGURE LAYOUT (1 row, 3 columns)
    # -------------------------
    # Increase figure width to accommodate 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
    axA, axB, axAlpha = axes

    # -------------------------
    # Panel (a) and (b): Unraveling A and B
    # -------------------------
    # Store the pcm for colorbar
    pcm_AB = None 
    
    from scipy.ndimage import gaussian_filter

    for ax, Z, title, panel in [
        (axA, Z_A, "Unraveling A", "(a)"),
        (axB, Z_B, "Unraveling B", "(b)"),
    ]:
        pcm_AB = ax.pcolormesh(
            dp_edges,
            L_edges,
            Z,
            shading="auto",
            cmap="magma_r",
            vmin=vmin,
            vmax=vmax,
            rasterized=True
        )

        ax.set_xscale("log")
        ax.set_ylim(min(L_list) - 2, max(L_list) + 5)

        style_prx(ax)
        ax.set_title(title, pad=6)

        # Panel label
        ax.text(
            0.04, 0.96, panel,
            transform=ax.transAxes,
            ha="left", va="top",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        )

        # Clean y ticks at your L samples
        ax.set_yticks(L_arr)

        # Log x ticks
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.tick_params(which="minor", length=2.0)

        # Contours + labels
        # Smooth the data for contours to reduce jaggedness
        Z_smooth = gaussian_filter(Z, sigma=1.0)
        add_contours(ax, DPc, Lc, Z_smooth)

    # -------------------------
    # Panel (c): Alpha Ratio
    # -------------------------
    # Calculate ratio
    Z_B_safe = Z_B.copy()
    Z_B_safe[np.abs(Z_B_safe) < 1e-12] = np.nan
    Z_alpha = Z_A / Z_B_safe

    # Dynamic scaling for contrast
    # Use percentiles to ignore outliers and stretch contrast
    vmin_alpha = np.nanpercentile(Z_alpha, 2)
    vmax_alpha = np.nanpercentile(Z_alpha, 98)
    # Ensure reasonable bounds if data is flat
    vmin_alpha = 1.0
    vmax_alpha = 2.5

    # Plot Heatmap
    # Use 'plasma' or 'inferno' for high contrast in this range
    pcm_alpha = axAlpha.pcolormesh(
        dp_edges,
        L_edges,
        Z_alpha,
        shading="auto",
        cmap="cividis", # cividis is good for perceptual uniformity
        vmin=vmin_alpha,
        vmax=vmax_alpha,
        rasterized=True
    )
    
    axAlpha.set_xscale("log")
    style_prx(axAlpha)
    
    # Panel label
    axAlpha.text(
        0.04, 0.96, "(c)",
        transform=axAlpha.transAxes,
        ha="left", va="top",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )
    
    axAlpha.set_yticks(L_arr)
    axAlpha.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    axAlpha.xaxis.set_major_formatter(LogFormatterMathtext())
    axAlpha.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    axAlpha.tick_params(which="minor", length=2.0)

    # Add contours to Alpha plot for precision
    # Smooth alpha for better contours
    Z_alpha_smooth = gaussian_filter(Z_alpha, sigma=1.0)
    # Define contour levels based on range
    alpha_levels = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5] # np.linspace(vmin_alpha, vmax_alpha, 6)
    cs_alpha = axAlpha.contour(DPc, Lc, Z_alpha_smooth, levels=alpha_levels, colors="white", linewidths=0.8, alpha=0.8)
    axAlpha.clabel(
        cs_alpha,
        fmt=lambda v: f"{v:.2f}",
        inline=True,
        fontsize=7,
        colors="white",
    )

    # -------------------------
    # Labels
    # -------------------------
    axA.set_ylabel(r"$L$")
    for ax in axes:
        ax.set_xlabel(r"$\delta p=\gamma\,\delta t$")

    # -------------------------
    # Colorbars
    # -------------------------
    # Layout adjustment to make room
    # left, bottom, right, top, wspace, hspace
    plt.subplots_adjust(left=0.06, right=0.91, bottom=0.18, top=0.88, wspace=0.15)
    
    # 1) Colorbar for A & B (placed between B and C? or far right?)
    # Request: "first unraveling, second unraveling, their shared colorbar, heatmap alpha, its colorbar"
    # So: [A] [B] [cbarAB] [Alpha] [cbarAlpha]
    # This implies we need to adjust subplot positioning manually or use gridspec.
    # Let's try to fit cbarAB between B and Alpha, and cbarAlpha on the right.
    
    # We can use inset_axes to place colorbars explicitly relative to axes
    
    # Cbar for A & B attached to B (right side)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # However, forcing 3 subplots and squeezing a colorbar in between is tricky with standard subplots.
    # Let's restart the figure layout using GridSpec for precise control if we want that specific order.
    # Actually, standard tight_layout might not work well with "cbar in middle".
    # Let's stick to subplots and place cbarAB to the right of B, and cbarAlpha to the right of C.
    
    # Get positions
    posB = axB.get_position()
    posC = axAlpha.get_position()
    
    # Manually adjust posC to the right to make room for cbarAB?
    # Or just use the space provided by wspace.
    
    # Let's try adding axes specifically.
    
    # Cbar for A&B
    # Position: Right of B.
    # [left, bottom, width, height]
    # We'll attach it to axB
    cax_AB = axB.inset_axes([1.03, 0.0, 0.05, 1.0])
    cbar_AB = fig.colorbar(pcm_AB, cax=cax_AB)
    cbar_AB.ax.set_title(r"$\overline{\chi}_{peak}$", pad=5, fontsize=10)
    # cbar_AB.set_label(r"$\overline{\chi}_{peak}$")
    # cbar_AB.ax.yaxis.set_ticks_position('right') # Default
    
    # Cbar for Alpha
    # Position: Right of C
    cax_Alpha = axAlpha.inset_axes([1.03, 0.0, 0.05, 1.0])
    cbar_Alpha = fig.colorbar(pcm_alpha, cax=cax_Alpha)
    cbar_Alpha.ax.set_title(r"$\alpha$", pad=5, fontsize=10)
    # cbar_Alpha.set_label(r"$\alpha$")
    
    # Adjust wspace to prevent overlap between cbarAB and axC
    # The inset_axes are inside the "bbox" of the parent usually, or relative to it.
    # If we put it at x=1.03, it's outside.
    # We need enough wspace.
    plt.subplots_adjust(wspace=0.25)

    fig.savefig("large_scale.pdf")
    plt.show()

