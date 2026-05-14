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
        fmt=lambda v: r"$\overline{\chi}_{\mathrm{max}}={int(v)}$",
        inline=True,
        fontsize=8,
        colors="white",
    )


if __name__ == "__main__":
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
    import matplotlib.patheffects as pe
    from scipy.ndimage import gaussian_filter
    from matplotlib.ticker import FixedLocator, FixedFormatter, LogLocator, NullFormatter, LogFormatterMathtext

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
    })

    # 1. Load Data
    try:
        Z_A = load_heatmap(u=1, L_list=L_list, dp_list=dp_list)
        Z_B = load_heatmap(u=2, L_list=L_list, dp_list=dp_list)
    except Exception as e:
        print(f"Error loading data: {e}. Ensure pickles exist.")
        exit(1)

    # 2. Coordinate Geometry (Bin Edges)
    def get_log_edges(centers):
        log_c = np.log10(centers)
        d_log = np.diff(log_c)
        edges_log = np.concatenate([
            [log_c[0] - d_log[0]/2],
            log_c[:-1] + d_log/2,
            [log_c[-1] + d_log[-1]/2]
        ])
        return 10**edges_log

    def get_lin_edges(centers):
        d = np.diff(centers)
        edges = np.concatenate([
            [centers[0] - d[0]/2],
            centers[:-1] + d/2,
            [centers[-1] + d[-1]/2]
        ])
        return edges

    dp_edges = get_log_edges(dp_list)
    L_edges = get_lin_edges(L_list)
    DPc, Lc = np.meshgrid(dp_list, L_list)

    # 3. Figure Layout
    fig = plt.figure(figsize=(12, 3.8))
    gs = fig.add_gridspec(
        1, 6,
        left=0.08, right=0.92, bottom=0.2, top=0.85,
        width_ratios=[1.0, 1.0, 0.04, 0.15, 1.0, 0.04],
        wspace=0.15
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    cax_chi = fig.add_subplot(gs[0, 2])
    axAlpha = fig.add_subplot(gs[0, 4])
    cax_alpha = fig.add_subplot(gs[0, 5])

    # 4. Heatmaps A & B
    all_Z = np.concatenate([Z_A.flatten(), Z_B.flatten()])
    vmin_chi = max(1.0, np.nanpercentile(all_Z, 1))
    vmax_chi = np.nanpercentile(all_Z, 99)
    norm_chi = LogNorm(vmin=vmin_chi, vmax=vmax_chi)
    
    # Cleaner pcolormesh settings (no white grid)
    pc_opts = dict(shading="flat", edgecolors="none", antialiased=False)
    
    pcmA = axA.pcolormesh(dp_edges, L_edges, Z_A, cmap="magma_r", norm=norm_chi, **pc_opts)
    pcmB = axB.pcolormesh(dp_edges, L_edges, Z_B, cmap="magma_r", norm=norm_chi, **pc_opts)

    # 5. Inflation Heatmap (Panel C)
    # Masking where Z_A or Z_B are problematic (<= 0 or NaN)
    # Goal is alpha = chi_A / chi_B to emphasize inflation in the 1.0 - 2.5 range
    Z_alpha = np.divide(Z_A, Z_B, out=np.full_like(Z_A, np.nan), where=(Z_A > 0) & (Z_B > 0))
    
    norm_al = Normalize(vmin=1.0, vmax=2)
    cmap_al = plt.get_cmap("viridis").copy()
    cmap_al.set_bad(color="0.85") # Light gray for invalid points
    
    pcm_al = axAlpha.pcolormesh(dp_edges, L_edges, Z_alpha, cmap=cmap_al, norm=norm_al, **pc_opts)

    # 6. Contours (Smoother chi lines, cleaner labels)
    chi_levels = [16, 32, 48, 64, 80]
    chi_rotations = 10  # <--- Adjust this by hand to rotate chi labels (e.g., -20)
    for ax, Z in [(axA, Z_A), (axB, Z_B)]:
        # Subtle internal grid to see bins without edge lines
        ax.grid(which="both", color="w", alpha=0.15, linewidth=0.5)
        
        Z_smooth = gaussian_filter(Z, sigma=0.8)
        cs = ax.contour(DPc, Lc, Z_smooth, levels=chi_levels, colors="white", linewidths=1.2, alpha=0.95)
        # Label more levels to be informative
        labels = ax.clabel(cs, levels=[16, 32, 48, 64, 80], fmt=lambda v: rf"$\chi={int(v)}$", 
                          inline=True, fontsize=8, colors="white", inline_spacing=2)
        if labels:
            for l in labels:
                l.set_rotation(chi_rotations)
                l.set_path_effects([pe.withStroke(linewidth=2, foreground="black", alpha=0.5)])

    alpha_levels = [1.1, 1.2, 1.3]
    
    # Smooth contours by smoothing A and B separately in log-space (keeping heatmap discrete)
    # chi_floor helps mask unstable ratio regions where signal is low
    chi_floor = 10.0
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.log(Z_A)
        B = np.log(Z_B)
        
        # Replace NaNs for smoothing
        A_f = np.nan_to_num(A, nan=np.nanmean(A) if not np.all(np.isnan(A)) else 0)
        B_f = np.nan_to_num(B, nan=np.nanmean(B) if not np.all(np.isnan(B)) else 0)
        
        A_s = gaussian_filter(A_f, sigma=(0.8, 1.0))
        B_s = gaussian_filter(B_f, sigma=(0.8, 1.0))
        Z_alpha_smooth = np.exp(A_s - B_s)
        
        # Mask unstable regions and original invalid points
        unstable_mask = (Z_A < chi_floor) | (Z_B < chi_floor) | np.isnan(Z_alpha)
        Z_alpha_smooth[unstable_mask] = np.nan

    # Anchor at alpha=1
    cs_anchor = axAlpha.contour(DPc, Lc, Z_alpha_smooth, levels=[1.0], colors="black", linewidths=1.3, zorder=6)
    axAlpha.clabel(cs_anchor, fmt={1.0: r"$\alpha=1$"}, inline=True, fontsize=8, colors="black")
    
    cs_al2 = axAlpha.contour(DPc, Lc, Z_alpha_smooth, levels=alpha_levels, colors="white", linewidths=0.8, alpha=0.8)
    labels_al = axAlpha.clabel(cs_al2, fmt=lambda v: f"{v:.1f}", inline=True, fontsize=7, colors="white")
    if labels_al:
        for l in labels_al:
            l.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black", alpha=0.4)])

    # 7. Axes & Labels
    L_loc = FixedLocator(L_list)
    L_fmt = FixedFormatter([f"{L}" for L in L_list])
    
    # Exact dp_list ticks (or subset)
    dp_loc = FixedLocator(dp_list[::2]) # subset to avoid crowding
    dp_fmt = FixedFormatter([f"{v:g}" for v in dp_list[::2]])

    for ax, label in zip([axA, axB, axAlpha], ["(a)", "(b)", "(c)"]):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\delta p = \gamma\,\delta t$")
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2,10)*0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

    axA.set_ylabel(r"System Size $L$")
    axA.yaxis.set_major_locator(L_loc)
    axA.yaxis.set_major_formatter(L_fmt)
    
    axB.tick_params(labelleft=False)
    axAlpha.tick_params(labelleft=False)
    axAlpha.yaxis.set_major_locator(L_loc)
    axAlpha.yaxis.set_major_formatter(L_fmt)

    axA.set_title("Unraveling A")
    axB.set_title("Unraveling B")
    axAlpha.set_title(r"Bond Inflation $\alpha$")

    # 8. Colorbars
    cb_chi = fig.colorbar(pcmA, cax=cax_chi)
    cb_chi.set_ticks([4, 8, 16, 32, 64])
    cb_chi.set_ticklabels(["4", "8", "16", "32", "64"])
    cax_chi.set_title(r"$\overline{\chi}_{\mathrm{max}}$", pad=12, fontsize=10)
    
    cb_al = fig.colorbar(pcm_al, cax=cax_alpha, ticks=[1.0, 1.5, 2.0, 2.5])
    cax_alpha.set_title(r"$\alpha$", pad=12, fontsize=10)

    # 9. Stats for Alpha (computed on plotted masked data)
    valid_alpha = Z_alpha[~np.isnan(Z_alpha)]
    mean_al = np.mean(valid_alpha) if valid_alpha.size > 0 else 0
    std_al = np.std(valid_alpha) if valid_alpha.size > 0 else 0
    stats_text = rf"$\mu_\alpha = {mean_al:.2f}$"+"\n"+rf"$\sigma_\alpha = {std_al:.2f}$"
    # Move to top-right corner
    axAlpha.text(0.95, 0.95, stats_text, transform=axAlpha.transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    plt.savefig("large_scale.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("large_scale.png", dpi=300, bbox_inches="tight")
    plt.show()


