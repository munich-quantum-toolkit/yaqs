from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Data (same as your heatmap script)
# -------------------------
L_list = [5, 10, 15, 20, 25, 30, 40, 60, 80]
dp_list = np.logspace(-3, 0, 20)
data_dir = Path(".")


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


def nearest_dp_indices(dp_grid: np.ndarray, targets: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each target dp, return nearest index and the actual dp value used.
    """
    dp_grid = np.asarray(dp_grid, dtype=float)
    t = np.asarray(list(targets), dtype=float)

    idx = np.array([int(np.argmin(np.abs(dp_grid - x))) for x in t], dtype=int)
    used = dp_grid[idx]
    return idx, used


def power_law_fit(L: np.ndarray, chi: np.ndarray) -> Tuple[float, float]:
    """
    Fit chi ~ a * L^b via log-log least squares (ignoring non-positive entries).
    Returns (a, b).
    """
    L = np.asarray(L, dtype=float)
    chi = np.asarray(chi, dtype=float)

    m = (L > 0) & (chi > 0)
    if np.count_nonzero(m) < 2:
        return np.nan, np.nan

    x = np.log(L[m])
    y = np.log(chi[m])
    b, loga = np.polyfit(x, y, 1)
    a = float(np.exp(loga))
    return a, float(b)


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

    # -------------------------
    # Load the same grids as the heatmap script
    # -------------------------
    Z_A = load_heatmap(u=1, L_list=L_list, dp_list=dp_list)  # Unraveling A
    Z_B = load_heatmap(u=2, L_list=L_list, dp_list=dp_list)  # Unraveling B

    L_arr = np.asarray(L_list, dtype=float)

    # Only use L > 15  (with your L_list this means L >= 20)
    mask_L = L_arr > 1
    L_sel = L_arr[mask_L]

    # Targets (requested)
    dp_targets = np.array([1e-3, 1e-2, 1e-1, 1e0], dtype=float)
    dp_idx, dp_used = nearest_dp_indices(dp_list, dp_targets)

    print("Requested dp targets:", dp_targets)
    print("Nearest dp grid used:", dp_used)
    if np.any(np.abs(np.log10(dp_used / dp_targets)) > 1e-6):
        print("NOTE: Some requested dp values are not exactly on dp_list; nearest values were used.")

    # Extract slices: shape (n_dp, n_L_selected)
    A_slices = np.stack([Z_A[mask_L, j] for j in dp_idx], axis=0)
    B_slices = np.stack([Z_B[mask_L, j] for j in dp_idx], axis=0)

    # -------------------------
    # Plot: 2 panels (A/B), 4 curves (dp values) each
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    axA, axB = axes

    # Use a clean set of linestyles so we don’t rely on color alone
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^"]

    for ax, slices, title, panel in [
        (axA, A_slices, "Unraveling A", "(a)"),
        (axB, B_slices, "Unraveling B", "(b)"),
    ]:
        for k, (chi_vals, dp_val, ls, mk) in enumerate(zip(slices, dp_used, linestyles, markers)):
            ax.plot(
                L_sel,
                chi_vals,
                linestyle=ls,
                marker=mk,
                markersize=4.5,
                linewidth=1.2,
                label=rf"$dp \approx {dp_val:.2g}$",
            )

            # Optional quick power-law fit on the shown points (prints to console)
            a, b = power_law_fit(L_sel, chi_vals)
            if np.isfinite(b):
                print(f"{title}: dp≈{dp_val:.2g}  fit chi ~ a*L^b  =>  b={b:.3f}, a={a:.3g}")

        style_prx(ax)
        ax.set_title(title, pad=4)
        ax.set_xlabel(r"$L$")

        ax.text(
            0.02, 0.98, panel,
            transform=ax.transAxes,
            ha="left", va="top",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        )

        # nice discrete ticks at your sample points
        ax.set_xticks(L_sel)

    axA.set_ylabel(r"$\overline{\chi}_{peak}$")
    axB.legend(frameon=False, fontsize=8, loc="upper left")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.86, wspace=0.18)
    fig.savefig("chi_vs_L_slices.pdf")
    plt.show()
