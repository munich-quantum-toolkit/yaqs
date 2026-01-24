from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Experiment metadata
# -------------------------
DATA_DIR = Path(".")
L_LIST = [5, 10, 15, 20, 25, 30, 40, 60, 80]
DP_VALUES = [1e-3, 1e-2, 1e-1, 1e0]   # exactly what you saved
T_LIST = [2, 5, 8]


def style_prx(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3.0, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_law_grid(*, u: int, T: int | float, L_list: Sequence[int], dp_values: Sequence[float]) -> np.ndarray:
    """
    Load grid[i, j] = mean(max_bond) for L=L_list[i], dp=dp_values[j]
    from files: u{u}_law_T{T}_L{L}.pickle

    Each pickle contains a list over dp_values, where each entry is a list of Observables.
    We assume the first observable is Observable("max_bond") as in your runner.
    """
    grid = np.zeros((len(L_list), len(dp_values)), dtype=float)

    for i, L in enumerate(L_list):
        fname = DATA_DIR / f"u{u}_law_T{T}_L{L}.pickle"
        if not fname.exists():
            raise FileNotFoundError(f"Missing file: {fname}")

        with open(fname, "rb") as f:
            results = pickle.load(f)

        if len(results) != len(dp_values):
            raise ValueError(f"{fname} has {len(results)} dp entries, expected {len(dp_values)}")

        for j, obs_list in enumerate(results):
            # obs_list is whatever tdvp_simulator returned; in your code it's `sim_params.observables`
            # and you set measurements=[Observable("max_bond")] so index 0 is max_bond.
            max_bond_obs = obs_list[0]
            vals = np.asarray(max_bond_obs.results, dtype=float)
            grid[i, j] = float(np.mean(vals))

    return grid


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

    L_arr = np.asarray(L_LIST, dtype=float)
    dp_arr = np.asarray(DP_VALUES, dtype=float)

    # Optional: drop tiny-L region if you want (set to 15 or 20)
    L_MIN = 5
    mask_L = L_arr >= L_MIN
    L_sel = L_arr[mask_L]

    # Line styles (don’t rely only on color)
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^"]

    # -------------------------
    # 2x3 layout: rows=unravelings, cols=T
    # -------------------------
    fig, axes = plt.subplots(
        2, len(T_LIST),
        figsize=(7.2, 4.2),
        sharex=True, sharey="row",
    )

    # Collect legend handles once
    legend_handles = None
    legend_labels = None

    for col, T in enumerate(T_LIST):
        # load grids for this T
        ZA = load_law_grid(u=1, T=T, L_list=L_LIST, dp_values=DP_VALUES)
        ZB = load_law_grid(u=2, T=T, L_list=L_LIST, dp_values=DP_VALUES)

        for row, (Z, title) in enumerate([(ZA, "Unraveling A"), (ZB, "Unraveling B")]):
            ax = axes[row, col]

            for k, (dp, ls, mk) in enumerate(zip(dp_arr, linestyles, markers)):
                y = Z[mask_L, k]
                (line,) = ax.plot(
                    L_sel, y,
                    linestyle=ls,
                    marker=mk,
                    markersize=4.5,
                    linewidth=1.2,
                    label=rf"$dp = {dp:.0e}$",
                )

            style_prx(ax)

            if row == 0:
                ax.set_title(rf"$T={T}$", pad=4)

            if col == 0:
                ax.set_ylabel(r"$\overline{\chi}_{\mathrm{peak}}$" + "\n" + title)

            ax.set_xticks(L_sel)

            # grab legend entries from the first axis only
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    # Common x label
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$L$")

    # One shared legend at top center
    fig.legend(
        legend_handles, legend_labels,
        ncol=4,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
        handlelength=2.5,
        columnspacing=1.5,
    )

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.12, top=0.86, wspace=0.18, hspace=0.22)
    fig.savefig("chi_vs_L_slices_Tscan.pdf")
    plt.show()
