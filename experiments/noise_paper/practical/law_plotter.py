from __future__ import annotations

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# ------------------------
# Data + parameters (same)
# ------------------------
dt_list = [0.01, 0.0125, 0.02, 0.025, 0.04, 0.05, 0.1]
gamma_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
data_dir = Path(".")

dt = np.array(dt_list, dtype=float)
g = np.array(gamma_list, dtype=float)

# Choose which fixed gammas to slice (edit freely)
gamma_slices = [0.02, 0.1, 0.5, 2.0]

# Optional: show constant-dp guides on the chi-vs-dt plot
show_dp_guides = True
dp_levels = [1e-3, 1e-2, 1e-1, 1.0]


# ------------------------
# Loader (same logic as your heatmap code)
# ------------------------
def load_gamma_dt_heatmaps_for_u(u_tag: str):
    bond_grid = np.full((len(gamma_list), len(dt_list)), np.nan, dtype=float)
    time_grid = np.full_like(bond_grid, np.nan)

    gamma_to_idx = {gg: i for i, gg in enumerate(gamma_list)}

    for k, _dt in enumerate(dt_list):
        fname = data_dir / f"practical_{u_tag}_{k}.pickle"
        if not fname.exists():
            continue

        with open(fname, "rb") as f:
            results = pickle.load(f)

        if len(results) != len(gamma_list):
            raise ValueError(f"{fname} has {len(results)} entries, expected {len(gamma_list)}")

        for j, obs_list in enumerate(results):
            if obs_list is None:
                continue

            # max bond dimension over time
            obs_bond = obs_list[0]
            vals = np.asarray(obs_bond[0].results, dtype=float)
            row = gamma_to_idx[gamma_list[j]]
            bond_grid[row, k] = float(np.max(vals))

            # wall time per trajectory (kept for completeness)
            time_grid[row, k] = float(np.asarray(obs_list[1]))

    return bond_grid, time_grid


# ------------------------
# Helpers
# ------------------------
def style_prx(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3.0, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def nearest_gamma_idx(gamma_values: np.ndarray, target: float) -> tuple[int, float]:
    gamma_values = np.asarray(gamma_values, dtype=float)
    idx = int(np.argmin(np.abs(gamma_values - target)))
    return idx, float(gamma_values[idx])


def panel_label(ax, s: str) -> None:
    t = ax.text(
        0.02, 0.98, s, transform=ax.transAxes,
        ha="left", va="top", fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2)
    )
    t.set_path_effects([pe.withStroke(linewidth=1.3, foreground="white")])


def add_dp_guides(ax: plt.Axes, *, gamma: float, dt: np.ndarray, dp_levels: list[float]) -> None:
    """
    At fixed gamma, constant dp corresponds to dt = dp/gamma.
    We draw vertical guide lines at those dt values (if they fall within dt-range).
    """
    dt_min, dt_max = float(np.min(dt)), float(np.max(dt))
    for dp in dp_levels:
        dt_star = dp / gamma
        if dt_min <= dt_star <= dt_max:
            ax.axvline(dt_star, ls="--", lw=0.9, alpha=0.35, color="k")


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
    })

    u1_bond, _ = load_gamma_dt_heatmaps_for_u("u1")
    u2_bond, _ = load_gamma_dt_heatmaps_for_u("u2")

    # Figure: two panels
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    axA, axB = axes

    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "D", "^"]

    # For each gamma slice, plot chi vs dt for both unravelings
    for k, gamma_target in enumerate(gamma_slices):
        gi, gamma_used = nearest_gamma_idx(g, gamma_target)
        ls = linestyles[k % len(linestyles)]
        mk = markers[k % len(markers)]

        y1 = u1_bond[gi, :]
        y2 = u2_bond[gi, :]

        axA.plot(dt, y1, linestyle=ls, marker=mk, markersize=4.5, linewidth=1.2,
                 label=rf"$\gamma \approx {gamma_used:g}$")
        axB.plot(dt, y2, linestyle=ls, marker=mk, markersize=4.5, linewidth=1.2,
                 label=rf"$\gamma \approx {gamma_used:g}$")

        # Optional constant-dp guides (vertical lines) to show where dp would be fixed
        if show_dp_guides:
            add_dp_guides(axA, gamma=gamma_used, dt=dt, dp_levels=dp_levels)
            add_dp_guides(axB, gamma=gamma_used, dt=dt, dp_levels=dp_levels)

    # Cosmetics
    for ax, title, panel in [(axA, "Unraveling A", "(a)"), (axB, "Unraveling B", "(b)")]:
        style_prx(ax)
        ax.set_title(title, pad=4)
        ax.set_xlabel(r"$\Delta t$")
        ax.set_xticks(dt)
        ax.set_xticklabels([f"{x:g}" for x in dt_list], rotation=0, ha="center")
        panel_label(ax, panel)

    axA.set_ylabel(r"$\chi_{\max}$")

    # Legend once
    axB.legend(frameon=False, fontsize=8, loc="upper left")

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.22, top=0.86, wspace=0.18)
    fig.savefig("chi_vs_dt_fixed_gamma.pdf", dpi=300)
    plt.show()
