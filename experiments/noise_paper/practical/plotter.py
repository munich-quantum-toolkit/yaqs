import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# --- parameters must match the data-generation script ---
dt_list = [0.005, 0.01, 0.0125, 0.02, 0.025, 0.04, 0.05, 0.1, 0.2, 0.25]
gamma_list = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

T = 5.0  # total evolution time (fixed)

data_dir = Path(".")   # adjust if needed
prefix = "u1_practical_"  # from your data generation


def load_gamma_dt_heatmap(
    dt_list=dt_list,
    gamma_list=gamma_list,
    prefix=prefix,
    data_dir=data_dir,
    which_obs="max_bond",
):
    grid = np.full((len(gamma_list), len(dt_list)), np.nan, dtype=float)

    obs_idx_map = {
        "Z": 0,
        "magnetization": 0,
        "entropy": 1,
        "max_bond": 2,
    }
    obs_idx = obs_idx_map[which_obs]

    for k, dt in enumerate(dt_list):
        fname = data_dir / f"{prefix}{k}.pickle"

        with open(fname, "rb") as f:
            results = pickle.load(f)

        if len(results) != len(gamma_list):
            raise ValueError(f"{fname} has {len(results)} entries, expected {len(gamma_list)}")

        for j, obs_list in enumerate(results):
            if obs_list is None:
                continue

            obs = obs_list[obs_idx]
            vals = np.array(obs.results)
            grid[j, k] = float(vals.mean())

    return grid


# --- load grid ---
grid = load_gamma_dt_heatmap(which_obs="max_bond")

# --- scale to χ_avg * T / dt ---
scaled_grid = np.full_like(grid, np.nan)
for k, dt in enumerate(dt_list):
    scaled_grid[:, k] = grid[:, k] * T / dt

# -----------------------------
# PLOT WITH dp = γ Δt LINES
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 4))

vmin = np.nanmin(scaled_grid)
vmax = np.nanmax(scaled_grid)

im = ax.imshow(
    scaled_grid,
    origin="lower",
    aspect="auto",
    cmap="magma_r",
    vmin=vmin,
    vmax=vmax,
)

ax.set_xticks(np.arange(len(dt_list)))
ax.set_xticklabels([f"{dt:.3g}" for dt in dt_list])
ax.set_yticks(np.arange(len(gamma_list)))
ax.set_yticklabels([f"{g:.3g}" for g in gamma_list])

ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel(r"$\gamma$")
ax.set_title(r"$\chi_{\mathrm{avg}}\, T / \Delta t$ (num\_traj = 10)")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$\chi_{\mathrm{avg}}\, T / \Delta t$")

# ----------------------------------------------------
# Add dp = γ Δt *lines* in index coordinates
# ----------------------------------------------------
dp_levels = [1e-3, 1e-2, 1e-1, 1.0]

dt_arr = np.array(dt_list)
gamma_arr = np.array(gamma_list)

# continuous parameter for dt along the x-axis range
dt_min, dt_max = dt_arr.min(), dt_arr.max()
dt_dense = np.linspace(dt_min, dt_max, 500)

for dp in dp_levels:
    # gamma(dp, dt) = dp / dt
    gamma_dense = dp / dt_dense

    # keep only points within gamma range
    mask = (gamma_dense >= gamma_arr.min()) & (gamma_dense <= gamma_arr.max())
    if not np.any(mask):
        continue

    dt_line = dt_dense[mask]
    gamma_line = gamma_dense[mask]

    # map physical dt, gamma → imshow index coords (x=k, y=j)
    x = np.interp(dt_line, dt_arr, np.arange(len(dt_arr)))
    y = np.interp(gamma_line, gamma_arr, np.arange(len(gamma_arr)))

    ax.plot(x, y, linestyle="--", linewidth=1.0, color="white", alpha=0.9)

# optional: label one end of each line
for dp in dp_levels:
    # choose a representative dt near the left side to place label
    dt_label = dt_arr[1]  # second column to avoid edge
    gamma_label = dp / dt_label
    if gamma_label < gamma_arr.min() or gamma_label > gamma_arr.max():
        continue
    x_lab = np.interp(dt_label, dt_arr, np.arange(len(dt_arr)))
    y_lab = np.interp(gamma_label, gamma_arr, np.arange(len(gamma_arr)))
    ax.text(
        x_lab + 0.1,
        y_lab + 0.1,
        rf"$dp={dp:g}$",
        color="white",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
    )

fig.tight_layout()
plt.show()
