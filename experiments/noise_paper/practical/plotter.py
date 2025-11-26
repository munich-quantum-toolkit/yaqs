import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# --- parameters must match the data-generation scripts ---
dt_list = [0.01, 0.0125, 0.02, 0.025, 0.04, 0.05, 0.1, 0.2, 0.25]

# low-gamma data lives in separate files
low_gamma_list = [0.02, 0.05]
high_gamma_list = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

gamma_list = low_gamma_list + high_gamma_list  # combined list

T = 5.0  # total evolution time (fixed)

data_dir = Path(".")   # adjust if needed
prefix_high = "u1_practical_"          # original data
prefix_low = "u1_low_gamma_practical_" # new low-gamma data


def load_gamma_dt_heatmap(
    dt_list=dt_list,
    gamma_list_full=gamma_list,
    prefix_high=prefix_high,
    prefix_low=prefix_low,
    low_gamma_list=low_gamma_list,
    high_gamma_list=high_gamma_list,
    data_dir=data_dir,
    which_obs="max_bond",
):
    """
    Build a (len(gamma_list_full) x len(dt_list)) array of average observable values.

    For each dt index k:
      - low-gamma data is read from  prefix_low + f"{k}.pickle"
      - high-gamma data is read from prefix_high + f"{k}.pickle"

    Each pickle is expected to contain a list over the corresponding gamma set
    (low or high), ordered as in low_gamma_list / high_gamma_list.
    """
    grid = np.full((len(gamma_list_full), len(dt_list)), np.nan, dtype=float)

    # map gamma value -> row index in the final grid
    gamma_to_idx = {g: i for i, g in enumerate(gamma_list_full)}

    obs_idx_map = {
        "Z": 0,
        "magnetization": 0,
        "entropy": 1,
        "max_bond": 2,
    }
    obs_idx = obs_idx_map[which_obs]

    for k, dt in enumerate(dt_list):
        # --------- load LOW-gamma file (if it exists) ----------
        fname_low = data_dir / f"{prefix_low}{k}.pickle"
        if fname_low.exists():
            with open(fname_low, "rb") as f:
                results_low = pickle.load(f)

            if len(results_low) != len(low_gamma_list):
                raise ValueError(
                    f"{fname_low} has {len(results_low)} entries, "
                    f"expected {len(low_gamma_list)}"
                )

            for j, obs_list in enumerate(results_low):
                if obs_list is None:
                    continue
                obs = obs_list[obs_idx]
                vals = np.array(obs.results)
                g = low_gamma_list[j]
                row = gamma_to_idx[g]
                grid[row, k] = float(vals.mean())

        # --------- load HIGH-gamma file (if it exists) ----------
        fname_high = data_dir / f"{prefix_high}{k}.pickle"
        if fname_high.exists():
            with open(fname_high, "rb") as f:
                results_high = pickle.load(f)

            if len(results_high) != len(high_gamma_list):
                raise ValueError(
                    f"{fname_high} has {len(results_high)} entries, "
                    f"expected {len(high_gamma_list)}"
                )

            for j, obs_list in enumerate(results_high):
                if obs_list is None:
                    continue
                obs = obs_list[obs_idx]
                vals = np.array(obs.results)
                g = high_gamma_list[j]
                row = gamma_to_idx[g]
                grid[row, k] = float(vals.mean())

    return grid


# --- load grid ---
grid = load_gamma_dt_heatmap(which_obs="max_bond")

# --- scale to χ_avg * T / dt ---
scaled_grid = np.full_like(grid, np.nan)
for k, dt in enumerate(dt_list):
    scaled_grid[:, k] = grid[:, k] # * T / dt

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

ax.set_xlabel(r"$dt$")
ax.set_ylabel(r"$\gamma$")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$\chi_{\mathrm{avg}}$")

# ----------------------------------------------------
# Add dp = γ Δt *lines* in index coordinates
# ----------------------------------------------------
dp_levels = [1e-3, 1e-2, 1e-1, 1.0]

dt_arr = np.array(dt_list)
gamma_arr = np.array(gamma_list)

dt_min, dt_max = dt_arr.min(), dt_arr.max()
dt_dense = np.linspace(dt_min, dt_max, 500)

for dp in dp_levels:
    gamma_dense = dp / dt_dense

    mask = (gamma_dense >= gamma_arr.min()) & (gamma_dense <= gamma_arr.max())
    if not np.any(mask):
        continue

    dt_line = dt_dense[mask]
    gamma_line = gamma_dense[mask]

    x = np.interp(dt_line, dt_arr, np.arange(len(dt_arr)))
    y = np.interp(gamma_line, gamma_arr, np.arange(len(gamma_arr)))

    ax.plot(x, y, linestyle="--", linewidth=1.0, color="black", alpha=0.9)

for dp in dp_levels:
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
