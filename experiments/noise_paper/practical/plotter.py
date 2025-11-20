import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# --- parameters must match the data-generation script ---
dt_list = [0.005, 0.01, 0.02, 0.05, 0.1]
gamma_list = [0.5, 1, 2, 5, 10, 20, 50, 100]

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
    """
    Build a (len(gamma_list) x len(dt_list)) array of average observable values.

    Each file is expected as f"{prefix}{k}.pickle" for dt_list[k].
    Each file contains:
        results[j] == sim_params.observables for gamma_list[j],
        where sim_params.observables = [Z_obs, entropy_obs, max_bond_obs].

    which_obs:
        "max_bond"  -> use observables[2]
        "entropy"   -> use observables[1]
        "magnetization" or "Z" -> use observables[0]
    """
    grid = np.full((len(gamma_list), len(dt_list)), np.nan, dtype=float)

    obs_idx_map = {
        "Z": 0,
        "magnetization": 0,
        "entropy": 1,
        "max_bond": 2,
    }
    obs_idx = obs_idx_map[which_obs]

    for k, dt in enumerate(dt_list):
        fname = data_dir / f"{prefix}{k}.pickle"   # <-- make sure this matches your filenames
        # if your files are actually u1_practical_1.pickle, 2, ... then use:
        # fname = data_dir / f"{prefix}{k+1}.pickle"

        with open(fname, "rb") as f:
            results = pickle.load(f)  # list over gamma

        if len(results) != len(gamma_list):
            raise ValueError(f"{fname} has {len(results)} entries, expected {len(gamma_list)}")

        for j, obs_list in enumerate(results):
            if obs_list is None:
                # was skipped because γ Δt > 1
                continue

            obs = obs_list[obs_idx]  # pick Z / entropy / max_bond
            vals = np.array(obs.results)

            # average over trajectories and time; flatten in case it's 2D
            grid[j, k] = float(vals.mean())

    return grid


# --- load grid (choose which observable you want) ---
grid = load_gamma_dt_heatmap(which_obs="max_bond")   # χ_avg

# --- rescale to χ_avg * T / Δt ---
scaled_grid = np.full_like(grid, np.nan, dtype=float)
for k, dt in enumerate(dt_list):
    if dt == 0:
        continue
    scaled_grid[:, k] = grid[:, k] * T / dt

# --- plotting ---
fig, ax = plt.subplots(figsize=(5, 4))

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
ax.set_yticklabels([str(g) for g in gamma_list])

ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel(r"$\gamma$")
ax.set_title(r"$\chi_{\mathrm{avg}} \, T / \Delta t$ (num\_traj = 10)")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$\chi_{\mathrm{avg}} \, T / \Delta t$")

fig.tight_layout()
plt.show()
