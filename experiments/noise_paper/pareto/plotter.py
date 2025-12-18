import numpy as np
import pickle
from pathlib import Path

L_list = range(5, 55, 5)
dps = np.logspace(-3, 0, 20)
data_dir = Path(".")  # adjust if needed

def load_heatmap(u, t, L_list=L_list, dps=dps, prefix=""):
    """
    Build a (len(L_list) x len(dps)) array of average max bond dimension.

    Files are expected as: f"{prefix}u{u}t{t}_{L}.pickle"
    Each file contains:
        results[j] == [entropy_obs, max_bond_obs]  for dp index j
    """
    grid = np.zeros((len(L_list), len(dps)), dtype=float)

    for i, L in enumerate(L_list):
        fname = data_dir / f"{prefix}u{u}_{L}.pickle"
        with open(fname, "rb") as f:
            results = pickle.load(f)  # list over dp

        if len(results) != len(dps):
            raise ValueError(f"{fname} has {len(results)} entries, expected {len(dps)}")

        for j, (z, entropy_obs, max_bond_obs) in enumerate(results):
            # max_bond_obs.results: list/array over trajectories
            vals = np.array(max_bond_obs.results)
            grid[i, j] = float(vals.mean())

    return grid

import matplotlib.pyplot as plt
import numpy as np

# Load available combinations
heat_u1_t1 = load_heatmap(u=1, t=1)  # unraveling 1, truncation 1
heat_u2_t1 = load_heatmap(u=2, t=1)  # unraveling 2, truncation 1
# heat_u1_t2 = load_heatmap(u=1, t=2)
# heat_u2_t2 = load_heatmap(u=2, t=2)

fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True, sharey=True)
axes = axes.reshape(2, 1)

panels = {
    (0, 0): (heat_u1_t1, "Pauli Jump Unraveling", "Discarded weight"),
    (1, 0): (heat_u2_t1, "Projective Unraveling", "Discarded weight"),
    # (0, 1): (heat_u1_t2, "Unraveling 1", "Relative cutoff"),
    # (1, 1): (heat_u2_t2, "Unraveling 2", "Relative cutoff"),
}

vmin = 2 # min(np.min(arr) for (arr, _, _) in panels.values())
vmax = 128 # max(np.max(arr) for (arr, _, _) in panels.values())

for (r, c), (grid, row_label, col_label) in panels.items():
    ax = axes[r, c]
    # ax.contour(grid, levels=[4, 8, 16, 32, 48], colors="white", linewidths=0.5, alpha=0.8)

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="magma_r",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(dps)))
    ax.set_xticklabels([f"{dp:.1g}" for dp in dps], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(L_list)))
    ax.set_yticklabels([str(L) for L in L_list])

    if c == 0:
        ax.set_ylabel(f"L\n({row_label})")
    # if r == 0:
    #     ax.set_title(col_label)

    for threshold in [16, 32, 48, 64]:
        # Contour expects x=dp index, y=L index
        cs = ax.contour(
            grid,
            levels=[threshold],
            colors="white",
            linewidths=1.0,
        )

        ax.clabel(cs, fmt={threshold: f"χ={threshold}"}, inline=True, fontsize=8)



for ax in axes[-1, :]:
    ax.set_xlabel("dp = γ Δt")
fig.tight_layout()

# Add at the end, replacing your existing colorbar code
fig.subplots_adjust(right=0.86)  # make room on the right

cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_title("$\\chi_{\\text{max}}$")

plt.show()
