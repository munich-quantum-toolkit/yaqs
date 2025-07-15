import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# ---- INPUT FILES ----
file_names = [
    "slimmed_reduced_tjm_results_L100_T100000_factor5_25trajectories_leonhard-12_pack0.pkl",
    "slimmed_reduced_tjm_results_L100_T100000_factor5_25trajectories_leonhard-14_pack0.pkl",
    "slimmed_reduced_tjm_results_L100_T100000_factor5_25trajectories_leonhard-15_pack0.pkl",
    "slimmed_reduced_tjm_results_L100_T100000_factor5_25trajectories_leonhard-15_pack1.pkl"
]

# ---- PLOT STYLE ----
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 1,
    "legend.fontsize": 10,
})

def steady_state(L):
    return np.cos(np.pi * np.arange(L) / (L - 1))

# ---- LOAD DATA ----
all_runs = []
times = None
for fn in file_names:
    with open(fn, "rb") as f:
        data = pickle.load(f)
    if times is None:
        times = np.array(data["times"])
    obs = data["sim_params"].observables
    runs = [np.array(o.results) for o in obs]
    all_runs.append(runs)

all_runs = np.array(all_runs)              # shape (n_files, L, T)
mean_Z   = all_runs.mean(axis=0)           # shape (L, T)
Z_ss     = steady_state(mean_Z.shape[0])   # length L
Z_ss = Z_ss[::-1]
# ---- SUBSET AND COLORS ----
sites = list(range(0, 100, 5))
cmap  = plt.cm.viridis(np.linspace(0, 1, 100))

# ---- MAKE FIGURE (one-column width ~3.5") ----
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(3.5, 6),
    sharex=True,
    gridspec_kw={'hspace': 0.1}     # ← almost zero vertical gap
)

# (a) LEFT: dynamics + steady lines
for j in sites:
    ax1.plot(times, mean_Z[j], color=cmap[j], alpha=1)
    ax1.hlines(
        Z_ss[j], times[0], times[-1],
        linestyles='--', color='k', alpha=0.4, linewidth=1
    )

ax1.text(0.9, 0.97, "\\textbf{(a)}", transform=ax1.transAxes,
         fontweight="bold", va="top")
ax1.set_ylabel("$\\langle Z_j\\rangle$")
ax1.grid(alpha=0.2)
ax1.tick_params(labelbottom=False)

# (b) RIGHT: absolute error
t1, t2 = 1, 100000
i1 = np.searchsorted(times, t1)
i2 = np.searchsorted(times, t2)
tc = times[i1:i2]

# ---- SUBSET AND COLORS ----
sites = list(range(0, 100))
cmap  = plt.cm.viridis(np.linspace(0, 1, 100))
for j in sites:
    err = np.abs(mean_Z[j, i1:i2] - Z_ss[j])
    ax2.plot(tc[0:-1:10000], err[0:-1:10000], color=cmap[j], alpha=1)

ax2.hlines(0.01, tc[0], tc[-1], linestyles='--', color='k', linewidth=1)
ax2.set_yscale('log')
ax2.text(0.9, 0.97, "\\textbf{(b)}", transform=ax2.transAxes,
         fontweight="bold", va="top")
ax2.set_xlabel("Time $(Jt)$")
ax2.set_ylabel(
    "$\\Delta_\\textrm{steady}$"
)
ax2.grid(alpha=0.2)

# build a discrete colormap of 20 colors from viridis
sites = list(range(0, 100, 5))
base_cmap = plt.cm.viridis
colors = base_cmap(np.linspace(0,1,len(sites)))
disc_cmap = mpl.colors.ListedColormap(colors)

# boundaries halfway between site indices
boundaries = [s - 2.5 for s in sites] + [sites[-1] + 2.5]
norm = mpl.colors.BoundaryNorm(boundaries, disc_cmap.N)

# create the ScalarMappable for the colorbar
sm = mpl.cm.ScalarMappable(cmap=disc_cmap, norm=norm)
sm.set_array([])

# add the colorbar
cbar_ax = fig.add_axes([0.95, 0.125, 0.03, 0.7])
cbar = fig.colorbar(
    sm,
    ax=[ax1, ax2],
    orientation="vertical",
    pad=0.02,
    fraction=0.05,
    boundaries=boundaries,
    ticks=sites,
    cax = cbar_ax
)
cbar.ax.set_title(r"$j$", fontsize=14, pad=10)

# plt.tight_layout()
plt.savefig("Results_AnalyticalComparison.pdf", dpi=600, bbox_inches="tight")
