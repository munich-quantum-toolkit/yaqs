import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.patheffects as pe

# ------------------------
# Data + parameters
# ------------------------
dt_list = [0.01, 0.0125, 0.02, 0.025, 0.04, 0.05, 0.1, 0.125]
gamma_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
gamma_min = 0.01
gamma_max = 50

data_dir = Path(".")
dp_levels = [1e-3, 1e-2, 1e-1, 1.0]

dt = np.array(dt_list, dtype=float)
g = np.array(gamma_list, dtype=float)

# ------------------------
# Loader for U1 / U2
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

            # max bond dimension
            obs_bond = obs_list[0]
            vals = np.asarray(obs_bond[0].results, dtype=float)
            row = gamma_to_idx[gamma_list[j]]
            bond_grid[row, k] = float(np.max(vals))

            # wall time per trajectory
            time_grid[row, k] = float(np.asarray(obs_list[1]))

    return bond_grid, time_grid

u1_bond, u1_time = load_gamma_dt_heatmaps_for_u("u1")
u2_bond, u2_time = load_gamma_dt_heatmaps_for_u("u2")

# ------------------------
# PRX-ish style
# ------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

def panel_label(ax, s):
    t = ax.text(
        0.02, 0.98, s, transform=ax.transAxes,
        ha="left", va="top", fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5)
    )
    t.set_path_effects([pe.withStroke(linewidth=1.3, foreground="white")])

# ------------------------
# pcolormesh edges
# (dt: linear edges; gamma: geometric edges for log axis)
# ------------------------
def edges_linear(x):
    x = np.asarray(x, float)
    dx = np.diff(x)
    left = x[0] - dx[0] / 2
    right = x[-1] + dx[-1] / 2
    mids = (x[:-1] + x[1:]) / 2
    return np.concatenate([[left], mids, [right]])

def edges_geo(x):
    x = np.asarray(x, float)
    r = x[1:] / x[:-1]
    left = x[0] / np.sqrt(r[0])
    right = x[-1] * np.sqrt(r[-1])
    mids = np.sqrt(x[:-1] * x[1:])
    return np.concatenate([[left], mids, [right]])

dt_edges = edges_linear(dt)   # linear dt axis
g_edges = edges_geo(g)        # log-friendly gamma edges

# ------------------------
# Norms (shared per column)
# ------------------------
bond_norm = Normalize(
    vmin=np.nanmin([u1_bond, u2_bond]),
    vmax=np.nanmax([u1_bond, u2_bond]),
)

# Keep log wall time (usually more PRX-like). If you insist on linear: swap to Normalize.
time_norm = LogNorm(
    vmin=max(1e-6, np.nanmin([u1_time, u2_time])),
    vmax=np.nanmax([u1_time, u2_time]),
)

# ------------------------
# dp lines in true (dt, gamma) coordinates
# ------------------------
def add_dp_lines_true(ax, *, add_labels=False):
    dt_dense = np.linspace(dt.min(), dt.max(), 600)  # linear dt (since axis is linear)
    for dp in dp_levels:
        gamma_dense = dp / dt_dense
        mask = (gamma_dense >= gamma_min) & (gamma_dense <= gamma_max)
        if not np.any(mask):
            continue
        ax.plot(dt_dense[mask], gamma_dense[mask], "--", color="k", lw=1.0, alpha=0.65)

    if add_labels:
        dt_label = dt[1]
        for dp in dp_levels:
            gamma_label = dp / dt_label
            if not (gamma_min <= gamma_label <= gamma_max):
                continue
            txt = ax.text(
                dt_label * 1.01, gamma_label * 1.05,
                rf"$dp={dp:g}$", color="w", fontsize=8
            )
            txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black", alpha=0.6)])

# ------------------------
# Heatmap helper
# ------------------------
def heat(ax, Z, *, cmap, norm):
    m = ax.pcolormesh(dt_edges, g_edges, Z, cmap=cmap, norm=norm, shading="auto")
    ax.set_yscale("log")
    ax.set_xlim(dt_edges[0], dt_edges[-1])
    ax.set_ylim(gamma_min, gamma_max)
    ax.tick_params(direction="out")
    return m

# ------------------------
# Figure 1: 2x2 heatmaps
# ------------------------
fig = plt.figure(figsize=(7.2, 6.2), layout="constrained")  # PRX single-column-ish
gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 0.06, 0.06], wspace=0.15, hspace=0.20)

ax_u1_bond = fig.add_subplot(gs[0, 0])
ax_u1_time = fig.add_subplot(gs[0, 1])
ax_u2_bond = fig.add_subplot(gs[1, 0])
ax_u2_time = fig.add_subplot(gs[1, 1])

cax_bond = fig.add_subplot(gs[:, 2])  # shared colorbar for bond column
cax_time = fig.add_subplot(gs[:, 3])  # shared colorbar for time column

m1 = heat(ax_u1_bond, u1_bond, cmap="magma_r", norm=bond_norm)
m2 = heat(ax_u1_time, u1_time, cmap="coolwarm", norm=time_norm)
m3 = heat(ax_u2_bond, u2_bond, cmap="magma_r", norm=bond_norm)
m4 = heat(ax_u2_time, u2_time, cmap="coolwarm", norm=time_norm)

ax_u1_bond.set_title("U1: max bond dimension")
ax_u1_time.set_title("U1: wall time")
ax_u2_bond.set_title("U2: max bond dimension")
ax_u2_time.set_title("U2: wall time")

# ticks: reduce redundancy
for ax in (ax_u1_bond, ax_u1_time):
    ax.set_xticklabels([])

for ax in (ax_u1_time, ax_u2_time):
    ax.set_yticklabels([])

# dt ticks (linear)
for ax in (ax_u2_bond, ax_u2_time):
    ax.set_xticks(dt)
    ax.set_xticklabels([f"{x:g}" for x in dt], rotation=45, ha="right")

# gamma ticks
for ax in (ax_u1_bond, ax_u2_bond):
    ax.set_yticks(g)
    ax.set_yticklabels([f"{x:g}" for x in g])

# dp overlays
for ax in (ax_u1_bond, ax_u1_time, ax_u2_bond, ax_u2_time):
    add_dp_lines_true(ax, add_labels=False)
add_dp_lines_true(ax_u1_bond, add_labels=True)

# panel letters
panel_label(ax_u1_bond, "(a)")
panel_label(ax_u1_time, "(b)")
panel_label(ax_u2_bond, "(c)")
panel_label(ax_u2_time, "(d)")

# shared colorbars
cb_b = fig.colorbar(m1, cax=cax_bond)
cb_b.set_label(r"$\overline{\chi}$")

cb_t = fig.colorbar(m2, cax=cax_time)
cb_t.set_label("Wall time (s)")

# shared axis labels
fig.supxlabel(r"$dt$")
fig.supylabel(r"$\gamma$")

plt.show()
