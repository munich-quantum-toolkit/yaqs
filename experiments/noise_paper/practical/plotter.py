import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.patheffects as pe

# ------------------------
# Data + parameters
# ------------------------
dt_list = [0.01, 0.0125, 0.02, 0.025, 0.04, 0.05, 0.1]
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
print(np.min(u1_bond / u2_bond), np.max(u1_bond / u2_bond), np.mean(u1_bond / u2_bond))
max_time = np.nanmax([u1_time, u2_time])
u1_time = u1_time / np.nanmax(u1_time)
u2_time = u2_time / np.nanmax(u2_time)

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
# pcolormesh edges (index axes for dt AND gamma -> perfect grid alignment)
# ------------------------
dt_centers = np.arange(len(dt_list), dtype=float)
dt_edges   = np.arange(len(dt_list) + 1, dtype=float) - 0.5

g_centers  = np.arange(len(gamma_list), dtype=float)
g_edges    = np.arange(len(gamma_list) + 1, dtype=float) - 0.5

# ------------------------
# Norms (shared per column)
# ------------------------
bond_norm = Normalize(
    vmin=np.nanmin([u1_bond, u2_bond]),
    vmax=np.nanmax([u1_bond, u2_bond]),
)

time_norm = Normalize(
    vmin=np.nanmin([u1_time, u2_time]),
    vmax=np.nanmax([u1_time, u2_time]),
)

# ------------------------
# dp lines in true (dt, gamma) coordinates
# ------------------------
def add_dp_lines_true(ax, *, add_labels=False, gamma_top=10.0):
    dt_dense = np.linspace(dt.min(), dt.max(), 800)
    x_dense = np.interp(dt_dense, dt, dt_centers)

    logg = np.log(g)
    for dp in dp_levels:
        gamma_dense = dp / dt_dense
        mask = (gamma_dense >= gamma_min) & (gamma_dense <= gamma_top)
        if not np.any(mask):
            continue

        y_dense = np.interp(np.log(gamma_dense[mask]), logg, g_centers)
        ax.plot(x_dense[mask], y_dense, "--", color="k", lw=1.0, alpha=0.65)

    if add_labels:
        dt_label = dt[1]
        x_lab = np.interp(dt_label, dt, dt_centers)
        for dp in dp_levels:
            gamma_label = dp / dt_label
            if not (gamma_min <= gamma_label <= gamma_top):
                continue
            y_lab = np.interp(np.log(gamma_label), logg, g_centers)
            txt = ax.text(
                x_lab + 0.15, y_lab + 0.10,
                rf"$dp={dp:g}$", color="w", fontsize=8
            )
            txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black", alpha=0.6)])

# ------------------------
# Heatmap helper
# ------------------------
def heat(ax, Z, *, cmap, norm, gamma_top=10.0):
    m = ax.pcolormesh(dt_edges, g_edges, Z, cmap=cmap, norm=norm, shading="auto")
    ax.set_xlim(dt_edges[0], dt_edges[-1])

    top_idx = int(np.where(g == gamma_top)[0][0])
    ax.set_ylim(-0.5, top_idx + 0.5)

    ax.tick_params(direction="out")
    return m

# ------------------------
# Figure
# ------------------------
fig = plt.figure(figsize=(8.2, 5.4))

gs = fig.add_gridspec(
    2, 5,
    left=0.085, right=0.955, bottom=0.135, top=0.965,
    width_ratios=[1.0, 0.035, 0.07, 1.0, 0.035],
    wspace=0.06, hspace=0.20
)

ax_u1_bond = fig.add_subplot(gs[0, 0])
ax_u1_time = fig.add_subplot(gs[0, 3])
ax_u2_bond = fig.add_subplot(gs[1, 0])
ax_u2_time = fig.add_subplot(gs[1, 3])

cax_bond = fig.add_subplot(gs[:, 1])
cax_time = fig.add_subplot(gs[:, 4])

ax_spacer = fig.add_subplot(gs[:, 2])
ax_spacer.axis("off")

pos = cax_bond.get_position()
cax_bond.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.95])

pos = cax_time.get_position()
cax_time.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.95])

gamma_top = 10.0

m1 = heat(ax_u1_bond, u1_bond, cmap="magma_r", norm=bond_norm, gamma_top=gamma_top)
m2 = heat(ax_u1_time, u1_time, cmap="coolwarm", norm=time_norm, gamma_top=gamma_top)
m3 = heat(ax_u2_bond, u2_bond, cmap="magma_r", norm=bond_norm, gamma_top=gamma_top)
m4 = heat(ax_u2_time, u2_time, cmap="coolwarm", norm=time_norm, gamma_top=gamma_top)

for ax in (ax_u1_bond, ax_u1_time):
    ax.tick_params(labelbottom=False)

for ax in (ax_u1_time, ax_u2_time):
    ax.tick_params(labelleft=False)

tick_idx = np.arange(len(dt_list))
tick_labels = [f"{x:g}" for x in dt_list]
for ax in (ax_u2_bond, ax_u2_time):
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlabel(r"$dt$", labelpad=4)

top_idx = int(np.where(g == gamma_top)[0][0])
g_tick_idx = np.arange(top_idx + 1)
g_tick_labels = [f"{x:g}" for x in g[:top_idx + 1]]
for ax in (ax_u1_bond, ax_u2_bond):
    ax.set_yticks(g_tick_idx)
    ax.set_yticklabels(g_tick_labels)
    ax.set_ylabel(r"$\gamma$", labelpad=4)

for ax in (ax_u1_bond, ax_u1_time, ax_u2_bond, ax_u2_time):
    add_dp_lines_true(ax, add_labels=False, gamma_top=gamma_top)
add_dp_lines_true(ax_u1_bond, add_labels=True, gamma_top=gamma_top)

panel_label(ax_u1_bond, "(a)")
panel_label(ax_u1_time, "(b)")
panel_label(ax_u2_bond, "(c)")
panel_label(ax_u2_time, "(d)")

cb_b = fig.colorbar(m1, cax=cax_bond)
cb_b.set_label("")
cax_bond.text(0.5, 1.02, r"$\bar{\chi}$", transform=cax_bond.transAxes,
              ha="center", va="bottom")

cb_t = fig.colorbar(m2, cax=cax_time)
cb_t.set_label("")
cax_time.text(0.5, 1.02, "$\\tau$", transform=cax_time.transAxes,
              ha="center", va="bottom")

plt.show()
