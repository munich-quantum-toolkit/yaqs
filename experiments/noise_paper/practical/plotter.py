import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

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
max_time = np.nanmax([u1_time, u2_time])
u1_time = u1_time / np.nanmax(max_time)
u2_time = u2_time / np.nanmax(max_time)

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

time_norm = LogNorm(
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
                rf"$\delta p={dp:g}$", color="w", fontsize=8
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
# ------------------------
# Alpha Calculation
# ------------------------
# Bond Alpha
alpha_bond = u1_bond / u2_bond  
# Handle 0/0 or inf? usually bond > 1.

# Time Alpha
alpha_time = u1_time / u2_time

# ------------------------
# Figure
# ------------------------
# Figure
# ------------------------
# Figure: make it true PRX figure* width (double column)
fig = plt.figure(figsize=(10.5, 2.6)) # Single row height

# 1 row (Bond)
# Layout: [U1] [U2] [CbarShared] [Sp] [Alpha] [CbarAlpha]
gs = fig.add_gridspec(
    1, 6,
    left=0.06, right=0.92, bottom=0.18, top=0.88,
    width_ratios=[1.0, 1.0, 0.03, 0.12, 1.0, 0.03], 
    wspace=0.08
)

# Row 0: Bond
ax_u1_bond = fig.add_subplot(gs[0, 0])
ax_u2_bond = fig.add_subplot(gs[0, 1])
cax_bond   = fig.add_subplot(gs[0, 2])
# Spacer at 3
ax_al_bond = fig.add_subplot(gs[0, 4])
cax_al_bond= fig.add_subplot(gs[0, 5])

gamma_top = 10.0

# --- Plot Bond ---
m1 = heat(ax_u1_bond, u1_bond, cmap="magma_r", norm=bond_norm, gamma_top=gamma_top)
m3 = heat(ax_u2_bond, u2_bond, cmap="magma_r", norm=bond_norm, gamma_top=gamma_top)

# Alpha Bond
from scipy.ndimage import gaussian_filter
alpha_bond_smooth = gaussian_filter(alpha_bond, sigma=0.8) # Light smoothing
vm_ab = 1 # np.nanpercentile(alpha_bond, 2)
vx_ab = 2 # np.nanpercentile(alpha_bond, 98)
if vx_ab <= vm_ab: vx_ab = vm_ab + 1.0
m5 = heat(ax_al_bond, alpha_bond, cmap="cividis", norm=Normalize(vmin=vm_ab, vmax=vx_ab), gamma_top=gamma_top)

# --- Formatting ---

# X-Labels (now on all plots)
tick_idx = np.arange(len(dt_list))
tick_labels = [f"{x:g}" for x in dt_list]
for ax in (ax_u1_bond, ax_u2_bond, ax_al_bond):
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_xlabel(r"$\delta t$", labelpad=4)

# Y-Labels (only left col)
for ax in (ax_u2_bond, ax_al_bond):
    ax.tick_params(labelleft=False)

top_idx = int(np.where(g == gamma_top)[0][0])
g_tick_idx = np.arange(top_idx + 1)
g_tick_labels = [f"{x:g}" for x in g[:top_idx + 1]]
for ax in (ax_u1_bond,):
    ax.set_yticks(g_tick_idx)
    ax.set_yticklabels(g_tick_labels)
    ax.set_ylabel(r"$\gamma$", labelpad=4)

# Contours for Alphas
# Need grid for contour
DT_c, G_c = np.meshgrid(dt_centers, g_centers[:top_idx+1])
# We need to slice alpha to gamma_top
alpha_bond_cut = alpha_bond_smooth[:top_idx+1, :]

# Contour Bond
lev_ab = [1.0, 1.2, 1.4, 1.6, 1.8] # np.linspace(vm_ab, vx_ab, 5)
cs_ab = ax_al_bond.contour(DT_c, G_c, alpha_bond_cut, levels=lev_ab, colors="white", linewidths=0.8, alpha=0.8)
ax_al_bond.clabel(cs_ab, fmt="%.2f", inline=True, fontsize=7, colors="white")


# dp lines (ONLY on U1/U2 plots, REMOVE from Alpha)
for ax in (ax_u1_bond, ax_u2_bond):
    add_dp_lines_true(ax, add_labels=False, gamma_top=gamma_top)

# Only add labels to first one?
add_dp_lines_true(ax_u1_bond, add_labels=True, gamma_top=gamma_top)

# Titles / Panels
panel_label(ax_u1_bond, "(a)")
panel_label(ax_u2_bond, "(b)")
panel_label(ax_al_bond, "(c)")

# Titles
ax_u1_bond.set_title("Unraveling A", fontsize=10)
ax_u2_bond.set_title("Unraveling B", fontsize=10)
# No title for Alpha column per request
# ax_al_bond.set_title("", fontsize=10)

# Colorbars
# 1. Bond Shared
cb_b = fig.colorbar(m1, cax=cax_bond)
cax_bond.set_title(r"$\bar{\chi}$", pad=5, fontsize=10)

# 2. Alpha Bond
cb_ab = fig.colorbar(m5, cax=cax_al_bond)
cax_al_bond.set_title(r"$\alpha$", pad=5, fontsize=10) # Renamed

fig.savefig("gamma_dt_alpha.pdf", dpi=300)
plt.show()
