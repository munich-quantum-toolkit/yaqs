import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# ------------------------
# Data + parameters
# ------------------------
dt_list = [0.001, 0.00125, 0.0025, 0.004, 0.005, 0.01, 0.0125, 0.02, 0.025, 0.04, 0.05, 0.1]
gamma_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
gamma_min = 0.01
gamma_max = 50

T = 5.0
data_dir = Path(".")
prefix = "u1_practical_"

def load_gamma_dt_heatmap(
    dt_list=dt_list,
    gamma_list=gamma_list,
    prefix=prefix,
    data_dir=data_dir,
    which_obs="max_bond",
):
    grid = np.full((len(gamma_list), len(dt_list)), np.nan, dtype=float)

    gamma_to_idx = {g: i for i, g in enumerate(gamma_list)}
    obs_idx_map = {"Z": 0, "magnetization": 0, "entropy": 1, "max_bond": 2}
    obs_idx = obs_idx_map[which_obs]

    for k, dt in enumerate(dt_list):
        fname = data_dir / f"{prefix}{k}.pickle"
        if fname.exists():
            with open(fname, "rb") as f:
                results = pickle.load(f)

            if len(results) != len(gamma_list):
                raise ValueError(
                    f"{fname} has {len(results)} entries, "
                    f"expected {len(gamma_list)}"
                )

            for j, obs_list in enumerate(results):
                if obs_list is None:
                    continue
                obs = obs_list[obs_idx]
                vals = np.array(obs.results)
                g = gamma_list[j]
                row = gamma_to_idx[g]
                grid[row, k] = float(vals.mean())

    return grid

grid = load_gamma_dt_heatmap(which_obs="max_bond")                 # χ_avg
scaled_grid = np.full_like(grid, np.nan)
for k, dt in enumerate(dt_list):
    scaled_grid[:, k] = grid[:, k] * T / dt                         # χ_avg T/dt

dt_arr = np.array(dt_list)
gamma_arr = np.array(gamma_list)
dp_levels = [1e-3, 1e-2, 1e-1, 1.0]

# ------------------------
# Helper: add dp lines
# ------------------------
def add_dp_lines(ax):
    dt_dense = np.linspace(dt_arr.min(), dt_arr.max(), 500)
    for dp in dp_levels:
        gamma_dense = dp / dt_dense
        mask = (gamma_dense >= gamma_arr.min()) & (gamma_dense <= gamma_arr.max())
        if not np.any(mask):
            continue
        dt_line = dt_dense[mask]
        gamma_line = gamma_dense[mask]
        x = np.interp(dt_line, dt_arr, np.arange(len(dt_arr)))
        y = np.interp(gamma_line, gamma_arr, np.arange(len(gamma_arr)))
        ax.plot(x, y, linestyle="--", linewidth=1.0, color="black", alpha=0.8)

    # label dp on second dt column to avoid edges
    for dp in dp_levels:
        dt_label = dt_arr[1]
        gamma_label = dp / dt_label
        if gamma_label < gamma_arr.min() or gamma_label > gamma_arr.max():
            continue
        x_lab = np.interp(dt_label, dt_arr, np.arange(len(dt_arr)))
        y_lab = np.interp(gamma_label, gamma_arr, np.arange(len(gamma_arr)))
        ax.text(
            x_lab + 0.2,
            y_lab + 0.2,
            rf"$dp={dp:g}$",
            color="white",
            fontsize=7,
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
        )

# ------------------------
# Figure with 3 panels
# ------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4.5),
    gridspec_kw={"width_ratios": [1.0, 1.0, 1.1]}
)

# ------------------------
# (a) Max bond dimension heatmap
# ------------------------
ax = axes[0]
vmin_a = np.nanmin(grid)
vmax_a = np.nanmax(grid)

im_a = ax.imshow(
    grid,
    origin="lower",
    aspect="auto",
    cmap="magma_r",
    vmin=vmin_a,
    vmax=vmax_a,
)

ax.set_xticks(np.arange(len(dt_list)))
ax.set_xticklabels([f"{dt:.3g}" for dt in dt_list], rotation=45, ha="right")
ax.set_yticks(np.arange(len(gamma_list)))
ax.set_yticklabels([f"{g:.3g}" for g in gamma_list])

ax.set_xlabel(r"$dt$")
ax.set_ylabel(r"$\gamma$")

cbar_a = fig.colorbar(im_a, ax=ax)
cbar_a.set_label(r"$\overline{\chi}$")

add_dp_lines(ax)
ax.text(0.02, 0.98, "(a)", transform=ax.transAxes,
        ha="left", va="top", fontsize=12, fontweight="bold")

ymin = np.interp(gamma_min, gamma_arr, np.arange(len(gamma_arr)))
ymax = np.interp(gamma_max, gamma_arr, np.arange(len(gamma_arr)))

ax.set_ylim(ymin, ymax)

# ------------------------
# (b) Cost heatmap χ_avg T/dt
# ------------------------
ax = axes[1]
vmin_b = np.nanmin(scaled_grid)
vmax_b = np.nanmax(scaled_grid)

im_b = ax.imshow(
    scaled_grid,
    origin="lower",
    aspect="auto",
    cmap="coolwarm",
    vmin=vmin_b,
    vmax=vmax_b,
)

ax.set_xticks(np.arange(len(dt_list)))
ax.set_xticklabels([f"{dt:.3g}" for dt in dt_list], rotation=45, ha="right")
ax.set_yticks(np.arange(len(gamma_list)))
ax.set_yticklabels([f"{g:.3g}" for g in gamma_list])

ax.set_xlabel(r"$dt$")
ax.set_ylabel(r"$\gamma$")

cbar_b = fig.colorbar(im_b, ax=ax)
cbar_b.set_label(r"$\overline{\chi}\,T/dt$")

add_dp_lines(ax)
ax.text(0.02, 0.98, "(b)", transform=ax.transAxes,
        ha="left", va="top", fontsize=12, fontweight="bold")

ymin = np.interp(gamma_min, gamma_arr, np.arange(len(gamma_arr)))
ymax = np.interp(gamma_max, gamma_arr, np.arange(len(gamma_arr)))

ax.set_ylim(ymin, ymax)

# ------------------------
# (c) Abstract phase diagram
# ------------------------
ax = axes[2]

# --- Updated phase colors ---
c_coherent   = "#fff2a0"    # yellow
c_zeno       = "#9ecae1"    # blue
c_noiseind   = "#c7e9c0"    # green (noise-independent)
c_inaccurate = "#fdae6b"    # orange (inaccurate large-dt regime)
c_dp         = "#fcbba1"    # red (dp>1 wedge)

col_coherent = np.array(mcolors.to_rgb(c_coherent))
col_zeno     = np.array(mcolors.to_rgb(c_zeno))
col_noiseind = np.array(mcolors.to_rgb(c_noiseind))
col_inacc    = np.array(mcolors.to_rgb(c_inaccurate))
col_dp       = np.array(mcolors.to_rgb(c_dp))

# Conceptual grid in normalized coordinates
nx_c, ny_c = 400, 300
x_c = np.linspace(0, 1, nx_c)
y_c = np.linspace(0, 1, ny_c)
X_c, Y_c = np.meshgrid(x_c, y_c)

# Boundaries
xc_boundary = 0.25  # dt crossover (left vs right)
yc_boundary = 0.35  # coherent ↔ Zeno
wx = 0.10
wy = 0.10

# --- NEW: "inaccurate" band on far right (large dt) ---
x_inacc = 0.80   # where the band starts (normalized dt)
w_inacc = 0.05   # half-width for smooth blend into inaccurate band

# Base image initialization
img = np.zeros((ny_c, nx_c, 3))

mask_left     = X_c < xc_boundary
mask_right    = ~mask_left
mask_coherent = mask_left & (Y_c < yc_boundary)
mask_zeno     = mask_left & (Y_c >= yc_boundary)

# Assign base colors
img[mask_coherent] = col_coherent
img[mask_zeno]     = col_zeno
img[mask_right]    = col_noiseind

# --- Smooth vertical blending (left → noise-independent) ---
t_x = np.clip((X_c - (xc_boundary - wx)) / (2 * wx), 0.0, 1.0)[..., None]
left_colors = np.zeros_like(img)
left_colors[mask_coherent] = col_coherent
left_colors[mask_zeno]     = col_zeno
left_colors[mask_right]    = col_noiseind
img = (1 - t_x) * left_colors + t_x * col_noiseind

# --- Smooth horizontal blending (coherent ↔ Zeno) ---
t_y = np.clip((Y_c - (yc_boundary - wy)) / (2 * wy), 0.0, 1.0)[..., None]
blend_left_mask = X_c < (xc_boundary + wx)
blend_left_mask_3d = blend_left_mask[..., None]
blend_colors = (1 - t_y) * col_coherent + t_y * col_zeno
img = np.where(blend_left_mask_3d, blend_colors, img)

# --------------------------------------------------------
# 1) Inaccurate large-dt band (apply first)
# --------------------------------------------------------
t_inacc = np.clip((X_c - (x_inacc - w_inacc)) / (2 * w_inacc), 0.0, 1.0)[..., None]
mask_inacc_domain = (X_c >= (x_inacc - w_inacc))[..., None]
img = np.where(mask_inacc_domain, (1 - t_inacc) * img + t_inacc * col_inacc, img)

# --------------------------------------------------------
# 2) dp>1 wedge (apply second so it remains distinct on top)
# --------------------------------------------------------
# Triangle vertices: (xc_boundary,1), (1,1), (1,0.7)
tri_x1, tri_y1 = xc_boundary, 1.0
tri_x2, tri_y2 = 1.0,       1.0
tri_x3, tri_y3 = 1.0,       0.7

den = (tri_y2 - tri_y3)*(tri_x1 - tri_x3) + (tri_x3 - tri_x2)*(tri_y1 - tri_y3)
w1 = ((tri_y2 - tri_y3)*(X_c - tri_x3) + (tri_x3 - tri_x2)*(Y_c - tri_y3)) / den
w2 = ((tri_y3 - tri_y1)*(X_c - tri_x3) + (tri_x1 - tri_x3)*(Y_c - tri_y3)) / den
w3 = 1 - w1 - w2
mask_dp = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)

img[mask_dp] = col_dp  # overwrite (keeps dp wedge separate even inside the band)

# --- Draw figure ---
ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), aspect="auto")

props = dict(ha="center", va="center", fontsize=11, color="black")

ax.text(0.25, 0.20,
        "Coherent regime\nHigh $\\overline{\\chi}$, many ops\n(Memory & CPU limited)",
        **props)

ax.text(0.25, 0.75,
        "Zeno regime\nLow $\\overline{\\chi}$, many ops\n(CPU limited)",
        **props)

ax.text(0.62, 0.45,
        "Noise-independent regime\n$\\overline{\\chi}$ saturated, few ops\n(Memory limited)",
        **props)

ax.text(0.90, 0.30,
        "Low accuracy regime",
        **props)

ax.text(0.86, 0.90,
        "$dp>1$",
        **props)

ax.set_xlabel("Small timestep   $\\longrightarrow$   Large timestep",
              fontsize=12, labelpad=10)
ax.set_ylabel("Weak noise   $\\longrightarrow$   Strong noise",
              fontsize=12, labelpad=10)

ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

ax.text(0.02, 0.98, "(c)", transform=ax.transAxes,
        ha="left", va="top", fontsize=12, fontweight="bold")

fig.tight_layout()
plt.show()
