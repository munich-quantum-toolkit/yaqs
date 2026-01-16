import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
})

def panel_label(ax, s):
    t = ax.text(
        0.02, 0.98, s, transform=ax.transAxes,
        ha="left", va="top", fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5)
    )
    t.set_path_effects([pe.withStroke(linewidth=1.3, foreground="white")])

# Colors
c_coherent   = "#fff2a0"
c_zeno       = "#9ecae1"
c_noiseind   = "#c7e9c0"
c_inaccurate = "#fdae6b"
c_dp         = "#fcbba1"

col_coherent = np.array(mcolors.to_rgb(c_coherent))
col_zeno     = np.array(mcolors.to_rgb(c_zeno))
col_noiseind = np.array(mcolors.to_rgb(c_noiseind))
col_inacc    = np.array(mcolors.to_rgb(c_inaccurate))
col_dp       = np.array(mcolors.to_rgb(c_dp))

fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")  # PRX standard standalone

nx_c, ny_c = 500, 350
x_c = np.linspace(0, 1, nx_c)
y_c = np.linspace(0, 1, ny_c)
X_c, Y_c = np.meshgrid(x_c, y_c)

# Boundaries
xc_boundary, yc_boundary = 0.25, 0.35
wx, wy = 0.10, 0.10
x_inacc, w_inacc = 0.80, 0.05

img = np.zeros((ny_c, nx_c, 3))
mask_left     = X_c < xc_boundary
mask_right    = ~mask_left
mask_coherent = mask_left & (Y_c < yc_boundary)
mask_zeno     = mask_left & (Y_c >= yc_boundary)

img[mask_coherent] = col_coherent
img[mask_zeno]     = col_zeno
img[mask_right]    = col_noiseind

# smooth blend left→noiseind
t_x = np.clip((X_c - (xc_boundary - wx)) / (2 * wx), 0.0, 1.0)[..., None]
img = (1 - t_x) * img + t_x * col_noiseind

# smooth coherent↔zeno blend
t_y = np.clip((Y_c - (yc_boundary - wy)) / (2 * wy), 0.0, 1.0)[..., None]
blend_left_mask = (X_c < (xc_boundary + wx))[..., None]
blend_colors = (1 - t_y) * col_coherent + t_y * col_zeno
img = np.where(blend_left_mask, blend_colors, img)

# inaccurate band
t_inacc = np.clip((X_c - (x_inacc - w_inacc)) / (2 * w_inacc), 0.0, 1.0)[..., None]
mask_inacc_domain = (X_c >= (x_inacc - w_inacc))[..., None]
img = np.where(mask_inacc_domain, (1 - t_inacc) * img + t_inacc * col_inacc, img)

# dp>1 wedge
tri_x1, tri_y1 = xc_boundary, 1.0
tri_x2, tri_y2 = 1.0,       1.0
tri_x3, tri_y3 = 1.0,       0.7
den = (tri_y2 - tri_y3)*(tri_x1 - tri_x3) + (tri_x3 - tri_x2)*(tri_y1 - tri_y3)
w1 = ((tri_y2 - tri_y3)*(X_c - tri_x3) + (tri_x3 - tri_x2)*(Y_c - tri_y3)) / den
w2 = ((tri_y3 - tri_y1)*(X_c - tri_x3) + (tri_x1 - tri_x3)*(Y_c - tri_y3)) / den
w3 = 1 - w1 - w2
img[(w1 >= 0) & (w2 >= 0) & (w3 >= 0)] = col_dp

ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), aspect="auto")

panel_label(ax, "(a)")

box = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.6)
ax.text(0.20, 0.18, "Coherent\n(high $\\bar\\chi$)", ha="center", va="center", bbox=box)
ax.text(0.20, 0.72, "Zeno\n(low $\\bar\\chi$)", ha="center", va="center", bbox=box)
ax.text(0.62, 0.45, "Noise-independent\n($\\bar\\chi$ saturated)", ha="center", va="center", bbox=box)
ax.text(0.90, 0.30, "Low accuracy", ha="center", va="center", bbox=box)
ax.text(0.85, 0.88, "$dp>1$", ha="center", va="center", bbox=box)

ax.set_xlabel("Small timestep  $\\rightarrow$  Large timestep")
ax.set_ylabel("Weak noise  $\\rightarrow$  Strong noise")
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_linewidth(0.8)

plt.show()
