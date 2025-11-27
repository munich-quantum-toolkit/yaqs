import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# Phase colors
c_coherent = "#fdae91"      # soft orange/red
c_zeno = "#9ecae1"          # soft blue
c_reslim = "#fee6a8"        # warm light yellow

# Convert hex -> RGB triplets
col_coherent = np.array(mcolors.to_rgb(c_coherent))
col_zeno     = np.array(mcolors.to_rgb(c_zeno))
col_reslim   = np.array(mcolors.to_rgb(c_reslim))

# Grid in "timestep" (x) and "noise" (y), normalized to [0,1]
nx, ny = 400, 300
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Parameters for where/over what width to blend
x_c = 0.4       # center of dt crossover
w_x = 0.10       # half-width of vertical blending region
y_c = 0.50       # boundary between coherent and Zeno
w_y = 0.1       # half-width of horizontal blending region

# Start with base colors (piecewise constant)
img = np.zeros((ny, nx, 3))

# Left side: coherent (bottom) and Zeno (top)
mask_left = X < x_c
mask_coherent = mask_left & (Y < y_c)
mask_zeno = mask_left & (Y >= y_c)

img[mask_coherent] = col_coherent
img[mask_zeno] = col_zeno

# Right side: resolution-limited
mask_right = X >= x_c
img[mask_right] = col_reslim

# --- Vertical blending between left regimes and resolution-limited ---
# Blend factor t_x goes from 0 (pure left regime) to 1 (pure resolution-limited)
t_x = np.clip((X - (x_c - w_x)) / (2 * w_x), 0.0, 1.0)
t_x = t_x[..., None]  # broadcast

# Color on the "left" side before blending (coherent or Zeno)
left_colors = np.zeros_like(img)
left_colors[mask_coherent] = col_coherent
left_colors[mask_zeno] = col_zeno
left_colors[mask_right] = col_reslim   # so outside blend region nothing changes

img = (1 - t_x) * left_colors + t_x * col_reslim

# --- Horizontal blending between coherent and Zeno on the left ---
# Only apply on left part (where dt is small enough)
t_y = np.clip((Y - (y_c - w_y)) / (2 * w_y), 0.0, 1.0)
t_y = t_y[..., None]

# Target colors for bottom/top
bottom = col_coherent
top = col_zeno

# Only blend on the left side (X < x_c + w_x so the transition still visible)
blend_left_mask = X < (x_c + w_x)
blend_left_mask_3d = blend_left_mask[..., None]

blend_colors = (1 - t_y) * bottom + t_y * top
img = np.where(blend_left_mask_3d, blend_colors, img)

# --------------------------------------------------------
# Plot
# --------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), aspect="auto")

# Text labels
props = dict(ha='center', va='center', fontsize=12, color='black')

# ax.text(0.18, 0.20,
#         "Coherent phase\nHigh $\overline{\\chi}$, many ops",
#         **props)

# ax.text(0.18, 0.75,
#         "Zeno phase\nLow $\overline{\\chi}$, many ops",
#         **props)

# ax.text(0.72, 0.50,
#         "Resolution-limited phase\nHigh $\overline{\\chi}$, few ops",
#         **props)

ax.text(0.25, 0.20,
        "Coherent phase\nHigh $\overline{\\chi}$, many ops\n(Memory and CPU limited)",
        **props)

ax.text(0.25, 0.75,
        "Zeno phase\nLow $\overline{\\chi}$, many ops\n(CPU limited)",
        **props)

ax.text(0.75, 0.50,
        "Noise-independent phase\nHigh $\overline{\\chi}$, few ops\n(Memory limited)",
        **props)
        
# Axis labels
ax.set_xlabel("Small timestep     $\\longrightarrow$     Large timestep",
              fontsize=13, labelpad=12)
ax.set_ylabel("Weak noise     $\\longrightarrow$     Strong noise",
              fontsize=13, labelpad=14)

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

# Thin frame
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.show()
