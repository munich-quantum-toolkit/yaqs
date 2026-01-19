import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.patheffects as pe

# --- PRX Quantum-ish single-column sizing ---
# PRX single-column width is ~3.375 in (≈ 8.6 cm). Pick an aspect that reads well.
FIG_W = 3.375
FIG_H = 2.35  # tweak 2.2–2.5 depending on your layout

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,   # embed fonts nicely
    "ps.fonttype": 42,
})

def outlined_text(ax, x, y, s, **kw):
    t = ax.text(x, y, s, **kw)
    t.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])
    return t

# --- Colors ---
c_cpu = "#9ecae1"   # CPU-limited
c_ram = "#c7e9c0"   # RAM-limited
c_dp  = "#fcbba1"   # dp>1

col_cpu = np.array(mcolors.to_rgb(c_cpu))
col_ram = np.array(mcolors.to_rgb(c_ram))
col_dp  = np.array(mcolors.to_rgb(c_dp))

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

nx, ny = 600, 380
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# --- CPU ↔ RAM blend (horizontal fade) ---
center_x = 0.50
w = 0.18
t = np.clip((X - (center_x - w)) / (2 * w), 0.0, 1.0)[..., None]
img = (1 - t) * col_cpu + t * col_ram

# --- dp>1 wedge: extend to top-left corner ---
tri_x1, tri_y1 = 0.00, 1.00
tri_x2, tri_y2 = 1.00, 1.00
tri_x3, tri_y3 = 1.00, 0.70  # controls thickness

den = (tri_y2 - tri_y3) * (tri_x1 - tri_x3) + (tri_x3 - tri_x2) * (tri_y1 - tri_y3)
w1 = ((tri_y2 - tri_y3) * (X - tri_x3) + (tri_x3 - tri_x2) * (Y - tri_y3)) / den
w2 = ((tri_y3 - tri_y1) * (X - tri_x3) + (tri_x1 - tri_x3) * (Y - tri_y3)) / den
w3 = 1 - w1 - w2
mask_dp = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
img[mask_dp] = col_dp

ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), aspect="auto")

# --- Labels (compact for single-column) ---
box = dict(
    boxstyle="round,pad=0.20",
    facecolor="white",
    edgecolor="black",
    linewidth=0.7,
    alpha=0.78,
)

ax.text(
    0.24, 0.56,
    "Zeno-resolved\n\n"
    "CPU-limited\n"
    "Low $\\chi$\n"
    "Many ops",
    ha="center", va="center", bbox=box
)

ax.text(
    0.78, 0.45,
    "Noise-independent\n\n"
    "RAM-limited\n"
    "High $\\chi$\n"
    "Few ops",
    ha="center", va="center", bbox=box
)

# small top label; outline helps it stay readable on the wedge
outlined_text(ax, 0.86, 0.90, "dp>1",
              ha="center", va="center", fontsize=8.5, bbox=box)

# --- Axes styling ---
ax.set_xlabel(r"Small timestep $\rightarrow$ Large timestep", labelpad=2.5)
ax.set_ylabel(r"Weak noise $\rightarrow$ Strong noise", labelpad=2.5)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_linewidth(0.8)

# Optional: save as vector PDF for PRX
plt.savefig("gamma_dt_summary.pdf", dpi=300)
plt.show()
