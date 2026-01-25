import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.patheffects as pe

# --- PRX Quantum-ish single-column sizing ---
FIG_W = 3.375
FIG_H = 2.20

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def outlined_text(ax, x, y, s, **kw):
    t = ax.text(x, y, s, **kw)
    t.set_path_effects([pe.withStroke(linewidth=1.3, foreground="white")])
    return t

# --- Colors ---
c_cpu = "#9ecae1"   # CPU-limited
c_ram = "#c7e9c0"   # RAM-limited
c_dp  = "#fcbba1"   # dp>1
c_bal = "#efbcff"   # balanced / minimal walltime

col_cpu = np.array(mcolors.to_rgb(c_cpu))
col_ram = np.array(mcolors.to_rgb(c_ram))
col_dp  = np.array(mcolors.to_rgb(c_dp))
col_bal = np.array(mcolors.to_rgb(c_bal))

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

nx, ny = 650, 420
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# --- CPU ↔ RAM blend (horizontal fade) ---
center_x = 0.52
w = 0.20
t = np.clip((X - (center_x - w)) / (2 * w), 0.0, 1.0)[..., None]
img = (1 - t) * col_cpu + t * col_ram

# --- dp>1 wedge: ORIGINAL style ---
tri_x1, tri_y1 = 0.00, 1.00
tri_x2, tri_y2 = 1.00, 1.00
tri_x3, tri_y3 = 1.00, 0.70  # thickness control

den = (tri_y2 - tri_y3) * (tri_x1 - tri_x3) + (tri_x3 - tri_x2) * (tri_y1 - tri_y3)
w1 = ((tri_y2 - tri_y3) * (X - tri_x3) + (tri_x3 - tri_x2) * (Y - tri_y3)) / den
w2 = ((tri_y3 - tri_y1) * (X - tri_x3) + (tri_x1 - tri_x3) * (Y - tri_y3)) / den
w3 = 1 - w1 - w2
mask_dp = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
img[mask_dp] = col_dp

# --- Balanced regime: dashed minimal-wall-time curve across noise ---

# Parametrize a curve from near dp>1 down to weak noise
ys = np.linspace(0.02, 0.84, 300)   # bottom → just below dp>1
ys1 = np.linspace(0.02, 0.88, 300)   # bottom → just below dp>1
ys2 = np.linspace(0.02, 0.8, 300)   # bottom → just below dp>1

# Slight rightward drift with decreasing noise (matches your heatmap intuition)
# Tune these two numbers only:
x0 = 0.54        # center near strong noise
slope = 0.0     # how much it shifts right as noise decreases

xs = x0 + slope * (0.86 - ys)

ax.plot(
    xs, ys,
    linestyle=(0, (3, 3)),   # dashed, PRX-safe
    linewidth=1.4,
    color="k",
    alpha=0.9,
    zorder=6,
)

# Optional: make it a *band* instead of a single curve
band = 0.1
ax.plot(xs - band, ys1, linestyle=(0, (2, 3)), lw=1.0, color="k", alpha=0.5)
ax.plot(xs + band, ys2, linestyle=(0, (2, 3)), lw=1.0, color="k", alpha=0.5)

i = len(xs) // 2   # pick a point slightly above center
dx = 0.015               # small horizontal offset from the dashed line

ax.text(
    xs[i]-dx,
    ys[i],
    r"Minimal wall time",
    rotation=90,         # vertical text
    ha="right",
    va="center",
    fontsize=8.5,
    color="black",
    alpha=0.9,
    zorder=7,
)

ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), aspect="auto")

# --- Label boxes (original text restored) ---
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

# ax.text(
#     0.78, 0.45,
#     "Noise-independent\n\n"
#     "RAM-limited\n"
#     "High $\\chi$\n"
#     "Few ops",
#     ha="center", va="center", bbox=box
# )

# # --- Balanced label: minimal text only ---
# ax.text(
#     bal_x, bal_y,
#     r"Minimal wall time",
#     ha="center", va="center", bbox=box
# )
# --- (A) Lower the noise-independent box (was y=0.45) ---
ax.text(
    0.78, 0.25,   # <-- lowered from 0.45
    "Noise-independent\n\n"
    "RAM-limited\n"
    "High $\\chi$\n"
    "Few ops",
    ha="center", va="center", bbox=box
)


# --- dp label: back to dp>1 ---
outlined_text(ax, 0.86, 0.90, "dp>1",
              ha="center", va="center", fontsize=8.5, bbox=box)

# --- Axes styling ---
ax.set_xlabel(r"Small timestep $\rightarrow$ Large timestep", labelpad=2.5)
ax.set_ylabel(r"Weak noise $\rightarrow$ Strong noise", labelpad=2.5)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_linewidth(0.8)

plt.savefig("gamma_dt_summary.pdf", dpi=300)
plt.show()
