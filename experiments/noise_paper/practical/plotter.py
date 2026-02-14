import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import matplotlib.patheffects as pe
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter, FixedLocator, FixedFormatter
from scipy.ndimage import gaussian_filter

# ------------------------
# Data + parameters
# ------------------------
dt_list = [0.01, 0.0125, 0.02, 0.025, 0.04, 0.05, 0.1]
gamma_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
gamma_min = 0.01
gamma_max = 10.0 # Standard plotting range

data_dir = Path(".")
dp_levels = [1e-3, 1e-2, 1e-1, 1.0]

dt = np.array(dt_list, dtype=float)
g = np.array(gamma_list, dtype=float)

def get_log_edges(centers):
    centers = np.asarray(centers)
    log_c = np.log10(centers)
    d_log = np.diff(log_c)
    edges_log = np.concatenate([
        [log_c[0] - d_log[0]/2],
        log_c[:-1] + d_log/2,
        [log_c[-1] + d_log[-1]/2]
    ])
    return 10**edges_log

dt_edges = get_log_edges(dt_list)
g_edges = get_log_edges(gamma_list)
DTc, Gc = np.meshgrid(dt, g)

# ------------------------
# Loader for U1 / U2
# ------------------------
def load_gamma_dt_heatmaps_for_u(u_tag: str):
    bond_grid = np.full((len(gamma_list), len(dt_list)), np.nan, dtype=float)
    time_grid = np.full_like(bond_grid, np.nan)
    gamma_to_idx = {gg: i for i, gg in enumerate(gamma_list)}

    for k, _dt in enumerate(dt_list):
        fname = data_dir / f"practical_{u_tag}_{k}.pickle"
        if not fname.exists(): continue
        with open(fname, "rb") as f:
            results = pickle.load(f)
        for j, obs_list in enumerate(results):
            if obs_list is None: continue
            obs_bond = obs_list[0]
            vals = np.asarray(obs_bond[0].results, dtype=float)
            row = gamma_to_idx[gamma_list[j]]
            bond_grid[row, k] = float(np.max(vals))
    return bond_grid

u1_bond = load_gamma_dt_heatmaps_for_u("u1")
u2_bond = load_gamma_dt_heatmaps_for_u("u2")

# ------------------------
# PRX Style Params
# ------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
})

# ------------------------
# Normalization
# ------------------------
all_bond = np.concatenate([u1_bond.flatten(), u2_bond.flatten()])
vmin_chi = 8.0
vmax_chi = np.nanpercentile(all_bond, 99)
norm_chi = LogNorm(vmin=vmin_chi, vmax=vmax_chi)

# Alpha: A/B ratio
with np.errstate(divide='ignore', invalid='ignore'):
    Z_alpha = np.divide(u1_bond, u2_bond, out=np.full_like(u1_bond, np.nan), where=(u1_bond > 0) & (u2_bond > 0))

norm_al = Normalize(vmin=1.0, vmax=2)
cmap_al = plt.get_cmap("viridis").copy()
cmap_al.set_bad(color="0.85")

# ------------------------
# Layout
# ------------------------
fig = plt.figure(figsize=(12, 3.8))
gs = fig.add_gridspec(
    1, 6,
    left=0.08, right=0.92, bottom=0.2, top=0.85,
    width_ratios=[1.0, 1.0, 0.04, 0.15, 1.0, 0.04],
    wspace=0.15
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
cax_chi = fig.add_subplot(gs[0, 2])
axAlpha = fig.add_subplot(gs[0, 4])
cax_alpha = fig.add_subplot(gs[0, 5])

pc_opts = dict(shading="flat", edgecolors="none", antialiased=False)

# ------------------------
# Plot Panels
# ------------------------
pcmA = axA.pcolormesh(dt_edges, g_edges, u1_bond, cmap="magma_r", norm=norm_chi, **pc_opts)
pcmB = axB.pcolormesh(dt_edges, g_edges, u2_bond, cmap="magma_r", norm=norm_chi, **pc_opts)
pcm_al = axAlpha.pcolormesh(dt_edges, g_edges, Z_alpha, cmap=cmap_al, norm=norm_al, **pc_opts)

# Subtle Grid
for ax in (axA, axB, axAlpha):
    ax.grid(which="both", color="w", alpha=0.12, linewidth=0.5)

# ------------------------
# Contours
# ------------------------
chi_levels = [16, 32, 48, 64, 80]
chi_rotations = 10
# for ax, Z in [(axA, u1_bond), (axB, u2_bond)]:
#     Z_smooth = gaussian_filter(Z, sigma=0.8)
#     cs = ax.contour(DTc, Gc, Z_smooth, levels=chi_levels, colors="white", linewidths=1.2, alpha=0.95)
#     labels = ax.clabel(cs, levels=chi_levels, fmt=lambda v: rf"$\chi={int(v)}$", inline=True, fontsize=8, colors="white", inline_spacing=2)
#     if labels:
#         for l in labels:
#             l.set_rotation(chi_rotations)
#             l.set_path_effects([pe.withStroke(linewidth=2, foreground="black", alpha=0.5)])

# Alpha Contours: Smooth A and B in log space
alpha_levels = [1.1, 1.3, 1.5]
chi_floor = 2.0
with np.errstate(divide='ignore', invalid='ignore'):
    A_log = np.log(u1_bond)
    B_log = np.log(u2_bond)
    A_f = np.nan_to_num(A_log, nan=np.nanmean(A_log) if not np.all(np.isnan(A_log)) else 0)
    B_f = np.nan_to_num(B_log, nan=np.nanmean(B_log) if not np.all(np.isnan(B_log)) else 0)
    A_s = gaussian_filter(A_f, sigma=(0.8, 1.0))
    B_s = gaussian_filter(B_f, sigma=(0.8, 1.0))
    Z_al_smooth = np.exp(A_s - B_s)
    # Mask unstable regions
    unstable_mask = (u1_bond < chi_floor) | (u2_bond < chi_floor) | np.isnan(Z_alpha)
    Z_al_smooth[unstable_mask] = np.nan

cs_al = axAlpha.contour(DTc, Gc, Z_al_smooth, levels=alpha_levels, colors="white", linewidths=0.8, alpha=0.8)
labels_al = axAlpha.clabel(cs_al, fmt=lambda v: f"{v:.1f}", inline=True, fontsize=7, colors="white")
if labels_al:
    for l in labels_al:
        l.set_path_effects([pe.withStroke(linewidth=2, foreground="black", alpha=0.5)])

# Anchor alpha=1
cs_anchor = axAlpha.contour(DTc, Gc, Z_al_smooth, levels=[1.0], colors="black", linewidths=1.3, zorder=6)
axAlpha.clabel(cs_anchor, fmt={1.0: r"$\alpha=1$"}, inline=True, fontsize=8, colors="black")

# ------------------------
# Axes & Ticking
# ------------------------
def set_scientific_log_ticks(ax, axis='x'):
    target = ax.xaxis if axis == 'x' else ax.yaxis
    target.set_major_locator(LogLocator(base=10))
    target.set_major_formatter(LogFormatterMathtext(base=10))
    target.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    target.set_minor_formatter(NullFormatter())

for ax, label in zip([axA, axB, axAlpha], ["(a)", "(b)", "(c)"]):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Time step $\delta t$")
    set_scientific_log_ticks(ax, 'x')
    set_scientific_log_ticks(ax, 'y')
    ax.tick_params(axis='x', rotation=15)
    
    # Internal Labels
    ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

axA.set_ylabel(r"Noise strength $\gamma$")
axB.tick_params(labelleft=False)
axAlpha.tick_params(labelleft=False)

axA.set_title("Unraveling A")
axB.set_title("Unraveling B")
axAlpha.set_title("Bond Inflation")

# ------------------------
# DP Lines (from convergence)
# ------------------------
def add_dp_lines(ax, annotate=False):
    dt_fine = np.logspace(np.log10(min(dt_list)), np.log10(max(dt_list)), 100)
    dt_label = 0.04
    for dp in dp_levels:
        g_fine = dp / dt_fine
        mask = (g_fine >= min(gamma_list)) & (g_fine <= max(gamma_list))
        if np.any(mask):
            # Consistent guide styling
            ax.plot(dt_fine[mask], g_fine[mask], "w--", lw=1.1, alpha=0.7, zorder=5)
            if annotate and dp in [1e-3, 1e-2, 1e-1]:
                g_at_dt = 1.5 * (dp / dt_label)
                if min(gamma_list) <= g_at_dt <= max(gamma_list):
                    label_text = rf"$\delta p = 10^{{{int(np.log10(dp))}}}$"
                    txt = ax.text(dt_label, g_at_dt, label_text, fontsize=8, rotation=-15, ha='center', va='center',
                                  color='white', weight='bold')
                    txt.set_path_effects([pe.withStroke(linewidth=2, foreground="black", alpha=0.6)])

add_dp_lines(axA, annotate=True)
add_dp_lines(axB, annotate=False)

# ------------------------
# Colorbars
# ------------------------
cb_chi = fig.colorbar(pcmA, cax=cax_chi)
cb_chi.set_ticks([8, 16, 32, 64])
cb_chi.set_ticklabels(["8", "16", "32", "64"])
cax_chi.set_title(r"$\overline{\chi}_{\mathrm{max}}$", pad=12, fontsize=10)

cb_al = fig.colorbar(pcm_al, cax=cax_alpha, ticks=[1.0, 1.5, 2.0])
cax_alpha.set_title(r"$\alpha = \chi_A / \chi_B$", pad=12, fontsize=10)

# ------------------------
# Alpha Stats
# ------------------------
valid_alpha = Z_alpha[~np.isnan(Z_alpha)]
mean_al = np.nanmean(valid_alpha) if valid_alpha.size > 0 else 0
std_al = np.nanstd(valid_alpha) if valid_alpha.size > 0 else 0
stats_text = rf"$\mu_\alpha = {mean_al:.2f}$"+"\n"+rf"$\sigma_\alpha = {std_al:.2f}$"
axAlpha.text(0.95, 0.95, stats_text, transform=axAlpha.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

plt.savefig("gamma_dt_alpha.pdf", dpi=300, bbox_inches="tight")
plt.savefig("gamma_dt_alpha.png", dpi=300, bbox_inches="tight")
plt.show()
