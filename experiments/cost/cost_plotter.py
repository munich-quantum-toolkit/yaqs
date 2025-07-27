import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import make_interp_spline

import pickle

filename = "results_48.pickle"
with open(filename, 'rb') as f:
    results = pickle.load(f)

results = np.array(results).T

# ----- Configure global style -----
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "font.size": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.linewidth": 1.0,
    "legend.fontsize": 12,
    "figure.figsize": (6, 4)
})

# ----- Input data -----
results = np.array(results)
num_curves = len(results)

# Set up figure and axis
fig, ax = plt.subplots()

# Create colormap and normalize
cmap = plt.get_cmap('magma_r')
norm = mcolors.Normalize(vmin=0, vmax=num_curves - 1)
colors = cmap(np.linspace(0, 1, num_curves))
gammas = np.logspace(-3, 3, 30)
# Smooth and plot each curve
for j in range(num_curves):
    x = np.array(gammas)
    y = results[j]
    # if len(x) >= 4:
    #     x_smooth = np.logspace(np.log10(x[0]), np.log10(x[-1]), 300)
    #     spline = make_interp_spline(np.log10(x), y, k=3)
    #     y_smooth = spline(np.log10(x_smooth))
    #     ax.plot(x_smooth, y_smooth, color=colors[j], lw=1.5)
    # else:
    ax.plot(x, y, color=colors[j], lw=1.5)

# Log scale and labels
ax.set_ylim(1e1, 1e9)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$\\gamma$ (Depolarizing noise)', labelpad=5)
ax.set_ylabel('Runtime cost', labelpad=5)

# Clean up axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Trotter steps', labelpad=5)
cbar.ax.tick_params(direction='out', length=3)

# Tight layout
plt.tight_layout()
plt.show()
