import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import os

def combine_trajectories(dir):
    dir = dir + '/'
    data = []
    for filename in os.listdir(dir):
        filepath = dir + filename
        batch = pickle.load(open(filepath, "rb"))
        for trajectory in batch['trajectories']:
            data.append(trajectory['expectation_values'])
    # [Trajectory, Expectation Values]
    return np.array(data)

### Overall Plot
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
    "lines.linewidth": 3
})

fig = plt.figure(figsize=(4, 3), layout='constrained')  # A size often acceptable for Nature
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])

ax1.set_xlabel('$s_\\text{max}$', fontsize=12)
ax1.set_ylabel("$\\epsilon(s_\\text{max})$", fontsize=12)
ax1.tick_params(labelsize=10)

# ---- Example function to plot one dataset with shaded variance/std dev
def plot_with_variance(ax, filepath, label, line_style='-', alpha_fill=0.2):
    data = pickle.load(open(filepath, "rb"))
    thresholds = data['thresholds']
    avg_errs = data['average_errors']
    
    # Example: if your data dictionary has 'error_variance'
    # or 'error_std_dev'; adapt as needed:
    if 'error_variance' in data:
        var_errs = data['error_variance']
        std_errs = var_errs
    elif 'error_std_dev' in data:
        std_errs = data['error_std_dev']
    else:
        # If no variance info is found, default to zero
        std_errs = np.zeros_like(avg_errs)
    
    ax.plot(thresholds, avg_errs, label=label, linestyle=line_style)
    ax.fill_between(
        thresholds,
        avg_errs - std_errs,
        avg_errs + std_errs,
        alpha=alpha_fill
    )

# ---- Plot each dataset with a fill_between for variance
plot_with_variance(ax1, "5_sites/linear.pickle", label='Linear', line_style='-')
plot_with_variance(ax1, "5_sites/sca.pickle",    label='SCA',    line_style='--')
plot_with_variance(ax1, "5_sites/full.pickle",   label='Full',   line_style=':')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.savefig("results.pdf", dpi=300, format="pdf")
plt.show()
