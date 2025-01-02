import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
import os
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter

def combine_trajectories(dir):
    dir = dir + '/'
    data = []
    for filename in os.listdir(dir):
        filepath = dir + filename
        batch = pickle.load(open(filepath, "rb" ))
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
fig  = plt.figure(figsize=(7.2, 5))  # a size often acceptable for Nature

gs = GridSpec(1, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
# ax4 = fig.add_subplot(gs[1, 2])

axes = [ax1, ax2, ax3]
L = 100
axes[0].set_xlabel('$\\gamma$', fontsize=12)
axes[1].set_xlabel('$\\gamma$', fontsize=12)
axes[2].set_xlabel('$\\gamma$', fontsize=12)

axes[0].set_ylabel("Site", fontsize=12)
# from matplotlib.ticker import ScalarFormatter
# for axis in [axes[0].xaxis, axes[0].yaxis]:
#     axis.set_major_formatter(ScalarFormatter())
# axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# axes[0].set_xticks([x for x in list(np.logspace(-4, 1, 5))], np.logspace(-4, 1, 5))
# axes[1].set_xticks([x for x in list(range(0, 101, 10))], range(0, 11, 1))
# axes[2].set_xticks([x for x in list(range(0, 101, 10))], range(0, 11, 1))

axes[0].set_yticks([x-0.5 for x in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]], [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
axes[1].set_yticks([])
axes[2].set_yticks([])

axes[0].tick_params(labelsize=10)

axes[0].set_title('$Jt=1$')
axes[1].set_title('$Jt=5$')
axes[2].set_title('$Jt=10$')

desired_logvals = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]

x = np.logspace(-4, 1, 100)
rotation = 30
# For each of those log-values, find which column in [0..99] is the best match
# i.e., the column whose x_log is closest to that value.
def closest_col(value, x_array):
    return np.argmin(np.abs(x_array - value))

tick_positions = [closest_col(val, x) for val in desired_logvals]

# Turn them into nice log-like labels: 10^-4, 10^-3, ...
tick_labels = [fr'$10^{{{int(np.log10(val))}}}$' for val in desired_logvals]

#########################################################################################

heatmap = pickle.load(open("100L_T1_wall.pickle", "rb"))['heatmap']#
im = axes[0].imshow(heatmap, aspect='auto', extent=(0, 100, 100, 0), vmin=-1, vmax=1)
axes[0].set_xticks(tick_positions)
axes[0].set_xticklabels(tick_labels, rotation=rotation)

heatmap = pickle.load(open("100L_T5_wall.pickle", "rb"))['heatmap']
im = axes[1].imshow(heatmap, aspect='auto', extent=(0, 100, 100, 0), vmin=-1, vmax=1)
axes[1].set_xticks(tick_positions)
axes[1].set_xticklabels(tick_labels, rotation=rotation)

heatmap = pickle.load(open("100L_T10_wall.pickle", "rb"))['heatmap']
im = axes[2].imshow(heatmap, aspect='auto', extent=(0, 100, 100, 0), vmin=-1, vmax=1)
axes[2].set_xticks(tick_positions)
axes[2].set_xticklabels(tick_labels, rotation=rotation)

fig.subplots_adjust(top=0.95, right=0.88)
cbar_ax = fig.add_axes([0.9, 0.11, 0.025, 0.8])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title('$\\langle Z \\rangle$')
# axes[0].legend(title='$\\delta t$', loc='upper right')
# axes[0].legend([p1_10, p1_100, p1_1000], ['$10^{-1}$', '$10^{-2}$', '$10^{-3}$'],
#                handler_map={tuple: HandlerTuple(ndivide=None)},
#                 loc='upper right',
#                 title='$\\delta t$')

# axes[0].text(0.85, 0.9, '$Jt=1$', fontsize=12, fontweight='bold', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
##########################################################################################################
# axes[0].legend()
plt.savefig("results.pdf", dpi=300, format="pdf")
plt.show()