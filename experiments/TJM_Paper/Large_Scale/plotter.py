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
fig  = plt.figure(figsize=(7.2, 4.2))  # a size often acceptable for Nature

gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
# ax4 = fig.add_subplot(gs[1, 2])

axes = [ax1, ax2, ax3]
L = 100
axes[2].set_xlabel('Time $(Jt)$', fontsize=12)

axes[0].set_ylabel("Site", fontsize=12)
axes[1].set_ylabel("Site", fontsize=12)
axes[2].set_ylabel("Site", fontsize=12)

axes[0].set_xticks([])
axes[1].set_xticks([])
axes[2].set_xticks([x for x in list(range(0, 101, 10))], range(0, 11, 1))

axes[0].set_yticks([x-0.5 for x in [1, 25, 50, 75, 100]], [1, 250, 500, 750, 1000])
axes[1].set_yticks([x-0.5 for x in [1, 25, 50, 75, 100]], [1, 250, 500, 750, 1000])
axes[2].set_yticks([x-0.5 for x in [1, 25, 50, 75, 100]], [1, 250, 500, 750, 1000])

axes[0].tick_params(labelsize=10)

# axes[0].set_title('$Jt=1$')
# axes[1].set_title('$Jt=5$')
# axes[2].set_title('$Jt=10$')

#########################################################################################

data = pickle.load(open("Wall_T10/1000L_Exact.pickle", "rb"))
heatmap_exact = []
for observable in data['sim_params'].observables:
    heatmap_exact.append(observable.results)
heatmap_exact = np.array(heatmap_exact)
im_exact = axes[0].imshow(heatmap_exact, aspect='auto', extent=(0, 100, 100, 0), vmin=-1, vmax=1)

data = pickle.load(open("Wall_T10/1000L_Gamma01_Batch1_50N.pickle", "rb"))
heatmap = []
for observable in data['sim_params'].observables:
    heatmap.append(observable.results)
heatmap = np.array(heatmap)
im = axes[1].imshow(heatmap, aspect='auto', extent=(0, 100, 100, 0), vmin=-1, vmax=1)

im = axes[2].imshow(heatmap-heatmap_exact, cmap='coolwarm', aspect='auto', extent=(0, 100, 100, 0), vmin=-0.1, vmax=0.1)

axes[0].text(-0.125, 0.5, "$\\gamma = 0$", fontsize=12, transform=axes[0].transAxes, va='center', ha='center', rotation=90)
axes[1].text(-0.125, 0.5, "$\\gamma = 0.1$", fontsize=12, transform=axes[1].transAxes, va='center', ha='center', rotation=90)
# axes[2].text(-0.125, 0.5, "$ \\Delta $", fontsize=12, transform=axes[2].transAxes, va='center', ha='center', rotation=90)


fig.subplots_adjust(top=0.95, right=0.88)
cbar_ax = fig.add_axes([0.9, 0.42, 0.025, 0.48])
cbar = fig.colorbar(im_exact, cax=cbar_ax)
cbar.ax.set_title('$\\langle Z \\rangle$')

cbar_ax = fig.add_axes([0.9, 0.12, 0.025, 0.225])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title('$\\Delta$')

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