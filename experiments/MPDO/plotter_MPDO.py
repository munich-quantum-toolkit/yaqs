import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
import os
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

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
fig  = plt.figure(figsize=(7.2, 4.2))  # a size often acceptable for Nature

gs = GridSpec(3, 1, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
# ax4 = fig.add_subplot(gs[1, 2])

axes = [ax1, ax2, ax3]
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['lines.linewidth'] = 3
# plt.rcParams['pdf.fonttype'] = 42  # ensures fonts are embedded
# axes[0].set_frame_on(False)
# axes[0].tick_params(labeltop=False, top=False, labelright=False, right=False)

L = 30
# axes[0].set_xlabel('Time (Jt)', fontsize=12)
# axes[1].set_xlabel('Time (Jt)', fontsize=12)
axes[2].set_xlabel('Time ($Jt$)', fontsize=12)

axes[0].set_ylabel("Site", fontsize=12)
axes[1].set_ylabel("Site", fontsize=12)
axes[2].set_ylabel("Site", fontsize=12)

axes[0].set_xticks([])
axes[1].set_xticks([])
axes[2].set_xticks([x for x in list(range(0, 101, 10))], range(0, 11, 1))

axes[0].set_yticks([x-0.5 for x in [1, 10, 20, 29]], [1, 10, 20, 30])
axes[1].set_yticks([x-0.5 for x in [1, 10, 20, 29]], [1, 10, 20, 30])
axes[2].set_yticks([x-0.5 for x in [1, 10, 20, 29]], [1, 10, 20, 30])

axes[0].tick_params(labelsize=10)

# axes[0].set_title('TJM $(\\gamma=0)$')
# axes[1].set_title('TJM $(\\gamma=0.1)$')
# axes[2].set_title('MPDO $(\\gamma=0.1)$')

#########################################################################################
x = np.logspace(-2, 4)

data = pickle.load(open("30L_NoNoise.pickle", "rb"))
heatmap = []
for observable in data['sim_params'].observables:
    heatmap.append(observable.results)

im = axes[0].imshow(heatmap, aspect='auto', vmin=-1, vmax=1)

data = pickle.load(open("30L_Noise.pickle", "rb"))
heatmap = []
for observable in data['sim_params'].observables:
    heatmap.append(observable.results)

im = axes[1].imshow(heatmap, aspect='auto', vmin=-1, vmax=1)

data = pickle.load(open("30L_Noise.pickle", "rb"))
heatmap = []
for observable in data['sim_params'].observables:
    heatmap.append(observable.results)

im = axes[2].imshow(heatmap, aspect='auto', vmin=-1, vmax=1)

# Add vertical annotations on the left side above the "Site" label
axes[0].text(-0.125, 0.5, "TJM \n ($\\gamma = 0$)", fontsize=12, transform=axes[0].transAxes, va='center', ha='center', rotation=90)
axes[1].text(-0.125, 0.5, "TJM \n ($\\gamma = 0.1$)", fontsize=12, transform=axes[1].transAxes, va='center', ha='center', rotation=90)
axes[2].text(-0.125, 0.5, "MPDO \n ($\\gamma = 0.1$)", fontsize=12, transform=axes[2].transAxes, va='center', ha='center', rotation=90)


fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.9, 0.11, 0.025, 0.75])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title('$\\langle X \\rangle$')

plt.savefig("results.pdf", dpi=300, format="pdf")
plt.show()