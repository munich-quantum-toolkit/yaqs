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


# def average_random_samples(data, max_sample_size=10, num_samples=100):
#     avg_values = []
    
#     for sample_size in range(2, max_sample_size + 1):
#         averages = []
#         for _ in range(num_samples):
#             sample = np.random.choice(data, sample_size, replace=False)
#             averages.append(np.mean(sample))
#         avg_values.append(np.mean(averages))
    
#     return avg_values


### Overall Plot
# fig = plt.figure()

fig, axes = plt.subplots(3, 3)
# gs = GridSpec(3, 2, figure=fig)
# ax1 = fig.add_subplot(gs[:, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[1, 1])
# ax4 = fig.add_subplot(gs[2, 1])

L = 10
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['lines.linewidth'] = 2


axes[0, 0].set_title("$\\chi=2$", fontsize=12)
axes[0, 1].set_title("$\\chi=4$", fontsize=12)
axes[0, 2].set_title("$\\chi=8$", fontsize=12)
axes[0, 0].set_ylabel("$N=10$", fontsize=12)
axes[1, 0].set_ylabel("$N=100$", fontsize=12)
axes[2, 0].set_ylabel("$N=1000$", fontsize=12)
axes[0, 0].set_ylabel("Site", fontsize=12)
axes[1, 0].set_ylabel("Site", fontsize=12)
axes[2, 0].set_ylabel("Site", fontsize=12)
axes[0, 0].set_yticks([x-0.5 for x in list(range(2,L+2, 2))], range(2,L+2, 2))
axes[1, 0].set_yticks([x-0.5 for x in list(range(2,L+2, 2))], range(2,L+2, 2))
axes[2, 0].set_yticks([x-0.5 for x in list(range(2,L+2, 2))], range(2,L+2, 2))
axes[2, 0].set_xlabel("Time (Jt)", fontsize=12)
axes[2, 1].set_xlabel("Time (Jt)", fontsize=12)
axes[2, 2].set_xlabel("Time (Jt)", fontsize=12)

axes[0, 0].set_xticks([])
axes[0, 1].set_xticks([])
axes[0, 2].set_xticks([])
axes[1, 0].set_xticks([])
axes[1, 1].set_xticks([])
axes[1, 2].set_xticks([])
axes[0, 1].set_yticks([])
axes[0, 2].set_yticks([])
axes[1, 1].set_yticks([])
axes[1, 2].set_yticks([])
axes[2, 1].set_yticks([])
axes[2, 2].set_yticks([])

# axes[0].set_xlabel('Trajectories (N)', fontsize=12)
# axes[0].set_ylabel("$|\\langle X^{[5]} \\rangle - \\langle \\tilde{X}^{[5]} \\rangle|$", fontsize=12)
# axes[0].tick_params(labelsize=10)
# axes[0].set_xlim(1, 1e3)
# axes[0].yaxis.grid(linestyle='--')
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')
# axes[0].set_ylim(1e-6, 1)
# axes[1].tick_params(labelsize=10)
# axes[2].tick_params(labelsize=10)
# axes[3].tick_params(labelsize=10)
# axes[3].set_xlabel('Time (Jt)', fontsize=12)
# axes[1].set_ylabel('Site', fontsize=12, labelpad=-2)
# axes[2].set_ylabel('Site', fontsize=12, labelpad=-2)
# axes[3].set_ylabel('Site', fontsize=12, labelpad=-2)
# L = 10


# axes[1].set_yticks([x-0.5 for x in list(range(2,L+2, 2))], range(2,L+2, 2))
# axes[2].set_yticks([x-0.5 for x in list(range(2,L+2, 2))], range(2,L+2, 2))
# axes[3].set_yticks([x-0.5 for x in list(range(2,L+2, 2))], range(2,L+2, 2))
# axes[1].set_xticks([])
# axes[2].set_xticks([])

# axes[1].set_title('$N=10$', y=0.95)
# axes[2].set_title('$N=100$', y=0.95)
# axes[3].set_title('$N=1000$', y=0.95)


##########################################################################################################
L = 10
data = pickle.load(open("QuTip_exact_convergence.pickle", "rb"))
heatmap_exact = []
for site in range(L):
    heatmap_exact.append(data['observables'][site])

data = pickle.load(open("TJM_convergence_Bond2.pickle", "rb"))
heatmap1000 = []
for observable in data['sim_params'].observables:
        heatmap1000.append(observable.results)

heatmap1000 = np.array(heatmap1000)
heatmap_exact = np.array(heatmap_exact)
im = axes[2, 0].imshow(np.abs(heatmap_exact-heatmap1000), cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1e-1))

trajectories = 100
num_samples = 100
error_heatmap = []
for _ in range(num_samples):
    indices = np.random.choice(data['sim_params'].observables[0].trajectories.shape[0], trajectories, replace=False)
    heatmap = []
    for site in range(L):
        samples = data['sim_params'].observables[site].trajectories[indices]
        heatmap.append(np.mean(samples, axis=0))
    heatmap = np.array(heatmap)
    error_heatmap.append(np.abs(heatmap_exact-heatmap))
error_heatmap = np.array(error_heatmap)
error_heatmap = np.mean(error_heatmap, axis=0)
im = axes[0, 0].imshow(error_heatmap, cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1e-1))

trajectories = 1000
num_samples = 100
error_heatmap = []
for _ in range(num_samples):
    indices = np.random.choice(data['sim_params'].observables[0].trajectories.shape[0], trajectories, replace=False)
    heatmap = []
    for site in range(L):
        samples = data['sim_params'].observables[site].trajectories[indices]
        heatmap.append(np.mean(samples, axis=0))
    heatmap = np.array(heatmap)
    error_heatmap.append(np.abs(heatmap_exact-heatmap))
error_heatmap = np.array(error_heatmap)
error_heatmap = np.mean(error_heatmap, axis=0)
im = axes[1, 0].imshow(error_heatmap, cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1e-1))

data = pickle.load(open("TJM_convergence_Bond4.pickle", "rb"))
heatmap1000 = []
for observable in data['sim_params'].observables:
        heatmap1000.append(observable.results)

heatmap1000 = np.array(heatmap1000)
heatmap_exact = np.array(heatmap_exact)
im = axes[2, 1].imshow(np.abs(heatmap_exact-heatmap1000), cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1e-1))

trajectories = 100
num_samples = 100
error_heatmap = []
for _ in range(num_samples):
    indices = np.random.choice(data['sim_params'].observables[0].trajectories.shape[0], trajectories, replace=False)
    heatmap = []
    for site in range(L):
        samples = data['sim_params'].observables[site].trajectories[indices]
        heatmap.append(np.mean(samples, axis=0))
    heatmap = np.array(heatmap)
    error_heatmap.append(np.abs(heatmap_exact-heatmap))
error_heatmap = np.array(error_heatmap)
error_heatmap = np.mean(error_heatmap, axis=0)
im = axes[0, 1].imshow(error_heatmap, cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1e-1))

trajectories = 1000
num_samples = 100
error_heatmap = []
for _ in range(num_samples):
    indices = np.random.choice(data['sim_params'].observables[0].trajectories.shape[0], trajectories, replace=False)
    heatmap = []
    for site in range(L):
        samples = data['sim_params'].observables[site].trajectories[indices]
        heatmap.append(np.mean(samples, axis=0))
    heatmap = np.array(heatmap)
    error_heatmap.append(np.abs(heatmap_exact-heatmap))
error_heatmap = np.array(error_heatmap)
error_heatmap = np.mean(error_heatmap, axis=0)
im = axes[1, 1].imshow(error_heatmap, cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1e-1))

data = pickle.load(open("TJM_convergence_Bond8.pickle", "rb"))
heatmap1000 = []
for observable in data['sim_params'].observables:
        heatmap1000.append(observable.results)

heatmap1000 = np.array(heatmap1000)
heatmap_exact = np.array(heatmap_exact)
im = axes[2, 2].imshow(np.abs(heatmap_exact-heatmap1000), cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1e-1))

trajectories = 100
num_samples = 100
error_heatmap = []
for _ in range(num_samples):
    indices = np.random.choice(data['sim_params'].observables[0].trajectories.shape[0], trajectories, replace=False)
    heatmap = []
    for site in range(L):
        samples = data['sim_params'].observables[site].trajectories[indices]
        heatmap.append(np.mean(samples, axis=0))
    heatmap = np.array(heatmap)
    error_heatmap.append(np.abs(heatmap_exact-heatmap))
error_heatmap = np.array(error_heatmap)
error_heatmap = np.mean(error_heatmap, axis=0)
im = axes[0, 2].imshow(error_heatmap, cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1e-1))

trajectories = 1000
num_samples = 100
error_heatmap = []
for _ in range(num_samples):
    indices = np.random.choice(data['sim_params'].observables[0].trajectories.shape[0], trajectories, replace=False)
    heatmap = []
    for site in range(L):
        samples = data['sim_params'].observables[site].trajectories[indices]
        heatmap.append(np.mean(samples, axis=0))
    heatmap = np.array(heatmap)
    error_heatmap.append(np.abs(heatmap_exact-heatmap))
error_heatmap = np.array(error_heatmap)
error_heatmap = np.mean(error_heatmap, axis=0)
im = axes[1, 2].imshow(error_heatmap, cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1e-1))

# Adjust the layout to make room for vertically rotated labels
fig.subplots_adjust(left=0.2, right=0.875, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)

# Add vertical annotations on the left side above the "Site" label
axes[0, 0].text(-0.55, 0.5, "$N=100$", fontsize=12, transform=axes[0, 0].transAxes, va='center', ha='center', rotation=90)
axes[1, 0].text(-0.55, 0.5, "$N=1000$", fontsize=12, transform=axes[1, 0].transAxes, va='center', ha='center', rotation=90)
axes[2, 0].text(-0.55, 0.5, "$N=10000$", fontsize=12, transform=axes[2, 0].transAxes, va='center', ha='center', rotation=90)

# Add "Average" and "Typical" above the vertical labels
axes[0, 0].text(-0.75, 0.5, "Average", fontsize=12, transform=axes[0, 0].transAxes, va='center', ha='center', rotation=90)
axes[1, 0].text(-0.75, 0.5, "Average", fontsize=12, transform=axes[1, 0].transAxes, va='center', ha='center', rotation=90)
axes[2, 0].text(-0.75, 0.5, "Typical", fontsize=12, transform=axes[2, 0].transAxes, va='center', ha='center', rotation=90)

cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.75])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title('$\\epsilon$')

# Add a dotted line between row 2 and row 3
fig_width, fig_height = fig.get_size_inches()
line_y = 0.35  # Approximate position between the second and third rows (normalized figure coordinate)

# Draw the dotted line across the entire figure width
fig.add_artist(plt.Line2D([0.025, 0.875], [line_y, line_y], transform=fig.transFigure, color='black', linestyle='dotted', linewidth=1))

plt.savefig("results.pdf", dpi=300, format="pdf")
plt.show()