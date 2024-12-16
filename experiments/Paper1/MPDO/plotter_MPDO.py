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

fig = plt.figure()
gs = GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

data = pickle.load(open("TJM_MPDO_Bond2.pickle", "rb"))
heatmap1000 = []
for observable in data['sim_params'].observables:
        heatmap1000.append(observable.results)

heatmap = np.array(heatmap1000)
L = 30
im = ax1.imshow(heatmap1000, cmap='viridis', aspect='auto', extent=[0, data['sim_params'].T, L, 0])

data = pickle.load(open("TJM_MPDO_Bond4.pickle", "rb"))
heatmap1000 = []
for observable in data['sim_params'].observables:
        heatmap1000.append(observable.results)

heatmap = np.array(heatmap1000)
L = 30
im = ax2.imshow(heatmap1000, cmap='viridis', aspect='auto', extent=[0, data['sim_params'].T, L, 0])

# ### Overall Plot
# fig = plt.figure()

# gs = GridSpec(3, 2, figure=fig)
# ax1 = fig.add_subplot(gs[:, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[1, 1])
# ax4 = fig.add_subplot(gs[2, 1])

# axes = [ax1, ax2, ax3, ax4]
# plt.rcParams.update({'font.size': 10})
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Arial'
# plt.rcParams['lines.linewidth'] = 2

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

# #########################################################################################
# colors = {'10': 'darkturquoise', '100': 'cornflowerblue', '1000': 'blue', 'Exact': 'k'}

# markers = {'10': 'o', '100': '^', '1000': 's', 'Exact': '*'}
# linestyles = {'10': '-', '100': '-', 'Exact': '--'}

# x = np.linspace(0.01, 1001)
# # Load data from file
# data10 = pickle.load(open("timesteps_10/trajectories_timesteps_10.pkl", "rb"))
# dt = 1 / data10['timesteps']
# exp_value_exact = data10['result_lindblad_interp'][-1]
# times10 = data10['time_points']

# # [Trajectories, Times]
# trajectories10 = np.array(data10['all_trajectories'])

# # Trajectories at T=1
# trajectoriesT = trajectories10[:, -1]

# # Use average_random_samples to calculate averaged errors with shaded error bars
# max_sample_size = 1000
# num_samples = 100
# errors = []
# std_devs = []

# for sample_size in range(1, max_sample_size + 1):
#     sample_errors = []
#     for _ in range(num_samples):
#         sample = np.random.choice(trajectoriesT, sample_size, replace=False)
#         exp_value_stochastic = np.mean(sample)
#         error = np.abs(exp_value_stochastic - exp_value_exact)
#         sample_errors.append(error)
#     errors.append(np.mean(sample_errors))
#     std_devs.append(np.std(sample_errors))

# x_values = range(1, max_sample_size + 1)
# errors = np.array(errors)
# std_devs = np.array(std_devs)
# p1_10, = axes[0].plot(x_values, errors, label='$10^{-1}$', color=colors['10'])
# axes[0].fill_between(x_values, errors - std_devs, errors + std_devs, color=colors['10'], alpha=0.3)

# # Load data from file for data100
# data100 = pickle.load(open("timesteps_100/trajectories_timesteps_100.pkl", "rb"))
# dt = 1 / data100['timesteps']
# exp_value_exact_100 = data100['result_lindblad_interp'][-1]
# times = data100['time_points']

# # [Trajectories, Times]
# trajectories100 = np.array(data100['all_trajectories'])

# # Trajectories at T=1
# trajectoriesT_100 = trajectories100[:, -1]

# # Use average_random_samples to calculate averaged errors for data100 with shaded error bars
# errors_100 = []
# std_devs_100 = []

# for sample_size in range(1, max_sample_size + 1):
#     sample_errors = []
#     for _ in range(num_samples):
#         sample = np.random.choice(trajectoriesT_100, sample_size, replace=False)
#         exp_value_stochastic = np.mean(sample)
#         error = np.abs(exp_value_stochastic - exp_value_exact_100)
#         sample_errors.append(error)
#     errors_100.append(np.mean(sample_errors))
#     std_devs_100.append(np.std(sample_errors))

# errors_100 = np.array(errors_100)
# std_devs_100 = np.array(std_devs_100)
# p1_100, = axes[0].plot(x_values, errors_100, label='$10^{-2}$', color=colors['100'])
# axes[0].fill_between(x_values, errors_100 - std_devs_100, errors_100 + std_devs_100, color=colors['100'], alpha=0.3)

# # Load data from file for data1000
# trajectories1000 = combine_trajectories("timesteps_1000")
# exp_value_exact_1000 = data10['result_lindblad_interp'][-1]  # Assuming same exact value for all datasets

# # Trajectories at T=1
# trajectoriesT_1000 = trajectories1000[:, -1]

# # Use average_random_samples to calculate averaged errors for data1000 with shaded error bars
# errors_1000 = []
# std_devs_1000 = []

# for sample_size in range(1, max_sample_size + 1):
#     sample_errors = []
#     for _ in range(num_samples):
#         sample = np.random.choice(trajectoriesT_1000, sample_size, replace=False)
#         exp_value_stochastic = np.mean(sample)
#         error = np.abs(exp_value_stochastic - exp_value_exact_1000)
#         sample_errors.append(error)
#     errors_1000.append(np.mean(sample_errors))
#     std_devs_1000.append(np.std(sample_errors))

# errors_1000 = np.array(errors_1000)
# std_devs_1000 = np.array(std_devs_1000)
# p1_1000, = axes[0].plot(x_values, errors_1000, label='$10^{-3}$', color=colors['1000'])
# axes[0].fill_between(x_values, errors_1000 - std_devs_1000, errors_1000 + std_devs_1000, color=colors['1000'], alpha=0.3)

# p1_exact = axes[0].plot(x, 0.1/np.sqrt(x), linestyle=linestyles['Exact'], label="$0.1/ \\sqrt{N}$", color=colors['Exact'], linewidth=2)

# axes[0].legend([p1_10, p1_100, p1_1000], ['$10^{-1}$', '$10^{-2}$', '$10^{-3}$'],
#                handler_map={tuple: HandlerTuple(ndivide=None)},
#                 loc='lower right',
#                 title='$\delta t$')

# axes[0].text(0.85, 0.95, '$Jt=1$', fontsize=12, fontweight='bold', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
# ##########################################################################################################

# data = pickle.load(open("QuTip_exact_convergence.pickle", "rb"))
# heatmap_exact = []
# for site in range(L):
#         heatmap_exact.append(data['observables'][site])

# data = pickle.load(open("TJM_convergence.pickle", "rb"))
# heatmap1000 = []
# for observable in data['sim_params'].observables:
#         heatmap1000.append(observable.results)

# heatmap1000 = np.array(heatmap1000)
# heatmap_exact = np.array(heatmap_exact)
# im = axes[3].imshow(np.abs(heatmap_exact-heatmap1000), cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1))

# heatmap10 = []
# trajectories = 10
# num_samples = 100
# for site in range(L):
#     sample_values = []
#     for _ in range(num_samples):
#         # Pick random trajectories for a given site
#         # sample = (trajectories, times)
#         indices = np.random.choice(data['sim_params'].observables[site].trajectories.shape[0], trajectories, replace=False)
#         samples = data['sim_params'].observables[site].trajectories[indices]

#         # Calculate average value at each time t
#         average = np.mean(samples, axis=0)

#         # Save result of sample
#         sample_values.append(average)

#     # Calculate average
#     heatmap10.append(np.mean(sample_values, axis=0))
# heatmap10 = np.array(heatmap10)
# im = axes[1].imshow(np.abs(heatmap_exact-heatmap10), cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1e-1))

# heatmap100 = []
# trajectories = 100
# num_samples = 100
# for site in range(L):
#     sample_values = []
#     for _ in range(num_samples):
#         # Pick random trajectories for a given site
#         # sample = (trajectories, times)
#         indices = np.random.choice(data['sim_params'].observables[site].trajectories.shape[0], sample_size, replace=False)
#         samples = data['sim_params'].observables[site].trajectories[indices]

#         # Calculate average value at each time t
#         average = np.mean(samples, axis=0)

#         # Save result of sample
#         sample_values.append(average)

#     # Calculate average
#     heatmap100.append(np.mean(sample_values, axis=0))
# heatmap100 = np.array(heatmap100)
# im = axes[2].imshow(np.abs(heatmap_exact-heatmap100), cmap='Reds', aspect='auto', extent=[0, data['sim_params'].T, L, 0], norm=LogNorm(vmin=1e-3, vmax=1e-1))

# fig.subplots_adjust(right=0.875)
# cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.75])
# cbar = fig.colorbar(im, cax=cbar_ax)
# cbar.ax.set_title('$\\epsilon$')
plt.savefig("results.pdf", dpi=300, format="pdf")
plt.show()