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
fig = plt.figure(layout='compressed')

gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1, 0])
# ax3 = fig.add_subplot(gs[1, 1])
# ax4 = fig.add_subplot(gs[1, 2])

axes = [ax1]
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['lines.linewidth'] = 2

axes[0].set_xlabel('Trajectories (N)', fontsize=12)
axes[0].set_ylabel("$|\\langle X^{[5]} \\rangle - \\langle \\tilde{X}^{[5]} \\rangle|$", fontsize=12)
axes[0].tick_params(labelsize=10)
axes[0].set_xlim(1, 1e4)
axes[0].yaxis.grid(linestyle='--')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_ylim(1e-3, 1e-1)
# axes[1].tick_params(labelsize=10)
# axes[2].tick_params(labelsize=10)
# axes[3].tick_params(labelsize=10)
# axes[1].set_xlabel('Time (Jt)', fontsize=12)
# axes[2].set_xlabel('Time (Jt)', fontsize=12)
# axes[3].set_xlabel('Time (Jt)', fontsize=12)
# axes[1].set_ylabel('Site', fontsize=12)
# axes[2].set_ylabel('Site', fontsize=12, labelpad=-2)
# axes[3].set_ylabel('Site', fontsize=12, labelpad=-2)
# L = 10
# axes[1].set_yticks([x-0.5 for x in list(range(2,L+2, 2))], range(2,L+2, 2))
# # axes[2].set_yticks([x-0.5 for x in list(range(2,L+2, 2))], range(2,L+2, 2))
# # axes[3].set_yticks([x-0.5 for x in list(range(2,L+2, 2))], range(2,L+2, 2))
# axes[2].set_yticks([])
# axes[3].set_yticks([])

# axes[1].set_title('$N=10$')
# axes[2].set_title('$N=100$')
# axes[3].set_title('$N=1000$')

#########################################################################################
colors = {'10': 'darkturquoise', '100': 'cornflowerblue', '1000': 'blue', 'Exact': 'k'}

markers = {'10': 'o', '100': '^', '1000': 's', 'Exact': '*'}
linestyles = {'10': '-', '100': '-', 'Exact': '--'}

x = np.linspace(0.01, 10001)
# Load data from file
data = pickle.load(open("QuTip_exact_convergence.pickle", "rb"))
# Site 5, Time T=1
exp_value_exact = data['observables'][4][10]


# data = pickle.load(open("TJM_Convergence_dt1.pickle", "rb"))
# times = data['sim_params'].observables[4].times

# # [Trajectories, Times]
# trajectories =  data['sim_params'].observables[4].trajectories.squeeze()

# # Use average_random_samples to calculate averaged errors with shaded error bars
max_sample_size = 10000
num_samples = 100
# errors = []
# std_devs = []

# for sample_size in range(1, max_sample_size + 1):
#     sample_errors = []
#     for _ in range(num_samples):
#         sample = np.random.choice(trajectories, sample_size, replace=False)
#         exp_value_stochastic = np.mean(sample)
#         error = np.abs(exp_value_stochastic - exp_value_exact)
#         sample_errors.append(error)
#     errors.append(np.mean(sample_errors))
#     std_devs.append(np.std(sample_errors))

# x_values = range(1, max_sample_size + 1)
# errors = np.array(errors)
# std_devs = np.array(std_devs)
# p1_1, = axes[0].plot(x_values, errors, label='$1$')
# axes[0].fill_between(x_values, errors - std_devs, errors + std_devs, alpha=0.3)

# data = pickle.load(open("TJM_Convergence_dt05.pickle", "rb"))
# times = data['sim_params'].observables[4].times

# # [Trajectories, Times]
# trajectories =  data['sim_params'].observables[4].trajectories.squeeze()

# # Use average_random_samples to calculate averaged errors with shaded error bars
# errors = []
# std_devs = []

# for sample_size in range(1, max_sample_size + 1):
#     sample_errors = []
#     for _ in range(num_samples):
#         sample = np.random.choice(trajectories, sample_size, replace=False)
#         exp_value_stochastic = np.mean(sample)
#         error = np.abs(exp_value_stochastic - exp_value_exact)
#         sample_errors.append(error)
#     errors.append(np.mean(sample_errors))
#     std_devs.append(np.std(sample_errors))

# x_values = range(1, max_sample_size + 1)
# errors = np.array(errors)
# std_devs = np.array(std_devs)
# p1_05, = axes[0].plot(x_values, errors, label='$0.5$')
# axes[0].fill_between(x_values, errors - std_devs, errors + std_devs, alpha=0.3)

data = pickle.load(open("2nd_Order/TJM_Convergence_dt01.pickle", "rb"))
times = data['sim_params'].observables[4].times

# [Trajectories, Times]
trajectories =  data['sim_params'].observables[4].trajectories.squeeze()

# Use average_random_samples to calculate averaged errors with shaded error bars
errors = []
std_devs = []

for sample_size in range(1, max_sample_size + 1):
    sample_errors = []
    for _ in range(num_samples):
        sample = np.random.choice(trajectories, sample_size, replace=False)
        exp_value_stochastic = np.mean(sample)
        error = np.abs(exp_value_stochastic - exp_value_exact)
        sample_errors.append(error)
    errors.append(np.mean(sample_errors))
    std_devs.append(np.std(sample_errors))

x_values = range(1, max_sample_size + 1)
errors = np.array(errors)
std_devs = np.array(std_devs)
p1_05, = axes[0].plot(x_values, errors, label='$0.1$')
axes[0].fill_between(x_values, errors - std_devs, errors + std_devs, alpha=0.3)

data = pickle.load(open("TJM_Convergence_dt01.pickle", "rb"))
times = data['sim_params'].observables[4].times

# [Trajectories, Times]
trajectories =  data['sim_params'].observables[4].trajectories.squeeze()

# Use average_random_samples to calculate averaged errors with shaded error bars
errors = []
std_devs = []

for sample_size in range(1, max_sample_size + 1):
    sample_errors = []
    for _ in range(num_samples):
        sample = np.random.choice(trajectories, sample_size, replace=False)
        exp_value_stochastic = np.mean(sample)
        error = np.abs(exp_value_stochastic - exp_value_exact)
        sample_errors.append(error)
    errors.append(np.mean(sample_errors))
    std_devs.append(np.std(sample_errors))

x_values = range(1, max_sample_size + 1)
errors = np.array(errors)
std_devs = np.array(std_devs)
p1_05, = axes[0].plot(x_values, errors, label='$0.1$')
axes[0].fill_between(x_values, errors - std_devs, errors + std_devs, alpha=0.3)

# data = pickle.load(open("TJM_Convergence_dt005.pickle", "rb"))
# times = data['sim_params'].observables[4].times

# # [Trajectories, Times]
# trajectories =  data['sim_params'].observables[4].trajectories.squeeze()

# # Use average_random_samples to calculate averaged errors with shaded error bars
# errors = []
# std_devs = []

# for sample_size in range(1, max_sample_size + 1):
#     sample_errors = []
#     for _ in range(num_samples):
#         sample = np.random.choice(trajectories, sample_size, replace=False)
#         exp_value_stochastic = np.mean(sample)
#         error = np.abs(exp_value_stochastic - exp_value_exact)
#         sample_errors.append(error)
#     errors.append(np.mean(sample_errors))
#     std_devs.append(np.std(sample_errors))

# x_values = range(1, max_sample_size + 1)
# errors = np.array(errors)
# std_devs = np.array(std_devs)
# p1_05, = axes[0].plot(x_values, errors, label='$0.05$')
# axes[0].fill_between(x_values, errors - std_devs, errors + std_devs, alpha=0.3)

p1_exact = axes[0].plot(x, 0.1/np.sqrt(x), linestyle=linestyles['Exact'], label="$0.1/ \\sqrt{N}$", color=colors['Exact'], linewidth=2)

# axes[0].legend([p1_10, p1_100, p1_1000], ['$10^{-1}$', '$10^{-2}$', '$10^{-3}$'],
#                handler_map={tuple: HandlerTuple(ndivide=None)},
#                 loc='upper right',
#                 title='$\\delta t$')

# axes[0].text(0.85, 0.9, '$Jt=1$', fontsize=12, fontweight='bold', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
##########################################################################################################
# axes[0].legend()
plt.savefig("results.pdf", dpi=300, format="pdf")
plt.show()