import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
import os
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter

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
fig  = plt.figure(figsize=(7.2, 3))  # a size often acceptable for Nature

gs = GridSpec(2, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4= fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])
axes = [[ax1, ax2, ax3], [ax4, ax5, ax6]]

axes[0][0].set(ylabel='$\\overline{T}$ (s)')
axes[1][0].set(ylabel='$\\sigma(T)$ (s)')
axes[1][0].set(xlabel='$|g|_{\\text{removed}}$')
axes[1][1].set(xlabel='$\\theta_{\\text{error}} / \\pi$')
axes[1][2].set(xlabel='$n_{\\text{SWAP}}$')

plt.setp(axes, ylim=(1e-1, 1e2), yscale='log')
for i in range(2):
    for j in range(3):
        axes[i][j].xaxis.grid()
        axes[i][j].yaxis.grid()
        if j != 0:
            axes[i][j].tick_params('y', labelleft=False)
        if i == 0:
            axes[i][j].tick_params('x', labelbottom=False)

# Error Injection
x = np.linspace(0, 16)

TN6 = pickle.load( open("TN6.p", "rb" ) )
TN6_qubits = np.array(TN6['N'])
TN6_times = np.mean(np.array(TN6['t']), axis=0)
axes[0][0].scatter(TN6_qubits, TN6_times, color='lightsalmon', marker='o')
a, b, c = np.polyfit(TN6_qubits, TN6_times, 2)
axes[0][0].plot(x, a*x**2 + b*x + c, color='lightsalmon')

TN_std = np.std(np.array(TN6['t']), axis=0)
axes[1][0].scatter(TN6_qubits, TN_std, color='lightsalmon', marker='o')
a, b = np.polyfit(TN6_qubits, TN_std, 1)
axes[1][0].plot(x, a*x + b, color='lightsalmon')

TN3 = pickle.load( open("TN3.p", "rb" ) )
TN3_qubits = np.array(TN3['N'])
TN3_times = np.mean(np.array(TN3['t']), axis=0)
axes[0][0].scatter(TN3_qubits, TN3_times, color='tab:red', marker='o')
a, b = np.polyfit(TN3_qubits, TN3_times, 1)
axes[0][0].plot(x, a*x + b, color='tab:red')

TN_std = np.std(np.array(TN3['t']), axis=0)
axes[1][0].scatter(TN3_qubits, TN_std, color='tab:red', marker='o')
a, b = np.polyfit(TN3_qubits, TN_std, 1)
axes[1][0].plot(x, a*x + b, color='tab:red')

TN1 = pickle.load( open("TN1.p", "rb" ) )
TN1_qubits = np.array(TN1['N'])
TN1_times = np.mean(np.array(TN1['t']), axis=0)
axes[0][0].scatter(TN1_qubits, TN1_times, color='black', marker='o')
a, b = np.polyfit(TN1_qubits, TN1_times, 1)
axes[0][0].plot(x, a*x + b, color='black')

TN_std = np.std(np.array(TN1['t']), axis=0)
axes[1][0].scatter(TN1_qubits, TN_std, color='black', marker='o')
a, b, c = np.polyfit(TN1_qubits, TN_std, 2)
axes[1][0].plot(x, a*x + b*x + c, color='black')

axes[0][0].set_xlim(-1, 12)
axes[1][0].set_xlim(-1, 12)
axes[1][0].set_ylim(1e-2, 1e1)

# # Phase Error
x = np.logspace(-3, 1, 100)*2
negative_x = np.array([-val for val in x])
TN6 = pickle.load( open("TN6_phase.p", "rb" ) )
TN6_qubits = np.array(TN6['N'])
TN6_times = np.mean(np.array(TN6['t']), axis=0)
axes[0][1].plot(TN6_qubits, TN6_times, color='lightsalmon', marker='o')

TN_std = np.std(np.array(TN6['t']), axis=0)
axes[1][1].scatter(TN6_qubits, TN_std, color='lightsalmon', marker='o')
axes[1][1].plot(TN6_qubits, TN_std, color='lightsalmon', marker='o')

x = np.logspace(-2, 1, 100)
negative_x = np.array([-val for val in x])
TN3 = pickle.load( open("TN3_phase.p", "rb" ) )
TN3_qubits = np.array(TN3['N'])
TN3_times = np.mean(np.array(TN3['t']), axis=0)
axes[0][1].scatter(TN3_qubits, TN3_times, color='tab:red', marker='o')
axes[0][1].plot(TN3_qubits[3:8], TN3_times[3:8], color='tab:red', marker='o')
a, b, c = np.polyfit(TN3_qubits[0:6], TN3_times[0:6], 2)
axes[0][1].plot(negative_x, a*negative_x**2 + b*negative_x + c, color='tab:red')
a, b, c = np.polyfit(TN3_qubits[5::], TN3_times[5::], 2)
axes[0][1].plot(x, a*x**2 + b*x + c, color='tab:red')

x = np.logspace(-3, 1, 100)*np.pi
negative_x = np.array([-val for val in x])
TN_std = np.std(np.array(TN3['t']), axis=0)
axes[1][1].scatter(TN3_qubits, TN_std, color='tab:red', marker='o')
a, b, c = np.polyfit(TN3_qubits[0:6], TN_std[0:6], 2)
axes[1][1].plot(negative_x, a*negative_x**2 + b*negative_x + c, color='tab:red')
a, b, c = np.polyfit(TN3_qubits[4::], TN_std[4::], 2)
axes[1][1].plot(x, a*x**2 + b*x + c, color='tab:red')
axes[1][1].plot(TN3_qubits[4:7], TN_std[4:7], color='tab:red', marker='o')

x = np.logspace(-2, 1, 100)
negative_x = np.array([-val for val in x])
TN1 = pickle.load( open("TN1_phase.p", "rb" ) )
TN1_qubits = np.array(TN1['N'], dtype=float)
TN1_times = np.mean(np.array(TN1['t']), axis=0)
axes[0][1].scatter(TN1_qubits, TN1_times, color='black', marker='o')
axes[0][1].plot(TN1_qubits[3:8], TN1_times[3:8], color='black', marker='o')
a, b, c = np.polyfit(TN1_qubits[0:6], TN1_times[0:6], 2)
axes[0][1].plot(negative_x, a*negative_x**2 + b*negative_x + c, color='black')
a, b, c = np.polyfit(TN1_qubits[5::], TN1_times[5::], 2)
axes[0][1].plot(x, a*x**2 + b*x + c, color='black')

x = np.logspace(-2, 1, 100)*0.5
negative_x = np.array([-val for val in x])
TN_std = np.std(np.array(TN1['t']), axis=0)
axes[1][1].scatter(TN1_qubits, TN_std, color='black', marker='o')
a, b, c = np.polyfit(TN1_qubits[0:6], TN_std[0:6], 2)
axes[1][1].plot(negative_x, a*negative_x**2 + b*negative_x + c, color='black')
a, b, c = np.polyfit(TN1_qubits[5::], TN_std[5::], 2)
axes[1][1].plot(x, a*x**2 + b*x + c, color='black')
axes[1][1].plot(TN1_qubits[3:7], TN_std[3:7], color='black', marker='o')

axes[0][1].set_xlim(-2*np.pi, 2*np.pi)
axes[0][1].set_xscale('symlog', linthresh=1e-3)

axes[1][1].set_xlim(-2*np.pi, 2*np.pi)
axes[1][1].set_xscale('symlog', linthresh=1e-3)
axes[0][1].set_xticks([-np.pi, -np.pi/10, -np.pi/100, -np.pi/1000, 0, np.pi/1000, np.pi/100, np.pi/10, np.pi], labels=['$-1$', '$-10^{-1}$', '$-10^{-2}$', '$-10^{-3}$', '0', '$10^{-3}$','$10^{-2}$', '$10^{-1}$', '$1$'], fontsize=8, rotation=45)
axes[1][1].set_xticks([-np.pi, -np.pi/10, -np.pi/100, -np.pi/1000, 0, np.pi/1000, np.pi/100, np.pi/10, np.pi], labels=['$-1$', '$-10^{-1}$', '$-10^{-2}$', '$-10^{-3}$', '0', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$'], fontsize=8, rotation=45)
axes[0][1].minorticks_off()
axes[1][1].minorticks_off()
axes[1][1].set_ylim(1e-2, 1e1)


# # Permutation Error
x = np.linspace(0, 100)
TN6 = pickle.load( open("TN6_permutation.p", "rb" ))
TN6_qubits = np.array(TN6['N'], dtype=float)
TN6_times = np.mean(np.array(TN6['t']), axis=0)
axes[0][2].scatter(TN6_qubits, TN6_times, color='lightsalmon', marker='o', label='$10^{-6}$')
a, b, c = np.polyfit(TN6_qubits, TN6_times, 2)
axes[0][2].plot(x, a*x**2 + b*x + c, color='lightsalmon')

TN_std = np.std(np.array(TN6['t']), axis=0)
axes[1][2].scatter(TN6_qubits, TN_std, color='lightsalmon', marker='o')
a, b = np.polyfit(TN6_qubits, TN_std, 1)
axes[1][2].plot(x, a*x + b, color='lightsalmon')

TN3 = pickle.load( open("TN3_permutation.p", "rb" ))
TN3_qubits = np.array(TN3['N'], dtype=float)
TN3_times = np.mean(np.array(TN3['t']), axis=0)
axes[0][2].scatter(TN3_qubits, TN3_times, color='tab:red', marker='o', label='$10^{-3}$')
a, b, c = np.polyfit(TN3_qubits, TN3_times, 2)
axes[0][2].plot(x, a*x**2 + b*x + c, color='tab:red')

TN_std = np.std(np.array(TN3['t']), axis=0)
axes[1][2].scatter(TN3_qubits, TN_std, color='tab:red', marker='o')
a, b  = np.polyfit(TN3_qubits, TN_std, 1)
axes[1][2].plot(x, a*x + b, color='tab:red')

TN1 = pickle.load( open("TN1_permutation.p", "rb" ))
TN1_qubits = np.array(TN1['N'], dtype=float)
TN1_times = np.mean(np.array(TN1['t']), axis=0)
axes[0][2].scatter(TN1_qubits, TN1_times, color='black', marker='o', label='$10^{-1}$')
a, b, c  = np.polyfit(TN1_qubits, TN1_times, 2)
axes[0][2].plot(x, a*x**2 + b*x + c, color='black')

TN_std = np.std(np.array(TN1['t']), axis=0)
axes[1][2].scatter(TN1_qubits, TN_std, color='black', marker='o')
a, b, c = np.polyfit(TN1_qubits, TN_std, 2)
axes[1][2].plot(x, a*x**2 + b*x + c, color='black')

axes[0][2].set_xlim(-5, 60)
axes[0][2].set_yscale('log')
# axes[0][2].set_xticks([0, 10, 20, 30, 40, 50])

axes[1][2].set_xlim(-5, 60)
axes[1][2].set_yscale('log')
# axes[1][2].set_xticks([0, 10, 20, 30, 40, 50])
axes[1][2].set_ylim(1e-2, 1e1)

fig.legend(loc=7, title='$s_\\text{max}$')
fig.tight_layout()
fig.subplots_adjust(right=0.86)
plt.savefig("results.pdf", format="pdf")

plt.show()