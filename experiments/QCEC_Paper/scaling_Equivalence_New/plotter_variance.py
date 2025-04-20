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

x = np.linspace(0, 25)
y = x
# fig, axes = plt.subplots(2, 3, figsize=set_size(514.1736, subplots=(2, 3)))
axes[0][0].set_title('Linear')
axes[0][1].set_title('SCA')
axes[0][2].set_title('Full')

axes[0][0].set(ylabel='$\\overline{T}$ (s)')
axes[1][0].set(ylabel='$\\sigma(T)$ (s)')
axes[1][0].set(xlabel='n')
axes[1][1].set(xlabel='n')
axes[1][2].set(xlabel='n')

plt.setp(axes, xlim=(1, 32), ylim=(1e-3, 1e2), yscale='log')
for i in range(2):
    for j in range(3):
        axes[i][j].xaxis.grid()
        axes[i][j].yaxis.grid()
        if j != 0:
            axes[i][j].tick_params('y', labelleft=False)
        if i == 0:
            axes[i][j].tick_params('x', labelbottom=False)

# Linear
x = np.linspace(2, 50)

ZX = pickle.load( open("ZX_Lin.p", "rb" ) )
ZX_qubits = np.array(ZX['N'])
ZX_times = np.mean(np.array(ZX['t']), axis=0)
a, b, c = np.polyfit(np.log10(ZX_qubits), np.log10(ZX_times), 2)
axes[0][0].scatter(ZX_qubits, ZX_times, marker='X', color='tab:green')
axes[0][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle='--', color='tab:green')

ZX_std = np.std(np.array(ZX['t']), axis=0)
a, b, c = np.polyfit(np.log10(ZX_qubits), np.log10(ZX_std), 2)
axes[1][0].scatter(ZX_qubits, ZX_std, color='tab:green', linestyle='--', marker='X')
axes[1][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle='--', color='tab:green')

DD = pickle.load( open("DD_Lin.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
axes[0][0].scatter(DD_qubits, DD_times, marker='^', color='tab:blue')
a, b, c, d = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 3)
axes[0][0].plot(x, 10**(a*np.log10(x)**3 + b*np.log10(x)**2 + c*np.log10(x) + d), linestyle=':', color='tab:blue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c, d = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 3)
axes[1][0].scatter(DD_qubits, DD_std, color='tab:blue', marker='^')
axes[1][0].plot(x, 10**(a*np.log10(x)**3 + b*np.log10(x)**2 + c*np.log10(x) + d), linestyle=':', color='tab:blue')

proportional = pickle.load( open("TN_Lin.p", "rb" ) )
proportional_qubits = np.array(proportional['N'])
proportional_times = np.mean(np.array(proportional['t']), axis=0)
a, b, c = np.polyfit(np.log10(proportional_qubits), np.log10(proportional_times), 2)
axes[0][0].scatter(proportional_qubits, proportional_times, color='tab:red', marker='o')
axes[0][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

TN_std = np.std(np.array(proportional['t']), axis=0)
a, b, c = np.polyfit(np.log10(proportional_qubits), np.log10(TN_std), 2)
axes[1][0].scatter(proportional_qubits, TN_std, color='tab:red', marker='o')
axes[1][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

# SCA

ZX = pickle.load( open("ZX_SCA.p", "rb" ) )
ZX_qubits = np.array(ZX['N'])
ZX_times = np.mean(np.array(ZX['t']), axis=0)
a, b, c = np.polyfit(np.log10(ZX_qubits), np.log10(ZX_times), 2)
axes[0][1].scatter(ZX_qubits, ZX_times, marker='X', color='tab:green')
axes[0][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle='--', color='tab:green')

ZX_std = np.std(np.array(ZX['t']), axis=0)
a, b, c = np.polyfit(np.log10(ZX_qubits), np.log10(ZX_std), 2)
axes[1][1].scatter(ZX_qubits, ZX_std, color='tab:green', linestyle='--', marker='X')
axes[1][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle='--', color='tab:green')

DD = pickle.load( open("DD_SCA.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
axes[0][1].scatter(DD_qubits, DD_times, marker='^', color='tab:blue')
a, b, c, d = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 3)
axes[0][1].plot(x, 10**(a*np.log10(x)**3 + b*np.log10(x)**2 + c*np.log10(x) + d), linestyle=':', color='tab:blue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c, d = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 3)
axes[1][1].scatter(DD_qubits, DD_std, color='tab:blue', marker='^')
axes[1][1].plot(x, 10**(a*np.log10(x)**3 + b*np.log10(x)**2 + c*np.log10(x) + d), linestyle=':', color='tab:blue')

proportional = pickle.load( open("TN_SCA.p", "rb" ) )
proportional_qubits = np.array(proportional['N'])
proportional_times = np.mean(np.array(proportional['t']), axis=0)
a, b, c = np.polyfit(np.log10(proportional_qubits), np.log10(proportional_times), 2)
axes[0][1].scatter(proportional_qubits, proportional_times, color='tab:red', marker='o')
axes[0][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

TN_std = np.std(np.array(proportional['t']), axis=0)
a, b, c = np.polyfit(np.log10(proportional_qubits), np.log10(TN_std), 2)
axes[1][1].scatter(proportional_qubits, TN_std, color='tab:red', marker='o')
axes[1][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')


# # # # # Full

ZX = pickle.load( open("ZX_Full.p", "rb" ) )
ZX_qubits = np.array(ZX['N'])
ZX_times = np.mean(np.array(ZX['t']), axis=0)
a, b, c = np.polyfit(np.log10(ZX_qubits), np.log10(ZX_times), 2)
p1_ZX = axes[0][2].scatter(ZX_qubits, ZX_times, marker='X', color='tab:green', label='ZX')
axes[0][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle='--', color='tab:green')

ZX_std = np.std(np.array(ZX['t']), axis=0)
a, b, c = np.polyfit(np.log10(ZX_qubits), np.log10(ZX_std), 2)
axes[1][2].scatter(ZX_qubits, ZX_std, color='tab:green', linestyle='--', marker='X')
axes[1][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle='--', color='tab:green')

DD = pickle.load( open("DD_Full.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p1_DD = axes[0][2].scatter(DD_qubits, DD_times, marker='^', color='tab:blue', label='DD')
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='tab:blue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c, d = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 3)
axes[1][2].scatter(DD_qubits, DD_std, color='tab:blue', marker='^')
axes[1][2].plot(x, 10**(a*np.log10(x)**3 + b*np.log10(x)**2 + c*np.log10(x) + d), linestyle=':', color='tab:blue')

proportional = pickle.load( open("TN_Full.p", "rb" ) )
proportional_qubits = np.array(proportional['N'])
proportional_times = np.mean(np.array(proportional['t']), axis=0)
a, b, c = np.polyfit(np.log10(proportional_qubits), np.log10(proportional_times), 2)
axes[0][2].scatter(proportional_qubits, proportional_times, color='tab:red', marker='o', label='MPO')
axes[0][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

TN_std = np.std(np.array(proportional['t']), axis=0)
a, b = np.polyfit(proportional_qubits, np.log10(TN_std), 1)
p1_TN = axes[1][2].scatter(proportional_qubits, TN_std, color='tab:red', marker='o')
axes[1][2].plot(x, 10**(a*x + b), color='tab:red')


fig.legend([p1_TN, p1_ZX, p1_DD], ['MPO', 'ZX', 'DD'],
               handler_map={tuple: HandlerTuple(ndivide=None)}, loc=7, title='Method')
fig.tight_layout()
fig.subplots_adjust(right=0.86)
plt.savefig("results.pdf", format="pdf")


# plt.xlabel('Qubits')
# plt.ylabel('Runtime (s)')
# plt.legend()
plt.show()