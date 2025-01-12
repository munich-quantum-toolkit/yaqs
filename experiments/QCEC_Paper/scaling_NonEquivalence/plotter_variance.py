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

### Overall Plot
x = np.linspace(0, 100)
y = x
# axes[0][0].set_title('Gate Error')
# axes[0][1].set_title('Rotation Angle Error')
# axes[0][2].set_title('Permutation Error')

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


# Gate Removal
x = np.linspace(2, 100)

# DD = pickle.load( open("DD0.p", "rb" ) )
# DD_qubits = np.array(DD['N'])
# DD_times = np.mean(np.array(DD['t']), axis=0)
# a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
# axes[0][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='black', linewidth=2)

# DD_std = np.std(np.array(DD['t']), axis=0)
# a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
# axes[1][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='black', linewidth=2)

TN1 = pickle.load( open("TN1.p", "rb" ) )
TN1_qubits = np.array(TN1['N'])
TN1_times = np.mean(np.array(TN1['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN1_times), 2)
p1_TN = axes[0][0].scatter(TN1_qubits, TN1_times, marker='o', color='black')
axes[0][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black')

TN_std = np.std(np.array(TN1['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN_std), 2)
axes[1][0].scatter(TN1_qubits, TN_std, color='black', marker='o')
axes[1][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black')

TN5 = pickle.load( open("TN5.p", "rb" ) )
TN5_qubits = np.array(TN5['N'])
TN5_times = np.mean(np.array(TN5['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN5_qubits), np.log10(TN5_times), 2)
axes[0][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')
p2_TN = axes[0][0].scatter(TN5_qubits, TN5_times, marker='o', color='tab:red')

TN_std = np.std(np.array(TN5['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN5_qubits), np.log10(TN_std), 2)
axes[1][0].scatter(TN5_qubits, TN_std, color='tab:red', marker='o')
axes[1][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

TN10 = pickle.load( open("TN10.p", "rb" ) )
TN10_qubits = np.array(TN10['N'])
TN10_times = np.mean(np.array(TN10['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN10_qubits), np.log10(TN10_times), 2)
p3_TN = axes[0][0].scatter(TN10_qubits, TN10_times, marker='o', color='lightsalmon')
axes[0][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='lightsalmon')

TN_std = np.std(np.array(TN10['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN10_qubits), np.log10(TN_std), 2)
axes[1][0].scatter(TN10_qubits, TN_std, color='lightsalmon', marker='o')
axes[1][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='lightsalmon')

DD = pickle.load( open("DD1.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p1_DD = axes[0][0].scatter(DD_qubits, DD_times, marker='^', color='darkblue', )
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='darkblue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
axes[1][0].scatter(DD_qubits, DD_std, color='darkblue', marker='^')
axes[1][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='darkblue')

DD = pickle.load( open("DD5.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p2_DD = axes[0][0].scatter(DD_qubits, DD_times, marker='^', color='tab:blue', )
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='tab:blue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
axes[1][0].scatter(DD_qubits, DD_std, color='tab:blue', marker='^')
axes[1][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='tab:blue')

DD = pickle.load( open("DD10.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p3_DD = axes[0][0].scatter(DD_qubits, DD_times, marker='^', color='lightsteelblue', )
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='lightsteelblue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
axes[1][0].scatter(DD_qubits, DD_std, color='lightsteelblue', marker='^')
axes[1][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='lightsteelblue')

# TN0 = pickle.load( open("TN0.p", "rb" ) )
# TN0_qubits = np.array(TN0['N'])
# TN0_times = np.mean(np.array(TN0['t']), axis=0)
# a, b, c = np.polyfit(np.log10(TN0_qubits), np.log10(TN0_times), 2)
# axes[0][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black', linewidth=2)

# TN_std = np.std(np.array(TN0['t']), axis=0)
# a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN_std), 2)
# axes[1][0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black', linewidth=2)

axes[0][1].set_yscale('log')
axes[1][1].set_yscale('log')
# axes[0][0].grid(visible=True, axis='y')
axes[0][0].legend([(p1_TN, p1_DD), (p2_TN, p2_DD), (p3_TN, p3_DD)], ['1', '5', '10'], title='$|g|_\\text{removed}$',
               handler_map={tuple: HandlerTuple(ndivide=None)},
                ncol=3, fontsize='small', title_fontsize='small', columnspacing=1, handletextpad=0.5,
                loc='upper center', bbox_to_anchor=(0.5, 1.7))


# Angle Error
# DD = pickle.load( open("DD0.p", "rb" ) )
# DD_qubits = np.array(DD['N'])
# DD_times = np.mean(np.array(DD['t']), axis=0)
# a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
# axes[0][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='black', linewidth=2)

# DD_std = np.std(np.array(DD['t']), axis=0)
# a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
# axes[1][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='black', linewidth=2)

TN1 = pickle.load( open("TN1e-3.p", "rb" ) )
TN1_qubits = np.array(TN1['N'])
TN1_times = np.mean(np.array(TN1['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN1_times), 2)
p1_TN = axes[0][1].scatter(TN1_qubits, TN1_times, marker='o', color='black')
axes[0][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black')

TN_std = np.std(np.array(TN1['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN_std), 2)
axes[1][1].scatter(TN1_qubits, TN_std, color='black', marker='o')
axes[1][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black')

TN5 = pickle.load( open("TN1e-2.p", "rb" ) )
TN5_qubits = np.array(TN5['N'])
TN5_times = np.mean(np.array(TN5['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN5_qubits), np.log10(TN5_times), 2)
axes[0][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')
p2_TN = axes[0][1].scatter(TN5_qubits, TN5_times, marker='o', color='tab:red')

TN_std = np.std(np.array(TN5['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN5_qubits), np.log10(TN_std), 2)
axes[1][1].scatter(TN5_qubits, TN_std, color='tab:red', marker='o')
axes[1][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

TN10 = pickle.load( open("TN1e-1.p", "rb" ) )
TN10_qubits = np.array(TN10['N'])
TN10_times = np.mean(np.array(TN10['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN10_qubits), np.log10(TN10_times), 2)
p3_TN = axes[0][1].scatter(TN10_qubits, TN10_times, marker='o', color='lightsalmon')
axes[0][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='lightsalmon')

TN_std = np.std(np.array(TN10['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN10_qubits), np.log10(TN_std), 2)
axes[1][1].scatter(TN10_qubits, TN_std, color='lightsalmon', marker='o')
axes[1][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='lightsalmon')

DD = pickle.load( open("DD1e-3.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p1_DD = axes[0][1].scatter(DD_qubits, DD_times, marker='^', color='darkblue', )
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='darkblue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
axes[1][1].scatter(DD_qubits, DD_std, color='darkblue', marker='^')
axes[1][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='darkblue')

DD = pickle.load( open("DD1e-2.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p2_DD = axes[0][1].scatter(DD_qubits, DD_times, marker='^', color='tab:blue', )
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='tab:blue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
axes[1][1].scatter(DD_qubits, DD_std, color='tab:blue', marker='^')
axes[1][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='tab:blue')

DD = pickle.load( open("DD1e-1.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p3_DD = axes[0][1].scatter(DD_qubits, DD_times, marker='^', color='lightsteelblue', )
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='lightsteelblue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
axes[1][1].scatter(DD_qubits, DD_std, color='lightsteelblue', marker='^')
axes[1][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='lightsteelblue')

# TN0 = pickle.load( open("TN0.p", "rb" ) )
# TN0_qubits = np.array(TN0['N'])
# TN0_times = np.mean(np.array(TN0['t']), axis=0)
# a, b, c = np.polyfit(np.log10(TN0_qubits), np.log10(TN0_times), 2)
# axes[0][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black', linewidth=2)

# TN_std = np.std(np.array(TN0['t']), axis=0)
# a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN_std), 2)
# axes[1][1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black', linewidth=2)

axes[0][1].set_yscale('log')
axes[1][1].set_yscale('log')
axes[0][1].legend([(p1_TN, p1_DD), (p2_TN, p2_DD), (p3_TN, p3_DD)], ['$10^{-3}$', '$10^{-2}$', '$10^{-1}$'], title='$\\theta_\\text{error} / \\pi$',
               handler_map={tuple: HandlerTuple(ndivide=None)},
                ncol=3, fontsize='small', title_fontsize='small', columnspacing=1, handletextpad=0.5,
                loc='upper center', bbox_to_anchor=(0.5, 1.7))

# Permutation
# DD = pickle.load( open("DD0.p", "rb" ) )
# DD_qubits = np.array(DD['N'])
# DD_times = np.mean(np.array(DD['t']), axis=0)
# a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
# axes[0][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='black', linewidth=2)

# DD_std = np.std(np.array(DD['t']), axis=0)
# a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
# axes[1][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='black', linewidth=2)

TN1 = pickle.load( open("TN1_permutation.p", "rb" ) )
TN1_qubits = np.array(TN1['N'])
TN1_times = np.mean(np.array(TN1['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN1_times), 2)
p1_TN = axes[0][2].scatter(TN1_qubits, TN1_times, marker='o', color='black')
axes[0][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black')

TN_std = np.std(np.array(TN1['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN_std), 2)
axes[1][2].scatter(TN1_qubits, TN_std, color='black', marker='o')
axes[1][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black')

TN5 = pickle.load( open("TN5_permutation.p", "rb" ) )
TN5_qubits = np.array(TN5['N'])
TN5_times = np.mean(np.array(TN5['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN5_qubits), np.log10(TN5_times), 2)
axes[0][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')
p2_TN = axes[0][2].scatter(TN5_qubits, TN5_times, marker='o', color='tab:red')

TN_std = np.std(np.array(TN5['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN5_qubits), np.log10(TN_std), 2)
axes[1][2].scatter(TN5_qubits, TN_std, color='tab:red', marker='o')
axes[1][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

TN10 = pickle.load( open("TN10_permutation.p", "rb" ) )
TN10_qubits = np.array(TN10['N'])
TN10_times = np.mean(np.array(TN10['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN10_qubits), np.log10(TN10_times), 2)
p3_TN = axes[0][2].scatter(TN10_qubits, TN10_times, marker='o', color='lightsalmon')
axes[0][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='lightsalmon')

TN_std = np.std(np.array(TN10['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN10_qubits), np.log10(TN_std), 2)
axes[1][2].scatter(TN10_qubits, TN_std, color='lightsalmon', marker='o')
axes[1][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='lightsalmon')

DD = pickle.load( open("DD1_permutation.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p1_DD = axes[0][2].scatter(DD_qubits, DD_times, marker='^', color='darkblue', )
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='darkblue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
axes[1][2].scatter(DD_qubits, DD_std, color='darkblue', marker='^')
axes[1][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='darkblue')

DD = pickle.load( open("DD5_permutation.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p2_DD = axes[0][2].scatter(DD_qubits, DD_times, marker='^', color='tab:blue', )
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='tab:blue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
axes[1][2].scatter(DD_qubits, DD_std, color='tab:blue', marker='^')
axes[1][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='tab:blue')

DD = pickle.load( open("DD10_permutation.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p3_DD = axes[0][2].scatter(DD_qubits, DD_times, marker='^', color='lightsteelblue', )
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='lightsteelblue')

DD_std = np.std(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_std), 2)
axes[1][2].scatter(DD_qubits, DD_std, color='lightsteelblue', marker='^')
axes[1][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='lightsteelblue')

# TN0 = pickle.load( open("TN0.p", "rb" ) )
# TN0_qubits = np.array(TN0['N'])
# TN0_times = np.mean(np.array(TN0['t']), axis=0)
# a, b, c = np.polyfit(np.log10(TN0_qubits), np.log10(TN0_times), 2)
# axes[0][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black', linewidth=2)

# TN_std = np.std(np.array(TN0['t']), axis=0)
# a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN_std), 2)
# axes[1][2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black', linewidth=2)

axes[0][2].set_yscale('log')
axes[1][2].set_yscale('log')
axes[0][2].legend([(p1_TN, p1_DD), (p2_TN, p2_DD), (p3_TN, p3_DD)], ['1', '5', '10'], title='$n_\\text{SWAP}$',
               handler_map={tuple: HandlerTuple(ndivide=None)},
                ncol=3, fontsize='small', title_fontsize='small', columnspacing=1, handletextpad=0.5,
                loc='upper center', bbox_to_anchor=(0.5, 1.7))

fig.legend([(p1_TN, p2_TN, p3_TN), (p1_DD, p2_DD, p3_DD)], ['MPO', 'DD'], handler_map={tuple: HandlerTuple(ndivide=None)}, title='Method', loc=7)
fig.tight_layout()
fig.subplots_adjust(top=0.8, right=0.86)
plt.savefig("results.pdf", format="pdf")


# plt.xlabel('Qubits')
# plt.ylabel('Runtime (s)')
# plt.legend()
plt.show()