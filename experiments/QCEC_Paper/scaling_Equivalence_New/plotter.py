import pickle
import matplotlib.pyplot as plt
import numpy as np


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = 1*(fig_width_in * golden_ratio * (subplots[0] / subplots[1]))

    return (fig_width_in, fig_height_in)

### Overall Plot
x = np.linspace(0, 25)
y = x
fig, axes = plt.subplots(1, 3, figsize=set_size(514.1736, subplots=(2, 3)))
axes[0].set_title('Linear')
axes[1].set_title('SCA')
axes[2].set_title('Full')

axes[0].set(xlabel='N', ylabel='Runtime (s)')
axes[1].set(xlabel='N')
axes[2].set(xlabel='N')

# axes[1, 0].set(xlabel="Errors ($N=60$)", ylabel='Runtime (s)')
# axes[1, 1].set(xlabel='Errors ($N=20$)')
# axes[1, 2].set(xlabel='Errors ($N=10$)')

for i, ax in enumerate(axes):
    if i != 0:
        ax.tick_params('y', labelleft=False)

# Adjust misaligned y axis labels
# labelx = -0.3
# for j in range(2):
#     axes[j, 0].yaxis.set_label_coords(labelx, 0.5)

# ### Linear plot
# x = np.linspace(2, 100)
# proportional = pickle.load( open("proportional.p", "rb" ) )
# proportional_qubits = np.array(proportional['N'])
# proportional_times = np.mean(np.array(proportional['t']), axis=0)
# a, b, c = np.polyfit(proportional_qubits, proportional_times, 2)
# plt.scatter(proportional_qubits, proportional_times, color='tab:red', marker='o', label='TN')
# # plt.plot(x, a*x**2 + b*x + c, color='tab:red')
# plt.plot(x, a*x**2 + b*x + c, color='tab:red')

# # total = pickle.load( open("total.p", "rb" ) )
# # total_qubits = np.array(total['N'])
# # total_times = np.mean(np.array(total['t']), axis=0)
# # a, b, c = np.polyfit(total_qubits, total_times, 2)
# # plt.scatter(total_qubits, total_times, color='tab:red', marker='x', label='TN: Total Strategy')
# # plt.plot(x, a*x**2 + b*x + c, color='tab:red', linestyle='--')

# ZX = pickle.load( open("zx.p", "rb" ) )
# ZX_qubits = np.array(ZX['N'])
# ZX_times = np.mean(np.array(ZX['t']), axis=0)
# a, b, c = np.polyfit(ZX_qubits, ZX_times, 2)
# plt.scatter(ZX_qubits, ZX_times, marker='*', color='tab:green', label='ZX')
# plt.plot(x, a*x**2 + b*x + c, color='tab:green')

# DD = pickle.load( open("dd.p", "rb" ) )
# DD_qubits = np.array(DD['N'])
# DD_times = np.mean(np.array(DD['t']), axis=0)
# plt.scatter(DD_qubits, DD_times, marker='^', color='tab:blue', label='DD')
# a, b, c = np.polyfit(DD_qubits, DD_times, 2)
# plt.plot(x, a*x**2 + b*x + c, color='tab:blue')

# plt.title('Verification of Equivalent Circuits, Opt=1, Samples=10')
# # plt.plot(proportional_qubits, proportional_times, marker='o', color='tab:red', label='TN: Proportional Strategy')
# # # plt.plot(total_qubits, total_times, marker='x', color='tab:red', linestyle='--', label='TN: Total Bond Strategy')
# # plt.plot(ZX_qubits, ZX_times, marker='*', color='tab:green', label='ZX')
# # plt.plot(DD_qubits, DD_times, marker='^', color='tab:blue', label='DD: Best Strategy')

# # plt.yscale('log')
# plt.xlabel('Qubits')
# plt.ylabel('Runtime (s)')
# plt.legend()
# plt.show()


# Linear
x = np.linspace(2, 50)
proportional = pickle.load( open("TN_Lin.p", "rb" ) )
proportional_qubits = np.array(proportional['N'])
proportional_times = np.mean(np.array(proportional['t']), axis=0)
TN_std = np.std(np.array(proportional['t']), axis=0)
a, b, c = np.polyfit(np.log10(proportional_qubits), np.log10(proportional_times), 2)
axes[0].scatter(proportional_qubits, proportional_times, color='tab:red', marker='o')
axes[0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

ZX = pickle.load( open("ZX_Lin.p", "rb" ) )
ZX_qubits = np.array(ZX['N'])
ZX_times = np.mean(np.array(ZX['t']), axis=0)
a, b, c = np.polyfit(np.log10(ZX_qubits), np.log10(ZX_times), 2)
axes[0].scatter(ZX_qubits, ZX_times, marker='X', color='tab:green')
axes[0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle='--', color='tab:green')

DD = pickle.load( open("DD_Lin.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
axes[0].scatter(DD_qubits, DD_times, marker='^', color='tab:blue')
a, b, c, d = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 3)
axes[0].plot(x, 10**(a*np.log10(x)**3 + b*np.log10(x)**2 + c*np.log10(x) + d), linestyle=':', color='tab:blue')


axes[0].set_xlim(1, 32)
axes[0].set_ylim(1e-3, 1e2)
axes[0].set_yscale('log')

# SCA
proportional = pickle.load( open("TN_SCA.p", "rb" ) )
proportional_qubits = np.array(proportional['N'])
proportional_times = np.mean(np.array(proportional['t']), axis=0)
a, b, c = np.polyfit(np.log10(proportional_qubits), np.log10(proportional_times), 2)
axes[1].scatter(proportional_qubits, proportional_times, color='tab:red', marker='o')
axes[1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

ZX = pickle.load( open("ZX_SCA.p", "rb" ) )
ZX_qubits = np.array(ZX['N'])
ZX_times = np.mean(np.array(ZX['t']), axis=0)
a, b, c = np.polyfit(np.log10(ZX_qubits), np.log10(ZX_times), 2)
axes[1].scatter(ZX_qubits, ZX_times, marker='X', color='tab:green')
axes[1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle='--', color='tab:green')

DD = pickle.load( open("DD_SCA.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
axes[1].scatter(DD_qubits, DD_times, marker='^', color='tab:blue')
a, b, c, d = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 3)
axes[1].plot(x, 10**(a*np.log10(x)**3 + b*np.log10(x)**2 + c*np.log10(x) + d), linestyle=':', color='tab:blue')


axes[1].set_xlim(1, 32)
axes[1].set_ylim(1e-3, 1e2)
axes[1].set_yscale('log')

# # # # # Full
proportional = pickle.load( open("TN_Full.p", "rb" ) )
proportional_qubits = np.array(proportional['N'])
proportional_times = np.mean(np.array(proportional['t']), axis=0)
a, b, c = np.polyfit(np.log10(proportional_qubits), np.log10(proportional_times), 2)
axes[2].scatter(proportional_qubits, proportional_times, color='tab:red', marker='o', label='MPO')
axes[2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

ZX = pickle.load( open("ZX_Full.p", "rb" ) )
ZX_qubits = np.array(ZX['N'])
ZX_times = np.mean(np.array(ZX['t']), axis=0)
a, b, c = np.polyfit(np.log10(ZX_qubits), np.log10(ZX_times), 2)
axes[2].scatter(ZX_qubits, ZX_times, marker='X', color='tab:green', label='ZX')
axes[2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle='--', color='tab:green')

DD = pickle.load( open("DD_Full.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
axes[2].scatter(DD_qubits, DD_times, marker='^', color='tab:blue', label='DD')
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='tab:blue')

axes[2].set_xlim(1, 32)
axes[2].set_ylim(1e-3, 1e2)
axes[2].set_yscale('log')
axes[2].legend(loc='lower right')
# fig.legend(loc=7)
fig.tight_layout()
# fig.subplots_adjust(right=0.86)
plt.savefig("results.pdf", format="pdf")


# plt.xlabel('Qubits')
# plt.ylabel('Runtime (s)')
# plt.legend()
plt.show()