import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


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
# axes[0].set_title('Gate Removal')
# axes[1].set_title('Phase Error')
# axes[2].set_title('Full')

axes[0].set(xlabel='$|g|_{\\text{removed}}$', ylabel='Runtime (s)')
axes[1].set(xlabel='$\\theta_{\\text{error}} / \pi$')
axes[2].set(xlabel='$n_{\\text{SWAP}}$')

# axes[1, 0].set(xlabel="Errors ($N=60$)", ylabel='Runtime (s)')
# axes[1, 1].set(xlabel='Errors ($N=20$)')
# axes[1, 2].set(xlabel='Errors ($N=10$)')

for i, ax in enumerate(axes):
    if i != 0:
        ax.tick_params('y', labelleft=False)

# Error Injection
x = np.linspace(0, 16)
TN1 = pickle.load( open("TN1.p", "rb" ) )
TN1_qubits = np.array(TN1['N'])
TN1_times = np.mean(np.array(TN1['t']), axis=0)
axes[0].scatter(TN1_qubits, TN1_times, color='tab:red', marker='o', label='$10^{-1}$')
a, b = np.polyfit(TN1_qubits, TN1_times, 1)
axes[0].plot(x, a*x + b, color='tab:red')

TN3 = pickle.load( open("TN3.p", "rb" ) )
TN3_qubits = np.array(TN3['N'])
TN3_times = np.mean(np.array(TN3['t']), axis=0)
axes[0].scatter(TN3_qubits, TN3_times, color='tab:blue', marker='^', label='$10^{-3}$')
a, b = np.polyfit(TN3_qubits, TN3_times, 1)
axes[0].plot(x, a*x + b, color='tab:blue')

TN6 = pickle.load( open("TN6.p", "rb" ) )
TN6_qubits = np.array(TN6['N'])
TN6_times = np.mean(np.array(TN6['t']), axis=0)
axes[0].scatter(TN6_qubits, TN6_times, color='tab:green', marker='X', label='$10^{-6}$')
a, b, c = np.polyfit(TN6_qubits, TN6_times, 2)
axes[0].plot(x, a*x**2 + b*x + c, color='tab:green')

axes[0].set_xlim(0, 11)
axes[0].set_ylim(1e-1, 1e2)
axes[0].set_yscale('log')

# # Phase Error
x = np.logspace(-2, 1, 100)*2
negative_x = np.array([-val for val in x])

TN1 = pickle.load( open("TN1_phase.p", "rb" ) )
TN1_qubits = np.array(TN1['N'], dtype=float)
TN1_times = np.mean(np.array(TN1['t']), axis=0)
axes[1].scatter(TN1_qubits, TN1_times, color='tab:red', marker='o')
axes[1].plot(TN1_qubits[3:8], TN1_times[3:8], color='tab:red', marker='o', label='$10^{-1}$')
a, b, c = np.polyfit(TN1_qubits[0:6], TN1_times[0:6], 2)
axes[1].plot(negative_x, a*negative_x**2 + b*negative_x + c, color='tab:red')
a, b, c = np.polyfit(TN1_qubits[5::], TN1_times[5::], 2)
axes[1].plot(x, a*x**2 + b*x + c, color='tab:red')

x = np.logspace(-3, 1, 100)*2.5
negative_x = np.array([-val for val in x])
TN3 = pickle.load( open("TN3_phase.p", "rb" ) )
TN3_qubits = np.array(TN3['N'])
TN3_times = np.mean(np.array(TN3['t']), axis=0)
axes[1].scatter(TN3_qubits, TN3_times, color='tab:blue', marker='^')
axes[1].plot(TN3_qubits[4:7], TN3_times[4:7], color='tab:blue', marker='^', label='$10^{-1}$')
a, b, c = np.polyfit(TN3_qubits[0:6], TN3_times[0:6], 2)
axes[1].plot(negative_x, a*negative_x**2 + b*negative_x + c, color='tab:blue')
a, b, c = np.polyfit(TN3_qubits[5::], TN3_times[5::], 2)
axes[1].plot(x, a*x**2 + b*x + c, color='tab:blue')

TN6 = pickle.load( open("TN6_phase.p", "rb" ) )
TN6_qubits = np.array(TN6['N'])
TN6_times = np.mean(np.array(TN6['t']), axis=0)
axes[1].plot(TN6_qubits, TN6_times, color='tab:green', marker='X', label='$10^{-6}$')

axes[1].set_xlim(-2*np.pi, 2*np.pi)
axes[1].set_ylim(1e-1, 1e2)
axes[1].set_xscale('symlog', linthresh=1e-3)
axes[1].set_yscale('log')
axes[1].set_xticks([-1e0, -1e-2, 0, 1e-2, 1e0], labels=['-1', '-0.01', '0', '0.01', '1'])
axes[1].set_xticks([-np.pi, -np.pi/100, 0, np.pi/100, np.pi], labels=['-1', '-0.01', '0', '0.01', '1'])


# # Permutation Error
x = np.linspace(-5, 100)
TN1 = pickle.load( open("TN1_permutation_test.p", "rb" ))
TN1_qubits = np.array(TN1['N'], dtype=float)
TN1_times = np.mean(np.array(TN1['t']), axis=0)
axes[2].scatter(TN1_qubits, TN1_times, color='tab:red', marker='o', label='$10^{-1}$')
a, b, c  = np.polyfit(TN1_qubits, TN1_times, 2)
axes[2].plot(x, a*x**2 + b*x + c, color='tab:red')

TN3 = pickle.load( open("TN3_permutation_test.p", "rb" ))
TN3_qubits = np.array(TN3['N'], dtype=float)
TN3_times = np.mean(np.array(TN3['t']), axis=0)
axes[2].scatter(TN3_qubits, TN3_times, color='tab:blue', marker='^', label='$10^{-3}$')
a, b, c = np.polyfit(TN3_qubits, TN3_times, 2)
axes[2].plot(x, a*x**2 + b*x + c, color='tab:blue')

TN6 = pickle.load( open("TN6_permutation_test.p", "rb" ))
TN6_qubits = np.array(TN6['N'], dtype=float)
TN6_times = np.mean(np.array(TN6['t']), axis=0)
axes[2].scatter(TN6_qubits, TN6_times, color='tab:green', marker='X', label='$10^{-6}$')
a, b, c = np.polyfit(TN6_qubits, TN6_times, 2)
axes[2].plot(x, a*x**2 + b*x + c, color='tab:green')

axes[2].set_xlim(-1, 55)
axes[2].set_ylim(1e-1, 1e2)
axes[2].set_yscale('log')
axes[2].set_xticks([0, 10, 20, 30, 40, 50])

axes[0].legend()
# fig.legend(loc=7)
fig.tight_layout()
# fig.subplots_adjust(right=0.86)
plt.savefig("results.pdf", format="pdf")


# plt.xlabel('Qubits')
# plt.ylabel('Runtime (s)')
# plt.legend()
plt.show()