import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple


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
axes[0].set_title('Gate Error')
axes[1].set_title('Angle Error')
axes[2].set_title('Permutation Error')

axes[0].set(xlabel='N', ylabel='Runtime (s)')
axes[1].set(xlabel='N')
axes[2].set(xlabel='N')

# axes[1, 0].set(xlabel="Errors ($N=60$)", ylabel='Runtime (s)')
# axes[1, 1].set(xlabel='Errors ($N=20$)')
# axes[1, 2].set(xlabel='Errors ($N=10$)')

for i, ax in enumerate(axes):
    if i != 0:
        ax.tick_params('y', labelleft=False)

# Linear
x = np.linspace(2, 50)

DD = pickle.load( open("DD0.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='black', linewidth=2)

TN1 = pickle.load( open("TN1.p", "rb" ) )
TN1_qubits = np.array(TN1['N'])
TN1_times = np.mean(np.array(TN1['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN1_times), 2)
p1_TN = axes[0].scatter(TN1_qubits, TN1_times, marker='o', color='darkred')
axes[0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='darkred')

TN5 = pickle.load( open("TN5.p", "rb" ) )
TN5_qubits = np.array(TN5['N'])
TN5_times = np.mean(np.array(TN5['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN5_qubits), np.log10(TN5_times), 2)
axes[0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')
p2_TN = axes[0].scatter(TN5_qubits, TN5_times, marker='o', color='tab:red')

TN10 = pickle.load( open("TN10.p", "rb" ) )
TN10_qubits = np.array(TN10['N'])
TN10_times = np.mean(np.array(TN10['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN10_qubits), np.log10(TN10_times), 2)
p3_TN = axes[0].scatter(TN10_qubits, TN10_times, marker='o', color='salmon')
axes[0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='salmon')

DD = pickle.load( open("DD1.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p1_DD = axes[0].scatter(DD_qubits, DD_times, marker='^', color='darkblue', label='DD')
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='darkblue')

DD = pickle.load( open("DD5.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p2_DD = axes[0].scatter(DD_qubits, DD_times, marker='^', color='tab:blue', label='DD')
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='tab:blue')

DD = pickle.load( open("DD10.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p3_DD = axes[0].scatter(DD_qubits, DD_times, marker='^', color='lightsteelblue', label='DD')
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='lightsteelblue')

TN0 = pickle.load( open("TN0.p", "rb" ) )
TN0_qubits = np.array(TN0['N'])
TN0_times = np.mean(np.array(TN0['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN0_qubits), np.log10(TN0_times), 2)
axes[0].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black', linewidth=2)


axes[0].set_xlim(2, 32)
axes[0].set_ylim(1e-3, 1e2)
axes[0].set_yscale('log')
# axes[0].grid(visible=True, axis='y')
axes[0].legend([(p1_TN, p1_DD), (p2_TN, p2_DD), (p3_TN, p3_DD)], ['1', '5', '10'], title='$|g|_\\text{removed}$',
               handler_map={tuple: HandlerTuple(ndivide=None)})

# Angle Error
DD = pickle.load( open("DD0.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='black', linewidth=2)

TN1 = pickle.load( open("TN1e-3.p", "rb" ) )
TN1_qubits = np.array(TN1['N'])
TN1_times = np.mean(np.array(TN1['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN1_times), 2)
p1_TN = axes[1].scatter(TN1_qubits, TN1_times, marker='o', color='darkred')
axes[1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='darkred')

TN10 = pickle.load( open("TN1e-2.p", "rb" ) )
TN10_qubits = np.array(TN10['N'])
TN10_times = np.mean(np.array(TN10['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN10_qubits), np.log10(TN10_times), 2)
p2_TN = axes[1].scatter(TN10_qubits, TN10_times, marker='o', color='tab:red')
axes[1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

TN5 = pickle.load( open("TN1e-1.p", "rb" ) )
TN5_qubits = np.array(TN5['N'])
TN5_times = np.mean(np.array(TN5['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN5_qubits), np.log10(TN5_times), 2)
p3_TN = axes[1].scatter(TN5_qubits, TN5_times, marker='o', color='salmon')
axes[1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='salmon')

DD = pickle.load( open("DD1e-3.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p1_DD = axes[1].scatter(DD_qubits, DD_times, marker='^', color='darkblue', label='DD')
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='darkblue')

DD = pickle.load( open("DD1e-1.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p2_DD = axes[1].scatter(DD_qubits, DD_times, marker='^', color='tab:blue', label='DD')
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='tab:blue')

DD = pickle.load( open("DD1e-2.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p3_DD = axes[1].scatter(DD_qubits, DD_times, marker='^', color='lightsteelblue', label='DD')
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='lightsteelblue')

TN0 = pickle.load( open("TN0.p", "rb" ) )
TN0_qubits = np.array(TN0['N'])
TN0_times = np.mean(np.array(TN0['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN0_qubits), np.log10(TN0_times), 2)
axes[1].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black', linewidth=2)


axes[1].set_xlim(2, 32)
axes[1].set_ylim(1e-3, 1e2)
axes[1].set_yscale('log')
axes[1].legend([(p1_TN, p1_DD), (p2_TN, p2_DD), (p3_TN, p3_DD)], ['$10^{-3} \pi$', '$10^{-2} \pi$', '$10^{-1} \pi$'], title='$\\theta_\\text{error}$',
               handler_map={tuple: HandlerTuple(ndivide=None)})

# Permutation
DD = pickle.load( open("DD0.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='black', linewidth=2)

TN1 = pickle.load( open("TN1_permutation.p", "rb" ) )
TN1_qubits = np.array(TN1['N'])
TN1_times = np.mean(np.array(TN1['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN1_qubits), np.log10(TN1_times), 2)
p1_TN = axes[2].scatter(TN1_qubits, TN1_times, marker='o', color='darkred')
axes[2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='darkred')

TN5 = pickle.load( open("TN5_permutation.p", "rb" ) )
TN5_qubits = np.array(TN5['N'])
TN5_times = np.mean(np.array(TN5['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN5_qubits), np.log10(TN5_times), 2)
p2_TN = axes[2].scatter(TN5_qubits, TN5_times, marker='o', color='tab:red')
axes[2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='tab:red')

TN10 = pickle.load( open("TN10_permutation.p", "rb" ) )
TN10_qubits = np.array(TN10['N'])
TN10_times = np.mean(np.array(TN10['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN10_qubits), np.log10(TN10_times), 2)
p3_TN = axes[2].scatter(TN10_qubits, TN10_times, marker='o', color='salmon')
axes[2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='salmon')

DD = pickle.load( open("DD1_permutation.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p1_DD = axes[2].scatter(DD_qubits, DD_times, marker='^', color='darkblue', label='DD')
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='darkblue')

DD = pickle.load( open("DD5_permutation.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p2_DD = axes[2].scatter(DD_qubits, DD_times, marker='^', color='tab:blue', label='DD')
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='tab:blue')

DD = pickle.load( open("DD10_permutation.p", "rb" ) )
DD_qubits = np.array(DD['N'])
DD_times = np.mean(np.array(DD['t']), axis=0)
p3_DD = axes[2].scatter(DD_qubits, DD_times, marker='^', color='lightsteelblue', label='DD')
a, b, c = np.polyfit(np.log10(DD_qubits), np.log10(DD_times), 2)
axes[2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), linestyle=':', color='lightsteelblue')

TN0 = pickle.load( open("TN0.p", "rb" ) )
TN0_qubits = np.array(TN0['N'])
TN0_times = np.mean(np.array(TN0['t']), axis=0)
a, b, c = np.polyfit(np.log10(TN0_qubits), np.log10(TN0_times), 2)
axes[2].plot(x, 10**(a*np.log10(x)**2 + b*np.log10(x) + c), color='black', linewidth=2)

axes[2].set_xlim(2, 32)
axes[2].set_ylim(1e-3, 1e2)
axes[2].set_yscale('log')
axes[2].legend([(p1_TN, p1_DD), (p2_TN, p2_DD), (p3_TN, p3_DD)], ['1', '5', '10'], title='$n_\\text{SWAP}$',
               handler_map={tuple: HandlerTuple(ndivide=None)})
# axes[2].legend()
fig.legend([p2_TN, p2_DD], ['MPO', 'DD'], handler_map={tuple: HandlerTuple(ndivide=None)}, loc=7)
fig.tight_layout()
fig.subplots_adjust(right=0.86)
plt.savefig("results.pdf", format="pdf")


# plt.xlabel('Qubits')
# plt.ylabel('Runtime (s)')
# plt.legend()
plt.show()