import matplotlib.pyplot as plt
import operator
import pickle


def plot_error_vs_depth() -> None:
    plt.rcParams.update({
        'figure.figsize': (7.0, 3.2),           # width ~2-column width, height for a single row of three panels
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'lines.linewidth': 2,
        'lines.markersize': 5,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4
    })

    # Data configuration
    bond_dims = [2, 4, 8, 16, 32]
    color_map = {2: "lightsalmon", 4: "salmon", 8: "red", 16: "darkred", 32: "black"}
    marker_map = {"TEBD": "^", "TDVP": "o"}

    files = [
        'heisenberg_error.pickle',
        'periodic_heisenberg_error.pickle',
        '2d_ising_error.pickle'
    ]
    model_titles = [
        'Heisenberg',
        'Periodic Heisenberg',
        '2D Ising'
    ]

    # Create figure with one row, three columns
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
    fig.subplots_adjust(wspace=0.3)

    for ax, fname, title in zip(axes, files, model_titles):
        # Load results
        with open(fname, 'rb') as f:
            data = pickle.load(f)['results']

        # Plot TDVP error for each bond dimension
        for bd in bond_dims:
            tdvp_data = [(depth, err) for (bd0, depth, inf, err) in data['TDVP'] if bd0 == bd]
            depths = [d for d, v in tdvp_data]
            vals = [v for d, v in tdvp_data]
            ax.plot(
                depths,
                vals,
                label=f"$\\chi$={bd}",
                color=color_map[bd], marker=marker_map['TDVP'],
                linestyle='-', linewidth=2
            )

        # Plot TEBD error
        tebd_data = [(depth, err) for (depth, inf, err) in data['TEBD']]
        tebd_data.sort(key=operator.itemgetter(0))
        depths_tebd = [d for d, v in tebd_data]
        vals_tebd = [v for d, v in tebd_data]
        ax.plot(
            depths_tebd,
            vals_tebd,
            label='TEBD', color='k', marker=marker_map['TEBD'],
            linestyle='--', linewidth=2
        )

        # Log scale and limits
        ax.set_yscale('log')
        ax.set_ylim(1e-16, 5e-1)
        ax.set_title(title)

    # Axis labels and legend
    axes[0].set_ylabel('Error (log scale)')
    for ax in axes:
        ax.set_xlabel('Circuit depth (Trotter steps)')

    # Show legend only on first panel
    axes[0].legend(ncol=1, loc='lower right')

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_error_vs_depth()