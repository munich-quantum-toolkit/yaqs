import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import os

# Ensure the script’s working directory is the location of this file:
os.chdir(os.path.dirname(__file__))

def compute_error_heatmap(pickle_path, heatmap_exact, L, num_samples, num_trajectories):
    """
    - Load trajectories from `pickle_path`.
    - For `num_samples` random draws of size `num_trajectories`, compute the 
      site‐by‐time average, then calculate |exact − stochastic|.
    - Return the mean‐over‐samples of that absolute error, shaped (L, T).
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # data['sim_params'].observables is a list of length L
    # Each observable has attributes:
    #   - .trajectories: array of shape (n_total_trajectories, T)
    #   - .results (we don’t actually use .results here, only trajectories)
    all_trajs = [obs.trajectories for obs in data['sim_params'].observables]
    n_total = all_trajs[0].shape[0]   # total number of available trajectories
    T = all_trajs[0].shape[1]         # number of time‐steps

    # Pre‐allocate an array to collect |exact − stochastic| for each sample
    # Shape: (num_samples, L, T)
    sample_errors = np.zeros((num_samples, L, T))

    for i in range(num_samples):
        # Randomly choose `num_trajectories` distinct indices
        indices = np.random.choice(n_total, size=num_trajectories, replace=False)

        # Build a “sampled” heatmap: for each site, take mean over selected trajectories
        sampled_heatmap = np.zeros((L, T))
        for s in range(L):
            sampled_heatmap[s, :] = all_trajs[s][indices, :].mean(axis=0)

        # Absolute error w.r.t. exact
        sample_errors[i, :, :] = np.abs(heatmap_exact - sampled_heatmap)

    # Return the sample‐mean of absolute errors at each (site, time)
    return sample_errors.mean(axis=0)


def plot_bonddimension_data():
    # ——— Global style settings ———
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}",
        "font.size": 12,           # base font size
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,        # embed TrueType fonts in PDF
    })

    # ——— Figure / Axes layout ———
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(7.2, 6.0),          # a bit taller to avoid cramped y‐axis labels
        constrained_layout=False
    )

    # Adjust margins manually so labels + colorbar fit nicely
    fig.subplots_adjust(left=0.10, right=0.88, top=0.92, bottom=0.10,
                        wspace=0.35, hspace=0.35)

    # Titles at the top of each column: χ = 4, 8, 16
    chi_values = [4, 8, 16]
    for col, chi in enumerate(chi_values):
        axes[0, col].set_title(r"$\chi={}$".format(chi), fontsize=14)

    # Y‐axis label on the leftmost column
    for row in range(3):
        axes[row, 0].set_ylabel("Bond", fontsize=14)

    # X‐axis label on the bottom row
    for col in range(3):
        axes[2, col].set_xlabel(r"Time ($Jt$)", fontsize=14)

    # We will display ticks only on the “outer” edges:
    #  - No x‐ticks on rows 0 & 1
    #  - No y‐ticks on columns 1 & 2
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            if row < 2:
                ax.set_xticks([])
            if col > 0:
                ax.set_yticks([])

    # ——— Load “exact” data once (L × T) ———
    with open("QuTip_exact_convergence.pickle", "rb") as f_exact:
        data_exact = pickle.load(f_exact)

    # We assume observables[site] is a 1‐D array of length T
    # For L = 9 bonds, build heatmap_exact of shape (L, T)
    bonds = 9
    heatmap_exact = np.stack([ data_exact['observables'][site] for site in range(bonds) ], axis=0)

    # ——— Normalization + colormap for all heatmaps ———
    norm = LogNorm(vmin=1e-3, vmax=1e-1)
    cmap = plt.get_cmap("coolwarm")  # Perceptually uniform sequential

    # ——— Data parameters ———
    num_samples = 1000              # number of random sub‐samples to average over
    traj_counts = [100, 1000, 10000]  # rows correspond to these
    # Ensure these are in the same order as row indices (0,1,2)

    # ——— Loop over χ columns and N‐rows ———
    for col, chi in enumerate(chi_values):
        # Each χ has its own pickle file
        pickle_path = f"TJM_convergence_Bond{chi}.pickle"
        # Load just the “time axis” from one of the observables:
        with open(pickle_path, "rb") as f_chi:
            data_chi = pickle.load(f_chi)
        elapsed_time = data_chi['sim_params'].elapsed_time  # scalar, total time T_max
        T = data_chi['sim_params'].observables[0].trajectories.shape[1]

        # Build an “extent” that maps column index → time in [0, T_max]
        extent = [0, elapsed_time, bonds, 0]  # y goes from 0 (top) down to L

        for row, N in enumerate(traj_counts):
            ax = axes[row, col]

            # Compute the average error‐heatmap for (χ, N)
            error_hmap = compute_error_heatmap(
                pickle_path=pickle_path,
                heatmap_exact=heatmap_exact,
                L=bonds,
                num_samples=num_samples,
                num_trajectories=N
            )

            # Display with consistent normalization & colormap
            im = ax.imshow(
                error_hmap,
                cmap=cmap,
                norm=norm,
                aspect="auto",
                extent=extent
            )

            # Set y‐ticks every 2 bonds, centered on “rows”:
            yticks_sites = list(range(2, bonds + 1, 2))
            ax.set_yticks([y - 0.5 for y in yticks_sites], yticks_sites)

            # (Optional) On the bottom row, add x‐ticks at “nice” time‐intervals
            if row == 2:
                # Example: place ticks at 0, 0.25 T_max, 0.5 T_max, 0.75 T_max, T_max
                tick_positions = np.linspace(0, elapsed_time, num=5)
                tick_labels = [f"{pos:.1f}" for pos in tick_positions]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=0)

            # Annotate the first column of each row with N=… text, rotated 90°
            if col == 0:
                ax.text(
                    -0.38, 0.5,
                    rf"$N={N}$",
                    transform=ax.transAxes,
                    fontsize=14,
                    va="center",
                    ha="center",
                    rotation=90
                )

    # ——— Colorbar on the right, spanning all subplots ———
    cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.80])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical"
    )
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_title(r"$\epsilon$", fontsize=14, pad=10)

    # ——— Save & Show ———
    fig.savefig("Benchmark_BondDimension.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_bonddimension_data()
