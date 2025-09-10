import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

def compute_mean_errors(pickle_path, exp_value_exact, x_values, num_samples=1000):
    """
    Load trajectories from pickle_path, then for each sample size in x_values:
    - draw num_samples random subsets
    - compute absolute error |mean(sample) - exp_value_exact|
    - return arrays of mean errors and standard deviations
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    trajectories = data[0][0].trajectories.squeeze()
    trajectories = trajectories[:, -5]

    errors = []
    std_devs = []
    for sample_size in x_values:
        sample_errors = []
        for _ in range(num_samples):
            sample = np.random.choice(trajectories, sample_size, replace=False)
            error = abs(np.mean(sample) - exp_value_exact)
            sample_errors.append(error)
        errors.append(np.mean(sample_errors))
        std_devs.append(np.std(sample_errors))

    return np.array(errors), np.array(std_devs)


def plot_convergence_data():
    # ——— Global plotting parameters ———
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}",
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2,
        "legend.fontsize": 10,
        })

    fig = plt.figure(figsize=(3.5, 3), constrained_layout=True)
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, 1e4)
    ax.set_ylim(1e-3, 1e0)
    ax.set_xlabel(r"Trajectories ($N$)")
    ax.set_ylabel(r"$\bigl|\langle Z \rangle\bigr|$")
    ax.grid(which='both', linestyle='--', color='0.8', linewidth=0.8, alpha=0.7)


    # ——— Define x_values (trajectory counts) ———
    x_values = [*range(1, 10),
                *range(10, 110, 10),
                *range(100, 1100, 100)]
                # *range(1000, 11000, 1000)]

    # ——— Configuration: (pickle file, label, linestyle, color, fill_color) ———
    configs = [
        {
            "path": "pauli_convergence_8.pickle",
            "label": r"$8$",
            "linestyle": "-",
            "color": "tab:blue",
            "fill_color": None
        },
        {
            "path": "pauli_convergence_16.pickle",
            "label": r"$16$",
            "linestyle": "-",
            "color": "tab:orange",
            "fill_color": None
        },
        {
            "path": "pauli_convergence_32.pickle",
            "label": r"$32$",
            "linestyle": "-",
            "color": "tab:green",
            "fill_color": "tab:green"
        }
    ]

    # ——— Loop over each configuration, compute and plot errors ———
    num_samples = 50
    for cfg in configs:
        if "pauli" in cfg["path"]:
            with open("pauli_convergence_32.pickle", "rb") as f:
                data = pickle.load(f)
            exp_val_exact = data[0][0].results
            exp_val_exact = exp_val_exact[-5]

        errors, std_devs = compute_mean_errors(
            pickle_path=cfg["path"],
            exp_value_exact=exp_val_exact,
            x_values=x_values,
            num_samples=num_samples
        )

        ax.plot(
            x_values,
            errors,
            label=cfg["label"],
            linestyle=cfg["linestyle"],
            color=cfg["color"]
            )

        if cfg["fill_color"] is not None:
            ax.fill_between(
                x_values,
                errors - std_devs,
                errors + std_devs,
                alpha=0.2,
                color=cfg["fill_color"]
            )

    # ——— Plot the theoretical 1/√N reference curve ———
    x_ref = np.logspace(-2, 4)
    ax.plot(
        x_ref,
        1 / np.sqrt(x_ref),
        linestyle='-',
        color='black',
        linewidth=1.5,
    )
    ax.text(
        50,                     # manually pick an x‐location
        1 / np.sqrt(50) * 1.35,  # y‐position slightly above
        r"$1/\sqrt{N}$",
        color="black",
        fontsize=11,
        rotation=-35,
        va="center",
        ha="center"
    )
    legend = ax.legend(loc='upper right', frameon=True, title=r"$L$")

    plt.savefig("convergence.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_convergence_data()