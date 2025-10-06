import numpy as np
from typing import Dict
import matplotlib.pyplot as plt

def pick_subplot_indices(L: int) -> list[int]:
    if L < 4:
        return sorted(set([0, L - 1] + [max(0, L // 2 - 1), min(L - 1, L // 2)]))
    return [0, L // 2 - 1, L // 2, L - 1]

def plot_series_against_exact(
    exact: np.ndarray,
    series_by_label: dict[str, np.ndarray],
    *,
    num_qubits: int,
    num_layers: int,
) -> None:
    """Plot YAQS series against exact baseline for a few representative qubits.

    Uses the same output structure as run_yaqs/run_qiskit: arrays of shape
    (num_qubits, num_layers) that omit the initial t=0 point. The plot re-adds
    the initial value 1.0 for visualization.
    """
    import matplotlib.pyplot as plt

    indices = pick_subplot_indices(num_qubits)
    layers = np.arange(0, num_layers + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    axs = axs.ravel()
    for ax, q in zip(axs, indices):
        for label, arr in series_by_label.items():
            y = np.concatenate(([1.0], arr[q]))
            ax.plot(layers, y, label=label)
        y_exact = np.concatenate(([1.0], exact[q]))
        ax.plot(layers, y_exact, label="exact", linestyle="--", color="black")
        ax.set_title(f"qubit {q}")
        ax.grid(True, linestyle="--", alpha=0.6)
    fig.supxlabel("Layer")
    fig.supylabel("<Z>")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    plt.show()

def plot_avg_bond_dims(
    *,
    num_layers: int,
    qiskit_bonds: Dict[str, np.ndarray],
    yaqs_bonds_by_label: Dict[str, np.ndarray | None],
) -> None:
    import matplotlib.pyplot as plt

    layers = np.arange(1, num_layers + 1)
    plt.figure(figsize=(10, 3.5))

    # Qiskit MPS: mean per layer provided in bonds dict
    if "per_layer_mean_across_shots" in qiskit_bonds:
        q_mean = np.asarray(qiskit_bonds["per_layer_mean_across_shots"])  # (num_layers,)
        plt.plot(layers, q_mean[: num_layers], label="qiskit_mps", linewidth=2)

    # YAQS: mean across trajectories; drop initial and final columns
    for label, arr in yaqs_bonds_by_label.items():
        if arr is None:
            continue
        mean_per_col = np.mean(arr, axis=0)
        if mean_per_col.size >= 2:
            mean_layers = mean_per_col[1:-1]
        else:
            mean_layers = mean_per_col
        plt.plot(layers, mean_layers[: num_layers], label=label)

    plt.xlabel("Layer")
    plt.ylabel("avg max bond dim")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", ncol=3)
    plt.tight_layout()
    plt.show()


def plot_stochastic_variances(
    *,
    num_layers: int,
    qiskit_var: np.ndarray,  # (num_qubits, num_layers)
    yaqs_var_by_label: Dict[str, np.ndarray],  # each (num_qubits, num_layers)
) -> None:
    import matplotlib.pyplot as plt

    # include initial layer at 0
    layers = np.arange(0, num_layers + 1)
    plt.figure(figsize=(10, 3.5))

    q_mean = np.mean(qiskit_var, axis=0)
    plt.plot(layers[: q_mean.shape[0]], q_mean, label="qiskit_mps", linewidth=2)

    for label, arr in yaqs_var_by_label.items():
        mean_var = np.mean(arr, axis=0)
        plt.plot(layers[: mean_var.shape[0]], mean_var, label=label)

    plt.xlabel("Layer")
    plt.ylabel("trajectory variance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", ncol=3)
    plt.tight_layout()
    plt.show()

def plot_z_expectations_all_qubits(z: np.ndarray) -> None:
    """
    Plot ⟨Z_i⟩ for all qubits across layers.
    Expects z.shape == (num_qubits, num_layers).
    """
    import matplotlib.pyplot as plt

    num_qubits, T = z.shape
    # Layers index: 0..T-1 where 0 corresponds to initial state if included
    layers = np.arange(T)
    plt.figure(figsize=(10, 6))
    for q in range(num_qubits):
        plt.plot(layers, z[q], label=f"q{q}", alpha=0.8)
    plt.xlabel("Layer")
    plt.ylabel("<Z>")
    plt.title("Z expectation values per qubit")
    # Hide legend if too many qubits
    if num_qubits <= 12:
        plt.legend(ncol=3, fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

