import copy

import numpy as np
from qiskit.quantum_info import Pauli
from qiskit_aer.noise.errors import PauliLindbladError
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit import QuantumCircuit

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit, create_heisenberg_circuit
from mqt.yaqs.core.libraries.gate_library import Z

def run_yaqs(
    basis,
    num_qubits: int,
    num_layers: int,
    nm: NoiseModel,
    *,
    parallel = True,
    num_traj = 1024,
    max_bond_dim = 256,
) -> np.ndarray:
    circ = basis.copy()
    # sample after the first layer
    circ.barrier(label="SAMPLE_OBSERVABLES")
    for _ in range(1, num_layers):
        circ.compose(basis, qubits=range(num_qubits), inplace=True)
        circ.barrier(label="SAMPLE_OBSERVABLES")

    obs = [Observable(Z(), i) for i in range(num_qubits)]
    sim = StrongSimParams(observables=obs, num_traj=num_traj, max_bond_dim=max_bond_dim, sample_layers=True)
    state = MPS(num_qubits, state="zeros", pad=2)
    simulator.run(state, circ, sim, nm, parallel=parallel)
    # shape (Q, num_layers+?)
    res = np.stack([np.real(o.results) for o in sim.observables])
    # drop initial column and final aggregate → keep exactly num_layers points
    return res[:, 1:-1]


def run_qiskit(num_qubits: int, num_layers: int, basis, noise_model: QiskitNoiseModel, method = "density_matrix") -> np.ndarray:
    from worker_functions.qiskit_noisy_sim import qiskit_noisy_simulator

    baseline = [[1.0] for _ in range(num_qubits)]
    for i in range(num_layers):
        qc = basis.copy()
        for _ in range(i):
            qc = qc.compose(basis)
        vals = np.real(np.asarray(qiskit_noisy_simulator(qc, noise_model, num_qubits, 1, method=method))).flatten()
        for q in range(num_qubits):
            baseline[q].append(float(vals[q]))
    arr = np.stack([np.asarray(b) for b in baseline])
    # drop initial t=0 → 1..num_layers
    return arr[:, 1:]


def compute_variance_vs_exact(yaqs: np.ndarray, exact: np.ndarray) -> float:
    return float(np.var(yaqs - exact))


def compute_mean_error_vs_exact(yaqs: np.ndarray, exact: np.ndarray) -> float:
    """Mean absolute error across all qubits and layers."""
    return float(np.mean(np.abs(yaqs - exact)))


def pick_subplot_indices(L: int) -> list[int]:
    if L < 4:
        return sorted(set([0, L - 1] + [max(0, L // 2 - 1), min(L - 1, L // 2)]))
    return [0, L // 2 - 1, L // 2, L - 1]


def print_variances_against_exact(exact: np.ndarray, series_by_label: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute and print variance of each YAQS series vs exact baseline.

    Parameters
    ----------
    exact : np.ndarray
        Array of shape (num_qubits, num_layers) from run_qiskit (taken as exact).
    series_by_label : dict[str, np.ndarray]
        Mapping from label (e.g., 'standard', 'projector', ...) to YAQS array of
        shape (num_qubits, num_layers) as returned by run_yaqs.

    Returns
    -------
    dict[str, float]
        Mapping from label to variance value (np.var(yaqs - exact)).
    """
    variances: dict[str, float] = {}
    print("=== Variance vs exact (lower is better) ===")
    for label, arr in series_by_label.items():
        var = compute_variance_vs_exact(arr, exact)
        variances[label] = var
        print(f"{label:>12}: {var:.6e}")
    return variances


def print_mean_errors_against_exact(exact: np.ndarray, series_by_label: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute and print mean absolute error of each series vs exact baseline."""
    mean_errors: dict[str, float] = {}
    print("=== Mean absolute error vs exact (lower is better) ===")
    for label, arr in series_by_label.items():
        mae = compute_mean_error_vs_exact(arr, exact)
        mean_errors[label] = mae
        print(f"{label:>12}: {mae:.6e}")
    return mean_errors


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


def build_noise_models(processes):
    # Always deep-copy; each NoiseModel gets its own process list.
    procs_std  = copy.deepcopy(processes)
    procs_proj = copy.deepcopy(processes)
    procs_2pt  = copy.deepcopy(processes)
    procs_gaus = copy.deepcopy(processes)

    # (1) standard (whatever your default is)
    noise_model_normal = NoiseModel(procs_std)

    # (2) projector unraveling: same Lindblad rate γ per process
    for p in procs_proj:
        p["unraveling"] = "projector"
    for p in procs_2pt:
        p["unraveling"] = "unitary_2pt"
    for p in procs_gaus:
        p["unraveling"] = "unitary_gauss"
        # strength unchanged
    noise_model_projector = NoiseModel(procs_proj)
    noise_model_unitary_2pt = NoiseModel(procs_2pt)
    noise_model_unitary_gauss = NoiseModel(procs_gaus)

    return (noise_model_normal,
            noise_model_projector,
            noise_model_unitary_2pt,
            noise_model_unitary_gauss)



def variance_print():
    print("=== Variance vs baseline (lower is better) ===")
    for k, (_, var) in results.items():
        print(f"{k:>12}: {var:.6e}")

def plotting():
        import matplotlib.pyplot as plt
        indices = pick_subplot_indices(args.num_qubits)
        layers = np.arange(0, args.layers + 1)
        fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
        axs = axs.ravel()
        for ax, q in zip(axs, indices):
            for label, (series, _) in results.items():
                y = np.concatenate(([1.0], series[q]))
                ax.plot(layers, y, label=label)
            y_exact = np.concatenate(([1.0], baseline[q]))
            ax.plot(layers, y_exact, label="exact", linestyle="--", color="black")
            ax.set_title(f"qubit {q}")
            ax.grid(True, linestyle="--", alpha=0.6)
        fig.supxlabel("Layer")
        fig.supylabel("<Z>")
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":

    L = 4
    num_layers = 15
    noise_strength = 0.1

    from qiskit import QuantumCircuit

    def x_commuting_brickwork(n_qubits: int,
                            depth: int,
                            alpha: float = 0.37,
                            beta: float = 0.28,
                            add_barriers: bool = False) -> QuantumCircuit:
        """
        A scalable circuit built from Rx and RXX gates only.
        For X-Pauli noise and Z-observables, projector unraveling attains very low variance:
        the first projector jump pins qubits into X-eigenstates and <Z> stays 0 thereafter.

        Parameters
        ----------
        n_qubits : number of qubits
        depth    : number of repeated layers
        alpha    : per-layer single-qubit Rx angle
        beta     : per-layer two-qubit RXX angle
        add_barriers : insert barriers labeled 'SAMPLE_OBSERVABLES' after each layer

        Returns
        -------
        QuantumCircuit
        """
        qc = QuantumCircuit(n_qubits)
        for layer in range(depth):
            # Local rotations generated by X (commute with X_i)
            for q in range(n_qubits):
                qc.rx(alpha, q)

            # Brickwork of RXX couplings (also commute with every X_i)
            # even bonds
            for q in range(0, n_qubits - 1, 2):
                qc.rxx(beta, q, q + 1)
            # odd bonds
            for q in range(1, n_qubits - 1, 2):
                qc.rxx(beta, q, q + 1)

            if add_barriers:
                qc.barrier(label="SAMPLE_OBSERVABLES")  # sample ⟨Z_i⟩ after each layer

        return qc



    # basis_circuit = create_heisenberg_circuit(L, 1, 0.5, 0.5, 0.1, 0.1, 1)
    basis_circuit = x_commuting_brickwork(L, 1, add_barriers=False)
    # for i in range(L):
    #     basis_circuit.rx(np.pi/2, i)
    #     basis_circuit.barrier()
    # for i in range(L-1):
    #     basis_circuit.rzz(np.pi/2, i, i+1)
    #     basis_circuit.barrier()

    

    processes = [
        {"name": "pauli_x", "sites": [i], "strength": noise_strength}
        for i in range(L)] + [{"name": "crosstalk_xx", "sites": [i, i+1], "strength": noise_strength}
        for i in range(L-1) 
        ]
    noise_model_normal, noise_model_projector, noise_model_unitary_2pt, noise_model_unitary_gauss = build_noise_models(processes)

    qiskit_noise_model = QiskitNoiseModel()
    TwoQubit_XX_error = PauliLindbladError([Pauli("IX"), Pauli("XI"), Pauli("XX")], [noise_strength, noise_strength, noise_strength])
    #TwoQubit_YY_error = PauliLindbladError([Pauli("IY"), Pauli("YI"), Pauli("YY")], [noise_strength, noise_strength, noise_strength])
    #TwoQubit_ZZ_error = PauliLindbladError([Pauli("IZ"), Pauli("ZI"), Pauli("ZZ")], [noise_strength, noise_strength, noise_strength])
    for qubit in range(L-1):
        qiskit_noise_model.add_quantum_error(TwoQubit_XX_error, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])
        # qiskit_noise_model.add_quantum_error(TwoQubit_YY_error, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])
        # qiskit_noise_model.add_quantum_error(TwoQubit_ZZ_error, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])


    exact = run_qiskit(L, num_layers, basis_circuit, qiskit_noise_model, method="density_matrix")
    qiskit_mps = run_qiskit(L, num_layers, basis_circuit, qiskit_noise_model, method="matrix_product_state")
    yaqs_results_normal = run_yaqs(basis_circuit, L, num_layers, noise_model_normal)
    yaqs_results_projector = run_yaqs(basis_circuit, L, num_layers, noise_model_projector)
    yaqs_results_unitary_2pt = run_yaqs(basis_circuit, L, num_layers, noise_model_unitary_2pt)
    yaqs_results_unitary_gauss = run_yaqs(basis_circuit, L, num_layers, noise_model_unitary_gauss)


    series_by_label = {
    "standard": yaqs_results_normal,
    "projector": yaqs_results_projector,
    "unitary_2pt": yaqs_results_unitary_2pt,
    "unitary_gauss": yaqs_results_unitary_gauss,
    "qiskit_mps": qiskit_mps,
    }

    # print variances (lower is better)
    variances = print_variances_against_exact(exact, series_by_label)
    # print mean absolute errors (lower is better)
    mean_errors = print_mean_errors_against_exact(exact, series_by_label)

    # plot results vs exact
    plot_series_against_exact(
        exact,
        series_by_label,
        num_qubits=L,
        num_layers=num_layers,
    )


