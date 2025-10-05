import numpy as np
import matplotlib.pyplot as plt
from qutip import sigmax, sigmay, sigmaz, qeye, tensor, sesolve, basis

from ..worker_functions.qiskit_simulators import run_qiskit_exact, run_qiskit_mps
from ..worker_functions.yaqs_simulator import run_yaqs, build_noise_models
from ..worker_functions.qiskit_noisy_sim import qiskit_noisy_simulator
from ..worker_functions.plotting import plot_avg_bond_dims

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer.noise.errors import PauliLindbladError
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel

def staggered_magnetization(z, num_qubits):
    return np.sum([(-1)**i * z[i] for i in range(num_qubits)]) / num_qubits


def xy_trotter_layer(N, tau, order="YX") -> QuantumCircuit:
    """Create one Trotter step for the XY Hamiltonian."""
    qc = QuantumCircuit(N)
    even = [(i, i+1) for i in range(0, N-1, 2)]
    odd  = [(i, i+1) for i in range(1, N-1, 2)]

    def apply_pairwise(gate_name):
        for a, b in even: 
            getattr(qc, gate_name)(2*tau, a, b)
        for a, b in odd:  
            getattr(qc, gate_name)(2*tau, a, b)

    if order == "YX":
        apply_pairwise("ryy")
        apply_pairwise("rxx")
    else:
        apply_pairwise("rxx")
        apply_pairwise("ryy")
    
    return qc


def compute_mse(pred, exact):
    """Compute Mean Squared Error between prediction and exact solution."""
    return np.mean([(pred[i] - exact[i])**2 for i in range(len(pred))])


def find_required_trajectories(
    method_name,
    simulator_func,
    exact_stag,
    threshold_mse,
    init_circuit,
    trotter_step,
    num_qubits,
    num_layers,
    noise_model,
    qiskit_noise_model,
    stag_initial,
    max_traj=1000
):
    """
    Find minimum number of trajectories needed to achieve target MSE.
    
    Incrementally tests 1, 2, 3, ... trajectories until threshold is met.
    
    Args:
        method_name: Name of the method for logging
        simulator_func: Function to run simulation (run_yaqs or run_qiskit_mps)
        exact_stag: Exact staggered magnetization reference
        threshold_mse: Target MSE threshold
        max_traj: Maximum trajectories to try
    
    Returns:
        (num_trajectories_needed, final_mse, stag_values)
    """
    for num_traj in range(1, max_traj + 1):
        print(f"  {method_name}: Testing with {num_traj} trajectory(ies)...")
        
        # Run simulation
        if method_name.startswith("YAQS"):
            results, bond_dims, _ = simulator_func(
                init_circuit, trotter_step, num_qubits, num_layers, 
                noise_model, num_traj=num_traj, parallel=True
            )
        else:  # Qiskit MPS
            results, bond_dims, _ = simulator_func(
                num_qubits, num_layers, init_circuit, trotter_step,
                qiskit_noise_model, num_traj=num_traj
            )
        
        # Compute staggered magnetization
        stag = [stag_initial] + [staggered_magnetization(results[:, t], num_qubits) 
                                  for t in range(num_layers)]
        
        # Compute MSE
        mse = compute_mse(stag, exact_stag)
        
        print(f"    MSE = {mse:.6e} (threshold = {threshold_mse:.6e})")
        
        # Check if threshold is met
        if mse < threshold_mse:
            print(f"  ✓ {method_name}: Target reached with {num_traj} trajectory(ies)!")
            return num_traj, mse, stag, bond_dims
    
    print(f"  ✗ {method_name}: Failed to reach threshold with {max_traj} trajectories")
    return max_traj, mse, stag, bond_dims


if __name__ == "__main__":
    # Simulation parameters
    num_qubits = 8
    num_layers = 50
    tau = 0.1
    noise_strength = 0.1
    threshold_mse = 5e-4  # Target MSE threshold
    
    print("="*70)
    print("Trajectory Efficiency Comparison for Unraveling Methods")
    print("="*70)
    print(f"System: {num_qubits} qubits, {num_layers} layers")
    print(f"Noise strength: {noise_strength}")
    print(f"Target MSE threshold: {threshold_mse:.2e}")
    print("="*70)

    # Prepare initial state circuit
    init_circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        if i % 4 == 3:
            init_circuit.x(i)
    
    # One Trotter step
    trotter_step = xy_trotter_layer(num_qubits, tau)

    # Initialize noise models (YAQS)
    processes = [
        {"name": "pauli_x", "sites": [i], "strength": noise_strength}
        for i in range(num_qubits)
    ] + [
        {"name": "crosstalk_xx", "sites": [i, i+1], "strength": noise_strength}
        for i in range(num_qubits - 1)
    ]
    noise_model_normal, noise_model_projector, noise_model_unitary_2pt, noise_model_unitary_gauss = build_noise_models(processes)

    # Initialize Qiskit noise model
    qiskit_noise_model = QiskitNoiseModel()
    TwoQubit_XX_error = PauliLindbladError(
        [Pauli("IX"), Pauli("XI"), Pauli("XX")],
        [noise_strength, noise_strength, noise_strength]
    )
    for qubit in range(num_qubits):
        next_qubit = (qubit + 1) % num_qubits
        qiskit_noise_model.add_quantum_error(
            TwoQubit_XX_error,
            ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"],
            [qubit, next_qubit]
        )

    # Run exact density matrix simulation (reference)
    print("\nRunning exact density matrix simulation (reference)...")
    z_expvals_exact = run_qiskit_exact(
        num_qubits, num_layers, init_circuit, trotter_step, 
        qiskit_noise_model, method="density_matrix"
    )
    
    # Compute exact staggered magnetization
    z_initial = np.array([1.0 if i % 4 != 3 else -1.0 for i in range(num_qubits)])
    stag_initial = staggered_magnetization(z_initial, num_qubits)
    exact_stag = [stag_initial] + [staggered_magnetization(z_expvals_exact[:, t], num_qubits) 
                                     for t in range(num_layers)]

    print("Exact reference computed.\n")

    # Test each method
    results = {}
    
    print("Finding minimum trajectories for each method...")
    print("-"*70)
    
    # Qiskit MPS
    print("\n1. Qiskit MPS (Standard Unraveling)")
    num_traj_mps, mse_mps, stag_mps, bonds_mps = find_required_trajectories(
        "Qiskit MPS",
        run_qiskit_mps,
        exact_stag,
        threshold_mse,
        init_circuit,
        trotter_step,
        num_qubits,
        num_layers,
        None,
        qiskit_noise_model,
        stag_initial,
        max_traj=1000
    )
    results["Qiskit MPS"] = {"trajectories": num_traj_mps, "mse": mse_mps, "stag": stag_mps, "bonds": bonds_mps}
    
    # YAQS Standard
    print("\n2. YAQS Standard Unraveling")
    num_traj_std, mse_std, stag_std, bonds_std = find_required_trajectories(
        "YAQS Standard",
        run_yaqs,
        exact_stag,
        threshold_mse,
        init_circuit,
        trotter_step,
        num_qubits,
        num_layers,
        noise_model_normal,
        None,
        stag_initial,
        max_traj=1000
    )
    results["YAQS Standard"] = {"trajectories": num_traj_std, "mse": mse_std, "stag": stag_std, "bonds": bonds_std}
    
    # YAQS Projector
    print("\n3. YAQS Projector Unraveling")
    num_traj_proj, mse_proj, stag_proj, bonds_proj = find_required_trajectories(
        "YAQS Projector",
        run_yaqs,
        exact_stag,
        threshold_mse,
        init_circuit,
        trotter_step,
        num_qubits,
        num_layers,
        noise_model_projector,
        None,
        stag_initial,
        max_traj=1000
    )
    results["YAQS Projector"] = {"trajectories": num_traj_proj, "mse": mse_proj, "stag": stag_proj, "bonds": bonds_proj}
    
    # YAQS Unitary 2pt
    print("\n4. YAQS Unitary 2pt Unraveling")
    num_traj_2pt, mse_2pt, stag_2pt, bonds_2pt = find_required_trajectories(
        "YAQS Unitary 2pt",
        run_yaqs,
        exact_stag,
        threshold_mse,
        init_circuit,
        trotter_step,
        num_qubits,
        num_layers,
        noise_model_unitary_2pt,
        None,
        stag_initial,
        max_traj=1000
    )
    results["YAQS Unitary 2pt"] = {"trajectories": num_traj_2pt, "mse": mse_2pt, "stag": stag_2pt, "bonds": bonds_2pt}
    
    # YAQS Unitary Gauss
    print("\n5. YAQS Unitary Gauss Unraveling")
    num_traj_gauss, mse_gauss, stag_gauss, bonds_gauss = find_required_trajectories(
        "YAQS Unitary Gauss",
        run_yaqs,
        exact_stag,
        threshold_mse,
        init_circuit,
        trotter_step,
        num_qubits,
        num_layers,
        noise_model_unitary_gauss,
        None,
        stag_initial,
        max_traj=1000
    )
    results["YAQS Unitary Gauss"] = {"trajectories": num_traj_gauss, "mse": mse_gauss, "stag": stag_gauss, "bonds": bonds_gauss}

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<25} {'Trajectories':<15} {'Final MSE':<15} {'Speedup':<10}")
    print("-"*70)
    
    baseline_traj = results["YAQS Standard"]["trajectories"]
    for method, data in results.items():
        speedup = baseline_traj / data["trajectories"]
        print(f"{method:<25} {data['trajectories']:<15} {data['mse']:<15.2e} {speedup:<10.2f}x")
    
    print("="*70)

    # Organize bond dimension data in the format expected by plot_avg_bond_dims
    qiskit_bonds = results["Qiskit MPS"]["bonds"]
    yaqs_bonds_by_label = {
        "standard": results["YAQS Standard"]["bonds"],
        "projector": results["YAQS Projector"]["bonds"],
        "unitary_2pt": results["YAQS Unitary 2pt"]["bonds"],
        "unitary_gauss": results["YAQS Unitary Gauss"]["bonds"],
    }
    
    # Process bond dimensions for plotting (using same logic as plot_avg_bond_dims)
    layers = np.arange(1, num_layers + 1)
    bond_data_for_plot = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    method_names = list(results.keys())
    
    # Process Qiskit MPS bonds
    if qiskit_bonds is not None and "per_layer_mean_across_shots" in qiskit_bonds:
        q_mean = np.asarray(qiskit_bonds["per_layer_mean_across_shots"])
        bond_data_for_plot["Qiskit MPS"] = q_mean[:num_layers]
    
    # Process YAQS bonds (mean across trajectories; drop initial and final columns)
    yaqs_method_map = {
        "standard": "YAQS Standard",
        "projector": "YAQS Projector",
        "unitary_2pt": "YAQS Unitary 2pt",
        "unitary_gauss": "YAQS Unitary Gauss",
    }
    
    for label, arr in yaqs_bonds_by_label.items():
        method_name = yaqs_method_map[label]
        if arr is None:
            continue
        mean_per_col = np.mean(arr, axis=0)
        if mean_per_col.size >= 2:
            mean_layers = mean_per_col[1:-1]
        else:
            mean_layers = mean_per_col
        bond_data_for_plot[method_name] = mean_layers[:num_layers]

    # Create visualization with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Subplot 1: Bar chart of required trajectories
    trajectories = [results[m]["trajectories"] for m in method_names]
    
    bars = ax1.bar(range(len(method_names)), trajectories, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(method_names)))
    ax1.set_xticklabels(method_names, rotation=45, ha='right')
    ax1.set_ylabel("Number of Trajectories", fontsize=12)
    ax1.set_title(f"Trajectories Required to Reach MSE < {threshold_mse:.2e}", fontsize=13)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for i, (bar, traj) in enumerate(zip(bars, trajectories)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(traj)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Subplot 2: Staggered magnetization comparison
    times = np.arange(num_layers + 1) * tau
    ax2.plot(times, exact_stag, '-', label="Exact (Density Matrix)", 
             alpha=1.0, linewidth=3, color='red', zorder=10)
    
    for i, (method, data) in enumerate(results.items()):
        ax2.plot(times, data["stag"], '-o', label=f"{method} ({data['trajectories']} traj)", 
                 alpha=0.7, markersize=3, color=colors[i])
    
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel(r"$S^z(\pi)$", fontsize=12)
    ax2.set_title("Staggered Magnetization at Threshold", fontsize=13)
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 3: Bond dimension growth (using same logic as plot_avg_bond_dims)
    for i, method in enumerate(method_names):
        if method in bond_data_for_plot:
            bond_avg = bond_data_for_plot[method]
            ax3.plot(layers, bond_avg[:num_layers], '-o', label=method, 
                     alpha=0.8, markersize=4, color=colors[i], linewidth=2)
    
    ax3.set_xlabel("Layer", fontsize=12)
    ax3.set_ylabel("avg max bond dim", fontsize=12)
    ax3.set_title("Bond Dimension Growth", fontsize=13)
    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("unraveling_trajectory_efficiency.png", dpi=300)
    plt.show()
    
    print("\nPlot saved as 'unraveling_trajectory_efficiency.png'")

