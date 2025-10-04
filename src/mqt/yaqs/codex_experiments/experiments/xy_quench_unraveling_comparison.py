import numpy as np
import matplotlib.pyplot as plt
from qutip import sigmax, sigmay, sigmaz, qeye, tensor, sesolve, basis

from ..worker_functions.qiskit_simulators import run_qiskit_exact, run_qiskit_mps
from ..worker_functions.yaqs_simulator import run_yaqs, build_noise_models
from ..worker_functions.qiskit_noisy_sim import qiskit_noisy_simulator

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer.noise.errors import PauliLindbladError
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel

def staggered_magnetization(z, num_qubits):
    return np.sum([(-1)**i * z[i] for i in range(num_qubits)]) / num_qubits


def run_qutip_exact(N, times, init_state=None):
    from qutip import sigmax, sigmay, sigmaz, qeye, tensor, sesolve, basis

    X, Y, Z, I = sigmax(), sigmay(), sigmaz(), qeye(2)

    H = 0
    # open-chain nearest neighbors (OBC)
    for i in range(N - 1):
        H += tensor(*([I]*i + [X, X] + [I]*(N - i - 2)))
        H += tensor(*([I]*i + [Y, Y] + [I]*(N - i - 2)))

    psi0 = tensor(*[basis(2, 1 if i % 4 == 3 else 0) for i in range(N)]) if init_state is None else init_state

    eops = [tensor(*([I]*i + [Z] + [I]*(N - i - 1))) for i in range(N)]
    result = sesolve(H, psi0, times, e_ops=eops)
    return np.array([result.expect[i] for i in range(N)])



def xy_trotter_layer(N, tau, order="YX") -> QuantumCircuit:
    """
    Create one Trotter step for the XY Hamiltonian: H = sum_i (X_i X_{i+1} + Y_i Y_{i+1})
    
    Args:
        N: Number of qubits
        tau: Trotter time step
        order: Order of applying gates, either "YX" or "XY"
    
    Returns:
        QuantumCircuit implementing exp(-i*tau*H)
    """
    qc = QuantumCircuit(N)
    # Open boundary conditions (adjacent sites only)
    even = [(i, i+1) for i in range(0, N-1, 2)]
    odd  = [(i, i+1) for i in range(1, N-1, 2)]

    def apply_pairwise(gate_name):
        # Apply gates in two sublayers to avoid overlapping qubits
        for a, b in even: 
            getattr(qc, gate_name)(2*tau, a, b)
        for a, b in odd:  
            getattr(qc, gate_name)(2*tau, a, b)

    if order == "YX":
        apply_pairwise("ryy")          # exp(-i τ sum_i Y_i Y_{i+1})
        apply_pairwise("rxx")          # exp(-i τ sum_i X_i X_{i+1})
    else:
        apply_pairwise("rxx")
        apply_pairwise("ryy")

    # qc.draw(output="mpl")
    # plt.show()
    
    return qc



if __name__ == "__main__":
    num_qubits = 10
    num_layers = 50
    tau = 0.1
    noise_strength = 0.1
    num_traj = 24

    # Prepare initial state circuit
    init_circuit = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        if i % 4 == 3:  # Down-spin at every 4th site
            init_circuit.x(i)
    
    # One Trotter step (without initialization)
    trotter_step = xy_trotter_layer(num_qubits, tau)

    # Initialize noise models (YAQS)
    processes = [
        {"name": "pauli_x", "sites": [i], "strength": noise_strength}
        for i in range(num_qubits)
    ] + [
        {"name": "crosstalk_xx", "sites": [i, i+1], "strength": noise_strength}
        for i in range(num_qubits - 1)  # OBC: only adjacent sites, no wrap-around
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

    # Run all trajectory methods
    print("Running Qiskit MPS...")
    qiskit_mps, _, _ = run_qiskit_mps(
        num_qubits, num_layers, init_circuit, trotter_step, qiskit_noise_model,
        num_traj=num_traj
    )
    
    print("Running YAQS standard unraveling...")
    yaqs_results_normal, _, _ = run_yaqs(init_circuit, trotter_step, num_qubits, num_layers, noise_model_normal, num_traj=num_traj)
    
    print("Running YAQS projector unraveling...")
    yaqs_results_projector, _, _ = run_yaqs(init_circuit, trotter_step, num_qubits, num_layers, noise_model_projector, num_traj=num_traj)
    
    print("Running YAQS unitary 2pt unraveling...")
    yaqs_results_unitary_2pt, _, _ = run_yaqs(init_circuit, trotter_step, num_qubits, num_layers, noise_model_unitary_2pt, num_traj=num_traj)
    
    print("Running YAQS unitary gauss unraveling...")
    yaqs_results_unitary_gauss, _, _ = run_yaqs(init_circuit, trotter_step, num_qubits, num_layers, noise_model_unitary_gauss, num_traj=num_traj)

    # Run exact density matrix simulation with noise (reference)
    print("Running exact density matrix (noisy reference)...")
    z_expvals_qiskit_exact_noisy = run_qiskit_exact(
        num_qubits, num_layers, init_circuit, trotter_step, qiskit_noise_model, method="density_matrix"
    )

    # Run noiseless reference (Qiskit statevector for Trotter)
    print("Running noiseless reference...")
    z_expvals_qiskit_list = [[] for _ in range(num_qubits)]
    for layer in range(1, num_layers + 1):
        qc = init_circuit.copy()
        for _ in range(layer):
            composed = qc.compose(trotter_step)
            assert composed is not None
            qc = composed
        vals = np.real(np.asarray(qiskit_noisy_simulator(qc, None, num_qubits, 1, method="statevector"))).flatten()
        for q in range(num_qubits):
            z_expvals_qiskit_list[q].append(float(vals[q]))
    z_expvals_qiskit_noiseless = np.array(z_expvals_qiskit_list)

    # Run QuTiP exact simulation  
    print("Running QuTiP exact...")
    times = np.arange(num_layers + 1) * tau
    z_expvals_qutip = run_qutip_exact(num_qubits, times)
    
    # Compute initial staggered magnetization (t=0) for all methods
    # Initial state: |00010001⟩ with qubits 3,7 in |1⟩ (Z=-1), others in |0⟩ (Z=+1)
    z_initial = np.array([1.0 if i % 4 != 3 else -1.0 for i in range(num_qubits)])
    stag_initial = staggered_magnetization(z_initial, num_qubits)
    
    # Compute staggered magnetization for all methods (including t=0)
    # Exact noisy reference (density matrix)
    qiskit_exact_noisy_stag = [stag_initial] + [staggered_magnetization(z_expvals_qiskit_exact_noisy[:, t], num_qubits) for t in range(num_layers)]
    
    # Qiskit MPS (noisy trajectory): prepend initial value
    qiskit_mps_stag = [stag_initial] + [staggered_magnetization(qiskit_mps[:, t], num_qubits) for t in range(num_layers)]
    
    # YAQS methods (noisy trajectories): prepend initial value
    yaqs_normal_stag = [stag_initial] + [staggered_magnetization(yaqs_results_normal[:, t], num_qubits) for t in range(num_layers)]
    yaqs_projector_stag = [stag_initial] + [staggered_magnetization(yaqs_results_projector[:, t], num_qubits) for t in range(num_layers)]
    yaqs_unitary_2pt_stag = [stag_initial] + [staggered_magnetization(yaqs_results_unitary_2pt[:, t], num_qubits) for t in range(num_layers)]
    yaqs_unitary_gauss_stag = [stag_initial] + [staggered_magnetization(yaqs_results_unitary_gauss[:, t], num_qubits) for t in range(num_layers)]
    
    # Noiseless references (include t=0)
    qiskit_noiseless_stag = [stag_initial] + [staggered_magnetization(z_expvals_qiskit_noiseless[:, t], num_qubits) for t in range(num_layers)]
    qutip_stag = [staggered_magnetization(z_expvals_qutip[:, t], num_qubits) for t in range(num_layers + 1)]

    # Compute absolute errors relative to exact density matrix solution
    qiskit_mps_error = [abs(qiskit_mps_stag[i] - qiskit_exact_noisy_stag[i]) for i in range(len(times))]
    yaqs_normal_error = [abs(yaqs_normal_stag[i] - qiskit_exact_noisy_stag[i]) for i in range(len(times))]
    yaqs_projector_error = [abs(yaqs_projector_stag[i] - qiskit_exact_noisy_stag[i]) for i in range(len(times))]
    yaqs_unitary_2pt_error = [abs(yaqs_unitary_2pt_stag[i] - qiskit_exact_noisy_stag[i]) for i in range(len(times))]
    yaqs_unitary_gauss_error = [abs(yaqs_unitary_gauss_stag[i] - qiskit_exact_noisy_stag[i]) for i in range(len(times))]
    
    # Compute MSE (Mean Squared Error) for each method
    def compute_mse(pred, exact):
        return np.mean([(pred[i] - exact[i])**2 for i in range(len(pred))])
    
    mse_qiskit_mps = compute_mse(qiskit_mps_stag, qiskit_exact_noisy_stag)
    mse_yaqs_normal = compute_mse(yaqs_normal_stag, qiskit_exact_noisy_stag)
    mse_yaqs_projector = compute_mse(yaqs_projector_stag, qiskit_exact_noisy_stag)
    mse_yaqs_unitary_2pt = compute_mse(yaqs_unitary_2pt_stag, qiskit_exact_noisy_stag)
    mse_yaqs_unitary_gauss = compute_mse(yaqs_unitary_gauss_stag, qiskit_exact_noisy_stag)
    
    # Print MSE values
    print("\n" + "="*60)
    print("MSE (Mean Squared Error) vs Exact Density Matrix:")
    print("="*60)
    print(f"Qiskit MPS (Trajectories):    {mse_qiskit_mps:.6e}")
    print(f"YAQS Standard:                {mse_yaqs_normal:.6e}")
    print(f"YAQS Projector:               {mse_yaqs_projector:.6e}")
    print(f"YAQS Unitary 2pt:             {mse_yaqs_unitary_2pt:.6e}")
    print(f"YAQS Unitary Gauss:           {mse_yaqs_unitary_gauss:.6e}")
    print("="*60)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Subplot 1: Staggered magnetization
    ax1.plot(times, qiskit_exact_noisy_stag, '-', label="Exact (Density Matrix, Noisy)", 
             alpha=1.0, linewidth=3, color='red', zorder=10)
    ax1.plot(times, qutip_stag, '-', label="QuTiP (Exact Hamiltonian)", alpha=0.6, linewidth=2, color='black')
    ax1.plot(times, qiskit_noiseless_stag, '--', label="Qiskit Trotter (Noiseless)", alpha=0.5, linewidth=1.5, color='gray')
    ax1.plot(times, qiskit_mps_stag, '-o', label="Qiskit MPS (Trajectories)", alpha=0.8, markersize=3)
    ax1.plot(times, yaqs_normal_stag, '-s', label="YAQS Standard", alpha=0.8, markersize=3)
    ax1.plot(times, yaqs_projector_stag, '-^', label="YAQS Projector", alpha=0.8, markersize=3)
    ax1.plot(times, yaqs_unitary_2pt_stag, '-d', label="YAQS Unitary 2pt", alpha=0.8, markersize=3)
    ax1.plot(times, yaqs_unitary_gauss_stag, '-v', label="YAQS Unitary Gauss", alpha=0.8, markersize=3)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel(r"$S^z(\pi) = \sum_i (-1)^i \langle Z_i \rangle / N$", fontsize=12)
    ax1.set_title(f"XY Model Quench: Staggered Magnetization (N={num_qubits}, noise={noise_strength})", fontsize=13)
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: Absolute errors
    ax2.plot(times, qiskit_mps_error, '-o', label=f"Qiskit MPS (MSE={mse_qiskit_mps:.2e})", alpha=0.8, markersize=3)
    ax2.plot(times, yaqs_normal_error, '-s', label=f"YAQS Standard (MSE={mse_yaqs_normal:.2e})", alpha=0.8, markersize=3)
    ax2.plot(times, yaqs_projector_error, '-^', label=f"YAQS Projector (MSE={mse_yaqs_projector:.2e})", alpha=0.8, markersize=3)
    ax2.plot(times, yaqs_unitary_2pt_error, '-d', label=f"YAQS Unitary 2pt (MSE={mse_yaqs_unitary_2pt:.2e})", alpha=0.8, markersize=3)
    ax2.plot(times, yaqs_unitary_gauss_error, '-v', label=f"YAQS Unitary Gauss (MSE={mse_yaqs_unitary_gauss:.2e})", alpha=0.8, markersize=3)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_ylabel("Absolute Error", fontsize=12)
    ax2.set_title("Absolute Error vs Exact Density Matrix", fontsize=13)
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("xy_quench_unraveling_comparison.png", dpi=300)
    plt.show()

    print("\nPlot saved as 'xy_quench_unraveling_comparison.png'")
    