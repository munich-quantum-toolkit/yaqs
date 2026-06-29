import numpy as np

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_noisy_state_preparation_batch
)
from mqt.yaqs.optimization.parameterized_circuit import (
    ParameterizedCircuit,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
)

def get_disordered_tfim_ground_state(num_qubits: int, seed: int):
    rng = np.random.RandomState(seed)
    dim = 2**num_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    
    J = rng.uniform(0.8, 1.2, size=num_qubits - 1)
    h = rng.uniform(0.8, 1.2, size=num_qubits)
    
    for i in range(num_qubits - 1):
        op = [I]*num_qubits
        op[i] = Z
        op[i+1] = Z
        term = op[0]
        for j in range(1, num_qubits):
            term = np.kron(term, op[j])
        H -= J[i] * term
        
    for i in range(num_qubits):
        op = [I]*num_qubits
        op[i] = X
        term = op[0]
        for j in range(1, num_qubits):
            term = np.kron(term, op[j])
        H -= h[i] * term
        
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvectors[:, 0]

def build_hardware_noise_model(hardware: str, num_qubits: int):
    processes = []
    if hardware == "IBM":
        p_1q = 3e-4 / 3.0  
        p_2q = 3e-3 / 2.0  
        for i in range(num_qubits):
            for pauli in ["pauli_x", "pauli_y", "pauli_z"]:
                processes.append({"name": pauli, "sites": [i], "strength": p_1q})
        for i in range(num_qubits - 1):
            for crosstalk in ["crosstalk_xx", "crosstalk_zz"]:
                processes.append({"name": crosstalk, "sites": [i, i+1], "strength": p_2q})
    return NoiseModel(processes)

def run_ablations():
    num_qubits = 8
    seed = 400 
    statevec = get_disordered_tfim_ground_state(num_qubits, seed=seed)
    
    print(f"\\n{'='*50}", flush=True)
    print(f"ABLATION STUDIES: Disordered TFIM (Seed={seed})", flush=True)
    print(f"{'='*50}", flush=True)

    # ---------------------------------------------------------
    # Ablation A: CRN vs No-CRN
    # ---------------------------------------------------------
    print("\\n>>> Ablation A: CRN vs No-CRN under Noisy Fine-Tuning <<<", flush=True)
    noise_ibm = build_hardware_noise_model("IBM", num_qubits)
    
    circuit_d4 = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, 4, initial_single_qubit_layer=True)
    rng = np.random.RandomState(999)
    theta_d4 = rng.randn(circuit_d4.num_params) * 0.1
    
    options_finetune = KrotovOptions(max_iterations=150, batch_step_size=0.2, batch_schedule="exp", batch_decay=0.01, seed=1)
    
    tjm_with_crn = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=True)
    res_crn = train_krotov_noisy_state_preparation_batch(
        circuit_d4, statevec, noise_ibm, tjm_with_crn, initial_theta=theta_d4, options=options_finetune
    )
    print(f"  Fidelity WITH CRN stabilization: {res_crn.trace['fidelity'][-1]:.4f}", flush=True)
    
    tjm_no_crn = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=False)
    res_nocrn = train_krotov_noisy_state_preparation_batch(
        circuit_d4, statevec, noise_ibm, tjm_no_crn, initial_theta=theta_d4, options=options_finetune
    )
    print(f"  Fidelity WITHOUT CRN stabilization: {res_nocrn.trace['fidelity'][-1]:.4f}", flush=True)


    # ---------------------------------------------------------
    # Ablation B: Layerwise Pre-Training vs All-at-Once
    # ---------------------------------------------------------
    print("\\n>>> Ablation B: Layerwise vs All-at-Once Pre-Training (Depth 16) <<<", flush=True)
    noise_none = NoiseModel([])
    tjm_noiseless = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    
    # 1. All-at-Once
    circuit_d16 = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, 16, initial_single_qubit_layer=True)
    rng = np.random.RandomState(42)
    theta_d16 = rng.randn(circuit_d16.num_params) * 0.1
    options_all = KrotovOptions(max_iterations=1000, batch_step_size=10.0, batch_schedule="constant", seed=10)
    
    print("  Running All-at-Once (Depth-16 for 1000 iterations)...", flush=True)
    res_all = train_krotov_noisy_state_preparation_batch(
        circuit_d16, statevec, noise_none, tjm_noiseless, initial_theta=theta_d16, options=options_all
    )
    print(f"  Result -> All-at-Once Fidelity: {res_all.trace['fidelity'][-1]:.4f}", flush=True)
    
    # 2. Layerwise (We already know the result from our previous scripts, but let's re-run it quickly)
    print("  Running Layerwise Expansion (Depth 4 -> 8 -> 12 -> 16)...", flush=True)
    depths = [4, 8, 12, 16]
    step_sizes = [10.0, 2.0, 0.5, 0.1]
    max_iters_per_depth = 250
    current_theta = None
    
    for i, d in enumerate(depths):
        circuit_layer = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, d, initial_single_qubit_layer=True)
        if current_theta is None:
            rng = np.random.RandomState(100)
            initial_theta = rng.randn(circuit_layer.num_params) * 0.05
        else:
            initial_theta = np.zeros(circuit_layer.num_params)
            initial_theta[:len(current_theta)] = current_theta
            rng = np.random.RandomState(200 + d)
            initial_theta[len(current_theta):] = rng.randn(circuit_layer.num_params - len(current_theta)) * 0.001
            
        options_layer = KrotovOptions(
            max_iterations=max_iters_per_depth, batch_step_size=step_sizes[i],
            batch_schedule="exp", batch_decay=0.01, seed=300
        )
        res_layer = train_krotov_noisy_state_preparation_batch(
            circuit_layer, statevec, noise_none, tjm_noiseless, initial_theta=initial_theta, options=options_layer
        )
        current_theta = res_layer.theta
        print(f"    Depth {d} fidelity: {res_layer.trace['fidelity'][-1]:.4f}", flush=True)

if __name__ == "__main__":
    run_ablations()
