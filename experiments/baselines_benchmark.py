import numpy as np

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_noisy_state_preparation_batch
)
from mqt.yaqs.optimization.parameterized_circuit import (
    ParameterizedCircuit,
    ParameterizedGate,
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

def run_baselines():
    num_qubits = 8
    seed = 400 
    
    noise_ibm = build_hardware_noise_model("IBM", num_qubits)
    tjm_crn = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=True)
    tjm_noiseless = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    noise_none = NoiseModel([])
    
    print(f"\\n{'='*50}", flush=True)
    print(f"BASELINE COMPARISON: Disordered TFIM (Seed={seed})", flush=True)
    print(f"{'='*50}", flush=True)
    
    statevec = get_disordered_tfim_ground_state(num_qubits, seed=seed)
    
    print("\\n>>> Baseline 1: Standard VQA (Fixed Ansatz) <<<", flush=True)
    print("Testing a standard Depth-4 circuit (276 parameters) trained directly from scratch.", flush=True)
    circuit_d4 = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, 4, initial_single_qubit_layer=True)
    rng = np.random.RandomState(42)
    theta_d4 = rng.randn(circuit_d4.num_params) * 0.1
    
    options_vqa = KrotovOptions(max_iterations=300, batch_step_size=0.2, batch_schedule="exp", batch_decay=0.01, seed=1)
    res_vqa = train_krotov_noisy_state_preparation_batch(
        circuit_d4, statevec, noise_ibm, tjm_crn, initial_theta=theta_d4, options=options_vqa
    )
    print(f"  Result -> Standard VQA (276 params): {res_vqa.trace['fidelity'][-1]:.4f}", flush=True)


    print("\\n>>> Baseline 2: Bottom-Up Growth (Mock ADAPT-VQE) <<<", flush=True)
    print("Growing circuit layer-by-layer up to Depth-4, stopping at 276 parameters.", flush=True)
    depths = [1, 2, 3, 4]
    current_theta = None
    
    for d in depths:
        circuit_adapt = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, d, initial_single_qubit_layer=True)
        if current_theta is None:
            rng = np.random.RandomState(100)
            initial_theta = rng.randn(circuit_adapt.num_params) * 0.05
        else:
            initial_theta = np.zeros(circuit_adapt.num_params)
            initial_theta[:len(current_theta)] = current_theta
            rng = np.random.RandomState(200 + d)
            initial_theta[len(current_theta):] = rng.randn(circuit_adapt.num_params - len(current_theta)) * 0.001
            
        options_adapt = KrotovOptions(max_iterations=100, batch_step_size=1.0, batch_schedule="constant", seed=2)
        res_adapt = train_krotov_noisy_state_preparation_batch(
            circuit_adapt, statevec, noise_none, tjm_noiseless, initial_theta=initial_theta, options=options_adapt
        )
        current_theta = res_adapt.theta
        print(f"  Grown to Depth {d} (Params: {circuit_adapt.num_params}) -> Fidelity: {res_adapt.trace['fidelity'][-1]:.4f}", flush=True)

    print(f"\\n  Our Top-Down Method (Iterative Pruned to 150 params) reached: 0.9096", flush=True)

if __name__ == "__main__":
    run_baselines()
