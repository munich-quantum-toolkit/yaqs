import numpy as np

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_noisy_state_preparation_batch
)
from mqt.yaqs.optimization.parameterized_circuit import (
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

def test_adapt():
    num_qubits = 8
    seed = 400
    statevec = get_disordered_tfim_ground_state(num_qubits, seed=seed)
    noise_ibm = build_hardware_noise_model("IBM", num_qubits)
    tjm_crn = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=True)
    tjm_noiseless = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    noise_none = NoiseModel([])

    print("Growing circuit layer-by-layer up to Depth-2 (150 parameters)...")
    depths = [1, 2]
    current_theta = None
    circuit_adapt = None
    
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
        print(f"  Grown to Depth {d} (Params: {circuit_adapt.num_params}) -> Noiseless Fidelity: {res_adapt.trace['fidelity'][-1]:.4f}")
        
    print("\\nEvaluating the Depth-2 ADAPT circuit under IBM Heron noise (Standard, NO CRN)...")
    tjm_noisy_nocrn = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=False)
    options_eval = KrotovOptions(max_iterations=100, batch_step_size=0.1, batch_schedule="constant", seed=800)
    res_eval = train_krotov_noisy_state_preparation_batch(
        circuit_adapt, statevec, noise_ibm, tjm_noisy_nocrn, initial_theta=current_theta, options=options_eval
    )
    print(f"  Final Noisy Fidelity of ADAPT (150 params): {res_eval.trace['fidelity'][-1]:.4f}")

if __name__ == "__main__":
    test_adapt()
