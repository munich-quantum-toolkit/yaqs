import numpy as np
import sys

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

def impact_aware_prune(circuit: ParameterizedCircuit, theta: np.ndarray, statevec: np.ndarray, keep_k: int):
    noise_none = NoiseModel([])
    tjm_none = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    
    impacts = np.zeros(circuit.num_params)
    base_options = KrotovOptions(max_iterations=1, batch_step_size=0.0)
    
    eps = np.pi / 2
    for i in range(circuit.num_params):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        res_plus = train_krotov_noisy_state_preparation_batch(
            circuit, statevec, noise_none, tjm_none, initial_theta=theta_plus, options=base_options
        )
        fid_plus = res_plus.trace['fidelity'][-1]
        
        theta_minus = theta.copy()
        theta_minus[i] -= eps
        res_minus = train_krotov_noisy_state_preparation_batch(
            circuit, statevec, noise_none, tjm_none, initial_theta=theta_minus, options=base_options
        )
        fid_minus = res_minus.trace['fidelity'][-1]
        
        grad = (fid_plus - fid_minus) / 2.0
        impacts[i] = np.abs(theta[i] * grad)
        
    threshold = np.sort(impacts)[-keep_k]
    keep_indices = np.where(impacts >= threshold)[0]
    if len(keep_indices) > keep_k:
        keep_indices = keep_indices[:keep_k]
        
    keep_indices = set(keep_indices)
    new_gates = []
    new_theta = []
    old_to_new_index = {}
    
    current_new_idx = 0
    for i in range(circuit.num_params):
        if i in keep_indices:
            old_to_new_index[i] = current_new_idx
            current_new_idx += 1
            new_theta.append(theta[i])
            
    for gate in circuit.gates:
        if isinstance(gate, ParameterizedGate):
            if gate.param_index in keep_indices:
                new_gate = ParameterizedGate(
                    name=gate.name, sites=gate.sites, param_index=old_to_new_index[gate.param_index],
                    angle_scale=gate.angle_scale, angle_offset=gate.angle_offset,
                    data_map=gate.data_map, fixed_params=gate.fixed_params
                )
                new_gates.append(new_gate)
        else:
            new_gates.append(gate)
            
    pruned_circuit = ParameterizedCircuit(
        num_qubits=circuit.num_qubits, gates=new_gates, num_params=len(new_theta)
    )
            
    return pruned_circuit, np.array(new_theta, dtype=np.float64)

def iterative_impact_prune(circuit: ParameterizedCircuit, theta: np.ndarray, statevec: np.ndarray, target_k: int):
    current_circuit = circuit
    current_theta = theta
    noise_none = NoiseModel([])
    tjm_none = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    
    while current_circuit.num_params > target_k:
        drop_count = min(150, current_circuit.num_params - target_k)
        next_k = current_circuit.num_params - drop_count
        print(f"    Iterative Prune: {current_circuit.num_params} -> {next_k}...", flush=True)
        
        pruned_circuit, new_theta = impact_aware_prune(current_circuit, current_theta, statevec, keep_k=next_k)
        
        if next_k > target_k:
            options = KrotovOptions(max_iterations=40, batch_step_size=0.2, batch_schedule="constant", seed=42)
            res = train_krotov_noisy_state_preparation_batch(
                pruned_circuit, statevec, noise_none, tjm_none, initial_theta=new_theta, options=options
            )
            current_theta = res.theta
        else:
            current_theta = new_theta
            
        current_circuit = pruned_circuit
        
    return current_circuit, current_theta

def find_hero_state():
    num_qubits = 8
    noise_ibm = build_hardware_noise_model("IBM", num_qubits)
    tjm_crn = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=True)
    tjm_noisy_nocrn = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=False)
    tjm_noiseless = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    noise_none = NoiseModel([])
    
    seeds = [300]
    
    for seed in seeds:
        print(f"\\n{'='*50}", flush=True)
        print(f"TESTING SEED: {seed}", flush=True)
        print(f"{'='*50}", flush=True)
        
        statevec = get_disordered_tfim_ground_state(num_qubits, seed=seed)
        
        # 1. Test Standard VQA (Depth 4)
        print(">>> 1. Standard VQA (276 gates, Noisy + CRN) <<<", flush=True)
        circuit_vqa = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, 4, initial_single_qubit_layer=True)
        rng = np.random.RandomState(seed + 1)
        theta_vqa = rng.randn(circuit_vqa.num_params) * 0.1
        
        options_vqa = KrotovOptions(max_iterations=300, batch_step_size=0.2, batch_schedule="exp", batch_decay=0.01, seed=seed + 2)
        res_vqa = train_krotov_noisy_state_preparation_batch(
            circuit_vqa, statevec, noise_ibm, tjm_crn, initial_theta=theta_vqa, options=options_vqa
        )
        fid_vqa = res_vqa.trace['fidelity'][-1]
        print(f"  VQA Fidelity: {fid_vqa:.4f}", flush=True)
        
        # 2. Test ADAPT-VQE (Depth 2)
        print("\\n>>> 2. ADAPT-VQE (150 gates, Noisy, No CRN) <<<", flush=True)
        depths = [1, 2]
        current_theta = None
        circuit_adapt = None
        for d in depths:
            circuit_adapt = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, d, initial_single_qubit_layer=True)
            if current_theta is None:
                rng = np.random.RandomState(seed + 100)
                initial_theta = rng.randn(circuit_adapt.num_params) * 0.05
            else:
                initial_theta = np.zeros(circuit_adapt.num_params)
                initial_theta[:len(current_theta)] = current_theta
                rng = np.random.RandomState(seed + 200 + d)
                initial_theta[len(current_theta):] = rng.randn(circuit_adapt.num_params - len(current_theta)) * 0.001
                
            options_adapt = KrotovOptions(max_iterations=100, batch_step_size=1.0, batch_schedule="constant", seed=seed + 300)
            res_adapt = train_krotov_noisy_state_preparation_batch(
                circuit_adapt, statevec, noise_none, tjm_noiseless, initial_theta=initial_theta, options=options_adapt
            )
            current_theta = res_adapt.theta
            
        options_eval = KrotovOptions(max_iterations=100, batch_step_size=0.1, batch_schedule="constant", seed=seed + 400)
        res_eval = train_krotov_noisy_state_preparation_batch(
            circuit_adapt, statevec, noise_ibm, tjm_noisy_nocrn, initial_theta=current_theta, options=options_eval
        )
        fid_adapt = res_eval.trace['fidelity'][-1]
        print(f"  ADAPT Fidelity: {fid_adapt:.4f}", flush=True)
        
        # 3. Test Our Method
        print("\\n>>> 3. Top-Down Pruning (150 gates, Noisy + CRN) <<<", flush=True)
        depths_ours = [4, 8, 12, 16]
        step_sizes = [5.0, 2.0, 0.5, 0.2]
        current_theta = None
        circuit_ours = None
        for i, d in enumerate(depths_ours):
            circuit_ours = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, d, initial_single_qubit_layer=True)
            if current_theta is None:
                rng = np.random.RandomState(seed + 1000)
                initial_theta = rng.randn(circuit_ours.num_params) * 0.05
            else:
                initial_theta = np.zeros(circuit_ours.num_params)
                initial_theta[:len(current_theta)] = current_theta
                rng = np.random.RandomState(seed + 2000 + d)
                initial_theta[len(current_theta):] = rng.randn(circuit_ours.num_params - len(current_theta)) * 0.001
                
            options = KrotovOptions(max_iterations=150, batch_step_size=step_sizes[i], batch_schedule="exp", batch_decay=0.01, seed=seed + 3000)
            res = train_krotov_noisy_state_preparation_batch(
                circuit_ours, statevec, noise_none, tjm_noiseless, initial_theta=initial_theta, options=options
            )
            current_theta = res.theta
            print(f"    Pre-train Depth {d} fidelity: {res.trace['fidelity'][-1]:.4f}", flush=True)
            
        pruned_circuit, theta_pruned = iterative_impact_prune(circuit_ours, current_theta, statevec, target_k=150)
        
        options_finetune = KrotovOptions(max_iterations=200, batch_step_size=0.1, batch_schedule="constant", seed=seed + 5000)
        res_ours = train_krotov_noisy_state_preparation_batch(
            pruned_circuit, statevec, noise_ibm, tjm_crn, initial_theta=theta_pruned, options=options_finetune
        )
        fid_ours = res_ours.trace['fidelity'][-1]
        print(f"  OURS Fidelity: {fid_ours:.4f}", flush=True)
        
        print(f"\\n!!! FINAL RESULTS FOR SEED {seed} !!!", flush=True)
        print(f"  VQA (276 gates):   {fid_vqa*100:.2f}%")
        print(f"  ADAPT (150 gates): {fid_adapt*100:.2f}%")
        print(f"  OURS (150 gates):  {fid_ours*100:.2f}%")
            
    print("FINISHED ALL SEEDS.")

if __name__ == "__main__":
    find_hero_state()
