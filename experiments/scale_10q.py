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

def impact_aware_prune(circuit: ParameterizedCircuit, theta: np.ndarray, statevec: np.ndarray, keep_k: int):
    noise_none = NoiseModel([])
    tjm_none = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    
    impacts = np.zeros(circuit.num_params)
    base_options = KrotovOptions(max_iterations=1, batch_step_size=0.0)
    base_res = train_krotov_noisy_state_preparation_batch(
        circuit, statevec, noise_none, tjm_none, initial_theta=theta, options=base_options
    )
    base_fid = base_res.trace['fidelity'][-1]
    
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
    for i in keep_indices:
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
        drop_count = min(300, current_circuit.num_params - target_k)
        next_k = current_circuit.num_params - drop_count
        print(f"\\n--- Iterative Prune: {current_circuit.num_params} -> {next_k} ---", flush=True)
        
        pruned_circuit, new_theta = impact_aware_prune(current_circuit, current_theta, statevec, keep_k=next_k)
        
        if next_k > target_k:
            print("Relaxing circuit...", flush=True)
            options = KrotovOptions(max_iterations=40, batch_step_size=0.2, batch_schedule="constant", seed=42)
            res = train_krotov_noisy_state_preparation_batch(
                pruned_circuit, statevec, noise_none, tjm_none, initial_theta=new_theta, options=options
            )
            current_theta = res.theta
            print(f"Relaxed fidelity: {res.trace['fidelity'][-1]:.4f}", flush=True)
        else:
            current_theta = new_theta
            
        current_circuit = pruned_circuit
        
    return current_circuit, current_theta

def scale_test():
    num_qubits = 10
    seed = 400
    statevec = get_disordered_tfim_ground_state(num_qubits, seed=seed)
    
    print(f"\\n{'='*50}", flush=True)
    print(f"SCALING BOUNDARY: {num_qubits}-Qubit Disordered TFIM (Seed={seed})", flush=True)
    print(f"{'='*50}", flush=True)
    
    # Pre-training configs
    noise_strength_pre = 0.0005
    processes_pre = [{"name": "pauli_x", "sites": [i], "strength": noise_strength_pre} for i in range(num_qubits)]
    noise_pretrain = NoiseModel(processes_pre)
    tjm_crn_pretrain = KrotovTJMOptions(num_trajectories=1, trajectory_update="cross", use_crn=True)
    
    print(f"--- Phase 1: Pre-training Layerwise to Depth-16 ---", flush=True)
    depths = [4, 8, 12, 16]
    step_sizes = [5.0, 2.0, 0.5, 0.2]
    max_iters_per_depth = 150
    
    current_theta = None
    final_pretrain_circuit = None
    
    for i, d in enumerate(depths):
        circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, d, initial_single_qubit_layer=True)
        final_pretrain_circuit = circuit
        
        if current_theta is None:
            rng = np.random.RandomState(seed + 1000)
            initial_theta = rng.randn(circuit.num_params) * 0.05
        else:
            initial_theta = np.zeros(circuit.num_params)
            initial_theta[:len(current_theta)] = current_theta
            rng = np.random.RandomState(seed + 2000 + d)
            initial_theta[len(current_theta):] = rng.randn(circuit.num_params - len(current_theta)) * 0.001
            
        options = KrotovOptions(
            max_iterations=max_iters_per_depth, batch_step_size=step_sizes[i],
            batch_schedule="exp", batch_decay=0.01, seed=seed + 3000
        )
        res = train_krotov_noisy_state_preparation_batch(
            circuit, statevec, noise_pretrain, tjm_crn_pretrain, initial_theta=initial_theta, options=options
        )
        current_theta = res.theta
        print(f"  Depth {d} final fidelity: {res.trace['fidelity'][-1]:.4f} (Params: {circuit.num_params})", flush=True)
        
    target_k = 250 # Scale up the pruned gate count a bit for 10 qubits
    pruned_circuit, theta_pruned = iterative_impact_prune(final_pretrain_circuit, current_theta, statevec, target_k)
    print(f"\\n--- Phase 2: Iterative Impact-Aware Pruned to {target_k} gates ---", flush=True)
    
    noise_ibm = build_hardware_noise_model("IBM", num_qubits)
    tjm_crn_finetune = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=True)
    options_finetune = KrotovOptions(max_iterations=200, batch_step_size=0.1, batch_schedule="constant", seed=800)
    
    print(f"\\n  >>> Evaluating Hardware: IBM Heron (10 qubits) <<<", flush=True)
    res_final = train_krotov_noisy_state_preparation_batch(
        pruned_circuit, statevec, noise_ibm, tjm_crn_finetune, initial_theta=theta_pruned, options=options_finetune
    )
    print(f"  Result -> Impact-Pruned (250 gates) on IBM: {res_final.trace['fidelity'][-1]:.4f}", flush=True)

if __name__ == "__main__":
    scale_test()
