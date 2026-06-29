import os
import sys
import time
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

def evaluate_fidelity(circuit, theta, statevec):
    noise_none = NoiseModel([])
    tjm_none = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    options = KrotovOptions(max_iterations=0)
    res = train_krotov_noisy_state_preparation_batch(
        circuit, statevec, noise_none, tjm_none, initial_theta=theta, options=options
    )
    return res.trace['fidelity'][-1]

def impact_aware_prune(circuit: ParameterizedCircuit, theta: np.ndarray, statevec: np.ndarray, keep_k: int):
    print("Computing gradients via Parameter Shift Rule...", flush=True)
    start = time.time()
    gradients = np.zeros(circuit.num_params)
    
    for gate in circuit.gates:
        if gate.is_trainable:
            i = gate.param_index
            if gradients[i] != 0: continue # Already computed
            
            theta_plus = theta.copy()
            theta_plus[i] += np.pi / 2
            fid_plus = evaluate_fidelity(circuit, theta_plus, statevec)
            
            theta_minus = theta.copy()
            theta_minus[i] -= np.pi / 2
            fid_minus = evaluate_fidelity(circuit, theta_minus, statevec)
            
            gradients[i] = 0.5 * (fid_plus - fid_minus)
    print(f"Computed {circuit.num_params} gradients in {time.time()-start:.1f}s", flush=True)

    impacts = []
    for gate in circuit.gates:
        if gate.is_trainable:
            i = gate.param_index
            impact = abs(theta[i] * gradients[i])
            impacts.append((impact, i))
            
    impacts.sort(reverse=True, key=lambda x: x[0])
    keep_indices = set([x[1] for x in impacts[:keep_k]])

    new_gates = []
    new_theta = []
    old_to_new_index = {}
    for gate in circuit.gates:
        if gate.is_trainable:
            if gate.param_index not in keep_indices:
                continue
            if gate.param_index not in old_to_new_index:
                old_to_new_index[gate.param_index] = len(new_theta)
                new_theta.append(theta[gate.param_index])
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
        drop_count = min(200, current_circuit.num_params - target_k)
        next_k = current_circuit.num_params - drop_count
        print(f"\\n--- Iterative Prune: {current_circuit.num_params} -> {next_k} ---", flush=True)
        
        pruned_circuit, new_theta = impact_aware_prune(current_circuit, current_theta, statevec, keep_k=next_k)
        
        # Relax the circuit with a short noiseless finetuning
        if next_k > target_k:
            print("Relaxing circuit...", flush=True)
            options = KrotovOptions(max_iterations=50, batch_step_size=0.1, batch_schedule="constant", seed=42)
            res = train_krotov_noisy_state_preparation_batch(
                pruned_circuit, statevec, noise_none, tjm_none, initial_theta=new_theta, options=options
            )
            current_theta = res.theta
            print(f"Relaxed fidelity: {res.trace['fidelity'][-1]:.4f}", flush=True)
        else:
            current_theta = new_theta
            
        current_circuit = pruned_circuit
        
    return current_circuit, current_theta

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
    elif hardware == "Google":
        p_1q = 6e-4 / 3.0
        p_2q = 5e-3 / 3.0
        for i in range(num_qubits):
            for pauli in ["pauli_x", "pauli_y", "pauli_z"]:
                processes.append({"name": pauli, "sites": [i], "strength": p_1q})
        for i in range(num_qubits - 1):
            for crosstalk in ["crosstalk_xx", "crosstalk_yy", "crosstalk_zz"]:
                processes.append({"name": crosstalk, "sites": [i, i+1], "strength": p_2q})
    return NoiseModel(processes)

def run_impact_benchmark():
    num_qubits = 8
    seed = 400 # The seed that failed catastrophically with magnitude pruning
    hardwares = ["IBM", "Google"]
    
    noise_strength_pre = 0.0005
    processes_pre = [{"name": "pauli_x", "sites": [i], "strength": noise_strength_pre} for i in range(num_qubits)]
    noise_pretrain = NoiseModel(processes_pre)
    tjm_crn_pretrain = KrotovTJMOptions(num_trajectories=1, trajectory_update="cross", use_crn=True)
    tjm_crn_finetune = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=True)
    
    print(f"\\n{'='*50}", flush=True)
    print(f"STATE 4/5 RETEST: Disordered TFIM (Seed={seed}) [Iterative Impact Pruning]", flush=True)
    print(f"{'='*50}", flush=True)
    
    statevec = get_disordered_tfim_ground_state(num_qubits, seed=seed)
    
    print(f"--- Phase 1: Pre-training Depth-16 Monolith ---", flush=True)
    depths = [4, 8, 12, 16]
    step_sizes = [10.0, 2.0, 0.5, 0.1]
    max_iters_per_depth = 250
    
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
        print(f"  Depth {d} final fidelity: {res.trace['fidelity'][-1]:.4f}", flush=True)
        
    target_k = 150
    pruned_circuit, theta_pruned = iterative_impact_prune(final_pretrain_circuit, current_theta, statevec, target_k)
    print(f"--- Phase 2: Iterative Impact-Aware Pruned to {target_k} gates ---", flush=True)
    
    for hw in hardwares:
        print(f"\\n  >>> Evaluating Hardware: {hw} <<<", flush=True)
        noise_hw = build_hardware_noise_model(hw, num_qubits)
        
        options_finetune = KrotovOptions(
            max_iterations=300, batch_step_size=0.2,
            batch_schedule="exp", batch_decay=0.01, seed=999
        )
        res_finetune = train_krotov_noisy_state_preparation_batch(
            pruned_circuit, statevec, noise_hw, tjm_crn_finetune, initial_theta=theta_pruned, options=options_finetune
        )
        finetuned_fid = res_finetune.trace["fidelity"][-1]
        
        print(f"  Result -> Impact-Pruned ({target_k}): {finetuned_fid:.4f}", flush=True)
        
if __name__ == "__main__":
    run_impact_benchmark()
