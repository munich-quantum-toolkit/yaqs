import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def get_tfim_ground_state(num_qubits: int, g: float = 1.0):
    dim = 2**num_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    
    # ZZ interactions
    for i in range(num_qubits - 1):
        op = [I]*num_qubits
        op[i] = Z
        op[i+1] = Z
        term = op[0]
        for j in range(1, num_qubits):
            term = np.kron(term, op[j])
        H -= term
        
    # X field
    for i in range(num_qubits):
        op = [I]*num_qubits
        op[i] = X
        term = op[0]
        for j in range(1, num_qubits):
            term = np.kron(term, op[j])
        H -= g * term
        
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvectors[:, 0]

def prune_circuit_top_k(circuit: ParameterizedCircuit, theta: np.ndarray, keep_k: int):
    gate_angles = []
    for gate in circuit.gates:
        if gate.is_trainable:
            angle = circuit.angle(gate, theta, None)
            angle_mod = angle % (2 * np.pi)
            if angle_mod < 0:
                angle_mod += 2 * np.pi
            dist_to_0 = min(angle_mod, 2 * np.pi - angle_mod)
            gate_angles.append(dist_to_0)
            
    gate_angles.sort(reverse=True)
    if keep_k < len(gate_angles):
        threshold = gate_angles[keep_k - 1]
    else:
        threshold = 0.0

    new_gates = []
    new_theta = []
    old_to_new_index = {}
    
    for gate in circuit.gates:
        if gate.is_trainable:
            angle = circuit.angle(gate, theta, None)
            angle_mod = angle % (2 * np.pi)
            if angle_mod < 0:
                angle_mod += 2 * np.pi
            dist_to_0 = min(angle_mod, 2 * np.pi - angle_mod)
            
            if dist_to_0 < threshold:
                continue # Prune this gate
            
            if gate.param_index not in old_to_new_index:
                old_to_new_index[gate.param_index] = len(new_theta)
                new_theta.append(theta[gate.param_index])
                
            new_gate = ParameterizedGate(
                name=gate.name,
                sites=gate.sites,
                param_index=old_to_new_index[gate.param_index],
                angle_scale=gate.angle_scale,
                angle_offset=gate.angle_offset,
                data_map=gate.data_map,
                fixed_params=gate.fixed_params
            )
            new_gates.append(new_gate)
        else:
            new_gates.append(gate)
            
    pruned_circuit = ParameterizedCircuit(
        num_qubits=circuit.num_qubits, 
        gates=new_gates, 
        num_params=len(new_theta)
    )
    return pruned_circuit, np.array(new_theta, dtype=np.float64)

def run_adaptive_pruning():
    print("=== Aggressive Top-K Pruning Sweep for TFIM Ground State ===", flush=True)
    num_qubits = 8
    
    # 1. Generate Structured Target State (TFIM Ground State)
    statevec = get_tfim_ground_state(num_qubits, g=1.0)
    
    # 2. Base Noise Model (0.05% for pre-training)
    noise_strength = 0.0005
    processes_pretrain = [{"name": "pauli_x", "sites": [i], "strength": noise_strength} for i in range(num_qubits)]
    noise_pretrain = NoiseModel(processes_pretrain)
    tjm_crn_pretrain = KrotovTJMOptions(num_trajectories=1, trajectory_update="cross", use_crn=True)
    
    print(f"\\n=== Phase 1: Deep Layerwise Pre-Training ===", flush=True)
    depths = [4, 8, 12, 16] 
    max_iters_per_depth = 250
    step_sizes = [10.0, 5.0, 2.5, 1.25]
    seed = 42
    
    current_theta = None
    pretrain_fidelities = []
    final_pretrain_circuit = None
    
    for i, d in enumerate(depths):
        print(f"--- Training Depth {d} (Step Size: {step_sizes[i]}) ---", flush=True)
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
            max_iterations=max_iters_per_depth, 
            batch_step_size=step_sizes[i],
            batch_schedule="exp",
            batch_decay=0.01,
            seed=seed + 3000
        )
        
        start_t = time.time()
        res = train_krotov_noisy_state_preparation_batch(
            circuit, statevec, noise_pretrain, tjm_crn_pretrain, initial_theta=initial_theta, options=options
        )
        dur = time.time() - start_t
        fids = res.trace["fidelity"]
        pretrain_fidelities.extend(fids)
        current_theta = res.theta
        print(f"Depth {d} finished in {dur:.1f}s. Final fidelity: {fids[-1]:.6f}", flush=True)

    print(f"\\n=== Phase 2 & 3: Sweeping Top-K Pruning ===", flush=True)
    original_gate_count = len(final_pretrain_circuit.gates)
    print(f"Original Gates: {original_gate_count}", flush=True)
    
    k_values = [20, 30, 40, 50, 60, 80, 100, 150]
    final_fidelities = []
    noiseless_fidelities = []
    
    noise_realistic_strength = 0.005
    processes_realistic = [{"name": "pauli_x", "sites": [i], "strength": noise_realistic_strength} for i in range(num_qubits)]
    noise_realistic = NoiseModel(processes_realistic)
    tjm_crn_realistic = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=True)
    
    # Empty noise model for noiseless limit check
    noise_none = NoiseModel([])
    tjm_crn_none = KrotovTJMOptions(num_trajectories=1, trajectory_update="cross", use_crn=False)
    
    options_finetune = KrotovOptions(
        max_iterations=400, 
        batch_step_size=0.5,
        batch_schedule="exp",
        batch_decay=0.01,
        seed=999
    )
    
    for target_k in k_values:
        print(f"\\n--- Evaluating K = {target_k} ---", flush=True)
        pruned_circuit, theta_pruned = prune_circuit_top_k(final_pretrain_circuit, current_theta, keep_k=target_k)
        
        res_finetune = train_krotov_noisy_state_preparation_batch(
            pruned_circuit, statevec, noise_realistic, tjm_crn_realistic, initial_theta=theta_pruned, options=options_finetune
        )
        
        fid = res_finetune.trace["fidelity"][-1]
        final_fidelities.append(fid)
        
        # Check noiseless limit of the fine-tuned parameters
        res_noiseless = train_krotov_noisy_state_preparation_batch(
            pruned_circuit, statevec, noise_none, tjm_crn_none, initial_theta=res_finetune.theta, 
            options=KrotovOptions(max_iterations=1, batch_step_size=0.0, seed=999)
        )
        n_fid = res_noiseless.trace["fidelity"][-1]
        noiseless_fidelities.append(n_fid)
        
        print(f"K={target_k} -> Noisy Fidelity: {fid:.6f}, Noiseless Limit: {n_fid:.6f}", flush=True)
        
    print(f"\\n=== Baseline Check: Unpruned Monolith under 0.5% Noise ===", flush=True)
    options_baseline = KrotovOptions(max_iterations=1, batch_step_size=0.0, seed=999)
    res_baseline = train_krotov_noisy_state_preparation_batch(
        final_pretrain_circuit, statevec, noise_realistic, tjm_crn_realistic, initial_theta=current_theta, options=options_baseline
    )
    baseline_fidelity = res_baseline.trace["fidelity"][0]
    print(f"Monolith Fidelity at 0.5% noise: {baseline_fidelity:.6f}", flush=True)
    
    # Save results
    os.makedirs("experiments/results", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    plt.plot(k_values, final_fidelities, marker='o', label="Pruned & Fine-Tuned (0.5% Noise)", color="green")
    plt.plot(k_values, noiseless_fidelities, marker='s', linestyle=':', label="Noiseless Expressivity Limit", color="blue")
    plt.axhline(y=baseline_fidelity, color='red', linestyle='--', label=f"Unpruned Monolith ({original_gate_count} gates)")
    plt.axhline(y=0.80, color='k', linestyle='-', label="80% SOTA Goal")
    
    plt.xlabel("Number of Kept Gates (K)")
    plt.ylabel("Fidelity")
    plt.title(f"Adaptive Top-K Pruning Trade-off for TFIM Ground State")
    plt.legend()
    plt.grid(True)
    plt.savefig("experiments/results/pruning_tradeoff_tfim.png", dpi=300)
    print("Saved plot to experiments/results/pruning_tradeoff_tfim.png", flush=True)

if __name__ == "__main__":
    run_adaptive_pruning()
