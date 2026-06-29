import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def run_hardware_benchmark():
    num_qubits = 8
    state_seeds = [100, 200, 300, 400, 500]
    hardwares = ["IBM", "Google"]
    
    noise_strength_pre = 0.0005
    processes_pre = [{"name": "pauli_x", "sites": [i], "strength": noise_strength_pre} for i in range(num_qubits)]
    noise_pretrain = NoiseModel(processes_pre)
    tjm_crn_pretrain = KrotovTJMOptions(num_trajectories=1, trajectory_update="cross", use_crn=True)
    tjm_crn_finetune = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=True)
    
    results = []
    
    for state_idx, seed in enumerate(state_seeds):
        print(f"\\n{'='*50}", flush=True)
        print(f"STATE {state_idx+1}/5: Disordered TFIM (Seed={seed})", flush=True)
        print(f"{'='*50}", flush=True)
        
        statevec = get_disordered_tfim_ground_state(num_qubits, seed=seed)
        
        print(f"--- Phase 1: Pre-training Depth-16 Monolith ---", flush=True)
        depths = [4, 8, 12, 16]
        step_sizes = [10.0, 2.0, 0.5, 0.1] # Safer step sizes to avoid divergence after warm-starts
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
            
        # Verify pretraining succeeded
        pretrain_fid = res.trace['fidelity'][-1]
        if pretrain_fid < 0.9:
            print(f"  WARNING: Pre-training fidelity low ({pretrain_fid:.4f}). This will bound the fine-tuned fidelity.", flush=True)
            
        target_k = 150
        pruned_circuit, theta_pruned = prune_circuit_top_k(final_pretrain_circuit, current_theta, keep_k=target_k)
        print(f"--- Phase 2: Pruned to {target_k} gates ---", flush=True)
        
        for hw in hardwares:
            print(f"\\n  >>> Evaluating Hardware: {hw} <<<", flush=True)
            noise_hw = build_hardware_noise_model(hw, num_qubits)
            
            options_finetune = KrotovOptions(
                max_iterations=300, batch_step_size=0.2, # Safer finetuning step size
                batch_schedule="exp", batch_decay=0.01, seed=999
            )
            res_finetune = train_krotov_noisy_state_preparation_batch(
                pruned_circuit, statevec, noise_hw, tjm_crn_finetune, initial_theta=theta_pruned, options=options_finetune
            )
            finetuned_fid = res_finetune.trace["fidelity"][-1]
            
            options_baseline = KrotovOptions(max_iterations=1, batch_step_size=0.0, seed=999)
            res_baseline = train_krotov_noisy_state_preparation_batch(
                final_pretrain_circuit, statevec, noise_hw, tjm_crn_finetune, initial_theta=current_theta, options=options_baseline
            )
            baseline_fid = res_baseline.trace["fidelity"][-1]
            
            print(f"  Result -> Monolith: {baseline_fid:.4f} | Pruned ({target_k}): {finetuned_fid:.4f}", flush=True)
            results.append({
                "State_Seed": seed,
                "Hardware": hw,
                "Baseline_Fidelity": baseline_fid,
                "Pruned_Fidelity": finetuned_fid
            })
            
    df = pd.DataFrame(results)
    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/hardware_benchmark_5states.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(state_seeds))
    width = 0.35
    
    ibm_pruned = df[df["Hardware"] == "IBM"]["Pruned_Fidelity"].values
    google_pruned = df[df["Hardware"] == "Google"]["Pruned_Fidelity"].values
    ibm_baseline = df[df["Hardware"] == "IBM"]["Baseline_Fidelity"].values
    
    plt.bar(x - width/2, ibm_pruned, width, label='IBM Heron (Pruned, K=150)', color='blue')
    plt.bar(x + width/2, google_pruned, width, label='Google Willow (Pruned, K=150)', color='green')
    
    plt.scatter(x - width/2, ibm_baseline, color='red', marker='x', s=100, label='IBM Heron (Unpruned)')
    plt.scatter(x + width/2, df[df["Hardware"] == "Google"]["Baseline_Fidelity"].values, color='orange', marker='x', s=100, label='Google Willow (Unpruned)')
    
    plt.axhline(y=0.80, color='k', linestyle='--', label='80% SOTA Target')
    
    plt.xlabel("Disordered TFIM Target State (Seed)")
    plt.ylabel("Final Fidelity")
    plt.title("Hardware Benchmarks: IBM vs Google across 5 TFIM States")
    plt.xticks(x, [f"Seed={s}" for s in state_seeds])
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig("experiments/results/hardware_benchmark_5states.png", dpi=300)
    print("Saved plot to experiments/results/hardware_benchmark_5states.png", flush=True)

if __name__ == "__main__":
    run_hardware_benchmark()
