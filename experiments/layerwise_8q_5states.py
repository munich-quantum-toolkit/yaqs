import sys
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.parameterized_circuit import create_brickwall_matrix_product_disentangler_parameterized_circuit
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_noisy_state_preparation_batch
)

def run_5_states():
    print("Starting 8-Qubit Noisy Layerwise Training for 5 Random States...", flush=True)
    num_qubits = 8
    noise_strength = 0.0005
    
    seeds = [42, 101, 2024, 777, 999]
    depths = [4, 8, 12, 16, 20, 24]
    max_iters_per_depth = 250
    step_sizes = [10.0, 5.0, 2.5, 1.25, 0.6, 0.3]
    
    processes = [{"name": "pauli_x", "sites": [i], "strength": noise_strength} for i in range(num_qubits)]
    noise_model = NoiseModel(processes)
    tjm_crn = KrotovTJMOptions(num_trajectories=1, trajectory_update="cross", use_crn=True)
    
    results = {}
    
    for seed in seeds:
        print(f"\n======================================", flush=True)
        print(f"=== Optimizing State (Seed: {seed}) ===", flush=True)
        print(f"======================================", flush=True)
        
        np.random.seed(seed)
        statevec = np.random.randn(2**num_qubits) + 1j * np.random.randn(2**num_qubits)
        statevec /= np.linalg.norm(statevec)
        
        current_theta = None
        state_fidelities = []
        
        for i, d in enumerate(depths):
            print(f"\n--- Training Depth {d} (Step Size: {step_sizes[i]}) ---", flush=True)
            circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, d, initial_single_qubit_layer=True)
            
            if current_theta is None:
                # Use a fixed inner seed for reproducible circuit initialization
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
                circuit, statevec, noise_model, tjm_crn, initial_theta=initial_theta, options=options
            )
            dur = time.time() - start_t
            fids = res.trace["fidelity"]
            state_fidelities.extend(fids)
            current_theta = res.theta
            print(f"Depth {d} finished in {dur:.1f}s. Final fidelity: {fids[-1]:.6f}", flush=True)
            
        results[f"State_{seed}"] = state_fidelities
        
        # Save incrementally
        df_temp = pd.DataFrame(results)
        os.makedirs("experiments/results", exist_ok=True)
        df_temp.to_csv("experiments/results/layerwise_8q_5states.csv", index_label="Iteration")
        
        plt.figure(figsize=(12, 7))
        for s in results.keys():
            plt.plot(df_temp.index, df_temp[s], label=s)
            
        for i in range(1, len(depths)):
            plt.axvline(x=i * max_iters_per_depth, color='k', linestyle='--', alpha=0.3)
            
        plt.axhline(y=0.85, color='r', linestyle='-', linewidth=2, label='85% Goal')
            
        plt.xlabel("Iteration")
        plt.ylabel("Fidelity")
        plt.title("8-Qubit Krotov Layerwise Noisy Training (5 Random States)")
        plt.legend()
        plt.grid(True)
        plt.savefig("experiments/results/layerwise_8q_5states.png", dpi=300)
        plt.close()
        print(f"Incrementally saved results up to Seed {seed}.", flush=True)

    print("Fully completed 5 states.", flush=True)

if __name__ == "__main__":
    run_5_states()
