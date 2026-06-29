import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.parameterized_circuit import create_brickwall_matrix_product_disentangler_parameterized_circuit
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_noisy_state_preparation_batch,
    train_krotov_noisy_state_preparation_hybrid
)

def run_search():
    print("Starting 8-Qubit CRN Grid Search...", flush=True)
    num_qubits = 8
    depth = 4
    max_iterations = 30
    num_trajectories = 3
    noise_strength = 0.02
    
    np.random.seed(42)
    statevec = np.random.randn(2**num_qubits) + 1j * np.random.randn(2**num_qubits)
    statevec /= np.linalg.norm(statevec)
    
    print("Creating circuit...", flush=True)
    circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, depth, initial_single_qubit_layer=True)
    initial_theta = np.random.randn(circuit.num_params) * 0.05
    
    processes = [{"name": "pauli_x", "sites": [i], "strength": noise_strength} for i in range(num_qubits)]
    noise_model = NoiseModel(processes)
    tjm_crn = KrotovTJMOptions(num_trajectories=num_trajectories, trajectory_update="cross", use_crn=True)

    # Grid search config
    step_sizes = [5.0, 10.0]
    modes = ["batch", "hybrid"]
    
    results = {}
    
    for mode in modes:
        for step in step_sizes:
            name = f"{mode}_step{step}"
            print(f"Running {name}...", flush=True)
            
            options = KrotovOptions(
                max_iterations=max_iterations, 
                batch_step_size=step, 
                online_step_size=step, 
                seed=42
            )
            
            start_t = time.time()
            if mode == "batch":
                res = train_krotov_noisy_state_preparation_batch(
                    circuit, statevec, noise_model, tjm_crn, initial_theta=initial_theta, options=options
                )
            else:
                options.switch_iteration = max_iterations // 2
                res = train_krotov_noisy_state_preparation_hybrid(
                    circuit, statevec, noise_model, tjm_crn, initial_theta=initial_theta, options=options
                )
            dur = time.time() - start_t
            
            fidelities = res.trace["fidelity"]
            results[name] = fidelities
            print(f"  {name} finished in {dur:.1f}s. Final fidelity: {fidelities[-1]:.6f}", flush=True)

    df = pd.DataFrame(results)
    import os
    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/scale_up_8q_search.csv", index_label="Iteration")
    
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, marker='.')
    plt.xlabel("Iteration")
    plt.ylabel("Fidelity")
    plt.title("8-Qubit Krotov Grid Search (CRN)")
    plt.legend()
    plt.grid(True)
    plt.savefig("experiments/results/scale_up_8q_search.png", dpi=300)
    print("Saved grid search results.", flush=True)

if __name__ == "__main__":
    run_search()
