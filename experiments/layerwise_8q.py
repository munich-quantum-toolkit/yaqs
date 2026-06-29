import sys
import numpy as np
import time
import pandas as pd
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.parameterized_circuit import create_brickwall_matrix_product_disentangler_parameterized_circuit
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_noisy_state_preparation_batch
)

def run_layerwise():
    print("Starting 8-Qubit Noisy Layerwise Training...", flush=True)
    num_qubits = 8
    noise_strength = 0.002
    
    np.random.seed(42)
    statevec = np.random.randn(2**num_qubits) + 1j * np.random.randn(2**num_qubits)
    statevec /= np.linalg.norm(statevec)
    
    processes = [{"name": "pauli_x", "sites": [i], "strength": noise_strength} for i in range(num_qubits)]
    noise_model = NoiseModel(processes)
    tjm_crn = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=True)
    
    depths = [4, 8, 12, 16]
    max_iters_per_depth = 150
    step_sizes = [10.0, 5.0, 2.5, 1.0]
    
    current_theta = None
    all_fidelities = []
    
    for i, d in enumerate(depths):
        print(f"\n--- Training Depth {d} ---", flush=True)
        circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, d, initial_single_qubit_layer=True)
        
        if current_theta is None:
            initial_theta = np.random.randn(circuit.num_params) * 0.05
        else:
            initial_theta = np.zeros(circuit.num_params)
            initial_theta[:len(current_theta)] = current_theta
            # Add a tiny bit of noise to the new identity layers to break symmetry
            initial_theta[len(current_theta):] = np.random.randn(circuit.num_params - len(current_theta)) * 0.001
            
        options = KrotovOptions(
            max_iterations=max_iters_per_depth, 
            batch_step_size=step_sizes[i],
            batch_schedule="exp",
            batch_decay=0.01,
            seed=42
        )
        
        start_t = time.time()
        res = train_krotov_noisy_state_preparation_batch(
            circuit, statevec, noise_model, tjm_crn, initial_theta=initial_theta, options=options
        )
        dur = time.time() - start_t
        fids = res.trace["fidelity"]
        all_fidelities.extend(fids)
        current_theta = res.theta
        print(f"Depth {d} finished in {dur:.1f}s. Final fidelity: {fids[-1]:.6f}", flush=True)

    df = pd.DataFrame({"Layerwise_CRN": all_fidelities})
    import os
    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/layerwise_8q_noisy.csv", index_label="Iteration")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Layerwise_CRN"], label="Layerwise CRN", color="red")
    for i in range(1, len(depths)):
        plt.axvline(x=i * max_iters_per_depth, color='k', linestyle='--', alpha=0.3)
    plt.xlabel("Iteration")
    plt.ylabel("Fidelity")
    plt.title("8-Qubit Krotov Layerwise Noisy Training")
    plt.legend()
    plt.grid(True)
    plt.savefig("experiments/results/layerwise_8q_noisy.png", dpi=300)
    print("Saved layerwise results.", flush=True)

if __name__ == "__main__":
    run_layerwise()
