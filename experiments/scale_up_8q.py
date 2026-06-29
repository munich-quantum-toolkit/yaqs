import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.parameterized_circuit import create_brickwall_matrix_product_disentangler_parameterized_circuit
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_noisy_state_preparation_batch
)

def run_scale_up():
    print("Starting 8-Qubit Scale-Up with CRN Stabilization...", flush=True)
    np.random.seed(42)
    num_qubits = 8
    depth = 4
    max_iterations = 50
    num_trajectories = 10
    noise_strength = 0.02
    step_size = 0.1
    
    # 1. Random Gaussian State (Dense 256-dim complex vector)
    print("Generating random target state...", flush=True)
    statevec = np.random.randn(2**num_qubits) + 1j * np.random.randn(2**num_qubits)
    statevec /= np.linalg.norm(statevec)
    
    # 2. Circuit Ansatz
    print(f"Creating BMPD parameterized circuit (qubits={num_qubits}, depth={depth})...", flush=True)
    circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, depth, initial_single_qubit_layer=True)
    initial_theta = np.random.randn(circuit.num_params) * 0.05
    
    # 3. Noise Model
    processes = [{"name": "pauli_x", "sites": [i], "strength": noise_strength} for i in range(num_qubits)]
    noise_model = NoiseModel(processes)
    
    # 4. Krotov Configurations
    options = KrotovOptions(max_iterations=max_iterations, batch_step_size=step_size, seed=42)
    tjm_crn = KrotovTJMOptions(num_trajectories=num_trajectories, trajectory_update="cross", use_crn=True)
    
    # 5. Training
    print("Training Krotov Noisy State Preparation (Cross + CRN)...", flush=True)
    res_crn = train_krotov_noisy_state_preparation_batch(circuit, statevec, noise_model, tjm_crn, initial_theta=initial_theta, options=options)
    
    # 6. Results & Plotting
    print("Training finished. Saving results...", flush=True)
    import os
    os.makedirs("experiments/results", exist_ok=True)
    
    fidelities = res_crn.trace["fidelity"]
    df = pd.DataFrame({"Noisy_Cross_CRN": fidelities})
    df.to_csv("experiments/results/scale_up_8q.csv", index_label="Iteration")
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Noisy_Cross_CRN"], label="Noisy Cross CRN (8 qubits)", marker='o', color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Fidelity")
    plt.title("8-Qubit Krotov State Preparation with CRN Stabilization")
    plt.legend()
    plt.grid(True)
    plt.savefig("experiments/results/scale_up_8q.png", dpi=300)
    print("Plot saved to experiments/results/scale_up_8q.png", flush=True)
    print("Final Fidelity:", fidelities[-1], flush=True)

if __name__ == "__main__":
    run_scale_up()
