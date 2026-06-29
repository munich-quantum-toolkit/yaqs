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
    train_krotov_noisy_state_preparation_batch
)

def run_deep():
    print("Starting 8-Qubit CRN Deep Convergence Run...", flush=True)
    num_qubits = 8
    depth = 4
    max_iterations = 300
    num_trajectories = 3
    noise_strength = 0.02
    step_size = 10.0
    
    np.random.seed(42)
    statevec = np.random.randn(2**num_qubits) + 1j * np.random.randn(2**num_qubits)
    statevec /= np.linalg.norm(statevec)
    
    print("Creating circuit...", flush=True)
    circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, depth, initial_single_qubit_layer=True)
    initial_theta = np.random.randn(circuit.num_params) * 0.05
    
    processes = [{"name": "pauli_x", "sites": [i], "strength": noise_strength} for i in range(num_qubits)]
    noise_model = NoiseModel(processes)
    tjm_crn = KrotovTJMOptions(num_trajectories=num_trajectories, trajectory_update="cross", use_crn=True)

    options = KrotovOptions(
        max_iterations=500, 
        batch_step_size=step_size, 
        batch_schedule="exp",
        batch_decay=0.01,
        seed=42
    )
    
    start_t = time.time()
    res = train_krotov_noisy_state_preparation_batch(
        circuit, statevec, noise_model, tjm_crn, initial_theta=initial_theta, options=options
    )
    dur = time.time() - start_t
    
    fidelities = res.trace["fidelity"]
    print(f"Deep run finished in {dur:.1f}s. Final fidelity: {fidelities[-1]:.6f}", flush=True)

    df = pd.DataFrame({"Noisy_Cross_CRN_Deep": fidelities})
    import os
    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/scale_up_8q_deep.csv", index_label="Iteration")
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Noisy_Cross_CRN_Deep"], label=f"CRN Batch (step={step_size})", color="purple", marker='.')
    plt.xlabel("Iteration")
    plt.ylabel("Fidelity")
    plt.title("8-Qubit Krotov CRN Deep Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("experiments/results/scale_up_8q_deep.png", dpi=300)
    print("Saved deep run results.", flush=True)

if __name__ == "__main__":
    run_deep()
