import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.parameterized_circuit import create_brickwall_matrix_product_disentangler_parameterized_circuit
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_state_preparation_batch,
    train_krotov_noisy_state_preparation_batch
)

def run_all():
    np.random.seed(42)
    num_qubits = 3
    depth = 1
    max_iterations = 20
    num_trajectories = 5
    noise_strength = 0.05
    step_size = 0.1
    
    statevec = np.random.randn(2**num_qubits) + 1j * np.random.randn(2**num_qubits)
    statevec /= np.linalg.norm(statevec)
    from mqt.yaqs.optimization.krotov import _mps_from_statevector
    target_state = _mps_from_statevector(statevec)

    circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, depth, initial_single_qubit_layer=True)
    initial_theta = np.random.randn(circuit.num_params) * 0.1
    options = KrotovOptions(max_iterations=max_iterations, batch_step_size=step_size, seed=42)
    
    results = {}
    
    print("Running Noiseless...")
    res = train_krotov_state_preparation_batch(circuit, target_state, initial_theta=initial_theta, options=options)
    results["Noiseless"] = res.trace["fidelity"]
    
    processes = [{"name": "pauli_x", "sites": [i], "strength": noise_strength} for i in range(num_qubits)]
    noise_model = NoiseModel(processes)
    
    print("Running Noisy Independent (No CRN)...")
    tjm_indep = KrotovTJMOptions(num_trajectories=num_trajectories, trajectory_update="independent", use_crn=False)
    res_indep = train_krotov_noisy_state_preparation_batch(circuit, target_state, noise_model, tjm_indep, initial_theta=initial_theta, options=options)
    results["Noisy_Independent"] = res_indep.trace["fidelity"]
    
    print("Running Noisy Cross (No CRN)...")
    tjm_cross = KrotovTJMOptions(num_trajectories=num_trajectories, trajectory_update="cross", use_crn=False)
    res_cross = train_krotov_noisy_state_preparation_batch(circuit, target_state, noise_model, tjm_cross, initial_theta=initial_theta, options=options)
    results["Noisy_Cross"] = res_cross.trace["fidelity"]
    
    print("Running Noisy Cross (With CRN)...")
    tjm_crn = KrotovTJMOptions(num_trajectories=num_trajectories, trajectory_update="cross", use_crn=True)
    res_crn = train_krotov_noisy_state_preparation_batch(circuit, target_state, noise_model, tjm_crn, initial_theta=initial_theta, options=options)
    results["Noisy_Cross_CRN"] = res_crn.trace["fidelity"]
    
    df = pd.DataFrame(results)
    import os
    os.makedirs("experiments", exist_ok=True)
    df.to_csv("experiments/crn_comparison.csv", index_label="Iteration")
    
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Fidelity")
    plt.title("Krotov State Preparation with CRN Stabilization")
    plt.legend()
    plt.grid(True)
    plt.savefig("experiments/crn_comparison.png", dpi=300)
    print("Plot saved to experiments/crn_comparison.png")

if __name__ == "__main__":
    run_all()
