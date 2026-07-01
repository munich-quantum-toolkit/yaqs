import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath("."))

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_noisy_state_preparation_batch
)
from mqt.yaqs.optimization.parameterized_circuit import ParameterizedCircuit
from experiments.find_hero_state import get_disordered_tfim_ground_state, build_hardware_noise_model, iterative_impact_prune
from mqt.yaqs.optimization.parameterized_circuit import create_brickwall_matrix_product_disentangler_parameterized_circuit

def run_trajectory_sweep():
    num_qubits = 6
    num_states = 2
    seeds = [42, 43]
    trajectory_counts = [1, 10, 25, 50, 100]
    
    noise_ibm = build_hardware_noise_model("IBM", num_qubits)
    noise_none = NoiseModel([])
    tjm_noiseless = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    
    # Store results: results[t_idx][state_idx]
    results_fidelities = np.zeros((len(trajectory_counts), num_states))
    
    target_k = 60  # For 6 qubits, target a small gate count
    
    for s_idx, seed in enumerate(seeds):
        print(f"\n==================================================")
        print(f"TESTING STATE {s_idx + 1}/{num_states} (Seed {seed})")
        print(f"==================================================")
        
        statevec = get_disordered_tfim_ground_state(num_qubits, seed=seed)
        
        # --- 1. Noiseless Pre-training ---
        print("  [Step 1] Pre-training noiseless depth-8 circuit...", flush=True)
        circuit_ours = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, 8, initial_single_qubit_layer=True)
        
        rng = np.random.RandomState(seed * 10)
        initial_theta = rng.randn(circuit_ours.num_params) * 0.05
        
        options_pre = KrotovOptions(max_iterations=150, batch_step_size=1.0, batch_schedule="exp", batch_decay=0.01, seed=seed*100)
        res_pre = train_krotov_noisy_state_preparation_batch(
            circuit_ours, statevec, noise_none, tjm_noiseless, initial_theta=initial_theta, options=options_pre
        )
        current_theta = res_pre.theta
        print(f"    Pre-train Fidelity: {res_pre.trace['fidelity'][-1]:.4f}")
        
        # --- 2. Impact Pruning ---
        print(f"  [Step 2] Pruning from {circuit_ours.num_params} to {target_k} gates...", flush=True)
        pruned_circuit, theta_pruned = iterative_impact_prune(circuit_ours, current_theta, statevec, target_k=target_k)
        print(f"    Pruning complete.")
        
        # --- 3. Noisy Fine-Tuning Sweep ---
        print("  [Step 3] Fine-tuning under IBM noise with various trajectory counts...", flush=True)
        for t_idx, num_traj in enumerate(trajectory_counts):
            tjm_crn = KrotovTJMOptions(num_trajectories=num_traj, trajectory_update="cross", use_crn=True)
            
            # Use constant seed so randomness is identical across comparisons where possible
            options_finetune = KrotovOptions(max_iterations=50, batch_step_size=0.1, batch_schedule="constant", seed=seed*1000 + num_traj)
            res_ours = train_krotov_noisy_state_preparation_batch(
                pruned_circuit, statevec, noise_ibm, tjm_crn, initial_theta=theta_pruned, options=options_finetune
            )
            fid = res_ours.trace['fidelity'][-1]
            results_fidelities[t_idx, s_idx] = fid
            print(f"    Trajectories = {num_traj:2d}  ->  Fidelity: {fid:.4f}", flush=True)
            
    # Save the raw data
    np.save("experiments/trajectory_sweep_results.npy", results_fidelities)
    print("\nSweep Complete! Results saved to trajectory_sweep_results.npy")
    
    # Simple summary
    means = np.mean(results_fidelities, axis=1)
    stds = np.std(results_fidelities, axis=1)
    print("\n--- SUMMARY ---")
    for t_idx, num_traj in enumerate(trajectory_counts):
        print(f"Trajectories: {num_traj:2d} | Avg Fidelity: {means[t_idx]:.4f} +/- {stds[t_idx]:.4f}")

if __name__ == "__main__":
    run_trajectory_sweep()
