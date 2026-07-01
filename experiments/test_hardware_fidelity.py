import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_noisy_state_preparation_batch,
    noisy_state_preparation_metrics,
)
from experiments.find_hero_state import get_disordered_tfim_ground_state, build_hardware_noise_model, iterative_impact_prune
from mqt.yaqs.optimization.parameterized_circuit import create_brickwall_matrix_product_disentangler_parameterized_circuit

def run_ml_evaluation():
    num_qubits = 6
    seed = 42
    
    noise_ibm = build_hardware_noise_model("IBM", num_qubits)
    noise_none = NoiseModel([])
    
    print("==================================================")
    print("ML PIPELINE EVALUATION (TRAIN VS TEST FIDELITY)")
    print("==================================================")
    
    statevec = get_disordered_tfim_ground_state(num_qubits, seed=seed)
    
    # --- 1. Noiseless Pre-training ---
    print("\n[Step 1] Pre-training noiseless depth-8 circuit...")
    circuit_ours = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, 8, initial_single_qubit_layer=True)
    rng = np.random.RandomState(seed * 10)
    initial_theta = rng.randn(circuit_ours.num_params) * 0.05
    
    tjm_noiseless = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    options_pre = KrotovOptions(max_iterations=100, batch_step_size=1.0, batch_schedule="exp", batch_decay=0.01, seed=seed*100)
    res_pre = train_krotov_noisy_state_preparation_batch(
        circuit_ours, statevec, noise_none, tjm_noiseless, initial_theta=initial_theta, options=options_pre
    )
    current_theta = res_pre.theta
    
    # --- 2. Impact Pruning ---
    target_k = 60
    print(f"\n[Step 2] Pruning from {circuit_ours.num_params} to {target_k} gates...")
    pruned_circuit, theta_pruned = iterative_impact_prune(circuit_ours, current_theta, statevec, target_k=target_k)
    
    # --- 3. Noisy "Training" (Fine-Tuning) ---
    print("\n[Step 3] Fine-tuning under IBM noise (Training)...")
    num_train_traj = 10
    tjm_crn = KrotovTJMOptions(num_trajectories=num_train_traj, trajectory_update="cross", use_crn=True)
    
    # Fast fine-tuning for demonstration
    options_finetune = KrotovOptions(max_iterations=30, batch_step_size=0.1, batch_schedule="constant", seed=seed*1000)
    res_ours = train_krotov_noisy_state_preparation_batch(
        pruned_circuit, statevec, noise_ibm, tjm_crn, initial_theta=theta_pruned, options=options_finetune
    )
    
    theta_opt = res_ours.theta
    training_fidelity = res_ours.trace['fidelity'][-1]
    print(f"-> Reported Training Fidelity (with CRN, N={num_train_traj}): {training_fidelity:.4f}")
    
    # --- 4. Noisy "Testing" (True Hardware Fidelity) ---
    print("\n[Step 4] Evaluating True Hardware Fidelity (Testing)...")
    num_test_traj = 1000
    tjm_test = KrotovTJMOptions(num_trajectories=num_test_traj, use_crn=False) # Unseeded independent trajectories!
    
    _, test_fidelity, _ = noisy_state_preparation_metrics(
        pruned_circuit, theta_opt, statevec, noise_ibm, tjm_test
    )
    print(f"-> Evaluated Test Fidelity (Unseeded, N={num_test_traj}): {test_fidelity:.4f}")
    
    # --- 5. Noiseless Evaluation (Coherent Bound) ---
    print("\n[Step 5] Evaluating Noiseless Fidelity (Coherent Bound)...")
    _, noiseless_fidelity, _ = noisy_state_preparation_metrics(
        pruned_circuit, theta_opt, statevec, noise_none, tjm_noiseless
    )
    print(f"-> Evaluated Noiseless Fidelity (Algorithmic Limit): {noiseless_fidelity:.4f}")
    
    print("\n==================================================")
    print("PIPELINE SUMMARY:")
    print(f" Algorithmic Coherent Fidelity: {noiseless_fidelity:.4f}")
    print(f" True Hardware Test Fidelity  : {test_fidelity:.4f}")
    print(f" Hardware Error Gap           : {(noiseless_fidelity - test_fidelity):.4f}")
    print("==================================================")

if __name__ == "__main__":
    run_ml_evaluation()
