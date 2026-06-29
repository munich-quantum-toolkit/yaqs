import sys
import numpy as np
import time
import pandas as pd
from mqt.yaqs.optimization.parameterized_circuit import create_brickwall_matrix_product_disentangler_parameterized_circuit
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    train_krotov_state_preparation_batch
)

def run_layerwise_noiseless():
    print("Starting 8-Qubit Noiseless Layerwise Training...", flush=True)
    num_qubits = 8
    
    np.random.seed(42)
    statevec = np.random.randn(2**num_qubits) + 1j * np.random.randn(2**num_qubits)
    statevec /= np.linalg.norm(statevec)
    from mqt.yaqs.optimization.krotov import _mps_from_statevector
    target_state = _mps_from_statevector(statevec)
    
    depths = [4, 8, 12, 16]
    max_iters_per_depth = 150
    
    current_theta = None
    all_fidelities = []
    
    for d in depths:
        print(f"\n--- Training Depth {d} ---", flush=True)
        circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, d, initial_single_qubit_layer=True)
        
        if current_theta is None:
            initial_theta = np.random.randn(circuit.num_params) * 0.05
        else:
            initial_theta = np.zeros(circuit.num_params)
            initial_theta[:len(current_theta)] = current_theta
            # Add a tiny bit of noise to break symmetry
            initial_theta[len(current_theta):] = np.random.randn(circuit.num_params - len(current_theta)) * 0.001
            
        options = KrotovOptions(
            max_iterations=max_iters_per_depth, 
            batch_step_size=10.0,
            batch_schedule="exp",
            batch_decay=0.01,
            seed=42
        )
        
        start_t = time.time()
        res = train_krotov_state_preparation_batch(
            circuit, target_state, initial_theta=initial_theta, options=options
        )
        dur = time.time() - start_t
        fids = res.trace["fidelity"]
        all_fidelities.extend(fids)
        current_theta = res.theta
        print(f"Depth {d} finished in {dur:.1f}s. Final fidelity: {fids[-1]:.6f}", flush=True)

    df = pd.DataFrame({"Layerwise_Noiseless": all_fidelities})
    import os
    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/layerwise_8q_noiseless.csv", index_label="Iteration")
    print("Saved layerwise noiseless results.", flush=True)

if __name__ == "__main__":
    run_layerwise_noiseless()
