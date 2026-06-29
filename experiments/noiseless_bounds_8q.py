import sys
import numpy as np
import time
import pandas as pd
from mqt.yaqs.optimization.parameterized_circuit import create_brickwall_matrix_product_disentangler_parameterized_circuit
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    train_krotov_state_preparation_batch
)

def run_noiseless_bounds():
    print("Evaluating Noiseless Circuit Expressivity for 8 Qubits...", flush=True)
    num_qubits = 8
    max_iterations = 150
    step_size = 5.0
    
    np.random.seed(42)
    statevec = np.random.randn(2**num_qubits) + 1j * np.random.randn(2**num_qubits)
    statevec /= np.linalg.norm(statevec)
    from mqt.yaqs.optimization.krotov import _mps_from_statevector
    target_state = _mps_from_statevector(statevec)
    
    depths = [4, 8, 12, 16]
    results = {}
    
    for d in depths:
        print(f"Testing depth={d}...", flush=True)
        circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, d, initial_single_qubit_layer=True)
        initial_theta = np.random.randn(circuit.num_params) * 0.05
        
        options = KrotovOptions(
            max_iterations=max_iterations, 
            batch_step_size=step_size,
            batch_schedule="exp",
            batch_decay=0.01,
            seed=42
        )
        
        start_t = time.time()
        res = train_krotov_state_preparation_batch(
            circuit, target_state, initial_theta=initial_theta, options=options
        )
        dur = time.time() - start_t
        fidelities = res.trace["fidelity"]
        results[f"depth_{d}"] = fidelities
        print(f"  Depth {d} finished in {dur:.1f}s. Final fidelity: {fidelities[-1]:.6f}", flush=True)

    df = pd.DataFrame(results)
    import os
    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/noiseless_bounds_8q.csv", index_label="Iteration")
    print("Saved noiseless bounds.", flush=True)

if __name__ == "__main__":
    run_noiseless_bounds()
