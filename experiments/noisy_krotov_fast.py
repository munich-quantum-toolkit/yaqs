import sys
import copy
import numpy as np

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.state import MPS
from mqt.yaqs.optimization.parameterized_circuit import create_brickwall_matrix_product_disentangler_parameterized_circuit
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_state_preparation_batch,
    train_krotov_noisy_state_preparation_batch
)

np.random.seed(42)
num_qubits = 2
depth = 1

statevec = np.random.randn(2**num_qubits) + 1j * np.random.randn(2**num_qubits)
statevec /= np.linalg.norm(statevec)

from mqt.yaqs.optimization.krotov import _mps_from_statevector
target_state = _mps_from_statevector(statevec)

circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, depth, initial_single_qubit_layer=True)
initial_theta = np.random.randn(circuit.num_params) * 0.1

print(f"Testing noiseless Krotov batch on {num_qubits} qubits, depth {depth}...")
sys.stdout.flush()

options = KrotovOptions(max_iterations=5, batch_step_size=0.1, seed=42)
res = train_krotov_state_preparation_batch(circuit, target_state, initial_theta=initial_theta, options=options)
print(f"Noiseless final fidelity: {res.trace['fidelity'][-1]}")
sys.stdout.flush()

print(f"Testing noisy Krotov batch...")
sys.stdout.flush()
processes = [{"name": "pauli_x", "sites": [i], "strength": 0.05} for i in range(num_qubits)]
noise_model = NoiseModel(processes)
tjm_options = KrotovTJMOptions(num_trajectories=5, trajectory_update="cross", use_crn=True)
res_noisy = train_krotov_noisy_state_preparation_batch(circuit, target_state, noise_model, tjm_options, initial_theta=initial_theta, options=options)
print(f"Noiseless trace: {res.trace['fidelity']}")
print(f"Noisy cross trace: {res_noisy.trace['fidelity']}")
sys.stdout.flush()
