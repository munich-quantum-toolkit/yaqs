from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator
from qiskit_aer import Aer
import numpy as np

def simulate_noisy_layers_with_estimator(base_circuit: QuantumCircuit,
                                         noise_model: NoiseModel,
                                         num_layers: int) -> np.ndarray:
    """Simulates a noisy quantum circuit using EstimatorV2 and returns Z expectations."""
    
    num_qubits = base_circuit.num_qubits
    z_expectations = np.zeros((num_layers, num_qubits))

    # Setup simulator and estimator
    backend = Aer.get_backend("aer_simulator")
    estimator = Estimator(backend_options={"noise_model": noise_model})

    cumulative_circuit = QuantumCircuit(num_qubits)

    for layer in range(num_layers):
        # Compose another copy of the circuit
        cumulative_circuit = cumulative_circuit.compose(base_circuit)

        # For each qubit, define <Z_i>
        for qubit in range(num_qubits):
            observable = SparsePauliOp(f'{"I"*qubit}Z{"I"*(num_qubits - qubit - 1)}')
            result = estimator.run([cumulative_circuit], [observable], shots = None).result()
            z_expectations[layer, qubit] = result.values[0]

    return z_expectations

def simulate_noisy_layers_with_estimator(base_circuit: QuantumCircuit,
                                         noise_model: NoiseModel,
                                         num_layers: int) -> np.ndarray:
    """Simulates a noisy quantum circuit using EstimatorV2 and returns Z expectations."""
    
    num_qubits = base_circuit.num_qubits
    z_expectations = np.zeros((num_layers, num_qubits))

    # Setup simulator and estimator
    backend = Aer.get_backend("aer_simulator")
    estimator = Estimator(backend_options={"noise_model": noise_model})

    cumulative_circuit = QuantumCircuit(num_qubits)

    for layer in range(num_layers):
        # Compose another copy of the circuit
        cumulative_circuit = cumulative_circuit.compose(base_circuit)

        # For each qubit, define <Z_i>
        for qubit in range(num_qubits):
            observable = SparsePauliOp(f'{"I"*qubit}Z{"I"*(num_qubits - qubit - 1)}')
            result = estimator.run([cumulative_circuit], [observable], shots = None).result()
            z_expectations[layer, qubit] = result.values[0]

    return z_expectations

if __name__ == "__main__":


    # Create a simple quantum circuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Noise model with depolarizing noise
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['id', 'rz', 'sx', 'x'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ['cx'])



    z_vals = simulate_noisy_layers_with_estimator(qc, noise_model, num_layers=5)
    print("Z expectations per layer:\n", z_vals)




