from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit_aer.noise.errors import PauliError
from qiskit_aer.noise import depolarizing_error
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator
from qiskit_aer import Aer
import numpy as np

def qiskit_dephasing_noise(num_qubits: int, noise_strengths: list) -> QiskitNoiseModel:
    """Create a Qiskit noise model with dephasing noise for single qubits and qubit pairs."""

    noise_model = QiskitNoiseModel()
    single_qubit_strength = noise_strengths[0]
    pair_qubit_strength = noise_strengths[1] if len(noise_strengths) > 1 else single_qubit_strength


    # Single qubit dephasing
    single_qubit_dephasing = PauliError([Pauli('I'), Pauli('Z')], [1-single_qubit_strength, single_qubit_strength])
    # Two qubit ZZ dephasing
    two_qubit_dephasing = PauliError([Pauli('II'), Pauli('ZZ')], [1-pair_qubit_strength, pair_qubit_strength])

    for qubit in range(num_qubits):
        noise_model.add_quantum_error(single_qubit_dephasing, ['id'], [qubit])
    for qubit in range(num_qubits - 1):
        noise_model.add_quantum_error(two_qubit_dephasing, ['cx'], [qubit, qubit + 1])

    return noise_model






