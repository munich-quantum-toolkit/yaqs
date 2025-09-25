from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.backends.aer_simulator import AerSimulator
from qiskit_aer import Aer
import numpy as np

def qiskit_noisy_simulator_stepwise(circuit, noise_model, num_qubits, method="automatic"):
    """
    Helper function to get Z expectations using Qiskit's Aer simulator.
    """
    observables = []
    for i in range(num_qubits):
        pauli_str = "I"*i + "Z" + "I"*(num_qubits - i - 1)
        observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
        observables.append(observable)

    z_expectations = []
    qc_copy = circuit.copy()

    # print('circut:')
    # print(qc_copy.draw())
    # exact_estimator = Estimator()
    noisy_estimator = Estimator(options=dict(backend_options=dict(noise_model=noise_model, method=method)))
    pub = (qc_copy, observables)
    job = noisy_estimator.run([pub])
    result = job.result()
    pub_result = result[0] 

    # .data is a DataBin
    data = pub_result.data

    # The Z expectation values
    evs = np.array(data.evs).squeeze()  # This is a numpy array of shape (num_qubits,)
    evs = evs.reshape(-1)

    return evs[::-1]

def qiskit_noisy_simulator(circuit, noise_model, num_qubits, num_layers, method="automatic"):
    z_expectations = []
    for layer in range(num_layers):
        qc_copy = circuit.copy()
        for j in range(layer):
            qc_copy = qc_copy.compose(circuit)
        z_expectations.append(qiskit_noisy_simulator_stepwise(qc_copy, noise_model, num_qubits, method=method))
    return np.array(z_expectations)


