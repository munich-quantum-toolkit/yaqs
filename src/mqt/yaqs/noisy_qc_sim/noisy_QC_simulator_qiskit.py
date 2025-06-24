from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.backends.aer_simulator import AerSimulator
from qiskit_aer import Aer
import numpy as np

def qiskit_noisy_simulator(circuit, noise_model, num_qubits):
    """
    Helper function to get Z expectations using Qiskit's Aer simulator.
    """
    observables = []
    for i in range(num_qubits):
        pauli_str = "I"*i + "Z" + "I"*(num_qubits - i - 1)
        observable = SparsePauliOp.from_list([(pauli_str, 1.0)])
        observables.append(observable)

    z_expectations = np.zeros(num_qubits)
    qc_copy = circuit.copy()

    exact_estimator = Estimator()
    pub = (qc_copy, observables)
    job = exact_estimator.run([pub])
    result = job.result()
    pub_result = result[0] 

    # .data is a DataBin
    data = pub_result.data

    # The Z expectation values
    z_expectations = data.evs  # This is a numpy array of shape (num_qubits,)

    return z_expectations


if __name__ == "__main__":

    from mqt.yaqs.noisy_qc_sim.qiskit_noisemodels import qiskit_dephasing_noise
    # Create a simple quantum circuit

    qc = QuantumCircuit(2)
    qc.h(0)

   
    noise_model = qiskit_dephasing_noise(num_qubits=2, noise_strengths=[0.1, 0.2])
    # noise_model = None
   

    z_vals = qiskit_noisy_simulator(qc, noise_model, 2)
    print("Z expectations per layer:\n", z_vals)




