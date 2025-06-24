from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, depolarizing_error
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

    # exact_estimator = Estimator()
    noisy_estimator = Estimator(options=dict(backend_options=dict(noise_model=noise_model)))
    pub = (qc_copy, observables)
    job = noisy_estimator.run([pub])
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
    num_qubits = 1

    qc = QuantumCircuit(num_qubits)
    qc.h(0)



  


   
    # noise_model = qiskit_dephasing_noise(num_qubits=2, noise_strengths=[0.2, 0.25])
    noise_model = None
    depolarizing_prob = 0.01
    noise_model = NoiseModel()
    # noise_model.add_all_qubit_quantum_error(
    # depolarizing_error(depolarizing_prob, 2), ["cx"]
    # )
    noise_model.add_all_qubit_quantum_error(
    depolarizing_error(depolarizing_prob, 1), ["h"]
    )
   

    z_vals = qiskit_noisy_simulator(qc, noise_model, num_qubits)
    print("Z expectations per layer:\n", z_vals)




