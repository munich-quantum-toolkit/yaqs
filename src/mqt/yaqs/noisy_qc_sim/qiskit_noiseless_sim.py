from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
import numpy as np


def get_qiskit_z_expectations(circuit, num_qubits, num_layers=1):
    """
    Simulate local Z expectations per qubit after each layer (circuit repetition) with Qiskit's StatevectorEstimator.

    Args:
        circuit (QuantumCircuit): The circuit (no measurements).
        num_qubits (int): Number of qubits.
        num_layers (int): Number of circuit repetitions.

    Returns:
        z_vals (list of lists): Each sublist is [<Z_0>, <Z_1>, ..., <Z_{n-1}>] after each layer.
    """
    # Prepare Z observables for each qubit
    z_observables = []
    for i in range(num_qubits):
        pauli_str = "I" * i + "Z" + "I" * (num_qubits - i - 1)
        z_observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))

    # StatevectorEstimator
    estimator = StatevectorEstimator()

    z_vals = []

    # Start with an empty circuit (all |0>)
    qc_current = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        qc_current = qc_current.compose(circuit)

        # Evaluate all Z expectations for this layer
        pub = (qc_current, z_observables)
        job = estimator.run([pub])
        result = job.result()
        # Get the list of <Z_i> for this layer
        z_layer = [float(val) for val in result[0].data.evs]
        z_vals.append(z_layer)

    return z_vals


