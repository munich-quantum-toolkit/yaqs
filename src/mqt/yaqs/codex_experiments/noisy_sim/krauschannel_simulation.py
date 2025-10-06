import numpy as np
from qiskit.quantum_info import Operator
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit


from mqt.yaqs.circuits.utils.dag_utils import convert_dag_to_tensor_algorithm
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.libraries.noise_library import NoiseLibrary



from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.backends.aer_simulator import AerSimulator
import matplotlib.pyplot as plt


def apply(self, rho):
    """Apply the channel to a density matrix rho."""
    result = np.zeros_like(rho, dtype=complex)
    for k, prob in zip(self.kraus_ops, self.probabilities):
        result += prob * k @ rho @ k.conj().T
    return result

import numpy as np

def expand_operator(local_op, site, n_qubits):
    """Expand a single-qubit operator to act on 'site' in an n-qubit system."""
    ops = [np.eye(2)] * n_qubits
    ops[site] = local_op
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def KrausChannel(rho, noisemodel, sites):
    """
    Apply a Kraus channel (from NoiseModel) selectively to 'sites' of the density matrix 'rho'.
    If len(sites)==1, all local ops for that site are used (standard local channel).
    If len(sites)==2, Kraus ops are all local ops for site 0 (embedded), and all for site 1 (embedded),
    but **never** tensor products where both are non-identity.
    """
    n_qubits = int(np.log2(rho.shape[0]))
    kraus_ops_global = []
    if len(sites) == 1:
        site = sites[0]
        for i, process in enumerate(noisemodel.processes[site]):
            local_K = np.sqrt(noisemodel.strengths[site][i]) * getattr(NoiseLibrary, process)().matrix
            global_K = expand_operator(local_K, site, n_qubits)
            kraus_ops_global.append(global_K)
    elif len(sites) == 2:
        # For each site, embed each local Kraus operator individually (identity elsewhere)
        for idx, site in enumerate(sites):
            for i, process in enumerate(noisemodel.processes[site]):
                local_K = np.sqrt(noisemodel.strengths[site][i]) * getattr(NoiseLibrary, process)().matrix
                global_K = expand_operator(local_K, site, n_qubits)
                kraus_ops_global.append(global_K)
    else:
        raise ValueError("This function currently supports only 1 or 2 sites.")

    # Kraus channel application
    result = np.zeros_like(rho, dtype=complex)
    for K in kraus_ops_global:
        result += K @ rho @ K.conj().T

    return result

    
def create_all_zero_density_matrix(n_qubits):
    """
    Creates the density matrix for an n-qubit all-zero state (|0...0><0...0|).

    Args:
        n_qubits (int): The number of qubits.

    Returns:
        np.ndarray: The (2^n x 2^n) density matrix.
    """
    dim = 2**n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0  # Set the |0...0><0...0| element to 1
    return rho

def circuit_to_unitary_list(circuit):
    """
    Convert a Qiskit circuit to a list of gates with qubit indices.
    Returns a list of tuples (unitary, [qubit indices])
    """
    dag = circuit_to_dag(circuit)
    return convert_dag_to_tensor_algorithm(dag)




def evolve_noisy_circuit(rho0, gate_list, noisemodel):
    """
    Evolve a density matrix rho0 through the list of gates,
    applying Kraus noise after every gate as specified in kraus_channel_map.
    kraus_channel_map: dict with keys '1q' and '2q' for 1- and 2-qubit noise
    Returns: final density matrix
    """
    n = int(np.log2(rho0.shape[0]))
    rho = np.copy(rho0)
    for gate in gate_list:
        # Expand gate to full Hilbert space
        if len(gate.sites) == 1:
            U = np.eye(1)
            for i in range(n):
                U = np.kron(U, gate.matrix if i == gate.sites[0] else np.eye(2))
     
        elif len(gate.sites) == 2:
            if np.abs(gate.sites[-1] - gate.sites[0]) > 1:
                raise ValueError("Non-adjacent two-qubit gates not supported")

            idx0, idx1 = gate.sites[0], gate.sites[1]
            if idx0 > idx1:
                idx0, idx1 = idx1, idx0
                gate.matrix = two_qubit_reverse(gate.matrix)
            U = np.eye(1)
            i = 0
            while i < n:
                if len(gate.sites) == 2 and i == idx0:
                    U = np.kron(U, gate.matrix)
                    i += 2  # skip both qubits (idx0, idx1)
                else:
                    U = np.kron(U, np.eye(2))
                    i += 1
        else:
            raise ValueError("Only 1- and 2-qubit gates supported")
        # Apply unitary    
        rho = U @ rho @ U.conj().T

        # Apply noise
        rho = KrausChannel(rho, noisemodel, gate.sites)

    return rho

def z_expectations(rho, num_qubits):
    """
    Compute <Z> for each qubit for the given density matrix.
    """
    z_vals = []
    sz = np.array([[1,0],[0,-1]])
    I = np.eye(2)
    for i in range(num_qubits):
        op = 1
        for j in range(num_qubits):
            op = np.kron(op, sz if i == j else I)
        z_vals.append(np.real(np.trace(rho @ op)))
    return np.array(z_vals)

def two_qubit_reverse(mat):
    """
    For a 4x4 gate acting on qubits (a, b), swap a and b.
    Only needed if cx order is reversed.
    """
    SWAP = np.array([[1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]])
    mat = SWAP @ mat @ SWAP 
    return mat

def get_qiskit_z_expectations(circuit, num_qubits):
    """
    Helper function to get Z expectations using Qiskit's Aer simulator.
    """
    z_expectations = np.zeros(num_qubits)
    simulator = AerSimulator(method='density_matrix')
    qc_with_save = circuit.copy()
    qc_with_save.save_density_matrix() 

    # Run the circuit on the simulator
    job = simulator.run(qc_with_save) 
    result = job.result()

    density_matrix = result.data(0)['density_matrix'].data

    z_vals = []
    sz = np.array([[1,0],[0,-1]])
    I = np.eye(2)
    
    for i in range(num_qubits):
        op = 1
        for j in range(num_qubits):
            # Construct the Z operator for the current qubit
            op = np.kron(op, sz if i == j else I)
    
        # Compute the expectation value <Z_i>
        z_val = np.real(np.trace(density_matrix @ op))
        z_vals.append(z_val)
    return np.array(z_vals)



