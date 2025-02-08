import copy
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.n_local import TwoLocal

from yaqs.core.data_structures.networks import MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import WeakSimParams

from yaqs import Simulator


# Define the circuit
num_qubits = 10
circuit = QuantumCircuit(num_qubits)

# Example: Two-Local Circuit
twolocal = TwoLocal(num_qubits, ['rx'], ['rzz'], entanglement='linear', reps=num_qubits).decompose()        
num_pars = len(twolocal.parameters)
values = np.random.uniform(low=-np.pi, high=np.pi, size=num_pars)
circuit = copy.deepcopy(twolocal).assign_parameters(values)
circuit.measure_all()

# Define the noise model
gamma = 0.1
noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma, gamma])

# Define the initial state
state = MPS(num_qubits, state='zeros')

# Define the simulation parameters
shots = 1024
max_bond_dim = 4
threshold = 1e-6
window_size = 0
sim_params = WeakSimParams(shots, max_bond_dim, threshold, window_size)

if __name__ == "__main__":
    Simulator.run(state, circuit, sim_params, noise_model)

    plt.bar(sim_params.results.keys(), sim_params.results.values())
    plt.xlabel("Bitstring")
    plt.ylabel("Counts")
    plt.show()