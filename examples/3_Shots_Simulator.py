import copy
import numpy as np
import qiskit.circuit
from qiskit.circuit.library.n_local import TwoLocal
import matplotlib.pyplot as plt

from yaqs.general.data_structures.networks import MPS
from yaqs.general.data_structures.noise_model import NoiseModel
from yaqs.general.data_structures.simulation_parameters import Observable, WeakSimParams
from yaqs.circuits.simulation import simulator

# Define the circuit
num_qubits = 5
depth = num_qubits
circuit = qiskit.circuit.QuantumCircuit(num_qubits)

# Example: Two-Local Circuit
twolocal = TwoLocal(num_qubits, ['rx'], ['rzz'], entanglement='linear', reps=depth).decompose()        
num_pars = len(twolocal.parameters)
values = np.random.uniform(low=-np.pi, high=np.pi, size=num_pars)
circuit = copy.deepcopy(twolocal).assign_parameters(values)

# Define the noise model
gamma = 0
noise_model = NoiseModel(['relaxation'], [gamma])

# Define the initial state
state = MPS(num_qubits, state='zeros')

# Define the simulation parameters
shots = 10000
max_bond_dim = 4
threshold = 1e-6
sim_params = WeakSimParams(shots, max_bond_dim, threshold)

if __name__ == "__main__":
    simulator.run(state, circuit, sim_params, noise_model)

    plt.bar(sim_params.results.keys(), sim_params.results.values())
    plt.xlabel("Bitstring")
    plt.ylabel("Counts")
    plt.show()