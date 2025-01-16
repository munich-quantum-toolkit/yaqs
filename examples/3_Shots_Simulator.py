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
num_qubits = 10
depth = num_qubits
circuit = qiskit.circuit.QuantumCircuit(num_qubits)

# Example: Two-Local Circuit
twolocal = TwoLocal(num_qubits, ['rx'], ['rzz'], entanglement='linear', reps=depth).decompose()        
num_pars = len(twolocal.parameters)
values = np.random.uniform(low=-np.pi, high=np.pi, size=num_pars)
circuit = copy.deepcopy(twolocal).assign_parameters(values)
circuit.measure_all()

# L = 10
# d = 2
# J = 1
# g = 0.5
# H_0 = MPO()
# H_0.init_Ising(L, d, J, g)

# Define the initial state
state = MPS(num_qubits, state='zeros')

# Define the noise model
# gamma = 0.1
# noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma, gamma])

# Define the simulation parameters
shots = 10000
N = 1
max_bond_dim = 4
threshold = 1e-6
# measurements = [Observable('x', site) for site in range(num_qubits)]
sim_params = WeakSimParams(shots, N, max_bond_dim, threshold)

if __name__ == "__main__":
    simulator.run(state, circuit, sim_params)

    plt.bar(sim_params.prob_dists[0].keys(), sim_params.prob_dists[0].values())
    plt.xlabel("Bitstring")
    plt.ylabel("Counts")
    plt.show()