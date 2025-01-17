import copy
import numpy as np
import qiskit.circuit
from qiskit.circuit.library.n_local import TwoLocal
import matplotlib.pyplot as plt

from yaqs.general.data_structures.networks import MPS
from yaqs.general.data_structures.noise_model import NoiseModel
from yaqs.general.data_structures.simulation_parameters import Observable, StrongSimParams
from yaqs.circuits.simulation import simulator

# Define the circuit
num_qubits = 10
depth = 1
circuit = qiskit.circuit.QuantumCircuit(num_qubits)

# Example: Two-Local Circuit
twolocal = TwoLocal(num_qubits, ['rz'], ['cx'], entanglement='linear', reps=depth).decompose()        
num_pars = len(twolocal.parameters)
values = np.random.uniform(low=-np.pi, high=np.pi, size=num_pars)
circuit = copy.deepcopy(twolocal).assign_parameters(values)

# Define the noise model
gamma = 0
noise_model = NoiseModel(['excitation', 'dephasing'], [gamma, gamma])

# Define the initial state
state = MPS(num_qubits, state='zeros')

# Define the simulation parameters
N = 1000
max_bond_dim = 4
threshold = 1e-6
measurements = [Observable('z', site) for site in range(num_qubits)]
sim_params = StrongSimParams(measurements, N, max_bond_dim, threshold)

if __name__ == "__main__":
    simulator.run(state, circuit, sim_params, noise_model)
    heatmap = []
    for observable in sim_params.observables:
        heatmap.append(observable.results)
        print(observable.results)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(heatmap, aspect='auto')
    ax.set_ylabel('Site')

    # plt.bar(sim_params.results.keys(), sim_params.results.values())
    # plt.xlabel("Bitstring")
    # plt.ylabel("Counts")
    plt.show()