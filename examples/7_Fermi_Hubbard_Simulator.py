import numpy as np
import qiskit.circuit
import matplotlib.pyplot as plt

from yaqs.core.data_structures.networks import MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from yaqs.core.libraries.circuit_library import create_2D_Fermi_Hubbard_circuit

from yaqs import Simulator

# Define the model
t = -1.0        # kinetic hopping
v = 0.5         # chemical potential
u = 2.0         # onsite interaction
(x, y) = (2,2)  # lattice dimensions

# Define the circuit
num_sites = x*y
num_qubits = 2*num_sites
num_trotter_steps = 1
timesteps = 100
circuit = qiskit.circuit.QuantumCircuit(num_qubits)

model = {'name': '2D_Fermi_Hubbard', 'Lx': x, 'Ly': y, 'mu': v, 'u': u, 't': t, 'num_trotter_steps': num_trotter_steps}
circuit = create_2D_Fermi_Hubbard_circuit(model, dt=0.1, timesteps=timesteps)
circuit.measure_all()

# Define the initial state
state = MPS(num_qubits, state='zeros')

# Define the simulation parameters
N = 100
max_bond_dim = 4
threshold = 1e-6
window_size = 0
measurements = [Observable('z', site) for site in range(num_qubits)]

if __name__ == "__main__":
    sim_params = StrongSimParams(measurements, N, max_bond_dim, threshold, window_size)
    gamma = 1e-5
    noise_model = NoiseModel(['relaxation'], [gamma])
    Simulator.run(state, circuit, sim_params, noise_model)

    heatmap = []
    for observable in sim_params.observables:
        heatmap.append(observable.results)

    im = plt.imshow(heatmap, aspect='auto', extent=[0, timesteps, num_qubits, 0], vmin=0, vmax=0.5)
    y_labels = []
    for i in range(1, num_sites+1):
        y_labels.append(f'{i}↑')
        y_labels.append(f'{i}↓')
    plt.yticks([x-0.5 for x in list(range(1,num_qubits+1))], y_labels)
    plt.xlabel('t')
    cbar = plt.colorbar(im)
    cbar.set_label("$\\langle Z \\rangle$")

    plt.show()