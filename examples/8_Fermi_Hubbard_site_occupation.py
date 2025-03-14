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
Lx, Ly = 2, 2   # lattice dimensions

# Define the circuit
num_sites = Lx * Ly
num_qubits = 2*num_sites
num_trotter_steps = 10
timesteps = 10
dt = 1
total_time = dt * timesteps

# Define the initial state
state = MPS(num_qubits, state='wall')

# Define the simulation parameters
N = 100
max_bond_dim = 4
threshold = 1e-6
window_size = 0
measurements = [Observable('z', site) for site in range(num_qubits)]

#heatmap = np.empty((num_qubits, timesteps))
heatmap = np.zeros((num_sites, timesteps))

print(state.)

if __name__ == "__main__":
    for timestep in range(timesteps):
        print("Timestep: " + str(timestep))
        model = {'name': '2D_Fermi_Hubbard', 'Lx': Lx, 'Ly': Ly, 'mu': v, 'u': u, 't': t, 'num_trotter_steps': num_trotter_steps}
        circuit = create_2D_Fermi_Hubbard_circuit(model, dt=dt, timesteps=timestep)
        circuit.measure_all()

        sim_params = StrongSimParams(measurements, N, max_bond_dim, threshold, window_size)
        #gamma = 1e-5
        #noise_model = NoiseModel(['relaxation'], [gamma])
        Simulator.run(state, circuit, sim_params, None)

        for observable in sim_params.observables:
            index = observable.site // 2
            print(str(observable.site) + " -> " + str(index))
            heatmap[index, timestep] += (1/2 * (1 - observable.results.item()))

    print(heatmap.shape)
    plt.plot(heatmap.transpose())
    plt.show()