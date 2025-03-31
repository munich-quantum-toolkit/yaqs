import numpy as np
import matplotlib.pyplot as plt

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.circuit_library import create_2D_Fermi_Hubbard_circuit

from mqt.yaqs import simulator

# Define the model
t = 1.0         # kinetic hopping
v = 0.5         # chemical potential
u = 4.0         # onsite interaction
Lx, Ly = 2, 2   # lattice dimensions

# Define the circuit
num_sites = Lx * Ly
num_qubits = 2 * num_sites
num_trotter_steps = 1
timesteps = 1000
dt = 0.01
total_time = dt * timesteps
print("Total time: ", total_time)

# Define the initial state
state = MPS(num_qubits, state='wall', pad=32)

# Define the simulation parameters
N = 1
max_bond_dim = 16
threshold = 1e-6
window_size = 0
measurements = [Observable('z', site) for site in range(num_qubits)]

occupations = np.zeros((num_qubits, timesteps), dtype='complex')

if __name__ == "__main__":
    for timestep in range(timesteps):
        print("Timestep: " + str(timestep))
        model = {'name': '2D_Fermi_Hubbard', 'Lx': Lx, 'Ly': Ly, 'mu': -v, 'u': u, 't': -t, 'num_trotter_steps': num_trotter_steps}
        circuit = create_2D_Fermi_Hubbard_circuit(model, dt=dt, timesteps=1)

        sim_params = StrongSimParams(measurements, N, max_bond_dim, threshold, window_size, get_state=True)

        simulator.run(state, circuit, sim_params, None)

        for observable in sim_params.observables:
            index = observable.site
            occupations[index, timestep] = 0.5 * (1 - observable.results.item())

        state = MPS(num_qubits, sim_params.output_state.tensors, pad=32)

plt.figure(figsize=(10, 5))
for i in range(num_qubits):
    site = i // 2
    spin = "↑" if i % 2 == 0 else '↓'
    plt.plot(occupations[i, :], label=f"Site {site} " + spin)

plt.xlabel("Time")
plt.ylabel("Occupation")
plt.title("2D Hubbard Model: Time Evolution of Site Occupations (mqt-yaqs)")
plt.legend()
plt.show()