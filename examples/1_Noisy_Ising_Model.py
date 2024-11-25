import matplotlib.pyplot as plt

from yaqs.data_structures.MPO import MPO
from yaqs.data_structures.MPS import MPS
from yaqs.data_structures.noise_model import NoiseModel
from yaqs.data_structures.simulation_parameters import Observable, SimulationParams
from yaqs.methods.TJM import TJM


# Define the system Hamiltonian
L = 10
d = 2
J = 1
g = 0.5
H_0 = MPO()
H_0.init_Ising(L, d, J, g)

# Define the initial state
state = MPS(L, state='zeros')

# Define the noise model
noise_model = NoiseModel(['relaxation', 'dephasing'], [0.1, 0.1])

# Define the simulation parameters
T = 10
dt = 0.1
max_bond_dim = 4
N = 100
threshold = 1e-6
measurements = [Observable('x', site) for site in range(L)]
sim_params = SimulationParams(measurements, T, dt, N, max_bond_dim, threshold)

if __name__ == "__main__":
    TJM(state, H_0, noise_model, sim_params)

    heatmap = []
    for observable in sim_params.observables:
        heatmap.append(observable.results)

    plt.imshow(heatmap, aspect='auto', extent=[0, T, L, 0])
    # Centers site ticks
    plt.yticks([x-0.5 for x in list(range(1,L+1))], range(1,L+1))
    cbar = plt.colorbar()
    cbar.ax.set_title('$\\langle X \\rangle$')
    plt.xlabel('t')
    plt.ylabel('Site')
    plt.show()
