import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pickle

from yaqs.general.data_structures.MPO import MPO
from yaqs.general.data_structures.MPS import MPS
from yaqs.general.data_structures.noise_model import NoiseModel
from yaqs.general.data_structures.simulation_parameters import Observable, SimulationParams
from yaqs.physics.methods.TJM import TJM


# Define the system Hamiltonian
L = 30
d = 2
J = 1
g = 0.5
H_0 = MPO()
# H_0.init_Ising(L, d, J, g)
H_0.init_Heisenberg(L, d, J, J, J, g)

# Define the initial state
state = MPS(L, state='x')

# Define the noise model
gamma_relaxation = 0.1
gamma_dephasing = 0.1
noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma_relaxation, gamma_dephasing])

# Define the simulation parameters
T = 10
dt = 0.1
sample_timesteps = True
N = 100
max_bond_dim = 4
threshold = 1e-6
order = 2
measurements = [Observable('x', site) for site in range(L)]
sim_params = SimulationParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

fig, ax = plt.subplots(1, 1)
if __name__ == "__main__":
    ########## TJM Example #################
    TJM(state, H_0, noise_model, sim_params)

    heatmap = []
    for observable in sim_params.observables:
        heatmap.append(observable.results)

    im = ax.imshow(heatmap, aspect='auto', extent=[0, T, L, 0])
    ax.set_ylabel('Site')

    # Centers site ticks
    ax.set_yticks([x-0.5 for x in list(range(1,L+1))], range(1,L+1))
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_title('$\\langle Z \\rangle$')
    #########################################
    filename = f"30L_Noise.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'sim_params': sim_params,
        }, f)

    plt.show()
