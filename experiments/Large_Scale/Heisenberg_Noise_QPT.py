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
L = 100
d = 2
J = 1
g = 0.5
H_0 = MPO()
# H_0.init_Ising(L, d, J, g)
H_0.init_Heisenberg(L, d, J, J, J, g)

# Define the initial state
state = MPS(L, state='x')

# Define the simulation parameters
T = 10
dt = 0.5
sample_timesteps = False
N = 100
max_bond_dim = 4
threshold = 1e-6
order = 2
measurements = [Observable('x', site) for site in range(L)]

fig, ax = plt.subplots(1, 1)
gammas = np.logspace(-4, 1, 100)
if __name__ == "__main__":
    heatmap = np.empty((L, len(gammas)))
    for j, gamma in enumerate(gammas):
        print("Gamma =", gamma)
        # Define the noise model
        noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma, gamma])
        sim_params = SimulationParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

        ########## TJM Example #################
        TJM(state, H_0, noise_model, sim_params)
        for i, observable in enumerate(sim_params.observables):
            heatmap[i, j] = observable.results[0]

    filename = f"100L_T10.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'heatmap': heatmap,
        }, f)

    im = ax.imshow(heatmap, aspect='auto')
    ax.set_xlabel('$\\gamma$')
    ax.set_ylabel('Site')

    # Centers site ticks
    # ax.set_xticks([x-0.5 for x in list(range(len(gammas)))], gammas)
    ax.set_yticks([x-0.5 for x in list(range(1,L+1))], range(1,L+1))
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_title('$\\langle X \\rangle$')
    #########################################

    plt.show()
