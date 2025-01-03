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
H_0.init_Heisenberg(L, d, -J, -J, -J, -g)

# Define the initial state
state = MPS(L, state='wall')

# Define the simulation parameters
dt = 0.1
sample_timesteps = False
N = 100
max_bond_dim = 4
threshold = 1e-6
order = 2
measurements = [Observable('z', site) for site in range(L)]

fig, ax = plt.subplots(1, 1)
if __name__ == "__main__":
    # T = 1
    # gammas = np.logspace(-2, -1, 20)
    # heatmap = np.empty((L, len(gammas)))
    # for j, gamma in enumerate(gammas):
    #     print("Gamma =", gamma)
    #     # Define the noise model
    #     noise_model = NoiseModel(['relaxation', 'excitation'], [gamma, gamma])
    #     sim_params = SimulationParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

    #     ########## TJM Example #################
    #     TJM(state, H_0, noise_model, sim_params)
    #     for i, observable in enumerate(sim_params.observables):
    #         heatmap[i, j] = observable.results[0]

    # filename = f"100L_T1_wall_fixed.pickle"
    # with open(filename, 'wb') as f:
    #     pickle.dump({
    #         'heatmap': heatmap,
    #     }, f)

    # gammas = np.logspace(-2, -1, 20)
    # T = 5
    # heatmap = np.empty((L, len(gammas)))
    # for j, gamma in enumerate(gammas):
    #     print("Gamma =", gamma)
    #     # Define the noise model
    #     noise_model = NoiseModel(['relaxation', 'excitation'], [gamma, gamma])
    #     sim_params = SimulationParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

    #     ########## TJM Example #################
    #     TJM(state, H_0, noise_model, sim_params)
    #     for i, observable in enumerate(sim_params.observables):
    #         heatmap[i, j] = observable.results[0]

    # filename = f"100L_T5_wall_fixed.pickle"
    # with open(filename, 'wb') as f:
    #     pickle.dump({
    #         'heatmap': heatmap,
    #     }, f)

    T = 10
    gammas = np.logspace(-4, 1, 100)
    gammas = gammas[55:65]
    heatmap = np.empty((L, len(gammas)))
    for j, gamma in enumerate(gammas):
        print("Gamma =", gamma)
        # Define the noise model
        noise_model = NoiseModel(['relaxation', 'excitation'], [gamma, gamma])
        sim_params = SimulationParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

        ########## TJM Example #################
        TJM(state, H_0, noise_model, sim_params)
        for i, observable in enumerate(sim_params.observables):
            heatmap[i, j] = observable.results[0]

    filename = f"100L_T10_wall_fixed.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'heatmap': heatmap,
        }, f)


    # T = 1
    # heatmap = np.empty((L, len(gammas)))
    # for j, gamma in enumerate(gammas):
    #     print("Gamma =", gamma)
    #     # Define the noise model
    #     noise_model = NoiseModel(['relaxation', 'excitation'], [gamma, gamma])
    #     sim_params = SimulationParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

    #     ########## TJM Example #################
    #     TJM(state, H_0, noise_model, sim_params)
    #     for i, observable in enumerate(sim_params.observables):
    #         heatmap[i, j] = observable.results[0]

    # filename = f"100L_T1_wall.pickle"
    # with open(filename, 'wb') as f:
    #     pickle.dump({
    #         'heatmap': heatmap,
    #     }, f)

    # T = 5
    # heatmap = np.empty((L, len(gammas)))
    # for j, gamma in enumerate(gammas):
    #     print("Gamma =", gamma)
    #     # Define the noise model
    #     noise_model = NoiseModel(['relaxation', 'excitation'], [gamma, gamma])
    #     sim_params = SimulationParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

    #     ########## TJM Example #################
    #     TJM(state, H_0, noise_model, sim_params)
    #     for i, observable in enumerate(sim_params.observables):
    #         heatmap[i, j] = observable.results[0]

    # filename = f"100L_T5_wall.pickle"
    # with open(filename, 'wb') as f:
    #     pickle.dump({
    #         'heatmap': heatmap,
    #     }, f)

    # T = 10
    # heatmap = np.empty((L, len(gammas)))
    # for j, gamma in enumerate(gammas):
    #     print("Gamma =", gamma)
    #     # Define the noise model
    #     noise_model = NoiseModel(['relaxation', 'excitation'], [gamma, gamma])
    #     sim_params = SimulationParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

    #     ########## TJM Example #################
    #     TJM(state, H_0, noise_model, sim_params)
    #     for i, observable in enumerate(sim_params.observables):
    #         heatmap[i, j] = observable.results[0]

    # filename = f"100L_T10_wall.pickle"
    # with open(filename, 'wb') as f:
    #     pickle.dump({
    #         'heatmap': heatmap,
    #     }, f)
