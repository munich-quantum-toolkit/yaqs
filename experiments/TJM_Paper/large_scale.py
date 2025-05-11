import numpy as np
import pickle

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs import simulator

def run_large_scale():
    # Define the system Hamiltonian
    L = 1000
    J = 1
    g = 0.5
    H_0 = MPO()
    # H_0.init_Ising(L, d, J, g)
    H_0.init_heisenberg(L, J, J, J, g)

    # Define the initial state
    state = MPS(L, state='wall')

    # Define the simulation parameters
    T = 10
    dt = 0.1
    sample_timesteps = True
    N = 100
    max_bond_dim = 8
    threshold = 0
    order = 2
    measurements = [Observable(Z(), site) for site in range(L)]

     # Define the noise parameters
    gammas= [0.1, 0]
    for gamma in gammas:

        noise_model = NoiseModel(['relaxation', 'excitation'], [gamma, gamma])
        sim_params = PhysicsSimParams(measurements, T, dt, N, max_bond_dim, threshold, order, sample_timesteps=sample_timesteps)


        ########## TJM Example #################
        simulator.run(state, H_0, sim_params, noise_model)

        if gamma == 0:
            filename = f"results/large_scale/TJM_1000L_Exact.pickle"
        else:
            filename = f"results/large_scale/TJM_1000L_Gamma01.pickle"
        with open(filename, 'wb') as f:
            pickle.dump({
                'sim_params': sim_params,
            }, f)