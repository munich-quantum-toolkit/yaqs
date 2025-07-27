from mqt.yaqs.core.data_structures.networks import MPS, MPO
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Z, RuntimeCost
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

import matplotlib.pyplot as plt
import numpy as np
import pickle

def tdvp_simulator(H_0, noise_model, state=None):
    L = H_0.length
    num_traj = 10
    max_bond_dim = 2**(L-1)
    min_bond_dim = 2
    threshold = 1e-6
    sample_timesteps = True
    order = 1

    elapsed_time = 5
    dt = 0.1

    state = MPS(length=L)

    results = {}
    # for j in range(int(elapsed_time / dt)):
    # time = dt

    measurements = [Observable("runtime_cost")]
    sim_params = AnalogSimParams(measurements, elapsed_time, dt, num_traj, max_bond_dim, min_bond_dim, threshold, order, sample_timesteps=sample_timesteps)

    simulator.run(state, H_0, sim_params, noise_model=noise_model)

    cost = sim_params.observables[0].results
    return cost


L_list = [8, 16, 24, 32, 40, 48]
for L in L_list:
    print(L)
    state = MPS(length=L, state="zeros")
    J = 1
    h = 1
    H_0 = MPO()
    H_0.init_ising(L, J, h)

    gammas = np.logspace(-3, 3, 30)
    results = []
    for j, gamma in enumerate(gammas):
        print(j, "of", len(gammas))
        # Define the noise model
        noise_model = NoiseModel([
            {"name": name, "sites": [i], "strength": gamma} for i in range(L) for name in ["dephasing", "bitflip", "bitphaseflip"]
            ])

        cost = tdvp_simulator(H_0, noise_model)
        results.append(cost)
    filename = f"results_{L}.pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle)