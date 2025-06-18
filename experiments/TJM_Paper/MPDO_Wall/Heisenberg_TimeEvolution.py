# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import pickle

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs.core.libraries.gate_library import Z

if __name__ == "__main__":
    # Define the system Hamiltonian
    L = 30
    J = 1
    g = 0.5
    H_0 = MPO()
    # H_0.init_Ising(L, d, J, g)
    H_0.init_heisenberg(L, J, J, J, g)

    # Define the initial state
    state = MPS(L, state="wall")

    # Define the noise model
    gamma_relaxation = 0.1
    gamma_dephasing = 0.1
    noise_model = NoiseModel(["relaxation", "excitation"], [gamma_relaxation, gamma_dephasing])

    # Define the simulation parameters
    T = 10
    dt = 0.1
    sample_timesteps = True
    N = 100
    max_bond_dim = 8
    threshold = 0
    order = 2
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T, dt, N, max_bond_dim, threshold, order, sample_timesteps=True)

    # TJM Example #################
    simulator.run(state, H_0, sim_params, noise_model)

    filename = "30L_NoNoise.pickle"
    with open(filename, "wb") as f:
        pickle.dump(
            {
                "sim_params": sim_params,
            },
            f,
        )
