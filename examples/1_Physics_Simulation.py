# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Example: Noisy Hamiltonian Simulation with YAQS.

This module demonstrates how to run a Hamiltonian simulation using the YAQS Simulator
and visualize the results. In this example, an Ising Hamiltonian is initialized as an MPO, and an MPS
state is prepared in the |0> state. A noise model is applied, and simulation parameters are defined
for a physics simulation using the Tensor Jump Method (TJM). After running the simulation, the
expectation values of the X observable are extracted and displayed as a heatmap.

Usage:
    Run this module as a script to execute the simulation and display the resulting heatmap.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams

# Define the system Hamiltonian
L = 10
J = 1
g = 0.5
H_0 = MPO()
H_0.init_ising(L, J, g)

# Define the initial state
state = MPS(L, state="zeros")

# Define the noise model
gamma = 0.1
noise_model = NoiseModel(["relaxation", "dephasing"], [gamma, gamma])

# Define the simulation parameters
T = 10
dt = 0.1
sample_timesteps = True
N = 100
max_bond_dim = 4
threshold = 1e-6
order = 2
measurements = [Observable("x", site) for site in range(L)]
sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)

fig, ax = plt.subplots(1, 1)
if __name__ == "__main__":
    # TJM Example
    Simulator.run(state, H_0, sim_params, noise_model)

    heatmap = [observable.results for observable in sim_params.observables]

    im = plt.imshow(heatmap, aspect="auto", extent=[0, T, L, 0], vmin=0, vmax=0.5)
    plt.xlabel("Site")
    plt.yticks([x - 0.5 for x in list(range(1, L + 1))], range(1, L + 1))
    plt.ylabel("t")

    fig.subplots_adjust(top=0.95, right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.11, 0.025, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title("$\\langle X \\rangle$")

    plt.show()
