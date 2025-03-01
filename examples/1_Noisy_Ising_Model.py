# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams

# Define the system Hamiltonian
L = 10
J = 1
g = 0.5
H_0 = MPO()
H_0.init_Ising(L, J, g)

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

fig, ax = plt.subplots(2, 1)
if __name__ == "__main__":
    # TJM Example #################
    Simulator.run(state, H_0, sim_params, noise_model)

    heatmap = [observable.results for observable in sim_params.observables]

    im = ax[0].imshow(heatmap, aspect="auto", extent=[0, T, L, 0], vmin=0, vmax=0.5)
    ax[0].set_ylabel("Site")

    # Centers site ticks
    ax[0].set_yticks([x - 0.5 for x in list(range(1, L + 1))], range(1, L + 1))
    #########################################

    # ######### QuTip Exact Solver ############
    # Time vector
    t = np.arange(0, sim_params.T + sim_params.dt, sim_params.dt)

    # Define Pauli matrices
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # Construct the Ising Hamiltonian
    H = 0
    for i in range(L - 1):
        H += -J * qt.tensor([sz if n == i or n == i + 1 else qt.qeye(2) for n in range(L)])
    for i in range(L):
        H += -g * qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)])

    # Construct collapse operators

    # Relaxation operators
    c_ops = [np.sqrt(gamma) * qt.tensor([qt.destroy(2) if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    # Dephasing operators
    c_ops.extend(np.sqrt(gamma) * qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L))

    # Initial state
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

    # Define measurement operators
    sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    # Exact Lindblad solution
    result_lindblad = qt.mesolve(H, psi0, t, c_ops, sx_list, progress_bar=True)
    heatmap2 = [result_lindblad.expect[site] for site in range(len(sx_list))]

    # Error heatmap
    heatmap = np.array(heatmap)
    heatmap2 = np.array(heatmap2)
    im2 = ax[1].imshow(heatmap2, aspect="auto", extent=[0, T, L, 0], vmin=0, vmax=0.5)

    ax[1].set_yticks([x - 0.5 for x in list(range(1, L + 1))], range(1, L + 1))

    fig.subplots_adjust(top=0.95, right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.11, 0.025, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title("$\\langle X \\rangle$")

    ax[1].set_xlabel("t")
    ax[1].set_ylabel("Site")
    plt.show()
