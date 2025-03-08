# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Example: Weak Quantum Circuit Simulation (Shots) with YAQS.

This module demonstrates how to run a weak simulation using the YAQS simulator
with a TwoLocal circuit generated via Qiskit's circuit library. An MPS is initialized
in the |0> state, a noise model is applied, and weak simulation parameters are set.
After running the simulation, the measurement results (bitstring counts) are displayed
as a bar chart.

Usage:
    Run this module as a script to execute the simulation and display the results.
"""

from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library.n_local import TwoLocal

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import WeakSimParams

if __name__ == "__main__":
    num_qubits = 10

    # Create the circuit.
    twolocal = TwoLocal(num_qubits, ["rx"], ["rzz"], entanglement="linear", reps=num_qubits).decompose()
    num_pars = len(twolocal.parameters)
    rng = np.random.default_rng()
    values = rng.uniform(-np.pi, np.pi, size=num_pars)
    circuit = copy.deepcopy(twolocal).assign_parameters(values)
    circuit.measure_all()

    # Define the noise model.
    gamma = 0.1
    noise_model = NoiseModel(["relaxation", "dephasing"], [gamma, gamma])

    # Define the initial state.
    state = MPS(num_qubits, state="zeros")

    # Define the simulation parameters.
    shots = 1024
    max_bond_dim = 4
    threshold = 1e-6
    window_size = 0
    sim_params = WeakSimParams(shots, max_bond_dim, threshold, window_size)

    # Run the simulation.
    simulator.run(state, circuit, sim_params, noise_model)

    # Plot the measurement outcomes as a bar chart.
    plt.bar(sim_params.results.keys(), sim_params.results.values())
    plt.xlabel("Bitstring")
    plt.ylabel("Counts")
    plt.title("Measurement Results")
    plt.show()
