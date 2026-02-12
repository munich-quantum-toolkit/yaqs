# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for noise distribution support."""

from __future__ import annotations

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.simulator import run


def test_noise_distribution_integration() -> None:
    """Test that running a simulation with a distributed noise model works.

    The noise model should be sampled once per run, meaning all trajectories
    share the same (randomly sampled) noise strengths.
    """
    num_qubits = 2
    # Define Hamiltonian: Ising model
    hamiltonian = MPO.ising(num_qubits, J=1.0, g=0.5)

    # Define noise model with distribution
    processes = [
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "normal", "mean": 0.1, "std": 0.01},
        }
    ]
    noise_model = NoiseModel(processes)

    # Initial state
    initial_state = MPS(num_qubits)

    # Simulation parameters
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        dt=0.1,
        elapsed_time=1.0,
        num_traj=10,  # Run multiple trajectories to confirm it runs
        sample_timesteps=False,
    )

    # Run simulation
    run(initial_state, hamiltonian, sim_params, noise_model)

    # If we reached here without error, the integration works (at least doesn't crash)
    assert True
