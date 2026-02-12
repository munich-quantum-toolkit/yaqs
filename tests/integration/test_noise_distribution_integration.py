from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from mqt.yaqs.simulator import run
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z

def test_noise_distribution_integration():
    """Test that running a simulation with a distributed noise model works."""
    num_qubits = 2
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

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
    sim_params = StrongSimParams(
        observables=[Observable(Z(), 0)],
        sample_layers=False,
        num_traj=10,  # Run multiple trajectories to trigger sampling
    )

    # Run simulation
    run(initial_state, qc, sim_params, noise_model)

    # If we reached here without error, the integration works (at least doesn't crash)
    assert True
