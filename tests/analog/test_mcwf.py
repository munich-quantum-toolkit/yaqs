# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Monte Carlo Wavefunction (MCWF) Solver."""

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.simulator import run


def test_mcwf_amplitude_damping() -> None:
    """Test single qubit amplitude damping with MCWF solver.
    
    Checks if the average of many trajectories matches the analytical solution.
    """
    n_sites = 1
    initial_state = MPS(n_sites, state="ones")  # |1>
    hamiltonian = MPO()
    hamiltonian.identity(n_sites)
    for i in range(len(hamiltonian.tensors)):
        hamiltonian.tensors[i] *= 0.0  # H = 0

    # Noise: Amplitude Damping
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    gamma = 1.0
    noise_processes = [{"name": "destroy", "sites": [0], "strength": gamma, "matrix": sigma_minus}]
    noise_model = NoiseModel(processes=noise_processes)

    t_max = 2.0
    dt = 0.05
    obs = Observable("z", sites=[0])

    # We need enough trajectories to converge reasonably well
    num_traj = 200
    sim_params = AnalogSimParams(
        observables=[obs],
        elapsed_time=t_max,
        dt=dt,
        solver="MCWF",
        num_traj=num_traj,
        show_progress=False
    )

    run(initial_state, hamiltonian, sim_params, noise_model, parallel=False)

    times = sim_params.times
    sigma_z_sim = obs.results
    assert sigma_z_sim is not None
    
    # Analytical solution for <sigma_z>:
    # rho_11(t) = exp(-gamma t)
    # rho_00(t) = 1 - exp(-gamma t)
    # <Z> = rho_00 - rho_11 = 1 - 2 exp(-gamma t)
    delta_exact = 1 - 2 * np.exp(-gamma * times)

    # Comparison (allow larger tolerance due to stochastic noise ~ 1/sqrt(N))
    # 1/sqrt(200) ~ 0.07. 3 sigma ~ 0.21.
    diff = np.abs(sigma_z_sim - delta_exact)
    assert np.all(diff < 0.25), f"Max diff: {np.max(diff)}"


def test_mcwf_unitary_rabi() -> None:
    """Test single qubit unitary evolution (Rabi oscillation) with no noise.
    
    Should be deterministic and match exact solution tightly.
    """
    n_sites = 1
    initial_state = MPS(n_sites, state="zeros")  # |0>

    hamiltonian = MPO.ising(n_sites, J=0.0, g=-1.0)
    # H = +1.0 * X

    t_max = 2.0 * np.pi
    dt = 0.01
    obs = Observable("z", sites=[0])

    sim_params = AnalogSimParams(
        observables=[obs], 
        elapsed_time=t_max, 
        dt=dt, 
        solver="MCWF",
        num_traj=1 # Deterministic
    )

    run(initial_state, hamiltonian, sim_params, None)

    times = sim_params.times
    sigma_z_sim = obs.results
    assert sigma_z_sim is not None
    sigma_z_exact = np.cos(2 * times)

    # Should match very well
    diff = np.abs(sigma_z_sim - sigma_z_exact)
    assert np.all(diff < 1e-4), f"Max diff: {np.max(diff)}"


def test_mcwf_dephasing() -> None:
    """Test 2-qubit system with local dephasing on one qubit."""
    n_sites = 2
    initial_state = MPS(n_sites, state="x+")  # |++>

    hamiltonian = MPO()
    hamiltonian.identity(n_sites)
    for i in range(len(hamiltonian.tensors)):
        hamiltonian.tensors[i] *= 0.0

    # Dephasing on qubit 0 (sigma_z)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    gamma = 0.5
    noise_processes = [{"name": "dephasing", "sites": [0], "strength": gamma, "matrix": sigma_z}]

    noise_model = NoiseModel(processes=noise_processes)

    t_max = 2.0
    dt = 0.05
    obs0 = Observable("x", sites=[0])
    obs1 = Observable("x", sites=[1])

    num_traj = 200
    sim_params = AnalogSimParams(
        observables=[obs0, obs1], 
        elapsed_time=t_max, 
        dt=dt, 
        solver="MCWF",
        num_traj=num_traj,
        show_progress=False
    )



    # Use parallel=True to verify infrastructure
    run(initial_state, hamiltonian, sim_params, noise_model, parallel=True)

    times = sim_params.times
    x0_sim = obs0.results
    x1_sim = obs1.results
    assert x0_sim is not None
    assert x1_sim is not None

    x0_exact = np.exp(-2 * gamma * times)
    x1_exact = np.ones_like(times)

    # Stochastic tolerance
    assert np.all(np.abs(x0_sim - x0_exact) < 0.2), f"Qubit 0 failed. Max diff: {np.max(np.abs(x0_sim - x0_exact))}"
    assert np.all(np.abs(x1_sim - x1_exact) < 0.05), f"Qubit 1 failed. Max diff: {np.max(np.abs(x1_sim - x1_exact))}" # Should be constant 1, maybe small fluctuations if jumps happen but they are Z jumps on + state -> - state... wait.
    # Z on |+> gives |->. |+> = (|0>+|1>), Z|+> = (|0>-|1>) = |->.
    # expectation <+|X|+> = 1. <|X|> = -1.
    # If jumps happen, X flips sign. Average should decay.
    # Wait, Z dephasing causes decay of X expectation.
    # My exact solution is correct.
    # Qubit 1 has NO noise. It should evolve unitarily (identity). So it should stay |+>. <X> = 1.
    # But initialization might have noise? No, noise model is on site 0.
