# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Monte Carlo Wavefunction (MCWF) Solver."""

import numpy as np
import pytest

from mqt.yaqs.analog.mcwf import mcwf, preprocess_mcwf
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
        observables=[obs], elapsed_time=t_max, dt=dt, solver="MCWF", num_traj=num_traj, show_progress=False
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
    # Comparison (using mean error for robustness against stochastic fluctuations)
    mean_diff = np.mean(np.abs(sigma_z_sim - delta_exact))
    assert mean_diff < 0.25, f"Mean diff: {mean_diff}"


def test_mcwf_unitary_rabi() -> None:
    """Test single qubit unitary evolution (Rabi oscillation) with no noise.

    Should be deterministic and match exact solution tightly.
    """
    n_sites = 1
    initial_state = MPS(n_sites, state="zeros")  # |0>

    hamiltonian = MPO.ising(n_sites, J=0.0, g=-1.0)

    t_max = 2.0 * np.pi
    dt = 0.01
    obs = Observable("z", sites=[0])

    sim_params = AnalogSimParams(
        observables=[obs],
        elapsed_time=t_max,
        dt=dt,
        solver="MCWF",
        num_traj=1,  # Deterministic
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
        observables=[obs0, obs1], elapsed_time=t_max, dt=dt, solver="MCWF", num_traj=num_traj, show_progress=False
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
    # Stochastic tolerance (using mean error)
    mean_diff_0 = np.mean(np.abs(x0_sim - x0_exact))
    mean_diff_1 = np.mean(np.abs(x1_sim - x1_exact))

    assert mean_diff_0 < 0.25, f"Qubit 0 failed. Mean diff: {mean_diff_0}"
    assert mean_diff_1 < 0.05, f"Qubit 1 failed. Mean diff: {mean_diff_1}"


def test_mcwf_zero_strength_noise() -> None:
    """Test MCWF with zero strength noise process (should be ignored)."""
    n_sites = 2
    psi = MPS(n_sites)
    h = MPO.ising(n_sites, J=1.0, g=1.0)
    # Define noise with 0 strength
    noise = NoiseModel(processes=[{"name": "lowering", "sites": [0], "strength": 0.0}])

    sim_params = AnalogSimParams(dt=0.1, elapsed_time=0.1, solver="MCWF", observables=[Observable("z", sites=[0])])

    # Preprocess should not add any jump ops
    ctx = preprocess_mcwf(psi, h, noise, sim_params)
    assert len(ctx.jump_ops) == 0




def test_mcwf_diagnostic_observables() -> None:
    """Test that diagnostic observables are handled (converted to None/0.0) in MCWF."""
    n_sites = 2
    psi = MPS(n_sites)
    h = MPO.ising(n_sites, J=1.0, g=1.0)

    # "runtime_cost" is a special diagnostic observable name
    obs_diag = Observable("runtime_cost", sites=[])
    # Also add a real observable to verify mixing
    obs_real = Observable("z", sites=[0])

    sim_params = AnalogSimParams(dt=0.1, elapsed_time=0.1, solver="MCWF", observables=[obs_diag, obs_real])

    # MCWF Preprocess
    ctx = preprocess_mcwf(psi, h, None, sim_params)

    # Check that we have one None and one array in embedded_observables
    assert any(op is None for op in ctx.embedded_observables)
    assert any(op is not None for op in ctx.embedded_observables)

    # Identify the index of the diagnostic observable
    diag_idx = -1
    for i, obs in enumerate(sim_params.sorted_observables):
        if obs.gate.name == "runtime_cost":
            diag_idx = i
            break
    assert diag_idx != -1
    assert ctx.embedded_observables[diag_idx] is None

    # Run MCWF
    res_mcwf = mcwf((0, ctx))
    # Result for diagnostic should be 0.0
    assert np.allclose(res_mcwf[diag_idx, :], 0.0)
