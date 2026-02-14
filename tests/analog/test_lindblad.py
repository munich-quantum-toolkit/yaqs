# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Exact Lindblad Solver."""

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.simulator import run


def test_lindblad_amplitude_damping() -> None:
    """Test single qubit amplitude damping with exact Lindblad solver."""
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
    dt = 0.05  # Reduced dt for better accuracy
    obs = Observable("z", sites=[0])

    sim_params = AnalogSimParams(
        observables=[obs],
        elapsed_time=t_max,
        dt=dt,
        solver="Lindblad",
        num_traj=1,  # Deterministic
    )

    run(initial_state, hamiltonian, sim_params, noise_model)

    times = sim_params.times
    sigma_z_sim = obs.results
    assert sigma_z_sim is not None
    delta_exact = 1 - 2 * np.exp(-gamma * times)

    # Comparison
    diff = np.abs(sigma_z_sim - delta_exact)
    assert np.all(diff < 1e-4), f"Max diff: {np.max(diff)}"


def test_lindblad_unitary_rabi() -> None:
    """Test single qubit unitary evolution (Rabi oscillation) with no noise."""
    n_sites = 1
    initial_state = MPS(n_sites, state="zeros")  # |0>

    hamiltonian = MPO.ising(n_sites, J=0.0, g=-1.0)
    # MPO.ising returns H = -J ZZ - g X.
    # We want H = +1.0 * X. So set g = -1.0.

    t_max = 2.0 * np.pi
    dt = 0.05
    obs = Observable("z", sites=[0])

    sim_params = AnalogSimParams(observables=[obs], elapsed_time=t_max, dt=dt, solver="Lindblad")

    run(initial_state, hamiltonian, sim_params, None)

    times = sim_params.times
    sigma_z_sim = obs.results
    assert sigma_z_sim is not None
    sigma_z_exact = np.cos(2 * times)

    diff = np.abs(sigma_z_sim - sigma_z_exact)
    assert np.all(diff < 1e-4), f"Max diff: {np.max(diff)}"


def test_lindblad_dephasing() -> None:
    """Test 2-qubit system with local dephasing on one qubit."""
    n_sites = 2
    initial_state = MPS(n_sites, state="x+")  # |++> = (|0>+|1>)(|0>+|1>)/2

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

    sim_params = AnalogSimParams(observables=[obs0, obs1], elapsed_time=t_max, dt=dt, solver="Lindblad")

    run(initial_state, hamiltonian, sim_params, noise_model)

    times = sim_params.times
    x0_sim = obs0.results
    x1_sim = obs1.results
    assert x0_sim is not None
    assert x1_sim is not None

    x0_exact = np.exp(-2 * gamma * times)
    x1_exact = np.ones_like(times)

    assert np.allclose(x0_sim, x0_exact, atol=1e-4), f"Qubit 0 failed. Max diff: {np.max(np.abs(x0_sim - x0_exact))}"
    assert np.allclose(x1_sim, x1_exact, atol=1e-4), f"Qubit 1 failed. Max diff: {np.max(np.abs(x1_sim - x1_exact))}"


def test_lindblad_dephasing_both_qubits() -> None:
    """Test 2-qubit system with dephasing on both qubits (multiple jump operators).

    This test exposes potential anti-commutator over-counting bugs when
    multiple jump operators are present. Both qubits should decay identically.
    """
    n_sites = 2
    initial_state = MPS(n_sites, state="x+")  # |++> = (|0>+|1>)(|0>+|1>)/2

    hamiltonian = MPO()
    hamiltonian.identity(n_sites)
    for i in range(len(hamiltonian.tensors)):
        hamiltonian.tensors[i] *= 0.0

    # Dephasing on BOTH qubits with the same gamma and sigma_z
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    gamma = 0.5
    noise_processes = [
        {"name": "dephasing_0", "sites": [0], "strength": gamma, "matrix": sigma_z},
        {"name": "dephasing_1", "sites": [1], "strength": gamma, "matrix": sigma_z},
    ]

    noise_model = NoiseModel(processes=noise_processes)

    t_max = 2.0
    dt = 0.05
    obs0 = Observable("x", sites=[0])
    obs1 = Observable("x", sites=[1])

    sim_params = AnalogSimParams(observables=[obs0, obs1], elapsed_time=t_max, dt=dt, solver="Lindblad")

    run(initial_state, hamiltonian, sim_params, noise_model)

    times = sim_params.times
    x0_sim = obs0.results
    x1_sim = obs1.results
    assert x0_sim is not None
    assert x1_sim is not None

    # Both qubits should decay identically
    x_exact = np.exp(-2 * gamma * times)

    assert np.allclose(x0_sim, x_exact, atol=1e-4), f"Qubit 0 failed. Max diff: {np.max(np.abs(x0_sim - x_exact))}"
    assert np.allclose(x1_sim, x_exact, atol=1e-4), f"Qubit 1 failed. Max diff: {np.max(np.abs(x1_sim - x_exact))}"


def test_lindblad_system_size_error() -> None:
    """Test that Lindblad solver raises error for large systems."""
    n_sites = 11
    initial_state = MPS(n_sites)
    hamiltonian = MPO()
    hamiltonian.identity(n_sites)
    sim_params = AnalogSimParams(solver="Lindblad", get_state=True)

    with pytest.raises(ValueError, match="System size too large"):
        run(initial_state, hamiltonian, sim_params, None)


def test_lindblad_zero_strength_noise() -> None:
    """Test Lindblad with zero strength noise process."""
    n_sites = 2
    psi = MPS(n_sites)
    h = MPO.ising(n_sites, J=1.0, g=1.0)
    noise = NoiseModel(processes=[{"name": "lowering", "sites": [0], "strength": 0.0}])

    obs = Observable("z", sites=[0])
    sim_params = AnalogSimParams(dt=0.1, elapsed_time=0.1, solver="Lindblad", observables=[obs])

    args = (0, psi, noise, sim_params, h)
    # Should run without error
    lindblad(args)


def test_lindblad_system_size_limit() -> None:
    """Test that Lindblad raises ValueError for system size > 12."""
    n_sites = 13
    psi = MPS(n_sites)
    h = MPO.ising(n_sites, J=1.0, g=1.0)
    sim_params = AnalogSimParams(dt=0.1, elapsed_time=1.0, observables=[Observable("z", sites=[0])])

    # Lindblad args: (traj_idx, psi, noise_model, sim_params, hamiltonian)
    args = (0, psi, None, sim_params, h)

    with pytest.raises(ValueError, match="System size too large"):
        lindblad(args)


def test_lindblad_diagnostic_observables() -> None:
    """Test that diagnostic observables are handled (converted to None/0.0) in Lindblad."""
    n_sites = 2
    psi = MPS(n_sites)
    h = MPO.ising(n_sites, J=1.0, g=1.0)

    # "runtime_cost" is a special diagnostic observable name
    obs_diag = Observable("runtime_cost", sites=[])
    # Also add a real observable to verify mixing
    obs_real = Observable("z", sites=[0])

    sim_params = AnalogSimParams(dt=0.1, elapsed_time=0.1, solver="Lindblad", observables=[obs_diag, obs_real])

    # Lindblad args: (traj_idx, psi, noise_model, sim_params, hamiltonian)
    args = (0, psi, None, sim_params, h)
    res_lindblad = lindblad(args)

    # Identify the index of the diagnostic observable
    diag_idx = -1
    for i, obs in enumerate(sim_params.sorted_observables):
        if obs.gate.name == "runtime_cost":
            diag_idx = i
            break

    assert diag_idx != -1
    # Result for diagnostic should be 0.0
    assert np.all(res_lindblad[diag_idx, :] == 0.0)
