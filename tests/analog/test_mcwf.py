# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Monte Carlo Wavefunction (MCWF) Solver."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

import mqt.yaqs.analog.mcwf as mcwf_mod
from mqt.yaqs import (
    AnalogSimParams,
    Hamiltonian,
    NoiseModel,
    Observable,
    Simulator,
    State,
)
from mqt.yaqs.analog.mcwf import MAX_PRECOMPUTE_DIM, preprocess_mcwf
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from tests.conftest import YAQS_TEST_SEED


def test_mcwf_amplitude_damping() -> None:
    """Test single qubit amplitude damping with MCWF solver.

    Checks if the average of many trajectories matches the analytical solution.
    """
    n_sites = 1
    initial_state = State(n_sites, initial="ones", representation="vector")  # |1>
    mpo = MPO.identity(n_sites)
    for i in range(len(mpo.tensors)):
        mpo.tensors[i] *= 0.0  # H = 0
    hamiltonian = Hamiltonian.from_mpo(mpo)

    # Noise: Amplitude Damping
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    gamma = 1.0
    noise_processes = [{"name": "destroy", "sites": [0], "strength": gamma, "matrix": sigma_minus}]
    noise_model = NoiseModel(processes=noise_processes)

    t_max = 2.0
    dt = 0.05
    obs = Observable("z", sites=[0])

    num_traj = 100
    sim_params = AnalogSimParams(
        observables=[obs],
        elapsed_time=t_max,
        dt=dt,
        num_traj=num_traj,
        random_seed=YAQS_TEST_SEED,
    )

    result = Simulator(parallel=False, show_progress=False).run(initial_state, hamiltonian, sim_params, noise_model)

    times = sim_params.times
    sigma_z_sim = result.expectation_values[0]
    assert sigma_z_sim is not None

    # Analytical solution for <sigma_z>:
    # rho_11(t) = exp(-gamma t)
    # rho_00(t) = 1 - exp(-gamma t)
    # <Z> = rho_00 - rho_11 = 1 - 2 exp(-gamma t)
    delta_exact = 1 - 2 * np.exp(-gamma * times)

    # Comparison (allow larger tolerance due to stochastic noise ~ 1/sqrt(N))
    # 1/sqrt(100) ~ 0.1. 3 sigma ~ 0.3; mean check uses 0.25 margin.
    # Comparison (using mean error for robustness against stochastic fluctuations)
    mean_diff = np.mean(np.abs(sigma_z_sim - delta_exact))
    assert mean_diff < 0.25, f"Mean diff: {mean_diff}"


def test_mcwf_unitary_rabi() -> None:
    """Test single qubit unitary evolution (Rabi oscillation) with no noise.

    Should be deterministic and match exact solution tightly.
    """
    n_sites = 1
    initial_state = State(n_sites, initial="zeros", representation="vector")  # |0>

    hamiltonian = Hamiltonian.ising(n_sites, J=0.0, g=-1.0)

    t_max = 2.0 * np.pi
    dt = 0.01
    obs = Observable("z", sites=[0])

    sim_params = AnalogSimParams(
        observables=[obs],
        elapsed_time=t_max,
        dt=dt,
        num_traj=1,  # Deterministic
    )

    result = Simulator(show_progress=False).run(initial_state, hamiltonian, sim_params, None)

    times = sim_params.times
    sigma_z_sim = result.expectation_values[0]
    assert sigma_z_sim is not None
    sigma_z_exact = np.cos(2 * times)

    # Should match very well
    diff = np.abs(sigma_z_sim - sigma_z_exact)
    assert np.all(diff < 1e-4), f"Max diff: {np.max(diff)}"


def test_mcwf_dephasing() -> None:
    """Test 2-qubit system with local dephasing on one qubit."""
    n_sites = 2
    initial_state = State(n_sites, initial="x+", representation="vector")  # |++>

    mpo = MPO.identity(n_sites)
    for i in range(len(mpo.tensors)):
        mpo.tensors[i] *= 0.0
    hamiltonian = Hamiltonian.from_mpo(mpo)

    # Dephasing on qubit 0 (sigma_z)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    gamma = 0.5
    noise_processes = [{"name": "dephasing", "sites": [0], "strength": gamma, "matrix": sigma_z}]

    noise_model = NoiseModel(processes=noise_processes)

    t_max = 2.0
    dt = 0.05
    obs0 = Observable("x", sites=[0])
    obs1 = Observable("x", sites=[1])

    num_traj = 100
    sim_params = AnalogSimParams(
        observables=[obs0, obs1],
        elapsed_time=t_max,
        dt=dt,
        num_traj=num_traj,
        random_seed=YAQS_TEST_SEED,
    )

    # Use parallel=True to verify infrastructure
    result = Simulator(parallel=True, show_progress=False).run(initial_state, hamiltonian, sim_params, noise_model)

    times = sim_params.times
    x0_sim = result.expectation_values[0]
    x1_sim = result.expectation_values[1]
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

    sim_params = AnalogSimParams(dt=0.1, elapsed_time=0.1, observables=[Observable("z", sites=[0])])

    ctx = preprocess_mcwf(psi, h, noise, sim_params)
    assert len(ctx.jump_ops) == 0
    assert ctx.is_unitary
    dim = 2**n_sites
    assert ctx.step_propagator is not None
    assert ctx.step_propagator.shape == (dim, dim)


def test_preprocess_mcwf_rejects_mismatched_h_sparse_shape() -> None:
    """h_sparse must match the Hilbert dimension implied by the initial state."""
    psi = MPS(2, state="zeros")
    sim_params = AnalogSimParams(dt=0.1, elapsed_time=0.1, observables=[])
    bad_h = scipy.sparse.eye(8, format="csr")
    with pytest.raises(ValueError, match=r"h_sparse must have shape \(4, 4\)"):
        preprocess_mcwf(psi, None, None, sim_params, h_sparse=bad_h)


def test_preprocess_mcwf_sets_propagator_small_system() -> None:
    """Small systems precompute a fixed time-step propagator."""
    n_sites = 3
    psi = MPS(n_sites, state="zeros")
    h = MPO.ising(n_sites, J=1.0, g=0.5)
    sim_params = AnalogSimParams(
        dt=0.05,
        elapsed_time=0.1,
        observables=[Observable("z", sites=[0])],
    )
    ctx = preprocess_mcwf(psi, h, None, sim_params)
    dim = 2**n_sites
    assert dim <= MAX_PRECOMPUTE_DIM
    assert ctx.step_propagator is not None
    assert ctx.step_propagator.shape == (dim, dim)
    assert ctx.is_unitary


def test_mcwf_noisy_system_has_propagator() -> None:
    """Open-system runs on small Hilbert spaces also use the precomputed propagator."""
    n_sites = 2
    psi = MPS(n_sites, state="x+")
    h = MPO.identity(n_sites)
    for i in range(len(h.tensors)):
        h.tensors[i] *= 0.0
    noise = NoiseModel(processes=[{"name": "pauli_z", "sites": [0], "strength": 0.2}])
    sim_params = AnalogSimParams(dt=0.1, elapsed_time=0.1, observables=[])
    ctx = preprocess_mcwf(psi, h, noise, sim_params)
    assert not ctx.is_unitary
    assert ctx.step_propagator is not None


def test_mcwf_result_has_no_auto_diagnostics() -> None:
    """Vector MCWF runs do not populate Result bond diagnostics."""
    length = 2
    state = State(length, initial="zeros", representation="vector")
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
    sim_params = AnalogSimParams(
        observables=[Observable("z", sites=0)],
        elapsed_time=0.1,
        dt=0.1,
        sample_timesteps=False,
    )
    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params)
    assert result.runtime_cost is None
    assert result.max_bond is None
    assert result.total_bond is None


def test_mcwf_trajectory_rng_seeding_reproducible() -> None:
    """Two runs with the same ``random_seed`` produce identical MCWF trajectories."""
    n_sites = 1
    state = State(n_sites, initial="x+", representation="vector")
    mpo = MPO.identity(n_sites)
    for i in range(len(mpo.tensors)):
        mpo.tensors[i] *= 0.0
    hamiltonian = Hamiltonian.from_mpo(mpo)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    noise = NoiseModel(processes=[{"name": "dephasing", "sites": [0], "strength": 2.0, "matrix": sigma_z}])

    def _params() -> AnalogSimParams:
        return AnalogSimParams(
            dt=0.02,
            elapsed_time=0.2,
            observables=[Observable("x", sites=[0])],
            num_traj=4,
            sample_timesteps=True,
            random_seed=YAQS_TEST_SEED,
        )

    sim = Simulator(parallel=False, show_progress=False)
    res_a = sim.run(state, hamiltonian, _params(), noise)
    res_b = sim.run(state, hamiltonian, _params(), noise)
    np.testing.assert_allclose(res_a.expectation_values[0], res_b.expectation_values[0])


def test_mcwf_noisy_evolution_with_propagator_via_simulator() -> None:
    """Noisy open-system evolution through the Simulator returns finite trajectories."""
    n_sites = 2
    state = State(n_sites, initial="x+", representation="vector")
    mpo = MPO.identity(n_sites)
    for i in range(len(mpo.tensors)):
        mpo.tensors[i] *= 0.0
    hamiltonian = Hamiltonian.from_mpo(mpo)
    noise = NoiseModel(processes=[{"name": "pauli_z", "sites": [0], "strength": 0.3}])

    sim_params = AnalogSimParams(
        dt=0.05,
        elapsed_time=0.15,
        observables=[Observable("z", sites=[0])],
        num_traj=2,
        sample_timesteps=True,
        random_seed=YAQS_TEST_SEED,
    )
    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params, noise)
    expectation = result.expectation_values[0]
    assert expectation is not None
    assert expectation.shape == (len(sim_params.times),)
    assert np.all(np.isfinite(expectation))


def test_mcwf_unitary_krylov_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``U_step`` is not precomputed, unitary evolution uses ``expm_krylov`` per step."""
    monkeypatch.setattr(mcwf_mod, "MAX_PRECOMPUTE_DIM", 4)
    n_sites = 3
    state = State(n_sites, initial="zeros", representation="vector")
    hamiltonian = Hamiltonian.ising(n_sites, J=1.0, g=0.5)

    sim_params = AnalogSimParams(
        dt=0.05,
        elapsed_time=0.1,
        observables=[Observable("z", sites=[0])],
        num_traj=1,
        sample_timesteps=True,
    )
    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params, None)
    expectation = result.expectation_values[0]
    assert expectation is not None
    assert expectation.shape == (len(sim_params.times),)
    assert np.all(np.isfinite(expectation))


def test_mcwf_noisy_arnoldi_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``U_step`` is not precomputed, noisy evolution falls back to ``expm_arnoldi``."""
    monkeypatch.setattr(mcwf_mod, "MAX_PRECOMPUTE_DIM", 4)
    n_sites = 3
    state = State(n_sites, initial="x+", representation="vector")
    mpo = MPO.identity(n_sites)
    for i in range(len(mpo.tensors)):
        mpo.tensors[i] *= 0.0
    hamiltonian = Hamiltonian.from_mpo(mpo)
    noise = NoiseModel(processes=[{"name": "pauli_z", "sites": [0], "strength": 0.2}])

    sim_params = AnalogSimParams(
        dt=0.05,
        elapsed_time=0.1,
        observables=[Observable("z", sites=[0])],
        num_traj=1,
        sample_timesteps=True,
        random_seed=YAQS_TEST_SEED,
    )
    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params, noise)
    expectation = result.expectation_values[0]
    assert expectation is not None
    assert expectation.shape == (len(sim_params.times),)
    assert np.all(np.isfinite(expectation))
