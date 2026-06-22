# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Exact Lindblad Solver."""

from __future__ import annotations

import numpy as np
import pytest

import mqt.yaqs.analog.lindblad as lindblad_mod
from mqt.yaqs import (
    AnalogSimParams,
    Hamiltonian,
    NoiseModel,
    Observable,
    Simulator,
    State,
)
from mqt.yaqs.analog.lindblad import (
    MAX_LIOUVILLIAN_VECTOR_DIM,
    _rho_vec_at_elapsed_time,
    lindblad,
    lindblad_evolve,
    preprocess_lindblad,
)
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS


def test_lindblad_amplitude_damping() -> None:
    """Test single qubit amplitude damping with exact Lindblad solver."""
    n_sites = 1
    initial_state = State(n_sites, initial="ones", representation="density_matrix")
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
    dt = 0.05  # Reduced dt for better accuracy
    obs = Observable("z", sites=[0])

    sim_params = AnalogSimParams(
        observables=[obs],
        elapsed_time=t_max,
        dt=dt,
        num_traj=1,  # Deterministic
    )

    result = Simulator(show_progress=False).run(initial_state, hamiltonian, sim_params, noise_model)

    times = sim_params.times
    sigma_z_sim = result.expectation_values[0]
    assert sigma_z_sim is not None
    delta_exact = 1 - 2 * np.exp(-gamma * times)

    # Comparison
    diff = np.abs(sigma_z_sim - delta_exact)
    assert np.all(diff < 1e-4), f"Max diff: {np.max(diff)}"


def test_lindblad_unitary_rabi() -> None:
    """Test single qubit unitary evolution (Rabi oscillation) with no noise."""
    n_sites = 1
    initial_state = State(n_sites, initial="zeros", representation="density_matrix")

    hamiltonian = Hamiltonian.ising(n_sites, J=0.0, g=-1.0)
    # MPO.ising returns H = -J ZZ - g X.
    # We want H = +1.0 * X. So set g = -1.0.

    t_max = 2.0 * np.pi
    dt = 0.05
    obs = Observable("z", sites=[0])

    sim_params = AnalogSimParams(observables=[obs], elapsed_time=t_max, dt=dt)

    result = Simulator(show_progress=False).run(initial_state, hamiltonian, sim_params, None)

    times = sim_params.times
    sigma_z_sim = result.expectation_values[0]
    assert sigma_z_sim is not None
    sigma_z_exact = np.cos(2 * times)

    diff = np.abs(sigma_z_sim - sigma_z_exact)
    assert np.all(diff < 1e-4), f"Max diff: {np.max(diff)}"


def test_lindblad_dephasing() -> None:
    """Test 2-qubit system with local dephasing on one qubit."""
    n_sites = 2
    initial_state = State(n_sites, initial="x+", representation="density_matrix")

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

    sim_params = AnalogSimParams(observables=[obs0, obs1], elapsed_time=t_max, dt=dt)

    result = Simulator(show_progress=False).run(initial_state, hamiltonian, sim_params, noise_model)

    times = sim_params.times
    x0_sim = result.expectation_values[0]
    x1_sim = result.expectation_values[1]
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
    initial_state = State(n_sites, initial="x+", representation="density_matrix")

    mpo = MPO.identity(n_sites)
    for i in range(len(mpo.tensors)):
        mpo.tensors[i] *= 0.0
    hamiltonian = Hamiltonian.from_mpo(mpo)

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

    sim_params = AnalogSimParams(observables=[obs0, obs1], elapsed_time=t_max, dt=dt)

    result = Simulator(show_progress=False).run(initial_state, hamiltonian, sim_params, noise_model)

    times = sim_params.times
    x0_sim = result.expectation_values[0]
    x1_sim = result.expectation_values[1]
    assert x0_sim is not None
    assert x1_sim is not None

    # Both qubits should decay identically
    x_exact = np.exp(-2 * gamma * times)

    assert np.allclose(x0_sim, x_exact, atol=1e-4), f"Qubit 0 failed. Max diff: {np.max(np.abs(x0_sim - x_exact))}"
    assert np.allclose(x1_sim, x_exact, atol=1e-4), f"Qubit 1 failed. Max diff: {np.max(np.abs(x1_sim - x_exact))}"


def test_lindblad_zero_strength_noise_runs_via_simulator() -> None:
    """Zero-strength noise processes are pruned; the Simulator path still produces results."""
    n_sites = 2
    psi = MPS(n_sites)
    h = MPO.ising(n_sites, J=1.0, g=1.0)
    noise = NoiseModel(processes=[{"name": "lowering", "sites": [0], "strength": 0.0}])

    obs = Observable("z", sites=[0])
    sim_params = AnalogSimParams(dt=0.1, elapsed_time=0.1, observables=[obs])

    ctx = preprocess_lindblad(psi, h, noise, sim_params)
    assert len(ctx.jump_ops) == 0
    assert ctx.is_unitary
    dim = 2**n_sites
    assert dim * dim <= MAX_LIOUVILLIAN_VECTOR_DIM
    assert ctx.step_propagator is not None
    assert ctx.step_propagator.shape == (dim * dim, dim * dim)

    state = State(n_sites, initial="zeros", representation="density_matrix")
    hamiltonian = Hamiltonian.from_mpo(h)
    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params, noise)
    assert result.expectation_values[0] is not None


def test_lindblad_result_has_no_auto_diagnostics() -> None:
    """Density-matrix Lindblad runs do not populate Result bond diagnostics."""
    length = 2
    state = State(length, initial="zeros", representation="density_matrix")
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


def test_preprocess_lindblad_sets_propagator_small_system() -> None:
    """Small systems precompute a fixed Liouvillian step propagator."""
    n_sites = 3
    psi = MPS(n_sites, state="zeros")
    h = MPO.ising(n_sites, J=1.0, g=0.5)
    sim_params = AnalogSimParams(
        dt=0.05,
        elapsed_time=0.1,
        observables=[Observable("z", sites=[0])],
    )
    ctx = preprocess_lindblad(psi, h, None, sim_params)
    vec_dim = (2**n_sites) ** 2
    assert vec_dim <= MAX_LIOUVILLIAN_VECTOR_DIM
    assert ctx.step_propagator is not None
    assert ctx.step_propagator.shape == (vec_dim, vec_dim)
    assert ctx.is_unitary


def test_lindblad_noisy_small_system_has_propagator() -> None:
    """Open-system runs on small Hilbert spaces also use the precomputed propagator."""
    n_sites = 2
    psi = MPS(n_sites, state="x+")
    h = MPO.identity(n_sites)
    for i in range(len(h.tensors)):
        h.tensors[i] *= 0.0
    noise = NoiseModel(processes=[{"name": "pauli_z", "sites": [0], "strength": 0.2}])
    sim_params = AnalogSimParams(dt=0.1, elapsed_time=0.1, observables=[])
    ctx = preprocess_lindblad(psi, h, noise, sim_params)
    assert not ctx.is_unitary
    assert ctx.step_propagator is not None


def test_noiseless_mps_matches_density_matrix() -> None:
    """Noiseless Hamiltonian evolution agrees between mps and density_matrix representations."""
    n_sites = 3
    psi_mps = State(n_sites, initial="zeros", representation="mps")
    psi_rho = State(n_sites, initial="zeros", representation="density_matrix")
    h = Hamiltonian.ising(n_sites, J=1.0, g=0.5)
    obs = Observable("z", sites=[0])
    t_max = 0.5
    dt = 0.1

    params_mps = AnalogSimParams(
        observables=[obs],
        elapsed_time=t_max,
        dt=dt,
        max_bond_dim=32,
    )
    sim = Simulator(show_progress=False)
    result_mps = sim.run(psi_mps, h, params_mps, None)
    assert result_mps.expectation_values[0] is not None
    z_mps = result_mps.expectation_values[0][-1]

    obs_rho = Observable("z", sites=[0])
    params_rho = AnalogSimParams(
        observables=[obs_rho],
        elapsed_time=t_max,
        dt=dt,
    )
    result_rho = sim.run(psi_rho, h, params_rho, None)
    assert result_rho.expectation_values[0] is not None
    z_rho = result_rho.expectation_values[0][-1]

    assert z_mps is not None
    assert z_rho is not None
    assert np.isclose(z_mps, z_rho, atol=1e-4), f"mps={z_mps}, density_matrix={z_rho}"


def test_lindblad_propagator_records_all_timepoints() -> None:
    """Propagator path records observables at every entry in ``sim_params.times``."""
    n_sites = 1
    psi = MPS(n_sites, state="ones")
    h = MPO.identity(n_sites)
    for i in range(len(h.tensors)):
        h.tensors[i] *= 0.0

    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    noise = NoiseModel(processes=[{"name": "destroy", "sites": [0], "strength": 1.0, "matrix": sigma_minus}])
    obs = Observable("z", sites=[0])
    sim_params = AnalogSimParams(
        observables=[obs],
        elapsed_time=0.2,
        dt=0.05,
        sample_timesteps=True,
    )
    ctx = preprocess_lindblad(psi, h, noise, sim_params)
    assert ctx.step_propagator is not None

    state = State(n_sites, initial="ones", representation="density_matrix")
    hamiltonian = Hamiltonian.from_mpo(h)
    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params, noise)
    expectation = result.expectation_values[0]
    assert expectation is not None
    assert expectation.shape == (len(sim_params.times),)
    assert np.all(np.isfinite(expectation))
    assert not np.isclose(expectation[0], expectation[-1])


def test_lindblad_ode_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``vec(rho)`` is too large for ``exp(L dt)``, the Simulator falls back to RK45."""
    monkeypatch.setattr(lindblad_mod, "MAX_LIOUVILLIAN_VECTOR_DIM", 4)
    n_sites = 2
    psi = MPS(n_sites, state="ones")
    h = MPO.identity(n_sites)
    for i in range(len(h.tensors)):
        h.tensors[i] *= 0.0

    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    gamma = 1.0
    noise = NoiseModel(processes=[{"name": "destroy", "sites": [0], "strength": gamma, "matrix": sigma_minus}])
    obs = Observable("z", sites=[0])
    sim_params = AnalogSimParams(
        observables=[obs],
        elapsed_time=0.2,
        dt=0.05,
        sample_timesteps=True,
        get_state=True,
    )

    ctx = preprocess_lindblad(psi, h, noise, sim_params)
    assert ctx.step_propagator is None

    state = State(n_sites, initial="ones", representation="density_matrix")
    hamiltonian = Hamiltonian.from_mpo(h)
    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params, noise)
    expectation = result.expectation_values[0]
    assert expectation is not None
    assert expectation.shape == (len(sim_params.times),)
    assert np.all(np.isfinite(expectation))
    assert result.output_state is not None
    assert result.output_state.density_matrix.shape == (4, 4)


def test_lindblad_evolve_get_state_false_returns_no_matrix() -> None:
    """``lindblad_evolve`` omits the density matrix unless ``get_state`` is requested."""
    n_sites = 1
    psi = MPS(n_sites, state="ones")
    h = MPO.identity(n_sites)
    for i in range(len(h.tensors)):
        h.tensors[i] *= 0.0
    noise = NoiseModel(
        processes=[{"name": "destroy", "sites": [0], "strength": 1.0, "matrix": np.array([[0, 1], [0, 0]], dtype=complex)}],
    )
    sim_params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.1,
        dt=0.1,
        get_state=False,
        sample_timesteps=False,
    )
    ctx = preprocess_lindblad(psi, h, noise, sim_params)
    obs, diag, rho = lindblad_evolve(ctx)
    assert obs.shape == (1, 1)
    assert diag is None
    assert rho is None


def test_rho_vec_at_elapsed_time_returns_initial_state_at_zero() -> None:
    """``_rho_vec_at_elapsed_time`` returns the initial vector when ``elapsed_time`` is zero."""
    n_sites = 1
    psi = MPS(n_sites, state="ones")
    h = MPO.identity(n_sites)
    sim_params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.0,
        dt=0.1,
        get_state=True,
    )
    ctx = preprocess_lindblad(psi, h, None, sim_params)
    rho_vec = _rho_vec_at_elapsed_time(ctx)
    np.testing.assert_allclose(rho_vec, ctx.rho_initial)


def test_rho_vec_at_elapsed_time_fractional_step() -> None:
    """Fractional elapsed times use an extra ``expm(L * remainder)`` after full ``dt`` steps."""
    n_sites = 1
    initial_state = State(n_sites, initial="ones", representation="density_matrix")
    psi = MPS(n_sites, state="ones")
    h = MPO.identity(n_sites)
    for i in range(len(h.tensors)):
        h.tensors[i] *= 0.0
    hamiltonian = Hamiltonian.from_mpo(h)
    gamma = 1.0
    elapsed_time = 0.25
    noise = NoiseModel(
        processes=[{"name": "destroy", "sites": [0], "strength": gamma, "matrix": np.array([[0, 1], [0, 0]], dtype=complex)}],
    )
    sim_params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=elapsed_time,
        dt=0.1,
        get_state=True,
    )
    ctx = preprocess_lindblad(
        psi,
        hamiltonian.mpo,
        noise,
        sim_params,
        rho_initial=initial_state.density_matrix,
        num_sites=n_sites,
    )
    assert ctx.step_propagator is not None
    rho_vec = _rho_vec_at_elapsed_time(ctx)
    rho = rho_vec.reshape((2, 2), order="F")
    expected = np.array(
        [[1.0 - np.exp(-gamma * elapsed_time), 0.0], [0.0, np.exp(-gamma * elapsed_time)]],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(rho, expected, atol=1e-4)


def test_lindblad_entry_point_returns_density_matrix() -> None:
    """The ``lindblad`` worker entry point forwards ``get_state`` output."""
    n_sites = 1
    psi = MPS(n_sites, state="ones")
    h = MPO.identity(n_sites)
    for i in range(len(h.tensors)):
        h.tensors[i] *= 0.0
    noise = NoiseModel(
        processes=[{"name": "destroy", "sites": [0], "strength": 1.0, "matrix": np.array([[0, 1], [0, 0]], dtype=complex)}],
    )
    sim_params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.2,
        dt=0.1,
        get_state=True,
        sample_timesteps=False,
    )
    obs, diag, rho = lindblad((0, psi, noise, sim_params, h))
    assert obs.shape == (1, 1)
    assert diag is None
    assert rho is not None
    assert rho.shape == (2, 2)
    assert np.isclose(np.trace(rho), 1.0)
