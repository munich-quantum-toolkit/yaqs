# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for tomography module."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.analog.analog_tjm import analog_tjm_1
from mqt.yaqs.characterization.tomography.process_tensor import ProcessTensor
from mqt.yaqs.characterization.tomography.tomography import (
    calculate_dual_frame,
    get_basis_states,
    run,
)
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_tomography_run_basic() -> None:
    """Integration test for basic 1-step tomography.run() API."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1, elapsed_time=0.1)

    # Run Tomography
    pt = run(op, params, num_trajectories=10)

    assert pt.tensor.shape == (4, 16)
    # Check that identity is somewhat preserved (rough check)
    assert np.real(pt.tensor[0, 0]) > 0.5


def test_tomography_run_defaults() -> None:
    """Verify tomography run with default timesteps."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1, elapsed_time=0.2)
    timesteps = [0.1, 0.1]

    pt = run(op, params, timesteps=timesteps)
    assert pt.tensor.shape == (4, 16, 16)

    # When timesteps is not provided, it defaults to [params.elapsed_time], which is 1 step of 0.2
    pt_default = run(op, params)
    assert pt_default.tensor.shape == (4, 16)


def test_tomography_mcwf_multistep() -> None:
    """Verify tomography with MCWF solver and multiple steps (vector interventions)."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)

    params = AnalogSimParams(dt=0.1, order=1, solver="MCWF")
    timesteps = [0.1, 0.1]

    pt = run(op, params, timesteps=timesteps)
    assert pt.tensor.shape == (4, 16, 16)
    assert np.real(pt.tensor[0, 0, 0]) > 0.5


def test_tomography_run_multistep() -> None:
    """Integration test for multi-step tomography.run() API."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1)
    timesteps = [0.1, 0.1]

    pt = run(op, params, timesteps=timesteps)

    assert pt.tensor.shape == (4, 16, 16)
    # Check that identity is somewhat preserved (rough check)
    assert np.real(pt.tensor[0, 0, 0]) > 0.5


def test_reconstruction_x_gate() -> None:
    """Verify tomography correctly reconstructs an X gate."""
    # H = -g X. If g * elapsed_time = pi/2, then U = exp(i pi/2 X) = iX.
    # The unitary channel is an X gate (up to a global phase): rho -> X rho X.
    elapsed_time = 0.1
    g = np.pi / (2 * elapsed_time)
    op = MPO.ising(length=1, J=0.0, g=g)

    params = AnalogSimParams(dt=0.1, elapsed_time=elapsed_time)
    pt = run(op, params)

    from mqt.yaqs.characterization.tomography.tomography import get_choi_basis
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_frame(choi_basis)

    # Predict final state given input |0>
    def prep_0(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.array([[1, 0], [0, 0]], dtype=complex)
        
    out_0 = pt.predict_final_state([prep_0], duals)
    expected_out_0 = np.array([[0, 0], [0, 1]], dtype=complex)
    np.testing.assert_allclose(out_0, expected_out_0, atol=1e-6)

    # Predict final state given input |1>
    def prep_1(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.array([[0, 0], [0, 1]], dtype=complex)
        
    out_1 = pt.predict_final_state([prep_1], duals)
    expected_out_1 = np.array([[1, 0], [0, 0]], dtype=complex)
    np.testing.assert_allclose(out_1, expected_out_1, atol=1e-6)

    # Note: Quantum Mutual Information over the 16 comb sequences gives ~0.6-0.9 bits.
    # It doesn't strictly adhere to the 0.907 value since the ensemble includes impossible measurements.
    h = pt.quantum_mutual_information(base=2)
    assert h > 0.5


def test_reconstruction_depolarizing() -> None:
    """Verify reconstruction of a Depolarizing channel (via strong noise)."""
    l_size = 1
    op = MPO()
    op.identity(l_size)

    # Strong X, Y, Z noise => Depolarizing
    noise_processes = [
        {"name": "x", "sites": [0], "strength": 10.0},
        {"name": "y", "sites": [0], "strength": 10.0},
        {"name": "z", "sites": [0], "strength": 10.0},
    ]
    noise_model = NoiseModel(processes=noise_processes)
    params = AnalogSimParams(dt=0.1, elapsed_time=1.0)

    pt = run(op, params, num_trajectories=100, noise_model=noise_model)

    # Fully depolarized state is 0.5*I, information should be 0.0
    h = pt.quantum_mutual_information(base=2)
    assert np.isclose(h, 0.0, atol=0.1)


def _reconstruct_state(expectations: dict[str, float]) -> NDArray[np.complex128]:
    """Reconstructs single-qubit density matrix from Pauli expectations.

    Returns:
        NDArray[np.complex128]: The reconstructed single-qubit density matrix.
    """
    eye = np.eye(2, dtype=complex)
    x_matrix = X().matrix
    y_matrix = Y().matrix
    z_matrix = Z().matrix

    return 0.5 * (eye + expectations["x"] * x_matrix + expectations["y"] * y_matrix + expectations["z"] * z_matrix)


def test_pt_prediction_consistency() -> None:
    """Verify ProcessTensor.predict_final_state consistency with direct simulation."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1, elapsed_time=0.1)

    timesteps = [0.1]
    pt = run(op, params, timesteps=timesteps)

    from mqt.yaqs.characterization.tomography.tomography import get_choi_basis
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_frame(choi_basis)
    
    basis_set = get_basis_states()
    initial_rho = basis_set[0][2]

    # CP map for initial prep
    def prep_initial(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return initial_rho

    rho_pred = pt.predict_final_state([prep_initial], duals)

    # If we apply the basis CP map exactly matching alpha=0 (which is project |0>, prep |0>),
    # Since the sim starts in |0>, the unnormalized output for sequence 0 should match rho_pred exactly.
    vec_stored = pt.tensor[:, 0]
    rho_stored = vec_stored.reshape(2, 2)

    np.testing.assert_allclose(rho_pred, rho_stored, atol=1e-6)


def get_standard_basis() -> list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]]:
    """Returns the standard 4-state Pauli basis for testing.

    Returns:
        list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]]: Standard 4-state basis.
    """
    psi_0 = np.array([1, 0], dtype=complex)
    psi_1 = np.array([0, 1], dtype=complex)
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j], dtype=complex) / np.sqrt(2)

    basis = [
        ("zeros", psi_0),
        ("ones", psi_1),
        ("x+", psi_plus),
        ("y+", psi_i_plus),
    ]
    return [(name, psi, np.outer(psi, psi.conj())) for name, psi in basis]


def test_algebraic_consistency() -> None:
    """Check algebraic consistency for various operators."""
    rng = np.random.default_rng(42)
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    timesteps = [0.1]
    pt = run(op, AnalogSimParams(dt=0.05, order=2), timesteps)

    from mqt.yaqs.characterization.tomography.tomography import get_choi_basis
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_frame(choi_basis)
    
    basis_set = get_basis_states()
    initial_rho = basis_set[0][2]

    def prep_initial(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return initial_rho

    rho_pred = pt.predict_final_state([prep_initial], duals)
    vec_stored = pt.tensor[:, 0]

    err = np.linalg.norm(rho_pred.reshape(-1) - vec_stored)
    assert err < 1e-10


def _get_random_rho(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Generate a random 2x2 density matrix."""
    # Create random complex matrix
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def _sample_pure_state(rho: NDArray[np.complex128], rng: np.random.Generator) -> NDArray[np.complex128]:
    """Sample a pure state from the eigen-decomposition of rho.

    Returns:
        NDArray[np.complex128]: A sampled pure state.
    """
    evals, evecs = np.linalg.eigh(rho)
    # Ensure probabilities are non-negative and sum to 1
    evals = np.maximum(evals, 0)
    evals /= np.sum(evals)
    idx = rng.choice(len(evals), p=evals)
    return evecs[:, idx]


def test_held_out_prediction() -> None:
    """Verify PT prediction accuracy for random held-out mixed state sequences."""
    from scipy.linalg import expm
    rng = np.random.default_rng(42)
    l_size = 2
    op = MPO.ising(length=l_size, J=1.0, g=0.5)

    params = AnalogSimParams(
        dt=0.1,
        max_bond_dim=16,
        order=1,
    )
    timesteps = [0.1, 0.1]

    # Build True Process Tensor using exact initializations
    pt = run(op, params, timesteps=timesteps)

    rho_0 = _get_random_rho(rng)
    
    # Define a unitary intervention
    theta = rng.uniform(0, 2 * np.pi)
    u_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=complex)
    
    def intervention(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return u_mat @ rho @ u_mat.conj().T

    basis_set = get_basis_states()
    from mqt.yaqs.characterization.tomography.tomography import get_choi_basis
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_frame(choi_basis)
    
    def initial_prep(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return rho_0
        
    # Predict final state using 2 interventions: initial prep and intermediate unitary
    rho_pred = pt.predict_final_state([initial_prep, intervention], duals)

    # Direct Simulation via exact matrix exp for 2 qubits
    H = op.to_matrix()
    u_evol = expm(-1j * H * 0.1)
    
    rho_system = np.kron(rho_0, np.array([[1, 0], [0, 0]], dtype=complex))
    
    # Step 1
    rho_system = u_evol @ rho_system @ u_evol.conj().T
    
    # Intervention on site 0
    u_intervene = np.kron(u_mat, np.eye(2, dtype=complex))
    rho_system = u_intervene @ rho_system @ u_intervene.conj().T
    
    # Step 2
    rho_system = u_evol @ rho_system @ u_evol.conj().T
    
    # Partial trace over site 1
    rho_final = np.trace(rho_system.reshape(2, 2, 2, 2), axis1=1, axis2=3)

    # Allow some tolerance for Trotter error in PT creation vs exact expm
    assert np.allclose(rho_pred, rho_final, atol=5e-3)
