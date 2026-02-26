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
    calculate_dual_choi_basis,
    get_basis_states,
    get_choi_basis,
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
    assert np.all(np.isfinite(pt.tensor))
    assert np.all(pt.weights >= 0)


def test_tomography_run_defaults() -> None:
    """Verify tomography run with default timesteps."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1, elapsed_time=0.2)
    timesteps = [0.1, 0.1]

    pt = run(op, params, timesteps=timesteps)
    assert pt.tensor.shape == (4, 16, 16)
    assert np.all(np.isfinite(pt.tensor))

    # When timesteps is not provided, it defaults to [params.elapsed_time], which is 1 step of 0.2
    pt_default = run(op, params)
    assert pt_default.tensor.shape == (4, 16)
    assert np.all(np.isfinite(pt_default.tensor))


def test_tomography_mcwf_multistep() -> None:
    """Verify tomography with MCWF solver and multiple steps (vector interventions)."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)

    params = AnalogSimParams(dt=0.1, order=1, solver="MCWF")
    timesteps = [0.1, 0.1]

    pt = run(op, params, timesteps=timesteps)
    assert pt.tensor.shape == (4, 16, 16)
    assert np.all(np.isfinite(pt.tensor))


def test_tomography_run_multistep() -> None:
    """Integration test for multi-step tomography.run() API."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1)
    timesteps = [0.1, 0.1]

    pt = run(op, params, timesteps=timesteps)

    assert pt.tensor.shape == (4, 16, 16)
    assert np.all(np.isfinite(pt.tensor))


def test_basis_reproduction() -> None:
    """Verify tomography recovers specific basis trajectories exactly."""
    l_size = 1
    op = MPO.ising(length=l_size, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, elapsed_time=0.1)
    pt = run(op, params)

    # Choose a specific alpha sequence (e.g., alpha=5)
    alpha = 5
    p, m = pt.choi_indices[alpha]
    
    basis_set = get_basis_states()
    rho_p = basis_set[p][2]
    E_m = basis_set[m][2]

    # Build CP map A_alpha using np.trace(E_m @ rho) * rho_p
    def A_alpha(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.trace(E_m @ rho) * rho_p
        
    # Predict using the constructed instrument map
    rho_pred = pt.predict_final_state([A_alpha])
    
    # Compare against stored unnormalized branch output
    stored_rho = pt.tensor[:, alpha].reshape(2, 2)
    np.testing.assert_allclose(rho_pred, stored_rho, atol=1e-10)


def test_predict_linearity() -> None:
    """Verify multilinear comb behavior of predict_final_state."""
    l_size = 1
    op = MPO.ising(length=l_size, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, elapsed_time=0.1)
    pt = run(op, params)
    
    rng = np.random.default_rng(42)
    
    u1 = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    def random_map1(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return u1 @ rho @ u1.conj().T
        
    u2 = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    def random_map2(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return u2 @ rho @ u2.conj().T
        
    a, b = 0.5, 0.5
    
    def combined_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return a * random_map1(rho) + b * random_map2(rho)
        
    pred1 = pt.predict_final_state([random_map1])
    pred2 = pt.predict_final_state([random_map2])
    pred_comb = pt.predict_final_state([combined_map])
    
    np.testing.assert_allclose(pred_comb, a * pred1 + b * pred2, atol=1e-10)


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

    basis_set = get_basis_states()
    initial_rho = basis_set[0][2]

    # CP map for initial prep
    def prep_initial(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.trace(rho) * initial_rho

    rho_pred = pt.predict_final_state([prep_initial])

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

    basis_set = get_basis_states()
    initial_rho = basis_set[0][2]

    def prep_initial(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.trace(rho) * initial_rho

    rho_pred = pt.predict_final_state([prep_initial])
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
    """Verify PT prediction accuracy for a held-out initial state (1-step sanity check).

    Uses a single evolution step, which PT can predict to machine precision
    (no Trotter vs exact-expm ambiguity at the boundary).
    """
    from scipy.linalg import expm
    rng = np.random.default_rng(42)
    l_size = 2
    op = MPO.ising(length=l_size, J=1.0, g=0.5)

    params = AnalogSimParams(
        dt=0.1,
        max_bond_dim=16,
        order=1,
    )
    timesteps = [0.1]   # single step

    pt = run(op, params, timesteps=timesteps)

    rho_0 = _get_random_rho(rng)

    def initial_prep(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.trace(rho) * rho_0

    # PT prediction
    rho_pred = pt.predict_final_state([initial_prep])

    # Exact direct simulation: start with rho_0 ⊗ |0><0|, evolve once, trace over site 1
    # MPO.to_matrix() for 2 qubits uses kron(site0, site1) with site 0 as the left factor
    H = op.to_matrix()
    u_evol = expm(-1j * H * 0.1)

    env = np.array([[1, 0], [0, 0]], dtype=complex)
    rho_full = np.kron(rho_0, env)
    rho_full = u_evol @ rho_full @ u_evol.conj().T
    rho_site0 = np.trace(rho_full.reshape(2, 2, 2, 2), axis1=1, axis2=3)

    np.testing.assert_allclose(rho_pred, rho_site0, atol=1e-6)


def test_dual_frame_biorthogonality() -> None:
    """Verify that the dual frame {D_k} and basis {B_k} are biorthogonal: Tr(D_i^dag B_j) = delta_{ij}."""
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    for i in range(16):
        for j in range(16):
            inner = np.trace(duals[i].conj().T @ choi_basis[j])
            expected = 1.0 if i == j else 0.0
            np.testing.assert_allclose(inner, expected, atol=1e-10)


def test_dual_frame_reconstruction() -> None:
    """Verify that any random Choi matrix J can be reconstructed via the dual frame."""
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    rng = np.random.default_rng(42)
    # Random 4x4 matrix
    j_mat = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))

    # Coefficients: c_k = Tr(D_k^dag @ J)
    coeffs = [np.trace(d.conj().T @ j_mat) for d in duals]

    # Reconstruct: J_rec = sum_k c_k * B_k
    j_rec = np.zeros((4, 4), dtype=complex)
    for c, b in zip(coeffs, choi_basis):
        j_rec += c * b

    np.testing.assert_allclose(j_rec, j_mat, atol=1e-10)
