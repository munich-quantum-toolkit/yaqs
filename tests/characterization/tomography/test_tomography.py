# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
"""Tests for the Process Tomography Module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

from mqt.yaqs.characterization.tomography.tomography import (
    calculate_dual_choi_basis,
    get_basis_states,
    get_choi_basis,
    run,
)
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def _get_random_rho(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Generate a random 2x2 density matrix.

    Args:
        rng: The random number generator.

    Returns:
        NDArray[np.complex128]: The resulting 2x2 density matrix.
    """
    # General mixed state via Ginibre ensemble
    z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = z @ z.conj().T
    return rho / np.trace(rho)


def _apply_local_map_site0(
    rho_full: NDArray[np.complex128],
    local_map: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
) -> NDArray[np.complex128]:
    """Apply (local_map ⊗ I) to a 2-qubit density matrix rho_full.

    Assumes rho_full uses ordering kron(site0, site1).
    Reshape convention: rho[i, k, j, l] where
      i,j index site0 (row/col), k,l index site1 (row/col).

    Args:
        rho_full: The 2-qubit density matrix.
        local_map: The local CP map to apply to site 0.

    Returns:
        NDArray[np.complex128]: The resulting 2-qubit density matrix.

    Raises:
        ValueError: If local_map does not return a (2,2) matrix.
    """
    rho4 = rho_full.reshape(2, 2, 2, 2)  # (i,k,j,l)
    out4 = np.zeros_like(rho4, dtype=np.complex128)

    for i in range(2):
        for j in range(2):
            e_ij = np.zeros((2, 2), dtype=np.complex128)
            e_ij[i, j] = 1.0

            a_eij = local_map(e_ij)
            if a_eij.shape != (2, 2):
                msg = f"local_map must return (2,2), got {a_eij.shape}"
                raise ValueError(msg)

            block_ij = rho4[i, :, j, :]  # (k,l) environment operator
            out4 += np.einsum("ab,kl->akbl", a_eij, block_ij)

    return out4.reshape(4, 4)


def test_tomography_run_basic() -> None:
    """Test standard single-step process tomography."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1])

    assert pt.tensor.shape == (4, 16)
    assert pt.weights.shape == (16,)
    assert len(pt.timesteps) == 1


def test_tomography_run_defaults() -> None:
    """Test defaults (timesteps=None -> single step)."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, elapsed_time=0.1)
    pt = run(op, params)
    assert pt.tensor.shape == (4, 16)


def test_tomography_mcwf_multistep() -> None:
    """Test multi-step process tomography with MCWF solver."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", num_traj=10)
    pt = run(op, params, timesteps=[0.1, 0.1])
    assert pt.tensor.shape == (4, 16, 16)


def test_tomography_run_multistep() -> None:
    """Test structure of multi-step PT."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1, 0.2])

    assert pt.tensor.shape == (4, 16, 16)
    assert pt.weights.shape == (16, 16)
    assert len(pt.timesteps) == 2


def test_basis_reproduction() -> None:
    """Verify that identity map yields correct prediction."""
    op = MPO.ising(length=2, J=0.0, g=0.0)  # Zero Hamiltonian
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1])

    def identity_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return rho

    rho_pred = pt.predict_final_state([identity_map])
    # Expect state to be |0><0| (initial)
    expected = np.array([[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(rho_pred, expected, atol=1e-10)


def test_predict_linearity() -> None:
    """Ensure predict_final_state is linear in the intervention maps."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1])

    def map1(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return rho

    def map2(_rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.zeros((2, 2), dtype=complex)

    def sum_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return 0.5 * map1(rho) + 0.3 * map2(rho)

    rho1 = pt.predict_final_state([map1])
    rho2 = pt.predict_final_state([map2])
    rho_sum = pt.predict_final_state([sum_map])

    np.testing.assert_allclose(rho_sum, 0.5 * rho1 + 0.3 * rho2, atol=1e-10)


def test_reconstruction_depolarizing() -> None:
    """Test reconstruction of a depolarizing channel via PT."""
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1])

    def depolarize(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return 0.5 * np.trace(rho) * np.eye(2)

    rho_pred = pt.predict_final_state([depolarize])
    expected = 0.5 * np.eye(2)
    np.testing.assert_allclose(rho_pred, expected, atol=1e-10)


def test_choi_duality_biorthogonality() -> None:
    """Verify dual frame biorthogonality: Tr(D_i^dag B_j) = delta_ij."""
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    for i, d_i in enumerate(duals):
        for j, b_j in enumerate(choi_basis):
            inner = np.trace(d_i.conj().T @ b_j)
            expected = 1.0 if i == j else 0.0
            np.testing.assert_allclose(inner, expected, atol=1e-10)


def test_reconstruction_identity_random_choi() -> None:
    """Verify Choi matrix reconstruction: J = sum_k Tr(D_k^dag J) B_k."""
    rng = np.random.default_rng(42)
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    # Random 4x4 matrix
    j_rand = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))

    coeffs = np.array([np.trace(d.conj().T @ j_rand) for d in duals])
    j_rec = np.zeros((4, 4), dtype=complex)
    for c, b in zip(coeffs, choi_basis, strict=False):
        j_rec += c * b

    np.testing.assert_allclose(j_rec, j_rand, atol=1e-10)


def test_dual_extracts_one_hot_for_basis_maps() -> None:
    """Verify duals extract one-hot coefficients for basis maps under the strict Choi build convention.

    This locks the `predict_final_state` Choi builder convention to the duals.
    """
    basis = get_basis_states()
    choi_basis, choi_indices = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    for alpha in range(16):
        p, m = choi_indices[alpha]
        rho_p = basis[p][2]
        e_m = basis[m][2]

        def a_alpha(
            rho: NDArray[np.complex128],
            e_m_sub: NDArray[np.complex128] = e_m,
            rho_p_sub: NDArray[np.complex128] = rho_p,
        ) -> NDArray[np.complex128]:
            return np.trace(e_m_sub @ rho) * rho_p_sub

        j_choi = np.zeros((4, 4), dtype=complex)
        for i in range(2):
            for j in range(2):
                e = np.zeros((2, 2), dtype=complex)
                e[i, j] = 1.0
                j_choi += np.kron(a_alpha(e), e)  # NO transpose

        c = np.array([np.trace(d.conj().T @ j_choi) for d in duals])
        expected = np.zeros(16, dtype=complex)
        expected[alpha] = 1.0
        np.testing.assert_allclose(c, expected, atol=1e-10)


def test_held_out_prediction() -> None:
    """Test PT prediction against direct evolution for a random preparation map (1-step)."""
    rng = np.random.default_rng(42)
    op = MPO.ising(length=2, J=1.0, g=0.5)

    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=4)
    pt = run(op, params, timesteps=[0.1])

    # Hold-out intervention: prepare arbitrary mixed state rho_0
    rho_0 = _get_random_rho(rng)

    def prep_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.trace(rho) * rho_0

    # 1) PT prediction
    rho_pred = pt.predict_final_state([prep_map])

    # 2) Direct evolution
    h_mat = expm(-1j * op.to_matrix() * 0.1)
    # Start with rho_0 on site 0, |0> on site 1
    rho_init = np.kron(rho_0, np.array([[1.0, 0.0], [0.0, 0.0]]))
    rho_final_full = h_mat @ rho_init @ h_mat.conj().T
    # Partial trace over site 1
    rho_final = np.trace(rho_final_full.reshape(2, 2, 2, 2), axis1=1, axis2=3)

    np.testing.assert_allclose(rho_pred, rho_final, atol=1e-10)


def test_multi_step_correctness() -> None:
    """Verify 2-step PT correctness against explicit global evolution."""
    rng = np.random.default_rng(42)
    op = MPO.ising(length=2, J=1.0, g=0.5)

    # Use order=2 for TJM 2
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=2)
    pt = run(op, params, timesteps=[0.1, 0.1])

    rho_0 = _get_random_rho(rng)

    # Make a proper unitary intervention for step 1
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    u_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.complex128)

    def a0_prep(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        # Prep random mixed state
        return np.trace(rho) * rho_0

    def a1_unitary(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        # Unitary channel
        return u_mat @ rho @ u_mat.conj().T

    # 1) PT prediction
    rho_pred = pt.predict_final_state([a0_prep, a1_unitary])

    # 2) Direct dense evolution
    h_mat = op.to_matrix()
    u_evol = expm(-1j * h_mat * 0.1)

    # Start from |00><00|
    rho_full = np.zeros((4, 4), dtype=np.complex128)
    rho_full[0, 0] = 1.0

    # Apply A0 on site0 (step 0 intervention)
    rho_full = _apply_local_map_site0(rho_full, a0_prep)
    # Evolve 1
    rho_full = u_evol @ rho_full @ u_evol.conj().T
    # Apply A1 on site0 (step 1 intervention)
    rho_full = _apply_local_map_site0(rho_full, a1_unitary)
    # Evolve 2
    rho_full = u_evol @ rho_full @ u_evol.conj().T

    # Partial trace over site 1
    rho_final = np.trace(rho_full.reshape(2, 2, 2, 2), axis1=1, axis2=3)

    # Assert tight tolerance
    np.testing.assert_allclose(rho_pred, rho_final, atol=1e-6)


def test_unnormalized_branch_semantics_h0() -> None:
    """Verify trace-weight consistency in the deterministic H=0 case.

    For each basis map A_{p,m}(rho) = Tr(E_m rho) rho_p, starting from |0><0| on site 0,
    the unnormalized output branch rho_out should have:
    - trace(rho_out) == pt.weights[alpha]
    - trace(rho_out) == Tr(E_m |0><0|)
    """
    op = MPO.ising(length=2, J=0.0, g=0.0)  # H = 0
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)
    pt = run(op, params, timesteps=[0.1])

    basis = get_basis_states()
    _, choi_indices = get_choi_basis()

    # Initial state is |0><0| on site 0
    rho_0 = np.zeros((2, 2), dtype=complex)
    rho_0[0, 0] = 1.0

    for alpha in range(16):
        # Extract the branch density matrix (unnormalized output for this basis map)
        rho_branch = pt.tensor[:, alpha].reshape(2, 2)
        weight = pt.weights[alpha]

        p, m = choi_indices[alpha]
        e_m = basis[m][2]
        rho_p = basis[p][2]  # The expected output state proportional to rho_p

        expected_trace = np.trace(e_m @ rho_0)

        # Assert trace matches weight
        np.testing.assert_allclose(np.trace(rho_branch), weight, atol=1e-10)
        # Assert trace matches theoretical expectation Tr(E_m rho_0)
        np.testing.assert_allclose(np.trace(rho_branch), expected_trace, atol=1e-10)

        # Optionally, verify the state itself is proportional to rho_p
        expected_rho_branch = expected_trace * rho_p
        np.testing.assert_allclose(rho_branch, expected_rho_branch, atol=1e-10)


def test_tomography_with_noise() -> None:
    """Verify that the tomography pipeline runs correctly with a noise model and multiple trajectories.

    This is an integration test to ensure that the parallelized worker handles `num_trajectories > 1`
    and stochastic noise operators without crashing. It does not perform an exact arithmetic assertion
    against the output.
    """
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)

    # Create a simple noise model (e.g. amplitude damping on site 0)
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.05}])

    # Run tomography computationally with noise
    pt = run(op, params, timesteps=[0.1], num_trajectories=5, noise_model=noise_model)

    # Check that the tensor built properly without None outputs
    assert pt.tensor.shape == (4, 16)
    assert not np.isnan(pt.tensor).any()
    assert pt.weights.shape == (16,)
    assert not np.isnan(pt.weights).any()
