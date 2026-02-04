# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Krylov subspace methods used for matrix exponential calculations.

This module provides unit tests for the internal functions `lanczos_iteration` and `expm_krylov`,
which are utilized in YAQS for efficient computation of matrix exponentials.

The tests verify that:
- The Lanczos iteration correctly generates orthonormal bases and respects expected shapes.
- Early termination of the Lanczos iteration occurs appropriately when convergence conditions are met.
- Krylov subspace approximations to matrix exponentials match exact computations when the subspace dimension
  equals the full space, and remain within acceptable error bounds for smaller subspace dimensions.

These tests ensure reliable numerical behavior and accuracy of Krylov-based algorithms within YAQS.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import scipy.linalg

from mqt.yaqs.core.methods import matrix_exponential
from mqt.yaqs.core.methods.matrix_exponential import expm_krylov


def test_expm_krylov_2x2_exact() -> None:
    """Test exact Krylov matrix exponential.

    For a 2x2 Hermitian matrix, when lanczos_iterations equals the full dimension (2),
    expm_krylov should yield a result that matches the direct matrix exponential exactly.
    """
    mat = np.array([[2.0, 1.0], [1.0, 3.0]])

    def matrix_free_operator(x: np.ndarray) -> np.ndarray:
        return mat @ x

    v = np.array([1.0, 0.0], dtype=complex)
    dt = 0.1
    lanczos_iterations = 2  # full subspace

    approx = expm_krylov(matrix_free_operator, v, dt, max_lanczos_iterations=lanczos_iterations)
    direct = scipy.linalg.expm(-1j * dt * mat) @ v

    np.testing.assert_allclose(
        approx,
        direct,
        atol=1e-12,
        err_msg="Krylov expm approximation should match direct exponential for 2x2, lanczos_iterations=2.",
    )


def test_expm_krylov_smaller_subspace() -> None:
    """Test small subspace Krylov matrix exponential.

    For a 2x2 Hermitian matrix, if lanczos_iterations is less than the full dimension,
    the expm_krylov result will be approximate. For small dt, the approximation
    should be within a tolerance of 1e-1.
    """
    mat = np.array([[2.0, 1.0], [1.0, 3.0]])

    def matrix_free_operator(x: np.ndarray) -> np.ndarray:
        return mat @ x

    v = np.array([1.0, 1.0], dtype=complex)
    dt = 0.05
    lanczos_iterations = 1  # subspace dimension smaller than the full space

    approx = expm_krylov(matrix_free_operator, v, dt, max_lanczos_iterations=lanczos_iterations)
    direct = scipy.linalg.expm(-1j * dt * mat) @ v

    np.testing.assert_allclose(
        approx,
        direct,
        atol=1e-1,
        err_msg="Krylov with subspace < dimension might be approximate, but should be within 1e-1 for small dt.",
    )


def test_expm_krylov_zero_norm() -> None:
    """Test that zero vector input returns zero vector immediately."""
    v = np.zeros(10, dtype=complex)
    dt = 0.1

    # matrix_free_operator shouldn't even be called
    mock_op = MagicMock()

    res = expm_krylov(mock_op, v, dt)

    np.testing.assert_array_equal(res, v)
    mock_op.assert_not_called()


def test_expm_krylov_numba_execution() -> None:
    """Test execution path when Numba is enabled (large vector)."""
    # Create a vector larger than NUMBA_THRESHOLD (4096)
    # We'll patch NUMBA_THRESHOLD effectively by using a large vector or patching the constant
    # Patching constant is safer to avoid huge allocations in test

    size = 100
    v = np.ones(size, dtype=complex)
    dt = 0.1
    mat = np.eye(size)

    def op(x: np.ndarray) -> np.ndarray:
        return mat @ x

    # We need to verify that numba logic is triggered.
    # The simplest way is to mock lanczos_numba.orthogonalize_step or check coverage.
    # But since we can't easily mock imported cached modules inside the function,
    # we can try to patch the constant in the module.

    with patch("mqt.yaqs.core.methods.matrix_exponential.NUMBA_THRESHOLD", 50):
        # Trigger numba path because size=100 > 50

        # We also need to ensure lanczos_numba can be imported.
        # Ideally we assume it is since we are testing it.

        res = expm_krylov(op, v, dt, max_lanczos_iterations=5)

        # Exact solution for Identity matrix exp(-i*dt*I) v = e^{-i*dt} v
        expected = np.exp(-1j * dt) * v
        np.testing.assert_allclose(res, expected)


def test_expm_krylov_numba_early_convergence() -> None:
    """Test early convergence (breakdown) in Numba path."""
    # If starting vector is eigenvector, breakdown happens at step 1
    size = 60
    v = np.zeros(size, dtype=complex)
    v[0] = 1.0  # Standard basis vector
    mat = np.diag(np.arange(size))  # Diagonal matrix, standard basis vectors are eigenvectors

    # With v=e_0, Av = 0*e_0 = 0. Krylov subspace is 1D.
    # Should converge immediately.

    def op(x: np.ndarray) -> np.ndarray:
        return mat @ x

    with patch("mqt.yaqs.core.methods.matrix_exponential.NUMBA_THRESHOLD", 50):
        res = expm_krylov(op, v, dt=0.1, max_lanczos_iterations=10)

        # Exact: exp(-i*0.1*0) * e_0 = 1 * e_0 = e_0
        np.testing.assert_allclose(res, v)


def test_expm_krylov_linalg_error_fallback() -> None:
    """Test fallback to 'stebz' driver when 'stemr' fails with LinAlgError."""
    # We need to patch scipy.linalg.eigh_tridiagonal to raise LinAlgError on first call
    # and succeed on second call.

    # Mock return values must match the dimension implied by alpha (len 2)
    # alpha size 2 -> returns 2 eigenvalues and 2x2 eigenvectors
    mock_evals = np.array([1.0, 2.0])
    mock_evecs = np.eye(2)

    mock_eigh = MagicMock(side_effect=[scipy.linalg.LinAlgError("Test Error"), (mock_evals, mock_evecs)])

    with patch("scipy.linalg.eigh_tridiagonal", mock_eigh):
        # Minimal inputs to reach the eigh call
        size = 10
        # Minimal inputs to reach the eigh call
        size = 10

        # Force single iteration to reach _compute_krylov_result

        alpha = np.array([1.0, 1.0])
        beta = np.array([0.5])
        # lanczos_mat should be (size, k) where k=len(alpha)=2
        # But _compute_krylov_result takes lanczos_mat which is (size, m_max) usually,
        # or (size, k) if sliced.
        # In the function signature: lanczos_mat: NDArray[np.complex128]
        # logic: return np.asarray(lanczos_mat @ (u_hess @ coeffs), dtype=np.complex128)
        # u_hess is (k, k), coeffs is (k,). u_hess @ coeffs -> (k,)
        # lanczos_mat must be (N, k).
        lanczos_mat = np.zeros((size, 2), dtype=complex)
        nrm = 1.0
        dt = 0.1

        matrix_exponential._compute_krylov_result(alpha, beta, lanczos_mat, nrm, dt)  # noqa: SLF001

        assert mock_eigh.call_count == 2
        # First call should be stemr
        assert mock_eigh.call_args_list[0][1]["lapack_driver"] == "stemr"
        # Second call should be stebz
        assert mock_eigh.call_args_list[1][1]["lapack_driver"] == "stebz"
