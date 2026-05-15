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
import pytest
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
    # Verify Numba module is available
    # Verify Numba module is available
    pytest.importorskip("mqt.yaqs.core.methods.lanczos_numba")
    from mqt.yaqs.core.methods.lanczos_numba import orthogonalize_step  # noqa: PLC0415

    size = 100
    v = np.ones(size, dtype=complex)
    dt = 0.1
    mat = np.eye(size)

    def op(x: np.ndarray) -> np.ndarray:
        return mat @ x

    # Patch NUMBA_THRESHOLD to force Numba path for smaller vector
    with (
        patch("mqt.yaqs.core.methods.matrix_exponential.NUMBA_THRESHOLD", 50),
        # Spy on orthogonalize_step to ensure it's called
        patch("mqt.yaqs.core.methods.lanczos_numba.orthogonalize_step", wraps=orthogonalize_step) as mock_ortho,
    ):
        res = expm_krylov(op, v, dt, max_lanczos_iterations=5)

        # Assert expected numerical result
        expected = np.exp(-1j * dt) * v
        np.testing.assert_allclose(res, expected)

        # Assert Numba kernel was actually used
        assert mock_ortho.called, "Numba-accelerated orthogonalize_step should have been called"


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


def test_expm_arnoldi_2x2_exact() -> None:
    """Test exact Arnoldi matrix exponential for a Hermitian matrix.

    Arnoldi should match Lanczos and direct expm for Hermitian matrices when
    iterations >= dimension.
    """
    mat = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=complex)

    def op(v: np.ndarray) -> np.ndarray:
        return mat @ v

    vec = np.array([1.0, 0.0], dtype=complex)
    dt = 0.1

    approx = matrix_exponential.expm_arnoldi(op, vec, dt, max_arnoldi_iterations=2)
    direct = scipy.linalg.expm(-1j * dt * mat) @ vec

    np.testing.assert_allclose(approx, direct, atol=1e-12)


def test_expm_arnoldi_non_hermitian() -> None:
    """Test Arnoldi matrix exponential for a non-Hermitian matrix."""
    # A simple non-Hermitian matrix (e.g., from an effective Hamiltonian)
    mat = np.array([[1.0 + 0.5j, 0.2], [-0.1j, 2.0 - 0.3j]], dtype=complex)

    def op(v: np.ndarray) -> np.ndarray:
        return mat @ v

    vec = np.array([0.6, 0.8j], dtype=complex)
    dt = 0.05

    approx = matrix_exponential.expm_arnoldi(op, vec, dt, max_arnoldi_iterations=5)
    direct = scipy.linalg.expm(-1j * dt * mat) @ vec

    np.testing.assert_allclose(approx, direct, atol=1e-10)


def test_expm_arnoldi_zero_norm() -> None:
    """Test that Arnoldi handles zero vector input correctly."""
    vec = np.zeros(4, dtype=complex)
    mock_op = MagicMock()

    res = matrix_exponential.expm_arnoldi(mock_op, vec, dt=0.1)

    np.testing.assert_array_equal(res, vec)
    mock_op.assert_not_called()


def test_expm_arnoldi_breakdown() -> None:
    """Test early convergence (breakdown) in Arnoldi iteration."""
    # Start vector is an eigenvector
    mat = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

    def op(v: np.ndarray) -> np.ndarray:
        return mat @ v

    # Breakdown should happen at j=1 (one dimensional subspace)
    res = matrix_exponential.expm_arnoldi(op, vec, dt=0.1, max_arnoldi_iterations=10)

    expected = np.exp(-1j * 0.1 * 1.0) * vec
    np.testing.assert_allclose(res, expected, atol=1e-12)


def test_expm_arnoldi_convergence_tol() -> None:
    """Test that Arnoldi respects the tolerance parameter."""
    # Large random matrix where small iteration count isn't exact
    rng = np.random.default_rng(42)
    size = 20
    mat = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))

    def op(v: np.ndarray) -> np.ndarray:
        return mat @ v

    vec = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    vec /= np.linalg.norm(vec)

    dt = 0.01

    # Large tolerance -> fewer iterations
    res_quick = matrix_exponential.expm_arnoldi(op, vec, dt, max_arnoldi_iterations=10, tol=1e-2)

    # Tight tolerance -> more accurate
    res_tight = matrix_exponential.expm_arnoldi(op, vec, dt, max_arnoldi_iterations=15, tol=1e-12)

    direct = scipy.linalg.expm(-1j * dt * mat) @ vec

    err_quick = np.linalg.norm(res_quick - direct)
    err_tight = np.linalg.norm(res_tight - direct)

    assert err_tight < err_quick
    assert err_tight < 1e-10


def test_compute_arnoldi_result() -> None:
    """Test the internal _compute_arnoldi_result helper."""
    h_mat = np.array([[1.0, 0.5], [0.1, 1.2]], dtype=complex)
    v_mat = np.eye(2, dtype=complex)
    nrm = 2.0
    dt = 0.1

    res = matrix_exponential._compute_arnoldi_result(h_mat, v_mat, nrm, dt)  # noqa: SLF001
    direct = scipy.linalg.expm(-1j * dt * h_mat) @ np.array([nrm, 0.0], dtype=complex)

    np.testing.assert_allclose(res, direct, atol=1e-12)
