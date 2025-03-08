# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm

from mqt.yaqs.core.methods.matrix_exponential import _lanczos_iteration, expm_krylov


def test_lanczos_iteration_small() -> None:
    """Check that _lanczos_iteration produces correct shapes and orthonormal vectors
    for a small 2x2 Hermitian matrix.
    """
    A = np.array([[2.0, 1.0], [1.0, 3.0]])

    def Afunc(x):
        return A @ x

    vstart = np.array([1.0, 1.0], dtype=complex)
    numiter = 2

    alpha, beta, V = _lanczos_iteration(Afunc, vstart, numiter)
    # alpha should have shape (2,), beta shape (1,), and V shape (2, 2)
    assert alpha.shape == (2,)
    assert beta.shape == (1,)
    assert V.shape == (2, 2)

    # Check first Lanczos vector is normalized.
    np.testing.assert_allclose(norm(V[:, 0]), 1.0, atol=1e-12)
    # Check second vector is orthogonal to the first.
    dot_01 = np.vdot(V[:, 0], V[:, 1])
    np.testing.assert_allclose(dot_01, 0.0, atol=1e-12)
    np.testing.assert_allclose(norm(V[:, 1]), 1.0, atol=1e-12)


def test_lanczos_early_termination() -> None:
    """Check that _lanczos_iteration terminates early when beta[j] is nearly zero.

    Using a diagonal matrix so that if the starting vector is an eigenvector, the
    iteration can terminate early. In this case, with vstart aligned with [1, 0],
    the iteration should stop after one step.
    """
    A = np.diag([1.0, 2.0])

    def Afunc(x):
        return A @ x

    vstart = np.array([1.0, 0.0], dtype=complex)
    numiter = 5

    alpha, beta, V = _lanczos_iteration(Afunc, vstart, numiter)
    # Expect termination after 1 iteration: alpha shape (1,), beta shape (0,), V shape (2, 1)
    assert alpha.shape == (1,)
    assert beta.shape == (0,)
    assert V.shape == (2, 1), "Should have truncated V to 1 Lanczos vector."


def test_expm_krylov_2x2_exact() -> None:
    """For a 2x2 Hermitian matrix, when numiter equals the full dimension (2),
    expm_krylov should yield a result that matches the direct matrix exponential exactly.
    """
    A = np.array([[2.0, 1.0], [1.0, 3.0]])

    def Afunc(x):
        return A @ x

    v = np.array([1.0, 0.0], dtype=complex)
    dt = 0.1
    numiter = 2  # full subspace

    approx = expm_krylov(Afunc, v, dt, numiter)
    direct = expm(-1j * dt * A) @ v

    np.testing.assert_allclose(
        approx,
        direct,
        atol=1e-12,
        err_msg="Krylov expm approximation should match direct exponential for 2x2, numiter=2.",
    )


def test_expm_krylov_smaller_subspace() -> None:
    """For a 2x2 Hermitian matrix, if numiter is less than the full dimension,
    the expm_krylov result will be approximate. For small dt, the approximation
    should be within a tolerance of 1e-1.
    """
    A = np.array([[2.0, 1.0], [1.0, 3.0]])

    def Afunc(x):
        return A @ x

    v = np.array([1.0, 1.0], dtype=complex)
    dt = 0.05
    numiter = 1  # subspace dimension smaller than the full space

    approx = expm_krylov(Afunc, v, dt, numiter)
    direct = expm(-1j * dt * A) @ v

    np.testing.assert_allclose(
        approx,
        direct,
        atol=1e-1,
        err_msg="Krylov with subspace < dimension might be approximate, but should be within 1e-1 for small dt.",
    )
