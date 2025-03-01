import pytest
import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm

from mqt.yaqs.core.methods.matrix_exponential import (
    _lanczos_iteration,
    expm_krylov
)

##############################################################################
# Tests for _lanczos_iteration
##############################################################################

def test_lanczos_iteration_small():
    """
    Check that _lanczos_iteration produces correct shapes and orthonormal vectors
    for a small 2x2 Hermitian matrix.
    """
    A = np.array([[2.0, 1.0],
                  [1.0, 3.0]])
    def Afunc(x):
        return A @ x

    vstart = np.array([1.0, 1.0], dtype=complex)
    numiter = 2

    alpha, beta, V = _lanczos_iteration(Afunc, vstart, numiter)
    # alpha => (2,), beta => (1,), V => shape (dimension=2, numiter=2)
    assert alpha.shape == (2,)
    assert beta.shape == (1,)
    assert V.shape == (2, 2)

    # Check first vector is normalized
    np.testing.assert_allclose(norm(V[:,0]), 1.0, atol=1e-12)
    # Check second vector is orthonormal to the first
    dot_01 = np.vdot(V[:,0], V[:,1])
    np.testing.assert_allclose(dot_01, 0.0, atol=1e-12)
    np.testing.assert_allclose(norm(V[:,1]), 1.0, atol=1e-12)


def test_lanczos_early_termination():
    """
    If beta[j] is nearly zero, the code should return early with truncated alpha, beta, V.
    Create a diagonal matrix so that if vstart is one of the eigenvectors, 
    the iteration can terminate early.
    """
    A = np.diag([1.0, 2.0])
    def Afunc(x):
        return A @ x

    # Start vector aligned with eigenvector [1,0].
    vstart = np.array([1.0, 0.0], dtype=complex)
    numiter = 5

    alpha, beta, V = _lanczos_iteration(Afunc, vstart, numiter)
    # We expect it to stop after j=1 => effectively only 1 step
    # so alpha => shape(1,), beta => shape(0,), V => shape(2,1)
    assert alpha.shape == (1,)
    assert beta.shape == (0,)
    assert V.shape == (2,1), "Should have truncated V to 1 Lanczos vector."


##############################################################################
# Tests for expm_krylov
##############################################################################

def test_expm_krylov_2x2_exact():
    """
    For a 2x2 Hermitian matrix, if numiter=2 (full dimension),
    expm_krylov should match the direct matrix exponential exactly.
    """
    A = np.array([[2.0, 1.0],
                  [1.0, 3.0]])
    def Afunc(x):
        return A @ x

    v = np.array([1.0, 0.0], dtype=complex)
    dt = 0.1
    numiter = 2  # covers the full space for a 2x2

    approx = expm_krylov(Afunc, v, dt, numiter)
    direct = expm(-1j*dt*A) @ v

    np.testing.assert_allclose(approx, direct, atol=1e-12,
        err_msg="Krylov expm approximation should match direct exponential for 2x2, numiter=2.")


def test_expm_krylov_smaller_subspace():
    """
    If numiter < dimension, the result might be approximate. We'll allow
    some tolerance, but it shouldn't be too far off for small dt.
    """
    A = np.array([[2.0, 1.0],
                  [1.0, 3.0]])
    def Afunc(x):
        return A @ x

    v = np.array([1.0, 1.0], dtype=complex)
    dt = 0.05
    numiter = 1  # a smaller Krylov subspace than the dimension => approximate

    approx = expm_krylov(Afunc, v, dt, numiter)
    direct = expm(-1j*dt*A) @ v

    # We expect them to be close, but not perfect
    np.testing.assert_allclose(approx, direct, atol=1e-1,
        err_msg="Krylov with subspace < dimension might be approximate, but should be within 1e-1 for small dt.")
