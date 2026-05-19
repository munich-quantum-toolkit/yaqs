# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for BLAS-thread-safe dense exponentials."""

from __future__ import annotations

import numpy as np
import scipy.linalg

from mqt.yaqs.core.numerics.blas_safe import (
    expm_dense,
    is_hermitian_matrix,
    unitary_propagator_from_hermitian,
)


def test_expm_dense_matches_scipy_small_random() -> None:
    """``expm_dense`` should match SciPy's dense exponential on a small matrix."""
    rng = np.random.default_rng(0)
    n = 5
    m = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    expected = scipy.linalg.expm(m)
    got = expm_dense(m.astype(np.complex128))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_unitary_propagator_matches_expm_for_hermitian() -> None:
    """Hermitian fast path should match ``expm(-1j*dt*H)`` for a random Hamiltonian."""
    rng = np.random.default_rng(1)
    n = 4
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h = 0.5 * (a + a.conj().T)
    dt = 0.03
    ref = scipy.linalg.expm(-1j * dt * h)
    got = unitary_propagator_from_hermitian(h.astype(np.complex128), dt)
    np.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-11)


def test_is_hermitian_matrix_examples() -> None:
    """``is_hermitian_matrix`` should accept the identity and reject a non-Hermitian matrix."""
    assert is_hermitian_matrix(np.eye(2, dtype=np.complex128)) is True
    assert is_hermitian_matrix(np.array([[0, 1], [0, 0]], dtype=np.complex128)) is False
