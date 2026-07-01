# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for SciPy-style BLAS-thread-safe matrix exponential helpers."""

from __future__ import annotations

import numpy as np
import scipy.linalg

from mqt.yaqs.core import linalg


def test_expm_matches_scipy_small_random() -> None:
    """``linalg.expm`` should match SciPy's dense exponential on a small matrix."""
    rng = np.random.default_rng(0)
    n = 5
    m = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    expected = scipy.linalg.expm(m)
    got = linalg.expm(m.astype(np.complex128))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_expm_hermitian_matches_expm_for_hermitian() -> None:
    """``expm_hermitian`` should match ``expm(-1j*dt*H)`` for a random Hamiltonian."""
    rng = np.random.default_rng(1)
    n = 4
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h = 0.5 * (a + a.conj().T)
    dt = 0.03
    ref = scipy.linalg.expm(-1j * dt * h)
    got = linalg.expm_hermitian(h.astype(np.complex128), dt)
    np.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-11)


def test_ishermitian_examples() -> None:
    """``ishermitian`` should accept the identity and reject a non-Hermitian matrix."""
    assert linalg.ishermitian(np.eye(2, dtype=np.complex128)) is True
    assert linalg.ishermitian(np.array([[0, 1], [0, 0]], dtype=np.complex128)) is False


def test_ishermitian_rtol_atol_kwargs() -> None:
    """``ishermitian`` should honor ``rtol`` and ``atol`` like SciPy."""
    nearly_hermitian = np.array([[1.0, 1e-6], [0.0, 1.0]], dtype=np.complex128)
    assert linalg.ishermitian(nearly_hermitian, rtol=1e-5, atol=1e-8) is False
    assert linalg.ishermitian(nearly_hermitian, rtol=1e-4, atol=1e-4) is True
