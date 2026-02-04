# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

import numpy as np

from mqt.yaqs.core.methods.lanczos_numba import normalize_and_store, orthogonalize_step


def test_orthogonalize_step() -> None:
    """Test orthogonalize_step function.

    Note: The function assumes the underlying operator is Hermitian, implying:
      1. <v_j, w> is real (becomes alpha[j])
      2. <v_{j-1}, w> is real and equals beta[j-1]
    The test must construct w to satisfy these conditions.
    """
    dim = 10
    num_vecs = 5

    # Create random orthonormal vectors
    rng = np.random.default_rng(42)
    v = rng.standard_normal((dim, num_vecs)) + 1j * rng.standard_normal((dim, num_vecs))
    v, _ = np.linalg.qr(v)

    j = 2

    # Define expected projections (must be real)
    expected_alpha = 1.5
    expected_beta_prev = 0.8

    # Create a vector w_perp orthogonal to all v
    w_random = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    # Project out all v components to get pure noise orthogonal to subspace
    for k in range(num_vecs):
        proj = np.vdot(v[:, k], w_random)
        w_random -= proj * v[:, k]
    w_perp = w_random

    # Construct input w:
    # w = w_perp + alpha * v_j + beta_prev * v_{j-1}
    w_input = w_perp + expected_alpha * v[:, j] + expected_beta_prev * v[:, j - 1]

    # Arrays for output
    alpha = np.zeros(num_vecs, dtype=np.float64)
    beta = np.zeros(num_vecs - 1, dtype=np.float64)
    beta[j - 1] = expected_beta_prev  # Input assumption

    # Run Function with a COPY of w_input since it modifies in-place
    w_test = w_input.copy()
    res_beta = orthogonalize_step(v, w_test, j, alpha, beta)

    # Check alpha
    assert np.isclose(alpha[j], expected_alpha)

    # Check orthogonality against v[j]
    # Since we constructed w with real projection, residual projection should be 0
    dot_j = np.vdot(v[:, j], w_test)
    assert np.abs(dot_j) < 1e-10

    # Check orthogonality against v[j-1]
    dot_prev = np.vdot(v[:, j - 1], w_test)
    assert np.abs(dot_prev) < 1e-10

    # Check that the remaining vector is w_perp
    # The function subtracts alpha*v_j and beta_prev*v_{j-1}
    # So w_test should be w_perp
    np.testing.assert_allclose(w_test, w_perp, atol=1e-10)

    # Check return beta
    expected_norm = np.linalg.norm(w_perp)
    assert np.isclose(res_beta, expected_norm)
    assert np.isclose(beta[j], expected_norm)


def test_normalize_and_store() -> None:
    """Test normalize_and_store function."""
    dim = 10
    num_vecs = 5
    j = 2

    v = np.zeros((dim, num_vecs), dtype=np.complex128, order="F")
    rng = np.random.default_rng(43)
    w = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

    norm_w = np.linalg.norm(w)

    # Run function
    normalize_and_store(v, w, j, norm_w)

    # Check stored vector
    expected_v_next = w / norm_w
    np.testing.assert_allclose(v[:, j + 1], expected_v_next)

    # Test case with 0 norm (should not modify v)
    v_copy = v.copy()
    normalize_and_store(v, w, j, 0.0)
    np.testing.assert_array_equal(v, v_copy)
