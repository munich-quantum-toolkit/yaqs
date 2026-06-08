# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Numba-accelerated TDVP kernels."""

import numpy as np

from mqt.yaqs.core.methods.tdvp.numba import (
    build_dense_heff_bond_numba,
    build_dense_heff_site_numba,
)


def test_build_dense_heff_site_numba() -> None:
    """Test the Numba-accelerated single-site effective Hamiltonian construction."""
    o_dim, p_dim = 2, 2
    mpo_l, mpo_r = 4, 4
    a_in, a_out = 3, 3
    b_in, b_out = 3, 3

    rng = np.random.default_rng(42)
    left_env = rng.standard_normal((a_in, mpo_l, a_out)) + 1j * rng.standard_normal((a_in, mpo_l, a_out))
    right_env = rng.standard_normal((b_in, mpo_r, b_out)) + 1j * rng.standard_normal((b_in, mpo_r, b_out))
    op = rng.standard_normal((o_dim, p_dim, mpo_l, mpo_r)) + 1j * rng.standard_normal((o_dim, p_dim, mpo_l, mpo_r))

    heff_numba = build_dense_heff_site_numba(left_env, right_env, op)

    heff_ref = np.einsum("oplr,alA,brB->oABpab", op, left_env, right_env)

    rows = o_dim * a_out * b_out
    cols = p_dim * a_in * b_in
    heff_ref_flat = heff_ref.reshape(rows, cols)

    np.testing.assert_allclose(heff_numba, heff_ref_flat, rtol=1e-12, atol=1e-12)


def test_build_dense_heff_bond_numba() -> None:
    """Test the Numba-accelerated bond effective Hamiltonian construction."""
    u_dim, p_dim = 3, 2
    a_dim = 4
    v_dim, w_dim = 3, 3

    rng = np.random.default_rng(123)
    left_env = rng.standard_normal((u_dim, a_dim, p_dim)) + 1j * rng.standard_normal((u_dim, a_dim, p_dim))
    right_env = rng.standard_normal((v_dim, a_dim, w_dim)) + 1j * rng.standard_normal((v_dim, a_dim, w_dim))

    heff_numba = build_dense_heff_bond_numba(left_env, right_env)

    heff_ref = np.einsum("uap,vaw->pwuv", left_env, right_env)

    rows = p_dim * w_dim
    cols = u_dim * v_dim
    heff_ref_flat = heff_ref.reshape(rows, cols)

    np.testing.assert_allclose(heff_numba, heff_ref_flat, rtol=1e-12, atol=1e-12)
