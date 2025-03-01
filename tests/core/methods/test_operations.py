# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.methods.operations import local_expval, measure, measure_single_shot, scalar_product


def test_scalar_product_same_state() -> None:
    """If we compute <psi|psi> for a normalized product state, we expect 1."""
    # For qubits, let's pick |0> => [1, 0].
    # Let's say length=3 for a 3-qubit product state of all |0>.
    psi_mps = MPS(length=3, state="random")

    val = scalar_product(psi_mps, psi_mps)
    np.testing.assert_allclose(val, 1.0, atol=1e-12)


def test_scalar_product_orthogonal_states() -> None:
    """Orthogonal product states (e.g., |0> vs |1>) => scalar product = 0."""
    psi_mps_0 = MPS(length=3, state="zeros")
    psi_mps_1 = MPS(length=3, state="ones")

    val = scalar_product(psi_mps_0, psi_mps_1)
    np.testing.assert_allclose(val, 0.0, atol=1e-12)


def test_scalar_product_partial_site() -> None:
    """If we specify a 'site' argument, the code does a single-site contraction:
    result = oe.contract('ijk, ijk', A_copy.tensors[site], B_copy.tensors[site])
    We'll test site=0 for a 2-site MPS, comparing it to direct np.dot on that site.
    """
    psi_mps = MPS(length=3, state="x+")

    site = 0
    partial_val = scalar_product(psi_mps, psi_mps, site=site)
    np.testing.assert_allclose(partial_val, 1.0, atol=1e-12)


def test_local_expval_z_on_zero_state() -> None:
    """Expectation value of Z on |0> is +1.
    For a product state of all |0>, local_expval => +1 (real).
    """
    # Pauli-Z in computational basis
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # 2-qubit MPS, each site => |0>
    psi_mps = MPS(length=2, state="zeros")

    # local_expval(psi, Z, site=0) => <0|Z|0> = 1
    val = local_expval(psi_mps, Z, site=0)
    np.testing.assert_allclose(val, 1.0, atol=1e-12)

    # On the same 2-qubit MPS, if site=1 is also |0>, local_expval => 1 for site=1 as well
    val_site1 = local_expval(psi_mps, Z, site=1)
    np.testing.assert_allclose(val_site1, 1.0, atol=1e-12)


def test_local_expval_x_on_plus_state() -> None:
    """|+> = 1/sqrt(2)(|0> + |1>),
    Expectation value of X on |+> is +1.
    We'll do a single-site MPS for simplicity.
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    # single-qubit MPS => |+>
    [1 / np.sqrt(2), 1 / np.sqrt(2)]
    psi_mps = MPS(length=3, state="x+")

    val = local_expval(psi_mps, X, site=0)
    np.testing.assert_allclose(val, 1.0, atol=1e-12)


def test_single_shot() -> None:
    psi_mps = MPS(length=3, state="zeros")
    val = measure_single_shot(psi_mps)
    np.testing.assert_allclose(val, 0, atol=1e-12)


def test_multi_shot() -> None:
    psi_mps = MPS(length=3, state="ones")
    shots_dict = measure(psi_mps, shots=10)
    assert shots_dict[7]
    with pytest.raises(KeyError):
        assert shots_dict[0]
