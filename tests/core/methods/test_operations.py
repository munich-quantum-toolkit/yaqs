# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for basic quantum operations on Matrix Product States (MPS).

This module contains tests verifying fundamental operations on Matrix Product States (MPS)
implemented in YAQS. Specifically, it tests:

- Scalar products between MPS, including cases of identical states, orthogonal states,
  and partial scalar products at specified sites.
- Calculation of local expectation values for common quantum observables (Pauli X and Z)
  on standard quantum states (|0>, |1>, |+>).
- Quantum measurements (single-shot and multi-shot) on MPS representations of simple product states.

These tests ensure correctness and reliability of basic quantum operations within YAQS.
"""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.methods.operations import local_expval, measure, measure_single_shot, scalar_product


def test_scalar_product_same_state() -> None:
    """Test that the scalar product of a normalized state with itself equals 1.

    For a normalized product state (here constructed as an MPS in 'random' state), the inner product
    <psi|psi> should be 1.
    """
    psi_mps = MPS(length=3, state="random")
    val = scalar_product(psi_mps, psi_mps)
    np.testing.assert_allclose(val, 1.0, atol=1e-12)


def test_scalar_product_orthogonal_states() -> None:
    """Test that the scalar product between orthogonal product states is 0.

    This test creates two MPS objects initialized in orthogonal states ("zeros" and "ones")
    and verifies that their inner product is 0.
    """
    psi_mps_0 = MPS(length=3, state="zeros")
    psi_mps_1 = MPS(length=3, state="ones")
    val = scalar_product(psi_mps_0, psi_mps_1)
    np.testing.assert_allclose(val, 0.0, atol=1e-12)


def test_scalar_product_partial_site() -> None:
    """Test the scalar product function when specifying a single site.

    For a given site (here site 0 of a 3-site MPS), the scalar product computed by
    scalar_product should equal the direct contraction of the tensor at that site,
    which for a normalized state is 1.
    """
    psi_mps = MPS(length=3, state="x+")
    site = 0
    partial_val = scalar_product(psi_mps, psi_mps, site=site)
    np.testing.assert_allclose(partial_val, 1.0, atol=1e-12)


def test_local_expval_z_on_zero_state() -> None:
    """Test the local expectation value of the Z observable on a |0> state.

    For the computational basis state |0>, the expectation value of Z is +1.
    This test verifies that local_expval returns +1 for site 0 and site 1 of a 2-qubit MPS
    initialized in the "zeros" state.
    """
    # Pauli-Z in computational basis.
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    psi_mps = MPS(length=2, state="zeros")
    val = local_expval(psi_mps, Z, site=0)
    np.testing.assert_allclose(val, 1.0, atol=1e-12)
    val_site1 = local_expval(psi_mps, Z, site=1)
    np.testing.assert_allclose(val_site1, 1.0, atol=1e-12)


def test_local_expval_x_on_plus_state() -> None:
    """Test the local expectation value of the X observable on a |+> state.

    For the |+> state, defined as 1/√2 (|0> + |1>), the expectation value of the X observable is +1.
    This test verifies that local_expval returns +1 for a single-qubit MPS initialized in the "x+" state.
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    psi_mps = MPS(length=3, state="x+")
    val = local_expval(psi_mps, X, site=0)
    np.testing.assert_allclose(val, 1.0, atol=1e-12)


def test_single_shot() -> None:
    """Test measure_single_shot on an MPS initialized in the |0> state.

    For an MPS representing the state |0> on all qubits, a single-shot measurement should yield 0.
    """
    psi_mps = MPS(length=3, state="zeros")
    val = measure_single_shot(psi_mps)
    np.testing.assert_allclose(val, 0, atol=1e-12)


def test_multi_shot() -> None:
    """Test measure over multiple shots on an MPS initialized in the |1> state.

    This test performs 10 measurement shots on an MPS in the "ones" state and verifies that
    the measurement result for the corresponding basis state (here, 7) is present, while an unexpected
    key (e.g., 0) should not be present.
    """
    psi_mps = MPS(length=3, state="ones")
    shots_dict = measure(psi_mps, shots=10)
    # Assuming that in the "ones" state the measurement outcome is encoded as 7.
    assert shots_dict[7]
    with pytest.raises(KeyError):
        _ = shots_dict[0]
