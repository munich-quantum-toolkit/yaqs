# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Gate Library utility functions and gates in the YAQS core.

This module includes unit tests for:
- Utility functions `split_tensor` and `extend_gate`, ensuring correct tensor operations for MPO construction.
- Quantum gates provided by `GateLibrary`, verifying the correctness of tensors, matrices, and gate operations.

Specifically, tests ensure:
- Tensors are correctly split and reshaped for MPO representations.
- Identity tensors are correctly inserted between gates when necessary.
- Gates have correctly set parameters, sites, and tensors.
- Gate tensors match their expected unitary matrix representations.
- Gates behave correctly when their sites are specified in reversed order.

These tests validate the internal consistency of quantum gate representations used within the YAQS framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import BaseGate, Destroy, GateLibrary, X, Y, Z, extend_gate, split_tensor

if TYPE_CHECKING:
    from numpy.typing import NDArray


def testsplit_tensor_valid_shape() -> None:
    """Test that split_tensor correctly splits a tensor of shape (2,2,2,2) into two tensors.

    The test creates a tensor with values 0..15 reshaped to (2,2,2,2) and then applies split_tensor.
    It verifies that the result is a list of two tensors, where the first tensor has a dummy dimension added
    (expected shape (2, 2, 1, r)) and the second tensor has shape (2, 2, r, 1).
    """
    # Create a simple tensor of shape (2,2,2,2) with values 0..15.
    tensor = np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    tensors = split_tensor(tensor)
    # Expect a list of two tensors.
    assert isinstance(tensors, list)
    assert len(tensors) == 2

    t1, t2 = tensors
    # t1 should have 4 dimensions and its first two dimensions equal 2.
    assert t1.ndim == 4
    assert t1.shape[0] == 2
    assert t1.shape[1] == 2

    # t2 should also be 4-dimensional with its first two dimensions equal 2.
    assert t2.ndim == 4
    assert t2.shape[0] == 2
    assert t2.shape[1] == 2


def testsplit_tensor_invalid_shape() -> None:
    """Test that split_tensor raises an AssertionError when the input tensor does not have shape (2,2,2,2).

    The test creates a tensor of shape (2,2,2) and expects an assertion error.
    """
    tensor = np.zeros((2, 2, 2))
    with pytest.raises(AssertionError):
        split_tensor(tensor)


def testextend_gate_no_identity() -> None:
    """Test extend_gate when no identity tensor is required.

    This test uses a simple tensor (4x4 identity reshaped to (2,2,2,2)) and sites such that the gap between sites is 1.
    It checks that the resulting MPO has exactly two tensors.
    """
    tensor = np.eye(4).reshape(2, 2, 2, 2)
    sites = [0, 1]  # No gap, so no identity tensor should be added.
    mpo = extend_gate(tensor, sites)
    assert isinstance(mpo, MPO)
    assert len(mpo.tensors) == 2


def testextend_gate_with_identity() -> None:
    """Test extend_gate when an identity tensor is required.

    This test uses a simple tensor (4x4 identity reshaped to (2,2,2,2)) with sites that have a gap (difference 2),
    which should insert one identity tensor. It verifies that the MPO contains three tensors and that the
    identity tensor has the expected structure.
    """
    tensor = np.eye(4).reshape(2, 2, 2, 2)
    sites = [0, 2]  # Gap present, one identity tensor inserted.
    mpo = extend_gate(tensor, sites)
    assert isinstance(mpo, MPO)
    assert len(mpo.tensors) == 3

    identity_tensor = mpo.tensors[1]
    prev_bond = mpo.tensors[0].shape[3]
    assert identity_tensor.shape == (2, 2, prev_bond, prev_bond)
    for i in range(prev_bond):
        assert_array_equal(identity_tensor[:, :, i, i], np.eye(2))


def testextend_gate_reverse_order() -> None:
    """Test that extend_gate correctly handles reverse ordering of sites.

    This test applies extend_gate with sites provided in reverse order, reverses the resulting MPO tensors,
    and verifies that each tensor matches the transpose of the forward-order result on axes (0,1,3,2).
    """
    tensor = np.eye(4).reshape(2, 2, 2, 2)
    mpo_forward = extend_gate(tensor, [0, 1])
    mpo_reverse = extend_gate(tensor, [1, 0])
    mpo_reverse.tensors.reverse()
    for t_f, t_r in zip(mpo_forward.tensors, mpo_reverse.tensors):
        assert_allclose(t_r, np.transpose(t_f, (0, 1, 3, 2)))


def test_gate_x() -> None:
    """Test the X gate from GateLibrary.

    This test creates an X gate, sets its site, and verifies that the sites attribute is correct.
    It also checks that the gate's tensor is equal to its matrix.
    """
    gate = GateLibrary.x()
    gate.set_sites(0)
    assert gate.sites == [0]
    assert_array_equal(gate.tensor, gate.matrix)

    base_gate = BaseGate.x()
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_y() -> None:
    """Test the Y gate from GateLibrary.

    This test creates a Y gate, sets its site, and checks that its tensor equals its matrix.
    """
    gate = GateLibrary.y()
    gate.set_sites(0)
    assert gate.sites == [0]
    assert_array_equal(gate.tensor, gate.matrix)

    base_gate = BaseGate.y()
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_z() -> None:
    """Test the Z gate from GateLibrary.

    This test creates a Z gate, sets its site, and checks that its tensor equals its matrix.
    """
    gate = GateLibrary.z()
    gate.set_sites(0)
    assert gate.sites == [0]
    assert_array_equal(gate.tensor, gate.matrix)

    base_gate = BaseGate.z()
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_id() -> None:
    """Test the identity gate from GateLibrary.

    This test creates the identity gate, sets its site, and verifies that its tensor matches its matrix.
    """
    gate = GateLibrary.id()
    gate.set_sites(0)
    assert gate.sites == [0]
    assert_array_equal(gate.tensor, gate.matrix)

    base_gate = BaseGate.id()
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_sx() -> None:
    """Test the square-root X (sx) gate from GateLibrary.

    This test creates an sx gate, sets its site, and checks that its tensor equals its matrix.
    """
    gate = GateLibrary.sx()
    gate.set_sites(0)
    assert gate.sites == [0]
    assert_array_equal(gate.tensor, gate.matrix)

    base_gate = BaseGate.sx()
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_h() -> None:
    """Test the Hadamard (H) gate from GateLibrary.

    This test creates an H gate, sets its site, and verifies that its tensor equals its matrix.
    """
    gate = GateLibrary.h()
    gate.set_sites(0)
    assert gate.sites == [0]
    assert_array_equal(gate.tensor, gate.matrix)

    base_gate = BaseGate.h()
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_create() -> None:
    """Test the Create gate from GateLibrary."""
    gate = GateLibrary.create()
    gate.set_sites(0)
    assert gate.sites == [0]
    assert_array_equal(gate.tensor, gate.matrix)

    base_gate = BaseGate.create()
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_destroy() -> None:
    """Test the Create gate from GateLibrary."""
    gate = GateLibrary.destroy()
    gate.set_sites(0)
    assert gate.sites == [0]
    assert_array_equal(gate.tensor, gate.matrix)

    base_gate = BaseGate.destroy()
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_phase() -> None:
    """Test the phase (p) gate from GateLibrary.

    This test sets a rotation parameter for the phase gate, sets its site, and verifies that its generator
    is computed correctly. It also confirms that the tensor equals the matrix.
    """
    theta = np.pi / 3
    gate = GateLibrary.phase([theta])
    gate.set_sites(4)
    assert gate.sites == [4]
    assert_array_equal(gate.tensor, gate.matrix)

    base_gate = BaseGate.phase([theta])
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_rx() -> None:
    """Test the Rx gate from GateLibrary.

    This test sets a rotation parameter for the Rx gate, sets its site, and verifies that its tensor
    equals the expected rotation matrix.
    """
    theta = np.pi / 2
    gate = GateLibrary.rx([theta])
    gate.set_sites(1)
    expected = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
    assert_allclose(gate.tensor, expected)

    base_gate = BaseGate.rx([theta])
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_ry() -> None:
    """Test the Ry gate from GateLibrary.

    This test sets a rotation parameter for the Ry gate, sets its site, and verifies that both its matrix
    and tensor match the expected rotation matrix.
    """
    theta = np.pi / 3
    gate = GateLibrary.ry([theta])
    gate.set_sites(1)
    expected = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])
    assert_allclose(gate.matrix, expected)
    assert_allclose(gate.tensor, expected)

    base_gate = BaseGate.ry([theta])
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_rz() -> None:
    """Test the Rz gate from GateLibrary.

    This test sets a rotation parameter for the Rz gate, sets its site, and verifies that both its matrix
    and tensor match the expected diagonal rotation matrix.
    """
    theta = np.pi / 4
    gate = GateLibrary.rz([theta])
    gate.set_sites(2)
    expected = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
    assert_allclose(gate.matrix, expected)
    assert_allclose(gate.tensor, expected)

    base_gate = BaseGate.rz([theta])
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_u3() -> None:
    """Test the U3 gate.

    This test sets the parameters (theta, phi, lambda) for the U3 gate and verifies that its tensor,
    which is equivalent to the 2x2 unitary matrix representation, matches the expected matrix.
    """
    theta = np.pi / 2
    phi = np.pi / 4
    lam = np.pi / 3
    gate = GateLibrary.u3([theta, phi, lam])
    gate.set_sites(0)  # For a single-qubit gate, only one site is needed.

    expected = np.array([
        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
    ])

    assert_allclose(gate.tensor, expected)

    base_gate = BaseGate.u3([theta, phi, lam])
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_cx() -> None:
    """Test the CX (controlled-NOT) gate from GateLibrary.

    This test sets the control and target sites for a CX gate and verifies that:
      - The sites attribute is correctly set.
      - The tensor is reshaped to (2,2,2,2).
      - An MPO has been constructed and contains at least two tensors.
    """
    gate = GateLibrary.cx()
    gate.set_sites(0, 1)
    assert gate.sites == [0, 1]
    assert gate.tensor.shape == (2, 2, 2, 2)
    assert hasattr(gate, "mpo")
    assert isinstance(gate.mpo, MPO)
    assert len(gate.mpo.tensors) >= 2

    base_gate = BaseGate.cx()
    assert_array_equal(gate.matrix, base_gate.matrix)

    with pytest.raises(ValueError, match="Number of sites 3 must be equal to the interaction level 2"):
        gate.set_sites(0, 1, 2)


def test_gate_cz() -> None:
    """Test the CZ (controlled-Z) gate from GateLibrary.

    This test sets a CZ gate in forward order and then in reverse order,
    verifying that in the reverse order the tensor is the transpose (axes (1,0,3,2)) of the forward tensor.
    """
    # Forward order
    gate = GateLibrary.cz()
    gate.set_sites(0, 1)
    tensor_forward = gate.tensor.copy()
    assert gate.sites == [0, 1]
    # Reverse order: tensor should be transposed on axes (1,0,3,2)
    gate_rev = GateLibrary.cz()
    gate_rev.set_sites(1, 0)
    expected = np.transpose(tensor_forward, (1, 0, 3, 2))
    np.testing.assert_allclose(gate_rev.tensor, expected)

    base_gate = BaseGate.cz()
    assert_array_equal(gate.matrix, base_gate.matrix)

    with pytest.raises(ValueError, match="Number of sites 3 must be equal to the interaction level 2"):
        gate.set_sites(0, 1, 2)


def test_gate_swap() -> None:
    """Test the SWAP gate from GateLibrary.

    This test sets the sites for a SWAP gate and verifies that:
      - The sites attribute is correct.
      - The tensor matches the matrix reshaped to (2,2,2,2).
    """
    gate = GateLibrary.swap()
    gate.set_sites(2, 3)
    assert gate.sites == [2, 3]
    expected = gate.matrix.reshape(2, 2, 2, 2)
    assert_array_equal(gate.tensor, expected)

    base_gate = BaseGate.swap()
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_rxx() -> None:
    """Test the Rxx gate from GateLibrary.

    This test sets a rotation parameter for the Rxx gate and verifies that its tensor,
    when reshaped to (2,2,2,2), matches the expected matrix.
    """
    theta = np.pi / 3
    gate = GateLibrary.rxx([theta])
    gate.set_sites(0, 1)
    expected = gate.matrix.reshape(2, 2, 2, 2)
    assert_allclose(gate.tensor, expected)

    base_gate = BaseGate.rxx([theta])
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_ryy() -> None:
    """Test the Ryy gate from GateLibrary.

    This test sets a rotation parameter for the Ryy gate and verifies that its tensor,
    when reshaped to (2,2,2,2), matches the expected matrix.
    """
    theta = np.pi / 4
    gate = GateLibrary.ryy([theta])
    gate.set_sites(1, 2)
    expected = gate.matrix.reshape(2, 2, 2, 2)
    assert_allclose(gate.tensor, expected)

    base_gate = BaseGate.ryy([theta])
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_rzz() -> None:
    """Test the Rzz gate from GateLibrary.

    This test sets a rotation parameter for the Rzz gate and verifies that its tensor,
    when reshaped to (2,2,2,2), matches the expected matrix.
    """
    theta = np.pi / 6
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, 1)
    expected = gate.matrix.reshape(2, 2, 2, 2)
    assert_allclose(gate.tensor, expected)

    base_gate = BaseGate.rzz([theta])
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_cphase_forward() -> None:
    """Test the forward order of the CPhase gate from GateLibrary.

    This test sets the parameters for a CPhase gate and sets the sites in forward order.
    It verifies that the tensor, when reshaped to (2,2,2,2), matches the expected matrix.
    """
    theta = np.pi / 2
    gate = GateLibrary.cphase([theta])
    gate.set_sites(0, 1)  # Forward order
    expected: NDArray[np.complex128] = np.reshape(gate.matrix, (2, 2, 2, 2))
    assert_array_equal(gate.tensor, expected)

    base_gate = BaseGate.cphase([theta])
    assert_array_equal(gate.matrix, base_gate.matrix)


def test_gate_cphase_reverse() -> None:
    """Test the reverse order of the CPhase gate from GateLibrary.

    This test sets the parameters for a CPhase gate and sets the sites in reverse order.
    It then verifies that the tensor is correctly transposed (axes (1,0,3,2)) relative to the forward order.
    """
    theta = np.pi / 2
    gate = GateLibrary.cphase([theta])
    gate.set_sites(1, 0)  # Reverse order; tensor should be transposed on (1,0,3,2)
    expected: NDArray[np.complex128] = np.reshape(gate.matrix, (2, 2, 2, 2))
    expected = np.transpose(expected, (1, 0, 3, 2))
    assert_allclose(gate.tensor, expected)


def test_gate_constructor() -> None:
    """Test the constructor of the GateLibrary."""
    # Testing 1-site gates
    one_site_matrix = np.array([[1, 2], [3, 4]])
    one_site_gate = BaseGate(one_site_matrix)

    assert_array_equal(one_site_gate.matrix, one_site_matrix)
    assert_array_equal(one_site_gate.tensor, one_site_matrix)

    assert one_site_gate.interaction == 1, "Failed to set interaction level for one site"

    # Testing multiple-sites gates
    two_sites_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    two_sites_gate = BaseGate(two_sites_matrix)

    assert_array_equal(two_sites_gate.matrix, two_sites_matrix)
    assert_array_equal(two_sites_gate.tensor, two_sites_matrix)

    assert two_sites_gate.interaction == 2, "Failed to set interaction level for two sites"

    # Testing error messages for invalid matrices
    non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]])

    non_power_of_2_matrix = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])

    # Test for non-square matrix
    with pytest.raises(ValueError, match="Matrix must be square"):
        BaseGate(non_square_matrix)
    # Test for matrix size not being a power of 2
    with pytest.raises(ValueError, match="Matrix must have a size that is a power of 2"):
        BaseGate(non_power_of_2_matrix)


def test_set_sites() -> None:
    """Test the set_sites method of the BaseGate class.

    This test creates a BaseGate instance and sets its sites using the set_sites method.
    It verifies that the sites are correctly set and that the tensor is equal to the matrix.
    """
    # Testing multiple-sites gates
    two_sites_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    two_sites_gate = BaseGate(two_sites_matrix)

    assert two_sites_gate.interaction == 2, "Failed to set interaction level for two sites"

    two_sites_gate.set_sites(0, 1)
    assert two_sites_gate.sites == [0, 1], "Failed to set multiple sites"

    with pytest.raises(ValueError, match="Number of sites 3 must be equal to the interaction level 2"):
        two_sites_gate.set_sites(0, 1, 2)


def test_gate_operations() -> None:
    """Test the operations of the GateLibrary.

    This test creates instances of the Destroy, X, Y, and Z gates, and verifies that the
    resulting matrices from performing addition, multiplication and adjoint operations on them are correct.
    """
    rel = Destroy()

    x = X()
    y = Y()
    z = Z()

    jump_list = [rel, z]

    obs_list = [x, y, z]

    matrices = []

    for lk in jump_list:
        for on in obs_list:
            res = lk.dag() * on * lk - 0.5 * on * lk.dag() * lk - 0.5 * lk.dag() * lk * on
            matrices.append(res.matrix)

    # Check the resulting matrices
    assert len(matrices) == 6  # 3 jump operators * 2 observables

    assert_array_equal(matrices[0], np.array([[0.0 + 0.0j, -0.5 + 0.0j], [-0.5 + 0.0j, 0.0 + 0.0j]]))
    assert_array_equal(matrices[1], np.array([[0.0 + 0.0j, 0.0 + 0.5j], [0.0 - 0.5j, 0.0 + 0.0j]]))
    assert_array_equal(matrices[2], np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 2.0 + 0.0j]]))
    assert_array_equal(matrices[3], np.array([[0.0 + 0.0j, -2.0 + 0.0j], [-2.0 + 0.0j, 0.0 + 0.0j]]))
    assert_array_equal(matrices[4], np.array([[0.0 + 0.0j, 0.0 + 2.0j], [0.0 - 2.0j, 0.0 + 0.0j]]))
    assert_array_equal(matrices[5], np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]))

    assert_array_equal(x.conj().matrix, np.conj(x.matrix))
    assert_array_equal(x.trans().matrix, x.matrix.T)

    assert_array_equal((x + y).matrix, (y + x).matrix)
    assert_array_equal((x + y).matrix, x.matrix + y.matrix)


def test_gate_observable() -> None:
    """Test the observable property of the GateLibrary.

    This test creates an instance of the X gate and verifies that its observable property is set correctly.
    """
    gate = X()

    site = 3

    obs = Observable(gate, site)

    assert_array_equal(obs.gate.matrix, gate.matrix)
    assert obs.site == site


def test_basegate_operations_with_different_interaction() -> None:
    """Test that BaseGate raises ValueError for operations with different interaction levels."""
    # Create two gates with different interaction levels
    gate1 = BaseGate(np.eye(2))  # Single-qubit gate (interaction = 1)

    gate2 = BaseGate(np.eye(4))  # Two-qubit gate (interaction = 2)

    # Test addition
    with pytest.raises(ValueError, match="Cannot add gates with different interaction"):
        _ = gate1 + gate2

    # Test subtraction
    with pytest.raises(ValueError, match="Cannot subtract gates with different interaction"):
        _ = gate1 - gate2

    # Test multiplication
    with pytest.raises(ValueError, match="Cannot multiply gates with different interaction"):
        _ = gate1 * gate2
