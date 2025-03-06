# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.libraries.gate_library import GateLibrary, _extend_gate, _split_tensor


def test_split_tensor_valid_shape() -> None:
    # Create a simple tensor of shape (2,2,2,2).
    # Here we use a tensor with values 0..15.
    tensor = np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    tensors = _split_tensor(tensor)
    # Expect a list of two tensors.
    assert isinstance(tensors, list)
    assert len(tensors) == 2

    t1, t2 = tensors
    # t1 is obtained by reshaping U then expanding a dummy dimension.
    # Its shape should be (2, 2, 1, r) for some r.
    assert t1.ndim == 4
    assert t1.shape[0] == 2
    assert t1.shape[1] == 2

    # t2 is reshaped from V, then transposed and expanded.
    # Its shape should be (2, 2, r, 1).
    assert t2.ndim == 4
    assert t2.shape[0] == 2
    assert t2.shape[1] == 2


def test_split_tensor_invalid_shape() -> None:
    # A tensor that does not have shape (2,2,2,2) should trigger the assertion.
    tensor = np.zeros((2, 2, 2))
    with pytest.raises(AssertionError):
        _split_tensor(tensor)


def test_extend_gate_no_identity() -> None:
    # Use a simple tensor. For example, use the 4x4 identity reshaped to (2,2,2,2).
    tensor = np.eye(4).reshape(2, 2, 2, 2)
    # Choose sites such that |site0 - site1| == 1 so no identity tensor is added.
    sites = [0, 1]
    mpo = _extend_gate(tensor, sites)
    # Check that we got an MPO instance.
    assert isinstance(mpo, MPO)
    # With no gap, _split_tensor returns 2 tensors and no identity is inserted.
    assert len(mpo.tensors) == 2


def test_extend_gate_with_identity() -> None:
    # Use a simple tensor.
    tensor = np.eye(4).reshape(2, 2, 2, 2)
    # Use sites with a gap (difference 2 → one identity tensor inserted).
    sites = [0, 2]
    mpo = _extend_gate(tensor, sites)
    assert isinstance(mpo, MPO)
    # Expected list:
    #   [first tensor from _split_tensor,
    #    one identity tensor,
    #    second tensor from _split_tensor]
    assert len(mpo.tensors) == 3

    # Verify that the inserted identity tensor has the expected structure.
    identity_tensor = mpo.tensors[1]
    # The identity tensor shape is (2,2,prev_bond,prev_bond)
    prev_bond = mpo.tensors[0].shape[3]
    assert identity_tensor.shape == (2, 2, prev_bond, prev_bond)
    # Check that each diagonal slice is a 2x2 identity.
    for i in range(prev_bond):
        assert_array_equal(identity_tensor[:, :, i, i], np.eye(2))


def test_extend_gate_reverse_order() -> None:
    # Check that if sites are provided in reverse order, the MPO tensors are reversed
    # and each tensor is transposed on its last two indices.
    tensor = np.eye(4).reshape(2, 2, 2, 2)
    mpo_forward = _extend_gate(tensor, [0, 1])
    mpo_reverse = _extend_gate(tensor, [1, 0])
    mpo_reverse.tensors.reverse()
    # For each pair of corresponding tensors, the reverse version should be the
    # transpose of the forward one on axes (0, 1, 3, 2).
    for t_f, t_r in zip(mpo_forward.tensors, mpo_reverse.tensors):
        assert_allclose(t_r, np.transpose(t_f, (0, 1, 3, 2)))


def test_gate_x() -> None:
    gate = GateLibrary.x()
    gate.set_sites(0)
    assert gate.sites == [0]
    # For X, the tensor should equal the matrix.
    assert_array_equal(gate.tensor, gate.matrix)


def test_gate_y() -> None:
    gate = GateLibrary.y()
    gate.set_sites(0)
    assert gate.sites == [0]
    # For Y, the tensor should equal the matrix.
    assert_array_equal(gate.tensor, gate.matrix)


def test_gate_z() -> None:
    gate = GateLibrary.z()
    gate.set_sites(0)
    assert gate.sites == [0]
    # For Z, the tensor should equal the matrix.
    assert_array_equal(gate.tensor, gate.matrix)


def test_gate_id() -> None:
    gate = GateLibrary.id()
    gate.set_sites(0)
    assert gate.sites == [0]
    assert_array_equal(gate.tensor, gate.matrix)


def test_gate_sx() -> None:
    gate = GateLibrary.sx()
    gate.set_sites(0)
    assert gate.sites == [0]
    assert_array_equal(gate.tensor, gate.matrix)


def test_gate_h() -> None:
    gate = GateLibrary.h()
    gate.set_sites(0)
    assert gate.sites == [0]
    assert_array_equal(gate.tensor, gate.matrix)


def test_gate_phase() -> None:
    gate = GateLibrary.p()
    theta = np.pi / 3
    gate.set_params([theta])
    gate.set_sites(4)
    assert gate.sites == [4]
    expected_gen = (theta / 2) * np.array([[1, 0], [0, -1]])
    assert_allclose(gate.generator, expected_gen)
    # For Phase, tensor equals matrix.
    assert_array_equal(gate.tensor, gate.matrix)


def test_gate_rx() -> None:
    gate = GateLibrary.rx()
    theta = np.pi / 2
    gate.set_params([theta])
    gate.set_sites(1)
    expected = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
    assert_allclose(gate.tensor, expected)


def test_gate_ry() -> None:
    gate = GateLibrary.ry()
    theta = np.pi / 3
    gate.set_params([theta])
    gate.set_sites(1)
    expected = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]])
    assert_allclose(gate.matrix, expected)
    # For Ry, tensor equals matrix.
    assert_allclose(gate.tensor, expected)


def test_gate_rz() -> None:
    gate = GateLibrary.rz()
    theta = np.pi / 4
    gate.set_params([theta])
    gate.set_sites(2)
    expected = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
    assert_allclose(gate.matrix, expected)
    # For Rz, tensor equals matrix.
    assert_allclose(gate.tensor, expected)


def test_gate_cx() -> None:
    gate = GateLibrary.cx()
    gate.set_sites(0, 1)
    assert gate.sites == [0, 1]
    # Check that the tensor is reshaped to (2,2,2,2)
    assert gate.tensor.shape == (2, 2, 2, 2)
    # The gate should also have an MPO constructed.
    assert hasattr(gate, "mpo")
    assert isinstance(gate.mpo, MPO)
    # Basic check: there should be at least 2 tensors in the MPO.
    assert len(gate.mpo.tensors) >= 2


def test_gate_cz() -> None:
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


def test_gate_swap() -> None:
    gate = GateLibrary.swap()
    gate.set_sites(2, 3)
    assert gate.sites == [2, 3]
    # Check that tensor is equal to the matrix reshaped to (2,2,2,2)
    expected = gate.matrix.reshape(2, 2, 2, 2)
    assert_array_equal(gate.tensor, expected)


def test_gate_rxx() -> None:
    gate = GateLibrary.rxx()
    theta = np.pi / 3
    gate.set_params([theta])
    gate.set_sites(0, 1)
    expected = gate.matrix.reshape(2, 2, 2, 2)
    assert_allclose(gate.tensor, expected)


def test_gate_ryy() -> None:
    gate = GateLibrary.ryy()
    theta = np.pi / 4
    gate.set_params([theta])
    gate.set_sites(1, 2)
    expected = gate.matrix.reshape(2, 2, 2, 2)
    assert_allclose(gate.tensor, expected)


def test_gate_rzz() -> None:
    gate = GateLibrary.rzz()
    theta = np.pi / 6
    gate.set_params([theta])
    gate.set_sites(0, 1)
    expected = gate.matrix.reshape(2, 2, 2, 2)
    assert_allclose(gate.tensor, expected)


def test_gate_cphase_forward() -> None:
    gate = GateLibrary.cp()
    theta = np.pi / 2
    gate.set_params([theta])
    gate.set_sites(0, 1)  # Forward order (site0 < site1) so no transpose
    expected: NDArray[np.complex128] = np.reshape(gate.matrix, (2, 2, 2, 2))
    assert_array_equal(gate.tensor, expected)


def test_gate_cphase_reverse() -> None:
    gate = GateLibrary.cp()
    theta = np.pi / 2
    gate.set_params([theta])
    gate.set_sites(1, 0)  # Reverse order; tensor should be transposed on (1,0,3,2)
    expected: NDArray[np.complex128] = np.reshape(gate.matrix, (2, 2, 2, 2))
    expected = np.transpose(expected, (1, 0, 3, 2))
    assert_allclose(gate.tensor, expected)
