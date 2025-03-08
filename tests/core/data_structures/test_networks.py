# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module provides unit tests for the Matrix Product State (MPS) class and its associated methods.
It verifies correct initialization, custom tensor assignment, bond dimension computation, network flipping,
orthogonality center shifting, normalization, observable measurement, and overall validity of MPS objects.
These tests ensure that the MPS class functions as expected in various simulation scenarios.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import GateLibrary

I = GateLibrary.id.matrix
X = GateLibrary.x.matrix
Y = GateLibrary.y.matrix
Z = GateLibrary.z.matrix


def untranspose_block(mpo_tensor: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Reverse the transposition of an MPO tensor.

    MPO tensors are stored in the order (sigma, sigma', row, col). This function transposes
    the tensor to the order (row, col, sigma, sigma') so that the first two indices can be interpreted
    as a block matrix of operators.

    Args:
        mpo_tensor (NDArray[np.complex128]): The MPO tensor in (sigma, sigma', row, col) order.

    Returns:
        NDArray[np.complex128]: The MPO tensor in (row, col, sigma, sigma') order.
    """
    return np.transpose(mpo_tensor, (2, 3, 0, 1))


##############################################################################
# Tests for the MPO class
##############################################################################


def test_init_Ising() -> None:
    """Test that init_Ising creates the correct MPO for the Ising model.

    This test initializes an Ising MPO with a given length, coupling constant (J), and transverse field (g).
    It verifies that:
      - The MPO has the expected length and physical dimension.
      - The left boundary tensor, inner tensors, and right boundary tensor have the correct shapes.
      - The operator blocks (after untransposing) match the expected values: identity, -J*Z, and -g*X.
    """
    mpo = MPO()
    length = 4
    J = 1.0
    g = 0.5

    mpo.init_Ising(length, J, g)

    assert mpo.length == length
    assert mpo.physical_dimension == 2
    assert len(mpo.tensors) == length

    minus_J = -J  # -1.0
    minus_g = -g  # -0.5

    # Check left boundary: shape (2,2,1,3) -> untransposed to (1,3,2,2)
    left_block = untranspose_block(mpo.tensors[0])
    assert left_block.shape == (1, 3, 2, 2)

    block_I = left_block[0, 0]
    block_JZ = left_block[0, 1]
    block_gX = left_block[0, 2]

    assert np.allclose(block_I, I)
    assert np.allclose(block_JZ, minus_J * Z)
    assert np.allclose(block_gX, minus_g * X)

    # Check an inner tensor (if length > 2): shape (2,2,3,3) -> untransposed to (3,3,2,2)
    if length > 2:
        inner_block = untranspose_block(mpo.tensors[1])
        assert inner_block.shape == (3, 3, 2, 2)
        assert np.allclose(inner_block[0, 0], I)
        assert np.allclose(inner_block[0, 1], minus_J * Z)
        assert np.allclose(inner_block[0, 2], minus_g * X)
        assert np.allclose(inner_block[1, 2], Z)
        assert np.allclose(inner_block[2, 2], I)

    # Check right boundary: shape (2,2,3,1) -> untransposed to (3,1,2,2)
    right_block = untranspose_block(mpo.tensors[-1])
    assert right_block.shape == (3, 1, 2, 2)

    block_gX = right_block[0, 0]
    block_Z = right_block[1, 0]
    block_I = right_block[2, 0]

    assert np.allclose(block_gX, minus_g * X)
    assert np.allclose(block_Z, Z)
    assert np.allclose(block_I, I)


def test_init_Heisenberg() -> None:
    """Test that init_Heisenberg creates the correct MPO for the Heisenberg model.

    This test initializes a Heisenberg MPO with given coupling constants (Jx, Jy, Jz) and field h.
    It verifies that:
      - The MPO has the expected length and physical dimension.
      - The left boundary tensor (after untransposition) has the correct shape and
        contains the expected operators: [I, -Jx*X, -Jy*Y, -Jz*Z, -h*Z].
      - Inner and right boundary tensors have the expected shapes.
    """
    mpo = MPO()
    length = 5
    Jx, Jy, Jz, h = 1.0, 0.5, 0.3, 0.2

    mpo.init_Heisenberg(length, Jx, Jy, Jz, h)

    assert mpo.length == length
    assert mpo.physical_dimension == 2
    assert len(mpo.tensors) == length

    left_block = untranspose_block(mpo.tensors[0])
    assert left_block.shape == (1, 5, 2, 2)

    block_I = left_block[0, 0]
    block_JxX = left_block[0, 1]
    block_JyY = left_block[0, 2]
    block_JzZ = left_block[0, 3]
    block_hZ = left_block[0, 4]

    minus_Jx = -Jx
    minus_Jy = -Jy
    minus_Jz = -Jz
    minus_h = -h

    assert np.allclose(block_I, I)
    assert np.allclose(block_JxX, minus_Jx * X)
    assert np.allclose(block_JyY, minus_Jy * Y)
    assert block_JyY.shape == (2, 2)
    assert np.allclose(block_JzZ, minus_Jz * Z)
    assert np.allclose(block_hZ, minus_h * Z)

    for i, tensor in enumerate(mpo.tensors):
        if i == 0:
            assert tensor.shape == (2, 2, 1, 5)
        elif i == length - 1:
            assert tensor.shape == (2, 2, 5, 1)
        else:
            assert tensor.shape == (2, 2, 5, 5)


def test_init_identity() -> None:
    """Test that init_identity initializes an identity MPO correctly.

    This test checks that an identity MPO has the correct length, physical dimension,
    and that each tensor corresponds to the identity operator.
    """
    mpo = MPO()
    length = 3
    pdim = 2

    mpo.init_identity(length, physical_dimension=pdim)

    assert mpo.length == length
    assert mpo.physical_dimension == pdim
    assert len(mpo.tensors) == length

    for tensor in mpo.tensors:
        assert tensor.shape == (2, 2, 1, 1)
        assert np.allclose(np.squeeze(tensor), I)


def test_init_custom_Hamiltonian() -> None:
    """Test initializing a custom Hamiltonian MPO using user-provided boundary and inner tensors.

    This test creates random tensors for the left boundary, inner sites, and right boundary,
    initializes the MPO with these using init_custom_Hamiltonian, and verifies that the tensors
    have the expected shapes and values (after appropriate transposition).
    """
    length = 4
    pdim = 2

    left_bound = np.random.rand(1, 2, pdim, pdim)
    inner = np.random.rand(2, 2, pdim, pdim)
    right_bound = np.random.rand(2, 1, pdim, pdim)

    mpo = MPO()
    mpo.init_custom_Hamiltonian(length, left_bound, inner, right_bound)

    assert mpo.length == length
    assert len(mpo.tensors) == length

    assert mpo.tensors[0].shape == (pdim, pdim, 1, 2)
    for i in range(1, length - 1):
        assert mpo.tensors[i].shape == (pdim, pdim, 2, 2)
    assert mpo.tensors[-1].shape == (pdim, pdim, 2, 1)

    assert np.allclose(mpo.tensors[0], np.transpose(left_bound, (2, 3, 0, 1)))
    for i in range(1, length - 1):
        assert np.allclose(mpo.tensors[i], np.transpose(inner, (2, 3, 0, 1)))
    assert np.allclose(mpo.tensors[-1], np.transpose(right_bound, (2, 3, 0, 1)))


def test_init_custom() -> None:
    """Test that init_custom correctly sets up an MPO from a user-provided list of tensors.

    This test provides a list of tensors for the left boundary, middle, and right boundary,
    initializes the MPO, and checks that the shapes and values of the MPO tensors match the inputs.
    """
    length = 3
    pdim = 2
    tensors = [
        np.random.rand(1, 2, pdim, pdim),
        np.random.rand(2, 2, pdim, pdim),
        np.random.rand(2, 1, pdim, pdim),
    ]

    mpo = MPO()
    mpo.init_custom(tensors)

    assert mpo.length == length
    assert mpo.physical_dimension == pdim
    assert len(mpo.tensors) == length

    for original, created in zip(tensors, mpo.tensors):
        assert original.shape == created.shape
        assert np.allclose(original, created)


def test_convert_to_MPS() -> None:
    """Test converting an MPO to an MPS.

    This test initializes an MPO using init_Ising, converts it to an MPS via convert_to_MPS,
    and verifies that the resulting MPS has the correct length and that each tensor has been reshaped
    to the expected dimensions.
    """
    mpo = MPO()
    length = 3
    J, g = 1.0, 0.5

    mpo.init_Ising(length, J, g)
    mps = mpo.convert_to_MPS()

    assert isinstance(mps, MPS)
    assert mps.length == length

    for i, tensor in enumerate(mps.tensors):
        original_mpo_tensor = mpo.tensors[i]
        pdim2 = original_mpo_tensor.shape[0] * original_mpo_tensor.shape[1]
        bond_in = original_mpo_tensor.shape[2]
        bond_out = original_mpo_tensor.shape[3]
        assert tensor.shape == (pdim2, bond_in, bond_out)


def test_check_if_valid_MPO() -> None:
    """Test that a valid MPO passes the check_if_valid_MPO method without raising errors.

    This test initializes an Ising MPO and calls check_if_valid_MPO, which should validate the MPO.
    """
    mpo = MPO()
    length = 4
    J, g = 1.0, 0.5

    mpo.init_Ising(length, J, g)
    mpo.check_if_valid_MPO()


def test_rotate() -> None:
    """Test the rotate method for an MPO.

    This test checks that rotating an MPO (without conjugation) transposes each tensor as expected,
    and that rotating back with conjugation returns tensors with the original physical dimensions.
    """
    mpo = MPO()
    length = 3
    J, g = 1.0, 0.5

    mpo.init_Ising(length, J, g)
    original_tensors = [t.copy() for t in mpo.tensors]

    mpo.rotate(conjugate=False)
    for orig, rotated in zip(original_tensors, mpo.tensors):
        assert rotated.shape == (orig.shape[1], orig.shape[0], orig.shape[2], orig.shape[3])
        np.testing.assert_allclose(rotated, np.transpose(orig, (1, 0, 2, 3)))

    mpo.rotate(conjugate=True)
    for tensor in mpo.tensors:
        assert tensor.shape[0:2] == (2, 2)


def test_check_if_identity() -> None:
    """Test that an identity MPO is recognized as identity by check_if_identity.

    This test initializes an identity MPO and verifies that check_if_identity returns True
    when a fidelity threshold is provided.
    """
    mpo = MPO()
    length = 3
    pdim = 2

    mpo.init_identity(length, pdim)
    fidelity_threshold = 0.9
    assert mpo.check_if_identity(fidelity_threshold) is True


##############################################################################
# Tests for the MPS class
##############################################################################


@pytest.mark.parametrize("state", ["zeros", "ones", "x+", "x-", "y+", "y-", "Neel", "wall"])
def test_mps_initialization(state: str) -> None:
    """Test that MPS initializes with the correct chain length, physical dimensions, and tensor shapes.

    This test creates an MPS with 4 sites using a specified default state and verifies that each tensor
    is of rank 3 and has dimensions corresponding to the physical dimension and default bond dimensions.

    Args:
        state (str): The default state to initialize (e.g., "zeros", "ones", "x+", etc.).
    """
    length = 4
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim] * length, state=state)

    assert mps.length == length
    assert len(mps.tensors) == length
    assert all(d == pdim for d in mps.physical_dimensions)

    for tensor in mps.tensors:
        assert tensor.ndim == 3
        assert tensor.shape[0] == pdim
        assert tensor.shape[1] == 1
        assert tensor.shape[2] == 1


def test_mps_custom_tensors() -> None:
    """Test that an MPS can be initialized with custom tensors.

    This test provides a list of custom rank-3 tensors for an MPS and verifies that the MPS
    retains these tensors correctly.
    """
    length = 3
    pdim = 2
    t1 = np.random.rand(pdim, 1, 2)
    t2 = np.random.rand(pdim, 2, 2)
    t3 = np.random.rand(pdim, 2, 1)
    tensors = [t1, t2, t3]

    mps = MPS(length=length, tensors=tensors, physical_dimensions=[pdim] * length)
    assert mps.length == length
    assert len(mps.tensors) == length
    for i, tensor in enumerate(mps.tensors):
        assert np.allclose(tensor, tensors[i])


def test_write_max_bond_dim() -> None:
    """Test that write_max_bond_dim returns the maximum bond dimension of an MPS.

    Constructs an MPS with varying bond dimensions and checks that the maximum bond dimension is reported correctly.
    """
    length = 3
    pdim = 2
    t1 = np.random.rand(pdim, 1, 4)
    t2 = np.random.rand(pdim, 4, 5)
    t3 = np.random.rand(pdim, 5, 2)
    mps = MPS(length, tensors=[t1, t2, t3], physical_dimensions=[pdim] * length)

    max_bond = mps.write_max_bond_dim()
    assert max_bond == 5


def test_flip_network() -> None:
    """Test the flip_network method of MPS.

    This test reverses the order of the MPS tensors and transposes each tensor's bond dimensions.
    Flipping the network twice should restore the original order and tensor values.
    """
    length = 3
    pdim = 2
    t1 = np.random.rand(pdim, 1, 2)
    t2 = np.random.rand(pdim, 2, 2)
    t3 = np.random.rand(pdim, 2, 1)
    original_tensors = [t1, t2, t3]
    mps = MPS(length, tensors=copy.deepcopy(original_tensors), physical_dimensions=[pdim] * length)

    mps.flip_network()
    flipped_tensors = mps.tensors
    assert len(flipped_tensors) == length
    assert flipped_tensors[0].shape == (pdim, original_tensors[2].shape[2], original_tensors[2].shape[1])
    mps.flip_network()
    for orig, now in zip(original_tensors, mps.tensors):
        assert np.allclose(orig, now)


def test_shift_orthogonality_center_right() -> None:
    """Test shifting the orthogonality center to the right in an MPS.

    This test verifies that shifting the orthogonality center does not change the rank of the tensors.
    """
    length = 4
    pdim = 2
    t1 = np.random.rand(pdim, 1, 2)
    t2 = np.random.rand(pdim, 2, 3)
    t3 = np.random.rand(pdim, 3, 3)
    t4 = np.random.rand(pdim, 3, 1)
    mps = MPS(length, tensors=[t1, t2, t3, t4], physical_dimensions=[pdim] * length)

    mps.shift_orthogonality_center_right(current_orthogonality_center=0)
    for tensor in mps.tensors:
        assert tensor.ndim == 3


def test_shift_orthogonality_center_left() -> None:
    """Test shifting the orthogonality center to the left in an MPS.

    This test ensures that the left shift operation does not alter the rank (3) of the MPS tensors.
    """
    length = 4
    pdim = 2
    t1 = np.random.rand(pdim, 1, 3)
    t2 = np.random.rand(pdim, 3, 3)
    t3 = np.random.rand(pdim, 3, 2)
    t4 = np.random.rand(pdim, 2, 1)
    mps = MPS(length, [t1, t2, t3, t4], [pdim] * length)

    mps.shift_orthogonality_center_left(current_orthogonality_center=3)
    for tensor in mps.tensors:
        assert tensor.ndim == 3


def test_set_canonical_form() -> None:
    """Test that set_canonical_form correctly sets the MPS into a canonical form without altering tensor shapes.

    This test initializes an MPS with a default state and applies the canonical form procedure, ensuring
    that tensor ranks remain unchanged.
    """
    length = 4
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim] * length, state="zeros")
    mps.set_canonical_form(orthogonality_center=2)
    for tensor in mps.tensors:
        assert tensor.ndim == 3


def test_normalize() -> None:
    """Test that normalize brings an MPS to unit norm without changing tensor ranks.

    This test normalizes an MPS (using 'B' normalization) and verifies that the overall norm is 1 and
    that all tensors remain rank-3.
    """
    length = 4
    pdim = 2
    t1 = np.random.rand(pdim, 1, 3)
    t2 = np.random.rand(pdim, 3, 3)
    t3 = np.random.rand(pdim, 3, 2)
    t4 = np.random.rand(pdim, 2, 1)
    mps = MPS(length, [t1, t2, t3, t4], [pdim] * length)

    mps.normalize(form="B")
    assert np.isclose(mps.norm(), 1)
    for tensor in mps.tensors:
        assert tensor.ndim == 3


def test_measure() -> None:
    """Test that the measure method of an MPS returns the expected observable value.

    This test creates an MPS initialized in the 'x+' state, measures the X observable on site 0,
    and verifies that the measured value is close to 1.
    """
    length = 2
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim] * length, state="x+")
    obs = Observable(site=0, name="x")
    val = mps.measure(obs)
    assert np.isclose(val, 1)


def test_norm() -> None:
    """Test that the norm of an MPS initialized in the 'zeros' state is 1.

    This test checks the norm method of an MPS.
    """
    length = 3
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim] * length, state="zeros")
    val = mps.norm()
    assert val == 1


def test_check_if_valid_MPS() -> None:
    """Test that an MPS with consistent bond dimensions passes the validity check.

    This test creates an MPS with carefully constructed tensors and verifies that check_if_valid_MPS
    does not raise an exception.
    """
    length = 3
    pdim = 2
    t1 = np.random.rand(pdim, 1, 2)
    t2 = np.random.rand(pdim, 2, 3)
    t3 = np.random.rand(pdim, 3, 1)
    mps = MPS(length, tensors=[t1, t2, t3], physical_dimensions=[pdim] * length)
    mps.check_if_valid_MPS()


def test_check_canonical_form() -> None:
    """Test that check_canonical_form executes without error and returns canonical information.

    This test initializes an MPS and calls check_canonical_form to ensure it produces output
    (e.g., debug information or canonical indices) without crashing.
    """
    length = 3
    pdim = 2
    mps = MPS(length, physical_dimensions=[pdim] * length, state="zeros")
    res = mps.check_canonical_form()
    assert res is not None
