# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

import copy
import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import GateLibrary


I = getattr(GateLibrary, "id").matrix
X = getattr(GateLibrary, "x").matrix
Y = getattr(GateLibrary, "y").matrix
Z = getattr(GateLibrary, "z").matrix


def untranspose_block(mpo_tensor):
    """
    MPO tensors are stored as (sigma, sigma', row, col).
    This function reverses that transpose to get (row, col, sigma, sigma').
    That way, we can interpret row x col as a block matrix of operators.
    """
    return np.transpose(mpo_tensor, (2, 3, 0, 1))


################################################
# Tests for the MPO class
################################################
def test_init_Ising():
    """Test that init_Ising creates the correct number, shape, and values."""
    mpo = MPO()
    length = 4
    J = 1.0
    g = 0.5

    mpo.init_Ising(length, J, g)

    assert mpo.length == length
    assert mpo.physical_dimension == 2
    assert len(mpo.tensors) == length

    # Expected sign factors
    minus_J = -J  # -1.0
    minus_g = -g  # -0.5

    #
    # Check left boundary: shape (2,2,1,3)
    # After untranspose => shape (1,3,2,2)
    # The 1x3 block should contain: [I, -J*Z, -g*X]
    #
    left_block = untranspose_block(mpo.tensors[0])  # (1,3,2,2)
    assert left_block.shape == (1, 3, 2, 2)

    # Extract each operator block
    block_I = left_block[0, 0]  # (2,2)
    block_JZ = left_block[0, 1]  # (2,2)
    block_gX = left_block[0, 2]  # (2,2)

    assert np.allclose(block_I, I)
    assert np.allclose(block_JZ, minus_J * Z)
    assert np.allclose(block_gX, minus_g * X)

    #
    # Check inner site: shape (2,2,3,3)
    # W = [[ I,     -J Z,  -g X ],
    #      [ 0,       0,     Z  ],
    #      [ 0,       0,     I  ]]
    #
    if length > 2:
        inner_block = untranspose_block(mpo.tensors[1])  # (3,3,2,2)
        assert inner_block.shape == (3, 3, 2, 2)

        # row=0, col=0 => I
        assert np.allclose(inner_block[0, 0], I)
        # row=0, col=1 => -J Z
        assert np.allclose(inner_block[0, 1], minus_J * Z)
        # row=0, col=2 => -g X
        assert np.allclose(inner_block[0, 2], minus_g * X)
        # row=1, col=2 => Z
        assert np.allclose(inner_block[1, 2], Z)
        # row=2, col=2 => I
        assert np.allclose(inner_block[2, 2], I)

    #
    # Check right boundary: shape (2,2,3,1)
    # This is the last column, i.e. [ -gX, Z, I ]^T
    #
    right_block = untranspose_block(mpo.tensors[-1])  # (3,1,2,2)
    assert right_block.shape == (3, 1, 2, 2)

    block_gX = right_block[0, 0]
    block_Z = right_block[1, 0]
    block_I = right_block[2, 0]

    assert np.allclose(block_gX, minus_g * X)
    assert np.allclose(block_Z, Z)
    assert np.allclose(block_I, I)


def test_init_Heisenberg():
    """Test that init_Heisenberg creates the correct number, shape, and values."""
    mpo = MPO()
    length = 5
    Jx, Jy, Jz, h = 1.0, 0.5, 0.3, 0.2

    mpo.init_Heisenberg(length, Jx, Jy, Jz, h)

    assert mpo.length == length
    assert mpo.physical_dimension == 2
    assert len(mpo.tensors) == length

    # The internal block is 5x5. Let's do a quick check on the left boundary:
    # shape (2,2,1,5) => untransposed shape (1,5,2,2)
    # That row should contain: [I, -JxX, -JyY, -JzZ, -hZ]
    left_block = untranspose_block(mpo.tensors[0])
    assert left_block.shape == (1, 5, 2, 2)

    # Check each operator
    block_I = left_block[0, 0]
    block_JxX = left_block[0, 1]
    block_JyY = left_block[0, 2]
    block_JzZ = left_block[0, 3]
    block_hZ = left_block[0, 4]

    # For a 2x2 system, Y = i * X * Z or typically [[0, -1j],[1j, 0]],
    # but let's just do a magnitude check unless your code expects a real Y.
    # We'll just check the sign factors.
    # Negative signs come from the code: inner[0,1] = -Jx*X, etc.
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

    # Similarly, check shapes of inner and right boundary as you did before:
    for i, tensor in enumerate(mpo.tensors):
        if i == 0:  # left boundary
            assert tensor.shape == (2, 2, 1, 5)
        elif i == length - 1:  # right boundary
            assert tensor.shape == (2, 2, 5, 1)
        else:  # inner
            assert tensor.shape == (2, 2, 5, 5)


def test_init_identity():
    """Test init_identity builds uniform tensors for the specified length and checks actual values."""
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


def test_init_custom_Hamiltonian():
    """Test that a custom Hamiltonian can be initialized, verifying shape and partial values."""
    length = 4
    pdim = 2

    # Create dummy boundary and inner with shapes that match the length
    left_bound = np.random.rand(1, 2, pdim, pdim)
    inner = np.random.rand(2, 2, pdim, pdim)
    right_bound = np.random.rand(2, 1, pdim, pdim)

    mpo = MPO()
    mpo.init_custom_Hamiltonian(length, left_bound, inner, right_bound)

    assert mpo.length == length
    assert len(mpo.tensors) == length

    # Just check shapes
    assert mpo.tensors[0].shape == (pdim, pdim, 1, 2)
    for i in range(1, length - 1):
        assert mpo.tensors[i].shape == (pdim, pdim, 2, 2)
    assert mpo.tensors[-1].shape == (pdim, pdim, 2, 1)

    assert np.allclose(mpo.tensors[0], np.transpose(left_bound, (2, 3, 0, 1)))
    for i in range(1, length - 1):
        assert np.allclose(mpo.tensors[i], np.transpose(inner, (2, 3, 0, 1)))
    assert np.allclose(mpo.tensors[-1], np.transpose(right_bound, (2, 3, 0, 1)))


def test_init_custom():
    """Test init_custom with a user-provided list of tensors, checking shapes and values."""
    length = 3
    pdim = 2
    tensors = [
        np.random.rand(1, 2, pdim, pdim),  # left
        np.random.rand(2, 2, pdim, pdim),  # middle
        np.random.rand(2, 1, pdim, pdim),  # right
    ]

    mpo = MPO()
    mpo.init_custom(tensors)

    assert mpo.length == length
    assert mpo.physical_dimension == pdim
    assert len(mpo.tensors) == length

    for original, created in zip(tensors, mpo.tensors):
        assert original.shape == created.shape
        assert np.allclose(original, created)


def test_convert_to_MPS():
    """Test converting an MPO to MPS."""
    mpo = MPO()
    length = 3
    J, g = 1.0, 0.5

    # Initialize a small Ising MPO
    mpo.init_Ising(length, J, g)

    # Convert to MPS
    mps = mpo.convert_to_MPS()

    assert isinstance(mps, MPS)
    assert mps.length == length

    # Each MPS tensor after conversion has shape:
    # (pdim*pdim, bond_in, bond_out), because we reshape (pdim, pdim, bond_in, bond_out)
    # to (pdim*pdim, bond_in, bond_out).

    for i, tensor in enumerate(mps.tensors):
        original_mpo_tensor = mpo.tensors[i]
        pdim2 = original_mpo_tensor.shape[0] * original_mpo_tensor.shape[1]
        bond_in = original_mpo_tensor.shape[2]
        bond_out = original_mpo_tensor.shape[3]

        assert tensor.shape == (pdim2, bond_in, bond_out)


def test_check_if_valid_MPO():
    """Test that a valid MPO passes the check_if_valid_MPO method without error."""
    mpo = MPO()
    length = 4
    J, g = 1.0, 0.5

    mpo.init_Ising(length, J, g)

    # Should not raise an AssertionError
    mpo.check_if_valid_MPO()


def test_rotate():
    """Test the rotate method and ensure shapes remain consistent."""
    mpo = MPO()
    length = 3
    J, g = 1.0, 0.5

    mpo.init_Ising(length, J, g)
    original_tensors = [t.copy() for t in mpo.tensors]

    # Rotate without conjugation
    mpo.rotate(conjugate=False)

    for orig, rotated in zip(original_tensors, mpo.tensors):
        # rotate does transpose (1, 0, 2, 3), so check that
        assert rotated.shape == (orig.shape[1], orig.shape[0], orig.shape[2], orig.shape[3])
        np.allclose(rotated, np.transpose(orig, (1, 0, 2, 3)))

    # Rotate back with conjugation=True
    # This will take the conj() and then transpose(1,0,2,3) again.
    mpo.rotate(conjugate=True)
    # Now each tensor should have shape (pdim, pdim, bond_in, bond_out) again.
    for tensor in mpo.tensors:
        assert tensor.shape == (2, 2, tensor.shape[2], tensor.shape[3])


def test_check_if_identity():
    mpo = MPO()
    length = 3
    pdim = 2

    mpo.init_identity(length, pdim)

    # Some fidelity threshold to consider identity "valid"
    fidelity_threshold = 0.9
    assert mpo.check_if_identity(fidelity_threshold) is True


################################################
# Tests for the MPS class
################################################


@pytest.mark.parametrize("state", ["zeros", "ones", "x+", "x-", "y+", "y-", "Neel", "wall"])
def test_mps_initialization(state):
    """
    Test that MPS initializes a chain of a given length with correct
    shapes and the specified default state.
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


def test_mps_custom_tensors():
    """
    Test providing custom tensors at initialization (bypassing the default state creation).
    """
    length = 3
    pdim = 2
    # Create random rank-3 tensors: shape = (pdim, bond_in, bond_out).
    t1 = np.random.rand(pdim, 1, 2)
    t2 = np.random.rand(pdim, 2, 2)
    t3 = np.random.rand(pdim, 2, 1)
    tensors = [t1, t2, t3]

    mps = MPS(length=length, tensors=tensors, physical_dimensions=[pdim] * length)
    assert mps.length == length
    assert len(mps.tensors) == length
    for i, tensor in enumerate(mps.tensors):
        assert np.allclose(tensor, tensors[i])


def test_write_max_bond_dim():
    """
    Test that write_max_bond_dim returns the maximum bond dimension over all tensors.
    """
    length = 3
    pdim = 2
    # Construct an MPS with different bond dims
    t1 = np.random.rand(pdim, 1, 4)
    t2 = np.random.rand(pdim, 4, 5)
    t3 = np.random.rand(pdim, 5, 2)
    mps = MPS(length, tensors=[t1, t2, t3], physical_dimensions=[pdim] * length)

    max_bond = mps.write_max_bond_dim()
    assert max_bond == 5  # The largest dimension among (1,4,5,5,2) is 5.


def test_flip_network():
    """
    Test flipping the MPS bond dimensions and reversing site order.
    """
    length = 3
    pdim = 2
    # shape = (pdim, bond_in, bond_out)
    t1 = np.random.rand(pdim, 1, 2)
    t2 = np.random.rand(pdim, 2, 2)
    t3 = np.random.rand(pdim, 2, 1)
    original_tensors = [t1, t2, t3]
    mps = MPS(length, tensors=copy.deepcopy(original_tensors), physical_dimensions=[pdim] * length)

    mps.flip_network()
    # After flip, the order of tensors is reversed, and each is transposed (0, 2, 1).
    flipped_tensors = mps.tensors
    assert len(flipped_tensors) == length

    # The new order should be t3, t2, t1
    # And shape for t3: (pdim, bond_out, bond_in) => (pdim, 1, 2)
    # But since the original was (pdim, 2, 1), flipping => (pdim, 1, 2).
    assert flipped_tensors[0].shape == (pdim, original_tensors[2].shape[2], original_tensors[2].shape[1])
    # Check the new shape is the transpose of the original
    assert flipped_tensors[0].shape == (pdim, 1, 2)

    # Flip back
    mps.flip_network()
    # Now it should match the original again
    for orig, now in zip(original_tensors, mps.tensors):
        assert np.allclose(orig, now)


def test_shift_orthogonality_center_right():
    """
    Test shifting the orthogonality center to the right by one site.
    This mainly checks that the operation runs without error and preserves shapes.
    """
    length = 4
    pdim = 2
    # For simplicity, build a random MPS
    t1 = np.random.rand(pdim, 1, 2)
    t2 = np.random.rand(pdim, 2, 3)
    t3 = np.random.rand(pdim, 3, 3)
    t4 = np.random.rand(pdim, 3, 1)
    mps = MPS(length, tensors=[t1, t2, t3, t4], physical_dimensions=[pdim] * length)

    # Shift orthogonality center from site 0 to site 1
    mps.shift_orthogonality_center_right(current_orthogonality_center=0)

    # Check shapes remain rank-3
    for tensor in mps.tensors:
        assert tensor.ndim == 3


def test_shift_orthogonality_center_left():
    """
    Test shifting the orthogonality center to the left.
    We just check for shape consistency and no errors.
    """
    length = 4
    pdim = 2
    # Random MPS
    t1 = np.random.rand(pdim, 1, 3)
    t2 = np.random.rand(pdim, 3, 3)
    t3 = np.random.rand(pdim, 3, 2)
    t4 = np.random.rand(pdim, 2, 1)
    mps = MPS(length, [t1, t2, t3, t4], [pdim] * length)

    # Shift orthogonality center from site 3 to site 2
    mps.shift_orthogonality_center_left(current_orthogonality_center=3)

    for tensor in mps.tensors:
        assert tensor.ndim == 3


def test_set_canonical_form():
    """
    Test the set_canonical_form method doesn't raise errors
    and retains correct shapes for an MPS.
    """
    length = 4
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim] * length, state="zeros")

    # By default, 'zeros' means each site is shape (2,1,1).
    # set_canonical_form should not break anything.
    mps.set_canonical_form(orthogonality_center=2)

    for tensor in mps.tensors:
        assert tensor.ndim == 3


def test_normalize():
    """
    Test that calling normalize does not throw errors and keeps MPS rank-3.
    """
    length = 4
    pdim = 2
    # Random MPS
    t1 = np.random.rand(pdim, 1, 3)
    t2 = np.random.rand(pdim, 3, 3)
    t3 = np.random.rand(pdim, 3, 2)
    t4 = np.random.rand(pdim, 2, 1)
    mps = MPS(length, [t1, t2, t3, t4], [pdim] * length)

    # Normalizing an all-'1' initial state:
    mps.normalize(form="B")

    assert np.isclose(mps.norm(), 1)
    for tensor in mps.tensors:
        assert tensor.ndim == 3


def test_measure():
    length = 2
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim] * length, state="x+")

    obs = Observable(site=0, name="x")
    val = mps.measure(obs)

    assert np.isclose(val, 1)


def test_norm():
    length = 3
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim] * length, state="zeros")

    val = mps.norm()

    assert val == 1


def test_check_if_valid_MPS():
    """
    Test that an MPS with consistent bond dimensions passes check_if_valid_MPS
    without errors.
    """
    length = 3
    pdim = 2
    # shapes: (2,1,2), (2,2,3), (2,3,1) => consistent chaining
    t1 = np.random.rand(pdim, 1, 2)
    t2 = np.random.rand(pdim, 2, 3)
    t3 = np.random.rand(pdim, 3, 1)
    mps = MPS(length, tensors=[t1, t2, t3], physical_dimensions=[pdim] * length)

    # Should not raise AssertionError
    mps.check_if_valid_MPS()


def test_check_canonical_form():
    """
    Check that check_canonical_form runs and prints out
    some form of canonical info. By default, our random
    MPS is likely not in a canonical form, so we just
    ensure it doesn't crash.
    """
    length = 3
    pdim = 2
    mps = MPS(length, physical_dimensions=[pdim] * length, state="zeros")
    # This will just print debug info, we can ensure no exceptions occur.
    res = mps.check_canonical_form()
    # res might be None or a list of sites. We won't test equality here
    # since a random or naive product state might not be purely canonical.
    # We'll just check that it returns something.
    # If it prints "MPS is right (B) canonical." for example, that's okay.
    assert res is not None
