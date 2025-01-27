import copy
import numpy as np
import pytest

from yaqs.general.data_structures.MPO import MPO
from yaqs.general.data_structures.MPS import MPS
from yaqs.general.data_structures.simulation_parameters import Observable

################################################
# Tests for the MPS class
################################################

def test_init_Ising():
    """Test that init_Ising creates the correct number and shape of tensors."""
    mpo = MPO()
    length = 4
    pdim = 2
    J = 1.0
    g = 0.5

    mpo.init_Ising(length, pdim, J, g)

    assert mpo.length == length
    assert mpo.physical_dimension == pdim
    assert len(mpo.tensors) == length

    # Check shapes of each tensor:
    # After transpose, each tensor is (pdim, pdim, bond_in, bond_out).
    # For a length-4 chain:
    #   - The left boundary has shape (pdim, pdim, 1, 3)
    #   - The middle sites have shape (pdim, pdim, 3, 3)
    #   - The right boundary has shape (pdim, pdim, 3, 1)

    for i, tensor in enumerate(mpo.tensors):
        if i == 0:  # left boundary
            assert tensor.shape == (pdim, pdim, 1, 3)
        elif i == length - 1:  # right boundary
            assert tensor.shape == (pdim, pdim, 3, 1)
        else:  # inner
            assert tensor.shape == (pdim, pdim, 3, 3)


def test_init_Heisenberg():
    """Test that init_Heisenberg creates the correct number and shape of tensors."""
    mpo = MPO()
    length = 5
    pdim = 2
    Jx, Jy, Jz, h = 1.0, 0.5, 0.3, 0.2

    mpo.init_Heisenberg(length, pdim, Jx, Jy, Jz, h)

    assert mpo.length == length
    assert mpo.physical_dimension == pdim
    assert len(mpo.tensors) == length

    # After transpose, each tensor is (pdim, pdim, bond_in, bond_out).
    # For Heisenberg with a 5x5 internal block, the boundary shapes differ:
    #   - Left boundary: (pdim, pdim, 1, 5)
    #   - Inner: (pdim, pdim, 5, 5)
    #   - Right boundary: (pdim, pdim, 5, 1)

    for i, tensor in enumerate(mpo.tensors):
        if i == 0:  # left boundary
            assert tensor.shape == (pdim, pdim, 1, 5)
        elif i == length - 1:  # right boundary
            assert tensor.shape == (pdim, pdim, 5, 1)
        else:  # inner
            assert tensor.shape == (pdim, pdim, 5, 5)


def test_init_identity():
    """Test init_identity builds uniform tensors for the specified length."""
    mpo = MPO()
    length = 3
    pdim = 2

    mpo.init_identity(length, physical_dimension=pdim)

    assert mpo.length == length
    assert mpo.physical_dimension == pdim
    assert len(mpo.tensors) == length

    # Each tensor for identity is shape (2,2) with 2D expansions,
    # but note that we do not transpose here in the code. The method
    # uses `np.expand_dims(M, (2,3))`, so each final shape is (2,2,1,1).
    for tensor in mpo.tensors:
        assert tensor.shape == (2, 2, 1, 1)


def test_init_custom_Hamiltonian():
    """Test that a custom Hamiltonian can be initialized."""
    length = 4
    pdim = 2

    # Create dummy boundary and inner with shapes that match the length
    # Suppose each site has a 2D bond dimension just as an example
    left_bound = np.random.rand(1, 2, pdim, pdim)
    inner = np.random.rand(2, 2, pdim, pdim)
    right_bound = np.random.rand(2, 1, pdim, pdim)

    mpo = MPO()
    mpo.init_custom_Hamiltonian(length, left_bound, inner, right_bound)

    assert mpo.length == length
    assert len(mpo.tensors) == length

    # After init, no transpose is done explicitly in init_custom_Hamiltonian.
    # If you need a transpose like in the other inits, you might do that externally
    # or within the method. We'll assume shapes remain as-is for a direct test.
    assert mpo.tensors[0].shape == (1, 2, pdim, pdim)
    for i in range(1, length - 1):
        assert mpo.tensors[i].shape == (2, 2, pdim, pdim)
    assert mpo.tensors[-1].shape == (2, 1, pdim, pdim)


def test_init_custom():
    """Test init_custom with a user-provided list of tensors."""
    length = 3
    pdim = 2
    tensors = [
        np.random.rand(pdim, pdim, 1, 2),   # left
        np.random.rand(pdim, pdim, 2, 2),   # middle
        np.random.rand(pdim, pdim, 2, 1)    # right
    ]

    mpo = MPO()
    mpo.init_custom(tensors)

    assert mpo.length == length
    assert mpo.physical_dimension == pdim
    assert len(mpo.tensors) == length

    for original, created in zip(tensors, mpo.tensors):
        assert np.allclose(original, created)


def test_convert_to_MPS():
    """Test converting an MPO to MPS."""
    mpo = MPO()
    length = 3
    pdim = 2
    J, g = 1.0, 0.5

    # Initialize a small Ising MPO
    mpo.init_Ising(length, pdim, J, g)

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
    pdim = 2
    J, g = 1.0, 0.5

    mpo.init_Ising(length, pdim, J, g)

    # Should not raise an AssertionError
    mpo.check_if_valid_MPO()


def test_rotate():
    """Test the rotate method and ensure shapes remain consistent."""
    mpo = MPO()
    length = 3
    pdim = 2
    J, g = 1.0, 0.5

    mpo.init_Ising(length, pdim, J, g)
    original_tensors = [t.copy() for t in mpo.tensors]

    # Rotate without conjugation
    mpo.rotate(conjugate=False)

    for orig, rotated in zip(original_tensors, mpo.tensors):
        # rotate does transpose (1, 0, 2, 3), so check that
        assert rotated.shape == (orig.shape[1], orig.shape[0], orig.shape[2], orig.shape[3])
        # If we wanted to test exact values, we could check each element:
        # np.allclose(rotated, np.transpose(orig, (1,0,2,3)))

    # Rotate back with conjugation=True
    # This will take the conj() and then transpose(1,0,2,3) again.
    mpo.rotate(conjugate=True)
    # Now each tensor should have shape (pdim, pdim, bond_in, bond_out) again.
    for tensor in mpo.tensors:
        assert tensor.shape == (pdim, pdim, tensor.shape[2], tensor.shape[3])


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


@pytest.mark.parametrize("state", ["zeros", "ones", "x", "Neel", "wall"])
def test_mps_initialization(state):
    """
    Test that MPS initializes a chain of a given length with correct
    shapes and the specified default state.
    """
    length = 4
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim]*length, state=state)

    assert mps.length == length
    assert len(mps.tensors) == length
    assert all(d == pdim for d in mps.physical_dimensions)

    # For a freshly initialized MPS, each tensor is shape (pdim, 1, 1) 
    # if we have no entanglement (i.e., just a single site 'ket').
    # But if we specify length=4, we might expect the shape to be 
    # (pdim, bond_in, bond_out). For the default code, bond_in=1, bond_out=1
    # for each site. 
    for tensor in mps.tensors:
        assert tensor.ndim == 3
        assert tensor.shape[0] == pdim
        # Usually for a "product state" MPS, the middle dims are all (1,1) 
        # except if there's some code that introduces entanglement. 
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

    mps = MPS(length=length, tensors=tensors, physical_dimensions=[pdim]*length)
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
    mps = MPS(length, tensors=[t1, t2, t3], physical_dimensions=[pdim]*length)

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
    mps = MPS(length, tensors=copy.deepcopy(original_tensors), physical_dimensions=[pdim]*length)

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
    mps = MPS(length, tensors=[t1, t2, t3, t4], physical_dimensions=[pdim]*length)

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
    mps = MPS(length, [t1, t2, t3, t4], [pdim]*length)

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
    mps = MPS(length=length, physical_dimensions=[pdim]*length, state='zeros')

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
    mps = MPS(length=length, physical_dimensions=[pdim]*length, state='ones')

    # Normalizing an all-'1' initial state:
    mps.normalize(form='B')

    for tensor in mps.tensors:
        assert tensor.ndim == 3


def test_measure():
    length = 2
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim]*length, state='x')

    obs = Observable(site=0, name="x")
    val = mps.measure(obs)

    assert np.isclose(val, 1)
    

def test_norm():
    length = 3
    pdim = 2
    mps = MPS(length=length, physical_dimensions=[pdim]*length, state='zeros')

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
    mps = MPS(length, tensors=[t1, t2, t3], physical_dimensions=[pdim]*length)

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
    mps = MPS(length, physical_dimensions=[pdim]*length, state='zeros')
    # This will just print debug info, we can ensure no exceptions occur.
    res = mps.check_canonical_form()
    # res might be None or a list of sites. We won't test equality here
    # since a random or naive product state might not be purely canonical.
    # We'll just check that it returns something.
    # If it prints "MPS is right (B) canonical." for example, that's okay.
    assert res is not None
