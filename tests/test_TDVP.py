import numpy as np
import pytest

from yaqs.core.data_structures.networks import MPO, MPS
from yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams

from yaqs.core.methods.TDVP import (
    _split_mps_tensor,
    _merge_mps_tensor_pair,
    _merge_mpo_tensor_pair,
    _contraction_operator_step_right,
    _contraction_operator_step_left,
    _compute_right_operator_blocks,
    _apply_local_hamiltonian,
    _apply_local_bond_contraction,
    _local_hamiltonian_step,
    _local_bond_step,
    single_site_TDVP,
    two_site_TDVP,
)

def test_split_mps_tensor_left_right_sqrt():
    # Create an input tensor A with shape (d0*d1, D0, D2).
    # Let d0 = d1 = 2 so that A.shape[0]=4, and choose D0=3, D2=5.
    A = np.random.randn(4, 3, 5)
    # For each svd_distr option, run the splitting and then reconstruct A.
    for distr in ['left', 'right', 'sqrt']:
        A0, A1 = _split_mps_tensor(A, svd_distr=distr, threshold=1e-8)
        # A0 should have shape (2, 3, r) and A1 should have shape (2, r, 5),
        # where r is the effective rank.
        assert A0.ndim == 3
        assert A1.ndim == 3
        r = A0.shape[2]
        assert A1.shape[1] == r
        # Reconstruct A: undo the transpose on A1 then contract A0 and A1.
        A1_recon = A1.transpose((1, 0, 2))  # now shape (r, 2, 5)
        # Contract along the rank index:
        # A0 has indices: (d0, D0, r), A1_recon has indices: (r, d1, D2).
        # Form a tensor of shape (d0, d1, D0, D2) then reshape back to (4, 3, 5)
        A_recon = np.tensordot(A0, A1_recon, axes=(2, 0))  # shape (2, 3, 2, 5)
        A_recon = A_recon.transpose((0,2,1,3)).reshape(4, 3, 5)
        # Up to SVD sign and ordering ambiguities, the reconstruction should be close.
        np.testing.assert_allclose(A, A_recon, atol=1e-6)

def test_split_mps_tensor_invalid_shape():
    # If A.shape[0] is not divisible by 2, the function should raise a ValueError.
    A = np.random.randn(3, 3, 5)
    with pytest.raises(ValueError):
        _split_mps_tensor(A, svd_distr='left')

def test_merge_mps_tensor_pair():
    # Let A0 have shape (2, 3, 4) and A1 have shape (5, 4, 7).
    # In _merge_mps_tensor_pair the arrays are interpreted with label tuples
    # (0,2,3) for A0 and (1,3,4) for A1, so contraction is over the third axis of A0 and
    # the second axis of A1.
    A0 = np.random.randn(2, 3, 4)
    A1 = np.random.randn(5, 4, 7)
    merged = _merge_mps_tensor_pair(A0, A1)
    # Expected shape: first two axes merged from A0[0] and A1[0]:
    # output shape = ((2*5), 3, 7) i.e. (10, 3, 7)
    assert merged.shape == (10, 3, 7)

def test_merge_mpo_tensor_pair():
    # Let A0 be a 4D array with shape (2, 3, 4, 5) and
    # A1 be a 4D array with shape (7, 8, 5, 9).
    # The contract call uses label tuples (0,2,4,6) and (1,3,6,5) and contracts label 6.
    A0 = np.random.randn(2, 3, 4, 5)
    A1 = np.random.randn(7, 8, 5, 9)
    merged = _merge_mpo_tensor_pair(A0, A1)
    # Expected shape: merge first two axes of the result, where result (before reshape)
    # should have shape (2,7,3,8,4,9), then merged to (2*7, 3*8, 4,9) = (14,24,4,9).
    assert merged.shape == (14, 24, 4, 9)

def test_contraction_operator_step_right():
    # Choose shapes as described in the function.
    A = np.random.randn(2, 3, 4)
    # R: contract A's last axis (size 4) with R's first axis.
    R = np.random.randn(4, 5, 6)
    # W must have shape with axes (1,3) matching T from tensordot(A, R, 1) of shape (2,3,5,6):
    # We require W.shape[1]=2 and W.shape[3]=5. Let W = (7,2,8,5)
    W = np.random.randn(7, 2, 8, 5)
    # B: choose shape so that B.conj() has axes (0,2) matching T from later step.
    # After the previous steps, T becomes shape (3,8,7,6); we contract with B.conj() axes ((2,3),(0,2)).
    # So let B be of shape (7, 9, 6).
    B = np.random.randn(7, 9, 6)
    Rnext = _contraction_operator_step_right(A, B, W, R)
    # Expected shape: (3,8,9) (from the discussion above).
    assert Rnext.shape == (3, 8, 9)

def test_contraction_operator_step_left():
    # Set up dummy arrays with shapes so that the contraction is valid.
    # Let A be shape (3,4,10)
    A = np.random.randn(3, 4, 10)
    # Let B be shape (7, 6, 8) so that B.conj() has shape (7,6,8).
    B = np.random.randn(7, 6, 8)
    # Let L be shape (4,5,6) and require that L.shape[2] (6) matches B.shape[1] (6).
    L_arr = np.random.randn(4, 5, 6)
    # Let W be shape (7,3,5,9) so that we contract axes ((0,2),(2,1)) with T later.
    W = np.random.randn(7, 3, 5, 9)
    Rnext = _contraction_operator_step_left(A, B, W, L_arr)
    # The expected shape from our reasoning is (10,9,8) (A's remaining axis 2 becomes output along with leftover dims from T).
    # We check that the result is 3-dimensional.
    assert Rnext.ndim == 3

def test_apply_local_hamiltonian():
    # Let A: shape (2,3,4); R: shape (4,5,6) as before.
    A = np.random.randn(2, 3, 4)
    R = np.random.randn(4, 5, 6)
    # Let W be shape (7,2,8,5) as in test above.
    W = np.random.randn(7, 2, 8, 5)
    # Let L be shape (3,8,9) so that tensordot works.
    L_arr = np.random.randn(3, 8, 9)
    out = _apply_local_hamiltonian(L_arr, R, W, A)
    # The function transposes the final result so we expect a 3D array.
    assert out.ndim == 3

def test_apply_local_bond_contraction():
    # Let C: shape (2,3)
    C = np.random.randn(2, 3)
    # Let R: shape (3,4,5)
    R = np.random.randn(3, 4, 5)
    # Let L: shape (2,4,6)
    L_arr = np.random.randn(2, 4, 6)
    out = _apply_local_bond_contraction(L_arr, R, C)
    # Expected output shape: contraction gives shape (6,5)
    assert out.shape == (6, 5)

def test_local_hamiltonian_step():
    # We choose an MPS tensor A with shape (d0, d0, d1) where d0=2, d1=4.
    # (The requirement here is that the first two dimensions are equal,
    # so that after the contraction chain the operator is square.)
    A = np.random.randn(2, 2, 4)   # total elements: 2*2*4 = 16
    # Choose R of shape (d1, X, d1) with d1=4 and X=1.
    R = np.random.randn(4, 1, 4)     # shape: (4,1,4)
    # Choose W of shape (d0, d0, X, X) with d0=2 and X=1.
    W = np.random.randn(2, 2, 1, 1)   # shape: (2,2,1,1)
    # Choose L so that the contraction makes sense.
    # In our contraction chain, after:
    #   T1 = tensordot(A, R, axes=1)  → shape (2,2,1,4)
    #   T2 = tensordot(W, T1, axes=((1,3),(0,2))) → shape (2,1,2,4)
    #   T3 = T2.transpose((2,1,0,3)) → shape (2,1,2,4) (here the permutation reorders axes)
    # Then we contract T3 with L along axes ((2,1),(0,1)).
    # To contract T3’s axes (axis2, axis1) = (2,1) we need L with shape (2,1,r).
    # Then T4 will have shape (remaining T3 axes: (axis0, axis3)) plus L’s remaining axis, i.e. (2,4,r).
    # Finally, a transpose (here, (0,2,1)) gives shape (2, r, 4).
    # We want the final shape to equal A’s shape (2,2,4), so we set r=2.
    L_arr = np.random.randn(2, 1, 2)  # shape: (2,1,2)
    dt = 0.05
    numiter = 10
    out = _local_hamiltonian_step(L_arr, R, W, A, dt, numiter)
    # The operator should be square, so out.shape should equal A.shape.
    assert out.shape == A.shape, f"Expected shape {A.shape}, got {out.shape}"

def test_local_bond_step():
    # For the bond step we want the operator to be square.
    # Let C be a matrix of shape (p, q). We choose C to be square.
    C = np.random.randn(2, 2)  # total elements: 4
    # Choose R of shape (q, r, s). To contract C and get back the same number of elements,
    # we can choose R such that q=2, r=2, s=2.
    R = np.random.randn(2, 2, 2)  # shape: (2,2,2)
    # Choose L of shape (p, r, t) with p=2 and r=2.
    # We want the final result to have shape (p, q) = (2,2); so t must equal q=2.
    L_arr = np.random.randn(2, 2, 2)  # shape: (2,2,2)
    dt = 0.05
    numiter = 10
    out = _local_bond_step(L_arr, R, C, dt, numiter)
    # The output shape should equal the input shape.
    assert out.shape == C.shape, f"Expected shape {C.shape}, got {out.shape}"

def test_single_site_TDVP():
    L = 5
    d = 2
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, d, J, g)
    state = MPS(L)
    measurements = [Observable('z', site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T=0.2, dt=0.1, sample_timesteps=True, N=1, max_bond_dim=4, threshold=1e-6, order=1)
    single_site_TDVP(state, H, sim_params, numiter_lanczos=5)
    # Check that state still has L tensors and that each tensor is a numpy array.
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)

    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after single-site TDVP, "
        f"but got canonical site: {canonical_site}"
    )

def test_two_site_TDVP():
    L = 5
    d = 2
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, d, J, g)
    state = MPS(L)
    measurements = [Observable('z', site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T=0.2, dt=0.1, sample_timesteps=True, N=1, max_bond_dim=4, threshold=1e-6, order=1)
    two_site_TDVP(state, H, sim_params, numiter_lanczos=5)
    # Check that state still has L tensors and that each tensor is a numpy array.
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)

    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after two-site TDVP, "
        f"but got canonical site: {canonical_site}"
    )