# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs.core.methods.TDVP import (
    merge_mpo_tensors,
    merge_mps_tensors,
    project_bond,
    project_site,
    single_site_TDVP,
    split_mps_tensor,
    two_site_TDVP,
    update_bond,
    update_left_environment,
    update_right_environment,
    update_site,
)


def test_split_mps_tensor_left_right_sqrt() -> None:
    """Test splitting of an MPS tensor using different singular value distribution options.

    This test creates a random tensor A with shape (4, 3, 5), corresponding to (d0*d1, D0, D2)
    with d0 = d1 = 2, D0 = 3, and D2 = 5. For each SVD distribution option ("left", "right", "sqrt"),
    the function split_mps_tensor is called to decompose A into two tensors A0 and A1. The test then
    reconstructs A from A0 and A1 by undoing the transpose on A1 and contracting over the singular value index.
    The reconstructed tensor is compared to the original A.
    """
    A = np.random.randn(4, 3, 5)
    for distr in ["left", "right", "sqrt"]:
        A0, A1 = split_mps_tensor(A, svd_distribution=distr, threshold=1e-8)
        # A0 should have shape (2, 3, r) and A1 should have shape (2, r, 5), where r is the effective rank.
        assert A0.ndim == 3
        assert A1.ndim == 3
        r = A0.shape[2]
        assert A1.shape[1] == r
        # Reconstruct A: first undo the transpose on A1 so that its shape becomes (r, 2, 5)
        A1_recon = A1.transpose((1, 0, 2))
        # Contract A0 (indices: d0, D0, r) with A1_recon (indices: r, d1, D2) over the rank index r.
        # This yields a tensor of shape (d0, d1, D0, D2). Then, we transpose to (d0*d1, D0, D2)
        A_recon = np.tensordot(A0, A1_recon, axes=(2, 0))  # shape (2, 3, 2, 5)
        A_recon = A_recon.transpose((0, 2, 1, 3)).reshape(4, 3, 5)
        np.testing.assert_allclose(A, A_recon, atol=1e-6)


def test_split_mps_tensor_invalid_shape() -> None:
    """Test that split_mps_tensor raises a ValueError when the input tensor's first dimension is not divisible by 2.

    This test creates a tensor A with shape (3, 3, 5) and expects the function to raise an error.
    """
    A = np.random.randn(3, 3, 5)
    with pytest.raises(ValueError):
        split_mps_tensor(A, svd_distribution="left")


def test_merge_mps_tensors() -> None:
    """Test the merge_mps_tensors function.

    This test creates two tensors A0 and A1 with shapes (2, 3, 4) and (5, 4, 7), respectively.
    It then merges them via merge_mps_tensors. The expected shape is (10, 3, 7) because
    the contraction is performed over the third axis of A0 and the second axis of A1.
    """
    A0 = np.random.randn(2, 3, 4)
    A1 = np.random.randn(5, 4, 7)
    merged = merge_mps_tensors(A0, A1)
    assert merged.shape == (10, 3, 7)


def test_merge_mpo_tensors() -> None:
    """Test the merge_mpo_tensors function.

    This test creates two 4D arrays A0 and A1 with shapes (2, 3, 4, 5) and (7, 8, 5, 9), respectively.
    After merging via merge_mpo_tensors, the expected shape is (14, 24, 4, 9).
    """
    A0 = np.random.randn(2, 3, 4, 5)
    A1 = np.random.randn(7, 8, 5, 9)
    merged = merge_mpo_tensors(A0, A1)
    assert merged.shape == (14, 24, 4, 9)


def test_update_right_environment() -> None:
    """Test the update_right_environment function.

    This test creates dummy arrays A, B, W, and R with compatible shapes for the contraction
    operations defined in update_right_environment. It then verifies that the resulting tensor
    has the expected shape (3, 8, 9).
    """
    A = np.random.randn(2, 3, 4)
    R = np.random.randn(4, 5, 6)
    W = np.random.randn(7, 2, 8, 5)
    B = np.random.randn(7, 9, 6)
    Rnext = update_right_environment(A, B, W, R)
    assert Rnext.shape == (3, 8, 9)


def test_update_left_environment() -> None:
    """Test the update_left_environment function.

    This test constructs dummy arrays A, B, W, and L with compatible shapes for the contraction.
    It then verifies that the output is a 3D tensor.
    """
    A = np.random.randn(3, 4, 10)
    B = np.random.randn(7, 6, 8)
    L_arr = np.random.randn(4, 5, 6)
    W = np.random.randn(7, 3, 5, 9)
    Rnext = update_left_environment(A, B, W, L_arr)
    assert Rnext.ndim == 3


def test_project_site() -> None:
    """Test the project_site function.

    This test creates dummy tensors A, R, W, and L with appropriate shapes and checks that
    the output of project_site is a 3D tensor.
    """
    A = np.random.randn(2, 3, 4)
    R = np.random.randn(4, 5, 6)
    W = np.random.randn(7, 2, 8, 5)
    L_arr = np.random.randn(3, 8, 9)
    out = project_site(L_arr, R, W, A)
    assert out.ndim == 3


def test_project_bond() -> None:
    """Test the project_bond function.

    This test creates a bond tensor C and dummy tensors L and R with compatible shapes,
    and verifies that the output has the expected shape (6, 5).
    """
    C = np.random.randn(2, 3)
    R = np.random.randn(3, 4, 5)
    L_arr = np.random.randn(2, 4, 6)
    out = project_bond(L_arr, R, C)
    assert out.shape == (6, 5)


def test_update_site() -> None:
    """Test the update_site function.

    This test creates a dummy MPS tensor A (shape (2,2,4)), along with tensors L, R, and W,
    and applies update_site with a small time step and a fixed number of Lanczos iterations.
    The output should have the same shape as the input tensor A.
    """
    A = np.random.randn(2, 2, 4)
    R = np.random.randn(4, 1, 4)
    W = np.random.randn(2, 2, 1, 1)
    L_arr = np.random.randn(2, 1, 2)
    dt = 0.05
    numiter = 10
    out = update_site(L_arr, R, W, A, dt, numiter)
    assert out.shape == A.shape, f"Expected shape {A.shape}, got {out.shape}"


def test_update_bond() -> None:
    """Test the update_bond function.

    This test creates a square bond tensor C and compatible dummy tensors R and L.
    It applies update_bond and checks that the output shape matches that of C.
    """
    C = np.random.randn(2, 2)
    R = np.random.randn(2, 2, 2)
    L_arr = np.random.randn(2, 2, 2)
    dt = 0.05
    numiter = 10
    out = update_bond(L_arr, R, C, dt, numiter)
    assert out.shape == C.shape, f"Expected shape {C.shape}, got {out.shape}"


def test_single_site_TDVP() -> None:
    """Test the single_site_TDVP function.

    This test initializes an Ising MPO and an MPS of length 5 (initialized to 'zeros'),
    along with PhysicsSimParams configured for a single trajectory update.
    It runs single_site_TDVP and verifies that the MPS remains of the same length, all tensors are numpy arrays,
    and the MPS is left in a canonical form with the orthogonality center at site 0.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, J, g)
    state = MPS(L, state="zeros")
    measurements = [Observable("z", site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements, T=0.2, dt=0.1, sample_timesteps=True, N=1, max_bond_dim=4, threshold=1e-6, order=1
    )
    single_site_TDVP(state, H, sim_params, numiter_lanczos=5)
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)
    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after single-site TDVP, but got canonical site: {canonical_site}"
    )


def test_two_site_TDVP() -> None:
    """Test the two_site_TDVP function.

    This test initializes an Ising MPO and an MPS of length 5, sets up PhysicsSimParams,
    and runs two_site_TDVP. It checks that the MPS retains the correct number of tensors,
    that all tensors remain numpy arrays, and that the MPS is in canonical form with the orthogonality center at site 0.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_Ising(L, J, g)
    state = MPS(L, state="zeros")
    measurements = [Observable("z", site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements, T=0.2, dt=0.1, sample_timesteps=True, N=1, max_bond_dim=4, threshold=1e-6, order=1
    )
    two_site_TDVP(state, H, sim_params, numiter_lanczos=5)
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)
    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after two-site TDVP, but got canonical site: {canonical_site}"
    )
