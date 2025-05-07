# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Time-Dependent Variational Principle (TDVP) methods in YAQS.

This module contains unit tests for verifying various components of TDVP-based methods,
including:

- Tensor decomposition (splitting and merging) routines for Matrix Product States (MPS)
   and Matrix Product Operators (MPO).
- Environment updates (left and right) and projection routines essential to efficient TDVP implementations.
- Single-site and two-site TDVP algorithms for evolving quantum states under given Hamiltonians.

The tests ensure that:
- Tensor reshaping and decomposition operations maintain numerical accuracy.
- Environment tensors are updated with correct shapes and dimensions.
- TDVP routines yield canonical MPS states with the correct orthogonality center.

These tests confirm the correctness and stability of TDVP-based simulations within the YAQS framework.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs.core.libraries.gate_library import X, Z
from mqt.yaqs.core.methods.tdvp import (
    global_dynamic_tdvp,
    merge_mpo_tensors,
    merge_mps_tensors,
    project_bond,
    project_site,
    single_site_tdvp,
    split_mps_tensor,
    two_site_tdvp,
    update_bond,
    update_left_environment,
    update_right_environment,
    update_site,
)

rng = np.random.default_rng()


def test_split_mps_tensor_left_right_sqrt() -> None:
    """Test splitting of an MPS tensor using different singular value distribution options.

    This test creates a random tensor A with shape (4, 3, 5), corresponding to (d0*d1, D0, D2)
    with d0 = d1 = 2, D0 = 3, and D2 = 5. For each SVD distribution option ("left", "right", "sqrt"),
    the function split_mps_tensor is called to decompose A into two tensors A0 and A1. The test then
    reconstructs A from A0 and A1 by undoing the transpose on A1 and contracting over the singular value index.
    The reconstructed tensor is compared to the original A.
    """
    A = rng.random(size=(4, 3, 5))
    # Placeholder
    measurements = [Observable(Z(), site) for site in range(1)]
    sim_params = PhysicsSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=100,
        threshold=1e-16,
        order=1,
    )
    for distr in ["left", "right", "sqrt"]:
        A0, A1 = split_mps_tensor(A, svd_distribution=distr, sim_params=sim_params, dynamic=False)
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
    A = rng.random(size=(3, 3, 5))
    # Placeholder
    measurements = [Observable(Z(), site) for site in range(1)]
    sim_params = PhysicsSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=100,
        threshold=1e-8,
        order=1,
    )
    with pytest.raises(ValueError, match=r"The first dimension of the tensor must be divisible by 2."):
        split_mps_tensor(A, svd_distribution="left", sim_params=sim_params, dynamic=False)


def test_merge_mps_tensors() -> None:
    """Test the merge_mps_tensors function.

    This test creates two tensors A0 and A1 with shapes (2, 3, 4) and (5, 4, 7), respectively.
    It then merges them via merge_mps_tensors. The expected shape is (10, 3, 7) because
    the contraction is performed over the third axis of A0 and the second axis of A1.
    """
    A0 = rng.random(size=(2, 3, 4))
    A1 = rng.random(size=(5, 4, 7))
    merged = merge_mps_tensors(A0, A1)
    assert merged.shape == (10, 3, 7)


def test_merge_mpo_tensors() -> None:
    """Test the merge_mpo_tensors function.

    This test creates two 4D arrays A0 and A1 with shapes (2, 3, 4, 5) and (7, 8, 5, 9), respectively.
    After merging via merge_mpo_tensors, the expected shape is (14, 24, 4, 9).
    """
    A0 = rng.random(size=(2, 3, 4, 5))
    A1 = rng.random(size=(7, 8, 5, 9))
    merged = merge_mpo_tensors(A0, A1)
    assert merged.shape == (14, 24, 4, 9)


def test_update_right_environment() -> None:
    """Test the update_right_environment function.

    This test creates dummy arrays A, B, W, and R with compatible shapes for the contraction
    operations defined in update_right_environment. It then verifies that the resulting tensor
    has the expected shape (3, 8, 9).
    """
    A = rng.random(size=(2, 3, 4))
    R = rng.random(size=(4, 5, 6))
    W = rng.random(size=(7, 2, 8, 5))
    B = rng.random(size=(7, 9, 6))
    Rnext = update_right_environment(A, B, W, R)
    assert Rnext.shape == (3, 8, 9)


def test_update_left_environment() -> None:
    """Test the update_left_environment function.

    This test constructs dummy arrays A, B, W, and L with compatible shapes for the contraction.
    It then verifies that the output is a 3D tensor.
    """
    A = rng.random(size=(3, 4, 10))
    B = rng.random(size=(7, 6, 8))
    L_arr = rng.random(size=(4, 5, 6))
    W = rng.random(size=(7, 3, 5, 9))
    Rnext = update_left_environment(A, B, W, L_arr)
    assert Rnext.ndim == 3


def test_project_site() -> None:
    """Test the project_site function.

    This test creates dummy tensors A, R, W, and L with appropriate shapes and checks that
    the output of project_site is a 3D tensor.
    """
    A = rng.random(size=(2, 3, 4))
    R = rng.random(size=(4, 5, 6))
    W = rng.random(size=(7, 2, 8, 5))
    L_arr = rng.random(size=(3, 8, 9))
    out = project_site(L_arr, R, W, A)
    assert out.ndim == 3


def test_project_bond() -> None:
    """Test the project_bond function.

    This test creates a bond tensor C and dummy tensors L and R with compatible shapes,
    and verifies that the output has the expected shape (6, 5).
    """
    C = rng.random(size=(2, 3))
    R = rng.random(size=(3, 4, 5))
    L_arr = rng.random(size=(2, 4, 6))
    out = project_bond(L_arr, R, C)
    assert out.shape == (6, 5)


def test_update_site() -> None:
    """Test the update_site function.

    This test creates a dummy MPS tensor A (shape (2,2,4)), along with tensors L, R, and W,
    and applies update_site with a small time step and a fixed number of Lanczos iterations.
    The output should have the same shape as the input tensor A.
    """
    A = rng.random(size=(2, 2, 4))
    R = rng.random(size=(4, 1, 4))
    W = rng.random(size=(2, 2, 1, 1))
    L_arr = rng.random(size=(2, 1, 2))
    dt = 0.05
    lanczos_iterations = 10
    out = update_site(L_arr, R, W, A, dt, lanczos_iterations)
    assert out.shape == A.shape, f"Expected shape {A.shape}, got {out.shape}"


def test_update_bond() -> None:
    """Test the update_bond function.

    This test creates a square bond tensor C and compatible dummy tensors R and L.
    It applies update_bond and checks that the output shape matches that of C.
    """
    C = rng.random(size=(2, 2))
    R = rng.random(size=(2, 2, 2))
    L_arr = rng.random(size=(2, 2, 2))
    dt = 0.05
    lanczos_iterations = 10
    out = update_bond(L_arr, R, C, dt, lanczos_iterations)
    assert out.shape == C.shape, f"Expected shape {C.shape}, got {out.shape}"


def test_single_site_tdvp() -> None:
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
    H.init_ising(L, J, g)
    state = MPS(L, state="zeros")
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=4,
        threshold=1e-6,
        order=1,
    )
    single_site_tdvp(state, H, sim_params, numiter_lanczos=5)
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)
    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after single-site TDVP, but got canonical site: {canonical_site}"
    )


def test_two_site_tdvp() -> None:
    """Test the two_site_TDVP function.

    This test initializes an Ising MPO and an MPS of length 5, sets up PhysicsSimParams,
    and runs two_site_TDVP. It checks that the MPS retains the correct number of tensors,
    that all tensors remain numpy arrays, and that the MPS is in canonical form with the orthogonality center at site 0.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L, state="zeros")
    ref_mps = deepcopy(state)
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=16,
        threshold=1e-12,
        order=1,
    )
    two_site_tdvp(state, H, sim_params, numiter_lanczos=25)
    assert state.length == L
    for tensor in state.tensors:
        assert isinstance(tensor, np.ndarray)
    canonical_site = state.check_canonical_form()[0]
    assert canonical_site == 0, (
        f"MPS should be site-canonical at site 0 after two-site TDVP, but got canonical site: {canonical_site}"
    )
    # Check against exact evolution
    state_vec = ref_mps.to_vec()
    H_mat = H.to_matrix()
    U = expm(-1j * sim_params.dt * H_mat)
    state_vec = U @ state_vec
    found = state.to_vec()
    assert np.allclose(state_vec, found)


def test_dynamic_tdvp_one_site() -> None:
    """Test dynamic TDVP, single site.

    Test that dynamic_TDVP calls single_site_TDVP exactly once when the current maximum bond dimension
    exceeds sim_params.max_bond_dim.

    In this test, sim_params.max_bond_dim is set to 0 so that the current maximum bond dimension of the MPS,
    computed by state.write_max_bond_dim(), is greater than 0. Therefore, the else branch of dynamic_TDVP should be
    taken, and single_site_tdvp should be called exactly once.
    """
    # Define the system Hamiltonian.
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)

    # Define the initial state.
    state = MPS(L, state="zeros")

    # Define simulation parameters with max_bond_dim set to 0.
    elapsed_time = 0.2
    dt = 0.1
    sample_timesteps = False
    num_traj = 1
    max_bond_dim = 0  # Force condition for single_site_TDVP.
    threshold = 1e-6
    order = 1
    measurements = [Observable(X(), site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements, elapsed_time, dt, num_traj, max_bond_dim, threshold, order, sample_timesteps=sample_timesteps
    )

    with patch("mqt.yaqs.core.methods.tdvp.single_site_tdvp") as mock_single_site:
        global_dynamic_tdvp(state, H, sim_params)
        mock_single_site.assert_called_once_with(state, H, sim_params)


def test_dynamic_tdvp_two_site() -> None:
    """Test dynamic TDVP, two site.

    Test that dynamic_TDVP calls two_site_TDVP exactly once when the current maximum bond dimension
    is less than or equal to sim_params.max_bond_dim.

    In this test, sim_params.max_bond_dim is set to 2, so if the current maximum bond dimension is ≤ 2,
    the if branch of dynamic_TDVP is executed and two_site_TDVP is called exactly once.
    """
    # Define the system Hamiltonian.
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)

    # Define the initial state.
    state = MPS(L, state="zeros")

    # Define simulation parameters with max_bond_dim set to 2.
    elapsed_time = 0.2
    dt = 0.1
    sample_timesteps = False
    num_traj = 1
    max_bond_dim = 8  # Force condition for two_site_tdvp.
    threshold = 1e-6
    order = 1
    measurements = [Observable(X(), site) for site in range(L)]
    sim_params = PhysicsSimParams(
        measurements, elapsed_time, dt, num_traj, max_bond_dim, threshold, order, sample_timesteps=sample_timesteps
    )

    with patch("mqt.yaqs.core.methods.tdvp.two_site_tdvp") as mock_two_site:
        global_dynamic_tdvp(state, H, sim_params)
        mock_two_site.assert_called_once_with(state, H, sim_params, dynamic=True)
