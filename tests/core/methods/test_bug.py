# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Basis-Update Galerkin (BUG) method."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import PhysicsSimParams
from mqt.yaqs.core.methods.bug import (
    bug,
    build_basis_change_tensor,
    choose_stack_tensor,
    find_new_q,
    local_update,
    prepare_canonical_site_tensors,
)
from mqt.yaqs.core.methods.decompositions import right_qr
from mqt.yaqs.core.methods.tdvp import update_left_environment

if TYPE_CHECKING:
    from numpy.typing import NDArray


def crandn(
    size: int | tuple[int, ...], *args: int, seed: np.random.Generator | int | None = None
) -> NDArray[np.complex128]:
    """Draw random samples from the standard complex normal distribution.

    Args:
        size (int |Tuple[int,...]): The size/shape of the output array.
        *args (int): Additional dimensions for the output array.
        seed (Generator | int): The seed for the random number generator.

    Returns:
        NDArray[np.complex128]: The array of random complex numbers.
    """
    if isinstance(size, int) and len(args) > 0:
        size = (size, *list(args))
    elif isinstance(size, int):
        size = (size,)
    rng = np.random.default_rng(seed)
    # 1/sqrt(2) is a normalization factor
    return (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)


def random_mps(shapes: list[tuple[int, int, int]]) -> MPS:
    """Create a random MPS with the given shapes.

    Args:
        shapes (List[Tuple[int, int, int]]): The shapes of the tensors in the
            MPS.

    Returns:
        MPS: The random MPS.
    """
    tensors = [crandn(shape) for shape in shapes]
    mps = MPS(len(shapes), tensors=tensors)
    mps.normalize()
    return mps


def random_mpo(shapes: list[tuple[int, int, int, int]]) -> MPO:
    """Create a random MPO with the given shapes.

    Args:
        shapes (List[Tuple[int, int, int, int]]): The shapes of the tensors in
            the MPO.

    Returns:
        MPO: The random MPO.
    """
    tensors = [crandn(shape) for shape in shapes]
    mpo = MPO()
    mpo.init_custom(tensors, transpose=False)
    return mpo


def test_prepare_canonical_site_tensors_single_site() -> None:
    """Tests the preparation for a single site MPS.

    The the preparation of the canonical sites tensors and left envs for a
    length 1 MPS.
    """
    mps_tensor = crandn(2, 3, 4)
    mps = MPS(1, tensors=[mps_tensor])
    ref_mps = deepcopy(mps)
    mpo_tensor = crandn(2, 2, 1, 1)
    mpo = MPO()
    mpo.init_custom([mpo_tensor])
    canon_sites, left_envs = prepare_canonical_site_tensors(mps, mpo)
    assert mps.almost_equal(ref_mps)
    assert len(left_envs) == 1
    assert len(canon_sites) == 1
    correct_env = np.eye(3).reshape(3, 1, 3)
    assert np.allclose(correct_env, left_envs[0])
    correct_canon = mps_tensor
    assert np.allclose(correct_canon, canon_sites[0])


def test_prepare_canonical_site_tensors_three_sites() -> None:
    """Tests the preparation for a three site MPS.

    The preparation of the canonical sites tensors and left envs for a
    length 3 MPS.
    """
    shapes = [(2, 3, 4), (2, 4, 5), (2, 5, 3)]
    mps_tensors = [crandn(shape) for shape in shapes]
    mps = MPS(3, tensors=mps_tensors)
    ref_mps = deepcopy(mps)
    shapes2 = [(2, 2, 1, 3), (2, 2, 3, 4), (2, 2, 4, 1)]
    mpo_tensors = [crandn(shape) for shape in shapes2]
    mpo = MPO()
    mpo.init_custom(mpo_tensors, transpose=False)
    canon_sites, left_envs = prepare_canonical_site_tensors(mps, mpo)
    assert mps.almost_equal(ref_mps)
    assert len(left_envs) == 3
    assert len(canon_sites) == 3
    # Correct envs and canon sites
    # Site 0
    correct_env = np.eye(3).reshape(3, 1, 3)
    correct_canon = mps_tensors[0]
    assert np.allclose(correct_env, left_envs[0])
    assert np.allclose(correct_canon, canon_sites[0])
    # Site 1
    q_last, r_matrix = right_qr(mps_tensors[0])
    correct_canon = np.tensordot(r_matrix, mps_tensors[1], axes=(1, 1)).transpose(1, 0, 2)
    correct_env = update_left_environment(q_last, q_last, mpo_tensors[0], left_envs[0])
    assert np.allclose(correct_env, left_envs[1])
    assert np.allclose(correct_canon, canon_sites[1])
    # Site 2
    q_last, r_matrix = right_qr(correct_canon)
    correct_canon = np.tensordot(r_matrix, mps_tensors[2], axes=(1, 1)).transpose(1, 0, 2)
    correct_env = update_left_environment(q_last, q_last, mpo_tensors[1], left_envs[1])
    assert np.allclose(correct_env, left_envs[2])
    assert np.allclose(correct_canon, correct_canon)


def test_choose_stack_tensor_last_site() -> None:
    """Tests the choice of the stack tensor for the last site.

    In case of the last site, the stack tensor should be the MPS tensor, when
    the state was in left-canonical form.
    """
    num_sites = 3
    mps_tensors = [crandn(2, 3, 4) for _ in range(num_sites)]
    mps = MPS(num_sites, tensors=mps_tensors)
    canon_center_tensors = [crandn(2, 3, 4) for _ in range(num_sites)]
    # Found tensor
    found_tensor = choose_stack_tensor(num_sites - 1, canon_center_tensors, mps)
    assert np.allclose(mps_tensors[-1], found_tensor)


def test_choose_stack_tensor_middle_site() -> None:
    """Test the choice of the stack tensor for a middle site.

    For any site that is not the last, the tensor chosen should be the MPS
    tensor, when this site was the canonical center.
    """
    num_sites = 3
    mps_tensors = [crandn(2, 3, 4) for _ in range(num_sites)]
    mps = MPS(num_sites, tensors=mps_tensors)
    canon_center_tensors = [crandn(2, 3, 4) for _ in range(num_sites)]
    # Found tensor
    found_tensor = choose_stack_tensor(1, canon_center_tensors, mps)
    assert np.allclose(canon_center_tensors[1], found_tensor)


def test_find_new_q() -> None:
    """Tests finding the new q tensor.

    The new q should be 'left-canonical' and the left leg should be the
    addition of the input tensors.
    """
    old_tensor = crandn(2, 3, 5)
    new_tensor = crandn(2, 4, 5)
    q_tensor = find_new_q(old_tensor, new_tensor)
    # Test shape
    assert q_tensor.ndim == 3
    assert q_tensor.shape[0] == 2
    assert q_tensor.shape[2] == 5
    assert q_tensor.shape[1] == 7
    # Check that q_tensor is unitary
    iden = np.eye(q_tensor.shape[1])
    q_prod = np.tensordot(q_tensor, q_tensor.conj(), axes=([0, 2], [0, 2]))
    assert np.allclose(q_prod, iden)


def test_build_basis_change_tensor() -> None:
    """The basis change tensor construction.

    The basis change tensor should have the old basis as first leg and the new
    basis as its last leg.
    """
    old_q = crandn(2, 3, 4)
    new_q = crandn(2, 7, 5)
    old_m = crandn(4, 5)
    basis_change = build_basis_change_tensor(old_q, new_q, old_m)
    assert basis_change.ndim == 2
    assert basis_change.shape[0] == 3
    assert basis_change.shape[1] == 7
    # Reference
    ref_basis_change = np.tensordot(old_q, old_m, axes=(2, 0))
    ref_basis_change = np.tensordot(ref_basis_change, new_q.conj(), axes=([0, 2], [0, 2]))
    assert np.allclose(ref_basis_change, basis_change)


def test_local_update() -> None:
    """Test the local update.

    Tests that it correctly changes input lists and returns the
        updated environment blocks.

    """
    mps = random_mps([(2, 5, 4), (2, 4, 3), (2, 3, 5)])
    mps.set_canonical_form(0)
    ref_mps = deepcopy(mps)
    mpo = random_mpo([(2, 2, 1, 3), (2, 2, 3, 4), (2, 2, 4, 1)])
    canon_sites, left_envs = prepare_canonical_site_tensors(mps, mpo)
    ref_canon_sites = deepcopy(canon_sites)
    right_block = np.eye(5).reshape(5, 1, 5)
    site = 2
    right_m_block = np.eye(5)
    sim_params = PhysicsSimParams(observables=[], elapsed_time=1)
    # Perform the local update
    result = local_update(
        mps, mpo, left_envs, right_block, canon_sites, site, right_m_block, sim_params, numiter_lanczos=25
    )
    # General Change Check
    assert not mps.almost_equal(ref_mps)
    assert canon_sites[site - 1].shape != ref_canon_sites[site - 1].shape
    # Check for correct shapes
    # Last left leg dimension should be doubled
    assert mps.tensors[site].shape == (2, 6, 5)
    assert canon_sites[site - 1].shape == (2, 4, 6)
    # Check results
    assert len(result) == 2
    assert result[0].shape == (3, 6)
    assert result[1].shape == (6, 4, 6)


def test_bug_single_site() -> None:
    """Tests the BUG on a single site MPS against an exact time evolution."""
    mps = random_mps([(2, 1, 1)])
    ref_mps = deepcopy(mps)
    mpo = MPO()
    mpo.init_ising(1, 1, 0.5)
    ref_mpo = deepcopy(mpo)
    sim_params = PhysicsSimParams(observables=[], elapsed_time=1, threshold=1e-16, max_bond_dim=10)
    # Perform BUG
    bug(mps, mpo, sim_params, numiter_lanczos=25)
    # Check against exact evolution
    state_vec = ref_mps.to_vec()
    ham_matrix = ref_mpo.to_matrix()
    time_evo_op = expm(-1j * sim_params.dt * ham_matrix)
    new_state_vec = time_evo_op @ state_vec
    assert np.allclose(mps.to_vec(), new_state_vec)

def test_bug_three_sites() -> None:
    """Tests the BUG on a three site MPS against an exact time evolution."""
    mps = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 1)])
    ref_mps = deepcopy(mps)
    mpo = MPO()
    mpo.init_ising(3, 1, 0.5)
    ref_mpo = deepcopy(mpo)
    sim_params = PhysicsSimParams(observables=[], elapsed_time=1, threshold=1e-16, max_bond_dim=10)
    # Perform BUG
    bug(mps, mpo, sim_params, numiter_lanczos=25)
    # Check against exact evolution
    state_vec = ref_mps.to_vec()
    ham_matrix = ref_mpo.to_matrix()
    time_evo_op = expm(-1j * sim_params.dt * ham_matrix)
    new_state_vec = time_evo_op @ state_vec
    # Check the result
    assert [0] == mps.check_canonical_form()
    assert np.allclose(mps.to_vec(), new_state_vec)

def test_bug_ten_sites() -> None:
    """Tests a single BUG time step on ten sites."""
    num_sites = 10
    shapes = [(2,1,4)] + (num_sites-2)*[(2,4,4)] + [(2,4,1)]
    mps = random_mps(shapes)
    ref_mps = deepcopy(mps)
    mpo = MPO()
    mpo.init_ising(num_sites, 1, 0.5)
    ref_mpo = deepcopy(mpo)
    sim_params = PhysicsSimParams(observables=[], elapsed_time=1, threshold=0, max_bond_dim=10000, dt=0.001)
    # Perform BUG
    bug(mps, mpo, sim_params, numiter_lanczos=25)
    # Check against exact evolution
    state_vec = ref_mps.to_vec()
    ham_matrix = ref_mpo.to_matrix()
    time_evo_op = expm(-1j * sim_params.dt * ham_matrix)
    new_state_vec = time_evo_op @ state_vec
    # Check the result
    assert [0] == mps.check_canonical_form()
    print(max(np.abs(mps.to_vec() - new_state_vec)))
    assert np.allclose(mps.to_vec(), new_state_vec)
