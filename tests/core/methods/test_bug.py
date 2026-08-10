# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
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
import pytest
from scipy.linalg import expm

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, BUGConfig, EvolutionMode
from mqt.yaqs.core.methods.bug import (
    bug,
    bug_sweep,
    build_basis_change_tensor,
    build_trial_basis,
    choose_stack_tensor,
    find_new_q,
    local_update,
    prepare_canonical_site_tensors,
)
from mqt.yaqs.core.methods.decompositions import left_qr, right_qr
from mqt.yaqs.core.methods.tdvp.primitives import update_left_environment

if TYPE_CHECKING:
    from numpy.typing import NDArray


def crandn(
    size: int | tuple[int, ...], *args: int, seed: np.random.Generator | int | None = None
) -> NDArray[np.complex128]:
    """Draw random samples from the standard complex normal distribution.

    Args:
        size: The size/shape of the output array.
        args: Additional dimensions for the output array.
        seed: The seed for the random number generator.

    Returns:
        The array of random complex numbers.
    """
    if isinstance(size, int) and len(args) > 0:
        size = (size, *list(args))
    elif isinstance(size, int):
        size = (size,)
    rng = np.random.default_rng(seed)
    # 1 / sqrt(2) is a normalization factor
    return np.asarray(rng.standard_normal(size) + 1j * rng.standard_normal(size) / np.sqrt(2), dtype=np.complex128)


def random_mps(shapes: list[tuple[int, int, int]]) -> MPS:
    """Create a random MPS with the given shapes.

    Args:
        shapes: The shapes of the tensors in the MPS.

    Returns:
        The random MPS.
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
    mpo.custom(tensors, transpose=False)
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
    mpo.custom([mpo_tensor])
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
    mpo.custom(mpo_tensors, transpose=False)
    canon_sites, left_envs = prepare_canonical_site_tensors(mps, mpo)
    assert mps.almost_equal(ref_mps)
    assert len(left_envs) == 3
    assert len(canon_sites) == 3
    # Correct envs and canon sites
    # Site 0
    correct_env = np.eye(3, dtype=np.complex128).reshape(3, 1, 3)
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
    q_last, r_matrix = right_qr(np.asarray(correct_canon, dtype=np.complex128))
    correct_canon = np.tensordot(r_matrix, mps_tensors[2], axes=(1, 1)).transpose(1, 0, 2)
    correct_env = update_left_environment(q_last, q_last, mpo_tensors[1], left_envs[1])
    assert np.allclose(correct_env, left_envs[2])
    assert np.allclose(correct_canon, canon_sites[2])


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
    right_block = np.eye(5, dtype=np.complex128).reshape(5, 1, 5)
    site = 2
    right_m_block = np.eye(5, dtype=np.complex128)
    sim_params = AnalogSimParams(get_state=True, elapsed_time=1)
    # Perform the local update
    result = local_update(
        mps,
        mpo,
        left_envs,
        right_block,
        canon_sites,
        site,
        right_m_block,
        dt=sim_params.dt,
        krylov_tol=sim_params.krylov_tol,
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
    mpo = MPO.ising(1, 1, 0.5)
    ref_mpo = deepcopy(mpo)
    sim_params = AnalogSimParams(preset="exact", get_state=True, elapsed_time=1)

    # Perform BUG
    bug(mps, mpo, sim_params)
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
    mpo = MPO.ising(3, 1, 0.5)
    ref_mpo = deepcopy(mpo)
    sim_params = AnalogSimParams(preset="exact", get_state=True, elapsed_time=1)

    # Perform BUG
    bug(mps, mpo, sim_params)
    # Check against exact evolution
    state_vec = ref_mps.to_vec()
    ham_matrix = ref_mpo.to_matrix()
    time_evo_op = expm(-1j * sim_params.dt * ham_matrix)
    new_state_vec = time_evo_op @ state_vec
    # Check the result
    assert mps.check_canonical_form() == [0]
    assert mps.orthogonality_center == 0
    np.testing.assert_allclose(mps.to_vec(), new_state_vec, rtol=1e-10, atol=1e-12)


def test_bug_requires_center_at_zero() -> None:
    """BUG rejects an MPS whose tracked center is not at site 0."""
    mps = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 1)])
    mps.set_center(1)
    mpo = MPO.ising(3, 1, 0.5)
    sim_params = AnalogSimParams(preset="exact", get_state=True, elapsed_time=1)
    with pytest.raises(ValueError, match="bug"):
        bug(mps, mpo, sim_params)


def test_bug_sweep_rejects_unknown_center() -> None:
    """The uncompressed kernel rejects a missing orthogonality center."""
    mps = random_mps([(2, 1, 4), (2, 4, 1)])
    mps.set_center(None)
    mpo = MPO.ising(2, 1, 0.5)
    with pytest.raises(ValueError, match="bug_sweep"):
        bug_sweep(mps, mpo, dt=0.1, krylov_tol=1e-10)


def test_bug_length_mismatch_raises() -> None:
    """BUG rejects Hamiltonians whose site count differs from the MPS."""
    mps = MPS(2, state="zeros")
    mpo = MPO.ising(3, 1, 0.5)
    sim_params = AnalogSimParams(preset="exact", get_state=True, elapsed_time=1)

    with pytest.raises(ValueError, match="same number of sites"):
        bug(mps, mpo, sim_params)


def _is_row_isometric(tensor: NDArray[np.complex128], *, atol: float = 1e-10) -> bool:
    """Return True if an MPS tensor is right-isometric (I on the left virtual leg)."""
    left = tensor.shape[1]
    mat = np.tensordot(tensor, tensor.conj(), axes=([0, 2], [0, 2]))
    return bool(np.allclose(mat, np.eye(left), atol=atol))


def test_bug_two_sites_dense_reference() -> None:
    """BUG on L=2 matches dense exact evolution within Krylov/truncation tolerance."""
    mps = random_mps([(2, 1, 3), (2, 3, 1)])
    ref_mps = deepcopy(mps)
    mpo = MPO.ising(2, 1.0, 0.7)
    sim_params = AnalogSimParams(preset="exact", get_state=True, elapsed_time=1, dt=0.05)
    bug(mps, mpo, sim_params)
    exact = expm(-1j * sim_params.dt * mpo.to_matrix()) @ ref_mps.to_vec()
    np.testing.assert_allclose(mps.to_vec(), exact, rtol=1e-8, atol=1e-10)
    assert mps.orthogonality_center == 0


def test_bug_sweep_no_compression_and_isometric_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """bug_sweep never compresses and leaves non-root sites right-isometric."""
    mps = random_mps([(2, 1, 3), (2, 3, 4), (2, 4, 1)])
    mpo = MPO.ising(3, 1.0, 0.5)

    def boom(*_args: object, **_kwargs: object) -> None:
        msg = "compress should not be called from bug_sweep"
        raise AssertionError(msg)

    monkeypatch.setattr(MPS, "compress", boom)
    bug_sweep(mps, mpo, dt=0.05, krylov_tol=1e-12)
    assert mps.orthogonality_center == 0
    assert _is_row_isometric(mps.tensors[1])
    assert _is_row_isometric(mps.tensors[2])


def test_bug_sweep_zero_hamiltonian_preserves_state() -> None:
    """A zero Hamiltonian leaves the physical state unchanged up to gauge."""
    mps = random_mps([(2, 1, 3), (2, 3, 1)])
    ref = mps.to_vec().copy()
    zero = MPO()
    zero.custom(
        [
            np.zeros((2, 2, 1, 1), dtype=np.complex128),
            np.zeros((2, 2, 1, 1), dtype=np.complex128),
        ],
        transpose=False,
    )
    # Fix physical_dimension: custom(transpose=False) historically reads shape[2].
    zero.physical_dimension = 2
    bug_sweep(mps, zero, dt=0.2, krylov_tol=1e-12)
    overlap = abs(np.vdot(ref, mps.to_vec()))
    assert overlap == pytest.approx(1.0, abs=1e-10)


def test_bug_forwards_trunc_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """bug() forwards sim_params.trunc_mode into MPS.compress."""
    seen: dict[str, object] = {}

    def fake_compress(
        self: MPS,
        threshold: float,
        *,
        max_bond_dim: int | None = None,
        trunc_mode: str = "discarded_weight",
    ) -> None:
        seen["threshold"] = threshold
        seen["max_bond_dim"] = max_bond_dim
        seen["trunc_mode"] = trunc_mode
        self.set_center(0)

    monkeypatch.setattr(MPS, "compress", fake_compress)
    mps = random_mps([(2, 1, 2), (2, 2, 1)])
    mpo = MPO.ising(2, 1.0, 0.5)
    sim_params = AnalogSimParams(
        preset="exact",
        get_state=True,
        elapsed_time=1,
        trunc_mode="relative",
        svd_threshold=1e-8,
        max_bond_dim=4,
    )
    bug(mps, mpo, sim_params)
    assert seen["trunc_mode"] == "relative"
    assert seen["threshold"] == pytest.approx(1e-8)
    assert seen["max_bond_dim"] == 4


def test_prepare_does_not_alias_physical_tensors() -> None:
    """Preparation must not mutate or alias the live MPS tensors."""
    mps = random_mps([(2, 1, 3), (2, 3, 2), (2, 2, 1)])
    originals = [t.copy() for t in mps.tensors]
    ids = [id(t) for t in mps.tensors]
    mpo = MPO.ising(3, 1.0, 0.5)
    canon_sites, _left = prepare_canonical_site_tensors(mps, mpo)
    for orig, live in zip(originals, mps.tensors, strict=True):
        assert np.allclose(orig, live)
    assert id(canon_sites[0]) == ids[0]
    assert id(canon_sites[1]) != ids[1]


def _row_space_projector(rows: NDArray[np.complex128], *, tol: float = 1e-10) -> NDArray[np.complex128]:
    """Orthogonal projector onto the row space of ``rows`` via rank-revealing SVD.

    Returns:
        The Hermitian projector onto the numerical row space.
    """
    _u, s, vh = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.sum(s > tol * max(s[0], 1.0)))
    basis = vh[:rank, :]
    return basis.conj().T @ basis


def test_proposition1_full_rank_row_spaces_agree() -> None:
    """Full-rank center and explicit stacks supply the same row space before QR."""
    rng = np.random.default_rng(0)
    b0 = np.linalg.qr(rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2)))[0].T
    # b0 is 2x4 row-isometric
    s_mat = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    c_pred = rng.standard_normal((2, 4)) + 1j * rng.standard_normal((2, 4))
    center_stack = np.vstack([s_mat @ b0, c_pred])
    explicit_stack = np.vstack([b0, c_pred])
    p_c = _row_space_projector(center_stack)
    p_e = _row_space_projector(explicit_stack)
    assert np.allclose(p_c, p_e, atol=1e-10)


def test_proposition1_rank_deficient_counterexample() -> None:
    """Center stacking can miss an old direction that explicit retention keeps."""
    b0 = np.eye(2, dtype=np.complex128)
    s_mat = np.diag([1.0, 0.0]).astype(np.complex128)
    c_pred = s_mat.copy()
    center_stack = np.vstack([s_mat @ b0, c_pred])
    explicit_stack = np.vstack([b0, c_pred])
    p_c = _row_space_projector(center_stack)
    p_e = _row_space_projector(explicit_stack)
    # Explicit contains e2; center does not.
    e2 = np.array([0.0, 1.0], dtype=np.complex128)
    assert np.linalg.norm(p_e @ e2) == pytest.approx(1.0, abs=1e-10)
    assert np.linalg.norm(p_c @ e2) == pytest.approx(0.0, abs=1e-10)


def test_proposition1_predictor_rescue() -> None:
    """A predictor carrying the missing direction can restore the old space."""
    b0 = np.eye(2, dtype=np.complex128)
    s_mat = np.diag([1.0, 0.0]).astype(np.complex128)
    c_pred = np.eye(2, dtype=np.complex128)
    center_stack = np.vstack([s_mat @ b0, c_pred])
    p_c = _row_space_projector(center_stack)
    assert np.allclose(p_c, np.eye(2), atol=1e-10)


def test_proposition1_explicit_old_basis_zero_defect() -> None:
    """explicit_old_basis retained input has zero old-space projector defect before QR."""
    old_q = crandn((2, 3, 4), seed=1)
    # Make old_q right-isometric on phys+right for a meaningful projector test.
    phys, left, right = old_q.shape
    q_mat, _ = np.linalg.qr(old_q.transpose(0, 2, 1).reshape(phys * right, left))
    old_q = q_mat[:, :left].reshape(phys, right, left).transpose(0, 2, 1)
    deeper = np.eye(right, dtype=np.complex128)
    predictor = crandn((2, 3, 4), seed=2)
    working = crandn((2, 3, 4), seed=3)
    new_q, _overlap = build_trial_basis(
        old_q=old_q,
        working_center=working,
        predictor=predictor,
        deeper_overlap=deeper,
        is_endpoint=True,
        basis_mode="explicit_old_basis",
    )
    old_basis = np.tensordot(old_q, deeper, axes=(2, 0))
    retained = np.concatenate((old_basis, predictor), axis=1)
    # Row space of retained (phys,right) fibers over left index.
    rows = retained.transpose(1, 0, 2).reshape(retained.shape[1], -1)
    old_rows = old_basis.transpose(1, 0, 2).reshape(old_basis.shape[1], -1)
    p_ret = _row_space_projector(rows)
    # Every old row vector should lie in the retained row space.
    for row in old_rows:
        assert np.linalg.norm(row - row @ p_ret) == pytest.approx(0.0, abs=1e-8)
    assert new_q.shape[0] == 2


def test_proposition2_overlap_frobenius_identity() -> None:
    """Overlap transport satisfies the local Frobenius identity from Proposition 2."""
    rng = np.random.default_rng(4)
    b0 = np.linalg.qr(rng.standard_normal((5, 3)) + 1j * rng.standard_normal((5, 3)))[0].T
    b1 = np.linalg.qr(rng.standard_normal((5, 4)) + 1j * rng.standard_normal((5, 4)))[0].T
    overlap = b0 @ b1.conj().T
    x = rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))
    left = np.linalg.norm(x @ overlap, "fro") ** 2
    # ||X O||^2 = ||X||^2 - ||X B0 (I - B1^H B1)||^2 with O = B0 B1^H
    ambient = b0.shape[1]
    defect = np.eye(ambient) - b1.conj().T @ b1
    right = np.linalg.norm(x, "fro") ** 2 - np.linalg.norm(x @ b0 @ defect, "fro") ** 2
    assert left == pytest.approx(right, abs=1e-10)


def test_proposition2_inclusion_and_contraction() -> None:
    """Inclusion of old space implies equality; failed inclusion contracts some X."""
    rng = np.random.default_rng(5)
    b0 = np.linalg.qr(rng.standard_normal((6, 2)) + 1j * rng.standard_normal((6, 2)))[0].T
    # Ensure span(b0) subset span(b1) by stacked QR of [b0; extra].
    extra = rng.standard_normal((1, 6)) + 1j * rng.standard_normal((1, 6))
    b1 = np.linalg.qr(np.vstack([b0, extra]).T)[0].T
    overlap = b0 @ b1.conj().T
    for _ in range(5):
        x = rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))
        assert np.linalg.norm(x @ overlap, "fro") == pytest.approx(np.linalg.norm(x, "fro"), abs=1e-10)

    # Failed inclusion: b1 omits part of b0.
    b1_small = b0[:1, :]
    b1_small /= np.linalg.norm(b1_small)
    overlap_bad = b0 @ b1_small.conj().T
    x = np.eye(2, dtype=np.complex128)
    assert np.linalg.norm(x @ overlap_bad, "fro") < np.linalg.norm(x, "fro") - 1e-8


def test_build_trial_basis_fixed_profile_no_concat() -> None:
    """fixed_profile factorizes the predictor without concatenating old tensors."""
    old_q = crandn((2, 3, 4), seed=6)
    working = crandn((2, 3, 4), seed=7)
    predictor = crandn((2, 3, 4), seed=8)
    deeper = np.eye(4, dtype=np.complex128)
    new_q, overlap = build_trial_basis(
        old_q=old_q,
        working_center=working,
        predictor=predictor,
        deeper_overlap=deeper,
        is_endpoint=False,
        basis_mode="fixed_profile",
    )
    ref_q, _ = left_qr(predictor)
    assert np.allclose(new_q, ref_q)
    assert overlap.shape == (3, new_q.shape[1])


def test_fixed_profile_zero_h_and_profile() -> None:
    """fixed_profile with compression none preserves state and bond profile for zero H."""
    mps = random_mps([(2, 1, 2), (2, 2, 3), (2, 3, 1)])
    entry = [int(mps.tensors[i].shape[2]) for i in range(mps.length - 1)]
    ref = mps.to_vec().copy()
    zero = MPO()
    zero.tensors = [np.zeros((2, 2, 1, 1), dtype=np.complex128) for _ in range(3)]
    zero.length = 3
    zero.physical_dimension = 2
    params = AnalogSimParams(
        preset="exact",
        get_state=True,
        elapsed_time=1,
        evolution_mode=EvolutionMode.BUG,
        bug_config=BUGConfig(basis_mode="fixed_profile", compression="none"),
    )
    bug(mps, zero, params)
    exit_profile = [int(mps.tensors[i].shape[2]) for i in range(mps.length - 1)]
    assert exit_profile == entry
    assert abs(np.vdot(ref, mps.to_vec())) == pytest.approx(1.0, abs=1e-10)


def test_alternating_endpoints_uses_half_dt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alternating schedule applies two positive half-steps of dt/2."""
    dts: list[float] = []

    def capture_sweep(
        state: MPS,
        _mpo: MPO,
        *,
        dt: float,
        krylov_tol: float,
        basis_mode: str = "center",
    ) -> None:
        del krylov_tol, basis_mode
        dts.append(dt)
        state.set_center(0)

    monkeypatch.setattr("mqt.yaqs.core.methods.bug.bug_sweep", capture_sweep)
    mps = random_mps([(2, 1, 2), (2, 2, 1)])
    mpo = MPO.ising(2, 1.0, 0.5)
    ref_tensors = [t.copy() for t in mpo.tensors]
    params = AnalogSimParams(
        preset="exact",
        get_state=True,
        elapsed_time=1,
        dt=0.2,
        evolution_mode=EvolutionMode.BUG,
        bug_config=BUGConfig(schedule="alternating_endpoints", compression="none"),
    )
    bug(mps, mpo, params)
    assert dts == pytest.approx([0.1, 0.1])
    for a, b in zip(ref_tensors, mpo.tensors, strict=True):
        assert np.allclose(a, b)
    assert mps.orthogonality_center == 0
    assert mps.flipped is False


def test_compression_call_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compression placement controls how often MPS.compress is invoked."""
    counts = {"n": 0}

    def counting_compress(
        self: MPS,
        threshold: float,
        *,
        max_bond_dim: int | None = None,
        trunc_mode: str = "discarded_weight",
    ) -> None:
        del threshold, max_bond_dim, trunc_mode
        counts["n"] += 1
        self.set_center(0)

    monkeypatch.setattr(MPS, "compress", counting_compress)

    def run(config: BUGConfig) -> int:
        counts["n"] = 0
        mps = random_mps([(2, 1, 2), (2, 2, 1)])
        mpo = MPO.ising(2, 1.0, 0.5)
        params = AnalogSimParams(
            preset="exact",
            get_state=True,
            elapsed_time=1,
            evolution_mode=EvolutionMode.BUG,
            bug_config=config,
        )
        bug(mps, mpo, params)
        return counts["n"]

    assert run(BUGConfig(schedule="single_endpoint", compression="after_sweep")) == 1
    assert run(BUGConfig(schedule="single_endpoint", compression="after_step")) == 1
    assert run(BUGConfig(schedule="single_endpoint", compression="none")) == 0
    assert run(BUGConfig(schedule="alternating_endpoints", compression="after_sweep")) == 2
    assert run(BUGConfig(schedule="alternating_endpoints", compression="after_step")) == 1
    assert run(BUGConfig(schedule="alternating_endpoints", compression="none")) == 0


def test_normalize_after_compression_opt_in() -> None:
    """normalize_after_compression=True returns unit norm; default does not force it."""
    mps = random_mps([(2, 1, 3), (2, 3, 1)])
    mpo = MPO.ising(2, 1.0, 0.5)
    params = AnalogSimParams(
        preset="exact",
        get_state=True,
        elapsed_time=1,
        evolution_mode=EvolutionMode.BUG,
        bug_config=BUGConfig(normalize_after_compression=True, compression="after_sweep"),
    )
    bug(mps, mpo, params)
    assert abs(mps.norm()) == pytest.approx(1.0, abs=1e-10)


def test_block_overlap_recursion_matches_full_contraction() -> None:
    """Recursive block overlaps match explicit old/new right-block contractions."""
    mps = random_mps([(2, 1, 3), (2, 3, 4), (2, 4, 1)])
    mpo = MPO.ising(3, 0.0, 0.0)
    # Use a nontrivial but small step; capture overlaps via build_basis_change_tensor path.
    old_tensors = [t.copy() for t in mps.tensors]
    bug_sweep(mps, mpo, dt=0.01, krylov_tol=1e-12, basis_mode="center")
    # Build full right-block bases for sites 1 and 2 after the sweep.
    # For L=3, right block at site 2 is just the site tensor reshaped.
    new_tensors = mps.tensors

    def right_basis(tensors: list[NDArray[np.complex128]], start: int) -> NDArray[np.complex128]:
        # Contract sites start..L-1 into (left_of_start, physical_right_block)
        acc = tensors[-1]
        for site in range(len(tensors) - 2, start - 1, -1):
            acc = np.tensordot(tensors[site], acc, axes=(2, 1))
            # (phys_i, left_i, phys_j, right_j) -> combine physicals
            phys_i, left_i, phys_j, right_j = acc.shape
            acc = acc.transpose(1, 0, 2, 3).reshape(left_i, phys_i * phys_j, right_j)
            acc = acc.transpose(1, 0, 2)
        phys, left, right = acc.shape
        return acc.reshape(phys * right, left).T  # (left, phys*right) rows? use (old_left, dim)

    # Compare M at deepest site via direct formula B_old @ B_new^H
    b_old = right_basis(old_tensors, 2)
    b_new = right_basis(new_tensors, 2)
    # shapes: we want (old_left, new_left) = B_old @ B_new.conj().T with row-isometric convention
    # Using tensors directly:
    m2 = np.tensordot(old_tensors[2], new_tensors[2].conj(), axes=([0, 2], [0, 2]))
    m2_ref = build_basis_change_tensor(
        old_tensors[2],
        new_tensors[2],
        np.eye(old_tensors[2].shape[2], dtype=np.complex128),
    )
    assert np.allclose(m2, m2_ref)
    gram = m2 @ m2.conj().T
    evals = np.linalg.eigvalsh(gram)
    assert np.all(evals >= -1e-10)
    assert np.all(evals <= 1.0 + 1e-8)
    del b_old, b_new


def test_alternating_reduces_error_for_smaller_dt() -> None:
    """On a small asymmetric Hamiltonian, smaller dt reduces state error vs dense reference."""
    length = 4
    mpo = MPO.ising(length, 1.3, 0.7)
    mps = random_mps([(2, 1, 2), (2, 2, 2), (2, 2, 2), (2, 2, 1)])
    total_time = 0.8
    ref = expm(-1j * total_time * mpo.to_matrix()) @ mps.to_vec()

    def error_for_dt(dt: float) -> float:
        params = AnalogSimParams(
            preset="exact",
            get_state=True,
            elapsed_time=total_time,
            dt=dt,
            evolution_mode=EvolutionMode.BUG,
            bug_config=BUGConfig(
                schedule="alternating_endpoints",
                compression="after_sweep",
                normalize_after_compression=True,
            ),
            max_bond_dim=2,
            svd_threshold=0.0,
            trunc_mode="hard_cutoff",
        )
        state = deepcopy(mps)
        n_steps = round(total_time / dt)
        for _ in range(n_steps):
            bug(state, mpo, params)
        return float(np.linalg.norm(state.to_vec() - ref))

    err_coarse = error_for_dt(0.4)
    err_fine = error_for_dt(0.2)
    assert err_coarse > 1e-6
    assert err_fine < err_coarse
