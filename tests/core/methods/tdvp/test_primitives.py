# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for TDVP low-level primitives and dense effective operators."""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806, PLC2701

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.methods.decompositions import merge_two_site, split_two_site
from mqt.yaqs.core.methods.tdvp.primitives import (
    _build_dense_effective_operator,
    build_dense_heff_bond,
    build_dense_heff_site,
    initialize_right_environments,
    merge_mpo_tensors,
    project_bond,
    project_site,
    update_bond,
    update_left_environment,
    update_right_environment,
    update_site,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.methods.decompositions import TruncMode

rng = np.random.default_rng()


def test_split_two_site_invalid_shape() -> None:
    """``split_two_site`` raises when the first axis does not match ``physical_dimensions``."""
    A = rng.random(size=(3, 3, 5)).astype(np.complex128)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
    )
    physical_dimensions = [3, 3]
    with pytest.raises(
        ValueError, match=r"The first dimension of the tensor must be a combination of the given physical dimensions."
    ):
        split_two_site(
            A,
            physical_dimensions,
            svd_distribution="left",
            trunc_mode=cast("TruncMode", sim_params.trunc_mode),
            threshold=sim_params.svd_threshold,
            max_bond_dim=sim_params.max_bond_dim,
        )


def test_merge_two_site() -> None:
    """Test :func:`mqt.yaqs.core.methods.decompositions.merge_two_site`.

    This test creates two tensors A0 and A1 with shapes (2, 3, 4) and (5, 4, 7), respectively.
    After merging, the expected shape is (10, 3, 7) because the contraction is over the bond
    between the two site tensors.
    """
    A0 = rng.random(size=(2, 3, 4)).astype(np.complex128)
    A1 = rng.random(size=(5, 4, 7)).astype(np.complex128)
    merged = merge_two_site(A0, A1)
    assert merged.shape == (10, 3, 7)


def test_merge_mpo_tensors() -> None:
    """Test the merge_mpo_tensors function.

    This test creates two 4D arrays A0 and A1 with shapes (2, 3, 4, 5) and (7, 8, 5, 9), respectively.
    After merging via merge_mpo_tensors, the expected shape is (14, 24, 4, 9).
    """
    A0 = rng.random(size=(2, 3, 4, 5)).astype(np.complex128)
    A1 = rng.random(size=(7, 8, 5, 9)).astype(np.complex128)
    merged = merge_mpo_tensors(A0, A1)
    assert merged.shape == (14, 24, 4, 9)


def test_update_right_environment() -> None:
    """Test the update_right_environment function.

    This test creates dummy arrays A, B, W, and R with compatible shapes for the contraction
    operations defined in update_right_environment. It then verifies that the resulting tensor
    has the expected shape (3, 8, 9).
    """
    A = rng.random(size=(2, 3, 4)).astype(np.complex128)
    R = rng.random(size=(4, 5, 6)).astype(np.complex128)
    W = rng.random(size=(7, 2, 8, 5)).astype(np.complex128)
    B = rng.random(size=(7, 9, 6)).astype(np.complex128)
    Rnext = update_right_environment(A, B, W, R)
    assert Rnext.shape == (3, 8, 9)


def test_update_left_environment() -> None:
    """Test the update_left_environment function.

    This test constructs dummy arrays A, B, W, and L with compatible shapes for the contraction.
    It then verifies that the output is a 3D tensor.
    """
    A = rng.random(size=(3, 4, 10)).astype(np.complex128)
    B = rng.random(size=(7, 6, 8)).astype(np.complex128)
    L_arr = rng.random(size=(4, 5, 6)).astype(np.complex128)
    W = rng.random(size=(7, 3, 5, 9)).astype(np.complex128)
    Rnext = update_left_environment(A, B, W, L_arr)
    assert Rnext.ndim == 3


def test_project_site() -> None:
    """Test the project_site function.

    This test creates dummy tensors A, R, W, and L with appropriate shapes and checks that
    the output of project_site is a 3D tensor.
    """
    A = rng.random(size=(2, 3, 4)).astype(np.complex128)
    R = rng.random(size=(4, 5, 6)).astype(np.complex128)
    W = rng.random(size=(7, 2, 8, 5)).astype(np.complex128)
    L_arr = rng.random(size=(3, 8, 9)).astype(np.complex128)
    out = project_site(L_arr, R, W, A)
    assert out.ndim == 3


def test_project_bond() -> None:
    """Test the project_bond function.

    This test creates a bond tensor C and dummy tensors L and R with compatible shapes,
    and verifies that the output has the expected shape (6, 5).
    """
    C = rng.random(size=(2, 3)).astype(np.complex128)
    R = rng.random(size=(3, 4, 5)).astype(np.complex128)
    L_arr = rng.random(size=(2, 4, 6)).astype(np.complex128)
    out = project_bond(L_arr, R, C)
    assert out.shape == (6, 5)


def test_update_site() -> None:
    """Test the update_site function.

    This test creates a dummy MPS tensor A (shape (2,2,4)), along with tensors L, R, and W,
    and applies update_site with a small time step and a fixed number of Lanczos iterations.
    The output should have the same shape as the input tensor A.
    """
    A = rng.random(size=(2, 2, 4)).astype(np.complex128)
    R = rng.random(size=(4, 1, 4)).astype(np.complex128)
    W = rng.random(size=(2, 2, 1, 1)).astype(np.complex128)
    L_arr = rng.random(size=(2, 1, 2)).astype(np.complex128)
    dt = 0.05
    out = update_site(L_arr, R, W, A, dt, krylov_tol=1e-12)
    assert out.shape == A.shape, f"Expected shape {A.shape}, got {out.shape}"


def test_initialize_right_environments_rejects_length_mismatch() -> None:
    """Right-environment initialization requires matching MPS and MPO lengths."""
    psi = MPS(3, state="zeros")
    op = MPO.ising(4, 1.0, 0.5)
    with pytest.raises(ValueError, match="lengths"):
        initialize_right_environments(psi, op)


def test_update_site_matrix_free_above_dense_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large local tensors use the matrix-free Krylov path instead of dense H_eff."""
    monkeypatch.setattr("mqt.yaqs.core.methods.tdvp.primitives.DENSE_THRESHOLD", 8)
    ket = rng.random(size=(2, 4, 4)).astype(np.complex128)
    left_env = rng.random(size=(4, 2, 4)).astype(np.complex128)
    right_env = rng.random(size=(4, 2, 4)).astype(np.complex128)
    op = rng.random(size=(2, 2, 2, 2)).astype(np.complex128)
    out = update_site(left_env, right_env, op, ket, 0.05, krylov_tol=1e-10)
    assert out.shape == ket.shape


def test_update_bond() -> None:
    """Test the update_bond function.

    This test creates a square bond tensor C and compatible dummy tensors R and L.
    It applies update_bond and checks that the output shape matches that of C.
    """
    C = rng.random(size=(2, 2)).astype(np.complex128)
    R = rng.random(size=(2, 2, 2)).astype(np.complex128)
    L_arr = rng.random(size=(2, 2, 2)).astype(np.complex128)
    dt = 0.05
    out = update_bond(L_arr, R, C, dt, krylov_tol=1e-12)
    assert out.shape == C.shape, f"Expected shape {C.shape}, got {out.shape}"


def test_dense_vs_project_site() -> None:
    """Dense H_eff should match the action of project_site on a local tensor."""
    phys_dim, bond_left_dim, bond_right_dim = 2, 2, 3
    chi_a_left = chi_a_right = 2

    rng_local = np.random.default_rng(1234)

    ket = rng_local.normal(size=(phys_dim, bond_left_dim, bond_right_dim)) + 1j * rng_local.normal(
        size=(phys_dim, bond_left_dim, bond_right_dim)
    )
    left_env = rng_local.normal(size=(bond_left_dim, chi_a_left, bond_left_dim)) + 1j * rng_local.normal(
        size=(bond_left_dim, chi_a_left, bond_left_dim)
    )
    right_env = rng_local.normal(size=(bond_right_dim, chi_a_right, bond_right_dim)) + 1j * rng_local.normal(
        size=(bond_right_dim, chi_a_right, bond_right_dim)
    )
    op = rng_local.normal(size=(phys_dim, phys_dim, chi_a_left, chi_a_right)) + 1j * rng_local.normal(
        size=(phys_dim, phys_dim, chi_a_left, chi_a_right)
    )

    H_eff = _build_dense_effective_operator(
        project_site,
        (left_env, right_env, op),
        ket.shape,
    )

    y1 = project_site(left_env, right_env, op, ket).reshape(-1)
    y2 = H_eff @ ket.reshape(-1)

    np.testing.assert_allclose(y1, y2, atol=1e-12)


def test_dense_vs_project_bond() -> None:
    """Dense H_eff should match the action of project_bond on a bond tensor."""
    bond_left_dim, bond_right_dim = 3, 4
    chi_a = 2

    rng_local = np.random.default_rng(5678)

    C = rng_local.normal(size=(bond_left_dim, bond_right_dim)) + 1j * rng_local.normal(
        size=(bond_left_dim, bond_right_dim)
    )
    left_env = rng_local.normal(size=(bond_left_dim, chi_a, bond_left_dim)) + 1j * rng_local.normal(
        size=(bond_left_dim, chi_a, bond_left_dim)
    )
    right_env = rng_local.normal(size=(bond_right_dim, chi_a, bond_right_dim)) + 1j * rng_local.normal(
        size=(bond_right_dim, chi_a, bond_right_dim)
    )

    H_eff = _build_dense_effective_operator(
        project_bond,
        (left_env, right_env),
        C.shape,
    )

    y1 = project_bond(left_env, right_env, C).reshape(-1)
    y2 = H_eff @ C.reshape(-1)

    np.testing.assert_allclose(y1, y2, atol=1e-12)


def test_build_dense_heff_site_matches_project_site(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_dense_heff_site: vec(project_site(..., X)) == H_eff @ vec(X) for random small tensors."""
    monkeypatch.setattr(
        "mqt.yaqs.core.methods.tdvp.primitives.NUMBA_DENSE_HEFF_MIN_DIM",
        1,
    )
    phys_in, phys_out = 2, 2
    bond_left_dim, bond_right_dim = 3, 4
    chi_left, chi_right = 2, 3

    rng_local = np.random.default_rng(4321)

    ket = rng_local.normal(size=(phys_in, bond_left_dim, bond_right_dim)) + 1j * rng_local.normal(
        size=(phys_in, bond_left_dim, bond_right_dim)
    )
    ket = np.asarray(ket, dtype=np.complex128)

    left_env = rng_local.normal(size=(bond_left_dim, chi_left, bond_left_dim)) + 1j * rng_local.normal(
        size=(bond_left_dim, chi_left, bond_left_dim)
    )
    left_env = np.asarray(left_env, dtype=np.complex128)

    right_env = rng_local.normal(size=(bond_right_dim, chi_right, bond_right_dim)) + 1j * rng_local.normal(
        size=(bond_right_dim, chi_right, bond_right_dim)
    )
    right_env = np.asarray(right_env, dtype=np.complex128)

    op = rng_local.normal(size=(phys_out, phys_in, chi_left, chi_right)) + 1j * rng_local.normal(
        size=(phys_out, phys_in, chi_left, chi_right)
    )
    op = np.asarray(op, dtype=np.complex128)

    H_eff = build_dense_heff_site(left_env, right_env, op)

    y1 = project_site(left_env, right_env, op, ket).reshape(-1)
    y2 = H_eff @ ket.reshape(-1)

    np.testing.assert_allclose(y1, y2, atol=1e-12)


def test_build_dense_heff_bond_matches_project_bond() -> None:
    """build_dense_heff_bond: vec(project_bond(..., C)) == H_eff @ vec(C) for random small tensors."""
    bond_left_dim, bond_right_dim = 3, 4
    chi = 2

    rng_local = np.random.default_rng(8765)

    C = rng_local.normal(size=(bond_left_dim, bond_right_dim)) + 1j * rng_local.normal(
        size=(bond_left_dim, bond_right_dim)
    )
    C = np.asarray(C, dtype=np.complex128)

    left_env = rng_local.normal(size=(bond_left_dim, chi, bond_left_dim)) + 1j * rng_local.normal(
        size=(bond_left_dim, chi, bond_left_dim)
    )
    left_env = np.asarray(left_env, dtype=np.complex128)

    right_env = rng_local.normal(size=(bond_right_dim, chi, bond_right_dim)) + 1j * rng_local.normal(
        size=(bond_right_dim, chi, bond_right_dim)
    )
    right_env = np.asarray(right_env, dtype=np.complex128)

    H_eff = build_dense_heff_bond(left_env, right_env)

    y1 = project_bond(left_env, right_env, C).reshape(-1)
    y2 = H_eff @ C.reshape(-1)

    np.testing.assert_allclose(y1, y2, atol=1e-12)


def test_build_dense_effective_operator_uses_generic_fallback() -> None:
    """_build_dense_effective_operator: uses generic fallback for unknown projector (basis-loop path)."""
    rng_local = np.random.default_rng(2025)

    tensor_shape = (2, 3)
    n_loc = int(np.prod(tensor_shape))

    A = rng_local.normal(size=(n_loc, n_loc)) + 1j * rng_local.normal(size=(n_loc, n_loc))
    A = np.asarray(A, dtype=np.complex128)

    calls = {"n": 0}

    def custom_projector(mat: NDArray[np.complex128], x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Apply a dense linear map and count basis-loop projector calls.

        Returns:
            Projected tensor with the input ``tensor_shape``.

        """
        calls["n"] += 1
        y = mat @ x.reshape(-1)
        return y.reshape(tensor_shape)

    H_eff = _build_dense_effective_operator(
        projector=custom_projector,
        proj_args=(A,),
        tensor_shape=tensor_shape,
    )

    assert calls["n"] == n_loc, f"Expected {n_loc} projector calls (basis loop), got {calls['n']}"
    assert H_eff.shape == (n_loc, n_loc)


def test_build_dense_effective_operator_generic_fallback_correctness() -> None:
    """_build_dense_effective_operator: generic fallback reconstructs the operator exactly."""
    rng_local = np.random.default_rng(2026)

    tensor_shape = (2, 2, 2)
    n_loc = int(np.prod(tensor_shape))

    A = rng_local.normal(size=(n_loc, n_loc)) + 1j * rng_local.normal(size=(n_loc, n_loc))
    A = np.asarray(A, dtype=np.complex128)

    def custom_projector(mat: NDArray[np.complex128], x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Apply a dense linear map for generic fallback correctness checks.

        Returns:
            Projected tensor with the input ``tensor_shape``.

        """
        y = mat @ x.reshape(-1)
        return y.reshape(tensor_shape)

    H_eff = _build_dense_effective_operator(
        projector=custom_projector,
        proj_args=(A,),
        tensor_shape=tensor_shape,
    )

    x = rng_local.normal(size=n_loc) + 1j * rng_local.normal(size=n_loc)
    x = np.asarray(x, dtype=np.complex128)

    y1 = custom_projector(A, x.reshape(tensor_shape)).reshape(-1)
    y2 = H_eff @ x

    np.testing.assert_allclose(y1, y2, atol=1e-12)
