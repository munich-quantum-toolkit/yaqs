# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for decompositions.

This module covers the left and right QR decompositions, ``split_two_site``
behavior (full-rank reconstruction, ``max_bond_dim`` truncation, error
handling), and ``linalg.svd`` edge-cases (reduced/full shapes, unitarity,
reconstruction, and the gesdd-to-gesvd fallback path).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pytest
import scipy.linalg

from mqt.yaqs.core import linalg
from mqt.yaqs.core.methods.decompositions import (
    left_qr,
    merge_two_site,
    right_qr,
    split_two_site,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.methods.decompositions import (
        SvdDistribution,
    )


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
    # 1 / sqrt(2) is a normalization factor
    return np.asarray((rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2), dtype=np.complex128)


def test_right_qr() -> None:
    """Tests the right qr decomposition.

    Ensures that it produces tensors of the correct shape and a unitary tensor.
    Also checks that the decomposition is actually the original tensor.
    """
    shape = (2, 3, 4)
    tensor = crandn(shape)
    q_tensor, r_matrix = right_qr(tensor)
    assert q_tensor.ndim == 3
    assert r_matrix.ndim == 2
    assert q_tensor.shape[0] == shape[0]
    assert q_tensor.shape[1] == shape[1]
    assert r_matrix.shape[1] == shape[2]
    assert q_tensor.shape[2] == r_matrix.shape[0]
    # Check that q_tensor is unitary
    iden = np.eye(q_tensor.shape[2])
    q_matrix = q_tensor.reshape(q_tensor.shape[0] * q_tensor.shape[1], -1)
    assert np.allclose(q_matrix.conj().T @ q_matrix, iden)
    # Check that qr = tensor
    contr = np.tensordot(q_tensor, r_matrix, axes=(2, 0))
    assert np.allclose(contr, tensor)


def test_left_qr() -> None:
    """Tests the left qr decomposition.

    Ensures that it produces tensors of the correct shape and a unitary tensor.
    Also checks that the decomposition is actually the original tensor.
    """
    shape = (2, 3, 4)
    tensor = crandn(shape)
    q_tensor, r_matrix = left_qr(tensor)
    assert q_tensor.ndim == 3
    assert r_matrix.ndim == 2
    assert q_tensor.shape[0] == shape[0]
    assert q_tensor.shape[2] == shape[2]
    assert r_matrix.shape[0] == shape[1]
    assert q_tensor.shape[1] == r_matrix.shape[1]
    # Check that q_tensor is unitary
    iden = np.eye(q_tensor.shape[1])
    q_matrix = q_tensor.transpose(0, 2, 1)
    q_matrix = q_matrix.reshape(-1, q_tensor.shape[1])
    assert np.allclose(q_matrix.T.conj() @ q_matrix, iden)
    # Check that qr = tensor
    contr = np.tensordot(q_tensor, r_matrix, axes=(1, 1))
    contr = contr.transpose(0, 2, 1)
    assert np.allclose(contr, tensor)


@pytest.mark.parametrize("distr", ["left", "right", "sqrt"])
def test_split_two_site_reconstructs_full_rank(distr: str) -> None:
    """``split_two_site`` with no truncation reconstructs the merged tensor for each distribution."""
    a = crandn(2, 3, 4)
    b = crandn(2, 4, 5)
    merged = merge_two_site(a, b)
    a_new, b_new = split_two_site(
        merged,
        [a.shape[0], b.shape[0]],
        svd_distribution=cast("SvdDistribution", distr),
        trunc_mode="discarded_weight",
        threshold=0.0,
        max_bond_dim=None,
    )
    merged_back = merge_two_site(a_new, b_new)
    np.testing.assert_allclose(merged_back, merged, atol=1e-10, rtol=1e-8)


def test_split_two_site_truncates_to_max_bond_dim() -> None:
    """``max_bond_dim`` caps the bond dimension returned by ``split_two_site``."""
    svs = np.array([1.0, 0.9, 0.8, 0.7], dtype=np.float64)
    d0, d1, d_left, d_right = 2, 2, 3, 4
    theta = _theta_from_singulars(svs, d0 * d_left, d1 * d_right, seed=21)
    merged = _as_merged_two_site(theta, d0, d1, d_left, d_right)
    k = 2
    a_new, b_new = split_two_site(
        merged,
        [d0, d1],
        svd_distribution="sqrt",
        trunc_mode="discarded_weight",
        threshold=0.0,
        max_bond_dim=k,
    )
    assert a_new.shape[2] == k
    assert b_new.shape[1] == k


def test_split_two_site_min_keep() -> None:
    """``min_keep`` enforces a truncation floor even when threshold would drop further."""
    svs = np.array([1.0, 1e-12, 1e-13, 1e-14], dtype=np.float64)
    d0, d1, d_left, d_right = 2, 2, 2, 2
    theta = _theta_from_singulars(svs, d0 * d_left, d1 * d_right, seed=31)
    merged = _as_merged_two_site(theta, d0, d1, d_left, d_right)
    a_new, b_new = split_two_site(
        merged,
        [d0, d1],
        svd_distribution="sqrt",
        trunc_mode="relative",
        threshold=1e-6,
        max_bond_dim=None,
        min_keep=2,
    )
    assert a_new.shape[2] == 2
    assert b_new.shape[1] == 2


def test_split_two_site_unknown_mode_raises() -> None:
    """Invalid ``trunc_mode`` raises ``ValueError``."""
    merged = crandn(4, 3, 5)
    with pytest.raises(ValueError, match="Unknown truncation mode"):
        split_two_site(
            merged,
            [2, 2],
            svd_distribution="right",
            trunc_mode=cast("Any", "invalid"),
            threshold=1e-9,
            max_bond_dim=None,
        )


def _rand_unitary_like(m: int, n: int, *, seed: int) -> NDArray[np.complex128]:
    """Build a random complex matrix with orthonormal columns via QR.

    Args:
        m: Number of rows.
        n: Number of columns (``n <= m``); the first ``n`` columns of the
            Q factor are returned.
        seed: Seed for the random number generator.

    Returns:
        Complex matrix of shape ``(m, n)`` whose columns are orthonormal.
    """
    rng_local = np.random.default_rng(seed)
    mat = rng_local.normal(size=(m, n)) + 1j * rng_local.normal(size=(m, n))
    q, _ = np.linalg.qr(mat)
    q = np.asarray(q, dtype=np.complex128)
    return cast("NDArray[np.complex128]", q[:, :n])


def _theta_from_singulars(s: NDArray[np.float64], m: int, n: int, *, seed: int) -> NDArray[np.complex128]:
    """Construct a complex matrix with a prescribed singular spectrum.

    Builds ``theta = U @ diag(s) @ V^H`` where ``U`` and ``V`` are random
    orthonormal factors, so ``theta``'s singular values match ``s`` (truncated
    to ``min(len(s), m, n)`` if needed).

    Args:
        s: Target singular values (non-increasing, non-negative).
        m: Number of rows of the output matrix.
        n: Number of columns of the output matrix.
        seed: Seed for the random number generator; ``seed + 1`` is used for
            the right factor so ``U`` and ``V`` are independent.

    Returns:
        Complex matrix of shape ``(m, n)`` with singular values ``s``.
    """
    r = min(len(s), m, n)
    u = _rand_unitary_like(m, r, seed=seed)
    v = _rand_unitary_like(n, r, seed=seed + 1)
    sigma = np.diag(s[:r].astype(np.complex128))
    return cast("NDArray[np.complex128]", (u @ sigma @ v.conj().T).astype(np.complex128, copy=False))


def _as_merged_two_site(
    theta: NDArray[np.complex128], d0: int, d1: int, d_left: int, d_right: int
) -> NDArray[np.complex128]:
    """Reshape a flat ``theta`` matrix into a merged two-site MPS tensor.

    Interprets ``theta`` as having combined physical/bond indices
    ``(d0 * d_left, d1 * d_right)`` and converts it to the canonical merged
    two-site layout expected by ``split_two_site``.

    Args:
        theta: Matrix of shape ``(d0 * d_left, d1 * d_right)``.
        d0: Physical dimension of the left site.
        d1: Physical dimension of the right site.
        d_left: Left bond dimension.
        d_right: Right bond dimension.

    Returns:
        Merged two-site tensor of shape ``(d0 * d1, d_left, d_right)``.
    """
    tensor = theta.reshape(d0, d_left, d1, d_right).transpose(0, 2, 1, 3)
    return cast("NDArray[np.complex128]", tensor.reshape(d0 * d1, d_left, d_right))


def test_linalg_svd_reduced_shapes_unitary_and_reconstruction() -> None:
    """linalg.svd: reduced SVD has correct shapes, unitary factors, and reconstructs A."""
    a = crandn(7, 5)  # m > n (k = 5)
    u, s, vh = linalg.svd(a, full_matrices=False)

    k = min(a.shape)
    assert u.shape == (a.shape[0], k)
    assert s.shape == (k,)
    assert vh.shape == (k, a.shape[1])

    # U is column-orthonormal: U^H U = I_k
    iden_k = np.eye(k, dtype=np.complex128)
    assert np.allclose(u.conj().T @ u, iden_k)

    # Vh has orthonormal rows: Vh Vh^H = I_k
    assert np.allclose(vh @ vh.conj().T, iden_k)

    # Singular values are non-increasing and non-negative (within numerical tolerance)
    assert np.all(np.diff(s) <= 1e-12)
    assert np.all(s >= -1e-12)

    # Reconstruct A
    a_rec = u @ (np.diag(s) @ vh)
    assert np.allclose(a_rec, a)


def test_linalg_svd_full_shapes_unitary_and_reconstruction() -> None:
    """linalg.svd: full SVD has correct shapes, unitary factors, and reconstructs A.

    Reconstruction uses the standard full-SVD identity:
        A = U[:, :k] @ diag(s) @ Vh[:k, :]
    where k = min(m, n).
    """
    a = crandn(4, 6)
    u, s, vh = linalg.svd(a, full_matrices=True)

    m, n = a.shape
    k = min(m, n)

    assert u.shape == (m, m)
    assert s.shape == (k,)
    assert vh.shape == (n, n)

    iden_m = np.eye(m, dtype=np.complex128)
    iden_n = np.eye(n, dtype=np.complex128)

    assert np.allclose(u.conj().T @ u, iden_m)
    assert np.allclose(vh.conj().T @ vh, iden_n)

    a_rec = u[:, :k] @ (np.diag(s) @ vh[:k, :])
    assert np.allclose(a_rec, a)


LapackDriver = Literal["gesdd", "gesvd"]


def test_linalg_svd_falls_back_to_gesvd(monkeypatch: pytest.MonkeyPatch) -> None:
    """linalg.svd: if the fast driver fails, it retries with the robust driver."""
    calls: list[tuple[LapackDriver, bool]] = []
    real_svd = scipy.linalg.svd

    def fake_svd(
        a_mat: NDArray[np.complex128],
        *,
        full_matrices: bool,
        compute_uv: bool = True,
        lapack_driver: LapackDriver,
        check_finite: bool,
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.complex128]]:
        calls.append((lapack_driver, check_finite))
        if lapack_driver == "gesdd":
            msg = "forced failure in fast driver"
            raise scipy.linalg.LinAlgError(msg)

        u, s, vh = real_svd(
            a_mat,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            lapack_driver=lapack_driver,
            check_finite=check_finite,
        )

        return (u, s, vh)

    monkeypatch.setattr(scipy.linalg, "svd", fake_svd)

    a = crandn(6, 6)
    u, s, vh = linalg.svd(a, full_matrices=False)

    assert calls[0] == ("gesdd", False)
    assert calls[1] == ("gesvd", True)

    a_rec = u @ (np.diag(s) @ vh)
    assert np.allclose(a_rec, a)
