# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for decompositions.

This module tests the left and right QR decompositions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.linalg

from mqt.yaqs.core import linalg
from mqt.yaqs.core.methods.decompositions import left_qr, right_qr

if TYPE_CHECKING:
    import pytest
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
