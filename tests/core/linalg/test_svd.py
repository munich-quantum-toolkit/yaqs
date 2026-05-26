# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for SciPy-style BLAS-thread-safe SVD."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from mqt.yaqs.core import linalg

if TYPE_CHECKING:
    import pytest


def test_svd_matches_scipy_reduced() -> None:
    """``linalg.svd`` with default driver should match SciPy on a small matrix."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal((6, 4)) + 1j * rng.standard_normal((6, 4))
    a = a.astype(np.complex128)
    u_ref, s_ref, vh_ref = scipy.linalg.svd(a, full_matrices=False, lapack_driver="gesdd", check_finite=False)
    u, s, vh = linalg.svd(a, full_matrices=False)
    np.testing.assert_allclose(s, s_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.abs(u), np.abs(u_ref), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.abs(vh), np.abs(vh_ref), rtol=1e-10, atol=1e-10)


def test_svd_compute_uv_false() -> None:
    """``compute_uv=False`` should return singular values only."""
    rng = np.random.default_rng(43)
    a = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    a = a.astype(np.complex128)
    s_only = linalg.svd(a, full_matrices=False, compute_uv=False)
    _, s_ref, _ = scipy.linalg.svd(a, full_matrices=False, compute_uv=True)
    np.testing.assert_allclose(s_only, s_ref, rtol=1e-10, atol=1e-10)


def test_svd_explicit_gesvd_driver() -> None:
    """Explicit ``lapack_driver='gesvd'`` should return consistent singular values."""
    rng = np.random.default_rng(44)
    a = rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))
    a = a.astype(np.complex128)
    _, s_gesvd, _ = linalg.svd(a, full_matrices=False, lapack_driver="gesvd", check_finite=True)
    _, s_ref, _ = scipy.linalg.svd(a, full_matrices=False, lapack_driver="gesdd", check_finite=False)
    np.testing.assert_allclose(s_gesvd, s_ref, rtol=1e-9, atol=1e-9)


def test_svd_gesdd_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``gesdd`` fails, ``linalg.svd`` retries with ``gesvd``."""
    calls: list[tuple[str, bool]] = []
    real_svd = scipy.linalg.svd

    def fake_svd(
        a_mat: np.ndarray,
        *,
        full_matrices: bool,
        compute_uv: bool = True,
        lapack_driver: str = "gesdd",
        check_finite: bool = True,
        **_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        calls.append((lapack_driver, check_finite))
        if lapack_driver == "gesdd":
            msg = "forced failure"
            raise scipy.linalg.LinAlgError(msg)
        return real_svd(
            a_mat,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            lapack_driver=lapack_driver,
            check_finite=check_finite,
        )

    monkeypatch.setattr(scipy.linalg, "svd", fake_svd)
    a = np.eye(3, dtype=np.complex128)
    u, s, vh = linalg.svd(a, full_matrices=False)
    assert calls[0] == ("gesdd", False)
    assert calls[1] == ("gesvd", True)
    np.testing.assert_allclose(u @ np.diag(s) @ vh, a, atol=1e-12)
