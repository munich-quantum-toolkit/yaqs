# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for SciPy-style Hermitian tridiagonal eigensolve."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from mqt.yaqs.core import linalg

if TYPE_CHECKING:
    import pytest


def test_eigh_tridiagonal_matches_scipy_stemr() -> None:
    """Default ``stemr`` path should match SciPy for a small tridiagonal."""
    rng = np.random.default_rng(0)
    n = 8
    d = rng.standard_normal(n).astype(np.float64)
    e = rng.standard_normal(n - 1).astype(np.float64)
    w_ref, v_ref = scipy.linalg.eigh_tridiagonal(d, e, lapack_driver="stemr", check_finite=False)
    w, v = linalg.eigh_tridiagonal(d, e)
    np.testing.assert_allclose(w, w_ref, rtol=1e-10, atol=1e-10)
    # Eigenvectors defined up to sign
    np.testing.assert_allclose(np.abs(v), np.abs(v_ref), rtol=1e-10, atol=1e-10)


def test_eigh_tridiagonal_explicit_stebz() -> None:
    """Explicit ``lapack_driver='stebz'`` should return consistent eigenvalues."""
    rng = np.random.default_rng(1)
    n = 6
    d = rng.standard_normal(n).astype(np.float64)
    e = rng.standard_normal(n - 1).astype(np.float64)
    w_stebz, _ = linalg.eigh_tridiagonal(d, e, lapack_driver="stebz", check_finite=False)
    w_ref, _ = scipy.linalg.eigh_tridiagonal(d, e, lapack_driver="stemr", check_finite=False)
    np.testing.assert_allclose(w_stebz, w_ref, rtol=1e-9, atol=1e-9)


def test_eigh_tridiagonal_stemr_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``stemr`` fails, ``linalg.eigh_tridiagonal`` falls back to ``stebz``."""
    calls: list[str] = []
    real_eigh = scipy.linalg.eigh_tridiagonal

    def fake_eigh(
        d: np.ndarray,
        e: np.ndarray,
        *,
        lapack_driver: str,
        check_finite: bool,
        **_kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(lapack_driver)
        if lapack_driver == "stemr":
            msg = "forced"
            raise scipy.linalg.LinAlgError(msg)
        return real_eigh(d, e, lapack_driver=lapack_driver, check_finite=check_finite)

    monkeypatch.setattr(scipy.linalg, "eigh_tridiagonal", fake_eigh)
    d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    e = np.array([0.1, 0.2], dtype=np.float64)
    w, v = linalg.eigh_tridiagonal(d, e)
    assert calls[0] == "stemr"
    assert calls[1] == "stebz"
    assert w.shape == (3,)
    assert v.shape == (3, 3)
