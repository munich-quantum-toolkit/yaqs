# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""SciPy-style dense matrix exponential helpers with BLAS-thread-safe defaults.

Dense matrix exponentials call multi-threaded LAPACK/OpenBLAS; when many
processes run concurrently (pytest-xdist, nested parallelism), some platform
wheels have shown intermittent segmentation faults specifically in
``scipy.linalg.expm``. Callers should use :func:`expm` here instead of
:func:`scipy.linalg.expm` directly so the BLAS thread cap is applied.

:func:`expm_hermitian` is a YAQS-specific shortcut (no SciPy analogue) that
computes ``exp(-1j * dt * H)`` for Hermitian ``H`` via an eigensolve.

Other linalg helpers in this package (``svd``, ``eigh_tridiagonal``) keep the
default multi-threaded BLAS for performance; only the matrix-exponential calls
have shown the crash signature that motivated the cap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from ._threading import threadpool_limits_one

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["expm", "expm_hermitian", "ishermitian"]


def ishermitian(a: NDArray[np.complex128], *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Return whether ``a`` is Hermitian within tolerance.

    Mirrors :func:`scipy.linalg.ishermitian` keyword arguments. Non-2-D or
    non-square inputs return ``False`` rather than raising.

    Args:
        a: Square matrix.
        rtol: Relative tolerance passed to :func:`numpy.allclose`.
        atol: Absolute tolerance passed to :func:`numpy.allclose`.

    Returns:
        True if ``a`` is Hermitian within tolerance, False otherwise (including
        for non-2-D or non-square inputs).
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return False
    return bool(np.allclose(a, a.conj().T, rtol=rtol, atol=atol))


def expm_hermitian(h: NDArray[np.complex128], dt: float) -> NDArray[np.complex128]:
    """Compute ``U = exp(-1j * dt * H)`` for Hermitian ``H`` via an eigensolve.

    YAQS-specific helper (no direct SciPy equivalent). Prefer :func:`expm` when
    ``H`` is not Hermitian.

    Args:
        h: Dense Hermitian Hamiltonian.
        dt: Time step.

    Returns:
        Dense unitary propagator of shape ``h.shape``.
    """
    with threadpool_limits_one():
        evals, evecs = np.linalg.eigh(h)
        phases = np.exp(-1j * dt * evals)
        return (evecs * phases) @ evecs.conj().T


def expm(a: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Dense matrix exponential with BLAS limited to a single thread.

    Mirrors :func:`scipy.linalg.expm`.

    Args:
        a: Square complex matrix.

    Returns:
        Matrix exponential of ``a``.
    """
    with threadpool_limits_one():
        return scipy.linalg.expm(a)
