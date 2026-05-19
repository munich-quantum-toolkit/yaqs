# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""SciPy-style dense linear algebra with BLAS-thread-safe defaults.

This module mirrors :mod:`scipy.linalg` for the subset of operations YAQS uses
internally. Dense matrix exponentials call multi-threaded LAPACK/OpenBLAS; when
many processes run concurrently (pytest-xdist, nested parallelism), some
platform wheels have shown intermittent segmentation faults. Callers should use
:func:`expm` and related helpers here instead of :func:`scipy.linalg.expm`
directly.

:func:`expm_hermitian` is a YAQS-specific shortcut (no SciPy analogue) that
computes ``exp(-1j * dt * H)`` for Hermitian ``H`` via an eigensolve.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["expm", "expm_hermitian", "ishermitian"]


def _threadpool_limits_one() -> contextlib.AbstractContextManager[None]:
    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)


def ishermitian(a: NDArray[np.complex128], *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Return whether ``a`` is Hermitian within tolerance.

    Mirrors :func:`scipy.linalg.ishermitian` keyword arguments.

    Args:
        a: Square matrix.
        rtol: Relative tolerance passed to :func:`numpy.allclose`.
        atol: Absolute tolerance passed to :func:`numpy.allclose`.

    Returns:
        True if ``a`` is Hermitian within tolerance.
    """
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
    with _threadpool_limits_one():
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
    with _threadpool_limits_one():
        return scipy.linalg.expm(a)
