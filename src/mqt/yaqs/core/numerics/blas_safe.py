# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""BLAS-thread-safe dense linear algebra used from analog and Krylov backends.

SciPy's dense matrix exponential calls multi-threaded LAPACK/OpenBLAS. When many
pytest-xdist (or nested) processes each use multi-threaded BLAS concurrently,
some platform wheels (notably Linux aarch64 with older CPython) have shown
intermittent segmentation faults. Capping BLAS threads during these calls and
preferring :func:`numpy.linalg.eigh` for Hermitian Hamiltonians avoids the
highest-risk ``scipy.linalg.expm`` path for the common noiseless case.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _threadpool_limits_one() -> contextlib.AbstractContextManager[None]:
    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)


def is_hermitian_matrix(a: NDArray[np.complex128], *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Return whether ``a`` is Hermitian within tolerance."""
    return bool(np.allclose(a, a.conj().T, rtol=rtol, atol=atol))


def unitary_propagator_from_hermitian(h: NDArray[np.complex128], dt: float) -> NDArray[np.complex128]:
    """Compute ``U = exp(-1j * dt * H)`` for Hermitian ``H`` via an eigensolve.

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


def expm_dense(mat: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Dense matrix exponential with BLAS limited to a single thread.

    Args:
        mat: Square complex matrix.

    Returns:
        ``expm(mat)`` computed with :func:`scipy.linalg.expm`.
    """
    with _threadpool_limits_one():
        return scipy.linalg.expm(mat)
