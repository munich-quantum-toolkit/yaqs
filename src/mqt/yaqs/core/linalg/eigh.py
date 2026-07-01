# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""SciPy-style Hermitian tridiagonal eigensolve with stemr-to-stebz fallback.

Uses multi-threaded LAPACK/OpenBLAS by default; the BLAS pool is *not* capped
here. The wrapper retries with the slower-but-more-robust ``stebz`` driver
when ``stemr`` raises :class:`scipy.linalg.LinAlgError` on ill-conditioned
inputs (a pattern that occurs in the Krylov / Lanczos paths).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import scipy.linalg

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

LapackDriver = Literal["stemr", "stebz"]


def eigh_tridiagonal(
    d: NDArray[np.float64],
    e: NDArray[np.float64],
    *,
    lapack_driver: LapackDriver = "stemr",
    check_finite: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenvalues and eigenvectors of a Hermitian tridiagonal matrix.

    When ``lapack_driver`` is ``"stemr"`` (default), falls back to ``"stebz"``
    on :class:`scipy.linalg.LinAlgError`.

    Mirrors :func:`scipy.linalg.eigh_tridiagonal` for the supported arguments.

    Args:
        d: Diagonal elements of the symmetric tridiagonal matrix.
        e: Off-diagonal elements of the symmetric tridiagonal matrix.
        lapack_driver: LAPACK driver to use (``"stemr"`` or ``"stebz"``).
            Defaults to ``"stemr"``.
        check_finite: Whether to check input arrays for finite values.
            Defaults to ``False``.

    Returns:
        Tuple ``(w, v)`` of eigenvalues (ascending) and the corresponding
        orthonormal eigenvectors as columns.
    """
    if lapack_driver != "stemr":
        return scipy.linalg.eigh_tridiagonal(
            d,
            e,
            lapack_driver=lapack_driver,
            check_finite=check_finite,
        )
    try:
        return scipy.linalg.eigh_tridiagonal(
            d,
            e,
            lapack_driver="stemr",
            check_finite=check_finite,
        )
    except scipy.linalg.LinAlgError:
        return scipy.linalg.eigh_tridiagonal(
            d,
            e,
            lapack_driver="stebz",
            check_finite=check_finite,
        )
