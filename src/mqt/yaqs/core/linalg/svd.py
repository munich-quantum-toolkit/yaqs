# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""SciPy-style SVD with gesdd-to-gesvd fallback.

Uses multi-threaded LAPACK/OpenBLAS by default; the BLAS pool is *not* capped
here. The wrapper retries with the more robust ``gesvd`` driver (and
``check_finite=True``) when ``gesdd`` fails to converge on ill-conditioned
inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import scipy.linalg

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

LapackDriver = Literal["gesdd", "gesvd"]


@overload
def svd(
    a: NDArray[np.complex128],
    *,
    full_matrices: bool = ...,
    compute_uv: Literal[True] = ...,
    lapack_driver: LapackDriver = ...,
    check_finite: bool = ...,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.complex128]]: ...


@overload
def svd(
    a: NDArray[np.complex128],
    *,
    full_matrices: bool = ...,
    compute_uv: Literal[False],
    lapack_driver: LapackDriver = ...,
    check_finite: bool = ...,
) -> NDArray[np.float64]: ...


def svd(
    a: NDArray[np.complex128],
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    lapack_driver: LapackDriver = "gesdd",
    check_finite: bool = False,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.complex128]] | NDArray[np.float64]:
    """Singular value decomposition with automatic driver fallback.

    When ``lapack_driver`` is ``"gesdd"`` (default), retries with ``"gesvd"``
    and ``check_finite=True`` on failure for ill-conditioned inputs.

    Mirrors :func:`scipy.linalg.svd` for the supported keyword arguments.

    Args:
        a: Complex matrix to decompose.
        full_matrices: If ``True`` (default), return full-size ``U`` and ``Vh``;
            otherwise return reduced (thin) factors.
        compute_uv: If ``True`` (default), compute and return ``U`` and ``Vh``;
            if ``False``, return only the singular values.
        lapack_driver: LAPACK driver to use (``"gesdd"`` or ``"gesvd"``).
            Defaults to ``"gesdd"``.
        check_finite: Whether to check input arrays for finite values.
            Defaults to ``False``.

    Returns:
        ``(U, s, Vh)`` when ``compute_uv`` is ``True``, else the 1D array of
        singular values ``s``.
    """
    if lapack_driver != "gesdd":
        return scipy.linalg.svd(
            a,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            lapack_driver=lapack_driver,
            check_finite=check_finite,
        )
    try:
        return scipy.linalg.svd(
            a,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            lapack_driver="gesdd",
            check_finite=check_finite,
        )
    except (scipy.linalg.LinAlgError, ValueError, FloatingPointError):
        return scipy.linalg.svd(
            a,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            lapack_driver="gesvd",
            check_finite=True,
        )
