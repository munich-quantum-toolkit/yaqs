# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""SciPy-style SVD with gesdd-to-gesvd fallback and BLAS thread cap."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from ._threading import threadpool_limits_one

LapackDriver = Literal["gesdd", "gesvd"]


def svd(
    a: NDArray[np.complex128],
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    lapack_driver: LapackDriver = "gesdd",
    check_finite: bool = False,
) -> Any:
    """Singular value decomposition with BLAS limited to one thread.

    When ``lapack_driver`` is ``\"gesdd\"`` (default), retries with ``\"gesvd\"``
    and ``check_finite=True`` on failure, matching the former ``robust_svd`` path.

    Mirrors :func:`scipy.linalg.svd` for the supported keyword arguments.

    Returns:
        ``(U, s, Vh)`` when ``compute_uv`` is True, else the 1D singular values ``s``.
    """
    with threadpool_limits_one():
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
