# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""SciPy-style Hermitian tridiagonal eigensolve with stemr-to-stebz fallback."""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from ._threading import threadpool_limits_one

LapackDriver = Literal["stemr", "stebz"]


def eigh_tridiagonal(
    d: NDArray[np.float64],
    e: NDArray[np.float64],
    *,
    lapack_driver: LapackDriver = "stemr",
    check_finite: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eigenvalues and eigenvectors of a Hermitian tridiagonal matrix.

    When ``lapack_driver`` is ``\"stemr\"`` (default), falls back to ``\"stebz\"``
    on :class:`scipy.linalg.LinAlgError`.

    Mirrors :func:`scipy.linalg.eigh_tridiagonal` for the supported arguments.
    """
    with threadpool_limits_one():
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
