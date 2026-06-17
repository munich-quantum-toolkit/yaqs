# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""SciPy-style dense linear algebra with BLAS-thread-safe defaults.

This package mirrors :mod:`scipy.linalg` for the subset of operations YAQS uses
internally; submodules group related helpers (e.g. :mod:`.expm`, :mod:`.svd`).
"""

from __future__ import annotations

from .eigh import eigh_tridiagonal
from .expm import expm, expm_hermitian, ishermitian
from .svd import svd
from .svd_utils import truncate

__all__ = ["eigh_tridiagonal", "expm", "expm_hermitian", "ishermitian", "svd", "truncate"]
