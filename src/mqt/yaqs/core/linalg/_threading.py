# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""BLAS thread-pool limiter for dense matrix exponentials.

Used by :mod:`mqt.yaqs.core.linalg.expm` to avoid intermittent segmentation
faults observed when multi-threaded OpenBLAS ``expm`` runs concurrently across
many processes (pytest-xdist, nested parallelism). Other linalg helpers in this
package do not need the cap.
"""

from __future__ import annotations

import contextlib

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None  # ty: ignore[invalid-assignment]


def threadpool_limits_one() -> contextlib.AbstractContextManager[None]:
    """Return a context manager that caps BLAS/OpenMP pools to one thread."""
    if threadpool_limits is None:
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)
