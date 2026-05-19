# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared BLAS thread-pool limiter for dense linear algebra."""

from __future__ import annotations

import contextlib


def threadpool_limits_one() -> contextlib.AbstractContextManager[None]:
    """Return a context manager that caps BLAS/OpenMP pools to one thread."""
    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=1)
