# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for parallel simulator and equivalence-checker execution."""

from __future__ import annotations

import contextlib
import importlib
import multiprocessing
import os
import sys
from typing import Literal

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:
    threadpool_limits = None  # ty: ignore[invalid-assignment]
    threadpool_info = None

MPContext = Literal["fork", "spawn", "auto"]

THREAD_ENV_VARS: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}

__all__ = [
    "THREAD_ENV_VARS",
    "MPContext",
    "available_cpus",
    "get_parallel_context",
    "limit_worker_threads",
    "safe_set_numba_threads",
]


def available_cpus() -> int:
    """Return the number of CPUs available for parallel work."""
    if "YAQS_MAX_WORKERS" in os.environ:
        try:
            val = int(os.environ["YAQS_MAX_WORKERS"])
            if val > 0:
                return val
        except ValueError:
            pass

    if os.environ.get("PYTEST_XDIST_WORKER", ""):
        return 1

    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(var, "").strip()
        if value:
            try:
                n = int(value)
                if n > 0:
                    return n
            except ValueError:
                pass

    fn = getattr(os, "sched_getaffinity", None)
    if fn is not None:
        try:
            n = len(fn(0))
            if n > 0:
                return n
        except OSError:
            pass

    try:
        return os.cpu_count() or multiprocessing.cpu_count() or 1
    except (NotImplementedError, OSError):
        return 1


def get_parallel_context(mp_context: MPContext = "auto") -> multiprocessing.context.BaseContext:
    """Return a multiprocessing context for worker processes.

    Args:
        mp_context: Start method selector. ``"auto"`` uses ``"fork"`` on Linux and
            ``"spawn"`` elsewhere; ``"fork"`` or ``"spawn"`` select that method explicitly.

    Returns:
        A :class:`~multiprocessing.context.BaseContext` for creating worker processes.
    """
    if mp_context == "auto":
        if sys.platform == "linux":
            return multiprocessing.get_context("fork")
        return multiprocessing.get_context("spawn")
    return multiprocessing.get_context(mp_context)


def limit_worker_threads(n_threads: int = 1) -> None:
    """Limit BLAS/OpenMP thread pools in the current process.

    Sets environment variables and optional runtime hooks (numexpr, MKL,
    threadpoolctl) to avoid oversubscription when many worker processes run
    concurrently.

    Args:
        n_threads: Maximum threads per numerical library (default ``1``).
    """
    for key in THREAD_ENV_VARS:
        os.environ[key] = str(n_threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"

    with contextlib.suppress(Exception):
        numexpr = importlib.import_module("numexpr")
        numexpr.set_num_threads(n_threads)

    with contextlib.suppress(Exception):
        mkl = importlib.import_module("mkl")
        mkl.set_num_threads(n_threads)

    if threadpool_limits is not None:
        with contextlib.suppress(Exception):
            threadpool_limits(limits=n_threads)

    if os.environ.get("YAQS_THREAD_DEBUG", "") == "1" and threadpool_info is not None:
        with contextlib.suppress(Exception):
            threadpool_info()


def safe_set_numba_threads(n_threads: int) -> None:
    """Set Numba's thread count when the runtime pool allows it.

    Numba initializes its parallel thread pool on first use and may refuse
    later changes (for example when ``NUMBA_NUM_THREADS`` pinned the pool at
    process start). Failures are ignored so callers can still proceed with the
    existing pool size.

    Args:
        n_threads: Desired Numba thread count.
    """
    with contextlib.suppress(ImportError, AttributeError, ValueError, RuntimeError):
        numba = importlib.import_module("numba")
        numba.set_num_threads(n_threads)
