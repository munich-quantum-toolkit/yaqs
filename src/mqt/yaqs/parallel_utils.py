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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import TYPE_CHECKING, Literal, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:
    threadpool_limits = None  # ty: ignore[invalid-assignment]
    threadpool_info = None

TArg = TypeVar("TArg")
TRes = TypeVar("TRes")

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
    "parallel_worker_init",
    "run_parallel_tasks",
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
    """Return a multiprocessing context for worker processes."""
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
        os.environ.setdefault(key, str(n_threads))
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")

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


def parallel_worker_init() -> None:
    """Cap BLAS/OpenMP thread pools inside worker processes (pool initializer)."""
    limit_worker_threads(1)


_parallel_worker_init = parallel_worker_init


def _run_on_executor(
    executor: ProcessPoolExecutor,
    tasks: list[TArg],
    worker_fn: Callable[[TArg], TRes],
) -> list[TRes]:
    """Submit ``tasks`` to an existing executor and return results in order.

    Args:
        executor: Executor used to submit work.
        tasks: Picklable task payloads, one per submission.
        worker_fn: Callable invoked as ``worker_fn(task)`` in worker processes.

    Returns:
        Worker results in the same order as ``tasks``.

    Raises:
        RuntimeError: If any task slot is left unset after all futures complete.
    """
    indexed = list(enumerate(tasks))
    results: list[TRes | None] = [None] * len(tasks)
    future_to_index = {executor.submit(worker_fn, task): index for index, task in indexed}
    while future_to_index:
        done, _ = wait(future_to_index, return_when=FIRST_COMPLETED)
        for future in done:
            index = future_to_index.pop(future)
            results[index] = future.result()
    if any(result is None for result in results):
        msg = "Parallel worker pool returned incomplete results."
        raise RuntimeError(msg)
    return cast("list[TRes]", results)


def run_parallel_tasks(
    tasks: list[TArg],
    worker_fn: Callable[[TArg], TRes],
    *,
    max_workers: int | None = None,
    mp_context: MPContext = "auto",
    executor: ProcessPoolExecutor | None = None,
) -> list[TRes]:
    """Run ``worker_fn`` on each task in a process pool and return results in order.

    Args:
        tasks: Picklable task payloads.
        worker_fn: Top-level callable ``worker_fn(task) -> result``.
        max_workers: Process count; defaults to :func:`available_cpus`.
        mp_context: Multiprocessing start method (ignored if ``executor`` is set).
        executor: Optional existing pool; when set, tasks are submitted without
            creating a new executor (avoids repeated spawn/fork overhead).

    Returns:
        Results in the same order as ``tasks``.
    """
    if not tasks:
        return []

    workers = max_workers if max_workers is not None else available_cpus()
    workers = max(1, min(workers, len(tasks)))

    if workers == 1:
        return [worker_fn(task) for task in tasks]

    if executor is not None:
        return _run_on_executor(executor, tasks, worker_fn)

    ctx = get_parallel_context(mp_context)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_parallel_worker_init,
    ) as pool:
        return _run_on_executor(pool, tasks, worker_fn)
