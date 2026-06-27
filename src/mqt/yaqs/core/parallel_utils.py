# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for parallel simulator, characterization, and equivalence-checker execution."""

from __future__ import annotations

import contextlib
import importlib
import multiprocessing
import os
import sys
from concurrent.futures import FIRST_COMPLETED, CancelledError, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from tqdm import tqdm

from mqt.yaqs.core.linalg._threading import threadpool_limits_one  # noqa: PLC2701

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from concurrent.futures import Future

TRes = TypeVar("TRes")

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
    "reassemble_indexed",
    "resolve_worker_ctx",
    "safe_set_numba_threads",
    "unpack_flat_job",
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


# ---------------------------------------------------------------------------
# Process-pool orchestration (used by Simulator, MemoryCharacterizer, and internals)
# ---------------------------------------------------------------------------

# Global worker state (initialized once per worker process).
WORKER_CTX: dict[str, Any] = {}


@dataclass(frozen=True)
class ExecutionConfig:
    """Internal execution-side configuration for parallel job dispatch."""

    parallel: bool = True
    max_workers: int | None = None
    show_progress: bool = True
    mp_context: MPContext = "auto"
    max_retries: int = 10
    retry_exceptions: tuple[type[BaseException], ...] = (CancelledError, TimeoutError, OSError)

    def resolved_max_workers(self) -> int:
        """Return the effective worker count."""
        if self.max_workers is not None:
            return max(1, int(self.max_workers))
        return max(1, available_cpus() - 1)


class _UnsetType:
    """Sentinel for optional merge fields that distinguish unset from explicit ``None``."""


_UNSET = _UnsetType()


def merge_execution_config(
    execution: ExecutionConfig | None,
    *,
    parallel: bool | None = None,
    show_progress: bool | None = None,
    max_workers: int | _UnsetType | None = _UNSET,
    mp_context: MPContext | None = None,
    max_retries: int | None = None,
) -> ExecutionConfig:
    """Merge optional overrides into an :class:`ExecutionConfig`.

    Returns:
        Updated execution configuration.
    """
    base = execution or ExecutionConfig()
    updates: dict[str, Any] = {}
    if parallel is not None:
        updates["parallel"] = bool(parallel)
    if show_progress is not None:
        updates["show_progress"] = bool(show_progress)
    if max_workers is not _UNSET:
        if isinstance(max_workers, int):
            updates["max_workers"] = max_workers
        else:
            updates["max_workers"] = None
    if mp_context is not None:
        updates["mp_context"] = mp_context
    if max_retries is not None:
        updates["max_retries"] = int(max_retries)
    return replace(base, **updates) if updates else base


# ---------------------------------------------------------------------------
# Worker helpers — shared by characterization pool workers
# ---------------------------------------------------------------------------
# Pool workers take ``(job_idx, payload=None)``. When ``payload`` is omitted they
# read :data:`WORKER_CTX`, installed once per process by :func:`worker_init`.
# Flat job indices encode ``sequence_index * num_trajectories + trajectory_index``;
# use :func:`unpack_flat_job` and :func:`reassemble_indexed` to map results back.
def resolve_worker_ctx(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``payload`` when set, otherwise the process-pool :data:`WORKER_CTX`."""
    return WORKER_CTX if payload is None else payload


def unpack_flat_job(job_idx: int, num_trajectories: int) -> tuple[int, int]:
    """Unpack a flat job index into ``(sequence_index, trajectory_index)``.

    Returns:
        Tuple ``(sequence_index, trajectory_index)``.
    """
    n_traj = int(num_trajectories)
    idx = int(job_idx)
    return idx // n_traj, idx % n_traj


def reassemble_indexed(
    results: dict[int, TRes],
    n_jobs: int,
    *,
    label: str,
) -> list[TRes]:
    """Build an ordered result list from a job-index map; raise if any slot is missing.

    Returns:
        Results ordered by job index ``0 .. n_jobs-1``.

    Raises:
        RuntimeError: If any job index is missing from ``results``.
    """
    n = int(n_jobs)
    missing = [i for i in range(n) if i not in results]
    if missing:
        msg = f"{label}: parallel execution incomplete (missing indices: {missing[:8]})."
        raise RuntimeError(msg)
    return [results[i] for i in range(n)]


def worker_init(payload: dict[str, Any], n_threads: int = 1) -> None:
    """Initialize worker process thread caps and shared payload context."""
    limit_worker_threads(n_threads)
    WORKER_CTX.clear()
    WORKER_CTX.update(payload)
    safe_set_numba_threads(n_threads)


def call_serial_capped(fn: Callable[..., TRes], /, *args: object, n_threads: int = 1) -> TRes:
    """Invoke ``fn(*args)`` under Numba/BLAS thread caps.

    Returns:
        Value returned by ``fn``.
    """
    safe_set_numba_threads(n_threads)
    with threadpool_limits_one():
        return fn(*args)


def run_backend_parallel(
    worker_fn: Callable[[int], TRes],
    *,
    payload: dict[str, Any] | None,
    n_jobs: int,
    max_workers: int,
    show_progress: bool = True,
    desc: str,
    max_retries: int = 10,
    retry_exceptions: tuple[type[BaseException], ...] = (CancelledError, TimeoutError, OSError),
    mp_context: MPContext = "auto",
) -> Iterator[tuple[int, TRes]]:
    """Execute indexed jobs in parallel with bounded submission and retry logic.

    Yields:
        Pairs ``(job_index, result)`` as jobs complete.
    """
    ctx = get_parallel_context(mp_context)
    inflight_factor = 2
    max_inflight = max_workers * inflight_factor

    with (
        ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=worker_init,
            initargs=(payload or {}, 1),
        ) as ex,
        tqdm(total=n_jobs, desc=desc, ncols=80, disable=(not show_progress)) as pbar,
    ):
        retries = dict.fromkeys(range(n_jobs), 0)
        futures: dict[Future[TRes], int] = {}
        next_job_idx = 0

        def submit_job(idx: int) -> None:
            futures[ex.submit(worker_fn, idx)] = idx

        while next_job_idx < n_jobs and len(futures) < max_inflight:
            submit_job(next_job_idx)
            next_job_idx += 1

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                i = futures.pop(fut)
                try:
                    res = fut.result()
                except retry_exceptions:
                    if retries[i] < max_retries:
                        retries[i] += 1
                        submit_job(i)
                        continue
                    raise

                yield i, res
                pbar.update(1)

                if next_job_idx < n_jobs:
                    submit_job(next_job_idx)
                    next_job_idx += 1


# ---------------------------------------------------------------------------
# Indexed job dispatch — parallel or serial (characterization entry point)
# ---------------------------------------------------------------------------
def run_indexed_jobs(
    worker_fn: Callable[..., TRes],
    *,
    payload: dict[str, Any],
    n_jobs: int,
    config: ExecutionConfig,
    desc: str,
) -> dict[int, TRes]:
    """Run indexed jobs in parallel or serially, returning results keyed by job index.

    Returns:
        Mapping from job index to worker result.
    """
    results: dict[int, TRes] = {}
    max_workers = config.resolved_max_workers()
    if config.parallel and n_jobs > 1 and max_workers > 1:
        results.update(
            dict(
                run_backend_parallel(
                    worker_fn=worker_fn,
                    payload=payload,
                    n_jobs=n_jobs,
                    max_workers=max_workers,
                    show_progress=config.show_progress,
                    desc=desc,
                    max_retries=config.max_retries,
                    retry_exceptions=config.retry_exceptions,
                    mp_context=config.mp_context,
                )
            )
        )
        return results

    for job_idx in tqdm(
        range(n_jobs),
        desc=f"{desc} (serial)",
        ncols=80,
        disable=(not config.show_progress),
    ):
        results[job_idx] = call_serial_capped(worker_fn, job_idx, payload)
    return results
