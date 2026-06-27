# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared parallel execution helpers."""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import sys

import numba
import pytest

from mqt.yaqs.core import parallel_utils
from mqt.yaqs.core.parallel_utils import (
    ExecutionConfig,
    available_cpus,
    call_serial_capped,
    get_parallel_context,
    merge_execution_config,
    resolve_worker_ctx,
    run_indexed_jobs,
    unpack_flat_job,
    worker_init,
)
from mqt.yaqs.simulator import available_cpus as simulator_available_cpus


def test_available_cpus_without_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without overrides, ``available_cpus`` falls back to affinity or ``cpu_count``."""
    monkeypatch.delenv("YAQS_MAX_WORKERS", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(4)), raising=False)

    assert parallel_utils.available_cpus() == 4


def test_available_cpus_with_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SLURM_CPUS_ON_NODE`` is honoured when higher-priority overrides are absent."""
    monkeypatch.delenv("YAQS_MAX_WORKERS", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")

    assert parallel_utils.available_cpus() == 8


def test_available_cpus_yaqs_max_workers_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``YAQS_MAX_WORKERS`` env var takes priority over xdist/SLURM/affinity."""
    monkeypatch.setenv("YAQS_MAX_WORKERS", "4")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert parallel_utils.available_cpus() == 4


def test_available_cpus_yaqs_max_workers_malformed_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed ``YAQS_MAX_WORKERS`` is ignored; later detection logic runs."""
    monkeypatch.setenv("YAQS_MAX_WORKERS", "not-a-number")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    assert parallel_utils.available_cpus() == 1


def test_available_cpus_xdist_worker_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running inside an xdist worker pins ``available_cpus`` to 1."""
    monkeypatch.delenv("YAQS_MAX_WORKERS", raising=False)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    assert parallel_utils.available_cpus() == 1


def test_available_cpus_slurm_malformed_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed SLURM_* values are ignored; the function falls back to affinity/cpu_count."""
    monkeypatch.delenv("YAQS_MAX_WORKERS", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "0")
    assert parallel_utils.available_cpus() >= 1


def test_simulator_reexports_available_cpus() -> None:
    """``simulator.available_cpus`` remains a public alias for the core helper."""
    assert simulator_available_cpus is parallel_utils.available_cpus


def test_threading_config() -> None:
    """Verify correct multiprocessing context and Numba threading configuration."""
    ctx = get_parallel_context()
    if sys.platform == "linux":
        assert ctx.get_start_method() == "fork"
    else:
        assert ctx.get_start_method() == "spawn"

    original_numba_threads = numba.get_num_threads()
    env_snapshot = os.environ.copy()

    try:
        worker_init({}, n_threads=1)
        assert numba.get_num_threads() == 1
        assert os.environ.get("NUMBA_NUM_THREADS") == "1"
    finally:
        for key in list(os.environ):
            if key not in env_snapshot:
                del os.environ[key]

        for key, value in env_snapshot.items():
            if os.environ.get(key) != value:
                os.environ[key] = value

        with contextlib.suppress(Exception):
            numba.set_num_threads(original_numba_threads)


def test_resolve_worker_ctx_and_unpack_flat_job() -> None:
    """Worker helpers resolve pool context and flat job indices."""
    payload = {"num_trajectories": 2, "x": 1}
    worker_init(payload)
    assert resolve_worker_ctx(None)["x"] == 1
    assert resolve_worker_ctx({"y": 2})["y"] == 2
    assert unpack_flat_job(3, 2) == (1, 1)


def test_reassemble_indexed_raises_on_missing() -> None:
    """Incomplete parallel maps raise with a descriptive label."""
    with pytest.raises(RuntimeError, match="incomplete"):
        parallel_utils.reassemble_indexed({0: 1}, 2, label="test_job")


def test_get_parallel_context_explicit_fork_and_spawn() -> None:
    """Explicit ``mp_context`` overrides platform auto-detection."""
    spawn_ctx = get_parallel_context("spawn")
    assert spawn_ctx.get_start_method() == "spawn"

    try:
        multiprocessing.get_context("fork")
    except ValueError:
        with pytest.raises(ValueError, match="cannot find context"):
            get_parallel_context("fork")
    else:
        fork_ctx = get_parallel_context("fork")
        assert fork_ctx.get_start_method() == "fork"


def test_merge_execution_config_applies_overrides() -> None:
    """merge_execution_config overlays parallel and worker settings."""
    base = ExecutionConfig(parallel=True, max_workers=3, show_progress=True)
    merged = merge_execution_config(base, parallel=False, max_workers=2, show_progress=False)
    assert merged.parallel is False
    assert merged.max_workers == 2
    assert merged.show_progress is False


def test_merge_execution_config_clears_max_workers() -> None:
    """Explicit max_workers=None restores the default worker policy."""
    base = ExecutionConfig(parallel=True, max_workers=2, show_progress=True)
    merged = merge_execution_config(base, max_workers=None)
    assert merged.max_workers is None
    assert merged.resolved_max_workers() == max(1, available_cpus() - 1)


def test_call_serial_capped_preserves_order() -> None:
    """Serial capped execution returns the worker result in-process."""
    seen: list[int] = []

    def worker(x: int) -> int:
        seen.append(x)
        return x * 2

    assert call_serial_capped(worker, 3) == 6
    assert seen == [3]


def test_run_indexed_jobs_serial_ordering() -> None:
    """Serial run_indexed_jobs executes every index and preserves results."""
    calls: list[int] = []

    def worker(job_idx: int, payload: dict[str, int]) -> int:
        calls.append(job_idx)
        return job_idx + payload["offset"]

    config = ExecutionConfig(parallel=False, show_progress=False)
    results = run_indexed_jobs(
        worker,
        payload={"offset": 10},
        n_jobs=4,
        config=config,
        desc="test",
    )
    assert calls == [0, 1, 2, 3]
    assert results == {i: i + 10 for i in range(4)}


def test_run_indexed_jobs_uses_serial_when_max_workers_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single resolved worker avoids spawning a process pool."""
    monkeypatch.setattr(parallel_utils, "available_cpus", lambda: 1)
    calls: list[int] = []

    def worker(job_idx: int, _payload: dict[str, int]) -> int:
        calls.append(job_idx)
        return job_idx

    config = ExecutionConfig(parallel=True, show_progress=False)
    results = run_indexed_jobs(
        worker,
        payload={},
        n_jobs=3,
        config=config,
        desc="test",
    )
    assert calls == [0, 1, 2]
    assert results == {0: 0, 1: 1, 2: 2}
