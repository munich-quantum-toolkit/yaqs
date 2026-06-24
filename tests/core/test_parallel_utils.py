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
from mqt.yaqs.core.parallel_utils import get_parallel_context, worker_init
from mqt.yaqs.simulator import available_cpus as simulator_available_cpus


def test_available_cpus_without_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without SLURM env vars, ``available_cpus`` falls back to ``cpu_count``."""
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

    assert parallel_utils.available_cpus() == multiprocessing.cpu_count()


def test_available_cpus_with_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SLURM_CPUS_ON_NODE`` is honoured when set."""
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("YAQS_MAX_WORKERS", raising=False)

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
