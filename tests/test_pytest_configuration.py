# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the repository's pytest configuration."""

from __future__ import annotations

from typing import cast

import pytest

from tests import conftest


def test_numerical_thread_pools_default_to_one_thread() -> None:
    """Numerical thread pools use one thread when the runner has no override."""
    environment: dict[str, str] = {}

    conftest.set_default_thread_limits(environment)

    assert environment
    assert set(environment.values()) == {"1"}


def test_numerical_thread_pool_defaults_preserve_runner_overrides() -> None:
    """Explicit local or CI thread limits take precedence over test defaults."""
    environment = {"OMP_NUM_THREADS": "runner-value"}

    conftest.set_default_thread_limits(environment)

    assert environment["OMP_NUM_THREADS"] == "runner-value"


@pytest.mark.parametrize(
    ("environment", "worker_limit"),
    [
        pytest.param(
            {"YAQS_MAX_WORKERS": "4", "SLURM_CPUS_PER_TASK": "3", "SLURM_CPUS_ON_NODE": "2"},
            4,
            id="yaqs-precedes-slurm",
        ),
        pytest.param(
            {"SLURM_CPUS_PER_TASK": "3", "SLURM_CPUS_ON_NODE": "2"},
            3,
            id="slurm-task-precedes-node",
        ),
        pytest.param({"SLURM_CPUS_ON_NODE": "2"}, 2, id="slurm-node"),
    ],
)
def test_xdist_auto_worker_count_respects_shared_limits(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], worker_limit: int
) -> None:
    """Configured YAQS and SLURM limits cap local and GitHub Actions workers."""
    for name in ("YAQS_MAX_WORKERS", "PYTEST_XDIST_WORKER", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    config = cast("pytest.Config", object())

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert conftest.pytest_xdist_auto_num_workers(config) == worker_limit

    monkeypatch.delenv("GITHUB_ACTIONS")
    physical_cpus = conftest.psutil.cpu_count(logical=False) or worker_limit
    expected_local_workers = max(1, min(physical_cpus, worker_limit) - 1)
    assert conftest.pytest_xdist_auto_num_workers(config) == expected_local_workers
