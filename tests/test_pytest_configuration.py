# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the repository's pytest configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from tests import conftest

if TYPE_CHECKING:
    import pytest


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


def test_local_xdist_auto_worker_count_reserves_one_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local automatic worker selection keeps one available physical CPU free."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    available_logical_cpus = conftest.available_logical_cpus()
    physical_cpus = conftest.psutil.cpu_count(logical=False) or available_logical_cpus
    expected_workers = max(1, min(physical_cpus, available_logical_cpus) - 1)
    config = cast("pytest.Config", object())
    assert conftest.pytest_xdist_auto_num_workers(config) == expected_workers


def test_github_actions_uses_its_full_cpu_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """GitHub Actions keeps its existing automatic worker allocation."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    config = cast("pytest.Config", object())

    assert conftest.pytest_xdist_auto_num_workers(config) == conftest.available_logical_cpus()
