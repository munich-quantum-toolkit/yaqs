# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Mandatory governed-source inventory tests for WP22."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from benchmarks.state_preparation.phase2.source_lock import (
    WP22_GOVERNED_ANALYSIS_ENTRY_POINT,
    WP22_GOVERNED_ENTRY_POINT,
    capture_execution_source_manifest,
    capture_governed_execution_source_manifest,
    verify_governed_execution_source_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


def _git(repository: Path, *arguments: str) -> None:
    """Run one Git command in the isolated repository.

    Args:
        repository: Exact test repository root.
        arguments: Git subcommand and arguments.
    """
    executable = shutil.which("git")
    assert executable is not None
    subprocess.run(  # noqa: S603 - test-only Git executable is resolved with shutil.which and shell remains disabled
        (executable, "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
    )


def _write(repository: Path, repo_path: str, payload: str) -> None:
    """Write one governed fixture file below the test repository.

    Args:
        repository: Exact test repository root.
        repo_path: Repository-relative fixture path.
        payload: Text to write.
    """
    path = repository / repo_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


@pytest.fixture
def governed_repository(tmp_path: Path) -> Path:
    """Create a minimal clean checkout with the mandatory WP22 path classes.

    Args:
        tmp_path: Pytest-managed temporary directory.

    Returns:
        Clean committed repository root.
    """
    repository = tmp_path / "repository"
    repository.mkdir()
    _write(repository, WP22_GOVERNED_ENTRY_POINT, "def main():\n    return 0\n")
    _write(repository, WP22_GOVERNED_ANALYSIS_ENTRY_POINT, "ANALYSIS_VERSION = 1\n")
    _write(repository, "benchmarks/state_preparation/phase2/runtime.py", "RUNTIME_VERSION = 1\n")
    _write(
        repository,
        "benchmarks/state_preparation/phase2/data/initial_preregistration_v1.json",
        '{"protocol":1}\n',
    )
    _write(repository, "benchmarks/state_preparation/phase2/data/secondary.json", '{"secondary":1}\n')
    _write(repository, "src/mqt/yaqs/__init__.py", "RUNTIME_VERSION = 1\n")
    _write(repository, "src/mqt/yaqs/core/runtime.py", "CORE_VERSION = 1\n")
    _write(repository, "src/mqt/yaqs/core/table.bin", "tracked runtime data\n")
    _write(repository, "pyproject.toml", '[project]\nname = "governed-test"\n')
    _write(repository, "uv.lock", "version = 1\n")
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "WP22 Test")
    _git(repository, "config", "user.email", "wp22@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "initial")
    return repository


def test_governed_capture_includes_dynamic_code_data_and_locks(governed_repository: Path) -> None:
    """The mandatory capture expands over all tracked scientific dependencies."""
    manifest = capture_governed_execution_source_manifest(
        governed_repository,
        manifest_id="wp22_governed_sources_v1",
    )
    roles = {source.repo_path: source.role for source in manifest.source_files}
    assert manifest.entry_point == WP22_GOVERNED_ENTRY_POINT
    assert roles[WP22_GOVERNED_ANALYSIS_ENTRY_POINT] == "analysis_source"
    assert roles["benchmarks/state_preparation/phase2/runtime.py"] == "execution_source"
    assert roles["src/mqt/yaqs/core/table.bin"] == "execution_source"
    assert roles["benchmarks/state_preparation/phase2/data/secondary.json"] == "sealed_input"
    assert roles["pyproject.toml"] == roles["uv.lock"] == "dependency_lock"
    assert verify_governed_execution_source_manifest(manifest, governed_repository)


def test_governed_verification_rejects_caller_selected_subset(governed_repository: Path) -> None:
    """A valid generic source manifest cannot omit governed runtime dependencies."""
    subset = capture_execution_source_manifest(
        governed_repository,
        manifest_id="caller_selected_subset_v1",
        entry_point=WP22_GOVERNED_ENTRY_POINT,
        execution_source_paths=(WP22_GOVERNED_ENTRY_POINT, "src/mqt/yaqs/__init__.py"),
        analysis_source_paths=(WP22_GOVERNED_ANALYSIS_ENTRY_POINT,),
        dependency_lock_paths=("pyproject.toml", "uv.lock"),
        sealed_input_paths=("benchmarks/state_preparation/phase2/data/initial_preregistration_v1.json",),
    )
    with pytest.raises(ValueError, match="omits or misroles"):
        verify_governed_execution_source_manifest(subset, governed_repository)
