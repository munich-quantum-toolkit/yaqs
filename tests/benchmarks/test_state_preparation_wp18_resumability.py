# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the explicit Phase II resumability fingerprint."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation.phase2.resumability import (
    ExecutionSourceEntry,
    NonScientificResumeOverride,
    ResumabilityFingerprint,
    ResumabilityMismatchError,
    capture_resumability_fingerprint,
    require_resumability_match,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


PREFIX_A = f"phase2_pipeline_prefix_{'a' * 64}"
PREFIX_B = f"phase2_pipeline_prefix_{'b' * 64}"
DEPENDENCIES = {"numpy": "2.3.2", "python": "3.13.5", "yaqs": "0.10.0"}


def _run_git(repository: Path, *arguments: str) -> str:
    """Run Git in one isolated test repository and return stdout.

    Returns:
        Stripped command stdout.
    """
    executable = shutil.which("git")
    assert executable is not None
    completed = subprocess.run(  # ruff: ignore[S603] -- resolved executable; no shell interpretation
        (executable, "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def source_repository(tmp_path: Path) -> tuple[Path, str]:
    """Create a committed repository with one file for every manifest role.

    Returns:
        The repository root and initial commit.
    """
    repository = tmp_path / "repository"
    repository.mkdir()
    _run_git(repository, "init", "--quiet")
    _run_git(repository, "config", "user.name", "WP18 Test")
    _run_git(repository, "config", "user.email", "wp18@example.invalid")
    (repository / "src").mkdir()
    (repository / "config").mkdir()
    (repository / "src" / "method.py").write_text("METHOD_VERSION = 1\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (repository / "config" / "protocol.json").write_text('{"protocol":1}\n', encoding="utf-8")
    _run_git(repository, "add", "src/method.py", "uv.lock", "config/protocol.json")
    _run_git(repository, "commit", "--quiet", "-m", "initial")
    return repository, _run_git(repository, "rev-parse", "HEAD")


def _capture(
    source_repository: tuple[Path, str],
    *,
    prefix: str = PREFIX_A,
    dependencies: Mapping[str, str] = DEPENDENCIES,
    output_root: Path | None = None,
) -> ResumabilityFingerprint:
    """Capture the standard three-file test fingerprint.

    Returns:
        The captured fingerprint.
    """
    repository, commit = source_repository
    return capture_resumability_fingerprint(
        repository,
        output_root=output_root or repository / "output" / "run",
        starting_commit=commit,
        pipeline_prefix_id=prefix,
        dependency_versions=dependencies,
        execution_source_paths=(Path("src/method.py"),),
        lockfile_paths=(repository / "uv.lock",),
        sealed_input_paths=(Path("config/protocol.json"),),
    )


def test_capture_is_canonical_immutable_and_round_trips(
    source_repository: tuple[Path, str],
) -> None:
    """Explicit entries and dependency versions form one immutable record."""
    repository, commit = source_repository
    fingerprint = _capture(source_repository)

    assert fingerprint.starting_commit == commit
    assert tuple(entry.role for entry in fingerprint.entries) == (
        "execution_source",
        "lockfile",
        "sealed_input",
    )
    assert tuple(entry.repository_path for entry in fingerprint.entries) == (
        "src/method.py",
        "uv.lock",
        "config/protocol.json",
    )
    assert isinstance(fingerprint.dependency_versions, MappingProxyType)
    assert fingerprint.execution_sources[0].starting_git_blob_id == _run_git(
        repository,
        "rev-parse",
        f"{commit}:src/method.py",
    )
    assert fingerprint.execution_sources[0].content_checksum == (
        f"sha256:{hashlib.sha256((repository / 'src' / 'method.py').read_bytes()).hexdigest()}"
    )
    assert ResumabilityFingerprint.from_dict(fingerprint.to_dict()) == fingerprint
    assert ResumabilityFingerprint.from_json(fingerprint.to_json()) == fingerprint
    assert (
        ExecutionSourceEntry.from_json(fingerprint.execution_sources[0].to_json()) == (fingerprint.execution_sources[0])
    )
    assert (
        fingerprint.to_json()
        == ResumabilityFingerprint(
            starting_commit=commit,
            pipeline_prefix_id=PREFIX_A,
            dependency_versions=dict(reversed(tuple(DEPENDENCIES.items()))),
            entries=tuple(reversed(fingerprint.entries)),
        ).to_json()
    )

    with pytest.raises(TypeError):
        cast("dict[str, str]", fingerprint.dependency_versions)["numpy"] = "changed"
    attribute_name = "starting_commit"
    with pytest.raises(FrozenInstanceError):
        setattr(fingerprint, attribute_name, "0" * 40)


def test_generated_output_never_invalidates_an_explicit_fingerprint(
    source_repository: tuple[Path, str],
) -> None:
    """Files below output_root are absent by construction and cannot self-invalidate."""
    repository, _commit = source_repository
    output_root = repository / "output" / "run"
    before = _capture(source_repository, output_root=output_root)
    output_root.mkdir(parents=True)
    (output_root / "results.jsonl").write_text('{"generated":true}\n', encoding="utf-8")
    (output_root / "checkpoint.npz").write_bytes(b"generated checkpoint")

    after = _capture(source_repository, output_root=output_root)

    assert after == before
    assert after.content_checksum == before.content_checksum


@pytest.mark.parametrize(
    ("relative_path", "expected_category"),
    [
        (Path("src/method.py"), "method_implementation"),
        (Path("uv.lock"), "lockfiles"),
        (Path("config/protocol.json"), "study_protocol"),
    ],
)
def test_each_tracked_input_role_independently_invalidates_resume(
    source_repository: tuple[Path, str],
    relative_path: Path,
    expected_category: str,
) -> None:
    """Working-tree changes are detected even before a new commit exists."""
    repository, _commit = source_repository
    stored = _capture(source_repository)
    path = repository / relative_path
    path.write_bytes(path.read_bytes() + b"changed\n")

    current = _capture(source_repository)
    diagnostics = stored.mismatch_diagnostics(current)

    assert tuple(diagnostics) == (expected_category,)
    assert diagnostics[expected_category][0] != diagnostics[expected_category][1]
    assert current.entries[0].starting_git_blob_id == stored.entries[0].starting_git_blob_id


def test_prefix_dependency_and_starting_commit_mismatches_are_diagnostic(
    source_repository: tuple[Path, str],
) -> None:
    """Non-file resume boundaries receive distinct stable diagnostics."""
    repository, _commit = source_repository
    stored = _capture(source_repository)
    changed_dependencies = _capture(
        source_repository,
        prefix=PREFIX_B,
        dependencies={**DEPENDENCIES, "numpy": "2.4.0"},
    )
    assert stored.mismatch_categories(changed_dependencies) == (
        "pipeline_prefix",
        "dependency_versions",
    )

    (repository / "unrelated.txt").write_text("new commit\n", encoding="utf-8")
    _run_git(repository, "add", "unrelated.txt")
    _run_git(repository, "commit", "--quiet", "-m", "advance start")
    later_commit = _run_git(repository, "rev-parse", "HEAD")
    later_context = (repository, later_commit)
    changed_commit = _capture(later_context)
    assert stored.mismatch_categories(changed_commit) == ("starting_commit",)


def test_resume_requires_an_exact_checksum_sealed_non_scientific_override(
    source_repository: tuple[Path, str],
) -> None:
    """A mismatch is rejected unless its exact pair and reason are recorded."""
    stored = _capture(source_repository)
    current = _capture(
        source_repository,
        dependencies={**DEPENDENCIES, "numpy": "2.4.0"},
    )

    with pytest.raises(ResumabilityMismatchError, match="dependency_versions") as error:
        require_resumability_match(stored, current)
    assert tuple(error.value.diagnostics) == ("dependency_versions",)

    override = NonScientificResumeOverride(
        stored_fingerprint=stored,
        current_fingerprint=current,
        reason="Diagnostic recovery only; excluded from scientific analysis.",
    )
    assert override.classification == "non_scientific"
    assert override.mismatch_categories == ("dependency_versions",)
    assert NonScientificResumeOverride.from_json(override.to_json()) == override
    require_resumability_match(stored, current, override=override)

    unrelated_current = _capture(
        source_repository,
        dependencies={**DEPENDENCIES, "numpy": "2.5.0"},
    )
    with pytest.raises(ValueError, match="does not bind"):
        require_resumability_match(stored, unrelated_current, override=override)
    with pytest.raises(ValueError, match="non-whitespace"):
        NonScientificResumeOverride(
            stored_fingerprint=stored,
            current_fingerprint=current,
            reason="   ",
        )
    with pytest.raises(ValueError, match="at least one"):
        NonScientificResumeOverride(
            stored_fingerprint=stored,
            current_fingerprint=stored,
            reason="Unnecessary override",
        )


def test_pipeline_prefix_mismatch_cannot_be_overridden(
    source_repository: tuple[Path, str],
) -> None:
    """A non-scientific override never authorizes a different pipeline prefix."""
    stored = _capture(source_repository)
    current = _capture(source_repository, prefix=PREFIX_B)

    with pytest.raises(ValueError, match="pipeline-prefix mismatch"):
        NonScientificResumeOverride(
            stored_fingerprint=stored,
            current_fingerprint=current,
            reason="Diagnostic recovery only; excluded from scientific analysis.",
        )
    with pytest.raises(ResumabilityMismatchError, match="cannot be overridden; use a separate artifact store"):
        require_resumability_match(stored, current)


def test_tampered_fingerprint_and_override_are_rejected(
    source_repository: tuple[Path, str],
) -> None:
    """Outer checksums and derived mismatch categories prevent silent edits."""
    stored = _capture(source_repository)
    current = _capture(
        source_repository,
        dependencies={**DEPENDENCIES, "numpy": "2.4.0"},
    )
    fingerprint_data = stored.to_dict()
    fingerprint_data["pipeline_prefix_id"] = PREFIX_B
    with pytest.raises(ValueError, match="content checksum mismatch"):
        ResumabilityFingerprint.from_dict(fingerprint_data)

    override = NonScientificResumeOverride(
        stored_fingerprint=stored,
        current_fingerprint=current,
        reason="Diagnostic recovery only",
    )
    override_data = override.to_dict()
    override_data["reason"] = "Changed reason"
    with pytest.raises(ValueError, match="content checksum mismatch"):
        NonScientificResumeOverride.from_dict(override_data)


def test_capture_rejects_untracked_duplicate_outside_and_output_paths(
    source_repository: tuple[Path, str],
    tmp_path: Path,
) -> None:
    """The explicit manifest cannot admit ambiguous or generated inputs."""
    repository, commit = source_repository
    untracked = repository / "untracked.py"
    untracked.write_text("untracked = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Git could not inspect"):
        capture_resumability_fingerprint(
            repository,
            output_root=repository / "output",
            starting_commit=commit,
            pipeline_prefix_id=PREFIX_A,
            dependency_versions=DEPENDENCIES,
            execution_source_paths=(Path("untracked.py"),),
            lockfile_paths=(Path("uv.lock"),),
            sealed_input_paths=(Path("config/protocol.json"),),
        )

    outside = tmp_path / "outside.py"
    outside.write_text("outside = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside repository_root"):
        capture_resumability_fingerprint(
            repository,
            output_root=repository / "output",
            starting_commit=commit,
            pipeline_prefix_id=PREFIX_A,
            dependency_versions=DEPENDENCIES,
            execution_source_paths=(outside,),
            lockfile_paths=(Path("uv.lock"),),
            sealed_input_paths=(Path("config/protocol.json"),),
        )

    with pytest.raises(ValueError, match="exactly one resumability role"):
        capture_resumability_fingerprint(
            repository,
            output_root=repository / "output",
            starting_commit=commit,
            pipeline_prefix_id=PREFIX_A,
            dependency_versions=DEPENDENCIES,
            execution_source_paths=(Path("src/method.py"),),
            lockfile_paths=(Path("src/method.py"),),
            sealed_input_paths=(Path("config/protocol.json"),),
        )

    tracked_output = repository / "managed" / "sealed.json"
    tracked_output.parent.mkdir()
    tracked_output.write_text("{}\n", encoding="utf-8")
    _run_git(repository, "add", "managed/sealed.json")
    _run_git(repository, "commit", "--quiet", "-m", "track would-be output")
    new_commit = _run_git(repository, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="overlaps output_root"):
        capture_resumability_fingerprint(
            repository,
            output_root=repository / "managed",
            starting_commit=new_commit,
            pipeline_prefix_id=PREFIX_A,
            dependency_versions=DEPENDENCIES,
            execution_source_paths=(Path("src/method.py"),),
            lockfile_paths=(Path("uv.lock"),),
            sealed_input_paths=(Path("managed/sealed.json"),),
        )


def test_direct_record_validation_rejects_missing_roles_and_malformed_identifiers(
    source_repository: tuple[Path, str],
) -> None:
    """Strict constructors reject incomplete manifests and malformed identities."""
    fingerprint = _capture(source_repository)
    with pytest.raises(ValueError, match="every role"):
        replace(fingerprint, entries=fingerprint.execution_sources)
    with pytest.raises(ValueError, match="pipeline_prefix_id"):
        replace(fingerprint, pipeline_prefix_id="short")
    with pytest.raises(ValueError, match="at least one resolved dependency"):
        replace(fingerprint, dependency_versions={})


def test_matching_fingerprints_accept_resume_without_an_override(
    source_repository: tuple[Path, str],
) -> None:
    """An exact fingerprint match is the only scientific resume path."""
    fingerprint = _capture(source_repository)
    require_resumability_match(fingerprint, ResumabilityFingerprint.from_json(fingerprint.to_json()))
    override = NonScientificResumeOverride(
        stored_fingerprint=fingerprint,
        current_fingerprint=_capture(
            source_repository,
            dependencies={**DEPENDENCIES, "numpy": "2.4.0"},
        ),
        reason="Diagnostic recovery only",
    )
    with pytest.raises(ValueError, match="cannot be recorded"):
        require_resumability_match(fingerprint, fingerprint, override=override)
