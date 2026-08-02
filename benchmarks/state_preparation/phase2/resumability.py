# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Explicit, checksum-sealed resumability fingerprints for Phase II runs.

Only caller-enumerated tracked files participate in these fingerprints.  This
keeps generated output from invalidating its own run while making changes to
execution source, lockfiles, and sealed study inputs independently visible.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .validation import (
    require_checksum,
    require_git_blob,
    require_git_commit,
    require_mapping,
    require_nonempty_text,
    require_relative_path,
    require_string,
)

EXECUTION_SOURCE_ENTRY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.execution_source_entry.v1"
RESUMABILITY_FINGERPRINT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.resumability_fingerprint.v1"
NON_SCIENTIFIC_RESUME_OVERRIDE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.non_scientific_resume_override.v1"

SOURCE_ROLES = ("execution_source", "lockfile", "sealed_input")
MISMATCH_CATEGORIES = (
    "starting_commit",
    "pipeline_prefix",
    "dependency_versions",
    "method_implementation",
    "lockfiles",
    "study_protocol",
)

SourceRole = Literal["execution_source", "lockfile", "sealed_input"]
MismatchCategory = Literal[
    "starting_commit",
    "pipeline_prefix",
    "dependency_versions",
    "method_implementation",
    "lockfiles",
    "study_protocol",
]

_PIPELINE_PREFIX_PATTERN = re.compile(r"^phase2_pipeline_prefix_[0-9a-f]{64}$")
_ENTRY_KEYS = frozenset({
    "schema_version",
    "role",
    "repository_path",
    "starting_git_blob_id",
    "content_checksum",
})
_FINGERPRINT_KEYS = frozenset({
    "schema_version",
    "starting_commit",
    "pipeline_prefix_id",
    "dependency_versions",
    "entries",
    "method_implementation_checksum",
    "lockfile_checksum",
    "study_protocol_checksum",
    "tracked_source_manifest_checksum",
    "dependency_versions_checksum",
    "content_checksum",
})
_OVERRIDE_KEYS = frozenset({
    "schema_version",
    "classification",
    "stored_fingerprint",
    "current_fingerprint",
    "mismatch_categories",
    "reason",
    "content_checksum",
})
_ROLE_ORDER = {role: index for index, role in enumerate(SOURCE_ROLES)}


def _sha256(payload: bytes) -> str:
    """Return a prefixed SHA-256 checksum for exact bytes."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _immutable_dependency_versions(value: object) -> Mapping[str, str]:
    """Validate and freeze a nonempty dependency-version mapping.

    Args:
        value: Candidate package-to-version mapping.

    Returns:
        A sorted immutable copy of the mapping.

    Raises:
        TypeError: If the value is not a string mapping.
        ValueError: If the mapping or one of its strings is empty.
    """
    if not isinstance(value, Mapping):
        msg = "dependency_versions must be a mapping."
        raise TypeError(msg)
    if not value:
        msg = "dependency_versions must contain at least one resolved dependency."
        raise ValueError(msg)
    normalized: dict[str, str] = {}
    for raw_name, raw_version in value.items():
        if type(raw_name) is not str or type(raw_version) is not str:
            msg = "dependency_versions must contain only string keys and values."
            raise TypeError(msg)
        name = require_string(raw_name, "dependency_versions key")
        version = require_string(raw_version, f"dependency_versions[{name!r}]")
        normalized[name] = version
    return MappingProxyType(dict(sorted(normalized.items())))


def _require_pipeline_prefix(value: object) -> str:
    """Validate one complete Phase II pipeline-prefix identifier.

    Returns:
        The validated prefix.

    Raises:
        ValueError: If the prefix does not use the Phase II format.
    """
    prefix = require_string(value, "pipeline_prefix_id")
    if _PIPELINE_PREFIX_PATTERN.fullmatch(prefix) is None:
        msg = "pipeline_prefix_id must be 'phase2_pipeline_prefix_' followed by 64 lowercase hexadecimal digits."
        raise ValueError(msg)
    return prefix


@dataclass(frozen=True, slots=True)
class ExecutionSourceEntry:
    """One tracked file participating in a Phase II resume decision.

    ``starting_git_blob_id`` records the file at the starting commit, while
    ``content_checksum`` identifies the exact working-tree bytes used by the
    execution.  Consequently a dirty tracked source remains representable but
    produces a distinct fingerprint.
    """

    role: SourceRole
    repository_path: str
    starting_git_blob_id: str
    content_checksum: str
    schema_version: str = field(default=EXECUTION_SOURCE_ENTRY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the role, normalized path, Git blob, and byte checksum.

        Raises:
            ValueError: If a field is malformed or the role is unsupported.
        """
        if self.role not in SOURCE_ROLES:
            msg = f"role must be one of {SOURCE_ROLES!r}."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "repository_path",
            require_relative_path(self.repository_path, "repository_path"),
        )
        object.__setattr__(
            self,
            "starting_git_blob_id",
            require_git_blob(self.starting_git_blob_id, "starting_git_blob_id"),
        )
        object.__setattr__(
            self,
            "content_checksum",
            require_checksum(self.content_checksum, "content_checksum"),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered entry fields."""
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "repository_path": self.repository_path,
            "starting_git_blob_id": self.starting_git_blob_id,
        }

    @property
    def record_checksum(self) -> str:
        """Checksum of the entry metadata and exact current file content."""
        return canonical_checksum({
            **self._content_dict(),
            "content_checksum": self.content_checksum,
        })

    def to_dict(self) -> dict[str, object]:
        """Return the exact canonical entry representation.

        The ``content_checksum`` field identifies the referenced file bytes;
        the enclosing fingerprint seals this entry's metadata as well.
        """
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> ExecutionSourceEntry:
        """Construct a strict execution-source entry from serialized data.

        Returns:
            The validated entry.

        Raises:
            ValueError: If fields or values are invalid.
        """
        mapping = require_mapping(data, "execution source entry")
        if frozenset(mapping) != _ENTRY_KEYS:
            missing = sorted(_ENTRY_KEYS - frozenset(mapping))
            extra = sorted(frozenset(mapping) - _ENTRY_KEYS)
            msg = f"execution source entry fields do not match the schema: missing={missing!r}, extra={extra!r}."
            raise ValueError(msg)
        if mapping["schema_version"] != EXECUTION_SOURCE_ENTRY_SCHEMA_VERSION:
            msg = f"schema_version must be {EXECUTION_SOURCE_ENTRY_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        return cls(
            role=cast("SourceRole", mapping["role"]),
            repository_path=cast("str", mapping["repository_path"]),
            starting_git_blob_id=cast("str", mapping["starting_git_blob_id"]),
            content_checksum=cast("str", mapping["content_checksum"]),
        )

    def to_json(self) -> str:
        """Return canonical JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> ExecutionSourceEntry:
        """Construct an entry from canonical JSON text.

        Returns:
            The validated entry.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class ResumabilityFingerprint:
    """Explicit scientific context that must match before a pipeline resumes."""

    starting_commit: str
    pipeline_prefix_id: str
    dependency_versions: Mapping[str, str]
    entries: tuple[ExecutionSourceEntry, ...]
    schema_version: str = field(default=RESUMABILITY_FINGERPRINT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate and canonically order the complete explicit manifest.

        Raises:
            TypeError: If entries or dependency versions have invalid types.
            ValueError: If an identity is invalid or a manifest role is absent.
        """
        object.__setattr__(self, "starting_commit", require_git_commit(self.starting_commit, "starting_commit"))
        object.__setattr__(self, "pipeline_prefix_id", _require_pipeline_prefix(self.pipeline_prefix_id))
        object.__setattr__(
            self,
            "dependency_versions",
            _immutable_dependency_versions(self.dependency_versions),
        )
        if isinstance(self.entries, (str, bytes)) or not isinstance(self.entries, Sequence):
            msg = "entries must be a sequence of ExecutionSourceEntry records."
            raise TypeError(msg)
        entries = tuple(self.entries)
        if not entries or not all(isinstance(entry, ExecutionSourceEntry) for entry in entries):
            msg = "entries must contain only ExecutionSourceEntry records."
            raise TypeError(msg)
        paths = tuple(entry.repository_path for entry in entries)
        if len(paths) != len(set(paths)):
            msg = "A repository path may appear in exactly one resumability role."
            raise ValueError(msg)
        missing_roles = tuple(role for role in SOURCE_ROLES if not any(entry.role == role for entry in entries))
        if missing_roles:
            msg = f"entries must contain at least one file for every role; missing {missing_roles!r}."
            raise ValueError(msg)
        ordered = tuple(sorted(entries, key=lambda entry: (_ROLE_ORDER[entry.role], entry.repository_path)))
        object.__setattr__(self, "entries", ordered)

    def _role_entries(self, role: SourceRole) -> tuple[ExecutionSourceEntry, ...]:
        """Return entries belonging to one manifest role."""
        return tuple(entry for entry in self.entries if entry.role == role)

    @property
    def execution_sources(self) -> tuple[ExecutionSourceEntry, ...]:
        """Tracked method implementation files in canonical path order."""
        return self._role_entries("execution_source")

    @property
    def lockfiles(self) -> tuple[ExecutionSourceEntry, ...]:
        """Tracked dependency lockfiles in canonical path order."""
        return self._role_entries("lockfile")

    @property
    def sealed_inputs(self) -> tuple[ExecutionSourceEntry, ...]:
        """Tracked sealed protocol/configuration files in canonical path order."""
        return self._role_entries("sealed_input")

    @staticmethod
    def _entry_group_checksum(entries: Sequence[ExecutionSourceEntry]) -> str:
        """Checksum one canonically ordered group of source entries.

        Returns:
            The group checksum.
        """
        return canonical_checksum([entry.to_dict() for entry in entries])

    @property
    def method_implementation_checksum(self) -> str:
        """Checksum of only the tracked executable implementation files."""
        return self._entry_group_checksum(self.execution_sources)

    @property
    def lockfile_checksum(self) -> str:
        """Checksum of only the dependency lockfiles."""
        return self._entry_group_checksum(self.lockfiles)

    @property
    def study_protocol_checksum(self) -> str:
        """Checksum of only the sealed configuration and protocol inputs."""
        return self._entry_group_checksum(self.sealed_inputs)

    @property
    def tracked_source_manifest_checksum(self) -> str:
        """Checksum of every explicitly tracked source-manifest entry."""
        return self._entry_group_checksum(self.entries)

    @property
    def dependency_versions_checksum(self) -> str:
        """Checksum of the sorted resolved dependency versions."""
        return canonical_checksum(dict(self.dependency_versions))

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered fingerprint content."""
        return {
            "schema_version": self.schema_version,
            "starting_commit": self.starting_commit,
            "pipeline_prefix_id": self.pipeline_prefix_id,
            "dependency_versions": dict(self.dependency_versions),
            "entries": [entry.to_dict() for entry in self.entries],
            "method_implementation_checksum": self.method_implementation_checksum,
            "lockfile_checksum": self.lockfile_checksum,
            "study_protocol_checksum": self.study_protocol_checksum,
            "tracked_source_manifest_checksum": self.tracked_source_manifest_checksum,
            "dependency_versions_checksum": self.dependency_versions_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete resume decision context."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed canonical fingerprint."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> ResumabilityFingerprint:
        """Construct and checksum-verify a strict fingerprint.

        Returns:
            The validated fingerprint.

        Raises:
            TypeError: If a serialized collection has the wrong type.
            ValueError: If the document or a derived checksum is invalid.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_FINGERPRINT_KEYS,
            name="resumability fingerprint",
        )
        if mapping["schema_version"] != RESUMABILITY_FINGERPRINT_SCHEMA_VERSION:
            msg = f"schema_version must be {RESUMABILITY_FINGERPRINT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        raw_entries = mapping["entries"]
        if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, Sequence):
            msg = "entries must be a sequence."
            raise TypeError(msg)
        result = cls(
            starting_commit=cast("str", mapping["starting_commit"]),
            pipeline_prefix_id=cast("str", mapping["pipeline_prefix_id"]),
            dependency_versions=cast("Mapping[str, str]", mapping["dependency_versions"]),
            entries=tuple(ExecutionSourceEntry.from_dict(entry) for entry in raw_entries),
        )
        derived = {
            "method_implementation_checksum": result.method_implementation_checksum,
            "lockfile_checksum": result.lockfile_checksum,
            "study_protocol_checksum": result.study_protocol_checksum,
            "tracked_source_manifest_checksum": result.tracked_source_manifest_checksum,
            "dependency_versions_checksum": result.dependency_versions_checksum,
            "content_checksum": result.content_checksum,
        }
        for name, expected in derived.items():
            if mapping[name] != expected:
                msg = f"Serialized {name} does not match the reconstructed resumability fingerprint."
                raise ValueError(msg)
        return result

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> ResumabilityFingerprint:
        """Construct a fingerprint from canonical JSON text.

        Returns:
            The validated fingerprint.
        """
        return cls.from_dict(load_canonical_json_object(payload))

    def mismatch_diagnostics(self, current: ResumabilityFingerprint) -> Mapping[str, tuple[str, str]]:
        """Compare a stored fingerprint with the current execution context.

        Args:
            current: Newly captured context proposed for resume.

        Returns:
            Immutable category-to-``(stored, current)`` diagnostics in the
            canonical mismatch-category order.

        Raises:
            TypeError: If ``current`` is not a fingerprint.
        """
        if not isinstance(current, ResumabilityFingerprint):
            msg = "current must be a ResumabilityFingerprint."
            raise TypeError(msg)
        candidates = {
            "starting_commit": (self.starting_commit, current.starting_commit),
            "pipeline_prefix": (self.pipeline_prefix_id, current.pipeline_prefix_id),
            "dependency_versions": (
                self.dependency_versions_checksum,
                current.dependency_versions_checksum,
            ),
            "method_implementation": (
                self.method_implementation_checksum,
                current.method_implementation_checksum,
            ),
            "lockfiles": (self.lockfile_checksum, current.lockfile_checksum),
            "study_protocol": (self.study_protocol_checksum, current.study_protocol_checksum),
        }
        return MappingProxyType({
            category: candidates[category]
            for category in MISMATCH_CATEGORIES
            if candidates[category][0] != candidates[category][1]
        })

    def mismatch_categories(self, current: ResumabilityFingerprint) -> tuple[str, ...]:
        """Return only the ordered mismatch category names."""
        return tuple(self.mismatch_diagnostics(current))


class ResumabilityMismatchError(ValueError):
    """Raised when a resume context differs without a recorded override."""

    def __init__(self, diagnostics: Mapping[str, tuple[str, str]]) -> None:
        """Store immutable diagnostics and build an actionable error message."""
        self.diagnostics = MappingProxyType(dict(diagnostics))
        categories = ", ".join(self.diagnostics)
        if "pipeline_prefix" in self.diagnostics:
            guidance = "Pipeline-prefix drift cannot be overridden; use a separate artifact store."
        else:
            guidance = "Create an explicit NonScientificResumeOverride to continue as non-scientific."
        super().__init__(f"Resume fingerprint mismatch in: {categories}. {guidance}")


@dataclass(frozen=True, slots=True)
class NonScientificResumeOverride:
    """Checksum-sealed acknowledgement of runtime drift within one pipeline."""

    stored_fingerprint: ResumabilityFingerprint
    current_fingerprint: ResumabilityFingerprint
    reason: str
    mismatch_categories: tuple[str, ...] = field(init=False)
    classification: str = field(default="non_scientific", init=False)
    schema_version: str = field(default=NON_SCIENTIFIC_RESUME_OVERRIDE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate both fingerprints and derive the mismatch declaration.

        Raises:
            TypeError: If either embedded record is not a fingerprint.
            ValueError: If the reason is invalid or the fingerprints match.
        """
        if not isinstance(self.stored_fingerprint, ResumabilityFingerprint):
            msg = "stored_fingerprint must be a ResumabilityFingerprint."
            raise TypeError(msg)
        if not isinstance(self.current_fingerprint, ResumabilityFingerprint):
            msg = "current_fingerprint must be a ResumabilityFingerprint."
            raise TypeError(msg)
        reason = require_nonempty_text(self.reason, "reason")
        if reason != reason.strip() or any(character.isspace() and character != " " for character in reason):
            msg = "reason must not contain surrounding or control whitespace."
            raise ValueError(msg)
        categories = self.stored_fingerprint.mismatch_categories(self.current_fingerprint)
        if not categories:
            msg = "A non-scientific override requires at least one fingerprint mismatch."
            raise ValueError(msg)
        if "pipeline_prefix" in categories:
            msg = "A pipeline-prefix mismatch cannot be authorized by a non-scientific override."
            raise ValueError(msg)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "mismatch_categories", categories)

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered override content."""
        return {
            "schema_version": self.schema_version,
            "classification": self.classification,
            "stored_fingerprint": self.stored_fingerprint.to_dict(),
            "current_fingerprint": self.current_fingerprint.to_dict(),
            "mismatch_categories": list(self.mismatch_categories),
            "reason": self.reason,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete mismatch acknowledgement."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed override record."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> NonScientificResumeOverride:
        """Construct and checksum-verify an override record.

        Returns:
            The validated override.

        Raises:
            TypeError: If a serialized collection has the wrong type.
            ValueError: If content or derived mismatch categories are invalid.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_OVERRIDE_KEYS,
            name="non-scientific resume override",
        )
        if mapping["schema_version"] != NON_SCIENTIFIC_RESUME_OVERRIDE_SCHEMA_VERSION:
            msg = f"schema_version must be {NON_SCIENTIFIC_RESUME_OVERRIDE_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        if mapping["classification"] != "non_scientific":
            msg = "classification must be 'non_scientific'."
            raise ValueError(msg)
        override = cls(
            stored_fingerprint=ResumabilityFingerprint.from_dict(mapping["stored_fingerprint"]),
            current_fingerprint=ResumabilityFingerprint.from_dict(mapping["current_fingerprint"]),
            reason=cast("str", mapping["reason"]),
        )
        serialized_categories = mapping["mismatch_categories"]
        if isinstance(serialized_categories, (str, bytes)) or not isinstance(serialized_categories, Sequence):
            msg = "mismatch_categories must be a sequence."
            raise TypeError(msg)
        if tuple(serialized_categories) != override.mismatch_categories:
            msg = "mismatch_categories do not match the embedded fingerprints."
            raise ValueError(msg)
        if mapping["content_checksum"] != override.content_checksum:
            msg = "Non-scientific resume override checksum changed during normalization."
            raise ValueError(msg)
        return override

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> NonScientificResumeOverride:
        """Construct an override from canonical JSON text.

        Returns:
            The validated override.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _run_git(repository_root: Path, *arguments: str) -> bytes:
    """Run a read-only Git inspection command and return stdout.

    Returns:
        Exact command stdout.

    Raises:
        ValueError: If Git is unavailable or cannot inspect the repository.
    """
    executable = shutil.which("git")
    if executable is None:
        msg = "Could not capture resumability provenance because Git was not found."
        raise ValueError(msg)
    try:
        completed = subprocess.run(  # ruff: ignore[S603] -- resolved executable; no shell interpretation
            (executable, "-C", os.fspath(repository_root), *arguments),
            check=True,
            capture_output=True,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        msg = f"Git could not inspect resumability provenance in {repository_root}: {error}."
        raise ValueError(msg) from error
    return completed.stdout


def _require_path_sequence(value: object, name: str) -> tuple[Path, ...]:
    """Validate a nonempty sequence containing only ``Path`` objects.

    Returns:
        The validated path tuple.

    Raises:
        TypeError: If the value is not a path sequence.
        ValueError: If the sequence is empty.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of pathlib.Path values."
        raise TypeError(msg)
    paths = tuple(value)
    if not paths:
        msg = f"{name} must contain at least one path."
        raise ValueError(msg)
    if not all(isinstance(path, Path) for path in paths):
        msg = f"{name} must contain only pathlib.Path values."
        raise TypeError(msg)
    return cast("tuple[Path, ...]", paths)


def _resolve_source_path(
    repository_root: Path,
    output_root: Path,
    path: Path,
    *,
    name: str,
) -> tuple[Path, str]:
    """Resolve one source path and reject escapes, symlinks, and output overlap.

    Returns:
        The resolved path and its repository-relative POSIX spelling.

    Raises:
        ValueError: If the path is invalid, unsafe, missing, or generated.
    """
    candidate = path if path.is_absolute() else repository_root / path
    if candidate.is_symlink():
        msg = f"{name} must not be a symbolic link."
        raise ValueError(msg)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(repository_root):
        msg = f"{name} must resolve inside repository_root."
        raise ValueError(msg)
    if resolved == output_root or resolved.is_relative_to(output_root):
        msg = f"{name} overlaps output_root and must be excluded from the resumability manifest."
        raise ValueError(msg)
    if not resolved.is_file():
        msg = f"{name} must identify an existing regular file."
        raise ValueError(msg)
    relative = require_relative_path(resolved.relative_to(repository_root).as_posix(), name)
    return resolved, relative


def _starting_blob_id(repository_root: Path, starting_commit: str, relative_path: str) -> str:
    """Return a regular-file blob ID from the exact starting commit.

    Returns:
        The complete lowercase Git blob identifier.

    Raises:
        ValueError: If the path is absent or not a regular tracked file.
    """
    listing = _run_git(repository_root, "ls-tree", "-z", starting_commit, "--", relative_path)
    records = tuple(record for record in listing.split(b"\0") if record)
    if len(records) != 1:
        msg = f"Source path {relative_path!r} is not tracked exactly once at the starting commit."
        raise ValueError(msg)
    try:
        header, encoded_path = records[0].split(b"\t", 1)
        mode, object_type, blob_id = header.split(b" ", 2)
        listed_path = os.fsdecode(encoded_path)
    except ValueError as error:
        msg = f"Git returned malformed tree metadata for {relative_path!r}."
        raise ValueError(msg) from error
    if mode not in {b"100644", b"100755"} or object_type != b"blob" or listed_path != relative_path:
        msg = f"Source path {relative_path!r} must be a regular tracked file at the starting commit."
        raise ValueError(msg)
    return require_git_blob(blob_id.decode("ascii"), f"starting blob for {relative_path}")


def _capture_entries(
    repository_root: Path,
    output_root: Path,
    starting_commit: str,
    role: SourceRole,
    paths: Sequence[Path],
) -> tuple[ExecutionSourceEntry, ...]:
    """Capture exact bytes and starting blobs for one explicit file role.

    Returns:
        The captured source entries.

    Raises:
        ValueError: If a path cannot be verified or read.
    """
    entries: list[ExecutionSourceEntry] = []
    for index, path in enumerate(paths):
        resolved, relative = _resolve_source_path(
            repository_root,
            output_root,
            path,
            name=f"{role}_paths[{index}]",
        )
        _run_git(repository_root, "ls-files", "--error-unmatch", "--", relative)
        try:
            payload = resolved.read_bytes()
        except OSError as error:
            msg = f"Could not read resumability source {relative!r}: {error}."
            raise ValueError(msg) from error
        entries.append(
            ExecutionSourceEntry(
                role=role,
                repository_path=relative,
                starting_git_blob_id=_starting_blob_id(repository_root, starting_commit, relative),
                content_checksum=_sha256(payload),
            )
        )
    return tuple(entries)


def capture_resumability_fingerprint(
    repository_root: Path,
    *,
    output_root: Path,
    starting_commit: str,
    pipeline_prefix_id: str,
    dependency_versions: Mapping[str, str],
    execution_source_paths: Sequence[Path],
    lockfile_paths: Sequence[Path],
    sealed_input_paths: Sequence[Path],
) -> ResumabilityFingerprint:
    """Capture one explicit, output-independent Phase II resume fingerprint.

    Args:
        repository_root: Root of the Git worktree used for execution.
        output_root: Configured root for generated run artifacts. Manifest
            entries at or below this path are rejected.
        starting_commit: Exact commit at which the run began.
        pipeline_prefix_id: Stable configured stage-prefix identity.
        dependency_versions: Resolved package/runtime versions.
        execution_source_paths: Explicit tracked implementation files.
        lockfile_paths: Explicit tracked dependency lockfiles.
        sealed_input_paths: Explicit tracked study/configuration inputs.

    Returns:
        A checksum-sealed immutable fingerprint.

    Raises:
        TypeError: If paths or mappings have unsupported types.
        ValueError: If Git provenance, source tracking, or output isolation is
            invalid.
    """
    if not isinstance(repository_root, Path):
        msg = "repository_root must be a pathlib.Path."
        raise TypeError(msg)
    if not isinstance(output_root, Path):
        msg = "output_root must be a pathlib.Path."
        raise TypeError(msg)
    root = repository_root.resolve()
    if not root.is_dir():
        msg = "repository_root must be an existing directory."
        raise ValueError(msg)
    output = output_root.resolve()
    commit = require_git_commit(starting_commit, "starting_commit")
    resolved_commit = _run_git(root, "rev-parse", "--verify", f"{commit}^{{commit}}").decode("ascii").strip()
    if resolved_commit != commit:
        msg = "starting_commit does not resolve to the supplied exact commit."
        raise ValueError(msg)
    prefix = _require_pipeline_prefix(pipeline_prefix_id)
    sources = _require_path_sequence(execution_source_paths, "execution_source_paths")
    lockfiles = _require_path_sequence(lockfile_paths, "lockfile_paths")
    inputs = _require_path_sequence(sealed_input_paths, "sealed_input_paths")
    entries = (
        *_capture_entries(root, output, commit, "execution_source", sources),
        *_capture_entries(root, output, commit, "lockfile", lockfiles),
        *_capture_entries(root, output, commit, "sealed_input", inputs),
    )
    return ResumabilityFingerprint(
        starting_commit=commit,
        pipeline_prefix_id=prefix,
        dependency_versions=dependency_versions,
        entries=entries,
    )


def require_resumability_match(
    stored: ResumabilityFingerprint,
    current: ResumabilityFingerprint,
    *,
    override: NonScientificResumeOverride | None = None,
) -> None:
    """Require matching resume provenance or its exact recorded override.

    Args:
        stored: Fingerprint persisted with completed work.
        current: Newly captured execution context.
        override: Optional checksum-sealed non-scientific acknowledgement.

    Raises:
        TypeError: If a record has the wrong type.
        ValueError: If an override is unnecessary or belongs to another pair.
        ResumabilityMismatchError: If fingerprints differ without an override.
    """
    if not isinstance(stored, ResumabilityFingerprint) or not isinstance(current, ResumabilityFingerprint):
        msg = "stored and current must be ResumabilityFingerprint records."
        raise TypeError(msg)
    diagnostics = stored.mismatch_diagnostics(current)
    if not diagnostics:
        if override is not None:
            msg = "A non-scientific override cannot be recorded for matching fingerprints."
            raise ValueError(msg)
        return
    if "pipeline_prefix" in diagnostics:
        raise ResumabilityMismatchError(diagnostics)
    if override is None:
        raise ResumabilityMismatchError(diagnostics)
    if not isinstance(override, NonScientificResumeOverride):
        msg = "override must be a NonScientificResumeOverride or None."
        raise TypeError(msg)
    if override.stored_fingerprint != stored or override.current_fingerprint != current:
        msg = "The non-scientific override does not bind the stored and current fingerprints."
        raise ValueError(msg)


__all__ = [
    "EXECUTION_SOURCE_ENTRY_SCHEMA_VERSION",
    "MISMATCH_CATEGORIES",
    "NON_SCIENTIFIC_RESUME_OVERRIDE_SCHEMA_VERSION",
    "RESUMABILITY_FINGERPRINT_SCHEMA_VERSION",
    "SOURCE_ROLES",
    "ExecutionSourceEntry",
    "NonScientificResumeOverride",
    "ResumabilityFingerprint",
    "ResumabilityMismatchError",
    "capture_resumability_fingerprint",
    "require_resumability_match",
]
