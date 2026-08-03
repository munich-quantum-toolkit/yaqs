# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Clean-checkout source locking for the WP22 final confirmation seal.

The final confirmation seal must address the exact execution, analysis,
dependency, and scientific-input bytes that will be used after unblinding.
Capture therefore fails closed unless the supplied repository is an exactly
clean Git checkout.  Cleanliness is checked before any governed file is read
and again after capture so that neither modified nor untracked inputs can be
silently omitted from the frozen source universe.
"""

from __future__ import annotations

import hashlib
import operator
import shutil
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .protocol import (
    AnalysisSourceFileRef,
    AnalysisSourceManifest,
    FinalConfirmationSeal,
    verify_analysis_source_files,
)
from .validation import (
    require_bool,
    require_checksum,
    require_git_blob,
    require_git_commit,
    require_relative_path,
    require_slug,
)

EXECUTION_SOURCE_FILE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.execution_source_file.v1"
EXECUTION_SOURCE_MANIFEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.execution_source_manifest.v1"
EXECUTION_SOURCE_ROLES = (
    "execution_source",
    "analysis_source",
    "dependency_lock",
    "sealed_input",
)

WP22_GOVERNED_ENTRY_POINT = "benchmarks/state_preparation/training_runner.py"
WP22_GOVERNED_ANALYSIS_ENTRY_POINT = "benchmarks/state_preparation/phase2/primary_analysis.py"
WP22_GOVERNED_DEPENDENCY_LOCK_PATHS = ("pyproject.toml", "uv.lock")
WP22_GOVERNED_PREREGISTRATION_PATH = "benchmarks/state_preparation/phase2/data/initial_preregistration_v1.json"
WP22_GOVERNED_REQUIRED_PATHS = (
    WP22_GOVERNED_ENTRY_POINT,
    WP22_GOVERNED_ANALYSIS_ENTRY_POINT,
    "src/mqt/yaqs/__init__.py",
    *WP22_GOVERNED_DEPENDENCY_LOCK_PATHS,
    WP22_GOVERNED_PREREGISTRATION_PATH,
)

ExecutionSourceRole = Literal[
    "execution_source",
    "analysis_source",
    "dependency_lock",
    "sealed_input",
]

_FILE_KEYS = frozenset({
    "schema_version",
    "role",
    "repo_path",
    "git_blob_id",
    "source_checksum",
    "content_checksum",
})
_MANIFEST_KEYS = frozenset({
    "schema_version",
    "manifest_id",
    "source_commit",
    "entry_point",
    "source_files",
    "environment_lock_checksum",
    "tracked_source_manifest_checksum",
    "clean_worktree",
    "content_checksum",
})


def _sha256(payload: bytes) -> str:
    """Return the prefixed SHA-256 checksum of exact bytes."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ExecutionSourceFileRef:
    """One role-typed, commit-addressed, checksum-sealed source file."""

    role: ExecutionSourceRole
    repo_path: str
    git_blob_id: str
    source_checksum: str
    schema_version: str = field(default=EXECUTION_SOURCE_FILE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the role, normalized path, Git blob, and byte digest.

        Raises:
            ValueError: If a role or immutable source identity is invalid.
        """
        if self.role not in EXECUTION_SOURCE_ROLES:
            msg = f"role must be one of {EXECUTION_SOURCE_ROLES!r}."
            raise ValueError(msg)
        object.__setattr__(self, "repo_path", require_relative_path(self.repo_path, "repo_path"))
        object.__setattr__(self, "git_blob_id", require_git_blob(self.git_blob_id, "git_blob_id"))
        object.__setattr__(
            self,
            "source_checksum",
            require_checksum(self.source_checksum, "source_checksum"),
        )

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete role-typed source reference."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered source-reference field."""
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "repo_path": self.repo_path,
            "git_blob_id": self.git_blob_id,
            "source_checksum": self.source_checksum,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the sealed JSON-native source reference."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ExecutionSourceFileRef:
        """Construct and checksum-verify one source reference.

        Args:
            data: Exact sealed source-reference document.

        Returns:
            The validated immutable source reference.

        Raises:
            ValueError: If fields, schema, or checksum differ.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_FILE_KEYS, name="execution source file")
        if mapping["schema_version"] != EXECUTION_SOURCE_FILE_SCHEMA_VERSION:
            msg = f"schema_version must be {EXECUTION_SOURCE_FILE_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        source_file = cls(
            role=cast("ExecutionSourceRole", mapping["role"]),
            repo_path=cast("str", mapping["repo_path"]),
            git_blob_id=cast("str", mapping["git_blob_id"]),
            source_checksum=cast("str", mapping["source_checksum"]),
        )
        supplied = cast("str", mapping["content_checksum"])
        if source_file.content_checksum != supplied:
            msg = f"Execution-source file checksum changed during normalization: expected {supplied}."
            raise ValueError(msg)
        return source_file

    @classmethod
    def from_json(cls, payload: str) -> ExecutionSourceFileRef:
        """Construct a source reference from canonical JSON.

        Args:
            payload: Canonical sealed JSON text.

        Returns:
            The validated immutable source reference.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _tracked_manifest_checksum(source_files: Sequence[ExecutionSourceFileRef]) -> str:
    """Return the separately addressable ordered source-inventory checksum."""
    return canonical_checksum({"source_files": [source_file.to_dict() for source_file in source_files]})


def _environment_lock_checksum(source_files: Sequence[ExecutionSourceFileRef]) -> str:
    """Return the separately addressable dependency-lock inventory checksum."""
    dependency_locks = [source_file.to_dict() for source_file in source_files if source_file.role == "dependency_lock"]
    return canonical_checksum({"dependency_locks": dependency_locks})


@dataclass(frozen=True, slots=True)
class ExecutionSourceManifest:
    """Immutable inventory captured from one exactly clean Git checkout."""

    manifest_id: str
    source_commit: str
    entry_point: str
    source_files: tuple[ExecutionSourceFileRef, ...]
    environment_lock_checksum: str
    tracked_source_manifest_checksum: str
    clean_worktree: bool = field(default=True)
    schema_version: str = field(default=EXECUTION_SOURCE_MANIFEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate completeness, deterministic order, and derived checksums.

        Raises:
            TypeError: If source files have unsupported record types.
            ValueError: If roles, paths, entry point, or checksums are inconsistent.
        """
        object.__setattr__(self, "manifest_id", require_slug(self.manifest_id, "manifest_id"))
        object.__setattr__(self, "source_commit", require_git_commit(self.source_commit, "source_commit"))
        object.__setattr__(self, "entry_point", require_relative_path(self.entry_point, "entry_point"))
        source_files = tuple(self.source_files)
        if not source_files or not all(isinstance(item, ExecutionSourceFileRef) for item in source_files):
            msg = "source_files must contain ExecutionSourceFileRef values."
            raise TypeError(msg)
        paths = tuple(source_file.repo_path for source_file in source_files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            msg = "source_files must have unique paths in lexical order."
            raise ValueError(msg)
        roles = {source_file.role for source_file in source_files}
        if roles != set(EXECUTION_SOURCE_ROLES):
            missing = sorted(set(EXECUTION_SOURCE_ROLES) - roles)
            msg = f"source_files must cover every final-lock role; missing={missing!r}."
            raise ValueError(msg)
        entry_matches = [
            source_file
            for source_file in source_files
            if source_file.repo_path == self.entry_point and source_file.role == "execution_source"
        ]
        if len(entry_matches) != 1:
            msg = "entry_point must identify exactly one execution_source file."
            raise ValueError(msg)
        object.__setattr__(self, "source_files", source_files)
        environment_checksum = require_checksum(
            self.environment_lock_checksum,
            "environment_lock_checksum",
        )
        expected_environment = _environment_lock_checksum(source_files)
        if environment_checksum != expected_environment:
            msg = "environment_lock_checksum is not derived from the exact dependency-lock inventory."
            raise ValueError(msg)
        object.__setattr__(self, "environment_lock_checksum", environment_checksum)
        manifest_checksum = require_checksum(
            self.tracked_source_manifest_checksum,
            "tracked_source_manifest_checksum",
        )
        expected_manifest = _tracked_manifest_checksum(source_files)
        if manifest_checksum != expected_manifest:
            msg = "tracked_source_manifest_checksum is not derived from the exact ordered source inventory."
            raise ValueError(msg)
        object.__setattr__(self, "tracked_source_manifest_checksum", manifest_checksum)
        if require_bool(self.clean_worktree, "clean_worktree") is not True:
            msg = "Execution sources must be captured from an exactly clean worktree."
            raise ValueError(msg)

    @property
    def content_checksum(self) -> str:
        """Checksum used by ``FinalConfirmationSeal.execution_source_checksum``."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered execution-source field."""
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "source_commit": self.source_commit,
            "entry_point": self.entry_point,
            "source_files": [source_file.to_dict() for source_file in self.source_files],
            "environment_lock_checksum": self.environment_lock_checksum,
            "tracked_source_manifest_checksum": self.tracked_source_manifest_checksum,
            "clean_worktree": self.clean_worktree,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the sealed JSON-native execution-source manifest."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ExecutionSourceManifest:
        """Construct and checksum-verify an execution-source manifest.

        Args:
            data: Exact sealed manifest document.

        Returns:
            The validated immutable manifest.

        Raises:
            ValueError: If fields, schema, derived identities, or checksum differ.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_MANIFEST_KEYS, name="execution source manifest")
        if mapping["schema_version"] != EXECUTION_SOURCE_MANIFEST_SCHEMA_VERSION:
            msg = f"schema_version must be {EXECUTION_SOURCE_MANIFEST_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        files = cast("Sequence[object]", mapping["source_files"])
        manifest = cls(
            manifest_id=cast("str", mapping["manifest_id"]),
            source_commit=cast("str", mapping["source_commit"]),
            entry_point=cast("str", mapping["entry_point"]),
            source_files=tuple(ExecutionSourceFileRef.from_dict(item) for item in files),
            environment_lock_checksum=cast("str", mapping["environment_lock_checksum"]),
            tracked_source_manifest_checksum=cast("str", mapping["tracked_source_manifest_checksum"]),
            clean_worktree=cast("bool", mapping["clean_worktree"]),
        )
        supplied = cast("str", mapping["content_checksum"])
        if manifest.content_checksum != supplied:
            msg = f"Execution-source manifest checksum changed during normalization: expected {supplied}."
            raise ValueError(msg)
        return manifest

    @classmethod
    def from_json(cls, payload: str) -> ExecutionSourceManifest:
        """Construct an execution-source manifest from canonical JSON.

        Args:
            payload: Canonical sealed JSON text.

        Returns:
            The validated immutable manifest.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _git(repository_root: Path, *arguments: str) -> bytes:
    """Run one non-shell Git command and return exact stdout bytes.

    Returns:
        Exact standard-output bytes.

    Raises:
        ValueError: If Git is unavailable or the command fails.
    """
    executable = shutil.which("git")
    if executable is None:
        msg = "Git is required to capture final execution sources."
        raise ValueError(msg)
    completed = subprocess.run(  # noqa: S603 - executable is resolved with shutil.which and shell remains disabled
        (executable, "-C", str(repository_root), *arguments),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip() or completed.stdout.decode(errors="replace").strip()
        msg = f"Git command {arguments!r} failed: {detail}."
        raise ValueError(msg)
    return completed.stdout


def _require_clean_checkout(repository_root: Path) -> str:
    """Require an exact clean repository root and return its HEAD commit.

    This check intentionally runs before any governed source file is opened.

    Returns:
        Complete HEAD commit identifier.

    Raises:
        TypeError: If ``repository_root`` is not a path.
        ValueError: If the path is not the clean root of a committed checkout.
    """
    if not isinstance(repository_root, Path):
        msg = f"repository_root must be a pathlib.Path, got {type(repository_root).__name__}."
        raise TypeError(msg)
    root = repository_root.resolve()
    if not root.is_dir():
        msg = "repository_root must identify an existing directory."
        raise ValueError(msg)
    top_level = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top_level != root:
        msg = "repository_root must be the exact top level of the Git checkout."
        raise ValueError(msg)
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status:
        msg = "Final execution sources require an exactly clean worktree with no modified or untracked files."
        raise ValueError(msg)
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    return require_git_commit(commit, "source_commit")


def _path_text(value: str | Path, name: str) -> str:
    """Normalize a caller-supplied repository-relative path spelling.

    Returns:
        Normalized repository-relative POSIX path.
    """
    if isinstance(value, Path):
        value = value.as_posix()
    return require_relative_path(value, name)


def _capture_source_file(
    repository_root: Path,
    source_commit: str,
    role: ExecutionSourceRole,
    repo_path: str,
) -> ExecutionSourceFileRef:
    """Capture one tracked regular file after the checkout passed cleanliness.

    Returns:
        Exact role-typed source reference.

    Raises:
        ValueError: If the path is untracked, non-regular, symlinked, or differs from HEAD.
    """
    tree_output = _git(repository_root, "ls-tree", "-z", source_commit, "--", repo_path)
    entries = tuple(item for item in tree_output.split(b"\0") if item)
    if len(entries) != 1:
        msg = f"Source path {repo_path!r} must identify exactly one tracked file at HEAD."
        raise ValueError(msg)
    try:
        header, raw_tree_path = entries[0].split(b"\t", maxsplit=1)
        mode, object_type, blob_bytes = header.decode().split(" ")
        tree_path = raw_tree_path.decode()
    except (UnicodeDecodeError, ValueError) as error:
        msg = f"Git returned an invalid tree record for source path {repo_path!r}."
        raise ValueError(msg) from error
    if tree_path != repo_path:
        msg = f"Git resolved source path {repo_path!r} to a different tree entry."
        raise ValueError(msg)
    if mode not in {"100644", "100755"} or object_type != "blob":
        msg = f"Source path {repo_path!r} must be a tracked regular file, not mode {mode!r}."
        raise ValueError(msg)
    git_blob_id = require_git_blob(blob_bytes, f"git blob for {repo_path}")

    worktree_path = repository_root
    for part in PurePosixPath(repo_path).parts:
        worktree_path /= part
        if worktree_path.is_symlink():
            msg = f"Source path {repo_path!r} must not contain a symbolic-link component."
            raise ValueError(msg)
    try:
        metadata = worktree_path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        msg = f"Tracked source path {repo_path!r} is missing from the worktree."
        raise ValueError(msg) from error
    if not stat.S_ISREG(metadata.st_mode):
        msg = f"Source path {repo_path!r} must be a regular worktree file."
        raise ValueError(msg)
    head_payload = _git(repository_root, "cat-file", "blob", git_blob_id)
    worktree_payload = worktree_path.read_bytes()
    if worktree_payload != head_payload:
        msg = f"Worktree bytes for source path {repo_path!r} differ from the recorded HEAD blob."
        raise ValueError(msg)
    return ExecutionSourceFileRef(
        role=role,
        repo_path=repo_path,
        git_blob_id=git_blob_id,
        source_checksum=_sha256(worktree_payload),
    )


def _role_path_specs(
    *,
    execution_source_paths: Sequence[str | Path],
    analysis_source_paths: Sequence[str | Path],
    dependency_lock_paths: Sequence[str | Path],
    sealed_input_paths: Sequence[str | Path],
) -> tuple[tuple[ExecutionSourceRole, str], ...]:
    """Validate and deterministically order the complete role/path universe.

    Returns:
        Role/path pairs ordered by normalized repository path.

    Raises:
        TypeError: If a role's paths are not a sequence.
        ValueError: If a role is empty or paths are invalid or duplicated.
    """
    role_values: tuple[tuple[ExecutionSourceRole, Sequence[str | Path]], ...] = (
        ("execution_source", execution_source_paths),
        ("analysis_source", analysis_source_paths),
        ("dependency_lock", dependency_lock_paths),
        ("sealed_input", sealed_input_paths),
    )
    specs: list[tuple[ExecutionSourceRole, str]] = []
    for role, values in role_values:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            msg = f"{role}_paths must be a sequence of repository-relative paths."
            raise TypeError(msg)
        if not values:
            msg = f"{role}_paths must contain at least one path."
            raise ValueError(msg)
        specs.extend((role, _path_text(value, f"{role}_paths item")) for value in values)
    paths = tuple(path for _role, path in specs)
    if len(paths) != len(set(paths)):
        msg = "A tracked source path must have exactly one final-lock role."
        raise ValueError(msg)
    return tuple(sorted(specs, key=operator.itemgetter(1)))


def capture_execution_source_manifest(
    repository_root: Path,
    *,
    manifest_id: str,
    entry_point: str | Path,
    execution_source_paths: Sequence[str | Path],
    analysis_source_paths: Sequence[str | Path],
    dependency_lock_paths: Sequence[str | Path],
    sealed_input_paths: Sequence[str | Path],
) -> ExecutionSourceManifest:
    """Capture the final tracked source universe from an exactly clean checkout.

    Args:
        repository_root: Exact Git worktree root.
        manifest_id: Stable final execution-source manifest identifier.
        entry_point: Repository-relative Phase II execution entry point.
        execution_source_paths: Runtime implementation files.
        analysis_source_paths: Frozen primary-analysis implementation files.
        dependency_lock_paths: Environment lockfiles.
        sealed_input_paths: Preregistration and other frozen scientific inputs.

    Returns:
        The checksum-sealed execution-source manifest.

    Raises:
        ValueError: If the repository is dirty or an input is not the exact tracked HEAD file.
    """
    source_commit = _require_clean_checkout(repository_root)
    root = repository_root.resolve()
    specs = _role_path_specs(
        execution_source_paths=execution_source_paths,
        analysis_source_paths=analysis_source_paths,
        dependency_lock_paths=dependency_lock_paths,
        sealed_input_paths=sealed_input_paths,
    )
    source_files = tuple(_capture_source_file(root, source_commit, role, repo_path) for role, repo_path in specs)
    normalized_entry_point = _path_text(entry_point, "entry_point")
    final_commit = _require_clean_checkout(root)
    if final_commit != source_commit:
        msg = "Repository HEAD changed while final execution sources were captured."
        raise ValueError(msg)
    return ExecutionSourceManifest(
        manifest_id=manifest_id,
        source_commit=source_commit,
        entry_point=normalized_entry_point,
        source_files=source_files,
        environment_lock_checksum=_environment_lock_checksum(source_files),
        tracked_source_manifest_checksum=_tracked_manifest_checksum(source_files),
    )


def verify_execution_source_manifest(
    manifest: ExecutionSourceManifest,
    repository_root: Path,
) -> tuple[str, ...]:
    """Verify a final execution-source manifest against a clean checkout.

    Args:
        manifest: Previously captured final source universe.
        repository_root: Exact Git worktree root to verify.

    Returns:
        Verified repository paths in canonical order.

    Raises:
        TypeError: If an input has an unsupported type.
        ValueError: If checkout state, HEAD, a blob, or source bytes differ.
    """
    if not isinstance(manifest, ExecutionSourceManifest):
        msg = f"manifest must be an ExecutionSourceManifest, got {type(manifest).__name__}."
        raise TypeError(msg)
    source_commit = _require_clean_checkout(repository_root)
    if source_commit != manifest.source_commit:
        msg = "The clean checkout HEAD differs from the sealed execution-source commit."
        raise ValueError(msg)
    root = repository_root.resolve()
    actual_files = tuple(
        _capture_source_file(root, source_commit, source_file.role, source_file.repo_path)
        for source_file in manifest.source_files
    )
    actual = ExecutionSourceManifest(
        manifest_id=manifest.manifest_id,
        source_commit=source_commit,
        entry_point=manifest.entry_point,
        source_files=actual_files,
        environment_lock_checksum=_environment_lock_checksum(actual_files),
        tracked_source_manifest_checksum=_tracked_manifest_checksum(actual_files),
    )
    if actual.content_checksum != manifest.content_checksum:
        msg = "The clean checkout does not reproduce the sealed execution-source manifest."
        raise ValueError(msg)
    final_commit = _require_clean_checkout(root)
    if final_commit != source_commit:
        msg = "Repository HEAD changed while final execution sources were verified."
        raise ValueError(msg)
    return tuple(source_file.repo_path for source_file in actual_files)


def _tracked_paths_below(repository_root: Path, repo_path: str) -> tuple[str, ...]:
    """Return every tracked regular-file path below one governed root.

    Args:
        repository_root: Exact Git worktree root.
        repo_path: Repository-relative root to inspect.

    Returns:
        Canonically ordered tracked paths.

    Raises:
        ValueError: If Git reports an invalid path or required files are absent.
    """
    payload = _git(repository_root, "ls-files", "-z", "--", repo_path)
    try:
        paths = tuple(sorted(item.decode() for item in payload.split(b"\0") if item))
    except UnicodeDecodeError as error:
        msg = f"Git returned a non-UTF-8 governed path below {repo_path!r}."
        raise ValueError(msg) from error
    for path in paths:
        require_relative_path(path, "governed tracked path")
    return paths


def _governed_role_paths(repository_root: Path) -> dict[ExecutionSourceRole, tuple[str, ...]]:
    """Derive the mandatory WP22 source universe from tracked project files.

    The inventory deliberately includes every tracked Python implementation in
    the state-preparation benchmark and every tracked YAQS runtime file.  This
    closes the caller-selected-file loophole while allowing future committed
    implementation dependencies to enter the lock automatically.

    Returns:
        Mandatory paths grouped by their unique final-lock role.

    Raises:
        ValueError: If a fixed governed entry point, lock, or input is absent.
    """
    benchmark_paths = _tracked_paths_below(repository_root, "benchmarks/state_preparation")
    runtime_paths = _tracked_paths_below(repository_root, "src/mqt/yaqs")
    fixed_root_paths = tuple(
        path
        for required in WP22_GOVERNED_DEPENDENCY_LOCK_PATHS
        for path in _tracked_paths_below(repository_root, required)
    )
    tracked = set(benchmark_paths) | set(runtime_paths) | set(fixed_root_paths)
    missing = sorted(set(WP22_GOVERNED_REQUIRED_PATHS) - tracked)
    if missing:
        msg = f"The checkout is missing mandatory WP22 governed files: {missing!r}."
        raise ValueError(msg)

    analysis_paths = (WP22_GOVERNED_ANALYSIS_ENTRY_POINT,)
    dependency_paths = WP22_GOVERNED_DEPENDENCY_LOCK_PATHS
    sealed_paths = tuple(
        path for path in benchmark_paths if path.startswith("benchmarks/state_preparation/phase2/data/")
    )
    if WP22_GOVERNED_PREREGISTRATION_PATH not in sealed_paths:
        msg = "The immutable WP22 preregistration is absent from the governed sealed-input inventory."
        raise ValueError(msg)

    excluded = set(analysis_paths) | set(dependency_paths) | set(sealed_paths)
    execution_paths = tuple(
        sorted(
            path
            for path in tracked
            if path not in excluded
            and (
                path.startswith("src/mqt/yaqs/")
                or (path.startswith("benchmarks/state_preparation/") and path.endswith(".py"))
            )
        )
    )
    if WP22_GOVERNED_ENTRY_POINT not in execution_paths:
        msg = "The Phase II training entry point is absent from the governed execution inventory."
        raise ValueError(msg)
    return {
        "execution_source": execution_paths,
        "analysis_source": analysis_paths,
        "dependency_lock": dependency_paths,
        "sealed_input": sealed_paths,
    }


def capture_governed_execution_source_manifest(
    repository_root: Path,
    *,
    manifest_id: str,
) -> ExecutionSourceManifest:
    """Capture the complete mandatory WP22 execution and analysis universe.

    Args:
        repository_root: Exact clean Git worktree root.
        manifest_id: Stable final execution-source manifest identifier.

    Returns:
        A clean-checkout manifest containing every governed project source,
        runtime dependency, environment lock, and sealed Phase II input.

    """
    _require_clean_checkout(repository_root)
    roles = _governed_role_paths(repository_root.resolve())
    return capture_execution_source_manifest(
        repository_root,
        manifest_id=manifest_id,
        entry_point=WP22_GOVERNED_ENTRY_POINT,
        execution_source_paths=roles["execution_source"],
        analysis_source_paths=roles["analysis_source"],
        dependency_lock_paths=roles["dependency_lock"],
        sealed_input_paths=roles["sealed_input"],
    )


def verify_governed_execution_source_manifest(
    manifest: ExecutionSourceManifest,
    repository_root: Path,
) -> tuple[str, ...]:
    """Verify source bytes and mandatory WP22 inventory completeness.

    Args:
        manifest: Previously captured final source universe.
        repository_root: Exact clean Git worktree root to verify.

    Returns:
        Verified mandatory paths in canonical order.

    Raises:
        ValueError: If source bytes, roles, or the mandatory inventory differ.
    """
    verified = verify_execution_source_manifest(manifest, repository_root)
    expected_roles = _governed_role_paths(repository_root.resolve())
    actual_roles = {source_file.repo_path: source_file.role for source_file in manifest.source_files}
    expected_pairs = {path: role for role, paths in expected_roles.items() for path in paths}
    missing_or_changed = {
        path: {"expected": role, "actual": actual_roles.get(path)}
        for path, role in expected_pairs.items()
        if actual_roles.get(path) != role
    }
    if missing_or_changed:
        msg = f"Execution-source manifest omits or misroles mandatory WP22 files: {missing_or_changed!r}."
        raise ValueError(msg)
    if manifest.entry_point != WP22_GOVERNED_ENTRY_POINT:
        msg = "Execution-source manifest does not use the mandatory Phase II training entry point."
        raise ValueError(msg)
    return tuple(path for path in verified if path in expected_pairs)


def build_analysis_source_manifest(
    execution_manifest: ExecutionSourceManifest,
    *,
    manifest_id: str,
    preregistration_checksum: str,
    analysis_template_checksum: str,
    analysis_entry_point: str | Path,
) -> AnalysisSourceManifest:
    """Build the existing WP15 analysis manifest from a final source lock.

    Args:
        execution_manifest: Complete WP22 execution-source universe.
        manifest_id: Stable analysis-source manifest identifier.
        preregistration_checksum: Governing protocol checksum.
        analysis_template_checksum: Separately frozen natural-language analysis checksum.
        analysis_entry_point: Repository-relative frozen analysis entry point.

    Returns:
        An ``AnalysisSourceManifest`` exactly cross-linked to the execution lock.

    Raises:
        TypeError: If ``execution_manifest`` has the wrong type.
        ValueError: If the requested analysis entry point is not analysis source.
    """
    if not isinstance(execution_manifest, ExecutionSourceManifest):
        msg = "execution_manifest must be an ExecutionSourceManifest."
        raise TypeError(msg)
    entry_point = _path_text(analysis_entry_point, "analysis_entry_point")
    analysis_files = tuple(
        AnalysisSourceFileRef(
            repo_path=source_file.repo_path,
            git_blob_id=source_file.git_blob_id,
            content_checksum=source_file.source_checksum,
        )
        for source_file in execution_manifest.source_files
        if source_file.role == "analysis_source"
    )
    if entry_point not in {source_file.repo_path for source_file in analysis_files}:
        msg = "analysis_entry_point must identify a file with the analysis_source role."
        raise ValueError(msg)
    return AnalysisSourceManifest(
        manifest_id=manifest_id,
        preregistration_checksum=preregistration_checksum,
        analysis_template_checksum=analysis_template_checksum,
        source_commit=execution_manifest.source_commit,
        entry_point=entry_point,
        source_files=analysis_files,
        environment_lock_checksum=execution_manifest.environment_lock_checksum,
        execution_source_manifest_checksum=execution_manifest.content_checksum,
        clean_worktree=True,
    )


def verify_analysis_source_bridge(
    execution_manifest: ExecutionSourceManifest,
    analysis_manifest: AnalysisSourceManifest,
    repository_root: Path,
) -> tuple[str, ...]:
    """Verify exact execution-to-analysis source linkage and committed bytes.

    Args:
        execution_manifest: Complete WP22 execution source lock.
        analysis_manifest: Existing WP15 analysis-source view.
        repository_root: Exact clean Git checkout root.

    Returns:
        Verified analysis-source paths in canonical order.

    Raises:
        TypeError: If a manifest has an unsupported type.
        ValueError: If any cross-link or source identity differs.
    """
    if not isinstance(analysis_manifest, AnalysisSourceManifest):
        msg = "analysis_manifest must be an AnalysisSourceManifest."
        raise TypeError(msg)
    verify_governed_execution_source_manifest(execution_manifest, repository_root)
    expected = build_analysis_source_manifest(
        execution_manifest,
        manifest_id=analysis_manifest.manifest_id,
        preregistration_checksum=analysis_manifest.preregistration_checksum,
        analysis_template_checksum=analysis_manifest.analysis_template_checksum,
        analysis_entry_point=analysis_manifest.entry_point,
    )
    if expected != analysis_manifest:
        msg = "AnalysisSourceManifest is not the exact analysis_source projection of the execution lock."
        raise ValueError(msg)
    verified = verify_analysis_source_files(analysis_manifest, repository_root)
    final_commit = _require_clean_checkout(repository_root)
    if final_commit != execution_manifest.source_commit:
        msg = "Repository HEAD changed while the analysis-source bridge was verified."
        raise ValueError(msg)
    return verified


def verify_final_seal_source_lock(
    final_seal: FinalConfirmationSeal,
    execution_manifest: ExecutionSourceManifest,
    analysis_manifest: AnalysisSourceManifest,
    repository_root: Path,
) -> tuple[str, ...]:
    """Verify the existing final-seal source fields against the WP22 lock.

    Args:
        final_seal: Existing checksum-sealed confirmation design.
        execution_manifest: Complete final tracked-source inventory.
        analysis_manifest: Frozen executable primary-analysis projection.
        repository_root: Exact clean Git checkout root.

    Returns:
        Verified analysis-source paths.

    Raises:
        TypeError: If ``final_seal`` has the wrong type.
        ValueError: If a final-seal checksum is not the exact source-lock link.
    """
    if not isinstance(final_seal, FinalConfirmationSeal):
        msg = "final_seal must be a FinalConfirmationSeal."
        raise TypeError(msg)
    if final_seal.execution_source_checksum != execution_manifest.content_checksum:
        msg = "Final seal does not reference the exact execution-source manifest."
        raise ValueError(msg)
    if final_seal.analysis_source_manifest_checksum != analysis_manifest.content_checksum:
        msg = "Final seal does not reference the exact analysis-source manifest."
        raise ValueError(msg)
    if final_seal.analysis_template_checksum != analysis_manifest.analysis_template_checksum:
        msg = "Final seal and analysis source use different analysis-template checksums."
        raise ValueError(msg)
    return verify_analysis_source_bridge(execution_manifest, analysis_manifest, repository_root)


__all__ = [
    "EXECUTION_SOURCE_FILE_SCHEMA_VERSION",
    "EXECUTION_SOURCE_MANIFEST_SCHEMA_VERSION",
    "EXECUTION_SOURCE_ROLES",
    "WP22_GOVERNED_ANALYSIS_ENTRY_POINT",
    "WP22_GOVERNED_DEPENDENCY_LOCK_PATHS",
    "WP22_GOVERNED_ENTRY_POINT",
    "WP22_GOVERNED_PREREGISTRATION_PATH",
    "WP22_GOVERNED_REQUIRED_PATHS",
    "ExecutionSourceFileRef",
    "ExecutionSourceManifest",
    "ExecutionSourceRole",
    "build_analysis_source_manifest",
    "capture_execution_source_manifest",
    "capture_governed_execution_source_manifest",
    "verify_analysis_source_bridge",
    "verify_execution_source_manifest",
    "verify_final_seal_source_lock",
    "verify_governed_execution_source_manifest",
]
