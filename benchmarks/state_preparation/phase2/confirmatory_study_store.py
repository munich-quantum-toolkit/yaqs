# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Append-only filesystem custody for the locked WP23 confirmatory study.

The scientific study schema lives in :mod:`.confirmatory_study`.  This module
is the only production collector: it authenticates the canonical terminal job
prefix from disk, closes the exact output tree, and publishes content-addressed
snapshots.  A snapshot never hashes itself.  Instead it points to the previous
snapshot receipt and inventories all non-snapshot output members, which keeps
the custody graph acyclic and resumable.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .confirmatory_study import LockedConfirmatoryStudyManifest, PriorTargetExposureInventory
from .execution_context import ConfirmationExecutionContext
from .production_executors import (
    CONFIRMATION_PLAN_SESSION_NAME,
    ProductionAttemptStore,
    ReopenedProductionResult,
    validate_confirmation_plan_session,
    validate_existing_confirmation_outcome,
)
from .training_orchestration import (
    JOB_ATTEMPTS_DIRECTORY_NAME,
    JOB_RESULT_NAME,
    TrainingJob,
    TrainingJobOutcome,
    load_training_job_outcome_history,
    training_job_attempt_path,
)
from .validation import require_checksum, require_int, require_relative_path

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFIRMATORY_OUTPUT_ENTRY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.confirmatory_output_entry.v1"
LOCKED_CONFIRMATORY_STUDY_SNAPSHOT_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.locked_confirmatory_study_snapshot.v1"
)
LOCKED_CONFIRMATORY_STUDY_SNAPSHOT_REF_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.locked_confirmatory_study_snapshot_ref.v1"
)
CONFIRMATORY_STUDY_DIRECTORY_NAME = "confirmation_study"

_SNAPSHOT_PATTERN = re.compile(r"^snapshot_(?P<ordinal>[0-9]{8})_(?P<digest>[0-9a-f]{64})\.json$")
_RECOVERABLE_ATTEMPT_DIRECTORY_SUFFIXES = frozenset({
    "diagnostics",
    "diagnostics/maps",
    "evaluation",
    "map_evidence",
    "maps",
    "runtime",
    "schedule",
    "smoke",
    "smoke/maps",
    "structural_prefix",
    "structural_prefix/maps",
})
_OUTPUT_ENTRY_KEYS = frozenset({
    "schema_version",
    "relative_path",
    "entry_kind",
    "byte_count",
    "file_checksum",
    "content_checksum",
})
_SNAPSHOT_REF_KEYS = frozenset({
    "schema_version",
    "relative_path",
    "ordinal",
    "file_checksum",
    "snapshot_content_checksum",
    "content_checksum",
})
_SNAPSHOT_KEYS = frozenset({
    "schema_version",
    "ordinal",
    "authorized_output_root",
    "session_marker_content_checksum",
    "previous_snapshot",
    "study_manifest",
    "output_entries",
    "filesystem_inventory_root",
    "content_checksum",
})
_CLI_HEAD_CUSTODY_WRAPPER_KEYS = frozenset({
    "attempted",
    "external_study_head_custody_required",
    "failed",
    "locked_study_snapshot_reference",
    "planned",
    "skipped",
    "succeeded",
})


def _sha256_bytes(payload: bytes) -> str:
    """Return the repository checksum spelling for exact bytes."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _strict_sequence(value: object, name: str) -> tuple[object, ...]:
    """Return a canonical-decoded JSON sequence.

    Returns:
        The immutable decoded sequence.

    Raises:
        TypeError: If the decoded value is not a JSON sequence.
    """
    if not isinstance(value, tuple):
        msg = f"{name} must be a JSON array."
        raise TypeError(msg)
    return value


@dataclass(frozen=True, slots=True)
class ConfirmatoryOutputEntry:
    """One exact directory or regular-file member outside snapshot files."""

    relative_path: str
    entry_kind: Literal["directory", "file"]
    byte_count: int | None
    file_checksum: str | None
    schema_version: str = field(default=CONFIRMATORY_OUTPUT_ENTRY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate path, kind, and file receipt closure.

        Raises:
            ValueError: If the path, kind, or receipt fields are inconsistent.
        """
        object.__setattr__(self, "relative_path", require_relative_path(self.relative_path, "relative_path"))
        if self.entry_kind not in {"directory", "file"}:
            msg = "entry_kind must be directory or file."
            raise ValueError(msg)
        if self.entry_kind == "directory":
            if self.byte_count is not None or self.file_checksum is not None:
                msg = "Directory output entries cannot claim byte receipts."
                raise ValueError(msg)
            return
        if self.byte_count is None or self.file_checksum is None:
            msg = "File output entries require byte_count and file_checksum."
            raise ValueError(msg)
        object.__setattr__(self, "byte_count", require_int(self.byte_count, "byte_count"))
        object.__setattr__(self, "file_checksum", require_checksum(self.file_checksum, "file_checksum"))

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact path/type/byte receipt."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "entry_kind": self.entry_kind,
            "byte_count": self.byte_count,
            "file_checksum": self.file_checksum,
        }

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native output-entry data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> ConfirmatoryOutputEntry:
        """Decode and verify one output entry.

        Returns:
            The verified entry.

        Raises:
            ValueError: If schema or checksum verification fails.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_OUTPUT_ENTRY_KEYS, name="confirmatory output entry")
        if mapping["schema_version"] != CONFIRMATORY_OUTPUT_ENTRY_SCHEMA_VERSION:
            msg = "Confirmatory output entry uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            relative_path=cast("str", mapping["relative_path"]),
            entry_kind=cast("Literal['directory', 'file']", mapping["entry_kind"]),
            byte_count=cast("int | None", mapping["byte_count"]),
            file_checksum=cast("str | None", mapping["file_checksum"]),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Confirmatory output entry checksum changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class LockedConfirmatoryStudySnapshotRef:
    """Content-addressed reference to one append-only study snapshot."""

    relative_path: str
    ordinal: int
    file_checksum: str
    snapshot_content_checksum: str
    schema_version: str = field(default=LOCKED_CONFIRMATORY_STUDY_SNAPSHOT_REF_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the path, ordinal, and both snapshot checksums.

        Raises:
            ValueError: If the address is not content-derived.
        """
        path = require_relative_path(self.relative_path, "relative_path")
        ordinal = require_int(self.ordinal, "ordinal")
        file_checksum = require_checksum(self.file_checksum, "file_checksum")
        snapshot_checksum = require_checksum(self.snapshot_content_checksum, "snapshot_content_checksum")
        expected = _snapshot_relative_path(ordinal, snapshot_checksum)
        if path != expected:
            msg = "Snapshot reference path is not derived from its ordinal and content checksum."
            raise ValueError(msg)
        object.__setattr__(self, "relative_path", path)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "file_checksum", file_checksum)
        object.__setattr__(self, "snapshot_content_checksum", snapshot_checksum)

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete snapshot address."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "ordinal": self.ordinal,
            "file_checksum": self.file_checksum,
            "snapshot_content_checksum": self.snapshot_content_checksum,
        }

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native reference data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed snapshot-reference JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> LockedConfirmatoryStudySnapshotRef:
        """Decode and verify one snapshot reference.

        Returns:
            The verified reference.

        Raises:
            ValueError: If schema or checksum verification fails.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_SNAPSHOT_REF_KEYS, name="study snapshot reference")
        if mapping["schema_version"] != LOCKED_CONFIRMATORY_STUDY_SNAPSHOT_REF_SCHEMA_VERSION:
            msg = "Study snapshot reference uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            relative_path=cast("str", mapping["relative_path"]),
            ordinal=cast("int", mapping["ordinal"]),
            file_checksum=cast("str", mapping["file_checksum"]),
            snapshot_content_checksum=cast("str", mapping["snapshot_content_checksum"]),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Study snapshot reference checksum changed during normalization."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> LockedConfirmatoryStudySnapshotRef:
        """Decode one canonical snapshot-reference document.

        Returns:
            The verified reference.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class LockedConfirmatoryStudySnapshot:
    """One immutable aggregate manifest plus exact non-snapshot output inventory."""

    ordinal: int
    authorized_output_root: str
    session_marker_content_checksum: str
    previous_snapshot: LockedConfirmatoryStudySnapshotRef | None
    study_manifest: LockedConfirmatoryStudyManifest
    output_entries: tuple[ConfirmatoryOutputEntry, ...]
    filesystem_inventory_root: str
    schema_version: str = field(default=LOCKED_CONFIRMATORY_STUDY_SNAPSHOT_SCHEMA_VERSION, init=False)
    _content_checksum: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the chain link, context roots, and exact inventory root.

        Raises:
            TypeError: If embedded objects have unsupported types.
            ValueError: If chain or inventory identities are inconsistent.
        """
        ordinal = require_int(self.ordinal, "ordinal")
        output_root = Path(self.authorized_output_root)
        if not output_root.is_absolute() or output_root != output_root.resolve():
            msg = "authorized_output_root must be an absolute canonical path."
            raise ValueError(msg)
        marker = require_checksum(self.session_marker_content_checksum, "session_marker_content_checksum")
        if not isinstance(self.study_manifest, LockedConfirmatoryStudyManifest):
            msg = "study_manifest must be a LockedConfirmatoryStudyManifest."
            raise TypeError(msg)
        previous = self.previous_snapshot
        if ordinal == 0 and previous is not None:
            msg = "The initial study snapshot cannot reference a predecessor."
            raise ValueError(msg)
        if ordinal > 0 and (
            not isinstance(previous, LockedConfirmatoryStudySnapshotRef) or previous.ordinal != ordinal - 1
        ):
            msg = "A later study snapshot must reference the immediately preceding ordinal."
            raise ValueError(msg)
        entries = tuple(self.output_entries)
        if not all(isinstance(item, ConfirmatoryOutputEntry) for item in entries):
            msg = "output_entries must contain ConfirmatoryOutputEntry values."
            raise TypeError(msg)
        paths = tuple(item.relative_path for item in entries)
        if paths != tuple(sorted(set(paths))):
            msg = "Output entries must have unique paths in lexical order."
            raise ValueError(msg)
        expected_inventory = canonical_checksum({
            "entry_checksums": [item.content_checksum for item in entries],
        })
        if require_checksum(self.filesystem_inventory_root, "filesystem_inventory_root") != expected_inventory:
            msg = "filesystem_inventory_root is not derived from every exact output entry."
            raise ValueError(msg)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "authorized_output_root", str(output_root))
        object.__setattr__(self, "session_marker_content_checksum", marker)
        object.__setattr__(self, "output_entries", entries)
        object.__setattr__(self, "filesystem_inventory_root", expected_inventory)
        object.__setattr__(self, "_content_checksum", canonical_checksum(self._content_dict()))

    @property
    def content_checksum(self) -> str:
        """Checksum of the acyclic snapshot document."""
        return self._content_checksum

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ordinal": self.ordinal,
            "authorized_output_root": self.authorized_output_root,
            "session_marker_content_checksum": self.session_marker_content_checksum,
            "previous_snapshot": None if self.previous_snapshot is None else self.previous_snapshot.to_dict(),
            "study_manifest": self.study_manifest.to_dict(),
            "output_entries": [item.to_dict() for item in self.output_entries],
            "filesystem_inventory_root": self.filesystem_inventory_root,
        }

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native snapshot data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical snapshot JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> LockedConfirmatoryStudySnapshot:
        """Decode and verify one aggregate snapshot.

        Returns:
            The verified snapshot.

        Raises:
            ValueError: If schema, chain, or checksum verification fails.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_SNAPSHOT_KEYS, name="locked study snapshot")
        if mapping["schema_version"] != LOCKED_CONFIRMATORY_STUDY_SNAPSHOT_SCHEMA_VERSION:
            msg = "Locked study snapshot uses an unsupported schema version."
            raise ValueError(msg)
        raw_previous = mapping["previous_snapshot"]
        result = cls(
            ordinal=cast("int", mapping["ordinal"]),
            authorized_output_root=cast("str", mapping["authorized_output_root"]),
            session_marker_content_checksum=cast("str", mapping["session_marker_content_checksum"]),
            previous_snapshot=(
                None if raw_previous is None else LockedConfirmatoryStudySnapshotRef.from_dict(raw_previous)
            ),
            study_manifest=LockedConfirmatoryStudyManifest.from_dict(mapping["study_manifest"]),
            output_entries=tuple(
                ConfirmatoryOutputEntry.from_dict(item)
                for item in _strict_sequence(mapping["output_entries"], "output_entries")
            ),
            filesystem_inventory_root=cast("str", mapping["filesystem_inventory_root"]),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Locked study snapshot checksum changed during normalization."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> LockedConfirmatoryStudySnapshot:
        """Decode one canonical snapshot document.

        Returns:
            The verified snapshot.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _snapshot_relative_path(ordinal: int, content_checksum: str) -> str:
    digest = require_checksum(content_checksum, "content_checksum").removeprefix("sha256:")
    return f"{CONFIRMATORY_STUDY_DIRECTORY_NAME}/snapshot_{ordinal:08d}_{digest}.json"


def _snapshot_ref(snapshot: LockedConfirmatoryStudySnapshot, payload: bytes) -> LockedConfirmatoryStudySnapshotRef:
    return LockedConfirmatoryStudySnapshotRef(
        relative_path=_snapshot_relative_path(snapshot.ordinal, snapshot.content_checksum),
        ordinal=snapshot.ordinal,
        file_checksum=_sha256_bytes(payload),
        snapshot_content_checksum=snapshot.content_checksum,
    )


def _read_locked_study_head_custody(
    path: Path,
) -> tuple[LockedConfirmatoryStudySnapshotRef, bytes] | None:
    """Read one exact external head document through a pinned descriptor.

    Returns:
        The checksum-verified reference and canonical bytes, or ``None`` when
        absent. Both the raw reference and strict prior-CLI-summary wrapper are
        accepted; publication always normalizes custody back to the raw form.

    Raises:
        ValueError: If custody is linked, non-regular, multiply linked,
            noncanonical, or changes while it is read.
    """
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        msg = "External locked-study head custody is unavailable during inspection."
        raise ValueError(msg) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        msg = "External locked-study head custody must be a single-link regular file."
        raise ValueError(msg)
    try:
        payload, closed = _read_pinned_file(path, metadata, str(path))
        current = path.lstat()
    except (FileNotFoundError, OSError) as error:
        msg = "External locked-study head custody changed during inspection."
        raise ValueError(msg) from error
    identity = (metadata.st_dev, metadata.st_ino, metadata.st_size)
    if (
        not stat.S_ISREG(closed.st_mode)
        or closed.st_nlink != 1
        or current.st_nlink != 1
        or (closed.st_dev, closed.st_ino, closed.st_size) != identity
        or (current.st_dev, current.st_ino, current.st_size) != identity
    ):
        msg = "External locked-study head custody changed during inspection."
        raise ValueError(msg)
    try:
        reference = _decode_locked_study_head_document(payload)
    except (TypeError, UnicodeDecodeError, ValueError) as error:
        msg = "External locked-study head custody is not a valid snapshot reference."
        raise ValueError(msg) from error
    return reference, payload


def _decode_locked_study_head_document(payload: bytes) -> LockedConfirmatoryStudySnapshotRef:
    """Decode a raw head reference or the exact prior-CLI-summary wrapper.

    Returns:
        The checksum-verified embedded reference.

    Raises:
        ValueError: If canonical JSON, wrapper shape, or reference validation fails.
    """
    document = load_canonical_json_object(payload.decode())
    if "locked_study_snapshot_reference" not in document:
        return LockedConfirmatoryStudySnapshotRef.from_dict(document)
    if frozenset(document) != _CLI_HEAD_CUSTODY_WRAPPER_KEYS:
        msg = "External locked-study CLI custody wrapper has unexpected fields."
        raise ValueError(msg)
    if document["external_study_head_custody_required"] is not True:
        msg = "External locked-study CLI custody wrapper must require external custody."
        raise ValueError(msg)
    for name in ("planned", "attempted", "succeeded", "failed", "skipped"):
        require_int(document[name], f"external locked-study CLI custody wrapper.{name}")
    return LockedConfirmatoryStudySnapshotRef.from_dict(document["locked_study_snapshot_reference"])


def _load_locked_study_head_custody(path: Path) -> LockedConfirmatoryStudySnapshotRef | None:
    """Return the verified reference from pinned external custody when present."""
    loaded = _read_locked_study_head_custody(path)
    return None if loaded is None else loaded[0]


def _validate_head_custody_against_chain(
    context: ConfirmationExecutionContext,
    chain: Sequence[tuple[LockedConfirmatoryStudySnapshot, LockedConfirmatoryStudySnapshotRef]],
) -> LockedConfirmatoryStudySnapshotRef | None:
    """Authenticate external head custody as a member of the internal chain.

    Returns:
        The externally retained reference, or ``None`` only before snapshot
        zero has completed external publication.

    Raises:
        ValueError: If external custody is foreign, rolled back past a
            recoverable snapshot-zero publication gap, or precedes no chain.
    """
    external = _load_locked_study_head_custody(context.locked_study_head_custody_path)
    references = tuple(reference for _snapshot, reference in chain)
    if not references:
        if external is not None:
            msg = "External locked-study head custody exists before the internal snapshot chain."
            raise ValueError(msg)
        return None
    if external is None:
        if len(references) == 1:
            snapshot, reference = chain[0]
            _validate_initial_snapshot_identity(
                context,
                snapshot.session_marker_content_checksum,
                snapshot,
                reference,
            )
            return None
        msg = "External locked-study head custody is missing from an established snapshot chain."
        raise ValueError(msg)
    if external not in references:
        msg = "External locked-study head custody is not a verified member of the snapshot chain."
        raise ValueError(msg)
    if external != references[-1] and (len(references) < 2 or external != references[-2]):
        msg = "External locked-study head custody lags beyond one recoverable publication crash window."
        raise ValueError(msg)
    return external


def _validate_recoverable_external_publication_gap(
    chain: Sequence[tuple[LockedConfirmatoryStudySnapshot, LockedConfirmatoryStudySnapshotRef]],
    external_head: LockedConfirmatoryStudySnapshotRef | None,
    manifest: LockedConfirmatoryStudyManifest,
    entries: Sequence[ConfirmatoryOutputEntry],
    *,
    has_orphan_attempt: bool,
) -> None:
    """Limit absent external custody to exact snapshot-zero publication state.

    Raises:
        ValueError: If any result or crash prefix appeared before snapshot-zero
            external custody completed.
    """
    if external_head is not None or not chain:
        return
    initial = chain[0][0]
    if (
        len(chain) != 1
        or has_orphan_attempt
        or manifest != initial.study_manifest
        or tuple(entries) != initial.output_entries
    ):
        msg = "Missing external head custody is recoverable only for the unchanged all-unattempted snapshot zero."
        raise ValueError(msg)


def _publish_locked_study_head_custody(
    path: Path,
    reference: LockedConfirmatoryStudySnapshotRef,
    expected_previous: LockedConfirmatoryStudySnapshotRef | None,
) -> None:
    """Atomically advance exact external head custody in its own directory.

    Raises:
        ValueError: If existing custody changed, is unsafe, or cannot be
            durably replaced with the new canonical reference.
    """
    loaded = _read_locked_study_head_custody(path)
    current = None if loaded is None else loaded[0]
    payload = f"{reference.to_json()}\n".encode()
    if current == reference and loaded is not None and loaded[1] == payload:
        return
    if current != expected_previous:
        msg = "External locked-study head custody changed before publication."
        raise ValueError(msg)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        _replace_staged_locked_study_head(
            descriptor,
            temporary,
            path,
            payload,
            expected_previous,
            reference,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _replace_staged_locked_study_head(
    descriptor: int,
    temporary: Path,
    path: Path,
    payload: bytes,
    expected_previous: LockedConfirmatoryStudySnapshotRef | None,
    reference: LockedConfirmatoryStudySnapshotRef,
) -> None:
    """Stage, atomically replace, and reopen one external head reference.

    Raises:
        ValueError: If staging, concurrent state, or the published bytes differ.
    """
    _stage_snapshot_payload(descriptor, payload)
    staged = temporary.lstat()
    if not stat.S_ISREG(staged.st_mode) or staged.st_nlink != 1:
        msg = "Staged external locked-study head custody is unsafe."
        raise ValueError(msg)
    if _load_locked_study_head_custody(path) != expected_previous:
        msg = "External locked-study head custody changed during publication."
        raise ValueError(msg)
    temporary.replace(path)
    _fsync_directory(path.parent)
    if _load_locked_study_head_custody(path) != reference:
        msg = "External locked-study head custody differs after publication."
        raise ValueError(msg)


def _path_ancestors(relative_path: str) -> set[str]:
    path = PurePosixPath(relative_path)
    return {str(parent) for parent in path.parents if str(parent) != "."}


def _canonical_crash_directory_candidates(job: TrainingJob, store: ProductionAttemptStore) -> set[str]:
    """Return the bounded directory state machine for one first attempt.

    Returns:
        Every plan-derived job ancestor, outer-history directory, attempt
        ancestor, and repository-known nested artifact directory that a crash
        may leave before its first byte member is created.
    """
    job_root = PurePosixPath(job.output_path)
    attempt_root = job_root / PurePosixPath(store.relative_attempt_directory)
    candidates = {
        *_path_ancestors(str(job_root)),
        str(job_root),
        str(job_root / JOB_ATTEMPTS_DIRECTORY_NAME),
        *_path_ancestors(str(attempt_root)),
        str(attempt_root),
    }
    candidates.update(str(attempt_root / suffix) for suffix in _RECOVERABLE_ATTEMPT_DIRECTORY_SUFFIXES)
    return candidates


def _present_canonical_crash_directories(
    root: Path,
    job: TrainingJob,
    store: ProductionAttemptStore,
) -> set[str]:
    """Return safely present members of one bounded crash-directory universe.

    Returns:
        Exact relative paths for present canonical directories.

    Raises:
        ValueError: If a canonical directory position is linked, special, or
            unavailable during inspection.
    """
    present: set[str] = set()
    for relative_path in sorted(_canonical_crash_directory_candidates(job, store)):
        path = root / relative_path
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            msg = "A canonical confirmation crash directory is unavailable during inspection."
            raise ValueError(msg) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            msg = f"Canonical confirmation crash directory {relative_path!r} is linked or non-directory."
            raise ValueError(msg)
        present.add(relative_path)
    return present


def _file_entry(root: Path, relative_path: str) -> ConfirmatoryOutputEntry:
    return _file_payload_and_entry(root, relative_path)[1]


def _file_payload_and_entry(root: Path, relative_path: str) -> tuple[bytes, ConfirmatoryOutputEntry]:
    """Read and receipt one single-link regular file without following it.

    Returns:
        Exact bytes and their path-bound output entry.

    Raises:
        ValueError: If the member is missing, unsafe, linked, or changes while read.
    """
    path = root / relative_path
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        msg = f"Confirmation output file {relative_path!r} is missing, linked, or non-regular."
        raise ValueError(msg)
    payload, closed = _read_pinned_file(path, metadata, relative_path)
    current = path.lstat()
    if current.st_nlink != 1 or (closed.st_dev, closed.st_ino, closed.st_size) != (
        current.st_dev,
        current.st_ino,
        current.st_size,
    ):
        msg = f"Confirmation output file {relative_path!r} changed during inventory."
        raise ValueError(msg)
    entry = ConfirmatoryOutputEntry(relative_path, "file", len(payload), _sha256_bytes(payload))
    return payload, entry


def _read_pinned_file(path: Path, metadata: os.stat_result, relative_path: str) -> tuple[bytes, os.stat_result]:
    """Read a nofollow regular file and return its final descriptor identity.

    Returns:
        Exact bytes and post-read descriptor metadata.
    """
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        _validate_opened_file(opened, metadata, relative_path)
        return _read_descriptor(descriptor), os.fstat(descriptor)
    finally:
        os.close(descriptor)


def _validate_opened_file(opened: os.stat_result, expected: os.stat_result, relative_path: str) -> None:
    """Require an opened file to retain its scanned single-link identity.

    Raises:
        ValueError: If type, link count, inode, or size changed.
    """
    expected_identity = (expected.st_dev, expected.st_ino, expected.st_size)
    actual_identity = (opened.st_dev, opened.st_ino, opened.st_size)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or actual_identity != expected_identity:
        msg = f"Confirmation output file {relative_path!r} changed during open."
        raise ValueError(msg)


def _read_descriptor(descriptor: int) -> bytes:
    """Read all bytes from a pinned descriptor without taking ownership.

    Returns:
        The exact descriptor bytes.
    """
    with os.fdopen(descriptor, "rb", closefd=False) as handle:
        return handle.read()


def _scan_output_tree(root: Path) -> tuple[ConfirmatoryOutputEntry, ...]:
    """Enumerate every non-snapshot output member without following links.

    Returns:
        Lexically ordered exact directory and regular-file receipts.
    """
    entries: list[ConfirmatoryOutputEntry] = []

    def visit(directory: Path, relative_directory: PurePosixPath) -> None:
        with os.scandir(directory) as children:
            ordered = sorted(children, key=lambda item: item.name)
        for child in ordered:
            relative = relative_directory / child.name
            relative_text = str(relative)
            metadata = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                msg = f"Confirmation output contains a symlink: {relative_text}."
                raise ValueError(msg)
            if stat.S_ISDIR(metadata.st_mode):
                entries.append(ConfirmatoryOutputEntry(relative_text, "directory", None, None))
                visit(Path(child.path), relative)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                msg = f"Confirmation output contains a special member: {relative_text}."
                raise ValueError(msg)
            if relative.parts[:1] == (CONFIRMATORY_STUDY_DIRECTORY_NAME,):
                if len(relative.parts) != 2 or _SNAPSHOT_PATTERN.fullmatch(child.name) is None:
                    msg = f"Confirmation study directory contains a foreign member: {relative_text}."
                    raise ValueError(msg)
                continue
            entries.append(_file_entry(root, relative_text))

    visit(root, PurePosixPath())
    return tuple(sorted(entries, key=lambda item: item.relative_path))


def _validate_latest_projection(job_directory: Path, outcome: TrainingJobOutcome) -> bool:
    path = job_directory / JOB_RESULT_NAME
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or not path.is_file():
        msg = "Confirmation latest-outcome projection is linked or non-regular."
        raise ValueError(msg)
    expected = f"{canonical_json(outcome.to_dict())}\n".encode()
    if path.read_bytes() != expected:
        msg = "Confirmation latest-outcome projection differs from immutable history."
        raise ValueError(msg)
    return True


def _terminal_paths(
    root: Path,
    job: TrainingJob,
    outcome: TrainingJobOutcome,
    reopened: ReopenedProductionResult,
) -> tuple[set[str], set[str]]:
    job_root = PurePosixPath(job.output_path)
    history_path = training_job_attempt_path(root / job.output_path, 1).relative_to(root).as_posix()
    expected_history = f"{canonical_json(outcome.to_dict())}\n".encode()
    if (root / history_path).read_bytes() != expected_history:
        msg = "Confirmation outer attempt bytes differ from the authenticated outcome."
        raise ValueError(msg)
    paths = {
        history_path,
        str(job_root / reopened.reference.manifest_path),
        *(str(job_root / ref.path) for ref in reopened.manifest.blobs),
    }
    if _validate_latest_projection(root / job.output_path, outcome):
        paths.add(str(job_root / JOB_RESULT_NAME))
    request = job.confirm_execution_request
    if request is None:
        msg = "Terminal confirmation custody requires a sealed request."
        raise ValueError(msg)
    directories = (
        _present_canonical_crash_directories(
            root,
            job,
            ProductionAttemptStore(root / job.output_path, request.content_checksum, 1),
        )
        if outcome.status == "failure"
        else set()
    )
    return paths, directories


def _orphan_attempt_paths(
    context: ConfirmationExecutionContext,
    job: TrainingJob,
) -> tuple[set[str], set[str]]:
    request = job.confirm_execution_request
    if request is None:
        msg = "Orphan confirmation custody requires a sealed request."
        raise ValueError(msg)
    job_directory = context.authorized_output_root / job.output_path
    store = ProductionAttemptStore(job_directory, request.content_checksum, 1)
    directories = _present_canonical_crash_directories(context.authorized_output_root, job, store)
    if not store.attempt_directory_exists():
        return set(), directories
    paths: set[str] = set()
    if store.terminal_manifest_exists():
        reference = store.derive_existing_ref()
        provisional = TrainingJobOutcome(
            job_checksum=job.content_checksum,
            status=reference.status,
            result_artifact_checksum=(reference.content_checksum if reference.status == "success" else None),
            exception_type=None if reference.status == "success" else "executor_failure",
            message=(
                None
                if reference.status == "success"
                else "executor failed; secret-bearing diagnostics are intentionally not persisted"
            ),
            attempt=1,
        )
        reopened = validate_existing_confirmation_outcome(
            context,
            job,
            provisional,
            job_directory,
        )
        paths.add(str(PurePosixPath(job.output_path) / reopened.reference.manifest_path))
        paths.update(str(PurePosixPath(job.output_path) / ref.path) for ref in reopened.manifest.blobs)
        return paths, directories
    refs = store.inventory_closed_members()
    paths.update(str(PurePosixPath(job.output_path) / ref.path) for ref in refs)
    return paths, directories


def _collect_study(
    context: ConfirmationExecutionContext,
    exposure_inventory: PriorTargetExposureInventory,
) -> tuple[LockedConfirmatoryStudyManifest, set[str], set[str], bool]:
    outcomes: dict[str, TrainingJobOutcome] = {}
    reopened_results: dict[str, ReopenedProductionResult] = {}
    allowed_files = {
        CONFIRMATION_PLAN_SESSION_NAME,
    }
    allowed_directories: set[str] = set()
    first_unattempted: TrainingJob | None = None
    gap_seen = False
    for job in context.plan.jobs:
        job_directory = context.authorized_output_root / job.output_path
        history = load_training_job_outcome_history(job_directory, job)
        if len(history) > 1:
            msg = "Confirmatory output contains more than one outer attempt."
            raise ValueError(msg)
        if not history:
            gap_seen = True
            if first_unattempted is None:
                first_unattempted = job
            continue
        if gap_seen:
            msg = "Confirmatory terminal jobs must form one contiguous canonical plan prefix."
            raise ValueError(msg)
        outcome = history[0]
        reopened = validate_existing_confirmation_outcome(context, job, outcome, job_directory)
        outcomes[job.content_checksum] = outcome
        reopened_results[job.content_checksum] = reopened
        terminal_files, terminal_directories = _terminal_paths(
            context.authorized_output_root,
            job,
            outcome,
            reopened,
        )
        allowed_files.update(terminal_files)
        allowed_directories.update(terminal_directories)

    has_orphan_attempt = False
    if first_unattempted is not None:
        orphan_files, orphan_directories = _orphan_attempt_paths(context, first_unattempted)
        established_directories: set[str] = set()
        for path in {*allowed_files, *orphan_files}:
            established_directories.update(_path_ancestors(path))
        has_orphan_attempt = bool(orphan_files or (orphan_directories - established_directories))
        allowed_files.update(orphan_files)
        allowed_directories.update(orphan_directories)
    manifest = LockedConfirmatoryStudyManifest._from_authenticated_reopened_results(  # noqa: SLF001
        context=context,
        exposure_inventory=exposure_inventory,
        outcomes_by_job=outcomes,
        reopened_results_by_job=reopened_results,
    )
    if manifest.status == "incomplete_resource_limit" and has_orphan_attempt:
        msg = "No production attempt may follow the authenticated confirmatory resource stop."
        raise ValueError(msg)
    return manifest, allowed_files, allowed_directories, has_orphan_attempt


def _expected_entry_paths(
    allowed_files: set[str],
    explicit_directories: set[str],
) -> tuple[set[str], set[str]]:
    allowed_directories = {CONFIRMATORY_STUDY_DIRECTORY_NAME, *explicit_directories}
    for path in allowed_files:
        allowed_directories.update(_path_ancestors(path))
    for path in explicit_directories:
        allowed_directories.update(_path_ancestors(path))
    return allowed_files, allowed_directories


def _close_current_output(
    context: ConfirmationExecutionContext,
    exposure_inventory: PriorTargetExposureInventory,
) -> tuple[LockedConfirmatoryStudyManifest, tuple[ConfirmatoryOutputEntry, ...], bool]:
    manifest, allowed_files, explicit_directories, has_orphan_attempt = _collect_study(
        context,
        exposure_inventory,
    )
    entries = _scan_output_tree(context.authorized_output_root)
    actual_files = {item.relative_path for item in entries if item.entry_kind == "file"}
    actual_directories = {item.relative_path for item in entries if item.entry_kind == "directory"}
    expected_files, expected_directories = _expected_entry_paths(allowed_files, explicit_directories)
    if actual_files != expected_files or actual_directories != expected_directories:
        msg = (
            "Confirmation output tree differs from the exact plan and custody universe: "
            f"extra_files={sorted(actual_files - expected_files)!r}, "
            f"missing_files={sorted(expected_files - actual_files)!r}, "
            f"extra_directories={sorted(actual_directories - expected_directories)!r}, "
            f"missing_directories={sorted(expected_directories - actual_directories)!r}."
        )
        raise ValueError(msg)
    return manifest, entries, has_orphan_attempt


def _verify_snapshot_entries(root: Path, entries: Sequence[ConfirmatoryOutputEntry]) -> None:
    for entry in entries:
        path = root / entry.relative_path
        if entry.entry_kind == "directory":
            if path.is_symlink() or not path.is_dir():
                msg = f"Previously inventoried directory changed: {entry.relative_path}."
                raise ValueError(msg)
            continue
        current = _file_entry(root, entry.relative_path)
        if current != entry:
            msg = f"Previously inventoried file changed: {entry.relative_path}."
            raise ValueError(msg)


def _load_snapshot_chain(
    context: ConfirmationExecutionContext,
    exposure_inventory: PriorTargetExposureInventory | None,
    session_marker_content_checksum: str,
) -> tuple[tuple[LockedConfirmatoryStudySnapshot, LockedConfirmatoryStudySnapshotRef], ...]:
    directory = context.authorized_output_root / CONFIRMATORY_STUDY_DIRECTORY_NAME
    if directory.is_symlink() or not directory.is_dir():
        msg = "Confirmation study directory is missing, linked, or non-directory."
        raise ValueError(msg)
    paths = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    chain: list[tuple[LockedConfirmatoryStudySnapshot, LockedConfirmatoryStudySnapshotRef]] = []
    for expected_ordinal, path in enumerate(paths):
        match = _SNAPSHOT_PATTERN.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            msg = "Confirmation study directory contains a foreign or unsafe member."
            raise ValueError(msg)
        if int(match.group("ordinal")) != expected_ordinal:
            msg = "Confirmation study snapshot ordinals must be contiguous and start at zero."
            raise ValueError(msg)
        relative_path = f"{CONFIRMATORY_STUDY_DIRECTORY_NAME}/{path.name}"
        payload, _entry = _file_payload_and_entry(context.authorized_output_root, relative_path)
        try:
            snapshot = LockedConfirmatoryStudySnapshot.from_json(payload.decode())
        except (UnicodeDecodeError, TypeError, ValueError) as error:
            msg = f"Confirmation study snapshot {path.name!r} is corrupt."
            raise ValueError(msg) from error
        reference = _snapshot_ref(snapshot, payload)
        if path.name != PurePosixPath(reference.relative_path).name:
            msg = "Confirmation study snapshot filename differs from its content address."
            raise ValueError(msg)
        if snapshot.authorized_output_root != str(context.authorized_output_root):
            msg = "Confirmation study snapshot belongs to another output root."
            raise ValueError(msg)
        if snapshot.session_marker_content_checksum != session_marker_content_checksum:
            msg = "Confirmation study snapshot differs from the verified whole-plan session marker."
            raise ValueError(msg)
        exposure_matches = (
            snapshot.study_manifest.exposure_inventory == exposure_inventory
            if exposure_inventory is not None
            else snapshot.study_manifest.exposure_inventory.content_checksum
            == context.prior_target_exposure_inventory_checksum
        )
        if (
            snapshot.study_manifest.plan != context.plan
            or snapshot.study_manifest.final_seal != context.final_seal
            or snapshot.study_manifest.target_manifest != context.target_manifest
            or not exposure_matches
            or snapshot.study_manifest.execution_source_manifest_checksum
            != context.execution_source_manifest.content_checksum
            or snapshot.study_manifest.analysis_source_manifest_checksum
            != context.analysis_source_manifest.content_checksum
        ):
            msg = "Confirmation study snapshot differs from the exact execution context."
            raise ValueError(msg)
        expected_previous = None if not chain else chain[-1][1]
        if snapshot.previous_snapshot != expected_previous:
            msg = "Confirmation study snapshot chain is missing, reordered, or changed."
            raise ValueError(msg)
        _verify_snapshot_entries(context.authorized_output_root, snapshot.output_entries)
        chain.append((snapshot, reference))
    return tuple(chain)


def _write_snapshot(
    root: Path,
    snapshot: LockedConfirmatoryStudySnapshot,
) -> LockedConfirmatoryStudySnapshotRef:
    payload = f"{snapshot.to_json()}\n".encode()
    reference = _snapshot_ref(snapshot, payload)
    destination = root / reference.relative_path
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
            msg = "Refusing to replace a changed immutable confirmation study snapshot."
            raise ValueError(msg)
        return reference
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".wp22-confirmatory-study-",
        suffix=".tmp",
        dir=root.parent,
    )
    temporary = Path(temporary_name)
    try:
        _stage_snapshot_payload(descriptor, payload)
        _rename_staged_snapshot(temporary, destination, payload)
    finally:
        temporary.unlink(missing_ok=True)
    return reference


def _stage_snapshot_payload(descriptor: int, payload: bytes) -> None:
    """Durably stage exact snapshot bytes outside the scientific output root."""
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _rename_staged_snapshot(temporary: Path, destination: Path, payload: bytes) -> None:
    """Atomically rename a complete staged snapshot into its immutable address.

    Raises:
        ValueError: If an existing immutable address contains other bytes.
    """
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
            msg = "Refusing to replace an existing confirmation study snapshot."
            raise ValueError(msg)
        return
    # Publication runs under the whole-plan lock on the same filesystem.  The
    # rename eliminates the link/unlink crash window that could otherwise leave
    # an immutable snapshot with an external second hard link.
    temporary.rename(destination)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def publish_locked_confirmatory_study_snapshot(
    context: ConfirmationExecutionContext,
    exposure_inventory: PriorTargetExposureInventory,
) -> LockedConfirmatoryStudySnapshotRef:
    """Authenticate the full output universe and append its aggregate snapshot.

    The caller must hold the whole-run lock.  The function is idempotent when
    no row or output receipt changed since the current head.

    Returns:
        The content-addressed head snapshot reference.

    Raises:
        TypeError: If the context or inventory has the wrong protocol type.
        ValueError: If session, study, or output custody is inconsistent.
    """
    if not isinstance(context, ConfirmationExecutionContext):
        msg = "context must be a ConfirmationExecutionContext."
        raise TypeError(msg)
    if not isinstance(exposure_inventory, PriorTargetExposureInventory):
        msg = "exposure_inventory must be a PriorTargetExposureInventory."
        raise TypeError(msg)
    if context.prior_target_exposure_inventory_checksum != exposure_inventory.content_checksum:
        msg = "Exposure inventory differs from the exact confirmation session root."
        raise ValueError(msg)
    marker = validate_confirmation_plan_session(context)
    directory = context.authorized_output_root / CONFIRMATORY_STUDY_DIRECTORY_NAME
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        msg = "Confirmation study path is linked or non-directory."
        raise ValueError(msg)
    directory_was_absent = not directory.exists()
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory_was_absent:
        _fsync_directory(context.authorized_output_root)
    chain = _load_snapshot_chain(context, exposure_inventory, marker.marker_content_checksum)
    external_head = _validate_head_custody_against_chain(context, chain)
    manifest, entries, has_orphan_attempt = _close_current_output(context, exposure_inventory)
    _validate_recoverable_external_publication_gap(
        chain,
        external_head,
        manifest,
        entries,
        has_orphan_attempt=has_orphan_attempt,
    )
    if has_orphan_attempt:
        msg = "An interrupted production attempt must be recovered before a new locked-study snapshot."
        raise ValueError(msg)
    if not chain and manifest.terminal_job_count != 0:
        msg = "The initial locked-study snapshot must precede every confirmatory terminal result."
        raise ValueError(msg)
    if chain and chain[-1][0].study_manifest == manifest and chain[-1][0].output_entries == entries:
        reference = chain[-1][1]
        _publish_locked_study_head_custody(
            context.locked_study_head_custody_path,
            reference,
            external_head,
        )
        return reference
    previous = None if not chain else chain[-1][1]
    ordinal = len(chain)
    inventory_root = canonical_checksum({"entry_checksums": [item.content_checksum for item in entries]})
    snapshot = LockedConfirmatoryStudySnapshot(
        ordinal=ordinal,
        authorized_output_root=str(context.authorized_output_root),
        session_marker_content_checksum=marker.marker_content_checksum,
        previous_snapshot=previous,
        study_manifest=manifest,
        output_entries=entries,
        filesystem_inventory_root=inventory_root,
    )
    reference = _write_snapshot(context.authorized_output_root, snapshot)
    _publish_locked_study_head_custody(
        context.locked_study_head_custody_path,
        reference,
        external_head,
    )
    return reference


def validate_initial_locked_confirmatory_study_snapshot(
    context: ConfirmationExecutionContext,
) -> LockedConfirmatoryStudySnapshotRef:
    """Require the all-unattempted session-bound snapshot before dispatch.

    This is the direct-executor gate.  It prevents a caller with only a valid
    whole-plan marker from materializing any target or producing a scientific
    result before aggregate study custody has been initialized.

    Returns:
        The verified content-addressed initial snapshot reference.

    Raises:
        TypeError: If ``context`` is not a confirmation execution context.
        ValueError: If snapshot zero is absent, unsafe, changed, or post-result.
    """
    if not isinstance(context, ConfirmationExecutionContext):
        msg = "context must be a ConfirmationExecutionContext."
        raise TypeError(msg)
    marker = validate_confirmation_plan_session(context)
    chain = _load_snapshot_chain(context, None, marker.marker_content_checksum)
    if not chain:
        msg = "Real confirmation requires exactly one content-addressed initial study snapshot."
        raise ValueError(msg)
    snapshot, reference = chain[0]
    _validate_initial_snapshot_identity(context, marker.marker_content_checksum, snapshot, reference)
    external_head = _validate_head_custody_against_chain(context, chain)
    if external_head is None:
        msg = "Real confirmation requires externally published locked-study head custody before dispatch."
        raise ValueError(msg)
    return reference


def _validate_initial_snapshot_identity(
    context: ConfirmationExecutionContext,
    marker_content_checksum: str,
    snapshot: LockedConfirmatoryStudySnapshot,
    reference: LockedConfirmatoryStudySnapshotRef,
) -> None:
    """Bind snapshot zero to the exact session, context, and empty study.

    Raises:
        ValueError: If any initial snapshot identity or count differs.
    """
    manifest = snapshot.study_manifest
    if (
        snapshot.ordinal != 0
        or snapshot.previous_snapshot is not None
        or snapshot.authorized_output_root != str(context.authorized_output_root)
        or snapshot.session_marker_content_checksum != marker_content_checksum
        or PurePosixPath(reference.relative_path).name
        != PurePosixPath(_snapshot_relative_path(0, snapshot.content_checksum)).name
        or manifest.plan != context.plan
        or manifest.final_seal != context.final_seal
        or manifest.target_manifest != context.target_manifest
        or manifest.exposure_inventory.content_checksum != context.prior_target_exposure_inventory_checksum
        or manifest.execution_source_manifest_checksum != context.execution_source_manifest.content_checksum
        or manifest.analysis_source_manifest_checksum != context.analysis_source_manifest.content_checksum
        or manifest.terminal_job_count != 0
        or manifest.unattempted_job_count != len(context.plan.jobs)
        or manifest.status != "incomplete"
    ):
        msg = "Initial locked-study snapshot differs from the all-unattempted confirmation session."
        raise ValueError(msg)


def validate_locked_confirmatory_study_output(
    context: ConfirmationExecutionContext,
    exposure_inventory: PriorTargetExposureInventory,
    expected_head: LockedConfirmatoryStudySnapshotRef | None = None,
) -> LockedConfirmatoryStudySnapshotRef | None:
    """Read-only validate an existing session, snapshot chain, and output tree.

    Returns:
        The verified head reference, or ``None`` for an absent root or a
        marker-only pre-snapshot initialization crash.

    Raises:
        TypeError: If the expected head has the wrong protocol type.
        ValueError: If existing session, snapshot, or output custody changed.
    """
    if not context.authorized_output_root.exists():
        if expected_head is not None:
            msg = "Externally retained study head exists but the confirmation output root is absent."
            raise ValueError(msg)
        return None
    if expected_head is not None and not isinstance(expected_head, LockedConfirmatoryStudySnapshotRef):
        msg = "expected_head must be a LockedConfirmatoryStudySnapshotRef."
        raise TypeError(msg)
    if not any(context.authorized_output_root.iterdir()):
        if expected_head is not None:
            msg = "Externally retained study head exists but the confirmation output root is empty."
            raise ValueError(msg)
        return None
    marker = validate_confirmation_plan_session(context)
    directory = context.authorized_output_root / CONFIRMATORY_STUDY_DIRECTORY_NAME
    if not directory.exists():
        if expected_head is not None:
            msg = "Externally retained study head exists but the snapshot directory is absent."
            raise ValueError(msg)
        allowed = {CONFIRMATION_PLAN_SESSION_NAME}
        existing = {path.name for path in context.authorized_output_root.iterdir()}
        if existing <= allowed:
            return None
        msg = "Confirmation-owned state exists after removal or before creation of snapshot custody."
        raise ValueError(msg)
    chain = _load_snapshot_chain(context, exposure_inventory, marker.marker_content_checksum)
    external_head = _validate_head_custody_against_chain(context, chain)
    manifest, entries, has_orphan_attempt = _close_current_output(context, exposure_inventory)
    _validate_recoverable_external_publication_gap(
        chain,
        external_head,
        manifest,
        entries,
        has_orphan_attempt=has_orphan_attempt,
    )
    if not chain:
        if expected_head is not None:
            msg = "Externally retained study head exists but the snapshot chain is empty."
            raise ValueError(msg)
        initialization_entries = {
            (CONFIRMATION_PLAN_SESSION_NAME, "file"),
            (CONFIRMATORY_STUDY_DIRECTORY_NAME, "directory"),
        }
        actual_entries = {(entry.relative_path, entry.entry_kind) for entry in entries}
        if manifest.terminal_job_count == 0 and not has_orphan_attempt and actual_entries == initialization_entries:
            return None
        msg = "An established confirmation study directory lacks its initial snapshot."
        raise ValueError(msg)
    references = tuple(reference for _snapshot, reference in chain)
    head, reference = chain[-1]
    if expected_head is not None and expected_head not in references:
        msg = "Externally retained trusted head is not a verified ancestor of the on-disk study head."
        raise ValueError(msg)
    if expected_head is not None and (external_head is None or external_head.ordinal < expected_head.ordinal):
        msg = "External locked-study head custody rolled back behind the trusted expected head."
        raise ValueError(msg)
    _validate_snapshot_extension(head, manifest, entries)
    return reference


def _republish_current_locked_confirmatory_study_head(
    context: ConfirmationExecutionContext,
    exposure_inventory: PriorTargetExposureInventory,
    expected_head: LockedConfirmatoryStudySnapshotRef | None,
) -> LockedConfirmatoryStudySnapshotRef:
    """Repair one validated external-head publication gap without appending.

    The caller must hold the whole-run lock. Full output validation admits the
    bounded first-unattempted orphan state, while the external-head validator
    proves that custody is either already current or exactly one contiguous
    snapshot behind. This helper never chooses or creates a snapshot branch.

    Returns:
        The fully validated existing snapshot-chain tip.

    Raises:
        ValueError: If custody is absent, branched, rolled back by more than
            one snapshot, or otherwise differs from the exact output universe.
    """
    reference = validate_locked_confirmatory_study_output(
        context,
        exposure_inventory,
        expected_head,
    )
    if reference is None:
        msg = "Cannot repair external head custody without an existing locked-study snapshot."
        raise ValueError(msg)
    external_head = _load_locked_study_head_custody(context.locked_study_head_custody_path)
    if external_head is None:
        msg = "External head repair requires a validated current or immediate-predecessor reference."
        raise ValueError(msg)
    if external_head != reference and external_head.ordinal + 1 != reference.ordinal:
        msg = "External head repair cannot cross more than one contiguous snapshot publication gap."
        raise ValueError(msg)
    _publish_locked_study_head_custody(
        context.locked_study_head_custody_path,
        reference,
        external_head,
    )
    return reference


def confirmation_output_has_interrupted_attempt(
    context: ConfirmationExecutionContext,
    exposure_inventory: PriorTargetExposureInventory,
) -> bool:
    """Return whether the first unattempted cell has recoverable production state.

    The caller must first validate the exact output tree under the whole-run
    lock.  This narrow query lets orchestration require an explicit resume
    request before authentic interrupted-attempt recovery.

    Returns:
        Whether an interrupted canonical first attempt is present.
    """
    _manifest, _files, _directories, has_orphan_attempt = _collect_study(
        context,
        exposure_inventory,
    )
    return has_orphan_attempt


def _validate_snapshot_extension(
    head: LockedConfirmatoryStudySnapshot,
    current_manifest: LockedConfirmatoryStudyManifest,
    current_entries: Sequence[ConfirmatoryOutputEntry],
) -> None:
    """Require current output to be an append-only canonical extension of head.

    Raises:
        ValueError: If a previously terminal row changed or an old receipt disappeared.
    """
    if head.study_manifest.plan != current_manifest.plan:
        msg = "Current confirmation study plan differs from the locked snapshot head."
        raise ValueError(msg)
    for prior, current in zip(head.study_manifest.rows, current_manifest.rows, strict=True):
        if prior.terminal_state != "unattempted" and prior != current:
            msg = "A previously terminal confirmatory study row changed after snapshot publication."
            raise ValueError(msg)
    current_by_path = {entry.relative_path: entry for entry in current_entries}
    if any(current_by_path.get(entry.relative_path) != entry for entry in head.output_entries):
        msg = "A previously snapshotted confirmation output receipt disappeared or changed."
        raise ValueError(msg)


def confirmation_output_has_owned_state(output_root: Path) -> bool:
    """Return whether a root contains any confirmation-owned state.

    Returns:
        ``True`` when a session, study, job, temporary, or foreign member exists.

    Raises:
        TypeError: If ``output_root`` is not a path.
        ValueError: If the existing root is linked or non-directory.
    """
    if not isinstance(output_root, Path):
        msg = "output_root must be a pathlib.Path."
        raise TypeError(msg)
    if not output_root.exists():
        return False
    if output_root.is_symlink() or not output_root.is_dir():
        msg = "Confirmation output root is linked or non-directory."
        raise ValueError(msg)
    return any(output_root.iterdir())


def _fsync_directory(directory: Path) -> None:
    """Durably publish directory-entry changes within ``directory``."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CONFIRMATORY_OUTPUT_ENTRY_SCHEMA_VERSION",
    "CONFIRMATORY_STUDY_DIRECTORY_NAME",
    "LOCKED_CONFIRMATORY_STUDY_SNAPSHOT_REF_SCHEMA_VERSION",
    "LOCKED_CONFIRMATORY_STUDY_SNAPSHOT_SCHEMA_VERSION",
    "ConfirmatoryOutputEntry",
    "LockedConfirmatoryStudySnapshot",
    "LockedConfirmatoryStudySnapshotRef",
    "confirmation_output_has_interrupted_attempt",
    "confirmation_output_has_owned_state",
    "publish_locked_confirmatory_study_snapshot",
    "validate_initial_locked_confirmatory_study_snapshot",
    "validate_locked_confirmatory_study_output",
]
