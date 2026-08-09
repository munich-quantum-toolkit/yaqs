# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Atomic append-only artifact bundles for the WP22H ceremony.

The store is deliberately independent of scientific artifact types.  A caller
supplies immutable member bytes and a checksum-linked stage manifest; this
module closes those bytes into one exact directory whose terminal bundle index
is written last.  Complete directories are published by a same-filesystem
rename, so a crash exposes either no bundle or one fully reopenable bundle.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

from filelock import FileLock

from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .validation import require_checksum, require_int, require_relative_path, require_slug

if TYPE_CHECKING:
    from collections.abc import Sequence

CEREMONY_MEMBER_RECEIPT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.ceremony_member_receipt.v1"
CEREMONY_STAGE_MANIFEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.ceremony_stage_manifest.v1"
CEREMONY_BUNDLE_INDEX_SCHEMA_VERSION = "yaqs.state_preparation.phase2.ceremony_bundle_index.v1"

CEREMONY_STAGE_MANIFEST_NAME = "stage_manifest.json"
CEREMONY_BUNDLE_INDEX_NAME = "bundle_index.json"

_RESERVED_MEMBER_PATHS = frozenset({CEREMONY_STAGE_MANIFEST_NAME, CEREMONY_BUNDLE_INDEX_NAME})
_RECEIPT_KEYS = frozenset({
    "schema_version",
    "relative_path",
    "role",
    "byte_count",
    "file_checksum",
    "content_checksum",
})
_STAGE_MANIFEST_KEYS = frozenset({
    "schema_version",
    "ceremony_id",
    "stage_id",
    "stage_ordinal",
    "predecessor_stage_manifest_checksum",
    "members",
    "member_inventory_checksum",
    "content_checksum",
})
_BUNDLE_INDEX_KEYS = frozenset({
    "schema_version",
    "ceremony_id",
    "stage_id",
    "stage_ordinal",
    "predecessor_stage_manifest_checksum",
    "stage_manifest_receipt",
    "stage_manifest_content_checksum",
    "bundle_inventory_checksum",
    "content_checksum",
})


def _is_reserved_member_path(relative_path: str) -> bool:
    """Return whether a member claims or descends from a store custody file."""
    path = PurePosixPath(relative_path)
    return any(
        path == PurePosixPath(reserved) or PurePosixPath(reserved) in path.parents
        for reserved in _RESERVED_MEMBER_PATHS
    )


def _sha256_bytes(payload: bytes) -> str:
    """Return the prefixed SHA-256 checksum of exact file bytes."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _inventory_checksum(receipts: Sequence[CeremonyMemberReceipt]) -> str:
    """Return the order-sensitive checksum of one canonical receipt inventory."""
    return canonical_checksum({"receipt_checksums": [receipt.content_checksum for receipt in receipts]})


@dataclass(frozen=True, slots=True)
class CeremonyMemberReceipt:
    """Path-bound receipt for one immutable regular single-link bundle member."""

    relative_path: str
    role: str
    byte_count: int
    file_checksum: str
    schema_version: str = field(default=CEREMONY_MEMBER_RECEIPT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the canonical path, role, size, and byte checksum."""
        relative_path = require_relative_path(self.relative_path, "relative_path")
        object.__setattr__(self, "relative_path", relative_path)
        object.__setattr__(self, "role", require_slug(self.role, "role"))
        object.__setattr__(self, "byte_count", require_int(self.byte_count, "byte_count", minimum=0))
        object.__setattr__(self, "file_checksum", require_checksum(self.file_checksum, "file_checksum"))

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered receipt field."""
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "role": self.role,
            "byte_count": self.byte_count,
            "file_checksum": self.file_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the member's path, role, size, and exact byte checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed JSON-native receipt."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> CeremonyMemberReceipt:
        """Decode and checksum-verify one member receipt.

        Returns:
            The verified immutable member receipt.

        Raises:
            ValueError: If the schema, fields, or checksum are invalid.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_RECEIPT_KEYS, name="ceremony member receipt")
        if mapping["schema_version"] != CEREMONY_MEMBER_RECEIPT_SCHEMA_VERSION:
            msg = "Ceremony member receipt uses an unsupported schema version."
            raise ValueError(msg)
        receipt = cls(
            relative_path=cast("str", mapping["relative_path"]),
            role=cast("str", mapping["role"]),
            byte_count=cast("int", mapping["byte_count"]),
            file_checksum=cast("str", mapping["file_checksum"]),
        )
        if mapping["content_checksum"] != receipt.content_checksum:
            msg = "Ceremony member receipt checksum changed during normalization."
            raise ValueError(msg)
        return receipt

    @classmethod
    def from_json(cls, payload: str) -> CeremonyMemberReceipt:
        """Decode a canonical JSON receipt.

        Returns:
            The verified immutable member receipt.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class CeremonyBundleMember:
    """In-memory bytes and public role supplied for one bundle publication."""

    relative_path: str
    role: str
    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        """Validate and detach the exact member bytes.

        Raises:
            TypeError: If the payload is not exact bytes.
            ValueError: If the member claims a reserved custody path.
        """
        if type(self.payload) is not bytes:
            msg = f"payload must be bytes, got {type(self.payload).__name__}."
            raise TypeError(msg)
        payload = bytes(self.payload)
        receipt = CeremonyMemberReceipt(
            relative_path=self.relative_path,
            role=self.role,
            byte_count=len(payload),
            file_checksum=_sha256_bytes(payload),
        )
        if _is_reserved_member_path(receipt.relative_path):
            msg = f"Ceremony member path {receipt.relative_path!r} is reserved for store custody."
            raise ValueError(msg)
        object.__setattr__(self, "relative_path", receipt.relative_path)
        object.__setattr__(self, "role", receipt.role)
        object.__setattr__(self, "payload", payload)

    @property
    def receipt(self) -> CeremonyMemberReceipt:
        """Deterministic receipt for these exact bytes."""
        return CeremonyMemberReceipt(
            relative_path=self.relative_path,
            role=self.role,
            byte_count=len(self.payload),
            file_checksum=_sha256_bytes(self.payload),
        )


def _validate_receipt_inventory(receipts: Sequence[CeremonyMemberReceipt]) -> tuple[CeremonyMemberReceipt, ...]:
    """Return a canonical collision-free member inventory.

    Returns:
        Receipts sorted by relative path.

    Raises:
        TypeError: If a member is not a receipt.
        ValueError: If paths are duplicated, unsorted, or collide as file and directory.
    """
    normalized = tuple(receipts)
    if not normalized:
        msg = "A ceremony stage must contain at least one immutable member."
        raise ValueError(msg)
    if any(not isinstance(receipt, CeremonyMemberReceipt) for receipt in normalized):
        msg = "Every ceremony stage member must be a CeremonyMemberReceipt."
        raise TypeError(msg)
    ordered = tuple(sorted(normalized, key=lambda receipt: receipt.relative_path))
    if normalized != ordered:
        msg = "Ceremony stage member receipts must use canonical relative-path order."
        raise ValueError(msg)
    paths = tuple(PurePosixPath(receipt.relative_path) for receipt in ordered)
    if any(_is_reserved_member_path(receipt.relative_path) for receipt in ordered):
        msg = "Ceremony stage member inventory cannot claim a reserved custody path."
        raise ValueError(msg)
    if len(paths) != len(set(paths)):
        msg = "Ceremony stage member paths must be unique."
        raise ValueError(msg)
    for index, path in enumerate(paths):
        if any(path in candidate.parents for candidate in paths[index + 1 :]):
            msg = "A ceremony member path cannot also be another member's directory."
            raise ValueError(msg)
    return ordered


@dataclass(frozen=True, slots=True)
class CeremonyStageManifest:
    """Immutable ceremony stage identity and exact member inventory."""

    ceremony_id: str
    stage_id: str
    stage_ordinal: int
    predecessor_stage_manifest_checksum: str | None
    members: tuple[CeremonyMemberReceipt, ...]
    member_inventory_checksum: str
    schema_version: str = field(default=CEREMONY_STAGE_MANIFEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate stage ordering, predecessor identity, and exact inventory.

        Raises:
            ValueError: If stage identity, ordering, or inventory is invalid.
        """
        object.__setattr__(self, "ceremony_id", require_slug(self.ceremony_id, "ceremony_id"))
        object.__setattr__(self, "stage_id", require_slug(self.stage_id, "stage_id"))
        ordinal = require_int(self.stage_ordinal, "stage_ordinal", minimum=0)
        object.__setattr__(self, "stage_ordinal", ordinal)
        predecessor = self.predecessor_stage_manifest_checksum
        if ordinal == 0:
            if predecessor is not None:
                msg = "The initial ceremony stage cannot name a predecessor."
                raise ValueError(msg)
        elif predecessor is None:
            msg = "Every noninitial ceremony stage requires its exact predecessor checksum."
            raise ValueError(msg)
        else:
            predecessor = require_checksum(predecessor, "predecessor_stage_manifest_checksum")
        object.__setattr__(self, "predecessor_stage_manifest_checksum", predecessor)
        members = _validate_receipt_inventory(self.members)
        object.__setattr__(self, "members", members)
        inventory_checksum = require_checksum(self.member_inventory_checksum, "member_inventory_checksum")
        if inventory_checksum != _inventory_checksum(members):
            msg = "Ceremony stage member inventory checksum is inconsistent."
            raise ValueError(msg)
        object.__setattr__(self, "member_inventory_checksum", inventory_checksum)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered stage field."""
        return {
            "schema_version": self.schema_version,
            "ceremony_id": self.ceremony_id,
            "stage_id": self.stage_id,
            "stage_ordinal": self.stage_ordinal,
            "predecessor_stage_manifest_checksum": self.predecessor_stage_manifest_checksum,
            "members": [member.to_dict() for member in self.members],
            "member_inventory_checksum": self.member_inventory_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the stage identity, predecessor, and member inventory."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed JSON-native stage manifest."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> CeremonyStageManifest:
        """Decode and checksum-verify a ceremony stage manifest.

        Returns:
            The verified immutable stage manifest.

        Raises:
            TypeError: If serialized members do not form a sequence.
            ValueError: If the schema, fields, inventory, or checksum are invalid.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_STAGE_MANIFEST_KEYS, name="ceremony stage manifest")
        if mapping["schema_version"] != CEREMONY_STAGE_MANIFEST_SCHEMA_VERSION:
            msg = "Ceremony stage manifest uses an unsupported schema version."
            raise ValueError(msg)
        raw_members = mapping["members"]
        if type(raw_members) is not tuple:
            msg = "Ceremony stage members must be a serialized sequence."
            raise TypeError(msg)
        manifest = cls(
            ceremony_id=cast("str", mapping["ceremony_id"]),
            stage_id=cast("str", mapping["stage_id"]),
            stage_ordinal=cast("int", mapping["stage_ordinal"]),
            predecessor_stage_manifest_checksum=cast("str | None", mapping["predecessor_stage_manifest_checksum"]),
            members=tuple(CeremonyMemberReceipt.from_dict(member) for member in raw_members),
            member_inventory_checksum=cast("str", mapping["member_inventory_checksum"]),
        )
        if mapping["content_checksum"] != manifest.content_checksum:
            msg = "Ceremony stage manifest checksum changed during normalization."
            raise ValueError(msg)
        return manifest

    @classmethod
    def from_json(cls, payload: str) -> CeremonyStageManifest:
        """Decode a canonical JSON ceremony stage manifest.

        Returns:
            The verified immutable stage manifest.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def build_ceremony_stage_manifest(
    ceremony_id: str,
    stage_id: str,
    members: Sequence[CeremonyBundleMember],
    *,
    predecessor: CeremonyStageManifest | None = None,
) -> CeremonyStageManifest:
    """Build the next exact stage manifest from immutable member bytes.

    Returns:
        The checksum-linked initial or successor stage manifest.

    Raises:
        TypeError: If a member or predecessor has the wrong type.
    """
    supplied = tuple(members)
    if any(not isinstance(member, CeremonyBundleMember) for member in supplied):
        msg = "Every supplied ceremony member must be a CeremonyBundleMember."
        raise TypeError(msg)
    if predecessor is not None and not isinstance(predecessor, CeremonyStageManifest):
        msg = "predecessor must be a CeremonyStageManifest or None."
        raise TypeError(msg)
    receipts = tuple(sorted((member.receipt for member in supplied), key=lambda receipt: receipt.relative_path))
    ordinal = 0 if predecessor is None else predecessor.stage_ordinal + 1
    predecessor_checksum = None if predecessor is None else predecessor.content_checksum
    manifest = CeremonyStageManifest(
        ceremony_id=ceremony_id,
        stage_id=stage_id,
        stage_ordinal=ordinal,
        predecessor_stage_manifest_checksum=predecessor_checksum,
        members=receipts,
        member_inventory_checksum=_inventory_checksum(receipts),
    )
    if predecessor is not None:
        validate_ceremony_stage_transition(predecessor, manifest)
    return manifest


def validate_ceremony_stage_transition(
    predecessor: CeremonyStageManifest,
    successor: CeremonyStageManifest,
) -> None:
    """Require one exact contiguous transition in a single ceremony.

    Raises:
        TypeError: If either stage has the wrong type.
        ValueError: If ceremony, ordinal, or predecessor identity differs.
    """
    if not isinstance(predecessor, CeremonyStageManifest) or not isinstance(successor, CeremonyStageManifest):
        msg = "Ceremony stage transitions require two CeremonyStageManifest objects."
        raise TypeError(msg)
    if (
        successor.ceremony_id != predecessor.ceremony_id
        or successor.stage_ordinal != predecessor.stage_ordinal + 1
        or successor.predecessor_stage_manifest_checksum != predecessor.content_checksum
    ):
        msg = "Ceremony stages do not form one exact contiguous predecessor chain."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CeremonyBundleIndex:
    """Terminal commit record for one complete immutable ceremony bundle."""

    ceremony_id: str
    stage_id: str
    stage_ordinal: int
    predecessor_stage_manifest_checksum: str | None
    stage_manifest_receipt: CeremonyMemberReceipt
    stage_manifest_content_checksum: str
    bundle_inventory_checksum: str
    schema_version: str = field(default=CEREMONY_BUNDLE_INDEX_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the terminal index and its stage-manifest receipt.

        Raises:
            TypeError: If the stage-manifest receipt has the wrong type.
            ValueError: If stage identity, ordering, receipts, or checksums are invalid.
        """
        object.__setattr__(self, "ceremony_id", require_slug(self.ceremony_id, "ceremony_id"))
        object.__setattr__(self, "stage_id", require_slug(self.stage_id, "stage_id"))
        ordinal = require_int(self.stage_ordinal, "stage_ordinal", minimum=0)
        object.__setattr__(self, "stage_ordinal", ordinal)
        predecessor = self.predecessor_stage_manifest_checksum
        if predecessor is not None:
            predecessor = require_checksum(predecessor, "predecessor_stage_manifest_checksum")
        if (ordinal == 0) != (predecessor is None):
            msg = "Ceremony bundle index predecessor identity differs from its stage ordinal."
            raise ValueError(msg)
        object.__setattr__(self, "predecessor_stage_manifest_checksum", predecessor)
        if not isinstance(self.stage_manifest_receipt, CeremonyMemberReceipt):
            msg = "stage_manifest_receipt must be a CeremonyMemberReceipt."
            raise TypeError(msg)
        if (
            self.stage_manifest_receipt.relative_path != CEREMONY_STAGE_MANIFEST_NAME
            or self.stage_manifest_receipt.role != "ceremony-stage-manifest"
        ):
            msg = "Bundle index must receipt the canonical ceremony stage manifest."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "stage_manifest_content_checksum",
            require_checksum(self.stage_manifest_content_checksum, "stage_manifest_content_checksum"),
        )
        object.__setattr__(
            self,
            "bundle_inventory_checksum",
            require_checksum(self.bundle_inventory_checksum, "bundle_inventory_checksum"),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered terminal-index field."""
        return {
            "schema_version": self.schema_version,
            "ceremony_id": self.ceremony_id,
            "stage_id": self.stage_id,
            "stage_ordinal": self.stage_ordinal,
            "predecessor_stage_manifest_checksum": self.predecessor_stage_manifest_checksum,
            "stage_manifest_receipt": self.stage_manifest_receipt.to_dict(),
            "stage_manifest_content_checksum": self.stage_manifest_content_checksum,
            "bundle_inventory_checksum": self.bundle_inventory_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact terminal bundle commitment."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed JSON-native terminal index."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> CeremonyBundleIndex:
        """Decode and checksum-verify a ceremony bundle index.

        Returns:
            The verified immutable terminal index.

        Raises:
            ValueError: If the schema, fields, receipts, or checksum are invalid.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_BUNDLE_INDEX_KEYS, name="ceremony bundle index")
        if mapping["schema_version"] != CEREMONY_BUNDLE_INDEX_SCHEMA_VERSION:
            msg = "Ceremony bundle index uses an unsupported schema version."
            raise ValueError(msg)
        index = cls(
            ceremony_id=cast("str", mapping["ceremony_id"]),
            stage_id=cast("str", mapping["stage_id"]),
            stage_ordinal=cast("int", mapping["stage_ordinal"]),
            predecessor_stage_manifest_checksum=cast("str | None", mapping["predecessor_stage_manifest_checksum"]),
            stage_manifest_receipt=CeremonyMemberReceipt.from_dict(mapping["stage_manifest_receipt"]),
            stage_manifest_content_checksum=cast("str", mapping["stage_manifest_content_checksum"]),
            bundle_inventory_checksum=cast("str", mapping["bundle_inventory_checksum"]),
        )
        if mapping["content_checksum"] != index.content_checksum:
            msg = "Ceremony bundle index checksum changed during normalization."
            raise ValueError(msg)
        return index

    @classmethod
    def from_json(cls, payload: str) -> CeremonyBundleIndex:
        """Decode a canonical JSON ceremony bundle index.

        Returns:
            The verified immutable terminal index.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class ReopenedCeremonyBundle:
    """Fully authenticated immutable view of one ceremony stage bundle."""

    bundle_directory: Path
    manifest: CeremonyStageManifest
    index: CeremonyBundleIndex


def _canonical_bundle_directory(bundle_directory: Path) -> Path:
    """Return one absolute, non-symlink bundle path with a safe parent.

    Returns:
        The validated absolute path.

    Raises:
        TypeError: If the path has the wrong type.
        ValueError: If the path is relative, linked, noncanonical, or has an unsafe parent.
    """
    if not isinstance(bundle_directory, Path):
        msg = f"bundle_directory must be a pathlib.Path, got {type(bundle_directory).__name__}."
        raise TypeError(msg)
    if not bundle_directory.is_absolute():
        msg = "bundle_directory must be an absolute path."
        raise ValueError(msg)
    absolute = bundle_directory.absolute()
    if absolute.resolve() != absolute:
        msg = "bundle_directory cannot contain symlink or noncanonical components."
        raise ValueError(msg)
    parent_metadata = absolute.parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        msg = "Ceremony bundle parent must be a non-symlink directory."
        raise ValueError(msg)
    return absolute


def _member_path(root: Path, relative_path: str) -> Path:
    """Resolve a validated POSIX member path below one bundle root.

    Returns:
        The local filesystem member path.
    """
    return root.joinpath(*PurePosixPath(relative_path).parts)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return identity and mutation timestamps required by pinned reads."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_pinned_regular_file(path: Path, label: str) -> bytes:
    """Read a stable single-link regular file without following its final link.

    Returns:
        Exact immutable file bytes.

    Raises:
        ValueError: If the file is absent, linked, non-regular, or changes while read.
    """
    try:
        before = path.lstat()
    except OSError as error:
        msg = f"{label} is missing or unavailable."
        raise ValueError(msg) from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        msg = f"{label} must be a single-link regular file."
        raise ValueError(msg)
    flags = os.O_RDONLY | cast("int", getattr(os, "O_NOFOLLOW", 0)) | cast("int", getattr(os, "O_NONBLOCK", 0))
    descriptor = os.open(path, flags)
    try:
        payload, closed = _read_verified_descriptor(descriptor, before, label)
    finally:
        os.close(descriptor)
    after = path.lstat()
    before_identity = _file_identity(before)
    closed_identity = _file_identity(closed)
    after_identity = _file_identity(after)
    if closed_identity != before_identity or after_identity != before_identity:
        msg = f"{label} changed while it was read."
        raise ValueError(msg)
    return payload


def _read_verified_descriptor(
    descriptor: int,
    before: os.stat_result,
    label: str,
) -> tuple[bytes, os.stat_result]:
    """Read one descriptor after matching it to its pre-open identity.

    Returns:
        Exact bytes and the post-read descriptor metadata.

    Raises:
        ValueError: If the opened descriptor differs from the expected file.
    """
    opened = os.fstat(descriptor)
    before_identity = _file_identity(before)
    opened_identity = _file_identity(opened)
    if not stat.S_ISREG(opened.st_mode) or opened_identity != before_identity:
        msg = f"{label} changed while it was opened."
        raise ValueError(msg)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks), os.fstat(descriptor)


def _expected_directories(paths: Sequence[str]) -> set[str]:
    """Return every expected relative directory ancestor for file paths."""
    directories: set[str] = set()
    for value in paths:
        for parent in PurePosixPath(value).parents:
            if str(parent) != ".":
                directories.add(str(parent))
    return directories


def _scan_bundle_tree(root: Path) -> tuple[set[str], set[str]]:
    """Inventory all bundle paths without following links or opening special files.

    Returns:
        Exact relative regular-file and directory path sets.

    Raises:
        ValueError: If the tree contains a link, special file, or multiply linked file.
    """
    root_metadata = root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        msg = "Ceremony bundle root must be a non-symlink directory."
        raise ValueError(msg)
    files: set[str] = set()
    directories: set[str] = set()
    pending = [(root, PurePosixPath())]
    while pending:
        directory, relative_directory = pending.pop()
        for child in directory.iterdir():
            relative = PurePosixPath(child.name) if str(relative_directory) == "." else relative_directory / child.name
            metadata = child.lstat()
            relative_text = str(relative)
            if stat.S_ISLNK(metadata.st_mode):
                msg = f"Ceremony bundle contains linked member {relative_text!r}."
                raise ValueError(msg)
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative_text)
                pending.append((child, relative))
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                files.add(relative_text)
            else:
                msg = f"Ceremony bundle contains special or multiply linked member {relative_text!r}."
                raise ValueError(msg)
    return files, directories


def _verify_receipted_file(root: Path, receipt: CeremonyMemberReceipt) -> bytes:
    """Read and verify one exact path-bound member receipt.

    Returns:
        Exact authenticated member bytes.

    Raises:
        ValueError: If the member differs from its receipt.
    """
    payload = _read_pinned_regular_file(_member_path(root, receipt.relative_path), receipt.relative_path)
    if len(payload) != receipt.byte_count or _sha256_bytes(payload) != receipt.file_checksum:
        msg = f"Ceremony bundle member {receipt.relative_path!r} differs from its immutable receipt."
        raise ValueError(msg)
    return payload


def reopen_ceremony_bundle(
    bundle_directory: Path,
    *,
    expected_index_checksum: str | None = None,
    expected_stage_manifest_checksum: str | None = None,
) -> ReopenedCeremonyBundle:
    """Reopen and authenticate one complete ceremony stage bundle.

    Returns:
        The fully verified manifest and terminal bundle index.

    Raises:
        ValueError: If any schema, checksum, member, type, link, or inventory differs.
    """
    root = _canonical_bundle_directory(bundle_directory)
    index_payload = _read_pinned_regular_file(root / CEREMONY_BUNDLE_INDEX_NAME, "Ceremony bundle index")
    try:
        index = CeremonyBundleIndex.from_json(index_payload.decode())
    except (TypeError, UnicodeError, ValueError) as error:
        msg = "Ceremony bundle index is invalid or noncanonical."
        raise ValueError(msg) from error
    if index_payload != f"{index.to_json()}\n".encode():
        msg = "Ceremony bundle index bytes differ from the one canonical terminal encoding."
        raise ValueError(msg)
    if expected_index_checksum is not None and index.content_checksum != require_checksum(
        expected_index_checksum,
        "expected_index_checksum",
    ):
        msg = "Ceremony bundle index differs from the expected terminal custody."
        raise ValueError(msg)
    manifest_payload = _verify_receipted_file(root, index.stage_manifest_receipt)
    try:
        manifest = CeremonyStageManifest.from_json(manifest_payload.decode())
    except (TypeError, UnicodeError, ValueError) as error:
        msg = "Ceremony stage manifest is invalid or noncanonical."
        raise ValueError(msg) from error
    _validate_index_manifest_binding(index, manifest)
    if expected_stage_manifest_checksum is not None and manifest.content_checksum != require_checksum(
        expected_stage_manifest_checksum,
        "expected_stage_manifest_checksum",
    ):
        msg = "Ceremony stage manifest differs from the expected predecessor or stage custody."
        raise ValueError(msg)
    expected_files = {
        CEREMONY_STAGE_MANIFEST_NAME,
        CEREMONY_BUNDLE_INDEX_NAME,
        *(receipt.relative_path for receipt in manifest.members),
    }
    expected_directories = _expected_directories(tuple(expected_files))
    actual_files, actual_directories = _scan_bundle_tree(root)
    if actual_files != expected_files or actual_directories != expected_directories:
        msg = (
            "Ceremony bundle differs from its exact immutable inventory: "
            f"extra_files={sorted(actual_files - expected_files)!r}, "
            f"missing_files={sorted(expected_files - actual_files)!r}, "
            f"extra_directories={sorted(actual_directories - expected_directories)!r}, "
            f"missing_directories={sorted(expected_directories - actual_directories)!r}."
        )
        raise ValueError(msg)
    for receipt in manifest.members:
        _verify_receipted_file(root, receipt)
    return ReopenedCeremonyBundle(root, manifest, index)


def _validate_index_manifest_binding(index: CeremonyBundleIndex, manifest: CeremonyStageManifest) -> None:
    """Require a terminal index to identify its one exact stage manifest.

    Raises:
        ValueError: If any repeated stage or inventory identity differs.
    """
    receipts = (*manifest.members, index.stage_manifest_receipt)
    if (
        index.ceremony_id != manifest.ceremony_id
        or index.stage_id != manifest.stage_id
        or index.stage_ordinal != manifest.stage_ordinal
        or index.predecessor_stage_manifest_checksum != manifest.predecessor_stage_manifest_checksum
        or index.stage_manifest_content_checksum != manifest.content_checksum
        or index.bundle_inventory_checksum != _inventory_checksum(receipts)
    ):
        msg = "Ceremony bundle index differs from its exact stage manifest or inventory."
        raise ValueError(msg)


def read_ceremony_bundle_member(reopened: ReopenedCeremonyBundle, relative_path: str) -> bytes:
    """Read one manifest-owned member through its authenticated pinned receipt.

    The bundle is reopened against the caller's terminal index and stage
    checksums before the selected member is returned.  A stale reopened object
    therefore cannot authorize reads after any filesystem mutation.

    Returns:
        Exact bytes of the requested immutable member.

    Raises:
        TypeError: If ``reopened`` has the wrong type.
        ValueError: If the path is absent or any bundle custody has changed.
    """
    if not isinstance(reopened, ReopenedCeremonyBundle):
        msg = "reopened must be a ReopenedCeremonyBundle."
        raise TypeError(msg)
    requested = require_relative_path(relative_path, "relative_path")
    current = reopen_ceremony_bundle(
        reopened.bundle_directory,
        expected_index_checksum=reopened.index.content_checksum,
        expected_stage_manifest_checksum=reopened.manifest.content_checksum,
    )
    matches = tuple(receipt for receipt in current.manifest.members if receipt.relative_path == requested)
    if len(matches) != 1:
        msg = f"Ceremony bundle manifest does not own member {requested!r}."
        raise ValueError(msg)
    return _verify_receipted_file(current.bundle_directory, matches[0])


def _write_exclusive_file(path: Path, payload: bytes) -> None:
    """Create, fsync, and close one new file without overwrite."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    """Durably close directory-entry publication where supported."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish one sibling directory without replacing a name.

    Raises:
        FileExistsError: If the destination appeared before publication.
        OSError: If the host lacks a safe no-replace rename primitive or the
            same-filesystem publication fails.
        ValueError: If source and destination are not sibling directories.
    """
    if source.parent != destination.parent:
        msg = "Ceremony bundle staging and publication paths must be siblings."
        raise ValueError(msg)
    directory_flags = os.O_RDONLY | cast("int", getattr(os, "O_DIRECTORY", 0))
    parent_descriptor = os.open(source.parent, directory_flags)
    try:
        result = _call_atomic_rename_no_replace(parent_descriptor, source.name, destination.name)
    finally:
        os.close(parent_descriptor)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        msg = "Immutable ceremony bundle already exists."
        raise FileExistsError(error_number, msg, destination.name)
    raise OSError(error_number, os.strerror(error_number), destination.name)


def _call_atomic_rename_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> int:
    """Call the platform no-replace primitive for two sibling names.

    Returns:
        The native zero-or-error return code.

    Raises:
        OSError: If the host lacks the required primitive.
    """
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform == "darwin":
        rename = getattr(library, "renameatx_np", None)
        flag = 0x00000004  # RENAME_EXCL from <stdio.h> on Darwin.
        primitive = "renameatx_np"
    elif sys.platform.startswith("linux"):
        rename = getattr(library, "renameat2", None)
        flag = 1  # RENAME_NOREPLACE from <linux/fs.h>.
        primitive = "renameat2"
    else:
        msg = "WP22H requires an atomic no-replace directory rename primitive."
        raise OSError(errno.ENOTSUP, msg)
    if rename is None:
        msg = f"WP22H requires the host {primitive} no-replace primitive."
        raise OSError(errno.ENOTSUP, msg)
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    return cast(
        "int",
        rename(
            parent_descriptor,
            source,
            parent_descriptor,
            destination,
            flag,
        ),
    )


def _complete_staging_and_publish(
    staging: Path,
    root: Path,
    manifest: CeremonyStageManifest,
    members: Sequence[CeremonyBundleMember],
) -> CeremonyBundleIndex:
    """Close and atomically publish one staged immutable bundle.

    Returns:
        The terminal index published with the bundle.

    Raises:
        FileExistsError: If a destination appears before the no-replace rename.
    """
    index = _write_staged_bundle(staging, manifest, members)
    reopen_ceremony_bundle(
        staging,
        expected_index_checksum=index.content_checksum,
        expected_stage_manifest_checksum=manifest.content_checksum,
    )
    if root.exists() or root.is_symlink():
        msg = "Immutable ceremony bundle appeared before publication."
        raise FileExistsError(msg)
    _atomic_rename_directory_no_replace(staging, root)
    _fsync_directory(root.parent)
    return index


def _write_staged_bundle(
    staging: Path,
    manifest: CeremonyStageManifest,
    members: Sequence[CeremonyBundleMember],
) -> CeremonyBundleIndex:
    """Write one complete off-tree bundle with its terminal index last.

    Returns:
        The terminal index written after all other durable members.
    """
    by_path = {member.relative_path: member for member in members}
    for receipt in manifest.members:
        member = by_path[receipt.relative_path]
        destination = _member_path(staging, receipt.relative_path)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_exclusive_file(destination, member.payload)
    manifest_payload = f"{manifest.to_json()}\n".encode()
    manifest_receipt = CeremonyMemberReceipt(
        relative_path=CEREMONY_STAGE_MANIFEST_NAME,
        role="ceremony-stage-manifest",
        byte_count=len(manifest_payload),
        file_checksum=_sha256_bytes(manifest_payload),
    )
    _write_exclusive_file(staging / CEREMONY_STAGE_MANIFEST_NAME, manifest_payload)
    for directory in sorted(
        (path for path in staging.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(staging)
    index = CeremonyBundleIndex(
        ceremony_id=manifest.ceremony_id,
        stage_id=manifest.stage_id,
        stage_ordinal=manifest.stage_ordinal,
        predecessor_stage_manifest_checksum=manifest.predecessor_stage_manifest_checksum,
        stage_manifest_receipt=manifest_receipt,
        stage_manifest_content_checksum=manifest.content_checksum,
        bundle_inventory_checksum=_inventory_checksum((*manifest.members, manifest_receipt)),
    )
    _write_exclusive_file(staging / CEREMONY_BUNDLE_INDEX_NAME, f"{index.to_json()}\n".encode())
    _fsync_directory(staging)
    return index


def _validate_requested_publication(
    manifest: CeremonyStageManifest,
    members: Sequence[CeremonyBundleMember],
) -> tuple[CeremonyBundleMember, ...]:
    """Require supplied payloads to reproduce the exact stage receipts.

    Returns:
        Members in canonical relative-path order.

    Raises:
        TypeError: If any supplied member has the wrong type.
        ValueError: If the payload receipts differ from the manifest.
    """
    normalized = tuple(members)
    if any(not isinstance(member, CeremonyBundleMember) for member in normalized):
        msg = "Every supplied ceremony member must be a CeremonyBundleMember."
        raise TypeError(msg)
    ordered = tuple(sorted(normalized, key=lambda member: member.relative_path))
    receipts = tuple(member.receipt for member in ordered)
    if receipts != manifest.members:
        msg = "Supplied ceremony member bytes differ from the exact stage manifest."
        raise ValueError(msg)
    return ordered


def _reopen_idempotent_publication(
    root: Path,
    manifest: CeremonyStageManifest,
    members: Sequence[CeremonyBundleMember],
) -> ReopenedCeremonyBundle:
    """Return an existing exact same-byte publication or reject overwrite.

    Returns:
        The fully authenticated existing bundle.

    Raises:
        ValueError: If existing custody is incomplete or byte-different.
    """
    try:
        reopened = reopen_ceremony_bundle(
            root,
            expected_stage_manifest_checksum=manifest.content_checksum,
        )
    except (OSError, TypeError, ValueError) as error:
        msg = "Existing ceremony bundle is incomplete or different and cannot be overwritten."
        raise ValueError(msg) from error
    by_path = {member.relative_path: member.payload for member in members}
    if reopened.manifest != manifest or any(
        _verify_receipted_file(root, receipt) != by_path[receipt.relative_path] for receipt in manifest.members
    ):
        msg = "Existing ceremony bundle bytes differ and cannot be overwritten."
        raise ValueError(msg)
    return reopened


def publish_ceremony_bundle(
    bundle_directory: Path,
    manifest: CeremonyStageManifest,
    members: Sequence[CeremonyBundleMember],
) -> ReopenedCeremonyBundle:
    """Atomically publish one immutable ceremony stage bundle.

    Publication is idempotent only when an existing bundle reopens to the exact
    same stage manifest and member bytes.  New bytes are written to a sibling
    staging directory, with the bundle index written last, before one atomic
    rename makes the complete directory visible.

    Returns:
        The fully reopened terminal bundle.

    Raises:
        TypeError: If the manifest, directory, or member inputs are invalid.
    """
    if not isinstance(manifest, CeremonyStageManifest):
        msg = "manifest must be a CeremonyStageManifest."
        raise TypeError(msg)
    root = _canonical_bundle_directory(bundle_directory)
    ordered = _validate_requested_publication(manifest, members)
    lock_path = root.parent / f".{root.name}.wp22h-ceremony.lock"
    with FileLock(str(lock_path)):
        if root.exists() or root.is_symlink():
            return _reopen_idempotent_publication(root, manifest, ordered)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.wp22h-stage-", dir=root.parent))
        try:
            index = _complete_staging_and_publish(staging, root, manifest, ordered)
        except FileExistsError:
            return _reopen_idempotent_publication(root, manifest, ordered)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return reopen_ceremony_bundle(
            root,
            expected_index_checksum=index.content_checksum,
            expected_stage_manifest_checksum=manifest.content_checksum,
        )


__all__ = [
    "CEREMONY_BUNDLE_INDEX_NAME",
    "CEREMONY_BUNDLE_INDEX_SCHEMA_VERSION",
    "CEREMONY_MEMBER_RECEIPT_SCHEMA_VERSION",
    "CEREMONY_STAGE_MANIFEST_NAME",
    "CEREMONY_STAGE_MANIFEST_SCHEMA_VERSION",
    "CeremonyBundleIndex",
    "CeremonyBundleMember",
    "CeremonyMemberReceipt",
    "CeremonyStageManifest",
    "ReopenedCeremonyBundle",
    "build_ceremony_stage_manifest",
    "publish_ceremony_bundle",
    "read_ceremony_bundle_member",
    "reopen_ceremony_bundle",
    "validate_ceremony_stage_transition",
]
