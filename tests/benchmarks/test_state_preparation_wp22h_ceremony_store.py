# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Atomic custody tests for the WP22H ceremony bundle store."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from benchmarks.state_preparation.phase2 import ceremony_store
from benchmarks.state_preparation.phase2.ceremony_store import (
    CEREMONY_BUNDLE_INDEX_NAME,
    CEREMONY_STAGE_MANIFEST_NAME,
    CeremonyBundleIndex,
    CeremonyBundleMember,
    CeremonyMemberReceipt,
    CeremonyStageManifest,
    build_ceremony_stage_manifest,
    publish_ceremony_bundle,
    read_ceremony_bundle_member,
    reopen_ceremony_bundle,
    validate_ceremony_stage_transition,
)

if TYPE_CHECKING:
    from pathlib import Path


def _members(*, suffix: bytes = b"") -> tuple[CeremonyBundleMember, ...]:
    """Return a small nested public-artifact bundle."""
    return (
        CeremonyBundleMember("artifacts/analysis.json", "analysis-source", b'{"analysis":1}\n' + suffix),
        CeremonyBundleMember("artifacts/execution.json", "execution-source", b'{"execution":1}\n'),
    )


def _manifest(
    members: tuple[CeremonyBundleMember, ...] | None = None,
    *,
    predecessor: CeremonyStageManifest | None = None,
    stage_id: str = "source-lock",
) -> CeremonyStageManifest:
    """Build one canonical test stage.

    Returns:
        The checksum-linked test stage manifest.
    """
    return build_ceremony_stage_manifest(
        "paper-confirm-ceremony",
        stage_id,
        _members() if members is None else members,
        predecessor=predecessor,
    )


def test_ceremony_schemas_roundtrip_and_reject_checksum_tamper(tmp_path: Path) -> None:
    """Every strict receipt, manifest, and terminal index roundtrips canonically."""
    members = _members()
    manifest = _manifest(members)
    reopened = publish_ceremony_bundle(tmp_path / "bundle", manifest, members)

    receipt = manifest.members[0]
    assert CeremonyMemberReceipt.from_json(receipt.to_json()) == receipt
    assert CeremonyStageManifest.from_json(manifest.to_json()) == manifest
    assert CeremonyBundleIndex.from_json(reopened.index.to_json()) == reopened.index

    changed = manifest.to_dict()
    changed["stage_id"] = "changed-stage"
    with pytest.raises(ValueError, match="content checksum mismatch"):
        CeremonyStageManifest.from_dict(changed)


def test_stage_builder_binds_one_contiguous_predecessor_identity() -> None:
    """Successor stages cannot skip ordinals, ceremonies, or parent checksums."""
    first = _manifest()
    second = _manifest(predecessor=first, stage_id="public-seal")
    validate_ceremony_stage_transition(first, second)
    assert second.stage_ordinal == 1
    assert second.predecessor_stage_manifest_checksum == first.content_checksum

    foreign = CeremonyStageManifest(
        ceremony_id="foreign-ceremony",
        stage_id=second.stage_id,
        stage_ordinal=second.stage_ordinal,
        predecessor_stage_manifest_checksum=second.predecessor_stage_manifest_checksum,
        members=second.members,
        member_inventory_checksum=second.member_inventory_checksum,
    )
    with pytest.raises(ValueError, match="exact contiguous predecessor chain"):
        validate_ceremony_stage_transition(first, foreign)
    with pytest.raises(ValueError, match=r"noninitial.*predecessor"):
        CeremonyStageManifest(
            ceremony_id=first.ceremony_id,
            stage_id="invalid-successor",
            stage_ordinal=1,
            predecessor_stage_manifest_checksum=None,
            members=first.members,
            member_inventory_checksum=first.member_inventory_checksum,
        )


def test_publication_is_same_byte_idempotent_and_never_overwrites(tmp_path: Path) -> None:
    """An existing path authenticates exact bytes or rejects without mutation."""
    root = tmp_path / "bundle"
    members = _members()
    manifest = _manifest(members)
    first = publish_ceremony_bundle(root, manifest, members)
    second = publish_ceremony_bundle(root, manifest, members)
    assert second == first

    original_index = (root / CEREMONY_BUNDLE_INDEX_NAME).read_bytes()
    changed_members = _members(suffix=b"changed")
    changed_manifest = _manifest(changed_members)
    with pytest.raises(ValueError, match="cannot be overwritten"):
        publish_ceremony_bundle(root, changed_manifest, changed_members)
    assert (root / CEREMONY_BUNDLE_INDEX_NAME).read_bytes() == original_index
    assert reopen_ceremony_bundle(root) == first

    reserved = tmp_path / "reserved"
    reserved.mkdir()
    with pytest.raises(ValueError, match="cannot be overwritten"):
        publish_ceremony_bundle(reserved, manifest, members)
    assert not tuple(reserved.iterdir())


@pytest.mark.parametrize("tamper_kind", ["changed", "missing", "extra", "symlink", "hardlink", "fifo"])
def test_reopen_rejects_tamper_and_nonexact_file_types(tmp_path: Path, tamper_kind: str) -> None:
    """Changed inventory, links, and special files all fail closed on reopen."""
    root = tmp_path / "bundle"
    members = _members()
    manifest = _manifest(members)
    publish_ceremony_bundle(root, manifest, members)
    member_path = root / manifest.members[0].relative_path

    if tamper_kind == "changed":
        member_path.write_bytes(b"changed")
    elif tamper_kind == "missing":
        member_path.unlink()
    elif tamper_kind == "extra":
        (root / "foreign.json").write_bytes(b"{}")
    elif tamper_kind == "symlink":
        (root / "foreign-link").symlink_to(member_path)
    elif tamper_kind == "hardlink":
        os.link(member_path, root / "foreign-hardlink")
    else:
        os.mkfifo(root / "foreign-fifo")

    with pytest.raises(ValueError, match=r"Ceremony bundle|missing or unavailable"):
        reopen_ceremony_bundle(root)


def test_manifest_and_index_are_canonical_single_link_members(tmp_path: Path) -> None:
    """The terminal documents are exact, regular, single-link canonical JSON."""
    root = tmp_path / "bundle"
    reopened = publish_ceremony_bundle(root, _manifest(), _members())
    manifest_path = root / CEREMONY_STAGE_MANIFEST_NAME
    index_path = root / CEREMONY_BUNDLE_INDEX_NAME

    assert manifest_path.stat().st_nlink == 1
    assert index_path.stat().st_nlink == 1
    assert manifest_path.read_bytes() == f"{reopened.manifest.to_json()}\n".encode()
    assert index_path.read_bytes() == f"{reopened.index.to_json()}\n".encode()

    index_path.write_bytes(reopened.index.to_json().encode())
    with pytest.raises(ValueError, match="canonical terminal encoding"):
        reopen_ceremony_bundle(root)


def test_pinned_reader_rejects_same_size_in_place_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation timestamps expose an in-place rewrite between lstat and fstat."""
    root = tmp_path / "bundle"
    members = _members()
    reopened = publish_ceremony_bundle(root, _manifest(members), members)
    relative_path = members[0].relative_path
    member_path = root / relative_path
    original_payload = member_path.read_bytes()
    before_rewrite = member_path.stat()
    original_open = ceremony_store.os.open

    def rewrite_after_open(path: Path, flags: int) -> int:
        """Rewrite the canonical inode immediately after its descriptor opens.

        Returns:
            The descriptor that now observes the same-size rewrite.
        """
        descriptor = original_open(path, flags)
        if path == member_path:
            member_path.write_bytes(b"x" * len(original_payload))
            os.utime(
                member_path,
                ns=(before_rewrite.st_atime_ns, before_rewrite.st_mtime_ns + 1_000_000_000),
            )
        return descriptor

    monkeypatch.setattr(ceremony_store.os, "open", rewrite_after_open)
    with pytest.raises(ValueError, match="changed while it was opened"):
        read_ceremony_bundle_member(reopened, relative_path)


def test_public_member_reader_requires_manifest_ownership_and_current_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public member reads authenticate both membership and the current bundle."""
    root = tmp_path / "bundle"
    members = _members()
    reopened = publish_ceremony_bundle(root, _manifest(members), members)
    relative_path = members[0].relative_path
    opened_flags: list[int] = []
    original_open = ceremony_store.os.open

    def recording_open(path: Path, flags: int) -> int:
        """Record descriptor policy before opening the real custody member.

        Returns:
            The real read-only file descriptor.
        """
        opened_flags.append(flags)
        return original_open(path, flags)

    monkeypatch.setattr(ceremony_store.os, "open", recording_open)

    assert read_ceremony_bundle_member(reopened, relative_path) == members[0].payload
    nofollow = getattr(ceremony_store.os, "O_NOFOLLOW", 0)
    nonblocking = getattr(ceremony_store.os, "O_NONBLOCK", 0)
    assert opened_flags
    assert all(not nofollow or flags & nofollow for flags in opened_flags)
    assert all(not nonblocking or flags & nonblocking for flags in opened_flags)
    with pytest.raises(ValueError, match="does not own"):
        read_ceremony_bundle_member(reopened, "foreign.json")

    (root / relative_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match=r"differs|invalid"):
        read_ceremony_bundle_member(reopened, relative_path)


def test_crash_before_atomic_rename_exposes_no_partial_bundle_and_retry_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kill-window failure leaves no scientific root and a clean retry succeeds."""
    root = tmp_path / "bundle"
    members = _members()
    manifest = _manifest(members)
    saw_complete_staging = False

    def crash_before_rename(source: Path, target: Path) -> None:
        """Prove the staged index is terminal, then simulate process failure.

        Raises:
            OSError: Always, to model interruption before atomic publication.
        """
        nonlocal saw_complete_staging
        assert target == root
        staged = reopen_ceremony_bundle(
            source,
            expected_stage_manifest_checksum=manifest.content_checksum,
        )
        assert staged.manifest == manifest
        saw_complete_staging = True
        msg = "simulated rename kill window"
        raise OSError(msg)

    monkeypatch.setattr(ceremony_store, "_atomic_rename_directory_no_replace", crash_before_rename)
    with pytest.raises(OSError, match="kill window"):
        publish_ceremony_bundle(root, manifest, members)
    assert saw_complete_staging
    assert not root.exists()

    monkeypatch.undo()
    assert publish_ceremony_bundle(root, manifest, members).manifest == manifest


def test_atomic_publication_race_never_replaces_a_concurrent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destination created in the final race window remains untouched."""
    root = tmp_path / "bundle"
    members = _members()
    manifest = _manifest(members)

    def publish_concurrent_directory(_source: Path, destination: Path) -> None:
        """Create the competing name and model no-replace EEXIST.

        Raises:
            FileExistsError: Always, after creating the competing directory.
        """
        destination.mkdir()
        msg = "concurrent immutable destination"
        raise FileExistsError(msg)

    monkeypatch.setattr(
        ceremony_store,
        "_atomic_rename_directory_no_replace",
        publish_concurrent_directory,
    )
    with pytest.raises(ValueError, match="cannot be overwritten"):
        publish_ceremony_bundle(root, manifest, members)
    assert root.is_dir()
    assert not tuple(root.iterdir())


def test_member_paths_are_sorted_unique_collision_free_and_reserved() -> None:
    """Caller bytes cannot claim store custody paths or file/directory aliases."""
    with pytest.raises(ValueError, match="reserved"):
        CeremonyBundleMember(CEREMONY_BUNDLE_INDEX_NAME, "artifact", b"{}")
    with pytest.raises(ValueError, match="reserved"):
        CeremonyBundleMember(f"{CEREMONY_STAGE_MANIFEST_NAME}/child", "artifact", b"{}")

    parent = CeremonyBundleMember("artifact", "artifact", b"parent")
    child = CeremonyBundleMember("artifact/child.json", "artifact", b"child")
    with pytest.raises(ValueError, match="another member's directory"):
        build_ceremony_stage_manifest("ceremony", "stage", (parent, child))

    duplicate = CeremonyBundleMember("same.json", "artifact", b"same")
    with pytest.raises(ValueError, match="unique"):
        build_ceremony_stage_manifest("ceremony", "stage", (duplicate, duplicate))
