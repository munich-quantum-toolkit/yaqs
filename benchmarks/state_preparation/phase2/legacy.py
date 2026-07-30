# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Commit-addressed audit records for pre-Phase-I state-preparation evidence."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    read_canonical_json_object,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .validation import (
    require_checksum,
    require_exact_keys,
    require_git_blob,
    require_git_commit,
    require_mapping,
    require_nonempty_text,
    require_relative_path,
    require_slug,
    require_string_sequence,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LEGACY_AUDIT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.legacy_evidence_audit.v1"
LEGACY_NAMESPACE = "legacy"
LEGACY_CLASSIFICATIONS = ("reproduced", "discrepant", "unreproducible")
LEGACY_ARTIFACT_ROLES = ("environment", "implementation", "result", "figure", "manuscript")

DEFAULT_LEGACY_AUDIT_PATH = Path(__file__).with_name("data") / "legacy_evidence_audit_v1.json"
TRUSTED_LEGACY_AUDIT_CHECKSUM = "sha256:a294080bf54a62b2bad0df85faa2f75ade5098b6a9afd84dc81fbb29bafdda1c"

_ARTIFACT_KEYS = frozenset({
    "artifact_id",
    "repo_path",
    "source_commit",
    "git_blob_id",
    "content_checksum",
    "role",
})
_CLAIM_KEYS = frozenset({
    "claim_id",
    "statement",
    "classification",
    "artifact_ids",
    "manuscript_locations",
    "configuration",
    "configuration_checksum",
    "limitations",
})
_AUDIT_KEYS = frozenset({
    "schema_version",
    "audit_id",
    "namespace",
    "source_commit",
    "environment_lock_artifact_id",
    "artifacts",
    "claims",
    "missing_provenance",
    "content_checksum",
})


@dataclass(frozen=True, slots=True)
class LegacyArtifactRef:
    """One immutable reference to evidence at a historical Git commit."""

    artifact_id: str
    repo_path: str
    source_commit: str
    git_blob_id: str
    content_checksum: str
    role: Literal["environment", "implementation", "result", "figure", "manuscript"]

    def __post_init__(self) -> None:
        """Validate identity, path, commit, digest, and evidence role.

        Raises:
            ValueError: If the evidence role is unsupported.
        """
        object.__setattr__(self, "artifact_id", require_slug(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "repo_path", require_relative_path(self.repo_path, "repo_path"))
        object.__setattr__(self, "source_commit", require_git_commit(self.source_commit, "source_commit"))
        object.__setattr__(self, "git_blob_id", require_git_blob(self.git_blob_id))
        object.__setattr__(self, "content_checksum", require_checksum(self.content_checksum, "content_checksum"))
        if self.role not in LEGACY_ARTIFACT_ROLES:
            msg = f"role must be one of {LEGACY_ARTIFACT_ROLES}, got {self.role!r}."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native artifact reference."""
        return {
            "artifact_id": self.artifact_id,
            "repo_path": self.repo_path,
            "source_commit": self.source_commit,
            "git_blob_id": self.git_blob_id,
            "content_checksum": self.content_checksum,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: object) -> LegacyArtifactRef:
        """Construct an artifact reference from an exact JSON object.

        Args:
            data: Serialized artifact reference.

        Returns:
            The validated immutable reference.
        """
        mapping = require_mapping(data, "legacy artifact")
        require_exact_keys(mapping, _ARTIFACT_KEYS, "legacy artifact")
        return cls(
            artifact_id=cast("str", mapping["artifact_id"]),
            repo_path=cast("str", mapping["repo_path"]),
            source_commit=cast("str", mapping["source_commit"]),
            git_blob_id=cast("str", mapping["git_blob_id"]),
            content_checksum=cast("str", mapping["content_checksum"]),
            role=cast("Literal['environment', 'implementation', 'result', 'figure', 'manuscript']", mapping["role"]),
        )


@dataclass(frozen=True, slots=True)
class LegacyClaimAudit:
    """Classification and configuration evidence for one historical claim."""

    claim_id: str
    statement: str
    classification: Literal["reproduced", "discrepant", "unreproducible"]
    artifact_ids: tuple[str, ...]
    manuscript_locations: tuple[str, ...]
    configuration: Mapping[str, object] | None
    configuration_checksum: str | None
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate references, classification, configuration, and limitations.

        Raises:
            ValueError: If the classification or configuration evidence is inconsistent.
        """
        object.__setattr__(self, "claim_id", require_slug(self.claim_id, "claim_id"))
        object.__setattr__(self, "statement", require_nonempty_text(self.statement, "statement"))
        if self.classification not in LEGACY_CLASSIFICATIONS:
            msg = f"classification must be one of {LEGACY_CLASSIFICATIONS}, got {self.classification!r}."
            raise ValueError(msg)
        artifact_ids = require_string_sequence(
            self.artifact_ids,
            "artifact_ids",
            minimum_length=1,
            unique=True,
            slugs=True,
        )
        manuscript_locations = require_string_sequence(
            self.manuscript_locations,
            "manuscript_locations",
            unique=True,
        )
        limitations = require_string_sequence(
            self.limitations,
            "limitations",
            minimum_length=1,
            unique=True,
        )
        object.__setattr__(self, "artifact_ids", artifact_ids)
        object.__setattr__(self, "manuscript_locations", manuscript_locations)
        object.__setattr__(self, "limitations", limitations)

        if self.configuration is None:
            if self.configuration_checksum is not None:
                msg = "configuration_checksum must be null when configuration is null."
                raise ValueError(msg)
            msg = "configuration and configuration_checksum are required for every audited claim."
            raise ValueError(msg)
        frozen_configuration = freeze_json_mapping(self.configuration, "configuration")
        checksum = require_checksum(self.configuration_checksum, "configuration_checksum")
        expected = canonical_checksum(frozen_configuration)
        if checksum != expected:
            msg = f"configuration_checksum mismatch: expected {checksum}, computed {expected}."
            raise ValueError(msg)
        object.__setattr__(self, "configuration", frozen_configuration)
        object.__setattr__(self, "configuration_checksum", checksum)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native claim record."""
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "classification": self.classification,
            "artifact_ids": list(self.artifact_ids),
            "manuscript_locations": list(self.manuscript_locations),
            "configuration": None if self.configuration is None else thaw_json_mapping(self.configuration),
            "configuration_checksum": self.configuration_checksum,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, data: object) -> LegacyClaimAudit:
        """Construct a claim audit from an exact JSON object.

        Args:
            data: Serialized claim audit.

        Returns:
            The validated immutable claim.
        """
        mapping = require_mapping(data, "legacy claim")
        require_exact_keys(mapping, _CLAIM_KEYS, "legacy claim")
        configuration_value = mapping["configuration"]
        configuration = None if configuration_value is None else require_mapping(configuration_value, "configuration")
        return cls(
            claim_id=cast("str", mapping["claim_id"]),
            statement=cast("str", mapping["statement"]),
            classification=cast("Literal['reproduced', 'discrepant', 'unreproducible']", mapping["classification"]),
            artifact_ids=cast("tuple[str, ...]", mapping["artifact_ids"]),
            manuscript_locations=cast("tuple[str, ...]", mapping["manuscript_locations"]),
            configuration=configuration,
            configuration_checksum=cast("str | None", mapping["configuration_checksum"]),
            limitations=cast("tuple[str, ...]", mapping["limitations"]),
        )


@dataclass(frozen=True, slots=True)
class LegacyEvidenceAudit:
    """Sealed, commit-addressed inventory of historical publication evidence."""

    audit_id: str
    source_commit: str
    environment_lock_artifact_id: str
    artifacts: tuple[LegacyArtifactRef, ...]
    claims: tuple[LegacyClaimAudit, ...]
    missing_provenance: tuple[str, ...]
    schema_version: str = field(default=LEGACY_AUDIT_SCHEMA_VERSION, init=False)
    namespace: str = field(default=LEGACY_NAMESPACE, init=False)

    def __post_init__(self) -> None:
        """Validate unique identities and all claim-to-artifact references.

        Raises:
            TypeError: If an artifact or claim has the wrong record type.
            ValueError: If required records are absent, duplicated, or inconsistently referenced.
        """
        object.__setattr__(self, "audit_id", require_slug(self.audit_id, "audit_id"))
        object.__setattr__(self, "source_commit", require_git_commit(self.source_commit, "source_commit"))
        object.__setattr__(
            self,
            "environment_lock_artifact_id",
            require_slug(self.environment_lock_artifact_id, "environment_lock_artifact_id"),
        )
        artifacts = tuple(self.artifacts)
        claims = tuple(self.claims)
        if not artifacts:
            msg = "artifacts must contain at least one reference."
            raise ValueError(msg)
        if not claims:
            msg = "claims must contain at least one audit record."
            raise ValueError(msg)
        if not all(isinstance(artifact, LegacyArtifactRef) for artifact in artifacts):
            msg = "artifacts must contain only LegacyArtifactRef values."
            raise TypeError(msg)
        if not all(isinstance(claim, LegacyClaimAudit) for claim in claims):
            msg = "claims must contain only LegacyClaimAudit values."
            raise TypeError(msg)
        artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
        claim_ids = tuple(claim.claim_id for claim in claims)
        if len(artifact_ids) != len(set(artifact_ids)):
            msg = "artifacts must have unique artifact_id values."
            raise ValueError(msg)
        if len(claim_ids) != len(set(claim_ids)):
            msg = "claims must have unique claim_id values."
            raise ValueError(msg)
        if self.environment_lock_artifact_id not in artifact_ids:
            msg = "environment_lock_artifact_id must reference an artifact in the audit."
            raise ValueError(msg)
        environment_lock = next(
            artifact for artifact in artifacts if artifact.artifact_id == self.environment_lock_artifact_id
        )
        if environment_lock.role != "environment":
            msg = "environment_lock_artifact_id must reference an environment artifact."
            raise ValueError(msg)
        if any(artifact.source_commit != self.source_commit for artifact in artifacts):
            msg = "Every legacy artifact must use the audit source_commit."
            raise ValueError(msg)
        artifact_id_set = frozenset(artifact_ids)
        for claim in claims:
            unknown = sorted(set(claim.artifact_ids) - artifact_id_set)
            if unknown:
                msg = f"Claim {claim.claim_id!r} references unknown artifacts {unknown!r}."
                raise ValueError(msg)
        missing_provenance = require_string_sequence(
            self.missing_provenance,
            "missing_provenance",
            minimum_length=1,
            unique=True,
        )
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "missing_provenance", missing_provenance)

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete audit payload."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return the checksum-covered audit content."""
        return {
            "schema_version": self.schema_version,
            "audit_id": self.audit_id,
            "namespace": self.namespace,
            "source_commit": self.source_commit,
            "environment_lock_artifact_id": self.environment_lock_artifact_id,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "claims": [claim.to_dict() for claim in self.claims],
            "missing_provenance": list(self.missing_provenance),
        }

    def to_dict(self) -> dict[str, object]:
        """Return the sealed JSON-native audit record."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    def artifact(self, artifact_id: str) -> LegacyArtifactRef:
        """Return one artifact by stable identifier.

        Args:
            artifact_id: Artifact identifier to resolve.

        Returns:
            The matching artifact.

        Raises:
            KeyError: If the identifier is absent.
        """
        normalized = require_slug(artifact_id, "artifact_id")
        for artifact in self.artifacts:
            if artifact.artifact_id == normalized:
                return artifact
        raise KeyError(normalized)

    def claim(self, claim_id: str) -> LegacyClaimAudit:
        """Return one claim by stable identifier.

        Args:
            claim_id: Claim identifier to resolve.

        Returns:
            The matching claim.

        Raises:
            KeyError: If the identifier is absent.
        """
        normalized = require_slug(claim_id, "claim_id")
        for claim in self.claims:
            if claim.claim_id == normalized:
                return claim
        raise KeyError(normalized)

    @classmethod
    def from_dict(cls, data: object) -> LegacyEvidenceAudit:
        """Construct and checksum-verify an audit from serialized data.

        Args:
            data: Sealed audit mapping.

        Returns:
            The validated immutable audit.

        Raises:
            ValueError: If the schema, namespace, or normalized checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_AUDIT_KEYS, name="legacy evidence audit")
        if mapping["schema_version"] != LEGACY_AUDIT_SCHEMA_VERSION:
            msg = f"schema_version must be {LEGACY_AUDIT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        if mapping["namespace"] != LEGACY_NAMESPACE:
            msg = f"namespace must be {LEGACY_NAMESPACE!r}."
            raise ValueError(msg)
        artifact_values = cast("Sequence[object]", mapping["artifacts"])
        claim_values = cast("Sequence[object]", mapping["claims"])
        audit = cls(
            audit_id=cast("str", mapping["audit_id"]),
            source_commit=cast("str", mapping["source_commit"]),
            environment_lock_artifact_id=cast("str", mapping["environment_lock_artifact_id"]),
            artifacts=tuple(LegacyArtifactRef.from_dict(value) for value in artifact_values),
            claims=tuple(LegacyClaimAudit.from_dict(value) for value in claim_values),
            missing_provenance=cast("tuple[str, ...]", mapping["missing_provenance"]),
        )
        supplied = cast("str", mapping["content_checksum"])
        if audit.content_checksum != supplied:
            msg = (
                "Legacy audit checksum changed during normalization: "
                f"expected {supplied}, got {audit.content_checksum}."
            )
            raise ValueError(msg)
        return audit

    @classmethod
    def from_json(cls, payload: str) -> LegacyEvidenceAudit:
        """Construct an audit from canonical sealed JSON text.

        Args:
            payload: Canonical JSON document.

        Returns:
            The validated immutable audit.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def load_legacy_evidence_audit(path: Path = DEFAULT_LEGACY_AUDIT_PATH) -> LegacyEvidenceAudit:
    """Load the checked-in canonical legacy-evidence audit.

    Args:
        path: Canonical audit document.

    Returns:
        The validated immutable audit.

    Raises:
        ValueError: If the document differs from the trusted checked-in digest.
    """
    audit = LegacyEvidenceAudit.from_dict(read_canonical_json_object(path))
    if audit.content_checksum != TRUSTED_LEGACY_AUDIT_CHECKSUM:
        msg = (
            "Checked-in legacy-audit digest differs from the trusted runtime constant: "
            f"expected {TRUSTED_LEGACY_AUDIT_CHECKSUM}, got {audit.content_checksum}."
        )
        raise ValueError(msg)
    return audit


def verify_legacy_evidence_sources(
    audit: LegacyEvidenceAudit,
    repository_root: Path,
) -> tuple[str, ...]:
    """Verify every artifact against its commit-addressed Git blob and SHA-256.

    Args:
        audit: Audit whose references are verified.
        repository_root: Git worktree containing the historical commit.

    Returns:
        Verified artifact identifiers in audit order.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If Git is unavailable or an artifact reference differs.
    """
    if not isinstance(audit, LegacyEvidenceAudit):
        msg = f"audit must be a LegacyEvidenceAudit, got {type(audit).__name__}."
        raise TypeError(msg)
    if not isinstance(repository_root, Path):
        msg = f"repository_root must be a pathlib.Path, got {type(repository_root).__name__}."
        raise TypeError(msg)
    git_executable = shutil.which("git")
    if git_executable is None:
        msg = "Git is required to verify commit-addressed legacy evidence."
        raise ValueError(msg)

    verified: list[str] = []
    for artifact in audit.artifacts:
        revision_path = f"{artifact.source_commit}:{artifact.repo_path}"
        blob_result = subprocess.run(  # noqa: S603 -- executable and revision are resolved and strictly validated
            [git_executable, "-C", str(repository_root), "rev-parse", revision_path],
            check=False,
            capture_output=True,
            text=True,
        )
        if blob_result.returncode != 0:
            detail = blob_result.stderr.strip() or blob_result.stdout.strip()
            msg = f"Could not resolve legacy artifact {revision_path!r}: {detail}."
            raise ValueError(msg)
        actual_blob = blob_result.stdout.strip()
        if actual_blob != artifact.git_blob_id:
            msg = (
                f"Legacy artifact {artifact.artifact_id!r} blob mismatch: "
                f"expected {artifact.git_blob_id}, got {actual_blob}."
            )
            raise ValueError(msg)

        content_result = subprocess.run(  # noqa: S603 -- executable and blob identifier are strictly validated
            [git_executable, "-C", str(repository_root), "cat-file", "blob", actual_blob],
            check=False,
            capture_output=True,
        )
        if content_result.returncode != 0:
            detail = content_result.stderr.decode(errors="replace").strip()
            msg = f"Could not read legacy blob {actual_blob}: {detail}."
            raise ValueError(msg)
        actual_checksum = f"sha256:{hashlib.sha256(content_result.stdout).hexdigest()}"
        if actual_checksum != artifact.content_checksum:
            msg = (
                f"Legacy artifact {artifact.artifact_id!r} checksum mismatch: "
                f"expected {artifact.content_checksum}, got {actual_checksum}."
            )
            raise ValueError(msg)
        verified.append(artifact.artifact_id)
    return tuple(verified)


__all__ = [
    "DEFAULT_LEGACY_AUDIT_PATH",
    "LEGACY_ARTIFACT_ROLES",
    "LEGACY_AUDIT_SCHEMA_VERSION",
    "LEGACY_CLASSIFICATIONS",
    "LEGACY_NAMESPACE",
    "TRUSTED_LEGACY_AUDIT_CHECKSUM",
    "LegacyArtifactRef",
    "LegacyClaimAudit",
    "LegacyEvidenceAudit",
    "load_legacy_evidence_audit",
    "verify_legacy_evidence_sources",
]
