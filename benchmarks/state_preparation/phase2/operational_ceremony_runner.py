# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Path-oriented execution of the non-numerical WP22H artifact ceremony.

The four commands in this module do not execute training.  They capture or
reopen the exact production inputs, call the repository-owned pilot and screen
closure seams, and publish one immutable predecessor-linked bundle per
artifact stage.  Confirmatory inputs are deliberately limited to the public
population configuration and checksum/count commitment.
"""

from __future__ import annotations

import argparse
import itertools
import os
import stat
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from filelock import FileLock

from .binding_catalog import RepositoryBindingCatalog
from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .ceremony_store import (
    CeremonyBundleMember,
    CeremonyStageManifest,
    ReopenedCeremonyBundle,
    build_ceremony_stage_manifest,
    publish_ceremony_bundle,
    read_ceremony_bundle_member,
    reopen_ceremony_bundle,
    validate_ceremony_stage_transition,
)
from .confirmatory_study import PriorTargetExposureInventory
from .execution_bindings import TrainingExecutionProfile
from .execution_context import ExternalEntropyKeyring, TrainingExecutionContext
from .execution_registry import (
    build_paper_pilot_execution_registry,
    build_paper_screen_execution_registry,
    derive_screening_optimization_seeds,
    derive_screening_seed_root,
)
from .legacy import LegacyEvidenceAudit, load_legacy_evidence_audit, verify_legacy_evidence_sources
from .operational_ceremony import (
    ProductionConfirmationReadiness,
    ProductionPilotClosure,
    ProductionScreenClosure,
    build_ceremony_resumability_fingerprint,
    build_ceremony_training_context,
    close_production_pilot,
    close_production_screen,
    finalize_confirmation_readiness,
    verify_confirmation_readiness,
)
from .pilot import PilotNuisanceSummary
from .protocol import (
    AnalysisSourceManifest,
    FinalConfigurationExecutionManifest,
    FinalConfirmationSeal,
    InitialPreregistration,
    PromotionDecision,
    SampleSizeDesign,
    ScreeningEvidence,
    ScreeningManifest,
    load_initial_preregistration,
)
from .resumability import ResumabilityFingerprint
from .screening import (
    PilotNormalizedComputeCalibration,
    ProductionResourceCalibration,
    build_screening_manifest,
)
from .source_lock import (
    WP22_GOVERNED_ANALYSIS_ENTRY_POINT,
    WP22_GOVERNED_PREREGISTRATION_PATH,
    ExecutionSourceManifest,
    build_analysis_source_manifest,
    capture_governed_execution_source_manifest,
    verify_analysis_source_bridge,
    verify_governed_execution_source_manifest,
)
from .targets import TargetPopulationCommitment, TargetPopulationConfig, TargetPopulationManifest
from .training_orchestration import TrainingRunPlan
from .validation import require_checksum

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

WP22H_OPERATIONAL_PATHS_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22h_operational_paths.v1"
WP22H_STAGE_RUN_RECEIPT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22h_stage_run_receipt.v1"
WP22H_PAPER_CONFIRM_HANDOFF_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22h_paper_confirm_handoff.v1"
WP22H_CEREMONY_ID = "wp22h-paper-confirmation"

_LEGACY_AUDIT_REPOSITORY_PATH = "benchmarks/state_preparation/phase2/data/legacy_evidence_audit_v1.json"
_PREPARE_DIRECTORY = "00-prepare-pilot"
_PILOT_DIRECTORY = "01-close-pilot-prepare-screen"
_SCREEN_DIRECTORY = "02-close-screen-seal"
_READY_DIRECTORY = "03-verify-ready"


class WP22HCeremonyStage(str, Enum):
    """The four non-numerical artifact stages exposed by WP22H."""

    PREPARE_PILOT = "prepare-pilot"
    CLOSE_PILOT_PREPARE_SCREEN = "close-pilot-prepare-screen"
    CLOSE_SCREEN_SEAL = "close-screen-seal"
    VERIFY_READY = "verify-ready"


_STAGE_DIRECTORY = {
    WP22HCeremonyStage.PREPARE_PILOT: _PREPARE_DIRECTORY,
    WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN: _PILOT_DIRECTORY,
    WP22HCeremonyStage.CLOSE_SCREEN_SEAL: _SCREEN_DIRECTORY,
    WP22HCeremonyStage.VERIFY_READY: _READY_DIRECTORY,
}


@dataclass(frozen=True, slots=True)
class PreparePilotOptions:
    """Path-only inputs for capturing and publishing the frozen pilot."""

    repository_root: Path
    ceremony_root: Path
    pilot_primary_target_config_path: Path
    pilot_primary_target_manifest_path: Path
    pilot_secondary_target_config_path: Path
    pilot_secondary_target_manifest_path: Path
    pilot_primary_entropy_path: Path = field(repr=False)
    pilot_secondary_entropy_path: Path = field(repr=False)

    def __post_init__(self) -> None:
        """Require exact ``Path`` values without touching any input."""
        _require_path_fields(self)


@dataclass(frozen=True, slots=True)
class ClosePilotPrepareScreenOptions:
    """Operational paths and retained custody for closing the pilot."""

    repository_root: Path
    ceremony_root: Path
    expected_predecessor_index_checksum: str
    pilot_output_root: Path
    pilot_primary_entropy_path: Path = field(repr=False)
    pilot_secondary_entropy_path: Path = field(repr=False)
    screening_target_config_path: Path
    screening_target_manifest_path: Path
    screening_entropy_path: Path = field(repr=False)

    def __post_init__(self) -> None:
        """Validate path types and the retained predecessor assertion."""
        _require_path_fields(self)
        object.__setattr__(
            self,
            "expected_predecessor_index_checksum",
            require_checksum(self.expected_predecessor_index_checksum, "expected_predecessor_index_checksum"),
        )


@dataclass(frozen=True, slots=True)
class CloseScreenSealOptions:
    """Operational paths and public confirmatory inputs for final sealing."""

    repository_root: Path
    ceremony_root: Path
    expected_predecessor_index_checksum: str
    screen_output_root: Path
    pilot_primary_entropy_path: Path = field(repr=False)
    pilot_secondary_entropy_path: Path = field(repr=False)
    screening_entropy_path: Path = field(repr=False)
    confirmatory_target_config_path: Path
    confirmatory_target_commitment_path: Path

    def __post_init__(self) -> None:
        """Validate path types and the retained predecessor assertion."""
        _require_path_fields(self)
        object.__setattr__(
            self,
            "expected_predecessor_index_checksum",
            require_checksum(self.expected_predecessor_index_checksum, "expected_predecessor_index_checksum"),
        )


@dataclass(frozen=True, slots=True)
class VerifyReadyOptions:
    """Paths and retained custody required to reproduce readiness."""

    repository_root: Path
    ceremony_root: Path
    expected_predecessor_index_checksum: str
    pilot_primary_entropy_path: Path = field(repr=False)
    pilot_secondary_entropy_path: Path = field(repr=False)
    screening_entropy_path: Path = field(repr=False)

    def __post_init__(self) -> None:
        """Validate path types and the retained predecessor assertion."""
        _require_path_fields(self)
        object.__setattr__(
            self,
            "expected_predecessor_index_checksum",
            require_checksum(self.expected_predecessor_index_checksum, "expected_predecessor_index_checksum"),
        )


CeremonyOptions = PreparePilotOptions | ClosePilotPrepareScreenOptions | CloseScreenSealOptions | VerifyReadyOptions


def _require_path_fields(options: CeremonyOptions) -> None:
    """Reject non-``Path`` option values before filesystem access.

    Raises:
        TypeError: If any path option is not an exact ``Path`` value.
    """
    for option_field in fields(options):
        name = option_field.name
        if name.endswith(("_path", "_root")):
            value = getattr(options, name)
            if not isinstance(value, Path):
                msg = f"{name} must be a pathlib.Path."
                raise TypeError(msg)


_OPERATIONAL_PATH_KEYS = frozenset({
    "schema_version",
    "repository_root",
    "ceremony_root",
    "pilot_output_root",
    "screen_output_root",
    "content_checksum",
})


@dataclass(frozen=True, slots=True)
class WP22HOperationalPaths:
    """Immutable absolute operational roots bound into each ceremony stage."""

    repository_root: Path
    ceremony_root: Path
    pilot_output_root: Path | None = None
    screen_output_root: Path | None = None
    schema_version: str = field(default=WP22H_OPERATIONAL_PATHS_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require lexical absolute canonical paths and ordered availability.

        Raises:
            TypeError: If a ledger path has the wrong type.
            ValueError: If a path is noncanonical or stage order is invalid.
        """
        for name in ("repository_root", "ceremony_root", "pilot_output_root", "screen_output_root"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, Path):
                msg = f"{name} must be a pathlib.Path or None."
                raise TypeError(msg)
            if not value.is_absolute() or value != Path(os.path.normpath(value)):
                msg = f"{name} must be an absolute canonical path."
                raise ValueError(msg)
        if self.screen_output_root is not None and self.pilot_output_root is None:
            msg = "screen_output_root requires the predecessor pilot output root."
            raise ValueError(msg)
        if (
            self.pilot_output_root is not None
            and self.screen_output_root is not None
            and (
                self.pilot_output_root == self.screen_output_root
                or self.pilot_output_root in self.screen_output_root.parents
                or self.screen_output_root in self.pilot_output_root.parents
            )
        ):
            msg = "Pilot and screening output roots must be pairwise disjoint."
            raise ValueError(msg)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered path field."""
        return {
            "schema_version": self.schema_version,
            "repository_root": str(self.repository_root),
            "ceremony_root": str(self.ceremony_root),
            "pilot_output_root": None if self.pilot_output_root is None else str(self.pilot_output_root),
            "screen_output_root": None if self.screen_output_root is None else str(self.screen_output_root),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact operational path ledger."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed JSON-native ledger."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> WP22HOperationalPaths:
        """Decode and verify one operational path ledger.

        Returns:
            The strict immutable path ledger.

        Raises:
            TypeError: If a serialized path has the wrong type.
            ValueError: If schema, path, or checksum validation fails.
        """
        mapping = verify_sealed_mapping(
            load_canonical_json_object(payload),
            expected_keys=_OPERATIONAL_PATH_KEYS,
            name="WP22H operational paths",
        )
        if mapping["schema_version"] != WP22H_OPERATIONAL_PATHS_SCHEMA_VERSION:
            msg = "WP22H operational paths use an unsupported schema version."
            raise ValueError(msg)

        def optional_path(value: object, name: str) -> Path | None:
            if value is None:
                return None
            if not isinstance(value, str):
                msg = f"{name} must be a path string or null."
                raise TypeError(msg)
            return Path(value)

        repository = mapping["repository_root"]
        ceremony = mapping["ceremony_root"]
        if not isinstance(repository, str) or not isinstance(ceremony, str):
            msg = "repository_root and ceremony_root must be path strings."
            raise TypeError(msg)
        ledger = cls(
            repository_root=Path(repository),
            ceremony_root=Path(ceremony),
            pilot_output_root=optional_path(mapping["pilot_output_root"], "pilot_output_root"),
            screen_output_root=optional_path(mapping["screen_output_root"], "screen_output_root"),
        )
        if mapping["content_checksum"] != ledger.content_checksum:
            msg = "WP22H operational path checksum changed during normalization."
            raise ValueError(msg)
        return ledger


_RUN_RECEIPT_KEYS = frozenset({
    "schema_version",
    "ceremony_id",
    "stage",
    "bundle_directory",
    "stage_manifest_checksum",
    "bundle_index_checksum",
    "predecessor_stage_manifest_checksum",
    "content_checksum",
})

_HANDOFF_ARTIFACT_PATHS = {
    "preregistration": "handoff/preregistration.json",
    "execution_source_manifest": "handoff/execution_source_manifest.json",
    "analysis_source_manifest": "handoff/analysis_source_manifest.json",
    "paper_screen_binding_catalog": "handoff/screen_execution_catalog.json",
    "sample_size_design": "handoff/sample_size_design.json",
    "pilot_compute_calibration": "handoff/pilot_calibration.json",
    "screening_manifest": "handoff/screening_manifest.json",
    "screening_evidence": "handoff/screening_evidence.json",
    "promotion_decision": "handoff/promotion_decision.json",
    "resource_calibration": "handoff/resource_calibration.json",
    "configuration_execution_manifest": "handoff/configuration_execution_manifest.json",
    "final_confirmation_seal": "handoff/final_confirmation_seal.json",
    "prior_target_exposure_inventory": "handoff/prior_target_exposure_inventory.json",
    "confirmatory_target_configuration": "handoff/confirmatory_target_config.json",
    "confirmatory_target_commitment": "handoff/confirmatory_target_commitment.json",
    "wp22h_readiness_receipt": "readiness/receipt.json",
}
_HANDOFF_KEYS = frozenset({"schema_version", "artifacts", "content_checksum"})
_HANDOFF_ENTRY_KEYS = frozenset({"relative_path", "content_checksum"})


@dataclass(frozen=True, slots=True)
class WP22HPaperConfirmHandoff:
    """Path-and-checksum map for the public dormant WP23 handoff."""

    artifact_checksums: Mapping[str, str]
    schema_version: str = field(default=WP22H_PAPER_CONFIRM_HANDOFF_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require every fixed public handoff role exactly once.

        Raises:
            TypeError: If the artifact map is not a mapping.
            ValueError: If roles or checksums differ from the fixed handoff.
        """
        if not isinstance(self.artifact_checksums, Mapping):
            msg = "artifact_checksums must be a mapping."
            raise TypeError(msg)
        if set(self.artifact_checksums) != set(_HANDOFF_ARTIFACT_PATHS):
            msg = "Paper-confirm handoff must contain every fixed public artifact exactly once."
            raise ValueError(msg)
        normalized = {
            role: require_checksum(self.artifact_checksums[role], f"artifact_checksums.{role}")
            for role in sorted(_HANDOFF_ARTIFACT_PATHS)
        }
        object.__setattr__(self, "artifact_checksums", normalized)

    def _content_dict(self) -> dict[str, object]:
        """Return the fixed path-and-checksum handoff registry."""
        return {
            "schema_version": self.schema_version,
            "artifacts": {
                role: {
                    "relative_path": _HANDOFF_ARTIFACT_PATHS[role],
                    "content_checksum": self.artifact_checksums[role],
                }
                for role in sorted(_HANDOFF_ARTIFACT_PATHS)
            },
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete acyclic handoff map."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native handoff data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> WP22HPaperConfirmHandoff:
        """Decode and verify one strict paper-confirm handoff map.

        Returns:
            The checksum-verified public handoff.

        Raises:
            TypeError: If the registry is not a mapping of exact entries.
            ValueError: If a role, path, checksum, schema, or seal differs.
        """
        mapping = verify_sealed_mapping(
            load_canonical_json_object(payload),
            expected_keys=_HANDOFF_KEYS,
            name="WP22H paper-confirm handoff",
        )
        if mapping["schema_version"] != WP22H_PAPER_CONFIRM_HANDOFF_SCHEMA_VERSION:
            msg = "WP22H paper-confirm handoff uses an unsupported schema version."
            raise ValueError(msg)
        artifacts = mapping["artifacts"]
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(_HANDOFF_ARTIFACT_PATHS):
            msg = "WP22H paper-confirm handoff has an invalid artifact registry."
            raise TypeError(msg)
        artifact_mapping = cast("Mapping[str, object]", artifacts)
        checksums: dict[str, str] = {}
        for role, expected_path in _HANDOFF_ARTIFACT_PATHS.items():
            raw_entry = artifact_mapping[role]
            if not isinstance(raw_entry, Mapping) or set(raw_entry) != _HANDOFF_ENTRY_KEYS:
                msg = f"WP22H handoff entry {role!r} has an invalid schema."
                raise TypeError(msg)
            entry = cast("Mapping[str, object]", raw_entry)
            if entry["relative_path"] != expected_path:
                msg = f"WP22H handoff entry {role!r} uses a changed path."
                raise ValueError(msg)
            checksum = entry["content_checksum"]
            if not isinstance(checksum, str):
                msg = f"WP22H handoff entry {role!r} checksum must be a string."
                raise TypeError(msg)
            checksums[role] = checksum
        handoff = cls(checksums)
        if mapping["content_checksum"] != handoff.content_checksum:
            msg = "WP22H paper-confirm handoff checksum changed during normalization."
            raise ValueError(msg)
        return handoff


@dataclass(frozen=True, slots=True)
class WP22HStageRunReceipt:
    """Operator-retained custody returned after one atomic publication."""

    stage: WP22HCeremonyStage
    bundle_directory: Path
    stage_manifest_checksum: str
    bundle_index_checksum: str
    predecessor_stage_manifest_checksum: str | None
    ceremony_id: str = field(default=WP22H_CEREMONY_ID, init=False)
    schema_version: str = field(default=WP22H_STAGE_RUN_RECEIPT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the stage, canonical path, and exact custody checksums.

        Raises:
            TypeError: If stage or path has the wrong type.
            ValueError: If directory, predecessor, or checksum custody is invalid.
        """
        if not isinstance(self.stage, WP22HCeremonyStage):
            msg = "stage must be a WP22HCeremonyStage."
            raise TypeError(msg)
        if not isinstance(self.bundle_directory, Path):
            msg = "bundle_directory must be a pathlib.Path."
            raise TypeError(msg)
        if (
            not self.bundle_directory.is_absolute()
            or self.bundle_directory != Path(os.path.normpath(self.bundle_directory))
            or self.bundle_directory.name != _STAGE_DIRECTORY[self.stage]
        ):
            msg = "bundle_directory must be the exact canonical directory for its stage."
            raise ValueError(msg)
        for name in ("stage_manifest_checksum", "bundle_index_checksum"):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        predecessor = self.predecessor_stage_manifest_checksum
        if (self.stage is WP22HCeremonyStage.PREPARE_PILOT) != (predecessor is None):
            msg = "Only prepare-pilot may have null predecessor custody."
            raise ValueError(msg)
        if predecessor is not None:
            object.__setattr__(
                self,
                "predecessor_stage_manifest_checksum",
                require_checksum(predecessor, "predecessor_stage_manifest_checksum"),
            )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered run receipt field."""
        return {
            "schema_version": self.schema_version,
            "ceremony_id": self.ceremony_id,
            "stage": self.stage.value,
            "bundle_directory": str(self.bundle_directory),
            "stage_manifest_checksum": self.stage_manifest_checksum,
            "bundle_index_checksum": self.bundle_index_checksum,
            "predecessor_stage_manifest_checksum": self.predecessor_stage_manifest_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of this externally retained custody receipt."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> WP22HStageRunReceipt:
        """Decode one operator-retained stage receipt.

        Returns:
            The checksum-verified receipt.

        Raises:
            TypeError: If a serialized stage or path has the wrong type.
            ValueError: If schema, custody, or checksum verification fails.
        """
        mapping = verify_sealed_mapping(
            load_canonical_json_object(payload),
            expected_keys=_RUN_RECEIPT_KEYS,
            name="WP22H stage run receipt",
        )
        if (
            mapping["schema_version"] != WP22H_STAGE_RUN_RECEIPT_SCHEMA_VERSION
            or mapping["ceremony_id"] != WP22H_CEREMONY_ID
        ):
            msg = "WP22H stage run receipt uses an unsupported schema or ceremony."
            raise ValueError(msg)
        stage = mapping["stage"]
        directory = mapping["bundle_directory"]
        if not isinstance(stage, str) or not isinstance(directory, str):
            msg = "WP22H stage and bundle directory must be strings."
            raise TypeError(msg)
        receipt = cls(
            stage=WP22HCeremonyStage(stage),
            bundle_directory=Path(directory),
            stage_manifest_checksum=cast("str", mapping["stage_manifest_checksum"]),
            bundle_index_checksum=cast("str", mapping["bundle_index_checksum"]),
            predecessor_stage_manifest_checksum=cast("str | None", mapping["predecessor_stage_manifest_checksum"]),
        )
        if mapping["content_checksum"] != receipt.content_checksum:
            msg = "WP22H stage run receipt checksum changed during normalization."
            raise ValueError(msg)
        return receipt


class _CanonicalArtifact(Protocol):
    """Structural type for the ceremony's strict JSON artifacts."""

    def to_json(self) -> str:
        """Return canonical JSON."""


_ArtifactT = TypeVar("_ArtifactT", bound=_CanonicalArtifact)


def _canonical_artifact_bytes(artifact: _CanonicalArtifact) -> bytes:
    """Return the one bundle encoding used for every JSON artifact."""
    return f"{artifact.to_json()}\n".encode()


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return the filesystem identity fields required by pinned reads."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_pinned_descriptor(
    descriptor: int,
    expected_identity: tuple[int, int, int, int, int, int],
    label: str,
) -> tuple[bytes, os.stat_result]:
    """Read one descriptor while retaining its final identity for validation.

    Returns:
        Exact descriptor bytes and its post-read metadata.

    Raises:
        ValueError: If the descriptor does not identify the expected regular file.
    """
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or _file_identity(opened) != expected_identity:
        msg = f"{label} changed while it was opened."
        raise ValueError(msg)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks), os.fstat(descriptor)


def _read_pinned_regular_file(path: Path, label: str) -> bytes:
    """Read a stable, single-link regular file without following its final link.

    Returns:
        Exact file bytes.

    Raises:
        TypeError: If ``path`` is not a ``Path``.
        ValueError: If the source is absent, linked, non-regular, or unstable.
    """
    if not isinstance(path, Path):
        msg = f"{label} path must be a pathlib.Path."
        raise TypeError(msg)
    try:
        before = path.lstat()
    except OSError as error:
        msg = f"{label} is missing or unavailable."
        raise ValueError(msg) from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        msg = f"{label} must be a single-link regular file."
        raise ValueError(msg)
    flags = os.O_RDONLY | cast("int", getattr(os, "O_NOFOLLOW", 0)) | cast("int", getattr(os, "O_NONBLOCK", 0))
    identity = _file_identity(before)
    descriptor = os.open(path, flags)
    try:
        payload, closed = _read_pinned_descriptor(descriptor, identity, label)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if _file_identity(closed) != identity or _file_identity(after) != identity:
        msg = f"{label} changed while it was read."
        raise ValueError(msg)
    return payload


def _decode_canonical_artifact(
    payload: bytes,
    loader: Callable[[str], _ArtifactT],
    label: str,
) -> _ArtifactT:
    """Decode one strict artifact and require its canonical newline encoding.

    Returns:
        The typed checksum-verified artifact.

    Raises:
        ValueError: If parsing or canonical byte validation fails.
    """
    try:
        artifact = loader(payload.decode("utf-8"))
    except (TypeError, UnicodeError, ValueError) as error:
        msg = f"{label} is not a valid canonical artifact."
        raise ValueError(msg) from error
    if payload != _canonical_artifact_bytes(artifact):
        msg = f"{label} bytes are not the canonical newline encoding."
        raise ValueError(msg)
    return artifact


def _load_external_artifact(path: Path, loader: Callable[[str], _ArtifactT], label: str) -> _ArtifactT:
    """Securely load and canonically decode one external public artifact.

    Returns:
        The strict typed artifact.
    """
    return _decode_canonical_artifact(_read_pinned_regular_file(path, label), loader, label)


def _load_bundle_artifact(
    bundle: ReopenedCeremonyBundle,
    relative_path: str,
    loader: Callable[[str], _ArtifactT],
) -> _ArtifactT:
    """Read one manifest-owned member and decode its exact canonical bytes.

    Returns:
        The strict typed artifact.
    """
    return _decode_canonical_artifact(
        read_ceremony_bundle_member(bundle, relative_path),
        loader,
        relative_path,
    )


def _member(relative_path: str, role: str, artifact: _CanonicalArtifact) -> CeremonyBundleMember:
    """Build one canonical immutable ceremony member.

    Returns:
        The path- and role-bound member bytes.
    """
    return CeremonyBundleMember(relative_path, role, _canonical_artifact_bytes(artifact))


def _canonical_existing_directory(path: Path, label: str) -> Path:
    """Return an absolute existing non-symlink directory.

    Returns:
        The canonical absolute directory.

    Raises:
        ValueError: If the path is absent, linked, relative, or noncanonical.
    """
    if not path.is_absolute():
        msg = f"{label} must be an absolute path."
        raise ValueError(msg)
    absolute = path.absolute()
    try:
        if absolute.resolve(strict=True) != absolute:
            msg = f"{label} cannot contain symlink or noncanonical components."
            raise ValueError(msg)
        metadata = absolute.lstat()
    except OSError as error:
        msg = f"{label} must be an existing directory."
        raise ValueError(msg) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        msg = f"{label} must be a non-symlink directory."
        raise ValueError(msg)
    return absolute


def _validated_roots(repository_root: Path, ceremony_root: Path) -> tuple[Path, Path]:
    """Validate the clean-source location and disjoint ceremony custody root.

    Returns:
        Canonical repository and ceremony roots.

    Raises:
        ValueError: If either root is unsafe or they are not disjoint.
    """
    repository = _canonical_existing_directory(repository_root, "repository_root")
    ceremony = _canonical_existing_directory(ceremony_root, "ceremony_root")
    if ceremony == repository or ceremony in repository.parents or repository in ceremony.parents:
        msg = "ceremony_root must be disjoint from and outside repository_root."
        raise ValueError(msg)
    return repository, ceremony


def _validate_ceremony_root_inventory(ceremony_root: Path, stage: WP22HCeremonyStage) -> None:
    """Reject foreign, future, linked, or special ceremony-root state.

    The store's per-bundle lock files and crash-left staging directories are
    operational state, not scientific members.  Only names belonging to the
    fixed prefix through the requested transition are admitted.

    Raises:
        ValueError: If a root entry is foreign, future, linked, or special.
    """
    ordinal = _STAGE_SPEC[stage][1]
    admitted_directories = {
        directory for directory, candidate_ordinal, _ in _STAGE_SPEC.values() if candidate_ordinal <= ordinal
    }
    admitted_locks = {f".{directory}.wp22h-ceremony.lock" for directory in admitted_directories}
    admitted_staging_prefixes = tuple(f".{directory}.wp22h-stage-" for directory in admitted_directories)
    for entry in ceremony_root.iterdir():
        metadata = entry.lstat()
        name = entry.name
        if stat.S_ISLNK(metadata.st_mode):
            msg = f"Ceremony root contains linked entry {name!r}."
            raise ValueError(msg)
        if name in admitted_directories and stat.S_ISDIR(metadata.st_mode):
            continue
        if name in admitted_locks and stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            continue
        if name.startswith(admitted_staging_prefixes) and stat.S_ISDIR(metadata.st_mode):
            continue
        msg = f"Ceremony root contains foreign, future, or special entry {name!r}."
        raise ValueError(msg)


@contextmanager
def _locked_transition(options: CeremonyOptions, stage: WP22HCeremonyStage) -> Iterator[None]:
    """Hold one off-tree whole-ceremony lock around a complete transition.

    Yields:
        Control after the fixed ceremony-root inventory is closed.
    """
    _, ceremony = _validated_roots(options.repository_root, options.ceremony_root)
    lock_path = ceremony.parent / f".{ceremony.name}.wp22h-operational.lock"
    with FileLock(str(lock_path)):
        _validate_ceremony_root_inventory(ceremony, stage)
        yield


def _validate_output_root(path: Path, repository_root: Path, ceremony_root: Path, label: str) -> Path:
    """Return an existing production root disjoint from source and ceremony custody.

    Returns:
        The canonical production output root.

    Raises:
        ValueError: If the root is unsafe or overlaps protected custody.
    """
    root = _canonical_existing_directory(path, label)
    for protected, protected_label in ((repository_root, "repository"), (ceremony_root, "ceremony")):
        if root == protected or root in protected.parents or protected in root.parents:
            msg = f"{label} must be disjoint from the {protected_label} root."
            raise ValueError(msg)
    return root


_PREPARE_SHAPE = frozenset({
    ("operational/paths.json", "operational-paths"),
    ("source/preregistration.json", "preregistration"),
    ("source/execution_source_manifest.json", "execution-source-manifest"),
    ("source/analysis_source_manifest.json", "analysis-source-manifest"),
    ("pilot/execution_catalog.json", "pilot-execution-catalog"),
    ("pilot/execution_profile.json", "pilot-execution-profile"),
    ("pilot/primary_target_config.json", "pilot-primary-target-config"),
    ("pilot/primary_target_manifest.json", "pilot-primary-target-manifest"),
    ("pilot/secondary_target_config.json", "pilot-secondary-target-config"),
    ("pilot/secondary_target_manifest.json", "pilot-secondary-target-manifest"),
    ("pilot/resumability_fingerprint.json", "pilot-resumability-fingerprint"),
    ("pilot/training_plan.json", "pilot-training-plan"),
})

_PILOT_SHAPE = frozenset({
    ("operational/paths.json", "operational-paths"),
    ("pilot/nuisance_summary.json", "pilot-nuisance-summary"),
    ("pilot/sample_size_design.json", "sample-size-design"),
    ("pilot/compute_calibration.json", "pilot-compute-calibration"),
    ("screen/target_config.json", "screen-target-config"),
    ("screen/target_manifest.json", "screen-target-manifest"),
    ("screen/execution_catalog.json", "screen-execution-catalog"),
    ("screen/execution_profile.json", "screen-execution-profile"),
    ("screen/screening_manifest.json", "screening-manifest"),
    ("screen/resumability_fingerprint.json", "screen-resumability-fingerprint"),
    ("screen/training_plan.json", "screen-training-plan"),
})

_SCREEN_SHAPE = frozenset({
    ("operational/paths.json", "operational-paths"),
    ("screen/screening_evidence.json", "screening-evidence"),
    ("screen/promotion_decision.json", "promotion-decision"),
    ("screen/resource_calibration.json", "resource-calibration"),
    ("screen/configuration_execution_manifest.json", "configuration-execution-manifest"),
    ("confirmation/public_target_config.json", "confirmatory-target-config"),
    ("confirmation/public_target_commitment.json", "confirmatory-target-commitment"),
    ("confirmation/final_seal.json", "final-confirmation-seal"),
    ("confirmation/prior_target_exposure_inventory.json", "prior-target-exposure-inventory"),
})

_READY_SHAPE = frozenset({
    ("paper_confirm_handoff.json", "paper-confirm-handoff"),
    ("readiness/receipt.json", "wp22h-readiness-receipt"),
    ("handoff/preregistration.json", "preregistration"),
    ("handoff/execution_source_manifest.json", "execution-source-manifest"),
    ("handoff/analysis_source_manifest.json", "analysis-source-manifest"),
    ("handoff/screen_execution_catalog.json", "screen-execution-catalog"),
    ("handoff/sample_size_design.json", "sample-size-design"),
    ("handoff/pilot_calibration.json", "pilot-compute-calibration"),
    ("handoff/screening_manifest.json", "screening-manifest"),
    ("handoff/screening_evidence.json", "screening-evidence"),
    ("handoff/promotion_decision.json", "promotion-decision"),
    ("handoff/resource_calibration.json", "resource-calibration"),
    ("handoff/configuration_execution_manifest.json", "configuration-execution-manifest"),
    ("handoff/confirmatory_target_config.json", "confirmatory-target-config"),
    ("handoff/confirmatory_target_commitment.json", "confirmatory-target-commitment"),
    ("handoff/final_confirmation_seal.json", "final-confirmation-seal"),
    ("handoff/prior_target_exposure_inventory.json", "prior-target-exposure-inventory"),
})

_STAGE_SPEC = {
    WP22HCeremonyStage.PREPARE_PILOT: (_PREPARE_DIRECTORY, 0, _PREPARE_SHAPE),
    WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN: (_PILOT_DIRECTORY, 1, _PILOT_SHAPE),
    WP22HCeremonyStage.CLOSE_SCREEN_SEAL: (_SCREEN_DIRECTORY, 2, _SCREEN_SHAPE),
    WP22HCeremonyStage.VERIFY_READY: (_READY_DIRECTORY, 3, _READY_SHAPE),
}


def _assert_bundle_identity(bundle: ReopenedCeremonyBundle, stage: WP22HCeremonyStage) -> None:
    """Require one fixed stage identity and exact path/role inventory.

    Raises:
        ValueError: If identity, ordinal, or member inventory differs.
    """
    _, ordinal, shape = _STAGE_SPEC[stage]
    manifest = bundle.manifest
    actual = frozenset((item.relative_path, item.role) for item in manifest.members)
    if (
        manifest.ceremony_id != WP22H_CEREMONY_ID
        or manifest.stage_id != stage.value
        or manifest.stage_ordinal != ordinal
        or actual != shape
    ):
        msg = f"Ceremony bundle is not the exact {stage.value!r} stage inventory."
        raise ValueError(msg)


def _reopen_stage(
    ceremony_root: Path,
    stage: WP22HCeremonyStage,
    *,
    expected_index_checksum: str | None = None,
) -> ReopenedCeremonyBundle:
    """Reopen one fixed stage path with optional external head custody.

    Returns:
        The authenticated fixed stage bundle.
    """
    directory, _, _ = _STAGE_SPEC[stage]
    bundle = reopen_ceremony_bundle(
        ceremony_root / directory,
        expected_index_checksum=expected_index_checksum,
    )
    _assert_bundle_identity(bundle, stage)
    return bundle


def _reopen_chain(
    ceremony_root: Path,
    terminal_stage: WP22HCeremonyStage,
    expected_terminal_index_checksum: str,
) -> tuple[ReopenedCeremonyBundle, ...]:
    """Reopen one contiguous fixed-path chain through the requested terminal.

    Returns:
        Authenticated bundles in increasing ordinal order.
    """
    terminal_ordinal = _STAGE_SPEC[terminal_stage][1]
    stages = tuple(stage for stage in WP22HCeremonyStage if _STAGE_SPEC[stage][1] <= terminal_ordinal)
    bundles = tuple(
        _reopen_stage(
            ceremony_root,
            stage,
            expected_index_checksum=(expected_terminal_index_checksum if stage is terminal_stage else None),
        )
        for stage in stages
    )
    for predecessor, successor in itertools.pairwise(bundles):
        validate_ceremony_stage_transition(predecessor.manifest, successor.manifest)
    return bundles


def _publish_stage(
    ceremony_root: Path,
    stage: WP22HCeremonyStage,
    members: Sequence[CeremonyBundleMember],
    predecessor: CeremonyStageManifest | None,
) -> WP22HStageRunReceipt:
    """Build and atomically publish one fixed immutable stage.

    Returns:
        Externally retainable terminal index custody.
    """
    directory, _, _ = _STAGE_SPEC[stage]
    manifest = build_ceremony_stage_manifest(
        WP22H_CEREMONY_ID,
        stage.value,
        members,
        predecessor=predecessor,
    )
    reopened = publish_ceremony_bundle(ceremony_root / directory, manifest, members)
    _assert_bundle_identity(reopened, stage)
    return WP22HStageRunReceipt(
        stage=stage,
        bundle_directory=reopened.bundle_directory,
        stage_manifest_checksum=reopened.manifest.content_checksum,
        bundle_index_checksum=reopened.index.content_checksum,
        predecessor_stage_manifest_checksum=reopened.manifest.predecessor_stage_manifest_checksum,
    )


def _require_target_pair(
    preregistration: InitialPreregistration,
    config: TargetPopulationConfig,
    manifest: TargetPopulationManifest,
    *,
    data_role: str,
    population_scope: str,
) -> None:
    """Require one exact public config/manifest identity before authorization.

    Raises:
        ValueError: If role, scope, preregistration, or config binding differs.
    """
    if (
        config.preregistration_checksum != preregistration.content_checksum
        or config.data_role != data_role
        or config.population_scope != population_scope
        or manifest.preregistration_checksum != preregistration.content_checksum
        or manifest.population_config_checksum != config.content_checksum
        or manifest.data_role != data_role
        or manifest.population_scope != population_scope
    ):
        msg = f"Target config and manifest differ from the frozen {data_role}/{population_scope} slot."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class _PilotContextState:
    """Rebuilt source-locked pilot state from stage zero."""

    preregistration: InitialPreregistration
    execution_source: ExecutionSourceManifest
    analysis_source: AnalysisSourceManifest
    catalog: RepositoryBindingCatalog
    context: TrainingExecutionContext
    paths: WP22HOperationalPaths


def _load_pilot_context(
    prepare: ReopenedCeremonyBundle,
    repository_root: Path,
    ceremony_root: Path,
    primary_entropy_path: Path,
    secondary_entropy_path: Path,
) -> _PilotContextState:
    """Reopen, rederive, and authorize the exact stage-zero pilot context.

    Returns:
        The fully reconstructed non-serializable pilot state.

    Raises:
        ValueError: If source, bundle, target, registry, or plan validation fails.
    """
    preregistration = _load_bundle_artifact(
        prepare,
        "source/preregistration.json",
        InitialPreregistration.from_json,
    )
    execution_source = _load_bundle_artifact(
        prepare,
        "source/execution_source_manifest.json",
        ExecutionSourceManifest.from_json,
    )
    analysis_source = _load_bundle_artifact(
        prepare,
        "source/analysis_source_manifest.json",
        AnalysisSourceManifest.from_json,
    )
    paths = _load_bundle_artifact(prepare, "operational/paths.json", WP22HOperationalPaths.from_json)
    if paths.repository_root != repository_root or paths.ceremony_root != ceremony_root:
        msg = "The caller roots differ from the immutable ceremony path ledger."
        raise ValueError(msg)
    verify_governed_execution_source_manifest(execution_source, repository_root)
    verify_analysis_source_bridge(execution_source, analysis_source, repository_root)
    _, derived_catalog = build_paper_pilot_execution_registry(preregistration)
    persisted_catalog = _load_bundle_artifact(
        prepare,
        "pilot/execution_catalog.json",
        RepositoryBindingCatalog.from_json,
    )
    persisted_profile = _load_bundle_artifact(
        prepare,
        "pilot/execution_profile.json",
        TrainingExecutionProfile.from_json,
    )
    if persisted_catalog != derived_catalog or persisted_profile != derived_catalog.profile:
        msg = "Persisted pilot execution catalog differs from the frozen repository compiler."
        raise ValueError(msg)
    primary_config = _load_bundle_artifact(
        prepare,
        "pilot/primary_target_config.json",
        TargetPopulationConfig.from_json,
    )
    primary_manifest = _load_bundle_artifact(
        prepare,
        "pilot/primary_target_manifest.json",
        TargetPopulationManifest.from_json,
    )
    secondary_config = _load_bundle_artifact(
        prepare,
        "pilot/secondary_target_config.json",
        TargetPopulationConfig.from_json,
    )
    secondary_manifest = _load_bundle_artifact(
        prepare,
        "pilot/secondary_target_manifest.json",
        TargetPopulationManifest.from_json,
    )
    _require_target_pair(
        preregistration,
        primary_config,
        primary_manifest,
        data_role="development",
        population_scope="primary_q6",
    )
    _require_target_pair(
        preregistration,
        secondary_config,
        secondary_manifest,
        data_role="screening_selection",
        population_scope="secondary_q12",
    )
    fingerprint = build_ceremony_resumability_fingerprint(execution_source, derived_catalog)
    persisted_fingerprint = _load_bundle_artifact(
        prepare,
        "pilot/resumability_fingerprint.json",
        ResumabilityFingerprint.from_json,
    )
    if persisted_fingerprint != fingerprint:
        msg = "Persisted pilot resumability fingerprint differs from the source-locked derivation."
        raise ValueError(msg)
    keyring = ExternalEntropyKeyring.from_files({
        (primary_config.data_role, primary_config.population_scope): primary_entropy_path,
        (secondary_config.data_role, secondary_config.population_scope): secondary_entropy_path,
    })
    context = build_ceremony_training_context(
        preregistration=preregistration,
        catalog=derived_catalog,
        execution_source_manifest=execution_source,
        target_configurations=(primary_config, secondary_config),
        target_manifests=(primary_manifest, secondary_manifest),
        external_entropy_keyring=keyring,
        resumability_fingerprint=fingerprint,
    )
    persisted_plan = _load_bundle_artifact(prepare, "pilot/training_plan.json", TrainingRunPlan.from_json)
    if context.plan != persisted_plan or len(context.plan.jobs) != 1_080:
        msg = "Persisted pilot plan differs from the exact frozen 1,080-job derivation."
        raise ValueError(msg)
    return _PilotContextState(preregistration, execution_source, analysis_source, derived_catalog, context, paths)


@dataclass(frozen=True, slots=True)
class _ScreenContextState:
    """Rebuilt source-locked screen state from stages zero and one."""

    catalog: RepositoryBindingCatalog
    screening_manifest: ScreeningManifest
    context: TrainingExecutionContext
    paths: WP22HOperationalPaths


def _load_screen_context(
    pilot_state: _PilotContextState,
    pilot_closure: ProductionPilotClosure,
    pilot_bundle: ReopenedCeremonyBundle,
    screening_entropy_path: Path,
) -> _ScreenContextState:
    """Reopen and rederive the exact calibrated screen context.

    Returns:
        The fully reconstructed non-serializable screening state.

    Raises:
        ValueError: If pilot closure or any persisted screen derivation differs.
    """
    expected_summary = _load_bundle_artifact(
        pilot_bundle,
        "pilot/nuisance_summary.json",
        PilotNuisanceSummary.from_json,
    )
    expected_design = _load_bundle_artifact(
        pilot_bundle,
        "pilot/sample_size_design.json",
        SampleSizeDesign.from_json,
    )
    expected_calibration = _load_bundle_artifact(
        pilot_bundle,
        "pilot/compute_calibration.json",
        PilotNormalizedComputeCalibration.from_json,
    )
    if (
        pilot_closure.nuisance_summary != expected_summary
        or pilot_closure.sample_size_design != expected_design
        or pilot_closure.pilot_calibration != expected_calibration
    ):
        msg = "Reopened pilot custody does not reproduce the persisted pilot closure."
        raise ValueError(msg)
    config = _load_bundle_artifact(
        pilot_bundle,
        "screen/target_config.json",
        TargetPopulationConfig.from_json,
    )
    target_manifest = _load_bundle_artifact(
        pilot_bundle,
        "screen/target_manifest.json",
        TargetPopulationManifest.from_json,
    )
    _require_target_pair(
        pilot_state.preregistration,
        config,
        target_manifest,
        data_role="screening_selection",
        population_scope="primary_q6",
    )
    candidates, catalog = build_paper_screen_execution_registry(
        pilot_state.preregistration,
        pilot_closure.sample_size_design,
        pilot_closure.pilot_calibration,
    )
    persisted_catalog = _load_bundle_artifact(
        pilot_bundle,
        "screen/execution_catalog.json",
        RepositoryBindingCatalog.from_json,
    )
    persisted_profile = _load_bundle_artifact(
        pilot_bundle,
        "screen/execution_profile.json",
        TrainingExecutionProfile.from_json,
    )
    if catalog != persisted_catalog or persisted_profile != catalog.profile:
        msg = "Persisted screen execution catalog differs from the pilot-calibrated compiler."
        raise ValueError(msg)
    screening_manifest = build_screening_manifest(
        pilot_state.preregistration,
        target_manifest,
        candidates,
        optimization_seeds=derive_screening_optimization_seeds(pilot_state.preregistration),
        screening_seed_root=derive_screening_seed_root(
            pilot_state.preregistration,
            catalog.profile,
            target_manifest,
        ),
    )
    persisted_manifest = _load_bundle_artifact(
        pilot_bundle,
        "screen/screening_manifest.json",
        ScreeningManifest.from_json,
    )
    if screening_manifest != persisted_manifest:
        msg = "Persisted screening manifest differs from the frozen deterministic derivation."
        raise ValueError(msg)
    fingerprint = build_ceremony_resumability_fingerprint(pilot_state.execution_source, catalog)
    persisted_fingerprint = _load_bundle_artifact(
        pilot_bundle,
        "screen/resumability_fingerprint.json",
        ResumabilityFingerprint.from_json,
    )
    if fingerprint != persisted_fingerprint:
        msg = "Persisted screen resumability fingerprint differs from its source lock."
        raise ValueError(msg)
    keyring = ExternalEntropyKeyring.from_files({
        (config.data_role, config.population_scope): screening_entropy_path,
    })
    context = build_ceremony_training_context(
        preregistration=pilot_state.preregistration,
        catalog=catalog,
        execution_source_manifest=pilot_state.execution_source,
        target_configurations=(config,),
        target_manifests=(target_manifest,),
        external_entropy_keyring=keyring,
        resumability_fingerprint=fingerprint,
        screening_manifest=screening_manifest,
        sample_size_design=pilot_closure.sample_size_design,
    )
    persisted_plan = _load_bundle_artifact(pilot_bundle, "screen/training_plan.json", TrainingRunPlan.from_json)
    if context.plan != persisted_plan or len(context.plan.jobs) != 1_296:
        msg = "Persisted screen plan differs from the exact calibrated 1,296-job derivation."
        raise ValueError(msg)
    paths = _load_bundle_artifact(pilot_bundle, "operational/paths.json", WP22HOperationalPaths.from_json)
    if (
        paths.repository_root != pilot_state.paths.repository_root
        or paths.ceremony_root != pilot_state.paths.ceremony_root
        or paths.pilot_output_root is None
        or paths.screen_output_root is not None
    ):
        msg = "Pilot-close operational paths do not extend the exact preparation ledger."
        raise ValueError(msg)
    return _ScreenContextState(catalog, screening_manifest, context, paths)


def _confirmatory_counts(config: TargetPopulationConfig) -> dict[str, int]:
    """Return family totals from one public confirmation configuration."""
    counts: dict[str, int] = {}
    for allocation in config.allocations:
        counts[allocation.family_id] = counts.get(allocation.family_id, 0) + allocation.instance_count
    return counts


def _validate_public_confirmation(
    preregistration: InitialPreregistration,
    design: SampleSizeDesign,
    config: TargetPopulationConfig,
    commitment: TargetPopulationCommitment,
) -> None:
    """Require the sole public confirmatory inputs to match pilot-derived counts.

    Raises:
        ValueError: If role, scope, preregistration, or family counts differ.
    """
    expected_counts = {key: cast("int", value) for key, value in design.target_count_by_family.items()}
    commitment_counts = {key: cast("int", value) for key, value in commitment.target_count_by_family.items()}
    if (
        config.preregistration_checksum != preregistration.content_checksum
        or config.data_role != "confirmatory"
        or config.population_scope != "primary_q6"
        or _confirmatory_counts(config) != expected_counts
        or commitment_counts != expected_counts
    ):
        msg = "Public confirmatory configuration and commitment differ from the pilot-derived design."
        raise ValueError(msg)


def _legacy_audit(repository_root: Path) -> LegacyEvidenceAudit:
    """Load and source-verify the trusted legacy audit from this checkout.

    Returns:
        The trusted and source-verified legacy audit.
    """
    audit = load_legacy_evidence_audit(repository_root / _LEGACY_AUDIT_REPOSITORY_PATH)
    verify_legacy_evidence_sources(audit, repository_root)
    return audit


def _prepare_pilot_locked(options: PreparePilotOptions) -> WP22HStageRunReceipt:
    """Capture source locks and publish the exact 1,080-job pilot bundle.

    Returns:
        The immutable stage-zero custody receipt.

    Raises:
        TypeError: If the option object has the wrong schema.
        ValueError: If any source, input, derivation, or custody check fails.
    """
    if not isinstance(options, PreparePilotOptions):
        msg = "options must be PreparePilotOptions."
        raise TypeError(msg)
    repository, ceremony = _validated_roots(options.repository_root, options.ceremony_root)
    preregistration = load_initial_preregistration(repository / WP22_GOVERNED_PREREGISTRATION_PATH)
    execution_source = capture_governed_execution_source_manifest(
        repository,
        manifest_id="wp22h-governed-execution-source-v1",
    )
    analysis_source = build_analysis_source_manifest(
        execution_source,
        manifest_id="wp22h-primary-analysis-source-v1",
        preregistration_checksum=preregistration.content_checksum,
        analysis_template_checksum=preregistration.analysis_template_checksum,
        analysis_entry_point=WP22_GOVERNED_ANALYSIS_ENTRY_POINT,
    )
    verify_analysis_source_bridge(execution_source, analysis_source, repository)
    primary_config = _load_external_artifact(
        options.pilot_primary_target_config_path,
        TargetPopulationConfig.from_json,
        "pilot primary target config",
    )
    primary_manifest = _load_external_artifact(
        options.pilot_primary_target_manifest_path,
        TargetPopulationManifest.from_json,
        "pilot primary target manifest",
    )
    secondary_config = _load_external_artifact(
        options.pilot_secondary_target_config_path,
        TargetPopulationConfig.from_json,
        "pilot secondary target config",
    )
    secondary_manifest = _load_external_artifact(
        options.pilot_secondary_target_manifest_path,
        TargetPopulationManifest.from_json,
        "pilot secondary target manifest",
    )
    _require_target_pair(
        preregistration,
        primary_config,
        primary_manifest,
        data_role="development",
        population_scope="primary_q6",
    )
    _require_target_pair(
        preregistration,
        secondary_config,
        secondary_manifest,
        data_role="screening_selection",
        population_scope="secondary_q12",
    )
    _, catalog = build_paper_pilot_execution_registry(preregistration)
    fingerprint = build_ceremony_resumability_fingerprint(execution_source, catalog)
    keyring = ExternalEntropyKeyring.from_files({
        (primary_config.data_role, primary_config.population_scope): options.pilot_primary_entropy_path,
        (secondary_config.data_role, secondary_config.population_scope): options.pilot_secondary_entropy_path,
    })
    context = build_ceremony_training_context(
        preregistration=preregistration,
        catalog=catalog,
        execution_source_manifest=execution_source,
        target_configurations=(primary_config, secondary_config),
        target_manifests=(primary_manifest, secondary_manifest),
        external_entropy_keyring=keyring,
        resumability_fingerprint=fingerprint,
    )
    if len(context.plan.jobs) != 1_080:
        msg = "The frozen paper-pilot compiler did not produce exactly 1,080 jobs."
        raise ValueError(msg)
    paths = WP22HOperationalPaths(repository, ceremony)
    members = (
        _member("operational/paths.json", "operational-paths", paths),
        _member("source/preregistration.json", "preregistration", preregistration),
        _member("source/execution_source_manifest.json", "execution-source-manifest", execution_source),
        _member("source/analysis_source_manifest.json", "analysis-source-manifest", analysis_source),
        _member("pilot/execution_catalog.json", "pilot-execution-catalog", catalog),
        _member("pilot/execution_profile.json", "pilot-execution-profile", catalog.profile),
        _member("pilot/primary_target_config.json", "pilot-primary-target-config", primary_config),
        _member("pilot/primary_target_manifest.json", "pilot-primary-target-manifest", primary_manifest),
        _member("pilot/secondary_target_config.json", "pilot-secondary-target-config", secondary_config),
        _member("pilot/secondary_target_manifest.json", "pilot-secondary-target-manifest", secondary_manifest),
        _member("pilot/resumability_fingerprint.json", "pilot-resumability-fingerprint", fingerprint),
        _member("pilot/training_plan.json", "pilot-training-plan", context.plan),
    )
    verify_governed_execution_source_manifest(execution_source, repository)
    verify_analysis_source_bridge(execution_source, analysis_source, repository)
    return _publish_stage(ceremony, WP22HCeremonyStage.PREPARE_PILOT, members, None)


def _close_pilot_prepare_screen_locked(options: ClosePilotPrepareScreenOptions) -> WP22HStageRunReceipt:
    """Reopen production pilot custody and publish the calibrated screen.

    Returns:
        The immutable stage-one custody receipt.

    Raises:
        TypeError: If the option object has the wrong schema.
        ValueError: If predecessor, production custody, or derivation validation fails.
    """
    if not isinstance(options, ClosePilotPrepareScreenOptions):
        msg = "options must be ClosePilotPrepareScreenOptions."
        raise TypeError(msg)
    repository, ceremony = _validated_roots(options.repository_root, options.ceremony_root)
    (prepare,) = _reopen_chain(
        ceremony,
        WP22HCeremonyStage.PREPARE_PILOT,
        options.expected_predecessor_index_checksum,
    )
    pilot_state = _load_pilot_context(
        prepare,
        repository,
        ceremony,
        options.pilot_primary_entropy_path,
        options.pilot_secondary_entropy_path,
    )
    pilot_output = _validate_output_root(options.pilot_output_root, repository, ceremony, "pilot_output_root")
    pilot = close_production_pilot(pilot_state.context, pilot_output)
    screen_config = _load_external_artifact(
        options.screening_target_config_path,
        TargetPopulationConfig.from_json,
        "screening target config",
    )
    screen_targets = _load_external_artifact(
        options.screening_target_manifest_path,
        TargetPopulationManifest.from_json,
        "screening target manifest",
    )
    _require_target_pair(
        pilot_state.preregistration,
        screen_config,
        screen_targets,
        data_role="screening_selection",
        population_scope="primary_q6",
    )
    candidates, screen_catalog = build_paper_screen_execution_registry(
        pilot_state.preregistration,
        pilot.sample_size_design,
        pilot.pilot_calibration,
    )
    screening_manifest = build_screening_manifest(
        pilot_state.preregistration,
        screen_targets,
        candidates,
        optimization_seeds=derive_screening_optimization_seeds(pilot_state.preregistration),
        screening_seed_root=derive_screening_seed_root(
            pilot_state.preregistration,
            screen_catalog.profile,
            screen_targets,
        ),
    )
    fingerprint = build_ceremony_resumability_fingerprint(pilot_state.execution_source, screen_catalog)
    screen_keyring = ExternalEntropyKeyring.from_files({
        (screen_config.data_role, screen_config.population_scope): options.screening_entropy_path,
    })
    screen_context = build_ceremony_training_context(
        preregistration=pilot_state.preregistration,
        catalog=screen_catalog,
        execution_source_manifest=pilot_state.execution_source,
        target_configurations=(screen_config,),
        target_manifests=(screen_targets,),
        external_entropy_keyring=screen_keyring,
        resumability_fingerprint=fingerprint,
        screening_manifest=screening_manifest,
        sample_size_design=pilot.sample_size_design,
    )
    if len(screen_context.plan.jobs) != 1_296:
        msg = "The calibrated paper-screen compiler did not produce exactly 1,296 jobs."
        raise ValueError(msg)
    paths = WP22HOperationalPaths(repository, ceremony, pilot_output)
    members = (
        _member("operational/paths.json", "operational-paths", paths),
        _member("pilot/nuisance_summary.json", "pilot-nuisance-summary", pilot.nuisance_summary),
        _member("pilot/sample_size_design.json", "sample-size-design", pilot.sample_size_design),
        _member("pilot/compute_calibration.json", "pilot-compute-calibration", pilot.pilot_calibration),
        _member("screen/target_config.json", "screen-target-config", screen_config),
        _member("screen/target_manifest.json", "screen-target-manifest", screen_targets),
        _member("screen/execution_catalog.json", "screen-execution-catalog", screen_catalog),
        _member("screen/execution_profile.json", "screen-execution-profile", screen_catalog.profile),
        _member("screen/screening_manifest.json", "screening-manifest", screening_manifest),
        _member("screen/resumability_fingerprint.json", "screen-resumability-fingerprint", fingerprint),
        _member("screen/training_plan.json", "screen-training-plan", screen_context.plan),
    )
    verify_governed_execution_source_manifest(pilot_state.execution_source, repository)
    verify_analysis_source_bridge(pilot_state.execution_source, pilot_state.analysis_source, repository)
    return _publish_stage(
        ceremony,
        WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
        members,
        prepare.manifest,
    )


def _rebuild_through_screen_context(
    repository: Path,
    ceremony: Path,
    prepare: ReopenedCeremonyBundle,
    pilot_bundle: ReopenedCeremonyBundle,
    primary_entropy_path: Path,
    secondary_entropy_path: Path,
    screening_entropy_path: Path,
) -> tuple[_PilotContextState, ProductionPilotClosure, _ScreenContextState]:
    """Rebuild pilot closure and calibrated screen context from bundle members.

    Returns:
        Pilot state, freshly reopened pilot closure, and screen state.

    Raises:
        ValueError: If operational roots or fresh closure projections differ.
    """
    pilot_state = _load_pilot_context(
        prepare,
        repository,
        ceremony,
        primary_entropy_path,
        secondary_entropy_path,
    )
    paths = _load_bundle_artifact(pilot_bundle, "operational/paths.json", WP22HOperationalPaths.from_json)
    if paths.pilot_output_root is None:
        msg = "Pilot-close stage does not identify its production output root."
        raise ValueError(msg)
    pilot_output = _validate_output_root(paths.pilot_output_root, repository, ceremony, "pilot_output_root")
    pilot = close_production_pilot(pilot_state.context, pilot_output)
    screen_state = _load_screen_context(pilot_state, pilot, pilot_bundle, screening_entropy_path)
    return pilot_state, pilot, screen_state


def _close_screen_seal_locked(options: CloseScreenSealOptions) -> WP22HStageRunReceipt:
    """Reopen production screening custody and publish the public final seal.

    Returns:
        The immutable stage-two custody receipt.

    Raises:
        TypeError: If the option object has the wrong schema.
    """
    if not isinstance(options, CloseScreenSealOptions):
        msg = "options must be CloseScreenSealOptions."
        raise TypeError(msg)
    repository, ceremony = _validated_roots(options.repository_root, options.ceremony_root)
    prepare, pilot_bundle = _reopen_chain(
        ceremony,
        WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
        options.expected_predecessor_index_checksum,
    )
    public_config = _load_external_artifact(
        options.confirmatory_target_config_path,
        TargetPopulationConfig.from_json,
        "public confirmatory target config",
    )
    public_commitment = _load_external_artifact(
        options.confirmatory_target_commitment_path,
        TargetPopulationCommitment.from_json,
        "public confirmatory target commitment",
    )
    pilot_state, pilot, screen_state = _rebuild_through_screen_context(
        repository,
        ceremony,
        prepare,
        pilot_bundle,
        options.pilot_primary_entropy_path,
        options.pilot_secondary_entropy_path,
        options.screening_entropy_path,
    )
    _validate_public_confirmation(
        pilot_state.preregistration,
        pilot.sample_size_design,
        public_config,
        public_commitment,
    )
    screen_output = _validate_output_root(options.screen_output_root, repository, ceremony, "screen_output_root")
    paths = WP22HOperationalPaths(
        repository,
        ceremony,
        pilot_output_root=screen_state.paths.pilot_output_root,
        screen_output_root=screen_output,
    )
    screen = close_production_screen(pilot, screen_state.context, screen_output)
    readiness = finalize_confirmation_readiness(
        pilot=pilot,
        screen=screen,
        paper_screen_binding_catalog=screen_state.catalog,
        confirmatory_target_configuration=public_config,
        confirmatory_target_commitment=public_commitment,
        analysis_source_manifest=pilot_state.analysis_source,
        legacy_evidence_audit=_legacy_audit(repository),
        repository_root=repository,
        pre_seal_chain_head_stage_manifest_checksum=pilot_bundle.manifest.content_checksum,
        close_screen_operational_paths_checksum=paths.content_checksum,
    )
    members = (
        _member("operational/paths.json", "operational-paths", paths),
        _member("screen/screening_evidence.json", "screening-evidence", screen.screening_evidence),
        _member("screen/promotion_decision.json", "promotion-decision", screen.promotion_decision),
        _member("screen/resource_calibration.json", "resource-calibration", screen.resource_calibration),
        _member(
            "screen/configuration_execution_manifest.json",
            "configuration-execution-manifest",
            screen.configuration_execution_manifest,
        ),
        _member("confirmation/public_target_config.json", "confirmatory-target-config", public_config),
        _member(
            "confirmation/public_target_commitment.json",
            "confirmatory-target-commitment",
            public_commitment,
        ),
        _member("confirmation/final_seal.json", "final-confirmation-seal", readiness.final_seal),
        _member(
            "confirmation/prior_target_exposure_inventory.json",
            "prior-target-exposure-inventory",
            readiness.prior_target_exposure_inventory,
        ),
    )
    verify_governed_execution_source_manifest(pilot_state.execution_source, repository)
    verify_analysis_source_bridge(pilot_state.execution_source, pilot_state.analysis_source, repository)
    return _publish_stage(
        ceremony,
        WP22HCeremonyStage.CLOSE_SCREEN_SEAL,
        members,
        pilot_bundle.manifest,
    )


def _require_screen_closure_matches(
    screen: ProductionScreenClosure,
    screen_bundle: ReopenedCeremonyBundle,
) -> None:
    """Compare every persisted screen projection to fresh custody derivation.

    Raises:
        ValueError: If any persisted projection differs.
    """
    expected = (
        (
            screen.screening_evidence,
            _load_bundle_artifact(screen_bundle, "screen/screening_evidence.json", ScreeningEvidence.from_json),
        ),
        (
            screen.promotion_decision,
            _load_bundle_artifact(screen_bundle, "screen/promotion_decision.json", PromotionDecision.from_json),
        ),
        (
            screen.resource_calibration,
            _load_bundle_artifact(
                screen_bundle,
                "screen/resource_calibration.json",
                ProductionResourceCalibration.from_json,
            ),
        ),
        (
            screen.configuration_execution_manifest,
            _load_bundle_artifact(
                screen_bundle,
                "screen/configuration_execution_manifest.json",
                FinalConfigurationExecutionManifest.from_json,
            ),
        ),
    )
    if any(actual != persisted for actual, persisted in expected):
        msg = "Reopened screen custody does not reproduce every persisted screen projection."
        raise ValueError(msg)


def _rederive_readiness(
    repository: Path,
    ceremony: Path,
    prepare: ReopenedCeremonyBundle,
    pilot_bundle: ReopenedCeremonyBundle,
    screen_bundle: ReopenedCeremonyBundle,
    primary_entropy_path: Path,
    secondary_entropy_path: Path,
    screening_entropy_path: Path,
) -> tuple[
    _PilotContextState,
    ProductionPilotClosure,
    _ScreenContextState,
    ProductionScreenClosure,
    TargetPopulationConfig,
    TargetPopulationCommitment,
    ProductionConfirmationReadiness,
]:
    """Reopen every stage member and mechanically reproduce final readiness.

    Returns:
        The complete typed, source-verified artifact chain.

    Raises:
        ValueError: If any bundle member or mechanical rederivation differs.
    """
    pilot_state, pilot, screen_state = _rebuild_through_screen_context(
        repository,
        ceremony,
        prepare,
        pilot_bundle,
        primary_entropy_path,
        secondary_entropy_path,
        screening_entropy_path,
    )
    paths = _load_bundle_artifact(screen_bundle, "operational/paths.json", WP22HOperationalPaths.from_json)
    if (
        paths.repository_root != repository
        or paths.ceremony_root != ceremony
        or paths.pilot_output_root != screen_state.paths.pilot_output_root
        or paths.screen_output_root is None
    ):
        msg = "Screen-close operational paths do not extend the exact pilot ledger."
        raise ValueError(msg)
    screen_output = _validate_output_root(paths.screen_output_root, repository, ceremony, "screen_output_root")
    screen = close_production_screen(pilot, screen_state.context, screen_output)
    _require_screen_closure_matches(screen, screen_bundle)
    public_config = _load_bundle_artifact(
        screen_bundle,
        "confirmation/public_target_config.json",
        TargetPopulationConfig.from_json,
    )
    public_commitment = _load_bundle_artifact(
        screen_bundle,
        "confirmation/public_target_commitment.json",
        TargetPopulationCommitment.from_json,
    )
    _validate_public_confirmation(
        pilot_state.preregistration,
        pilot.sample_size_design,
        public_config,
        public_commitment,
    )
    readiness = finalize_confirmation_readiness(
        pilot=pilot,
        screen=screen,
        paper_screen_binding_catalog=screen_state.catalog,
        confirmatory_target_configuration=public_config,
        confirmatory_target_commitment=public_commitment,
        analysis_source_manifest=pilot_state.analysis_source,
        legacy_evidence_audit=_legacy_audit(repository),
        repository_root=repository,
        pre_seal_chain_head_stage_manifest_checksum=pilot_bundle.manifest.content_checksum,
        close_screen_operational_paths_checksum=paths.content_checksum,
    )
    expected_seal = _load_bundle_artifact(
        screen_bundle,
        "confirmation/final_seal.json",
        FinalConfirmationSeal.from_json,
    )
    expected_exposure = _load_bundle_artifact(
        screen_bundle,
        "confirmation/prior_target_exposure_inventory.json",
        PriorTargetExposureInventory.from_json,
    )
    if readiness.final_seal != expected_seal or readiness.prior_target_exposure_inventory != expected_exposure:
        msg = "Fresh final sealing does not reproduce the immutable stage-two artifacts."
        raise ValueError(msg)
    verify_confirmation_readiness(
        readiness,
        execution_source_manifest=pilot_state.execution_source,
        analysis_source_manifest=pilot_state.analysis_source,
        repository_root=repository,
        pre_seal_chain_head_stage_manifest_checksum=pilot_bundle.manifest.content_checksum,
        close_screen_operational_paths_checksum=paths.content_checksum,
    )
    return (
        pilot_state,
        pilot,
        screen_state,
        screen,
        public_config,
        public_commitment,
        readiness,
    )


def _verify_ready_locked(options: VerifyReadyOptions) -> WP22HStageRunReceipt:
    """Reopen and rederive the full chain before publishing readiness.

    Returns:
        The immutable stage-three custody receipt for independent review.

    Raises:
        TypeError: If the option object has the wrong schema.
    """
    if not isinstance(options, VerifyReadyOptions):
        msg = "options must be VerifyReadyOptions."
        raise TypeError(msg)
    repository, ceremony = _validated_roots(options.repository_root, options.ceremony_root)
    prepare, pilot_bundle, screen_bundle = _reopen_chain(
        ceremony,
        WP22HCeremonyStage.CLOSE_SCREEN_SEAL,
        options.expected_predecessor_index_checksum,
    )
    pilot_state, pilot, screen_state, screen, public_config, public_commitment, readiness = _rederive_readiness(
        repository,
        ceremony,
        prepare,
        pilot_bundle,
        screen_bundle,
        options.pilot_primary_entropy_path,
        options.pilot_secondary_entropy_path,
        options.screening_entropy_path,
    )
    handoff = WP22HPaperConfirmHandoff({
        "preregistration": pilot_state.preregistration.content_checksum,
        "execution_source_manifest": pilot_state.execution_source.content_checksum,
        "analysis_source_manifest": pilot_state.analysis_source.content_checksum,
        "paper_screen_binding_catalog": screen_state.catalog.content_checksum,
        "sample_size_design": pilot.sample_size_design.content_checksum,
        "pilot_compute_calibration": pilot.pilot_calibration.content_checksum,
        "screening_manifest": screen_state.screening_manifest.content_checksum,
        "screening_evidence": screen.screening_evidence.content_checksum,
        "promotion_decision": screen.promotion_decision.content_checksum,
        "resource_calibration": screen.resource_calibration.content_checksum,
        "configuration_execution_manifest": screen.configuration_execution_manifest.content_checksum,
        "final_confirmation_seal": readiness.final_seal.content_checksum,
        "prior_target_exposure_inventory": readiness.prior_target_exposure_inventory.content_checksum,
        "confirmatory_target_configuration": public_config.content_checksum,
        "confirmatory_target_commitment": public_commitment.content_checksum,
        "wp22h_readiness_receipt": readiness.receipt.content_checksum,
    })
    members = (
        _member("paper_confirm_handoff.json", "paper-confirm-handoff", handoff),
        _member("readiness/receipt.json", "wp22h-readiness-receipt", readiness.receipt),
        _member("handoff/preregistration.json", "preregistration", pilot_state.preregistration),
        _member(
            "handoff/execution_source_manifest.json",
            "execution-source-manifest",
            pilot_state.execution_source,
        ),
        _member("handoff/analysis_source_manifest.json", "analysis-source-manifest", pilot_state.analysis_source),
        _member("handoff/screen_execution_catalog.json", "screen-execution-catalog", screen_state.catalog),
        _member("handoff/sample_size_design.json", "sample-size-design", pilot.sample_size_design),
        _member("handoff/pilot_calibration.json", "pilot-compute-calibration", pilot.pilot_calibration),
        _member("handoff/screening_manifest.json", "screening-manifest", screen_state.screening_manifest),
        _member("handoff/screening_evidence.json", "screening-evidence", screen.screening_evidence),
        _member("handoff/promotion_decision.json", "promotion-decision", screen.promotion_decision),
        _member("handoff/resource_calibration.json", "resource-calibration", screen.resource_calibration),
        _member(
            "handoff/configuration_execution_manifest.json",
            "configuration-execution-manifest",
            screen.configuration_execution_manifest,
        ),
        _member(
            "handoff/confirmatory_target_config.json",
            "confirmatory-target-config",
            public_config,
        ),
        _member(
            "handoff/confirmatory_target_commitment.json",
            "confirmatory-target-commitment",
            public_commitment,
        ),
        _member("handoff/final_confirmation_seal.json", "final-confirmation-seal", readiness.final_seal),
        _member(
            "handoff/prior_target_exposure_inventory.json",
            "prior-target-exposure-inventory",
            readiness.prior_target_exposure_inventory,
        ),
    )
    return _publish_stage(
        ceremony,
        WP22HCeremonyStage.VERIFY_READY,
        members,
        screen_bundle.manifest,
    )


def prepare_pilot(options: PreparePilotOptions) -> WP22HStageRunReceipt:
    """Publish stage zero under the whole-ceremony custody lock.

    Returns:
        The immutable stage-zero custody receipt.

    Raises:
        TypeError: If the option object has the wrong schema.
    """
    if not isinstance(options, PreparePilotOptions):
        msg = "options must be PreparePilotOptions."
        raise TypeError(msg)
    with _locked_transition(options, WP22HCeremonyStage.PREPARE_PILOT):
        return _prepare_pilot_locked(options)


def close_pilot_prepare_screen(options: ClosePilotPrepareScreenOptions) -> WP22HStageRunReceipt:
    """Publish stage one under the whole-ceremony custody lock.

    Returns:
        The immutable stage-one custody receipt.

    Raises:
        TypeError: If the option object has the wrong schema.
    """
    if not isinstance(options, ClosePilotPrepareScreenOptions):
        msg = "options must be ClosePilotPrepareScreenOptions."
        raise TypeError(msg)
    with _locked_transition(options, WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN):
        return _close_pilot_prepare_screen_locked(options)


def close_screen_seal(options: CloseScreenSealOptions) -> WP22HStageRunReceipt:
    """Publish stage two under the whole-ceremony custody lock.

    Returns:
        The immutable stage-two custody receipt.

    Raises:
        TypeError: If the option object has the wrong schema.
    """
    if not isinstance(options, CloseScreenSealOptions):
        msg = "options must be CloseScreenSealOptions."
        raise TypeError(msg)
    with _locked_transition(options, WP22HCeremonyStage.CLOSE_SCREEN_SEAL):
        return _close_screen_seal_locked(options)


def verify_ready(options: VerifyReadyOptions) -> WP22HStageRunReceipt:
    """Publish stage three under the whole-ceremony custody lock.

    Returns:
        The immutable readiness custody receipt.

    Raises:
        TypeError: If the option object has the wrong schema.
    """
    if not isinstance(options, VerifyReadyOptions):
        msg = "options must be VerifyReadyOptions."
        raise TypeError(msg)
    with _locked_transition(options, WP22HCeremonyStage.VERIFY_READY):
        return _verify_ready_locked(options)


def run_operational_ceremony(options: CeremonyOptions) -> WP22HStageRunReceipt:
    """Dispatch one exact typed ceremony stage without dependency injection.

    Returns:
        The new immutable stage custody receipt.

    Raises:
        TypeError: If ``options`` does not use an exact stage schema.
    """
    if isinstance(options, PreparePilotOptions):
        return prepare_pilot(options)
    if isinstance(options, ClosePilotPrepareScreenOptions):
        return close_pilot_prepare_screen(options)
    if isinstance(options, CloseScreenSealOptions):
        return close_screen_seal(options)
    if isinstance(options, VerifyReadyOptions):
        return verify_ready(options)
    msg = "options must be one exact WP22H ceremony option schema."
    raise TypeError(msg)


def _add_common_paths(parser: argparse.ArgumentParser, *, predecessor: bool) -> None:
    """Add the fixed non-scientific custody options shared by commands."""
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--ceremony-root", type=Path, required=True)
    if predecessor:
        parser.add_argument("--expected-predecessor-index-checksum", required=True)


def _add_pilot_entropy_paths(parser: argparse.ArgumentParser) -> None:
    """Add the two explicitly nonconfirmatory pilot entropy paths."""
    parser.add_argument("--pilot-primary-entropy", type=Path, required=True)
    parser.add_argument("--pilot-secondary-entropy", type=Path, required=True)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the path-oriented four-command WP22H parser.

    Returns:
        An argument parser with no scientific or held-confirmation options.
    """
    parser = argparse.ArgumentParser(prog="python -m benchmarks.state_preparation.phase2.operational_ceremony_runner")
    commands = parser.add_subparsers(dest="stage", required=True)

    prepare = commands.add_parser(WP22HCeremonyStage.PREPARE_PILOT.value)
    _add_common_paths(prepare, predecessor=False)
    prepare.add_argument("--pilot-primary-target-config", type=Path, required=True)
    prepare.add_argument("--pilot-primary-target-manifest", type=Path, required=True)
    prepare.add_argument("--pilot-secondary-target-config", type=Path, required=True)
    prepare.add_argument("--pilot-secondary-target-manifest", type=Path, required=True)
    _add_pilot_entropy_paths(prepare)

    pilot = commands.add_parser(WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN.value)
    _add_common_paths(pilot, predecessor=True)
    pilot.add_argument("--pilot-output-root", type=Path, required=True)
    _add_pilot_entropy_paths(pilot)
    pilot.add_argument("--screening-target-config", type=Path, required=True)
    pilot.add_argument("--screening-target-manifest", type=Path, required=True)
    pilot.add_argument("--screening-entropy", type=Path, required=True)

    screen = commands.add_parser(WP22HCeremonyStage.CLOSE_SCREEN_SEAL.value)
    _add_common_paths(screen, predecessor=True)
    screen.add_argument("--screen-output-root", type=Path, required=True)
    _add_pilot_entropy_paths(screen)
    screen.add_argument("--screening-entropy", type=Path, required=True)
    screen.add_argument("--confirmatory-target-config", type=Path, required=True)
    screen.add_argument("--confirmatory-target-commitment", type=Path, required=True)

    ready = commands.add_parser(WP22HCeremonyStage.VERIFY_READY.value)
    _add_common_paths(ready, predecessor=True)
    _add_pilot_entropy_paths(ready)
    ready.add_argument("--screening-entropy", type=Path, required=True)
    return parser


def _options_from_namespace(namespace: argparse.Namespace) -> CeremonyOptions:
    """Translate one parsed command to its exact typed path schema.

    Returns:
        One immutable stage option object.
    """
    stage = WP22HCeremonyStage(namespace.stage)
    common = {
        "repository_root": namespace.repository_root,
        "ceremony_root": namespace.ceremony_root,
    }
    if stage is WP22HCeremonyStage.PREPARE_PILOT:
        return PreparePilotOptions(
            **common,
            pilot_primary_target_config_path=namespace.pilot_primary_target_config,
            pilot_primary_target_manifest_path=namespace.pilot_primary_target_manifest,
            pilot_secondary_target_config_path=namespace.pilot_secondary_target_config,
            pilot_secondary_target_manifest_path=namespace.pilot_secondary_target_manifest,
            pilot_primary_entropy_path=namespace.pilot_primary_entropy,
            pilot_secondary_entropy_path=namespace.pilot_secondary_entropy,
        )
    predecessor = namespace.expected_predecessor_index_checksum
    if stage is WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN:
        return ClosePilotPrepareScreenOptions(
            **common,
            expected_predecessor_index_checksum=predecessor,
            pilot_output_root=namespace.pilot_output_root,
            pilot_primary_entropy_path=namespace.pilot_primary_entropy,
            pilot_secondary_entropy_path=namespace.pilot_secondary_entropy,
            screening_target_config_path=namespace.screening_target_config,
            screening_target_manifest_path=namespace.screening_target_manifest,
            screening_entropy_path=namespace.screening_entropy,
        )
    if stage is WP22HCeremonyStage.CLOSE_SCREEN_SEAL:
        return CloseScreenSealOptions(
            **common,
            expected_predecessor_index_checksum=predecessor,
            screen_output_root=namespace.screen_output_root,
            pilot_primary_entropy_path=namespace.pilot_primary_entropy,
            pilot_secondary_entropy_path=namespace.pilot_secondary_entropy,
            screening_entropy_path=namespace.screening_entropy,
            confirmatory_target_config_path=namespace.confirmatory_target_config,
            confirmatory_target_commitment_path=namespace.confirmatory_target_commitment,
        )
    return VerifyReadyOptions(
        **common,
        expected_predecessor_index_checksum=predecessor,
        pilot_primary_entropy_path=namespace.pilot_primary_entropy,
        pilot_secondary_entropy_path=namespace.pilot_secondary_entropy,
        screening_entropy_path=namespace.screening_entropy,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one WP22H artifact command and print its retained custody receipt.

    Returns:
        Zero after successful atomic publication.
    """
    options = _options_from_namespace(build_argument_parser().parse_args(argv))
    receipt = run_operational_ceremony(options)
    sys.stdout.write(f"{receipt.to_json()}\n")
    return 0


__all__ = [
    "WP22H_CEREMONY_ID",
    "WP22H_OPERATIONAL_PATHS_SCHEMA_VERSION",
    "WP22H_PAPER_CONFIRM_HANDOFF_SCHEMA_VERSION",
    "WP22H_STAGE_RUN_RECEIPT_SCHEMA_VERSION",
    "CeremonyOptions",
    "ClosePilotPrepareScreenOptions",
    "CloseScreenSealOptions",
    "PreparePilotOptions",
    "VerifyReadyOptions",
    "WP22HCeremonyStage",
    "WP22HOperationalPaths",
    "WP22HPaperConfirmHandoff",
    "WP22HStageRunReceipt",
    "build_argument_parser",
    "close_pilot_prepare_screen",
    "close_screen_seal",
    "main",
    "prepare_pilot",
    "run_operational_ceremony",
    "verify_ready",
]


if __name__ == "__main__":
    raise SystemExit(main())
