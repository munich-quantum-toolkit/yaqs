# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fail-closed execution context and external target-entropy custody for WP22D.

The records in this module deliberately separate public, checksum-bearing
scientific identities from secret role-master entropy.  The keyring and the
complete context cannot be serialized or pickled.  Only non-secret checksums
are copied into a :class:`~.training_orchestration.TrainingRunPlan`.
"""

from __future__ import annotations

import contextlib
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NoReturn, SupportsIndex, cast

from .binding_catalog import ExecutableScopedBinding, RepositoryBindingCatalog
from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .execution_bindings import TrainingExecutionProfile
from .protocol import (
    AnalysisSourceManifest,
    ConfirmationAuthorization,
    FinalConfigurationExecutionManifest,
    FinalConfirmationSeal,
    InitialPreregistration,
    SampleSizeDesign,
    ScreeningCell,
    ScreeningManifest,
    validate_final_configuration_execution_manifest,
)
from .resumability import ResumabilityFingerprint
from .scheduled_execution import ScheduledExecutionProgram, ScheduledJobSeedSet
from .source_lock import ExecutionSourceManifest, verify_execution_source_manifest, verify_final_seal_source_lock
from .targets import (
    MaterializedTarget,
    TargetMaterializationAuthorization,
    TargetPopulationConfig,
    TargetPopulationManifest,
    materialize_target_population,
    role_master_entropy_commitment,
)
from .training_orchestration import (
    ConfirmExecutionRequest,
    TrainingJob,
    TrainingRunPlan,
    build_paper_confirm_plan,
    validate_confirm_execution_request,
)
from .validation import require_checksum, require_relative_path, require_slug

if TYPE_CHECKING:
    from .execution_protocol import FreshEvaluationPolicy
    from .training_schedules import TrainingStrategySchedule

TRAINING_CANDIDATE_REF_SCHEMA_VERSION = "yaqs.state_preparation.phase2.training_candidate_ref.v1"
TRAINING_PREFLIGHT_REPORT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.training_preflight_report.v1"

_CANDIDATE_KEYS = frozenset({
    "schema_version",
    "candidate_schema_version",
    "candidate_checksum",
    "method_id",
    "content_checksum",
})
_PREFLIGHT_KEYS = frozenset({
    "schema_version",
    "plan_checksum",
    "execution_profile_checksum",
    "execution_source_manifest_checksum",
    "source_fingerprint_checksum",
    "job_count",
    "target_population_count",
    "compiled_program_count",
    "content_checksum",
})
_TARGET_SCOPE_BY_QUBIT_COUNT = {6: "primary_q6", 12: "secondary_q12"}
_EVALUATION_PURPOSE_BY_PRESET = {
    "training-smoke": "smoke_evaluation",
    "paper-pilot": "pilot_fresh_evaluation",
    "paper-screen": "screening_outer",
}
_EXACT_JOB_COUNT_BY_PRESET = {
    "training-smoke": 10,
    "historical-layerwise-reproduction": 5,
    "paper-pilot": 1_080,
    "paper-screen": 1_296,
}


def _serialization_forbidden() -> NoReturn:
    """Reject attempts to serialize secret-bearing runtime state.

    Raises:
        TypeError: Always, because secret-bearing runtime state is not serializable.
    """
    msg = "External target entropy and its execution context are intentionally non-serializable."
    raise TypeError(msg)


def _normalize_entropy(value: bytes) -> bytes:
    """Return one exact 256-bit role key without echoing rejected bytes.

    Returns:
        A detached immutable 32-byte key.

    Raises:
        TypeError: If ``value`` is not bytes.
        ValueError: If it is not an exact raw or lowercase-hex 256-bit key.
    """
    if type(value) is not bytes:
        msg = "External entropy must be supplied as opaque bytes."
        raise TypeError(msg)
    if len(value) == 32:
        return bytes(value)
    if len(value) == 64:
        try:
            text = value.decode("ascii")
            decoded = bytes.fromhex(text)
        except (UnicodeDecodeError, ValueError) as error:
            msg = "External entropy file has an invalid opaque key encoding."
            raise ValueError(msg) from error
        if text != text.lower() or decoded.hex() != text or len(decoded) != 32:
            msg = "External entropy file has an invalid opaque key encoding."
            raise ValueError(msg)
        return decoded
    msg = "External entropy file must contain exactly one 256-bit key."
    raise ValueError(msg)


def _entropy_slot(data_role: str, population_scope: str) -> tuple[str, str]:
    """Validate and return one public target-custody slot.

    Returns:
        The normalized ``(data_role, population_scope)`` pair.

    Raises:
        ValueError: If the role/scope pair is not an authorized target slot.
    """
    role = require_slug(data_role, "data_role")
    scope = require_slug(population_scope, "population_scope")
    allowed = {
        ("development", "primary_q6"),
        ("screening_selection", "primary_q6"),
        ("screening_selection", "secondary_q12"),
        ("confirmatory", "primary_q6"),
    }
    if (role, scope) not in allowed:
        msg = "External entropy slot is not an authorized WP22 target population."
        raise ValueError(msg)
    return role, scope


def _entropy_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    """Return the complete stable identity required for an entropy file."""
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_nlink


def _validate_entropy_file_metadata(metadata: os.stat_result, slot: tuple[str, str]) -> None:
    """Require one exact, single-link regular entropy file.

    Confirmatory entropy has stricter custody than development and screening
    entropy: no group or other permission bit may be set.

    Raises:
        ValueError: If the file type, size, link count, or confirmatory mode is
            unsafe.
    """
    confirmatory_is_shared = stat.S_IMODE(metadata.st_mode) & (stat.S_IRWXG | stat.S_IRWXO)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size not in {32, 64}
        or (slot == ("confirmatory", "primary_q6") and confirmatory_is_shared)
    ):
        msg = "External entropy source is unavailable or unsafe."
        raise ValueError(msg)


def _stat_external_entropy_file(path: Path) -> os.stat_result:
    """Return no-follow path metadata with a redacted failure.

    Returns:
        Exact metadata for the final path component.

    Raises:
        ValueError: If the path cannot be inspected safely.
    """
    try:
        return path.stat(follow_symlinks=False)
    except OSError:
        msg = "External entropy source is unavailable or unsafe."
        raise ValueError(msg) from None


def _open_external_entropy_file(path: Path) -> int:
    """Open one entropy path through an alias-resistant descriptor.

    Returns:
        A caller-owned read-only descriptor.

    Raises:
        ValueError: If the source cannot be opened safely.
    """
    flags = os.O_RDONLY
    for optional_flag in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK", "O_BINARY"):
        flags |= cast("int", getattr(os, optional_flag, 0))
    try:
        return os.open(path, flags)
    except OSError:
        msg = "External entropy source is unavailable or unsafe."
        raise ValueError(msg) from None


def _fstat_external_entropy_file(descriptor: int) -> os.stat_result:
    """Return descriptor metadata with a redacted failure.

    Returns:
        Exact metadata for the opened file description.

    Raises:
        ValueError: If the descriptor cannot be inspected safely.
    """
    try:
        return os.fstat(descriptor)
    except OSError:
        msg = "External entropy source is unavailable or unsafe."
        raise ValueError(msg) from None


def _read_external_entropy_descriptor(descriptor: int) -> bytes:
    """Read at most one byte beyond the largest accepted key encoding.

    Returns:
        At most 65 bytes read from the pinned descriptor.

    Raises:
        ValueError: If the descriptor cannot be read safely.
    """
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read(65)
    except OSError:
        msg = "External entropy source is unavailable or unsafe."
        raise ValueError(msg) from None


def _read_pinned_external_entropy_file(
    descriptor: int,
    before: os.stat_result,
    slot: tuple[str, str],
) -> tuple[bytes, tuple[int, int, int, int]]:
    """Read and recheck one descriptor against its pre-open identity.

    Returns:
        The bounded payload and stable pre-open identity.

    Raises:
        ValueError: If the opened file is unsafe or changes while read.
    """
    identity = _entropy_file_identity(before)
    opened = _fstat_external_entropy_file(descriptor)
    _validate_entropy_file_metadata(opened, slot)
    if _entropy_file_identity(opened) != identity:
        msg = "External entropy source is unavailable or unsafe."
        raise ValueError(msg)
    payload = _read_external_entropy_descriptor(descriptor)
    closed = _fstat_external_entropy_file(descriptor)
    _validate_entropy_file_metadata(closed, slot)
    if _entropy_file_identity(closed) != identity:
        msg = "External entropy source is unavailable or unsafe."
        raise ValueError(msg)
    return payload, identity


def _read_external_entropy_file(path: Path, slot: tuple[str, str]) -> bytes:
    """Read one entropy key through a pinned no-follow descriptor.

    Returns:
        Exactly 32 raw bytes or 64 lowercase hexadecimal bytes.

    Raises:
        ValueError: If the source is unavailable, unsafe, linked, changes while
            read, or has an unsupported size.
    """
    before = _stat_external_entropy_file(path)
    _validate_entropy_file_metadata(before, slot)
    descriptor = _open_external_entropy_file(path)
    try:
        payload, identity = _read_pinned_external_entropy_file(descriptor, before, slot)
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)
    after = _stat_external_entropy_file(path)
    _validate_entropy_file_metadata(after, slot)
    if _entropy_file_identity(after) != identity or len(payload) != before.st_size:
        msg = "External entropy source is unavailable or unsafe."
        raise ValueError(msg)
    return payload


class ExternalEntropyKeyring:
    """Non-serializable in-memory custody for independent role-master keys.

    The object exposes only exact slot lookup and public SHA-256 commitments.
    Its representation contains neither bytes, encodings, paths, nor
    commitments, preventing accidental disclosure through ordinary logging.
    """

    __slots__ = ("_entropy",)

    def __init__(self, entropy_by_slot: Mapping[tuple[str, str], bytes]) -> None:
        """Copy and validate externally supplied opaque entropy.

        Args:
            entropy_by_slot: Public role/scope slots mapped to raw or lowercase
                hexadecimal 256-bit keys encoded as bytes.

        Raises:
            TypeError: If the mapping or a key has an unsupported type.
            ValueError: If a slot, key width, or duplicate secret is invalid.
        """
        if not isinstance(entropy_by_slot, Mapping) or not entropy_by_slot:
            msg = "entropy_by_slot must be a nonempty mapping."
            raise TypeError(msg)
        normalized: dict[tuple[str, str], bytes] = {}
        for raw_slot, raw_entropy in entropy_by_slot.items():
            if type(raw_slot) is not tuple or len(raw_slot) != 2:
                msg = "Every external entropy key requires an exact role/scope slot."
                raise TypeError(msg)
            slot = _entropy_slot(raw_slot[0], raw_slot[1])
            normalized[slot] = _normalize_entropy(raw_entropy)
        if len(set(normalized.values())) != len(normalized):
            msg = "Independent target populations cannot reuse role-master entropy."
            raise ValueError(msg)
        self._entropy = MappingProxyType(normalized)

    @classmethod
    def from_files(
        cls,
        entropy_files: Mapping[tuple[str, str], Path],
    ) -> ExternalEntropyKeyring:
        """Load exact small regular files without decoding JSON or text metadata.

        Args:
            entropy_files: Public role/scope slots mapped to private key files.

        Returns:
            A redacted, non-serializable keyring.

        Raises:
            TypeError: If the mapping or a path has an unsupported type.
        """
        if not isinstance(entropy_files, Mapping) or not entropy_files:
            msg = "entropy_files must be a nonempty mapping."
            raise TypeError(msg)
        loaded: dict[tuple[str, str], bytes] = {}
        for raw_slot, path in entropy_files.items():
            if not isinstance(path, Path):
                msg = "Every external entropy source must be a pathlib.Path."
                raise TypeError(msg)
            slot = _entropy_slot(*raw_slot)
            payload = _read_external_entropy_file(path, slot)
            loaded[slot] = _normalize_entropy(payload)
        return cls(loaded)

    def entropy_for(self, data_role: str, population_scope: str) -> bytes:
        """Return the exact role key for an authorized public slot.

        Returns:
            The immutable 32-byte role key.

        Raises:
            KeyError: If the required independent role key is absent.
        """
        slot = _entropy_slot(data_role, population_scope)
        try:
            return self._entropy[slot]
        except KeyError:
            msg = "Required external entropy slot is absent."
            raise KeyError(msg) from None

    def commitment_for(self, data_role: str, population_scope: str) -> str:
        """Return only the public commitment for one slot.

        Returns:
            The prefixed SHA-256 role-key commitment.
        """
        return role_master_entropy_commitment(self.entropy_for(data_role, population_scope))

    def __repr__(self) -> str:
        """Return a constant redacted representation."""
        return "ExternalEntropyKeyring(<redacted>)"

    __str__ = __repr__

    def __getstate__(self) -> NoReturn:
        """Reject state extraction used by serializers."""
        _serialization_forbidden()

    def __reduce__(self) -> NoReturn:
        """Reject pickle reduction."""
        _serialization_forbidden()

    def __reduce_ex__(self, _protocol: SupportsIndex) -> NoReturn:
        """Reject protocol-specific pickle reduction."""
        _serialization_forbidden()


@dataclass(frozen=True, slots=True)
class TrainingCandidateRef:
    """Minimal complete publication-candidate identity retained by a context."""

    candidate_schema_version: str
    candidate_checksum: str
    method_id: str
    schema_version: str = field(default=TRAINING_CANDIDATE_REF_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the schema, checksum, and method identity."""
        object.__setattr__(
            self,
            "candidate_schema_version",
            require_slug(self.candidate_schema_version, "candidate_schema_version"),
        )
        object.__setattr__(self, "candidate_checksum", require_checksum(self.candidate_checksum, "candidate_checksum"))
        object.__setattr__(self, "method_id", require_slug(self.method_id, "method_id"))

    @classmethod
    def from_binding(cls, executable_binding: ExecutableScopedBinding) -> TrainingCandidateRef:
        """Project one executable binding onto its publication identity.

        Returns:
            The non-executable candidate reference.

        Raises:
            TypeError: If the input is not an executable scoped binding.
        """
        if not isinstance(executable_binding, ExecutableScopedBinding):
            msg = "executable_binding must be an ExecutableScopedBinding."
            raise TypeError(msg)
        binding = executable_binding.binding
        return cls(
            candidate_schema_version=binding.publication_candidate_schema_version,
            candidate_checksum=binding.publication_candidate_checksum,
            method_id=binding.publication_method_id,
        )

    def _payload(self) -> dict[str, object]:
        """Return all checksum-covered fields."""
        return {
            "schema_version": self.schema_version,
            "candidate_schema_version": self.candidate_schema_version,
            "candidate_checksum": self.candidate_checksum,
            "method_id": self.method_id,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the publication-candidate reference."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return {**self._payload(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> TrainingCandidateRef:
        """Decode and verify one candidate reference.

        Returns:
            The verified candidate reference.

        Raises:
            ValueError: If the schema or checksum seal is invalid.
        """
        mapping = verify_sealed_mapping(value, expected_keys=_CANDIDATE_KEYS, name="training candidate reference")
        if mapping["schema_version"] != TRAINING_CANDIDATE_REF_SCHEMA_VERSION:
            msg = "Training candidate reference uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            candidate_schema_version=cast("str", mapping["candidate_schema_version"]),
            candidate_checksum=cast("str", mapping["candidate_checksum"]),
            method_id=cast("str", mapping["method_id"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Training candidate reference checksum changed during normalization."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> TrainingCandidateRef:
        """Decode canonical JSON into a candidate reference.

        Returns:
            The verified candidate reference.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class AuthorizedTargetMaterialization:
    """One target config/manifest pair carrying its opaque authorization token."""

    target_configuration: TargetPopulationConfig
    target_manifest: TargetPopulationManifest
    authorization: TargetMaterializationAuthorization = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Require exact config, manifest, role, and authorization agreement.

        Raises:
            TypeError: If a nested record has the wrong typed schema.
            ValueError: If the config, manifest, and authorization differ.
        """
        if not isinstance(self.target_configuration, TargetPopulationConfig):
            msg = "target_configuration must be a TargetPopulationConfig."
            raise TypeError(msg)
        if not isinstance(self.target_manifest, TargetPopulationManifest):
            msg = "target_manifest must be a TargetPopulationManifest."
            raise TypeError(msg)
        if not isinstance(self.authorization, TargetMaterializationAuthorization):
            msg = "authorization must be a TargetMaterializationAuthorization."
            raise TypeError(msg)
        config = self.target_configuration
        manifest = self.target_manifest
        token = self.authorization
        if (
            manifest.population_config_checksum != config.content_checksum
            or manifest.preregistration_checksum != config.preregistration_checksum
            or manifest.data_role != config.data_role
            or manifest.population_scope != config.population_scope
            or manifest.role_master_entropy_commitment != config.role_master_entropy_commitment
            or token.preregistration_checksum != config.preregistration_checksum
            or token.population_config_checksum != config.content_checksum
            or token.target_manifest_checksum != manifest.content_checksum
            or token.data_role != config.data_role
        ):
            msg = "Target materialization authorization differs from its exact config or manifest."
            raise ValueError(msg)


def candidate_refs_from_bindings(
    executable_bindings: Sequence[ExecutableScopedBinding],
) -> tuple[TrainingCandidateRef, ...]:
    """Return the distinct candidates in first profile-occurrence order.

    Returns:
        Candidate references with q6/q12 projections deduplicated.
    """
    result: list[TrainingCandidateRef] = []
    seen: set[str] = set()
    for executable_binding in executable_bindings:
        candidate = TrainingCandidateRef.from_binding(executable_binding)
        if candidate.candidate_checksum not in seen:
            result.append(candidate)
            seen.add(candidate.candidate_checksum)
    return tuple(result)


def schedules_from_bindings(
    executable_bindings: Sequence[ExecutableScopedBinding],
) -> tuple[TrainingStrategySchedule, ...]:
    """Return distinct schedules in first profile-occurrence order.

    Returns:
        The complete deduplicated typed schedule universe.
    """
    result: list[TrainingStrategySchedule] = []
    seen: set[str] = set()
    for executable_binding in executable_bindings:
        schedule = executable_binding.binding.strategy_schedule
        if schedule.content_checksum not in seen:
            result.append(schedule)
            seen.add(schedule.content_checksum)
    return tuple(result)


def source_fingerprint_checksum(
    execution_source_manifest: ExecutionSourceManifest,
    resumability_fingerprints: Sequence[ResumabilityFingerprint],
) -> str:
    """Checksum the exact source lock and ordered resumability contexts.

    Returns:
        The non-secret combined source-fingerprint root.

    Raises:
        TypeError: If the source manifest or fingerprint records are untyped.
        ValueError: If resumability fingerprints are duplicated.
    """
    if not isinstance(execution_source_manifest, ExecutionSourceManifest):
        msg = "execution_source_manifest must be an ExecutionSourceManifest."
        raise TypeError(msg)
    fingerprints = tuple(resumability_fingerprints)
    if not fingerprints or any(not isinstance(item, ResumabilityFingerprint) for item in fingerprints):
        msg = "resumability_fingerprints must contain typed nonempty records."
        raise TypeError(msg)
    if len({item.content_checksum for item in fingerprints}) != len(fingerprints):
        msg = "resumability_fingerprints must be checksum-distinct."
        raise ValueError(msg)
    ordered = tuple(sorted(fingerprints, key=lambda item: (item.pipeline_prefix_id, item.content_checksum)))
    return canonical_checksum({
        "execution_source_manifest_checksum": execution_source_manifest.content_checksum,
        "resumability_fingerprint_checksums": [item.content_checksum for item in ordered],
    })


def validate_resumability_source_fingerprints(
    execution_source_manifest: ExecutionSourceManifest,
    resumability_fingerprints: Sequence[ResumabilityFingerprint],
) -> None:
    """Cross-link resumability entries to the immutable execution source lock.

    Raises:
        ValueError: If a fingerprint commit, role, path, or byte checksum differs.
    """
    source_fingerprint_checksum(execution_source_manifest, resumability_fingerprints)
    source_by_path = {item.repo_path: item for item in execution_source_manifest.source_files}
    expected_source_role = {
        "execution_source": "execution_source",
        "lockfile": "dependency_lock",
        "sealed_input": "sealed_input",
    }
    for fingerprint in resumability_fingerprints:
        if fingerprint.starting_commit != execution_source_manifest.source_commit:
            msg = "A resumability fingerprint starts from a different source commit."
            raise ValueError(msg)
        for entry in fingerprint.entries:
            source = source_by_path.get(entry.repository_path)
            if (
                source is None
                or source.role != expected_source_role[entry.role]
                or source.source_checksum != entry.content_checksum
            ):
                msg = "A resumability source entry is absent from or differs from the execution source lock."
                raise ValueError(msg)


def _target_scope(job: TrainingJob) -> str:
    """Return the exact binding scope for one job.

    Returns:
        ``primary_q6`` or ``secondary_q12``.

    Raises:
        ValueError: If the job width is outside the WP22 execution universe.
    """
    try:
        return _TARGET_SCOPE_BY_QUBIT_COUNT[job.qubit_count]
    except KeyError:
        msg = "WP22 execution bindings support only q6 and q12 jobs."
        raise ValueError(msg) from None


def _evaluation_policy(executable_binding: ExecutableScopedBinding, preset: str) -> FreshEvaluationPolicy:
    """Resolve the single preset-level fresh evaluation policy.

    Returns:
        The exact outer/fresh policy for the job.

    Raises:
        ValueError: If the preset or binding has no unique matching policy.
    """
    try:
        purpose = _EVALUATION_PURPOSE_BY_PRESET[preset]
    except KeyError:
        msg = "The preset does not have a WP22D executable-binding evaluation policy."
        raise ValueError(msg) from None
    policies = tuple(policy for policy in executable_binding.binding.evaluation_policies if policy.purpose == purpose)
    if len(policies) != 1:
        msg = "Executable binding does not contain exactly one preset-level fresh evaluation policy."
        raise ValueError(msg)
    return policies[0]


def _compiled_program(
    executable_binding: ExecutableScopedBinding,
    job: TrainingJob,
) -> ScheduledExecutionProgram:
    """Compile the exact WP22C program for one binding and optimization seed.

    Returns:
        The deterministic schedule program.
    """
    return ScheduledExecutionProgram.compile(
        executable_binding,
        executable_binding.binding.strategy_schedule,
        ScheduledJobSeedSet(job.optimization_seed),
    )


def bind_training_plan_fingerprints(
    plan: TrainingRunPlan,
    *,
    execution_profile: TrainingExecutionProfile,
    executable_bindings: Sequence[ExecutableScopedBinding],
    target_configurations: Sequence[TargetPopulationConfig],
    target_manifests: Sequence[TargetPopulationManifest],
    execution_source_manifest: ExecutionSourceManifest,
    resumability_fingerprints: Sequence[ResumabilityFingerprint],
    required_sample_size_design: SampleSizeDesign | None,
) -> TrainingRunPlan:
    """Copy every complete WP22D identity into a plan and all of its jobs.

    This function is pure.  It resolves all bindings and compiles each distinct
    binding/optimization-seed program before returning the newly addressed
    plan, so a mismatch cannot create an output directory.

    Returns:
        A new checksum-sealed plan with complete execution fingerprints.

    Raises:
        TypeError: If a nested input has the wrong typed schema.
        ValueError: If any job lacks an exact profile binding, config, policy,
            source fingerprint, or compilable WP22C schedule.
    """
    if not isinstance(plan, TrainingRunPlan):
        msg = "plan must be a TrainingRunPlan."
        raise TypeError(msg)
    if not isinstance(execution_profile, TrainingExecutionProfile):
        msg = "execution_profile must be a TrainingExecutionProfile."
        raise TypeError(msg)
    links = tuple(executable_bindings)
    if not links or any(not isinstance(link, ExecutableScopedBinding) for link in links):
        msg = "executable_bindings must contain ExecutableScopedBinding records."
        raise TypeError(msg)
    if tuple(link.binding for link in links) != execution_profile.bindings:
        msg = "Executable bindings must exactly and in order close the execution profile."
        raise ValueError(msg)
    if (
        plan.preset != execution_profile.preset
        or plan.preregistration_checksum != execution_profile.preregistration_checksum
    ):
        msg = "Training plan and execution profile preset or preregistration differ."
        raise ValueError(msg)
    configs = tuple(target_configurations)
    if not configs or any(not isinstance(config, TargetPopulationConfig) for config in configs):
        msg = "target_configurations must contain TargetPopulationConfig records."
        raise TypeError(msg)
    config_by_checksum = {config.content_checksum: config for config in configs}
    if len(config_by_checksum) != len(configs):
        msg = "target_configurations must be checksum-distinct."
        raise ValueError(msg)
    manifests = tuple(target_manifests)
    if not manifests or any(not isinstance(manifest, TargetPopulationManifest) for manifest in manifests):
        msg = "target_manifests must contain TargetPopulationManifest records."
        raise TypeError(msg)
    manifest_by_checksum = {manifest.content_checksum: manifest for manifest in manifests}
    if len(manifest_by_checksum) != len(manifests):
        msg = "target_manifests must be checksum-distinct."
        raise ValueError(msg)
    if tuple(manifest_by_checksum) != plan.target_manifest_checksums:
        msg = "target_manifests differ from the plan's ordered target roots."
        raise ValueError(msg)
    source_root = source_fingerprint_checksum(execution_source_manifest, resumability_fingerprints)
    validate_resumability_source_fingerprints(execution_source_manifest, resumability_fingerprints)
    link_by_key: dict[tuple[str, str], ExecutableScopedBinding] = {
        (link.binding.publication_candidate_checksum, link.binding.target_scope_id): link for link in links
    }
    if len(link_by_key) != len(links):
        msg = "Executable binding keys must be unique."
        raise ValueError(msg)
    program_cache: dict[tuple[str, int], ScheduledExecutionProgram] = {}
    jobs = []
    for job in plan.jobs:
        key = (job.candidate_configuration_checksum, _target_scope(job))
        try:
            link = link_by_key[key]
        except KeyError:
            msg = "A planned job has no exact executable scoped binding."
            raise ValueError(msg) from None
        binding = link.binding
        try:
            manifest = manifest_by_checksum[job.target_manifest_checksum]
            config = config_by_checksum[manifest.population_config_checksum]
        except KeyError:
            msg = "A planned job has no exact target-population configuration and manifest pair."
            raise ValueError(msg) from None
        policy = _evaluation_policy(link, plan.preset)
        program_key = (link.content_checksum, job.optimization_seed)
        program = program_cache.get(program_key)
        if program is None:
            program = _compiled_program(link, job)
            program_cache[program_key] = program
        rebound = replace(
            job,
            method_id=binding.publication_method_id,
            implementation_kind=(
                "phase2_pipeline"
                if binding.implementation_artifact.implementation_kind.startswith("phase2_pipeline")
                else "operator_growth"
            ),
            candidate_configuration_checksum=binding.publication_candidate_checksum,
            implementation_checksum=binding.implementation_checksum,
            strategy_schedule_checksum=binding.strategy_schedule.content_checksum,
            strategy_schedule=binding.strategy_schedule,
            execution_profile_checksum=execution_profile.content_checksum,
            scoped_binding_checksum=binding.content_checksum,
            executable_binding_checksum=link.content_checksum,
            evaluation_policy_checksum=policy.content_checksum,
            target_configuration_checksum=config.content_checksum,
            source_fingerprint_checksum=source_root,
            scheduled_execution_program_checksum=program.content_checksum,
        )
        address_payload = {
            key: value
            for key, value in rebound.to_dict().items()
            if key not in {"content_checksum", "job_id", "output_path"}
        }
        job_id = f"wp22_job_{canonical_checksum(address_payload).removeprefix('sha256:')}"
        jobs.append(
            replace(
                rebound,
                job_id=job_id,
                output_path=(f"roles/{rebound.data_role}/{rebound.family_id}/{rebound.target_instance_id}/{job_id}"),
            )
        )
    design_checksum = None if required_sample_size_design is None else required_sample_size_design.content_checksum
    return replace(
        plan,
        execution_profile_checksum=execution_profile.content_checksum,
        scoped_binding_checksums=tuple(sorted({cast("str", job.scoped_binding_checksum) for job in jobs})),
        executable_binding_checksums=tuple(sorted({cast("str", job.executable_binding_checksum) for job in jobs})),
        implementation_checksums=tuple(sorted({job.implementation_checksum for job in jobs})),
        evaluation_policy_checksums=tuple(sorted({cast("str", job.evaluation_policy_checksum) for job in jobs})),
        target_configuration_checksums=tuple(sorted({cast("str", job.target_configuration_checksum) for job in jobs})),
        source_fingerprint_checksums=(source_root,),
        scheduled_execution_program_checksums=tuple(
            sorted({cast("str", job.scheduled_execution_program_checksum) for job in jobs})
        ),
        sample_size_design_checksum=design_checksum,
        execution_source_checksum=execution_source_manifest.content_checksum,
        jobs=tuple(sorted(jobs, key=lambda job: job.sort_key)),
    )


@dataclass(frozen=True, slots=True)
class TrainingPreflightReport:
    """Non-secret checksum report produced only after complete preflight."""

    plan_checksum: str
    execution_profile_checksum: str
    execution_source_manifest_checksum: str
    source_fingerprint_checksum: str
    job_count: int
    target_population_count: int
    compiled_program_count: int
    schema_version: str = field(default=TRAINING_PREFLIGHT_REPORT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate every public checksum and positive count.

        Raises:
            ValueError: If a checksum or count is invalid.
        """
        for name in (
            "plan_checksum",
            "execution_profile_checksum",
            "execution_source_manifest_checksum",
            "source_fingerprint_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        for name in ("job_count", "target_population_count", "compiled_program_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                msg = f"{name} must be a positive integer."
                raise ValueError(msg)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered public field."""
        return {
            "schema_version": self.schema_version,
            "plan_checksum": self.plan_checksum,
            "execution_profile_checksum": self.execution_profile_checksum,
            "execution_source_manifest_checksum": self.execution_source_manifest_checksum,
            "source_fingerprint_checksum": self.source_fingerprint_checksum,
            "job_count": self.job_count,
            "target_population_count": self.target_population_count,
            "compiled_program_count": self.compiled_program_count,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete public preflight report."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return {**self._payload(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> TrainingPreflightReport:
        """Decode and verify a preflight report.

        Returns:
            The verified public report.

        Raises:
            ValueError: If the schema or checksum seal is invalid.
        """
        mapping = verify_sealed_mapping(value, expected_keys=_PREFLIGHT_KEYS, name="training preflight report")
        if mapping["schema_version"] != TRAINING_PREFLIGHT_REPORT_SCHEMA_VERSION:
            msg = "Training preflight report uses an unsupported schema version."
            raise ValueError(msg)
        report = cls(
            plan_checksum=cast("str", mapping["plan_checksum"]),
            execution_profile_checksum=cast("str", mapping["execution_profile_checksum"]),
            execution_source_manifest_checksum=cast("str", mapping["execution_source_manifest_checksum"]),
            source_fingerprint_checksum=cast("str", mapping["source_fingerprint_checksum"]),
            job_count=cast("int", mapping["job_count"]),
            target_population_count=cast("int", mapping["target_population_count"]),
            compiled_program_count=cast("int", mapping["compiled_program_count"]),
        )
        if mapping["content_checksum"] != report.content_checksum:
            msg = "Training preflight report checksum changed during normalization."
            raise ValueError(msg)
        return report


@dataclass(frozen=True, slots=True)
class TrainingExecutionContext:
    """Complete non-serializable scientific authority for one WP22 run plan."""

    plan: TrainingRunPlan
    execution_profile: TrainingExecutionProfile
    preregistration: InitialPreregistration
    candidates: tuple[TrainingCandidateRef, ...]
    schedules: tuple[TrainingStrategySchedule, ...]
    scoped_bindings: tuple[ExecutableScopedBinding, ...]
    target_configurations: tuple[TargetPopulationConfig, ...]
    target_manifests: tuple[TargetPopulationManifest, ...]
    authorized_materializations: tuple[AuthorizedTargetMaterialization, ...]
    screening_manifest: ScreeningManifest | None
    screening_cells: tuple[ScreeningCell, ...]
    required_sample_size_design: SampleSizeDesign | None
    execution_source_manifest: ExecutionSourceManifest
    resumability_fingerprints: tuple[ResumabilityFingerprint, ...]
    external_entropy_keyring: ExternalEntropyKeyring = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate structural closure without reading source bytes or output.

        Raises:
            TypeError: If a nested context artifact has the wrong typed schema.
            ValueError: If any context identity or preset invariant differs.
        """
        if not isinstance(self.plan, TrainingRunPlan):
            msg = "plan must be a TrainingRunPlan."
            raise TypeError(msg)
        if not isinstance(self.execution_profile, TrainingExecutionProfile):
            msg = "execution_profile must be a TrainingExecutionProfile."
            raise TypeError(msg)
        if not isinstance(self.preregistration, InitialPreregistration):
            msg = "preregistration must be an InitialPreregistration."
            raise TypeError(msg)
        if not isinstance(self.external_entropy_keyring, ExternalEntropyKeyring):
            msg = "external_entropy_keyring must be an ExternalEntropyKeyring."
            raise TypeError(msg)
        links = tuple(self.scoped_bindings)
        if tuple(link.binding for link in links) != self.execution_profile.bindings:
            msg = "scoped_bindings must exactly close every execution-profile binding in order."
            raise ValueError(msg)
        expected_candidates = candidate_refs_from_bindings(links)
        if tuple(self.candidates) != expected_candidates:
            msg = "candidates differ from the execution profile's exact publication universe."
            raise ValueError(msg)
        expected_schedules = schedules_from_bindings(links)
        if tuple(self.schedules) != expected_schedules:
            msg = "schedules differ from the execution profile's exact schedule universe."
            raise ValueError(msg)
        configs = tuple(self.target_configurations)
        manifests = tuple(self.target_manifests)
        authorized = tuple(self.authorized_materializations)
        if any(not isinstance(item, TargetPopulationConfig) for item in configs):
            msg = "target_configurations must contain TargetPopulationConfig records."
            raise TypeError(msg)
        if any(not isinstance(item, TargetPopulationManifest) for item in manifests):
            msg = "target_manifests must contain TargetPopulationManifest records."
            raise TypeError(msg)
        if any(not isinstance(item, AuthorizedTargetMaterialization) for item in authorized):
            msg = "authorized_materializations must contain typed authorization records."
            raise TypeError(msg)
        if (
            tuple(item.target_configuration for item in authorized) != configs
            or tuple(item.target_manifest for item in authorized) != manifests
        ):
            msg = "Authorized target materializations differ from the ordered config/manifest universe."
            raise ValueError(msg)
        if self.plan.preset != self.execution_profile.preset:
            msg = "Plan and execution profile presets differ."
            raise ValueError(msg)
        if (
            self.plan.preregistration_checksum != self.preregistration.content_checksum
            or self.execution_profile.preregistration_checksum != self.preregistration.content_checksum
        ):
            msg = "Plan, profile, and preregistration checksums differ."
            raise ValueError(msg)
        if self.plan.target_manifest_checksums != tuple(manifest.content_checksum for manifest in manifests):
            msg = "Plan target-manifest roots differ from the ordered context manifests."
            raise ValueError(msg)
        expected_count = _EXACT_JOB_COUNT_BY_PRESET.get(self.plan.preset)
        if expected_count is not None and len(self.plan.jobs) != expected_count:
            msg = f"{self.plan.preset} requires exactly {expected_count} jobs."
            raise ValueError(msg)
        if self.plan.preset == "paper-screen":
            if not isinstance(self.screening_manifest, ScreeningManifest):
                msg = "paper-screen requires its complete ScreeningManifest."
                raise TypeError(msg)
            if tuple(self.screening_cells) != self.screening_manifest.cells:
                msg = "screening_cells must equal the complete ScreeningManifest cell sequence."
                raise ValueError(msg)
            if not isinstance(self.required_sample_size_design, SampleSizeDesign):
                msg = "paper-screen requires the pilot-derived SampleSizeDesign."
                raise TypeError(msg)
            if self.plan.screening_manifest_checksum != self.screening_manifest.content_checksum:
                msg = "Plan and context screening-manifest checksums differ."
                raise ValueError(msg)
            if self.plan.sample_size_design_checksum != self.required_sample_size_design.content_checksum:
                msg = "Plan and context sample-size-design checksums differ."
                raise ValueError(msg)
            expected_screen_policies = (
                canonical_checksum({
                    "endpoint": self.preregistration.primary_endpoint,
                    "failure_policy": self.preregistration.failure_policy,
                    "noise": self.preregistration.primary_noise_condition,
                }),
                canonical_checksum(self.preregistration.primary_resource_constraint),
            )
            if (
                self.screening_manifest.evaluation_policy_checksum,
                self.screening_manifest.resource_policy_checksum,
            ) != expected_screen_policies:
                msg = "Screening evaluation or resource policy differs from the preregistration."
                raise ValueError(msg)
            if self.required_sample_size_design.preregistration_checksum != self.preregistration.content_checksum:
                msg = "Sample-size design belongs to a different preregistration."
                raise ValueError(msg)
        elif self.screening_manifest is not None or self.screening_cells:
            msg = "Screening manifest and cells are accepted only by paper-screen."
            raise ValueError(msg)
        elif self.required_sample_size_design is not None:
            msg = "A sample-size design is accepted only by paper-screen in the WP22D profile universe."
            raise ValueError(msg)
        if not isinstance(self.execution_source_manifest, ExecutionSourceManifest):
            msg = "execution_source_manifest must be an ExecutionSourceManifest."
            raise TypeError(msg)
        raw_fingerprints = tuple(self.resumability_fingerprints)
        source_root = source_fingerprint_checksum(self.execution_source_manifest, raw_fingerprints)
        fingerprints = tuple(
            sorted(
                raw_fingerprints,
                key=lambda item: (item.pipeline_prefix_id, item.content_checksum),
            )
        )
        object.__setattr__(self, "resumability_fingerprints", fingerprints)
        validate_resumability_source_fingerprints(self.execution_source_manifest, fingerprints)
        if self.plan.execution_profile_checksum != self.execution_profile.content_checksum:
            msg = "Plan does not bind the exact execution profile."
            raise ValueError(msg)
        if self.plan.execution_source_checksum != self.execution_source_manifest.content_checksum:
            msg = "Plan does not bind the exact execution-source manifest."
            raise ValueError(msg)
        if self.plan.source_fingerprint_checksums != (source_root,):
            msg = "Plan does not bind the exact combined source fingerprint."
            raise ValueError(msg)

    @property
    def source_fingerprint_checksum(self) -> str:
        """Combined public source and resumability root."""
        return source_fingerprint_checksum(self.execution_source_manifest, self.resumability_fingerprints)

    def _validate_entropy_and_authorization(self) -> None:
        """Verify every required key commitment and opaque authorization link.

        Raises:
            ValueError: If an external role key differs from its commitment.
        """
        for item in self.authorized_materializations:
            config = item.target_configuration
            commitment = self.external_entropy_keyring.commitment_for(config.data_role, config.population_scope)
            if commitment != config.role_master_entropy_commitment:
                msg = "External entropy does not match a target-population commitment."
                raise ValueError(msg)

    def _validate_source_fingerprints(self) -> None:
        """Cross-link resumability entries to the immutable execution source lock."""
        validate_resumability_source_fingerprints(
            self.execution_source_manifest,
            self.resumability_fingerprints,
        )

    def _validate_jobs_and_programs(self) -> int:
        """Recompile every distinct program and verify all job checksum aliases.

        Returns:
            The number of distinct compiled programs.

        Raises:
            ValueError: If any job fingerprint, target, policy, or program differs.
        """
        links: dict[tuple[str, str], ExecutableScopedBinding] = {
            (link.binding.publication_candidate_checksum, link.binding.target_scope_id): link
            for link in self.scoped_bindings
        }
        configs = {config.content_checksum: config for config in self.target_configurations}
        manifests = {manifest.content_checksum: manifest for manifest in self.target_manifests}
        cache: dict[tuple[str, int], ScheduledExecutionProgram] = {}
        for job in self.plan.jobs:
            try:
                link = links[job.candidate_configuration_checksum, _target_scope(job)]
                config = configs[cast("str", job.target_configuration_checksum)]
                manifest = manifests[job.target_manifest_checksum]
            except KeyError:
                msg = "A planned job is outside the context binding or target universe."
                raise ValueError(msg) from None
            binding = link.binding
            if manifest.population_config_checksum != config.content_checksum:
                msg = "A job target manifest differs from its target configuration."
                raise ValueError(msg)
            target = next(
                (item for item in manifest.instances if item.target_instance_id == job.target_instance_id),
                None,
            )
            if target is None or target.content_checksum != job.target_spec_checksum:
                msg = "A job target is absent from its exact target manifest."
                raise ValueError(msg)
            policy = _evaluation_policy(link, self.plan.preset)
            program_key = (link.content_checksum, job.optimization_seed)
            program = cache.get(program_key)
            if program is None:
                program = _compiled_program(link, job)
                cache[program_key] = program
            expected = (
                (job.execution_profile_checksum, self.execution_profile.content_checksum),
                (job.scoped_binding_checksum, binding.content_checksum),
                (job.executable_binding_checksum, link.content_checksum),
                (job.implementation_checksum, binding.implementation_checksum),
                (job.strategy_schedule_checksum, binding.strategy_schedule.content_checksum),
                (job.evaluation_policy_checksum, policy.content_checksum),
                (job.target_configuration_checksum, config.content_checksum),
                (job.source_fingerprint_checksum, self.source_fingerprint_checksum),
                (job.scheduled_execution_program_checksum, program.content_checksum),
            )
            if (
                any(actual != required for actual, required in expected)
                or job.strategy_schedule != binding.strategy_schedule
            ):
                msg = "A job fingerprint differs from its complete execution context."
                raise ValueError(msg)
        if self.plan.preset == "paper-screen":
            design = cast("SampleSizeDesign", self.required_sample_size_design)
            outer_counts = {_evaluation_policy(link, "paper-screen").trajectory_count for link in self.scoped_bindings}
            if outer_counts != {design.fixed_test_trajectory_count}:
                msg = "Screening outer trajectory count differs from the required sample-size design."
                raise ValueError(msg)
        return len(cache)

    def preflight(self, repository_root: Path, output_root: Path) -> TrainingPreflightReport:
        """Verify complete custody, source bytes, and schedules before mutation.

        Args:
            repository_root: Exact checkout containing the sealed source bytes.
            output_root: Prospective output location; it is inspected but never
                created or modified by preflight.

        Returns:
            A non-secret checksum report.

        Raises:
            TypeError: If either path has the wrong type.
            ValueError: If any context, source, authorization, schedule, count,
                or existing output path is unsafe.
        """
        if not isinstance(repository_root, Path) or not isinstance(output_root, Path):
            msg = "repository_root and output_root must be pathlib.Path values."
            raise TypeError(msg)
        if output_root.is_symlink() or (output_root.exists() and not output_root.is_dir()):
            msg = "output_root must be absent or an existing non-symlink directory."
            raise ValueError(msg)
        self._validate_entropy_and_authorization()
        self._validate_source_fingerprints()
        verify_execution_source_manifest(self.execution_source_manifest, repository_root)
        compiled = self._validate_jobs_and_programs()
        return TrainingPreflightReport(
            plan_checksum=self.plan.content_checksum,
            execution_profile_checksum=self.execution_profile.content_checksum,
            execution_source_manifest_checksum=self.execution_source_manifest.content_checksum,
            source_fingerprint_checksum=self.source_fingerprint_checksum,
            job_count=len(self.plan.jobs),
            target_population_count=len(self.target_manifests),
            compiled_program_count=compiled,
        )

    def __repr__(self) -> str:
        """Return only public roots and counts, never entropy-bearing members."""
        return (
            "TrainingExecutionContext("
            f"plan_checksum={self.plan.content_checksum!r}, "
            f"execution_profile_checksum={self.execution_profile.content_checksum!r}, "
            f"job_count={len(self.plan.jobs)}, target_population_count={len(self.target_manifests)}, "
            "external_entropy_keyring=<redacted>)"
        )

    def __getstate__(self) -> NoReturn:
        """Reject state extraction used by serializers."""
        _serialization_forbidden()

    def __reduce__(self) -> NoReturn:
        """Reject pickle reduction."""
        _serialization_forbidden()

    def __reduce_ex__(self, _protocol: SupportsIndex) -> NoReturn:
        """Reject protocol-specific pickle reduction."""
        _serialization_forbidden()


@dataclass(frozen=True, slots=True)
class ConfirmationExecutionContext:
    """Narrow non-serializable authority for frozen real confirmation.

    The context is deliberately distinct from :class:`TrainingExecutionContext`:
    confirmation reuses the exact promoted paper-screen implementations and
    schedules, but it has no execution profile, screening cell, or opportunity
    to introduce a new training or evaluation policy.  Target vectors remain
    unmaterialized until :meth:`materialize_targets` is called by the
    repository-owned production executor.
    """

    plan: TrainingRunPlan
    preregistration: InitialPreregistration
    final_seal: FinalConfirmationSeal
    configuration_execution_manifest: FinalConfigurationExecutionManifest
    execution_source_manifest: ExecutionSourceManifest
    analysis_source_manifest: AnalysisSourceManifest
    repository_binding_catalog: RepositoryBindingCatalog
    target_configuration: TargetPopulationConfig
    target_manifest: TargetPopulationManifest
    prior_target_exposure_inventory_checksum: str
    authorized_output_root: Path
    locked_study_head_custody_path: Path
    confirmation_authorization: ConfirmationAuthorization = field(repr=False)
    target_materialization_authorization: TargetMaterializationAuthorization = field(repr=False)
    external_entropy_keyring: ExternalEntropyKeyring = field(repr=False, compare=False)
    _selected_bindings: tuple[ExecutableScopedBinding, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Close every final-seal, executable, target, and opaque authority link.

        Raises:
            TypeError: If a nested value does not use its exact protocol type.
            ValueError: If any final-seal, catalog, target, or authorization
                identity differs from the already sealed confirmatory design.
        """
        typed_values = (
            (self.plan, TrainingRunPlan, "plan"),
            (self.preregistration, InitialPreregistration, "preregistration"),
            (self.final_seal, FinalConfirmationSeal, "final_seal"),
            (
                self.configuration_execution_manifest,
                FinalConfigurationExecutionManifest,
                "configuration_execution_manifest",
            ),
            (self.execution_source_manifest, ExecutionSourceManifest, "execution_source_manifest"),
            (self.analysis_source_manifest, AnalysisSourceManifest, "analysis_source_manifest"),
            (self.repository_binding_catalog, RepositoryBindingCatalog, "repository_binding_catalog"),
            (self.target_configuration, TargetPopulationConfig, "target_configuration"),
            (self.target_manifest, TargetPopulationManifest, "target_manifest"),
            (self.authorized_output_root, Path, "authorized_output_root"),
            (
                self.locked_study_head_custody_path,
                Path,
                "locked_study_head_custody_path",
            ),
            (self.confirmation_authorization, ConfirmationAuthorization, "confirmation_authorization"),
            (
                self.target_materialization_authorization,
                TargetMaterializationAuthorization,
                "target_materialization_authorization",
            ),
            (self.external_entropy_keyring, ExternalEntropyKeyring, "external_entropy_keyring"),
        )
        for value, expected_type, name in typed_values:
            if not isinstance(value, expected_type):
                msg = f"{name} must be a {expected_type.__name__}."
                raise TypeError(msg)

        object.__setattr__(
            self,
            "prior_target_exposure_inventory_checksum",
            require_checksum(
                self.prior_target_exposure_inventory_checksum,
                "prior_target_exposure_inventory_checksum",
            ),
        )
        self._validate_authorized_output_root(self.authorized_output_root)
        self._validate_locked_study_head_custody_path(
            self.locked_study_head_custody_path,
            self.authorized_output_root,
        )

        seal = self.final_seal
        plan = self.plan
        target = self.target_manifest
        config = self.target_configuration
        if (
            plan.preset != "paper-confirm"
            or plan.preregistration_checksum != self.preregistration.content_checksum
            or plan.final_confirmation_seal_checksum != seal.content_checksum
            or plan.execution_source_checksum != seal.execution_source_checksum
            or plan.target_manifest_checksums != (target.content_checksum,)
        ):
            msg = "Confirmation plan does not reproduce the exact final-seal and target roots."
            raise ValueError(msg)
        expected_plan = build_paper_confirm_plan(
            seal=seal,
            target_manifest=target,
            configuration_execution_manifest=self.configuration_execution_manifest,
        )
        if plan != expected_plan:
            msg = "Confirmation plan differs from the exact final-seal Cartesian request universe."
            raise ValueError(msg)
        validate_final_configuration_execution_manifest(seal, self.configuration_execution_manifest)
        if (
            self.execution_source_manifest.content_checksum != seal.execution_source_checksum
            or self.analysis_source_manifest.content_checksum != seal.analysis_source_manifest_checksum
            or self.analysis_source_manifest.execution_source_manifest_checksum
            != self.execution_source_manifest.content_checksum
        ):
            msg = "Confirmation execution and analysis source manifests differ from the final seal."
            raise ValueError(msg)
        if (
            config.data_role != "confirmatory"
            or config.population_scope != "primary_q6"
            or target.data_role != "confirmatory"
            or target.population_scope != "primary_q6"
            or target.population_config_checksum != config.content_checksum
            or config.preregistration_checksum != self.preregistration.content_checksum
            or target.preregistration_checksum != self.preregistration.content_checksum
        ):
            msg = "Confirmation target configuration and revealed manifest do not form the exact primary-q6 pair."
            raise ValueError(msg)
        confirmation = self.confirmation_authorization
        materialization = self.target_materialization_authorization
        if (
            confirmation.preregistration_checksum != self.preregistration.content_checksum
            or confirmation.final_seal_checksum != seal.content_checksum
            or confirmation.target_manifest_checksum != target.content_checksum
            or confirmation.execution_source_checksum != seal.execution_source_checksum
            or materialization.preregistration_checksum != self.preregistration.content_checksum
            or materialization.population_config_checksum != config.content_checksum
            or materialization.target_manifest_checksum != target.content_checksum
            or materialization.data_role != "confirmatory"
        ):
            msg = "Confirmation or target-materialization authority differs from the exact sealed target."
            raise ValueError(msg)
        if (
            self.external_entropy_keyring.commitment_for("confirmatory", "primary_q6")
            != config.role_master_entropy_commitment
        ):
            msg = "Confirmatory external entropy differs from its sealed target-population commitment."
            raise ValueError(msg)

        catalog = self.repository_binding_catalog
        if (
            catalog.profile.preset != "paper-screen"
            or catalog.profile.preregistration_checksum != self.preregistration.content_checksum
        ):
            msg = "Confirmation must reuse the exact preregistration-bound paper-screen catalog."
            raise ValueError(msg)
        selected: list[ExecutableScopedBinding] = []
        for execution in self.configuration_execution_manifest.entries:
            matches = tuple(
                link
                for link in catalog.bindings
                if link.binding.publication_candidate_checksum == execution.configuration_checksum
            )
            if len(matches) != 1:
                msg = "A final configuration has no unique exact paper-screen executable binding."
                raise ValueError(msg)
            link = matches[0]
            alias = catalog.implementation_catalog.resolve(
                "paper-confirm",
                execution.method_id,
                "primary_q6",
            )
            if (
                link.binding.publication_method_id != execution.method_id
                or link.binding.target_scope_id != "primary_q6"
                or link.binding.strategy_schedule != execution.strategy_schedule
                or link.binding.implementation_checksum != execution.implementation_checksum
                or link.binding.content_checksum != execution.scoped_binding_checksum
                or link.content_checksum != execution.executable_binding_checksum
                or link.implementation_entry != alias
            ):
                msg = "A final configuration differs from its exact dormant repository confirmation alias."
                raise ValueError(msg)
            selected.append(link)
        object.__setattr__(self, "_selected_bindings", tuple(selected))

    @property
    def executable_bindings(self) -> tuple[ExecutableScopedBinding, ...]:
        """Exact promoted-plus-comparator repository bindings in manifest order."""
        return self._selected_bindings

    def executable_binding(self, configuration_checksum: str) -> ExecutableScopedBinding:
        """Resolve one exact final configuration without accepting a new route.

        Returns:
            The unique screened executable binding reused by confirmation.

        Raises:
            KeyError: If the configuration is outside the final manifest.
        """
        checksum = require_checksum(configuration_checksum, "configuration_checksum")
        for binding in self.executable_bindings:
            if binding.binding.publication_candidate_checksum == checksum:
                return binding
        raise KeyError(checksum)

    def scheduled_program_checksum(self, request: ConfirmExecutionRequest) -> str:
        """Recompile and return the exact program root for one owned request.

        This provides the external trust anchor needed when reopening real
        production evidence; an attempt cannot authenticate a substituted
        snapshot merely by resealing its own internal program checksum.

        Returns:
            The exact compiled :class:`ScheduledExecutionProgram` checksum.

        """
        link = self._owned_request_binding(request)
        execution = self.configuration_execution_manifest.entry(request.configuration_checksum)
        return ScheduledExecutionProgram.compile(
            link,
            execution.strategy_schedule,
            ScheduledJobSeedSet(request.optimization_seed),
        ).content_checksum

    def artifact_kind(self, request: ConfirmExecutionRequest) -> Literal["pipeline", "operator_growth"]:
        """Return the exact production artifact family for one owned request.

        This second external trust anchor prevents an internally resealed
        pipeline/operator-growth substitution from authenticating itself when
        a real confirmatory attempt is reopened.

        Returns:
            The only production artifact kind admitted by the sealed binding.

        Raises:
            ValueError: If the binding has no real production artifact family.
        """
        implementation_kind = self._owned_request_binding(request).binding.implementation_artifact.implementation_kind
        if implementation_kind == "operator_growth":
            return "operator_growth"
        if implementation_kind == "phase2_pipeline":
            return "pipeline"
        msg = "Final confirmation binding uses an unsupported production artifact kind."
        raise ValueError(msg)

    def _owned_request_binding(self, request: ConfirmExecutionRequest) -> ExecutableScopedBinding:
        """Validate request ownership and return its exact screened binding.

        Returns:
            The exact repository binding named by the context-owned request.

        Raises:
            TypeError: If ``request`` has the wrong protocol type.
            ValueError: If it is not the exact nested request object in this plan.
        """
        if not isinstance(request, ConfirmExecutionRequest):
            msg = "request must be a ConfirmExecutionRequest."
            raise TypeError(msg)
        if not any(job.confirm_execution_request is request for job in self.plan.jobs):
            msg = "Confirmation resolution accepts only an exact context-owned request object."
            raise ValueError(msg)
        validate_confirm_execution_request(
            request,
            self.final_seal,
            self.target_manifest,
            self.configuration_execution_manifest,
        )
        return self.executable_binding(request.configuration_checksum)

    def materialize_targets(self) -> tuple[MaterializedTarget, ...]:
        """Materialize the revealed population through the opaque sealed authority.

        Returns:
            Immutable target vectors for the already revealed manifest.
        """
        entropy = self.external_entropy_keyring.entropy_for("confirmatory", "primary_q6")
        return materialize_target_population(
            self.target_configuration,
            self.preregistration,
            self.target_manifest,
            entropy,
            self.target_materialization_authorization,
        ).targets

    @staticmethod
    def _validate_authorized_output_root(output_root: Path) -> None:
        """Require one absolute lexical root with no extant symlink component.

        Raises:
            TypeError: If the root is not a pathlib path.
            ValueError: If the root is noncanonical or contains an unsafe
                existing component.
        """
        if not isinstance(output_root, Path):
            msg = "authorized_output_root must be a pathlib.Path."
            raise TypeError(msg)
        canonical = Path(Path(output_root).resolve())
        if not output_root.is_absolute() or output_root != canonical:
            msg = "authorized_output_root must be an absolute canonical path."
            raise ValueError(msg)
        current = Path(output_root.anchor)
        for component in output_root.parts[1:]:
            current /= component
            if not current.exists() and not current.is_symlink():
                continue
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                msg = "authorized_output_root cannot contain a symlink component."
                raise ValueError(msg)
            if current != output_root and not stat.S_ISDIR(metadata.st_mode):
                msg = "authorized_output_root has a non-directory parent component."
                raise ValueError(msg)
        if output_root.exists() and not output_root.is_dir():
            msg = "authorized_output_root must be absent or an existing directory."
            raise ValueError(msg)

    @staticmethod
    def _validate_locked_study_head_custody_path(custody_path: Path, output_root: Path) -> None:
        """Require one external canonical single-link head-custody path.

        Raises:
            TypeError: If a path has the wrong type.
            ValueError: If the custody location is noncanonical, linked, inside
                the mutable scientific output, or lacks a safe existing parent.
        """
        if not isinstance(custody_path, Path) or not isinstance(output_root, Path):
            msg = "locked-study custody and output roots must be pathlib.Path values."
            raise TypeError(msg)
        canonical = Path(custody_path.resolve())
        if not custody_path.is_absolute() or custody_path != canonical:
            msg = "locked_study_head_custody_path must be an absolute canonical path."
            raise ValueError(msg)
        if custody_path.is_relative_to(output_root):
            msg = "Locked-study head custody must remain outside the mutable confirmation output."
            raise ValueError(msg)
        parent = custody_path.parent
        if parent.is_symlink() or not parent.is_dir():
            msg = "Locked-study head custody requires an existing non-symlink parent directory."
            raise ValueError(msg)
        current = Path(parent.anchor)
        for component in parent.parts[1:]:
            current /= component
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                msg = "Locked-study head custody cannot contain a linked or non-directory parent."
                raise ValueError(msg)
        if custody_path.is_symlink():
            msg = "Locked-study head custody cannot be a symlink."
            raise ValueError(msg)
        if custody_path.exists():
            metadata = custody_path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                msg = "Locked-study head custody must be a single-link regular file."
                raise ValueError(msg)

    def preflight(self, repository_root: Path, output_root: Path) -> None:
        """Recheck frozen source and output-role isolation before mutation.

        Raises:
            TypeError: If a path has the wrong type.
            ValueError: If source bytes, repository routes, or output role
                separation changed after the context was built.
        """
        if not isinstance(repository_root, Path) or not isinstance(output_root, Path):
            msg = "repository_root and output_root must be pathlib.Path values."
            raise TypeError(msg)
        if output_root != self.authorized_output_root:
            msg = "output_root differs from the confirmation context's authorized output root."
            raise ValueError(msg)
        self._validate_authorized_output_root(output_root)
        self._validate_locked_study_head_custody_path(
            self.locked_study_head_custody_path,
            output_root,
        )
        repository = repository_root.resolve()
        if self.locked_study_head_custody_path.is_relative_to(repository):
            msg = "Locked-study head custody must remain outside the governed repository."
            raise ValueError(msg)
        verify_final_seal_source_lock(
            self.final_seal,
            self.execution_source_manifest,
            self.analysis_source_manifest,
            repository_root,
        )
        validate_final_configuration_execution_manifest(
            self.final_seal,
            self.configuration_execution_manifest,
        )
        for link in self.executable_bindings:
            link.resolve_callable()
        roles = output_root / "roles"
        if roles.is_symlink() or (roles.exists() and not roles.is_dir()):
            msg = "Confirmation output roles must be absent or a non-symlink directory."
            raise ValueError(msg)
        if roles.exists() and any(path.name != "confirmatory" for path in roles.iterdir()):
            msg = "Development, screening, and confirmation outputs cannot share one output root."
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Return public roots only, never external confirmatory entropy."""
        return (
            "ConfirmationExecutionContext("
            f"plan_checksum={self.plan.content_checksum!r}, "
            f"final_seal_checksum={self.final_seal.content_checksum!r}, "
            f"target_manifest_checksum={self.target_manifest.content_checksum!r}, "
            f"prior_target_exposure_inventory_checksum={self.prior_target_exposure_inventory_checksum!r}, "
            f"authorized_output_root={str(self.authorized_output_root)!r}, "
            f"locked_study_head_custody_path={str(self.locked_study_head_custody_path)!r}, "
            "external_entropy_keyring=<redacted>)"
        )

    def __getstate__(self) -> NoReturn:
        """Reject state extraction used by serializers."""
        _serialization_forbidden()

    def __reduce__(self) -> NoReturn:
        """Reject pickle reduction."""
        _serialization_forbidden()

    def __reduce_ex__(self, _protocol: SupportsIndex) -> NoReturn:
        """Reject protocol-specific pickle reduction."""
        _serialization_forbidden()


def parse_entropy_file_specs(specifications: Sequence[str]) -> Mapping[tuple[str, str], Path]:
    """Parse CLI-only ``ROLE/SCOPE=PATH`` references without opening files.

    Returns:
        An immutable slot-to-path mapping.

    Raises:
        TypeError: If specifications is not a string sequence.
        ValueError: If spelling, slots, paths, or uniqueness are invalid.
    """
    if isinstance(specifications, (str, bytes)) or not isinstance(specifications, Sequence):
        msg = "External entropy specifications must be a sequence of strings."
        raise TypeError(msg)
    result: dict[tuple[str, str], Path] = {}
    for specification in specifications:
        if type(specification) is not str or specification.count("=") != 1:
            msg = "External entropy references must use ROLE/SCOPE=PATH."
            raise ValueError(msg)
        slot_text, path_text = specification.split("=", maxsplit=1)
        if slot_text.count("/") != 1 or not path_text:
            msg = "External entropy references must use ROLE/SCOPE=PATH."
            raise ValueError(msg)
        role, scope = slot_text.split("/", maxsplit=1)
        slot = _entropy_slot(role, scope)
        if slot in result:
            msg = "An external entropy slot may be specified only once."
            raise ValueError(msg)
        # This validates relative spellings while still allowing explicit
        # absolute private-file paths at the CLI boundary.
        if not Path(path_text).is_absolute():
            require_relative_path(path_text, "external entropy path")
        result[slot] = Path(path_text)
    return MappingProxyType(result)


__all__ = [
    "TRAINING_CANDIDATE_REF_SCHEMA_VERSION",
    "TRAINING_PREFLIGHT_REPORT_SCHEMA_VERSION",
    "AuthorizedTargetMaterialization",
    "ConfirmationExecutionContext",
    "ExternalEntropyKeyring",
    "TrainingCandidateRef",
    "TrainingExecutionContext",
    "TrainingPreflightReport",
    "bind_training_plan_fingerprints",
    "candidate_refs_from_bindings",
    "parse_entropy_file_specs",
    "schedules_from_bindings",
    "source_fingerprint_checksum",
    "validate_resumability_source_fingerprints",
]
