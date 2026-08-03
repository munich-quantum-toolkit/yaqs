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

import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn, SupportsIndex, cast

from .binding_catalog import ExecutableScopedBinding
from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .execution_bindings import TrainingExecutionProfile
from .protocol import InitialPreregistration, SampleSizeDesign, ScreeningCell, ScreeningManifest
from .resumability import ResumabilityFingerprint
from .scheduled_execution import ScheduledExecutionProgram, ScheduledJobSeedSet
from .source_lock import ExecutionSourceManifest, verify_execution_source_manifest
from .targets import (
    TargetMaterializationAuthorization,
    TargetPopulationConfig,
    TargetPopulationManifest,
    role_master_entropy_commitment,
)
from .training_orchestration import TrainingJob, TrainingRunPlan
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
            ValueError: If a path is missing, linked, non-regular, oversized, or
                does not contain exactly one supported key encoding.
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
            try:
                metadata = path.stat(follow_symlinks=False)
            except OSError:
                msg = "External entropy source is unavailable or unsafe."
                raise ValueError(msg) from None
            if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size not in {32, 64}:
                msg = "External entropy source is unavailable or unsafe."
                raise ValueError(msg)
            try:
                payload = path.read_bytes()
            except OSError:
                msg = "External entropy source is unavailable or unsafe."
                raise ValueError(msg) from None
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
