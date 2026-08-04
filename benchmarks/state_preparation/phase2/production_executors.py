# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Repository-owned WP22E training executors and immutable evidence custody.

The orchestration ABI predates typed result references and consequently asks an
executor for a checksum string.  This module keeps that compatibility boundary
deliberately narrow: :class:`ProductionTrainingExecutor` returns a typed
:class:`ResultArtifactRef`; the registry adapter immediately reopens the whole
attempt through that reference and returns only the verified reference
checksum.

No caller may supply a result checksum or summary.  All result identities are
derived from immutable attempt manifests which in turn enumerate every raw
numerical sidecar, map ensemble, schedule snapshot, resource record, and
failure record.  Reopening verifies paths, file bytes, logical checksums, and
the exact manifest member set before exposing a result.
"""

# The public codecs below delegate detailed scalar validation to validation.py.
# The executor boundary records ordinary numerical failures before reraising.
# ruff: noqa: DOC201, DOC501, PLW0717

from __future__ import annotations

import hashlib
import math
import os
import secrets
import stat
import time
import tracemalloc
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from benchmarks.state_preparation.constants import NOISELESS_NOISE_ID
from benchmarks.state_preparation.noise import create_scaled_standard_noise_provider
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KROTOV_MAP_ROLES,
    KrotovFixedMapEnsemble,
    KrotovTJMOptions,
    KrotovTruncation,
    derive_krotov_trajectory_seed,
    forward_tjm_trajectory,
    noisy_state_preparation_contribution,
    noisy_state_preparation_loss,
    noisy_state_preparation_metrics,
    noisy_state_preparation_metrics_with_maps,
    state_preparation_contribution,
    state_preparation_loss,
    state_preparation_metrics,
)

from .artifacts import StageExecutionEvidence
from .canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
    verify_sealed_mapping,
)
from .competitor_optimizers import ParameterShiftAdamConfig, SPSAConfig
from .execution_context import ConfirmationExecutionContext, TrainingExecutionContext
from .execution_protocol import FreshEvaluationPolicy, OperatorGrowthExecutionSpec
from .implementation_catalog import (
    OperatorGrowthSmokeExecution,
    OperatorGrowthSmokeRuntimeProgram,
    PipelineSmokeRuntimeProgram,
)
from .layerwise_bmpd import create_bmpd_circuit_binding
from .noisy_krotov import (
    NoisyKrotovCircuitBinding,
    NoisyKrotovObjectiveBinding,
    NoisyKrotovStageExecution,
    NoisyKrotovStageFailure,
    decode_noisy_krotov_circuit_binding_document,
    noisy_krotov_computational_zero_state_checksum,
)
from .operator_growth import (
    CandidateGradient,
    OperatorGrowthSpec,
    OperatorPoolSpec,
    materialize_operator_growth_circuit,
)
from .pipeline import TrainingPipelineConfig, TrainingPipelineTemplate, TrainingStageConfig
from .scheduled_execution import (
    OPERATOR_GROWTH_SEGMENTED_SNAPSHOT_SCHEMA_VERSION,
    AdamOptimizerPayload,
    KrotovOptimizerPayload,
    KrotovScheduledUpdateAdapter,
    OperatorGrowthSegmentedObjectiveEvidence,
    OperatorGrowthSegmentedObjectiveRequest,
    OperatorGrowthSegmentedObjectiveResult,
    OperatorGrowthSegmentedSnapshot,
    OperatorGrowthSelectionRequest,
    OperatorGrowthSelectionResult,
    OptimizerInitialization,
    ParameterShiftAdamScheduledUpdateAdapter,
    ScheduledExecutionProgram,
    ScheduledExecutionSnapshot,
    ScheduledJobSeedSet,
    ScheduledTrainingGradientRequest,
    ScheduledTrainingGradientResult,
    ScheduledTrainingObjectiveRequest,
    ScheduledTrainingObjectiveResult,
    ScheduledValidationRequest,
    ScheduledValidationResult,
    SPSAOptimizerPayload,
    SPSAScheduledUpdateAdapter,
    execute_operator_growth_segmented_program,
    execute_scheduled_program,
    initialize_scheduled_execution,
)
from .targets import (
    MaterializedTarget,
    TargetInstanceSpec,
    TargetPopulationConfig,
    TargetPopulationManifest,
    materialize_target_population,
)
from .training_orchestration import (
    ConfirmExecutionRequest,
    JobExecutionControls,
    TrainingExecutorRegistry,
    TrainingJob,
    confirmatory_evaluation_policy_checksum,
    validate_confirm_execution_request,
)
from .training_schedules import (
    PILOT_DIAGNOSTIC_SEED_POLICY_ID,
    ExecutionSeedPolicySuite,
    derive_role_seed,
)
from .validation import (
    require_checksum,
    require_float,
    require_int,
    require_mapping,
    require_relative_path,
    require_slug,
)
from .wp20_resources import CircuitResourceMetrics, measure_circuit_resources

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.yaqs.optimization import GateNoiseProvider, KrotovMapRole, KrotovNoiseMap

    from .binding_catalog import ExecutableScopedBinding
    from .protocol import ScreeningCell
    from .training_schedules import TrajectoryEnsembleMembership

    PipelineRunnerFactory = Callable[
        [TrainingPipelineConfig, MaterializedTarget],
        Callable[[TrainingStageConfig, np.ndarray | None], object],
    ]
    ProductionDataRole = Literal[
        "development",
        "checkpoint_validation",
        "screening_selection",
        "secondary_benchmark",
        "confirmatory",
    ]


ARTIFACT_BLOB_REF_SCHEMA_VERSION = "yaqs.state_preparation.phase2.production_blob_ref.v1"
SCHEDULED_MAP_EVIDENCE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_map_evidence.v1"
PILOT_DIAGNOSTIC_EVIDENCE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.pilot_diagnostic_evidence.v1"
PRODUCTION_NUMERICAL_EVIDENCE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.production_evidence.v1"
ATTEMPT_ARTIFACT_MANIFEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.production_attempt_manifest.v1"
RESULT_ARTIFACT_REF_SCHEMA_VERSION = "yaqs.state_preparation.phase2.production_result_ref.v1"
SYNTHETIC_CONFIRM_FIXTURE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.synthetic_confirm_fixture.v1"
PRODUCTION_DOCUMENT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.production_document.v1"

ATTEMPT_DIRECTORY_NAME = "production_attempts"
ATTEMPT_MANIFEST_NAME = "attempt_manifest.json"
OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT = 200

ArtifactStatus = Literal["success", "failure"]
ArtifactKind = Literal["pipeline", "operator_growth", "synthetic_confirmation"]

_BLOB_REF_KEYS = frozenset({
    "schema_version",
    "role",
    "media_type",
    "path",
    "byte_count",
    "file_checksum",
    "logical_checksum",
    "content_checksum",
})
_MAP_EVIDENCE_KEYS = frozenset({
    "schema_version",
    "request_checksum",
    "policy_checksum",
    "membership_checksum",
    "component_membership_checksums",
    "member_seeds",
    "component_member_seeds",
    "map_role",
    "resolved_seeds",
    "circuit_checksum",
    "provider_checksums",
    "ensemble_refs",
    "numerical_result_checksum",
    "trajectory_fidelities",
    "content_checksum",
})
_PILOT_DIAGNOSTIC_KEYS = frozenset({
    "schema_version",
    "job_checksum",
    "policy_checksum",
    "checkpoint_parameter_checksum",
    "parameter_vector_checksum",
    "circuit_checksum",
    "provider_checksum",
    "estimator_checksum",
    "member_seeds",
    "ensemble_refs",
    "pathwise_update_vectors",
    "content_checksum",
})
_EVIDENCE_KEYS = frozenset({
    "schema_version",
    "job_checksum",
    "attempt",
    "artifact_kind",
    "status",
    "execution_source_manifest_checksum",
    "source_fingerprint_checksum",
    "executable_binding_checksum",
    "scheduled_program_checksum",
    "target_identity",
    "evaluation_policy_checksum",
    "structural_prefix_checksums",
    "schedule_snapshot_ref",
    "map_evidence_refs",
    "diagnostic_refs",
    "raw_trajectory_ref",
    "resource_ref",
    "derived_metrics",
    "failure",
    "content_checksum",
})
_ATTEMPT_MANIFEST_KEYS = frozenset({
    "schema_version",
    "job_checksum",
    "attempt",
    "artifact_kind",
    "status",
    "execution_source_manifest_checksum",
    "source_fingerprint_checksum",
    "blobs",
    "evidence_ref",
    "content_checksum",
})
_RESULT_REF_KEYS = frozenset({
    "schema_version",
    "job_checksum",
    "attempt",
    "artifact_kind",
    "status",
    "execution_source_manifest_checksum",
    "source_fingerprint_checksum",
    "manifest_path",
    "manifest_file_checksum",
    "manifest_content_checksum",
    "evidence_checksum",
    "content_checksum",
})
_SYNTHETIC_FIXTURE_KEYS = frozenset({
    "schema_version",
    "request_checksum",
    "trajectory_fidelities",
    "content_checksum",
})
_PRODUCTION_DOCUMENT_KEYS = frozenset({"schema_version", "document_type", "payload", "content_checksum"})
_REAL_RAW_TRAJECTORY_KEYS = frozenset({
    "job_checksum",
    "evaluation_policy_checksum",
    "evaluation_configuration_checksum",
    "data_role",
    "seed_domain",
    "evaluation_seed",
    "trajectory_count",
    "trajectory_fidelities",
    "fixed_map_ensemble_checksum",
    "sampled_nonidentity_events",
})
_SYNTHETIC_RAW_TRAJECTORY_KEYS = frozenset({
    "request_checksum",
    "evaluation_policy_checksum",
    "data_role",
    "seed_domain",
    "evaluation_seed",
    "trajectory_count",
    "trajectory_fidelities",
    "synthetic_fixture_checksum",
})
_REAL_RESOURCE_KEYS = frozenset({
    "job_checksum",
    "source_fingerprint_checksum",
    "wall_time_seconds",
    "peak_memory_bytes",
    "normalized_work",
    "failure_phase",
    "partial_receipts",
    "circuit",
})
_SYNTHETIC_RESOURCE_KEYS = frozenset({
    "request_checksum",
    "source_fingerprint_checksum",
    "wall_time_seconds",
    "peak_memory_bytes",
    "normalized_work",
    "synthetic_fixture",
    "circuit",
})
_RESOURCE_CIRCUIT_KEYS = frozenset({
    "circuit_binding_checksum",
    "topology_id",
    "qubit_count",
    "parameter_count",
    "logical_gate_count",
    "logical_two_qubit_gate_count",
    "noisy_gate_indices",
    "compiled_resources",
    "compiled_resources_checksum",
    "native_two_qubit_gates_per_chain_edge",
})


def _sha256_bytes(payload: bytes) -> str:
    """Return the prefixed SHA-256 digest of exact file bytes."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sealed(payload: dict[str, object]) -> dict[str, object]:
    """Return a detached canonical-checksum-sealed mapping."""
    return {**payload, "content_checksum": canonical_checksum(payload)}


def _json_bytes(value: Mapping[str, object]) -> bytes:
    """Encode one canonical JSON document with a single terminal newline."""
    return f"{canonical_json(value)}\n".encode()


def _strict_tuple(value: object, name: str) -> tuple[object, ...]:
    """Decode a canonical JSON array after canonical.py has frozen it."""
    if type(value) is not tuple:
        msg = f"{name} must be a JSON array."
        raise TypeError(msg)
    return value


def _attempt_number(controls: JobExecutionControls) -> int:
    """Derive the append-only attempt number solely from orchestration state."""
    if not isinstance(controls, JobExecutionControls):
        msg = "controls must be JobExecutionControls."
        raise TypeError(msg)
    state = controls.schedule_resume_state
    if state is None:
        return 1
    if controls.resume and state.prior_status == "success":
        return state.prior_attempt
    return state.prior_attempt + 1


def _target_scope(qubit_count: int) -> Literal["primary_q6", "secondary_q12"]:
    """Map the only authorized WP22 widths to their binding scopes."""
    if qubit_count == 6:
        return "primary_q6"
    if qubit_count == 12:
        return "secondary_q12"
    msg = "Production WP22 executors support only q6 and q12."
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ArtifactBlobRef:
    """Immutable exact-byte reference to one member of an attempt manifest."""

    role: str
    media_type: str
    path: str
    byte_count: int
    file_checksum: str
    logical_checksum: str
    schema_version: str = field(default=ARTIFACT_BLOB_REF_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate bounded spelling, size, and both checksum identities."""
        object.__setattr__(self, "role", require_slug(self.role, "role"))
        if self.media_type not in {"application/json", "application/octet-stream"}:
            msg = "media_type must identify canonical JSON or an exact binary sidecar."
            raise ValueError(msg)
        path = require_relative_path(self.path, "path")
        if PurePosixPath(path).parts[:1] != (ATTEMPT_DIRECTORY_NAME,):
            msg = "Artifact blob paths must remain under production_attempts."
            raise ValueError(msg)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "byte_count", require_int(self.byte_count, "byte_count", minimum=1))
        object.__setattr__(self, "file_checksum", require_checksum(self.file_checksum, "file_checksum"))
        object.__setattr__(self, "logical_checksum", require_checksum(self.logical_checksum, "logical_checksum"))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "media_type": self.media_type,
            "path": self.path,
            "byte_count": self.byte_count,
            "file_checksum": self.file_checksum,
            "logical_checksum": self.logical_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering path, exact bytes, and logical content."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ArtifactBlobRef:
        """Decode and verify one exact-byte member reference."""
        mapping = verify_sealed_mapping(value, expected_keys=_BLOB_REF_KEYS, name="production blob reference")
        if mapping["schema_version"] != ARTIFACT_BLOB_REF_SCHEMA_VERSION:
            msg = "Production blob reference uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            role=cast("str", mapping["role"]),
            media_type=cast("str", mapping["media_type"]),
            path=cast("str", mapping["path"]),
            byte_count=cast("int", mapping["byte_count"]),
            file_checksum=cast("str", mapping["file_checksum"]),
            logical_checksum=cast("str", mapping["logical_checksum"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Production blob reference checksum changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class ScheduledMapEvidence:
    """Exact scheduled membership-to-realized-map provenance for one request."""

    request_checksum: str
    policy_checksum: str
    membership_checksum: str
    component_membership_checksums: tuple[str, ...]
    member_seeds: tuple[int, ...]
    component_member_seeds: tuple[tuple[int, ...], ...]
    map_role: KrotovMapRole
    resolved_seeds: tuple[int, ...]
    circuit_checksum: str
    provider_checksums: tuple[str, ...]
    ensemble_refs: tuple[ArtifactBlobRef, ...]
    numerical_result_checksum: str | None = None
    trajectory_fidelities: tuple[float, ...] = ()
    schema_version: str = field(default=SCHEDULED_MAP_EVIDENCE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require complete ordered component partition and immutable map refs."""
        for name in ("request_checksum", "policy_checksum", "membership_checksum", "circuit_checksum"):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        components = tuple(
            require_checksum(item, "component_membership_checksum") for item in self.component_membership_checksums
        )
        members = tuple(require_int(item, "member_seed") for item in self.member_seeds)
        partitions = tuple(
            tuple(require_int(seed, "component_member_seed") for seed in part) for part in self.component_member_seeds
        )
        providers = tuple(require_checksum(item, "provider_checksum") for item in self.provider_checksums)
        if self.map_role not in KROTOV_MAP_ROLES:
            msg = f"map_role must be one of {KROTOV_MAP_ROLES!r}."
            raise ValueError(msg)
        resolved = tuple(require_int(seed, "resolved_seed") for seed in self.resolved_seeds)
        refs = tuple(self.ensemble_refs)
        if not members or len(set(members)) != len(members):
            msg = "Scheduled map evidence requires a nonempty unique member sequence."
            raise ValueError(msg)
        if components:
            if len(partitions) != len(components) or tuple(seed for part in partitions for seed in part) != members:
                msg = "Component member partitions must reproduce aggregate membership in order."
                raise ValueError(msg)
        elif partitions != (members,):
            msg = "A non-mixture map request requires its aggregate membership as the sole partition."
            raise ValueError(msg)
        if len(refs) != len(partitions) or len(providers) != len(partitions) or len(resolved) != len(partitions):
            msg = "Every component partition requires one provider, resolved seed, and fixed-map ensemble."
            raise ValueError(msg)
        if any(not isinstance(ref, ArtifactBlobRef) or ref.role != "fixed_map_ensemble" for ref in refs):
            msg = "ensemble_refs must contain only fixed-map blob references."
            raise TypeError(msg)
        paths = tuple(ref.path for ref in refs)
        if len(paths) != len(set(paths)) or paths != tuple(sorted(paths)):
            msg = "Scheduled map ensembles must be unique and ordered by canonical path."
            raise ValueError(msg)
        numerical_result_checksum = self.numerical_result_checksum
        fidelities = tuple(
            require_float(value, "trajectory_fidelity", minimum=0.0, maximum=1.0)
            for value in self.trajectory_fidelities
        )
        if numerical_result_checksum is None:
            if fidelities:
                msg = "Raw scheduled fidelities require their exact numerical result checksum."
                raise ValueError(msg)
        else:
            numerical_result_checksum = require_checksum(
                numerical_result_checksum,
                "numerical_result_checksum",
            )
            if len(fidelities) != len(members):
                msg = "Result-bound scheduled map evidence requires one fidelity per declared member."
                raise ValueError(msg)
        object.__setattr__(self, "component_membership_checksums", components)
        object.__setattr__(self, "member_seeds", members)
        object.__setattr__(self, "component_member_seeds", partitions)
        object.__setattr__(self, "map_role", self.map_role)
        object.__setattr__(self, "resolved_seeds", resolved)
        object.__setattr__(self, "provider_checksums", providers)
        object.__setattr__(self, "ensemble_refs", refs)
        object.__setattr__(self, "numerical_result_checksum", numerical_result_checksum)
        object.__setattr__(self, "trajectory_fidelities", fidelities)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_checksum": self.request_checksum,
            "policy_checksum": self.policy_checksum,
            "membership_checksum": self.membership_checksum,
            "component_membership_checksums": list(self.component_membership_checksums),
            "member_seeds": list(self.member_seeds),
            "component_member_seeds": [list(part) for part in self.component_member_seeds],
            "map_role": self.map_role,
            "resolved_seeds": list(self.resolved_seeds),
            "circuit_checksum": self.circuit_checksum,
            "provider_checksums": list(self.provider_checksums),
            "ensemble_refs": [ref.to_dict() for ref in self.ensemble_refs],
            "numerical_result_checksum": self.numerical_result_checksum,
            "trajectory_fidelities": list(self.trajectory_fidelities),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum closing explicit members to exact persisted maps."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ScheduledMapEvidence:
        """Decode and verify one explicit scheduled map record."""
        mapping = verify_sealed_mapping(value, expected_keys=_MAP_EVIDENCE_KEYS, name="scheduled map evidence")
        if mapping["schema_version"] != SCHEDULED_MAP_EVIDENCE_SCHEMA_VERSION:
            msg = "Scheduled map evidence uses an unsupported schema version."
            raise ValueError(msg)
        raw_parts = _strict_tuple(mapping["component_member_seeds"], "component_member_seeds")
        result = cls(
            request_checksum=cast("str", mapping["request_checksum"]),
            policy_checksum=cast("str", mapping["policy_checksum"]),
            membership_checksum=cast("str", mapping["membership_checksum"]),
            component_membership_checksums=cast("tuple[str, ...]", mapping["component_membership_checksums"]),
            member_seeds=cast("tuple[int, ...]", mapping["member_seeds"]),
            component_member_seeds=tuple(
                cast("tuple[int, ...]", _strict_tuple(part, "component_member_seeds item")) for part in raw_parts
            ),
            map_role=cast("KrotovMapRole", mapping["map_role"]),
            resolved_seeds=cast("tuple[int, ...]", mapping["resolved_seeds"]),
            circuit_checksum=cast("str", mapping["circuit_checksum"]),
            provider_checksums=cast("tuple[str, ...]", mapping["provider_checksums"]),
            ensemble_refs=tuple(
                ArtifactBlobRef.from_dict(item) for item in _strict_tuple(mapping["ensemble_refs"], "ensemble_refs")
            ),
            numerical_result_checksum=cast("str | None", mapping["numerical_result_checksum"]),
            trajectory_fidelities=cast("tuple[float, ...]", mapping["trajectory_fidelities"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Scheduled map evidence checksum changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class PilotDiagnosticEvidence:
    """Complete q6 pilot pathwise-update vectors and exact fixed-map custody."""

    job_checksum: str
    policy_checksum: str
    checkpoint_parameter_checksum: str
    parameter_vector_checksum: str
    circuit_checksum: str
    provider_checksum: str
    estimator_checksum: str
    member_seeds: tuple[int, ...]
    ensemble_refs: tuple[ArtifactBlobRef, ...]
    pathwise_update_vectors: tuple[tuple[float, ...], ...]
    schema_version: str = field(default=PILOT_DIAGNOSTIC_EVIDENCE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require one complete vector and fixed-map ensemble per declared member."""
        for name in (
            "job_checksum",
            "policy_checksum",
            "checkpoint_parameter_checksum",
            "parameter_vector_checksum",
            "circuit_checksum",
            "provider_checksum",
            "estimator_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        if self.parameter_vector_checksum != self.checkpoint_parameter_checksum:
            msg = "Pilot diagnostic vector checksum differs from the selected checkpoint."
            raise ValueError(msg)
        seeds = tuple(require_int(seed, "member_seed") for seed in self.member_seeds)
        refs = tuple(self.ensemble_refs)
        vectors = tuple(
            tuple(require_float(value, "pathwise_update_coordinate") for value in vector)
            for vector in self.pathwise_update_vectors
        )
        if not seeds or len(seeds) != len(set(seeds)):
            msg = "Pilot diagnostic member seeds must be nonempty and unique."
            raise ValueError(msg)
        if len(refs) != len(seeds) or len(vectors) != len(seeds):
            msg = "Every pilot diagnostic member requires one map ensemble and update vector."
            raise ValueError(msg)
        if any(not isinstance(ref, ArtifactBlobRef) or ref.role != "fixed_map_ensemble" for ref in refs):
            msg = "Pilot diagnostic ensembles must be fixed-map blob references."
            raise TypeError(msg)
        paths = tuple(ref.path for ref in refs)
        if len(paths) != len(set(paths)) or paths != tuple(sorted(paths)):
            msg = "Pilot diagnostic ensembles must be unique and ordered by canonical path."
            raise ValueError(msg)
        widths = {len(vector) for vector in vectors}
        if widths != {len(vectors[0])} or not vectors[0]:
            msg = "Pilot diagnostic vectors must have one common positive parameter width."
            raise ValueError(msg)
        object.__setattr__(self, "member_seeds", seeds)
        object.__setattr__(self, "ensemble_refs", refs)
        object.__setattr__(self, "pathwise_update_vectors", vectors)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "job_checksum": self.job_checksum,
            "policy_checksum": self.policy_checksum,
            "checkpoint_parameter_checksum": self.checkpoint_parameter_checksum,
            "parameter_vector_checksum": self.parameter_vector_checksum,
            "circuit_checksum": self.circuit_checksum,
            "provider_checksum": self.provider_checksum,
            "estimator_checksum": self.estimator_checksum,
            "member_seeds": list(self.member_seeds),
            "ensemble_refs": [ref.to_dict() for ref in self.ensemble_refs],
            "pathwise_update_vectors": [list(vector) for vector in self.pathwise_update_vectors],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum closing every diagnostic vector to its exact sampled map."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed diagnostic evidence."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> PilotDiagnosticEvidence:
        """Decode and verify complete q6 pilot diagnostic evidence."""
        mapping = verify_sealed_mapping(
            value,
            expected_keys=_PILOT_DIAGNOSTIC_KEYS,
            name="pilot diagnostic evidence",
        )
        if mapping["schema_version"] != PILOT_DIAGNOSTIC_EVIDENCE_SCHEMA_VERSION:
            msg = "Pilot diagnostic evidence uses an unsupported schema version."
            raise ValueError(msg)
        vectors = tuple(
            cast("tuple[float, ...]", _strict_tuple(item, "pathwise_update_vector"))
            for item in _strict_tuple(mapping["pathwise_update_vectors"], "pathwise_update_vectors")
        )
        evidence = cls(
            job_checksum=cast("str", mapping["job_checksum"]),
            policy_checksum=cast("str", mapping["policy_checksum"]),
            checkpoint_parameter_checksum=cast("str", mapping["checkpoint_parameter_checksum"]),
            parameter_vector_checksum=cast("str", mapping["parameter_vector_checksum"]),
            circuit_checksum=cast("str", mapping["circuit_checksum"]),
            provider_checksum=cast("str", mapping["provider_checksum"]),
            estimator_checksum=cast("str", mapping["estimator_checksum"]),
            member_seeds=cast("tuple[int, ...]", mapping["member_seeds"]),
            ensemble_refs=tuple(
                ArtifactBlobRef.from_dict(item) for item in _strict_tuple(mapping["ensemble_refs"], "ensemble_refs")
            ),
            pathwise_update_vectors=vectors,
        )
        if mapping["content_checksum"] != evidence.content_checksum:
            msg = "Pilot diagnostic evidence checksum changed during normalization."
            raise ValueError(msg)
        return evidence


@dataclass(frozen=True, slots=True)
class ProductionNumericalEvidence:
    """Typed source-addressed summary whose raw members remain dereferenceable."""

    job_checksum: str
    attempt: int
    artifact_kind: ArtifactKind
    status: ArtifactStatus
    execution_source_manifest_checksum: str
    source_fingerprint_checksum: str
    executable_binding_checksum: str
    scheduled_program_checksum: str
    target_identity: Mapping[str, object]
    evaluation_policy_checksum: str
    structural_prefix_checksums: tuple[str, ...]
    schedule_snapshot_ref: ArtifactBlobRef | None
    map_evidence_refs: tuple[ArtifactBlobRef, ...]
    diagnostic_refs: tuple[ArtifactBlobRef, ...]
    raw_trajectory_ref: ArtifactBlobRef | None
    resource_ref: ArtifactBlobRef
    derived_metrics: Mapping[str, object]
    failure: Mapping[str, object] | None
    schema_version: str = field(default=PRODUCTION_NUMERICAL_EVIDENCE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate source, result membership, status, and failure separation."""
        for name in (
            "job_checksum",
            "execution_source_manifest_checksum",
            "source_fingerprint_checksum",
            "executable_binding_checksum",
            "scheduled_program_checksum",
            "evaluation_policy_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(self, "attempt", require_int(self.attempt, "attempt", minimum=1))
        if self.artifact_kind not in {"pipeline", "operator_growth", "synthetic_confirmation"}:
            msg = "artifact_kind is unsupported."
            raise ValueError(msg)
        if self.status not in {"success", "failure"}:
            msg = "status must be success or failure."
            raise ValueError(msg)
        identity = dict(require_mapping(self.target_identity, "target_identity"))
        metrics = dict(require_mapping(self.derived_metrics, "derived_metrics"))
        prefix = tuple(
            require_checksum(item, "structural_prefix_checksum") for item in self.structural_prefix_checksums
        )
        map_refs = tuple(self.map_evidence_refs)
        if any(not isinstance(ref, ArtifactBlobRef) or ref.role != "scheduled_map_evidence" for ref in map_refs):
            msg = "map_evidence_refs must contain only scheduled-map evidence blobs."
            raise TypeError(msg)
        diagnostic_refs = tuple(self.diagnostic_refs)
        if any(
            not isinstance(ref, ArtifactBlobRef) or ref.role != "pilot_diagnostic_sidecar" for ref in diagnostic_refs
        ):
            msg = "diagnostic_refs must contain only pilot-diagnostic sidecar references."
            raise TypeError(msg)
        if not isinstance(self.resource_ref, ArtifactBlobRef) or self.resource_ref.role != "runtime_resources":
            msg = "resource_ref must be a runtime-resources blob."
            raise TypeError(msg)
        if self.schedule_snapshot_ref is not None and (
            not isinstance(self.schedule_snapshot_ref, ArtifactBlobRef)
            or self.schedule_snapshot_ref.role != "schedule_snapshot"
        ):
            msg = "schedule_snapshot_ref must be a schedule-snapshot blob or None."
            raise TypeError(msg)
        if self.raw_trajectory_ref is not None and (
            not isinstance(self.raw_trajectory_ref, ArtifactBlobRef)
            or self.raw_trajectory_ref.role != "raw_trajectory_sidecar"
        ):
            msg = "raw_trajectory_ref must be a raw-trajectory sidecar blob or None."
            raise TypeError(msg)
        failure = None if self.failure is None else dict(require_mapping(self.failure, "failure"))
        if (self.status == "failure") != (failure is not None):
            msg = "Failure details are present exactly for a failed attempt."
            raise ValueError(msg)
        if self.status == "success":
            preset = metrics.get("execution_preset")
            if self.artifact_kind != "synthetic_confirmation" and preset not in {
                "training-smoke",
                "paper-pilot",
                "paper-screen",
                "paper-confirm",
            }:
                msg = "Successful production evidence requires its exact execution preset."
                raise ValueError(msg)
            if (
                self.artifact_kind != "operator_growth" or preset != "training-smoke"
            ) and self.raw_trajectory_ref is None:
                msg = "Successful numerical evidence requires a raw trajectory sidecar."
                raise ValueError(msg)
            if self.artifact_kind == "pipeline" and preset != "training-smoke":
                if self.schedule_snapshot_ref is None:
                    msg = "Successful production pipeline evidence requires a schedule snapshot."
                    raise ValueError(msg)
                if metrics.get("scheduled_noisy_training") is True and not map_refs:
                    msg = "Successful noisy scheduled training requires map evidence."
                    raise ValueError(msg)
            if self.artifact_kind == "operator_growth" and not prefix:
                msg = "Successful operator-growth evidence requires its structural result checksum."
                raise ValueError(msg)
            if metrics.get("pilot_diagnostic_required") is True and not diagnostic_refs:
                msg = "Successful primary pilot evidence requires its pathwise diagnostic."
                raise ValueError(msg)
        object.__setattr__(self, "target_identity", identity)
        object.__setattr__(self, "derived_metrics", metrics)
        object.__setattr__(self, "structural_prefix_checksums", prefix)
        object.__setattr__(self, "map_evidence_refs", map_refs)
        object.__setattr__(self, "diagnostic_refs", diagnostic_refs)
        object.__setattr__(self, "failure", failure)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "job_checksum": self.job_checksum,
            "attempt": self.attempt,
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "execution_source_manifest_checksum": self.execution_source_manifest_checksum,
            "source_fingerprint_checksum": self.source_fingerprint_checksum,
            "executable_binding_checksum": self.executable_binding_checksum,
            "scheduled_program_checksum": self.scheduled_program_checksum,
            "target_identity": dict(self.target_identity),
            "evaluation_policy_checksum": self.evaluation_policy_checksum,
            "structural_prefix_checksums": list(self.structural_prefix_checksums),
            "schedule_snapshot_ref": (
                None if self.schedule_snapshot_ref is None else self.schedule_snapshot_ref.to_dict()
            ),
            "map_evidence_refs": [ref.to_dict() for ref in self.map_evidence_refs],
            "diagnostic_refs": [ref.to_dict() for ref in self.diagnostic_refs],
            "raw_trajectory_ref": None if self.raw_trajectory_ref is None else self.raw_trajectory_ref.to_dict(),
            "resource_ref": self.resource_ref.to_dict(),
            "derived_metrics": dict(self.derived_metrics),
            "failure": None if self.failure is None else dict(self.failure),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum closing the summary to all raw artifact references."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ProductionNumericalEvidence:
        """Decode and verify typed production evidence."""
        mapping = verify_sealed_mapping(value, expected_keys=_EVIDENCE_KEYS, name="production numerical evidence")
        if mapping["schema_version"] != PRODUCTION_NUMERICAL_EVIDENCE_SCHEMA_VERSION:
            msg = "Production numerical evidence uses an unsupported schema version."
            raise ValueError(msg)
        snapshot = mapping["schedule_snapshot_ref"]
        raw = mapping["raw_trajectory_ref"]
        result = cls(
            job_checksum=cast("str", mapping["job_checksum"]),
            attempt=cast("int", mapping["attempt"]),
            artifact_kind=cast("ArtifactKind", mapping["artifact_kind"]),
            status=cast("ArtifactStatus", mapping["status"]),
            execution_source_manifest_checksum=cast("str", mapping["execution_source_manifest_checksum"]),
            source_fingerprint_checksum=cast("str", mapping["source_fingerprint_checksum"]),
            executable_binding_checksum=cast("str", mapping["executable_binding_checksum"]),
            scheduled_program_checksum=cast("str", mapping["scheduled_program_checksum"]),
            target_identity=cast("Mapping[str, object]", mapping["target_identity"]),
            evaluation_policy_checksum=cast("str", mapping["evaluation_policy_checksum"]),
            structural_prefix_checksums=cast("tuple[str, ...]", mapping["structural_prefix_checksums"]),
            schedule_snapshot_ref=None if snapshot is None else ArtifactBlobRef.from_dict(snapshot),
            map_evidence_refs=tuple(
                ArtifactBlobRef.from_dict(item)
                for item in _strict_tuple(mapping["map_evidence_refs"], "map_evidence_refs")
            ),
            diagnostic_refs=tuple(
                ArtifactBlobRef.from_dict(item) for item in _strict_tuple(mapping["diagnostic_refs"], "diagnostic_refs")
            ),
            raw_trajectory_ref=None if raw is None else ArtifactBlobRef.from_dict(raw),
            resource_ref=ArtifactBlobRef.from_dict(mapping["resource_ref"]),
            derived_metrics=cast("Mapping[str, object]", mapping["derived_metrics"]),
            failure=cast("Mapping[str, object] | None", mapping["failure"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Production numerical evidence checksum changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class AttemptArtifactManifest:
    """Terminal append-only manifest enumerating an exact attempt universe."""

    job_checksum: str
    attempt: int
    artifact_kind: ArtifactKind
    status: ArtifactStatus
    execution_source_manifest_checksum: str
    source_fingerprint_checksum: str
    blobs: tuple[ArtifactBlobRef, ...]
    evidence_ref: ArtifactBlobRef
    schema_version: str = field(default=ATTEMPT_ARTIFACT_MANIFEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require a unique, sorted, closed blob universe and exact evidence member."""
        object.__setattr__(self, "job_checksum", require_checksum(self.job_checksum, "job_checksum"))
        object.__setattr__(self, "attempt", require_int(self.attempt, "attempt", minimum=1))
        object.__setattr__(
            self,
            "execution_source_manifest_checksum",
            require_checksum(self.execution_source_manifest_checksum, "execution_source_manifest_checksum"),
        )
        object.__setattr__(
            self,
            "source_fingerprint_checksum",
            require_checksum(self.source_fingerprint_checksum, "source_fingerprint_checksum"),
        )
        if self.artifact_kind not in {"pipeline", "operator_growth", "synthetic_confirmation"}:
            msg = "artifact_kind is unsupported."
            raise ValueError(msg)
        if self.status not in {"success", "failure"}:
            msg = "status must be success or failure."
            raise ValueError(msg)
        blobs = tuple(self.blobs)
        if not blobs or any(not isinstance(ref, ArtifactBlobRef) for ref in blobs):
            msg = "blobs must contain typed artifact references."
            raise TypeError(msg)
        paths = tuple(ref.path for ref in blobs)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            msg = "Manifest blob paths must be unique and canonically sorted."
            raise ValueError(msg)
        if not isinstance(self.evidence_ref, ArtifactBlobRef) or self.evidence_ref.role != "production_evidence":
            msg = "evidence_ref must identify production numerical evidence."
            raise TypeError(msg)
        if blobs.count(self.evidence_ref) != 1:
            msg = "The exact evidence reference must occur once in the manifest universe."
            raise ValueError(msg)
        object.__setattr__(self, "blobs", blobs)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "job_checksum": self.job_checksum,
            "attempt": self.attempt,
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "execution_source_manifest_checksum": self.execution_source_manifest_checksum,
            "source_fingerprint_checksum": self.source_fingerprint_checksum,
            "blobs": [ref.to_dict() for ref in self.blobs],
            "evidence_ref": self.evidence_ref.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete attempt member set."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> AttemptArtifactManifest:
        """Decode and verify one terminal attempt manifest."""
        mapping = verify_sealed_mapping(value, expected_keys=_ATTEMPT_MANIFEST_KEYS, name="attempt artifact manifest")
        if mapping["schema_version"] != ATTEMPT_ARTIFACT_MANIFEST_SCHEMA_VERSION:
            msg = "Attempt artifact manifest uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            job_checksum=cast("str", mapping["job_checksum"]),
            attempt=cast("int", mapping["attempt"]),
            artifact_kind=cast("ArtifactKind", mapping["artifact_kind"]),
            status=cast("ArtifactStatus", mapping["status"]),
            execution_source_manifest_checksum=cast("str", mapping["execution_source_manifest_checksum"]),
            source_fingerprint_checksum=cast("str", mapping["source_fingerprint_checksum"]),
            blobs=tuple(ArtifactBlobRef.from_dict(item) for item in _strict_tuple(mapping["blobs"], "blobs")),
            evidence_ref=ArtifactBlobRef.from_dict(mapping["evidence_ref"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Attempt artifact manifest checksum changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class ResultArtifactRef:
    """Source-addressed typed handle to one verified immutable attempt."""

    job_checksum: str
    attempt: int
    artifact_kind: ArtifactKind
    status: ArtifactStatus
    execution_source_manifest_checksum: str
    source_fingerprint_checksum: str
    manifest_path: str
    manifest_file_checksum: str
    manifest_content_checksum: str
    evidence_checksum: str
    schema_version: str = field(default=RESULT_ARTIFACT_REF_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact source, manifest, and evidence addressing."""
        for name in (
            "job_checksum",
            "execution_source_manifest_checksum",
            "source_fingerprint_checksum",
            "manifest_file_checksum",
            "manifest_content_checksum",
            "evidence_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(self, "attempt", require_int(self.attempt, "attempt", minimum=1))
        if self.artifact_kind not in {"pipeline", "operator_growth", "synthetic_confirmation"}:
            msg = "artifact_kind is unsupported."
            raise ValueError(msg)
        if self.status not in {"success", "failure"}:
            msg = "status must be success or failure."
            raise ValueError(msg)
        path = require_relative_path(self.manifest_path, "manifest_path")
        expected = f"{ATTEMPT_DIRECTORY_NAME}/attempt_{self.attempt:06d}/{ATTEMPT_MANIFEST_NAME}"
        if path != expected:
            msg = "manifest_path does not match the canonical immutable attempt address."
            raise ValueError(msg)
        object.__setattr__(self, "manifest_path", path)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "job_checksum": self.job_checksum,
            "attempt": self.attempt,
            "artifact_kind": self.artifact_kind,
            "status": self.status,
            "execution_source_manifest_checksum": self.execution_source_manifest_checksum,
            "source_fingerprint_checksum": self.source_fingerprint_checksum,
            "manifest_path": self.manifest_path,
            "manifest_file_checksum": self.manifest_file_checksum,
            "manifest_content_checksum": self.manifest_content_checksum,
            "evidence_checksum": self.evidence_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum used by the legacy orchestration outcome ABI."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ResultArtifactRef:
        """Decode and verify one source-addressed result handle."""
        mapping = verify_sealed_mapping(value, expected_keys=_RESULT_REF_KEYS, name="result artifact reference")
        if mapping["schema_version"] != RESULT_ARTIFACT_REF_SCHEMA_VERSION:
            msg = "Result artifact reference uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            job_checksum=cast("str", mapping["job_checksum"]),
            attempt=cast("int", mapping["attempt"]),
            artifact_kind=cast("ArtifactKind", mapping["artifact_kind"]),
            status=cast("ArtifactStatus", mapping["status"]),
            execution_source_manifest_checksum=cast("str", mapping["execution_source_manifest_checksum"]),
            source_fingerprint_checksum=cast("str", mapping["source_fingerprint_checksum"]),
            manifest_path=cast("str", mapping["manifest_path"]),
            manifest_file_checksum=cast("str", mapping["manifest_file_checksum"]),
            manifest_content_checksum=cast("str", mapping["manifest_content_checksum"]),
            evidence_checksum=cast("str", mapping["evidence_checksum"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Result artifact reference checksum changed during normalization."
            raise ValueError(msg)
        return result


class ProductionAttemptStore:
    """Append-only attempt writer and complete dereferencing verifier."""

    def __init__(self, job_directory: Path, job_checksum: str, attempt: int) -> None:
        """Bind one store to a single exact job and append-only attempt."""
        if not isinstance(job_directory, Path):
            msg = "job_directory must be a pathlib.Path."
            raise TypeError(msg)
        self.job_directory = job_directory
        self.job_checksum = require_checksum(job_checksum, "job_checksum")
        self.attempt = require_int(attempt, "attempt", minimum=1)
        self.relative_attempt_directory = f"{ATTEMPT_DIRECTORY_NAME}/attempt_{self.attempt:06d}"
        self._written_refs: list[ArtifactBlobRef] = []

    @property
    def manifest_relative_path(self) -> str:
        """Canonical relative path of this attempt's terminal manifest."""
        return f"{self.relative_attempt_directory}/{ATTEMPT_MANIFEST_NAME}"

    @property
    def written_refs(self) -> tuple[ArtifactBlobRef, ...]:
        """Complete members written by this exact in-process attempt."""
        return tuple(self._written_refs)

    def terminal_manifest_exists(self) -> bool:
        """Return whether this exact attempt already has a terminal manifest."""
        path = self._safe_path(self.manifest_relative_path)
        return path.exists() or path.is_symlink()

    def attempt_directory_exists(self) -> bool:
        """Return whether any state already exists for this exact attempt."""
        path = self._safe_path(self.relative_attempt_directory)
        return path.exists() or path.is_symlink()

    def member_paths(self) -> tuple[str, ...]:
        """Enumerate exact regular members through one pinned attempt directory."""
        if self._descriptor_creation_supported():
            try:
                root_descriptor, _manifest_name = self._open_parent_descriptor(
                    self.manifest_relative_path,
                    create=False,
                )
            except FileNotFoundError:
                return ()
            except OSError as error:
                msg = "Production attempt directory is unavailable or unsafe."
                raise ValueError(msg) from error
            root_metadata = os.fstat(root_descriptor)
            members: list[str] = []

            def visit(directory_descriptor: int, relative_directory: PurePosixPath) -> None:
                for name in sorted(os.listdir(directory_descriptor)):
                    try:
                        metadata = os.stat(
                            name,
                            dir_fd=directory_descriptor,
                            follow_symlinks=False,
                        )
                    except OSError as error:
                        msg = "Production attempt member changed during enumeration."
                        raise ValueError(msg) from error
                    relative = relative_directory / name
                    if stat.S_ISLNK(metadata.st_mode):
                        msg = "Production attempt contains a symlink member."
                        raise ValueError(msg)
                    if stat.S_ISDIR(metadata.st_mode):
                        try:
                            child_descriptor = os.open(
                                name,
                                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory_descriptor,
                            )
                        except OSError as error:
                            msg = "Production attempt directory changed during enumeration."
                            raise ValueError(msg) from error
                        try:
                            opened = os.fstat(child_descriptor)
                            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                                msg = "Production attempt directory identity changed during enumeration."
                                raise ValueError(msg)
                            visit(child_descriptor, relative)
                            current = os.stat(
                                name,
                                dir_fd=directory_descriptor,
                                follow_symlinks=False,
                            )
                            if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                                msg = "Production attempt directory changed during enumeration."
                                raise ValueError(msg)
                        finally:
                            os.close(child_descriptor)
                        continue
                    if not stat.S_ISREG(metadata.st_mode):
                        msg = "Production attempt contains a non-regular member."
                        raise ValueError(msg)
                    normalized = str(relative)
                    if normalized != ATTEMPT_MANIFEST_NAME:
                        members.append(f"{self.relative_attempt_directory}/{normalized}")

            try:
                visit(root_descriptor, PurePosixPath())
                verification_descriptor, verification_name = self._open_parent_descriptor(
                    self.manifest_relative_path,
                    create=False,
                )
                try:
                    verification = os.fstat(verification_descriptor)
                    if verification_name != ATTEMPT_MANIFEST_NAME or (
                        verification.st_dev,
                        verification.st_ino,
                    ) != (root_metadata.st_dev, root_metadata.st_ino):
                        msg = "Production attempt root changed during member enumeration."
                        raise ValueError(msg)
                finally:
                    os.close(verification_descriptor)
            finally:
                os.close(root_descriptor)
            return tuple(sorted(members))

        root = self._safe_path(self.relative_attempt_directory)
        if not root.exists():
            return ()
        root_metadata = root.lstat()
        if not stat.S_ISDIR(root_metadata.st_mode):
            msg = "Production attempt root is not a directory."
            raise ValueError(msg)
        members: list[str] = []
        directory_identities: dict[Path, tuple[int, int]] = {}

        def visit_fallback(directory: Path) -> None:
            metadata = directory.lstat()
            directory_identities[directory] = (metadata.st_dev, metadata.st_ino)
            with os.scandir(directory) as entries:
                for entry in sorted(entries, key=lambda item: item.name):
                    path = Path(entry.path)
                    entry_metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(entry_metadata.st_mode):
                        msg = "Production attempt contains a symlink member."
                        raise ValueError(msg)
                    if stat.S_ISDIR(entry_metadata.st_mode):
                        visit_fallback(path)
                        current = path.lstat()
                        if (current.st_dev, current.st_ino) != directory_identities[path]:
                            msg = "Production attempt directory changed during enumeration."
                            raise ValueError(msg)
                        continue
                    if not stat.S_ISREG(entry_metadata.st_mode):
                        msg = "Production attempt contains a non-regular member."
                        raise ValueError(msg)
                    relative = str(path.relative_to(self.job_directory)).replace(os.sep, "/")
                    if relative != self.manifest_relative_path:
                        members.append(relative)

        visit_fallback(root)
        current_root = self._safe_path(self.relative_attempt_directory).lstat()
        if (current_root.st_dev, current_root.st_ino) != (root_metadata.st_dev, root_metadata.st_ino):
            msg = "Production attempt root changed during member enumeration."
            raise ValueError(msg)
        for directory, identity in directory_identities.items():
            current = directory.lstat()
            if (current.st_dev, current.st_ino) != identity:
                msg = "Production attempt directory changed during enumeration."
                raise ValueError(msg)
        return tuple(sorted(members))

    @staticmethod
    def _descriptor_creation_supported() -> bool:
        """Return whether secure descriptor-relative creation is available."""
        return (
            hasattr(os, "O_DIRECTORY")
            and hasattr(os, "O_NOFOLLOW")
            and hasattr(os, "supports_dir_fd")
            and all(function in os.supports_dir_fd for function in (os.open, os.mkdir, os.link, os.stat, os.unlink))
        )

    @staticmethod
    def _open_child_directory(parent_descriptor: int, component: str, *, create: bool) -> int:
        """Open one non-link child directory relative to an already pinned parent."""
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            return os.open(component, flags, dir_fd=parent_descriptor)
        except FileNotFoundError:
            if not create:
                raise
            with suppress(FileExistsError):
                os.mkdir(component, 0o700, dir_fd=parent_descriptor)
            return os.open(component, flags, dir_fd=parent_descriptor)

    def _open_job_directory_descriptor(self, *, create: bool) -> int:
        """Pin the job directory by walking every absolute component without links."""
        absolute = self.job_directory.absolute()
        parts = absolute.parts
        if not parts or parts[0] != os.sep:
            msg = "Descriptor-relative production custody requires an absolute POSIX path."
            raise ValueError(msg)
        descriptor = os.open(os.sep, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for component in parts[1:]:
                next_descriptor = self._open_child_directory(descriptor, component, create=create)
                os.close(descriptor)
                descriptor = next_descriptor
        except Exception:
            os.close(descriptor)
            raise
        return descriptor

    def _open_parent_descriptor(self, relative_path: str, *, create: bool) -> tuple[int, str]:
        """Pin the exact parent of one bounded production member."""
        normalized = require_relative_path(relative_path, "relative_path")
        parts = PurePosixPath(normalized).parts
        if parts[:1] != (ATTEMPT_DIRECTORY_NAME,) or len(parts) < 2:
            msg = "Production artifacts must stay under production_attempts."
            raise ValueError(msg)
        descriptor = self._open_job_directory_descriptor(create=create)
        try:
            for component in parts[:-1]:
                next_descriptor = self._open_child_directory(descriptor, component, create=create)
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor, parts[-1]
        except Exception:
            os.close(descriptor)
            raise

    def _fallback_directory_identities(self, relative_directory: str) -> tuple[tuple[int, int], ...]:
        """Capture every fallback parent identity for post-open swap detection."""
        normalized = require_relative_path(relative_directory, "relative_directory")
        parts = PurePosixPath(normalized).parts
        if parts[:1] != (ATTEMPT_DIRECTORY_NAME,):
            msg = "Production artifacts must stay under production_attempts."
            raise ValueError(msg)
        identities: list[tuple[int, int]] = []
        path = self.job_directory
        for component in parts:
            path /= component
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                msg = "Production artifact parent is unavailable or unsafe."
                raise ValueError(msg)
            identities.append((metadata.st_dev, metadata.st_ino))
        return tuple(identities)

    def _safe_path(self, relative_path: str, *, require_file: bool = False) -> Path:
        """Resolve a bounded path while rejecting every symlink component."""
        normalized = require_relative_path(relative_path, "relative_path")
        path = self.job_directory
        parts = PurePosixPath(normalized).parts
        if parts[:1] != (ATTEMPT_DIRECTORY_NAME,):
            msg = "Production artifacts must stay under production_attempts."
            raise ValueError(msg)
        if self.job_directory.is_symlink() or (self.job_directory.exists() and not self.job_directory.is_dir()):
            msg = "Job artifact directory is unavailable or unsafe."
            raise ValueError(msg)
        for index, component in enumerate(parts):
            path /= component
            if not path.exists() and not path.is_symlink():
                continue
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                msg = "Production artifact path contains a symlink."
                raise ValueError(msg)
            final = index == len(parts) - 1
            if final and require_file and not stat.S_ISREG(metadata.st_mode):
                msg = "Production artifact member is not a regular file."
                raise ValueError(msg)
            if not final and not stat.S_ISDIR(metadata.st_mode):
                msg = "Production artifact parent is not a directory."
                raise ValueError(msg)
        return path

    def _mkdirs(self, relative_directory: str) -> None:
        """Create missing bounded directories one component at a time."""
        path = self.job_directory
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            msg = "Job artifact directory is unavailable or unsafe."
            raise ValueError(msg)
        path.mkdir(parents=True, exist_ok=True)
        for component in PurePosixPath(require_relative_path(relative_directory, "relative_directory")).parts:
            path /= component
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                msg = "Production artifact parent is unavailable or unsafe."
                raise ValueError(msg)
            path.mkdir(exist_ok=True)

    def write_blob(
        self,
        relative_name: str,
        payload: bytes,
        *,
        role: str,
        logical_checksum: str,
        media_type: str = "application/json",
    ) -> ArtifactBlobRef:
        """Write one exact immutable member with exclusive-create semantics."""
        if type(payload) is not bytes or not payload:
            msg = "payload must be nonempty immutable bytes."
            raise TypeError(msg)
        name = require_relative_path(relative_name, "relative_name")
        relative = f"{self.relative_attempt_directory}/{name}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        pinned_parent_descriptor: int | None = None
        if self._descriptor_creation_supported():
            parent_descriptor, filename = self._open_parent_descriptor(relative, create=True)
            pinned_parent_descriptor = parent_descriptor
            try:
                descriptor = os.open(filename, flags, 0o600, dir_fd=parent_descriptor)
            except FileExistsError:
                os.close(parent_descriptor)
                msg = f"Immutable production artifact already exists: {relative}."
                raise ValueError(msg) from None
            except Exception:
                os.close(parent_descriptor)
                raise
        else:
            path = self._safe_path(relative)
            self._mkdirs(str(PurePosixPath(relative).parent))
            parent_identities = self._fallback_directory_identities(str(PurePosixPath(relative).parent))
            try:
                descriptor = os.open(path, flags, 0o600)
            except FileExistsError:
                msg = f"Immutable production artifact already exists: {relative}."
                raise ValueError(msg) from None
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or self._fallback_directory_identities(str(PurePosixPath(relative).parent)) != parent_identities
            ):
                os.close(descriptor)
                msg = "Production artifact parent changed during exclusive creation."
                raise ValueError(msg)
            disk = self._safe_path(relative, require_file=True).lstat()
            if (disk.st_dev, disk.st_ino) != (opened.st_dev, opened.st_ino):
                os.close(descriptor)
                msg = "Production artifact identity changed during exclusive creation."
                raise ValueError(msg)
        # A failed write deliberately leaves its partial exclusive file.  The
        # missing terminal manifest keeps the attempt non-authoritative and the
        # retained member prevents silent publication under the same attempt.
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if pinned_parent_descriptor is not None:
                os.fsync(pinned_parent_descriptor)
                verification_descriptor, verification_name = self._open_parent_descriptor(
                    relative,
                    create=False,
                )
                try:
                    if verification_name != filename or (
                        os.fstat(verification_descriptor).st_dev,
                        os.fstat(verification_descriptor).st_ino,
                    ) != (
                        os.fstat(pinned_parent_descriptor).st_dev,
                        os.fstat(pinned_parent_descriptor).st_ino,
                    ):
                        msg = "Production artifact parent changed during descriptor-relative creation."
                        raise ValueError(msg)
                finally:
                    os.close(verification_descriptor)
        finally:
            if pinned_parent_descriptor is not None:
                os.close(pinned_parent_descriptor)
        ref = ArtifactBlobRef(
            role=role,
            media_type=media_type,
            path=relative,
            byte_count=len(payload),
            file_checksum=_sha256_bytes(payload),
            logical_checksum=logical_checksum,
        )
        self._written_refs.append(ref)
        return ref

    def write_json_blob(self, relative_name: str, value: Mapping[str, object], *, role: str) -> ArtifactBlobRef:
        """Write one checksum-bearing canonical JSON member."""
        supplied = value.get("content_checksum")
        logical = require_checksum(supplied, "value.content_checksum")
        return self.write_blob(relative_name, _json_bytes(value), role=role, logical_checksum=logical)

    def _read_regular_file(self, relative_path: str) -> bytes:
        """Read through nofollow directory descriptors, closing path-swap races."""
        normalized = require_relative_path(relative_path, "relative_path")
        parts = PurePosixPath(normalized).parts
        if parts[:1] != (ATTEMPT_DIRECTORY_NAME,):
            msg = "Production artifacts must stay under production_attempts."
            raise ValueError(msg)
        supports_descriptor_walk = self._descriptor_creation_supported()
        if not supports_descriptor_walk:
            path = self._safe_path(normalized, require_file=True)
            parent = str(PurePosixPath(normalized).parent)
            parent_identities = self._fallback_directory_identities(parent)
            try:
                with path.open("rb") as handle:
                    metadata = os.fstat(handle.fileno())
                    if not stat.S_ISREG(metadata.st_mode):
                        msg = "Production artifact member is not a regular file."
                        raise ValueError(msg)
                    payload = handle.read()
                    disk = self._safe_path(normalized, require_file=True).lstat()
                    if self._fallback_directory_identities(parent) != parent_identities or (
                        disk.st_dev,
                        disk.st_ino,
                    ) != (metadata.st_dev, metadata.st_ino):
                        msg = "Production artifact identity changed during fallback read."
                        raise ValueError(msg)
                    return payload
            except OSError as error:
                msg = "Production artifact path is unavailable or unsafe."
                raise ValueError(msg) from error
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | os.O_NOFOLLOW
        directory_descriptor: int | None = None
        try:
            directory_descriptor = self._open_job_directory_descriptor(create=False)
            for component in parts[:-1]:
                next_descriptor = os.open(component, directory_flags, dir_fd=directory_descriptor)
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
            descriptor = os.open(parts[-1], file_flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = None
        except OSError as error:
            if directory_descriptor is not None:
                os.close(directory_descriptor)
            msg = "Production artifact path is unavailable or unsafe."
            raise ValueError(msg) from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                msg = "Production artifact member is not a regular file."
                raise ValueError(msg)
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                return handle.read()
        finally:
            os.close(descriptor)

    def _verified_bytes(self, ref: ArtifactBlobRef) -> bytes:
        """Read and verify one manifest member without following links."""
        payload = self._read_regular_file(ref.path)
        if len(payload) != ref.byte_count:
            msg = "Production artifact byte size or file type changed."
            raise ValueError(msg)
        if _sha256_bytes(payload) != ref.file_checksum:
            msg = "Production artifact bytes failed their immutable checksum."
            raise ValueError(msg)
        return payload

    def read_written_receipt(self, ref: ArtifactBlobRef) -> bytes:
        """Read one closed member created by this exact in-process attempt."""
        if ref not in self._written_refs:
            msg = "Partial-work receipt was not written by this production attempt."
            raise ValueError(msg)
        return self._verified_bytes(ref)

    def _publish_terminal_payload(self, payload: bytes) -> None:
        """Atomically link one fully synced terminal payload into its exclusive address."""
        token = secrets.token_hex(16)
        temporary_name = f".wp22e-terminal-{self.attempt:06d}-{token}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if self._descriptor_creation_supported():
            job_descriptor = self._open_job_directory_descriptor(create=True)
            parent_descriptor, manifest_name = self._open_parent_descriptor(
                self.manifest_relative_path,
                create=True,
            )
            temporary_descriptor: int | None = None
            try:
                temporary_descriptor = os.open(
                    temporary_name,
                    flags,
                    0o600,
                    dir_fd=job_descriptor,
                )
                with os.fdopen(temporary_descriptor, "wb", closefd=False) as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary_metadata = os.fstat(temporary_descriptor)
                named_metadata = os.stat(
                    temporary_name,
                    dir_fd=job_descriptor,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(temporary_metadata.st_mode) or (
                    temporary_metadata.st_dev,
                    temporary_metadata.st_ino,
                ) != (named_metadata.st_dev, named_metadata.st_ino):
                    msg = "Terminal staging identity changed before atomic publication."
                    raise ValueError(msg)
                try:
                    os.link(
                        temporary_name,
                        manifest_name,
                        src_dir_fd=job_descriptor,
                        dst_dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    msg = "Immutable production attempt manifest already exists."
                    raise ValueError(msg) from None
                published_descriptor = os.open(
                    manifest_name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
                try:
                    published_metadata = os.fstat(published_descriptor)
                    if (published_metadata.st_dev, published_metadata.st_ino) != (
                        temporary_metadata.st_dev,
                        temporary_metadata.st_ino,
                    ):
                        msg = "Atomic terminal link does not identify the synced staging file."
                        raise ValueError(msg)
                finally:
                    os.close(published_descriptor)
                os.fsync(parent_descriptor)
                verification_descriptor, verification_name = self._open_parent_descriptor(
                    self.manifest_relative_path,
                    create=False,
                )
                try:
                    verification_metadata = os.fstat(verification_descriptor)
                    pinned_metadata = os.fstat(parent_descriptor)
                    if verification_name != manifest_name or (
                        verification_metadata.st_dev,
                        verification_metadata.st_ino,
                    ) != (pinned_metadata.st_dev, pinned_metadata.st_ino):
                        msg = "Terminal parent changed during atomic descriptor publication."
                        raise ValueError(msg)
                finally:
                    os.close(verification_descriptor)
            finally:
                if temporary_descriptor is not None:
                    os.close(temporary_descriptor)
                with suppress(FileNotFoundError):
                    os.unlink(temporary_name, dir_fd=job_descriptor)
                os.close(parent_descriptor)
                os.close(job_descriptor)
            return

        self.job_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = self.job_directory / temporary_name
        manifest_path = self._safe_path(self.manifest_relative_path)
        self._mkdirs(self.relative_attempt_directory)
        parent_identities = self._fallback_directory_identities(self.relative_attempt_directory)
        descriptor = os.open(temporary_path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged = os.fstat(descriptor)
            named = temporary_path.lstat()
            if (staged.st_dev, staged.st_ino) != (named.st_dev, named.st_ino):
                msg = "Terminal staging identity changed before fallback publication."
                raise ValueError(msg)
            try:
                os.link(temporary_path, manifest_path, follow_symlinks=False)
            except FileExistsError:
                msg = "Immutable production attempt manifest already exists."
                raise ValueError(msg) from None
            published = self._safe_path(self.manifest_relative_path, require_file=True).lstat()
            if self._fallback_directory_identities(self.relative_attempt_directory) != parent_identities or (
                published.st_dev,
                published.st_ino,
            ) != (staged.st_dev, staged.st_ino):
                msg = "Fallback terminal publication changed parent or file identity."
                raise ValueError(msg)
        finally:
            os.close(descriptor)
            temporary_path.unlink(missing_ok=True)

    def publish(
        self,
        *,
        artifact_kind: ArtifactKind,
        status: ArtifactStatus,
        execution_source_manifest_checksum: str,
        source_fingerprint_checksum: str,
        blobs: Sequence[ArtifactBlobRef],
        evidence_ref: ArtifactBlobRef,
    ) -> ResultArtifactRef:
        """Publish the terminal manifest last and return its typed address."""
        ordered = tuple(sorted(blobs, key=lambda ref: ref.path))
        if self.member_paths() != tuple(ref.path for ref in ordered):
            msg = "Attempt members differ from the exact terminal manifest universe."
            raise ValueError(msg)
        manifest = AttemptArtifactManifest(
            job_checksum=self.job_checksum,
            attempt=self.attempt,
            artifact_kind=artifact_kind,
            status=status,
            execution_source_manifest_checksum=execution_source_manifest_checksum,
            source_fingerprint_checksum=source_fingerprint_checksum,
            blobs=ordered,
            evidence_ref=evidence_ref,
        )
        payload = _json_bytes(manifest.to_dict())
        self._publish_terminal_payload(payload)
        return ResultArtifactRef(
            job_checksum=self.job_checksum,
            attempt=self.attempt,
            artifact_kind=artifact_kind,
            status=status,
            execution_source_manifest_checksum=execution_source_manifest_checksum,
            source_fingerprint_checksum=source_fingerprint_checksum,
            manifest_path=self.manifest_relative_path,
            manifest_file_checksum=_sha256_bytes(payload),
            manifest_content_checksum=manifest.content_checksum,
            evidence_checksum=evidence_ref.logical_checksum,
        )

    def derive_existing_ref(self) -> ResultArtifactRef:
        """Derive a typed ref from a crash-published terminal manifest."""
        payload = self._read_regular_file(self.manifest_relative_path)
        manifest = AttemptArtifactManifest.from_dict(load_canonical_json_object(payload.decode()))
        if manifest.job_checksum != self.job_checksum or manifest.attempt != self.attempt:
            msg = "Existing terminal manifest belongs to a different job or attempt."
            raise ValueError(msg)
        return ResultArtifactRef(
            job_checksum=manifest.job_checksum,
            attempt=manifest.attempt,
            artifact_kind=manifest.artifact_kind,
            status=manifest.status,
            execution_source_manifest_checksum=manifest.execution_source_manifest_checksum,
            source_fingerprint_checksum=manifest.source_fingerprint_checksum,
            manifest_path=self.manifest_relative_path,
            manifest_file_checksum=_sha256_bytes(payload),
            manifest_content_checksum=manifest.content_checksum,
            evidence_checksum=manifest.evidence_ref.logical_checksum,
        )

    def reopen(self, reference: ResultArtifactRef) -> tuple[AttemptArtifactManifest, ProductionNumericalEvidence]:
        """Dereference and verify every manifest member and the typed evidence."""
        if not isinstance(reference, ResultArtifactRef):
            msg = "reference must be a ResultArtifactRef."
            raise TypeError(msg)
        if reference.job_checksum != self.job_checksum or reference.attempt != self.attempt:
            msg = "Result reference belongs to a different job or attempt."
            raise ValueError(msg)
        manifest_payload = self._read_regular_file(reference.manifest_path)
        if _sha256_bytes(manifest_payload) != reference.manifest_file_checksum:
            msg = "Attempt manifest bytes differ from the result reference."
            raise ValueError(msg)
        manifest = AttemptArtifactManifest.from_dict(load_canonical_json_object(manifest_payload.decode()))
        aliases = (
            (manifest.content_checksum, reference.manifest_content_checksum),
            (manifest.job_checksum, reference.job_checksum),
            (manifest.attempt, reference.attempt),
            (manifest.artifact_kind, reference.artifact_kind),
            (manifest.status, reference.status),
            (
                manifest.execution_source_manifest_checksum,
                reference.execution_source_manifest_checksum,
            ),
            (manifest.source_fingerprint_checksum, reference.source_fingerprint_checksum),
            (manifest.evidence_ref.logical_checksum, reference.evidence_checksum),
        )
        if any(actual != expected for actual, expected in aliases):
            msg = "Attempt manifest aliases differ from the typed result reference."
            raise ValueError(msg)
        actual_files = self.member_paths()
        expected_files = tuple(ref.path for ref in manifest.blobs)
        if actual_files != expected_files:
            msg = "Attempt directory member set differs from its immutable manifest."
            raise ValueError(msg)
        evidence: ProductionNumericalEvidence | None = None
        ensembles_by_path: dict[str, KrotovFixedMapEnsemble] = {}
        snapshots_by_path: dict[str, ScheduledExecutionSnapshot | OperatorGrowthSegmentedSnapshot] = {}
        for ref in manifest.blobs:
            payload = self._verified_bytes(ref)
            if ref.media_type == "application/json":
                document = load_canonical_json_object(payload.decode())
                supplied = require_checksum(document.get("content_checksum"), "artifact content_checksum")
                if supplied != ref.logical_checksum:
                    msg = "Artifact logical checksum differs from its manifest reference."
                    raise ValueError(msg)
                if ref.role == "fixed_map_ensemble":
                    ensemble = KrotovFixedMapEnsemble.from_json(payload.decode())
                    if ensemble.content_checksum != ref.logical_checksum:
                        msg = "Fixed-map ensemble differs from its logical manifest checksum."
                        raise ValueError(msg)
                    ensembles_by_path[ref.path] = ensemble
                elif ref.role == "schedule_snapshot":
                    snapshot = _decode_schedule_snapshot(payload)
                    if snapshot.content_checksum != ref.logical_checksum:
                        msg = "Schedule snapshot differs from its logical manifest checksum."
                        raise ValueError(msg)
                    snapshots_by_path[ref.path] = snapshot
                if ref == manifest.evidence_ref:
                    evidence = ProductionNumericalEvidence.from_dict(document)
        if evidence is None or evidence.content_checksum != reference.evidence_checksum:
            msg = "Attempt manifest does not reopen its exact typed evidence."
            raise ValueError(msg)
        evidence_aliases = (
            (evidence.job_checksum, manifest.job_checksum),
            (evidence.attempt, manifest.attempt),
            (evidence.artifact_kind, manifest.artifact_kind),
            (evidence.status, manifest.status),
            (
                evidence.execution_source_manifest_checksum,
                manifest.execution_source_manifest_checksum,
            ),
            (evidence.source_fingerprint_checksum, manifest.source_fingerprint_checksum),
        )
        if any(actual != expected for actual, expected in evidence_aliases):
            msg = "Production evidence aliases differ from its terminal manifest."
            raise ValueError(msg)
        manifest_by_path = {ref.path: ref for ref in manifest.blobs}
        evidence_refs = {
            ref.path
            for ref in (
                *(evidence.map_evidence_refs),
                *(evidence.diagnostic_refs),
                evidence.resource_ref,
                *(() if evidence.schedule_snapshot_ref is None else (evidence.schedule_snapshot_ref,)),
                *(() if evidence.raw_trajectory_ref is None else (evidence.raw_trajectory_ref,)),
            )
        }
        manifest_paths = {ref.path for ref in manifest.blobs}
        if not evidence_refs <= manifest_paths:
            msg = "Production evidence references a blob outside its exact manifest."
            raise ValueError(msg)
        referenced = (
            *(evidence.map_evidence_refs),
            *(evidence.diagnostic_refs),
            evidence.resource_ref,
            *(() if evidence.schedule_snapshot_ref is None else (evidence.schedule_snapshot_ref,)),
            *(() if evidence.raw_trajectory_ref is None else (evidence.raw_trajectory_ref,)),
        )
        if any(manifest_by_path.get(ref.path) != ref for ref in referenced):
            msg = "Production evidence blob aliases differ from the terminal manifest."
            raise ValueError(msg)
        authoritative_snapshot: ScheduledExecutionSnapshot | OperatorGrowthSegmentedSnapshot | None = None
        if evidence.schedule_snapshot_ref is not None:
            authoritative_snapshot = snapshots_by_path.get(evidence.schedule_snapshot_ref.path)
            if authoritative_snapshot is None:
                msg = "Production evidence does not resolve its authoritative schedule snapshot."
                raise ValueError(msg)
            if authoritative_snapshot.program_checksum != evidence.scheduled_program_checksum:
                msg = "Schedule snapshot belongs to a different production program."
                raise ValueError(msg)
        decoded_map_evidence: list[ScheduledMapEvidence] = []
        for map_ref in evidence.map_evidence_refs:
            document = load_canonical_json_object(self._verified_bytes(map_ref).decode())
            map_evidence = ScheduledMapEvidence.from_dict(document)
            decoded_map_evidence.append(map_evidence)
            if any(manifest_by_path.get(ref.path) != ref for ref in map_evidence.ensemble_refs):
                msg = "Scheduled map evidence references a non-manifest ensemble alias."
                raise ValueError(msg)
            for index, ensemble_ref in enumerate(map_evidence.ensemble_refs):
                ensemble = ensembles_by_path.get(ensemble_ref.path)
                if ensemble is None:
                    msg = "Scheduled map evidence does not resolve to a decoded fixed-map ensemble."
                    raise ValueError(msg)
                expected = (
                    map_evidence.map_role,
                    map_evidence.resolved_seeds[index],
                    map_evidence.circuit_checksum,
                    map_evidence.provider_checksums[index],
                    len(map_evidence.component_member_seeds[index]),
                )
                actual = (
                    ensemble.role,
                    ensemble.resolved_seed,
                    ensemble.circuit_checksum,
                    ensemble.provider_checksum,
                    ensemble.trajectory_count,
                )
                if actual != expected:
                    msg = (
                        "Scheduled map ensemble role, resolved/member seeds, circuit, provider, "
                        "or trajectory count differs from its evidence."
                    )
                    raise ValueError(msg)
        resource_document = load_canonical_json_object(self._verified_bytes(evidence.resource_ref).decode())
        resource_custody = _validate_resource_document(resource_document, evidence=evidence)
        if evidence.artifact_kind == "pipeline" and authoritative_snapshot is not None:
            if not isinstance(authoritative_snapshot, ScheduledExecutionSnapshot):
                msg = "Pipeline production requires its authoritative scheduled-execution snapshot."
                raise TypeError(msg)
            if resource_custody.circuit_checksum is None or resource_custody.circuit_gate_count is None:
                msg = "Pipeline scheduled-map replay requires its exact runtime circuit resources."
                raise ValueError(msg)
            expected_pipeline_links = _pipeline_snapshot_numerical_links(
                authoritative_snapshot,
                target_identity=evidence.target_identity,
                circuit_checksum=resource_custody.circuit_checksum,
                circuit_gate_count=resource_custody.circuit_gate_count,
            )
            if len(decoded_map_evidence) != len(expected_pipeline_links):
                msg = "Pipeline callback count differs from persisted scheduled-map evidence."
                raise ValueError(msg)
            for map_evidence, expected_link in zip(
                decoded_map_evidence,
                expected_pipeline_links,
                strict=True,
            ):
                _validate_pipeline_numerical_link(
                    map_evidence,
                    expected_link,
                    tuple(ensembles_by_path[ref.path] for ref in map_evidence.ensemble_refs),
                )
            if evidence.status == "success":
                multistart = authoritative_snapshot.multistart_evidence
                if multistart is None:
                    msg = "Successful pipeline production cannot reference an incomplete snapshot."
                    raise ValueError(msg)
                metrics = evidence.derived_metrics
                if (
                    metrics.get("selected_start_index") != multistart.selected_start_index
                    or metrics.get("selected_update") != multistart.selected_update
                    or metrics.get("selected_parameter_checksum") != multistart.selected_parameter_checksum
                    or metrics.get("total_normalized_training_work") != multistart.total_normalized_work
                ):
                    msg = "Pipeline derived selection/work aliases differ from its authoritative snapshot."
                    raise ValueError(msg)
        if (
            evidence.artifact_kind == "operator_growth"
            and evidence.status == "success"
            and (evidence.derived_metrics.get("execution_preset") != "training-smoke")
        ):
            if not isinstance(authoritative_snapshot, OperatorGrowthSegmentedSnapshot):
                msg = "Successful operator production requires its authoritative segmented snapshot."
                raise ValueError(msg)
            if not authoritative_snapshot.complete:
                msg = "Successful operator production cannot reference an incomplete segmented snapshot."
                raise ValueError(msg)
            if evidence.structural_prefix_checksums != (authoritative_snapshot.content_checksum,):
                msg = "Operator structural result alias differs from its authoritative segmented snapshot."
                raise ValueError(msg)
            expected_links = _operator_snapshot_numerical_links(authoritative_snapshot)
            if len(decoded_map_evidence) != len(expected_links):
                msg = "Operator objective/validation count differs from persisted scheduled-map evidence."
                raise ValueError(msg)
            for map_evidence, expected_link in zip(decoded_map_evidence, expected_links, strict=True):
                _validate_operator_numerical_link(
                    map_evidence,
                    expected_link,
                    tuple(ensembles_by_path[ensemble_ref.path] for ensemble_ref in map_evidence.ensemble_refs),
                )
        diagnostic_member_count = 0
        for diagnostic_ref in evidence.diagnostic_refs:
            document = load_canonical_json_object(self._verified_bytes(diagnostic_ref).decode())
            diagnostic = PilotDiagnosticEvidence.from_dict(document)
            diagnostic_member_count += len(diagnostic.member_seeds)
            if diagnostic.job_checksum != manifest.job_checksum:
                msg = "Pilot diagnostic evidence belongs to a different production job."
                raise ValueError(msg)
            if any(manifest_by_path.get(ref.path) != ref for ref in diagnostic.ensemble_refs):
                msg = "Pilot diagnostic evidence references a non-manifest ensemble alias."
                raise ValueError(msg)
            for index, (member_seed, ensemble_ref) in enumerate(
                zip(diagnostic.member_seeds, diagnostic.ensemble_refs, strict=True)
            ):
                ensemble = ensembles_by_path.get(ensemble_ref.path)
                if ensemble is None:
                    msg = "Pilot diagnostic does not resolve to a decoded fixed-map ensemble."
                    raise ValueError(msg)
                expected = (
                    "pilot_evaluation",
                    member_seed,
                    diagnostic.circuit_checksum,
                    diagnostic.provider_checksum,
                    1,
                    index,
                )
                actual = (
                    ensemble.role,
                    ensemble.resolved_seed,
                    ensemble.circuit_checksum,
                    ensemble.provider_checksum,
                    ensemble.trajectory_count,
                    ensemble.ensemble_index,
                )
                if actual != expected:
                    msg = "Pilot diagnostic member seed, map role, circuit, provider, or path order changed."
                    raise ValueError(msg)
        raw_custody: _RawTrajectoryCustody | None = None
        if evidence.raw_trajectory_ref is not None:
            raw_document = load_canonical_json_object(self._verified_bytes(evidence.raw_trajectory_ref).decode())
            raw_custody = _validate_raw_trajectory_document(
                raw_document,
                evidence=evidence,
                ensembles_by_path=ensembles_by_path,
                resource=resource_custody,
                snapshot=authoritative_snapshot,
            )
        if isinstance(authoritative_snapshot, OperatorGrowthSegmentedSnapshot):
            selected_binding = _operator_circuit_binding(
                authoritative_snapshot.pool,
                authoritative_snapshot.selected_operator_ids,
            )
            metrics = evidence.derived_metrics
            if (
                metrics.get("segmented_snapshot_checksum") != authoritative_snapshot.content_checksum
                or metrics.get("selected_prefix_index") != authoritative_snapshot.selected_prefix_index
                or metrics.get("selected_operator_ids") != authoritative_snapshot.selected_operator_ids
                or resource_custody.circuit_checksum != selected_binding.content_checksum
                or resource_custody.circuit_gate_count != len(selected_binding.circuit.gates)
            ):
                msg = "Fresh operator evaluation differs from the validation-selected snapshot prefix."
                raise ValueError(msg)
        _validate_normalized_work(
            evidence=evidence,
            resource=resource_custody,
            raw=raw_custody,
            diagnostic_member_count=diagnostic_member_count,
        )
        return manifest, evidence


@dataclass(frozen=True, slots=True)
class ReopenedProductionResult:
    """Fully verified typed view of a source-addressed production attempt."""

    reference: ResultArtifactRef
    manifest: AttemptArtifactManifest
    evidence: ProductionNumericalEvidence
    raw_trajectory: Mapping[str, object] | None
    resources: Mapping[str, object]
    scheduled_map_evidence: tuple[ScheduledMapEvidence, ...]
    diagnostic_documents: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        """Require mutually consistent reopened identities and document roles."""
        if not isinstance(self.reference, ResultArtifactRef):
            msg = "reference must be a ResultArtifactRef."
            raise TypeError(msg)
        if not isinstance(self.manifest, AttemptArtifactManifest):
            msg = "manifest must be an AttemptArtifactManifest."
            raise TypeError(msg)
        if not isinstance(self.evidence, ProductionNumericalEvidence):
            msg = "evidence must be ProductionNumericalEvidence."
            raise TypeError(msg)
        raw = None if self.raw_trajectory is None else dict(require_mapping(self.raw_trajectory, "raw_trajectory"))
        resources = dict(require_mapping(self.resources, "resources"))
        maps = tuple(self.scheduled_map_evidence)
        diagnostics = tuple(dict(require_mapping(item, "diagnostic_document")) for item in self.diagnostic_documents)
        if any(not isinstance(item, ScheduledMapEvidence) for item in maps):
            msg = "scheduled_map_evidence must contain typed records."
            raise TypeError(msg)
        if len(maps) != len(self.evidence.map_evidence_refs):
            msg = "Reopened scheduled-map evidence count differs from its typed references."
            raise ValueError(msg)
        if len(diagnostics) != len(self.evidence.diagnostic_refs):
            msg = "Reopened diagnostic document count differs from its typed references."
            raise ValueError(msg)
        object.__setattr__(self, "raw_trajectory", raw)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "scheduled_map_evidence", maps)
        object.__setattr__(self, "diagnostic_documents", diagnostics)

    @property
    def pilot_diagnostics(self) -> tuple[PilotDiagnosticEvidence, ...]:
        """Typed q6 pathwise diagnostic records verified during reopening."""
        return tuple(PilotDiagnosticEvidence.from_dict(item) for item in self.diagnostic_documents)


def derive_result_artifact_ref(
    job_directory: Path,
    job_checksum: str,
    attempt: int,
    *,
    expected_reference_checksum: str | None = None,
) -> ResultArtifactRef:
    """Derive the sole typed result address from an immutable terminal manifest."""
    store = ProductionAttemptStore(job_directory, job_checksum, attempt)
    reference = store.derive_existing_ref()
    if expected_reference_checksum is not None and reference.content_checksum != require_checksum(
        expected_reference_checksum,
        "expected_reference_checksum",
    ):
        msg = "Derived result reference differs from the orchestration outcome checksum."
        raise ValueError(msg)
    store.reopen(reference)
    return reference


def reopen_result_artifact(reference: ResultArtifactRef, job_directory: Path) -> ReopenedProductionResult:
    """Dereference a result and expose only manifest-verified numerical documents."""
    if not isinstance(reference, ResultArtifactRef):
        msg = "reference must be a ResultArtifactRef."
        raise TypeError(msg)
    store = ProductionAttemptStore(job_directory, reference.job_checksum, reference.attempt)
    manifest, evidence = store.reopen(reference)

    def document(ref: ArtifactBlobRef) -> Mapping[str, object]:
        return load_canonical_json_object(
            store._verified_bytes(ref).decode()  # noqa: SLF001
        )

    raw = None if evidence.raw_trajectory_ref is None else document(evidence.raw_trajectory_ref)
    resources = document(evidence.resource_ref)
    maps = tuple(ScheduledMapEvidence.from_dict(document(ref)) for ref in evidence.map_evidence_refs)
    diagnostics = tuple(document(ref) for ref in evidence.diagnostic_refs)
    return ReopenedProductionResult(reference, manifest, evidence, raw, resources, maps, diagnostics)


@dataclass(frozen=True, slots=True)
class ResolvedProductionJob:
    """Exact context-owned executable, target, evaluation, and schedule closure."""

    job: TrainingJob
    executable_binding: ExecutableScopedBinding
    target_configuration: TargetPopulationConfig
    target_manifest: TargetPopulationManifest
    target_spec: TargetInstanceSpec
    target: MaterializedTarget
    evaluation_policy: FreshEvaluationPolicy
    scheduled_program: ScheduledExecutionProgram
    execution_source_manifest_checksum: str
    screening_cell: ScreeningCell | None
    confirm_request: ConfirmExecutionRequest | None = None

    def __post_init__(self) -> None:
        """Validate the direct execution-source manifest identity."""
        object.__setattr__(
            self,
            "execution_source_manifest_checksum",
            require_checksum(self.execution_source_manifest_checksum, "execution_source_manifest_checksum"),
        )
        if self.confirm_request is not None:
            if not isinstance(self.confirm_request, ConfirmExecutionRequest):
                msg = "confirm_request must be a ConfirmExecutionRequest."
                raise TypeError(msg)
            if self.job.confirm_execution_request is not self.confirm_request or self.job.preset != "paper-confirm":
                msg = "A real confirmation resolution must retain its exact context-owned nested request."
                raise ValueError(msg)

    @property
    def evidence_identity_checksum(self) -> str:
        """Request identity for confirmation, otherwise the enclosing job identity."""
        if self.confirm_request is not None:
            return self.confirm_request.content_checksum
        return self.job.content_checksum

    @property
    def source_fingerprint_checksum(self) -> str:
        """Frozen source identity used by terminal evidence."""
        if self.confirm_request is not None:
            return self.confirm_request.execution_source_checksum
        return require_checksum(self.job.source_fingerprint_checksum, "job.source_fingerprint_checksum")

    @property
    def executable_binding_checksum(self) -> str:
        """Exact executable binding root used by terminal evidence."""
        if self.confirm_request is not None:
            return self.confirm_request.executable_binding_checksum
        return require_checksum(self.job.executable_binding_checksum, "job.executable_binding_checksum")

    @property
    def strategy_schedule_checksum(self) -> str:
        """Exact sealed strategy schedule underlying the compiled program."""
        if self.confirm_request is not None:
            return self.confirm_request.hyperparameters_checksum
        return self.job.strategy_schedule_checksum

    @property
    def evaluation_policy_checksum(self) -> str:
        """Request-bound confirmation identity or ordinary policy checksum."""
        if self.confirm_request is not None:
            return confirmatory_evaluation_policy_checksum(self.confirm_request)
        return self.evaluation_policy.content_checksum


class ProductionExecutionAuthority:
    """Resolve only exact plan member objects and cache authorized targets."""

    def __init__(self, context: TrainingExecutionContext) -> None:
        """Bind a non-serializable complete WP22D authority."""
        if not isinstance(context, TrainingExecutionContext):
            msg = "context must be a TrainingExecutionContext."
            raise TypeError(msg)
        self.context = context
        self._materializations: dict[str, tuple[MaterializedTarget, ...]] = {}

    def _materialized_targets(
        self,
        config: TargetPopulationConfig,
        manifest: TargetPopulationManifest,
    ) -> tuple[MaterializedTarget, ...]:
        cached = self._materializations.get(manifest.content_checksum)
        if cached is not None:
            return cached
        authorized = next(
            (
                item
                for item in self.context.authorized_materializations
                if item.target_configuration == config and item.target_manifest == manifest
            ),
            None,
        )
        if authorized is None:
            msg = "Target config and manifest lack exact materialization authority."
            raise ValueError(msg)
        entropy = self.context.external_entropy_keyring.entropy_for(config.data_role, config.population_scope)
        population = materialize_target_population(
            config,
            self.context.preregistration,
            manifest,
            entropy,
            authorized.authorization,
        )
        targets = population.targets
        self._materializations[manifest.content_checksum] = targets
        return targets

    def resolve(self, job: TrainingJob) -> ResolvedProductionJob:
        """Resolve one exact object member before any output mutation."""
        if not isinstance(job, TrainingJob):
            msg = "job must be a TrainingJob."
            raise TypeError(msg)
        planned = next((candidate for candidate in self.context.plan.jobs if candidate is job), None)
        if planned is None:
            msg = "Production execution accepts only the exact TrainingJob object owned by its context."
            raise ValueError(msg)
        scope = _target_scope(job.qubit_count)
        links = tuple(
            link
            for link in self.context.scoped_bindings
            if link.binding.publication_candidate_checksum == job.candidate_configuration_checksum
            and link.binding.target_scope_id == scope
        )
        if len(links) != 1:
            msg = "Job has no unique exact executable scoped binding."
            raise ValueError(msg)
        link = links[0]
        link.resolve_callable()
        binding = link.binding
        configs = tuple(
            config
            for config in self.context.target_configurations
            if config.content_checksum == job.target_configuration_checksum
        )
        manifests = tuple(
            manifest
            for manifest in self.context.target_manifests
            if manifest.content_checksum == job.target_manifest_checksum
        )
        if len(configs) != 1 or len(manifests) != 1:
            msg = "Job lacks a unique exact target config/manifest pair."
            raise ValueError(msg)
        config, manifest = configs[0], manifests[0]
        if manifest.population_config_checksum != config.content_checksum:
            msg = "Target manifest does not belong to the job target configuration."
            raise ValueError(msg)
        specs = tuple(item for item in manifest.instances if item.target_instance_id == job.target_instance_id)
        if len(specs) != 1 or specs[0].content_checksum != job.target_spec_checksum:
            msg = "Job target is absent from its exact manifest."
            raise ValueError(msg)
        purpose = {
            "training-smoke": "smoke_evaluation",
            "paper-pilot": "pilot_fresh_evaluation",
            "paper-screen": "screening_outer",
        }.get(job.preset)
        policies = tuple(policy for policy in binding.evaluation_policies if policy.purpose == purpose)
        if len(policies) != 1 or policies[0].content_checksum != job.evaluation_policy_checksum:
            msg = "Job lacks its unique exact fresh-evaluation policy."
            raise ValueError(msg)
        program = ScheduledExecutionProgram.compile(
            link,
            binding.strategy_schedule,
            ScheduledJobSeedSet(job.optimization_seed),
        )
        expected = (
            (job.execution_profile_checksum, self.context.execution_profile.content_checksum),
            (job.scoped_binding_checksum, binding.content_checksum),
            (job.executable_binding_checksum, link.content_checksum),
            (job.implementation_checksum, binding.implementation_checksum),
            (job.strategy_schedule_checksum, binding.strategy_schedule.content_checksum),
            (job.source_fingerprint_checksum, self.context.source_fingerprint_checksum),
            (job.scheduled_execution_program_checksum, program.content_checksum),
        )
        if any(actual != required for actual, required in expected):
            msg = "Job fingerprint differs from its exact production execution closure."
            raise ValueError(msg)
        target = next(
            item
            for item in self._materialized_targets(config, manifest)
            if item.target_instance_id == job.target_instance_id
        )
        cell = None
        if job.preset == "paper-screen":
            cells = tuple(
                candidate
                for candidate in self.context.screening_cells
                if candidate.target_instance_id == job.target_instance_id
                and candidate.optimization_seed == job.optimization_seed
                and candidate.screening_seed == job.evaluation_seed
                and candidate.family_id == job.family_id
                and candidate.stratum_id == job.stratum_id
            )
            if len(cells) != 1:
                msg = "Paper-screen job has no unique exact ScreeningCell."
                raise ValueError(msg)
            cell = cells[0]
        return ResolvedProductionJob(
            job,
            link,
            config,
            manifest,
            specs[0],
            target,
            policies[0],
            program,
            self.context.execution_source_manifest.content_checksum,
            cell,
        )


class ProductionConfirmationAuthority:
    """Resolve only exact context-owned real confirmatory request objects."""

    def __init__(self, context: ConfirmationExecutionContext) -> None:
        """Bind the narrow final-seal authority without materializing targets."""
        if not isinstance(context, ConfirmationExecutionContext):
            msg = "context must be a ConfirmationExecutionContext."
            raise TypeError(msg)
        self.context = context
        self._materialized_targets: tuple[MaterializedTarget, ...] | None = None

    def _targets(self) -> tuple[MaterializedTarget, ...]:
        """Lazily materialize the externally revealed confirmatory population."""
        if self._materialized_targets is None:
            self._materialized_targets = self.context.materialize_targets()
        return self._materialized_targets

    def resolve(self, request: ConfirmExecutionRequest) -> ResolvedProductionJob:
        """Close one exact nested request to frozen code, target, and policies.

        Returns:
            The repository-owned production execution closure.

        Raises:
            ValueError: If the request is not the exact object owned by the
                complete final-seal Cartesian plan or any sealed root differs.
        """
        if not isinstance(request, ConfirmExecutionRequest):
            msg = "request must be a ConfirmExecutionRequest."
            raise TypeError(msg)
        jobs = tuple(job for job in self.context.plan.jobs if job.confirm_execution_request is request)
        if len(jobs) != 1:
            msg = "Production confirmation accepts only an exact context-owned request object."
            raise ValueError(msg)
        job = jobs[0]
        validate_confirm_execution_request(
            request,
            self.context.final_seal,
            self.context.target_manifest,
            self.context.configuration_execution_manifest,
        )
        link = self.context.executable_binding(request.configuration_checksum)
        link.resolve_callable()
        binding = link.binding
        execution = self.context.configuration_execution_manifest.entry(request.configuration_checksum)
        expected_cap = cast("float", request.primary_resource_budget["normalized_compute_cap"])
        if (
            binding.strategy_schedule != execution.strategy_schedule
            or binding.execution_budget.normalized_compute_cap is None
            or float(binding.execution_budget.normalized_compute_cap).hex() != float(expected_cap).hex()
            or binding.resource_policy.to_dict()["cap_per_chain_edge"]
            != request.primary_resource_budget["cap_per_chain_edge"]
        ):
            msg = "Confirmatory binding schedule or resource limits differ from the final request."
            raise ValueError(msg)
        policy = FreshEvaluationPolicy.confirmatory(request.fixed_test_trajectory_count)
        if canonical_json(policy.noise_condition) != canonical_json(request.primary_noise_condition):
            msg = "Confirmatory fresh-evaluation noise differs from the final request."
            raise ValueError(msg)
        program = ScheduledExecutionProgram.compile(
            link,
            execution.strategy_schedule,
            ScheduledJobSeedSet(request.optimization_seed),
        )
        specs = tuple(
            spec
            for spec in self.context.target_manifest.instances
            if spec.target_instance_id == request.target_instance_id
        )
        targets = tuple(target for target in self._targets() if target.target_instance_id == request.target_instance_id)
        if (
            len(specs) != 1
            or specs[0].content_checksum != request.target_spec_checksum
            or len(targets) != 1
            or targets[0].target_instance_spec_checksum != request.target_spec_checksum
        ):
            msg = "Confirmatory request does not resolve one exact authorized target vector."
            raise ValueError(msg)
        return ResolvedProductionJob(
            job=job,
            executable_binding=link,
            target_configuration=self.context.target_configuration,
            target_manifest=self.context.target_manifest,
            target_spec=specs[0],
            target=targets[0],
            evaluation_policy=policy,
            scheduled_program=program,
            execution_source_manifest_checksum=self.context.execution_source_manifest.content_checksum,
            screening_cell=None,
            confirm_request=request,
        )


@dataclass(frozen=True, slots=True)
class _MeasuredCall:
    """One callback result and authoritative local runtime measurements."""

    value: object
    wall_time_seconds: float
    peak_memory_bytes: int


def _measure_call(callback: Callable[[], object]) -> _MeasuredCall:
    """Execute a callback while measuring monotonic time and Python allocations."""
    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
        tracemalloc.reset_peak()
    baseline, _ = tracemalloc.get_traced_memory()
    started = time.perf_counter()
    try:
        value = callback()
    finally:
        elapsed = time.perf_counter() - started
        current, peak = tracemalloc.get_traced_memory()
        measured_peak = max(current - baseline, peak - baseline, 0)
        if owns_tracing:
            tracemalloc.stop()
    return _MeasuredCall(value, max(float(elapsed), 0.0), measured_peak)


def _typed_document(document_type: str, payload: Mapping[str, object]) -> dict[str, object]:
    """Seal one small repository-owned evidence document."""
    return _sealed({
        "schema_version": PRODUCTION_DOCUMENT_SCHEMA_VERSION,
        "document_type": require_slug(document_type, "document_type"),
        "payload": dict(payload),
    })


def _production_document_payload(
    document: object,
    *,
    document_type: str,
    payload_keys: frozenset[str],
) -> Mapping[str, object]:
    """Decode one strict checksum-sealed production sidecar payload."""
    mapping = verify_sealed_mapping(
        document,
        expected_keys=_PRODUCTION_DOCUMENT_KEYS,
        name=f"{document_type} production document",
    )
    if mapping["schema_version"] != PRODUCTION_DOCUMENT_SCHEMA_VERSION:
        msg = f"{document_type} uses an unsupported production document schema."
        raise ValueError(msg)
    if mapping["document_type"] != document_type:
        msg = f"Production document has the wrong {document_type} role."
        raise ValueError(msg)
    payload = require_mapping(mapping["payload"], f"{document_type} payload")
    if set(payload) != payload_keys:
        missing = sorted(payload_keys - set(payload))
        extra = sorted(set(payload) - payload_keys)
        msg = f"{document_type} payload keys differ: missing={missing!r}, extra={extra!r}."
        raise ValueError(msg)
    return payload


@dataclass(frozen=True, slots=True)
class _ResourceCustody:
    """Mechanically verified resource aliases needed by other custody checks."""

    normalized_work: float
    circuit_checksum: str | None
    circuit_gate_count: int | None


def _validate_resource_circuit(
    value: object,
    *,
    evidence: ProductionNumericalEvidence,
) -> tuple[str, int]:
    """Decode and cross-check the complete logical and compiled circuit record."""
    circuit = require_mapping(value, "runtime resource circuit")
    if set(circuit) != _RESOURCE_CIRCUIT_KEYS:
        msg = "Runtime resource circuit fields differ from the complete WP22E schema."
        raise ValueError(msg)
    binding_checksum = require_checksum(circuit["circuit_binding_checksum"], "circuit_binding_checksum")
    require_slug(circuit["topology_id"], "topology_id")
    qubit_count = require_int(circuit["qubit_count"], "qubit_count", minimum=1)
    parameter_count = require_int(circuit["parameter_count"], "parameter_count")
    logical_gate_count = require_int(circuit["logical_gate_count"], "logical_gate_count")
    logical_two_qubit_gate_count = require_int(
        circuit["logical_two_qubit_gate_count"],
        "logical_two_qubit_gate_count",
    )
    noisy_gate_indices = tuple(
        require_int(item, "noisy_gate_index")
        for item in _strict_tuple(circuit["noisy_gate_indices"], "noisy_gate_indices")
    )
    if len(noisy_gate_indices) != len(set(noisy_gate_indices)) or any(
        index >= logical_gate_count for index in noisy_gate_indices
    ):
        msg = "Runtime noisy-gate indices are duplicate or outside the logical circuit."
        raise ValueError(msg)
    compiled = CircuitResourceMetrics.from_dict(circuit["compiled_resources"])
    compiled_checksum = require_checksum(circuit["compiled_resources_checksum"], "compiled_resources_checksum")
    edge_counts = tuple(
        require_int(item, "native_two_qubit_gates_per_chain_edge")
        for item in _strict_tuple(
            circuit["native_two_qubit_gates_per_chain_edge"],
            "native_two_qubit_gates_per_chain_edge",
        )
    )
    target_qubits = require_int(evidence.target_identity.get("qubit_count"), "target_identity.qubit_count", minimum=1)
    if (
        qubit_count != target_qubits
        or compiled.qubit_count != qubit_count
        or compiled.trainable_parameter_count != parameter_count
        or len(compiled.logical_events) != logical_gate_count
        or compiled.logical_two_qubit_gates != logical_two_qubit_gate_count
        or compiled.content_checksum != compiled_checksum
        or compiled.native_two_qubit_gates_per_chain_edge != edge_counts
    ):
        msg = "Runtime circuit aliases differ from mechanically decoded WP20 resources."
        raise ValueError(msg)
    fresh_binding = evidence.derived_metrics.get("fresh_circuit_binding_checksum")
    if fresh_binding is not None and fresh_binding != binding_checksum:
        msg = "Runtime circuit differs from the fresh-evaluation circuit alias."
        raise ValueError(msg)
    return binding_checksum, logical_gate_count


def _validate_resource_document(
    document: object,
    *,
    evidence: ProductionNumericalEvidence,
) -> _ResourceCustody:
    """Validate runtime identity, work scalars, failure state, and circuit resources."""
    if evidence.artifact_kind == "synthetic_confirmation":
        payload = _production_document_payload(
            document,
            document_type="runtime_resources",
            payload_keys=_SYNTHETIC_RESOURCE_KEYS,
        )
        if (
            payload["request_checksum"] != evidence.job_checksum
            or payload["source_fingerprint_checksum"] != evidence.source_fingerprint_checksum
            or payload["synthetic_fixture"] is not True
            or payload["circuit"] is not None
        ):
            msg = "Synthetic runtime resources differ from their exact request or fixture."
            raise ValueError(msg)
        wall_time = require_float(payload["wall_time_seconds"], "wall_time_seconds", minimum=0.0)
        peak_memory = require_int(payload["peak_memory_bytes"], "peak_memory_bytes")
        normalized_work = require_float(payload["normalized_work"], "normalized_work", minimum=0.0)
        if (
            not math.isclose(wall_time, 0.0, rel_tol=0.0, abs_tol=0.0)
            or peak_memory != 0
            or not math.isclose(normalized_work, 0.0, rel_tol=0.0, abs_tol=0.0)
        ):
            msg = "Synthetic runtime resources must record exactly zero measured work."
            raise ValueError(msg)
        return _ResourceCustody(normalized_work, None, None)

    payload = _production_document_payload(
        document,
        document_type="runtime_resources",
        payload_keys=_REAL_RESOURCE_KEYS,
    )
    if (
        payload["job_checksum"] != evidence.job_checksum
        or payload["source_fingerprint_checksum"] != evidence.source_fingerprint_checksum
    ):
        msg = "Runtime resource job or source identity differs from production evidence."
        raise ValueError(msg)
    require_float(payload["wall_time_seconds"], "wall_time_seconds", minimum=0.0)
    require_int(payload["peak_memory_bytes"], "peak_memory_bytes")
    normalized_work = require_float(payload["normalized_work"], "normalized_work", minimum=0.0)
    failure_phase = payload["failure_phase"]
    partial_receipts = payload["partial_receipts"]
    if evidence.status == "failure":
        failure = require_mapping(evidence.failure, "failure")
        if failure_phase != failure.get("phase") or partial_receipts is None:
            msg = "Failure resources do not reproduce the structured failure phase and partial receipts."
            raise ValueError(msg)
        require_mapping(partial_receipts, "partial_receipts")
    elif failure_phase is not None or partial_receipts is not None:
        msg = "Successful runtime resources cannot carry failure-only fields."
        raise ValueError(msg)

    circuit_checksum: str | None = None
    circuit_gate_count: int | None = None
    if payload["circuit"] is not None:
        circuit_checksum, circuit_gate_count = _validate_resource_circuit(payload["circuit"], evidence=evidence)
    preset = evidence.derived_metrics.get("execution_preset")
    circuit_required = evidence.status == "success" and not (
        evidence.artifact_kind == "operator_growth" and preset == "training-smoke"
    )
    if circuit_required and circuit_checksum is None:
        msg = "Runtime circuit presence differs from the successful execution family."
        raise ValueError(msg)
    if (
        evidence.status == "success"
        and evidence.artifact_kind == "operator_growth"
        and preset == "training-smoke"
        and circuit_checksum is not None
    ):
        msg = "Operator-growth smoke resources cannot claim an unmeasured circuit."
        raise ValueError(msg)
    return _ResourceCustody(normalized_work, circuit_checksum, circuit_gate_count)


@dataclass(frozen=True, slots=True)
class _RawTrajectoryCustody:
    """Verified fresh trajectory values and their exact fixed-map ensemble."""

    trajectory_count: int
    trajectory_fidelities: tuple[float, ...]
    fresh_ensemble: KrotovFixedMapEnsemble | None


def _selected_snapshot_parameters(
    snapshot: ScheduledExecutionSnapshot | OperatorGrowthSegmentedSnapshot | None,
) -> tuple[float, ...] | None:
    """Return validation-selected parameters from either complete snapshot family."""
    if isinstance(snapshot, OperatorGrowthSegmentedSnapshot):
        return snapshot.selected_parameters
    if isinstance(snapshot, ScheduledExecutionSnapshot) and snapshot.multistart_evidence is not None:
        return snapshot.multistart_evidence.selected_parameter_artifact.parameters
    return None


def _validate_raw_metrics(
    *,
    evidence: ProductionNumericalEvidence,
    fidelities: tuple[float, ...],
    fresh_ensemble: KrotovFixedMapEnsemble,
) -> None:
    """Recompute every raw-vector-derived reporting metric."""
    metrics = evidence.derived_metrics
    count = len(fidelities)
    mean_fidelity = float(np.mean(fidelities))
    reporting_prefixes = tuple(
        require_int(item, "reporting_prefix")
        for item in _strict_tuple(metrics.get("reporting_prefixes"), "reporting_prefixes")
    )
    if not reporting_prefixes or any(prefix <= 0 or prefix > count for prefix in reporting_prefixes):
        msg = "Reporting prefixes lie outside the complete raw trajectory vector."
        raise ValueError(msg)
    prefix_metrics = require_mapping(metrics.get("prefix_mean_fidelities"), "prefix_mean_fidelities")
    expected_prefix_keys = {str(prefix) for prefix in reporting_prefixes}
    if set(prefix_metrics) != expected_prefix_keys:
        msg = "Prefix mean fidelity keys differ from the declared reporting prefixes."
        raise ValueError(msg)
    if (
        require_int(metrics.get("trajectory_count"), "derived trajectory_count") != count
        or not math.isclose(
            require_float(metrics.get("noisy_fidelity"), "noisy_fidelity", minimum=0.0, maximum=1.0),
            mean_fidelity,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or metrics.get("fresh_ensemble_checksum") != fresh_ensemble.content_checksum
        or metrics.get("provider_checksum") != fresh_ensemble.provider_checksum
        or require_int(metrics.get("sampled_nonidentity_events"), "sampled_nonidentity_events")
        != fresh_ensemble.nonidentity_event_count
        or any(
            not math.isclose(
                require_float(prefix_metrics[str(prefix)], f"prefix_mean_fidelities[{prefix}]"),
                float(np.mean(fidelities[:prefix])),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for prefix in reporting_prefixes
        )
    ):
        msg = "Derived fresh-evaluation metrics differ from the complete raw fidelity vector."
        raise ValueError(msg)


def _validate_raw_trajectory_document(
    document: object,
    *,
    evidence: ProductionNumericalEvidence,
    ensembles_by_path: Mapping[str, KrotovFixedMapEnsemble],
    resource: _ResourceCustody,
    snapshot: ScheduledExecutionSnapshot | OperatorGrowthSegmentedSnapshot | None,
) -> _RawTrajectoryCustody:
    """Validate raw trajectory identity, values, map custody, and selected parameters."""
    synthetic = evidence.artifact_kind == "synthetic_confirmation"
    payload = _production_document_payload(
        document,
        document_type="raw_trajectory_fidelities",
        payload_keys=_SYNTHETIC_RAW_TRAJECTORY_KEYS if synthetic else _REAL_RAW_TRAJECTORY_KEYS,
    )
    identity_key = "request_checksum" if synthetic else "job_checksum"
    if (
        payload[identity_key] != evidence.job_checksum
        or payload["evaluation_policy_checksum"] != evidence.evaluation_policy_checksum
    ):
        msg = "Raw trajectory job/request or evaluation-policy identity differs from production evidence."
        raise ValueError(msg)
    count = require_int(payload["trajectory_count"], "trajectory_count", minimum=1)
    fidelities = tuple(
        require_float(value, "trajectory_fidelity", minimum=0.0, maximum=1.0)
        for value in _strict_tuple(payload["trajectory_fidelities"], "trajectory_fidelities")
    )
    if len(fidelities) != count:
        msg = "Raw trajectory count differs from its complete fidelity vector."
        raise ValueError(msg)
    require_int(payload["evaluation_seed"], "evaluation_seed")

    if synthetic:
        if (
            payload["data_role"] != "confirmatory"
            or payload["seed_domain"] != "confirmatory_test"
            or payload["evaluation_seed"] != evidence.derived_metrics.get("evaluation_seed")
            or payload["data_role"] != evidence.derived_metrics.get("evaluation_data_role")
            or payload["seed_domain"] != evidence.derived_metrics.get("evaluation_seed_domain")
            or payload["synthetic_fixture_checksum"] != evidence.derived_metrics.get("synthetic_fixture_checksum")
            or require_int(evidence.derived_metrics.get("trajectory_count"), "derived trajectory_count") != count
            or not math.isclose(
                require_float(
                    evidence.derived_metrics.get("noisy_fidelity"),
                    "noisy_fidelity",
                    minimum=0.0,
                    maximum=1.0,
                ),
                float(np.mean(fidelities)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            msg = "Synthetic raw trajectories differ from their fixture-bound evidence."
            raise ValueError(msg)
        return _RawTrajectoryCustody(count, fidelities, None)

    fixed_checksum = require_checksum(payload["fixed_map_ensemble_checksum"], "fixed_map_ensemble_checksum")
    fresh_matches = tuple(
        ensemble
        for path, ensemble in ensembles_by_path.items()
        if path.endswith("/evaluation/fresh_fixed_map_ensemble.json") and ensemble.content_checksum == fixed_checksum
    )
    if len(fresh_matches) != 1:
        msg = "Fresh raw sidecar does not identify one exact manifest fixed-map ensemble."
        raise ValueError(msg)
    fresh = fresh_matches[0]
    role_by_data_role = {
        "development": "pilot_evaluation",
        "secondary_benchmark": "pilot_evaluation",
        "screening_selection": "screening_selection",
        "confirmatory": "confirmatory_test",
        "checkpoint_validation": "checkpoint_validation",
    }
    data_role = require_slug(payload["data_role"], "data_role")
    expected_role = role_by_data_role.get(data_role)
    expected_seed = require_int(payload["evaluation_seed"], "evaluation_seed")
    evaluation_checksum = require_checksum(
        payload["evaluation_configuration_checksum"],
        "evaluation_configuration_checksum",
    )
    if (
        expected_role is None
        or payload["seed_domain"] != expected_role
        or fresh.role != expected_role
        or fresh.resolved_seed != expected_seed
        or fresh.trajectory_count != count
        or fresh.stage_configuration_checksum != evaluation_checksum
        or evaluation_checksum != evidence.derived_metrics.get("evaluation_configuration_checksum")
        or expected_seed != evidence.derived_metrics.get("evaluation_seed")
        or data_role != evidence.derived_metrics.get("evaluation_data_role")
        or payload["seed_domain"] != evidence.derived_metrics.get("evaluation_seed_domain")
        or require_int(payload["sampled_nonidentity_events"], "sampled_nonidentity_events")
        != fresh.nonidentity_event_count
        or resource.circuit_checksum != fresh.circuit_checksum
        or resource.circuit_gate_count != fresh.gate_count
    ):
        msg = "Fresh manifest ensemble role, seed, circuit, or trajectory aliases differ from its raw sidecar."
        raise ValueError(msg)
    selected_parameters = _selected_snapshot_parameters(snapshot)
    if selected_parameters is not None:
        expected_evaluation_checksum = canonical_checksum({
            "job_checksum": evidence.job_checksum,
            "evaluation_policy_checksum": evidence.evaluation_policy_checksum,
            "circuit_checksum": fresh.circuit_checksum,
            "parameter_checksum": canonical_checksum({
                "parameters": [float(value) for value in selected_parameters],
            }),
        })
        if evaluation_checksum != expected_evaluation_checksum:
            msg = "Fresh evaluation configuration differs from the validation-selected snapshot parameters."
            raise ValueError(msg)
    _validate_raw_metrics(evidence=evidence, fidelities=fidelities, fresh_ensemble=fresh)
    return _RawTrajectoryCustody(count, fidelities, fresh)


def _validate_normalized_work(
    *,
    evidence: ProductionNumericalEvidence,
    resource: _ResourceCustody,
    raw: _RawTrajectoryCustody | None,
    diagnostic_member_count: int,
) -> None:
    """Recompute complete successful normalized work from typed evidence."""
    if evidence.status != "success":
        return
    if evidence.artifact_kind == "synthetic_confirmation":
        expected = 0.0
    elif (
        evidence.artifact_kind == "operator_growth"
        and evidence.derived_metrics.get("execution_preset") == "training-smoke"
    ):
        expected = float(
            require_int(evidence.derived_metrics.get("training_trajectory_count"), "training_trajectory_count")
        )
    else:
        if raw is None:
            msg = "Successful production work accounting requires raw trajectory evidence."
            raise ValueError(msg)
        if evidence.derived_metrics.get("execution_preset") == "training-smoke":
            expected = float(
                require_int(
                    evidence.derived_metrics.get("training_trajectory_count"),
                    "training_trajectory_count",
                )
            )
        else:
            expected = require_float(
                evidence.derived_metrics.get("total_normalized_training_work"),
                "normalized training work",
                minimum=0.0,
            )
        expected += raw.trajectory_count + diagnostic_member_count
    if not math.isclose(resource.normalized_work, expected, rel_tol=0.0, abs_tol=1e-12):
        msg = "Runtime normalized work differs from training, fresh-evaluation, and diagnostic receipts."
        raise ValueError(msg)


def _provider_checksum(provider: object) -> str:
    """Return the stable checksum required from a standard-noise provider."""
    return require_checksum(getattr(provider, "content_checksum", None), "provider.content_checksum")


def _operator_circuit_binding(
    pool: OperatorPoolSpec,
    selected_operator_ids: Sequence[str],
) -> NoisyKrotovCircuitBinding:
    """Materialize one exact ordered operator prefix as a production circuit."""
    if not isinstance(pool, OperatorPoolSpec):
        msg = "pool must be an OperatorPoolSpec."
        raise TypeError(msg)
    selected_ids = tuple(require_slug(item, "selected_operator_id") for item in selected_operator_ids)
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        msg = "Operator production circuits require one nonempty unique selected prefix."
        raise ValueError(msg)
    by_id = {operator.operator_id: operator for operator in pool.operators}
    try:
        operators = tuple(by_id[operator_id] for operator_id in selected_ids)
    except KeyError as error:
        msg = "Selected operator prefix contains an identity outside its exact pool."
        raise ValueError(msg) from error
    topology_suffix = canonical_checksum({"selected_operator_ids": list(selected_ids)}).removeprefix("sha256:")[:16]
    return NoisyKrotovCircuitBinding(
        materialize_operator_growth_circuit(pool.num_qubits, operators),
        f"operator_growth_q6_prefix_{topology_suffix}",
    )


def _operator_candidate_metadata(
    pool: OperatorPoolSpec,
    spec: OperatorGrowthSpec,
    selected_operator_ids: tuple[str, ...],
) -> tuple[tuple[int, object, bool], ...]:
    """Return the complete remaining ordered pool with native-cap feasibility."""
    if not isinstance(pool, OperatorPoolSpec) or not isinstance(spec, OperatorGrowthSpec):
        msg = "Operator selection requires exact pool and growth-spec records."
        raise TypeError(msg)
    selected = set(selected_operator_ids)
    by_id = {operator.operator_id: operator for operator in pool.operators}
    if len(selected) != len(selected_operator_ids) or any(operator_id not in by_id for operator_id in selected):
        msg = "Selected operator prefix is not a unique subset of its exact pool."
        raise ValueError(msg)
    edge_counts = [0] * (pool.num_qubits - 1)
    for operator_id in selected_operator_ids:
        operator = by_id[operator_id]
        if len(operator.sites) == 2:
            edge_counts[operator.sites[0]] += operator.native_two_qubit_gates
    result: list[tuple[int, object, bool]] = []
    for pool_index, operator in enumerate(pool.operators):
        if operator.operator_id in selected:
            continue
        feasible = True
        if len(operator.sites) == 2 and spec.native_two_qubit_cap_per_edge is not None:
            edge = operator.sites[0]
            feasible = edge_counts[edge] + operator.native_two_qubit_gates <= spec.native_two_qubit_cap_per_edge
        result.append((pool_index, operator, feasible))
    return tuple(result)


class _OperatorStructuralSelection:
    """Complete deterministic pool scan using the segmented objective seam."""

    def __init__(
        self,
        pool: OperatorPoolSpec,
        spec: OperatorGrowthSpec,
        selection_sink: Callable[[int, tuple[str, ...]], None],
    ) -> None:
        """Bind the exact pool/spec and selected-prefix custody callback."""
        self.pool = pool
        self.spec = spec
        self.selection_sink = selection_sink

    def quote_normalized_work(self, request: OperatorGrowthSelectionRequest) -> float:
        """Quote all feasible parameter-shift pairs and the appended-zero baseline."""
        metadata = _operator_candidate_metadata(self.pool, self.spec, request.selected_operator_ids)
        feasible_count = sum(feasible for _index, _operator, feasible in metadata)
        if not metadata or not feasible_count:
            msg = "Operator structural selection has no feasible remaining candidate."
            raise ValueError(msg)
        return float((2 * feasible_count + 1) * request.policy.trajectory_count)

    def __call__(
        self,
        request: OperatorGrowthSelectionRequest,
        objective_executor: Callable[
            [OperatorGrowthSegmentedObjectiveRequest],
            OperatorGrowthSegmentedObjectiveResult,
        ],
    ) -> OperatorGrowthSelectionResult:
        """Evaluate every candidate in pool order and retain the deterministic winner."""
        metadata = _operator_candidate_metadata(self.pool, self.spec, request.selected_operator_ids)
        candidates: list[CandidateGradient] = []
        feasible_candidates: list[CandidateGradient] = []
        evidence: list[OperatorGrowthSegmentedObjectiveEvidence] = []
        for pool_index, candidate, feasible in metadata:
            operator_id = require_slug(getattr(candidate, "operator_id", None), "candidate.operator_id")
            native_increment = require_int(
                getattr(candidate, "native_two_qubit_gates", None),
                "candidate.native_two_qubit_gates",
            )
            if not feasible:
                candidates.append(
                    CandidateGradient(
                        operator_id=operator_id,
                        pool_index=pool_index,
                        gradient=None,
                        absolute_gradient=None,
                        native_two_qubit_increment=native_increment,
                        native_cap_feasible=False,
                    )
                )
                continue
            selected_ids = (*request.selected_operator_ids, operator_id)
            plus_request = OperatorGrowthSegmentedObjectiveRequest(
                program_checksum=request.program_checksum,
                structural_state_checksum=request.content_checksum,
                selected_operator_ids=selected_ids,
                prefix_index=request.prefix_index,
                global_update=request.global_update_start,
                local_update=0,
                evaluation_stage="structural_selection",
                evaluation_kind="gradient_plus",
                parameter_index=request.prefix_index,
                parameters=(*request.parameters, math.pi / 2.0),
                policy=request.policy,
            )
            minus_request = OperatorGrowthSegmentedObjectiveRequest(
                program_checksum=request.program_checksum,
                structural_state_checksum=request.content_checksum,
                selected_operator_ids=selected_ids,
                prefix_index=request.prefix_index,
                global_update=request.global_update_start,
                local_update=0,
                evaluation_stage="structural_selection",
                evaluation_kind="gradient_minus",
                parameter_index=request.prefix_index,
                parameters=(*request.parameters, -math.pi / 2.0),
                policy=request.policy,
            )
            plus = OperatorGrowthSegmentedObjectiveEvidence(plus_request, objective_executor(plus_request))
            minus = OperatorGrowthSegmentedObjectiveEvidence(minus_request, objective_executor(minus_request))
            evidence.extend((plus, minus))
            gradient = 0.5 * (plus.result.objective - minus.result.objective)
            record = CandidateGradient(
                operator_id=operator_id,
                pool_index=pool_index,
                gradient=gradient,
                absolute_gradient=abs(gradient),
                native_two_qubit_increment=native_increment,
                native_cap_feasible=True,
            )
            candidates.append(record)
            feasible_candidates.append(record)
        chosen = max(feasible_candidates, key=lambda item: cast("float", item.absolute_gradient))
        selected_ids = (*request.selected_operator_ids, chosen.operator_id)
        baseline_request = OperatorGrowthSegmentedObjectiveRequest(
            program_checksum=request.program_checksum,
            structural_state_checksum=request.content_checksum,
            selected_operator_ids=selected_ids,
            prefix_index=request.prefix_index,
            global_update=request.global_update_start,
            local_update=0,
            evaluation_stage="structural_selection",
            evaluation_kind="post_update",
            parameter_index=request.prefix_index + 1,
            parameters=(*request.parameters, 0.0),
            policy=request.policy,
        )
        baseline = OperatorGrowthSegmentedObjectiveEvidence(
            baseline_request,
            objective_executor(baseline_request),
        )
        evidence.append(baseline)
        self.selection_sink(request.prefix_index, selected_ids)
        return OperatorGrowthSelectionResult(
            request_checksum=request.content_checksum,
            candidate_gradients=tuple(candidates),
            selected_operator_id=chosen.operator_id,
            selected_gradient=cast("float", chosen.gradient),
            objective_before_reoptimization=baseline.result.objective,
            objective_evidence=tuple(evidence),
            normalized_work=self.quote_normalized_work(request),
        )


class _OperatorScheduledNumerics:
    """Real fixed-CRN objective and role-separated validation callbacks."""

    def __init__(
        self,
        *,
        store: ProductionAttemptStore,
        execution_spec: OperatorGrowthExecutionSpec,
        program_checksum: str,
        target: MaterializedTarget,
    ) -> None:
        """Bind exact operator policy, target, scheduled program, and custody store."""
        self.store = store
        self.execution_spec = execution_spec
        self.program_checksum = require_checksum(program_checksum, "program_checksum")
        self.target = target
        self.map_blobs: list[ArtifactBlobRef] = []
        self.map_evidence_blobs: list[ArtifactBlobRef] = []
        self.map_evidence_records: list[ScheduledMapEvidence] = []
        self._request_index = 0
        self._selected_ids_by_prefix: dict[int, tuple[str, ...]] = {}

    @property
    def truncation(self) -> KrotovTruncation:
        """Exact no-truncation operator-growth policy."""
        return KrotovTruncation()

    def register_selected_prefix(self, prefix_index: int, selected_operator_ids: tuple[str, ...]) -> None:
        """Retain the exact structural choice needed by its later validation callback."""
        prefix = require_int(prefix_index, "prefix_index")
        selected = tuple(require_slug(item, "selected_operator_id") for item in selected_operator_ids)
        if len(selected) != prefix + 1:
            msg = "Registered operator prefix has the wrong structural width."
            raise ValueError(msg)
        existing = self._selected_ids_by_prefix.get(prefix)
        if existing is not None and existing != selected:
            msg = "A structural prefix cannot be rebound to different operator identities."
            raise ValueError(msg)
        if prefix and self._selected_ids_by_prefix.get(prefix - 1) != selected[:-1]:
            msg = "Operator prefix registration does not extend its predecessor."
            raise ValueError(msg)
        self._selected_ids_by_prefix[prefix] = selected

    def _provider(self, strength_scale: float) -> GateNoiseProvider:
        """Resolve and verify the exact primary-noise provider."""
        noise = self.execution_spec.training_noise_condition
        noise_id = cast("str", noise["noise_id"])
        expected_scale = cast("float", noise["strength_scale"])
        if not math.isclose(strength_scale, expected_scale, rel_tol=0.0, abs_tol=0.0):
            msg = "Scheduled operator objective changed the frozen primary-noise strength."
            raise ValueError(msg)
        provider = create_scaled_standard_noise_provider(noise_id, strength_scale)
        if _provider_checksum(provider) != self.execution_spec.provider_identity["content_checksum"]:
            msg = "Resolved operator-growth provider differs from its execution specification."
            raise ValueError(msg)
        return cast("GateNoiseProvider", provider)

    def _evaluate_members(
        self,
        *,
        request_checksum: str,
        policy_checksum: str,
        membership: TrajectoryEnsembleMembership,
        circuit_binding: NoisyKrotovCircuitBinding,
        parameters: tuple[float, ...],
        provider: GateNoiseProvider,
        map_role: KrotovMapRole,
        prefix_index: int,
        global_update: int,
        stage_id: str,
    ) -> tuple[float, float, tuple[float, ...], ArtifactBlobRef]:
        """Evaluate each explicit member once and persist maps from that same pass."""
        theta = np.asarray(parameters, dtype=np.float64)
        noise = self.execution_spec.training_noise_condition
        trajectory_maps: list[list[KrotovNoiseMap]] = []
        trajectory_fidelities: list[float] = []
        for member_seed in membership.member_seeds:
            metrics = noisy_state_preparation_metrics_with_maps(
                circuit_binding.circuit,
                theta,
                self.target.state_vector_copy(),
                None,
                KrotovTJMOptions(
                    num_trajectories=1,
                    random_seed=member_seed,
                    dt=cast("float", noise["tjm_dt"]),
                    noisy_gate_indices=circuit_binding.noisy_gate_indices,
                    trajectory_update="independent",
                    differentiate_jump_normalization=False,
                    use_crn=False,
                ),
                initial_state=MPS(circuit_binding.circuit.num_qubits),
                truncation=self.truncation,
                iteration=0,
                noise_provider=provider,
            )
            if (
                len(metrics.trajectory_fidelities) != 1
                or len(metrics.realized_noise_maps) != 1
                or len(metrics.realized_noise_maps[0]) != len(circuit_binding.circuit.gates)
            ):
                msg = "Single-member operator evaluation returned non-singleton evidence."
                raise ValueError(msg)
            trajectory_fidelities.append(metrics.trajectory_fidelities[0])
            trajectory_maps.append(list(metrics.realized_noise_maps[0]))
        stage_checksum = canonical_checksum({
            "execution_spec_checksum": self.execution_spec.content_checksum,
            "program_checksum": self.program_checksum,
            "request_checksum": request_checksum,
            "policy_checksum": policy_checksum,
            "stage_id": stage_id,
        })
        ensemble = KrotovFixedMapEnsemble(
            role=map_role,
            resolved_seed=membership.map_seed,
            stage_index=prefix_index,
            stage_id=stage_id,
            stage_configuration_checksum=stage_checksum,
            circuit_checksum=circuit_binding.content_checksum,
            provider_checksum=_provider_checksum(provider),
            ensemble_index=self._request_index,
            refresh_index=0,
            global_iteration_start=global_update,
            trajectory_maps=trajectory_maps,
        )
        ref = self.store.write_blob(
            f"maps/request_{self._request_index:08d}_component_000.json",
            f"{ensemble.to_json()}\n".encode(),
            role="fixed_map_ensemble",
            logical_checksum=ensemble.content_checksum,
        )
        self.map_blobs.append(ref)
        mean_fidelity = float(np.mean(trajectory_fidelities))
        return 1.0 - mean_fidelity, mean_fidelity, tuple(trajectory_fidelities), ref

    def _persist_map_evidence(
        self,
        *,
        request_checksum: str,
        policy_checksum: str,
        membership: TrajectoryEnsembleMembership,
        map_role: KrotovMapRole,
        circuit_binding: NoisyKrotovCircuitBinding,
        provider: GateNoiseProvider,
        ensemble_ref: ArtifactBlobRef,
        numerical_result_checksum: str,
        trajectory_fidelities: Sequence[float],
    ) -> None:
        """Close one exact request/result pair over raw maps and fidelities."""
        evidence = ScheduledMapEvidence(
            request_checksum=request_checksum,
            policy_checksum=policy_checksum,
            membership_checksum=membership.content_checksum,
            component_membership_checksums=(),
            member_seeds=membership.member_seeds,
            component_member_seeds=(membership.member_seeds,),
            map_role=map_role,
            resolved_seeds=(membership.map_seed,),
            circuit_checksum=circuit_binding.content_checksum,
            provider_checksums=(_provider_checksum(provider),),
            ensemble_refs=(ensemble_ref,),
            numerical_result_checksum=numerical_result_checksum,
            trajectory_fidelities=tuple(float(value) for value in trajectory_fidelities),
        )
        evidence_ref = self.store.write_json_blob(
            f"map_evidence/request_{self._request_index:08d}.json",
            evidence.to_dict(),
            role="scheduled_map_evidence",
        )
        self.map_evidence_blobs.append(evidence_ref)
        self.map_evidence_records.append(evidence)
        self._request_index += 1

    def objective(
        self,
        request: OperatorGrowthSegmentedObjectiveRequest,
    ) -> OperatorGrowthSegmentedObjectiveResult:
        """Evaluate one exact noisy segmented objective and persist all raw evidence."""
        if request.program_checksum != self.program_checksum:
            msg = "Segmented objective belongs to a different scheduled program."
            raise ValueError(msg)
        membership = request.policy.training_membership
        if membership is None:
            msg = "Segmented noisy objective requires explicit fixed-CRN membership."
            raise ValueError(msg)
        circuit_binding = _operator_circuit_binding(self.execution_spec.pool, request.selected_operator_ids)
        provider = self._provider(request.policy.noise_strength_scale)
        loss, mean_fidelity, trajectory_fidelities, ensemble_ref = self._evaluate_members(
            request_checksum=request.content_checksum,
            policy_checksum=request.policy.content_checksum,
            membership=membership,
            circuit_binding=circuit_binding,
            parameters=request.parameters,
            provider=provider,
            map_role="training_trajectory",
            prefix_index=request.prefix_index,
            global_update=request.global_update,
            stage_id=f"operator_growth_{request.evaluation_stage}",
        )
        if len(trajectory_fidelities) != membership.trajectory_count or not math.isclose(
            float(mean_fidelity),
            float(np.mean(trajectory_fidelities)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            msg = "Operator objective returned incomplete or inconsistent trajectory fidelities."
            raise ValueError(msg)
        result = OperatorGrowthSegmentedObjectiveResult.for_request(request, float(loss))
        self._persist_map_evidence(
            request_checksum=request.content_checksum,
            policy_checksum=request.policy.content_checksum,
            membership=membership,
            map_role="training_trajectory",
            circuit_binding=circuit_binding,
            provider=provider,
            ensemble_ref=ensemble_ref,
            numerical_result_checksum=result.content_checksum,
            trajectory_fidelities=trajectory_fidelities,
        )
        return result

    def validate(self, request: ScheduledValidationRequest) -> ScheduledValidationResult:
        """Evaluate one completed prefix only on its separate 256-member maps."""
        if request.program_checksum != self.program_checksum:
            msg = "Prefix validation belongs to a different scheduled program."
            raise ValueError(msg)
        prefix_index = (request.update + 1) // 100 - 1
        selected_ids = self._selected_ids_by_prefix.get(prefix_index)
        if selected_ids is None:
            msg = "Prefix validation has no completed structural selection."
            raise ValueError(msg)
        circuit_binding = _operator_circuit_binding(self.execution_spec.pool, selected_ids)
        provider = self._provider(cast("float", self.execution_spec.training_noise_condition["strength_scale"]))
        _loss, mean_fidelity, trajectory_fidelities, ensemble_ref = self._evaluate_members(
            request_checksum=request.content_checksum,
            policy_checksum=request.program_checksum,
            membership=request.membership,
            circuit_binding=circuit_binding,
            parameters=request.parameter_artifact.parameters,
            provider=provider,
            map_role="checkpoint_validation",
            prefix_index=prefix_index,
            global_update=request.update,
            stage_id="operator_growth_prefix_validation",
        )
        if len(trajectory_fidelities) != request.membership.trajectory_count or not math.isclose(
            float(mean_fidelity),
            float(np.mean(trajectory_fidelities)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            msg = "Prefix validation returned incomplete or inconsistent trajectory fidelities."
            raise ValueError(msg)
        result = ScheduledValidationResult.for_request(request, float(mean_fidelity))
        self._persist_map_evidence(
            request_checksum=request.content_checksum,
            policy_checksum=request.program_checksum,
            membership=request.membership,
            map_role="checkpoint_validation",
            circuit_binding=circuit_binding,
            provider=provider,
            ensemble_ref=ensemble_ref,
            numerical_result_checksum=result.content_checksum,
            trajectory_fidelities=trajectory_fidelities,
        )
        return result


@dataclass(frozen=True, slots=True)
class _PipelineNumericalLink:
    """Expected map-custody aliases for one pipeline numerical callback."""

    request_checksum: str
    policy_checksum: str
    membership_checksum: str
    component_membership_checksums: tuple[str, ...]
    member_seeds: tuple[int, ...]
    component_member_seeds: tuple[tuple[int, ...], ...]
    map_role: KrotovMapRole
    resolved_seeds: tuple[int, ...]
    circuit_checksum: str
    provider_checksums: tuple[str, ...]
    circuit_gate_count: int


def _pipeline_objective_checksum(target_identity: Mapping[str, object]) -> str:
    """Reconstruct the target-bound pure-state objective from sealed identity."""
    identity = dict(require_mapping(target_identity, "target_identity"))
    qubit_count = require_int(identity.get("qubit_count"), "target_identity.qubit_count", minimum=1)
    vector_checksum = require_checksum(identity.get("vector_checksum"), "target_identity.vector_checksum")
    target_state_checksum = canonical_checksum({
        "amplitude_count": 2**qubit_count,
        "data_sha256": vector_checksum.removeprefix("sha256:"),
        "dtype": "<c16",
    })
    return NoisyKrotovObjectiveBinding(
        target_state_checksum=target_state_checksum,
        initial_state_policy="computational_zero_v1",
        initial_state_checksum=noisy_krotov_computational_zero_state_checksum(qubit_count),
        materialized_target_identity=identity,
    ).objective_checksum


def _pipeline_training_link(
    request: ScheduledTrainingGradientRequest | ScheduledTrainingObjectiveRequest,
    *,
    circuit_checksum: str,
    circuit_gate_count: int,
) -> _PipelineNumericalLink:
    """Project one reconstructed training request onto its exact map aliases."""
    membership = request.policy.training_membership
    if membership is None:
        msg = "A persisted noisy pipeline request requires exact aggregate membership."
        raise ValueError(msg)
    components = request.policy.component_memberships
    if not components:
        msg = "A persisted noisy pipeline request requires component-local membership."
        raise ValueError(msg)
    providers = tuple(
        _provider_checksum(
            create_scaled_standard_noise_provider(component.noise_id, request.policy.noise_strength_scale)
        )
        for component in components
    )
    return _PipelineNumericalLink(
        request_checksum=request.content_checksum,
        policy_checksum=request.policy.content_checksum,
        membership_checksum=membership.content_checksum,
        component_membership_checksums=tuple(component.content_checksum for component in components),
        member_seeds=membership.member_seeds,
        component_member_seeds=tuple(component.member_seeds for component in components),
        map_role="training_trajectory",
        resolved_seeds=(membership.map_seed,) * len(components),
        circuit_checksum=circuit_checksum,
        provider_checksums=providers,
        circuit_gate_count=circuit_gate_count,
    )


def _pipeline_validation_link(
    request: ScheduledValidationRequest,
    *,
    circuit_checksum: str,
    circuit_gate_count: int,
) -> _PipelineNumericalLink:
    """Project one snapshot validation request onto its fixed primary-noise map."""
    provider = create_scaled_standard_noise_provider("depolarizing_1s_all", 1.0)
    membership = request.membership
    return _PipelineNumericalLink(
        request_checksum=request.content_checksum,
        policy_checksum=request.program_checksum,
        membership_checksum=membership.content_checksum,
        component_membership_checksums=(),
        member_seeds=membership.member_seeds,
        component_member_seeds=(membership.member_seeds,),
        map_role="checkpoint_validation",
        resolved_seeds=(membership.map_seed,),
        circuit_checksum=circuit_checksum,
        provider_checksums=(_provider_checksum(provider),),
        circuit_gate_count=circuit_gate_count,
    )


def _pipeline_snapshot_numerical_links(
    snapshot: ScheduledExecutionSnapshot,
    *,
    target_identity: Mapping[str, object],
    circuit_checksum: str,
    circuit_gate_count: int,
) -> tuple[_PipelineNumericalLink, ...]:
    """Replay every map-producing optimizer and validation request in exact order."""
    if not isinstance(snapshot, ScheduledExecutionSnapshot):
        msg = "snapshot must be a ScheduledExecutionSnapshot."
        raise TypeError(msg)
    checksum = require_checksum(circuit_checksum, "circuit_checksum")
    gate_count = require_int(circuit_gate_count, "circuit_gate_count", minimum=1)
    objective_checksum = _pipeline_objective_checksum(target_identity)
    links: list[_PipelineNumericalLink] = []
    for state in snapshot.states:
        for receipt in state.receipts:
            update_request = receipt.request
            policy = update_request.policy
            sampled = policy.trajectory_count > 0 and not np.isclose(policy.noise_strength_scale, 0.0)
            payload = update_request.optimizer_payload
            if sampled and isinstance(payload, KrotovOptimizerPayload):
                request = ScheduledTrainingGradientRequest(
                    update_request.content_checksum,
                    objective_checksum,
                    policy,
                    payload.parameters,
                )
                links.append(
                    _pipeline_training_link(
                        request,
                        circuit_checksum=checksum,
                        circuit_gate_count=gate_count,
                    )
                )
            elif sampled and isinstance(payload, AdamOptimizerPayload):
                qubit_count = require_int(target_identity.get("qubit_count"), "target_identity.qubit_count", minimum=1)
                positive_count = 3 * qubit_count
                if len(payload.parameters) < positive_count:
                    msg = "Adam snapshot parameter width is smaller than the BMPD rotation prefix."
                    raise ValueError(msg)
                scales = (1.0,) * positive_count + (-1.0,) * (len(payload.parameters) - positive_count)
                for index, scale in enumerate(scales):
                    shift = math.pi / (2.0 * scale)
                    plus = list(payload.parameters)
                    minus = list(payload.parameters)
                    plus[index] += shift
                    minus[index] -= shift
                    pair_seed = derive_role_seed(
                        update_request.seed_bundle.training_trajectory_seed,
                        "training_trajectory",
                        purpose="scheduled_objective_pair",
                        stream_index=policy.start_index,
                        epoch=policy.update,
                        member_index=index,
                    )
                    for parameters, evaluation_kind in (
                        (tuple(plus), "gradient_plus"),
                        (tuple(minus), "gradient_minus"),
                    ):
                        request = ScheduledTrainingObjectiveRequest(
                            update_request.content_checksum,
                            objective_checksum,
                            policy,
                            parameters,
                            evaluation_kind,
                            index,
                            pair_seed,
                        )
                        links.append(
                            _pipeline_training_link(
                                request,
                                circuit_checksum=checksum,
                                circuit_gate_count=gate_count,
                            )
                        )
            elif sampled and isinstance(payload, SPSAOptimizerPayload):
                iteration = payload.completed_updates + 1
                seed_checksum = canonical_checksum({
                    "derivation_version": "yaqs.state_preparation.phase2.wp22_spsa_perturbation.v1",
                    "program_checksum": update_request.program_checksum,
                    "optimizer_ordering_seed": payload.initialization.seed_bundle.optimizer_ordering_seed,
                    "iteration": iteration,
                })
                perturbation_seed = int(seed_checksum.removeprefix("sha256:")[:16], 16)
                rng = np.random.Generator(np.random.PCG64(perturbation_seed))
                perturbation = tuple(float(2 * item - 1) for item in rng.integers(0, 2, size=len(payload.parameters)))
                _learning_rate, scale = payload.config.gains(iteration)
                plus = tuple(
                    parameter + scale * delta for parameter, delta in zip(payload.parameters, perturbation, strict=True)
                )
                minus = tuple(
                    parameter - scale * delta for parameter, delta in zip(payload.parameters, perturbation, strict=True)
                )
                for parameters, evaluation_kind in (
                    (plus, "gradient_plus"),
                    (minus, "gradient_minus"),
                ):
                    request = ScheduledTrainingObjectiveRequest(
                        update_request.content_checksum,
                        objective_checksum,
                        policy,
                        parameters,
                        evaluation_kind,
                        0,
                        perturbation_seed,
                    )
                    links.append(
                        _pipeline_training_link(
                            request,
                            circuit_checksum=checksum,
                            circuit_gate_count=gate_count,
                        )
                    )
            elif sampled:
                msg = "Pipeline snapshot contains an unsupported map-producing optimizer payload."
                raise TypeError(msg)
            if receipt.validation_request is not None:
                links.append(
                    _pipeline_validation_link(
                        receipt.validation_request,
                        circuit_checksum=checksum,
                        circuit_gate_count=gate_count,
                    )
                )
    return tuple(links)


def _validate_pipeline_numerical_link(
    map_evidence: ScheduledMapEvidence,
    expected_link: _PipelineNumericalLink,
    ensembles: tuple[KrotovFixedMapEnsemble, ...],
) -> None:
    """Compare one persisted pipeline map record to its reconstructed callback."""
    actual = (
        map_evidence.request_checksum,
        map_evidence.policy_checksum,
        map_evidence.membership_checksum,
        map_evidence.component_membership_checksums,
        map_evidence.member_seeds,
        map_evidence.component_member_seeds,
        map_evidence.map_role,
        map_evidence.resolved_seeds,
        map_evidence.circuit_checksum,
        map_evidence.provider_checksums,
    )
    expected = (
        expected_link.request_checksum,
        expected_link.policy_checksum,
        expected_link.membership_checksum,
        expected_link.component_membership_checksums,
        expected_link.member_seeds,
        expected_link.component_member_seeds,
        expected_link.map_role,
        expected_link.resolved_seeds,
        expected_link.circuit_checksum,
        expected_link.provider_checksums,
    )
    if (
        actual != expected
        or map_evidence.numerical_result_checksum is not None
        or map_evidence.trajectory_fidelities
        or len(ensembles) != len(expected_link.component_member_seeds)
        or any(ensemble.gate_count != expected_link.circuit_gate_count for ensemble in ensembles)
    ):
        msg = "Pipeline scheduled map evidence differs from its exact snapshot callback."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class _OperatorNumericalLink:
    """Expected one-to-one snapshot link for an operator numerical callback."""

    request_checksum: str
    result_checksum: str
    policy_checksum: str
    membership_checksum: str
    member_seeds: tuple[int, ...]
    map_role: KrotovMapRole
    circuit_checksum: str
    circuit_gate_count: int
    expected_mean_fidelity: float


def _validate_operator_numerical_link(
    map_evidence: ScheduledMapEvidence,
    expected_link: _OperatorNumericalLink,
    ensembles: tuple[KrotovFixedMapEnsemble, ...],
) -> None:
    """Validate one persisted operator callback against its snapshot identity."""
    actual_link = (
        map_evidence.request_checksum,
        map_evidence.numerical_result_checksum,
        map_evidence.policy_checksum,
        map_evidence.membership_checksum,
        map_evidence.member_seeds,
        map_evidence.map_role,
        map_evidence.circuit_checksum,
    )
    expected = (
        expected_link.request_checksum,
        expected_link.result_checksum,
        expected_link.policy_checksum,
        expected_link.membership_checksum,
        expected_link.member_seeds,
        expected_link.map_role,
        expected_link.circuit_checksum,
    )
    if (
        actual_link != expected
        or len(ensembles) != len(map_evidence.ensemble_refs)
        or not math.isclose(
            float(np.mean(map_evidence.trajectory_fidelities)),
            expected_link.expected_mean_fidelity,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or any(ensemble.gate_count != expected_link.circuit_gate_count for ensemble in ensembles)
    ):
        msg = "Operator scheduled map/fidelity evidence differs from its exact snapshot callback."
        raise ValueError(msg)


def _operator_snapshot_numerical_links(
    snapshot: OperatorGrowthSegmentedSnapshot,
) -> tuple[_OperatorNumericalLink, ...]:
    """Replay exact callback order from one authoritative segmented snapshot."""
    if not isinstance(snapshot, OperatorGrowthSegmentedSnapshot):
        msg = "snapshot must be an OperatorGrowthSegmentedSnapshot."
        raise TypeError(msg)
    links: list[_OperatorNumericalLink] = []

    def append_objective(evidence: OperatorGrowthSegmentedObjectiveEvidence) -> None:
        request = evidence.request
        membership = request.policy.training_membership
        if membership is None:
            msg = "Segmented snapshot objective omitted its explicit membership."
            raise ValueError(msg)
        circuit = _operator_circuit_binding(snapshot.pool, request.selected_operator_ids)
        links.append(
            _OperatorNumericalLink(
                request.content_checksum,
                evidence.result.content_checksum,
                request.policy.content_checksum,
                membership.content_checksum,
                membership.member_seeds,
                "training_trajectory",
                circuit.content_checksum,
                len(circuit.circuit.gates),
                require_float(
                    1.0 - evidence.result.objective,
                    "objective mean fidelity",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        )

    for prefix_index, transition in enumerate(snapshot.transitions):
        for evidence in transition.result.objective_evidence:
            append_objective(evidence)
        start = prefix_index * 100
        stop = min((prefix_index + 1) * 100, len(snapshot.receipts))
        for receipt in snapshot.receipts[start:stop]:
            for evidence in receipt.objective_evidence:
                append_objective(evidence)
            validation = receipt.prefix_validation
            if validation is None:
                continue
            request = validation.request
            circuit = _operator_circuit_binding(snapshot.pool, transition.selected_operator_ids)
            links.append(
                _OperatorNumericalLink(
                    request.content_checksum,
                    validation.result.content_checksum,
                    request.program_checksum,
                    request.membership.content_checksum,
                    request.membership.member_seeds,
                    "checkpoint_validation",
                    circuit.content_checksum,
                    len(circuit.circuit.gates),
                    validation.result.score,
                )
            )
    return tuple(links)


def _decode_schedule_snapshot(
    payload: bytes,
) -> ScheduledExecutionSnapshot | OperatorGrowthSegmentedSnapshot:
    """Decode either supported authoritative schedule-snapshot schema."""
    document = load_canonical_json_object(payload.decode())
    if document.get("schema_version") == OPERATOR_GROWTH_SEGMENTED_SNAPSHOT_SCHEMA_VERSION:
        return OperatorGrowthSegmentedSnapshot.from_dict(document)
    return ScheduledExecutionSnapshot.from_json(payload.decode())


def _pipeline_config(resolved: ResolvedProductionJob) -> TrainingPipelineConfig:
    """Resolve a production pipeline template for the exact job target cell."""
    payload = resolved.executable_binding.binding.implementation_artifact.implementation_payload
    if not isinstance(payload, TrainingPipelineTemplate):
        msg = "Production pipeline execution requires a TrainingPipelineTemplate payload."
        raise TypeError(msg)
    job = resolved.job
    return payload.resolve(
        target_namespace="phase2",
        target_manifest=resolved.target_manifest,
        target_instance_id=job.target_instance_id,
        target_population_manifest_checksum=job.target_manifest_checksum,
        target_instance_spec_checksum=job.target_spec_checksum,
        target_family_id=job.family_id,
        target_stratum_id=job.stratum_id,
        qubit_count=job.qubit_count,
        optimization_block_id=job.optimization_block_id,
        optimization_seed=job.optimization_seed,
        data_role=cast("ProductionDataRole", job.data_role),
    )


def _stage_outcome_checksum(outcome: object) -> str:
    """Return a deterministic checksum for one genuine repository stage outcome."""
    checksum = getattr(outcome, "content_checksum", None)
    if isinstance(checksum, str):
        return require_checksum(checksum, "stage outcome checksum")
    if isinstance(outcome, StageExecutionEvidence):
        return canonical_checksum({
            "stage_configuration_checksum": outcome.stage.configuration_checksum,
            "selected_parameter_checksum": outcome.selected_parameter_checksum,
            "circuit_binding_checksum": outcome.circuit_binding_checksum,
            "training_summary": dict(outcome.training_summary),
            "normalized_work": dict(outcome.normalized_work),
        })
    msg = "Repository stage outcome has no stable production identity."
    raise TypeError(msg)


def _stage_outcome_document(outcome: object) -> dict[str, object]:
    """Project one genuine stage outcome without dropping raw numerical aliases."""
    if isinstance(outcome, NoisyKrotovStageExecution):
        return _typed_document(
            "noisy_krotov_stage",
            {
                "execution": outcome.to_dict(),
                "circuit_binding_document": dict(outcome.circuit_binding_document),
                "initial_parameters": [float(value) for value in outcome.initial_theta],
                "final_parameters": [float(value) for value in outcome.final_theta],
                "selected_parameters": [float(value) for value in outcome.selected_theta],
                "training_ensemble_checksums": list(outcome.training_ensemble_checksums),
                "checkpoint_validation_ensemble_checksums": list(outcome.checkpoint_validation_ensemble_checksums),
            },
        )
    if isinstance(outcome, StageExecutionEvidence):
        return _typed_document(
            "stage_execution_evidence",
            {
                "stage_configuration_checksum": outcome.stage.configuration_checksum,
                "source_parameter_checksum": outcome.source_parameter_checksum,
                "initial_parameter_checksum": outcome.initial_parameter_checksum,
                "final_parameter_checksum": outcome.final_parameter_checksum,
                "selected_parameter_checksum": outcome.selected_parameter_checksum,
                "initial_parameters": [float(value) for value in outcome.initial_parameters],
                "final_parameters": [float(value) for value in outcome.final_parameters],
                "selected_parameters": [float(value) for value in outcome.selected_parameters],
                "trace": [dict(item) for item in outcome.trace],
                "normalized_work": dict(outcome.normalized_work),
                "training_summary": dict(outcome.training_summary),
                "checkpoint_validation_summary": (
                    None
                    if outcome.checkpoint_validation_summary is None
                    else dict(outcome.checkpoint_validation_summary)
                ),
                "circuit_topology": dict(outcome.circuit_topology),
                "circuit_statistics": dict(outcome.circuit_statistics),
            },
        )
    to_dict = getattr(outcome, "to_dict", None)
    if callable(to_dict):
        document = to_dict()
        if isinstance(document, Mapping):
            return _typed_document("repository_execution", cast("Mapping[str, object]", document))
    msg = "Repository outcome cannot be projected into production evidence."
    raise TypeError(msg)


def _stage_parameters(outcome: object) -> np.ndarray:
    """Return the exact selected handoff vector from one genuine stage outcome."""
    if isinstance(outcome, NoisyKrotovStageExecution):
        return outcome.selected_theta
    if isinstance(outcome, StageExecutionEvidence):
        return outcome.selected_parameters
    msg = "Stage outcome does not expose selected handoff parameters."
    raise TypeError(msg)


def _stage_binding(outcome: object) -> NoisyKrotovCircuitBinding:
    """Reconstruct the exact output circuit binding from a genuine outcome."""
    if isinstance(outcome, NoisyKrotovStageExecution):
        return decode_noisy_krotov_circuit_binding_document(outcome.circuit_binding_document)
    if isinstance(outcome, StageExecutionEvidence):
        return decode_noisy_krotov_circuit_binding_document(outcome.circuit_topology)
    msg = "Stage outcome does not expose an exact circuit binding."
    raise TypeError(msg)


def _outcome_ensembles(outcome: object) -> tuple[KrotovFixedMapEnsemble, ...]:
    """Return every fixed-map ensemble embedded by a genuine stage outcome."""
    if isinstance(outcome, NoisyKrotovStageExecution):
        return (*outcome.training_ensembles, *outcome.checkpoint_validation_ensembles)
    if isinstance(outcome, StageExecutionEvidence):
        return (*outcome.training_ensembles, *outcome.checkpoint_validation_ensembles)
    return ()


class _ScheduledNumerics:
    """Target-bound WP22C callbacks with explicit-member map persistence."""

    def __init__(
        self,
        *,
        store: ProductionAttemptStore,
        stage: TrainingStageConfig,
        circuit_binding: NoisyKrotovCircuitBinding,
        target: MaterializedTarget,
        cross_trajectory: bool,
    ) -> None:
        """Bind one controlled stage, exact circuit, target, and attempt store."""
        self.store = store
        self.stage = stage
        self.circuit_binding = circuit_binding
        self.target = target
        self.cross_trajectory = cross_trajectory
        self.objective_binding = NoisyKrotovObjectiveBinding.from_inputs(
            target,
            None,
            num_qubits=circuit_binding.circuit.num_qubits,
        )
        self.map_blobs: list[ArtifactBlobRef] = []
        self.map_evidence_blobs: list[ArtifactBlobRef] = []
        self._request_index = 0

    @property
    def objective_checksum(self) -> str:
        """Exact target and computational-zero objective checksum."""
        return self.objective_binding.objective_checksum

    @property
    def truncation(self) -> KrotovTruncation:
        """Controlled-stage truncation policy."""
        return KrotovTruncation(
            max_bond_dim=self.stage.max_bond_dimension,
            svd_threshold=self.stage.svd_threshold,
            trunc_mode=self.stage.truncation_mode,
            min_bond_dim=self.stage.min_bond_dimension,
        )

    def _sample_component(
        self,
        *,
        membership: TrajectoryEnsembleMembership,
        member_seeds: tuple[int, ...],
        provider: GateNoiseProvider,
        parameters: np.ndarray,
        component_index: int,
        role: KrotovMapRole,
    ) -> tuple[KrotovFixedMapEnsemble, ArtifactBlobRef]:
        """Sample one component by using every declared member seed directly."""
        options = KrotovTJMOptions(
            num_trajectories=1,
            random_seed=0,
            dt=1.0 if self.stage.tjm_dt is None else self.stage.tjm_dt,
            apply_noise_to="all",
            noisy_gate_indices=self.circuit_binding.noisy_gate_indices,
            trajectory_update="cross" if self.cross_trajectory else "independent",
            differentiate_jump_normalization=False,
            use_crn=False,
        )
        trajectory_maps: list[list[KrotovNoiseMap]] = []
        for member_seed in member_seeds:
            trajectory = forward_tjm_trajectory(
                self.circuit_binding.circuit,
                parameters,
                np.array([], dtype=np.float64),
                MPS(self.circuit_binding.circuit.num_qubits),
                self.truncation,
                None,
                options,
                np.random.Generator(np.random.PCG64(member_seed)),
                noise_provider=provider,
            )
            trajectory_maps.append(trajectory.noise_maps)
        ensemble = KrotovFixedMapEnsemble(
            role=role,
            resolved_seed=membership.map_seed,
            stage_index=self.stage.stage_index,
            stage_id=self.stage.stage_id,
            stage_configuration_checksum=self.stage.configuration_checksum,
            circuit_checksum=self.circuit_binding.content_checksum,
            provider_checksum=_provider_checksum(provider),
            ensemble_index=self._request_index,
            refresh_index=component_index,
            global_iteration_start=membership.update,
            trajectory_maps=trajectory_maps,
        )
        blob = self.store.write_blob(
            f"maps/request_{self._request_index:08d}_component_{component_index:03d}.json",
            f"{ensemble.to_json()}\n".encode(),
            role="fixed_map_ensemble",
            logical_checksum=ensemble.content_checksum,
        )
        self.map_blobs.append(blob)
        return ensemble, blob

    def _maps_for_request(
        self,
        request: ScheduledTrainingGradientRequest | ScheduledTrainingObjectiveRequest,
    ) -> list[list[KrotovNoiseMap]]:
        """Persist exact member/component maps and return their ordered replay lists."""
        membership = request.policy.training_membership
        if membership is None:
            msg = "A noisy scheduled request requires exact aggregate membership."
            raise ValueError(msg)
        components = request.policy.component_memberships
        if components:
            component_specs = tuple(
                (component.noise_id, component.member_seeds, component.content_checksum) for component in components
            )
            component_checksums = tuple(component.content_checksum for component in components)
        else:
            component_specs = ((self.stage.training_noise_id, membership.member_seeds, None),)
            component_checksums = ()
        ensembles: list[KrotovFixedMapEnsemble] = []
        ensemble_refs: list[ArtifactBlobRef] = []
        provider_checksums: list[str] = []
        for component_index, (noise_id, member_seeds, _component_checksum) in enumerate(component_specs):
            if noise_id == NOISELESS_NOISE_ID:
                msg = "A positive-trajectory scheduled request cannot select a noiseless provider."
                raise ValueError(msg)
            provider = create_scaled_standard_noise_provider(noise_id, request.policy.noise_strength_scale)
            ensemble, ref = self._sample_component(
                membership=membership,
                member_seeds=member_seeds,
                provider=cast("GateNoiseProvider", provider),
                parameters=np.asarray(request.parameters, dtype=np.float64),
                component_index=component_index,
                role="training_trajectory",
            )
            ensembles.append(ensemble)
            ensemble_refs.append(ref)
            provider_checksums.append(_provider_checksum(provider))
        evidence = ScheduledMapEvidence(
            request_checksum=request.content_checksum,
            policy_checksum=request.policy.content_checksum,
            membership_checksum=membership.content_checksum,
            component_membership_checksums=component_checksums,
            member_seeds=membership.member_seeds,
            component_member_seeds=tuple(spec[1] for spec in component_specs),
            map_role="training_trajectory",
            resolved_seeds=(membership.map_seed,) * len(component_specs),
            circuit_checksum=self.circuit_binding.content_checksum,
            provider_checksums=tuple(provider_checksums),
            ensemble_refs=tuple(ensemble_refs),
        )
        evidence_ref = self.store.write_json_blob(
            f"map_evidence/request_{self._request_index:08d}.json",
            evidence.to_dict(),
            role="scheduled_map_evidence",
        )
        self.map_evidence_blobs.append(evidence_ref)
        self._request_index += 1
        return [maps for ensemble in ensembles for maps in ensemble.replay_maps()]

    def gradient(self, request: ScheduledTrainingGradientRequest) -> ScheduledTrainingGradientResult:
        """Evaluate one real Krotov contribution through scheduled fixed maps."""
        parameters = np.asarray(request.parameters, dtype=np.float64)
        if request.policy.trajectory_count == 0 or np.isclose(request.policy.noise_strength_scale, 0.0):
            contribution = state_preparation_contribution(
                self.circuit_binding.circuit,
                parameters,
                self.target.state_vector_copy(),
                MPS(self.circuit_binding.circuit.num_qubits),
                self.truncation,
            )[0]
        else:
            maps = self._maps_for_request(request)
            contribution = noisy_state_preparation_contribution(
                self.circuit_binding.circuit,
                parameters,
                self.target.state_vector_copy(),
                None,
                KrotovTJMOptions(
                    num_trajectories=len(maps),
                    dt=1.0 if self.stage.tjm_dt is None else self.stage.tjm_dt,
                    noisy_gate_indices=self.circuit_binding.noisy_gate_indices,
                    trajectory_update="cross" if self.cross_trajectory else "independent",
                ),
                MPS(self.circuit_binding.circuit.num_qubits),
                self.truncation,
                fixed_noise_maps=maps,
            )[0]
        return ScheduledTrainingGradientResult.for_request(
            request,
            tuple(float(value) for value in contribution),
        )

    def objective(self, request: ScheduledTrainingObjectiveRequest) -> ScheduledTrainingObjectiveResult:
        """Evaluate one real state-preparation loss through scheduled fixed maps."""
        parameters = np.asarray(request.parameters, dtype=np.float64)
        if request.policy.trajectory_count == 0 or np.isclose(request.policy.noise_strength_scale, 0.0):
            loss = state_preparation_loss(
                self.circuit_binding.circuit,
                parameters,
                self.target.state_vector_copy(),
                truncation=self.truncation,
            )
        else:
            maps = self._maps_for_request(request)
            loss = noisy_state_preparation_loss(
                self.circuit_binding.circuit,
                parameters,
                self.target.state_vector_copy(),
                None,
                KrotovTJMOptions(
                    num_trajectories=len(maps),
                    dt=1.0 if self.stage.tjm_dt is None else self.stage.tjm_dt,
                    noisy_gate_indices=self.circuit_binding.noisy_gate_indices,
                ),
                truncation=self.truncation,
                fixed_noise_maps=maps,
            )
        return ScheduledTrainingObjectiveResult.for_request(request, float(loss))

    def validate(self, request: ScheduledValidationRequest) -> ScheduledValidationResult:
        """Evaluate a checkpoint only through its validation-only explicit membership."""
        membership = request.membership
        parameters = np.asarray(request.parameter_artifact.parameters, dtype=np.float64)
        provider = create_scaled_standard_noise_provider("depolarizing_1s_all", 1.0)
        # Validation has no training policy, but it uses the same explicit-member
        # sampler.  A tiny proxy object is unnecessary: sample directly and bind
        # the resulting ensemble to the exact validation request checksum.
        options = KrotovTJMOptions(
            num_trajectories=1,
            dt=1.0,
            noisy_gate_indices=self.circuit_binding.noisy_gate_indices,
        )
        maps: list[list[KrotovNoiseMap]] = []
        for member_seed in membership.member_seeds:
            trajectory = forward_tjm_trajectory(
                self.circuit_binding.circuit,
                parameters,
                np.array([], dtype=np.float64),
                MPS(self.circuit_binding.circuit.num_qubits),
                self.truncation,
                None,
                options,
                np.random.Generator(np.random.PCG64(member_seed)),
                noise_provider=cast("GateNoiseProvider", provider),
            )
            maps.append(trajectory.noise_maps)
        ensemble = KrotovFixedMapEnsemble(
            role="checkpoint_validation",
            resolved_seed=membership.map_seed,
            stage_index=self.stage.stage_index,
            stage_id=self.stage.stage_id,
            stage_configuration_checksum=self.stage.configuration_checksum,
            circuit_checksum=self.circuit_binding.content_checksum,
            provider_checksum=_provider_checksum(provider),
            ensemble_index=self._request_index,
            refresh_index=0,
            global_iteration_start=membership.update,
            trajectory_maps=maps,
        )
        ensemble_ref = self.store.write_blob(
            f"maps/request_{self._request_index:08d}_validation.json",
            f"{ensemble.to_json()}\n".encode(),
            role="fixed_map_ensemble",
            logical_checksum=ensemble.content_checksum,
        )
        self.map_blobs.append(ensemble_ref)
        evidence = ScheduledMapEvidence(
            request_checksum=request.content_checksum,
            policy_checksum=request.program_checksum,
            membership_checksum=membership.content_checksum,
            component_membership_checksums=(),
            member_seeds=membership.member_seeds,
            component_member_seeds=(membership.member_seeds,),
            map_role="checkpoint_validation",
            resolved_seeds=(membership.map_seed,),
            circuit_checksum=self.circuit_binding.content_checksum,
            provider_checksums=(_provider_checksum(provider),),
            ensemble_refs=(ensemble_ref,),
        )
        evidence_ref = self.store.write_json_blob(
            f"map_evidence/request_{self._request_index:08d}.json",
            evidence.to_dict(),
            role="scheduled_map_evidence",
        )
        self.map_evidence_blobs.append(evidence_ref)
        self._request_index += 1
        _loss, fidelity, _fidelities = noisy_state_preparation_metrics(
            self.circuit_binding.circuit,
            parameters,
            self.target.state_vector_copy(),
            None,
            KrotovTJMOptions(
                num_trajectories=len(maps),
                dt=1.0,
                noisy_gate_indices=self.circuit_binding.noisy_gate_indices,
            ),
            truncation=self.truncation,
            fixed_noise_maps=maps,
        )
        return ScheduledValidationResult.for_request(request, float(fidelity))


@dataclass(frozen=True, slots=True)
class _FreshEvaluation:
    """Raw fresh-evaluation artifacts and derived reporting metrics."""

    ensemble_ref: ArtifactBlobRef
    trajectory_ref: ArtifactBlobRef
    metrics: Mapping[str, object]


def _fresh_evaluate(
    *,
    store: ProductionAttemptStore,
    resolved: ResolvedProductionJob,
    circuit_binding: NoisyKrotovCircuitBinding,
    parameters: Sequence[float],
) -> _FreshEvaluation:
    """Run the exact role-separated fresh fixed-sample evaluator."""
    policy = resolved.evaluation_policy
    evaluation_policy_checksum = resolved.evaluation_policy_checksum
    noise = policy.noise_condition
    noise_id = cast("str", noise["noise_id"])
    strength = cast("float", noise["strength_scale"])
    tjm_dt = cast("float", noise["tjm_dt"])
    provider = create_scaled_standard_noise_provider(noise_id, strength)
    theta = np.asarray(parameters, dtype=np.float64)
    truncation = KrotovTruncation(
        max_bond_dim=cast("int | None", policy.truncation_policy["max_bond_dimension"]),
        svd_threshold=cast("float", policy.truncation_policy["svd_threshold"]),
        trunc_mode=cast(
            "Literal['discarded_weight', 'relative']",
            policy.truncation_policy["truncation_mode"],
        ),
        min_bond_dim=cast("int", policy.truncation_policy["min_bond_dimension"]),
    )
    role_by_data_role: dict[str, KrotovMapRole] = {
        "development": "pilot_evaluation",
        "secondary_benchmark": "pilot_evaluation",
        "screening_selection": "screening_selection",
        "confirmatory": "confirmatory_test",
        "checkpoint_validation": "checkpoint_validation",
    }
    map_role = role_by_data_role[policy.data_role]
    evaluation_configuration_checksum = canonical_checksum({
        "job_checksum": resolved.evidence_identity_checksum,
        "evaluation_policy_checksum": evaluation_policy_checksum,
        "circuit_checksum": circuit_binding.content_checksum,
        "parameter_checksum": canonical_checksum({
            "parameters": [float(value) for value in theta],
        }),
    })
    trajectory_fidelities: list[float] = []
    realized_maps: list[list[KrotovNoiseMap]] = []
    for trajectory_index in range(policy.trajectory_count):
        member_seed = derive_krotov_trajectory_seed(
            role=map_role,
            resolved_seed=resolved.job.evaluation_seed,
            stage_index=0,
            ensemble_index=0,
            trajectory_index=trajectory_index,
            refresh_index=0,
        )
        member_metrics = noisy_state_preparation_metrics_with_maps(
            circuit_binding.circuit,
            theta,
            resolved.target.state_vector_copy(),
            None,
            KrotovTJMOptions(
                num_trajectories=1,
                random_seed=member_seed,
                dt=tjm_dt,
                apply_noise_to="all",
                noisy_gate_indices=circuit_binding.noisy_gate_indices,
                trajectory_update="independent",
                differentiate_jump_normalization=False,
                use_crn=False,
            ),
            initial_state=MPS(circuit_binding.circuit.num_qubits),
            truncation=truncation,
            iteration=0,
            noise_provider=cast("GateNoiseProvider", provider),
        )
        if (
            len(member_metrics.trajectory_fidelities) != 1
            or len(member_metrics.realized_noise_maps) != 1
            or len(member_metrics.realized_noise_maps[0]) != len(circuit_binding.circuit.gates)
        ):
            msg = "Fresh single-member evaluation returned non-singleton evidence."
            raise ValueError(msg)
        trajectory_fidelities.append(member_metrics.trajectory_fidelities[0])
        realized_maps.append(list(member_metrics.realized_noise_maps[0]))
    ensemble = KrotovFixedMapEnsemble(
        role=map_role,
        resolved_seed=resolved.job.evaluation_seed,
        stage_index=0,
        stage_id="fresh_evaluation",
        stage_configuration_checksum=evaluation_configuration_checksum,
        circuit_checksum=circuit_binding.content_checksum,
        provider_checksum=_provider_checksum(provider),
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
        trajectory_maps=realized_maps,
    )
    ensemble_ref = store.write_blob(
        "evaluation/fresh_fixed_map_ensemble.json",
        f"{ensemble.to_json()}\n".encode(),
        role="fixed_map_ensemble",
        logical_checksum=ensemble.content_checksum,
    )
    _noiseless_loss, noiseless_fidelity = state_preparation_metrics(
        circuit_binding.circuit,
        theta,
        resolved.target.state_vector_copy(),
        truncation=truncation,
    )
    noisy_fidelity = float(np.mean(trajectory_fidelities))
    sidecar = _typed_document(
        "raw_trajectory_fidelities",
        {
            "job_checksum": resolved.evidence_identity_checksum,
            "evaluation_policy_checksum": evaluation_policy_checksum,
            "evaluation_configuration_checksum": evaluation_configuration_checksum,
            "data_role": policy.data_role,
            "seed_domain": policy.seed_domain,
            "evaluation_seed": resolved.job.evaluation_seed,
            "trajectory_count": policy.trajectory_count,
            "trajectory_fidelities": [float(value) for value in trajectory_fidelities],
            "fixed_map_ensemble_checksum": ensemble.content_checksum,
            "sampled_nonidentity_events": ensemble.nonidentity_event_count,
        },
    )
    trajectory_ref = store.write_json_blob(
        "evaluation/raw_trajectory_fidelities.json",
        sidecar,
        role="raw_trajectory_sidecar",
    )
    metrics: dict[str, object] = {
        "noiseless_fidelity": float(noiseless_fidelity),
        "noisy_fidelity": float(noisy_fidelity),
        "evaluation_configuration_checksum": evaluation_configuration_checksum,
        "evaluation_data_role": policy.data_role,
        "evaluation_seed_domain": policy.seed_domain,
        "evaluation_seed": resolved.job.evaluation_seed,
        "trajectory_count": policy.trajectory_count,
        "reporting_prefixes": list(policy.reporting_prefixes),
        "prefix_mean_fidelities": {
            str(prefix): float(np.mean(trajectory_fidelities[:prefix])) for prefix in policy.reporting_prefixes
        },
        "sampled_nonidentity_events": ensemble.nonidentity_event_count,
        "provider_checksum": _provider_checksum(provider),
        "fresh_ensemble_checksum": ensemble.content_checksum,
    }
    return _FreshEvaluation(ensemble_ref, trajectory_ref, metrics)


def _pilot_diagnostic(
    *,
    store: ProductionAttemptStore,
    resolved: ResolvedProductionJob,
    circuit_binding: NoisyKrotovCircuitBinding,
    parameters: Sequence[float],
    checkpoint_parameter_checksum: str,
) -> tuple[tuple[ArtifactBlobRef, ...], ArtifactBlobRef] | None:
    """Persist the frozen 32-vector q6 pathwise diagnostic at the selected checkpoint."""
    policy = resolved.executable_binding.binding.pilot_diagnostic_policy
    if resolved.job.preset != "paper-pilot" or policy is None or not policy.enabled:
        return None
    if resolved.job.qubit_count != 6 or policy.trajectory_count != 32:
        msg = "Enabled pilot diagnostics require the frozen q6 32-vector policy."
        raise ValueError(msg)
    noise = policy.noise_condition
    provider_identity = policy.provider_identity
    if noise is None or provider_identity is None:
        msg = "Enabled pilot diagnostic policy lacks its exact primary-noise provider."
        raise ValueError(msg)
    noise_id = cast("str", noise["noise_id"])
    strength = cast("float", noise["strength_scale"])
    tjm_dt = cast("float", noise["tjm_dt"])
    provider = create_scaled_standard_noise_provider(noise_id, strength)
    provider_checksum = _provider_checksum(provider)
    if provider_checksum != provider_identity["content_checksum"]:
        msg = "Pilot diagnostic provider differs from its frozen policy identity."
        raise ValueError(msg)
    theta = np.asarray(parameters, dtype=np.float64)
    if theta.shape != (circuit_binding.circuit.num_params,) or not np.all(np.isfinite(theta)):
        msg = "Pilot diagnostic parameters differ from the materialized selected checkpoint."
        raise ValueError(msg)
    parameter_vector_checksum = canonical_checksum({
        "dtype": "float64",
        "parameters": [float(value) for value in theta],
    })
    if parameter_vector_checksum != checkpoint_parameter_checksum:
        msg = "Pilot diagnostic parameters do not reproduce the selected WP22C checkpoint checksum."
        raise ValueError(msg)
    evaluation_truncation = resolved.evaluation_policy.truncation_policy
    truncation = KrotovTruncation(
        max_bond_dim=cast("int | None", evaluation_truncation["max_bond_dimension"]),
        svd_threshold=cast("float", evaluation_truncation["svd_threshold"]),
        trunc_mode=cast(
            "Literal['discarded_weight', 'relative']",
            evaluation_truncation["truncation_mode"],
        ),
        min_bond_dim=cast("int", evaluation_truncation["min_bond_dimension"]),
    )
    options = KrotovTJMOptions(
        num_trajectories=1,
        dt=tjm_dt,
        noisy_gate_indices=circuit_binding.noisy_gate_indices,
        trajectory_update="independent",
        use_crn=False,
    )
    stage_configuration_checksum = canonical_checksum({
        "job_checksum": resolved.evidence_identity_checksum,
        "policy_checksum": policy.content_checksum,
        "checkpoint_parameter_checksum": checkpoint_parameter_checksum,
        "parameter_vector_checksum": parameter_vector_checksum,
        "circuit_checksum": circuit_binding.content_checksum,
    })
    seed_suite = ExecutionSeedPolicySuite.frozen()
    member_seeds: list[int] = []
    vectors: list[tuple[float, ...]] = []
    ensemble_refs: list[ArtifactBlobRef] = []
    for repetition in range(policy.trajectory_count):
        member_seed = seed_suite.derive(
            PILOT_DIAGNOSTIC_SEED_POLICY_ID,
            {
                "target_manifest_checksum": resolved.job.target_manifest_checksum,
                "target_instance_spec_checksum": resolved.job.target_spec_checksum,
                "optimization_seed": resolved.job.optimization_seed,
                "publication_candidate_checksum": resolved.job.candidate_configuration_checksum,
                "repetition": repetition,
            },
        )
        trajectory = forward_tjm_trajectory(
            circuit_binding.circuit,
            theta,
            np.array([], dtype=np.float64),
            MPS(circuit_binding.circuit.num_qubits),
            truncation,
            None,
            options,
            np.random.Generator(np.random.PCG64(member_seed)),
            noise_provider=cast("GateNoiseProvider", provider),
        )
        ensemble = KrotovFixedMapEnsemble(
            role="pilot_evaluation",
            resolved_seed=member_seed,
            stage_index=0,
            stage_id="pilot_diagnostic",
            stage_configuration_checksum=stage_configuration_checksum,
            circuit_checksum=circuit_binding.content_checksum,
            provider_checksum=provider_checksum,
            ensemble_index=repetition,
            refresh_index=0,
            global_iteration_start=0,
            trajectory_maps=(trajectory.noise_maps,),
        )
        ensemble_ref = store.write_blob(
            f"diagnostics/maps/pathwise_{repetition:03d}.json",
            f"{ensemble.to_json()}\n".encode(),
            role="fixed_map_ensemble",
            logical_checksum=ensemble.content_checksum,
        )
        contribution = noisy_state_preparation_contribution(
            circuit_binding.circuit,
            theta,
            resolved.target.state_vector_copy(),
            None,
            options,
            MPS(circuit_binding.circuit.num_qubits),
            truncation,
            fixed_noise_maps=ensemble.replay_maps(),
        )[0]
        member_seeds.append(member_seed)
        ensemble_refs.append(ensemble_ref)
        vectors.append(tuple(float(value) for value in contribution))
    estimator_checksum = canonical_checksum({
        "endpoint": policy.endpoint,
        "checkpoint_rule": policy.checkpoint_rule,
        "estimator_id": policy.estimator_id,
        "estimator_version": policy.estimator_version,
        "parameter_ordering": policy.parameter_ordering,
        "coordinate_variance_rule": policy.coordinate_variance_rule,
        "summary_statistics": list(policy.summary_statistics),
        "provider_checksum": provider_checksum,
    })
    diagnostic = PilotDiagnosticEvidence(
        job_checksum=resolved.evidence_identity_checksum,
        policy_checksum=policy.content_checksum,
        checkpoint_parameter_checksum=checkpoint_parameter_checksum,
        parameter_vector_checksum=parameter_vector_checksum,
        circuit_checksum=circuit_binding.content_checksum,
        provider_checksum=provider_checksum,
        estimator_checksum=estimator_checksum,
        member_seeds=tuple(member_seeds),
        ensemble_refs=tuple(ensemble_refs),
        pathwise_update_vectors=tuple(vectors),
    )
    diagnostic_ref = store.write_json_blob(
        "diagnostics/pathwise_update_vectors.json",
        diagnostic.to_dict(),
        role="pilot_diagnostic_sidecar",
    )
    return ((*ensemble_refs, diagnostic_ref), diagnostic_ref)


def _optimizer_payloads(
    program: ScheduledExecutionProgram,
    stage: TrainingStageConfig,
    predecessor: np.ndarray | None,
    source_checksum: str,
) -> tuple[KrotovOptimizerPayload | AdamOptimizerPayload | SPSAOptimizerPayload, ...]:
    """Construct the exact binding-owned update-zero payload for every start."""
    result: list[KrotovOptimizerPayload | AdamOptimizerPayload | SPSAOptimizerPayload] = []
    for bundle in program.start_seed_bundles:
        if predecessor is None:
            scale = cast("float", stage.optimizer_hyperparameters["initialization_scale"])
            initialization = OptimizerInitialization.normal(bundle, stage.output_parameter_count, scale=scale)
        else:
            initialization = OptimizerInitialization.warm_start(
                bundle,
                tuple(float(value) for value in predecessor),
                source_checksum=source_checksum,
            )
        hyperparameters = stage.optimizer_hyperparameters
        if stage.optimizer_id == "krotov":
            result.append(
                KrotovOptimizerPayload.initialize(
                    initialization,
                    learning_rate=cast("float", hyperparameters["learning_rate"]),
                    learning_rate_schedule=cast(
                        "Literal['constant', 'inverse', 'exp']",
                        hyperparameters.get("schedule", "constant"),
                    ),
                    decay=cast("float", hyperparameters.get("decay", 0.0)),
                )
            )
        elif stage.optimizer_id == "parameter_shift_adam":
            result.append(
                AdamOptimizerPayload.initialize(
                    initialization,
                    ParameterShiftAdamConfig.from_stage(stage),
                )
            )
        elif stage.optimizer_id == "spsa":
            result.append(SPSAOptimizerPayload.initialize(initialization, SPSAConfig.from_stage(stage)))
        else:
            msg = f"Unsupported controlled pipeline optimizer {stage.optimizer_id!r}."
            raise ValueError(msg)
    return tuple(result)


def _run_structural_prefix(
    resolved: ResolvedProductionJob,
    pipeline: TrainingPipelineConfig,
    store: ProductionAttemptStore,
) -> tuple[object, np.ndarray | None, NoisyKrotovCircuitBinding, tuple[str, ...], list[ArtifactBlobRef]]:
    """Execute every genuine repository stage before the controlled terminal stage."""
    if any(stage.training_noise_id == "ballarin_coupled" for stage in pipeline.stages):
        msg = "ballarin_coupled is evaluation-only and cannot enter production training."
        raise ValueError(msg)
    runner_factory = cast("PipelineRunnerFactory", resolved.executable_binding.resolve_callable())
    runner = runner_factory(pipeline, resolved.target)
    predecessor: np.ndarray | None = None
    latest_outcome: object | None = None
    prefix_checksums: list[str] = []
    refs: list[ArtifactBlobRef] = []
    for stage in pipeline.stages[:-1]:
        outcome = runner(stage, predecessor)
        if isinstance(outcome, NoisyKrotovStageFailure):
            msg = f"Structural prefix stage {stage.stage_id!r} failed before the controlled schedule."
            raise TypeError(msg)
        checksum = _stage_outcome_checksum(outcome)
        prefix_checksums.append(checksum)
        refs.append(
            store.write_json_blob(
                f"structural_prefix/stage_{stage.stage_index:03d}.json",
                _stage_outcome_document(outcome),
                role="structural_stage",
            )
        )
        for map_index, ensemble in enumerate(_outcome_ensembles(outcome)):
            refs.append(
                store.write_blob(
                    f"structural_prefix/maps/stage_{stage.stage_index:03d}_{map_index:05d}.json",
                    f"{ensemble.to_json()}\n".encode(),
                    role="fixed_map_ensemble",
                    logical_checksum=ensemble.content_checksum,
                )
            )
        predecessor = _stage_parameters(outcome)
        latest_outcome = outcome
    controlled = pipeline.stages[-1]
    if latest_outcome is None:
        topology = controlled.output_topology_id
        try:
            depth = int(topology.rsplit("_d", maxsplit=1)[1])
        except (IndexError, ValueError):
            msg = "An initial controlled pipeline stage requires a BMPD output topology."
            raise ValueError(msg) from None
        binding = create_bmpd_circuit_binding(pipeline.qubit_count, depth)
    else:
        binding = _stage_binding(latest_outcome)
    if (
        binding.topology_id != controlled.output_topology_id
        or binding.circuit.num_params != controlled.output_parameter_count
    ):
        msg = "Structural prefix circuit does not match the controlled terminal stage."
        raise ValueError(msg)
    return runner, predecessor, binding, tuple(prefix_checksums), refs


def _scheduled_pipeline_execution(
    resolved: ResolvedProductionJob,
    store: ProductionAttemptStore,
) -> tuple[
    ScheduledExecutionSnapshot,
    NoisyKrotovCircuitBinding,
    tuple[float, ...],
    tuple[str, ...],
    list[ArtifactBlobRef],
    float,
    int,
]:
    """Run structural prefix then the exact target-bound WP22C controlled stage."""
    pipeline = _pipeline_config(resolved)
    _runner, predecessor, circuit_binding, prefix_checksums, refs = _run_structural_prefix(
        resolved,
        pipeline,
        store,
    )
    stage = pipeline.stages[-1]
    source_checksum = canonical_checksum({
        "prefix_checksums": list(prefix_checksums),
        "circuit_binding_checksum": circuit_binding.content_checksum,
        "predecessor_parameter_checksum": (
            None if predecessor is None else canonical_checksum({"parameters": [float(value) for value in predecessor]})
        ),
    })
    payloads = _optimizer_payloads(resolved.scheduled_program, stage, predecessor, source_checksum)
    snapshot = initialize_scheduled_execution(resolved.scheduled_program, payloads)
    numerics = _ScheduledNumerics(
        store=store,
        stage=stage,
        circuit_binding=circuit_binding,
        target=resolved.target,
        cross_trajectory=stage.trajectory_update == "cross",
    )
    if stage.optimizer_id == "krotov":
        adapter = KrotovScheduledUpdateAdapter(
            numerics.objective_checksum,
            numerics.gradient,
            cross_trajectory=stage.trajectory_update == "cross",
        )
    elif stage.optimizer_id == "parameter_shift_adam":
        scales = (1.0,) * (3 * pipeline.qubit_count) + (-1.0,) * (
            stage.output_parameter_count - 3 * pipeline.qubit_count
        )
        adapter = ParameterShiftAdamScheduledUpdateAdapter(
            numerics.objective_checksum,
            scales,
            numerics.objective,
        )
    elif stage.optimizer_id == "spsa":
        adapter = SPSAScheduledUpdateAdapter(numerics.objective_checksum, numerics.objective)
    else:
        msg = f"Unsupported controlled pipeline optimizer {stage.optimizer_id!r}."
        raise ValueError(msg)
    measured = _measure_call(
        lambda: execute_scheduled_program(
            resolved.scheduled_program,
            snapshot,
            adapter,
            validation_executor=numerics.validate,
        )
    )
    result = cast("ScheduledExecutionSnapshot", measured.value)
    if not result.complete or result.multistart_evidence is None:
        msg = "Production schedule execution did not reach a terminal complete snapshot."
        raise RuntimeError(msg)
    refs.extend((*numerics.map_blobs, *numerics.map_evidence_blobs))
    selected = result.multistart_evidence.selected_parameter_artifact.parameters
    return (
        result,
        circuit_binding,
        selected,
        prefix_checksums,
        refs,
        measured.wall_time_seconds,
        measured.peak_memory_bytes,
    )


def _runtime_resource_document(
    *,
    resolved: ResolvedProductionJob,
    circuit_binding: NoisyKrotovCircuitBinding | None,
    wall_time_seconds: float,
    peak_memory_bytes: int,
    normalized_work: float,
    failure_phase: str | None = None,
    partial_receipts: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one authoritative runtime/resource/work sidecar."""
    circuit_payload: dict[str, object] | None = None
    if circuit_binding is not None:
        circuit = circuit_binding.circuit
        compiled_resources = measure_circuit_resources(circuit)
        circuit_payload = {
            "circuit_binding_checksum": circuit_binding.content_checksum,
            "topology_id": circuit_binding.topology_id,
            "qubit_count": circuit.num_qubits,
            "parameter_count": circuit.num_params,
            "logical_gate_count": len(circuit.gates),
            "logical_two_qubit_gate_count": sum(len(gate.sites) == 2 for gate in circuit.gates),
            "noisy_gate_indices": list(circuit_binding.noisy_gate_indices),
            "compiled_resources": compiled_resources.to_dict(),
            "compiled_resources_checksum": compiled_resources.content_checksum,
            "native_two_qubit_gates_per_chain_edge": list(compiled_resources.native_two_qubit_gates_per_chain_edge),
        }
    return _typed_document(
        "runtime_resources",
        {
            "job_checksum": resolved.evidence_identity_checksum,
            "source_fingerprint_checksum": resolved.source_fingerprint_checksum,
            "wall_time_seconds": require_float(wall_time_seconds, "wall_time_seconds", minimum=0.0),
            "peak_memory_bytes": require_int(peak_memory_bytes, "peak_memory_bytes"),
            "normalized_work": require_float(normalized_work, "normalized_work", minimum=0.0),
            "failure_phase": failure_phase,
            "partial_receipts": None if partial_receipts is None else dict(partial_receipts),
            "circuit": circuit_payload,
        },
    )


def _publish_attempt(
    *,
    store: ProductionAttemptStore,
    resolved: ResolvedProductionJob,
    artifact_kind: ArtifactKind,
    status: ArtifactStatus,
    blobs: Sequence[ArtifactBlobRef],
    prefix_checksums: tuple[str, ...],
    schedule_snapshot_ref: ArtifactBlobRef | None,
    map_evidence_refs: tuple[ArtifactBlobRef, ...],
    raw_trajectory_ref: ArtifactBlobRef | None,
    resource_ref: ArtifactBlobRef,
    derived_metrics: Mapping[str, object],
    diagnostic_refs: tuple[ArtifactBlobRef, ...] = (),
    failure: Mapping[str, object] | None = None,
) -> ResultArtifactRef:
    """Write typed evidence then publish its complete terminal manifest."""
    job = resolved.job
    metrics = dict(derived_metrics)
    metrics.update({
        "execution_preset": job.preset,
        "scheduled_noisy_training": any(
            policy.trajectory_count > 0 for policy in resolved.scheduled_program.update_policies
        ),
        "pilot_diagnostic_required": job.preset == "paper-pilot" and job.qubit_count == 6,
    })
    if status == "success":
        if artifact_kind == "pipeline" and raw_trajectory_ref is None:
            msg = "Successful pipeline evidence requires raw fresh-evaluation trajectories."
            raise ValueError(msg)
        if artifact_kind == "pipeline" and job.preset != "training-smoke" and schedule_snapshot_ref is None:
            msg = "Successful production pipeline evidence requires its complete schedule snapshot."
            raise ValueError(msg)
        if (
            artifact_kind == "pipeline"
            and metrics["scheduled_noisy_training"]
            and job.preset != "training-smoke"
            and not map_evidence_refs
        ):
            msg = "Successful noisy scheduled training requires explicit-member map evidence."
            raise ValueError(msg)
        if artifact_kind == "operator_growth" and job.preset != "training-smoke" and raw_trajectory_ref is None:
            msg = "Successful production operator growth requires raw outer-evaluation trajectories."
            raise ValueError(msg)
        if artifact_kind == "operator_growth" and not prefix_checksums:
            msg = "Successful operator growth requires its exact structural numerical result."
            raise ValueError(msg)
        if metrics["pilot_diagnostic_required"] and not diagnostic_refs:
            msg = "Successful primary-q6 pilot execution requires its frozen pathwise diagnostic."
            raise ValueError(msg)
    evidence = ProductionNumericalEvidence(
        job_checksum=resolved.evidence_identity_checksum,
        attempt=store.attempt,
        artifact_kind=artifact_kind,
        status=status,
        execution_source_manifest_checksum=resolved.execution_source_manifest_checksum,
        source_fingerprint_checksum=resolved.source_fingerprint_checksum,
        executable_binding_checksum=resolved.executable_binding_checksum,
        scheduled_program_checksum=resolved.scheduled_program.content_checksum,
        target_identity=resolved.target.identity_dict(),
        evaluation_policy_checksum=resolved.evaluation_policy_checksum,
        structural_prefix_checksums=prefix_checksums,
        schedule_snapshot_ref=schedule_snapshot_ref,
        map_evidence_refs=map_evidence_refs,
        diagnostic_refs=diagnostic_refs,
        raw_trajectory_ref=raw_trajectory_ref,
        resource_ref=resource_ref,
        derived_metrics={**metrics, "strategy_schedule_checksum": resolved.strategy_schedule_checksum},
        failure=failure,
    )
    evidence_ref = store.write_json_blob(
        "production_evidence.json",
        evidence.to_dict(),
        role="production_evidence",
    )
    return store.publish(
        artifact_kind=artifact_kind,
        status=status,
        execution_source_manifest_checksum=resolved.execution_source_manifest_checksum,
        source_fingerprint_checksum=resolved.source_fingerprint_checksum,
        blobs=(*blobs, evidence_ref),
        evidence_ref=evidence_ref,
    )


def _execute_pipeline_attempt(
    resolved: ResolvedProductionJob,
    store: ProductionAttemptStore,
) -> ResultArtifactRef:
    """Execute one full production pipeline and publish raw fresh evidence."""
    (
        snapshot,
        circuit_binding,
        selected,
        prefix_checksums,
        refs,
        wall_time,
        peak_memory,
    ) = _scheduled_pipeline_execution(resolved, store)
    snapshot_ref = store.write_blob(
        "schedule/snapshot.json",
        f"{snapshot.to_json()}\n".encode(),
        role="schedule_snapshot",
        logical_checksum=snapshot.content_checksum,
    )
    refs.append(snapshot_ref)
    fresh_measured = _measure_call(
        lambda: _fresh_evaluate(
            store=store,
            resolved=resolved,
            circuit_binding=circuit_binding,
            parameters=tuple(float(value) for value in selected),
        )
    )
    fresh = cast("_FreshEvaluation", fresh_measured.value)
    refs.extend((fresh.ensemble_ref, fresh.trajectory_ref))
    multistart = snapshot.multistart_evidence
    assert multistart is not None
    diagnostic_measured = _measure_call(
        lambda: _pilot_diagnostic(
            store=store,
            resolved=resolved,
            circuit_binding=circuit_binding,
            parameters=selected,
            checkpoint_parameter_checksum=multistart.selected_parameter_checksum,
        )
    )
    diagnostic_result = cast(
        "tuple[tuple[ArtifactBlobRef, ...], ArtifactBlobRef] | None",
        diagnostic_measured.value,
    )
    diagnostic_refs: tuple[ArtifactBlobRef, ...] = ()
    diagnostic_count = 0
    if diagnostic_result is not None:
        diagnostic_blobs, diagnostic_ref = diagnostic_result
        refs.extend(diagnostic_blobs)
        diagnostic_refs = (diagnostic_ref,)
        diagnostic_count = 32
    resource = _runtime_resource_document(
        resolved=resolved,
        circuit_binding=circuit_binding,
        wall_time_seconds=(wall_time + fresh_measured.wall_time_seconds + diagnostic_measured.wall_time_seconds),
        peak_memory_bytes=max(
            peak_memory,
            fresh_measured.peak_memory_bytes,
            diagnostic_measured.peak_memory_bytes,
        ),
        normalized_work=(
            multistart.total_normalized_work + resolved.evaluation_policy.trajectory_count + diagnostic_count
        ),
    )
    resource_ref = store.write_json_blob(
        "runtime/resources.json",
        resource,
        role="runtime_resources",
    )
    refs.append(resource_ref)
    map_evidence_refs = tuple(ref for ref in refs if ref.role == "scheduled_map_evidence")
    return _publish_attempt(
        store=store,
        resolved=resolved,
        artifact_kind="pipeline",
        status="success",
        blobs=refs,
        prefix_checksums=prefix_checksums,
        schedule_snapshot_ref=snapshot_ref,
        map_evidence_refs=map_evidence_refs,
        raw_trajectory_ref=fresh.trajectory_ref,
        resource_ref=resource_ref,
        diagnostic_refs=diagnostic_refs,
        derived_metrics={
            **dict(fresh.metrics),
            "selected_start_index": multistart.selected_start_index,
            "selected_update": multistart.selected_update,
            "selected_parameter_checksum": multistart.selected_parameter_checksum,
            "total_normalized_training_work": multistart.total_normalized_work,
            "pilot_diagnostic_checksum": (None if not diagnostic_refs else diagnostic_refs[0].logical_checksum),
        },
    )


def _execute_pipeline_smoke_attempt(
    resolved: ResolvedProductionJob,
    store: ProductionAttemptStore,
) -> ResultArtifactRef:
    """Run one real repository pipeline-family smoke stage and fresh evaluation."""
    runtime = resolved.executable_binding.smoke_runtime_program()
    if not isinstance(runtime, PipelineSmokeRuntimeProgram):
        msg = "Pipeline smoke dispatch resolved a non-pipeline runtime."
        raise TypeError(msg)
    bound = runtime.bind(
        resolved.target_manifest,
        resolved.target,
        optimization_seed=resolved.job.optimization_seed,
    )
    measured = _measure_call(bound.execute)
    outcome = measured.value
    if isinstance(outcome, NoisyKrotovStageFailure):
        msg = "Genuine pipeline smoke stage returned structured numerical failure."
        raise TypeError(msg)
    outcome_document = _stage_outcome_document(outcome)
    outcome_ref = store.write_json_blob(
        "smoke/repository_outcome.json",
        outcome_document,
        role="structural_stage",
    )
    refs = [outcome_ref]
    for index, ensemble in enumerate(_outcome_ensembles(outcome)):
        refs.append(
            store.write_blob(
                f"smoke/maps/ensemble_{index:05d}.json",
                f"{ensemble.to_json()}\n".encode(),
                role="fixed_map_ensemble",
                logical_checksum=ensemble.content_checksum,
            )
        )
    circuit_binding = _stage_binding(outcome)
    selected = _stage_parameters(outcome)
    fresh_measured = _measure_call(
        lambda: _fresh_evaluate(
            store=store,
            resolved=resolved,
            circuit_binding=circuit_binding,
            parameters=tuple(float(value) for value in selected),
        )
    )
    fresh = cast("_FreshEvaluation", fresh_measured.value)
    refs.extend((fresh.ensemble_ref, fresh.trajectory_ref))
    resource_ref = store.write_json_blob(
        "runtime/resources.json",
        _runtime_resource_document(
            resolved=resolved,
            circuit_binding=circuit_binding,
            wall_time_seconds=measured.wall_time_seconds + fresh_measured.wall_time_seconds,
            peak_memory_bytes=max(measured.peak_memory_bytes, fresh_measured.peak_memory_bytes),
            normalized_work=float(runtime.training_trajectory_count + resolved.evaluation_policy.trajectory_count),
        ),
        role="runtime_resources",
    )
    refs.append(resource_ref)
    return _publish_attempt(
        store=store,
        resolved=resolved,
        artifact_kind="pipeline",
        status="success",
        blobs=refs,
        prefix_checksums=(_stage_outcome_checksum(outcome),),
        schedule_snapshot_ref=None,
        map_evidence_refs=(),
        raw_trajectory_ref=fresh.trajectory_ref,
        resource_ref=resource_ref,
        derived_metrics={
            **dict(fresh.metrics),
            "smoke_runtime_program_checksum": runtime.content_checksum,
            "repository_outcome_checksum": _stage_outcome_checksum(outcome),
            "training_trajectory_count": runtime.training_trajectory_count,
            "promotion_eligible": False,
        },
    )


def _operator_smoke_outcome(resolved: ResolvedProductionJob) -> OperatorGrowthSmokeExecution:
    """Execute the exact real projector or Energy-ADAPT bounded smoke callback."""
    runtime = resolved.executable_binding.smoke_runtime_program()
    if not isinstance(runtime, OperatorGrowthSmokeRuntimeProgram):
        msg = "Operator smoke dispatch resolved a non-growth runtime."
        raise TypeError(msg)
    if runtime.runner_family == "tfim_energy_adapt":
        return runtime.execute_energy(resolved.target, resolved.target_spec)
    return runtime.execute_projector(
        resolved.target,
        optimization_block_id=resolved.job.optimization_block_id,
        optimization_seed=resolved.job.optimization_seed,
        resource_stratum_id="primary_cap_12",
        trajectory_seed=resolved.scheduled_program.start_seed_bundles[0].training_trajectory_seed,
    )


def _execute_operator_smoke_attempt(
    resolved: ResolvedProductionJob,
    store: ProductionAttemptStore,
) -> ResultArtifactRef:
    """Persist one genuine bounded operator-family smoke execution."""
    runtime = resolved.executable_binding.smoke_runtime_program()
    measured = _measure_call(lambda: _operator_smoke_outcome(resolved))
    outcome = cast("OperatorGrowthSmokeExecution", measured.value)
    outcome_ref = store.write_json_blob(
        "smoke/repository_outcome.json",
        _typed_document("operator_growth_smoke", outcome.to_dict()),
        role="structural_stage",
    )
    resource_ref = store.write_json_blob(
        "runtime/resources.json",
        _runtime_resource_document(
            resolved=resolved,
            circuit_binding=None,
            wall_time_seconds=measured.wall_time_seconds,
            peak_memory_bytes=measured.peak_memory_bytes,
            normalized_work=float(outcome.work.total_sampled_trajectories),
        ),
        role="runtime_resources",
    )
    return _publish_attempt(
        store=store,
        resolved=resolved,
        artifact_kind="operator_growth",
        status="success",
        blobs=(outcome_ref, resource_ref),
        prefix_checksums=(outcome.content_checksum,),
        schedule_snapshot_ref=None,
        map_evidence_refs=(),
        raw_trajectory_ref=None,
        resource_ref=resource_ref,
        derived_metrics={
            "smoke_runtime_program_checksum": runtime.content_checksum,
            "repository_outcome_checksum": outcome.content_checksum,
            "selected_operator_count": len(outcome.selected_operator_ids),
            "trace_count": outcome.trace_count,
            "training_trajectory_count": outcome.work.total_sampled_trajectories,
            "promotion_eligible": False,
        },
    )


def _execute_operator_attempt(
    resolved: ResolvedProductionJob,
    store: ProductionAttemptStore,
) -> ResultArtifactRef:
    """Execute two exact noisy growth prefixes and fresh-evaluate the selected one."""
    payload = resolved.executable_binding.binding.implementation_artifact.implementation_payload
    if not isinstance(payload, OperatorGrowthExecutionSpec):
        msg = "Production operator growth requires its complete execution specification."
        raise TypeError(msg)
    binding_spec = resolved.executable_binding.binding.operator_growth_spec
    if not isinstance(binding_spec, OperatorGrowthExecutionSpec) or binding_spec != payload:
        msg = "Operator implementation payload differs from its scoped execution specification."
        raise ValueError(msg)
    if resolved.scheduled_program.total_updates_per_start != OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT:
        msg = "Operator growth must be governed by exactly 200 global scheduled updates."
        raise ValueError(msg)
    if OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT % payload.growth_spec.reoptimization_steps:
        msg = "The 200-update cap must contain complete operator reoptimization prefixes."
        raise ValueError(msg)
    numerics = _OperatorScheduledNumerics(
        store=store,
        execution_spec=payload,
        program_checksum=resolved.scheduled_program.content_checksum,
        target=resolved.target,
    )
    selector = _OperatorStructuralSelection(
        payload.pool,
        payload.growth_spec,
        numerics.register_selected_prefix,
    )
    initial = OperatorGrowthSegmentedSnapshot.initialize(resolved.scheduled_program)
    measured = _measure_call(
        lambda: execute_operator_growth_segmented_program(
            resolved.scheduled_program,
            initial,
            selector,
            numerics.objective,
            numerics.validate,
        )
    )
    snapshot = cast("OperatorGrowthSegmentedSnapshot", measured.value)
    if (
        not snapshot.complete
        or len(snapshot.receipts) != OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT
        or len(snapshot.transitions) != 2
        or len(snapshot.prefix_validations) != 2
        or snapshot.selected_prefix_index is None
        or not snapshot.selected_operator_ids
    ):
        msg = "Production operator growth did not complete its exact two-prefix schedule."
        raise RuntimeError(msg)
    expected_links = _operator_snapshot_numerical_links(snapshot)
    if len(numerics.map_blobs) != len(expected_links) or len(numerics.map_evidence_records) != len(expected_links):
        msg = "Operator numerical calls and persisted map/fidelity evidence are not one-to-one."
        raise ValueError(msg)
    for record, expected in zip(numerics.map_evidence_records, expected_links, strict=True):
        if (
            record.request_checksum != expected.request_checksum
            or record.numerical_result_checksum != expected.result_checksum
            or record.circuit_checksum != expected.circuit_checksum
            or not math.isclose(
                float(np.mean(record.trajectory_fidelities)),
                expected.expected_mean_fidelity,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            msg = "Operator in-memory numerical evidence differs from its authoritative snapshot."
            raise ValueError(msg)
    snapshot_ref = store.write_blob(
        "schedule/operator_growth_segmented_snapshot.json",
        f"{snapshot.to_json()}\n".encode(),
        role="schedule_snapshot",
        logical_checksum=snapshot.content_checksum,
    )
    selected_binding = _operator_circuit_binding(payload.pool, snapshot.selected_operator_ids)
    fresh_measured = _measure_call(
        lambda: _fresh_evaluate(
            store=store,
            resolved=resolved,
            circuit_binding=selected_binding,
            parameters=snapshot.selected_parameters,
        )
    )
    fresh = cast("_FreshEvaluation", fresh_measured.value)
    normalized_work = snapshot.total_normalized_work + resolved.evaluation_policy.trajectory_count
    resource_ref = store.write_json_blob(
        "runtime/resources.json",
        _runtime_resource_document(
            resolved=resolved,
            circuit_binding=selected_binding,
            wall_time_seconds=measured.wall_time_seconds + fresh_measured.wall_time_seconds,
            peak_memory_bytes=max(measured.peak_memory_bytes, fresh_measured.peak_memory_bytes),
            normalized_work=normalized_work,
        ),
        role="runtime_resources",
    )
    refs = [
        *numerics.map_blobs,
        *numerics.map_evidence_blobs,
        snapshot_ref,
        fresh.ensemble_ref,
        fresh.trajectory_ref,
        resource_ref,
    ]
    selected_validation = snapshot.prefix_validations[snapshot.selected_prefix_index]
    objective_count = sum(link.map_role == "training_trajectory" for link in expected_links)
    validation_count = sum(link.map_role == "checkpoint_validation" for link in expected_links)
    return _publish_attempt(
        store=store,
        resolved=resolved,
        artifact_kind="operator_growth",
        status="success",
        blobs=refs,
        prefix_checksums=(snapshot.content_checksum,),
        schedule_snapshot_ref=snapshot_ref,
        map_evidence_refs=tuple(numerics.map_evidence_blobs),
        raw_trajectory_ref=fresh.trajectory_ref,
        resource_ref=resource_ref,
        derived_metrics={
            **dict(fresh.metrics),
            "segmented_snapshot_checksum": snapshot.content_checksum,
            "transition_checksums": [transition.content_checksum for transition in snapshot.transitions],
            "selected_prefix_index": snapshot.selected_prefix_index,
            "selected_operator_ids": list(snapshot.selected_operator_ids),
            "selected_parameter_checksum": selected_validation.request.parameter_artifact.parameter_checksum,
            "selected_validation_fidelity": snapshot.selected_validation_score,
            "active_operator_ids": list(snapshot.active_operator_ids),
            "scheduled_objective_call_count": objective_count,
            "scheduled_validation_call_count": validation_count,
            "total_normalized_training_work": snapshot.total_normalized_work,
            "fresh_circuit_binding_checksum": selected_binding.content_checksum,
            "promotion_eligible": True,
        },
    )


class PersistedProductionAttemptError(RuntimeError):
    """Raised when an immutable terminal attempt records structured failure."""


def _artifact_kind_for_job(job: TrainingJob) -> Literal["pipeline", "operator_growth"]:
    """Resolve the only two repository-owned production executor families."""
    if job.implementation_kind == "phase2_pipeline":
        return "pipeline"
    if job.implementation_kind == "operator_growth":
        return "operator_growth"
    msg = f"Unsupported repository-owned implementation kind {job.implementation_kind!r}."
    raise ValueError(msg)


def _artifact_kind_for_binding(resolved: ResolvedProductionJob) -> Literal["pipeline", "operator_growth"]:
    """Resolve a repository family from an authenticated executable binding."""
    kind = resolved.executable_binding.binding.implementation_artifact.implementation_kind
    if kind.startswith("phase2_pipeline"):
        return "pipeline"
    if kind.startswith("operator_growth"):
        return "operator_growth"
    msg = f"Unsupported repository-owned implementation kind {kind!r}."
    raise ValueError(msg)


def _failure_partial_receipts(store: ProductionAttemptStore) -> tuple[float, Mapping[str, object]]:
    """Recover conservative completed work and custody facts from closed members."""
    refs = store.written_refs
    role_counts: dict[str, int] = {}
    for ref in refs:
        role_counts[ref.role] = role_counts.get(ref.role, 0) + 1
    components: list[dict[str, object]] = []
    runtime_values: list[float] = []
    accounted_paths: set[str] = set()
    for ref in refs:
        if ref.role == "runtime_resources":
            document = load_canonical_json_object(store.read_written_receipt(ref).decode())
            payload = require_mapping(document.get("payload"), "runtime resource payload")
            value = require_float(payload.get("normalized_work"), "normalized_work", minimum=0.0)
            runtime_values.append(value)
            accounted_paths.add(ref.path)
            components.append({"kind": "cumulative_runtime_resource", "path": ref.path, "work": value})
    if runtime_values:
        normalized_work = max(runtime_values)
        unavailable_roles: tuple[str, ...] = ()
        lower_bound = False
    else:
        incremental_values: list[float] = []
        for ref in refs:
            if ref.role == "schedule_snapshot":
                snapshot = _decode_schedule_snapshot(store.read_written_receipt(ref))
                value = (
                    snapshot.total_normalized_work
                    if isinstance(snapshot, OperatorGrowthSegmentedSnapshot)
                    else math.fsum(state.total_normalized_work for state in snapshot.states)
                )
                incremental_values.append(value)
                accounted_paths.add(ref.path)
                components.append({"kind": "scheduled_snapshot", "path": ref.path, "work": value})
            elif ref.role == "raw_trajectory_sidecar":
                document = load_canonical_json_object(store.read_written_receipt(ref).decode())
                payload = require_mapping(document.get("payload"), "raw trajectory payload")
                if "trajectory_count" in payload:
                    count = require_int(payload["trajectory_count"], "trajectory_count")
                    value = float(count)
                    incremental_values.append(value)
                    accounted_paths.add(ref.path)
                    components.append({"kind": "fresh_trajectory_vector", "path": ref.path, "work": value})
            elif ref.role == "scheduled_map_evidence":
                try:
                    map_evidence = ScheduledMapEvidence.from_dict(
                        load_canonical_json_object(store.read_written_receipt(ref).decode())
                    )
                except (TypeError, ValueError):
                    continue
                if map_evidence.numerical_result_checksum is None:
                    continue
                value = float(len(map_evidence.trajectory_fidelities))
                incremental_values.append(value)
                accounted_paths.add(ref.path)
                accounted_paths.update(ensemble_ref.path for ensemble_ref in map_evidence.ensemble_refs)
                components.append({"kind": "scheduled_map_fidelity_vector", "path": ref.path, "work": value})
            elif ref.role == "pilot_diagnostic_sidecar":
                try:
                    diagnostic = PilotDiagnosticEvidence.from_dict(
                        load_canonical_json_object(store.read_written_receipt(ref).decode())
                    )
                except (TypeError, ValueError):
                    continue
                value = float(len(diagnostic.member_seeds))
                incremental_values.append(value)
                accounted_paths.add(ref.path)
                components.append({"kind": "pilot_pathwise_vectors", "path": ref.path, "work": value})
        normalized_work = math.fsum(incremental_values)
        unavailable_roles = tuple(sorted({ref.role for ref in refs if ref.path not in accounted_paths}))
        lower_bound = bool(refs)
    summary: dict[str, object] = {
        "closed_artifact_count": len(refs),
        "closed_artifact_bytes": sum(ref.byte_count for ref in refs),
        "closed_role_counts": dict(sorted(role_counts.items())),
        "normalized_work_components": components,
        "normalized_work_unavailable": not components,
        "normalized_work_is_lower_bound": lower_bound,
        "unavailable_partial_work_roles": list(unavailable_roles),
    }
    return normalized_work, summary


def _publish_failure_attempt(
    *,
    resolved: ResolvedProductionJob,
    store: ProductionAttemptStore,
    error: Exception,
    elapsed_seconds: float,
    peak_memory_bytes: int,
) -> ResultArtifactRef:
    """Publish a redacted failure only when every already-written member is closed."""
    if any(ref.role == "production_evidence" for ref in store.written_refs):
        msg = "A partially published terminal evidence record makes this attempt incomplete."
        raise ValueError(msg)
    expected_before = tuple(sorted(ref.path for ref in store.written_refs))
    if store.member_paths() != expected_before:
        msg = "A partial or foreign member prevents terminal failure publication."
        raise ValueError(msg)
    normalized_work, partial_receipts = _failure_partial_receipts(store)
    resource_ref = store.write_json_blob(
        "runtime/failure_resources.json",
        _runtime_resource_document(
            resolved=resolved,
            circuit_binding=None,
            wall_time_seconds=max(float(elapsed_seconds), 0.0),
            peak_memory_bytes=max(peak_memory_bytes, 0),
            normalized_work=normalized_work,
            failure_phase="production_execution",
            partial_receipts=partial_receipts,
        ),
        role="runtime_resources",
    )
    existing_refs = store.written_refs
    schedule_refs = tuple(ref for ref in existing_refs if ref.role == "schedule_snapshot")
    map_refs = tuple(ref for ref in existing_refs if ref.role == "scheduled_map_evidence")
    return _publish_attempt(
        store=store,
        resolved=resolved,
        artifact_kind=_artifact_kind_for_binding(resolved),
        status="failure",
        blobs=existing_refs,
        prefix_checksums=tuple(ref.logical_checksum for ref in existing_refs if ref.role == "structural_stage"),
        schedule_snapshot_ref=None if not schedule_refs else schedule_refs[-1],
        map_evidence_refs=map_refs,
        raw_trajectory_ref=None,
        resource_ref=resource_ref,
        diagnostic_refs=(),
        derived_metrics={
            "failure_treatment": "structured_failure_zero_fidelity_for_intention_to_treat",
            "promotion_eligible": False,
        },
        failure={
            "phase": "production_execution",
            "exception_type": type(error).__name__,
            "message": "production executor failed; diagnostics are intentionally redacted",
        },
    )


def _dispatch_production_attempt(
    resolved: ResolvedProductionJob,
    store: ProductionAttemptStore,
    artifact_kind: Literal["pipeline", "operator_growth"],
) -> ResultArtifactRef:
    """Dispatch one already-authorized job to its exact repository execution path."""
    if artifact_kind == "pipeline":
        if resolved.job.preset == "training-smoke":
            return _execute_pipeline_smoke_attempt(resolved, store)
        return _execute_pipeline_attempt(resolved, store)
    if resolved.job.preset == "training-smoke":
        return _execute_operator_smoke_attempt(resolved, store)
    return _execute_operator_attempt(resolved, store)


class ProductionTrainingExecutor:
    """Context-bound typed executor for every frozen WP22 pipeline and growth job."""

    def __init__(self, context: TrainingExecutionContext) -> None:
        """Bind the exact non-serializable execution authority."""
        self.authority = ProductionExecutionAuthority(context)

    @staticmethod
    def reopen(reference: ResultArtifactRef, job_directory: Path) -> ReopenedProductionResult:
        """Reopen one result through complete immutable-manifest verification."""
        return reopen_result_artifact(reference, job_directory)

    def execute(
        self,
        job: TrainingJob,
        job_directory: Path,
        controls: JobExecutionControls,
    ) -> ResultArtifactRef:
        """Execute or reopen one exact context-owned production attempt."""
        if not isinstance(job_directory, Path):
            msg = "job_directory must be a pathlib.Path."
            raise TypeError(msg)
        if not isinstance(controls, JobExecutionControls):
            msg = "controls must be JobExecutionControls."
            raise TypeError(msg)
        resolved = self.authority.resolve(job)
        artifact_kind = _artifact_kind_for_job(job)
        attempt = _attempt_number(controls)
        store = ProductionAttemptStore(job_directory, job.content_checksum, attempt)
        if store.terminal_manifest_exists():
            reference = store.derive_existing_ref()
            reopened = reopen_result_artifact(reference, job_directory)
            if reopened.evidence.status == "failure":
                msg = "The exact immutable production attempt already records structured failure."
                raise PersistedProductionAttemptError(msg)
            return reference
        if store.attempt_directory_exists():
            msg = "An incomplete immutable production attempt already exists and cannot be overwritten."
            raise ValueError(msg)

        owns_tracing = not tracemalloc.is_tracing()
        if owns_tracing:
            tracemalloc.start()
            tracemalloc.reset_peak()
        baseline, _baseline_peak = tracemalloc.get_traced_memory()
        started = time.perf_counter()
        try:
            reference = _dispatch_production_attempt(resolved, store, artifact_kind)
        except Exception as error:
            current, peak = tracemalloc.get_traced_memory()
            measured_peak = max(current - baseline, peak - baseline, 0)
            try:
                _publish_failure_attempt(
                    resolved=resolved,
                    store=store,
                    error=error,
                    elapsed_seconds=time.perf_counter() - started,
                    peak_memory_bytes=measured_peak,
                )
            except Exception as custody_error:
                raise error from custody_error
            finally:
                if owns_tracing:
                    tracemalloc.stop()
            raise
        if owns_tracing:
            tracemalloc.stop()
        reopened = reopen_result_artifact(reference, job_directory)
        if reopened.evidence.status != "success":
            msg = "A successful production dispatch reopened a non-success terminal attempt."
            raise ValueError(msg)
        return reference


class ProductionConfirmationExecutor:
    """Final-seal-bound real executor reusing the frozen WP22 production paths."""

    def __init__(self, context: ConfirmationExecutionContext) -> None:
        """Bind the narrow non-serializable confirmation authority."""
        self.authority = ProductionConfirmationAuthority(context)

    def execute(
        self,
        request: ConfirmExecutionRequest,
        job_directory: Path,
        controls: JobExecutionControls,
    ) -> ResultArtifactRef:
        """Execute or reopen the authoritative first real confirmatory attempt."""
        if not isinstance(job_directory, Path):
            msg = "job_directory must be a pathlib.Path."
            raise TypeError(msg)
        if not isinstance(controls, JobExecutionControls):
            msg = "controls must be JobExecutionControls."
            raise TypeError(msg)
        if controls.overwrite:
            msg = "Real confirmation forbids overwrite; its first terminal attempt is authoritative."
            raise ValueError(msg)
        resolved = self.authority.resolve(request)
        store = ProductionAttemptStore(job_directory, request.content_checksum, 1)
        if store.terminal_manifest_exists():
            reference = store.derive_existing_ref()
            reopened = reopen_result_artifact(reference, job_directory)
            if reopened.evidence.status == "failure":
                msg = "The authoritative real confirmation attempt records structured failure."
                raise PersistedProductionAttemptError(msg)
            return reference
        if store.attempt_directory_exists():
            msg = "An incomplete authoritative real confirmation attempt already exists."
            raise ValueError(msg)
        state = controls.schedule_resume_state
        if state is not None and state.prior_attempt > 0:
            msg = "Real confirmation cannot create a later attempt after prior orchestration state."
            raise ValueError(msg)

        owns_tracing = not tracemalloc.is_tracing()
        if owns_tracing:
            tracemalloc.start()
            tracemalloc.reset_peak()
        baseline, _baseline_peak = tracemalloc.get_traced_memory()
        started = time.perf_counter()
        try:
            reference = _dispatch_production_attempt(
                resolved,
                store,
                _artifact_kind_for_binding(resolved),
            )
        except Exception as error:
            current, peak = tracemalloc.get_traced_memory()
            measured_peak = max(current - baseline, peak - baseline, 0)
            try:
                _publish_failure_attempt(
                    resolved=resolved,
                    store=store,
                    error=error,
                    elapsed_seconds=time.perf_counter() - started,
                    peak_memory_bytes=measured_peak,
                )
            except Exception as custody_error:
                raise error from custody_error
            finally:
                if owns_tracing:
                    tracemalloc.stop()
            raise
        if owns_tracing:
            tracemalloc.stop()
        reopened = reopen_result_artifact(reference, job_directory)
        if reopened.evidence.status != "success":
            msg = "A successful real confirmation dispatch reopened a non-success terminal attempt."
            raise ValueError(msg)
        return reference


@dataclass(frozen=True, slots=True)
class SyntheticConfirmationFixture:
    """Checksum-bound raw fidelities for exercising dormant confirmation only."""

    request_checksum: str
    trajectory_fidelities: tuple[float, ...]
    schema_version: str = field(default=SYNTHETIC_CONFIRM_FIXTURE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate an exact request binding and bounded raw fidelity vector."""
        object.__setattr__(self, "request_checksum", require_checksum(self.request_checksum, "request_checksum"))
        values = tuple(
            require_float(value, "trajectory_fidelity", minimum=0.0, maximum=1.0)
            for value in self.trajectory_fidelities
        )
        if len(values) < 2:
            msg = "Synthetic confirmation fixtures require at least two trajectory fidelities."
            raise ValueError(msg)
        object.__setattr__(self, "trajectory_fidelities", values)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_checksum": self.request_checksum,
            "trajectory_fidelities": list(self.trajectory_fidelities),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum binding every synthetic value to one exact confirm request."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed fixture data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> SyntheticConfirmationFixture:
        """Decode and verify one synthetic confirmation fixture."""
        mapping = verify_sealed_mapping(
            value,
            expected_keys=_SYNTHETIC_FIXTURE_KEYS,
            name="synthetic confirmation fixture",
        )
        if mapping["schema_version"] != SYNTHETIC_CONFIRM_FIXTURE_SCHEMA_VERSION:
            msg = "Synthetic confirmation fixture uses an unsupported schema version."
            raise ValueError(msg)
        fixture = cls(
            request_checksum=cast("str", mapping["request_checksum"]),
            trajectory_fidelities=cast("tuple[float, ...]", mapping["trajectory_fidelities"]),
        )
        if mapping["content_checksum"] != fixture.content_checksum:
            msg = "Synthetic confirmation fixture checksum changed during normalization."
            raise ValueError(msg)
        return fixture


class SyntheticConfirmationExecutor:
    """Dormant typed confirmation path that never opens a held target manifest."""

    def __init__(self, fixture: SyntheticConfirmationFixture) -> None:
        """Bind exactly one sealed synthetic request fixture."""
        if not isinstance(fixture, SyntheticConfirmationFixture):
            msg = "fixture must be a SyntheticConfirmationFixture."
            raise TypeError(msg)
        self.fixture = fixture

    def execute(
        self,
        request: ConfirmExecutionRequest,
        job_directory: Path,
        controls: JobExecutionControls,
    ) -> ResultArtifactRef:
        """Publish or reopen the authoritative first synthetic confirm attempt."""
        if not isinstance(request, ConfirmExecutionRequest):
            msg = "request must be a ConfirmExecutionRequest."
            raise TypeError(msg)
        if not isinstance(job_directory, Path):
            msg = "job_directory must be a pathlib.Path."
            raise TypeError(msg)
        if not isinstance(controls, JobExecutionControls):
            msg = "controls must be JobExecutionControls."
            raise TypeError(msg)
        if request.content_checksum != self.fixture.request_checksum:
            msg = "Synthetic fixture is not bound to this exact confirm execution request."
            raise ValueError(msg)
        if len(self.fixture.trajectory_fidelities) != request.fixed_test_trajectory_count:
            msg = "Synthetic fixture count differs from the sealed fixed confirmatory test count."
            raise ValueError(msg)
        if controls.overwrite:
            msg = "Synthetic confirmation forbids overwrite; its first terminal attempt is authoritative."
            raise ValueError(msg)

        store = ProductionAttemptStore(job_directory, request.content_checksum, 1)
        if store.terminal_manifest_exists():
            reference = store.derive_existing_ref()
            reopened = reopen_result_artifact(reference, job_directory)
            if reopened.evidence.status == "failure":
                msg = "The authoritative synthetic confirmation attempt records failure."
                raise PersistedProductionAttemptError(msg)
            return reference
        if store.attempt_directory_exists():
            msg = "An incomplete authoritative synthetic confirmation attempt already exists."
            raise ValueError(msg)
        state = controls.schedule_resume_state
        if state is not None and state.prior_attempt > 0:
            msg = "Synthetic confirmation cannot create a later attempt after prior orchestration state."
            raise ValueError(msg)

        evaluation_policy_checksum = confirmatory_evaluation_policy_checksum(request)
        raw_ref = store.write_json_blob(
            "evaluation/raw_trajectory_fidelities.json",
            _typed_document(
                "raw_trajectory_fidelities",
                {
                    "request_checksum": request.content_checksum,
                    "evaluation_policy_checksum": evaluation_policy_checksum,
                    "data_role": "confirmatory",
                    "seed_domain": "confirmatory_test",
                    "evaluation_seed": request.evaluation_seed,
                    "trajectory_count": request.fixed_test_trajectory_count,
                    "trajectory_fidelities": list(self.fixture.trajectory_fidelities),
                    "synthetic_fixture_checksum": self.fixture.content_checksum,
                },
            ),
            role="raw_trajectory_sidecar",
        )
        resource_ref = store.write_json_blob(
            "runtime/resources.json",
            _typed_document(
                "runtime_resources",
                {
                    "request_checksum": request.content_checksum,
                    "source_fingerprint_checksum": request.execution_source_checksum,
                    "wall_time_seconds": 0.0,
                    "peak_memory_bytes": 0,
                    "normalized_work": 0.0,
                    "synthetic_fixture": True,
                    "circuit": None,
                },
            ),
            role="runtime_resources",
        )
        mean_fidelity = float(np.mean(self.fixture.trajectory_fidelities))
        evidence = ProductionNumericalEvidence(
            job_checksum=request.content_checksum,
            attempt=1,
            artifact_kind="synthetic_confirmation",
            status="success",
            execution_source_manifest_checksum=request.execution_source_checksum,
            source_fingerprint_checksum=request.execution_source_checksum,
            executable_binding_checksum=request.executable_binding_checksum,
            scheduled_program_checksum=request.hyperparameters_checksum,
            target_identity={
                "synthetic_fixture": True,
                "request_checksum": request.content_checksum,
                "target_instance_id": request.target_instance_id,
                "target_spec_checksum": request.target_spec_checksum,
                "qubit_count": request.qubit_count,
            },
            evaluation_policy_checksum=evaluation_policy_checksum,
            structural_prefix_checksums=(),
            schedule_snapshot_ref=None,
            map_evidence_refs=(),
            diagnostic_refs=(),
            raw_trajectory_ref=raw_ref,
            resource_ref=resource_ref,
            derived_metrics={
                "noisy_fidelity": mean_fidelity,
                "evaluation_data_role": "confirmatory",
                "evaluation_seed_domain": "confirmatory_test",
                "evaluation_seed": request.evaluation_seed,
                "trajectory_count": request.fixed_test_trajectory_count,
                "synthetic_fixture_checksum": self.fixture.content_checksum,
                "strategy_schedule_checksum": request.hyperparameters_checksum,
                "promotion_eligible": False,
            },
            failure=None,
        )
        evidence_ref = store.write_json_blob(
            "production_evidence.json",
            evidence.to_dict(),
            role="production_evidence",
        )
        reference = store.publish(
            artifact_kind="synthetic_confirmation",
            status="success",
            execution_source_manifest_checksum=request.execution_source_checksum,
            source_fingerprint_checksum=request.execution_source_checksum,
            blobs=(raw_ref, resource_ref, evidence_ref),
            evidence_ref=evidence_ref,
        )
        reopen_result_artifact(reference, job_directory)
        return reference


def create_synthetic_confirmation_executor(
    fixture: SyntheticConfirmationFixture,
) -> Callable[[ConfirmExecutionRequest, Path, JobExecutionControls], str]:
    """Create the legacy-ABI wrapper for one sealed synthetic confirmation fixture."""
    executor = SyntheticConfirmationExecutor(fixture)

    def execute_and_verify(
        request: ConfirmExecutionRequest,
        directory: Path,
        controls: JobExecutionControls,
    ) -> str:
        reference = executor.execute(request, directory, controls)
        reopened = reopen_result_artifact(reference, directory)
        if reopened.evidence.status != "success":
            msg = "Synthetic confirmation did not reopen as a successful typed attempt."
            raise PersistedProductionAttemptError(msg)
        return reference.content_checksum

    return execute_and_verify


def create_default_training_executor_registry(
    context: TrainingExecutionContext | ConfirmationExecutionContext,
    *,
    synthetic_confirmation_fixture: SyntheticConfirmationFixture | None = None,
) -> TrainingExecutorRegistry:
    """Create the repository-owned registry while preserving the legacy string ABI."""
    if isinstance(context, ConfirmationExecutionContext):
        if synthetic_confirmation_fixture is not None:
            msg = "A real confirmation context cannot accept a synthetic fixture."
            raise ValueError(msg)
        confirmation = ProductionConfirmationExecutor(context)

        def execute_real_confirmation(
            request: ConfirmExecutionRequest,
            directory: Path,
            controls: JobExecutionControls,
        ) -> str:
            reference = confirmation.execute(request, directory, controls)
            reopened = reopen_result_artifact(reference, directory)
            if reopened.evidence.status != "success":
                msg = "Real confirmation did not reopen as a successful typed attempt."
                raise PersistedProductionAttemptError(msg)
            return reference.content_checksum

        return TrainingExecutorRegistry(confirm_executor=execute_real_confirmation)
    if not isinstance(context, TrainingExecutionContext):
        msg = "context must be a TrainingExecutionContext or ConfirmationExecutionContext."
        raise TypeError(msg)
    executor = ProductionTrainingExecutor(context)

    def reject_unheld_confirmation(
        request: ConfirmExecutionRequest,
        directory: Path,
        controls: JobExecutionControls,
    ) -> str:
        if not isinstance(request, ConfirmExecutionRequest):
            msg = "request must be a ConfirmExecutionRequest."
            raise TypeError(msg)
        if not isinstance(directory, Path) or not isinstance(controls, JobExecutionControls):
            msg = "Confirmation dispatch requires a pathlib directory and typed controls."
            raise TypeError(msg)
        msg = "Real held confirmation is dormant; supply an exact synthetic fixture in WP22."
        raise ValueError(msg)

    def execute_and_verify(
        job: TrainingJob,
        directory: Path,
        controls: JobExecutionControls,
    ) -> str:
        reference = executor.execute(job, directory, controls)
        reopened = executor.reopen(reference, directory)
        if reopened.evidence.status != "success":
            msg = "The registry cannot project a failed typed result into a success checksum."
            raise PersistedProductionAttemptError(msg)
        return reference.content_checksum

    return TrainingExecutorRegistry(
        phase2_pipeline_executor=execute_and_verify,
        operator_growth_executor=execute_and_verify,
        confirm_executor=(
            reject_unheld_confirmation
            if synthetic_confirmation_fixture is None
            else create_synthetic_confirmation_executor(synthetic_confirmation_fixture)
        ),
    )
