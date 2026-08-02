# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Immutable artifacts and canonical result storage for Phase II pipelines.

This module is deliberately separate from the Phase I report store.  A
checksum-sealed JSONL stage ledger is the authority for resumability; derived
manifests and CSV files can therefore be rebuilt after interruption without
replaying a successfully committed stage.
"""

# The public records carry strict validation contracts; repeating every
# validator-raised exception in this persistence module would obscure the
# stage-commit and resume protocol.
# ruff: noqa: DOC201, DOC501

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import math
import shutil
import statistics
import traceback as traceback_module
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from filelock import FileLock, Timeout

from benchmarks.state_preparation.constants import NOISELESS_NOISE_ID
from benchmarks.state_preparation.reporting import atomic_write_bytes
from mqt.yaqs.optimization import KrotovFixedMapEnsemble

from .artifact_codecs import (
    StageParameterCheckpoint,
    create_phase2_trajectory_sidecar,
    read_phase2_trajectory_sidecar,
)
from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    seal_mapping,
    thaw_json,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .noisy_krotov import (
    NOISY_KROTOV_ADAPTER_VERSION,
    NOISY_KROTOV_EXECUTION_SCHEMA_VERSION,
    NOISY_KROTOV_TRACE_SCHEMA_VERSION,
    KrotovWorkLedger,
    NoisyKrotovCheckpointSelection,
    NoisyKrotovIterationRecord,
    NoisyKrotovObjectiveBinding,
    NoisyKrotovResumeState,
    NoisyKrotovStageExecution,
    NoisyKrotovStageFailure,
    noisy_krotov_computational_zero_state_checksum,
    validate_noisy_krotov_execution_trace,
)
from .pipeline import (
    EXTERNAL_CHECKPOINT_REF_SCHEMA_VERSION,
    PIPELINE_CSV_COLUMNS,
    PipelineBenchmarkFailure,
    PipelineBenchmarkResult,
    PipelineEvaluationConfig,
    TrainingPipelineConfig,
    TrainingPipelineResult,
    TrainingStageConfig,
    TrainingStageResult,
    pipeline_benchmark_record_from_json,
)
from .resumability import (
    NonScientificResumeOverride,
    ResumabilityFingerprint,
    require_resumability_match,
)
from .validation import (
    require_bool,
    require_checksum,
    require_exact_keys,
    require_float,
    require_int,
    require_mapping,
    require_nonempty_text,
    require_relative_path,
    require_slug,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray

    from .noisy_krotov import (
        UpdateSignalKind,
    )
    from .pipeline import (
        PipelineBenchmarkRecord,
    )

PHASE2_ARTIFACT_MANIFEST_FORMAT = "yaqs.state_preparation.phase2.artifact_manifest.v1"
PHASE2_STAGE_ARTIFACT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.stage_artifact.v1"
PHASE2_STAGE_FAILURE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.stage_failure.v1"
PHASE2_MATERIALIZATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.materialized_circuit.v1"
PHASE2_MATERIALIZATION_ATTEMPT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.materialization_attempt.v1"
PHASE2_EVALUATION_EVIDENCE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.evaluation_evidence.v1"
PHASE2_STAGE_METADATA_SCHEMA_VERSION = "yaqs.state_preparation.phase2.stage_metadata.v2"
PHASE2_TRACE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.optimizer_trace.v1"

PIPELINE_CONFIG_NAME = "pipeline.json"
RESUMABILITY_FINGERPRINT_NAME = "resumability.json"
RESUME_OVERRIDE_STREAM_NAME = "resume_overrides.jsonl"
STAGE_RESULT_STREAM_NAME = "stage_results.jsonl"
STAGE_FAILURE_STREAM_NAME = "stage_failures.jsonl"
MATERIALIZATION_STREAM_NAME = "materializations.jsonl"
MATERIALIZATION_ATTEMPT_STREAM_NAME = "materialization_attempts.jsonl"
RESULTS_JSONL_NAME = "results.jsonl"
RESULTS_CSV_NAME = "results.csv"
EVALUATION_FAILURE_STREAM_NAME = "evaluation_failures.jsonl"
EVALUATION_EVIDENCE_STREAM_NAME = "evaluation_evidence.jsonl"
MANIFEST_NAME = "manifest.json"

_STORE_LOCK_NAME = ".phase2-artifact-store.lock"
_STORE_LOCK_TIMEOUT_SECONDS = 30.0

CHECKPOINT_DIRECTORY = "checkpoints"
TRACE_DIRECTORY = "traces"
STAGE_METADATA_DIRECTORY = "stage_metadata"
FIXED_MAP_DIRECTORY = "fixed_maps"
CIRCUIT_DIRECTORY = "circuits"
TRAJECTORY_DIRECTORY = "trajectories"

_MAX_CHECKPOINT_SIZE = 64 * 1024 * 1024
_MAX_TRACE_SIZE = 256 * 1024 * 1024
_MAX_STAGE_METADATA_SIZE = 16 * 1024 * 1024
_MAX_FIXED_MAP_SIZE = 512 * 1024 * 1024
_MAX_CIRCUIT_SIZE = 512 * 1024 * 1024

_MANAGED_DIRECTORIES = (
    CHECKPOINT_DIRECTORY,
    TRACE_DIRECTORY,
    STAGE_METADATA_DIRECTORY,
    FIXED_MAP_DIRECTORY,
    CIRCUIT_DIRECTORY,
    TRAJECTORY_DIRECTORY,
)
_MANAGED_ROOT_FILES = (
    PIPELINE_CONFIG_NAME,
    RESUMABILITY_FINGERPRINT_NAME,
    RESUME_OVERRIDE_STREAM_NAME,
    STAGE_RESULT_STREAM_NAME,
    STAGE_FAILURE_STREAM_NAME,
    MATERIALIZATION_STREAM_NAME,
    MATERIALIZATION_ATTEMPT_STREAM_NAME,
    RESULTS_JSONL_NAME,
    RESULTS_CSV_NAME,
    EVALUATION_FAILURE_STREAM_NAME,
    EVALUATION_EVIDENCE_STREAM_NAME,
    MANIFEST_NAME,
)
_CANONICAL_LEDGER_NAMES = (
    RESUME_OVERRIDE_STREAM_NAME,
    STAGE_RESULT_STREAM_NAME,
    STAGE_FAILURE_STREAM_NAME,
    MATERIALIZATION_STREAM_NAME,
    MATERIALIZATION_ATTEMPT_STREAM_NAME,
    RESULTS_JSONL_NAME,
    EVALUATION_FAILURE_STREAM_NAME,
    EVALUATION_EVIDENCE_STREAM_NAME,
)
_STAGE_ARTIFACT_KEYS = frozenset({
    "schema_version",
    "stage_result",
    "runtime_fingerprint_checksum",
    "checkpoint_file_checksum",
    "trace_file_checksum",
    "metadata_file_checksum",
    "fixed_map_artifacts",
    "content_checksum",
})
_FIXED_MAP_REF_KEYS = frozenset({
    "role",
    "ensemble_id",
    "content_checksum",
    "path",
    "file_checksum",
})
_STAGE_FAILURE_KEYS = frozenset({
    "schema_version",
    "failure_id",
    "pipeline_training_id",
    "pipeline_configuration_checksum",
    "pipeline_prefix_id",
    "stage_index",
    "stage_id",
    "stage_configuration_checksum",
    "phase",
    "exception_type",
    "message",
    "traceback",
    "retryable",
    "attempt",
    "partial_work",
    "completed_stage_artifact_checksums",
    "wall_time_seconds",
    "runtime_fingerprint_checksum",
    "content_checksum",
})
_MATERIALIZATION_KEYS = frozenset({
    "schema_version",
    "materialized_circuit_id",
    "pipeline_training_id",
    "pipeline_result_checksum",
    "final_checkpoint_checksum",
    "materialization_policy_checksum",
    "path",
    "payload_checksum",
    "wall_time_seconds",
    "peak_memory_bytes",
    "runtime_fingerprint_checksum",
    "content_checksum",
})
_MATERIALIZATION_ATTEMPT_KEYS = frozenset({
    "schema_version",
    "attempt_id",
    "materialized_circuit_id",
    "pipeline_training_id",
    "pipeline_result_checksum",
    "attempt",
    "status",
    "phase",
    "payload_checksum",
    "exception_type",
    "message",
    "wall_time_seconds",
    "peak_memory_bytes",
    "runtime_fingerprint_checksum",
    "content_checksum",
})
_EVALUATION_EVIDENCE_KEYS = frozenset({
    "schema_version",
    "evaluation_row_id",
    "record_checksum",
    "pipeline_result_checksum",
    "materialization_checksum",
    "evaluation_provider_checksum",
    "evaluation_map_artifacts",
    "content_checksum",
})
_TRACE_KEYS = frozenset({
    "schema_version",
    "pipeline_training_id",
    "pipeline_prefix_id",
    "stage_configuration_checksum",
    "trace",
    "optimizer_state",
    "content_checksum",
})
_STAGE_METADATA_KEYS = frozenset({
    "schema_version",
    "pipeline_training_id",
    "pipeline_prefix_id",
    "stage_configuration_checksum",
    "circuit_binding_checksum",
    "provider_checksum",
    "objective_checksum",
    "objective_binding",
    "source_parameter_checksum",
    "initial_parameter_checksum",
    "final_parameter_checksum",
    "selected_parameter_checksum",
    "selected_global_iteration",
    "completed_global_iteration",
    "selected_checkpoint_validation_fidelity",
    "circuit_topology",
    "circuit_statistics",
    "training_map_artifacts",
    "checkpoint_validation_map_artifacts",
    "checkpoint_validation_provider_checksum",
    "cumulative_cross_trajectory_pairings",
    "runtime_fingerprint_checksum",
    "content_checksum",
})
_MANIFEST_KEYS = frozenset({
    "manifest_format",
    "pipeline_training_id",
    "pipeline_configuration_checksum",
    "completed_stage_count",
    "completed_pipeline_result_checksum",
    "active_runtime_fingerprint_checksum",
    "resume_override_checksums",
    "canonical_stage_stream",
    "stage_failure_stream",
    "canonical_result_stream",
    "evaluation_failure_stream",
    "derived_csv",
    "materialization_stream",
    "materialization_attempt_stream",
    "evaluation_evidence_stream",
    "completed_stage_artifact_checksums",
    "stage_failure_ids",
    "stage_failure_checksums",
    "materialization_checksums",
    "materialization_attempt_checksums",
    "evaluation_record_index",
    "evaluation_failure_attempt_checksums",
    "evaluation_evidence_checksums",
    "successful_evaluation_row_ids",
    "failed_evaluation_row_ids",
    "record_count",
    "artifact_inventory",
    "canonical_stream_checksums",
    "timing_convention",
    "timing",
    "content_checksum",
})
_MAP_ROLES = frozenset({
    "training_trajectory",
    "checkpoint_validation",
    "pilot_evaluation",
    "screening_selection",
    "confirmatory_test",
})
_NOISY_TRACE_ROW_KEYS = frozenset({
    "schema_version",
    "local_iteration",
    "global_iteration",
    "parameter_checksum",
    "learning_rate",
    "monitoring_loss",
    "monitoring_fidelity",
    "checkpoint_validation_fidelity",
    "update_signal",
    "update_signal_kind",
    "update_signal_norm",
    "gradient_norm",
    "cross_dense_sum_norm",
    "update_norm",
    "trajectory_count",
    "nonidentity_events",
    "training_ensemble_id",
    "training_ensemble_checksum",
    "checkpoint_validation_ensemble_checksum",
    "cumulative_work",
    "training_ensemble_sampled",
    "checkpoint_validation_ensemble_sampled",
    "cross_trajectory_pairings",
    "cumulative_cross_trajectory_pairings",
})


class Phase2ArtifactError(RuntimeError):
    """Base class for Phase II persistence failures."""


class Phase2ConcurrentMutationError(Phase2ArtifactError):
    """Raised when another writer changed or currently owns the artifact store."""


class Phase2ResumeMismatchError(Phase2ArtifactError):
    """Raised when stored work is not scientifically resumable."""


class Phase2ArtifactVerificationError(Phase2ArtifactError):
    """Raised when a referenced artifact is missing or corrupt."""


class Phase2DuplicateRecordError(Phase2ArtifactError):
    """Raised when an immutable scientific identifier is written twice."""


class Phase2DerivedArtifactError(Phase2ArtifactError):
    """Raised after a canonical commit when rebuilding a derived view fails."""


def _sha256(payload: bytes) -> str:
    """Return a canonical checksum for exact file bytes."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _require_artifact_size(payload: bytes, maximum_size: int, name: str) -> None:
    """Reject payloads that a later verified reopen would refuse to read."""
    if len(payload) > maximum_size:
        msg = f"{name} exceeds its configured verification limit."
        raise Phase2ArtifactVerificationError(msg)


def validate_materialized_circuit_payload(payload: object) -> bytes:
    """Return exact circuit bytes only when the artifact store can reopen them."""
    if type(payload) is not bytes:
        msg = "Materialized circuit payload must be exact bytes."
        raise TypeError(msg)
    _require_artifact_size(payload, _MAX_CIRCUIT_SIZE, "Materialized circuit")
    return payload


def _empty_work() -> dict[str, int]:
    """Return a new zero-valued normalized-work ledger."""
    return {
        "objective_evaluations": 0,
        "gradient_evaluations": 0,
        "training_trajectories": 0,
        "checkpoint_validation_trajectories": 0,
        "test_trajectories": 0,
        "trajectory_gate_applications": 0,
    }


def _sum_work(items: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Return the component-wise sum of normalized-work mappings."""
    total = _empty_work()
    for item in items:
        if set(item) != set(total):
            msg = "normalized_work fields do not match the Phase II work ledger."
            raise ValueError(msg)
        for key, value in item.items():
            total[key] += require_int(value, f"normalized_work.{key}")
    return total


def _map_schedule_coordinates(
    policy: str,
    refresh_interval: int | None,
    point_count: int,
) -> tuple[tuple[int, int, int], ...]:
    """Return exact ensemble, refresh, and window coordinates for a map schedule."""
    if point_count == 0 or policy == "none":
        return ()
    if policy == "crn_fixed":
        return ((0, 0, 0),)
    if policy == "resampled":
        return tuple((index, index, index) for index in range(point_count))
    if policy == "crn_refresh":
        interval = cast("int", refresh_interval)
        return tuple((index, index, index * interval) for index in range(math.ceil(point_count / interval)))
    msg = f"Unsupported Phase II fixed-map schedule {policy!r}."
    raise ValueError(msg)


def _immutable_vector(value: object, name: str) -> NDArray[np.float64]:
    """Validate and detach one finite parameter vector."""
    try:
        vector = np.asarray(value, dtype=np.dtype("<f8"))
    except (TypeError, ValueError) as error:
        msg = f"{name} must be convertible to float64."
        raise TypeError(msg) from error
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        msg = f"{name} must be a finite one-dimensional parameter vector."
        raise ValueError(msg)
    return np.ascontiguousarray(vector).copy()


def _vector_checksum(value: NDArray[np.float64]) -> str:
    """Checksum canonical little-endian vector bytes."""
    return _sha256(np.ascontiguousarray(value, dtype=np.dtype("<f8")).tobytes())


def _validated_circuit_topology(
    value: object,
    *,
    stage: TrainingStageConfig,
    circuit_binding_checksum: str | None,
) -> Mapping[str, object]:
    """Verify and freeze one complete checksum-sealed circuit topology."""
    if circuit_binding_checksum is None:
        msg = "A persisted stage requires a checksum-bound circuit topology."
        raise ValueError(msg)
    topology = freeze_json_mapping(value, "circuit_topology")
    supplied = require_checksum(topology.get("content_checksum"), "circuit_topology.content_checksum")
    payload = {key: thaw_json(item) for key, item in topology.items() if key != "content_checksum"}
    if supplied != circuit_binding_checksum or canonical_checksum(payload) != circuit_binding_checksum:
        msg = "circuit_topology does not reproduce circuit_binding_checksum."
        raise ValueError(msg)
    if topology.get("topology_id") != stage.output_topology_id or topology.get("num_params") != (
        stage.output_parameter_count
    ):
        msg = "circuit_topology does not match the configured output topology and parameter count."
        raise ValueError(msg)
    require_int(topology.get("num_qubits"), "circuit_topology.num_qubits", minimum=1)
    gates = topology.get("gates")
    if type(gates) is not tuple:
        msg = "circuit_topology.gates must be a serialized gate sequence."
        raise TypeError(msg)
    if not all(isinstance(gate, Mapping) for gate in gates):
        msg = "circuit_topology.gates must contain only gate mappings."
        raise TypeError(msg)
    return topology


def _decode_noisy_trace_rows(value: object) -> tuple[NoisyKrotovIterationRecord, ...]:
    """Decode strict WP17 trace rows and re-run their semantic validators."""
    if type(value) is not tuple:
        msg = "WP17 optimizer trace must be a serialized sequence."
        raise TypeError(msg)
    records: list[NoisyKrotovIterationRecord] = []
    for index, raw_row in enumerate(value):
        row = require_mapping(raw_row, f"trace[{index}]")
        require_exact_keys(row, _NOISY_TRACE_ROW_KEYS, f"trace[{index}]")
        if row["schema_version"] != NOISY_KROTOV_TRACE_SCHEMA_VERSION:
            msg = f"trace[{index}] uses an unsupported WP17 schema."
            raise ValueError(msg)
        cumulative = _sum_work((require_mapping(row["cumulative_work"], f"trace[{index}].cumulative_work"),))
        update_signal = row["update_signal"]
        if type(update_signal) is not tuple:
            msg = f"trace[{index}].update_signal must be a serialized sequence."
            raise TypeError(msg)
        record = NoisyKrotovIterationRecord(
            local_iteration=cast("int", row["local_iteration"]),
            global_iteration=cast("int", row["global_iteration"]),
            parameter_checksum=cast("str", row["parameter_checksum"]),
            learning_rate=cast("float", row["learning_rate"]),
            monitoring_loss=cast("float", row["monitoring_loss"]),
            monitoring_fidelity=cast("float", row["monitoring_fidelity"]),
            checkpoint_validation_fidelity=cast("float | None", row["checkpoint_validation_fidelity"]),
            update_signal=cast("tuple[float, ...]", update_signal),
            update_signal_kind=cast("UpdateSignalKind", row["update_signal_kind"]),
            update_signal_norm=cast("float", row["update_signal_norm"]),
            gradient_norm=cast("float | None", row["gradient_norm"]),
            cross_dense_sum_norm=cast("float | None", row["cross_dense_sum_norm"]),
            update_norm=cast("float", row["update_norm"]),
            trajectory_count=cast("int", row["trajectory_count"]),
            nonidentity_events=cast("int", row["nonidentity_events"]),
            training_ensemble_id=cast("str | None", row["training_ensemble_id"]),
            training_ensemble_checksum=cast("str | None", row["training_ensemble_checksum"]),
            checkpoint_validation_ensemble_checksum=cast(
                "str | None",
                row["checkpoint_validation_ensemble_checksum"],
            ),
            cumulative_work=KrotovWorkLedger(**cumulative),
            training_ensemble_sampled=cast("bool", row["training_ensemble_sampled"]),
            checkpoint_validation_ensemble_sampled=cast(
                "bool",
                row["checkpoint_validation_ensemble_sampled"],
            ),
            cross_trajectory_pairings=cast("int", row["cross_trajectory_pairings"]),
            cumulative_cross_trajectory_pairings=cast(
                "int",
                row["cumulative_cross_trajectory_pairings"],
            ),
        )
        if canonical_checksum(record.to_dict()) != canonical_checksum(row):
            msg = f"trace[{index}] changes during WP17 semantic normalization."
            raise ValueError(msg)
        records.append(record)
    if not records:
        msg = "WP17 optimizer trace must not be empty."
        raise ValueError(msg)
    return tuple(records)


def _validate_noisy_trace_semantics(
    *,
    stage: TrainingStageConfig,
    trace: object,
    training_ensembles: Sequence[KrotovFixedMapEnsemble],
    validation_ensembles: Sequence[KrotovFixedMapEnsemble],
    normalized_work: Mapping[str, object],
    initial_parameter_checksum: str,
    final_parameter_checksum: str,
    selected_parameter_checksum: str,
    completed_iteration: int,
    selected_iteration: int,
    selected_fidelity: float | None,
    cumulative_pairings: int,
    circuit_topology: Mapping[str, object],
    provider_checksum: str | None,
) -> tuple[NoisyKrotovIterationRecord, ...]:
    """Verify trace progress, work, maps, and checkpoint-selection semantics."""
    records = _decode_noisy_trace_rows(trace)
    global_iterations = tuple(row.global_iteration for row in records)
    local_iterations = tuple(row.local_iteration for row in records)
    expected_iterations = tuple(range(stage.iteration_budget + 1))
    if global_iterations != expected_iterations or local_iterations != expected_iterations:
        msg = "WP17 trace must contain every global and local stage iteration exactly once."
        raise ValueError(msg)
    if records[-1].global_iteration != completed_iteration:
        msg = "WP17 trace does not end at the completed stage iteration."
        raise ValueError(msg)
    selected_records = tuple(row for row in records if row.global_iteration == selected_iteration)
    if (
        records[0].parameter_checksum != initial_parameter_checksum
        or records[-1].parameter_checksum != final_parameter_checksum
        or len(selected_records) != 1
        or selected_records[0].parameter_checksum != selected_parameter_checksum
    ):
        msg = "WP17 trace parameter checksums do not bind the initial, selected, and final states."
        raise ValueError(msg)
    exact_work = KrotovWorkLedger(**_sum_work((normalized_work,)))
    validate_noisy_krotov_execution_trace(
        stage=stage,
        circuit_binding_document=circuit_topology,
        provider_checksum=provider_checksum,
        trace=records,
        training_ensembles=training_ensembles,
        validation_ensembles=validation_ensembles,
        normalized_work=exact_work,
        input_resume_state=None,
    )
    if records[-1].cumulative_cross_trajectory_pairings != cumulative_pairings:
        msg = "WP17 trace-derived cross-trajectory pairings do not match stage evidence."
        raise ValueError(msg)

    validation_records = tuple(row for row in records if row.checkpoint_validation_fidelity is not None)
    if stage.checkpoint_validation.enabled:
        if not validation_records or selected_fidelity is None:
            msg = "Enabled checkpoint validation requires trace-backed selection evidence."
            raise ValueError(msg)
        cadence = cast("int", stage.checkpoint_validation.cadence)
        expected_validation_iterations = tuple(
            sorted({0, stage.iteration_budget, *range(cadence, stage.iteration_budget, cadence)})
        )
        if tuple(row.global_iteration for row in validation_records) != expected_validation_iterations:
            msg = "WP17 trace does not contain the exact configured checkpoint-validation cadence."
            raise ValueError(msg)
        best = max(cast("float", row.checkpoint_validation_fidelity) for row in validation_records)
        candidates = tuple(
            row for row in validation_records if cast("float", row.checkpoint_validation_fidelity) == best
        )
        selected_record = (
            candidates[0] if stage.checkpoint_validation.tie_breaker == "earliest_iteration" else candidates[-1]
        )
        if selected_record.global_iteration != selected_iteration or best != selected_fidelity:
            msg = "Validation-selected checkpoint does not implement the configured winner and tie-break rule."
            raise ValueError(msg)
    elif validation_records:
        msg = "A stage without checkpoint validation cannot contain validation trace outcomes."
        raise ValueError(msg)
    return records


def _noisy_execution_checksum(
    *,
    stage: TrainingStageConfig,
    circuit_binding_checksum: str,
    provider_checksum: str | None,
    objective_checksum: str,
    objective_binding_checksum: str,
    initial_parameter_checksum: str,
    final_parameter_checksum: str,
    selected_parameter_checksum: str,
    selected_iteration: int,
    selected_fidelity: float | None,
    trace: Sequence[NoisyKrotovIterationRecord],
    training_ensembles: Sequence[KrotovFixedMapEnsemble],
    validation_ensembles: Sequence[KrotovFixedMapEnsemble],
    normalized_work: Mapping[str, object],
) -> str:
    """Recompute the exact WP17 successful-execution audit checksum."""
    return canonical_checksum({
        "schema_version": NOISY_KROTOV_EXECUTION_SCHEMA_VERSION,
        "adapter_version": NOISY_KROTOV_ADAPTER_VERSION,
        "stage_index": stage.stage_index,
        "stage_id": stage.stage_id,
        "stage_configuration_checksum": stage.configuration_checksum,
        "circuit_binding_checksum": circuit_binding_checksum,
        "provider_checksum": provider_checksum,
        "objective_checksum": objective_checksum,
        "objective_binding_checksum": objective_binding_checksum,
        "initial_parameter_checksum": initial_parameter_checksum,
        "final_parameter_checksum": final_parameter_checksum,
        "selected_parameter_checksum": selected_parameter_checksum,
        "selected_global_iteration": selected_iteration,
        "selected_checkpoint_validation_fidelity": selected_fidelity,
        "trace": [row.to_dict() for row in trace],
        "training_ensemble_checksums": [item.content_checksum for item in training_ensembles],
        "checkpoint_validation_ensemble_checksums": [item.content_checksum for item in validation_ensembles],
        "normalized_work": dict(normalized_work),
        "input_resume_state_checksum": None,
    })


def _noisy_execution_summaries(
    *,
    stage: TrainingStageConfig,
    trace: Sequence[NoisyKrotovIterationRecord],
    adapter_checksum: str,
    selected_iteration: int,
    selected_fidelity: float | None,
    selected_parameter_checksum: str,
    final_parameter_checksum: str,
    cumulative_pairings: int,
    validation_ensembles: Sequence[KrotovFixedMapEnsemble],
) -> tuple[Mapping[str, object], Mapping[str, object] | None]:
    """Derive the only valid WP17 training and validation summary aliases."""
    final_row = trace[-1]
    training_summary: Mapping[str, object] = {
        "adapter_execution_checksum": adapter_checksum,
        "completed_iterations": final_row.global_iteration,
        "final_monitoring_fidelity": final_row.monitoring_fidelity,
        "final_monitoring_loss": final_row.monitoring_loss,
        "selected_iteration": selected_iteration,
        "selected_parameter_checksum": selected_parameter_checksum,
        "final_parameter_checksum": final_parameter_checksum,
        "cumulative_cross_trajectory_pairings": cumulative_pairings,
    }
    validation_summary: Mapping[str, object] | None = None
    if stage.checkpoint_validation.enabled:
        validation_summary = {
            "evaluation_count": sum(row.checkpoint_validation_fidelity is not None for row in trace),
            "selected_iteration": selected_iteration,
            "selected_fidelity": selected_fidelity,
            "ensemble_checksums": tuple(item.content_checksum for item in validation_ensembles),
        }
    return training_summary, validation_summary


@dataclass(frozen=True, slots=True)
class FixedMapArtifactRef:
    """Filesystem and scientific identity of one persisted fixed-map ensemble."""

    role: str
    ensemble_id: str
    content_checksum: str
    path: str
    file_checksum: str

    def __post_init__(self) -> None:
        """Validate the role, identifiers, path, and checksums."""
        if self.role not in _MAP_ROLES:
            msg = f"role must be one of {sorted(_MAP_ROLES)!r}."
            raise ValueError(msg)
        object.__setattr__(self, "ensemble_id", require_slug(self.ensemble_id, "ensemble_id"))
        object.__setattr__(self, "content_checksum", require_checksum(self.content_checksum, "content_checksum"))
        object.__setattr__(self, "path", require_relative_path(self.path, "path"))
        object.__setattr__(self, "file_checksum", require_checksum(self.file_checksum, "file_checksum"))

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native reference."""
        return {
            "role": self.role,
            "ensemble_id": self.ensemble_id,
            "content_checksum": self.content_checksum,
            "path": self.path,
            "file_checksum": self.file_checksum,
        }

    @classmethod
    def from_dict(cls, data: object) -> FixedMapArtifactRef:
        """Construct a strict fixed-map reference."""
        mapping = require_mapping(data, "fixed-map artifact reference")
        require_exact_keys(mapping, _FIXED_MAP_REF_KEYS, "fixed-map artifact reference")
        return cls(
            role=cast("str", mapping["role"]),
            ensemble_id=cast("str", mapping["ensemble_id"]),
            content_checksum=cast("str", mapping["content_checksum"]),
            path=cast("str", mapping["path"]),
            file_checksum=cast("str", mapping["file_checksum"]),
        )


@dataclass(frozen=True, slots=True, init=False)
class StageExecutionEvidence:
    """In-memory, optimizer-independent evidence for one completed stage."""

    stage: TrainingStageConfig
    source_parameter_checksum: str | None
    initial_parameter_checksum: str
    final_parameter_checksum: str
    selected_parameter_checksum: str
    selected_global_iteration: int
    completed_global_iteration: int
    selected_checkpoint_validation_fidelity: float | None
    circuit_binding_checksum: str | None
    provider_checksum: str | None
    checkpoint_validation_provider_checksum: str | None
    objective_checksum: str | None
    objective_binding: NoisyKrotovObjectiveBinding | None
    trace: tuple[Mapping[str, object], ...]
    training_ensembles: tuple[KrotovFixedMapEnsemble, ...]
    checkpoint_validation_ensembles: tuple[KrotovFixedMapEnsemble, ...]
    normalized_work: Mapping[str, object]
    training_summary: Mapping[str, object]
    checkpoint_validation_summary: Mapping[str, object] | None
    circuit_topology: Mapping[str, object]
    circuit_statistics: Mapping[str, object]
    optimizer_state: Mapping[str, object] | None
    cumulative_cross_trajectory_pairings: int
    _initial_parameters: NDArray[np.float64] = field(repr=False)
    _final_parameters: NDArray[np.float64] = field(repr=False)
    _selected_parameters: NDArray[np.float64] = field(repr=False)

    def __init__(
        self,
        *,
        stage: TrainingStageConfig,
        source_parameters: object | None,
        initial_parameters: object,
        final_parameters: object,
        selected_parameters: object,
        selected_global_iteration: int,
        completed_global_iteration: int,
        selected_checkpoint_validation_fidelity: float | None,
        circuit_binding_checksum: str | None,
        provider_checksum: str | None,
        objective_checksum: str | None,
        trace: Sequence[Mapping[str, object]],
        training_ensembles: Sequence[KrotovFixedMapEnsemble],
        checkpoint_validation_ensembles: Sequence[KrotovFixedMapEnsemble],
        normalized_work: Mapping[str, object],
        training_summary: Mapping[str, object],
        checkpoint_validation_summary: Mapping[str, object] | None,
        circuit_topology: Mapping[str, object],
        circuit_statistics: Mapping[str, object],
        optimizer_state: Mapping[str, object] | None = None,
        cumulative_cross_trajectory_pairings: int = 0,
        objective_binding: NoisyKrotovObjectiveBinding | None = None,
    ) -> None:
        """Validate and defensively freeze arrays and JSON evidence."""
        if not isinstance(stage, TrainingStageConfig):
            msg = "stage must be a TrainingStageConfig."
            raise TypeError(msg)
        source = None if source_parameters is None else _immutable_vector(source_parameters, "source_parameters")
        if stage.input_parameter_count == 0:
            if source is not None:
                msg = "A stage without an input topology cannot declare source parameters."
                raise ValueError(msg)
        elif source is None or source.size != stage.input_parameter_count:
            msg = "source_parameters must match the configured stage input parameter count."
            raise ValueError(msg)
        initial = _immutable_vector(initial_parameters, "initial_parameters")
        final = _immutable_vector(final_parameters, "final_parameters")
        selected = _immutable_vector(selected_parameters, "selected_parameters")
        if final.size != stage.output_parameter_count or selected.size != stage.output_parameter_count:
            msg = "Final and selected parameter counts must match the configured stage output."
            raise ValueError(msg)
        if initial.size != stage.output_parameter_count:
            msg = "Stage execution parameters must already be bound to the output topology."
            raise ValueError(msg)
        if (
            source is not None
            and stage.parameter_transfer_rule in {"copy", "load_checkpoint"}
            and (source.shape != initial.shape or not np.array_equal(source, initial))
        ):
            msg = "Copy/load parameter transfer must bind the exact source vector before optimization."
            raise ValueError(msg)
        if (
            source is not None
            and stage.stage_kind == "grow"
            and not np.array_equal(
                source,
                initial[: stage.input_parameter_count],
            )
        ):
            msg = "Growth parameter transfer must preserve the exact predecessor prefix."
            raise ValueError(msg)
        selected_iteration = require_int(selected_global_iteration, "selected_global_iteration")
        completed_iteration = require_int(completed_global_iteration, "completed_global_iteration")
        if selected_iteration > completed_iteration:
            msg = "selected_global_iteration cannot exceed completed_global_iteration."
            raise ValueError(msg)
        if completed_iteration != stage.iteration_budget:
            msg = "A completed stage must reach its configured iteration budget."
            raise ValueError(msg)
        fidelity = selected_checkpoint_validation_fidelity
        if fidelity is not None:
            fidelity = require_float(
                float(fidelity), "selected_checkpoint_validation_fidelity", minimum=0.0, maximum=1.0
            )
        if stage.checkpoint_validation.enabled != (fidelity is not None):
            msg = "Checkpoint-selection fidelity presence must match the checkpoint-validation policy."
            raise ValueError(msg)
        if not stage.checkpoint_validation.enabled:
            if selected_iteration != completed_iteration or not np.array_equal(selected, final):
                msg = "A stage without checkpoint validation must select its completed final parameters."
                raise ValueError(msg)
        else:
            cadence = cast("int", stage.checkpoint_validation.cadence)
            if selected_iteration not in {0, completed_iteration} and selected_iteration % cadence != 0:
                msg = "Selected checkpoint iteration must be zero, a validation cadence, or the final iteration."
                raise ValueError(msg)
        for name, checksum in (
            ("circuit_binding_checksum", circuit_binding_checksum),
            ("provider_checksum", provider_checksum),
            ("objective_checksum", objective_checksum),
        ):
            if checksum is not None:
                require_checksum(checksum, name)
        if stage.optimizer_id != "none" and (circuit_binding_checksum is None or objective_checksum is None):
            msg = "Optimized stage evidence requires circuit-binding and objective provenance."
            raise ValueError(msg)
        topology_mapping = _validated_circuit_topology(
            circuit_topology,
            stage=stage,
            circuit_binding_checksum=circuit_binding_checksum,
        )
        trace_rows = tuple(freeze_json_mapping(row, f"trace[{index}]") for index, row in enumerate(trace))
        training_maps = tuple(training_ensembles)
        validation_maps = tuple(checkpoint_validation_ensembles)
        if not all(isinstance(item, KrotovFixedMapEnsemble) for item in (*training_maps, *validation_maps)):
            msg = "Fixed-map collections must contain KrotovFixedMapEnsemble values."
            raise TypeError(msg)
        if any(item.role != "training_trajectory" for item in training_maps):
            msg = "training_ensembles may contain only training_trajectory maps."
            raise ValueError(msg)
        if any(item.role != "checkpoint_validation" for item in validation_maps):
            msg = "checkpoint_validation_ensembles may contain only checkpoint_validation maps."
            raise ValueError(msg)
        all_maps = (*training_maps, *validation_maps)
        if len({item.ensemble_id for item in all_maps}) != len(all_maps) or len({
            item.content_checksum for item in all_maps
        }) != len(all_maps):
            msg = "A stage cannot reuse a fixed-map identity or checksum across evidence roles."
            raise ValueError(msg)
        if all_maps and circuit_binding_checksum is None:
            msg = "Fixed-map evidence requires an exact circuit binding checksum."
            raise ValueError(msg)
        if any(item.circuit_checksum != circuit_binding_checksum for item in all_maps):
            msg = "Fixed-map evidence does not match the stage circuit binding."
            raise ValueError(msg)
        if training_maps and provider_checksum is None:
            msg = "Training fixed maps require an exact training provider checksum."
            raise ValueError(msg)
        if any(item.provider_checksum != provider_checksum for item in training_maps):
            msg = "Training fixed maps do not match the stage noise provider."
            raise ValueError(msg)
        validation_provider_checksums = {item.provider_checksum for item in validation_maps}
        if len(validation_provider_checksums) > 1:
            msg = "Checkpoint-validation fixed maps must share one provider checksum."
            raise ValueError(msg)
        work = freeze_json_mapping(_sum_work([normalized_work]), "normalized_work")
        training = freeze_json_mapping(training_summary, "training_summary")
        if not training:
            msg = "training_summary must not be empty."
            raise ValueError(msg)
        validation = (
            None
            if checkpoint_validation_summary is None
            else freeze_json_mapping(checkpoint_validation_summary, "checkpoint_validation_summary")
        )
        if stage.checkpoint_validation.enabled != (validation is not None):
            msg = "Checkpoint-validation summary presence must match the stage policy."
            raise ValueError(msg)
        if validation is not None:
            if validation.get("selected_iteration") != selected_iteration:
                msg = "checkpoint_validation_summary selected_iteration does not match the selected checkpoint."
                raise ValueError(msg)
            summary_fidelity = validation.get("selected_fidelity")
            if summary_fidelity is not None and summary_fidelity != fidelity:
                msg = "checkpoint_validation_summary selected_fidelity does not match the selected checkpoint."
                raise ValueError(msg)
        statistics_mapping = freeze_json_mapping(circuit_statistics, "circuit_statistics")
        if not statistics_mapping:
            msg = "circuit_statistics must not be empty."
            raise ValueError(msg)
        if (
            statistics_mapping.get("topology_id") != stage.output_topology_id
            or statistics_mapping.get("parameter_count") != stage.output_parameter_count
        ):
            msg = "circuit_statistics must identify the configured output topology and parameter count."
            raise ValueError(msg)
        adapter_checksum = training.get("adapter_execution_checksum")
        if (stage.optimizer_id == "krotov") != (adapter_checksum is not None):
            msg = "optimizer_id='krotov' requires exact WP17 adapter execution evidence, exclusively."
            raise ValueError(msg)
        if adapter_checksum is not None:
            expected_adapter_checksum = require_checksum(adapter_checksum, "adapter_execution_checksum")
            if (
                circuit_binding_checksum is None
                or objective_checksum is None
                or not isinstance(objective_binding, NoisyKrotovObjectiveBinding)
            ):
                msg = "WP17 adapter evidence requires circuit and sealed objective provenance."
                raise ValueError(msg)
            if objective_binding.objective_checksum != objective_checksum:
                msg = "WP17 objective binding does not reproduce objective_checksum."
                raise ValueError(msg)
            noisy_trace = _validate_noisy_trace_semantics(
                stage=stage,
                trace=trace_rows,
                training_ensembles=training_maps,
                validation_ensembles=validation_maps,
                normalized_work=work,
                initial_parameter_checksum=_vector_checksum(initial),
                final_parameter_checksum=_vector_checksum(final),
                selected_parameter_checksum=_vector_checksum(selected),
                completed_iteration=completed_iteration,
                selected_iteration=selected_iteration,
                selected_fidelity=fidelity,
                cumulative_pairings=cumulative_cross_trajectory_pairings,
                circuit_topology=topology_mapping,
                provider_checksum=provider_checksum,
            )
            actual_adapter_checksum = _noisy_execution_checksum(
                stage=stage,
                circuit_binding_checksum=circuit_binding_checksum,
                provider_checksum=provider_checksum,
                objective_checksum=objective_checksum,
                objective_binding_checksum=objective_binding.content_checksum,
                initial_parameter_checksum=_vector_checksum(initial),
                final_parameter_checksum=_vector_checksum(final),
                selected_parameter_checksum=_vector_checksum(selected),
                selected_iteration=selected_iteration,
                selected_fidelity=fidelity,
                trace=noisy_trace,
                training_ensembles=training_maps,
                validation_ensembles=validation_maps,
                normalized_work=work,
            )
            if actual_adapter_checksum != expected_adapter_checksum:
                msg = "WP17 adapter execution checksum does not close over the supplied stage evidence."
                raise ValueError(msg)
            expected_training_summary, expected_validation_summary = _noisy_execution_summaries(
                stage=stage,
                trace=noisy_trace,
                adapter_checksum=actual_adapter_checksum,
                selected_iteration=selected_iteration,
                selected_fidelity=fidelity,
                selected_parameter_checksum=_vector_checksum(selected),
                final_parameter_checksum=_vector_checksum(final),
                cumulative_pairings=cumulative_cross_trajectory_pairings,
                validation_ensembles=validation_maps,
            )
            if training != expected_training_summary:
                msg = "WP17 training summary is not exactly implied by the adapter trace and checkpoints."
                raise ValueError(msg)
            if validation != expected_validation_summary:
                msg = "WP17 validation summary is not exactly implied by the trace and fixed maps."
                raise ValueError(msg)
        elif objective_binding is not None:
            msg = "Only genuine WP17 adapter evidence may carry an objective binding."
            raise ValueError(msg)
        optimizer = None if optimizer_state is None else freeze_json_mapping(optimizer_state, "optimizer_state")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(
            self,
            "source_parameter_checksum",
            None if source is None else _vector_checksum(source),
        )
        object.__setattr__(self, "initial_parameter_checksum", _vector_checksum(initial))
        object.__setattr__(self, "final_parameter_checksum", _vector_checksum(final))
        object.__setattr__(self, "selected_parameter_checksum", _vector_checksum(selected))
        object.__setattr__(self, "selected_global_iteration", selected_iteration)
        object.__setattr__(self, "completed_global_iteration", completed_iteration)
        object.__setattr__(self, "selected_checkpoint_validation_fidelity", fidelity)
        object.__setattr__(self, "circuit_binding_checksum", circuit_binding_checksum)
        object.__setattr__(self, "provider_checksum", provider_checksum)
        object.__setattr__(
            self,
            "checkpoint_validation_provider_checksum",
            None if not validation_provider_checksums else next(iter(validation_provider_checksums)),
        )
        object.__setattr__(self, "objective_checksum", objective_checksum)
        object.__setattr__(self, "objective_binding", objective_binding)
        object.__setattr__(self, "trace", trace_rows)
        object.__setattr__(self, "training_ensembles", training_maps)
        object.__setattr__(self, "checkpoint_validation_ensembles", validation_maps)
        object.__setattr__(self, "normalized_work", work)
        object.__setattr__(self, "training_summary", training)
        object.__setattr__(self, "checkpoint_validation_summary", validation)
        object.__setattr__(self, "circuit_topology", topology_mapping)
        object.__setattr__(self, "circuit_statistics", statistics_mapping)
        object.__setattr__(self, "optimizer_state", optimizer)
        object.__setattr__(
            self,
            "cumulative_cross_trajectory_pairings",
            require_int(cumulative_cross_trajectory_pairings, "cumulative_cross_trajectory_pairings"),
        )
        object.__setattr__(self, "_initial_parameters", initial)
        object.__setattr__(self, "_final_parameters", final)
        object.__setattr__(self, "_selected_parameters", selected)

    @property
    def initial_parameters(self) -> NDArray[np.float64]:
        """Return a detached initial parameter vector."""
        return self._initial_parameters.copy()

    @property
    def final_parameters(self) -> NDArray[np.float64]:
        """Return a detached last-iteration parameter vector."""
        return self._final_parameters.copy()

    @property
    def selected_parameters(self) -> NDArray[np.float64]:
        """Return a detached validation-selected handoff vector."""
        return self._selected_parameters.copy()

    @classmethod
    def from_noisy_krotov(
        cls,
        stage: TrainingStageConfig,
        execution: NoisyKrotovStageExecution,
        *,
        source_parameters: object | None,
        circuit_statistics: Mapping[str, object],
        optimizer_state: Mapping[str, object] | None = None,
    ) -> StageExecutionEvidence:
        """Translate a successful WP17 execution into persistence evidence."""
        if not isinstance(execution, NoisyKrotovStageExecution):
            msg = "execution must be a NoisyKrotovStageExecution."
            raise TypeError(msg)
        if (
            execution.stage_index != stage.stage_index
            or execution.stage_id != stage.stage_id
            or execution.stage_configuration_checksum != stage.configuration_checksum
        ):
            msg = "Noisy Krotov execution does not identify the supplied stage."
            raise ValueError(msg)
        trace = tuple(row.to_dict() for row in execution.trace)
        final_row = execution.trace[-1]
        training_summary = {
            "adapter_execution_checksum": execution.content_checksum,
            "completed_iterations": final_row.global_iteration,
            "final_monitoring_fidelity": final_row.monitoring_fidelity,
            "final_monitoring_loss": final_row.monitoring_loss,
            "selected_iteration": execution.selected_global_iteration,
            "selected_parameter_checksum": execution.selected_parameter_checksum,
            "final_parameter_checksum": execution.final_parameter_checksum,
            "cumulative_cross_trajectory_pairings": execution.cross_trajectory_pairings,
        }
        validation_summary = None
        if stage.checkpoint_validation.enabled:
            validation_rows = [row for row in execution.trace if row.checkpoint_validation_fidelity is not None]
            validation_summary = {
                "evaluation_count": len(validation_rows),
                "selected_iteration": execution.selected_global_iteration,
                "selected_fidelity": execution.selected_checkpoint_validation_fidelity,
                "ensemble_checksums": list(execution.checkpoint_validation_ensemble_checksums),
            }
        return cls(
            stage=stage,
            source_parameters=source_parameters,
            initial_parameters=execution.initial_theta,
            final_parameters=execution.final_theta,
            selected_parameters=execution.selected_theta,
            selected_global_iteration=execution.selected_global_iteration,
            completed_global_iteration=final_row.global_iteration,
            selected_checkpoint_validation_fidelity=execution.selected_checkpoint_validation_fidelity,
            circuit_binding_checksum=execution.circuit_binding_checksum,
            provider_checksum=execution.provider_checksum,
            objective_checksum=execution.objective_checksum,
            objective_binding=execution.objective_binding,
            trace=trace,
            training_ensembles=execution.training_ensembles,
            checkpoint_validation_ensembles=execution.checkpoint_validation_ensembles,
            normalized_work=execution.normalized_work,
            training_summary=training_summary,
            checkpoint_validation_summary=validation_summary,
            circuit_topology=execution.circuit_binding_document,
            circuit_statistics=circuit_statistics,
            optimizer_state=optimizer_state,
            cumulative_cross_trajectory_pairings=execution.cross_trajectory_pairings,
        )

    @classmethod
    def for_parameter_transform(
        cls,
        stage: TrainingStageConfig,
        *,
        initial_parameters: object,
        output_parameters: object,
        circuit_binding_checksum: str,
        circuit_topology: Mapping[str, object],
        circuit_statistics: Mapping[str, object],
        summary: Mapping[str, object],
    ) -> StageExecutionEvidence:
        """Create evidence for a deterministic zero-iteration grow or prune stage."""
        if stage.iteration_budget != 0 or stage.optimizer_id != "none":
            msg = "Parameter-transform evidence requires a zero-iteration optimizer='none' stage."
            raise ValueError(msg)
        output = _immutable_vector(output_parameters, "output_parameters")
        return cls(
            stage=stage,
            source_parameters=initial_parameters,
            initial_parameters=output,
            final_parameters=output,
            selected_parameters=output,
            selected_global_iteration=0,
            completed_global_iteration=0,
            selected_checkpoint_validation_fidelity=None,
            circuit_binding_checksum=circuit_binding_checksum,
            provider_checksum=None,
            objective_checksum=None,
            trace=(
                {
                    "event": "parameter_transform",
                    "input_parameter_checksum": _vector_checksum(
                        _immutable_vector(initial_parameters, "initial_parameters")
                    ),
                },
            ),
            training_ensembles=(),
            checkpoint_validation_ensembles=(),
            normalized_work=_empty_work(),
            training_summary=summary,
            checkpoint_validation_summary=None,
            circuit_topology=circuit_topology,
            circuit_statistics=circuit_statistics,
        )


@dataclass(frozen=True, slots=True)
class PersistedStageArtifact:
    """Canonical row linking a WP16 stage result to all persisted evidence."""

    stage_result: TrainingStageResult
    runtime_fingerprint_checksum: str
    checkpoint_file_checksum: str
    trace_file_checksum: str
    metadata_file_checksum: str
    fixed_map_artifacts: tuple[FixedMapArtifactRef, ...]
    schema_version: str = field(default=PHASE2_STAGE_ARTIFACT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate all redundant file and scientific links."""
        if not isinstance(self.stage_result, TrainingStageResult):
            msg = "stage_result must be a TrainingStageResult."
            raise TypeError(msg)
        runtime = require_checksum(self.runtime_fingerprint_checksum, "runtime_fingerprint_checksum")
        checkpoint = require_checksum(self.checkpoint_file_checksum, "checkpoint_file_checksum")
        trace_checksum = require_checksum(self.trace_file_checksum, "trace_file_checksum")
        metadata_checksum = require_checksum(self.metadata_file_checksum, "metadata_file_checksum")
        if checkpoint != self.stage_result.produced_checkpoint_checksum:
            msg = "checkpoint_file_checksum must match the stage result checkpoint."
            raise ValueError(msg)
        if trace_checksum != self.stage_result.optimizer_trace_checksum:
            msg = "trace_file_checksum must match the stage result optimizer trace."
            raise ValueError(msg)
        if metadata_checksum != self.stage_result.diagnostic_sidecar_checksum:
            msg = "metadata_file_checksum must match the stage result diagnostic sidecar."
            raise ValueError(msg)
        refs = tuple(self.fixed_map_artifacts)
        if len({ref.path for ref in refs}) != len(refs):
            msg = "fixed_map_artifacts paths must be unique."
            raise ValueError(msg)
        if len({ref.ensemble_id for ref in refs}) != len(refs):
            msg = "fixed_map_artifacts ensemble identities must be unique."
            raise ValueError(msg)
        object.__setattr__(self, "runtime_fingerprint_checksum", runtime)
        object.__setattr__(self, "checkpoint_file_checksum", checkpoint)
        object.__setattr__(self, "trace_file_checksum", trace_checksum)
        object.__setattr__(self, "metadata_file_checksum", metadata_checksum)
        object.__setattr__(self, "fixed_map_artifacts", refs)

    @property
    def content_checksum(self) -> str:
        """Checksum the complete canonical stage-artifact row."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered field."""
        return {
            "schema_version": self.schema_version,
            "stage_result": self.stage_result.to_dict(),
            "runtime_fingerprint_checksum": self.runtime_fingerprint_checksum,
            "checkpoint_file_checksum": self.checkpoint_file_checksum,
            "trace_file_checksum": self.trace_file_checksum,
            "metadata_file_checksum": self.metadata_file_checksum,
            "fixed_map_artifacts": [ref.to_dict() for ref in self.fixed_map_artifacts],
        }

    def to_dict(self) -> dict[str, object]:
        """Return a checksum-sealed JSON-native row."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> PersistedStageArtifact:
        """Construct and checksum-verify one canonical stage row."""
        mapping = verify_sealed_mapping(data, expected_keys=_STAGE_ARTIFACT_KEYS, name="persisted stage artifact")
        if mapping["schema_version"] != PHASE2_STAGE_ARTIFACT_SCHEMA_VERSION:
            msg = f"schema_version must be {PHASE2_STAGE_ARTIFACT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        raw_refs = mapping["fixed_map_artifacts"]
        if type(raw_refs) is not tuple:
            msg = "fixed_map_artifacts must be a serialized sequence."
            raise TypeError(msg)
        result = cls(
            stage_result=TrainingStageResult.from_dict(mapping["stage_result"]),
            runtime_fingerprint_checksum=cast("str", mapping["runtime_fingerprint_checksum"]),
            checkpoint_file_checksum=cast("str", mapping["checkpoint_file_checksum"]),
            trace_file_checksum=cast("str", mapping["trace_file_checksum"]),
            metadata_file_checksum=cast("str", mapping["metadata_file_checksum"]),
            fixed_map_artifacts=tuple(FixedMapArtifactRef.from_dict(item) for item in raw_refs),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Persisted stage artifact checksum changed during normalization."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> PersistedStageArtifact:
        """Construct a stage artifact from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class StageFailureArtifact:
    """Append-only structured failure for one attempted pipeline stage."""

    pipeline_training_id: str
    pipeline_configuration_checksum: str
    pipeline_prefix_id: str
    stage_index: int
    stage_id: str
    stage_configuration_checksum: str
    phase: str
    exception_type: str
    message: str
    traceback: str
    retryable: bool
    attempt: int
    partial_work: Mapping[str, object]
    completed_stage_artifact_checksums: tuple[str, ...]
    wall_time_seconds: float
    runtime_fingerprint_checksum: str
    schema_version: str = field(default=PHASE2_STAGE_FAILURE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate immutable pipeline links and diagnostics."""
        object.__setattr__(
            self, "pipeline_training_id", require_slug(self.pipeline_training_id, "pipeline_training_id")
        )
        for name in (
            "pipeline_configuration_checksum",
            "stage_configuration_checksum",
            "runtime_fingerprint_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(self, "pipeline_prefix_id", require_slug(self.pipeline_prefix_id, "pipeline_prefix_id"))
        object.__setattr__(self, "stage_index", require_int(self.stage_index, "stage_index"))
        object.__setattr__(self, "stage_id", require_slug(self.stage_id, "stage_id"))
        for name in ("phase", "exception_type", "message", "traceback"):
            object.__setattr__(self, name, require_nonempty_text(getattr(self, name), name))
        object.__setattr__(self, "retryable", require_bool(self.retryable, "retryable"))
        object.__setattr__(self, "attempt", require_int(self.attempt, "attempt", minimum=1))
        object.__setattr__(self, "partial_work", freeze_json_mapping(self.partial_work, "partial_work"))
        completed = tuple(
            require_checksum(value, f"completed_stage_artifact_checksums[{index}]")
            for index, value in enumerate(self.completed_stage_artifact_checksums)
        )
        object.__setattr__(self, "completed_stage_artifact_checksums", completed)
        object.__setattr__(
            self,
            "wall_time_seconds",
            require_float(float(self.wall_time_seconds), "wall_time_seconds", minimum=0.0),
        )

    @property
    def failure_id(self) -> str:
        """Stable identity of one ordered stage attempt."""
        return "phase2_stage_failure_" + canonical_checksum({
            "pipeline_training_id": self.pipeline_training_id,
            "pipeline_prefix_id": self.pipeline_prefix_id,
            "attempt": self.attempt,
        }).removeprefix("sha256:")

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered failure field."""
        return {
            "schema_version": self.schema_version,
            "failure_id": self.failure_id,
            "pipeline_training_id": self.pipeline_training_id,
            "pipeline_configuration_checksum": self.pipeline_configuration_checksum,
            "pipeline_prefix_id": self.pipeline_prefix_id,
            "stage_index": self.stage_index,
            "stage_id": self.stage_id,
            "stage_configuration_checksum": self.stage_configuration_checksum,
            "phase": self.phase,
            "exception_type": self.exception_type,
            "message": self.message,
            "traceback": self.traceback,
            "retryable": self.retryable,
            "attempt": self.attempt,
            "partial_work": thaw_json_mapping(self.partial_work),
            "completed_stage_artifact_checksums": list(self.completed_stage_artifact_checksums),
            "wall_time_seconds": self.wall_time_seconds,
            "runtime_fingerprint_checksum": self.runtime_fingerprint_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum the complete stage failure."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return a checksum-sealed failure record."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> StageFailureArtifact:
        """Construct and verify a strict serialized stage failure."""
        mapping = verify_sealed_mapping(data, expected_keys=_STAGE_FAILURE_KEYS, name="stage failure artifact")
        if mapping["schema_version"] != PHASE2_STAGE_FAILURE_SCHEMA_VERSION:
            msg = f"schema_version must be {PHASE2_STAGE_FAILURE_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        completed = mapping["completed_stage_artifact_checksums"]
        if type(completed) is not tuple:
            msg = "completed_stage_artifact_checksums must be a serialized sequence."
            raise TypeError(msg)
        artifact = cls(
            pipeline_training_id=cast("str", mapping["pipeline_training_id"]),
            pipeline_configuration_checksum=cast("str", mapping["pipeline_configuration_checksum"]),
            pipeline_prefix_id=cast("str", mapping["pipeline_prefix_id"]),
            stage_index=cast("int", mapping["stage_index"]),
            stage_id=cast("str", mapping["stage_id"]),
            stage_configuration_checksum=cast("str", mapping["stage_configuration_checksum"]),
            phase=cast("str", mapping["phase"]),
            exception_type=cast("str", mapping["exception_type"]),
            message=cast("str", mapping["message"]),
            traceback=cast("str", mapping["traceback"]),
            retryable=cast("bool", mapping["retryable"]),
            attempt=cast("int", mapping["attempt"]),
            partial_work=cast("Mapping[str, object]", mapping["partial_work"]),
            completed_stage_artifact_checksums=cast("tuple[str, ...]", completed),
            wall_time_seconds=cast("float", mapping["wall_time_seconds"]),
            runtime_fingerprint_checksum=cast("str", mapping["runtime_fingerprint_checksum"]),
        )
        if mapping["failure_id"] != artifact.failure_id or mapping["content_checksum"] != artifact.content_checksum:
            msg = "Stage failure identity or checksum changed during normalization."
            raise ValueError(msg)
        return artifact

    @classmethod
    def from_json(cls, payload: str) -> StageFailureArtifact:
        """Construct a failure from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class MaterializedCircuitArtifact:
    """One shared, checksum-verified final-circuit materialization."""

    materialized_circuit_id: str
    pipeline_training_id: str
    pipeline_result_checksum: str
    final_checkpoint_checksum: str
    materialization_policy_checksum: str
    path: str
    payload_checksum: str
    wall_time_seconds: float
    peak_memory_bytes: int
    runtime_fingerprint_checksum: str
    schema_version: str = field(default=PHASE2_MATERIALIZATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate artifact identity, provenance, and resource observations."""
        object.__setattr__(
            self, "materialized_circuit_id", require_slug(self.materialized_circuit_id, "materialized_circuit_id")
        )
        object.__setattr__(
            self, "pipeline_training_id", require_slug(self.pipeline_training_id, "pipeline_training_id")
        )
        for name in (
            "pipeline_result_checksum",
            "final_checkpoint_checksum",
            "materialization_policy_checksum",
            "payload_checksum",
            "runtime_fingerprint_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(self, "path", require_relative_path(self.path, "path"))
        object.__setattr__(
            self,
            "wall_time_seconds",
            require_float(float(self.wall_time_seconds), "wall_time_seconds", minimum=0.0),
        )
        object.__setattr__(self, "peak_memory_bytes", require_int(self.peak_memory_bytes, "peak_memory_bytes"))

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered materialization fields."""
        return {
            "schema_version": self.schema_version,
            "materialized_circuit_id": self.materialized_circuit_id,
            "pipeline_training_id": self.pipeline_training_id,
            "pipeline_result_checksum": self.pipeline_result_checksum,
            "final_checkpoint_checksum": self.final_checkpoint_checksum,
            "materialization_policy_checksum": self.materialization_policy_checksum,
            "path": self.path,
            "payload_checksum": self.payload_checksum,
            "wall_time_seconds": self.wall_time_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "runtime_fingerprint_checksum": self.runtime_fingerprint_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum this materialization record."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return a sealed materialization record."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> MaterializedCircuitArtifact:
        """Construct and verify a strict materialization record."""
        mapping = verify_sealed_mapping(data, expected_keys=_MATERIALIZATION_KEYS, name="materialized circuit artifact")
        if mapping["schema_version"] != PHASE2_MATERIALIZATION_SCHEMA_VERSION:
            msg = f"schema_version must be {PHASE2_MATERIALIZATION_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        artifact = cls(
            materialized_circuit_id=cast("str", mapping["materialized_circuit_id"]),
            pipeline_training_id=cast("str", mapping["pipeline_training_id"]),
            pipeline_result_checksum=cast("str", mapping["pipeline_result_checksum"]),
            final_checkpoint_checksum=cast("str", mapping["final_checkpoint_checksum"]),
            materialization_policy_checksum=cast("str", mapping["materialization_policy_checksum"]),
            path=cast("str", mapping["path"]),
            payload_checksum=cast("str", mapping["payload_checksum"]),
            wall_time_seconds=cast("float", mapping["wall_time_seconds"]),
            peak_memory_bytes=cast("int", mapping["peak_memory_bytes"]),
            runtime_fingerprint_checksum=cast("str", mapping["runtime_fingerprint_checksum"]),
        )
        if mapping["content_checksum"] != artifact.content_checksum:
            msg = "Materialized circuit checksum changed during normalization."
            raise ValueError(msg)
        return artifact

    @classmethod
    def from_json(cls, payload: str) -> MaterializedCircuitArtifact:
        """Construct a materialization from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class MaterializationAttemptArtifact:
    """One non-overlapping shared final-circuit materialization attempt."""

    materialized_circuit_id: str
    pipeline_training_id: str
    pipeline_result_checksum: str
    attempt: int
    status: Literal["success", "failure"]
    phase: Literal["materialization", "serialization"]
    payload_checksum: str | None
    exception_type: str | None
    message: str | None
    wall_time_seconds: float
    peak_memory_bytes: int
    runtime_fingerprint_checksum: str
    schema_version: str = field(default=PHASE2_MATERIALIZATION_ATTEMPT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate identity, outcome fields, and exclusive resource timing."""
        object.__setattr__(
            self,
            "materialized_circuit_id",
            require_slug(self.materialized_circuit_id, "materialized_circuit_id"),
        )
        object.__setattr__(
            self, "pipeline_training_id", require_slug(self.pipeline_training_id, "pipeline_training_id")
        )
        object.__setattr__(
            self,
            "pipeline_result_checksum",
            require_checksum(self.pipeline_result_checksum, "pipeline_result_checksum"),
        )
        object.__setattr__(self, "attempt", require_int(self.attempt, "attempt", minimum=1))
        if self.status not in {"success", "failure"}:
            msg = "status must be 'success' or 'failure'."
            raise ValueError(msg)
        if self.phase not in {"materialization", "serialization"}:
            msg = "phase must be 'materialization' or 'serialization'."
            raise ValueError(msg)
        payload = None if self.payload_checksum is None else require_checksum(self.payload_checksum, "payload_checksum")
        exception_type = (
            None if self.exception_type is None else require_nonempty_text(self.exception_type, "exception_type")
        )
        message = None if self.message is None else require_nonempty_text(self.message, "message")
        if (self.status == "success") != (payload is not None and exception_type is None and message is None):
            msg = "Successful materialization attempts require a payload and no exception fields."
            raise ValueError(msg)
        if (self.status == "failure") != (payload is None and exception_type is not None and message is not None):
            msg = "Failed materialization attempts require exception fields and no payload."
            raise ValueError(msg)
        object.__setattr__(self, "payload_checksum", payload)
        object.__setattr__(self, "exception_type", exception_type)
        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "wall_time_seconds",
            require_float(float(self.wall_time_seconds), "wall_time_seconds", minimum=0.0),
        )
        object.__setattr__(self, "peak_memory_bytes", require_int(self.peak_memory_bytes, "peak_memory_bytes"))
        object.__setattr__(
            self,
            "runtime_fingerprint_checksum",
            require_checksum(self.runtime_fingerprint_checksum, "runtime_fingerprint_checksum"),
        )

    @property
    def attempt_id(self) -> str:
        """Stable identity for this circuit-local attempt number."""
        identity = {"circuit": self.materialized_circuit_id, "attempt": self.attempt}
        return f"phase2_materialization_attempt_{canonical_checksum(identity)[7:]}"

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered attempt field."""
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "materialized_circuit_id": self.materialized_circuit_id,
            "pipeline_training_id": self.pipeline_training_id,
            "pipeline_result_checksum": self.pipeline_result_checksum,
            "attempt": self.attempt,
            "status": self.status,
            "phase": self.phase,
            "payload_checksum": self.payload_checksum,
            "exception_type": self.exception_type,
            "message": self.message,
            "wall_time_seconds": self.wall_time_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "runtime_fingerprint_checksum": self.runtime_fingerprint_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum the complete materialization attempt."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return a checksum-sealed materialization attempt."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> MaterializationAttemptArtifact:
        """Construct and verify one strict materialization attempt."""
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_MATERIALIZATION_ATTEMPT_KEYS,
            name="materialization attempt",
        )
        if mapping["schema_version"] != PHASE2_MATERIALIZATION_ATTEMPT_SCHEMA_VERSION:
            msg = f"schema_version must be {PHASE2_MATERIALIZATION_ATTEMPT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        artifact = cls(
            materialized_circuit_id=cast("str", mapping["materialized_circuit_id"]),
            pipeline_training_id=cast("str", mapping["pipeline_training_id"]),
            pipeline_result_checksum=cast("str", mapping["pipeline_result_checksum"]),
            attempt=cast("int", mapping["attempt"]),
            status=cast("Literal['success', 'failure']", mapping["status"]),
            phase=cast("Literal['materialization', 'serialization']", mapping["phase"]),
            payload_checksum=cast("str | None", mapping["payload_checksum"]),
            exception_type=cast("str | None", mapping["exception_type"]),
            message=cast("str | None", mapping["message"]),
            wall_time_seconds=cast("float", mapping["wall_time_seconds"]),
            peak_memory_bytes=cast("int", mapping["peak_memory_bytes"]),
            runtime_fingerprint_checksum=cast("str", mapping["runtime_fingerprint_checksum"]),
        )
        if mapping["attempt_id"] != artifact.attempt_id or mapping["content_checksum"] != artifact.content_checksum:
            msg = "Materialization attempt identity or checksum changed during normalization."
            raise ValueError(msg)
        return artifact

    @classmethod
    def from_json(cls, payload: str) -> MaterializationAttemptArtifact:
        """Construct an attempt from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class EvaluationEvidenceArtifact:
    """Artifact ledger that supplements one typed Phase II result row."""

    evaluation_row_id: str
    record_checksum: str
    pipeline_result_checksum: str
    materialization_checksum: str
    evaluation_provider_checksum: str | None
    evaluation_map_artifacts: tuple[FixedMapArtifactRef, ...]
    schema_version: str = field(default=PHASE2_EVALUATION_EVIDENCE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate result, pipeline, materialization, and map links."""
        object.__setattr__(self, "evaluation_row_id", require_slug(self.evaluation_row_id, "evaluation_row_id"))
        for name in ("record_checksum", "pipeline_result_checksum", "materialization_checksum"):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        if self.evaluation_provider_checksum is not None:
            object.__setattr__(
                self,
                "evaluation_provider_checksum",
                require_checksum(self.evaluation_provider_checksum, "evaluation_provider_checksum"),
            )
        refs = tuple(self.evaluation_map_artifacts)
        if len({ref.ensemble_id for ref in refs}) != len(refs):
            msg = "evaluation_map_artifacts must not reuse an ensemble identity."
            raise ValueError(msg)
        object.__setattr__(self, "evaluation_map_artifacts", refs)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered evidence link."""
        return {
            "schema_version": self.schema_version,
            "evaluation_row_id": self.evaluation_row_id,
            "record_checksum": self.record_checksum,
            "pipeline_result_checksum": self.pipeline_result_checksum,
            "materialization_checksum": self.materialization_checksum,
            "evaluation_provider_checksum": self.evaluation_provider_checksum,
            "evaluation_map_artifacts": [ref.to_dict() for ref in self.evaluation_map_artifacts],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum this evidence ledger row."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return a sealed evidence record."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> EvaluationEvidenceArtifact:
        """Construct and verify a strict evidence record."""
        mapping = verify_sealed_mapping(data, expected_keys=_EVALUATION_EVIDENCE_KEYS, name="evaluation evidence")
        if mapping["schema_version"] != PHASE2_EVALUATION_EVIDENCE_SCHEMA_VERSION:
            msg = f"schema_version must be {PHASE2_EVALUATION_EVIDENCE_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        refs = mapping["evaluation_map_artifacts"]
        if type(refs) is not tuple:
            msg = "evaluation_map_artifacts must be a serialized sequence."
            raise TypeError(msg)
        evidence = cls(
            evaluation_row_id=cast("str", mapping["evaluation_row_id"]),
            record_checksum=cast("str", mapping["record_checksum"]),
            pipeline_result_checksum=cast("str", mapping["pipeline_result_checksum"]),
            materialization_checksum=cast("str", mapping["materialization_checksum"]),
            evaluation_provider_checksum=cast("str | None", mapping["evaluation_provider_checksum"]),
            evaluation_map_artifacts=tuple(FixedMapArtifactRef.from_dict(item) for item in refs),
        )
        if mapping["content_checksum"] != evidence.content_checksum:
            msg = "Evaluation evidence checksum changed during normalization."
            raise ValueError(msg)
        return evidence

    @classmethod
    def from_json(cls, payload: str) -> EvaluationEvidenceArtifact:
        """Construct evidence from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))


Decoder = Callable[[str], object]


class Phase2ArtifactStore:
    """Atomic artifact store and resume authority for one Phase II pipeline.

    Args:
        output_directory: Dedicated output root. An external directory is
            preferred; when it lies inside the checkout it must be excluded by
            the explicit resumability fingerprint.
        pipeline: Fully resolved Phase II pipeline configuration.
        fingerprint: Explicit tracked-source/dependency fingerprint.
        resume: Reopen and verify an existing store.
        overwrite: Remove only this store's versioned managed outputs first.
        resume_override: Checksum-sealed non-scientific override authorizing a
            specifically identified fingerprint mismatch.

    Raises:
        Phase2ResumeMismatchError: If existing output is not explicitly resumed
            or its pipeline/fingerprint differs.
        Phase2ArtifactVerificationError: If canonical evidence or a referenced
            artifact does not verify.
    """

    def __init__(
        self,
        output_directory: Path,
        pipeline: TrainingPipelineConfig,
        fingerprint: ResumabilityFingerprint,
        *,
        resume: bool = False,
        overwrite: bool = False,
        resume_override: NonScientificResumeOverride | None = None,
    ) -> None:
        """Initialize a new store or verify an existing one."""
        if not isinstance(output_directory, Path):
            msg = "output_directory must be a pathlib.Path."
            raise TypeError(msg)
        if not isinstance(pipeline, TrainingPipelineConfig):
            msg = "pipeline must be a TrainingPipelineConfig."
            raise TypeError(msg)
        if not isinstance(fingerprint, ResumabilityFingerprint):
            msg = "fingerprint must be a ResumabilityFingerprint."
            raise TypeError(msg)
        expected_prefix = pipeline.prefix_id(len(pipeline.stages) - 1)
        if fingerprint.pipeline_prefix_id != expected_prefix:
            msg = "Resumability fingerprint does not bind the pipeline's complete configured stage prefix."
            raise Phase2ResumeMismatchError(msg)
        if type(resume) is not bool or type(overwrite) is not bool:
            msg = "resume and overwrite must be bool values."
            raise TypeError(msg)
        if resume and overwrite:
            msg = "resume and overwrite are mutually exclusive."
            raise ValueError(msg)
        self.output_directory = output_directory.resolve()
        self.pipeline = pipeline
        self.fingerprint = fingerprint
        self.pipeline_config_path = self.output_directory / PIPELINE_CONFIG_NAME
        self.fingerprint_path = self.output_directory / RESUMABILITY_FINGERPRINT_NAME
        self.resume_override_stream_path = self.output_directory / RESUME_OVERRIDE_STREAM_NAME
        self.stage_result_stream_path = self.output_directory / STAGE_RESULT_STREAM_NAME
        self.stage_failure_stream_path = self.output_directory / STAGE_FAILURE_STREAM_NAME
        self.materialization_stream_path = self.output_directory / MATERIALIZATION_STREAM_NAME
        self.materialization_attempt_stream_path = self.output_directory / MATERIALIZATION_ATTEMPT_STREAM_NAME
        self.results_jsonl_path = self.output_directory / RESULTS_JSONL_NAME
        self.results_csv_path = self.output_directory / RESULTS_CSV_NAME
        self.evaluation_failure_stream_path = self.output_directory / EVALUATION_FAILURE_STREAM_NAME
        self.evaluation_evidence_stream_path = self.output_directory / EVALUATION_EVIDENCE_STREAM_NAME
        self.manifest_path = self.output_directory / MANIFEST_NAME
        self.checkpoint_directory = self.output_directory / CHECKPOINT_DIRECTORY
        self.trace_directory = self.output_directory / TRACE_DIRECTORY
        self.stage_metadata_directory = self.output_directory / STAGE_METADATA_DIRECTORY
        self.fixed_map_directory = self.output_directory / FIXED_MAP_DIRECTORY
        self.circuit_directory = self.output_directory / CIRCUIT_DIRECTORY
        self.trajectory_directory = self.output_directory / TRAJECTORY_DIRECTORY

        self._validate_managed_storage_roots()
        preflight_managed_exists = self._managed_output_exists()
        if preflight_managed_exists and not resume and not overwrite:
            msg = "Existing Phase II output requires resume=True or overwrite=True."
            raise Phase2ResumeMismatchError(msg)
        if resume and not preflight_managed_exists:
            msg = "resume=True requires an existing Phase II artifact store."
            raise Phase2ResumeMismatchError(msg)
        external_checkpoint_payload: bytes | None = None
        if self.pipeline.stages[0].input_checkpoint_checksum is not None and (
            overwrite or not preflight_managed_exists
        ):
            external_checkpoint_payload = self._read_external_checkpoint_source()

        self.output_directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_directory / _STORE_LOCK_NAME
        if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
            msg = "Phase II artifact-store lock must be a regular file, never a symbolic link."
            raise Phase2ArtifactVerificationError(msg)
        self._store_lock = FileLock(lock_path, timeout=_STORE_LOCK_TIMEOUT_SECONDS)
        self._retained_manifest_file_checksum: str | None = None
        self._mutation_requires_reopen = False
        with self._exclusive_store_lock():
            self._initialize_locked(
                pipeline=pipeline,
                resume=resume,
                overwrite=overwrite,
                resume_override=resume_override,
                external_checkpoint_payload=external_checkpoint_payload,
            )
            self._retained_manifest_file_checksum = self._verified_commit_baseline_checksum()

    def _initialize_locked(
        self,
        *,
        pipeline: TrainingPipelineConfig,
        resume: bool,
        overwrite: bool,
        resume_override: NonScientificResumeOverride | None,
        external_checkpoint_payload: bytes | None,
    ) -> None:
        """Create, recover, or verify this store while holding its writer lock."""
        self._validate_managed_storage_roots()
        managed_exists = self._managed_output_exists()
        if managed_exists and not resume and not overwrite:
            msg = "Existing Phase II output requires resume=True or overwrite=True."
            raise Phase2ResumeMismatchError(msg)
        if resume and not managed_exists:
            msg = "resume=True requires an existing Phase II artifact store."
            raise Phase2ResumeMismatchError(msg)
        if (
            self.pipeline.stages[0].input_checkpoint_checksum is not None
            and (overwrite or not managed_exists)
            and external_checkpoint_payload is None
        ):
            external_checkpoint_payload = self._read_external_checkpoint_source()
        if overwrite:
            self._remove_managed_outputs()
            managed_exists = False

        for name in _MANAGED_DIRECTORIES:
            directory = self.output_directory / name
            if directory.is_symlink():
                msg = f"Managed Phase II directory {name!r} must not be a symbolic link."
                raise Phase2ArtifactVerificationError(msg)
            directory.mkdir(parents=True, exist_ok=True)
        self._cleanup_temporary_files()

        self._resume_overrides: list[NonScientificResumeOverride] = []
        self._resume_override_write_pending = False
        self._fingerprint_write_pending = False
        stored_manifest: Mapping[str, object] | None = None
        stored_fingerprint: ResumabilityFingerprint | None = None
        if managed_exists:
            stored_fingerprint = self._open_existing(resume_override)
            stored_manifest = self._read_stored_manifest()
        else:
            atomic_write_bytes(self.pipeline_config_path, f"{pipeline.to_json()}\n".encode())
            atomic_write_bytes(self.fingerprint_path, f"{self.fingerprint.to_json()}\n".encode())
            self._initialize_empty_ledgers()
        if self.pipeline.stages[0].input_checkpoint_checksum is not None:
            if managed_exists:
                self._verify_sealed_external_checkpoint()
            else:
                assert external_checkpoint_payload is not None
                self._ingest_external_checkpoint(external_checkpoint_payload)

        raw_stage_artifacts, stage_tail = self._read_stream(
            self.stage_result_stream_path,
            PersistedStageArtifact.from_json,
            "stage-result stream",
        )
        self._stage_artifacts = cast("list[PersistedStageArtifact]", raw_stage_artifacts)
        raw_stage_failures, failure_tail = self._read_stream(
            self.stage_failure_stream_path,
            StageFailureArtifact.from_json,
            "stage-failure stream",
        )
        self._stage_failures = cast("list[StageFailureArtifact]", raw_stage_failures)
        raw_materializations, materialization_tail = self._read_stream(
            self.materialization_stream_path,
            MaterializedCircuitArtifact.from_json,
            "materialization stream",
        )
        self._materializations = cast("list[MaterializedCircuitArtifact]", raw_materializations)
        raw_materialization_attempts, materialization_attempt_tail = self._read_stream(
            self.materialization_attempt_stream_path,
            MaterializationAttemptArtifact.from_json,
            "materialization-attempt stream",
        )
        self._materialization_attempts = cast("list[MaterializationAttemptArtifact]", raw_materialization_attempts)
        raw_records, result_tail = self._read_stream(
            self.results_jsonl_path,
            pipeline_benchmark_record_from_json,
            "evaluation-result stream",
        )
        self._records = cast("list[PipelineBenchmarkRecord]", raw_records)
        raw_evaluation_failures, evaluation_failure_tail = self._read_stream(
            self.evaluation_failure_stream_path,
            pipeline_benchmark_record_from_json,
            "evaluation-failure stream",
        )
        if not all(isinstance(item, PipelineBenchmarkFailure) for item in raw_evaluation_failures):
            msg = "Evaluation-failure stream may contain only structured failure records."
            raise Phase2ArtifactVerificationError(msg)
        self._evaluation_failures = cast("list[PipelineBenchmarkFailure]", raw_evaluation_failures)
        raw_evidence, evidence_tail = self._read_stream(
            self.evaluation_evidence_stream_path,
            EvaluationEvidenceArtifact.from_json,
            "evaluation-evidence stream",
        )
        self._evaluation_evidence = cast("list[EvaluationEvidenceArtifact]", raw_evidence)
        recovered_materialization_attempt, recovered_result = self._reconcile_cross_stream_commits()
        materialization_attempt_tail = materialization_attempt_tail or recovered_materialization_attempt
        result_tail = result_tail or recovered_result
        if stored_manifest is not None:
            assert stored_fingerprint is not None
            self._verify_manifest_baseline(stored_manifest, stored_fingerprint)
        successful_row_ids = {
            record.evaluation_row_id for record in self._records if isinstance(record, PipelineBenchmarkResult)
        }
        committed_evidence = [
            item for item in self._evaluation_evidence if item.evaluation_row_id in successful_row_ids
        ]
        if len(committed_evidence) != len(self._evaluation_evidence):
            self._evaluation_evidence = committed_evidence
            evidence_tail = True
        self._validate_loaded_state()
        self._verify_referenced_artifacts()
        self._cleanup_orphan_artifacts()
        if stage_tail:
            self._write_stage_stream()
        if failure_tail:
            self._write_stage_failure_stream()
        if materialization_tail:
            self._write_materialization_stream()
        if materialization_attempt_tail:
            self._write_materialization_attempt_stream()
        if result_tail:
            self._write_result_stream()
        if evaluation_failure_tail:
            self._write_evaluation_failure_stream()
        if evidence_tail:
            self._write_evaluation_evidence_stream()
        if self._resume_override_write_pending:
            self._write_generic_stream(self.resume_override_stream_path, self._resume_overrides)
        if self._fingerprint_write_pending:
            atomic_write_bytes(self.fingerprint_path, f"{self.fingerprint.to_json()}\n".encode())
        self._write_csv()
        self._write_manifest()

    @contextlib.contextmanager
    def _exclusive_store_lock(self) -> Iterator[None]:
        """Serialize one constructor or mutation across threads and processes."""
        try:
            with self._store_lock:
                yield
        except Timeout as error:
            msg = "Another writer currently owns this Phase II artifact store; retry after it finishes."
            raise Phase2ConcurrentMutationError(msg) from error

    def _verified_commit_baseline_checksum(self) -> str:
        """Verify the retained commit roots and return the exact manifest checksum."""
        manifest = self._read_stored_manifest()
        stream_checksums = require_mapping(manifest["canonical_stream_checksums"], "canonical_stream_checksums")
        require_exact_keys(stream_checksums, frozenset(_CANONICAL_LEDGER_NAMES), "canonical_stream_checksums")
        for name in _CANONICAL_LEDGER_NAMES:
            try:
                payload = (self.output_directory / name).read_bytes()
            except OSError as error:
                msg = f"Canonical stream {name!r} could not be read while checking the writer baseline: {error}."
                raise Phase2ArtifactVerificationError(msg) from error
            expected = require_checksum(stream_checksums[name], f"canonical_stream_checksums.{name}")
            if _sha256(payload) != expected:
                msg = f"Canonical stream {name!r} changed without a matching manifest commit."
                raise Phase2ArtifactVerificationError(msg)
        try:
            stored_pipeline = TrainingPipelineConfig.from_json(self.pipeline_config_path.read_text(encoding="utf-8"))
            stored_fingerprint = ResumabilityFingerprint.from_json(self.fingerprint_path.read_text(encoding="utf-8"))
            manifest_payload = self.manifest_path.read_bytes()
        except (OSError, TypeError, ValueError) as error:
            msg = f"Could not verify the Phase II writer baseline roots: {error}."
            raise Phase2ArtifactVerificationError(msg) from error
        if (
            manifest["pipeline_training_id"] != stored_pipeline.training_id
            or manifest["pipeline_configuration_checksum"] != stored_pipeline.configuration_checksum
        ):
            msg = "The retained manifest does not match the stored pipeline root."
            raise Phase2ArtifactVerificationError(msg)
        if manifest["active_runtime_fingerprint_checksum"] != stored_fingerprint.content_checksum:
            msg = "The retained manifest does not match the stored runtime fingerprint root."
            raise Phase2ArtifactVerificationError(msg)
        return _sha256(manifest_payload)

    def _require_fresh_baseline_unlocked(self) -> str:
        """Return the current baseline only when this handle may still write."""
        if self._mutation_requires_reopen:
            msg = "This Phase II artifact-store handle observed an incomplete commit and must be reopened."
            raise Phase2ConcurrentMutationError(msg)
        try:
            current = self._verified_commit_baseline_checksum()
        except Phase2ArtifactVerificationError as error:
            msg = "The Phase II artifact store changed outside this handle or needs recovery; reopen it."
            raise Phase2ConcurrentMutationError(msg) from error
        if self._retained_manifest_file_checksum != current:
            msg = "The Phase II artifact store advanced after this handle opened; reopen before writing."
            raise Phase2ConcurrentMutationError(msg)
        return current

    def require_fresh_handle(self) -> None:
        """Fail before expensive work if this handle no longer owns the commit baseline."""
        with self._exclusive_store_lock():
            self._require_fresh_baseline_unlocked()

    @contextlib.contextmanager
    def _mutation_guard(self) -> Iterator[None]:
        """Require a fresh retained commit before and after one store mutation."""
        with self._exclusive_store_lock():
            before = self._require_fresh_baseline_unlocked()
            try:
                yield
            except BaseException:
                try:
                    after_failure = self._verified_commit_baseline_checksum()
                except Phase2ArtifactVerificationError:
                    self._mutation_requires_reopen = True
                else:
                    if after_failure != before:
                        self._mutation_requires_reopen = True
                raise
            try:
                self._retained_manifest_file_checksum = self._verified_commit_baseline_checksum()
            except Phase2ArtifactVerificationError as error:
                self._mutation_requires_reopen = True
                msg = "The Phase II mutation did not close with a complete manifest commit; reopen the store."
                raise Phase2ConcurrentMutationError(msg) from error

    def _reconcile_cross_stream_commits(self) -> tuple[bool, bool]:
        """Recover canonical paired-ledger commits after a process interruption.

        Materializations and evaluation-failure history are written before
        their redundant attempt/current-row ledgers.  They are therefore the
        authorities when an interruption lands between those atomic writes.
        """
        recovered_attempt = False
        for materialization in self._materializations:
            successful_attempts = tuple(
                attempt
                for attempt in self._materialization_attempts
                if attempt.materialized_circuit_id == materialization.materialized_circuit_id
                and attempt.status == "success"
            )
            if successful_attempts:
                continue
            circuit_attempts = tuple(
                attempt.attempt
                for attempt in self._materialization_attempts
                if attempt.materialized_circuit_id == materialization.materialized_circuit_id
            )
            self._materialization_attempts.append(
                MaterializationAttemptArtifact(
                    materialized_circuit_id=materialization.materialized_circuit_id,
                    pipeline_training_id=materialization.pipeline_training_id,
                    pipeline_result_checksum=materialization.pipeline_result_checksum,
                    attempt=max(circuit_attempts, default=0) + 1,
                    status="success",
                    phase="materialization",
                    payload_checksum=materialization.payload_checksum,
                    exception_type=None,
                    message=None,
                    wall_time_seconds=materialization.wall_time_seconds,
                    peak_memory_bytes=materialization.peak_memory_bytes,
                    runtime_fingerprint_checksum=materialization.runtime_fingerprint_checksum,
                )
            )
            recovered_attempt = True

        latest_failure_by_row: dict[str, PipelineBenchmarkFailure] = {}
        for failure in self._evaluation_failures:
            previous = latest_failure_by_row.get(failure.evaluation_row_id)
            if previous is None or failure.attempt > previous.attempt:
                latest_failure_by_row[failure.evaluation_row_id] = failure
        recovered_result = False
        for row_id, failure in latest_failure_by_row.items():
            current_index = next(
                (index for index, record in enumerate(self._records) if record.evaluation_row_id == row_id),
                None,
            )
            if current_index is not None and isinstance(self._records[current_index], PipelineBenchmarkResult):
                continue
            if current_index is None:
                self._records.append(failure)
                recovered_result = True
            elif self._records[current_index] != failure:
                self._records[current_index] = failure
                recovered_result = True
        return recovered_attempt, recovered_result

    def _initialize_empty_ledgers(self) -> None:
        """Create every canonical ledger so later deletion is detectable."""
        for name in _CANONICAL_LEDGER_NAMES:
            atomic_write_bytes(self.output_directory / name, b"")

    def _read_stored_manifest(self) -> Mapping[str, object]:
        """Read the prior checksum-sealed commit baseline on resume."""
        try:
            payload = self.manifest_path.read_bytes()
        except OSError as error:
            msg = f"Required Phase II manifest could not be read: {error}."
            raise Phase2ArtifactVerificationError(msg) from error
        if not payload.endswith(b"\n"):
            msg = "Stored Phase II manifest is not a complete canonical document."
            raise Phase2ArtifactVerificationError(msg)
        try:
            mapping = verify_sealed_mapping(
                load_canonical_json_object(payload.decode("utf-8")),
                expected_keys=_MANIFEST_KEYS,
                name="Phase II artifact manifest",
            )
        except (UnicodeDecodeError, TypeError, ValueError) as error:
            msg = f"Stored Phase II manifest is invalid: {error}."
            raise Phase2ArtifactVerificationError(msg) from error
        if mapping["manifest_format"] != PHASE2_ARTIFACT_MANIFEST_FORMAT:
            msg = "Stored Phase II manifest uses an unsupported format."
            raise Phase2ArtifactVerificationError(msg)
        return mapping

    @staticmethod
    def _manifest_sequence(manifest: Mapping[str, object], name: str) -> tuple[object, ...]:
        """Return one strictly serialized manifest sequence."""
        value = manifest[name]
        if type(value) is not tuple:
            msg = f"Manifest field {name!r} must be a serialized sequence."
            raise Phase2ArtifactVerificationError(msg)
        return value

    @staticmethod
    def _require_manifest_prefix(
        stored: tuple[object, ...],
        current: tuple[object, ...],
        name: str,
    ) -> None:
        """Reject removal or rewriting of a previously committed ledger prefix."""
        if len(current) < len(stored) or current[: len(stored)] != stored:
            msg = f"Canonical {name} no longer contains its manifest-committed prefix."
            raise Phase2ArtifactVerificationError(msg)

    @staticmethod
    def _record_index(records: Sequence[PipelineBenchmarkRecord]) -> tuple[dict[str, object], ...]:
        """Return ordered identity/checksum metadata for current evaluation rows."""
        return tuple(
            {
                "evaluation_row_id": record.evaluation_row_id,
                "record_checksum": record.content_checksum,
                "status": "success" if isinstance(record, PipelineBenchmarkResult) else "failure",
                "attempt": None if isinstance(record, PipelineBenchmarkResult) else record.attempt,
            }
            for record in records
        )

    def _verify_manifest_baseline(
        self,
        manifest: Mapping[str, object],
        stored_fingerprint: ResumabilityFingerprint,
    ) -> None:
        """Reject deletion or mutation of evidence committed by the prior manifest."""
        if (
            manifest["pipeline_training_id"] != self.pipeline.training_id
            or manifest["pipeline_configuration_checksum"] != self.pipeline.configuration_checksum
        ):
            msg = "Stored manifest does not identify the active training pipeline."
            raise Phase2ArtifactVerificationError(msg)
        try:
            manifest_runtime = require_checksum(
                manifest["active_runtime_fingerprint_checksum"],
                "active_runtime_fingerprint_checksum",
            )
        except (TypeError, ValueError) as error:
            msg = f"Stored manifest runtime fingerprint is invalid: {error}."
            raise Phase2ArtifactVerificationError(msg) from error
        stored_runtime = stored_fingerprint.content_checksum
        current_runtime = self.runtime_fingerprint_checksum
        self._validate_resume_override_runtime_state(
            manifest_runtime=manifest_runtime,
            stored_fingerprint=stored_fingerprint,
        )
        if manifest_runtime not in {stored_runtime, current_runtime}:
            manifest_fingerprint = next(
                (
                    fingerprint
                    for override in self._resume_overrides
                    for fingerprint in (override.stored_fingerprint, override.current_fingerprint)
                    if fingerprint.content_checksum == manifest_runtime
                ),
                None,
            )
            explained_transition = any(
                override.stored_fingerprint == manifest_fingerprint
                and override.current_fingerprint == stored_fingerprint
                for override in self._resume_overrides
            )
            if manifest_fingerprint is None or not explained_transition:
                msg = "Stored resumability fingerprint rolls back the manifest's active runtime without an override."
                raise Phase2ArtifactVerificationError(msg)

        sequences = (
            (
                "resume_override_checksums",
                tuple(item.content_checksum for item in self._resume_overrides),
                RESUME_OVERRIDE_STREAM_NAME,
            ),
            (
                "completed_stage_artifact_checksums",
                tuple(artifact.content_checksum for artifact in self._stage_artifacts),
                STAGE_RESULT_STREAM_NAME,
            ),
            (
                "stage_failure_checksums",
                tuple(failure.content_checksum for failure in self._stage_failures),
                STAGE_FAILURE_STREAM_NAME,
            ),
            (
                "materialization_checksums",
                tuple(item.content_checksum for item in self._materializations),
                MATERIALIZATION_STREAM_NAME,
            ),
            (
                "materialization_attempt_checksums",
                tuple(item.content_checksum for item in self._materialization_attempts),
                MATERIALIZATION_ATTEMPT_STREAM_NAME,
            ),
            (
                "evaluation_failure_attempt_checksums",
                tuple(item.content_checksum for item in self._evaluation_failures),
                EVALUATION_FAILURE_STREAM_NAME,
            ),
            (
                "evaluation_evidence_checksums",
                tuple(item.content_checksum for item in self._evaluation_evidence),
                EVALUATION_EVIDENCE_STREAM_NAME,
            ),
        )
        stream_checksums = require_mapping(manifest["canonical_stream_checksums"], "canonical_stream_checksums")
        require_exact_keys(stream_checksums, frozenset(_CANONICAL_LEDGER_NAMES), "canonical_stream_checksums")
        for field_name, current, stream_name in sequences:
            stored = self._manifest_sequence(manifest, field_name)
            self._require_manifest_prefix(stored, cast("tuple[object, ...]", current), field_name)
            if stored == current:
                expected = require_checksum(stream_checksums[stream_name], f"canonical_stream_checksums.{stream_name}")
                try:
                    actual = _sha256((self.output_directory / stream_name).read_bytes())
                except OSError as error:
                    msg = f"Canonical stream {stream_name!r} could not be read: {error}."
                    raise Phase2ArtifactVerificationError(msg) from error
                if actual != expected:
                    msg = f"Canonical stream {stream_name!r} differs from its committed manifest checksum."
                    raise Phase2ArtifactVerificationError(msg)

        stored_records = self._manifest_sequence(manifest, "evaluation_record_index")
        current_records = self._record_index(self._records)
        if len(current_records) < len(stored_records):
            msg = "Canonical evaluation results lost a manifest-committed row."
            raise Phase2ArtifactVerificationError(msg)
        failure_checksums = {failure.content_checksum for failure in self._evaluation_failures}
        for index, raw_stored in enumerate(stored_records):
            stored = require_mapping(raw_stored, f"evaluation_record_index[{index}]")
            require_exact_keys(
                stored,
                frozenset({"evaluation_row_id", "record_checksum", "status", "attempt"}),
                f"evaluation_record_index[{index}]",
            )
            current = current_records[index]
            if current["evaluation_row_id"] != stored["evaluation_row_id"]:
                msg = "Canonical evaluation-result order differs from its committed manifest."
                raise Phase2ArtifactVerificationError(msg)
            if stored["status"] == "success":
                if current != stored:
                    msg = "A successful evaluation row changed after its manifest commit."
                    raise Phase2ArtifactVerificationError(msg)
            elif stored["status"] == "failure":
                old_checksum = require_checksum(stored["record_checksum"], "record_checksum")
                if old_checksum not in failure_checksums:
                    msg = "A manifest-committed evaluation failure disappeared from append-only history."
                    raise Phase2ArtifactVerificationError(msg)
                if current["status"] == "failure" and cast("int", current["attempt"]) < require_int(
                    stored["attempt"],
                    "attempt",
                    minimum=1,
                ):
                    msg = "A current evaluation row regressed to an older failure attempt."
                    raise Phase2ArtifactVerificationError(msg)
            else:
                msg = "Manifest evaluation status must be 'success' or 'failure'."
                raise Phase2ArtifactVerificationError(msg)
        if tuple(stored_records) == cast("tuple[object, ...]", current_records):
            expected_results = require_checksum(
                stream_checksums[RESULTS_JSONL_NAME],
                f"canonical_stream_checksums.{RESULTS_JSONL_NAME}",
            )
            if _sha256(self.results_jsonl_path.read_bytes()) != expected_results:
                msg = "Canonical evaluation-result stream differs from its committed manifest checksum."
                raise Phase2ArtifactVerificationError(msg)

        stored_inventory = set(self._manifest_sequence(manifest, "artifact_inventory"))
        if not stored_inventory.issubset(self._referenced_relative_paths()):
            msg = "Manifest-committed artifact references were removed from canonical evidence."
            raise Phase2ArtifactVerificationError(msg)

    def _validate_resume_override_runtime_state(
        self,
        *,
        manifest_runtime: str,
        stored_fingerprint: ResumabilityFingerprint,
    ) -> None:
        """Bind manifest, fingerprint file, and requested runtime to the chain endpoint."""
        if not self._resume_overrides:
            return
        last_override = self._resume_overrides[-1]
        predecessor = last_override.stored_fingerprint.content_checksum
        endpoint = last_override.current_fingerprint.content_checksum
        stored_runtime = stored_fingerprint.content_checksum
        requested_runtime = self.runtime_fingerprint_checksum
        if manifest_runtime == endpoint:
            if requested_runtime != endpoint or stored_runtime not in {predecessor, endpoint}:
                msg = "Stored resumability fingerprint rolls back the manifest's active runtime override chain."
                raise Phase2ArtifactVerificationError(msg)
            return
        if manifest_runtime == predecessor:
            if requested_runtime != endpoint or stored_runtime not in {predecessor, endpoint}:
                msg = "Only the final torn-forward resume-override transition may be recovered."
                raise Phase2ArtifactVerificationError(msg)
            return
        msg = "Resume-override chain does not extend the manifest's active runtime endpoint."
        raise Phase2ArtifactVerificationError(msg)

    @property
    def runtime_fingerprint_checksum(self) -> str:
        """Active execution/dependency fingerprint checksum."""
        return self.fingerprint.content_checksum

    @property
    def stage_artifacts(self) -> tuple[PersistedStageArtifact, ...]:
        """Verified completed-stage rows in pipeline order."""
        return tuple(self._stage_artifacts)

    @property
    def stage_failures(self) -> tuple[StageFailureArtifact, ...]:
        """Append-only verified stage-attempt failures."""
        return tuple(self._stage_failures)

    @property
    def records(self) -> tuple[PipelineBenchmarkRecord, ...]:
        """Canonical current evaluation rows."""
        return tuple(self._records)

    @property
    def evaluation_failures(self) -> tuple[PipelineBenchmarkFailure, ...]:
        """Append-only history of every failed evaluation attempt."""
        return tuple(self._evaluation_failures)

    @property
    def materializations(self) -> tuple[MaterializedCircuitArtifact, ...]:
        """Verified shared circuit materializations."""
        return tuple(self._materializations)

    @property
    def materialization_attempts(self) -> tuple[MaterializationAttemptArtifact, ...]:
        """Append-only shared materialization-attempt timing ledger."""
        return tuple(self._materialization_attempts)

    @property
    def completed_stage_count(self) -> int:
        """Length of the verified committed stage prefix."""
        return len(self._stage_artifacts)

    @property
    def pipeline_result(self) -> TrainingPipelineResult | None:
        """Complete typed pipeline result, or ``None`` for a partial prefix."""
        if len(self._stage_artifacts) != len(self.pipeline.stages):
            return None
        stage_results = tuple(artifact.stage_result for artifact in self._stage_artifacts)
        final = stage_results[-1]
        return TrainingPipelineResult(
            config=self.pipeline,
            stage_results=stage_results,
            final_checkpoint_path=final.produced_checkpoint_path,
            final_checkpoint_checksum=final.produced_checkpoint_checksum,
            final_checkpoint_provenance_checksum=final.checkpoint_provenance_checksum,
            wall_time_seconds=sum(result.wall_time_seconds for result in stage_results),
            peak_memory_bytes=max(result.peak_memory_bytes for result in stage_results),
            normalized_work=_sum_work([result.normalized_work for result in stage_results]),
        )

    def is_stage_completed(self, stage_index: int) -> bool:
        """Whether a zero-based stage already belongs to the verified prefix."""
        index = require_int(stage_index, "stage_index")
        return index < len(self._stage_artifacts)

    def _managed_output_exists(self) -> bool:
        """Whether any versioned root file exists."""
        return any((self.output_directory / name).exists() for name in _MANAGED_ROOT_FILES)

    def _validate_managed_storage_roots(self) -> None:
        """Reject aliases and non-regular entries at every managed storage root."""
        for name in _MANAGED_ROOT_FILES:
            path = self.output_directory / name
            if path.is_symlink() or (path.exists() and not path.is_file()):
                msg = f"Managed Phase II root file {name!r} must be a regular file, never a symbolic link."
                raise Phase2ArtifactVerificationError(msg)
        for name in _MANAGED_DIRECTORIES:
            path = self.output_directory / name
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                msg = f"Managed Phase II directory {name!r} must be a real directory, never a symbolic link."
                raise Phase2ArtifactVerificationError(msg)

    def _remove_managed_outputs(self) -> None:
        """Remove only known versioned files and directories."""
        for name in _MANAGED_ROOT_FILES:
            (self.output_directory / name).unlink(missing_ok=True)
        for name in _MANAGED_DIRECTORIES:
            path = self.output_directory / name
            if path.exists():
                shutil.rmtree(path)

    def _cleanup_temporary_files(self) -> None:
        """Remove abandoned atomic-write temporary files under managed roots."""
        for name in _MANAGED_ROOT_FILES:
            for path in self.output_directory.glob(f".{name}.*.tmp"):
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)
        for name in _MANAGED_DIRECTORIES:
            root = self.output_directory / name
            if not root.exists():
                continue
            for path in root.rglob(".*.tmp"):
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)

    def _open_existing(
        self,
        resume_override: NonScientificResumeOverride | None,
    ) -> ResumabilityFingerprint:
        """Verify immutable pipeline/fingerprint roots and record an override."""
        try:
            stored_pipeline = TrainingPipelineConfig.from_json(self.pipeline_config_path.read_text(encoding="utf-8"))
            stored_fingerprint = ResumabilityFingerprint.from_json(self.fingerprint_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as error:
            msg = f"Could not verify stored pipeline identity: {error}."
            raise Phase2ArtifactVerificationError(msg) from error
        if (
            stored_pipeline.training_id != self.pipeline.training_id
            or stored_pipeline.configuration_checksum != self.pipeline.configuration_checksum
        ):
            msg = "Stored pipeline configuration or stage prefix differs from the requested pipeline."
            raise Phase2ResumeMismatchError(msg)
        self.pipeline = stored_pipeline
        overrides, _ = self._read_stream(
            self.resume_override_stream_path,
            NonScientificResumeOverride.from_json,
            "resume-override stream",
        )
        self._resume_overrides = cast("list[NonScientificResumeOverride]", overrides)
        self._validate_resume_override_chain()
        last_override = self._resume_overrides[-1] if self._resume_overrides else None
        matching_recorded_override = (
            last_override
            if last_override is not None
            and last_override.stored_fingerprint == stored_fingerprint
            and last_override.current_fingerprint == self.fingerprint
            else None
        )
        if (
            stored_fingerprint == self.fingerprint
            and resume_override is not None
            and resume_override in self._resume_overrides
        ):
            effective_override = None
        else:
            effective_override = resume_override if resume_override is not None else matching_recorded_override
        try:
            require_resumability_match(
                stored_fingerprint,
                self.fingerprint,
                override=effective_override,
            )
        except ValueError as error:
            msg = str(error)
            raise Phase2ResumeMismatchError(msg) from error
        if resume_override is not None and resume_override not in self._resume_overrides:
            if self._resume_overrides and self._resume_overrides[-1].current_fingerprint != stored_fingerprint:
                msg = "A new resume override must extend the current endpoint of the recorded override chain."
                raise Phase2ArtifactVerificationError(msg)
            self._resume_overrides.append(resume_override)
            self._validate_resume_override_chain()
            self._resume_override_write_pending = True
        if stored_fingerprint != self.fingerprint:
            self._fingerprint_write_pending = True
        return stored_fingerprint

    def _validate_resume_override_chain(self) -> None:
        """Require one ordered, acyclic runtime-fingerprint transition chain."""
        seen_fingerprints: set[str] = set()
        previous: NonScientificResumeOverride | None = None
        for index, override in enumerate(self._resume_overrides):
            if previous is not None and override.stored_fingerprint != previous.current_fingerprint:
                msg = f"Resume-override row {index + 1} is disconnected from or forks the recorded transition chain."
                raise Phase2ArtifactVerificationError(msg)
            stored_checksum = override.stored_fingerprint.content_checksum
            current_checksum = override.current_fingerprint.content_checksum
            if index == 0:
                seen_fingerprints.add(stored_checksum)
            if current_checksum in seen_fingerprints:
                msg = f"Resume-override row {index + 1} revisits an earlier runtime fingerprint."
                raise Phase2ArtifactVerificationError(msg)
            seen_fingerprints.add(current_checksum)
            previous = override

    def _read_stream(self, path: Path, decoder: Decoder, name: str) -> tuple[list[object], bool]:
        """Read a complete canonical JSONL stream without hiding corruption."""
        if not path.exists():
            msg = f"Required canonical {name} is missing."
            raise Phase2ArtifactVerificationError(msg)
        try:
            payload = path.read_bytes()
        except OSError as error:
            msg = f"Could not read {name}: {error}."
            raise Phase2ArtifactVerificationError(msg) from error
        if payload and not payload.endswith(b"\n"):
            msg = f"{name} is not terminated by a complete canonical row."
            raise Phase2ArtifactVerificationError(msg)
        rows: list[object] = []
        lines = payload.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                msg = f"{name} contains an empty row at line {index + 1}."
                raise Phase2ArtifactVerificationError(msg)
            try:
                row = decoder(line.decode("utf-8"))
            except (UnicodeDecodeError, TypeError, ValueError) as error:
                msg = f"{name} is invalid at line {index + 1}: {error}."
                raise Phase2ArtifactVerificationError(msg) from error
            if self._serialize_row(row).encode() != line:
                msg = f"{name} contains non-canonical bytes at line {index + 1}."
                raise Phase2ArtifactVerificationError(msg)
            rows.append(row)
        return rows, False

    def _accepted_runtime_fingerprints(self) -> frozenset[str]:
        """Return active and explicitly overridden runtime fingerprints."""
        checksums = {self.runtime_fingerprint_checksum}
        for raw_override in self._resume_overrides:
            stored = getattr(raw_override, "stored_fingerprint", None)
            current = getattr(raw_override, "current_fingerprint", None)
            for fingerprint in (stored, current):
                checksum = getattr(fingerprint, "content_checksum", None)
                if type(checksum) is str:
                    checksums.add(checksum)
        return frozenset(checksums)

    def _validate_loaded_state(self) -> None:
        """Validate canonical streams as one contiguous scientific ledger."""
        accepted_runtime = self._accepted_runtime_fingerprints()
        previous: TrainingStageResult | None = None
        seen_stage_rows: set[str] = set()
        for index, artifact in enumerate(self._stage_artifacts):
            result = artifact.stage_result
            stage = self.pipeline.stages[index] if index < len(self.pipeline.stages) else None
            if stage is None:
                msg = "Stage-result stream is longer than the configured pipeline."
                raise Phase2ArtifactVerificationError(msg)
            expected = {
                "pipeline_training_id": self.pipeline.training_id,
                "pipeline_prefix_id": self.pipeline.prefix_id(index),
                "stage_index": stage.stage_index,
                "stage_id": stage.stage_id,
                "stage_configuration_checksum": stage.configuration_checksum,
                "output_topology_id": stage.output_topology_id,
                "output_parameter_count": stage.output_parameter_count,
            }
            if any(getattr(result, name) != value for name, value in expected.items()):
                msg = f"Stored stage {index} does not match its configured pipeline prefix."
                raise Phase2ArtifactVerificationError(msg)
            expected_input_checksum = (
                stage.input_checkpoint_checksum if previous is None else previous.produced_checkpoint_checksum
            )
            expected_input_provenance = (
                stage.input_checkpoint_provenance_checksum
                if previous is None
                else previous.checkpoint_provenance_checksum
            )
            if (
                result.input_checkpoint_checksum != expected_input_checksum
                or result.input_checkpoint_provenance_checksum != expected_input_provenance
            ):
                msg = f"Stored stage {index} does not continue its verified predecessor checkpoint."
                raise Phase2ArtifactVerificationError(msg)
            if artifact.runtime_fingerprint_checksum not in accepted_runtime:
                msg = f"Stored stage {index} uses an unacknowledged runtime fingerprint."
                raise Phase2ArtifactVerificationError(msg)
            if artifact.content_checksum in seen_stage_rows:
                msg = "Stage-result stream contains a duplicate immutable artifact."
                raise Phase2ArtifactVerificationError(msg)
            seen_stage_rows.add(artifact.content_checksum)
            previous = result

        failure_ids: set[str] = set()
        attempts: dict[str, set[int]] = {}
        for failure in self._stage_failures:
            if (
                failure.pipeline_training_id != self.pipeline.training_id
                or failure.pipeline_configuration_checksum != self.pipeline.configuration_checksum
                or failure.stage_index >= len(self.pipeline.stages)
            ):
                msg = "Stage-failure stream contains a failure from another pipeline."
                raise Phase2ArtifactVerificationError(msg)
            stage = self.pipeline.stages[failure.stage_index]
            if (
                failure.pipeline_prefix_id != self.pipeline.prefix_id(failure.stage_index)
                or failure.stage_id != stage.stage_id
                or failure.stage_configuration_checksum != stage.configuration_checksum
            ):
                msg = "Stage failure does not identify its exact configured prefix."
                raise Phase2ArtifactVerificationError(msg)
            if failure.runtime_fingerprint_checksum not in accepted_runtime:
                msg = "Stage failure uses an unacknowledged runtime fingerprint."
                raise Phase2ArtifactVerificationError(msg)
            if failure.failure_id in failure_ids:
                msg = "Stage-failure stream contains a duplicate failure identity."
                raise Phase2ArtifactVerificationError(msg)
            failure_ids.add(failure.failure_id)
            stage_attempts = attempts.setdefault(failure.pipeline_prefix_id, set())
            if failure.attempt in stage_attempts:
                msg = "Stage-failure stream repeats an attempt number for one stage prefix."
                raise Phase2ArtifactVerificationError(msg)
            stage_attempts.add(failure.attempt)
            expected_prefix_checksums = tuple(
                artifact.content_checksum for artifact in self._stage_artifacts[: failure.stage_index]
            )
            if failure.completed_stage_artifact_checksums != expected_prefix_checksums:
                msg = "Stage failure does not link to the exact completed predecessor prefix."
                raise Phase2ArtifactVerificationError(msg)
        if any(values != set(range(1, max(values) + 1)) for values in attempts.values()):
            msg = "Stage-failure attempt numbers must be contiguous from one."
            raise Phase2ArtifactVerificationError(msg)

        materialization_ids = [item.materialized_circuit_id for item in self._materializations]
        if len(materialization_ids) != len(set(materialization_ids)):
            msg = "Materialization stream contains a duplicate circuit identity."
            raise Phase2ArtifactVerificationError(msg)
        result = self.pipeline_result
        if self._materializations or self._records or self._evaluation_evidence:
            if result is None:
                msg = "Evaluation evidence cannot exist before the training pipeline is complete."
                raise Phase2ArtifactVerificationError(msg)
            for materialization in self._materializations:
                if (
                    materialization.pipeline_training_id != result.training_id
                    or materialization.pipeline_result_checksum != result.content_checksum
                    or materialization.final_checkpoint_checksum != result.final_checkpoint_checksum
                    or materialization.runtime_fingerprint_checksum not in accepted_runtime
                ):
                    msg = "Materialization does not link to the complete active pipeline artifact."
                    raise Phase2ArtifactVerificationError(msg)

        materialization_attempts: dict[str, set[int]] = {}
        attempt_ids: set[str] = set()
        materialization_by_id = {item.materialized_circuit_id: item for item in self._materializations}
        for attempt in self._materialization_attempts:
            if result is None:
                msg = "Materialization attempts cannot exist before the training pipeline is complete."
                raise Phase2ArtifactVerificationError(msg)
            if (
                attempt.pipeline_training_id != result.training_id
                or attempt.pipeline_result_checksum != result.content_checksum
                or attempt.runtime_fingerprint_checksum not in accepted_runtime
            ):
                msg = "Materialization attempt does not link to the active complete pipeline."
                raise Phase2ArtifactVerificationError(msg)
            if attempt.attempt_id in attempt_ids:
                msg = "Materialization-attempt stream contains a duplicate attempt identity."
                raise Phase2ArtifactVerificationError(msg)
            attempt_ids.add(attempt.attempt_id)
            circuit_attempts = materialization_attempts.setdefault(attempt.materialized_circuit_id, set())
            if attempt.attempt in circuit_attempts:
                msg = "Materialization-attempt stream repeats a circuit-local attempt number."
                raise Phase2ArtifactVerificationError(msg)
            circuit_attempts.add(attempt.attempt)
            if attempt.status == "success":
                materialization = materialization_by_id.get(attempt.materialized_circuit_id)
                if materialization is None or attempt.payload_checksum != materialization.payload_checksum:
                    msg = "Successful materialization attempt does not identify its persisted circuit artifact."
                    raise Phase2ArtifactVerificationError(msg)
        if any(values != set(range(1, max(values) + 1)) for values in materialization_attempts.values()):
            msg = "Materialization attempt numbers must be contiguous from one."
            raise Phase2ArtifactVerificationError(msg)

        row_ids: set[str] = set()
        for record in self._records:
            if record.evaluation_row_id in row_ids:
                msg = "Canonical evaluation stream contains a duplicate evaluation-row identity."
                raise Phase2ArtifactVerificationError(msg)
            row_ids.add(record.evaluation_row_id)
            assert result is not None
            record.config.validate_against_pipeline(result)
            if record.runtime_fingerprint_checksum not in accepted_runtime:
                msg = "Evaluation row uses an unacknowledged runtime fingerprint."
                raise Phase2ArtifactVerificationError(msg)
        failure_checksums: set[str] = set()
        evaluation_attempts: dict[str, set[int]] = {}
        for failure in self._evaluation_failures:
            assert result is not None
            failure.config.validate_against_pipeline(result)
            if failure.runtime_fingerprint_checksum not in accepted_runtime:
                msg = "Evaluation failure uses an unacknowledged runtime fingerprint."
                raise Phase2ArtifactVerificationError(msg)
            if failure.content_checksum in failure_checksums:
                msg = "Evaluation-failure history contains a duplicate attempt."
                raise Phase2ArtifactVerificationError(msg)
            failure_checksums.add(failure.content_checksum)
            row_attempts = evaluation_attempts.setdefault(failure.evaluation_row_id, set())
            if failure.attempt in row_attempts:
                msg = "Evaluation-failure history repeats an attempt number for one row."
                raise Phase2ArtifactVerificationError(msg)
            row_attempts.add(failure.attempt)
        if any(values != set(range(1, max(values) + 1)) for values in evaluation_attempts.values()):
            msg = "Evaluation-failure attempt numbers must be contiguous from one."
            raise Phase2ArtifactVerificationError(msg)
        latest_failure_by_row = {
            row_id: max(
                (failure for failure in self._evaluation_failures if failure.evaluation_row_id == row_id),
                key=lambda failure: failure.attempt,
            )
            for row_id in evaluation_attempts
        }
        for record in self._records:
            if (
                isinstance(record, PipelineBenchmarkFailure)
                and latest_failure_by_row.get(record.evaluation_row_id) != record
            ):
                msg = "A current failed row must equal its latest append-only failure attempt."
                raise Phase2ArtifactVerificationError(msg)
        evidence_ids = [item.evaluation_row_id for item in self._evaluation_evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            msg = "Evaluation-evidence stream contains a duplicate row identity."
            raise Phase2ArtifactVerificationError(msg)
        record_by_id = {record.evaluation_row_id: record for record in self._records}
        materialization_checksums = {item.content_checksum for item in self._materializations}
        for evidence in self._evaluation_evidence:
            record = record_by_id.get(evidence.evaluation_row_id)
            if not isinstance(record, PipelineBenchmarkResult):
                msg = "Evaluation evidence must link to a current successful result row."
                raise Phase2ArtifactVerificationError(msg)
            if evidence.record_checksum != record.content_checksum:
                msg = "Evaluation evidence record checksum does not match its result row."
                raise Phase2ArtifactVerificationError(msg)
            assert result is not None
            if (
                evidence.pipeline_result_checksum != result.content_checksum
                or evidence.materialization_checksum not in materialization_checksums
            ):
                msg = "Evaluation evidence is not linked to the complete pipeline and materialization."
                raise Phase2ArtifactVerificationError(msg)
        successful_ids = {
            record.evaluation_row_id for record in self._records if isinstance(record, PipelineBenchmarkResult)
        }
        if successful_ids != set(evidence_ids):
            msg = "Every successful result row requires exactly one evaluation-evidence row."
            raise Phase2ArtifactVerificationError(msg)
        self._validate_global_map_isolation()

    def _validate_global_map_isolation(self) -> None:
        """Reject fixed-map identity or content reuse across scientific roles."""
        refs = [ref for artifact in self._stage_artifacts for ref in artifact.fixed_map_artifacts]
        refs.extend(ref for evidence in self._evaluation_evidence for ref in evidence.evaluation_map_artifacts)
        by_id: dict[str, str] = {}
        by_checksum: dict[str, str] = {}
        for ref in refs:
            prior_role = by_id.setdefault(ref.ensemble_id, ref.role)
            prior_checksum_role = by_checksum.setdefault(ref.content_checksum, ref.role)
            if prior_role != ref.role or prior_checksum_role != ref.role:
                msg = "A fixed-map identity or checksum is reused across scientific roles."
                raise Phase2ArtifactVerificationError(msg)
            if prior_role == ref.role and sum(item.ensemble_id == ref.ensemble_id for item in refs) > 1:
                msg = "A fixed-map ensemble is reused by more than one stored evidence record."
                raise Phase2ArtifactVerificationError(msg)

    def _resolve_managed_relative(self, relative_path: str) -> Path:
        """Resolve a regular artifact path without accepting filesystem aliases."""
        relative = require_relative_path(relative_path, "relative_path")
        path = self.output_directory / relative
        current = self.output_directory
        for part in Path(relative).parts:
            current /= part
            if current.is_symlink():
                msg = f"Managed artifact path {relative!r} must not contain symbolic links."
                raise Phase2ArtifactVerificationError(msg)
        path = path.resolve()
        if not path.is_relative_to(self.output_directory):
            msg = f"Managed artifact path {relative!r} escapes the output directory."
            raise Phase2ArtifactVerificationError(msg)
        if path.exists() and not path.is_file():
            msg = f"Managed artifact path {relative!r} must identify a regular file."
            raise Phase2ArtifactVerificationError(msg)
        return path

    def _read_verified_bytes(self, relative_path: str, checksum: str, *, maximum_size: int) -> bytes:
        """Read bounded exact bytes and verify their file checksum."""
        path = self._resolve_managed_relative(relative_path)
        try:
            with path.open("rb") as stream:
                payload = stream.read(maximum_size + 1)
        except OSError as error:
            msg = f"Required Phase II artifact {relative_path!r} could not be read: {error}."
            raise Phase2ArtifactVerificationError(msg) from error
        if len(payload) > maximum_size:
            msg = f"Artifact {relative_path!r} exceeds its configured verification limit."
            raise Phase2ArtifactVerificationError(msg)
        actual = _sha256(payload)
        if actual != checksum:
            msg = f"Artifact checksum mismatch for {relative_path!r}: expected {checksum}, computed {actual}."
            raise Phase2ArtifactVerificationError(msg)
        return payload

    def _verify_map_ref(self, ref: FixedMapArtifactRef) -> KrotovFixedMapEnsemble:
        """Checksum, decode, and role-verify one fixed-map artifact."""
        payload = self._read_verified_bytes(ref.path, ref.file_checksum, maximum_size=_MAX_FIXED_MAP_SIZE)
        try:
            ensemble = KrotovFixedMapEnsemble.from_json(payload.decode("utf-8"))
        except (UnicodeDecodeError, TypeError, ValueError) as error:
            msg = f"Fixed-map artifact {ref.path!r} is invalid: {error}."
            raise Phase2ArtifactVerificationError(msg) from error
        if (
            ensemble.role != ref.role
            or ensemble.ensemble_id != ref.ensemble_id
            or ensemble.content_checksum != ref.content_checksum
        ):
            msg = f"Fixed-map artifact {ref.path!r} does not match its canonical reference."
            raise Phase2ArtifactVerificationError(msg)
        return ensemble

    def _verify_referenced_artifacts(self) -> None:
        """Verify every checkpoint, trace, metadata, map, circuit, and sidecar."""
        optimized_objective_checksums: set[str] = set()
        for artifact in self._stage_artifacts:
            result = artifact.stage_result
            checkpoint_payload = self._read_verified_bytes(
                result.produced_checkpoint_path,
                artifact.checkpoint_file_checksum,
                maximum_size=_MAX_CHECKPOINT_SIZE,
            )
            try:
                checkpoint = StageParameterCheckpoint.from_bytes(
                    checkpoint_payload,
                    expected_checksum=artifact.checkpoint_file_checksum,
                )
            except (TypeError, ValueError) as error:
                msg = f"Stage checkpoint {result.produced_checkpoint_path!r} is invalid: {error}."
                raise Phase2ArtifactVerificationError(msg) from error
            if (
                checkpoint.pipeline_training_id != result.pipeline_training_id
                or checkpoint.pipeline_prefix_id != result.pipeline_prefix_id
                or checkpoint.stage_index != result.stage_index
                or checkpoint.stage_id != result.stage_id
                or checkpoint.stage_configuration_checksum != result.stage_configuration_checksum
                or checkpoint.selected_theta.size != result.output_parameter_count
            ):
                msg = "Stage checkpoint identity or parameter count does not match its canonical stage row."
                raise Phase2ArtifactVerificationError(msg)

            assert result.optimizer_trace_path is not None
            assert result.optimizer_trace_checksum is not None
            trace_payload = self._read_verified_bytes(
                result.optimizer_trace_path,
                result.optimizer_trace_checksum,
                maximum_size=_MAX_TRACE_SIZE,
            )
            try:
                trace_mapping = verify_sealed_mapping(
                    load_canonical_json_object(trace_payload.decode("utf-8")),
                    expected_keys=_TRACE_KEYS,
                    name="optimizer trace artifact",
                )
            except (UnicodeDecodeError, TypeError, ValueError) as error:
                msg = f"Optimizer trace {result.optimizer_trace_path!r} is invalid: {error}."
                raise Phase2ArtifactVerificationError(msg) from error
            if (
                trace_mapping["schema_version"] != PHASE2_TRACE_SCHEMA_VERSION
                or trace_mapping["pipeline_training_id"] != result.pipeline_training_id
                or trace_mapping["pipeline_prefix_id"] != result.pipeline_prefix_id
                or trace_mapping["stage_configuration_checksum"] != result.stage_configuration_checksum
            ):
                msg = "Optimizer trace does not match its canonical stage row."
                raise Phase2ArtifactVerificationError(msg)

            assert result.diagnostic_sidecar_path is not None
            assert result.diagnostic_sidecar_checksum is not None
            metadata_payload = self._read_verified_bytes(
                result.diagnostic_sidecar_path,
                result.diagnostic_sidecar_checksum,
                maximum_size=_MAX_STAGE_METADATA_SIZE,
            )
            try:
                metadata = verify_sealed_mapping(
                    load_canonical_json_object(metadata_payload.decode("utf-8")),
                    expected_keys=_STAGE_METADATA_KEYS,
                    name="stage metadata artifact",
                )
            except (UnicodeDecodeError, TypeError, ValueError) as error:
                msg = f"Stage metadata {result.diagnostic_sidecar_path!r} is invalid: {error}."
                raise Phase2ArtifactVerificationError(msg) from error
            if (
                metadata["schema_version"] != PHASE2_STAGE_METADATA_SCHEMA_VERSION
                or metadata["pipeline_training_id"] != result.pipeline_training_id
                or metadata["pipeline_prefix_id"] != result.pipeline_prefix_id
                or metadata["stage_configuration_checksum"] != result.stage_configuration_checksum
                or (
                    result.stage_index > 0
                    and metadata["source_parameter_checksum"]
                    != self.load_stage_checkpoint(result.stage_index - 1).selected_parameter_checksum
                )
                or (
                    result.stage_index == 0
                    and (
                        (self.pipeline.stages[0].input_checkpoint_checksum is None)
                        != (metadata["source_parameter_checksum"] is None)
                    )
                )
                or metadata["selected_parameter_checksum"] != checkpoint.selected_parameter_checksum
                or metadata["final_parameter_checksum"] != checkpoint.final_parameter_checksum
                or metadata["selected_global_iteration"] != checkpoint.selected_global_iteration
                or metadata["completed_global_iteration"] != checkpoint.completed_global_iteration
                or metadata["selected_checkpoint_validation_fidelity"]
                != checkpoint.selected_checkpoint_validation_fidelity
                or metadata["circuit_binding_checksum"] != checkpoint.circuit_binding_checksum
                or metadata["provider_checksum"] != checkpoint.provider_checksum
                or metadata["objective_checksum"] != checkpoint.objective_checksum
                or checkpoint.stage_execution_checksum != result.training_summary.get("adapter_execution_checksum")
                or metadata["runtime_fingerprint_checksum"] != artifact.runtime_fingerprint_checksum
            ):
                msg = "Stage metadata does not match its canonical checkpoint and stage row."
                raise Phase2ArtifactVerificationError(msg)
            if result.stage_index == 0 and self.pipeline.stages[0].input_checkpoint_checksum is not None:
                external = self.load_external_checkpoint()
                if metadata["source_parameter_checksum"] != external.selected_parameter_checksum:
                    msg = "Stage metadata does not identify the verified external source parameters."
                    raise Phase2ArtifactVerificationError(msg)
            stage = self.pipeline.stages[result.stage_index]
            if stage.optimizer_id != "none":
                optimized_objective_checksums.add(
                    require_checksum(metadata["objective_checksum"], "objective_checksum")
                )
            adapter_checksum = result.training_summary.get("adapter_execution_checksum")
            objective_binding: NoisyKrotovObjectiveBinding | None = None
            if adapter_checksum is None:
                if metadata["objective_binding"] is not None:
                    msg = "Only genuine WP17 evidence may persist an objective binding."
                    raise Phase2ArtifactVerificationError(msg)
            else:
                try:
                    objective_binding = NoisyKrotovObjectiveBinding.from_dict(metadata["objective_binding"])
                    self._validate_wp17_objective_binding(
                        objective_binding,
                        objective_checksum=cast("str | None", metadata["objective_checksum"]),
                    )
                except (TypeError, ValueError) as error:
                    msg = f"Stage objective binding does not verify: {error}."
                    raise Phase2ArtifactVerificationError(msg) from error
            try:
                _validated_circuit_topology(
                    metadata["circuit_topology"],
                    stage=stage,
                    circuit_binding_checksum=cast("str | None", metadata["circuit_binding_checksum"]),
                )
                statistics_mapping = require_mapping(metadata["circuit_statistics"], "circuit_statistics")
            except (TypeError, ValueError) as error:
                msg = f"Stage topology or statistics are invalid: {error}."
                raise Phase2ArtifactVerificationError(msg) from error
            if (
                statistics_mapping.get("topology_id") != stage.output_topology_id
                or statistics_mapping.get("parameter_count") != stage.output_parameter_count
            ):
                msg = "Stage statistics do not match the configured output topology."
                raise Phase2ArtifactVerificationError(msg)
            training_refs = tuple(ref for ref in artifact.fixed_map_artifacts if ref.role == "training_trajectory")
            validation_refs = tuple(ref for ref in artifact.fixed_map_artifacts if ref.role == "checkpoint_validation")
            if canonical_checksum(metadata["training_map_artifacts"]) != canonical_checksum([
                ref.to_dict() for ref in training_refs
            ]) or canonical_checksum(metadata["checkpoint_validation_map_artifacts"]) != canonical_checksum([
                ref.to_dict() for ref in validation_refs
            ]):
                msg = "Stage metadata fixed-map references do not match the canonical stage artifact."
                raise Phase2ArtifactVerificationError(msg)
            expected_training_checksums = (
                tuple(ref.content_checksum for ref in training_refs)
                if self.pipeline.stages[result.stage_index].sampling_policy in {"crn_fixed", "crn_refresh"}
                else ()
            )
            expected_validation_checksum = (
                canonical_checksum({
                    "role": "checkpoint_validation",
                    "ensemble_checksums": [ref.content_checksum for ref in validation_refs],
                })
                if self.pipeline.stages[result.stage_index].checkpoint_validation.enabled
                and self.pipeline.stages[result.stage_index].checkpoint_validation.sampling_policy
                in {"crn_fixed", "crn_refresh"}
                else None
            )
            if (
                result.training_ensemble_checksums != expected_training_checksums
                or result.checkpoint_validation_ensemble_checksum != expected_validation_checksum
            ):
                msg = "Stage result fixed-map checksum aliases do not match its persisted map artifacts."
                raise Phase2ArtifactVerificationError(msg)
            decoded_maps: list[KrotovFixedMapEnsemble] = []
            for ref in artifact.fixed_map_artifacts:
                ensemble = self._verify_map_ref(ref)
                decoded_maps.append(ensemble)
                if (
                    ensemble.stage_configuration_checksum != result.stage_configuration_checksum
                    or ensemble.stage_index != result.stage_index
                    or ensemble.stage_id != result.stage_id
                    or ensemble.circuit_checksum != metadata["circuit_binding_checksum"]
                ):
                    msg = "Fixed-map artifact does not bind the stage that references it."
                    raise Phase2ArtifactVerificationError(msg)
            stage = self.pipeline.stages[result.stage_index]
            training_maps = tuple(item for item in decoded_maps if item.role == "training_trajectory")
            validation_maps = tuple(item for item in decoded_maps if item.role == "checkpoint_validation")
            training_coordinates = _map_schedule_coordinates(
                stage.sampling_policy,
                stage.crn_refresh_interval,
                stage.iteration_budget,
            )
            validation_coordinates: tuple[tuple[int, int, int], ...] = ()
            if stage.checkpoint_validation.enabled:
                validation_calls = 1 + math.ceil(
                    stage.iteration_budget / cast("int", stage.checkpoint_validation.cadence)
                )
                validation_coordinates = _map_schedule_coordinates(
                    stage.checkpoint_validation.sampling_policy,
                    stage.checkpoint_validation.ensemble_refresh_interval,
                    validation_calls,
                )
            for maps, coordinates, seed, count, provider_checksum in (
                (
                    training_maps,
                    training_coordinates,
                    stage.training_seed,
                    stage.trajectory_count,
                    metadata["provider_checksum"],
                ),
                (
                    validation_maps,
                    validation_coordinates,
                    stage.checkpoint_validation.seed,
                    stage.checkpoint_validation.trajectory_count,
                    metadata["checkpoint_validation_provider_checksum"],
                ),
            ):
                if tuple(
                    (item.ensemble_index, item.refresh_index, item.global_iteration_start) for item in maps
                ) != coordinates or any(
                    item.resolved_seed != seed
                    or item.trajectory_count != count
                    or item.provider_checksum != provider_checksum
                    for item in maps
                ):
                    msg = "Fixed-map artifacts do not reproduce the configured stage schedule and bindings."
                    raise Phase2ArtifactVerificationError(msg)
            if adapter_checksum is not None:
                assert objective_binding is not None
                try:
                    noisy_trace = _validate_noisy_trace_semantics(
                        stage=stage,
                        trace=trace_mapping["trace"],
                        training_ensembles=training_maps,
                        validation_ensembles=validation_maps,
                        normalized_work=result.normalized_work,
                        initial_parameter_checksum=require_checksum(
                            metadata["initial_parameter_checksum"],
                            "initial_parameter_checksum",
                        ),
                        final_parameter_checksum=checkpoint.final_parameter_checksum,
                        selected_parameter_checksum=checkpoint.selected_parameter_checksum,
                        completed_iteration=checkpoint.completed_global_iteration,
                        selected_iteration=checkpoint.selected_global_iteration,
                        selected_fidelity=checkpoint.selected_checkpoint_validation_fidelity,
                        cumulative_pairings=require_int(
                            metadata["cumulative_cross_trajectory_pairings"],
                            "cumulative_cross_trajectory_pairings",
                        ),
                        circuit_topology=cast("Mapping[str, object]", metadata["circuit_topology"]),
                        provider_checksum=cast("str | None", metadata["provider_checksum"]),
                    )
                    recomputed_execution_checksum = _noisy_execution_checksum(
                        stage=stage,
                        circuit_binding_checksum=require_checksum(
                            metadata["circuit_binding_checksum"],
                            "circuit_binding_checksum",
                        ),
                        provider_checksum=cast("str | None", metadata["provider_checksum"]),
                        objective_checksum=require_checksum(metadata["objective_checksum"], "objective_checksum"),
                        objective_binding_checksum=objective_binding.content_checksum,
                        initial_parameter_checksum=require_checksum(
                            metadata["initial_parameter_checksum"],
                            "initial_parameter_checksum",
                        ),
                        final_parameter_checksum=checkpoint.final_parameter_checksum,
                        selected_parameter_checksum=checkpoint.selected_parameter_checksum,
                        selected_iteration=checkpoint.selected_global_iteration,
                        selected_fidelity=checkpoint.selected_checkpoint_validation_fidelity,
                        trace=noisy_trace,
                        training_ensembles=training_maps,
                        validation_ensembles=validation_maps,
                        normalized_work=result.normalized_work,
                    )
                except (TypeError, ValueError) as error:
                    msg = f"WP17 stage trace does not verify: {error}."
                    raise Phase2ArtifactVerificationError(msg) from error
                if recomputed_execution_checksum != adapter_checksum:
                    msg = "WP17 adapter execution checksum does not close over persisted stage evidence."
                    raise Phase2ArtifactVerificationError(msg)
                expected_training_summary, expected_validation_summary = _noisy_execution_summaries(
                    stage=stage,
                    trace=noisy_trace,
                    adapter_checksum=recomputed_execution_checksum,
                    selected_iteration=checkpoint.selected_global_iteration,
                    selected_fidelity=checkpoint.selected_checkpoint_validation_fidelity,
                    selected_parameter_checksum=checkpoint.selected_parameter_checksum,
                    final_parameter_checksum=checkpoint.final_parameter_checksum,
                    cumulative_pairings=require_int(
                        metadata["cumulative_cross_trajectory_pairings"],
                        "cumulative_cross_trajectory_pairings",
                    ),
                    validation_ensembles=validation_maps,
                )
                if (
                    result.training_summary != expected_training_summary
                    or result.checkpoint_validation_summary != expected_validation_summary
                ):
                    msg = "WP17 stage summaries are not exactly implied by persisted trace and checkpoints."
                    raise Phase2ArtifactVerificationError(msg)

        if len(optimized_objective_checksums) > 1:
            msg = "Optimized stages in one pipeline must share one exact target/objective binding."
            raise Phase2ArtifactVerificationError(msg)

        for materialization in self._materializations:
            self._read_verified_bytes(
                materialization.path,
                materialization.payload_checksum,
                maximum_size=_MAX_CIRCUIT_SIZE,
            )
        for evidence in self._evaluation_evidence:
            record = next(item for item in self._records if item.evaluation_row_id == evidence.evaluation_row_id)
            assert isinstance(record, PipelineBenchmarkResult)
            evaluation_maps = tuple(self._verify_map_ref(ref) for ref in evidence.evaluation_map_artifacts)
            try:
                self._validate_evaluation_maps(
                    record.config,
                    evaluation_maps,
                    evidence.evaluation_provider_checksum,
                )
            except (TypeError, ValueError) as error:
                msg = f"Evaluation maps for {record.evaluation_row_id!r} are invalid: {error}."
                raise Phase2ArtifactVerificationError(msg) from error
            if record.sampled_nonidentity_events != sum(item.nonidentity_event_count for item in evaluation_maps):
                msg = "Evaluation sampled-event count does not match its persisted fixed maps."
                raise Phase2ArtifactVerificationError(msg)
            if record.trajectory_sidecar_path is not None:
                assert record.trajectory_sidecar_checksum is not None
                sidecar = self._read_verified_bytes(
                    record.trajectory_sidecar_path,
                    record.trajectory_sidecar_checksum,
                    maximum_size=max(4096, record.config.trajectory_budget * 16 + 16384),
                )
                expected_role = self._evaluation_map_role(record.config.data_role)
                map_partitions = tuple(
                    {
                        "ensemble_id": item.ensemble_id,
                        "content_checksum": item.content_checksum,
                        "trajectory_count": item.trajectory_count,
                    }
                    for item in evaluation_maps
                )
                try:
                    fidelities = read_phase2_trajectory_sidecar(
                        sidecar,
                        expected_evaluation_row_id=record.evaluation_row_id,
                        expected_pipeline_training_id=record.config.pipeline_training_id,
                        expected_map_role=expected_role,
                        expected_map_partitions=map_partitions,
                        expected_count=record.config.trajectory_budget,
                    )
                except (TypeError, ValueError) as error:
                    msg = f"Trajectory sidecar {record.trajectory_sidecar_path!r} is invalid: {error}."
                    raise Phase2ArtifactVerificationError(msg) from error
                expected_statistics = self._trajectory_statistics(record.config, fidelities)
                recorded_statistics = (
                    record.test_noisy_fidelity,
                    record.noisy_fidelity_standard_deviation,
                    record.noisy_fidelity_standard_error,
                    record.confidence_interval_lower,
                    record.confidence_interval_upper,
                )
                if recorded_statistics != expected_statistics:
                    msg = "Evaluation trajectory sidecar does not reproduce the canonical row statistics."
                    raise Phase2ArtifactVerificationError(msg)

    def _referenced_relative_paths(self) -> frozenset[str]:
        """Return every managed artifact path referenced by canonical streams."""
        paths: set[str] = set()
        external_path = self._external_checkpoint_relative_path()
        if external_path is not None:
            paths.add(external_path)
        for artifact in self._stage_artifacts:
            result = artifact.stage_result
            paths.update({
                result.produced_checkpoint_path,
                cast("str", result.optimizer_trace_path),
                cast("str", result.diagnostic_sidecar_path),
            })
            paths.update(ref.path for ref in artifact.fixed_map_artifacts)
        paths.update(item.path for item in self._materializations)
        for evidence in self._evaluation_evidence:
            paths.update(ref.path for ref in evidence.evaluation_map_artifacts)
        for record in self._records:
            if isinstance(record, PipelineBenchmarkResult) and record.trajectory_sidecar_path is not None:
                paths.add(record.trajectory_sidecar_path)
        return frozenset(paths)

    def _cleanup_orphan_artifacts(self) -> None:
        """Remove uncommitted files without touching completed artifacts."""
        referenced = self._referenced_relative_paths()
        for directory_name in _MANAGED_DIRECTORIES:
            directory = self.output_directory / directory_name
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if not path.is_file() and not path.is_symlink():
                    continue
                relative = path.relative_to(self.output_directory).as_posix()
                if relative not in referenced:
                    path.unlink(missing_ok=True)
            for path in sorted(directory.rglob("*"), reverse=True):
                if path.is_dir():
                    with contextlib.suppress(OSError):
                        path.rmdir()

    @staticmethod
    def _serialize_row(row: object) -> str:
        """Serialize an object exposing a canonical ``to_json`` method."""
        serializer = getattr(row, "to_json", None)
        if not callable(serializer):
            msg = f"Stream row {type(row).__name__} does not expose to_json()."
            raise TypeError(msg)
        payload = serializer()
        if type(payload) is not str:
            msg = "to_json() must return text."
            raise TypeError(msg)
        return payload

    def _write_generic_stream(self, path: Path, rows: Sequence[object]) -> None:
        """Atomically rewrite one canonical JSONL stream."""
        payload = "".join(f"{self._serialize_row(row)}\n" for row in rows).encode()
        atomic_write_bytes(path, payload)

    def _write_stage_stream(self) -> None:
        """Atomically rewrite the canonical completed-stage ledger."""
        self._write_generic_stream(self.stage_result_stream_path, self._stage_artifacts)

    def _write_stage_failure_stream(self) -> None:
        """Atomically rewrite the append-only stage-failure ledger."""
        self._write_generic_stream(self.stage_failure_stream_path, self._stage_failures)

    def _write_materialization_stream(self) -> None:
        """Atomically rewrite the materialization ledger."""
        self._write_generic_stream(self.materialization_stream_path, self._materializations)

    def _write_materialization_attempt_stream(self) -> None:
        """Atomically rewrite the append-only materialization-attempt ledger."""
        self._write_generic_stream(self.materialization_attempt_stream_path, self._materialization_attempts)

    def _write_result_stream(self) -> None:
        """Atomically rewrite the canonical evaluation-result stream."""
        self._write_generic_stream(self.results_jsonl_path, self._records)

    def _write_evaluation_failure_stream(self) -> None:
        """Atomically rewrite append-only evaluation-failure history."""
        self._write_generic_stream(self.evaluation_failure_stream_path, self._evaluation_failures)

    def _write_evaluation_evidence_stream(self) -> None:
        """Atomically rewrite supplemental evaluation evidence."""
        self._write_generic_stream(self.evaluation_evidence_stream_path, self._evaluation_evidence)

    def _write_csv(self) -> None:
        """Atomically derive the union CSV from canonical evaluation rows."""
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=PIPELINE_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in self._records:
            writer.writerow(record.to_csv_row())
        atomic_write_bytes(self.results_csv_path, buffer.getvalue().encode())

    def _timing_summary(self) -> dict[str, float]:
        """Return the documented exact, non-overlapping wall-time convention."""
        stage_time = sum(artifact.stage_result.wall_time_seconds for artifact in self._stage_artifacts) + sum(
            failure.wall_time_seconds for failure in self._stage_failures
        )
        materialization_time = sum(item.wall_time_seconds for item in self._materialization_attempts)
        evaluation_time = sum(
            record.evaluation_wall_time_seconds
            for record in self._records
            if isinstance(record, PipelineBenchmarkResult)
        ) + sum(failure.wall_time_seconds for failure in self._evaluation_failures)
        return {
            "stage_execution_seconds": stage_time,
            "circuit_materialization_seconds": materialization_time,
            "row_evaluation_seconds": evaluation_time,
            "total_wall_time_seconds": stage_time + materialization_time + evaluation_time,
        }

    def _manifest_dict(self) -> dict[str, object]:
        """Build the complete derived store manifest."""
        result = self.pipeline_result
        successful_ids = sorted(
            record.evaluation_row_id for record in self._records if isinstance(record, PipelineBenchmarkResult)
        )
        failed_ids = sorted(
            record.evaluation_row_id for record in self._records if isinstance(record, PipelineBenchmarkFailure)
        )
        artifact_inventory = sorted(self._referenced_relative_paths())
        payload = {
            "manifest_format": PHASE2_ARTIFACT_MANIFEST_FORMAT,
            "pipeline_training_id": self.pipeline.training_id,
            "pipeline_configuration_checksum": self.pipeline.configuration_checksum,
            "completed_stage_count": len(self._stage_artifacts),
            "completed_pipeline_result_checksum": None if result is None else result.content_checksum,
            "active_runtime_fingerprint_checksum": self.runtime_fingerprint_checksum,
            "resume_override_checksums": [item.content_checksum for item in self._resume_overrides],
            "canonical_stage_stream": STAGE_RESULT_STREAM_NAME,
            "stage_failure_stream": STAGE_FAILURE_STREAM_NAME,
            "canonical_result_stream": RESULTS_JSONL_NAME,
            "evaluation_failure_stream": EVALUATION_FAILURE_STREAM_NAME,
            "derived_csv": RESULTS_CSV_NAME,
            "materialization_stream": MATERIALIZATION_STREAM_NAME,
            "materialization_attempt_stream": MATERIALIZATION_ATTEMPT_STREAM_NAME,
            "evaluation_evidence_stream": EVALUATION_EVIDENCE_STREAM_NAME,
            "completed_stage_artifact_checksums": [artifact.content_checksum for artifact in self._stage_artifacts],
            "stage_failure_ids": [failure.failure_id for failure in self._stage_failures],
            "stage_failure_checksums": [failure.content_checksum for failure in self._stage_failures],
            "materialization_checksums": [item.content_checksum for item in self._materializations],
            "materialization_attempt_checksums": [
                attempt.content_checksum for attempt in self._materialization_attempts
            ],
            "evaluation_record_index": list(self._record_index(self._records)),
            "evaluation_failure_attempt_checksums": [failure.content_checksum for failure in self._evaluation_failures],
            "evaluation_evidence_checksums": [item.content_checksum for item in self._evaluation_evidence],
            "successful_evaluation_row_ids": successful_ids,
            "failed_evaluation_row_ids": failed_ids,
            "record_count": len(self._records),
            "artifact_inventory": artifact_inventory,
            "canonical_stream_checksums": {
                name: _sha256((self.output_directory / name).read_bytes()) for name in _CANONICAL_LEDGER_NAMES
            },
            "timing_convention": (
                "total_wall_time_seconds = stage_execution_seconds + "
                "circuit_materialization_seconds + row_evaluation_seconds; atomic store I/O is excluded"
            ),
            "timing": self._timing_summary(),
        }
        return seal_mapping(payload)

    def _write_manifest(self) -> None:
        """Atomically rebuild the checksum-sealed derived manifest."""
        atomic_write_bytes(self.manifest_path, f"{canonical_json(self._manifest_dict())}\n".encode())

    def _write_derived_evaluation_views(self) -> None:
        """Rebuild CSV and manifest without changing canonical scientific rows."""
        try:
            self._write_csv()
            self._write_manifest()
        except Exception as error:
            msg = "Canonical evidence committed, but a derived CSV or manifest rebuild failed."
            raise Phase2DerivedArtifactError(msg) from error

    def _write_verified_artifact(self, relative_path: str, payload: bytes, checksum: str) -> None:
        """Publish exact bytes or verify an identical immutable existing artifact."""
        expected = require_checksum(checksum, "checksum")
        actual = _sha256(payload)
        if expected != actual:
            msg = f"Artifact payload checksum mismatch: expected {expected}, computed {actual}."
            raise Phase2ArtifactVerificationError(msg)
        path = self._resolve_managed_relative(relative_path)
        if path.exists():
            existing = self._read_verified_bytes(relative_path, expected, maximum_size=max(len(payload), 1))
            if existing != payload:
                msg = f"Existing immutable artifact {relative_path!r} differs from the supplied payload."
                raise Phase2ArtifactVerificationError(msg)
            return
        atomic_write_bytes(path, payload)

    @staticmethod
    def _checkpoint_provenance_checksum(
        *,
        pipeline_prefix_id: str,
        stage_id: str,
        stage_configuration_checksum: str,
        input_checkpoint_checksum: str | None,
        input_checkpoint_provenance_checksum: str | None,
        produced_checkpoint_checksum: str,
    ) -> str:
        """Derive the WP16 immutable checkpoint-lineage checksum."""
        return canonical_checksum({
            "pipeline_prefix_id": pipeline_prefix_id,
            "stage_id": stage_id,
            "stage_configuration_checksum": stage_configuration_checksum,
            "input_checkpoint_checksum": input_checkpoint_checksum,
            "input_checkpoint_provenance_checksum": input_checkpoint_provenance_checksum,
            "produced_checkpoint_checksum": produced_checkpoint_checksum,
        })

    def _validate_wp17_objective_binding(
        self,
        binding: NoisyKrotovObjectiveBinding | None,
        *,
        objective_checksum: str | None,
    ) -> None:
        """Require genuine WP17 evidence to use this pipeline's authorized objective."""
        if not isinstance(binding, NoisyKrotovObjectiveBinding):
            msg = "Genuine WP17 evidence requires a sealed objective binding."
            raise TypeError(msg)
        if objective_checksum is None or binding.objective_checksum != objective_checksum:
            msg = "WP17 objective binding does not reproduce the stage objective checksum."
            raise ValueError(msg)
        target_ref = self.pipeline.target_ref
        target_identity = binding.materialized_target_identity
        if self.pipeline.target_namespace != "phase2" or target_ref is None or target_identity is None:
            msg = "Published WP17 evidence requires an authorized materialized Phase II target."
            raise ValueError(msg)
        expected_target_identity = {
            "target_instance_id": self.pipeline.target_instance_id,
            "target_instance_spec_checksum": self.pipeline.target_instance_spec_checksum,
            "population_config_checksum": target_ref.population_config_checksum,
            "target_manifest_checksum": self.pipeline.target_population_manifest_checksum,
            "parameter_checksum": canonical_checksum(target_ref.target_spec.parameters),
            "family_id": self.pipeline.target_family_id,
            "stratum_id": self.pipeline.target_stratum_id,
            "qubit_count": self.pipeline.qubit_count,
        }
        if any(target_identity[name] != value for name, value in expected_target_identity.items()):
            msg = "WP17 materialized target identity does not match the configured pipeline target."
            raise ValueError(msg)
        expected_initial_checksum = noisy_krotov_computational_zero_state_checksum(self.pipeline.qubit_count)
        if (
            binding.initial_state_policy != "computational_zero_v1"
            or binding.initial_state_checksum != expected_initial_checksum
        ):
            msg = "Published WP17 evidence must use the computational-zero initial-state policy."
            raise ValueError(msg)

    def _stage_checkpoint(self, evidence: StageExecutionEvidence) -> StageParameterCheckpoint:
        """Build a generic or resumable safe checkpoint from stage evidence."""
        resume_state = None
        if evidence.objective_checksum is not None:
            assert evidence.circuit_binding_checksum is not None
            selection = None
            if evidence.selected_checkpoint_validation_fidelity is not None:
                selection = NoisyKrotovCheckpointSelection(
                    stage_configuration_checksum=evidence.stage.configuration_checksum,
                    circuit_binding_checksum=evidence.circuit_binding_checksum,
                    provider_checksum=evidence.provider_checksum,
                    objective_checksum=evidence.objective_checksum,
                    global_iteration=evidence.selected_global_iteration,
                    validation_fidelity=evidence.selected_checkpoint_validation_fidelity,
                    theta=evidence.selected_parameters,
                )
            resume_state = NoisyKrotovResumeState(
                stage_configuration_checksum=evidence.stage.configuration_checksum,
                circuit_binding_checksum=evidence.circuit_binding_checksum,
                provider_checksum=evidence.provider_checksum,
                objective_checksum=evidence.objective_checksum,
                completed_global_iteration=evidence.completed_global_iteration,
                final_parameter_checksum=evidence.final_parameter_checksum,
                checkpoint_selection=selection,
                cumulative_work=KrotovWorkLedger(
                    objective_evaluations=cast("int", evidence.normalized_work["objective_evaluations"]),
                    gradient_evaluations=cast("int", evidence.normalized_work["gradient_evaluations"]),
                    training_trajectories=cast("int", evidence.normalized_work["training_trajectories"]),
                    checkpoint_validation_trajectories=cast(
                        "int",
                        evidence.normalized_work["checkpoint_validation_trajectories"],
                    ),
                    test_trajectories=cast("int", evidence.normalized_work["test_trajectories"]),
                    trajectory_gate_applications=cast(
                        "int",
                        evidence.normalized_work["trajectory_gate_applications"],
                    ),
                ),
                cumulative_cross_trajectory_pairings=evidence.cumulative_cross_trajectory_pairings,
            )
        return StageParameterCheckpoint(
            pipeline_training_id=self.pipeline.training_id,
            pipeline_prefix_id=self.pipeline.prefix_id(evidence.stage.stage_index),
            stage_index=evidence.stage.stage_index,
            stage_id=evidence.stage.stage_id,
            stage_configuration_checksum=evidence.stage.configuration_checksum,
            selected_theta=evidence.selected_parameters,
            final_theta=evidence.final_parameters,
            selected_global_iteration=evidence.selected_global_iteration,
            completed_global_iteration=evidence.completed_global_iteration,
            circuit_binding_checksum=evidence.circuit_binding_checksum,
            provider_checksum=evidence.provider_checksum,
            objective_checksum=evidence.objective_checksum,
            stage_execution_checksum=cast("str | None", evidence.training_summary.get("adapter_execution_checksum")),
            resume_state=resume_state,
        )

    def _persist_fixed_maps(
        self,
        pipeline_prefix_id: str,
        ensembles: Sequence[KrotovFixedMapEnsemble],
    ) -> tuple[FixedMapArtifactRef, ...]:
        """Persist exact fixed maps and return their role-bound references."""
        references: list[FixedMapArtifactRef] = []
        for ensemble in ensembles:
            relative = f"{FIXED_MAP_DIRECTORY}/{pipeline_prefix_id}/{ensemble.role}-{ensemble.ensemble_id}.json"
            payload = ensemble.to_json().encode()
            _require_artifact_size(payload, _MAX_FIXED_MAP_SIZE, "Fixed-map artifact")
            file_checksum = _sha256(payload)
            self._write_verified_artifact(relative, payload, file_checksum)
            references.append(
                FixedMapArtifactRef(
                    role=ensemble.role,
                    ensemble_id=ensemble.ensemble_id,
                    content_checksum=ensemble.content_checksum,
                    path=relative,
                    file_checksum=file_checksum,
                )
            )
        return tuple(references)

    def publish_stage(
        self,
        evidence: StageExecutionEvidence,
        *,
        wall_time_seconds: float,
        peak_memory_bytes: int,
    ) -> PersistedStageArtifact:
        """Commit one stage only if this handle still owns the retained baseline."""
        with self._mutation_guard():
            return self._publish_stage_unlocked(
                evidence,
                wall_time_seconds=wall_time_seconds,
                peak_memory_bytes=peak_memory_bytes,
            )

    def _publish_stage_unlocked(
        self,
        evidence: StageExecutionEvidence,
        *,
        wall_time_seconds: float,
        peak_memory_bytes: int,
    ) -> PersistedStageArtifact:
        """Atomically commit one completed stage and all of its immutable evidence.

        The stage row is written only after every referenced artifact is safely
        present. If interruption occurs earlier, reopening removes only those
        unreferenced files and reruns the unfinished stage.

        Returns:
            The canonical persisted-stage artifact.

        Raises:
            Phase2DuplicateRecordError: If the stage is not the next prefix.
            ValueError: If the execution does not match the configured stage.
        """
        if not isinstance(evidence, StageExecutionEvidence):
            msg = "evidence must be StageExecutionEvidence."
            raise TypeError(msg)
        next_index = len(self._stage_artifacts)
        if next_index >= len(self.pipeline.stages):
            msg = "Every configured pipeline stage is already complete."
            raise Phase2DuplicateRecordError(msg)
        stage = self.pipeline.stages[next_index]
        if evidence.stage != stage:
            msg = "Stage evidence is not the next configured pipeline stage."
            raise ValueError(msg)
        if next_index == 0:
            if stage.input_checkpoint_checksum is None and evidence.source_parameter_checksum is not None:
                msg = "An initialized first stage cannot claim predecessor parameters."
                raise ValueError(msg)
            if stage.input_checkpoint_checksum is not None:
                external_checkpoint = self.load_external_checkpoint()
                if evidence.source_parameter_checksum != external_checkpoint.selected_parameter_checksum:
                    msg = "Stage evidence does not consume the verified external checkpoint parameters."
                    raise ValueError(msg)
        else:
            predecessor_checkpoint = self.load_stage_checkpoint(next_index - 1)
            if evidence.source_parameter_checksum != predecessor_checkpoint.selected_parameter_checksum:
                msg = "Stage evidence does not consume the verified predecessor's selected parameters."
                raise ValueError(msg)
        if evidence.training_summary.get("adapter_execution_checksum") is not None:
            self._validate_wp17_objective_binding(
                evidence.objective_binding,
                objective_checksum=evidence.objective_checksum,
            )
        elapsed = require_float(float(wall_time_seconds), "wall_time_seconds", minimum=0.0)
        peak = require_int(peak_memory_bytes, "peak_memory_bytes")
        prefix_id = self.pipeline.prefix_id(next_index)

        checkpoint = self._stage_checkpoint(evidence)
        checkpoint_payload = checkpoint.to_bytes()
        _require_artifact_size(checkpoint_payload, _MAX_CHECKPOINT_SIZE, "Stage checkpoint")
        checkpoint_checksum = checkpoint.content_checksum
        checkpoint_relative = f"{CHECKPOINT_DIRECTORY}/{prefix_id}.npz"

        trace_document = seal_mapping({
            "schema_version": PHASE2_TRACE_SCHEMA_VERSION,
            "pipeline_training_id": self.pipeline.training_id,
            "pipeline_prefix_id": prefix_id,
            "stage_configuration_checksum": stage.configuration_checksum,
            "trace": [thaw_json_mapping(row) for row in evidence.trace],
            "optimizer_state": (
                None if evidence.optimizer_state is None else thaw_json_mapping(evidence.optimizer_state)
            ),
        })
        trace_payload = canonical_json(trace_document).encode()
        _require_artifact_size(trace_payload, _MAX_TRACE_SIZE, "Optimizer trace")
        trace_checksum = _sha256(trace_payload)
        trace_relative = f"{TRACE_DIRECTORY}/{prefix_id}.json"

        training_coordinates = _map_schedule_coordinates(
            stage.sampling_policy,
            stage.crn_refresh_interval,
            stage.iteration_budget,
        )
        validation_coordinates: tuple[tuple[int, int, int], ...] = ()
        if stage.checkpoint_validation.enabled:
            validation_calls = 1 + math.ceil(stage.iteration_budget / cast("int", stage.checkpoint_validation.cadence))
            validation_coordinates = _map_schedule_coordinates(
                stage.checkpoint_validation.sampling_policy,
                stage.checkpoint_validation.ensemble_refresh_interval,
                validation_calls,
            )
        for maps, role, seed, trajectory_count, coordinates, provider_checksum in (
            (
                evidence.training_ensembles,
                "training_trajectory",
                stage.training_seed,
                stage.trajectory_count,
                training_coordinates,
                evidence.provider_checksum,
            ),
            (
                evidence.checkpoint_validation_ensembles,
                "checkpoint_validation",
                stage.checkpoint_validation.seed,
                stage.checkpoint_validation.trajectory_count,
                validation_coordinates,
                evidence.checkpoint_validation_provider_checksum,
            ),
        ):
            actual_coordinates = tuple(
                (item.ensemble_index, item.refresh_index, item.global_iteration_start) for item in maps
            )
            if actual_coordinates != coordinates:
                msg = f"{role} fixed-map artifacts do not match the configured sampling schedule."
                raise ValueError(msg)
            if any(
                item.role != role
                or item.resolved_seed != seed
                or item.trajectory_count != trajectory_count
                or item.stage_index != stage.stage_index
                or item.stage_id != stage.stage_id
                or item.stage_configuration_checksum != stage.configuration_checksum
                or item.circuit_checksum != evidence.circuit_binding_checksum
                or item.provider_checksum != provider_checksum
                for item in maps
            ):
                msg = f"{role} fixed-map artifacts do not bind the exact configured stage context."
                raise ValueError(msg)

        map_refs = self._persist_fixed_maps(
            prefix_id,
            (*evidence.training_ensembles, *evidence.checkpoint_validation_ensembles),
        )
        training_refs = tuple(ref for ref in map_refs if ref.role == "training_trajectory")
        validation_refs = tuple(ref for ref in map_refs if ref.role == "checkpoint_validation")
        metadata_document = seal_mapping({
            "schema_version": PHASE2_STAGE_METADATA_SCHEMA_VERSION,
            "pipeline_training_id": self.pipeline.training_id,
            "pipeline_prefix_id": prefix_id,
            "stage_configuration_checksum": stage.configuration_checksum,
            "circuit_binding_checksum": evidence.circuit_binding_checksum,
            "provider_checksum": evidence.provider_checksum,
            "checkpoint_validation_provider_checksum": evidence.checkpoint_validation_provider_checksum,
            "objective_checksum": evidence.objective_checksum,
            "objective_binding": (None if evidence.objective_binding is None else evidence.objective_binding.to_dict()),
            "source_parameter_checksum": evidence.source_parameter_checksum,
            "initial_parameter_checksum": evidence.initial_parameter_checksum,
            "final_parameter_checksum": evidence.final_parameter_checksum,
            "selected_parameter_checksum": evidence.selected_parameter_checksum,
            "selected_global_iteration": evidence.selected_global_iteration,
            "completed_global_iteration": evidence.completed_global_iteration,
            "selected_checkpoint_validation_fidelity": evidence.selected_checkpoint_validation_fidelity,
            "circuit_topology": thaw_json_mapping(evidence.circuit_topology),
            "circuit_statistics": thaw_json_mapping(evidence.circuit_statistics),
            "training_map_artifacts": [ref.to_dict() for ref in training_refs],
            "checkpoint_validation_map_artifacts": [ref.to_dict() for ref in validation_refs],
            "cumulative_cross_trajectory_pairings": evidence.cumulative_cross_trajectory_pairings,
            "runtime_fingerprint_checksum": self.runtime_fingerprint_checksum,
        })
        metadata_payload = canonical_json(metadata_document).encode()
        _require_artifact_size(metadata_payload, _MAX_STAGE_METADATA_SIZE, "Stage metadata")
        metadata_checksum = _sha256(metadata_payload)
        metadata_relative = f"{STAGE_METADATA_DIRECTORY}/{prefix_id}.json"

        previous = None if not self._stage_artifacts else self._stage_artifacts[-1].stage_result
        input_checksum = stage.input_checkpoint_checksum if previous is None else previous.produced_checkpoint_checksum
        input_provenance = (
            stage.input_checkpoint_provenance_checksum if previous is None else previous.checkpoint_provenance_checksum
        )
        result = TrainingStageResult(
            pipeline_training_id=self.pipeline.training_id,
            pipeline_prefix_id=prefix_id,
            stage_index=stage.stage_index,
            stage_id=stage.stage_id,
            stage_configuration_checksum=stage.configuration_checksum,
            input_checkpoint_checksum=input_checksum,
            input_checkpoint_provenance_checksum=input_provenance,
            produced_checkpoint_path=checkpoint_relative,
            produced_checkpoint_checksum=checkpoint_checksum,
            checkpoint_provenance_checksum=self._checkpoint_provenance_checksum(
                pipeline_prefix_id=prefix_id,
                stage_id=stage.stage_id,
                stage_configuration_checksum=stage.configuration_checksum,
                input_checkpoint_checksum=input_checksum,
                input_checkpoint_provenance_checksum=input_provenance,
                produced_checkpoint_checksum=checkpoint_checksum,
            ),
            output_topology_id=stage.output_topology_id,
            output_parameter_count=stage.output_parameter_count,
            training_summary=evidence.training_summary,
            checkpoint_validation_summary=evidence.checkpoint_validation_summary,
            training_ensemble_checksums=(
                tuple(ref.content_checksum for ref in training_refs)
                if stage.sampling_policy in {"crn_fixed", "crn_refresh"}
                else ()
            ),
            checkpoint_validation_ensemble_checksum=(
                None
                if not stage.checkpoint_validation.enabled
                or stage.checkpoint_validation.sampling_policy not in {"crn_fixed", "crn_refresh"}
                else canonical_checksum({
                    "role": "checkpoint_validation",
                    "ensemble_checksums": [ref.content_checksum for ref in validation_refs],
                })
            ),
            optimizer_trace_path=trace_relative,
            optimizer_trace_checksum=trace_checksum,
            diagnostic_sidecar_path=metadata_relative,
            diagnostic_sidecar_checksum=metadata_checksum,
            wall_time_seconds=elapsed,
            peak_memory_bytes=peak,
            normalized_work=evidence.normalized_work,
        )
        artifact = PersistedStageArtifact(
            stage_result=result,
            runtime_fingerprint_checksum=self.runtime_fingerprint_checksum,
            checkpoint_file_checksum=checkpoint_checksum,
            trace_file_checksum=trace_checksum,
            metadata_file_checksum=metadata_checksum,
            fixed_map_artifacts=map_refs,
        )
        self._write_verified_artifact(checkpoint_relative, checkpoint_payload, checkpoint_checksum)
        self._write_verified_artifact(trace_relative, trace_payload, trace_checksum)
        self._write_verified_artifact(metadata_relative, metadata_payload, metadata_checksum)
        self._stage_artifacts.append(artifact)
        try:
            self._validate_loaded_state()
            self._verify_referenced_artifacts()
            self._write_stage_stream()
        except BaseException:
            self._stage_artifacts.pop()
            raise
        self._write_manifest()
        return artifact

    def _external_checkpoint_relative_path(self) -> str | None:
        """Return the deterministic managed path for the sealed external input."""
        stage = self.pipeline.stages[0]
        if stage.input_checkpoint_checksum is None:
            return None
        return f"{CHECKPOINT_DIRECTORY}/external-{stage.input_checkpoint_checksum[7:]}.npz"

    def _decode_external_checkpoint(self, payload: bytes) -> StageParameterCheckpoint:
        """Decode external bytes and verify their complete producer reference."""
        stage = self.pipeline.stages[0]
        if stage.input_checkpoint_checksum is None:
            msg = "The first configured stage does not consume an external checkpoint."
            raise Phase2ArtifactVerificationError(msg)
        assert stage.input_checkpoint_pipeline_prefix is not None
        try:
            checkpoint = StageParameterCheckpoint.from_bytes(
                payload,
                expected_checksum=stage.input_checkpoint_checksum,
                expected_pipeline_prefix_id=stage.input_checkpoint_pipeline_prefix,
            )
        except (TypeError, ValueError) as error:
            msg = f"External checkpoint is invalid: {error}."
            raise Phase2ArtifactVerificationError(msg) from error
        if checkpoint.parameter_count != stage.input_parameter_count:
            msg = "External checkpoint parameter count does not match the consuming stage."
            raise Phase2ArtifactVerificationError(msg)
        assert stage.input_checkpoint_provenance_checksum is not None
        expected_reference = canonical_checksum({
            "schema_version": EXTERNAL_CHECKPOINT_REF_SCHEMA_VERSION,
            "producer_pipeline_prefix_id": checkpoint.pipeline_prefix_id,
            "producer_stage_index": checkpoint.stage_index,
            "producer_stage_id": checkpoint.stage_id,
            "producer_stage_configuration_checksum": checkpoint.stage_configuration_checksum,
            "produced_checkpoint_checksum": stage.input_checkpoint_checksum,
            "checkpoint_provenance_checksum": stage.input_checkpoint_provenance_checksum,
            "output_topology_id": stage.input_topology_id,
            "output_parameter_count": stage.input_parameter_count,
        })
        if expected_reference != stage.input_checkpoint_ref_checksum:
            msg = "External checkpoint bytes do not reproduce the sealed producer reference."
            raise Phase2ArtifactVerificationError(msg)
        return checkpoint

    def _read_external_checkpoint_source(self) -> bytes:
        """Preflight the source path and return exact verified checkpoint bytes."""
        stage = self.pipeline.stages[0]
        if stage.input_checkpoint_path is None or stage.input_checkpoint_checksum is None:
            msg = "The first configured stage does not contain a complete external checkpoint reference."
            raise Phase2ArtifactVerificationError(msg)
        input_root = Path.cwd().resolve()
        candidate = input_root / Path(stage.input_checkpoint_path)
        resolved_source = candidate.resolve()
        if not resolved_source.is_relative_to(input_root):
            msg = "External checkpoint path must remain inside its launch-time input root."
            raise Phase2ArtifactVerificationError(msg)
        relative_parts = Path(stage.input_checkpoint_path).parts
        current = input_root
        for part in relative_parts:
            current /= part
            if current.is_symlink():
                msg = "External checkpoint path must not contain symbolic links."
                raise Phase2ArtifactVerificationError(msg)
        try:
            with resolved_source.open("rb") as stream:
                payload = stream.read(_MAX_CHECKPOINT_SIZE + 1)
        except OSError as error:
            msg = f"External checkpoint {stage.input_checkpoint_path!r} could not be read: {error}."
            raise Phase2ArtifactVerificationError(msg) from error
        if len(payload) > _MAX_CHECKPOINT_SIZE:
            msg = "External checkpoint exceeds the configured verification limit."
            raise Phase2ArtifactVerificationError(msg)
        self._decode_external_checkpoint(payload)
        return payload

    def _ingest_external_checkpoint(self, payload: bytes) -> None:
        """Atomically seal preflight-verified external input bytes in the store."""
        stage = self.pipeline.stages[0]
        if stage.input_checkpoint_checksum is None:
            msg = "The first configured stage does not consume an external checkpoint."
            raise Phase2ArtifactVerificationError(msg)
        self._decode_external_checkpoint(payload)
        relative = cast("str", self._external_checkpoint_relative_path())
        self._write_verified_artifact(relative, payload, stage.input_checkpoint_checksum)

    def _verify_sealed_external_checkpoint(self) -> StageParameterCheckpoint:
        """Verify the store-local external input without consulting its source path."""
        stage = self.pipeline.stages[0]
        relative = self._external_checkpoint_relative_path()
        if relative is None or stage.input_checkpoint_checksum is None:
            msg = "The first configured stage does not consume an external checkpoint."
            raise Phase2ArtifactVerificationError(msg)
        payload = self._read_verified_bytes(
            relative,
            stage.input_checkpoint_checksum,
            maximum_size=_MAX_CHECKPOINT_SIZE,
        )
        return self._decode_external_checkpoint(payload)

    def load_external_checkpoint(self) -> StageParameterCheckpoint:
        """Load the verified store-local copy of the external first-stage input."""
        return self._verify_sealed_external_checkpoint()

    def load_stage_checkpoint(self, stage_index: int) -> StageParameterCheckpoint:
        """Checksum-verify and decode one completed-stage checkpoint."""
        index = require_int(stage_index, "stage_index")
        if index >= len(self._stage_artifacts):
            msg = f"Stage {index} is not part of the completed verified prefix."
            raise Phase2ArtifactVerificationError(msg)
        artifact = self._stage_artifacts[index]
        result = artifact.stage_result
        payload = self._read_verified_bytes(
            result.produced_checkpoint_path,
            result.produced_checkpoint_checksum,
            maximum_size=_MAX_CHECKPOINT_SIZE,
        )
        try:
            return StageParameterCheckpoint.from_bytes(
                payload,
                expected_checksum=result.produced_checkpoint_checksum,
                expected_pipeline_training_id=self.pipeline.training_id,
                expected_pipeline_prefix_id=self.pipeline.prefix_id(index),
                expected_stage_configuration_checksum=result.stage_configuration_checksum,
            )
        except (TypeError, ValueError) as error:
            msg = f"Could not verify stage {index} checkpoint: {error}."
            raise Phase2ArtifactVerificationError(msg) from error

    def load_final_parameters(self) -> NDArray[np.float64]:
        """Return selected parameters from the complete final stage."""
        result = self.pipeline_result
        if result is None:
            msg = "Final parameters are unavailable until every pipeline stage is complete."
            raise Phase2ArtifactVerificationError(msg)
        return self.load_stage_checkpoint(len(self._stage_artifacts) - 1).selected_theta

    def write_stage_failure(
        self,
        stage: TrainingStageConfig,
        failure: BaseException | NoisyKrotovStageFailure,
        *,
        wall_time_seconds: float,
        retryable: bool | None = None,
    ) -> StageFailureArtifact:
        """Record one stage failure only against this handle's retained baseline."""
        with self._mutation_guard():
            return self._write_stage_failure_unlocked(
                stage,
                failure,
                wall_time_seconds=wall_time_seconds,
                retryable=retryable,
            )

    def _write_stage_failure_unlocked(
        self,
        stage: TrainingStageConfig,
        failure: BaseException | NoisyKrotovStageFailure,
        *,
        wall_time_seconds: float,
        retryable: bool | None = None,
    ) -> StageFailureArtifact:
        """Append and preserve one structured failure for the next stage attempt."""
        if not isinstance(stage, TrainingStageConfig):
            msg = "stage must be a TrainingStageConfig."
            raise TypeError(msg)
        next_index = len(self._stage_artifacts)
        if next_index >= len(self.pipeline.stages) or stage != self.pipeline.stages[next_index]:
            msg = "Failures may be recorded only for the next unfinished configured stage."
            raise ValueError(msg)
        if not isinstance(failure, (BaseException, NoisyKrotovStageFailure)):
            msg = "failure must be an exception or NoisyKrotovStageFailure."
            raise TypeError(msg)
        prefix_id = self.pipeline.prefix_id(stage.stage_index)
        attempt = 1 + sum(item.pipeline_prefix_id == prefix_id for item in self._stage_failures)
        if isinstance(failure, NoisyKrotovStageFailure):
            if (
                failure.stage_index != stage.stage_index
                or failure.stage_id != stage.stage_id
                or failure.stage_configuration_checksum != stage.configuration_checksum
            ):
                msg = "Noisy Krotov failure does not identify the supplied stage."
                raise ValueError(msg)
            phase = failure.phase
            exception_type = failure.exception_type
            message = failure.message
            traceback_text = failure.traceback_text
            partial_work = failure.partial_work
            resolved_retryable = failure.retryable if retryable is None else retryable
        else:
            phase = "stage_execution"
            exception_type = type(failure).__name__
            message = str(failure) or type(failure).__name__
            traceback_text = "".join(traceback_module.format_exception(failure))
            partial_work = _empty_work()
            resolved_retryable = False if retryable is None else retryable
        artifact = StageFailureArtifact(
            pipeline_training_id=self.pipeline.training_id,
            pipeline_configuration_checksum=self.pipeline.configuration_checksum,
            pipeline_prefix_id=prefix_id,
            stage_index=stage.stage_index,
            stage_id=stage.stage_id,
            stage_configuration_checksum=stage.configuration_checksum,
            phase=phase,
            exception_type=exception_type,
            message=message,
            traceback=traceback_text,
            retryable=resolved_retryable,
            attempt=attempt,
            partial_work=partial_work,
            completed_stage_artifact_checksums=tuple(item.content_checksum for item in self._stage_artifacts),
            wall_time_seconds=float(wall_time_seconds),
            runtime_fingerprint_checksum=self.runtime_fingerprint_checksum,
        )
        self._stage_failures.append(artifact)
        try:
            self._validate_loaded_state()
            self._write_stage_failure_stream()
        except BaseException:
            self._stage_failures.pop()
            raise
        self._write_manifest()
        return artifact

    @staticmethod
    def _evaluation_map_role(data_role: str) -> str:
        """Map one data role to its reserved fixed-map random-stream role."""
        roles = {
            "confirmatory": "confirmatory_test",
            "screening_selection": "screening_selection",
            "checkpoint_validation": "checkpoint_validation",
            "development": "pilot_evaluation",
            "secondary_benchmark": "pilot_evaluation",
        }
        try:
            return roles[data_role]
        except KeyError as error:
            msg = f"Unsupported Phase II evaluation data role {data_role!r}."
            raise ValueError(msg) from error

    def _next_materialization_attempt(self, materialized_circuit_id: str) -> int:
        """Return the next contiguous attempt number for one circuit identity."""
        return 1 + sum(
            item.materialized_circuit_id == materialized_circuit_id for item in self._materialization_attempts
        )

    def record_materialization_failure(
        self,
        *,
        config: PipelineEvaluationConfig,
        exception: BaseException,
        phase: Literal["materialization", "serialization"],
        wall_time_seconds: float,
        peak_memory_bytes: int = 0,
    ) -> MaterializationAttemptArtifact:
        """Record one materialization failure against the retained baseline."""
        with self._mutation_guard():
            return self._record_materialization_failure_unlocked(
                config=config,
                exception=exception,
                phase=phase,
                wall_time_seconds=wall_time_seconds,
                peak_memory_bytes=peak_memory_bytes,
            )

    def _record_materialization_failure_unlocked(
        self,
        *,
        config: PipelineEvaluationConfig,
        exception: BaseException,
        phase: Literal["materialization", "serialization"],
        wall_time_seconds: float,
        peak_memory_bytes: int = 0,
    ) -> MaterializationAttemptArtifact:
        """Persist one shared failed materialization attempt exactly once."""
        result = self.pipeline_result
        if result is None:
            msg = "Materialization failures require a complete training pipeline artifact."
            raise Phase2ArtifactVerificationError(msg)
        if not isinstance(config, PipelineEvaluationConfig):
            msg = "config must be a PipelineEvaluationConfig."
            raise TypeError(msg)
        config.validate_against_pipeline(result)
        if not isinstance(exception, BaseException):
            msg = "exception must be a BaseException."
            raise TypeError(msg)
        artifact = MaterializationAttemptArtifact(
            materialized_circuit_id=config.materialized_circuit_id,
            pipeline_training_id=result.training_id,
            pipeline_result_checksum=result.content_checksum,
            attempt=self._next_materialization_attempt(config.materialized_circuit_id),
            status="failure",
            phase=phase,
            payload_checksum=None,
            exception_type=type(exception).__name__,
            message=str(exception) or type(exception).__name__,
            wall_time_seconds=wall_time_seconds,
            peak_memory_bytes=peak_memory_bytes,
            runtime_fingerprint_checksum=self.runtime_fingerprint_checksum,
        )
        self._materialization_attempts.append(artifact)
        try:
            self._validate_loaded_state()
            self._write_materialization_attempt_stream()
        except BaseException:
            self._materialization_attempts.pop()
            raise
        try:
            self._write_manifest()
        except Exception as error:
            msg = "Materialization failure committed, but the derived manifest rebuild failed."
            raise Phase2DerivedArtifactError(msg) from error
        return artifact

    def publish_materialized_circuit(
        self,
        *,
        config: PipelineEvaluationConfig,
        payload: bytes,
        wall_time_seconds: float,
        peak_memory_bytes: int,
    ) -> MaterializedCircuitArtifact:
        """Publish one materialization only against the retained baseline."""
        with self._mutation_guard():
            return self._publish_materialized_circuit_unlocked(
                config=config,
                payload=payload,
                wall_time_seconds=wall_time_seconds,
                peak_memory_bytes=peak_memory_bytes,
            )

    def _publish_materialized_circuit_unlocked(
        self,
        *,
        config: PipelineEvaluationConfig,
        payload: bytes,
        wall_time_seconds: float,
        peak_memory_bytes: int,
    ) -> MaterializedCircuitArtifact:
        """Publish or verify the one shared materialized circuit for evaluations."""
        result = self.pipeline_result
        if result is None:
            msg = "A circuit cannot be materialized before the training pipeline is complete."
            raise Phase2ArtifactVerificationError(msg)
        if not isinstance(config, PipelineEvaluationConfig):
            msg = "config must be a PipelineEvaluationConfig."
            raise TypeError(msg)
        config.validate_against_pipeline(result)
        validated_payload = validate_materialized_circuit_payload(payload)
        payload_checksum = _sha256(validated_payload)
        if payload_checksum != config.materialized_circuit_checksum:
            msg = "Materialized circuit bytes do not match the planned circuit checksum."
            raise ValueError(msg)
        elapsed = require_float(float(wall_time_seconds), "wall_time_seconds", minimum=0.0)
        peak = require_int(peak_memory_bytes, "peak_memory_bytes")
        existing = next(
            (item for item in self._materializations if item.materialized_circuit_id == config.materialized_circuit_id),
            None,
        )
        if existing is not None:
            if existing.payload_checksum != payload_checksum:
                msg = "An existing materialized-circuit identity refers to different bytes."
                raise Phase2ArtifactVerificationError(msg)
            self._read_verified_bytes(existing.path, existing.payload_checksum, maximum_size=_MAX_CIRCUIT_SIZE)
            artifact = existing
        else:
            relative = f"{CIRCUIT_DIRECTORY}/{config.materialized_circuit_id}.bin"
            artifact = MaterializedCircuitArtifact(
                materialized_circuit_id=config.materialized_circuit_id,
                pipeline_training_id=result.training_id,
                pipeline_result_checksum=result.content_checksum,
                final_checkpoint_checksum=result.final_checkpoint_checksum,
                materialization_policy_checksum=config.final_materialization_policy_checksum,
                path=relative,
                payload_checksum=payload_checksum,
                wall_time_seconds=elapsed,
                peak_memory_bytes=peak,
                runtime_fingerprint_checksum=self.runtime_fingerprint_checksum,
            )
            self._write_verified_artifact(relative, validated_payload, payload_checksum)
        attempt = MaterializationAttemptArtifact(
            materialized_circuit_id=config.materialized_circuit_id,
            pipeline_training_id=result.training_id,
            pipeline_result_checksum=result.content_checksum,
            attempt=self._next_materialization_attempt(config.materialized_circuit_id),
            status="success",
            phase="materialization",
            payload_checksum=payload_checksum,
            exception_type=None,
            message=None,
            wall_time_seconds=elapsed,
            peak_memory_bytes=peak,
            runtime_fingerprint_checksum=self.runtime_fingerprint_checksum,
        )
        previous_materializations = list(self._materializations)
        previous_attempts = list(self._materialization_attempts)
        if existing is None:
            self._materializations.append(artifact)
        self._materialization_attempts.append(attempt)
        try:
            self._validate_loaded_state()
            if existing is None:
                self._write_materialization_stream()
            self._write_materialization_attempt_stream()
        except BaseException:
            self._materializations = previous_materializations
            self._materialization_attempts = previous_attempts
            with contextlib.suppress(Exception):
                self._write_materialization_attempt_stream()
            with contextlib.suppress(Exception):
                self._write_materialization_stream()
            raise
        try:
            self._write_manifest()
        except Exception as error:
            msg = "Materialization committed, but the derived manifest rebuild failed."
            raise Phase2DerivedArtifactError(msg) from error
        return artifact

    def _require_materialization(
        self,
        config: PipelineEvaluationConfig,
        materialization: MaterializedCircuitArtifact,
    ) -> None:
        """Verify a materialization against one planned row and this store."""
        if not isinstance(materialization, MaterializedCircuitArtifact):
            msg = "materialization must be a MaterializedCircuitArtifact."
            raise TypeError(msg)
        if materialization not in self._materializations:
            msg = "Materialization is not part of this artifact store."
            raise ValueError(msg)
        if (
            materialization.materialized_circuit_id != config.materialized_circuit_id
            or materialization.payload_checksum != config.materialized_circuit_checksum
            or materialization.materialization_policy_checksum != config.final_materialization_policy_checksum
        ):
            msg = "Materialization does not match the planned evaluation circuit."
            raise ValueError(msg)

    @staticmethod
    def _trajectory_statistics(
        config: PipelineEvaluationConfig,
        fidelities: Sequence[float],
    ) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        """Compute the exact fixed-sample mean, dispersion, and configured CI."""
        noisy = config.test_noise_id != NOISELESS_NOISE_ID
        if not noisy:
            if fidelities:
                msg = "Noiseless evaluation cannot contain trajectory fidelities."
                raise ValueError(msg)
            return None, None, None, None, None
        try:
            values = tuple(float(value) for value in fidelities)
        except (TypeError, ValueError) as error:
            msg = "trajectory_fidelities must contain finite real values."
            raise TypeError(msg) from error
        if len(values) != config.trajectory_budget:
            msg = "Trajectory-fidelity count must equal the fixed evaluation budget."
            raise ValueError(msg)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            msg = "Trajectory fidelities must be finite and lie in [0, 1]."
            raise ValueError(msg)
        mean = statistics.fmean(values)
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        standard_error = deviation / math.sqrt(len(values))
        lower: float | None = None
        upper: float | None = None
        if config.evaluation_policy == "confidence_interval":
            method = config.confidence_interval_method
            if method not in {"normal", "normal_clipped"}:
                msg = f"Unsupported Phase II confidence interval method {method!r}."
                raise ValueError(msg)
            assert config.confidence_level is not None
            critical = statistics.NormalDist().inv_cdf((1.0 + config.confidence_level) / 2.0)
            lower = mean - critical * standard_error
            upper = mean + critical * standard_error
            if method == "normal_clipped":
                lower = max(0.0, lower)
                upper = min(1.0, upper)
        return mean, deviation, standard_error, lower, upper

    @staticmethod
    def _evaluation_work(
        config: PipelineEvaluationConfig,
        normalized_work: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Validate final-evaluation work independently of evaluator callbacks."""
        work = _sum_work((normalized_work,))
        expected_test_trajectories = config.trajectory_budget if config.test_noise_id != NOISELESS_NOISE_ID else 0
        if work["test_trajectories"] != expected_test_trajectories:
            msg = "Normalized test-trajectory work must equal the configured evaluation budget."
            raise ValueError(msg)
        if work["training_trajectories"] != 0 or work["checkpoint_validation_trajectories"] != 0:
            msg = "Final-evaluation work cannot include training or checkpoint-validation trajectories."
            raise ValueError(msg)
        return work

    def _validate_evaluation_maps(
        self,
        config: PipelineEvaluationConfig,
        ensembles: Sequence[KrotovFixedMapEnsemble],
        evaluation_provider_checksum: str | None,
    ) -> tuple[KrotovFixedMapEnsemble, ...]:
        """Validate final-test role, seed, budget, circuit, and provider bindings."""
        maps = tuple(ensembles)
        noisy = config.test_noise_id != NOISELESS_NOISE_ID
        if not noisy:
            if maps or evaluation_provider_checksum is not None:
                msg = "Noiseless evaluation cannot consume fixed noisy maps or a noise provider."
                raise ValueError(msg)
            return ()
        provider_checksum = require_checksum(evaluation_provider_checksum, "evaluation_provider_checksum")
        if not maps or not all(isinstance(item, KrotovFixedMapEnsemble) for item in maps):
            msg = "Noisy evaluation requires fresh KrotovFixedMapEnsemble evidence."
            raise TypeError(msg)
        expected_role = self._evaluation_map_role(config.data_role)
        if any(item.role != expected_role for item in maps):
            msg = f"Evaluation maps must use the reserved {expected_role!r} role."
            raise ValueError(msg)
        if any(item.resolved_seed != config.evaluation_seed for item in maps):
            msg = "Evaluation fixed maps do not use the planned evaluation seed."
            raise ValueError(msg)
        if any(
            item.stage_configuration_checksum != config.configuration_checksum
            or item.circuit_checksum != config.materialized_circuit_checksum
            or item.provider_checksum != provider_checksum
            for item in maps
        ):
            msg = "Evaluation fixed maps do not bind the planned row, circuit, and noise provider."
            raise ValueError(msg)
        if sum(item.trajectory_count for item in maps) != config.trajectory_budget:
            msg = "Evaluation fixed-map trajectories must equal the fixed evaluation budget."
            raise ValueError(msg)
        if len({item.ensemble_id for item in maps}) != len(maps) or len({
            item.content_checksum for item in maps
        }) != len(maps):
            msg = "Evaluation fixed-map ensembles must be mutually distinct."
            raise ValueError(msg)
        return maps

    def _require_fresh_evaluation_maps(
        self,
        config: PipelineEvaluationConfig,
        ensembles: Sequence[KrotovFixedMapEnsemble],
        evaluation_provider_checksum: str | None,
    ) -> tuple[KrotovFixedMapEnsemble, ...]:
        """Validate final-test maps and reject reuse of any stored ensemble."""
        maps = self._validate_evaluation_maps(config, ensembles, evaluation_provider_checksum)
        existing_refs = [ref for artifact in self._stage_artifacts for ref in artifact.fixed_map_artifacts]
        existing_refs.extend(ref for evidence in self._evaluation_evidence for ref in evidence.evaluation_map_artifacts)
        existing_ids = {ref.ensemble_id for ref in existing_refs}
        existing_checksums = {ref.content_checksum for ref in existing_refs}
        if any(item.ensemble_id in existing_ids or item.content_checksum in existing_checksums for item in maps):
            msg = "Evaluation attempted to reuse a training, validation, screening, or test map ensemble."
            raise ValueError(msg)
        return maps

    def _replace_current_record(self, record: PipelineBenchmarkRecord) -> None:
        """Upsert one current row while preserving failure history separately."""
        for index, existing in enumerate(self._records):
            if existing.evaluation_row_id != record.evaluation_row_id:
                continue
            if isinstance(existing, PipelineBenchmarkResult):
                msg = f"Evaluation row {record.evaluation_row_id!r} is already successful."
                raise Phase2DuplicateRecordError(msg)
            self._records[index] = record
            return
        self._records.append(record)

    def write_evaluation_success(
        self,
        *,
        config: PipelineEvaluationConfig,
        materialization: MaterializedCircuitArtifact,
        test_noiseless_fidelity: float,
        trajectory_fidelities: Sequence[float],
        sampled_nonidentity_events: int,
        normalized_work: Mapping[str, object],
        evaluation_wall_time_seconds: float,
        peak_memory_bytes: int,
        evaluation_provider_checksum: str | None,
        evaluation_ensembles: Sequence[KrotovFixedMapEnsemble] = (),
    ) -> PipelineBenchmarkResult:
        """Publish one successful row only against the retained baseline."""
        with self._mutation_guard():
            return self._write_evaluation_success_unlocked(
                config=config,
                materialization=materialization,
                test_noiseless_fidelity=test_noiseless_fidelity,
                trajectory_fidelities=trajectory_fidelities,
                sampled_nonidentity_events=sampled_nonidentity_events,
                normalized_work=normalized_work,
                evaluation_wall_time_seconds=evaluation_wall_time_seconds,
                peak_memory_bytes=peak_memory_bytes,
                evaluation_provider_checksum=evaluation_provider_checksum,
                evaluation_ensembles=evaluation_ensembles,
            )

    def _write_evaluation_success_unlocked(
        self,
        *,
        config: PipelineEvaluationConfig,
        materialization: MaterializedCircuitArtifact,
        test_noiseless_fidelity: float,
        trajectory_fidelities: Sequence[float],
        sampled_nonidentity_events: int,
        normalized_work: Mapping[str, object],
        evaluation_wall_time_seconds: float,
        peak_memory_bytes: int,
        evaluation_provider_checksum: str | None,
        evaluation_ensembles: Sequence[KrotovFixedMapEnsemble] = (),
    ) -> PipelineBenchmarkResult:
        """Publish a fully linked final-test success and optional sidecars."""
        result = self.pipeline_result
        if result is None:
            msg = "Evaluation is unavailable until every pipeline stage is complete."
            raise Phase2ArtifactVerificationError(msg)
        if not isinstance(config, PipelineEvaluationConfig):
            msg = "config must be a PipelineEvaluationConfig."
            raise TypeError(msg)
        config.validate_against_pipeline(result)
        if any(
            isinstance(item, PipelineBenchmarkResult) and item.evaluation_row_id == config.evaluation_row_id
            for item in self._records
        ):
            msg = f"Evaluation row {config.evaluation_row_id!r} is already successful."
            raise Phase2DuplicateRecordError(msg)
        if any(item.evaluation_row_id == config.evaluation_row_id for item in self._evaluation_evidence):
            msg = f"Evaluation evidence {config.evaluation_row_id!r} is already committed."
            raise Phase2DuplicateRecordError(msg)
        self._require_materialization(config, materialization)
        maps = self._require_fresh_evaluation_maps(
            config,
            evaluation_ensembles,
            evaluation_provider_checksum,
        )
        expected_events = sum(item.nonidentity_event_count for item in maps)
        if sampled_nonidentity_events != expected_events:
            msg = "sampled_nonidentity_events must equal the persisted final-evaluation maps."
            raise ValueError(msg)
        noisy_mean, deviation, standard_error, lower, upper = self._trajectory_statistics(
            config,
            trajectory_fidelities,
        )
        evaluation_work = self._evaluation_work(config, normalized_work)
        map_refs = self._persist_fixed_maps(config.evaluation_row_id, maps)
        sidecar_relative: str | None = None
        sidecar_checksum: str | None = None
        if config.sidecar_storage_policy == "trajectory_fidelities":
            sidecar_relative = f"{TRAJECTORY_DIRECTORY}/{config.evaluation_row_id}.npz"
            sidecar_payload = create_phase2_trajectory_sidecar(
                evaluation_row_id=config.evaluation_row_id,
                pipeline_training_id=config.pipeline_training_id,
                map_role=self._evaluation_map_role(config.data_role),
                map_partitions=tuple(
                    {
                        "ensemble_id": item.ensemble_id,
                        "content_checksum": item.content_checksum,
                        "trajectory_count": item.trajectory_count,
                    }
                    for item in maps
                ),
                fidelities=trajectory_fidelities,
            )
            sidecar_checksum = _sha256(sidecar_payload)
            _require_artifact_size(
                sidecar_payload,
                max(4096, config.trajectory_budget * 16 + 16384),
                "Trajectory sidecar",
            )
            self._write_verified_artifact(sidecar_relative, sidecar_payload, sidecar_checksum)
        record = PipelineBenchmarkResult(
            config=config,
            materialized_circuit_path=materialization.path,
            test_noiseless_fidelity=float(test_noiseless_fidelity),
            test_noisy_fidelity=noisy_mean,
            noisy_fidelity_standard_deviation=deviation,
            noisy_fidelity_standard_error=standard_error,
            confidence_interval_lower=lower,
            confidence_interval_upper=upper,
            sampled_nonidentity_events=sampled_nonidentity_events,
            trajectory_sidecar_path=sidecar_relative,
            trajectory_sidecar_checksum=sidecar_checksum,
            evaluation_wall_time_seconds=float(evaluation_wall_time_seconds),
            peak_memory_bytes=peak_memory_bytes,
            normalized_work=evaluation_work,
            runtime_fingerprint_checksum=self.runtime_fingerprint_checksum,
        )
        evidence = EvaluationEvidenceArtifact(
            evaluation_row_id=record.evaluation_row_id,
            record_checksum=record.content_checksum,
            pipeline_result_checksum=result.content_checksum,
            materialization_checksum=materialization.content_checksum,
            evaluation_provider_checksum=evaluation_provider_checksum,
            evaluation_map_artifacts=map_refs,
        )
        previous_records = list(self._records)
        previous_evidence = list(self._evaluation_evidence)
        self._evaluation_evidence.append(evidence)
        try:
            self._replace_current_record(record)
            self._validate_loaded_state()
            self._write_evaluation_evidence_stream()
            self._write_result_stream()
        except BaseException:
            self._records = previous_records
            self._evaluation_evidence = previous_evidence
            with contextlib.suppress(Exception):
                self._write_result_stream()
            with contextlib.suppress(Exception):
                self._write_evaluation_evidence_stream()
            raise
        self._write_derived_evaluation_views()
        return record

    def write_evaluation_failure(
        self,
        *,
        config: PipelineEvaluationConfig,
        exception: BaseException,
        phase: Literal["pipeline_loading", "materialization", "evaluation", "serialization"],
        wall_time_seconds: float,
        materialization: MaterializedCircuitArtifact | None = None,
        retryable: bool = False,
    ) -> PipelineBenchmarkFailure:
        """Publish one failed row only against the retained baseline."""
        with self._mutation_guard():
            return self._write_evaluation_failure_unlocked(
                config=config,
                exception=exception,
                phase=phase,
                wall_time_seconds=wall_time_seconds,
                materialization=materialization,
                retryable=retryable,
            )

    def _write_evaluation_failure_unlocked(
        self,
        *,
        config: PipelineEvaluationConfig,
        exception: BaseException,
        phase: Literal["pipeline_loading", "materialization", "evaluation", "serialization"],
        wall_time_seconds: float,
        materialization: MaterializedCircuitArtifact | None = None,
        retryable: bool = False,
    ) -> PipelineBenchmarkFailure:
        """Append one linked evaluation failure and update its current row."""
        result = self.pipeline_result
        if result is None:
            msg = "Evaluation failures require a complete training pipeline artifact."
            raise Phase2ArtifactVerificationError(msg)
        if not isinstance(config, PipelineEvaluationConfig):
            msg = "config must be a PipelineEvaluationConfig."
            raise TypeError(msg)
        config.validate_against_pipeline(result)
        if not isinstance(exception, BaseException):
            msg = "exception must be a BaseException."
            raise TypeError(msg)
        if materialization is not None:
            self._require_materialization(config, materialization)
        if any(
            isinstance(item, PipelineBenchmarkResult) and item.evaluation_row_id == config.evaluation_row_id
            for item in self._records
        ):
            msg = f"Evaluation row {config.evaluation_row_id!r} is already successful."
            raise Phase2DuplicateRecordError(msg)
        attempt = 1 + sum(item.evaluation_row_id == config.evaluation_row_id for item in self._evaluation_failures)
        failure = PipelineBenchmarkFailure.from_exception(
            config=config,
            failure_phase=phase,
            exception=exception,
            runtime_fingerprint_checksum=self.runtime_fingerprint_checksum,
            traceback="".join(traceback_module.format_exception(exception)),
            retryable=retryable,
            attempt=attempt,
            materialized_circuit_path=None if materialization is None else materialization.path,
            materialized_circuit_checksum=(None if materialization is None else materialization.payload_checksum),
            wall_time_seconds=float(wall_time_seconds),
        )
        previous_records = list(self._records)
        previous_failures = list(self._evaluation_failures)
        self._evaluation_failures.append(failure)
        try:
            self._replace_current_record(failure)
            self._validate_loaded_state()
            self._write_evaluation_failure_stream()
            self._write_result_stream()
        except BaseException:
            self._records = previous_records
            self._evaluation_failures = previous_failures
            with contextlib.suppress(Exception):
                self._write_result_stream()
            with contextlib.suppress(Exception):
                self._write_evaluation_failure_stream()
            raise
        self._write_derived_evaluation_views()
        return failure
