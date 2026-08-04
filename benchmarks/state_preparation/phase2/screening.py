# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""WP22 screening orchestration and final-confirmation seal construction.

This module deliberately operates on typed, checksum-sealed artifacts.  It
does not accept training scores, selected-target summaries, or confirmatory
target manifests.  The only confirmatory-target input accepted before the
final seal is the public checksum/count commitment created by WP16.
"""

from __future__ import annotations

import base64
import binascii
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from mqt.yaqs.optimization import KrotovFixedMapEnsemble

from .artifact_codecs import artifact_checksum, read_phase2_trajectory_sidecar
from .canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .execution_context import TrainingExecutionContext
from .pilot import (
    PILOT_CALCULATION_SOURCE_CHECKSUM,
    PILOT_PRIMARY_JOB_COUNT,
    PILOT_PRIMARY_TRAJECTORY_COUNT,
    PILOT_SECONDARY_TRAJECTORY_COUNT,
    PilotNuisanceSummary,
    ProductionPilotCustody,
    build_cluster_aware_paired_difference_v1,
    reestimate_cluster_aware_paired_difference_v1,
)
from .pipeline import (
    PipelineBenchmarkFailure,
    PipelineBenchmarkRecord,
    PipelineBenchmarkResult,
    TrainingPipelineResult,
    TrainingPipelineTemplate,
    pipeline_benchmark_record_from_dict,
)
from .protocol import (
    AnalysisSourceManifest,
    FinalComparatorRef,
    FinalConfigurationExecutionManifest,
    FinalConfigurationExecutionRef,
    FinalConfirmationSeal,
    FinalResourceCalibrationManifest,
    InitialPreregistration,
    PrimaryContrastBinding,
    PromotionDecision,
    PromotionObservation,
    SampleSizeDesign,
    ScreeningCandidateRef,
    ScreeningCell,
    ScreeningEvidence,
    ScreeningManifest,
    authorize_confirmation,
    select_promoted_candidate,
)
from .result_custody import (
    ProductionResultCustody,
    TrajectoryFidelityEvidence,
    reopen_terminal_production_attempt,
    validate_production_job_custody,
)
from .screening_design import (
    ADAPT_STYLE_PUBLICATION_METHOD_ID,
    IMPACT_PRUNING_PUBLICATION_METHOD_ID,
    WP22_CANDIDATE_CONFIGURATION_SCHEMA_VERSION,
    WP22_OPERATOR_GROWTH_TEMPLATE_SCHEMA_VERSION,
    WP22_PUBLICATION_PRUNING_MAPPING_VERSION,
    OperatorGrowthScreeningTemplate,
    WP22CandidateConfiguration,
    build_screening_manifest,
)
from .source_lock import ExecutionSourceManifest, verify_analysis_source_bridge
from .targets import TargetPopulationCommitment, TargetPopulationManifest
from .training_orchestration import TrainingJob, TrainingJobOutcome, load_training_job_outcome_history
from .validation import require_checksum, require_float, require_int, require_mapping, require_slug
from .wp20_resources import CircuitResourceMetrics, WP20WorkLedger

if TYPE_CHECKING:
    from .artifacts import EvaluationEvidenceArtifact, MaterializedCircuitArtifact
    from .operator_growth_pipeline import OperatorGrowthPipelineArtifact

WP22_SCREENING_OUTCOME_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_screening_outcome.v1"
WP18_SCREENING_SOURCE_ARTIFACT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp18_screening_source_artifact.v1"
WP22_SCREENING_SOURCE_RECORD_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_screening_source_record.v2"
WP22_PRODUCTION_SCREENING_SOURCE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_production_screening_source.v1"
PRODUCTION_RESOURCE_PROJECTION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.production_resource_projection.v1"
PILOT_NORMALIZED_COMPUTE_CALIBRATION_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.pilot_normalized_compute_calibration.v1"
)
PRODUCTION_RESOURCE_CALIBRATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.production_resource_calibration.v1"
_PRODUCTION_RESOURCE_PROJECTION_KEYS = frozenset({
    "schema_version",
    "job_checksum",
    "result_reference_checksum",
    "resource_document_checksum",
    "execution_source_manifest_checksum",
    "method_id",
    "candidate_configuration_checksum",
    "data_role",
    "family_id",
    "stratum_id",
    "qubit_count",
    "status",
    "normalized_work",
    "structural_prefix_checksums",
    "circuit_binding_checksum",
    "compiled_resources_checksum",
    "native_two_qubit_gates_per_chain_edge",
    "content_checksum",
})
_PRODUCTION_RESOURCE_CALIBRATION_KEYS = frozenset({
    "schema_version",
    "preregistration_checksum",
    "execution_source_manifest_checksum",
    "pilot_plan_checksum",
    "pilot_custody_checksum",
    "pilot_calibration_checksum",
    "screening_plan_checksum",
    "screening_manifest_checksum",
    "screening_custody_checksum",
    "calculation_rule_id",
    "normalized_compute_cap",
    "pilot_q6_resources",
    "screening_resources",
    "content_checksum",
})
_PILOT_NORMALIZED_COMPUTE_CALIBRATION_KEYS = frozenset({
    "schema_version",
    "preregistration_checksum",
    "execution_source_manifest_checksum",
    "pilot_plan_checksum",
    "pilot_custody_checksum",
    "calculation_rule_id",
    "normalized_compute_cap",
    "pilot_q6_resources",
    "content_checksum",
})
_OUTCOME_KEYS = frozenset({
    "schema_version",
    "candidate_configuration_checksum",
    "cell_id",
    "data_role",
    "evaluation_seed",
    "result_schema_version",
    "result_record_checksum",
    "evaluation_evidence_checksum",
    "materialization_checksum",
    "status",
    "noisy_fidelity",
    "resource_value",
    "normalized_work",
    "failure_code",
    "circuit_resources",
    "work_ledger",
    "protocol_violations",
    "content_checksum",
})
_WP18_SOURCE_ARTIFACT_KEYS = frozenset({
    "schema_version",
    "candidate",
    "cell",
    "template",
    "pipeline_result",
    "record",
    "circuit_resources",
    "work_ledger",
    "evaluation_evidence",
    "materialization",
    "preregistration",
    "evaluation_maps",
    "trajectory_sidecar_base64",
    "protocol_violations",
    "content_checksum",
})
_SOURCE_RECORD_KEYS = frozenset({"schema_version", "source_kind", "source_artifact", "content_checksum"})


def _ordered_strings(value: object, name: str) -> tuple[str, ...]:
    """Return a duplicate-free ordered tuple of slug strings.

    Raises:
        TypeError: If ``value`` is not a sequence.
        ValueError: If a value is invalid or duplicated.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence."
        raise TypeError(msg)
    result = tuple(require_slug(item, f"{name} item") for item in value)
    if len(result) != len(set(result)):
        msg = f"{name} must not contain duplicates."
        raise ValueError(msg)
    return result


def screening_trajectory_context_checksum(
    *,
    candidate_configuration_checksum: str,
    cell_id: str,
    result_schema_version: str,
    result_record_checksum: str,
) -> str:
    """Bind raw screening trajectories to their exact result and cell.

    Returns:
        The canonical evaluation-context checksum.
    """
    return canonical_checksum({
        "schema_version": "yaqs.state_preparation.phase2.screening_trajectory_context.v1",
        "candidate_configuration_checksum": require_checksum(
            candidate_configuration_checksum,
            "candidate_configuration_checksum",
        ),
        "cell_id": require_slug(cell_id, "cell_id"),
        "result_schema_version": require_slug(result_schema_version, "result_schema_version"),
        "result_record_checksum": require_checksum(result_record_checksum, "result_record_checksum"),
    })


@dataclass(frozen=True, slots=True)
class VerifiedScreeningOutcome:
    """One artifact-linked held-out screening outcome used for promotion."""

    candidate_configuration_checksum: str
    cell_id: str
    data_role: str
    evaluation_seed: int
    result_schema_version: str
    result_record_checksum: str
    evaluation_evidence_checksum: str | None
    materialization_checksum: str | None
    status: Literal["success", "failure"]
    noisy_fidelity: float | None
    resource_value: float | None
    normalized_work: float
    failure_code: str | None
    circuit_resources: CircuitResourceMetrics | None
    work_ledger: WP20WorkLedger
    protocol_violations: tuple[str, ...] = ()
    schema_version: str = field(default=WP22_SCREENING_OUTCOME_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate outcome status, source links, and derived work/resources.

        Raises:
            TypeError: If a successful row lacks typed resource evidence.
            ValueError: If identities, links, roles, or derived values disagree.
        """
        object.__setattr__(
            self,
            "candidate_configuration_checksum",
            require_checksum(self.candidate_configuration_checksum, "candidate_configuration_checksum"),
        )
        object.__setattr__(self, "cell_id", require_slug(self.cell_id, "cell_id"))
        if self.data_role != "screening_selection":
            msg = "WP22 promotion outcomes must use only the outer screening_selection role."
            raise ValueError(msg)
        object.__setattr__(self, "evaluation_seed", require_int(self.evaluation_seed, "evaluation_seed", minimum=0))
        object.__setattr__(
            self,
            "result_schema_version",
            require_slug(self.result_schema_version, "result_schema_version"),
        )
        object.__setattr__(
            self,
            "result_record_checksum",
            require_checksum(self.result_record_checksum, "result_record_checksum"),
        )
        for name in ("evaluation_evidence_checksum", "materialization_checksum"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_checksum(value, name))
        if self.status not in {"success", "failure"}:
            msg = "status must be 'success' or 'failure'."
            raise ValueError(msg)
        if not isinstance(self.work_ledger, WP20WorkLedger):
            msg = "work_ledger must be a WP20WorkLedger."
            raise TypeError(msg)
        expected_work = self.work_ledger.normalized_compute()
        normalized = require_float(self.normalized_work, "normalized_work", minimum=0.0)
        if float(normalized).hex() != float(expected_work).hex():
            msg = "normalized_work must be mechanically derived from the WP20 work ledger."
            raise ValueError(msg)
        object.__setattr__(self, "normalized_work", normalized)
        if self.status == "success":
            if not isinstance(self.circuit_resources, CircuitResourceMetrics):
                msg = "Successful screening outcomes require compiler-derived CircuitResourceMetrics."
                raise TypeError(msg)
            fidelity = require_float(self.noisy_fidelity, "noisy_fidelity", minimum=0.0, maximum=1.0)
            counts = self.circuit_resources.native_two_qubit_gates_per_chain_edge
            resource = float(max(counts, default=0))
            supplied_resource = float(require_float(self.resource_value, "resource_value", minimum=0.0))
            if supplied_resource.hex() != resource.hex():
                msg = "resource_value must equal the largest compiler-derived native per-edge count."
                raise ValueError(msg)
            if self.failure_code is not None:
                msg = "Successful screening outcomes cannot carry a failure code."
                raise ValueError(msg)
            if self.evaluation_evidence_checksum is None or self.materialization_checksum is None:
                msg = "Successful pipeline outcomes require evaluation and materialization evidence links."
                raise ValueError(msg)
            object.__setattr__(self, "noisy_fidelity", fidelity)
            object.__setattr__(self, "resource_value", resource)
        else:
            if self.noisy_fidelity is not None or self.resource_value is not None or self.circuit_resources is not None:
                msg = "Failed screening outcomes cannot fabricate fidelity or circuit resources."
                raise ValueError(msg)
            if self.failure_code is None:
                msg = "Failed screening outcomes require a failure code."
                raise ValueError(msg)
            object.__setattr__(self, "failure_code", require_slug(self.failure_code, "failure_code"))
        object.__setattr__(
            self,
            "protocol_violations",
            _ordered_strings(self.protocol_violations, "protocol_violations"),
        )

    @classmethod
    def from_pipeline_record(
        cls,
        *,
        candidate: WP22CandidateConfiguration,
        cell: ScreeningCell,
        template: TrainingPipelineTemplate,
        pipeline_result: TrainingPipelineResult,
        record: PipelineBenchmarkRecord,
        work_ledger: WP20WorkLedger,
        circuit_resources: CircuitResourceMetrics | None,
        evaluation_evidence: EvaluationEvidenceArtifact | None,
        materialization: MaterializedCircuitArtifact | None,
        preregistration: InitialPreregistration,
        protocol_violations: Sequence[str] = (),
    ) -> VerifiedScreeningOutcome:
        """Create one promotion input from cross-linked WP18/WP20 artifacts.

        Returns:
            A verified outer-screening outcome.

        Raises:
            TypeError: If required artifacts have the wrong types or are absent.
            ValueError: If artifact links, roles, seeds, noise, or work disagree.
        """
        if not isinstance(candidate, WP22CandidateConfiguration):
            msg = "candidate must be a WP22CandidateConfiguration."
            raise TypeError(msg)
        if not isinstance(cell, ScreeningCell):
            msg = "cell must be a ScreeningCell."
            raise TypeError(msg)
        if not isinstance(template, TrainingPipelineTemplate):
            msg = "template must be a TrainingPipelineTemplate."
            raise TypeError(msg)
        if not isinstance(pipeline_result, TrainingPipelineResult):
            msg = "pipeline_result must be a TrainingPipelineResult."
            raise TypeError(msg)
        if not isinstance(record, (PipelineBenchmarkResult, PipelineBenchmarkFailure)):
            msg = "record must be a PipelineBenchmarkResult or PipelineBenchmarkFailure."
            raise TypeError(msg)
        if not isinstance(preregistration, InitialPreregistration):
            msg = "preregistration must be an InitialPreregistration."
            raise TypeError(msg)
        config = record.config
        pipeline = pipeline_result.config
        if (
            candidate.implementation_kind != "phase2_pipeline"
            or candidate.implementation_method_id != template.method_id
            or candidate.implementation_checksum != template.configuration_checksum
            or pipeline.template != template
            or pipeline.method_id != candidate.implementation_method_id
            or pipeline.target_instance_id != cell.target_instance_id
            or pipeline.target_family_id != cell.family_id
            or pipeline.target_stratum_id != cell.stratum_id
            or pipeline.qubit_count != cell.qubit_count
            or pipeline.optimization_block_id != cell.cell_id
            or pipeline.optimization_seed != cell.optimization_seed
            or pipeline.data_role != "screening_selection"
            or config.pipeline_configuration_checksum != pipeline.configuration_checksum
            or config.pipeline_result_checksum != pipeline_result.content_checksum
        ):
            msg = "Candidate, template, trained pipeline, and screening cell do not form one identity chain."
            raise ValueError(msg)
        config.validate_against_pipeline(pipeline_result)
        noise = preregistration.primary_noise_condition
        if (
            config.data_role != "screening_selection"
            or config.evaluation_seed != cell.screening_seed
            or config.test_noise_id != noise["noise_id"]
            or config.noise_definition_version != noise["definition_version"]
            or canonical_json(config.noise_strength_scale) != canonical_json(noise["strength_scale"])
            or canonical_json(config.tjm_dt) != canonical_json(noise["tjm_dt"])
        ):
            msg = "Pipeline result is not the cell's fresh outer primary-noise screening evaluation."
            raise ValueError(msg)
        if isinstance(record, PipelineBenchmarkResult):
            if evaluation_evidence is None or materialization is None or circuit_resources is None:
                msg = "Successful pipeline screening requires evidence, materialization, and resources."
                raise TypeError(msg)
            if (
                evaluation_evidence.evaluation_row_id != record.evaluation_row_id
                or evaluation_evidence.record_checksum != record.content_checksum
                or evaluation_evidence.materialization_checksum != materialization.content_checksum
                or materialization.materialized_circuit_id != config.materialized_circuit_id
                or materialization.payload_checksum != config.materialized_circuit_checksum
                or record.materialized_circuit_path != materialization.path
            ):
                msg = "WP18 result, evaluation evidence, and materialization links do not agree."
                raise ValueError(msg)
            if not evaluation_evidence.evaluation_map_artifacts:
                msg = "Noisy screening success requires persisted fresh outer map artifacts."
                raise ValueError(msg)
            if any(ref.role != "screening_selection" for ref in evaluation_evidence.evaluation_map_artifacts):
                msg = "Screening evidence contains a non-screening trajectory-map role."
                raise ValueError(msg)
            if dict(record.normalized_work) != work_ledger.phase2_projection():
                msg = "WP18 normalized work does not match the detailed WP20 ledger."
                raise ValueError(msg)
            counts = circuit_resources.native_two_qubit_gates_per_chain_edge
            return cls(
                candidate_configuration_checksum=candidate.content_checksum,
                cell_id=cell.cell_id,
                data_role="screening_selection",
                evaluation_seed=cell.screening_seed,
                result_schema_version=record.schema_version,
                result_record_checksum=record.content_checksum,
                evaluation_evidence_checksum=evaluation_evidence.content_checksum,
                materialization_checksum=materialization.content_checksum,
                status="success",
                noisy_fidelity=record.test_noisy_fidelity,
                resource_value=float(max(counts, default=0)),
                normalized_work=work_ledger.normalized_compute(),
                failure_code=None,
                circuit_resources=circuit_resources,
                work_ledger=work_ledger,
                protocol_violations=tuple(protocol_violations),
            )
        if evaluation_evidence is not None:
            msg = "Failed evaluation rows cannot claim committed successful evaluation evidence."
            raise ValueError(msg)
        if materialization is not None and (
            record.materialized_circuit_checksum != materialization.payload_checksum
            or record.materialized_circuit_path != materialization.path
        ):
            msg = "Failed row materialization link does not match the supplied artifact."
            raise ValueError(msg)
        return cls(
            candidate_configuration_checksum=candidate.content_checksum,
            cell_id=cell.cell_id,
            data_role="screening_selection",
            evaluation_seed=cell.screening_seed,
            result_schema_version=record.schema_version,
            result_record_checksum=record.content_checksum,
            evaluation_evidence_checksum=None,
            materialization_checksum=None if materialization is None else materialization.content_checksum,
            status="failure",
            noisy_fidelity=None,
            resource_value=None,
            normalized_work=work_ledger.normalized_compute(),
            failure_code=record.failure_phase,
            circuit_resources=None,
            work_ledger=work_ledger,
            protocol_violations=tuple(protocol_violations),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered outcome fields."""
        return {
            "schema_version": self.schema_version,
            "candidate_configuration_checksum": self.candidate_configuration_checksum,
            "cell_id": self.cell_id,
            "data_role": self.data_role,
            "evaluation_seed": self.evaluation_seed,
            "result_schema_version": self.result_schema_version,
            "result_record_checksum": self.result_record_checksum,
            "evaluation_evidence_checksum": self.evaluation_evidence_checksum,
            "materialization_checksum": self.materialization_checksum,
            "status": self.status,
            "noisy_fidelity": self.noisy_fidelity,
            "resource_value": self.resource_value,
            "normalized_work": self.normalized_work,
            "failure_code": self.failure_code,
            "circuit_resources": None if self.circuit_resources is None else self.circuit_resources.to_dict(),
            "work_ledger": self.work_ledger.to_dict(),
            "protocol_violations": list(self.protocol_violations),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of source links and derived promotion values."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    def promotion_observation(self) -> PromotionObservation:
        """Project the verified artifact outcome onto the WP15 raw-evidence row.

        Returns:
            The frozen-protocol promotion observation.
        """
        return PromotionObservation(
            configuration_checksum=self.candidate_configuration_checksum,
            cell_id=self.cell_id,
            result_schema_version=self.result_schema_version,
            result_record_checksum=self.result_record_checksum,
            status=self.status,
            noisy_fidelity=self.noisy_fidelity,
            resource_value=self.resource_value,
            normalized_work=self.normalized_work,
            failure_code=self.failure_code,
            protocol_violations=self.protocol_violations,
        )

    @classmethod
    def from_dict(cls, data: object) -> VerifiedScreeningOutcome:
        """Decode and checksum-verify a persisted screening outcome.

        Returns:
            The verified screening outcome.

        Raises:
            ValueError: If the schema or checksum is invalid.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_OUTCOME_KEYS, name="WP22 screening outcome")
        if mapping["schema_version"] != WP22_SCREENING_OUTCOME_SCHEMA_VERSION:
            msg = "WP22 screening outcome uses an unsupported schema version."
            raise ValueError(msg)
        resources = mapping["circuit_resources"]
        outcome = cls(
            candidate_configuration_checksum=cast("str", mapping["candidate_configuration_checksum"]),
            cell_id=cast("str", mapping["cell_id"]),
            data_role=cast("str", mapping["data_role"]),
            evaluation_seed=cast("int", mapping["evaluation_seed"]),
            result_schema_version=cast("str", mapping["result_schema_version"]),
            result_record_checksum=cast("str", mapping["result_record_checksum"]),
            evaluation_evidence_checksum=cast("str | None", mapping["evaluation_evidence_checksum"]),
            materialization_checksum=cast("str | None", mapping["materialization_checksum"]),
            status=cast("Literal['success', 'failure']", mapping["status"]),
            noisy_fidelity=cast("float | None", mapping["noisy_fidelity"]),
            resource_value=cast("float | None", mapping["resource_value"]),
            normalized_work=cast("float", mapping["normalized_work"]),
            failure_code=cast("str | None", mapping["failure_code"]),
            circuit_resources=(None if resources is None else CircuitResourceMetrics.from_dict(resources)),
            work_ledger=WP20WorkLedger.from_dict(mapping["work_ledger"]),
            protocol_violations=cast("tuple[str, ...]", mapping["protocol_violations"]),
        )
        if mapping["content_checksum"] != outcome.content_checksum:
            msg = "WP22 screening outcome checksum changed during normalization."
            raise ValueError(msg)
        return outcome

    @classmethod
    def from_json(cls, payload: str) -> VerifiedScreeningOutcome:
        """Decode canonical checksum-sealed JSON.

        Returns:
            The verified screening outcome.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class _LegacyScreeningSourceRecord:
    """Authoritative screening input with an embedded raw trajectory source.

    ``VerifiedScreeningOutcome`` remains the persisted promotion projection,
    but only this source record may enter evidence construction.  Successful
    scalar fidelities are recomputed from ``trajectory_evidence`` every time.
    """

    candidate_configuration_checksum: str
    cell_id: str
    data_role: Literal["screening_selection"]
    evaluation_seed: int
    result_schema_version: str
    result_record_checksum: str
    evaluation_evidence_checksum: str | None
    materialization_checksum: str | None
    status: Literal["success", "failure"]
    trajectory_evidence: TrajectoryFidelityEvidence | None
    circuit_resources: CircuitResourceMetrics | None
    work_ledger: WP20WorkLedger
    failure_code: str | None
    protocol_violations: tuple[str, ...] = ()
    schema_version: str = field(default=WP22_SCREENING_SOURCE_RECORD_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the exact source identity and status-dependent evidence.

        Raises:
            TypeError: If a typed work, resource, or trajectory artifact is absent.
            ValueError: If an identity, role, status, or source link is invalid.
        """
        object.__setattr__(
            self,
            "candidate_configuration_checksum",
            require_checksum(self.candidate_configuration_checksum, "candidate_configuration_checksum"),
        )
        object.__setattr__(self, "cell_id", require_slug(self.cell_id, "cell_id"))
        if self.data_role != "screening_selection":
            msg = "Screening source records must use the screening_selection role."
            raise ValueError(msg)
        object.__setattr__(self, "evaluation_seed", require_int(self.evaluation_seed, "evaluation_seed", minimum=0))
        object.__setattr__(
            self,
            "result_schema_version",
            require_slug(self.result_schema_version, "result_schema_version"),
        )
        object.__setattr__(
            self,
            "result_record_checksum",
            require_checksum(self.result_record_checksum, "result_record_checksum"),
        )
        for name in ("evaluation_evidence_checksum", "materialization_checksum"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_checksum(value, name))
        if self.status not in {"success", "failure"}:
            msg = "status must be 'success' or 'failure'."
            raise ValueError(msg)
        if not isinstance(self.work_ledger, WP20WorkLedger):
            msg = "work_ledger must be a WP20WorkLedger."
            raise TypeError(msg)
        if self.status == "success":
            if not isinstance(self.trajectory_evidence, TrajectoryFidelityEvidence):
                msg = "Successful screening source records require raw trajectory evidence."
                raise TypeError(msg)
            if not isinstance(self.circuit_resources, CircuitResourceMetrics):
                msg = "Successful screening source records require compiler-derived resources."
                raise TypeError(msg)
            expected_context = screening_trajectory_context_checksum(
                candidate_configuration_checksum=self.candidate_configuration_checksum,
                cell_id=self.cell_id,
                result_schema_version=self.result_schema_version,
                result_record_checksum=self.result_record_checksum,
            )
            if (
                self.trajectory_evidence.evaluation_context_checksum != expected_context
                or self.trajectory_evidence.data_role != self.data_role
                or self.trajectory_evidence.evaluation_seed != self.evaluation_seed
            ):
                msg = "Raw screening trajectories do not identify this exact result, role, and seed."
                raise ValueError(msg)
            if self.evaluation_evidence_checksum is None or self.materialization_checksum is None:
                msg = "Successful screening source records require evaluation and materialization identities."
                raise ValueError(msg)
            if self.failure_code is not None:
                msg = "Successful screening source records cannot carry a failure code."
                raise ValueError(msg)
        else:
            if self.trajectory_evidence is not None or self.circuit_resources is not None:
                msg = "Failed screening source records cannot carry trajectory or resource evidence."
                raise ValueError(msg)
            if self.evaluation_evidence_checksum is not None:
                msg = "Failed screening source records cannot claim successful evaluation evidence."
                raise ValueError(msg)
            if self.failure_code is None:
                msg = "Failed screening source records require a failure code."
                raise ValueError(msg)
            object.__setattr__(self, "failure_code", require_slug(self.failure_code, "failure_code"))
        object.__setattr__(
            self,
            "protocol_violations",
            _ordered_strings(self.protocol_violations, "protocol_violations"),
        )

    @classmethod
    def from_pipeline_record(
        cls,
        *,
        trajectory_evidence: TrajectoryFidelityEvidence | None,
        candidate: WP22CandidateConfiguration,
        cell: ScreeningCell,
        template: TrainingPipelineTemplate,
        pipeline_result: TrainingPipelineResult,
        record: PipelineBenchmarkRecord,
        work_ledger: WP20WorkLedger,
        circuit_resources: CircuitResourceMetrics | None,
        evaluation_evidence: EvaluationEvidenceArtifact | None,
        materialization: MaterializedCircuitArtifact | None,
        preregistration: InitialPreregistration,
        protocol_violations: Sequence[str] = (),
    ) -> _LegacyScreeningSourceRecord:
        """Authenticate a WP18 result and its raw fixed-trajectory evidence.

        Returns:
            The authoritative screening source record.

        Raises:
            TypeError: If a mandatory source artifact has the wrong type.
            ValueError: If source identities or the raw aggregate disagree.
        """
        outcome = VerifiedScreeningOutcome.from_pipeline_record(
            candidate=candidate,
            cell=cell,
            template=template,
            pipeline_result=pipeline_result,
            record=record,
            work_ledger=work_ledger,
            circuit_resources=circuit_resources,
            evaluation_evidence=evaluation_evidence,
            materialization=materialization,
            preregistration=preregistration,
            protocol_violations=protocol_violations,
        )
        if outcome.status == "success":
            if not isinstance(trajectory_evidence, TrajectoryFidelityEvidence):
                msg = "Successful WP18 screening results require raw trajectory evidence."
                raise TypeError(msg)
            expected_context = screening_trajectory_context_checksum(
                candidate_configuration_checksum=outcome.candidate_configuration_checksum,
                cell_id=outcome.cell_id,
                result_schema_version=outcome.result_schema_version,
                result_record_checksum=outcome.result_record_checksum,
            )
            if (
                trajectory_evidence.evaluation_context_checksum != expected_context
                or trajectory_evidence.data_role != "screening_selection"
                or trajectory_evidence.evaluation_seed != outcome.evaluation_seed
                or len(trajectory_evidence.trajectory_fidelities) != record.config.trajectory_budget
                or float(trajectory_evidence.mean_fidelity).hex() != float(cast("float", outcome.noisy_fidelity)).hex()
            ):
                msg = "Raw trajectories do not reproduce the exact WP18 screening result."
                raise ValueError(msg)
        elif trajectory_evidence is not None:
            msg = "Failed WP18 screening results cannot carry raw trajectory evidence."
            raise ValueError(msg)
        return cls._from_verified_outcome(outcome, trajectory_evidence)

    @classmethod
    def from_operator_growth_artifact(cls, artifact: object) -> _LegacyScreeningSourceRecord:
        """Authenticate a standalone operator-growth artifact and raw evaluation.

        Returns:
            The authoritative screening source record.

        Raises:
            TypeError: If ``artifact`` is not the exact typed operator wrapper.
            ValueError: If its raw evaluation does not reproduce the promotion row.
        """
        from .operator_growth_pipeline import (  # noqa: PLC0415 -- runtime import avoids a module cycle
            OperatorGrowthPipelineArtifact,
        )

        if not isinstance(artifact, OperatorGrowthPipelineArtifact):
            msg = "artifact must be an OperatorGrowthPipelineArtifact."
            raise TypeError(msg)
        verified_artifact = OperatorGrowthPipelineArtifact.from_dict(artifact.to_dict())
        outcome = verified_artifact.verified_outcome
        trajectory_evidence = (
            None
            if verified_artifact.outer_evaluation is None
            else verified_artifact.outer_evaluation.trajectory_evidence
        )
        if outcome.status == "success":
            if trajectory_evidence is None:
                msg = "Successful operator growth requires raw outer-screening trajectories."
                raise ValueError(msg)
            if float(trajectory_evidence.mean_fidelity).hex() != float(cast("float", outcome.noisy_fidelity)).hex():
                msg = "Raw trajectories do not reproduce the operator-growth screening result."
                raise ValueError(msg)
        elif trajectory_evidence is not None:
            msg = "Failed operator-growth screening cannot carry successful raw trajectories."
            raise ValueError(msg)
        return cls._from_verified_outcome(outcome, trajectory_evidence)

    @classmethod
    def _from_verified_outcome(
        cls,
        outcome: VerifiedScreeningOutcome,
        trajectory_evidence: TrajectoryFidelityEvidence | None,
    ) -> _LegacyScreeningSourceRecord:
        """Build the authoritative wrapper after its typed source is checked.

        Returns:
            The normalized authoritative source record.
        """
        return cls(
            candidate_configuration_checksum=outcome.candidate_configuration_checksum,
            cell_id=outcome.cell_id,
            data_role="screening_selection",
            evaluation_seed=outcome.evaluation_seed,
            result_schema_version=outcome.result_schema_version,
            result_record_checksum=outcome.result_record_checksum,
            evaluation_evidence_checksum=outcome.evaluation_evidence_checksum,
            materialization_checksum=outcome.materialization_checksum,
            status=outcome.status,
            trajectory_evidence=trajectory_evidence,
            circuit_resources=outcome.circuit_resources,
            work_ledger=outcome.work_ledger,
            failure_code=outcome.failure_code,
            protocol_violations=outcome.protocol_violations,
        )

    def verified_outcome(self) -> VerifiedScreeningOutcome:
        """Mechanically derive the non-authoritative promotion projection.

        Returns:
            The promotion projection derived from raw source evidence.
        """
        evidence = self.trajectory_evidence
        resources = self.circuit_resources
        success = self.status == "success"
        if success:
            assert evidence is not None
            assert resources is not None
            noisy_fidelity: float | None = evidence.mean_fidelity
            resource_value: float | None = float(max(resources.native_two_qubit_gates_per_chain_edge, default=0))
        else:
            noisy_fidelity = None
            resource_value = None
        return VerifiedScreeningOutcome(
            candidate_configuration_checksum=self.candidate_configuration_checksum,
            cell_id=self.cell_id,
            data_role=self.data_role,
            evaluation_seed=self.evaluation_seed,
            result_schema_version=self.result_schema_version,
            result_record_checksum=self.result_record_checksum,
            evaluation_evidence_checksum=self.evaluation_evidence_checksum,
            materialization_checksum=self.materialization_checksum,
            status=self.status,
            noisy_fidelity=noisy_fidelity,
            resource_value=resource_value,
            normalized_work=self.work_ledger.normalized_compute(),
            failure_code=self.failure_code,
            circuit_resources=resources,
            work_ledger=self.work_ledger,
            protocol_violations=self.protocol_violations,
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered authoritative source field."""
        return {
            "schema_version": self.schema_version,
            "candidate_configuration_checksum": self.candidate_configuration_checksum,
            "cell_id": self.cell_id,
            "data_role": self.data_role,
            "evaluation_seed": self.evaluation_seed,
            "result_schema_version": self.result_schema_version,
            "result_record_checksum": self.result_record_checksum,
            "evaluation_evidence_checksum": self.evaluation_evidence_checksum,
            "materialization_checksum": self.materialization_checksum,
            "status": self.status,
            "trajectory_evidence": None if self.trajectory_evidence is None else self.trajectory_evidence.to_dict(),
            "circuit_resources": None if self.circuit_resources is None else self.circuit_resources.to_dict(),
            "work_ledger": self.work_ledger.to_dict(),
            "failure_code": self.failure_code,
            "protocol_violations": list(self.protocol_violations),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the raw evidence and exact promotion-source identity."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native source data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> _LegacyScreeningSourceRecord:
        """Decode and checksum-verify one authoritative screening source.

        Returns:
            The verified source record.

        Raises:
            ValueError: If its schema or checksum differs.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_SOURCE_RECORD_KEYS, name="WP22 screening source record")
        if mapping["schema_version"] != WP22_SCREENING_SOURCE_RECORD_SCHEMA_VERSION:
            msg = "WP22 screening source record uses an unsupported schema version."
            raise ValueError(msg)
        raw_evidence = mapping["trajectory_evidence"]
        raw_resources = mapping["circuit_resources"]
        source = cls(
            candidate_configuration_checksum=cast("str", mapping["candidate_configuration_checksum"]),
            cell_id=cast("str", mapping["cell_id"]),
            data_role=cast("Literal['screening_selection']", mapping["data_role"]),
            evaluation_seed=cast("int", mapping["evaluation_seed"]),
            result_schema_version=cast("str", mapping["result_schema_version"]),
            result_record_checksum=cast("str", mapping["result_record_checksum"]),
            evaluation_evidence_checksum=cast("str | None", mapping["evaluation_evidence_checksum"]),
            materialization_checksum=cast("str | None", mapping["materialization_checksum"]),
            status=cast("Literal['success', 'failure']", mapping["status"]),
            trajectory_evidence=(None if raw_evidence is None else TrajectoryFidelityEvidence.from_dict(raw_evidence)),
            circuit_resources=(None if raw_resources is None else CircuitResourceMetrics.from_dict(raw_resources)),
            work_ledger=WP20WorkLedger.from_dict(mapping["work_ledger"]),
            failure_code=cast("str | None", mapping["failure_code"]),
            protocol_violations=cast("tuple[str, ...]", mapping["protocol_violations"]),
        )
        if mapping["content_checksum"] != source.content_checksum:
            msg = "WP22 screening source record checksum changed during normalization."
            raise ValueError(msg)
        return source

    @classmethod
    def from_json(cls, payload: str) -> _LegacyScreeningSourceRecord:
        """Decode canonical checksum-sealed source JSON.

        Returns:
            The verified authoritative source record.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _wp18_trajectory_statistics(
    record: PipelineBenchmarkResult,
    fidelities: Sequence[float],
) -> tuple[float, float, float, float | None, float | None]:
    """Replay the exact WP18 noisy fixed-sample statistics.

    Returns:
        The mean, sample deviation, standard error, and optional confidence bounds.

    Raises:
        ValueError: If values or fixed-sample policy differ from the WP18 row.
    """
    values = tuple(float(value) for value in fidelities)
    if len(values) != record.config.trajectory_budget:
        msg = "WP18 trajectory sidecar count differs from the fixed screening budget."
        raise ValueError(msg)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        msg = "WP18 trajectory sidecar fidelities must be finite and lie in [0, 1]."
        raise ValueError(msg)
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    standard_error = deviation / math.sqrt(len(values))
    lower: float | None = None
    upper: float | None = None
    if record.config.evaluation_policy == "confidence_interval":
        method = record.config.confidence_interval_method
        if method not in {"normal", "normal_clipped"}:
            msg = f"Unsupported WP18 confidence interval method {method!r}."
            raise ValueError(msg)
        assert record.config.confidence_level is not None
        critical = statistics.NormalDist().inv_cdf((1.0 + record.config.confidence_level) / 2.0)
        lower = mean - critical * standard_error
        upper = mean + critical * standard_error
        if method == "normal_clipped":
            lower = max(0.0, lower)
            upper = min(1.0, upper)
    return mean, deviation, standard_error, lower, upper


@dataclass(frozen=True, slots=True)
class WP18ScreeningSourceArtifact:
    """Self-contained authoritative WP18 source for one screening row.

    The exact persisted fixed maps and trajectory-sidecar bytes are embedded,
    checksum-verified against the WP18 evidence ledger, and decoded here. A
    caller therefore cannot supply an unrelated tuple that merely shares the
    recorded scalar mean.
    """

    candidate: WP22CandidateConfiguration
    cell: ScreeningCell
    template: TrainingPipelineTemplate
    pipeline_result: TrainingPipelineResult
    record: PipelineBenchmarkRecord
    circuit_resources: CircuitResourceMetrics | None
    work_ledger: WP20WorkLedger
    evaluation_evidence: EvaluationEvidenceArtifact | None
    materialization: MaterializedCircuitArtifact | None
    preregistration: InitialPreregistration
    evaluation_maps: tuple[KrotovFixedMapEnsemble, ...]
    trajectory_sidecar_payload: bytes | None
    protocol_violations: tuple[str, ...] = ()
    verified_outcome: VerifiedScreeningOutcome = field(init=False, repr=False)
    trajectory_evidence: TrajectoryFidelityEvidence | None = field(init=False, repr=False)
    schema_version: str = field(default=WP18_SCREENING_SOURCE_ARTIFACT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Replay the complete typed source chain and raw sidecar custody.

        Raises:
            TypeError: If any source-chain artifact has the wrong typed schema.
            ValueError: If source links, maps, bytes, or derived statistics differ.
        """
        from .artifacts import (  # noqa: PLC0415 - runtime codec validation avoids a module cycle
            EvaluationEvidenceArtifact,
            MaterializedCircuitArtifact,
        )

        if not isinstance(self.candidate, WP22CandidateConfiguration):
            msg = "candidate must be a WP22CandidateConfiguration."
            raise TypeError(msg)
        if not isinstance(self.cell, ScreeningCell):
            msg = "cell must be a ScreeningCell."
            raise TypeError(msg)
        if not isinstance(self.template, TrainingPipelineTemplate):
            msg = "template must be a TrainingPipelineTemplate."
            raise TypeError(msg)
        if not isinstance(self.pipeline_result, TrainingPipelineResult):
            msg = "pipeline_result must be a TrainingPipelineResult."
            raise TypeError(msg)
        if not isinstance(self.record, (PipelineBenchmarkResult, PipelineBenchmarkFailure)):
            msg = "record must be a typed WP18 benchmark record."
            raise TypeError(msg)
        if not isinstance(self.work_ledger, WP20WorkLedger):
            msg = "work_ledger must be a WP20WorkLedger."
            raise TypeError(msg)
        if self.circuit_resources is not None and not isinstance(self.circuit_resources, CircuitResourceMetrics):
            msg = "circuit_resources must be CircuitResourceMetrics or None."
            raise TypeError(msg)
        if self.evaluation_evidence is not None and not isinstance(
            self.evaluation_evidence,
            EvaluationEvidenceArtifact,
        ):
            msg = "evaluation_evidence must be EvaluationEvidenceArtifact or None."
            raise TypeError(msg)
        if self.materialization is not None and not isinstance(self.materialization, MaterializedCircuitArtifact):
            msg = "materialization must be MaterializedCircuitArtifact or None."
            raise TypeError(msg)
        if not isinstance(self.preregistration, InitialPreregistration):
            msg = "preregistration must be an InitialPreregistration."
            raise TypeError(msg)
        maps = tuple(self.evaluation_maps)
        if not all(isinstance(item, KrotovFixedMapEnsemble) for item in maps):
            msg = "evaluation_maps must contain KrotovFixedMapEnsemble values."
            raise TypeError(msg)
        object.__setattr__(self, "evaluation_maps", maps)
        object.__setattr__(
            self,
            "protocol_violations",
            _ordered_strings(self.protocol_violations, "protocol_violations"),
        )
        outcome = VerifiedScreeningOutcome.from_pipeline_record(
            candidate=self.candidate,
            cell=self.cell,
            template=self.template,
            pipeline_result=self.pipeline_result,
            record=self.record,
            work_ledger=self.work_ledger,
            circuit_resources=self.circuit_resources,
            evaluation_evidence=self.evaluation_evidence,
            materialization=self.materialization,
            preregistration=self.preregistration,
            protocol_violations=self.protocol_violations,
        )
        if isinstance(self.record, PipelineBenchmarkResult):
            evidence = cast("EvaluationEvidenceArtifact", self.evaluation_evidence)
            sidecar = self.trajectory_sidecar_payload
            if (
                type(sidecar) is not bytes
                or self.record.trajectory_sidecar_path is None
                or self.record.trajectory_sidecar_checksum is None
            ):
                msg = "Successful WP18 screening requires its exact persisted trajectory sidecar bytes."
                raise TypeError(msg)
            refs = evidence.evaluation_map_artifacts
            if not maps or len(maps) != len(refs):
                msg = "WP18 screening source must embed every exact persisted evaluation map."
                raise ValueError(msg)
            provider_checksum = evidence.evaluation_provider_checksum
            if provider_checksum is None:
                msg = "Noisy WP18 screening requires its exact evaluation provider checksum."
                raise ValueError(msg)
            config = self.record.config
            for ref, ensemble in zip(refs, maps, strict=True):
                if (
                    ref.role != "screening_selection"
                    or ensemble.role != ref.role
                    or ensemble.ensemble_id != ref.ensemble_id
                    or ensemble.content_checksum != ref.content_checksum
                    or artifact_checksum(ensemble.to_json().encode()) != ref.file_checksum
                    or ensemble.resolved_seed != config.evaluation_seed
                    or ensemble.stage_configuration_checksum != config.configuration_checksum
                    or ensemble.circuit_checksum != config.materialized_circuit_checksum
                    or ensemble.provider_checksum != provider_checksum
                ):
                    msg = "Embedded WP18 evaluation maps differ from their authenticated row, file, or provider links."
                    raise ValueError(msg)
            if (
                sum(item.trajectory_count for item in maps) != config.trajectory_budget
                or len({item.ensemble_id for item in maps}) != len(maps)
                or len({item.content_checksum for item in maps}) != len(maps)
                or self.record.sampled_nonidentity_events != sum(item.nonidentity_event_count for item in maps)
            ):
                msg = "Embedded WP18 maps do not form the exact fixed, distinct evaluation ensemble."
                raise ValueError(msg)
            partitions = tuple(
                {
                    "ensemble_id": item.ensemble_id,
                    "content_checksum": item.content_checksum,
                    "trajectory_count": item.trajectory_count,
                }
                for item in maps
            )
            fidelities = read_phase2_trajectory_sidecar(
                sidecar,
                expected_evaluation_row_id=self.record.evaluation_row_id,
                expected_pipeline_training_id=config.pipeline_training_id,
                expected_map_role="screening_selection",
                expected_map_partitions=partitions,
                expected_count=config.trajectory_budget,
                expected_checksum=self.record.trajectory_sidecar_checksum,
            )
            recorded_statistics = (
                self.record.test_noisy_fidelity,
                self.record.noisy_fidelity_standard_deviation,
                self.record.noisy_fidelity_standard_error,
                self.record.confidence_interval_lower,
                self.record.confidence_interval_upper,
            )
            if recorded_statistics != _wp18_trajectory_statistics(self.record, fidelities):
                msg = "Authenticated WP18 trajectory sidecar does not reproduce the complete result statistics."
                raise ValueError(msg)
            trajectory_evidence = TrajectoryFidelityEvidence(
                evaluation_context_checksum=screening_trajectory_context_checksum(
                    candidate_configuration_checksum=outcome.candidate_configuration_checksum,
                    cell_id=outcome.cell_id,
                    result_schema_version=outcome.result_schema_version,
                    result_record_checksum=outcome.result_record_checksum,
                ),
                data_role="screening_selection",
                evaluation_seed=outcome.evaluation_seed,
                trajectory_fidelities=fidelities,
            )
        else:
            if maps or self.trajectory_sidecar_payload is not None:
                msg = "Failed WP18 screening records cannot carry successful map or trajectory sidecars."
                raise ValueError(msg)
            trajectory_evidence = None
        object.__setattr__(self, "verified_outcome", outcome)
        object.__setattr__(self, "trajectory_evidence", trajectory_evidence)

    def _content_dict(self) -> dict[str, object]:
        """Return the complete canonical source chain and exact sidecar bytes."""
        return {
            "schema_version": self.schema_version,
            "candidate": self.candidate.to_dict(),
            "cell": self.cell.to_dict(),
            "template": self.template.to_dict(),
            "pipeline_result": self.pipeline_result.to_dict(),
            "record": self.record.to_dict(),
            "circuit_resources": None if self.circuit_resources is None else self.circuit_resources.to_dict(),
            "work_ledger": self.work_ledger.to_dict(),
            "evaluation_evidence": (None if self.evaluation_evidence is None else self.evaluation_evidence.to_dict()),
            "materialization": None if self.materialization is None else self.materialization.to_dict(),
            "preregistration": self.preregistration.to_dict(),
            "evaluation_maps": [item.to_dict() for item in self.evaluation_maps],
            "trajectory_sidecar_base64": (
                None
                if self.trajectory_sidecar_payload is None
                else base64.b64encode(self.trajectory_sidecar_payload).decode("ascii")
            ),
            "protocol_violations": list(self.protocol_violations),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum the complete authoritative WP18 source package."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return a checksum-sealed self-contained source artifact."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> WP18ScreeningSourceArtifact:
        """Decode and fully replay one persisted WP18 source artifact.

        Returns:
            The verified self-contained source artifact.

        Raises:
            TypeError: If a serialized collection or scalar has the wrong type.
            ValueError: If schema, base64, source links, or checksums differ.
        """
        from .artifacts import (  # noqa: PLC0415 - runtime codec validation avoids a module cycle
            EvaluationEvidenceArtifact,
            MaterializedCircuitArtifact,
        )

        mapping = verify_sealed_mapping(
            data,
            expected_keys=_WP18_SOURCE_ARTIFACT_KEYS,
            name="WP18 screening source artifact",
        )
        if mapping["schema_version"] != WP18_SCREENING_SOURCE_ARTIFACT_SCHEMA_VERSION:
            msg = "WP18 screening source artifact uses an unsupported schema version."
            raise ValueError(msg)
        raw_maps = mapping["evaluation_maps"]
        if type(raw_maps) is not tuple:
            msg = "evaluation_maps must be a serialized sequence."
            raise TypeError(msg)
        encoded_sidecar = mapping["trajectory_sidecar_base64"]
        if encoded_sidecar is not None and type(encoded_sidecar) is not str:
            msg = "trajectory_sidecar_base64 must be a string or null."
            raise TypeError(msg)
        try:
            sidecar = (
                None if encoded_sidecar is None else base64.b64decode(encoded_sidecar.encode("ascii"), validate=True)
            )
        except (UnicodeEncodeError, binascii.Error) as error:
            msg = "trajectory_sidecar_base64 is not canonical base64."
            raise ValueError(msg) from error
        raw_resources = mapping["circuit_resources"]
        raw_evidence = mapping["evaluation_evidence"]
        raw_materialization = mapping["materialization"]
        source = cls(
            candidate=WP22CandidateConfiguration.from_dict(mapping["candidate"]),
            cell=ScreeningCell.from_dict(mapping["cell"]),
            template=TrainingPipelineTemplate.from_dict(mapping["template"]),
            pipeline_result=TrainingPipelineResult.from_dict(mapping["pipeline_result"]),
            record=pipeline_benchmark_record_from_dict(mapping["record"]),
            circuit_resources=(None if raw_resources is None else CircuitResourceMetrics.from_dict(raw_resources)),
            work_ledger=WP20WorkLedger.from_dict(mapping["work_ledger"]),
            evaluation_evidence=(None if raw_evidence is None else EvaluationEvidenceArtifact.from_dict(raw_evidence)),
            materialization=(
                None if raw_materialization is None else MaterializedCircuitArtifact.from_dict(raw_materialization)
            ),
            preregistration=InitialPreregistration.from_dict(mapping["preregistration"]),
            evaluation_maps=tuple(
                KrotovFixedMapEnsemble.from_dict(thaw_json_mapping(cast("Mapping[str, object]", item)))
                for item in raw_maps
            ),
            trajectory_sidecar_payload=sidecar,
            protocol_violations=cast("tuple[str, ...]", mapping["protocol_violations"]),
        )
        if mapping["content_checksum"] != source.content_checksum:
            msg = "WP18 screening source artifact checksum changed during normalization."
            raise ValueError(msg)
        return source

    @classmethod
    def from_json(cls, payload: str) -> WP18ScreeningSourceArtifact:
        """Decode canonical checksum-sealed source JSON.

        Returns:
            The verified self-contained source artifact.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class ScreeningSourceRecord:
    """Typed wrapper around a complete WP18 or operator-growth source.

    The wrapper contains no caller-authored scientific projection. Every
    property is replayed from one complete typed source artifact.
    """

    source_artifact: WP18ScreeningSourceArtifact | OperatorGrowthPipelineArtifact
    schema_version: str = field(default=WP22_SCREENING_SOURCE_RECORD_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Deep-verify and normalize the complete nested source artifact.

        Raises:
            TypeError: If the nested artifact is not an accepted production source.
        """
        from .operator_growth_pipeline import (  # noqa: PLC0415 - module cycle
            OperatorGrowthPipelineArtifact,
        )

        artifact = self.source_artifact
        if isinstance(artifact, WP18ScreeningSourceArtifact):
            verified: WP18ScreeningSourceArtifact | OperatorGrowthPipelineArtifact = (
                WP18ScreeningSourceArtifact.from_dict(artifact.to_dict())
            )
        elif isinstance(artifact, OperatorGrowthPipelineArtifact):
            verified = OperatorGrowthPipelineArtifact.from_dict(artifact.to_dict())
        else:
            msg = "source_artifact must be a complete WP18ScreeningSourceArtifact or OperatorGrowthPipelineArtifact."
            raise TypeError(msg)
        object.__setattr__(self, "source_artifact", verified)

    @property
    def source_kind(self) -> Literal["wp18_pipeline", "operator_growth"]:
        """Authoritative source discriminator."""
        return "wp18_pipeline" if isinstance(self.source_artifact, WP18ScreeningSourceArtifact) else "operator_growth"

    def verified_outcome(self) -> VerifiedScreeningOutcome:
        """Replay the promotion projection from the complete typed source.

        Returns:
            The mechanically source-derived promotion outcome.
        """
        return self.source_artifact.verified_outcome

    @property
    def trajectory_evidence(self) -> TrajectoryFidelityEvidence | None:
        """Raw trajectories dereferenced from the authoritative source."""
        if isinstance(self.source_artifact, WP18ScreeningSourceArtifact):
            return self.source_artifact.trajectory_evidence
        evaluation = self.source_artifact.outer_evaluation
        return None if evaluation is None else evaluation.trajectory_evidence

    @property
    def candidate_configuration_checksum(self) -> str:
        """Source-derived candidate identity."""
        return self.verified_outcome().candidate_configuration_checksum

    @property
    def cell_id(self) -> str:
        """Source-derived screening cell identity."""
        return self.verified_outcome().cell_id

    @property
    def data_role(self) -> Literal["screening_selection"]:
        """Mechanically enforced outer-screening role."""
        return cast("Literal['screening_selection']", self.verified_outcome().data_role)

    @property
    def evaluation_seed(self) -> int:
        """Source-derived outer evaluation seed."""
        return self.verified_outcome().evaluation_seed

    @property
    def result_schema_version(self) -> str:
        """Complete source result schema."""
        return self.verified_outcome().result_schema_version

    @property
    def result_record_checksum(self) -> str:
        """Complete source result checksum."""
        return self.verified_outcome().result_record_checksum

    @property
    def evaluation_evidence_checksum(self) -> str | None:
        """Authenticated evaluation-evidence identity."""
        return self.verified_outcome().evaluation_evidence_checksum

    @property
    def materialization_checksum(self) -> str | None:
        """Authenticated materialization identity."""
        return self.verified_outcome().materialization_checksum

    @property
    def status(self) -> Literal["success", "failure"]:
        """Typed source status."""
        return self.verified_outcome().status

    @property
    def circuit_resources(self) -> CircuitResourceMetrics | None:
        """Source-linked compiler-derived resources."""
        return self.verified_outcome().circuit_resources

    @property
    def work_ledger(self) -> WP20WorkLedger:
        """Source-linked detailed work ledger."""
        return self.verified_outcome().work_ledger

    @property
    def failure_code(self) -> str | None:
        """Source-derived failure code."""
        return self.verified_outcome().failure_code

    @property
    def protocol_violations(self) -> tuple[str, ...]:
        """Source-derived protocol violations."""
        return self.verified_outcome().protocol_violations

    @classmethod
    def from_pipeline_record(
        cls,
        *,
        candidate: WP22CandidateConfiguration,
        cell: ScreeningCell,
        template: TrainingPipelineTemplate,
        pipeline_result: TrainingPipelineResult,
        record: PipelineBenchmarkRecord,
        work_ledger: WP20WorkLedger,
        circuit_resources: CircuitResourceMetrics | None,
        evaluation_evidence: EvaluationEvidenceArtifact | None,
        materialization: MaterializedCircuitArtifact | None,
        preregistration: InitialPreregistration,
        evaluation_maps: Sequence[KrotovFixedMapEnsemble] = (),
        trajectory_sidecar_payload: bytes | None = None,
        protocol_violations: Sequence[str] = (),
    ) -> ScreeningSourceRecord:
        """Build a wrapper only from the complete authenticated WP18 source.

        Returns:
            The deeply verified authoritative screening source.
        """
        return cls(
            WP18ScreeningSourceArtifact(
                candidate=candidate,
                cell=cell,
                template=template,
                pipeline_result=pipeline_result,
                record=record,
                circuit_resources=circuit_resources,
                work_ledger=work_ledger,
                evaluation_evidence=evaluation_evidence,
                materialization=materialization,
                preregistration=preregistration,
                evaluation_maps=tuple(evaluation_maps),
                trajectory_sidecar_payload=trajectory_sidecar_payload,
                protocol_violations=tuple(protocol_violations),
            )
        )

    @classmethod
    def from_wp18_artifact(cls, artifact: WP18ScreeningSourceArtifact) -> ScreeningSourceRecord:
        """Wrap a complete typed WP18 source artifact.

        Returns:
            The deeply verified authoritative screening source.
        """
        return cls(artifact)

    @classmethod
    def from_operator_growth_artifact(cls, artifact: object) -> ScreeningSourceRecord:
        """Wrap only a complete typed standalone operator-growth artifact.

        Returns:
            The deeply verified authoritative screening source.

        Raises:
            TypeError: If the artifact is not the production operator wrapper.
        """
        from .operator_growth_pipeline import (  # noqa: PLC0415 - module cycle
            OperatorGrowthPipelineArtifact,
        )

        if not isinstance(artifact, OperatorGrowthPipelineArtifact):
            msg = "artifact must be an OperatorGrowthPipelineArtifact."
            raise TypeError(msg)
        return cls(artifact)

    def _content_dict(self) -> dict[str, object]:
        """Return the complete nested source; no scalar projection is stored."""
        return {
            "schema_version": self.schema_version,
            "source_kind": self.source_kind,
            "source_artifact": self.source_artifact.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum the complete authoritative source package."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native source data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ScreeningSourceRecord:
        """Decode and fully verify the nested authoritative source.

        Returns:
            The deeply verified authoritative screening source.

        Raises:
            ValueError: If source kind, schema, nested source, or checksum differs.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_SOURCE_RECORD_KEYS, name="WP22 screening source record")
        if mapping["schema_version"] != WP22_SCREENING_SOURCE_RECORD_SCHEMA_VERSION:
            msg = "WP22 screening source record uses an unsupported schema version."
            raise ValueError(msg)
        kind = mapping["source_kind"]
        if kind == "wp18_pipeline":
            artifact: WP18ScreeningSourceArtifact | OperatorGrowthPipelineArtifact = (
                WP18ScreeningSourceArtifact.from_dict(mapping["source_artifact"])
            )
        elif kind == "operator_growth":
            from .operator_growth_pipeline import (  # noqa: PLC0415 - module cycle
                OperatorGrowthPipelineArtifact,
            )

            artifact = OperatorGrowthPipelineArtifact.from_dict(mapping["source_artifact"])
        else:
            msg = "source_kind must identify a WP18 pipeline or operator-growth artifact."
            raise ValueError(msg)
        source = cls(artifact)
        if mapping["content_checksum"] != source.content_checksum:
            msg = "WP22 screening source record checksum changed during normalization."
            raise ValueError(msg)
        return source

    @classmethod
    def from_json(cls, payload: str) -> ScreeningSourceRecord:
        """Decode canonical checksum-sealed source JSON.

        Returns:
            The deeply verified authoritative screening source.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True, init=False)
class ProductionScreeningSourceRecord:
    """One promotion row derived only by reopening a WP22E terminal attempt."""

    candidate: ScreeningCandidateRef
    cell: ScreeningCell
    job: TrainingJob
    outcome: TrainingJobOutcome
    result_custody: ProductionResultCustody
    fixed_trajectory_count: int
    circuit_resources: CircuitResourceMetrics | None
    schema_version: str = field(default=WP22_PRODUCTION_SCREENING_SOURCE_SCHEMA_VERSION, init=False)

    def __init__(
        self,
        candidate: ScreeningCandidateRef,
        cell: ScreeningCell,
        job: TrainingJob,
        job_directory: Path,
        *,
        fixed_trajectory_count: int,
    ) -> None:
        """Load the first durable outcome and its immutable production manifest.

        Raises:
            TypeError: If an identity or path has the wrong typed schema.
            ValueError: If the job, first attempt, raw evidence, or resources differ.
        """
        if not isinstance(candidate, ScreeningCandidateRef) or not isinstance(cell, ScreeningCell):
            msg = "candidate and cell must be typed screening-manifest records."
            raise TypeError(msg)
        if not isinstance(job, TrainingJob) or not isinstance(job_directory, Path):
            msg = "job and job_directory must be typed production inputs."
            raise TypeError(msg)
        count = require_int(fixed_trajectory_count, "fixed_trajectory_count", minimum=1)
        expected_job_fields = (
            (job.preset, "paper-screen"),
            (job.data_role, "screening_selection"),
            (job.qubit_count, 6),
            (job.candidate_configuration_checksum, candidate.configuration_checksum),
            (job.target_instance_id, cell.target_instance_id),
            (job.family_id, cell.family_id),
            (job.stratum_id, cell.stratum_id),
            (job.optimization_seed, cell.optimization_seed),
            (job.evaluation_seed, cell.screening_seed),
        )
        if any(actual != expected for actual, expected in expected_job_fields):
            msg = "Paper-screen job differs from its exact candidate/cell identity."
            raise ValueError(msg)
        history = load_training_job_outcome_history(job_directory, job)
        if not history:
            msg = "Paper-screen job has no durable first terminal outcome."
            raise ValueError(msg)
        outcome = history[0]
        custody = reopen_terminal_production_attempt(job, outcome, job_directory)
        expected_kind = "operator_growth" if job.implementation_kind == "operator_growth" else "pipeline"
        validate_production_job_custody(
            custody,
            job,
            outcome,
            expected_data_role="screening_selection",
            expected_trajectory_count=count,
            allowed_artifact_kinds=(expected_kind,),
        )
        resources: CircuitResourceMetrics | None = None
        if outcome.status == "success":
            circuit = require_mapping(custody.resource_payload.get("circuit"), "runtime_resources.circuit")
            resources = CircuitResourceMetrics.from_dict(circuit.get("compiled_resources"))
            supplied_checksum = require_checksum(
                circuit.get("compiled_resources_checksum"),
                "compiled_resources_checksum",
            )
            raw_counts = circuit.get("native_two_qubit_gates_per_chain_edge")
            if (
                supplied_checksum != resources.content_checksum
                or not isinstance(raw_counts, tuple)
                or tuple(raw_counts) != resources.native_two_qubit_gates_per_chain_edge
            ):
                msg = "Screening resource projection differs from compiler-derived manifest evidence."
                raise ValueError(msg)
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "job", job)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "result_custody", custody)
        object.__setattr__(self, "fixed_trajectory_count", count)
        object.__setattr__(self, "circuit_resources", resources)
        object.__setattr__(self, "schema_version", WP22_PRODUCTION_SCREENING_SOURCE_SCHEMA_VERSION)

    @property
    def content_checksum(self) -> str:
        """Checksum of exact plan, outcome, and reopened-manifest identities."""
        return canonical_checksum({
            "schema_version": self.schema_version,
            "candidate_checksum": canonical_checksum(self.candidate.to_dict()),
            "cell_checksum": canonical_checksum(self.cell.to_dict()),
            "job_checksum": self.job.content_checksum,
            "outcome_checksum": self.outcome.content_checksum,
            "result_custody_checksum": self.result_custody.content_checksum,
            "fixed_trajectory_count": self.fixed_trajectory_count,
        })

    def promotion_observation(self) -> PromotionObservation:
        """Project the mechanically reopened first attempt onto the WP15 row.

        Returns:
            A scalar promotion row derived only from immutable raw evidence.
        """
        custody = self.result_custody
        if self.outcome.status == "success":
            resources = cast("CircuitResourceMetrics", self.circuit_resources)
            fidelity = cast("float", custody.mean_fidelity)
            resource_value = float(max(resources.native_two_qubit_gates_per_chain_edge, default=0))
            failure_code = None
        else:
            fidelity = None
            resource_value = None
            failure_code = self.outcome.exception_type
        return PromotionObservation(
            configuration_checksum=self.candidate.configuration_checksum,
            cell_id=self.cell.cell_id,
            result_schema_version=custody.reference.schema_version,
            result_record_checksum=custody.reference.content_checksum,
            status=self.outcome.status,
            noisy_fidelity=fidelity,
            resource_value=resource_value,
            normalized_work=require_float(
                custody.resource_payload.get("normalized_work"),
                "normalized_work",
                minimum=0.0,
            ),
            failure_code=failure_code,
            protocol_violations=(),
        )

    def to_dict(self) -> dict[str, object]:
        """Return an audit record which requires manifest reopening for reuse."""
        return {
            "schema_version": self.schema_version,
            "candidate": self.candidate.to_dict(),
            "cell": self.cell.to_dict(),
            "job": self.job.to_dict(),
            "outcome": self.outcome.to_dict(),
            "result_custody": self.result_custody.to_dict(),
            "fixed_trajectory_count": self.fixed_trajectory_count,
            "content_checksum": self.content_checksum,
        }


@dataclass(frozen=True, slots=True, init=False)
class ProductionScreeningCustody:
    """Complete context-owned replay of all 1,296 paper-screen jobs."""

    context: TrainingExecutionContext
    records: tuple[ProductionScreeningSourceRecord, ...]

    def __init__(
        self,
        context: TrainingExecutionContext,
        records: Sequence[ProductionScreeningSourceRecord],
    ) -> None:
        """Bind every reopened row to its exact context job and manifest object.

        Raises:
            TypeError: If context, manifest, design, targets, or records use unsupported schemas.
            ValueError: If any job, candidate, cell, target, policy, count, attempt, or result root differs.
        """
        if not isinstance(context, TrainingExecutionContext):
            msg = "context must be a TrainingExecutionContext."
            raise TypeError(msg)
        manifest = context.screening_manifest
        design = context.required_sample_size_design
        if context.plan.preset != "paper-screen":
            msg = "Production screening custody requires a paper-screen execution context."
            raise ValueError(msg)
        if not isinstance(manifest, ScreeningManifest) or not isinstance(design, SampleSizeDesign):
            msg = "Paper-screen custody requires its exact manifest and pilot-derived design."
            raise TypeError(msg)
        values = tuple(records)
        if len(values) != 1_296 or not all(isinstance(item, ProductionScreeningSourceRecord) for item in values):
            msg = "Production screening custody requires exactly 1,296 reopened source records."
            raise TypeError(msg)
        if len(context.plan.jobs) != len(values):
            msg = "Paper-screen context and production records differ from the exact 1,296-job universe."
            raise ValueError(msg)
        targets = tuple(
            item
            for item in context.target_manifests
            if item.content_checksum == manifest.screening_target_manifest_checksum
        )
        if len(targets) != 1 or not isinstance(targets[0], TargetPopulationManifest):
            msg = "Paper-screen context lacks its exact screening target manifest."
            raise TypeError(msg)
        target_manifest = targets[0]
        target_manifest_checksum = target_manifest.content_checksum
        execution_source_checksum = context.execution_source_manifest.content_checksum
        target_by_id = {item.target_instance_id: item for item in target_manifest.instances}
        target_checksum_by_id = {item.target_instance_id: item.content_checksum for item in target_manifest.instances}
        candidate_by_checksum = {item.configuration_checksum: item for item in manifest.candidates}
        cell_by_id = {item.cell_id: item for item in manifest.cells}
        expected_pairs = {
            (candidate.configuration_checksum, cell.cell_id)
            for candidate in manifest.candidates
            for cell in manifest.cells
        }
        actual_pairs: set[tuple[str, str]] = set()
        for job, record in zip(context.plan.jobs, values, strict=True):
            candidate = candidate_by_checksum.get(job.candidate_configuration_checksum)
            cell = cell_by_id.get(job.optimization_block_id)
            target = target_by_id.get(job.target_instance_id)
            if candidate is None or cell is None or target is None:
                msg = "A paper-screen job lies outside its exact candidate, cell, or target universe."
                raise ValueError(msg)
            if record.job is not job or record.candidate is not candidate or record.cell is not cell:
                msg = "A production screening record is not owned by the exact context job and manifest objects."
                raise ValueError(msg)
            exact_fields = (
                (job.target_manifest_checksum, target_manifest_checksum),
                (job.target_configuration_checksum, target_manifest.population_config_checksum),
                (job.target_spec_checksum, target_checksum_by_id[target.target_instance_id]),
                (job.target_instance_id, cell.target_instance_id),
                (job.family_id, target.family_id),
                (job.family_id, cell.family_id),
                (job.stratum_id, target.stratum_id),
                (job.stratum_id, cell.stratum_id),
                (job.qubit_count, target.qubit_count),
                (job.qubit_count, cell.qubit_count),
                (job.optimization_seed, cell.optimization_seed),
                (job.evaluation_seed, cell.screening_seed),
                (job.data_role, "screening_selection"),
                (record.fixed_trajectory_count, design.fixed_test_trajectory_count),
                (record.outcome.attempt, 1),
                (record.result_custody.reference.attempt, 1),
            )
            if any(actual != expected for actual, expected in exact_fields):
                msg = "Production screening custody differs from its target, policy, count, or first attempt."
                raise ValueError(msg)
            validate_production_job_custody(
                record.result_custody,
                job,
                record.outcome,
                expected_data_role="screening_selection",
                expected_trajectory_count=design.fixed_test_trajectory_count,
                expected_execution_source_manifest_checksum=execution_source_checksum,
                allowed_artifact_kinds=(
                    "operator_growth" if job.implementation_kind == "operator_growth" else "pipeline",
                ),
            )
            actual_pairs.add((candidate.configuration_checksum, cell.cell_id))
        if actual_pairs != expected_pairs:
            msg = "Production screening custody differs from the exact candidate-by-cell Cartesian universe."
            raise ValueError(msg)
        references = tuple(item.result_custody.reference.content_checksum for item in values)
        if len(set(references)) != len(values):
            msg = "Every screening job must have a distinct immutable first-attempt result reference."
            raise ValueError(msg)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "records", values)

    @classmethod
    def reopen(
        cls,
        context: TrainingExecutionContext,
        output_root: Path,
    ) -> ProductionScreeningCustody:
        """Reopen all context jobs and return their aggregate custody.

        Returns:
            The exact context-owned production screening custody.

        Raises:
            TypeError: If context or output root has the wrong type.
        """
        if not isinstance(context, TrainingExecutionContext) or not isinstance(output_root, Path):
            msg = "context and output_root must be typed production inputs."
            raise TypeError(msg)
        manifest = context.screening_manifest
        design = context.required_sample_size_design
        if not isinstance(manifest, ScreeningManifest) or not isinstance(design, SampleSizeDesign):
            msg = "Paper-screen context lacks its exact manifest or pilot-derived design."
            raise TypeError(msg)
        candidate_by_checksum = {item.configuration_checksum: item for item in manifest.candidates}
        cell_by_id = {item.cell_id: item for item in manifest.cells}
        records = tuple(
            ProductionScreeningSourceRecord(
                candidate_by_checksum[job.candidate_configuration_checksum],
                cell_by_id[job.optimization_block_id],
                job,
                output_root / job.output_path,
                fixed_trajectory_count=design.fixed_test_trajectory_count,
            )
            for job in context.plan.jobs
        )
        return cls(context, records)

    def build_evidence(
        self,
        *,
        evidence_id: str = "wp22_paper_screen_evidence_v1",
    ) -> tuple[ScreeningEvidence, PromotionDecision]:
        """Build the mechanical promotion projection from this aggregate.

        Returns:
            Complete screening evidence and its unique promotion decision.
        """
        return build_production_screening_evidence_from_records(
            self.context.preregistration,
            cast("ScreeningManifest", self.context.screening_manifest),
            self.records,
            evidence_id=evidence_id,
        )


@dataclass(frozen=True, slots=True)
class ProductionResourceProjection:
    """One source-locked runtime and compiler-resource projection."""

    job_checksum: str
    result_reference_checksum: str
    resource_document_checksum: str
    execution_source_manifest_checksum: str
    method_id: str
    candidate_configuration_checksum: str
    data_role: str
    family_id: str
    stratum_id: str
    qubit_count: int
    status: Literal["success", "failure"]
    normalized_work: float
    structural_prefix_checksums: tuple[str, ...]
    circuit_binding_checksum: str | None
    compiled_resources_checksum: str | None
    native_two_qubit_gates_per_chain_edge: tuple[int, ...]
    _content_checksum: str = field(init=False, repr=False, compare=False)
    schema_version: str = field(default=PRODUCTION_RESOURCE_PROJECTION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the exact source, result, work, and compiler identities.

        Raises:
            ValueError: If a checksum, role, status, work value, or resource projection is invalid.
        """
        for name in (
            "job_checksum",
            "result_reference_checksum",
            "resource_document_checksum",
            "execution_source_manifest_checksum",
            "candidate_configuration_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(self, "method_id", require_slug(self.method_id, "method_id"))
        object.__setattr__(self, "data_role", require_slug(self.data_role, "data_role"))
        object.__setattr__(self, "family_id", require_slug(self.family_id, "family_id"))
        object.__setattr__(self, "stratum_id", require_slug(self.stratum_id, "stratum_id"))
        object.__setattr__(self, "qubit_count", require_int(self.qubit_count, "qubit_count", minimum=1))
        if self.status not in {"success", "failure"}:
            msg = "status must be success or failure."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "normalized_work",
            require_float(self.normalized_work, "normalized_work", minimum=0.0),
        )
        prefixes = tuple(
            require_checksum(item, "structural_prefix_checksum") for item in self.structural_prefix_checksums
        )
        object.__setattr__(self, "structural_prefix_checksums", prefixes)
        circuit = self.circuit_binding_checksum
        compiled = self.compiled_resources_checksum
        if circuit is not None:
            object.__setattr__(self, "circuit_binding_checksum", require_checksum(circuit, "circuit_binding_checksum"))
        if compiled is not None:
            object.__setattr__(
                self,
                "compiled_resources_checksum",
                require_checksum(compiled, "compiled_resources_checksum"),
            )
        counts = tuple(
            require_int(item, "native_two_qubit_gate_count", minimum=0)
            for item in self.native_two_qubit_gates_per_chain_edge
        )
        object.__setattr__(self, "native_two_qubit_gates_per_chain_edge", counts)
        if self.status == "success" and (circuit is None or compiled is None):
            msg = "Successful resource projections require circuit and compiler identities."
            raise ValueError(msg)
        if compiled is None and counts:
            msg = "Native compiler counts require a compiled-resource identity."
            raise ValueError(msg)
        object.__setattr__(self, "_content_checksum", canonical_checksum(self._content_dict()))

    @property
    def content_checksum(self) -> str:
        """Checksum the complete source-linked resource projection."""
        return self._content_checksum

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered projection field."""
        return {
            "schema_version": self.schema_version,
            "job_checksum": self.job_checksum,
            "result_reference_checksum": self.result_reference_checksum,
            "resource_document_checksum": self.resource_document_checksum,
            "execution_source_manifest_checksum": self.execution_source_manifest_checksum,
            "method_id": self.method_id,
            "candidate_configuration_checksum": self.candidate_configuration_checksum,
            "data_role": self.data_role,
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "qubit_count": self.qubit_count,
            "status": self.status,
            "normalized_work": self.normalized_work,
            "structural_prefix_checksums": list(self.structural_prefix_checksums),
            "circuit_binding_checksum": self.circuit_binding_checksum,
            "compiled_resources_checksum": self.compiled_resources_checksum,
            "native_two_qubit_gates_per_chain_edge": list(self.native_two_qubit_gates_per_chain_edge),
        }

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native resource evidence."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> ProductionResourceProjection:
        """Decode and verify one persisted production resource projection.

        Returns:
            The strict typed resource projection.

        Raises:
            ValueError: If its schema, checksum, or normalized content differs.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_PRODUCTION_RESOURCE_PROJECTION_KEYS,
            name="production resource projection",
        )
        if mapping["schema_version"] != PRODUCTION_RESOURCE_PROJECTION_SCHEMA_VERSION:
            msg = "Production resource projection uses an unsupported schema version."
            raise ValueError(msg)
        projection = cls(
            job_checksum=cast("str", mapping["job_checksum"]),
            result_reference_checksum=cast("str", mapping["result_reference_checksum"]),
            resource_document_checksum=cast("str", mapping["resource_document_checksum"]),
            execution_source_manifest_checksum=cast("str", mapping["execution_source_manifest_checksum"]),
            method_id=cast("str", mapping["method_id"]),
            candidate_configuration_checksum=cast("str", mapping["candidate_configuration_checksum"]),
            data_role=cast("str", mapping["data_role"]),
            family_id=cast("str", mapping["family_id"]),
            stratum_id=cast("str", mapping["stratum_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            status=cast("Literal['success', 'failure']", mapping["status"]),
            normalized_work=cast("float", mapping["normalized_work"]),
            structural_prefix_checksums=cast("tuple[str, ...]", mapping["structural_prefix_checksums"]),
            circuit_binding_checksum=cast("str | None", mapping["circuit_binding_checksum"]),
            compiled_resources_checksum=cast("str | None", mapping["compiled_resources_checksum"]),
            native_two_qubit_gates_per_chain_edge=cast(
                "tuple[int, ...]",
                mapping["native_two_qubit_gates_per_chain_edge"],
            ),
        )
        if mapping["content_checksum"] != projection.content_checksum:
            msg = "Production resource projection checksum changed during normalization."
            raise ValueError(msg)
        return projection


def _production_resource_projection(
    record: object,
) -> ProductionResourceProjection:
    """Derive one resource projection from a reopened pilot or screen record.

    Returns:
        The exact source-linked runtime/compiler projection.

    Raises:
        TypeError: If the record does not expose typed production custody.
        ValueError: If compiler projections differ from their resource document.
    """
    job = getattr(record, "job", None)
    outcome = getattr(record, "outcome", None)
    custody = getattr(record, "result_custody", None)
    if (
        not isinstance(job, TrainingJob)
        or not isinstance(outcome, TrainingJobOutcome)
        or not isinstance(
            custody,
            ProductionResultCustody,
        )
    ):
        msg = "Resource calibration accepts only reopened production job records."
        raise TypeError(msg)
    raw_circuit = custody.resource_payload.get("circuit")
    circuit_checksum: str | None = None
    compiled_checksum: str | None = None
    counts: tuple[int, ...] = ()
    if raw_circuit is not None:
        circuit = require_mapping(raw_circuit, "runtime_resources.circuit")
        circuit_checksum = require_checksum(
            circuit.get("circuit_binding_checksum"),
            "runtime_resources.circuit.circuit_binding_checksum",
        )
        compiled = CircuitResourceMetrics.from_dict(circuit.get("compiled_resources"))
        compiled_checksum = require_checksum(
            circuit.get("compiled_resources_checksum"),
            "runtime_resources.circuit.compiled_resources_checksum",
        )
        raw_counts = circuit.get("native_two_qubit_gates_per_chain_edge")
        if not isinstance(raw_counts, tuple):
            msg = "Runtime native two-qubit resource counts must be a canonical sequence."
            raise TypeError(msg)
        counts = tuple(require_int(item, "native_two_qubit_gate_count", minimum=0) for item in raw_counts)
        if compiled_checksum != compiled.content_checksum or counts != compiled.native_two_qubit_gates_per_chain_edge:
            msg = "Runtime compiler resource projections differ from their exact typed artifact."
            raise ValueError(msg)
    return ProductionResourceProjection(
        job_checksum=job.content_checksum,
        result_reference_checksum=custody.reference.content_checksum,
        resource_document_checksum=custody.resource_document_checksum,
        execution_source_manifest_checksum=custody.reference.execution_source_manifest_checksum,
        method_id=job.method_id,
        candidate_configuration_checksum=job.candidate_configuration_checksum,
        data_role=job.data_role,
        family_id=job.family_id,
        stratum_id=job.stratum_id,
        qubit_count=job.qubit_count,
        status=outcome.status,
        normalized_work=require_float(custody.resource_payload.get("normalized_work"), "normalized_work", minimum=0.0),
        structural_prefix_checksums=custody.production_evidence.structural_prefix_checksums,
        circuit_binding_checksum=circuit_checksum,
        compiled_resources_checksum=compiled_checksum,
        native_two_qubit_gates_per_chain_edge=counts,
    )


@dataclass(frozen=True, slots=True)
class PilotNormalizedComputeCalibration:
    """Pilot-only cap artifact available before paper-screen execution."""

    preregistration_checksum: str
    execution_source_manifest_checksum: str
    pilot_plan_checksum: str
    pilot_custody_checksum: str
    calculation_rule_id: Literal["maximum_successful_q6_pilot_normalized_work_v1"]
    normalized_compute_cap: float
    pilot_q6_resources: tuple[ProductionResourceProjection, ...]
    _content_checksum: str = field(init=False, repr=False, compare=False)
    schema_version: str = field(default=PILOT_NORMALIZED_COMPUTE_CALIBRATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the q6-only prospective screen-cap derivation.

        Raises:
            TypeError: If resource projections use the wrong typed schema.
            ValueError: If roots, source, universe, successes, rule, or cap differ.
        """
        for name in (
            "preregistration_checksum",
            "execution_source_manifest_checksum",
            "pilot_plan_checksum",
            "pilot_custody_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        resources = tuple(self.pilot_q6_resources)
        if len(resources) != PILOT_PRIMARY_JOB_COUNT or not all(
            isinstance(item, ProductionResourceProjection) for item in resources
        ):
            msg = "Pilot cap calibration requires exactly 720 typed q6 resource projections."
            raise TypeError(msg)
        if any(
            item.qubit_count != 6
            or item.data_role != "development"
            or item.execution_source_manifest_checksum != self.execution_source_manifest_checksum
            for item in resources
        ):
            msg = "Pilot cap calibration differs from its q6 role or execution source."
            raise ValueError(msg)
        if len({item.job_checksum for item in resources}) != len(resources):
            msg = "Pilot cap calibration cannot reuse an exact q6 job."
            raise ValueError(msg)
        if self.calculation_rule_id != "maximum_successful_q6_pilot_normalized_work_v1":
            msg = "Pilot cap calibration uses an unsupported prospective rule."
            raise ValueError(msg)
        successful = tuple(item for item in resources if item.status == "success")
        configured = {item.candidate_configuration_checksum for item in resources}
        successful_configured = {item.candidate_configuration_checksum for item in successful}
        if not successful or successful_configured != configured:
            msg = "Pilot cap calibration requires a successful q6 receipt for every configuration."
            raise ValueError(msg)
        cap = require_float(self.normalized_compute_cap, "normalized_compute_cap", minimum=0.0)
        if float(cap).hex() != float(max(item.normalized_work for item in successful)).hex():
            msg = "Pilot normalized compute cap must equal the maximum successful q6 receipt."
            raise ValueError(msg)
        object.__setattr__(self, "normalized_compute_cap", cap)
        object.__setattr__(self, "pilot_q6_resources", resources)
        object.__setattr__(self, "_content_checksum", canonical_checksum(self._content_dict()))

    @property
    def content_checksum(self) -> str:
        """Checksum the exact pilot custody root and q6 resource universe."""
        return self._content_checksum

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered pilot calibration field."""
        return {
            "schema_version": self.schema_version,
            "preregistration_checksum": self.preregistration_checksum,
            "execution_source_manifest_checksum": self.execution_source_manifest_checksum,
            "pilot_plan_checksum": self.pilot_plan_checksum,
            "pilot_custody_checksum": self.pilot_custody_checksum,
            "calculation_rule_id": self.calculation_rule_id,
            "normalized_compute_cap": self.normalized_compute_cap,
            "pilot_q6_resources": [item.to_dict() for item in self.pilot_q6_resources],
        }

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native pilot calibration evidence."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical pilot calibration JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> PilotNormalizedComputeCalibration:
        """Decode and verify a persisted pilot-only calibration.

        Returns:
            The strict pilot cap calibration.

        Raises:
            TypeError: If the q6 projection universe is not a serialized sequence.
            ValueError: If schema, derivation, or checksum differs.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_PILOT_NORMALIZED_COMPUTE_CALIBRATION_KEYS,
            name="pilot normalized compute calibration",
        )
        if mapping["schema_version"] != PILOT_NORMALIZED_COMPUTE_CALIBRATION_SCHEMA_VERSION:
            msg = "Pilot normalized compute calibration uses an unsupported schema version."
            raise ValueError(msg)
        raw_resources = mapping["pilot_q6_resources"]
        if not isinstance(raw_resources, tuple):
            msg = "Pilot q6 resource projections must be a serialized sequence."
            raise TypeError(msg)
        calibration = cls(
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            execution_source_manifest_checksum=cast("str", mapping["execution_source_manifest_checksum"]),
            pilot_plan_checksum=cast("str", mapping["pilot_plan_checksum"]),
            pilot_custody_checksum=cast("str", mapping["pilot_custody_checksum"]),
            calculation_rule_id=cast(
                "Literal['maximum_successful_q6_pilot_normalized_work_v1']",
                mapping["calculation_rule_id"],
            ),
            normalized_compute_cap=cast("float", mapping["normalized_compute_cap"]),
            pilot_q6_resources=tuple(ProductionResourceProjection.from_dict(item) for item in raw_resources),
        )
        if mapping["content_checksum"] != calibration.content_checksum:
            msg = "Pilot normalized compute calibration checksum changed during normalization."
            raise ValueError(msg)
        return calibration

    @classmethod
    def from_json(cls, payload: str) -> PilotNormalizedComputeCalibration:
        """Decode canonical pilot calibration JSON.

        Returns:
            The strict pilot cap calibration.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def build_pilot_normalized_compute_calibration(
    pilot_custody: ProductionPilotCustody,
) -> PilotNormalizedComputeCalibration:
    """Derive the prospective paper-screen cap from production q6 pilot custody.

    Returns:
        The persisted pilot-only normalized-compute calibration.

    Raises:
        TypeError: If pilot custody has the wrong runtime schema.
        ValueError: If pilot custody has no successful primary-q6 receipt.
    """
    if not isinstance(pilot_custody, ProductionPilotCustody):
        msg = "pilot_custody must be ProductionPilotCustody."
        raise TypeError(msg)
    context = pilot_custody.context
    records = tuple(item for item in pilot_custody.records if item.job.qubit_count == 6)
    resources = tuple(_production_resource_projection(item) for item in records)
    successful = tuple(item for item in resources if item.status == "success")
    if not successful:
        msg = "Pilot normalized-compute calibration requires successful q6 receipts."
        raise ValueError(msg)
    return PilotNormalizedComputeCalibration(
        preregistration_checksum=context.preregistration.content_checksum,
        execution_source_manifest_checksum=context.execution_source_manifest.content_checksum,
        pilot_plan_checksum=context.plan.content_checksum,
        pilot_custody_checksum=canonical_checksum({
            "q6_record_checksums": [item.content_checksum for item in records],
        }),
        calculation_rule_id="maximum_successful_q6_pilot_normalized_work_v1",
        normalized_compute_cap=max(item.normalized_work for item in successful),
        pilot_q6_resources=resources,
    )


@dataclass(frozen=True, slots=True)
class ProductionResourceCalibration(FinalResourceCalibrationManifest):
    """Typed q6-pilot calibration and complete paper-screen resource manifest."""

    preregistration_checksum: str
    execution_source_manifest_checksum: str
    pilot_plan_checksum: str
    pilot_custody_checksum: str
    pilot_calibration_checksum: str
    screening_plan_checksum: str
    screening_manifest_checksum: str
    screening_custody_checksum: str
    calculation_rule_id: Literal["maximum_successful_q6_pilot_normalized_work_v1"]
    normalized_compute_cap: float
    pilot_q6_resources: tuple[ProductionResourceProjection, ...]
    screening_resources: tuple[ProductionResourceProjection, ...]
    _content_checksum: str = field(init=False, repr=False, compare=False)
    schema_version: str = field(default=PRODUCTION_RESOURCE_CALIBRATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the exact q6-only calibration and full screen universe.

        Raises:
            TypeError: If resource projections use the wrong typed schema.
            ValueError: If counts, roots, sources, roles, work, or cap differ.
        """
        for name in (
            "preregistration_checksum",
            "execution_source_manifest_checksum",
            "pilot_plan_checksum",
            "pilot_custody_checksum",
            "pilot_calibration_checksum",
            "screening_plan_checksum",
            "screening_manifest_checksum",
            "screening_custody_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        pilot = tuple(self.pilot_q6_resources)
        screen = tuple(self.screening_resources)
        if not all(isinstance(item, ProductionResourceProjection) for item in (*pilot, *screen)):
            msg = "Resource calibration requires typed ProductionResourceProjection values."
            raise TypeError(msg)
        if len(pilot) != PILOT_PRIMARY_JOB_COUNT or len(screen) != 1_296:
            msg = "Resource calibration requires exactly 720 q6 pilot and 1,296 paper-screen projections."
            raise ValueError(msg)
        if any(item.qubit_count != 6 for item in pilot) or any(item.data_role != "development" for item in pilot):
            msg = "Pilot calibration must contain only the primary-q6 development universe."
            raise ValueError(msg)
        if any(item.data_role != "screening_selection" for item in screen):
            msg = "Screen resource calibration must contain only screening-selection records."
            raise ValueError(msg)
        if self.calculation_rule_id != "maximum_successful_q6_pilot_normalized_work_v1":
            msg = "Resource calibration uses an unsupported prospective cap rule."
            raise ValueError(msg)
        if any(
            item.execution_source_manifest_checksum != self.execution_source_manifest_checksum
            for item in (*pilot, *screen)
        ):
            msg = "Resource calibration projections use a foreign execution-source manifest."
            raise ValueError(msg)
        if len({item.job_checksum for item in pilot}) != len(pilot) or len({
            item.job_checksum for item in screen
        }) != len(screen):
            msg = "Resource calibration cannot reuse a pilot or screening job."
            raise ValueError(msg)
        cap = require_float(self.normalized_compute_cap, "normalized_compute_cap", minimum=0.0)
        successful_pilot = tuple(item for item in pilot if item.status == "success")
        configured = {item.candidate_configuration_checksum for item in pilot}
        successful_configured = {item.candidate_configuration_checksum for item in successful_pilot}
        if not successful_pilot or successful_configured != configured:
            msg = "Resource calibration requires a successful q6 pilot receipt for every piloted configuration."
            raise ValueError(msg)
        expected_cap = max(item.normalized_work for item in successful_pilot)
        if float(cap).hex() != float(expected_cap).hex():
            msg = "Normalized compute cap must be the maximum verified primary-q6 pilot work."
            raise ValueError(msg)
        if any(item.normalized_work > cap for item in screen):
            msg = "A paper-screen result exceeds the pilot-calibrated normalized compute cap."
            raise ValueError(msg)
        pilot_calibration = PilotNormalizedComputeCalibration(
            preregistration_checksum=self.preregistration_checksum,
            execution_source_manifest_checksum=self.execution_source_manifest_checksum,
            pilot_plan_checksum=self.pilot_plan_checksum,
            pilot_custody_checksum=self.pilot_custody_checksum,
            calculation_rule_id=self.calculation_rule_id,
            normalized_compute_cap=cap,
            pilot_q6_resources=pilot,
        )
        if pilot_calibration.content_checksum != self.pilot_calibration_checksum:
            msg = "Final resource calibration differs from its persisted pilot-only cap artifact."
            raise ValueError(msg)
        object.__setattr__(self, "normalized_compute_cap", cap)
        object.__setattr__(self, "pilot_q6_resources", pilot)
        object.__setattr__(self, "screening_resources", screen)
        object.__setattr__(self, "_content_checksum", canonical_checksum(self._content_dict()))

    @property
    def content_checksum(self) -> str:
        """Checksum all source, custody, compiler, and work projections."""
        return self._content_checksum

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered calibration field."""
        return {
            "schema_version": self.schema_version,
            "preregistration_checksum": self.preregistration_checksum,
            "execution_source_manifest_checksum": self.execution_source_manifest_checksum,
            "pilot_plan_checksum": self.pilot_plan_checksum,
            "pilot_custody_checksum": self.pilot_custody_checksum,
            "pilot_calibration_checksum": self.pilot_calibration_checksum,
            "screening_plan_checksum": self.screening_plan_checksum,
            "screening_manifest_checksum": self.screening_manifest_checksum,
            "screening_custody_checksum": self.screening_custody_checksum,
            "calculation_rule_id": self.calculation_rule_id,
            "normalized_compute_cap": self.normalized_compute_cap,
            "pilot_q6_resources": [item.to_dict() for item in self.pilot_q6_resources],
            "screening_resources": [item.to_dict() for item in self.screening_resources],
        }

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native calibration evidence."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed calibration JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ProductionResourceCalibration:
        """Decode and verify a persisted production resource calibration.

        Returns:
            The concrete typed calibration accepted by final authorization.

        Raises:
            TypeError: If a projection universe is not serialized as a sequence.
            ValueError: If schema, counts, derivation, or checksum differs.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_PRODUCTION_RESOURCE_CALIBRATION_KEYS,
            name="production resource calibration",
        )
        if mapping["schema_version"] != PRODUCTION_RESOURCE_CALIBRATION_SCHEMA_VERSION:
            msg = "Production resource calibration uses an unsupported schema version."
            raise ValueError(msg)
        raw_pilot = mapping["pilot_q6_resources"]
        raw_screen = mapping["screening_resources"]
        if not isinstance(raw_pilot, tuple) or not isinstance(raw_screen, tuple):
            msg = "Production resource calibration projections must be serialized sequences."
            raise TypeError(msg)
        calibration = cls(
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            execution_source_manifest_checksum=cast("str", mapping["execution_source_manifest_checksum"]),
            pilot_plan_checksum=cast("str", mapping["pilot_plan_checksum"]),
            pilot_custody_checksum=cast("str", mapping["pilot_custody_checksum"]),
            pilot_calibration_checksum=cast("str", mapping["pilot_calibration_checksum"]),
            screening_plan_checksum=cast("str", mapping["screening_plan_checksum"]),
            screening_manifest_checksum=cast("str", mapping["screening_manifest_checksum"]),
            screening_custody_checksum=cast("str", mapping["screening_custody_checksum"]),
            calculation_rule_id=cast(
                "Literal['maximum_successful_q6_pilot_normalized_work_v1']",
                mapping["calculation_rule_id"],
            ),
            normalized_compute_cap=cast("float", mapping["normalized_compute_cap"]),
            pilot_q6_resources=tuple(ProductionResourceProjection.from_dict(item) for item in raw_pilot),
            screening_resources=tuple(ProductionResourceProjection.from_dict(item) for item in raw_screen),
        )
        if mapping["content_checksum"] != calibration.content_checksum:
            msg = "Production resource calibration checksum changed during normalization."
            raise ValueError(msg)
        return calibration

    @classmethod
    def from_json(cls, payload: str) -> ProductionResourceCalibration:
        """Decode canonical persisted calibration JSON.

        Returns:
            The concrete strict production resource calibration.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def build_production_resource_calibration(
    pilot_custody: ProductionPilotCustody,
    screening_custody: ProductionScreeningCustody,
) -> ProductionResourceCalibration:
    """Derive the final resource calibration from exact production custody.

    Returns:
        The q12-invariant pilot calibration and full screening resource manifest.

    Raises:
        TypeError: If custody inputs have the wrong typed schemas.
        ValueError: If source roots, screen caps, or resource projections differ.
    """
    if not isinstance(pilot_custody, ProductionPilotCustody) or not isinstance(
        screening_custody,
        ProductionScreeningCustody,
    ):
        msg = "Resource calibration requires production pilot and screening custody."
        raise TypeError(msg)
    pilot_context = pilot_custody.context
    screen_context = screening_custody.context
    source_checksum = pilot_context.execution_source_manifest.content_checksum
    if screen_context.preregistration.content_checksum != pilot_context.preregistration.content_checksum:
        msg = "Pilot and screening calibration custody use different preregistrations."
        raise ValueError(msg)
    if screen_context.execution_source_manifest.content_checksum != source_checksum:
        msg = "Pilot and screening calibration custody use different execution sources."
        raise ValueError(msg)
    pilot_records = tuple(item for item in pilot_custody.records if item.job.qubit_count == 6)
    pilot_calibration = build_pilot_normalized_compute_calibration(pilot_custody)
    pilot_resources = pilot_calibration.pilot_q6_resources
    screen_resources = tuple(_production_resource_projection(item) for item in screening_custody.records)
    derived_cap = pilot_calibration.normalized_compute_cap
    manifest = cast("ScreeningManifest", screen_context.screening_manifest)
    candidate_checksums = {item.configuration_checksum for item in manifest.candidates}
    bindings = tuple(
        link
        for link in screen_context.scoped_bindings
        if link.binding.publication_candidate_checksum in candidate_checksums
    )
    caps = {link.binding.execution_budget.normalized_compute_cap for link in bindings}
    if len(bindings) != len(candidate_checksums) or caps != {derived_cap}:
        msg = "Every paper-screen binding must use the unique pilot-calibrated normalized compute cap."
        raise ValueError(msg)
    return ProductionResourceCalibration(
        preregistration_checksum=pilot_context.preregistration.content_checksum,
        execution_source_manifest_checksum=source_checksum,
        pilot_plan_checksum=pilot_context.plan.content_checksum,
        pilot_custody_checksum=canonical_checksum({
            "q6_record_checksums": [item.content_checksum for item in pilot_records],
        }),
        pilot_calibration_checksum=pilot_calibration.content_checksum,
        screening_plan_checksum=screen_context.plan.content_checksum,
        screening_manifest_checksum=manifest.content_checksum,
        screening_custody_checksum=canonical_checksum({
            "record_checksums": [item.content_checksum for item in screening_custody.records],
        }),
        calculation_rule_id="maximum_successful_q6_pilot_normalized_work_v1",
        normalized_compute_cap=derived_cap,
        pilot_q6_resources=pilot_resources,
        screening_resources=screen_resources,
    )


def build_production_screening_evidence_from_records(
    preregistration: InitialPreregistration,
    manifest: ScreeningManifest,
    records: Sequence[ProductionScreeningSourceRecord],
    *,
    evidence_id: str,
) -> tuple[ScreeningEvidence, PromotionDecision]:
    """Build promotion evidence from a complete reopened production universe.

    Returns:
        The raw evidence projection and mechanical promotion decision.

    Raises:
        TypeError: If records do not use the production source schema.
        ValueError: If records are missing, duplicated, foreign, or reused.
    """
    values = tuple(records)
    if not values or not all(isinstance(item, ProductionScreeningSourceRecord) for item in values):
        msg = "Production promotion requires manifest-reopened ProductionScreeningSourceRecord values."
        raise TypeError(msg)
    by_pair = {(item.candidate.configuration_checksum, item.cell.cell_id): item for item in values}
    expected = {
        (candidate.configuration_checksum, cell.cell_id) for candidate in manifest.candidates for cell in manifest.cells
    }
    if len(values) != 1296 or len(by_pair) != len(values) or set(by_pair) != expected:
        msg = "Production screening sources must cover the exact 1,296-cell Cartesian universe."
        raise ValueError(msg)
    if len({item.result_custody.reference.content_checksum for item in values}) != len(values):
        msg = "Every screening cell must dereference a distinct immutable result reference."
        raise ValueError(msg)
    observations = tuple(
        by_pair[candidate.configuration_checksum, cell.cell_id].promotion_observation()
        for candidate in manifest.candidates
        for cell in manifest.cells
    )
    evidence = ScreeningEvidence(
        evidence_id=require_slug(evidence_id, "evidence_id"),
        preregistration_checksum=preregistration.content_checksum,
        screening_manifest_checksum=manifest.content_checksum,
        observations=observations,
    )
    return evidence, select_promoted_candidate(preregistration, manifest, evidence)


def build_production_screening_evidence(
    context: TrainingExecutionContext,
    output_root: Path,
    *,
    evidence_id: str = "wp22_paper_screen_evidence_v1",
) -> tuple[ScreeningEvidence, PromotionDecision, tuple[ProductionScreeningSourceRecord, ...]]:
    """Replay the complete 1,296-cell screen from immutable production attempts.

    Returns:
        Screening evidence, the mechanical decision, and source audit records.

    """
    custody = ProductionScreeningCustody.reopen(context, output_root)
    evidence, decision = custody.build_evidence(evidence_id=evidence_id)
    return evidence, decision, custody.records


def build_screening_evidence(
    preregistration: InitialPreregistration,
    manifest: ScreeningManifest,
    source_records: Sequence[ScreeningSourceRecord],
    *,
    evidence_id: str = "wp22_paper_screen_evidence_v1",
) -> tuple[ScreeningEvidence, PromotionDecision]:
    """Build complete source-addressed evidence and apply the frozen rule.

    Returns:
        The complete screening evidence and its mechanical promotion decision.

    Raises:
        TypeError: If inputs have the wrong artifact types.
        ValueError: If sources duplicate, omit, add, or alter sealed cells.
    """
    if not isinstance(preregistration, InitialPreregistration):
        msg = "preregistration must be an InitialPreregistration."
        raise TypeError(msg)
    if not isinstance(manifest, ScreeningManifest):
        msg = "manifest must be a ScreeningManifest."
        raise TypeError(msg)
    values = tuple(source_records)
    if not values or not all(isinstance(item, ScreeningSourceRecord) for item in values):
        msg = "source_records must contain authoritative ScreeningSourceRecord values."
        raise TypeError(msg)
    by_pair = {(item.candidate_configuration_checksum, item.cell_id): item for item in values}
    if len(by_pair) != len(values):
        msg = "Screening source records must not duplicate candidate/cell pairs."
        raise ValueError(msg)
    if len({item.content_checksum for item in values}) != len(values):
        msg = "Screening source records must have unique authoritative identities."
        raise ValueError(msg)
    if len({item.result_record_checksum for item in values}) != len(values):
        msg = "Screening source records must reference distinct result records."
        raise ValueError(msg)
    successful_evidence = tuple(
        cast("TrajectoryFidelityEvidence", item.trajectory_evidence).content_checksum
        for item in values
        if item.status == "success"
    )
    if len(set(successful_evidence)) != len(successful_evidence):
        msg = "Successful screening source records must contain distinct raw trajectory evidence."
        raise ValueError(msg)
    expected = {
        (candidate.configuration_checksum, cell.cell_id) for candidate in manifest.candidates for cell in manifest.cells
    }
    if set(by_pair) != expected:
        missing = sorted(expected - set(by_pair))
        extra = sorted(set(by_pair) - expected)
        msg = f"Screening outcomes do not cover the sealed Cartesian universe: missing={missing!r}, extra={extra!r}."
        raise ValueError(msg)
    cell_by_id = {cell.cell_id: cell for cell in manifest.cells}
    observations: list[PromotionObservation] = []
    for candidate in manifest.candidates:
        for cell in manifest.cells:
            source = by_pair[candidate.configuration_checksum, cell.cell_id]
            if source.evaluation_seed != cell.screening_seed or source.data_role != "screening_selection":
                msg = "Screening outcome uses a changed outer cell seed or data role."
                raise ValueError(msg)
            if cell_by_id[source.cell_id] != cell:
                msg = "Screening outcome cell identity changed."
                raise ValueError(msg)
            observations.append(source.verified_outcome().promotion_observation())
    evidence = ScreeningEvidence(
        evidence_id=require_slug(evidence_id, "evidence_id"),
        preregistration_checksum=preregistration.content_checksum,
        screening_manifest_checksum=manifest.content_checksum,
        observations=tuple(observations),
    )
    return evidence, select_promoted_candidate(preregistration, manifest, evidence)


def _candidate_by_method(manifest: ScreeningManifest, method_id: str) -> ScreeningCandidateRef:
    """Return the unique screened candidate for one method.

    Raises:
        ValueError: If the method does not identify exactly one candidate.
    """
    matches = tuple(candidate for candidate in manifest.candidates if candidate.method_id == method_id)
    if len(matches) != 1:
        msg = f"Screening manifest must contain exactly one {method_id!r} candidate."
        raise ValueError(msg)
    return matches[0]


def _final_comparators(
    manifest: ScreeningManifest,
    decision: PromotionDecision,
) -> tuple[FinalComparatorRef, ...]:
    """Build the exact de-duplicated primary comparator set.

    Returns:
        The comparator set after removing any promoted-baseline self-comparison.
    """
    baseline = _candidate_by_method(manifest, "layerwise_bmpd_crn_v2")
    noiseless = _candidate_by_method(manifest, "layerwise_bmpd_noiseless")
    comparators: list[FinalComparatorRef] = []
    if decision.promoted_configuration_checksum != baseline.configuration_checksum:
        comparators.append(
            FinalComparatorRef(
                role="layerwise_v2_reference",
                method_id=baseline.method_id,
                configuration_schema_version=baseline.configuration_schema_version,
                configuration_checksum=baseline.configuration_checksum,
                matched_to_configuration_checksum=noiseless.configuration_checksum,
                matching_projection_checksum=baseline.matching_projection_checksum,
            )
        )
    comparators.append(
        FinalComparatorRef(
            role="matched_noiseless_control",
            method_id=noiseless.method_id,
            configuration_schema_version=noiseless.configuration_schema_version,
            configuration_checksum=noiseless.configuration_checksum,
            matched_to_configuration_checksum=baseline.configuration_checksum,
            matching_projection_checksum=noiseless.matching_projection_checksum,
        )
    )
    return tuple(comparators)


def _primary_contrasts(
    preregistration: InitialPreregistration,
    manifest: ScreeningManifest,
    decision: PromotionDecision,
) -> tuple[PrimaryContrastBinding, ...]:
    """Bind every applicable primary contrast to exact screened configurations.

    Returns:
        The de-duplicated primary contrast bindings.
    """
    baseline = _candidate_by_method(manifest, "layerwise_bmpd_crn_v2")
    noiseless = _candidate_by_method(manifest, "layerwise_bmpd_noiseless")
    contrasts = [
        PrimaryContrastBinding(
            contrast_id="noisy_vs_noiseless",
            treatment_configuration_checksum=baseline.configuration_checksum,
            control_configuration_checksum=noiseless.configuration_checksum,
            paired_block_policy_checksum=preregistration.paired_block_policy_checksum,
            matching_projection_checksum=baseline.matching_projection_checksum,
        )
    ]
    if decision.promoted_configuration_checksum != baseline.configuration_checksum:
        contrasts.append(
            PrimaryContrastBinding(
                contrast_id="promoted_vs_layerwise_v2_if_distinct",
                treatment_configuration_checksum=decision.promoted_configuration_checksum,
                control_configuration_checksum=baseline.configuration_checksum,
                paired_block_policy_checksum=preregistration.paired_block_policy_checksum,
                matching_projection_checksum=None,
            )
        )
    return tuple(contrasts)


def build_final_configuration_execution_manifest(
    screening_custody: ProductionScreeningCustody,
    decision: PromotionDecision,
    *,
    manifest_id: str = "phase2_final_configuration_execution_v1",
) -> FinalConfigurationExecutionManifest:
    """Bind every final configuration to its exact screened executable schedule.

    Returns:
        The canonical promoted-plus-comparator execution manifest.

    Raises:
        TypeError: If custody or decision has the wrong typed schema.
        ValueError: If a final configuration lacks one exact context binding or its jobs differ.
    """
    if not isinstance(screening_custody, ProductionScreeningCustody):
        msg = "screening_custody must be ProductionScreeningCustody."
        raise TypeError(msg)
    if not isinstance(decision, PromotionDecision):
        msg = "decision must be a PromotionDecision."
        raise TypeError(msg)
    context = screening_custody.context
    manifest = cast("ScreeningManifest", context.screening_manifest)
    candidates = {item.configuration_checksum: item for item in manifest.candidates}
    final_checksums = {
        decision.promoted_configuration_checksum,
        *(item.configuration_checksum for item in _final_comparators(manifest, decision)),
    }
    entries: list[FinalConfigurationExecutionRef] = []
    for configuration_checksum in final_checksums:
        candidate = candidates.get(configuration_checksum)
        if candidate is None:
            msg = "A final configuration is absent from the exact screening manifest."
            raise ValueError(msg)
        links = tuple(
            item
            for item in context.scoped_bindings
            if item.binding.publication_candidate_checksum == configuration_checksum
        )
        if len(links) != 1:
            msg = "Each final configuration requires one exact context-owned executable binding."
            raise ValueError(msg)
        link = links[0]
        binding = link.binding
        if binding.publication_method_id != candidate.method_id:
            msg = "Final execution binding method differs from its screened candidate."
            raise ValueError(msg)
        jobs = tuple(item.job for item in screening_custody.records if item.candidate is candidate)
        expected_job_fields = (
            ("method_id", candidate.method_id),
            ("implementation_checksum", binding.implementation_checksum),
            ("strategy_schedule_checksum", binding.strategy_schedule.content_checksum),
            ("scoped_binding_checksum", binding.content_checksum),
            ("executable_binding_checksum", link.content_checksum),
        )
        if not jobs or any(
            getattr(job, field_name) != expected for job in jobs for field_name, expected in expected_job_fields
        ):
            msg = "Final execution reference differs from its exact paper-screen job closure."
            raise ValueError(msg)
        entries.append(
            FinalConfigurationExecutionRef(
                method_id=candidate.method_id,
                configuration_schema_version=candidate.configuration_schema_version,
                configuration_checksum=candidate.configuration_checksum,
                strategy_schedule=binding.strategy_schedule,
                implementation_checksum=binding.implementation_checksum,
                scoped_binding_checksum=binding.content_checksum,
                executable_binding_checksum=link.content_checksum,
            )
        )
    return FinalConfigurationExecutionManifest(
        manifest_id=require_slug(manifest_id, "manifest_id"),
        entries=tuple(sorted(entries, key=lambda item: (item.configuration_checksum, item.method_id))),
    )


def create_final_confirmation_seal(
    *,
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    promotion_decision: PromotionDecision,
    pilot_nuisance_summary: PilotNuisanceSummary,
    sample_size_design: SampleSizeDesign,
    confirmatory_target_commitment: TargetPopulationCommitment,
    analysis_source_manifest: AnalysisSourceManifest,
    execution_source_manifest: ExecutionSourceManifest,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
    repository_root: Path,
    production_screening_custody: ProductionScreeningCustody | None = None,
    production_pilot_custody: ProductionPilotCustody | None = None,
    parent_sample_size_design: SampleSizeDesign | None = None,
    seal_id: str = "phase2_confirmation_v1",
) -> FinalConfirmationSeal:
    """Create and fully cross-verify the immutable final confirmation seal.

    ``confirmatory_target_commitment`` intentionally exposes only a checksum
    and family counts.  This function has no API through which target seeds,
    instance identifiers, parameters, or vectors can be supplied.

    Raw WP15 screening projections are not accepted in production: the complete
    context-owned raw-trajectory-backed WP22 custody is projected internally. The
    nuisance summary is likewise rebuilt from the complete production pilot
    custody; only its q6 records enter inference, while q12 result bytes remain
    an external scaling archive. The sample-size design must byte-match a fresh
    run of the frozen pilot calculator over that reproduced summary. The one
    preregistered halfway re-estimation additionally requires its initial
    parent design. Bare, legacy, synthetic, or mixed source rows are never an
    authorizing input to this factory.

    Returns:
        The authorized immutable final confirmation seal.

    Raises:
        TypeError: If any supplied artifact has the wrong type.
        ValueError: If promotion, pilot derivation, counts, sources, or seal cross-links disagree.
    """
    if not isinstance(preregistration, InitialPreregistration):
        msg = "preregistration must be an InitialPreregistration."
        raise TypeError(msg)
    if not isinstance(screening_manifest, ScreeningManifest):
        msg = "screening_manifest must be a ScreeningManifest."
        raise TypeError(msg)
    if not isinstance(promotion_decision, PromotionDecision):
        msg = "promotion_decision must be a PromotionDecision."
        raise TypeError(msg)
    if not isinstance(pilot_nuisance_summary, PilotNuisanceSummary):
        msg = "pilot_nuisance_summary must be a PilotNuisanceSummary."
        raise TypeError(msg)
    if not isinstance(sample_size_design, SampleSizeDesign):
        msg = "sample_size_design must be a SampleSizeDesign."
        raise TypeError(msg)
    if parent_sample_size_design is not None and not isinstance(parent_sample_size_design, SampleSizeDesign):
        msg = "parent_sample_size_design must be a SampleSizeDesign or None."
        raise TypeError(msg)
    if not isinstance(confirmatory_target_commitment, TargetPopulationCommitment):
        msg = "confirmatory_target_commitment must be a checksum-only TargetPopulationCommitment."
        raise TypeError(msg)
    if not isinstance(analysis_source_manifest, AnalysisSourceManifest):
        msg = "analysis_source_manifest must be an AnalysisSourceManifest."
        raise TypeError(msg)
    if not isinstance(execution_source_manifest, ExecutionSourceManifest):
        msg = "execution_source_manifest must be an ExecutionSourceManifest."
        raise TypeError(msg)
    if not isinstance(configuration_execution_manifest, FinalConfigurationExecutionManifest):
        msg = "configuration_execution_manifest must be a FinalConfigurationExecutionManifest."
        raise TypeError(msg)
    if not isinstance(repository_root, Path):
        msg = "repository_root must be a pathlib.Path."
        raise TypeError(msg)
    if production_screening_custody is not None and not isinstance(
        production_screening_custody,
        ProductionScreeningCustody,
    ):
        msg = "production_screening_custody must be ProductionScreeningCustody or None."
        raise TypeError(msg)
    if production_pilot_custody is None:
        msg = "Final sealing requires manifest-reopened production pilot custody."
        raise TypeError(msg)
    if not isinstance(production_pilot_custody, ProductionPilotCustody):
        msg = "production_pilot_custody must be ProductionPilotCustody or None."
        raise TypeError(msg)
    if production_screening_custody is None:
        msg = "Final sealing requires context-owned production screening custody."
        raise TypeError(msg)
    screening_context = production_screening_custody.context
    source_checksum = execution_source_manifest.content_checksum
    if analysis_source_manifest.execution_source_manifest_checksum != source_checksum:
        msg = "Analysis source is not bound to the supplied final execution-source manifest."
        raise ValueError(msg)
    custody_sources = (
        production_pilot_custody.context.execution_source_manifest,
        screening_context.execution_source_manifest,
    )
    if any(
        not isinstance(source, ExecutionSourceManifest) or source.content_checksum != source_checksum
        for source in custody_sources
    ):
        msg = "Production pilot or screening custody uses a foreign execution-source manifest."
        raise ValueError(msg)
    if production_pilot_custody.context.preregistration.content_checksum != preregistration.content_checksum:
        msg = "Production pilot custody differs from the final-seal preregistration."
        raise ValueError(msg)
    if (
        screening_context.preregistration.content_checksum != preregistration.content_checksum
        or screening_context.screening_manifest is not screening_manifest
        or screening_context.required_sample_size_design != sample_size_design
    ):
        msg = "Production screening custody differs from the final-seal protocol, manifest, or design."
        raise ValueError(msg)
    if sample_size_design.reestimation_kind == "initial" and parent_sample_size_design is not None:
        msg = "An initial sample-size design cannot have a parent design at final-seal creation."
        raise ValueError(msg)
    if sample_size_design.reestimation_kind != "initial" and parent_sample_size_design is None:
        msg = "A blinded nuisance-only sample-size design requires its initial parent design."
        raise ValueError(msg)
    reproduced_execution_manifest = build_final_configuration_execution_manifest(
        production_screening_custody,
        promotion_decision,
        manifest_id=configuration_execution_manifest.manifest_id,
    )
    if reproduced_execution_manifest.to_json() != configuration_execution_manifest.to_json():
        msg = "Final configuration execution manifest differs from exact screened bindings and schedules."
        raise ValueError(msg)
    reproduced_pilot = production_pilot_custody.build_nuisance_summary(
        pilot_nuisance_summary.contrast_bindings,
        summary_id=pilot_nuisance_summary.summary_id,
    )
    if reproduced_pilot.to_json().encode("utf-8") != pilot_nuisance_summary.to_json().encode("utf-8"):
        msg = "Pilot nuisance summary is not the exact q6 inference projection of production custody."
        raise ValueError(msg)
    screening_evidence, recomputed_promotion = production_screening_custody.build_evidence(
        evidence_id="wp22_paper_screen_evidence_v1",
    )
    if recomputed_promotion.content_checksum != promotion_decision.content_checksum:
        msg = "Promotion decision is not the exact mechanical result of the supplied raw screening evidence."
        raise ValueError(msg)
    if sample_size_design.reestimation_kind == "initial":
        recomputed_design = build_cluster_aware_paired_difference_v1(
            preregistration,
            pilot_nuisance_summary,
            design_id=sample_size_design.design_id,
            calculation_source_checksum=PILOT_CALCULATION_SOURCE_CHECKSUM,
        )
    else:
        parent_design = cast("SampleSizeDesign", parent_sample_size_design)
        recomputed_design = reestimate_cluster_aware_paired_difference_v1(
            preregistration,
            pilot_nuisance_summary,
            parent_design,
            information_fraction=0.5,
            design_id=sample_size_design.design_id,
            calculation_source_checksum=PILOT_CALCULATION_SOURCE_CHECKSUM,
        )
    if recomputed_design.to_json().encode("utf-8") != sample_size_design.to_json().encode("utf-8"):
        msg = "Sample-size design is not the exact result of the supplied pilot nuisance evidence and frozen source."
        raise ValueError(msg)
    if confirmatory_target_commitment.target_count_by_family != sample_size_design.target_count_by_family:
        msg = "Confirmatory target commitment counts differ from the pilot-derived sample-size design."
        raise ValueError(msg)
    for record in production_pilot_custody.records:
        validate_production_job_custody(
            record.result_custody,
            record.job,
            record.outcome,
            expected_data_role=record.job.data_role,
            expected_trajectory_count=(
                PILOT_PRIMARY_TRAJECTORY_COUNT if record.job.qubit_count == 6 else PILOT_SECONDARY_TRAJECTORY_COUNT
            ),
            expected_execution_source_manifest_checksum=source_checksum,
            allowed_artifact_kinds=(
                "operator_growth" if record.job.implementation_kind == "operator_growth" else "pipeline",
            ),
        )
    for record in production_screening_custody.records:
        validate_production_job_custody(
            record.result_custody,
            record.job,
            record.outcome,
            expected_data_role="screening_selection",
            expected_trajectory_count=sample_size_design.fixed_test_trajectory_count,
            expected_execution_source_manifest_checksum=source_checksum,
            allowed_artifact_kinds=(
                "operator_growth" if record.job.implementation_kind == "operator_growth" else "pipeline",
            ),
        )
    resource_calibration = build_production_resource_calibration(
        production_pilot_custody,
        production_screening_custody,
    )
    verify_analysis_source_bridge(execution_source_manifest, analysis_source_manifest, repository_root)
    resource = preregistration.primary_resource_constraint
    seal = FinalConfirmationSeal(
        seal_id=require_slug(seal_id, "seal_id"),
        preregistration_checksum=preregistration.content_checksum,
        promotion_decision_checksum=promotion_decision.content_checksum,
        promoted_method_id=promotion_decision.promoted_method_id,
        promoted_configuration_checksum=promotion_decision.promoted_configuration_checksum,
        comparators=_final_comparators(screening_manifest, promotion_decision),
        primary_contrasts=_primary_contrasts(preregistration, screening_manifest, promotion_decision),
        confirmatory_target_manifest_checksum=confirmatory_target_commitment.target_manifest_checksum,
        target_count_by_family=sample_size_design.target_count_by_family,
        optimization_seed_count=sample_size_design.optimization_seed_count,
        fixed_test_trajectory_count=sample_size_design.fixed_test_trajectory_count,
        primary_noise_condition=preregistration.primary_noise_condition,
        primary_resource_budget={
            "metric": resource["metric"],
            "cap_per_chain_edge": resource["cap_per_chain_edge"],
            "normalized_compute_cap": resource_calibration.normalized_compute_cap,
            "reachable_stratum_manifest_checksum": resource_calibration.content_checksum,
        },
        hyperparameters_checksum=configuration_execution_manifest.content_checksum,
        execution_source_checksum=source_checksum,
        analysis_template_checksum=preregistration.analysis_template_checksum,
        analysis_source_manifest_checksum=analysis_source_manifest.content_checksum,
        sample_size_design_checksum=sample_size_design.content_checksum,
        failure_policy_checksum=preregistration.failure_policy_checksum,
    )
    authorize_confirmation(
        preregistration,
        screening_manifest,
        screening_evidence,
        promotion_decision,
        sample_size_design,
        analysis_source_manifest,
        seal,
        configuration_execution_manifest,
        resource_calibration,
        repository_root,
    )
    return seal


__all__ = [
    "ADAPT_STYLE_PUBLICATION_METHOD_ID",
    "IMPACT_PRUNING_PUBLICATION_METHOD_ID",
    "PILOT_NORMALIZED_COMPUTE_CALIBRATION_SCHEMA_VERSION",
    "PRODUCTION_RESOURCE_CALIBRATION_SCHEMA_VERSION",
    "PRODUCTION_RESOURCE_PROJECTION_SCHEMA_VERSION",
    "WP18_SCREENING_SOURCE_ARTIFACT_SCHEMA_VERSION",
    "WP22_CANDIDATE_CONFIGURATION_SCHEMA_VERSION",
    "WP22_OPERATOR_GROWTH_TEMPLATE_SCHEMA_VERSION",
    "WP22_PRODUCTION_SCREENING_SOURCE_SCHEMA_VERSION",
    "WP22_PUBLICATION_PRUNING_MAPPING_VERSION",
    "WP22_SCREENING_OUTCOME_SCHEMA_VERSION",
    "WP22_SCREENING_SOURCE_RECORD_SCHEMA_VERSION",
    "OperatorGrowthScreeningTemplate",
    "PilotNormalizedComputeCalibration",
    "ProductionResourceCalibration",
    "ProductionResourceProjection",
    "ProductionScreeningCustody",
    "ProductionScreeningSourceRecord",
    "ScreeningSourceRecord",
    "VerifiedScreeningOutcome",
    "WP18ScreeningSourceArtifact",
    "WP22CandidateConfiguration",
    "build_final_configuration_execution_manifest",
    "build_pilot_normalized_compute_calibration",
    "build_production_resource_calibration",
    "build_production_screening_evidence",
    "build_production_screening_evidence_from_records",
    "build_screening_evidence",
    "build_screening_manifest",
    "create_final_confirmation_seal",
    "screening_trajectory_context_checksum",
]
