# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Typed WP18 source fixtures for WP22 screening-custody tests."""

from __future__ import annotations

import hashlib
import statistics
from copy import copy
from dataclasses import replace
from functools import cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np

from benchmarks.state_preparation.phase2.artifact_codecs import (
    artifact_checksum,
    create_phase2_trajectory_sidecar,
)
from benchmarks.state_preparation.phase2.artifacts import (
    EvaluationEvidenceArtifact,
    FixedMapArtifactRef,
    MaterializedCircuitArtifact,
)
from benchmarks.state_preparation.phase2.canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
)
from benchmarks.state_preparation.phase2.execution_context import TrainingExecutionContext
from benchmarks.state_preparation.phase2.pipeline import (
    PipelineBenchmarkFailure,
    PipelineBenchmarkResult,
    TrainingPipelineTemplate,
)
from benchmarks.state_preparation.phase2.production_executors import (
    ProductionAttemptStore,
    ProductionNumericalEvidence,
)
from benchmarks.state_preparation.phase2.pruning import TOPDOWN_IMPACT_ITERATIVE_METHOD_ID
from benchmarks.state_preparation.phase2.result_custody import ProductionResultCustody
from benchmarks.state_preparation.phase2.screening import (
    IMPACT_PRUNING_PUBLICATION_METHOD_ID,
    WP22_PUBLICATION_PRUNING_MAPPING_VERSION,
    ProductionScreeningCustody,
    ProductionScreeningSourceRecord,
    ScreeningSourceRecord,
    WP22CandidateConfiguration,
)
from benchmarks.state_preparation.phase2.topdown_pruning import build_topdown_impact_iterative_template
from benchmarks.state_preparation.phase2.training_orchestration import (
    TrainingJob,
    TrainingJobOutcome,
    training_job_attempt_path,
)
from benchmarks.state_preparation.phase2.training_schedules import (
    CheckpointValidationPolicy,
    LimitedMultistartPlan,
    NoiselessPretrainNoisyFinetune,
    NoiseMixtureComponent,
    NoiseStrengthContinuation,
    StandardNoiseMixture,
    TrainingStrategySchedule,
    TrajectoryCountCurriculum,
    TrajectoryCountStep,
    TrajectorySamplingPolicy,
)
from benchmarks.state_preparation.phase2.wp20_resources import (
    CircuitResourceMetrics,
    LogicalEventSignature,
    WP20WorkLedger,
)
from mqt.yaqs.optimization import KrotovFixedMapEnsemble, KrotovNoiseMap
from tests.benchmarks.test_state_preparation_phase2_pipeline import (
    _evaluation,
    _pipeline_result,
    _template,
)

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.state_preparation.phase2.protocol import (
        InitialPreregistration,
        SampleSizeDesign,
        ScreeningCandidateRef,
        ScreeningCell,
        ScreeningManifest,
    )
    from benchmarks.state_preparation.phase2.source_lock import ExecutionSourceManifest
    from benchmarks.state_preparation.phase2.targets import TargetPopulationManifest


def _checksum(label: str) -> str:
    """Return one deterministic prefixed SHA-256 checksum."""
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


def _set_frozen_slot(instance: object, name: str, value: object) -> None:
    """Set one slot while constructing a deliberately bounded typed test seam."""
    object.__setattr__(instance, name, value)  # noqa: PLC2801 -- frozen init bypass for test-only seams


class _BoundedScheduledExecutionSnapshot:
    """Minimal typed schedule snapshot used only by the manifest-backed F seam."""

    schema_version = "yaqs.state_preparation.phase2.wp22f_bounded_schedule_snapshot.v1"

    def __init__(self, program_checksum: str) -> None:
        self.program_checksum = program_checksum
        self.states = ()
        self.multistart_evidence = SimpleNamespace(
            selected_start_index=0,
            selected_update=0,
            selected_parameter_checksum=canonical_checksum({
                "bounded fixture selected parameters": program_checksum,
            }),
            selected_parameter_artifact=SimpleNamespace(parameters=()),
            total_normalized_work=0.0,
        )

    @property
    def content_checksum(self) -> str:
        """Checksum the bounded snapshot's exact scheduled program identity."""
        return canonical_checksum({
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
        })

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json({
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "content_checksum": self.content_checksum,
        })

    @classmethod
    def from_json(cls, payload: str) -> _BoundedScheduledExecutionSnapshot:
        """Decode the narrow bounded snapshot used by the test seam.

        Returns:
            The verified bounded snapshot.

        Raises:
            ValueError: If its schema or checksum changed.
        """
        mapping = load_canonical_json_object(payload)
        snapshot = cls(cast("str", mapping["program_checksum"]))
        if (
            mapping.get("schema_version") != cls.schema_version
            or mapping.get("content_checksum") != snapshot.content_checksum
        ):
            msg = "Bounded schedule snapshot schema or checksum changed."
            raise ValueError(msg)
        return snapshot


def _candidate_template(method_id: str, *, noisy: bool) -> TrainingPipelineTemplate:
    """Return a small typed pipeline template for a synthetic publication method."""
    if method_id == IMPACT_PRUNING_PUBLICATION_METHOD_ID:
        return build_topdown_impact_iterative_template(
            deep_depth=1,
            pretrain_iterations=1,
            relaxation_iterations=1,
            fine_tune_mode="fixed_crn",
            fine_tune_iterations=1,
            fine_tune_trajectory_count=1,
            resource_stratum_id="primary_cap_12",
        )
    implementation = method_id
    return _template(
        noisy=noisy,
        method_id=implementation,
        template_id=f"wp22_fixture_{implementation}",
    )


@cache
def _production_strategy_schedule() -> TrainingStrategySchedule:
    """Return one small valid schedule for production-custody fixture jobs."""
    return TrainingStrategySchedule(
        schedule_id="wp22_production_screening_fixture",
        noise_continuation=NoiseStrengthContinuation(
            start_update=0,
            end_update=0,
            start_strength_scale=1.0,
            target_strength_scale=1.0,
            interpolation="constant",
        ),
        trajectory_curriculum=TrajectoryCountCurriculum((TrajectoryCountStep(0, 2),)),
        sampling_policy=TrajectorySamplingPolicy("fixed_crn"),
        checkpoint_validation=CheckpointValidationPolicy(patience=1, min_delta=0.0),
        phase_boundary=NoiselessPretrainNoisyFinetune(
            noiseless_pretrain_updates=0,
            noisy_finetune_updates=1,
        ),
        multistart=LimitedMultistartPlan(start_count=1, declared_cap=3),
        training_noise=StandardNoiseMixture(
            "matched",
            (NoiseMixtureComponent("depolarizing_1s_all", 1.0),),
        ),
    )


def _production_document(document_type: str, payload: dict[str, object]) -> dict[str, object]:
    """Return one WP22E-compatible checksum-sealed typed document."""
    content = {
        "schema_version": "yaqs.state_preparation.phase2.production_document.v1",
        "document_type": document_type,
        "payload": payload,
    }
    return {**content, "content_checksum": canonical_checksum(content)}


def _failure_partial_receipts(normalized_work: float) -> dict[str, object]:
    """Return bounded structured partial-work custody for a failed fixture."""
    return {
        "closed_artifact_count": 0,
        "closed_artifact_bytes": 0,
        "closed_role_counts": {},
        "normalized_work_components": [{"kind": "bounded_fixture", "work": normalized_work}],
        "normalized_work_unavailable": False,
        "normalized_work_is_lower_bound": False,
        "unavailable_partial_work_roles": [],
    }


def production_screening_job(
    candidate: ScreeningCandidateRef,
    cell: ScreeningCell,
    *,
    target_manifest: TargetPopulationManifest | None = None,
) -> TrainingJob:
    """Return one fully fingerprinted paper-screen fixture job.

    Returns:
        A job whose scientific identity matches the manifest candidate/cell.
    """
    schedule = _production_strategy_schedule()
    identity = canonical_checksum({
        "candidate": candidate.configuration_checksum,
        "cell": cell.cell_id,
    }).removeprefix("sha256:")
    job_id = f"wp22_production_screen_{identity}"
    target = (
        None
        if target_manifest is None
        else next(item for item in target_manifest.instances if item.target_instance_id == cell.target_instance_id)
    )
    return TrainingJob(
        job_id=job_id,
        preset="paper-screen",
        method_id=candidate.method_id,
        implementation_kind="phase2_pipeline",
        candidate_configuration_checksum=candidate.configuration_checksum,
        implementation_checksum=_checksum(
            f"production screening fixture implementation {candidate.configuration_checksum}",
        ),
        strategy_schedule_checksum=schedule.content_checksum,
        strategy_schedule=schedule,
        target_manifest_checksum=(
            _checksum("production screening fixture target manifest")
            if target_manifest is None
            else target_manifest.content_checksum
        ),
        target_instance_id=cell.target_instance_id,
        target_spec_checksum=(
            _checksum(f"production screening target {cell.target_instance_id}")
            if target is None
            else target.content_checksum
        ),
        family_id=cell.family_id,
        stratum_id=cell.stratum_id,
        qubit_count=cell.qubit_count,
        data_role="screening_selection",
        optimization_block_id=cell.cell_id,
        optimization_seed=cell.optimization_seed,
        evaluation_seed=cell.screening_seed,
        output_path=f"roles/screening_selection/{cell.family_id}/{cell.target_instance_id}/{job_id}",
        execution_profile_checksum=_checksum("production screening execution profile"),
        scoped_binding_checksum=_checksum(
            f"production screening scoped binding {candidate.configuration_checksum}",
        ),
        executable_binding_checksum=_checksum(
            f"production screening executable binding {candidate.configuration_checksum}",
        ),
        evaluation_policy_checksum=_checksum("production screening evaluation policy"),
        target_configuration_checksum=(
            _checksum("production screening target configuration")
            if target_manifest is None
            else target_manifest.population_config_checksum
        ),
        source_fingerprint_checksum=_checksum("production screening source fingerprint"),
        scheduled_execution_program_checksum=_checksum("production screening scheduled program"),
    )


def production_screening_source(
    candidate: ScreeningCandidateRef,
    cell: ScreeningCell,
    job_directory: Path,
    *,
    fidelity: float | None,
    fixed_trajectory_count: int = 2,
    raw_trajectory_count: int | None = None,
    raw_data_role: str = "screening_selection",
    source_fingerprint_checksum: str | None = None,
    execution_source_manifest_checksum: str | None = None,
    outcome_attempt: int = 1,
) -> ProductionScreeningSourceRecord:
    """Publish and reopen one real WP22E manifest-backed screening record.

    Returns:
        The F-layer record initialized through immutable E-layer custody.
    """
    job = production_screening_job(candidate, cell)
    source_checksum = (
        cast("str", job.source_fingerprint_checksum)
        if source_fingerprint_checksum is None
        else source_fingerprint_checksum
    )
    execution_source_checksum = (
        _checksum("production screening execution source manifest")
        if execution_source_manifest_checksum is None
        else execution_source_manifest_checksum
    )
    store = ProductionAttemptStore(job_directory, job.content_checksum, 1)
    stored_count = fixed_trajectory_count if raw_trajectory_count is None else raw_trajectory_count
    resource_normalized_work = float(stored_count) if fidelity is not None else 1.0
    resources = CircuitResourceMetrics(
        qubit_count=cell.qubit_count,
        trainable_parameter_count=0,
        logical_events=(
            LogicalEventSignature(
                ordinal=0,
                logical_gate_id=0,
                name="rx",
                sites=(0,),
                parameter_index=None,
                angle_scale=1.0,
                angle_offset=0.0,
                fixed_parameters=(0.0,),
                noise_enabled=True,
            ),
        ),
        native_events=(),
    )
    circuit_checksum = canonical_checksum({"screening fixture circuit": job.content_checksum})
    resource_ref = store.write_json_blob(
        "runtime/resources.json",
        _production_document(
            "runtime_resources",
            {
                "job_checksum": job.content_checksum,
                "source_fingerprint_checksum": source_checksum,
                "wall_time_seconds": 0.0,
                "peak_memory_bytes": 0,
                "normalized_work": resource_normalized_work,
                "failure_phase": None if fidelity is not None else "operator_growth_execution",
                "partial_receipts": (
                    None if fidelity is not None else _failure_partial_receipts(resource_normalized_work)
                ),
                "circuit": {
                    "circuit_binding_checksum": circuit_checksum,
                    "topology_id": "wp22_fixture_topology",
                    "qubit_count": resources.qubit_count,
                    "parameter_count": resources.trainable_parameter_count,
                    "logical_gate_count": len(resources.logical_events),
                    "logical_two_qubit_gate_count": resources.logical_two_qubit_gates,
                    "noisy_gate_indices": [],
                    "compiled_resources": resources.to_dict(),
                    "compiled_resources_checksum": resources.content_checksum,
                    "native_two_qubit_gates_per_chain_edge": list(
                        resources.native_two_qubit_gates_per_chain_edge,
                    ),
                },
            },
        ),
        role="runtime_resources",
    )
    raw_ref = None
    schedule_snapshot_ref = None
    blobs = [resource_ref]
    derived_metrics: dict[str, object] = {
        "execution_preset": "paper-screen",
        "promotion_eligible": False,
    }
    if fidelity is not None:
        snapshot = _BoundedScheduledExecutionSnapshot(
            cast("str", job.scheduled_execution_program_checksum),
        )
        schedule_snapshot_ref = store.write_blob(
            "schedule/snapshot.json",
            f"{snapshot.to_json()}\n".encode(),
            role="schedule_snapshot",
            logical_checksum=snapshot.content_checksum,
        )
        blobs.append(schedule_snapshot_ref)
        values = tuple(float(fidelity) for _ in range(stored_count))
        evaluation_seed_domain = "screening_selection"
        evaluation_configuration_checksum = canonical_checksum({
            "job_checksum": job.content_checksum,
            "evaluation_policy_checksum": job.evaluation_policy_checksum,
            "circuit_checksum": circuit_checksum,
            "parameter_checksum": canonical_checksum({"parameters": []}),
        })
        provider_checksum = canonical_checksum({"bounded fixture provider": job.content_checksum})
        fresh_ensemble = KrotovFixedMapEnsemble(
            role="screening_selection",
            resolved_seed=job.evaluation_seed,
            stage_index=0,
            stage_id="fresh_evaluation",
            stage_configuration_checksum=evaluation_configuration_checksum,
            circuit_checksum=circuit_checksum,
            provider_checksum=provider_checksum,
            ensemble_index=0,
            refresh_index=0,
            global_iteration_start=0,
            trajectory_maps=tuple(
                (KrotovNoiseMap(source_gate_index=0, is_identity=True),) for _ in range(stored_count)
            ),
        )
        fresh_ensemble_ref = store.write_blob(
            "evaluation/fresh_fixed_map_ensemble.json",
            f"{fresh_ensemble.to_json()}\n".encode(),
            role="fixed_map_ensemble",
            logical_checksum=fresh_ensemble.content_checksum,
        )
        blobs.append(fresh_ensemble_ref)
        raw_ref = store.write_json_blob(
            "evaluation/raw_trajectory_fidelities.json",
            _production_document(
                "raw_trajectory_fidelities",
                {
                    "job_checksum": job.content_checksum,
                    "evaluation_policy_checksum": job.evaluation_policy_checksum,
                    "evaluation_configuration_checksum": evaluation_configuration_checksum,
                    "data_role": raw_data_role,
                    "seed_domain": evaluation_seed_domain,
                    "evaluation_seed": job.evaluation_seed,
                    "trajectory_count": len(values),
                    "trajectory_fidelities": list(values),
                    "fixed_map_ensemble_checksum": fresh_ensemble.content_checksum,
                    "sampled_nonidentity_events": fresh_ensemble.nonidentity_event_count,
                },
            ),
            role="raw_trajectory_sidecar",
        )
        blobs.append(raw_ref)
        derived_metrics.update({
            "noisy_fidelity": float(np.mean(np.asarray(values, dtype=np.float64))),
            "trajectory_count": len(values),
            "evaluation_configuration_checksum": evaluation_configuration_checksum,
            "evaluation_seed": job.evaluation_seed,
            "evaluation_data_role": raw_data_role,
            "evaluation_seed_domain": evaluation_seed_domain,
            "reporting_prefixes": [len(values)],
            "prefix_mean_fidelities": {str(len(values)): float(fidelity)},
            "fresh_ensemble_checksum": fresh_ensemble.content_checksum,
            "provider_checksum": provider_checksum,
            "sampled_nonidentity_events": fresh_ensemble.nonidentity_event_count,
            "total_normalized_training_work": 0.0,
            "selected_start_index": snapshot.multistart_evidence.selected_start_index,
            "selected_update": snapshot.multistart_evidence.selected_update,
            "selected_parameter_checksum": snapshot.multistart_evidence.selected_parameter_checksum,
        })
    failure = (
        None
        if fidelity is not None
        else {
            "phase": "operator_growth_execution",
            "exception_type": "SyntheticFixtureFailure",
            "message": "bounded manifest-backed failure fixture",
        }
    )
    evidence = ProductionNumericalEvidence(
        job_checksum=job.content_checksum,
        attempt=1,
        artifact_kind="pipeline",
        status="success" if fidelity is not None else "failure",
        execution_source_manifest_checksum=execution_source_checksum,
        source_fingerprint_checksum=source_checksum,
        executable_binding_checksum=cast("str", job.executable_binding_checksum),
        scheduled_program_checksum=cast("str", job.scheduled_execution_program_checksum),
        target_identity={
            "target_instance_id": job.target_instance_id,
            "target_instance_spec_checksum": job.target_spec_checksum,
            "population_config_checksum": job.target_configuration_checksum,
            "target_manifest_checksum": job.target_manifest_checksum,
            "family_id": job.family_id,
            "stratum_id": job.stratum_id,
            "qubit_count": job.qubit_count,
            "parameter_checksum": canonical_checksum({"bounded fixture parameters": job.target_instance_id}),
            "vector_checksum": canonical_checksum({"bounded fixture target": job.target_instance_id}),
            "norm": 1.0,
        },
        evaluation_policy_checksum=cast("str", job.evaluation_policy_checksum),
        structural_prefix_checksums=(_checksum(f"production prefix {job.content_checksum}"),),
        schedule_snapshot_ref=schedule_snapshot_ref,
        map_evidence_refs=(),
        diagnostic_refs=(),
        raw_trajectory_ref=raw_ref,
        resource_ref=resource_ref,
        derived_metrics=derived_metrics,
        failure=failure,
    )
    evidence_ref = store.write_json_blob(
        "production_evidence.json",
        evidence.to_dict(),
        role="production_evidence",
    )
    blobs.append(evidence_ref)
    reference = store.publish(
        artifact_kind="pipeline",
        status=evidence.status,
        execution_source_manifest_checksum=execution_source_checksum,
        source_fingerprint_checksum=source_checksum,
        blobs=blobs,
        evidence_ref=evidence_ref,
    )
    outcome = TrainingJobOutcome(
        job_checksum=job.content_checksum,
        status=evidence.status,
        result_artifact_checksum=(reference.content_checksum if evidence.status == "success" else None),
        exception_type=(None if evidence.status == "success" else "executor_failure"),
        message=(None if evidence.status == "success" else "bounded manifest-backed failure fixture"),
        attempt=outcome_attempt,
    )
    outcome_path = training_job_attempt_path(job_directory, 1)
    outcome_path.parent.mkdir(parents=True, exist_ok=True)
    outcome_path.write_text(f"{canonical_json(outcome.to_dict())}\n", encoding="utf-8")
    with patch(
        "benchmarks.state_preparation.phase2.production_executors.ScheduledExecutionSnapshot",
        _BoundedScheduledExecutionSnapshot,
    ):
        return ProductionScreeningSourceRecord(
            candidate,
            cell,
            job,
            job_directory,
            fixed_trajectory_count=fixed_trajectory_count,
        )


def production_screening_records_fixture(
    manifest: ScreeningManifest,
    *,
    target_manifest: TargetPopulationManifest,
    fixed_trajectory_count: int,
    execution_source_manifest_checksum: str,
    promoted_method_id: str = "fixed_depth_bmpd_crn",
) -> tuple[ProductionScreeningSourceRecord, ...]:
    """Return the complete typed production-record universe for seal tests.

    Representative tests reopen actual WP22E manifests.  This bounded full-grid
    seam retains exact production record types while avoiding 1,296 redundant
    filesystem publications in the final-seal integration test.

    Returns:
        Exactly one typed record for each manifest candidate/cell pair.
    """
    records: list[ProductionScreeningSourceRecord] = []
    resources = CircuitResourceMetrics(
        qubit_count=6,
        trainable_parameter_count=0,
        logical_events=(),
        native_events=(),
    )
    for candidate in manifest.candidates:
        for cell in manifest.cells:
            fidelity = (
                0.82
                if candidate.method_id == promoted_method_id
                else 0.80
                if candidate.method_id in {"layerwise_bmpd_crn_v2", "layerwise_bmpd_noiseless"}
                else None
            )
            job = production_screening_job(candidate, cell, target_manifest=target_manifest)
            reference_checksum = canonical_checksum({
                "production screening result": job.content_checksum,
            })
            outcome = TrainingJobOutcome(
                job_checksum=job.content_checksum,
                status="success" if fidelity is not None else "failure",
                result_artifact_checksum=reference_checksum if fidelity is not None else None,
                exception_type=None if fidelity is not None else "executor_failure",
                message=None if fidelity is not None else "bounded manifest-backed failure fixture",
                attempt=1,
            )
            custody = object.__new__(ProductionResultCustody)
            reference = SimpleNamespace(
                schema_version="yaqs.state_preparation.phase2.production_result_ref.v1",
                content_checksum=reference_checksum,
                attempt=1,
                job_checksum=job.content_checksum,
                status=outcome.status,
                execution_source_manifest_checksum=execution_source_manifest_checksum,
            )
            production_evidence = SimpleNamespace(
                artifact_kind="pipeline",
                execution_source_manifest_checksum=execution_source_manifest_checksum,
                structural_prefix_checksums=(canonical_checksum({"prefix": job.content_checksum}),),
                derived_metrics={"execution_preset": "paper-screen"},
                source_fingerprint_checksum=job.source_fingerprint_checksum,
                executable_binding_checksum=job.executable_binding_checksum,
                scheduled_program_checksum=job.scheduled_execution_program_checksum,
                evaluation_policy_checksum=job.evaluation_policy_checksum,
                target_identity={
                    "target_instance_id": job.target_instance_id,
                    "target_instance_spec_checksum": job.target_spec_checksum,
                    "population_config_checksum": job.target_configuration_checksum,
                    "target_manifest_checksum": job.target_manifest_checksum,
                    "family_id": job.family_id,
                    "stratum_id": job.stratum_id,
                    "qubit_count": job.qubit_count,
                },
            )
            raw_payload = (
                None
                if fidelity is None
                else {
                    "data_role": "screening_selection",
                    "evaluation_seed": job.evaluation_seed,
                    "trajectory_count": fixed_trajectory_count,
                    "trajectory_fidelities": (fidelity,) * fixed_trajectory_count,
                }
            )
            _set_frozen_slot(custody, "reference", reference)
            _set_frozen_slot(custody, "production_evidence", production_evidence)
            _set_frozen_slot(custody, "result_evidence_checksum", canonical_checksum({"evidence": reference_checksum}))
            _set_frozen_slot(custody, "raw_trajectory_payload", raw_payload)
            _set_frozen_slot(
                custody,
                "raw_trajectory_document_checksum",
                None if raw_payload is None else canonical_checksum({"raw": reference_checksum}),
            )
            _set_frozen_slot(
                custody,
                "resource_payload",
                {
                    "normalized_work": 1.0,
                    "circuit": {
                        "circuit_binding_checksum": canonical_checksum({"circuit": job.content_checksum}),
                        "compiled_resources": resources.to_dict(),
                        "compiled_resources_checksum": resources.content_checksum,
                        "native_two_qubit_gates_per_chain_edge": resources.native_two_qubit_gates_per_chain_edge,
                    },
                },
            )
            _set_frozen_slot(
                custody,
                "resource_document_checksum",
                canonical_checksum({"resource": reference_checksum}),
            )
            _set_frozen_slot(custody, "pilot_diagnostics", ())
            _set_frozen_slot(
                custody,
                "schema_version",
                "yaqs.state_preparation.phase2.production_result_custody.v1",
            )
            record = object.__new__(ProductionScreeningSourceRecord)
            _set_frozen_slot(record, "candidate", candidate)
            _set_frozen_slot(record, "cell", cell)
            _set_frozen_slot(record, "job", job)
            _set_frozen_slot(record, "outcome", outcome)
            _set_frozen_slot(record, "result_custody", custody)
            _set_frozen_slot(record, "fixed_trajectory_count", fixed_trajectory_count)
            _set_frozen_slot(record, "circuit_resources", resources if fidelity is not None else None)
            _set_frozen_slot(
                record,
                "schema_version",
                "yaqs.state_preparation.phase2.wp22_production_screening_source.v1",
            )
            records.append(record)
    return tuple(records)


def production_screening_custody_fixture(
    preregistration: InitialPreregistration,
    manifest: ScreeningManifest,
    target_manifest: TargetPopulationManifest,
    design: SampleSizeDesign,
    execution_source_manifest: ExecutionSourceManifest,
    *,
    normalized_compute_cap: float = 1_000.0,
) -> ProductionScreeningCustody:
    """Return an exact-context aggregate around the bounded typed full grid.

    Returns:
        A typed production screening custody accepted by final sealing.
    """
    records = production_screening_records_fixture(
        manifest,
        target_manifest=target_manifest,
        fixed_trajectory_count=design.fixed_test_trajectory_count,
        execution_source_manifest_checksum=execution_source_manifest.content_checksum,
    )
    context = object.__new__(TrainingExecutionContext)
    plan_jobs = tuple(item.job for item in records)
    _set_frozen_slot(
        context,
        "plan",
        SimpleNamespace(
            preset="paper-screen",
            jobs=plan_jobs,
            content_checksum=canonical_checksum({"paper_screen_jobs": [item.content_checksum for item in plan_jobs]}),
        ),
    )
    _set_frozen_slot(context, "preregistration", preregistration)
    _set_frozen_slot(context, "screening_manifest", manifest)
    _set_frozen_slot(context, "required_sample_size_design", design)
    _set_frozen_slot(context, "target_manifests", (target_manifest,))
    _set_frozen_slot(context, "execution_source_manifest", execution_source_manifest)
    _set_frozen_slot(
        context,
        "scoped_bindings",
        tuple(
            SimpleNamespace(
                content_checksum=next(
                    item.job.executable_binding_checksum for item in records if item.candidate is candidate
                ),
                binding=SimpleNamespace(
                    publication_method_id=candidate.method_id,
                    publication_candidate_checksum=candidate.configuration_checksum,
                    strategy_schedule=next(
                        item.job.strategy_schedule for item in records if item.candidate is candidate
                    ),
                    implementation_checksum=next(
                        item.job.implementation_checksum for item in records if item.candidate is candidate
                    ),
                    content_checksum=next(
                        item.job.scoped_binding_checksum for item in records if item.candidate is candidate
                    ),
                    execution_budget=SimpleNamespace(normalized_compute_cap=normalized_compute_cap),
                ),
            )
            for candidate in manifest.candidates
        ),
    )
    return ProductionScreeningCustody(context, records)


def production_screening_record_with_fixed_count(
    record: ProductionScreeningSourceRecord,
    fixed_trajectory_count: int,
) -> ProductionScreeningSourceRecord:
    """Copy one typed record with only its seal-time count projection changed.

    Returns:
        An otherwise identical production record for negative closure tests.
    """
    changed = copy(record)
    _set_frozen_slot(changed, "fixed_trajectory_count", fixed_trajectory_count)
    return changed


def candidate_configurations(
    preregistration: InitialPreregistration,
) -> tuple[WP22CandidateConfiguration, ...]:
    """Build one explicit typed WP18 fixture configuration per frozen method.

    The production operator-growth path is tested with its real standalone
    artifact elsewhere. This synthetic Cartesian fixture deliberately binds
    every publication method, including ADAPT-style, to a complete WP18 test
    pipeline so all 1,296 rows retain one uniform authoritative source type.

    Returns:
        The nine typed synthetic candidate configurations.
    """
    matching = _checksum("wp22 fixture matched noisy/noiseless projection")
    schedule = _checksum("wp22 fixture strategy schedule")
    candidates: list[WP22CandidateConfiguration] = []
    for policy in preregistration.candidate_methods:
        if policy["scope"] != "all_families":
            continue
        method_id = cast("str", policy["method_id"])
        noisy = cast("bool", policy["noisy_training"])
        template = _candidate_template(method_id, noisy=noisy)
        publication_mapping: dict[str, object] = {}
        if method_id == IMPACT_PRUNING_PUBLICATION_METHOD_ID:
            publication_mapping = {
                "mapping_version": WP22_PUBLICATION_PRUNING_MAPPING_VERSION,
                "publication_method_id": IMPACT_PRUNING_PUBLICATION_METHOD_ID,
                "implementation_method_id": TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
                "pruning_rule": "impact_iterative",
                "minimum_pruning_rounds": 2,
                "required_final_finetune_sampling": "crn_fixed",
            }
        candidates.append(
            WP22CandidateConfiguration(
                method_id=method_id,
                implementation_kind="phase2_pipeline",
                implementation_method_id=template.method_id,
                implementation_schema_version=template.schema_version,
                implementation_checksum=template.configuration_checksum,
                strategy_schedule_checksum=schedule,
                resource_stratum_id=template.resource_stratum_id,
                noisy_training=noisy,
                matching_projection_checksum=(
                    matching if method_id in {"layerwise_bmpd_crn_v2", "layerwise_bmpd_noiseless"} else None
                ),
                publication_mapping=publication_mapping,
            )
        )
    return tuple(candidates)


def _resolved_template(candidate: WP22CandidateConfiguration) -> TrainingPipelineTemplate:
    """Rebuild and verify the deterministic template bound by a fixture candidate.

    Returns:
        The exact template checksum-bound by the candidate.

    Raises:
        ValueError: If the candidate is outside the explicit fixture registry.
    """
    template = _candidate_template(candidate.method_id, noisy=candidate.noisy_training)
    if (
        candidate.implementation_kind != "phase2_pipeline"
        or candidate.implementation_method_id != template.method_id
        or candidate.implementation_checksum != template.configuration_checksum
    ):
        msg = "candidate is not one of the explicit typed WP18 screening fixtures."
        raise ValueError(msg)
    return template


def wp18_source(
    preregistration: InitialPreregistration,
    target_manifest: TargetPopulationManifest,
    candidate: WP22CandidateConfiguration,
    cell: ScreeningCell,
    fidelity: float | None,
) -> ScreeningSourceRecord:
    """Build one complete typed WP18 success or failure source fixture.

    Args:
        preregistration: Trusted protocol artifact.
        target_manifest: Exact typed outer-screening target population.
        candidate: Exact fixture candidate configuration.
        cell: Exact outer-screening cell.
        fidelity: Two-trajectory mean, or ``None`` for a typed WP18 failure.

    Returns:
        A fully replayable production-schema screening source.

    Raises:
        ValueError: If the cell target is absent from the typed target manifest.
    """
    template = _resolved_template(candidate)
    target = next(
        (item for item in target_manifest.instances if item.target_instance_id == cell.target_instance_id),
        None,
    )
    if target is None:
        msg = "screening cell target is absent from the typed target manifest."
        raise ValueError(msg)
    pipeline = template.resolve(
        target_namespace="phase2",
        target_manifest=target_manifest,
        target_instance_id=cell.target_instance_id,
        target_population_manifest_checksum=target_manifest.content_checksum,
        target_instance_spec_checksum=target.content_checksum,
        target_family_id=cell.family_id,
        target_stratum_id=cell.stratum_id,
        qubit_count=cell.qubit_count,
        optimization_block_id=cell.cell_id,
        optimization_seed=cell.optimization_seed,
        data_role="screening_selection",
    )
    result = _pipeline_result(pipeline)
    config = replace(_evaluation(result, seed=cell.screening_seed), trajectory_budget=2)
    runtime_checksum = _checksum("wp22 fixture runtime")
    if fidelity is None:
        record = PipelineBenchmarkFailure(
            config=config,
            failure_phase="evaluation",
            exception_type="RuntimeError",
            message="typed synthetic screening failure",
            traceback=None,
            retryable=False,
            attempt=1,
            materialized_circuit_path=None,
            materialized_circuit_checksum=None,
            wall_time_seconds=0.0,
            runtime_fingerprint_checksum=runtime_checksum,
        )
        return ScreeningSourceRecord.from_pipeline_record(
            candidate=candidate,
            cell=cell,
            template=template,
            pipeline_result=result,
            record=record,
            work_ledger=WP20WorkLedger(),
            circuit_resources=None,
            evaluation_evidence=None,
            materialization=None,
            preregistration=preregistration,
        )

    fidelities = (float(fidelity) - 0.01, float(fidelity) + 0.01)
    provider_checksum = _checksum("wp22 fixture evaluation provider")
    noise_map = KrotovNoiseMap(source_gate_index=0, is_identity=True)
    ensemble = KrotovFixedMapEnsemble(
        role="screening_selection",
        resolved_seed=cell.screening_seed,
        stage_index=0,
        stage_id="final_evaluation",
        stage_configuration_checksum=config.configuration_checksum,
        circuit_checksum=config.materialized_circuit_checksum,
        provider_checksum=provider_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
        trajectory_maps=((noise_map,), (noise_map,)),
    )
    map_partition = {
        "ensemble_id": ensemble.ensemble_id,
        "content_checksum": ensemble.content_checksum,
        "trajectory_count": ensemble.trajectory_count,
    }
    sidecar = create_phase2_trajectory_sidecar(
        evaluation_row_id=config.evaluation_row_id,
        pipeline_training_id=config.pipeline_training_id,
        map_role="screening_selection",
        map_partitions=(map_partition,),
        fidelities=fidelities,
    )
    work = WP20WorkLedger(
        forward_circuit_evaluations=2,
        test_trajectories=2,
        objective_calls=1,
    )
    record = PipelineBenchmarkResult(
        config=config,
        materialized_circuit_path=f"circuits/{config.materialized_circuit_id}.bin",
        test_noiseless_fidelity=0.93,
        test_noisy_fidelity=statistics.fmean(fidelities),
        noisy_fidelity_standard_deviation=statistics.stdev(fidelities),
        noisy_fidelity_standard_error=statistics.stdev(fidelities) / 2**0.5,
        confidence_interval_lower=None,
        confidence_interval_upper=None,
        sampled_nonidentity_events=0,
        trajectory_sidecar_path=f"trajectories/{config.evaluation_row_id}.npz",
        trajectory_sidecar_checksum=artifact_checksum(sidecar),
        evaluation_wall_time_seconds=0.0,
        peak_memory_bytes=0,
        normalized_work=work.phase2_projection(),
        runtime_fingerprint_checksum=runtime_checksum,
    )
    materialization = MaterializedCircuitArtifact(
        materialized_circuit_id=config.materialized_circuit_id,
        pipeline_training_id=pipeline.training_id,
        pipeline_result_checksum=result.content_checksum,
        final_checkpoint_checksum=result.final_checkpoint_checksum,
        materialization_policy_checksum=config.final_materialization_policy_checksum,
        path=record.materialized_circuit_path,
        payload_checksum=config.materialized_circuit_checksum,
        wall_time_seconds=0.0,
        peak_memory_bytes=0,
        runtime_fingerprint_checksum=runtime_checksum,
    )
    map_ref = FixedMapArtifactRef(
        role="screening_selection",
        ensemble_id=ensemble.ensemble_id,
        content_checksum=ensemble.content_checksum,
        path=f"maps/{ensemble.ensemble_id}.json",
        file_checksum=artifact_checksum(ensemble.to_json().encode()),
    )
    evidence = EvaluationEvidenceArtifact(
        evaluation_row_id=record.evaluation_row_id,
        record_checksum=record.content_checksum,
        pipeline_result_checksum=result.content_checksum,
        materialization_checksum=materialization.content_checksum,
        evaluation_provider_checksum=provider_checksum,
        evaluation_map_artifacts=(map_ref,),
    )
    return ScreeningSourceRecord.from_pipeline_record(
        candidate=candidate,
        cell=cell,
        template=template,
        pipeline_result=result,
        record=record,
        work_ledger=work,
        circuit_resources=CircuitResourceMetrics(
            qubit_count=6,
            trainable_parameter_count=0,
            logical_events=(),
            native_events=(),
        ),
        evaluation_evidence=evidence,
        materialization=materialization,
        preregistration=preregistration,
        evaluation_maps=(ensemble,),
        trajectory_sidecar_payload=sidecar,
    )


def complete_screening_sources(
    preregistration: InitialPreregistration,
    target_manifest: TargetPopulationManifest,
    candidates: tuple[WP22CandidateConfiguration, ...],
    cells: tuple[ScreeningCell, ...],
    *,
    promoted_method_id: str = "fixed_depth_bmpd_crn",
) -> tuple[ScreeningSourceRecord, ...]:
    """Build a complete 1,296-row typed source universe with one improvement.

    Returns:
        Complete candidate-by-cell source records.
    """
    return tuple(
        wp18_source(
            preregistration,
            target_manifest,
            candidate,
            cell,
            (
                0.82
                if candidate.method_id == promoted_method_id
                else 0.80
                if candidate.method_id in {"layerwise_bmpd_crn_v2", "layerwise_bmpd_noiseless"}
                else None
            ),
        )
        for candidate in candidates
        for cell in cells
    )


__all__ = [
    "candidate_configurations",
    "complete_screening_sources",
    "production_screening_custody_fixture",
    "production_screening_job",
    "production_screening_record_with_fixed_count",
    "production_screening_records_fixture",
    "production_screening_source",
    "wp18_source",
]
