# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Rigorous synthetic tests for the frozen WP22 primary analysis."""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np
import pytest

import benchmarks.state_preparation.phase2.primary_analysis as primary_analysis_module
from benchmarks.state_preparation import phase2
from benchmarks.state_preparation.phase2.canonical import canonical_checksum, canonical_json
from benchmarks.state_preparation.phase2.execution_context import ConfirmationExecutionContext
from benchmarks.state_preparation.phase2.primary_analysis import (
    ConfirmatoryEvaluationArtifact,
    ConfirmatoryObservation,
    ConfirmatoryProductionAttemptReceipt,
    ConfirmatoryResultArtifact,
    PrimaryAnalysisResult,
    _analyze_confirmatory_statistics,  # noqa: PLC2701 -- explicitly private statistical test seam
    _ConfirmatoryAnalysisStatistics,  # noqa: PLC2701 -- explicitly private non-production view
    analyze_production_confirmatory_results,
)
from benchmarks.state_preparation.phase2.production_executors import (
    ArtifactBlobRef,
    AttemptArtifactManifest,
    ProductionAttemptStore,
    ProductionNumericalEvidence,
    ReopenedProductionResult,
    ResultArtifactRef,
    SyntheticConfirmationExecutor,
    SyntheticConfirmationFixture,
)
from benchmarks.state_preparation.phase2.protocol import (
    PRIMARY_TARGET_FAMILIES,
    FinalComparatorRef,
    FinalConfigurationExecutionManifest,
    FinalConfigurationExecutionRef,
    FinalConfirmationSeal,
    PrimaryContrastBinding,
    load_initial_preregistration,
)
from benchmarks.state_preparation.phase2.result_custody import (
    ProductionResultCustody,
    reopen_confirmatory_production_attempt,
)
from benchmarks.state_preparation.phase2.targets import (
    TargetPopulationManifest,
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.training_orchestration import (
    JobExecutionControls,
    TrainingJobOutcome,
    TrainingRunPlan,
    build_paper_confirm_plan,
    confirmatory_evaluation_policy_checksum,
)
from benchmarks.state_preparation.phase2.training_schedules import FrozenTrainingPolicyUniverse
from tests.benchmarks.wp22_confirmation_test_support import build_confirmation_context_fixture

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from benchmarks.state_preparation.phase2.protocol import InitialPreregistration
    from benchmarks.state_preparation.phase2.training_orchestration import TrainingJob

PROMOTED_CONFIGURATION = canonical_checksum({"configuration": "promoted"})
V2_CONFIGURATION = canonical_checksum({"configuration": "layerwise-v2"})
NOISELESS_CONFIGURATION = canonical_checksum({"configuration": "matched-noiseless"})
MATCHING_PROJECTION = canonical_checksum({"projection": "noisy-to-noiseless"})
_CONFIRMATORY_MASTER = bytes(range(32))
Custody: TypeAlias = tuple[
    TargetPopulationManifest,
    TrainingRunPlan,
    tuple[TrainingJobOutcome, ...],
    tuple[ConfirmatoryResultArtifact, ...],
    FinalConfigurationExecutionManifest,
]
AnalysisResult: TypeAlias = _ConfirmatoryAnalysisStatistics
ProductionBundle: TypeAlias = tuple[
    PrimaryAnalysisResult,
    ConfirmationExecutionContext,
    dict[str, ProductionResultCustody],
    str,
    str,
]


def _production_document(document_type: str, payload: dict[str, object]) -> dict[str, object]:
    """Return one checksum-sealed WP22E production document."""
    content = {
        "schema_version": "yaqs.state_preparation.phase2.production_document.v1",
        "document_type": document_type,
        "payload": payload,
    }
    return {**content, "content_checksum": canonical_checksum(content)}


def _blob_ref(role: str, relative_name: str, logical_checksum: str) -> ArtifactBlobRef:
    """Return one typed in-memory attempt-member reference."""
    return ArtifactBlobRef(
        role=role,
        media_type="application/json",
        path=f"production_attempts/attempt_000001/{relative_name}",
        byte_count=1,
        file_checksum=canonical_checksum({"test_file": relative_name, "logical": logical_checksum}),
        logical_checksum=logical_checksum,
    )


def _real_confirmation_custody(
    job: TrainingJob,
    context: ConfirmationExecutionContext,
    trajectory_fidelities: tuple[float, ...] | None,
) -> tuple[ProductionResultCustody, str | None]:
    """Construct a fully typed reopened real-attempt projection in memory.

    Returns:
        The production custody and, for failure, the manifest root without its
        partial member so the test can prove complete-manifest addressing.
    """
    request = job.confirm_execution_request
    assert request is not None
    status: Literal["success", "failure"] = "success" if trajectory_fidelities is not None else "failure"
    artifact_kind = context.artifact_kind(request)
    policy_checksum = confirmatory_evaluation_policy_checksum(request)
    source_checksum = request.execution_source_checksum
    resource_document = _production_document(
        "runtime_resources",
        {
            "job_checksum": request.content_checksum,
            "source_fingerprint_checksum": source_checksum,
            "wall_time_seconds": 1.0,
            "peak_memory_bytes": 1,
            "normalized_work": 1.0,
            "circuit": None,
        },
    )
    resource_ref = _blob_ref(
        "runtime_resources",
        "runtime_resources.json",
        cast("str", resource_document["content_checksum"]),
    )
    raw_document: dict[str, object] | None = None
    raw_ref: ArtifactBlobRef | None = None
    schedule_ref: ArtifactBlobRef | None = None
    partial_ref: ArtifactBlobRef | None = None
    if trajectory_fidelities is not None:
        raw_document = _production_document(
            "raw_trajectory_fidelities",
            {
                "job_checksum": request.content_checksum,
                "evaluation_policy_checksum": policy_checksum,
                "data_role": "confirmatory",
                "seed_domain": "confirmatory_test",
                "evaluation_seed": request.evaluation_seed,
                "trajectory_count": len(trajectory_fidelities),
                "trajectory_fidelities": tuple(trajectory_fidelities),
            },
        )
        raw_ref = _blob_ref(
            "raw_trajectory_sidecar",
            "raw_trajectory_fidelities.json",
            cast("str", raw_document["content_checksum"]),
        )
        schedule_ref = _blob_ref(
            "schedule_snapshot",
            "schedule_snapshot.json",
            canonical_checksum({"schedule_snapshot": request.content_checksum}),
        )
    else:
        partial_ref = _blob_ref(
            "partial_checkpoint",
            "partial_checkpoint.json",
            canonical_checksum({"partial_checkpoint": request.content_checksum}),
        )
    metrics: dict[str, object] = {
        "execution_preset": "paper-confirm",
        "scheduled_noisy_training": False,
        "pilot_diagnostic_required": False,
        "strategy_schedule_checksum": request.hyperparameters_checksum,
    }
    if trajectory_fidelities is not None:
        metrics["noisy_fidelity"] = float(np.mean(np.asarray(trajectory_fidelities, dtype=np.float64)))
        metrics.update({
            "evaluation_data_role": "confirmatory",
            "evaluation_seed_domain": "confirmatory_test",
            "evaluation_seed": request.evaluation_seed,
            "trajectory_count": request.fixed_test_trajectory_count,
        })
    evidence = ProductionNumericalEvidence(
        job_checksum=request.content_checksum,
        attempt=1,
        artifact_kind=artifact_kind,
        status=status,
        execution_source_manifest_checksum=source_checksum,
        source_fingerprint_checksum=source_checksum,
        executable_binding_checksum=request.executable_binding_checksum,
        scheduled_program_checksum=context.scheduled_program_checksum(request),
        target_identity={
            "target_instance_id": request.target_instance_id,
            "target_instance_spec_checksum": request.target_spec_checksum,
            "target_manifest_checksum": request.target_manifest_checksum,
            "family_id": request.family_id,
            "stratum_id": request.stratum_id,
            "qubit_count": request.qubit_count,
        },
        evaluation_policy_checksum=policy_checksum,
        structural_prefix_checksums=(),
        schedule_snapshot_ref=schedule_ref,
        map_evidence_refs=(),
        diagnostic_refs=(),
        raw_trajectory_ref=raw_ref,
        resource_ref=resource_ref,
        derived_metrics=metrics,
        failure=(
            None
            if status == "success"
            else {
                "phase": "production_execution",
                "exception_type": "RuntimeError",
                "message": "redacted test failure",
            }
        ),
    )
    evidence_ref = _blob_ref("production_evidence", "production_evidence.json", evidence.content_checksum)
    member_refs = tuple(
        sorted(
            (
                evidence_ref,
                resource_ref,
                *((raw_ref,) if raw_ref is not None else ()),
                *((schedule_ref,) if schedule_ref is not None else ()),
                *((partial_ref,) if partial_ref is not None else ()),
            ),
            key=lambda item: item.path,
        )
    )
    manifest = AttemptArtifactManifest(
        job_checksum=request.content_checksum,
        attempt=1,
        artifact_kind=artifact_kind,
        status=status,
        execution_source_manifest_checksum=source_checksum,
        source_fingerprint_checksum=source_checksum,
        blobs=member_refs,
        evidence_ref=evidence_ref,
    )
    manifest_without_partial_checksum: str | None = None
    if partial_ref is not None:
        manifest_without_partial_checksum = AttemptArtifactManifest(
            job_checksum=request.content_checksum,
            attempt=1,
            artifact_kind=artifact_kind,
            status=status,
            execution_source_manifest_checksum=source_checksum,
            source_fingerprint_checksum=source_checksum,
            blobs=tuple(item for item in member_refs if item != partial_ref),
            evidence_ref=evidence_ref,
        ).content_checksum
    reference = ResultArtifactRef(
        job_checksum=request.content_checksum,
        attempt=1,
        artifact_kind=artifact_kind,
        status=status,
        execution_source_manifest_checksum=source_checksum,
        source_fingerprint_checksum=source_checksum,
        manifest_path="production_attempts/attempt_000001/attempt_manifest.json",
        manifest_file_checksum=canonical_checksum({"manifest_file": manifest.content_checksum}),
        manifest_content_checksum=manifest.content_checksum,
        evidence_checksum=evidence.content_checksum,
    )
    reopened = ReopenedProductionResult(
        reference=reference,
        manifest=manifest,
        evidence=evidence,
        raw_trajectory=raw_document,
        resources=resource_document,
        scheduled_map_evidence=(),
        diagnostic_documents=(),
    )
    return ProductionResultCustody(reopened), manifest_without_partial_checksum


def _publish_resealed_confirmation_alias_attack(
    job: TrainingJob,
    job_directory: Path,
    mutation: str,
) -> TrainingJobOutcome:
    """Publish a storage-valid attempt whose request custody has been resealed.

    Returns:
        The outer job outcome pointing to the resealed request-addressed attempt.
    """
    request = job.confirm_execution_request
    assert request is not None
    values = (0.1,) * request.fixed_test_trajectory_count
    exact_source = request.execution_source_checksum
    exact_policy = confirmatory_evaluation_policy_checksum(request)
    source = canonical_checksum({"resealed": "execution source"}) if mutation == "source" else exact_source
    evaluation_policy = (
        canonical_checksum({"resealed": "evaluation policy"}) if mutation == "evaluation_policy" else exact_policy
    )
    executable_binding = (
        canonical_checksum({"resealed": "executable binding"})
        if mutation == "executable_binding"
        else request.executable_binding_checksum
    )
    scheduled_program = (
        canonical_checksum({"resealed": "scheduled program"})
        if mutation == "scheduled_program"
        else request.hyperparameters_checksum
    )
    fixture_checksum = canonical_checksum({"resealed fixture": mutation})
    store = ProductionAttemptStore(job_directory, request.content_checksum, 1)
    raw_ref = store.write_json_blob(
        "evaluation/raw_trajectory_fidelities.json",
        _production_document(
            "raw_trajectory_fidelities",
            {
                "request_checksum": request.content_checksum,
                "evaluation_policy_checksum": evaluation_policy,
                "data_role": "confirmatory",
                "seed_domain": "confirmatory_test",
                "evaluation_seed": request.evaluation_seed,
                "trajectory_count": request.fixed_test_trajectory_count,
                "trajectory_fidelities": list(values),
                "synthetic_fixture_checksum": fixture_checksum,
            },
        ),
        role="raw_trajectory_sidecar",
    )
    resource_ref = store.write_json_blob(
        "runtime/resources.json",
        _production_document(
            "runtime_resources",
            {
                "request_checksum": request.content_checksum,
                "source_fingerprint_checksum": source,
                "wall_time_seconds": 0.0,
                "peak_memory_bytes": 0,
                "normalized_work": 0.0,
                "synthetic_fixture": True,
                "circuit": None,
            },
        ),
        role="runtime_resources",
    )
    target_identity = {
        "synthetic_fixture": True,
        "request_checksum": request.content_checksum,
        "target_instance_id": request.target_instance_id,
        "target_spec_checksum": request.target_spec_checksum,
        "qubit_count": request.qubit_count,
    }
    if mutation == "target_identity":
        target_identity["target_instance_id"] = "resealed_foreign_target"
    evidence = ProductionNumericalEvidence(
        job_checksum=request.content_checksum,
        attempt=1,
        artifact_kind="synthetic_confirmation",
        status="success",
        execution_source_manifest_checksum=source,
        source_fingerprint_checksum=source,
        executable_binding_checksum=executable_binding,
        scheduled_program_checksum=scheduled_program,
        target_identity=target_identity,
        evaluation_policy_checksum=evaluation_policy,
        structural_prefix_checksums=(),
        schedule_snapshot_ref=None,
        map_evidence_refs=(),
        diagnostic_refs=(),
        raw_trajectory_ref=raw_ref,
        resource_ref=resource_ref,
        derived_metrics={
            "noisy_fidelity": float(np.mean(np.asarray(values, dtype=np.float64))),
            "evaluation_data_role": "confirmatory",
            "evaluation_seed_domain": "confirmatory_test",
            "evaluation_seed": request.evaluation_seed,
            "trajectory_count": request.fixed_test_trajectory_count,
            "synthetic_fixture_checksum": fixture_checksum,
            "promotion_eligible": False,
            "strategy_schedule_checksum": request.hyperparameters_checksum,
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
        execution_source_manifest_checksum=source,
        source_fingerprint_checksum=source,
        blobs=(raw_ref, resource_ref, evidence_ref),
        evidence_ref=evidence_ref,
    )
    return TrainingJobOutcome(
        job_checksum=job.content_checksum,
        status="success",
        result_artifact_checksum=reference.content_checksum,
        exception_type=None,
        message=None,
        attempt=1,
    )


def _configuration_execution_manifest() -> FinalConfigurationExecutionManifest:
    """Return distinct exact execution identities for the three final configurations."""
    schedules = {item.schedule_id: item for item in FrozenTrainingPolicyUniverse.frozen().schedules}
    entries = (
        FinalConfigurationExecutionRef(
            method_id="spsa_layerwise",
            configuration_schema_version="yaqs.state_preparation.phase2.training_pipeline.v1",
            configuration_checksum=PROMOTED_CONFIGURATION,
            strategy_schedule=schedules["resampled_each_update"],
            implementation_checksum=canonical_checksum({"implementation": "promoted"}),
            scoped_binding_checksum=canonical_checksum({"scoped": "promoted"}),
            executable_binding_checksum=canonical_checksum({"executable": "promoted"}),
        ),
        FinalConfigurationExecutionRef(
            method_id="layerwise_bmpd_crn_v2",
            configuration_schema_version="yaqs.state_preparation.phase2.training_pipeline.v1",
            configuration_checksum=V2_CONFIGURATION,
            strategy_schedule=schedules["direct_matched_fixed_crn"],
            implementation_checksum=canonical_checksum({"implementation": "v2"}),
            scoped_binding_checksum=canonical_checksum({"scoped": "v2"}),
            executable_binding_checksum=canonical_checksum({"executable": "v2"}),
        ),
        FinalConfigurationExecutionRef(
            method_id="layerwise_bmpd_noiseless",
            configuration_schema_version="yaqs.state_preparation.phase2.training_pipeline.v1",
            configuration_checksum=NOISELESS_CONFIGURATION,
            strategy_schedule=schedules["direct_noiseless_control"],
            implementation_checksum=canonical_checksum({"implementation": "noiseless"}),
            scoped_binding_checksum=canonical_checksum({"scoped": "noiseless"}),
            executable_binding_checksum=canonical_checksum({"executable": "noiseless"}),
        ),
    )
    return FinalConfigurationExecutionManifest(
        manifest_id="synthetic_primary_analysis_execution",
        entries=tuple(sorted(entries, key=lambda item: (item.configuration_checksum, item.method_id))),
    )


def _seal(
    preregistration: InitialPreregistration,
    target_manifest: TargetPopulationManifest,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
) -> FinalConfirmationSeal:
    """Return a valid distinct-promoted synthetic final seal."""
    return FinalConfirmationSeal(
        seal_id="synthetic-wp22-primary-analysis-v1",
        preregistration_checksum=preregistration.content_checksum,
        promotion_decision_checksum=canonical_checksum({"promotion": "promoted"}),
        promoted_method_id="spsa_layerwise",
        promoted_configuration_checksum=PROMOTED_CONFIGURATION,
        comparators=(
            FinalComparatorRef(
                role="layerwise_v2_reference",
                method_id="layerwise_bmpd_crn_v2",
                configuration_schema_version="yaqs.state_preparation.phase2.training_pipeline.v1",
                configuration_checksum=V2_CONFIGURATION,
                matched_to_configuration_checksum=NOISELESS_CONFIGURATION,
                matching_projection_checksum=MATCHING_PROJECTION,
            ),
            FinalComparatorRef(
                role="matched_noiseless_control",
                method_id="layerwise_bmpd_noiseless",
                configuration_schema_version="yaqs.state_preparation.phase2.training_pipeline.v1",
                configuration_checksum=NOISELESS_CONFIGURATION,
                matched_to_configuration_checksum=V2_CONFIGURATION,
                matching_projection_checksum=MATCHING_PROJECTION,
            ),
        ),
        primary_contrasts=(
            PrimaryContrastBinding(
                contrast_id="noisy_vs_noiseless",
                treatment_configuration_checksum=V2_CONFIGURATION,
                control_configuration_checksum=NOISELESS_CONFIGURATION,
                paired_block_policy_checksum=preregistration.paired_block_policy_checksum,
                matching_projection_checksum=MATCHING_PROJECTION,
            ),
            PrimaryContrastBinding(
                contrast_id="promoted_vs_layerwise_v2_if_distinct",
                treatment_configuration_checksum=PROMOTED_CONFIGURATION,
                control_configuration_checksum=V2_CONFIGURATION,
                paired_block_policy_checksum=preregistration.paired_block_policy_checksum,
                matching_projection_checksum=None,
            ),
        ),
        confirmatory_target_manifest_checksum=target_manifest.content_checksum,
        target_count_by_family=dict.fromkeys(PRIMARY_TARGET_FAMILIES, 24),
        optimization_seed_count=3,
        fixed_test_trajectory_count=256,
        primary_noise_condition=dict(preregistration.primary_noise_condition),
        primary_resource_budget={
            "metric": preregistration.primary_resource_constraint["metric"],
            "cap_per_chain_edge": preregistration.primary_resource_constraint["cap_per_chain_edge"],
            "normalized_compute_cap": 1_000_000.0,
            "reachable_stratum_manifest_checksum": canonical_checksum({"reachable": "pilot"}),
        },
        hyperparameters_checksum=configuration_execution_manifest.content_checksum,
        execution_source_checksum=canonical_checksum({"execution": "frozen"}),
        analysis_template_checksum=preregistration.analysis_template_checksum,
        analysis_source_manifest_checksum=canonical_checksum({"analysis-source": "frozen"}),
        sample_size_design_checksum=canonical_checksum({"sample-size": "pilot-derived"}),
        failure_policy_checksum=preregistration.failure_policy_checksum,
    )


def _evidence(
    seal: FinalConfirmationSeal,
    target_manifest: TargetPopulationManifest,
    run_plan: TrainingRunPlan,
    *,
    promoted_base: float = 0.745,
) -> tuple[tuple[ConfirmatoryResultArtifact, ...], tuple[TrainingJobOutcome, ...]]:
    """Return exact typed successful results and outcomes for the plan."""
    bases = {
        NOISELESS_CONFIGURATION: 0.70,
        V2_CONFIGURATION: 0.75,
        PROMOTED_CONFIGURATION: promoted_base,
    }
    specs = {spec.target_instance_id: spec for spec in target_manifest.instances}
    family_indices = {family_id: index for index, family_id in enumerate(PRIMARY_TARGET_FAMILIES)}
    target_indices: dict[str, int] = {}
    for family_id in PRIMARY_TARGET_FAMILIES:
        family_targets = [spec for spec in target_manifest.instances if spec.family_id == family_id]
        target_indices.update({spec.target_instance_id: index for index, spec in enumerate(family_targets)})
    seeds_by_target = {
        target_id: tuple(
            sorted({job.optimization_seed for job in run_plan.jobs if job.target_instance_id == target_id})
        )
        for target_id in specs
    }
    results: list[ConfirmatoryResultArtifact] = []
    outcomes: list[TrainingJobOutcome] = []
    for job in run_plan.jobs:
        spec = specs[job.target_instance_id]
        configuration_checksum = job.candidate_configuration_checksum
        target_index = target_indices[spec.target_instance_id]
        seed_index = seeds_by_target[spec.target_instance_id].index(job.optimization_seed)
        centered_target = (target_index - 11.5) * 0.0001
        common_effect = centered_target + (seed_index - 1) * 0.0002 + family_indices[spec.family_id] * 0.0001
        interaction = (
            centered_target
            if configuration_checksum == V2_CONFIGURATION
            else -0.5 * centered_target
            if configuration_checksum == PROMOTED_CONFIGURATION
            else 0.0
        )
        fidelity = bases[configuration_checksum] + common_effect + interaction
        source_evaluation = ConfirmatoryEvaluationArtifact.create(
            job,
            seal,
            trajectory_fidelities=(fidelity,) * seal.fixed_test_trajectory_count,
        )
        result_reference_checksum = canonical_checksum({
            "nonproduction_test_result_reference": job.content_checksum,
        })
        result = ConfirmatoryResultArtifact(
            source_evaluation=source_evaluation,
            source_result_reference_checksum=result_reference_checksum,
            source_production_evidence_checksum=canonical_checksum({
                "nonproduction_test_evidence": job.content_checksum,
            }),
        )
        results.append(result)
        outcome = TrainingJobOutcome(
            job_checksum=job.content_checksum,
            status="success",
            result_artifact_checksum=result_reference_checksum,
            exception_type=None,
            message=None,
            attempt=1,
        )
        outcomes.append(outcome)
    return tuple(results), tuple(outcomes)


@pytest.fixture(scope="module")
def preregistration() -> InitialPreregistration:
    """Return the trusted Phase II protocol."""
    return load_initial_preregistration()


@pytest.fixture(scope="module")
def target_manifest(preregistration: InitialPreregistration) -> TargetPopulationManifest:
    """Return the revealed minimum-size confirmatory target manifest."""
    config = build_target_population_config(
        preregistration,
        "confirmatory",
        role_master_entropy_commitment=role_master_entropy_commitment(_CONFIRMATORY_MASTER),
        confirmatory_target_count_by_family=dict.fromkeys(PRIMARY_TARGET_FAMILIES, 24),
    )
    return create_target_population_manifest(config, preregistration, _CONFIRMATORY_MASTER)


@pytest.fixture(scope="module")
def configuration_execution_manifest() -> FinalConfigurationExecutionManifest:
    """Return the exact final executable configuration universe."""
    return _configuration_execution_manifest()


@pytest.fixture(scope="module")
def seal(
    preregistration: InitialPreregistration,
    target_manifest: TargetPopulationManifest,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
) -> FinalConfirmationSeal:
    """Return the synthetic distinct-promoted final seal."""
    return _seal(preregistration, target_manifest, configuration_execution_manifest)


@pytest.fixture(scope="module")
def run_plan(
    seal: FinalConfirmationSeal,
    target_manifest: TargetPopulationManifest,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
) -> TrainingRunPlan:
    """Return the only confirmatory plan authorized by the seal and revealed targets."""
    return build_paper_confirm_plan(
        seal=seal,
        target_manifest=target_manifest,
        configuration_execution_manifest=configuration_execution_manifest,
    )


@pytest.fixture(scope="module")
def evidence(
    seal: FinalConfirmationSeal,
    target_manifest: TargetPopulationManifest,
    run_plan: TrainingRunPlan,
) -> tuple[tuple[ConfirmatoryResultArtifact, ...], tuple[TrainingJobOutcome, ...]]:
    """Return the complete source-linked confirmatory evidence universe."""
    return _evidence(seal, target_manifest, run_plan)


@pytest.fixture(scope="module")
def job_outcomes(
    evidence: tuple[tuple[ConfirmatoryResultArtifact, ...], tuple[TrainingJobOutcome, ...]],
) -> tuple[TrainingJobOutcome, ...]:
    """Return one typed outcome for every exact confirmatory job."""
    return evidence[1]


@pytest.fixture(scope="module")
def result_artifacts(
    evidence: tuple[tuple[ConfirmatoryResultArtifact, ...], tuple[TrainingJobOutcome, ...]],
) -> tuple[ConfirmatoryResultArtifact, ...]:
    """Return one typed fresh-test result for every successful job."""
    return evidence[0]


@pytest.fixture(scope="module")
def custody(
    target_manifest: TargetPopulationManifest,
    run_plan: TrainingRunPlan,
    job_outcomes: tuple[TrainingJobOutcome, ...],
    result_artifacts: tuple[ConfirmatoryResultArtifact, ...],
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
) -> Custody:
    """Return all mandatory typed custody inputs for primary analysis."""
    return target_manifest, run_plan, job_outcomes, result_artifacts, configuration_execution_manifest


@pytest.fixture(scope="module")
def production_bundle(tmp_path_factory: pytest.TempPathFactory) -> ProductionBundle:
    """Issue one full production-schema analysis from typed reopened custody.

    Returns:
        The issued result, authority, reopened custodies, and failure-manifest roots.
    """
    root = tmp_path_factory.mktemp("wp22_primary_production")
    context = build_confirmation_context_fixture(root).context
    outcomes: dict[str, TrainingJobOutcome] = {}
    custodies: dict[str, ProductionResultCustody] = {}
    failed_job_checksum = context.plan.jobs[-1].content_checksum
    failure_manifest_without_partial = ""
    for index, job in enumerate(context.plan.jobs):
        if job.content_checksum == failed_job_checksum:
            values = None
        elif index == 0:
            values = (0.1,) * context.final_seal.fixed_test_trajectory_count
        else:
            fidelity = (
                0.75
                if job.candidate_configuration_checksum == context.final_seal.promoted_configuration_checksum
                else 0.70
            )
            values = (fidelity,) * context.final_seal.fixed_test_trajectory_count
        attempt_custody, without_partial = _real_confirmation_custody(job, context, values)
        custodies[job.content_checksum] = attempt_custody
        outcomes[job.content_checksum] = TrainingJobOutcome(
            job_checksum=job.content_checksum,
            status=attempt_custody.reference.status,
            result_artifact_checksum=(
                attempt_custody.reference.content_checksum if attempt_custody.reference.status == "success" else None
            ),
            exception_type=None if attempt_custody.reference.status == "success" else "runtime_error",
            message=None if attempt_custody.reference.status == "success" else "redacted test failure",
            attempt=1,
        )
        if without_partial is not None:
            failure_manifest_without_partial = without_partial

    def history(_directory: Path, job: TrainingJob) -> tuple[TrainingJobOutcome, ...]:
        return (outcomes[job.content_checksum],)

    def reopen(
        job: TrainingJob,
        outcome: TrainingJobOutcome,
        _directory: Path,
        authority: ConfirmationExecutionContext,
    ) -> ProductionResultCustody:
        assert authority is context
        assert outcome is outcomes[job.content_checksum]
        return custodies[job.content_checksum]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(primary_analysis_module, "load_training_job_outcome_history", history)
        monkeypatch.setattr(primary_analysis_module, "reopen_confirmatory_production_attempt", reopen)
        result = analyze_production_confirmatory_results(context, root / "results")
    assert failure_manifest_without_partial
    return result, context, custodies, failed_job_checksum, failure_manifest_without_partial


def _analyze(
    seal: FinalConfirmationSeal,
    custody: Custody,
    *,
    outcomes: tuple[TrainingJobOutcome, ...] | list[TrainingJobOutcome] | None = None,
    results: tuple[ConfirmatoryResultArtifact, ...] | list[ConfirmatoryResultArtifact] | None = None,
) -> AnalysisResult:
    """Run analysis with the mandatory typed custody bridge.

    Returns:
        The validated primary-analysis result.
    """
    target_manifest, run_plan, default_outcomes, default_results, configuration_execution_manifest = custody
    return _analyze_confirmatory_statistics(
        seal,
        configuration_execution_manifest,
        target_manifest,
        run_plan,
        default_outcomes if outcomes is None else outcomes,
        default_results if results is None else results,
    )


def _failed_job(
    outcome: TrainingJobOutcome,
) -> TrainingJobOutcome:
    """Convert one successful job outcome into an exact durable failure.

    Returns:
        The typed failed outcome.
    """
    return TrainingJobOutcome(
        job_checksum=outcome.job_checksum,
        status="failure",
        result_artifact_checksum=None,
        exception_type="optimizer-failed",
        message="synthetic optimizer failure",
        attempt=outcome.attempt,
    )


@pytest.fixture(scope="module")
def baseline_result(
    seal: FinalConfirmationSeal,
    custody: Custody,
) -> AnalysisResult:
    """Return the fully authenticated all-success analysis result."""
    return _analyze(seal, custody)


@pytest.fixture(scope="module")
def observations(baseline_result: AnalysisResult) -> tuple[ConfirmatoryObservation, ...]:
    """Return the mechanically constructed confirmatory observation universe."""
    return baseline_result.observations


def _contrast(result: AnalysisResult, contrast_id: str) -> Mapping[str, object]:
    """Resolve one contrast result by identifier.

    Returns:
        The requested derived contrast mapping.
    """
    return next(item for item in result.contrast_results if item["contrast_id"] == contrast_id)


def _failure_result(result: AnalysisResult, configuration_checksum: str) -> Mapping[str, object]:
    """Resolve one configuration failure-rate result.

    Returns:
        The requested failure-rate endpoint mapping.
    """
    return next(
        item for item in result.failure_rate_results if item["configuration_checksum"] == configuration_checksum
    )


def test_confirmatory_observation_round_trip_and_status_semantics(
    observations: tuple[ConfirmatoryObservation, ...],
) -> None:
    """Cell evidence is sealed and rejects wrong roles and status payloads."""
    observation = observations[0]
    assert ConfirmatoryObservation.from_json(observation.to_json()) == observation
    with pytest.raises(ValueError, match="confirmatory data role"):
        replace(observation, data_role="screening_selection")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="null fidelity"):
        replace(observation, status="failure", failure_code="optimizer-failed")
    with pytest.raises(ValueError, match="cannot carry"):
        replace(observation, failure_code="unexpected")


def test_primary_analysis_is_deterministic_clustered_and_holm_controlled(
    seal: FinalConfirmationSeal,
    custody: Custody,
    baseline_result: AnalysisResult,
) -> None:
    """Paired estimates use target clusters, equal families, and sealed Holm tests."""
    result = baseline_result
    reversed_result = _analyze(
        seal,
        custody,
        outcomes=tuple(reversed(custody[2])),
        results=tuple(reversed(custody[3])),
    )

    assert result.content_checksum == reversed_result.content_checksum
    assert result.observations == reversed_result.observations
    noisy = _contrast(result, "noisy_vs_noiseless")
    promoted = _contrast(result, "promoted_vs_layerwise_v2_if_distinct")
    assert cast("float", noisy["estimate"]) == pytest.approx(0.05)
    assert cast("float", promoted["estimate"]) == pytest.approx(-0.005)
    assert cast("float", noisy["standard_error"]) > 0.0
    assert cast("float", promoted["standard_error"]) > 0.0
    assert cast("float", noisy["confidence_interval_lower"]) < cast("float", noisy["estimate"])
    assert cast("float", noisy["confidence_interval_upper"]) > cast("float", noisy["estimate"])
    assert noisy["claim"] == "superior"
    assert promoted["claim"] == "noninferior"
    assert noisy["reject_null"] is True
    assert promoted["reject_null"] is True
    assert result.analysis_policy["family_weighting"] == "equal"
    assert result.analysis_policy["cluster_unit"] == "target_instance"


def test_holm_stops_after_a_nonrejected_sealed_noninferiority_contrast(
    seal: FinalConfirmationSeal,
    custody: Custody,
) -> None:
    """A promoted effect below its -0.01 margin produces no confirmatory claim."""
    results, outcomes = _evidence(seal, custody[0], custody[1], promoted_base=0.735)
    result = _analyze(seal, custody, outcomes=outcomes, results=results)
    noisy = _contrast(result, "noisy_vs_noiseless")
    promoted = _contrast(result, "promoted_vs_layerwise_v2_if_distinct")

    assert noisy["claim"] == "superior"
    assert promoted["claim"] == "not_established"
    assert promoted["reject_null"] is False
    assert cast("float", promoted["estimate"]) == pytest.approx(-0.015)
    assert cast("float", noisy["holm_threshold"]) == pytest.approx(0.025)


def test_failures_are_retained_as_zero_in_itt_and_failure_endpoint(
    seal: FinalConfirmationSeal,
    custody: Custody,
    baseline_result: AnalysisResult,
) -> None:
    """A failed cell remains in both the fidelity contrast and failure endpoint."""
    failed_row = next(item for item in baseline_result.observations if item.configuration_checksum == V2_CONFIGURATION)
    original_fidelity = cast("float", failed_row.fresh_test_noisy_fidelity)
    outcomes = list(custody[2])
    outcome_index = next(index for index, item in enumerate(outcomes) if item.job_checksum == failed_row.job_checksum)
    outcomes[outcome_index] = _failed_job(outcomes[outcome_index])
    results = [item for item in custody[3] if item.job_checksum != failed_row.job_checksum]
    result = _analyze(seal, custody, outcomes=outcomes, results=results)
    noisy = _contrast(result, "noisy_vs_noiseless")
    v2_failure = _failure_result(result, V2_CONFIGURATION)

    assert cast("float", noisy["estimate"]) == pytest.approx(0.05 - original_fidelity / 288.0)
    assert v2_failure["failure_count"] == 1
    assert cast("float", v2_failure["failure_rate"]) == pytest.approx(1.0 / 288.0)
    assert cast("float", v2_failure["standard_error"]) > 0.0
    assert cast("float", v2_failure["confidence_interval_upper"]) > cast("float", v2_failure["failure_rate"])


def test_failure_intervals_remain_finite_sample_conservative_at_boundaries(
    seal: FinalConfirmationSeal,
    custody: Custody,
    baseline_result: AnalysisResult,
) -> None:
    """All-success and all-failure samples retain nonzero Wilson uncertainty floors."""
    all_success = _failure_result(baseline_result, V2_CONFIGURATION)
    assert all_success["failure_count"] == 0
    assert cast("float", all_success["failure_rate"]) == pytest.approx(0.0)
    assert cast("float", all_success["standard_error"]) == pytest.approx(0.0)
    assert cast("float", all_success["confidence_interval_lower"]) == pytest.approx(0.0)
    assert 0.0 < cast("float", all_success["confidence_interval_upper"]) < 1.0
    assert all_success["effective_target_cluster_count"] == 96
    assert all_success["confidence_interval_method"] == ("cluster_normal_enveloped_by_target_cluster_wilson_score")

    outcomes = list(custody[2])
    outcome_index = {outcome.job_checksum: index for index, outcome in enumerate(outcomes)}
    failed_job_checksums: set[str] = set()
    for row in baseline_result.observations:
        if row.configuration_checksum != V2_CONFIGURATION:
            continue
        index = outcome_index[row.job_checksum]
        outcomes[index] = _failed_job(outcomes[index])
        failed_job_checksums.add(row.job_checksum)
    results = [item for item in custody[3] if item.job_checksum not in failed_job_checksums]
    all_failure = _failure_result(
        _analyze(seal, custody, outcomes=outcomes, results=results),
        V2_CONFIGURATION,
    )
    assert all_failure["failure_count"] == 288
    assert cast("float", all_failure["failure_rate"]) == pytest.approx(1.0)
    assert cast("float", all_failure["standard_error"]) == pytest.approx(0.0)
    assert 0.0 < cast("float", all_failure["confidence_interval_lower"]) < 1.0
    assert cast("float", all_failure["confidence_interval_upper"]) == pytest.approx(1.0)


def test_primary_analysis_requires_typed_results_not_caller_authored_fidelity(
    seal: FinalConfirmationSeal,
    custody: Custody,
    baseline_result: AnalysisResult,
) -> None:
    """Opaque checksums, copied rows, and unbound fidelity cannot cross custody."""
    artifact = custody[3][0]
    assert ConfirmatoryResultArtifact.from_json(artifact.to_json()) == artifact
    assert ConfirmatoryEvaluationArtifact.from_json(artifact.source_evaluation.to_json()) == artifact.source_evaluation
    raw = artifact.source_evaluation.trajectory_evidence
    fabricated_raw = replace(
        raw,
        trajectory_fidelities=(min(1.0, raw.trajectory_fidelities[0] + 0.1), *raw.trajectory_fidelities[1:]),
    )
    fabricated = ConfirmatoryResultArtifact(
        source_evaluation=replace(artifact.source_evaluation, trajectory_evidence=fabricated_raw),
        source_result_reference_checksum=artifact.source_result_reference_checksum,
        source_production_evidence_checksum=artifact.source_production_evidence_checksum,
    )
    detached_statistics = _analyze(seal, custody, results=[fabricated, *custody[3][1:]])
    assert not hasattr(detached_statistics, "to_dict")
    with pytest.raises(TypeError, match="ProductionResultCustody"):
        ConfirmatoryResultArtifact.create(
            custody[1].jobs[0],
            seal,
            cast("ProductionResultCustody", fabricated.source_evaluation),
            cast("ConfirmationExecutionContext", object()),
        )
    with pytest.raises(TypeError, match="verified production custody"):
        PrimaryAnalysisResult(
            final_seal=seal,
            target_manifest_checksum=custody[0].content_checksum,
            run_plan_checksum=custody[1].content_checksum,
            job_outcomes=detached_statistics.job_outcomes,
            production_attempt_receipts=(),
            confirmatory_results=detached_statistics.confirmatory_results,
            observations=detached_statistics.observations,
            _construction_authority=object(),
        )

    copied = [custody[3][1], *custody[3][1:]]
    with pytest.raises(ValueError, match="duplicate successful results"):
        _analyze(seal, custody, results=copied)

    document = artifact.to_dict()
    document["fresh_test_noisy_fidelity"] = min(1.0, artifact.fresh_test_noisy_fidelity + 0.1)
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })
    with pytest.raises(ValueError, match="not derived from its raw trajectory evidence"):
        ConfirmatoryResultArtifact.from_dict(document)

    with pytest.raises(TypeError, match="ConfirmatoryResultArtifact"):
        _analyze_confirmatory_statistics(
            seal,
            custody[4],
            custody[0],
            custody[1],
            custody[2],
            cast("tuple[ConfirmatoryResultArtifact, ...]", baseline_result.observations),
        )


def test_public_production_analysis_binds_all_attempts_and_wp22e_float64_mean(
    production_bundle: ProductionBundle,
) -> None:
    """The public schema covers failures and preserves the exact float64 mean."""
    result, context, custodies, failed_job_checksum, manifest_without_partial = production_bundle
    assert len(result.production_attempt_receipts) == len(context.plan.jobs)
    assert len(result.job_outcomes) == len(context.plan.jobs)
    assert len(result.observations) == len(context.plan.jobs)
    assert not hasattr(phase2, "analyze_confirmatory_results")
    assert "analyze_confirmatory_results" not in primary_analysis_module.__all__

    failed_receipt = next(
        receipt for receipt in result.production_attempt_receipts if receipt.job_checksum == failed_job_checksum
    )
    assert failed_receipt.status == "failure"
    assert failed_receipt.raw_trajectory_document_checksum is None
    assert failed_receipt.result_reference.manifest_content_checksum != manifest_without_partial
    assert failed_job_checksum not in {item.job_checksum for item in result.confirmatory_results}

    first_job = context.plan.jobs[0]
    values = custodies[first_job.content_checksum].trajectory_fidelities
    assert values is not None
    assert float(statistics.fmean(values)).hex() != float(np.mean(np.asarray(values, dtype=np.float64))).hex()
    expected = float(np.mean(np.asarray(values, dtype=np.float64)))
    observation = next(item for item in result.observations if item.job_checksum == first_job.content_checksum)
    assert observation.fresh_test_noisy_fidelity is not None
    assert float(observation.fresh_test_noisy_fidelity).hex() == expected.hex()
    source_result = next(
        item for item in result.confirmatory_results if item.job_checksum == first_job.content_checksum
    )
    assert float(source_result.fresh_test_noisy_fidelity).hex() == expected.hex()
    assert ConfirmatoryProductionAttemptReceipt.from_json(failed_receipt.to_json()) == failed_receipt
    assert PrimaryAnalysisResult.from_json(result.to_json()) == result


@pytest.mark.parametrize("root_name", ["raw", "diagnostic", "resource", "reference"])
def test_production_attempt_receipt_rejects_resealed_root_tampering(
    production_bundle: ProductionBundle,
    root_name: str,
) -> None:
    """Every raw, diagnostic, resource, and reference root is mechanically closed."""
    result = production_bundle[0]
    receipt = next(item for item in result.production_attempt_receipts if item.status == "success")
    document = receipt.to_dict()
    changed_checksum = canonical_checksum({"tampered_receipt_root": root_name})
    if root_name == "raw":
        document["raw_trajectory_document_checksum"] = changed_checksum
    elif root_name == "diagnostic":
        document["pilot_diagnostic_checksums"] = [changed_checksum]
    elif root_name == "resource":
        document["resource_document_checksum"] = changed_checksum
    else:
        reference = cast("dict[str, object]", document["result_reference"])
        reference["manifest_content_checksum"] = changed_checksum
        reference["content_checksum"] = canonical_checksum({
            key: value for key, value in reference.items() if key != "content_checksum"
        })
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })
    with pytest.raises(ValueError, match=r"receipt|custody|exact first attempt"):
        ConfirmatoryProductionAttemptReceipt.from_dict(document)


def test_production_custody_replays_wp22e_float64_mean_and_rejects_resealed_raw_bytes(
    seal: FinalConfirmationSeal,
    run_plan: TrainingRunPlan,
    production_bundle: ProductionBundle,
    tmp_path: Path,
) -> None:
    """A real WP22E attempt uses one mean algorithm and immutable raw bytes."""
    job = run_plan.jobs[0]
    request = job.confirm_execution_request
    assert request is not None
    values = (0.1,) * request.fixed_test_trajectory_count
    assert float(statistics.fmean(values)).hex() != float(np.mean(values)).hex()
    executor = SyntheticConfirmationExecutor(
        SyntheticConfirmationFixture(
            request_checksum=request.content_checksum,
            trajectory_fidelities=values,
        ),
    )
    job_directory = tmp_path / job.output_path
    reference = executor.execute(
        request,
        job_directory,
        JobExecutionControls(resume=False, overwrite=False),
    )
    outcome = TrainingJobOutcome(
        job_checksum=job.content_checksum,
        status="success",
        result_artifact_checksum=reference.content_checksum,
        exception_type=None,
        message=None,
        attempt=1,
    )
    custody = reopen_confirmatory_production_attempt(job, outcome, job_directory)
    mean_fidelity = custody.mean_fidelity
    assert isinstance(mean_fidelity, float)
    recorded_fidelity = custody.production_evidence.derived_metrics["noisy_fidelity"]
    assert isinstance(recorded_fidelity, float)
    assert mean_fidelity.hex() == recorded_fidelity.hex()
    assert custody.production_evidence.artifact_kind == "synthetic_confirmation"
    assert custody.production_evidence.derived_metrics["promotion_eligible"] is False
    with pytest.raises(ValueError, match="first-attempt"):
        reopen_confirmatory_production_attempt(job, replace(outcome, attempt=2), job_directory)

    with pytest.raises(ValueError, match="Synthetic custody"):
        ConfirmatoryResultArtifact.create(
            job,
            seal,
            custody,
            production_bundle[1],
        )

    raw_ref = custody.production_evidence.raw_trajectory_ref
    assert raw_ref is not None
    raw_path = job_directory / raw_ref.path
    raw_document = cast("dict[str, object]", json.loads(raw_path.read_text(encoding="utf-8")))
    raw_payload = cast("dict[str, object]", raw_document["payload"])
    raw_values = cast("list[float]", raw_payload["trajectory_fidelities"])
    raw_values[0] = 0.2
    raw_document["content_checksum"] = canonical_checksum({
        key: value for key, value in raw_document.items() if key != "content_checksum"
    })
    raw_path.write_text(canonical_json(raw_document), encoding="utf-8")
    with pytest.raises(ValueError, match=r"checksum|byte"):
        reopen_confirmatory_production_attempt(job, outcome, job_directory)


def test_public_production_analysis_rejects_mixed_real_and_synthetic_custody(
    production_bundle: ProductionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One synthetic attempt poisons an otherwise real reopened custody sequence."""
    result, context, real_custodies, _failed_job, _without_partial = production_bundle
    first_job, synthetic_job = context.plan.jobs[:2]
    request = synthetic_job.confirm_execution_request
    assert request is not None
    executor = SyntheticConfirmationExecutor(
        SyntheticConfirmationFixture(
            request_checksum=request.content_checksum,
            trajectory_fidelities=(0.1,) * request.fixed_test_trajectory_count,
        )
    )
    synthetic_directory = tmp_path / "synthetic"
    reference = executor.execute(
        request,
        synthetic_directory,
        JobExecutionControls(resume=False, overwrite=False),
    )
    synthetic_outcome = TrainingJobOutcome(
        job_checksum=synthetic_job.content_checksum,
        status="success",
        result_artifact_checksum=reference.content_checksum,
        exception_type=None,
        message=None,
        attempt=1,
    )
    synthetic_custody = reopen_confirmatory_production_attempt(
        synthetic_job,
        synthetic_outcome,
        synthetic_directory,
    )
    first_outcome = next(item for item in result.job_outcomes if item.job_checksum == first_job.content_checksum)
    outcomes = {
        first_job.content_checksum: first_outcome,
        synthetic_job.content_checksum: synthetic_outcome,
    }
    custodies = {
        first_job.content_checksum: real_custodies[first_job.content_checksum],
        synthetic_job.content_checksum: synthetic_custody,
    }
    monkeypatch.setattr(
        primary_analysis_module,
        "load_training_job_outcome_history",
        lambda _directory, job: (outcomes[job.content_checksum],),
    )
    monkeypatch.setattr(
        primary_analysis_module,
        "reopen_confirmatory_production_attempt",
        lambda job, _outcome, _directory, _authority: custodies[job.content_checksum],
    )
    with pytest.raises(ValueError, match="Synthetic custody"):
        analyze_production_confirmatory_results(context, tmp_path / "mixed-results")


def test_public_production_analysis_rejects_a_second_terminal_attempt(
    production_bundle: ProductionBundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later terminal retry cannot be hidden behind the authoritative first row."""
    result, context, _custodies, _failed_job, _without_partial = production_bundle
    first_job = context.plan.jobs[0]
    first_outcome = next(item for item in result.job_outcomes if item.job_checksum == first_job.content_checksum)
    retry = replace(first_outcome, attempt=2)
    monkeypatch.setattr(
        primary_analysis_module,
        "load_training_job_outcome_history",
        lambda _directory, _job: (first_outcome, retry),
    )
    with pytest.raises(ValueError, match="exactly one authoritative terminal outcome"):
        analyze_production_confirmatory_results(context, tmp_path / "retried-results")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("source", "execution links"),
        ("executable_binding", "execution links"),
        ("scheduled_program", "execution links"),
        ("evaluation_policy", "execution links"),
        ("target_identity", "target aliases"),
    ],
)
def test_confirmatory_custody_rejects_resealed_request_aliases(
    run_plan: TrainingRunPlan,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Storage-valid resealing cannot replace any v2 request execution link."""
    job = run_plan.jobs[0]
    job_directory = tmp_path / mutation
    outcome = _publish_resealed_confirmation_alias_attack(job, job_directory, mutation)
    with pytest.raises(ValueError, match=message):
        reopen_confirmatory_production_attempt(job, outcome, job_directory)


def test_primary_analysis_requires_exact_generated_plan_and_typed_outcome_universe(
    seal: FinalConfirmationSeal,
    custody: Custody,
) -> None:
    """A changed plan job or missing outcome cannot masquerade as confirmatory execution."""
    target_manifest, run_plan, outcomes, results, configuration_execution_manifest = custody
    request = run_plan.jobs[0].confirm_execution_request
    assert request is not None
    changed_request = replace(request, evaluation_seed=request.evaluation_seed + 1)
    changed_job = replace(
        run_plan.jobs[0],
        evaluation_seed=run_plan.jobs[0].evaluation_seed + 1,
        confirm_execution_request=changed_request,
    )
    changed_plan = replace(run_plan, jobs=(changed_job, *run_plan.jobs[1:]))
    with pytest.raises(ValueError, match="ConfirmExecutionRequest"):
        _analyze_confirmatory_statistics(
            seal,
            configuration_execution_manifest,
            target_manifest,
            changed_plan,
            outcomes,
            results,
        )

    with pytest.raises(ValueError, match="exact seal-generated job universe"):
        _analyze_confirmatory_statistics(
            seal,
            configuration_execution_manifest,
            target_manifest,
            run_plan,
            outcomes[:-1],
            results,
        )

    copied = (*outcomes, outcomes[0])
    with pytest.raises(ValueError, match="duplicate outcomes"):
        _analyze_confirmatory_statistics(
            seal,
            configuration_execution_manifest,
            target_manifest,
            run_plan,
            copied,
            results,
        )

    retried = (replace(outcomes[0], attempt=2), *outcomes[1:])
    with pytest.raises(ValueError, match="first terminal"):
        _analyze_confirmatory_statistics(
            seal,
            configuration_execution_manifest,
            target_manifest,
            run_plan,
            retried,
            results,
        )

    with pytest.raises(ValueError, match="exactly the successful"):
        _analyze_confirmatory_statistics(
            seal,
            configuration_execution_manifest,
            target_manifest,
            run_plan,
            outcomes,
            results[:-1],
        )


def test_primary_analysis_rejects_unsealed_analysis_choices(
    seal: FinalConfirmationSeal,
    custody: Custody,
) -> None:
    """Changing the analysis-template checksum cannot create a primary result."""
    changed = replace(seal, analysis_template_checksum=canonical_checksum({"analysis": "changed"}))
    with pytest.raises(ValueError, match="analysis template"):
        _analyze(changed, custody)


def test_resealed_caller_authored_analysis_is_recomputed(
    production_bundle: ProductionBundle,
) -> None:
    """A resealed caller-supplied estimate is rejected against raw-row recomputation."""
    baseline_result = production_bundle[0]
    changed_observation = baseline_result.to_dict()
    observations = cast("list[dict[str, object]]", changed_observation["observations"])
    successful = next(item for item in observations if item["status"] == "success")
    successful["fresh_test_noisy_fidelity"] = 0.999
    successful["content_checksum"] = canonical_checksum({
        key: value for key, value in successful.items() if key != "content_checksum"
    })
    changed_observation["content_checksum"] = canonical_checksum({
        key: value for key, value in changed_observation.items() if key != "content_checksum"
    })
    with pytest.raises(ValueError, match="not dereferenced from its typed raw result source"):
        PrimaryAnalysisResult.from_dict(changed_observation)

    document = baseline_result.to_dict()
    contrasts = cast("list[dict[str, object]]", document["contrast_results"])
    contrasts[0]["estimate"] = 0.999
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })

    with pytest.raises(ValueError, match="not derived"):
        PrimaryAnalysisResult.from_dict(document)
