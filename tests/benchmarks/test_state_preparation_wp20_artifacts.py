# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Artifact-boundary tests for publishable WP20 optimizer competitors."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.phase2.artifact_codecs import StageParameterCheckpoint
from benchmarks.state_preparation.phase2.artifacts import (
    Phase2ArtifactStore,
    Phase2ArtifactVerificationError,
    StageExecutionEvidence,
)
from benchmarks.state_preparation.phase2.canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
    seal_mapping,
    thaw_json_mapping,
)
from benchmarks.state_preparation.phase2.competitor_optimizers import (
    BMPDCompetitorStageRunner,
    build_spsa_fixed_template,
)
from benchmarks.state_preparation.phase2.noisy_krotov import (
    KrotovWorkLedger,
    NoisyKrotovCheckpointSelection,
    NoisyKrotovObjectiveBinding,
    NoisyKrotovResumeState,
)
from benchmarks.state_preparation.phase2.protocol import load_initial_preregistration
from benchmarks.state_preparation.phase2.resumability import ExecutionSourceEntry, ResumabilityFingerprint
from benchmarks.state_preparation.phase2.targets import (
    authorize_target_materialization,
    build_target_population_config,
    materialize_target_population,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.wp20_resources import PairedBlockIdentity, TrainingRandomnessRecord
from mqt.yaqs.optimization import KrotovFixedMapEnsemble
from tests.benchmarks.test_state_preparation_phase2_pipeline import _screening_target_manifest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineConfig


def _fingerprint(pipeline: TrainingPipelineConfig) -> ResumabilityFingerprint:
    """Return a complete compact fingerprint for the focused artifact store."""
    entries = tuple(
        ExecutionSourceEntry(
            role=role,
            repository_path=path,
            starting_git_blob_id=character * 40,
            content_checksum=f"sha256:{character * 64}",
        )
        for role, path, character in (
            ("execution_source", "benchmarks/wp20_artifact_test.py", "a"),
            ("lockfile", "uv.lock", "b"),
            ("sealed_input", "benchmarks/state_preparation_preregistration.json", "c"),
        )
    )
    return ResumabilityFingerprint(
        starting_commit="d" * 40,
        pipeline_prefix_id=pipeline.prefix_id(0),
        dependency_versions={"numpy": "2.4.6", "yaqs": "0.5.0"},
        entries=entries,
    )


@pytest.fixture(scope="module")
def competitor_case() -> tuple[TrainingPipelineConfig, StageExecutionEvidence]:
    """Execute one genuine target-bound noisy SPSA stage.

    Returns:
        The resolved pipeline and its publishable one-stage evidence.
    """
    master_entropy = bytes(reversed(range(32)))
    preregistration = load_initial_preregistration()
    population = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(master_entropy),
        population_scope="primary_q6",
    )
    manifest = _screening_target_manifest()
    authorization = authorize_target_materialization(
        preregistration,
        population,
        manifest,
        master_entropy,
    )
    target = materialize_target_population(
        population,
        preregistration,
        manifest,
        master_entropy,
        authorization,
    ).targets[0]
    target_spec = next(item for item in manifest.instances if item.target_instance_id == target.target_instance_id)
    template = build_spsa_fixed_template(
        iteration_budget=1,
        training_trajectory_count=1,
        checkpoint_validation_trajectory_count=1,
    )
    pipeline = template.resolve(
        target_namespace="phase2",
        target_manifest=manifest,
        target_instance_id=target_spec.target_instance_id,
        target_population_manifest_checksum=manifest.content_checksum,
        target_instance_spec_checksum=target_spec.content_checksum,
        target_family_id=target_spec.family_id,
        target_stratum_id=target_spec.stratum_id,
        qubit_count=target_spec.qubit_count,
        optimization_block_id="wp20_artifact_boundary_test",
        optimization_seed=123,
        data_role="screening_selection",
    )
    stage = pipeline.stages[0]
    evidence = BMPDCompetitorStageRunner(pipeline, target)(stage, None)
    return pipeline, evidence


def _copy_evidence(
    evidence: StageExecutionEvidence,
    *,
    training_summary: Mapping[str, object],
    training_ensembles: Sequence[KrotovFixedMapEnsemble],
    objective_binding: NoisyKrotovObjectiveBinding | None,
    final_parameters: np.ndarray | None = None,
    selected_parameters: np.ndarray | None = None,
    trace: Sequence[Mapping[str, object]] | None = None,
) -> StageExecutionEvidence:
    """Reconstruct evidence while varying one artifact-boundary input.

    Returns:
        A newly validated immutable stage-evidence value.
    """
    return StageExecutionEvidence(
        stage=evidence.stage,
        source_parameters=None,
        initial_parameters=evidence.initial_parameters,
        final_parameters=evidence.final_parameters if final_parameters is None else final_parameters,
        selected_parameters=evidence.selected_parameters if selected_parameters is None else selected_parameters,
        selected_global_iteration=evidence.selected_global_iteration,
        completed_global_iteration=evidence.completed_global_iteration,
        selected_checkpoint_validation_fidelity=evidence.selected_checkpoint_validation_fidelity,
        circuit_binding_checksum=evidence.circuit_binding_checksum,
        provider_checksum=evidence.provider_checksum,
        objective_checksum=evidence.objective_checksum,
        objective_binding=objective_binding,
        trace=evidence.trace if trace is None else trace,
        training_ensembles=training_ensembles,
        checkpoint_validation_ensembles=evidence.checkpoint_validation_ensembles,
        normalized_work=evidence.normalized_work,
        training_summary=training_summary,
        checkpoint_validation_summary=evidence.checkpoint_validation_summary,
        circuit_topology=evidence.circuit_topology,
        circuit_statistics=evidence.circuit_statistics,
        optimizer_state=evidence.optimizer_state,
        cumulative_cross_trajectory_pairings=evidence.cumulative_cross_trajectory_pairings,
    )


def test_competitor_maps_objective_and_execution_reopen_as_one_artifact(
    tmp_path: Path,
    competitor_case: tuple[TrainingPipelineConfig, StageExecutionEvidence],
) -> None:
    """The store persists and re-verifies the complete target-bound SPSA stage."""
    pipeline, evidence = competitor_case
    assert isinstance(evidence.objective_binding, NoisyKrotovObjectiveBinding)
    assert isinstance(evidence.training_ensembles, tuple)
    assert len(evidence.training_ensembles) == 1
    assert isinstance(evidence.checkpoint_validation_ensembles, tuple)
    assert len(evidence.checkpoint_validation_ensembles) == 1

    store = Phase2ArtifactStore(tmp_path / "wp20_competitor", pipeline, _fingerprint(pipeline))
    artifact = store.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=1024)
    assert tuple(item.role for item in artifact.fixed_map_artifacts) == (
        "training_trajectory",
        "checkpoint_validation",
    )
    execution_checksum = cast("str", evidence.training_summary["competitor_execution_checksum"])
    assert store.load_stage_checkpoint(0).stage_execution_checksum == execution_checksum

    reopened = Phase2ArtifactStore(store.output_directory, pipeline, store.fingerprint, resume=True)
    assert reopened.completed_stage_count == 1
    assert reopened.load_stage_checkpoint(0).stage_execution_checksum == execution_checksum


def test_resealed_competitor_checkpoint_cannot_add_krotov_resume_state(
    tmp_path: Path,
    competitor_case: tuple[TrainingPipelineConfig, StageExecutionEvidence],
) -> None:
    """Competitor checkpoints cannot masquerade as resumable Krotov state."""
    pipeline, evidence = competitor_case
    store = Phase2ArtifactStore(tmp_path / "wp20_resumable_competitor", pipeline, _fingerprint(pipeline))
    artifact = store.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=1024)
    checkpoint = store.load_stage_checkpoint(0)
    assert checkpoint.resume_state_checksum is None

    selection = None
    if checkpoint.selected_checkpoint_validation_fidelity is not None:
        assert checkpoint.circuit_binding_checksum is not None
        assert checkpoint.objective_checksum is not None
        selection = NoisyKrotovCheckpointSelection(
            stage_configuration_checksum=checkpoint.stage_configuration_checksum,
            circuit_binding_checksum=checkpoint.circuit_binding_checksum,
            provider_checksum=checkpoint.provider_checksum,
            objective_checksum=checkpoint.objective_checksum,
            global_iteration=checkpoint.selected_global_iteration,
            validation_fidelity=checkpoint.selected_checkpoint_validation_fidelity,
            theta=checkpoint.selected_theta,
        )
    work = evidence.normalized_work
    resume_state = NoisyKrotovResumeState(
        stage_configuration_checksum=checkpoint.stage_configuration_checksum,
        circuit_binding_checksum=cast("str", checkpoint.circuit_binding_checksum),
        provider_checksum=checkpoint.provider_checksum,
        objective_checksum=cast("str", checkpoint.objective_checksum),
        completed_global_iteration=checkpoint.completed_global_iteration,
        final_parameter_checksum=checkpoint.final_parameter_checksum,
        checkpoint_selection=selection,
        cumulative_work=KrotovWorkLedger(
            objective_evaluations=cast("int", work["objective_evaluations"]),
            gradient_evaluations=cast("int", work["gradient_evaluations"]),
            training_trajectories=cast("int", work["training_trajectories"]),
            checkpoint_validation_trajectories=cast("int", work["checkpoint_validation_trajectories"]),
            test_trajectories=cast("int", work["test_trajectories"]),
            trajectory_gate_applications=cast("int", work["trajectory_gate_applications"]),
        ),
        cumulative_cross_trajectory_pairings=0,
    )
    resumable = StageParameterCheckpoint(
        pipeline_training_id=checkpoint.pipeline_training_id,
        pipeline_prefix_id=checkpoint.pipeline_prefix_id,
        stage_index=checkpoint.stage_index,
        stage_id=checkpoint.stage_id,
        stage_configuration_checksum=checkpoint.stage_configuration_checksum,
        selected_theta=checkpoint.selected_theta,
        final_theta=checkpoint.final_theta,
        selected_global_iteration=checkpoint.selected_global_iteration,
        completed_global_iteration=checkpoint.completed_global_iteration,
        circuit_binding_checksum=checkpoint.circuit_binding_checksum,
        provider_checksum=checkpoint.provider_checksum,
        objective_checksum=checkpoint.objective_checksum,
        stage_execution_checksum=checkpoint.stage_execution_checksum,
        resume_state=resume_state,
    )
    assert resumable.resume_state_checksum is not None
    checkpoint_payload = resumable.to_bytes()
    checkpoint_checksum = resumable.content_checksum
    checkpoint_path = store.output_directory / artifact.stage_result.produced_checkpoint_path
    checkpoint_path.write_bytes(checkpoint_payload)

    result = artifact.stage_result
    checkpoint_provenance_checksum = canonical_checksum({
        "pipeline_prefix_id": result.pipeline_prefix_id,
        "stage_id": result.stage_id,
        "stage_configuration_checksum": result.stage_configuration_checksum,
        "input_checkpoint_checksum": result.input_checkpoint_checksum,
        "input_checkpoint_provenance_checksum": result.input_checkpoint_provenance_checksum,
        "produced_checkpoint_checksum": checkpoint_checksum,
    })
    resealed_result = replace(
        result,
        produced_checkpoint_checksum=checkpoint_checksum,
        checkpoint_provenance_checksum=checkpoint_provenance_checksum,
    )
    resealed_artifact = replace(
        artifact,
        stage_result=resealed_result,
        checkpoint_file_checksum=checkpoint_checksum,
    )
    stage_stream_payload = f"{resealed_artifact.to_json()}\n".encode()
    store.stage_result_stream_path.write_bytes(stage_stream_payload)

    pipeline_result = store.pipeline_result
    assert pipeline_result is not None
    resealed_pipeline_result = replace(
        pipeline_result,
        stage_results=(resealed_result,),
        final_checkpoint_checksum=checkpoint_checksum,
        final_checkpoint_provenance_checksum=checkpoint_provenance_checksum,
    )
    manifest = dict(load_canonical_json_object(store.manifest_path.read_text(encoding="utf-8")))
    manifest.pop("content_checksum")
    manifest["completed_stage_artifact_checksums"] = [resealed_artifact.content_checksum]
    manifest["completed_pipeline_result_checksum"] = resealed_pipeline_result.content_checksum
    stream_checksums = dict(cast("Mapping[str, object]", manifest["canonical_stream_checksums"]))
    stream_checksums[store.stage_result_stream_path.name] = "sha256:" + hashlib.sha256(stage_stream_payload).hexdigest()
    manifest["canonical_stream_checksums"] = stream_checksums
    store.manifest_path.write_text(f"{canonical_json(seal_mapping(manifest))}\n", encoding="utf-8")

    with pytest.raises(Phase2ArtifactVerificationError, match="requires a non-resumable checkpoint"):
        Phase2ArtifactStore(store.output_directory, pipeline, store.fingerprint, resume=True)


def test_competitor_execution_document_tampering_is_rejected(
    competitor_case: tuple[TrainingPipelineConfig, StageExecutionEvidence],
) -> None:
    """A changed execution alias cannot retain the original sealed checksum."""
    _, evidence = competitor_case
    tampered_summary = thaw_json_mapping(evidence.training_summary)
    execution_document = cast("dict[str, object]", tampered_summary["competitor_execution_document"])
    execution_document["final_parameter_checksum"] = canonical_checksum({"tampered": True})
    with pytest.raises(ValueError, match="checksum does not cover"):
        _copy_evidence(
            evidence,
            training_summary=tampered_summary,
            training_ensembles=evidence.training_ensembles,
            objective_binding=evidence.objective_binding,
        )


def test_resealed_competitor_trace_cannot_claim_an_arbitrary_final_vector(
    competitor_case: tuple[TrainingPipelineConfig, StageExecutionEvidence],
) -> None:
    """Persisted update equations, not caller-selected hashes, determine every state."""
    _, evidence = competitor_case
    final = evidence.final_parameters
    final[0] += 0.125
    parameter_checksum = (
        "sha256:" + hashlib.sha256(np.ascontiguousarray(final, dtype=np.dtype("<f8")).tobytes(order="C")).hexdigest()
    )
    summary = thaw_json_mapping(evidence.training_summary)
    document = cast("dict[str, object]", summary["competitor_execution_document"])
    trace = [dict(cast("Mapping[str, object]", row)) for row in cast("Sequence[object]", document["trace"])]
    trace[-1]["parameters"] = final.tolist()
    trace[-1]["parameter_checksum"] = parameter_checksum
    document["trace"] = trace
    document["final_parameter_checksum"] = parameter_checksum
    summary["final_parameter_checksum"] = parameter_checksum
    selected = evidence.selected_parameters
    if evidence.selected_global_iteration == evidence.completed_global_iteration:
        selected = final.copy()
        document["selected_parameter_checksum"] = parameter_checksum
        summary["selected_parameter_checksum"] = parameter_checksum
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })
    summary["competitor_execution_checksum"] = document["content_checksum"]

    with pytest.raises(ValueError, match="resulting parameters"):
        _copy_evidence(
            evidence,
            training_summary=summary,
            training_ensembles=evidence.training_ensembles,
            objective_binding=evidence.objective_binding,
            final_parameters=final,
            selected_parameters=selected,
            trace=trace,
        )


def test_competitor_requires_scheduled_maps_and_genuine_objective(
    competitor_case: tuple[TrainingPipelineConfig, StageExecutionEvidence],
) -> None:
    """Sampled competitors cannot publish empty maps or an unbound callback objective."""
    _, evidence = competitor_case
    with pytest.raises(ValueError, match="fixed maps do not match the configured sampling schedule"):
        _copy_evidence(
            evidence,
            training_summary=evidence.training_summary,
            training_ensembles=(),
            objective_binding=evidence.objective_binding,
        )
    with pytest.raises(ValueError, match="sealed objective provenance"):
        _copy_evidence(
            evidence,
            training_summary=evidence.training_summary,
            training_ensembles=evidence.training_ensembles,
            objective_binding=None,
        )


def test_competitor_rejects_resealed_maps_that_omit_bound_circuit_gates(
    competitor_case: tuple[TrainingPipelineConfig, StageExecutionEvidence],
) -> None:
    """A matching circuit checksum cannot replace complete per-gate map evidence."""
    _, evidence = competitor_case
    original = evidence.training_ensembles[0]
    incomplete = KrotovFixedMapEnsemble(
        role=original.role,
        resolved_seed=original.resolved_seed,
        stage_index=original.stage_index,
        stage_id=original.stage_id,
        stage_configuration_checksum=original.stage_configuration_checksum,
        circuit_checksum=original.circuit_checksum,
        provider_checksum=original.provider_checksum,
        ensemble_index=original.ensemble_index,
        refresh_index=original.refresh_index,
        global_iteration_start=original.global_iteration_start,
        trajectory_maps=[[] for _ in range(original.trajectory_count)],
    )
    summary = thaw_json_mapping(evidence.training_summary)
    document = cast("dict[str, object]", summary["competitor_execution_document"])
    document["training_ensemble_checksums"] = [incomplete.content_checksum]
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })
    summary["competitor_execution_checksum"] = document["content_checksum"]
    summary["training_ensemble_checksums"] = [incomplete.content_checksum]

    with pytest.raises(ValueError, match="do not cover every gate"):
        _copy_evidence(
            evidence,
            training_summary=summary,
            training_ensembles=(incomplete,),
            objective_binding=evidence.objective_binding,
        )


def test_competitor_randomness_record_is_derived_from_complete_stage_evidence(
    competitor_case: tuple[TrainingPipelineConfig, StageExecutionEvidence],
) -> None:
    """Pairing provenance includes optimizer, trajectory, and checkpoint roles."""
    pipeline, evidence = competitor_case
    block = PairedBlockIdentity(
        target_instance_id=pipeline.target_instance_id,
        target_manifest_checksum=pipeline.target_population_manifest_checksum,
        target_spec_checksum=pipeline.target_instance_spec_checksum,
        optimization_block_id=pipeline.optimization_block_id,
        optimization_seed=pipeline.optimization_seed,
        test_noise_id="depolarizing_1s_all",
        test_protocol_checksum=canonical_checksum({"test_protocol": "wp20_artifact_test"}),
        resource_stratum_id=pipeline.template.resource_stratum_id,
    )
    record = TrainingRandomnessRecord.from_stage_evidence(block, pipeline, (evidence,))

    assert record.optimizer_seeds == (pipeline.stages[0].optimizer_seed,)
    assert record.training_seeds == (pipeline.stages[0].training_seed,)
    assert record.checkpoint_validation_seeds == (pipeline.stages[0].checkpoint_validation.seed,)
    assert record.training_ensemble_checksums == tuple(item.content_checksum for item in evidence.training_ensembles)
    assert record.checkpoint_validation_ensemble_checksums == tuple(
        item.content_checksum for item in evidence.checkpoint_validation_ensembles
    )
