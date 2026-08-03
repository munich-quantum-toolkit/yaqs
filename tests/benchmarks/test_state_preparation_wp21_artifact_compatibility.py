# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Backward-compatibility and algorithm-typing tests for WP21 artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

import benchmarks.state_preparation.phase2.artifacts as artifact_module
from benchmarks.state_preparation.phase2.artifacts import (
    PHASE2_STAGE_METADATA_SCHEMA_VERSION,
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
from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineTemplate
from benchmarks.state_preparation.phase2.pruning import PruningStagePolicy
from mqt.yaqs.optimization import KrotovFixedMapEnsemble
from tests.benchmarks.test_state_preparation_phase2_pipeline import (
    _materialization_policy,
    _pipeline,
    _seed_domains,
    _stage_template,
    _template,
)
from tests.benchmarks.test_state_preparation_wp18_artifacts import (
    _checksum,
    _fingerprint,
    _pipeline_with_every_stage_kind,
    _stage_evidence,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineConfig


_STAGE_METADATA_V2 = "yaqs.state_preparation.phase2.stage_metadata.v2"


def _file_checksum(payload: bytes) -> str:
    """Return the artifact checksum of exact file bytes."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _rewrite_stage_metadata_as_v2(store: Phase2ArtifactStore, stage_index: int) -> bytes:
    """Downgrade one valid v3 sidecar and consistently reseal its ledgers.

    Returns:
        The exact legacy sidecar bytes written to the store.
    """
    artifacts = list(store.stage_artifacts)
    artifact = artifacts[stage_index]
    result = artifact.stage_result
    assert result.diagnostic_sidecar_path is not None
    metadata_path = store.output_directory / result.diagnostic_sidecar_path
    metadata = thaw_json_mapping(load_canonical_json_object(metadata_path.read_text(encoding="utf-8")))
    metadata.pop("content_checksum")
    assert metadata.pop("schema_version") == PHASE2_STAGE_METADATA_SCHEMA_VERSION
    assert metadata.pop("map_circuit_binding_checksum") == metadata["circuit_binding_checksum"]
    metadata["schema_version"] = _STAGE_METADATA_V2
    metadata_payload = canonical_json(seal_mapping(metadata)).encode()
    metadata_checksum = _file_checksum(metadata_payload)
    metadata_path.write_bytes(metadata_payload)

    resealed_result = replace(result, diagnostic_sidecar_checksum=metadata_checksum)
    artifacts[stage_index] = replace(
        artifact,
        stage_result=resealed_result,
        metadata_file_checksum=metadata_checksum,
    )
    stage_stream_payload = "".join(f"{item.to_json()}\n" for item in artifacts).encode()
    store.stage_result_stream_path.write_bytes(stage_stream_payload)

    manifest = thaw_json_mapping(load_canonical_json_object(store.manifest_path.read_text(encoding="utf-8")))
    manifest.pop("content_checksum")
    manifest["completed_stage_artifact_checksums"] = [item.content_checksum for item in artifacts]
    stream_checksums = cast("dict[str, object]", manifest["canonical_stream_checksums"])
    stream_checksums[store.stage_result_stream_path.name] = _file_checksum(stage_stream_payload)
    store.manifest_path.write_text(f"{canonical_json(seal_mapping(manifest))}\n", encoding="utf-8")
    return metadata_payload


@pytest.mark.parametrize(
    ("completed_stage_count", "converted_stage_index", "expected_fixed_map_count"),
    [(1, 0, 0), (3, 2, 2)],
)
def test_v2_stage_metadata_reopens_without_rewriting_its_sealed_bytes(
    tmp_path: Path,
    completed_stage_count: int,
    converted_stage_index: int,
    expected_fixed_map_count: int,
) -> None:
    """Both no-map and fixed-map WP18/WP20 v2 sidecars remain reopenable."""
    pipeline = _pipeline_with_every_stage_kind()
    store = Phase2ArtifactStore(
        tmp_path / f"v2_stage_{converted_stage_index}",
        pipeline,
        _fingerprint(pipeline),
    )
    predecessor: NDArray[np.float64] | None = None
    for stage in pipeline.stages[:completed_stage_count]:
        evidence = _stage_evidence(stage, predecessor)
        store.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=1024)
        predecessor = evidence.selected_parameters
    assert len(store.stage_artifacts[converted_stage_index].fixed_map_artifacts) == expected_fixed_map_count

    legacy_payload = _rewrite_stage_metadata_as_v2(store, converted_stage_index)
    reopened = Phase2ArtifactStore(store.output_directory, pipeline, store.fingerprint, resume=True)

    assert reopened.completed_stage_count == completed_stage_count
    metadata_path = reopened.output_directory / cast(
        "str",
        reopened.stage_artifacts[converted_stage_index].stage_result.diagnostic_sidecar_path,
    )
    assert metadata_path.read_bytes() == legacy_payload


def _wp21_random_pipeline() -> TrainingPipelineConfig:
    """Build a compact WP21 pipeline whose final stage is policy-typed pruning.

    Returns:
        A resolved random-pruning pipeline with generic preparation stages.
    """
    base = _template(
        method_id="impact_pruning_crn",
        template_id="wp21_artifact_typing_base",
    )
    preparation_stages = []
    for stage_template in base.stages:
        policy = thaw_json_mapping(stage_template.stage_policy)
        policy["optimizer_id"] = "wp21_artifact_test_optimizer"
        preparation_stages.append(replace(stage_template, stage_policy=policy))
    pruning = _stage_template(
        index=3,
        stage_id="prune_random",
        kind="prune",
        input_topology="bmpd_d2",
        output_topology="bmpd_d2_pruned",
        input_parameters=108,
        output_parameters=90,
        transfer="apply_pruning_mask",
        iterations=0,
        pruning_rule="random",
        pruning_threshold=18.0,
    )
    pruning_policy = thaw_json_mapping(pruning.stage_policy)
    pruning_policy["optimizer_hyperparameters"] = PruningStagePolicy(
        pruning_unit="parameter",
        scoring_objective_kind="none",
        removal_schedule="fixed_count",
        removal_count=18,
        removal_fraction=None,
        relax_after_round=False,
    ).to_mapping()
    typed_pruning = replace(pruning, stage_policy=pruning_policy)
    template = TrainingPipelineTemplate(
        template_id="wp21_artifact_typed_random",
        preregistration_checksum=base.preregistration_checksum,
        target_scope_id=base.target_scope_id,
        ansatz_family=base.ansatz_family,
        method_id="topdown_random",
        method_version="1",
        resource_stratum_id=base.resource_stratum_id,
        stages=(*preparation_stages, typed_pruning),
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )
    return _pipeline(template, block_id="wp21_artifact_typing")


def _predecessor_for_final_stage(
    pipeline: TrainingPipelineConfig,
) -> tuple[NDArray[np.float64], tuple[StageExecutionEvidence, ...]]:
    """Build generic preparation evidence and return its selected handoff.

    Returns:
        The selected predecessor parameters and preparation-stage evidence.
    """
    predecessor: NDArray[np.float64] | None = None
    evidence_rows: list[StageExecutionEvidence] = []
    for stage in pipeline.stages[:-1]:
        evidence = _stage_evidence(stage, predecessor)
        evidence_rows.append(evidence)
        predecessor = evidence.selected_parameters
    assert predecessor is not None
    return predecessor, tuple(evidence_rows)


def test_wp21_policy_and_pruning_execution_evidence_are_exactly_equivalent() -> None:
    """Typed policies require execution seals, and legacy transforms forbid them."""
    typed_pipeline = _wp21_random_pipeline()
    typed_predecessor, _ = _predecessor_for_final_stage(typed_pipeline)
    with pytest.raises(ValueError, match="requires exact pruning execution evidence"):
        _stage_evidence(typed_pipeline.stages[-1], typed_predecessor)

    legacy_pipeline = _pipeline_with_every_stage_kind()
    legacy_predecessor, _ = _predecessor_for_final_stage(legacy_pipeline)
    legacy_evidence = _stage_evidence(legacy_pipeline.stages[-1], legacy_predecessor)
    fake_summary: Mapping[str, object] = {
        "pruning_execution_checksum": _checksum("e"),
        "pruning_execution_document": {"unexpected": "legacy execution seal"},
    }
    with pytest.raises(ValueError, match="requires exact pruning execution evidence"):
        StageExecutionEvidence(
            stage=legacy_evidence.stage,
            source_parameters=legacy_predecessor,
            initial_parameters=legacy_evidence.initial_parameters,
            final_parameters=legacy_evidence.final_parameters,
            selected_parameters=legacy_evidence.selected_parameters,
            selected_global_iteration=legacy_evidence.selected_global_iteration,
            completed_global_iteration=legacy_evidence.completed_global_iteration,
            selected_checkpoint_validation_fidelity=None,
            circuit_binding_checksum=legacy_evidence.circuit_binding_checksum,
            provider_checksum=None,
            objective_checksum=None,
            trace=legacy_evidence.trace,
            training_ensembles=(),
            checkpoint_validation_ensembles=(),
            normalized_work=legacy_evidence.normalized_work,
            training_summary=fake_summary,
            checkpoint_validation_summary=None,
            circuit_topology=legacy_evidence.circuit_topology,
            circuit_statistics=legacy_evidence.circuit_statistics,
        )


def test_fully_sealed_wp21_stage_cannot_omit_pruning_execution_on_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reopen verifier rejects a checksum-consistent pre-fix omission."""
    pipeline = _wp21_random_pipeline()
    predecessor, preparation_evidence = _predecessor_for_final_stage(pipeline)
    store = Phase2ArtifactStore(tmp_path / "wp21_omitted_execution", pipeline, _fingerprint(pipeline))
    for evidence in preparation_evidence:
        store.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=1024)

    with monkeypatch.context() as context:
        context.setattr(artifact_module, "_is_wp21_pruning_stage", lambda _stage: False)
        vulnerable_evidence = _stage_evidence(pipeline.stages[-1], predecessor)
        store.publish_stage(vulnerable_evidence, wall_time_seconds=1.0, peak_memory_bytes=1024)
    assert store.pipeline_result is not None
    assert "pruning_execution_checksum" not in store.stage_artifacts[-1].stage_result.training_summary

    with pytest.raises(Phase2ArtifactVerificationError, match="requires exact pruning execution evidence"):
        Phase2ArtifactStore(store.output_directory, pipeline, store.fingerprint, resume=True)


def test_copy_stage_rejects_a_valid_but_different_circuit_binding(tmp_path: Path) -> None:
    """Parameter continuity cannot hide a circuit change across a copy handoff."""
    pipeline = _pipeline_with_every_stage_kind()
    store = Phase2ArtifactStore(tmp_path / "copy_circuit_handoff", pipeline, _fingerprint(pipeline))
    predecessor: NDArray[np.float64] | None = None
    for stage in pipeline.stages[:2]:
        evidence = _stage_evidence(stage, predecessor)
        store.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=1024)
        predecessor = evidence.selected_parameters
    assert predecessor is not None

    source = _stage_evidence(pipeline.stages[2], predecessor)
    topology = thaw_json_mapping(source.circuit_topology)
    topology.pop("content_checksum")
    gates = cast("list[dict[str, object]]", topology["gates"])
    gates[0]["name"] = "different_valid_gate"
    different_checksum = canonical_checksum(topology)
    topology["content_checksum"] = different_checksum

    def rebind(ensemble: KrotovFixedMapEnsemble) -> KrotovFixedMapEnsemble:
        """Return the same complete maps under the alternate circuit identity.

        Args:
            ensemble: Original stage map ensemble.

        Returns:
            A checksum-sealed ensemble bound to the alternate circuit.
        """
        return KrotovFixedMapEnsemble(
            role=ensemble.role,
            resolved_seed=ensemble.resolved_seed,
            stage_index=ensemble.stage_index,
            stage_id=ensemble.stage_id,
            stage_configuration_checksum=ensemble.stage_configuration_checksum,
            circuit_checksum=different_checksum,
            provider_checksum=ensemble.provider_checksum,
            ensemble_index=ensemble.ensemble_index,
            refresh_index=ensemble.refresh_index,
            global_iteration_start=ensemble.global_iteration_start,
            trajectory_maps=ensemble.replay_maps(),
        )

    forged = StageExecutionEvidence(
        stage=source.stage,
        source_parameters=predecessor,
        initial_parameters=source.initial_parameters,
        final_parameters=source.final_parameters,
        selected_parameters=source.selected_parameters,
        selected_global_iteration=source.selected_global_iteration,
        completed_global_iteration=source.completed_global_iteration,
        selected_checkpoint_validation_fidelity=source.selected_checkpoint_validation_fidelity,
        circuit_binding_checksum=different_checksum,
        provider_checksum=source.provider_checksum,
        objective_checksum=source.objective_checksum,
        objective_binding=source.objective_binding,
        trace=source.trace,
        training_ensembles=tuple(rebind(item) for item in source.training_ensembles),
        checkpoint_validation_ensembles=tuple(rebind(item) for item in source.checkpoint_validation_ensembles),
        normalized_work=source.normalized_work,
        training_summary=source.training_summary,
        checkpoint_validation_summary=source.checkpoint_validation_summary,
        circuit_topology=topology,
        circuit_statistics=source.circuit_statistics,
        optimizer_state=source.optimizer_state,
        cumulative_cross_trajectory_pairings=source.cumulative_cross_trajectory_pairings,
    )
    assert forged.circuit_binding_checksum != store.load_stage_checkpoint(1).circuit_binding_checksum
    with pytest.raises(ValueError, match="preserve its exact source circuit binding"):
        store.publish_stage(forged, wall_time_seconds=1.0, peak_memory_bytes=1024)
