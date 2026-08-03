# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end fresh noisy final-evaluation test for WP21 pruning."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast

import numpy as np

from benchmarks.state_preparation.noise import (
    FIXED_RATE_NOISE_DEFINITION_VERSION,
    create_scaled_standard_noise_provider,
)
from benchmarks.state_preparation.phase2.artifacts import (
    EvaluationEvidenceArtifact,
    FixedMapArtifactRef,
    Phase2ArtifactStore,
)
from benchmarks.state_preparation.phase2.evaluator import (
    MaterializedCircuitPayload,
    ParallelPhase2Evaluator,
    PipelineEvaluationMeasurement,
)
from benchmarks.state_preparation.phase2.execution import Phase2PipelineExecutor
from benchmarks.state_preparation.phase2.historical_reproduction import (
    LayerwiseMaterializedCircuit,
    decode_layerwise_materialized_circuit,
    encode_layerwise_materialized_circuit,
)
from benchmarks.state_preparation.phase2.pipeline import (
    PipelineBenchmarkResult,
    PipelineEvaluationConfig,
    TrainingPipelineResult,
)
from benchmarks.state_preparation.phase2.topdown_pruning import (
    TopDownPruningStageExecution,
    TopDownPruningStageRunner,
    build_topdown_impact_one_shot_template,
)
from mqt.yaqs.optimization import (
    GateNoiseProvider,
    KrotovFixedMapEnsemble,
    KrotovTJMOptions,
    KrotovTruncation,
    noisy_state_preparation_metrics,
    sample_krotov_fixed_map_ensemble,
    state_preparation_metrics,
)
from tests.benchmarks.test_state_preparation_phase2_pipeline import _screening_target_manifest
from tests.benchmarks.test_state_preparation_wp18_artifacts import (
    _fingerprint,
    _screening_materialized_targets,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from benchmarks.state_preparation.phase2.targets import MaterializedTarget


_FINAL_EVALUATION_SEED = 987_654_321


def _load_fixed_map(store: Phase2ArtifactStore, ref: FixedMapArtifactRef) -> KrotovFixedMapEnsemble:
    """Load one persisted fixed-map artifact through its canonical codec.

    Returns:
        The checksum-sealed persisted ensemble.
    """
    payload = (store.output_directory / ref.path).read_text(encoding="utf-8")
    ensemble = KrotovFixedMapEnsemble.from_json(payload)
    assert ensemble.ensemble_id == ref.ensemble_id
    assert ensemble.content_checksum == ref.content_checksum
    return ensemble


def _fresh_noisy_measurement(
    config: PipelineEvaluationConfig,
    runtime_circuit: object,
    target: MaterializedTarget,
) -> PipelineEvaluationMeasurement:
    """Run the real fixed-map final evaluator on one materialized circuit.

    Returns:
        Final-test fidelity and fresh fixed-map evidence.
    """
    assert isinstance(runtime_circuit, LayerwiseMaterializedCircuit)
    assert config.evaluation_seed is not None
    assert config.noise_strength_scale is not None
    assert config.tjm_dt is not None
    binding = runtime_circuit.circuit_binding
    circuit = binding.circuit
    theta = runtime_circuit.selected_parameters
    provider = create_scaled_standard_noise_provider(config.test_noise_id, config.noise_strength_scale)
    truncation = KrotovTruncation(
        max_bond_dim=config.max_bond_dimension,
        svd_threshold=config.svd_threshold,
        trunc_mode=config.truncation_mode,
        min_bond_dim=config.min_bond_dimension,
    )
    options = KrotovTJMOptions(
        num_trajectories=config.trajectory_budget,
        random_seed=config.evaluation_seed,
        dt=config.tjm_dt,
        apply_noise_to="all",
        noisy_gate_indices=binding.noisy_gate_indices,
        trajectory_update="independent",
        differentiate_jump_normalization=False,
        use_crn=False,
    )
    ensemble = sample_krotov_fixed_map_ensemble(
        circuit,
        theta,
        None,
        truncation,
        cast("GateNoiseProvider", provider),
        options,
        role="screening_selection",
        resolved_seed=config.evaluation_seed,
        stage_index=0,
        stage_id="final_evaluation",
        stage_configuration_checksum=config.configuration_checksum,
        circuit_checksum=config.materialized_circuit_checksum,
        provider_checksum=provider.content_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
    )
    target_vector = target.state_vector_copy()
    _loss, noiseless_fidelity = state_preparation_metrics(
        circuit,
        theta,
        target_vector,
        truncation=truncation,
    )
    _noisy_loss, _mean_fidelity, trajectory_fidelities = noisy_state_preparation_metrics(
        circuit,
        theta,
        target_vector,
        None,
        options,
        truncation=truncation,
        iteration=0,
        fixed_noise_maps=ensemble.replay_maps(),
        noise_provider=cast("GateNoiseProvider", provider),
    )
    return PipelineEvaluationMeasurement(
        noiseless_fidelity=float(noiseless_fidelity),
        trajectory_fidelities=tuple(float(value) for value in trajectory_fidelities),
        sampled_nonidentity_events=ensemble.nonidentity_event_count,
        provider_checksum=provider.content_checksum,
        normalized_work={
            "objective_evaluations": 2,
            "gradient_evaluations": 0,
            "training_trajectories": 0,
            "checkpoint_validation_trajectories": 0,
            "test_trajectories": config.trajectory_budget,
            "trajectory_gate_applications": 2 * config.trajectory_budget * len(circuit.gates),
        },
        fixed_map_ensembles=(ensemble,),
        wall_time_seconds=0.0,
        peak_memory_bytes=0,
    )


def test_pruned_noisy_finetuned_circuit_receives_fresh_noisy_final_test(tmp_path: Path) -> None:
    """WP21 evaluates the exact pruned payload with a fresh final-test ensemble."""
    target = _screening_materialized_targets()[0]
    manifest = _screening_target_manifest()
    spec = next(item for item in manifest.instances if item.target_instance_id == target.target_instance_id)
    template = build_topdown_impact_one_shot_template(
        deep_depth=1,
        pruning_unit="parameter",
        removal_count=1,
        scoring_objective_kind="fixed_map_sample_average_fidelity",
        scoring_trajectory_count=1,
        pretrain_iterations=1,
        fine_tune_mode="fixed_crn",
        fine_tune_iterations=1,
        fine_tune_trajectory_count=1,
    )
    pipeline = template.resolve(
        target_namespace="phase2",
        target_manifest=manifest,
        target_instance_id=spec.target_instance_id,
        target_population_manifest_checksum=manifest.content_checksum,
        target_instance_spec_checksum=spec.content_checksum,
        target_family_id=spec.family_id,
        target_stratum_id=spec.stratum_id,
        qubit_count=spec.qubit_count,
        optimization_block_id="wp21_fresh_noisy_final_test",
        optimization_seed=21,
        data_role="screening_selection",
    )
    fingerprint = _fingerprint(pipeline)
    store = Phase2ArtifactStore(tmp_path / "wp21_fresh_final", pipeline, fingerprint)
    runner = TopDownPruningStageRunner(pipeline, target, artifact_store=store)
    training_result = Phase2PipelineExecutor(store).execute(
        runner,
        circuit_statistics=runner.circuit_statistics,
    )
    assert isinstance(training_result, TrainingPipelineResult)

    prune_index = next(stage.stage_index for stage in pipeline.stages if stage.stage_kind == "prune")
    prune_artifact = store.stage_artifacts[prune_index]
    pruning_execution = TopDownPruningStageExecution.from_dict(
        prune_artifact.stage_result.training_summary["pruning_execution_document"]
    )
    pruned_binding = pruning_execution.round.output_circuit_binding
    assert pruned_binding.content_checksum != pruning_execution.round.input_circuit_binding.content_checksum
    assert pruned_binding.circuit.num_params == pipeline.stages[-1].output_parameter_count

    final_checkpoint = store.load_stage_checkpoint(len(pipeline.stages) - 1)
    assert final_checkpoint.circuit_binding_checksum == pruned_binding.content_checksum
    final_parameters = store.load_final_parameters()
    planned_payload = encode_layerwise_materialized_circuit(pruned_binding, final_parameters)
    payload_checksum = f"sha256:{hashlib.sha256(planned_payload).hexdigest()}"

    prune_refs = prune_artifact.fixed_map_artifacts
    fine_tune_refs = store.stage_artifacts[-1].fixed_map_artifacts
    assert len(prune_refs) == len(fine_tune_refs) == 1
    pruning_map = _load_fixed_map(store, prune_refs[0])
    fine_tune_map = _load_fixed_map(store, fine_tune_refs[0])
    assert pruning_execution.training_ensemble_checksums == (pruning_map.content_checksum,)
    assert pruning_map.circuit_checksum == pruning_execution.round.input_circuit_binding.content_checksum
    assert fine_tune_map.circuit_checksum == pruned_binding.content_checksum
    assert pruning_map.resolved_seed == pipeline.stages[prune_index].training_seed
    assert fine_tune_map.resolved_seed == pipeline.stages[-1].training_seed

    stage_seeds = {
        seed
        for stage in pipeline.stages
        for seed in (
            stage.initialization_seed,
            stage.optimizer_seed,
            stage.training_seed,
            stage.checkpoint_validation.seed,
        )
        if seed is not None
    }
    assert _FINAL_EVALUATION_SEED not in stage_seeds
    evaluation_config = PipelineEvaluationConfig.for_pipeline(
        pipeline=training_result,
        materialized_circuit_checksum=payload_checksum,
        test_noise_id="depolarizing_1s_all",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        evaluation_seed=_FINAL_EVALUATION_SEED,
        evaluation_seed_domain=cast("str", pipeline.seed_domains["screening_selection"]),
        repetition=0,
        trajectory_budget=2,
        evaluation_policy="fixed_sample",
        confidence_level=None,
        confidence_interval_method=None,
        sidecar_storage_policy="trajectory_fidelities",
        max_bond_dimension=pipeline.stages[-1].max_bond_dimension,
        svd_threshold=pipeline.stages[-1].svd_threshold,
        truncation_mode=pipeline.stages[-1].truncation_mode,
        min_bond_dimension=pipeline.stages[-1].min_bond_dimension,
    )

    observed_circuits: list[LayerwiseMaterializedCircuit] = []
    observed_ensembles: list[KrotovFixedMapEnsemble] = []

    def materialize(
        complete_pipeline: TrainingPipelineResult,
        selected_parameters: NDArray[np.float64],
    ) -> MaterializedCircuitPayload:
        assert complete_pipeline.content_checksum == training_result.content_checksum
        np.testing.assert_array_equal(selected_parameters, final_parameters)
        return MaterializedCircuitPayload(
            serialized_bytes=planned_payload,
            wall_time_seconds=0.0,
            peak_memory_bytes=0,
        )

    def evaluate(config: PipelineEvaluationConfig, runtime_circuit: object) -> PipelineEvaluationMeasurement:
        assert isinstance(runtime_circuit, LayerwiseMaterializedCircuit)
        assert runtime_circuit.circuit_binding.to_dict() == pruned_binding.to_dict()
        np.testing.assert_array_equal(runtime_circuit.selected_parameters, final_parameters)
        observed_circuits.append(runtime_circuit)
        measurement = _fresh_noisy_measurement(config, runtime_circuit, target)
        observed_ensembles.extend(measurement.fixed_map_ensembles)
        return measurement

    records = ParallelPhase2Evaluator(store, decode_layerwise_materialized_circuit).evaluate(
        (evaluation_config,),
        materialize,
        evaluate,
        max_workers=1,
    )
    assert len(records) == 1
    assert isinstance(records[0], PipelineBenchmarkResult)
    assert len(observed_circuits) == len(observed_ensembles) == 1

    final_map = observed_ensembles[0]
    training_maps = (pruning_map, fine_tune_map)
    assert final_map.role == "screening_selection"
    assert final_map.resolved_seed == _FINAL_EVALUATION_SEED
    assert final_map.circuit_checksum == payload_checksum
    assert final_map.stage_configuration_checksum == evaluation_config.configuration_checksum
    assert final_map.ensemble_id not in {item.ensemble_id for item in training_maps}
    assert final_map.content_checksum not in {item.content_checksum for item in training_maps}

    evidence_rows = store.evaluation_evidence_stream_path.read_text(encoding="utf-8").splitlines()
    assert len(evidence_rows) == 1
    evidence = EvaluationEvidenceArtifact.from_json(evidence_rows[0])
    assert evidence.evaluation_row_id == evaluation_config.evaluation_row_id
    assert len(evidence.evaluation_map_artifacts) == 1
    persisted_final_map = _load_fixed_map(store, evidence.evaluation_map_artifacts[0])
    assert persisted_final_map.to_dict() == final_map.to_dict()

    reopened = Phase2ArtifactStore(store.output_directory, pipeline, fingerprint, resume=True)
    assert reopened.records == records
    assert len(reopened.materializations) == 1
    materialization = reopened.materializations[0]
    assert materialization.payload_checksum == payload_checksum
    materialized_bytes = (reopened.output_directory / materialization.path).read_bytes()
    assert materialized_bytes == planned_payload
    decoded = decode_layerwise_materialized_circuit(materialized_bytes)
    assert decoded.circuit_binding.to_dict() == pruned_binding.to_dict()
    np.testing.assert_array_equal(decoded.selected_parameters, final_parameters)
