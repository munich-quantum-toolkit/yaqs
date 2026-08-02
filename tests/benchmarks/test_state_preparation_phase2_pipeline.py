# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Phase II staged-training and evaluation identities."""

from __future__ import annotations

import csv
import io
from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.phase2.canonical import (
    canonical_checksum,
    canonical_json,
)
from benchmarks.state_preparation.phase2.layerwise_bmpd import resolve_layerwise_bmpd_crn_legacy_v1_pipeline
from benchmarks.state_preparation.phase2.pipeline import (
    PHASE1_FIXTURE_MANIFEST_CHECKSUM,
    PIPELINE_CSV_COLUMNS,
    TRAINING_PIPELINE_TEMPLATE_SCHEMA_VERSION,
    CheckpointValidationConfig,
    ExternalCheckpointRef,
    PipelineBenchmarkFailure,
    PipelineBenchmarkResult,
    PipelineEvaluationConfig,
    TrainingPipelineConfig,
    TrainingPipelineResult,
    TrainingPipelineTemplate,
    TrainingStageConfig,
    TrainingStageResult,
    TrainingStageTemplate,
    fixture_target_spec_checksum,
    pipeline_benchmark_record_from_csv_row,
    pipeline_benchmark_record_from_json,
    validate_screening_resolution,
)
from benchmarks.state_preparation.phase2.protocol import (
    TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
    ScreeningCandidateRef,
    ScreeningCell,
    ScreeningManifest,
    load_initial_preregistration,
)
from benchmarks.state_preparation.phase2.targets import (
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)

if TYPE_CHECKING:
    from benchmarks.state_preparation.phase2.targets import (
        TargetPopulationManifest,
    )

_NOISE_VERSION = "yaqs.state_preparation.noise.v1"
_SCREENING_MASTER = bytes(reversed(range(32)))
_GENERATION_RUNTIME_AVAILABLE = np.__version__ == "2.4.6"


def _checksum(character: str) -> str:
    return f"sha256:{character * 64}"


def _seed_domains() -> dict[str, object]:
    return {
        "initialization": "initialization",
        "optimizer_ordering": "optimizer_ordering",
        "training_trajectory": "training_trajectory",
        "checkpoint_validation": "checkpoint_validation",
        "pilot_evaluation": "pilot_evaluation",
        "screening_selection": "screening_selection",
        "confirmatory_test": "confirmatory_test",
    }


def _materialization_policy() -> dict[str, object]:
    return {
        "policy_id": "native_chain_v1",
        "compiler_policy_id": "quantinuum_rzz_chain_v1",
        "connectivity_id": "linear_chain",
        "routing_policy_id": "identity_no_swap",
        "optimization_level": 0,
        "noise_placement": "logical_parameterized_gates",
        "parameter_source": "selected_checkpoint",
    }


def _checkpoint_policy(*, enabled: bool = False) -> dict[str, object]:
    config = (
        CheckpointValidationConfig(
            noise_id="depolarizing_1s_all",
            noise_definition_version=_NOISE_VERSION,
            noise_strength_scale=1.0,
            tjm_dt=1.0,
            trajectory_count=64,
            seed=37,
            sampling_policy="crn_fixed",
            ensemble_refresh_interval=None,
            cadence=10,
            selection_rule="best_validation_fidelity",
            tie_breaker="earliest_iteration",
        )
        if enabled
        else CheckpointValidationConfig.disabled()
    )
    payload = config.to_dict()
    del payload["seed"]
    return payload


def _stage_template(
    *,
    index: int,
    stage_id: str,
    kind: str,
    input_topology: str | None,
    output_topology: str,
    input_parameters: int,
    output_parameters: int,
    transfer: str,
    iterations: int,
    noisy: bool = False,
    validation: bool = False,
    pruning_rule: str = "none",
    pruning_threshold: float | None = None,
) -> TrainingStageTemplate:
    random_transfer = transfer in {
        "initialize_random_uniform",
        "initialize_random_normal",
        "append_random_uniform",
        "append_random_normal",
    }
    return TrainingStageTemplate(
        stage_policy={
            "stage_index": index,
            "stage_id": stage_id,
            "stage_kind": kind,
            "input_topology_id": input_topology,
            "output_topology_id": output_topology,
            "input_parameter_count": input_parameters,
            "output_parameter_count": output_parameters,
            "parameter_transfer_rule": transfer,
            "optimizer_id": "none" if kind == "prune" else "krotov",
            "optimizer_hyperparameters": ({} if kind == "prune" else {"learning_rate": 0.01}),
            "iteration_budget": 0 if kind == "prune" else iterations,
            "training_noise_id": "depolarizing_1s_all" if noisy else "noiseless",
            "noise_definition_version": _NOISE_VERSION,
            "noise_strength_scale": 1.0 if noisy else None,
            "tjm_dt": 1.0 if noisy else None,
            "trajectory_count": 32 if noisy else 0,
            "trajectory_update": "independent" if noisy else None,
            "sampling_policy": "crn_fixed" if noisy else "none",
            "crn_refresh_interval": None,
            "checkpoint_validation_policy": _checkpoint_policy(enabled=validation),
            "pruning_rule": pruning_rule,
            "pruning_threshold": pruning_threshold,
            "max_bond_dimension": 64,
            "svd_threshold": 0.0,
            "truncation_mode": "discarded_weight",
            "min_bond_dimension": 1,
        },
        seed_bindings={
            "initialization": f"{stage_id}_initialization" if random_transfer else None,
            "optimizer": (f"{stage_id}_optimizer" if kind != "prune" or pruning_rule == "random" else None),
            "training": f"{stage_id}_training" if noisy else None,
            "checkpoint_validation": (f"{stage_id}_validation" if validation else None),
        },
    )


def _template(
    *,
    noisy: bool = True,
    method_id: str | None = None,
    template_id: str | None = None,
    growth_learning_rate: float = 0.01,
) -> TrainingPipelineTemplate:
    first = _stage_template(
        index=0,
        stage_id="grow_d1",
        kind="optimize",
        input_topology=None,
        output_topology="bmpd_d1",
        input_parameters=0,
        output_parameters=63,
        transfer="initialize_random_normal",
        iterations=50,
    )
    growth = _stage_template(
        index=1,
        stage_id="grow_d2",
        kind="grow",
        input_topology="bmpd_d1",
        output_topology="bmpd_d2",
        input_parameters=63,
        output_parameters=108,
        transfer="append_random_normal",
        iterations=50,
    )
    if canonical_json(growth_learning_rate) != "0.01":
        policy = dict(growth.stage_policy)
        policy["optimizer_hyperparameters"] = {"learning_rate": growth_learning_rate}
        growth = TrainingStageTemplate(
            stage_policy=policy,
            seed_bindings=growth.seed_bindings,
        )
    final = _stage_template(
        index=2,
        stage_id="final_finetune",
        kind="optimize",
        input_topology="bmpd_d2",
        output_topology="bmpd_d2",
        input_parameters=108,
        output_parameters=108,
        transfer="copy",
        iterations=200,
        noisy=noisy,
        validation=True,
    )
    resolved_method = method_id or ("layerwise_bmpd_crn_v2" if noisy else "layerwise_bmpd_noiseless")
    return TrainingPipelineTemplate(
        template_id=template_id or ("layerwise_noisy_candidate" if noisy else "layerwise_noiseless_control"),
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id="primary_q6",
        ansatz_family="bmpd_brickwall",
        method_id=resolved_method,
        method_version="1",
        resource_stratum_id="primary_cap_12",
        stages=(first, growth, final),
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )


@lru_cache(maxsize=1)
def _screening_target_manifest() -> TargetPopulationManifest:
    """Build the genuine deterministic primary-q6 WP16 screening manifest.

    Returns:
        The complete seed-bearing screening target manifest.
    """
    if not _GENERATION_RUNTIME_AVAILABLE:
        pytest.skip("Target-backed pipeline fixtures require preregistered NumPy 2.4.6.")
    preregistration = load_initial_preregistration()
    config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(_SCREENING_MASTER),
        population_scope="primary_q6",
    )
    return create_target_population_manifest(config, preregistration, _SCREENING_MASTER)


def _typed_screening_universe(
    template: TrainingPipelineTemplate,
) -> tuple[ScreeningManifest, ScreeningCandidateRef, ScreeningCell, TargetPopulationManifest]:
    """Construct the complete typed WP15 universe over a genuine WP16 manifest.

    Returns:
        Screening manifest, selected candidate, one cell, and target manifest.
    """
    preregistration = load_initial_preregistration()
    target_manifest = _screening_target_manifest()
    noiseless = _template(noisy=False)
    candidates: list[ScreeningCandidateRef] = []
    selected: ScreeningCandidateRef | None = None
    for policy in preregistration.candidate_methods:
        if policy["scope"] != "all_families":
            continue
        method_id = cast("str", policy["method_id"])
        candidate_template = (
            template if method_id == template.method_id else noiseless if method_id == noiseless.method_id else None
        )
        candidate = ScreeningCandidateRef(
            configuration_schema_version=(
                candidate_template.schema_version
                if candidate_template is not None
                else TRAINING_PIPELINE_TEMPLATE_SCHEMA_VERSION
            ),
            configuration_checksum=(
                candidate_template.configuration_checksum
                if candidate_template is not None
                else canonical_checksum({"test_candidate_method_id": method_id})
            ),
            method_id=method_id,
            noisy_training=cast("bool", policy["noisy_training"]),
            resource_stratum_id=template.resource_stratum_id,
            matching_projection_checksum=(
                candidate_template.matching_projection_checksum
                if candidate_template is not None and method_id in {"layerwise_bmpd_crn_v2", "layerwise_bmpd_noiseless"}
                else None
            ),
        )
        candidates.append(candidate)
        if method_id == template.method_id:
            selected = candidate
    assert selected is not None

    cells: list[ScreeningCell] = []
    screening_seed = 500_000
    for spec in target_manifest.instances:
        for optimization_index, optimization_seed in enumerate((101, 102, 103), start=1):
            screening_seed += 1
            cells.append(
                ScreeningCell(
                    cell_id=f"{spec.target_instance_id}_optimization_{optimization_index}",
                    family_id=spec.family_id,
                    stratum_id=spec.stratum_id,
                    qubit_count=spec.qubit_count,
                    target_instance_id=spec.target_instance_id,
                    optimization_seed=optimization_seed,
                    screening_seed=screening_seed,
                )
            )
    manifest = ScreeningManifest(
        manifest_id="phase2_screening_manifest_pipeline_test_v1",
        preregistration_checksum=preregistration.content_checksum,
        screening_target_manifest_checksum=target_manifest.content_checksum,
        evaluation_policy_checksum=canonical_checksum({
            "endpoint": preregistration.primary_endpoint,
            "failure_policy": preregistration.failure_policy,
            "noise": preregistration.primary_noise_condition,
        }),
        resource_policy_checksum=canonical_checksum(preregistration.primary_resource_constraint),
        baseline_configuration_checksum=template.configuration_checksum,
        candidates=tuple(candidates),
        cells=tuple(cells),
    )
    return manifest, selected, cells[0], target_manifest


def _pipeline(
    template: TrainingPipelineTemplate | None = None,
    *,
    target_character: str = "a",
    block_id: str = "screening_block_0",
    optimization_seed: int = 101,
) -> TrainingPipelineConfig:
    candidate = template or _template()
    target_manifest = _screening_target_manifest()
    target_index = int(target_character, 16) % len(target_manifest.instances)
    target_spec = target_manifest.instances[target_index]
    return candidate.resolve(
        target_namespace="phase2",
        target_manifest=target_manifest,
        target_instance_id=target_spec.target_instance_id,
        target_population_manifest_checksum=target_manifest.content_checksum,
        target_instance_spec_checksum=target_spec.content_checksum,
        target_family_id=target_spec.family_id,
        target_stratum_id=target_spec.stratum_id,
        qubit_count=target_spec.qubit_count,
        optimization_block_id=block_id,
        optimization_seed=optimization_seed,
        data_role="screening_selection",
    )


def _work(
    *,
    objective: int = 0,
    gradient: int = 0,
    training: int = 0,
    validation: int = 0,
    test: int = 0,
    gates: int = 0,
) -> dict[str, object]:
    return {
        "objective_evaluations": objective,
        "gradient_evaluations": gradient,
        "training_trajectories": training,
        "checkpoint_validation_trajectories": validation,
        "test_trajectories": test,
        "trajectory_gate_applications": gates,
    }


def _stage_results(
    pipeline: TrainingPipelineConfig,
) -> tuple[TrainingStageResult, ...]:
    results: list[TrainingStageResult] = []
    previous: TrainingStageResult | None = None
    for stage in pipeline.stages:
        input_checksum = stage.input_checkpoint_checksum if previous is None else previous.produced_checkpoint_checksum
        if previous is None and input_checksum is not None:
            input_provenance = canonical_checksum({
                "external_checkpoint_checksum": input_checksum,
                "external_pipeline_prefix": stage.input_checkpoint_pipeline_prefix,
            })
        else:
            input_provenance = None if previous is None else previous.checkpoint_provenance_checksum
        checkpoint_checksum = _checksum(str(stage.stage_index + 1))
        provenance = canonical_checksum({
            "pipeline_prefix_id": pipeline.prefix_id(stage.stage_index),
            "stage_id": stage.stage_id,
            "stage_configuration_checksum": stage.configuration_checksum,
            "input_checkpoint_checksum": input_checksum,
            "input_checkpoint_provenance_checksum": input_provenance,
            "produced_checkpoint_checksum": checkpoint_checksum,
        })
        result = TrainingStageResult(
            pipeline_training_id=pipeline.training_id,
            pipeline_prefix_id=pipeline.prefix_id(stage.stage_index),
            stage_index=stage.stage_index,
            stage_id=stage.stage_id,
            stage_configuration_checksum=stage.configuration_checksum,
            input_checkpoint_checksum=input_checksum,
            input_checkpoint_provenance_checksum=input_provenance,
            produced_checkpoint_path=f"checkpoints/{stage.stage_id}.npy",
            produced_checkpoint_checksum=checkpoint_checksum,
            checkpoint_provenance_checksum=provenance,
            output_topology_id=stage.output_topology_id,
            output_parameter_count=stage.output_parameter_count,
            training_summary={
                "completed_iterations": stage.iteration_budget,
                "final_objective": 0.2,
            },
            checkpoint_validation_summary=(
                {"evaluation_count": 20, "selected_iteration": 180} if stage.checkpoint_validation.enabled else None
            ),
            training_ensemble_checksums=((_checksum("d"),) if stage.sampling_policy == "crn_fixed" else ()),
            checkpoint_validation_ensemble_checksum=(_checksum("e") if stage.checkpoint_validation.enabled else None),
            optimizer_trace_path=f"traces/{stage.stage_id}.json",
            optimizer_trace_checksum=_checksum("f"),
            diagnostic_sidecar_path=None,
            diagnostic_sidecar_checksum=None,
            wall_time_seconds=1.0,
            peak_memory_bytes=1000 + stage.stage_index,
            normalized_work=_work(
                objective=stage.iteration_budget,
                gradient=stage.iteration_budget,
                training=stage.iteration_budget * stage.trajectory_count,
                validation=(stage.checkpoint_validation.trajectory_count if stage.checkpoint_validation.enabled else 0),
                gates=100,
            ),
        )
        results.append(result)
        previous = result
    return tuple(results)


def _pipeline_result(
    pipeline: TrainingPipelineConfig | None = None,
) -> TrainingPipelineResult:
    config = pipeline or _pipeline()
    stages = _stage_results(config)
    final = stages[-1]
    return TrainingPipelineResult(
        config=config,
        stage_results=stages,
        final_checkpoint_path=final.produced_checkpoint_path,
        final_checkpoint_checksum=final.produced_checkpoint_checksum,
        final_checkpoint_provenance_checksum=final.checkpoint_provenance_checksum,
        wall_time_seconds=sum(stage.wall_time_seconds for stage in stages),
        peak_memory_bytes=max(stage.peak_memory_bytes for stage in stages),
        normalized_work={
            key: sum(cast("int", stage.normalized_work[key]) for stage in stages) for key in stages[0].normalized_work
        },
    )


def _evaluation(
    result: TrainingPipelineResult | None = None,
    *,
    seed: int = 909,
    repetition: int = 0,
) -> PipelineEvaluationConfig:
    pipeline = result or _pipeline_result()
    return PipelineEvaluationConfig.for_pipeline(
        pipeline=pipeline,
        materialized_circuit_checksum=_checksum("9"),
        test_noise_id="depolarizing_1s_all",
        noise_definition_version=_NOISE_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        evaluation_seed=seed,
        evaluation_seed_domain="screening_selection",
        repetition=repetition,
        trajectory_budget=256,
        evaluation_policy="fixed_sample",
        confidence_level=None,
        confidence_interval_method=None,
        sidecar_storage_policy="trajectory_fidelities",
        max_bond_dimension=64,
        svd_threshold=0.0,
        truncation_mode="discarded_weight",
        min_bond_dimension=1,
    )


def _benchmark_result(
    evaluation: PipelineEvaluationConfig | None = None,
) -> PipelineBenchmarkResult:
    config = evaluation or _evaluation()
    return PipelineBenchmarkResult(
        config=config,
        materialized_circuit_path="circuits/final.json",
        test_noiseless_fidelity=0.93,
        test_noisy_fidelity=0.87,
        noisy_fidelity_standard_deviation=0.08,
        noisy_fidelity_standard_error=0.005,
        confidence_interval_lower=None,
        confidence_interval_upper=None,
        sampled_nonidentity_events=11,
        trajectory_sidecar_path="trajectories/final.npz",
        trajectory_sidecar_checksum=_checksum("8"),
        evaluation_wall_time_seconds=2.0,
        peak_memory_bytes=2048,
        normalized_work=_work(test=256, gates=5000),
        runtime_fingerprint_checksum=_checksum("7"),
    )


def test_candidate_template_is_target_independent_and_round_trips() -> None:
    """Candidate templates remain stable across concrete screening cells."""
    template = _template()
    assert TrainingPipelineTemplate.from_json(template.to_json()) == template
    assert template.to_json() == TrainingPipelineTemplate.from_dict(template.to_dict()).to_json()
    assert template.schema_version == TRAINING_PIPELINE_TEMPLATE_SCHEMA_VERSION

    first = _pipeline(template, target_character="a", block_id="block_a")
    second = _pipeline(template, target_character="d", block_id="block_b")
    assert first.template_checksum == second.template_checksum
    assert first.training_id != second.training_id
    assert first.stages[0].initialization_seed != second.stages[0].initialization_seed


def test_v1_pipeline_serialization_and_identity_goldens() -> None:
    """The first WP16 schema freezes representative serialization and identifiers."""
    template = _template()
    pipeline = _pipeline(template)
    result = _pipeline_result(pipeline)
    evaluation = _evaluation(result)
    assert template.configuration_checksum == "sha256:764e7a52e9baec43ab35b4f5f7368352cbd0113eee6fb5ff2207536147923ce0"
    assert (
        canonical_checksum(template.to_dict())
        == "sha256:30fac76d0f924c653a3cb45520324e8752832adde53ac482e485b4f2ec5b1081"
    )
    assert pipeline.training_id == "phase2_training_0c51da0b22558b0f0648092814c18c4566b1245fa6cdb1c63a2674e68ec3e214"
    assert (
        canonical_checksum(pipeline.to_dict())
        == "sha256:aa0edf97adf3629bae472fd9dc818952224a84a0fd049412bd0b622f9ae90444"
    )
    prefix_id_zero = "phase2_pipeline_prefix_4b" + "a2de1bc75e301d95df72bbf536c1c11be9ce6196e31958af0210e0781c87f9"
    assert pipeline.prefix_id(0) == prefix_id_zero
    assert pipeline.prefix_id(2) == (
        "phase2_pipeline_prefix_b0da0e241b7eb66fa5b3957cd2ee0ca58758604d9ebfe72292a19080d6fb6f76"
    )
    assert result.content_checksum == "sha256:4c364e7630b0c8a58d67e87733587fbdf51a3e23f8c17c16ce11b76a532485b1"
    assert evaluation.evaluation_row_id == (
        "phase2_evaluation_57f899f8056457eb1dd91767c8bf3c999c86055f5b7b59de7cdb35dc2038b552"
    )


def test_matching_projection_changes_only_exact_final_training_treatment() -> None:
    """The noisy and noiseless controls share only the sealed matching projection."""
    noisy = _template(noisy=True)
    noiseless = _template(noisy=False)
    assert noisy.configuration_checksum != noiseless.configuration_checksum
    assert noisy.matching_projection_checksum == noiseless.matching_projection_checksum

    noisy_pipeline = _pipeline(noisy)
    noiseless_pipeline = _pipeline(noiseless)
    assert noisy_pipeline.stages[:2] == noiseless_pipeline.stages[:2]
    assert noisy_pipeline.prefix_id(1) == noiseless_pipeline.prefix_id(1)
    assert noisy_pipeline.training_id != noiseless_pipeline.training_id


def test_matching_projection_retains_growth_and_validation_policy() -> None:
    """Growth and checkpoint-selection differences remain visible when matching."""
    reference = _template(noisy=True)
    changed_growth = _template(noisy=True, growth_learning_rate=0.02)
    assert reference.matching_projection_checksum != changed_growth.matching_projection_checksum

    data = reference.to_dict()
    stages = cast("list[dict[str, object]]", data["stages"])
    final = stages[-1]
    policy = cast("dict[str, object]", final["stage_policy"])
    validation = cast("dict[str, object]", policy["checkpoint_validation_policy"])
    validation["cadence"] = 20
    stage = TrainingStageTemplate(
        stage_policy=policy,
        seed_bindings=cast("dict[str, object]", final["seed_bindings"]),
    )
    with pytest.raises(ValueError, match="checkpoint validation"):
        TrainingPipelineTemplate(
            template_id=reference.template_id,
            preregistration_checksum=reference.preregistration_checksum,
            target_scope_id=reference.target_scope_id,
            ansatz_family=reference.ansatz_family,
            method_id=reference.method_id,
            method_version=reference.method_version,
            resource_stratum_id=reference.resource_stratum_id,
            stages=(*reference.stages[:-1], stage),
            seed_domains=reference.seed_domains,
            final_materialization_policy=reference.final_materialization_policy,
        ).matching_projection()


def test_matched_pair_rejects_noisy_growth_and_noiseless_version_aliases() -> None:
    """The matched control is canonical noiseless growth plus one exact treatment change."""
    template = _template(noisy=False)
    growth = template.stages[1]
    policy = dict(growth.stage_policy)
    policy.update({
        "training_noise_id": "depolarizing_1s_all",
        "noise_strength_scale": 1.0,
        "tjm_dt": 1.0,
        "trajectory_count": 8,
        "trajectory_update": "independent",
        "sampling_policy": "crn_fixed",
    })
    bindings = dict(growth.seed_bindings)
    bindings["training"] = "grow_d2_training"
    noisy_growth = TrainingStageTemplate(stage_policy=policy, seed_bindings=bindings)
    with pytest.raises(ValueError, match="pre-final stage"):
        replace(template, stages=(template.stages[0], noisy_growth, template.stages[2])).matching_projection()

    final = template.stages[-1]
    final_policy = dict(final.stage_policy)
    final_policy["noise_definition_version"] = "yaqs.state_preparation.noise.alias_v1"
    aliased = TrainingStageTemplate(stage_policy=final_policy, seed_bindings=final.seed_bindings)
    with pytest.raises(ValueError, match="exact noiseless final treatment"):
        replace(template, stages=(*template.stages[:-1], aliased)).matching_projection()


def test_future_stage_changes_do_not_perturb_earlier_prefixes() -> None:
    """Suffix-only policy changes preserve earlier streams and checkpoint prefixes."""
    reference = _pipeline(_template(growth_learning_rate=0.01))
    changed = _pipeline(_template(growth_learning_rate=0.02))
    assert reference.stages[0] == changed.stages[0]
    assert reference.prefix_id(0) == changed.prefix_id(0)
    assert reference.stages[1].optimizer_seed != changed.stages[1].optimizer_seed
    assert reference.prefix_id(1) != changed.prefix_id(1)


def test_future_stage_changes_preserve_earlier_checkpoint_reuse_identity() -> None:
    """Suffix-only changes preserve prior artifacts while changing later audit roots."""
    reference_pipeline = _pipeline(_template(growth_learning_rate=0.01))
    changed_pipeline = _pipeline(_template(growth_learning_rate=0.02))
    reference_result = _pipeline_result(reference_pipeline)
    changed_result = _pipeline_result(changed_pipeline)
    reference_stage = reference_result.stage_results[0]
    changed_stage = changed_result.stage_results[0]

    assert reference_pipeline.training_id != changed_pipeline.training_id
    assert reference_result.content_checksum != changed_result.content_checksum
    assert reference_stage.produced_checkpoint_checksum == changed_stage.produced_checkpoint_checksum
    assert reference_stage.checkpoint_provenance_checksum == changed_stage.checkpoint_provenance_checksum
    assert (
        reference_result.stage_results[1].checkpoint_provenance_checksum
        != changed_result.stage_results[1].checkpoint_provenance_checksum
    )

    reference = ExternalCheckpointRef.from_pipeline_result(reference_result, 0)
    changed = ExternalCheckpointRef.from_pipeline_result(changed_result, 0)
    assert reference.provenance_ref_checksum == changed.provenance_ref_checksum
    assert reference.content_checksum != changed.content_checksum


def test_method_family_and_preregistration_cannot_be_forged() -> None:
    """Derived method families and the trusted protocol link reject aliases."""
    data = _template().to_dict()
    data["method_family_id"] = "forged"
    with pytest.raises(ValueError, match="method_family_id"):
        TrainingPipelineTemplate.from_dict(data)

    with pytest.raises(ValueError, match="trusted Phase II preregistration"):
        replace(_template(), preregistration_checksum=_checksum("0"))


def test_concrete_pipeline_round_trip_and_immutable_state() -> None:
    """Concrete pipelines serialize losslessly and recursively freeze mappings."""
    pipeline = _pipeline()
    assert TrainingPipelineConfig.from_json(pipeline.to_json()) == pipeline
    frozen_setattr = pipeline.__setattr__
    with pytest.raises(FrozenInstanceError):
        frozen_setattr("qubit_count", 12)
    with pytest.raises(TypeError):
        cast("dict[str, object]", pipeline.template.seed_domains)["training_trajectory"] = "changed"


def test_target_role_and_scope_guards_reject_leakage() -> None:
    """Primary q6 targets cannot be replaced by secondary or legacy fixtures."""
    template = _template()
    target_manifest = _screening_target_manifest()
    target_spec = target_manifest.instances[0]
    valid = _pipeline()
    with pytest.raises(ValueError, match="typed target manifest"):
        replace(valid, target_instance_id="phase2_target_" + "0" * 64)
    with pytest.raises(ValueError, match="typed target manifest"):
        replace(valid, target_family_id="haar_random", target_stratum_id="dense_complex")
    with pytest.raises(TypeError, match="target_ref"):
        template.resolve(
            target_namespace="phase2",
            target_manifest=None,
            target_instance_id=target_spec.target_instance_id,
            target_population_manifest_checksum=target_manifest.content_checksum,
            target_instance_spec_checksum=target_spec.content_checksum,
            target_family_id=target_spec.family_id,
            target_stratum_id=target_spec.stratum_id,
            qubit_count=target_spec.qubit_count,
            optimization_block_id="missing_typed_manifest",
            optimization_seed=1,
            data_role="screening_selection",
        )
    assert valid.target_ref is not None
    target_ref_data = valid.target_ref.to_dict()
    assert "target_manifest_id" not in target_ref_data
    target_ref_data["target_manifest_id"] = "phase2_target_population_" + "0" * 64
    with pytest.raises(ValueError, match="fields do not match"):
        type(valid.target_ref).from_dict(target_ref_data)
    with pytest.raises(ValueError, match="typed target manifest"):
        template.resolve(
            target_namespace="phase2",
            target_manifest=target_manifest,
            target_instance_id=target_spec.target_instance_id,
            target_population_manifest_checksum=target_manifest.content_checksum,
            target_instance_spec_checksum=target_spec.content_checksum,
            target_family_id=target_spec.family_id,
            target_stratum_id=target_spec.stratum_id,
            qubit_count=12,
            optimization_block_id="bad_q12_screen",
            optimization_seed=1,
            data_role="screening_selection",
        )
    with pytest.raises(ValueError, match="only for secondary"):
        template.resolve(
            target_namespace="phase1_fixture",
            target_manifest=None,
            target_instance_id="gaussian_mu0p5_sigma0p1",
            target_population_manifest_checksum=PHASE1_FIXTURE_MANIFEST_CHECKSUM,
            target_instance_spec_checksum=fixture_target_spec_checksum("phase1_fixture", "gaussian_mu0p5_sigma0p1", 6),
            target_family_id="gaussian_amplitude",
            target_stratum_id="interior",
            qubit_count=6,
            optimization_block_id="phase1_holdout",
            optimization_seed=1,
            data_role="confirmatory",
        )


def test_seed_domains_and_stage_order_are_strict() -> None:
    """Random-stream domain names and ordered stage indices are frozen."""
    domains = _seed_domains()
    domains["training_trajectory"] = "renamed_training"
    with pytest.raises(ValueError, match="exact preregistered"):
        replace(_template(), seed_domains=domains)

    with pytest.raises(ValueError, match="indices"):
        replace(_template(), stages=tuple(reversed(_template().stages)))


def test_physical_indices_counts_work_and_seeds_are_nonnegative() -> None:
    """Signed aliases cannot enter random streams or scientific work ledgers."""
    pipeline = _pipeline()
    noisy_stage = pipeline.stages[-1]
    for changes in (
        {"stage_index": -1},
        {"iteration_budget": -1},
        {"trajectory_count": -1},
        {"training_seed": -1},
    ):
        with pytest.raises(ValueError, match="at least 0"):
            replace(noisy_stage, **changes)
    with pytest.raises(ValueError, match="at least 0"):
        pipeline.prefix_id(-1)
    with pytest.raises(ValueError, match="at least 0"):
        replace(_evaluation(), repetition=-1)
    with pytest.raises(ValueError, match="at least 0"):
        replace(_evaluation(), trajectory_budget=-1)

    stage_result = _pipeline_result(pipeline).stage_results[0]
    bad_work = dict(stage_result.normalized_work)
    bad_work["objective_evaluations"] = -1
    with pytest.raises(ValueError, match="at least 0"):
        replace(stage_result, normalized_work=bad_work)
    with pytest.raises(ValueError, match="at least 0"):
        replace(stage_result, peak_memory_bytes=-1)
    with pytest.raises(ValueError, match="at least 0"):
        replace(_benchmark_result(), sampled_nonidentity_events=-1)
    with pytest.raises(ValueError, match="at least 0"):
        ExternalCheckpointRef.from_pipeline_result(_pipeline_result(pipeline), -1)


def test_ballarin_and_invalid_noisy_crn_training_are_rejected() -> None:
    """Training rejects evaluation-only Ballarin noise and incomplete CRN refresh."""
    stage = _template().stages[-1]
    policy = dict(stage.stage_policy)
    policy["training_noise_id"] = "ballarin_coupled"
    policy["noise_strength_scale"] = None
    policy["tjm_dt"] = None
    with pytest.raises(ValueError, match="evaluation-only"):
        TrainingStageTemplate(stage_policy=policy, seed_bindings=stage.seed_bindings)

    policy = dict(stage.stage_policy)
    policy["sampling_policy"] = "crn_refresh"
    with pytest.raises(ValueError, match="crn_refresh"):
        TrainingStageTemplate(stage_policy=policy, seed_bindings=stage.seed_bindings)


def test_checkpoint_validation_configuration_changes_candidate_identity() -> None:
    """Checkpoint selection inputs participate in candidate and training identity."""
    template = _template()
    final = template.stages[-1]
    policy = dict(final.stage_policy)
    validation = dict(cast("dict[str, object]", policy["checkpoint_validation_policy"]))
    validation["trajectory_count"] = 128
    policy["checkpoint_validation_policy"] = validation
    changed_stage = TrainingStageTemplate(
        stage_policy=policy,
        seed_bindings=final.seed_bindings,
    )
    changed = replace(template, stages=(*template.stages[:-1], changed_stage))
    assert changed.configuration_checksum != template.configuration_checksum
    assert _pipeline(changed).training_id != _pipeline(template).training_id


def test_random_pruning_requires_and_resolves_optimizer_ordering_seed() -> None:
    """Random pruning uses an explicit deterministic ordering stream."""
    pruning = _stage_template(
        index=0,
        stage_id="random_prune",
        kind="prune",
        input_topology="bmpd_d2",
        output_topology="bmpd_d2_pruned",
        input_parameters=108,
        output_parameters=90,
        transfer="apply_pruning_mask",
        iterations=0,
        pruning_rule="random",
        pruning_threshold=0.1,
    )
    bindings = dict(pruning.seed_bindings)
    bindings["optimizer"] = None
    with pytest.raises(ValueError, match=r"seed_bindings\.optimizer"):
        TrainingStageTemplate(
            stage_policy=pruning.stage_policy,
            seed_bindings=bindings,
        )
    resolved = pruning.resolve(
        optimization_seed=7,
        seed_domains=_seed_domains(),
        resolution_context_checksum=_checksum("a"),
    )
    assert resolved.optimizer_id == "none"
    assert resolved.optimizer_seed is not None


def test_pipeline_result_checkpoint_chain_and_round_trip() -> None:
    """Stage results form a lossless checksum and provenance chain."""
    result = _pipeline_result()
    assert TrainingPipelineResult.from_json(result.to_json()) == result
    for predecessor, successor in zip(
        result.stage_results[:-1],
        result.stage_results[1:],
        strict=True,
    ):
        assert successor.input_checkpoint_checksum == predecessor.produced_checkpoint_checksum
        assert successor.input_checkpoint_provenance_checksum == predecessor.checkpoint_provenance_checksum


def test_external_checkpoint_reference_is_producer_verified_and_compact() -> None:
    """Resumed pipelines consume a typed producer ref but store only path-free provenance."""
    producer = _pipeline_result()
    reference = ExternalCheckpointRef.from_pipeline_result(producer, 0)
    assert ExternalCheckpointRef.from_dict(reference.to_dict()) == reference

    continuation_policy = dict(
        _stage_template(
            index=0,
            stage_id="resume_d1",
            kind="optimize",
            input_topology="bmpd_d1",
            output_topology="bmpd_d1",
            input_parameters=63,
            output_parameters=63,
            transfer="copy",
            iterations=20,
        ).stage_policy
    )
    continuation_policy["parameter_transfer_rule"] = "load_checkpoint"
    continuation_stage = TrainingStageTemplate(
        stage_policy=continuation_policy,
        seed_bindings={
            "initialization": None,
            "optimizer": "resume_d1_optimizer",
            "training": None,
            "checkpoint_validation": None,
        },
    )
    continuation = TrainingPipelineTemplate(
        template_id="fixed_depth_resume_candidate",
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id="primary_q6",
        ansatz_family="bmpd_brickwall",
        method_id="fixed_depth_bmpd_crn",
        method_version="1",
        resource_stratum_id="primary_cap_12",
        stages=(continuation_stage,),
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )
    manifest = _screening_target_manifest()
    spec = reference.producer_result.config.target_ref
    assert spec is not None
    target_spec = spec.target_spec

    def resolve(checkpoint_ref: ExternalCheckpointRef) -> TrainingPipelineConfig:
        return continuation.resolve(
            target_namespace="phase2",
            target_manifest=manifest,
            target_instance_id=target_spec.target_instance_id,
            target_population_manifest_checksum=manifest.content_checksum,
            target_instance_spec_checksum=target_spec.content_checksum,
            target_family_id=target_spec.family_id,
            target_stratum_id=target_spec.stratum_id,
            qubit_count=target_spec.qubit_count,
            optimization_block_id=producer.config.optimization_block_id,
            optimization_seed=producer.config.optimization_seed,
            data_role="screening_selection",
            input_checkpoint_path="relocated/resume.npy",
            input_checkpoint_ref=checkpoint_ref,
        )

    resumed = resolve(reference)
    assert resumed.stages[0].input_checkpoint_ref_checksum == reference.provenance_ref_checksum
    assert "producer_result" not in resumed.stages[0].to_dict()

    changed_stage = replace(
        producer.stage_results[0],
        produced_checkpoint_path="other/location.npy",
        training_summary={"completed_iterations": 50, "final_objective": 0.7},
    )
    changed_producer = replace(producer, stage_results=(changed_stage, *producer.stage_results[1:]))
    changed_reference = ExternalCheckpointRef.from_pipeline_result(changed_producer, 0)
    assert changed_reference.provenance_ref_checksum == reference.provenance_ref_checksum
    assert changed_reference.content_checksum != reference.content_checksum
    assert resolve(changed_reference).to_json() == resumed.to_json()

    producer_stage = producer.stage_results[0]
    for changes in (
        {"pipeline_prefix_id": "phase2_pipeline_prefix_" + "0" * 64},
        {"checkpoint_provenance_checksum": _checksum("0")},
        {"produced_checkpoint_checksum": _checksum("0")},
    ):
        with pytest.raises(ValueError, match="provenance"):
            replace(producer_stage, **changes)

    wrong_topology_policy = dict(continuation_policy)
    wrong_topology_policy["input_topology_id"] = "bmpd_d2"
    wrong_topology_policy["output_topology_id"] = "bmpd_d2"
    wrong_topology_policy["input_parameter_count"] = 108
    wrong_topology_policy["output_parameter_count"] = 108
    wrong_topology = TrainingStageTemplate(
        stage_policy=wrong_topology_policy,
        seed_bindings=continuation_stage.seed_bindings,
    )
    with pytest.raises(ValueError, match="topology and parameter count"):
        wrong_topology.resolve(
            optimization_seed=101,
            seed_domains=_seed_domains(),
            resolution_context_checksum=_checksum("a"),
            input_checkpoint_path="relocated/resume.npy",
            input_checkpoint_ref=reference,
        )


def test_pipeline_templates_reject_implicit_or_mid_pipeline_checkpoint_state() -> None:
    """Only a typed first-stage load may introduce state from outside the pipeline."""
    implicit = _stage_template(
        index=0,
        stage_id="implicit_copy",
        kind="optimize",
        input_topology="bmpd_d1",
        output_topology="bmpd_d1",
        input_parameters=63,
        output_parameters=63,
        transfer="copy",
        iterations=10,
    )
    with pytest.raises(ValueError, match="typed external checkpoint"):
        replace(_template(method_id="fixed_depth_bmpd_crn"), stages=(implicit,))

    later = _stage_template(
        index=1,
        stage_id="later_copy",
        kind="optimize",
        input_topology="bmpd_d1",
        output_topology="bmpd_d1",
        input_parameters=63,
        output_parameters=63,
        transfer="copy",
        iterations=10,
    )
    later_policy = dict(later.stage_policy)
    later_policy["parameter_transfer_rule"] = "load_checkpoint"
    later_load = TrainingStageTemplate(stage_policy=later_policy, seed_bindings=later.seed_bindings)
    with pytest.raises(ValueError, match="Only the first"):
        replace(
            _template(method_id="fixed_depth_bmpd_crn"),
            stages=(_template().stages[0], later_load),
        )


def test_pipeline_result_rejects_wrong_prefix_checkpoint_and_provenance() -> None:
    """Pipeline results reject reordered stages and forged predecessor provenance."""
    result = _pipeline_result()
    middle = result.stage_results[1]
    with pytest.raises(ValueError, match="pipeline"):
        replace(
            result,
            stage_results=(result.stage_results[0], replace(middle, stage_id="wrong"), *result.stage_results[2:]),
        )
    with pytest.raises(ValueError, match="predecessor"):
        replace(
            result,
            stage_results=(
                result.stage_results[0],
                replace(
                    middle,
                    input_checkpoint_provenance_checksum=_checksum("0"),
                    checkpoint_provenance_checksum=canonical_checksum({
                        "pipeline_prefix_id": middle.pipeline_prefix_id,
                        "stage_id": middle.stage_id,
                        "stage_configuration_checksum": middle.stage_configuration_checksum,
                        "input_checkpoint_checksum": middle.input_checkpoint_checksum,
                        "input_checkpoint_provenance_checksum": _checksum("0"),
                        "produced_checkpoint_checksum": middle.produced_checkpoint_checksum,
                    }),
                ),
                *result.stage_results[2:],
            ),
        )


def test_stage_outcomes_and_paths_do_not_change_training_identity() -> None:
    """Observed outcomes and output path spelling stay outside training identity."""
    pipeline = _pipeline()
    first = _pipeline_result(pipeline)
    changed_stage = replace(
        first.stage_results[0],
        training_summary={"completed_iterations": 50, "final_objective": 0.8},
        produced_checkpoint_path="other/checkpoint.npy",
    )
    changed = replace(
        first,
        stage_results=(changed_stage, *first.stage_results[1:]),
    )
    assert changed.training_id == first.training_id
    assert changed.content_checksum != first.content_checksum


def test_unrelated_candidates_use_distinct_resolved_seed_streams() -> None:
    """Unmatched candidate configurations receive distinct derived streams."""
    layerwise = _pipeline(_template())
    fixed_depth = _pipeline(
        _template(
            method_id="fixed_depth_bmpd_crn",
            template_id="fixed_depth_candidate",
        )
    )
    assert layerwise.stages[0].initialization_seed != fixed_depth.stages[0].initialization_seed
    assert layerwise.stages[0].optimizer_seed != fixed_depth.stages[0].optimizer_seed


def test_phase1_and_legacy_scopes_cannot_cross_namespaces() -> None:
    """Phase I and legacy fixture scopes remain mutually isolated."""
    phase1_template = replace(_template(), target_scope_id="phase1_fixture")
    phase1 = phase1_template.resolve(
        target_namespace="phase1_fixture",
        target_manifest=None,
        target_instance_id="gaussian_mu0p5_sigma0p1",
        target_population_manifest_checksum=PHASE1_FIXTURE_MANIFEST_CHECKSUM,
        target_instance_spec_checksum=fixture_target_spec_checksum("phase1_fixture", "gaussian_mu0p5_sigma0p1", 6),
        target_family_id="gaussian_amplitude",
        target_stratum_id="interior",
        qubit_count=6,
        optimization_block_id="phase1_secondary",
        optimization_seed=1,
        data_role="secondary_benchmark",
    )
    assert phase1.target_namespace == "phase1_fixture"
    with pytest.raises(ValueError, match="18 target"):
        replace(phase1, target_population_manifest_checksum=_checksum("0"))
    with pytest.raises(ValueError, match="five immutable"):
        replace(phase1, target_namespace="legacy_reproduction")

    legacy = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(100)
    assert legacy.target_namespace == "legacy_reproduction"
    with pytest.raises(ValueError, match="five immutable"):
        replace(legacy, target_instance_spec_checksum=_checksum("0"))
    with pytest.raises(ValueError, match="phase1_fixture"):
        replace(legacy, template=phase1_template)
    with pytest.raises(ValueError, match="Legacy method identities"):
        replace(_template(), method_id="standard_vqa")


def test_stage_configuration_json_is_strict_and_path_free_identity_is_stable() -> None:
    """Stage JSON checks identities and external checkpoint semantics strictly."""
    stage = _pipeline().stages[0]
    assert TrainingStageConfig.from_json(stage.to_json()) == stage
    data = stage.to_dict()
    data["trajectory_count"] = 64
    with pytest.raises(ValueError, match="Noiseless training"):
        TrainingStageConfig.from_dict(data)

    with pytest.raises(ValueError, match="typed reference"):
        replace(
            stage,
            input_checkpoint_path="checkpoints/input.npy",
            input_checkpoint_checksum=_checksum("5"),
            input_checkpoint_pipeline_prefix=("phase2_pipeline_prefix_" + "6" * 64),
        )


def test_evaluation_factory_derives_identity_and_round_trips() -> None:
    """Evaluation factories derive circuit and row identities mechanically."""
    pipeline_result = _pipeline_result()
    evaluation = _evaluation(pipeline_result)
    assert PipelineEvaluationConfig.from_json(evaluation.to_json()) == evaluation
    assert evaluation.evaluation_row_id.startswith("phase2_evaluation_")
    evaluation.validate_against_pipeline(pipeline_result)

    with pytest.raises(ValueError, match="materialized_circuit_id"):
        replace(
            evaluation,
            materialized_circuit_id="phase2_circuit_" + "0" * 64,
        )


def test_pipeline_result_checksum_is_not_an_evaluation_row_field() -> None:
    """Observed pipeline-result content cannot alter a planned evaluation row."""
    evaluation = _evaluation()
    changed = replace(evaluation, pipeline_result_checksum=_checksum("0"))
    assert changed.evaluation_row_id == evaluation.evaluation_row_id
    assert changed.configuration_checksum != evaluation.configuration_checksum


def test_final_test_fanout_changes_only_evaluation_identity() -> None:
    """Repeated final tests share one training identity but have distinct rows."""
    pipeline_result = _pipeline_result()
    first = _evaluation(pipeline_result, seed=909, repetition=0)
    repeated = _evaluation(pipeline_result, seed=910, repetition=1)
    assert first.pipeline_training_id == repeated.pipeline_training_id
    assert first.evaluation_row_id != repeated.evaluation_row_id
    assert pipeline_result.training_id == first.pipeline_training_id


def test_noiseless_and_ballarin_evaluation_rules() -> None:
    """Noiseless and evaluation-only Ballarin conditions retain distinct rules."""
    pipeline_result = _pipeline_result()
    noiseless = PipelineEvaluationConfig.for_pipeline(
        pipeline=pipeline_result,
        materialized_circuit_checksum=_checksum("9"),
        test_noise_id="noiseless",
        noise_definition_version=_NOISE_VERSION,
        noise_strength_scale=None,
        tjm_dt=None,
        evaluation_seed=None,
        evaluation_seed_domain=None,
        repetition=0,
        trajectory_budget=0,
        evaluation_policy="fixed_sample",
        confidence_level=None,
        confidence_interval_method=None,
        sidecar_storage_policy="none",
        max_bond_dimension=64,
        svd_threshold=0.0,
        truncation_mode="discarded_weight",
        min_bond_dimension=1,
    )
    assert noiseless.test_noise_id == "noiseless"

    ballarin = PipelineEvaluationConfig.for_pipeline(
        pipeline=pipeline_result,
        materialized_circuit_checksum=_checksum("9"),
        test_noise_id="ballarin_coupled",
        noise_definition_version=_NOISE_VERSION,
        noise_strength_scale=None,
        tjm_dt=None,
        evaluation_seed=911,
        evaluation_seed_domain="screening_selection",
        repetition=0,
        trajectory_budget=32,
        evaluation_policy="fixed_sample",
        confidence_level=None,
        confidence_interval_method=None,
        sidecar_storage_policy="none",
        max_bond_dimension=64,
        svd_threshold=0.0,
        truncation_mode="discarded_weight",
        min_bond_dimension=1,
    )
    assert ballarin.test_noise_id == "ballarin_coupled"

    with pytest.raises(ValueError, match="Noiseless evaluation"):
        replace(noiseless, trajectory_budget=1)


def test_confidence_interval_policy_has_a_fixed_budget() -> None:
    """Confidence intervals use a fixed sample budget rather than optional stopping."""
    evaluation = _evaluation()
    interval = replace(
        evaluation,
        evaluation_policy="confidence_interval",
        confidence_level=0.95,
        confidence_interval_method="normal_clipped",
    )
    assert interval.trajectory_budget == 256
    with pytest.raises(ValueError, match="complete CI policy"):
        replace(interval, trajectory_budget=1)


def test_success_result_json_csv_and_path_identity_round_trips() -> None:
    """Successful evaluations round-trip through canonical JSON and union CSV."""
    result = _benchmark_result()
    assert pipeline_benchmark_record_from_json(result.to_json()) == result
    assert (
        result.evaluation_row_id
        == replace(
            result,
            materialized_circuit_path="other/final.json",
            trajectory_sidecar_path="other/final.npz",
        ).evaluation_row_id
    )

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=PIPELINE_CSV_COLUMNS)
    writer.writeheader()
    writer.writerow(result.to_csv_row())
    stream.seek(0)
    decoded = pipeline_benchmark_record_from_csv_row(next(csv.DictReader(stream)))
    assert decoded == result


def test_failure_result_json_csv_retains_row_without_fidelity() -> None:
    """Structured failures retain the planned row without fabricated outcomes."""
    config = _evaluation()
    failure = PipelineBenchmarkFailure.from_exception(
        config=config,
        failure_phase="evaluation",
        exception=RuntimeError("trajectory failed"),
        runtime_fingerprint_checksum=_checksum("7"),
        traceback="trace",
        retryable=True,
        attempt=2,
        materialized_circuit_path="circuits/final.json",
        materialized_circuit_checksum=config.materialized_circuit_checksum,
        wall_time_seconds=1.5,
    )
    assert pipeline_benchmark_record_from_json(failure.to_json()) == failure
    assert "test_noisy_fidelity" not in failure.to_dict()
    assert failure.evaluation_row_id == config.evaluation_row_id
    with pytest.raises(ValueError, match="non-whitespace"):
        replace(failure, traceback="")

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=PIPELINE_CSV_COLUMNS)
    writer.writeheader()
    writer.writerow(failure.to_csv_row())
    stream.seek(0)
    decoded = pipeline_benchmark_record_from_csv_row(next(csv.DictReader(stream)))
    assert decoded == failure


def test_csv_alias_and_schema_tampering_are_rejected() -> None:
    """CSV decoding rejects aliases and columns that diverge from typed config."""
    result = _benchmark_result()
    row = result.to_csv_row()
    row["pipeline_training_id"] = "phase2_training_" + "0" * 64
    with pytest.raises(ValueError, match="aliases"):
        pipeline_benchmark_record_from_csv_row(row)

    row = result.to_csv_row()
    del row["repetition"]
    with pytest.raises(ValueError, match="fields"):
        pipeline_benchmark_record_from_csv_row(row)

    row = result.to_csv_row()
    row["failure_phase"] = "evaluation"
    with pytest.raises(ValueError, match="failure-only"):
        pipeline_benchmark_record_from_csv_row(row)

    failure = PipelineBenchmarkFailure.from_exception(
        config=result.config,
        failure_phase="evaluation",
        exception=RuntimeError("failed"),
        runtime_fingerprint_checksum=_checksum("7"),
    )
    row = failure.to_csv_row()
    row["test_noisy_fidelity"] = 0.5
    with pytest.raises(ValueError, match="success-only"):
        pipeline_benchmark_record_from_csv_row(row)


def test_sealed_result_rejects_tampering_and_unknown_fields() -> None:
    """Sealed evaluation records reject changed content and schema extensions."""
    result = _benchmark_result()
    data = result.to_dict()
    data["test_noisy_fidelity"] = 0.1
    with pytest.raises(ValueError, match="checksum mismatch"):
        PipelineBenchmarkResult.from_dict(data)

    data = result.to_dict()
    data["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        PipelineBenchmarkResult.from_dict(data)


def test_evaluation_seed_must_use_the_role_domain_and_avoid_stage_streams() -> None:
    """Final-test streams use the correct role and never collide with training."""
    pipeline_result = _pipeline_result()
    evaluation = _evaluation(pipeline_result)
    with pytest.raises(ValueError, match="seed domain"):
        replace(
            evaluation,
            evaluation_seed_domain="confirmatory_test",
        ).validate_against_pipeline(pipeline_result)
    stage_seed = pipeline_result.config.stages[0].optimizer_seed
    assert stage_seed is not None
    with pytest.raises(ValueError, match="disjoint"):
        replace(
            evaluation,
            evaluation_seed=stage_seed,
        ).validate_against_pipeline(pipeline_result)


def test_typed_screening_resolution_binds_manifests_specs_seeds_and_endpoint() -> None:
    """A genuine WP15/WP16 screening cell resolves without caller-controlled aliases."""
    template = _template()
    screening, candidate, cell, target_manifest = _typed_screening_universe(template)
    spec = next(item for item in target_manifest.instances if item.target_instance_id == cell.target_instance_id)
    pipeline = template.resolve(
        target_namespace="phase2",
        target_manifest=target_manifest,
        target_instance_id=spec.target_instance_id,
        target_population_manifest_checksum=target_manifest.content_checksum,
        target_instance_spec_checksum=spec.content_checksum,
        target_family_id=spec.family_id,
        target_stratum_id=spec.stratum_id,
        qubit_count=spec.qubit_count,
        optimization_block_id=cell.cell_id,
        optimization_seed=cell.optimization_seed,
        data_role="screening_selection",
    )
    result = _pipeline_result(pipeline)
    evaluation = _evaluation(result, seed=cell.screening_seed)
    validate_screening_resolution(
        screening_manifest=screening,
        target_manifest=target_manifest,
        candidate=candidate,
        cell=cell,
        template=template,
        pipeline=pipeline,
        pipeline_result=result,
        evaluation=evaluation,
    )

    for field_name, forged_value in (
        ("policy_id", "forged_policy"),
        ("compiler_policy_id", "forged_compiler"),
        ("connectivity_id", "all_to_all"),
        ("routing_policy_id", "swap_routing"),
        ("optimization_level", 1),
        ("noise_placement", "compiled_native_gates"),
        ("parameter_source", "random_parameters"),
    ):
        materialization = dict(template.final_materialization_policy)
        materialization[field_name] = forged_value
        with pytest.raises(ValueError, match=rf"materialization {field_name}"):
            validate_screening_resolution(
                screening_manifest=screening,
                target_manifest=target_manifest,
                candidate=candidate,
                cell=cell,
                template=replace(template, final_materialization_policy=materialization),
                pipeline=pipeline,
            )

    for field_name in ("target_population_manifest_checksum", "target_instance_spec_checksum"):
        with pytest.raises(ValueError, match="typed target manifest"):
            validate_screening_resolution(
                screening_manifest=screening,
                target_manifest=target_manifest,
                candidate=candidate,
                cell=cell,
                template=template,
                pipeline=replace(pipeline, **{field_name: _checksum("0")}),
            )

    with pytest.raises(ValueError, match="seed domain"):
        validate_screening_resolution(
            screening_manifest=screening,
            target_manifest=target_manifest,
            candidate=candidate,
            cell=cell,
            template=template,
            pipeline=pipeline,
            pipeline_result=result,
            evaluation=replace(evaluation, evaluation_seed_domain="confirmatory_test"),
        )
    with pytest.raises(ValueError, match="fresh-test noisy-fidelity endpoint"):
        validate_screening_resolution(
            screening_manifest=screening,
            target_manifest=target_manifest,
            candidate=candidate,
            cell=cell,
            template=template,
            pipeline=pipeline,
            pipeline_result=result,
            evaluation=replace(evaluation, test_noise_id="dephasing_1s_all"),
        )
