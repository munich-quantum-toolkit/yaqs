# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused tests for checksum-sealed WP22A execution bindings."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation.phase2.binding_catalog import (
    ExecutableScopedBinding,
    RepositoryBindingCatalog,
)
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.competitor_optimizers import (
    build_parameter_shift_adam_layerwise_template,
    build_spsa_layerwise_template,
)
from benchmarks.state_preparation.phase2.execution_bindings import (
    FROZEN_IMPLEMENTATION_PLAN_COMMIT,
    PILOT_METHOD_IDS,
    SCREEN_CELL_COUNT,
    SCREEN_METHOD_IDS,
    SMOKE_EXECUTION_LIMITS_SCHEMA_VERSION,
    SMOKE_METHOD_IDS,
    BindingResourcePolicy,
    ControlledTrainingStage,
    EnergyAdaptSmokeSpec,
    ExecutionBudget,
    ExecutionImplementationArtifact,
    ExecutionRole,
    ImplementationKind,
    InferenceRole,
    ManifestRole,
    OperatorGrowthSmokeSpec,
    PipelineSmokeSpec,
    Preset,
    QubitTreatmentProjection,
    ScopedImplementationBinding,
    SmokeExecutionLimits,
    TargetScope,
    TrainingExecutionProfile,
)
from benchmarks.state_preparation.phase2.execution_bindings import (
    __all__ as execution_bindings_public_api,
)
from benchmarks.state_preparation.phase2.execution_protocol import (
    FreshEvaluationPolicy,
    OperationalProtocolAmendment,
    OperatorGrowthExecutionSpec,
    PilotDiagnosticPolicy,
)
from benchmarks.state_preparation.phase2.fair_controls import (
    build_fixed_depth_bmpd_crn_template,
    build_layerwise_bmpd_cross_crn_template,
    build_layerwise_bmpd_noiseless_template,
    build_layerwise_bmpd_resampled_template,
)
from benchmarks.state_preparation.phase2.implementation_catalog import (
    RepositoryImplementationCatalog,
    RepositoryRunnerAdapter,
)
from benchmarks.state_preparation.phase2.layerwise_bmpd import (
    bmpd_parameter_count,
    bmpd_topology_id,
    build_layerwise_bmpd_crn_v2_template,
)
from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineTemplate, TrainingStageTemplate
from benchmarks.state_preparation.phase2.protocol import TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM
from benchmarks.state_preparation.phase2.topdown_pruning import build_topdown_impact_iterative_template
from benchmarks.state_preparation.phase2.training_schedules import (
    CheckpointValidationPolicy,
    FrozenTrainingPolicyUniverse,
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

if TYPE_CHECKING:
    from collections.abc import Mapping


def _schedule(method_id: str, scope: TargetScope, preset: Preset) -> TrainingStrategySchedule:
    """Return one complete 200-update production schedule."""
    noisy = method_id not in {"layerwise_bmpd_noiseless", "energy_adapt_vqe"}
    if preset != "training-smoke":
        schedule_id = (
            "direct_noiseless_control"
            if not noisy
            else "resampled_each_update"
            if method_id in {"layerwise_bmpd_resampled", "spsa_layerwise"}
            else "direct_matched_fixed_crn"
        )
        return next(
            schedule
            for schedule in FrozenTrainingPolicyUniverse.frozen().schedules
            if schedule.schedule_id == schedule_id
        )
    update_count = 1
    trajectory_count = 1 if noisy else 0
    return TrainingStrategySchedule(
        schedule_id=f"training_smoke_{method_id}_{scope}",
        noise_continuation=NoiseStrengthContinuation(
            start_update=0,
            end_update=update_count - 1,
            start_strength_scale=1.0 if noisy else 0.0,
            target_strength_scale=1.0 if noisy else 0.0,
            interpolation="constant",
        ),
        trajectory_curriculum=TrajectoryCountCurriculum((TrajectoryCountStep(0, trajectory_count),)),
        sampling_policy=TrajectorySamplingPolicy(
            "resampled" if method_id in {"layerwise_bmpd_resampled", "spsa_layerwise"} else "fixed_crn"
        ),
        checkpoint_validation=CheckpointValidationPolicy(patience=None),
        phase_boundary=NoiselessPretrainNoisyFinetune(
            noiseless_pretrain_updates=0 if noisy else update_count,
            noisy_finetune_updates=update_count if noisy else 0,
        ),
        multistart=LimitedMultistartPlan(start_count=1, declared_cap=1),
        training_noise=(
            StandardNoiseMixture("matched", (NoiseMixtureComponent("depolarizing_1s_all", 1.0),))
            if noisy
            else StandardNoiseMixture("noiseless", ())
        ),
    )


def _candidate_checksum(method_id: str) -> str:
    """Return one width-independent publication candidate identity."""
    return canonical_checksum({"publication_candidate": method_id, "version": "wp22a"})


def _pipeline_payload(method_id: str, scope: TargetScope, preset: Preset) -> TrainingPipelineTemplate:
    """Return a strict repository pipeline payload for one synthetic binding."""
    smoke = preset == "training-smoke"
    training_count = 1 if smoke else 8
    validation_count = 1 if smoke else 256
    update_count = 1 if smoke else 200
    builders = {
        "layerwise_bmpd_crn_v2": lambda: build_layerwise_bmpd_crn_v2_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        ),
        "layerwise_bmpd_noiseless": lambda: build_layerwise_bmpd_noiseless_template(
            checkpoint_validation_trajectory_count=validation_count,
        ),
        "fixed_depth_bmpd_crn": lambda: build_fixed_depth_bmpd_crn_template(
            iteration_budget=update_count,
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        ),
        "layerwise_bmpd_resampled": lambda: build_layerwise_bmpd_resampled_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        ),
        "layerwise_bmpd_cross_crn": lambda: build_layerwise_bmpd_cross_crn_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        ),
        "parameter_shift_adam_layerwise": lambda: build_parameter_shift_adam_layerwise_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        ),
        "spsa_layerwise": lambda: build_spsa_layerwise_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        ),
        "impact_pruning_crn": lambda: build_topdown_impact_iterative_template(
            fine_tune_mode="fixed_crn",
            fine_tune_iterations=update_count,
            fine_tune_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        ),
    }
    template = builders[method_id]()
    if scope == "secondary_q12":
        stages: list[TrainingStageTemplate] = []
        for stage in template.stages:
            policy = dict(stage.stage_policy)
            output_depth = int(cast("str", policy["output_topology_id"]).rsplit("_d", maxsplit=1)[1])
            policy["output_topology_id"] = bmpd_topology_id(12, output_depth)
            policy["output_parameter_count"] = bmpd_parameter_count(12, output_depth)
            if policy["input_topology_id"] is not None:
                input_depth = int(cast("str", policy["input_topology_id"]).rsplit("_d", maxsplit=1)[1])
                policy["input_topology_id"] = bmpd_topology_id(12, input_depth)
                policy["input_parameter_count"] = bmpd_parameter_count(12, input_depth)
            stages.append(TrainingStageTemplate(stage_policy=policy, seed_bindings=stage.seed_bindings))
        template = replace(
            template,
            template_id=f"{template.template_id}_q12_projection",
            target_scope_id="secondary_q12",
            stages=tuple(stages),
        )
    return template


def _binding(
    method_id: str,
    scope: TargetScope,
    *,
    preset: Preset = "paper-pilot",
    normalized_compute_cap: float | None = None,
) -> ScopedImplementationBinding:
    """Return one complete synthetic binding using production policy objects."""
    schedule = _schedule(method_id, scope, preset)
    candidate_checksum = _candidate_checksum(method_id)
    implementation_kind: ImplementationKind = (
        "operator_growth_smoke"
        if method_id == "adapt_style_state_preparation" and preset == "training-smoke"
        else "operator_growth"
        if method_id == "adapt_style_state_preparation"
        else "tfim_operator_growth"
        if method_id == "energy_adapt_vqe"
        else "phase2_pipeline_smoke"
        if preset == "training-smoke"
        else "phase2_pipeline"
    )
    smoke_count = 2
    operator_spec = (
        OperatorGrowthExecutionSpec.for_screening(256)
        if method_id == "adapt_style_state_preparation" and preset == "paper-screen"
        else None
    )
    implementation_payload = (
        operator_spec
        if operator_spec is not None
        else OperatorGrowthSmokeSpec.frozen(smoke_count)
        if method_id == "adapt_style_state_preparation"
        else EnergyAdaptSmokeSpec.frozen(smoke_count)
        if method_id == "energy_adapt_vqe"
        else PipelineSmokeSpec.frozen(_pipeline_payload(method_id, scope, preset), smoke_count)
        if preset == "training-smoke"
        else _pipeline_payload(method_id, scope, preset)
    )
    assert implementation_payload is not None
    implementation_method_id = "topdown_impact_iterative" if method_id == "impact_pruning_crn" else method_id
    artifact = ExecutionImplementationArtifact(
        artifact_id=f"{preset}_{method_id}_{scope}",
        preset=preset,
        publication_method_id=method_id,
        implementation_kind=implementation_kind,
        implementation_method_id=implementation_method_id,
        target_scope_id=scope,
        strategy_schedule_checksum=schedule.content_checksum,
        implementation_payload=implementation_payload,
    )
    if preset == "training-smoke":
        evaluation_policies = (FreshEvaluationPolicy.smoke(smoke_count),)
        diagnostic = None
        primary_checksum = artifact.content_checksum
        inference_role = "primary"
        checkpoint_count = 0
    elif scope == "primary_q6":
        evaluation_policies = (
            FreshEvaluationPolicy.checkpoint_validation(),
            (
                FreshEvaluationPolicy.primary_q6_pilot()
                if preset == "paper-pilot"
                else FreshEvaluationPolicy.screening(256)
            ),
        )
        diagnostic = PilotDiagnosticPolicy.primary_q6() if preset == "paper-pilot" else None
        primary_checksum = artifact.content_checksum
        inference_role: InferenceRole = "primary"
        checkpoint_count = 256
    else:
        evaluation_policies = (
            FreshEvaluationPolicy.checkpoint_validation("secondary_q12"),
            FreshEvaluationPolicy.secondary_q12_pilot(),
        )
        diagnostic = PilotDiagnosticPolicy.secondary_q12()
        # The profile replaces this placeholder with the paired q6 checksum.
        primary_checksum = canonical_checksum({"primary": method_id})
        inference_role = "secondary_descriptive_only"
        checkpoint_count = 256
    screening_eligible = preset == "paper-screen" and scope == "primary_q6"
    promotion_eligible = screening_eligible and method_id != "layerwise_bmpd_noiseless"
    manifest_data_role: ManifestRole = (
        "screening_selection" if scope == "secondary_q12" or preset == "paper-screen" else "development"
    )
    execution_data_role: ExecutionRole = (
        "secondary_benchmark"
        if scope == "secondary_q12"
        else "screening_selection"
        if preset == "paper-screen"
        else "development"
    )
    return ScopedImplementationBinding(
        binding_id=f"binding_{preset}_{method_id}_{scope}",
        preset=preset,
        publication_candidate_schema_version="test.publication_candidate.v1",
        publication_candidate_checksum=candidate_checksum,
        publication_method_id=method_id,
        target_scope_id=scope,
        qubit_count=6 if scope == "primary_q6" else 12,
        manifest_data_role=manifest_data_role,
        execution_data_role=execution_data_role,
        implementation_artifact=artifact,
        strategy_schedule=schedule,
        controlled_stage=ControlledTrainingStage.complete_schedule(schedule, artifact),
        evaluation_policies=evaluation_policies,
        pilot_diagnostic_policy=diagnostic,
        execution_budget=ExecutionBudget(
            total_update_count=schedule.phase_boundary.total_updates,
            maximum_training_trajectory_count=max(
                step.trajectory_count for step in schedule.trajectory_curriculum.steps
            ),
            checkpoint_validation_trajectory_count=checkpoint_count,
            multistart_count=1,
            normalized_compute_cap=normalized_compute_cap,
        ),
        resource_policy=BindingResourcePolicy(),
        treatment_projection=QubitTreatmentProjection(
            publication_candidate_checksum=candidate_checksum,
            publication_method_id=method_id,
            target_scope_id=scope,
            primary_q6_implementation_checksum=primary_checksum,
            inference_role=inference_role,
            screening_eligible=screening_eligible,
            promotion_eligible=promotion_eligible,
        ),
        operator_growth_spec=operator_spec,
    )


def _pilot_bindings() -> tuple[ScopedImplementationBinding, ...]:
    """Return the exact three-method q6/q12 pilot registry."""
    bindings: list[ScopedImplementationBinding] = []
    for method_id in PILOT_METHOD_IDS:
        q6 = _binding(method_id, "primary_q6")
        q12 = _binding(method_id, "secondary_q12")
        q12 = replace(
            q12,
            treatment_projection=replace(
                q12.treatment_projection,
                primary_q6_implementation_checksum=q6.implementation_checksum,
            ),
        )
        bindings.extend((q6, q12))
    return tuple(bindings)


def _smoke_bindings() -> tuple[ScopedImplementationBinding, ...]:
    """Return all ten q6 implementation identities under tiny smoke budgets."""
    return tuple(_binding(method_id, "primary_q6", preset="training-smoke") for method_id in SMOKE_METHOD_IDS)


def _profile(bindings: tuple[ScopedImplementationBinding, ...]) -> TrainingExecutionProfile:
    """Return a frozen-amendment execution profile."""
    return TrainingExecutionProfile(
        profile_id="wp22a_test_profile",
        preset=bindings[0].preset,
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        implementation_plan_commit=FROZEN_IMPLEMENTATION_PLAN_COMMIT,
        operational_protocol_amendment=OperationalProtocolAmendment.frozen(),
        bindings=bindings,
    )


@pytest.fixture(scope="module")
def implementation_catalog() -> RepositoryImplementationCatalog:
    """Return the exact WP22B catalog matching the WP22A test profiles."""
    return RepositoryImplementationCatalog.frozen(
        screening_outer_trajectory_count=256,
        smoke_evaluation_trajectory_count=2,
    )


def _replace_primary_artifact(
    binding: ScopedImplementationBinding,
    artifact: ExecutionImplementationArtifact,
) -> ScopedImplementationBinding:
    """Rebind all primary-q6 identity links after replacing an artifact.

    Returns:
        The binding with every artifact-dependent identity replaced.
    """
    return replace(
        binding,
        implementation_artifact=artifact,
        controlled_stage=ControlledTrainingStage.complete_schedule(binding.strategy_schedule, artifact),
        treatment_projection=replace(
            binding.treatment_projection,
            primary_q6_implementation_checksum=artifact.content_checksum,
        ),
    )


def _replace_binding_schedule(
    binding: ScopedImplementationBinding,
    schedule: TrainingStrategySchedule,
) -> ScopedImplementationBinding:
    """Rebind an artifact and its budget to one changed schedule.

    Returns:
        The binding with schedule-dependent identities and limits replaced.
    """
    artifact = replace(
        binding.implementation_artifact,
        strategy_schedule_checksum=schedule.content_checksum,
    )
    return replace(
        binding,
        implementation_artifact=artifact,
        strategy_schedule=schedule,
        controlled_stage=ControlledTrainingStage.complete_schedule(schedule, artifact),
        execution_budget=replace(
            binding.execution_budget,
            total_update_count=schedule.phase_boundary.total_updates,
            maximum_training_trajectory_count=max(
                step.trajectory_count for step in schedule.trajectory_curriculum.steps
            ),
            multistart_count=schedule.multistart.start_count,
        ),
        treatment_projection=replace(
            binding.treatment_projection,
            primary_q6_implementation_checksum=artifact.content_checksum,
        ),
    )


def _replace_pipeline_stage(
    template: TrainingPipelineTemplate,
    stage_index: int,
    **policy_updates: object,
) -> TrainingPipelineTemplate:
    """Return a template with one validated stage policy changed."""
    stages = list(template.stages)
    policy = dict(stages[stage_index].stage_policy)
    policy.update(policy_updates)
    stages[stage_index] = TrainingStageTemplate(
        stage_policy=policy,
        seed_bindings=stages[stage_index].seed_bindings,
    )
    return replace(template, stages=tuple(stages))


def _retopologize_topdown(
    template: TrainingPipelineTemplate,
    *,
    depth: int,
    removal_count: int,
) -> TrainingPipelineTemplate:
    """Return a self-consistent but noncanonical top-down topology sequence."""
    stages: list[TrainingStageTemplate] = []
    current_topology = bmpd_topology_id(6, depth)
    current_parameter_count = bmpd_parameter_count(6, depth)
    current_round = 0
    for index, stage in enumerate(template.stages):
        policy = dict(stage.stage_policy)
        if index == 0:
            policy["output_topology_id"] = current_topology
            policy["output_parameter_count"] = current_parameter_count
        else:
            policy["input_topology_id"] = current_topology
            policy["input_parameter_count"] = current_parameter_count
            if policy["stage_kind"] == "prune":
                current_round += 1
                current_parameter_count -= removal_count
                current_topology = f"topdown_q6_d{depth}_r{current_round}_p{current_parameter_count}"
            policy["output_topology_id"] = current_topology
            policy["output_parameter_count"] = current_parameter_count
        stages.append(TrainingStageTemplate(stage_policy=policy, seed_bindings=stage.seed_bindings))
    return replace(template, stages=tuple(stages))


def test_pilot_profile_roundtrip_and_lookup_separate_publication_from_width() -> None:
    """The exact q6/q12 pilot roundtrips and resolves by all three key fields."""
    profile = _profile(_pilot_bindings())
    restored = TrainingExecutionProfile.from_json(profile.to_json())
    q6 = next(
        binding
        for binding in profile.bindings
        if binding.publication_method_id == PILOT_METHOD_IDS[0] and binding.target_scope_id == "primary_q6"
    )
    q12 = profile.binding("paper-pilot", q6.publication_candidate_checksum, "secondary_q12")

    assert restored == profile
    assert q12.publication_candidate_checksum == q6.publication_candidate_checksum
    assert q12.implementation_checksum != q6.implementation_checksum
    assert q12.treatment_projection.primary_q6_implementation_checksum == q6.implementation_checksum
    assert not q12.treatment_projection.screening_eligible
    assert not q12.treatment_projection.promotion_eligible


def test_q12_is_exactly_three_method_pilot_only_and_never_promotable() -> None:
    """Wrong q12 methods, roles, presets, or eligibility fail at construction."""
    with pytest.raises(ValueError, match=r"selected paper preset|three-method paper pilot"):
        _binding("spsa_layerwise", "secondary_q12")
    with pytest.raises(ValueError, match=r"selected paper preset|three-method paper pilot"):
        _binding(PILOT_METHOD_IDS[0], "secondary_q12", preset="paper-screen", normalized_compute_cap=10.0)

    q12 = _binding(PILOT_METHOD_IDS[0], "secondary_q12")
    with pytest.raises(ValueError, match="descriptive only"):
        replace(
            q12.treatment_projection,
            screening_eligible=True,
            promotion_eligible=True,
        )
    with pytest.raises(ValueError, match="three-method paper pilot"):
        replace(q12, execution_data_role="screening_selection")


def test_paper_screen_is_exact_nonadaptive_nine_method_q6_universe() -> None:
    """All and only nine q6 candidates form the amendment's 1,296 screen cells."""
    bindings = tuple(
        _binding(method_id, "primary_q6", preset="paper-screen", normalized_compute_cap=1_000.0)
        for method_id in SCREEN_METHOD_IDS
    )
    profile = _profile(bindings)

    assert profile.to_dict()["screen_design"] == {
        "adaptation": "none",
        "target_scope_id": "primary_q6",
        "candidate_count": 9,
        "target_count": 48,
        "optimization_seed_count": 3,
        "paired_block_count": 144,
        "cell_count": SCREEN_CELL_COUNT,
        "job_count": 1296,
    }
    promotion_by_method = {
        binding.publication_method_id: binding.treatment_projection.promotion_eligible for binding in bindings
    }
    assert not promotion_by_method["layerwise_bmpd_noiseless"]
    assert all(
        eligible for method_id, eligible in promotion_by_method.items() if method_id != "layerwise_bmpd_noiseless"
    )
    with pytest.raises(ValueError, match="exactly the nine"):
        _profile(bindings[:-1])


def test_profile_rejects_duplicate_lookup_key_and_incomplete_q12_projection() -> None:
    """Registry ambiguity and q12-to-q6 projection drift are rejected."""
    bindings = _pilot_bindings()
    with pytest.raises(ValueError, match="unique by preset"):
        _profile((*bindings, bindings[0]))

    q12_index = next(index for index, binding in enumerate(bindings) if binding.target_scope_id == "secondary_q12")
    changed = list(bindings)
    changed[q12_index] = replace(
        changed[q12_index],
        treatment_projection=replace(
            changed[q12_index].treatment_projection,
            primary_q6_implementation_checksum=canonical_checksum({"wrong": "primary"}),
        ),
    )
    with pytest.raises(ValueError, match="exact q6 treatment"):
        _profile(tuple(changed))


def test_implementation_artifact_rejects_untyped_sensitive_payload() -> None:
    """Free-form callbacks, target vectors, and entropy cannot enter typed implementation payloads."""
    schedule = _schedule(PILOT_METHOD_IDS[0], "primary_q6", "paper-pilot")
    with pytest.raises(TypeError, match="TrainingPipelineTemplate"):
        ExecutionImplementationArtifact(
            artifact_id="sensitive_implementation",
            preset="paper-pilot",
            publication_method_id=PILOT_METHOD_IDS[0],
            implementation_kind="phase2_pipeline",
            implementation_method_id=PILOT_METHOD_IDS[0],
            target_scope_id="primary_q6",
            strategy_schedule_checksum=schedule.content_checksum,
            implementation_payload=cast(
                "TrainingPipelineTemplate",
                {"target_vector": [0.0, 1.0], "role_master_entropy": 3},
            ),
        )


def test_nested_implementation_configuration_and_profile_amendment_are_checksum_bound() -> None:
    """Nested implementation drift and amendment drift cannot survive decoding."""
    profile = _profile(_pilot_bindings())
    artifact = profile.bindings[0].implementation_artifact
    tampered_artifact = artifact.to_dict()
    implementation = cast("dict[str, object]", tampered_artifact["implementation_payload"])
    implementation["method_version"] = "changed"
    with pytest.raises(ValueError, match="checksum mismatch"):
        ExecutionImplementationArtifact.from_dict(tampered_artifact)

    tampered_profile = profile.to_dict()
    tampered_profile["operational_protocol_amendment_checksum"] = canonical_checksum({"changed": True})
    with pytest.raises(ValueError, match="checksum mismatch"):
        TrainingExecutionProfile.from_dict(tampered_profile)


def test_smoke_profile_is_complete_and_preset_scope_changes_implementation_identity() -> None:
    """Smoke covers all ten identities and stays distinct from q6/q12 paper implementations."""
    smoke = _profile(_smoke_bindings())
    q6, q12 = _pilot_bindings()[:2]
    smoke_same_candidate = next(
        binding for binding in smoke.bindings if binding.publication_method_id == q6.publication_method_id
    )

    assert len(smoke.bindings) == 10
    assert {binding.publication_method_id for binding in smoke.bindings} == set(SMOKE_METHOD_IDS)
    assert q6.publication_candidate_checksum == q12.publication_candidate_checksum
    assert smoke_same_candidate.publication_candidate_checksum == q6.publication_candidate_checksum
    assert (
        len({q6.implementation_checksum, q12.implementation_checksum, smoke_same_candidate.implementation_checksum})
        == 3
    )
    assert TrainingExecutionProfile.from_json(smoke.to_json()) == smoke


def test_paper_bindings_use_only_exact_method_specific_rooted_schedules() -> None:
    """Production bindings reject development variants even when their outer budgets match."""
    expected_schedule_ids = {
        "layerwise_bmpd_crn_v2": "direct_matched_fixed_crn",
        "layerwise_bmpd_noiseless": "direct_noiseless_control",
        "fixed_depth_bmpd_crn": "direct_matched_fixed_crn",
        "layerwise_bmpd_resampled": "resampled_each_update",
        "layerwise_bmpd_cross_crn": "direct_matched_fixed_crn",
        "parameter_shift_adam_layerwise": "direct_matched_fixed_crn",
        "spsa_layerwise": "resampled_each_update",
        "adapt_style_state_preparation": "direct_matched_fixed_crn",
        "impact_pruning_crn": "direct_matched_fixed_crn",
    }
    for method_id, expected_schedule_id in expected_schedule_ids.items():
        binding = _binding(
            method_id,
            "primary_q6",
            preset="paper-screen",
            normalized_compute_cap=1_000.0,
        )
        assert binding.strategy_schedule.schedule_id == expected_schedule_id

    production = _binding(PILOT_METHOD_IDS[0], "primary_q6")
    development_schedule = next(
        schedule
        for schedule in FrozenTrainingPolicyUniverse.frozen().schedules
        if schedule.schedule_id == "continuation_fixed_crn"
    )
    with pytest.raises(ValueError, match="exact method-specific schedule"):
        replace(
            production.implementation_artifact,
            strategy_schedule_checksum=development_schedule.content_checksum,
        )

    operator_growth = _binding(
        "adapt_style_state_preparation",
        "primary_q6",
        preset="paper-screen",
        normalized_compute_cap=1_000.0,
    )
    with pytest.raises(ValueError, match="exact method-specific rooted training schedule"):
        _replace_binding_schedule(operator_growth, development_schedule)


def test_artifact_method_alias_and_preset_kind_matrix_fail_closed() -> None:
    """Only the impact alias and exact preset/kind/method/scope combinations are accepted."""
    pilot = _binding(PILOT_METHOD_IDS[0], "primary_q6")
    impact = _binding(
        "impact_pruning_crn",
        "primary_q6",
        preset="paper-screen",
        normalized_compute_cap=1_000.0,
    )
    assert impact.implementation_artifact.implementation_method_id == "topdown_impact_iterative"
    with pytest.raises(ValueError, match="maps exactly"):
        replace(impact.implementation_artifact, implementation_method_id="impact_pruning_crn")
    with pytest.raises(ValueError, match="maps exactly"):
        replace(pilot.implementation_artifact, implementation_method_id="topdown_impact_iterative")

    for deferred_preset in ("historical-layerwise-reproduction", "paper-confirm"):
        with pytest.raises(ValueError, match="deferred beyond WP22A"):
            replace(pilot.implementation_artifact, preset=deferred_preset)
        with pytest.raises(ValueError, match="intentionally unavailable"):
            replace(pilot, preset=deferred_preset)

    with pytest.raises(ValueError, match="restricted to paper pilot and screen"):
        replace(pilot.implementation_artifact, preset="training-smoke")
    q12 = _binding(PILOT_METHOD_IDS[0], "secondary_q12")
    with pytest.raises(ValueError, match="selected paper preset"):
        replace(q12.implementation_artifact, preset="paper-screen")

    operator_growth = _binding(
        "adapt_style_state_preparation",
        "primary_q6",
        preset="paper-screen",
        normalized_compute_cap=1_000.0,
    )
    with pytest.raises(TypeError, match="complete primary-q6 projector"):
        replace(operator_growth.implementation_artifact, preset="paper-pilot")

    pilot_template = cast("TrainingPipelineTemplate", pilot.implementation_artifact.implementation_payload)
    for operator_method in ("adapt_style_state_preparation", "energy_adapt_vqe"):
        relabeled = replace(pilot_template, method_id=operator_method)
        with pytest.raises(ValueError, match="selected paper preset"):
            replace(
                pilot.implementation_artifact,
                publication_method_id=operator_method,
                implementation_method_id=operator_method,
                implementation_payload=relabeled,
            )


def test_pipeline_family_topology_and_optimizer_substitutions_fail_closed() -> None:
    """Relabeled families, custom topology grammar, and noncanonical top-down evolution fail."""
    layerwise = _binding("layerwise_bmpd_crn_v2", "primary_q6")
    fixed = _binding("fixed_depth_bmpd_crn", "primary_q6")
    layerwise_template = cast(
        "TrainingPipelineTemplate",
        layerwise.implementation_artifact.implementation_payload,
    )
    fixed_template = cast("TrainingPipelineTemplate", fixed.implementation_artifact.implementation_payload)

    relabeled_fixed = replace(layerwise_template, method_id="fixed_depth_bmpd_crn")
    with pytest.raises(ValueError, match="exact implementation method family"):
        replace(fixed.implementation_artifact, implementation_payload=relabeled_fixed)

    custom_width = _replace_pipeline_stage(
        fixed_template,
        0,
        output_topology_id="custom_width6_d4",
    )
    with pytest.raises(ValueError, match="neither canonical BMPD nor canonical width-matched top-down"):
        replace(fixed.implementation_artifact, implementation_payload=custom_width)

    for competitor_method, optimizer_match in (
        ("parameter_shift_adam_layerwise", "optimizer"),
        ("spsa_layerwise", "optimizer|sampling"),
    ):
        competitor = _binding(
            competitor_method,
            "primary_q6",
            preset="paper-screen",
            normalized_compute_cap=1_000.0,
        )
        relabeled_competitor = replace(layerwise_template, method_id=competitor_method)
        with pytest.raises(ValueError, match=optimizer_match):
            replace(
                competitor.implementation_artifact,
                implementation_payload=relabeled_competitor,
            )

    impact = _binding(
        "impact_pruning_crn",
        "primary_q6",
        preset="paper-screen",
        normalized_compute_cap=1_000.0,
    )
    impact_template = cast("TrainingPipelineTemplate", impact.implementation_artifact.implementation_payload)
    noncanonical_depth = _retopologize_topdown(impact_template, depth=999, removal_count=1)
    with pytest.raises(ValueError, match="frozen default BMPD depth"):
        replace(impact.implementation_artifact, implementation_payload=noncanonical_depth)
    noncanonical_removal = _retopologize_topdown(impact_template, depth=4, removal_count=2)
    with pytest.raises(ValueError, match="fixed-count pruning evolution"):
        replace(impact.implementation_artifact, implementation_payload=noncanonical_removal)


def test_production_terminal_policy_is_exact_and_cross_bound_to_schedule() -> None:
    """Terminal update, noise, checkpoint, optimizer, and sampling semantics cannot drift."""
    expected_terminal = {
        "layerwise_bmpd_crn_v2": ("krotov", "crn_fixed", "independent"),
        "layerwise_bmpd_noiseless": ("krotov", "none", None),
        "fixed_depth_bmpd_crn": ("krotov", "crn_fixed", "independent"),
        "layerwise_bmpd_resampled": ("krotov", "resampled", "independent"),
        "layerwise_bmpd_cross_crn": ("krotov", "crn_fixed", "cross"),
        "parameter_shift_adam_layerwise": ("parameter_shift_adam", "crn_fixed", "independent"),
        "spsa_layerwise": ("spsa", "resampled", "independent"),
        "impact_pruning_crn": ("krotov", "crn_fixed", "independent"),
    }
    for method_id, expected in expected_terminal.items():
        binding = _binding(
            method_id,
            "primary_q6",
            preset="paper-screen",
            normalized_compute_cap=1_000.0,
        )
        template = cast("TrainingPipelineTemplate", binding.implementation_artifact.implementation_payload)
        terminal = template.stages[-1].stage_policy
        assert (terminal["optimizer_id"], terminal["sampling_policy"], terminal["trajectory_update"]) == expected
        assert terminal["iteration_budget"] == 200
        assert terminal["trajectory_count"] == (0 if method_id == "layerwise_bmpd_noiseless" else 8)

    binding = _binding("layerwise_bmpd_crn_v2", "primary_q6")
    template = cast("TrainingPipelineTemplate", binding.implementation_artifact.implementation_payload)
    wrong_updates = _replace_pipeline_stage(template, -1, iteration_budget=199)
    with pytest.raises(ValueError, match="contradicts its frozen schedule"):
        replace(binding.implementation_artifact, implementation_payload=wrong_updates)

    checkpoint = dict(cast("Mapping[str, object]", template.stages[-1].stage_policy["checkpoint_validation_policy"]))
    checkpoint["cadence"] = 11
    wrong_checkpoint = _replace_pipeline_stage(
        template,
        -1,
        checkpoint_validation_policy=checkpoint,
    )
    with pytest.raises(ValueError, match="contradicts its frozen schedule"):
        replace(binding.implementation_artifact, implementation_payload=wrong_checkpoint)


@pytest.mark.parametrize(
    ("method_id", "optimizer_field", "changed_value"),
    [
        ("layerwise_bmpd_crn_v2", "learning_rate", 9.9),
        ("parameter_shift_adam_layerwise", "learning_rate", 9.9),
        ("spsa_layerwise", "a", 9.9),
    ],
)
def test_production_optimizer_hyperparameters_are_exact_repository_policy(
    method_id: str,
    optimizer_field: str,
    changed_value: float,
) -> None:
    """Resealed Krotov, Adam, and SPSA hyperparameter drift fails at the artifact boundary."""
    preset: Preset = "paper-pilot" if method_id == "layerwise_bmpd_crn_v2" else "paper-screen"
    binding = _binding(
        method_id,
        "primary_q6",
        preset=preset,
        normalized_compute_cap=None if preset == "paper-pilot" else 1_000.0,
    )
    template = cast("TrainingPipelineTemplate", binding.implementation_artifact.implementation_payload)
    optimizer = dict(cast("Mapping[str, object]", template.stages[-1].stage_policy["optimizer_hyperparameters"]))
    optimizer[optimizer_field] = changed_value
    changed = _replace_pipeline_stage(
        template,
        -1,
        optimizer_hyperparameters=optimizer,
    )
    with pytest.raises(ValueError, match="exact repository-rooted template"):
        replace(binding.implementation_artifact, implementation_payload=changed)


def test_production_seed_bindings_and_minimally_adapted_family_substitution_fail() -> None:
    """Symbolic seed preimages and family-specific seed namespaces are frozen exactly."""
    pilot = _binding("layerwise_bmpd_crn_v2", "primary_q6")
    template = cast("TrainingPipelineTemplate", pilot.implementation_artifact.implementation_payload)
    stages = list(template.stages)
    terminal_seeds = dict(stages[-1].seed_bindings)
    terminal_seeds["training"] = "arbitrary_changed_training_stream"
    stages[-1] = TrainingStageTemplate(
        stage_policy=stages[-1].stage_policy,
        seed_bindings=terminal_seeds,
    )
    seed_drift = replace(template, stages=tuple(stages))
    with pytest.raises(ValueError, match="exact repository-rooted template"):
        replace(pilot.implementation_artifact, implementation_payload=seed_drift)

    source = _binding(
        "layerwise_bmpd_crn_v2",
        "primary_q6",
        preset="paper-screen",
        normalized_compute_cap=1_000.0,
    )
    resampled = _binding(
        "layerwise_bmpd_resampled",
        "primary_q6",
        preset="paper-screen",
        normalized_compute_cap=1_000.0,
    )
    source_template = cast("TrainingPipelineTemplate", source.implementation_artifact.implementation_payload)
    minimally_adapted = _replace_pipeline_stage(
        source_template,
        -1,
        sampling_policy="resampled",
    )
    minimally_adapted = replace(
        minimally_adapted,
        template_id="layerwise_bmpd_resampled_default",
        method_id="layerwise_bmpd_resampled",
    )
    with pytest.raises(ValueError, match="exact repository-rooted template"):
        replace(
            resampled.implementation_artifact,
            implementation_payload=minimally_adapted,
        )


def test_controlled_stage_names_the_actual_terminal_stage_or_smoke_sentinel() -> None:
    """A self-consistent but unrelated controlled-stage name cannot replace the implementation stage."""
    binding = _binding(PILOT_METHOD_IDS[0], "primary_q6")
    unrelated = ControlledTrainingStage(
        stage_id="controlled_unrelated_stage",
        implementation_stage_id="unrelated_stage",
        strategy_schedule_checksum=binding.strategy_schedule.content_checksum,
        start_update=0,
        stop_update_exclusive=200,
    )
    with pytest.raises(ValueError, match="exact implementation stage or adapter"):
        replace(binding, controlled_stage=unrelated)

    smoke = _binding(PILOT_METHOD_IDS[0], "primary_q6", preset="training-smoke")
    assert smoke.controlled_stage.implementation_stage_id == "pipeline_smoke_adapter"
    operator_smoke = _binding(
        "adapt_style_state_preparation",
        "primary_q6",
        preset="training-smoke",
    )
    assert operator_smoke.controlled_stage.implementation_stage_id == "operator_growth_reoptimization"


def test_smoke_wrappers_truthfully_freeze_effective_limits_and_runtime_boundary() -> None:
    """Every structural wrapper enforces its exact noisy/noiseless tiny-work boundary."""
    noisy = _binding("layerwise_bmpd_crn_v2", "primary_q6", preset="training-smoke")
    noiseless = _binding("layerwise_bmpd_noiseless", "primary_q6", preset="training-smoke")
    noisy_payload = cast("PipelineSmokeSpec", noisy.implementation_artifact.implementation_payload)
    noiseless_payload = cast("PipelineSmokeSpec", noiseless.implementation_artifact.implementation_payload)
    with pytest.raises(ValueError, match="zero exactly for the noiseless comparator"):
        replace(
            noisy_payload,
            effective_limits=SmokeExecutionLimits(2, training_trajectory_count=0),
        )
    with pytest.raises(ValueError, match="zero exactly for the noiseless comparator"):
        replace(
            noiseless_payload,
            effective_limits=SmokeExecutionLimits(2, training_trajectory_count=1),
        )

    operator = _binding("adapt_style_state_preparation", "primary_q6", preset="training-smoke")
    operator_payload = cast("OperatorGrowthSmokeSpec", operator.implementation_artifact.implementation_payload)
    with pytest.raises(ValueError, match="exactly one matched-noise"):
        replace(
            operator_payload,
            effective_limits=SmokeExecutionLimits(2, training_trajectory_count=0),
        )
    energy = _binding("energy_adapt_vqe", "primary_q6", preset="training-smoke")
    energy_payload = cast("EnergyAdaptSmokeSpec", energy.implementation_artifact.implementation_payload)
    with pytest.raises(ValueError, match="zero training trajectories"):
        replace(
            energy_payload,
            effective_limits=SmokeExecutionLimits(2, training_trajectory_count=1),
        )

    for payload in (noisy_payload, noiseless_payload, operator_payload, energy_payload):
        limits = payload.effective_limits
        assert limits.training_update_count == 1
        assert limits.checkpoint_validation_trajectory_count == 0
        assert limits.maximum_growth_steps == 1
        assert limits.reoptimization_steps_per_growth == 1
        assert limits.runtime_adapter_status == "wp22b_required_before_execution"
        assert not payload.promotion_eligible


def test_smoke_pipeline_reference_rejects_minimally_adapted_family_substitution() -> None:
    """Smoke cannot claim resampled-family coverage with relabeled CRN-v2 seeds and stages."""
    source = _binding("layerwise_bmpd_crn_v2", "primary_q6", preset="training-smoke")
    resampled = _binding("layerwise_bmpd_resampled", "primary_q6", preset="training-smoke")
    source_payload = cast("PipelineSmokeSpec", source.implementation_artifact.implementation_payload)
    resampled_payload = cast("PipelineSmokeSpec", resampled.implementation_artifact.implementation_payload)
    minimally_adapted = _replace_pipeline_stage(
        source_payload.structural_template_reference,
        -1,
        sampling_policy="resampled",
    )
    minimally_adapted = replace(
        minimally_adapted,
        template_id="layerwise_bmpd_resampled_default",
        method_id="layerwise_bmpd_resampled",
    )
    with pytest.raises(ValueError, match="exact repository-rooted tiny template"):
        replace(
            resampled_payload,
            structural_template_reference=minimally_adapted,
        )


def test_smoke_schedule_noise_sampling_checkpoint_and_multistart_are_exact() -> None:
    """Smoke uses exact method-specific sampling and rejects familiar-policy drift."""
    bindings = _smoke_bindings()
    for binding in bindings:
        expected_sampling = (
            "resampled"
            if binding.publication_method_id in {"layerwise_bmpd_resampled", "spsa_layerwise"}
            else "fixed_crn"
        )
        assert binding.strategy_schedule.sampling_policy.kind == expected_sampling

    noisy = _binding("layerwise_bmpd_crn_v2", "primary_q6", preset="training-smoke")
    dephasing = replace(
        noisy.strategy_schedule,
        training_noise=StandardNoiseMixture(
            "matched",
            (NoiseMixtureComponent("dephasing_1s_all", 1.0),),
        ),
    )
    with pytest.raises(ValueError, match="one-update structural preflight"):
        _replace_binding_schedule(noisy, dephasing)

    checkpoint_drift = replace(
        noisy.strategy_schedule,
        checkpoint_validation=CheckpointValidationPolicy(patience=None, min_delta=0.1),
    )
    with pytest.raises(ValueError, match="one-update structural preflight"):
        _replace_binding_schedule(noisy, checkpoint_drift)
    multistart_drift = replace(
        noisy.strategy_schedule,
        multistart=LimitedMultistartPlan(start_count=1, declared_cap=2),
    )
    with pytest.raises(ValueError, match="one-update structural preflight"):
        _replace_binding_schedule(noisy, multistart_drift)

    resampled = _binding("spsa_layerwise", "primary_q6", preset="training-smoke")
    wrong_sampling = replace(
        resampled.strategy_schedule,
        sampling_policy=TrajectorySamplingPolicy("fixed_crn"),
    )
    with pytest.raises(ValueError, match="one-update structural preflight"):
        _replace_binding_schedule(resampled, wrong_sampling)


def test_smoke_cannot_substitute_raw_production_payloads_and_public_api_is_complete() -> None:
    """Structural smoke kinds remain distinct and their schema types are public."""
    smoke = _binding("layerwise_bmpd_crn_v2", "primary_q6", preset="training-smoke")
    wrapper = cast("PipelineSmokeSpec", smoke.implementation_artifact.implementation_payload)
    with pytest.raises(ValueError, match="restricted to paper pilot and screen"):
        replace(
            smoke.implementation_artifact,
            implementation_kind="phase2_pipeline",
            implementation_payload=wrapper.structural_template_reference,
        )

    assert SMOKE_EXECUTION_LIMITS_SCHEMA_VERSION.endswith("smoke_execution_limits.v1")
    assert {
        "SMOKE_EXECUTION_LIMITS_SCHEMA_VERSION",
        "PIPELINE_SMOKE_SPEC_SCHEMA_VERSION",
        "OPERATOR_GROWTH_SMOKE_SPEC_SCHEMA_VERSION",
        "SmokeExecutionLimits",
        "PipelineSmokeSpec",
        "OperatorGrowthSmokeSpec",
    } <= set(execution_bindings_public_api)


def test_executable_binding_catalog_closes_pilot_q6_q12_and_roundtrips(
    implementation_catalog: RepositoryImplementationCatalog,
) -> None:
    """Every paired pilot treatment closes to one independently resolved runner."""
    profile = _profile(_pilot_bindings())
    catalog = RepositoryBindingCatalog.from_profile(profile, implementation_catalog)

    assert len(catalog.bindings) == 2 * len(PILOT_METHOD_IDS)
    for method_id in PILOT_METHOD_IDS:
        q6_binding = next(
            binding
            for binding in profile.bindings
            if binding.publication_method_id == method_id and binding.target_scope_id == "primary_q6"
        )
        q6 = catalog.resolve("paper-pilot", q6_binding.publication_candidate_checksum, "primary_q6")
        q12 = catalog.resolve("paper-pilot", q6_binding.publication_candidate_checksum, "secondary_q12")

        assert q6.implementation_entry is implementation_catalog.resolve("paper-pilot", method_id, "primary_q6")
        assert q12.implementation_entry is implementation_catalog.resolve("paper-pilot", method_id, "secondary_q12")
        assert q6.resolve_callable() is q6.implementation_entry.resolve_callable()
        assert q12.resolve_callable() is q12.implementation_entry.resolve_callable()
        assert q12.binding.treatment_projection.primary_q6_implementation_checksum == (
            q6.binding.implementation_checksum
        )
        assert not q12.binding.treatment_projection.promotion_eligible

    assert ExecutableScopedBinding.from_json(catalog.bindings[0].to_json()) == catalog.bindings[0]
    assert RepositoryBindingCatalog.from_json(catalog.to_json()) == catalog


def test_executable_binding_catalog_accepts_complete_smoke_and_screen_profiles(
    implementation_catalog: RepositoryImplementationCatalog,
) -> None:
    """Smoke and screen close exactly while analytic energy smoke stays non-promotional."""
    smoke_profile = _profile(_smoke_bindings())
    smoke = RepositoryBindingCatalog.from_profile(smoke_profile, implementation_catalog)
    screen_bindings = tuple(
        _binding(
            method_id,
            "primary_q6",
            preset="paper-screen",
            normalized_compute_cap=1_000.0,
        )
        for method_id in SCREEN_METHOD_IDS
    )
    screen = RepositoryBindingCatalog.from_profile(
        _profile(screen_bindings),
        implementation_catalog,
    )

    assert len(smoke.bindings) == len(SMOKE_METHOD_IDS)
    assert len(screen.bindings) == len(SCREEN_METHOD_IDS)
    assert all(link.binding.target_scope_id == "primary_q6" for link in screen.bindings)
    energy_binding = next(
        binding for binding in smoke_profile.bindings if binding.publication_method_id == "energy_adapt_vqe"
    )
    energy = smoke.resolve(
        "training-smoke",
        energy_binding.publication_candidate_checksum,
        "primary_q6",
    )
    energy_payload = cast("EnergyAdaptSmokeSpec", energy.binding.implementation_artifact.implementation_payload)
    assert energy_payload.effective_limits.training_trajectory_count == 0
    assert energy.binding.execution_budget.maximum_training_trajectory_count == 0
    assert not energy.binding.treatment_projection.screening_eligible
    assert not energy.binding.treatment_projection.promotion_eligible
    assert energy.smoke_runtime_program() == energy.implementation_entry.smoke_runtime_program()

    with pytest.raises(KeyError, match="No executable scoped binding"):
        screen.resolve(
            "paper-confirm",
            screen.bindings[0].binding.publication_candidate_checksum,
            "primary_q6",
        )
    assert screen.implementation_catalog.resolve(
        "paper-confirm",
        "layerwise_bmpd_noiseless",
        "primary_q6",
    ) is screen.implementation_catalog.resolve(
        "paper-screen",
        "layerwise_bmpd_noiseless",
        "primary_q6",
    )
    tampered = screen.to_dict()
    tampered["paper_confirm_execution_authorized"] = True
    tampered_without_checksum = {key: value for key, value in tampered.items() if key != "content_checksum"}
    tampered["content_checksum"] = canonical_checksum(tampered_without_checksum)
    with pytest.raises(ValueError, match="confirmation authorization"):
        RepositoryBindingCatalog.from_dict(tampered)


def test_executable_binding_catalog_rejects_missing_duplicate_foreign_and_payload_drift(
    implementation_catalog: RepositoryImplementationCatalog,
) -> None:
    """Catalog closure fails before execution for incomplete or forged profile links."""
    pilot = RepositoryBindingCatalog.from_profile(
        _profile(_pilot_bindings()),
        implementation_catalog,
    )
    with pytest.raises(ValueError, match="exactly and in order"):
        replace(pilot, bindings=pilot.bindings[:-1])
    with pytest.raises(ValueError, match="must be unique"):
        replace(pilot, bindings=(*pilot.bindings[:-1], pilot.bindings[0]))

    smoke = RepositoryBindingCatalog.from_profile(
        _profile(_smoke_bindings()),
        implementation_catalog,
    )
    with pytest.raises(ValueError, match="exactly and in order"):
        replace(pilot, bindings=(smoke.bindings[0], *pilot.bindings[1:]))

    base = _binding("layerwise_bmpd_crn_v2", "primary_q6", preset="training-smoke")
    changed_schedule = replace(
        base.strategy_schedule,
        schedule_id="training_smoke_layerwise_bmpd_crn_v2_primary_q6_drift",
    )
    with pytest.raises(ValueError, match="exact same strategy schedule"):
        ExecutableScopedBinding.close(
            _replace_binding_schedule(base, changed_schedule),
            implementation_catalog.resolve(
                "training-smoke",
                "layerwise_bmpd_crn_v2",
                "primary_q6",
            ),
        )

    payload = cast("PipelineSmokeSpec", base.implementation_artifact.implementation_payload)
    changed_payload = PipelineSmokeSpec.frozen(payload.structural_template_reference, 3)
    changed_artifact = replace(base.implementation_artifact, implementation_payload=changed_payload)
    changed_binding = replace(
        base,
        implementation_artifact=changed_artifact,
        controlled_stage=ControlledTrainingStage.complete_schedule(
            base.strategy_schedule,
            changed_artifact,
        ),
        evaluation_policies=(FreshEvaluationPolicy.smoke(3),),
        treatment_projection=replace(
            base.treatment_projection,
            primary_q6_implementation_checksum=changed_artifact.content_checksum,
        ),
    )
    canonical_entry = implementation_catalog.resolve(
        "training-smoke",
        "layerwise_bmpd_crn_v2",
        "primary_q6",
    )
    with pytest.raises(ValueError, match="exact same typed payload"):
        ExecutableScopedBinding.close(changed_binding, canonical_entry)

    wrong_adapter = RepositoryRunnerAdapter.for_artifact(
        implementation_catalog.resolve(
            "training-smoke",
            "fixed_depth_bmpd_crn",
            "primary_q6",
        ).implementation_artifact
    )
    with pytest.raises(ValueError, match="re-derived"):
        ExecutableScopedBinding(
            binding=base,
            implementation_entry=canonical_entry,
            runner_adapter=wrong_adapter,
        )
