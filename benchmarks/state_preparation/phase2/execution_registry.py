# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deterministic production execution registries for the WP22 paper stages.

This module projects the repository-owned implementation catalog onto the
publication candidates and scoped bindings required by the pilot and screen.
It introduces no scientific choices: counts, schedules, policies, candidate
membership, and seed derivations all come from already frozen protocol types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from .binding_catalog import RepositoryBindingCatalog
from .execution_bindings import (
    FROZEN_IMPLEMENTATION_PLAN_COMMIT,
    PILOT_METHOD_IDS,
    SCREEN_METHOD_IDS,
    BindingResourcePolicy,
    ControlledTrainingStage,
    ExecutionBudget,
    QubitTreatmentProjection,
    ScopedImplementationBinding,
    TrainingExecutionProfile,
)
from .execution_protocol import (
    CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
    FreshEvaluationPolicy,
    OperationalProtocolAmendment,
    OperatorGrowthExecutionSpec,
    PilotDiagnosticPolicy,
)
from .implementation_catalog import ExecutableImplementationEntry, RepositoryImplementationCatalog
from .pilot import PilotContrastBinding
from .pipeline import TrainingPipelineTemplate
from .protocol import InitialPreregistration, SampleSizeDesign
from .screening import PilotNormalizedComputeCalibration
from .screening_design import WP22CandidateConfiguration
from .targets import TargetPopulationManifest
from .training_orchestration import TrainingRunPlan
from .training_schedules import (
    SCREEN_OPTIMIZATION_SEED_POLICY_ID,
    SCREENING_ROOT_SEED_POLICY_ID,
    ExecutionSeedPolicySuite,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

ExecutionRegistryCompilation = tuple[
    tuple[WP22CandidateConfiguration, ...],
    RepositoryBindingCatalog,
]
ProductionPreset = Literal["paper-pilot", "paper-screen"]


def _frozen_amendment(preregistration: InitialPreregistration) -> OperationalProtocolAmendment:
    """Return the amendment after binding it to the supplied preregistration.

    Returns:
        The trusted operational amendment.

    Raises:
        TypeError: If ``preregistration`` has the wrong protocol type.
        ValueError: If it is not the preregistration frozen by the amendment.
    """
    if not isinstance(preregistration, InitialPreregistration):
        msg = "preregistration must be an InitialPreregistration."
        raise TypeError(msg)
    amendment = OperationalProtocolAmendment.frozen()
    if amendment.preregistration_checksum != preregistration.content_checksum:
        msg = "The production registry requires the preregistration frozen by the operational amendment."
        raise ValueError(msg)
    return amendment


def _candidate_from_entry(entry: ExecutableImplementationEntry) -> WP22CandidateConfiguration:
    """Project one primary-q6 catalog entry onto its publication identity.

    Returns:
        The exact publication candidate addressed by the execution entry.

    Raises:
        TypeError: If the production entry has an unsupported payload type.
        ValueError: If it is not a primary-q6 pilot or screening entry.
    """
    if not isinstance(entry, ExecutableImplementationEntry):
        msg = "entry must be an ExecutableImplementationEntry."
        raise TypeError(msg)
    if entry.target_scope_id != "primary_q6" or entry.preset not in {"paper-pilot", "paper-screen"}:
        msg = "Publication candidates are derived only from primary-q6 paper entries."
        raise ValueError(msg)
    artifact = entry.implementation_artifact
    payload = artifact.implementation_payload
    if isinstance(payload, TrainingPipelineTemplate):
        return WP22CandidateConfiguration.from_pipeline(
            payload,
            strategy_schedule_checksum=entry.strategy_schedule.content_checksum,
            publication_method_id=entry.publication_method_id,
        )
    if isinstance(payload, OperatorGrowthExecutionSpec):
        return WP22CandidateConfiguration(
            method_id=entry.publication_method_id,
            implementation_kind="operator_growth",
            implementation_method_id=payload.method_id,
            implementation_schema_version=payload.schema_version,
            implementation_checksum=payload.content_checksum,
            strategy_schedule_checksum=entry.strategy_schedule.content_checksum,
            resource_stratum_id="primary_cap_12",
            noisy_training=True,
            matching_projection_checksum=None,
            publication_mapping={},
        )
    msg = "Paper execution entries must contain a pipeline template or operator-growth execution spec."
    raise TypeError(msg)


def _evaluation_policies(
    entry: ExecutableImplementationEntry,
    screening_outer_trajectory_count: int,
) -> tuple[FreshEvaluationPolicy, ...]:
    """Return the frozen evaluation policies for one catalog entry.

    Returns:
        The exact checkpoint-plus-stage evaluation policy sequence.
    """
    if entry.preset == "paper-pilot":
        return (
            FreshEvaluationPolicy.checkpoint_validation(entry.target_scope_id),
            (
                FreshEvaluationPolicy.primary_q6_pilot()
                if entry.target_scope_id == "primary_q6"
                else FreshEvaluationPolicy.secondary_q12_pilot()
            ),
        )
    return (
        FreshEvaluationPolicy.checkpoint_validation(),
        FreshEvaluationPolicy.screening(screening_outer_trajectory_count),
    )


def _pilot_diagnostic_policy(entry: ExecutableImplementationEntry) -> PilotDiagnosticPolicy | None:
    """Return the exact pilot diagnostic policy, or ``None`` for screening.

    Returns:
        The role-specific pilot diagnostic policy when applicable.
    """
    if entry.preset != "paper-pilot":
        return None
    if entry.target_scope_id == "primary_q6":
        return PilotDiagnosticPolicy.primary_q6()
    return PilotDiagnosticPolicy.secondary_q12()


def _scoped_binding(
    entry: ExecutableImplementationEntry,
    candidate: WP22CandidateConfiguration,
    *,
    primary_q6_implementation_checksum: str,
    screening_outer_trajectory_count: int,
    normalized_compute_cap: float | None,
) -> ScopedImplementationBinding:
    """Close one catalog entry to its publication and treatment projection.

    Returns:
        A complete production scoped implementation binding.
    """
    schedule = entry.strategy_schedule
    artifact = entry.implementation_artifact
    is_screen = entry.preset == "paper-screen"
    is_primary = entry.target_scope_id == "primary_q6"
    policies = _evaluation_policies(entry, screening_outer_trajectory_count)
    return ScopedImplementationBinding(
        binding_id=f"wp22h_{entry.preset}_{entry.publication_method_id}_{entry.target_scope_id}",
        preset=entry.preset,
        publication_candidate_schema_version=candidate.schema_version,
        publication_candidate_checksum=candidate.content_checksum,
        publication_method_id=entry.publication_method_id,
        target_scope_id=entry.target_scope_id,
        qubit_count=6 if is_primary else 12,
        manifest_data_role=("screening_selection" if is_screen or not is_primary else "development"),
        execution_data_role=(
            "screening_selection" if is_screen else "development" if is_primary else "secondary_benchmark"
        ),
        implementation_artifact=artifact,
        strategy_schedule=schedule,
        controlled_stage=ControlledTrainingStage.complete_schedule(schedule, artifact),
        evaluation_policies=policies,
        pilot_diagnostic_policy=_pilot_diagnostic_policy(entry),
        execution_budget=ExecutionBudget(
            total_update_count=schedule.phase_boundary.total_updates,
            maximum_training_trajectory_count=max(
                step.trajectory_count for step in schedule.trajectory_curriculum.steps
            ),
            checkpoint_validation_trajectory_count=CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
            multistart_count=schedule.multistart.start_count,
            normalized_compute_cap=normalized_compute_cap,
        ),
        resource_policy=BindingResourcePolicy(),
        treatment_projection=QubitTreatmentProjection(
            publication_candidate_checksum=candidate.content_checksum,
            publication_method_id=entry.publication_method_id,
            target_scope_id=entry.target_scope_id,
            primary_q6_implementation_checksum=primary_q6_implementation_checksum,
            inference_role="primary" if is_primary else "secondary_descriptive_only",
            screening_eligible=is_screen,
            promotion_eligible=is_screen and entry.publication_method_id != "layerwise_bmpd_noiseless",
        ),
        operator_growth_spec=(
            cast("OperatorGrowthExecutionSpec", artifact.implementation_payload)
            if artifact.implementation_kind == "operator_growth"
            else None
        ),
    )


def _compile_registry(
    preregistration: InitialPreregistration,
    *,
    preset: ProductionPreset,
    screening_outer_trajectory_count: int,
    normalized_compute_cap: float | None,
) -> ExecutionRegistryCompilation:
    """Compile one production preset from the canonical repository catalog.

    Returns:
        Ordered publication candidates and their executable binding catalog.
    """
    amendment = _frozen_amendment(preregistration)
    implementation_catalog = RepositoryImplementationCatalog.frozen(
        screening_outer_trajectory_count=screening_outer_trajectory_count,
    )
    method_ids: Sequence[str] = PILOT_METHOD_IDS if preset == "paper-pilot" else SCREEN_METHOD_IDS
    candidates = tuple(
        _candidate_from_entry(implementation_catalog.resolve(preset, method_id, "primary_q6"))
        for method_id in method_ids
    )
    candidate_by_method = {candidate.method_id: candidate for candidate in candidates}
    bindings: list[ScopedImplementationBinding] = []
    for method_id in method_ids:
        candidate = candidate_by_method[method_id]
        primary_entry = implementation_catalog.resolve(preset, method_id, "primary_q6")
        primary_checksum = primary_entry.implementation_artifact.content_checksum
        bindings.append(
            _scoped_binding(
                primary_entry,
                candidate,
                primary_q6_implementation_checksum=primary_checksum,
                screening_outer_trajectory_count=screening_outer_trajectory_count,
                normalized_compute_cap=normalized_compute_cap,
            )
        )
        if preset == "paper-pilot":
            bindings.append(
                _scoped_binding(
                    implementation_catalog.resolve(preset, method_id, "secondary_q12"),
                    candidate,
                    primary_q6_implementation_checksum=primary_checksum,
                    screening_outer_trajectory_count=screening_outer_trajectory_count,
                    normalized_compute_cap=None,
                )
            )
    profile = TrainingExecutionProfile(
        profile_id=f"wp22h_{preset.replace('-', '_')}_execution_profile",
        preset=preset,
        preregistration_checksum=preregistration.content_checksum,
        implementation_plan_commit=FROZEN_IMPLEMENTATION_PLAN_COMMIT,
        operational_protocol_amendment=amendment,
        bindings=tuple(bindings),
    )
    return candidates, RepositoryBindingCatalog.from_profile(profile, implementation_catalog)


def build_paper_pilot_execution_registry(
    preregistration: InitialPreregistration,
) -> ExecutionRegistryCompilation:
    """Compile the exact three-method, paired-q6/q12 pilot registry.

    The complete repository catalog also contains dormant screen entries. Its
    outer count is set to the frozen protocol minimum because pilot artifacts
    and policies do not depend on that future pilot-derived screen choice.

    Returns:
        The three publication candidates and six-binding executable catalog.
    """
    amendment = _frozen_amendment(preregistration)
    return _compile_registry(
        preregistration,
        preset="paper-pilot",
        screening_outer_trajectory_count=amendment.outer_trajectory_count_min,
        normalized_compute_cap=None,
    )


def build_paper_screen_execution_registry(
    preregistration: InitialPreregistration,
    sample_size_design: SampleSizeDesign,
    pilot_calibration: PilotNormalizedComputeCalibration,
) -> ExecutionRegistryCompilation:
    """Compile the exact nine-method q6 screen from pilot-derived authorities.

    Args:
        preregistration: Frozen Phase II protocol.
        sample_size_design: Pilot-derived fixed outer trajectory count.
        pilot_calibration: Pilot-only normalized-compute cap calibration.

    Returns:
        The nine publication candidates and nine-binding executable catalog.

    Raises:
        TypeError: If either pilot-derived artifact has the wrong type.
        ValueError: If either artifact belongs to another preregistration or
            the fixed outer count is outside the frozen bounds.
    """
    amendment = _frozen_amendment(preregistration)
    if not isinstance(sample_size_design, SampleSizeDesign):
        msg = "sample_size_design must be a SampleSizeDesign."
        raise TypeError(msg)
    if not isinstance(pilot_calibration, PilotNormalizedComputeCalibration):
        msg = "pilot_calibration must be a PilotNormalizedComputeCalibration."
        raise TypeError(msg)
    if (
        sample_size_design.preregistration_checksum != preregistration.content_checksum
        or pilot_calibration.preregistration_checksum != preregistration.content_checksum
    ):
        msg = "Screen registry authorities belong to a different preregistration."
        raise ValueError(msg)
    outer_count = sample_size_design.fixed_test_trajectory_count
    if not amendment.outer_trajectory_count_min <= outer_count <= amendment.outer_trajectory_count_max:
        msg = "The pilot-derived fixed outer trajectory count is outside the frozen protocol bounds."
        raise ValueError(msg)
    return _compile_registry(
        preregistration,
        preset="paper-screen",
        screening_outer_trajectory_count=outer_count,
        normalized_compute_cap=pilot_calibration.normalized_compute_cap,
    )


def build_paper_pilot_contrast_bindings(
    pilot_plan: TrainingRunPlan,
) -> tuple[PilotContrastBinding, ...]:
    """Derive both frozen pilot planning contrasts from the exact pilot plan.

    Returns:
        Noisy-versus-noiseless followed by fixed-depth-versus-noisy-v2.

    Raises:
        TypeError: If ``pilot_plan`` has the wrong type.
        ValueError: If fixed-depth has no unique plan configuration.
    """
    if not isinstance(pilot_plan, TrainingRunPlan):
        msg = "pilot_plan must be a TrainingRunPlan."
        raise TypeError(msg)
    fixed_depth_configurations = {
        job.candidate_configuration_checksum for job in pilot_plan.jobs if job.method_id == "fixed_depth_bmpd_crn"
    }
    if len(fixed_depth_configurations) != 1:
        msg = "The paper pilot must contain one unique fixed-depth planning configuration."
        raise ValueError(msg)
    fixed_depth_configuration = next(iter(fixed_depth_configurations))
    return (
        PilotContrastBinding.noisy_vs_noiseless(pilot_plan),
        PilotContrastBinding.promoted_vs_layerwise_v2(
            pilot_plan,
            treatment_method_id="fixed_depth_bmpd_crn",
            treatment_configuration_checksum=fixed_depth_configuration,
        ),
    )


def derive_screening_optimization_seeds(
    preregistration: InitialPreregistration,
) -> tuple[int, ...]:
    """Derive the frozen ordered three-seed screening schedule.

    Returns:
        Three distinct domain-separated unsigned optimization seeds.

    Raises:
        RuntimeError: If the reviewed derivation unexpectedly collides.
    """
    amendment = _frozen_amendment(preregistration)
    suite = ExecutionSeedPolicySuite.frozen()
    seeds = tuple(
        suite.derive(
            SCREEN_OPTIMIZATION_SEED_POLICY_ID,
            {
                "preregistration_checksum": preregistration.content_checksum,
                "seed_index": index,
            },
        )
        for index in range(amendment.screen_optimization_seed_count)
    )
    if len(set(seeds)) != len(seeds):
        msg = "The frozen screening optimization policy derived duplicate seeds."
        raise RuntimeError(msg)
    return seeds


def derive_screening_seed_root(
    preregistration: InitialPreregistration,
    screen_execution_profile: TrainingExecutionProfile,
    screening_target_manifest: TargetPopulationManifest,
) -> int:
    """Derive the outer-screen root from the exact profile and target manifest.

    Returns:
        The domain-separated unsigned screening root seed.

    Raises:
        TypeError: If profile or target manifest has the wrong type.
        ValueError: If either input is outside the frozen primary-q6 screen.
    """
    _frozen_amendment(preregistration)
    if not isinstance(screen_execution_profile, TrainingExecutionProfile):
        msg = "screen_execution_profile must be a TrainingExecutionProfile."
        raise TypeError(msg)
    if not isinstance(screening_target_manifest, TargetPopulationManifest):
        msg = "screening_target_manifest must be a TargetPopulationManifest."
        raise TypeError(msg)
    if (
        screen_execution_profile.preset != "paper-screen"
        or screen_execution_profile.preregistration_checksum != preregistration.content_checksum
        or screening_target_manifest.preregistration_checksum != preregistration.content_checksum
        or screening_target_manifest.data_role != "screening_selection"
        or screening_target_manifest.population_scope != "primary_q6"
    ):
        msg = "Screening seed authority differs from the frozen primary-q6 screen."
        raise ValueError(msg)
    return ExecutionSeedPolicySuite.frozen().derive(
        SCREENING_ROOT_SEED_POLICY_ID,
        {
            "preregistration_checksum": preregistration.content_checksum,
            "screen_execution_profile_checksum": screen_execution_profile.content_checksum,
            "screening_target_manifest_checksum": screening_target_manifest.content_checksum,
        },
    )


__all__ = [
    "ExecutionRegistryCompilation",
    "build_paper_pilot_contrast_bindings",
    "build_paper_pilot_execution_registry",
    "build_paper_screen_execution_registry",
    "derive_screening_optimization_seeds",
    "derive_screening_seed_root",
]
