# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Checksum-sealed WP22A implementation bindings and execution profiles.

These records keep the publication candidate identity separate from its typed
width- and preset-specific implementation reference. Production references
bind executable repository pipelines; smoke references remain structural
until WP22B supplies their runtime adapters. They contain no targets or entropy
and are therefore safe to inspect before target materialization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, cast

from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    verify_sealed_mapping,
)
from .competitor_optimizers import (
    build_parameter_shift_adam_layerwise_template,
    build_spsa_layerwise_template,
)
from .execution_protocol import (
    CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
    PRODUCTION_TRAINING_TRAJECTORY_COUNT,
    PRODUCTION_UPDATE_COUNT,
    FreshEvaluationPolicy,
    OperationalProtocolAmendment,
    OperatorGrowthExecutionSpec,
    PilotDiagnosticPolicy,
)
from .fair_controls import (
    build_fixed_depth_bmpd_crn_template,
    build_layerwise_bmpd_cross_crn_template,
    build_layerwise_bmpd_noiseless_template,
    build_layerwise_bmpd_resampled_template,
)
from .layerwise_bmpd import (
    bmpd_parameter_count,
    bmpd_topology_id,
    build_layerwise_bmpd_crn_v2_template,
)
from .operator_growth import (
    ENERGY_ADAPT_METHOD_ID,
    OperatorGrowthSpec,
    OperatorPoolSpec,
    build_tfim_real_operator_pool,
)
from .pipeline import TrainingPipelineTemplate, TrainingStageTemplate
from .topdown_pruning import TOPDOWN_DEFAULT_DEEP_DEPTH, build_topdown_impact_iterative_template
from .training_schedules import (
    CheckpointValidationPolicy,
    FrozenTrainingPolicyUniverse,
    LimitedMultistartPlan,
    TrainingStrategySchedule,
)
from .validation import (
    require_bool,
    require_checksum,
    require_exact_keys,
    require_float,
    require_git_commit,
    require_int,
    require_mapping,
    require_slug,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

EXECUTION_IMPLEMENTATION_ARTIFACT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.execution_implementation_artifact.v1"
SMOKE_EXECUTION_LIMITS_SCHEMA_VERSION = "yaqs.state_preparation.phase2.smoke_execution_limits.v1"
PIPELINE_SMOKE_SPEC_SCHEMA_VERSION = "yaqs.state_preparation.phase2.pipeline_smoke_spec.v1"
OPERATOR_GROWTH_SMOKE_SPEC_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_smoke_spec.v1"
ENERGY_ADAPT_SMOKE_SPEC_SCHEMA_VERSION = "yaqs.state_preparation.phase2.energy_adapt_smoke_spec.v1"
CONTROLLED_TRAINING_STAGE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.controlled_training_stage.v1"
EXECUTION_BUDGET_SCHEMA_VERSION = "yaqs.state_preparation.phase2.execution_budget.v1"
RESOURCE_POLICY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.binding_resource_policy.v1"
QUBIT_TREATMENT_PROJECTION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.qubit_treatment_projection.v1"
SCOPED_IMPLEMENTATION_BINDING_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scoped_implementation_binding.v1"
TRAINING_EXECUTION_PROFILE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.training_execution_profile.v1"

FROZEN_IMPLEMENTATION_PLAN_COMMIT = "93fae0bb1cd4a2af12a7ac11e3383a1180bd4f3e"

TRAINING_PRESETS = (
    "training-smoke",
    "historical-layerwise-reproduction",
    "paper-pilot",
    "paper-screen",
    "paper-confirm",
)
PILOT_METHOD_IDS = (
    "layerwise_bmpd_crn_v2",
    "layerwise_bmpd_noiseless",
    "fixed_depth_bmpd_crn",
)
SCREEN_METHOD_IDS = (
    "layerwise_bmpd_crn_v2",
    "layerwise_bmpd_noiseless",
    "fixed_depth_bmpd_crn",
    "layerwise_bmpd_resampled",
    "layerwise_bmpd_cross_crn",
    "parameter_shift_adam_layerwise",
    "spsa_layerwise",
    "adapt_style_state_preparation",
    "impact_pruning_crn",
)
SMOKE_METHOD_IDS = (*SCREEN_METHOD_IDS, "energy_adapt_vqe")

# The paper pilot and nonadaptive screen use only these method-specific members
# of the wider development-policy universe.  Continuation, curriculum, rolling,
# mixture, and multistart variants remain development-only until a later package
# explicitly freezes a candidate using them.
_PAPER_SCHEDULE_ID_BY_METHOD = {
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
_IMPLEMENTATION_METHOD_BY_PUBLICATION_METHOD = {
    **{method_id: method_id for method_id in SMOKE_METHOD_IDS},
    "impact_pruning_crn": "topdown_impact_iterative",
}
_PIPELINE_PUBLICATION_METHOD_IDS = frozenset(SMOKE_METHOD_IDS) - {
    "adapt_style_state_preparation",
    "energy_adapt_vqe",
}
_WP22A_DEFERRED_PRESETS = frozenset({"historical-layerwise-reproduction", "paper-confirm"})
_TOPDOWN_TOPOLOGY_PATTERN = re.compile(
    r"^topdown_q(?P<qubits>[1-9][0-9]*)_d(?P<depth>[1-9][0-9]*)_r(?P<round>[1-9][0-9]*)_p(?P<parameters>[1-9][0-9]*)$"
)
_PIPELINE_TERMINAL_SAMPLING_BY_METHOD = {
    "layerwise_bmpd_crn_v2": ("crn_fixed", "independent"),
    "layerwise_bmpd_noiseless": ("none", None),
    "fixed_depth_bmpd_crn": ("crn_fixed", "independent"),
    "layerwise_bmpd_resampled": ("resampled", "independent"),
    "layerwise_bmpd_cross_crn": ("crn_fixed", "cross"),
    "parameter_shift_adam_layerwise": ("crn_fixed", "independent"),
    "spsa_layerwise": ("resampled", "independent"),
    "impact_pruning_crn": ("crn_fixed", "independent"),
}
_LAYERWISE_PIPELINE_STAGE_SEQUENCE = (
    ("grow_d1", "optimize"),
    ("grow_d2", "grow"),
    ("grow_d3", "grow"),
    ("grow_d4", "grow"),
    ("final_finetune", "optimize"),
)
_PIPELINE_STAGE_SEQUENCE_BY_IMPLEMENTATION_METHOD = {
    "layerwise_bmpd_crn_v2": _LAYERWISE_PIPELINE_STAGE_SEQUENCE,
    "layerwise_bmpd_noiseless": _LAYERWISE_PIPELINE_STAGE_SEQUENCE,
    "fixed_depth_bmpd_crn": (("direct_depth4_noisy_training", "optimize"),),
    "layerwise_bmpd_resampled": _LAYERWISE_PIPELINE_STAGE_SEQUENCE,
    "layerwise_bmpd_cross_crn": _LAYERWISE_PIPELINE_STAGE_SEQUENCE,
    "parameter_shift_adam_layerwise": _LAYERWISE_PIPELINE_STAGE_SEQUENCE,
    "spsa_layerwise": _LAYERWISE_PIPELINE_STAGE_SEQUENCE,
    "topdown_impact_iterative": (
        ("deep_pretrain", "optimize"),
        ("prune_round_1", "prune"),
        ("relax_round_1", "optimize"),
        ("prune_round_2", "prune"),
        ("final_finetune", "optimize"),
    ),
}
_PIPELINE_TERMINAL_OPTIMIZER_BY_IMPLEMENTATION_METHOD = {
    **dict.fromkeys(_PIPELINE_STAGE_SEQUENCE_BY_IMPLEMENTATION_METHOD, "krotov"),
    "parameter_shift_adam_layerwise": "parameter_shift_adam",
    "spsa_layerwise": "spsa",
}

SCREEN_CANDIDATE_COUNT = 9
SCREEN_TARGET_COUNT = 48
SCREEN_OPTIMIZATION_SEED_COUNT = 3
SCREEN_PAIRED_BLOCK_COUNT = 144
SCREEN_CELL_COUNT = 1296
SCREEN_JOB_COUNT = 1296

Preset = Literal[
    "training-smoke",
    "historical-layerwise-reproduction",
    "paper-pilot",
    "paper-screen",
    "paper-confirm",
]
TargetScope = Literal["primary_q6", "secondary_q12"]
ManifestRole = Literal["development", "screening_selection", "confirmatory"]
ExecutionRole = Literal["development", "screening_selection", "secondary_benchmark", "confirmatory"]
ImplementationKind = Literal[
    "phase2_pipeline",
    "phase2_pipeline_smoke",
    "operator_growth",
    "operator_growth_smoke",
    "tfim_operator_growth",
]
InferenceRole = Literal["primary", "secondary_descriptive_only"]

_PRESETS = frozenset(TRAINING_PRESETS)
_TARGET_SCOPES = frozenset({"primary_q6", "secondary_q12"})
_MANIFEST_ROLES = frozenset({"development", "screening_selection", "confirmatory"})
_EXECUTION_ROLES = frozenset({"development", "screening_selection", "secondary_benchmark", "confirmatory"})
_IMPLEMENTATION_KINDS = frozenset({
    "phase2_pipeline",
    "phase2_pipeline_smoke",
    "operator_growth",
    "operator_growth_smoke",
    "tfim_operator_growth",
})
_IMPLEMENTATION_ARTIFACT_KEYS = frozenset({
    "schema_version",
    "artifact_id",
    "preset",
    "publication_method_id",
    "implementation_kind",
    "implementation_method_id",
    "target_scope_id",
    "strategy_schedule_checksum",
    "implementation_payload",
    "implementation_payload_checksum",
    "content_checksum",
})
_ENERGY_ADAPT_SMOKE_SPEC_KEYS = frozenset({
    "schema_version",
    "target_scope_id",
    "qubit_count",
    "method_id",
    "pool",
    "growth_spec",
    "outer_evaluation_policy",
    "effective_limits",
    "promotion_eligible",
    "content_checksum",
})
_SMOKE_EXECUTION_LIMITS_KEYS = frozenset({
    "schema_version",
    "training_update_count",
    "training_trajectory_count",
    "checkpoint_validation_trajectory_count",
    "maximum_growth_steps",
    "reoptimization_steps_per_growth",
    "outer_evaluation_trajectory_count",
    "runtime_adapter_status",
    "content_checksum",
})
_OPERATOR_GROWTH_SMOKE_SPEC_KEYS = frozenset({
    "schema_version",
    "target_scope_id",
    "qubit_count",
    "method_id",
    "production_pool",
    "production_growth_spec",
    "outer_evaluation_policy",
    "effective_limits",
    "promotion_eligible",
    "content_checksum",
})
_PIPELINE_SMOKE_SPEC_KEYS = frozenset({
    "schema_version",
    "target_scope_id",
    "qubit_count",
    "method_id",
    "structural_template_reference",
    "outer_evaluation_policy",
    "effective_limits",
    "runtime_stage_sentinel",
    "promotion_eligible",
    "content_checksum",
})
_CONTROLLED_STAGE_KEYS = frozenset({
    "schema_version",
    "stage_id",
    "implementation_stage_id",
    "strategy_schedule_checksum",
    "start_update",
    "stop_update_exclusive",
    "schedule_application",
    "optimizer_state_rule",
    "resume_rule",
    "content_checksum",
})
_EXECUTION_BUDGET_KEYS = frozenset({
    "schema_version",
    "total_update_count",
    "maximum_training_trajectory_count",
    "checkpoint_validation_trajectory_count",
    "multistart_count",
    "normalized_compute_cap",
    "content_checksum",
})
_RESOURCE_POLICY_KEYS = frozenset({
    "schema_version",
    "metric",
    "cap_per_chain_edge",
    "comparison_rule",
    "compiler_policy_id",
    "connectivity",
    "routing_policy",
    "normalized_compute_cap_source",
    "residual_gap_reporting",
    "content_checksum",
})
_TREATMENT_PROJECTION_KEYS = frozenset({
    "schema_version",
    "publication_candidate_checksum",
    "publication_method_id",
    "target_scope_id",
    "primary_q6_implementation_checksum",
    "inference_role",
    "screening_eligible",
    "promotion_eligible",
    "content_checksum",
})
_BINDING_KEYS = frozenset({
    "schema_version",
    "binding_id",
    "preset",
    "publication_candidate_schema_version",
    "publication_candidate_checksum",
    "publication_method_id",
    "target_scope_id",
    "qubit_count",
    "manifest_data_role",
    "execution_data_role",
    "implementation_artifact",
    "strategy_schedule",
    "controlled_stage",
    "evaluation_policies",
    "pilot_diagnostic_policy",
    "execution_budget",
    "resource_policy",
    "treatment_projection",
    "operator_growth_spec",
    "content_checksum",
})
_PROFILE_KEYS = frozenset({
    "schema_version",
    "profile_id",
    "preset",
    "preregistration_checksum",
    "implementation_plan_commit",
    "operational_protocol_amendment",
    "operational_protocol_amendment_checksum",
    "screen_design",
    "bindings",
    "content_checksum",
})


def _sealed(payload: dict[str, object]) -> dict[str, object]:
    """Attach the checksum of a complete JSON-native payload.

    Returns:
        A detached mapping containing the payload and its checksum.
    """
    return {**payload, "content_checksum": canonical_checksum(payload)}


def _verify(
    value: object,
    *,
    keys: frozenset[str],
    version: str,
    name: str,
) -> Mapping[str, object]:
    """Verify one exact checksum-sealed schema.

    Returns:
        The frozen, verified schema mapping.

    Raises:
        ValueError: If the schema version is unsupported.
    """
    mapping = verify_sealed_mapping(value, expected_keys=keys, name=name)
    if mapping["schema_version"] != version:
        msg = f"{name} uses an unsupported schema version."
        raise ValueError(msg)
    return mapping


def _require_preset(value: object, name: str = "preset") -> Preset:
    """Return one exact repository-owned WP22 preset.

    Raises:
        ValueError: If the value is not a supported preset.
    """
    preset = require_slug(value, name)
    match preset:
        case "training-smoke":
            return "training-smoke"
        case "historical-layerwise-reproduction":
            return "historical-layerwise-reproduction"
        case "paper-pilot":
            return "paper-pilot"
        case "paper-screen":
            return "paper-screen"
        case "paper-confirm":
            return "paper-confirm"
        case _:
            msg = f"{name} is not a supported WP22 preset."
            raise ValueError(msg)


def _require_target_scope(value: object, name: str = "target_scope_id") -> TargetScope:
    """Return one preregistered q6 or q12 target scope.

    Raises:
        ValueError: If the value is not a preregistered target scope.
    """
    scope = require_slug(value, name)
    if scope not in _TARGET_SCOPES:
        msg = f"{name} must be primary_q6 or secondary_q12."
        raise ValueError(msg)
    return cast("TargetScope", scope)


def _validate_pipeline_width(template: TrainingPipelineTemplate, scope: TargetScope) -> None:
    """Reject BMPD topology or parameter counts that disagree with target scope.

    Raises:
        ValueError: If topology syntax, width, depth, or parameter evolution is invalid.
    """
    qubit_count = 6 if scope == "primary_q6" else 12
    for stage in template.stages:
        policy = stage.stage_policy
        for prefix in ("input", "output"):
            topology = policy[f"{prefix}_topology_id"]
            parameter_count = policy[f"{prefix}_parameter_count"]
            if topology is None:
                if parameter_count != 0:
                    msg = "An absent pipeline input topology must have zero parameters."
                    raise ValueError(msg)
                continue
            topology_text = cast("str", topology)
            if topology_text.startswith("bmpd_q"):
                try:
                    depth = int(topology_text.rsplit("_d", maxsplit=1)[1])
                except (IndexError, ValueError) as error:
                    msg = "BMPD topology identifiers must encode their exact depth."
                    raise ValueError(msg) from error
                if topology_text != bmpd_topology_id(qubit_count, depth) or parameter_count != bmpd_parameter_count(
                    qubit_count, depth
                ):
                    msg = "Pipeline BMPD topology or parameter count disagrees with target scope width."
                    raise ValueError(msg)
                continue
            topdown_match = _TOPDOWN_TOPOLOGY_PATTERN.fullmatch(topology_text)
            if (
                template.method_id != "topdown_impact_iterative"
                or topdown_match is None
                or int(topdown_match.group("qubits")) != qubit_count
                or int(topdown_match.group("parameters")) != parameter_count
            ):
                msg = "Phase2 pipeline topology is neither canonical BMPD nor canonical width-matched top-down."
                raise ValueError(msg)
    if template.method_id != "topdown_impact_iterative":
        return
    root_topology = cast("str", template.stages[0].stage_policy["output_topology_id"])
    root_parameter_count = cast("int", template.stages[0].stage_policy["output_parameter_count"])
    if root_topology != bmpd_topology_id(
        qubit_count, TOPDOWN_DEFAULT_DEEP_DEPTH
    ) or root_parameter_count != bmpd_parameter_count(qubit_count, TOPDOWN_DEFAULT_DEEP_DEPTH):
        msg = "Top-down pipeline root must use the frozen default BMPD depth and parameter count."
        raise ValueError(msg)
    expected_sequence = (
        ("deep_pretrain", "optimize"),
        ("prune_round_1", "prune"),
        ("relax_round_1", "optimize"),
        ("prune_round_2", "prune"),
        ("final_finetune", "optimize"),
    )
    if tuple((stage.stage_id, stage.stage_policy["stage_kind"]) for stage in template.stages) != expected_sequence:
        msg = "Top-down pipeline stages differ from the frozen two-round impact sequence."
        raise ValueError(msg)
    current_round = 0
    current_topology = root_topology
    current_parameter_count = root_parameter_count
    for stage in template.stages[1:]:
        policy = stage.stage_policy
        stage_kind = policy["stage_kind"]
        input_topology = cast("str", policy["input_topology_id"])
        output_topology = cast("str", policy["output_topology_id"])
        input_parameter_count = cast("int", policy["input_parameter_count"])
        output_parameter_count = cast("int", policy["output_parameter_count"])
        if input_topology != current_topology or input_parameter_count != current_parameter_count:
            msg = "Top-down input topology does not match the exact preceding stage output."
            raise ValueError(msg)
        if stage_kind == "prune":
            current_round += 1
            current_parameter_count -= 1
            current_topology = (
                f"topdown_q{qubit_count}_d{TOPDOWN_DEFAULT_DEEP_DEPTH}_r{current_round}_p{current_parameter_count}"
            )
        if output_topology != current_topology or output_parameter_count != current_parameter_count:
            msg = "Top-down output topology differs from the exact fixed-count pruning evolution."
            raise ValueError(msg)


def _validate_pipeline_method_family(template: TrainingPipelineTemplate) -> None:
    """Reject relabeled templates that do not match their exact method family.

    Raises:
        ValueError: If stage, optimizer, sampling, or noise policy differs from the method family.
    """
    expected_sequence = _PIPELINE_STAGE_SEQUENCE_BY_IMPLEMENTATION_METHOD.get(template.method_id)
    actual_sequence = tuple((stage.stage_id, stage.stage_policy["stage_kind"]) for stage in template.stages)
    if expected_sequence is None or actual_sequence != expected_sequence:
        msg = "Pipeline stage sequence differs from its exact implementation method family."
        raise ValueError(msg)
    publication_method_id = (
        "impact_pruning_crn" if template.method_id == "topdown_impact_iterative" else template.method_id
    )
    expected_sampling = _PIPELINE_TERMINAL_SAMPLING_BY_METHOD[publication_method_id]
    expected_optimizer = _PIPELINE_TERMINAL_OPTIMIZER_BY_IMPLEMENTATION_METHOD[template.method_id]
    terminal = template.stages[-1].stage_policy
    noiseless = publication_method_id == "layerwise_bmpd_noiseless"
    if (
        terminal["optimizer_id"] != expected_optimizer
        or terminal["sampling_policy"] != expected_sampling[0]
        or terminal["trajectory_update"] != expected_sampling[1]
        or terminal["training_noise_id"] != ("noiseless" if noiseless else "depolarizing_1s_all")
        or (terminal["trajectory_count"] == 0) != noiseless
    ):
        msg = "Pipeline terminal optimizer, sampling, or noise differs from its exact method family."
        raise ValueError(msg)
    optimizer = require_mapping(terminal["optimizer_hyperparameters"], "optimizer_hyperparameters")
    if "sampling_policy" in optimizer and optimizer["sampling_policy"] != expected_sampling[0]:
        msg = "Pipeline optimizer sampling differs from its exact method family."
        raise ValueError(msg)
    if (
        "gradient_trajectory_count" in optimizer
        and optimizer["gradient_trajectory_count"] != terminal["trajectory_count"]
    ):
        msg = "Pipeline optimizer trajectory count differs from its terminal stage."
        raise ValueError(msg)


def _canonical_production_pipeline(
    publication_method_id: str,
    scope: TargetScope,
) -> TrainingPipelineTemplate:
    """Build the exact repository-rooted production template for one treatment.

    Returns:
        The canonical q6 template or its exact secondary-q12 projection.

    Raises:
        ValueError: If the method or requested secondary projection is unsupported.
    """
    if publication_method_id == "layerwise_bmpd_crn_v2":
        template = build_layerwise_bmpd_crn_v2_template(
            training_trajectory_count=PRODUCTION_TRAINING_TRAJECTORY_COUNT,
            checkpoint_validation_trajectory_count=CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
        )
    elif publication_method_id == "layerwise_bmpd_noiseless":
        template = build_layerwise_bmpd_noiseless_template(
            checkpoint_validation_trajectory_count=CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
        )
    elif publication_method_id == "fixed_depth_bmpd_crn":
        template = build_fixed_depth_bmpd_crn_template(
            iteration_budget=PRODUCTION_UPDATE_COUNT,
            training_trajectory_count=PRODUCTION_TRAINING_TRAJECTORY_COUNT,
            checkpoint_validation_trajectory_count=CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
        )
    elif publication_method_id == "layerwise_bmpd_resampled":
        template = build_layerwise_bmpd_resampled_template(
            training_trajectory_count=PRODUCTION_TRAINING_TRAJECTORY_COUNT,
            checkpoint_validation_trajectory_count=CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
        )
    elif publication_method_id == "layerwise_bmpd_cross_crn":
        template = build_layerwise_bmpd_cross_crn_template(
            training_trajectory_count=PRODUCTION_TRAINING_TRAJECTORY_COUNT,
            checkpoint_validation_trajectory_count=CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
        )
    elif publication_method_id == "parameter_shift_adam_layerwise":
        template = build_parameter_shift_adam_layerwise_template(
            training_trajectory_count=PRODUCTION_TRAINING_TRAJECTORY_COUNT,
            checkpoint_validation_trajectory_count=CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
        )
    elif publication_method_id == "spsa_layerwise":
        template = build_spsa_layerwise_template(
            training_trajectory_count=PRODUCTION_TRAINING_TRAJECTORY_COUNT,
            checkpoint_validation_trajectory_count=CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
        )
    elif publication_method_id == "impact_pruning_crn":
        template = build_topdown_impact_iterative_template(
            fine_tune_mode="fixed_crn",
            fine_tune_iterations=PRODUCTION_UPDATE_COUNT,
            fine_tune_trajectory_count=PRODUCTION_TRAINING_TRAJECTORY_COUNT,
            checkpoint_validation_trajectory_count=CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
        )
    else:
        msg = "Publication method has no canonical production pipeline."
        raise ValueError(msg)
    if scope == "primary_q6":
        return template
    if publication_method_id not in PILOT_METHOD_IDS:
        msg = "Only pilot methods have a canonical secondary-q12 projection."
        raise ValueError(msg)
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
    return replace(
        template,
        template_id=f"{template.template_id}_q12_projection",
        target_scope_id="secondary_q12",
        stages=tuple(stages),
    )


def _canonical_smoke_pipeline(implementation_method_id: str) -> TrainingPipelineTemplate:
    """Build the exact repository-rooted q6 structural smoke reference.

    Returns:
        The canonical tiny structural template for the implementation method.

    Raises:
        ValueError: If the implementation method has no pipeline smoke reference.
    """
    if implementation_method_id == "layerwise_bmpd_crn_v2":
        return build_layerwise_bmpd_crn_v2_template(
            training_trajectory_count=1,
            checkpoint_validation_trajectory_count=1,
        )
    if implementation_method_id == "layerwise_bmpd_noiseless":
        return build_layerwise_bmpd_noiseless_template(
            checkpoint_validation_trajectory_count=1,
        )
    if implementation_method_id == "fixed_depth_bmpd_crn":
        return build_fixed_depth_bmpd_crn_template(
            iteration_budget=1,
            training_trajectory_count=1,
            checkpoint_validation_trajectory_count=1,
        )
    if implementation_method_id == "layerwise_bmpd_resampled":
        return build_layerwise_bmpd_resampled_template(
            training_trajectory_count=1,
            checkpoint_validation_trajectory_count=1,
        )
    if implementation_method_id == "layerwise_bmpd_cross_crn":
        return build_layerwise_bmpd_cross_crn_template(
            training_trajectory_count=1,
            checkpoint_validation_trajectory_count=1,
        )
    if implementation_method_id == "parameter_shift_adam_layerwise":
        return build_parameter_shift_adam_layerwise_template(
            training_trajectory_count=1,
            checkpoint_validation_trajectory_count=1,
        )
    if implementation_method_id == "spsa_layerwise":
        return build_spsa_layerwise_template(
            training_trajectory_count=1,
            checkpoint_validation_trajectory_count=1,
        )
    if implementation_method_id == "topdown_impact_iterative":
        return build_topdown_impact_iterative_template(
            fine_tune_mode="fixed_crn",
            fine_tune_iterations=1,
            fine_tune_trajectory_count=1,
            checkpoint_validation_trajectory_count=1,
        )
    msg = "Implementation method has no canonical structural smoke pipeline."
    raise ValueError(msg)


def _validate_production_terminal_policy(
    template: TrainingPipelineTemplate,
    publication_method_id: str,
    schedule: TrainingStrategySchedule,
) -> None:
    """Bind a production pipeline's terminal stage to its exact schedule and budget.

    Raises:
        ValueError: If no policy exists or the terminal stage contradicts the frozen policy.
    """
    expected_sampling = _PIPELINE_TERMINAL_SAMPLING_BY_METHOD.get(publication_method_id)
    if expected_sampling is None:
        msg = "Publication method has no frozen terminal pipeline policy."
        raise ValueError(msg)
    terminal = template.stages[-1].stage_policy
    sampling_policy, trajectory_update = expected_sampling
    noiseless = publication_method_id == "layerwise_bmpd_noiseless"
    expected_checkpoint = {
        "schema_version": "yaqs.state_preparation.phase2.checkpoint_validation_config.v1",
        "noise_id": "depolarizing_1s_all",
        "noise_definition_version": "yaqs.state_preparation.noise.v1",
        "noise_strength_scale": 1.0,
        "tjm_dt": 1.0,
        "trajectory_count": CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
        "sampling_policy": "crn_fixed",
        "ensemble_refresh_interval": None,
        "cadence": 10,
        "selection_rule": "best_validation_fidelity",
        "tie_breaker": "earliest_iteration",
    }
    checkpoint = require_mapping(terminal["checkpoint_validation_policy"], "checkpoint_validation_policy")
    optimizer = require_mapping(terminal["optimizer_hyperparameters"], "optimizer_hyperparameters")
    expected_schedule_sampling = "resampled" if sampling_policy == "resampled" else "fixed_crn"
    valid = (
        terminal["stage_kind"] == "optimize"
        and terminal["iteration_budget"] == PRODUCTION_UPDATE_COUNT
        and terminal["training_noise_id"] == ("noiseless" if noiseless else "depolarizing_1s_all")
        and terminal["noise_definition_version"] == "yaqs.state_preparation.noise.v1"
        and terminal["noise_strength_scale"] == (None if noiseless else 1.0)
        and terminal["tjm_dt"] == (None if noiseless else 1.0)
        and terminal["trajectory_count"] == (0 if noiseless else PRODUCTION_TRAINING_TRAJECTORY_COUNT)
        and terminal["trajectory_update"] == trajectory_update
        and terminal["sampling_policy"] == sampling_policy
        and dict(checkpoint) == expected_checkpoint
        and schedule.sampling_policy.kind == expected_schedule_sampling
    )
    if "sampling_policy" in optimizer:
        valid = valid and optimizer["sampling_policy"] == sampling_policy
    if "gradient_trajectory_count" in optimizer:
        valid = valid and optimizer["gradient_trajectory_count"] == PRODUCTION_TRAINING_TRAJECTORY_COUNT
    if not valid:
        msg = "Production terminal pipeline stage contradicts its frozen schedule, noise, or evaluation budget."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SmokeExecutionLimits:
    """Truthful effective limits for structural smoke preflight.

    These limits override no production protocol.  They describe the future
    WP22B smoke adapter's effective work and explicitly mark that adapter as a
    prerequisite before the profile can execute.
    """

    outer_evaluation_trajectory_count: int
    training_trajectory_count: int = 1
    training_update_count: int = field(default=1, init=False)
    checkpoint_validation_trajectory_count: int = field(default=0, init=False)
    maximum_growth_steps: int = field(default=1, init=False)
    reoptimization_steps_per_growth: int = field(default=1, init=False)
    runtime_adapter_status: str = field(default="wp22b_required_before_execution", init=False)
    schema_version: str = field(default=SMOKE_EXECUTION_LIMITS_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require a positive fixed outer count and exact tiny effective limits.

        Raises:
            ValueError: If the training trajectory count is not zero or one.
        """
        object.__setattr__(
            self,
            "outer_evaluation_trajectory_count",
            require_int(
                self.outer_evaluation_trajectory_count,
                "outer_evaluation_trajectory_count",
                minimum=1,
            ),
        )
        training_count = require_int(self.training_trajectory_count, "training_trajectory_count")
        if training_count not in {0, 1}:
            msg = "Smoke training_trajectory_count must be exactly zero or one."
            raise ValueError(msg)
        object.__setattr__(self, "training_trajectory_count", training_count)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered effective smoke limit."""
        return {
            "schema_version": self.schema_version,
            "training_update_count": self.training_update_count,
            "training_trajectory_count": self.training_trajectory_count,
            "checkpoint_validation_trajectory_count": self.checkpoint_validation_trajectory_count,
            "maximum_growth_steps": self.maximum_growth_steps,
            "reoptimization_steps_per_growth": self.reoptimization_steps_per_growth,
            "outer_evaluation_trajectory_count": self.outer_evaluation_trajectory_count,
            "runtime_adapter_status": self.runtime_adapter_status,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the effective smoke limits."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> SmokeExecutionLimits:
        """Decode and verify exact effective smoke limits.

        Returns:
            The verified effective smoke limits.

        Raises:
            ValueError: If normalized limits differ from the sealed boundary.
        """
        mapping = _verify(
            value,
            keys=_SMOKE_EXECUTION_LIMITS_KEYS,
            version=SMOKE_EXECUTION_LIMITS_SCHEMA_VERSION,
            name="smoke execution limits",
        )
        limits = cls(
            outer_evaluation_trajectory_count=cast("int", mapping["outer_evaluation_trajectory_count"]),
            training_trajectory_count=cast("int", mapping["training_trajectory_count"]),
        )
        if mapping != freeze_json_mapping(limits.to_dict(), "expected smoke execution limits"):
            msg = "Effective smoke execution limits differ from the exact WP22A preflight boundary."
            raise ValueError(msg)
        return limits


@dataclass(frozen=True, slots=True)
class PipelineSmokeSpec:
    """Typed pipeline reference for structural smoke preflight only.

    The referenced repository template identifies the implementation family,
    but WP22A does not authorize running all of its stages.  The exact sentinel
    and effective limits describe the future WP22B adapter boundary.
    """

    structural_template_reference: TrainingPipelineTemplate
    outer_evaluation_policy: FreshEvaluationPolicy
    effective_limits: SmokeExecutionLimits
    target_scope_id: TargetScope = field(default="primary_q6", init=False)
    qubit_count: int = field(default=6, init=False)
    runtime_stage_sentinel: str = field(default="pipeline_smoke_adapter", init=False)
    promotion_eligible: bool = field(default=False, init=False)
    schema_version: str = field(default=PIPELINE_SMOKE_SPEC_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the typed q6 reference and truthful effective boundary.

        Raises:
            TypeError: If the template, policy, or effective limits use the wrong record type.
            ValueError: If the reference or effective boundary differs from the frozen smoke policy.
        """
        template = self.structural_template_reference
        if not isinstance(template, TrainingPipelineTemplate):
            msg = "structural_template_reference must be a TrainingPipelineTemplate."
            raise TypeError(msg)
        if template.target_scope_id != "primary_q6":
            msg = "Pipeline structural smoke is restricted to primary_q6."
            raise ValueError(msg)
        _validate_pipeline_width(template, "primary_q6")
        _validate_pipeline_method_family(template)
        if template != _canonical_smoke_pipeline(template.method_id):
            msg = "Pipeline smoke reference differs from the exact repository-rooted tiny template."
            raise ValueError(msg)
        if (
            not isinstance(self.outer_evaluation_policy, FreshEvaluationPolicy)
            or self.outer_evaluation_policy.purpose != "smoke_evaluation"
        ):
            msg = "Pipeline structural smoke requires a role-specific smoke evaluation policy."
            raise ValueError(msg)
        if not isinstance(self.effective_limits, SmokeExecutionLimits):
            msg = "effective_limits must be SmokeExecutionLimits."
            raise TypeError(msg)
        expected_training_count = 0 if template.method_id == "layerwise_bmpd_noiseless" else 1
        if self.effective_limits.training_trajectory_count != expected_training_count:
            msg = "Pipeline smoke training trajectories must be zero exactly for the noiseless comparator."
            raise ValueError(msg)
        if self.effective_limits.outer_evaluation_trajectory_count != self.outer_evaluation_policy.trajectory_count:
            msg = "Pipeline smoke outer evaluation and effective limits disagree."
            raise ValueError(msg)

    @property
    def method_id(self) -> str:
        """Repository implementation method identified by the structural template."""
        return self.structural_template_reference.method_id

    @classmethod
    def frozen(
        cls,
        template: TrainingPipelineTemplate,
        trajectory_count: int,
    ) -> PipelineSmokeSpec:
        """Build a typed structural reference under exact tiny effective limits.

        Returns:
            The canonical structural pipeline smoke wrapper.
        """
        return cls(
            structural_template_reference=template,
            outer_evaluation_policy=FreshEvaluationPolicy.smoke(trajectory_count),
            effective_limits=SmokeExecutionLimits(
                trajectory_count,
                training_trajectory_count=(0 if template.method_id == "layerwise_bmpd_noiseless" else 1),
            ),
        )

    def _payload(self) -> dict[str, object]:
        """Return the checksum-covered structural reference and adapter boundary."""
        return {
            "schema_version": self.schema_version,
            "target_scope_id": self.target_scope_id,
            "qubit_count": self.qubit_count,
            "method_id": self.method_id,
            "structural_template_reference": self.structural_template_reference.to_dict(),
            "outer_evaluation_policy": self.outer_evaluation_policy.to_dict(),
            "effective_limits": self.effective_limits.to_dict(),
            "runtime_stage_sentinel": self.runtime_stage_sentinel,
            "promotion_eligible": self.promotion_eligible,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the structural pipeline smoke wrapper."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> PipelineSmokeSpec:
        """Decode and verify a structural pipeline smoke wrapper.

        Returns:
            The verified structural pipeline smoke wrapper.

        Raises:
            ValueError: If fixed identity fields or the normalized checksum differ.
        """
        mapping = _verify(
            value,
            keys=_PIPELINE_SMOKE_SPEC_KEYS,
            version=PIPELINE_SMOKE_SPEC_SCHEMA_VERSION,
            name="pipeline smoke implementation",
        )
        fixed = {
            "target_scope_id": "primary_q6",
            "qubit_count": 6,
            "runtime_stage_sentinel": "pipeline_smoke_adapter",
            "promotion_eligible": False,
        }
        if any(mapping[name] != expected for name, expected in fixed.items()):
            msg = "Pipeline smoke scope, adapter sentinel, or promotion status changed."
            raise ValueError(msg)
        spec = cls(
            structural_template_reference=TrainingPipelineTemplate.from_dict(mapping["structural_template_reference"]),
            outer_evaluation_policy=FreshEvaluationPolicy.from_dict(mapping["outer_evaluation_policy"]),
            effective_limits=SmokeExecutionLimits.from_dict(mapping["effective_limits"]),
        )
        if mapping["method_id"] != spec.method_id or mapping["content_checksum"] != spec.content_checksum:
            msg = "Pipeline smoke method or checksum changed during normalization."
            raise ValueError(msg)
        return spec


@dataclass(frozen=True, slots=True)
class OperatorGrowthSmokeSpec:
    """Production projector-growth core with truthful tiny effective limits."""

    production_pool: OperatorPoolSpec
    production_growth_spec: OperatorGrowthSpec
    outer_evaluation_policy: FreshEvaluationPolicy
    effective_limits: SmokeExecutionLimits
    target_scope_id: TargetScope = field(default="primary_q6", init=False)
    qubit_count: int = field(default=6, init=False)
    method_id: str = field(default="adapt_style_state_preparation", init=False)
    promotion_eligible: bool = field(default=False, init=False)
    schema_version: str = field(default=OPERATOR_GROWTH_SMOKE_SPEC_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Preserve the production core while sealing effective smoke work.

        Raises:
            TypeError: If effective limits do not use the required typed record.
            ValueError: If the production core, evaluation policy, or tiny limits differ.
        """
        production = OperatorGrowthExecutionSpec.for_screening(256)
        if self.production_pool != production.pool or self.production_growth_spec != production.growth_spec:
            msg = "Operator-growth smoke must reference the exact production screening pool and growth core."
            raise ValueError(msg)
        if (
            not isinstance(self.outer_evaluation_policy, FreshEvaluationPolicy)
            or self.outer_evaluation_policy.purpose != "smoke_evaluation"
        ):
            msg = "Operator-growth smoke requires its role-specific fresh smoke evaluation."
            raise ValueError(msg)
        if not isinstance(self.effective_limits, SmokeExecutionLimits):
            msg = "effective_limits must be SmokeExecutionLimits."
            raise TypeError(msg)
        if self.effective_limits.training_trajectory_count != 1:
            msg = "Operator-growth smoke requires exactly one matched-noise training trajectory."
            raise ValueError(msg)
        if self.effective_limits.outer_evaluation_trajectory_count != self.outer_evaluation_policy.trajectory_count:
            msg = "Operator-growth smoke outer evaluation and effective limits disagree."
            raise ValueError(msg)

    @classmethod
    def frozen(cls, trajectory_count: int) -> OperatorGrowthSmokeSpec:
        """Build the exact production reference under tiny effective limits.

        Returns:
            The canonical operator-growth smoke wrapper.
        """
        production = OperatorGrowthExecutionSpec.for_screening(256)
        return cls(
            production_pool=production.pool,
            production_growth_spec=production.growth_spec,
            outer_evaluation_policy=FreshEvaluationPolicy.smoke(trajectory_count),
            effective_limits=SmokeExecutionLimits(trajectory_count),
        )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered production reference and effective limit."""
        return {
            "schema_version": self.schema_version,
            "target_scope_id": self.target_scope_id,
            "qubit_count": self.qubit_count,
            "method_id": self.method_id,
            "production_pool": self.production_pool.to_dict(),
            "production_growth_spec": self.production_growth_spec.to_dict(),
            "outer_evaluation_policy": self.outer_evaluation_policy.to_dict(),
            "effective_limits": self.effective_limits.to_dict(),
            "promotion_eligible": self.promotion_eligible,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete operator-growth smoke wrapper."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthSmokeSpec:
        """Decode and verify the exact operator-growth smoke wrapper.

        Returns:
            The verified operator-growth smoke wrapper.

        Raises:
            ValueError: If identity fields or the normalized checksum differ.
        """
        mapping = _verify(
            value,
            keys=_OPERATOR_GROWTH_SMOKE_SPEC_KEYS,
            version=OPERATOR_GROWTH_SMOKE_SPEC_SCHEMA_VERSION,
            name="operator-growth smoke implementation",
        )
        fixed = {
            "target_scope_id": "primary_q6",
            "qubit_count": 6,
            "method_id": "adapt_style_state_preparation",
            "promotion_eligible": False,
        }
        if any(mapping[name] != expected for name, expected in fixed.items()):
            msg = "Operator-growth smoke identity or promotion status changed."
            raise ValueError(msg)
        spec = cls(
            production_pool=OperatorPoolSpec.from_dict(mapping["production_pool"]),
            production_growth_spec=OperatorGrowthSpec.from_dict(mapping["production_growth_spec"]),
            outer_evaluation_policy=FreshEvaluationPolicy.from_dict(mapping["outer_evaluation_policy"]),
            effective_limits=SmokeExecutionLimits.from_dict(mapping["effective_limits"]),
        )
        if mapping["content_checksum"] != spec.content_checksum:
            msg = "Operator-growth smoke checksum changed during normalization."
            raise ValueError(msg)
        return spec


@dataclass(frozen=True, slots=True)
class EnergyAdaptSmokeSpec:
    """Target-independent q6 TFIM energy-ADAPT smoke implementation."""

    pool: OperatorPoolSpec
    growth_spec: OperatorGrowthSpec
    outer_evaluation_policy: FreshEvaluationPolicy
    effective_limits: SmokeExecutionLimits
    target_scope_id: TargetScope = field(default="primary_q6", init=False)
    qubit_count: int = field(default=6, init=False)
    method_id: str = field(default=ENERGY_ADAPT_METHOD_ID, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    schema_version: str = field(default=ENERGY_ADAPT_SMOKE_SPEC_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require the exact target-independent pool, growth, and smoke wrapper.

        Raises:
            TypeError: If pool, growth, or limits do not use their required typed records.
            ValueError: If the implementation or effective smoke boundary differs.
        """
        if not isinstance(self.pool, OperatorPoolSpec) or not isinstance(self.growth_spec, OperatorGrowthSpec):
            msg = "Energy ADAPT smoke requires typed operator-pool and growth artifacts."
            raise TypeError(msg)
        expected_pool = build_tfim_real_operator_pool(6)
        expected_growth = OperatorGrowthSpec.for_pool(
            expected_pool,
            gradient_tolerance=1e-10,
            max_operators=min(16, len(expected_pool.operators)),
            native_two_qubit_cap_per_edge=12,
            reoptimization_steps=100,
            learning_rate=0.08,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
        )
        if self.pool != expected_pool or self.growth_spec != expected_growth:
            msg = "Energy ADAPT smoke pool or growth settings differ from the frozen q6 implementation."
            raise ValueError(msg)
        if (
            not isinstance(self.outer_evaluation_policy, FreshEvaluationPolicy)
            or self.outer_evaluation_policy.purpose != "smoke_evaluation"
        ):
            msg = "Energy ADAPT smoke requires a role-specific q6 smoke evaluation policy."
            raise ValueError(msg)
        if not isinstance(self.effective_limits, SmokeExecutionLimits):
            msg = "effective_limits must be SmokeExecutionLimits."
            raise TypeError(msg)
        if self.effective_limits.training_trajectory_count != 1:
            msg = "Energy ADAPT smoke requires exactly one matched-noise training trajectory."
            raise ValueError(msg)
        if self.effective_limits.outer_evaluation_trajectory_count != self.outer_evaluation_policy.trajectory_count:
            msg = "Energy ADAPT smoke outer evaluation and effective limits disagree."
            raise ValueError(msg)

    @classmethod
    def frozen(cls, trajectory_count: int) -> EnergyAdaptSmokeSpec:
        """Build the exact target-independent TFIM energy-ADAPT smoke wrapper.

        Returns:
            The canonical TFIM energy-ADAPT smoke wrapper.
        """
        pool = build_tfim_real_operator_pool(6)
        return cls(
            pool=pool,
            growth_spec=OperatorGrowthSpec.for_pool(
                pool,
                gradient_tolerance=1e-10,
                max_operators=min(16, len(pool.operators)),
                native_two_qubit_cap_per_edge=12,
                reoptimization_steps=100,
                learning_rate=0.08,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8,
            ),
            outer_evaluation_policy=FreshEvaluationPolicy.smoke(trajectory_count),
            effective_limits=SmokeExecutionLimits(trajectory_count),
        )

    def _payload(self) -> dict[str, object]:
        """Return every target-independent smoke implementation choice."""
        return {
            "schema_version": self.schema_version,
            "target_scope_id": self.target_scope_id,
            "qubit_count": self.qubit_count,
            "method_id": self.method_id,
            "pool": self.pool.to_dict(),
            "growth_spec": self.growth_spec.to_dict(),
            "outer_evaluation_policy": self.outer_evaluation_policy.to_dict(),
            "effective_limits": self.effective_limits.to_dict(),
            "promotion_eligible": self.promotion_eligible,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete target-independent smoke implementation."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> EnergyAdaptSmokeSpec:
        """Decode and verify the exact TFIM energy-ADAPT smoke implementation.

        Returns:
            The verified TFIM energy-ADAPT smoke wrapper.

        Raises:
            ValueError: If identity fields or the normalized checksum differ.
        """
        mapping = _verify(
            value,
            keys=_ENERGY_ADAPT_SMOKE_SPEC_KEYS,
            version=ENERGY_ADAPT_SMOKE_SPEC_SCHEMA_VERSION,
            name="energy ADAPT smoke implementation",
        )
        fixed = {
            "target_scope_id": "primary_q6",
            "qubit_count": 6,
            "method_id": ENERGY_ADAPT_METHOD_ID,
            "promotion_eligible": False,
        }
        if any(mapping[name] != expected for name, expected in fixed.items()):
            msg = "Energy ADAPT smoke aliases or promotion status changed."
            raise ValueError(msg)
        spec = cls(
            pool=OperatorPoolSpec.from_dict(mapping["pool"]),
            growth_spec=OperatorGrowthSpec.from_dict(mapping["growth_spec"]),
            outer_evaluation_policy=FreshEvaluationPolicy.from_dict(mapping["outer_evaluation_policy"]),
            effective_limits=SmokeExecutionLimits.from_dict(mapping["effective_limits"]),
        )
        if mapping["content_checksum"] != spec.content_checksum:
            msg = "Energy ADAPT smoke checksum changed during normalization."
            raise ValueError(msg)
        return spec


ImplementationPayload = (
    TrainingPipelineTemplate
    | PipelineSmokeSpec
    | OperatorGrowthExecutionSpec
    | OperatorGrowthSmokeSpec
    | EnergyAdaptSmokeSpec
)


@dataclass(frozen=True, slots=True)
class ExecutionImplementationArtifact:
    """Typed preset- and width-specific implementation reference.

    Production variants describe executable repository pipelines. Smoke
    variants describe structural preflight only and require the later WP22B
    runtime adapter before execution.
    """

    artifact_id: str
    preset: Preset
    publication_method_id: str
    implementation_kind: ImplementationKind
    implementation_method_id: str
    target_scope_id: TargetScope
    strategy_schedule_checksum: str
    implementation_payload: ImplementationPayload
    schema_version: str = field(default=EXECUTION_IMPLEMENTATION_ARTIFACT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate one supported, strict, target-independent payload type.

        Raises:
            TypeError: If the implementation payload does not match its discriminator.
            ValueError: If preset, identity, scope, schedule, or canonical policy differs.
        """
        object.__setattr__(self, "artifact_id", require_slug(self.artifact_id, "artifact_id"))
        preset = _require_preset(self.preset)
        if preset in _WP22A_DEFERRED_PRESETS:
            msg = f"{preset} implementation artifacts are deferred beyond WP22A."
            raise ValueError(msg)
        object.__setattr__(self, "preset", preset)
        object.__setattr__(
            self,
            "publication_method_id",
            require_slug(self.publication_method_id, "publication_method_id"),
        )
        if self.implementation_kind not in _IMPLEMENTATION_KINDS:
            msg = (
                "implementation_kind must be phase2_pipeline, operator_growth, "
                "phase2_pipeline_smoke, operator_growth_smoke, or tfim_operator_growth."
            )
            raise ValueError(msg)
        object.__setattr__(
            self,
            "implementation_method_id",
            require_slug(self.implementation_method_id, "implementation_method_id"),
        )
        expected_implementation_method = _IMPLEMENTATION_METHOD_BY_PUBLICATION_METHOD.get(self.publication_method_id)
        if expected_implementation_method is None or self.implementation_method_id != expected_implementation_method:
            msg = (
                "publication_method_id must map to the same implementation method, except "
                "impact_pruning_crn which maps exactly to topdown_impact_iterative."
            )
            raise ValueError(msg)
        object.__setattr__(self, "target_scope_id", _require_target_scope(self.target_scope_id))
        object.__setattr__(
            self,
            "strategy_schedule_checksum",
            require_checksum(self.strategy_schedule_checksum, "strategy_schedule_checksum"),
        )
        payload = self.implementation_payload
        if self.implementation_kind == "phase2_pipeline":
            if preset not in {"paper-pilot", "paper-screen"}:
                msg = "Production phase2_pipeline artifacts are restricted to paper pilot and screen."
                raise ValueError(msg)
            expected_methods = (
                frozenset(PILOT_METHOD_IDS) if preset == "paper-pilot" else _PIPELINE_PUBLICATION_METHOD_IDS
            )
            if self.publication_method_id not in expected_methods or (
                preset == "paper-screen" and self.target_scope_id != "primary_q6"
            ):
                msg = "Production pipeline method is not part of the selected paper preset."
                raise ValueError(msg)
            if not isinstance(payload, TrainingPipelineTemplate):
                msg = "phase2_pipeline implementations require a TrainingPipelineTemplate payload."
                raise TypeError(msg)
            if payload.method_id != self.implementation_method_id or payload.target_scope_id != self.target_scope_id:
                msg = "Pipeline payload method and target scope must match its implementation envelope."
                raise ValueError(msg)
            _validate_pipeline_width(payload, self.target_scope_id)
            _validate_pipeline_method_family(payload)
            expected_schedule = next(
                schedule
                for schedule in FrozenTrainingPolicyUniverse.frozen().schedules
                if schedule.schedule_id == _PAPER_SCHEDULE_ID_BY_METHOD[self.publication_method_id]
            )
            if self.strategy_schedule_checksum != expected_schedule.content_checksum:
                msg = "Production pipeline artifact does not bind its exact method-specific schedule."
                raise ValueError(msg)
            _validate_production_terminal_policy(payload, self.publication_method_id, expected_schedule)
            if payload != _canonical_production_pipeline(self.publication_method_id, self.target_scope_id):
                msg = "Production pipeline payload differs from the exact repository-rooted template."
                raise ValueError(msg)
        elif self.implementation_kind == "phase2_pipeline_smoke":
            if (
                self.publication_method_id not in _PIPELINE_PUBLICATION_METHOD_IDS
                or not isinstance(payload, PipelineSmokeSpec)
                or payload.method_id != self.implementation_method_id
                or self.target_scope_id != "primary_q6"
                or self.preset != "training-smoke"
            ):
                msg = "phase2_pipeline_smoke requires a truthful q6 structural pipeline wrapper."
                raise TypeError(msg)
        elif self.implementation_kind == "operator_growth":
            if (
                preset != "paper-screen"
                or not isinstance(payload, OperatorGrowthExecutionSpec)
                or self.implementation_method_id != "adapt_style_state_preparation"
                or self.target_scope_id != "primary_q6"
            ):
                msg = "operator_growth requires the complete primary-q6 projector execution spec."
                raise TypeError(msg)
        elif self.implementation_kind == "operator_growth_smoke":
            if (
                not isinstance(payload, OperatorGrowthSmokeSpec)
                or self.implementation_method_id != "adapt_style_state_preparation"
                or self.target_scope_id != "primary_q6"
                or self.preset != "training-smoke"
            ):
                msg = "operator_growth_smoke requires the truthful target-independent q6 smoke wrapper."
                raise TypeError(msg)
        elif (
            not isinstance(payload, EnergyAdaptSmokeSpec)
            or self.implementation_method_id != ENERGY_ADAPT_METHOD_ID
            or self.target_scope_id != "primary_q6"
            or self.preset != "training-smoke"
        ):
            msg = "tfim_operator_growth requires the target-independent q6 energy-ADAPT smoke spec."
            raise TypeError(msg)

    @property
    def implementation_payload_checksum(self) -> str:
        """Checksum of the complete typed implementation payload."""
        payload = self.implementation_payload
        if isinstance(payload, TrainingPipelineTemplate):
            return canonical_checksum(payload.to_dict())
        return payload.content_checksum

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered artifact field."""
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "preset": self.preset,
            "publication_method_id": self.publication_method_id,
            "implementation_kind": self.implementation_kind,
            "implementation_method_id": self.implementation_method_id,
            "target_scope_id": self.target_scope_id,
            "strategy_schedule_checksum": self.strategy_schedule_checksum,
            "implementation_payload": self.implementation_payload.to_dict(),
            "implementation_payload_checksum": self.implementation_payload_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Preset- and width-specific implementation checksum."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ExecutionImplementationArtifact:
        """Decode and verify one implementation envelope.

        Returns:
            The verified implementation artifact.

        Raises:
            ValueError: If the discriminator or normalized checksums differ.
        """
        mapping = _verify(
            value,
            keys=_IMPLEMENTATION_ARTIFACT_KEYS,
            version=EXECUTION_IMPLEMENTATION_ARTIFACT_SCHEMA_VERSION,
            name="execution implementation artifact",
        )
        kind = cast("ImplementationKind", mapping["implementation_kind"])
        raw_payload = mapping["implementation_payload"]
        if kind == "phase2_pipeline":
            implementation_payload: ImplementationPayload = TrainingPipelineTemplate.from_dict(raw_payload)
        elif kind == "phase2_pipeline_smoke":
            implementation_payload = PipelineSmokeSpec.from_dict(raw_payload)
        elif kind == "operator_growth":
            implementation_payload = OperatorGrowthExecutionSpec.from_dict(raw_payload)
        elif kind == "operator_growth_smoke":
            implementation_payload = OperatorGrowthSmokeSpec.from_dict(raw_payload)
        elif kind == "tfim_operator_growth":
            implementation_payload = EnergyAdaptSmokeSpec.from_dict(raw_payload)
        else:
            msg = "implementation_kind is not a supported typed payload discriminator."
            raise ValueError(msg)
        artifact = cls(
            artifact_id=cast("str", mapping["artifact_id"]),
            preset=cast("Preset", mapping["preset"]),
            publication_method_id=cast("str", mapping["publication_method_id"]),
            implementation_kind=kind,
            implementation_method_id=cast("str", mapping["implementation_method_id"]),
            target_scope_id=cast("TargetScope", mapping["target_scope_id"]),
            strategy_schedule_checksum=cast("str", mapping["strategy_schedule_checksum"]),
            implementation_payload=implementation_payload,
        )
        if mapping["implementation_payload_checksum"] != artifact.implementation_payload_checksum:
            msg = "Serialized implementation payload checksum is inconsistent."
            raise ValueError(msg)
        if mapping["content_checksum"] != artifact.content_checksum:
            msg = "Implementation artifact checksum changed during normalization."
            raise ValueError(msg)
        return artifact

    @classmethod
    def from_json(cls, payload: str) -> ExecutionImplementationArtifact:
        """Decode canonical JSON into a verified implementation envelope.

        Returns:
            The verified implementation artifact.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class ControlledTrainingStage:
    """Update-aware stage boundary that preserves optimizer and resume state."""

    stage_id: str
    implementation_stage_id: str
    strategy_schedule_checksum: str
    start_update: int
    stop_update_exclusive: int
    schema_version: str = field(default=CONTROLLED_TRAINING_STAGE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require a nonempty stage using the frozen update-aware adapter.

        Raises:
            ValueError: If stage identity or update boundaries are inconsistent.
        """
        stage_id = require_slug(self.stage_id, "stage_id")
        implementation_stage_id = require_slug(self.implementation_stage_id, "implementation_stage_id")
        if stage_id != f"controlled_{implementation_stage_id}":
            msg = "stage_id must be derived exactly from implementation_stage_id."
            raise ValueError(msg)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "implementation_stage_id", implementation_stage_id)
        object.__setattr__(
            self,
            "strategy_schedule_checksum",
            require_checksum(self.strategy_schedule_checksum, "strategy_schedule_checksum"),
        )
        start = require_int(self.start_update, "start_update")
        stop = require_int(self.stop_update_exclusive, "stop_update_exclusive", minimum=1)
        if stop <= start:
            msg = "stop_update_exclusive must be greater than start_update."
            raise ValueError(msg)
        object.__setattr__(self, "start_update", start)
        object.__setattr__(self, "stop_update_exclusive", stop)

    @classmethod
    def complete_schedule(
        cls,
        schedule: TrainingStrategySchedule,
        implementation_artifact: ExecutionImplementationArtifact,
    ) -> ControlledTrainingStage:
        """Bind a full schedule to one exact implementation stage or adapter.

        Returns:
            The update-aware full-schedule implementation-stage binding.

        Raises:
            TypeError: If the schedule or implementation artifact has the wrong type.
        """
        if not isinstance(schedule, TrainingStrategySchedule):
            msg = "schedule must be a TrainingStrategySchedule."
            raise TypeError(msg)
        if not isinstance(implementation_artifact, ExecutionImplementationArtifact):
            msg = "implementation_artifact must be an ExecutionImplementationArtifact."
            raise TypeError(msg)
        payload = implementation_artifact.implementation_payload
        if isinstance(payload, TrainingPipelineTemplate):
            implementation_stage_id = payload.stages[-1].stage_id
        elif isinstance(payload, PipelineSmokeSpec):
            implementation_stage_id = payload.runtime_stage_sentinel
        elif isinstance(payload, (OperatorGrowthExecutionSpec, OperatorGrowthSmokeSpec)):
            implementation_stage_id = "operator_growth_reoptimization"
        else:
            implementation_stage_id = "energy_adapt_reoptimization"
        return cls(
            stage_id=f"controlled_{implementation_stage_id}",
            implementation_stage_id=implementation_stage_id,
            strategy_schedule_checksum=schedule.content_checksum,
            start_update=0,
            stop_update_exclusive=schedule.phase_boundary.total_updates,
        )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered stage field."""
        return {
            "schema_version": self.schema_version,
            "stage_id": self.stage_id,
            "implementation_stage_id": self.implementation_stage_id,
            "strategy_schedule_checksum": self.strategy_schedule_checksum,
            "start_update": self.start_update,
            "stop_update_exclusive": self.stop_update_exclusive,
            "schedule_application": "update_aware_adapter",
            "optimizer_state_rule": "preserve_across_schedule_boundaries",
            "resume_rule": "byte_identical_membership_and_optimizer_state",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the controlled stage."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ControlledTrainingStage:
        """Decode and verify one controlled stage.

        Returns:
            The verified controlled training stage.

        Raises:
            ValueError: If frozen execution or resume semantics differ.
        """
        mapping = _verify(
            value,
            keys=_CONTROLLED_STAGE_KEYS,
            version=CONTROLLED_TRAINING_STAGE_SCHEMA_VERSION,
            name="controlled training stage",
        )
        expected = {
            "schedule_application": "update_aware_adapter",
            "optimizer_state_rule": "preserve_across_schedule_boundaries",
            "resume_rule": "byte_identical_membership_and_optimizer_state",
        }
        if any(mapping[key] != expected_value for key, expected_value in expected.items()):
            msg = "Controlled stage execution or resume semantics changed."
            raise ValueError(msg)
        return cls(
            stage_id=cast("str", mapping["stage_id"]),
            implementation_stage_id=cast("str", mapping["implementation_stage_id"]),
            strategy_schedule_checksum=cast("str", mapping["strategy_schedule_checksum"]),
            start_update=cast("int", mapping["start_update"]),
            stop_update_exclusive=cast("int", mapping["stop_update_exclusive"]),
        )


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    """Finite update, trajectory, multistart, and normalized-work limits."""

    total_update_count: int
    maximum_training_trajectory_count: int
    checkpoint_validation_trajectory_count: int
    multistart_count: int
    normalized_compute_cap: float | None
    schema_version: str = field(default=EXECUTION_BUDGET_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate every finite execution limit."""
        object.__setattr__(
            self,
            "total_update_count",
            require_int(self.total_update_count, "total_update_count", minimum=1),
        )
        object.__setattr__(
            self,
            "maximum_training_trajectory_count",
            require_int(self.maximum_training_trajectory_count, "maximum_training_trajectory_count"),
        )
        object.__setattr__(
            self,
            "checkpoint_validation_trajectory_count",
            require_int(self.checkpoint_validation_trajectory_count, "checkpoint_validation_trajectory_count"),
        )
        object.__setattr__(
            self,
            "multistart_count",
            require_int(self.multistart_count, "multistart_count", minimum=1),
        )
        if self.normalized_compute_cap is not None:
            object.__setattr__(
                self,
                "normalized_compute_cap",
                require_float(self.normalized_compute_cap, "normalized_compute_cap", minimum=0.0),
            )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered budget field."""
        return {
            "schema_version": self.schema_version,
            "total_update_count": self.total_update_count,
            "maximum_training_trajectory_count": self.maximum_training_trajectory_count,
            "checkpoint_validation_trajectory_count": self.checkpoint_validation_trajectory_count,
            "multistart_count": self.multistart_count,
            "normalized_compute_cap": self.normalized_compute_cap,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering all execution budgets."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ExecutionBudget:
        """Decode and verify finite execution budgets.

        Returns:
            The verified finite execution budget.
        """
        mapping = _verify(
            value,
            keys=_EXECUTION_BUDGET_KEYS,
            version=EXECUTION_BUDGET_SCHEMA_VERSION,
            name="execution budget",
        )
        return cls(
            total_update_count=cast("int", mapping["total_update_count"]),
            maximum_training_trajectory_count=cast("int", mapping["maximum_training_trajectory_count"]),
            checkpoint_validation_trajectory_count=cast(
                "int",
                mapping["checkpoint_validation_trajectory_count"],
            ),
            multistart_count=cast("int", mapping["multistart_count"]),
            normalized_compute_cap=cast("float | None", mapping["normalized_compute_cap"]),
        )


@dataclass(frozen=True, slots=True)
class BindingResourcePolicy:
    """Primary native-two-qubit resource policy carried by every binding."""

    normalized_compute_cap_source: str = "pilot_final_seal"
    schema_version: str = field(default=RESOURCE_POLICY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require the preregistered source of the normalized compute cap.

        Raises:
            ValueError: If the normalized compute cap source differs.
        """
        source = require_slug(self.normalized_compute_cap_source, "normalized_compute_cap_source")
        if source != "pilot_final_seal":
            msg = "normalized_compute_cap_source must remain pilot_final_seal."
            raise ValueError(msg)

    def _payload(self) -> dict[str, object]:
        """Return the exact preregistered resource fields."""
        return {
            "schema_version": self.schema_version,
            "metric": "native_two_qubit_gates_per_chain_edge",
            "cap_per_chain_edge": 12.0,
            "comparison_rule": "largest_reachable_at_or_below_cap",
            "compiler_policy_id": "quantinuum_rzz_chain_v1",
            "connectivity": "linear_chain",
            "routing_policy": "identity_no_swap",
            "normalized_compute_cap_source": self.normalized_compute_cap_source,
            "residual_gap_reporting": True,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the exact resource policy."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> BindingResourcePolicy:
        """Decode and verify the exact primary resource policy.

        Returns:
            The verified binding resource policy.

        Raises:
            ValueError: If a fixed resource-policy field differs.
        """
        mapping = _verify(
            value,
            keys=_RESOURCE_POLICY_KEYS,
            version=RESOURCE_POLICY_SCHEMA_VERSION,
            name="binding resource policy",
        )
        expected = cls().to_dict()
        if mapping != freeze_json_mapping(expected, "expected binding resource policy"):
            msg = "Binding resource policy differs from the frozen preregistration."
            raise ValueError(msg)
        return cls()


@dataclass(frozen=True, slots=True)
class QubitTreatmentProjection:
    """Primary-q6 or descriptive-q12 projection of one publication treatment."""

    publication_candidate_checksum: str
    publication_method_id: str
    target_scope_id: TargetScope
    primary_q6_implementation_checksum: str
    inference_role: InferenceRole
    screening_eligible: bool
    promotion_eligible: bool
    schema_version: str = field(default=QUBIT_TREATMENT_PROJECTION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Keep q12 descriptive and categorically outside promotion.

        Raises:
            ValueError: If role or eligibility contradicts the target scope.
        """
        object.__setattr__(
            self,
            "publication_candidate_checksum",
            require_checksum(self.publication_candidate_checksum, "publication_candidate_checksum"),
        )
        object.__setattr__(
            self,
            "publication_method_id",
            require_slug(self.publication_method_id, "publication_method_id"),
        )
        scope = _require_target_scope(self.target_scope_id)
        object.__setattr__(self, "target_scope_id", scope)
        object.__setattr__(
            self,
            "primary_q6_implementation_checksum",
            require_checksum(
                self.primary_q6_implementation_checksum,
                "primary_q6_implementation_checksum",
            ),
        )
        if self.inference_role not in {"primary", "secondary_descriptive_only"}:
            msg = "inference_role must be primary or secondary_descriptive_only."
            raise ValueError(msg)
        screening = require_bool(self.screening_eligible, "screening_eligible")
        promotion = require_bool(self.promotion_eligible, "promotion_eligible")
        if scope == "secondary_q12" and (self.inference_role != "secondary_descriptive_only" or screening or promotion):
            msg = "secondary_q12 is descriptive only and cannot enter screening or promotion."
            raise ValueError(msg)
        if scope == "primary_q6" and self.inference_role != "primary":
            msg = "primary_q6 must retain the primary treatment projection."
            raise ValueError(msg)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered projection field."""
        return {
            "schema_version": self.schema_version,
            "publication_candidate_checksum": self.publication_candidate_checksum,
            "publication_method_id": self.publication_method_id,
            "target_scope_id": self.target_scope_id,
            "primary_q6_implementation_checksum": self.primary_q6_implementation_checksum,
            "inference_role": self.inference_role,
            "screening_eligible": self.screening_eligible,
            "promotion_eligible": self.promotion_eligible,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the treatment projection."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> QubitTreatmentProjection:
        """Decode and verify one treatment projection.

        Returns:
            The verified treatment projection.
        """
        mapping = _verify(
            value,
            keys=_TREATMENT_PROJECTION_KEYS,
            version=QUBIT_TREATMENT_PROJECTION_SCHEMA_VERSION,
            name="qubit treatment projection",
        )
        return cls(
            publication_candidate_checksum=cast("str", mapping["publication_candidate_checksum"]),
            publication_method_id=cast("str", mapping["publication_method_id"]),
            target_scope_id=cast("TargetScope", mapping["target_scope_id"]),
            primary_q6_implementation_checksum=cast(
                "str",
                mapping["primary_q6_implementation_checksum"],
            ),
            inference_role=cast("InferenceRole", mapping["inference_role"]),
            screening_eligible=cast("bool", mapping["screening_eligible"]),
            promotion_eligible=cast("bool", mapping["promotion_eligible"]),
        )


@dataclass(frozen=True, slots=True)
class ScopedImplementationBinding:
    """One publication candidate bound to a typed preset/width treatment.

    Production treatments bind executable repository pipelines. Smoke
    treatments remain structural preflight records until WP22B supplies their
    runtime adapters.
    """

    binding_id: str
    preset: Preset
    publication_candidate_schema_version: str
    publication_candidate_checksum: str
    publication_method_id: str
    target_scope_id: TargetScope
    qubit_count: int
    manifest_data_role: ManifestRole
    execution_data_role: ExecutionRole
    implementation_artifact: ExecutionImplementationArtifact
    strategy_schedule: TrainingStrategySchedule
    controlled_stage: ControlledTrainingStage
    evaluation_policies: tuple[FreshEvaluationPolicy, ...]
    pilot_diagnostic_policy: PilotDiagnosticPolicy | None
    execution_budget: ExecutionBudget
    resource_policy: BindingResourcePolicy
    treatment_projection: QubitTreatmentProjection
    operator_growth_spec: OperatorGrowthExecutionSpec | None = None
    schema_version: str = field(default=SCOPED_IMPLEMENTATION_BINDING_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the complete publication-to-execution identity chain.

        Raises:
            TypeError: If a nested policy or implementation uses the wrong record type.
            ValueError: If any identity, policy, budget, role, or eligibility link differs.
        """
        object.__setattr__(self, "binding_id", require_slug(self.binding_id, "binding_id"))
        preset = _require_preset(self.preset)
        if preset in _WP22A_DEFERRED_PRESETS:
            msg = f"{preset} is intentionally unavailable until its owning post-WP22A package freezes execution."
            raise ValueError(msg)
        object.__setattr__(self, "preset", preset)
        object.__setattr__(
            self,
            "publication_candidate_schema_version",
            require_slug(self.publication_candidate_schema_version, "publication_candidate_schema_version"),
        )
        candidate_checksum = require_checksum(
            self.publication_candidate_checksum,
            "publication_candidate_checksum",
        )
        object.__setattr__(self, "publication_candidate_checksum", candidate_checksum)
        method = require_slug(self.publication_method_id, "publication_method_id")
        object.__setattr__(self, "publication_method_id", method)
        scope = _require_target_scope(self.target_scope_id)
        object.__setattr__(self, "target_scope_id", scope)
        qubits = require_int(self.qubit_count, "qubit_count", minimum=1)
        expected_qubits = 6 if scope == "primary_q6" else 12
        if qubits != expected_qubits:
            msg = f"{scope} requires qubit_count={expected_qubits}."
            raise ValueError(msg)
        object.__setattr__(self, "qubit_count", qubits)
        manifest_role = require_slug(self.manifest_data_role, "manifest_data_role")
        execution_role = require_slug(self.execution_data_role, "execution_data_role")
        if manifest_role not in _MANIFEST_ROLES or execution_role not in _EXECUTION_ROLES:
            msg = "manifest_data_role or execution_data_role is unsupported."
            raise ValueError(msg)
        object.__setattr__(self, "manifest_data_role", cast("ManifestRole", manifest_role))
        object.__setattr__(self, "execution_data_role", cast("ExecutionRole", execution_role))
        if scope == "secondary_q12" and (
            preset != "paper-pilot"
            or method not in PILOT_METHOD_IDS
            or manifest_role != "screening_selection"
            or execution_role != "secondary_benchmark"
        ):
            msg = (
                "secondary_q12 is restricted to the three-method paper pilot with "
                "screening_selection custody and secondary_benchmark execution."
            )
            raise ValueError(msg)
        expected_primary_roles = {
            "training-smoke": ("development", "development"),
            "historical-layerwise-reproduction": ("development", "development"),
            "paper-pilot": ("development", "development"),
            "paper-screen": ("screening_selection", "screening_selection"),
            "paper-confirm": ("confirmatory", "confirmatory"),
        }
        if scope == "primary_q6" and (manifest_role, execution_role) != expected_primary_roles[preset]:
            msg = "Primary-q6 manifest and execution roles differ from the selected preset."
            raise ValueError(msg)
        if not isinstance(self.implementation_artifact, ExecutionImplementationArtifact):
            msg = "implementation_artifact must be an ExecutionImplementationArtifact."
            raise TypeError(msg)
        artifact = self.implementation_artifact
        if artifact.preset != preset or artifact.publication_method_id != method or artifact.target_scope_id != scope:
            msg = "Implementation artifact preset, publication method, and target scope must match the binding."
            raise ValueError(msg)
        if artifact.content_checksum == candidate_checksum:
            msg = "Publication candidate and concrete execution implementation must remain separate identities."
            raise ValueError(msg)
        if not isinstance(self.strategy_schedule, TrainingStrategySchedule):
            msg = "strategy_schedule must be a TrainingStrategySchedule."
            raise TypeError(msg)
        if artifact.strategy_schedule_checksum != self.strategy_schedule.content_checksum:
            msg = "Implementation artifact does not bind the complete strategy schedule."
            raise ValueError(msg)
        if preset in {"paper-pilot", "paper-screen"}:
            expected_schedule_id = _PAPER_SCHEDULE_ID_BY_METHOD.get(method)
            schedule_by_id = {
                schedule.schedule_id: schedule for schedule in FrozenTrainingPolicyUniverse.frozen().schedules
            }
            if (
                expected_schedule_id is None
                or self.strategy_schedule.schedule_id != expected_schedule_id
                or self.strategy_schedule != schedule_by_id[expected_schedule_id]
            ):
                msg = "Paper bindings require the exact method-specific rooted training schedule."
                raise ValueError(msg)
            if isinstance(artifact.implementation_payload, TrainingPipelineTemplate):
                _validate_production_terminal_policy(
                    artifact.implementation_payload,
                    method,
                    self.strategy_schedule,
                )
        if not isinstance(self.controlled_stage, ControlledTrainingStage):
            msg = "controlled_stage must be a ControlledTrainingStage."
            raise TypeError(msg)
        if (
            self.controlled_stage.strategy_schedule_checksum != self.strategy_schedule.content_checksum
            or self.controlled_stage.start_update != 0
            or self.controlled_stage.stop_update_exclusive != self.strategy_schedule.phase_boundary.total_updates
        ):
            msg = "Controlled stage must span the exact complete strategy schedule."
            raise ValueError(msg)
        payload = artifact.implementation_payload
        expected_implementation_stage_id = (
            payload.stages[-1].stage_id
            if isinstance(payload, TrainingPipelineTemplate)
            else payload.runtime_stage_sentinel
            if isinstance(payload, PipelineSmokeSpec)
            else "operator_growth_reoptimization"
            if isinstance(payload, (OperatorGrowthExecutionSpec, OperatorGrowthSmokeSpec))
            else "energy_adapt_reoptimization"
        )
        if self.controlled_stage.implementation_stage_id != expected_implementation_stage_id:
            msg = "Controlled stage does not identify the exact implementation stage or adapter."
            raise ValueError(msg)
        policies = tuple(self.evaluation_policies)
        if not policies or any(not isinstance(policy, FreshEvaluationPolicy) for policy in policies):
            msg = "evaluation_policies must contain FreshEvaluationPolicy records."
            raise TypeError(msg)
        policy_payloads = tuple(policy.to_dict() for policy in policies)
        policy_ids = tuple(cast("str", payload["policy_id"]) for payload in policy_payloads)
        if len(policy_ids) != len(set(policy_ids)):
            msg = "evaluation_policies must have unique policy identities."
            raise ValueError(msg)
        if any(payload["target_scope"] != scope or payload["qubit_count"] != qubits for payload in policy_payloads):
            msg = "Every evaluation policy must match the binding target scope and width."
            raise ValueError(msg)
        if preset == "paper-pilot":
            expected_policies = (
                FreshEvaluationPolicy.checkpoint_validation(scope),
                (
                    FreshEvaluationPolicy.primary_q6_pilot()
                    if scope == "primary_q6"
                    else FreshEvaluationPolicy.secondary_q12_pilot()
                ),
            )
            if policies != expected_policies:
                msg = "paper-pilot bindings require the exact checkpoint and fresh-pilot policies."
                raise ValueError(msg)
        elif preset == "paper-screen":
            outer = tuple(policy for policy in policies if policy.purpose == "screening_outer")
            if len(outer) != 1 or policies != (
                FreshEvaluationPolicy.checkpoint_validation(),
                FreshEvaluationPolicy.screening(outer[0].trajectory_count),
            ):
                msg = "paper-screen bindings require exact checkpoint and fixed outer-screen policies."
                raise ValueError(msg)
        elif preset == "training-smoke":
            if len(policies) != 1 or policies[0].purpose != "smoke_evaluation":
                msg = "training-smoke bindings require exactly one role-specific smoke policy."
                raise ValueError(msg)
        object.__setattr__(self, "evaluation_policies", policies)
        if preset == "paper-pilot":
            if not isinstance(self.pilot_diagnostic_policy, PilotDiagnosticPolicy):
                msg = "paper-pilot bindings require a PilotDiagnosticPolicy."
                raise TypeError(msg)
            diagnostic = self.pilot_diagnostic_policy.to_dict()
            if diagnostic["target_scope"] != scope or diagnostic["qubit_count"] != qubits:
                msg = "Pilot diagnostic policy must match the binding width."
                raise ValueError(msg)
        elif self.pilot_diagnostic_policy is not None:
            msg = "Pilot diagnostics are valid only for paper-pilot bindings."
            raise ValueError(msg)
        if not isinstance(self.execution_budget, ExecutionBudget):
            msg = "execution_budget must be an ExecutionBudget."
            raise TypeError(msg)
        expected_trajectory_ceiling = max(
            step.trajectory_count for step in self.strategy_schedule.trajectory_curriculum.steps
        )
        if (
            self.execution_budget.total_update_count != self.strategy_schedule.phase_boundary.total_updates
            or self.execution_budget.maximum_training_trajectory_count != expected_trajectory_ceiling
            or self.execution_budget.multistart_count != self.strategy_schedule.multistart.start_count
        ):
            msg = "Execution budget disagrees with the exact strategy schedule."
            raise ValueError(msg)
        if preset in {"paper-pilot", "paper-screen"}:
            expected_training_count = (
                0 if method == "layerwise_bmpd_noiseless" else PRODUCTION_TRAINING_TRAJECTORY_COUNT
            )
            if (
                self.execution_budget.total_update_count != PRODUCTION_UPDATE_COUNT
                or self.execution_budget.maximum_training_trajectory_count != expected_training_count
                or self.execution_budget.checkpoint_validation_trajectory_count
                != CHECKPOINT_VALIDATION_TRAJECTORY_COUNT
                or self.execution_budget.multistart_count != 1
                or self.strategy_schedule.checkpoint_validation.patience is not None
            ):
                msg = "Paper pilot and screen bindings require the frozen 200-update production budget."
                raise ValueError(msg)
        checkpoint_counts = {
            cast("int", payload["trajectory_count"])
            for payload in policy_payloads
            if payload["data_role"] == "checkpoint_validation"
        }
        expected_checkpoint_counts = (
            set() if preset == "training-smoke" else {self.execution_budget.checkpoint_validation_trajectory_count}
        )
        if checkpoint_counts != expected_checkpoint_counts:
            msg = "Execution budget requires one exact checkpoint-validation trajectory count."
            raise ValueError(msg)
        if preset == "training-smoke" and self.execution_budget.checkpoint_validation_trajectory_count != 0:
            msg = "training-smoke uses its separate tiny-budget evaluation policy."
            raise ValueError(msg)
        if preset == "training-smoke":
            expected_smoke_training_count = 0 if method == "layerwise_bmpd_noiseless" else 1
            expected_smoke_sampling = (
                "resampled" if method in {"layerwise_bmpd_resampled", "spsa_layerwise"} else "fixed_crn"
            )
            continuation = self.strategy_schedule.noise_continuation
            noise = self.strategy_schedule.training_noise
            exact_noiseless = (
                noise.mode == "noiseless"
                and not noise.components
                and continuation.interpolation == "constant"
                and continuation.start_update == 0
                and continuation.end_update == 0
                and float(continuation.start_strength_scale).hex() == float(0).hex()
                and float(continuation.target_strength_scale).hex() == float(0).hex()
            )
            exact_primary_noise = (
                noise.mode == "matched"
                and len(noise.components) == 1
                and noise.components[0].noise_id == "depolarizing_1s_all"
                and float(noise.components[0].weight).hex() == float(1).hex()
                and continuation.interpolation == "constant"
                and continuation.start_update == 0
                and continuation.end_update == 0
                and float(continuation.start_strength_scale).hex() == float(1).hex()
                and float(continuation.target_strength_scale).hex() == float(1).hex()
            )
            if (
                self.execution_budget.total_update_count != 1
                or self.execution_budget.maximum_training_trajectory_count != expected_smoke_training_count
                or self.execution_budget.multistart_count != 1
                or self.execution_budget.normalized_compute_cap is not None
                or self.strategy_schedule.phase_boundary.total_updates != 1
                or len(self.strategy_schedule.trajectory_curriculum.steps) != 1
                or self.strategy_schedule.trajectory_curriculum.steps[0].start_update != 0
                or self.strategy_schedule.trajectory_curriculum.steps[0].trajectory_count
                != expected_smoke_training_count
                or self.strategy_schedule.sampling_policy.kind != expected_smoke_sampling
                or self.strategy_schedule.multistart != LimitedMultistartPlan(1, 1)
                or self.strategy_schedule.checkpoint_validation != CheckpointValidationPolicy(patience=None)
                or (not exact_noiseless if method == "layerwise_bmpd_noiseless" else not exact_primary_noise)
            ):
                msg = (
                    "training-smoke is a one-update structural preflight only; WP22B owns executable runtime "
                    "adapters and production evidence."
                )
                raise ValueError(msg)
            if isinstance(payload, PipelineSmokeSpec):
                limits = payload.effective_limits
                if (
                    artifact.implementation_kind != "phase2_pipeline_smoke"
                    or payload.outer_evaluation_policy not in policies
                    or limits.training_update_count != self.execution_budget.total_update_count
                    or limits.training_trajectory_count != self.execution_budget.maximum_training_trajectory_count
                    or limits.checkpoint_validation_trajectory_count
                    != self.execution_budget.checkpoint_validation_trajectory_count
                    or limits.maximum_growth_steps != 1
                    or limits.reoptimization_steps_per_growth != 1
                ):
                    msg = "Pipeline smoke effective limits disagree with its structural schedule or budget."
                    raise ValueError(msg)
        if not isinstance(self.resource_policy, BindingResourcePolicy):
            msg = "resource_policy must be a BindingResourcePolicy."
            raise TypeError(msg)
        if not isinstance(self.treatment_projection, QubitTreatmentProjection):
            msg = "treatment_projection must be a QubitTreatmentProjection."
            raise TypeError(msg)
        projection = self.treatment_projection
        if (
            projection.publication_candidate_checksum != candidate_checksum
            or projection.publication_method_id != method
            or projection.target_scope_id != scope
        ):
            msg = "Treatment projection does not reference the binding publication identity and target scope."
            raise ValueError(msg)
        if scope == "primary_q6" and projection.primary_q6_implementation_checksum != artifact.content_checksum:
            msg = "A primary-q6 projection must identify its own execution implementation."
            raise ValueError(msg)
        if preset == "paper-screen":
            expected_promotion = method != "layerwise_bmpd_noiseless"
            if (
                scope != "primary_q6"
                or not projection.screening_eligible
                or projection.promotion_eligible != expected_promotion
            ):
                msg = "paper-screen eligibility must reproduce the preregistered candidate policy."
                raise ValueError(msg)
            if self.execution_budget.normalized_compute_cap is None:
                msg = "paper-screen bindings require the pilot-frozen normalized compute cap."
                raise ValueError(msg)
        elif projection.screening_eligible or projection.promotion_eligible:
            msg = "Only paper-screen bindings can enter screening or promotion."
            raise ValueError(msg)
        if method == "adapt_style_state_preparation":
            if preset == "paper-screen":
                if (
                    artifact.implementation_kind != "operator_growth"
                    or not isinstance(self.operator_growth_spec, OperatorGrowthExecutionSpec)
                    or not isinstance(artifact.implementation_payload, OperatorGrowthExecutionSpec)
                    or artifact.implementation_payload != self.operator_growth_spec
                ):
                    msg = "Paper-screen operator growth requires its exact production execution spec."
                    raise TypeError(msg)
                if self.operator_growth_spec.outer_evaluation_policy not in policies:
                    msg = "Operator-growth execution must use a fresh-evaluation policy bound by this treatment."
                    raise ValueError(msg)
            elif preset == "training-smoke":
                smoke_payload = artifact.implementation_payload
                if (
                    artifact.implementation_kind != "operator_growth_smoke"
                    or not isinstance(smoke_payload, OperatorGrowthSmokeSpec)
                    or self.operator_growth_spec is not None
                ):
                    msg = "Operator-growth smoke requires its truthful tiny-limit wrapper only."
                    raise TypeError(msg)
                limits = smoke_payload.effective_limits
                if (
                    smoke_payload.outer_evaluation_policy not in policies
                    or limits.training_update_count != self.execution_budget.total_update_count
                    or limits.training_trajectory_count != self.execution_budget.maximum_training_trajectory_count
                    or limits.checkpoint_validation_trajectory_count
                    != self.execution_budget.checkpoint_validation_trajectory_count
                    or limits.maximum_growth_steps != 1
                    or limits.reoptimization_steps_per_growth != 1
                ):
                    msg = "Operator-growth smoke effective limits disagree with its schedule or budget."
                    raise ValueError(msg)
            else:
                msg = "The operator-growth method is available only for structural smoke and paper screen."
                raise ValueError(msg)
            schedule = self.strategy_schedule
            if (
                schedule.sampling_policy.kind != "fixed_crn"
                or schedule.training_noise.mode != "matched"
                or schedule.noise_continuation.interpolation != "constant"
                or float(schedule.noise_continuation.start_strength_scale).hex() != float(1).hex()
                or float(schedule.noise_continuation.target_strength_scale).hex() != float(1).hex()
                or schedule.multistart.start_count != 1
            ):
                msg = "Operator growth supports only one fixed matched-noise CRN schedule."
                raise ValueError(msg)
        elif method == "energy_adapt_vqe":
            if preset != "training-smoke" or artifact.implementation_kind != "tfim_operator_growth":
                msg = "TFIM energy ADAPT is restricted to its non-promotional smoke implementation."
                raise ValueError(msg)
            if self.operator_growth_spec is not None:
                msg = "The projector-growth spec cannot be substituted for TFIM energy ADAPT."
                raise ValueError(msg)
            payload = artifact.implementation_payload
            if not isinstance(payload, EnergyAdaptSmokeSpec) or payload.outer_evaluation_policy not in policies:
                msg = "Energy ADAPT smoke must bind its exact typed implementation and evaluation policy."
                raise ValueError(msg)
            limits = payload.effective_limits
            if (
                limits.training_update_count != self.execution_budget.total_update_count
                or limits.training_trajectory_count != self.execution_budget.maximum_training_trajectory_count
                or limits.checkpoint_validation_trajectory_count
                != self.execution_budget.checkpoint_validation_trajectory_count
                or limits.maximum_growth_steps != 1
                or limits.reoptimization_steps_per_growth != 1
            ):
                msg = "Energy ADAPT smoke effective limits disagree with its schedule or budget."
                raise ValueError(msg)
        elif self.operator_growth_spec is not None or artifact.implementation_kind != (
            "phase2_pipeline_smoke" if preset == "training-smoke" else "phase2_pipeline"
        ):
            msg = "operator-growth artifacts and specs are reserved for adapt_style_state_preparation."
            raise ValueError(msg)

    @property
    def key(self) -> tuple[Preset, str, TargetScope]:
        """Unique registry key for this binding."""
        return (self.preset, self.publication_candidate_checksum, self.target_scope_id)

    @property
    def implementation_checksum(self) -> str:
        """Checksum of the actual preset- and width-specific implementation."""
        return self.implementation_artifact.content_checksum

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered binding field."""
        return {
            "schema_version": self.schema_version,
            "binding_id": self.binding_id,
            "preset": self.preset,
            "publication_candidate_schema_version": self.publication_candidate_schema_version,
            "publication_candidate_checksum": self.publication_candidate_checksum,
            "publication_method_id": self.publication_method_id,
            "target_scope_id": self.target_scope_id,
            "qubit_count": self.qubit_count,
            "manifest_data_role": self.manifest_data_role,
            "execution_data_role": self.execution_data_role,
            "implementation_artifact": self.implementation_artifact.to_dict(),
            "strategy_schedule": self.strategy_schedule.to_dict(),
            "controlled_stage": self.controlled_stage.to_dict(),
            "evaluation_policies": [policy.to_dict() for policy in self.evaluation_policies],
            "pilot_diagnostic_policy": (
                None if self.pilot_diagnostic_policy is None else self.pilot_diagnostic_policy.to_dict()
            ),
            "execution_budget": self.execution_budget.to_dict(),
            "resource_policy": self.resource_policy.to_dict(),
            "treatment_projection": self.treatment_projection.to_dict(),
            "operator_growth_spec": None if self.operator_growth_spec is None else self.operator_growth_spec.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete publication-to-execution binding."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ScopedImplementationBinding:
        """Decode and verify one complete scoped implementation binding.

        Returns:
            The verified scoped implementation binding.

        Raises:
            TypeError: If a serialized collection has the wrong JSON-native type.
            ValueError: If the normalized binding checksum differs.
        """
        mapping = _verify(
            value,
            keys=_BINDING_KEYS,
            version=SCOPED_IMPLEMENTATION_BINDING_SCHEMA_VERSION,
            name="scoped implementation binding",
        )
        raw_policies = mapping["evaluation_policies"]
        if type(raw_policies) is not tuple:
            msg = "evaluation_policies must be a JSON array."
            raise TypeError(msg)
        raw_diagnostic = mapping["pilot_diagnostic_policy"]
        raw_operator = mapping["operator_growth_spec"]
        binding = cls(
            binding_id=cast("str", mapping["binding_id"]),
            preset=cast("Preset", mapping["preset"]),
            publication_candidate_schema_version=cast(
                "str",
                mapping["publication_candidate_schema_version"],
            ),
            publication_candidate_checksum=cast("str", mapping["publication_candidate_checksum"]),
            publication_method_id=cast("str", mapping["publication_method_id"]),
            target_scope_id=cast("TargetScope", mapping["target_scope_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            manifest_data_role=cast("ManifestRole", mapping["manifest_data_role"]),
            execution_data_role=cast("ExecutionRole", mapping["execution_data_role"]),
            implementation_artifact=ExecutionImplementationArtifact.from_dict(mapping["implementation_artifact"]),
            strategy_schedule=TrainingStrategySchedule.from_dict(mapping["strategy_schedule"]),
            controlled_stage=ControlledTrainingStage.from_dict(mapping["controlled_stage"]),
            evaluation_policies=tuple(FreshEvaluationPolicy.from_dict(policy) for policy in raw_policies),
            pilot_diagnostic_policy=(
                None if raw_diagnostic is None else PilotDiagnosticPolicy.from_dict(raw_diagnostic)
            ),
            execution_budget=ExecutionBudget.from_dict(mapping["execution_budget"]),
            resource_policy=BindingResourcePolicy.from_dict(mapping["resource_policy"]),
            treatment_projection=QubitTreatmentProjection.from_dict(mapping["treatment_projection"]),
            operator_growth_spec=(
                None if raw_operator is None else OperatorGrowthExecutionSpec.from_dict(raw_operator)
            ),
        )
        if mapping["content_checksum"] != binding.content_checksum:
            msg = "Scoped implementation binding checksum changed during normalization."
            raise ValueError(msg)
        return binding

    @classmethod
    def from_json(cls, payload: str) -> ScopedImplementationBinding:
        """Decode canonical JSON into a verified scoped binding.

        Returns:
            The verified scoped implementation binding.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class TrainingExecutionProfile:
    """Repository-owned registry for one complete execution preset."""

    profile_id: str
    preset: Preset
    preregistration_checksum: str
    implementation_plan_commit: str
    operational_protocol_amendment: OperationalProtocolAmendment
    bindings: tuple[ScopedImplementationBinding, ...]
    schema_version: str = field(default=TRAINING_EXECUTION_PROFILE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Enforce unique lookup, exact pilot widths, and the nonadaptive screen.

        Raises:
            TypeError: If the amendment or bindings use the wrong record type.
            ValueError: If profile identity, membership, amendment, or preset policy differs.
        """
        object.__setattr__(self, "profile_id", require_slug(self.profile_id, "profile_id"))
        preset = _require_preset(self.preset)
        if preset in _WP22A_DEFERRED_PRESETS:
            msg = f"{preset} profiles are intentionally unavailable until their owning post-WP22A package."
            raise ValueError(msg)
        object.__setattr__(self, "preset", preset)
        preregistration_checksum = require_checksum(self.preregistration_checksum, "preregistration_checksum")
        object.__setattr__(self, "preregistration_checksum", preregistration_checksum)
        commit = require_git_commit(self.implementation_plan_commit, "implementation_plan_commit")
        if commit != FROZEN_IMPLEMENTATION_PLAN_COMMIT:
            msg = "Training execution profiles must bind the frozen WP22A implementation-plan commit."
            raise ValueError(msg)
        object.__setattr__(self, "implementation_plan_commit", commit)
        if not isinstance(self.operational_protocol_amendment, OperationalProtocolAmendment):
            msg = "operational_protocol_amendment must be an OperationalProtocolAmendment."
            raise TypeError(msg)
        amendment = self.operational_protocol_amendment.to_dict()
        if amendment["preregistration_checksum"] != preregistration_checksum:
            msg = "Operational protocol amendment and execution profile preregistration differ."
            raise ValueError(msg)
        if amendment["implementation_plan_commit"] != commit:
            msg = "Operational protocol amendment and execution profile plan commits differ."
            raise ValueError(msg)
        expected_amendment_values = {
            "pilot_method_ids": list(PILOT_METHOD_IDS),
            "screen_method_ids": list(SCREEN_METHOD_IDS),
            "screen_method_count": SCREEN_CANDIDATE_COUNT,
            "screen_target_count": SCREEN_TARGET_COUNT,
            "screen_optimization_seed_count": SCREEN_OPTIMIZATION_SEED_COUNT,
            "screen_cell_count": SCREEN_CELL_COUNT,
            "screen_adaptive": False,
            "q12_inference_eligible": False,
            "q12_screening_eligible": False,
            "q12_promotion_eligible": False,
        }
        if any(amendment[name] != expected for name, expected in expected_amendment_values.items()):
            msg = "Operational protocol amendment differs from the frozen binding universe."
            raise ValueError(msg)
        bindings = tuple(self.bindings)
        if not bindings or any(not isinstance(binding, ScopedImplementationBinding) for binding in bindings):
            msg = "bindings must contain ScopedImplementationBinding records."
            raise TypeError(msg)
        if any(binding.preset != preset for binding in bindings):
            msg = "Every binding must belong to the execution profile's single preset."
            raise ValueError(msg)
        keys = tuple(binding.key for binding in bindings)
        if len(keys) != len(set(keys)):
            msg = "Bindings must be unique by preset, publication candidate checksum, and target scope."
            raise ValueError(msg)
        implementation_checksums = tuple(binding.implementation_checksum for binding in bindings)
        if len(implementation_checksums) != len(set(implementation_checksums)):
            msg = "Every width and preset binding requires a distinct concrete implementation checksum."
            raise ValueError(msg)
        candidate_methods: dict[str, str] = {}
        for binding in bindings:
            previous = candidate_methods.setdefault(
                binding.publication_candidate_checksum,
                binding.publication_method_id,
            )
            if previous != binding.publication_method_id:
                msg = "One publication candidate checksum cannot identify multiple methods."
                raise ValueError(msg)
        object.__setattr__(self, "bindings", bindings)
        self._validate_smoke_bindings()
        self._validate_pilot_bindings()
        self._validate_screen_bindings()

    def _validate_smoke_bindings(self) -> None:
        """Require all q6 identities for structural preflight, not execution evidence.

        WP22A seals the tiny limits and typed implementation references only.
        WP22B must still provide and test the runtime adapters before this preset
        can be described as executable or used as scientific evidence.

        Raises:
            ValueError: If smoke membership or scope differs from the exact universe.
        """
        if self.preset != "training-smoke":
            return
        methods = tuple(binding.publication_method_id for binding in self.bindings)
        if (
            len(self.bindings) != len(SMOKE_METHOD_IDS)
            or set(methods) != set(SMOKE_METHOD_IDS)
            or any(binding.target_scope_id != "primary_q6" for binding in self.bindings)
        ):
            msg = "training-smoke requires all ten frozen q6 implementation identities."
            raise ValueError(msg)

    def _validate_pilot_bindings(self) -> None:
        """Require the complete three-method q6/q12 pilot treatment projection.

        Raises:
            ValueError: If pilot membership or q12-to-q6 projection differs.
        """
        if self.preset != "paper-pilot":
            return
        pilot = self.bindings
        expected = {(method, scope) for method in PILOT_METHOD_IDS for scope in _TARGET_SCOPES}
        actual = {(binding.publication_method_id, binding.target_scope_id) for binding in pilot}
        if actual != expected or len(pilot) != len(expected):
            msg = "paper-pilot requires exactly the three frozen methods at both q6 and q12."
            raise ValueError(msg)
        for method in PILOT_METHOD_IDS:
            q6 = next(
                binding
                for binding in pilot
                if binding.publication_method_id == method and binding.target_scope_id == "primary_q6"
            )
            q12 = next(
                binding
                for binding in pilot
                if binding.publication_method_id == method and binding.target_scope_id == "secondary_q12"
            )
            if q6.publication_candidate_checksum != q12.publication_candidate_checksum:
                msg = "q6 and q12 pilot projections must share one publication-candidate identity."
                raise ValueError(msg)
            if q12.treatment_projection.primary_q6_implementation_checksum != q6.implementation_checksum:
                msg = "Each q12 pilot implementation must project explicitly to its exact q6 treatment."
                raise ValueError(msg)

    def _validate_screen_bindings(self) -> None:
        """Require the exact nine-candidate, q6-only, nonadaptive paper screen.

        Raises:
            ValueError: If screen membership, scope, role, or cell identity differs.
        """
        if self.preset != "paper-screen":
            return
        screen = self.bindings
        methods = tuple(binding.publication_method_id for binding in screen)
        if (
            len(screen) != SCREEN_CANDIDATE_COUNT
            or set(methods) != set(SCREEN_METHOD_IDS)
            or any(binding.target_scope_id != "primary_q6" for binding in screen)
        ):
            msg = "paper-screen requires exactly the nine frozen q6 candidates and excludes q12."
            raise ValueError(msg)
        outer_counts = {
            policy.trajectory_count
            for binding in screen
            for policy in binding.evaluation_policies
            if policy.purpose == "screening_outer"
        }
        compute_caps = {binding.execution_budget.normalized_compute_cap for binding in screen}
        if len(outer_counts) != 1 or len(compute_caps) != 1:
            msg = "paper-screen requires one common fixed outer count and normalized compute cap."
            raise ValueError(msg)

    @property
    def operational_protocol_amendment_checksum(self) -> str:
        """Checksum of the bound frozen operational amendment."""
        return self.operational_protocol_amendment.content_checksum

    def binding(
        self,
        preset: Preset,
        publication_candidate_checksum: str,
        target_scope_id: TargetScope,
    ) -> ScopedImplementationBinding:
        """Resolve one binding by its complete unique key.

        Returns:
            The exact scoped implementation binding.

        Raises:
            KeyError: If the profile contains no such binding.
        """
        key = (
            _require_preset(preset),
            require_checksum(publication_candidate_checksum, "publication_candidate_checksum"),
            _require_target_scope(target_scope_id),
        )
        for binding in self.bindings:
            if binding.key == key:
                return binding
        msg = f"No scoped implementation binding exists for key {key!r}."
        raise KeyError(msg)

    @staticmethod
    def _screen_design() -> dict[str, object]:
        """Return the frozen nonadaptive q6 screen universe."""
        return {
            "adaptation": "none",
            "target_scope_id": "primary_q6",
            "candidate_count": SCREEN_CANDIDATE_COUNT,
            "target_count": SCREEN_TARGET_COUNT,
            "optimization_seed_count": SCREEN_OPTIMIZATION_SEED_COUNT,
            "paired_block_count": SCREEN_PAIRED_BLOCK_COUNT,
            "cell_count": SCREEN_CELL_COUNT,
            "job_count": SCREEN_JOB_COUNT,
        }

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered profile field."""
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "preset": self.preset,
            "preregistration_checksum": self.preregistration_checksum,
            "implementation_plan_commit": self.implementation_plan_commit,
            "operational_protocol_amendment": self.operational_protocol_amendment.to_dict(),
            "operational_protocol_amendment_checksum": self.operational_protocol_amendment_checksum,
            "screen_design": self._screen_design(),
            "bindings": [binding.to_dict() for binding in self.bindings],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete execution registry."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> TrainingExecutionProfile:
        """Decode and verify a complete execution profile.

        Returns:
            The verified training execution profile.

        Raises:
            TypeError: If serialized bindings do not use a JSON array.
            ValueError: If amendment, design, or profile checksums differ.
        """
        mapping = _verify(
            value,
            keys=_PROFILE_KEYS,
            version=TRAINING_EXECUTION_PROFILE_SCHEMA_VERSION,
            name="training execution profile",
        )
        raw_bindings = mapping["bindings"]
        if type(raw_bindings) is not tuple:
            msg = "bindings must be a JSON array."
            raise TypeError(msg)
        profile = cls(
            profile_id=cast("str", mapping["profile_id"]),
            preset=cast("Preset", mapping["preset"]),
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            implementation_plan_commit=cast("str", mapping["implementation_plan_commit"]),
            operational_protocol_amendment=OperationalProtocolAmendment.from_dict(
                mapping["operational_protocol_amendment"]
            ),
            bindings=tuple(ScopedImplementationBinding.from_dict(binding) for binding in raw_bindings),
        )
        if mapping["operational_protocol_amendment_checksum"] != profile.operational_protocol_amendment_checksum:
            msg = "Serialized operational protocol amendment checksum is inconsistent."
            raise ValueError(msg)
        screen_design = require_mapping(mapping["screen_design"], "screen_design")
        require_exact_keys(screen_design, frozenset(profile._screen_design()), "screen_design")
        if dict(screen_design) != profile._screen_design():
            msg = "Serialized screen design differs from the frozen nonadaptive 1,296-job universe."
            raise ValueError(msg)
        if mapping["content_checksum"] != profile.content_checksum:
            msg = "Training execution profile checksum changed during normalization."
            raise ValueError(msg)
        return profile

    @classmethod
    def from_json(cls, payload: str) -> TrainingExecutionProfile:
        """Decode canonical JSON into a verified execution profile.

        Returns:
            The verified training execution profile.
        """
        return cls.from_dict(load_canonical_json_object(payload))


__all__ = [
    "CONTROLLED_TRAINING_STAGE_SCHEMA_VERSION",
    "ENERGY_ADAPT_SMOKE_SPEC_SCHEMA_VERSION",
    "EXECUTION_BUDGET_SCHEMA_VERSION",
    "EXECUTION_IMPLEMENTATION_ARTIFACT_SCHEMA_VERSION",
    "FROZEN_IMPLEMENTATION_PLAN_COMMIT",
    "OPERATOR_GROWTH_SMOKE_SPEC_SCHEMA_VERSION",
    "PILOT_METHOD_IDS",
    "PIPELINE_SMOKE_SPEC_SCHEMA_VERSION",
    "QUBIT_TREATMENT_PROJECTION_SCHEMA_VERSION",
    "RESOURCE_POLICY_SCHEMA_VERSION",
    "SCOPED_IMPLEMENTATION_BINDING_SCHEMA_VERSION",
    "SCREEN_CANDIDATE_COUNT",
    "SCREEN_CELL_COUNT",
    "SCREEN_JOB_COUNT",
    "SCREEN_METHOD_IDS",
    "SCREEN_OPTIMIZATION_SEED_COUNT",
    "SCREEN_PAIRED_BLOCK_COUNT",
    "SCREEN_TARGET_COUNT",
    "SMOKE_EXECUTION_LIMITS_SCHEMA_VERSION",
    "SMOKE_METHOD_IDS",
    "TRAINING_EXECUTION_PROFILE_SCHEMA_VERSION",
    "TRAINING_PRESETS",
    "BindingResourcePolicy",
    "ControlledTrainingStage",
    "EnergyAdaptSmokeSpec",
    "ExecutionBudget",
    "ExecutionImplementationArtifact",
    "ExecutionRole",
    "ImplementationKind",
    "InferenceRole",
    "ManifestRole",
    "OperatorGrowthSmokeSpec",
    "PipelineSmokeSpec",
    "Preset",
    "QubitTreatmentProjection",
    "ScopedImplementationBinding",
    "SmokeExecutionLimits",
    "TargetScope",
    "TrainingExecutionProfile",
]
