# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Strict pipeline-schema tests for the four WP21 top-down methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.pipeline import (
    CheckpointValidationConfig,
    TrainingPipelineTemplate,
    TrainingStageTemplate,
)
from benchmarks.state_preparation.phase2.protocol import TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM
from benchmarks.state_preparation.phase2.pruning import (
    TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
    TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID,
    TOPDOWN_MAGNITUDE_METHOD_ID,
    TOPDOWN_RANDOM_METHOD_ID,
    PruningStagePolicy,
    PruningStageSpec,
    ScoringObjectiveKind,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_NOISE_VERSION = "yaqs.state_preparation.noise.v1"

_METHOD_RULES = {
    TOPDOWN_RANDOM_METHOD_ID: "random",
    TOPDOWN_MAGNITUDE_METHOD_ID: "magnitude",
    TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID: "impact_one_shot",
    TOPDOWN_IMPACT_ITERATIVE_METHOD_ID: "impact_iterative",
}


def _seed_domains() -> dict[str, object]:
    """Return the exact Phase II seed-domain registry.

    Returns:
        A detached seed-domain mapping.
    """
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
    """Return the frozen native materialization policy.

    Returns:
        A detached materialization-policy mapping.
    """
    return {
        "policy_id": "native_chain_v1",
        "compiler_policy_id": "quantinuum_rzz_chain_v1",
        "connectivity_id": "linear_chain",
        "routing_policy_id": "identity_no_swap",
        "optimization_level": 0,
        "noise_placement": "logical_parameterized_gates",
        "parameter_source": "selected_checkpoint",
    }


def _disabled_checkpoint_policy() -> dict[str, object]:
    """Return the template form of disabled checkpoint validation.

    Returns:
        A detached policy without a resolved seed.
    """
    result = CheckpointValidationConfig.disabled().to_dict()
    del result["seed"]
    return result


def _pruning_policy(
    objective: ScoringObjectiveKind,
    *,
    relax_after_round: bool = False,
) -> PruningStagePolicy:
    """Build the common fixed-count gate-pruning policy.

    Returns:
        A strict embedded WP21 policy.
    """
    return PruningStagePolicy(
        pruning_unit="gate",
        scoring_objective_kind=objective,
        removal_schedule="fixed_count",
        removal_count=2,
        removal_fraction=None,
        relax_after_round=relax_after_round,
    )


def _optimize_stage(
    index: int,
    stage_id: str,
    *,
    input_topology: str | None,
    output_topology: str,
    parameter_count: int,
) -> TrainingStageTemplate:
    """Build an active noiseless initialization or relaxation stage.

    Returns:
        A validated stage template.
    """
    first = input_topology is None
    return TrainingStageTemplate(
        stage_policy={
            "stage_index": index,
            "stage_id": stage_id,
            "stage_kind": "optimize",
            "input_topology_id": input_topology,
            "output_topology_id": output_topology,
            "input_parameter_count": 0 if first else parameter_count,
            "output_parameter_count": parameter_count,
            "parameter_transfer_rule": "initialize_zeros" if first else "copy",
            "optimizer_id": "krotov",
            "optimizer_hyperparameters": {"learning_rate": 0.01},
            "iteration_budget": 2,
            "training_noise_id": "noiseless",
            "noise_definition_version": _NOISE_VERSION,
            "noise_strength_scale": None,
            "tjm_dt": None,
            "trajectory_count": 0,
            "trajectory_update": None,
            "sampling_policy": "none",
            "crn_refresh_interval": None,
            "checkpoint_validation_policy": _disabled_checkpoint_policy(),
            "pruning_rule": "none",
            "pruning_threshold": None,
            "max_bond_dimension": 64,
            "svd_threshold": 0.0,
            "truncation_mode": "discarded_weight",
            "min_bond_dimension": 1,
        },
        seed_bindings={
            "initialization": None,
            "optimizer": f"{stage_id}_optimizer",
            "training": None,
            "checkpoint_validation": None,
        },
    )


def _prune_stage(
    index: int,
    stage_id: str,
    *,
    rule: str,
    objective: ScoringObjectiveKind,
    input_topology: str,
    output_topology: str,
    input_parameters: int,
    output_parameters: int,
    relax_after_round: bool = False,
    noisy: bool = False,
    policy_mapping: Mapping[str, object] | None = None,
) -> TrainingStageTemplate:
    """Build one schema-validated WP21 pruning transform.

    Returns:
        A validated pruning-stage template.
    """
    embedded = (
        _pruning_policy(objective, relax_after_round=relax_after_round).to_mapping()
        if policy_mapping is None
        else dict(policy_mapping)
    )
    return TrainingStageTemplate(
        stage_policy={
            "stage_index": index,
            "stage_id": stage_id,
            "stage_kind": "prune",
            "input_topology_id": input_topology,
            "output_topology_id": output_topology,
            "input_parameter_count": input_parameters,
            "output_parameter_count": output_parameters,
            "parameter_transfer_rule": "apply_pruning_mask",
            "optimizer_id": "none",
            "optimizer_hyperparameters": embedded,
            "iteration_budget": 0,
            "training_noise_id": "depolarizing_1s_all" if noisy else "noiseless",
            "noise_definition_version": _NOISE_VERSION,
            "noise_strength_scale": 1.0 if noisy else None,
            "tjm_dt": 1.0 if noisy else None,
            "trajectory_count": 8 if noisy else 0,
            "trajectory_update": "independent" if noisy else None,
            "sampling_policy": "crn_fixed" if noisy else "none",
            "crn_refresh_interval": None,
            "checkpoint_validation_policy": _disabled_checkpoint_policy(),
            "pruning_rule": rule,
            "pruning_threshold": 2.0,
            "max_bond_dimension": 64,
            "svd_threshold": 0.0,
            "truncation_mode": "discarded_weight",
            "min_bond_dimension": 1,
        },
        seed_bindings={
            "initialization": None,
            "optimizer": f"{stage_id}_ordering" if rule == "random" else None,
            "training": f"{stage_id}_training" if noisy else None,
            "checkpoint_validation": None,
        },
    )


def _pipeline(
    method_id: str,
    stages: Sequence[TrainingStageTemplate],
    *,
    identity_suffix: str,
) -> TrainingPipelineTemplate:
    """Build one top-down pipeline around an already contiguous stage chain.

    Returns:
        A strict pipeline template.
    """
    return TrainingPipelineTemplate(
        template_id=f"{method_id}_{identity_suffix}",
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id="primary_q6",
        ansatz_family="bmpd_brickwall",
        method_id=method_id,
        method_version="wp21-v1",
        resource_stratum_id="primary_cap_12",
        stages=tuple(stages),
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )


def _single_round_pipeline(
    method_id: str,
    *,
    objective: ScoringObjectiveKind,
    noisy: bool = False,
) -> TrainingPipelineTemplate:
    """Build a valid single-round pipeline for one registered method.

    Returns:
        A validated pipeline template.
    """
    initial = _optimize_stage(
        0,
        "train_deep_start",
        input_topology=None,
        output_topology="deep_start",
        parameter_count=8,
    )
    pruning = _prune_stage(
        1,
        "prune_round_1",
        rule=_METHOD_RULES[method_id],
        objective=objective,
        input_topology="deep_start",
        output_topology="pruned_round_1",
        input_parameters=8,
        output_parameters=6,
        noisy=noisy,
    )
    return _pipeline(method_id, (initial, pruning), identity_suffix="single_round")


def _iterative_pipeline(
    *,
    terminal_relaxation: bool,
    terminal_policy_relax: bool,
    terminal_stage_id: str,
) -> TrainingPipelineTemplate:
    """Build a two-round iterative chain with a configurable terminal successor.

    Returns:
        The validated template when terminal semantics agree.
    """
    initial = _optimize_stage(
        0,
        "train_deep_start",
        input_topology=None,
        output_topology="deep_start",
        parameter_count=8,
    )
    first_prune = _prune_stage(
        1,
        "prune_round_1",
        rule="impact_iterative",
        objective="noiseless_fidelity",
        input_topology="deep_start",
        output_topology="pruned_round_1",
        input_parameters=8,
        output_parameters=6,
        relax_after_round=True,
    )
    first_relax = _optimize_stage(
        2,
        "relax_round_1",
        input_topology="pruned_round_1",
        output_topology="pruned_round_1",
        parameter_count=6,
    )
    second_prune = _prune_stage(
        3,
        "prune_round_2",
        rule="impact_iterative",
        objective="noiseless_fidelity",
        input_topology="pruned_round_1",
        output_topology="pruned_round_2",
        input_parameters=6,
        output_parameters=4,
        relax_after_round=terminal_policy_relax,
    )
    terminal = _optimize_stage(
        4,
        terminal_stage_id,
        input_topology="pruned_round_2",
        output_topology="pruned_round_2",
        parameter_count=4,
    )
    suffix = "terminal_relax" if terminal_relaxation else "final_finetune"
    return _pipeline(
        TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
        (initial, first_prune, first_relax, second_prune, terminal),
        identity_suffix=suffix,
    )


def test_policy_mapping_round_trip_and_exact_field_rejection() -> None:
    """Only the exact checksum-sealed pruning-policy schema is accepted."""
    policy = _pruning_policy("noiseless_fidelity", relax_after_round=True)
    assert PruningStagePolicy.from_mapping(policy.to_mapping()) == policy

    unknown = policy.to_mapping()
    unknown["undocumented_choice"] = "forbidden"
    with pytest.raises(ValueError, match="fields do not match"):
        PruningStagePolicy.from_mapping(unknown)

    mutated = policy.to_mapping()
    mutated["tie_break_rule"] = "unstable_input_order_v1"
    mutated["policy_checksum"] = canonical_checksum({
        key: value for key, value in mutated.items() if key != "policy_checksum"
    })
    with pytest.raises(ValueError, match="tie_break_rule"):
        PruningStagePolicy.from_mapping(mutated)


@pytest.mark.parametrize(
    ("method_id", "rule", "objective", "noisy"),
    [
        (TOPDOWN_RANDOM_METHOD_ID, "random", "none", False),
        (TOPDOWN_MAGNITUDE_METHOD_ID, "magnitude", "none", False),
        (TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID, "impact_one_shot", "noiseless_fidelity", False),
        (
            TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
            "impact_iterative",
            "noiseless_fidelity",
            False,
        ),
    ],
)
def test_all_four_method_rule_pairs_are_strict_and_round_trip(
    method_id: str,
    rule: str,
    objective: ScoringObjectiveKind,
    *,
    noisy: bool,
) -> None:
    """Every registered method accepts only its matching pruning rule."""
    if method_id == TOPDOWN_IMPACT_ITERATIVE_METHOD_ID:
        template = _iterative_pipeline(
            terminal_relaxation=False,
            terminal_policy_relax=False,
            terminal_stage_id="final_finetune",
        )
    else:
        template = _single_round_pipeline(method_id, objective=objective, noisy=noisy)
    assert template.method_id == method_id
    pruning_stages = tuple(stage for stage in template.stages if stage.stage_policy["stage_kind"] == "prune")
    assert pruning_stages
    assert all(stage.stage_policy["pruning_rule"] == rule for stage in pruning_stages)
    assert TrainingPipelineTemplate.from_dict(template.to_dict()) == template

    if method_id == TOPDOWN_IMPACT_ITERATIVE_METHOD_ID:
        return
    with pytest.raises(ValueError, match="method_id must agree"):
        _pipeline(
            method_id,
            (
                template.stages[0],
                _prune_stage(
                    1,
                    "mismatched_prune",
                    rule="magnitude" if rule != "magnitude" else "random",
                    objective="none",
                    input_topology="deep_start",
                    output_topology="pruned_round_1",
                    input_parameters=8,
                    output_parameters=6,
                ),
            ),
            identity_suffix="mismatch",
        )


def test_random_seed_is_required_only_for_random_pruning() -> None:
    """The optimizer-ordering seed is neither implicit nor shared by other rules."""
    random_stage = _single_round_pipeline(TOPDOWN_RANDOM_METHOD_ID, objective="none").stages[-1]
    missing = dict(random_stage.seed_bindings)
    missing["optimizer"] = None
    with pytest.raises(ValueError, match=r"seed_bindings\.optimizer"):
        TrainingStageTemplate(stage_policy=random_stage.stage_policy, seed_bindings=missing)

    magnitude_stage = _single_round_pipeline(TOPDOWN_MAGNITUDE_METHOD_ID, objective="none").stages[-1]
    surplus = dict(magnitude_stage.seed_bindings)
    surplus["optimizer"] = "forbidden_magnitude_ordering"
    with pytest.raises(ValueError, match=r"seed_bindings\.optimizer"):
        TrainingStageTemplate(stage_policy=magnitude_stage.stage_policy, seed_bindings=surplus)

    random_policy = _pruning_policy("none").to_mapping()
    with pytest.raises(ValueError, match="random seed"):
        PruningStageSpec.from_mapping(
            random_policy,
            method_id=TOPDOWN_RANDOM_METHOD_ID,
            score_rule="random",
            random_seed=None,
        )
    with pytest.raises(ValueError, match="random seed"):
        PruningStageSpec.from_mapping(
            random_policy,
            method_id=TOPDOWN_MAGNITUDE_METHOD_ID,
            score_rule="magnitude",
            random_seed=9,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampling_policy", "resampled"),
        ("trajectory_update", "cross"),
    ],
)
def test_fixed_map_noisy_impact_requires_independent_fixed_crn(field: str, value: str) -> None:
    """Noisy impact scores bind one independent CRN map to the round input."""
    valid = _single_round_pipeline(
        TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID,
        objective="fixed_map_sample_average_fidelity",
        noisy=True,
    ).stages[-1]
    policy = dict(valid.stage_policy)
    policy[field] = value
    with pytest.raises(ValueError, match="Fixed-map impact scoring"):
        TrainingStageTemplate(stage_policy=policy, seed_bindings=valid.seed_bindings)


def test_fixed_map_and_noiseless_impact_noise_constraints_are_disjoint() -> None:
    """Fixed-map impact is noisy, whereas noiseless impact forbids all sampling."""
    with pytest.raises(ValueError, match="Fixed-map impact scoring"):
        _prune_stage(
            0,
            "invalid_fixed_map_noiseless",
            rule="impact_one_shot",
            objective="fixed_map_sample_average_fidelity",
            input_topology="deep_start",
            output_topology="pruned_round_1",
            input_parameters=8,
            output_parameters=6,
        )

    with pytest.raises(ValueError, match="Only fixed-map impact scoring"):
        _prune_stage(
            0,
            "invalid_noiseless_sampled",
            rule="impact_one_shot",
            objective="noiseless_fidelity",
            input_topology="deep_start",
            output_topology="pruned_round_1",
            input_parameters=8,
            output_parameters=6,
            noisy=True,
        )


def test_one_shot_methods_require_exactly_one_pruning_transform() -> None:
    """A one-shot identity cannot silently encode a second scoring/removal round."""
    valid = _single_round_pipeline(
        TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID,
        objective="noiseless_fidelity",
    )
    assert len(valid.stages) == 2
    second = _prune_stage(
        2,
        "prune_round_2",
        rule="impact_one_shot",
        objective="noiseless_fidelity",
        input_topology="pruned_round_1",
        output_topology="pruned_round_2",
        input_parameters=6,
        output_parameters=4,
    )
    with pytest.raises(ValueError, match="exactly one"):
        _pipeline(
            TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID,
            (*valid.stages, second),
            identity_suffix="two_rounds",
        )


def test_iterative_method_rejects_a_single_pruning_round() -> None:
    """An iterative identity cannot collapse to one-shot impact pruning."""
    with pytest.raises(ValueError, match="at least two"):
        _single_round_pipeline(
            TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
            objective="noiseless_fidelity",
        )


def test_iterative_rounds_alternate_with_relaxation_and_distinguish_final_finetuning() -> None:
    """Relaxation names and flags distinguish algorithm rounds from final fine-tuning."""
    terminal_relax = _iterative_pipeline(
        terminal_relaxation=True,
        terminal_policy_relax=True,
        terminal_stage_id="relax_round_2",
    )
    final_finetune = _iterative_pipeline(
        terminal_relaxation=False,
        terminal_policy_relax=False,
        terminal_stage_id="final_finetune",
    )
    assert terminal_relax.stages[-1].stage_id == "relax_round_2"
    assert final_finetune.stages[-1].stage_id == "final_finetune"
    assert terminal_relax.configuration_checksum != final_finetune.configuration_checksum

    with pytest.raises(ValueError, match="terminal iterative pruning relaxation"):
        _iterative_pipeline(
            terminal_relaxation=False,
            terminal_policy_relax=True,
            terminal_stage_id="final_finetune",
        )
    with pytest.raises(ValueError, match="terminal iterative pruning relaxation"):
        _iterative_pipeline(
            terminal_relaxation=True,
            terminal_policy_relax=False,
            terminal_stage_id="relax_round_2",
        )


def test_iterative_rounds_reject_a_missing_intermediate_relaxation() -> None:
    """Two iterative pruning transforms cannot be adjacent in the stage chain."""
    initial = _optimize_stage(
        0,
        "train_deep_start",
        input_topology=None,
        output_topology="deep_start",
        parameter_count=8,
    )
    first = _prune_stage(
        1,
        "prune_round_1",
        rule="impact_iterative",
        objective="noiseless_fidelity",
        input_topology="deep_start",
        output_topology="pruned_round_1",
        input_parameters=8,
        output_parameters=6,
        relax_after_round=True,
    )
    second = _prune_stage(
        2,
        "prune_round_2",
        rule="impact_iterative",
        objective="noiseless_fidelity",
        input_topology="pruned_round_1",
        output_topology="pruned_round_2",
        input_parameters=6,
        output_parameters=4,
    )
    with pytest.raises(ValueError, match="alternate with active relaxation"):
        _pipeline(
            TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
            (initial, first, second),
            identity_suffix="missing_relaxation",
        )


def test_all_topdown_method_identities_remain_distinct() -> None:
    """The four algorithms cannot collapse to one candidate identity."""
    templates = (
        _single_round_pipeline(TOPDOWN_RANDOM_METHOD_ID, objective="none"),
        _single_round_pipeline(TOPDOWN_MAGNITUDE_METHOD_ID, objective="none"),
        _single_round_pipeline(TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID, objective="noiseless_fidelity"),
        _iterative_pipeline(
            terminal_relaxation=False,
            terminal_policy_relax=False,
            terminal_stage_id="final_finetune",
        ),
    )
    assert len({template.method_id for template in templates}) == 4
    assert len({template.configuration_checksum for template in templates}) == 4
