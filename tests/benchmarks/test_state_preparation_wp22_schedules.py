# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for sealed WP22 training schedules and deterministic CRN membership."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from benchmarks.state_preparation.phase2.canonical import seal_mapping
from benchmarks.state_preparation.phase2.protocol import TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM
from benchmarks.state_preparation.phase2.training_schedules import (
    CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID,
    CONFIRMATORY_OPTIMIZATION_SEED_POLICY_ID,
    EXECUTION_SEED_POLICY_IDS,
    MAX_MULTISTART_COUNT,
    PILOT_DIAGNOSTIC_SEED_POLICY_ID,
    PILOT_FRESH_EVALUATION_SEED_POLICY_ID,
    PILOT_OPTIMIZATION_SEED_POLICY_ID,
    SCHEDULE_SEED_DERIVATION_POLICY_ID,
    SCREEN_OPTIMIZATION_SEED_POLICY_ID,
    SCREENING_CELL_SEED_POLICY_ID,
    SCREENING_ROOT_SEED_POLICY_ID,
    SMOKE_FRESH_EVALUATION_SEED_POLICY_ID,
    SMOKE_OPTIMIZATION_SEED_POLICY_ID,
    STAGE_SEED_DERIVATION_POLICY_ID,
    CheckpointValidationPolicy,
    CheckpointValidationTracker,
    ExecutionSeedPolicySuite,
    FrozenTrainingPolicyUniverse,
    LimitedMultistartPlan,
    NoiselessPretrainNoisyFinetune,
    NoiseMixtureAllocation,
    NoiseMixtureComponent,
    NoiseStrengthContinuation,
    SeedDerivationPolicy,
    StandardNoiseMixture,
    TrainingStrategySchedule,
    TrajectoryCountCurriculum,
    TrajectoryCountStep,
    TrajectoryEnsembleMembership,
    TrajectorySamplingPolicy,
    ValidationCheckpoint,
    build_trajectory_membership,
    derive_map_seed,
    derive_member_seed,
    derive_role_seed,
)


def _schedule(*, policy: TrajectorySamplingPolicy | None = None) -> TrainingStrategySchedule:
    """Return one small internally consistent complete schedule."""
    return TrainingStrategySchedule(
        schedule_id="wp22_test",
        noise_continuation=NoiseStrengthContinuation(start_update=3, end_update=5, target_strength_scale=1.0),
        trajectory_curriculum=TrajectoryCountCurriculum((TrajectoryCountStep(0, 4), TrajectoryCountStep(4, 8))),
        sampling_policy=policy or TrajectorySamplingPolicy("fixed_crn"),
        checkpoint_validation=CheckpointValidationPolicy(patience=3, min_delta=0.01),
        phase_boundary=NoiselessPretrainNoisyFinetune(
            noiseless_pretrain_updates=3,
            noisy_finetune_updates=5,
        ),
        multistart=LimitedMultistartPlan(start_count=2, declared_cap=3),
        training_noise=StandardNoiseMixture(
            "matched",
            (NoiseMixtureComponent("depolarizing_1s_all", 1.0),),
        ),
    )


def _next_membership(
    policy: TrajectorySamplingPolicy,
    update: int,
    previous: TrajectoryEnsembleMembership | None,
    *,
    count: int = 6,
) -> TrajectoryEnsembleMembership:
    """Build one member of a deterministic training-trajectory stream.

    Returns:
        The requested exact ensemble membership.
    """
    return build_trajectory_membership(
        policy,
        master_seed=91,
        role="training_trajectory",
        update=update,
        trajectory_count=count,
        previous=previous,
    )


def test_noise_strength_continuation_has_exact_endpoints_and_roundtrip() -> None:
    """Linear continuation clamps and checksum-seals exact endpoint semantics."""
    continuation = NoiseStrengthContinuation(4, 8, 0.75)

    assert continuation.strength_at(0) == pytest.approx(0.0, abs=0.0)
    assert continuation.strength_at(4) == pytest.approx(0.0, abs=0.0)
    assert continuation.strength_at(6) == pytest.approx(0.375, abs=0.0)
    assert continuation.strength_at(8) == pytest.approx(0.75, abs=0.0)
    assert continuation.strength_at(20) == pytest.approx(0.75, abs=0.0)
    assert NoiseStrengthContinuation.from_json(continuation.to_json()) == continuation

    tampered = continuation.to_dict()
    tampered["target_strength_scale"] = 0.5
    with pytest.raises(ValueError, match="checksum mismatch"):
        NoiseStrengthContinuation.from_dict(tampered)
    with pytest.raises(ValueError, match="Linear continuation"):
        NoiseStrengthContinuation(3, 3, 1.0)


def test_trajectory_curriculum_uses_inclusive_monotone_boundaries() -> None:
    """Counts change exactly at declared boundaries and cannot decrease."""
    curriculum = TrajectoryCountCurriculum((
        TrajectoryCountStep(0, 3),
        TrajectoryCountStep(5, 7),
        TrajectoryCountStep(9, 7),
    ))

    assert [curriculum.count_at(update) for update in (0, 4, 5, 8, 9, 50)] == [3, 3, 7, 7, 7, 7]
    assert TrajectoryCountCurriculum.from_json(curriculum.to_json()) == curriculum
    with pytest.raises(ValueError, match="monotone"):
        TrajectoryCountCurriculum((TrajectoryCountStep(0, 5), TrajectoryCountStep(2, 4)))
    with pytest.raises(ValueError, match="update zero"):
        TrajectoryCountCurriculum((TrajectoryCountStep(1, 4),))


def test_sampling_policy_options_and_epoch_boundaries_are_strict() -> None:
    """Each sampling family accepts only its own refresh and retention fields."""
    fixed = TrajectorySamplingPolicy("fixed_crn")
    periodic = TrajectorySamplingPolicy("periodic_full_refresh", refresh_interval=3)
    rolling = TrajectorySamplingPolicy("rolling_ensemble", refresh_interval=2, retain_count=2)
    resampled = TrajectorySamplingPolicy("resampled")

    assert [fixed.epoch_at(update) for update in range(4)] == [0, 0, 0, 0]
    assert [periodic.epoch_at(update) for update in range(5)] == [0, 0, 0, 1, 1]
    assert [rolling.epoch_at(update) for update in range(5)] == [0, 0, 1, 1, 2]
    assert [resampled.epoch_at(update) for update in range(4)] == [0, 1, 2, 3]
    assert TrajectorySamplingPolicy.from_json(rolling.to_json()) == rolling
    with pytest.raises(ValueError, match="does not accept"):
        TrajectorySamplingPolicy("fixed_crn", refresh_interval=2)
    with pytest.raises(ValueError, match="exactly one"):
        TrajectorySamplingPolicy("rolling_ensemble", refresh_interval=2)
    with pytest.raises(ValueError, match="cannot retain"):
        TrajectorySamplingPolicy("periodic_full_refresh", refresh_interval=2, retain_count=1)


def test_fixed_periodic_and_resampled_membership_transitions() -> None:
    """Fixed CRN is stable, periodic CRN fully refreshes, and resampling is fresh."""
    fixed = TrajectorySamplingPolicy("fixed_crn")
    fixed_zero = _next_membership(fixed, 0, None)
    fixed_one = _next_membership(fixed, 1, fixed_zero)
    assert fixed_one.member_seeds == fixed_zero.member_seeds
    assert fixed_one.map_seed == fixed_zero.map_seed
    assert fixed_one.retained_member_count == 6

    periodic = TrajectorySamplingPolicy("periodic_full_refresh", refresh_interval=2)
    periodic_zero = _next_membership(periodic, 0, None)
    periodic_one = _next_membership(periodic, 1, periodic_zero)
    periodic_two = _next_membership(periodic, 2, periodic_one)
    assert periodic_one.member_seeds == periodic_zero.member_seeds
    assert periodic_one.retained_member_count == 6
    assert set(periodic_two.member_seeds).isdisjoint(periodic_one.member_seeds)
    assert periodic_two.retained_member_count == 0

    resampled = TrajectorySamplingPolicy("resampled")
    resampled_zero = _next_membership(resampled, 0, None)
    resampled_one = _next_membership(resampled, 1, resampled_zero)
    assert set(resampled_one.member_seeds).isdisjoint(resampled_zero.member_seeds)
    assert resampled_one.retained_member_count == 0


def test_fixed_membership_decrease_is_ranked_and_later_growth_preserves_members() -> None:
    """Count changes use deterministic retention and append-only growth semantics."""
    policy = TrajectorySamplingPolicy("fixed_crn")
    first = _next_membership(policy, 0, None, count=6)
    reduced = _next_membership(policy, 1, first, count=4)
    repeated = _next_membership(
        policy,
        1,
        TrajectoryEnsembleMembership.from_json(first.to_json()),
        count=4,
    )
    expanded = _next_membership(policy, 2, reduced, count=8)

    assert reduced == repeated
    assert reduced.retained_member_count == 4
    assert set(reduced.member_seeds).issubset(first.member_seeds)
    assert expanded.member_seeds[:4] == reduced.member_seeds
    assert expanded.retained_member_count == 4


@pytest.mark.parametrize(
    ("policy", "expected_retained"),
    [
        (TrajectorySamplingPolicy("rolling_ensemble", refresh_interval=1, retain_count=2), 2),
        (TrajectorySamplingPolicy("rolling_ensemble", refresh_interval=1, retain_fraction=0.5), 3),
    ],
)
def test_rolling_membership_retains_exact_declared_amount_across_resume(
    policy: TrajectorySamplingPolicy,
    expected_retained: int,
) -> None:
    """Persisted predecessor state reproduces exact deterministic rolling membership."""
    first = _next_membership(policy, 0, None)
    restored = TrajectoryEnsembleMembership.from_json(first.to_json())
    uninterrupted = _next_membership(policy, 1, first)
    resumed = _next_membership(policy, 1, restored)

    assert resumed == uninterrupted
    assert resumed.retained_member_count == expected_retained
    assert len(set(resumed.member_seeds) & set(first.member_seeds)) == expected_retained
    with pytest.raises(ValueError, match="exact integer"):
        TrajectorySamplingPolicy(
            "rolling_ensemble",
            refresh_interval=1,
            retain_fraction=0.3,
        ).retained_count(6, 6)


def test_post_zero_membership_requires_exact_predecessor_stream() -> None:
    """A resume cannot skip an update or cross policy and role domains."""
    policy = TrajectorySamplingPolicy("fixed_crn")
    first = _next_membership(policy, 0, None)
    with pytest.raises(TypeError, match="preceding"):
        _next_membership(policy, 1, None)
    with pytest.raises(ValueError, match="consecutive"):
        _next_membership(policy, 2, first)
    with pytest.raises(ValueError, match="consecutive"):
        build_trajectory_membership(
            policy,
            master_seed=91,
            role="checkpoint_validation",
            update=1,
            trajectory_count=6,
            previous=first,
        )


def test_seed_derivation_separates_role_map_member_and_multistart_domains() -> None:
    """Equal roots cannot alias scientific roles, maps, members, or starts."""
    role_seeds = {
        derive_role_seed(12, role, purpose="test")
        for role in (
            "initialization",
            "optimizer_ordering",
            "training_trajectory",
            "checkpoint_validation",
            "pilot_evaluation",
            "screening_selection",
            "confirmatory_test",
        )
    }
    assert len(role_seeds) == 7
    map_seed = derive_map_seed(12, "training_trajectory")
    member_seeds = {derive_member_seed(12, "training_trajectory", member_index=index) for index in range(5)}
    assert map_seed not in member_seeds
    assert len(member_seeds) == 5

    bundles = LimitedMultistartPlan(3, 3).seed_bundles(12)
    seeds = {
        value
        for bundle in bundles
        for value in (bundle.initialization_seed, bundle.optimizer_ordering_seed, bundle.training_trajectory_seed)
    }
    assert len(seeds) == 9
    assert bundles == LimitedMultistartPlan(3, 3).seed_bundles(12)


def test_execution_seed_policy_suite_freezes_exact_universe_and_golden_vectors() -> None:
    """Every execution purpose has one ordered sealed policy with reviewed derivations."""
    suite = ExecutionSeedPolicySuite.frozen()
    zero = "sha256:" + "0" * 64
    one = "sha256:" + "1" * 64
    two = "sha256:" + "2" * 64
    three = "sha256:" + "3" * 64

    assert tuple(policy.policy_id for policy in suite.policies) == EXECUTION_SEED_POLICY_IDS
    assert ExecutionSeedPolicySuite.from_json(suite.to_json()) == suite
    assert tuple(
        suite.derive(
            PILOT_OPTIMIZATION_SEED_POLICY_ID,
            {"preregistration_checksum": TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM, "seed_index": index},
        )
        for index in range(5)
    ) == (
        7318385955052047882,
        7744228800727108459,
        6477127604988620024,
        8093673719055324565,
        895920682149875227,
    )
    assert tuple(
        suite.derive(
            SCREEN_OPTIMIZATION_SEED_POLICY_ID,
            {"preregistration_checksum": TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM, "seed_index": index},
        )
        for index in range(3)
    ) == (7329660033858372585, 4524389697880734114, 4579802874124897325)

    golden_cases = (
        (SMOKE_OPTIMIZATION_SEED_POLICY_ID, {"publication_candidate_checksum": zero}, 17953725567951375462),
        (
            CONFIRMATORY_OPTIMIZATION_SEED_POLICY_ID,
            {"final_seal_checksum": zero, "target_instance_spec_checksum": one, "seed_index": 2},
            5441059875348405186,
        ),
        (
            SMOKE_FRESH_EVALUATION_SEED_POLICY_ID,
            {"publication_candidate_checksum": zero},
            17325230508059404107,
        ),
        (
            PILOT_FRESH_EVALUATION_SEED_POLICY_ID,
            {
                "target_manifest_checksum": zero,
                "target_instance_spec_checksum": one,
                "optimization_seed": 17,
                "publication_candidate_checksum": two,
            },
            4383180969964957079,
        ),
        (
            SCREENING_ROOT_SEED_POLICY_ID,
            {
                "preregistration_checksum": TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
                "screen_execution_profile_checksum": zero,
                "screening_target_manifest_checksum": one,
            },
            15409197646567244506,
        ),
        (
            SCREENING_CELL_SEED_POLICY_ID,
            {"root_seed": 17, "target_instance_id": "target-003", "optimization_seed": 23},
            1543422226519030536,
        ),
        (
            CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID,
            {
                "final_seal_checksum": zero,
                "target_instance_spec_checksum": one,
                "seed_index": 2,
                "configuration_checksum": two,
            },
            7739550078624138684,
        ),
        (
            PILOT_DIAGNOSTIC_SEED_POLICY_ID,
            {
                "target_manifest_checksum": zero,
                "target_instance_spec_checksum": one,
                "optimization_seed": 17,
                "publication_candidate_checksum": two,
                "repetition": 5,
            },
            707336837789888076,
        ),
        (
            STAGE_SEED_DERIVATION_POLICY_ID,
            {
                "optimization_seed": 17,
                "domain_id": "training_trajectory",
                "binding": "training",
                "resolution_context_checksum": three,
            },
            5627049241039278462,
        ),
        (
            SCHEDULE_SEED_DERIVATION_POLICY_ID,
            {
                "master_seed": 17,
                "role": "training_trajectory",
                "purpose": "trajectory_member",
                "stream_index": 2,
                "epoch": 3,
                "member_index": 5,
            },
            10139570943990584380,
        ),
    )
    for policy_id, coordinates, expected in golden_cases:
        assert suite.derive(policy_id, coordinates) == expected


def test_seed_policy_rejects_resealed_algorithm_domain_order_and_coordinate_drift() -> None:
    """A valid outer checksum cannot authorize changed seed semantics or malformed coordinates."""
    suite = ExecutionSeedPolicySuite.frozen()
    schedule_policy = suite.policy(SCHEDULE_SEED_DERIVATION_POLICY_ID)

    algorithm_drift = schedule_policy.to_dict()
    algorithm_drift.pop("content_checksum")
    algorithm_drift["output_width_bits"] = 32
    with pytest.raises(ValueError, match="algorithm or extraction"):
        SeedDerivationPolicy.from_dict(seal_mapping(algorithm_drift))

    policy_drift = suite.policy(PILOT_OPTIMIZATION_SEED_POLICY_ID).to_dict()
    policy_drift.pop("content_checksum")
    policy_drift["constant_fields"] = {"domain": "wp22_paper_pilot_optimization_drift"}
    altered_policy = seal_mapping(policy_drift)
    suite_drift = suite.to_dict()
    raw_policies = cast("list[object]", suite_drift["policies"])
    assert isinstance(raw_policies, list)
    raw_policies[0] = altered_policy
    suite_drift.pop("content_checksum")
    with pytest.raises(ValueError, match="reviewed WP22 universe"):
        ExecutionSeedPolicySuite.from_dict(seal_mapping(suite_drift))

    coordinates = {
        "master_seed": 17,
        "role": "training_trajectory",
        "purpose": "trajectory_member",
        "stream_index": 2,
        "epoch": 3,
        "member_index": 5,
    }
    with pytest.raises(ValueError, match="fields do not match"):
        schedule_policy.derive({**coordinates, "extra": 0})
    with pytest.raises(TypeError, match="must be an int"):
        schedule_policy.derive({**coordinates, "member_index": "5"})


def test_persisted_membership_seeds_are_direct_sampler_seeds_without_rederivation() -> None:
    """The training universe binds persisted membership values directly to sampler input."""
    universe = FrozenTrainingPolicyUniverse.frozen()
    suite = ExecutionSeedPolicySuite.frozen()
    membership = build_trajectory_membership(
        TrajectorySamplingPolicy("fixed_crn"),
        master_seed=91,
        role="training_trajectory",
        update=0,
        trajectory_count=3,
    )

    assert universe.schedule_seed_policy_id == SCHEDULE_SEED_DERIVATION_POLICY_ID
    assert universe.schedule_seed_policy_checksum == suite.policy(SCHEDULE_SEED_DERIVATION_POLICY_ID).content_checksum
    assert universe.persisted_membership_seed_usage == "direct_sampler_seed"
    assert universe.sampler_seed_rederivation is False
    assert membership.member_seeds == tuple(
        suite.derive(
            SCHEDULE_SEED_DERIVATION_POLICY_ID,
            {
                "master_seed": 91,
                "role": "training_trajectory",
                "purpose": "trajectory_member",
                "stream_index": 0,
                "epoch": 0,
                "member_index": index,
            },
        )
        for index in range(3)
    )


def test_validation_only_early_stopping_and_earliest_best_tie() -> None:
    """Patience observes validation only and exact best-score ties retain the earliest update."""
    policy = CheckpointValidationPolicy(patience=2, min_delta=0.02)
    checkpoints = (
        ValidationCheckpoint(0, 0.5),
        ValidationCheckpoint(2, 0.6),
        ValidationCheckpoint(4, 0.6),
        ValidationCheckpoint(6, 0.61),
        ValidationCheckpoint(8, 0.9),
    )

    selection = policy.select(checkpoints)
    assert selection.best_update == 6
    assert selection.best_score == pytest.approx(0.61)
    assert selection.stopped_early
    assert selection.stop_update == 6
    assert selection.observed_checkpoint_count == 4
    assert selection == type(selection).from_dict(selection.to_dict())

    tampered = ValidationCheckpoint(0, 0.5).to_dict()
    tampered.pop("content_checksum")
    tampered["data_role"] = "training"
    with pytest.raises(ValueError, match="validation"):
        ValidationCheckpoint.from_dict(seal_mapping(tampered))


def test_validation_tracker_is_immutable_resumable_and_terminal() -> None:
    """Persisted tracker state reproduces the stop decision and rejects later observations."""
    tracker = CheckpointValidationTracker(CheckpointValidationPolicy(patience=2))
    for checkpoint in (
        ValidationCheckpoint(0, 0.7),
        ValidationCheckpoint(2, 0.6),
        ValidationCheckpoint(4, 0.6),
    ):
        tracker = tracker.observe(checkpoint)

    restored = CheckpointValidationTracker.from_json(tracker.to_json())
    assert restored == tracker
    assert restored.should_stop
    assert restored.selection is not None
    assert restored.selection.best_update == 0
    with pytest.raises(ValueError, match="after validation"):
        restored.observe(ValidationCheckpoint(6, 0.9))


def test_phase_boundary_and_multistart_caps_are_explicit() -> None:
    """The phase transition is exact and multistart cannot exceed either cap."""
    boundary = NoiselessPretrainNoisyFinetune(3, 5)
    assert boundary.phase_at(2) == "noiseless_pretrain"
    assert boundary.phase_at(3) == "noisy_finetune"
    assert boundary.total_updates == 8
    assert NoiselessPretrainNoisyFinetune.from_dict(boundary.to_dict()) == boundary
    assert NoiselessPretrainNoisyFinetune(0, 5).mode == "noisy_only"
    assert NoiselessPretrainNoisyFinetune(5, 0).mode == "noiseless_only"
    with pytest.raises(ValueError, match="At least one"):
        NoiselessPretrainNoisyFinetune(0, 0)
    with pytest.raises(ValueError, match="outside"):
        boundary.phase_at(8)
    with pytest.raises(ValueError, match="declared_cap"):
        LimitedMultistartPlan(3, 2)
    with pytest.raises(ValueError, match="maximum"):
        LimitedMultistartPlan(1, MAX_MULTISTART_COUNT + 1)


def test_standard_noise_mixture_preserves_exact_order_and_weights() -> None:
    """Mixture order and largest-remainder allocation are sealed identity."""
    mixture = StandardNoiseMixture(
        "frozen_mixture",
        (
            NoiseMixtureComponent("depolarizing_1s_all", 0.5),
            NoiseMixtureComponent("dephasing_1s_all", 0.5),
        ),
    )
    reversed_mixture = StandardNoiseMixture("frozen_mixture", tuple(reversed(mixture.components)))
    allocation = mixture.allocate(9)

    assert StandardNoiseMixture.from_dict(mixture.to_dict()) == mixture
    assert mixture.content_checksum != reversed_mixture.content_checksum
    assert allocation.component_ids == ("depolarizing_1s_all", "dephasing_1s_all")
    assert allocation.component_counts == (5, 4)
    assert NoiseMixtureAllocation.from_dict(allocation.to_dict()) == allocation
    with pytest.raises(ValueError, match="at least one"):
        mixture.allocate(1)
    with pytest.raises(ValueError, match="sum exactly"):
        StandardNoiseMixture(
            "frozen_mixture",
            (
                NoiseMixtureComponent("dephasing_1s_all", 0.2),
                NoiseMixtureComponent("depolarizing_1s_all", 0.7),
            ),
        )
    with pytest.raises(ValueError, match="standard"):
        NoiseMixtureComponent("ballarin_coupled", 1.0)


def test_complete_schedule_is_strict_sealed_and_resume_deterministic() -> None:
    """Top-level identity binds every component and reconstructs an uninterrupted rolling stream."""
    policy = TrajectorySamplingPolicy("rolling_ensemble", refresh_interval=2, retain_fraction=0.5)
    schedule = _schedule(policy=policy)
    restored = TrainingStrategySchedule.from_json(schedule.to_json())

    assert restored == schedule
    assert restored.content_checksum == schedule.content_checksum
    at_five = restored.trajectory_membership_at(
        master_seed=101,
        role="training_trajectory",
        update=5,
    )
    repeated = schedule.trajectory_membership_at(
        master_seed=101,
        role="training_trajectory",
        update=5,
    )
    assert at_five == repeated
    assert at_five.trajectory_count == 8

    tampered = schedule.to_dict()
    raw_nested = tampered["noise_continuation"]
    assert isinstance(raw_nested, dict)
    nested = dict(raw_nested)
    nested["target_strength_scale"] = 0.5
    tampered["noise_continuation"] = nested
    tampered.pop("content_checksum")
    with pytest.raises(ValueError, match="checksum mismatch"):
        TrainingStrategySchedule.from_dict(seal_mapping(tampered))


def test_direct_noisy_and_noiseless_controls_are_representable() -> None:
    """Optional schedule components preserve honest direct-control identities."""
    noisy = replace(
        _schedule(),
        schedule_id="direct_noisy",
        noise_continuation=NoiseStrengthContinuation(
            0,
            4,
            1.0,
            start_strength_scale=1.0,
            interpolation="constant",
        ),
        phase_boundary=NoiselessPretrainNoisyFinetune(0, 5),
    )
    noiseless = replace(
        _schedule(),
        schedule_id="direct_noiseless",
        noise_continuation=NoiseStrengthContinuation(
            0,
            4,
            0.0,
            start_strength_scale=0.0,
            interpolation="constant",
        ),
        trajectory_curriculum=TrajectoryCountCurriculum((TrajectoryCountStep(0, 0),)),
        phase_boundary=NoiselessPretrainNoisyFinetune(5, 0),
        training_noise=StandardNoiseMixture("noiseless", ()),
    )

    assert noisy.noise_continuation.strength_at(0) == pytest.approx(1.0, abs=0.0)
    assert noisy.phase_boundary.mode == "noisy_only"
    assert noiseless.noise_continuation.strength_at(4) == pytest.approx(0.0, abs=0.0)
    assert noiseless.phase_boundary.mode == "noiseless_only"
    assert TrainingStrategySchedule.from_json(noisy.to_json()) == noisy
    assert TrainingStrategySchedule.from_json(noiseless.to_json()) == noiseless


def test_frozen_production_continuation_and_curriculum_have_exact_boundaries() -> None:
    """The prospective 200-update continuation reaches a sealed plateau at update 49."""
    schedule = TrainingStrategySchedule(
        schedule_id="wp22_production_continuation",
        noise_continuation=NoiseStrengthContinuation(0, 49, 1.0),
        trajectory_curriculum=TrajectoryCountCurriculum((
            TrajectoryCountStep(0, 2),
            TrajectoryCountStep(50, 4),
            TrajectoryCountStep(100, 8),
        )),
        sampling_policy=TrajectorySamplingPolicy(
            "rolling_ensemble",
            refresh_interval=20,
            retain_fraction=0.5,
        ),
        checkpoint_validation=CheckpointValidationPolicy(patience=None),
        phase_boundary=NoiselessPretrainNoisyFinetune(0, 200),
        multistart=LimitedMultistartPlan(1, 3),
        training_noise=StandardNoiseMixture(
            "matched",
            (NoiseMixtureComponent("depolarizing_1s_all", 1.0),),
        ),
    )

    assert [schedule.noise_continuation.strength_at(index) for index in (0, 49, 50, 199)] == [
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    assert [schedule.trajectory_curriculum.count_at(index) for index in (0, 49, 50, 99, 100, 199)] == [
        2,
        2,
        4,
        4,
        8,
        8,
    ]
    assert schedule.sampling_policy.epoch_at(19) == 0
    assert schedule.sampling_policy.epoch_at(20) == 1
    assert TrainingStrategySchedule.from_json(schedule.to_json()) == schedule


def test_frozen_training_policy_universe_roots_every_reviewed_schedule_choice() -> None:
    """All exact direct, exploratory, mixture, multistart, and control policies are sealed together."""
    universe = FrozenTrainingPolicyUniverse.frozen()
    by_id = {schedule.schedule_id: schedule for schedule in universe.schedules}

    assert tuple(by_id) == (
        "direct_matched_fixed_crn",
        "continuation_fixed_crn",
        "curriculum_fixed_crn",
        "periodic_refresh_20",
        "rolling_half_refresh_20",
        "resampled_each_update",
        "frozen_half_depolarizing_half_dephasing",
        "limited_multistart_3",
        "direct_noiseless_control",
    )
    assert by_id["continuation_fixed_crn"].noise_continuation.end_update == 49
    assert by_id["periodic_refresh_20"].sampling_policy.refresh_interval == 20
    assert by_id["rolling_half_refresh_20"].sampling_policy.retain_fraction == pytest.approx(0.5, abs=0.0)
    assert by_id["limited_multistart_3"].multistart.start_count == 3
    mixture = by_id["frozen_half_depolarizing_half_dephasing"].training_noise.allocate(8)
    assert mixture.component_counts == (4, 4)
    assert mixture.component_seed_domains == (
        "training_trajectory.mixture.depolarizing_1s_all",
        "training_trajectory.mixture.dephasing_1s_all",
    )
    assert FrozenTrainingPolicyUniverse.from_json(universe.to_json()) == universe

    multistart_payload = by_id["limited_multistart_3"].multistart.to_dict()
    assert multistart_payload["tie_rules"] == ["earliest_checkpoint", "lowest_start_index"]
    assert multistart_payload["work_accounting"] == "all_starts"


def test_schedule_membership_skips_a_zero_count_noiseless_prefix() -> None:
    """The first noisy ensemble starts explicitly after unsampled noiseless pretraining."""
    schedule = replace(
        _schedule(),
        trajectory_curriculum=TrajectoryCountCurriculum((
            TrajectoryCountStep(0, 0),
            TrajectoryCountStep(3, 4),
            TrajectoryCountStep(4, 8),
        )),
    )

    first = schedule.trajectory_membership_at(master_seed=4, role="training_trajectory", update=3)
    later = schedule.trajectory_membership_at(master_seed=4, role="training_trajectory", update=5)
    assert first.predecessor_checksum is None
    assert later.trajectory_count == 8
    with pytest.raises(ValueError, match="unavailable"):
        schedule.trajectory_membership_at(master_seed=4, role="training_trajectory", update=2)


def test_complete_schedule_allows_a_plateau_and_rejects_boundary_mismatches() -> None:
    """Continuation may plateau after its ramp but must start at the noisy boundary."""
    schedule = _schedule()
    plateau = replace(schedule, noise_continuation=NoiseStrengthContinuation(3, 4, 1.0))
    assert plateau.noise_continuation.strength_at(7) == pytest.approx(1.0, abs=0.0)
    with pytest.raises(ValueError, match="Two-phase training"):
        replace(schedule, noise_continuation=NoiseStrengthContinuation(2, 7, 1.0))
    with pytest.raises(ValueError, match="within the training budget"):
        replace(schedule, noise_continuation=NoiseStrengthContinuation(3, 8, 1.0))
    with pytest.raises(ValueError, match="positive trajectory"):
        replace(
            schedule,
            trajectory_curriculum=TrajectoryCountCurriculum((TrajectoryCountStep(0, 0), TrajectoryCountStep(4, 8))),
        )


def test_strict_schema_rejects_unknown_top_level_fields() -> None:
    """Even a correctly resealed document cannot introduce unversioned schedule fields."""
    data = _schedule().to_dict()
    data.pop("content_checksum")
    data["unsupported"] = True
    with pytest.raises(ValueError, match="fields do not match"):
        TrainingStrategySchedule.from_dict(seal_mapping(data))
