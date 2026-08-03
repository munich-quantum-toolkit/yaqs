# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the prospective checksum-sealed WP22A execution protocol."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Protocol, cast

import pytest

from benchmarks.state_preparation import phase2 as phase2_api
from benchmarks.state_preparation.phase2.canonical import seal_mapping
from benchmarks.state_preparation.phase2.execution_protocol import (
    CHECKPOINT_VALIDATION_UPDATES,
    DEFAULT_OPERATIONAL_PROTOCOL_AMENDMENT_PATH,
    PILOT_METHOD_IDS,
    PRIMARY_Q6_PILOT_PREFIXES,
    SCREEN_METHOD_IDS,
    TRUSTED_OPERATIONAL_PROTOCOL_AMENDMENT_CHECKSUM,
    WP22_IMPLEMENTATION_PLAN_COMMIT,
    FreshEvaluationPolicy,
    OperationalProtocolAmendment,
    OperatorGrowthExecutionSpec,
    PilotDiagnosticPolicy,
    bounded_outer_trajectory_count,
    load_operational_protocol_amendment,
)
from benchmarks.state_preparation.phase2.operator_growth import OperatorGrowthSpec
from benchmarks.state_preparation.phase2.protocol import TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM
from benchmarks.state_preparation.phase2.training_schedules import (
    PILOT_DIAGNOSTIC_SEED_POLICY_ID,
    PILOT_FRESH_EVALUATION_SEED_POLICY_ID,
    PILOT_OPTIMIZATION_SEED_POLICY_ID,
    SCHEDULE_SEED_DERIVATION_POLICY_ID,
    SCREEN_OPTIMIZATION_SEED_POLICY_ID,
    STAGE_SEED_DERIVATION_POLICY_ID,
    ExecutionSeedPolicySuite,
)


def _reseal(document: Mapping[str, object]) -> dict[str, object]:
    """Return a detached document with a checksum over its current payload."""
    return seal_mapping({key: value for key, value in document.items() if key != "content_checksum"})


def _walk_mappings(value: object) -> tuple[Mapping[str, object], ...]:
    """Return every mapping nested in a JSON-native value."""
    if isinstance(value, Mapping):
        descendants = tuple(child for item in value.values() for child in _walk_mappings(item))
        return (cast("Mapping[str, object]", value), *descendants)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(child for item in value for child in _walk_mappings(item))
    return ()


def _assert_exact_float(value: object, expected: float) -> None:
    """Require an exact built-in float with the expected binary value."""
    assert type(value) is float
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=0.0)


class _SealedJsonArtifact(Protocol):
    """Common test-facing interface of checksum-sealed JSON artifacts."""

    @property
    def content_checksum(self) -> str:
        """Artifact checksum."""

    def to_json(self) -> str:
        """Return canonical JSON."""


@pytest.mark.parametrize(
    ("artifact", "decoder"),
    [
        (FreshEvaluationPolicy.primary_q6_pilot(), FreshEvaluationPolicy.from_json),
        (FreshEvaluationPolicy.secondary_q12_pilot(), FreshEvaluationPolicy.from_json),
        (FreshEvaluationPolicy.checkpoint_validation(), FreshEvaluationPolicy.from_json),
        (FreshEvaluationPolicy.checkpoint_validation("secondary_q12"), FreshEvaluationPolicy.from_json),
        (FreshEvaluationPolicy.smoke(2), FreshEvaluationPolicy.from_json),
        (FreshEvaluationPolicy.screening(1024), FreshEvaluationPolicy.from_json),
        (PilotDiagnosticPolicy.primary_q6(), PilotDiagnosticPolicy.from_json),
        (PilotDiagnosticPolicy.secondary_q12(), PilotDiagnosticPolicy.from_json),
        (OperatorGrowthExecutionSpec.for_screening(1024), OperatorGrowthExecutionSpec.from_json),
        (OperatorGrowthExecutionSpec.for_smoke(2), OperatorGrowthExecutionSpec.from_json),
        (OperationalProtocolAmendment.frozen(), OperationalProtocolAmendment.from_json),
    ],
)
def test_wp22a_policy_round_trips_are_strict_and_canonical(
    artifact: _SealedJsonArtifact,
    decoder: Callable[[str], _SealedJsonArtifact],
) -> None:
    """Every WP22A policy round-trips without changing bytes or checksum."""
    restored = decoder(artifact.to_json())
    assert restored == artifact
    assert restored.to_json() == artifact.to_json()
    assert restored.content_checksum == artifact.content_checksum


def test_wp22a_q6_and_q12_pilot_roles_counts_and_diagnostics_are_exact() -> None:
    """The q12 scaling collection cannot acquire primary roles or inference diagnostics."""
    q6 = FreshEvaluationPolicy.primary_q6_pilot()
    q12 = FreshEvaluationPolicy.secondary_q12_pilot()
    q6_diagnostic = PilotDiagnosticPolicy.primary_q6()
    q12_diagnostic = PilotDiagnosticPolicy.secondary_q12()

    assert (q6.target_scope, q6.qubit_count, q6.data_role, q6.seed_domain) == (
        "primary_q6",
        6,
        "development",
        "pilot_evaluation",
    )
    assert q6.trajectory_count == 1024
    assert q6.reporting_prefixes == (64, 128, 256, 512, 1024)
    assert (q12.target_scope, q12.qubit_count, q12.data_role, q12.seed_domain) == (
        "secondary_q12",
        12,
        "secondary_benchmark",
        "pilot_evaluation",
    )
    assert q12.trajectory_count == 256
    assert q12.reporting_prefixes == (256,)
    q12_checkpoint = FreshEvaluationPolicy.checkpoint_validation("secondary_q12")
    assert (q12_checkpoint.target_scope, q12_checkpoint.qubit_count) == ("secondary_q12", 12)
    assert (q12_checkpoint.data_role, q12_checkpoint.seed_domain) == (
        "checkpoint_validation",
        "checkpoint_validation",
    )
    assert q12_checkpoint.trajectory_count == 256
    assert q6_diagnostic.enabled
    assert q6_diagnostic.trajectory_count == 32
    assert q6_diagnostic.endpoint == "post_training_primary_noise_pathwise_update_variance"
    assert q6_diagnostic.summary_statistics == ("arithmetic_mean", "maximum")
    assert not q6_diagnostic.promotion_eligible
    assert not q12_diagnostic.enabled
    assert q12_diagnostic.trajectory_count == 0
    assert q12_diagnostic.endpoint is None
    assert q12_diagnostic.summary_statistics == ()
    assert q6.seed_derivation_policy_id == PILOT_FRESH_EVALUATION_SEED_POLICY_ID
    assert q6_diagnostic.seed_derivation_policy_id == PILOT_DIAGNOSTIC_SEED_POLICY_ID
    assert q12_diagnostic.seed_derivation_policy_id is None
    assert q12_diagnostic.seed_derivation_policy_checksum is None

    with pytest.raises(ValueError, match="purpose is not allowed"):
        FreshEvaluationPolicy(
            policy_id="invalid_q12_screen",
            purpose="screening_outer",
            target_scope="secondary_q12",
            qubit_count=12,
            data_role="screening_selection",
            seed_domain="screening_selection",
            trajectory_count=256,
            reporting_prefixes=(256,),
        )

    with pytest.raises(ValueError, match="forbids trajectory optional stopping"):
        replace(FreshEvaluationPolicy.smoke(2), trajectory_optional_stopping=True)


def test_wp22a_fresh_policy_rejects_missing_count_and_resealed_substitutions() -> None:
    """Fixed counts and nested provider identity survive hostile outer resealing."""
    missing_count = FreshEvaluationPolicy.primary_q6_pilot().to_dict()
    missing_count["trajectory_count"] = None
    with pytest.raises(TypeError, match="trajectory_count"):
        FreshEvaluationPolicy.from_dict(_reseal(missing_count))

    changed_count = FreshEvaluationPolicy.primary_q6_pilot().to_dict()
    changed_count["trajectory_count"] = 512
    changed_count["reporting_prefixes"] = [64, 128, 256, 512]
    with pytest.raises(ValueError, match="differ from WP22A"):
        FreshEvaluationPolicy.from_dict(_reseal(changed_count))

    changed_provider = FreshEvaluationPolicy.primary_q6_pilot().to_dict()
    raw_provider = changed_provider["provider_identity"]
    assert isinstance(raw_provider, Mapping)
    provider = dict(cast("Mapping[str, object]", raw_provider))
    provider["strength_scale"] = 0.5
    changed_provider["provider_identity"] = provider
    with pytest.raises(ValueError, match="provider_identity"):
        FreshEvaluationPolicy.from_dict(_reseal(changed_provider))

    extra_field = FreshEvaluationPolicy.primary_q6_pilot().to_dict()
    extra_field["target_vector"] = [1.0]
    with pytest.raises(ValueError, match="fields do not match"):
        FreshEvaluationPolicy.from_dict(_reseal(extra_field))

    changed_seed_policy = FreshEvaluationPolicy.primary_q6_pilot().to_dict()
    changed_seed_policy["seed_derivation_policy_id"] = STAGE_SEED_DERIVATION_POLICY_ID
    with pytest.raises(ValueError, match="seed policy reference changed"):
        FreshEvaluationPolicy.from_dict(_reseal(changed_seed_policy))


def test_wp22a_operational_amendment_freezes_reviewed_population_and_budget() -> None:
    """The addendum exactly reproduces the reviewed pilot and screen cross products."""
    amendment = OperationalProtocolAmendment.frozen()

    assert amendment.preregistration_checksum == TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM
    assert amendment.implementation_plan_commit == WP22_IMPLEMENTATION_PLAN_COMMIT
    assert amendment.pilot_method_ids == PILOT_METHOD_IDS
    assert len(PILOT_METHOD_IDS) == 3
    assert amendment.screen_method_ids == SCREEN_METHOD_IDS
    assert len(SCREEN_METHOD_IDS) == 9
    assert amendment.production_update_count == 200
    assert amendment.production_terminal_update == 199
    assert amendment.training_trajectory_count == 8
    assert amendment.checkpoint_validation_trajectory_count == 256
    assert amendment.checkpoint_validation_cadence == 10
    assert amendment.checkpoint_validation_updates == (*range(0, 199, 10), 199)
    assert amendment.checkpoint_validation_updates == CHECKPOINT_VALIDATION_UPDATES
    assert amendment.primary_q6_pilot_job_count == 48 * 5 * 3 == 720
    assert amendment.secondary_q12_pilot_job_count == 24 * 5 * 3 == 360
    assert amendment.screen_cell_count == 48 * 3 * 9 == 1296
    assert amendment.pilot_optimization_seed_policy_id == PILOT_OPTIMIZATION_SEED_POLICY_ID
    assert amendment.screen_optimization_seed_policy_id == SCREEN_OPTIMIZATION_SEED_POLICY_ID
    assert amendment.execution_seed_policy_suite == ExecutionSeedPolicySuite.frozen()
    assert amendment.execution_seed_policy_suite_checksum == amendment.execution_seed_policy_suite.content_checksum
    assert not amendment.screen_adaptive
    assert not amendment.q12_inference_eligible
    assert not amendment.q12_screening_eligible
    assert not amendment.q12_promotion_eligible
    assert not amendment.outer_trajectory_optional_stopping
    assert amendment.primary_q6_fresh_evaluation_policy.reporting_prefixes == PRIMARY_Q6_PILOT_PREFIXES


def test_wp22a_operational_amendment_has_an_independent_checked_in_trust_root() -> None:
    """The reviewed JSON and literal checksum prevent code-only redefinition of schema v1."""
    amendment = load_operational_protocol_amendment()

    assert amendment.content_checksum == TRUSTED_OPERATIONAL_PROTOCOL_AMENDMENT_CHECKSUM
    assert DEFAULT_OPERATIONAL_PROTOCOL_AMENDMENT_PATH.read_text(encoding="utf-8").strip() == amendment.to_json()


def test_wp22a_operational_amendment_rejects_resealed_nested_substitution() -> None:
    """A valid policy from another scope cannot be substituted and outer-resealed."""
    document = OperationalProtocolAmendment.frozen().to_dict()
    document["secondary_q12_fresh_evaluation_policy"] = FreshEvaluationPolicy.primary_q6_pilot().to_dict()
    with pytest.raises(ValueError, match="differs from the reviewed"):
        OperationalProtocolAmendment.from_dict(_reseal(document))

    changed_seed_binding = OperationalProtocolAmendment.frozen().to_dict()
    changed_seed_binding["pilot_optimization_seed_policy_id"] = SCREEN_OPTIMIZATION_SEED_POLICY_ID
    with pytest.raises(ValueError, match="differs from the reviewed"):
        OperationalProtocolAmendment.from_dict(_reseal(changed_seed_binding))


def test_wp22a_bounded_outer_count_matches_the_frozen_theorem_10_rule() -> None:
    """The union-bound rule produces the exact power-of-two endpoints without optional stopping."""
    zero_variances = [0.0] * 720
    radius = math.sqrt(2.0 * math.log(720 / 0.05) / 1023)
    variance_upper = min(0.25, radius**2)
    required = max(256, math.ceil(variance_upper / 0.005**2))
    expected = 1 << (required - 1).bit_length()

    assert expected == 1024
    assert bounded_outer_trajectory_count(zero_variances) == expected
    assert bounded_outer_trajectory_count([0.25] * 720) == 16384
    with pytest.raises(ValueError, match="exactly 720"):
        bounded_outer_trajectory_count([0.0] * 719)
    with pytest.raises(TypeError, match=r"unbiased_variances\[0\]"):
        bounded_outer_trajectory_count(cast("list[float]", [0] * 720))


def test_wp22a_operator_growth_spec_is_complete_and_rejects_nested_replacement() -> None:
    """The wrapper seals the full q6 WP20 core, noise, validation, work, and outer policy."""
    spec = OperatorGrowthExecutionSpec.for_screening(1024)

    assert spec.target_scope == "primary_q6"
    assert spec.qubit_count == 6
    assert spec.pool.one_qubit_generators == ("x", "y", "z")
    assert spec.pool.two_qubit_generators == ("xx", "yy", "zz")
    assert spec.pool.selection_reuse_policy == "without_replacement"
    assert len(spec.pool.operators) == 33
    _assert_exact_float(spec.growth_spec.gradient_tolerance, 1e-10)
    assert spec.growth_spec.max_operators == 16
    assert spec.growth_spec.native_two_qubit_cap_per_edge == 12
    assert spec.growth_spec.reoptimization_steps == 100
    _assert_exact_float(spec.growth_spec.learning_rate, 0.08)
    _assert_exact_float(spec.growth_spec.adam_beta1, 0.9)
    _assert_exact_float(spec.growth_spec.adam_beta2, 0.999)
    _assert_exact_float(spec.growth_spec.adam_epsilon, 1e-8)
    assert spec.training_trajectory_count == 8
    assert spec.training_sampling_policy.kind == "fixed_crn"
    assert spec.training_seed_derivation_policy_id == STAGE_SEED_DERIVATION_POLICY_ID
    assert spec.trajectory_member_seed_policy_id == SCHEDULE_SEED_DERIVATION_POLICY_ID
    assert spec.checkpoint_validation_policy.trajectory_count == 256
    assert spec.outer_evaluation_policy.trajectory_count == 1024
    _assert_exact_float(spec.resource_policy["cap_per_chain_edge"], 12.0)

    smoke = OperatorGrowthExecutionSpec.for_smoke(2)
    assert smoke.outer_evaluation_policy.purpose == "smoke_evaluation"
    assert smoke.outer_evaluation_policy.data_role == "development"
    assert smoke.outer_evaluation_policy.trajectory_count == 2
    assert smoke.growth_spec == spec.growth_spec

    replacement = OperatorGrowthSpec.for_pool(
        spec.pool,
        gradient_tolerance=1e-10,
        max_operators=15,
        native_two_qubit_cap_per_edge=12,
        reoptimization_steps=100,
        learning_rate=0.08,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    )
    document = spec.to_dict()
    document["growth_spec"] = replacement.to_dict()
    with pytest.raises(ValueError, match="growth_spec differs"):
        OperatorGrowthExecutionSpec.from_dict(_reseal(document))


def test_wp22a_policy_artifacts_never_materialize_targets_or_secret_entropy() -> None:
    """Policy records contain target scopes and checksums but no target values or role secrets."""
    artifacts = (
        OperationalProtocolAmendment.frozen().to_dict(),
        OperatorGrowthExecutionSpec.for_screening(1024).to_dict(),
    )
    forbidden_keys = {
        "target_instance_id",
        "target_vector",
        "target_vector_checksum",
        "target_seed",
        "role_master_entropy",
        "master_entropy",
        "materialized_target",
    }
    for artifact in artifacts:
        for mapping in _walk_mappings(artifact):
            assert forbidden_keys.isdisjoint(mapping)


def test_wp22a_records_are_available_from_the_phase2_public_api() -> None:
    """Later packages can import the reviewed WP22A records from one stable API."""
    expected = {
        "CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID",
        "CONFIRMATORY_OPTIMIZATION_SEED_POLICY_ID",
        "EXECUTION_SEED_POLICY_IDS",
        "EXECUTION_SEED_POLICY_SUITE_SCHEMA_VERSION",
        "OPERATOR_GROWTH_SMOKE_SPEC_SCHEMA_VERSION",
        "PILOT_DIAGNOSTIC_SEED_POLICY_ID",
        "PILOT_FRESH_EVALUATION_SEED_POLICY_ID",
        "PILOT_OPTIMIZATION_SEED_POLICY_ID",
        "PIPELINE_SMOKE_SPEC_SCHEMA_VERSION",
        "SCHEDULE_SEED_DERIVATION_POLICY_ID",
        "SCREENING_CELL_SEED_POLICY_ID",
        "SCREENING_ROOT_SEED_POLICY_ID",
        "SCREEN_OPTIMIZATION_SEED_POLICY_ID",
        "SEED_DERIVATION_POLICY_SCHEMA_VERSION",
        "SMOKE_EXECUTION_LIMITS_SCHEMA_VERSION",
        "SMOKE_FRESH_EVALUATION_SEED_POLICY_ID",
        "SMOKE_OPTIMIZATION_SEED_POLICY_ID",
        "STAGE_SEED_DERIVATION_POLICY_ID",
        "ExecutionSeedPolicySuite",
        "FreshEvaluationPolicy",
        "FrozenTrainingPolicyUniverse",
        "OperationalProtocolAmendment",
        "OperatorGrowthExecutionSpec",
        "OperatorGrowthSmokeSpec",
        "PipelineSmokeSpec",
        "ScopedImplementationBinding",
        "SeedDerivationPolicy",
        "SmokeExecutionLimits",
        "TrainingExecutionProfile",
        "TrainingStrategySchedule",
        "bounded_outer_trajectory_count",
        "load_operational_protocol_amendment",
    }

    assert expected <= set(phase2_api.__all__)
    assert all(getattr(phase2_api, name) is not None for name in expected)
