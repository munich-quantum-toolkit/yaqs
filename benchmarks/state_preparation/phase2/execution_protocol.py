# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Prospective WP22A execution policies without target materialization.

The records in this module close operational choices that were deliberately
left outside the immutable WP15 preregistration.  They describe only policies
and implementation-independent numerical constants: no target vector, target
seed, role-master entropy, or result is accepted by any constructor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from benchmarks.state_preparation.noise import create_scaled_standard_noise_provider
from mqt.yaqs.optimization import (
    KROTOV_FIXED_MAP_ENSEMBLE_IDENTITY_VERSION,
    KROTOV_FIXED_MAP_ENSEMBLE_SCHEMA_VERSION,
)

from .artifact_codecs import PHASE2_TRAJECTORY_SIDECAR_SCHEMA_VERSION
from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    read_canonical_json_object,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .operator_growth import (
    ADAPT_STYLE_METHOD_ID,
    PROJECTOR_COST_ID,
    OperatorGrowthSpec,
    OperatorPoolSpec,
    build_projector_operator_pool,
)
from .protocol import TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM
from .training_schedules import (
    CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID,
    PILOT_DIAGNOSTIC_SEED_POLICY_ID,
    PILOT_FRESH_EVALUATION_SEED_POLICY_ID,
    PILOT_OPTIMIZATION_SEED_POLICY_ID,
    SCHEDULE_SEED_DERIVATION_POLICY_ID,
    SCREEN_OPTIMIZATION_SEED_POLICY_ID,
    SCREENING_CELL_SEED_POLICY_ID,
    SMOKE_FRESH_EVALUATION_SEED_POLICY_ID,
    STAGE_SEED_DERIVATION_POLICY_ID,
    ExecutionSeedPolicySuite,
    FrozenTrainingPolicyUniverse,
    TrajectorySamplingPolicy,
)
from .validation import (
    require_bool,
    require_float,
    require_int,
    require_slug,
)
from .wp20_resources import NormalizedComputePolicy

FRESH_EVALUATION_POLICY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_fresh_evaluation_policy.v1"
PILOT_DIAGNOSTIC_POLICY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_pilot_diagnostic_policy.v1"
OPERATOR_GROWTH_EXECUTION_SPEC_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_operator_growth_execution_spec.v1"
OPERATIONAL_PROTOCOL_AMENDMENT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_operational_protocol_amendment.v1"

WP22_IMPLEMENTATION_PLAN_COMMIT = "93fae0bb1cd4a2af12a7ac11e3383a1180bd4f3e"
DEFAULT_OPERATIONAL_PROTOCOL_AMENDMENT_PATH = (
    Path(__file__).with_name("data") / "operational_protocol_amendment_v1.json"
)
TRUSTED_OPERATIONAL_PROTOCOL_AMENDMENT_CHECKSUM = (
    "sha256:49e7256f7191ba5ff3362e3dc93942e2c9fb9c7ba0423f8a8f14c2cafcf34899"
)

PRODUCTION_UPDATE_COUNT = 200
PRODUCTION_TERMINAL_UPDATE = 199
PRODUCTION_TRAINING_TRAJECTORY_COUNT = 8
CHECKPOINT_VALIDATION_TRAJECTORY_COUNT = 256
CHECKPOINT_VALIDATION_CADENCE = 10
CHECKPOINT_VALIDATION_UPDATES = (*range(0, PRODUCTION_TERMINAL_UPDATE, CHECKPOINT_VALIDATION_CADENCE), 199)

PRIMARY_Q6_PILOT_TARGET_COUNT = 48
SECONDARY_Q12_PILOT_TARGET_COUNT = 24
PILOT_OPTIMIZATION_SEED_COUNT = 5
PILOT_CONFIGURATION_COUNT = 3
PRIMARY_Q6_PILOT_JOB_COUNT = 720
SECONDARY_Q12_PILOT_JOB_COUNT = 360
PRIMARY_Q6_PILOT_TRAJECTORY_COUNT = 1024
SECONDARY_Q12_PILOT_TRAJECTORY_COUNT = 256
PRIMARY_Q6_PILOT_PREFIXES = (64, 128, 256, 512, 1024)
PRIMARY_Q6_DIAGNOSTIC_VECTOR_COUNT = 32

SCREEN_METHOD_COUNT = 9
SCREEN_TARGET_COUNT = 48
SCREEN_OPTIMIZATION_SEED_COUNT = 3
SCREEN_CELL_COUNT = 1296

OUTER_VARIANCE_PLANNED_JOB_COUNT = PRIMARY_Q6_PILOT_JOB_COUNT
OUTER_VARIANCE_ALPHA = 0.05
OUTER_TRAJECTORY_MCSE_TARGET = 0.005
OUTER_TRAJECTORY_COUNT_MIN = 256
OUTER_TRAJECTORY_COUNT_MAX = 16384

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

_PRIMARY_NOISE_CONDITION = {
    "noise_id": "depolarizing_1s_all",
    "definition_version": "yaqs.state_preparation.noise.v1",
    "strength_scale": 1.0,
    "tjm_dt": 1.0,
    "training_placement": "logical_parameterized_gates",
    "test_placement": "logical_parameterized_gates",
}
_PRIMARY_RESOURCE_POLICY = {
    "metric": "native_two_qubit_gates_per_chain_edge",
    "cap_per_chain_edge": 12.0,
    "comparison_rule": "largest_reachable_at_or_below_cap",
    "compiler_policy_id": "quantinuum_rzz_chain_v1",
    "connectivity": "linear_chain",
    "routing_policy": "identity_no_swap",
    "residual_gap_reporting": True,
    "normalized_compute_cap_source": "pilot_final_seal",
}
_TRUNCATION_POLICY = {
    "max_bond_dimension": None,
    "svd_threshold": 0.0,
    "truncation_mode": "discarded_weight",
    "min_bond_dimension": 1,
}
_FRESH_EVALUATION_KEYS = frozenset({
    "schema_version",
    "policy_id",
    "purpose",
    "target_scope",
    "qubit_count",
    "data_role",
    "seed_domain",
    "noise_condition",
    "provider_identity",
    "trajectory_count",
    "reporting_prefixes",
    "seed_derivation_policy_id",
    "seed_derivation_policy_checksum",
    "map_schema_version",
    "map_identity_version",
    "truncation_policy",
    "worker_count",
    "sidecar_schema_version",
    "evaluation_policy",
    "trajectory_optional_stopping",
    "failure_treatment",
    "content_checksum",
})
_PILOT_DIAGNOSTIC_KEYS = frozenset({
    "schema_version",
    "policy_id",
    "target_scope",
    "qubit_count",
    "data_role",
    "enabled",
    "endpoint",
    "trajectory_count",
    "seed_domain",
    "seed_derivation_policy_id",
    "seed_derivation_policy_checksum",
    "noise_condition",
    "provider_identity",
    "map_schema_version",
    "checkpoint_rule",
    "estimator_id",
    "estimator_version",
    "parameter_ordering",
    "coordinate_variance_rule",
    "summary_statistics",
    "store_complete_vectors",
    "successful_jobs_only",
    "promotion_eligible",
    "content_checksum",
})
_OPERATOR_GROWTH_EXECUTION_KEYS = frozenset({
    "schema_version",
    "target_scope",
    "qubit_count",
    "method_id",
    "objective_id",
    "pool",
    "growth_spec",
    "training_noise_condition",
    "provider_identity",
    "training_trajectory_count",
    "training_sampling_policy",
    "training_seed_derivation_policy_id",
    "training_seed_derivation_policy_checksum",
    "trajectory_member_seed_policy_id",
    "trajectory_member_seed_policy_checksum",
    "training_ensemble_rule",
    "checkpoint_validation_policy",
    "outer_evaluation_policy",
    "checkpoint_selection_rule",
    "validation_after_each_completed_prefix",
    "resource_policy",
    "normalized_compute_policy",
    "supported_schedule_mode",
    "explicitly_unsupported_schedule_modes",
    "content_checksum",
})
_OPERATIONAL_AMENDMENT_KEYS = frozenset({
    "schema_version",
    "amendment_id",
    "preregistration_checksum",
    "implementation_plan_commit",
    "prospective_status",
    "subpackage_order",
    "pilot_method_ids",
    "screen_method_ids",
    "production_update_count",
    "production_terminal_update",
    "training_trajectory_count",
    "checkpoint_validation_trajectory_count",
    "checkpoint_validation_cadence",
    "checkpoint_validation_updates",
    "checkpoint_selection_tie_rule",
    "primary_methods_stop_early",
    "optimizer_state_preserved_across_schedule_boundaries",
    "unsupported_composition_action",
    "resume_identity_requirement",
    "primary_q6_pilot_target_count",
    "secondary_q12_pilot_target_count",
    "pilot_optimization_seed_count",
    "pilot_optimization_seed_policy_id",
    "pilot_optimization_seed_policy_checksum",
    "pilot_configuration_count",
    "primary_q6_pilot_job_count",
    "secondary_q12_pilot_job_count",
    "primary_q6_fresh_evaluation_policy",
    "secondary_q12_fresh_evaluation_policy",
    "primary_q6_diagnostic_policy",
    "secondary_q12_diagnostic_policy",
    "execution_seed_policy_suite",
    "execution_seed_policy_suite_checksum",
    "training_policy_universe",
    "q12_inference_eligible",
    "q12_screening_eligible",
    "q12_promotion_eligible",
    "screen_target_count",
    "screen_optimization_seed_count",
    "screen_optimization_seed_policy_id",
    "screen_optimization_seed_policy_checksum",
    "screen_method_count",
    "screen_cell_count",
    "screen_adaptive",
    "outer_variance_method",
    "outer_variance_planned_job_count",
    "outer_variance_alpha",
    "outer_trajectory_mcse_target",
    "outer_trajectory_count_min",
    "outer_trajectory_count_max",
    "outer_trajectory_optional_stopping",
    "content_checksum",
})

FreshPurpose = Literal[
    "pilot_fresh_evaluation",
    "checkpoint_validation",
    "screening_outer",
    "confirmatory_fresh_evaluation",
    "smoke_evaluation",
]
TargetScope = Literal["primary_q6", "secondary_q12"]
EvaluationRole = Literal[
    "development",
    "checkpoint_validation",
    "screening_selection",
    "secondary_benchmark",
    "confirmatory",
]


def _primary_noise_condition() -> Mapping[str, object]:
    """Return an immutable detached copy of the preregistered noise condition."""
    return freeze_json_mapping(_PRIMARY_NOISE_CONDITION, "primary noise condition")


def _primary_resource_policy() -> Mapping[str, object]:
    """Return an immutable detached copy of the preregistered resource policy."""
    return freeze_json_mapping(_PRIMARY_RESOURCE_POLICY, "primary resource policy")


def _truncation_policy() -> Mapping[str, object]:
    """Return the exact no-truncation tensor-network policy."""
    return freeze_json_mapping(_TRUNCATION_POLICY, "truncation policy")


def _provider_identity() -> Mapping[str, object]:
    """Return the exact provider identity and its implementation checksum."""
    provider = create_scaled_standard_noise_provider("depolarizing_1s_all", 1.0)
    return freeze_json_mapping(
        {**provider.to_dict(), "content_checksum": provider.content_checksum},
        "provider identity",
    )


def _require_string_tuple(value: object, name: str) -> tuple[str, ...]:
    """Return a strict ordered tuple of unique nonempty strings.

    Raises:
        TypeError: If ``value`` is not a tuple.
        ValueError: If an item is empty or duplicated.
    """
    if type(value) is not tuple:
        msg = f"{name} must be a tuple."
        raise TypeError(msg)
    result = tuple(require_slug(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        msg = f"{name} must not contain duplicates."
        raise ValueError(msg)
    return result


def _require_int_tuple(value: object, name: str) -> tuple[int, ...]:
    """Return a strict ordered tuple of nonnegative integers.

    Raises:
        TypeError: If ``value`` is not a tuple.
    """
    if type(value) is not tuple:
        msg = f"{name} must be a tuple."
        raise TypeError(msg)
    return tuple(require_int(item, f"{name}[{index}]") for index, item in enumerate(value))


def _same_mapping(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Compare frozen JSON mappings with exact scalar types.

    Returns:
        Whether the mappings have identical canonical JSON representations.
    """
    return canonical_json(left) == canonical_json(right)


@dataclass(frozen=True, slots=True)
class FreshEvaluationPolicy:
    """Complete target-independent policy for one fixed fresh evaluation."""

    policy_id: str
    purpose: FreshPurpose
    target_scope: TargetScope
    qubit_count: int
    data_role: EvaluationRole
    seed_domain: str
    trajectory_count: int
    reporting_prefixes: tuple[int, ...]
    noise_condition: Mapping[str, object] = field(default_factory=_primary_noise_condition)
    provider_identity: Mapping[str, object] = field(default_factory=_provider_identity)
    map_schema_version: str = KROTOV_FIXED_MAP_ENSEMBLE_SCHEMA_VERSION
    map_identity_version: str = KROTOV_FIXED_MAP_ENSEMBLE_IDENTITY_VERSION
    truncation_policy: Mapping[str, object] = field(default_factory=_truncation_policy)
    worker_count: int = 1
    sidecar_schema_version: str = PHASE2_TRAJECTORY_SIDECAR_SCHEMA_VERSION
    evaluation_policy: str = "fixed_sample"
    trajectory_optional_stopping: bool = False
    failure_treatment: str = "structured_failure_zero_fidelity_for_intention_to_treat"
    schema_version: str = field(default=FRESH_EVALUATION_POLICY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate role separation, fixed counts, and complete runtime identity.

        Raises:
            ValueError: If any policy field differs from the frozen protocol.
        """
        object.__setattr__(self, "policy_id", require_slug(self.policy_id, "policy_id"))
        allowed_purposes = {
            "pilot_fresh_evaluation",
            "checkpoint_validation",
            "screening_outer",
            "confirmatory_fresh_evaluation",
            "smoke_evaluation",
        }
        if self.purpose not in allowed_purposes:
            msg = "purpose is not a supported fresh-evaluation purpose."
            raise ValueError(msg)
        if self.target_scope not in {"primary_q6", "secondary_q12"}:
            msg = "target_scope must be primary_q6 or secondary_q12."
            raise ValueError(msg)
        qubits = require_int(self.qubit_count, "qubit_count", minimum=1)
        expected_qubits = 6 if self.target_scope == "primary_q6" else 12
        if qubits != expected_qubits:
            msg = "qubit_count does not match target_scope."
            raise ValueError(msg)
        object.__setattr__(self, "qubit_count", qubits)
        allowed_roles = {
            "development",
            "checkpoint_validation",
            "screening_selection",
            "secondary_benchmark",
            "confirmatory",
        }
        if self.data_role not in allowed_roles:
            msg = "data_role is not a supported WP22 execution role."
            raise ValueError(msg)
        expected_role_and_domain = {
            ("pilot_fresh_evaluation", "primary_q6"): ("development", "pilot_evaluation"),
            ("pilot_fresh_evaluation", "secondary_q12"): ("secondary_benchmark", "pilot_evaluation"),
            ("checkpoint_validation", "primary_q6"): ("checkpoint_validation", "checkpoint_validation"),
            ("checkpoint_validation", "secondary_q12"): ("checkpoint_validation", "checkpoint_validation"),
            ("screening_outer", "primary_q6"): ("screening_selection", "screening_selection"),
            ("confirmatory_fresh_evaluation", "primary_q6"): ("confirmatory", "confirmatory_test"),
            ("smoke_evaluation", "primary_q6"): ("development", "pilot_evaluation"),
        }
        expected = expected_role_and_domain.get((self.purpose, self.target_scope))
        if expected is None:
            msg = "purpose is not allowed for this target scope."
            raise ValueError(msg)
        seed_domain = require_slug(self.seed_domain, "seed_domain")
        if (self.data_role, seed_domain) != expected:
            msg = "data_role or seed_domain differs from the frozen purpose/scope separation."
            raise ValueError(msg)
        object.__setattr__(self, "seed_domain", seed_domain)
        count = require_int(self.trajectory_count, "trajectory_count", minimum=1)
        prefixes = _require_int_tuple(self.reporting_prefixes, "reporting_prefixes")
        if not prefixes or tuple(sorted(set(prefixes))) != prefixes or prefixes[-1] != count:
            msg = "reporting_prefixes must be strictly increasing and terminate at trajectory_count."
            raise ValueError(msg)
        if self.purpose == "pilot_fresh_evaluation":
            expected_count = (
                PRIMARY_Q6_PILOT_TRAJECTORY_COUNT
                if self.target_scope == "primary_q6"
                else SECONDARY_Q12_PILOT_TRAJECTORY_COUNT
            )
            expected_prefixes = PRIMARY_Q6_PILOT_PREFIXES if self.target_scope == "primary_q6" else (256,)
            if count != expected_count or prefixes != expected_prefixes:
                msg = "Pilot fresh-evaluation count or reporting prefixes differ from WP22A."
                raise ValueError(msg)
        elif self.purpose == "checkpoint_validation":
            if count != CHECKPOINT_VALIDATION_TRAJECTORY_COUNT or prefixes != (256,):
                msg = "Checkpoint validation requires exactly 256 fixed trajectories."
                raise ValueError(msg)
        elif self.purpose in {"screening_outer", "confirmatory_fresh_evaluation"}:
            if count < OUTER_TRAJECTORY_COUNT_MIN or count > OUTER_TRAJECTORY_COUNT_MAX or count & (count - 1):
                msg = "Paper fresh evaluation requires a bounded power-of-two trajectory count."
                raise ValueError(msg)
        object.__setattr__(self, "trajectory_count", count)
        object.__setattr__(self, "reporting_prefixes", prefixes)

        noise = freeze_json_mapping(self.noise_condition, "noise_condition")
        provider = freeze_json_mapping(self.provider_identity, "provider_identity")
        truncation = freeze_json_mapping(self.truncation_policy, "truncation_policy")
        if not _same_mapping(noise, _primary_noise_condition()):
            msg = "noise_condition must equal the preregistered primary noise condition."
            raise ValueError(msg)
        if not _same_mapping(provider, _provider_identity()):
            msg = "provider_identity does not reproduce the primary standard-noise provider."
            raise ValueError(msg)
        if not _same_mapping(truncation, _truncation_policy()):
            msg = "truncation_policy differs from the exact WP22 no-truncation policy."
            raise ValueError(msg)
        object.__setattr__(self, "noise_condition", noise)
        object.__setattr__(self, "provider_identity", provider)
        object.__setattr__(self, "truncation_policy", truncation)

        fixed_values = {
            "map_schema_version": KROTOV_FIXED_MAP_ENSEMBLE_SCHEMA_VERSION,
            "map_identity_version": KROTOV_FIXED_MAP_ENSEMBLE_IDENTITY_VERSION,
            "sidecar_schema_version": PHASE2_TRAJECTORY_SIDECAR_SCHEMA_VERSION,
            "evaluation_policy": "fixed_sample",
            "failure_treatment": "structured_failure_zero_fidelity_for_intention_to_treat",
        }
        for name, expected_value in fixed_values.items():
            if getattr(self, name) != expected_value:
                msg = f"{name} differs from the frozen WP22A value."
                raise ValueError(msg)
        if require_int(self.worker_count, "worker_count", minimum=1) != 1:
            msg = "WP22A fresh evaluations use one deterministic worker."
            raise ValueError(msg)
        if require_bool(self.trajectory_optional_stopping, "trajectory_optional_stopping"):
            msg = "Fresh evaluation forbids trajectory optional stopping."
            raise ValueError(msg)

    @property
    def seed_derivation_policy_id(self) -> str:
        """Stable identity of the exact derivation policy for this purpose."""
        policy_ids = {
            "pilot_fresh_evaluation": PILOT_FRESH_EVALUATION_SEED_POLICY_ID,
            "checkpoint_validation": STAGE_SEED_DERIVATION_POLICY_ID,
            "screening_outer": SCREENING_CELL_SEED_POLICY_ID,
            "confirmatory_fresh_evaluation": CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID,
            "smoke_evaluation": SMOKE_FRESH_EVALUATION_SEED_POLICY_ID,
        }
        return policy_ids[self.purpose]

    @property
    def seed_derivation_policy_checksum(self) -> str:
        """Checksum of the exact derivation policy for this purpose."""
        return ExecutionSeedPolicySuite.frozen().policy(self.seed_derivation_policy_id).content_checksum

    @classmethod
    def primary_q6_pilot(cls) -> FreshEvaluationPolicy:
        """Build the exact 1,024-trajectory primary-q6 pilot policy.

        Returns:
            The frozen primary-q6 pilot evaluation policy.
        """
        return cls(
            policy_id="primary_q6_pilot_fresh_evaluation",
            purpose="pilot_fresh_evaluation",
            target_scope="primary_q6",
            qubit_count=6,
            data_role="development",
            seed_domain="pilot_evaluation",
            trajectory_count=1024,
            reporting_prefixes=PRIMARY_Q6_PILOT_PREFIXES,
        )

    @classmethod
    def secondary_q12_pilot(cls) -> FreshEvaluationPolicy:
        """Build the exact 256-trajectory secondary-q12 pilot policy.

        Returns:
            The frozen secondary-q12 pilot evaluation policy.
        """
        return cls(
            policy_id="secondary_q12_pilot_fresh_evaluation",
            purpose="pilot_fresh_evaluation",
            target_scope="secondary_q12",
            qubit_count=12,
            data_role="secondary_benchmark",
            seed_domain="pilot_evaluation",
            trajectory_count=256,
            reporting_prefixes=(256,),
        )

    @classmethod
    def checkpoint_validation(
        cls,
        target_scope: TargetScope = "primary_q6",
    ) -> FreshEvaluationPolicy:
        """Build the exact fixed checkpoint-validation policy for one width.

        Returns:
            The frozen checkpoint-validation policy for ``target_scope``.

        Raises:
            ValueError: If ``target_scope`` is not a supported target width.
        """
        scope = cast("TargetScope", require_slug(target_scope, "target_scope"))
        if scope not in {"primary_q6", "secondary_q12"}:
            msg = "target_scope must be primary_q6 or secondary_q12."
            raise ValueError(msg)
        qubit_count = 6 if scope == "primary_q6" else 12
        return cls(
            policy_id=f"{scope}_checkpoint_validation",
            purpose="checkpoint_validation",
            target_scope=scope,
            qubit_count=qubit_count,
            data_role="checkpoint_validation",
            seed_domain="checkpoint_validation",
            trajectory_count=256,
            reporting_prefixes=(256,),
        )

    @classmethod
    def smoke(cls, trajectory_count: int) -> FreshEvaluationPolicy:
        """Build a role-specific q6 smoke-evaluation policy.

        Returns:
            A primary-q6 smoke-evaluation policy with the fixed requested count.
        """
        return cls(
            policy_id="primary_q6_smoke_evaluation",
            purpose="smoke_evaluation",
            target_scope="primary_q6",
            qubit_count=6,
            data_role="development",
            seed_domain="pilot_evaluation",
            trajectory_count=trajectory_count,
            reporting_prefixes=(trajectory_count,),
        )

    @classmethod
    def screening(cls, trajectory_count: int) -> FreshEvaluationPolicy:
        """Build a q6 outer-screen policy with one pilot-derived fixed count.

        Returns:
            A primary-q6 screening policy with the fixed requested count.
        """
        return cls(
            policy_id="primary_q6_screening_outer_evaluation",
            purpose="screening_outer",
            target_scope="primary_q6",
            qubit_count=6,
            data_role="screening_selection",
            seed_domain="screening_selection",
            trajectory_count=trajectory_count,
            reporting_prefixes=(trajectory_count,),
        )

    @classmethod
    def confirmatory(cls, trajectory_count: int) -> FreshEvaluationPolicy:
        """Build the frozen q6 confirmatory fixed-sample policy.

        Returns:
            A role-separated primary-q6 policy with the already sealed fixed
            trajectory count.
        """
        return cls(
            policy_id="primary_q6_confirmatory_fresh_evaluation",
            purpose="confirmatory_fresh_evaluation",
            target_scope="primary_q6",
            qubit_count=6,
            data_role="confirmatory",
            seed_domain="confirmatory_test",
            trajectory_count=trajectory_count,
            reporting_prefixes=(trajectory_count,),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered fresh-evaluation choice."""
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "purpose": self.purpose,
            "target_scope": self.target_scope,
            "qubit_count": self.qubit_count,
            "data_role": self.data_role,
            "seed_domain": self.seed_domain,
            "noise_condition": thaw_json_mapping(self.noise_condition),
            "provider_identity": thaw_json_mapping(self.provider_identity),
            "trajectory_count": self.trajectory_count,
            "reporting_prefixes": list(self.reporting_prefixes),
            "seed_derivation_policy_id": self.seed_derivation_policy_id,
            "seed_derivation_policy_checksum": self.seed_derivation_policy_checksum,
            "map_schema_version": self.map_schema_version,
            "map_identity_version": self.map_identity_version,
            "truncation_policy": thaw_json_mapping(self.truncation_policy),
            "worker_count": self.worker_count,
            "sidecar_schema_version": self.sidecar_schema_version,
            "evaluation_policy": self.evaluation_policy,
            "trajectory_optional_stopping": self.trajectory_optional_stopping,
            "failure_treatment": self.failure_treatment,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing the complete fresh-evaluation policy."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> FreshEvaluationPolicy:
        """Decode and verify a complete fresh-evaluation policy.

        Returns:
            The verified fresh-evaluation policy.

        Raises:
            ValueError: If the schema, seed-policy reference, or checksum is invalid.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_FRESH_EVALUATION_KEYS,
            name="WP22 fresh-evaluation policy",
        )
        if mapping["schema_version"] != FRESH_EVALUATION_POLICY_SCHEMA_VERSION:
            msg = "Fresh-evaluation policy uses an unsupported schema version."
            raise ValueError(msg)
        purpose = cast("FreshPurpose", mapping["purpose"])
        expected_policy_id = {
            "pilot_fresh_evaluation": PILOT_FRESH_EVALUATION_SEED_POLICY_ID,
            "checkpoint_validation": STAGE_SEED_DERIVATION_POLICY_ID,
            "screening_outer": SCREENING_CELL_SEED_POLICY_ID,
            "confirmatory_fresh_evaluation": CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID,
            "smoke_evaluation": SMOKE_FRESH_EVALUATION_SEED_POLICY_ID,
        }.get(purpose)
        if expected_policy_id is None:
            msg = "Fresh-evaluation purpose has no reviewed seed policy."
            raise ValueError(msg)
        expected_policy_checksum = ExecutionSeedPolicySuite.frozen().policy(expected_policy_id).content_checksum
        if (
            mapping["seed_derivation_policy_id"] != expected_policy_id
            or mapping["seed_derivation_policy_checksum"] != expected_policy_checksum
        ):
            msg = "Fresh-evaluation seed policy reference changed."
            raise ValueError(msg)
        policy = cls(
            policy_id=cast("str", mapping["policy_id"]),
            purpose=purpose,
            target_scope=cast("TargetScope", mapping["target_scope"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            data_role=cast("EvaluationRole", mapping["data_role"]),
            seed_domain=cast("str", mapping["seed_domain"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            reporting_prefixes=cast("tuple[int, ...]", mapping["reporting_prefixes"]),
            noise_condition=cast("Mapping[str, object]", mapping["noise_condition"]),
            provider_identity=cast("Mapping[str, object]", mapping["provider_identity"]),
            map_schema_version=cast("str", mapping["map_schema_version"]),
            map_identity_version=cast("str", mapping["map_identity_version"]),
            truncation_policy=cast("Mapping[str, object]", mapping["truncation_policy"]),
            worker_count=cast("int", mapping["worker_count"]),
            sidecar_schema_version=cast("str", mapping["sidecar_schema_version"]),
            evaluation_policy=cast("str", mapping["evaluation_policy"]),
            trajectory_optional_stopping=cast("bool", mapping["trajectory_optional_stopping"]),
            failure_treatment=cast("str", mapping["failure_treatment"]),
        )
        if mapping["content_checksum"] != policy.content_checksum:
            msg = "Fresh-evaluation policy checksum changed during normalization."
            raise ValueError(msg)
        return policy

    @classmethod
    def from_json(cls, payload: str) -> FreshEvaluationPolicy:
        """Decode canonical JSON into a verified fresh-evaluation policy.

        Returns:
            The verified fresh-evaluation policy.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class PilotDiagnosticPolicy:
    """Exact q6 pathwise-update diagnostic or explicit q12 exclusion."""

    policy_id: str
    target_scope: TargetScope
    qubit_count: int
    data_role: EvaluationRole
    enabled: bool
    endpoint: str | None
    trajectory_count: int
    seed_domain: str | None
    noise_condition: Mapping[str, object] | None
    provider_identity: Mapping[str, object] | None
    map_schema_version: str | None
    checkpoint_rule: str | None
    estimator_id: str | None
    estimator_version: str | None
    parameter_ordering: str | None
    coordinate_variance_rule: str | None
    summary_statistics: tuple[str, ...]
    store_complete_vectors: bool
    successful_jobs_only: bool
    promotion_eligible: bool = False
    schema_version: str = field(default=PILOT_DIAGNOSTIC_POLICY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require the exact primary diagnostic and mechanical q12 exclusion.

        Raises:
            ValueError: If the diagnostic differs from its frozen target-scope policy.
        """
        object.__setattr__(self, "policy_id", require_slug(self.policy_id, "policy_id"))
        if self.target_scope not in {"primary_q6", "secondary_q12"}:
            msg = "target_scope must be primary_q6 or secondary_q12."
            raise ValueError(msg)
        qubits = require_int(self.qubit_count, "qubit_count", minimum=1)
        expected_qubits = 6 if self.target_scope == "primary_q6" else 12
        if qubits != expected_qubits:
            msg = "Diagnostic qubit_count does not match target_scope."
            raise ValueError(msg)
        object.__setattr__(self, "qubit_count", qubits)
        enabled = require_bool(self.enabled, "enabled")
        count = require_int(self.trajectory_count, "trajectory_count")
        summaries = _require_string_tuple(self.summary_statistics, "summary_statistics")
        object.__setattr__(self, "summary_statistics", summaries)
        for name in ("store_complete_vectors", "successful_jobs_only", "promotion_eligible"):
            object.__setattr__(self, name, require_bool(getattr(self, name), name))

        if self.target_scope == "secondary_q12":
            if self.data_role != "secondary_benchmark" or enabled or count != 0:
                msg = "Secondary-q12 pilot diagnostics must be explicitly disabled."
                raise ValueError(msg)
            optional_values = (
                self.endpoint,
                self.seed_domain,
                self.noise_condition,
                self.provider_identity,
                self.map_schema_version,
                self.checkpoint_rule,
                self.estimator_id,
                self.estimator_version,
                self.parameter_ordering,
                self.coordinate_variance_rule,
            )
            if any(value is not None for value in optional_values) or summaries:
                msg = "Disabled q12 diagnostics cannot carry estimator or noise choices."
                raise ValueError(msg)
            if self.store_complete_vectors or self.successful_jobs_only or self.promotion_eligible:
                msg = "Disabled q12 diagnostics cannot claim storage, filtering, or promotion."
                raise ValueError(msg)
            return

        expected_values = {
            "data_role": "development",
            "enabled": True,
            "endpoint": "post_training_primary_noise_pathwise_update_variance",
            "trajectory_count": PRIMARY_Q6_DIAGNOSTIC_VECTOR_COUNT,
            "seed_domain": "pilot_evaluation",
            "map_schema_version": KROTOV_FIXED_MAP_ENSEMBLE_SCHEMA_VERSION,
            "checkpoint_rule": "inner_validation_selected_checkpoint",
            "estimator_id": "independent_single_trajectory_pathwise_update_vector",
            "estimator_version": "yaqs.state_preparation.phase2.pathwise_update_estimator.v1",
            "parameter_ordering": "materialized_circuit_parameter_order",
            "coordinate_variance_rule": "unbiased_sample_variance_ddof_1",
            "summary_statistics": ("arithmetic_mean", "maximum"),
            "store_complete_vectors": True,
            "successful_jobs_only": True,
            "promotion_eligible": False,
        }
        for name, expected_value in expected_values.items():
            if getattr(self, name) != expected_value:
                msg = f"{name} differs from the frozen primary-q6 diagnostic."
                raise ValueError(msg)
        if self.noise_condition is None or self.provider_identity is None:
            msg = "Primary-q6 diagnostic requires complete primary-noise provider identity."
            raise ValueError(msg)
        noise = freeze_json_mapping(self.noise_condition, "noise_condition")
        provider = freeze_json_mapping(self.provider_identity, "provider_identity")
        if not _same_mapping(noise, _primary_noise_condition()) or not _same_mapping(provider, _provider_identity()):
            msg = "Primary-q6 diagnostic noise or provider differs from the preregistered condition."
            raise ValueError(msg)
        object.__setattr__(self, "noise_condition", noise)
        object.__setattr__(self, "provider_identity", provider)

    @property
    def seed_derivation_policy_id(self) -> str | None:
        """Stable derivation-policy identity, absent for the disabled q12 diagnostic."""
        return PILOT_DIAGNOSTIC_SEED_POLICY_ID if self.enabled else None

    @property
    def seed_derivation_policy_checksum(self) -> str | None:
        """Checksum of the enabled diagnostic derivation policy."""
        if self.seed_derivation_policy_id is None:
            return None
        return ExecutionSeedPolicySuite.frozen().policy(self.seed_derivation_policy_id).content_checksum

    @classmethod
    def primary_q6(cls) -> PilotDiagnosticPolicy:
        """Build the exact 32-vector primary-q6 diagnostic policy.

        Returns:
            The frozen primary-q6 pathwise-update diagnostic policy.
        """
        return cls(
            policy_id="primary_q6_pathwise_update_diagnostic",
            target_scope="primary_q6",
            qubit_count=6,
            data_role="development",
            enabled=True,
            endpoint="post_training_primary_noise_pathwise_update_variance",
            trajectory_count=32,
            seed_domain="pilot_evaluation",
            noise_condition=_primary_noise_condition(),
            provider_identity=_provider_identity(),
            map_schema_version=KROTOV_FIXED_MAP_ENSEMBLE_SCHEMA_VERSION,
            checkpoint_rule="inner_validation_selected_checkpoint",
            estimator_id="independent_single_trajectory_pathwise_update_vector",
            estimator_version="yaqs.state_preparation.phase2.pathwise_update_estimator.v1",
            parameter_ordering="materialized_circuit_parameter_order",
            coordinate_variance_rule="unbiased_sample_variance_ddof_1",
            summary_statistics=("arithmetic_mean", "maximum"),
            store_complete_vectors=True,
            successful_jobs_only=True,
        )

    @classmethod
    def secondary_q12(cls) -> PilotDiagnosticPolicy:
        """Build the explicit no-diagnostic secondary-q12 policy.

        Returns:
            The frozen policy that disables secondary-q12 diagnostics.
        """
        return cls(
            policy_id="secondary_q12_no_pilot_diagnostic",
            target_scope="secondary_q12",
            qubit_count=12,
            data_role="secondary_benchmark",
            enabled=False,
            endpoint=None,
            trajectory_count=0,
            seed_domain=None,
            noise_condition=None,
            provider_identity=None,
            map_schema_version=None,
            checkpoint_rule=None,
            estimator_id=None,
            estimator_version=None,
            parameter_ordering=None,
            coordinate_variance_rule=None,
            summary_statistics=(),
            store_complete_vectors=False,
            successful_jobs_only=False,
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered pilot diagnostic choice."""
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "target_scope": self.target_scope,
            "qubit_count": self.qubit_count,
            "data_role": self.data_role,
            "enabled": self.enabled,
            "endpoint": self.endpoint,
            "trajectory_count": self.trajectory_count,
            "seed_domain": self.seed_domain,
            "seed_derivation_policy_id": self.seed_derivation_policy_id,
            "seed_derivation_policy_checksum": self.seed_derivation_policy_checksum,
            "noise_condition": None if self.noise_condition is None else thaw_json_mapping(self.noise_condition),
            "provider_identity": (
                None if self.provider_identity is None else thaw_json_mapping(self.provider_identity)
            ),
            "map_schema_version": self.map_schema_version,
            "checkpoint_rule": self.checkpoint_rule,
            "estimator_id": self.estimator_id,
            "estimator_version": self.estimator_version,
            "parameter_ordering": self.parameter_ordering,
            "coordinate_variance_rule": self.coordinate_variance_rule,
            "summary_statistics": list(self.summary_statistics),
            "store_complete_vectors": self.store_complete_vectors,
            "successful_jobs_only": self.successful_jobs_only,
            "promotion_eligible": self.promotion_eligible,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing the complete diagnostic policy."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> PilotDiagnosticPolicy:
        """Decode and verify a complete pilot diagnostic policy.

        Returns:
            The verified pilot diagnostic policy.

        Raises:
            ValueError: If the schema, seed-policy reference, or checksum is invalid.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_PILOT_DIAGNOSTIC_KEYS,
            name="WP22 pilot diagnostic policy",
        )
        if mapping["schema_version"] != PILOT_DIAGNOSTIC_POLICY_SCHEMA_VERSION:
            msg = "Pilot diagnostic policy uses an unsupported schema version."
            raise ValueError(msg)
        enabled = cast("bool", mapping["enabled"])
        expected_policy_id = PILOT_DIAGNOSTIC_SEED_POLICY_ID if enabled else None
        expected_policy_checksum = (
            None
            if expected_policy_id is None
            else ExecutionSeedPolicySuite.frozen().policy(expected_policy_id).content_checksum
        )
        if (
            mapping["seed_derivation_policy_id"] != expected_policy_id
            or mapping["seed_derivation_policy_checksum"] != expected_policy_checksum
        ):
            msg = "Pilot diagnostic seed policy reference changed."
            raise ValueError(msg)
        raw_noise = mapping["noise_condition"]
        raw_provider = mapping["provider_identity"]
        policy = cls(
            policy_id=cast("str", mapping["policy_id"]),
            target_scope=cast("TargetScope", mapping["target_scope"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            data_role=cast("EvaluationRole", mapping["data_role"]),
            enabled=enabled,
            endpoint=cast("str | None", mapping["endpoint"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            seed_domain=cast("str | None", mapping["seed_domain"]),
            noise_condition=None if raw_noise is None else cast("Mapping[str, object]", raw_noise),
            provider_identity=None if raw_provider is None else cast("Mapping[str, object]", raw_provider),
            map_schema_version=cast("str | None", mapping["map_schema_version"]),
            checkpoint_rule=cast("str | None", mapping["checkpoint_rule"]),
            estimator_id=cast("str | None", mapping["estimator_id"]),
            estimator_version=cast("str | None", mapping["estimator_version"]),
            parameter_ordering=cast("str | None", mapping["parameter_ordering"]),
            coordinate_variance_rule=cast("str | None", mapping["coordinate_variance_rule"]),
            summary_statistics=cast("tuple[str, ...]", mapping["summary_statistics"]),
            store_complete_vectors=cast("bool", mapping["store_complete_vectors"]),
            successful_jobs_only=cast("bool", mapping["successful_jobs_only"]),
            promotion_eligible=cast("bool", mapping["promotion_eligible"]),
        )
        if mapping["content_checksum"] != policy.content_checksum:
            msg = "Pilot diagnostic policy checksum changed during normalization."
            raise ValueError(msg)
        return policy

    @classmethod
    def from_json(cls, payload: str) -> PilotDiagnosticPolicy:
        """Decode canonical JSON into a verified pilot diagnostic policy.

        Returns:
            The verified pilot diagnostic policy.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class OperatorGrowthExecutionSpec:
    """Complete role-specific q6 execution policy around the exact WP20 core."""

    pool: OperatorPoolSpec
    growth_spec: OperatorGrowthSpec
    outer_evaluation_policy: FreshEvaluationPolicy
    training_noise_condition: Mapping[str, object] = field(default_factory=_primary_noise_condition)
    provider_identity: Mapping[str, object] = field(default_factory=_provider_identity)
    training_sampling_policy: TrajectorySamplingPolicy = field(
        default_factory=lambda: TrajectorySamplingPolicy(kind="fixed_crn")
    )
    checkpoint_validation_policy: FreshEvaluationPolicy = field(
        default_factory=FreshEvaluationPolicy.checkpoint_validation
    )
    resource_policy: Mapping[str, object] = field(default_factory=_primary_resource_policy)
    normalized_compute_policy: NormalizedComputePolicy = field(default_factory=NormalizedComputePolicy)
    target_scope: TargetScope = field(default="primary_q6", init=False)
    qubit_count: int = field(default=6, init=False)
    method_id: str = field(default=ADAPT_STYLE_METHOD_ID, init=False)
    objective_id: str = field(default=PROJECTOR_COST_ID, init=False)
    training_trajectory_count: int = field(default=PRODUCTION_TRAINING_TRAJECTORY_COUNT, init=False)
    training_ensemble_rule: str = field(default="one_fixed_primary_noise_crn_ensemble", init=False)
    checkpoint_selection_rule: str = field(
        default="greatest_validation_fidelity_earliest_growth_step",
        init=False,
    )
    validation_after_each_completed_prefix: bool = field(default=True, init=False)
    supported_schedule_mode: str = field(default="fixed_matched_noise", init=False)
    explicitly_unsupported_schedule_modes: tuple[str, ...] = field(
        default=("continuation", "rolling", "frozen_mixture", "width_other_than_q6"),
        init=False,
    )
    schema_version: str = field(default=OPERATOR_GROWTH_EXECUTION_SPEC_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Bind the exact pool, optimizer, noise, validation, and resource policies.

        Raises:
            TypeError: If a nested policy does not have its required typed representation.
            ValueError: If a nested policy differs from the frozen operator-growth protocol.
        """
        if not isinstance(self.pool, OperatorPoolSpec) or not isinstance(self.growth_spec, OperatorGrowthSpec):
            msg = "pool and growth_spec must be exact WP20 operator-growth artifacts."
            raise TypeError(msg)
        expected_pool = build_projector_operator_pool(6)
        if self.pool.content_checksum != expected_pool.content_checksum:
            msg = "Operator growth requires the exact q6 projector pool."
            raise ValueError(msg)
        expected_growth = OperatorGrowthSpec.for_pool(
            expected_pool,
            gradient_tolerance=1e-10,
            max_operators=16,
            native_two_qubit_cap_per_edge=12,
            reoptimization_steps=100,
            learning_rate=0.08,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
        )
        if self.growth_spec.content_checksum != expected_growth.content_checksum:
            msg = "growth_spec differs from the frozen WP22 operator-growth settings."
            raise ValueError(msg)
        if not isinstance(self.training_sampling_policy, TrajectorySamplingPolicy):
            msg = "training_sampling_policy must be a TrajectorySamplingPolicy."
            raise TypeError(msg)
        if self.training_sampling_policy != TrajectorySamplingPolicy(kind="fixed_crn"):
            msg = "Operator-growth training requires one fixed CRN ensemble."
            raise ValueError(msg)
        if not isinstance(self.checkpoint_validation_policy, FreshEvaluationPolicy):
            msg = "checkpoint_validation_policy must be a FreshEvaluationPolicy."
            raise TypeError(msg)
        if self.checkpoint_validation_policy != FreshEvaluationPolicy.checkpoint_validation():
            msg = "Operator-growth checkpoint validation differs from the exact 256-trajectory policy."
            raise ValueError(msg)
        if not isinstance(self.outer_evaluation_policy, FreshEvaluationPolicy):
            msg = "outer_evaluation_policy must be a FreshEvaluationPolicy."
            raise TypeError(msg)
        if self.outer_evaluation_policy.purpose not in {"screening_outer", "smoke_evaluation"}:
            msg = "Operator-growth execution requires a q6 screening or smoke outer policy."
            raise ValueError(msg)
        if not isinstance(self.normalized_compute_policy, NormalizedComputePolicy):
            msg = "normalized_compute_policy must be a NormalizedComputePolicy."
            raise TypeError(msg)
        if self.normalized_compute_policy != NormalizedComputePolicy():
            msg = "Operator growth requires the exact WP20 normalized-compute policy."
            raise ValueError(msg)
        noise = freeze_json_mapping(self.training_noise_condition, "training_noise_condition")
        provider = freeze_json_mapping(self.provider_identity, "provider_identity")
        resource = freeze_json_mapping(self.resource_policy, "resource_policy")
        if not _same_mapping(noise, _primary_noise_condition()) or not _same_mapping(provider, _provider_identity()):
            msg = "Operator-growth training must use the preregistered primary-noise provider."
            raise ValueError(msg)
        if not _same_mapping(resource, _primary_resource_policy()):
            msg = "Operator-growth resource policy differs from the preregistered q6 policy."
            raise ValueError(msg)
        object.__setattr__(self, "training_noise_condition", noise)
        object.__setattr__(self, "provider_identity", provider)
        object.__setattr__(self, "resource_policy", resource)

    @property
    def training_seed_derivation_policy_id(self) -> str:
        """Policy that derives role-specific stage roots from an optimization seed."""
        return STAGE_SEED_DERIVATION_POLICY_ID

    @property
    def training_seed_derivation_policy_checksum(self) -> str:
        """Checksum of the role-specific stage-root policy."""
        return ExecutionSeedPolicySuite.frozen().policy(self.training_seed_derivation_policy_id).content_checksum

    @property
    def trajectory_member_seed_policy_id(self) -> str:
        """Policy that derives persisted direct sampler seeds within a schedule."""
        return SCHEDULE_SEED_DERIVATION_POLICY_ID

    @property
    def trajectory_member_seed_policy_checksum(self) -> str:
        """Checksum of the persisted membership seed policy."""
        return ExecutionSeedPolicySuite.frozen().policy(self.trajectory_member_seed_policy_id).content_checksum

    @classmethod
    def for_screening(cls, outer_trajectory_count: int) -> OperatorGrowthExecutionSpec:
        """Build the exact publication spec with its pilot-derived outer count.

        Returns:
            The frozen operator-growth screening specification.
        """
        pool = build_projector_operator_pool(6)
        growth = OperatorGrowthSpec.for_pool(
            pool,
            gradient_tolerance=1e-10,
            max_operators=16,
            native_two_qubit_cap_per_edge=12,
            reoptimization_steps=100,
            learning_rate=0.08,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
        )
        return cls(
            pool=pool,
            growth_spec=growth,
            outer_evaluation_policy=FreshEvaluationPolicy.screening(outer_trajectory_count),
        )

    @classmethod
    def for_smoke(cls, outer_trajectory_count: int) -> OperatorGrowthExecutionSpec:
        """Build the exact core under a role-specific tiny-budget smoke wrapper.

        Returns:
            The frozen operator-growth smoke specification.
        """
        pool = build_projector_operator_pool(6)
        growth = OperatorGrowthSpec.for_pool(
            pool,
            gradient_tolerance=1e-10,
            max_operators=16,
            native_two_qubit_cap_per_edge=12,
            reoptimization_steps=100,
            learning_rate=0.08,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
        )
        return cls(
            pool=pool,
            growth_spec=growth,
            outer_evaluation_policy=FreshEvaluationPolicy.smoke(outer_trajectory_count),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered operator-growth execution choice."""
        return {
            "schema_version": self.schema_version,
            "target_scope": self.target_scope,
            "qubit_count": self.qubit_count,
            "method_id": self.method_id,
            "objective_id": self.objective_id,
            "pool": self.pool.to_dict(),
            "growth_spec": self.growth_spec.to_dict(),
            "training_noise_condition": thaw_json_mapping(self.training_noise_condition),
            "provider_identity": thaw_json_mapping(self.provider_identity),
            "training_trajectory_count": self.training_trajectory_count,
            "training_sampling_policy": self.training_sampling_policy.to_dict(),
            "training_seed_derivation_policy_id": self.training_seed_derivation_policy_id,
            "training_seed_derivation_policy_checksum": self.training_seed_derivation_policy_checksum,
            "trajectory_member_seed_policy_id": self.trajectory_member_seed_policy_id,
            "trajectory_member_seed_policy_checksum": self.trajectory_member_seed_policy_checksum,
            "training_ensemble_rule": self.training_ensemble_rule,
            "checkpoint_validation_policy": self.checkpoint_validation_policy.to_dict(),
            "outer_evaluation_policy": self.outer_evaluation_policy.to_dict(),
            "checkpoint_selection_rule": self.checkpoint_selection_rule,
            "validation_after_each_completed_prefix": self.validation_after_each_completed_prefix,
            "resource_policy": thaw_json_mapping(self.resource_policy),
            "normalized_compute_policy": self.normalized_compute_policy.to_dict(),
            "supported_schedule_mode": self.supported_schedule_mode,
            "explicitly_unsupported_schedule_modes": list(self.explicitly_unsupported_schedule_modes),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing the complete operator-growth execution spec."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> OperatorGrowthExecutionSpec:
        """Decode and verify a complete operator-growth execution specification.

        Returns:
            The verified operator-growth execution specification.

        Raises:
            ValueError: If a fixed alias, nested policy, or checksum is invalid.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_OPERATOR_GROWTH_EXECUTION_KEYS,
            name="WP22 operator-growth execution spec",
        )
        if mapping["schema_version"] != OPERATOR_GROWTH_EXECUTION_SPEC_SCHEMA_VERSION:
            msg = "Operator-growth execution spec uses an unsupported schema version."
            raise ValueError(msg)
        fixed_aliases = {
            "target_scope": "primary_q6",
            "qubit_count": 6,
            "method_id": ADAPT_STYLE_METHOD_ID,
            "objective_id": PROJECTOR_COST_ID,
            "training_trajectory_count": 8,
            "training_seed_derivation_policy_id": STAGE_SEED_DERIVATION_POLICY_ID,
            "training_seed_derivation_policy_checksum": (
                ExecutionSeedPolicySuite.frozen().policy(STAGE_SEED_DERIVATION_POLICY_ID).content_checksum
            ),
            "trajectory_member_seed_policy_id": SCHEDULE_SEED_DERIVATION_POLICY_ID,
            "trajectory_member_seed_policy_checksum": (
                ExecutionSeedPolicySuite.frozen().policy(SCHEDULE_SEED_DERIVATION_POLICY_ID).content_checksum
            ),
            "training_ensemble_rule": "one_fixed_primary_noise_crn_ensemble",
            "checkpoint_selection_rule": "greatest_validation_fidelity_earliest_growth_step",
            "validation_after_each_completed_prefix": True,
            "supported_schedule_mode": "fixed_matched_noise",
            "explicitly_unsupported_schedule_modes": (
                "continuation",
                "rolling",
                "frozen_mixture",
                "width_other_than_q6",
            ),
        }
        if any(mapping[name] != expected for name, expected in fixed_aliases.items()):
            msg = "Operator-growth execution aliases differ from the frozen WP22A policy."
            raise ValueError(msg)
        outer_policy = FreshEvaluationPolicy.from_dict(mapping["outer_evaluation_policy"])
        spec = cls(
            pool=OperatorPoolSpec.from_dict(mapping["pool"]),
            growth_spec=OperatorGrowthSpec.from_dict(mapping["growth_spec"]),
            outer_evaluation_policy=outer_policy,
            training_noise_condition=cast("Mapping[str, object]", mapping["training_noise_condition"]),
            provider_identity=cast("Mapping[str, object]", mapping["provider_identity"]),
            training_sampling_policy=TrajectorySamplingPolicy.from_dict(mapping["training_sampling_policy"]),
            checkpoint_validation_policy=FreshEvaluationPolicy.from_dict(mapping["checkpoint_validation_policy"]),
            resource_policy=cast("Mapping[str, object]", mapping["resource_policy"]),
            normalized_compute_policy=NormalizedComputePolicy.from_dict(mapping["normalized_compute_policy"]),
        )
        if mapping["content_checksum"] != spec.content_checksum:
            msg = "Operator-growth execution checksum changed during normalization."
            raise ValueError(msg)
        return spec

    @classmethod
    def from_json(cls, payload: str) -> OperatorGrowthExecutionSpec:
        """Decode canonical JSON into a verified operator-growth execution spec.

        Returns:
            The verified operator-growth execution specification.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def bounded_outer_trajectory_count(unbiased_variances: Sequence[float]) -> int:
    """Apply the frozen Maurer--Pontil bounded-data outer-count rule.

    Args:
        unbiased_variances: Exactly 720 q6 per-job unbiased fidelity variances.

    Returns:
        The next power-of-two fixed count in the closed interval [256, 16384].

    Raises:
        TypeError: If the variances are not a sequence of strict floats.
        ValueError: If the evidence is incomplete or contains an invalid variance.
        RuntimeError: If the derived count exceeds the preregistered maximum.
    """
    if isinstance(unbiased_variances, (str, bytes)) or not isinstance(unbiased_variances, Sequence):
        msg = "unbiased_variances must be a sequence of strict floats."
        raise TypeError(msg)
    if len(unbiased_variances) != OUTER_VARIANCE_PLANNED_JOB_COUNT:
        msg = f"unbiased_variances must contain exactly {OUTER_VARIANCE_PLANNED_JOB_COUNT} pilot jobs."
        raise ValueError(msg)
    checked = tuple(
        require_float(value, f"unbiased_variances[{index}]", minimum=0.0)
        for index, value in enumerate(unbiased_variances)
    )
    confidence_radius = math.sqrt(
        2.0
        * math.log(OUTER_VARIANCE_PLANNED_JOB_COUNT / OUTER_VARIANCE_ALPHA)
        / (PRIMARY_Q6_PILOT_TRAJECTORY_COUNT - 1)
    )
    variance_upper_bound = max(min(0.25, (math.sqrt(variance) + confidence_radius) ** 2) for variance in checked)
    required = max(
        OUTER_TRAJECTORY_COUNT_MIN,
        math.ceil(variance_upper_bound / OUTER_TRAJECTORY_MCSE_TARGET**2),
    )
    count = 1 << (required - 1).bit_length()
    if count > OUTER_TRAJECTORY_COUNT_MAX:
        msg = "Pilot-derived outer trajectory count exceeds the preregistered maximum."
        raise RuntimeError(msg)
    return count


@dataclass(frozen=True, slots=True)
class OperationalProtocolAmendment:
    """Checksum-sealed prospective WP22 operational addendum."""

    primary_q6_fresh_evaluation_policy: FreshEvaluationPolicy
    secondary_q12_fresh_evaluation_policy: FreshEvaluationPolicy
    primary_q6_diagnostic_policy: PilotDiagnosticPolicy
    secondary_q12_diagnostic_policy: PilotDiagnosticPolicy
    execution_seed_policy_suite: ExecutionSeedPolicySuite
    training_policy_universe: FrozenTrainingPolicyUniverse
    amendment_id: str = field(default="wp22_execution_protocol_closure", init=False)
    preregistration_checksum: str = field(default=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM, init=False)
    implementation_plan_commit: str = field(default=WP22_IMPLEMENTATION_PLAN_COMMIT, init=False)
    prospective_status: str = field(default="replanned_before_wp22_execution", init=False)
    subpackage_order: tuple[str, ...] = field(
        default=("wp22a", "wp22b", "wp22c", "wp22d", "wp22e", "wp22f"),
        init=False,
    )
    pilot_method_ids: tuple[str, ...] = field(default=PILOT_METHOD_IDS, init=False)
    screen_method_ids: tuple[str, ...] = field(default=SCREEN_METHOD_IDS, init=False)
    production_update_count: int = field(default=PRODUCTION_UPDATE_COUNT, init=False)
    production_terminal_update: int = field(default=PRODUCTION_TERMINAL_UPDATE, init=False)
    training_trajectory_count: int = field(default=PRODUCTION_TRAINING_TRAJECTORY_COUNT, init=False)
    checkpoint_validation_trajectory_count: int = field(
        default=CHECKPOINT_VALIDATION_TRAJECTORY_COUNT,
        init=False,
    )
    checkpoint_validation_cadence: int = field(default=CHECKPOINT_VALIDATION_CADENCE, init=False)
    checkpoint_validation_updates: tuple[int, ...] = field(default=CHECKPOINT_VALIDATION_UPDATES, init=False)
    checkpoint_selection_tie_rule: str = field(default="earliest_update", init=False)
    primary_methods_stop_early: bool = field(default=False, init=False)
    optimizer_state_preserved_across_schedule_boundaries: bool = field(default=True, init=False)
    unsupported_composition_action: str = field(default="reject_without_approximation", init=False)
    resume_identity_requirement: str = field(default="byte_identical_program_state_and_result", init=False)
    primary_q6_pilot_target_count: int = field(default=PRIMARY_Q6_PILOT_TARGET_COUNT, init=False)
    secondary_q12_pilot_target_count: int = field(default=SECONDARY_Q12_PILOT_TARGET_COUNT, init=False)
    pilot_optimization_seed_count: int = field(default=PILOT_OPTIMIZATION_SEED_COUNT, init=False)
    pilot_configuration_count: int = field(default=PILOT_CONFIGURATION_COUNT, init=False)
    primary_q6_pilot_job_count: int = field(default=PRIMARY_Q6_PILOT_JOB_COUNT, init=False)
    secondary_q12_pilot_job_count: int = field(default=SECONDARY_Q12_PILOT_JOB_COUNT, init=False)
    q12_inference_eligible: bool = field(default=False, init=False)
    q12_screening_eligible: bool = field(default=False, init=False)
    q12_promotion_eligible: bool = field(default=False, init=False)
    screen_target_count: int = field(default=SCREEN_TARGET_COUNT, init=False)
    screen_optimization_seed_count: int = field(default=SCREEN_OPTIMIZATION_SEED_COUNT, init=False)
    screen_method_count: int = field(default=SCREEN_METHOD_COUNT, init=False)
    screen_cell_count: int = field(default=SCREEN_CELL_COUNT, init=False)
    screen_adaptive: bool = field(default=False, init=False)
    outer_variance_method: str = field(default="maurer_pontil_2009_theorem_10_union_bound", init=False)
    outer_variance_planned_job_count: int = field(default=OUTER_VARIANCE_PLANNED_JOB_COUNT, init=False)
    outer_variance_alpha: float = field(default=OUTER_VARIANCE_ALPHA, init=False)
    outer_trajectory_mcse_target: float = field(default=OUTER_TRAJECTORY_MCSE_TARGET, init=False)
    outer_trajectory_count_min: int = field(default=OUTER_TRAJECTORY_COUNT_MIN, init=False)
    outer_trajectory_count_max: int = field(default=OUTER_TRAJECTORY_COUNT_MAX, init=False)
    outer_trajectory_optional_stopping: bool = field(default=False, init=False)
    schema_version: str = field(default=OPERATIONAL_PROTOCOL_AMENDMENT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Verify nested policies and every cross-product constant.

        Raises:
            ValueError: If a nested policy or population cross-product differs from the review.
        """
        expected_policies = (
            (self.primary_q6_fresh_evaluation_policy, FreshEvaluationPolicy.primary_q6_pilot()),
            (self.secondary_q12_fresh_evaluation_policy, FreshEvaluationPolicy.secondary_q12_pilot()),
            (self.primary_q6_diagnostic_policy, PilotDiagnosticPolicy.primary_q6()),
            (self.secondary_q12_diagnostic_policy, PilotDiagnosticPolicy.secondary_q12()),
        )
        if any(type(actual) is not type(expected) or actual != expected for actual, expected in expected_policies):
            msg = "Operational amendment contains a changed or incomplete nested pilot policy."
            raise ValueError(msg)
        if self.execution_seed_policy_suite != ExecutionSeedPolicySuite.frozen():
            msg = "Operational amendment contains a changed execution seed-policy suite."
            raise ValueError(msg)
        if self.training_policy_universe != FrozenTrainingPolicyUniverse.frozen():
            msg = "Operational amendment contains a changed training-policy universe."
            raise ValueError(msg)
        schedule_policy = self.execution_seed_policy_suite.policy(self.training_policy_universe.schedule_seed_policy_id)
        if schedule_policy.content_checksum != self.training_policy_universe.schedule_seed_policy_checksum:
            msg = "Training universe is not bound to the amendment seed-policy suite."
            raise ValueError(msg)
        if self.primary_q6_pilot_job_count != (
            self.primary_q6_pilot_target_count * self.pilot_optimization_seed_count * self.pilot_configuration_count
        ):
            msg = "Primary-q6 pilot job count does not reproduce its Cartesian population."
            raise ValueError(msg)
        if self.secondary_q12_pilot_job_count != (
            self.secondary_q12_pilot_target_count * self.pilot_optimization_seed_count * self.pilot_configuration_count
        ):
            msg = "Secondary-q12 pilot job count does not reproduce its Cartesian population."
            raise ValueError(msg)
        if self.screen_cell_count != (
            self.screen_target_count * self.screen_optimization_seed_count * self.screen_method_count
        ):
            msg = "Screen cell count does not reproduce the nonadaptive Cartesian population."
            raise ValueError(msg)

    @property
    def pilot_optimization_seed_policy_id(self) -> str:
        """Policy bound to the ordered five-seed pilot population."""
        return PILOT_OPTIMIZATION_SEED_POLICY_ID

    @property
    def execution_seed_policy_suite_checksum(self) -> str:
        """Checksum of the complete seed suite nested in this amendment."""
        return self.execution_seed_policy_suite.content_checksum

    @property
    def pilot_optimization_seed_policy_checksum(self) -> str:
        """Checksum of the ordered five-seed pilot derivation policy."""
        return self.execution_seed_policy_suite.policy(self.pilot_optimization_seed_policy_id).content_checksum

    @property
    def screen_optimization_seed_policy_id(self) -> str:
        """Policy bound to the ordered three-seed screening population."""
        return SCREEN_OPTIMIZATION_SEED_POLICY_ID

    @property
    def screen_optimization_seed_policy_checksum(self) -> str:
        """Checksum of the ordered three-seed screening derivation policy."""
        return self.execution_seed_policy_suite.policy(self.screen_optimization_seed_policy_id).content_checksum

    @classmethod
    def frozen(cls) -> OperationalProtocolAmendment:
        """Load the checked-in reviewed prospective WP22A amendment.

        Returns:
            The trusted checked-in operational protocol amendment.
        """
        return load_operational_protocol_amendment()

    @classmethod
    def _reviewed(cls) -> OperationalProtocolAmendment:
        """Build the code-side expectation used to reject semantic drift.

        Returns:
            The exact code-side operational protocol amendment.
        """
        return cls(
            primary_q6_fresh_evaluation_policy=FreshEvaluationPolicy.primary_q6_pilot(),
            secondary_q12_fresh_evaluation_policy=FreshEvaluationPolicy.secondary_q12_pilot(),
            primary_q6_diagnostic_policy=PilotDiagnosticPolicy.primary_q6(),
            secondary_q12_diagnostic_policy=PilotDiagnosticPolicy.secondary_q12(),
            execution_seed_policy_suite=ExecutionSeedPolicySuite.frozen(),
            training_policy_universe=FrozenTrainingPolicyUniverse.frozen(),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered operational choice."""
        return {
            "schema_version": self.schema_version,
            "amendment_id": self.amendment_id,
            "preregistration_checksum": self.preregistration_checksum,
            "implementation_plan_commit": self.implementation_plan_commit,
            "prospective_status": self.prospective_status,
            "subpackage_order": list(self.subpackage_order),
            "pilot_method_ids": list(self.pilot_method_ids),
            "screen_method_ids": list(self.screen_method_ids),
            "production_update_count": self.production_update_count,
            "production_terminal_update": self.production_terminal_update,
            "training_trajectory_count": self.training_trajectory_count,
            "checkpoint_validation_trajectory_count": self.checkpoint_validation_trajectory_count,
            "checkpoint_validation_cadence": self.checkpoint_validation_cadence,
            "checkpoint_validation_updates": list(self.checkpoint_validation_updates),
            "checkpoint_selection_tie_rule": self.checkpoint_selection_tie_rule,
            "primary_methods_stop_early": self.primary_methods_stop_early,
            "optimizer_state_preserved_across_schedule_boundaries": (
                self.optimizer_state_preserved_across_schedule_boundaries
            ),
            "unsupported_composition_action": self.unsupported_composition_action,
            "resume_identity_requirement": self.resume_identity_requirement,
            "primary_q6_pilot_target_count": self.primary_q6_pilot_target_count,
            "secondary_q12_pilot_target_count": self.secondary_q12_pilot_target_count,
            "pilot_optimization_seed_count": self.pilot_optimization_seed_count,
            "pilot_optimization_seed_policy_id": self.pilot_optimization_seed_policy_id,
            "pilot_optimization_seed_policy_checksum": self.pilot_optimization_seed_policy_checksum,
            "pilot_configuration_count": self.pilot_configuration_count,
            "primary_q6_pilot_job_count": self.primary_q6_pilot_job_count,
            "secondary_q12_pilot_job_count": self.secondary_q12_pilot_job_count,
            "primary_q6_fresh_evaluation_policy": self.primary_q6_fresh_evaluation_policy.to_dict(),
            "secondary_q12_fresh_evaluation_policy": self.secondary_q12_fresh_evaluation_policy.to_dict(),
            "primary_q6_diagnostic_policy": self.primary_q6_diagnostic_policy.to_dict(),
            "secondary_q12_diagnostic_policy": self.secondary_q12_diagnostic_policy.to_dict(),
            "execution_seed_policy_suite": self.execution_seed_policy_suite.to_dict(),
            "execution_seed_policy_suite_checksum": self.execution_seed_policy_suite_checksum,
            "training_policy_universe": self.training_policy_universe.to_dict(),
            "q12_inference_eligible": self.q12_inference_eligible,
            "q12_screening_eligible": self.q12_screening_eligible,
            "q12_promotion_eligible": self.q12_promotion_eligible,
            "screen_target_count": self.screen_target_count,
            "screen_optimization_seed_count": self.screen_optimization_seed_count,
            "screen_optimization_seed_policy_id": self.screen_optimization_seed_policy_id,
            "screen_optimization_seed_policy_checksum": self.screen_optimization_seed_policy_checksum,
            "screen_method_count": self.screen_method_count,
            "screen_cell_count": self.screen_cell_count,
            "screen_adaptive": self.screen_adaptive,
            "outer_variance_method": self.outer_variance_method,
            "outer_variance_planned_job_count": self.outer_variance_planned_job_count,
            "outer_variance_alpha": self.outer_variance_alpha,
            "outer_trajectory_mcse_target": self.outer_trajectory_mcse_target,
            "outer_trajectory_count_min": self.outer_trajectory_count_min,
            "outer_trajectory_count_max": self.outer_trajectory_count_max,
            "outer_trajectory_optional_stopping": self.outer_trajectory_optional_stopping,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing the complete operational amendment."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> OperationalProtocolAmendment:
        """Decode and verify the exact reviewed WP22 operational amendment.

        Returns:
            The verified operational protocol amendment.

        Raises:
            ValueError: If the payload differs from the reviewed amendment.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_OPERATIONAL_AMENDMENT_KEYS,
            name="WP22 operational protocol amendment",
        )
        if mapping["schema_version"] != OPERATIONAL_PROTOCOL_AMENDMENT_SCHEMA_VERSION:
            msg = "Operational protocol amendment uses an unsupported schema version."
            raise ValueError(msg)
        expected = cls._reviewed()
        supplied_payload = {key: value for key, value in mapping.items() if key != "content_checksum"}
        if canonical_checksum(supplied_payload) != expected.content_checksum:
            msg = "Operational protocol amendment differs from the reviewed WP22A plan."
            raise ValueError(msg)
        amendment = cls(
            primary_q6_fresh_evaluation_policy=FreshEvaluationPolicy.from_dict(
                mapping["primary_q6_fresh_evaluation_policy"]
            ),
            secondary_q12_fresh_evaluation_policy=FreshEvaluationPolicy.from_dict(
                mapping["secondary_q12_fresh_evaluation_policy"]
            ),
            primary_q6_diagnostic_policy=PilotDiagnosticPolicy.from_dict(mapping["primary_q6_diagnostic_policy"]),
            secondary_q12_diagnostic_policy=PilotDiagnosticPolicy.from_dict(mapping["secondary_q12_diagnostic_policy"]),
            execution_seed_policy_suite=ExecutionSeedPolicySuite.from_dict(mapping["execution_seed_policy_suite"]),
            training_policy_universe=FrozenTrainingPolicyUniverse.from_dict(mapping["training_policy_universe"]),
        )
        if mapping["execution_seed_policy_suite_checksum"] != amendment.execution_seed_policy_suite.content_checksum:
            msg = "Operational amendment seed-policy suite checksum reference changed."
            raise ValueError(msg)
        if mapping["content_checksum"] != amendment.content_checksum:
            msg = "Operational protocol amendment checksum changed during normalization."
            raise ValueError(msg)
        return amendment

    @classmethod
    def from_json(cls, payload: str) -> OperationalProtocolAmendment:
        """Decode canonical JSON into the reviewed WP22 operational amendment.

        Returns:
            The verified operational protocol amendment.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def load_operational_protocol_amendment(
    path: Path = DEFAULT_OPERATIONAL_PROTOCOL_AMENDMENT_PATH,
) -> OperationalProtocolAmendment:
    """Load the checked-in WP22A amendment against its independent trust root.

    Args:
        path: Canonical amendment JSON path.

    Returns:
        The strict reviewed operational amendment.

    Raises:
        ValueError: If content or semantics differ from the reviewed seal.
    """
    amendment = OperationalProtocolAmendment.from_dict(read_canonical_json_object(path))
    if amendment.content_checksum != TRUSTED_OPERATIONAL_PROTOCOL_AMENDMENT_CHECKSUM:
        msg = (
            "Operational protocol amendment does not match its trusted checksum: "
            f"expected {TRUSTED_OPERATIONAL_PROTOCOL_AMENDMENT_CHECKSUM}, "
            f"got {amendment.content_checksum}."
        )
        raise ValueError(msg)
    return amendment


__all__ = [
    "CHECKPOINT_VALIDATION_CADENCE",
    "CHECKPOINT_VALIDATION_TRAJECTORY_COUNT",
    "CHECKPOINT_VALIDATION_UPDATES",
    "DEFAULT_OPERATIONAL_PROTOCOL_AMENDMENT_PATH",
    "FRESH_EVALUATION_POLICY_SCHEMA_VERSION",
    "OPERATIONAL_PROTOCOL_AMENDMENT_SCHEMA_VERSION",
    "OPERATOR_GROWTH_EXECUTION_SPEC_SCHEMA_VERSION",
    "OUTER_TRAJECTORY_COUNT_MAX",
    "OUTER_TRAJECTORY_COUNT_MIN",
    "PILOT_DIAGNOSTIC_POLICY_SCHEMA_VERSION",
    "PILOT_METHOD_IDS",
    "PRIMARY_Q6_DIAGNOSTIC_VECTOR_COUNT",
    "PRIMARY_Q6_PILOT_JOB_COUNT",
    "PRIMARY_Q6_PILOT_PREFIXES",
    "PRIMARY_Q6_PILOT_TRAJECTORY_COUNT",
    "PRODUCTION_TRAINING_TRAJECTORY_COUNT",
    "PRODUCTION_UPDATE_COUNT",
    "SCREEN_CELL_COUNT",
    "SCREEN_METHOD_IDS",
    "SECONDARY_Q12_PILOT_JOB_COUNT",
    "SECONDARY_Q12_PILOT_TRAJECTORY_COUNT",
    "TRUSTED_OPERATIONAL_PROTOCOL_AMENDMENT_CHECKSUM",
    "WP22_IMPLEMENTATION_PLAN_COMMIT",
    "FreshEvaluationPolicy",
    "OperationalProtocolAmendment",
    "OperatorGrowthExecutionSpec",
    "PilotDiagnosticPolicy",
    "bounded_outer_trajectory_count",
    "load_operational_protocol_amendment",
]
