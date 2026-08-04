# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Checksum-sealed WP22C schedule execution and exact restart.

The engine in this module owns optimizer-update scheduling only.  It supplies
an executor with the exact noise strength, trajectory membership, mixture
assignment, and checkpoint request for each update of the binding's declared
controlled stage.  Structural circuit growth and pruning remain outside this
engine and retain their independently sealed semantics.

The callback boundary deliberately has no final-test input.  Checkpoint
selection accepts only :class:`~.training_schedules.ValidationCheckpoint`
records, so confirmatory evidence cannot influence optimizer state, early
stopping, or multistart selection.

WP22 uses zero-based *optimizer-update* coordinates: checkpoint 0 observes the
state after optimizer update 0, and checkpoint 199 observes the terminal state
after the 200th update.  This intentionally differs from the older WP17--WP21
global-iteration convention, which labeled the initialized state iteration 0.
"""

# Strict persisted schemas repeat validation at every decode and construction
# boundary; enumerating every delegated validator exception would obscure the
# scientific contracts documented by the records themselves.
# ruff: noqa: DOC501

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast

import numpy as np

from .binding_catalog import ExecutableScopedBinding
from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .competitor_optimizers import ParameterShiftAdamConfig, SPSAConfig
from .execution_bindings import EnergyAdaptSmokeSpec, OperatorGrowthSmokeSpec, PipelineSmokeSpec
from .execution_protocol import (
    CHECKPOINT_VALIDATION_UPDATES,
    OperatorGrowthExecutionSpec,
)
from .implementation_catalog import PipelineSmokeRuntimeProgram
from .operator_growth import CandidateGradient, OperatorGrowthSpec, OperatorPoolSpec, PoolOperator
from .pipeline import TrainingPipelineTemplate
from .training_schedules import (
    CheckpointValidationTracker,
    FrozenTrainingPolicyUniverse,
    MultistartSeedBundle,
    NoiseMixtureAllocation,
    TrainingStrategySchedule,
    TrajectoryEnsembleMembership,
    TrajectorySamplingPolicy,
    ValidationCheckpoint,
    build_trajectory_membership,
    derive_role_seed,
)
from .validation import (
    require_checksum,
    require_float,
    require_int,
    require_slug,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .pipeline import TrainingStageTemplate

SCHEDULED_JOB_SEED_SET_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_job_seed_set.v1"
COMPONENT_TRAJECTORY_MEMBERSHIP_SCHEMA_VERSION = "yaqs.state_preparation.phase2.component_trajectory_membership.v1"
SCHEDULED_UPDATE_POLICY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_update_policy.v1"
SCHEDULED_TRAINING_POLICY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_training_policy.v1"
SCHEDULED_EXECUTION_PROGRAM_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_execution_program.v1"
OPTIMIZER_INITIALIZATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.optimizer_initialization.v1"
KROTOV_OPTIMIZER_PAYLOAD_SCHEMA_VERSION = "yaqs.state_preparation.phase2.krotov_optimizer_payload.v1"
ADAM_OPTIMIZER_PAYLOAD_SCHEMA_VERSION = "yaqs.state_preparation.phase2.adam_optimizer_payload.v1"
SPSA_OPTIMIZER_PAYLOAD_SCHEMA_VERSION = "yaqs.state_preparation.phase2.spsa_optimizer_payload.v1"
OPERATOR_GROWTH_OPTIMIZER_PAYLOAD_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_optimizer_payload.v1"
SCHEDULED_UPDATE_REQUEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_update_request.v1"
SCHEDULED_UPDATE_RESULT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_update_result.v1"
SCHEDULED_TRAINING_OBJECTIVE_REQUEST_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.scheduled_training_objective_request.v1"
)
SCHEDULED_TRAINING_OBJECTIVE_RESULT_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.scheduled_training_objective_result.v1"
)
SCHEDULED_TRAINING_GRADIENT_REQUEST_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.scheduled_training_gradient_request.v1"
)
SCHEDULED_TRAINING_GRADIENT_RESULT_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.scheduled_training_gradient_result.v1"
)
SCHEDULED_VALIDATION_REQUEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_validation_request.v1"
SCHEDULED_VALIDATION_RESULT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_validation_result.v1"
PARAMETER_CHECKPOINT_ARTIFACT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.parameter_checkpoint_artifact.v1"
SCHEDULED_UPDATE_RECEIPT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_update_receipt.v1"
SCHEDULED_OPTIMIZER_STATE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_optimizer_state.v1"
MULTISTART_START_EVIDENCE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.multistart_start_evidence.v1"
MULTISTART_WORK_EVIDENCE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.multistart_work_evidence.v1"
SCHEDULED_EXECUTION_SNAPSHOT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.scheduled_execution_snapshot.v1"
OPERATOR_GROWTH_SELECTION_REQUEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_selection_request.v1"
OPERATOR_GROWTH_SELECTION_RESULT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_selection_result.v1"
OPERATOR_GROWTH_PREFIX_TRANSITION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_prefix_transition.v1"
OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_REQUEST_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.operator_growth_segmented_objective_request.v1"
)
OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_RESULT_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.operator_growth_segmented_objective_result.v1"
)
OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_EVIDENCE_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.operator_growth_segmented_objective_evidence.v1"
)
OPERATOR_GROWTH_SEGMENTED_UPDATE_RECEIPT_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.operator_growth_segmented_update_receipt.v1"
)
OPERATOR_GROWTH_PREFIX_VALIDATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_prefix_validation.v1"
OPERATOR_GROWTH_SEGMENTED_SNAPSHOT_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.operator_growth_segmented_snapshot.v1"
)

OPERATOR_GROWTH_PREFIX_UPDATE_COUNT = 100
OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT = 200
OPERATOR_GROWTH_TRAINING_TRAJECTORY_COUNT = 8
OPERATOR_GROWTH_VALIDATION_TRAJECTORY_COUNT = 256

ExecutionScope = Literal["scoped_binding", "development_schedule_trace"]
TrainingPhase = Literal["noiseless_pretrain", "noisy_finetune"]
TerminalReason = Literal["budget_complete", "validation_early_stop"]
OptimizerKind = Literal["krotov", "parameter_shift_adam", "spsa", "operator_growth_adam"]
InitializerKind = Literal["normal_pcg64", "sealed_warm_start"]
LearningRateSchedule = Literal["constant", "inverse", "exp"]
ObjectiveEvaluationKind = Literal["gradient_plus", "gradient_minus"]

_EXECUTION_SCOPES = frozenset({"scoped_binding", "development_schedule_trace"})
_TERMINAL_REASONS = frozenset({"budget_complete", "validation_early_stop"})
_OPTIMIZER_KINDS = frozenset({"krotov", "parameter_shift_adam", "spsa", "operator_growth_adam"})
_SUPPORTED_IMPLEMENTATION_KINDS = frozenset({
    "phase2_pipeline",
    "phase2_pipeline_smoke",
    "operator_growth",
    "operator_growth_smoke",
    "tfim_operator_growth",
})
_OPERATOR_GROWTH_IMPLEMENTATION_KINDS = frozenset({
    "operator_growth",
    "operator_growth_smoke",
    "tfim_operator_growth",
})


def _sealed(payload: dict[str, object]) -> dict[str, object]:
    """Return a detached payload with its canonical checksum."""
    return {**payload, "content_checksum": canonical_checksum(payload)}


def _verify(
    value: object,
    *,
    keys: frozenset[str],
    version: str,
    name: str,
) -> Mapping[str, object]:
    """Verify one strict sealed record and schema version.

    Returns:
        The verified immutable record mapping.
    """
    mapping = verify_sealed_mapping(value, expected_keys=keys, name=name)
    if mapping["schema_version"] != version:
        msg = f"{name} uses an unsupported schema version."
        raise ValueError(msg)
    return mapping


def _uint64(value: object, name: str) -> int:
    """Return one strict unsigned 64-bit integer."""
    result = require_int(value, name)
    if result >= 2**64:
        msg = f"{name} must fit an unsigned 64-bit integer."
        raise ValueError(msg)
    return result


def _mapping_checksum(value: Mapping[str, object]) -> str:
    """Return the canonical checksum of one already frozen mapping."""
    return canonical_checksum(thaw_json_mapping(value))


def _float_tuple(value: object, name: str, *, length: int | None = None) -> tuple[float, ...]:
    """Return one strict finite float tuple.

    Returns:
        The normalized immutable vector.
    """
    if type(value) is not tuple:
        msg = f"{name} must be a tuple."
        raise TypeError(msg)
    result = tuple(require_float(item, f"{name}[{index}]") for index, item in enumerate(value))
    if not result:
        msg = f"{name} must be nonempty."
        raise ValueError(msg)
    if length is not None and len(result) != length:
        msg = f"{name} must contain exactly {length} values."
        raise ValueError(msg)
    return result


def _possibly_empty_float_tuple(
    value: object,
    name: str,
    *,
    length: int | None = None,
) -> tuple[float, ...]:
    """Return a strict finite float tuple whose empty value is meaningful."""
    if type(value) is not tuple:
        msg = f"{name} must be a tuple."
        raise TypeError(msg)
    result = tuple(require_float(item, f"{name}[{index}]") for index, item in enumerate(value))
    if length is not None and len(result) != length:
        msg = f"{name} must contain exactly {length} values."
        raise ValueError(msg)
    return result


def _vector_checksum(parameters: tuple[float, ...]) -> str:
    """Return the canonical recoverable float-vector checksum."""
    return canonical_checksum({"dtype": "float64", "parameters": list(parameters)})


def _zeros(length: int) -> tuple[float, ...]:
    """Return one immutable all-zero vector."""
    return (0.0,) * length


@dataclass(frozen=True, slots=True)
class ScheduledJobSeedSet:
    """Root seed material for one binding-target-optimization job.

    The checkpoint root is derived, rather than supplied, so callers cannot
    introduce an unreviewed validation stream.  Multistart initialization,
    ordering, and training roots are derived by the sealed schedule plan.
    """

    optimization_seed: int
    schema_version: str = field(default=SCHEDULED_JOB_SEED_SET_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the exact unsigned root seed."""
        object.__setattr__(self, "optimization_seed", _uint64(self.optimization_seed, "optimization_seed"))

    @property
    def checkpoint_validation_seed(self) -> int:
        """Common validation root shared fairly by all optimizer starts."""
        return derive_role_seed(
            self.optimization_seed,
            "checkpoint_validation",
            purpose="scheduled_checkpoint_validation",
        )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered seed field."""
        return {
            "schema_version": self.schema_version,
            "optimization_seed": self.optimization_seed,
            "checkpoint_validation_seed": self.checkpoint_validation_seed,
            "checkpoint_sharing_rule": "common_across_optimizer_starts",
            "final_test_seed_access": "forbidden",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete job seed set."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ScheduledJobSeedSet:
        """Decode and verify one strict job seed set.

        Returns:
            The verified seed set.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "optimization_seed",
                "checkpoint_validation_seed",
                "checkpoint_sharing_rule",
                "final_test_seed_access",
                "content_checksum",
            }),
            version=SCHEDULED_JOB_SEED_SET_SCHEMA_VERSION,
            name="scheduled job seed set",
        )
        result = cls(optimization_seed=cast("int", mapping["optimization_seed"]))
        if (
            mapping["checkpoint_validation_seed"] != result.checkpoint_validation_seed
            or mapping["checkpoint_sharing_rule"] != "common_across_optimizer_starts"
            or mapping["final_test_seed_access"] != "forbidden"
        ):
            msg = "Scheduled job seed derivation or final-test isolation changed."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> ScheduledJobSeedSet:
        """Decode canonical JSON into a verified job seed set.

        Returns:
            The verified seed set.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class ComponentTrajectoryMembership:
    """Exact ordered assignment of aggregate trajectories to one noise component."""

    update: int
    component_index: int
    noise_id: str
    weight: float
    allocation_checksum: str
    aggregate_membership_checksum: str
    member_seeds: tuple[int, ...]
    predecessor_checksum: str | None
    schema_version: str = field(default=COMPONENT_TRAJECTORY_MEMBERSHIP_SCHEMA_VERSION, init=False)
    _cached_content_checksum: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate component identity, exact seeds, and predecessor link."""
        object.__setattr__(self, "update", require_int(self.update, "update"))
        object.__setattr__(self, "component_index", require_int(self.component_index, "component_index"))
        object.__setattr__(self, "noise_id", require_slug(self.noise_id, "noise_id"))
        object.__setattr__(self, "weight", require_float(self.weight, "weight", minimum=0.0, maximum=1.0))
        object.__setattr__(
            self,
            "allocation_checksum",
            require_checksum(self.allocation_checksum, "allocation_checksum"),
        )
        object.__setattr__(
            self,
            "aggregate_membership_checksum",
            require_checksum(self.aggregate_membership_checksum, "aggregate_membership_checksum"),
        )
        if type(self.member_seeds) is not tuple or not self.member_seeds:
            msg = "member_seeds must be a nonempty tuple."
            raise TypeError(msg)
        members = tuple(_uint64(seed, "member_seed") for seed in self.member_seeds)
        if len(set(members)) != len(members):
            msg = "Component member seeds must be unique."
            raise ValueError(msg)
        object.__setattr__(self, "member_seeds", members)
        if self.predecessor_checksum is not None:
            object.__setattr__(
                self,
                "predecessor_checksum",
                require_checksum(self.predecessor_checksum, "predecessor_checksum"),
            )
        object.__setattr__(self, "_cached_content_checksum", canonical_checksum(self._payload()))

    @property
    def trajectory_count(self) -> int:
        """Number of trajectories assigned to this component."""
        return len(self.member_seeds)

    @property
    def seed_domain(self) -> str:
        """Stable component-local seed-domain label."""
        return f"training_trajectory.mixture.{self.noise_id}"

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered component membership field."""
        return {
            "schema_version": self.schema_version,
            "update": self.update,
            "component_index": self.component_index,
            "noise_id": self.noise_id,
            "weight": self.weight,
            "seed_domain": self.seed_domain,
            "allocation_checksum": self.allocation_checksum,
            "aggregate_membership_checksum": self.aggregate_membership_checksum,
            "trajectory_count": self.trajectory_count,
            "member_seeds": list(self.member_seeds),
            "predecessor_checksum": self.predecessor_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering exact component-local membership."""
        return self._cached_content_checksum

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return {**self._payload(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> ComponentTrajectoryMembership:
        """Decode and verify one exact component membership.

        Returns:
            The verified component membership.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "update",
                "component_index",
                "noise_id",
                "weight",
                "seed_domain",
                "allocation_checksum",
                "aggregate_membership_checksum",
                "trajectory_count",
                "member_seeds",
                "predecessor_checksum",
                "content_checksum",
            }),
            version=COMPONENT_TRAJECTORY_MEMBERSHIP_SCHEMA_VERSION,
            name="component trajectory membership",
        )
        raw_members = mapping["member_seeds"]
        if type(raw_members) is not tuple:
            msg = "member_seeds must be a JSON array."
            raise TypeError(msg)
        result = cls(
            update=cast("int", mapping["update"]),
            component_index=cast("int", mapping["component_index"]),
            noise_id=cast("str", mapping["noise_id"]),
            weight=cast("float", mapping["weight"]),
            allocation_checksum=cast("str", mapping["allocation_checksum"]),
            aggregate_membership_checksum=cast("str", mapping["aggregate_membership_checksum"]),
            member_seeds=cast("tuple[int, ...]", raw_members),
            predecessor_checksum=cast("str | None", mapping["predecessor_checksum"]),
        )
        if mapping["trajectory_count"] != result.trajectory_count or mapping["seed_domain"] != result.seed_domain:
            msg = "Component membership derived count or seed domain changed."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class ScheduledUpdatePolicy:
    """One fully compiled, executable optimizer-update policy."""

    schedule_checksum: str
    controlled_stage_id: str
    controlled_stage_checksum: str
    start_index: int
    update: int
    phase: TrainingPhase
    noise_strength_scale: float
    trajectory_count: int
    sampling_epoch: int | None
    mixture_allocation: NoiseMixtureAllocation | None
    training_membership: TrajectoryEnsembleMembership | None
    component_memberships: tuple[ComponentTrajectoryMembership, ...]
    checkpoint_due: bool
    checkpoint_membership: TrajectoryEnsembleMembership | None
    schema_version: str = field(default=SCHEDULED_UPDATE_POLICY_SCHEMA_VERSION, init=False)
    _cached_content_checksum: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate exact aggregate, component, and checkpoint membership."""
        object.__setattr__(self, "schedule_checksum", require_checksum(self.schedule_checksum, "schedule_checksum"))
        object.__setattr__(self, "controlled_stage_id", require_slug(self.controlled_stage_id, "controlled_stage_id"))
        object.__setattr__(
            self,
            "controlled_stage_checksum",
            require_checksum(self.controlled_stage_checksum, "controlled_stage_checksum"),
        )
        object.__setattr__(self, "start_index", require_int(self.start_index, "start_index"))
        object.__setattr__(self, "update", require_int(self.update, "update"))
        if self.phase not in {"noiseless_pretrain", "noisy_finetune"}:
            msg = "phase must be noiseless_pretrain or noisy_finetune."
            raise ValueError(msg)
        scale = require_float(self.noise_strength_scale, "noise_strength_scale", minimum=0.0)
        object.__setattr__(self, "noise_strength_scale", scale)
        count = require_int(self.trajectory_count, "trajectory_count")
        object.__setattr__(self, "trajectory_count", count)
        if count == 0:
            if (
                self.sampling_epoch is not None
                or self.mixture_allocation is not None
                or self.training_membership is not None
                or self.component_memberships
            ):
                msg = "A zero-trajectory update cannot carry sampling or mixture membership."
                raise ValueError(msg)
        else:
            if self.sampling_epoch is None:
                msg = "A sampled update requires its exact sampling epoch."
                raise ValueError(msg)
            object.__setattr__(self, "sampling_epoch", require_int(self.sampling_epoch, "sampling_epoch"))
            if not isinstance(self.training_membership, TrajectoryEnsembleMembership):
                msg = "A sampled update requires exact aggregate trajectory membership."
                raise TypeError(msg)
            if not isinstance(self.mixture_allocation, NoiseMixtureAllocation):
                msg = "A sampled update requires its exact noise-mixture allocation."
                raise TypeError(msg)
            membership = self.training_membership
            allocation = self.mixture_allocation
            if (
                membership.update != self.update
                or membership.epoch != self.sampling_epoch
                or membership.trajectory_count != count
                or allocation.trajectory_count != count
            ):
                msg = "Aggregate membership, allocation, epoch, or update is inconsistent."
                raise ValueError(msg)
            components = self.component_memberships
            if type(components) is not tuple or any(
                not isinstance(component, ComponentTrajectoryMembership) for component in components
            ):
                msg = "component_memberships must be a tuple of exact component records."
                raise TypeError(msg)
            if (
                tuple(component.noise_id for component in components) != allocation.component_ids
                or tuple(component.trajectory_count for component in components) != allocation.component_counts
                or tuple(component.component_index for component in components) != tuple(range(len(components)))
                or any(component.update != self.update for component in components)
                or any(component.allocation_checksum != allocation.content_checksum for component in components)
                or any(
                    component.aggregate_membership_checksum != membership.content_checksum for component in components
                )
                or tuple(seed for component in components for seed in component.member_seeds) != membership.member_seeds
            ):
                msg = "Component-local memberships do not exactly partition the aggregate ensemble."
                raise ValueError(msg)
        if type(self.checkpoint_due) is not bool:
            msg = "checkpoint_due must be a bool."
            raise TypeError(msg)
        if self.checkpoint_due != (self.checkpoint_membership is not None):
            msg = "Checkpoint membership is present exactly at scheduled checkpoint updates."
            raise ValueError(msg)
        if self.checkpoint_membership is not None and self.checkpoint_membership.role != "checkpoint_validation":
            msg = "Checkpoint membership must use the checkpoint_validation role."
            raise ValueError(msg)
        object.__setattr__(self, "_cached_content_checksum", canonical_checksum(self._payload()))

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered update-policy field."""
        return {
            "schema_version": self.schema_version,
            "schedule_checksum": self.schedule_checksum,
            "controlled_stage_id": self.controlled_stage_id,
            "controlled_stage_checksum": self.controlled_stage_checksum,
            "schedule_application": "declared_controlled_stage_only",
            "start_index": self.start_index,
            "update": self.update,
            "phase": self.phase,
            "noise_strength_scale": self.noise_strength_scale,
            "trajectory_count": self.trajectory_count,
            "sampling_epoch": self.sampling_epoch,
            "mixture_allocation": None if self.mixture_allocation is None else self.mixture_allocation.to_dict(),
            "training_membership": None if self.training_membership is None else self.training_membership.to_dict(),
            "component_memberships": [item.to_dict() for item in self.component_memberships],
            "checkpoint_due": self.checkpoint_due,
            "checkpoint_timing": "after_corresponding_zero_based_update",
            "checkpoint_membership": (
                None if self.checkpoint_membership is None else self.checkpoint_membership.to_dict()
            ),
            "final_test_access": "forbidden",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering this exact optimizer-update policy."""
        return self._cached_content_checksum

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return {**self._payload(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> ScheduledUpdatePolicy:
        """Decode and verify one exact optimizer-update policy.

        Returns:
            The verified update policy.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "schedule_checksum",
                "controlled_stage_id",
                "controlled_stage_checksum",
                "schedule_application",
                "start_index",
                "update",
                "phase",
                "noise_strength_scale",
                "trajectory_count",
                "sampling_epoch",
                "mixture_allocation",
                "training_membership",
                "component_memberships",
                "checkpoint_due",
                "checkpoint_timing",
                "checkpoint_membership",
                "final_test_access",
                "content_checksum",
            }),
            version=SCHEDULED_UPDATE_POLICY_SCHEMA_VERSION,
            name="scheduled update policy",
        )
        if (
            mapping["schedule_application"] != "declared_controlled_stage_only"
            or mapping["checkpoint_timing"] != "after_corresponding_zero_based_update"
            or mapping["final_test_access"] != "forbidden"
        ):
            msg = "Scheduled update stage, checkpoint timing, or final-test isolation changed."
            raise ValueError(msg)
        raw_components = mapping["component_memberships"]
        if type(raw_components) is not tuple:
            msg = "component_memberships must be a JSON array."
            raise TypeError(msg)
        raw_allocation = mapping["mixture_allocation"]
        raw_training = mapping["training_membership"]
        raw_checkpoint = mapping["checkpoint_membership"]
        return cls(
            schedule_checksum=cast("str", mapping["schedule_checksum"]),
            controlled_stage_id=cast("str", mapping["controlled_stage_id"]),
            controlled_stage_checksum=cast("str", mapping["controlled_stage_checksum"]),
            start_index=cast("int", mapping["start_index"]),
            update=cast("int", mapping["update"]),
            phase=cast("TrainingPhase", mapping["phase"]),
            noise_strength_scale=cast("float", mapping["noise_strength_scale"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            sampling_epoch=cast("int | None", mapping["sampling_epoch"]),
            mixture_allocation=(None if raw_allocation is None else NoiseMixtureAllocation.from_dict(raw_allocation)),
            training_membership=(
                None if raw_training is None else TrajectoryEnsembleMembership.from_dict(raw_training)
            ),
            component_memberships=tuple(ComponentTrajectoryMembership.from_dict(item) for item in raw_components),
            checkpoint_due=cast("bool", mapping["checkpoint_due"]),
            checkpoint_membership=(
                None if raw_checkpoint is None else TrajectoryEnsembleMembership.from_dict(raw_checkpoint)
            ),
        )


@dataclass(frozen=True, slots=True)
class ScheduledTrainingPolicy:
    """Training-only projection of one compiled update policy.

    This is the sole policy object exposed to optimizer and objective adapters.
    It intentionally has no checkpoint flag, validation membership, validation
    seed, or generic data-role field.
    """

    schedule_checksum: str
    controlled_stage_id: str
    controlled_stage_checksum: str
    start_index: int
    update: int
    phase: TrainingPhase
    noise_strength_scale: float
    trajectory_count: int
    sampling_epoch: int | None
    mixture_allocation: NoiseMixtureAllocation | None
    training_membership: TrajectoryEnsembleMembership | None
    component_memberships: tuple[ComponentTrajectoryMembership, ...]
    schema_version: str = field(default=SCHEDULED_TRAINING_POLICY_SCHEMA_VERSION, init=False)
    _cached_content_checksum: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reconstruct and validate the training half of a policy."""
        shadow = ScheduledUpdatePolicy(
            schedule_checksum=self.schedule_checksum,
            controlled_stage_id=self.controlled_stage_id,
            controlled_stage_checksum=self.controlled_stage_checksum,
            start_index=self.start_index,
            update=self.update,
            phase=self.phase,
            noise_strength_scale=self.noise_strength_scale,
            trajectory_count=self.trajectory_count,
            sampling_epoch=self.sampling_epoch,
            mixture_allocation=self.mixture_allocation,
            training_membership=self.training_membership,
            component_memberships=self.component_memberships,
            checkpoint_due=False,
            checkpoint_membership=None,
        )
        for name in (
            "schedule_checksum",
            "controlled_stage_id",
            "controlled_stage_checksum",
            "start_index",
            "update",
            "phase",
            "noise_strength_scale",
            "trajectory_count",
            "sampling_epoch",
            "mixture_allocation",
            "training_membership",
            "component_memberships",
        ):
            object.__setattr__(self, name, getattr(shadow, name))
        object.__setattr__(self, "_cached_content_checksum", canonical_checksum(self._payload()))

    @classmethod
    def from_compiled(cls, policy: ScheduledUpdatePolicy) -> ScheduledTrainingPolicy:
        """Project one internal compiled policy onto training-only fields.

        Returns:
            The validation-blind training policy.
        """
        if not isinstance(policy, ScheduledUpdatePolicy):
            msg = "policy must be a ScheduledUpdatePolicy."
            raise TypeError(msg)
        return cls(
            schedule_checksum=policy.schedule_checksum,
            controlled_stage_id=policy.controlled_stage_id,
            controlled_stage_checksum=policy.controlled_stage_checksum,
            start_index=policy.start_index,
            update=policy.update,
            phase=policy.phase,
            noise_strength_scale=policy.noise_strength_scale,
            trajectory_count=policy.trajectory_count,
            sampling_epoch=policy.sampling_epoch,
            mixture_allocation=policy.mixture_allocation,
            training_membership=policy.training_membership,
            component_memberships=policy.component_memberships,
        )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered training-only field."""
        return {
            "schema_version": self.schema_version,
            "schedule_checksum": self.schedule_checksum,
            "controlled_stage_id": self.controlled_stage_id,
            "controlled_stage_checksum": self.controlled_stage_checksum,
            "start_index": self.start_index,
            "update": self.update,
            "phase": self.phase,
            "noise_strength_scale": self.noise_strength_scale,
            "trajectory_count": self.trajectory_count,
            "sampling_epoch": self.sampling_epoch,
            "mixture_allocation": None if self.mixture_allocation is None else self.mixture_allocation.to_dict(),
            "training_membership": None if self.training_membership is None else self.training_membership.to_dict(),
            "component_memberships": [item.to_dict() for item in self.component_memberships],
            "accessible_data_role": "training_trajectory",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the validation-blind policy."""
        return self._cached_content_checksum

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return {**self._payload(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> ScheduledTrainingPolicy:
        """Decode one strict training-only policy.

        Returns:
            The verified policy.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "schedule_checksum",
                "controlled_stage_id",
                "controlled_stage_checksum",
                "start_index",
                "update",
                "phase",
                "noise_strength_scale",
                "trajectory_count",
                "sampling_epoch",
                "mixture_allocation",
                "training_membership",
                "component_memberships",
                "accessible_data_role",
                "content_checksum",
            }),
            version=SCHEDULED_TRAINING_POLICY_SCHEMA_VERSION,
            name="scheduled training policy",
        )
        if mapping["accessible_data_role"] != "training_trajectory":
            msg = "Scheduled training policy may access only training trajectories."
            raise ValueError(msg)
        raw_components = mapping["component_memberships"]
        if type(raw_components) is not tuple:
            msg = "component_memberships must be a JSON array."
            raise TypeError(msg)
        raw_allocation = mapping["mixture_allocation"]
        raw_membership = mapping["training_membership"]
        return cls(
            schedule_checksum=cast("str", mapping["schedule_checksum"]),
            controlled_stage_id=cast("str", mapping["controlled_stage_id"]),
            controlled_stage_checksum=cast("str", mapping["controlled_stage_checksum"]),
            start_index=cast("int", mapping["start_index"]),
            update=cast("int", mapping["update"]),
            phase=cast("TrainingPhase", mapping["phase"]),
            noise_strength_scale=cast("float", mapping["noise_strength_scale"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            sampling_epoch=cast("int | None", mapping["sampling_epoch"]),
            mixture_allocation=(None if raw_allocation is None else NoiseMixtureAllocation.from_dict(raw_allocation)),
            training_membership=(
                None if raw_membership is None else TrajectoryEnsembleMembership.from_dict(raw_membership)
            ),
            component_memberships=tuple(ComponentTrajectoryMembership.from_dict(item) for item in raw_components),
        )


def _component_memberships(
    *,
    schedule: TrainingStrategySchedule,
    update: int,
    allocation: NoiseMixtureAllocation,
    aggregate: TrajectoryEnsembleMembership,
    previous: Mapping[str, ComponentTrajectoryMembership],
) -> tuple[ComponentTrajectoryMembership, ...]:
    """Build the exact ordered partition of one aggregate ensemble.

    Returns:
        Component-local memberships in the mixture's declared order.
    """
    components: list[ComponentTrajectoryMembership] = []
    cursor = 0
    weights = {component.noise_id: component.weight for component in schedule.training_noise.components}
    for component_index, (noise_id, count) in enumerate(
        zip(allocation.component_ids, allocation.component_counts, strict=True)
    ):
        member_seeds = aggregate.member_seeds[cursor : cursor + count]
        cursor += count
        predecessor = previous.get(noise_id)
        components.append(
            ComponentTrajectoryMembership(
                update=update,
                component_index=component_index,
                noise_id=noise_id,
                weight=weights[noise_id],
                allocation_checksum=allocation.content_checksum,
                aggregate_membership_checksum=aggregate.content_checksum,
                member_seeds=member_seeds,
                predecessor_checksum=None if predecessor is None else predecessor.content_checksum,
            )
        )
    return tuple(components)


def _compile_update_policies(
    *,
    schedule: TrainingStrategySchedule,
    controlled_stage_id: str,
    controlled_stage_checksum: str,
    job_seeds: ScheduledJobSeedSet,
    start_seed_bundles: tuple[MultistartSeedBundle, ...],
    checkpoint_validation_trajectory_count: int,
    checkpoint_updates: tuple[int, ...] = CHECKPOINT_VALIDATION_UPDATES,
) -> tuple[ScheduledUpdatePolicy, ...]:
    """Deterministically compile every optimizer start and update.

    Returns:
        Policies ordered by optimizer start and then update.
    """
    validation_membership = None
    if checkpoint_validation_trajectory_count:
        validation_membership = build_trajectory_membership(
            TrajectorySamplingPolicy("fixed_crn"),
            master_seed=job_seeds.checkpoint_validation_seed,
            role="checkpoint_validation",
            update=0,
            trajectory_count=checkpoint_validation_trajectory_count,
            allow_stream_start=False,
        )
    policies: list[ScheduledUpdatePolicy] = []
    for bundle in start_seed_bundles:
        previous_training: TrajectoryEnsembleMembership | None = None
        previous_components: dict[str, ComponentTrajectoryMembership] = {}
        for update in range(schedule.phase_boundary.total_updates):
            count = schedule.trajectory_curriculum.count_at(update)
            membership: TrajectoryEnsembleMembership | None = None
            allocation: NoiseMixtureAllocation | None = None
            components: tuple[ComponentTrajectoryMembership, ...] = ()
            epoch: int | None = None
            if count:
                membership = build_trajectory_membership(
                    schedule.sampling_policy,
                    master_seed=bundle.training_trajectory_seed,
                    role="training_trajectory",
                    update=update,
                    trajectory_count=count,
                    stream_index=bundle.start_index,
                    previous=previous_training,
                    allow_stream_start=previous_training is None,
                )
                allocation = schedule.training_noise.allocate(count)
                components = _component_memberships(
                    schedule=schedule,
                    update=update,
                    allocation=allocation,
                    aggregate=membership,
                    previous=previous_components,
                )
                previous_training = membership
                previous_components = {component.noise_id: component for component in components}
                epoch = membership.epoch
            checkpoint_due = validation_membership is not None and update in checkpoint_updates
            policies.append(
                ScheduledUpdatePolicy(
                    schedule_checksum=schedule.content_checksum,
                    controlled_stage_id=controlled_stage_id,
                    controlled_stage_checksum=controlled_stage_checksum,
                    start_index=bundle.start_index,
                    update=update,
                    phase=schedule.phase_boundary.phase_at(update),
                    noise_strength_scale=schedule.noise_continuation.strength_at(update),
                    trajectory_count=count,
                    sampling_epoch=epoch,
                    mixture_allocation=allocation,
                    training_membership=membership,
                    component_memberships=components,
                    checkpoint_due=checkpoint_due,
                    checkpoint_membership=validation_membership if checkpoint_due else None,
                )
            )
    return tuple(policies)


@dataclass(frozen=True, slots=True)
class ScheduledExecutionProgram:
    """Complete deterministic schedule program for one bound optimization job."""

    execution_scope: ExecutionScope
    binding_checksum: str
    implementation_checksum: str
    publication_method_id: str
    controlled_stage_id: str
    controlled_stage_checksum: str
    executable_binding: ExecutableScopedBinding | None
    normalized_compute_cap: float | None
    schedule: TrainingStrategySchedule
    job_seeds: ScheduledJobSeedSet
    checkpoint_validation_trajectory_count: int
    start_seed_bundles: tuple[MultistartSeedBundle, ...]
    update_policies: tuple[ScheduledUpdatePolicy, ...]
    schema_version: str = field(default=SCHEDULED_EXECUTION_PROGRAM_SCHEMA_VERSION, init=False)
    _cached_content_checksum: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Require a complete deterministic Cartesian update program."""
        if self.execution_scope not in _EXECUTION_SCOPES:
            msg = "execution_scope is not a supported scheduled-program scope."
            raise ValueError(msg)
        object.__setattr__(self, "binding_checksum", require_checksum(self.binding_checksum, "binding_checksum"))
        object.__setattr__(
            self,
            "implementation_checksum",
            require_checksum(self.implementation_checksum, "implementation_checksum"),
        )
        object.__setattr__(
            self, "publication_method_id", require_slug(self.publication_method_id, "publication_method_id")
        )
        object.__setattr__(self, "controlled_stage_id", require_slug(self.controlled_stage_id, "controlled_stage_id"))
        object.__setattr__(
            self,
            "controlled_stage_checksum",
            require_checksum(self.controlled_stage_checksum, "controlled_stage_checksum"),
        )
        if self.normalized_compute_cap is not None:
            object.__setattr__(
                self,
                "normalized_compute_cap",
                require_float(self.normalized_compute_cap, "normalized_compute_cap", minimum=0.0),
            )
        if not isinstance(self.schedule, TrainingStrategySchedule):
            msg = "schedule must be a TrainingStrategySchedule."
            raise TypeError(msg)
        if not isinstance(self.job_seeds, ScheduledJobSeedSet):
            msg = "job_seeds must be a ScheduledJobSeedSet."
            raise TypeError(msg)
        checkpoint_count = require_int(
            self.checkpoint_validation_trajectory_count,
            "checkpoint_validation_trajectory_count",
        )
        object.__setattr__(self, "checkpoint_validation_trajectory_count", checkpoint_count)
        if checkpoint_count and self.schedule.phase_boundary.total_updates != 200:
            msg = "Checkpoint observations are supported only for the frozen 200-update cadence."
            raise ValueError(msg)
        bundles = self.start_seed_bundles
        if type(bundles) is not tuple or any(not isinstance(bundle, MultistartSeedBundle) for bundle in bundles):
            msg = "start_seed_bundles must be a tuple of MultistartSeedBundle records."
            raise TypeError(msg)
        expected_bundles = self.schedule.multistart.seed_bundles(self.job_seeds.optimization_seed)
        if bundles != expected_bundles:
            msg = "Multistart seed bundles differ from the exact schedule-derived seed set."
            raise ValueError(msg)
        policies = self.update_policies
        if type(policies) is not tuple or any(not isinstance(policy, ScheduledUpdatePolicy) for policy in policies):
            msg = "update_policies must be a tuple of ScheduledUpdatePolicy records."
            raise TypeError(msg)
        expected_policies = _compile_update_policies(
            schedule=self.schedule,
            controlled_stage_id=self.controlled_stage_id,
            controlled_stage_checksum=self.controlled_stage_checksum,
            job_seeds=self.job_seeds,
            start_seed_bundles=bundles,
            checkpoint_validation_trajectory_count=checkpoint_count,
            checkpoint_updates=self.checkpoint_updates,
        )
        if policies != expected_policies:
            msg = "Scheduled update policies do not equal the deterministic compiled schedule."
            raise ValueError(msg)
        if self.execution_scope == "development_schedule_trace":
            expected_identity = _development_identity(self.schedule)
            if (
                self.executable_binding is not None
                or self.normalized_compute_cap is not None
                or self.binding_checksum != expected_identity[0]
                or self.implementation_checksum != expected_identity[1]
                or self.publication_method_id != "development_schedule_trace"
                or self.controlled_stage_id != "development_schedule_adapter"
                or self.controlled_stage_checksum != expected_identity[2]
            ):
                msg = "Development trace identity differs from the sealed non-evidence adapter."
                raise ValueError(msg)
        else:
            if not isinstance(self.executable_binding, ExecutableScopedBinding):
                msg = "Scientific schedule programs require a complete ExecutableScopedBinding."
                raise TypeError(msg)
            executable = self.executable_binding
            binding = executable.binding
            executable.resolve_callable()
            if (
                self.binding_checksum != binding.content_checksum
                or self.implementation_checksum != binding.implementation_checksum
                or self.publication_method_id != binding.publication_method_id
                or self.controlled_stage_id != binding.controlled_stage.implementation_stage_id
                or self.controlled_stage_checksum != binding.controlled_stage.content_checksum
                or self.schedule != binding.strategy_schedule
                or checkpoint_count != binding.execution_budget.checkpoint_validation_trajectory_count
                or self.normalized_compute_cap != binding.execution_budget.normalized_compute_cap
            ):
                msg = "Scheduled program identity differs from its complete executable binding."
                raise ValueError(msg)
        object.__setattr__(self, "_cached_content_checksum", canonical_checksum(self._payload()))

    @property
    def total_updates_per_start(self) -> int:
        """Number of optimizer updates in each start."""
        return self.schedule.phase_boundary.total_updates

    @property
    def start_count(self) -> int:
        """Number of complete optimizer starts."""
        return len(self.start_seed_bundles)

    @property
    def checkpoint_updates(self) -> tuple[int, ...]:
        """Exact checkpoint coordinates, or the empty tuple when disabled."""
        if not self.checkpoint_validation_trajectory_count:
            return ()
        if (
            self.executable_binding is not None
            and self.executable_binding.binding.implementation_artifact.implementation_kind
            in _OPERATOR_GROWTH_IMPLEMENTATION_KINDS
        ):
            return (99, 199)
        return CHECKPOINT_VALIDATION_UPDATES

    def policy(self, start_index: int, update: int) -> ScheduledUpdatePolicy:
        """Resolve one exact update policy by optimizer coordinates.

        Returns:
            The exact compiled policy.

        Raises:
            ValueError: If either coordinate lies outside the program.
        """
        start = require_int(start_index, "start_index")
        index = require_int(update, "update")
        if start >= self.start_count or index >= self.total_updates_per_start:
            msg = "Optimizer start or update lies outside the scheduled program."
            raise ValueError(msg)
        return self.update_policies[start * self.total_updates_per_start + index]

    @classmethod
    def compile(
        cls,
        executable_binding: ExecutableScopedBinding,
        schedule: TrainingStrategySchedule,
        job_seeds: ScheduledJobSeedSet,
    ) -> ScheduledExecutionProgram:
        """Compile one exact scoped binding without approximating its schedule.

        Returns:
            The complete checksum-sealed execution program.

        Raises:
            TypeError: If an input does not use its strict typed artifact.
            ValueError: If the binding/schedule pair or controlled stage is unsupported.
        """
        if not isinstance(executable_binding, ExecutableScopedBinding):
            msg = "executable_binding must be an ExecutableScopedBinding."
            raise TypeError(msg)
        binding = executable_binding.binding
        executable_binding.resolve_callable()
        if not isinstance(schedule, TrainingStrategySchedule):
            msg = "schedule must be a TrainingStrategySchedule."
            raise TypeError(msg)
        if not isinstance(job_seeds, ScheduledJobSeedSet):
            msg = "job_seeds must be a ScheduledJobSeedSet."
            raise TypeError(msg)
        if (
            schedule != binding.strategy_schedule
            or schedule.content_checksum != binding.controlled_stage.strategy_schedule_checksum
        ):
            msg = "The supplied schedule is not the binding's exact controlled-stage schedule."
            raise ValueError(msg)
        if binding.implementation_artifact.implementation_kind not in _SUPPORTED_IMPLEMENTATION_KINDS:
            msg = "The binding implementation kind has no exact scheduled-update semantics."
            raise ValueError(msg)
        if (
            binding.controlled_stage.start_update != 0
            or binding.controlled_stage.stop_update_exclusive != schedule.phase_boundary.total_updates
            or binding.execution_budget.total_update_count != schedule.phase_boundary.total_updates
            or binding.execution_budget.multistart_count != schedule.multistart.start_count
            or max(step.trajectory_count for step in schedule.trajectory_curriculum.steps)
            > binding.execution_budget.maximum_training_trajectory_count
        ):
            msg = "Controlled-stage, budget, schedule, or multistart boundaries disagree."
            raise ValueError(msg)
        if binding.implementation_artifact.implementation_kind.startswith("operator_growth") and (
            binding.controlled_stage.implementation_stage_id != "operator_growth_reoptimization"
        ):
            msg = "Operator growth may schedule only its sealed reoptimization stage."
            raise ValueError(msg)
        checkpoint_policies = tuple(
            policy for policy in binding.evaluation_policies if policy.purpose == "checkpoint_validation"
        )
        if len(checkpoint_policies) > 1:
            msg = "A binding cannot schedule multiple checkpoint-validation policies."
            raise ValueError(msg)
        checkpoint_count = 0 if not checkpoint_policies else checkpoint_policies[0].trajectory_count
        if checkpoint_count != binding.execution_budget.checkpoint_validation_trajectory_count:
            msg = "Checkpoint-validation policy and execution budget disagree."
            raise ValueError(msg)
        bundles = schedule.multistart.seed_bundles(job_seeds.optimization_seed)
        policies = _compile_update_policies(
            schedule=schedule,
            controlled_stage_id=binding.controlled_stage.implementation_stage_id,
            controlled_stage_checksum=binding.controlled_stage.content_checksum,
            job_seeds=job_seeds,
            start_seed_bundles=bundles,
            checkpoint_validation_trajectory_count=checkpoint_count,
            checkpoint_updates=(99, 199)
            if binding.implementation_artifact.implementation_kind in _OPERATOR_GROWTH_IMPLEMENTATION_KINDS
            else CHECKPOINT_VALIDATION_UPDATES,
        )
        return cls(
            execution_scope="scoped_binding",
            binding_checksum=binding.content_checksum,
            implementation_checksum=binding.implementation_checksum,
            publication_method_id=binding.publication_method_id,
            controlled_stage_id=binding.controlled_stage.implementation_stage_id,
            controlled_stage_checksum=binding.controlled_stage.content_checksum,
            executable_binding=executable_binding,
            normalized_compute_cap=binding.execution_budget.normalized_compute_cap,
            schedule=schedule,
            job_seeds=job_seeds,
            checkpoint_validation_trajectory_count=checkpoint_count,
            start_seed_bundles=bundles,
            update_policies=policies,
        )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered program field."""
        return {
            "schema_version": self.schema_version,
            "execution_scope": self.execution_scope,
            "binding_checksum": self.binding_checksum,
            "implementation_checksum": self.implementation_checksum,
            "publication_method_id": self.publication_method_id,
            "controlled_stage_id": self.controlled_stage_id,
            "controlled_stage_checksum": self.controlled_stage_checksum,
            "executable_binding": None if self.executable_binding is None else self.executable_binding.to_dict(),
            "executable_binding_checksum": (
                None if self.executable_binding is None else self.executable_binding.content_checksum
            ),
            "normalized_compute_cap": self.normalized_compute_cap,
            "normalized_compute_cap_enforcement": "prospective_atomic_before_complete_update",
            "structural_stage_semantics": "outside_engine_independently_sealed",
            "schedule": self.schedule.to_dict(),
            "schedule_checksum": self.schedule.content_checksum,
            "job_seeds": self.job_seeds.to_dict(),
            "job_seed_set_checksum": self.job_seeds.content_checksum,
            "checkpoint_validation_trajectory_count": self.checkpoint_validation_trajectory_count,
            "checkpoint_updates": list(self.checkpoint_updates),
            "update_index_semantics": "zero_based_post_update_state",
            "start_seed_bundles": [bundle.to_dict() for bundle in self.start_seed_bundles],
            "update_policies": [policy.to_dict() for policy in self.update_policies],
            "optimizer_state_rule": "preserve_across_every_schedule_boundary",
            "unsupported_schedule_action": "abort_without_approximation",
            "final_test_access": "forbidden",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete executable schedule program."""
        return self._cached_content_checksum

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return {**self._payload(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ScheduledExecutionProgram:
        """Decode and deterministically recompile one strict program.

        Returns:
            The verified execution program.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "execution_scope",
                "binding_checksum",
                "implementation_checksum",
                "publication_method_id",
                "controlled_stage_id",
                "controlled_stage_checksum",
                "executable_binding",
                "executable_binding_checksum",
                "normalized_compute_cap",
                "normalized_compute_cap_enforcement",
                "structural_stage_semantics",
                "schedule",
                "schedule_checksum",
                "job_seeds",
                "job_seed_set_checksum",
                "checkpoint_validation_trajectory_count",
                "checkpoint_updates",
                "update_index_semantics",
                "start_seed_bundles",
                "update_policies",
                "optimizer_state_rule",
                "unsupported_schedule_action",
                "final_test_access",
                "content_checksum",
            }),
            version=SCHEDULED_EXECUTION_PROGRAM_SCHEMA_VERSION,
            name="scheduled execution program",
        )
        expected_literals = {
            "structural_stage_semantics": "outside_engine_independently_sealed",
            "update_index_semantics": "zero_based_post_update_state",
            "optimizer_state_rule": "preserve_across_every_schedule_boundary",
            "unsupported_schedule_action": "abort_without_approximation",
            "final_test_access": "forbidden",
            "normalized_compute_cap_enforcement": "prospective_atomic_before_complete_update",
        }
        if any(mapping[name] != expected for name, expected in expected_literals.items()):
            msg = "Scheduled execution invariants changed."
            raise ValueError(msg)
        raw_bundles = mapping["start_seed_bundles"]
        raw_policies = mapping["update_policies"]
        if type(raw_bundles) is not tuple or type(raw_policies) is not tuple:
            msg = "start_seed_bundles and update_policies must be JSON arrays."
            raise TypeError(msg)
        program = cls(
            execution_scope=cast("ExecutionScope", mapping["execution_scope"]),
            binding_checksum=cast("str", mapping["binding_checksum"]),
            implementation_checksum=cast("str", mapping["implementation_checksum"]),
            publication_method_id=cast("str", mapping["publication_method_id"]),
            controlled_stage_id=cast("str", mapping["controlled_stage_id"]),
            controlled_stage_checksum=cast("str", mapping["controlled_stage_checksum"]),
            executable_binding=(
                None
                if mapping["executable_binding"] is None
                else ExecutableScopedBinding.from_dict(mapping["executable_binding"])
            ),
            normalized_compute_cap=cast("float | None", mapping["normalized_compute_cap"]),
            schedule=TrainingStrategySchedule.from_dict(mapping["schedule"]),
            job_seeds=ScheduledJobSeedSet.from_dict(mapping["job_seeds"]),
            checkpoint_validation_trajectory_count=cast("int", mapping["checkpoint_validation_trajectory_count"]),
            start_seed_bundles=tuple(MultistartSeedBundle.from_dict(item) for item in raw_bundles),
            update_policies=tuple(ScheduledUpdatePolicy.from_dict(item) for item in raw_policies),
        )
        if (
            mapping["schedule_checksum"] != program.schedule.content_checksum
            or mapping["job_seed_set_checksum"] != program.job_seeds.content_checksum
            or mapping["checkpoint_updates"] != program.checkpoint_updates
            or mapping["executable_binding_checksum"]
            != (None if program.executable_binding is None else program.executable_binding.content_checksum)
        ):
            msg = "Serialized program schedule, seed-set, or checkpoint identity is inconsistent."
            raise ValueError(msg)
        return program

    @classmethod
    def from_json(cls, payload: str) -> ScheduledExecutionProgram:
        """Decode canonical JSON into a deterministically verified program.

        Returns:
            The verified execution program.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _development_identity(schedule: TrainingStrategySchedule) -> tuple[str, str, str]:
    """Return sealed non-evidence identities for one frozen schedule trace."""
    binding = canonical_checksum({
        "scope": "development_schedule_trace",
        "schedule_checksum": schedule.content_checksum,
    })
    implementation = canonical_checksum({
        "adapter": "deterministic_callback",
        "schedule_checksum": schedule.content_checksum,
    })
    controlled = canonical_checksum({
        "controlled_stage_id": "development_schedule_adapter",
        "schedule_checksum": schedule.content_checksum,
    })
    return binding, implementation, controlled


def compile_frozen_schedule_trace(
    schedule: TrainingStrategySchedule,
    job_seeds: ScheduledJobSeedSet,
) -> ScheduledExecutionProgram:
    """Compile one frozen development schedule without claiming binding evidence.

    This entry point exists to exhaustively validate every reviewed schedule
    composition before a publication candidate adopts it.  Scientific runs
    must instead use :meth:`ScheduledExecutionProgram.compile` with a complete
    :class:`~.binding_catalog.ExecutableScopedBinding`.

    Returns:
        A deterministic development-only execution program.
    """
    if not isinstance(schedule, TrainingStrategySchedule):
        msg = "schedule must be a TrainingStrategySchedule."
        raise TypeError(msg)
    if not isinstance(job_seeds, ScheduledJobSeedSet):
        msg = "job_seeds must be a ScheduledJobSeedSet."
        raise TypeError(msg)
    universe = FrozenTrainingPolicyUniverse.frozen()
    if schedule not in universe.schedules:
        msg = "Only exact members of the frozen schedule universe can produce development traces."
        raise ValueError(msg)
    binding_checksum, implementation_checksum, controlled_checksum = _development_identity(schedule)
    bundles = schedule.multistart.seed_bundles(job_seeds.optimization_seed)
    policies = _compile_update_policies(
        schedule=schedule,
        controlled_stage_id="development_schedule_adapter",
        controlled_stage_checksum=controlled_checksum,
        job_seeds=job_seeds,
        start_seed_bundles=bundles,
        checkpoint_validation_trajectory_count=universe.checkpoint_validation_trajectory_count,
    )
    return ScheduledExecutionProgram(
        execution_scope="development_schedule_trace",
        binding_checksum=binding_checksum,
        implementation_checksum=implementation_checksum,
        publication_method_id="development_schedule_trace",
        controlled_stage_id="development_schedule_adapter",
        controlled_stage_checksum=controlled_checksum,
        executable_binding=None,
        normalized_compute_cap=None,
        schedule=schedule,
        job_seeds=job_seeds,
        checkpoint_validation_trajectory_count=universe.checkpoint_validation_trajectory_count,
        start_seed_bundles=bundles,
        update_policies=policies,
    )


def compile_development_schedule(
    schedule: TrainingStrategySchedule,
    job_seeds: ScheduledJobSeedSet,
    *,
    checkpoint_validation_trajectory_count: int = 0,
) -> ScheduledExecutionProgram:
    """Compile a typed non-evidence schedule for boundary integration tests.

    Unlike :func:`compile_frozen_schedule_trace`, this function deliberately
    accepts a nonfrozen schedule.  Its execution scope is permanently marked as
    development-only and it cannot carry an executable binding or compute cap.

    Returns:
        The deterministic development-only program.
    """
    if not isinstance(schedule, TrainingStrategySchedule):
        msg = "schedule must be a TrainingStrategySchedule."
        raise TypeError(msg)
    if not isinstance(job_seeds, ScheduledJobSeedSet):
        msg = "job_seeds must be a ScheduledJobSeedSet."
        raise TypeError(msg)
    checkpoint_count = require_int(
        checkpoint_validation_trajectory_count,
        "checkpoint_validation_trajectory_count",
    )
    binding_checksum, implementation_checksum, controlled_checksum = _development_identity(schedule)
    bundles = schedule.multistart.seed_bundles(job_seeds.optimization_seed)
    policies = _compile_update_policies(
        schedule=schedule,
        controlled_stage_id="development_schedule_adapter",
        controlled_stage_checksum=controlled_checksum,
        job_seeds=job_seeds,
        start_seed_bundles=bundles,
        checkpoint_validation_trajectory_count=checkpoint_count,
    )
    return ScheduledExecutionProgram(
        execution_scope="development_schedule_trace",
        binding_checksum=binding_checksum,
        implementation_checksum=implementation_checksum,
        publication_method_id="development_schedule_trace",
        controlled_stage_id="development_schedule_adapter",
        controlled_stage_checksum=controlled_checksum,
        executable_binding=None,
        normalized_compute_cap=None,
        schedule=schedule,
        job_seeds=job_seeds,
        checkpoint_validation_trajectory_count=checkpoint_count,
        start_seed_bundles=bundles,
        update_policies=policies,
    )


@dataclass(frozen=True, slots=True)
class OptimizerInitialization:
    """Recoverable initializer bound to all isolated multistart seeds."""

    seed_bundle: MultistartSeedBundle
    initializer_kind: InitializerKind
    initial_parameters: tuple[float, ...]
    normal_scale: float | None = None
    warm_start_source_checksum: str | None = None
    schema_version: str = field(default=OPTIMIZER_INITIALIZATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate seed binding and reproduce seeded normal initialization."""
        if not isinstance(self.seed_bundle, MultistartSeedBundle):
            msg = "seed_bundle must be a MultistartSeedBundle."
            raise TypeError(msg)
        if self.initializer_kind not in {"normal_pcg64", "sealed_warm_start"}:
            msg = "initializer_kind is unsupported."
            raise ValueError(msg)
        parameters = _float_tuple(self.initial_parameters, "initial_parameters")
        object.__setattr__(self, "initial_parameters", parameters)
        if self.initializer_kind == "normal_pcg64":
            if self.normal_scale is None or self.warm_start_source_checksum is not None:
                msg = "Seeded normal initialization requires only normal_scale."
                raise ValueError(msg)
            scale = require_float(self.normal_scale, "normal_scale", minimum=0.0)
            if scale <= 0.0:
                msg = "normal_scale must be positive."
                raise ValueError(msg)
            object.__setattr__(self, "normal_scale", scale)
            rng = np.random.Generator(np.random.PCG64(self.seed_bundle.initialization_seed))
            expected = tuple(float(item) for item in rng.normal(0.0, scale, size=len(parameters)))
            if parameters != expected:
                msg = "Initial parameters do not reproduce the bound PCG64 initialization seed."
                raise ValueError(msg)
        else:
            if self.normal_scale is not None or self.warm_start_source_checksum is None:
                msg = "A warm start requires only its sealed structural source checksum."
                raise ValueError(msg)
            object.__setattr__(
                self,
                "warm_start_source_checksum",
                require_checksum(self.warm_start_source_checksum, "warm_start_source_checksum"),
            )

    @classmethod
    def normal(
        cls,
        seed_bundle: MultistartSeedBundle,
        parameter_count: int,
        *,
        scale: float,
    ) -> OptimizerInitialization:
        """Generate a deterministic normal initializer.

        Returns:
            The exact recoverable initialization artifact.
        """
        count = require_int(parameter_count, "parameter_count", minimum=1)
        checked_scale = require_float(scale, "scale", minimum=0.0)
        if checked_scale <= 0.0:
            msg = "scale must be positive."
            raise ValueError(msg)
        rng = np.random.Generator(np.random.PCG64(seed_bundle.initialization_seed))
        parameters = tuple(float(item) for item in rng.normal(0.0, checked_scale, size=count))
        return cls(seed_bundle, "normal_pcg64", parameters, normal_scale=checked_scale)

    @classmethod
    def warm_start(
        cls,
        seed_bundle: MultistartSeedBundle,
        parameters: Sequence[float],
        *,
        source_checksum: str,
    ) -> OptimizerInitialization:
        """Bind a structural-stage warm start to the complete seed bundle.

        Returns:
            The exact recoverable initialization artifact.
        """
        return cls(
            seed_bundle,
            "sealed_warm_start",
            tuple(parameters),
            warm_start_source_checksum=source_checksum,
        )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered initializer field."""
        return {
            "schema_version": self.schema_version,
            "seed_bundle": self.seed_bundle.to_dict(),
            "seed_bundle_checksum": self.seed_bundle.content_checksum,
            "initializer_kind": self.initializer_kind,
            "initial_parameters": list(self.initial_parameters),
            "initial_parameter_checksum": _vector_checksum(self.initial_parameters),
            "normal_scale": self.normal_scale,
            "warm_start_source_checksum": self.warm_start_source_checksum,
            "rng_algorithm": "numpy_pcg64",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum binding parameters, source, and every random domain."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> OptimizerInitialization:
        """Decode one exact initializer.

        Returns:
            The verified initializer.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "seed_bundle",
                "seed_bundle_checksum",
                "initializer_kind",
                "initial_parameters",
                "initial_parameter_checksum",
                "normal_scale",
                "warm_start_source_checksum",
                "rng_algorithm",
                "content_checksum",
            }),
            version=OPTIMIZER_INITIALIZATION_SCHEMA_VERSION,
            name="optimizer initialization",
        )
        if mapping["rng_algorithm"] != "numpy_pcg64":
            msg = "Optimizer initialization RNG algorithm changed."
            raise ValueError(msg)
        result = cls(
            seed_bundle=MultistartSeedBundle.from_dict(mapping["seed_bundle"]),
            initializer_kind=cast("InitializerKind", mapping["initializer_kind"]),
            initial_parameters=cast("tuple[float, ...]", mapping["initial_parameters"]),
            normal_scale=cast("float | None", mapping["normal_scale"]),
            warm_start_source_checksum=cast("str | None", mapping["warm_start_source_checksum"]),
        )
        if mapping["seed_bundle_checksum"] != result.seed_bundle.content_checksum or mapping[
            "initial_parameter_checksum"
        ] != _vector_checksum(result.initial_parameters):
            msg = "Serialized initializer checksum aliases are inconsistent."
            raise ValueError(msg)
        return result


def _validate_common_optimizer_payload(
    initialization: OptimizerInitialization,
    parameters: object,
    completed_updates: object,
) -> tuple[tuple[float, ...], int]:
    """Validate fields shared by all method-specific optimizer payloads.

    Returns:
        The normalized parameters and completed-update counter.
    """
    if not isinstance(initialization, OptimizerInitialization):
        msg = "initialization must be an OptimizerInitialization."
        raise TypeError(msg)
    vector = _float_tuple(parameters, "parameters", length=len(initialization.initial_parameters))
    return vector, require_int(completed_updates, "completed_updates")


@dataclass(frozen=True, slots=True)
class KrotovOptimizerPayload:
    """Typed Krotov parameters and learning-rate restart state."""

    initialization: OptimizerInitialization
    parameters: tuple[float, ...]
    completed_updates: int
    learning_rate: float
    learning_rate_schedule: LearningRateSchedule = "constant"
    decay: float = 0.0
    optimizer_kind: Literal["krotov"] = field(default="krotov", init=False)
    schema_version: str = field(default=KROTOV_OPTIMIZER_PAYLOAD_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact Krotov restart fields."""
        parameters, completed = _validate_common_optimizer_payload(
            self.initialization,
            self.parameters,
            self.completed_updates,
        )
        rate = require_float(self.learning_rate, "learning_rate", minimum=0.0)
        decay = require_float(self.decay, "decay", minimum=0.0)
        if rate <= 0.0 or self.learning_rate_schedule not in {"constant", "inverse", "exp"}:
            msg = "Krotov learning-rate configuration is unsupported."
            raise ValueError(msg)
        if self.learning_rate_schedule == "constant" and not math.isclose(decay, 0.0, rel_tol=0.0, abs_tol=0.0):
            msg = "A constant Krotov schedule requires decay zero."
            raise ValueError(msg)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "completed_updates", completed)
        object.__setattr__(self, "learning_rate", rate)
        object.__setattr__(self, "decay", decay)

    @classmethod
    def initialize(
        cls,
        initialization: OptimizerInitialization,
        *,
        learning_rate: float,
        learning_rate_schedule: LearningRateSchedule = "constant",
        decay: float = 0.0,
    ) -> KrotovOptimizerPayload:
        """Create an update-zero Krotov payload.

        Returns:
            The exact typed payload.
        """
        return cls(
            initialization,
            initialization.initial_parameters,
            0,
            learning_rate,
            learning_rate_schedule,
            decay,
        )

    @property
    def parameter_checksum(self) -> str:
        """Checksum of the recoverable current parameter vector."""
        return _vector_checksum(self.parameters)

    @property
    def step_size(self) -> float:
        """Learning rate at the next zero-based optimizer update."""
        if self.learning_rate_schedule == "constant":
            return self.learning_rate
        if self.learning_rate_schedule == "inverse":
            return self.learning_rate / (1.0 + self.decay * self.completed_updates)
        return self.learning_rate * math.exp(-self.decay * self.completed_updates)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered Krotov field."""
        return {
            "schema_version": self.schema_version,
            "optimizer_kind": self.optimizer_kind,
            "initialization": self.initialization.to_dict(),
            "parameters": list(self.parameters),
            "parameter_checksum": self.parameter_checksum,
            "completed_updates": self.completed_updates,
            "learning_rate": self.learning_rate,
            "learning_rate_schedule": self.learning_rate_schedule,
            "decay": self.decay,
            "optimizer_rng_counter": self.completed_updates,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete Krotov restart payload."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())


@dataclass(frozen=True, slots=True)
class AdamOptimizerPayload:
    """Typed parameter-shift Adam parameters, moments, and seeds."""

    initialization: OptimizerInitialization
    parameters: tuple[float, ...]
    completed_updates: int
    config: ParameterShiftAdamConfig
    first_moment: tuple[float, ...]
    second_moment: tuple[float, ...]
    optimizer_kind: Literal["parameter_shift_adam"] = field(default="parameter_shift_adam", init=False)
    schema_version: str = field(default=ADAM_OPTIMIZER_PAYLOAD_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate Adam state dimensions and configuration."""
        parameters, completed = _validate_common_optimizer_payload(
            self.initialization,
            self.parameters,
            self.completed_updates,
        )
        if not isinstance(self.config, ParameterShiftAdamConfig):
            msg = "config must be a ParameterShiftAdamConfig."
            raise TypeError(msg)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "completed_updates", completed)
        object.__setattr__(
            self, "first_moment", _float_tuple(self.first_moment, "first_moment", length=len(parameters))
        )
        object.__setattr__(
            self,
            "second_moment",
            _float_tuple(self.second_moment, "second_moment", length=len(parameters)),
        )

    @classmethod
    def initialize(
        cls,
        initialization: OptimizerInitialization,
        config: ParameterShiftAdamConfig,
    ) -> AdamOptimizerPayload:
        """Create an update-zero Adam payload.

        Returns:
            The exact typed payload.
        """
        zeros = _zeros(len(initialization.initial_parameters))
        return cls(initialization, initialization.initial_parameters, 0, config, zeros, zeros)

    @property
    def parameter_checksum(self) -> str:
        """Checksum of the recoverable current parameter vector."""
        return _vector_checksum(self.parameters)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered Adam field."""
        return {
            "schema_version": self.schema_version,
            "optimizer_kind": self.optimizer_kind,
            "initialization": self.initialization.to_dict(),
            "parameters": list(self.parameters),
            "parameter_checksum": self.parameter_checksum,
            "completed_updates": self.completed_updates,
            "learning_rate": self.config.learning_rate,
            "beta1": self.config.beta1,
            "beta2": self.config.beta2,
            "epsilon": self.config.epsilon,
            "first_moment": list(self.first_moment),
            "second_moment": list(self.second_moment),
            "optimizer_rng_counter": self.completed_updates * len(self.parameters),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete Adam restart payload."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())


@dataclass(frozen=True, slots=True)
class SPSAOptimizerPayload:
    """Typed SPSA parameters and complete counter-derived RNG state."""

    initialization: OptimizerInitialization
    parameters: tuple[float, ...]
    completed_updates: int
    config: SPSAConfig
    last_perturbation_seed: int | None = None
    optimizer_kind: Literal["spsa"] = field(default="spsa", init=False)
    schema_version: str = field(default=SPSA_OPTIMIZER_PAYLOAD_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate SPSA counter and last perturbation seed."""
        parameters, completed = _validate_common_optimizer_payload(
            self.initialization,
            self.parameters,
            self.completed_updates,
        )
        if not isinstance(self.config, SPSAConfig):
            msg = "config must be an SPSAConfig."
            raise TypeError(msg)
        if (completed == 0) != (self.last_perturbation_seed is None):
            msg = "last_perturbation_seed is present exactly after an SPSA update."
            raise ValueError(msg)
        if self.last_perturbation_seed is not None:
            object.__setattr__(
                self,
                "last_perturbation_seed",
                _uint64(self.last_perturbation_seed, "last_perturbation_seed"),
            )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "completed_updates", completed)

    @classmethod
    def initialize(
        cls,
        initialization: OptimizerInitialization,
        config: SPSAConfig,
    ) -> SPSAOptimizerPayload:
        """Create an update-zero SPSA payload.

        Returns:
            The exact typed payload.
        """
        return cls(initialization, initialization.initial_parameters, 0, config)

    @property
    def parameter_checksum(self) -> str:
        """Checksum of the recoverable current parameter vector."""
        return _vector_checksum(self.parameters)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered SPSA field."""
        return {
            "schema_version": self.schema_version,
            "optimizer_kind": self.optimizer_kind,
            "initialization": self.initialization.to_dict(),
            "parameters": list(self.parameters),
            "parameter_checksum": self.parameter_checksum,
            "completed_updates": self.completed_updates,
            "a": self.config.a,
            "stability_constant": self.config.stability_constant,
            "alpha": self.config.alpha,
            "c": self.config.c,
            "gamma": self.config.gamma,
            "last_perturbation_seed": self.last_perturbation_seed,
            "optimizer_rng_counter": self.completed_updates,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete SPSA restart payload."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())


@dataclass(frozen=True, slots=True)
class OperatorGrowthOptimizerPayload:
    """Typed full-parameter Adam state for a sealed growth reoptimization."""

    initialization: OptimizerInitialization
    parameters: tuple[float, ...]
    completed_updates: int
    spec: OperatorGrowthSpec
    growth_step_index: int
    selected_operator_ids: tuple[str, ...]
    structural_state_checksum: str
    first_moment: tuple[float, ...]
    second_moment: tuple[float, ...]
    best_parameters: tuple[float, ...]
    best_objective: float
    optimizer_kind: Literal["operator_growth_adam"] = field(default="operator_growth_adam", init=False)
    schema_version: str = field(default=OPERATOR_GROWTH_OPTIMIZER_PAYLOAD_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate structural identity, current/best vectors, and moments."""
        parameters, completed = _validate_common_optimizer_payload(
            self.initialization,
            self.parameters,
            self.completed_updates,
        )
        if not isinstance(self.spec, OperatorGrowthSpec):
            msg = "spec must be an OperatorGrowthSpec."
            raise TypeError(msg)
        if self.initialization.initializer_kind != "sealed_warm_start":
            msg = "Operator-growth reoptimization requires a sealed structural warm start."
            raise ValueError(msg)
        growth = require_int(self.growth_step_index, "growth_step_index", minimum=1)
        if type(self.selected_operator_ids) is not tuple or not self.selected_operator_ids:
            msg = "selected_operator_ids must be a nonempty tuple."
            raise TypeError(msg)
        operators = tuple(require_slug(item, "selected_operator_id") for item in self.selected_operator_ids)
        if len(operators) != len(set(operators)):
            msg = "selected_operator_ids must be unique."
            raise ValueError(msg)
        structural = require_checksum(self.structural_state_checksum, "structural_state_checksum")
        if self.initialization.warm_start_source_checksum != structural:
            msg = "Operator-growth initialization and structural state checksums differ."
            raise ValueError(msg)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "completed_updates", completed)
        object.__setattr__(self, "growth_step_index", growth)
        object.__setattr__(self, "selected_operator_ids", operators)
        object.__setattr__(self, "structural_state_checksum", structural)
        object.__setattr__(
            self, "first_moment", _float_tuple(self.first_moment, "first_moment", length=len(parameters))
        )
        object.__setattr__(
            self,
            "second_moment",
            _float_tuple(self.second_moment, "second_moment", length=len(parameters)),
        )
        object.__setattr__(
            self, "best_parameters", _float_tuple(self.best_parameters, "best_parameters", length=len(parameters))
        )
        object.__setattr__(self, "best_objective", require_float(self.best_objective, "best_objective"))

    @classmethod
    def initialize(
        cls,
        initialization: OptimizerInitialization,
        spec: OperatorGrowthSpec,
        *,
        growth_step_index: int,
        selected_operator_ids: tuple[str, ...],
        structural_state_checksum: str,
        initial_objective: float,
    ) -> OperatorGrowthOptimizerPayload:
        """Create an update-zero growth-reoptimization payload.

        Returns:
            The exact typed payload.
        """
        zeros = _zeros(len(initialization.initial_parameters))
        return cls(
            initialization,
            initialization.initial_parameters,
            0,
            spec,
            growth_step_index,
            selected_operator_ids,
            structural_state_checksum,
            zeros,
            zeros,
            initialization.initial_parameters,
            initial_objective,
        )

    @property
    def parameter_checksum(self) -> str:
        """Checksum of the recoverable current parameter vector."""
        return _vector_checksum(self.parameters)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered reoptimization field."""
        return {
            "schema_version": self.schema_version,
            "optimizer_kind": self.optimizer_kind,
            "initialization": self.initialization.to_dict(),
            "parameters": list(self.parameters),
            "parameter_checksum": self.parameter_checksum,
            "completed_updates": self.completed_updates,
            "spec": self.spec.to_dict(),
            "spec_checksum": self.spec.content_checksum,
            "growth_step_index": self.growth_step_index,
            "selected_operator_ids": list(self.selected_operator_ids),
            "structural_state_checksum": self.structural_state_checksum,
            "first_moment": list(self.first_moment),
            "second_moment": list(self.second_moment),
            "best_parameters": list(self.best_parameters),
            "best_parameter_checksum": _vector_checksum(self.best_parameters),
            "best_objective": self.best_objective,
            "optimizer_rng_counter": self.completed_updates * len(self.parameters),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete reoptimization restart payload."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())


MethodOptimizerPayload: TypeAlias = (
    KrotovOptimizerPayload | AdamOptimizerPayload | SPSAOptimizerPayload | OperatorGrowthOptimizerPayload
)


def _method_payload_to_dict(payload: MethodOptimizerPayload) -> dict[str, object]:
    """Serialize a strict method-specific payload.

    Returns:
        Checksum-sealed JSON-native data.
    """
    return payload.to_dict()


def _method_payload_from_dict(value: object) -> MethodOptimizerPayload:
    """Decode a strict method-specific payload.

    Returns:
        The verified discriminated payload.
    """
    if not isinstance(value, Mapping):
        msg = "optimizer payload must be a mapping."
        raise TypeError(msg)
    mapping = cast("Mapping[str, object]", value)
    kind = mapping["optimizer_kind"]
    initialization = OptimizerInitialization.from_dict(mapping["initialization"])
    if kind == "krotov":
        result: MethodOptimizerPayload = KrotovOptimizerPayload(
            initialization=initialization,
            parameters=cast("tuple[float, ...]", mapping["parameters"]),
            completed_updates=cast("int", mapping["completed_updates"]),
            learning_rate=cast("float", mapping["learning_rate"]),
            learning_rate_schedule=cast("LearningRateSchedule", mapping["learning_rate_schedule"]),
            decay=cast("float", mapping["decay"]),
        )
    elif kind == "parameter_shift_adam":
        result = AdamOptimizerPayload(
            initialization=initialization,
            parameters=cast("tuple[float, ...]", mapping["parameters"]),
            completed_updates=cast("int", mapping["completed_updates"]),
            config=ParameterShiftAdamConfig(
                learning_rate=cast("float", mapping["learning_rate"]),
                beta1=cast("float", mapping["beta1"]),
                beta2=cast("float", mapping["beta2"]),
                epsilon=cast("float", mapping["epsilon"]),
            ),
            first_moment=cast("tuple[float, ...]", mapping["first_moment"]),
            second_moment=cast("tuple[float, ...]", mapping["second_moment"]),
        )
    elif kind == "spsa":
        result = SPSAOptimizerPayload(
            initialization=initialization,
            parameters=cast("tuple[float, ...]", mapping["parameters"]),
            completed_updates=cast("int", mapping["completed_updates"]),
            config=SPSAConfig(
                a=cast("float", mapping["a"]),
                stability_constant=cast("float", mapping["stability_constant"]),
                alpha=cast("float", mapping["alpha"]),
                c=cast("float", mapping["c"]),
                gamma=cast("float", mapping["gamma"]),
            ),
            last_perturbation_seed=cast("int | None", mapping["last_perturbation_seed"]),
        )
    elif kind == "operator_growth_adam":
        result = OperatorGrowthOptimizerPayload(
            initialization=initialization,
            parameters=cast("tuple[float, ...]", mapping["parameters"]),
            completed_updates=cast("int", mapping["completed_updates"]),
            spec=OperatorGrowthSpec.from_dict(mapping["spec"]),
            growth_step_index=cast("int", mapping["growth_step_index"]),
            selected_operator_ids=cast("tuple[str, ...]", mapping["selected_operator_ids"]),
            structural_state_checksum=cast("str", mapping["structural_state_checksum"]),
            first_moment=cast("tuple[float, ...]", mapping["first_moment"]),
            second_moment=cast("tuple[float, ...]", mapping["second_moment"]),
            best_parameters=cast("tuple[float, ...]", mapping["best_parameters"]),
            best_objective=cast("float", mapping["best_objective"]),
        )
    else:
        msg = "optimizer payload has an unsupported method discriminator."
        raise ValueError(msg)
    if mapping != freeze_json_mapping(result.to_dict(), "normalized optimizer payload"):
        msg = "Optimizer payload fields or checksum changed during normalization."
        raise ValueError(msg)
    return result


def _optimizer_kind_for_program(program: ScheduledExecutionProgram) -> OptimizerKind | None:
    """Return the repository optimizer required by a scientific program.

    Returns:
        The exact optimizer kind, or ``None`` for a development-only program.
    """
    if program.executable_binding is None:
        return None
    binding = program.executable_binding.binding
    if binding.implementation_artifact.implementation_kind in {
        "operator_growth",
        "operator_growth_smoke",
        "tfim_operator_growth",
    }:
        return "operator_growth_adam"
    stage = _controlled_pipeline_stage(program)
    if stage is None:
        msg = "A pipeline scheduled program must resolve one binding-owned controlled stage."
        raise ValueError(msg)
    optimizer = stage.stage_policy["optimizer_id"]
    kinds: dict[object, OptimizerKind] = {
        "krotov": "krotov",
        "parameter_shift_adam": "parameter_shift_adam",
        "spsa": "spsa",
    }
    if optimizer not in kinds:
        msg = "The controlled implementation stage has no scheduled optimizer adapter."
        raise ValueError(msg)
    return kinds[optimizer]


def _controlled_pipeline_stage(program: ScheduledExecutionProgram) -> TrainingStageTemplate | None:
    """Resolve the exact implementation-owned pipeline stage.

    Returns:
        The controlled production or derived smoke stage, or ``None`` for
        operator-growth and development-only programs.
    """
    if program.executable_binding is None:
        return None
    executable = program.executable_binding
    payload = executable.binding.implementation_artifact.implementation_payload
    if isinstance(payload, TrainingPipelineTemplate):
        candidates = tuple(stage for stage in payload.stages if stage.stage_id == program.controlled_stage_id)
    elif isinstance(payload, PipelineSmokeSpec):
        runtime = executable.smoke_runtime_program()
        if not isinstance(runtime, PipelineSmokeRuntimeProgram):
            msg = "A pipeline smoke binding did not derive its exact runtime pipeline."
            raise TypeError(msg)
        candidates = runtime.runtime_template.stages
    else:
        return None
    if len(candidates) != 1:
        msg = "The executable binding must identify exactly one controlled pipeline stage."
        raise ValueError(msg)
    return candidates[0]


def _operator_growth_spec_for_program(program: ScheduledExecutionProgram) -> OperatorGrowthSpec | None:
    """Resolve the exact binding-owned operator-growth optimizer settings.

    Returns:
        The inner WP20 growth specification, or ``None`` for a pipeline or
        development-only program.
    """
    if program.executable_binding is None:
        return None
    binding = program.executable_binding.binding
    if binding.operator_growth_spec is not None:
        return binding.operator_growth_spec.growth_spec
    payload = binding.implementation_artifact.implementation_payload
    if isinstance(payload, OperatorGrowthSmokeSpec):
        return payload.production_growth_spec
    if isinstance(payload, (OperatorGrowthExecutionSpec, EnergyAdaptSmokeSpec)):
        return payload.growth_spec
    return None


def _operator_growth_pool_ids_for_program(program: ScheduledExecutionProgram) -> frozenset[str] | None:
    """Resolve the binding-owned set of selectable operator identifiers.

    Returns:
        Exact operator identifiers, or ``None`` outside a scientific growth
        binding.
    """
    if program.executable_binding is None:
        return None
    binding = program.executable_binding.binding
    if binding.operator_growth_spec is not None:
        pool = binding.operator_growth_spec.pool
    else:
        payload = binding.implementation_artifact.implementation_payload
        if isinstance(payload, OperatorGrowthSmokeSpec):
            pool = payload.production_pool
        elif isinstance(payload, (OperatorGrowthExecutionSpec, EnergyAdaptSmokeSpec)):
            pool = payload.pool
        else:
            return None
    return frozenset(operator.operator_id for operator in pool.operators)


def _expected_parameter_shift_scales(program: ScheduledExecutionProgram, parameter_count: int) -> tuple[float, ...]:
    """Derive the repository BMPD angle-scale vector from the sealed topology.

    Returns:
        The exact state-preparation-adjoint scale for every terminal BMPD
        parameter.  The initial U3 product layer uses ``+1`` and every
        adjoint BMPD rotation uses ``-1``.
    """
    if program.executable_binding is None:
        msg = "Development programs do not own a repository parameter-shift topology."
        raise ValueError(msg)
    qubits = program.executable_binding.binding.qubit_count
    initial_layer_count = 3 * qubits
    if parameter_count < initial_layer_count:
        msg = "The controlled Adam stage is smaller than its required BMPD initial layer."
        raise ValueError(msg)
    return (1.0,) * initial_layer_count + (-1.0,) * (parameter_count - initial_layer_count)


def _validate_pipeline_payload(
    stage: TrainingStageTemplate,
    payload: MethodOptimizerPayload,
) -> None:
    """Bind vector size, initialization, and optimizer settings to one stage."""
    policy = stage.stage_policy
    parameter_count = cast("int", policy["output_parameter_count"])
    if len(payload.parameters) != parameter_count:
        msg = "Optimizer parameter dimension differs from the controlled implementation stage."
        raise ValueError(msg)
    initialization = payload.initialization
    transfer = policy["parameter_transfer_rule"]
    hyperparameters = cast("Mapping[str, object]", policy["optimizer_hyperparameters"])
    if transfer == "initialize_random_normal":
        if (
            initialization.initializer_kind != "normal_pcg64"
            or hyperparameters.get("initialization_rng") != "numpy_pcg64_standard_normal_v1"
            or initialization.normal_scale != hyperparameters.get("initialization_scale")
        ):
            msg = "Optimizer initialization differs from the stage's sealed PCG64 normal policy."
            raise ValueError(msg)
    elif initialization.initializer_kind != "sealed_warm_start":
        msg = "A controlled continuation or structural stage requires a sealed warm start."
        raise ValueError(msg)

    if isinstance(payload, KrotovOptimizerPayload):
        expected = (
            hyperparameters.get("learning_rate"),
            hyperparameters.get("schedule", "constant"),
            hyperparameters.get("decay", 0.0),
        )
        actual = (payload.learning_rate, payload.learning_rate_schedule, payload.decay)
        if actual != expected:
            msg = "Krotov hyperparameters differ from the controlled implementation stage."
            raise ValueError(msg)
    elif isinstance(payload, AdamOptimizerPayload):
        expected_config = ParameterShiftAdamConfig(
            learning_rate=cast("float", hyperparameters["learning_rate"]),
            beta1=cast("float", hyperparameters["beta1"]),
            beta2=cast("float", hyperparameters["beta2"]),
            epsilon=cast("float", hyperparameters["epsilon"]),
        )
        if payload.config != expected_config:
            msg = "Adam hyperparameters differ from the controlled implementation stage."
            raise ValueError(msg)
    elif isinstance(payload, SPSAOptimizerPayload):
        expected_config = SPSAConfig(
            a=cast("float", hyperparameters["a"]),
            stability_constant=cast("float", hyperparameters["A"]),
            alpha=cast("float", hyperparameters["alpha"]),
            c=cast("float", hyperparameters["c"]),
            gamma=cast("float", hyperparameters["gamma"]),
        )
        if payload.config != expected_config:
            msg = "SPSA hyperparameters differ from the controlled implementation stage."
            raise ValueError(msg)


def _validate_payload_for_program(
    program: ScheduledExecutionProgram,
    payload: MethodOptimizerPayload,
) -> None:
    """Require complete optimizer closure to the executable implementation."""
    expected = _optimizer_kind_for_program(program)
    if expected is not None and payload.optimizer_kind != expected:
        msg = "Optimizer payload kind differs from the executable binding's repository method."
        raise ValueError(msg)
    stage = _controlled_pipeline_stage(program)
    if stage is not None:
        _validate_pipeline_payload(stage, payload)
    if isinstance(payload, OperatorGrowthOptimizerPayload):
        binding_spec = _operator_growth_spec_for_program(program)
        pool_ids = _operator_growth_pool_ids_for_program(program)
        if binding_spec is not None and payload.spec != binding_spec:
            msg = "Operator-growth optimizer state differs from the binding's exact reoptimization specification."
            raise ValueError(msg)
        if (
            len(payload.parameters) != len(payload.selected_operator_ids)
            or payload.growth_step_index != len(payload.selected_operator_ids)
            or len(payload.selected_operator_ids) > payload.spec.max_operators
            or (pool_ids is not None and not set(payload.selected_operator_ids) <= pool_ids)
        ):
            msg = "Operator-growth optimizer dimension differs from its complete selected prefix."
            raise ValueError(msg)


def _validate_optimizer_transition(
    before: MethodOptimizerPayload,
    after: MethodOptimizerPayload,
) -> None:
    """Require one update to preserve every immutable optimizer field."""
    if type(before) is not type(after) or before.initialization != after.initialization:
        msg = "An optimizer update cannot change method type or initialization."
        raise ValueError(msg)
    if isinstance(before, KrotovOptimizerPayload) and isinstance(after, KrotovOptimizerPayload):
        if (
            before.learning_rate,
            before.learning_rate_schedule,
            before.decay,
        ) != (
            after.learning_rate,
            after.learning_rate_schedule,
            after.decay,
        ):
            msg = "A Krotov update cannot reset or change its learning-rate policy."
            raise ValueError(msg)
    elif isinstance(before, AdamOptimizerPayload) and isinstance(after, AdamOptimizerPayload):
        if before.config != after.config:
            msg = "An Adam update cannot reset or change its optimizer configuration."
            raise ValueError(msg)
    elif isinstance(before, SPSAOptimizerPayload) and isinstance(after, SPSAOptimizerPayload):
        if before.config != after.config:
            msg = "An SPSA update cannot reset or change its gain configuration."
            raise ValueError(msg)
    elif (
        isinstance(before, OperatorGrowthOptimizerPayload)
        and isinstance(after, OperatorGrowthOptimizerPayload)
        and (
            before.spec != after.spec
            or before.growth_step_index != after.growth_step_index
            or before.selected_operator_ids != after.selected_operator_ids
            or before.structural_state_checksum != after.structural_state_checksum
            or after.best_objective > before.best_objective
            or (after.best_objective == before.best_objective and after.best_parameters != before.best_parameters)
            or (after.best_objective < before.best_objective and after.best_parameters != after.parameters)
        )
    ):
        msg = "Operator-growth reoptimization changed its sealed prefix or violated strict best-state tracking."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ScheduledUpdateRequest:
    """Validation-blind callback input for one optimizer update."""

    program_checksum: str
    policy: ScheduledTrainingPolicy
    seed_bundle: MultistartSeedBundle
    optimizer_payload: MethodOptimizerPayload
    previous_receipt_checksum: str | None
    schema_version: str = field(default=SCHEDULED_UPDATE_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate coordinates, seeds, and typed pre-update state."""
        object.__setattr__(self, "program_checksum", require_checksum(self.program_checksum, "program_checksum"))
        if not isinstance(self.policy, ScheduledTrainingPolicy):
            msg = "policy must be a ScheduledTrainingPolicy."
            raise TypeError(msg)
        if not isinstance(self.seed_bundle, MultistartSeedBundle):
            msg = "seed_bundle must be a MultistartSeedBundle."
            raise TypeError(msg)
        if self.policy.start_index != self.seed_bundle.start_index:
            msg = "Update policy and multistart seed bundle identify different starts."
            raise ValueError(msg)
        if not isinstance(
            self.optimizer_payload,
            (KrotovOptimizerPayload, AdamOptimizerPayload, SPSAOptimizerPayload, OperatorGrowthOptimizerPayload),
        ):
            msg = "optimizer_payload must be a method-specific optimizer payload."
            raise TypeError(msg)
        if self.optimizer_payload.initialization.seed_bundle != self.seed_bundle:
            msg = "Optimizer initialization is not bound to this start's complete seed bundle."
            raise ValueError(msg)
        if self.optimizer_payload.completed_updates != self.policy.update:
            msg = "Optimizer payload counter differs from its requested update."
            raise ValueError(msg)
        if self.previous_receipt_checksum is not None:
            object.__setattr__(
                self,
                "previous_receipt_checksum",
                require_checksum(self.previous_receipt_checksum, "previous_receipt_checksum"),
            )

    @property
    def optimizer_payload_checksum(self) -> str:
        """Checksum of the exact pre-update optimizer state."""
        return self.optimizer_payload.content_checksum

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered callback input."""
        return {
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "policy": self.policy.to_dict(),
            "seed_bundle": self.seed_bundle.to_dict(),
            "optimizer_payload": _method_payload_to_dict(self.optimizer_payload),
            "optimizer_payload_checksum": self.optimizer_payload_checksum,
            "previous_receipt_checksum": self.previous_receipt_checksum,
            "accessible_data_role": "training_trajectory",
            "validation_membership_access": "forbidden_before_update",
            "evaluation_result_access": "forbidden",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the exact callback request."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ScheduledUpdateRequest:
        """Decode and verify one complete pre-update request.

        Returns:
            The verified validation-blind request.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "program_checksum",
                "policy",
                "seed_bundle",
                "optimizer_payload",
                "optimizer_payload_checksum",
                "previous_receipt_checksum",
                "accessible_data_role",
                "validation_membership_access",
                "evaluation_result_access",
                "content_checksum",
            }),
            version=SCHEDULED_UPDATE_REQUEST_SCHEMA_VERSION,
            name="scheduled update request",
        )
        if (
            mapping["accessible_data_role"] != "training_trajectory"
            or mapping["validation_membership_access"] != "forbidden_before_update"
            or mapping["evaluation_result_access"] != "forbidden"
        ):
            msg = "Scheduled update request data-role isolation changed."
            raise ValueError(msg)
        result = cls(
            program_checksum=cast("str", mapping["program_checksum"]),
            policy=ScheduledTrainingPolicy.from_dict(mapping["policy"]),
            seed_bundle=MultistartSeedBundle.from_dict(mapping["seed_bundle"]),
            optimizer_payload=_method_payload_from_dict(mapping["optimizer_payload"]),
            previous_receipt_checksum=cast("str | None", mapping["previous_receipt_checksum"]),
        )
        if mapping["optimizer_payload_checksum"] != result.optimizer_payload_checksum:
            msg = "Serialized update-request optimizer payload checksum is inconsistent."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class ScheduledUpdateResult:
    """One repository-adapter update with exact typed state and work."""

    optimizer_payload: MethodOptimizerPayload
    adapter_checksum: str
    normalized_work: float
    schema_version: str = field(default=SCHEDULED_UPDATE_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate method-specific state and finite nonnegative work."""
        if not isinstance(
            self.optimizer_payload,
            (KrotovOptimizerPayload, AdamOptimizerPayload, SPSAOptimizerPayload, OperatorGrowthOptimizerPayload),
        ):
            msg = "optimizer_payload must be a method-specific optimizer payload."
            raise TypeError(msg)
        object.__setattr__(self, "adapter_checksum", require_checksum(self.adapter_checksum, "adapter_checksum"))
        object.__setattr__(
            self,
            "normalized_work",
            require_float(self.normalized_work, "normalized_work", minimum=0.0),
        )

    @property
    def optimizer_payload_checksum(self) -> str:
        """Checksum of the exact post-update optimizer state."""
        return self.optimizer_payload.content_checksum

    @property
    def parameter_checksum(self) -> str:
        """Checksum of the recoverable post-update parameter vector."""
        return self.optimizer_payload.parameter_checksum

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered callback result."""
        return {
            "schema_version": self.schema_version,
            "optimizer_payload": _method_payload_to_dict(self.optimizer_payload),
            "optimizer_payload_checksum": self.optimizer_payload_checksum,
            "parameter_checksum": self.parameter_checksum,
            "adapter_checksum": self.adapter_checksum,
            "normalized_work": self.normalized_work,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the exact update result."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ScheduledUpdateResult:
        """Decode and verify one strict update result.

        Returns:
            The verified update result.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "optimizer_payload",
                "optimizer_payload_checksum",
                "parameter_checksum",
                "adapter_checksum",
                "normalized_work",
                "content_checksum",
            }),
            version=SCHEDULED_UPDATE_RESULT_SCHEMA_VERSION,
            name="scheduled update result",
        )
        result = cls(
            optimizer_payload=_method_payload_from_dict(mapping["optimizer_payload"]),
            adapter_checksum=cast("str", mapping["adapter_checksum"]),
            normalized_work=cast("float", mapping["normalized_work"]),
        )
        if (
            mapping["optimizer_payload_checksum"] != result.optimizer_payload_checksum
            or mapping["parameter_checksum"] != result.parameter_checksum
        ):
            msg = "Serialized post-update optimizer state checksum is inconsistent."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class ParameterCheckpointArtifact:
    """Recoverable parameter bytes at one completed update."""

    optimizer_kind: OptimizerKind
    start_index: int
    update: int
    parameters: tuple[float, ...]
    optimizer_payload_checksum: str
    schema_version: str = field(default=PARAMETER_CHECKPOINT_ARTIFACT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact method, coordinates, vector, and optimizer link."""
        if self.optimizer_kind not in _OPTIMIZER_KINDS:
            msg = "optimizer_kind is unsupported."
            raise ValueError(msg)
        object.__setattr__(self, "start_index", require_int(self.start_index, "start_index"))
        object.__setattr__(self, "update", require_int(self.update, "update"))
        object.__setattr__(self, "parameters", _float_tuple(self.parameters, "parameters"))
        object.__setattr__(
            self,
            "optimizer_payload_checksum",
            require_checksum(self.optimizer_payload_checksum, "optimizer_payload_checksum"),
        )

    @classmethod
    def from_payload(
        cls,
        start_index: int,
        update: int,
        payload: MethodOptimizerPayload,
    ) -> ParameterCheckpointArtifact:
        """Capture a complete post-update parameter artifact.

        Returns:
            The recoverable artifact.
        """
        return cls(
            optimizer_kind=payload.optimizer_kind,
            start_index=start_index,
            update=update,
            parameters=payload.parameters,
            optimizer_payload_checksum=payload.content_checksum,
        )

    @property
    def parameter_checksum(self) -> str:
        """Checksum of the recoverable parameter vector."""
        return _vector_checksum(self.parameters)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered artifact field."""
        return {
            "schema_version": self.schema_version,
            "optimizer_kind": self.optimizer_kind,
            "start_index": self.start_index,
            "update": self.update,
            "parameters": list(self.parameters),
            "parameter_checksum": self.parameter_checksum,
            "optimizer_payload_checksum": self.optimizer_payload_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering coordinates, parameters, and optimizer state."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ParameterCheckpointArtifact:
        """Decode one recoverable checkpoint artifact.

        Returns:
            The verified artifact.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "optimizer_kind",
                "start_index",
                "update",
                "parameters",
                "parameter_checksum",
                "optimizer_payload_checksum",
                "content_checksum",
            }),
            version=PARAMETER_CHECKPOINT_ARTIFACT_SCHEMA_VERSION,
            name="parameter checkpoint artifact",
        )
        result = cls(
            optimizer_kind=cast("OptimizerKind", mapping["optimizer_kind"]),
            start_index=cast("int", mapping["start_index"]),
            update=cast("int", mapping["update"]),
            parameters=cast("tuple[float, ...]", mapping["parameters"]),
            optimizer_payload_checksum=cast("str", mapping["optimizer_payload_checksum"]),
        )
        if mapping["parameter_checksum"] != result.parameter_checksum:
            msg = "Serialized checkpoint parameter checksum is inconsistent."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class ScheduledValidationRequest:
    """Validation-only callback input emitted at an exact checkpoint update."""

    program_checksum: str
    start_index: int
    update: int
    parameter_artifact: ParameterCheckpointArtifact
    membership: TrajectoryEnsembleMembership
    schema_version: str = field(default=SCHEDULED_VALIDATION_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require exact checkpoint-validation identity and immutable state."""
        object.__setattr__(self, "program_checksum", require_checksum(self.program_checksum, "program_checksum"))
        object.__setattr__(self, "start_index", require_int(self.start_index, "start_index"))
        object.__setattr__(self, "update", require_int(self.update, "update"))
        if not isinstance(self.parameter_artifact, ParameterCheckpointArtifact):
            msg = "parameter_artifact must be a ParameterCheckpointArtifact."
            raise TypeError(msg)
        if self.parameter_artifact.start_index != self.start_index or self.parameter_artifact.update != self.update:
            msg = "Validation coordinates differ from the post-update parameter artifact."
            raise ValueError(msg)
        if not isinstance(self.membership, TrajectoryEnsembleMembership):
            msg = "membership must be a TrajectoryEnsembleMembership."
            raise TypeError(msg)
        if self.membership.role != "checkpoint_validation":
            msg = "Scheduled validation accepts only checkpoint_validation membership."
            raise ValueError(msg)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered validation input."""
        return {
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "start_index": self.start_index,
            "update": self.update,
            "data_role": "checkpoint_validation",
            "parameter_artifact": self.parameter_artifact.to_dict(),
            "parameter_checksum": self.parameter_artifact.parameter_checksum,
            "post_update_optimizer_payload_checksum": self.parameter_artifact.optimizer_payload_checksum,
            "membership": self.membership.to_dict(),
            "membership_checksum": self.membership.content_checksum,
            "final_test_access": "forbidden",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the validation-only request."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ScheduledValidationRequest:
        """Decode one post-update validation request.

        Returns:
            The verified request.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "program_checksum",
                "start_index",
                "update",
                "data_role",
                "parameter_artifact",
                "parameter_checksum",
                "post_update_optimizer_payload_checksum",
                "membership",
                "membership_checksum",
                "final_test_access",
                "content_checksum",
            }),
            version=SCHEDULED_VALIDATION_REQUEST_SCHEMA_VERSION,
            name="scheduled validation request",
        )
        if mapping["data_role"] != "checkpoint_validation" or mapping["final_test_access"] != "forbidden":
            msg = "Scheduled validation data-role isolation changed."
            raise ValueError(msg)
        result = cls(
            program_checksum=cast("str", mapping["program_checksum"]),
            start_index=cast("int", mapping["start_index"]),
            update=cast("int", mapping["update"]),
            parameter_artifact=ParameterCheckpointArtifact.from_dict(mapping["parameter_artifact"]),
            membership=TrajectoryEnsembleMembership.from_dict(mapping["membership"]),
        )
        if (
            mapping["parameter_checksum"] != result.parameter_artifact.parameter_checksum
            or mapping["post_update_optimizer_payload_checksum"] != result.parameter_artifact.optimizer_payload_checksum
            or mapping["membership_checksum"] != result.membership.content_checksum
        ):
            msg = "Serialized validation request checksum aliases are inconsistent."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class ScheduledValidationResult:
    """Bounded checkpoint-validation fidelity sealed to its exact request."""

    request_checksum: str
    parameter_checksum: str
    membership_checksum: str
    score: float
    schema_version: str = field(default=SCHEDULED_VALIDATION_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require exact request links and a physical fidelity in [0, 1]."""
        for name in ("request_checksum", "parameter_checksum", "membership_checksum"):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(self, "score", require_float(self.score, "score", minimum=0.0, maximum=1.0))

    @classmethod
    def for_request(cls, request: ScheduledValidationRequest, score: float) -> ScheduledValidationResult:
        """Bind one bounded fidelity to its exact request.

        Returns:
            The sealed validation result.
        """
        return cls(
            request_checksum=request.content_checksum,
            parameter_checksum=request.parameter_artifact.parameter_checksum,
            membership_checksum=request.membership.content_checksum,
            score=score,
        )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered result field."""
        return {
            "schema_version": self.schema_version,
            "data_role": "checkpoint_validation",
            "request_checksum": self.request_checksum,
            "parameter_checksum": self.parameter_checksum,
            "membership_checksum": self.membership_checksum,
            "score": self.score,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the exact bounded validation observation."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> ScheduledValidationResult:
        """Decode one validation result.

        Returns:
            The verified result.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "data_role",
                "request_checksum",
                "parameter_checksum",
                "membership_checksum",
                "score",
                "content_checksum",
            }),
            version=SCHEDULED_VALIDATION_RESULT_SCHEMA_VERSION,
            name="scheduled validation result",
        )
        if mapping["data_role"] != "checkpoint_validation":
            msg = "Scheduled validation result must remain validation-only."
            raise ValueError(msg)
        return cls(
            request_checksum=cast("str", mapping["request_checksum"]),
            parameter_checksum=cast("str", mapping["parameter_checksum"]),
            membership_checksum=cast("str", mapping["membership_checksum"]),
            score=cast("float", mapping["score"]),
        )


@dataclass(frozen=True, slots=True)
class ScheduledUpdateReceipt:
    """Immutable evidence for one completed scheduled optimizer update."""

    program_checksum: str
    start_index: int
    update: int
    policy_checksum: str
    request: ScheduledUpdateRequest
    result: ScheduledUpdateResult
    parameter_artifact: ParameterCheckpointArtifact
    normalized_work: float
    validation_request: ScheduledValidationRequest | None
    validation_result: ScheduledValidationResult | None
    previous_receipt_checksum: str | None
    schema_version: str = field(default=SCHEDULED_UPDATE_RECEIPT_SCHEMA_VERSION, init=False)
    _cached_content_checksum: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the exact update chain and optional validation evidence."""
        for name in (
            "program_checksum",
            "policy_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(self, "start_index", require_int(self.start_index, "start_index"))
        object.__setattr__(self, "update", require_int(self.update, "update"))
        if not isinstance(self.request, ScheduledUpdateRequest) or not isinstance(self.result, ScheduledUpdateResult):
            msg = "Receipt request and result must use their complete typed records."
            raise TypeError(msg)
        if (
            self.request.program_checksum != self.program_checksum
            or self.request.policy.start_index != self.start_index
            or self.request.policy.update != self.update
            or self.result.optimizer_payload.completed_updates != self.update + 1
            or self.result.optimizer_payload.initialization != self.request.optimizer_payload.initialization
        ):
            msg = "Receipt request/result coordinates, counter, or initializer differ."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "normalized_work",
            require_float(self.normalized_work, "normalized_work", minimum=0.0),
        )
        if not isinstance(self.parameter_artifact, ParameterCheckpointArtifact):
            msg = "parameter_artifact must be a ParameterCheckpointArtifact."
            raise TypeError(msg)
        if (
            self.parameter_artifact.start_index != self.start_index
            or self.parameter_artifact.update != self.update
            or self.parameter_artifact.optimizer_payload_checksum != self.result.optimizer_payload_checksum
            or self.parameter_artifact.parameters != self.result.optimizer_payload.parameters
        ):
            msg = "Receipt coordinates or post-update state differ from its parameter artifact."
            raise ValueError(msg)
        if (self.validation_request is None) != (self.validation_result is None):
            msg = "Validation request and result must be present together."
            raise ValueError(msg)
        if self.validation_request is not None:
            if not isinstance(self.validation_request, ScheduledValidationRequest) or not isinstance(
                self.validation_result,
                ScheduledValidationResult,
            ):
                msg = "Validation evidence must use typed request and result records."
                raise TypeError(msg)
            result = self.validation_result
            if (
                self.validation_request.start_index != self.start_index
                or self.validation_request.update != self.update
                or self.validation_request.parameter_artifact != self.parameter_artifact
                or result.request_checksum != self.validation_request.content_checksum
                or result.parameter_checksum != self.parameter_artifact.parameter_checksum
                or result.membership_checksum != self.validation_request.membership.content_checksum
            ):
                msg = "Validation request/result links differ from their exact post-update artifact."
                raise ValueError(msg)
        if self.previous_receipt_checksum is not None:
            object.__setattr__(
                self,
                "previous_receipt_checksum",
                require_checksum(self.previous_receipt_checksum, "previous_receipt_checksum"),
            )
        if (self.update == 0) != (self.previous_receipt_checksum is None):
            msg = "Only update zero may begin a receipt chain."
            raise ValueError(msg)
        if self.request.previous_receipt_checksum != self.previous_receipt_checksum:
            msg = "Receipt and typed update request identify different predecessors."
            raise ValueError(msg)
        validation_work = 0 if self.validation_request is None else self.validation_request.membership.trajectory_count
        if not math.isclose(
            self.normalized_work,
            math.fsum((self.result.normalized_work, float(validation_work))),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            msg = "Receipt work does not equal its typed training result plus validation membership."
            raise ValueError(msg)
        object.__setattr__(self, "_cached_content_checksum", canonical_checksum(self._payload()))

    @property
    def request_checksum(self) -> str:
        """Checksum of the complete persisted pre-update request."""
        return self.request.content_checksum

    @property
    def result_checksum(self) -> str:
        """Checksum of the complete persisted post-update result."""
        return self.result.content_checksum

    @property
    def before_optimizer_payload_checksum(self) -> str:
        """Checksum of the recoverable pre-update optimizer payload."""
        return self.request.optimizer_payload_checksum

    @property
    def after_optimizer_payload_checksum(self) -> str:
        """Checksum of the recoverable post-update optimizer payload."""
        return self.result.optimizer_payload_checksum

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered receipt field."""
        return {
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "start_index": self.start_index,
            "update": self.update,
            "policy_checksum": self.policy_checksum,
            "request": self.request.to_dict(),
            "request_checksum": self.request_checksum,
            "result": self.result.to_dict(),
            "result_checksum": self.result_checksum,
            "before_optimizer_payload_checksum": self.before_optimizer_payload_checksum,
            "after_optimizer_payload_checksum": self.after_optimizer_payload_checksum,
            "parameter_artifact": self.parameter_artifact.to_dict(),
            "parameter_checksum": self.parameter_checksum,
            "normalized_work": self.normalized_work,
            "validation_request": None if self.validation_request is None else self.validation_request.to_dict(),
            "validation_result": None if self.validation_result is None else self.validation_result.to_dict(),
            "previous_receipt_checksum": self.previous_receipt_checksum,
        }

    @property
    def parameter_checksum(self) -> str:
        """Checksum of this receipt's recoverable parameter vector."""
        return self.parameter_artifact.parameter_checksum

    @property
    def validation_checkpoint(self) -> ValidationCheckpoint | None:
        """Tracker-compatible bounded checkpoint observation, if due."""
        return (
            None
            if self.validation_result is None
            else ValidationCheckpoint(update=self.update, score=self.validation_result.score)
        )

    @property
    def content_checksum(self) -> str:
        """Checksum covering the exact receipt chain link."""
        return self._cached_content_checksum

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return {**self._payload(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, value: object) -> ScheduledUpdateReceipt:
        """Decode and verify one exact update receipt.

        Returns:
            The verified receipt.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "program_checksum",
                "start_index",
                "update",
                "policy_checksum",
                "request",
                "request_checksum",
                "result",
                "result_checksum",
                "before_optimizer_payload_checksum",
                "after_optimizer_payload_checksum",
                "parameter_artifact",
                "parameter_checksum",
                "normalized_work",
                "validation_request",
                "validation_result",
                "previous_receipt_checksum",
                "content_checksum",
            }),
            version=SCHEDULED_UPDATE_RECEIPT_SCHEMA_VERSION,
            name="scheduled update receipt",
        )
        raw_request = mapping["validation_request"]
        raw_result = mapping["validation_result"]
        result = cls(
            program_checksum=cast("str", mapping["program_checksum"]),
            start_index=cast("int", mapping["start_index"]),
            update=cast("int", mapping["update"]),
            policy_checksum=cast("str", mapping["policy_checksum"]),
            request=ScheduledUpdateRequest.from_dict(mapping["request"]),
            result=ScheduledUpdateResult.from_dict(mapping["result"]),
            parameter_artifact=ParameterCheckpointArtifact.from_dict(mapping["parameter_artifact"]),
            normalized_work=cast("float", mapping["normalized_work"]),
            validation_request=(None if raw_request is None else ScheduledValidationRequest.from_dict(raw_request)),
            validation_result=(None if raw_result is None else ScheduledValidationResult.from_dict(raw_result)),
            previous_receipt_checksum=cast("str | None", mapping["previous_receipt_checksum"]),
        )
        if (
            mapping["request_checksum"] != result.request_checksum
            or mapping["result_checksum"] != result.result_checksum
            or mapping["before_optimizer_payload_checksum"] != result.before_optimizer_payload_checksum
            or mapping["after_optimizer_payload_checksum"] != result.after_optimizer_payload_checksum
            or mapping["parameter_checksum"] != result.parameter_checksum
        ):
            msg = "Serialized receipt request, result, optimizer, or parameter alias is inconsistent."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class ScheduledOptimizerState:
    """Complete immutable optimizer, membership, and validation restart state."""

    program_checksum: str
    start_index: int
    seed_bundle_checksum: str
    initial_optimizer_payload: MethodOptimizerPayload
    next_update: int
    optimizer_payload: MethodOptimizerPayload
    receipts: tuple[ScheduledUpdateReceipt, ...]
    validation_tracker: CheckpointValidationTracker
    last_training_membership: TrajectoryEnsembleMembership | None
    last_component_memberships: tuple[ComponentTrajectoryMembership, ...]
    total_normalized_work: float
    terminal_reason: TerminalReason | None
    schema_version: str = field(default=SCHEDULED_OPTIMIZER_STATE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact receipt, optimizer, membership, and tracker chains."""
        object.__setattr__(self, "program_checksum", require_checksum(self.program_checksum, "program_checksum"))
        object.__setattr__(self, "start_index", require_int(self.start_index, "start_index"))
        object.__setattr__(
            self,
            "seed_bundle_checksum",
            require_checksum(self.seed_bundle_checksum, "seed_bundle_checksum"),
        )
        if not isinstance(
            self.initial_optimizer_payload,
            (KrotovOptimizerPayload, AdamOptimizerPayload, SPSAOptimizerPayload, OperatorGrowthOptimizerPayload),
        ):
            msg = "initial_optimizer_payload must be a complete method-specific payload."
            raise TypeError(msg)
        if self.initial_optimizer_payload.completed_updates != 0:
            msg = "initial_optimizer_payload must precede every optimizer update."
            raise ValueError(msg)
        next_update = require_int(self.next_update, "next_update")
        object.__setattr__(self, "next_update", next_update)
        if not isinstance(
            self.optimizer_payload,
            (KrotovOptimizerPayload, AdamOptimizerPayload, SPSAOptimizerPayload, OperatorGrowthOptimizerPayload),
        ):
            msg = "optimizer_payload must be a method-specific optimizer payload."
            raise TypeError(msg)
        if self.optimizer_payload.completed_updates != next_update:
            msg = "Typed optimizer counter must equal next_update."
            raise ValueError(msg)
        receipts = self.receipts
        if type(receipts) is not tuple or any(not isinstance(item, ScheduledUpdateReceipt) for item in receipts):
            msg = "receipts must be a tuple of ScheduledUpdateReceipt records."
            raise TypeError(msg)
        if next_update != len(receipts) or tuple(item.update for item in receipts) != tuple(range(next_update)):
            msg = "Receipt updates must be complete and consecutive through next_update."
            raise ValueError(msg)
        for index, receipt in enumerate(receipts):
            expected_predecessor = None if index == 0 else receipts[index - 1].content_checksum
            predecessor_payload = (
                self.initial_optimizer_payload if index == 0 else receipts[index - 1].result.optimizer_payload
            )
            _validate_optimizer_transition(predecessor_payload, receipt.result.optimizer_payload)
            if (
                receipt.program_checksum != self.program_checksum
                or receipt.start_index != self.start_index
                or receipt.previous_receipt_checksum != expected_predecessor
                or receipt.request.optimizer_payload != predecessor_payload
            ):
                msg = "Receipt chain program, start, predecessor, or state continuity differs."
                raise ValueError(msg)
        if receipts and receipts[-1].result.optimizer_payload != self.optimizer_payload:
            msg = "Persisted optimizer payload differs from the last completed update."
            raise ValueError(msg)
        if not receipts and self.optimizer_payload != self.initial_optimizer_payload:
            msg = "An update-zero state must retain its exact initial optimizer payload."
            raise ValueError(msg)
        if not isinstance(self.validation_tracker, CheckpointValidationTracker):
            msg = "validation_tracker must be a CheckpointValidationTracker."
            raise TypeError(msg)
        observed = tuple(
            receipt.validation_checkpoint for receipt in receipts if receipt.validation_checkpoint is not None
        )
        if self.validation_tracker.checkpoints != observed:
            msg = "Validation tracker does not contain exactly the receipt observations."
            raise ValueError(msg)
        if self.last_training_membership is not None and not isinstance(
            self.last_training_membership, TrajectoryEnsembleMembership
        ):
            msg = "last_training_membership must be exact membership or None."
            raise TypeError(msg)
        if type(self.last_component_memberships) is not tuple or any(
            not isinstance(item, ComponentTrajectoryMembership) for item in self.last_component_memberships
        ):
            msg = "last_component_memberships must be a tuple of exact component records."
            raise TypeError(msg)
        total = require_float(self.total_normalized_work, "total_normalized_work", minimum=0.0)
        if not math.isclose(total, math.fsum(item.normalized_work for item in receipts), rel_tol=0.0, abs_tol=0.0):
            msg = "total_normalized_work does not equal the complete receipt work."
            raise ValueError(msg)
        object.__setattr__(self, "total_normalized_work", total)
        if self.terminal_reason is not None and self.terminal_reason not in _TERMINAL_REASONS:
            msg = "terminal_reason is unsupported."
            raise ValueError(msg)
        if (self.terminal_reason == "validation_early_stop") != self.validation_tracker.should_stop:
            msg = "validation_early_stop must equal the validation tracker decision."
            raise ValueError(msg)

    @classmethod
    def initialize(
        cls,
        program: ScheduledExecutionProgram,
        start_index: int,
        optimizer_payload: MethodOptimizerPayload,
    ) -> ScheduledOptimizerState:
        """Create one exact pre-update state for a program start.

        Returns:
            The initialized immutable restart state.
        """
        if not isinstance(program, ScheduledExecutionProgram):
            msg = "program must be a ScheduledExecutionProgram."
            raise TypeError(msg)
        start = require_int(start_index, "start_index")
        if start >= program.start_count:
            msg = "start_index lies outside the scheduled program."
            raise ValueError(msg)
        if not isinstance(
            optimizer_payload,
            (KrotovOptimizerPayload, AdamOptimizerPayload, SPSAOptimizerPayload, OperatorGrowthOptimizerPayload),
        ):
            msg = "optimizer_payload must be a method-specific optimizer payload."
            raise TypeError(msg)
        if (
            optimizer_payload.completed_updates != 0
            or optimizer_payload.initialization.seed_bundle != program.start_seed_bundles[start]
        ):
            msg = "Initial optimizer payload must be update zero and bound to the exact start seeds."
            raise ValueError(msg)
        _validate_payload_for_program(program, optimizer_payload)
        return cls(
            program_checksum=program.content_checksum,
            start_index=start,
            seed_bundle_checksum=program.start_seed_bundles[start].content_checksum,
            initial_optimizer_payload=optimizer_payload,
            next_update=0,
            optimizer_payload=optimizer_payload,
            receipts=(),
            validation_tracker=CheckpointValidationTracker(program.schedule.checkpoint_validation),
            last_training_membership=None,
            last_component_memberships=(),
            total_normalized_work=0.0,
            terminal_reason=None,
        )

    @property
    def optimizer_payload_checksum(self) -> str:
        """Checksum of the exact current optimizer payload."""
        return self.optimizer_payload.content_checksum

    @property
    def initial_optimizer_payload_checksum(self) -> str:
        """Checksum of the recoverable update-zero optimizer payload."""
        return self.initial_optimizer_payload.content_checksum

    @property
    def last_parameter_checksum(self) -> str | None:
        """Most recently completed parameter checksum, if any."""
        return None if not self.receipts else self.receipts[-1].parameter_checksum

    @property
    def is_terminal(self) -> bool:
        """Whether this start has a persisted terminal decision."""
        return self.terminal_reason is not None

    def validate_against_program(self, program: ScheduledExecutionProgram) -> None:
        """Verify every persisted link against the exact program.

        Raises:
            ValueError: If the state cannot resume this exact program.
        """
        if not isinstance(program, ScheduledExecutionProgram):
            msg = "program must be a ScheduledExecutionProgram."
            raise TypeError(msg)
        if (
            self.program_checksum != program.content_checksum
            or self.start_index >= program.start_count
            or self.seed_bundle_checksum != program.start_seed_bundles[self.start_index].content_checksum
            or self.initial_optimizer_payload.initialization.seed_bundle != program.start_seed_bundles[self.start_index]
            or self.optimizer_payload.initialization.seed_bundle != program.start_seed_bundles[self.start_index]
            or self.next_update > program.total_updates_per_start
            or self.validation_tracker.policy != program.schedule.checkpoint_validation
        ):
            msg = "Optimizer state does not belong to this exact execution program."
            raise ValueError(msg)
        _validate_payload_for_program(program, self.initial_optimizer_payload)
        _validate_payload_for_program(program, self.optimizer_payload)
        for receipt in self.receipts:
            policy = program.policy(self.start_index, receipt.update)
            if (
                receipt.policy_checksum != policy.content_checksum
                or (receipt.validation_checkpoint is not None) != policy.checkpoint_due
                or receipt.parameter_artifact.optimizer_kind != self.optimizer_payload.optimizer_kind
            ):
                msg = "Receipt schedule or validation boundary differs from the program."
                raise ValueError(msg)
        last_policy = None if not self.receipts else program.policy(self.start_index, self.next_update - 1)
        expected_training = None if last_policy is None else last_policy.training_membership
        expected_components = () if last_policy is None else last_policy.component_memberships
        if self.last_training_membership != expected_training or self.last_component_memberships != expected_components:
            msg = "Persisted trajectory or component membership differs from the last program update."
            raise ValueError(msg)
        expected_terminal = (
            "validation_early_stop"
            if self.validation_tracker.should_stop
            else "budget_complete"
            if self.next_update == program.total_updates_per_start
            else None
        )
        if self.terminal_reason != expected_terminal:
            msg = "Persisted terminal reason differs from program budget or validation state."
            raise ValueError(msg)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered restart field."""
        return {
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "start_index": self.start_index,
            "seed_bundle_checksum": self.seed_bundle_checksum,
            "initial_optimizer_payload": _method_payload_to_dict(self.initial_optimizer_payload),
            "initial_optimizer_payload_checksum": self.initial_optimizer_payload_checksum,
            "next_update": self.next_update,
            "optimizer_payload": _method_payload_to_dict(self.optimizer_payload),
            "optimizer_payload_checksum": self.optimizer_payload_checksum,
            "receipts": [receipt.to_dict() for receipt in self.receipts],
            "validation_tracker": self.validation_tracker.to_dict(),
            "last_training_membership": (
                None if self.last_training_membership is None else self.last_training_membership.to_dict()
            ),
            "last_component_memberships": [item.to_dict() for item in self.last_component_memberships],
            "total_normalized_work": self.total_normalized_work,
            "terminal_reason": self.terminal_reason,
            "resume_rule": "exact_json_state_only",
            "optimizer_reset_at_schedule_boundary": False,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering complete restart state."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed restart JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ScheduledOptimizerState:
        """Decode and verify complete intrinsic restart state.

        Returns:
            The verified optimizer state.  Call ``validate_against_program``
            before resuming execution.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "program_checksum",
                "start_index",
                "seed_bundle_checksum",
                "initial_optimizer_payload",
                "initial_optimizer_payload_checksum",
                "next_update",
                "optimizer_payload",
                "optimizer_payload_checksum",
                "receipts",
                "validation_tracker",
                "last_training_membership",
                "last_component_memberships",
                "total_normalized_work",
                "terminal_reason",
                "resume_rule",
                "optimizer_reset_at_schedule_boundary",
                "content_checksum",
            }),
            version=SCHEDULED_OPTIMIZER_STATE_SCHEMA_VERSION,
            name="scheduled optimizer state",
        )
        if (
            mapping["resume_rule"] != "exact_json_state_only"
            or mapping["optimizer_reset_at_schedule_boundary"] is not False
        ):
            msg = "Scheduled optimizer resume or boundary-state rule changed."
            raise ValueError(msg)
        raw_receipts = mapping["receipts"]
        raw_components = mapping["last_component_memberships"]
        if type(raw_receipts) is not tuple or type(raw_components) is not tuple:
            msg = "receipts and last_component_memberships must be JSON arrays."
            raise TypeError(msg)
        raw_training = mapping["last_training_membership"]
        state = cls(
            program_checksum=cast("str", mapping["program_checksum"]),
            start_index=cast("int", mapping["start_index"]),
            seed_bundle_checksum=cast("str", mapping["seed_bundle_checksum"]),
            initial_optimizer_payload=_method_payload_from_dict(mapping["initial_optimizer_payload"]),
            next_update=cast("int", mapping["next_update"]),
            optimizer_payload=_method_payload_from_dict(mapping["optimizer_payload"]),
            receipts=tuple(ScheduledUpdateReceipt.from_dict(item) for item in raw_receipts),
            validation_tracker=CheckpointValidationTracker.from_dict(mapping["validation_tracker"]),
            last_training_membership=(
                None if raw_training is None else TrajectoryEnsembleMembership.from_dict(raw_training)
            ),
            last_component_memberships=tuple(ComponentTrajectoryMembership.from_dict(item) for item in raw_components),
            total_normalized_work=cast("float", mapping["total_normalized_work"]),
            terminal_reason=cast("TerminalReason | None", mapping["terminal_reason"]),
        )
        if (
            mapping["initial_optimizer_payload_checksum"] != state.initial_optimizer_payload_checksum
            or mapping["optimizer_payload_checksum"] != state.optimizer_payload_checksum
        ):
            msg = "Serialized initial or current optimizer payload checksum is inconsistent."
            raise ValueError(msg)
        return state

    @classmethod
    def from_json(cls, payload: str) -> ScheduledOptimizerState:
        """Decode canonical JSON into verified intrinsic restart state.

        Returns:
            The verified optimizer state.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class MultistartStartEvidence:
    """Terminal work and validation selection evidence for one optimizer start."""

    program_checksum: str
    start_index: int
    state_checksum: str
    terminal_reason: TerminalReason
    completed_update_count: int
    total_normalized_work: float
    selected_update: int
    selected_parameter_artifact: ParameterCheckpointArtifact
    selected_parameter_checksum: str
    selected_validation_score: float | None
    schema_version: str = field(default=MULTISTART_START_EVIDENCE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate one terminal start selection and complete work scalar."""
        object.__setattr__(self, "program_checksum", require_checksum(self.program_checksum, "program_checksum"))
        object.__setattr__(self, "start_index", require_int(self.start_index, "start_index"))
        object.__setattr__(self, "state_checksum", require_checksum(self.state_checksum, "state_checksum"))
        if self.terminal_reason not in _TERMINAL_REASONS:
            msg = "terminal_reason is unsupported."
            raise ValueError(msg)
        updates = require_int(self.completed_update_count, "completed_update_count", minimum=1)
        object.__setattr__(self, "completed_update_count", updates)
        work = require_float(self.total_normalized_work, "total_normalized_work", minimum=0.0)
        object.__setattr__(self, "total_normalized_work", work)
        selected = require_int(self.selected_update, "selected_update")
        if selected >= updates:
            msg = "selected_update must identify a completed optimizer update."
            raise ValueError(msg)
        object.__setattr__(self, "selected_update", selected)
        if not isinstance(self.selected_parameter_artifact, ParameterCheckpointArtifact):
            msg = "selected_parameter_artifact must be a ParameterCheckpointArtifact."
            raise TypeError(msg)
        object.__setattr__(
            self,
            "selected_parameter_checksum",
            require_checksum(self.selected_parameter_checksum, "selected_parameter_checksum"),
        )
        if (
            self.selected_parameter_artifact.start_index != self.start_index
            or self.selected_parameter_artifact.update != selected
            or self.selected_parameter_artifact.parameter_checksum != self.selected_parameter_checksum
        ):
            msg = "Selected coordinates or checksum differ from the recoverable parameter artifact."
            raise ValueError(msg)
        if self.selected_validation_score is not None:
            object.__setattr__(
                self,
                "selected_validation_score",
                require_float(self.selected_validation_score, "selected_validation_score"),
            )

    @classmethod
    def from_state(
        cls,
        program: ScheduledExecutionProgram,
        state: ScheduledOptimizerState,
    ) -> MultistartStartEvidence:
        """Derive terminal start evidence from complete persisted state.

        Returns:
            The exact start-level evidence.
        """
        state.validate_against_program(program)
        if not state.is_terminal or not state.receipts:
            msg = "Multistart evidence requires a terminal start with completed work."
            raise ValueError(msg)
        selection = state.validation_tracker.selection
        if selection is None:
            selected_update = state.receipts[-1].update
            selected_score = None
        else:
            selected_update = selection.best_update
            selected_score = selection.best_score
        parameter_artifact = state.receipts[selected_update].parameter_artifact
        assert state.terminal_reason is not None
        return cls(
            program_checksum=program.content_checksum,
            start_index=state.start_index,
            state_checksum=state.content_checksum,
            terminal_reason=state.terminal_reason,
            completed_update_count=state.next_update,
            total_normalized_work=state.total_normalized_work,
            selected_update=selected_update,
            selected_parameter_artifact=parameter_artifact,
            selected_parameter_checksum=parameter_artifact.parameter_checksum,
            selected_validation_score=selected_score,
        )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered start-evidence field."""
        return {
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "start_index": self.start_index,
            "state_checksum": self.state_checksum,
            "terminal_reason": self.terminal_reason,
            "completed_update_count": self.completed_update_count,
            "total_normalized_work": self.total_normalized_work,
            "selected_update": self.selected_update,
            "selected_parameter_artifact": self.selected_parameter_artifact.to_dict(),
            "selected_parameter_checksum": self.selected_parameter_checksum,
            "selected_validation_score": self.selected_validation_score,
            "selection_data_role": "checkpoint_validation",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering one start's complete work and selection."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> MultistartStartEvidence:
        """Decode and verify one start-level evidence record.

        Returns:
            The verified start evidence.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "program_checksum",
                "start_index",
                "state_checksum",
                "terminal_reason",
                "completed_update_count",
                "total_normalized_work",
                "selected_update",
                "selected_parameter_artifact",
                "selected_parameter_checksum",
                "selected_validation_score",
                "selection_data_role",
                "content_checksum",
            }),
            version=MULTISTART_START_EVIDENCE_SCHEMA_VERSION,
            name="multistart start evidence",
        )
        if mapping["selection_data_role"] != "checkpoint_validation":
            msg = "Multistart start selection must remain validation-only."
            raise ValueError(msg)
        return cls(
            program_checksum=cast("str", mapping["program_checksum"]),
            start_index=cast("int", mapping["start_index"]),
            state_checksum=cast("str", mapping["state_checksum"]),
            terminal_reason=cast("TerminalReason", mapping["terminal_reason"]),
            completed_update_count=cast("int", mapping["completed_update_count"]),
            total_normalized_work=cast("float", mapping["total_normalized_work"]),
            selected_update=cast("int", mapping["selected_update"]),
            selected_parameter_artifact=ParameterCheckpointArtifact.from_dict(mapping["selected_parameter_artifact"]),
            selected_parameter_checksum=cast("str", mapping["selected_parameter_checksum"]),
            selected_validation_score=cast("float | None", mapping["selected_validation_score"]),
        )


@dataclass(frozen=True, slots=True)
class MultistartWorkEvidence:
    """Complete all-start work accounting and validation-only selection."""

    program_checksum: str
    expected_start_count: int
    starts: tuple[MultistartStartEvidence, ...]
    selected_start_index: int
    selected_update: int
    selected_parameter_artifact: ParameterCheckpointArtifact
    selected_parameter_checksum: str
    selected_validation_score: float | None
    total_normalized_work: float
    schema_version: str = field(default=MULTISTART_WORK_EVIDENCE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require all starts and reproduce the frozen tie rules."""
        object.__setattr__(self, "program_checksum", require_checksum(self.program_checksum, "program_checksum"))
        expected_count = require_int(self.expected_start_count, "expected_start_count", minimum=1)
        object.__setattr__(self, "expected_start_count", expected_count)
        if (
            type(self.starts) is not tuple
            or not self.starts
            or any(not isinstance(item, MultistartStartEvidence) for item in self.starts)
        ):
            msg = "starts must be a nonempty tuple of MultistartStartEvidence records."
            raise TypeError(msg)
        if len(self.starts) != expected_count or tuple(item.start_index for item in self.starts) != tuple(
            range(expected_count)
        ):
            msg = "Multistart evidence must contain every start in index order."
            raise ValueError(msg)
        if any(item.program_checksum != self.program_checksum for item in self.starts):
            msg = "Every start evidence record must belong to this program."
            raise ValueError(msg)
        selected_index = require_int(self.selected_start_index, "selected_start_index")
        if selected_index >= len(self.starts):
            msg = "selected_start_index lies outside complete start evidence."
            raise ValueError(msg)
        scores = tuple(item.selected_validation_score for item in self.starts)
        if len(self.starts) > 1 and any(score is None for score in scores):
            msg = "Multiple optimizer starts require validation scores for selection."
            raise ValueError(msg)
        expected = (
            0
            if len(self.starts) == 1 and scores[0] is None
            else min(
                range(len(self.starts)),
                key=lambda index: (
                    -cast("float", self.starts[index].selected_validation_score),
                    self.starts[index].selected_update,
                    self.starts[index].start_index,
                ),
            )
        )
        selected = self.starts[expected]
        if (
            selected_index != expected
            or self.selected_update != selected.selected_update
            or self.selected_parameter_artifact != selected.selected_parameter_artifact
            or self.selected_parameter_checksum != selected.selected_parameter_checksum
            or self.selected_validation_score != selected.selected_validation_score
        ):
            msg = "Multistart selection differs from validation score, earliest checkpoint, or lowest-start ties."
            raise ValueError(msg)
        object.__setattr__(self, "selected_update", require_int(self.selected_update, "selected_update"))
        if not isinstance(self.selected_parameter_artifact, ParameterCheckpointArtifact):
            msg = "selected_parameter_artifact must be a ParameterCheckpointArtifact."
            raise TypeError(msg)
        object.__setattr__(
            self,
            "selected_parameter_checksum",
            require_checksum(self.selected_parameter_checksum, "selected_parameter_checksum"),
        )
        if self.selected_validation_score is not None:
            object.__setattr__(
                self,
                "selected_validation_score",
                require_float(self.selected_validation_score, "selected_validation_score"),
            )
        total = require_float(self.total_normalized_work, "total_normalized_work", minimum=0.0)
        if not math.isclose(
            total, math.fsum(item.total_normalized_work for item in self.starts), rel_tol=0.0, abs_tol=0.0
        ):
            msg = "Multistart work must account for every optimizer start."
            raise ValueError(msg)
        object.__setattr__(self, "total_normalized_work", total)

    @classmethod
    def from_states(
        cls,
        program: ScheduledExecutionProgram,
        states: tuple[ScheduledOptimizerState, ...],
    ) -> MultistartWorkEvidence:
        """Build complete evidence after every optimizer start terminates.

        Returns:
            Complete validation-only multistart evidence.
        """
        if len(states) != program.start_count or tuple(state.start_index for state in states) != tuple(
            range(program.start_count)
        ):
            msg = "Complete multistart evidence requires the program's exact ordered start universe."
            raise ValueError(msg)
        starts = tuple(MultistartStartEvidence.from_state(program, state) for state in states)
        scores = tuple(item.selected_validation_score for item in starts)
        selected_index = (
            0
            if len(starts) == 1 and scores[0] is None
            else min(
                range(len(starts)),
                key=lambda index: (
                    -cast("float", starts[index].selected_validation_score),
                    starts[index].selected_update,
                    starts[index].start_index,
                ),
            )
        )
        selected = starts[selected_index]
        return cls(
            program_checksum=program.content_checksum,
            expected_start_count=program.start_count,
            starts=starts,
            selected_start_index=selected_index,
            selected_update=selected.selected_update,
            selected_parameter_artifact=selected.selected_parameter_artifact,
            selected_parameter_checksum=selected.selected_parameter_checksum,
            selected_validation_score=selected.selected_validation_score,
            total_normalized_work=math.fsum(item.total_normalized_work for item in starts),
        )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered all-start evidence field."""
        return {
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "expected_start_count": self.expected_start_count,
            "starts": [item.to_dict() for item in self.starts],
            "selected_start_index": self.selected_start_index,
            "selected_update": self.selected_update,
            "selected_parameter_artifact": self.selected_parameter_artifact.to_dict(),
            "selected_parameter_checksum": self.selected_parameter_checksum,
            "selected_validation_score": self.selected_validation_score,
            "total_normalized_work": self.total_normalized_work,
            "work_accounting": "all_starts",
            "selection_data_role": "checkpoint_validation",
            "tie_rules": ["earliest_checkpoint", "lowest_start_index"],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering complete multistart work and selection."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> MultistartWorkEvidence:
        """Decode and verify complete multistart evidence.

        Returns:
            The verified all-start evidence.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "program_checksum",
                "expected_start_count",
                "starts",
                "selected_start_index",
                "selected_update",
                "selected_parameter_artifact",
                "selected_parameter_checksum",
                "selected_validation_score",
                "total_normalized_work",
                "work_accounting",
                "selection_data_role",
                "tie_rules",
                "content_checksum",
            }),
            version=MULTISTART_WORK_EVIDENCE_SCHEMA_VERSION,
            name="multistart work evidence",
        )
        if (
            mapping["work_accounting"] != "all_starts"
            or mapping["selection_data_role"] != "checkpoint_validation"
            or mapping["tie_rules"] != ("earliest_checkpoint", "lowest_start_index")
        ):
            msg = "Multistart work, data-role, or tie semantics changed."
            raise ValueError(msg)
        raw_starts = mapping["starts"]
        if type(raw_starts) is not tuple:
            msg = "starts must be a JSON array."
            raise TypeError(msg)
        return cls(
            program_checksum=cast("str", mapping["program_checksum"]),
            expected_start_count=cast("int", mapping["expected_start_count"]),
            starts=tuple(MultistartStartEvidence.from_dict(item) for item in raw_starts),
            selected_start_index=cast("int", mapping["selected_start_index"]),
            selected_update=cast("int", mapping["selected_update"]),
            selected_parameter_artifact=ParameterCheckpointArtifact.from_dict(mapping["selected_parameter_artifact"]),
            selected_parameter_checksum=cast("str", mapping["selected_parameter_checksum"]),
            selected_validation_score=cast("float | None", mapping["selected_validation_score"]),
            total_normalized_work=cast("float", mapping["total_normalized_work"]),
        )


@dataclass(frozen=True, slots=True)
class ScheduledExecutionSnapshot:
    """Persistable interrupted or complete state for all optimizer starts."""

    program_checksum: str
    states: tuple[ScheduledOptimizerState, ...]
    multistart_evidence: MultistartWorkEvidence | None
    schema_version: str = field(default=SCHEDULED_EXECUTION_SNAPSHOT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require ordered unique starts and evidence only after completion."""
        object.__setattr__(self, "program_checksum", require_checksum(self.program_checksum, "program_checksum"))
        if (
            type(self.states) is not tuple
            or not self.states
            or any(not isinstance(state, ScheduledOptimizerState) for state in self.states)
        ):
            msg = "states must be a nonempty tuple of ScheduledOptimizerState records."
            raise TypeError(msg)
        if tuple(state.start_index for state in self.states) != tuple(range(len(self.states))):
            msg = "Snapshot states must contain every optimizer start in index order."
            raise ValueError(msg)
        if any(state.program_checksum != self.program_checksum for state in self.states):
            msg = "Every optimizer state must belong to the snapshot program."
            raise ValueError(msg)
        complete = all(state.is_terminal for state in self.states)
        if complete != (self.multistart_evidence is not None):
            msg = "Complete snapshots require multistart evidence; interrupted snapshots forbid it."
            raise ValueError(msg)
        if self.multistart_evidence is not None:
            if not isinstance(self.multistart_evidence, MultistartWorkEvidence):
                msg = "multistart_evidence must be MultistartWorkEvidence or None."
                raise TypeError(msg)
            if self.multistart_evidence.program_checksum != self.program_checksum or tuple(
                item.state_checksum for item in self.multistart_evidence.starts
            ) != tuple(state.content_checksum for state in self.states):
                msg = "Multistart evidence does not bind every exact terminal state."
                raise ValueError(msg)

    @property
    def complete(self) -> bool:
        """Whether every optimizer start is terminal."""
        return self.multistart_evidence is not None

    def validate_against_program(self, program: ScheduledExecutionProgram) -> None:
        """Verify every state and complete evidence against one exact program."""
        if self.program_checksum != program.content_checksum or len(self.states) != program.start_count:
            msg = "Snapshot does not contain this program's complete optimizer-start universe."
            raise ValueError(msg)
        unfinished_start_seen = False
        for state in self.states:
            if unfinished_start_seen and state.next_update != 0:
                msg = "Snapshot optimizer starts differ from the engine's exact sequential execution order."
                raise ValueError(msg)
            state.validate_against_program(program)
            unfinished_start_seen = unfinished_start_seen or not state.is_terminal
        expected = None if not self.complete else MultistartWorkEvidence.from_states(program, self.states)
        if self.multistart_evidence != expected:
            msg = "Snapshot multistart evidence differs from its exact program states."
            raise ValueError(msg)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered snapshot field."""
        return {
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "states": [state.to_dict() for state in self.states],
            "complete": self.complete,
            "multistart_evidence": (None if self.multistart_evidence is None else self.multistart_evidence.to_dict()),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering all exact restart states and optional evidence."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed snapshot JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ScheduledExecutionSnapshot:
        """Decode and verify intrinsic interrupted or complete snapshot state.

        Returns:
            The verified snapshot.  Call ``validate_against_program`` before resume.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "program_checksum",
                "states",
                "complete",
                "multistart_evidence",
                "content_checksum",
            }),
            version=SCHEDULED_EXECUTION_SNAPSHOT_SCHEMA_VERSION,
            name="scheduled execution snapshot",
        )
        raw_states = mapping["states"]
        if type(raw_states) is not tuple:
            msg = "states must be a JSON array."
            raise TypeError(msg)
        raw_evidence = mapping["multistart_evidence"]
        snapshot = cls(
            program_checksum=cast("str", mapping["program_checksum"]),
            states=tuple(ScheduledOptimizerState.from_dict(item) for item in raw_states),
            multistart_evidence=(None if raw_evidence is None else MultistartWorkEvidence.from_dict(raw_evidence)),
        )
        if mapping["complete"] != snapshot.complete:
            msg = "Serialized snapshot completion flag is inconsistent."
            raise ValueError(msg)
        return snapshot

    @classmethod
    def from_json(cls, payload: str) -> ScheduledExecutionSnapshot:
        """Decode canonical JSON into intrinsic snapshot state.

        Returns:
            The verified snapshot.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class ScheduledTrainingGradientRequest:
    """Training-only Krotov-gradient request with no validation seam."""

    update_request_checksum: str
    objective_checksum: str
    policy: ScheduledTrainingPolicy
    parameters: tuple[float, ...]
    schema_version: str = field(default=SCHEDULED_TRAINING_GRADIENT_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the objective, policy, and recoverable parameter vector."""
        object.__setattr__(
            self,
            "update_request_checksum",
            require_checksum(self.update_request_checksum, "update_request_checksum"),
        )
        object.__setattr__(self, "objective_checksum", require_checksum(self.objective_checksum, "objective_checksum"))
        if not isinstance(self.policy, ScheduledTrainingPolicy):
            msg = "policy must be a ScheduledTrainingPolicy."
            raise TypeError(msg)
        object.__setattr__(self, "parameters", _float_tuple(self.parameters, "parameters"))

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered training-gradient field."""
        return {
            "schema_version": self.schema_version,
            "update_request_checksum": self.update_request_checksum,
            "objective_checksum": self.objective_checksum,
            "policy": self.policy.to_dict(),
            "parameters": list(self.parameters),
            "parameter_checksum": _vector_checksum(self.parameters),
            "data_role": "training_trajectory",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete gradient request."""
        return canonical_checksum(self._payload())


@dataclass(frozen=True, slots=True)
class ScheduledTrainingGradientResult:
    """One typed gradient result bound to its exact training request."""

    request_checksum: str
    objective_checksum: str
    gradient: tuple[float, ...]
    schema_version: str = field(default=SCHEDULED_TRAINING_GRADIENT_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate request identity and finite gradient."""
        object.__setattr__(self, "request_checksum", require_checksum(self.request_checksum, "request_checksum"))
        object.__setattr__(self, "objective_checksum", require_checksum(self.objective_checksum, "objective_checksum"))
        object.__setattr__(self, "gradient", _float_tuple(self.gradient, "gradient"))

    @classmethod
    def for_request(
        cls,
        request: ScheduledTrainingGradientRequest,
        gradient: Sequence[float],
    ) -> ScheduledTrainingGradientResult:
        """Bind a numerical gradient to one request.

        Returns:
            The exact typed gradient result.
        """
        return cls(request.content_checksum, request.objective_checksum, tuple(gradient))

    @property
    def content_checksum(self) -> str:
        """Checksum covering the exact gradient result."""
        return canonical_checksum({
            "schema_version": self.schema_version,
            "request_checksum": self.request_checksum,
            "objective_checksum": self.objective_checksum,
            "gradient": list(self.gradient),
            "data_role": "training_trajectory",
        })


@dataclass(frozen=True, slots=True)
class ScheduledTrainingObjectiveRequest:
    """One paired training-loss request for Adam, SPSA, or growth."""

    update_request_checksum: str
    objective_checksum: str
    policy: ScheduledTrainingPolicy
    parameters: tuple[float, ...]
    evaluation_kind: ObjectiveEvaluationKind
    pair_index: int
    pair_random_seed: int
    schema_version: str = field(default=SCHEDULED_TRAINING_OBJECTIVE_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the CRN-paired training-only loss request."""
        object.__setattr__(
            self,
            "update_request_checksum",
            require_checksum(self.update_request_checksum, "update_request_checksum"),
        )
        object.__setattr__(self, "objective_checksum", require_checksum(self.objective_checksum, "objective_checksum"))
        if not isinstance(self.policy, ScheduledTrainingPolicy):
            msg = "policy must be a ScheduledTrainingPolicy."
            raise TypeError(msg)
        object.__setattr__(self, "parameters", _float_tuple(self.parameters, "parameters"))
        if self.evaluation_kind not in {"gradient_plus", "gradient_minus"}:
            msg = "evaluation_kind must be gradient_plus or gradient_minus."
            raise ValueError(msg)
        object.__setattr__(self, "pair_index", require_int(self.pair_index, "pair_index"))
        object.__setattr__(self, "pair_random_seed", _uint64(self.pair_random_seed, "pair_random_seed"))

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered training-loss field."""
        return {
            "schema_version": self.schema_version,
            "update_request_checksum": self.update_request_checksum,
            "objective_checksum": self.objective_checksum,
            "policy": self.policy.to_dict(),
            "parameters": list(self.parameters),
            "parameter_checksum": _vector_checksum(self.parameters),
            "evaluation_kind": self.evaluation_kind,
            "pair_index": self.pair_index,
            "pair_random_seed": self.pair_random_seed,
            "data_role": "training_trajectory",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete objective request."""
        return canonical_checksum(self._payload())


@dataclass(frozen=True, slots=True)
class ScheduledTrainingObjectiveResult:
    """Finite training loss sealed to its exact paired request."""

    request_checksum: str
    objective_checksum: str
    loss: float
    schema_version: str = field(default=SCHEDULED_TRAINING_OBJECTIVE_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact request/objective identity and finite loss."""
        object.__setattr__(self, "request_checksum", require_checksum(self.request_checksum, "request_checksum"))
        object.__setattr__(self, "objective_checksum", require_checksum(self.objective_checksum, "objective_checksum"))
        object.__setattr__(self, "loss", require_float(self.loss, "loss"))

    @classmethod
    def for_request(
        cls,
        request: ScheduledTrainingObjectiveRequest,
        loss: float,
    ) -> ScheduledTrainingObjectiveResult:
        """Bind one numerical loss to its exact request.

        Returns:
            The typed loss result.
        """
        return cls(request.content_checksum, request.objective_checksum, loss)


class ScheduledTrainingGradientExecutor(Protocol):
    """Target-bound numerical Krotov gradient seam."""

    def __call__(self, request: ScheduledTrainingGradientRequest) -> ScheduledTrainingGradientResult:
        """Evaluate one exact repository Krotov contribution."""
        ...


class ScheduledTrainingObjectiveExecutor(Protocol):
    """Target-bound numerical training-loss seam."""

    def __call__(self, request: ScheduledTrainingObjectiveRequest) -> ScheduledTrainingObjectiveResult:
        """Evaluate one exact repository state-preparation loss."""
        ...


def _trajectory_multiplier(policy: ScheduledTrainingPolicy) -> int:
    """Return one for noiseless work and R for sampled work."""
    return max(1, policy.trajectory_count)


def _pair_seed(request: ScheduledUpdateRequest, pair_index: int) -> int:
    """Derive one CRN seed shared by the plus/minus pair.

    Returns:
        The isolated unsigned pair seed.
    """
    return derive_role_seed(
        request.seed_bundle.training_trajectory_seed,
        "training_trajectory",
        purpose="scheduled_objective_pair",
        stream_index=request.policy.start_index,
        epoch=request.policy.update,
        member_index=pair_index,
    )


def _objective_loss(
    executor: ScheduledTrainingObjectiveExecutor,
    request: ScheduledTrainingObjectiveRequest,
) -> float:
    """Evaluate and verify one target-bound training loss.

    Returns:
        The verified finite loss.
    """
    result = executor(request)
    if not isinstance(result, ScheduledTrainingObjectiveResult):
        msg = "Training objective must return ScheduledTrainingObjectiveResult."
        raise TypeError(msg)
    if result.request_checksum != request.content_checksum or result.objective_checksum != request.objective_checksum:
        msg = "Training objective result is not bound to its exact request."
        raise ValueError(msg)
    return result.loss


@dataclass(frozen=True, slots=True)
class KrotovScheduledUpdateAdapter:
    """Repository-owned one-step Krotov batch-update adapter."""

    objective_checksum: str
    gradient_executor: ScheduledTrainingGradientExecutor = field(repr=False, compare=False)
    cross_trajectory: bool = False
    optimizer_kind: Literal["krotov"] = field(default="krotov", init=False)

    def __post_init__(self) -> None:
        """Validate exact objective identity and callable runtime seam."""
        object.__setattr__(self, "objective_checksum", require_checksum(self.objective_checksum, "objective_checksum"))
        if not callable(self.gradient_executor):
            msg = "gradient_executor must be callable."
            raise TypeError(msg)
        if type(self.cross_trajectory) is not bool:
            msg = "cross_trajectory must be a bool."
            raise TypeError(msg)

    @property
    def content_checksum(self) -> str:
        """Checksum of this repository adapter's numerical contract."""
        return canonical_checksum({
            "adapter": "krotov_scheduled_update",
            "objective_checksum": self.objective_checksum,
            "cross_trajectory": self.cross_trajectory,
            "scheduled_member_execution": "per_member_sample_then_fixed_map_replay",
        })

    def quote_normalized_work(self, request: ScheduledUpdateRequest) -> float:
        """Quote sampling, replay, and optional dense cross-pair work.

        Returns:
            The deterministic complete training work.
        """
        count = request.policy.trajectory_count
        if count == 0:
            return 1.0
        return float(2 * count + (count * count if self.cross_trajectory else 0))

    def __call__(self, request: ScheduledUpdateRequest) -> ScheduledUpdateResult:
        """Apply one full-batch Krotov contribution without resetting state.

        Returns:
            The exact post-update payload and deterministic work quote.
        """
        state = request.optimizer_payload
        if not isinstance(state, KrotovOptimizerPayload):
            msg = "Krotov adapter requires KrotovOptimizerPayload."
            raise TypeError(msg)
        gradient_request = ScheduledTrainingGradientRequest(
            request.content_checksum,
            self.objective_checksum,
            request.policy,
            state.parameters,
        )
        evaluation = self.gradient_executor(gradient_request)
        if not isinstance(evaluation, ScheduledTrainingGradientResult):
            msg = "gradient_executor must return ScheduledTrainingGradientResult."
            raise TypeError(msg)
        if (
            evaluation.request_checksum != gradient_request.content_checksum
            or evaluation.objective_checksum != self.objective_checksum
            or len(evaluation.gradient) != len(state.parameters)
        ):
            msg = "Krotov gradient result differs from its exact request or parameter dimension."
            raise ValueError(msg)
        parameters = tuple(
            parameter - state.step_size * gradient
            for parameter, gradient in zip(state.parameters, evaluation.gradient, strict=True)
        )
        next_state = replace(state, parameters=parameters, completed_updates=state.completed_updates + 1)
        return ScheduledUpdateResult(next_state, self.content_checksum, self.quote_normalized_work(request))


@dataclass(frozen=True, slots=True)
class ParameterShiftAdamScheduledUpdateAdapter:
    """Repository-owned one-step exact parameter-shift Adam adapter."""

    objective_checksum: str
    parameter_shift_scales: tuple[float, ...]
    objective_executor: ScheduledTrainingObjectiveExecutor = field(repr=False, compare=False)
    optimizer_kind: Literal["parameter_shift_adam"] = field(default="parameter_shift_adam", init=False)

    def __post_init__(self) -> None:
        """Validate objective, exact Pauli scales, and runtime seam."""
        object.__setattr__(self, "objective_checksum", require_checksum(self.objective_checksum, "objective_checksum"))
        scales = _float_tuple(self.parameter_shift_scales, "parameter_shift_scales")
        if any(math.isclose(scale, 0.0, rel_tol=0.0, abs_tol=0.0) for scale in scales):
            msg = "parameter_shift_scales must be nonzero."
            raise ValueError(msg)
        object.__setattr__(self, "parameter_shift_scales", scales)
        if not callable(self.objective_executor):
            msg = "objective_executor must be callable."
            raise TypeError(msg)

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact Adam update contract."""
        return canonical_checksum({
            "adapter": "parameter_shift_adam_scheduled_update",
            "objective_checksum": self.objective_checksum,
            "parameter_shift_scales": list(self.parameter_shift_scales),
        })

    def quote_normalized_work(self, request: ScheduledUpdateRequest) -> float:
        """Quote every paired parameter-shift objective evaluation.

        Returns:
            The deterministic complete training work.
        """
        return float(2 * len(self.parameter_shift_scales) * _trajectory_multiplier(request.policy))

    def __call__(self, request: ScheduledUpdateRequest) -> ScheduledUpdateResult:
        """Apply one exact Adam update and preserve both moment vectors.

        Returns:
            The exact post-update payload and deterministic work quote.
        """
        state = request.optimizer_payload
        if not isinstance(state, AdamOptimizerPayload):
            msg = "Adam adapter requires AdamOptimizerPayload."
            raise TypeError(msg)
        if len(state.parameters) != len(self.parameter_shift_scales):
            msg = "Adam parameter count differs from the exact parameter-shift scale vector."
            raise ValueError(msg)
        gradient: list[float] = []
        for index, scale in enumerate(self.parameter_shift_scales):
            shift = math.pi / (2.0 * scale)
            plus = list(state.parameters)
            minus = list(state.parameters)
            plus[index] += shift
            minus[index] -= shift
            seed = _pair_seed(request, index)
            plus_request = ScheduledTrainingObjectiveRequest(
                request.content_checksum,
                self.objective_checksum,
                request.policy,
                tuple(plus),
                "gradient_plus",
                index,
                seed,
            )
            minus_request = replace(plus_request, parameters=tuple(minus), evaluation_kind="gradient_minus")
            gradient.append(
                0.5
                * scale
                * (
                    _objective_loss(self.objective_executor, plus_request)
                    - _objective_loss(self.objective_executor, minus_request)
                )
            )
        config = state.config
        first = tuple(
            config.beta1 * moment + (1.0 - config.beta1) * value
            for moment, value in zip(state.first_moment, gradient, strict=True)
        )
        second = tuple(
            config.beta2 * moment + (1.0 - config.beta2) * value * value
            for moment, value in zip(state.second_moment, gradient, strict=True)
        )
        iteration = state.completed_updates + 1
        parameters = tuple(
            parameter
            - config.learning_rate
            * (moment / (1.0 - config.beta1**iteration))
            / (math.sqrt(square / (1.0 - config.beta2**iteration)) + config.epsilon)
            for parameter, moment, square in zip(state.parameters, first, second, strict=True)
        )
        next_state = replace(
            state,
            parameters=parameters,
            completed_updates=iteration,
            first_moment=first,
            second_moment=second,
        )
        return ScheduledUpdateResult(next_state, self.content_checksum, self.quote_normalized_work(request))


@dataclass(frozen=True, slots=True)
class SPSAScheduledUpdateAdapter:
    """Repository-owned one-step counter-seeded SPSA adapter."""

    objective_checksum: str
    objective_executor: ScheduledTrainingObjectiveExecutor = field(repr=False, compare=False)
    optimizer_kind: Literal["spsa"] = field(default="spsa", init=False)

    def __post_init__(self) -> None:
        """Validate objective identity and runtime seam."""
        object.__setattr__(self, "objective_checksum", require_checksum(self.objective_checksum, "objective_checksum"))
        if not callable(self.objective_executor):
            msg = "objective_executor must be callable."
            raise TypeError(msg)

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact SPSA update contract."""
        return canonical_checksum({
            "adapter": "spsa_scheduled_update",
            "objective_checksum": self.objective_checksum,
            "perturbation_derivation": "wp22_program_ordering_seed_one_based_update_sha256_v1",
        })

    @staticmethod
    def quote_normalized_work(request: ScheduledUpdateRequest) -> float:
        """Quote the paired SPSA objective evaluations.

        Returns:
            The deterministic complete training work.
        """
        return float(2 * _trajectory_multiplier(request.policy))

    def __call__(self, request: ScheduledUpdateRequest) -> ScheduledUpdateResult:
        """Apply one deterministic SPSA update and persist its exact RNG seed.

        Returns:
            The exact post-update payload and deterministic work quote.
        """
        state = request.optimizer_payload
        if not isinstance(state, SPSAOptimizerPayload):
            msg = "SPSA adapter requires SPSAOptimizerPayload."
            raise TypeError(msg)
        iteration = state.completed_updates + 1
        seed_checksum = canonical_checksum({
            "derivation_version": "yaqs.state_preparation.phase2.wp22_spsa_perturbation.v1",
            "program_checksum": request.program_checksum,
            "optimizer_ordering_seed": state.initialization.seed_bundle.optimizer_ordering_seed,
            "iteration": iteration,
        })
        perturbation_seed = int(seed_checksum.removeprefix("sha256:")[:16], 16)
        rng = np.random.Generator(np.random.PCG64(perturbation_seed))
        perturbation = tuple(float(2 * item - 1) for item in rng.integers(0, 2, size=len(state.parameters)))
        learning_rate, scale = state.config.gains(iteration)
        plus = tuple(parameter + scale * delta for parameter, delta in zip(state.parameters, perturbation, strict=True))
        minus = tuple(
            parameter - scale * delta for parameter, delta in zip(state.parameters, perturbation, strict=True)
        )
        plus_request = ScheduledTrainingObjectiveRequest(
            request.content_checksum,
            self.objective_checksum,
            request.policy,
            plus,
            "gradient_plus",
            0,
            perturbation_seed,
        )
        minus_request = replace(plus_request, parameters=minus, evaluation_kind="gradient_minus")
        factor = (
            _objective_loss(self.objective_executor, plus_request)
            - _objective_loss(self.objective_executor, minus_request)
        ) / (2.0 * scale)
        parameters = tuple(
            parameter - learning_rate * factor * delta
            for parameter, delta in zip(state.parameters, perturbation, strict=True)
        )
        next_state = replace(
            state,
            parameters=parameters,
            completed_updates=iteration,
            last_perturbation_seed=perturbation_seed,
        )
        return ScheduledUpdateResult(next_state, self.content_checksum, self.quote_normalized_work(request))


@dataclass(frozen=True, slots=True)
class OperatorGrowthAdamScheduledUpdateAdapter:
    """Repository-owned one-step full-prefix operator-growth reoptimizer."""

    objective_checksum: str
    objective_executor: ScheduledTrainingObjectiveExecutor = field(repr=False, compare=False)
    optimizer_kind: Literal["operator_growth_adam"] = field(default="operator_growth_adam", init=False)

    def __post_init__(self) -> None:
        """Validate objective identity and runtime seam."""
        object.__setattr__(self, "objective_checksum", require_checksum(self.objective_checksum, "objective_checksum"))
        if not callable(self.objective_executor):
            msg = "objective_executor must be callable."
            raise TypeError(msg)

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact operator-growth Adam contract."""
        return canonical_checksum({
            "adapter": "operator_growth_adam_scheduled_update",
            "objective_checksum": self.objective_checksum,
            "parameter_shift": "pauli_rotation_pi_over_2",
            "best_rule": "strict_objective_decrease",
        })

    @staticmethod
    def quote_normalized_work(request: ScheduledUpdateRequest) -> float:
        """Quote paired gradients plus one post-update objective.

        Returns:
            The deterministic complete training work.
        """
        count = len(request.optimizer_payload.parameters)
        return float((2 * count + 1) * _trajectory_multiplier(request.policy))

    def __call__(self, request: ScheduledUpdateRequest) -> ScheduledUpdateResult:
        """Apply one internal-Adam reoptimization step for the sealed prefix.

        Returns:
            The exact current/best state and deterministic work quote.
        """
        state = request.optimizer_payload
        if not isinstance(state, OperatorGrowthOptimizerPayload):
            msg = "Operator-growth adapter requires OperatorGrowthOptimizerPayload."
            raise TypeError(msg)
        gradient: list[float] = []
        for index in range(len(state.parameters)):
            plus = list(state.parameters)
            minus = list(state.parameters)
            plus[index] += math.pi / 2.0
            minus[index] -= math.pi / 2.0
            seed = _pair_seed(request, index)
            plus_request = ScheduledTrainingObjectiveRequest(
                request.content_checksum,
                self.objective_checksum,
                request.policy,
                tuple(plus),
                "gradient_plus",
                index,
                seed,
            )
            minus_request = replace(plus_request, parameters=tuple(minus), evaluation_kind="gradient_minus")
            gradient.append(
                0.5
                * (
                    _objective_loss(self.objective_executor, plus_request)
                    - _objective_loss(self.objective_executor, minus_request)
                )
            )
        spec = state.spec
        first = tuple(
            spec.adam_beta1 * moment + (1.0 - spec.adam_beta1) * value
            for moment, value in zip(state.first_moment, gradient, strict=True)
        )
        second = tuple(
            spec.adam_beta2 * moment + (1.0 - spec.adam_beta2) * value * value
            for moment, value in zip(state.second_moment, gradient, strict=True)
        )
        iteration = state.completed_updates + 1
        parameters = tuple(
            parameter
            - spec.learning_rate
            * (moment / (1.0 - spec.adam_beta1**iteration))
            / (math.sqrt(square / (1.0 - spec.adam_beta2**iteration)) + spec.adam_epsilon)
            for parameter, moment, square in zip(state.parameters, first, second, strict=True)
        )
        monitor = ScheduledTrainingObjectiveRequest(
            request.content_checksum,
            self.objective_checksum,
            request.policy,
            parameters,
            "gradient_plus",
            len(parameters),
            _pair_seed(request, len(parameters)),
        )
        objective = _objective_loss(self.objective_executor, monitor)
        improved = objective < state.best_objective
        next_state = replace(
            state,
            parameters=parameters,
            completed_updates=iteration,
            first_moment=first,
            second_moment=second,
            best_parameters=parameters if improved else state.best_parameters,
            best_objective=objective if improved else state.best_objective,
        )
        return ScheduledUpdateResult(next_state, self.content_checksum, self.quote_normalized_work(request))


RepositoryScheduledUpdateAdapter: TypeAlias = (
    KrotovScheduledUpdateAdapter
    | ParameterShiftAdamScheduledUpdateAdapter
    | SPSAScheduledUpdateAdapter
    | OperatorGrowthAdamScheduledUpdateAdapter
)
ScheduledUpdateExecutor: TypeAlias = RepositoryScheduledUpdateAdapter


class ScheduledValidationExecutor(Protocol):
    """Validation-only callback for one exact checkpoint request."""

    def __call__(self, request: ScheduledValidationRequest) -> ScheduledValidationResult:
        """Evaluate one checkpoint without final-test access."""
        ...


def initialize_scheduled_execution(
    program: ScheduledExecutionProgram,
    optimizer_payloads: tuple[MethodOptimizerPayload, ...],
) -> ScheduledExecutionSnapshot:
    """Initialize every optimizer start before any executor callback.

    Returns:
        An exact interrupted snapshot at update zero for all starts.
    """
    if not isinstance(program, ScheduledExecutionProgram):
        msg = "program must be a ScheduledExecutionProgram."
        raise TypeError(msg)
    if type(optimizer_payloads) is not tuple or len(optimizer_payloads) != program.start_count:
        msg = "optimizer_payloads must contain exactly one typed payload per optimizer start."
        raise ValueError(msg)
    states = tuple(
        ScheduledOptimizerState.initialize(program, start_index, payload)
        for start_index, payload in enumerate(optimizer_payloads)
    )
    return ScheduledExecutionSnapshot(
        program_checksum=program.content_checksum,
        states=states,
        multistart_evidence=None,
    )


class NormalizedComputeCapError(ValueError):
    """Prospective complete update exceeds the sealed job-wide work cap."""

    def __init__(self, *, cap: float, completed_work: float, prospective_update_work: float) -> None:
        """Store structured cap evidence without invoking a numerical callback."""
        self.cap = cap
        self.completed_work = completed_work
        self.prospective_update_work = prospective_update_work
        super().__init__(
            "Prospective complete scheduled update exceeds normalized_compute_cap "
            f"({completed_work} + {prospective_update_work} > {cap})."
        )


def _validate_adapter_for_program(
    program: ScheduledExecutionProgram,
    adapter: RepositoryScheduledUpdateAdapter,
) -> None:
    """Require a concrete repository adapter matching the executable binding."""
    if type(adapter) not in {
        KrotovScheduledUpdateAdapter,
        ParameterShiftAdamScheduledUpdateAdapter,
        SPSAScheduledUpdateAdapter,
        OperatorGrowthAdamScheduledUpdateAdapter,
    }:
        msg = "update_executor must be a concrete repository scheduled-update adapter."
        raise TypeError(msg)
    expected = _optimizer_kind_for_program(program)
    if expected is not None and adapter.optimizer_kind != expected:
        msg = "Scheduled-update adapter differs from the executable binding's optimizer family."
        raise ValueError(msg)
    stage = _controlled_pipeline_stage(program)
    if isinstance(adapter, KrotovScheduledUpdateAdapter) and stage is not None:
        cross_expected = stage.stage_policy["trajectory_update"] == "cross"
        if adapter.cross_trajectory != cross_expected:
            msg = "Krotov cross-trajectory adapter mode differs from the controlled implementation stage."
            raise ValueError(msg)
    if isinstance(adapter, ParameterShiftAdamScheduledUpdateAdapter) and stage is not None:
        parameter_count = cast("int", stage.stage_policy["output_parameter_count"])
        expected_scales = _expected_parameter_shift_scales(program, parameter_count)
        if adapter.parameter_shift_scales != expected_scales:
            msg = "Adam parameter-shift scales differ from the binding-owned BMPD implementation."
            raise ValueError(msg)


def _validate_state_history_for_adapter(
    program: ScheduledExecutionProgram,
    state: ScheduledOptimizerState,
    adapter: RepositoryScheduledUpdateAdapter,
) -> None:
    """Replay every persisted envelope and work quote without numerical calls."""
    payload = state.initial_optimizer_payload
    previous_receipt_checksum: str | None = None
    for receipt in state.receipts:
        policy = program.policy(state.start_index, receipt.update)
        expected_request = ScheduledUpdateRequest(
            program_checksum=program.content_checksum,
            policy=ScheduledTrainingPolicy.from_compiled(policy),
            seed_bundle=program.start_seed_bundles[state.start_index],
            optimizer_payload=payload,
            previous_receipt_checksum=previous_receipt_checksum,
        )
        if receipt.request != expected_request:
            msg = "Persisted update request differs from its exact program, seeds, or predecessor state."
            raise ValueError(msg)
        quoted_training_work = require_float(
            adapter.quote_normalized_work(expected_request),
            "historical quoted normalized training work",
            minimum=0.0,
        )
        validation_work = float(
            0 if policy.checkpoint_membership is None else policy.checkpoint_membership.trajectory_count
        )
        if (
            receipt.result.adapter_checksum != adapter.content_checksum
            or not math.isclose(
                receipt.result.normalized_work,
                quoted_training_work,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                receipt.normalized_work,
                math.fsum((quoted_training_work, validation_work)),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            msg = "Persisted update result or work differs from the currently bound repository adapter."
            raise ValueError(msg)
        _validate_optimizer_transition(payload, receipt.result.optimizer_payload)
        _validate_payload_for_program(program, receipt.result.optimizer_payload)
        expected_artifact = ParameterCheckpointArtifact.from_payload(
            state.start_index,
            receipt.update,
            receipt.result.optimizer_payload,
        )
        if receipt.parameter_artifact != expected_artifact:
            msg = "Persisted update result does not reproduce its recoverable parameter artifact."
            raise ValueError(msg)
        if policy.checkpoint_due:
            assert policy.checkpoint_membership is not None
            expected_validation = ScheduledValidationRequest(
                program_checksum=program.content_checksum,
                start_index=state.start_index,
                update=receipt.update,
                parameter_artifact=expected_artifact,
                membership=policy.checkpoint_membership,
            )
            if receipt.validation_request != expected_validation or receipt.validation_result is None:
                msg = "Persisted validation request/result differs from the exact post-update checkpoint."
                raise ValueError(msg)
        elif receipt.validation_request is not None or receipt.validation_result is not None:
            msg = "Persisted validation evidence appears outside the frozen checkpoint schedule."
            raise ValueError(msg)
        payload = receipt.result.optimizer_payload
        previous_receipt_checksum = receipt.content_checksum


def _advance_state(
    program: ScheduledExecutionProgram,
    state: ScheduledOptimizerState,
    update_executor: RepositoryScheduledUpdateAdapter,
    validation_executor: ScheduledValidationExecutor | None,
    current_job_work: float,
) -> ScheduledOptimizerState:
    """Execute one exact update and return its complete persisted state.

    Returns:
        The next immutable optimizer state.
    """
    policy = program.policy(state.start_index, state.next_update)
    training_policy = ScheduledTrainingPolicy.from_compiled(policy)
    previous_receipt_checksum = None if not state.receipts else state.receipts[-1].content_checksum
    request = ScheduledUpdateRequest(
        program_checksum=program.content_checksum,
        policy=training_policy,
        seed_bundle=program.start_seed_bundles[state.start_index],
        optimizer_payload=state.optimizer_payload,
        previous_receipt_checksum=previous_receipt_checksum,
    )
    training_work = require_float(
        update_executor.quote_normalized_work(request),
        "quoted normalized training work",
        minimum=0.0,
    )
    validation_work = float(
        0 if policy.checkpoint_membership is None else policy.checkpoint_membership.trajectory_count
    )
    complete_update_work = math.fsum((training_work, validation_work))
    cap = program.normalized_compute_cap
    if cap is not None and math.fsum((current_job_work, complete_update_work)) > cap:
        raise NormalizedComputeCapError(
            cap=cap,
            completed_work=current_job_work,
            prospective_update_work=complete_update_work,
        )
    result = update_executor(request)
    if not isinstance(result, ScheduledUpdateResult):
        msg = "update_executor must return a ScheduledUpdateResult."
        raise TypeError(msg)
    if (
        result.adapter_checksum != update_executor.content_checksum
        or result.normalized_work != training_work
        or result.optimizer_payload.optimizer_kind != state.optimizer_payload.optimizer_kind
        or result.optimizer_payload.completed_updates != state.next_update + 1
        or result.optimizer_payload.initialization != state.optimizer_payload.initialization
    ):
        msg = "Update result differs from its adapter quote, method, counter, or initializer."
        raise ValueError(msg)
    _validate_optimizer_transition(state.optimizer_payload, result.optimizer_payload)
    _validate_payload_for_program(program, result.optimizer_payload)
    parameter_artifact = ParameterCheckpointArtifact.from_payload(
        state.start_index,
        policy.update,
        result.optimizer_payload,
    )
    validation_request = None
    validation_result = None
    tracker = state.validation_tracker
    if policy.checkpoint_due:
        if validation_executor is None:
            msg = "A validation executor is required before scheduled execution begins."
            raise ValueError(msg)
        assert policy.checkpoint_membership is not None
        validation_request = ScheduledValidationRequest(
            program_checksum=program.content_checksum,
            start_index=state.start_index,
            update=policy.update,
            parameter_artifact=parameter_artifact,
            membership=policy.checkpoint_membership,
        )
        validation_result = validation_executor(validation_request)
        if not isinstance(validation_result, ScheduledValidationResult):
            msg = "validation_executor must return a ScheduledValidationResult."
            raise TypeError(msg)
        if (
            validation_result.request_checksum != validation_request.content_checksum
            or validation_result.parameter_checksum != parameter_artifact.parameter_checksum
            or validation_result.membership_checksum != policy.checkpoint_membership.content_checksum
        ):
            msg = "Validation result is not sealed to its exact post-update request."
            raise ValueError(msg)
        tracker = tracker.observe(ValidationCheckpoint(update=policy.update, score=validation_result.score))
    receipt = ScheduledUpdateReceipt(
        program_checksum=program.content_checksum,
        start_index=state.start_index,
        update=policy.update,
        policy_checksum=policy.content_checksum,
        request=request,
        result=result,
        parameter_artifact=parameter_artifact,
        normalized_work=complete_update_work,
        validation_request=validation_request,
        validation_result=validation_result,
        previous_receipt_checksum=previous_receipt_checksum,
    )
    next_update = state.next_update + 1
    terminal_reason: TerminalReason | None = (
        "validation_early_stop"
        if tracker.should_stop
        else "budget_complete"
        if next_update == program.total_updates_per_start
        else None
    )
    return ScheduledOptimizerState(
        program_checksum=program.content_checksum,
        start_index=state.start_index,
        seed_bundle_checksum=state.seed_bundle_checksum,
        initial_optimizer_payload=state.initial_optimizer_payload,
        next_update=next_update,
        optimizer_payload=result.optimizer_payload,
        receipts=(*state.receipts, receipt),
        validation_tracker=tracker,
        last_training_membership=policy.training_membership,
        last_component_memberships=policy.component_memberships,
        total_normalized_work=math.fsum(item.normalized_work for item in (*state.receipts, receipt)),
        terminal_reason=terminal_reason,
    )


def execute_scheduled_program(
    program: ScheduledExecutionProgram,
    snapshot: ScheduledExecutionSnapshot,
    update_executor: RepositoryScheduledUpdateAdapter,
    *,
    validation_executor: ScheduledValidationExecutor | None = None,
    stop_after_updates: int | None = None,
) -> ScheduledExecutionSnapshot:
    """Execute or resume a program through a bounded or terminal snapshot.

    All program and restart-state checks, including the required validation
    callback, happen before the first update callback.  ``stop_after_updates``
    bounds callbacks in this invocation only; resuming the returned canonical
    JSON snapshot is byte-identical to uninterrupted execution.

    Returns:
        The exact interrupted or complete all-start snapshot.

    Raises:
        TypeError: If callbacks or state artifacts use the wrong typed records.
        ValueError: If the program/state pair is inconsistent or validation is unavailable.
    """
    if not isinstance(program, ScheduledExecutionProgram):
        msg = "program must be a ScheduledExecutionProgram."
        raise TypeError(msg)
    if not isinstance(snapshot, ScheduledExecutionSnapshot):
        msg = "snapshot must be a ScheduledExecutionSnapshot."
        raise TypeError(msg)
    _validate_adapter_for_program(program, update_executor)
    if validation_executor is not None and not callable(validation_executor):
        msg = "validation_executor must be callable or None."
        raise TypeError(msg)
    snapshot.validate_against_program(program)
    for state in snapshot.states:
        _validate_state_history_for_adapter(program, state, update_executor)
    historical_job_work = math.fsum(receipt.normalized_work for state in snapshot.states for receipt in state.receipts)
    if program.normalized_compute_cap is not None and historical_job_work > program.normalized_compute_cap:
        msg = "Persisted historical work already exceeds the executable binding's job-wide compute cap."
        raise ValueError(msg)
    if snapshot.complete:
        return snapshot
    if program.checkpoint_updates and validation_executor is None:
        msg = "This program schedules checkpoint observations and requires a validation executor."
        raise ValueError(msg)
    callback_limit = None
    if stop_after_updates is not None:
        callback_limit = require_int(stop_after_updates, "stop_after_updates", minimum=1)
    completed_callbacks = 0
    current_job_work = math.fsum(state.total_normalized_work for state in snapshot.states)
    states: list[ScheduledOptimizerState] = []
    for initial_state in snapshot.states:
        state = initial_state
        while not state.is_terminal and (callback_limit is None or completed_callbacks < callback_limit):
            before_work = state.total_normalized_work
            state = _advance_state(program, state, update_executor, validation_executor, current_job_work)
            current_job_work = math.fsum((current_job_work, state.total_normalized_work - before_work))
            completed_callbacks += 1
        states.append(state)
    result_states = tuple(states)
    complete = all(state.is_terminal for state in result_states)
    evidence = MultistartWorkEvidence.from_states(program, result_states) if complete else None
    result = ScheduledExecutionSnapshot(
        program_checksum=program.content_checksum,
        states=result_states,
        multistart_evidence=evidence,
    )
    result.validate_against_program(program)
    return result


@dataclass(frozen=True, slots=True)
class OperatorGrowthSelectionRequest:
    """Validation-blind request to append one operator before a prefix."""

    program_checksum: str
    growth_spec_checksum: str
    pool_checksum: str
    prefix_index: int
    global_update_start: int
    selected_operator_ids: tuple[str, ...]
    parameters: tuple[float, ...]
    policy: ScheduledTrainingPolicy
    previous_transition_checksum: str | None
    schema_version: str = field(default=OPERATOR_GROWTH_SELECTION_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the exact pre-prefix state and training-only membership."""
        for name in ("program_checksum", "growth_spec_checksum", "pool_checksum"):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        prefix = require_int(self.prefix_index, "prefix_index")
        update = require_int(self.global_update_start, "global_update_start")
        if update != prefix * OPERATOR_GROWTH_PREFIX_UPDATE_COUNT:
            msg = "Operator-growth prefix and global update coordinates disagree."
            raise ValueError(msg)
        operators = tuple(require_slug(item, "selected_operator_id") for item in self.selected_operator_ids)
        if len(operators) != prefix or len(operators) != len(set(operators)):
            msg = "Pre-prefix operator sequence must contain exactly one operator per completed prefix."
            raise ValueError(msg)
        parameters = _possibly_empty_float_tuple(self.parameters, "parameters", length=len(operators))
        if not isinstance(self.policy, ScheduledTrainingPolicy):
            msg = "policy must be a ScheduledTrainingPolicy."
            raise TypeError(msg)
        if (
            self.policy.update != update
            or self.policy.start_index != 0
            or self.policy.trajectory_count != OPERATOR_GROWTH_TRAINING_TRAJECTORY_COUNT
            or self.policy.training_membership is None
        ):
            msg = "Structural selection requires the exact fixed eight-member prefix-start policy."
            raise ValueError(msg)
        if self.previous_transition_checksum is not None:
            object.__setattr__(
                self,
                "previous_transition_checksum",
                require_checksum(self.previous_transition_checksum, "previous_transition_checksum"),
            )
        if (prefix == 0) != (self.previous_transition_checksum is None):
            msg = "Only the first structural selection may omit a predecessor transition."
            raise ValueError(msg)
        object.__setattr__(self, "prefix_index", prefix)
        object.__setattr__(self, "global_update_start", update)
        object.__setattr__(self, "selected_operator_ids", operators)
        object.__setattr__(self, "parameters", parameters)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered selection input."""
        return {
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "growth_spec_checksum": self.growth_spec_checksum,
            "pool_checksum": self.pool_checksum,
            "prefix_index": self.prefix_index,
            "global_update_start": self.global_update_start,
            "selected_operator_ids": list(self.selected_operator_ids),
            "parameters": list(self.parameters),
            "policy": self.policy.to_dict(),
            "previous_transition_checksum": self.previous_transition_checksum,
            "accessible_data_role": "training_trajectory",
            "validation_access": "forbidden",
            "final_test_access": "forbidden",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete structural-selection request."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthSelectionRequest:
        """Decode one strict structural-selection request.

        Returns:
            The verified request.
        """
        mapping = verify_sealed_mapping(
            value,
            expected_keys=frozenset({
                "schema_version",
                "program_checksum",
                "growth_spec_checksum",
                "pool_checksum",
                "prefix_index",
                "global_update_start",
                "selected_operator_ids",
                "parameters",
                "policy",
                "previous_transition_checksum",
                "accessible_data_role",
                "validation_access",
                "final_test_access",
                "content_checksum",
            }),
            name="operator-growth selection request",
        )
        if (
            mapping["schema_version"] != OPERATOR_GROWTH_SELECTION_REQUEST_SCHEMA_VERSION
            or mapping["accessible_data_role"] != "training_trajectory"
            or mapping["validation_access"] != "forbidden"
            or mapping["final_test_access"] != "forbidden"
        ):
            msg = "Operator-growth selection request schema or data-role isolation changed."
            raise ValueError(msg)
        return cls(
            program_checksum=cast("str", mapping["program_checksum"]),
            growth_spec_checksum=cast("str", mapping["growth_spec_checksum"]),
            pool_checksum=cast("str", mapping["pool_checksum"]),
            prefix_index=cast("int", mapping["prefix_index"]),
            global_update_start=cast("int", mapping["global_update_start"]),
            selected_operator_ids=cast("tuple[str, ...]", mapping["selected_operator_ids"]),
            parameters=cast("tuple[float, ...]", mapping["parameters"]),
            policy=ScheduledTrainingPolicy.from_dict(mapping["policy"]),
            previous_transition_checksum=cast("str | None", mapping["previous_transition_checksum"]),
        )


@dataclass(frozen=True, slots=True)
class OperatorGrowthSelectionResult:
    """Complete ordered structural-gradient evidence and selected operator."""

    request_checksum: str
    candidate_gradients: tuple[CandidateGradient, ...]
    selected_operator_id: str
    selected_gradient: float
    objective_before_reoptimization: float
    objective_evidence: tuple[OperatorGrowthSegmentedObjectiveEvidence, ...]
    normalized_work: float
    schema_version: str = field(default=OPERATOR_GROWTH_SELECTION_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate complete candidate, objective, selection, and work evidence."""
        object.__setattr__(self, "request_checksum", require_checksum(self.request_checksum, "request_checksum"))
        candidates = tuple(self.candidate_gradients)
        if not candidates or any(not isinstance(item, CandidateGradient) for item in candidates):
            msg = "candidate_gradients must contain the complete nonempty CandidateGradient universe."
            raise TypeError(msg)
        if len({item.operator_id for item in candidates}) != len(candidates):
            msg = "candidate_gradients must not duplicate operator identities."
            raise ValueError(msg)
        object.__setattr__(
            self, "selected_operator_id", require_slug(self.selected_operator_id, "selected_operator_id")
        )
        gradient = require_float(self.selected_gradient, "selected_gradient")
        selected = tuple(item for item in candidates if item.operator_id == self.selected_operator_id)
        if len(selected) != 1 or selected[0].gradient != gradient or not selected[0].native_cap_feasible:
            msg = "selected operator and gradient must identify one feasible candidate."
            raise ValueError(msg)
        evidence = tuple(self.objective_evidence)
        if not evidence or any(not isinstance(item, OperatorGrowthSegmentedObjectiveEvidence) for item in evidence):
            msg = "objective_evidence must contain typed segmented objective evidence."
            raise TypeError(msg)
        object.__setattr__(self, "candidate_gradients", candidates)
        object.__setattr__(self, "selected_gradient", gradient)
        object.__setattr__(
            self,
            "objective_before_reoptimization",
            require_float(self.objective_before_reoptimization, "objective_before_reoptimization"),
        )
        object.__setattr__(self, "objective_evidence", evidence)
        object.__setattr__(
            self,
            "normalized_work",
            require_float(self.normalized_work, "normalized_work", minimum=0.0),
        )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered selection result field."""
        return {
            "schema_version": self.schema_version,
            "request_checksum": self.request_checksum,
            "candidate_gradients": [item.to_dict() for item in self.candidate_gradients],
            "selected_operator_id": self.selected_operator_id,
            "selected_gradient": self.selected_gradient,
            "objective_before_reoptimization": self.objective_before_reoptimization,
            "objective_evidence": [item.to_dict() for item in self.objective_evidence],
            "normalized_work": self.normalized_work,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the selected structural extension."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthSelectionResult:
        """Decode one strict structural-selection result.

        Returns:
            The verified selection result.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "request_checksum",
                "candidate_gradients",
                "selected_operator_id",
                "selected_gradient",
                "objective_before_reoptimization",
                "objective_evidence",
                "normalized_work",
                "content_checksum",
            }),
            version=OPERATOR_GROWTH_SELECTION_RESULT_SCHEMA_VERSION,
            name="operator-growth selection result",
        )
        raw_candidates = mapping["candidate_gradients"]
        raw_evidence = mapping["objective_evidence"]
        if type(raw_candidates) is not tuple or type(raw_evidence) is not tuple:
            msg = "Serialized selection candidates and objective evidence must be JSON arrays."
            raise TypeError(msg)
        return cls(
            request_checksum=cast("str", mapping["request_checksum"]),
            candidate_gradients=tuple(CandidateGradient.from_dict(item) for item in raw_candidates),
            selected_operator_id=cast("str", mapping["selected_operator_id"]),
            selected_gradient=cast("float", mapping["selected_gradient"]),
            objective_before_reoptimization=cast("float", mapping["objective_before_reoptimization"]),
            objective_evidence=tuple(OperatorGrowthSegmentedObjectiveEvidence.from_dict(item) for item in raw_evidence),
            normalized_work=cast("float", mapping["normalized_work"]),
        )


@dataclass(frozen=True, slots=True)
class OperatorGrowthPrefixTransition:
    """Checksum-sealed append transition opening one 100-update prefix."""

    request: OperatorGrowthSelectionRequest
    result: OperatorGrowthSelectionResult
    selected_operator_ids: tuple[str, ...]
    initial_parameters: tuple[float, ...]
    previous_transition_checksum: str | None
    schema_version: str = field(default=OPERATOR_GROWTH_PREFIX_TRANSITION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require append-only structure and exact Adam moment reset semantics."""
        if not isinstance(self.request, OperatorGrowthSelectionRequest) or not isinstance(
            self.result, OperatorGrowthSelectionResult
        ):
            msg = "Prefix transition requires typed selection request and result."
            raise TypeError(msg)
        expected_operators = (*self.request.selected_operator_ids, self.result.selected_operator_id)
        expected_parameters = (*self.request.parameters, 0.0)
        if (
            self.result.request_checksum != self.request.content_checksum
            or self.selected_operator_ids != expected_operators
            or self.initial_parameters != expected_parameters
            or self.previous_transition_checksum != self.request.previous_transition_checksum
        ):
            msg = "Prefix transition is not the exact append-at-zero structural extension."
            raise ValueError(msg)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered structural transition field."""
        return {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "request_checksum": self.request.content_checksum,
            "result": self.result.to_dict(),
            "result_checksum": self.result.content_checksum,
            "selected_operator_ids": list(self.selected_operator_ids),
            "initial_parameters": list(self.initial_parameters),
            "adam_first_moment_reset": [0.0] * len(self.initial_parameters),
            "adam_second_moment_reset": [0.0] * len(self.initial_parameters),
            "local_update_counter_reset": 0,
            "previous_transition_checksum": self.previous_transition_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the exact append and moment reset."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthPrefixTransition:
        """Decode one strict structural prefix transition.

        Returns:
            The verified transition.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "request",
                "request_checksum",
                "result",
                "result_checksum",
                "selected_operator_ids",
                "initial_parameters",
                "adam_first_moment_reset",
                "adam_second_moment_reset",
                "local_update_counter_reset",
                "previous_transition_checksum",
                "content_checksum",
            }),
            version=OPERATOR_GROWTH_PREFIX_TRANSITION_SCHEMA_VERSION,
            name="operator-growth prefix transition",
        )
        transition = cls(
            request=OperatorGrowthSelectionRequest.from_dict(mapping["request"]),
            result=OperatorGrowthSelectionResult.from_dict(mapping["result"]),
            selected_operator_ids=cast("tuple[str, ...]", mapping["selected_operator_ids"]),
            initial_parameters=cast("tuple[float, ...]", mapping["initial_parameters"]),
            previous_transition_checksum=cast("str | None", mapping["previous_transition_checksum"]),
        )
        zeros = (0.0,) * len(transition.initial_parameters)
        if (
            mapping["request_checksum"] != transition.request.content_checksum
            or mapping["result_checksum"] != transition.result.content_checksum
            or mapping["adam_first_moment_reset"] != zeros
            or mapping["adam_second_moment_reset"] != zeros
            or mapping["local_update_counter_reset"] != 0
        ):
            msg = "Serialized prefix transition aliases or Adam reset semantics changed."
            raise ValueError(msg)
        return transition


@dataclass(frozen=True, slots=True)
class OperatorGrowthSegmentedObjectiveRequest:
    """Training-only objective request for one selected operator prefix."""

    program_checksum: str
    structural_state_checksum: str
    selected_operator_ids: tuple[str, ...]
    prefix_index: int
    global_update: int
    local_update: int
    evaluation_stage: Literal["structural_selection", "prefix_reoptimization"]
    evaluation_kind: Literal["gradient_plus", "gradient_minus", "post_update"]
    parameter_index: int
    parameters: tuple[float, ...]
    policy: ScheduledTrainingPolicy
    schema_version: str = field(default=OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate coordinates, prefix identity, membership, and role isolation."""
        object.__setattr__(self, "program_checksum", require_checksum(self.program_checksum, "program_checksum"))
        object.__setattr__(
            self,
            "structural_state_checksum",
            require_checksum(self.structural_state_checksum, "structural_state_checksum"),
        )
        operators = tuple(require_slug(item, "selected_operator_id") for item in self.selected_operator_ids)
        prefix = require_int(self.prefix_index, "prefix_index")
        global_update = require_int(self.global_update, "global_update")
        local_update = require_int(self.local_update, "local_update")
        if (
            len(operators) != prefix + 1
            or global_update != prefix * OPERATOR_GROWTH_PREFIX_UPDATE_COUNT + local_update
            or not 0 <= local_update < OPERATOR_GROWTH_PREFIX_UPDATE_COUNT
        ):
            msg = "Segmented objective prefix and update coordinates disagree."
            raise ValueError(msg)
        if self.evaluation_stage not in {"structural_selection", "prefix_reoptimization"}:
            msg = "evaluation_stage is unsupported."
            raise ValueError(msg)
        if self.evaluation_kind not in {"gradient_plus", "gradient_minus", "post_update"}:
            msg = "evaluation_kind is unsupported."
            raise ValueError(msg)
        index = require_int(self.parameter_index, "parameter_index")
        if index > len(operators) or (self.evaluation_kind != "post_update" and index >= len(operators)):
            msg = "Objective parameter_index differs from its evaluation kind."
            raise ValueError(msg)
        parameters = _float_tuple(self.parameters, "parameters", length=len(operators))
        if (
            not isinstance(self.policy, ScheduledTrainingPolicy)
            or self.policy.update != global_update
            or self.policy.trajectory_count != OPERATOR_GROWTH_TRAINING_TRAJECTORY_COUNT
            or self.policy.training_membership is None
        ):
            msg = "Segmented objective requires the exact fixed eight-member scheduled policy."
            raise ValueError(msg)
        object.__setattr__(self, "selected_operator_ids", operators)
        object.__setattr__(self, "prefix_index", prefix)
        object.__setattr__(self, "global_update", global_update)
        object.__setattr__(self, "local_update", local_update)
        object.__setattr__(self, "parameter_index", index)
        object.__setattr__(self, "parameters", parameters)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered objective input."""
        return {
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "structural_state_checksum": self.structural_state_checksum,
            "selected_operator_ids": list(self.selected_operator_ids),
            "prefix_index": self.prefix_index,
            "global_update": self.global_update,
            "local_update": self.local_update,
            "evaluation_stage": self.evaluation_stage,
            "evaluation_kind": self.evaluation_kind,
            "parameter_index": self.parameter_index,
            "parameters": list(self.parameters),
            "policy": self.policy.to_dict(),
            "accessible_data_role": "training_trajectory",
            "validation_access": "forbidden",
            "final_test_access": "forbidden",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering one validation-blind objective request."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthSegmentedObjectiveRequest:
        """Decode one strict segmented objective request.

        Returns:
            The verified request.
        """
        mapping = verify_sealed_mapping(
            value,
            expected_keys=frozenset({
                "schema_version",
                "program_checksum",
                "structural_state_checksum",
                "selected_operator_ids",
                "prefix_index",
                "global_update",
                "local_update",
                "evaluation_stage",
                "evaluation_kind",
                "parameter_index",
                "parameters",
                "policy",
                "accessible_data_role",
                "validation_access",
                "final_test_access",
                "content_checksum",
            }),
            name="operator-growth segmented objective request",
        )
        if (
            mapping["schema_version"] != OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_REQUEST_SCHEMA_VERSION
            or mapping["accessible_data_role"] != "training_trajectory"
            or mapping["validation_access"] != "forbidden"
            or mapping["final_test_access"] != "forbidden"
        ):
            msg = "Segmented objective schema or data-role isolation changed."
            raise ValueError(msg)
        return cls(
            program_checksum=cast("str", mapping["program_checksum"]),
            structural_state_checksum=cast("str", mapping["structural_state_checksum"]),
            selected_operator_ids=cast("tuple[str, ...]", mapping["selected_operator_ids"]),
            prefix_index=cast("int", mapping["prefix_index"]),
            global_update=cast("int", mapping["global_update"]),
            local_update=cast("int", mapping["local_update"]),
            evaluation_stage=cast(
                'Literal["structural_selection", "prefix_reoptimization"]',
                mapping["evaluation_stage"],
            ),
            evaluation_kind=cast(
                'Literal["gradient_plus", "gradient_minus", "post_update"]', mapping["evaluation_kind"]
            ),
            parameter_index=cast("int", mapping["parameter_index"]),
            parameters=cast("tuple[float, ...]", mapping["parameters"]),
            policy=ScheduledTrainingPolicy.from_dict(mapping["policy"]),
        )


@dataclass(frozen=True, slots=True)
class OperatorGrowthSegmentedObjectiveResult:
    """Finite training objective value bound to one exact request."""

    request_checksum: str
    objective: float
    schema_version: str = field(default=OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact request identity and finite objective."""
        object.__setattr__(self, "request_checksum", require_checksum(self.request_checksum, "request_checksum"))
        object.__setattr__(self, "objective", require_float(self.objective, "objective"))

    @classmethod
    def for_request(
        cls,
        request: OperatorGrowthSegmentedObjectiveRequest,
        objective: float,
    ) -> OperatorGrowthSegmentedObjectiveResult:
        """Bind a finite objective to its exact request.

        Returns:
            The checksum-linked objective result.
        """
        return cls(request.content_checksum, objective)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered objective result field."""
        return {
            "schema_version": self.schema_version,
            "request_checksum": self.request_checksum,
            "objective": self.objective,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the exact objective result."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthSegmentedObjectiveResult:
        """Decode one strict segmented objective result.

        Returns:
            The verified result.
        """
        mapping = _verify(
            value,
            keys=frozenset({"schema_version", "request_checksum", "objective", "content_checksum"}),
            version=OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_RESULT_SCHEMA_VERSION,
            name="operator-growth segmented objective result",
        )
        return cls(
            request_checksum=cast("str", mapping["request_checksum"]),
            objective=cast("float", mapping["objective"]),
        )


@dataclass(frozen=True, slots=True)
class OperatorGrowthSegmentedObjectiveEvidence:
    """One checksum-linked training objective request and finite result."""

    request: OperatorGrowthSegmentedObjectiveRequest
    result: OperatorGrowthSegmentedObjectiveResult
    schema_version: str = field(
        default=OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_EVIDENCE_SCHEMA_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        """Require typed evidence with an exact request-result link."""
        if not isinstance(self.request, OperatorGrowthSegmentedObjectiveRequest) or not isinstance(
            self.result,
            OperatorGrowthSegmentedObjectiveResult,
        ):
            msg = "Segmented objective evidence requires typed request and result records."
            raise TypeError(msg)
        if self.result.request_checksum != self.request.content_checksum:
            msg = "Segmented objective result does not identify its exact request."
            raise ValueError(msg)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered evidence field."""
        return {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "request_checksum": self.request.content_checksum,
            "result": self.result.to_dict(),
            "result_checksum": self.result.content_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the exact objective evidence pair."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthSegmentedObjectiveEvidence:
        """Decode one strict objective evidence pair.

        Returns:
            The verified objective evidence.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "request",
                "request_checksum",
                "result",
                "result_checksum",
                "content_checksum",
            }),
            version=OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_EVIDENCE_SCHEMA_VERSION,
            name="operator-growth segmented objective evidence",
        )
        evidence = cls(
            request=OperatorGrowthSegmentedObjectiveRequest.from_dict(mapping["request"]),
            result=OperatorGrowthSegmentedObjectiveResult.from_dict(mapping["result"]),
        )
        if (
            mapping["request_checksum"] != evidence.request.content_checksum
            or mapping["result_checksum"] != evidence.result.content_checksum
        ):
            msg = "Serialized objective evidence checksum aliases changed."
            raise ValueError(msg)
        return evidence


@dataclass(frozen=True, slots=True)
class OperatorGrowthPrefixValidation:
    """Validation-only evidence emitted after one complete 100-update prefix."""

    prefix_index: int
    transition_checksum: str
    request: ScheduledValidationRequest
    result: ScheduledValidationResult
    normalized_work: float
    previous_validation_checksum: str | None
    schema_version: str = field(default=OPERATOR_GROWTH_PREFIX_VALIDATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require exact prefix coordinates, role isolation, and 256-member work."""
        prefix = require_int(self.prefix_index, "prefix_index")
        object.__setattr__(self, "prefix_index", prefix)
        object.__setattr__(
            self,
            "transition_checksum",
            require_checksum(self.transition_checksum, "transition_checksum"),
        )
        if not isinstance(self.request, ScheduledValidationRequest) or not isinstance(
            self.result,
            ScheduledValidationResult,
        ):
            msg = "Prefix validation requires typed validation request and result records."
            raise TypeError(msg)
        expected_update = (prefix + 1) * OPERATOR_GROWTH_PREFIX_UPDATE_COUNT - 1
        if (
            self.request.start_index != 0
            or self.request.update != expected_update
            or self.request.membership.trajectory_count != OPERATOR_GROWTH_VALIDATION_TRAJECTORY_COUNT
            or self.result.request_checksum != self.request.content_checksum
            or self.result.parameter_checksum != self.request.parameter_artifact.parameter_checksum
            or self.result.membership_checksum != self.request.membership.content_checksum
        ):
            msg = "Prefix validation coordinates, membership, or checksum links changed."
            raise ValueError(msg)
        work = require_float(self.normalized_work, "normalized_work", minimum=0.0)
        if not math.isclose(
            work,
            float(OPERATOR_GROWTH_VALIDATION_TRAJECTORY_COUNT),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            msg = "Prefix validation work must equal its exact 256-member ensemble."
            raise ValueError(msg)
        object.__setattr__(self, "normalized_work", work)
        if self.previous_validation_checksum is not None:
            object.__setattr__(
                self,
                "previous_validation_checksum",
                require_checksum(self.previous_validation_checksum, "previous_validation_checksum"),
            )
        if (prefix == 0) != (self.previous_validation_checksum is None):
            msg = "Only the first prefix validation may omit its predecessor."
            raise ValueError(msg)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered prefix-validation field."""
        return {
            "schema_version": self.schema_version,
            "prefix_index": self.prefix_index,
            "transition_checksum": self.transition_checksum,
            "request": self.request.to_dict(),
            "request_checksum": self.request.content_checksum,
            "result": self.result.to_dict(),
            "result_checksum": self.result.content_checksum,
            "normalized_work": self.normalized_work,
            "previous_validation_checksum": self.previous_validation_checksum,
            "selection_rule": "greatest_validation_fidelity_earliest_growth_step",
            "final_test_access": "forbidden",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering one completed-prefix validation."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthPrefixValidation:
        """Decode one strict completed-prefix validation.

        Returns:
            The verified prefix validation.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "prefix_index",
                "transition_checksum",
                "request",
                "request_checksum",
                "result",
                "result_checksum",
                "normalized_work",
                "previous_validation_checksum",
                "selection_rule",
                "final_test_access",
                "content_checksum",
            }),
            version=OPERATOR_GROWTH_PREFIX_VALIDATION_SCHEMA_VERSION,
            name="operator-growth prefix validation",
        )
        if (
            mapping["selection_rule"] != "greatest_validation_fidelity_earliest_growth_step"
            or mapping["final_test_access"] != "forbidden"
        ):
            msg = "Prefix-validation selection or data-role invariant changed."
            raise ValueError(msg)
        validation = cls(
            prefix_index=cast("int", mapping["prefix_index"]),
            transition_checksum=cast("str", mapping["transition_checksum"]),
            request=ScheduledValidationRequest.from_dict(mapping["request"]),
            result=ScheduledValidationResult.from_dict(mapping["result"]),
            normalized_work=cast("float", mapping["normalized_work"]),
            previous_validation_checksum=cast("str | None", mapping["previous_validation_checksum"]),
        )
        if (
            mapping["request_checksum"] != validation.request.content_checksum
            or mapping["result_checksum"] != validation.result.content_checksum
        ):
            msg = "Serialized prefix-validation checksum aliases changed."
            raise ValueError(msg)
        return validation


@dataclass(frozen=True, slots=True)
class OperatorGrowthSegmentedUpdateReceipt:
    """Exact Adam state transition and objective evidence for one global update."""

    program_checksum: str
    prefix_index: int
    global_update: int
    local_update: int
    transition_checksum: str
    selected_operator_ids: tuple[str, ...]
    policy: ScheduledTrainingPolicy
    parameters_before: tuple[float, ...]
    first_moment_before: tuple[float, ...]
    second_moment_before: tuple[float, ...]
    gradient: tuple[float, ...]
    parameters_after: tuple[float, ...]
    first_moment_after: tuple[float, ...]
    second_moment_after: tuple[float, ...]
    best_parameters: tuple[float, ...]
    best_objective: float
    objective_evidence: tuple[OperatorGrowthSegmentedObjectiveEvidence, ...]
    normalized_work: float
    prefix_validation: OperatorGrowthPrefixValidation | None
    previous_receipt_checksum: str | None
    schema_version: str = field(default=OPERATOR_GROWTH_SEGMENTED_UPDATE_RECEIPT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate coordinates, parameter-shift evidence, and complete work."""
        object.__setattr__(self, "program_checksum", require_checksum(self.program_checksum, "program_checksum"))
        prefix = require_int(self.prefix_index, "prefix_index")
        global_update = require_int(self.global_update, "global_update")
        local_update = require_int(self.local_update, "local_update")
        if (
            prefix not in {0, 1}
            or global_update != prefix * OPERATOR_GROWTH_PREFIX_UPDATE_COUNT + local_update
            or not 0 <= local_update < OPERATOR_GROWTH_PREFIX_UPDATE_COUNT
        ):
            msg = "Segmented receipt prefix and update coordinates disagree."
            raise ValueError(msg)
        object.__setattr__(self, "prefix_index", prefix)
        object.__setattr__(self, "global_update", global_update)
        object.__setattr__(self, "local_update", local_update)
        object.__setattr__(
            self,
            "transition_checksum",
            require_checksum(self.transition_checksum, "transition_checksum"),
        )
        operators = tuple(require_slug(item, "selected_operator_id") for item in self.selected_operator_ids)
        if len(operators) != prefix + 1 or len(operators) != len(set(operators)):
            msg = "Segmented receipt operator sequence differs from its prefix."
            raise ValueError(msg)
        object.__setattr__(self, "selected_operator_ids", operators)
        if (
            not isinstance(self.policy, ScheduledTrainingPolicy)
            or self.policy.start_index != 0
            or self.policy.update != global_update
            or self.policy.trajectory_count != OPERATOR_GROWTH_TRAINING_TRAJECTORY_COUNT
            or self.policy.training_membership is None
        ):
            msg = "Segmented receipt requires the exact fixed eight-member scheduled policy."
            raise ValueError(msg)
        vector_fields = (
            "parameters_before",
            "first_moment_before",
            "second_moment_before",
            "gradient",
            "parameters_after",
            "first_moment_after",
            "second_moment_after",
            "best_parameters",
        )
        for name in vector_fields:
            object.__setattr__(self, name, _float_tuple(getattr(self, name), name, length=len(operators)))
        object.__setattr__(self, "best_objective", require_float(self.best_objective, "best_objective"))
        evidence = tuple(self.objective_evidence)
        expected_count = 2 * len(operators) + 1
        if len(evidence) != expected_count or any(
            not isinstance(item, OperatorGrowthSegmentedObjectiveEvidence) for item in evidence
        ):
            msg = "Each Adam receipt requires two objectives per parameter and one post-update objective."
            raise ValueError(msg)
        for parameter_index in range(len(operators)):
            plus = evidence[2 * parameter_index].request
            minus = evidence[2 * parameter_index + 1].request
            expected_plus = list(self.parameters_before)
            expected_minus = list(self.parameters_before)
            expected_plus[parameter_index] += math.pi / 2.0
            expected_minus[parameter_index] -= math.pi / 2.0
            if (
                plus.evaluation_stage != "prefix_reoptimization"
                or minus.evaluation_stage != "prefix_reoptimization"
                or plus.evaluation_kind != "gradient_plus"
                or minus.evaluation_kind != "gradient_minus"
                or plus.parameter_index != parameter_index
                or minus.parameter_index != parameter_index
                or plus.parameters != tuple(expected_plus)
                or minus.parameters != tuple(expected_minus)
            ):
                msg = "Receipt parameter-shift requests changed their exact pair ordering or angles."
                raise ValueError(msg)
            expected_gradient = 0.5 * (
                evidence[2 * parameter_index].result.objective - evidence[2 * parameter_index + 1].result.objective
            )
            if not math.isclose(
                self.gradient[parameter_index],
                expected_gradient,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                msg = "Receipt gradient differs from its retained parameter-shift objectives."
                raise ValueError(msg)
        monitor = evidence[-1].request
        if (
            monitor.evaluation_stage != "prefix_reoptimization"
            or monitor.evaluation_kind != "post_update"
            or monitor.parameter_index != len(operators)
            or monitor.parameters != self.parameters_after
        ):
            msg = "Receipt post-update monitor differs from the persisted Adam parameters."
            raise ValueError(msg)
        for item in evidence:
            request = item.request
            if (
                request.program_checksum != self.program_checksum
                or request.structural_state_checksum != self.transition_checksum
                or request.selected_operator_ids != operators
                or request.prefix_index != prefix
                or request.global_update != global_update
                or request.local_update != local_update
                or request.policy != self.policy
            ):
                msg = "Receipt objective evidence differs from its structural or schedule envelope."
                raise ValueError(msg)
        object.__setattr__(self, "objective_evidence", evidence)
        if self.prefix_validation is not None:
            if not isinstance(self.prefix_validation, OperatorGrowthPrefixValidation):
                msg = "prefix_validation must be typed or None."
                raise TypeError(msg)
            if (
                local_update != OPERATOR_GROWTH_PREFIX_UPDATE_COUNT - 1
                or self.prefix_validation.prefix_index != prefix
                or self.prefix_validation.transition_checksum != self.transition_checksum
            ):
                msg = "Prefix validation may occur only on its completed 100th update."
                raise ValueError(msg)
        elif local_update == OPERATOR_GROWTH_PREFIX_UPDATE_COUNT - 1:
            msg = "Every complete 100-update prefix requires validation evidence."
            raise ValueError(msg)
        expected_work = float(expected_count * OPERATOR_GROWTH_TRAINING_TRAJECTORY_COUNT)
        if self.prefix_validation is not None:
            expected_work = math.fsum((expected_work, self.prefix_validation.normalized_work))
        work = require_float(self.normalized_work, "normalized_work", minimum=0.0)
        if work != expected_work:
            msg = "Segmented receipt work omits or duplicates objective or validation trajectories."
            raise ValueError(msg)
        object.__setattr__(self, "normalized_work", work)
        if self.previous_receipt_checksum is not None:
            object.__setattr__(
                self,
                "previous_receipt_checksum",
                require_checksum(self.previous_receipt_checksum, "previous_receipt_checksum"),
            )
        if (global_update == 0) != (self.previous_receipt_checksum is None):
            msg = "Only global update zero may begin the segmented receipt chain."
            raise ValueError(msg)

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered segmented receipt field."""
        return {
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "prefix_index": self.prefix_index,
            "global_update": self.global_update,
            "local_update": self.local_update,
            "transition_checksum": self.transition_checksum,
            "selected_operator_ids": list(self.selected_operator_ids),
            "policy": self.policy.to_dict(),
            "policy_checksum": self.policy.content_checksum,
            "parameters_before": list(self.parameters_before),
            "first_moment_before": list(self.first_moment_before),
            "second_moment_before": list(self.second_moment_before),
            "gradient": list(self.gradient),
            "parameters_after": list(self.parameters_after),
            "first_moment_after": list(self.first_moment_after),
            "second_moment_after": list(self.second_moment_after),
            "best_parameters": list(self.best_parameters),
            "best_objective": self.best_objective,
            "objective_evidence": [item.to_dict() for item in self.objective_evidence],
            "normalized_work": self.normalized_work,
            "prefix_validation": (None if self.prefix_validation is None else self.prefix_validation.to_dict()),
            "prefix_validation_checksum": (
                None if self.prefix_validation is None else self.prefix_validation.content_checksum
            ),
            "previous_receipt_checksum": self.previous_receipt_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete global update receipt."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthSegmentedUpdateReceipt:
        """Decode one strict segmented update receipt.

        Returns:
            The verified update receipt.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "program_checksum",
                "prefix_index",
                "global_update",
                "local_update",
                "transition_checksum",
                "selected_operator_ids",
                "policy",
                "policy_checksum",
                "parameters_before",
                "first_moment_before",
                "second_moment_before",
                "gradient",
                "parameters_after",
                "first_moment_after",
                "second_moment_after",
                "best_parameters",
                "best_objective",
                "objective_evidence",
                "normalized_work",
                "prefix_validation",
                "prefix_validation_checksum",
                "previous_receipt_checksum",
                "content_checksum",
            }),
            version=OPERATOR_GROWTH_SEGMENTED_UPDATE_RECEIPT_SCHEMA_VERSION,
            name="operator-growth segmented update receipt",
        )
        raw_evidence = mapping["objective_evidence"]
        if type(raw_evidence) is not tuple:
            msg = "objective_evidence must be a JSON array."
            raise TypeError(msg)
        raw_validation = mapping["prefix_validation"]
        receipt = cls(
            program_checksum=cast("str", mapping["program_checksum"]),
            prefix_index=cast("int", mapping["prefix_index"]),
            global_update=cast("int", mapping["global_update"]),
            local_update=cast("int", mapping["local_update"]),
            transition_checksum=cast("str", mapping["transition_checksum"]),
            selected_operator_ids=cast("tuple[str, ...]", mapping["selected_operator_ids"]),
            policy=ScheduledTrainingPolicy.from_dict(mapping["policy"]),
            parameters_before=cast("tuple[float, ...]", mapping["parameters_before"]),
            first_moment_before=cast("tuple[float, ...]", mapping["first_moment_before"]),
            second_moment_before=cast("tuple[float, ...]", mapping["second_moment_before"]),
            gradient=cast("tuple[float, ...]", mapping["gradient"]),
            parameters_after=cast("tuple[float, ...]", mapping["parameters_after"]),
            first_moment_after=cast("tuple[float, ...]", mapping["first_moment_after"]),
            second_moment_after=cast("tuple[float, ...]", mapping["second_moment_after"]),
            best_parameters=cast("tuple[float, ...]", mapping["best_parameters"]),
            best_objective=cast("float", mapping["best_objective"]),
            objective_evidence=tuple(OperatorGrowthSegmentedObjectiveEvidence.from_dict(item) for item in raw_evidence),
            normalized_work=cast("float", mapping["normalized_work"]),
            prefix_validation=(
                None if raw_validation is None else OperatorGrowthPrefixValidation.from_dict(raw_validation)
            ),
            previous_receipt_checksum=cast("str | None", mapping["previous_receipt_checksum"]),
        )
        if mapping["policy_checksum"] != receipt.policy.content_checksum or mapping["prefix_validation_checksum"] != (
            None if receipt.prefix_validation is None else receipt.prefix_validation.content_checksum
        ):
            msg = "Serialized segmented receipt checksum aliases changed."
            raise ValueError(msg)
        return receipt


@dataclass(frozen=True, slots=True)
class OperatorGrowthSegmentedSnapshot:
    """Authoritative restart and selected-prefix result for two growth prefixes."""

    program_checksum: str
    pool: OperatorPoolSpec
    growth_spec: OperatorGrowthSpec
    active_operator_ids: tuple[str, ...]
    parameters: tuple[float, ...]
    first_moment: tuple[float, ...]
    second_moment: tuple[float, ...]
    best_training_parameters: tuple[float, ...]
    best_training_objective: float | None
    transitions: tuple[OperatorGrowthPrefixTransition, ...]
    receipts: tuple[OperatorGrowthSegmentedUpdateReceipt, ...]
    prefix_validations: tuple[OperatorGrowthPrefixValidation, ...]
    total_normalized_work: float
    selected_prefix_index: int | None
    selected_operator_ids: tuple[str, ...]
    selected_parameters: tuple[float, ...]
    selected_validation_score: float | None
    terminal_reason: Literal["update_budget"] | None
    schema_version: str = field(default=OPERATOR_GROWTH_SEGMENTED_SNAPSHOT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate every chain, active state, work total, and validation selection."""
        object.__setattr__(self, "program_checksum", require_checksum(self.program_checksum, "program_checksum"))
        if not isinstance(self.pool, OperatorPoolSpec) or not isinstance(self.growth_spec, OperatorGrowthSpec):
            msg = "Segmented snapshot requires exact pool and growth-spec documents."
            raise TypeError(msg)
        if (
            self.growth_spec.pool_checksum != self.pool.content_checksum
            or self.growth_spec.method_id != self.pool.method_id
        ):
            msg = "Segmented snapshot growth spec is not bound to its exact pool."
            raise ValueError(msg)
        transitions = tuple(self.transitions)
        receipts = tuple(self.receipts)
        validations = tuple(self.prefix_validations)
        if any(not isinstance(item, OperatorGrowthPrefixTransition) for item in transitions):
            msg = "transitions must contain only OperatorGrowthPrefixTransition records."
            raise TypeError(msg)
        if any(not isinstance(item, OperatorGrowthSegmentedUpdateReceipt) for item in receipts):
            msg = "receipts must contain only OperatorGrowthSegmentedUpdateReceipt records."
            raise TypeError(msg)
        if any(not isinstance(item, OperatorGrowthPrefixValidation) for item in validations):
            msg = "prefix_validations must contain only OperatorGrowthPrefixValidation records."
            raise TypeError(msg)
        if len(receipts) > OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT:
            msg = "Segmented snapshot exceeds the exact 200-update budget."
            raise ValueError(msg)
        expected_transition_count = (
            0 if not receipts else (len(receipts) - 1) // OPERATOR_GROWTH_PREFIX_UPDATE_COUNT + 1
        )
        if len(transitions) != expected_transition_count or len(validations) != len(receipts) // 100:
            msg = "Transition or validation count differs from completed segmented work."
            raise ValueError(msg)
        previous_transition_checksum: str | None = None
        for prefix, transition in enumerate(transitions):
            if (
                transition.request.prefix_index != prefix
                or transition.request.global_update_start != prefix * OPERATOR_GROWTH_PREFIX_UPDATE_COUNT
                or transition.previous_transition_checksum != previous_transition_checksum
            ):
                msg = "Structural transition chain or prefix coordinates changed."
                raise ValueError(msg)
            previous_transition_checksum = transition.content_checksum
        previous_receipt_checksum: str | None = None
        receipt_validations: list[OperatorGrowthPrefixValidation] = []
        for update, receipt in enumerate(receipts):
            prefix = update // OPERATOR_GROWTH_PREFIX_UPDATE_COUNT
            if (
                receipt.program_checksum != self.program_checksum
                or receipt.global_update != update
                or receipt.prefix_index != prefix
                or receipt.local_update != update % OPERATOR_GROWTH_PREFIX_UPDATE_COUNT
                or receipt.transition_checksum != transitions[prefix].content_checksum
                or receipt.selected_operator_ids != transitions[prefix].selected_operator_ids
                or receipt.previous_receipt_checksum != previous_receipt_checksum
            ):
                msg = "Segmented receipt chain differs from its program or structural prefix."
                raise ValueError(msg)
            previous_receipt_checksum = receipt.content_checksum
            if receipt.prefix_validation is not None:
                receipt_validations.append(receipt.prefix_validation)
        if tuple(receipt_validations) != validations:
            msg = "Snapshot validation aliases differ from validation-bearing receipts."
            raise ValueError(msg)
        previous_validation_checksum: str | None = None
        for prefix, validation in enumerate(validations):
            if (
                validation.prefix_index != prefix
                or validation.transition_checksum != transitions[prefix].content_checksum
                or validation.previous_validation_checksum != previous_validation_checksum
            ):
                msg = "Completed-prefix validation chain changed."
                raise ValueError(msg)
            previous_validation_checksum = validation.content_checksum
        active = tuple(require_slug(item, "active_operator_id") for item in self.active_operator_ids)
        expected_active = () if not transitions else transitions[-1].selected_operator_ids
        if active != expected_active:
            msg = "Active operator sequence differs from the latest structural transition."
            raise ValueError(msg)
        state_length = len(active)
        vector_fields = (
            "parameters",
            "first_moment",
            "second_moment",
            "best_training_parameters",
        )
        for name in vector_fields:
            object.__setattr__(
                self,
                name,
                _possibly_empty_float_tuple(getattr(self, name), name, length=state_length),
            )
        if receipts:
            last = receipts[-1]
            if (
                self.parameters != last.parameters_after
                or self.first_moment != last.first_moment_after
                or self.second_moment != last.second_moment_after
                or self.best_training_parameters != last.best_parameters
                or self.best_training_objective != last.best_objective
            ):
                msg = "Snapshot active Adam state differs from its latest receipt."
                raise ValueError(msg)
            object.__setattr__(
                self,
                "best_training_objective",
                require_float(self.best_training_objective, "best_training_objective"),
            )
        elif (
            active
            or self.parameters
            or self.first_moment
            or self.second_moment
            or self.best_training_parameters
            or self.best_training_objective is not None
        ):
            msg = "An update-zero segmented snapshot must contain no active optimizer state."
            raise ValueError(msg)
        expected_work = math.fsum(
            [transition.result.normalized_work for transition in transitions]
            + [receipt.normalized_work for receipt in receipts]
        )
        work = require_float(self.total_normalized_work, "total_normalized_work", minimum=0.0)
        if work != expected_work:
            msg = "Snapshot normalized work differs from all structural, training, and validation evidence."
            raise ValueError(msg)
        object.__setattr__(self, "total_normalized_work", work)
        if validations:
            expected_selected = max(
                range(len(validations)),
                key=lambda index: validations[index].result.score,
            )
            expected_selected_operators = transitions[expected_selected].selected_operator_ids
            expected_selected_parameters = validations[expected_selected].request.parameter_artifact.parameters
            expected_selected_score = validations[expected_selected].result.score
        else:
            expected_selected = None
            expected_selected_operators = ()
            expected_selected_parameters = ()
            expected_selected_score = None
        selected_ids = tuple(require_slug(item, "selected_operator_id") for item in self.selected_operator_ids)
        selected_parameters = _possibly_empty_float_tuple(
            self.selected_parameters,
            "selected_parameters",
            length=len(selected_ids),
        )
        if (
            self.selected_prefix_index != expected_selected
            or selected_ids != expected_selected_operators
            or selected_parameters != expected_selected_parameters
            or self.selected_validation_score != expected_selected_score
        ):
            msg = "Snapshot selected prefix differs from greatest validation and earliest-prefix tie-break."
            raise ValueError(msg)
        if self.selected_prefix_index is not None:
            object.__setattr__(
                self,
                "selected_prefix_index",
                require_int(self.selected_prefix_index, "selected_prefix_index"),
            )
            object.__setattr__(
                self,
                "selected_validation_score",
                require_float(
                    self.selected_validation_score,
                    "selected_validation_score",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(self, "active_operator_ids", active)
        object.__setattr__(self, "selected_operator_ids", selected_ids)
        object.__setattr__(self, "selected_parameters", selected_parameters)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "prefix_validations", validations)
        expected_terminal = "update_budget" if len(receipts) == OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT else None
        if self.terminal_reason != expected_terminal:
            msg = "Only the exact 200th global update may set update_budget termination."
            raise ValueError(msg)

    @property
    def next_global_update(self) -> int:
        """Next global optimizer-update coordinate."""
        return len(self.receipts)

    @property
    def complete(self) -> bool:
        """Whether both exact 100-update prefixes are complete."""
        return self.terminal_reason == "update_budget"

    @property
    def selected_transition(self) -> OperatorGrowthPrefixTransition | None:
        """The validation-selected structural prefix transition."""
        return None if self.selected_prefix_index is None else self.transitions[self.selected_prefix_index]

    @classmethod
    def initialize(cls, program: ScheduledExecutionProgram) -> OperatorGrowthSegmentedSnapshot:
        """Create the empty authoritative snapshot for one exact q6 program.

        Returns:
            The checksum-bound update-zero segmented snapshot.
        """
        execution_spec = _validate_operator_growth_segmented_program(program)
        return cls(
            program_checksum=program.content_checksum,
            pool=execution_spec.pool,
            growth_spec=execution_spec.growth_spec,
            active_operator_ids=(),
            parameters=(),
            first_moment=(),
            second_moment=(),
            best_training_parameters=(),
            best_training_objective=None,
            transitions=(),
            receipts=(),
            prefix_validations=(),
            total_normalized_work=0.0,
            selected_prefix_index=None,
            selected_operator_ids=(),
            selected_parameters=(),
            selected_validation_score=None,
            terminal_reason=None,
        )

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered snapshot field."""
        return {
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "pool": self.pool.to_dict(),
            "pool_checksum": self.pool.content_checksum,
            "growth_spec": self.growth_spec.to_dict(),
            "growth_spec_checksum": self.growth_spec.content_checksum,
            "active_operator_ids": list(self.active_operator_ids),
            "parameters": list(self.parameters),
            "first_moment": list(self.first_moment),
            "second_moment": list(self.second_moment),
            "best_training_parameters": list(self.best_training_parameters),
            "best_training_objective": self.best_training_objective,
            "transitions": [item.to_dict() for item in self.transitions],
            "receipts": [item.to_dict() for item in self.receipts],
            "prefix_validations": [item.to_dict() for item in self.prefix_validations],
            "total_normalized_work": self.total_normalized_work,
            "selected_prefix_index": self.selected_prefix_index,
            "selected_operator_ids": list(self.selected_operator_ids),
            "selected_parameters": list(self.selected_parameters),
            "selected_validation_score": self.selected_validation_score,
            "terminal_reason": self.terminal_reason,
            "selection_rule": "greatest_validation_fidelity_earliest_growth_step",
            "global_update_budget": OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT,
            "updates_per_prefix": OPERATOR_GROWTH_PREFIX_UPDATE_COUNT,
            "final_test_access": "forbidden",
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete authoritative structural result."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed restart JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> OperatorGrowthSegmentedSnapshot:
        """Decode one strict authoritative segmented snapshot.

        Returns:
            The verified snapshot.
        """
        mapping = _verify(
            value,
            keys=frozenset({
                "schema_version",
                "program_checksum",
                "pool",
                "pool_checksum",
                "growth_spec",
                "growth_spec_checksum",
                "active_operator_ids",
                "parameters",
                "first_moment",
                "second_moment",
                "best_training_parameters",
                "best_training_objective",
                "transitions",
                "receipts",
                "prefix_validations",
                "total_normalized_work",
                "selected_prefix_index",
                "selected_operator_ids",
                "selected_parameters",
                "selected_validation_score",
                "terminal_reason",
                "selection_rule",
                "global_update_budget",
                "updates_per_prefix",
                "final_test_access",
                "content_checksum",
            }),
            version=OPERATOR_GROWTH_SEGMENTED_SNAPSHOT_SCHEMA_VERSION,
            name="operator-growth segmented snapshot",
        )
        if (
            mapping["selection_rule"] != "greatest_validation_fidelity_earliest_growth_step"
            or mapping["global_update_budget"] != OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT
            or mapping["updates_per_prefix"] != OPERATOR_GROWTH_PREFIX_UPDATE_COUNT
            or mapping["final_test_access"] != "forbidden"
        ):
            msg = "Segmented snapshot protocol invariants changed."
            raise ValueError(msg)
        raw_transitions = mapping["transitions"]
        raw_receipts = mapping["receipts"]
        raw_validations = mapping["prefix_validations"]
        if type(raw_transitions) is not tuple or type(raw_receipts) is not tuple or type(raw_validations) is not tuple:
            msg = "Snapshot histories must be JSON arrays."
            raise TypeError(msg)
        snapshot = cls(
            program_checksum=cast("str", mapping["program_checksum"]),
            pool=OperatorPoolSpec.from_dict(mapping["pool"]),
            growth_spec=OperatorGrowthSpec.from_dict(mapping["growth_spec"]),
            active_operator_ids=cast("tuple[str, ...]", mapping["active_operator_ids"]),
            parameters=cast("tuple[float, ...]", mapping["parameters"]),
            first_moment=cast("tuple[float, ...]", mapping["first_moment"]),
            second_moment=cast("tuple[float, ...]", mapping["second_moment"]),
            best_training_parameters=cast("tuple[float, ...]", mapping["best_training_parameters"]),
            best_training_objective=cast("float | None", mapping["best_training_objective"]),
            transitions=tuple(OperatorGrowthPrefixTransition.from_dict(item) for item in raw_transitions),
            receipts=tuple(OperatorGrowthSegmentedUpdateReceipt.from_dict(item) for item in raw_receipts),
            prefix_validations=tuple(OperatorGrowthPrefixValidation.from_dict(item) for item in raw_validations),
            total_normalized_work=cast("float", mapping["total_normalized_work"]),
            selected_prefix_index=cast("int | None", mapping["selected_prefix_index"]),
            selected_operator_ids=cast("tuple[str, ...]", mapping["selected_operator_ids"]),
            selected_parameters=cast("tuple[float, ...]", mapping["selected_parameters"]),
            selected_validation_score=cast("float | None", mapping["selected_validation_score"]),
            terminal_reason=cast('Literal["update_budget"] | None', mapping["terminal_reason"]),
        )
        if (
            mapping["pool_checksum"] != snapshot.pool.content_checksum
            or mapping["growth_spec_checksum"] != snapshot.growth_spec.content_checksum
        ):
            msg = "Serialized snapshot pool or growth-spec checksum alias changed."
            raise ValueError(msg)
        return snapshot

    @classmethod
    def from_json(cls, payload: str) -> OperatorGrowthSegmentedSnapshot:
        """Decode canonical restart JSON.

        Returns:
            The verified segmented snapshot.
        """
        return cls.from_dict(load_canonical_json_object(payload))

    def validate_against_program(self, program: ScheduledExecutionProgram) -> None:
        """Replay every structural and numerical transition against its exact program."""
        _validate_operator_growth_segmented_history(program, self)


class OperatorGrowthSegmentedObjectiveExecutor(Protocol):
    """Training-only numerical seam shared by selection and reoptimization."""

    def __call__(
        self,
        request: OperatorGrowthSegmentedObjectiveRequest,
    ) -> OperatorGrowthSegmentedObjectiveResult:
        """Evaluate one exact fixed-CRN objective request."""
        ...


class OperatorGrowthStructuralSelectionExecutor(Protocol):
    """Numerical structural-selection seam with a deterministic work quote."""

    def quote_normalized_work(self, request: OperatorGrowthSelectionRequest) -> float:
        """Quote all fixed-CRN candidate and appended-zero objectives."""
        ...

    def __call__(
        self,
        request: OperatorGrowthSelectionRequest,
        objective_executor: OperatorGrowthSegmentedObjectiveExecutor,
    ) -> OperatorGrowthSelectionResult:
        """Return complete ordered candidate and objective evidence."""
        ...


def _validate_operator_growth_segmented_program(
    program: ScheduledExecutionProgram,
) -> OperatorGrowthExecutionSpec:
    """Require the exact q6 scientific binding and two-prefix scheduled policy.

    Returns:
        The binding-owned exact q6 operator-growth execution specification.
    """
    if not isinstance(program, ScheduledExecutionProgram):
        msg = "program must be a ScheduledExecutionProgram."
        raise TypeError(msg)
    if program.execution_scope != "scoped_binding" or program.executable_binding is None:
        msg = "Segmented operator growth requires one complete scoped executable binding."
        raise ValueError(msg)
    binding = program.executable_binding.binding
    payload = binding.implementation_artifact.implementation_payload
    execution_spec = binding.operator_growth_spec
    if (
        binding.publication_method_id != "adapt_style_state_preparation"
        or binding.preset != "paper-screen"
        or binding.target_scope_id != "primary_q6"
        or binding.qubit_count != 6
        or binding.execution_data_role != "screening_selection"
        or binding.implementation_artifact.implementation_kind != "operator_growth"
        or binding.controlled_stage.implementation_stage_id != "operator_growth_reoptimization"
        or not isinstance(execution_spec, OperatorGrowthExecutionSpec)
        or not isinstance(payload, OperatorGrowthExecutionSpec)
        or payload != execution_spec
        or execution_spec.outer_evaluation_policy.purpose != "screening_outer"
    ):
        msg = "Segmented execution accepts only the exact primary-q6 operator-growth implementation."
        raise ValueError(msg)
    spec = execution_spec.growth_spec
    if (
        program.total_updates_per_start != OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT
        or program.start_count != 1
        or program.schedule.schedule_id != "direct_matched_fixed_crn"
        or spec.reoptimization_steps != OPERATOR_GROWTH_PREFIX_UPDATE_COUNT
        or spec.max_operators < 2
        or program.checkpoint_updates != (99, 199)
        or program.checkpoint_validation_trajectory_count != OPERATOR_GROWTH_VALIDATION_TRAJECTORY_COUNT
    ):
        msg = "Operator-growth program differs from the exact two-prefix WP22 schedule."
        raise ValueError(msg)
    memberships: list[tuple[int, ...]] = []
    for update in range(OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT):
        policy = program.policy(0, update)
        membership = policy.training_membership
        if (
            policy.trajectory_count != OPERATOR_GROWTH_TRAINING_TRAJECTORY_COUNT
            or membership is None
            or membership.role != "training_trajectory"
            or not math.isclose(policy.noise_strength_scale, 1.0, rel_tol=0.0, abs_tol=0.0)
            or policy.checkpoint_due != (update in {99, 199})
            or (policy.checkpoint_membership is not None) != (update in {99, 199})
        ):
            msg = "Operator-growth policies differ from fixed-eight CRN training or prefix validation."
            raise ValueError(msg)
        memberships.append(membership.member_seeds)
        if policy.checkpoint_membership is not None and (
            policy.checkpoint_membership.trajectory_count != OPERATOR_GROWTH_VALIDATION_TRAJECTORY_COUNT
        ):
            msg = "Operator-growth prefix validation must use exactly 256 trajectories."
            raise ValueError(msg)
    if len(set(memberships)) != 1:
        msg = "Operator-growth training membership must remain one fixed eight-member CRN ensemble."
        raise ValueError(msg)
    return execution_spec


def _operator_growth_candidate_metadata(
    pool: OperatorPoolSpec,
    spec: OperatorGrowthSpec,
    selected_operator_ids: tuple[str, ...],
) -> tuple[tuple[int, PoolOperator, bool], ...]:
    """Return the complete remaining pool order and native-cap feasibility."""
    selected = set(selected_operator_ids)
    by_id = {operator.operator_id: operator for operator in pool.operators}
    if len(selected) != len(selected_operator_ids) or any(item not in by_id for item in selected):
        msg = "Selected structural prefix is not a unique subsequence of the exact pool."
        raise ValueError(msg)
    edge_counts = [0] * (pool.num_qubits - 1)
    for operator_id in selected_operator_ids:
        operator = by_id[operator_id]
        if len(operator.sites) == 2:
            edge_counts[operator.sites[0]] += operator.native_two_qubit_gates
    result: list[tuple[int, PoolOperator, bool]] = []
    for pool_index, operator in enumerate(pool.operators):
        if operator.operator_id in selected:
            continue
        feasible = True
        if len(operator.sites) == 2 and spec.native_two_qubit_cap_per_edge is not None:
            feasible = (
                edge_counts[operator.sites[0]] + operator.native_two_qubit_gates <= spec.native_two_qubit_cap_per_edge
            )
        result.append((pool_index, operator, feasible))
    return tuple(result)


def _selection_expected_work(
    pool: OperatorPoolSpec,
    spec: OperatorGrowthSpec,
    request: OperatorGrowthSelectionRequest,
) -> float:
    """Return exact fixed-CRN work for all candidates plus appended-zero baseline."""
    candidates = _operator_growth_candidate_metadata(pool, spec, request.selected_operator_ids)
    feasible_count = sum(feasible for _index, _operator, feasible in candidates)
    if not candidates or not feasible_count:
        msg = "The exact two-prefix program requires at least one feasible remaining candidate."
        raise ValueError(msg)
    return float((2 * feasible_count + 1) * OPERATOR_GROWTH_TRAINING_TRAJECTORY_COUNT)


def _segmented_objective_evidence(
    executor: OperatorGrowthSegmentedObjectiveExecutor,
    request: OperatorGrowthSegmentedObjectiveRequest,
) -> OperatorGrowthSegmentedObjectiveEvidence:
    """Invoke one typed objective callback and verify its exact request link.

    Returns:
        The checksum-linked request and callback result.
    """
    result = executor(request)
    if not isinstance(result, OperatorGrowthSegmentedObjectiveResult):
        msg = "objective_executor must return OperatorGrowthSegmentedObjectiveResult."
        raise TypeError(msg)
    return OperatorGrowthSegmentedObjectiveEvidence(request, result)


def _validate_selection_result(
    program: ScheduledExecutionProgram,
    pool: OperatorPoolSpec,
    spec: OperatorGrowthSpec,
    request: OperatorGrowthSelectionRequest,
    result: OperatorGrowthSelectionResult,
    quoted_work: float,
) -> None:
    """Replay full pool order, objective pairs, feasibility, and deterministic selection."""
    if not isinstance(result, OperatorGrowthSelectionResult):
        msg = "selection_executor must return OperatorGrowthSelectionResult."
        raise TypeError(msg)
    if result.request_checksum != request.content_checksum or result.normalized_work != quoted_work:
        msg = "Structural selection result differs from its request or deterministic work quote."
        raise ValueError(msg)
    metadata = _operator_growth_candidate_metadata(pool, spec, request.selected_operator_ids)
    if len(result.candidate_gradients) != len(metadata):
        msg = "Structural selection omitted remaining pool candidates."
        raise ValueError(msg)
    evidence_index = 0
    feasible_candidates: list[CandidateGradient] = []
    parameter_index = request.prefix_index
    for candidate, (pool_index, operator, feasible) in zip(
        result.candidate_gradients,
        metadata,
        strict=True,
    ):
        operator_id = operator.operator_id
        native_increment = operator.native_two_qubit_gates
        if (
            candidate.operator_id != operator_id
            or candidate.pool_index != pool_index
            or candidate.native_two_qubit_increment != native_increment
            or candidate.native_cap_feasible != feasible
        ):
            msg = "Selection candidate ordering, identity, native cost, or feasibility changed."
            raise ValueError(msg)
        if not feasible:
            continue
        selected_ids = (*request.selected_operator_ids, operator_id)
        plus_parameters = (*request.parameters, math.pi / 2.0)
        minus_parameters = (*request.parameters, -math.pi / 2.0)
        expected_plus = OperatorGrowthSegmentedObjectiveRequest(
            program_checksum=program.content_checksum,
            structural_state_checksum=request.content_checksum,
            selected_operator_ids=selected_ids,
            prefix_index=request.prefix_index,
            global_update=request.global_update_start,
            local_update=0,
            evaluation_stage="structural_selection",
            evaluation_kind="gradient_plus",
            parameter_index=parameter_index,
            parameters=plus_parameters,
            policy=request.policy,
        )
        expected_minus = replace(
            expected_plus,
            evaluation_kind="gradient_minus",
            parameters=minus_parameters,
        )
        try:
            plus = result.objective_evidence[evidence_index]
            minus = result.objective_evidence[evidence_index + 1]
        except IndexError as error:
            msg = "Selection objective evidence ended before every feasible candidate pair."
            raise ValueError(msg) from error
        if plus.request != expected_plus or minus.request != expected_minus:
            msg = "Selection objective evidence changed its fixed-CRN candidate pair order."
            raise ValueError(msg)
        expected_gradient = 0.5 * (plus.result.objective - minus.result.objective)
        if candidate.gradient is None or not math.isclose(
            candidate.gradient,
            expected_gradient,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            msg = "Candidate gradient differs from its retained parameter-shift objectives."
            raise ValueError(msg)
        feasible_candidates.append(candidate)
        evidence_index += 2
    chosen = max(feasible_candidates, key=lambda item: cast("float", item.absolute_gradient))
    expected_baseline = OperatorGrowthSegmentedObjectiveRequest(
        program_checksum=program.content_checksum,
        structural_state_checksum=request.content_checksum,
        selected_operator_ids=(*request.selected_operator_ids, chosen.operator_id),
        prefix_index=request.prefix_index,
        global_update=request.global_update_start,
        local_update=0,
        evaluation_stage="structural_selection",
        evaluation_kind="post_update",
        parameter_index=parameter_index + 1,
        parameters=(*request.parameters, 0.0),
        policy=request.policy,
    )
    if len(result.objective_evidence) != evidence_index + 1:
        msg = "Selection objective evidence contains hidden or missing objective calls."
        raise ValueError(msg)
    baseline = result.objective_evidence[evidence_index]
    if (
        baseline.request != expected_baseline
        or result.selected_operator_id != chosen.operator_id
        or result.selected_gradient != chosen.gradient
        or result.objective_before_reoptimization != baseline.result.objective
        or cast("float", chosen.absolute_gradient) <= spec.gradient_tolerance
    ):
        msg = "Structural selection differs from largest-gradient, tie-break, or appended-zero rules."
        raise ValueError(msg)


def _segmented_checkpoint_state_checksum(
    transition_checksum: str,
    global_update: int,
    parameters: tuple[float, ...],
    first_moment: tuple[float, ...],
    second_moment: tuple[float, ...],
    best_objective: float,
) -> str:
    """Seal the optimizer state that yielded a recoverable best-prefix artifact.

    Returns:
        The canonical optimizer-state checksum.
    """
    return canonical_checksum({
        "optimizer_kind": "operator_growth_adam",
        "transition_checksum": transition_checksum,
        "global_update": global_update,
        "parameters": list(parameters),
        "first_moment": list(first_moment),
        "second_moment": list(second_moment),
        "best_objective": best_objective,
    })


def _validate_operator_growth_segmented_history(
    program: ScheduledExecutionProgram,
    snapshot: OperatorGrowthSegmentedSnapshot,
) -> None:
    """Mechanically replay exact structural selection, Adam, validation, and work."""
    if not isinstance(snapshot, OperatorGrowthSegmentedSnapshot):
        msg = "snapshot must be an OperatorGrowthSegmentedSnapshot."
        raise TypeError(msg)
    execution_spec = _validate_operator_growth_segmented_program(program)
    pool = execution_spec.pool
    spec = execution_spec.growth_spec
    if snapshot.program_checksum != program.content_checksum or snapshot.pool != pool or snapshot.growth_spec != spec:
        msg = "Segmented snapshot differs from its exact program, pool, or growth spec."
        raise ValueError(msg)
    previous_transition_checksum: str | None = None
    for prefix, transition in enumerate(snapshot.transitions):
        source_parameters = (
            () if prefix == 0 else snapshot.prefix_validations[prefix - 1].request.parameter_artifact.parameters
        )
        expected_request = OperatorGrowthSelectionRequest(
            program_checksum=program.content_checksum,
            growth_spec_checksum=spec.content_checksum,
            pool_checksum=pool.content_checksum,
            prefix_index=prefix,
            global_update_start=prefix * OPERATOR_GROWTH_PREFIX_UPDATE_COUNT,
            selected_operator_ids=() if prefix == 0 else snapshot.transitions[prefix - 1].selected_operator_ids,
            parameters=source_parameters,
            policy=ScheduledTrainingPolicy.from_compiled(
                program.policy(0, prefix * OPERATOR_GROWTH_PREFIX_UPDATE_COUNT)
            ),
            previous_transition_checksum=previous_transition_checksum,
        )
        expected_work = _selection_expected_work(pool, spec, expected_request)
        if transition.request != expected_request:
            msg = "Persisted structural-selection request differs from its exact prefix source."
            raise ValueError(msg)
        _validate_selection_result(
            program,
            pool,
            spec,
            transition.request,
            transition.result,
            expected_work,
        )
        if transition.selected_operator_ids != (
            *expected_request.selected_operator_ids,
            transition.result.selected_operator_id,
        ) or transition.initial_parameters != (*source_parameters, 0.0):
            msg = "Persisted transition differs from append-at-zero semantics."
            raise ValueError(msg)
        previous_transition_checksum = transition.content_checksum

    parameters: tuple[float, ...] = ()
    first_moment: tuple[float, ...] = ()
    second_moment: tuple[float, ...] = ()
    best_parameters: tuple[float, ...] = ()
    best_objective: float | None = None
    for update, receipt in enumerate(snapshot.receipts):
        prefix = update // OPERATOR_GROWTH_PREFIX_UPDATE_COUNT
        local = update % OPERATOR_GROWTH_PREFIX_UPDATE_COUNT
        transition = snapshot.transitions[prefix]
        if local == 0:
            parameters = transition.initial_parameters
            first_moment = _zeros(len(parameters))
            second_moment = _zeros(len(parameters))
            best_parameters = parameters
            best_objective = transition.result.objective_before_reoptimization
        assert best_objective is not None
        expected_policy = ScheduledTrainingPolicy.from_compiled(program.policy(0, update))
        if (
            receipt.policy != expected_policy
            or receipt.parameters_before != parameters
            or receipt.first_moment_before != first_moment
            or receipt.second_moment_before != second_moment
        ):
            msg = "Persisted Adam receipt differs from its exact program or predecessor state."
            raise ValueError(msg)
        expected_first = tuple(
            spec.adam_beta1 * moment + (1.0 - spec.adam_beta1) * gradient
            for moment, gradient in zip(first_moment, receipt.gradient, strict=True)
        )
        expected_second = tuple(
            spec.adam_beta2 * moment + (1.0 - spec.adam_beta2) * gradient * gradient
            for moment, gradient in zip(second_moment, receipt.gradient, strict=True)
        )
        adam_iteration = local + 1
        expected_parameters = tuple(
            parameter
            - spec.learning_rate
            * (moment / (1.0 - spec.adam_beta1**adam_iteration))
            / (math.sqrt(square / (1.0 - spec.adam_beta2**adam_iteration)) + spec.adam_epsilon)
            for parameter, moment, square in zip(
                parameters,
                expected_first,
                expected_second,
                strict=True,
            )
        )
        monitor_objective = receipt.objective_evidence[-1].result.objective
        improved = monitor_objective < best_objective
        expected_best_parameters = expected_parameters if improved else best_parameters
        expected_best_objective = monitor_objective if improved else best_objective
        if (
            receipt.first_moment_after != expected_first
            or receipt.second_moment_after != expected_second
            or receipt.parameters_after != expected_parameters
            or receipt.best_parameters != expected_best_parameters
            or receipt.best_objective != expected_best_objective
        ):
            msg = "Persisted Adam moments, bias correction, parameters, or best state changed."
            raise ValueError(msg)
        if local == OPERATOR_GROWTH_PREFIX_UPDATE_COUNT - 1:
            validation = receipt.prefix_validation
            assert validation is not None
            policy = program.policy(0, update)
            assert policy.checkpoint_membership is not None
            expected_artifact = ParameterCheckpointArtifact(
                optimizer_kind="operator_growth_adam",
                start_index=0,
                update=update,
                parameters=expected_best_parameters,
                optimizer_payload_checksum=_segmented_checkpoint_state_checksum(
                    transition.content_checksum,
                    update,
                    expected_parameters,
                    expected_first,
                    expected_second,
                    expected_best_objective,
                ),
            )
            expected_validation_request = ScheduledValidationRequest(
                program_checksum=program.content_checksum,
                start_index=0,
                update=update,
                parameter_artifact=expected_artifact,
                membership=policy.checkpoint_membership,
            )
            if validation.request != expected_validation_request:
                msg = "Persisted prefix validation differs from its best training parameters or membership."
                raise ValueError(msg)
        parameters = expected_parameters
        first_moment = expected_first
        second_moment = expected_second
        best_parameters = expected_best_parameters
        best_objective = expected_best_objective
    if snapshot.receipts and (
        snapshot.parameters != parameters
        or snapshot.first_moment != first_moment
        or snapshot.second_moment != second_moment
        or snapshot.best_training_parameters != best_parameters
        or snapshot.best_training_objective != best_objective
    ):
        msg = "Snapshot endpoint differs from mechanically replayed Adam history."
        raise ValueError(msg)
    if program.normalized_compute_cap is not None and snapshot.total_normalized_work > program.normalized_compute_cap:
        msg = "Persisted segmented work already exceeds the job-wide compute cap."
        raise ValueError(msg)


def _build_operator_growth_segmented_snapshot(
    program: ScheduledExecutionProgram,
    pool: OperatorPoolSpec,
    spec: OperatorGrowthSpec,
    transitions: tuple[OperatorGrowthPrefixTransition, ...],
    receipts: tuple[OperatorGrowthSegmentedUpdateReceipt, ...],
    validations: tuple[OperatorGrowthPrefixValidation, ...],
) -> OperatorGrowthSegmentedSnapshot:
    """Construct one immutable endpoint from complete histories.

    Returns:
        The strict authoritative endpoint snapshot.
    """
    if receipts:
        last = receipts[-1]
        active_ids = last.selected_operator_ids
        parameters = last.parameters_after
        first_moment = last.first_moment_after
        second_moment = last.second_moment_after
        best_parameters = last.best_parameters
        best_objective: float | None = last.best_objective
    else:
        active_ids = ()
        parameters = ()
        first_moment = ()
        second_moment = ()
        best_parameters = ()
        best_objective = None
    if validations:
        selected_prefix = max(range(len(validations)), key=lambda index: validations[index].result.score)
        selected_ids = transitions[selected_prefix].selected_operator_ids
        selected_parameters = validations[selected_prefix].request.parameter_artifact.parameters
        selected_score: float | None = validations[selected_prefix].result.score
    else:
        selected_prefix = None
        selected_ids = ()
        selected_parameters = ()
        selected_score = None
    return OperatorGrowthSegmentedSnapshot(
        program_checksum=program.content_checksum,
        pool=pool,
        growth_spec=spec,
        active_operator_ids=active_ids,
        parameters=parameters,
        first_moment=first_moment,
        second_moment=second_moment,
        best_training_parameters=best_parameters,
        best_training_objective=best_objective,
        transitions=transitions,
        receipts=receipts,
        prefix_validations=validations,
        total_normalized_work=math.fsum(
            [transition.result.normalized_work for transition in transitions]
            + [receipt.normalized_work for receipt in receipts]
        ),
        selected_prefix_index=selected_prefix,
        selected_operator_ids=selected_ids,
        selected_parameters=selected_parameters,
        selected_validation_score=selected_score,
        terminal_reason=("update_budget" if len(receipts) == OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT else None),
    )


def execute_operator_growth_segmented_program(
    program: ScheduledExecutionProgram,
    snapshot: OperatorGrowthSegmentedSnapshot,
    structural_selection_executor: OperatorGrowthStructuralSelectionExecutor,
    objective_executor: OperatorGrowthSegmentedObjectiveExecutor,
    validation_executor: ScheduledValidationExecutor,
    *,
    stop_after_updates: int | None = None,
) -> OperatorGrowthSegmentedSnapshot:
    """Execute or resume the exact two-prefix q6 operator-growth program.

    Structural selection is work-quoted and cap-checked before its callback.
    Every complete Adam update, including a mandatory boundary validation, is
    likewise checked against the job-wide cap before any objective callback.
    The callback interfaces expose training or checkpoint-validation records
    only and contain no final-test input.

    Returns:
        An interrupted or terminal checksum-sealed segmented snapshot.
    """
    execution_spec = _validate_operator_growth_segmented_program(program)
    if not isinstance(snapshot, OperatorGrowthSegmentedSnapshot):
        msg = "snapshot must be an OperatorGrowthSegmentedSnapshot."
        raise TypeError(msg)
    if not callable(structural_selection_executor) or not callable(
        getattr(structural_selection_executor, "quote_normalized_work", None)
    ):
        msg = "structural_selection_executor must be callable and expose quote_normalized_work."
        raise TypeError(msg)
    if not callable(objective_executor):
        msg = "objective_executor must be callable."
        raise TypeError(msg)
    if not callable(validation_executor):
        msg = "validation_executor must be callable."
        raise TypeError(msg)
    snapshot.validate_against_program(program)
    if snapshot.complete:
        return snapshot
    callback_limit = (
        None if stop_after_updates is None else require_int(stop_after_updates, "stop_after_updates", minimum=1)
    )
    pool = execution_spec.pool
    spec = execution_spec.growth_spec
    transitions = list(snapshot.transitions)
    receipts = list(snapshot.receipts)
    validations = list(snapshot.prefix_validations)
    current_work = snapshot.total_normalized_work
    completed_updates = 0
    while len(receipts) < OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT and (
        callback_limit is None or completed_updates < callback_limit
    ):
        global_update = len(receipts)
        prefix = global_update // OPERATOR_GROWTH_PREFIX_UPDATE_COUNT
        local = global_update % OPERATOR_GROWTH_PREFIX_UPDATE_COUNT
        if local == 0:
            source_parameters = () if prefix == 0 else validations[prefix - 1].request.parameter_artifact.parameters
            selection_request = OperatorGrowthSelectionRequest(
                program_checksum=program.content_checksum,
                growth_spec_checksum=spec.content_checksum,
                pool_checksum=pool.content_checksum,
                prefix_index=prefix,
                global_update_start=global_update,
                selected_operator_ids=() if prefix == 0 else transitions[prefix - 1].selected_operator_ids,
                parameters=source_parameters,
                policy=ScheduledTrainingPolicy.from_compiled(program.policy(0, global_update)),
                previous_transition_checksum=(None if not transitions else transitions[-1].content_checksum),
            )
            expected_selection_work = _selection_expected_work(pool, spec, selection_request)
            quoted_selection_work = require_float(
                structural_selection_executor.quote_normalized_work(selection_request),
                "quoted structural-selection normalized work",
                minimum=0.0,
            )
            if quoted_selection_work != expected_selection_work:
                msg = "Structural-selection work quote differs from the complete feasible candidate universe."
                raise ValueError(msg)
            cap = program.normalized_compute_cap
            first_update_work = float((2 * (prefix + 1) + 1) * OPERATOR_GROWTH_TRAINING_TRAJECTORY_COUNT)
            atomic_prefix_start_work = math.fsum((quoted_selection_work, first_update_work))
            if cap is not None and math.fsum((current_work, atomic_prefix_start_work)) > cap:
                raise NormalizedComputeCapError(
                    cap=cap,
                    completed_work=current_work,
                    prospective_update_work=atomic_prefix_start_work,
                )
            selection_result = structural_selection_executor(selection_request, objective_executor)
            _validate_selection_result(
                program,
                pool,
                spec,
                selection_request,
                selection_result,
                quoted_selection_work,
            )
            transition = OperatorGrowthPrefixTransition(
                request=selection_request,
                result=selection_result,
                selected_operator_ids=(
                    *selection_request.selected_operator_ids,
                    selection_result.selected_operator_id,
                ),
                initial_parameters=(*selection_request.parameters, 0.0),
                previous_transition_checksum=selection_request.previous_transition_checksum,
            )
            transitions.append(transition)
            current_work = math.fsum((current_work, quoted_selection_work))
            parameters = transition.initial_parameters
            first_moment = _zeros(len(parameters))
            second_moment = _zeros(len(parameters))
            best_parameters = parameters
            best_objective = selection_result.objective_before_reoptimization
        else:
            transition = transitions[prefix]
            previous = receipts[-1]
            parameters = previous.parameters_after
            first_moment = previous.first_moment_after
            second_moment = previous.second_moment_after
            best_parameters = previous.best_parameters
            best_objective = previous.best_objective
        policy = ScheduledTrainingPolicy.from_compiled(program.policy(0, global_update))
        training_work = float((2 * len(parameters) + 1) * OPERATOR_GROWTH_TRAINING_TRAJECTORY_COUNT)
        validation_work = (
            float(OPERATOR_GROWTH_VALIDATION_TRAJECTORY_COUNT)
            if local == OPERATOR_GROWTH_PREFIX_UPDATE_COUNT - 1
            else 0.0
        )
        complete_update_work = math.fsum((training_work, validation_work))
        cap = program.normalized_compute_cap
        if cap is not None and math.fsum((current_work, complete_update_work)) > cap:
            raise NormalizedComputeCapError(
                cap=cap,
                completed_work=current_work,
                prospective_update_work=complete_update_work,
            )
        evidence: list[OperatorGrowthSegmentedObjectiveEvidence] = []
        gradient: list[float] = []
        for parameter_index in range(len(parameters)):
            plus_parameters = list(parameters)
            minus_parameters = list(parameters)
            plus_parameters[parameter_index] += math.pi / 2.0
            minus_parameters[parameter_index] -= math.pi / 2.0
            plus_request = OperatorGrowthSegmentedObjectiveRequest(
                program_checksum=program.content_checksum,
                structural_state_checksum=transition.content_checksum,
                selected_operator_ids=transition.selected_operator_ids,
                prefix_index=prefix,
                global_update=global_update,
                local_update=local,
                evaluation_stage="prefix_reoptimization",
                evaluation_kind="gradient_plus",
                parameter_index=parameter_index,
                parameters=tuple(plus_parameters),
                policy=policy,
            )
            minus_request = replace(
                plus_request,
                evaluation_kind="gradient_minus",
                parameters=tuple(minus_parameters),
            )
            plus = _segmented_objective_evidence(objective_executor, plus_request)
            minus = _segmented_objective_evidence(objective_executor, minus_request)
            evidence.extend((plus, minus))
            gradient.append(0.5 * (plus.result.objective - minus.result.objective))
        first_after = tuple(
            spec.adam_beta1 * moment + (1.0 - spec.adam_beta1) * value
            for moment, value in zip(first_moment, gradient, strict=True)
        )
        second_after = tuple(
            spec.adam_beta2 * moment + (1.0 - spec.adam_beta2) * value * value
            for moment, value in zip(second_moment, gradient, strict=True)
        )
        adam_iteration = local + 1
        parameters_after = tuple(
            parameter
            - spec.learning_rate
            * (moment / (1.0 - spec.adam_beta1**adam_iteration))
            / (math.sqrt(square / (1.0 - spec.adam_beta2**adam_iteration)) + spec.adam_epsilon)
            for parameter, moment, square in zip(
                parameters,
                first_after,
                second_after,
                strict=True,
            )
        )
        monitor_request = OperatorGrowthSegmentedObjectiveRequest(
            program_checksum=program.content_checksum,
            structural_state_checksum=transition.content_checksum,
            selected_operator_ids=transition.selected_operator_ids,
            prefix_index=prefix,
            global_update=global_update,
            local_update=local,
            evaluation_stage="prefix_reoptimization",
            evaluation_kind="post_update",
            parameter_index=len(parameters_after),
            parameters=parameters_after,
            policy=policy,
        )
        monitor = _segmented_objective_evidence(objective_executor, monitor_request)
        evidence.append(monitor)
        improved = monitor.result.objective < best_objective
        next_best_parameters = parameters_after if improved else best_parameters
        next_best_objective = monitor.result.objective if improved else best_objective
        prefix_validation = None
        if local == OPERATOR_GROWTH_PREFIX_UPDATE_COUNT - 1:
            compiled_policy = program.policy(0, global_update)
            assert compiled_policy.checkpoint_membership is not None
            artifact = ParameterCheckpointArtifact(
                optimizer_kind="operator_growth_adam",
                start_index=0,
                update=global_update,
                parameters=next_best_parameters,
                optimizer_payload_checksum=_segmented_checkpoint_state_checksum(
                    transition.content_checksum,
                    global_update,
                    parameters_after,
                    first_after,
                    second_after,
                    next_best_objective,
                ),
            )
            validation_request = ScheduledValidationRequest(
                program_checksum=program.content_checksum,
                start_index=0,
                update=global_update,
                parameter_artifact=artifact,
                membership=compiled_policy.checkpoint_membership,
            )
            validation_result = validation_executor(validation_request)
            if not isinstance(validation_result, ScheduledValidationResult):
                msg = "validation_executor must return ScheduledValidationResult."
                raise TypeError(msg)
            prefix_validation = OperatorGrowthPrefixValidation(
                prefix_index=prefix,
                transition_checksum=transition.content_checksum,
                request=validation_request,
                result=validation_result,
                normalized_work=validation_work,
                previous_validation_checksum=(None if not validations else validations[-1].content_checksum),
            )
        receipt = OperatorGrowthSegmentedUpdateReceipt(
            program_checksum=program.content_checksum,
            prefix_index=prefix,
            global_update=global_update,
            local_update=local,
            transition_checksum=transition.content_checksum,
            selected_operator_ids=transition.selected_operator_ids,
            policy=policy,
            parameters_before=parameters,
            first_moment_before=first_moment,
            second_moment_before=second_moment,
            gradient=tuple(gradient),
            parameters_after=parameters_after,
            first_moment_after=first_after,
            second_moment_after=second_after,
            best_parameters=next_best_parameters,
            best_objective=next_best_objective,
            objective_evidence=tuple(evidence),
            normalized_work=complete_update_work,
            prefix_validation=prefix_validation,
            previous_receipt_checksum=None if not receipts else receipts[-1].content_checksum,
        )
        receipts.append(receipt)
        if prefix_validation is not None:
            validations.append(prefix_validation)
        current_work = math.fsum((current_work, complete_update_work))
        completed_updates += 1
    result = _build_operator_growth_segmented_snapshot(
        program,
        pool,
        spec,
        tuple(transitions),
        tuple(receipts),
        tuple(validations),
    )
    result.validate_against_program(program)
    return result


__all__ = [
    "ADAM_OPTIMIZER_PAYLOAD_SCHEMA_VERSION",
    "COMPONENT_TRAJECTORY_MEMBERSHIP_SCHEMA_VERSION",
    "KROTOV_OPTIMIZER_PAYLOAD_SCHEMA_VERSION",
    "MULTISTART_START_EVIDENCE_SCHEMA_VERSION",
    "MULTISTART_WORK_EVIDENCE_SCHEMA_VERSION",
    "OPERATOR_GROWTH_GLOBAL_UPDATE_COUNT",
    "OPERATOR_GROWTH_OPTIMIZER_PAYLOAD_SCHEMA_VERSION",
    "OPERATOR_GROWTH_PREFIX_TRANSITION_SCHEMA_VERSION",
    "OPERATOR_GROWTH_PREFIX_UPDATE_COUNT",
    "OPERATOR_GROWTH_PREFIX_VALIDATION_SCHEMA_VERSION",
    "OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_EVIDENCE_SCHEMA_VERSION",
    "OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_REQUEST_SCHEMA_VERSION",
    "OPERATOR_GROWTH_SEGMENTED_OBJECTIVE_RESULT_SCHEMA_VERSION",
    "OPERATOR_GROWTH_SEGMENTED_SNAPSHOT_SCHEMA_VERSION",
    "OPERATOR_GROWTH_SEGMENTED_UPDATE_RECEIPT_SCHEMA_VERSION",
    "OPERATOR_GROWTH_SELECTION_REQUEST_SCHEMA_VERSION",
    "OPERATOR_GROWTH_SELECTION_RESULT_SCHEMA_VERSION",
    "OPERATOR_GROWTH_TRAINING_TRAJECTORY_COUNT",
    "OPERATOR_GROWTH_VALIDATION_TRAJECTORY_COUNT",
    "OPTIMIZER_INITIALIZATION_SCHEMA_VERSION",
    "PARAMETER_CHECKPOINT_ARTIFACT_SCHEMA_VERSION",
    "SCHEDULED_EXECUTION_PROGRAM_SCHEMA_VERSION",
    "SCHEDULED_EXECUTION_SNAPSHOT_SCHEMA_VERSION",
    "SCHEDULED_JOB_SEED_SET_SCHEMA_VERSION",
    "SCHEDULED_OPTIMIZER_STATE_SCHEMA_VERSION",
    "SCHEDULED_TRAINING_GRADIENT_REQUEST_SCHEMA_VERSION",
    "SCHEDULED_TRAINING_GRADIENT_RESULT_SCHEMA_VERSION",
    "SCHEDULED_TRAINING_OBJECTIVE_REQUEST_SCHEMA_VERSION",
    "SCHEDULED_TRAINING_OBJECTIVE_RESULT_SCHEMA_VERSION",
    "SCHEDULED_TRAINING_POLICY_SCHEMA_VERSION",
    "SCHEDULED_UPDATE_POLICY_SCHEMA_VERSION",
    "SCHEDULED_UPDATE_RECEIPT_SCHEMA_VERSION",
    "SCHEDULED_UPDATE_REQUEST_SCHEMA_VERSION",
    "SCHEDULED_UPDATE_RESULT_SCHEMA_VERSION",
    "SCHEDULED_VALIDATION_REQUEST_SCHEMA_VERSION",
    "SCHEDULED_VALIDATION_RESULT_SCHEMA_VERSION",
    "SPSA_OPTIMIZER_PAYLOAD_SCHEMA_VERSION",
    "AdamOptimizerPayload",
    "ComponentTrajectoryMembership",
    "ExecutionScope",
    "InitializerKind",
    "KrotovOptimizerPayload",
    "KrotovScheduledUpdateAdapter",
    "LearningRateSchedule",
    "MethodOptimizerPayload",
    "MultistartStartEvidence",
    "MultistartWorkEvidence",
    "NormalizedComputeCapError",
    "OperatorGrowthAdamScheduledUpdateAdapter",
    "OperatorGrowthOptimizerPayload",
    "OperatorGrowthPrefixTransition",
    "OperatorGrowthPrefixValidation",
    "OperatorGrowthSegmentedObjectiveEvidence",
    "OperatorGrowthSegmentedObjectiveExecutor",
    "OperatorGrowthSegmentedObjectiveRequest",
    "OperatorGrowthSegmentedObjectiveResult",
    "OperatorGrowthSegmentedSnapshot",
    "OperatorGrowthSegmentedUpdateReceipt",
    "OperatorGrowthSelectionRequest",
    "OperatorGrowthSelectionResult",
    "OperatorGrowthStructuralSelectionExecutor",
    "OptimizerInitialization",
    "OptimizerKind",
    "ParameterCheckpointArtifact",
    "ParameterShiftAdamScheduledUpdateAdapter",
    "RepositoryScheduledUpdateAdapter",
    "SPSAOptimizerPayload",
    "SPSAScheduledUpdateAdapter",
    "ScheduledExecutionProgram",
    "ScheduledExecutionSnapshot",
    "ScheduledJobSeedSet",
    "ScheduledOptimizerState",
    "ScheduledTrainingGradientExecutor",
    "ScheduledTrainingGradientRequest",
    "ScheduledTrainingGradientResult",
    "ScheduledTrainingObjectiveExecutor",
    "ScheduledTrainingObjectiveRequest",
    "ScheduledTrainingObjectiveResult",
    "ScheduledTrainingPolicy",
    "ScheduledUpdateExecutor",
    "ScheduledUpdatePolicy",
    "ScheduledUpdateReceipt",
    "ScheduledUpdateRequest",
    "ScheduledUpdateResult",
    "ScheduledValidationExecutor",
    "ScheduledValidationRequest",
    "ScheduledValidationResult",
    "TerminalReason",
    "TrainingPhase",
    "compile_development_schedule",
    "compile_frozen_schedule_trace",
    "execute_operator_growth_segmented_program",
    "execute_scheduled_program",
    "initialize_scheduled_execution",
]
