# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fair resource accounting and paired-comparison records for WP20.

This module is deliberately additive.  It projects its detailed work ledger
onto the smaller WP16 ledger when needed, but it does not change any existing
Phase II schema.  Circuit resources are always measured after the frozen
Quantinuum compilation, and resource-stratum selection never observes a
fidelity or another scientific outcome.
"""

# The strict records below have deliberately small private validators. Their
# exception contracts are inherited from ``validation.py`` and repeating them
# in every helper would obscure the scientific schema documentation.
# ruff: noqa: DOC201, DOC501

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from benchmarks.state_preparation.circuits import LogicalToNativeMapping, compile_quantinuum_native
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .noisy_krotov import (
    PRIMARY_COMPILER_POLICY_ID,
    PRIMARY_CONNECTIVITY,
    PRIMARY_COUNTING_POLICY_ID,
    PRIMARY_ROUTING_POLICY_ID,
    NoisyKrotovStageExecution,
)
from .pipeline import PipelineEvaluationConfig, TrainingPipelineConfig, TrainingStageConfig
from .validation import (
    require_bool,
    require_checksum,
    require_float,
    require_int,
    require_mapping,
    require_slug,
    require_string,
)

if TYPE_CHECKING:
    from .operator_growth import OperatorGrowthResult

WP20_WORK_LEDGER_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_work_ledger.v1"
WP20_NORMALIZED_COMPUTE_POLICY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_normalized_compute_policy.v1"
WP20_NORMALIZED_COMPUTE_POLICY_ID = "circuit_and_trajectory_work_unit_v1"
WP20_LOGICAL_EVENT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_logical_event.v1"
WP20_NATIVE_EVENT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_native_event.v1"
WP20_CIRCUIT_RESOURCES_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_circuit_resources.v1"
WP20_RESOURCE_BUDGET_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_resource_budget.v1"
WP20_RESOURCE_STRATUM_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_resource_stratum.v1"
WP20_RESOURCE_SELECTION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_resource_selection.v1"
WP20_PARETO_POINT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_pareto_point.v1"
WP20_PAIRED_BLOCK_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_paired_block.v1"
WP20_TRAINING_RANDOMNESS_STAGE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_training_randomness_stage.v1"
WP20_OPERATOR_GROWTH_RANDOMNESS_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_operator_growth_randomness.v1"
WP20_TRAINING_RANDOMNESS_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_training_randomness.v2"
WP20_TEST_COUPLING_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp20_test_coupling.v1"

_WORK_LEDGER_COUNTER_FIELDS = (
    "forward_circuit_evaluations",
    "backward_circuit_evaluations",
    "trajectory_gate_applications",
    "training_trajectories",
    "checkpoint_validation_trajectories",
    "test_trajectories",
    "objective_calls",
    "gradient_calls",
    "cross_trajectory_pairings",
)

_KNOWN_NOISELESS_TRAINING_METHOD_IDS = frozenset({
    "layerwise_bmpd_noiseless",
    "phase1_noiseless_checkpoint_control",
    "unpruned_deep_bmpd",
})


def _require_identifier(value: object, name: str) -> int | str:
    """Validate a stable integer or string gate identifier."""
    if type(value) is int:
        return require_int(value, name)
    if type(value) is str:
        return require_string(value, name)
    msg = f"{name} must be a nonnegative int or nonempty string."
    raise TypeError(msg)


def _require_optional_int(value: object, name: str) -> int | None:
    """Validate an optional nonnegative built-in integer."""
    return None if value is None else require_int(value, name)


def _require_int_tuple(value: object, name: str, *, unique: bool = False) -> tuple[int, ...]:
    """Validate and detach a sequence of nonnegative integers."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of ints."
        raise TypeError(msg)
    result = tuple(require_int(item, f"{name}[{index}]") for index, item in enumerate(value))
    if unique and len(result) != len(set(result)):
        msg = f"{name} must not contain duplicates."
        raise ValueError(msg)
    return result


def _require_float_tuple(value: object, name: str) -> tuple[float, ...]:
    """Validate and detach a sequence of finite built-in floats."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of floats."
        raise TypeError(msg)
    return tuple(require_float(item, f"{name}[{index}]") for index, item in enumerate(value))


def _require_checksum_tuple(value: object, name: str) -> tuple[str, ...]:
    """Validate an ordered duplicate-free sequence of checksums."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of checksums."
        raise TypeError(msg)
    result = tuple(require_checksum(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        msg = f"{name} must not contain duplicates."
        raise ValueError(msg)
    return result


def _dependency_depth(qubit_count: int, sites: Sequence[tuple[int, ...]]) -> int:
    """Return dependency depth using the shared benchmark convention."""
    site_depths = [0] * qubit_count
    for event_sites in sites:
        gate_depth = 1 + max(site_depths[site] for site in event_sites)
        for site in event_sites:
            site_depths[site] = gate_depth
    return max(site_depths, default=0)


@dataclass(frozen=True, slots=True)
class WP20WorkLedger:
    """Detailed additive work and runtime ledger for one WP20 execution.

    ``plus`` adds work counters and elapsed time. Peak memory is a high-water
    mark and is therefore merged with ``max`` rather than summed.
    """

    forward_circuit_evaluations: int = 0
    backward_circuit_evaluations: int = 0
    trajectory_gate_applications: int = 0
    training_trajectories: int = 0
    checkpoint_validation_trajectories: int = 0
    test_trajectories: int = 0
    objective_calls: int = 0
    gradient_calls: int = 0
    cross_trajectory_pairings: int = 0
    wall_time_seconds: float = 0.0
    peak_memory_bytes: int = 0
    schema_version: str = field(default=WP20_WORK_LEDGER_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact nonnegative scalar types."""
        for name in _WORK_LEDGER_COUNTER_FIELDS:
            object.__setattr__(self, name, require_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "wall_time_seconds",
            require_float(self.wall_time_seconds, "wall_time_seconds", minimum=0.0),
        )
        object.__setattr__(
            self,
            "peak_memory_bytes",
            require_int(self.peak_memory_bytes, "peak_memory_bytes"),
        )

    @property
    def total_sampled_trajectories(self) -> int:
        """Return the mechanically derived total trajectory count."""
        return self.training_trajectories + self.checkpoint_validation_trajectories + self.test_trajectories

    def plus(self, **increments: float) -> WP20WorkLedger:
        """Return a ledger with validated increments applied.

        Args:
            **increments: Named counter, wall-time, or peak-memory updates.

        Returns:
            A new immutable ledger.

        Raises:
            ValueError: If an unknown field is supplied.
        """
        allowed = {*_WORK_LEDGER_COUNTER_FIELDS, "wall_time_seconds", "peak_memory_bytes"}
        unknown = set(increments) - allowed
        if unknown:
            msg = f"Unknown WP20 work counters: {sorted(unknown)!r}."
            raise ValueError(msg)
        fields: dict[str, int | float] = {
            name: cast("int", getattr(self, name)) for name in _WORK_LEDGER_COUNTER_FIELDS
        }
        fields["wall_time_seconds"] = self.wall_time_seconds
        fields["peak_memory_bytes"] = self.peak_memory_bytes
        for name, increment in increments.items():
            if name == "wall_time_seconds":
                fields[name] += require_float(
                    increment,
                    "increments.wall_time_seconds",
                    minimum=0.0,
                )
            elif name == "peak_memory_bytes":
                fields[name] = max(
                    cast("int", fields[name]),
                    require_int(increment, "increments.peak_memory_bytes"),
                )
            else:
                fields[name] = cast("int", fields[name]) + require_int(increment, f"increments.{name}")
        return WP20WorkLedger(
            forward_circuit_evaluations=cast("int", fields["forward_circuit_evaluations"]),
            backward_circuit_evaluations=cast("int", fields["backward_circuit_evaluations"]),
            trajectory_gate_applications=cast("int", fields["trajectory_gate_applications"]),
            training_trajectories=cast("int", fields["training_trajectories"]),
            checkpoint_validation_trajectories=cast("int", fields["checkpoint_validation_trajectories"]),
            test_trajectories=cast("int", fields["test_trajectories"]),
            objective_calls=cast("int", fields["objective_calls"]),
            gradient_calls=cast("int", fields["gradient_calls"]),
            cross_trajectory_pairings=cast("int", fields["cross_trajectory_pairings"]),
            wall_time_seconds=float(fields["wall_time_seconds"]),
            peak_memory_bytes=int(fields["peak_memory_bytes"]),
        )

    def phase2_projection(self) -> dict[str, int]:
        """Project onto the immutable six-counter WP16 normalized-work schema."""
        return {
            "objective_evaluations": self.objective_calls,
            "gradient_evaluations": self.gradient_calls,
            "training_trajectories": self.training_trajectories,
            "checkpoint_validation_trajectories": self.checkpoint_validation_trajectories,
            "test_trajectories": self.test_trajectories,
            "trajectory_gate_applications": self.trajectory_gate_applications,
        }

    def normalized_compute(self, policy: NormalizedComputePolicy | None = None) -> float:
        """Return normalized compute under the exact WP20 policy."""
        resolved = DEFAULT_NORMALIZED_COMPUTE_POLICY if policy is None else policy
        if not isinstance(resolved, NormalizedComputePolicy):
            msg = "policy must be a NormalizedComputePolicy."
            raise TypeError(msg)
        return resolved.compute(self)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered field."""
        return {
            "schema_version": self.schema_version,
            **{name: getattr(self, name) for name in _WORK_LEDGER_COUNTER_FIELDS},
            "total_sampled_trajectories": self.total_sampled_trajectories,
            "wall_time_seconds": self.wall_time_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
        }

    @property
    def content_checksum(self) -> str:
        """Return the checksum of the complete detailed ledger."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> WP20WorkLedger:
        """Decode and verify a detailed work ledger."""
        expected = frozenset({
            "schema_version",
            *_WORK_LEDGER_COUNTER_FIELDS,
            "total_sampled_trajectories",
            "wall_time_seconds",
            "peak_memory_bytes",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 work ledger")
        if mapping["schema_version"] != WP20_WORK_LEDGER_SCHEMA_VERSION:
            msg = f"schema_version must be {WP20_WORK_LEDGER_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        ledger = cls(
            **{name: cast("int", mapping[name]) for name in _WORK_LEDGER_COUNTER_FIELDS},
            wall_time_seconds=cast("float", mapping["wall_time_seconds"]),
            peak_memory_bytes=cast("int", mapping["peak_memory_bytes"]),
        )
        if mapping["total_sampled_trajectories"] != ledger.total_sampled_trajectories:
            msg = "total_sampled_trajectories is not the sum of the three trajectory roles."
            raise ValueError(msg)
        if mapping["content_checksum"] != ledger.content_checksum:
            msg = "WP20 work ledger checksum changed during normalization."
            raise ValueError(msg)
        return ledger

    def to_json(self) -> str:
        """Return canonical sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> WP20WorkLedger:
        """Decode canonical sealed JSON."""
        return cls.from_dict(load_canonical_json_object(payload))


def wp20_work_from_noisy_krotov(
    stage: TrainingStageConfig,
    execution: NoisyKrotovStageExecution,
    *,
    wall_time_seconds: float = 0.0,
    peak_memory_bytes: int = 0,
) -> WP20WorkLedger:
    """Project one verified WP17 Krotov execution onto the common WP20 ledger.

    Forward counts include every noisy trajectory replay and map-generation
    pass. Noiseless objective calls each contribute one forward circuit pass.
    Every completed Krotov update contributes one backward pass per trajectory,
    or one pass for a noiseless update; dense cross contractions remain in the
    separate quadratic pairing counter.

    Args:
        stage: Exact resolved stage used by the execution.
        execution: Successfully validated WP17 execution evidence.
        wall_time_seconds: Optional measured stage wall time.
        peak_memory_bytes: Optional measured stage peak memory.

    Returns:
        The mechanically derived common-method work ledger.
    """
    if not isinstance(stage, TrainingStageConfig):
        msg = "stage must be a TrainingStageConfig."
        raise TypeError(msg)
    if not isinstance(execution, NoisyKrotovStageExecution):
        msg = "execution must be a NoisyKrotovStageExecution."
        raise TypeError(msg)
    if (
        execution.stage_index != stage.stage_index
        or execution.stage_id != stage.stage_id
        or execution.stage_configuration_checksum != stage.configuration_checksum
    ):
        msg = "Noisy Krotov execution does not identify the supplied stage."
        raise ValueError(msg)
    raw_work = execution.normalized_work
    work = {
        name: require_int(raw_work[name], f"normalized_work.{name}")
        for name in (
            "objective_evaluations",
            "gradient_evaluations",
            "training_trajectories",
            "checkpoint_validation_trajectories",
            "test_trajectories",
            "trajectory_gate_applications",
        )
    }
    validation_evaluations = 0
    if stage.checkpoint_validation.enabled:
        count = require_int(
            stage.checkpoint_validation.trajectory_count,
            "checkpoint_validation.trajectory_count",
            minimum=1,
        )
        sampled = work["checkpoint_validation_trajectories"]
        if sampled % count:
            msg = "Checkpoint-validation trajectories do not divide into configured evaluations."
            raise ValueError(msg)
        validation_evaluations = sampled // count - len(execution.checkpoint_validation_ensembles)
        if validation_evaluations < 0:
            msg = "Checkpoint-validation map generation exceeds recorded trajectory work."
            raise ValueError(msg)
    training_evaluations = work["objective_evaluations"] - validation_evaluations
    if training_evaluations < 0:
        msg = "Krotov objective work contains fewer calls than checkpoint validation requires."
        raise ValueError(msg)
    forward = work["training_trajectories"] + work["checkpoint_validation_trajectories"] + work["test_trajectories"]
    if stage.trajectory_count == 0:
        forward += training_evaluations
    completed_updates = execution.trace[-1].global_iteration
    backward = completed_updates * max(1, stage.trajectory_count)
    return WP20WorkLedger(
        forward_circuit_evaluations=forward,
        backward_circuit_evaluations=backward,
        trajectory_gate_applications=work["trajectory_gate_applications"],
        training_trajectories=work["training_trajectories"],
        checkpoint_validation_trajectories=work["checkpoint_validation_trajectories"],
        test_trajectories=work["test_trajectories"],
        objective_calls=work["objective_evaluations"],
        gradient_calls=work["gradient_evaluations"],
        cross_trajectory_pairings=execution.cross_trajectory_pairings,
        wall_time_seconds=wall_time_seconds,
        peak_memory_bytes=peak_memory_bytes,
    )


@dataclass(frozen=True, slots=True)
class NormalizedComputePolicy:
    """Versioned conversion from detailed counters to one compute scalar.

    Objective, gradient, and trajectory counts remain diagnostics. They are not
    added a second time because their concrete simulation work is represented
    by circuit evaluations, trajectory-gate applications, and cross pairings.
    """

    policy_id: str = WP20_NORMALIZED_COMPUTE_POLICY_ID
    forward_circuit_evaluation_weight: float = 1.0
    backward_circuit_evaluation_weight: float = 1.0
    trajectory_gate_application_weight: float = 1.0
    cross_trajectory_pairing_weight: float = 1.0
    schema_version: str = field(default=WP20_NORMALIZED_COMPUTE_POLICY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require the frozen WP20 policy and exact unit weights."""
        if require_slug(self.policy_id, "policy_id") != WP20_NORMALIZED_COMPUTE_POLICY_ID:
            msg = f"policy_id must be {WP20_NORMALIZED_COMPUTE_POLICY_ID!r}."
            raise ValueError(msg)
        for name in (
            "forward_circuit_evaluation_weight",
            "backward_circuit_evaluation_weight",
            "trajectory_gate_application_weight",
            "cross_trajectory_pairing_weight",
        ):
            weight = float(require_float(getattr(self, name), name, minimum=0.0))
            if weight.hex() != (1.0).hex():
                msg = f"{name} must be the frozen unit weight 1.0."
                raise ValueError(msg)
            object.__setattr__(self, name, weight)

    def compute(self, work: WP20WorkLedger) -> float:
        """Compute the exact normalized-work scalar."""
        if not isinstance(work, WP20WorkLedger):
            msg = "work must be a WP20WorkLedger."
            raise TypeError(msg)
        return float(
            self.forward_circuit_evaluation_weight * work.forward_circuit_evaluations
            + self.backward_circuit_evaluation_weight * work.backward_circuit_evaluations
            + self.trajectory_gate_application_weight * work.trajectory_gate_applications
            + self.cross_trajectory_pairing_weight * work.cross_trajectory_pairings
        )

    def _content_dict(self) -> dict[str, object]:
        """Return all policy fields."""
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "forward_circuit_evaluation_weight": self.forward_circuit_evaluation_weight,
            "backward_circuit_evaluation_weight": self.backward_circuit_evaluation_weight,
            "trajectory_gate_application_weight": self.trajectory_gate_application_weight,
            "cross_trajectory_pairing_weight": self.cross_trajectory_pairing_weight,
        }

    @property
    def content_checksum(self) -> str:
        """Return the policy checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed policy data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> NormalizedComputePolicy:
        """Decode and verify the frozen policy."""
        expected = frozenset({
            "schema_version",
            "policy_id",
            "forward_circuit_evaluation_weight",
            "backward_circuit_evaluation_weight",
            "trajectory_gate_application_weight",
            "cross_trajectory_pairing_weight",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 normalized-compute policy")
        if mapping["schema_version"] != WP20_NORMALIZED_COMPUTE_POLICY_SCHEMA_VERSION:
            msg = f"schema_version must be {WP20_NORMALIZED_COMPUTE_POLICY_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        policy = cls(
            policy_id=cast("str", mapping["policy_id"]),
            forward_circuit_evaluation_weight=cast("float", mapping["forward_circuit_evaluation_weight"]),
            backward_circuit_evaluation_weight=cast("float", mapping["backward_circuit_evaluation_weight"]),
            trajectory_gate_application_weight=cast("float", mapping["trajectory_gate_application_weight"]),
            cross_trajectory_pairing_weight=cast("float", mapping["cross_trajectory_pairing_weight"]),
        )
        if mapping["content_checksum"] != policy.content_checksum:
            msg = "Normalized-compute policy checksum changed during normalization."
            raise ValueError(msg)
        return policy


DEFAULT_NORMALIZED_COMPUTE_POLICY = NormalizedComputePolicy()


@dataclass(frozen=True, slots=True)
class LogicalEventSignature:
    """Complete stable signature of one logical circuit event."""

    ordinal: int
    logical_gate_id: int | str
    name: str
    sites: tuple[int, ...]
    parameter_index: int | None
    angle_scale: float
    angle_offset: float
    fixed_parameters: tuple[float, ...]
    noise_enabled: bool
    schema_version: str = field(default=WP20_LOGICAL_EVENT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate event semantics and scalar types."""
        object.__setattr__(self, "ordinal", require_int(self.ordinal, "ordinal"))
        object.__setattr__(self, "logical_gate_id", _require_identifier(self.logical_gate_id, "logical_gate_id"))
        object.__setattr__(self, "name", require_slug(self.name, "name"))
        sites = _require_int_tuple(self.sites, "sites", unique=True)
        if len(sites) not in {1, 2}:
            msg = "sites must contain one or two unique qubit indices."
            raise ValueError(msg)
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "parameter_index", _require_optional_int(self.parameter_index, "parameter_index"))
        object.__setattr__(self, "angle_scale", require_float(self.angle_scale, "angle_scale"))
        object.__setattr__(self, "angle_offset", require_float(self.angle_offset, "angle_offset"))
        object.__setattr__(self, "fixed_parameters", _require_float_tuple(self.fixed_parameters, "fixed_parameters"))
        object.__setattr__(self, "noise_enabled", require_bool(self.noise_enabled, "noise_enabled"))

    def _content_dict(self) -> dict[str, object]:
        """Return signature content."""
        return {
            "schema_version": self.schema_version,
            "ordinal": self.ordinal,
            "logical_gate_id": self.logical_gate_id,
            "name": self.name,
            "sites": list(self.sites),
            "parameter_index": self.parameter_index,
            "angle_scale": self.angle_scale,
            "angle_offset": self.angle_offset,
            "fixed_parameters": list(self.fixed_parameters),
            "noise_enabled": self.noise_enabled,
        }

    @property
    def content_checksum(self) -> str:
        """Return the signature checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed signature data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> LogicalEventSignature:
        """Decode a sealed logical event."""
        expected = frozenset({
            "schema_version",
            "ordinal",
            "logical_gate_id",
            "name",
            "sites",
            "parameter_index",
            "angle_scale",
            "angle_offset",
            "fixed_parameters",
            "noise_enabled",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 logical event")
        if mapping["schema_version"] != WP20_LOGICAL_EVENT_SCHEMA_VERSION:
            msg = f"schema_version must be {WP20_LOGICAL_EVENT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        event = cls(
            ordinal=cast("int", mapping["ordinal"]),
            logical_gate_id=cast("int | str", mapping["logical_gate_id"]),
            name=cast("str", mapping["name"]),
            sites=cast("tuple[int, ...]", mapping["sites"]),
            parameter_index=cast("int | None", mapping["parameter_index"]),
            angle_scale=cast("float", mapping["angle_scale"]),
            angle_offset=cast("float", mapping["angle_offset"]),
            fixed_parameters=cast("tuple[float, ...]", mapping["fixed_parameters"]),
            noise_enabled=cast("bool", mapping["noise_enabled"]),
        )
        if mapping["content_checksum"] != event.content_checksum:
            msg = "Logical-event checksum changed during normalization."
            raise ValueError(msg)
        return event


@dataclass(frozen=True, slots=True)
class NativeEventSignature:
    """Full stable native-event signature used to authorize test coupling."""

    ordinal: int
    native_gate_id: int | str
    logical_gate_id: int | str
    name: str
    sites: tuple[int, ...]
    parameter_index: int | None
    angle_scale: float
    angle_offset: float
    fixed_parameters: tuple[float, ...]
    noise_enabled: bool
    source_logical_gate_index: int
    source_gate_name: str
    source_sites: tuple[int, ...]
    source_parameter_index: int | None
    basis_change_relationship: str
    schema_version: str = field(default=WP20_NATIVE_EVENT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the native event and complete source provenance."""
        object.__setattr__(self, "ordinal", require_int(self.ordinal, "ordinal"))
        object.__setattr__(self, "native_gate_id", _require_identifier(self.native_gate_id, "native_gate_id"))
        object.__setattr__(self, "logical_gate_id", _require_identifier(self.logical_gate_id, "logical_gate_id"))
        object.__setattr__(self, "name", require_slug(self.name, "name"))
        sites = _require_int_tuple(self.sites, "sites", unique=True)
        if len(sites) not in {1, 2}:
            msg = "sites must contain one or two unique qubit indices."
            raise ValueError(msg)
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "parameter_index", _require_optional_int(self.parameter_index, "parameter_index"))
        object.__setattr__(self, "angle_scale", require_float(self.angle_scale, "angle_scale"))
        object.__setattr__(self, "angle_offset", require_float(self.angle_offset, "angle_offset"))
        object.__setattr__(self, "fixed_parameters", _require_float_tuple(self.fixed_parameters, "fixed_parameters"))
        object.__setattr__(self, "noise_enabled", require_bool(self.noise_enabled, "noise_enabled"))
        object.__setattr__(
            self,
            "source_logical_gate_index",
            require_int(self.source_logical_gate_index, "source_logical_gate_index"),
        )
        object.__setattr__(self, "source_gate_name", require_slug(self.source_gate_name, "source_gate_name"))
        source_sites = _require_int_tuple(self.source_sites, "source_sites", unique=True)
        if len(source_sites) not in {1, 2}:
            msg = "source_sites must contain one or two unique qubit indices."
            raise ValueError(msg)
        object.__setattr__(self, "source_sites", source_sites)
        object.__setattr__(
            self,
            "source_parameter_index",
            _require_optional_int(self.source_parameter_index, "source_parameter_index"),
        )
        relationship = require_slug(self.basis_change_relationship, "basis_change_relationship")
        if relationship not in {"none", "rxx_h", "ryy_rx_pi_over_2"}:
            msg = "basis_change_relationship is not a frozen Quantinuum rewrite."
            raise ValueError(msg)
        object.__setattr__(self, "basis_change_relationship", relationship)

    def _content_dict(self) -> dict[str, object]:
        """Return signature content."""
        return {
            "schema_version": self.schema_version,
            "ordinal": self.ordinal,
            "native_gate_id": self.native_gate_id,
            "logical_gate_id": self.logical_gate_id,
            "name": self.name,
            "sites": list(self.sites),
            "parameter_index": self.parameter_index,
            "angle_scale": self.angle_scale,
            "angle_offset": self.angle_offset,
            "fixed_parameters": list(self.fixed_parameters),
            "noise_enabled": self.noise_enabled,
            "source_logical_gate_index": self.source_logical_gate_index,
            "source_gate_name": self.source_gate_name,
            "source_sites": list(self.source_sites),
            "source_parameter_index": self.source_parameter_index,
            "basis_change_relationship": self.basis_change_relationship,
        }

    @property
    def content_checksum(self) -> str:
        """Return the signature checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed signature data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> NativeEventSignature:
        """Decode a sealed native event."""
        expected = frozenset({
            "schema_version",
            "ordinal",
            "native_gate_id",
            "logical_gate_id",
            "name",
            "sites",
            "parameter_index",
            "angle_scale",
            "angle_offset",
            "fixed_parameters",
            "noise_enabled",
            "source_logical_gate_index",
            "source_gate_name",
            "source_sites",
            "source_parameter_index",
            "basis_change_relationship",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 native event")
        if mapping["schema_version"] != WP20_NATIVE_EVENT_SCHEMA_VERSION:
            msg = f"schema_version must be {WP20_NATIVE_EVENT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        event = cls(
            ordinal=cast("int", mapping["ordinal"]),
            native_gate_id=cast("int | str", mapping["native_gate_id"]),
            logical_gate_id=cast("int | str", mapping["logical_gate_id"]),
            name=cast("str", mapping["name"]),
            sites=cast("tuple[int, ...]", mapping["sites"]),
            parameter_index=cast("int | None", mapping["parameter_index"]),
            angle_scale=cast("float", mapping["angle_scale"]),
            angle_offset=cast("float", mapping["angle_offset"]),
            fixed_parameters=cast("tuple[float, ...]", mapping["fixed_parameters"]),
            noise_enabled=cast("bool", mapping["noise_enabled"]),
            source_logical_gate_index=cast("int", mapping["source_logical_gate_index"]),
            source_gate_name=cast("str", mapping["source_gate_name"]),
            source_sites=cast("tuple[int, ...]", mapping["source_sites"]),
            source_parameter_index=cast("int | None", mapping["source_parameter_index"]),
            basis_change_relationship=cast("str", mapping["basis_change_relationship"]),
        )
        if mapping["content_checksum"] != event.content_checksum:
            msg = "Native-event checksum changed during normalization."
            raise ValueError(msg)
        return event


def _event_parameter_upper_bound(events: Sequence[LogicalEventSignature | NativeEventSignature]) -> int:
    """Return the minimum declared parameter count implied by events."""
    indices = [event.parameter_index for event in events if event.parameter_index is not None]
    return 0 if not indices else max(cast("list[int]", indices)) + 1


@dataclass(frozen=True, slots=True)
class CircuitResourceMetrics:
    """Mechanically derived logical and Quantinuum-native circuit resources."""

    qubit_count: int
    trainable_parameter_count: int
    logical_events: tuple[LogicalEventSignature, ...]
    native_events: tuple[NativeEventSignature, ...]
    schema_version: str = field(default=WP20_CIRCUIT_RESOURCES_SCHEMA_VERSION, init=False)
    compiler_policy_id: str = field(default=PRIMARY_COMPILER_POLICY_ID, init=False)
    connectivity_id: str = field(default=PRIMARY_CONNECTIVITY, init=False)
    routing_policy_id: str = field(default=PRIMARY_ROUTING_POLICY_ID, init=False)
    counting_policy_id: str = field(default=PRIMARY_COUNTING_POLICY_ID, init=False)

    def __post_init__(self) -> None:
        """Validate event order, sites, provenance, and parameter bounds."""
        qubits = require_int(self.qubit_count, "qubit_count", minimum=1)
        parameters = require_int(self.trainable_parameter_count, "trainable_parameter_count")
        logical = tuple(self.logical_events)
        native = tuple(self.native_events)
        if not all(isinstance(event, LogicalEventSignature) for event in logical):
            msg = "logical_events must contain LogicalEventSignature values."
            raise TypeError(msg)
        if not all(isinstance(event, NativeEventSignature) for event in native):
            msg = "native_events must contain NativeEventSignature values."
            raise TypeError(msg)
        if tuple(event.ordinal for event in logical) != tuple(range(len(logical))):
            msg = "logical event ordinals must be contiguous and ordered."
            raise ValueError(msg)
        if tuple(event.ordinal for event in native) != tuple(range(len(native))):
            msg = "native event ordinals must be contiguous and ordered."
            raise ValueError(msg)
        if len({event.native_gate_id for event in native}) != len(native):
            msg = "native event identifiers must be unique."
            raise ValueError(msg)
        if any(site >= qubits for event in (*logical, *native) for site in event.sites):
            msg = "circuit event site lies outside the recorded qubit count."
            raise ValueError(msg)
        if any(event.source_logical_gate_index >= len(logical) for event in native):
            msg = "native event source index lies outside the logical circuit."
            raise ValueError(msg)
        for event in native:
            source = logical[event.source_logical_gate_index]
            if (
                event.logical_gate_id != source.logical_gate_id
                or event.source_gate_name != source.name
                or event.source_sites != source.sites
                or event.source_parameter_index != source.parameter_index
            ):
                msg = "native event provenance does not match its logical source event."
                raise ValueError(msg)
            if len(event.sites) == 2 and abs(event.sites[0] - event.sites[1]) != 1:
                msg = "Frozen Quantinuum native two-qubit events must lie on a linear-chain edge."
                raise ValueError(msg)
        if parameters < max(_event_parameter_upper_bound(logical), _event_parameter_upper_bound(native)):
            msg = "trainable_parameter_count is smaller than an event parameter index requires."
            raise ValueError(msg)
        object.__setattr__(self, "qubit_count", qubits)
        object.__setattr__(self, "trainable_parameter_count", parameters)
        object.__setattr__(self, "logical_events", logical)
        object.__setattr__(self, "native_events", native)

    @property
    def logical_one_qubit_gates(self) -> int:
        """Return the number of logical one-qubit gates."""
        return sum(len(event.sites) == 1 for event in self.logical_events)

    @property
    def logical_two_qubit_gates(self) -> int:
        """Return the number of logical two-qubit gates."""
        return sum(len(event.sites) == 2 for event in self.logical_events)

    @property
    def logical_depth(self) -> int:
        """Return logical dependency depth."""
        return _dependency_depth(self.qubit_count, tuple(event.sites for event in self.logical_events))

    @property
    def native_one_qubit_gates(self) -> int:
        """Return the number of compiled one-qubit gates."""
        return sum(len(event.sites) == 1 for event in self.native_events)

    @property
    def native_two_qubit_gates(self) -> int:
        """Return the number of compiled two-qubit gates."""
        return sum(len(event.sites) == 2 for event in self.native_events)

    @property
    def native_two_qubit_gates_per_chain_edge(self) -> tuple[int, ...]:
        """Return compiled two-qubit counts in chain-edge order.

        Entry ``i`` counts native two-qubit events on the physical edge
        ``(i, i + 1)``. The tuple is therefore mechanically fixed to length
        ``qubit_count - 1`` and its sum equals ``native_two_qubit_gates``.
        """
        counts = [0] * (self.qubit_count - 1)
        for event in self.native_events:
            if len(event.sites) != 2:
                continue
            left, right = sorted(event.sites)
            if right != left + 1:
                msg = "Frozen Quantinuum native two-qubit events must lie on a linear-chain edge."
                raise ValueError(msg)
            counts[left] += 1
        return tuple(counts)

    @property
    def native_depth(self) -> int:
        """Return compiled dependency depth."""
        return _dependency_depth(self.qubit_count, tuple(event.sites for event in self.native_events))

    @property
    def logical_circuit_checksum(self) -> str:
        """Return a checksum of the complete logical event stream."""
        return canonical_checksum({
            "qubit_count": self.qubit_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "events": [event.to_dict() for event in self.logical_events],
        })

    @property
    def native_circuit_checksum(self) -> str:
        """Return a checksum of compilation policy and every native event."""
        return canonical_checksum({
            "compiler_policy_id": self.compiler_policy_id,
            "connectivity_id": self.connectivity_id,
            "routing_policy_id": self.routing_policy_id,
            "events": [event.to_dict() for event in self.native_events],
        })

    def _content_dict(self) -> dict[str, object]:
        """Return complete resources and event evidence."""
        return {
            "schema_version": self.schema_version,
            "compiler_policy_id": self.compiler_policy_id,
            "connectivity_id": self.connectivity_id,
            "routing_policy_id": self.routing_policy_id,
            "counting_policy_id": self.counting_policy_id,
            "qubit_count": self.qubit_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "logical_one_qubit_gates": self.logical_one_qubit_gates,
            "logical_two_qubit_gates": self.logical_two_qubit_gates,
            "logical_depth": self.logical_depth,
            "native_one_qubit_gates": self.native_one_qubit_gates,
            "native_two_qubit_gates": self.native_two_qubit_gates,
            "native_two_qubit_gates_per_chain_edge": list(self.native_two_qubit_gates_per_chain_edge),
            "native_depth": self.native_depth,
            "logical_circuit_checksum": self.logical_circuit_checksum,
            "native_circuit_checksum": self.native_circuit_checksum,
            "logical_events": [event.to_dict() for event in self.logical_events],
            "native_events": [event.to_dict() for event in self.native_events],
        }

    @property
    def content_checksum(self) -> str:
        """Return the checksum of all resource evidence."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return strict sealed resource data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> CircuitResourceMetrics:
        """Decode and mechanically verify circuit resources."""
        aliases = {
            "logical_one_qubit_gates",
            "logical_two_qubit_gates",
            "logical_depth",
            "native_one_qubit_gates",
            "native_two_qubit_gates",
            "native_two_qubit_gates_per_chain_edge",
            "native_depth",
            "logical_circuit_checksum",
            "native_circuit_checksum",
        }
        expected = frozenset({
            "schema_version",
            "compiler_policy_id",
            "connectivity_id",
            "routing_policy_id",
            "counting_policy_id",
            "qubit_count",
            "trainable_parameter_count",
            "logical_events",
            "native_events",
            *aliases,
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 circuit resources")
        frozen = {
            "schema_version": WP20_CIRCUIT_RESOURCES_SCHEMA_VERSION,
            "compiler_policy_id": PRIMARY_COMPILER_POLICY_ID,
            "connectivity_id": PRIMARY_CONNECTIVITY,
            "routing_policy_id": PRIMARY_ROUTING_POLICY_ID,
            "counting_policy_id": PRIMARY_COUNTING_POLICY_ID,
        }
        for name, value in frozen.items():
            if mapping[name] != value:
                msg = f"{name} must be the frozen WP20 value {value!r}."
                raise ValueError(msg)
        logical_data = mapping["logical_events"]
        native_data = mapping["native_events"]
        if isinstance(logical_data, (str, bytes)) or not isinstance(logical_data, Sequence):
            msg = "logical_events must be a sequence."
            raise TypeError(msg)
        if isinstance(native_data, (str, bytes)) or not isinstance(native_data, Sequence):
            msg = "native_events must be a sequence."
            raise TypeError(msg)
        resources = cls(
            qubit_count=cast("int", mapping["qubit_count"]),
            trainable_parameter_count=cast("int", mapping["trainable_parameter_count"]),
            logical_events=tuple(LogicalEventSignature.from_dict(event) for event in logical_data),
            native_events=tuple(NativeEventSignature.from_dict(event) for event in native_data),
        )
        for name in aliases:
            serialized_value = mapping[name]
            if name == "native_two_qubit_gates_per_chain_edge":
                serialized_value = _require_int_tuple(serialized_value, name)
            if serialized_value != getattr(resources, name):
                msg = f"Serialized {name} is not derived from the event evidence."
                raise ValueError(msg)
        if mapping["content_checksum"] != resources.content_checksum:
            msg = "Circuit-resource checksum changed during normalization."
            raise ValueError(msg)
        return resources

    def to_json(self) -> str:
        """Return canonical sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> CircuitResourceMetrics:
        """Decode canonical sealed JSON."""
        return cls.from_dict(load_canonical_json_object(payload))


def _logical_event(gate: ParameterizedGate, ordinal: int) -> LogicalEventSignature:
    """Convert one logical gate to immutable event evidence."""
    if gate.data_map is not None:
        msg = "WP20 state-preparation resource accounting does not accept data-dependent gates."
        raise ValueError(msg)
    return LogicalEventSignature(
        ordinal=ordinal,
        logical_gate_id=ordinal if gate.logical_gate_id is None else gate.logical_gate_id,
        name=gate.name,
        sites=tuple(gate.sites),
        parameter_index=gate.param_index,
        angle_scale=float(gate.angle_scale),
        angle_offset=float(gate.angle_offset),
        fixed_parameters=tuple(float(value) for value in gate.fixed_params),
        noise_enabled=gate.noise_enabled,
    )


def _native_event(
    gate: ParameterizedGate,
    ordinal: int,
    source: LogicalToNativeMapping,
) -> NativeEventSignature:
    """Convert one compiled gate and its provenance to event evidence."""
    if gate.data_map is not None:
        msg = "WP20 native resource evidence cannot serialize data-dependent gates."
        raise ValueError(msg)
    if gate.native_gate_id is None:
        msg = "Frozen Quantinuum compilation must assign every native event a stable identifier."
        raise ValueError(msg)
    logical_gate_id = source.logical_gate_id
    return NativeEventSignature(
        ordinal=ordinal,
        native_gate_id=gate.native_gate_id,
        logical_gate_id=logical_gate_id,
        name=gate.name,
        sites=tuple(gate.sites),
        parameter_index=gate.param_index,
        angle_scale=float(gate.angle_scale),
        angle_offset=float(gate.angle_offset),
        fixed_parameters=tuple(float(value) for value in gate.fixed_params),
        noise_enabled=gate.noise_enabled,
        source_logical_gate_index=source.source_logical_gate_index,
        source_gate_name=source.source_gate_name,
        source_sites=source.source_sites,
        source_parameter_index=source.source_parameter_index,
        basis_change_relationship=source.basis_change_relationship,
    )


def measure_circuit_resources(circuit: ParameterizedCircuit) -> CircuitResourceMetrics:
    """Compile and mechanically count one circuit under the frozen policy.

    Args:
        circuit: Logical parameterized state-preparation circuit.

    Returns:
        Strict logical/native resource evidence and full event signatures.
    """
    if not isinstance(circuit, ParameterizedCircuit):
        msg = "circuit must be a ParameterizedCircuit."
        raise TypeError(msg)
    logical = tuple(_logical_event(gate, index) for index, gate in enumerate(circuit.gates))
    compilation = compile_quantinuum_native(circuit)
    source_by_native_index: dict[int, LogicalToNativeMapping] = {}
    for source in compilation.mapping:
        for native_index in source.native_gate_indices:
            if native_index in source_by_native_index:
                msg = "Quantinuum compilation maps one native event to multiple logical sources."
                raise ValueError(msg)
            source_by_native_index[native_index] = source
    if set(source_by_native_index) != set(range(len(compilation.circuit.gates))):
        msg = "Quantinuum compilation provenance does not cover every native event exactly once."
        raise ValueError(msg)
    native = tuple(
        _native_event(gate, index, source_by_native_index[index])
        for index, gate in enumerate(compilation.circuit.gates)
    )
    return CircuitResourceMetrics(
        qubit_count=circuit.num_qubits,
        trainable_parameter_count=circuit.num_params,
        logical_events=logical,
        native_events=native,
    )


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    """Frozen joint per-chain-edge native and normalized-compute budget."""

    native_two_qubit_gate_cap_per_chain_edge: int
    normalized_compute_cap: float
    normalized_compute_policy_checksum: str = field(
        default_factory=lambda: DEFAULT_NORMALIZED_COMPUTE_POLICY.content_checksum
    )
    schema_version: str = field(default=WP20_RESOURCE_BUDGET_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate caps and bind the only accepted compute policy."""
        object.__setattr__(
            self,
            "native_two_qubit_gate_cap_per_chain_edge",
            require_int(
                self.native_two_qubit_gate_cap_per_chain_edge,
                "native_two_qubit_gate_cap_per_chain_edge",
            ),
        )
        object.__setattr__(
            self,
            "normalized_compute_cap",
            require_float(self.normalized_compute_cap, "normalized_compute_cap", minimum=0.0),
        )
        checksum = require_checksum(
            self.normalized_compute_policy_checksum,
            "normalized_compute_policy_checksum",
        )
        if checksum != DEFAULT_NORMALIZED_COMPUTE_POLICY.content_checksum:
            msg = "Resource budgets must use the frozen WP20 normalized-compute policy."
            raise ValueError(msg)
        object.__setattr__(self, "normalized_compute_policy_checksum", checksum)

    def _content_dict(self) -> dict[str, object]:
        """Return budget content."""
        return {
            "schema_version": self.schema_version,
            "native_two_qubit_gate_cap_per_chain_edge": self.native_two_qubit_gate_cap_per_chain_edge,
            "normalized_compute_cap": self.normalized_compute_cap,
            "normalized_compute_policy_checksum": self.normalized_compute_policy_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Return the budget checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed budget data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> ResourceBudget:
        """Decode a sealed resource budget."""
        expected = frozenset({
            "schema_version",
            "native_two_qubit_gate_cap_per_chain_edge",
            "normalized_compute_cap",
            "normalized_compute_policy_checksum",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 resource budget")
        if mapping["schema_version"] != WP20_RESOURCE_BUDGET_SCHEMA_VERSION:
            msg = f"schema_version must be {WP20_RESOURCE_BUDGET_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        budget = cls(
            native_two_qubit_gate_cap_per_chain_edge=cast(
                "int",
                mapping["native_two_qubit_gate_cap_per_chain_edge"],
            ),
            normalized_compute_cap=cast("float", mapping["normalized_compute_cap"]),
            normalized_compute_policy_checksum=cast("str", mapping["normalized_compute_policy_checksum"]),
        )
        if mapping["content_checksum"] != budget.content_checksum:
            msg = "Resource-budget checksum changed during normalization."
            raise ValueError(msg)
        return budget


@dataclass(frozen=True, slots=True)
class ReachableResourceStratum:
    """One attempted and actually reachable circuit/work stratum."""

    stratum_id: str
    circuit_resources: CircuitResourceMetrics
    work: WP20WorkLedger
    schema_version: str = field(default=WP20_RESOURCE_STRATUM_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the exact typed resource evidence."""
        object.__setattr__(self, "stratum_id", require_slug(self.stratum_id, "stratum_id"))
        if not isinstance(self.circuit_resources, CircuitResourceMetrics):
            msg = "circuit_resources must be CircuitResourceMetrics."
            raise TypeError(msg)
        if not isinstance(self.work, WP20WorkLedger):
            msg = "work must be a WP20WorkLedger."
            raise TypeError(msg)

    @property
    def normalized_compute(self) -> float:
        """Return normalized compute from the sealed detailed ledger."""
        return self.work.normalized_compute()

    def _content_dict(self) -> dict[str, object]:
        """Return complete stratum evidence."""
        return {
            "schema_version": self.schema_version,
            "stratum_id": self.stratum_id,
            "circuit_resources": self.circuit_resources.to_dict(),
            "work": self.work.to_dict(),
            "normalized_compute": self.normalized_compute,
        }

    @property
    def content_checksum(self) -> str:
        """Return the stratum checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed stratum data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> ReachableResourceStratum:
        """Decode and verify a reachable stratum."""
        expected = frozenset({
            "schema_version",
            "stratum_id",
            "circuit_resources",
            "work",
            "normalized_compute",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 resource stratum")
        if mapping["schema_version"] != WP20_RESOURCE_STRATUM_SCHEMA_VERSION:
            msg = f"schema_version must be {WP20_RESOURCE_STRATUM_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        stratum = cls(
            stratum_id=cast("str", mapping["stratum_id"]),
            circuit_resources=CircuitResourceMetrics.from_dict(mapping["circuit_resources"]),
            work=WP20WorkLedger.from_dict(mapping["work"]),
        )
        if mapping["normalized_compute"] != stratum.normalized_compute:
            msg = "Stratum normalized_compute is not derived from its work ledger."
            raise ValueError(msg)
        if mapping["content_checksum"] != stratum.content_checksum:
            msg = "Resource-stratum checksum changed during normalization."
            raise ValueError(msg)
        return stratum


@dataclass(frozen=True, slots=True)
class SelectedResourceStratum:
    """Deterministically selected reachable stratum within both caps."""

    budget: ResourceBudget
    selected: ReachableResourceStratum
    native_two_qubit_residuals_per_chain_edge: tuple[int, ...]
    normalized_compute_residual: float
    status: Literal["selected"] = field(default="selected", init=False)
    schema_version: str = field(default=WP20_RESOURCE_SELECTION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate selection feasibility and exact residual gaps."""
        if not isinstance(self.budget, ResourceBudget):
            msg = "budget must be a ResourceBudget."
            raise TypeError(msg)
        if not isinstance(self.selected, ReachableResourceStratum):
            msg = "selected must be a ReachableResourceStratum."
            raise TypeError(msg)
        expected_native = tuple(
            self.budget.native_two_qubit_gate_cap_per_chain_edge - count
            for count in self.selected.circuit_resources.native_two_qubit_gates_per_chain_edge
        )
        expected_compute = self.budget.normalized_compute_cap - self.selected.normalized_compute
        if any(residual < 0 for residual in expected_native) or expected_compute < 0.0:
            msg = "A selected resource stratum must satisfy both caps."
            raise ValueError(msg)
        residuals = _require_int_tuple(
            self.native_two_qubit_residuals_per_chain_edge,
            "native_two_qubit_residuals_per_chain_edge",
        )
        if residuals != expected_native:
            msg = "native_two_qubit_residuals_per_chain_edge do not equal cap minus each edge count."
            raise ValueError(msg)
        residual = float(
            require_float(
                self.normalized_compute_residual,
                "normalized_compute_residual",
                minimum=0.0,
            )
        )
        if residual.hex() != float(expected_compute).hex():
            msg = "normalized_compute_residual does not equal cap minus selected work."
            raise ValueError(msg)
        object.__setattr__(self, "native_two_qubit_residuals_per_chain_edge", residuals)
        object.__setattr__(self, "normalized_compute_residual", residual)

    @property
    def exact_native_match(self) -> bool:
        """Whether every chain edge exactly reaches its native cap."""
        return all(residual == 0 for residual in self.native_two_qubit_residuals_per_chain_edge)

    @property
    def exact_compute_match(self) -> bool:
        """Whether the selected normalized work exactly reaches its cap."""
        return float(self.normalized_compute_residual).hex() == (0.0).hex()

    def _content_dict(self) -> dict[str, object]:
        """Return complete selected-outcome evidence."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "budget": self.budget.to_dict(),
            "selected": self.selected.to_dict(),
            "native_two_qubit_residuals_per_chain_edge": list(self.native_two_qubit_residuals_per_chain_edge),
            "normalized_compute_residual": self.normalized_compute_residual,
            "exact_native_match": self.exact_native_match,
            "exact_compute_match": self.exact_compute_match,
        }

    @property
    def content_checksum(self) -> str:
        """Return the selected outcome checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed selected-outcome data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> SelectedResourceStratum:
        """Decode a sealed selected outcome."""
        expected = frozenset({
            "schema_version",
            "status",
            "budget",
            "selected",
            "native_two_qubit_residuals_per_chain_edge",
            "normalized_compute_residual",
            "exact_native_match",
            "exact_compute_match",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 selected resource stratum")
        if mapping["schema_version"] != WP20_RESOURCE_SELECTION_SCHEMA_VERSION or mapping["status"] != "selected":
            msg = "Selected resource outcome has an unsupported schema or status."
            raise ValueError(msg)
        result = cls(
            budget=ResourceBudget.from_dict(mapping["budget"]),
            selected=ReachableResourceStratum.from_dict(mapping["selected"]),
            native_two_qubit_residuals_per_chain_edge=cast(
                "tuple[int, ...]",
                mapping["native_two_qubit_residuals_per_chain_edge"],
            ),
            normalized_compute_residual=cast("float", mapping["normalized_compute_residual"]),
        )
        if mapping["exact_native_match"] is not result.exact_native_match:
            msg = "exact_native_match is not derived from the residual."
            raise ValueError(msg)
        if mapping["exact_compute_match"] is not result.exact_compute_match:
            msg = "exact_compute_match is not derived from the residual."
            raise ValueError(msg)
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Selected resource outcome checksum changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class InfeasibleResourceBudget:
    """Typed evidence that no attempted stratum satisfies both caps."""

    budget: ResourceBudget
    attempted_strata: tuple[ReachableResourceStratum, ...]
    reason: str = "no_reachable_stratum_within_joint_caps"
    status: Literal["infeasible"] = field(default="infeasible", init=False)
    schema_version: str = field(default=WP20_RESOURCE_SELECTION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate attempts and prove that each violates at least one cap."""
        if not isinstance(self.budget, ResourceBudget):
            msg = "budget must be a ResourceBudget."
            raise TypeError(msg)
        attempts = tuple(self.attempted_strata)
        if not attempts or not all(isinstance(item, ReachableResourceStratum) for item in attempts):
            msg = "attempted_strata must contain ReachableResourceStratum values."
            raise TypeError(msg)
        identifiers = tuple(item.stratum_id for item in attempts)
        if len(identifiers) != len(set(identifiers)):
            msg = "attempted resource strata must have unique identifiers."
            raise ValueError(msg)
        if self.reason != "no_reachable_stratum_within_joint_caps":
            msg = "reason must identify the frozen joint-cap infeasibility."
            raise ValueError(msg)
        if any(_within_budget(item, self.budget) for item in attempts):
            msg = "An infeasible outcome cannot contain a stratum within both caps."
            raise ValueError(msg)
        object.__setattr__(self, "attempted_strata", attempts)

    def _content_dict(self) -> dict[str, object]:
        """Return complete infeasibility evidence."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "budget": self.budget.to_dict(),
            "attempted_strata": [item.to_dict() for item in self.attempted_strata],
            "reason": self.reason,
        }

    @property
    def content_checksum(self) -> str:
        """Return the infeasible-outcome checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed infeasible-outcome data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> InfeasibleResourceBudget:
        """Decode a sealed infeasible outcome."""
        expected = frozenset({
            "schema_version",
            "status",
            "budget",
            "attempted_strata",
            "reason",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 infeasible resource budget")
        if mapping["schema_version"] != WP20_RESOURCE_SELECTION_SCHEMA_VERSION or mapping["status"] != "infeasible":
            msg = "Infeasible resource outcome has an unsupported schema or status."
            raise ValueError(msg)
        attempts = mapping["attempted_strata"]
        if isinstance(attempts, (str, bytes)) or not isinstance(attempts, Sequence):
            msg = "attempted_strata must be a sequence."
            raise TypeError(msg)
        result = cls(
            budget=ResourceBudget.from_dict(mapping["budget"]),
            attempted_strata=tuple(ReachableResourceStratum.from_dict(item) for item in attempts),
            reason=cast("str", mapping["reason"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Infeasible resource outcome checksum changed during normalization."
            raise ValueError(msg)
        return result


ResourceSelectionOutcome: TypeAlias = SelectedResourceStratum | InfeasibleResourceBudget


def resource_selection_outcome_from_dict(data: object) -> ResourceSelectionOutcome:
    """Decode a selected or infeasible resource outcome by status."""
    mapping = require_mapping(data, "WP20 resource selection outcome")
    if mapping.get("status") == "selected":
        return SelectedResourceStratum.from_dict(mapping)
    if mapping.get("status") == "infeasible":
        return InfeasibleResourceBudget.from_dict(mapping)
    msg = "WP20 resource selection status must be 'selected' or 'infeasible'."
    raise ValueError(msg)


def _within_budget(stratum: ReachableResourceStratum, budget: ResourceBudget) -> bool:
    """Return whether one stratum satisfies both frozen caps."""
    return (
        all(
            count <= budget.native_two_qubit_gate_cap_per_chain_edge
            for count in stratum.circuit_resources.native_two_qubit_gates_per_chain_edge
        )
        and stratum.normalized_compute <= budget.normalized_compute_cap
    )


def select_reachable_resource_stratum(
    strata: Sequence[ReachableResourceStratum],
    budget: ResourceBudget,
) -> ResourceSelectionOutcome:
    """Select the richest reachable stratum without observing performance.

    Selection maximizes native two-qubit count, then normalized compute, and
    finally uses the lexicographically smallest stable stratum identifier.
    This is deterministic for non-monotonic growth paths and never claims an
    unreachable exact match.
    """
    if not isinstance(budget, ResourceBudget):
        msg = "budget must be a ResourceBudget."
        raise TypeError(msg)
    attempted = tuple(strata)
    if not attempted or not all(isinstance(item, ReachableResourceStratum) for item in attempted):
        msg = "strata must contain at least one ReachableResourceStratum."
        raise TypeError(msg)
    identifiers = tuple(item.stratum_id for item in attempted)
    if len(identifiers) != len(set(identifiers)):
        msg = "Reachable resource strata must have unique identifiers."
        raise ValueError(msg)
    eligible = [item for item in attempted if _within_budget(item, budget)]
    if not eligible:
        return InfeasibleResourceBudget(
            budget=budget,
            attempted_strata=tuple(sorted(attempted, key=lambda item: (item.stratum_id, item.content_checksum))),
        )
    selected = min(
        eligible,
        key=lambda item: (
            -item.circuit_resources.native_two_qubit_gates,
            -item.normalized_compute,
            item.stratum_id,
            item.content_checksum,
        ),
    )
    return SelectedResourceStratum(
        budget=budget,
        selected=selected,
        native_two_qubit_residuals_per_chain_edge=tuple(
            budget.native_two_qubit_gate_cap_per_chain_edge - count
            for count in selected.circuit_resources.native_two_qubit_gates_per_chain_edge
        ),
        normalized_compute_residual=budget.normalized_compute_cap - selected.normalized_compute,
    )


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One fidelity-bearing point over a reachable resource stratum."""

    stratum: ReachableResourceStratum
    fidelity: float
    schema_version: str = field(default=WP20_PARETO_POINT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate typed stratum evidence and bounded fidelity."""
        if not isinstance(self.stratum, ReachableResourceStratum):
            msg = "stratum must be a ReachableResourceStratum."
            raise TypeError(msg)
        object.__setattr__(self, "fidelity", require_float(self.fidelity, "fidelity", minimum=0.0, maximum=1.0))

    def _content_dict(self) -> dict[str, object]:
        """Return point content."""
        return {
            "schema_version": self.schema_version,
            "stratum": self.stratum.to_dict(),
            "fidelity": self.fidelity,
        }

    @property
    def content_checksum(self) -> str:
        """Return the Pareto-point checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed point data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}


def _dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
    """Return whether left is no worse in resources/fidelity and strictly better once."""
    left_gates = left.stratum.circuit_resources.native_two_qubit_gates
    right_gates = right.stratum.circuit_resources.native_two_qubit_gates
    left_compute = left.stratum.normalized_compute
    right_compute = right.stratum.normalized_compute
    no_worse = left_gates <= right_gates and left_compute <= right_compute and left.fidelity >= right.fidelity
    strict = left_gates < right_gates or left_compute < right_compute or left.fidelity > right.fidelity
    return no_worse and strict


def deterministic_pareto_frontier(points: Sequence[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    """Return the deterministic nondominated fidelity/resource frontier."""
    candidates = tuple(points)
    if not candidates or not all(isinstance(point, ParetoPoint) for point in candidates):
        msg = "points must contain at least one ParetoPoint."
        raise TypeError(msg)
    identifiers = tuple(point.stratum.stratum_id for point in candidates)
    if len(identifiers) != len(set(identifiers)):
        msg = "Pareto points must have unique stratum identifiers."
        raise ValueError(msg)
    frontier = [point for point in candidates if not any(_dominates(other, point) for other in candidates)]
    return tuple(
        sorted(
            frontier,
            key=lambda point: (
                point.stratum.circuit_resources.native_two_qubit_gates,
                point.stratum.normalized_compute,
                -point.fidelity,
                point.stratum.stratum_id,
                point.content_checksum,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class PairedBlockIdentity:
    """Exact target/seed/noise/resource block used for paired comparisons."""

    target_instance_id: str
    target_manifest_checksum: str
    target_spec_checksum: str
    optimization_block_id: str
    optimization_seed: int
    test_noise_id: str
    test_protocol_checksum: str
    resource_stratum_id: str
    schema_version: str = field(default=WP20_PAIRED_BLOCK_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate every identity-bearing pairing coordinate."""
        object.__setattr__(self, "target_instance_id", require_string(self.target_instance_id, "target_instance_id"))
        object.__setattr__(
            self,
            "target_manifest_checksum",
            require_checksum(self.target_manifest_checksum, "target_manifest_checksum"),
        )
        object.__setattr__(
            self,
            "target_spec_checksum",
            require_checksum(self.target_spec_checksum, "target_spec_checksum"),
        )
        object.__setattr__(
            self,
            "optimization_block_id",
            require_slug(self.optimization_block_id, "optimization_block_id"),
        )
        object.__setattr__(self, "optimization_seed", require_int(self.optimization_seed, "optimization_seed"))
        object.__setattr__(self, "test_noise_id", require_slug(self.test_noise_id, "test_noise_id"))
        object.__setattr__(
            self,
            "test_protocol_checksum",
            require_checksum(self.test_protocol_checksum, "test_protocol_checksum"),
        )
        object.__setattr__(self, "resource_stratum_id", require_slug(self.resource_stratum_id, "resource_stratum_id"))

    def _content_dict(self) -> dict[str, object]:
        """Return all pairing coordinates."""
        return {
            "schema_version": self.schema_version,
            "target_instance_id": self.target_instance_id,
            "target_manifest_checksum": self.target_manifest_checksum,
            "target_spec_checksum": self.target_spec_checksum,
            "optimization_block_id": self.optimization_block_id,
            "optimization_seed": self.optimization_seed,
            "test_noise_id": self.test_noise_id,
            "test_protocol_checksum": self.test_protocol_checksum,
            "resource_stratum_id": self.resource_stratum_id,
        }

    @property
    def content_checksum(self) -> str:
        """Return the paired-block checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed paired-block data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> PairedBlockIdentity:
        """Decode a sealed paired block."""
        expected = frozenset({
            "schema_version",
            "target_instance_id",
            "target_manifest_checksum",
            "target_spec_checksum",
            "optimization_block_id",
            "optimization_seed",
            "test_noise_id",
            "test_protocol_checksum",
            "resource_stratum_id",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 paired block")
        if mapping["schema_version"] != WP20_PAIRED_BLOCK_SCHEMA_VERSION:
            msg = f"schema_version must be {WP20_PAIRED_BLOCK_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        block = cls(
            target_instance_id=cast("str", mapping["target_instance_id"]),
            target_manifest_checksum=cast("str", mapping["target_manifest_checksum"]),
            target_spec_checksum=cast("str", mapping["target_spec_checksum"]),
            optimization_block_id=cast("str", mapping["optimization_block_id"]),
            optimization_seed=cast("int", mapping["optimization_seed"]),
            test_noise_id=cast("str", mapping["test_noise_id"]),
            test_protocol_checksum=cast("str", mapping["test_protocol_checksum"]),
            resource_stratum_id=cast("str", mapping["resource_stratum_id"]),
        )
        if mapping["content_checksum"] != block.content_checksum:
            msg = "Paired-block checksum changed during normalization."
            raise ValueError(msg)
        return block


@dataclass(frozen=True, slots=True)
class TrainingRandomnessStageEvidence:
    """One complete stage schedule and its source execution/map identities."""

    stage: TrainingStageConfig
    execution_checksum: str
    training_ensemble_checksums: tuple[str, ...]
    checkpoint_validation_ensemble_checksums: tuple[str, ...] = ()
    schema_version: str = field(default=WP20_TRAINING_RANDOMNESS_STAGE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Derive the exact expected map count from the sealed stage policy."""
        if not isinstance(self.stage, TrainingStageConfig):
            msg = "stage must be a TrainingStageConfig."
            raise TypeError(msg)
        object.__setattr__(
            self,
            "execution_checksum",
            require_checksum(self.execution_checksum, "execution_checksum"),
        )
        ensembles = _require_checksum_tuple(
            self.training_ensemble_checksums,
            "training_ensemble_checksums",
        )
        if self.stage.trajectory_count == 0:
            expected_count = 0
            if self.stage.training_seed is not None or self.stage.sampling_policy != "none":
                msg = "Noiseless stage randomness must use no training seed and sampling_policy='none'."
                raise ValueError(msg)
        else:
            if self.stage.training_seed is None:
                msg = "Noisy stage randomness requires its resolved training seed."
                raise ValueError(msg)
            if self.stage.sampling_policy == "crn_fixed":
                expected_count = 1
            elif self.stage.sampling_policy == "resampled":
                expected_count = self.stage.iteration_budget
            elif self.stage.sampling_policy == "crn_refresh":
                assert self.stage.crn_refresh_interval is not None
                expected_count = math.ceil(self.stage.iteration_budget / self.stage.crn_refresh_interval)
            else:
                msg = "A noisy stage requires a fixed, refreshed, or resampled map schedule."
                raise ValueError(msg)
        if len(ensembles) != expected_count:
            msg = "Training ensemble evidence is incomplete for the sealed stage sampling schedule."
            raise ValueError(msg)
        object.__setattr__(self, "training_ensemble_checksums", ensembles)
        validation_ensembles = _require_checksum_tuple(
            self.checkpoint_validation_ensemble_checksums,
            "checkpoint_validation_ensemble_checksums",
        )
        if not self.stage.checkpoint_validation.enabled:
            expected_validation_count = 0
        else:
            validation_calls = 1 + math.ceil(
                self.stage.iteration_budget / cast("int", self.stage.checkpoint_validation.cadence)
            )
            policy = self.stage.checkpoint_validation.sampling_policy
            if policy == "crn_fixed":
                expected_validation_count = 1
            elif policy == "resampled":
                expected_validation_count = validation_calls
            elif policy == "crn_refresh":
                interval = cast("int", self.stage.checkpoint_validation.ensemble_refresh_interval)
                expected_validation_count = math.ceil(validation_calls / interval)
            else:
                msg = "Enabled checkpoint validation requires a sampled fixed-map policy."
                raise ValueError(msg)
        if len(validation_ensembles) != expected_validation_count:
            msg = "Checkpoint-validation ensemble evidence is incomplete for the sealed stage schedule."
            raise ValueError(msg)
        object.__setattr__(self, "checkpoint_validation_ensemble_checksums", validation_ensembles)

    @property
    def training_seed(self) -> int | None:
        """Return the resolved root seed for this stage's trajectory maps."""
        return self.stage.training_seed

    @property
    def checkpoint_validation_seed(self) -> int | None:
        """Return the resolved checkpoint-selection trajectory root seed."""
        return self.stage.checkpoint_validation.seed

    def _content_dict(self) -> dict[str, object]:
        """Return complete stage-level randomness provenance."""
        return {
            "schema_version": self.schema_version,
            "stage": self.stage.to_dict(),
            "execution_checksum": self.execution_checksum,
            "training_ensemble_checksums": list(self.training_ensemble_checksums),
            "checkpoint_validation_ensemble_checksums": list(self.checkpoint_validation_ensemble_checksums),
        }

    @property
    def content_checksum(self) -> str:
        """Return the complete stage-randomness checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return strict sealed stage-randomness evidence."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> TrainingRandomnessStageEvidence:
        """Decode and verify stage-randomness evidence."""
        expected = frozenset({
            "schema_version",
            "stage",
            "execution_checksum",
            "training_ensemble_checksums",
            "checkpoint_validation_ensemble_checksums",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 stage training randomness")
        if mapping["schema_version"] != WP20_TRAINING_RANDOMNESS_STAGE_SCHEMA_VERSION:
            msg = f"schema_version must be {WP20_TRAINING_RANDOMNESS_STAGE_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        evidence = cls(
            stage=TrainingStageConfig.from_dict(mapping["stage"]),
            execution_checksum=cast("str", mapping["execution_checksum"]),
            training_ensemble_checksums=cast("tuple[str, ...]", mapping["training_ensemble_checksums"]),
            checkpoint_validation_ensemble_checksums=cast(
                "tuple[str, ...]",
                mapping["checkpoint_validation_ensemble_checksums"],
            ),
        )
        if mapping["content_checksum"] != evidence.content_checksum:
            msg = "Stage training-randomness checksum changed during normalization."
            raise ValueError(msg)
        return evidence


@dataclass(frozen=True, slots=True)
class OperatorGrowthRandomnessEvidence:
    """Complete target-bound noisy operator-growth randomness evidence."""

    result: OperatorGrowthResult
    schema_version: str = field(default=WP20_OPERATOR_GROWTH_RANDOMNESS_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require a completed promotion-eligible concrete noisy result."""
        from .operator_growth import OperatorGrowthResult  # noqa: PLC0415

        if not isinstance(self.result, OperatorGrowthResult):
            msg = "result must be an OperatorGrowthResult."
            raise TypeError(msg)
        result = self.result
        if (
            result.status != "completed"
            or result.execution_mode != "noisy_training"
            or not result.promotion_eligible
            or result.training_provenance is None
            or result.evaluator_binding is None
        ):
            msg = "Operator-growth randomness requires completed promotion-eligible target-bound noisy evidence."
            raise ValueError(msg)

    @property
    def method_id(self) -> str:
        """Return the exact operator-growth method identity."""
        return self.result.method_id

    @property
    def target_instance_id(self) -> str:
        """Return the evaluator-bound target instance identity."""
        assert self.result.evaluator_binding is not None
        return self.result.evaluator_binding.target_instance_id

    @property
    def target_spec_checksum(self) -> str:
        """Return the evaluator-bound target specification checksum."""
        assert self.result.evaluator_binding is not None
        return self.result.evaluator_binding.target_instance_spec_checksum

    @property
    def target_manifest_checksum(self) -> str:
        """Return the evaluator-bound target-manifest checksum."""
        assert self.result.evaluator_binding is not None
        return self.result.evaluator_binding.target_manifest_checksum

    @property
    def optimization_block_id(self) -> str:
        """Return the exact paired optimization-block identity."""
        assert self.result.training_provenance is not None
        return self.result.training_provenance.optimization_block_id

    @property
    def optimization_seed(self) -> int:
        """Return the exact paired outer optimization seed."""
        assert self.result.training_provenance is not None
        return self.result.training_provenance.optimization_seed

    @property
    def resource_stratum_id(self) -> str:
        """Return the exact paired resource-stratum identity."""
        assert self.result.training_provenance is not None
        return self.result.training_provenance.resource_stratum_id

    @property
    def training_seed(self) -> int:
        """Return the exact common-trajectory root seed."""
        assert self.result.training_provenance is not None
        return self.result.training_provenance.trajectory_seed

    @property
    def training_ensemble_checksum(self) -> str:
        """Return the exact common-trajectory ensemble identity."""
        assert self.result.training_provenance is not None
        return self.result.training_provenance.trajectory_ensemble_checksum

    @property
    def source_execution_checksum(self) -> str:
        """Return the checksum of the embedded complete execution result."""
        return self.result.content_checksum

    @property
    def training_configuration_checksum(self) -> str:
        """Derive the exact target, optimizer, objective, and noise configuration."""
        assert self.result.pool is not None
        assert self.result.growth_spec is not None
        assert self.result.objective_binding is not None
        assert self.result.evaluator_binding is not None
        return canonical_checksum({
            "identity_version": "yaqs.state_preparation.phase2.operator_growth_training_configuration.v1",
            "method_id": self.result.method_id,
            "pool": self.result.pool.to_dict(),
            "growth_spec": self.result.growth_spec.to_dict(),
            "objective_binding": self.result.objective_binding.to_dict(),
            "evaluator_binding": self.result.evaluator_binding.to_dict(),
        })

    @property
    def training_id(self) -> str:
        """Derive a stable training identity from the exact bound configuration."""
        checksum = canonical_checksum({
            "identity_version": "yaqs.state_preparation.phase2.operator_growth_training_identity.v1",
            "target_instance_id": self.target_instance_id,
            "target_spec_checksum": self.target_spec_checksum,
            "target_manifest_checksum": self.target_manifest_checksum,
            "method_id": self.method_id,
            "training_configuration_checksum": self.training_configuration_checksum,
        })
        return "phase2_operator_growth_training_" + checksum.removeprefix("sha256:")

    def _content_dict(self) -> dict[str, object]:
        """Return the full result plus every mechanically derived identity."""
        return {
            "schema_version": self.schema_version,
            "operator_growth_result": self.result.to_dict(),
            "source_execution_checksum": self.source_execution_checksum,
            "method_id": self.method_id,
            "training_id": self.training_id,
            "training_configuration_checksum": self.training_configuration_checksum,
            "target_instance_id": self.target_instance_id,
            "target_spec_checksum": self.target_spec_checksum,
            "target_manifest_checksum": self.target_manifest_checksum,
            "optimization_block_id": self.optimization_block_id,
            "optimization_seed": self.optimization_seed,
            "resource_stratum_id": self.resource_stratum_id,
            "training_seed": self.training_seed,
            "training_ensemble_checksum": self.training_ensemble_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Return the checksum of the complete operator-growth randomness evidence."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed operator-growth randomness evidence."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> OperatorGrowthRandomnessEvidence:
        """Decode and verify complete embedded operator-growth evidence."""
        from .operator_growth import OperatorGrowthResult  # noqa: PLC0415

        expected = frozenset({
            "schema_version",
            "operator_growth_result",
            "source_execution_checksum",
            "method_id",
            "training_id",
            "training_configuration_checksum",
            "target_instance_id",
            "target_spec_checksum",
            "target_manifest_checksum",
            "optimization_block_id",
            "optimization_seed",
            "resource_stratum_id",
            "training_seed",
            "training_ensemble_checksum",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 operator-growth randomness")
        if mapping["schema_version"] != WP20_OPERATOR_GROWTH_RANDOMNESS_SCHEMA_VERSION:
            msg = f"schema_version must be {WP20_OPERATOR_GROWTH_RANDOMNESS_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        evidence = cls(result=OperatorGrowthResult.from_dict(mapping["operator_growth_result"]))
        aliases = {
            "source_execution_checksum": evidence.source_execution_checksum,
            "method_id": evidence.method_id,
            "training_id": evidence.training_id,
            "training_configuration_checksum": evidence.training_configuration_checksum,
            "target_instance_id": evidence.target_instance_id,
            "target_spec_checksum": evidence.target_spec_checksum,
            "target_manifest_checksum": evidence.target_manifest_checksum,
            "optimization_block_id": evidence.optimization_block_id,
            "optimization_seed": evidence.optimization_seed,
            "resource_stratum_id": evidence.resource_stratum_id,
            "training_seed": evidence.training_seed,
            "training_ensemble_checksum": evidence.training_ensemble_checksum,
        }
        if any(mapping[name] != value for name, value in aliases.items()):
            msg = "Operator-growth randomness aliases are not derived from the embedded complete result."
            raise ValueError(msg)
        if mapping["content_checksum"] != evidence.content_checksum:
            msg = "Operator-growth randomness checksum changed during normalization."
            raise ValueError(msg)
        return evidence


@dataclass(frozen=True, slots=True)
class TrainingRandomnessRecord:
    """Method-wide training streams derived from every validated stage execution."""

    paired_block_checksum: str
    method_id: str
    training_id: str
    pipeline_configuration_checksum: str
    stages: tuple[TrainingRandomnessStageEvidence, ...]
    operator_growth_evidence: OperatorGrowthRandomnessEvidence | None = None
    schema_version: str = field(default=WP20_TRAINING_RANDOMNESS_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate method identity and a complete contiguous stage sequence."""
        object.__setattr__(
            self,
            "paired_block_checksum",
            require_checksum(self.paired_block_checksum, "paired_block_checksum"),
        )
        method = require_slug(self.method_id, "method_id")
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "training_id", require_string(self.training_id, "training_id"))
        object.__setattr__(
            self,
            "pipeline_configuration_checksum",
            require_checksum(self.pipeline_configuration_checksum, "pipeline_configuration_checksum"),
        )
        stages = tuple(self.stages)
        operator_evidence = self.operator_growth_evidence
        if bool(stages) == (operator_evidence is not None):
            msg = "Training randomness requires exactly one of staged execution or operator-growth evidence."
            raise ValueError(msg)
        if stages:
            if any(not isinstance(stage, TrainingRandomnessStageEvidence) for stage in stages):
                msg = "stages must contain every TrainingRandomnessStageEvidence value."
                raise TypeError(msg)
            if tuple(item.stage.stage_index for item in stages) != tuple(range(len(stages))):
                msg = "Training-randomness stages must be a complete contiguous pipeline sequence."
                raise ValueError(msg)
            if len({item.execution_checksum for item in stages}) != len(stages):
                msg = "Every training stage must bind a distinct source execution."
                raise ValueError(msg)
            all_ensembles = tuple(
                checksum
                for item in stages
                for checksum in (
                    *item.training_ensemble_checksums,
                    *item.checkpoint_validation_ensemble_checksums,
                )
            )
            if len(all_ensembles) != len(set(all_ensembles)):
                msg = "A method cannot reuse one sampled training ensemble across stages."
                raise ValueError(msg)
            training_seeds = tuple(item.training_seed for item in stages if item.training_seed is not None)
            if len(training_seeds) != len(set(training_seeds)):
                msg = "A method cannot reuse one training root seed across stages."
                raise ValueError(msg)
            noise_active = any(item.stage.trajectory_count > 0 for item in stages)
            if not noise_active and method not in _KNOWN_NOISELESS_TRAINING_METHOD_IDS:
                msg = f"Method {method!r} is not a registered noiseless-training control."
                raise ValueError(msg)
            if method in _KNOWN_NOISELESS_TRAINING_METHOD_IDS and noise_active:
                msg = f"Known noiseless-training method {method!r} cannot declare training noise."
                raise ValueError(msg)
        else:
            if not isinstance(operator_evidence, OperatorGrowthRandomnessEvidence):
                msg = "operator_growth_evidence must be OperatorGrowthRandomnessEvidence."
                raise TypeError(msg)
            if (
                method != operator_evidence.method_id
                or self.training_id != operator_evidence.training_id
                or self.pipeline_configuration_checksum != operator_evidence.training_configuration_checksum
            ):
                msg = "Operator-growth record identities must be derived from its complete evidence."
                raise ValueError(msg)
        object.__setattr__(self, "stages", stages)

    @property
    def training_noise_active(self) -> bool:
        """Return whether any stage samples training noise."""
        return self.operator_growth_evidence is not None or any(item.stage.trajectory_count > 0 for item in self.stages)

    @property
    def training_seeds(self) -> tuple[int, ...]:
        """Return one root seed per noisy stage, retaining stage order."""
        if self.operator_growth_evidence is not None:
            return (self.operator_growth_evidence.training_seed,)
        return tuple(item.training_seed for item in self.stages if item.training_seed is not None)

    @property
    def initialization_seeds(self) -> tuple[int, ...]:
        """Return every resolved stochastic parameter-transfer seed."""
        return tuple(
            item.stage.initialization_seed for item in self.stages if item.stage.initialization_seed is not None
        )

    @property
    def optimizer_seeds(self) -> tuple[int, ...]:
        """Return every resolved optimizer-ordering or perturbation seed."""
        return tuple(item.stage.optimizer_seed for item in self.stages if item.stage.optimizer_seed is not None)

    @property
    def checkpoint_validation_seeds(self) -> tuple[int, ...]:
        """Return every resolved checkpoint-selection trajectory seed."""
        return tuple(
            item.checkpoint_validation_seed for item in self.stages if item.checkpoint_validation_seed is not None
        )

    @property
    def training_ensemble_checksums(self) -> tuple[str, ...]:
        """Return every sampled training ensemble in stage/schedule order."""
        if self.operator_growth_evidence is not None:
            return (self.operator_growth_evidence.training_ensemble_checksum,)
        return tuple(checksum for item in self.stages for checksum in item.training_ensemble_checksums)

    @property
    def checkpoint_validation_ensemble_checksums(self) -> tuple[str, ...]:
        """Return every checkpoint-selection map ensemble in stage order."""
        return tuple(checksum for item in self.stages for checksum in item.checkpoint_validation_ensemble_checksums)

    @property
    def source_execution_checksums(self) -> tuple[str, ...]:
        """Return every source stage-execution checksum in pipeline order."""
        if self.operator_growth_evidence is not None:
            return (self.operator_growth_evidence.source_execution_checksum,)
        return tuple(item.execution_checksum for item in self.stages)

    @classmethod
    def from_stage_evidence(
        cls,
        block: PairedBlockIdentity,
        pipeline: TrainingPipelineConfig,
        evidence: Sequence[object],
    ) -> TrainingRandomnessRecord:
        """Derive method-wide randomness from a complete validated pipeline execution."""
        from .artifacts import StageExecutionEvidence  # noqa: PLC0415

        if not isinstance(block, PairedBlockIdentity):
            msg = "block must be a PairedBlockIdentity."
            raise TypeError(msg)
        if not isinstance(pipeline, TrainingPipelineConfig):
            msg = "pipeline must be a TrainingPipelineConfig."
            raise TypeError(msg)
        expected_block = (
            pipeline.target_instance_id,
            pipeline.target_population_manifest_checksum,
            pipeline.target_instance_spec_checksum,
            pipeline.optimization_block_id,
            pipeline.optimization_seed,
            pipeline.template.resource_stratum_id,
        )
        actual_block = (
            block.target_instance_id,
            block.target_manifest_checksum,
            block.target_spec_checksum,
            block.optimization_block_id,
            block.optimization_seed,
            block.resource_stratum_id,
        )
        if expected_block != actual_block:
            msg = "Resolved pipeline does not belong to the supplied paired block."
            raise ValueError(msg)
        rows = tuple(evidence)
        if len(rows) != len(pipeline.stages) or any(
            not isinstance(item, StageExecutionEvidence) or item.stage != pipeline.stages[index]
            for index, item in enumerate(rows)
        ):
            msg = "evidence must contain every validated pipeline stage in exact order."
            raise ValueError(msg)
        stages: list[TrainingRandomnessStageEvidence] = []
        for item in cast("tuple[StageExecutionEvidence, ...]", rows):
            execution_checksum = item.training_summary.get("adapter_execution_checksum")
            if execution_checksum is None:
                execution_checksum = item.training_summary.get("competitor_execution_checksum")
            if execution_checksum is None:
                msg = "Every randomized training stage requires a sealed optimizer execution checksum."
                raise ValueError(msg)
            stages.append(
                TrainingRandomnessStageEvidence(
                    stage=item.stage,
                    execution_checksum=cast("str", execution_checksum),
                    training_ensemble_checksums=tuple(
                        ensemble.content_checksum for ensemble in item.training_ensembles
                    ),
                    checkpoint_validation_ensemble_checksums=tuple(
                        ensemble.content_checksum for ensemble in item.checkpoint_validation_ensembles
                    ),
                )
            )
        return cls(
            paired_block_checksum=block.content_checksum,
            method_id=pipeline.method_id,
            training_id=pipeline.training_id,
            pipeline_configuration_checksum=pipeline.configuration_checksum,
            stages=tuple(stages),
        )

    @classmethod
    def from_operator_growth_result(
        cls,
        block: PairedBlockIdentity,
        result: OperatorGrowthResult,
    ) -> TrainingRandomnessRecord:
        """Derive one randomness record from a complete noisy operator-growth result."""
        if not isinstance(block, PairedBlockIdentity):
            msg = "block must be a PairedBlockIdentity."
            raise TypeError(msg)
        evidence = OperatorGrowthRandomnessEvidence(result=result)
        expected_target = (
            block.target_instance_id,
            block.target_spec_checksum,
            block.target_manifest_checksum,
            block.optimization_block_id,
            block.optimization_seed,
            block.resource_stratum_id,
        )
        actual_target = (
            evidence.target_instance_id,
            evidence.target_spec_checksum,
            evidence.target_manifest_checksum,
            evidence.optimization_block_id,
            evidence.optimization_seed,
            evidence.resource_stratum_id,
        )
        if actual_target != expected_target:
            msg = "Operator-growth result does not belong to the paired block's target and optimization coordinates."
            raise ValueError(msg)
        return cls(
            paired_block_checksum=block.content_checksum,
            method_id=evidence.method_id,
            training_id=evidence.training_id,
            pipeline_configuration_checksum=evidence.training_configuration_checksum,
            stages=(),
            operator_growth_evidence=evidence,
        )

    def _content_dict(self) -> dict[str, object]:
        """Return complete and mechanically derived method randomness."""
        return {
            "schema_version": self.schema_version,
            "paired_block_checksum": self.paired_block_checksum,
            "method_id": self.method_id,
            "training_id": self.training_id,
            "pipeline_configuration_checksum": self.pipeline_configuration_checksum,
            "stages": [stage.to_dict() for stage in self.stages],
            "operator_growth_evidence": (
                None if self.operator_growth_evidence is None else self.operator_growth_evidence.to_dict()
            ),
            "training_noise_active": self.training_noise_active,
            "initialization_seeds": list(self.initialization_seeds),
            "optimizer_seeds": list(self.optimizer_seeds),
            "training_seeds": list(self.training_seeds),
            "checkpoint_validation_seeds": list(self.checkpoint_validation_seeds),
            "training_ensemble_checksums": list(self.training_ensemble_checksums),
            "checkpoint_validation_ensemble_checksums": list(self.checkpoint_validation_ensemble_checksums),
            "source_execution_checksums": list(self.source_execution_checksums),
        }

    @property
    def content_checksum(self) -> str:
        """Return the randomness-record checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed randomness data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> TrainingRandomnessRecord:
        """Decode and verify a complete method randomness record."""
        expected = frozenset({
            "schema_version",
            "paired_block_checksum",
            "method_id",
            "training_id",
            "pipeline_configuration_checksum",
            "stages",
            "operator_growth_evidence",
            "training_noise_active",
            "initialization_seeds",
            "optimizer_seeds",
            "training_seeds",
            "checkpoint_validation_seeds",
            "training_ensemble_checksums",
            "checkpoint_validation_ensemble_checksums",
            "source_execution_checksums",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 training randomness")
        if mapping["schema_version"] != WP20_TRAINING_RANDOMNESS_SCHEMA_VERSION:
            msg = f"schema_version must be {WP20_TRAINING_RANDOMNESS_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        stage_values = mapping["stages"]
        if isinstance(stage_values, (str, bytes)) or not isinstance(stage_values, Sequence):
            msg = "stages must be a sequence."
            raise TypeError(msg)
        operator_value = mapping["operator_growth_evidence"]
        record = cls(
            paired_block_checksum=cast("str", mapping["paired_block_checksum"]),
            method_id=cast("str", mapping["method_id"]),
            training_id=cast("str", mapping["training_id"]),
            pipeline_configuration_checksum=cast("str", mapping["pipeline_configuration_checksum"]),
            stages=tuple(TrainingRandomnessStageEvidence.from_dict(item) for item in stage_values),
            operator_growth_evidence=(
                None if operator_value is None else OperatorGrowthRandomnessEvidence.from_dict(operator_value)
            ),
        )
        aliases = {
            "training_noise_active": record.training_noise_active,
            "initialization_seeds": record.initialization_seeds,
            "optimizer_seeds": record.optimizer_seeds,
            "training_seeds": record.training_seeds,
            "checkpoint_validation_seeds": record.checkpoint_validation_seeds,
            "training_ensemble_checksums": record.training_ensemble_checksums,
            "checkpoint_validation_ensemble_checksums": record.checkpoint_validation_ensemble_checksums,
            "source_execution_checksums": record.source_execution_checksums,
        }
        if any(mapping[name] != value for name, value in aliases.items()):
            msg = "Training-randomness aliases are not derived from complete stage evidence."
            raise ValueError(msg)
        if mapping["content_checksum"] != record.content_checksum:
            msg = "Training-randomness checksum changed during normalization."
            raise ValueError(msg)
        return record


def validate_training_randomness_isolation(
    block: PairedBlockIdentity,
    records: Sequence[TrainingRandomnessRecord],
) -> tuple[TrainingRandomnessRecord, ...]:
    """Validate a paired method block without sharing training randomness.

    The validator rejects reuse of initialization, optimizer, training, or
    checkpoint-validation seeds and sampled training or checkpoint ensembles
    across methods in the same comparison block.
    """
    if not isinstance(block, PairedBlockIdentity):
        msg = "block must be a PairedBlockIdentity."
        raise TypeError(msg)
    members = tuple(records)
    if len(members) < 2 or not all(isinstance(record, TrainingRandomnessRecord) for record in members):
        msg = "records must contain at least two TrainingRandomnessRecord values."
        raise TypeError(msg)
    if any(record.paired_block_checksum != block.content_checksum for record in members):
        msg = "Every training-randomness record must belong to the supplied paired block."
        raise ValueError(msg)
    methods = tuple(record.method_id for record in members)
    training_ids = tuple(record.training_id for record in members)
    if len(methods) != len(set(methods)):
        msg = "A paired block must contain each method exactly once."
        raise ValueError(msg)
    if len(training_ids) != len(set(training_ids)):
        msg = "Paired methods must have distinct training identities."
        raise ValueError(msg)
    observed_initialization_seeds: set[int] = set()
    observed_optimizer_seeds: set[int] = set()
    observed_training_seeds: set[int] = set()
    observed_validation_seeds: set[int] = set()
    observed_ensembles: set[str] = set()
    for record in members:
        seed_roles = (
            ("initialization", record.initialization_seeds, observed_initialization_seeds),
            ("optimizer", record.optimizer_seeds, observed_optimizer_seeds),
            ("training trajectory", record.training_seeds, observed_training_seeds),
            ("checkpoint validation", record.checkpoint_validation_seeds, observed_validation_seeds),
        )
        for role, seeds, observed in seed_roles:
            reused_seeds = observed.intersection(seeds)
            if reused_seeds:
                msg = f"Paired methods share {role} seeds {sorted(reused_seeds)!r}."
                raise ValueError(msg)
        ensembles = (*record.training_ensemble_checksums, *record.checkpoint_validation_ensemble_checksums)
        reused_ensembles = observed_ensembles.intersection(ensembles)
        if reused_ensembles:
            msg = "Paired methods share sampled training or checkpoint-map ensembles."
            raise ValueError(msg)
        for _, seeds, observed in seed_roles:
            observed.update(seeds)
        observed_ensembles.update(ensembles)
    return tuple(sorted(members, key=lambda record: (record.method_id, record.training_id, record.content_checksum)))


def _test_protocol_payload(config: PipelineEvaluationConfig) -> dict[str, object]:
    """Return the complete method-independent final-test protocol."""
    return {
        "evaluation_schema_version": config.schema_version,
        "data_role": config.data_role,
        "test_noise_id": config.test_noise_id,
        "noise_definition_version": config.noise_definition_version,
        "noise_strength_scale": config.noise_strength_scale,
        "tjm_dt": config.tjm_dt,
        "evaluation_seed": config.evaluation_seed,
        "evaluation_seed_domain": config.evaluation_seed_domain,
        "repetition": config.repetition,
        "trajectory_budget": config.trajectory_budget,
        "evaluation_policy": config.evaluation_policy,
        "confidence_level": config.confidence_level,
        "confidence_interval_method": config.confidence_interval_method,
        "sidecar_storage_policy": config.sidecar_storage_policy,
        "max_bond_dimension": config.max_bond_dimension,
        "svd_threshold": config.svd_threshold,
        "truncation_mode": config.truncation_mode,
        "min_bond_dimension": config.min_bond_dimension,
    }


@dataclass(frozen=True, slots=True)
class EventLevelTestCoupling:
    """Explicit decision to couple or separate final-test event streams."""

    paired_block: PairedBlockIdentity
    left_method_id: str
    right_method_id: str
    left_training_id: str
    right_training_id: str
    left_evaluation_id: str
    right_evaluation_id: str
    left_evaluation: PipelineEvaluationConfig
    right_evaluation: PipelineEvaluationConfig
    left_resources: CircuitResourceMetrics
    right_resources: CircuitResourceMetrics
    left_test_noise_id: str
    right_test_noise_id: str
    mode: Literal["event_level_coupled", "independent"]
    reason: str
    left_native_circuit_checksum: str
    right_native_circuit_checksum: str
    left_resource_checksum: str
    right_resource_checksum: str
    aligned_event_count: int
    alignment_checksum: str | None
    aligned_native_events: tuple[NativeEventSignature, ...] = ()
    schema_version: str = field(default=WP20_TEST_COUPLING_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate mode-dependent evidence."""
        if not isinstance(self.paired_block, PairedBlockIdentity):
            msg = "paired_block must be a PairedBlockIdentity."
            raise TypeError(msg)
        left_method = require_slug(self.left_method_id, "left_method_id")
        right_method = require_slug(self.right_method_id, "right_method_id")
        if left_method == right_method:
            msg = "Event-level coupling must compare two distinct method identities."
            raise ValueError(msg)
        left_training = require_string(self.left_training_id, "left_training_id")
        right_training = require_string(self.right_training_id, "right_training_id")
        left_evaluation = require_string(self.left_evaluation_id, "left_evaluation_id")
        right_evaluation = require_string(self.right_evaluation_id, "right_evaluation_id")
        if left_training == right_training or left_evaluation == right_evaluation:
            msg = "Compared methods require distinct training and evaluation identities."
            raise ValueError(msg)
        if not isinstance(self.left_evaluation, PipelineEvaluationConfig) or not isinstance(
            self.right_evaluation,
            PipelineEvaluationConfig,
        ):
            msg = "left_evaluation and right_evaluation must be PipelineEvaluationConfig values."
            raise TypeError(msg)
        if not isinstance(self.left_resources, CircuitResourceMetrics) or not isinstance(
            self.right_resources,
            CircuitResourceMetrics,
        ):
            msg = "left_resources and right_resources must be CircuitResourceMetrics values."
            raise TypeError(msg)
        if (
            left_training != self.left_evaluation.pipeline_training_id
            or right_training != self.right_evaluation.pipeline_training_id
            or left_evaluation != self.left_evaluation.evaluation_row_id
            or right_evaluation != self.right_evaluation.evaluation_row_id
        ):
            msg = "Training/evaluation aliases must be derived from the complete evaluation configurations."
            raise ValueError(msg)
        left_noise = require_slug(self.left_test_noise_id, "left_test_noise_id")
        right_noise = require_slug(self.right_test_noise_id, "right_test_noise_id")
        if (
            left_noise != self.paired_block.test_noise_id
            or right_noise != self.paired_block.test_noise_id
            or left_noise != self.left_evaluation.test_noise_id
            or right_noise != self.right_evaluation.test_noise_id
        ):
            msg = "Both declared test-noise identities must match the paired block."
            raise ValueError(msg)
        left_protocol = _test_protocol_payload(self.left_evaluation)
        right_protocol = _test_protocol_payload(self.right_evaluation)
        if left_protocol != right_protocol:
            msg = "Event-level coupling requires identical complete final-test protocols."
            raise ValueError(msg)
        if canonical_checksum(left_protocol) != self.paired_block.test_protocol_checksum:
            msg = "Final-test protocol does not match the paired-block commitment."
            raise ValueError(msg)
        if self.mode not in {"event_level_coupled", "independent"}:
            msg = "mode must be 'event_level_coupled' or 'independent'."
            raise ValueError(msg)
        allowed_reasons = {
            "identical_full_native_event_signatures",
            "no_stable_native_events",
            "native_event_count_mismatch",
            "native_event_signature_mismatch",
        }
        reason = require_slug(self.reason, "reason")
        if reason not in allowed_reasons:
            msg = "reason is not a supported WP20 coupling decision."
            raise ValueError(msg)
        left = require_checksum(self.left_native_circuit_checksum, "left_native_circuit_checksum")
        right = require_checksum(self.right_native_circuit_checksum, "right_native_circuit_checksum")
        left_resource = require_checksum(self.left_resource_checksum, "left_resource_checksum")
        right_resource = require_checksum(self.right_resource_checksum, "right_resource_checksum")
        if (
            left != self.left_resources.native_circuit_checksum
            or right != self.right_resources.native_circuit_checksum
            or left_resource != self.left_resources.content_checksum
            or right_resource != self.right_resources.content_checksum
        ):
            msg = "Circuit/resource aliases must be derived from complete resource evidence."
            raise ValueError(msg)
        count = require_int(self.aligned_event_count, "aligned_event_count")
        alignment = (
            None
            if self.alignment_checksum is None
            else require_checksum(
                self.alignment_checksum,
                "alignment_checksum",
            )
        )
        coupled = self.mode == "event_level_coupled"
        events = tuple(self.aligned_native_events)
        if any(not isinstance(event, NativeEventSignature) for event in events):
            msg = "aligned_native_events must contain NativeEventSignature values."
            raise TypeError(msg)
        expected_reason: str
        if not self.left_resources.native_events or not self.right_resources.native_events:
            expected_reason = "no_stable_native_events"
        elif len(self.left_resources.native_events) != len(self.right_resources.native_events):
            expected_reason = "native_event_count_mismatch"
        elif self.left_resources.native_events != self.right_resources.native_events:
            expected_reason = "native_event_signature_mismatch"
        else:
            expected_reason = "identical_full_native_event_signatures"
        if reason != expected_reason:
            msg = "Coupling reason is not mechanically derived from the complete native circuits."
            raise ValueError(msg)
        if coupled != (reason == "identical_full_native_event_signatures"):
            msg = "Only identical full native event signatures authorize coupling."
            raise ValueError(msg)
        if coupled != (alignment is not None):
            msg = "alignment_checksum is required exactly for event-level coupling."
            raise ValueError(msg)
        if coupled:
            if not events or events != self.left_resources.native_events or count != len(events):
                msg = "Event-level coupling requires every aligned native event as evidence."
                raise ValueError(msg)
            expected_native_checksum = canonical_checksum({
                "compiler_policy_id": PRIMARY_COMPILER_POLICY_ID,
                "connectivity_id": PRIMARY_CONNECTIVITY,
                "routing_policy_id": PRIMARY_ROUTING_POLICY_ID,
                "events": [event.to_dict() for event in events],
            })
            if left != expected_native_checksum or right != expected_native_checksum:
                msg = "Event-level coupling native checksums are not derived from aligned event evidence."
                raise ValueError(msg)
            expected_alignment = canonical_checksum({
                "coupling_version": WP20_TEST_COUPLING_SCHEMA_VERSION,
                "paired_block_checksum": self.paired_block.content_checksum,
                "left_method_id": left_method,
                "right_method_id": right_method,
                "left_training_id": left_training,
                "right_training_id": right_training,
                "left_evaluation_id": left_evaluation,
                "right_evaluation_id": right_evaluation,
                "left_resource_checksum": left_resource,
                "right_resource_checksum": right_resource,
                "test_noise_id": self.paired_block.test_noise_id,
                "native_events": [event.to_dict() for event in events],
            })
            if alignment != expected_alignment:
                msg = "Event-level coupling alignment checksum is not derived from its exact events."
                raise ValueError(msg)
        elif events:
            msg = "Independent test streams cannot claim aligned native-event evidence."
            raise ValueError(msg)
        elif count != min(len(self.left_resources.native_events), len(self.right_resources.native_events)):
            msg = "Independent aligned_event_count must equal the shared native-event prefix length."
            raise ValueError(msg)
        object.__setattr__(self, "left_method_id", left_method)
        object.__setattr__(self, "right_method_id", right_method)
        object.__setattr__(self, "left_training_id", left_training)
        object.__setattr__(self, "right_training_id", right_training)
        object.__setattr__(self, "left_evaluation_id", left_evaluation)
        object.__setattr__(self, "right_evaluation_id", right_evaluation)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "left_test_noise_id", left_noise)
        object.__setattr__(self, "right_test_noise_id", right_noise)
        object.__setattr__(self, "left_native_circuit_checksum", left)
        object.__setattr__(self, "right_native_circuit_checksum", right)
        object.__setattr__(self, "left_resource_checksum", left_resource)
        object.__setattr__(self, "right_resource_checksum", right_resource)
        object.__setattr__(self, "aligned_event_count", count)
        object.__setattr__(self, "alignment_checksum", alignment)
        object.__setattr__(self, "aligned_native_events", events)

    def _content_dict(self) -> dict[str, object]:
        """Return complete coupling evidence."""
        return {
            "schema_version": self.schema_version,
            "paired_block": self.paired_block.to_dict(),
            "left_method_id": self.left_method_id,
            "right_method_id": self.right_method_id,
            "left_training_id": self.left_training_id,
            "right_training_id": self.right_training_id,
            "left_evaluation_id": self.left_evaluation_id,
            "right_evaluation_id": self.right_evaluation_id,
            "left_evaluation": self.left_evaluation.to_dict(),
            "right_evaluation": self.right_evaluation.to_dict(),
            "left_resources": self.left_resources.to_dict(),
            "right_resources": self.right_resources.to_dict(),
            "left_test_noise_id": self.left_test_noise_id,
            "right_test_noise_id": self.right_test_noise_id,
            "mode": self.mode,
            "reason": self.reason,
            "left_native_circuit_checksum": self.left_native_circuit_checksum,
            "right_native_circuit_checksum": self.right_native_circuit_checksum,
            "left_resource_checksum": self.left_resource_checksum,
            "right_resource_checksum": self.right_resource_checksum,
            "aligned_event_count": self.aligned_event_count,
            "alignment_checksum": self.alignment_checksum,
            "aligned_native_events": [event.to_dict() for event in self.aligned_native_events],
        }

    @property
    def content_checksum(self) -> str:
        """Return the coupling-decision checksum."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return sealed coupling-decision data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> EventLevelTestCoupling:
        """Decode a sealed coupling decision."""
        expected = frozenset({
            "schema_version",
            "paired_block",
            "left_method_id",
            "right_method_id",
            "left_training_id",
            "right_training_id",
            "left_evaluation_id",
            "right_evaluation_id",
            "left_evaluation",
            "right_evaluation",
            "left_resources",
            "right_resources",
            "left_test_noise_id",
            "right_test_noise_id",
            "mode",
            "reason",
            "left_native_circuit_checksum",
            "right_native_circuit_checksum",
            "left_resource_checksum",
            "right_resource_checksum",
            "aligned_event_count",
            "alignment_checksum",
            "aligned_native_events",
            "content_checksum",
        })
        mapping = verify_sealed_mapping(data, expected_keys=expected, name="WP20 test coupling")
        if mapping["schema_version"] != WP20_TEST_COUPLING_SCHEMA_VERSION:
            msg = f"schema_version must be {WP20_TEST_COUPLING_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        decision = cls(
            paired_block=PairedBlockIdentity.from_dict(mapping["paired_block"]),
            left_method_id=cast("str", mapping["left_method_id"]),
            right_method_id=cast("str", mapping["right_method_id"]),
            left_training_id=cast("str", mapping["left_training_id"]),
            right_training_id=cast("str", mapping["right_training_id"]),
            left_evaluation_id=cast("str", mapping["left_evaluation_id"]),
            right_evaluation_id=cast("str", mapping["right_evaluation_id"]),
            left_evaluation=PipelineEvaluationConfig.from_dict(mapping["left_evaluation"]),
            right_evaluation=PipelineEvaluationConfig.from_dict(mapping["right_evaluation"]),
            left_resources=CircuitResourceMetrics.from_dict(mapping["left_resources"]),
            right_resources=CircuitResourceMetrics.from_dict(mapping["right_resources"]),
            left_test_noise_id=cast("str", mapping["left_test_noise_id"]),
            right_test_noise_id=cast("str", mapping["right_test_noise_id"]),
            mode=cast("Literal['event_level_coupled', 'independent']", mapping["mode"]),
            reason=cast("str", mapping["reason"]),
            left_native_circuit_checksum=cast("str", mapping["left_native_circuit_checksum"]),
            right_native_circuit_checksum=cast("str", mapping["right_native_circuit_checksum"]),
            left_resource_checksum=cast("str", mapping["left_resource_checksum"]),
            right_resource_checksum=cast("str", mapping["right_resource_checksum"]),
            aligned_event_count=cast("int", mapping["aligned_event_count"]),
            alignment_checksum=cast("str | None", mapping["alignment_checksum"]),
            aligned_native_events=tuple(
                NativeEventSignature.from_dict(item)
                for item in cast("Sequence[object]", mapping["aligned_native_events"])
            ),
        )
        if mapping["content_checksum"] != decision.content_checksum:
            msg = "Test-coupling checksum changed during normalization."
            raise ValueError(msg)
        return decision


def decide_event_level_test_coupling(
    block: PairedBlockIdentity,
    left: CircuitResourceMetrics,
    right: CircuitResourceMetrics,
    *,
    left_method_id: str,
    right_method_id: str,
    left_evaluation: PipelineEvaluationConfig,
    right_evaluation: PipelineEvaluationConfig,
) -> EventLevelTestCoupling:
    """Couple only complete, identical final-test protocols on aligned events."""
    if not isinstance(block, PairedBlockIdentity):
        msg = "block must be a PairedBlockIdentity."
        raise TypeError(msg)
    if not isinstance(left, CircuitResourceMetrics) or not isinstance(right, CircuitResourceMetrics):
        msg = "left and right must be CircuitResourceMetrics."
        raise TypeError(msg)
    if not isinstance(left_evaluation, PipelineEvaluationConfig) or not isinstance(
        right_evaluation,
        PipelineEvaluationConfig,
    ):
        msg = "left_evaluation and right_evaluation must be PipelineEvaluationConfig values."
        raise TypeError(msg)
    left_method = require_slug(left_method_id, "left_method_id")
    right_method = require_slug(right_method_id, "right_method_id")
    left_training = left_evaluation.pipeline_training_id
    right_training = right_evaluation.pipeline_training_id
    left_evaluation_id = left_evaluation.evaluation_row_id
    right_evaluation_id = right_evaluation.evaluation_row_id
    left_noise = left_evaluation.test_noise_id
    right_noise = right_evaluation.test_noise_id
    shared = min(len(left.native_events), len(right.native_events))
    if not left.native_events or not right.native_events:
        reason = "no_stable_native_events"
    elif len(left.native_events) != len(right.native_events):
        reason = "native_event_count_mismatch"
    elif left.native_events != right.native_events:
        reason = "native_event_signature_mismatch"
    else:
        alignment = canonical_checksum({
            "coupling_version": WP20_TEST_COUPLING_SCHEMA_VERSION,
            "paired_block_checksum": block.content_checksum,
            "left_method_id": left_method,
            "right_method_id": right_method,
            "left_training_id": left_training,
            "right_training_id": right_training,
            "left_evaluation_id": left_evaluation_id,
            "right_evaluation_id": right_evaluation_id,
            "left_resource_checksum": left.content_checksum,
            "right_resource_checksum": right.content_checksum,
            "test_noise_id": block.test_noise_id,
            "native_events": [event.to_dict() for event in left.native_events],
        })
        return EventLevelTestCoupling(
            paired_block=block,
            left_method_id=left_method,
            right_method_id=right_method,
            left_training_id=left_training,
            right_training_id=right_training,
            left_evaluation_id=left_evaluation_id,
            right_evaluation_id=right_evaluation_id,
            left_evaluation=left_evaluation,
            right_evaluation=right_evaluation,
            left_resources=left,
            right_resources=right,
            left_test_noise_id=left_noise,
            right_test_noise_id=right_noise,
            mode="event_level_coupled",
            reason="identical_full_native_event_signatures",
            left_native_circuit_checksum=left.native_circuit_checksum,
            right_native_circuit_checksum=right.native_circuit_checksum,
            left_resource_checksum=left.content_checksum,
            right_resource_checksum=right.content_checksum,
            aligned_event_count=len(left.native_events),
            alignment_checksum=alignment,
            aligned_native_events=left.native_events,
        )
    return EventLevelTestCoupling(
        paired_block=block,
        left_method_id=left_method,
        right_method_id=right_method,
        left_training_id=left_training,
        right_training_id=right_training,
        left_evaluation_id=left_evaluation_id,
        right_evaluation_id=right_evaluation_id,
        left_evaluation=left_evaluation,
        right_evaluation=right_evaluation,
        left_resources=left,
        right_resources=right,
        left_test_noise_id=left_noise,
        right_test_noise_id=right_noise,
        mode="independent",
        reason=reason,
        left_native_circuit_checksum=left.native_circuit_checksum,
        right_native_circuit_checksum=right.native_circuit_checksum,
        left_resource_checksum=left.content_checksum,
        right_resource_checksum=right.content_checksum,
        aligned_event_count=shared,
        alignment_checksum=None,
        aligned_native_events=(),
    )


__all__ = [
    "DEFAULT_NORMALIZED_COMPUTE_POLICY",
    "WP20_CIRCUIT_RESOURCES_SCHEMA_VERSION",
    "WP20_NATIVE_EVENT_SCHEMA_VERSION",
    "WP20_NORMALIZED_COMPUTE_POLICY_ID",
    "WP20_NORMALIZED_COMPUTE_POLICY_SCHEMA_VERSION",
    "WP20_OPERATOR_GROWTH_RANDOMNESS_SCHEMA_VERSION",
    "WP20_PAIRED_BLOCK_SCHEMA_VERSION",
    "WP20_PARETO_POINT_SCHEMA_VERSION",
    "WP20_RESOURCE_BUDGET_SCHEMA_VERSION",
    "WP20_RESOURCE_SELECTION_SCHEMA_VERSION",
    "WP20_RESOURCE_STRATUM_SCHEMA_VERSION",
    "WP20_TEST_COUPLING_SCHEMA_VERSION",
    "WP20_TRAINING_RANDOMNESS_SCHEMA_VERSION",
    "WP20_TRAINING_RANDOMNESS_STAGE_SCHEMA_VERSION",
    "WP20_WORK_LEDGER_SCHEMA_VERSION",
    "CircuitResourceMetrics",
    "EventLevelTestCoupling",
    "InfeasibleResourceBudget",
    "LogicalEventSignature",
    "NativeEventSignature",
    "NormalizedComputePolicy",
    "OperatorGrowthRandomnessEvidence",
    "PairedBlockIdentity",
    "ParetoPoint",
    "ReachableResourceStratum",
    "ResourceBudget",
    "ResourceSelectionOutcome",
    "SelectedResourceStratum",
    "TrainingRandomnessRecord",
    "TrainingRandomnessStageEvidence",
    "WP20WorkLedger",
    "decide_event_level_test_coupling",
    "deterministic_pareto_frontier",
    "measure_circuit_resources",
    "resource_selection_outcome_from_dict",
    "select_reachable_resource_stratum",
    "validate_training_randomness_isolation",
    "wp20_work_from_noisy_krotov",
]
