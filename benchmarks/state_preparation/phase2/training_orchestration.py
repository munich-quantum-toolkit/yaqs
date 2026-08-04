# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Safe WP22 planning, dispatch, and resumable job orchestration.

The runner in this module is deliberately scientific-artifact agnostic: an
execution callback receives an already sealed job, while this layer owns the
Cartesian fan-out, role-isolated paths, collision checks, and durable outcome
ledger.  Method-specific callbacks remain responsible for publishing their
WP18 stage and evaluation artifacts.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from filelock import FileLock

from benchmarks.state_preparation.constants import BALLARIN_NOISE_ID
from benchmarks.state_preparation.reporting import atomic_write_bytes

from .binding_catalog import ExecutableScopedBinding
from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .execution_bindings import PILOT_METHOD_IDS, SCREEN_METHOD_IDS, SMOKE_METHOD_IDS
from .layerwise_bmpd import resolve_layerwise_bmpd_crn_legacy_v1_pipeline
from .pipeline import TrainingPipelineConfig, TrainingPipelineTemplate
from .protocol import (
    FinalConfigurationExecutionManifest,
    FinalConfigurationExecutionRef,
    FinalConfirmationSeal,
    ScreeningCell,
    ScreeningManifest,
    validate_final_configuration_execution_manifest,
)
from .screening_design import WP22CandidateConfiguration
from .targets import TargetInstanceSpec, TargetPopulationManifest, verify_screening_target_population
from .training_schedules import (
    CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID,
    CONFIRMATORY_OPTIMIZATION_SEED_POLICY_ID,
    PILOT_FRESH_EVALUATION_SEED_POLICY_ID,
    PILOT_OPTIMIZATION_SEED_POLICY_ID,
    SMOKE_FRESH_EVALUATION_SEED_POLICY_ID,
    SMOKE_OPTIMIZATION_SEED_POLICY_ID,
    ExecutionSeedPolicySuite,
    TrainingStrategySchedule,
)
from .validation import (
    require_bool,
    require_checksum,
    require_exact_keys,
    require_float,
    require_int,
    require_nonempty_text,
    require_relative_path,
    require_slug,
)

if TYPE_CHECKING:
    from .execution_context import ConfirmationExecutionContext, TrainingExecutionContext

TRAINING_JOB_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_training_job.v2"
TRAINING_RUN_PLAN_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_training_run_plan.v2"
TRAINING_JOB_OUTCOME_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_training_job_outcome.v1"
CONFIRM_EXECUTION_REQUEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22_confirm_execution_request.v2"

PILOT_OPTIMIZATION_SEED_COUNT = 5

TRAINING_PRESETS = (
    "training-smoke",
    "historical-layerwise-reproduction",
    "paper-pilot",
    "paper-screen",
    "paper-confirm",
)
RUNNABLE_DATA_ROLES = (
    "development",
    "screening_selection",
    "secondary_benchmark",
    "confirmatory",
)
JOB_RESULT_NAME = "wp22_job_outcome.json"
JOB_ATTEMPTS_DIRECTORY_NAME = "wp22_job_outcomes"

_ATTEMPT_FILE_PATTERN = re.compile(r"^attempt_(?P<attempt>[0-9]{8})\.json$")

_PRIMARY_NOISE_KEYS = frozenset({
    "noise_id",
    "definition_version",
    "strength_scale",
    "tjm_dt",
    "training_placement",
    "test_placement",
})
_PRIMARY_RESOURCE_KEYS = frozenset({
    "metric",
    "cap_per_chain_edge",
    "normalized_compute_cap",
    "reachable_stratum_manifest_checksum",
})
_CONFIRM_REQUEST_KEYS = frozenset({
    "schema_version",
    "final_confirmation_seal_checksum",
    "preregistration_checksum",
    "promotion_decision_checksum",
    "execution_source_checksum",
    "analysis_source_manifest_checksum",
    "analysis_template_checksum",
    "configuration_execution_manifest_checksum",
    "hyperparameters_checksum",
    "implementation_checksum",
    "scoped_binding_checksum",
    "executable_binding_checksum",
    "sample_size_design_checksum",
    "failure_policy_checksum",
    "fixed_test_trajectory_count",
    "primary_noise_condition",
    "primary_resource_budget",
    "method_id",
    "configuration_checksum",
    "target_manifest_checksum",
    "target_instance_id",
    "target_spec_checksum",
    "family_id",
    "stratum_id",
    "qubit_count",
    "optimization_block_id",
    "optimization_seed_index",
    "optimization_seed",
    "evaluation_seed",
    "content_checksum",
})

_JOB_KEYS = frozenset({
    "schema_version",
    "job_id",
    "preset",
    "method_id",
    "implementation_kind",
    "candidate_configuration_checksum",
    "implementation_checksum",
    "strategy_schedule_checksum",
    "strategy_schedule",
    "confirm_execution_request",
    "target_manifest_checksum",
    "target_instance_id",
    "target_spec_checksum",
    "family_id",
    "stratum_id",
    "qubit_count",
    "data_role",
    "optimization_block_id",
    "optimization_seed",
    "evaluation_seed",
    "output_path",
    "execution_profile_checksum",
    "scoped_binding_checksum",
    "executable_binding_checksum",
    "evaluation_policy_checksum",
    "target_configuration_checksum",
    "source_fingerprint_checksum",
    "scheduled_execution_program_checksum",
    "content_checksum",
})
_PLAN_KEYS = frozenset({
    "schema_version",
    "plan_id",
    "preset",
    "preregistration_checksum",
    "target_manifest_checksums",
    "screening_manifest_checksum",
    "final_confirmation_seal_checksum",
    "execution_source_checksum",
    "execution_profile_checksum",
    "scoped_binding_checksums",
    "executable_binding_checksums",
    "implementation_checksums",
    "evaluation_policy_checksums",
    "target_configuration_checksums",
    "source_fingerprint_checksums",
    "scheduled_execution_program_checksums",
    "sample_size_design_checksum",
    "jobs",
    "content_checksum",
})
_OUTCOME_KEYS = frozenset({
    "schema_version",
    "job_checksum",
    "status",
    "result_artifact_checksum",
    "exception_type",
    "message",
    "attempt",
    "content_checksum",
})


def derive_pilot_optimization_seeds(preregistration_checksum: str, seed_count: int) -> tuple[int, ...]:
    """Derive the exact preregistered pilot optimization-seed schedule.

    Args:
        preregistration_checksum: Complete immutable preregistration identity.
        seed_count: Count read from the preregistered role-allocation policy.

    Returns:
        Five deterministic, domain-separated unsigned seeds.

    Raises:
        ValueError: If the preregistration does not declare the WP22 count.
    """
    root = require_checksum(preregistration_checksum, "preregistration_checksum")
    count = require_int(seed_count, "seed_count", minimum=1)
    if count != PILOT_OPTIMIZATION_SEED_COUNT:
        msg = f"The preregistered WP22 pilot must use exactly {PILOT_OPTIMIZATION_SEED_COUNT} optimization seeds."
        raise ValueError(msg)
    suite = ExecutionSeedPolicySuite.frozen()
    return tuple(
        suite.derive(
            PILOT_OPTIMIZATION_SEED_POLICY_ID,
            {"preregistration_checksum": root, "seed_index": index},
        )
        for index in range(count)
    )


def derive_confirmatory_optimization_seed(
    final_seal_checksum: str,
    target_spec_checksum: str,
    seed_index: int,
) -> int:
    """Derive one final-seal-bound confirmatory optimization seed.

    Returns:
        The domain-separated unsigned seed.
    """
    return ExecutionSeedPolicySuite.frozen().derive(
        CONFIRMATORY_OPTIMIZATION_SEED_POLICY_ID,
        {
            "final_seal_checksum": require_checksum(final_seal_checksum, "final_seal_checksum"),
            "target_instance_spec_checksum": require_checksum(target_spec_checksum, "target_spec_checksum"),
            "seed_index": require_int(seed_index, "seed_index", minimum=0),
        },
    )


def derive_confirmatory_evaluation_seed(
    final_seal_checksum: str,
    target_spec_checksum: str,
    seed_index: int,
    configuration_checksum: str,
) -> int:
    """Derive one independent final-seal-bound confirmatory test seed.

    Returns:
        The domain-separated unsigned seed.
    """
    return ExecutionSeedPolicySuite.frozen().derive(
        CONFIRMATORY_FRESH_EVALUATION_SEED_POLICY_ID,
        {
            "final_seal_checksum": require_checksum(final_seal_checksum, "final_seal_checksum"),
            "target_instance_spec_checksum": require_checksum(target_spec_checksum, "target_spec_checksum"),
            "seed_index": require_int(seed_index, "seed_index", minimum=0),
            "configuration_checksum": require_checksum(configuration_checksum, "configuration_checksum"),
        },
    )


@dataclass(frozen=True, slots=True)
class ConfirmExecutionRequest:
    """Seal-complete request for one authorized confirmatory method cell."""

    final_confirmation_seal_checksum: str
    preregistration_checksum: str
    promotion_decision_checksum: str
    execution_source_checksum: str
    analysis_source_manifest_checksum: str
    analysis_template_checksum: str
    configuration_execution_manifest_checksum: str
    hyperparameters_checksum: str
    implementation_checksum: str
    scoped_binding_checksum: str
    executable_binding_checksum: str
    sample_size_design_checksum: str
    failure_policy_checksum: str
    fixed_test_trajectory_count: int
    primary_noise_condition: Mapping[str, object]
    primary_resource_budget: Mapping[str, object]
    method_id: str
    configuration_checksum: str
    target_manifest_checksum: str
    target_instance_id: str
    target_spec_checksum: str
    family_id: str
    stratum_id: str
    qubit_count: int
    optimization_block_id: str
    optimization_seed_index: int
    optimization_seed: int
    evaluation_seed: int
    schema_version: str = field(default=CONFIRM_EXECUTION_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate every final-seal, method, target, and seed identity."""
        for name in (
            "final_confirmation_seal_checksum",
            "preregistration_checksum",
            "promotion_decision_checksum",
            "execution_source_checksum",
            "analysis_source_manifest_checksum",
            "analysis_template_checksum",
            "configuration_execution_manifest_checksum",
            "hyperparameters_checksum",
            "implementation_checksum",
            "scoped_binding_checksum",
            "executable_binding_checksum",
            "sample_size_design_checksum",
            "failure_policy_checksum",
            "configuration_checksum",
            "target_manifest_checksum",
            "target_spec_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        for name in (
            "method_id",
            "target_instance_id",
            "family_id",
            "stratum_id",
            "optimization_block_id",
        ):
            object.__setattr__(self, name, require_slug(getattr(self, name), name))
        object.__setattr__(
            self,
            "fixed_test_trajectory_count",
            require_int(self.fixed_test_trajectory_count, "fixed_test_trajectory_count", minimum=2),
        )
        object.__setattr__(self, "qubit_count", require_int(self.qubit_count, "qubit_count", minimum=2))
        object.__setattr__(
            self,
            "optimization_seed_index",
            require_int(self.optimization_seed_index, "optimization_seed_index", minimum=0),
        )
        object.__setattr__(
            self,
            "optimization_seed",
            require_int(self.optimization_seed, "optimization_seed", minimum=0),
        )
        object.__setattr__(self, "evaluation_seed", require_int(self.evaluation_seed, "evaluation_seed", minimum=0))
        noise = freeze_json_mapping(self.primary_noise_condition, "primary_noise_condition")
        require_exact_keys(noise, _PRIMARY_NOISE_KEYS, "primary_noise_condition")
        require_slug(noise["noise_id"], "primary_noise_condition.noise_id")
        require_float(noise["strength_scale"], "primary_noise_condition.strength_scale", minimum=0.0)
        resource = freeze_json_mapping(self.primary_resource_budget, "primary_resource_budget")
        require_exact_keys(resource, _PRIMARY_RESOURCE_KEYS, "primary_resource_budget")
        require_float(resource["cap_per_chain_edge"], "primary_resource_budget.cap_per_chain_edge", minimum=0.0)
        require_float(resource["normalized_compute_cap"], "primary_resource_budget.normalized_compute_cap", minimum=0.0)
        require_checksum(
            resource["reachable_stratum_manifest_checksum"],
            "primary_resource_budget.reachable_stratum_manifest_checksum",
        )
        object.__setattr__(self, "primary_noise_condition", noise)
        object.__setattr__(self, "primary_resource_budget", resource)

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered execution fields."""
        return {
            "schema_version": self.schema_version,
            "final_confirmation_seal_checksum": self.final_confirmation_seal_checksum,
            "preregistration_checksum": self.preregistration_checksum,
            "promotion_decision_checksum": self.promotion_decision_checksum,
            "execution_source_checksum": self.execution_source_checksum,
            "analysis_source_manifest_checksum": self.analysis_source_manifest_checksum,
            "analysis_template_checksum": self.analysis_template_checksum,
            "configuration_execution_manifest_checksum": self.configuration_execution_manifest_checksum,
            "hyperparameters_checksum": self.hyperparameters_checksum,
            "implementation_checksum": self.implementation_checksum,
            "scoped_binding_checksum": self.scoped_binding_checksum,
            "executable_binding_checksum": self.executable_binding_checksum,
            "sample_size_design_checksum": self.sample_size_design_checksum,
            "failure_policy_checksum": self.failure_policy_checksum,
            "fixed_test_trajectory_count": self.fixed_test_trajectory_count,
            "primary_noise_condition": thaw_json_mapping(self.primary_noise_condition),
            "primary_resource_budget": thaw_json_mapping(self.primary_resource_budget),
            "method_id": self.method_id,
            "configuration_checksum": self.configuration_checksum,
            "target_manifest_checksum": self.target_manifest_checksum,
            "target_instance_id": self.target_instance_id,
            "target_spec_checksum": self.target_spec_checksum,
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "qubit_count": self.qubit_count,
            "optimization_block_id": self.optimization_block_id,
            "optimization_seed_index": self.optimization_seed_index,
            "optimization_seed": self.optimization_seed,
            "evaluation_seed": self.evaluation_seed,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete confirmatory execution request."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native request data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed request JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ConfirmExecutionRequest:
        """Decode and verify one confirmatory execution request.

        Returns:
            The verified typed request.

        Raises:
            ValueError: If the schema or normalized checksum is invalid.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_CONFIRM_REQUEST_KEYS, name="confirm execution request")
        if mapping["schema_version"] != CONFIRM_EXECUTION_REQUEST_SCHEMA_VERSION:
            msg = "Confirm execution request uses an unsupported schema version."
            raise ValueError(msg)
        request = cls(
            final_confirmation_seal_checksum=cast("str", mapping["final_confirmation_seal_checksum"]),
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            promotion_decision_checksum=cast("str", mapping["promotion_decision_checksum"]),
            execution_source_checksum=cast("str", mapping["execution_source_checksum"]),
            analysis_source_manifest_checksum=cast("str", mapping["analysis_source_manifest_checksum"]),
            analysis_template_checksum=cast("str", mapping["analysis_template_checksum"]),
            configuration_execution_manifest_checksum=cast(
                "str",
                mapping["configuration_execution_manifest_checksum"],
            ),
            hyperparameters_checksum=cast("str", mapping["hyperparameters_checksum"]),
            implementation_checksum=cast("str", mapping["implementation_checksum"]),
            scoped_binding_checksum=cast("str", mapping["scoped_binding_checksum"]),
            executable_binding_checksum=cast("str", mapping["executable_binding_checksum"]),
            sample_size_design_checksum=cast("str", mapping["sample_size_design_checksum"]),
            failure_policy_checksum=cast("str", mapping["failure_policy_checksum"]),
            fixed_test_trajectory_count=cast("int", mapping["fixed_test_trajectory_count"]),
            primary_noise_condition=cast("Mapping[str, object]", mapping["primary_noise_condition"]),
            primary_resource_budget=cast("Mapping[str, object]", mapping["primary_resource_budget"]),
            method_id=cast("str", mapping["method_id"]),
            configuration_checksum=cast("str", mapping["configuration_checksum"]),
            target_manifest_checksum=cast("str", mapping["target_manifest_checksum"]),
            target_instance_id=cast("str", mapping["target_instance_id"]),
            target_spec_checksum=cast("str", mapping["target_spec_checksum"]),
            family_id=cast("str", mapping["family_id"]),
            stratum_id=cast("str", mapping["stratum_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            optimization_block_id=cast("str", mapping["optimization_block_id"]),
            optimization_seed_index=cast("int", mapping["optimization_seed_index"]),
            optimization_seed=cast("int", mapping["optimization_seed"]),
            evaluation_seed=cast("int", mapping["evaluation_seed"]),
        )
        if mapping["content_checksum"] != request.content_checksum:
            msg = "Confirm execution request checksum changed during normalization."
            raise ValueError(msg)
        return request

    @classmethod
    def from_json(cls, payload: str) -> ConfirmExecutionRequest:
        """Decode canonical confirmatory request JSON.

        Returns:
            The verified typed request.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def confirmatory_evaluation_policy_checksum(request: ConfirmExecutionRequest) -> str:
    """Bind the frozen confirmatory evaluator to one exact sealed request.

    Returns:
        The shared request-, seed-, count-, noise-, and role-bound evaluation
        policy checksum used by synthetic and real confirmation custody.

    Raises:
        TypeError: If ``request`` is not a confirm execution request.
    """
    if not isinstance(request, ConfirmExecutionRequest):
        msg = "request must be a ConfirmExecutionRequest."
        raise TypeError(msg)
    return canonical_checksum({
        "purpose": "confirmatory_fresh_evaluation",
        "request_checksum": request.content_checksum,
        "evaluation_seed": request.evaluation_seed,
        "fixed_test_trajectory_count": request.fixed_test_trajectory_count,
        "primary_noise_condition": dict(request.primary_noise_condition),
        "data_role": "confirmatory",
        "seed_domain": "confirmatory_test",
    })


@dataclass(frozen=True, slots=True)
class ConfirmExecutionContext:
    """Precomputed immutable index for efficient per-cell authentication."""

    seal: FinalConfirmationSeal
    target_manifest: TargetPopulationManifest
    configuration_execution_manifest: FinalConfigurationExecutionManifest
    final_seal_checksum: str = field(init=False)
    target_manifest_checksum: str = field(init=False)
    targets_by_id: Mapping[str, TargetInstanceSpec] = field(init=False, repr=False)
    methods_by_configuration: Mapping[str, str] = field(init=False, repr=False)
    execution_by_configuration: Mapping[str, FinalConfigurationExecutionRef] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate source schemas and cache their expensive roots once.

        Raises:
            TypeError: If an input has the wrong typed schema.
            ValueError: If the target manifest differs from the final seal.
        """
        if not isinstance(self.seal, FinalConfirmationSeal):
            msg = "seal must be a FinalConfirmationSeal."
            raise TypeError(msg)
        if not isinstance(self.target_manifest, TargetPopulationManifest):
            msg = "target_manifest must be a TargetPopulationManifest."
            raise TypeError(msg)
        validate_final_configuration_execution_manifest(self.seal, self.configuration_execution_manifest)
        seal_checksum = self.seal.content_checksum
        manifest_checksum = self.target_manifest.content_checksum
        if manifest_checksum != self.seal.confirmatory_target_manifest_checksum:
            msg = "Target manifest differs from the final seal."
            raise ValueError(msg)
        targets = {item.target_instance_id: item for item in self.target_manifest.instances}
        methods = {
            self.seal.promoted_configuration_checksum: self.seal.promoted_method_id,
            **{item.configuration_checksum: item.method_id for item in self.seal.comparators},
        }
        execution = {item.configuration_checksum: item for item in self.configuration_execution_manifest.entries}
        if {checksum: item.method_id for checksum, item in execution.items()} != methods:
            msg = "Final execution manifest method/configuration identities differ from the final seal."
            raise ValueError(msg)
        object.__setattr__(self, "final_seal_checksum", seal_checksum)
        object.__setattr__(self, "target_manifest_checksum", manifest_checksum)
        object.__setattr__(self, "targets_by_id", MappingProxyType(targets))
        object.__setattr__(self, "methods_by_configuration", MappingProxyType(methods))
        object.__setattr__(self, "execution_by_configuration", MappingProxyType(execution))


def build_confirm_execution_context(
    seal: FinalConfirmationSeal,
    target_manifest: TargetPopulationManifest,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
) -> ConfirmExecutionContext:
    """Build one cached context for a complete confirmatory result batch.

    Returns:
        The immutable O(1)-lookup authentication context.
    """
    return ConfirmExecutionContext(
        seal=seal,
        target_manifest=target_manifest,
        configuration_execution_manifest=configuration_execution_manifest,
    )


def validate_confirm_execution_request(
    request: ConfirmExecutionRequest,
    context_or_seal: ConfirmExecutionContext | FinalConfirmationSeal,
    target_manifest: TargetPopulationManifest | None = None,
    configuration_execution_manifest: FinalConfigurationExecutionManifest | None = None,
) -> None:
    """Authenticate one request without rebuilding the full confirmatory plan.

    Raises:
        TypeError: If an input does not have the required typed schema.
        ValueError: If any seal, method, target, resource, or seed field differs.
    """
    if not isinstance(request, ConfirmExecutionRequest):
        msg = "request must be a ConfirmExecutionRequest."
        raise TypeError(msg)
    if isinstance(context_or_seal, ConfirmExecutionContext):
        if target_manifest is not None or configuration_execution_manifest is not None:
            msg = "Target and execution manifests must be omitted with a ConfirmExecutionContext."
            raise ValueError(msg)
        context = context_or_seal
    else:
        if target_manifest is None or configuration_execution_manifest is None:
            msg = "Target and final-configuration execution manifests are required with a FinalConfirmationSeal."
            raise TypeError(msg)
        context = build_confirm_execution_context(
            context_or_seal,
            target_manifest,
            configuration_execution_manifest,
        )
    seal = context.seal
    target = context.targets_by_id.get(request.target_instance_id)
    if target is None:
        msg = "Confirm execution target is absent from the sealed manifest."
        raise ValueError(msg)
    method_id = context.methods_by_configuration.get(request.configuration_checksum)
    if method_id is None:
        msg = "Confirm execution configuration is absent from the final seal."
        raise ValueError(msg)
    execution = context.execution_by_configuration[request.configuration_checksum]
    if request.optimization_seed_index >= seal.optimization_seed_count:
        msg = "Confirm execution seed index exceeds the final-seal count."
        raise ValueError(msg)
    block = f"confirm_{target.target_instance_id}_seed_index_{request.optimization_seed_index}"
    expected = ConfirmExecutionRequest(
        final_confirmation_seal_checksum=context.final_seal_checksum,
        preregistration_checksum=seal.preregistration_checksum,
        promotion_decision_checksum=seal.promotion_decision_checksum,
        execution_source_checksum=seal.execution_source_checksum,
        analysis_source_manifest_checksum=seal.analysis_source_manifest_checksum,
        analysis_template_checksum=seal.analysis_template_checksum,
        configuration_execution_manifest_checksum=context.configuration_execution_manifest.content_checksum,
        hyperparameters_checksum=execution.strategy_schedule_checksum,
        implementation_checksum=execution.implementation_checksum,
        scoped_binding_checksum=execution.scoped_binding_checksum,
        executable_binding_checksum=execution.executable_binding_checksum,
        sample_size_design_checksum=seal.sample_size_design_checksum,
        failure_policy_checksum=seal.failure_policy_checksum,
        fixed_test_trajectory_count=seal.fixed_test_trajectory_count,
        primary_noise_condition=seal.primary_noise_condition,
        primary_resource_budget=seal.primary_resource_budget,
        method_id=method_id,
        configuration_checksum=request.configuration_checksum,
        target_manifest_checksum=context.target_manifest_checksum,
        target_instance_id=target.target_instance_id,
        target_spec_checksum=target.content_checksum,
        family_id=target.family_id,
        stratum_id=target.stratum_id,
        qubit_count=target.qubit_count,
        optimization_block_id=block,
        optimization_seed_index=request.optimization_seed_index,
        optimization_seed=derive_confirmatory_optimization_seed(
            context.final_seal_checksum,
            target.content_checksum,
            request.optimization_seed_index,
        ),
        evaluation_seed=derive_confirmatory_evaluation_seed(
            context.final_seal_checksum,
            target.content_checksum,
            request.optimization_seed_index,
            request.configuration_checksum,
        ),
    )
    if request != expected:
        msg = "ConfirmExecutionRequest differs from its exact final-seal cell."
        raise ValueError(msg)


def _job_sort_key(job: TrainingJob) -> tuple[object, ...]:
    """Return the canonical target, seed, and method ordering key."""
    return (
        job.data_role,
        job.family_id,
        job.stratum_id,
        job.qubit_count,
        job.target_instance_id,
        job.optimization_seed,
        job.method_id,
        job.candidate_configuration_checksum,
    )


@dataclass(frozen=True, slots=True)
class TrainingJob:
    """One target, optimization seed, candidate, and fresh evaluation cell."""

    job_id: str
    preset: str
    method_id: str
    implementation_kind: Literal[
        "phase2_pipeline",
        "operator_growth",
        "legacy_delegate",
        "sealed_configuration",
    ]
    candidate_configuration_checksum: str
    implementation_checksum: str
    strategy_schedule_checksum: str
    target_manifest_checksum: str
    target_instance_id: str
    target_spec_checksum: str
    family_id: str
    stratum_id: str
    qubit_count: int
    data_role: str
    optimization_block_id: str
    optimization_seed: int
    evaluation_seed: int
    output_path: str
    execution_profile_checksum: str | None = None
    scoped_binding_checksum: str | None = None
    executable_binding_checksum: str | None = None
    evaluation_policy_checksum: str | None = None
    target_configuration_checksum: str | None = None
    source_fingerprint_checksum: str | None = None
    scheduled_execution_program_checksum: str | None = None
    strategy_schedule: TrainingStrategySchedule | None = None
    confirm_execution_request: ConfirmExecutionRequest | None = None
    schema_version: str = field(default=TRAINING_JOB_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate identities and the role-isolated output path.

        Raises:
            TypeError: If a required typed schedule or confirm request is absent.
            ValueError: If an identity, preset, role, seed, or path is invalid.
        """
        object.__setattr__(self, "job_id", require_slug(self.job_id, "job_id"))
        if self.preset not in TRAINING_PRESETS:
            msg = f"preset must be one of {TRAINING_PRESETS!r}."
            raise ValueError(msg)
        if self.implementation_kind not in {
            "phase2_pipeline",
            "operator_growth",
            "legacy_delegate",
            "sealed_configuration",
        }:
            msg = "implementation_kind is not a registered WP22 executor family."
            raise ValueError(msg)
        for name in ("method_id", "target_instance_id", "family_id", "stratum_id", "optimization_block_id"):
            object.__setattr__(self, name, require_slug(getattr(self, name), name))
        for name in (
            "candidate_configuration_checksum",
            "implementation_checksum",
            "strategy_schedule_checksum",
            "target_manifest_checksum",
            "target_spec_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        fingerprint_names = (
            "execution_profile_checksum",
            "scoped_binding_checksum",
            "executable_binding_checksum",
            "evaluation_policy_checksum",
            "target_configuration_checksum",
            "source_fingerprint_checksum",
            "scheduled_execution_program_checksum",
        )
        populated_fingerprints = tuple(getattr(self, name) is not None for name in fingerprint_names)
        if any(populated_fingerprints) and not all(populated_fingerprints):
            msg = "WP22D job fingerprints must be either complete or entirely absent."
            raise ValueError(msg)
        for name in fingerprint_names:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_checksum(value, name))
        if self.data_role not in RUNNABLE_DATA_ROLES:
            msg = f"data_role must be one of {RUNNABLE_DATA_ROLES!r}."
            raise ValueError(msg)
        if self.preset == "paper-screen" and self.data_role != "screening_selection":
            msg = "paper-screen jobs must use the screening_selection target role."
            raise ValueError(msg)
        if self.preset == "paper-confirm" and self.data_role != "confirmatory":
            msg = "paper-confirm jobs must use the confirmatory target role."
            raise ValueError(msg)
        object.__setattr__(self, "qubit_count", require_int(self.qubit_count, "qubit_count", minimum=1))
        if self.preset == "paper-pilot" and (self.data_role, self.qubit_count) not in {
            ("development", 6),
            ("secondary_benchmark", 12),
        }:
            msg = "paper-pilot jobs must be primary-q6 development or secondary-q12 benchmark jobs."
            raise ValueError(msg)
        if self.preset == "paper-screen" and self.qubit_count != 6:
            msg = "paper-screen jobs must remain in the primary q6 population."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "optimization_seed",
            require_int(self.optimization_seed, "optimization_seed", minimum=0),
        )
        object.__setattr__(self, "evaluation_seed", require_int(self.evaluation_seed, "evaluation_seed", minimum=0))
        if self.implementation_kind in {"phase2_pipeline", "operator_growth"}:
            if not isinstance(self.strategy_schedule, TrainingStrategySchedule):
                msg = "Strategy candidate jobs require their complete typed TrainingStrategySchedule."
                raise TypeError(msg)
            if self.strategy_schedule.content_checksum != self.strategy_schedule_checksum:
                msg = "Job strategy_schedule differs from strategy_schedule_checksum."
                raise ValueError(msg)
        elif self.strategy_schedule is not None:
            msg = "Only strategy candidate jobs may embed a TrainingStrategySchedule."
            raise ValueError(msg)
        if self.preset == "paper-confirm":
            if not isinstance(self.confirm_execution_request, ConfirmExecutionRequest):
                msg = "paper-confirm jobs require a seal-complete ConfirmExecutionRequest."
                raise TypeError(msg)
            request = self.confirm_execution_request
            expected = (
                (request.method_id, self.method_id),
                (request.configuration_checksum, self.candidate_configuration_checksum),
                (request.implementation_checksum, self.implementation_checksum),
                (request.hyperparameters_checksum, self.strategy_schedule_checksum),
                (request.target_manifest_checksum, self.target_manifest_checksum),
                (request.target_instance_id, self.target_instance_id),
                (request.target_spec_checksum, self.target_spec_checksum),
                (request.family_id, self.family_id),
                (request.stratum_id, self.stratum_id),
                (request.qubit_count, self.qubit_count),
                (request.optimization_block_id, self.optimization_block_id),
                (request.optimization_seed, self.optimization_seed),
                (request.evaluation_seed, self.evaluation_seed),
            )
            if any(left != right for left, right in expected):
                msg = "ConfirmExecutionRequest does not match its enclosing TrainingJob identity."
                raise ValueError(msg)
        elif self.confirm_execution_request is not None:
            msg = "A ConfirmExecutionRequest may be attached only to paper-confirm jobs."
            raise ValueError(msg)
        path = require_relative_path(self.output_path, "output_path")
        expected_prefix = f"roles/{self.data_role}/"
        if not path.startswith(expected_prefix) or not path.endswith(self.job_id):
            msg = "output_path must be nested under its exact data role and end in job_id."
            raise ValueError(msg)
        object.__setattr__(self, "output_path", path)

    @property
    def sort_key(self) -> tuple[object, ...]:
        """Canonical target, seed, and method ordering key."""
        return _job_sort_key(self)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered job field."""
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "preset": self.preset,
            "method_id": self.method_id,
            "implementation_kind": self.implementation_kind,
            "candidate_configuration_checksum": self.candidate_configuration_checksum,
            "implementation_checksum": self.implementation_checksum,
            "strategy_schedule_checksum": self.strategy_schedule_checksum,
            "strategy_schedule": None if self.strategy_schedule is None else self.strategy_schedule.to_dict(),
            "confirm_execution_request": (
                None if self.confirm_execution_request is None else self.confirm_execution_request.to_dict()
            ),
            "target_manifest_checksum": self.target_manifest_checksum,
            "target_instance_id": self.target_instance_id,
            "target_spec_checksum": self.target_spec_checksum,
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "qubit_count": self.qubit_count,
            "data_role": self.data_role,
            "optimization_block_id": self.optimization_block_id,
            "optimization_seed": self.optimization_seed,
            "evaluation_seed": self.evaluation_seed,
            "output_path": self.output_path,
            "execution_profile_checksum": self.execution_profile_checksum,
            "scoped_binding_checksum": self.scoped_binding_checksum,
            "executable_binding_checksum": self.executable_binding_checksum,
            "evaluation_policy_checksum": self.evaluation_policy_checksum,
            "target_configuration_checksum": self.target_configuration_checksum,
            "source_fingerprint_checksum": self.source_fingerprint_checksum,
            "scheduled_execution_program_checksum": self.scheduled_execution_program_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete immutable job request."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> TrainingJob:
        """Decode and verify one training job.

        Returns:
            The verified job.

        Raises:
            ValueError: If its schema or checksum is invalid.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_JOB_KEYS, name="WP22 training job")
        if mapping["schema_version"] != TRAINING_JOB_SCHEMA_VERSION:
            msg = "WP22 training job uses an unsupported schema version."
            raise ValueError(msg)
        job = cls(
            job_id=cast("str", mapping["job_id"]),
            preset=cast("str", mapping["preset"]),
            method_id=cast("str", mapping["method_id"]),
            implementation_kind=cast(
                "Literal['phase2_pipeline', 'operator_growth', 'legacy_delegate', 'sealed_configuration']",
                mapping["implementation_kind"],
            ),
            candidate_configuration_checksum=cast("str", mapping["candidate_configuration_checksum"]),
            implementation_checksum=cast("str", mapping["implementation_checksum"]),
            strategy_schedule_checksum=cast("str", mapping["strategy_schedule_checksum"]),
            strategy_schedule=(
                None
                if mapping["strategy_schedule"] is None
                else TrainingStrategySchedule.from_dict(mapping["strategy_schedule"])
            ),
            confirm_execution_request=(
                None
                if mapping["confirm_execution_request"] is None
                else ConfirmExecutionRequest.from_dict(mapping["confirm_execution_request"])
            ),
            target_manifest_checksum=cast("str", mapping["target_manifest_checksum"]),
            target_instance_id=cast("str", mapping["target_instance_id"]),
            target_spec_checksum=cast("str", mapping["target_spec_checksum"]),
            family_id=cast("str", mapping["family_id"]),
            stratum_id=cast("str", mapping["stratum_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            data_role=cast("str", mapping["data_role"]),
            optimization_block_id=cast("str", mapping["optimization_block_id"]),
            optimization_seed=cast("int", mapping["optimization_seed"]),
            evaluation_seed=cast("int", mapping["evaluation_seed"]),
            output_path=cast("str", mapping["output_path"]),
            execution_profile_checksum=cast("str | None", mapping["execution_profile_checksum"]),
            scoped_binding_checksum=cast("str | None", mapping["scoped_binding_checksum"]),
            executable_binding_checksum=cast("str | None", mapping["executable_binding_checksum"]),
            evaluation_policy_checksum=cast("str | None", mapping["evaluation_policy_checksum"]),
            target_configuration_checksum=cast("str | None", mapping["target_configuration_checksum"]),
            source_fingerprint_checksum=cast("str | None", mapping["source_fingerprint_checksum"]),
            scheduled_execution_program_checksum=cast(
                "str | None",
                mapping["scheduled_execution_program_checksum"],
            ),
        )
        if mapping["content_checksum"] != job.content_checksum:
            msg = "WP22 training job checksum changed during normalization."
            raise ValueError(msg)
        return job


@dataclass(frozen=True, slots=True)
class TrainingRunPlan:
    """Deterministically ordered, checksum-sealed WP22 job fan-out."""

    plan_id: str
    preset: str
    preregistration_checksum: str
    target_manifest_checksums: tuple[str, ...]
    screening_manifest_checksum: str | None
    final_confirmation_seal_checksum: str | None
    execution_source_checksum: str | None
    jobs: tuple[TrainingJob, ...]
    execution_profile_checksum: str | None = None
    scoped_binding_checksums: tuple[str, ...] = ()
    executable_binding_checksums: tuple[str, ...] = ()
    implementation_checksums: tuple[str, ...] = ()
    evaluation_policy_checksums: tuple[str, ...] = ()
    target_configuration_checksums: tuple[str, ...] = ()
    source_fingerprint_checksums: tuple[str, ...] = ()
    scheduled_execution_program_checksums: tuple[str, ...] = ()
    sample_size_design_checksum: str | None = None
    schema_version: str = field(default=TRAINING_RUN_PLAN_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate preset-specific roots and canonical job ordering.

        Raises:
            TypeError: If jobs are not typed immutable records.
            ValueError: If roots, identities, or preset invariants disagree.
        """
        object.__setattr__(self, "plan_id", require_slug(self.plan_id, "plan_id"))
        if self.preset not in TRAINING_PRESETS:
            msg = f"preset must be one of {TRAINING_PRESETS!r}."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "preregistration_checksum",
            require_checksum(self.preregistration_checksum, "preregistration_checksum"),
        )
        manifests = tuple(require_checksum(item, "target_manifest_checksum") for item in self.target_manifest_checksums)
        if len(manifests) != len(set(manifests)):
            msg = "target_manifest_checksums must not contain duplicates."
            raise ValueError(msg)
        object.__setattr__(self, "target_manifest_checksums", manifests)
        for name in (
            "screening_manifest_checksum",
            "final_confirmation_seal_checksum",
            "execution_source_checksum",
            "execution_profile_checksum",
            "sample_size_design_checksum",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_checksum(value, name))
        jobs = tuple(self.jobs)
        if not jobs or not all(isinstance(job, TrainingJob) for job in jobs):
            msg = "jobs must be a nonempty tuple of TrainingJob records."
            raise TypeError(msg)
        if tuple(sorted(jobs, key=_job_sort_key)) != jobs:
            msg = "jobs must use the canonical target, seed, and method ordering."
            raise ValueError(msg)
        if len({job.job_id for job in jobs}) != len(jobs) or len({job.content_checksum for job in jobs}) != len(jobs):
            msg = "A run plan must not duplicate job identities."
            raise ValueError(msg)
        if any(job.preset != self.preset for job in jobs):
            msg = "Every job must belong to the plan preset."
            raise ValueError(msg)
        if {job.target_manifest_checksum for job in jobs} != set(manifests):
            msg = "Plan target-manifest roots differ from its jobs."
            raise ValueError(msg)
        if self.preset == "paper-screen" and self.screening_manifest_checksum is None:
            msg = "paper-screen requires its complete screening manifest checksum."
            raise ValueError(msg)
        if self.preset == "paper-confirm":
            if self.final_confirmation_seal_checksum is None or self.execution_source_checksum is None:
                msg = "paper-confirm requires both the final seal and frozen execution source."
                raise ValueError(msg)
            requests = tuple(job.confirm_execution_request for job in jobs)
            if any(request is None for request in requests):
                msg = "Every paper-confirm job requires a seal-complete execution request."
                raise ValueError(msg)
            typed_requests = cast("tuple[ConfirmExecutionRequest, ...]", requests)
            if any(
                request.final_confirmation_seal_checksum != self.final_confirmation_seal_checksum
                or request.execution_source_checksum != self.execution_source_checksum
                or request.preregistration_checksum != self.preregistration_checksum
                for request in typed_requests
            ):
                msg = "Confirm execution requests differ from the plan's sealed roots."
                raise ValueError(msg)
            invariant_fields = {
                (
                    request.analysis_source_manifest_checksum,
                    request.analysis_template_checksum,
                    request.configuration_execution_manifest_checksum,
                    request.fixed_test_trajectory_count,
                    canonical_checksum(request.primary_noise_condition),
                    canonical_checksum(request.primary_resource_budget),
                    request.sample_size_design_checksum,
                    request.failure_policy_checksum,
                )
                for request in typed_requests
            }
            if len(invariant_fields) != 1:
                msg = "Paper-confirm jobs do not share one exact final-seal execution policy."
                raise ValueError(msg)
            execution_by_configuration: dict[str, tuple[str, str, str, str]] = {}
            for request in typed_requests:
                execution_identity = (
                    request.implementation_checksum,
                    request.hyperparameters_checksum,
                    request.scoped_binding_checksum,
                    request.executable_binding_checksum,
                )
                previous = execution_by_configuration.setdefault(
                    request.configuration_checksum,
                    execution_identity,
                )
                if previous != execution_identity:
                    msg = "One confirmatory configuration cannot use multiple executable identities."
                    raise ValueError(msg)
        elif self.final_confirmation_seal_checksum is not None:
            msg = "A final confirmation seal may be attached only to paper-confirm."
            raise ValueError(msg)
        fingerprint_fields = (
            ("scoped_binding_checksums", "scoped_binding_checksum"),
            ("executable_binding_checksums", "executable_binding_checksum"),
            ("implementation_checksums", "implementation_checksum"),
            ("evaluation_policy_checksums", "evaluation_policy_checksum"),
            ("target_configuration_checksums", "target_configuration_checksum"),
            ("source_fingerprint_checksums", "source_fingerprint_checksum"),
            ("scheduled_execution_program_checksums", "scheduled_execution_program_checksum"),
        )
        bound_flags = tuple(job.execution_profile_checksum is not None for job in jobs)
        if any(bound_flags) and not all(bound_flags):
            msg = "A plan cannot mix WP22D-bound and unbound jobs."
            raise ValueError(msg)
        jobs_are_bound = all(bound_flags)
        if jobs_are_bound:
            if self.execution_profile_checksum is None or any(
                job.execution_profile_checksum != self.execution_profile_checksum for job in jobs
            ):
                msg = "Bound jobs and plan must share one exact execution-profile checksum."
                raise ValueError(msg)
            for plan_name, job_name in fingerprint_fields:
                raw_values = tuple(getattr(self, plan_name))
                values = tuple(require_checksum(value, f"{plan_name} item") for value in raw_values)
                if values != tuple(sorted(set(values))):
                    msg = f"{plan_name} must be a sorted checksum-distinct tuple."
                    raise ValueError(msg)
                expected = tuple(sorted({cast("str", getattr(job, job_name)) for job in jobs}))
                if values != expected:
                    msg = f"{plan_name} differs from the exact job fingerprint universe."
                    raise ValueError(msg)
                object.__setattr__(self, plan_name, values)
        elif self.execution_profile_checksum is not None or any(
            tuple(getattr(self, name)) for name, _ in fingerprint_fields
        ):
            msg = "An unbound plan cannot claim WP22D execution fingerprints."
            raise ValueError(msg)
        if self.preset == "paper-screen" and jobs_are_bound and self.sample_size_design_checksum is None:
            msg = "A bound paper-screen plan requires its pilot-derived sample-size design checksum."
            raise ValueError(msg)
        if self.preset != "paper-screen" and self.sample_size_design_checksum is not None:
            msg = "sample_size_design_checksum is accepted only by paper-screen before WP22 confirmation."
            raise ValueError(msg)
        object.__setattr__(self, "jobs", jobs)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered plan field."""
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "preset": self.preset,
            "preregistration_checksum": self.preregistration_checksum,
            "target_manifest_checksums": list(self.target_manifest_checksums),
            "screening_manifest_checksum": self.screening_manifest_checksum,
            "final_confirmation_seal_checksum": self.final_confirmation_seal_checksum,
            "execution_source_checksum": self.execution_source_checksum,
            "execution_profile_checksum": self.execution_profile_checksum,
            "scoped_binding_checksums": list(self.scoped_binding_checksums),
            "executable_binding_checksums": list(self.executable_binding_checksums),
            "implementation_checksums": list(self.implementation_checksums),
            "evaluation_policy_checksums": list(self.evaluation_policy_checksums),
            "target_configuration_checksums": list(self.target_configuration_checksums),
            "source_fingerprint_checksums": list(self.source_fingerprint_checksums),
            "scheduled_execution_program_checksums": list(self.scheduled_execution_program_checksums),
            "sample_size_design_checksum": self.sample_size_design_checksum,
            "jobs": [job.to_dict() for job in self.jobs],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact deterministic run plan."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native plan data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> TrainingRunPlan:
        """Decode and verify a persisted run plan.

        Returns:
            The verified run plan.

        Raises:
            ValueError: If the schema or checksum is invalid.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_PLAN_KEYS, name="WP22 training run plan")
        if mapping["schema_version"] != TRAINING_RUN_PLAN_SCHEMA_VERSION:
            msg = "WP22 training run plan uses an unsupported schema version."
            raise ValueError(msg)
        plan = cls(
            plan_id=cast("str", mapping["plan_id"]),
            preset=cast("str", mapping["preset"]),
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            target_manifest_checksums=cast("tuple[str, ...]", mapping["target_manifest_checksums"]),
            screening_manifest_checksum=cast("str | None", mapping["screening_manifest_checksum"]),
            final_confirmation_seal_checksum=cast("str | None", mapping["final_confirmation_seal_checksum"]),
            execution_source_checksum=cast("str | None", mapping["execution_source_checksum"]),
            jobs=tuple(TrainingJob.from_dict(item) for item in cast("Sequence[object]", mapping["jobs"])),
            execution_profile_checksum=cast("str | None", mapping["execution_profile_checksum"]),
            scoped_binding_checksums=cast("tuple[str, ...]", mapping["scoped_binding_checksums"]),
            executable_binding_checksums=cast("tuple[str, ...]", mapping["executable_binding_checksums"]),
            implementation_checksums=cast("tuple[str, ...]", mapping["implementation_checksums"]),
            evaluation_policy_checksums=cast("tuple[str, ...]", mapping["evaluation_policy_checksums"]),
            target_configuration_checksums=cast("tuple[str, ...]", mapping["target_configuration_checksums"]),
            source_fingerprint_checksums=cast("tuple[str, ...]", mapping["source_fingerprint_checksums"]),
            scheduled_execution_program_checksums=cast(
                "tuple[str, ...]",
                mapping["scheduled_execution_program_checksums"],
            ),
            sample_size_design_checksum=cast("str | None", mapping["sample_size_design_checksum"]),
        )
        if mapping["content_checksum"] != plan.content_checksum:
            msg = "WP22 training run plan checksum changed during normalization."
            raise ValueError(msg)
        return plan

    @classmethod
    def from_json(cls, payload: str) -> TrainingRunPlan:
        """Decode canonical plan JSON.

        Returns:
            The verified run plan.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _candidate_job(
    *,
    preset: str,
    candidate: WP22CandidateConfiguration,
    strategy_schedule: TrainingStrategySchedule,
    target_manifest: TargetPopulationManifest,
    target_index: int,
    optimization_seed: int,
    optimization_block_id: str,
    evaluation_seed: int,
) -> TrainingJob:
    """Build one candidate job without accepting caller-selected target metadata.

    The q12 pilot population remains custodied by its
    ``screening_selection/secondary_q12`` manifest, while execution uses the
    pipeline schema's non-promotional ``secondary_benchmark`` data role.

    Returns:
        The checksum-sealed job request.
    """
    target = target_manifest.instances[target_index]
    job_data_role = (
        "secondary_benchmark"
        if preset == "paper-pilot"
        and target_manifest.data_role == "screening_selection"
        and target_manifest.population_scope == "secondary_q12"
        else target_manifest.data_role
    )
    identity = {
        "preset": preset,
        "candidate": candidate.content_checksum,
        "target": target.content_checksum,
        "optimization_block_id": optimization_block_id,
        "optimization_seed": optimization_seed,
        "evaluation_seed": evaluation_seed,
    }
    job_id = f"wp22_job_{canonical_checksum(identity).removeprefix('sha256:')}"
    return TrainingJob(
        job_id=job_id,
        preset=preset,
        method_id=candidate.method_id,
        implementation_kind=candidate.implementation_kind,
        candidate_configuration_checksum=candidate.content_checksum,
        implementation_checksum=candidate.implementation_checksum,
        strategy_schedule_checksum=candidate.strategy_schedule_checksum,
        strategy_schedule=strategy_schedule,
        target_manifest_checksum=target_manifest.content_checksum,
        target_instance_id=target.target_instance_id,
        target_spec_checksum=target.content_checksum,
        family_id=target.family_id,
        stratum_id=target.stratum_id,
        qubit_count=target.qubit_count,
        data_role=job_data_role,
        optimization_block_id=optimization_block_id,
        optimization_seed=optimization_seed,
        evaluation_seed=evaluation_seed,
        output_path=f"roles/{job_data_role}/{target.family_id}/{target.target_instance_id}/{job_id}",
    )


def _binding_job(
    *,
    preset: str,
    executable_binding: ExecutableScopedBinding,
    target_manifest: TargetPopulationManifest,
    target_index: int,
    optimization_seed: int,
    optimization_block_id: str,
    evaluation_seed: int,
) -> TrainingJob:
    """Build one unbound plan job from an exact WP22B executable link.

    WP22D source and context fingerprints are attached later by
    :func:`execution_context.bind_training_plan_fingerprints`; this helper
    already replaces the obsolete publication-wrapper implementation checksum
    with the exact preset- and width-specific binding identity.

    Returns:
        The checksum-sealed unbound job request.

    Raises:
        TypeError: If the executable binding has the wrong typed schema.
        ValueError: If the binding and target scope differ.
    """
    if not isinstance(executable_binding, ExecutableScopedBinding):
        msg = "executable_binding must be an ExecutableScopedBinding."
        raise TypeError(msg)
    binding = executable_binding.binding
    target = target_manifest.instances[target_index]
    scope = "primary_q6" if target.qubit_count == 6 else "secondary_q12" if target.qubit_count == 12 else None
    if (
        binding.preset != preset
        or binding.target_scope_id != scope
        or binding.manifest_data_role != target_manifest.data_role
        or binding.qubit_count != target.qubit_count
    ):
        msg = "Executable binding does not match the preset and target-manifest scope."
        raise ValueError(msg)
    job_data_role = binding.execution_data_role
    identity = {
        "preset": preset,
        "candidate": binding.publication_candidate_checksum,
        "binding": binding.content_checksum,
        "target": target.content_checksum,
        "optimization_block_id": optimization_block_id,
        "optimization_seed": optimization_seed,
        "evaluation_seed": evaluation_seed,
    }
    job_id = f"wp22_job_{canonical_checksum(identity).removeprefix('sha256:')}"
    implementation_kind = (
        "phase2_pipeline"
        if binding.implementation_artifact.implementation_kind.startswith("phase2_pipeline")
        else "operator_growth"
    )
    return TrainingJob(
        job_id=job_id,
        preset=preset,
        method_id=binding.publication_method_id,
        implementation_kind=implementation_kind,
        candidate_configuration_checksum=binding.publication_candidate_checksum,
        implementation_checksum=binding.implementation_checksum,
        strategy_schedule_checksum=binding.strategy_schedule.content_checksum,
        strategy_schedule=binding.strategy_schedule,
        target_manifest_checksum=target_manifest.content_checksum,
        target_instance_id=target.target_instance_id,
        target_spec_checksum=target.content_checksum,
        family_id=target.family_id,
        stratum_id=target.stratum_id,
        qubit_count=target.qubit_count,
        data_role=job_data_role,
        optimization_block_id=optimization_block_id,
        optimization_seed=optimization_seed,
        evaluation_seed=evaluation_seed,
        output_path=f"roles/{job_data_role}/{target.family_id}/{target.target_instance_id}/{job_id}",
    )


def _executable_binding_tuple(
    values: Sequence[ExecutableScopedBinding],
    *,
    preset: str,
    expected_methods_and_scopes: set[tuple[str, str]],
) -> tuple[ExecutableScopedBinding, ...]:
    """Validate one exact preset binding universe.

    Returns:
        The caller order after complete membership validation.

    Raises:
        TypeError: If an entry has the wrong typed schema.
        ValueError: If the exact preset method/scope universe differs.
    """
    bindings = tuple(values)
    if not bindings or any(not isinstance(item, ExecutableScopedBinding) for item in bindings):
        msg = "executable_bindings must contain ExecutableScopedBinding records."
        raise TypeError(msg)
    actual = {(item.binding.publication_method_id, item.binding.target_scope_id) for item in bindings}
    if (
        len(actual) != len(bindings)
        or actual != expected_methods_and_scopes
        or any(item.binding.preset != preset for item in bindings)
    ):
        msg = f"{preset} executable bindings differ from the exact frozen method/scope universe."
        raise ValueError(msg)
    return bindings


def _candidate_schedule_map(
    candidates: Sequence[WP22CandidateConfiguration],
    schedules: Sequence[TrainingStrategySchedule],
) -> dict[str, TrainingStrategySchedule]:
    """Bind every strategy candidate to one complete typed schedule.

    Returns:
        A checksum-indexed exact schedule universe.

    Raises:
        TypeError: If candidates or schedules contain unsupported records.
        ValueError: If either checksum universe is incomplete or duplicated.
    """
    values = tuple(candidates)
    schedule_values = tuple(schedules)
    if not values or not all(isinstance(item, WP22CandidateConfiguration) for item in values):
        msg = "candidates must contain typed WP22 candidate configurations."
        raise TypeError(msg)
    if not schedule_values or not all(isinstance(item, TrainingStrategySchedule) for item in schedule_values):
        msg = "Strategy candidates require complete typed TrainingStrategySchedule records."
        raise TypeError(msg)
    schedule_map = {schedule.content_checksum: schedule for schedule in schedule_values}
    if len(schedule_map) != len(schedule_values):
        msg = "Training strategy schedules must have distinct content checksums."
        raise ValueError(msg)
    expected = {candidate.strategy_schedule_checksum for candidate in values}
    if set(schedule_map) != expected:
        msg = "Candidate and TrainingStrategySchedule checksum universes differ."
        raise ValueError(msg)
    for candidate in values:
        schedule = schedule_map[candidate.strategy_schedule_checksum]
        schedule_is_noisy = (
            schedule.phase_boundary.mode != "noiseless_only" and schedule.training_noise.mode != "noiseless"
        )
        if schedule_is_noisy != candidate.noisy_training:
            msg = f"Candidate {candidate.method_id!r} noisy_training disagrees with its complete schedule."
            raise ValueError(msg)
    return schedule_map


def build_training_smoke_plan(
    *,
    preregistration_checksum: str,
    target_manifest: TargetPopulationManifest,
    candidates: Sequence[WP22CandidateConfiguration] = (),
    schedules: Sequence[TrainingStrategySchedule] = (),
    executable_bindings: Sequence[ExecutableScopedBinding] = (),
) -> TrainingRunPlan:
    """Build one bounded target across all ten frozen smoke implementations.

    Returns:
        The deterministic smoke plan.

    Raises:
        ValueError: If the target role or candidate set is unsuitable.
    """
    if target_manifest.data_role != "development":
        msg = "training-smoke requires a development target manifest."
        raise ValueError(msg)
    target_index = next(
        (
            index
            for index, spec in enumerate(target_manifest.instances)
            if spec.qubit_count == 6 and spec.family_id == "tfim_ground_state"
        ),
        None,
    )
    if target_index is None:
        msg = "training-smoke requires one bounded q6 tfim_ground_state target shared by all methods."
        raise ValueError(msg)
    if executable_bindings:
        if candidates or schedules:
            msg = "Select executable_bindings or legacy candidate assertions, not both."
            raise ValueError(msg)
        links = _executable_binding_tuple(
            executable_bindings,
            preset="training-smoke",
            expected_methods_and_scopes={(method, "primary_q6") for method in SMOKE_METHOD_IDS},
        )
        suite = ExecutionSeedPolicySuite.frozen()
        jobs = tuple(
            sorted(
                (
                    _binding_job(
                        preset="training-smoke",
                        executable_binding=link,
                        target_manifest=target_manifest,
                        target_index=target_index,
                        optimization_seed=suite.derive(
                            SMOKE_OPTIMIZATION_SEED_POLICY_ID,
                            {"publication_candidate_checksum": link.binding.publication_candidate_checksum},
                        ),
                        optimization_block_id=f"smoke_{link.binding.publication_method_id}",
                        evaluation_seed=suite.derive(
                            SMOKE_FRESH_EVALUATION_SEED_POLICY_ID,
                            {"publication_candidate_checksum": link.binding.publication_candidate_checksum},
                        ),
                    )
                    for link in links
                ),
                key=_job_sort_key,
            )
        )
    else:
        values = tuple(candidates)
        schedule_map = _candidate_schedule_map(values, schedules)
        methods = tuple(item.method_id for item in values)
        if len(values) != len(SMOKE_METHOD_IDS) or set(methods) != set(SMOKE_METHOD_IDS):
            msg = "training-smoke requires exactly the ten frozen q6 method identities."
            raise ValueError(msg)
        suite = ExecutionSeedPolicySuite.frozen()
        jobs = tuple(
            sorted(
                (
                    _candidate_job(
                        preset="training-smoke",
                        candidate=candidate,
                        strategy_schedule=schedule_map[candidate.strategy_schedule_checksum],
                        target_manifest=target_manifest,
                        target_index=target_index,
                        optimization_seed=suite.derive(
                            SMOKE_OPTIMIZATION_SEED_POLICY_ID,
                            {"publication_candidate_checksum": candidate.content_checksum},
                        ),
                        optimization_block_id=f"smoke_{candidate.method_id}",
                        evaluation_seed=suite.derive(
                            SMOKE_FRESH_EVALUATION_SEED_POLICY_ID,
                            {"publication_candidate_checksum": candidate.content_checksum},
                        ),
                    )
                    for candidate in values
                ),
                key=_job_sort_key,
            )
        )
    if len(jobs) != 10:
        msg = "training-smoke must expand to exactly ten jobs."
        raise ValueError(msg)
    return TrainingRunPlan(
        plan_id="wp22_training_smoke_v1",
        preset="training-smoke",
        preregistration_checksum=preregistration_checksum,
        target_manifest_checksums=(target_manifest.content_checksum,),
        screening_manifest_checksum=None,
        final_confirmation_seal_checksum=None,
        execution_source_checksum=None,
        jobs=jobs,
    )


def build_paper_pilot_plan(
    *,
    preregistration_checksum: str,
    target_manifests: Sequence[TargetPopulationManifest],
    candidates: Sequence[WP22CandidateConfiguration] = (),
    schedules: Sequence[TrainingStrategySchedule] = (),
    optimization_seeds: Sequence[int],
    executable_bindings: Sequence[ExecutableScopedBinding] = (),
) -> TrainingRunPlan:
    """Build the primary-q6 pilot and preregistered secondary-q12 fan-out.

    The q12 population retains its custodied ``screening_selection`` target
    role, but it is scheduled only by this pilot preset.  Pilot inference
    accepts only the q6 development jobs, and :func:`build_paper_screen_plan`
    independently requires a primary-q6 manifest, so the q12 collection can
    never enter primary screening or promotion.

    Returns:
        The deterministic pilot plan.

    Raises:
        TypeError: If manifests or candidates have the wrong types.
        ValueError: If roles, manifests, candidates, or seeds are duplicated.
    """
    manifests = tuple(target_manifests)
    if not manifests or not all(isinstance(item, TargetPopulationManifest) for item in manifests):
        msg = "target_manifests must contain typed target manifests."
        raise TypeError(msg)
    if len({manifest.content_checksum for manifest in manifests}) != len(manifests):
        msg = "paper-pilot target manifests must be distinct."
        raise ValueError(msg)
    manifests_by_collection = {(manifest.data_role, manifest.population_scope): manifest for manifest in manifests}
    expected_collections = {
        ("development", "primary_q6"),
        ("screening_selection", "secondary_q12"),
    }
    if set(manifests_by_collection) != expected_collections or len(manifests_by_collection) != len(manifests):
        msg = (
            "paper-pilot requires exactly one development/primary_q6 manifest and one "
            "screening_selection/secondary_q12 manifest."
        )
        raise ValueError(msg)
    manifests = tuple(manifests_by_collection[key] for key in sorted(expected_collections))
    seeds = tuple(require_int(seed, "optimization_seed", minimum=0) for seed in optimization_seeds)
    if len(seeds) != PILOT_OPTIMIZATION_SEED_COUNT or len(seeds) != len(set(seeds)):
        msg = f"paper-pilot requires exactly {PILOT_OPTIMIZATION_SEED_COUNT} distinct optimization seeds."
        raise ValueError(msg)
    if executable_bindings:
        if candidates or schedules:
            msg = "Select executable_bindings or legacy candidate assertions, not both."
            raise ValueError(msg)
        links = _executable_binding_tuple(
            executable_bindings,
            preset="paper-pilot",
            expected_methods_and_scopes={
                (method, scope) for method in PILOT_METHOD_IDS for scope in ("primary_q6", "secondary_q12")
            },
        )
        candidates_by_scope: Mapping[str, Sequence[ExecutableScopedBinding]] = {
            scope: tuple(link for link in links if link.binding.target_scope_id == scope)
            for scope in ("primary_q6", "secondary_q12")
        }
        legacy_values: tuple[WP22CandidateConfiguration, ...] = ()
        schedule_map: Mapping[str, TrainingStrategySchedule] = {}
    else:
        legacy_values = tuple(candidates)
        if not all(isinstance(item, WP22CandidateConfiguration) for item in legacy_values):
            msg = "candidates must contain typed WP22 candidate configurations."
            raise TypeError(msg)
        if len(legacy_values) != len(PILOT_METHOD_IDS) or {item.method_id for item in legacy_values} != set(
            PILOT_METHOD_IDS
        ):
            msg = "paper-pilot requires exactly the three frozen publication candidates."
            raise ValueError(msg)
        schedule_map = _candidate_schedule_map(legacy_values, schedules)
        candidates_by_scope = {}
    suite = ExecutionSeedPolicySuite.frozen()
    jobs: list[TrainingJob] = []
    for manifest in manifests:
        scope = manifest.population_scope
        for target_index, target in enumerate(manifest.instances):
            for seed in seeds:
                block = f"pilot_{target.target_instance_id}_seed_{seed}"
                if executable_bindings:
                    jobs.extend(
                        _binding_job(
                            preset="paper-pilot",
                            executable_binding=link,
                            target_manifest=manifest,
                            target_index=target_index,
                            optimization_seed=seed,
                            optimization_block_id=block,
                            evaluation_seed=suite.derive(
                                PILOT_FRESH_EVALUATION_SEED_POLICY_ID,
                                {
                                    "target_manifest_checksum": manifest.content_checksum,
                                    "target_instance_spec_checksum": target.content_checksum,
                                    "optimization_seed": seed,
                                    "publication_candidate_checksum": (link.binding.publication_candidate_checksum),
                                },
                            ),
                        )
                        for link in candidates_by_scope[scope]
                    )
                else:
                    jobs.extend(
                        _candidate_job(
                            preset="paper-pilot",
                            candidate=candidate,
                            strategy_schedule=schedule_map[candidate.strategy_schedule_checksum],
                            target_manifest=manifest,
                            target_index=target_index,
                            optimization_seed=seed,
                            optimization_block_id=block,
                            evaluation_seed=suite.derive(
                                PILOT_FRESH_EVALUATION_SEED_POLICY_ID,
                                {
                                    "target_manifest_checksum": manifest.content_checksum,
                                    "target_instance_spec_checksum": target.content_checksum,
                                    "optimization_seed": seed,
                                    "publication_candidate_checksum": candidate.content_checksum,
                                },
                            ),
                        )
                        for candidate in legacy_values
                    )
    if len(jobs) != 1_080:
        msg = "paper-pilot must expand to exactly 1,080 jobs."
        raise ValueError(msg)
    return TrainingRunPlan(
        plan_id="wp22_paper_pilot_v1",
        preset="paper-pilot",
        preregistration_checksum=preregistration_checksum,
        target_manifest_checksums=tuple(manifest.content_checksum for manifest in manifests),
        screening_manifest_checksum=None,
        final_confirmation_seal_checksum=None,
        execution_source_checksum=None,
        jobs=tuple(sorted(jobs, key=_job_sort_key)),
    )


def build_paper_screen_plan(
    *,
    preregistration_checksum: str,
    target_manifest: TargetPopulationManifest,
    screening_manifest: ScreeningManifest,
    candidates: Sequence[WP22CandidateConfiguration] = (),
    schedules: Sequence[TrainingStrategySchedule] = (),
    executable_bindings: Sequence[ExecutableScopedBinding] = (),
) -> TrainingRunPlan:
    """Build the exact preregistered candidate-by-screening-cell matrix.

    Returns:
        The deterministic complete paper-screen plan.

    Raises:
        ValueError: If candidates, cells, targets, or preregistration differ.
    """
    verify_screening_target_population(screening_manifest, target_manifest)
    if screening_manifest.preregistration_checksum != preregistration_checksum:
        msg = "Screening manifest belongs to a different preregistration."
        raise ValueError(msg)
    target_index = {spec.target_instance_id: index for index, spec in enumerate(target_manifest.instances)}
    if executable_bindings:
        if candidates or schedules:
            msg = "Select executable_bindings or legacy candidate assertions, not both."
            raise ValueError(msg)
        links = _executable_binding_tuple(
            executable_bindings,
            preset="paper-screen",
            expected_methods_and_scopes={(method, "primary_q6") for method in SCREEN_METHOD_IDS},
        )
        by_checksum = {link.binding.publication_candidate_checksum: link for link in links}
        if set(by_checksum) != {item.configuration_checksum for item in screening_manifest.candidates}:
            msg = "Executable bindings do not implement the exact screening-manifest universe."
            raise ValueError(msg)
        jobs = [
            _binding_job(
                preset="paper-screen",
                executable_binding=by_checksum[candidate_ref.configuration_checksum],
                target_manifest=target_manifest,
                target_index=target_index[cell.target_instance_id],
                optimization_seed=cell.optimization_seed,
                optimization_block_id=cell.cell_id,
                evaluation_seed=cell.screening_seed,
            )
            for candidate_ref in screening_manifest.candidates
            for cell in screening_manifest.cells
        ]
    else:
        values = tuple(candidates)
        schedule_map = _candidate_schedule_map(values, schedules)
        if len(values) != len(SCREEN_METHOD_IDS) or {item.method_id for item in values} != set(SCREEN_METHOD_IDS):
            msg = "paper-screen requires exactly the nine frozen publication candidates."
            raise ValueError(msg)
        by_checksum_candidates = {candidate.content_checksum: candidate for candidate in values}
        if len(by_checksum_candidates) != len(values) or set(by_checksum_candidates) != {
            item.configuration_checksum for item in screening_manifest.candidates
        }:
            msg = "Candidate configurations do not implement the exact screening-manifest universe."
            raise ValueError(msg)
        jobs = [
            _candidate_job(
                preset="paper-screen",
                candidate=by_checksum_candidates[candidate_ref.configuration_checksum],
                strategy_schedule=schedule_map[
                    by_checksum_candidates[candidate_ref.configuration_checksum].strategy_schedule_checksum
                ],
                target_manifest=target_manifest,
                target_index=target_index[cell.target_instance_id],
                optimization_seed=cell.optimization_seed,
                optimization_block_id=cell.cell_id,
                evaluation_seed=cell.screening_seed,
            )
            for candidate_ref in screening_manifest.candidates
            for cell in screening_manifest.cells
        ]
    if len(jobs) != 1_296:
        msg = "paper-screen must expand to exactly 1,296 jobs."
        raise ValueError(msg)
    return TrainingRunPlan(
        plan_id="wp22_paper_screen_v1",
        preset="paper-screen",
        preregistration_checksum=preregistration_checksum,
        target_manifest_checksums=(target_manifest.content_checksum,),
        screening_manifest_checksum=screening_manifest.content_checksum,
        final_confirmation_seal_checksum=None,
        execution_source_checksum=None,
        jobs=tuple(sorted(jobs, key=_job_sort_key)),
    )


def build_historical_reproduction_plan(*, preregistration_checksum: str) -> TrainingRunPlan:
    """Build the exact five-row legacy delegation plan.

    Returns:
        The deterministic historical reproduction plan.
    """
    jobs: list[TrainingJob] = []
    for target_seed in (100, 200, 300, 400, 500):
        pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(target_seed)
        job_id = f"wp22_legacy_target_seed_{target_seed}"
        jobs.append(
            TrainingJob(
                job_id=job_id,
                preset="historical-layerwise-reproduction",
                method_id=pipeline.method_id,
                implementation_kind="legacy_delegate",
                candidate_configuration_checksum=pipeline.template.configuration_checksum,
                implementation_checksum=pipeline.template.configuration_checksum,
                strategy_schedule_checksum=canonical_checksum({"legacy_pipeline": pipeline.template.to_dict()}),
                target_manifest_checksum=pipeline.target_population_manifest_checksum,
                target_instance_id=pipeline.target_instance_id,
                target_spec_checksum=pipeline.target_instance_spec_checksum,
                family_id=pipeline.target_family_id,
                stratum_id=pipeline.target_stratum_id,
                qubit_count=pipeline.qubit_count,
                data_role="secondary_benchmark",
                optimization_block_id=pipeline.optimization_block_id,
                optimization_seed=pipeline.optimization_seed,
                evaluation_seed=0,
                output_path=f"roles/secondary_benchmark/tfim_ground_state/{pipeline.target_instance_id}/{job_id}",
            )
        )
    return TrainingRunPlan(
        plan_id="wp22_historical_layerwise_reproduction_v1",
        preset="historical-layerwise-reproduction",
        preregistration_checksum=preregistration_checksum,
        target_manifest_checksums=(jobs[0].target_manifest_checksum,),
        screening_manifest_checksum=None,
        final_confirmation_seal_checksum=None,
        execution_source_checksum=None,
        jobs=tuple(sorted(jobs, key=_job_sort_key)),
    )


def build_paper_confirm_plan(
    *,
    seal: FinalConfirmationSeal,
    target_manifest: TargetPopulationManifest,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
) -> TrainingRunPlan:
    """Build a dormant confirmatory plan only from an authorized final seal.

    The caller must validate source custody before loading ``target_manifest``;
    this factory then rejects every target/configuration/count mismatch.

    Returns:
        The deterministic confirmatory Cartesian plan.

    Raises:
        TypeError: If the seal or revealed manifest has the wrong type.
        ValueError: If targets, methods, or counts differ from the final seal.
    """
    if not isinstance(seal, FinalConfirmationSeal):
        msg = "seal must be a FinalConfirmationSeal."
        raise TypeError(msg)
    if not isinstance(target_manifest, TargetPopulationManifest):
        msg = "target_manifest must be a TargetPopulationManifest."
        raise TypeError(msg)
    if target_manifest.data_role != "confirmatory":
        msg = "Revealed confirmatory target manifest does not match the final seal."
        raise ValueError(msg)
    context = build_confirm_execution_context(
        seal,
        target_manifest,
        configuration_execution_manifest,
    )
    actual_counts = {
        family: sum(spec.family_id == family for spec in target_manifest.instances)
        for family in seal.target_count_by_family
    }
    if actual_counts != dict(seal.target_count_by_family):
        msg = "Revealed confirmatory family counts differ from the final seal."
        raise ValueError(msg)
    configurations = [context.execution_by_configuration[seal.promoted_configuration_checksum]]
    configurations.extend(context.execution_by_configuration[item.configuration_checksum] for item in seal.comparators)
    if len({item.configuration_checksum for item in configurations}) != len(configurations):
        msg = "Final seal contains duplicate confirmatory configurations."
        raise ValueError(msg)
    jobs: list[TrainingJob] = []
    for target in target_manifest.instances:
        target_checksum = target.content_checksum
        for seed_index in range(seal.optimization_seed_count):
            optimization_seed = derive_confirmatory_optimization_seed(
                context.final_seal_checksum,
                target_checksum,
                seed_index,
            )
            block = f"confirm_{target.target_instance_id}_seed_index_{seed_index}"
            for execution in configurations:
                method_id = execution.method_id
                configuration_checksum = execution.configuration_checksum
                evaluation_seed = derive_confirmatory_evaluation_seed(
                    context.final_seal_checksum,
                    target_checksum,
                    seed_index,
                    configuration_checksum,
                )
                identity = {
                    "seal": context.final_seal_checksum,
                    "configuration": configuration_checksum,
                    "target": target_checksum,
                    "optimization_seed": optimization_seed,
                    "evaluation_seed": evaluation_seed,
                }
                job_id = f"wp22_job_{canonical_checksum(identity).removeprefix('sha256:')}"
                request = ConfirmExecutionRequest(
                    final_confirmation_seal_checksum=context.final_seal_checksum,
                    preregistration_checksum=seal.preregistration_checksum,
                    promotion_decision_checksum=seal.promotion_decision_checksum,
                    execution_source_checksum=seal.execution_source_checksum,
                    analysis_source_manifest_checksum=seal.analysis_source_manifest_checksum,
                    analysis_template_checksum=seal.analysis_template_checksum,
                    configuration_execution_manifest_checksum=(
                        context.configuration_execution_manifest.content_checksum
                    ),
                    hyperparameters_checksum=execution.strategy_schedule_checksum,
                    implementation_checksum=execution.implementation_checksum,
                    scoped_binding_checksum=execution.scoped_binding_checksum,
                    executable_binding_checksum=execution.executable_binding_checksum,
                    sample_size_design_checksum=seal.sample_size_design_checksum,
                    failure_policy_checksum=seal.failure_policy_checksum,
                    fixed_test_trajectory_count=seal.fixed_test_trajectory_count,
                    primary_noise_condition=seal.primary_noise_condition,
                    primary_resource_budget=seal.primary_resource_budget,
                    method_id=method_id,
                    configuration_checksum=configuration_checksum,
                    target_manifest_checksum=context.target_manifest_checksum,
                    target_instance_id=target.target_instance_id,
                    target_spec_checksum=target_checksum,
                    family_id=target.family_id,
                    stratum_id=target.stratum_id,
                    qubit_count=target.qubit_count,
                    optimization_block_id=block,
                    optimization_seed_index=seed_index,
                    optimization_seed=optimization_seed,
                    evaluation_seed=evaluation_seed,
                )
                jobs.append(
                    TrainingJob(
                        job_id=job_id,
                        preset="paper-confirm",
                        method_id=method_id,
                        implementation_kind="sealed_configuration",
                        candidate_configuration_checksum=configuration_checksum,
                        implementation_checksum=execution.implementation_checksum,
                        strategy_schedule_checksum=execution.strategy_schedule_checksum,
                        target_manifest_checksum=context.target_manifest_checksum,
                        target_instance_id=target.target_instance_id,
                        target_spec_checksum=target_checksum,
                        family_id=target.family_id,
                        stratum_id=target.stratum_id,
                        qubit_count=target.qubit_count,
                        data_role="confirmatory",
                        optimization_block_id=block,
                        optimization_seed=optimization_seed,
                        evaluation_seed=evaluation_seed,
                        output_path=(f"roles/confirmatory/{target.family_id}/{target.target_instance_id}/{job_id}"),
                        confirm_execution_request=request,
                    )
                )
    return TrainingRunPlan(
        plan_id="wp22_paper_confirm_v1",
        preset="paper-confirm",
        preregistration_checksum=seal.preregistration_checksum,
        target_manifest_checksums=(context.target_manifest_checksum,),
        screening_manifest_checksum=None,
        final_confirmation_seal_checksum=context.final_seal_checksum,
        execution_source_checksum=seal.execution_source_checksum,
        jobs=tuple(sorted(jobs, key=_job_sort_key)),
    )


def reject_ballarin_training(template: TrainingPipelineTemplate) -> None:
    """Reject Ballarin's coupled benchmark model as a training objective.

    Raises:
        TypeError: If ``template`` has the wrong type.
        ValueError: If any stage requests Ballarin training.
    """
    if not isinstance(template, TrainingPipelineTemplate):
        msg = "template must be a TrainingPipelineTemplate."
        raise TypeError(msg)
    if any(stage.stage_policy["training_noise_id"] == BALLARIN_NOISE_ID for stage in template.stages):
        msg = "Ballarin noise is an evaluation benchmark only and cannot be used for Phase II training."
        raise ValueError(msg)


def validate_job_pipeline_binding(
    job: TrainingJob,
    candidate: WP22CandidateConfiguration,
    template: TrainingPipelineTemplate,
    pipeline: TrainingPipelineConfig,
    *,
    screening_manifest: ScreeningManifest | None = None,
    screening_cell: ScreeningCell | None = None,
) -> None:
    """Verify a concrete pipeline against its immutable WP22 job wrapper.

    Raises:
        TypeError: If an input has the wrong typed schema.
        ValueError: If candidate, template, target, seed, role, or cell differs.
    """
    if not isinstance(job, TrainingJob) or not isinstance(candidate, WP22CandidateConfiguration):
        msg = "job and candidate must be typed WP22 records."
        raise TypeError(msg)
    if not isinstance(template, TrainingPipelineTemplate) or not isinstance(pipeline, TrainingPipelineConfig):
        msg = "template and pipeline must be typed Phase II records."
        raise TypeError(msg)
    reject_ballarin_training(template)
    if (
        job.implementation_kind != "phase2_pipeline"
        or job.candidate_configuration_checksum != candidate.content_checksum
        or job.method_id != candidate.method_id
        or job.implementation_checksum != candidate.implementation_checksum
        or job.strategy_schedule_checksum != candidate.strategy_schedule_checksum
        or candidate.implementation_kind != "phase2_pipeline"
        or candidate.implementation_method_id != template.method_id
        or candidate.implementation_checksum != template.configuration_checksum
        or pipeline.template != template
        or pipeline.target_population_manifest_checksum != job.target_manifest_checksum
        or pipeline.target_instance_id != job.target_instance_id
        or pipeline.target_instance_spec_checksum != job.target_spec_checksum
        or pipeline.target_family_id != job.family_id
        or pipeline.target_stratum_id != job.stratum_id
        or pipeline.qubit_count != job.qubit_count
        or pipeline.data_role != job.data_role
        or pipeline.optimization_block_id != job.optimization_block_id
        or pipeline.optimization_seed != job.optimization_seed
    ):
        msg = "Concrete pipeline does not implement the exact sealed WP22 job."
        raise ValueError(msg)
    if (screening_manifest is None) != (screening_cell is None):
        msg = "screening_manifest and screening_cell must be supplied together."
        raise ValueError(msg)
    if (
        screening_manifest is not None
        and screening_cell is not None
        and (
            candidate.screening_ref() not in screening_manifest.candidates
            or screening_cell not in screening_manifest.cells
            or screening_cell.cell_id != job.optimization_block_id
            or screening_cell.target_instance_id != job.target_instance_id
            or screening_cell.optimization_seed != job.optimization_seed
            or screening_cell.screening_seed != job.evaluation_seed
        )
    ):
        msg = "WP22 job does not match its complete screening candidate/cell records."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TrainingJobOutcome:
    """Durable orchestration outcome for one exact job request."""

    job_checksum: str
    status: Literal["success", "failure"]
    result_artifact_checksum: str | None
    exception_type: str | None
    message: str | None
    attempt: int
    schema_version: str = field(default=TRAINING_JOB_OUTCOME_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate success/failure fields and attempt identity.

        Raises:
            ValueError: If status-specific fields or identifiers are invalid.
        """
        object.__setattr__(self, "job_checksum", require_checksum(self.job_checksum, "job_checksum"))
        object.__setattr__(self, "attempt", require_int(self.attempt, "attempt", minimum=1))
        if self.status == "success":
            if self.result_artifact_checksum is None or self.exception_type is not None or self.message is not None:
                msg = "Successful job outcomes require only a result artifact checksum."
                raise ValueError(msg)
            object.__setattr__(
                self,
                "result_artifact_checksum",
                require_checksum(self.result_artifact_checksum, "result_artifact_checksum"),
            )
        elif self.status == "failure":
            if self.result_artifact_checksum is not None or self.exception_type is None or self.message is None:
                msg = "Failed job outcomes require exception_type and message only."
                raise ValueError(msg)
            object.__setattr__(self, "exception_type", require_slug(self.exception_type, "exception_type"))
            object.__setattr__(self, "message", require_nonempty_text(self.message, "message"))
        else:
            msg = "status must be 'success' or 'failure'."
            raise ValueError(msg)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered outcome field."""
        return {
            "schema_version": self.schema_version,
            "job_checksum": self.job_checksum,
            "status": self.status,
            "result_artifact_checksum": self.result_artifact_checksum,
            "exception_type": self.exception_type,
            "message": self.message,
            "attempt": self.attempt,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact orchestration outcome."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native outcome data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> TrainingJobOutcome:
        """Decode and verify one durable job outcome.

        Returns:
            The verified outcome.

        Raises:
            ValueError: If the schema or checksum is invalid.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_OUTCOME_KEYS, name="WP22 job outcome")
        if mapping["schema_version"] != TRAINING_JOB_OUTCOME_SCHEMA_VERSION:
            msg = "WP22 job outcome uses an unsupported schema version."
            raise ValueError(msg)
        outcome = cls(
            job_checksum=cast("str", mapping["job_checksum"]),
            status=cast("Literal['success', 'failure']", mapping["status"]),
            result_artifact_checksum=cast("str | None", mapping["result_artifact_checksum"]),
            exception_type=cast("str | None", mapping["exception_type"]),
            message=cast("str | None", mapping["message"]),
            attempt=cast("int", mapping["attempt"]),
        )
        if mapping["content_checksum"] != outcome.content_checksum:
            msg = "WP22 job outcome checksum changed during normalization."
            raise ValueError(msg)
        return outcome


@dataclass(frozen=True, slots=True)
class TrainingScheduleResumeState:
    """Typed schedule state exposed at one executor resume boundary."""

    strategy_schedule_checksum: str
    schedule_id: str
    resume_requested: bool
    overwrite_requested: bool
    prior_attempt: int
    prior_outcome_checksum: str | None
    prior_status: Literal["success", "failure"] | None

    def __post_init__(self) -> None:
        """Validate schedule identity, controls, and prior-outcome linkage.

        Raises:
            ValueError: If controls conflict or prior fields are inconsistent.
        """
        object.__setattr__(
            self,
            "strategy_schedule_checksum",
            require_checksum(self.strategy_schedule_checksum, "strategy_schedule_checksum"),
        )
        object.__setattr__(self, "schedule_id", require_slug(self.schedule_id, "schedule_id"))
        resume = require_bool(self.resume_requested, "resume_requested")
        overwrite = require_bool(self.overwrite_requested, "overwrite_requested")
        if resume and overwrite:
            msg = "resume_requested and overwrite_requested are mutually exclusive."
            raise ValueError(msg)
        attempt = require_int(self.prior_attempt, "prior_attempt", minimum=0)
        object.__setattr__(self, "prior_attempt", attempt)
        if attempt == 0:
            if self.prior_outcome_checksum is not None or self.prior_status is not None:
                msg = "A fresh schedule state cannot reference a prior outcome."
                raise ValueError(msg)
        elif self.prior_outcome_checksum is None or self.prior_status not in {"success", "failure"}:
            msg = "A resumed or overwritten schedule state requires its typed prior outcome."
            raise ValueError(msg)
        else:
            object.__setattr__(
                self,
                "prior_outcome_checksum",
                require_checksum(self.prior_outcome_checksum, "prior_outcome_checksum"),
            )


@dataclass(frozen=True, slots=True)
class JobExecutionControls:
    """Non-scientific execution controls passed to one method executor."""

    resume: bool
    overwrite: bool
    schedule_resume_state: TrainingScheduleResumeState | None = None

    def __post_init__(self) -> None:
        """Validate mutually exclusive execution controls.

        Raises:
            TypeError: If schedule_resume_state has the wrong record type.
            ValueError: If resume and overwrite are both enabled.
        """
        object.__setattr__(self, "resume", require_bool(self.resume, "resume"))
        object.__setattr__(self, "overwrite", require_bool(self.overwrite, "overwrite"))
        if self.resume and self.overwrite:
            msg = "resume and overwrite are mutually exclusive."
            raise ValueError(msg)
        if self.schedule_resume_state is not None:
            if not isinstance(self.schedule_resume_state, TrainingScheduleResumeState):
                msg = "schedule_resume_state must be a TrainingScheduleResumeState."
                raise TypeError(msg)
            if (
                self.schedule_resume_state.resume_requested != self.resume
                or self.schedule_resume_state.overwrite_requested != self.overwrite
            ):
                msg = "Schedule resume state disagrees with its execution controls."
                raise ValueError(msg)


TrainingJobExecutor = Callable[[TrainingJob, Path, JobExecutionControls], str]
ConfirmJobExecutor = Callable[[ConfirmExecutionRequest, Path, JobExecutionControls], str]


@dataclass(frozen=True, slots=True)
class TrainingExecutorRegistry:
    """Explicit implementation-kind registry with a typed confirm dispatch."""

    phase2_pipeline_executor: TrainingJobExecutor | None = None
    operator_growth_executor: TrainingJobExecutor | None = None
    legacy_delegate_executor: TrainingJobExecutor | None = None
    confirm_executor: ConfirmJobExecutor | None = None

    def __post_init__(self) -> None:
        """Require every populated executor slot to be callable.

        Raises:
            TypeError: If a populated slot is not callable.
            ValueError: If the registry is empty.
        """
        executors = (
            self.phase2_pipeline_executor,
            self.operator_growth_executor,
            self.legacy_delegate_executor,
            self.confirm_executor,
        )
        if not any(executor is not None for executor in executors):
            msg = "TrainingExecutorRegistry must register at least one executor."
            raise ValueError(msg)
        if any(executor is not None and not callable(executor) for executor in executors):
            msg = "Every registered executor must be callable."
            raise TypeError(msg)

    def supports(self, job: TrainingJob) -> bool:
        """Return whether the exact job kind has a registered dispatcher."""
        if job.implementation_kind == "phase2_pipeline":
            return self.phase2_pipeline_executor is not None
        if job.implementation_kind == "operator_growth":
            return self.operator_growth_executor is not None
        if job.implementation_kind == "legacy_delegate":
            return self.legacy_delegate_executor is not None
        return self.confirm_executor is not None and job.confirm_execution_request is not None

    def dispatch(self, job: TrainingJob, directory: Path, controls: JobExecutionControls) -> str:
        """Dispatch one job through its explicitly registered typed boundary.

        Returns:
            The executor's result-artifact checksum.

        Raises:
            ValueError: If the exact implementation kind is unregistered.
        """
        if job.implementation_kind == "phase2_pipeline":
            executor = self.phase2_pipeline_executor
        elif job.implementation_kind == "operator_growth":
            executor = self.operator_growth_executor
        elif job.implementation_kind == "legacy_delegate":
            executor = self.legacy_delegate_executor
        else:
            if self.confirm_executor is None or job.confirm_execution_request is None:
                msg = "No seal-complete confirm executor is registered for this job."
                raise ValueError(msg)
            return self.confirm_executor(job.confirm_execution_request, directory, controls)
        if executor is None:
            msg = f"No executor is registered for implementation_kind={job.implementation_kind!r}."
            raise ValueError(msg)
        return executor(job, directory, controls)


@dataclass(frozen=True, slots=True)
class TrainingRunSummary:
    """Aggregate outcome of one deterministic sequential dispatch."""

    planned: int
    attempted: int
    succeeded: int
    failed: int
    skipped: int


def training_job_attempt_path(job_directory: Path, attempt: int) -> Path:
    """Return the immutable path for one exact orchestration attempt.

    Returns:
        The attempt-addressed outcome path.

    Raises:
        TypeError: If ``job_directory`` is not a path.
        ValueError: If ``attempt`` is outside the supported filename range.
    """
    if not isinstance(job_directory, Path):
        msg = "job_directory must be a pathlib.Path."
        raise TypeError(msg)
    normalized = require_int(attempt, "attempt", minimum=1)
    if normalized > 99_999_999:
        msg = "attempt exceeds the eight-digit append-only filename range."
        raise ValueError(msg)
    return job_directory / JOB_ATTEMPTS_DIRECTORY_NAME / f"attempt_{normalized:08d}.json"


def load_training_job_outcome_history(
    job_directory: Path,
    job: TrainingJob,
) -> tuple[TrainingJobOutcome, ...]:
    """Load the complete append-only outcome history for one sealed job.

    The attempt directory, rather than the mutable latest-outcome projection,
    is authoritative.  Histories must start at one, be contiguous, and bind
    every attempt to the exact same job checksum.

    Returns:
        Verified outcomes in increasing attempt order, or an empty tuple.

    Raises:
        TypeError: If inputs have unsupported types.
        ValueError: If a history path, sequence, schema, or job link is invalid.
    """
    if not isinstance(job_directory, Path):
        msg = "job_directory must be a pathlib.Path."
        raise TypeError(msg)
    if not isinstance(job, TrainingJob):
        msg = "job must be a TrainingJob."
        raise TypeError(msg)
    history_directory = job_directory / JOB_ATTEMPTS_DIRECTORY_NAME
    if not history_directory.exists():
        return ()
    if history_directory.is_symlink() or not history_directory.is_dir():
        msg = f"Outcome history {history_directory} must be a non-symlink directory."
        raise ValueError(msg)
    entries = tuple(sorted(history_directory.iterdir(), key=lambda path: path.name))
    outcomes: list[TrainingJobOutcome] = []
    for expected_attempt, path in enumerate(entries, start=1):
        match = _ATTEMPT_FILE_PATTERN.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            msg = f"Outcome history contains an unsupported entry: {path}."
            raise ValueError(msg)
        named_attempt = int(match.group("attempt"))
        if named_attempt != expected_attempt:
            msg = "Outcome history attempts must be contiguous and start at one."
            raise ValueError(msg)
        outcome = TrainingJobOutcome.from_dict(load_canonical_json_object(path.read_text(encoding="utf-8")))
        if outcome.attempt != named_attempt:
            msg = f"Outcome {path} differs from its attempt-addressed filename."
            raise ValueError(msg)
        if outcome.job_checksum != job.content_checksum:
            msg = f"Outcome {path} belongs to a different job."
            raise ValueError(msg)
        outcomes.append(outcome)
    return tuple(outcomes)


def _write_outcome_attempt(job_directory: Path, outcome: TrainingJobOutcome) -> Path:
    """Publish one outcome without replacing any prior attempt.

    Returns:
        The new immutable attempt path.

    Raises:
        ValueError: If the attempt path already exists or is unsafe.
    """
    history_directory = job_directory / JOB_ATTEMPTS_DIRECTORY_NAME
    history_directory.mkdir(parents=True, exist_ok=True)
    if history_directory.is_symlink() or not history_directory.is_dir():
        msg = f"Outcome history {history_directory} must be a non-symlink directory."
        raise ValueError(msg)
    attempt_path = training_job_attempt_path(job_directory, outcome.attempt)
    if attempt_path.exists() or attempt_path.is_symlink():
        msg = f"Refusing to replace immutable outcome attempt {attempt_path}."
        raise ValueError(msg)
    payload = f"{canonical_json(outcome.to_dict())}\n".encode()
    temporary = job_directory / f".{JOB_RESULT_NAME}.{outcome.content_checksum.removeprefix('sha256:')}.tmp"
    atomic_write_bytes(temporary, payload)
    try:
        os.link(temporary, attempt_path)
    except FileExistsError as error:
        msg = f"Refusing to replace immutable outcome attempt {attempt_path}."
        raise ValueError(msg) from error
    finally:
        temporary.unlink(missing_ok=True)
    return attempt_path


def _synchronize_latest_outcome(
    job_directory: Path,
    history: Sequence[TrainingJobOutcome],
) -> None:
    """Rebuild the non-authoritative latest projection from attempt history.

    Raises:
        ValueError: If a projection exists without authoritative history.
    """
    result_path = job_directory / JOB_RESULT_NAME
    if not history:
        if result_path.exists() or result_path.is_symlink():
            msg = f"Latest outcome projection {result_path} exists without append-only history."
            raise ValueError(msg)
        return
    payload = f"{canonical_json(history[-1].to_dict())}\n".encode()
    try:
        matches = not result_path.is_symlink() and result_path.read_bytes() == payload
    except OSError:
        matches = False
    if not matches:
        atomic_write_bytes(result_path, payload)


def _schedule_resume_state(
    job: TrainingJob,
    controls: JobExecutionControls,
    existing: TrainingJobOutcome | None,
) -> TrainingScheduleResumeState | None:
    """Build the typed state passed to a strategy-aware executor.

    Returns:
        The schedule state, or ``None`` for legacy and sealed confirm jobs.
    """
    if job.strategy_schedule is None:
        return None
    return TrainingScheduleResumeState(
        strategy_schedule_checksum=job.strategy_schedule.content_checksum,
        schedule_id=job.strategy_schedule.schedule_id,
        resume_requested=controls.resume,
        overwrite_requested=controls.overwrite,
        prior_attempt=0 if existing is None else existing.attempt,
        prior_outcome_checksum=None if existing is None else existing.content_checksum,
        prior_status=None if existing is None else existing.status,
    )


def _validate_executor_registration(
    plan: TrainingRunPlan,
    executor: TrainingJobExecutor | TrainingExecutorRegistry,
) -> None:
    """Reject incomplete dispatch registration before output mutation.

    Raises:
        TypeError: If an executor is invalid or confirmation lacks its typed registry.
        ValueError: If a registry omits a planned implementation kind.
    """
    if isinstance(executor, TrainingExecutorRegistry):
        unsupported = tuple(job.implementation_kind for job in plan.jobs if not executor.supports(job))
        if unsupported:
            msg = f"Executor registry does not cover planned implementation kinds: {sorted(set(unsupported))}."
            raise ValueError(msg)
        return
    if not callable(executor):
        msg = "executor must be callable or a TrainingExecutorRegistry."
        raise TypeError(msg)
    if plan.preset == "paper-confirm":
        msg = "paper-confirm requires a TrainingExecutorRegistry with a typed confirm_executor."
        raise TypeError(msg)


def _preflight_existing_outcomes(
    plan: TrainingRunPlan,
    output_root: Path,
    controls: JobExecutionControls,
) -> None:
    """Validate the complete existing outcome universe without mutation.

    Raises:
        ValueError: If the output root, a job path, history, or selected control
            would make execution unsafe.
    """
    if output_root.is_symlink() or (output_root.exists() and not output_root.is_dir()):
        msg = "output_root must be absent or an existing non-symlink directory."
        raise ValueError(msg)
    lock_path = output_root / ".wp22-training-runner.lock"
    if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
        msg = "The WP22 runner lock must be absent or an existing non-symlink regular file."
        raise ValueError(msg)
    for job in plan.jobs:
        job_directory = output_root
        for component in Path(job.output_path).parts:
            job_directory /= component
            if job_directory.is_symlink() or (job_directory.exists() and not job_directory.is_dir()):
                msg = f"Job output path for {job.job_id} must be absent or a non-symlink directory."
                raise ValueError(msg)
        history = load_training_job_outcome_history(job_directory, job)
        result_path = job_directory / JOB_RESULT_NAME
        if result_path.is_symlink() or (result_path.exists() and not result_path.is_file()):
            msg = f"Latest outcome projection for {job.job_id} must be absent or a non-symlink regular file."
            raise ValueError(msg)
        if result_path.exists() and not history:
            msg = f"Latest outcome projection for {job.job_id} exists without append-only history."
            raise ValueError(msg)
        if plan.preset == "paper-confirm" and len(history) > 1:
            msg = f"paper-confirm job {job.job_id} has more than one terminal attempt."
            raise ValueError(msg)
        if history and not controls.resume and not controls.overwrite:
            msg = f"Outcome already exists for {job.job_id}; select resume or overwrite."
            raise ValueError(msg)


def execute_training_plan(
    plan: TrainingRunPlan,
    output_root: Path,
    executor: TrainingJobExecutor | TrainingExecutorRegistry,
    *,
    resume: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    fail_fast: bool = False,
    context: TrainingExecutionContext | ConfirmationExecutionContext | None = None,
    repository_root: Path | None = None,
) -> TrainingRunSummary:
    """Execute or dry-run a sealed plan with atomic per-job outcomes.

    Returns:
        Aggregate attempted, successful, failed, and skipped counts.

    Raises:
        TypeError: If the plan, output path, or executor has the wrong type.
        ValueError: If controls conflict or an existing outcome is incompatible.
    """
    if not isinstance(plan, TrainingRunPlan):
        msg = "plan must be a TrainingRunPlan."
        raise TypeError(msg)
    if not isinstance(output_root, Path):
        msg = "output_root must be a pathlib.Path."
        raise TypeError(msg)
    controls = JobExecutionControls(resume=resume, overwrite=overwrite)
    dry = require_bool(dry_run, "dry_run")
    stop_early = require_bool(fail_fast, "fail_fast")
    if context is not None:
        from .execution_context import (  # noqa: PLC0415 - avoids a module import cycle
            ConfirmationExecutionContext,
            TrainingExecutionContext,
        )

        if not isinstance(context, (TrainingExecutionContext, ConfirmationExecutionContext)):
            msg = "context must be a TrainingExecutionContext or ConfirmationExecutionContext."
            raise TypeError(msg)
        if context.plan != plan:
            msg = "Execution context does not contain the supplied exact plan."
            raise ValueError(msg)
        if not isinstance(repository_root, Path):
            msg = "repository_root is required with a TrainingExecutionContext."
            raise TypeError(msg)
        context.preflight(repository_root, output_root)
    if plan.preset == "paper-confirm" and controls.overwrite:
        msg = "paper-confirm forbids overwrite because its first terminal attempt is authoritative."
        raise ValueError(msg)
    _validate_executor_registration(plan, executor)
    _preflight_existing_outcomes(plan, output_root, controls)
    if dry:
        return TrainingRunSummary(len(plan.jobs), 0, 0, 0, 0)
    output_root.mkdir(parents=True, exist_ok=True)
    attempted = succeeded = failed = skipped = 0
    with FileLock(str(output_root / ".wp22-training-runner.lock")):
        for job in plan.jobs:
            job_directory = output_root / job.output_path
            existing: TrainingJobOutcome | None = None
            history = load_training_job_outcome_history(job_directory, job)
            _synchronize_latest_outcome(job_directory, history)
            if plan.preset == "paper-confirm" and len(history) > 1:
                msg = f"paper-confirm job {job.job_id} has more than one terminal attempt."
                raise ValueError(msg)
            if history:
                existing = history[-1]
                if controls.resume and (existing.status == "success" or plan.preset == "paper-confirm"):
                    skipped += 1
                    continue
                if not controls.resume and not controls.overwrite:
                    msg = f"Outcome already exists for {job.job_id}; select resume or overwrite."
                    raise ValueError(msg)
            attempt = 1 if existing is None else existing.attempt + 1
            attempted += 1
            job_controls = JobExecutionControls(
                resume=controls.resume,
                overwrite=controls.overwrite,
                schedule_resume_state=_schedule_resume_state(job, controls, existing),
            )
            try:
                dispatched = (
                    executor.dispatch(job, job_directory, job_controls)
                    if isinstance(executor, TrainingExecutorRegistry)
                    else executor(job, job_directory, job_controls)
                )
                result_checksum = require_checksum(
                    dispatched,
                    "executor result artifact checksum",
                )
                outcome = TrainingJobOutcome(
                    job_checksum=job.content_checksum,
                    status="success",
                    result_artifact_checksum=result_checksum,
                    exception_type=None,
                    message=None,
                    attempt=attempt,
                )
                succeeded += 1
            except Exception:  # noqa: BLE001 - executor boundary must redact and persist arbitrary ordinary failures
                outcome = TrainingJobOutcome(
                    job_checksum=job.content_checksum,
                    status="failure",
                    result_artifact_checksum=None,
                    exception_type="executor_failure",
                    message="executor failed; secret-bearing diagnostics are intentionally not persisted",
                    attempt=attempt,
                )
                failed += 1
            job_directory.mkdir(parents=True, exist_ok=True)
            _write_outcome_attempt(job_directory, outcome)
            _synchronize_latest_outcome(job_directory, (*history, outcome))
            if outcome.status == "failure" and stop_early:
                break
    return TrainingRunSummary(len(plan.jobs), attempted, succeeded, failed, skipped)


__all__ = [
    "CONFIRM_EXECUTION_REQUEST_SCHEMA_VERSION",
    "JOB_ATTEMPTS_DIRECTORY_NAME",
    "JOB_RESULT_NAME",
    "PILOT_OPTIMIZATION_SEED_COUNT",
    "RUNNABLE_DATA_ROLES",
    "TRAINING_JOB_OUTCOME_SCHEMA_VERSION",
    "TRAINING_JOB_SCHEMA_VERSION",
    "TRAINING_PRESETS",
    "TRAINING_RUN_PLAN_SCHEMA_VERSION",
    "ConfirmExecutionContext",
    "ConfirmExecutionRequest",
    "ConfirmJobExecutor",
    "JobExecutionControls",
    "TrainingExecutorRegistry",
    "TrainingJob",
    "TrainingJobExecutor",
    "TrainingJobOutcome",
    "TrainingRunPlan",
    "TrainingRunSummary",
    "TrainingScheduleResumeState",
    "build_confirm_execution_context",
    "build_historical_reproduction_plan",
    "build_paper_confirm_plan",
    "build_paper_pilot_plan",
    "build_paper_screen_plan",
    "build_training_smoke_plan",
    "confirmatory_evaluation_policy_checksum",
    "derive_confirmatory_evaluation_seed",
    "derive_confirmatory_optimization_seed",
    "derive_pilot_optimization_seeds",
    "execute_training_plan",
    "load_training_job_outcome_history",
    "reject_ballarin_training",
    "training_job_attempt_path",
    "validate_confirm_execution_request",
    "validate_job_pipeline_binding",
]
