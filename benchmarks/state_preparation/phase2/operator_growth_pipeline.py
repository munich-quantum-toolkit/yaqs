# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Legacy standalone, durable WP22 orchestration for operator-growth screening.

The operator-growth implementation is intentionally not disguised as a Phase
II training pipeline.  This wrapper binds its own exact implementation,
target, screening-cell, schedule, and source identities; measures execution
resources authoritatively; and emits the same verified promotion-row type as
the ordinary WP18-backed candidates. Final confirmation authorization accepts
only WP22E production custody, never this standalone artifact.
"""

# Strict artifact decoders delegate scalar failures to ``validation.py``.
# ruff: noqa: DOC201, DOC501

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from filelock import FileLock

from benchmarks.state_preparation.reporting import atomic_write_bytes

from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .operator_growth import ADAPT_STYLE_METHOD_ID, OperatorGrowthResult
from .protocol import ScreeningCell
from .result_custody import TrajectoryFidelityEvidence
from .screening import (
    OperatorGrowthScreeningTemplate,
    VerifiedScreeningOutcome,
    WP22CandidateConfiguration,
    screening_trajectory_context_checksum,
)
from .validation import (
    require_bool,
    require_checksum,
    require_float,
    require_git_commit,
    require_int,
    require_nonempty_text,
    require_slug,
)
from .wp20_resources import CircuitResourceMetrics, WP20WorkLedger

OPERATOR_GROWTH_PIPELINE_REQUEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_pipeline_request.v1"
OUTER_SCREENING_EVALUATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.outer_screening_evaluation.v1"
OUTER_SCREENING_ENSEMBLE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.outer_screening_ensemble.v1"
OPERATOR_GROWTH_PIPELINE_ARTIFACT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_pipeline_artifact.v1"
OPERATOR_GROWTH_MATERIALIZATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.operator_growth_materialization.v1"

OPERATOR_GROWTH_ARTIFACT_NAME = "operator_growth_pipeline_artifact.json"
OPERATOR_GROWTH_LOCK_NAME = ".operator-growth-pipeline.lock"

_REQUEST_KEYS = frozenset({
    "schema_version",
    "request_id",
    "template",
    "candidate",
    "cell",
    "target_manifest_checksum",
    "target_spec_checksum",
    "target_vector_checksum",
    "strategy_schedule_checksum",
    "screening_evaluation_policy_checksum",
    "execution_source_manifest_checksum",
    "tracked_source_manifest_checksum",
    "source_commit",
    "optimization_block_id",
    "outer_evaluation_trajectory_count",
    "content_checksum",
})
_EVALUATION_KEYS = frozenset({
    "schema_version",
    "request_checksum",
    "candidate_configuration_checksum",
    "operator_growth_result_schema_version",
    "operator_growth_result_checksum",
    "cell_id",
    "target_instance_id",
    "data_role",
    "evaluation_seed",
    "trajectory_count",
    "trajectory_ensemble_checksum",
    "evaluation_policy_checksum",
    "trajectory_evidence",
    "circuit_resources_checksum",
    "work_ledger",
    "content_checksum",
})
_ARTIFACT_KEYS = frozenset({
    "schema_version",
    "request",
    "request_checksum",
    "attempt",
    "status",
    "failure_phase",
    "exception_type",
    "message",
    "operator_growth_result",
    "outer_evaluation",
    "circuit_resources",
    "training_work",
    "evaluation_work",
    "total_work",
    "materialization_checksum",
    "verified_outcome",
    "content_checksum",
})

FailurePhase = Literal["operator_growth_execution", "screening_evaluation"]


def _same_float(left: float, right: float) -> bool:
    """Return whether two finite floats have identical binary64 values."""
    return float(left).hex() == float(right).hex()


def _uint64(value: object, name: str) -> int:
    """Return one strict unsigned 64-bit integer."""
    result = require_int(value, name)
    if result >= 2**64:
        msg = f"{name} must fit an unsigned 64-bit integer."
        raise ValueError(msg)
    return result


def _ledger_with_measurement(base: WP20WorkLedger, wall_time: float, peak_memory: int) -> WP20WorkLedger:
    """Attach authoritative measurements while retaining exact work counters."""
    if not isinstance(base, WP20WorkLedger):
        msg = "base must be a WP20WorkLedger."
        raise TypeError(msg)
    return WP20WorkLedger(
        forward_circuit_evaluations=base.forward_circuit_evaluations,
        backward_circuit_evaluations=base.backward_circuit_evaluations,
        trajectory_gate_applications=base.trajectory_gate_applications,
        training_trajectories=base.training_trajectories,
        checkpoint_validation_trajectories=base.checkpoint_validation_trajectories,
        test_trajectories=base.test_trajectories,
        objective_calls=base.objective_calls,
        gradient_calls=base.gradient_calls,
        cross_trajectory_pairings=base.cross_trajectory_pairings,
        wall_time_seconds=require_float(wall_time, "wall_time", minimum=0.0),
        peak_memory_bytes=require_int(peak_memory, "peak_memory"),
    )


def _sum_ledgers(left: WP20WorkLedger, right: WP20WorkLedger) -> WP20WorkLedger:
    """Add work and elapsed time while retaining the maximum memory high-water mark."""
    return WP20WorkLedger(
        forward_circuit_evaluations=left.forward_circuit_evaluations + right.forward_circuit_evaluations,
        backward_circuit_evaluations=left.backward_circuit_evaluations + right.backward_circuit_evaluations,
        trajectory_gate_applications=left.trajectory_gate_applications + right.trajectory_gate_applications,
        training_trajectories=left.training_trajectories + right.training_trajectories,
        checkpoint_validation_trajectories=(
            left.checkpoint_validation_trajectories + right.checkpoint_validation_trajectories
        ),
        test_trajectories=left.test_trajectories + right.test_trajectories,
        objective_calls=left.objective_calls + right.objective_calls,
        gradient_calls=left.gradient_calls + right.gradient_calls,
        cross_trajectory_pairings=left.cross_trajectory_pairings + right.cross_trajectory_pairings,
        wall_time_seconds=left.wall_time_seconds + right.wall_time_seconds,
        peak_memory_bytes=max(left.peak_memory_bytes, right.peak_memory_bytes),
    )


@dataclass(frozen=True, slots=True)
class _MeasuredCall:
    """Internal result, ordinary failure, and authoritative process measurements."""

    value: object | None
    error: Exception | None
    wall_time_seconds: float
    peak_memory_bytes: int


def _measured_call(callback: Callable[[], object]) -> _MeasuredCall:
    """Execute a callback under monotonic time and Python allocation tracing."""
    owns_tracing = not tracemalloc.is_tracing()
    if owns_tracing:
        tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    if owns_tracing:
        tracemalloc.reset_peak()
        baseline = 0
    start = time.perf_counter()
    try:  # noqa: PLW0717 - timing and tracing cleanup must cover the callback as one atomic measurement
        try:
            value = callback()
            error: Exception | None = None
        except Exception as caught:  # noqa: BLE001 - ordinary failures become durable artifacts
            value = None
            error = caught
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
    finally:
        if owns_tracing:
            tracemalloc.stop()
    return _MeasuredCall(
        value=value,
        error=error,
        wall_time_seconds=float(max(elapsed, 0.0)),
        peak_memory_bytes=max(current - baseline, peak - baseline, 0),
    )


@dataclass(frozen=True, slots=True)
class OperatorGrowthPipelineRequest:
    """Complete standalone execution identity for one outer screening cell."""

    request_id: str
    template: OperatorGrowthScreeningTemplate
    candidate: WP22CandidateConfiguration
    cell: ScreeningCell
    target_manifest_checksum: str
    target_spec_checksum: str
    target_vector_checksum: str
    strategy_schedule_checksum: str
    screening_evaluation_policy_checksum: str
    execution_source_manifest_checksum: str
    tracked_source_manifest_checksum: str
    source_commit: str
    optimization_block_id: str
    outer_evaluation_trajectory_count: int
    schema_version: str = field(default=OPERATOR_GROWTH_PIPELINE_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate every implementation, target, schedule, cell, and source link."""
        object.__setattr__(self, "request_id", require_slug(self.request_id, "request_id"))
        if not isinstance(self.template, OperatorGrowthScreeningTemplate):
            msg = "template must be an OperatorGrowthScreeningTemplate."
            raise TypeError(msg)
        if not isinstance(self.candidate, WP22CandidateConfiguration):
            msg = "candidate must be a WP22CandidateConfiguration."
            raise TypeError(msg)
        if not isinstance(self.cell, ScreeningCell):
            msg = "cell must be a ScreeningCell."
            raise TypeError(msg)
        if (
            self.candidate.implementation_kind != "operator_growth"
            or self.candidate.method_id != ADAPT_STYLE_METHOD_ID
            or self.candidate.implementation_method_id != ADAPT_STYLE_METHOD_ID
            or not self.candidate.noisy_training
            or self.candidate.implementation_schema_version != self.template.schema_version
            or self.candidate.implementation_checksum != self.template.content_checksum
        ):
            msg = "candidate is not the noisy promotion-eligible implementation of the supplied template."
            raise ValueError(msg)
        if (
            self.template.pool_policy_id != "nearest_neighbor_pool"
            or self.template.growth_policy_id != "largest_projector_gradient"
        ):
            msg = "Standalone WP22 operator growth requires the registered projector-pool growth policy."
            raise ValueError(msg)
        for name in (
            "target_manifest_checksum",
            "target_spec_checksum",
            "target_vector_checksum",
            "strategy_schedule_checksum",
            "screening_evaluation_policy_checksum",
            "execution_source_manifest_checksum",
            "tracked_source_manifest_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        if self.strategy_schedule_checksum != self.candidate.strategy_schedule_checksum:
            msg = "strategy_schedule_checksum does not match the candidate configuration."
            raise ValueError(msg)
        object.__setattr__(self, "source_commit", require_git_commit(self.source_commit, "source_commit"))
        object.__setattr__(
            self,
            "optimization_block_id",
            require_slug(self.optimization_block_id, "optimization_block_id"),
        )
        object.__setattr__(
            self,
            "outer_evaluation_trajectory_count",
            require_int(
                self.outer_evaluation_trajectory_count,
                "outer_evaluation_trajectory_count",
                minimum=1,
            ),
        )
        if self.cell.data_role != "screening_selection":
            msg = "Standalone operator growth is authorized only for outer screening_selection cells."
            raise ValueError(msg)

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered request fields."""
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "template": self.template.to_dict(),
            "candidate": self.candidate.to_dict(),
            "cell": self.cell.to_dict(),
            "target_manifest_checksum": self.target_manifest_checksum,
            "target_spec_checksum": self.target_spec_checksum,
            "target_vector_checksum": self.target_vector_checksum,
            "strategy_schedule_checksum": self.strategy_schedule_checksum,
            "screening_evaluation_policy_checksum": self.screening_evaluation_policy_checksum,
            "execution_source_manifest_checksum": self.execution_source_manifest_checksum,
            "tracked_source_manifest_checksum": self.tracked_source_manifest_checksum,
            "source_commit": self.source_commit,
            "optimization_block_id": self.optimization_block_id,
            "outer_evaluation_trajectory_count": self.outer_evaluation_trajectory_count,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering every standalone execution identity."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native request data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed request JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> OperatorGrowthPipelineRequest:
        """Decode and checksum-verify one strict standalone request."""
        mapping = verify_sealed_mapping(data, expected_keys=_REQUEST_KEYS, name="operator-growth pipeline request")
        if mapping["schema_version"] != OPERATOR_GROWTH_PIPELINE_REQUEST_SCHEMA_VERSION:
            msg = "Operator-growth pipeline request uses an unsupported schema version."
            raise ValueError(msg)
        request = cls(
            request_id=cast("str", mapping["request_id"]),
            template=OperatorGrowthScreeningTemplate.from_dict(mapping["template"]),
            candidate=WP22CandidateConfiguration.from_dict(mapping["candidate"]),
            cell=ScreeningCell.from_dict(mapping["cell"]),
            target_manifest_checksum=cast("str", mapping["target_manifest_checksum"]),
            target_spec_checksum=cast("str", mapping["target_spec_checksum"]),
            target_vector_checksum=cast("str", mapping["target_vector_checksum"]),
            strategy_schedule_checksum=cast("str", mapping["strategy_schedule_checksum"]),
            screening_evaluation_policy_checksum=cast("str", mapping["screening_evaluation_policy_checksum"]),
            execution_source_manifest_checksum=cast("str", mapping["execution_source_manifest_checksum"]),
            tracked_source_manifest_checksum=cast("str", mapping["tracked_source_manifest_checksum"]),
            source_commit=cast("str", mapping["source_commit"]),
            optimization_block_id=cast("str", mapping["optimization_block_id"]),
            outer_evaluation_trajectory_count=cast("int", mapping["outer_evaluation_trajectory_count"]),
        )
        if mapping["content_checksum"] != request.content_checksum:
            msg = "Operator-growth request checksum changed during normalization."
            raise ValueError(msg)
        return request

    @classmethod
    def from_json(cls, payload: str) -> OperatorGrowthPipelineRequest:
        """Decode canonical request JSON."""
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class OuterScreeningEvaluation:
    """Fresh raw-trajectory outer-screening evaluation with exact test work."""

    request_checksum: str
    candidate_configuration_checksum: str
    operator_growth_result_schema_version: str
    operator_growth_result_checksum: str
    cell_id: str
    target_instance_id: str
    data_role: str
    evaluation_seed: int
    trajectory_count: int
    trajectory_ensemble_checksum: str
    evaluation_policy_checksum: str
    trajectory_evidence: TrajectoryFidelityEvidence
    circuit_resources_checksum: str
    work_ledger: WP20WorkLedger
    schema_version: str = field(default=OUTER_SCREENING_EVALUATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate role, seed, raw trajectories, and caller-supplied work."""
        for name in (
            "request_checksum",
            "candidate_configuration_checksum",
            "operator_growth_result_checksum",
            "trajectory_ensemble_checksum",
            "evaluation_policy_checksum",
            "circuit_resources_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(
            self,
            "operator_growth_result_schema_version",
            require_slug(self.operator_growth_result_schema_version, "operator_growth_result_schema_version"),
        )
        object.__setattr__(self, "cell_id", require_slug(self.cell_id, "cell_id"))
        object.__setattr__(
            self,
            "target_instance_id",
            require_slug(self.target_instance_id, "target_instance_id"),
        )
        if self.data_role != "screening_selection":
            msg = "Outer operator-growth evaluation must use screening_selection, never final-test data."
            raise ValueError(msg)
        object.__setattr__(self, "evaluation_seed", _uint64(self.evaluation_seed, "evaluation_seed"))
        count = require_int(self.trajectory_count, "trajectory_count", minimum=1)
        object.__setattr__(self, "trajectory_count", count)
        if not isinstance(self.trajectory_evidence, TrajectoryFidelityEvidence):
            msg = "trajectory_evidence must be raw TrajectoryFidelityEvidence."
            raise TypeError(msg)
        expected_context = screening_trajectory_context_checksum(
            candidate_configuration_checksum=self.candidate_configuration_checksum,
            cell_id=self.cell_id,
            result_schema_version=self.operator_growth_result_schema_version,
            result_record_checksum=self.operator_growth_result_checksum,
        )
        if (
            self.trajectory_evidence.evaluation_context_checksum != expected_context
            or self.trajectory_evidence.data_role != "screening_selection"
            or self.trajectory_evidence.evaluation_seed != self.evaluation_seed
            or len(self.trajectory_evidence.trajectory_fidelities) != count
        ):
            msg = "Raw outer-screening trajectories differ from the exact request, result, role, seed, or budget."
            raise ValueError(msg)
        if not isinstance(self.work_ledger, WP20WorkLedger):
            msg = "work_ledger must be a WP20WorkLedger."
            raise TypeError(msg)
        work = self.work_ledger
        if not _same_float(work.wall_time_seconds, 0.0) or work.peak_memory_bytes != 0:
            msg = "Evaluation callbacks cannot claim runtime or peak memory; the wrapper measures both."
            raise ValueError(msg)
        if (
            work.forward_circuit_evaluations != count
            or work.backward_circuit_evaluations != 0
            or work.training_trajectories != 0
            or work.checkpoint_validation_trajectories != 0
            or work.test_trajectories != count
            or work.objective_calls != 1
            or work.gradient_calls != 0
            or work.cross_trajectory_pairings != 0
        ):
            msg = "Outer evaluation work must describe one fresh aggregate test-trajectory objective."
            raise ValueError(msg)

    @classmethod
    def create(
        cls,
        request: OperatorGrowthPipelineRequest,
        result: OperatorGrowthResult,
        *,
        trajectory_fidelities: Sequence[float],
        trajectory_ensemble_checksum: str | None = None,
    ) -> OuterScreeningEvaluation:
        """Create exact outer evaluation evidence for the request's fixed budget."""
        if not isinstance(request, OperatorGrowthPipelineRequest):
            msg = "request must be an OperatorGrowthPipelineRequest."
            raise TypeError(msg)
        if not isinstance(result, OperatorGrowthResult) or result.circuit_resources is None:
            msg = "result must be a completed OperatorGrowthResult with circuit resources."
            raise TypeError(msg)
        count = request.outer_evaluation_trajectory_count
        expected_ensemble = derive_outer_screening_ensemble_checksum(request, result)
        if trajectory_ensemble_checksum is not None and trajectory_ensemble_checksum != expected_ensemble:
            msg = "trajectory_ensemble_checksum does not reproduce the request's screening role and seed."
            raise ValueError(msg)
        return cls(
            request_checksum=request.content_checksum,
            candidate_configuration_checksum=request.candidate.content_checksum,
            operator_growth_result_schema_version=result.schema_version,
            operator_growth_result_checksum=result.content_checksum,
            cell_id=request.cell.cell_id,
            target_instance_id=request.cell.target_instance_id,
            data_role="screening_selection",
            evaluation_seed=request.cell.screening_seed,
            trajectory_count=count,
            trajectory_ensemble_checksum=expected_ensemble,
            evaluation_policy_checksum=request.screening_evaluation_policy_checksum,
            trajectory_evidence=TrajectoryFidelityEvidence(
                evaluation_context_checksum=screening_trajectory_context_checksum(
                    candidate_configuration_checksum=request.candidate.content_checksum,
                    cell_id=request.cell.cell_id,
                    result_schema_version=result.schema_version,
                    result_record_checksum=result.content_checksum,
                ),
                data_role="screening_selection",
                evaluation_seed=request.cell.screening_seed,
                trajectory_fidelities=tuple(trajectory_fidelities),
            ),
            circuit_resources_checksum=result.circuit_resources.content_checksum,
            work_ledger=WP20WorkLedger(
                forward_circuit_evaluations=count,
                trajectory_gate_applications=count * len(result.circuit_resources.logical_events),
                test_trajectories=count,
                objective_calls=1,
            ),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered outer-evaluation field."""
        return {
            "schema_version": self.schema_version,
            "request_checksum": self.request_checksum,
            "candidate_configuration_checksum": self.candidate_configuration_checksum,
            "operator_growth_result_schema_version": self.operator_growth_result_schema_version,
            "operator_growth_result_checksum": self.operator_growth_result_checksum,
            "cell_id": self.cell_id,
            "target_instance_id": self.target_instance_id,
            "data_role": self.data_role,
            "evaluation_seed": self.evaluation_seed,
            "trajectory_count": self.trajectory_count,
            "trajectory_ensemble_checksum": self.trajectory_ensemble_checksum,
            "evaluation_policy_checksum": self.evaluation_policy_checksum,
            "trajectory_evidence": self.trajectory_evidence.to_dict(),
            "circuit_resources_checksum": self.circuit_resources_checksum,
            "work_ledger": self.work_ledger.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the fresh aggregate outer evaluation."""
        return canonical_checksum(self._content_dict())

    @property
    def noisy_fidelity(self) -> float:
        """Mean mechanically derived from raw trajectory evidence."""
        return self.trajectory_evidence.mean_fidelity

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native evaluation data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> OuterScreeningEvaluation:
        """Decode and checksum-verify strict outer-evaluation evidence."""
        mapping = verify_sealed_mapping(data, expected_keys=_EVALUATION_KEYS, name="outer screening evaluation")
        if mapping["schema_version"] != OUTER_SCREENING_EVALUATION_SCHEMA_VERSION:
            msg = "Outer screening evaluation uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            request_checksum=cast("str", mapping["request_checksum"]),
            candidate_configuration_checksum=cast("str", mapping["candidate_configuration_checksum"]),
            operator_growth_result_schema_version=cast("str", mapping["operator_growth_result_schema_version"]),
            operator_growth_result_checksum=cast("str", mapping["operator_growth_result_checksum"]),
            cell_id=cast("str", mapping["cell_id"]),
            target_instance_id=cast("str", mapping["target_instance_id"]),
            data_role=cast("str", mapping["data_role"]),
            evaluation_seed=cast("int", mapping["evaluation_seed"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            trajectory_ensemble_checksum=cast("str", mapping["trajectory_ensemble_checksum"]),
            evaluation_policy_checksum=cast("str", mapping["evaluation_policy_checksum"]),
            trajectory_evidence=TrajectoryFidelityEvidence.from_dict(mapping["trajectory_evidence"]),
            circuit_resources_checksum=cast("str", mapping["circuit_resources_checksum"]),
            work_ledger=WP20WorkLedger.from_dict(mapping["work_ledger"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Outer screening evaluation checksum changed during normalization."
            raise ValueError(msg)
        return result


def derive_outer_screening_ensemble_checksum(
    request: OperatorGrowthPipelineRequest,
    result: OperatorGrowthResult,
) -> str:
    """Derive the exact outer ensemble identity from role, seed, budget, and circuit."""
    if not isinstance(request, OperatorGrowthPipelineRequest):
        msg = "request must be an OperatorGrowthPipelineRequest."
        raise TypeError(msg)
    if not isinstance(result, OperatorGrowthResult) or result.circuit_resources is None:
        msg = "result must be a completed OperatorGrowthResult with circuit resources."
        raise TypeError(msg)
    return canonical_checksum({
        "schema_version": OUTER_SCREENING_ENSEMBLE_SCHEMA_VERSION,
        "request_checksum": request.content_checksum,
        "operator_growth_result_checksum": result.content_checksum,
        "target_instance_id": request.cell.target_instance_id,
        "target_vector_checksum": request.target_vector_checksum,
        "data_role": "screening_selection",
        "evaluation_seed": request.cell.screening_seed,
        "trajectory_count": request.outer_evaluation_trajectory_count,
        "evaluation_policy_checksum": request.screening_evaluation_policy_checksum,
        "circuit_resources_checksum": result.circuit_resources.content_checksum,
    })


def _validate_operator_result(request: OperatorGrowthPipelineRequest, result: OperatorGrowthResult) -> None:
    """Require the exact noisy, target-bound, promotion-eligible template result."""
    if not isinstance(result, OperatorGrowthResult):
        msg = "Operator-growth callback must return an OperatorGrowthResult."
        raise TypeError(msg)
    if (
        result.status != "completed"
        or result.method_id != ADAPT_STYLE_METHOD_ID
        or result.execution_mode != "noisy_training"
        or not result.promotion_eligible
        or result.training_provenance is None
        or result.evaluator_binding is None
        or result.growth_spec is None
        or result.pool is None
        or result.circuit_resources is None
    ):
        msg = "Operator-growth callback did not return completed promotion-eligible noisy adapt-style evidence."
        raise ValueError(msg)
    provenance = result.training_provenance
    binding = result.evaluator_binding
    spec = result.growth_spec
    resources = result.circuit_resources
    template = request.template
    if (
        provenance.target_instance_id != request.cell.target_instance_id
        or provenance.target_family_id != request.cell.family_id
        or provenance.qubit_count != request.cell.qubit_count
        or provenance.target_vector_checksum != request.target_vector_checksum
        or provenance.optimization_block_id != request.optimization_block_id
        or provenance.optimization_seed != request.cell.optimization_seed
        or provenance.resource_stratum_id != request.candidate.resource_stratum_id
        or provenance.trajectory_count != template.training_trajectory_count
        or binding.target_instance_spec_checksum != request.target_spec_checksum
        or binding.target_manifest_checksum != request.target_manifest_checksum
        or binding.target_stratum_id != request.cell.stratum_id
        or resources.qubit_count != request.cell.qubit_count
    ):
        msg = "Operator-growth result target, cell, optimization, or training identity differs from the request."
        raise ValueError(msg)
    cap = spec.native_two_qubit_cap_per_edge
    if (
        spec.max_operators != template.max_operators
        or spec.reoptimization_steps != template.reoptimization_steps
        or not _same_float(spec.gradient_tolerance, template.gradient_threshold)
        or cap is None
        or not _same_float(float(cap), template.native_two_qubit_cap_per_edge)
    ):
        msg = "Operator-growth result does not execute the exact sealed screening template."
        raise ValueError(msg)
    attained = max(resources.native_two_qubit_gates_per_chain_edge, default=0)
    if attained > template.native_two_qubit_cap_per_edge:
        msg = "Operator-growth result exceeds the template's native two-qubit cap."
        raise ValueError(msg)


def _validate_outer_evaluation(
    request: OperatorGrowthPipelineRequest,
    result: OperatorGrowthResult,
    evaluation: OuterScreeningEvaluation,
) -> None:
    """Require fresh role-isolated outer evidence for the exact result."""
    if not isinstance(evaluation, OuterScreeningEvaluation):
        msg = "Evaluation callback must return an OuterScreeningEvaluation."
        raise TypeError(msg)
    resources = result.circuit_resources
    provenance = result.training_provenance
    if resources is None or provenance is None:
        msg = "Outer evaluation requires a complete noisy operator-growth result."
        raise ValueError(msg)
    if (
        evaluation.request_checksum != request.content_checksum
        or evaluation.candidate_configuration_checksum != request.candidate.content_checksum
        or evaluation.operator_growth_result_schema_version != result.schema_version
        or evaluation.operator_growth_result_checksum != result.content_checksum
        or evaluation.cell_id != request.cell.cell_id
        or evaluation.target_instance_id != request.cell.target_instance_id
        or evaluation.data_role != "screening_selection"
        or evaluation.evaluation_seed != request.cell.screening_seed
        or evaluation.trajectory_count != request.outer_evaluation_trajectory_count
        or evaluation.evaluation_policy_checksum != request.screening_evaluation_policy_checksum
        or evaluation.circuit_resources_checksum != resources.content_checksum
    ):
        msg = "Outer evaluation role, seed, target, result, resources, or budget differs from the request."
        raise ValueError(msg)
    expected_ensemble = derive_outer_screening_ensemble_checksum(request, result)
    if evaluation.trajectory_ensemble_checksum != expected_ensemble:
        msg = "Outer evaluation ensemble does not reproduce its screening role, seed, policy, and budget."
        raise ValueError(msg)
    if evaluation.trajectory_ensemble_checksum == provenance.trajectory_ensemble_checksum:
        msg = "Outer screening evaluation must use a fresh ensemble independent of noisy training."
        raise ValueError(msg)
    expected_gate_applications = evaluation.trajectory_count * len(resources.logical_events)
    if evaluation.work_ledger.trajectory_gate_applications != expected_gate_applications:
        msg = "Outer evaluation trajectory-gate work does not match the materialized exact circuit."
        raise ValueError(msg)


def _materialization_checksum(result: OperatorGrowthResult) -> str:
    """Return a source-addressed identity for the exactly materializable result circuit."""
    resources = result.circuit_resources
    if resources is None:
        msg = "Completed operator growth requires circuit resources."
        raise ValueError(msg)
    return canonical_checksum({
        "schema_version": OPERATOR_GROWTH_MATERIALIZATION_SCHEMA_VERSION,
        "operator_growth_result_checksum": result.content_checksum,
        "selected_operator_ids": list(result.selected_operator_ids),
        "parameters": list(result.parameters),
        "logical_circuit_checksum": resources.logical_circuit_checksum,
        "native_circuit_checksum": resources.native_circuit_checksum,
        "circuit_resources_checksum": resources.content_checksum,
    })


def _failure_record_checksum(
    request: OperatorGrowthPipelineRequest,
    attempt: int,
    phase: FailurePhase,
    exception_type: str,
    message: str,
) -> str:
    """Return a stable record checksum independent of the enclosing artifact checksum."""
    return canonical_checksum({
        "schema_version": OPERATOR_GROWTH_PIPELINE_ARTIFACT_SCHEMA_VERSION,
        "request_checksum": request.content_checksum,
        "attempt": attempt,
        "status": "failure",
        "failure_phase": phase,
        "exception_type": exception_type,
        "message": message,
    })


def _expected_outcome(
    *,
    request: OperatorGrowthPipelineRequest,
    attempt: int,
    status: Literal["success", "failure"],
    failure_phase: FailurePhase | None,
    exception_type: str | None,
    message: str | None,
    result: OperatorGrowthResult | None,
    evaluation: OuterScreeningEvaluation | None,
    resources: CircuitResourceMetrics | None,
    work: WP20WorkLedger,
    materialization_checksum: str | None,
) -> VerifiedScreeningOutcome:
    """Mechanically construct the shared promotion-ready screening row."""
    if status == "success":
        if result is None or evaluation is None or resources is None or materialization_checksum is None:
            msg = "Successful promotion outcome requires complete result, evaluation, and materialization evidence."
            raise ValueError(msg)
        counts = resources.native_two_qubit_gates_per_chain_edge
        return VerifiedScreeningOutcome(
            candidate_configuration_checksum=request.candidate.content_checksum,
            cell_id=request.cell.cell_id,
            data_role="screening_selection",
            evaluation_seed=request.cell.screening_seed,
            result_schema_version=result.schema_version,
            result_record_checksum=result.content_checksum,
            evaluation_evidence_checksum=evaluation.content_checksum,
            materialization_checksum=materialization_checksum,
            status="success",
            noisy_fidelity=evaluation.noisy_fidelity,
            resource_value=float(max(counts, default=0)),
            normalized_work=work.normalized_compute(),
            failure_code=None,
            circuit_resources=resources,
            work_ledger=work,
        )
    if failure_phase is None or exception_type is None or message is None:
        msg = "Failed promotion outcome requires complete failure evidence."
        raise ValueError(msg)
    return VerifiedScreeningOutcome(
        candidate_configuration_checksum=request.candidate.content_checksum,
        cell_id=request.cell.cell_id,
        data_role="screening_selection",
        evaluation_seed=request.cell.screening_seed,
        result_schema_version=OPERATOR_GROWTH_PIPELINE_ARTIFACT_SCHEMA_VERSION,
        result_record_checksum=_failure_record_checksum(
            request,
            attempt,
            failure_phase,
            exception_type,
            message,
        ),
        evaluation_evidence_checksum=None,
        materialization_checksum=materialization_checksum,
        status="failure",
        noisy_fidelity=None,
        resource_value=None,
        normalized_work=work.normalized_compute(),
        failure_code=failure_phase,
        circuit_resources=None,
        work_ledger=work,
    )


@dataclass(frozen=True, slots=True)
class OperatorGrowthPipelineArtifact:
    """Durable success or failure from one authoritative standalone attempt."""

    request: OperatorGrowthPipelineRequest
    request_checksum: str
    attempt: int
    status: Literal["success", "failure"]
    failure_phase: FailurePhase | None
    exception_type: str | None
    message: str | None
    operator_growth_result: OperatorGrowthResult | None
    outer_evaluation: OuterScreeningEvaluation | None
    circuit_resources: CircuitResourceMetrics | None
    training_work: WP20WorkLedger
    evaluation_work: WP20WorkLedger
    total_work: WP20WorkLedger
    materialization_checksum: str | None
    verified_outcome: VerifiedScreeningOutcome
    schema_version: str = field(default=OPERATOR_GROWTH_PIPELINE_ARTIFACT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Replay source links, measurements, work, resources, and promotion row."""
        if not isinstance(self.request, OperatorGrowthPipelineRequest):
            msg = "request must be an OperatorGrowthPipelineRequest."
            raise TypeError(msg)
        checksum = require_checksum(self.request_checksum, "request_checksum")
        if checksum != self.request.content_checksum:
            msg = "request_checksum does not identify the nested exact request."
            raise ValueError(msg)
        object.__setattr__(self, "request_checksum", checksum)
        attempt = require_int(self.attempt, "attempt", minimum=1)
        if attempt != 1:
            msg = "Operator-growth pipeline custody accepts only the authoritative first attempt."
            raise ValueError(msg)
        object.__setattr__(self, "attempt", attempt)
        if self.status not in {"success", "failure"}:
            msg = "status must be success or failure."
            raise ValueError(msg)
        for name in ("training_work", "evaluation_work", "total_work"):
            if not isinstance(getattr(self, name), WP20WorkLedger):
                msg = f"{name} must be a WP20WorkLedger."
                raise TypeError(msg)
        if self.total_work != _sum_ledgers(self.training_work, self.evaluation_work):
            msg = "total_work must be the exact additive training/evaluation ledger."
            raise ValueError(msg)
        if not isinstance(self.verified_outcome, VerifiedScreeningOutcome):
            msg = "verified_outcome must be a VerifiedScreeningOutcome."
            raise TypeError(msg)

        result = self.operator_growth_result
        evaluation = self.outer_evaluation
        resources = self.circuit_resources
        materialization = self.materialization_checksum
        if result is not None:
            if not isinstance(result, OperatorGrowthResult):
                msg = "operator_growth_result must be an OperatorGrowthResult."
                raise TypeError(msg)
            rejected_training_result = self.status == "failure" and self.failure_phase == "operator_growth_execution"
            if rejected_training_result:
                if resources is not None or materialization is not None:
                    msg = "A request-incompatible training result cannot claim usable circuit materialization."
                    raise ValueError(msg)
            else:
                _validate_operator_result(self.request, result)
                if resources != result.circuit_resources:
                    msg = "circuit_resources must exactly equal the result's compiler-derived resources."
                    raise ValueError(msg)
            expected_training = _ledger_with_measurement(
                result.wp20_work,
                self.training_work.wall_time_seconds,
                self.training_work.peak_memory_bytes,
            )
            if self.training_work != expected_training:
                msg = "training_work counters do not reproduce the exact operator-growth result."
                raise ValueError(msg)
            if not rejected_training_result:
                expected_materialization = _materialization_checksum(result)
                if materialization != expected_materialization:
                    msg = "materialization_checksum does not reproduce the exact result circuit."
                    raise ValueError(msg)
        elif resources is not None or materialization is not None:
            msg = "A missing operator result cannot claim circuit resources or materialization."
            raise ValueError(msg)
        else:
            expected_training = _ledger_with_measurement(
                WP20WorkLedger(),
                self.training_work.wall_time_seconds,
                self.training_work.peak_memory_bytes,
            )
            if self.training_work != expected_training:
                msg = "A failed callback without a result cannot claim completed algorithmic work."
                raise ValueError(msg)

        if evaluation is not None:
            if result is None:
                msg = "Outer evaluation cannot exist without an operator-growth result."
                raise ValueError(msg)
            _validate_outer_evaluation(self.request, result, evaluation)
            expected_evaluation = _ledger_with_measurement(
                evaluation.work_ledger,
                self.evaluation_work.wall_time_seconds,
                self.evaluation_work.peak_memory_bytes,
            )
            if self.evaluation_work != expected_evaluation:
                msg = "evaluation_work counters do not reproduce the exact outer evidence."
                raise ValueError(msg)
        else:
            expected_evaluation = _ledger_with_measurement(
                WP20WorkLedger(),
                self.evaluation_work.wall_time_seconds,
                self.evaluation_work.peak_memory_bytes,
            )
            if self.evaluation_work != expected_evaluation:
                msg = "An incomplete outer callback cannot claim completed evaluation work."
                raise ValueError(msg)

        if self.status == "success":
            if (
                self.failure_phase is not None
                or self.exception_type is not None
                or self.message is not None
                or result is None
                or evaluation is None
                or resources is None
                or materialization is None
            ):
                msg = "Successful artifacts require complete evidence and no failure fields."
                raise ValueError(msg)
        else:
            if self.failure_phase not in {"operator_growth_execution", "screening_evaluation"}:
                msg = "Failed artifacts require an exact failure phase."
                raise ValueError(msg)
            if self.exception_type is None or self.message is None or evaluation is not None:
                msg = "Failed artifacts require exception details and cannot claim completed evaluation."
                raise ValueError(msg)
            object.__setattr__(
                self,
                "exception_type",
                require_nonempty_text(self.exception_type, "exception_type"),
            )
            object.__setattr__(self, "message", require_nonempty_text(self.message, "message"))
            if self.failure_phase == "screening_evaluation" and result is None:
                msg = "Screening-evaluation failures require the completed training result."
                raise ValueError(msg)

        expected_outcome = _expected_outcome(
            request=self.request,
            attempt=self.attempt,
            status=self.status,
            failure_phase=self.failure_phase,
            exception_type=self.exception_type,
            message=self.message,
            result=result,
            evaluation=evaluation,
            resources=resources,
            work=self.total_work,
            materialization_checksum=materialization,
        )
        if self.verified_outcome != expected_outcome:
            msg = "verified_outcome is not the mechanically derived source-linked promotion row."
            raise ValueError(msg)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered durable artifact field."""
        return {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "request_checksum": self.request_checksum,
            "attempt": self.attempt,
            "status": self.status,
            "failure_phase": self.failure_phase,
            "exception_type": self.exception_type,
            "message": self.message,
            "operator_growth_result": (
                None if self.operator_growth_result is None else self.operator_growth_result.to_dict()
            ),
            "outer_evaluation": None if self.outer_evaluation is None else self.outer_evaluation.to_dict(),
            "circuit_resources": None if self.circuit_resources is None else self.circuit_resources.to_dict(),
            "training_work": self.training_work.to_dict(),
            "evaluation_work": self.evaluation_work.to_dict(),
            "total_work": self.total_work.to_dict(),
            "materialization_checksum": self.materialization_checksum,
            "verified_outcome": self.verified_outcome.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete standalone attempt."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native artifact data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed artifact JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> OperatorGrowthPipelineArtifact:
        """Decode, checksum-verify, and mechanically replay a durable artifact."""
        mapping = verify_sealed_mapping(data, expected_keys=_ARTIFACT_KEYS, name="operator-growth pipeline artifact")
        if mapping["schema_version"] != OPERATOR_GROWTH_PIPELINE_ARTIFACT_SCHEMA_VERSION:
            msg = "Operator-growth pipeline artifact uses an unsupported schema version."
            raise ValueError(msg)
        raw_result = mapping["operator_growth_result"]
        raw_evaluation = mapping["outer_evaluation"]
        raw_resources = mapping["circuit_resources"]
        artifact = cls(
            request=OperatorGrowthPipelineRequest.from_dict(mapping["request"]),
            request_checksum=cast("str", mapping["request_checksum"]),
            attempt=cast("int", mapping["attempt"]),
            status=cast("Literal['success', 'failure']", mapping["status"]),
            failure_phase=cast("FailurePhase | None", mapping["failure_phase"]),
            exception_type=cast("str | None", mapping["exception_type"]),
            message=cast("str | None", mapping["message"]),
            operator_growth_result=(None if raw_result is None else OperatorGrowthResult.from_dict(raw_result)),
            outer_evaluation=(None if raw_evaluation is None else OuterScreeningEvaluation.from_dict(raw_evaluation)),
            circuit_resources=(None if raw_resources is None else CircuitResourceMetrics.from_dict(raw_resources)),
            training_work=WP20WorkLedger.from_dict(mapping["training_work"]),
            evaluation_work=WP20WorkLedger.from_dict(mapping["evaluation_work"]),
            total_work=WP20WorkLedger.from_dict(mapping["total_work"]),
            materialization_checksum=cast("str | None", mapping["materialization_checksum"]),
            verified_outcome=VerifiedScreeningOutcome.from_dict(mapping["verified_outcome"]),
        )
        if mapping["content_checksum"] != artifact.content_checksum:
            msg = "Operator-growth artifact checksum changed during normalization."
            raise ValueError(msg)
        return artifact

    @classmethod
    def from_json(cls, payload: str) -> OperatorGrowthPipelineArtifact:
        """Decode canonical artifact JSON."""
        return cls.from_dict(load_canonical_json_object(payload))


OperatorGrowthCallback = Callable[[OperatorGrowthPipelineRequest], OperatorGrowthResult]
OuterEvaluationCallback = Callable[
    [OperatorGrowthPipelineRequest, OperatorGrowthResult],
    OuterScreeningEvaluation,
]


def _failure_artifact(
    request: OperatorGrowthPipelineRequest,
    *,
    attempt: int,
    phase: FailurePhase,
    error: Exception,
    result: OperatorGrowthResult | None,
    training_work: WP20WorkLedger,
    evaluation_work: WP20WorkLedger,
) -> OperatorGrowthPipelineArtifact:
    """Construct one honest durable ordinary-failure artifact."""
    exception_type = type(error).__name__
    raw_message = str(error)
    message = raw_message if raw_message.strip() else exception_type
    usable_result = result is not None and phase == "screening_evaluation"
    resources = result.circuit_resources if usable_result else None
    materialization = _materialization_checksum(result) if usable_result and result is not None else None
    total = _sum_ledgers(training_work, evaluation_work)
    outcome = _expected_outcome(
        request=request,
        attempt=attempt,
        status="failure",
        failure_phase=phase,
        exception_type=exception_type,
        message=message,
        result=result,
        evaluation=None,
        resources=resources,
        work=total,
        materialization_checksum=materialization,
    )
    return OperatorGrowthPipelineArtifact(
        request=request,
        request_checksum=request.content_checksum,
        attempt=attempt,
        status="failure",
        failure_phase=phase,
        exception_type=exception_type,
        message=message,
        operator_growth_result=result,
        outer_evaluation=None,
        circuit_resources=resources,
        training_work=training_work,
        evaluation_work=evaluation_work,
        total_work=total,
        materialization_checksum=materialization,
        verified_outcome=outcome,
    )


def _load_artifact(path: Path, request: OperatorGrowthPipelineRequest) -> OperatorGrowthPipelineArtifact:
    """Load one existing artifact and require its exact request checksum."""
    artifact = OperatorGrowthPipelineArtifact.from_dict(load_canonical_json_object(path.read_text(encoding="utf-8")))
    if artifact.request_checksum != request.content_checksum:
        msg = "Existing operator-growth artifact belongs to a different exact request."
        raise ValueError(msg)
    return artifact


def execute_operator_growth_pipeline(
    request: OperatorGrowthPipelineRequest,
    output_directory: Path,
    operator_growth: OperatorGrowthCallback,
    outer_evaluation: OuterEvaluationCallback,
    *,
    resume: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> OperatorGrowthPipelineArtifact | None:
    """Execute, persist, resume, or dry-run one standalone operator-growth cell.

    Ordinary callback and validation exceptions become durable failure
    artifacts.  ``BaseException`` interruptions never publish a partial
    artifact, so a later invocation can safely recover from the exact request.
    The first terminal success or failure is immutable; resume returns it and
    overwrite is deliberately unsupported.
    """
    if not isinstance(request, OperatorGrowthPipelineRequest):
        msg = "request must be an OperatorGrowthPipelineRequest."
        raise TypeError(msg)
    if not isinstance(output_directory, Path):
        msg = "output_directory must be a pathlib.Path."
        raise TypeError(msg)
    if not callable(operator_growth) or not callable(outer_evaluation):
        msg = "operator_growth and outer_evaluation must be callable."
        raise TypeError(msg)
    resume_value = require_bool(resume, "resume")
    overwrite_value = require_bool(overwrite, "overwrite")
    dry_value = require_bool(dry_run, "dry_run")
    if resume_value and overwrite_value:
        msg = "resume and overwrite are mutually exclusive."
        raise ValueError(msg)
    if overwrite_value:
        msg = "Operator-growth first-terminal custody is immutable; overwrite is not supported."
        raise ValueError(msg)
    if dry_value:
        return None

    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = output_directory / OPERATOR_GROWTH_ARTIFACT_NAME
    lock_path = output_directory / OPERATOR_GROWTH_LOCK_NAME
    with FileLock(str(lock_path)):
        existing: OperatorGrowthPipelineArtifact | None = None
        if artifact_path.exists():
            existing = _load_artifact(artifact_path, request)
            if resume_value:
                return existing
            msg = "Operator-growth first-terminal artifact already exists; select resume to reopen it."
            raise ValueError(msg)
        attempt = 1

        growth_call = _measured_call(lambda: operator_growth(request))
        if growth_call.error is not None:
            artifact = _failure_artifact(
                request,
                attempt=attempt,
                phase="operator_growth_execution",
                error=growth_call.error,
                result=None,
                training_work=_ledger_with_measurement(
                    WP20WorkLedger(),
                    growth_call.wall_time_seconds,
                    growth_call.peak_memory_bytes,
                ),
                evaluation_work=WP20WorkLedger(),
            )
        else:
            try:
                result = cast("OperatorGrowthResult", growth_call.value)
                _validate_operator_result(request, result)
            except Exception as error:  # noqa: BLE001 - validation failures are durable evidence
                rejected_result = growth_call.value if isinstance(growth_call.value, OperatorGrowthResult) else None
                reported_work = WP20WorkLedger() if rejected_result is None else rejected_result.wp20_work
                artifact = _failure_artifact(
                    request,
                    attempt=attempt,
                    phase="operator_growth_execution",
                    error=error,
                    result=rejected_result,
                    training_work=_ledger_with_measurement(
                        reported_work,
                        growth_call.wall_time_seconds,
                        growth_call.peak_memory_bytes,
                    ),
                    evaluation_work=WP20WorkLedger(),
                )
            else:
                training_work = _ledger_with_measurement(
                    result.wp20_work,
                    growth_call.wall_time_seconds,
                    growth_call.peak_memory_bytes,
                )
                evaluation_call = _measured_call(lambda: outer_evaluation(request, result))
                if evaluation_call.error is not None:
                    artifact = _failure_artifact(
                        request,
                        attempt=attempt,
                        phase="screening_evaluation",
                        error=evaluation_call.error,
                        result=result,
                        training_work=training_work,
                        evaluation_work=_ledger_with_measurement(
                            WP20WorkLedger(),
                            evaluation_call.wall_time_seconds,
                            evaluation_call.peak_memory_bytes,
                        ),
                    )
                else:
                    try:
                        evaluation = cast("OuterScreeningEvaluation", evaluation_call.value)
                        _validate_outer_evaluation(request, result, evaluation)
                    except Exception as error:  # noqa: BLE001 - durable protocol failure
                        artifact = _failure_artifact(
                            request,
                            attempt=attempt,
                            phase="screening_evaluation",
                            error=error,
                            result=result,
                            training_work=training_work,
                            evaluation_work=_ledger_with_measurement(
                                WP20WorkLedger(),
                                evaluation_call.wall_time_seconds,
                                evaluation_call.peak_memory_bytes,
                            ),
                        )
                    else:
                        evaluation_work = _ledger_with_measurement(
                            evaluation.work_ledger,
                            evaluation_call.wall_time_seconds,
                            evaluation_call.peak_memory_bytes,
                        )
                        total = _sum_ledgers(training_work, evaluation_work)
                        resources = result.circuit_resources
                        assert resources is not None
                        materialization = _materialization_checksum(result)
                        outcome = _expected_outcome(
                            request=request,
                            attempt=attempt,
                            status="success",
                            failure_phase=None,
                            exception_type=None,
                            message=None,
                            result=result,
                            evaluation=evaluation,
                            resources=resources,
                            work=total,
                            materialization_checksum=materialization,
                        )
                        artifact = OperatorGrowthPipelineArtifact(
                            request=request,
                            request_checksum=request.content_checksum,
                            attempt=attempt,
                            status="success",
                            failure_phase=None,
                            exception_type=None,
                            message=None,
                            operator_growth_result=result,
                            outer_evaluation=evaluation,
                            circuit_resources=resources,
                            training_work=training_work,
                            evaluation_work=evaluation_work,
                            total_work=total,
                            materialization_checksum=materialization,
                            verified_outcome=outcome,
                        )

        atomic_write_bytes(artifact_path, f"{artifact.to_json()}\n".encode())
        return artifact


__all__ = [
    "OPERATOR_GROWTH_ARTIFACT_NAME",
    "OPERATOR_GROWTH_LOCK_NAME",
    "OPERATOR_GROWTH_MATERIALIZATION_SCHEMA_VERSION",
    "OPERATOR_GROWTH_PIPELINE_ARTIFACT_SCHEMA_VERSION",
    "OPERATOR_GROWTH_PIPELINE_REQUEST_SCHEMA_VERSION",
    "OUTER_SCREENING_ENSEMBLE_SCHEMA_VERSION",
    "OUTER_SCREENING_EVALUATION_SCHEMA_VERSION",
    "FailurePhase",
    "OperatorGrowthCallback",
    "OperatorGrowthPipelineArtifact",
    "OperatorGrowthPipelineRequest",
    "OuterEvaluationCallback",
    "OuterScreeningEvaluation",
    "derive_outer_screening_ensemble_checksum",
    "execute_operator_growth_pipeline",
]
