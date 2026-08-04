# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Checksum-sealed WP22 pilot evidence and cluster-aware sample-size design."""

from __future__ import annotations

import math
import operator
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Literal, Protocol, cast

from scipy.stats import chi2

from .canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
    verify_sealed_mapping,
)
from .execution_context import TrainingExecutionContext
from .protocol import (
    PRIMARY_FAMILY_STRATA,
    PRIMARY_TARGET_FAMILIES,
    InitialPreregistration,
    SampleAllocation,
    SampleSizeDesign,
    load_initial_preregistration,
)
from .result_custody import (
    ProductionResultCustody,
    production_noisy_fidelity,
    reopen_terminal_production_attempt,
    validate_production_job_custody,
)
from .targets import TargetPopulationManifest
from .training_schedules import PILOT_DIAGNOSTIC_SEED_POLICY_ID, ExecutionSeedPolicySuite
from .validation import (
    require_checksum,
    require_exact_keys,
    require_float,
    require_int,
    require_mapping,
    require_slug,
)

if TYPE_CHECKING:
    from .training_orchestration import TrainingJob, TrainingJobOutcome, TrainingRunPlan


class _TrainingOrchestrationModule(Protocol):
    """Runtime surface used lazily to avoid the screening/pilot import cycle."""

    TrainingJob: type[TrainingJob]
    TrainingJobOutcome: type[TrainingJobOutcome]
    TrainingRunPlan: type[TrainingRunPlan]
    TRAINING_JOB_OUTCOME_SCHEMA_VERSION: str
    PILOT_OPTIMIZATION_SEED_COUNT: int

    def derive_pilot_optimization_seeds(
        self,
        preregistration_checksum: str,
        seed_count: int,
    ) -> tuple[int, ...]: ...

    def load_training_job_outcome_history(
        self,
        job_directory: Path,
        job: TrainingJob,
    ) -> tuple[TrainingJobOutcome, ...]: ...


def _training_orchestration_module() -> _TrainingOrchestrationModule:
    """Load training custody schemas after module initialization.

    Returns:
        The runtime training-orchestration schema surface.
    """
    return cast("_TrainingOrchestrationModule", import_module(f"{__package__}.training_orchestration"))


PILOT_EVALUATION_EVIDENCE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.pilot_evaluation_evidence.v2"
PILOT_JOB_RESULT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.pilot_job_result.v2"
PILOT_OBSERVATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.pilot_observation.v2"
PILOT_CONTRAST_BINDING_SCHEMA_VERSION = "yaqs.state_preparation.phase2.pilot_contrast_binding.v1"
PILOT_NUISANCE_SUMMARY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.pilot_nuisance_summary.v3"
PILOT_INFERENCE_PROJECTION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.pilot_inference_projection.v1"
PILOT_CALCULATION_METHOD_ID = "cluster_aware_paired_difference_v1"
PILOT_DATA_ROLE = "development"
PILOT_QUBIT_COUNT = 6
VARIANCE_UCB_CONFIDENCE = 0.95
PILOT_PRIMARY_JOB_COUNT = 720
PILOT_PRIMARY_TRAJECTORY_COUNT = 1024
PILOT_SECONDARY_JOB_COUNT = 360
PILOT_SECONDARY_TRAJECTORY_COUNT = 256
MAURER_PONTIL_FAILURE_PROBABILITY = 0.05

FROZEN_CONTRAST_IDS = (
    "noisy_vs_noiseless",
    "promoted_vs_layerwise_v2_if_distinct",
)
_LAYERWISE_NOISY_METHOD_ID = "layerwise_bmpd_crn_v2"
_LAYERWISE_NOISELESS_METHOD_ID = "layerwise_bmpd_noiseless"

_CALCULATION_POLICY = {
    "method_id": PILOT_CALCULATION_METHOD_ID,
    "variance_bound": "normal_chi_square_one_sided_95_v1",
    "variance_model": "target_over_n_plus_optimizer_over_n_s_plus_mc_over_n_s_ntraj",
    "power": "normal_approximation_one_sided_worst_holm_alpha",
    "precision": "normal_two_sided_v1",
    "failure_precision": "pooled_method_marginal_clustered_binary_variance_with_wilson_floor_v1",
    "trajectory_count": "maurer_pontil_theorem_10_q6_720_job_union_bound_v1",
    "reestimation": "one_halfway_nuisance_only_non_decreasing_fixed_trajectories",
}
PILOT_CALCULATION_SOURCE_CHECKSUM = canonical_checksum(_CALCULATION_POLICY)

_JOB_RESULT_KEYS = frozenset({
    "schema_version",
    "job_checksum",
    "status",
    "result_evidence_schema_version",
    "result_evidence_checksum",
    "source_result_reference_checksum",
    "evaluation_evidence",
    "fresh_test_noisy_fidelity",
    "gradient_variance",
    "trajectory_mc_variance",
    "trajectory_count",
    "wall_time_seconds",
    "tracemalloc_peak_bytes",
    "content_checksum",
})
_EVALUATION_EVIDENCE_KEYS = frozenset({
    "schema_version",
    "job_checksum",
    "fresh_test_trajectory_fidelities",
    "gradient_samples",
    "content_checksum",
})
_CONTRAST_BINDING_KEYS = frozenset({
    "schema_version",
    "contrast_id",
    "pilot_plan_checksum",
    "treatment_method_id",
    "treatment_configuration_checksum",
    "comparator_method_id",
    "comparator_configuration_checksum",
    "content_checksum",
})
_OBSERVATION_KEYS = frozenset({
    "schema_version",
    "contrast_id",
    "treatment_job",
    "treatment_outcome",
    "treatment_result",
    "comparator_job",
    "comparator_outcome",
    "comparator_result",
    "content_checksum",
})
_SUMMARY_KEYS = frozenset({
    "schema_version",
    "summary_id",
    "preregistration_checksum",
    "target_manifest",
    "supplemental_target_manifest",
    "pilot_plan",
    "contrast_bindings",
    "observations",
    "nuisance_by_contrast",
    "runtime_summary",
    "content_checksum",
})
_FAMILY_COMPONENT_KEYS = frozenset({
    "target_count",
    "target_count_by_stratum",
    "observation_count",
    "failure_observation_count",
    "optimization_seed_count",
    "mean_fidelity_difference",
    "target_cluster_variance",
    "target_cluster_degrees_of_freedom",
    "optimization_seed_variance",
    "optimization_seed_degrees_of_freedom",
    "failure_rate",
    "failure_target_cluster_variance",
    "failure_optimization_seed_variance",
    "gradient_variance_mean",
    "gradient_variance_max",
    "trajectory_mc_variance_mean",
    "trajectory_mc_variance_max",
    "trajectory_count_min",
    "trajectory_count_max",
})


class PilotDesignInfeasibleError(ValueError):
    """Raised when the frozen WP22 bounds admit no valid sample-size design."""

    def __init__(self, reason_code: str, message: str) -> None:
        """Initialize a typed scientific infeasibility.

        Args:
            reason_code: Stable machine-readable reason.
            message: Human-readable explanation.
        """
        self.reason_code = require_slug(reason_code, "reason_code")
        super().__init__(message)


def _require_pilot_plan(plan: TrainingRunPlan, name: str = "pilot_plan") -> None:
    """Require a typed paper-pilot plan.

    Raises:
        TypeError: If ``plan`` is not a typed training plan.
        ValueError: If it is not the paper-pilot preset.
    """
    training = _training_orchestration_module()
    if not isinstance(plan, training.TrainingRunPlan):
        msg = f"{name} must be a TrainingRunPlan."
        raise TypeError(msg)
    if plan.preset != "paper-pilot":
        msg = f"{name} must use the paper-pilot preset."
        raise ValueError(msg)


def _unique_plan_configuration(plan: TrainingRunPlan, method_id: str) -> str:
    """Return the sole plan configuration for one prescribed method.

    Returns:
        The unique candidate-configuration checksum.

    Raises:
        ValueError: If the method is absent or has multiple configurations.
    """
    configurations = {job.candidate_configuration_checksum for job in plan.jobs if job.method_id == method_id}
    if len(configurations) != 1:
        msg = f"Pilot method {method_id!r} must have exactly one configuration in the exact plan."
        raise ValueError(msg)
    return next(iter(configurations))


@dataclass(frozen=True, slots=True)
class PilotContrastBinding:
    """Checksum-sealed treatment/control identity for one pilot contrast."""

    contrast_id: str
    pilot_plan_checksum: str
    treatment_method_id: str
    treatment_configuration_checksum: str
    comparator_method_id: str
    comparator_configuration_checksum: str
    schema_version: str = field(default=PILOT_CONTRAST_BINDING_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate immutable contrast, plan, method, and configuration identities.

        Raises:
            ValueError: If identities are unsupported, malformed, or not distinct.
        """
        contrast = require_slug(self.contrast_id, "contrast_id")
        if contrast not in FROZEN_CONTRAST_IDS:
            msg = f"contrast_id must be one of {FROZEN_CONTRAST_IDS!r}."
            raise ValueError(msg)
        object.__setattr__(self, "contrast_id", contrast)
        object.__setattr__(
            self,
            "pilot_plan_checksum",
            require_checksum(self.pilot_plan_checksum, "pilot_plan_checksum"),
        )
        for name in ("treatment_method_id", "comparator_method_id"):
            object.__setattr__(self, name, require_slug(getattr(self, name), name))
        for name in ("treatment_configuration_checksum", "comparator_configuration_checksum"):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        if self.treatment_configuration_checksum == self.comparator_configuration_checksum:
            msg = "Pilot contrast treatment and comparator configurations must be distinct."
            raise ValueError(msg)
        if contrast == "noisy_vs_noiseless":
            methods = (self.treatment_method_id, self.comparator_method_id)
            expected = (_LAYERWISE_NOISY_METHOD_ID, _LAYERWISE_NOISELESS_METHOD_ID)
            if methods != expected:
                msg = "noisy_vs_noiseless must bind layerwise_bmpd_crn_v2 against layerwise_bmpd_noiseless."
                raise ValueError(msg)
        elif self.comparator_method_id != _LAYERWISE_NOISY_METHOD_ID:
            msg = "promoted_vs_layerwise_v2_if_distinct must use layerwise_bmpd_crn_v2 as comparator."
            raise ValueError(msg)

    @classmethod
    def noisy_vs_noiseless(cls, pilot_plan: TrainingRunPlan) -> PilotContrastBinding:
        """Derive the mandatory noisy-versus-noiseless binding from one plan.

        Returns:
            The exact method-derived binding.
        """
        _require_pilot_plan(pilot_plan)
        return cls(
            contrast_id="noisy_vs_noiseless",
            pilot_plan_checksum=pilot_plan.content_checksum,
            treatment_method_id=_LAYERWISE_NOISY_METHOD_ID,
            treatment_configuration_checksum=_unique_plan_configuration(
                pilot_plan,
                _LAYERWISE_NOISY_METHOD_ID,
            ),
            comparator_method_id=_LAYERWISE_NOISELESS_METHOD_ID,
            comparator_configuration_checksum=_unique_plan_configuration(
                pilot_plan,
                _LAYERWISE_NOISELESS_METHOD_ID,
            ),
        )

    @classmethod
    def promoted_vs_layerwise_v2(
        cls,
        pilot_plan: TrainingRunPlan,
        *,
        treatment_method_id: str,
        treatment_configuration_checksum: str,
    ) -> PilotContrastBinding:
        """Bind an explicit promotion-eligible planning treatment against v2.

        Returns:
            A plan-linked binding for the future-promotion planning contrast.

        Raises:
            ValueError: If the explicit treatment is not an exact plan configuration.
        """
        _require_pilot_plan(pilot_plan)
        method = require_slug(treatment_method_id, "treatment_method_id")
        configuration = require_checksum(
            treatment_configuration_checksum,
            "treatment_configuration_checksum",
        )
        matching = {(job.method_id, job.candidate_configuration_checksum) for job in pilot_plan.jobs}
        if (method, configuration) not in matching:
            msg = "The promoted planning treatment is not an exact method/configuration in the pilot plan."
            raise ValueError(msg)
        return cls(
            contrast_id="promoted_vs_layerwise_v2_if_distinct",
            pilot_plan_checksum=pilot_plan.content_checksum,
            treatment_method_id=method,
            treatment_configuration_checksum=configuration,
            comparator_method_id=_LAYERWISE_NOISY_METHOD_ID,
            comparator_configuration_checksum=_unique_plan_configuration(
                pilot_plan,
                _LAYERWISE_NOISY_METHOD_ID,
            ),
        )

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete immutable contrast binding."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered binding content."""
        return {
            "schema_version": self.schema_version,
            "contrast_id": self.contrast_id,
            "pilot_plan_checksum": self.pilot_plan_checksum,
            "treatment_method_id": self.treatment_method_id,
            "treatment_configuration_checksum": self.treatment_configuration_checksum,
            "comparator_method_id": self.comparator_method_id,
            "comparator_configuration_checksum": self.comparator_configuration_checksum,
        }

    def to_dict(self) -> dict[str, object]:
        """Return sealed JSON-native binding data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed binding JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> PilotContrastBinding:
        """Decode and checksum-verify a pilot contrast binding.

        Returns:
            The verified binding.

        Raises:
            ValueError: If schema or checksum verification fails.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_CONTRAST_BINDING_KEYS,
            name="pilot contrast binding",
        )
        if mapping["schema_version"] != PILOT_CONTRAST_BINDING_SCHEMA_VERSION:
            msg = f"schema_version must be {PILOT_CONTRAST_BINDING_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        binding = cls(
            contrast_id=cast("str", mapping["contrast_id"]),
            pilot_plan_checksum=cast("str", mapping["pilot_plan_checksum"]),
            treatment_method_id=cast("str", mapping["treatment_method_id"]),
            treatment_configuration_checksum=cast("str", mapping["treatment_configuration_checksum"]),
            comparator_method_id=cast("str", mapping["comparator_method_id"]),
            comparator_configuration_checksum=cast("str", mapping["comparator_configuration_checksum"]),
        )
        if binding.content_checksum != mapping["content_checksum"]:
            msg = "Pilot contrast binding checksum changed during normalization."
            raise ValueError(msg)
        return binding

    @classmethod
    def from_json(cls, payload: str) -> PilotContrastBinding:
        """Decode a pilot contrast binding from canonical JSON text.

        Returns:
            The verified binding.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _require_exact_pilot_job(job: TrainingJob, name: str) -> None:
    """Require an exact paper-pilot q6 development job.

    Raises:
        TypeError: If ``job`` is not a typed training job.
        ValueError: If it is outside the frozen pilot population.
    """
    training = _training_orchestration_module()
    if not isinstance(job, training.TrainingJob):
        msg = f"{name} must be a TrainingJob."
        raise TypeError(msg)
    if job.preset != "paper-pilot" or job.data_role != PILOT_DATA_ROLE:
        msg = f"{name} must be an exact paper-pilot development job."
        raise ValueError(msg)
    if job.qubit_count != PILOT_QUBIT_COUNT:
        msg = "The primary WP22 pilot uses q=6 targets."
        raise ValueError(msg)
    if job.family_id not in PRIMARY_TARGET_FAMILIES:
        msg = f"family_id must be one of {PRIMARY_TARGET_FAMILIES!r}."
        raise ValueError(msg)
    if job.stratum_id not in PRIMARY_FAMILY_STRATA[job.family_id]:
        msg = f"stratum_id {job.stratum_id!r} is not registered for family {job.family_id!r}."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PilotEvaluationEvidence:
    """Raw per-trajectory and gradient evidence for one successful pilot job."""

    job_checksum: str
    fresh_test_trajectory_fidelities: tuple[float, ...]
    gradient_samples: tuple[tuple[float, ...], ...]
    schema_version: str = field(default=PILOT_EVALUATION_EVIDENCE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate source identity and finite raw samples.

        Raises:
            ValueError: If either raw sample collection has fewer than two values.
        """
        object.__setattr__(self, "job_checksum", require_checksum(self.job_checksum, "job_checksum"))
        fidelities = tuple(
            require_float(value, "fresh_test_trajectory_fidelity", minimum=0.0, maximum=1.0)
            for value in self.fresh_test_trajectory_fidelities
        )
        gradients = tuple(
            tuple(require_float(value, "pathwise_update_coordinate") for value in vector)
            for vector in self.gradient_samples
        )
        widths = {len(vector) for vector in gradients}
        if len(fidelities) < 2 or len(gradients) != 32 or len(widths) != 1 or widths == {0}:
            msg = "Pilot evidence requires fresh trajectories and exactly 32 equal-width pathwise update vectors."
            raise ValueError(msg)
        object.__setattr__(self, "fresh_test_trajectory_fidelities", fidelities)
        object.__setattr__(self, "gradient_samples", gradients)

    @property
    def fresh_test_noisy_fidelity(self) -> float:
        """WP22E float64 mean derived from raw trajectory evidence."""
        return production_noisy_fidelity(self.fresh_test_trajectory_fidelities)

    @property
    def trajectory_mc_variance(self) -> float:
        """Sample variance derived from raw fresh-test trajectories."""
        return float(statistics.variance(self.fresh_test_trajectory_fidelities))

    @property
    def trajectory_count(self) -> int:
        """Number of raw fresh-test trajectories."""
        return len(self.fresh_test_trajectory_fidelities)

    @property
    def gradient_variance(self) -> float:
        """Arithmetic mean of unbiased coordinate variances across 32 vectors."""
        coordinate_variances = tuple(
            statistics.variance(vector[index] for vector in self.gradient_samples)
            for index in range(len(self.gradient_samples[0]))
        )
        return float(statistics.fmean(coordinate_variances))

    @property
    def gradient_variance_max(self) -> float:
        """Largest unbiased coordinate variance across the pathwise vectors."""
        return float(
            max(
                statistics.variance(vector[index] for vector in self.gradient_samples)
                for index in range(len(self.gradient_samples[0]))
            )
        )

    @property
    def content_checksum(self) -> str:
        """Checksum of all raw pilot evaluation evidence."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered raw evidence content."""
        return {
            "schema_version": self.schema_version,
            "job_checksum": self.job_checksum,
            "fresh_test_trajectory_fidelities": list(self.fresh_test_trajectory_fidelities),
            "gradient_samples": [list(vector) for vector in self.gradient_samples],
        }

    def to_dict(self) -> dict[str, object]:
        """Return sealed JSON-native raw evaluation evidence."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed evaluation-evidence JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> PilotEvaluationEvidence:
        """Decode and checksum-verify raw pilot evaluation evidence.

        Returns:
            The verified evidence.

        Raises:
            TypeError: If a raw sample collection is not a sequence.
            ValueError: If schema or checksum verification fails.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_EVALUATION_EVIDENCE_KEYS,
            name="pilot evaluation evidence",
        )
        if mapping["schema_version"] != PILOT_EVALUATION_EVIDENCE_SCHEMA_VERSION:
            msg = f"schema_version must be {PILOT_EVALUATION_EVIDENCE_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        raw_fidelities = mapping["fresh_test_trajectory_fidelities"]
        raw_gradients = mapping["gradient_samples"]
        if (
            isinstance(raw_fidelities, (str, bytes))
            or not isinstance(raw_fidelities, Sequence)
            or isinstance(raw_gradients, (str, bytes))
            or not isinstance(raw_gradients, Sequence)
        ):
            msg = "Pilot evaluation raw samples must be sequences."
            raise TypeError(msg)
        evidence = cls(
            job_checksum=cast("str", mapping["job_checksum"]),
            fresh_test_trajectory_fidelities=tuple(cast("Sequence[float]", raw_fidelities)),
            gradient_samples=tuple(tuple(vector) for vector in cast("Sequence[Sequence[float]]", raw_gradients)),
        )
        if evidence.content_checksum != mapping["content_checksum"]:
            msg = "Pilot evaluation evidence checksum changed during normalization."
            raise ValueError(msg)
        return evidence

    @classmethod
    def from_json(cls, payload: str) -> PilotEvaluationEvidence:
        """Decode raw pilot evaluation evidence from canonical JSON text.

        Returns:
            The verified evidence.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class PilotJobResult:
    """Typed result and diagnostic evidence for one exact paper-pilot job.

    A successful orchestration outcome must checksum-address this record. A
    failed record instead checksum-addresses its typed ``TrainingJobOutcome``;
    this preserves partial runtime and convergence diagnostics without
    fabricating a successful result artifact.
    """

    job_checksum: str
    status: Literal["success", "failure"]
    result_evidence_schema_version: str
    result_evidence_checksum: str
    source_result_reference_checksum: str | None
    evaluation_evidence: PilotEvaluationEvidence | None
    fresh_test_noisy_fidelity: float | None
    gradient_variance: float
    trajectory_mc_variance: float
    trajectory_count: int
    wall_time_seconds: float
    tracemalloc_peak_bytes: int
    schema_version: str = field(default=PILOT_JOB_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate status-specific fidelity and finite diagnostics.

        Raises:
            TypeError: If successful evidence is not typed.
            ValueError: If status, fidelity, provenance, or diagnostics are invalid.
        """
        object.__setattr__(self, "job_checksum", require_checksum(self.job_checksum, "job_checksum"))
        if self.status == "success":
            evidence = self.evaluation_evidence
            if not isinstance(evidence, PilotEvaluationEvidence):
                msg = "Successful pilot results require embedded PilotEvaluationEvidence."
                raise TypeError(msg)
            if evidence.job_checksum != self.job_checksum:
                msg = "Successful pilot evaluation evidence does not reference the exact job."
                raise ValueError(msg)
            object.__setattr__(self, "result_evidence_schema_version", evidence.schema_version)
            object.__setattr__(self, "result_evidence_checksum", evidence.content_checksum)
            object.__setattr__(self, "fresh_test_noisy_fidelity", evidence.fresh_test_noisy_fidelity)
            object.__setattr__(self, "gradient_variance", evidence.gradient_variance)
            object.__setattr__(self, "trajectory_mc_variance", evidence.trajectory_mc_variance)
            object.__setattr__(self, "trajectory_count", evidence.trajectory_count)
        elif self.status == "failure":
            if self.evaluation_evidence is not None:
                msg = "Failed pilot results cannot contain successful evaluation evidence."
                raise ValueError(msg)
        else:
            msg = "status must be 'success' or 'failure'."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "result_evidence_schema_version",
            require_slug(self.result_evidence_schema_version, "result_evidence_schema_version"),
        )
        object.__setattr__(
            self,
            "result_evidence_checksum",
            require_checksum(self.result_evidence_checksum, "result_evidence_checksum"),
        )
        if self.source_result_reference_checksum is not None:
            object.__setattr__(
                self,
                "source_result_reference_checksum",
                require_checksum(
                    self.source_result_reference_checksum,
                    "source_result_reference_checksum",
                ),
            )
        if self.status == "success":
            fidelity = require_float(
                self.fresh_test_noisy_fidelity,
                "fresh_test_noisy_fidelity",
                minimum=0.0,
                maximum=1.0,
            )
            object.__setattr__(self, "fresh_test_noisy_fidelity", fidelity)
        elif self.status == "failure":
            if self.fresh_test_noisy_fidelity is not None:
                msg = "Failed pilot results must have null noisy fidelity."
                raise ValueError(msg)
        for name in ("gradient_variance", "trajectory_mc_variance", "wall_time_seconds"):
            object.__setattr__(self, name, require_float(getattr(self, name), name, minimum=0.0))
        object.__setattr__(
            self,
            "trajectory_count",
            require_int(self.trajectory_count, "trajectory_count", minimum=2),
        )
        object.__setattr__(
            self,
            "tracemalloc_peak_bytes",
            require_int(self.tracemalloc_peak_bytes, "tracemalloc_peak_bytes", minimum=0),
        )

    @classmethod
    def success(
        cls,
        job: TrainingJob,
        *,
        evaluation_evidence: PilotEvaluationEvidence,
        wall_time_seconds: float,
        tracemalloc_peak_bytes: int,
        source_result_reference_checksum: str | None = None,
    ) -> PilotJobResult:
        """Create a successful record whose job identity is mechanically derived.

        Returns:
            The checksum-sealed successful result.

        Raises:
            TypeError: If ``evaluation_evidence`` is not typed.
        """
        _require_exact_pilot_job(job, "job")
        if not isinstance(evaluation_evidence, PilotEvaluationEvidence):
            msg = "evaluation_evidence must be a PilotEvaluationEvidence."
            raise TypeError(msg)
        return cls(
            job_checksum=job.content_checksum,
            status="success",
            result_evidence_schema_version=evaluation_evidence.schema_version,
            result_evidence_checksum=evaluation_evidence.content_checksum,
            source_result_reference_checksum=source_result_reference_checksum,
            evaluation_evidence=evaluation_evidence,
            fresh_test_noisy_fidelity=evaluation_evidence.fresh_test_noisy_fidelity,
            gradient_variance=evaluation_evidence.gradient_variance,
            trajectory_mc_variance=evaluation_evidence.trajectory_mc_variance,
            trajectory_count=evaluation_evidence.trajectory_count,
            wall_time_seconds=wall_time_seconds,
            tracemalloc_peak_bytes=tracemalloc_peak_bytes,
        )

    @classmethod
    def failure(
        cls,
        job: TrainingJob,
        outcome: TrainingJobOutcome,
        *,
        gradient_variance: float,
        trajectory_mc_variance: float,
        trajectory_count: int,
        wall_time_seconds: float,
        tracemalloc_peak_bytes: int,
    ) -> PilotJobResult:
        """Create a failed record linked to its exact typed outcome.

        Returns:
            The checksum-sealed failure and partial-diagnostic record.

        Raises:
            TypeError: If ``outcome`` is not typed.
            ValueError: If the outcome is not this job's failure.
        """
        _require_exact_pilot_job(job, "job")
        training = _training_orchestration_module()
        if not isinstance(outcome, training.TrainingJobOutcome):
            msg = "outcome must be a TrainingJobOutcome."
            raise TypeError(msg)
        if outcome.job_checksum != job.content_checksum or outcome.status != "failure":
            msg = "A failed pilot result requires the exact failed outcome for its job."
            raise ValueError(msg)
        return cls(
            job_checksum=job.content_checksum,
            status="failure",
            result_evidence_schema_version=outcome.schema_version,
            result_evidence_checksum=outcome.content_checksum,
            source_result_reference_checksum=None,
            evaluation_evidence=None,
            fresh_test_noisy_fidelity=None,
            gradient_variance=gradient_variance,
            trajectory_mc_variance=trajectory_mc_variance,
            trajectory_count=trajectory_count,
            wall_time_seconds=wall_time_seconds,
            tracemalloc_peak_bytes=tracemalloc_peak_bytes,
        )

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete job result and diagnostics."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered result content."""
        return {
            "schema_version": self.schema_version,
            "job_checksum": self.job_checksum,
            "status": self.status,
            "result_evidence_schema_version": self.result_evidence_schema_version,
            "result_evidence_checksum": self.result_evidence_checksum,
            "source_result_reference_checksum": self.source_result_reference_checksum,
            "evaluation_evidence": (None if self.evaluation_evidence is None else self.evaluation_evidence.to_dict()),
            "fresh_test_noisy_fidelity": self.fresh_test_noisy_fidelity,
            "gradient_variance": self.gradient_variance,
            "trajectory_mc_variance": self.trajectory_mc_variance,
            "trajectory_count": self.trajectory_count,
            "wall_time_seconds": self.wall_time_seconds,
            "tracemalloc_peak_bytes": self.tracemalloc_peak_bytes,
        }

    def to_dict(self) -> dict[str, object]:
        """Return sealed JSON-native pilot-result data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed result JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> PilotJobResult:
        """Decode and checksum-verify a pilot-job result.

        Returns:
            The verified result and diagnostics.

        Raises:
            ValueError: If its schema or checksum is invalid.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_JOB_RESULT_KEYS, name="pilot job result")
        if mapping["schema_version"] != PILOT_JOB_RESULT_SCHEMA_VERSION:
            msg = f"schema_version must be {PILOT_JOB_RESULT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        result = cls(
            job_checksum=cast("str", mapping["job_checksum"]),
            status=cast("Literal['success', 'failure']", mapping["status"]),
            result_evidence_schema_version=cast("str", mapping["result_evidence_schema_version"]),
            result_evidence_checksum=cast("str", mapping["result_evidence_checksum"]),
            source_result_reference_checksum=cast(
                "str | None",
                mapping["source_result_reference_checksum"],
            ),
            evaluation_evidence=(
                None
                if mapping["evaluation_evidence"] is None
                else PilotEvaluationEvidence.from_dict(mapping["evaluation_evidence"])
            ),
            fresh_test_noisy_fidelity=cast("float | None", mapping["fresh_test_noisy_fidelity"]),
            gradient_variance=cast("float", mapping["gradient_variance"]),
            trajectory_mc_variance=cast("float", mapping["trajectory_mc_variance"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            wall_time_seconds=cast("float", mapping["wall_time_seconds"]),
            tracemalloc_peak_bytes=cast("int", mapping["tracemalloc_peak_bytes"]),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Pilot job result checksum changed during normalization."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> PilotJobResult:
        """Decode a pilot-job result from canonical JSON text.

        Returns:
            The verified result and diagnostics.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class PilotObservation:
    """A mechanically paired, source-addressed q6 pilot contrast.

    The record stores exact jobs, durable outcomes, and typed result evidence.
    Its intention-to-treat fidelities and failure indicators are properties,
    so callers cannot author or reseal those scientific outcomes directly.
    """

    contrast_id: str
    treatment_job: TrainingJob
    treatment_outcome: TrainingJobOutcome
    treatment_result: PilotJobResult
    comparator_job: TrainingJob
    comparator_outcome: TrainingJobOutcome
    comparator_result: PilotJobResult
    schema_version: str = field(default=PILOT_OBSERVATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Authenticate paired cell identity and every job/outcome/result link.

        Raises:
            TypeError: If any nested artifact has the wrong typed schema.
            ValueError: If the pair or provenance links do not agree exactly.
        """
        training = _training_orchestration_module()
        contrast = require_slug(self.contrast_id, "contrast_id")
        if contrast not in FROZEN_CONTRAST_IDS:
            msg = f"contrast_id must be one of {FROZEN_CONTRAST_IDS!r}."
            raise ValueError(msg)
        object.__setattr__(self, "contrast_id", contrast)
        nested = (
            ("treatment", self.treatment_job, self.treatment_outcome, self.treatment_result),
            ("comparator", self.comparator_job, self.comparator_outcome, self.comparator_result),
        )
        for role, job, outcome, result in nested:
            _require_exact_pilot_job(job, f"{role}_job")
            if not isinstance(outcome, training.TrainingJobOutcome):
                msg = f"{role}_outcome must be a TrainingJobOutcome."
                raise TypeError(msg)
            if not isinstance(result, PilotJobResult):
                msg = f"{role}_result must be a PilotJobResult."
                raise TypeError(msg)
            if outcome.job_checksum != job.content_checksum or result.job_checksum != job.content_checksum:
                msg = f"{role} outcome/result does not reference its exact paper-pilot job."
                raise ValueError(msg)
            if outcome.attempt != 1:
                msg = f"{role} pilot custody must use the authoritative first randomized attempt."
                raise ValueError(msg)
            if result.status != outcome.status:
                msg = f"{role} result status differs from its typed job outcome."
                raise ValueError(msg)
            if outcome.status == "success":
                expected_result_checksum = (
                    result.content_checksum
                    if result.source_result_reference_checksum is None
                    else result.source_result_reference_checksum
                )
                if outcome.result_artifact_checksum != expected_result_checksum:
                    msg = f"{role} successful outcome does not checksum-address its typed pilot result."
                    raise ValueError(msg)
            elif (
                result.result_evidence_schema_version != training.TRAINING_JOB_OUTCOME_SCHEMA_VERSION
                or result.result_evidence_checksum != outcome.content_checksum
            ):
                msg = f"{role} failed result does not reproduce its exact typed job outcome."
                raise ValueError(msg)
        if not isinstance(self.treatment_job, training.TrainingJob) or not isinstance(
            self.comparator_job,
            training.TrainingJob,
        ):
            msg = "Pilot jobs must be typed TrainingJob records."
            raise TypeError(msg)
        paired_fields = (
            "preset",
            "target_manifest_checksum",
            "target_instance_id",
            "target_spec_checksum",
            "family_id",
            "stratum_id",
            "qubit_count",
            "data_role",
            "optimization_block_id",
            "optimization_seed",
        )
        mismatches = [
            name for name in paired_fields if getattr(self.treatment_job, name) != getattr(self.comparator_job, name)
        ]
        if mismatches:
            msg = f"Paired pilot jobs disagree on required cell fields: {mismatches!r}."
            raise ValueError(msg)
        if self.treatment_job.content_checksum == self.comparator_job.content_checksum:
            msg = "Paired pilot treatment and comparator jobs must be distinct."
            raise ValueError(msg)
        if self.treatment_job.candidate_configuration_checksum == self.comparator_job.candidate_configuration_checksum:
            msg = "Paired pilot treatment and comparator configurations must be distinct."
            raise ValueError(msg)
        methods = (self.treatment_job.method_id, self.comparator_job.method_id)
        if contrast == "noisy_vs_noiseless":
            expected = (_LAYERWISE_NOISY_METHOD_ID, _LAYERWISE_NOISELESS_METHOD_ID)
            if methods != expected:
                msg = "noisy_vs_noiseless rows must use the prescribed noisy and noiseless layerwise methods."
                raise ValueError(msg)
        elif self.comparator_job.method_id != _LAYERWISE_NOISY_METHOD_ID:
            msg = "promoted planning rows must compare against layerwise_bmpd_crn_v2."
            raise ValueError(msg)

    @classmethod
    def from_paired_job_evidence(
        cls,
        *,
        contrast_id: str,
        treatment_job: TrainingJob,
        treatment_outcome: TrainingJobOutcome,
        treatment_result: PilotJobResult,
        comparator_job: TrainingJob,
        comparator_outcome: TrainingJobOutcome,
        comparator_result: PilotJobResult,
    ) -> PilotObservation:
        """Adapt exact paper-pilot custody evidence into one paired row.

        Returns:
            A checksum-sealed row with derived ITT and failure semantics.
        """
        return cls(
            contrast_id=contrast_id,
            treatment_job=treatment_job,
            treatment_outcome=treatment_outcome,
            treatment_result=treatment_result,
            comparator_job=comparator_job,
            comparator_outcome=comparator_outcome,
            comparator_result=comparator_result,
        )

    @property
    def data_role(self) -> Literal["development"]:
        """The mechanically verified pilot data role."""
        return cast("Literal['development']", self.treatment_job.data_role)

    @property
    def family_id(self) -> str:
        """The mechanically verified target family."""
        return self.treatment_job.family_id

    @property
    def stratum_id(self) -> str:
        """The mechanically verified target stratum."""
        return self.treatment_job.stratum_id

    @property
    def target_instance_id(self) -> str:
        """The mechanically verified target identifier."""
        return self.treatment_job.target_instance_id

    @property
    def qubit_count(self) -> int:
        """The mechanically verified q6 count."""
        return self.treatment_job.qubit_count

    @property
    def optimization_seed(self) -> int:
        """The mechanically verified paired optimization seed."""
        return self.treatment_job.optimization_seed

    @property
    def treatment_failed(self) -> bool:
        """Whether the typed treatment outcome failed."""
        return self.treatment_outcome.status == "failure"

    @property
    def comparator_failed(self) -> bool:
        """Whether the typed comparator outcome failed."""
        return self.comparator_outcome.status == "failure"

    @property
    def failed(self) -> bool:
        """Whether either pair member failed; retained as a derived diagnostic."""
        return self.treatment_failed or self.comparator_failed

    @property
    def treatment_intention_to_treat_fidelity(self) -> float:
        """Treatment fidelity with failures fixed mechanically to zero."""
        if self.treatment_failed:
            return 0.0
        return cast("float", self.treatment_result.fresh_test_noisy_fidelity)

    @property
    def comparator_intention_to_treat_fidelity(self) -> float:
        """Comparator fidelity with failures fixed mechanically to zero."""
        if self.comparator_failed:
            return 0.0
        return cast("float", self.comparator_result.fresh_test_noisy_fidelity)

    @property
    def fidelity_difference(self) -> float:
        """Derived paired treatment-minus-comparator ITT fidelity."""
        return self.treatment_intention_to_treat_fidelity - self.comparator_intention_to_treat_fidelity

    @property
    def gradient_variance(self) -> float:
        """Worst method-marginal gradient variance in this pair."""
        return max(self.treatment_result.gradient_variance, self.comparator_result.gradient_variance)

    @property
    def trajectory_mc_variance(self) -> float:
        """Conservative independent-arm MC variance of the paired ITT difference."""
        treatment = 0.0 if self.treatment_failed else self.treatment_result.trajectory_mc_variance
        comparator = 0.0 if self.comparator_failed else self.comparator_result.trajectory_mc_variance
        return treatment + comparator

    @property
    def trajectory_count(self) -> int:
        """Smaller of the two method-marginal fixed trajectory counts."""
        return min(self.treatment_result.trajectory_count, self.comparator_result.trajectory_count)

    @property
    def wall_time_seconds(self) -> float:
        """Total wall time across both jobs."""
        return self.treatment_result.wall_time_seconds + self.comparator_result.wall_time_seconds

    @property
    def tracemalloc_peak_bytes(self) -> int:
        """Worst Python-allocation peak across both jobs."""
        return max(
            self.treatment_result.tracemalloc_peak_bytes,
            self.comparator_result.tracemalloc_peak_bytes,
        )

    @property
    def identity(self) -> tuple[str, str, str, int, str]:
        """The unique paired pilot-cell identity."""
        return (
            self.family_id,
            self.stratum_id,
            self.target_instance_id,
            self.optimization_seed,
            self.contrast_id,
        )

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact nested custody evidence."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered observation content without derived outcomes."""
        return {
            "schema_version": self.schema_version,
            "contrast_id": self.contrast_id,
            "treatment_job": self.treatment_job.to_dict(),
            "treatment_outcome": self.treatment_outcome.to_dict(),
            "treatment_result": self.treatment_result.to_dict(),
            "comparator_job": self.comparator_job.to_dict(),
            "comparator_outcome": self.comparator_outcome.to_dict(),
            "comparator_result": self.comparator_result.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        """Return sealed JSON-native observation data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> PilotObservation:
        """Decode and checksum-verify exact paired pilot custody evidence.

        Returns:
            The validated observation.

        Raises:
            ValueError: If the schema, checksum, or normalized content is invalid.
        """
        training = _training_orchestration_module()
        mapping = verify_sealed_mapping(data, expected_keys=_OBSERVATION_KEYS, name="pilot observation")
        if mapping["schema_version"] != PILOT_OBSERVATION_SCHEMA_VERSION:
            msg = f"schema_version must be {PILOT_OBSERVATION_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        observation = cls.from_paired_job_evidence(
            contrast_id=cast("str", mapping["contrast_id"]),
            treatment_job=training.TrainingJob.from_dict(mapping["treatment_job"]),
            treatment_outcome=training.TrainingJobOutcome.from_dict(mapping["treatment_outcome"]),
            treatment_result=PilotJobResult.from_dict(mapping["treatment_result"]),
            comparator_job=training.TrainingJob.from_dict(mapping["comparator_job"]),
            comparator_outcome=training.TrainingJobOutcome.from_dict(mapping["comparator_outcome"]),
            comparator_result=PilotJobResult.from_dict(mapping["comparator_result"]),
        )
        if observation.content_checksum != mapping["content_checksum"]:
            msg = "Pilot observation checksum changed during normalization."
            raise ValueError(msg)
        return observation

    @classmethod
    def from_json(cls, payload: str) -> PilotObservation:
        """Decode a pilot observation from canonical JSON text.

        Returns:
            The validated observation.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _sample_variance(values: Sequence[float]) -> float:
    """Return a stable sample variance, requiring at least two values.

    Raises:
        ValueError: If fewer than two values are supplied.
    """
    if len(values) < 2:
        msg = "Cluster variance requires at least two observations."
        raise ValueError(msg)
    return float(statistics.variance(values))


def _family_components(observations: Sequence[PilotObservation]) -> dict[str, object]:
    """Derive nested-target nuisance components for one contrast and family.

    Returns:
        JSON-native clustered variance, failure, convergence, and resource components.

    Raises:
        ValueError: If target or optimization-seed replication is insufficient.
    """
    by_target: dict[str, list[PilotObservation]] = defaultdict(list)
    for observation in observations:
        by_target[observation.target_instance_id].append(observation)
    if len(by_target) < 2:
        msg = "Each pilot contrast/family requires at least two independent target clusters."
        raise ValueError(msg)

    seed_counts = {len({item.optimization_seed for item in rows}) for rows in by_target.values()}
    if len(seed_counts) != 1 or next(iter(seed_counts)) < 2:
        msg = "Every pilot target must have the same number of at least two optimization seeds."
        raise ValueError(msg)
    optimization_seed_count = next(iter(seed_counts))
    target_means: list[float] = []
    target_failure_means: list[float] = []
    within_squares = 0.0
    failure_within_squares = 0.0
    within_df = 0
    for rows in by_target.values():
        differences = [item.fidelity_difference for item in rows]
        failures = [(float(item.treatment_failed) + float(item.comparator_failed)) / 2.0 for item in rows]
        difference_mean = math.fsum(differences) / len(differences)
        failure_mean = math.fsum(failures) / len(failures)
        target_means.append(difference_mean)
        target_failure_means.append(failure_mean)
        within_squares += math.fsum((value - difference_mean) ** 2 for value in differences)
        failure_within_squares += math.fsum((value - failure_mean) ** 2 for value in failures)
        within_df += len(rows) - 1
    if within_df <= 0:
        msg = "Pilot optimization-seed variance has no residual degrees of freedom."
        raise ValueError(msg)

    target_counts_by_stratum: dict[str, set[str]] = defaultdict(set)
    for observation in observations:
        target_counts_by_stratum[observation.stratum_id].add(observation.target_instance_id)
    target_count_by_stratum = {
        stratum_id: len(target_ids) for stratum_id, target_ids in sorted(target_counts_by_stratum.items())
    }
    failures = sum(item.treatment_failed + item.comparator_failed for item in observations)
    failure_observation_count = 2 * len(observations)
    job_results = [
        result
        for observation in observations
        for result in (observation.treatment_result, observation.comparator_result)
    ]
    return {
        "target_count": len(by_target),
        "target_count_by_stratum": target_count_by_stratum,
        "observation_count": len(observations),
        "failure_observation_count": failure_observation_count,
        "optimization_seed_count": optimization_seed_count,
        "mean_fidelity_difference": math.fsum(item.fidelity_difference for item in observations) / len(observations),
        "target_cluster_variance": _sample_variance(target_means),
        "target_cluster_degrees_of_freedom": len(target_means) - 1,
        "optimization_seed_variance": within_squares / within_df,
        "optimization_seed_degrees_of_freedom": within_df,
        "failure_rate": failures / failure_observation_count,
        "failure_target_cluster_variance": _sample_variance(target_failure_means),
        "failure_optimization_seed_variance": failure_within_squares / within_df,
        "gradient_variance_mean": math.fsum(item.gradient_variance for item in job_results) / len(job_results),
        "gradient_variance_max": max(item.gradient_variance for item in job_results),
        "trajectory_mc_variance_mean": math.fsum(item.trajectory_mc_variance for item in observations)
        / len(observations),
        "trajectory_mc_variance_max": max(item.trajectory_mc_variance for item in observations),
        "trajectory_count_min": min(item.trajectory_count for item in observations),
        "trajectory_count_max": max(item.trajectory_count for item in observations),
    }


def _validate_exact_pilot_custody(
    preregistration: InitialPreregistration,
    target_manifest: TargetPopulationManifest,
    supplemental_target_manifest: TargetPopulationManifest,
    pilot_plan: TrainingRunPlan,
    bindings: tuple[PilotContrastBinding, ...],
    observations: tuple[PilotObservation, ...],
) -> tuple[PilotContrastBinding, ...]:
    """Validate the manifest, plan, contrast, and result universe as one unit.

    Returns:
        Contrast bindings in frozen contrast order.

    Raises:
        TypeError: If a custody artifact has the wrong typed schema.
        ValueError: If any manifest, plan, binding, grid, or result link differs.
    """
    training = _training_orchestration_module()
    if not isinstance(target_manifest, TargetPopulationManifest) or not isinstance(
        supplemental_target_manifest,
        TargetPopulationManifest,
    ):
        msg = "target_manifest and supplemental_target_manifest must be TargetPopulationManifest values."
        raise TypeError(msg)
    _require_pilot_plan(pilot_plan)
    if (
        target_manifest.data_role != PILOT_DATA_ROLE
        or target_manifest.population_scope != "primary_q6"
        or target_manifest.preregistration_checksum != preregistration.content_checksum
    ):
        msg = "Pilot evidence requires the exact primary-q6 development target manifest."
        raise ValueError(msg)
    if (
        supplemental_target_manifest.data_role != "screening_selection"
        or supplemental_target_manifest.population_scope != "secondary_q12"
        or supplemental_target_manifest.preregistration_checksum != preregistration.content_checksum
    ):
        msg = "The paper pilot requires its exact screening-selection secondary-q12 manifest."
        raise ValueError(msg)
    if (
        pilot_plan.preregistration_checksum != preregistration.content_checksum
        or pilot_plan.target_manifest_checksums
        != (target_manifest.content_checksum, supplemental_target_manifest.content_checksum)
    ):
        msg = "The paper-pilot plan is not rooted in the exact preregistration and development manifest."
        raise ValueError(msg)

    target_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    for target in target_manifest.instances:
        target_counts[target.family_id, target.stratum_id, target.qubit_count] += 1
    expected_target_counts = {
        (family_id, stratum_id, PILOT_QUBIT_COUNT): 12 // len(strata)
        for family_id, strata in PRIMARY_FAMILY_STRATA.items()
        for stratum_id in strata
    }
    if target_counts != expected_target_counts:
        msg = "The pilot manifest must contain exactly 12 balanced q=6 targets per primary family."
        raise ValueError(msg)
    supplemental_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    for target in supplemental_target_manifest.instances:
        supplemental_counts[target.family_id, target.stratum_id, target.qubit_count] += 1
    expected_supplemental_counts = {
        (family_id, stratum_id, 12): 6 // len(strata)
        for family_id, strata in PRIMARY_FAMILY_STRATA.items()
        for stratum_id in strata
    }
    if supplemental_counts != expected_supplemental_counts:
        msg = "The supplemental pilot manifest must contain exactly six balanced q=12 targets per family."
        raise ValueError(msg)

    if not bindings or not all(isinstance(item, PilotContrastBinding) for item in bindings):
        msg = "contrast_bindings must contain PilotContrastBinding values."
        raise TypeError(msg)
    binding_by_id = {binding.contrast_id: binding for binding in bindings}
    if len(binding_by_id) != len(bindings) or set(binding_by_id) != set(FROZEN_CONTRAST_IDS):
        msg = "Pilot contrast bindings must cover each frozen contrast exactly once."
        raise ValueError(msg)
    ordered_bindings = tuple(binding_by_id[contrast_id] for contrast_id in FROZEN_CONTRAST_IDS)
    if any(binding.pilot_plan_checksum != pilot_plan.content_checksum for binding in ordered_bindings):
        msg = "Every pilot contrast binding must checksum-address the exact paper-pilot plan."
        raise ValueError(msg)
    derived_noisy = PilotContrastBinding.noisy_vs_noiseless(pilot_plan)
    if binding_by_id["noisy_vs_noiseless"] != derived_noisy:
        msg = "The noisy-versus-noiseless binding is not derived from the prescribed plan methods."
        raise ValueError(msg)
    promoted = binding_by_id["promoted_vs_layerwise_v2_if_distinct"]
    if promoted.comparator_configuration_checksum != derived_noisy.treatment_configuration_checksum:
        msg = "The promoted planning contrast must reuse the exact noisy-v2 comparator configuration."
        raise ValueError(msg)
    promotion_eligible = {
        cast("str", policy["method_id"])
        for policy in preregistration.candidate_methods
        if cast("bool", policy["promotion_eligible"])
    }
    if (
        promoted.treatment_method_id not in promotion_eligible
        or promoted.treatment_method_id == _LAYERWISE_NOISY_METHOD_ID
    ):
        msg = "The promoted planning treatment must be a distinct preregistered promotion-eligible method."
        raise ValueError(msg)

    bound_pairs = {
        (binding.treatment_method_id, binding.treatment_configuration_checksum) for binding in ordered_bindings
    } | {(binding.comparator_method_id, binding.comparator_configuration_checksum) for binding in ordered_bindings}
    plan_pairs = {(job.method_id, job.candidate_configuration_checksum) for job in pilot_plan.jobs}
    if plan_pairs != bound_pairs:
        msg = "The exact paper-pilot plan must contain only and all contrast-bound configurations."
        raise ValueError(msg)

    allocation_policy = cast(
        "Mapping[str, object]",
        preregistration.target_population_policy["role_allocation_policy"],
    )
    seed_count = cast("int", allocation_policy["pilot_optimizer_seed_count"])
    expected_seeds = training.derive_pilot_optimization_seeds(
        preregistration.content_checksum,
        seed_count,
    )
    if seed_count != training.PILOT_OPTIMIZATION_SEED_COUNT:
        msg = "The preregistration does not retain the exact five-seed pilot policy."
        raise ValueError(msg)

    manifests = (target_manifest, supplemental_target_manifest)
    expected_job_keys = {
        (target.target_instance_id, seed, method_id, configuration_checksum)
        for manifest in manifests
        for target in manifest.instances
        for seed in expected_seeds
        for method_id, configuration_checksum in bound_pairs
    }
    all_targets_by_id = {
        target.target_instance_id: (target, manifest) for manifest in manifests for target in manifest.instances
    }
    if len(all_targets_by_id) != sum(len(manifest.instances) for manifest in manifests):
        msg = "Primary and supplemental pilot manifests must have disjoint target identities."
        raise ValueError(msg)
    actual_job_keys: set[tuple[str, int, str, str]] = set()
    for job in pilot_plan.jobs:
        resolved_target = all_targets_by_id.get(job.target_instance_id)
        if resolved_target is None:
            msg = "The paper-pilot plan contains a target outside its exact manifest."
            raise ValueError(msg)
        target, manifest = resolved_target
        expected_job_data_role = (
            "secondary_benchmark"
            if manifest.data_role == "screening_selection" and manifest.population_scope == "secondary_q12"
            else manifest.data_role
        )
        target_fields = (
            (job.target_manifest_checksum, manifest.content_checksum),
            (job.target_spec_checksum, target.content_checksum),
            (job.family_id, target.family_id),
            (job.stratum_id, target.stratum_id),
            (job.qubit_count, target.qubit_count),
            (job.data_role, expected_job_data_role),
            (job.optimization_block_id, f"pilot_{target.target_instance_id}_seed_{job.optimization_seed}"),
        )
        if any(actual != expected for actual, expected in target_fields):
            msg = "A paper-pilot job differs from its exact manifest target or paired block."
            raise ValueError(msg)
        actual_job_keys.add((
            job.target_instance_id,
            job.optimization_seed,
            job.method_id,
            job.candidate_configuration_checksum,
        ))
    if actual_job_keys != expected_job_keys or len(pilot_plan.jobs) != len(expected_job_keys):
        msg = "The paper-pilot plan is not the complete target-by-five-seed-by-configuration Cartesian grid."
        raise ValueError(msg)

    plan_jobs = {job.content_checksum: job for job in pilot_plan.jobs}
    observed_job_checksums: set[str] = set()
    evidence_by_job: dict[str, tuple[str, str]] = {}
    expected_observation_grid = {
        (contrast_id, target.target_instance_id, seed)
        for contrast_id in FROZEN_CONTRAST_IDS
        for target in target_manifest.instances
        for seed in expected_seeds
    }
    actual_observation_grid: set[tuple[str, str, int]] = set()
    for observation in observations:
        binding = binding_by_id[observation.contrast_id]
        expected_arms = (
            (
                observation.treatment_job,
                observation.treatment_outcome,
                observation.treatment_result,
                binding.treatment_method_id,
                binding.treatment_configuration_checksum,
            ),
            (
                observation.comparator_job,
                observation.comparator_outcome,
                observation.comparator_result,
                binding.comparator_method_id,
                binding.comparator_configuration_checksum,
            ),
        )
        for job, outcome, result, method_id, configuration_checksum in expected_arms:
            if (
                plan_jobs.get(job.content_checksum) != job
                or job.method_id != method_id
                or job.candidate_configuration_checksum != configuration_checksum
            ):
                msg = "A pilot observation arm does not match its exact plan-linked contrast binding."
                raise ValueError(msg)
            observed_job_checksums.add(job.content_checksum)
            evidence = (outcome.content_checksum, result.content_checksum)
            previous = evidence_by_job.setdefault(job.content_checksum, evidence)
            if previous != evidence:
                msg = "Repeated pilot jobs must reuse identical typed outcome and result evidence."
                raise ValueError(msg)
        actual_observation_grid.add((
            observation.contrast_id,
            observation.target_instance_id,
            observation.optimization_seed,
        ))
    if actual_observation_grid != expected_observation_grid:
        msg = "Pilot observations must cover every exact manifest target and all five derived seeds per contrast."
        raise ValueError(msg)
    primary_job_checksums = {
        job.content_checksum
        for job in pilot_plan.jobs
        if job.target_manifest_checksum == target_manifest.content_checksum
    }
    if observed_job_checksums != primary_job_checksums:
        msg = "Pilot observations do not consume the complete primary-q6 paper-pilot job universe."
        raise ValueError(msg)
    return ordered_bindings


@dataclass(frozen=True, slots=True, init=False)
class ProductionPilotJobRecord:
    """One exact pilot job replayed from its first immutable WP22E attempt."""

    job: TrainingJob
    outcome: TrainingJobOutcome
    result_custody: ProductionResultCustody
    pilot_result: PilotJobResult | None

    def __init__(
        self,
        context: TrainingExecutionContext,
        job: TrainingJob,
        job_directory: Path,
    ) -> None:
        """Reopen and validate one context-owned q6 or q12 pilot attempt.

        Raises:
            TypeError: If an input has the wrong typed schema.
            ValueError: If plan identity, raw evidence, diagnostics, or role differs.
        """
        training = _training_orchestration_module()
        if not isinstance(context, TrainingExecutionContext):
            msg = "context must be a TrainingExecutionContext."
            raise TypeError(msg)
        if not isinstance(job, training.TrainingJob) or not isinstance(job_directory, Path):
            msg = "job and job_directory must be typed pilot execution inputs."
            raise TypeError(msg)
        if context.plan.preset != "paper-pilot" or not any(item is job for item in context.plan.jobs):
            msg = "Production pilot replay accepts only an exact context-owned paper-pilot job object."
            raise ValueError(msg)
        history = training.load_training_job_outcome_history(job_directory, job)
        if not history:
            msg = "Pilot job has no durable first terminal outcome."
            raise ValueError(msg)
        outcome = history[0]
        custody = reopen_terminal_production_attempt(job, outcome, job_directory)
        trajectory_count = PILOT_PRIMARY_TRAJECTORY_COUNT if job.qubit_count == 6 else PILOT_SECONDARY_TRAJECTORY_COUNT
        expected_kind = "operator_growth" if job.implementation_kind == "operator_growth" else "pipeline"
        validate_production_job_custody(
            custody,
            job,
            outcome,
            expected_data_role=job.data_role,
            expected_trajectory_count=trajectory_count,
            expected_execution_source_manifest_checksum=context.execution_source_manifest.content_checksum,
            allowed_artifact_kinds=(expected_kind,),
        )
        links = tuple(
            link for link in context.scoped_bindings if link.content_checksum == job.executable_binding_checksum
        )
        if len(links) != 1:
            msg = "Pilot job lacks its unique context-owned executable binding."
            raise ValueError(msg)
        diagnostic_policy = links[0].binding.pilot_diagnostic_policy
        pilot_result: PilotJobResult | None = None
        wall_time = require_float(
            custody.resource_payload.get("wall_time_seconds"),
            "wall_time_seconds",
            minimum=0.0,
        )
        peak_memory = require_int(
            custody.resource_payload.get("peak_memory_bytes"),
            "peak_memory_bytes",
            minimum=0,
        )
        if job.qubit_count == 12:
            if diagnostic_policy is None or diagnostic_policy.enabled or custody.pilot_diagnostics:
                msg = "Secondary-q12 pilot jobs must remain diagnostic-free."
                raise ValueError(msg)
        elif outcome.status == "success":
            if diagnostic_policy is None or not diagnostic_policy.enabled or len(custody.pilot_diagnostics) != 1:
                msg = "Successful q6 pilot jobs require their exact frozen pathwise diagnostic."
                raise ValueError(msg)
            diagnostic = custody.pilot_diagnostics[0]
            selected_checksum = custody.production_evidence.derived_metrics.get("selected_parameter_checksum")
            provider_identity = diagnostic_policy.provider_identity
            if provider_identity is None:
                msg = "Q6 pilot diagnostic policy lacks its frozen provider identity."
                raise ValueError(msg)
            expected_provider_checksum = require_checksum(
                provider_identity.get("content_checksum"),
                "pilot_diagnostic_policy.provider_identity.content_checksum",
            )
            expected_estimator_checksum = canonical_checksum({
                "endpoint": diagnostic_policy.endpoint,
                "checkpoint_rule": diagnostic_policy.checkpoint_rule,
                "estimator_id": diagnostic_policy.estimator_id,
                "estimator_version": diagnostic_policy.estimator_version,
                "parameter_ordering": diagnostic_policy.parameter_ordering,
                "coordinate_variance_rule": diagnostic_policy.coordinate_variance_rule,
                "summary_statistics": list(diagnostic_policy.summary_statistics),
                "provider_checksum": expected_provider_checksum,
            })
            expected_seeds = tuple(
                ExecutionSeedPolicySuite.frozen().derive(
                    PILOT_DIAGNOSTIC_SEED_POLICY_ID,
                    {
                        "target_manifest_checksum": job.target_manifest_checksum,
                        "target_instance_spec_checksum": job.target_spec_checksum,
                        "optimization_seed": job.optimization_seed,
                        "publication_candidate_checksum": job.candidate_configuration_checksum,
                        "repetition": repetition,
                    },
                )
                for repetition in range(diagnostic_policy.trajectory_count)
            )
            circuit_payload = require_mapping(
                custody.resource_payload.get("circuit"),
                "runtime_resources.circuit",
            )
            expected_circuit_checksum = require_checksum(
                circuit_payload.get("circuit_binding_checksum"),
                "runtime_resources.circuit.circuit_binding_checksum",
            )
            if (
                diagnostic.policy_checksum != diagnostic_policy.content_checksum
                or diagnostic.job_checksum != job.content_checksum
                or diagnostic.checkpoint_parameter_checksum != selected_checksum
                or diagnostic.parameter_vector_checksum != selected_checksum
                or diagnostic.member_seeds != expected_seeds
                or diagnostic.provider_checksum != expected_provider_checksum
                or diagnostic.estimator_checksum != expected_estimator_checksum
                or diagnostic.circuit_checksum != expected_circuit_checksum
            ):
                msg = (
                    "Q6 pilot diagnostic differs from its frozen seeds, provider, estimator, circuit, "
                    "or selected checkpoint."
                )
                raise ValueError(msg)
            values = custody.trajectory_fidelities
            if values is None:
                msg = "Successful q6 pilot job lacks raw fresh trajectories."
                raise ValueError(msg)
            pilot_result = PilotJobResult.success(
                job,
                evaluation_evidence=PilotEvaluationEvidence(
                    job_checksum=job.content_checksum,
                    fresh_test_trajectory_fidelities=values,
                    gradient_samples=diagnostic.pathwise_update_vectors,
                ),
                wall_time_seconds=wall_time,
                tracemalloc_peak_bytes=peak_memory,
                source_result_reference_checksum=custody.reference.content_checksum,
            )
        else:
            if custody.pilot_diagnostics:
                msg = "Failed q6 pilot jobs cannot claim completed pathwise diagnostics."
                raise ValueError(msg)
            pilot_result = PilotJobResult.failure(
                job,
                outcome,
                gradient_variance=0.0,
                trajectory_mc_variance=0.25,
                trajectory_count=PILOT_PRIMARY_TRAJECTORY_COUNT,
                wall_time_seconds=wall_time,
                tracemalloc_peak_bytes=peak_memory,
            )
        object.__setattr__(self, "job", job)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "result_custody", custody)
        object.__setattr__(self, "pilot_result", pilot_result)

    @property
    def content_checksum(self) -> str:
        """Checksum of exact job, outcome, and reopened-manifest identities."""
        return canonical_checksum({
            "job_checksum": self.job.content_checksum,
            "outcome_checksum": self.outcome.content_checksum,
            "result_custody_checksum": self.result_custody.content_checksum,
            "pilot_result_checksum": None if self.pilot_result is None else self.pilot_result.content_checksum,
        })


@dataclass(frozen=True, slots=True, init=False)
class ProductionPilotCustody:
    """Complete 720-q6 plus 360-q12 first-attempt pilot replay."""

    context: TrainingExecutionContext
    records: tuple[ProductionPilotJobRecord, ...]
    _nuisance_summary_cache_key: tuple[str, ...] | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _nuisance_summary_cache: PilotNuisanceSummary | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __init__(self, context: TrainingExecutionContext, output_root: Path) -> None:
        """Reopen every job in the exact paper-pilot context.

        Raises:
            TypeError: If context or output root has the wrong type.
            ValueError: If the exact 1,080-job universe does not reopen uniquely.
        """
        if not isinstance(context, TrainingExecutionContext) or not isinstance(output_root, Path):
            msg = "context and output_root must be typed production inputs."
            raise TypeError(msg)
        if context.plan.preset != "paper-pilot" or len(context.plan.jobs) != (
            PILOT_PRIMARY_JOB_COUNT + PILOT_SECONDARY_JOB_COUNT
        ):
            msg = "Production pilot custody requires the exact 1,080-job paper-pilot context."
            raise ValueError(msg)
        records = tuple(
            ProductionPilotJobRecord(context, job, output_root / job.output_path) for job in context.plan.jobs
        )
        if len({item.job.content_checksum for item in records}) != len(records):
            msg = "Production pilot custody contains duplicate exact jobs."
            raise ValueError(msg)
        if len({item.result_custody.reference.content_checksum for item in records}) != len(records):
            msg = "Every pilot job must have a distinct immutable first-attempt result reference."
            raise ValueError(msg)
        primary_count = sum(item.job.qubit_count == 6 for item in records)
        secondary_count = sum(item.job.qubit_count == 12 for item in records)
        if (primary_count, secondary_count) != (PILOT_PRIMARY_JOB_COUNT, PILOT_SECONDARY_JOB_COUNT):
            msg = "Production pilot custody differs from the exact q6/q12 job split."
            raise ValueError(msg)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "_nuisance_summary_cache_key", None)
        object.__setattr__(self, "_nuisance_summary_cache", None)

    @property
    def secondary_archive_checksum(self) -> str:
        """Checksum of q12 scaling evidence, deliberately excluded from inference."""
        return canonical_checksum({
            "schema_version": "yaqs.state_preparation.phase2.secondary_q12_pilot_archive.v1",
            "record_checksums": [item.content_checksum for item in self.records if item.job.qubit_count == 12],
        })

    def build_nuisance_summary(
        self,
        contrast_bindings: Sequence[PilotContrastBinding],
        *,
        summary_id: str = "phase2_pilot_nuisance_v1",
    ) -> PilotNuisanceSummary:
        """Construct q6 inference rows while keeping q12 numerical evidence archived.

        Returns:
            The exact primary-q6 nuisance summary.

        Raises:
            ValueError: If contrast bindings or q6 record coverage differ.
        """
        bindings = tuple(contrast_bindings)
        binding_by_id = {item.contrast_id: item for item in bindings}
        if len(binding_by_id) != len(FROZEN_CONTRAST_IDS) or set(binding_by_id) != set(FROZEN_CONTRAST_IDS):
            msg = "Production pilot replay requires each frozen contrast binding exactly once."
            raise ValueError(msg)
        cache_key = (summary_id, *(item.content_checksum for item in bindings))
        if getattr(self, "_nuisance_summary_cache_key", None) == cache_key:
            cached = getattr(self, "_nuisance_summary_cache", None)
            if cached is not None:
                return cached
        primary_by_key = {
            (item.job.target_instance_id, item.job.optimization_seed, item.job.method_id): item
            for item in self.records
            if item.job.qubit_count == 6
        }
        observations: list[PilotObservation] = []
        primary_manifest = next(
            manifest for manifest in self.context.target_manifests if manifest.population_scope == "primary_q6"
        )
        supplemental_manifest = next(
            manifest for manifest in self.context.target_manifests if manifest.population_scope == "secondary_q12"
        )
        seeds = _training_orchestration_module().derive_pilot_optimization_seeds(
            self.context.preregistration.content_checksum,
            _training_orchestration_module().PILOT_OPTIMIZATION_SEED_COUNT,
        )
        for target in primary_manifest.instances:
            for seed in seeds:
                for contrast_id in FROZEN_CONTRAST_IDS:
                    binding = binding_by_id[contrast_id]
                    treatment = primary_by_key[target.target_instance_id, seed, binding.treatment_method_id]
                    comparator = primary_by_key[target.target_instance_id, seed, binding.comparator_method_id]
                    observations.append(
                        PilotObservation.from_paired_job_evidence(
                            contrast_id=contrast_id,
                            treatment_job=treatment.job,
                            treatment_outcome=treatment.outcome,
                            treatment_result=cast("PilotJobResult", treatment.pilot_result),
                            comparator_job=comparator.job,
                            comparator_outcome=comparator.outcome,
                            comparator_result=cast("PilotJobResult", comparator.pilot_result),
                        )
                    )
        summary = build_pilot_nuisance_summary(
            self.context.preregistration,
            primary_manifest,
            supplemental_manifest,
            self.context.plan,
            bindings,
            observations,
            summary_id=summary_id,
        )
        object.__setattr__(self, "_nuisance_summary_cache_key", cache_key)  # noqa: PLC2801 - frozen derived cache
        object.__setattr__(self, "_nuisance_summary_cache", summary)  # noqa: PLC2801 - frozen derived cache
        return summary


@dataclass(frozen=True, slots=True)
class PilotNuisanceSummary:
    """Deterministic raw-row-backed clustered pilot nuisance summary."""

    summary_id: str
    preregistration_checksum: str
    target_manifest: TargetPopulationManifest
    supplemental_target_manifest: TargetPopulationManifest
    pilot_plan: TrainingRunPlan
    contrast_bindings: tuple[PilotContrastBinding, ...]
    observations: tuple[PilotObservation, ...]
    _inference_checksum: str = field(init=False, repr=False, compare=False)
    _content_checksum: str = field(init=False, repr=False, compare=False)
    _canonical_json_cache: str | None = field(init=False, default=None, repr=False, compare=False)
    schema_version: str = field(default=PILOT_NUISANCE_SUMMARY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Sort observations and require a complete balanced paired pilot grid.

        Raises:
            TypeError: If an observation has the wrong record type.
            ValueError: If cells are duplicate, inconsistent, or incomplete.
        """
        object.__setattr__(self, "summary_id", require_slug(self.summary_id, "summary_id"))
        object.__setattr__(
            self,
            "preregistration_checksum",
            require_checksum(self.preregistration_checksum, "preregistration_checksum"),
        )
        preregistration = load_initial_preregistration()
        if self.preregistration_checksum != preregistration.content_checksum:
            msg = "Pilot nuisance evidence must reference the trusted checked-in preregistration."
            raise ValueError(msg)
        raw = tuple(self.observations)
        if not raw or not all(isinstance(item, PilotObservation) for item in raw):
            msg = "observations must contain PilotObservation values."
            raise TypeError(msg)
        ordered = tuple(sorted(raw, key=lambda item: item.identity))
        identities = tuple(item.identity for item in ordered)
        if len(identities) != len(set(identities)):
            msg = "Pilot observations must not duplicate a contrast/target/optimization-seed cell."
            raise ValueError(msg)

        ordered_bindings = _validate_exact_pilot_custody(
            preregistration,
            self.target_manifest,
            self.supplemental_target_manifest,
            self.pilot_plan,
            tuple(self.contrast_bindings),
            ordered,
        )

        target_identity: dict[str, tuple[str, str, int]] = {}
        grid_by_contrast: dict[str, set[tuple[str, int]]] = defaultdict(set)
        coverage: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for observation in ordered:
            identity = (observation.family_id, observation.stratum_id, observation.qubit_count)
            previous = target_identity.setdefault(observation.target_instance_id, identity)
            if previous != identity:
                msg = f"Target {observation.target_instance_id!r} has inconsistent pilot metadata."
                raise ValueError(msg)
            grid_by_contrast[observation.contrast_id].add((
                observation.target_instance_id,
                observation.optimization_seed,
            ))
            coverage[observation.contrast_id].add((observation.family_id, observation.stratum_id))
        expected_coverage = {
            (family_id, stratum_id) for family_id, strata in PRIMARY_FAMILY_STRATA.items() for stratum_id in strata
        }
        if set(grid_by_contrast) != set(FROZEN_CONTRAST_IDS):
            msg = "Pilot evidence must cover every frozen primary contrast."
            raise ValueError(msg)
        if any(coverage[contrast_id] != expected_coverage for contrast_id in FROZEN_CONTRAST_IDS):
            msg = "Every pilot contrast must cover every primary family and stratum."
            raise ValueError(msg)
        reference_grid = grid_by_contrast[FROZEN_CONTRAST_IDS[0]]
        if any(grid_by_contrast[contrast_id] != reference_grid for contrast_id in FROZEN_CONTRAST_IDS[1:]):
            msg = "All pilot contrasts must use the same paired target/optimization-seed grid."
            raise ValueError(msg)

        for contrast_id in FROZEN_CONTRAST_IDS:
            for family_id, strata in PRIMARY_FAMILY_STRATA.items():
                family_rows = [
                    item for item in ordered if item.contrast_id == contrast_id and item.family_id == family_id
                ]
                components = _family_components(family_rows)
                counts = cast("Mapping[str, int]", components["target_count_by_stratum"])
                if set(counts) != set(strata) or len(set(counts.values())) != 1 or sum(counts.values()) != 12:
                    msg = f"Pilot family {family_id!r} must contain exactly 12 targets balanced across strata."
                    raise ValueError(msg)
                if components["optimization_seed_count"] != 5:
                    msg = f"Pilot family {family_id!r} must use exactly five derived optimization seeds."
                    raise ValueError(msg)
        object.__setattr__(self, "contrast_bindings", ordered_bindings)
        object.__setattr__(self, "observations", ordered)
        object.__setattr__(self, "_inference_checksum", canonical_checksum(self.inference_projection))
        object.__setattr__(self, "_content_checksum", canonical_checksum(self._content_dict()))
        object.__setattr__(self, "_canonical_json_cache", None)

    @property
    def nuisance_by_contrast(self) -> dict[str, object]:
        """All derived family-level clustered nuisance components."""
        return {
            contrast_id: {
                family_id: _family_components([
                    item
                    for item in self.observations
                    if item.contrast_id == contrast_id and item.family_id == family_id
                ])
                for family_id in PRIMARY_TARGET_FAMILIES
            }
            for contrast_id in FROZEN_CONTRAST_IDS
        }

    @property
    def runtime_summary(self) -> dict[str, object]:
        """Deterministic bounded-runtime and Python-allocation diagnostics."""
        return {
            "observation_count": len(self.observations),
            "method_observation_count": 2 * len(self.observations),
            "failure_count": sum(item.treatment_failed + item.comparator_failed for item in self.observations),
            "treatment_failure_count": sum(item.treatment_failed for item in self.observations),
            "comparator_failure_count": sum(item.comparator_failed for item in self.observations),
            "wall_time_seconds_mean": math.fsum(item.wall_time_seconds for item in self.observations)
            / len(self.observations),
            "wall_time_seconds_max": max(item.wall_time_seconds for item in self.observations),
            "tracemalloc_peak_bytes_max": max(item.tracemalloc_peak_bytes for item in self.observations),
        }

    @property
    def inference_projection(self) -> dict[str, object]:
        """Canonical q6-only scientific projection.

        The full summary remains the audit-custody artifact and therefore
        retains its secondary-q12 manifest and complete mixed-width run plan.
        Sample-size inference instead binds only the primary manifest, q6
        jobs and observations, and contrast identities with the full-plan
        alias removed. Consequently, q12 result bytes and archive metadata
        cannot influence the confirmatory design or final-seal checksum.

        Returns:
            A JSON-native q6-only projection of all inferential inputs.
        """
        primary_jobs = tuple(job for job in self.pilot_plan.jobs if job.qubit_count == PILOT_QUBIT_COUNT)
        job_sources: dict[str, dict[str, object]] = {}
        observation_rows: list[dict[str, object]] = []
        for observation in self.observations:
            arms = (
                (
                    observation.treatment_job,
                    observation.treatment_outcome,
                    observation.treatment_result,
                ),
                (
                    observation.comparator_job,
                    observation.comparator_outcome,
                    observation.comparator_result,
                ),
            )
            for job, outcome, result in arms:
                job_sources.setdefault(
                    job.content_checksum,
                    {
                        "job_checksum": job.content_checksum,
                        "outcome_checksum": outcome.content_checksum,
                        "result_checksum": result.content_checksum,
                        "source_result_reference_checksum": result.source_result_reference_checksum,
                    },
                )
            observation_rows.append({
                "contrast_id": observation.contrast_id,
                "target_instance_id": observation.target_instance_id,
                "optimization_seed": observation.optimization_seed,
                "treatment_job_checksum": observation.treatment_job.content_checksum,
                "comparator_job_checksum": observation.comparator_job.content_checksum,
            })
        scientific_bindings = tuple(
            {
                "contrast_id": binding.contrast_id,
                "treatment_method_id": binding.treatment_method_id,
                "treatment_configuration_checksum": binding.treatment_configuration_checksum,
                "comparator_method_id": binding.comparator_method_id,
                "comparator_configuration_checksum": binding.comparator_configuration_checksum,
            }
            for binding in self.contrast_bindings
        )
        return {
            "schema_version": PILOT_INFERENCE_PROJECTION_SCHEMA_VERSION,
            "preregistration_checksum": self.preregistration_checksum,
            "primary_target_manifest_checksum": self.target_manifest.content_checksum,
            "primary_jobs": [
                {
                    "job_checksum": job.content_checksum,
                    "method_id": job.method_id,
                    "candidate_configuration_checksum": job.candidate_configuration_checksum,
                    "target_instance_id": job.target_instance_id,
                    "target_spec_checksum": job.target_spec_checksum,
                    "family_id": job.family_id,
                    "stratum_id": job.stratum_id,
                    "qubit_count": job.qubit_count,
                    "data_role": job.data_role,
                    "optimization_block_id": job.optimization_block_id,
                    "optimization_seed": job.optimization_seed,
                    "evaluation_seed": job.evaluation_seed,
                }
                for job in primary_jobs
            ],
            "primary_job_sources": [job_sources[key] for key in sorted(job_sources)],
            "scientific_contrast_bindings": list(scientific_bindings),
            "observations": observation_rows,
            "nuisance_by_contrast": self.nuisance_by_contrast,
        }

    @property
    def inference_checksum(self) -> str:
        """Checksum of q6 scientific inputs, excluding every q12 archive byte."""
        return self._inference_checksum

    @property
    def content_checksum(self) -> str:
        """Checksum of raw rows and every derived nuisance statistic."""
        return self._content_checksum

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered summary content."""
        return {
            "schema_version": self.schema_version,
            "summary_id": self.summary_id,
            "preregistration_checksum": self.preregistration_checksum,
            "target_manifest": self.target_manifest.to_dict(),
            "supplemental_target_manifest": self.supplemental_target_manifest.to_dict(),
            "pilot_plan": self.pilot_plan.to_dict(),
            "contrast_bindings": [binding.to_dict() for binding in self.contrast_bindings],
            "observations": [item.to_dict() for item in self.observations],
            "nuisance_by_contrast": self.nuisance_by_contrast,
            "runtime_summary": self.runtime_summary,
        }

    def to_dict(self) -> dict[str, object]:
        """Return sealed JSON-native summary data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        cached = self._canonical_json_cache
        if cached is None:
            cached = canonical_json(self.to_dict())
            object.__setattr__(self, "_canonical_json_cache", cached)  # noqa: PLC2801 - frozen derived cache
        return cached

    @classmethod
    def from_dict(cls, data: object) -> PilotNuisanceSummary:
        """Decode and verify a raw-row-backed nuisance summary.

        Args:
            data: Exact sealed summary mapping.

        Returns:
            The reconstructed deterministic summary.

        Raises:
            TypeError: If the raw observation collection is not a sequence.
            ValueError: If schemas, checksums, or derived statistics are inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_SUMMARY_KEYS, name="pilot nuisance summary")
        if mapping["schema_version"] != PILOT_NUISANCE_SUMMARY_SCHEMA_VERSION:
            msg = f"schema_version must be {PILOT_NUISANCE_SUMMARY_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        raw_observations = mapping["observations"]
        if isinstance(raw_observations, (str, bytes)) or not isinstance(raw_observations, Sequence):
            msg = "pilot nuisance summary observations must be a sequence."
            raise TypeError(msg)
        raw_bindings = mapping["contrast_bindings"]
        if isinstance(raw_bindings, (str, bytes)) or not isinstance(raw_bindings, Sequence):
            msg = "pilot nuisance summary contrast_bindings must be a sequence."
            raise TypeError(msg)
        training = _training_orchestration_module()
        summary = cls(
            summary_id=cast("str", mapping["summary_id"]),
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            target_manifest=TargetPopulationManifest.from_dict(mapping["target_manifest"]),
            supplemental_target_manifest=TargetPopulationManifest.from_dict(
                mapping["supplemental_target_manifest"],
            ),
            pilot_plan=training.TrainingRunPlan.from_dict(mapping["pilot_plan"]),
            contrast_bindings=tuple(PilotContrastBinding.from_dict(item) for item in raw_bindings),
            observations=tuple(PilotObservation.from_dict(item) for item in raw_observations),
        )
        if canonical_checksum(mapping["nuisance_by_contrast"]) != canonical_checksum(summary.nuisance_by_contrast):
            msg = "Pilot nuisance statistics are not derived from the sealed observations."
            raise ValueError(msg)
        if canonical_checksum(mapping["runtime_summary"]) != canonical_checksum(summary.runtime_summary):
            msg = "Pilot runtime statistics are not derived from the sealed observations."
            raise ValueError(msg)
        if summary.content_checksum != mapping["content_checksum"]:
            msg = "Pilot nuisance summary checksum changed during normalization."
            raise ValueError(msg)
        return summary

    @classmethod
    def from_json(cls, payload: str) -> PilotNuisanceSummary:
        """Decode a nuisance summary from canonical JSON text.

        Args:
            payload: Canonical sealed JSON.

        Returns:
            The reconstructed summary.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def build_pilot_nuisance_summary(
    preregistration: InitialPreregistration,
    target_manifest: TargetPopulationManifest,
    supplemental_target_manifest: TargetPopulationManifest,
    pilot_plan: TrainingRunPlan,
    contrast_bindings: Sequence[PilotContrastBinding],
    observations: Sequence[PilotObservation],
    *,
    summary_id: str = "phase2_pilot_nuisance_v1",
) -> PilotNuisanceSummary:
    """Build a deterministic nuisance summary bound to the governing protocol.

    Args:
        preregistration: Frozen Phase II preregistration.
        target_manifest: Exact seed-bearing q6 development manifest.
        supplemental_target_manifest: Exact seed-bearing secondary-q12 pilot manifest.
        pilot_plan: Exact paper-pilot job fan-out.
        contrast_bindings: Plan-linked treatment/control configuration bindings.
        observations: Complete paired development-pilot rows.
        summary_id: Stable artifact identifier.

    Returns:
        A checksum-sealed summary independent of input row order.

    Raises:
        TypeError: If ``preregistration`` has the wrong type.
        ValueError: If its contrast set differs from the WP22 calculation.
    """
    if not isinstance(preregistration, InitialPreregistration):
        msg = "preregistration must be an InitialPreregistration."
        raise TypeError(msg)
    contrast_definitions = cast(
        "Sequence[Mapping[str, object]]",
        preregistration.multiplicity_policy["contrast_definitions"],
    )
    contrast_ids = tuple(cast("str", item["contrast_id"]) for item in contrast_definitions)
    if contrast_ids != FROZEN_CONTRAST_IDS:
        msg = "The preregistered primary contrast set differs from the WP22 pilot calculation."
        raise ValueError(msg)
    return PilotNuisanceSummary(
        summary_id=summary_id,
        preregistration_checksum=preregistration.content_checksum,
        target_manifest=target_manifest,
        supplemental_target_manifest=supplemental_target_manifest,
        pilot_plan=pilot_plan,
        contrast_bindings=tuple(contrast_bindings),
        observations=tuple(observations),
    )


def _variance_upper_bound(variance: float, degrees_of_freedom: int) -> float:
    """Return the frozen one-sided 95% normal-variance upper confidence bound.

    Raises:
        PilotDesignInfeasibleError: If the variance quantile cannot be resolved.
    """
    resolved_variance = require_float(variance, "variance", minimum=0.0)
    df = require_int(degrees_of_freedom, "degrees_of_freedom", minimum=1)
    if resolved_variance <= 0.0:
        return 0.0
    lower_quantile = float(chi2.ppf(1.0 - VARIANCE_UCB_CONFIDENCE, df))
    if not math.isfinite(lower_quantile) or lower_quantile <= 0.0:
        msg = "variance_bound_unavailable"
        raise PilotDesignInfeasibleError(msg, "Could not resolve a finite variance bound.")
    return resolved_variance * df / lower_quantile


def _wilson_binary_variance_upper_bound(failures: int, count: int) -> float:
    """Return a one-sided Wilson-derived upper bound on Bernoulli variance.

    Raises:
        ValueError: If the failure count exceeds the observation count.
    """
    n = require_int(count, "count", minimum=1)
    failed = require_int(failures, "failures", minimum=0)
    if failed > n:
        msg = "failures cannot exceed count."
        raise ValueError(msg)
    probability = failed / n
    z_value = NormalDist().inv_cdf(VARIANCE_UCB_CONFIDENCE)
    denominator = 1.0 + z_value**2 / n
    center = probability + z_value**2 / (2.0 * n)
    radius = z_value * math.sqrt(probability * (1.0 - probability) / n + z_value**2 / (4.0 * n**2))
    upper_probability = min(1.0, (center + radius) / denominator)
    return 0.25 if upper_probability >= 0.5 else upper_probability * (1.0 - upper_probability)


def _component(
    nuisance_by_contrast: Mapping[str, object],
    contrast_id: str,
    family_id: str,
) -> Mapping[str, object]:
    """Resolve one verified summary component mapping.

    Returns:
        The exact family-level nuisance component.
    """
    families = require_mapping(
        nuisance_by_contrast[contrast_id],
        f"nuisance_by_contrast.{contrast_id}",
    )
    component = require_mapping(families[family_id], f"nuisance_by_contrast.{contrast_id}.{family_id}")
    require_exact_keys(component, _FAMILY_COMPONENT_KEYS, "pilot family nuisance component")
    return component


def _trajectory_variance_bound(summary: PilotNuisanceSummary) -> float:
    """Return the frozen q6-only Maurer--Pontil trajectory variance bound.

    Returns:
        The maximum Theorem-10 upper bound across exactly 720 q6 pilot jobs.

    Raises:
        ValueError: If the q6 job universe or a successful fixed batch differs.
    """
    result_by_job: dict[str, PilotJobResult] = {}
    for observation in summary.observations:
        for result in (observation.treatment_result, observation.comparator_result):
            previous = result_by_job.setdefault(result.job_checksum, result)
            if previous != result:
                msg = "Repeated q6 pilot jobs must retain identical raw result evidence."
                raise ValueError(msg)
    if len(result_by_job) != PILOT_PRIMARY_JOB_COUNT:
        msg = "Maurer--Pontil calibration requires exactly 720 primary-q6 pilot jobs."
        raise ValueError(msg)
    denominator = PILOT_PRIMARY_TRAJECTORY_COUNT - 1
    correction = math.sqrt(2.0 * math.log(PILOT_PRIMARY_JOB_COUNT / MAURER_PONTIL_FAILURE_PROBABILITY) / denominator)
    bounds: list[float] = []
    for result in result_by_job.values():
        if result.status == "failure":
            bounds.append(0.25)
            continue
        if result.trajectory_count != PILOT_PRIMARY_TRAJECTORY_COUNT:
            msg = "Every successful q6 pilot job must contain exactly 1,024 fresh trajectories."
            raise ValueError(msg)
        bounds.append(min(0.25, (math.sqrt(result.trajectory_mc_variance) + correction) ** 2))
    return max(bounds)


def _next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""
    required = require_int(value, "value", minimum=1)
    return 1 << (required - 1).bit_length()


def _contrast_effects(preregistration: InitialPreregistration) -> dict[str, float]:
    """Return the frozen planning distance from each null margin.

    Raises:
        PilotDesignInfeasibleError: If a contrast has no positive planning distance.
    """
    sample_policy = preregistration.sample_size_policy
    definitions = cast(
        "Sequence[Mapping[str, object]]",
        preregistration.multiplicity_policy["contrast_definitions"],
    )
    effects: dict[str, float] = {}
    for definition in definitions:
        contrast_id = cast("str", definition["contrast_id"])
        margin = cast("float", definition["margin"])
        if definition["hypothesis"] == "superiority":
            effect = cast("float", sample_policy["minimum_relevant_noisy_gain"]) - margin
        else:
            effect = cast("float", sample_policy["planned_noninferiority_true_difference"]) - margin
        if effect <= 0.0:
            msg = "nonpositive_planning_effect"
            raise PilotDesignInfeasibleError(
                msg,
                f"Contrast {contrast_id!r} has no positive planning distance from its null margin.",
            )
        effects[contrast_id] = effect
    return effects


def _design_metrics(
    preregistration: InitialPreregistration,
    *,
    targets_per_family: int,
    optimization_seed_count: int,
    trajectory_count: int,
    trajectory_variance: float,
    nuisance_by_contrast: Mapping[str, object],
) -> tuple[dict[str, float], float, float, float]:
    """Compute cluster-aware power and precision for one candidate design.

    Returns:
        Achieved powers, mean half-width, failure half-width, and trajectory MCSE.
    """
    weights = {
        family: cast("float", preregistration.target_family_weights[family]) for family in PRIMARY_TARGET_FAMILIES
    }
    planning_alpha = cast("float", preregistration.sample_size_policy["planning_alpha"])
    z_critical = NormalDist().inv_cdf(1.0 - planning_alpha)
    z_precision = NormalDist().inv_cdf(0.975)
    effects = _contrast_effects(preregistration)
    powers: dict[str, float] = {}
    mean_half_widths: list[float] = []
    failure_half_widths: list[float] = []
    for contrast_id in FROZEN_CONTRAST_IDS:
        estimator_variance = 0.0
        failure_variance = 0.0
        for family_id in PRIMARY_TARGET_FAMILIES:
            component = _component(nuisance_by_contrast, contrast_id, family_id)
            target_variance = _variance_upper_bound(
                cast("float", component["target_cluster_variance"]),
                cast("int", component["target_cluster_degrees_of_freedom"]),
            )
            optimizer_variance = _variance_upper_bound(
                cast("float", component["optimization_seed_variance"]),
                cast("int", component["optimization_seed_degrees_of_freedom"]),
            )
            weight_squared = weights[family_id] ** 2
            estimator_variance += weight_squared * (
                target_variance / targets_per_family
                + optimizer_variance / (targets_per_family * optimization_seed_count)
                + trajectory_variance / (targets_per_family * optimization_seed_count * trajectory_count)
            )

            failure_target_variance = _variance_upper_bound(
                cast("float", component["failure_target_cluster_variance"]),
                cast("int", component["target_cluster_degrees_of_freedom"]),
            )
            failure_seed_variance = _variance_upper_bound(
                cast("float", component["failure_optimization_seed_variance"]),
                cast("int", component["optimization_seed_degrees_of_freedom"]),
            )
            failure_observation_count = cast("int", component["failure_observation_count"])
            failures = round(cast("float", component["failure_rate"]) * failure_observation_count)
            binary_floor = _wilson_binary_variance_upper_bound(failures, failure_observation_count)
            clustered_failure = failure_target_variance / targets_per_family + failure_seed_variance / (
                targets_per_family * optimization_seed_count
            )
            failure_variance += weight_squared * max(
                clustered_failure,
                binary_floor / (2 * targets_per_family * optimization_seed_count),
            )
        standard_error = math.sqrt(max(0.0, estimator_variance))
        power = 1.0 if standard_error <= 0.0 else NormalDist().cdf(effects[contrast_id] / standard_error - z_critical)
        powers[contrast_id] = power
        mean_half_widths.append(z_precision * standard_error)
        failure_half_widths.append(z_precision * math.sqrt(max(0.0, failure_variance)))
    trajectory_mcse = math.sqrt(trajectory_variance / trajectory_count)
    return powers, max(mean_half_widths), max(failure_half_widths), trajectory_mcse


def _allocations(targets_per_family: int) -> tuple[SampleAllocation, ...]:
    """Return balanced q6 family/stratum allocations in stable order.

    Raises:
        PilotDesignInfeasibleError: If a family cannot receive a balanced allocation.
    """
    allocations: list[SampleAllocation] = []
    for family_id in PRIMARY_TARGET_FAMILIES:
        strata = PRIMARY_FAMILY_STRATA[family_id]
        if targets_per_family % len(strata) != 0:
            msg = "unbalanced_target_count"
            raise PilotDesignInfeasibleError(
                msg,
                f"{targets_per_family} targets cannot be balanced across {family_id!r} strata.",
            )
        per_stratum = targets_per_family // len(strata)
        allocations.extend(
            SampleAllocation(
                family_id=family_id,
                stratum_id=stratum_id,
                qubit_count=PILOT_QUBIT_COUNT,
                target_count=per_stratum,
            )
            for stratum_id in strata
        )
    return tuple(allocations)


def _select_design(
    preregistration: InitialPreregistration,
    summary: PilotNuisanceSummary,
    *,
    design_id: str,
    calculation_source_checksum: str,
    minimum_targets_per_family: int,
    minimum_optimization_seed_count: int,
    fixed_trajectory_count: int | None,
    reestimation_kind: Literal["initial", "blinded_nuisance_only"],
    reestimation_parent_checksum: str | None,
) -> SampleSizeDesign:
    """Search the bounded preregistered design grid deterministically.

    Returns:
        The minimum-work feasible frozen sample-size design.

    Raises:
        PilotDesignInfeasibleError: If no permitted allocation is feasible.
    """
    policy = preregistration.sample_size_policy
    trajectory_variance = _trajectory_variance_bound(summary)
    nuisance_by_contrast = summary.nuisance_by_contrast
    trajectory_target = cast("float", policy["trajectory_mcse_target"])
    required_trajectories = max(
        cast("int", policy["trajectory_count_min"]),
        math.ceil(trajectory_variance / trajectory_target**2),
    )
    selected_trajectory_count = (
        _next_power_of_two(required_trajectories) if fixed_trajectory_count is None else fixed_trajectory_count
    )
    maximum_trajectories = cast("int", policy["trajectory_count_max"])
    if selected_trajectory_count > maximum_trajectories:
        msg = "trajectory_budget_exceeded"
        raise PilotDesignInfeasibleError(
            msg,
            "The pilot MC variance requires more trajectories than the preregistered maximum.",
        )
    if fixed_trajectory_count is not None and selected_trajectory_count < required_trajectories:
        msg = "fixed_trajectory_budget_inadequate"
        raise PilotDesignInfeasibleError(
            msg,
            "Nuisance re-estimation cannot increase the already fixed per-cell trajectory count.",
        )

    allowed_seeds = tuple(
        seed
        for seed in cast("Sequence[int]", policy["allowed_optimization_seed_counts"])
        if seed >= minimum_optimization_seed_count
    )
    if not allowed_seeds:
        msg = "optimization_seed_bound_exceeded"
        raise PilotDesignInfeasibleError(
            msg,
            "No preregistered optimization-seed count satisfies the nondecreasing bound.",
        )
    increment = cast("int", policy["target_count_increment"])
    lower_targets = max(cast("int", policy["minimum_targets_per_family"]), minimum_targets_per_family)
    if lower_targets % increment:
        lower_targets += increment - lower_targets % increment
    upper_targets = cast("int", policy["maximum_targets_per_family"])
    candidates: list[tuple[int, int, int, dict[str, float], float, float, float]] = []
    for targets in range(lower_targets, upper_targets + 1, increment):
        for seeds in allowed_seeds:
            powers, mean_half_width, failure_half_width, trajectory_mcse = _design_metrics(
                preregistration,
                targets_per_family=targets,
                optimization_seed_count=seeds,
                trajectory_count=selected_trajectory_count,
                trajectory_variance=trajectory_variance,
                nuisance_by_contrast=nuisance_by_contrast,
            )
            if (
                all(power >= cast("float", policy["power"]) for power in powers.values())
                and mean_half_width <= cast("float", policy["target_mean_half_width"])
                and failure_half_width <= cast("float", policy["failure_rate_half_width"])
                and trajectory_mcse <= trajectory_target
            ):
                candidates.append((
                    targets * seeds,
                    targets,
                    seeds,
                    powers,
                    mean_half_width,
                    failure_half_width,
                    trajectory_mcse,
                ))
    if not candidates:
        msg = "sample_size_bounds_exhausted"
        raise PilotDesignInfeasibleError(
            msg,
            "No target/optimization-seed allocation meets every frozen power and precision requirement.",
        )
    _, targets, seeds, powers, mean_half_width, failure_half_width, trajectory_mcse = min(
        candidates,
        key=operator.itemgetter(0, 1, 2),
    )
    return SampleSizeDesign(
        design_id=design_id,
        preregistration_checksum=preregistration.content_checksum,
        pilot_nuisance_summary_checksum=summary.inference_checksum,
        calculation_method_id=PILOT_CALCULATION_METHOD_ID,
        calculation_source_checksum=calculation_source_checksum,
        contrast_set_checksum=preregistration.contrast_set_checksum,
        target_population_configuration_checksum=preregistration.target_population_configuration_checksum,
        allocations=_allocations(targets),
        optimization_seed_count=seeds,
        fixed_test_trajectory_count=selected_trajectory_count,
        achieved_power_by_contrast=powers,
        expected_primary_mean_half_width=mean_half_width,
        expected_overall_failure_rate_half_width=failure_half_width,
        expected_trajectory_mcse=trajectory_mcse,
        reestimation_kind=reestimation_kind,
        reestimation_parent_checksum=reestimation_parent_checksum,
    )


def build_cluster_aware_paired_difference_v1(
    preregistration: InitialPreregistration,
    summary: PilotNuisanceSummary,
    *,
    design_id: str = "phase2_initial_sample_size_v1",
    calculation_source_checksum: str = PILOT_CALCULATION_SOURCE_CHECKSUM,
) -> SampleSizeDesign:
    """Build the initial bounded cluster-aware confirmatory sample-size design.

    Args:
        preregistration: Frozen Phase II protocol.
        summary: Complete paired pilot nuisance evidence.
        design_id: Stable sample-size artifact identifier.
        calculation_source_checksum: Checksum of the frozen calculation source.

    Returns:
        The smallest-work feasible design on the preregistered grid.

    Raises:
        TypeError: If a protocol or summary has the wrong record type.
        ValueError: If pilot and protocol identities disagree.
    """
    if not isinstance(preregistration, InitialPreregistration):
        msg = "preregistration must be an InitialPreregistration."
        raise TypeError(msg)
    if not isinstance(summary, PilotNuisanceSummary):
        msg = "summary must be a PilotNuisanceSummary."
        raise TypeError(msg)
    if summary.preregistration_checksum != preregistration.content_checksum:
        msg = "Pilot nuisance evidence does not reference the supplied preregistration."
        raise ValueError(msg)
    if preregistration.sample_size_policy["method"] != PILOT_CALCULATION_METHOD_ID:
        msg = "The preregistration does not authorize the WP22 calculation method."
        raise ValueError(msg)
    checksum = require_checksum(calculation_source_checksum, "calculation_source_checksum")
    return _select_design(
        preregistration,
        summary,
        design_id=design_id,
        calculation_source_checksum=checksum,
        minimum_targets_per_family=cast("int", preregistration.sample_size_policy["minimum_targets_per_family"]),
        minimum_optimization_seed_count=cast(
            "int",
            preregistration.sample_size_policy["minimum_optimization_seed_count"],
        ),
        fixed_trajectory_count=None,
        reestimation_kind="initial",
        reestimation_parent_checksum=None,
    )


def reestimate_cluster_aware_paired_difference_v1(
    preregistration: InitialPreregistration,
    summary: PilotNuisanceSummary,
    parent_design: SampleSizeDesign,
    *,
    information_fraction: float = 0.5,
    design_id: str = "phase2_blinded_reestimation_v1",
    calculation_source_checksum: str = PILOT_CALCULATION_SOURCE_CHECKSUM,
) -> SampleSizeDesign:
    """Perform the one allowed halfway nuisance-only nondecreasing re-estimation.

    The calculation reads variances and failure indicators only. It cannot use
    candidate means, change the fixed trajectory count, reduce target or seed
    counts, or create a second re-estimation generation.

    Args:
        preregistration: Frozen Phase II protocol.
        summary: Blinded nuisance observations at the halfway point.
        parent_design: Initial sealed sample-size design.
        information_fraction: Frozen halfway trigger, exactly ``0.5``.
        design_id: Stable re-estimation artifact identifier.
        calculation_source_checksum: Checksum of the frozen calculation source.

    Returns:
        A parent-linked nondecreasing sample-size design.

    Raises:
        TypeError: If the parent design has the wrong record type.
        PilotDesignInfeasibleError: If re-estimation is disallowed or infeasible.
        ValueError: If identities or the halfway trigger disagree.
    """
    if not isinstance(parent_design, SampleSizeDesign):
        msg = "parent_design must be a SampleSizeDesign."
        raise TypeError(msg)
    if parent_design.reestimation_kind != "initial" or parent_design.reestimation_parent_checksum is not None:
        msg_0 = "reestimation_limit_exceeded"
        raise PilotDesignInfeasibleError(
            msg_0,
            "Only one nuisance-only re-estimation may follow the initial design.",
        )
    fraction = require_float(information_fraction, "information_fraction", minimum=0.0, maximum=1.0)
    expected_fraction = cast("float", preregistration.sample_size_policy["reestimation_trigger_fraction"])
    if not math.isclose(fraction, expected_fraction, rel_tol=0.0, abs_tol=0.0):
        msg = "Blinded nuisance re-estimation is permitted only at the frozen halfway trigger."
        raise ValueError(msg)
    if (
        parent_design.preregistration_checksum != preregistration.content_checksum
        or parent_design.calculation_method_id != PILOT_CALCULATION_METHOD_ID
        or parent_design.contrast_set_checksum != preregistration.contrast_set_checksum
        or parent_design.target_population_configuration_checksum
        != preregistration.target_population_configuration_checksum
    ):
        msg = "The parent sample-size design does not match the supplied preregistration."
        raise ValueError(msg)
    if summary.preregistration_checksum != preregistration.content_checksum:
        msg = "Re-estimation nuisance evidence does not reference the supplied preregistration."
        raise ValueError(msg)
    parent_counts = {cast("int", value) for value in parent_design.target_count_by_family.values()}
    if len(parent_counts) != 1:
        msg = "The parent design is not balanced by family."
        raise ValueError(msg)
    checksum = require_checksum(calculation_source_checksum, "calculation_source_checksum")
    if checksum != parent_design.calculation_source_checksum:
        msg = "Nuisance re-estimation must reuse the parent's frozen calculation source."
        raise ValueError(msg)
    return _select_design(
        preregistration,
        summary,
        design_id=design_id,
        calculation_source_checksum=checksum,
        minimum_targets_per_family=next(iter(parent_counts)),
        minimum_optimization_seed_count=parent_design.optimization_seed_count,
        fixed_trajectory_count=parent_design.fixed_test_trajectory_count,
        reestimation_kind="blinded_nuisance_only",
        reestimation_parent_checksum=parent_design.content_checksum,
    )


__all__ = [
    "FROZEN_CONTRAST_IDS",
    "PILOT_CALCULATION_METHOD_ID",
    "PILOT_CALCULATION_SOURCE_CHECKSUM",
    "PILOT_CONTRAST_BINDING_SCHEMA_VERSION",
    "PILOT_DATA_ROLE",
    "PILOT_EVALUATION_EVIDENCE_SCHEMA_VERSION",
    "PILOT_INFERENCE_PROJECTION_SCHEMA_VERSION",
    "PILOT_JOB_RESULT_SCHEMA_VERSION",
    "PILOT_NUISANCE_SUMMARY_SCHEMA_VERSION",
    "PILOT_OBSERVATION_SCHEMA_VERSION",
    "PilotContrastBinding",
    "PilotDesignInfeasibleError",
    "PilotEvaluationEvidence",
    "PilotJobResult",
    "PilotNuisanceSummary",
    "PilotObservation",
    "ProductionPilotCustody",
    "ProductionPilotJobRecord",
    "build_cluster_aware_paired_difference_v1",
    "build_pilot_nuisance_summary",
    "reestimate_cluster_aware_paired_difference_v1",
]
