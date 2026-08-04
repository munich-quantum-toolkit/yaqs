# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Frozen WP22 primary analysis for cell-level confirmatory evidence."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Literal, cast

from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .execution_context import ConfirmationExecutionContext
from .production_executors import ProductionNumericalEvidence, ResultArtifactRef
from .protocol import (
    PRIMARY_TARGET_FAMILIES,
    FinalConfirmationSeal,
    load_initial_preregistration,
)
from .result_custody import (
    PRODUCTION_RESULT_CUSTODY_SCHEMA_VERSION,
    ProductionResultCustody,
    TrajectoryFidelityEvidence,
    production_noisy_fidelity,
    reopen_confirmatory_production_attempt,
)
from .targets import TargetPopulationManifest
from .training_orchestration import (
    TRAINING_JOB_OUTCOME_SCHEMA_VERSION,
    ConfirmExecutionRequest,
    TrainingJob,
    TrainingJobOutcome,
    TrainingRunPlan,
    build_confirm_execution_context,
    confirmatory_evaluation_policy_checksum,
    load_training_job_outcome_history,
    validate_confirm_execution_request,
)
from .validation import (
    require_checksum,
    require_float,
    require_int,
    require_mapping,
    require_slug,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .protocol import (
        FinalConfigurationExecutionManifest,
        InitialPreregistration,
    )

CONFIRMATORY_OBSERVATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.confirmatory_observation.v2"
CONFIRMATORY_EVALUATION_ARTIFACT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.confirmatory_evaluation_artifact.v1"
CONFIRMATORY_RESULT_ARTIFACT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.confirmatory_result_artifact.v4"
CONFIRMATORY_PRODUCTION_ATTEMPT_RECEIPT_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.confirmatory_production_attempt_receipt.v1"
)
PRIMARY_ANALYSIS_RESULT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.primary_analysis_result.v3"
PRIMARY_ANALYSIS_POLICY_ID = "family_equal_target_cluster_holm_v1"
CONFIRMATORY_DATA_ROLE = "confirmatory"
CONFIRMATORY_QUBIT_COUNT = 6
FAMILYWISE_ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95
_CI_Z = NormalDist().inv_cdf(0.5 + CONFIDENCE_LEVEL / 2.0)

_CONFIRMATORY_EVALUATION_KEYS = frozenset({
    "schema_version",
    "job_checksum",
    "data_role",
    "evaluation_seed",
    "evaluation_policy_checksum",
    "test_trajectory_count",
    "primary_noise_condition_checksum",
    "primary_resource_budget_checksum",
    "trajectory_evidence",
    "content_checksum",
})
_CONFIRMATORY_RESULT_KEYS = frozenset({
    "schema_version",
    "source_evaluation",
    "source_result_reference_checksum",
    "source_production_evidence_checksum",
    "fresh_test_noisy_fidelity",
    "content_checksum",
})
_CONFIRMATORY_PRODUCTION_ATTEMPT_RECEIPT_KEYS = frozenset({
    "schema_version",
    "job_checksum",
    "request_checksum",
    "job_outcome_checksum",
    "status",
    "result_reference",
    "production_evidence",
    "raw_trajectory_payload",
    "raw_trajectory_document_checksum",
    "resource_payload",
    "resource_document_checksum",
    "pilot_diagnostic_checksums",
    "production_custody_checksum",
    "content_checksum",
})

_OBSERVATION_KEYS = frozenset({
    "schema_version",
    "data_role",
    "configuration_checksum",
    "target_manifest_checksum",
    "target_spec_checksum",
    "family_id",
    "stratum_id",
    "target_instance_id",
    "qubit_count",
    "optimization_seed",
    "evaluation_seed",
    "job_id",
    "job_checksum",
    "job_outcome_checksum",
    "primary_noise_condition_checksum",
    "primary_resource_budget_checksum",
    "test_trajectory_count",
    "status",
    "fresh_test_noisy_fidelity",
    "failure_code",
    "evaluation_evidence_checksum",
    "result_schema_version",
    "result_record_checksum",
    "content_checksum",
})
_RESULT_KEYS = frozenset({
    "schema_version",
    "analysis_policy",
    "final_seal",
    "target_manifest_checksum",
    "run_plan_checksum",
    "job_outcomes",
    "job_outcome_checksums",
    "production_attempt_receipts",
    "production_attempt_receipt_checksums",
    "confirmatory_results",
    "confirmatory_result_checksums",
    "observations",
    "contrast_results",
    "failure_rate_results",
    "content_checksum",
})
_ANALYSIS_POLICY = {
    "policy_id": PRIMARY_ANALYSIS_POLICY_ID,
    "endpoint": "fresh_test_noisy_fidelity",
    "failed_fidelity": 0.0,
    "cluster_unit": "target_instance",
    "nested_replicate": "optimization_seed",
    "target_weighting_within_family": "equal",
    "family_weighting": "equal",
    "confidence_level": CONFIDENCE_LEVEL,
    "multiplicity_method": "holm",
    "familywise_alpha": FAMILYWISE_ALPHA,
    "p_value_sidedness": "one_sided_against_sealed_margin",
    "failure_interval_method": "cluster_normal_enveloped_by_target_cluster_wilson_score",
}


def _confirmatory_trajectory_context_checksum(
    *,
    job_checksum: str,
    evaluation_policy_checksum: str,
    test_trajectory_count: int,
    primary_noise_condition_checksum: str,
    primary_resource_budget_checksum: str,
) -> str:
    """Return the exact context bound to raw confirmatory trajectories."""
    return canonical_checksum({
        "schema_version": "yaqs.state_preparation.phase2.confirmatory_trajectory_context.v1",
        "job_checksum": require_checksum(job_checksum, "job_checksum"),
        "evaluation_policy_checksum": require_checksum(
            evaluation_policy_checksum,
            "evaluation_policy_checksum",
        ),
        "test_trajectory_count": require_int(test_trajectory_count, "test_trajectory_count", minimum=2),
        "primary_noise_condition_checksum": require_checksum(
            primary_noise_condition_checksum,
            "primary_noise_condition_checksum",
        ),
        "primary_resource_budget_checksum": require_checksum(
            primary_resource_budget_checksum,
            "primary_resource_budget_checksum",
        ),
    })


@dataclass(frozen=True, slots=True)
class ConfirmatoryEvaluationArtifact:
    """Raw fixed-trajectory evaluation for one exact confirmatory job."""

    job_checksum: str
    evaluation_seed: int
    evaluation_policy_checksum: str
    test_trajectory_count: int
    primary_noise_condition_checksum: str
    primary_resource_budget_checksum: str
    trajectory_evidence: TrajectoryFidelityEvidence
    data_role: Literal["confirmatory"] = CONFIRMATORY_DATA_ROLE
    schema_version: str = field(default=CONFIRMATORY_EVALUATION_ARTIFACT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate policy provenance and the exact raw trajectory ensemble.

        Raises:
            TypeError: If raw evidence has the wrong typed schema.
            ValueError: If a policy, identity, role, seed, or budget is invalid.
        """
        if self.data_role != CONFIRMATORY_DATA_ROLE:
            msg = "Confirmatory evaluation artifacts must use the confirmatory data role."
            raise ValueError(msg)
        for name in (
            "job_checksum",
            "evaluation_policy_checksum",
            "primary_noise_condition_checksum",
            "primary_resource_budget_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(
            self,
            "evaluation_seed",
            require_int(self.evaluation_seed, "evaluation_seed", minimum=0),
        )
        count = require_int(self.test_trajectory_count, "test_trajectory_count", minimum=2)
        object.__setattr__(self, "test_trajectory_count", count)
        if not isinstance(self.trajectory_evidence, TrajectoryFidelityEvidence):
            msg = "Confirmatory evaluations require raw TrajectoryFidelityEvidence."
            raise TypeError(msg)
        expected_context = _confirmatory_trajectory_context_checksum(
            job_checksum=self.job_checksum,
            evaluation_policy_checksum=self.evaluation_policy_checksum,
            test_trajectory_count=count,
            primary_noise_condition_checksum=self.primary_noise_condition_checksum,
            primary_resource_budget_checksum=self.primary_resource_budget_checksum,
        )
        if (
            self.trajectory_evidence.evaluation_context_checksum != expected_context
            or self.trajectory_evidence.data_role != self.data_role
            or self.trajectory_evidence.evaluation_seed != self.evaluation_seed
            or len(self.trajectory_evidence.trajectory_fidelities) != count
        ):
            msg = "Raw confirmatory trajectories differ from the exact job, policy, role, seed, or budget."
            raise ValueError(msg)

    @property
    def fresh_test_noisy_fidelity(self) -> float:
        """WP22E float64 mean derived only from raw trajectories."""
        return production_noisy_fidelity(self.trajectory_evidence.trajectory_fidelities)

    @classmethod
    def create(
        cls,
        job: TrainingJob,
        seal: FinalConfirmationSeal,
        *,
        trajectory_fidelities: Sequence[float],
    ) -> ConfirmatoryEvaluationArtifact:
        """Create raw evidence from the exact seal-linked request policy.

        Returns:
            The raw-trajectory confirmatory evaluation artifact.

        Raises:
            TypeError: If the job or seal has the wrong typed schema.
            ValueError: If the job is not an exact seal-linked confirmation cell.
        """
        if not isinstance(job, TrainingJob) or not isinstance(seal, FinalConfirmationSeal):
            msg = "job and seal must be typed TrainingJob and FinalConfirmationSeal records."
            raise TypeError(msg)
        request = job.confirm_execution_request
        if (
            job.preset != "paper-confirm"
            or not isinstance(request, ConfirmExecutionRequest)
            or request.final_confirmation_seal_checksum != seal.content_checksum
        ):
            msg = "Confirmatory evaluations require an exact seal-linked paper-confirm job."
            raise ValueError(msg)
        job_checksum = job.content_checksum
        evaluation_policy_checksum = confirmatory_evaluation_policy_checksum(request)
        count = seal.fixed_test_trajectory_count
        noise_checksum = canonical_checksum(seal.primary_noise_condition)
        resource_checksum = canonical_checksum(seal.primary_resource_budget)
        return cls(
            job_checksum=job_checksum,
            evaluation_seed=job.evaluation_seed,
            evaluation_policy_checksum=evaluation_policy_checksum,
            test_trajectory_count=count,
            primary_noise_condition_checksum=noise_checksum,
            primary_resource_budget_checksum=resource_checksum,
            trajectory_evidence=TrajectoryFidelityEvidence(
                evaluation_context_checksum=_confirmatory_trajectory_context_checksum(
                    job_checksum=job_checksum,
                    evaluation_policy_checksum=evaluation_policy_checksum,
                    test_trajectory_count=count,
                    primary_noise_condition_checksum=noise_checksum,
                    primary_resource_budget_checksum=resource_checksum,
                ),
                data_role="confirmatory",
                evaluation_seed=job.evaluation_seed,
                trajectory_fidelities=tuple(trajectory_fidelities),
            ),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered raw evaluation field."""
        return {
            "schema_version": self.schema_version,
            "job_checksum": self.job_checksum,
            "data_role": self.data_role,
            "evaluation_seed": self.evaluation_seed,
            "evaluation_policy_checksum": self.evaluation_policy_checksum,
            "test_trajectory_count": self.test_trajectory_count,
            "primary_noise_condition_checksum": self.primary_noise_condition_checksum,
            "primary_resource_budget_checksum": self.primary_resource_budget_checksum,
            "trajectory_evidence": self.trajectory_evidence.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact raw confirmatory evaluation."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native raw evaluation data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed raw evaluation JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ConfirmatoryEvaluationArtifact:
        """Decode and checksum-verify one raw confirmatory evaluation.

        Returns:
            The verified raw evaluation artifact.

        Raises:
            ValueError: If its schema or checksum differs.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_CONFIRMATORY_EVALUATION_KEYS,
            name="confirmatory evaluation artifact",
        )
        if mapping["schema_version"] != CONFIRMATORY_EVALUATION_ARTIFACT_SCHEMA_VERSION:
            msg = "Confirmatory evaluation artifact uses an unsupported schema version."
            raise ValueError(msg)
        evaluation = cls(
            job_checksum=cast("str", mapping["job_checksum"]),
            evaluation_seed=cast("int", mapping["evaluation_seed"]),
            evaluation_policy_checksum=cast("str", mapping["evaluation_policy_checksum"]),
            test_trajectory_count=cast("int", mapping["test_trajectory_count"]),
            primary_noise_condition_checksum=cast("str", mapping["primary_noise_condition_checksum"]),
            primary_resource_budget_checksum=cast("str", mapping["primary_resource_budget_checksum"]),
            trajectory_evidence=TrajectoryFidelityEvidence.from_dict(mapping["trajectory_evidence"]),
            data_role=cast("Literal['confirmatory']", mapping["data_role"]),
        )
        if mapping["content_checksum"] != evaluation.content_checksum:
            msg = "Confirmatory evaluation artifact checksum changed during normalization."
            raise ValueError(msg)
        return evaluation

    @classmethod
    def from_json(cls, payload: str) -> ConfirmatoryEvaluationArtifact:
        """Decode canonical checksum-sealed raw evaluation JSON.

        Returns:
            The verified raw evaluation artifact.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _validate_production_confirmation_custody(
    job: TrainingJob,
    custody: ProductionResultCustody,
    confirmation_context: ConfirmationExecutionContext,
) -> ConfirmExecutionRequest:
    """Validate one real reopened attempt against its exact confirm request.

    Returns:
        The nested request authenticated by the production custody.

    Raises:
        TypeError: If the job, custody, or authority has the wrong typed schema.
        ValueError: If any production, target, policy, role, seed, or count link differs.
    """
    if not isinstance(job, TrainingJob):
        msg = "job must be a TrainingJob."
        raise TypeError(msg)
    if not isinstance(custody, ProductionResultCustody):
        msg = "custody must be reopened ProductionResultCustody."
        raise TypeError(msg)
    if not isinstance(confirmation_context, ConfirmationExecutionContext):
        msg = "confirmation_context must be a ConfirmationExecutionContext."
        raise TypeError(msg)
    request = job.confirm_execution_request
    if not isinstance(request, ConfirmExecutionRequest) or job.preset != "paper-confirm":
        msg = "Production confirmation custody requires an exact paper-confirm job."
        raise ValueError(msg)
    reference = custody.reference
    evidence = custody.production_evidence
    if evidence.artifact_kind == "synthetic_confirmation":
        msg = "Synthetic custody cannot authorize production primary analysis."
        raise ValueError(msg)
    if evidence.artifact_kind != confirmation_context.artifact_kind(request):
        msg = "Production confirmation artifact kind differs from its sealed executable binding."
        raise ValueError(msg)
    target = evidence.target_identity
    if (
        reference.attempt != 1
        or evidence.attempt != 1
        or reference.status != evidence.status
        or reference.job_checksum != request.content_checksum
        or evidence.job_checksum != request.content_checksum
        or reference.execution_source_manifest_checksum != request.execution_source_checksum
        or reference.source_fingerprint_checksum != request.execution_source_checksum
        or evidence.execution_source_manifest_checksum != request.execution_source_checksum
        or evidence.source_fingerprint_checksum != request.execution_source_checksum
        or evidence.executable_binding_checksum != request.executable_binding_checksum
        or evidence.scheduled_program_checksum != confirmation_context.scheduled_program_checksum(request)
        or evidence.derived_metrics.get("strategy_schedule_checksum") != request.hyperparameters_checksum
        or evidence.evaluation_policy_checksum != confirmatory_evaluation_policy_checksum(request)
        or target.get("target_instance_id") != request.target_instance_id
        or target.get("target_instance_spec_checksum") != request.target_spec_checksum
        or target.get("target_manifest_checksum") != request.target_manifest_checksum
        or target.get("family_id") != request.family_id
        or target.get("stratum_id") != request.stratum_id
        or target.get("qubit_count") != request.qubit_count
    ):
        msg = "Production confirmation custody differs from the exact sealed request."
        raise ValueError(msg)
    if evidence.status == "success" and (
        evidence.derived_metrics.get("evaluation_data_role") != CONFIRMATORY_DATA_ROLE
        or evidence.derived_metrics.get("evaluation_seed_domain") != "confirmatory_test"
        or evidence.derived_metrics.get("evaluation_seed") != request.evaluation_seed
        or evidence.derived_metrics.get("trajectory_count") != request.fixed_test_trajectory_count
    ):
        msg = "Production confirmation metrics differ from the sealed role, seed, or fixed count."
        raise ValueError(msg)
    raw = custody.raw_trajectory_payload
    if raw is not None:
        raw_count = require_int(raw.get("trajectory_count"), "trajectory_count", minimum=1)
        if (
            raw.get("job_checksum") != request.content_checksum
            or raw.get("evaluation_policy_checksum") != evidence.evaluation_policy_checksum
            or raw.get("data_role") != CONFIRMATORY_DATA_ROLE
            or raw.get("seed_domain") != "confirmatory_test"
            or raw.get("evaluation_seed") != request.evaluation_seed
            or raw_count > request.fixed_test_trajectory_count
            or (evidence.status == "success" and raw_count != request.fixed_test_trajectory_count)
        ):
            msg = "Production confirmation trajectories differ from the sealed role, seed, or count."
            raise ValueError(msg)
    elif evidence.status == "success":
        msg = "Successful production confirmation lacks manifest-addressed raw trajectories."
        raise ValueError(msg)
    return request


@dataclass(frozen=True, slots=True)
class ConfirmatoryResultArtifact:
    """Authoritative result receipt embedding its exact raw evaluation."""

    source_evaluation: ConfirmatoryEvaluationArtifact
    source_result_reference_checksum: str
    source_production_evidence_checksum: str
    schema_version: str = field(default=CONFIRMATORY_RESULT_ARTIFACT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require a typed raw evaluation and both production provenance roots.

        Raises:
            TypeError: If the source has the wrong typed schema.
        """
        if not isinstance(self.source_evaluation, ConfirmatoryEvaluationArtifact):
            msg = "source_evaluation must be a ConfirmatoryEvaluationArtifact."
            raise TypeError(msg)
        for name in ("source_result_reference_checksum", "source_production_evidence_checksum"):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))

    @property
    def job_checksum(self) -> str:
        """Exact seal-generated job identity."""
        return self.source_evaluation.job_checksum

    @property
    def data_role(self) -> Literal["confirmatory"]:
        """Fixed confirmatory data role."""
        return self.source_evaluation.data_role

    @property
    def evaluation_seed(self) -> int:
        """Fixed fresh-test evaluation seed."""
        return self.source_evaluation.evaluation_seed

    @property
    def evaluation_policy_checksum(self) -> str:
        """Sealed evaluation policy identity."""
        return self.source_evaluation.evaluation_policy_checksum

    @property
    def test_trajectory_count(self) -> int:
        """Fixed number of raw test trajectories."""
        return self.source_evaluation.test_trajectory_count

    @property
    def primary_noise_condition_checksum(self) -> str:
        """Sealed primary noise identity."""
        return self.source_evaluation.primary_noise_condition_checksum

    @property
    def primary_resource_budget_checksum(self) -> str:
        """Sealed primary resource-budget identity."""
        return self.source_evaluation.primary_resource_budget_checksum

    @property
    def evaluation_evidence_checksum(self) -> str:
        """Identity of the embedded typed raw evaluation source."""
        return self.source_evaluation.content_checksum

    @property
    def fresh_test_noisy_fidelity(self) -> float:
        """Mean mechanically dereferenced from raw trajectories."""
        return self.source_evaluation.fresh_test_noisy_fidelity

    @classmethod
    def create(
        cls,
        job: TrainingJob,
        seal: FinalConfirmationSeal,
        custody: ProductionResultCustody,
        confirmation_context: ConfirmationExecutionContext,
    ) -> ConfirmatoryResultArtifact:
        """Create a receipt only from reopened non-synthetic production custody.

        Returns:
            The checksum-sealed authoritative result receipt.

        Raises:
            TypeError: If an input has the wrong typed schema.
            ValueError: If custody differs from the job or sealed policy.
        """
        if not isinstance(job, TrainingJob) or not isinstance(seal, FinalConfirmationSeal):
            msg = "job and seal must be typed TrainingJob and FinalConfirmationSeal records."
            raise TypeError(msg)
        if not isinstance(custody, ProductionResultCustody):
            msg = "custody must be reopened ProductionResultCustody."
            raise TypeError(msg)
        _validate_production_confirmation_custody(job, custody, confirmation_context)
        evidence = custody.production_evidence
        values = custody.trajectory_fidelities
        if evidence.status != "success" or values is None:
            msg = "Confirmatory result artifacts require successful production trajectories."
            raise ValueError(msg)
        source_evaluation = ConfirmatoryEvaluationArtifact.create(
            job,
            seal,
            trajectory_fidelities=values,
        )
        return cls(
            source_evaluation=source_evaluation,
            source_result_reference_checksum=custody.reference.content_checksum,
            source_production_evidence_checksum=evidence.content_checksum,
        )

    def _content_dict(self) -> dict[str, object]:
        """Return the typed source and its mechanically derived endpoint."""
        return {
            "schema_version": self.schema_version,
            "source_evaluation": self.source_evaluation.to_dict(),
            "source_result_reference_checksum": self.source_result_reference_checksum,
            "source_production_evidence_checksum": self.source_production_evidence_checksum,
            "fresh_test_noisy_fidelity": self.fresh_test_noisy_fidelity,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact typed source and derived result."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native result data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed result JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ConfirmatoryResultArtifact:
        """Decode and recompute one authoritative confirmatory result.

        Returns:
            The verified result artifact.

        Raises:
            ValueError: If its schema, derived endpoint, or checksum differs.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_CONFIRMATORY_RESULT_KEYS,
            name="confirmatory result artifact",
        )
        if mapping["schema_version"] != CONFIRMATORY_RESULT_ARTIFACT_SCHEMA_VERSION:
            msg = "Confirmatory result artifact uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            source_evaluation=ConfirmatoryEvaluationArtifact.from_dict(mapping["source_evaluation"]),
            source_result_reference_checksum=cast(
                "str",
                mapping["source_result_reference_checksum"],
            ),
            source_production_evidence_checksum=cast(
                "str",
                mapping["source_production_evidence_checksum"],
            ),
        )
        supplied_fidelity = require_float(
            mapping["fresh_test_noisy_fidelity"],
            "fresh_test_noisy_fidelity",
            minimum=0.0,
            maximum=1.0,
        )
        if float(supplied_fidelity).hex() != float(result.fresh_test_noisy_fidelity).hex():
            msg = "Confirmatory result fidelity is not derived from its raw trajectory evidence."
            raise ValueError(msg)
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Confirmatory result artifact checksum changed during normalization."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> ConfirmatoryResultArtifact:
        """Decode canonical checksum-sealed result JSON.

        Returns:
            The verified result artifact.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class ConfirmatoryProductionAttemptReceipt:
    """Portable audit projection of one reopened first production attempt."""

    job_checksum: str
    request_checksum: str
    job_outcome_checksum: str
    status: Literal["success", "failure"]
    result_reference: ResultArtifactRef
    production_evidence: ProductionNumericalEvidence
    raw_trajectory_payload: Mapping[str, object] | None
    raw_trajectory_document_checksum: str | None
    resource_payload: Mapping[str, object]
    resource_document_checksum: str
    pilot_diagnostic_checksums: tuple[str, ...]
    production_custody_checksum: str
    schema_version: str = field(default=CONFIRMATORY_PRODUCTION_ATTEMPT_RECEIPT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Verify the typed reference, evidence, and every custody root.

        Raises:
            TypeError: If a typed source or checksum collection has the wrong shape.
            ValueError: If a first-attempt source or derived custody root differs.
        """
        for name in (
            "job_checksum",
            "request_checksum",
            "job_outcome_checksum",
            "resource_document_checksum",
            "production_custody_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        if self.status not in {"success", "failure"}:
            msg = "Confirmatory production receipt status must be success or failure."
            raise ValueError(msg)
        if not isinstance(self.result_reference, ResultArtifactRef):
            msg = "result_reference must be a ResultArtifactRef."
            raise TypeError(msg)
        if not isinstance(self.production_evidence, ProductionNumericalEvidence):
            msg = "production_evidence must be ProductionNumericalEvidence."
            raise TypeError(msg)
        diagnostics = tuple(
            require_checksum(value, "pilot_diagnostic_checksum") for value in self.pilot_diagnostic_checksums
        )
        object.__setattr__(self, "pilot_diagnostic_checksums", diagnostics)
        if diagnostics:
            msg = "Confirmatory primary-analysis receipts cannot contain pilot diagnostics."
            raise ValueError(msg)
        resource_payload = dict(require_mapping(self.resource_payload, "resource_payload"))
        object.__setattr__(self, "resource_payload", resource_payload)
        raw_payload = (
            None
            if self.raw_trajectory_payload is None
            else dict(require_mapping(self.raw_trajectory_payload, "raw_trajectory_payload"))
        )
        object.__setattr__(self, "raw_trajectory_payload", raw_payload)
        raw_checksum = self.raw_trajectory_document_checksum
        if raw_checksum is not None:
            raw_checksum = require_checksum(raw_checksum, "raw_trajectory_document_checksum")
            object.__setattr__(self, "raw_trajectory_document_checksum", raw_checksum)
        reference = self.result_reference
        evidence = self.production_evidence
        if evidence.artifact_kind not in {"pipeline", "operator_growth"}:
            msg = "Synthetic evidence cannot be represented as a production primary-analysis receipt."
            raise ValueError(msg)
        expected_raw = None if evidence.raw_trajectory_ref is None else evidence.raw_trajectory_ref.logical_checksum
        expected_diagnostics = tuple(ref.logical_checksum for ref in evidence.diagnostic_refs)
        resource_identity = resource_payload.get("job_checksum", resource_payload.get("request_checksum"))
        if (
            resource_identity != self.request_checksum
            or resource_payload.get("source_fingerprint_checksum") != evidence.source_fingerprint_checksum
        ):
            msg = "Confirmatory receipt resource payload differs from its request or source root."
            raise ValueError(msg)
        computed_resource_checksum = canonical_checksum({
            "schema_version": "yaqs.state_preparation.phase2.production_document.v1",
            "document_type": "runtime_resources",
            "payload": resource_payload,
        })
        computed_raw_checksum = (
            None
            if raw_payload is None
            else canonical_checksum({
                "schema_version": "yaqs.state_preparation.phase2.production_document.v1",
                "document_type": "raw_trajectory_fidelities",
                "payload": raw_payload,
            })
        )
        if raw_payload is not None:
            raw_values = raw_payload.get("trajectory_fidelities")
            if type(raw_values) is not tuple or (
                raw_payload.get("job_checksum") != self.request_checksum
                or raw_payload.get("evaluation_policy_checksum") != evidence.evaluation_policy_checksum
                or raw_payload.get("data_role") != CONFIRMATORY_DATA_ROLE
                or raw_payload.get("seed_domain") != "confirmatory_test"
                or raw_payload.get("trajectory_count") != len(raw_values)
            ):
                msg = "Confirmatory receipt raw payload differs from its request, policy, role, or count."
                raise ValueError(msg)
        if self.status == "success":
            if raw_payload is None:
                msg = "Successful confirmatory receipt requires its raw trajectory payload."
                raise ValueError(msg)
            values = cast("tuple[object, ...]", raw_payload["trajectory_fidelities"])
            fidelities = tuple(
                float(require_float(value, "trajectory_fidelity", minimum=0.0, maximum=1.0)) for value in values
            )
            recorded_fidelity = require_float(
                evidence.derived_metrics.get("noisy_fidelity"),
                "derived_metrics.noisy_fidelity",
                minimum=0.0,
                maximum=1.0,
            )
            if float(recorded_fidelity).hex() != float(production_noisy_fidelity(fidelities)).hex():
                msg = "Confirmatory receipt noisy fidelity is not the WP22E float64 raw-trajectory mean."
                raise ValueError(msg)
        if (
            reference.attempt != 1
            or evidence.attempt != 1
            or reference.job_checksum != self.request_checksum
            or evidence.job_checksum != self.request_checksum
            or reference.status != self.status
            or evidence.status != self.status
            or reference.artifact_kind != evidence.artifact_kind
            or reference.execution_source_manifest_checksum != evidence.execution_source_manifest_checksum
            or reference.source_fingerprint_checksum != evidence.source_fingerprint_checksum
            or reference.evidence_checksum != evidence.content_checksum
            or raw_checksum != expected_raw
            or raw_checksum != computed_raw_checksum
            or self.resource_document_checksum != evidence.resource_ref.logical_checksum
            or self.resource_document_checksum != computed_resource_checksum
            or diagnostics != expected_diagnostics
        ):
            msg = "Confirmatory production receipt does not close over one exact first attempt."
            raise ValueError(msg)
        expected_custody_checksum = canonical_checksum({
            "schema_version": PRODUCTION_RESULT_CUSTODY_SCHEMA_VERSION,
            "result_reference_checksum": reference.content_checksum,
            "result_evidence_checksum": evidence.content_checksum,
            "raw_trajectory_document_checksum": raw_checksum,
            "resource_document_checksum": self.resource_document_checksum,
            "pilot_diagnostic_checksums": list(diagnostics),
        })
        if self.production_custody_checksum != expected_custody_checksum:
            msg = "Confirmatory production receipt custody checksum is not mechanically derived."
            raise ValueError(msg)

    @classmethod
    def create(
        cls,
        job: TrainingJob,
        outcome: TrainingJobOutcome,
        custody: ProductionResultCustody,
        confirmation_context: ConfirmationExecutionContext,
    ) -> ConfirmatoryProductionAttemptReceipt:
        """Create a receipt only from one reopened real production attempt.

        Returns:
            The source-addressed receipt covering a success or failure attempt.

        Raises:
            TypeError: If an input has the wrong typed schema.
            ValueError: If the outcome differs from its exact reopened attempt.
        """
        if not isinstance(outcome, TrainingJobOutcome):
            msg = "outcome must be a TrainingJobOutcome."
            raise TypeError(msg)
        request = _validate_production_confirmation_custody(job, custody, confirmation_context)
        reference = custody.reference
        if (
            outcome.attempt != 1
            or outcome.job_checksum != job.content_checksum
            or outcome.status != reference.status
            or (outcome.status == "success" and outcome.result_artifact_checksum != reference.content_checksum)
            or (outcome.status == "failure" and outcome.result_artifact_checksum is not None)
        ):
            msg = "Confirmatory outcome differs from its reopened first production attempt."
            raise ValueError(msg)
        return cls(
            job_checksum=job.content_checksum,
            request_checksum=request.content_checksum,
            job_outcome_checksum=outcome.content_checksum,
            status=outcome.status,
            result_reference=reference,
            production_evidence=custody.production_evidence,
            raw_trajectory_payload=custody.raw_trajectory_payload,
            raw_trajectory_document_checksum=custody.raw_trajectory_document_checksum,
            resource_payload=custody.resource_payload,
            resource_document_checksum=custody.resource_document_checksum,
            pilot_diagnostic_checksums=tuple(item.content_checksum for item in custody.pilot_diagnostics),
            production_custody_checksum=custody.content_checksum,
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered production-attempt root."""
        return {
            "schema_version": self.schema_version,
            "job_checksum": self.job_checksum,
            "request_checksum": self.request_checksum,
            "job_outcome_checksum": self.job_outcome_checksum,
            "status": self.status,
            "result_reference": self.result_reference.to_dict(),
            "production_evidence": self.production_evidence.to_dict(),
            "raw_trajectory_payload": (
                None if self.raw_trajectory_payload is None else dict(self.raw_trajectory_payload)
            ),
            "raw_trajectory_document_checksum": self.raw_trajectory_document_checksum,
            "resource_payload": dict(self.resource_payload),
            "resource_document_checksum": self.resource_document_checksum,
            "pilot_diagnostic_checksums": list(self.pilot_diagnostic_checksums),
            "production_custody_checksum": self.production_custody_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete portable first-attempt projection."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native receipt data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed receipt JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ConfirmatoryProductionAttemptReceipt:
        """Decode and recompute one issued production-attempt receipt.

        Returns:
            The portable receipt after typed reference and evidence verification.

        Raises:
            TypeError: If diagnostic checksums are not a serialized sequence.
            ValueError: If a schema, source root, or receipt checksum differs.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_CONFIRMATORY_PRODUCTION_ATTEMPT_RECEIPT_KEYS,
            name="confirmatory production attempt receipt",
        )
        if mapping["schema_version"] != CONFIRMATORY_PRODUCTION_ATTEMPT_RECEIPT_SCHEMA_VERSION:
            msg = "Confirmatory production attempt receipt uses an unsupported schema version."
            raise ValueError(msg)
        diagnostics = mapping["pilot_diagnostic_checksums"]
        if type(diagnostics) is not tuple:
            msg = "pilot_diagnostic_checksums must be a serialized sequence."
            raise TypeError(msg)
        receipt = cls(
            job_checksum=cast("str", mapping["job_checksum"]),
            request_checksum=cast("str", mapping["request_checksum"]),
            job_outcome_checksum=cast("str", mapping["job_outcome_checksum"]),
            status=cast("Literal['success', 'failure']", mapping["status"]),
            result_reference=ResultArtifactRef.from_dict(mapping["result_reference"]),
            production_evidence=ProductionNumericalEvidence.from_dict(mapping["production_evidence"]),
            raw_trajectory_payload=cast("Mapping[str, object] | None", mapping["raw_trajectory_payload"]),
            raw_trajectory_document_checksum=cast("str | None", mapping["raw_trajectory_document_checksum"]),
            resource_payload=cast("Mapping[str, object]", mapping["resource_payload"]),
            resource_document_checksum=cast("str", mapping["resource_document_checksum"]),
            pilot_diagnostic_checksums=cast("tuple[str, ...]", diagnostics),
            production_custody_checksum=cast("str", mapping["production_custody_checksum"]),
        )
        if mapping["content_checksum"] != receipt.content_checksum:
            msg = "Confirmatory production attempt receipt checksum changed during normalization."
            raise ValueError(msg)
        return receipt

    @classmethod
    def from_json(cls, payload: str) -> ConfirmatoryProductionAttemptReceipt:
        """Decode canonical checksum-sealed receipt JSON.

        Returns:
            The portable typed production-attempt receipt.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class ConfirmatoryObservation:
    """One source-addressed target/configuration/optimization-seed outcome.

    This is deliberately a cell-level record. It contains the fixed trajectory
    budget as provenance but no trajectory index or individual trajectory
    outcome, preventing Monte Carlo trajectories from becoming pseudoreplicates.
    """

    configuration_checksum: str
    target_manifest_checksum: str
    target_spec_checksum: str
    family_id: str
    stratum_id: str
    target_instance_id: str
    qubit_count: int
    optimization_seed: int
    evaluation_seed: int
    job_id: str
    job_checksum: str
    job_outcome_checksum: str
    primary_noise_condition_checksum: str
    primary_resource_budget_checksum: str
    test_trajectory_count: int
    status: Literal["success", "failure"]
    fresh_test_noisy_fidelity: float | None
    failure_code: str | None
    evaluation_evidence_checksum: str | None
    result_schema_version: str
    result_record_checksum: str
    data_role: Literal["confirmatory"] = CONFIRMATORY_DATA_ROLE
    schema_version: str = field(default=CONFIRMATORY_OBSERVATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the source identity and status-dependent cell outcome.

        Raises:
            ValueError: If the role, identity, provenance, or outcome is invalid.
        """
        if self.data_role != CONFIRMATORY_DATA_ROLE:
            msg = "Primary-analysis observations must use the confirmatory data role."
            raise ValueError(msg)
        for name in (
            "configuration_checksum",
            "target_manifest_checksum",
            "target_spec_checksum",
            "primary_noise_condition_checksum",
            "primary_resource_budget_checksum",
            "job_checksum",
            "job_outcome_checksum",
            "result_record_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        family = require_slug(self.family_id, "family_id")
        if family not in PRIMARY_TARGET_FAMILIES:
            msg = f"family_id must be one of {PRIMARY_TARGET_FAMILIES!r}."
            raise ValueError(msg)
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "stratum_id", require_slug(self.stratum_id, "stratum_id"))
        object.__setattr__(self, "target_instance_id", require_slug(self.target_instance_id, "target_instance_id"))
        qubits = require_int(self.qubit_count, "qubit_count", minimum=2)
        if qubits != CONFIRMATORY_QUBIT_COUNT:
            msg = "The frozen primary analysis uses q=6 confirmatory targets."
            raise ValueError(msg)
        object.__setattr__(self, "qubit_count", qubits)
        object.__setattr__(
            self,
            "optimization_seed",
            require_int(self.optimization_seed, "optimization_seed", minimum=0),
        )
        object.__setattr__(
            self,
            "evaluation_seed",
            require_int(self.evaluation_seed, "evaluation_seed", minimum=0),
        )
        object.__setattr__(self, "job_id", require_slug(self.job_id, "job_id"))
        object.__setattr__(
            self,
            "test_trajectory_count",
            require_int(self.test_trajectory_count, "test_trajectory_count", minimum=2),
        )
        object.__setattr__(
            self,
            "result_schema_version",
            require_slug(self.result_schema_version, "result_schema_version"),
        )
        if self.status not in {"success", "failure"}:
            msg = "status must be 'success' or 'failure'."
            raise ValueError(msg)
        if self.status == "success":
            fidelity = require_float(
                self.fresh_test_noisy_fidelity,
                "fresh_test_noisy_fidelity",
                minimum=0.0,
                maximum=1.0,
            )
            if self.failure_code is not None:
                msg = "Successful confirmatory observations cannot carry a failure code."
                raise ValueError(msg)
            if self.evaluation_evidence_checksum is None:
                msg = "Successful confirmatory observations require typed evaluation evidence."
                raise ValueError(msg)
            object.__setattr__(
                self,
                "evaluation_evidence_checksum",
                require_checksum(self.evaluation_evidence_checksum, "evaluation_evidence_checksum"),
            )
            object.__setattr__(self, "fresh_test_noisy_fidelity", fidelity)
        else:
            if self.fresh_test_noisy_fidelity is not None:
                msg = "Failed confirmatory observations must have null fidelity."
                raise ValueError(msg)
            if self.failure_code is None:
                msg = "Failed confirmatory observations require a failure code."
                raise ValueError(msg)
            if self.evaluation_evidence_checksum is not None:
                msg = "Failed confirmatory observations cannot claim evaluation evidence."
                raise ValueError(msg)
            object.__setattr__(self, "failure_code", require_slug(self.failure_code, "failure_code"))

    @property
    def cell_identity(self) -> tuple[str, str, int]:
        """Configuration, target, and nested optimization-seed identity."""
        return self.configuration_checksum, self.target_instance_id, self.optimization_seed

    @property
    def sort_key(self) -> tuple[str, str, str, int]:
        """Canonical row ordering key."""
        return self.configuration_checksum, self.family_id, self.target_instance_id, self.optimization_seed

    @property
    def intention_to_treat_fidelity(self) -> float:
        """Fresh fidelity with the sealed zero contribution for failures."""
        return 0.0 if self.status == "failure" else cast("float", self.fresh_test_noisy_fidelity)

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete cell-level outcome."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered observation content."""
        return {
            "schema_version": self.schema_version,
            "data_role": self.data_role,
            "configuration_checksum": self.configuration_checksum,
            "target_manifest_checksum": self.target_manifest_checksum,
            "target_spec_checksum": self.target_spec_checksum,
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "target_instance_id": self.target_instance_id,
            "qubit_count": self.qubit_count,
            "optimization_seed": self.optimization_seed,
            "evaluation_seed": self.evaluation_seed,
            "job_id": self.job_id,
            "job_checksum": self.job_checksum,
            "job_outcome_checksum": self.job_outcome_checksum,
            "primary_noise_condition_checksum": self.primary_noise_condition_checksum,
            "primary_resource_budget_checksum": self.primary_resource_budget_checksum,
            "test_trajectory_count": self.test_trajectory_count,
            "status": self.status,
            "fresh_test_noisy_fidelity": self.fresh_test_noisy_fidelity,
            "failure_code": self.failure_code,
            "evaluation_evidence_checksum": self.evaluation_evidence_checksum,
            "result_schema_version": self.result_schema_version,
            "result_record_checksum": self.result_record_checksum,
        }

    def to_dict(self) -> dict[str, object]:
        """Return sealed JSON-native observation data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ConfirmatoryObservation:
        """Decode and checksum-verify one confirmatory observation.

        Args:
            data: Exact sealed observation mapping.

        Returns:
            The validated cell-level outcome.

        Raises:
            ValueError: If the schema, checksum, or normalized outcome is invalid.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_OBSERVATION_KEYS, name="confirmatory observation")
        if mapping["schema_version"] != CONFIRMATORY_OBSERVATION_SCHEMA_VERSION:
            msg = f"schema_version must be {CONFIRMATORY_OBSERVATION_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        observation = cls(
            configuration_checksum=cast("str", mapping["configuration_checksum"]),
            target_manifest_checksum=cast("str", mapping["target_manifest_checksum"]),
            target_spec_checksum=cast("str", mapping["target_spec_checksum"]),
            family_id=cast("str", mapping["family_id"]),
            stratum_id=cast("str", mapping["stratum_id"]),
            target_instance_id=cast("str", mapping["target_instance_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            optimization_seed=cast("int", mapping["optimization_seed"]),
            evaluation_seed=cast("int", mapping["evaluation_seed"]),
            job_id=cast("str", mapping["job_id"]),
            job_checksum=cast("str", mapping["job_checksum"]),
            job_outcome_checksum=cast("str", mapping["job_outcome_checksum"]),
            primary_noise_condition_checksum=cast("str", mapping["primary_noise_condition_checksum"]),
            primary_resource_budget_checksum=cast("str", mapping["primary_resource_budget_checksum"]),
            test_trajectory_count=cast("int", mapping["test_trajectory_count"]),
            status=cast("Literal['success', 'failure']", mapping["status"]),
            fresh_test_noisy_fidelity=cast("float | None", mapping["fresh_test_noisy_fidelity"]),
            failure_code=cast("str | None", mapping["failure_code"]),
            evaluation_evidence_checksum=cast("str | None", mapping["evaluation_evidence_checksum"]),
            result_schema_version=cast("str", mapping["result_schema_version"]),
            result_record_checksum=cast("str", mapping["result_record_checksum"]),
            data_role=cast("Literal['confirmatory']", mapping["data_role"]),
        )
        if observation.content_checksum != mapping["content_checksum"]:
            msg = "Confirmatory observation checksum changed during normalization."
            raise ValueError(msg)
        return observation

    @classmethod
    def from_json(cls, payload: str) -> ConfirmatoryObservation:
        """Decode an observation from canonical JSON text.

        Args:
            payload: Canonical sealed JSON.

        Returns:
            The validated observation.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _sealed_configuration_methods(seal: FinalConfirmationSeal) -> dict[str, str]:
    """Return the exact de-duplicated configuration-to-method universe.

    Raises:
        ValueError: If a comparator duplicates a configuration.
    """
    methods = {seal.promoted_configuration_checksum: seal.promoted_method_id}
    for comparator in seal.comparators:
        if comparator.configuration_checksum in methods:
            msg = "Final seal repeats a confirmatory configuration."
            raise ValueError(msg)
        methods[comparator.configuration_checksum] = comparator.method_id
    return methods


def _contrast_definitions(preregistration: InitialPreregistration) -> dict[str, Mapping[str, object]]:
    """Return the trusted preregistered hypothesis and margin definitions."""
    raw = cast(
        "Sequence[Mapping[str, object]]",
        preregistration.multiplicity_policy["contrast_definitions"],
    )
    return {cast("str", item["contrast_id"]): item for item in raw}


def _validate_seal_against_preregistration(
    seal: FinalConfirmationSeal,
    preregistration: InitialPreregistration,
) -> None:
    """Reject a final seal that changes any primary-analysis choice.

    Raises:
        ValueError: If a protocol, endpoint, contrast, or design field is unsealed.
    """
    if seal.preregistration_checksum != preregistration.content_checksum:
        msg = "Final seal does not reference the trusted preregistration."
        raise ValueError(msg)
    if seal.analysis_template_checksum != preregistration.analysis_template_checksum:
        msg = "Final seal changes the frozen primary-analysis template."
        raise ValueError(msg)
    if seal.failure_policy_checksum != preregistration.failure_policy_checksum:
        msg = "Final seal changes the frozen intention-to-treat failure policy."
        raise ValueError(msg)
    if canonical_checksum(seal.primary_noise_condition) != canonical_checksum(preregistration.primary_noise_condition):
        msg = "Final seal changes the preregistered primary noise condition."
        raise ValueError(msg)
    resource = preregistration.primary_resource_constraint
    if (
        seal.primary_resource_budget["metric"] != resource["metric"]
        or seal.primary_resource_budget["cap_per_chain_edge"] != resource["cap_per_chain_edge"]
    ):
        msg = "Final seal changes the preregistered primary native-resource condition."
        raise ValueError(msg)

    expected_contrasts = {"noisy_vs_noiseless"}
    if seal.promoted_method_id != "layerwise_bmpd_crn_v2":
        expected_contrasts.add("promoted_vs_layerwise_v2_if_distinct")
    actual_contrasts = {binding.contrast_id for binding in seal.primary_contrasts}
    if actual_contrasts != expected_contrasts:
        msg = "Final seal primary contrasts do not match the applicable preregistered set."
        raise ValueError(msg)
    definitions = _contrast_definitions(preregistration)
    if any(contrast_id not in definitions for contrast_id in actual_contrasts):
        msg = "Final seal contains an unregistered primary contrast."
        raise ValueError(msg)

    comparator_by_role = {comparator.role: comparator for comparator in seal.comparators}
    noiseless = comparator_by_role.get("matched_noiseless_control")
    if noiseless is None or noiseless.method_id != "layerwise_bmpd_noiseless":
        msg = "Final seal omits the exact matched noiseless control."
        raise ValueError(msg)
    if seal.promoted_method_id == "layerwise_bmpd_crn_v2":
        v2_checksum = seal.promoted_configuration_checksum
        if "layerwise_v2_reference" in comparator_by_role:
            msg = "A promoted v2 configuration must not be duplicated as a comparator."
            raise ValueError(msg)
    else:
        v2_reference = comparator_by_role.get("layerwise_v2_reference")
        if v2_reference is None or v2_reference.method_id != "layerwise_bmpd_crn_v2":
            msg = "Final seal omits the exact layerwise v2 reference configuration."
            raise ValueError(msg)
        v2_checksum = v2_reference.configuration_checksum
        if (
            v2_reference.matched_to_configuration_checksum != noiseless.configuration_checksum
            or v2_reference.matching_projection_checksum != noiseless.matching_projection_checksum
        ):
            msg = "The layerwise v2 reference is not paired with the matched noiseless control."
            raise ValueError(msg)
    if noiseless.matched_to_configuration_checksum != v2_checksum or noiseless.matching_projection_checksum is None:
        msg = "The noiseless control is not bound to the exact layerwise v2 configuration."
        raise ValueError(msg)
    binding_by_id = {binding.contrast_id: binding for binding in seal.primary_contrasts}
    noisy_binding = binding_by_id["noisy_vs_noiseless"]
    if (
        noisy_binding.treatment_configuration_checksum != v2_checksum
        or noisy_binding.control_configuration_checksum != noiseless.configuration_checksum
        or noisy_binding.paired_block_policy_checksum != preregistration.paired_block_policy_checksum
        or noisy_binding.matching_projection_checksum != noiseless.matching_projection_checksum
    ):
        msg = "The noisy-versus-noiseless contrast changes its sealed matched configurations."
        raise ValueError(msg)
    promoted_binding = binding_by_id.get("promoted_vs_layerwise_v2_if_distinct")
    if promoted_binding is not None and (
        promoted_binding.treatment_configuration_checksum != seal.promoted_configuration_checksum
        or promoted_binding.control_configuration_checksum != v2_checksum
        or promoted_binding.paired_block_policy_checksum != preregistration.paired_block_policy_checksum
        or promoted_binding.matching_projection_checksum is not None
    ):
        msg = "The promoted-versus-v2 contrast changes its sealed configurations."
        raise ValueError(msg)
    sample_policy = preregistration.sample_size_policy
    target_counts = tuple(cast("int", seal.target_count_by_family[family]) for family in PRIMARY_TARGET_FAMILIES)
    if len(set(target_counts)) != 1:
        msg = "Primary analysis requires equal target counts across families."
        raise ValueError(msg)
    targets = target_counts[0]
    minimum = cast("int", sample_policy["minimum_targets_per_family"])
    maximum = cast("int", sample_policy["maximum_targets_per_family"])
    increment = cast("int", sample_policy["target_count_increment"])
    if not minimum <= targets <= maximum or targets % increment != 0:
        msg = "Final seal target counts violate the frozen sample-size bounds."
        raise ValueError(msg)
    if seal.optimization_seed_count not in cast("Sequence[int]", sample_policy["allowed_optimization_seed_counts"]):
        msg = "Final seal optimization-seed count is not preregistered."
        raise ValueError(msg)
    trajectory_minimum = cast("int", sample_policy["trajectory_count_min"])
    trajectory_maximum = cast("int", sample_policy["trajectory_count_max"])
    trajectories = seal.fixed_test_trajectory_count
    if not trajectory_minimum <= trajectories <= trajectory_maximum or trajectories & (trajectories - 1) != 0:
        msg = "Final seal trajectory count violates the frozen fixed power-of-two budget."
        raise ValueError(msg)


def _validate_observation_universe(
    seal: FinalConfirmationSeal,
    observations: Sequence[ConfirmatoryObservation],
) -> tuple[ConfirmatoryObservation, ...]:
    """Validate the exact configuration-by-target-by-seed Cartesian universe.

    Returns:
        Observations in canonical deterministic order.

    Raises:
        TypeError: If an element is not a confirmatory observation.
        ValueError: If any cell is duplicated, missing, extra, or unsealed.
    """
    raw = tuple(observations)
    if not raw or not all(isinstance(item, ConfirmatoryObservation) for item in raw):
        msg = "observations must contain ConfirmatoryObservation values."
        raise TypeError(msg)
    ordered = tuple(sorted(raw, key=lambda item: item.sort_key))
    identities = tuple(item.cell_identity for item in ordered)
    if len(identities) != len(set(identities)):
        msg = "Confirmatory evidence must contain one cell-level outcome per configuration/target/seed."
        raise ValueError(msg)
    result_checksums = tuple(item.result_record_checksum for item in ordered)
    if len(result_checksums) != len(set(result_checksums)):
        msg = "Every confirmatory cell must reference a unique result record."
        raise ValueError(msg)

    methods = _sealed_configuration_methods(seal)
    expected_configurations = set(methods)
    if {item.configuration_checksum for item in ordered} != expected_configurations:
        msg = "Confirmatory evidence configuration universe differs from the final seal."
        raise ValueError(msg)
    noise_checksum = canonical_checksum(seal.primary_noise_condition)
    resource_checksum = canonical_checksum(seal.primary_resource_budget)
    for item in ordered:
        if item.target_manifest_checksum != seal.confirmatory_target_manifest_checksum:
            msg = "Confirmatory observation uses a target manifest not named by the final seal."
            raise ValueError(msg)
        if item.primary_noise_condition_checksum != noise_checksum:
            msg = "Confirmatory observation uses an unsealed primary noise condition."
            raise ValueError(msg)
        if item.primary_resource_budget_checksum != resource_checksum:
            msg = "Confirmatory observation uses an unsealed primary resource budget."
            raise ValueError(msg)
        if item.test_trajectory_count != seal.fixed_test_trajectory_count:
            msg = "Confirmatory observation changes the fixed test-trajectory count."
            raise ValueError(msg)

    cells_by_configuration: dict[str, set[tuple[str, int]]] = defaultdict(set)
    target_metadata: dict[str, str] = {}
    target_seeds: dict[str, set[int]] = defaultdict(set)
    for item in ordered:
        previous_family = target_metadata.setdefault(item.target_instance_id, item.family_id)
        if previous_family != item.family_id:
            msg = f"Target {item.target_instance_id!r} has inconsistent family metadata."
            raise ValueError(msg)
        cells_by_configuration[item.configuration_checksum].add((item.target_instance_id, item.optimization_seed))
        target_seeds[item.target_instance_id].add(item.optimization_seed)
    reference_cells = cells_by_configuration[seal.promoted_configuration_checksum]
    if any(cells != reference_cells for cells in cells_by_configuration.values()):
        msg = "Every sealed configuration must cover the identical target/optimization-seed universe."
        raise ValueError(msg)
    if {len(seeds) for seeds in target_seeds.values()} != {seal.optimization_seed_count}:
        msg = "Every confirmatory target must use the sealed number of paired optimization-seed replicates."
        raise ValueError(msg)
    targets_by_family: dict[str, set[str]] = defaultdict(set)
    for target_id, family_id in target_metadata.items():
        targets_by_family[family_id].add(target_id)
    if set(targets_by_family) != set(PRIMARY_TARGET_FAMILIES):
        msg = "Confirmatory evidence must cover every primary target family."
        raise ValueError(msg)
    for family_id in PRIMARY_TARGET_FAMILIES:
        if len(targets_by_family[family_id]) != cast("int", seal.target_count_by_family[family_id]):
            msg = f"Confirmatory target count for family {family_id!r} differs from the final seal."
            raise ValueError(msg)

    expected_cells = len(methods) * sum(cast("int", value) for value in seal.target_count_by_family.values())
    expected_cells *= seal.optimization_seed_count
    if len(ordered) != expected_cells:
        msg = "Confirmatory evidence is missing or contains extra Cartesian cells."
        raise ValueError(msg)
    contrast_configurations = {
        checksum
        for binding in seal.primary_contrasts
        for checksum in (binding.treatment_configuration_checksum, binding.control_configuration_checksum)
    }
    if not contrast_configurations <= expected_configurations:
        msg = "A primary contrast references a configuration outside the final seal."
        raise ValueError(msg)
    return ordered


def _validate_confirmatory_custody(
    seal: FinalConfirmationSeal,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
    target_manifest: TargetPopulationManifest,
    run_plan: TrainingRunPlan,
    job_outcomes: Sequence[TrainingJobOutcome],
    result_artifacts: Sequence[ConfirmatoryResultArtifact],
) -> tuple[tuple[TrainingJobOutcome, ...], tuple[ConfirmatoryResultArtifact, ...], tuple[ConfirmatoryObservation, ...]]:
    """Authenticate custody and mechanically construct the complete row universe.

    Returns:
        Canonical job outcomes, successful results, and derived observations.

    Raises:
        TypeError: If a custody artifact has the wrong typed schema.
        ValueError: If any target, job, outcome, or result link differs.
    """
    if not isinstance(target_manifest, TargetPopulationManifest):
        msg = "target_manifest must be the revealed TargetPopulationManifest."
        raise TypeError(msg)
    if not isinstance(run_plan, TrainingRunPlan):
        msg = "run_plan must be the seal-generated TrainingRunPlan."
        raise TypeError(msg)
    context = build_confirm_execution_context(
        seal,
        target_manifest,
        configuration_execution_manifest,
    )
    manifest_checksum = context.target_manifest_checksum
    seal_checksum = context.final_seal_checksum
    if target_manifest.data_role != CONFIRMATORY_DATA_ROLE:
        msg = "Primary analysis requires the revealed confirmatory target manifest."
        raise ValueError(msg)
    if manifest_checksum != seal.confirmatory_target_manifest_checksum:
        msg = "Revealed target manifest does not reproduce the final-seal commitment."
        raise ValueError(msg)
    if (
        run_plan.plan_id != "wp22_paper_confirm_v1"
        or run_plan.preset != "paper-confirm"
        or run_plan.preregistration_checksum != seal.preregistration_checksum
        or run_plan.target_manifest_checksums != (manifest_checksum,)
        or run_plan.screening_manifest_checksum is not None
        or run_plan.final_confirmation_seal_checksum != seal_checksum
        or run_plan.execution_source_checksum != seal.execution_source_checksum
    ):
        msg = "Training run plan roots do not reproduce the final seal and revealed target manifest."
        raise ValueError(msg)

    configurations = dict(context.methods_by_configuration)
    targets_by_id = dict(context.targets_by_id)
    target_checksums = {target_id: spec.content_checksum for target_id, spec in targets_by_id.items()}
    expected_cells = {
        (configuration_checksum, target_id, seed_index)
        for configuration_checksum in configurations
        for target_id in targets_by_id
        for seed_index in range(seal.optimization_seed_count)
    }
    jobs_by_cell: dict[tuple[str, str, int], tuple[TrainingJob, str]] = {}
    for job in run_plan.jobs:
        request = job.confirm_execution_request
        if not isinstance(request, ConfirmExecutionRequest):
            msg = "Every paper-confirm job requires its typed seal-complete execution request."
            raise TypeError(msg)
        spec = targets_by_id.get(job.target_instance_id)
        if spec is None:
            msg = "Confirmatory plan contains a target absent from the revealed manifest."
            raise ValueError(msg)
        spec_checksum = target_checksums[spec.target_instance_id]
        seed_index = request.optimization_seed_index
        method_id = configurations.get(job.candidate_configuration_checksum)
        if method_id is None:
            msg = "Confirmatory plan contains a configuration absent from the final seal."
            raise ValueError(msg)
        execution = context.execution_by_configuration[job.candidate_configuration_checksum]
        validate_confirm_execution_request(request, context)
        expected_job_id = "wp22_job_" + canonical_checksum({
            "seal": seal_checksum,
            "configuration": job.candidate_configuration_checksum,
            "target": spec_checksum,
            "optimization_seed": request.optimization_seed,
            "evaluation_seed": request.evaluation_seed,
        }).removeprefix("sha256:")
        expected_path = f"roles/confirmatory/{spec.family_id}/{spec.target_instance_id}/{expected_job_id}"
        if (
            job.job_id != expected_job_id
            or job.preset != "paper-confirm"
            or job.method_id != method_id
            or job.implementation_kind != "sealed_configuration"
            or job.implementation_checksum != execution.implementation_checksum
            or job.strategy_schedule_checksum != execution.strategy_schedule_checksum
            or job.strategy_schedule is not None
            or job.target_manifest_checksum != manifest_checksum
            or job.data_role != CONFIRMATORY_DATA_ROLE
            or job.output_path != expected_path
        ):
            msg = "Training job differs from its exact seal-generated confirmatory cell."
            raise ValueError(msg)
        key = (job.candidate_configuration_checksum, spec.target_instance_id, seed_index)
        if key in jobs_by_cell:
            msg = "Confirmatory plan duplicates one configuration/target/seed-index cell."
            raise ValueError(msg)
        jobs_by_cell[key] = job, job.content_checksum
    if set(jobs_by_cell) != expected_cells:
        msg = "Confirmatory plan does not cover the exact seal-generated Cartesian job universe."
        raise ValueError(msg)

    raw_outcomes = tuple(job_outcomes)
    if not raw_outcomes or not all(isinstance(item, TrainingJobOutcome) for item in raw_outcomes):
        msg = "job_outcomes must contain typed TrainingJobOutcome artifacts."
        raise TypeError(msg)
    outcomes_by_job: dict[str, TrainingJobOutcome] = {}
    for outcome in raw_outcomes:
        if outcome.attempt != 1:
            msg = "Primary analysis accepts only authoritative first terminal confirmatory attempts."
            raise ValueError(msg)
        if outcome.job_checksum in outcomes_by_job:
            msg = "Confirmatory custody contains duplicate outcomes for one exact job."
            raise ValueError(msg)
        outcomes_by_job[outcome.job_checksum] = outcome
    expected_job_checksums = {job_checksum for _job, job_checksum in jobs_by_cell.values()}
    if set(outcomes_by_job) != expected_job_checksums:
        msg = "Confirmatory outcomes do not cover the exact seal-generated job universe."
        raise ValueError(msg)

    raw_results = tuple(result_artifacts)
    if not all(isinstance(item, ConfirmatoryResultArtifact) for item in raw_results):
        msg = "result_artifacts must contain typed ConfirmatoryResultArtifact values."
        raise TypeError(msg)
    results_by_job: dict[str, tuple[ConfirmatoryResultArtifact, str]] = {}
    for result in raw_results:
        if result.job_checksum in results_by_job:
            msg = "Confirmatory custody contains duplicate successful results for one job."
            raise ValueError(msg)
        results_by_job[result.job_checksum] = result, result.content_checksum
    if len({result.source_evaluation.content_checksum for result in raw_results}) != len(raw_results):
        msg = "Every successful confirmatory job requires a distinct typed raw evaluation source."
        raise ValueError(msg)
    if len({result.source_result_reference_checksum for result in raw_results}) != len(raw_results):
        msg = "Every successful confirmatory job requires a distinct production result reference."
        raise ValueError(msg)
    if len({result.source_production_evidence_checksum for result in raw_results}) != len(raw_results):
        msg = "Every successful confirmatory job requires a distinct production evidence root."
        raise ValueError(msg)
    successful_jobs = {job_checksum for job_checksum, outcome in outcomes_by_job.items() if outcome.status == "success"}
    if set(results_by_job) != successful_jobs:
        msg = "Typed confirmatory results must cover exactly the successful job outcomes."
        raise ValueError(msg)

    noise_checksum = canonical_checksum(seal.primary_noise_condition)
    resource_checksum = canonical_checksum(seal.primary_resource_budget)
    ordered_outcomes: list[TrainingJobOutcome] = []
    ordered_results: list[ConfirmatoryResultArtifact] = []
    observations: list[ConfirmatoryObservation] = []
    for job in run_plan.jobs:
        request = cast("ConfirmExecutionRequest", job.confirm_execution_request)
        spec = targets_by_id[job.target_instance_id]
        job_checksum = jobs_by_cell[
            job.candidate_configuration_checksum,
            job.target_instance_id,
            request.optimization_seed_index,
        ][1]
        outcome = outcomes_by_job[job_checksum]
        outcome_checksum = outcome.content_checksum
        if outcome.status == "success":
            result, result_checksum = results_by_job[job_checksum]
            if (
                outcome.result_artifact_checksum != result.source_result_reference_checksum
                or result.evaluation_seed != job.evaluation_seed
                or result.evaluation_policy_checksum != confirmatory_evaluation_policy_checksum(request)
                or result.test_trajectory_count != seal.fixed_test_trajectory_count
                or result.primary_noise_condition_checksum != noise_checksum
                or result.primary_resource_budget_checksum != resource_checksum
            ):
                msg = "Successful result artifact differs from its job, outcome, or sealed evaluation policy."
                raise ValueError(msg)
            fidelity = result.fresh_test_noisy_fidelity
            failure_code = None
            evaluation_evidence_checksum = result.evaluation_evidence_checksum
            result_schema_version = result.schema_version
            result_record_checksum = result_checksum
            ordered_results.append(result)
        else:
            fidelity = None
            failure_code = outcome.exception_type
            evaluation_evidence_checksum = None
            result_schema_version = TRAINING_JOB_OUTCOME_SCHEMA_VERSION
            result_record_checksum = outcome_checksum
        observations.append(
            ConfirmatoryObservation(
                configuration_checksum=job.candidate_configuration_checksum,
                target_manifest_checksum=manifest_checksum,
                target_spec_checksum=target_checksums[spec.target_instance_id],
                family_id=spec.family_id,
                stratum_id=spec.stratum_id,
                target_instance_id=spec.target_instance_id,
                qubit_count=spec.qubit_count,
                optimization_seed=job.optimization_seed,
                evaluation_seed=job.evaluation_seed,
                job_id=job.job_id,
                job_checksum=job_checksum,
                job_outcome_checksum=outcome_checksum,
                primary_noise_condition_checksum=noise_checksum,
                primary_resource_budget_checksum=resource_checksum,
                test_trajectory_count=seal.fixed_test_trajectory_count,
                status=outcome.status,
                fresh_test_noisy_fidelity=fidelity,
                failure_code=failure_code,
                evaluation_evidence_checksum=evaluation_evidence_checksum,
                result_schema_version=result_schema_version,
                result_record_checksum=result_record_checksum,
            )
        )
        ordered_outcomes.append(outcome)
    return tuple(ordered_outcomes), tuple(ordered_results), tuple(observations)


def _clustered_equal_family_summary(
    values: Mapping[tuple[str, str, int], float],
) -> tuple[float, float, float, float, dict[str, object]]:
    """Aggregate nested seeds within target clusters and weight families equally.

    Returns:
        Estimate, cluster-aware standard error, confidence limits, and family summaries.

    Raises:
        ValueError: If a family has fewer than two target clusters.
    """
    by_family_target: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (family_id, target_id, _optimization_seed), value in values.items():
        by_family_target[family_id][target_id].append(value)
    family_estimates: dict[str, object] = {}
    estimator_variance = 0.0
    family_means: list[float] = []
    family_weight = 1.0 / len(PRIMARY_TARGET_FAMILIES)
    for family_id in PRIMARY_TARGET_FAMILIES:
        target_means = [
            math.fsum(seed_values) / len(seed_values)
            for _target_id, seed_values in sorted(by_family_target[family_id].items())
        ]
        if len(target_means) < 2:
            msg = "Cluster-aware analysis requires at least two targets per family."
            raise ValueError(msg)
        family_mean = math.fsum(target_means) / len(target_means)
        target_variance = float(statistics.variance(target_means))
        family_means.append(family_mean)
        estimator_variance += family_weight**2 * target_variance / len(target_means)
        family_estimates[family_id] = {
            "target_cluster_count": len(target_means),
            "estimate": family_mean,
            "target_cluster_variance": target_variance,
        }
    estimate = math.fsum(family_means) / len(family_means)
    standard_error = math.sqrt(max(0.0, estimator_variance))
    lower = estimate - _CI_Z * standard_error
    upper = estimate + _CI_Z * standard_error
    return estimate, standard_error, lower, upper, family_estimates


def _one_sided_p_value(estimate: float, standard_error: float, margin: float) -> tuple[float | None, float]:
    """Return the z statistic and one-sided p-value against a lower margin."""
    distance = estimate - margin
    if standard_error <= 0.0:
        if distance > 0.0:
            return None, 0.0
        if distance < 0.0:
            return None, 1.0
        return None, 0.5
    statistic = distance / standard_error
    return statistic, 1.0 - NormalDist().cdf(statistic)


def _raw_contrast_results(
    seal: FinalConfirmationSeal,
    preregistration: InitialPreregistration,
    observations: Sequence[ConfirmatoryObservation],
) -> list[dict[str, object]]:
    """Compute unadjusted sealed paired-contrast estimates and p-values.

    Returns:
        Contrast dictionaries before multiplicity adjustment.
    """
    by_configuration_cell = {
        (item.configuration_checksum, item.family_id, item.target_instance_id, item.optimization_seed): item
        for item in observations
    }
    definitions = _contrast_definitions(preregistration)
    results: list[dict[str, object]] = []
    for binding in sorted(seal.primary_contrasts, key=lambda item: item.contrast_id):
        definition = definitions[binding.contrast_id]
        paired_values: dict[tuple[str, str, int], float] = {}
        for item in observations:
            if item.configuration_checksum != binding.treatment_configuration_checksum:
                continue
            control = by_configuration_cell[
                binding.control_configuration_checksum,
                item.family_id,
                item.target_instance_id,
                item.optimization_seed,
            ]
            paired_values[item.family_id, item.target_instance_id, item.optimization_seed] = (
                item.intention_to_treat_fidelity - control.intention_to_treat_fidelity
            )
        estimate, standard_error, lower, upper, family_estimates = _clustered_equal_family_summary(paired_values)
        margin = cast("float", definition["margin"])
        statistic, p_value = _one_sided_p_value(estimate, standard_error, margin)
        results.append({
            "contrast_id": binding.contrast_id,
            "hypothesis": definition["hypothesis"],
            "margin": margin,
            "treatment_configuration_checksum": binding.treatment_configuration_checksum,
            "control_configuration_checksum": binding.control_configuration_checksum,
            "estimate": estimate,
            "standard_error": standard_error,
            "confidence_level": CONFIDENCE_LEVEL,
            "confidence_interval_lower": lower,
            "confidence_interval_upper": upper,
            "z_statistic": statistic,
            "unadjusted_p_value": p_value,
            "family_estimates": family_estimates,
        })
    return results


def _apply_holm(results: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Apply deterministic Holm FWER control and return contrast-id order.

    Returns:
        Holm-decorated contrast results in stable contrast-identifier order.
    """
    ranked = sorted(results, key=lambda item: (cast("float", item["unadjusted_p_value"]), item["contrast_id"]))
    count = len(ranked)
    adjusted_floor = 0.0
    continue_rejecting = True
    decorated: dict[str, dict[str, object]] = {}
    for rank, result in enumerate(ranked, start=1):
        p_value = cast("float", result["unadjusted_p_value"])
        threshold = FAMILYWISE_ALPHA / (count - rank + 1)
        adjusted_floor = max(adjusted_floor, min(1.0, (count - rank + 1) * p_value))
        rejected = continue_rejecting and p_value <= threshold
        if not rejected:
            continue_rejecting = False
        hypothesis = cast("str", result["hypothesis"])
        claim = (
            "superior"
            if rejected and hypothesis == "superiority"
            else "noninferior"
            if rejected and hypothesis == "noninferiority"
            else "not_established"
        )
        decorated[cast("str", result["contrast_id"])] = {
            **dict(result),
            "holm_rank": rank,
            "holm_threshold": threshold,
            "holm_adjusted_p_value": adjusted_floor,
            "reject_null": rejected,
            "claim": claim,
        }
    return tuple(decorated[contrast_id] for contrast_id in sorted(decorated))


def _wilson_score_interval(proportion: float, sample_size: int) -> tuple[float, float]:
    """Return a finite-sample Wilson interval for a bounded proportion.

    The failure endpoint's independent sampling unit is the target cluster, so
    ``sample_size`` is the total target-cluster count rather than the larger
    nested optimization-seed count. This deliberately supplies a conservative
    nonzero uncertainty floor when every observed cell succeeds or fails.

    Returns:
        Lower and upper Wilson score limits.

    Raises:
        ValueError: If no target cluster is available.
    """
    if sample_size <= 0:
        msg = "Wilson interval requires at least one target cluster."
        raise ValueError(msg)
    z_squared = _CI_Z**2
    denominator = 1.0 + z_squared / sample_size
    center = (proportion + z_squared / (2.0 * sample_size)) / denominator
    half_width = (
        _CI_Z
        * math.sqrt(proportion * (1.0 - proportion) / sample_size + z_squared / (4.0 * sample_size**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _failure_rate_results(
    seal: FinalConfirmationSeal,
    observations: Sequence[ConfirmatoryObservation],
) -> tuple[dict[str, object], ...]:
    """Compute family-equal clustered failure-rate endpoints by configuration.

    Returns:
        Configuration results in stable checksum order.
    """
    methods = _sealed_configuration_methods(seal)
    results: list[dict[str, object]] = []
    for configuration_checksum in sorted(methods):
        values = {
            (item.family_id, item.target_instance_id, item.optimization_seed): float(item.status == "failure")
            for item in observations
            if item.configuration_checksum == configuration_checksum
        }
        estimate, standard_error, lower, upper, family_estimates = _clustered_equal_family_summary(values)
        failures = sum(
            item.status == "failure" for item in observations if item.configuration_checksum == configuration_checksum
        )
        target_cluster_count = sum(
            cast("int", cast("Mapping[str, object]", family)["target_cluster_count"])
            for family in family_estimates.values()
        )
        wilson_lower, wilson_upper = _wilson_score_interval(estimate, target_cluster_count)
        results.append({
            "configuration_checksum": configuration_checksum,
            "method_id": methods[configuration_checksum],
            "failure_count": failures,
            "cell_count": len(values),
            "failure_rate": estimate,
            "standard_error": standard_error,
            "confidence_level": CONFIDENCE_LEVEL,
            "confidence_interval_method": _ANALYSIS_POLICY["failure_interval_method"],
            "effective_target_cluster_count": target_cluster_count,
            "confidence_interval_lower": max(0.0, min(lower, wilson_lower)),
            "confidence_interval_upper": min(1.0, max(upper, wilson_upper)),
            "family_estimates": family_estimates,
        })
    return tuple(results)


@dataclass(frozen=True, slots=True)
class _ConfirmatoryAnalysisStatistics:
    """Private non-production view used to test the frozen statistical seam."""

    final_seal: FinalConfirmationSeal
    target_manifest_checksum: str
    run_plan_checksum: str
    job_outcomes: tuple[TrainingJobOutcome, ...]
    confirmatory_results: tuple[ConfirmatoryResultArtifact, ...]
    observations: tuple[ConfirmatoryObservation, ...]

    @property
    def analysis_policy(self) -> dict[str, object]:
        """Frozen primary endpoint, clustering, weighting, and multiplicity policy."""
        preregistration = load_initial_preregistration()
        return {
            **_ANALYSIS_POLICY,
            "preregistration_checksum": preregistration.content_checksum,
            "analysis_template_checksum": preregistration.analysis_template_checksum,
            "failure_policy_checksum": preregistration.failure_policy_checksum,
            "contrast_set_checksum": preregistration.contrast_set_checksum,
        }

    @property
    def contrast_results(self) -> tuple[dict[str, object], ...]:
        """Paired estimates, clustered uncertainty, and Holm-controlled claims."""
        preregistration = load_initial_preregistration()
        return _apply_holm(_raw_contrast_results(self.final_seal, preregistration, self.observations))

    @property
    def failure_rate_results(self) -> tuple[dict[str, object], ...]:
        """Configuration-specific family-equal failure-rate endpoints."""
        return _failure_rate_results(self.final_seal, self.observations)

    @property
    def content_checksum(self) -> str:
        """Deterministic checksum explicitly tagged as non-production statistics."""
        return canonical_checksum({
            "artifact_kind": "nonproduction_detached_confirmatory_statistics",
            "final_seal_checksum": self.final_seal.content_checksum,
            "target_manifest_checksum": self.target_manifest_checksum,
            "run_plan_checksum": self.run_plan_checksum,
            "job_outcome_checksums": [item.content_checksum for item in self.job_outcomes],
            "confirmatory_result_checksums": [item.content_checksum for item in self.confirmatory_results],
            "observations": [item.to_dict() for item in self.observations],
            "contrast_results": list(self.contrast_results),
            "failure_rate_results": list(self.failure_rate_results),
        })


_PRODUCTION_ANALYSIS_CONSTRUCTION_AUTHORITY = object()
_PRIMARY_ANALYSIS_REOPEN_AUTHORITY = object()


@dataclass(frozen=True, slots=True, init=False)
class PrimaryAnalysisResult:
    """Raw-row-backed, checksum-sealed frozen confirmatory analysis result."""

    final_seal: FinalConfirmationSeal
    target_manifest_checksum: str
    run_plan_checksum: str
    job_outcomes: tuple[TrainingJobOutcome, ...]
    production_attempt_receipts: tuple[ConfirmatoryProductionAttemptReceipt, ...]
    confirmatory_results: tuple[ConfirmatoryResultArtifact, ...]
    observations: tuple[ConfirmatoryObservation, ...]
    schema_version: str = field(default=PRIMARY_ANALYSIS_RESULT_SCHEMA_VERSION, init=False)

    def __init__(
        self,
        *,
        final_seal: FinalConfirmationSeal,
        target_manifest_checksum: str,
        run_plan_checksum: str,
        job_outcomes: tuple[TrainingJobOutcome, ...],
        production_attempt_receipts: tuple[ConfirmatoryProductionAttemptReceipt, ...],
        confirmatory_results: tuple[ConfirmatoryResultArtifact, ...],
        observations: tuple[ConfirmatoryObservation, ...],
        _construction_authority: object,
    ) -> None:
        """Construct only from reopened production custody or portable decoding.

        Raises:
            TypeError: If a caller bypasses the authorized production or reopen paths.
        """
        if (
            _construction_authority is not _PRODUCTION_ANALYSIS_CONSTRUCTION_AUTHORITY
            and _construction_authority is not _PRIMARY_ANALYSIS_REOPEN_AUTHORITY
        ):
            msg = "PrimaryAnalysisResult creation requires verified production custody."
            raise TypeError(msg)
        object.__setattr__(self, "final_seal", final_seal)
        object.__setattr__(self, "target_manifest_checksum", target_manifest_checksum)
        object.__setattr__(self, "run_plan_checksum", run_plan_checksum)
        object.__setattr__(self, "job_outcomes", job_outcomes)
        object.__setattr__(self, "production_attempt_receipts", production_attempt_receipts)
        object.__setattr__(self, "confirmatory_results", confirmatory_results)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "schema_version", PRIMARY_ANALYSIS_RESULT_SCHEMA_VERSION)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate the seal and exact confirmatory Cartesian universe.

        Raises:
            TypeError: If the seal has the wrong record type.
            ValueError: If a checksum or committed evidence universe differs.
        """
        if not isinstance(self.final_seal, FinalConfirmationSeal):
            msg = "final_seal must be a FinalConfirmationSeal."
            raise TypeError(msg)
        preregistration = load_initial_preregistration()
        _validate_seal_against_preregistration(self.final_seal, preregistration)
        target_manifest_checksum = require_checksum(self.target_manifest_checksum, "target_manifest_checksum")
        if target_manifest_checksum != self.final_seal.confirmatory_target_manifest_checksum:
            msg = "Primary analysis target-manifest checksum differs from the final seal."
            raise ValueError(msg)
        object.__setattr__(self, "target_manifest_checksum", target_manifest_checksum)
        object.__setattr__(self, "run_plan_checksum", require_checksum(self.run_plan_checksum, "run_plan_checksum"))
        outcomes = tuple(self.job_outcomes)
        if not outcomes or not all(isinstance(item, TrainingJobOutcome) for item in outcomes):
            msg = "Primary analysis requires typed TrainingJobOutcome source artifacts."
            raise TypeError(msg)
        if any(item.attempt != 1 for item in outcomes):
            msg = "Primary analysis accepts only authoritative first terminal confirmatory attempts."
            raise ValueError(msg)
        outcomes = tuple(sorted(outcomes, key=lambda item: item.job_checksum))
        if len({item.job_checksum for item in outcomes}) != len(outcomes):
            msg = "Primary analysis requires one typed outcome per exact job."
            raise ValueError(msg)
        outcome_checksums = tuple(sorted(item.content_checksum for item in outcomes))
        if len(outcome_checksums) != len(set(outcome_checksums)):
            msg = "Primary analysis requires one distinct typed outcome checksum per job."
            raise ValueError(msg)
        object.__setattr__(self, "job_outcomes", outcomes)
        receipts = tuple(self.production_attempt_receipts)
        if not receipts or not all(isinstance(item, ConfirmatoryProductionAttemptReceipt) for item in receipts):
            msg = "Primary analysis requires one typed production-attempt receipt per job."
            raise TypeError(msg)
        receipts = tuple(sorted(receipts, key=lambda item: item.job_checksum))
        if len({item.job_checksum for item in receipts}) != len(receipts):
            msg = "Primary analysis requires one distinct production-attempt receipt per job."
            raise ValueError(msg)
        if len({item.request_checksum for item in receipts}) != len(receipts):
            msg = "Primary analysis requires one distinct confirm request per production attempt."
            raise ValueError(msg)
        receipt_checksums = tuple(sorted(item.content_checksum for item in receipts))
        if len(receipt_checksums) != len(set(receipt_checksums)):
            msg = "Primary analysis requires distinct production-attempt receipt checksums."
            raise ValueError(msg)
        object.__setattr__(self, "production_attempt_receipts", receipts)
        results = tuple(self.confirmatory_results)
        if not all(isinstance(item, ConfirmatoryResultArtifact) for item in results):
            msg = "Primary analysis requires authoritative ConfirmatoryResultArtifact sources."
            raise TypeError(msg)
        results = tuple(sorted(results, key=lambda item: item.job_checksum))
        if len({item.job_checksum for item in results}) != len(results):
            msg = "Primary analysis requires at most one confirmatory result per exact job."
            raise ValueError(msg)
        result_checksums = tuple(sorted(item.content_checksum for item in results))
        if len(result_checksums) != len(set(result_checksums)):
            msg = "Primary analysis requires distinct successful confirmatory result checksums."
            raise ValueError(msg)
        if len({item.source_evaluation.content_checksum for item in results}) != len(results):
            msg = "Primary analysis requires distinct typed raw evaluation source identities."
            raise ValueError(msg)
        if len({item.source_result_reference_checksum for item in results}) != len(results):
            msg = "Primary analysis requires distinct production result-reference roots."
            raise ValueError(msg)
        if len({item.source_production_evidence_checksum for item in results}) != len(results):
            msg = "Primary analysis requires distinct production-evidence roots."
            raise ValueError(msg)
        object.__setattr__(self, "confirmatory_results", results)
        observations = _validate_observation_universe(self.final_seal, self.observations)
        if set(outcome_checksums) != {item.job_outcome_checksum for item in observations}:
            msg = "Primary-analysis rows do not reference the exact committed job-outcome universe."
            raise ValueError(msg)
        successful_results = {item.result_record_checksum for item in observations if item.status == "success"}
        if set(result_checksums) != successful_results:
            msg = "Primary-analysis rows do not reference the exact committed successful-result universe."
            raise ValueError(msg)
        outcomes_by_job = {item.job_checksum: item for item in outcomes}
        receipts_by_job = {item.job_checksum: item for item in receipts}
        results_by_job = {item.job_checksum: item for item in results}
        if {item.job_checksum for item in observations} != set(outcomes_by_job) or set(receipts_by_job) != set(
            outcomes_by_job
        ):
            msg = "Primary-analysis rows and receipts do not cover the exact terminal job universe."
            raise ValueError(msg)
        for observation in observations:
            outcome = outcomes_by_job[observation.job_checksum]
            receipt = receipts_by_job[observation.job_checksum]
            if (
                observation.job_outcome_checksum != outcome.content_checksum
                or observation.status != outcome.status
                or receipt.job_outcome_checksum != outcome.content_checksum
                or receipt.status != outcome.status
            ):
                msg = "Primary-analysis row and receipt status are not dereferenced from the typed job outcome."
                raise ValueError(msg)
            if outcome.status == "success":
                result = results_by_job.get(observation.job_checksum)
                raw_payload = receipt.raw_trajectory_payload
                raw_values = None if raw_payload is None else raw_payload.get("trajectory_fidelities")
                if result is None or raw_payload is None or type(raw_values) is not tuple:
                    msg = "Primary-analysis success fidelity is not dereferenced from its typed raw result source."
                    raise ValueError(msg)
                if (
                    outcome.result_artifact_checksum != result.source_result_reference_checksum
                    or receipt.result_reference.content_checksum != result.source_result_reference_checksum
                    or receipt.production_evidence.content_checksum != result.source_production_evidence_checksum
                    or raw_values != result.source_evaluation.trajectory_evidence.trajectory_fidelities
                    or raw_payload.get("evaluation_seed") != result.evaluation_seed
                    or raw_payload.get("trajectory_count") != result.test_trajectory_count
                    or observation.result_record_checksum != result.content_checksum
                    or observation.result_schema_version != result.schema_version
                    or observation.evaluation_evidence_checksum != result.evaluation_evidence_checksum
                    or observation.fresh_test_noisy_fidelity is None
                    or float(observation.fresh_test_noisy_fidelity).hex()
                    != float(result.fresh_test_noisy_fidelity).hex()
                ):
                    msg = "Primary-analysis success fidelity is not dereferenced from its typed raw result source."
                    raise ValueError(msg)
            elif observation.job_checksum in results_by_job:
                msg = "Failed primary-analysis rows cannot reference a successful typed result source."
                raise ValueError(msg)
        object.__setattr__(self, "observations", observations)

    @property
    def job_outcome_checksums(self) -> tuple[str, ...]:
        """Sorted identities of the embedded typed terminal outcomes."""
        return tuple(sorted(item.content_checksum for item in self.job_outcomes))

    @property
    def production_attempt_receipt_checksums(self) -> tuple[str, ...]:
        """Sorted identities of all embedded first-attempt custody projections."""
        return tuple(sorted(item.content_checksum for item in self.production_attempt_receipts))

    @property
    def confirmatory_result_checksums(self) -> tuple[str, ...]:
        """Sorted identities of the embedded raw-trajectory result receipts."""
        return tuple(sorted(item.content_checksum for item in self.confirmatory_results))

    @property
    def analysis_policy(self) -> dict[str, object]:
        """Frozen primary endpoint, clustering, weighting, and multiplicity policy."""
        preregistration = load_initial_preregistration()
        return {
            **_ANALYSIS_POLICY,
            "preregistration_checksum": preregistration.content_checksum,
            "analysis_template_checksum": preregistration.analysis_template_checksum,
            "failure_policy_checksum": preregistration.failure_policy_checksum,
            "contrast_set_checksum": preregistration.contrast_set_checksum,
        }

    @property
    def contrast_results(self) -> tuple[dict[str, object], ...]:
        """Paired estimates, clustered uncertainty, and Holm-controlled claims."""
        preregistration = load_initial_preregistration()
        return _apply_holm(_raw_contrast_results(self.final_seal, preregistration, self.observations))

    @property
    def failure_rate_results(self) -> tuple[dict[str, object], ...]:
        """Configuration-specific family-equal failure-rate endpoints."""
        return _failure_rate_results(self.final_seal, self.observations)

    @property
    def content_checksum(self) -> str:
        """Checksum of the final seal, raw universe, and mechanically derived analysis."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered analysis content."""
        return {
            "schema_version": self.schema_version,
            "analysis_policy": self.analysis_policy,
            "final_seal": self.final_seal.to_dict(),
            "target_manifest_checksum": self.target_manifest_checksum,
            "run_plan_checksum": self.run_plan_checksum,
            "job_outcomes": [item.to_dict() for item in self.job_outcomes],
            "job_outcome_checksums": list(self.job_outcome_checksums),
            "production_attempt_receipts": [item.to_dict() for item in self.production_attempt_receipts],
            "production_attempt_receipt_checksums": list(self.production_attempt_receipt_checksums),
            "confirmatory_results": [item.to_dict() for item in self.confirmatory_results],
            "confirmatory_result_checksums": list(self.confirmatory_result_checksums),
            "observations": [item.to_dict() for item in self.observations],
            "contrast_results": list(self.contrast_results),
            "failure_rate_results": list(self.failure_rate_results),
        }

    def to_dict(self) -> dict[str, object]:
        """Return sealed JSON-native primary-analysis data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> PrimaryAnalysisResult:
        """Decode and recompute an already issued primary-analysis result.

        Portable decoding verifies the embedded typed custody and statistics;
        it does not authorize detached evidence as a new production analysis.

        Args:
            data: Exact sealed result mapping.

        Returns:
            The reconstructed result after raw-row recomputation.

        Raises:
            TypeError: If the observation collection is not a sequence.
            ValueError: If any schema, checksum, policy, or derived result differs.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_RESULT_KEYS, name="primary analysis result")
        if mapping["schema_version"] != PRIMARY_ANALYSIS_RESULT_SCHEMA_VERSION:
            msg = f"schema_version must be {PRIMARY_ANALYSIS_RESULT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        raw_observations = mapping["observations"]
        raw_outcomes = mapping["job_outcomes"]
        raw_outcome_checksums = mapping["job_outcome_checksums"]
        raw_receipts = mapping["production_attempt_receipts"]
        raw_receipt_checksums = mapping["production_attempt_receipt_checksums"]
        raw_results = mapping["confirmatory_results"]
        raw_result_checksums = mapping["confirmatory_result_checksums"]
        if isinstance(raw_observations, (str, bytes)) or not isinstance(raw_observations, Sequence):
            msg = "primary analysis observations must be a sequence."
            raise TypeError(msg)
        if isinstance(raw_outcome_checksums, (str, bytes)) or not isinstance(raw_outcome_checksums, Sequence):
            msg = "primary analysis job-outcome checksums must be a sequence."
            raise TypeError(msg)
        if isinstance(raw_outcomes, (str, bytes)) or not isinstance(raw_outcomes, Sequence):
            msg = "primary analysis typed job outcomes must be a sequence."
            raise TypeError(msg)
        if isinstance(raw_receipt_checksums, (str, bytes)) or not isinstance(raw_receipt_checksums, Sequence):
            msg = "primary analysis production-attempt receipt checksums must be a sequence."
            raise TypeError(msg)
        if isinstance(raw_receipts, (str, bytes)) or not isinstance(raw_receipts, Sequence):
            msg = "primary analysis typed production-attempt receipts must be a sequence."
            raise TypeError(msg)
        if isinstance(raw_result_checksums, (str, bytes)) or not isinstance(raw_result_checksums, Sequence):
            msg = "primary analysis confirmatory-result checksums must be a sequence."
            raise TypeError(msg)
        if isinstance(raw_results, (str, bytes)) or not isinstance(raw_results, Sequence):
            msg = "primary analysis typed confirmatory results must be a sequence."
            raise TypeError(msg)
        result = cls(
            final_seal=FinalConfirmationSeal.from_dict(mapping["final_seal"]),
            target_manifest_checksum=cast("str", mapping["target_manifest_checksum"]),
            run_plan_checksum=cast("str", mapping["run_plan_checksum"]),
            job_outcomes=tuple(TrainingJobOutcome.from_dict(item) for item in raw_outcomes),
            production_attempt_receipts=tuple(
                ConfirmatoryProductionAttemptReceipt.from_dict(item) for item in raw_receipts
            ),
            confirmatory_results=tuple(ConfirmatoryResultArtifact.from_dict(item) for item in raw_results),
            observations=tuple(ConfirmatoryObservation.from_dict(item) for item in raw_observations),
            _construction_authority=_PRIMARY_ANALYSIS_REOPEN_AUTHORITY,
        )
        if tuple(raw_outcome_checksums) != result.job_outcome_checksums:
            msg = "Primary-analysis outcome checksums are not derived from embedded typed outcomes."
            raise ValueError(msg)
        if tuple(raw_receipt_checksums) != result.production_attempt_receipt_checksums:
            msg = "Primary-analysis receipt checksums are not derived from embedded production custody."
            raise ValueError(msg)
        if tuple(raw_result_checksums) != result.confirmatory_result_checksums:
            msg = "Primary-analysis result checksums are not derived from embedded typed results."
            raise ValueError(msg)
        for name, derived in (
            ("analysis_policy", result.analysis_policy),
            ("contrast_results", result.contrast_results),
            ("failure_rate_results", result.failure_rate_results),
        ):
            if canonical_checksum(mapping[name]) != canonical_checksum(derived):
                msg = f"Primary analysis field {name!r} is not derived from the sealed raw universe."
                raise ValueError(msg)
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Primary-analysis checksum changed during normalization."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> PrimaryAnalysisResult:
        """Decode a primary-analysis result from canonical JSON.

        Args:
            payload: Canonical sealed JSON.

        Returns:
            The recomputed primary-analysis result.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _analyze_confirmatory_statistics(
    seal: FinalConfirmationSeal,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
    target_manifest: TargetPopulationManifest,
    run_plan: TrainingRunPlan,
    job_outcomes: Sequence[TrainingJobOutcome],
    result_artifacts: Sequence[ConfirmatoryResultArtifact],
) -> _ConfirmatoryAnalysisStatistics:
    """Exercise the frozen statistical seam without issuing a production schema.

    Args:
        seal: Immutable final confirmation seal.
        configuration_execution_manifest: Exact per-configuration execution identities.
        target_manifest: Revealed typed manifest matching the seal commitment.
        run_plan: Exact confirmatory plan generated from the seal and manifest.
        job_outcomes: One typed durable orchestration outcome per plan job.
        result_artifacts: One typed fresh-test result per successful job.

    Returns:
        A private non-production statistical view for focused verification.

    Raises:
        TypeError: If a seal or custody artifact has the wrong typed schema.
    """
    if not isinstance(seal, FinalConfirmationSeal):
        msg = "seal must be a FinalConfirmationSeal."
        raise TypeError(msg)
    _validate_seal_against_preregistration(seal, load_initial_preregistration())
    ordered_outcomes, ordered_results, derived_observations = _validate_confirmatory_custody(
        seal,
        configuration_execution_manifest,
        target_manifest,
        run_plan,
        job_outcomes,
        result_artifacts,
    )
    ordered_observations = _validate_observation_universe(seal, derived_observations)
    return _ConfirmatoryAnalysisStatistics(
        final_seal=seal,
        target_manifest_checksum=target_manifest.content_checksum,
        run_plan_checksum=run_plan.content_checksum,
        job_outcomes=ordered_outcomes,
        confirmatory_results=ordered_results,
        observations=ordered_observations,
    )


def analyze_production_confirmatory_results(
    confirmation_context: ConfirmationExecutionContext,
    output_root: Path,
) -> PrimaryAnalysisResult:
    """Run primary analysis only after reopening every first production attempt.

    Synthetic WP22 fixtures may test lower custody layers, but this public API
    rejects them and accepts only real pipeline or operator-growth attempts.

    Returns:
        The frozen cell-level primary analysis over manifest-derived raw means.

    Raises:
        TypeError: If the authority or output_root has the wrong type.
        ValueError: If an exact first attempt or production result is absent.
    """
    if not isinstance(confirmation_context, ConfirmationExecutionContext):
        msg = "confirmation_context must be a ConfirmationExecutionContext."
        raise TypeError(msg)
    if not isinstance(output_root, Path):
        msg = "output_root must be a pathlib.Path."
        raise TypeError(msg)
    seal = confirmation_context.final_seal
    configuration_execution_manifest = confirmation_context.configuration_execution_manifest
    target_manifest = confirmation_context.target_manifest
    run_plan = confirmation_context.plan
    outcomes: list[TrainingJobOutcome] = []
    receipts: list[ConfirmatoryProductionAttemptReceipt] = []
    results: list[ConfirmatoryResultArtifact] = []
    for job in run_plan.jobs:
        job_directory = output_root / job.output_path
        history = load_training_job_outcome_history(job_directory, job)
        if len(history) != 1:
            msg = "Confirmatory analysis requires exactly one authoritative terminal outcome per job."
            raise ValueError(msg)
        outcome = history[0]
        custody = reopen_confirmatory_production_attempt(
            job,
            outcome,
            job_directory,
            confirmation_context,
        )
        outcomes.append(outcome)
        receipts.append(ConfirmatoryProductionAttemptReceipt.create(job, outcome, custody, confirmation_context))
        if outcome.status == "failure":
            continue
        results.append(
            ConfirmatoryResultArtifact.create(
                job,
                seal,
                custody,
                confirmation_context,
            )
        )
    statistics_view = _analyze_confirmatory_statistics(
        seal,
        configuration_execution_manifest,
        target_manifest,
        run_plan,
        outcomes,
        results,
    )
    return PrimaryAnalysisResult(
        final_seal=statistics_view.final_seal,
        target_manifest_checksum=statistics_view.target_manifest_checksum,
        run_plan_checksum=statistics_view.run_plan_checksum,
        job_outcomes=statistics_view.job_outcomes,
        production_attempt_receipts=tuple(receipts),
        confirmatory_results=statistics_view.confirmatory_results,
        observations=statistics_view.observations,
        _construction_authority=_PRODUCTION_ANALYSIS_CONSTRUCTION_AUTHORITY,
    )


__all__ = [
    "CONFIRMATORY_DATA_ROLE",
    "CONFIRMATORY_EVALUATION_ARTIFACT_SCHEMA_VERSION",
    "CONFIRMATORY_OBSERVATION_SCHEMA_VERSION",
    "CONFIRMATORY_PRODUCTION_ATTEMPT_RECEIPT_SCHEMA_VERSION",
    "CONFIRMATORY_RESULT_ARTIFACT_SCHEMA_VERSION",
    "PRIMARY_ANALYSIS_POLICY_ID",
    "PRIMARY_ANALYSIS_RESULT_SCHEMA_VERSION",
    "ConfirmatoryEvaluationArtifact",
    "ConfirmatoryObservation",
    "ConfirmatoryProductionAttemptReceipt",
    "ConfirmatoryResultArtifact",
    "PrimaryAnalysisResult",
    "analyze_production_confirmatory_results",
]
