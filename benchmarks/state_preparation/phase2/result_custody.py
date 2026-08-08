# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Raw fixed-trajectory and immutable production-result custody for WP22."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from .canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
    verify_sealed_mapping,
)
from .execution_context import ConfirmationExecutionContext
from .production_executors import (
    PilotDiagnosticEvidence,
    ReopenedProductionResult,
    derive_result_artifact_ref,
    reopen_result_artifact,
    validate_existing_confirmation_outcome,
)
from .training_orchestration import (
    TrainingJob,
    TrainingJobOutcome,
    confirmatory_evaluation_policy_checksum,
)
from .validation import require_checksum, require_float, require_int, require_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from .production_executors import (
        ProductionNumericalEvidence,
        ResultArtifactRef,
    )

TRAJECTORY_FIDELITY_EVIDENCE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.trajectory_fidelity_evidence.v1"
PRODUCTION_RESULT_CUSTODY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.production_result_custody.v1"

_PRODUCTION_DOCUMENT_KEYS = frozenset({
    "schema_version",
    "document_type",
    "payload",
    "content_checksum",
})

_TRAJECTORY_EVIDENCE_KEYS = frozenset({
    "schema_version",
    "evaluation_context_checksum",
    "data_role",
    "evaluation_seed",
    "trajectory_fidelities",
    "content_checksum",
})


def production_noisy_fidelity(values: tuple[float, ...]) -> float:
    """Reproduce the WP22E float64 aggregation algorithm exactly.

    Returns:
        The NumPy float64 mean used by ``noisy_state_preparation_metrics``.
    """
    return float(np.mean(np.asarray(values, dtype=np.float64)))


@dataclass(frozen=True, slots=True)
class TrajectoryFidelityEvidence:
    """Checksum-sealed raw fidelities for one fixed evaluation ensemble.

    The aggregate is intentionally a property rather than a constructor field,
    so downstream custody records cannot pair a caller-authored mean with an
    unrelated checksum.
    """

    evaluation_context_checksum: str
    data_role: Literal["screening_selection", "confirmatory"]
    evaluation_seed: int
    trajectory_fidelities: tuple[float, ...]
    schema_version: str = field(default=TRAJECTORY_FIDELITY_EVIDENCE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the fixed ensemble identity and every raw fidelity.

        Raises:
            ValueError: If the context, role, seed, or trajectory values are invalid.
        """
        object.__setattr__(
            self,
            "evaluation_context_checksum",
            require_checksum(self.evaluation_context_checksum, "evaluation_context_checksum"),
        )
        if self.data_role not in {"screening_selection", "confirmatory"}:
            msg = "Trajectory fidelity evidence requires a screening_selection or confirmatory role."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "evaluation_seed",
            require_int(self.evaluation_seed, "evaluation_seed", minimum=0),
        )
        fidelities = tuple(
            require_float(value, f"trajectory_fidelities[{index}]", minimum=0.0, maximum=1.0)
            for index, value in enumerate(self.trajectory_fidelities)
        )
        if not fidelities:
            msg = "trajectory_fidelities must contain at least one fixed-ensemble result."
            raise ValueError(msg)
        object.__setattr__(self, "trajectory_fidelities", fidelities)

    @property
    def mean_fidelity(self) -> float:
        """WP22E float64 mean mechanically derived from raw trajectories."""
        return production_noisy_fidelity(self.trajectory_fidelities)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered raw-evidence field."""
        return {
            "schema_version": self.schema_version,
            "evaluation_context_checksum": self.evaluation_context_checksum,
            "data_role": self.data_role,
            "evaluation_seed": self.evaluation_seed,
            "trajectory_fidelities": list(self.trajectory_fidelities),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact context and raw fixed-trajectory values."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native raw evidence."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> TrajectoryFidelityEvidence:
        """Decode and checksum-verify raw fixed-trajectory evidence.

        Returns:
            The verified raw trajectory evidence.

        Raises:
            TypeError: If serialized trajectory values are not a sequence.
            ValueError: If its schema or checksum differs.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_TRAJECTORY_EVIDENCE_KEYS,
            name="trajectory fidelity evidence",
        )
        if mapping["schema_version"] != TRAJECTORY_FIDELITY_EVIDENCE_SCHEMA_VERSION:
            msg = "Trajectory fidelity evidence uses an unsupported schema version."
            raise ValueError(msg)
        values = mapping["trajectory_fidelities"]
        if type(values) is not tuple:
            msg = "trajectory_fidelities must be a serialized sequence."
            raise TypeError(msg)
        evidence = cls(
            evaluation_context_checksum=cast("str", mapping["evaluation_context_checksum"]),
            data_role=cast("Literal['screening_selection', 'confirmatory']", mapping["data_role"]),
            evaluation_seed=cast("int", mapping["evaluation_seed"]),
            trajectory_fidelities=cast("tuple[float, ...]", values),
        )
        if mapping["content_checksum"] != evidence.content_checksum:
            msg = "Trajectory fidelity evidence checksum changed during normalization."
            raise ValueError(msg)
        return evidence

    @classmethod
    def from_json(cls, payload: str) -> TrajectoryFidelityEvidence:
        """Decode canonical checksum-sealed raw evidence.

        Returns:
            The verified raw trajectory evidence.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _production_document_payload(
    document: Mapping[str, object],
    *,
    document_type: str,
    logical_checksum: str,
) -> dict[str, object]:
    """Reverify one manifest-addressed production document and return its payload.

    Returns:
        The exact checksum-verified document payload.

    Raises:
        ValueError: If the typed document differs from its manifest reference.
    """
    mapping = verify_sealed_mapping(
        document,
        expected_keys=_PRODUCTION_DOCUMENT_KEYS,
        name=f"{document_type} production document",
    )
    if (
        mapping["schema_version"] != "yaqs.state_preparation.phase2.production_document.v1"
        or mapping["document_type"] != document_type
        or mapping["content_checksum"] != logical_checksum
    ):
        msg = f"Production document does not reproduce its {document_type!r} manifest reference."
        raise ValueError(msg)
    return dict(require_mapping(mapping["payload"], f"{document_type}.payload"))


@dataclass(frozen=True, slots=True, init=False)
class ProductionResultCustody:
    """Runtime-only view obtained by reopening an immutable WP22E attempt.

    The constructor accepts only :class:`ReopenedProductionResult`, so pilot,
    screening, and confirmatory adapters cannot authenticate caller-authored
    summaries or detached checksums.  The complete attempt is reopened and all
    member bytes are checked by WP22E before this projection is created.
    """

    reference: ResultArtifactRef
    production_evidence: ProductionNumericalEvidence
    result_evidence_checksum: str
    raw_trajectory_payload: Mapping[str, object] | None
    raw_trajectory_document_checksum: str | None
    resource_payload: Mapping[str, object]
    resource_document_checksum: str
    pilot_diagnostics: tuple[PilotDiagnosticEvidence, ...]
    schema_version: str = field(default=PRODUCTION_RESULT_CUSTODY_SCHEMA_VERSION, init=False)

    def __init__(self, reopened: ReopenedProductionResult) -> None:
        """Recompute every scientific projection from one reopened attempt.

        Raises:
            TypeError: If the source is not a reopened WP22E result.
            ValueError: If the attempt is not first or a document alias differs.
        """
        if not isinstance(reopened, ReopenedProductionResult):
            msg = "reopened must be a ReopenedProductionResult."
            raise TypeError(msg)
        reference = reopened.reference
        evidence = reopened.evidence
        if reference.attempt != 1 or evidence.attempt != 1 or reopened.manifest.attempt != 1:
            msg = "Scientific WP22 custody accepts only the authoritative first attempt."
            raise ValueError(msg)
        if evidence.content_checksum != reference.evidence_checksum:
            msg = "Reopened numerical evidence differs from its typed result reference."
            raise ValueError(msg)

        resource_payload = _production_document_payload(
            reopened.resources,
            document_type="runtime_resources",
            logical_checksum=evidence.resource_ref.logical_checksum,
        )
        resource_identity = resource_payload.get("job_checksum", resource_payload.get("request_checksum"))
        if (
            resource_identity != reference.job_checksum
            or resource_payload.get("source_fingerprint_checksum") != reference.source_fingerprint_checksum
        ):
            msg = "Runtime resources differ from the exact job or source fingerprint."
            raise ValueError(msg)

        raw_payload: dict[str, object] | None = None
        raw_checksum: str | None = None
        raw_ref = evidence.raw_trajectory_ref
        if reopened.raw_trajectory is not None:
            if raw_ref is None:
                msg = "Reopened raw trajectories lack a typed manifest reference."
                raise ValueError(msg)
            raw_payload = _production_document_payload(
                reopened.raw_trajectory,
                document_type="raw_trajectory_fidelities",
                logical_checksum=raw_ref.logical_checksum,
            )
            raw_checksum = raw_ref.logical_checksum
            trajectory_values = raw_payload.get("trajectory_fidelities")
            if not isinstance(trajectory_values, tuple):
                msg = "Raw production trajectory fidelities must be a canonical JSON array."
                raise TypeError(msg)
            fidelities = tuple(
                require_float(value, "trajectory_fidelity", minimum=0.0, maximum=1.0) for value in trajectory_values
            )
            count = require_int(raw_payload.get("trajectory_count"), "trajectory_count", minimum=1)
            raw_identity = raw_payload.get("job_checksum", raw_payload.get("request_checksum"))
            if (
                len(fidelities) != count
                or raw_identity != reference.job_checksum
                or raw_payload.get("evaluation_policy_checksum") != evidence.evaluation_policy_checksum
            ):
                msg = "Raw trajectories differ from their exact job, policy, or fixed count."
                raise ValueError(msg)
            mean = production_noisy_fidelity(fidelities)
            supplied = require_float(
                evidence.derived_metrics.get("noisy_fidelity"),
                "derived_metrics.noisy_fidelity",
                minimum=0.0,
                maximum=1.0,
            )
            if float(mean).hex() != float(supplied).hex():
                msg = "Production noisy fidelity is not derived from the manifest-addressed raw trajectories."
                raise ValueError(msg)
            raw_payload["trajectory_fidelities"] = fidelities
        elif raw_ref is not None:
            msg = "Typed production evidence references raw trajectories that were not reopened."
            raise ValueError(msg)

        diagnostics = tuple(PilotDiagnosticEvidence.from_dict(item) for item in reopened.diagnostic_documents)
        if tuple(item.content_checksum for item in diagnostics) != tuple(
            ref.logical_checksum for ref in evidence.diagnostic_refs
        ):
            msg = "Reopened pilot diagnostics differ from their immutable manifest references."
            raise ValueError(msg)

        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "production_evidence", evidence)
        object.__setattr__(self, "result_evidence_checksum", evidence.content_checksum)
        object.__setattr__(self, "raw_trajectory_payload", raw_payload)
        object.__setattr__(self, "raw_trajectory_document_checksum", raw_checksum)
        object.__setattr__(self, "resource_payload", resource_payload)
        object.__setattr__(self, "resource_document_checksum", evidence.resource_ref.logical_checksum)
        object.__setattr__(self, "pilot_diagnostics", diagnostics)
        object.__setattr__(self, "schema_version", PRODUCTION_RESULT_CUSTODY_SCHEMA_VERSION)

    @property
    def trajectory_fidelities(self) -> tuple[float, ...] | None:
        """Exact fixed-count fidelities, or ``None`` for a result without an outer evaluation."""
        if self.raw_trajectory_payload is None:
            return None
        return cast("tuple[float, ...]", self.raw_trajectory_payload["trajectory_fidelities"])

    @property
    def mean_fidelity(self) -> float | None:
        """Mean mechanically derived from reopened raw trajectories."""
        values = self.trajectory_fidelities
        return None if values is None else production_noisy_fidelity(values)

    @property
    def content_checksum(self) -> str:
        """Checksum of immutable source references, never a replacement source."""
        return canonical_checksum({
            "schema_version": self.schema_version,
            "result_reference_checksum": self.reference.content_checksum,
            "result_evidence_checksum": self.result_evidence_checksum,
            "raw_trajectory_document_checksum": self.raw_trajectory_document_checksum,
            "resource_document_checksum": self.resource_document_checksum,
            "pilot_diagnostic_checksums": [item.content_checksum for item in self.pilot_diagnostics],
        })

    def to_dict(self) -> dict[str, object]:
        """Return an audit projection which must be reopened before scientific reuse."""
        return {
            "schema_version": self.schema_version,
            "result_reference": self.reference.to_dict(),
            "result_evidence": self.production_evidence.to_dict(),
            "raw_trajectory_document_checksum": self.raw_trajectory_document_checksum,
            "resource_document_checksum": self.resource_document_checksum,
            "pilot_diagnostic_checksums": [item.content_checksum for item in self.pilot_diagnostics],
            "content_checksum": self.content_checksum,
        }


def reopen_production_result(
    reference: ResultArtifactRef,
    job_directory: Path,
) -> ProductionResultCustody:
    """Reopen and authenticate one immutable first-attempt production result.

    Returns:
        Runtime-only custody over the fully reopened result.
    """
    return ProductionResultCustody(reopen_result_artifact(reference, job_directory))


def reopen_production_job_result(
    job: TrainingJob,
    outcome: TrainingJobOutcome,
    job_directory: Path,
) -> ProductionResultCustody:
    """Derive and reopen a success reference solely from its terminal manifest.

    Returns:
        Runtime custody bound to the exact orchestration checksum.

    Raises:
        TypeError: If job or outcome has the wrong typed schema.
        ValueError: If the outcome is not this job's first successful attempt.
    """
    if not isinstance(job, TrainingJob) or not isinstance(outcome, TrainingJobOutcome):
        msg = "job and outcome must be typed orchestration records."
        raise TypeError(msg)
    if (
        outcome.job_checksum != job.content_checksum
        or outcome.status != "success"
        or outcome.attempt != 1
        or outcome.result_artifact_checksum is None
    ):
        msg = "Production custody requires the exact first successful outcome for its job."
        raise ValueError(msg)
    reference = derive_result_artifact_ref(
        job_directory,
        job.content_checksum,
        1,
        expected_reference_checksum=outcome.result_artifact_checksum,
    )
    return reopen_production_result(reference, job_directory)


def reopen_terminal_production_attempt(
    job: TrainingJob,
    outcome: TrainingJobOutcome,
    job_directory: Path,
) -> ProductionResultCustody:
    """Derive the first terminal production manifest for success or failure.

    Returns:
        Runtime custody over the exact first production attempt.

    Raises:
        TypeError: If job or outcome has the wrong typed schema.
        ValueError: If the outcome does not belong to the job's first attempt.
    """
    if not isinstance(job, TrainingJob) or not isinstance(outcome, TrainingJobOutcome):
        msg = "job and outcome must be typed orchestration records."
        raise TypeError(msg)
    if outcome.job_checksum != job.content_checksum or outcome.attempt != 1:
        msg = "Production custody requires this job's authoritative first outcome."
        raise ValueError(msg)
    reference = derive_result_artifact_ref(
        job_directory,
        job.content_checksum,
        1,
        expected_reference_checksum=outcome.result_artifact_checksum,
    )
    if reference.status != outcome.status:
        msg = "Production terminal manifest status differs from orchestration custody."
        raise ValueError(msg)
    return reopen_production_result(reference, job_directory)


def reopen_confirmatory_production_attempt(
    job: TrainingJob,
    outcome: TrainingJobOutcome,
    job_directory: Path,
    confirmation_context: ConfirmationExecutionContext | None = None,
) -> ProductionResultCustody:
    """Reopen a seal-complete confirmation attempt through its request identity.

    The confirmation executor ABI receives the nested request rather than the
    enclosing job, so its immutable manifest is keyed by that request checksum.

    Returns:
        Runtime custody over the exact first confirmation attempt.

    Raises:
        TypeError: If inputs have the wrong typed schemas.
        ValueError: If job, request, outcome, or production evidence differs.
    """
    if not isinstance(job, TrainingJob) or not isinstance(outcome, TrainingJobOutcome):
        msg = "job and outcome must be typed orchestration records."
        raise TypeError(msg)
    request = job.confirm_execution_request
    if (
        request is None
        or job.preset != "paper-confirm"
        or outcome.job_checksum != job.content_checksum
        or outcome.attempt != 1
    ):
        msg = "Confirmation custody requires an exact first-attempt paper-confirm job."
        raise ValueError(msg)
    if confirmation_context is None:
        reference = derive_result_artifact_ref(
            job_directory,
            request.content_checksum,
            1,
            expected_reference_checksum=outcome.result_artifact_checksum,
        )
        custody = reopen_production_result(reference, job_directory)
    else:
        reopened = validate_existing_confirmation_outcome(
            confirmation_context,
            job,
            outcome,
            job_directory,
        )
        reference = reopened.reference
        custody = ProductionResultCustody(reopened)
    evidence = custody.production_evidence
    raw = custody.raw_trajectory_payload
    expected_evaluation_policy_checksum = confirmatory_evaluation_policy_checksum(request)
    is_synthetic = evidence.artifact_kind == "synthetic_confirmation"
    if confirmation_context is not None and not isinstance(confirmation_context, ConfirmationExecutionContext):
        msg = "confirmation_context must be a ConfirmationExecutionContext."
        raise TypeError(msg)
    if not is_synthetic and confirmation_context is None:
        msg = "Real confirmation custody requires its exact ConfirmationExecutionContext authority."
        raise ValueError(msg)
    expected_program_checksum = (
        request.hyperparameters_checksum
        if is_synthetic
        else cast("ConfirmationExecutionContext", confirmation_context).scheduled_program_checksum(request)
    )
    expected_artifact_kind = (
        "synthetic_confirmation"
        if is_synthetic
        else cast("ConfirmationExecutionContext", confirmation_context).artifact_kind(request)
    )
    if (
        reference.status != outcome.status
        or reference.artifact_kind != evidence.artifact_kind
        or evidence.artifact_kind != expected_artifact_kind
        or reference.job_checksum != request.content_checksum
        or reference.execution_source_manifest_checksum != request.execution_source_checksum
        or reference.source_fingerprint_checksum != request.execution_source_checksum
        or evidence.job_checksum != request.content_checksum
        or evidence.execution_source_manifest_checksum != request.execution_source_checksum
        or evidence.source_fingerprint_checksum != request.execution_source_checksum
        or evidence.executable_binding_checksum != request.executable_binding_checksum
        or evidence.scheduled_program_checksum != expected_program_checksum
        or evidence.derived_metrics.get("strategy_schedule_checksum") != request.hyperparameters_checksum
        or evidence.evaluation_policy_checksum != expected_evaluation_policy_checksum
    ):
        msg = "Confirmation production execution links differ from its sealed request or outcome."
        raise ValueError(msg)
    if is_synthetic and dict(evidence.target_identity) != {
        "synthetic_fixture": True,
        "request_checksum": request.content_checksum,
        "target_instance_id": request.target_instance_id,
        "target_spec_checksum": request.target_spec_checksum,
        "qubit_count": request.qubit_count,
    }:
        msg = "Synthetic confirmation target aliases differ from its sealed request."
        raise ValueError(msg)
    if not is_synthetic:
        target = evidence.target_identity
        if (
            target.get("target_instance_id") != request.target_instance_id
            or target.get("target_instance_spec_checksum") != request.target_spec_checksum
            or target.get("target_manifest_checksum") != request.target_manifest_checksum
            or target.get("family_id") != request.family_id
            or target.get("stratum_id") != request.stratum_id
            or target.get("qubit_count") != request.qubit_count
        ):
            msg = "Real confirmation target aliases differ from its sealed request."
            raise ValueError(msg)
    if outcome.status == "success" and (
        evidence.derived_metrics.get("evaluation_data_role") != "confirmatory"
        or evidence.derived_metrics.get("evaluation_seed_domain") != "confirmatory_test"
        or evidence.derived_metrics.get("evaluation_seed") != request.evaluation_seed
        or evidence.derived_metrics.get("trajectory_count") != request.fixed_test_trajectory_count
    ):
        msg = "Confirmation derived metrics differ from the sealed role, seed, or fixed count."
        raise ValueError(msg)
    if raw is not None:
        raw_count = require_int(raw.get("trajectory_count"), "trajectory_count", minimum=1)
        if (
            raw.get("request_checksum" if is_synthetic else "job_checksum") != request.content_checksum
            or raw.get("evaluation_policy_checksum") != expected_evaluation_policy_checksum
            or raw.get("data_role") != "confirmatory"
            or raw.get("seed_domain") != "confirmatory_test"
            or raw.get("evaluation_seed") != request.evaluation_seed
            or raw_count > request.fixed_test_trajectory_count
            or (outcome.status == "success" and raw_count != request.fixed_test_trajectory_count)
        ):
            msg = "Confirmation raw trajectories differ from the sealed policy, role, seed, or count."
            raise ValueError(msg)
    elif outcome.status == "success":
        msg = "Successful confirmation custody lacks raw trajectories."
        raise ValueError(msg)
    return custody


def validate_production_job_custody(
    custody: ProductionResultCustody,
    job: TrainingJob,
    outcome: TrainingJobOutcome,
    *,
    expected_data_role: str,
    expected_trajectory_count: int | None,
    expected_execution_source_manifest_checksum: str | None = None,
    allowed_artifact_kinds: tuple[str, ...] = ("pipeline", "operator_growth", "synthetic_confirmation"),
) -> None:
    """Bind a reopened result to one exact first terminal orchestration outcome.

    Raises:
        TypeError: If a source has the wrong typed schema.
        ValueError: If any job, source, result, role, count, or first-attempt link differs.
    """
    if not isinstance(custody, ProductionResultCustody):
        msg = "custody must be ProductionResultCustody."
        raise TypeError(msg)
    if not isinstance(job, TrainingJob) or not isinstance(outcome, TrainingJobOutcome):
        msg = "job and outcome must be typed orchestration records."
        raise TypeError(msg)
    if job.data_role != expected_data_role:
        msg = "Expected scientific role differs from the exact training job."
        raise ValueError(msg)
    reference = custody.reference
    reopened = reopen_result_artifact  # retain an explicit dependency in the custody module
    del reopened
    if outcome.attempt != 1 or reference.attempt != 1:
        msg = "Scientific custody accepts only the first terminal attempt."
        raise ValueError(msg)
    if (
        outcome.job_checksum != job.content_checksum
        or reference.job_checksum != job.content_checksum
        or outcome.status != reference.status
        or (outcome.status == "success" and outcome.result_artifact_checksum != reference.content_checksum)
    ):
        msg = "Production result reference differs from its exact job or orchestration outcome."
        raise ValueError(msg)
    evidence = custody.production_evidence
    if expected_execution_source_manifest_checksum is not None:
        source_checksum = require_checksum(
            expected_execution_source_manifest_checksum,
            "expected_execution_source_manifest_checksum",
        )
        if (
            reference.execution_source_manifest_checksum != source_checksum
            or evidence.execution_source_manifest_checksum != source_checksum
        ):
            msg = "Production result differs from the exact execution-source manifest."
            raise ValueError(msg)
    if evidence.artifact_kind not in allowed_artifact_kinds:
        msg = "Production result uses an artifact kind outside this scientific custody path."
        raise ValueError(msg)
    if evidence.derived_metrics.get("execution_preset") != job.preset:
        msg = "Production evidence execution preset differs from the exact scientific job."
        raise ValueError(msg)
    expected_fingerprints = (
        (evidence.source_fingerprint_checksum, job.source_fingerprint_checksum),
        (evidence.executable_binding_checksum, job.executable_binding_checksum),
        (evidence.scheduled_program_checksum, job.scheduled_execution_program_checksum),
        (evidence.evaluation_policy_checksum, job.evaluation_policy_checksum),
    )
    if any(actual != expected for actual, expected in expected_fingerprints):
        msg = "Production evidence differs from the job's exact execution closure."
        raise ValueError(msg)
    target = evidence.target_identity
    expected_target_fields = (
        (target.get("target_instance_id"), job.target_instance_id),
        (target.get("target_instance_spec_checksum"), job.target_spec_checksum),
        (target.get("population_config_checksum"), job.target_configuration_checksum),
        (target.get("target_manifest_checksum"), job.target_manifest_checksum),
        (target.get("family_id"), job.family_id),
        (target.get("stratum_id"), job.stratum_id),
        (target.get("qubit_count"), job.qubit_count),
    )
    if any(actual != expected for actual, expected in expected_target_fields):
        msg = "Production evidence target identity differs from the exact job target."
        raise ValueError(msg)
    raw = custody.raw_trajectory_payload
    if outcome.status == "success" and expected_trajectory_count is not None:
        if raw is None:
            msg = "This scientific result requires manifest-addressed raw trajectories."
            raise ValueError(msg)
        if (
            raw.get("data_role") != expected_data_role
            or raw.get("evaluation_seed") != job.evaluation_seed
            or raw.get("trajectory_count") != expected_trajectory_count
        ):
            msg = "Production raw evidence differs from the job role, seed, or fixed trajectory count."
            raise ValueError(msg)
    elif raw is not None and raw.get("data_role") != expected_data_role:
        msg = "Production raw evidence uses the wrong data role."
        raise ValueError(msg)
    if outcome.status == "failure" and raw is not None:
        msg = "Failed production attempts cannot claim successful raw outer trajectories."
        raise ValueError(msg)


__all__ = [
    "PRODUCTION_RESULT_CUSTODY_SCHEMA_VERSION",
    "TRAJECTORY_FIDELITY_EVIDENCE_SCHEMA_VERSION",
    "ProductionResultCustody",
    "TrajectoryFidelityEvidence",
    "production_noisy_fidelity",
    "reopen_confirmatory_production_attempt",
    "reopen_production_job_result",
    "reopen_production_result",
    "reopen_terminal_production_attempt",
    "validate_production_job_custody",
]
