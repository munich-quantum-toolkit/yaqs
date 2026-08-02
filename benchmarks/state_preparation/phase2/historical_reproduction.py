# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Checksum-sealed evidence for the WP19 historical reproduction.

This module deliberately contains no expensive optimizer or evaluation runner.
It converts five already produced evaluation rows into a comparison with the
commit-addressed legacy evidence and provides the deterministic circuit payload
used by a later opt-in runner.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from .artifact_codecs import MAX_STAGE_PARAMETER_COUNT
from .canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
    thaw_json,
    verify_sealed_mapping,
)
from .legacy import TRUSTED_LEGACY_AUDIT_CHECKSUM, load_legacy_evidence_audit
from .noisy_krotov import NoisyKrotovCircuitBinding, decode_noisy_krotov_circuit_binding_document
from .pipeline import PipelineBenchmarkFailure, PipelineBenchmarkResult
from .validation import (
    require_checksum,
    require_exact_keys,
    require_float,
    require_git_blob,
    require_git_commit,
    require_int,
    require_mapping,
    require_nonempty_text,
    require_relative_path,
    require_slug,
    require_string,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from .pipeline import PipelineBenchmarkRecord

LEGACY_LAYERWISE_METHOD_ID = "layerwise_bmpd_crn_legacy_v1"
LEGACY_REPRODUCTION_TARGET_SEEDS = (100, 200, 300, 400, 500)
LEGACY_ARITHMETIC_CLAIM_ID = "shared_protocol_five_target_arithmetic"
LEGACY_RESULT_ARTIFACT_ID = "result_rigorous_csv"

LEGACY_ARCHIVED_REFERENCE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.legacy_layerwise_archived_reference.v1"
LEGACY_REPRODUCTION_OUTCOME_SCHEMA_VERSION = "yaqs.state_preparation.phase2.legacy_reproduction_outcome.v1"
LEGACY_TARGET_COMPARISON_SCHEMA_VERSION = "yaqs.state_preparation.phase2.legacy_target_comparison.v1"
LEGACY_REPRODUCTION_REPORT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.legacy_reproduction_report.v1"
LAYERWISE_MATERIALIZED_CIRCUIT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.layerwise_materialized_circuit.v1"

MAX_LAYERWISE_MATERIALIZED_CIRCUIT_BYTES = 4 * 1024 * 1024

_ARCHIVED_REFERENCE_KEYS = frozenset({
    "schema_version",
    "method_id",
    "legacy_audit_checksum",
    "claim_id",
    "claim_configuration_checksum",
    "csv_artifact_id",
    "csv_repo_path",
    "csv_source_commit",
    "csv_git_blob_id",
    "csv_content_checksum",
    "target_seeds",
    "fidelities",
    "reference_mean",
    "content_checksum",
})
_OUTCOME_KEYS = frozenset({
    "schema_version",
    "target_seed",
    "status",
    "computed_fidelity",
    "source_record_id",
    "source_record_checksum",
    "runtime_fingerprint_checksum",
    "failure_type",
    "failure_message",
    "content_checksum",
})
_COMPARISON_KEYS = frozenset({
    "schema_version",
    "target_seed",
    "outcome",
    "reference_fidelity",
    "delta",
    "absolute_delta",
    "tolerance",
    "within_tolerance",
    "content_checksum",
})
_REPORT_KEYS = frozenset({
    "schema_version",
    "report_id",
    "method_id",
    "archived_reference",
    "comparison_tolerance",
    "tolerance_rationale",
    "computed_mean_policy",
    "target_comparisons",
    "computed_mean",
    "reference_mean",
    "mean_delta",
    "absolute_mean_delta",
    "classification",
    "source_manifest_checksum",
    "runtime_checksum",
    "content_checksum",
})
_MATERIALIZED_CIRCUIT_KEYS = frozenset({
    "schema_version",
    "circuit_binding",
    "selected_parameters",
    "selected_parameter_checksum",
    "content_checksum",
})
_PARAMETER_PAYLOAD_KEYS = frozenset({"data_base64", "dtype", "shape"})


def _same_float(left: float | None, right: float | None) -> bool:
    """Return whether optional floats have exactly the same binary value."""
    if left is None or right is None:
        return left is right
    return float(left).hex() == float(right).hex()


def _require_fidelity(value: object, name: str) -> float:
    """Return one exact finite fidelity."""
    return require_float(value, name, minimum=0.0, maximum=1.0)


def _require_optional_fidelity(value: object, name: str) -> float | None:
    """Return one optional exact finite fidelity."""
    return None if value is None else _require_fidelity(value, name)


def _parameter_bytes(parameters: NDArray[np.float64]) -> bytes:
    """Return canonical little-endian bytes for a parameter vector."""
    return np.ascontiguousarray(parameters, dtype=np.dtype("<f8")).tobytes(order="C")


def _parameter_checksum(parameters: NDArray[np.float64]) -> str:
    """Return the checksum of canonical parameter bytes."""
    return f"sha256:{hashlib.sha256(_parameter_bytes(parameters)).hexdigest()}"


@dataclass(frozen=True, slots=True)
class LegacyArchivedReference:
    """Five archived layerwise fidelities and their commit-addressed source."""

    method_id: str
    legacy_audit_checksum: str
    claim_id: str
    claim_configuration_checksum: str
    csv_artifact_id: str
    csv_repo_path: str
    csv_source_commit: str
    csv_git_blob_id: str
    csv_content_checksum: str
    target_seeds: tuple[int, ...]
    fidelities: tuple[float, ...]
    reference_mean: float
    schema_version: str = field(default=LEGACY_ARCHIVED_REFERENCE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate source provenance and exact five-target arithmetic.

        Raises:
            ValueError: If the provenance or five-target arithmetic is invalid.
        """
        if self.method_id != LEGACY_LAYERWISE_METHOD_ID:
            msg = f"method_id must be {LEGACY_LAYERWISE_METHOD_ID!r}."
            raise ValueError(msg)
        audit_checksum = require_checksum(self.legacy_audit_checksum, "legacy_audit_checksum")
        if audit_checksum != TRUSTED_LEGACY_AUDIT_CHECKSUM:
            msg = "Archived reference must use the trusted legacy-evidence audit."
            raise ValueError(msg)
        if self.claim_id != LEGACY_ARITHMETIC_CLAIM_ID:
            msg = f"claim_id must be {LEGACY_ARITHMETIC_CLAIM_ID!r}."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "claim_configuration_checksum",
            require_checksum(self.claim_configuration_checksum, "claim_configuration_checksum"),
        )
        if self.csv_artifact_id != LEGACY_RESULT_ARTIFACT_ID:
            msg = f"csv_artifact_id must be {LEGACY_RESULT_ARTIFACT_ID!r}."
            raise ValueError(msg)
        object.__setattr__(self, "csv_repo_path", require_relative_path(self.csv_repo_path, "csv_repo_path"))
        object.__setattr__(
            self,
            "csv_source_commit",
            require_git_commit(self.csv_source_commit, "csv_source_commit"),
        )
        object.__setattr__(self, "csv_git_blob_id", require_git_blob(self.csv_git_blob_id, "csv_git_blob_id"))
        object.__setattr__(
            self,
            "csv_content_checksum",
            require_checksum(self.csv_content_checksum, "csv_content_checksum"),
        )
        seeds = tuple(
            require_int(seed, f"target_seeds[{index}]", minimum=0) for index, seed in enumerate(self.target_seeds)
        )
        if seeds != LEGACY_REPRODUCTION_TARGET_SEEDS:
            msg = f"target_seeds must be the exact ordered legacy seeds {LEGACY_REPRODUCTION_TARGET_SEEDS!r}."
            raise ValueError(msg)
        fidelities = tuple(
            _require_fidelity(value, f"fidelities[{index}]") for index, value in enumerate(self.fidelities)
        )
        if len(fidelities) != len(seeds):
            msg = "Archived reference must contain exactly one fidelity per target seed."
            raise ValueError(msg)
        mean = _require_fidelity(self.reference_mean, "reference_mean")
        if not _same_float(mean, fmean(fidelities)):
            msg = "reference_mean does not equal the arithmetic mean of the five archived fidelities."
            raise ValueError(msg)
        object.__setattr__(self, "target_seeds", seeds)
        object.__setattr__(self, "fidelities", fidelities)
        object.__setattr__(self, "reference_mean", mean)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered reference field."""
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "legacy_audit_checksum": self.legacy_audit_checksum,
            "claim_id": self.claim_id,
            "claim_configuration_checksum": self.claim_configuration_checksum,
            "csv_artifact_id": self.csv_artifact_id,
            "csv_repo_path": self.csv_repo_path,
            "csv_source_commit": self.csv_source_commit,
            "csv_git_blob_id": self.csv_git_blob_id,
            "csv_content_checksum": self.csv_content_checksum,
            "target_seeds": list(self.target_seeds),
            "fidelities": list(self.fidelities),
            "reference_mean": self.reference_mean,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of values and their commit-addressed provenance."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the sealed archived reference."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> LegacyArchivedReference:
        """Decode and verify a sealed archived reference.

        Returns:
            The verified archived reference.

        Raises:
            ValueError: If the document is invalid, unsupported, or inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_ARCHIVED_REFERENCE_KEYS, name="archived reference")
        if mapping["schema_version"] != LEGACY_ARCHIVED_REFERENCE_SCHEMA_VERSION:
            msg = "Archived reference uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            method_id=cast("str", mapping["method_id"]),
            legacy_audit_checksum=cast("str", mapping["legacy_audit_checksum"]),
            claim_id=cast("str", mapping["claim_id"]),
            claim_configuration_checksum=cast("str", mapping["claim_configuration_checksum"]),
            csv_artifact_id=cast("str", mapping["csv_artifact_id"]),
            csv_repo_path=cast("str", mapping["csv_repo_path"]),
            csv_source_commit=cast("str", mapping["csv_source_commit"]),
            csv_git_blob_id=cast("str", mapping["csv_git_blob_id"]),
            csv_content_checksum=cast("str", mapping["csv_content_checksum"]),
            target_seeds=cast("tuple[int, ...]", mapping["target_seeds"]),
            fidelities=cast("tuple[float, ...]", mapping["fidelities"]),
            reference_mean=cast("float", mapping["reference_mean"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Archived reference changed during normalization."
            raise ValueError(msg)
        return result


def load_archived_layerwise_reference() -> LegacyArchivedReference:
    """Load the layerwise references mechanically from the trusted audit.

    Returns:
        The five audited fidelities together with the historical CSV blob
        provenance from which the WP15 audit derived them.

    Raises:
        TypeError: If the audited seeds or fidelities are not sequences.
        ValueError: If the trusted audit does not link the expected CSV evidence.
    """
    audit = load_legacy_evidence_audit()
    claim = audit.claim(LEGACY_ARITHMETIC_CLAIM_ID)
    artifact = audit.artifact(LEGACY_RESULT_ARTIFACT_ID)
    if artifact.artifact_id not in claim.artifact_ids or artifact.role != "result":
        msg = "The trusted arithmetic claim is not linked to the archived result CSV."
        raise ValueError(msg)
    configuration = cast("Mapping[str, object]", claim.configuration)
    method_fidelities = require_mapping(configuration["method_noisy_fidelities"], "method_noisy_fidelities")
    method_means = require_mapping(configuration["method_means"], "method_means")
    raw_seeds = configuration["target_seeds"]
    raw_fidelities = method_fidelities[LEGACY_LAYERWISE_METHOD_ID]
    if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes)):
        msg = "Trusted target_seeds must be a sequence."
        raise TypeError(msg)
    if not isinstance(raw_fidelities, Sequence) or isinstance(raw_fidelities, (str, bytes)):
        msg = "Trusted layerwise fidelities must be a sequence."
        raise TypeError(msg)
    return LegacyArchivedReference(
        method_id=LEGACY_LAYERWISE_METHOD_ID,
        legacy_audit_checksum=audit.content_checksum,
        claim_id=claim.claim_id,
        claim_configuration_checksum=cast("str", claim.configuration_checksum),
        csv_artifact_id=artifact.artifact_id,
        csv_repo_path=artifact.repo_path,
        csv_source_commit=artifact.source_commit,
        csv_git_blob_id=artifact.git_blob_id,
        csv_content_checksum=artifact.content_checksum,
        target_seeds=tuple(cast("Sequence[int]", raw_seeds)),
        fidelities=tuple(cast("Sequence[float]", raw_fidelities)),
        reference_mean=cast("float", method_means[LEGACY_LAYERWISE_METHOD_ID]),
    )


@dataclass(frozen=True, slots=True)
class LegacyReproductionOutcome:
    """One computed evaluation row or retained training/orchestration failure."""

    target_seed: int
    status: Literal["success", "failure"]
    computed_fidelity: float | None
    source_record_id: str
    source_record_checksum: str
    runtime_fingerprint_checksum: str
    failure_type: str | None = None
    failure_message: str | None = None
    schema_version: str = field(default=LEGACY_REPRODUCTION_OUTCOME_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate mutually exclusive success and failure evidence.

        Raises:
            ValueError: If the outcome fields are malformed or inconsistent.
        """
        seed = require_int(self.target_seed, "target_seed", minimum=0)
        if seed not in LEGACY_REPRODUCTION_TARGET_SEEDS:
            msg = f"target_seed must be one of {LEGACY_REPRODUCTION_TARGET_SEEDS!r}."
            raise ValueError(msg)
        if self.status not in {"success", "failure"}:
            msg = "status must be 'success' or 'failure'."
            raise ValueError(msg)
        fidelity = _require_optional_fidelity(self.computed_fidelity, "computed_fidelity")
        record_id = require_slug(self.source_record_id, "source_record_id")
        record_checksum = require_checksum(self.source_record_checksum, "source_record_checksum")
        runtime_checksum = require_checksum(self.runtime_fingerprint_checksum, "runtime_fingerprint_checksum")
        failure_type = None if self.failure_type is None else require_nonempty_text(self.failure_type, "failure_type")
        failure_message = (
            None if self.failure_message is None else require_nonempty_text(self.failure_message, "failure_message")
        )
        if self.status == "success" and (fidelity is None or failure_type is not None or failure_message is not None):
            msg = "A successful outcome requires a fidelity and forbids failure diagnostics."
            raise ValueError(msg)
        if self.status == "failure" and (fidelity is not None or failure_type is None or failure_message is None):
            msg = "A failed outcome forbids a fidelity and requires complete failure diagnostics."
            raise ValueError(msg)
        object.__setattr__(self, "target_seed", seed)
        object.__setattr__(self, "computed_fidelity", fidelity)
        object.__setattr__(self, "source_record_id", record_id)
        object.__setattr__(self, "source_record_checksum", record_checksum)
        object.__setattr__(self, "runtime_fingerprint_checksum", runtime_checksum)
        object.__setattr__(self, "failure_type", failure_type)
        object.__setattr__(self, "failure_message", failure_message)

    @classmethod
    def from_pipeline_record(
        cls,
        target_seed: int,
        record: PipelineBenchmarkRecord,
    ) -> LegacyReproductionOutcome:
        """Create an outcome from one actual Phase II evaluation record.

        Returns:
            The checksum-linked success or failure projection.

        Raises:
            TypeError: If ``record`` is not a supported pipeline record.
            ValueError: If a success record lacks a noisy fidelity.
        """
        if isinstance(record, PipelineBenchmarkResult):
            if record.test_noisy_fidelity is None:
                msg = "Historical reproduction requires a noisy evaluation result."
                raise ValueError(msg)
            return cls(
                target_seed=target_seed,
                status="success",
                computed_fidelity=record.test_noisy_fidelity,
                source_record_id=record.evaluation_row_id,
                source_record_checksum=record.content_checksum,
                runtime_fingerprint_checksum=record.runtime_fingerprint_checksum,
            )
        if isinstance(record, PipelineBenchmarkFailure):
            return cls(
                target_seed=target_seed,
                status="failure",
                computed_fidelity=None,
                source_record_id=record.evaluation_row_id,
                source_record_checksum=record.content_checksum,
                runtime_fingerprint_checksum=record.runtime_fingerprint_checksum,
                failure_type=record.exception_type,
                failure_message=record.message,
            )
        msg = "record must be a PipelineBenchmarkResult or PipelineBenchmarkFailure."
        raise TypeError(msg)

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered outcome fields."""
        return {
            "schema_version": self.schema_version,
            "target_seed": self.target_seed,
            "status": self.status,
            "computed_fidelity": self.computed_fidelity,
            "source_record_id": self.source_record_id,
            "source_record_checksum": self.source_record_checksum,
            "runtime_fingerprint_checksum": self.runtime_fingerprint_checksum,
            "failure_type": self.failure_type,
            "failure_message": self.failure_message,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the supplied result or failure projection."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the sealed outcome."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> LegacyReproductionOutcome:
        """Decode and verify a sealed outcome.

        Returns:
            The verified reproduction outcome.

        Raises:
            ValueError: If the document is invalid, unsupported, or inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_OUTCOME_KEYS, name="legacy reproduction outcome")
        if mapping["schema_version"] != LEGACY_REPRODUCTION_OUTCOME_SCHEMA_VERSION:
            msg = "Legacy reproduction outcome uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            target_seed=cast("int", mapping["target_seed"]),
            status=cast("Literal['success', 'failure']", mapping["status"]),
            computed_fidelity=cast("float | None", mapping["computed_fidelity"]),
            source_record_id=cast("str", mapping["source_record_id"]),
            source_record_checksum=cast("str", mapping["source_record_checksum"]),
            runtime_fingerprint_checksum=cast("str", mapping["runtime_fingerprint_checksum"]),
            failure_type=cast("str | None", mapping["failure_type"]),
            failure_message=cast("str | None", mapping["failure_message"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Legacy reproduction outcome changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class LegacyTargetComparison:
    """One computed outcome compared with its same-seed archived reference."""

    target_seed: int
    outcome: LegacyReproductionOutcome
    reference_fidelity: float
    delta: float | None
    absolute_delta: float | None
    tolerance: float
    within_tolerance: bool
    schema_version: str = field(default=LEGACY_TARGET_COMPARISON_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact derived delta and tolerance semantics.

        Raises:
            TypeError: If ``outcome`` is not a reproduction outcome.
            ValueError: If a field is malformed or not derived from the outcome.
        """
        seed = require_int(self.target_seed, "target_seed", minimum=0)
        if not isinstance(self.outcome, LegacyReproductionOutcome):
            msg = "outcome must be a LegacyReproductionOutcome."
            raise TypeError(msg)
        if seed != self.outcome.target_seed:
            msg = "Comparison seed does not match its computed outcome."
            raise ValueError(msg)
        reference = _require_fidelity(self.reference_fidelity, "reference_fidelity")
        tolerance = require_float(self.tolerance, "tolerance", minimum=0.0)
        if tolerance <= 0.0:
            msg = "tolerance must be strictly positive."
            raise ValueError(msg)
        expected_delta = None if self.outcome.computed_fidelity is None else self.outcome.computed_fidelity - reference
        expected_absolute = None if expected_delta is None else abs(expected_delta)
        expected_within = expected_absolute is not None and expected_absolute <= tolerance
        if (
            not _same_float(self.delta, expected_delta)
            or not _same_float(self.absolute_delta, expected_absolute)
            or type(self.within_tolerance) is not bool
            or self.within_tolerance != expected_within
        ):
            msg = "Comparison deltas or tolerance decision are not derived from the supplied outcome."
            raise ValueError(msg)
        object.__setattr__(self, "target_seed", seed)
        object.__setattr__(self, "reference_fidelity", reference)
        object.__setattr__(self, "tolerance", tolerance)

    def _content_dict(self) -> dict[str, object]:
        """Return all checksum-covered comparison fields."""
        return {
            "schema_version": self.schema_version,
            "target_seed": self.target_seed,
            "outcome": self.outcome.to_dict(),
            "reference_fidelity": self.reference_fidelity,
            "delta": self.delta,
            "absolute_delta": self.absolute_delta,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of computed and archived same-seed evidence."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the sealed comparison."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> LegacyTargetComparison:
        """Decode and verify one sealed comparison.

        Returns:
            The verified same-seed comparison.

        Raises:
            ValueError: If the document is invalid, unsupported, or inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_COMPARISON_KEYS, name="legacy target comparison")
        if mapping["schema_version"] != LEGACY_TARGET_COMPARISON_SCHEMA_VERSION:
            msg = "Legacy target comparison uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            target_seed=cast("int", mapping["target_seed"]),
            outcome=LegacyReproductionOutcome.from_dict(mapping["outcome"]),
            reference_fidelity=cast("float", mapping["reference_fidelity"]),
            delta=cast("float | None", mapping["delta"]),
            absolute_delta=cast("float | None", mapping["absolute_delta"]),
            tolerance=cast("float", mapping["tolerance"]),
            within_tolerance=cast("bool", mapping["within_tolerance"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Legacy target comparison changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class LegacyReproductionReport:
    """Complete five-target WP19 numerical comparison report."""

    report_id: str
    method_id: str
    archived_reference: LegacyArchivedReference
    comparison_tolerance: float
    tolerance_rationale: str
    computed_mean_policy: str
    target_comparisons: tuple[LegacyTargetComparison, ...]
    computed_mean: float | None
    reference_mean: float
    mean_delta: float | None
    absolute_mean_delta: float | None
    classification: Literal["reproduced", "discrepant"]
    source_manifest_checksum: str
    runtime_checksum: str
    schema_version: str = field(default=LEGACY_REPRODUCTION_REPORT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Recompute ordering, arithmetic, and final classification.

        Raises:
            TypeError: If nested evidence has an invalid type.
            ValueError: If the report is malformed or has inconsistent derivations.
        """
        object.__setattr__(self, "report_id", require_slug(self.report_id, "report_id"))
        if self.method_id != LEGACY_LAYERWISE_METHOD_ID:
            msg = f"method_id must be {LEGACY_LAYERWISE_METHOD_ID!r}."
            raise ValueError(msg)
        if not isinstance(self.archived_reference, LegacyArchivedReference):
            msg = "archived_reference must be a LegacyArchivedReference."
            raise TypeError(msg)
        tolerance = require_float(self.comparison_tolerance, "comparison_tolerance", minimum=0.0)
        if tolerance <= 0.0:
            msg = "comparison_tolerance must be strictly positive."
            raise ValueError(msg)
        rationale = require_nonempty_text(self.tolerance_rationale, "tolerance_rationale")
        if self.computed_mean_policy != "all_five_successes_required":
            msg = "computed_mean_policy must be 'all_five_successes_required'."
            raise ValueError(msg)
        comparisons = tuple(self.target_comparisons)
        if not all(isinstance(item, LegacyTargetComparison) for item in comparisons):
            msg = "target_comparisons must contain only LegacyTargetComparison values."
            raise TypeError(msg)
        seeds = tuple(item.target_seed for item in comparisons)
        if seeds != LEGACY_REPRODUCTION_TARGET_SEEDS:
            msg = "Comparisons must contain all five legacy targets exactly once in canonical seed order."
            raise ValueError(msg)
        record_ids = tuple(item.outcome.source_record_id for item in comparisons)
        if len(record_ids) != len(set(record_ids)):
            msg = "Computed outcomes must not reuse a source-record identity."
            raise ValueError(msg)
        runtime_fingerprints = tuple(item.outcome.runtime_fingerprint_checksum for item in comparisons)
        if len(runtime_fingerprints) != len(set(runtime_fingerprints)):
            msg = "Computed outcomes must not reuse a target runtime-fingerprint identity."
            raise ValueError(msg)
        for index, comparison in enumerate(comparisons):
            if not _same_float(comparison.tolerance, tolerance) or not _same_float(
                comparison.reference_fidelity, self.archived_reference.fidelities[index]
            ):
                msg = "Each comparison must use the report tolerance and same-seed archived reference."
                raise ValueError(msg)
        complete = all(item.outcome.status == "success" for item in comparisons)
        expected_computed_mean = (
            fmean(cast("float", item.outcome.computed_fidelity) for item in comparisons) if complete else None
        )
        expected_reference_mean = self.archived_reference.reference_mean
        expected_mean_delta = (
            None if expected_computed_mean is None else expected_computed_mean - expected_reference_mean
        )
        expected_absolute_mean_delta = None if expected_mean_delta is None else abs(expected_mean_delta)
        expected_classification = (
            "reproduced"
            if complete
            and all(item.within_tolerance for item in comparisons)
            and cast("float", expected_absolute_mean_delta) <= tolerance
            else "discrepant"
        )
        if (
            not _same_float(self.computed_mean, expected_computed_mean)
            or not _same_float(self.reference_mean, expected_reference_mean)
            or not _same_float(self.mean_delta, expected_mean_delta)
            or not _same_float(self.absolute_mean_delta, expected_absolute_mean_delta)
            or self.classification != expected_classification
        ):
            msg = "Report means or classification are not derived from the five supplied outcomes."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "source_manifest_checksum",
            require_checksum(self.source_manifest_checksum, "source_manifest_checksum"),
        )
        object.__setattr__(self, "runtime_checksum", require_checksum(self.runtime_checksum, "runtime_checksum"))
        object.__setattr__(self, "comparison_tolerance", tolerance)
        object.__setattr__(self, "tolerance_rationale", rationale)
        object.__setattr__(self, "target_comparisons", comparisons)

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered report field."""
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "method_id": self.method_id,
            "archived_reference": self.archived_reference.to_dict(),
            "comparison_tolerance": self.comparison_tolerance,
            "tolerance_rationale": self.tolerance_rationale,
            "computed_mean_policy": self.computed_mean_policy,
            "target_comparisons": [item.to_dict() for item in self.target_comparisons],
            "computed_mean": self.computed_mean,
            "reference_mean": self.reference_mean,
            "mean_delta": self.mean_delta,
            "absolute_mean_delta": self.absolute_mean_delta,
            "classification": self.classification,
            "source_manifest_checksum": self.source_manifest_checksum,
            "runtime_checksum": self.runtime_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete five-target comparison."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the sealed report."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> LegacyReproductionReport:
        """Decode and verify a complete five-target report.

        Returns:
            The verified reproduction report.

        Raises:
            TypeError: If the comparison collection is not a sequence.
            ValueError: If the document is invalid, unsupported, or inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_REPORT_KEYS, name="legacy reproduction report")
        if mapping["schema_version"] != LEGACY_REPRODUCTION_REPORT_SCHEMA_VERSION:
            msg = "Legacy reproduction report uses an unsupported schema version."
            raise ValueError(msg)
        raw_comparisons = mapping["target_comparisons"]
        if not isinstance(raw_comparisons, Sequence):
            msg = "target_comparisons must be a sequence."
            raise TypeError(msg)
        result = cls(
            report_id=cast("str", mapping["report_id"]),
            method_id=cast("str", mapping["method_id"]),
            archived_reference=LegacyArchivedReference.from_dict(mapping["archived_reference"]),
            comparison_tolerance=cast("float", mapping["comparison_tolerance"]),
            tolerance_rationale=cast("str", mapping["tolerance_rationale"]),
            computed_mean_policy=cast("str", mapping["computed_mean_policy"]),
            target_comparisons=tuple(LegacyTargetComparison.from_dict(item) for item in raw_comparisons),
            computed_mean=cast("float | None", mapping["computed_mean"]),
            reference_mean=cast("float", mapping["reference_mean"]),
            mean_delta=cast("float | None", mapping["mean_delta"]),
            absolute_mean_delta=cast("float | None", mapping["absolute_mean_delta"]),
            classification=cast("Literal['reproduced', 'discrepant']", mapping["classification"]),
            source_manifest_checksum=cast("str", mapping["source_manifest_checksum"]),
            runtime_checksum=cast("str", mapping["runtime_checksum"]),
        )
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Legacy reproduction report changed during normalization."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> LegacyReproductionReport:
        """Decode a canonical sealed report.

        Returns:
            The verified reproduction report.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def compare_legacy_reproduction(
    outcomes: Sequence[LegacyReproductionOutcome],
    *,
    tolerance: float,
    tolerance_rationale: str,
    report_id: str = "layerwise_bmpd_crn_legacy_v1_reproduction",
    source_manifest_checksum: str,
    runtime_checksum: str,
) -> LegacyReproductionReport:
    """Compare five supplied outcomes with the trusted archived values.

    Args:
        outcomes: Exactly five success/failure outcomes in canonical seed order.
        tolerance: Strictly positive absolute fidelity tolerance.
        tolerance_rationale: Scientific explanation for the numerical tolerance.
        report_id: Stable report identifier.
        source_manifest_checksum: Checksum of the job-level source snapshot.
        runtime_checksum: Checksum of the runtime bound to that snapshot.

    Returns:
        A checksum-sealed report derived solely from ``outcomes`` and the
        trusted archived reference.

    Raises:
        TypeError: If ``outcomes`` is not a sequence of reproduction outcomes.
        ValueError: If the outcome universe, tolerance, or rationale is invalid.
    """
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        msg = "outcomes must be a sequence of LegacyReproductionOutcome values."
        raise TypeError(msg)
    supplied = tuple(outcomes)
    if not all(isinstance(item, LegacyReproductionOutcome) for item in supplied):
        msg = "outcomes must contain only LegacyReproductionOutcome values."
        raise TypeError(msg)
    seeds = tuple(item.target_seed for item in supplied)
    if seeds != LEGACY_REPRODUCTION_TARGET_SEEDS:
        msg = "outcomes must contain all five targets exactly once in canonical seed order."
        raise ValueError(msg)
    normalized_tolerance = require_float(tolerance, "tolerance", minimum=0.0)
    if normalized_tolerance <= 0.0:
        msg = "tolerance must be strictly positive."
        raise ValueError(msg)
    rationale = require_nonempty_text(tolerance_rationale, "tolerance_rationale")
    reference = load_archived_layerwise_reference()
    comparisons = tuple(
        LegacyTargetComparison(
            target_seed=outcome.target_seed,
            outcome=outcome,
            reference_fidelity=reference.fidelities[index],
            delta=(
                None if outcome.computed_fidelity is None else outcome.computed_fidelity - reference.fidelities[index]
            ),
            absolute_delta=(
                None
                if outcome.computed_fidelity is None
                else abs(outcome.computed_fidelity - reference.fidelities[index])
            ),
            tolerance=normalized_tolerance,
            within_tolerance=(
                outcome.computed_fidelity is not None
                and abs(outcome.computed_fidelity - reference.fidelities[index]) <= normalized_tolerance
            ),
        )
        for index, outcome in enumerate(supplied)
    )
    complete = all(outcome.computed_fidelity is not None for outcome in supplied)
    computed_mean = fmean(cast("float", item.computed_fidelity) for item in supplied) if complete else None
    mean_delta = None if computed_mean is None else computed_mean - reference.reference_mean
    absolute_mean_delta = None if mean_delta is None else abs(mean_delta)
    classification: Literal["reproduced", "discrepant"] = (
        "reproduced"
        if complete
        and all(item.within_tolerance for item in comparisons)
        and cast("float", absolute_mean_delta) <= normalized_tolerance
        else "discrepant"
    )
    return LegacyReproductionReport(
        report_id=report_id,
        method_id=LEGACY_LAYERWISE_METHOD_ID,
        archived_reference=reference,
        comparison_tolerance=normalized_tolerance,
        tolerance_rationale=rationale,
        computed_mean_policy="all_five_successes_required",
        target_comparisons=comparisons,
        computed_mean=computed_mean,
        reference_mean=reference.reference_mean,
        mean_delta=mean_delta,
        absolute_mean_delta=absolute_mean_delta,
        classification=classification,
        source_manifest_checksum=source_manifest_checksum,
        runtime_checksum=runtime_checksum,
    )


@dataclass(frozen=True, slots=True, init=False)
class LayerwiseMaterializedCircuit:
    """Decoded deterministic circuit binding and selected parameter vector."""

    circuit_binding: NoisyKrotovCircuitBinding
    selected_parameter_checksum: str
    _selected_parameter_bytes: bytes = field(repr=False)
    schema_version: str = field(default=LAYERWISE_MATERIALIZED_CIRCUIT_SCHEMA_VERSION, init=False)

    def __init__(
        self,
        circuit_binding: NoisyKrotovCircuitBinding,
        selected_parameters: NDArray[np.float64],
    ) -> None:
        """Validate and defensively freeze the materialized circuit operands.

        Raises:
            TypeError: If either operand has an invalid type.
            ValueError: If the parameter shape, bound, or values are invalid.
        """
        if not isinstance(circuit_binding, NoisyKrotovCircuitBinding):
            msg = "circuit_binding must be a NoisyKrotovCircuitBinding."
            raise TypeError(msg)
        if not isinstance(selected_parameters, np.ndarray):
            msg = "selected_parameters must be a NumPy array."
            raise TypeError(msg)
        parameters = np.asarray(selected_parameters, dtype=np.float64)
        expected_count = circuit_binding.circuit.num_params
        if parameters.shape != (expected_count,) or parameters.size > MAX_STAGE_PARAMETER_COUNT:
            msg = f"selected_parameters must have shape ({expected_count},) within the codec bound."
            raise ValueError(msg)
        if not np.all(np.isfinite(parameters)):
            msg = "selected_parameters must contain only finite values."
            raise ValueError(msg)
        payload = _parameter_bytes(parameters)
        object.__setattr__(self, "circuit_binding", circuit_binding)
        object.__setattr__(self, "selected_parameter_checksum", _parameter_checksum(parameters))
        object.__setattr__(self, "_selected_parameter_bytes", payload)
        object.__setattr__(self, "schema_version", LAYERWISE_MATERIALIZED_CIRCUIT_SCHEMA_VERSION)

    @property
    def selected_parameters(self) -> NDArray[np.float64]:
        """Detached selected parameter vector."""
        return np.frombuffer(self._selected_parameter_bytes, dtype=np.dtype("<f8")).astype(np.float64, copy=True)

    def _content_dict(self) -> dict[str, object]:
        """Return the complete deterministic payload document."""
        return {
            "schema_version": self.schema_version,
            "circuit_binding": self.circuit_binding.to_dict(),
            "selected_parameters": {
                "data_base64": base64.b64encode(self._selected_parameter_bytes).decode("ascii"),
                "dtype": "<f8",
                "shape": [self.circuit_binding.circuit.num_params],
            },
            "selected_parameter_checksum": self.selected_parameter_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the binding and selected parameters."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return the sealed materialized circuit document."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_bytes(self) -> bytes:
        """Encode the materialized circuit as bounded canonical UTF-8 JSON.

        Returns:
            The deterministic canonical payload.

        Raises:
            ValueError: If the encoded payload exceeds the codec size bound.
        """
        payload = canonical_json(self.to_dict()).encode("utf-8")
        if len(payload) > MAX_LAYERWISE_MATERIALIZED_CIRCUIT_BYTES:
            msg = "Layerwise materialized-circuit payload exceeds the codec size bound."
            raise ValueError(msg)
        return payload

    @property
    def payload_checksum(self) -> str:
        """Checksum of the exact deterministic encoded bytes."""
        return f"sha256:{hashlib.sha256(self.to_bytes()).hexdigest()}"

    @classmethod
    def from_bytes(cls, payload: bytes) -> LayerwiseMaterializedCircuit:
        """Decode and verify bounded canonical materialized-circuit bytes.

        Returns:
            The verified materialized circuit.

        Raises:
            TypeError: If ``payload`` is not immutable bytes.
            ValueError: If the payload is malformed, oversized, or inconsistent.
        """
        if type(payload) is not bytes:
            msg = "payload must be immutable bytes."
            raise TypeError(msg)
        if not payload or len(payload) > MAX_LAYERWISE_MATERIALIZED_CIRCUIT_BYTES:
            msg = "payload must be nonempty and within the materialized-circuit size bound."
            raise ValueError(msg)
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            msg = "payload must contain canonical UTF-8 JSON."
            raise ValueError(msg) from error
        mapping = verify_sealed_mapping(
            load_canonical_json_object(text),
            expected_keys=_MATERIALIZED_CIRCUIT_KEYS,
            name="layerwise materialized circuit",
        )
        if mapping["schema_version"] != LAYERWISE_MATERIALIZED_CIRCUIT_SCHEMA_VERSION:
            msg = "Layerwise materialized circuit uses an unsupported schema version."
            raise ValueError(msg)
        binding = decode_noisy_krotov_circuit_binding_document(mapping["circuit_binding"])
        parameter_payload = require_mapping(mapping["selected_parameters"], "selected_parameters")
        require_exact_keys(parameter_payload, _PARAMETER_PAYLOAD_KEYS, "selected_parameters")
        if parameter_payload["dtype"] != "<f8":
            msg = "selected_parameters.dtype must be '<f8'."
            raise ValueError(msg)
        shape = parameter_payload["shape"]
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) != 1:
            msg = "selected_parameters.shape must contain exactly one dimension."
            raise ValueError(msg)
        count = require_int(shape[0], "selected_parameters.shape[0]", minimum=1)
        expected_count = binding.circuit.num_params
        if count != expected_count or count > MAX_STAGE_PARAMETER_COUNT:
            msg = "selected_parameters.shape does not match the decoded circuit or codec bound."
            raise ValueError(msg)
        encoded = require_string(parameter_payload["data_base64"], "selected_parameters.data_base64")
        try:
            parameter_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as error:
            msg = "selected_parameters.data_base64 is not canonical base64."
            raise ValueError(msg) from error
        if len(parameter_bytes) != count * np.dtype("<f8").itemsize:
            msg = "selected parameter bytes do not match the declared shape."
            raise ValueError(msg)
        parameters = np.frombuffer(parameter_bytes, dtype=np.dtype("<f8")).astype(np.float64, copy=True)
        result = cls(binding, parameters)
        if (
            mapping["selected_parameter_checksum"] != result.selected_parameter_checksum
            or mapping["content_checksum"] != result.content_checksum
            or canonical_json(thaw_json(mapping)).encode("utf-8") != payload
        ):
            msg = "Layerwise materialized circuit failed checksum or canonical-byte verification."
            raise ValueError(msg)
        return result


def encode_layerwise_materialized_circuit(
    circuit_binding: NoisyKrotovCircuitBinding,
    selected_parameters: NDArray[np.float64],
) -> bytes:
    """Encode one deterministic layerwise materialized circuit.

    Returns:
        The bounded canonical payload.
    """
    return LayerwiseMaterializedCircuit(circuit_binding, selected_parameters).to_bytes()


def decode_layerwise_materialized_circuit(payload: bytes) -> LayerwiseMaterializedCircuit:
    """Decode one deterministic layerwise materialized circuit.

    Returns:
        The verified materialized circuit.
    """
    return LayerwiseMaterializedCircuit.from_bytes(payload)


__all__ = [
    "LAYERWISE_MATERIALIZED_CIRCUIT_SCHEMA_VERSION",
    "LEGACY_ARCHIVED_REFERENCE_SCHEMA_VERSION",
    "LEGACY_LAYERWISE_METHOD_ID",
    "LEGACY_REPRODUCTION_OUTCOME_SCHEMA_VERSION",
    "LEGACY_REPRODUCTION_REPORT_SCHEMA_VERSION",
    "LEGACY_REPRODUCTION_TARGET_SEEDS",
    "LEGACY_TARGET_COMPARISON_SCHEMA_VERSION",
    "MAX_LAYERWISE_MATERIALIZED_CIRCUIT_BYTES",
    "LayerwiseMaterializedCircuit",
    "LegacyArchivedReference",
    "LegacyReproductionOutcome",
    "LegacyReproductionReport",
    "LegacyTargetComparison",
    "compare_legacy_reproduction",
    "decode_layerwise_materialized_circuit",
    "encode_layerwise_materialized_circuit",
    "load_archived_layerwise_reference",
]
