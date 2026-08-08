# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Pre-seal custody records for the locked WP23 confirmatory study.

This module contains no execution or file-system integration.  It closes the
prospective confirmatory universe over the exact pilot, screening, target,
source, outcome, and production-custody artifacts that already exist.  In
particular, novelty is checked from seed-bearing manifests before target
materialization and an outer failure is admitted only with authenticated
first-attempt production custody.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from benchmarks.state_preparation.constants import (
    SUPPORTED_QUBIT_COUNTS,
    TARGET_GENERATION_SEEDS,
    TARGET_IDS,
)

from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .execution_bindings import PILOT_METHOD_IDS, SCREEN_METHOD_IDS
from .execution_context import ConfirmationExecutionContext
from .legacy import TRUSTED_LEGACY_AUDIT_CHECKSUM, LegacyEvidenceAudit
from .legacy_targets import LEGACY_TARGET_QUBIT_COUNT, LEGACY_TARGET_SEEDS
from .pipeline import LEGACY_REPRODUCTION_TARGET_IDS, PHASE1_FIXTURE_MANIFEST_CHECKSUM
from .production_executors import (
    ConfirmationResourceLimitProof,
    ProductionNumericalEvidence,
    ReopenedProductionResult,
    ResultArtifactRef,
)
from .protocol import (
    FinalConfigurationExecutionManifest,
    FinalConfirmationSeal,
    InitialPreregistration,
    ScreeningManifest,
)
from .result_custody import PRODUCTION_RESULT_CUSTODY_SCHEMA_VERSION, ProductionResultCustody
from .screening import ProductionResourceCalibration
from .targets import TargetInstanceSpec, TargetPopulationManifest
from .training_orchestration import (
    PILOT_OPTIMIZATION_SEED_COUNT,
    ConfirmExecutionRequest,
    TrainingJob,
    TrainingJobOutcome,
    TrainingRunPlan,
    build_paper_confirm_plan,
    confirmatory_evaluation_policy_checksum,
)
from .validation import (
    require_checksum,
    require_int,
    require_nonempty_text,
    require_slug,
)
from .wp20_resources import EventLevelTestCoupling, PairedBlockIdentity

PRIOR_TARGET_EXPOSURE_RECORD_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22g_prior_target_exposure_record.v1"
PRIOR_TARGET_EXPOSURE_INVENTORY_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.wp22g_prior_target_exposure_inventory.v1"
)
CONFIRMATORY_PAIRABILITY_RECORD_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.wp22g_confirmatory_pairability_record.v1"
)
LOCKED_CONFIRMATORY_STUDY_ROW_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22g_locked_confirmatory_study_row.v1"
LOCKED_CONFIRMATORY_STUDY_MANIFEST_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.wp22g_locked_confirmatory_study_manifest.v1"
)

_ACTUAL_STREAM_MODE = "independent"
_INDEPENDENT_STREAM_REASON = "configuration_specific_evaluation_seeds"

_EXPOSURE_RECORD_KEYS = frozenset({
    "schema_version",
    "source_kind",
    "source_artifact_checksum",
    "target_instance_id",
    "instance_seed",
    "qubit_count",
    "content_checksum",
})
_EXPOSURE_INVENTORY_KEYS = frozenset({
    "schema_version",
    "preregistration_checksum",
    "legacy_evidence_audit_checksum",
    "phase_i_fixture_manifest_checksum",
    "pilot_plan",
    "pilot_primary_q6_target_manifest",
    "pilot_secondary_q12_target_manifest",
    "screening_plan",
    "screening_target_manifest",
    "screening_manifest",
    "resource_calibration_checksum",
    "resource_calibration_execution_source_checksum",
    "pilot_custody_checksum",
    "pilot_calibration_checksum",
    "screening_custody_checksum",
    "canonical_legacy_exposures",
    "content_checksum",
})
_PAIRABILITY_KEYS = frozenset({
    "schema_version",
    "contrast_id",
    "treatment_configuration_checksum",
    "control_configuration_checksum",
    "treatment_job_checksum",
    "control_job_checksum",
    "treatment_request_checksum",
    "control_request_checksum",
    "treatment_evaluation_seed",
    "control_evaluation_seed",
    "paired_block",
    "actual_stream_mode",
    "actual_stream_reason",
    "successful_resource_pair_checksum",
    "event_level_test_coupling",
    "content_checksum",
})
_STUDY_ROW_KEYS = frozenset({
    "schema_version",
    "job_checksum",
    "job_id",
    "request_checksum",
    "configuration_checksum",
    "method_id",
    "target_instance_id",
    "target_spec_checksum",
    "family_id",
    "stratum_id",
    "optimization_block_id",
    "optimization_seed_index",
    "optimization_seed",
    "evaluation_seed",
    "fixed_test_trajectory_count",
    "terminal_state",
    "outcome",
    "outer_outcome_checksum",
    "production_result_reference",
    "production_evidence",
    "production_custody_checksum",
    "raw_trajectory_document_checksum",
    "resource_document_checksum",
    "pilot_diagnostic_checksums",
    "partial_artifact_root",
    "observed_test_trajectory_count",
    "output_inventory_root",
    "content_checksum",
})
_STUDY_MANIFEST_KEYS = frozenset({
    "schema_version",
    "plan",
    "final_seal",
    "configuration_execution_manifest",
    "target_manifest",
    "exposure_inventory",
    "execution_source_manifest_checksum",
    "analysis_source_manifest_checksum",
    "analysis_template_checksum",
    "rows",
    "pairability_records",
    "status",
    "planned_job_count",
    "terminal_job_count",
    "successful_job_count",
    "failed_job_count",
    "unattempted_job_count",
    "planned_test_trajectory_count",
    "observed_test_trajectory_count",
    "output_inventory_root",
    "content_checksum",
})


def _strict_sequence(value: object, name: str) -> tuple[object, ...]:
    """Return a canonical-decoded JSON sequence.

    Returns:
        The immutable decoded sequence.

    Raises:
        TypeError: If ``value`` is not a tuple.
    """
    if not isinstance(value, tuple):
        msg = f"{name} must be a JSON array."
        raise TypeError(msg)
    return value


def _require_optional_checksum(value: str | None, name: str) -> str | None:
    """Validate an optional checksum.

    Returns:
        The normalized checksum or ``None``.
    """
    return None if value is None else require_checksum(value, name)


def _set_frozen(instance: object, name: str, value: object) -> None:
    """Set one field while constructing or normalizing a frozen custody record."""
    object.__setattr__(instance, name, value)  # noqa: PLC2801 -- required for frozen dataclass custody


@dataclass(frozen=True, slots=True)
class PriorTargetExposureRecord:
    """One canonical Phase-I or historical target identifier and seed."""

    source_kind: Literal["phase_i_fixture", "legacy_reproduction"]
    source_artifact_checksum: str
    target_instance_id: str
    instance_seed: str | None
    qubit_count: int
    schema_version: str = field(default=PRIOR_TARGET_EXPOSURE_RECORD_SCHEMA_VERSION, init=False)
    _content_checksum: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate one explicitly typed legacy exposure.

        Raises:
            ValueError: If a source identity, seed, or qubit count is invalid.
        """
        if self.source_kind not in {"phase_i_fixture", "legacy_reproduction"}:
            msg = "source_kind must identify a canonical Phase-I or legacy-reproduction source."
            raise ValueError(msg)
        _set_frozen(
            self,
            "source_artifact_checksum",
            require_checksum(self.source_artifact_checksum, "source_artifact_checksum"),
        )
        _set_frozen(self, "target_instance_id", require_slug(self.target_instance_id, "target_instance_id"))
        if self.instance_seed is not None:
            _set_frozen(self, "instance_seed", require_nonempty_text(self.instance_seed, "instance_seed"))
        _set_frozen(self, "qubit_count", require_int(self.qubit_count, "qubit_count", minimum=1))
        _set_frozen(self, "_content_checksum", canonical_checksum(self._content_dict()))

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_kind": self.source_kind,
            "source_artifact_checksum": self.source_artifact_checksum,
            "target_instance_id": self.target_instance_id,
            "instance_seed": self.instance_seed,
            "qubit_count": self.qubit_count,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of this exact prior exposure."""
        return self._content_checksum

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> PriorTargetExposureRecord:
        """Decode and verify one canonical prior exposure.

        Returns:
            The verified prior exposure record.

        Raises:
            ValueError: If the schema or checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_EXPOSURE_RECORD_KEYS, name="prior target exposure")
        if mapping["schema_version"] != PRIOR_TARGET_EXPOSURE_RECORD_SCHEMA_VERSION:
            msg = "Prior target exposure uses an unsupported schema version."
            raise ValueError(msg)
        result = cls(
            source_kind=cast("Literal['phase_i_fixture', 'legacy_reproduction']", mapping["source_kind"]),
            source_artifact_checksum=cast("str", mapping["source_artifact_checksum"]),
            target_instance_id=cast("str", mapping["target_instance_id"]),
            instance_seed=cast("str | None", mapping["instance_seed"]),
            qubit_count=cast("int", mapping["qubit_count"]),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Prior target exposure checksum changed during normalization."
            raise ValueError(msg)
        return result


def _canonical_legacy_exposures(legacy_audit_checksum: str) -> tuple[PriorTargetExposureRecord, ...]:
    """Return the closed Phase-I and five-target historical exposure universe."""
    phase_i = tuple(
        PriorTargetExposureRecord(
            source_kind="phase_i_fixture",
            source_artifact_checksum=PHASE1_FIXTURE_MANIFEST_CHECKSUM,
            target_instance_id=target_id,
            instance_seed=(
                None if TARGET_GENERATION_SEEDS[target_id] is None else str(TARGET_GENERATION_SEEDS[target_id])
            ),
            qubit_count=qubit_count,
        )
        for qubit_count in SUPPORTED_QUBIT_COUNTS
        for target_id in TARGET_IDS
    )
    historical = tuple(
        PriorTargetExposureRecord(
            source_kind="legacy_reproduction",
            source_artifact_checksum=legacy_audit_checksum,
            target_instance_id=target_id,
            instance_seed=str(seed),
            qubit_count=LEGACY_TARGET_QUBIT_COUNT,
        )
        for target_id, seed in zip(LEGACY_REPRODUCTION_TARGET_IDS, LEGACY_TARGET_SEEDS, strict=True)
    )
    return phase_i + historical


def _target_index(manifest: TargetPopulationManifest) -> dict[str, TargetInstanceSpec]:
    """Return a target identifier index without materializing target vectors."""
    return {spec.target_instance_id: spec for spec in manifest.instances}


def _validate_pilot_closure(
    plan: TrainingRunPlan,
    primary_q6: TargetPopulationManifest,
    secondary_q12: TargetPopulationManifest,
) -> None:
    """Validate exact paper-pilot roles, counts, roots, and job closure.

    Raises:
        ValueError: If any plan, manifest, role, count, or cell differs.
    """
    if (
        plan.plan_id != "wp22_paper_pilot_v1"
        or plan.preset != "paper-pilot"
        or len(plan.jobs) != 1_080
        or primary_q6.data_role != "development"
        or primary_q6.population_scope != "primary_q6"
        or secondary_q12.data_role != "screening_selection"
        or secondary_q12.population_scope != "secondary_q12"
        or len(primary_q6.instances) != 48
        or len(secondary_q12.instances) != 24
        or plan.target_manifest_checksums != (primary_q6.content_checksum, secondary_q12.content_checksum)
        or plan.screening_manifest_checksum is not None
        or plan.final_confirmation_seal_checksum is not None
    ):
        msg = "Prior exposure inventory requires the exact 1,080-job paper-pilot plan and q6/q12 manifests."
        raise ValueError(msg)
    targets = {**_target_index(primary_q6), **_target_index(secondary_q12)}
    expected_methods = set(PILOT_METHOD_IDS)
    primary_jobs = tuple(job for job in plan.jobs if job.target_manifest_checksum == primary_q6.content_checksum)
    secondary_jobs = tuple(job for job in plan.jobs if job.target_manifest_checksum == secondary_q12.content_checksum)
    if len(primary_jobs) != 720 or len(secondary_jobs) != 360:
        msg = "Paper-pilot closure requires 720 primary-q6 and 360 secondary-q12 jobs."
        raise ValueError(msg)
    by_target: dict[str, list[TrainingJob]] = {target_id: [] for target_id in targets}
    for job in plan.jobs:
        target = targets.get(job.target_instance_id)
        if target is None:
            msg = "A paper-pilot job references a target outside its exact manifests."
            raise ValueError(msg)
        expected_role = (
            "development" if job.target_manifest_checksum == primary_q6.content_checksum else "secondary_benchmark"
        )
        if (
            job.data_role != expected_role
            or job.target_spec_checksum != target.content_checksum
            or job.family_id != target.family_id
            or job.stratum_id != target.stratum_id
            or job.qubit_count != target.qubit_count
            or job.method_id not in expected_methods
        ):
            msg = "A paper-pilot job is not closed over its exact target, role, or method."
            raise ValueError(msg)
        by_target[job.target_instance_id].append(job)
    for jobs in by_target.values():
        configurations = {job.candidate_configuration_checksum for job in jobs}
        seeds = {job.optimization_seed for job in jobs}
        cells = {(job.candidate_configuration_checksum, job.optimization_seed) for job in jobs}
        if (
            {job.method_id for job in jobs} != expected_methods
            or len(configurations) != len(PILOT_METHOD_IDS)
            or len(seeds) != PILOT_OPTIMIZATION_SEED_COUNT
            or len(cells) != len(PILOT_METHOD_IDS) * PILOT_OPTIMIZATION_SEED_COUNT
            or len(jobs) != len(cells)
        ):
            msg = "Paper-pilot jobs do not form the exact method-by-optimization-seed target cells."
            raise ValueError(msg)


def _validate_screening_closure(
    plan: TrainingRunPlan,
    target_manifest: TargetPopulationManifest,
    screening_manifest: ScreeningManifest,
) -> None:
    """Validate exact paper-screen candidate/cell/target job closure.

    Raises:
        ValueError: If any plan, manifest, candidate, cell, or target differs.
    """
    if (
        plan.plan_id != "wp22_paper_screen_v1"
        or plan.preset != "paper-screen"
        or len(plan.jobs) != 1_296
        or target_manifest.data_role != "screening_selection"
        or target_manifest.population_scope != "primary_q6"
        or len(target_manifest.instances) != 48
        or plan.target_manifest_checksums != (target_manifest.content_checksum,)
        or plan.screening_manifest_checksum != screening_manifest.content_checksum
        or screening_manifest.screening_target_manifest_checksum != target_manifest.content_checksum
        or len(screening_manifest.candidates) != len(SCREEN_METHOD_IDS)
        or len(screening_manifest.cells) != 144
    ):
        msg = "Prior exposure inventory requires the exact 1,296-job paper-screen plan and target universe."
        raise ValueError(msg)
    candidates = {item.configuration_checksum: item for item in screening_manifest.candidates}
    cells = {item.cell_id: item for item in screening_manifest.cells}
    targets = _target_index(target_manifest)
    expected = {(configuration_checksum, cell_id) for configuration_checksum in candidates for cell_id in cells}
    actual: set[tuple[str, str]] = set()
    for job in plan.jobs:
        candidate = candidates.get(job.candidate_configuration_checksum)
        cell = cells.get(job.optimization_block_id)
        target = targets.get(job.target_instance_id)
        if target is None:
            msg = "A paper-screen job references a target outside its exact manifest."
            raise ValueError(msg)
        if (
            candidate is None
            or cell is None
            or job.method_id != candidate.method_id
            or job.data_role != "screening_selection"
            or job.target_manifest_checksum != target_manifest.content_checksum
            or job.target_spec_checksum != target.content_checksum
            or job.target_instance_id != cell.target_instance_id
            or job.family_id != cell.family_id
            or job.stratum_id != cell.stratum_id
            or job.qubit_count != cell.qubit_count
            or job.optimization_seed != cell.optimization_seed
            or job.evaluation_seed != cell.screening_seed
        ):
            msg = "A paper-screen job is not closed over its exact candidate, cell, and target."
            raise ValueError(msg)
        actual.add((job.candidate_configuration_checksum, job.optimization_block_id))
    if actual != expected or len(actual) != len(plan.jobs):
        msg = "Paper-screen jobs do not form the exact candidate-by-cell universe."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True, init=False)
class PriorTargetExposureInventory:
    """Exact prior pilot, screen, Phase-I, and legacy target exposure custody."""

    preregistration_checksum: str
    legacy_evidence_audit_checksum: str
    phase_i_fixture_manifest_checksum: str
    pilot_plan: TrainingRunPlan
    pilot_primary_q6_target_manifest: TargetPopulationManifest
    pilot_secondary_q12_target_manifest: TargetPopulationManifest
    screening_plan: TrainingRunPlan
    screening_target_manifest: TargetPopulationManifest
    screening_manifest: ScreeningManifest
    resource_calibration_checksum: str
    resource_calibration_execution_source_checksum: str
    pilot_custody_checksum: str
    pilot_calibration_checksum: str
    screening_custody_checksum: str
    canonical_legacy_exposures: tuple[PriorTargetExposureRecord, ...]
    schema_version: str = field(default=PRIOR_TARGET_EXPOSURE_INVENTORY_SCHEMA_VERSION, init=False)
    _content_checksum: str = field(init=False, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        *,
        preregistration: InitialPreregistration,
        legacy_evidence_audit: LegacyEvidenceAudit,
        pilot_plan: TrainingRunPlan,
        pilot_primary_q6_target_manifest: TargetPopulationManifest,
        pilot_secondary_q12_target_manifest: TargetPopulationManifest,
        screening_plan: TrainingRunPlan,
        screening_target_manifest: TargetPopulationManifest,
        screening_manifest: ScreeningManifest,
        resource_calibration: ProductionResourceCalibration,
    ) -> PriorTargetExposureInventory:
        """Derive a self-contained inventory from exact authenticated sources.

        The resource calibration itself is not copied: its checksum and all
        custody roots needed to reopen it are projected.  The factory accepts
        only the concrete production calibration type, so caller-authored root
        summaries cannot enter this boundary.

        Returns:
            The exact checksum-sealed prior-exposure inventory.

        Raises:
            TypeError: If an input is not its exact authenticated artifact type.
            ValueError: If roots, roles, scopes, counts, or job closure differ.
        """
        typed = (
            (preregistration, InitialPreregistration, "preregistration"),
            (legacy_evidence_audit, LegacyEvidenceAudit, "legacy_evidence_audit"),
            (pilot_plan, TrainingRunPlan, "pilot_plan"),
            (pilot_primary_q6_target_manifest, TargetPopulationManifest, "pilot_primary_q6_target_manifest"),
            (
                pilot_secondary_q12_target_manifest,
                TargetPopulationManifest,
                "pilot_secondary_q12_target_manifest",
            ),
            (screening_plan, TrainingRunPlan, "screening_plan"),
            (screening_target_manifest, TargetPopulationManifest, "screening_target_manifest"),
            (screening_manifest, ScreeningManifest, "screening_manifest"),
        )
        for value, expected, name in typed:
            if not isinstance(value, expected):
                msg = f"{name} must be a {expected.__name__}."
                raise TypeError(msg)
        if type(resource_calibration) is not ProductionResourceCalibration:
            msg = "resource_calibration must be the concrete ProductionResourceCalibration artifact."
            raise TypeError(msg)
        if (
            legacy_evidence_audit.content_checksum != TRUSTED_LEGACY_AUDIT_CHECKSUM
            or preregistration.legacy_evidence_audit_checksum != legacy_evidence_audit.content_checksum
        ):
            msg = "Legacy evidence audit does not match the preregistered trusted audit root."
            raise ValueError(msg)
        _validate_pilot_closure(
            pilot_plan,
            pilot_primary_q6_target_manifest,
            pilot_secondary_q12_target_manifest,
        )
        _validate_screening_closure(screening_plan, screening_target_manifest, screening_manifest)
        if any(
            artifact.preregistration_checksum != preregistration.content_checksum
            for artifact in (
                pilot_plan,
                pilot_primary_q6_target_manifest,
                pilot_secondary_q12_target_manifest,
                screening_plan,
                screening_target_manifest,
                screening_manifest,
            )
        ):
            msg = "A prior pilot or screening artifact belongs to another preregistration."
            raise ValueError(msg)
        if (
            resource_calibration.preregistration_checksum != preregistration.content_checksum
            or resource_calibration.pilot_plan_checksum != pilot_plan.content_checksum
            or resource_calibration.screening_plan_checksum != screening_plan.content_checksum
            or resource_calibration.screening_manifest_checksum != screening_manifest.content_checksum
        ):
            msg = "Production resource calibration does not close over the exact pilot and screening roots."
            raise ValueError(msg)
        return cls._build(
            preregistration_checksum=preregistration.content_checksum,
            legacy_evidence_audit_checksum=legacy_evidence_audit.content_checksum,
            phase_i_fixture_manifest_checksum=PHASE1_FIXTURE_MANIFEST_CHECKSUM,
            pilot_plan=pilot_plan,
            pilot_primary_q6_target_manifest=pilot_primary_q6_target_manifest,
            pilot_secondary_q12_target_manifest=pilot_secondary_q12_target_manifest,
            screening_plan=screening_plan,
            screening_target_manifest=screening_target_manifest,
            screening_manifest=screening_manifest,
            resource_calibration_checksum=resource_calibration.content_checksum,
            resource_calibration_execution_source_checksum=resource_calibration.execution_source_manifest_checksum,
            pilot_custody_checksum=resource_calibration.pilot_custody_checksum,
            pilot_calibration_checksum=resource_calibration.pilot_calibration_checksum,
            screening_custody_checksum=resource_calibration.screening_custody_checksum,
            canonical_legacy_exposures=_canonical_legacy_exposures(legacy_evidence_audit.content_checksum),
        )

    @classmethod
    def _build(cls, **values: object) -> PriorTargetExposureInventory:
        inventory = object.__new__(cls)
        for name, value in values.items():
            if name == "_content_checksum":
                continue
            _set_frozen(inventory, name, value)
        _set_frozen(inventory, "schema_version", PRIOR_TARGET_EXPOSURE_INVENTORY_SCHEMA_VERSION)
        inventory._validate()  # noqa: SLF001 -- class-owned invariant validation
        _set_frozen(
            inventory,
            "_content_checksum",
            canonical_checksum(inventory._content_dict()),  # noqa: SLF001 -- class-owned checksum derivation
        )
        return inventory

    def _validate(self) -> None:
        """Revalidate all embedded plan, manifest, and canonical-source closure.

        Raises:
            ValueError: If any checksum, exact artifact universe, or novelty source differs.
        """
        for name in (
            "preregistration_checksum",
            "legacy_evidence_audit_checksum",
            "phase_i_fixture_manifest_checksum",
            "resource_calibration_checksum",
            "resource_calibration_execution_source_checksum",
            "pilot_custody_checksum",
            "pilot_calibration_checksum",
            "screening_custody_checksum",
        ):
            _set_frozen(self, name, require_checksum(getattr(self, name), name))
        if self.legacy_evidence_audit_checksum != TRUSTED_LEGACY_AUDIT_CHECKSUM:
            msg = "Prior exposure inventory must retain the trusted legacy-audit root."
            raise ValueError(msg)
        if self.phase_i_fixture_manifest_checksum != PHASE1_FIXTURE_MANIFEST_CHECKSUM:
            msg = "Prior exposure inventory must retain the canonical Phase-I fixture root."
            raise ValueError(msg)
        _validate_pilot_closure(
            self.pilot_plan,
            self.pilot_primary_q6_target_manifest,
            self.pilot_secondary_q12_target_manifest,
        )
        _validate_screening_closure(self.screening_plan, self.screening_target_manifest, self.screening_manifest)
        embedded = (
            self.pilot_plan,
            self.pilot_primary_q6_target_manifest,
            self.pilot_secondary_q12_target_manifest,
            self.screening_plan,
            self.screening_target_manifest,
            self.screening_manifest,
        )
        if any(item.preregistration_checksum != self.preregistration_checksum for item in embedded):
            msg = "Embedded exposure artifacts differ from the inventory preregistration root."
            raise ValueError(msg)
        exposures = tuple(self.canonical_legacy_exposures)
        expected = _canonical_legacy_exposures(self.legacy_evidence_audit_checksum)
        if exposures != expected:
            msg = "Canonical Phase-I and legacy exposures are missing, duplicated, reordered, or changed."
            raise ValueError(msg)
        phase_i_count = sum(item.source_kind == "phase_i_fixture" for item in exposures)
        historical_count = sum(item.source_kind == "legacy_reproduction" for item in exposures)
        if phase_i_count != 18 or historical_count != 5:
            msg = "Canonical prior exposure inventory requires exactly 18 Phase-I and five legacy records."
            raise ValueError(msg)
        phase2_specs = tuple(
            spec
            for manifest in (
                self.pilot_primary_q6_target_manifest,
                self.pilot_secondary_q12_target_manifest,
                self.screening_target_manifest,
            )
            for spec in manifest.instances
        )
        if len({spec.target_instance_id for spec in phase2_specs}) != len(phase2_specs) or len({
            spec.instance_seed for spec in phase2_specs
        }) != len(phase2_specs):
            msg = "Pilot and screening Phase-II targets must not reuse an identifier or instance seed."
            raise ValueError(msg)

    @property
    def exposed_target_instance_ids(self) -> frozenset[str]:
        """Every exact prior target identifier without target vectors."""
        phase2 = (
            self.pilot_primary_q6_target_manifest.instances
            + self.pilot_secondary_q12_target_manifest.instances
            + self.screening_target_manifest.instances
        )
        return frozenset((
            *[spec.target_instance_id for spec in phase2],
            *[item.target_instance_id for item in self.canonical_legacy_exposures],
        ))

    @property
    def exposed_instance_seeds(self) -> frozenset[str]:
        """Every exact available prior target-generation seed."""
        phase2 = (
            self.pilot_primary_q6_target_manifest.instances
            + self.pilot_secondary_q12_target_manifest.instances
            + self.screening_target_manifest.instances
        )
        return frozenset((
            *[spec.instance_seed for spec in phase2],
            *[item.instance_seed for item in self.canonical_legacy_exposures if item.instance_seed is not None],
        ))

    def validate_confirmatory_novelty(self, confirmatory_manifest: TargetPopulationManifest) -> None:
        """Reject an exposed confirmatory identifier or instance seed before materialization.

        Raises:
            TypeError: If the prospective target manifest has the wrong type.
            ValueError: If its role/scope/root is wrong or a public identifier
                or seed was previously exposed.
        """
        if not isinstance(confirmatory_manifest, TargetPopulationManifest):
            msg = "confirmatory_manifest must be a TargetPopulationManifest."
            raise TypeError(msg)
        if (
            confirmatory_manifest.preregistration_checksum != self.preregistration_checksum
            or confirmatory_manifest.data_role != "confirmatory"
            or confirmatory_manifest.population_scope != "primary_q6"
        ):
            msg = "Novelty validation requires the preregistered confirmatory primary-q6 manifest."
            raise ValueError(msg)
        reused_ids = sorted(
            spec.target_instance_id
            for spec in confirmatory_manifest.instances
            if spec.target_instance_id in self.exposed_target_instance_ids
        )
        if reused_ids:
            msg = f"Confirmatory manifest reuses exposed target_instance_id values: {reused_ids!r}."
            raise ValueError(msg)
        reused_seeds = sorted(
            spec.instance_seed
            for spec in confirmatory_manifest.instances
            if spec.instance_seed in self.exposed_instance_seeds
        )
        if reused_seeds:
            msg = f"Confirmatory manifest reuses exposed instance_seed values: {reused_seeds!r}."
            raise ValueError(msg)

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "preregistration_checksum": self.preregistration_checksum,
            "legacy_evidence_audit_checksum": self.legacy_evidence_audit_checksum,
            "phase_i_fixture_manifest_checksum": self.phase_i_fixture_manifest_checksum,
            "pilot_plan": self.pilot_plan.to_dict(),
            "pilot_primary_q6_target_manifest": self.pilot_primary_q6_target_manifest.to_dict(),
            "pilot_secondary_q12_target_manifest": self.pilot_secondary_q12_target_manifest.to_dict(),
            "screening_plan": self.screening_plan.to_dict(),
            "screening_target_manifest": self.screening_target_manifest.to_dict(),
            "screening_manifest": self.screening_manifest.to_dict(),
            "resource_calibration_checksum": self.resource_calibration_checksum,
            "resource_calibration_execution_source_checksum": self.resource_calibration_execution_source_checksum,
            "pilot_custody_checksum": self.pilot_custody_checksum,
            "pilot_calibration_checksum": self.pilot_calibration_checksum,
            "screening_custody_checksum": self.screening_custody_checksum,
            "canonical_legacy_exposures": [item.to_dict() for item in self.canonical_legacy_exposures],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact prior-exposure closure."""
        return self._content_checksum

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native inventory data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> PriorTargetExposureInventory:
        """Decode and verify one self-contained prior-exposure inventory.

        Returns:
            The verified inventory.

        Raises:
            ValueError: If the schema, embedded artifacts, or checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_EXPOSURE_INVENTORY_KEYS, name="prior exposure inventory")
        if mapping["schema_version"] != PRIOR_TARGET_EXPOSURE_INVENTORY_SCHEMA_VERSION:
            msg = "Prior exposure inventory uses an unsupported schema version."
            raise ValueError(msg)
        result = cls._build(
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            legacy_evidence_audit_checksum=cast("str", mapping["legacy_evidence_audit_checksum"]),
            phase_i_fixture_manifest_checksum=cast("str", mapping["phase_i_fixture_manifest_checksum"]),
            pilot_plan=TrainingRunPlan.from_dict(mapping["pilot_plan"]),
            pilot_primary_q6_target_manifest=TargetPopulationManifest.from_dict(
                mapping["pilot_primary_q6_target_manifest"]
            ),
            pilot_secondary_q12_target_manifest=TargetPopulationManifest.from_dict(
                mapping["pilot_secondary_q12_target_manifest"]
            ),
            screening_plan=TrainingRunPlan.from_dict(mapping["screening_plan"]),
            screening_target_manifest=TargetPopulationManifest.from_dict(mapping["screening_target_manifest"]),
            screening_manifest=ScreeningManifest.from_dict(mapping["screening_manifest"]),
            resource_calibration_checksum=cast("str", mapping["resource_calibration_checksum"]),
            resource_calibration_execution_source_checksum=cast(
                "str", mapping["resource_calibration_execution_source_checksum"]
            ),
            pilot_custody_checksum=cast("str", mapping["pilot_custody_checksum"]),
            pilot_calibration_checksum=cast("str", mapping["pilot_calibration_checksum"]),
            screening_custody_checksum=cast("str", mapping["screening_custody_checksum"]),
            canonical_legacy_exposures=tuple(
                PriorTargetExposureRecord.from_dict(item)
                for item in _strict_sequence(mapping["canonical_legacy_exposures"], "canonical_legacy_exposures")
            ),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Prior exposure inventory checksum changed during normalization."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> PriorTargetExposureInventory:
        """Decode a prior-exposure inventory from canonical JSON.

        Returns:
            The verified inventory.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def validate_confirmatory_novelty(
    inventory: PriorTargetExposureInventory,
    confirmatory_manifest: TargetPopulationManifest,
) -> None:
    """Validate prospective target novelty through the explicit module API.

    Raises:
        TypeError: If ``inventory`` is not a prior-exposure inventory.
    """
    if not isinstance(inventory, PriorTargetExposureInventory):
        msg = "inventory must be a PriorTargetExposureInventory."
        raise TypeError(msg)
    inventory.validate_confirmatory_novelty(confirmatory_manifest)


def _paired_test_protocol_checksum(
    seal: FinalConfirmationSeal,
    contrast_id: str,
) -> str:
    """Checksum pairing-invariant final-test policy fields.

    Returns:
        The checksum of the sealed contrast-wide final-test policy.
    """
    contrast = next(item for item in seal.primary_contrasts if item.contrast_id == contrast_id)
    return canonical_checksum({
        "schema_version": "yaqs.state_preparation.phase2.wp22g_paired_test_protocol.v1",
        "final_confirmation_seal_checksum": seal.content_checksum,
        "paired_block_policy_checksum": contrast.paired_block_policy_checksum,
        "fixed_test_trajectory_count": seal.fixed_test_trajectory_count,
        "primary_noise_condition": seal.primary_noise_condition,
        "primary_resource_budget": seal.primary_resource_budget,
        "failure_policy_checksum": seal.failure_policy_checksum,
    })


@dataclass(frozen=True, slots=True, init=False)
class ConfirmatoryPairabilityRecord:
    """Actual stream custody for one contrast/target/optimization block.

    Configuration-specific evaluation seeds make the actual streams
    independent.  An optional WP20 record is only counterfactual resource
    eligibility evidence; it never changes ``actual_stream_mode``.
    """

    contrast_id: str
    treatment_configuration_checksum: str
    control_configuration_checksum: str
    treatment_job_checksum: str
    control_job_checksum: str
    treatment_request_checksum: str
    control_request_checksum: str
    treatment_evaluation_seed: int
    control_evaluation_seed: int
    paired_block: PairedBlockIdentity
    actual_stream_mode: Literal["independent"]
    actual_stream_reason: Literal["configuration_specific_evaluation_seeds"]
    successful_resource_pair_checksum: str | None
    event_level_test_coupling: EventLevelTestCoupling | None
    schema_version: str = field(default=CONFIRMATORY_PAIRABILITY_RECORD_SCHEMA_VERSION, init=False)
    _content_checksum: str = field(init=False, repr=False, compare=False)

    @classmethod
    def _build(cls, **values: object) -> ConfirmatoryPairabilityRecord:
        record = object.__new__(cls)
        for name, value in values.items():
            if name == "_content_checksum":
                continue
            _set_frozen(record, name, value)
        _set_frozen(record, "schema_version", CONFIRMATORY_PAIRABILITY_RECORD_SCHEMA_VERSION)
        record._validate()  # noqa: SLF001 -- class-owned invariant validation
        _set_frozen(
            record,
            "_content_checksum",
            canonical_checksum(record._content_dict()),  # noqa: SLF001 -- class-owned checksum derivation
        )
        return record

    def _validate(self) -> None:
        """Validate identities and preserve independent actual stream semantics.

        Raises:
            TypeError: If paired-block or WP20 evidence uses the wrong type.
            ValueError: If identities or actual independent-stream semantics differ.
        """
        _set_frozen(self, "contrast_id", require_slug(self.contrast_id, "contrast_id"))
        for name in (
            "treatment_configuration_checksum",
            "control_configuration_checksum",
            "treatment_job_checksum",
            "control_job_checksum",
            "treatment_request_checksum",
            "control_request_checksum",
        ):
            _set_frozen(self, name, require_checksum(getattr(self, name), name))
        if self.treatment_configuration_checksum == self.control_configuration_checksum:
            msg = "A confirmatory pair must compare distinct configurations."
            raise ValueError(msg)
        treatment_seed = require_int(self.treatment_evaluation_seed, "treatment_evaluation_seed", minimum=0)
        control_seed = require_int(self.control_evaluation_seed, "control_evaluation_seed", minimum=0)
        _set_frozen(self, "treatment_evaluation_seed", treatment_seed)
        _set_frozen(self, "control_evaluation_seed", control_seed)
        if treatment_seed == control_seed:
            msg = "Current confirmatory configuration streams must use distinct evaluation seeds."
            raise ValueError(msg)
        if not isinstance(self.paired_block, PairedBlockIdentity):
            msg = "paired_block must be a PairedBlockIdentity."
            raise TypeError(msg)
        if self.actual_stream_mode != _ACTUAL_STREAM_MODE or self.actual_stream_reason != _INDEPENDENT_STREAM_REASON:
            msg = "Actual confirmatory streams must record configuration-specific independent sampling."
            raise ValueError(msg)
        _set_frozen(
            self,
            "successful_resource_pair_checksum",
            _require_optional_checksum(
                self.successful_resource_pair_checksum,
                "successful_resource_pair_checksum",
            ),
        )
        coupling = self.event_level_test_coupling
        if coupling is not None:
            if not isinstance(coupling, EventLevelTestCoupling):
                msg = "event_level_test_coupling must be a WP20 EventLevelTestCoupling."
                raise TypeError(msg)
            if coupling.paired_block != self.paired_block or self.successful_resource_pair_checksum is None:
                msg = "WP20 eligibility requires the same paired block and successful resource evidence."
                raise ValueError(msg)

    @property
    def sort_key(self) -> tuple[object, ...]:
        """Canonical contrast, target, and optimization ordering."""
        return (
            self.contrast_id,
            self.paired_block.target_instance_id,
            self.paired_block.optimization_seed,
            self.treatment_configuration_checksum,
            self.control_configuration_checksum,
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contrast_id": self.contrast_id,
            "treatment_configuration_checksum": self.treatment_configuration_checksum,
            "control_configuration_checksum": self.control_configuration_checksum,
            "treatment_job_checksum": self.treatment_job_checksum,
            "control_job_checksum": self.control_job_checksum,
            "treatment_request_checksum": self.treatment_request_checksum,
            "control_request_checksum": self.control_request_checksum,
            "treatment_evaluation_seed": self.treatment_evaluation_seed,
            "control_evaluation_seed": self.control_evaluation_seed,
            "paired_block": self.paired_block.to_dict(),
            "actual_stream_mode": self.actual_stream_mode,
            "actual_stream_reason": self.actual_stream_reason,
            "successful_resource_pair_checksum": self.successful_resource_pair_checksum,
            "event_level_test_coupling": (
                None if self.event_level_test_coupling is None else self.event_level_test_coupling.to_dict()
            ),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the exact pairability record."""
        return self._content_checksum

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native pairability data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> ConfirmatoryPairabilityRecord:
        """Decode and verify one confirmatory pairability record.

        Returns:
            The verified pairability record.

        Raises:
            ValueError: If the schema, identities, or checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_PAIRABILITY_KEYS, name="confirmatory pairability record")
        if mapping["schema_version"] != CONFIRMATORY_PAIRABILITY_RECORD_SCHEMA_VERSION:
            msg = "Confirmatory pairability record uses an unsupported schema version."
            raise ValueError(msg)
        raw_coupling = mapping["event_level_test_coupling"]
        result = cls._build(
            contrast_id=cast("str", mapping["contrast_id"]),
            treatment_configuration_checksum=cast("str", mapping["treatment_configuration_checksum"]),
            control_configuration_checksum=cast("str", mapping["control_configuration_checksum"]),
            treatment_job_checksum=cast("str", mapping["treatment_job_checksum"]),
            control_job_checksum=cast("str", mapping["control_job_checksum"]),
            treatment_request_checksum=cast("str", mapping["treatment_request_checksum"]),
            control_request_checksum=cast("str", mapping["control_request_checksum"]),
            treatment_evaluation_seed=cast("int", mapping["treatment_evaluation_seed"]),
            control_evaluation_seed=cast("int", mapping["control_evaluation_seed"]),
            paired_block=PairedBlockIdentity.from_dict(mapping["paired_block"]),
            actual_stream_mode=cast("Literal['independent']", mapping["actual_stream_mode"]),
            actual_stream_reason=cast(
                "Literal['configuration_specific_evaluation_seeds']",
                mapping["actual_stream_reason"],
            ),
            successful_resource_pair_checksum=cast("str | None", mapping["successful_resource_pair_checksum"]),
            event_level_test_coupling=(
                None if raw_coupling is None else EventLevelTestCoupling.from_dict(raw_coupling)
            ),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Confirmatory pairability checksum changed during normalization."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True, init=False)
class LockedConfirmatoryStudyRow:
    """One planned confirmatory job with authenticated terminal or unattempted custody."""

    job_checksum: str
    job_id: str
    request_checksum: str
    configuration_checksum: str
    method_id: str
    target_instance_id: str
    target_spec_checksum: str
    family_id: str
    stratum_id: str
    optimization_block_id: str
    optimization_seed_index: int
    optimization_seed: int
    evaluation_seed: int
    fixed_test_trajectory_count: int
    terminal_state: Literal["success", "failure", "unattempted"]
    outcome: TrainingJobOutcome | None
    outer_outcome_checksum: str | None
    production_result_reference: ResultArtifactRef | None
    production_evidence: ProductionNumericalEvidence | None
    production_custody_checksum: str | None
    raw_trajectory_document_checksum: str | None
    resource_document_checksum: str | None
    pilot_diagnostic_checksums: tuple[str, ...]
    partial_artifact_root: str | None
    observed_test_trajectory_count: int
    output_inventory_root: str
    schema_version: str = field(default=LOCKED_CONFIRMATORY_STUDY_ROW_SCHEMA_VERSION, init=False)
    _content_checksum: str = field(init=False, repr=False, compare=False)

    @classmethod
    def from_execution(
        cls,
        context: ConfirmationExecutionContext,
        job: TrainingJob,
        outcome: TrainingJobOutcome | None,
        custody: ProductionResultCustody | None,
    ) -> LockedConfirmatoryStudyRow:
        """Derive one row from a sealed request and reopened first attempt.

        Returns:
            The authenticated terminal row or explicit unattempted row.

        Raises:
            TypeError: If context, job, outcome, or custody uses the wrong type.
            ValueError: If only one terminal artifact is supplied or any source,
                status, target, policy, attempt, or fixed-count identity differs.
        """
        if not isinstance(context, ConfirmationExecutionContext):
            msg = "context must be a ConfirmationExecutionContext."
            raise TypeError(msg)
        if not isinstance(job, TrainingJob) or not any(job is owned for owned in context.plan.jobs):
            msg = "job must be one exact TrainingJob in the confirmation plan."
            raise TypeError(msg)
        request = job.confirm_execution_request
        if not isinstance(request, ConfirmExecutionRequest):
            msg = "Confirmation study rows require a ConfirmExecutionRequest."
            raise TypeError(msg)
        if (outcome is None) != (custody is None):
            msg = "A terminal confirmatory row requires both outer outcome and authenticated production custody."
            raise ValueError(msg)
        if outcome is None:
            return cls._from_parts(job, request, None, None)
        if not isinstance(outcome, TrainingJobOutcome) or not isinstance(custody, ProductionResultCustody):
            msg = "Terminal rows require typed TrainingJobOutcome and ProductionResultCustody values."
            raise TypeError(msg)
        reference = custody.reference
        evidence = custody.production_evidence
        expected_policy = confirmatory_evaluation_policy_checksum(request)
        target_identity = evidence.target_identity
        target_spec_identity = target_identity.get(
            "target_instance_spec_checksum",
            target_identity.get("target_spec_checksum"),
        )
        if (
            outcome.job_checksum != job.content_checksum
            or outcome.attempt != 1
            or reference.attempt != 1
            or evidence.attempt != 1
            or reference.job_checksum != request.content_checksum
            or evidence.job_checksum != request.content_checksum
            or reference.evidence_checksum != evidence.content_checksum
            or custody.result_evidence_checksum != evidence.content_checksum
            or reference.status != outcome.status
            or evidence.status != outcome.status
            or reference.execution_source_manifest_checksum != request.execution_source_checksum
            or evidence.execution_source_manifest_checksum != request.execution_source_checksum
            or evidence.source_fingerprint_checksum != request.execution_source_checksum
            or reference.source_fingerprint_checksum != request.execution_source_checksum
            or evidence.executable_binding_checksum != request.executable_binding_checksum
            or evidence.scheduled_program_checksum != context.scheduled_program_checksum(request)
            or evidence.evaluation_policy_checksum != expected_policy
            or target_identity.get("target_instance_id") != request.target_instance_id
            or target_spec_identity != request.target_spec_checksum
        ):
            msg = "Terminal confirmatory custody differs from its exact first-attempt request or sealed source."
            raise ValueError(msg)
        if outcome.status == "success":
            if outcome.result_artifact_checksum != reference.content_checksum:
                msg = "Successful outer outcome does not address the reopened production result."
                raise ValueError(msg)
            values = custody.trajectory_fidelities
            if values is None or len(values) != request.fixed_test_trajectory_count:
                msg = "Successful confirmatory custody must contain the exact fixed test trajectory count."
                raise ValueError(msg)
        elif custody.raw_trajectory_payload is not None:
            msg = "Failed confirmatory custody cannot masquerade partial trajectories as a final test sample."
            raise ValueError(msg)
        return cls._from_parts(job, request, outcome, custody)

    @classmethod
    def _from_parts(
        cls,
        job: TrainingJob,
        request: ConfirmExecutionRequest,
        outcome: TrainingJobOutcome | None,
        custody: ProductionResultCustody | None,
    ) -> LockedConfirmatoryStudyRow:
        state: Literal["success", "failure", "unattempted"] = "unattempted" if outcome is None else outcome.status
        reference = None if custody is None else custody.reference
        evidence = None if custody is None else custody.production_evidence
        raw_root = None if custody is None else custody.raw_trajectory_document_checksum
        resource_root = None if custody is None else custody.resource_document_checksum
        diagnostic_checksums = (
            () if custody is None else tuple(item.content_checksum for item in custody.pilot_diagnostics)
        )
        partial_root = None if state != "failure" or reference is None else reference.manifest_content_checksum
        observed = 0 if custody is None or custody.trajectory_fidelities is None else len(custody.trajectory_fidelities)
        inventory_root = canonical_checksum({
            "job_checksum": job.content_checksum,
            "request_checksum": request.content_checksum,
            "terminal_state": state,
            "outer_outcome_checksum": None if outcome is None else outcome.content_checksum,
            "production_result_reference_checksum": None if reference is None else reference.content_checksum,
            "production_evidence_checksum": None if evidence is None else evidence.content_checksum,
            "production_custody_checksum": None if custody is None else custody.content_checksum,
            "raw_trajectory_document_checksum": raw_root,
            "resource_document_checksum": resource_root,
            "pilot_diagnostic_checksums": list(diagnostic_checksums),
            "partial_artifact_root": partial_root,
        })
        return cls._build(
            job_checksum=job.content_checksum,
            job_id=job.job_id,
            request_checksum=request.content_checksum,
            configuration_checksum=request.configuration_checksum,
            method_id=request.method_id,
            target_instance_id=request.target_instance_id,
            target_spec_checksum=request.target_spec_checksum,
            family_id=request.family_id,
            stratum_id=request.stratum_id,
            optimization_block_id=request.optimization_block_id,
            optimization_seed_index=request.optimization_seed_index,
            optimization_seed=request.optimization_seed,
            evaluation_seed=request.evaluation_seed,
            fixed_test_trajectory_count=request.fixed_test_trajectory_count,
            terminal_state=state,
            outcome=outcome,
            outer_outcome_checksum=None if outcome is None else outcome.content_checksum,
            production_result_reference=reference,
            production_evidence=evidence,
            production_custody_checksum=None if custody is None else custody.content_checksum,
            raw_trajectory_document_checksum=raw_root,
            resource_document_checksum=resource_root,
            pilot_diagnostic_checksums=diagnostic_checksums,
            partial_artifact_root=partial_root,
            observed_test_trajectory_count=observed,
            output_inventory_root=inventory_root,
        )

    @classmethod
    def _build(cls, **values: object) -> LockedConfirmatoryStudyRow:
        row = object.__new__(cls)
        for name, value in values.items():
            if name == "_content_checksum":
                continue
            _set_frozen(row, name, value)
        _set_frozen(row, "schema_version", LOCKED_CONFIRMATORY_STUDY_ROW_SCHEMA_VERSION)
        row._validate()  # noqa: SLF001 -- class-owned invariant validation
        _set_frozen(
            row,
            "_content_checksum",
            canonical_checksum(row._content_dict()),  # noqa: SLF001 -- class-owned checksum derivation
        )
        return row

    def _validate(self) -> None:
        """Validate row-local custody and derived inventory identities.

        Raises:
            ValueError: If identities, terminal custody, counts, or roots differ.
        """
        for name in ("job_checksum", "request_checksum", "configuration_checksum", "target_spec_checksum"):
            _set_frozen(self, name, require_checksum(getattr(self, name), name))
        for name in (
            "job_id",
            "method_id",
            "target_instance_id",
            "family_id",
            "stratum_id",
            "optimization_block_id",
        ):
            _set_frozen(self, name, require_slug(getattr(self, name), name))
        for name, minimum in (
            ("optimization_seed_index", 0),
            ("optimization_seed", 0),
            ("evaluation_seed", 0),
            ("fixed_test_trajectory_count", 2),
            ("observed_test_trajectory_count", 0),
        ):
            _set_frozen(self, name, require_int(getattr(self, name), name, minimum=minimum))
        if self.terminal_state not in {"success", "failure", "unattempted"}:
            msg = "terminal_state must be success, failure, or unattempted."
            raise ValueError(msg)
        for name in (
            "outer_outcome_checksum",
            "production_custody_checksum",
            "raw_trajectory_document_checksum",
            "resource_document_checksum",
            "partial_artifact_root",
        ):
            _set_frozen(self, name, _require_optional_checksum(getattr(self, name), name))
        diagnostic_checksums = tuple(
            require_checksum(item, "pilot_diagnostic_checksum") for item in self.pilot_diagnostic_checksums
        )
        if diagnostic_checksums:
            msg = "Confirmatory custody cannot contain pilot diagnostic members."
            raise ValueError(msg)
        _set_frozen(self, "pilot_diagnostic_checksums", diagnostic_checksums)
        _set_frozen(
            self,
            "output_inventory_root",
            require_checksum(self.output_inventory_root, "output_inventory_root"),
        )
        if self.terminal_state == "unattempted":
            if (
                any(
                    item is not None
                    for item in (
                        self.outcome,
                        self.outer_outcome_checksum,
                        self.production_result_reference,
                        self.production_evidence,
                        self.production_custody_checksum,
                        self.raw_trajectory_document_checksum,
                        self.resource_document_checksum,
                        self.partial_artifact_root,
                    )
                )
                or self.pilot_diagnostic_checksums
                or self.observed_test_trajectory_count != 0
            ):
                msg = "Unattempted rows cannot contain terminal outcome or production custody."
                raise ValueError(msg)
        else:
            if (
                not isinstance(self.outcome, TrainingJobOutcome)
                or not isinstance(self.production_result_reference, ResultArtifactRef)
                or not isinstance(self.production_evidence, ProductionNumericalEvidence)
                or self.outer_outcome_checksum != self.outcome.content_checksum
                or self.outcome.status != self.terminal_state
                or self.outcome.job_checksum != self.job_checksum
                or self.outcome.attempt != 1
                or self.production_result_reference.status != self.terminal_state
                or self.production_evidence.status != self.terminal_state
                or self.production_result_reference.job_checksum != self.request_checksum
                or self.production_evidence.job_checksum != self.request_checksum
                or self.production_result_reference.evidence_checksum != self.production_evidence.content_checksum
                or self.production_result_reference.attempt != 1
                or self.production_evidence.attempt != 1
                or self.production_custody_checksum is None
                or self.resource_document_checksum is None
            ):
                msg = "Terminal row lacks an authenticated first-attempt outcome and production custody closure."
                raise ValueError(msg)
            raw_ref = self.production_evidence.raw_trajectory_ref
            if (
                self.production_evidence.resource_ref.logical_checksum != self.resource_document_checksum
                or (raw_ref is None) != (self.raw_trajectory_document_checksum is None)
                or (raw_ref is not None and raw_ref.logical_checksum != self.raw_trajectory_document_checksum)
                or tuple(item.logical_checksum for item in self.production_evidence.diagnostic_refs)
                != self.pilot_diagnostic_checksums
            ):
                msg = "Production evidence member references differ from the row custody roots."
                raise ValueError(msg)
            expected_custody_checksum = canonical_checksum({
                "schema_version": PRODUCTION_RESULT_CUSTODY_SCHEMA_VERSION,
                "result_reference_checksum": self.production_result_reference.content_checksum,
                "result_evidence_checksum": self.production_evidence.content_checksum,
                "raw_trajectory_document_checksum": self.raw_trajectory_document_checksum,
                "resource_document_checksum": self.resource_document_checksum,
                "pilot_diagnostic_checksums": list(self.pilot_diagnostic_checksums),
            })
            if self.production_custody_checksum != expected_custody_checksum:
                msg = "production_custody_checksum is not derived from the embedded typed custody members."
                raise ValueError(msg)
            if self.terminal_state == "success":
                if (
                    self.outcome.result_artifact_checksum != self.production_result_reference.content_checksum
                    or self.raw_trajectory_document_checksum is None
                    or self.partial_artifact_root is not None
                    or self.observed_test_trajectory_count != self.fixed_test_trajectory_count
                ):
                    msg = "Successful row violates result-reference or fixed-trajectory custody."
                    raise ValueError(msg)
            elif (
                self.outcome.result_artifact_checksum is not None
                or self.raw_trajectory_document_checksum is not None
                or self.partial_artifact_root != self.production_result_reference.manifest_content_checksum
                or self.observed_test_trajectory_count != 0
            ):
                msg = "Failed row must preserve only its authenticated partial-attempt manifest root."
                raise ValueError(msg)
        expected_inventory = canonical_checksum({
            "job_checksum": self.job_checksum,
            "request_checksum": self.request_checksum,
            "terminal_state": self.terminal_state,
            "outer_outcome_checksum": self.outer_outcome_checksum,
            "production_result_reference_checksum": (
                None if self.production_result_reference is None else self.production_result_reference.content_checksum
            ),
            "production_evidence_checksum": (
                None if self.production_evidence is None else self.production_evidence.content_checksum
            ),
            "production_custody_checksum": self.production_custody_checksum,
            "raw_trajectory_document_checksum": self.raw_trajectory_document_checksum,
            "resource_document_checksum": self.resource_document_checksum,
            "pilot_diagnostic_checksums": list(self.pilot_diagnostic_checksums),
            "partial_artifact_root": self.partial_artifact_root,
        })
        if self.output_inventory_root != expected_inventory:
            msg = "Row output_inventory_root is not derived from the exact terminal custody roots."
            raise ValueError(msg)

    def validate_job(self, job: TrainingJob) -> None:
        """Validate this row against its exact embedded plan job.

        Raises:
            ValueError: If any row coordinate differs from the plan request.
        """
        request = job.confirm_execution_request
        if request is None or (
            self.job_checksum != job.content_checksum
            or self.job_id != job.job_id
            or self.request_checksum != request.content_checksum
            or self.configuration_checksum != request.configuration_checksum
            or self.method_id != request.method_id
            or self.target_instance_id != request.target_instance_id
            or self.target_spec_checksum != request.target_spec_checksum
            or self.family_id != request.family_id
            or self.stratum_id != request.stratum_id
            or self.optimization_block_id != request.optimization_block_id
            or self.optimization_seed_index != request.optimization_seed_index
            or self.optimization_seed != request.optimization_seed
            or self.evaluation_seed != request.evaluation_seed
            or self.fixed_test_trajectory_count != request.fixed_test_trajectory_count
        ):
            msg = "Locked study row differs from its exact planned ConfirmExecutionRequest."
            raise ValueError(msg)

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "job_checksum": self.job_checksum,
            "job_id": self.job_id,
            "request_checksum": self.request_checksum,
            "configuration_checksum": self.configuration_checksum,
            "method_id": self.method_id,
            "target_instance_id": self.target_instance_id,
            "target_spec_checksum": self.target_spec_checksum,
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "optimization_block_id": self.optimization_block_id,
            "optimization_seed_index": self.optimization_seed_index,
            "optimization_seed": self.optimization_seed,
            "evaluation_seed": self.evaluation_seed,
            "fixed_test_trajectory_count": self.fixed_test_trajectory_count,
            "terminal_state": self.terminal_state,
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
            "outer_outcome_checksum": self.outer_outcome_checksum,
            "production_result_reference": (
                None if self.production_result_reference is None else self.production_result_reference.to_dict()
            ),
            "production_evidence": None if self.production_evidence is None else self.production_evidence.to_dict(),
            "production_custody_checksum": self.production_custody_checksum,
            "raw_trajectory_document_checksum": self.raw_trajectory_document_checksum,
            "resource_document_checksum": self.resource_document_checksum,
            "pilot_diagnostic_checksums": list(self.pilot_diagnostic_checksums),
            "partial_artifact_root": self.partial_artifact_root,
            "observed_test_trajectory_count": self.observed_test_trajectory_count,
            "output_inventory_root": self.output_inventory_root,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of this exact planned-job outcome row."""
        return self._content_checksum

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native row data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> LockedConfirmatoryStudyRow:
        """Decode and verify one locked confirmatory study row.

        Returns:
            The verified row.  Its enclosing study subsequently closes it over
            the exact embedded plan job.

        Raises:
            ValueError: If the schema, custody fields, or checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_STUDY_ROW_KEYS, name="locked confirmatory study row")
        if mapping["schema_version"] != LOCKED_CONFIRMATORY_STUDY_ROW_SCHEMA_VERSION:
            msg = "Locked confirmatory study row uses an unsupported schema version."
            raise ValueError(msg)
        raw_outcome = mapping["outcome"]
        raw_reference = mapping["production_result_reference"]
        raw_evidence = mapping["production_evidence"]
        result = cls._build(
            job_checksum=cast("str", mapping["job_checksum"]),
            job_id=cast("str", mapping["job_id"]),
            request_checksum=cast("str", mapping["request_checksum"]),
            configuration_checksum=cast("str", mapping["configuration_checksum"]),
            method_id=cast("str", mapping["method_id"]),
            target_instance_id=cast("str", mapping["target_instance_id"]),
            target_spec_checksum=cast("str", mapping["target_spec_checksum"]),
            family_id=cast("str", mapping["family_id"]),
            stratum_id=cast("str", mapping["stratum_id"]),
            optimization_block_id=cast("str", mapping["optimization_block_id"]),
            optimization_seed_index=cast("int", mapping["optimization_seed_index"]),
            optimization_seed=cast("int", mapping["optimization_seed"]),
            evaluation_seed=cast("int", mapping["evaluation_seed"]),
            fixed_test_trajectory_count=cast("int", mapping["fixed_test_trajectory_count"]),
            terminal_state=cast("Literal['success', 'failure', 'unattempted']", mapping["terminal_state"]),
            outcome=None if raw_outcome is None else TrainingJobOutcome.from_dict(raw_outcome),
            outer_outcome_checksum=cast("str | None", mapping["outer_outcome_checksum"]),
            production_result_reference=(None if raw_reference is None else ResultArtifactRef.from_dict(raw_reference)),
            production_evidence=(None if raw_evidence is None else ProductionNumericalEvidence.from_dict(raw_evidence)),
            production_custody_checksum=cast("str | None", mapping["production_custody_checksum"]),
            raw_trajectory_document_checksum=cast(
                "str | None",
                mapping["raw_trajectory_document_checksum"],
            ),
            resource_document_checksum=cast("str | None", mapping["resource_document_checksum"]),
            pilot_diagnostic_checksums=tuple(
                cast(
                    "Sequence[str]",
                    _strict_sequence(mapping["pilot_diagnostic_checksums"], "pilot_diagnostic_checksums"),
                )
            ),
            partial_artifact_root=cast("str | None", mapping["partial_artifact_root"]),
            observed_test_trajectory_count=cast("int", mapping["observed_test_trajectory_count"]),
            output_inventory_root=cast("str", mapping["output_inventory_root"]),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Locked confirmatory study row checksum changed during normalization."
            raise ValueError(msg)
        return result


def _derive_pairability_records(
    plan: TrainingRunPlan,
    seal: FinalConfirmationSeal,
    rows: Sequence[LockedConfirmatoryStudyRow],
    exposure_inventory: PriorTargetExposureInventory,
) -> tuple[ConfirmatoryPairabilityRecord, ...]:
    """Derive the complete contrast/target/optimization pairing inventory.

    Returns:
        Canonically ordered pairability records for every sealed block.

    Raises:
        ValueError: If a planned contrast cell is missing or not pairable.
    """
    row_by_job = {row.job_checksum: row for row in rows}
    job_index: dict[tuple[str, str, int], TrainingJob] = {}
    for job in plan.jobs:
        request = cast("ConfirmExecutionRequest", job.confirm_execution_request)
        key = (request.configuration_checksum, request.target_instance_id, request.optimization_seed_index)
        if key in job_index:
            msg = "Confirmatory plan duplicates a configuration/target/optimization cell."
            raise ValueError(msg)
        job_index[key] = job
    target_ids = tuple(dict.fromkeys(job.target_instance_id for job in plan.jobs))
    screened_candidates = {
        item.configuration_checksum: item for item in exposure_inventory.screening_manifest.candidates
    }
    records: list[ConfirmatoryPairabilityRecord] = []
    for contrast in seal.primary_contrasts:
        protocol_checksum = _paired_test_protocol_checksum(seal, contrast.contrast_id)
        try:
            treatment_candidate = screened_candidates[contrast.treatment_configuration_checksum]
            control_candidate = screened_candidates[contrast.control_configuration_checksum]
        except KeyError as error:
            msg = "A confirmatory contrast configuration is absent from the exact screening manifest."
            raise ValueError(msg) from error
        if treatment_candidate.resource_stratum_id != control_candidate.resource_stratum_id:
            msg = "Contrasted screened candidates must use the same exact resource_stratum_id."
            raise ValueError(msg)
        resource_stratum_id = treatment_candidate.resource_stratum_id
        for target_id in target_ids:
            for seed_index in range(seal.optimization_seed_count):
                try:
                    treatment_job = job_index[contrast.treatment_configuration_checksum, target_id, seed_index]
                    control_job = job_index[contrast.control_configuration_checksum, target_id, seed_index]
                except KeyError as error:
                    msg = "Confirmatory contrast lacks one exact target/optimization configuration cell."
                    raise ValueError(msg) from error
                treatment = cast("ConfirmExecutionRequest", treatment_job.confirm_execution_request)
                control = cast("ConfirmExecutionRequest", control_job.confirm_execution_request)
                if (
                    treatment.optimization_seed != control.optimization_seed
                    or treatment.optimization_block_id != control.optimization_block_id
                    or treatment.target_spec_checksum != control.target_spec_checksum
                    or treatment.evaluation_seed == control.evaluation_seed
                ):
                    msg = "Contrasted jobs do not share one optimization block with distinct evaluation seeds."
                    raise ValueError(msg)
                treatment_row = row_by_job[treatment_job.content_checksum]
                control_row = row_by_job[control_job.content_checksum]
                successful_resource_pair_checksum: str | None = None
                if treatment_row.terminal_state == control_row.terminal_state == "success":
                    successful_resource_pair_checksum = canonical_checksum({
                        "treatment_resource_document_checksum": treatment_row.resource_document_checksum,
                        "control_resource_document_checksum": control_row.resource_document_checksum,
                    })
                paired_block = PairedBlockIdentity(
                    target_instance_id=target_id,
                    target_manifest_checksum=treatment.target_manifest_checksum,
                    target_spec_checksum=treatment.target_spec_checksum,
                    optimization_block_id=treatment.optimization_block_id,
                    optimization_seed=treatment.optimization_seed,
                    test_noise_id=cast("str", treatment.primary_noise_condition["noise_id"]),
                    test_protocol_checksum=protocol_checksum,
                    resource_stratum_id=resource_stratum_id,
                )
                records.append(
                    ConfirmatoryPairabilityRecord._build(  # noqa: SLF001 -- module-owned derived factory
                        contrast_id=contrast.contrast_id,
                        treatment_configuration_checksum=treatment.configuration_checksum,
                        control_configuration_checksum=control.configuration_checksum,
                        treatment_job_checksum=treatment_job.content_checksum,
                        control_job_checksum=control_job.content_checksum,
                        treatment_request_checksum=treatment.content_checksum,
                        control_request_checksum=control.content_checksum,
                        treatment_evaluation_seed=treatment.evaluation_seed,
                        control_evaluation_seed=control.evaluation_seed,
                        paired_block=paired_block,
                        actual_stream_mode=_ACTUAL_STREAM_MODE,
                        actual_stream_reason=_INDEPENDENT_STREAM_REASON,
                        successful_resource_pair_checksum=successful_resource_pair_checksum,
                        event_level_test_coupling=None,
                    )
                )
    return tuple(sorted(records, key=lambda item: item.sort_key))


@dataclass(frozen=True, slots=True, init=False)
class LockedConfirmatoryStudyManifest:
    """Complete locked confirmatory job/outcome/pairability custody manifest.

    ``output_inventory_root`` is the root of typed custody member identities;
    the later store integration must separately close the filesystem inventory
    over exact relative paths and file bytes.
    """

    plan: TrainingRunPlan
    final_seal: FinalConfirmationSeal
    configuration_execution_manifest: FinalConfigurationExecutionManifest
    target_manifest: TargetPopulationManifest
    exposure_inventory: PriorTargetExposureInventory
    execution_source_manifest_checksum: str
    analysis_source_manifest_checksum: str
    analysis_template_checksum: str
    rows: tuple[LockedConfirmatoryStudyRow, ...]
    pairability_records: tuple[ConfirmatoryPairabilityRecord, ...]
    status: Literal["complete", "incomplete", "incomplete_resource_limit"]
    planned_job_count: int
    terminal_job_count: int
    successful_job_count: int
    failed_job_count: int
    unattempted_job_count: int
    planned_test_trajectory_count: int
    observed_test_trajectory_count: int
    output_inventory_root: str
    schema_version: str = field(default=LOCKED_CONFIRMATORY_STUDY_MANIFEST_SCHEMA_VERSION, init=False)
    _content_checksum: str = field(init=False, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        *,
        context: ConfirmationExecutionContext,
        exposure_inventory: PriorTargetExposureInventory,
    ) -> LockedConfirmatoryStudyManifest:
        """Create only the initial all-unattempted confirmatory study manifest.

        Terminal evidence cannot cross this public constructor.  Operational
        store collection uses :meth:`_from_authenticated_reopened_results`.

        Returns:
            The fully derived initial manifest with every row unattempted.
        """
        return cls._from_authenticated_reopened_results(
            context=context,
            exposure_inventory=exposure_inventory,
            outcomes_by_job={},
            reopened_results_by_job={},
        )

    @classmethod
    def _from_authenticated_reopened_results(
        cls,
        *,
        context: ConfirmationExecutionContext,
        exposure_inventory: PriorTargetExposureInventory,
        outcomes_by_job: Mapping[str, TrainingJobOutcome],
        reopened_results_by_job: Mapping[str, ReopenedProductionResult],
    ) -> LockedConfirmatoryStudyManifest:
        """Derive terminal rows solely from authenticated reopened attempts.

        Operational store code must obtain every ``ReopenedProductionResult``
        by calling ``validate_existing_confirmation_outcome`` on the exact
        context-owned job and canonical output directory.  This constructor
        then creates ``ProductionResultCustody`` internally; detached custody
        summaries are not accepted at this boundary.

        Returns:
            The fully derived locked study manifest.

        Raises:
            TypeError: If context, exposure inventory, maps, outcomes, or reopened values use the wrong type.
            ValueError: If map membership or terminal custody is inconsistent.
        """
        if not isinstance(context, ConfirmationExecutionContext):
            msg = "context must be a ConfirmationExecutionContext."
            raise TypeError(msg)
        if not isinstance(exposure_inventory, PriorTargetExposureInventory):
            msg = "exposure_inventory must be a PriorTargetExposureInventory."
            raise TypeError(msg)
        if context.prior_target_exposure_inventory_checksum != exposure_inventory.content_checksum:
            msg = "Exposure inventory differs from the novelty root bound to the confirmation session."
            raise ValueError(msg)
        if not isinstance(outcomes_by_job, Mapping) or not isinstance(reopened_results_by_job, Mapping):
            msg = "outcomes_by_job and reopened_results_by_job must be mappings."
            raise TypeError(msg)
        outcomes = outcomes_by_job
        reopened_results = reopened_results_by_job
        known = {job.content_checksum for job in context.plan.jobs}
        unknown_outcomes = sorted(set(outcomes) - known)
        unknown_reopened = sorted(set(reopened_results) - known)
        if unknown_outcomes or unknown_reopened:
            msg = (
                "Terminal inputs contain jobs outside the sealed confirmation plan: "
                f"outcomes={unknown_outcomes!r}, reopened={unknown_reopened!r}."
            )
            raise ValueError(msg)
        if set(outcomes) != set(reopened_results):
            msg = "Every terminal confirmatory job requires both outcome and authenticated reopened result."
            raise ValueError(msg)
        custodies: dict[str, ProductionResultCustody] = {}
        for job_checksum, reopened in reopened_results.items():
            if not isinstance(reopened, ReopenedProductionResult):
                msg = "reopened_results_by_job must contain ReopenedProductionResult values."
                raise TypeError(msg)
            custodies[job_checksum] = ProductionResultCustody(reopened)
        exposure_inventory.validate_confirmatory_novelty(context.target_manifest)
        rows = tuple(
            LockedConfirmatoryStudyRow.from_execution(
                context,
                job,
                outcomes.get(job.content_checksum),
                custodies.get(job.content_checksum),
            )
            for job in context.plan.jobs
        )
        pairability = _derive_pairability_records(
            context.plan,
            context.final_seal,
            rows,
            exposure_inventory,
        )
        successful = sum(row.terminal_state == "success" for row in rows)
        failed = sum(row.terminal_state == "failure" for row in rows)
        unattempted = sum(row.terminal_state == "unattempted" for row in rows)
        terminal = successful + failed
        status = cls._derive_status(rows, context.final_seal)
        output_root = cls._output_root(context.plan, context.final_seal, rows, pairability)
        return cls._build(
            plan=context.plan,
            final_seal=context.final_seal,
            configuration_execution_manifest=context.configuration_execution_manifest,
            target_manifest=context.target_manifest,
            exposure_inventory=exposure_inventory,
            execution_source_manifest_checksum=context.execution_source_manifest.content_checksum,
            analysis_source_manifest_checksum=context.analysis_source_manifest.content_checksum,
            analysis_template_checksum=context.preregistration.analysis_template_checksum,
            rows=rows,
            pairability_records=pairability,
            status=status,
            planned_job_count=len(rows),
            terminal_job_count=terminal,
            successful_job_count=successful,
            failed_job_count=failed,
            unattempted_job_count=unattempted,
            planned_test_trajectory_count=sum(row.fixed_test_trajectory_count for row in rows),
            observed_test_trajectory_count=sum(row.observed_test_trajectory_count for row in rows),
            output_inventory_root=output_root,
        )

    @staticmethod
    def _derive_status(
        rows: Sequence[LockedConfirmatoryStudyRow],
        seal: FinalConfirmationSeal,
    ) -> Literal["complete", "incomplete", "incomplete_resource_limit"]:
        """Derive completeness and enforce resource-stop terminal custody.

        Returns:
            Complete, generic incomplete, or resource-limit incomplete status.

        Raises:
            TypeError: If ``seal`` is not a final confirmation seal.
            ValueError: If a terminal row occurs after authenticated resource-limit failure evidence.
        """
        unattempted = sum(row.terminal_state == "unattempted" for row in rows)
        if not isinstance(seal, FinalConfirmationSeal):
            msg = "seal must be a FinalConfirmationSeal."
            raise TypeError(msg)
        resource_stops: list[int] = []
        for index, row in enumerate(rows):
            if row.terminal_state != "failure" or row.production_evidence is None:
                continue
            failure = row.production_evidence.failure
            raw_proof = None if failure is None else failure.get("resource_limit_proof")
            if raw_proof is None:
                continue
            proof = ConfirmationResourceLimitProof.from_dict(raw_proof)
            expected_exception_type = (
                "NormalizedComputeCapError"
                if proof.proof_kind == "prospective_normalized_work"
                else "ConfirmationResourceLimitError"
            )
            if failure is None or failure.get("exception_type") != expected_exception_type:
                msg = "Confirmatory resource-limit proof differs from its typed failure exception family."
                raise ValueError(msg)
            expected_normalized_cap = cast("float", seal.primary_resource_budget["normalized_compute_cap"])
            expected_edge_cap = cast("float", seal.primary_resource_budget["cap_per_chain_edge"])
            if (
                proof.request_checksum != row.request_checksum
                or float(proof.normalized_compute_cap).hex() != float(expected_normalized_cap).hex()
                or float(proof.native_edge_gate_cap).hex() != float(expected_edge_cap).hex()
            ):
                msg = "Confirmatory resource-limit proof differs from its row or final-seal caps."
                raise ValueError(msg)
            resource_stops.append(index)
        if resource_stops:
            first_stop = resource_stops[0]
            if any(row.terminal_state != "unattempted" for row in rows[first_stop + 1 :]):
                msg = "No terminal confirmatory row may follow authenticated resource-limit failure evidence."
                raise ValueError(msg)
            if unattempted:
                return "incomplete_resource_limit"
        return "complete" if unattempted == 0 else "incomplete"

    @staticmethod
    def _output_root(
        plan: TrainingRunPlan,
        seal: FinalConfirmationSeal,
        rows: Sequence[LockedConfirmatoryStudyRow],
        pairability: Sequence[ConfirmatoryPairabilityRecord],
    ) -> str:
        """Derive the complete row and pairability output inventory root.

        Returns:
            The checksum of every typed row and pairability custody root.
        """
        return canonical_checksum({
            "plan_checksum": plan.content_checksum,
            "final_seal_checksum": seal.content_checksum,
            "row_checksums": [row.content_checksum for row in rows],
            "row_output_inventory_roots": [row.output_inventory_root for row in rows],
            "pairability_record_checksums": [record.content_checksum for record in pairability],
        })

    @classmethod
    def _build(cls, **values: object) -> LockedConfirmatoryStudyManifest:
        manifest = object.__new__(cls)
        for name, value in values.items():
            if name == "_content_checksum":
                continue
            _set_frozen(manifest, name, value)
        _set_frozen(manifest, "schema_version", LOCKED_CONFIRMATORY_STUDY_MANIFEST_SCHEMA_VERSION)
        manifest._validate()  # noqa: SLF001 -- class-owned invariant validation
        _set_frozen(
            manifest,
            "_content_checksum",
            canonical_checksum(manifest._content_dict()),  # noqa: SLF001 -- class-owned checksum derivation
        )
        return manifest

    def _validate(self) -> None:
        """Validate every root, row, count, and pairability projection.

        Raises:
            TypeError: If an embedded artifact or row uses the wrong type.
            ValueError: If any root, row universe, count, or pairing differs.
        """
        typed = (
            (self.plan, TrainingRunPlan, "plan"),
            (self.final_seal, FinalConfirmationSeal, "final_seal"),
            (
                self.configuration_execution_manifest,
                FinalConfigurationExecutionManifest,
                "configuration_execution_manifest",
            ),
            (self.target_manifest, TargetPopulationManifest, "target_manifest"),
            (self.exposure_inventory, PriorTargetExposureInventory, "exposure_inventory"),
        )
        for value, expected, name in typed:
            if not isinstance(value, expected):
                msg = f"{name} must be a {expected.__name__}."
                raise TypeError(msg)
        for name in (
            "execution_source_manifest_checksum",
            "analysis_source_manifest_checksum",
            "analysis_template_checksum",
            "output_inventory_root",
        ):
            _set_frozen(self, name, require_checksum(getattr(self, name), name))
        self.exposure_inventory.validate_confirmatory_novelty(self.target_manifest)
        expected_plan = build_paper_confirm_plan(
            seal=self.final_seal,
            target_manifest=self.target_manifest,
            configuration_execution_manifest=self.configuration_execution_manifest,
        )
        reachable_stratum_manifest_checksum = cast(
            "str",
            self.final_seal.primary_resource_budget["reachable_stratum_manifest_checksum"],
        )
        if (
            self.plan != expected_plan
            or self.plan.final_confirmation_seal_checksum != self.final_seal.content_checksum
            or self.final_seal.confirmatory_target_manifest_checksum != self.target_manifest.content_checksum
            or self.final_seal.execution_source_checksum != self.execution_source_manifest_checksum
            or self.final_seal.analysis_source_manifest_checksum != self.analysis_source_manifest_checksum
            or self.final_seal.analysis_template_checksum != self.analysis_template_checksum
            or self.final_seal.hyperparameters_checksum != self.configuration_execution_manifest.content_checksum
            or self.plan.preregistration_checksum != self.exposure_inventory.preregistration_checksum
            or reachable_stratum_manifest_checksum != self.exposure_inventory.resource_calibration_checksum
            or self.final_seal.execution_source_checksum
            != self.exposure_inventory.resource_calibration_execution_source_checksum
        ):
            msg = "Locked study roots do not reproduce the exact sealed confirmatory plan."
            raise ValueError(msg)
        rows = tuple(self.rows)
        if not all(isinstance(row, LockedConfirmatoryStudyRow) for row in rows):
            msg = "rows must contain LockedConfirmatoryStudyRow values."
            raise TypeError(msg)
        if len(rows) != len(self.plan.jobs):
            msg = "Locked study requires exactly one row for every planned job."
            raise ValueError(msg)
        if len({row.job_checksum for row in rows}) != len(rows):
            msg = "Locked study rows cannot duplicate a planned job."
            raise ValueError(msg)
        for row, job in zip(rows, self.plan.jobs, strict=True):
            row.validate_job(job)
        seen_unattempted = False
        for row in rows:
            if row.terminal_state == "unattempted":
                seen_unattempted = True
            elif seen_unattempted:
                msg = "Terminal confirmatory rows must form one exact contiguous plan-order prefix."
                raise ValueError(msg)
        pairability = tuple(self.pairability_records)
        if not all(isinstance(item, ConfirmatoryPairabilityRecord) for item in pairability):
            msg = "pairability_records must contain ConfirmatoryPairabilityRecord values."
            raise TypeError(msg)
        expected_pairability = _derive_pairability_records(
            self.plan,
            self.final_seal,
            rows,
            self.exposure_inventory,
        )
        if pairability != expected_pairability:
            msg = "Pairability records are missing, duplicated, reordered, or differ from the sealed contrasts."
            raise ValueError(msg)
        successful = sum(row.terminal_state == "success" for row in rows)
        failed = sum(row.terminal_state == "failure" for row in rows)
        unattempted = sum(row.terminal_state == "unattempted" for row in rows)
        terminal = successful + failed
        derived_status = self._derive_status(rows, self.final_seal)
        expected_counts = (
            (self.planned_job_count, len(rows), "planned_job_count"),
            (self.terminal_job_count, terminal, "terminal_job_count"),
            (self.successful_job_count, successful, "successful_job_count"),
            (self.failed_job_count, failed, "failed_job_count"),
            (self.unattempted_job_count, unattempted, "unattempted_job_count"),
            (
                self.planned_test_trajectory_count,
                sum(row.fixed_test_trajectory_count for row in rows),
                "planned_test_trajectory_count",
            ),
            (
                self.observed_test_trajectory_count,
                sum(row.observed_test_trajectory_count for row in rows),
                "observed_test_trajectory_count",
            ),
        )
        for supplied, expected, name in expected_counts:
            normalized = require_int(supplied, name, minimum=0)
            _set_frozen(self, name, normalized)
            if normalized != expected:
                msg = f"{name} must be derived from the exact canonical rows."
                raise ValueError(msg)
        if self.status != derived_status:
            msg = "Study status must be complete exactly when every planned row is terminal."
            raise ValueError(msg)
        expected_root = self._output_root(self.plan, self.final_seal, rows, pairability)
        if self.output_inventory_root != expected_root:
            msg = "Study output_inventory_root is not derived from every row and pairing record."
            raise ValueError(msg)
        _set_frozen(self, "rows", rows)
        _set_frozen(self, "pairability_records", pairability)

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan": self.plan.to_dict(),
            "final_seal": self.final_seal.to_dict(),
            "configuration_execution_manifest": self.configuration_execution_manifest.to_dict(),
            "target_manifest": self.target_manifest.to_dict(),
            "exposure_inventory": self.exposure_inventory.to_dict(),
            "execution_source_manifest_checksum": self.execution_source_manifest_checksum,
            "analysis_source_manifest_checksum": self.analysis_source_manifest_checksum,
            "analysis_template_checksum": self.analysis_template_checksum,
            "rows": [row.to_dict() for row in self.rows],
            "pairability_records": [record.to_dict() for record in self.pairability_records],
            "status": self.status,
            "planned_job_count": self.planned_job_count,
            "terminal_job_count": self.terminal_job_count,
            "successful_job_count": self.successful_job_count,
            "failed_job_count": self.failed_job_count,
            "unattempted_job_count": self.unattempted_job_count,
            "planned_test_trajectory_count": self.planned_test_trajectory_count,
            "observed_test_trajectory_count": self.observed_test_trajectory_count,
            "output_inventory_root": self.output_inventory_root,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete locked study custody."""
        return self._content_checksum

    def to_dict(self) -> dict[str, object]:
        """Return checksum-sealed JSON-native study data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> LockedConfirmatoryStudyManifest:
        """Decode and verify one locked confirmatory study manifest.

        Returns:
            The verified complete or incomplete study custody.

        Raises:
            ValueError: If the schema, embedded custody, or checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_STUDY_MANIFEST_KEYS, name="locked confirmatory study")
        if mapping["schema_version"] != LOCKED_CONFIRMATORY_STUDY_MANIFEST_SCHEMA_VERSION:
            msg = "Locked confirmatory study uses an unsupported schema version."
            raise ValueError(msg)
        result = cls._build(
            plan=TrainingRunPlan.from_dict(mapping["plan"]),
            final_seal=FinalConfirmationSeal.from_dict(mapping["final_seal"]),
            configuration_execution_manifest=FinalConfigurationExecutionManifest.from_dict(
                mapping["configuration_execution_manifest"]
            ),
            target_manifest=TargetPopulationManifest.from_dict(mapping["target_manifest"]),
            exposure_inventory=PriorTargetExposureInventory.from_dict(mapping["exposure_inventory"]),
            execution_source_manifest_checksum=cast("str", mapping["execution_source_manifest_checksum"]),
            analysis_source_manifest_checksum=cast("str", mapping["analysis_source_manifest_checksum"]),
            analysis_template_checksum=cast("str", mapping["analysis_template_checksum"]),
            rows=tuple(
                LockedConfirmatoryStudyRow.from_dict(item) for item in _strict_sequence(mapping["rows"], "rows")
            ),
            pairability_records=tuple(
                ConfirmatoryPairabilityRecord.from_dict(item)
                for item in _strict_sequence(mapping["pairability_records"], "pairability_records")
            ),
            status=cast(
                "Literal['complete', 'incomplete', 'incomplete_resource_limit']",
                mapping["status"],
            ),
            planned_job_count=cast("int", mapping["planned_job_count"]),
            terminal_job_count=cast("int", mapping["terminal_job_count"]),
            successful_job_count=cast("int", mapping["successful_job_count"]),
            failed_job_count=cast("int", mapping["failed_job_count"]),
            unattempted_job_count=cast("int", mapping["unattempted_job_count"]),
            planned_test_trajectory_count=cast("int", mapping["planned_test_trajectory_count"]),
            observed_test_trajectory_count=cast("int", mapping["observed_test_trajectory_count"]),
            output_inventory_root=cast("str", mapping["output_inventory_root"]),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Locked confirmatory study checksum changed during normalization."
            raise ValueError(msg)
        return result

    @classmethod
    def from_json(cls, payload: str) -> LockedConfirmatoryStudyManifest:
        """Decode a locked study from canonical JSON.

        Returns:
            The verified study custody.
        """
        return cls.from_dict(load_canonical_json_object(payload))


__all__ = [
    "CONFIRMATORY_PAIRABILITY_RECORD_SCHEMA_VERSION",
    "LOCKED_CONFIRMATORY_STUDY_MANIFEST_SCHEMA_VERSION",
    "LOCKED_CONFIRMATORY_STUDY_ROW_SCHEMA_VERSION",
    "PRIOR_TARGET_EXPOSURE_INVENTORY_SCHEMA_VERSION",
    "PRIOR_TARGET_EXPOSURE_RECORD_SCHEMA_VERSION",
    "ConfirmatoryPairabilityRecord",
    "LockedConfirmatoryStudyManifest",
    "LockedConfirmatoryStudyRow",
    "PriorTargetExposureInventory",
    "PriorTargetExposureRecord",
    "validate_confirmatory_novelty",
]
