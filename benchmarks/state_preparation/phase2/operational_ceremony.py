# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Non-numerical WP22H artifact ceremony over authenticated production runs.

This module does not execute training, screen candidates, open confirmatory
targets, or make scientific choices.  It reopens the exact authoritative
first-attempt pilot and screening output universes, invokes the already frozen
mechanical derivations, and emits a checksum-sealed readiness receipt for the
separate WP23 unblinding ceremony.
"""

from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from .binding_catalog import RepositoryBindingCatalog
from .canonical import canonical_checksum, canonical_json, load_canonical_json_object, verify_sealed_mapping
from .confirmatory_study import PriorTargetExposureInventory
from .execution_context import (
    AuthorizedTargetMaterialization,
    ExternalEntropyKeyring,
    TrainingExecutionContext,
    bind_training_plan_fingerprints,
    candidate_refs_from_bindings,
    schedules_from_bindings,
)
from .execution_registry import (
    build_paper_pilot_contrast_bindings,
    build_paper_screen_execution_registry,
)
from .legacy import LegacyEvidenceAudit
from .pilot import (
    PilotContrastBinding,
    PilotNuisanceSummary,
    ProductionPilotCustody,
    build_cluster_aware_paired_difference_v1,
)
from .protocol import (
    AnalysisSourceManifest,
    FinalConfigurationExecutionManifest,
    FinalConfirmationSeal,
    InitialPreregistration,
    PromotionDecision,
    SampleSizeDesign,
    ScreeningEvidence,
    ScreeningManifest,
)
from .resumability import ExecutionSourceEntry, ResumabilityFingerprint
from .screening import (
    PilotNormalizedComputeCalibration,
    ProductionResourceCalibration,
    ProductionScreeningCustody,
    build_final_configuration_execution_manifest,
    build_pilot_normalized_compute_calibration,
    build_production_resource_calibration,
    create_final_confirmation_seal,
)
from .source_lock import ExecutionSourceManifest, verify_final_seal_source_lock
from .targets import (
    TargetPopulationCommitment,
    TargetPopulationConfig,
    TargetPopulationManifest,
    authorize_target_materialization,
)
from .training_orchestration import (
    PILOT_OPTIMIZATION_SEED_COUNT,
    build_paper_pilot_plan,
    build_paper_screen_plan,
    derive_pilot_optimization_seeds,
)
from .validation import require_checksum, require_git_commit

if TYPE_CHECKING:
    from pathlib import Path

WP22H_READINESS_RECEIPT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22h_readiness_receipt.v1"
_LOGICAL_ARTIFACT_REGISTRY_SCHEMA_VERSION = "yaqs.state_preparation.phase2.wp22h_artifact_registry.v1"

_READINESS_ARTIFACT_CHECKSUM_FIELDS = (
    "preregistration_checksum",
    "execution_source_manifest_checksum",
    "analysis_source_manifest_checksum",
    "pilot_plan_checksum",
    "pilot_primary_target_manifest_checksum",
    "pilot_secondary_target_manifest_checksum",
    "pilot_custody_checksum",
    "pilot_secondary_archive_checksum",
    "pilot_nuisance_summary_checksum",
    "sample_size_design_checksum",
    "pilot_calibration_checksum",
    "screening_plan_checksum",
    "screening_target_manifest_checksum",
    "screening_manifest_checksum",
    "screening_custody_checksum",
    "screening_evidence_checksum",
    "promotion_decision_checksum",
    "resource_calibration_checksum",
    "configuration_execution_manifest_checksum",
    "paper_screen_binding_catalog_checksum",
    "confirmatory_target_configuration_checksum",
    "confirmatory_target_commitment_checksum",
    "final_confirmation_seal_checksum",
    "prior_target_exposure_inventory_checksum",
    "pre_seal_chain_head_stage_manifest_checksum",
    "close_screen_operational_paths_checksum",
)

_READINESS_KEYS = frozenset({
    "schema_version",
    "source_commit",
    "preregistration_checksum",
    "execution_source_manifest_checksum",
    "analysis_source_manifest_checksum",
    "pilot_plan_checksum",
    "pilot_primary_target_manifest_checksum",
    "pilot_secondary_target_manifest_checksum",
    "pilot_custody_checksum",
    "pilot_secondary_archive_checksum",
    "pilot_nuisance_summary_checksum",
    "sample_size_design_checksum",
    "pilot_calibration_checksum",
    "screening_plan_checksum",
    "screening_target_manifest_checksum",
    "screening_manifest_checksum",
    "screening_custody_checksum",
    "screening_evidence_checksum",
    "promotion_decision_checksum",
    "resource_calibration_checksum",
    "configuration_execution_manifest_checksum",
    "paper_screen_binding_catalog_checksum",
    "confirmatory_target_configuration_checksum",
    "confirmatory_target_commitment_checksum",
    "final_confirmation_seal_checksum",
    "prior_target_exposure_inventory_checksum",
    "pre_seal_chain_head_stage_manifest_checksum",
    "close_screen_operational_paths_checksum",
    "artifact_registry_checksum",
    "confirmatory_configuration_count",
    "confirmatory_target_count",
    "confirmatory_optimization_seed_count",
    "confirmatory_job_count",
    "held_target_manifest_opened",
    "held_entropy_opened",
    "numerical_execution_performed",
    "content_checksum",
})


def _artifact_registry_checksum(source_commit: str, artifacts: dict[str, str]) -> str:
    """Return the logical checksum registry for every ceremony artifact root."""
    return canonical_checksum({
        "schema_version": _LOGICAL_ARTIFACT_REGISTRY_SCHEMA_VERSION,
        "source_commit": source_commit,
        "artifacts": [
            {"artifact_id": name.removesuffix("_checksum"), "content_checksum": artifacts[name]}
            for name in _READINESS_ARTIFACT_CHECKSUM_FIELDS
        ],
    })


@dataclass(frozen=True, slots=True)
class ProductionPilotClosure:
    """Mechanically derived outputs of one complete production pilot."""

    custody: ProductionPilotCustody
    contrast_bindings: tuple[PilotContrastBinding, ...]
    nuisance_summary: PilotNuisanceSummary
    sample_size_design: SampleSizeDesign
    pilot_calibration: PilotNormalizedComputeCalibration

    def __post_init__(self) -> None:
        """Require one internally consistent, production-derived closure.

        Raises:
            TypeError: If a member has the wrong exact runtime schema.
            ValueError: If a derived artifact does not bind this pilot.
        """
        typed = (
            (self.custody, ProductionPilotCustody, "custody"),
            (self.nuisance_summary, PilotNuisanceSummary, "nuisance_summary"),
            (self.sample_size_design, SampleSizeDesign, "sample_size_design"),
            (self.pilot_calibration, PilotNormalizedComputeCalibration, "pilot_calibration"),
        )
        for value, expected, name in typed:
            if not isinstance(value, expected):
                msg = f"{name} must be {expected.__name__}."
                raise TypeError(msg)
        bindings = tuple(self.contrast_bindings)
        if not bindings or any(not isinstance(item, PilotContrastBinding) for item in bindings):
            msg = "contrast_bindings must contain PilotContrastBinding values."
            raise TypeError(msg)
        context = self.custody.context
        expected_bindings = build_paper_pilot_contrast_bindings(context.plan)
        if bindings != expected_bindings:
            msg = "Production pilot closure must use the canonical execution-registry contrasts."
            raise ValueError(msg)
        reproduced_summary = self.custody.build_nuisance_summary(
            expected_bindings,
            summary_id=self.nuisance_summary.summary_id,
        )
        reproduced_design = build_cluster_aware_paired_difference_v1(
            context.preregistration,
            reproduced_summary,
            design_id=self.sample_size_design.design_id,
        )
        reproduced_calibration = build_pilot_normalized_compute_calibration(self.custody)
        projections = (
            (self.nuisance_summary.to_json(), reproduced_summary.to_json()),
            (self.sample_size_design.to_json(), reproduced_design.to_json()),
            (self.pilot_calibration.to_json(), reproduced_calibration.to_json()),
        )
        if any(actual != expected for actual, expected in projections):
            msg = "Production pilot closure differs from its mechanically rederived custody projections."
            raise ValueError(msg)
        exact = (
            (self.nuisance_summary.pilot_plan.content_checksum, context.plan.content_checksum),
            (self.sample_size_design.pilot_nuisance_summary_checksum, self.nuisance_summary.inference_checksum),
            (self.pilot_calibration.pilot_plan_checksum, context.plan.content_checksum),
            (
                self.pilot_calibration.execution_source_manifest_checksum,
                context.execution_source_manifest.content_checksum,
            ),
        )
        if any(actual != expected for actual, expected in exact):
            msg = "Production pilot closure artifacts do not share one exact source-locked pilot root."
            raise ValueError(msg)
        object.__setattr__(self, "contrast_bindings", bindings)


@dataclass(frozen=True, slots=True)
class ProductionScreenClosure:
    """Mechanically derived outputs of one complete production screen."""

    custody: ProductionScreeningCustody
    screening_evidence: ScreeningEvidence
    promotion_decision: PromotionDecision
    resource_calibration: ProductionResourceCalibration
    configuration_execution_manifest: FinalConfigurationExecutionManifest

    def __post_init__(self) -> None:
        """Require exact production custody and mutually linked projections.

        Raises:
            TypeError: If a member has the wrong exact runtime schema.
            ValueError: If a projection belongs to another screen.
        """
        typed = (
            (self.custody, ProductionScreeningCustody, "custody"),
            (self.screening_evidence, ScreeningEvidence, "screening_evidence"),
            (self.promotion_decision, PromotionDecision, "promotion_decision"),
            (self.resource_calibration, ProductionResourceCalibration, "resource_calibration"),
            (
                self.configuration_execution_manifest,
                FinalConfigurationExecutionManifest,
                "configuration_execution_manifest",
            ),
        )
        for value, expected, name in typed:
            if not isinstance(value, expected):
                msg = f"{name} must be {expected.__name__}."
                raise TypeError(msg)
        context = self.custody.context
        manifest = context.screening_manifest
        if manifest is None:
            msg = "Production screen closure requires its exact screening manifest."
            raise ValueError(msg)
        reproduced_evidence, reproduced_decision = self.custody.build_evidence(
            evidence_id=self.screening_evidence.evidence_id,
        )
        reproduced_execution_manifest = build_final_configuration_execution_manifest(
            self.custody,
            reproduced_decision,
            manifest_id=self.configuration_execution_manifest.manifest_id,
        )
        projections = (
            (self.screening_evidence.to_json(), reproduced_evidence.to_json()),
            (self.promotion_decision.to_json(), reproduced_decision.to_json()),
            (self.configuration_execution_manifest.to_json(), reproduced_execution_manifest.to_json()),
        )
        if any(actual != expected for actual, expected in projections):
            msg = "Production screen closure differs from its mechanically rederived custody projections."
            raise ValueError(msg)
        exact = (
            (self.screening_evidence.screening_manifest_checksum, manifest.content_checksum),
            (self.promotion_decision.screening_evidence_checksum, self.screening_evidence.content_checksum),
            (self.resource_calibration.screening_plan_checksum, context.plan.content_checksum),
            (self.resource_calibration.screening_manifest_checksum, manifest.content_checksum),
        )
        if any(actual != expected for actual, expected in exact):
            msg = "Production screen closure artifacts do not share one exact authenticated screen root."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class WP22HReadinessReceipt:
    """Checksum-only proof that the nonconfirmatory ceremony is complete."""

    source_commit: str
    preregistration_checksum: str
    execution_source_manifest_checksum: str
    analysis_source_manifest_checksum: str
    pilot_plan_checksum: str
    pilot_primary_target_manifest_checksum: str
    pilot_secondary_target_manifest_checksum: str
    pilot_custody_checksum: str
    pilot_secondary_archive_checksum: str
    pilot_nuisance_summary_checksum: str
    sample_size_design_checksum: str
    pilot_calibration_checksum: str
    screening_plan_checksum: str
    screening_target_manifest_checksum: str
    screening_manifest_checksum: str
    screening_custody_checksum: str
    screening_evidence_checksum: str
    promotion_decision_checksum: str
    resource_calibration_checksum: str
    configuration_execution_manifest_checksum: str
    paper_screen_binding_catalog_checksum: str
    confirmatory_target_configuration_checksum: str
    confirmatory_target_commitment_checksum: str
    final_confirmation_seal_checksum: str
    prior_target_exposure_inventory_checksum: str
    pre_seal_chain_head_stage_manifest_checksum: str
    close_screen_operational_paths_checksum: str
    confirmatory_configuration_count: int
    confirmatory_target_count: int
    confirmatory_optimization_seed_count: int
    confirmatory_job_count: int
    held_target_manifest_opened: bool = field(default=False, init=False)
    held_entropy_opened: bool = field(default=False, init=False)
    numerical_execution_performed: bool = field(default=False, init=False)
    artifact_registry_checksum: str = field(init=False)
    schema_version: str = field(default=WP22H_READINESS_RECEIPT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate checksum syntax, counts, and the derived job cardinality.

        Raises:
            ValueError: If a checksum, commit, count, or dormancy flag is invalid.
        """
        for name in _READINESS_ARTIFACT_CHECKSUM_FIELDS:
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(self, "source_commit", require_git_commit(self.source_commit, "source_commit"))
        counts = (
            self.confirmatory_configuration_count,
            self.confirmatory_target_count,
            self.confirmatory_optimization_seed_count,
            self.confirmatory_job_count,
        )
        if any(type(value) is not int or value <= 0 for value in counts):
            msg = "Confirmatory readiness counts must be positive exact integers."
            raise ValueError(msg)
        expected_jobs = (
            self.confirmatory_configuration_count
            * self.confirmatory_target_count
            * self.confirmatory_optimization_seed_count
        )
        if self.confirmatory_job_count != expected_jobs:
            msg = "confirmatory_job_count must equal configurations times targets times optimization seeds."
            raise ValueError(msg)
        if self.held_target_manifest_opened or self.held_entropy_opened or self.numerical_execution_performed:
            msg = "WP22H readiness must remain dormant and non-numerical."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "artifact_registry_checksum",
            _artifact_registry_checksum(
                self.source_commit,
                {name: getattr(self, name) for name in _READINESS_ARTIFACT_CHECKSUM_FIELDS},
            ),
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every checksum-covered receipt field."""
        return {
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "preregistration_checksum": self.preregistration_checksum,
            "execution_source_manifest_checksum": self.execution_source_manifest_checksum,
            "analysis_source_manifest_checksum": self.analysis_source_manifest_checksum,
            "pilot_plan_checksum": self.pilot_plan_checksum,
            "pilot_primary_target_manifest_checksum": self.pilot_primary_target_manifest_checksum,
            "pilot_secondary_target_manifest_checksum": self.pilot_secondary_target_manifest_checksum,
            "pilot_custody_checksum": self.pilot_custody_checksum,
            "pilot_secondary_archive_checksum": self.pilot_secondary_archive_checksum,
            "pilot_nuisance_summary_checksum": self.pilot_nuisance_summary_checksum,
            "sample_size_design_checksum": self.sample_size_design_checksum,
            "pilot_calibration_checksum": self.pilot_calibration_checksum,
            "screening_plan_checksum": self.screening_plan_checksum,
            "screening_target_manifest_checksum": self.screening_target_manifest_checksum,
            "screening_manifest_checksum": self.screening_manifest_checksum,
            "screening_custody_checksum": self.screening_custody_checksum,
            "screening_evidence_checksum": self.screening_evidence_checksum,
            "promotion_decision_checksum": self.promotion_decision_checksum,
            "resource_calibration_checksum": self.resource_calibration_checksum,
            "configuration_execution_manifest_checksum": self.configuration_execution_manifest_checksum,
            "paper_screen_binding_catalog_checksum": self.paper_screen_binding_catalog_checksum,
            "confirmatory_target_configuration_checksum": self.confirmatory_target_configuration_checksum,
            "confirmatory_target_commitment_checksum": self.confirmatory_target_commitment_checksum,
            "final_confirmation_seal_checksum": self.final_confirmation_seal_checksum,
            "prior_target_exposure_inventory_checksum": self.prior_target_exposure_inventory_checksum,
            "pre_seal_chain_head_stage_manifest_checksum": self.pre_seal_chain_head_stage_manifest_checksum,
            "close_screen_operational_paths_checksum": self.close_screen_operational_paths_checksum,
            "artifact_registry_checksum": self.artifact_registry_checksum,
            "confirmatory_configuration_count": self.confirmatory_configuration_count,
            "confirmatory_target_count": self.confirmatory_target_count,
            "confirmatory_optimization_seed_count": self.confirmatory_optimization_seed_count,
            "confirmatory_job_count": self.confirmatory_job_count,
            "held_target_manifest_opened": self.held_target_manifest_opened,
            "held_entropy_opened": self.held_entropy_opened,
            "numerical_execution_performed": self.numerical_execution_performed,
        }

    @property
    def content_checksum(self) -> str:
        """Exact checksum of the readiness receipt."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native receipt data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed readiness JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> WP22HReadinessReceipt:
        """Decode and verify one readiness receipt.

        Returns:
            The strict normalized receipt.

        Raises:
            ValueError: If schema, fields, fixed flags, or checksum differ.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_READINESS_KEYS, name="WP22H readiness receipt")
        if mapping["schema_version"] != WP22H_READINESS_RECEIPT_SCHEMA_VERSION:
            msg = "WP22H readiness receipt uses an unsupported schema version."
            raise ValueError(msg)
        fixed = {
            "held_target_manifest_opened": False,
            "held_entropy_opened": False,
            "numerical_execution_performed": False,
        }
        if any(mapping[name] is not expected for name, expected in fixed.items()):
            msg = "WP22H readiness receipt cannot claim held access or numerical execution."
            raise ValueError(msg)
        receipt = cls(
            **{name: cast("str", mapping[name]) for name in _READINESS_ARTIFACT_CHECKSUM_FIELDS},
            source_commit=cast("str", mapping["source_commit"]),
            confirmatory_configuration_count=cast("int", mapping["confirmatory_configuration_count"]),
            confirmatory_target_count=cast("int", mapping["confirmatory_target_count"]),
            confirmatory_optimization_seed_count=cast("int", mapping["confirmatory_optimization_seed_count"]),
            confirmatory_job_count=cast("int", mapping["confirmatory_job_count"]),
        )
        if mapping["artifact_registry_checksum"] != receipt.artifact_registry_checksum:
            msg = "WP22H logical artifact registry checksum changed during normalization."
            raise ValueError(msg)
        if mapping["content_checksum"] != receipt.content_checksum:
            msg = "WP22H readiness receipt checksum changed during normalization."
            raise ValueError(msg)
        return receipt

    @classmethod
    def from_json(cls, payload: str) -> WP22HReadinessReceipt:
        """Decode canonical readiness JSON.

        Returns:
            The strict normalized readiness receipt.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _validate_resource_custody_roots(
    pilot: ProductionPilotClosure,
    screen: ProductionScreenClosure,
) -> None:
    """Compare the resource authority to its exact custody projections.

    Raises:
        ValueError: If a custody, source, plan, manifest, or pilot-calibration root differs.
    """
    resource = screen.resource_calibration
    pilot_context = pilot.custody.context
    screen_context = screen.custody.context
    screening_manifest = screen_context.screening_manifest
    expected_pilot_custody = canonical_checksum({
        "q6_record_checksums": [item.content_checksum for item in pilot.custody.records if item.job.qubit_count == 6],
    })
    expected_screening_custody = canonical_checksum({
        "record_checksums": [item.content_checksum for item in screen.custody.records],
    })
    exact = (
        (resource.preregistration_checksum, pilot_context.preregistration.content_checksum),
        (resource.execution_source_manifest_checksum, pilot_context.execution_source_manifest.content_checksum),
        (resource.pilot_plan_checksum, pilot_context.plan.content_checksum),
        (resource.pilot_custody_checksum, expected_pilot_custody),
        (resource.pilot_calibration_checksum, pilot.pilot_calibration.content_checksum),
        (resource.screening_plan_checksum, screen_context.plan.content_checksum),
        (
            resource.screening_manifest_checksum,
            None if screening_manifest is None else screening_manifest.content_checksum,
        ),
        (resource.screening_custody_checksum, expected_screening_custody),
    )
    if any(actual != expected for actual, expected in exact):
        msg = "Screen resource calibration differs from the exact pilot and screening custody roots."
        raise ValueError(msg)


def _validate_resource_seal_link(
    resource_calibration: ProductionResourceCalibration,
    final_seal: FinalConfirmationSeal,
) -> None:
    """Bind the caller-visible calibration to the factory-rederived seal root.

    Raises:
        ValueError: If the final seal was derived from another resource calibration.
    """
    if (
        final_seal.primary_resource_budget["reachable_stratum_manifest_checksum"]
        != resource_calibration.content_checksum
    ):
        msg = "Final seal resource authority differs from the supplied production calibration."
        raise ValueError(msg)


def _validate_paper_screen_catalog(
    catalog: RepositoryBindingCatalog,
    pilot: ProductionPilotClosure,
    screen: ProductionScreenClosure,
) -> None:
    """Require the canonical screen registry derived from the pilot authorities.

    Raises:
        TypeError: If the catalog has the wrong runtime schema.
        ValueError: If the catalog, candidates, or context bindings differ.
    """
    if not isinstance(catalog, RepositoryBindingCatalog):
        msg = "paper_screen_binding_catalog must be a RepositoryBindingCatalog."
        raise TypeError(msg)
    context = screen.custody.context
    preregistration = pilot.custody.context.preregistration
    candidates, expected_catalog = build_paper_screen_execution_registry(
        preregistration,
        pilot.sample_size_design,
        pilot.pilot_calibration,
    )
    if catalog.to_json() != expected_catalog.to_json():
        msg = "Paper-screen binding catalog differs from the canonical execution registry."
        raise ValueError(msg)
    manifest = context.screening_manifest
    if manifest is None or tuple(item.screening_ref() for item in candidates) != manifest.candidates:
        msg = "Paper-screen manifest candidates differ from the canonical execution registry."
        raise ValueError(msg)
    execution_profile = getattr(context, "execution_profile", None)
    if execution_profile is not None and (
        execution_profile != catalog.profile or tuple(context.scoped_bindings) != catalog.bindings
    ):
        msg = "Paper-screen custody context differs from its exact binding catalog."
        raise ValueError(msg)


def _validate_confirmatory_public_target(
    configuration: TargetPopulationConfig,
    commitment: TargetPopulationCommitment,
    pilot: ProductionPilotClosure,
) -> None:
    """Cross-check the public configuration and held-manifest commitment only.

    Raises:
        TypeError: If either public target artifact has the wrong schema.
        ValueError: If its role, scope, preregistration, or counts differ.
    """
    if not isinstance(configuration, TargetPopulationConfig):
        msg = "confirmatory_target_configuration must be a TargetPopulationConfig."
        raise TypeError(msg)
    if not isinstance(commitment, TargetPopulationCommitment):
        msg = "confirmatory_target_commitment must be a TargetPopulationCommitment."
        raise TypeError(msg)
    preregistration_checksum = pilot.custody.context.preregistration.content_checksum
    if (
        configuration.preregistration_checksum != preregistration_checksum
        or configuration.data_role != "confirmatory"
        or configuration.population_scope != "primary_q6"
    ):
        msg = "Confirmatory target configuration must be the public primary-q6 protocol commitment."
        raise ValueError(msg)
    allocation_counts: dict[str, int] = {}
    for allocation in configuration.allocations:
        allocation_counts[allocation.family_id] = (
            allocation_counts.get(allocation.family_id, 0) + allocation.instance_count
        )
    expected_counts = dict(pilot.sample_size_design.target_count_by_family)
    if allocation_counts != expected_counts or dict(commitment.target_count_by_family) != expected_counts:
        msg = "Confirmatory public target counts differ from the pilot-derived sample-size design."
        raise ValueError(msg)


def _ceremony_population_artifacts(
    pilot: ProductionPilotClosure,
    screen: ProductionScreenClosure,
) -> tuple[TargetPopulationManifest, TargetPopulationManifest, TargetPopulationManifest]:
    """Resolve the exact nonconfirmatory target manifests without held access.

    Returns:
        Pilot q6, pilot q12, and screening q6 target manifests.

    Raises:
        ValueError: If an exact nonconfirmatory population is absent.
    """
    pilot_manifests = {item.population_scope: item for item in pilot.custody.context.target_manifests}
    screening_manifest = screen.custody.context.screening_manifest
    if screening_manifest is None:
        msg = "Final readiness requires the exact paper-screen manifest."
        raise ValueError(msg)
    screen_manifests = tuple(
        item
        for item in screen.custody.context.target_manifests
        if item.content_checksum == screening_manifest.screening_target_manifest_checksum
    )
    try:
        primary = pilot_manifests["primary_q6"]
        secondary = pilot_manifests["secondary_q12"]
        (screening_targets,) = screen_manifests
    except (KeyError, ValueError):
        msg = "Final readiness lacks one exact pilot or screening target population."
        raise ValueError(msg) from None
    return primary, secondary, screening_targets


def _build_readiness_receipt(
    *,
    pilot: ProductionPilotClosure,
    screen: ProductionScreenClosure,
    paper_screen_binding_catalog: RepositoryBindingCatalog,
    confirmatory_target_configuration: TargetPopulationConfig,
    confirmatory_target_commitment: TargetPopulationCommitment,
    analysis_source_manifest: AnalysisSourceManifest,
    final_seal: FinalConfirmationSeal,
    exposure: PriorTargetExposureInventory,
    resource_calibration: ProductionResourceCalibration,
    pre_seal_chain_head_stage_manifest_checksum: str,
    close_screen_operational_paths_checksum: str,
) -> WP22HReadinessReceipt:
    """Project every ceremony artifact onto one complete readiness receipt.

    Returns:
        The complete checksum-only readiness receipt.

    Raises:
        ValueError: If the nonconfirmatory target universe is incomplete.
    """
    primary, secondary, screening_targets = _ceremony_population_artifacts(pilot, screen)
    pilot_context = pilot.custody.context
    screen_context = screen.custody.context
    source_manifest = pilot_context.execution_source_manifest
    screening_manifest = screen_context.screening_manifest
    if screening_manifest is None:  # pragma: no cover - resolved by _ceremony_population_artifacts
        msg = "Final readiness requires the exact paper-screen manifest."
        raise ValueError(msg)
    configuration_count = len(screen.configuration_execution_manifest.entries)
    target_count = sum(cast("int", value) for value in final_seal.target_count_by_family.values())
    return WP22HReadinessReceipt(
        source_commit=source_manifest.source_commit,
        preregistration_checksum=pilot_context.preregistration.content_checksum,
        execution_source_manifest_checksum=source_manifest.content_checksum,
        analysis_source_manifest_checksum=analysis_source_manifest.content_checksum,
        pilot_plan_checksum=pilot_context.plan.content_checksum,
        pilot_primary_target_manifest_checksum=primary.content_checksum,
        pilot_secondary_target_manifest_checksum=secondary.content_checksum,
        pilot_custody_checksum=resource_calibration.pilot_custody_checksum,
        pilot_secondary_archive_checksum=pilot.custody.secondary_archive_checksum,
        pilot_nuisance_summary_checksum=pilot.nuisance_summary.content_checksum,
        sample_size_design_checksum=pilot.sample_size_design.content_checksum,
        pilot_calibration_checksum=pilot.pilot_calibration.content_checksum,
        screening_plan_checksum=screen_context.plan.content_checksum,
        screening_target_manifest_checksum=screening_targets.content_checksum,
        screening_manifest_checksum=screening_manifest.content_checksum,
        screening_custody_checksum=resource_calibration.screening_custody_checksum,
        screening_evidence_checksum=screen.screening_evidence.content_checksum,
        promotion_decision_checksum=screen.promotion_decision.content_checksum,
        resource_calibration_checksum=resource_calibration.content_checksum,
        configuration_execution_manifest_checksum=screen.configuration_execution_manifest.content_checksum,
        paper_screen_binding_catalog_checksum=paper_screen_binding_catalog.content_checksum,
        confirmatory_target_configuration_checksum=confirmatory_target_configuration.content_checksum,
        confirmatory_target_commitment_checksum=confirmatory_target_commitment.content_checksum,
        final_confirmation_seal_checksum=final_seal.content_checksum,
        prior_target_exposure_inventory_checksum=exposure.content_checksum,
        pre_seal_chain_head_stage_manifest_checksum=pre_seal_chain_head_stage_manifest_checksum,
        close_screen_operational_paths_checksum=close_screen_operational_paths_checksum,
        confirmatory_configuration_count=configuration_count,
        confirmatory_target_count=target_count,
        confirmatory_optimization_seed_count=final_seal.optimization_seed_count,
        confirmatory_job_count=configuration_count * target_count * final_seal.optimization_seed_count,
    )


@dataclass(frozen=True, slots=True)
class ProductionConfirmationReadiness:
    """Complete public WP22H handoff, with no held manifest or entropy."""

    pilot: ProductionPilotClosure
    screen: ProductionScreenClosure
    paper_screen_binding_catalog: RepositoryBindingCatalog
    confirmatory_target_configuration: TargetPopulationConfig
    confirmatory_target_commitment: TargetPopulationCommitment
    analysis_source_manifest: AnalysisSourceManifest
    legacy_evidence_audit: LegacyEvidenceAudit
    final_seal: FinalConfirmationSeal
    prior_target_exposure_inventory: PriorTargetExposureInventory
    pre_seal_chain_head_stage_manifest_checksum: str
    close_screen_operational_paths_checksum: str
    receipt: WP22HReadinessReceipt

    def __post_init__(self) -> None:
        """Require typed public artifacts and a complete in-memory receipt closure.

        Raises:
            TypeError: If an artifact has the wrong runtime schema.
            ValueError: If a projection or receipt root differs.
        """
        typed = (
            (self.pilot, ProductionPilotClosure, "pilot"),
            (self.screen, ProductionScreenClosure, "screen"),
            (self.analysis_source_manifest, AnalysisSourceManifest, "analysis_source_manifest"),
            (self.legacy_evidence_audit, LegacyEvidenceAudit, "legacy_evidence_audit"),
            (self.final_seal, FinalConfirmationSeal, "final_seal"),
            (
                self.prior_target_exposure_inventory,
                PriorTargetExposureInventory,
                "prior_target_exposure_inventory",
            ),
            (self.receipt, WP22HReadinessReceipt, "receipt"),
        )
        for value, expected, name in typed:
            if not isinstance(value, expected):
                msg = f"{name} must be {expected.__name__}."
                raise TypeError(msg)
        object.__setattr__(
            self,
            "pre_seal_chain_head_stage_manifest_checksum",
            require_checksum(
                self.pre_seal_chain_head_stage_manifest_checksum,
                "pre_seal_chain_head_stage_manifest_checksum",
            ),
        )
        object.__setattr__(
            self,
            "close_screen_operational_paths_checksum",
            require_checksum(self.close_screen_operational_paths_checksum, "close_screen_operational_paths_checksum"),
        )
        _validate_paper_screen_catalog(self.paper_screen_binding_catalog, self.pilot, self.screen)
        _validate_confirmatory_public_target(
            self.confirmatory_target_configuration,
            self.confirmatory_target_commitment,
            self.pilot,
        )
        resource_calibration = self.screen.resource_calibration
        _validate_resource_custody_roots(self.pilot, self.screen)
        _validate_resource_seal_link(resource_calibration, self.final_seal)
        if (
            self.final_seal.confirmatory_target_manifest_checksum
            != self.confirmatory_target_commitment.target_manifest_checksum
            or self.prior_target_exposure_inventory.resource_calibration_checksum
            != resource_calibration.content_checksum
        ):
            msg = "WP22H final seal or exposure inventory differs from its public ceremony authority."
            raise ValueError(msg)
        expected_receipt = _build_readiness_receipt(
            pilot=self.pilot,
            screen=self.screen,
            paper_screen_binding_catalog=self.paper_screen_binding_catalog,
            confirmatory_target_configuration=self.confirmatory_target_configuration,
            confirmatory_target_commitment=self.confirmatory_target_commitment,
            analysis_source_manifest=self.analysis_source_manifest,
            final_seal=self.final_seal,
            exposure=self.prior_target_exposure_inventory,
            resource_calibration=resource_calibration,
            pre_seal_chain_head_stage_manifest_checksum=self.pre_seal_chain_head_stage_manifest_checksum,
            close_screen_operational_paths_checksum=self.close_screen_operational_paths_checksum,
        )
        if expected_receipt != self.receipt:
            msg = "WP22H receipt does not close over every supplied public ceremony artifact."
            raise ValueError(msg)


def build_ceremony_resumability_fingerprint(
    execution_source_manifest: ExecutionSourceManifest,
    catalog: RepositoryBindingCatalog,
) -> ResumabilityFingerprint:
    """Project one governed source lock onto a deterministic resume identity.

    Args:
        execution_source_manifest: Clean-HEAD governed source inventory.
        catalog: Exact preset-specific executable binding catalog.

    Returns:
        One source-complete, output-independent resumability fingerprint.

    Raises:
        TypeError: If either input has the wrong exact schema.
    """
    if not isinstance(execution_source_manifest, ExecutionSourceManifest):
        msg = "execution_source_manifest must be an ExecutionSourceManifest."
        raise TypeError(msg)
    if not isinstance(catalog, RepositoryBindingCatalog):
        msg = "catalog must be a RepositoryBindingCatalog."
        raise TypeError(msg)
    role_map = {
        "execution_source": "execution_source",
        "dependency_lock": "lockfile",
        "sealed_input": "sealed_input",
    }
    entries = tuple(
        ExecutionSourceEntry(
            role=cast("Literal['execution_source', 'lockfile', 'sealed_input']", role_map[source.role]),
            repository_path=source.repo_path,
            starting_git_blob_id=source.git_blob_id,
            content_checksum=source.source_checksum,
        )
        for source in execution_source_manifest.source_files
        if source.role in role_map
    )
    prefix_payload = canonical_json({
        "schema_version": "yaqs.state_preparation.phase2.wp22h_pipeline_prefix.v1",
        "source_manifest_checksum": execution_source_manifest.content_checksum,
        "binding_catalog_checksum": catalog.content_checksum,
    }).encode("utf-8")
    return ResumabilityFingerprint(
        starting_commit=execution_source_manifest.source_commit,
        pipeline_prefix_id=f"phase2_pipeline_prefix_{hashlib.sha256(prefix_payload).hexdigest()}",
        dependency_versions={
            "python": platform.python_version(),
            "source_commit": execution_source_manifest.source_commit,
        },
        entries=entries,
    )


def build_ceremony_training_context(
    *,
    preregistration: InitialPreregistration,
    catalog: RepositoryBindingCatalog,
    execution_source_manifest: ExecutionSourceManifest,
    target_configurations: tuple[TargetPopulationConfig, ...],
    target_manifests: tuple[TargetPopulationManifest, ...],
    external_entropy_keyring: ExternalEntropyKeyring,
    resumability_fingerprint: ResumabilityFingerprint,
    screening_manifest: object | None = None,
    sample_size_design: SampleSizeDesign | None = None,
) -> TrainingExecutionContext:
    """Construct the exact pilot or screen context from ceremony artifacts.

    This is the repository-owned counterpart of the path-oriented runner
    loader.  It accepts only already typed, checksum-verified artifacts and
    performs the same target authorization and full plan fingerprint binding.

    Returns:
        A non-serializable source-locked production execution context.

    Raises:
        TypeError: If an input uses the wrong exact runtime schema.
        ValueError: If catalog, target, design, or preset roots disagree.
    """
    typed = (
        (preregistration, InitialPreregistration, "preregistration"),
        (catalog, RepositoryBindingCatalog, "catalog"),
        (execution_source_manifest, ExecutionSourceManifest, "execution_source_manifest"),
        (external_entropy_keyring, ExternalEntropyKeyring, "external_entropy_keyring"),
        (resumability_fingerprint, ResumabilityFingerprint, "resumability_fingerprint"),
    )
    for value, expected, name in typed:
        if not isinstance(value, expected):
            msg = f"{name} must be {expected.__name__}."
            raise TypeError(msg)
    configs = tuple(target_configurations)
    manifests = tuple(target_manifests)
    if not configs or any(not isinstance(item, TargetPopulationConfig) for item in configs):
        msg = "target_configurations must contain TargetPopulationConfig values."
        raise TypeError(msg)
    if not manifests or any(not isinstance(item, TargetPopulationManifest) for item in manifests):
        msg = "target_manifests must contain TargetPopulationManifest values."
        raise TypeError(msg)
    profile = catalog.profile
    if profile.preregistration_checksum != preregistration.content_checksum:
        msg = "Execution profile belongs to another preregistration."
        raise ValueError(msg)
    config_by_checksum = {item.content_checksum: item for item in configs}
    try:
        ordered_configs = tuple(config_by_checksum[item.population_config_checksum] for item in manifests)
    except KeyError:
        msg = "A target manifest has no exact ceremony target configuration."
        raise ValueError(msg) from None
    authorized = tuple(
        AuthorizedTargetMaterialization(
            target_configuration=config,
            target_manifest=manifest,
            authorization=authorize_target_materialization(
                preregistration,
                config,
                manifest,
                external_entropy_keyring.entropy_for(config.data_role, config.population_scope),
            ),
        )
        for config, manifest in zip(ordered_configs, manifests, strict=True)
    )
    if profile.preset == "paper-pilot":
        if screening_manifest is not None or sample_size_design is not None or len(manifests) != 2:
            msg = "paper-pilot ceremony requires two targets and no screen artifacts."
            raise ValueError(msg)
        seeds = derive_pilot_optimization_seeds(
            preregistration.content_checksum,
            PILOT_OPTIMIZATION_SEED_COUNT,
        )
        unbound = build_paper_pilot_plan(
            preregistration_checksum=preregistration.content_checksum,
            target_manifests=manifests,
            optimization_seeds=seeds,
            executable_bindings=catalog.bindings,
        )
        manifest_by_checksum = {item.content_checksum: item for item in manifests}
        manifests = tuple(manifest_by_checksum[item] for item in unbound.target_manifest_checksums)
        config_by_checksum = {item.content_checksum: item for item in ordered_configs}
        ordered_configs = tuple(config_by_checksum[item.population_config_checksum] for item in manifests)
        authorization_by_manifest = {item.target_manifest.content_checksum: item for item in authorized}
        authorized = tuple(authorization_by_manifest[item.content_checksum] for item in manifests)
        screen = None
        cells = ()
    elif profile.preset == "paper-screen":
        if not isinstance(screening_manifest, ScreeningManifest) or not isinstance(
            sample_size_design,
            SampleSizeDesign,
        ):
            msg = "paper-screen ceremony requires typed screening and sample-size artifacts."
            raise TypeError(msg)
        if len(manifests) != 1:
            msg = "paper-screen ceremony requires exactly one q6 screening target manifest."
            raise ValueError(msg)
        unbound = build_paper_screen_plan(
            preregistration_checksum=preregistration.content_checksum,
            target_manifest=manifests[0],
            screening_manifest=screening_manifest,
            executable_bindings=catalog.bindings,
        )
        screen = screening_manifest
        cells = screening_manifest.cells
    else:
        msg = "WP22H constructs only paper-pilot and paper-screen contexts."
        raise ValueError(msg)
    bound = bind_training_plan_fingerprints(
        unbound,
        execution_profile=profile,
        executable_bindings=catalog.bindings,
        target_configurations=ordered_configs,
        target_manifests=manifests,
        execution_source_manifest=execution_source_manifest,
        resumability_fingerprints=(resumability_fingerprint,),
        required_sample_size_design=sample_size_design,
    )
    return TrainingExecutionContext(
        plan=bound,
        execution_profile=profile,
        preregistration=preregistration,
        candidates=candidate_refs_from_bindings(catalog.bindings),
        schedules=schedules_from_bindings(catalog.bindings),
        scoped_bindings=catalog.bindings,
        target_configurations=ordered_configs,
        target_manifests=manifests,
        authorized_materializations=authorized,
        screening_manifest=screen,
        screening_cells=cells,
        required_sample_size_design=sample_size_design,
        execution_source_manifest=execution_source_manifest,
        resumability_fingerprints=(resumability_fingerprint,),
        external_entropy_keyring=external_entropy_keyring,
    )


def frozen_pilot_contrast_bindings(context: TrainingExecutionContext) -> tuple[PilotContrastBinding, ...]:
    """Delegate the two frozen planning contrasts to the canonical registry.

    Returns:
        Noisy versus noiseless and fixed-depth versus layerwise-v2 bindings.

    Raises:
        TypeError: If ``context`` is not a training execution context.
    """
    if not isinstance(context, TrainingExecutionContext):
        msg = "context must be a TrainingExecutionContext."
        raise TypeError(msg)
    return build_paper_pilot_contrast_bindings(context.plan)


def close_production_pilot(
    context: TrainingExecutionContext,
    output_root: Path,
    *,
    design_id: str = "wp22h_cluster_aware_paired_difference_v1",
) -> ProductionPilotClosure:
    """Reopen all 1,080 first attempts and derive the frozen pilot outputs.

    Args:
        context: Exact clean-source-locked paper-pilot context.
        output_root: Existing production pilot result root.
        design_id: Stable identifier for the mechanically derived design.

    Returns:
        The production custody, nuisance summary, design, and screen cap.
    """
    custody = ProductionPilotCustody(context, output_root)
    bindings = frozen_pilot_contrast_bindings(context)
    summary = custody.build_nuisance_summary(bindings)
    design = build_cluster_aware_paired_difference_v1(
        context.preregistration,
        summary,
        design_id=design_id,
    )
    calibration = build_pilot_normalized_compute_calibration(custody)
    return ProductionPilotClosure(
        custody=custody,
        contrast_bindings=bindings,
        nuisance_summary=summary,
        sample_size_design=design,
        pilot_calibration=calibration,
    )


def close_production_screen(
    pilot: ProductionPilotClosure,
    context: TrainingExecutionContext,
    output_root: Path,
) -> ProductionScreenClosure:
    """Reopen all 1,296 screen attempts and derive promotion and resources.

    Args:
        pilot: Exact production pilot closure that calibrated the screen.
        context: Exact paper-screen execution context.
        output_root: Existing production screen result root.

    Returns:
        Authenticated screening custody and all mechanical projections.
    """
    custody = ProductionScreeningCustody.reopen(context, output_root)
    evidence, decision = custody.build_evidence()
    resource_calibration = build_production_resource_calibration(pilot.custody, custody)
    configuration_manifest = build_final_configuration_execution_manifest(custody, decision)
    return ProductionScreenClosure(
        custody=custody,
        screening_evidence=evidence,
        promotion_decision=decision,
        resource_calibration=resource_calibration,
        configuration_execution_manifest=configuration_manifest,
    )


def finalize_confirmation_readiness(
    *,
    pilot: ProductionPilotClosure,
    screen: ProductionScreenClosure,
    paper_screen_binding_catalog: RepositoryBindingCatalog,
    confirmatory_target_configuration: TargetPopulationConfig,
    confirmatory_target_commitment: TargetPopulationCommitment,
    analysis_source_manifest: AnalysisSourceManifest,
    legacy_evidence_audit: LegacyEvidenceAudit,
    repository_root: Path,
    pre_seal_chain_head_stage_manifest_checksum: str,
    close_screen_operational_paths_checksum: str,
) -> ProductionConfirmationReadiness:
    """Create the final seal and readiness receipt without held-target access.

    The only confirmatory inputs accepted here are the public population
    configuration and checksum/count commitment. There is intentionally no
    parameter for a held target manifest, entropy key, instance identifier,
    parameter record, or vector. The preregistration and execution-source
    manifest are derived from authenticated production custody.

    Returns:
        The final seal, prior-exposure inventory, and dormancy receipt.

    Raises:
        TypeError: If a supplied artifact has the wrong runtime schema.
        ValueError: If custody, registry, public target, or source roots differ.
    """
    if not isinstance(pilot, ProductionPilotClosure) or not isinstance(screen, ProductionScreenClosure):
        msg = "pilot and screen must be exact production closure artifacts."
        raise TypeError(msg)
    if not isinstance(analysis_source_manifest, AnalysisSourceManifest):
        msg = "analysis_source_manifest must be an AnalysisSourceManifest."
        raise TypeError(msg)
    if not isinstance(legacy_evidence_audit, LegacyEvidenceAudit):
        msg = "legacy_evidence_audit must be a LegacyEvidenceAudit."
        raise TypeError(msg)
    pilot_context = pilot.custody.context
    screen_context = screen.custody.context
    preregistration = pilot_context.preregistration
    execution_source_manifest = pilot_context.execution_source_manifest
    if (
        screen_context.preregistration.content_checksum != preregistration.content_checksum
        or screen_context.execution_source_manifest.content_checksum != execution_source_manifest.content_checksum
    ):
        msg = "Pilot and screening custody must share one preregistration and execution-source root."
        raise ValueError(msg)
    _validate_paper_screen_catalog(paper_screen_binding_catalog, pilot, screen)
    _validate_confirmatory_public_target(
        confirmatory_target_configuration,
        confirmatory_target_commitment,
        pilot,
    )
    resource_calibration = screen.resource_calibration
    _validate_resource_custody_roots(pilot, screen)
    screening_manifest = screen_context.screening_manifest
    if screening_manifest is None:
        msg = "Final readiness requires the exact paper-screen manifest."
        raise ValueError(msg)
    primary, secondary, screening_targets = _ceremony_population_artifacts(pilot, screen)
    final_seal = create_final_confirmation_seal(
        preregistration=preregistration,
        screening_manifest=screening_manifest,
        promotion_decision=screen.promotion_decision,
        pilot_nuisance_summary=pilot.nuisance_summary,
        sample_size_design=pilot.sample_size_design,
        confirmatory_target_commitment=confirmatory_target_commitment,
        analysis_source_manifest=analysis_source_manifest,
        execution_source_manifest=execution_source_manifest,
        configuration_execution_manifest=screen.configuration_execution_manifest,
        repository_root=repository_root,
        production_screening_custody=screen.custody,
        production_pilot_custody=pilot.custody,
    )
    _validate_resource_seal_link(resource_calibration, final_seal)
    exposure = PriorTargetExposureInventory.create(
        preregistration=preregistration,
        legacy_evidence_audit=legacy_evidence_audit,
        pilot_plan=pilot.custody.context.plan,
        pilot_primary_q6_target_manifest=primary,
        pilot_secondary_q12_target_manifest=secondary,
        screening_plan=screen_context.plan,
        screening_target_manifest=screening_targets,
        screening_manifest=screening_manifest,
        resource_calibration=resource_calibration,
    )
    receipt = _build_readiness_receipt(
        pilot=pilot,
        screen=screen,
        paper_screen_binding_catalog=paper_screen_binding_catalog,
        confirmatory_target_configuration=confirmatory_target_configuration,
        confirmatory_target_commitment=confirmatory_target_commitment,
        analysis_source_manifest=analysis_source_manifest,
        final_seal=final_seal,
        exposure=exposure,
        resource_calibration=resource_calibration,
        pre_seal_chain_head_stage_manifest_checksum=pre_seal_chain_head_stage_manifest_checksum,
        close_screen_operational_paths_checksum=close_screen_operational_paths_checksum,
    )
    return ProductionConfirmationReadiness(
        pilot=pilot,
        screen=screen,
        paper_screen_binding_catalog=paper_screen_binding_catalog,
        confirmatory_target_configuration=confirmatory_target_configuration,
        confirmatory_target_commitment=confirmatory_target_commitment,
        analysis_source_manifest=analysis_source_manifest,
        legacy_evidence_audit=legacy_evidence_audit,
        final_seal=final_seal,
        prior_target_exposure_inventory=exposure,
        pre_seal_chain_head_stage_manifest_checksum=pre_seal_chain_head_stage_manifest_checksum,
        close_screen_operational_paths_checksum=close_screen_operational_paths_checksum,
        receipt=receipt,
    )


def verify_confirmation_readiness(
    readiness: ProductionConfirmationReadiness,
    *,
    execution_source_manifest: ExecutionSourceManifest,
    analysis_source_manifest: AnalysisSourceManifest,
    repository_root: Path,
    pre_seal_chain_head_stage_manifest_checksum: str,
    close_screen_operational_paths_checksum: str,
) -> None:
    """Rebuild every ceremony projection and receipt field before WP23 handoff.

    Raises:
        TypeError: If a readiness input uses the wrong runtime schema.
        ValueError: If a source or checksum alias differs.
    """
    if not isinstance(readiness, ProductionConfirmationReadiness):
        msg = "readiness must be ProductionConfirmationReadiness."
        raise TypeError(msg)
    if not isinstance(execution_source_manifest, ExecutionSourceManifest):
        msg = "execution_source_manifest must be an ExecutionSourceManifest."
        raise TypeError(msg)
    if not isinstance(analysis_source_manifest, AnalysisSourceManifest):
        msg = "analysis_source_manifest must be an AnalysisSourceManifest."
        raise TypeError(msg)
    expected_pre_seal_head = require_checksum(
        pre_seal_chain_head_stage_manifest_checksum,
        "pre_seal_chain_head_stage_manifest_checksum",
    )
    expected_operational_paths = require_checksum(
        close_screen_operational_paths_checksum,
        "close_screen_operational_paths_checksum",
    )
    if (
        readiness.pre_seal_chain_head_stage_manifest_checksum != expected_pre_seal_head
        or readiness.close_screen_operational_paths_checksum != expected_operational_paths
        or readiness.receipt.pre_seal_chain_head_stage_manifest_checksum != expected_pre_seal_head
        or readiness.receipt.close_screen_operational_paths_checksum != expected_operational_paths
    ):
        msg = "WP22H readiness differs from the expected pre-seal chain or operational paths custody."
        raise ValueError(msg)
    custody_source = readiness.pilot.custody.context.execution_source_manifest
    if (
        custody_source.to_json() != execution_source_manifest.to_json()
        or readiness.screen.custody.context.execution_source_manifest.to_json() != execution_source_manifest.to_json()
        or readiness.analysis_source_manifest.to_json() != analysis_source_manifest.to_json()
    ):
        msg = "WP22H verification sources differ from the authenticated custody handoff."
        raise ValueError(msg)
    verify_final_seal_source_lock(
        readiness.final_seal,
        execution_source_manifest,
        analysis_source_manifest,
        repository_root,
    )
    pilot = readiness.pilot
    screen = readiness.screen
    expected_bindings = build_paper_pilot_contrast_bindings(pilot.custody.context.plan)
    if pilot.contrast_bindings != expected_bindings:
        msg = "WP22H pilot contrasts differ from the canonical execution registry."
        raise ValueError(msg)
    reproduced_pilot_calibration = build_pilot_normalized_compute_calibration(pilot.custody)
    if reproduced_pilot_calibration.to_json() != pilot.pilot_calibration.to_json():
        msg = "WP22H pilot calibration differs from exact production custody."
        raise ValueError(msg)
    reproduced_evidence, reproduced_decision = screen.custody.build_evidence(
        evidence_id=screen.screening_evidence.evidence_id,
    )
    reproduced_execution_manifest = build_final_configuration_execution_manifest(
        screen.custody,
        reproduced_decision,
        manifest_id=screen.configuration_execution_manifest.manifest_id,
    )
    screen_projections = (
        (screen.screening_evidence.to_json(), reproduced_evidence.to_json()),
        (screen.promotion_decision.to_json(), reproduced_decision.to_json()),
        (screen.configuration_execution_manifest.to_json(), reproduced_execution_manifest.to_json()),
    )
    if any(actual != expected for actual, expected in screen_projections):
        msg = "WP22H screening projections differ from exact production custody."
        raise ValueError(msg)
    _validate_paper_screen_catalog(readiness.paper_screen_binding_catalog, pilot, screen)
    _validate_confirmatory_public_target(
        readiness.confirmatory_target_configuration,
        readiness.confirmatory_target_commitment,
        pilot,
    )
    resource_calibration = screen.resource_calibration
    _validate_resource_custody_roots(pilot, screen)
    preregistration = pilot.custody.context.preregistration
    screen_context = screen.custody.context
    screening_manifest = screen_context.screening_manifest
    if screening_manifest is None:
        msg = "Final readiness requires the exact paper-screen manifest."
        raise ValueError(msg)
    reproduced_seal = create_final_confirmation_seal(
        preregistration=preregistration,
        screening_manifest=screening_manifest,
        promotion_decision=screen.promotion_decision,
        pilot_nuisance_summary=pilot.nuisance_summary,
        sample_size_design=pilot.sample_size_design,
        confirmatory_target_commitment=readiness.confirmatory_target_commitment,
        analysis_source_manifest=analysis_source_manifest,
        execution_source_manifest=execution_source_manifest,
        configuration_execution_manifest=screen.configuration_execution_manifest,
        repository_root=repository_root,
        production_screening_custody=screen.custody,
        production_pilot_custody=pilot.custody,
        seal_id=readiness.final_seal.seal_id,
    )
    if reproduced_seal.to_json() != readiness.final_seal.to_json():
        msg = "WP22H final confirmation seal differs from its rederived production authority."
        raise ValueError(msg)
    _validate_resource_seal_link(resource_calibration, reproduced_seal)
    primary, secondary, screening_targets = _ceremony_population_artifacts(pilot, screen)
    reproduced_exposure = PriorTargetExposureInventory.create(
        preregistration=preregistration,
        legacy_evidence_audit=readiness.legacy_evidence_audit,
        pilot_plan=pilot.custody.context.plan,
        pilot_primary_q6_target_manifest=primary,
        pilot_secondary_q12_target_manifest=secondary,
        screening_plan=screen_context.plan,
        screening_target_manifest=screening_targets,
        screening_manifest=screening_manifest,
        resource_calibration=resource_calibration,
    )
    if reproduced_exposure.to_json() != readiness.prior_target_exposure_inventory.to_json():
        msg = "WP22H prior-exposure inventory differs from its authenticated nonconfirmatory sources."
        raise ValueError(msg)
    reproduced_receipt = _build_readiness_receipt(
        pilot=pilot,
        screen=screen,
        paper_screen_binding_catalog=readiness.paper_screen_binding_catalog,
        confirmatory_target_configuration=readiness.confirmatory_target_configuration,
        confirmatory_target_commitment=readiness.confirmatory_target_commitment,
        analysis_source_manifest=analysis_source_manifest,
        final_seal=reproduced_seal,
        exposure=reproduced_exposure,
        resource_calibration=resource_calibration,
        pre_seal_chain_head_stage_manifest_checksum=expected_pre_seal_head,
        close_screen_operational_paths_checksum=expected_operational_paths,
    )
    if reproduced_receipt.to_json() != readiness.receipt.to_json():
        msg = "WP22H readiness receipt differs from the reverified production artifact chain."
        raise ValueError(msg)


__all__ = [
    "WP22H_READINESS_RECEIPT_SCHEMA_VERSION",
    "ProductionConfirmationReadiness",
    "ProductionPilotClosure",
    "ProductionScreenClosure",
    "WP22HReadinessReceipt",
    "build_ceremony_resumability_fingerprint",
    "build_ceremony_training_context",
    "close_production_pilot",
    "close_production_screen",
    "finalize_confirmation_readiness",
    "frozen_pilot_contrast_bindings",
    "verify_confirmation_readiness",
]
