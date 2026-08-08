# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bounded exact-artifact support for WP22G confirmatory-study tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Literal, cast

from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.confirmatory_study import PriorTargetExposureInventory
from benchmarks.state_preparation.phase2.legacy import load_legacy_evidence_audit
from benchmarks.state_preparation.phase2.production_executors import (
    AttemptArtifactManifest,
    ConfirmationResourceLimitProof,
    ReopenedProductionResult,
)
from benchmarks.state_preparation.phase2.result_custody import ProductionResultCustody
from benchmarks.state_preparation.phase2.screening import ProductionResourceCalibration
from benchmarks.state_preparation.phase2.screening_design import build_screening_manifest
from benchmarks.state_preparation.phase2.targets import (
    TargetPopulationManifest,
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.training_orchestration import (
    TrainingJob,
    TrainingJobOutcome,
    TrainingRunPlan,
    build_paper_screen_plan,
)
from tests.benchmarks.test_state_preparation_wp22_primary_analysis import (
    _real_confirmation_custody as _primary_analysis_confirmation_custody,
)
from tests.benchmarks.wp22_confirmation_test_support import _catalog
from tests.benchmarks.wp22_pilot_test_support import pilot_context

if TYPE_CHECKING:
    from benchmarks.state_preparation.phase2.execution_context import ConfirmationExecutionContext
    from benchmarks.state_preparation.phase2.protocol import InitialPreregistration, ScreeningManifest

_SCREENING_MASTER = bytes((index * 17 + 3) % 256 for index in range(32))


@dataclass(frozen=True, slots=True)
class PriorExposureFixture:
    """Exact pilot/screen artifacts plus their derived exposure inventory."""

    preregistration: InitialPreregistration
    screening_target_manifest: TargetPopulationManifest
    screening_plan: TrainingRunPlan
    screening_manifest: ScreeningManifest
    inventory: PriorTargetExposureInventory


def _set_frozen_slot(instance: object, name: str, value: object) -> None:
    """Set one slot for a deliberately bounded exact-type test seam."""
    object.__setattr__(instance, name, value)  # noqa: PLC2801 -- test-only exact-type seam


def _opaque_resource_calibration(
    preregistration_checksum: str,
    pilot_plan: TrainingRunPlan,
    screening_plan: TrainingRunPlan,
    screening_manifest: ScreeningManifest,
    *,
    resource_calibration_checksum: str,
    execution_source_manifest_checksum: str,
) -> ProductionResourceCalibration:
    """Project only authenticated roots through an exact-type calibration seam.

    The production constructor's 2,016 resource projections are covered by its
    own tests.  This seam exists only to test that WP22G accepts the concrete
    type and copies no caller-authored summaries.

    Returns:
        The bounded exact-type calibration root fixture.
    """
    calibration = object.__new__(ProductionResourceCalibration)
    values = {
        "preregistration_checksum": preregistration_checksum,
        "execution_source_manifest_checksum": execution_source_manifest_checksum,
        "pilot_plan_checksum": pilot_plan.content_checksum,
        "pilot_custody_checksum": canonical_checksum({"pilot custody": pilot_plan.content_checksum}),
        "pilot_calibration_checksum": canonical_checksum({"pilot calibration": pilot_plan.content_checksum}),
        "screening_plan_checksum": screening_plan.content_checksum,
        "screening_manifest_checksum": screening_manifest.content_checksum,
        "screening_custody_checksum": canonical_checksum({"screen custody": screening_plan.content_checksum}),
        "_content_checksum": resource_calibration_checksum,
    }
    for name, value in values.items():
        _set_frozen_slot(calibration, name, value)
    return calibration


@cache
def prior_exposure_fixture(
    *,
    resource_calibration_checksum: str,
    execution_source_manifest_checksum: str,
) -> PriorExposureFixture:
    """Build the exact 1,080-pilot/1,296-screen prior exposure universe.

    Returns:
        Cached exact plans, target manifests, screening manifest, and inventory.
    """
    preregistration, pilot_q6, pilot_q12, pilot_plan, _bindings = pilot_context()
    catalog, candidates = _catalog()
    config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(_SCREENING_MASTER),
        population_scope="primary_q6",
    )
    target_manifest = create_target_population_manifest(config, preregistration, _SCREENING_MASTER)
    screening_manifest = build_screening_manifest(
        preregistration,
        target_manifest,
        candidates,
        optimization_seeds=(10, 20, 30),
        screening_seed_root=42,
    )
    screening_plan = build_paper_screen_plan(
        preregistration_checksum=preregistration.content_checksum,
        target_manifest=target_manifest,
        screening_manifest=screening_manifest,
        executable_bindings=catalog.bindings,
    )
    calibration = _opaque_resource_calibration(
        preregistration.content_checksum,
        pilot_plan,
        screening_plan,
        screening_manifest,
        resource_calibration_checksum=resource_calibration_checksum,
        execution_source_manifest_checksum=execution_source_manifest_checksum,
    )
    inventory = PriorTargetExposureInventory.create(
        preregistration=preregistration,
        legacy_evidence_audit=load_legacy_evidence_audit(),
        pilot_plan=pilot_plan,
        pilot_primary_q6_target_manifest=pilot_q6,
        pilot_secondary_q12_target_manifest=pilot_q12,
        screening_plan=screening_plan,
        screening_target_manifest=target_manifest,
        screening_manifest=screening_manifest,
        resource_calibration=calibration,
    )
    return PriorExposureFixture(
        preregistration=preregistration,
        screening_target_manifest=target_manifest,
        screening_plan=screening_plan,
        screening_manifest=screening_manifest,
        inventory=inventory,
    )


def _production_document(document_type: str, payload: object) -> dict[str, object]:
    """Reconstruct one exact typed production document wrapper.

    Returns:
        The checksum-sealed document used by the bounded reopened-result seam.
    """
    content = {
        "schema_version": "yaqs.state_preparation.phase2.production_document.v1",
        "document_type": document_type,
        "payload": payload,
    }
    return {**content, "content_checksum": canonical_checksum(content)}


def _reopened_from_custody(custody: ProductionResultCustody) -> ReopenedProductionResult:
    """Recover a bounded exact-type reopened-result seam from test custody.

    The production reopen path is tested separately.  The opaque manifest is
    used here only because the shared primary-analysis fixture returned its
    already-derived custody projection rather than its in-memory reopened
    source.

    Returns:
        A typed reopened result that reproduces the exact custody projection.
    """
    manifest = object.__new__(AttemptArtifactManifest)
    _set_frozen_slot(manifest, "attempt", 1)
    raw = (
        None
        if custody.raw_trajectory_payload is None
        else _production_document("raw_trajectory_fidelities", custody.raw_trajectory_payload)
    )
    resources = _production_document("runtime_resources", custody.resource_payload)
    return ReopenedProductionResult(
        reference=custody.reference,
        manifest=manifest,
        evidence=custody.production_evidence,
        raw_trajectory=raw,
        resources=resources,
        scheduled_map_evidence=(),
        diagnostic_documents=(),
    )


def terminal_confirmation_custody(
    job: TrainingJob,
    context: ConfirmationExecutionContext,
    status: Literal["success", "failure"],
    *,
    trajectory_count: int = 256,
    failure_evidence_exception_type: str = "RuntimeError",
) -> tuple[TrainingJobOutcome, ReopenedProductionResult]:
    """Build bounded authenticated first-attempt confirmation custody.

    Returns:
        The exact outer outcome and bounded reopened production result.

    Raises:
        ValueError: If resource-limit custody lacks a confirmatory request.
    """
    values = (0.75,) * trajectory_count if status == "success" else None
    custody, _incomplete_root = _primary_analysis_confirmation_custody(job, context, values)
    if status == "failure" and failure_evidence_exception_type != "RuntimeError":
        request = job.confirm_execution_request
        if request is None:
            msg = "Resource-limit custody requires a confirmatory request."
            raise ValueError(msg)
        normalized_cap = cast("float", request.primary_resource_budget["normalized_compute_cap"])
        proof = ConfirmationResourceLimitProof(
            request_checksum=request.content_checksum,
            proof_kind="prospective_normalized_work",
            normalized_compute_cap=normalized_cap,
            native_edge_gate_cap=cast("float", request.primary_resource_budget["cap_per_chain_edge"]),
            completed_normalized_work=normalized_cap,
            prospective_normalized_work=1.0,
            measured_normalized_work=None,
            measured_native_edge_gate_counts=(),
            measurement_resource_ref=None,
            exceeded_dimensions=("normalized_work",),
        )
        evidence = replace(
            custody.production_evidence,
            failure={
                "phase": "production_execution",
                "exception_type": failure_evidence_exception_type,
                "message": "redacted test resource limit",
                "resource_limit_proof": proof.to_dict(),
            },
        )
        reference = replace(custody.reference, evidence_checksum=evidence.content_checksum)
        replaced_custody = object.__new__(ProductionResultCustody)
        values_by_slot = {
            "reference": reference,
            "production_evidence": evidence,
            "result_evidence_checksum": evidence.content_checksum,
            "raw_trajectory_payload": custody.raw_trajectory_payload,
            "raw_trajectory_document_checksum": custody.raw_trajectory_document_checksum,
            "resource_payload": custody.resource_payload,
            "resource_document_checksum": custody.resource_document_checksum,
            "pilot_diagnostics": custody.pilot_diagnostics,
            "schema_version": custody.schema_version,
        }
        for name, value in values_by_slot.items():
            _set_frozen_slot(replaced_custody, name, value)
        custody = replaced_custody
    outcome = TrainingJobOutcome(
        job_checksum=job.content_checksum,
        status=status,
        result_artifact_checksum=custody.reference.content_checksum if status == "success" else None,
        exception_type=(
            None
            if status == "success"
            else "confirmation-resource-limit-error"
            if failure_evidence_exception_type == "ConfirmationResourceLimitError"
            else "normalized-compute-cap-error"
            if failure_evidence_exception_type == "NormalizedComputeCapError"
            else "runtime-error"
        ),
        message=None if status == "success" else "bounded authenticated failure",
        attempt=1,
    )
    return outcome, _reopened_from_custody(custody)


__all__ = ["PriorExposureFixture", "prior_exposure_fixture", "terminal_confirmation_custody"]
