# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bounded lifecycle tests for the non-numerical WP22H ceremony."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.execution_registry import build_paper_screen_execution_registry
from benchmarks.state_preparation.phase2.legacy import load_legacy_evidence_audit
from benchmarks.state_preparation.phase2.operational_ceremony import (
    ProductionPilotClosure,
    ProductionScreenClosure,
    WP22HReadinessReceipt,
    finalize_confirmation_readiness,
    verify_confirmation_readiness,
)
from benchmarks.state_preparation.phase2.pilot import build_cluster_aware_paired_difference_v1
from benchmarks.state_preparation.phase2.protocol import load_initial_preregistration
from benchmarks.state_preparation.phase2.screening import (
    ProductionScreeningCustody,
    build_final_configuration_execution_manifest,
    build_pilot_normalized_compute_calibration,
    build_production_resource_calibration,
    build_screening_manifest,
)
from benchmarks.state_preparation.phase2.targets import (
    TargetPopulationCommitment,
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.training_orchestration import TrainingRunPlan
from tests.benchmarks.wp22_confirmation_test_support import _create_source_repository
from tests.benchmarks.wp22_pilot_test_support import (
    build_pilot_summary,
    pilot_observations,
    production_pilot_custody_fixture,
)
from tests.benchmarks.wp22_screening_test_support import (
    production_screening_custody_fixture,
)

if TYPE_CHECKING:
    from pathlib import Path


def _checksum(label: str) -> str:
    """Return one stable checksum for a focused receipt field."""
    return canonical_checksum({"wp22h_test": label})


def _receipt() -> WP22HReadinessReceipt:
    """Return a minimal internally consistent readiness receipt."""
    checksums = {
        name: _checksum(name)
        for name in (
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
    }
    return WP22HReadinessReceipt(
        source_commit="1" * 40,
        **checksums,
        confirmatory_configuration_count=2,
        confirmatory_target_count=96,
        confirmatory_optimization_seed_count=3,
        confirmatory_job_count=576,
    )


def _replace_bounded_screen_plan_with_typed_plan(
    custody: ProductionScreeningCustody,
    *,
    preregistration_checksum: str,
    target_manifest_checksum: str,
    screening_manifest_checksum: str,
    execution_source_checksum: str,
    sample_size_design_checksum: str,
) -> ProductionScreeningCustody:
    """Upgrade the bounded aggregate's namespace plan to the exact plan schema.

    Returns:
        Equivalent custody whose ordered records are owned by a TrainingRunPlan.
    """
    records = tuple(sorted(custody.records, key=lambda item: item.job.sort_key))
    jobs = tuple(item.job for item in records)

    def roots(name: str) -> tuple[str, ...]:
        return tuple(sorted({value for job in jobs if (value := getattr(job, name)) is not None}))

    plan = TrainingRunPlan(
        plan_id="wp22_paper_screen_v1",
        preset="paper-screen",
        preregistration_checksum=preregistration_checksum,
        target_manifest_checksums=(target_manifest_checksum,),
        screening_manifest_checksum=screening_manifest_checksum,
        final_confirmation_seal_checksum=None,
        execution_source_checksum=execution_source_checksum,
        jobs=jobs,
        execution_profile_checksum=jobs[0].execution_profile_checksum,
        scoped_binding_checksums=roots("scoped_binding_checksum"),
        executable_binding_checksums=roots("executable_binding_checksum"),
        implementation_checksums=roots("implementation_checksum"),
        evaluation_policy_checksums=roots("evaluation_policy_checksum"),
        target_configuration_checksums=roots("target_configuration_checksum"),
        source_fingerprint_checksums=roots("source_fingerprint_checksum"),
        scheduled_execution_program_checksums=roots("scheduled_execution_program_checksum"),
        sample_size_design_checksum=sample_size_design_checksum,
    )
    object.__setattr__(custody.context, "plan", plan)  # noqa: PLC2801 - bounded frozen fixture replacement
    return ProductionScreeningCustody(custody.context, records)


def test_readiness_receipt_roundtrip_rejects_tamper_and_scientific_activity() -> None:
    """The handoff receipt is strict, canonical, dormant, and count-derived."""
    receipt = _receipt()

    assert WP22HReadinessReceipt.from_json(receipt.to_json()) == receipt
    assert receipt.held_target_manifest_opened is False
    assert receipt.held_entropy_opened is False
    assert receipt.numerical_execution_performed is False

    tampered = receipt.to_dict()
    tampered["held_entropy_opened"] = True
    tampered["content_checksum"] = canonical_checksum({
        key: value for key, value in tampered.items() if key != "content_checksum"
    })
    with pytest.raises(ValueError, match="cannot claim held access"):
        WP22HReadinessReceipt.from_dict(tampered)
    with pytest.raises(ValueError, match="configurations times targets"):
        replace(receipt, confirmatory_job_count=575)
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        replace(receipt, pilot_plan_checksum=f"sha256:{'z' * 64}")
    registry_tamper = receipt.to_dict()
    registry_tamper["artifact_registry_checksum"] = _checksum("foreign logical registry")
    registry_tamper["content_checksum"] = canonical_checksum({
        key: value for key, value in registry_tamper.items() if key != "content_checksum"
    })
    with pytest.raises(ValueError, match="logical artifact registry"):
        WP22HReadinessReceipt.from_dict(registry_tamper)


def test_bounded_full_grid_finalization_emits_reverified_no_held_read_receipt(
    tmp_path: Path,
) -> None:
    """Exact typed pilot/screen grids derive one source-locked WP23 handoff."""
    preregistration = load_initial_preregistration()
    execution_source, analysis_source = _create_source_repository(tmp_path / "source")
    pilot_custody = production_pilot_custody_fixture(
        tmp_path / "pilot",
        execution_source_manifest=execution_source,
    )
    pilot_summary = build_pilot_summary(pilot_observations())
    design = build_cluster_aware_paired_difference_v1(
        preregistration,
        pilot_summary,
        design_id="wp22h_bounded_design",
    )
    pilot_calibration = build_pilot_normalized_compute_calibration(pilot_custody)
    pilot = ProductionPilotClosure(
        custody=pilot_custody,
        contrast_bindings=pilot_summary.contrast_bindings,
        nuisance_summary=pilot_summary,
        sample_size_design=design,
        pilot_calibration=pilot_calibration,
    )

    screening_master = bytes(reversed(range(32)))
    screening_config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(screening_master),
        population_scope="primary_q6",
    )
    screening_targets = create_target_population_manifest(
        screening_config,
        preregistration,
        screening_master,
    )
    screen_candidates, paper_screen_catalog = build_paper_screen_execution_registry(
        preregistration,
        design,
        pilot_calibration,
    )
    screening_manifest = build_screening_manifest(
        preregistration,
        screening_targets,
        screen_candidates,
        optimization_seeds=(101, 202, 303),
        screening_seed_root=10_000,
        manifest_id="wp22h_bounded_screen",
    )
    screening_custody = production_screening_custody_fixture(
        preregistration,
        screening_manifest,
        screening_targets,
        design,
        execution_source,
        normalized_compute_cap=pilot_calibration.normalized_compute_cap,
    )
    screening_custody = _replace_bounded_screen_plan_with_typed_plan(
        screening_custody,
        preregistration_checksum=preregistration.content_checksum,
        target_manifest_checksum=screening_targets.content_checksum,
        screening_manifest_checksum=screening_manifest.content_checksum,
        execution_source_checksum=execution_source.content_checksum,
        sample_size_design_checksum=design.content_checksum,
    )
    evidence, promotion = screening_custody.build_evidence()
    resources = build_production_resource_calibration(pilot_custody, screening_custody)
    configuration_manifest = build_final_configuration_execution_manifest(screening_custody, promotion)
    screen = ProductionScreenClosure(
        custody=screening_custody,
        screening_evidence=evidence,
        promotion_decision=promotion,
        resource_calibration=resources,
        configuration_execution_manifest=configuration_manifest,
    )
    target_commitment = TargetPopulationCommitment(
        target_manifest_checksum=_checksum("externally held confirmatory target manifest"),
        target_count_by_family=design.target_count_by_family,
    )
    confirmatory_config = build_target_population_config(
        preregistration,
        "confirmatory",
        role_master_entropy_commitment=_checksum("externally held confirmatory role master entropy"),
        confirmatory_target_count_by_family=design.target_count_by_family,
    )
    pre_seal_head_checksum = _checksum("ordinal-one pre-seal chain head")
    close_screen_paths_checksum = _checksum("close-screen operational paths")

    readiness = finalize_confirmation_readiness(
        pilot=pilot,
        screen=screen,
        paper_screen_binding_catalog=paper_screen_catalog,
        confirmatory_target_configuration=confirmatory_config,
        confirmatory_target_commitment=target_commitment,
        analysis_source_manifest=analysis_source,
        legacy_evidence_audit=load_legacy_evidence_audit(),
        repository_root=tmp_path / "source",
        pre_seal_chain_head_stage_manifest_checksum=pre_seal_head_checksum,
        close_screen_operational_paths_checksum=close_screen_paths_checksum,
    )

    assert WP22HReadinessReceipt.from_json(readiness.receipt.to_json()) == readiness.receipt
    assert readiness.receipt.confirmatory_job_count == (
        readiness.receipt.confirmatory_configuration_count
        * readiness.receipt.confirmatory_target_count
        * readiness.receipt.confirmatory_optimization_seed_count
    )
    assert readiness.receipt.paper_screen_binding_catalog_checksum == paper_screen_catalog.content_checksum
    assert readiness.receipt.confirmatory_target_configuration_checksum == confirmatory_config.content_checksum
    assert readiness.receipt.artifact_registry_checksum.startswith("sha256:")
    assert readiness.receipt.confirmatory_target_commitment_checksum == target_commitment.content_checksum
    assert readiness.receipt.pre_seal_chain_head_stage_manifest_checksum == pre_seal_head_checksum
    assert readiness.receipt.close_screen_operational_paths_checksum == close_screen_paths_checksum
    assert readiness.receipt.held_target_manifest_opened is False
    assert readiness.receipt.held_entropy_opened is False
    assert readiness.receipt.numerical_execution_performed is False
    verify_confirmation_readiness(
        readiness,
        execution_source_manifest=execution_source,
        analysis_source_manifest=analysis_source,
        repository_root=tmp_path / "source",
        pre_seal_chain_head_stage_manifest_checksum=pre_seal_head_checksum,
        close_screen_operational_paths_checksum=close_screen_paths_checksum,
    )
    with pytest.raises(ValueError, match="expected pre-seal chain"):
        verify_confirmation_readiness(
            readiness,
            execution_source_manifest=execution_source,
            analysis_source_manifest=analysis_source,
            repository_root=tmp_path / "source",
            pre_seal_chain_head_stage_manifest_checksum=_checksum("foreign pre-seal head"),
            close_screen_operational_paths_checksum=close_screen_paths_checksum,
        )
    with pytest.raises(ValueError, match="receipt does not close"):
        replace(
            readiness,
            receipt=replace(readiness.receipt, pilot_plan_checksum=_checksum("foreign pilot plan")),
        )

    split_resource = replace(
        resources,
        screening_custody_checksum=_checksum("foreign screening custody"),
    )
    split_screen = ProductionScreenClosure(
        custody=screening_custody,
        screening_evidence=evidence,
        promotion_decision=promotion,
        resource_calibration=split_resource,
        configuration_execution_manifest=configuration_manifest,
    )
    with pytest.raises(ValueError, match="differs from the exact pilot and screening custody"):
        finalize_confirmation_readiness(
            pilot=pilot,
            screen=split_screen,
            paper_screen_binding_catalog=paper_screen_catalog,
            confirmatory_target_configuration=confirmatory_config,
            confirmatory_target_commitment=target_commitment,
            analysis_source_manifest=analysis_source,
            legacy_evidence_audit=load_legacy_evidence_audit(),
            repository_root=tmp_path / "source",
            pre_seal_chain_head_stage_manifest_checksum=pre_seal_head_checksum,
            close_screen_operational_paths_checksum=close_screen_paths_checksum,
        )
