# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused non-numerical tests for WP22G confirmatory-study custody."""

from __future__ import annotations

import copy
from dataclasses import fields, replace
from inspect import signature
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.confirmatory_study import (
    ConfirmatoryPairabilityRecord,
    LockedConfirmatoryStudyManifest,
    LockedConfirmatoryStudyRow,
    PriorTargetExposureInventory,
    validate_confirmatory_novelty,
)
from tests.benchmarks.wp22_confirmation_test_support import (
    ConfirmationContextFixture,
    build_confirmation_context_fixture,
)
from tests.benchmarks.wp22g_confirmatory_study_test_support import (
    PriorExposureFixture,
    prior_exposure_fixture,
    terminal_confirmation_custody,
)

if TYPE_CHECKING:
    from _pytest.tmpdir import TempPathFactory

    from benchmarks.state_preparation.phase2.execution_context import ConfirmationExecutionContext
    from benchmarks.state_preparation.phase2.production_executors import ReopenedProductionResult
    from benchmarks.state_preparation.phase2.targets import TargetPopulationManifest
    from benchmarks.state_preparation.phase2.training_orchestration import TrainingJobOutcome


def _rebuild_inventory(
    inventory: PriorTargetExposureInventory,
    **changes: object,
) -> PriorTargetExposureInventory:
    """Re-run inventory invariants without reparsing 2,376 embedded jobs.

    Returns:
        The rebuilt inventory if all changed values remain valid.
    """
    values = {item.name: getattr(inventory, item.name) for item in fields(inventory) if item.name != "schema_version"}
    values.update(changes)
    return PriorTargetExposureInventory._build(**values)  # noqa: SLF001 -- adversarial invariant test


def _rebuild_study(
    study: LockedConfirmatoryStudyManifest,
    **changes: object,
) -> LockedConfirmatoryStudyManifest:
    """Re-run study invariants without reparsing every embedded source artifact.

    Returns:
        The rebuilt study if all changed values remain valid.
    """
    values = {item.name: getattr(study, item.name) for item in fields(study) if item.name != "schema_version"}
    values.update(changes)
    return LockedConfirmatoryStudyManifest._build(**values)  # noqa: SLF001 -- adversarial invariant test


def _rebuild_row(
    row: LockedConfirmatoryStudyRow,
    **changes: object,
) -> LockedConfirmatoryStudyRow:
    """Re-run row invariants with adversarial custody changes.

    Returns:
        The rebuilt row if all changed values remain valid.
    """
    values = {item.name: getattr(row, item.name) for item in fields(row) if item.name != "schema_version"}
    values.update(changes)
    return LockedConfirmatoryStudyRow._build(**values)  # noqa: SLF001 -- adversarial invariant test


def _operational_study(
    context: ConfirmationExecutionContext,
    exposure_inventory: PriorTargetExposureInventory,
    *,
    outcomes_by_job: dict[str, TrainingJobOutcome] | None = None,
    reopened_results_by_job: dict[str, ReopenedProductionResult] | None = None,
) -> LockedConfirmatoryStudyManifest:
    """Invoke the store-only builder with authenticated reopened-result fixtures.

    Returns:
        The fully derived terminal or incomplete study manifest.
    """
    return LockedConfirmatoryStudyManifest._from_authenticated_reopened_results(  # noqa: SLF001
        context=context,
        exposure_inventory=exposure_inventory,
        outcomes_by_job={} if outcomes_by_job is None else outcomes_by_job,
        reopened_results_by_job={} if reopened_results_by_job is None else reopened_results_by_job,
    )


@pytest.fixture(scope="module")
def base_confirmation(tmp_path_factory: TempPathFactory) -> ConfirmationContextFixture:
    """Return the generic source-locked confirmation support context."""
    return build_confirmation_context_fixture(tmp_path_factory.mktemp("wp22g-confirmation"))


@pytest.fixture(scope="module")
def prior(base_confirmation: ConfirmationContextFixture) -> PriorExposureFixture:
    """Return the exact cached pilot/screen exposure universe."""
    seal = base_confirmation.context.final_seal
    return prior_exposure_fixture(
        resource_calibration_checksum=cast(
            "str",
            seal.primary_resource_budget["reachable_stratum_manifest_checksum"],
        ),
        execution_source_manifest_checksum=seal.execution_source_checksum,
    )


@pytest.fixture(scope="module")
def confirmation(
    base_confirmation: ConfirmationContextFixture,
    prior: PriorExposureFixture,
) -> ConfirmationContextFixture:
    """Return a real source-locked, unmaterialized confirmatory context."""
    context = replace(
        base_confirmation.context,
        prior_target_exposure_inventory_checksum=prior.inventory.content_checksum,
    )
    return replace(base_confirmation, context=context)


def test_prior_exposure_inventory_closes_exact_plan_and_target_universes(
    prior: PriorExposureFixture,
) -> None:
    """Inventory embeds exact prior plans/manifests and canonical legacy sources."""
    inventory = prior.inventory
    assert len(inventory.pilot_plan.jobs) == 1_080
    assert len(inventory.screening_plan.jobs) == 1_296
    assert inventory.screening_plan.screening_manifest_checksum == inventory.screening_manifest.content_checksum
    assert (
        inventory.screening_manifest.screening_target_manifest_checksum
        == inventory.screening_target_manifest.content_checksum
    )
    assert inventory.pilot_plan.target_manifest_checksums == (
        inventory.pilot_primary_q6_target_manifest.content_checksum,
        inventory.pilot_secondary_q12_target_manifest.content_checksum,
    )
    assert sum(item.source_kind == "phase_i_fixture" for item in inventory.canonical_legacy_exposures) == 18
    assert sum(item.source_kind == "legacy_reproduction" for item in inventory.canonical_legacy_exposures) == 5
    assert inventory.content_checksum.startswith("sha256:")


def test_prior_exposure_inventory_rejects_missing_duplicate_reordered_and_tampered_sources(
    prior: PriorExposureFixture,
) -> None:
    """Strict decoding rejects incomplete, duplicate, reordered, or resealed roots."""
    missing = prior.inventory.to_dict()
    del missing["pilot_plan"]
    with pytest.raises(ValueError, match="schema"):
        PriorTargetExposureInventory.from_dict(missing)

    exposures = list(prior.inventory.canonical_legacy_exposures)
    exposures[-1] = exposures[-2]
    with pytest.raises(ValueError, match="missing, duplicated, reordered, or changed"):
        _rebuild_inventory(prior.inventory, canonical_legacy_exposures=tuple(exposures))

    with pytest.raises(ValueError, match="missing, duplicated, reordered, or changed"):
        _rebuild_inventory(
            prior.inventory,
            canonical_legacy_exposures=tuple(reversed(prior.inventory.canonical_legacy_exposures)),
        )

    tampered = prior.inventory.to_dict()
    tampered["screening_custody_checksum"] = canonical_checksum({"foreign": "custody"})
    with pytest.raises(ValueError, match="checksum"):
        PriorTargetExposureInventory.from_dict(tampered)


def _manifest_with_adversarial_spec(
    manifest: TargetPopulationManifest,
    *,
    target_instance_id: str | None = None,
    instance_seed: str | None = None,
) -> TargetPopulationManifest:
    """Return an exact-type post-construction corruption seam for novelty tests."""
    attacked = copy.copy(manifest)
    first = copy.copy(manifest.instances[0])
    if target_instance_id is not None:
        object.__setattr__(first, "target_instance_id", target_instance_id)  # noqa: PLC2801
    if instance_seed is not None:
        object.__setattr__(first, "instance_seed", instance_seed)  # noqa: PLC2801
    object.__setattr__(attacked, "instances", (first, *manifest.instances[1:]))  # noqa: PLC2801
    return attacked


def test_novelty_gate_rejects_exposed_id_and_seed_before_materialization(
    prior: PriorExposureFixture,
    confirmation: ConfirmationContextFixture,
) -> None:
    """Public manifest identifiers and instance seeds are rejected independently."""
    manifest = confirmation.context.target_manifest
    validate_confirmatory_novelty(prior.inventory, manifest)
    exposed = prior.inventory.pilot_primary_q6_target_manifest.instances[0]
    reused_id = _manifest_with_adversarial_spec(manifest, target_instance_id=exposed.target_instance_id)
    with pytest.raises(ValueError, match="target_instance_id"):
        validate_confirmatory_novelty(prior.inventory, reused_id)
    reused_seed = _manifest_with_adversarial_spec(manifest, instance_seed=exposed.instance_seed)
    with pytest.raises(ValueError, match="instance_seed"):
        validate_confirmatory_novelty(prior.inventory, reused_seed)


def test_incomplete_study_has_one_canonical_row_per_job_and_all_pairability_blocks(
    prior: PriorExposureFixture,
    confirmation: ConfirmationContextFixture,
) -> None:
    """Unattempted construction preserves full job and contrast block universes."""
    context = confirmation.context
    assert tuple(signature(LockedConfirmatoryStudyManifest.create).parameters) == (
        "context",
        "exposure_inventory",
    )
    study = LockedConfirmatoryStudyManifest.create(context=context, exposure_inventory=prior.inventory)
    assert study.status == "incomplete"
    assert study.planned_job_count == len(context.plan.jobs) == 576
    assert study.terminal_job_count == study.successful_job_count == study.failed_job_count == 0
    assert study.unattempted_job_count == 576
    assert study.planned_test_trajectory_count == 576 * 256
    assert study.observed_test_trajectory_count == 0
    assert tuple(row.job_checksum for row in study.rows) == tuple(job.content_checksum for job in context.plan.jobs)
    assert all(row.terminal_state == "unattempted" for row in study.rows)
    expected_pairs = (
        len(context.final_seal.primary_contrasts)
        * len(context.target_manifest.instances)
        * context.final_seal.optimization_seed_count
    )
    assert len(study.pairability_records) == expected_pairs == 288
    assert tuple(record.sort_key for record in study.pairability_records) == tuple(
        sorted(record.sort_key for record in study.pairability_records)
    )
    assert all(record.actual_stream_mode == "independent" for record in study.pairability_records)
    assert all(
        record.treatment_evaluation_seed != record.control_evaluation_seed for record in study.pairability_records
    )
    assert all(record.event_level_test_coupling is None for record in study.pairability_records)
    screened_by_configuration = {
        candidate.configuration_checksum: candidate for candidate in prior.inventory.screening_manifest.candidates
    }
    assert all(
        record.paired_block.resource_stratum_id
        == screened_by_configuration[record.treatment_configuration_checksum].resource_stratum_id
        == screened_by_configuration[record.control_configuration_checksum].resource_stratum_id
        for record in study.pairability_records
    )
    assert (
        ConfirmatoryPairabilityRecord.from_dict(study.pairability_records[0].to_dict())
        == (study.pairability_records[0])
    )
    foreign_context = replace(
        context,
        prior_target_exposure_inventory_checksum=canonical_checksum({"foreign": "exposure inventory"}),
    )
    with pytest.raises(ValueError, match="novelty root bound to the confirmation session"):
        LockedConfirmatoryStudyManifest.create(
            context=foreign_context,
            exposure_inventory=prior.inventory,
        )

    foreign_calibration = _rebuild_inventory(
        prior.inventory,
        resource_calibration_checksum=canonical_checksum({"foreign": "resource calibration"}),
    )
    with pytest.raises(ValueError, match="exact sealed confirmatory plan"):
        _rebuild_study(study, exposure_inventory=foreign_calibration)
    foreign_calibration_source = _rebuild_inventory(
        prior.inventory,
        resource_calibration_execution_source_checksum=canonical_checksum({"foreign": "execution source"}),
    )
    with pytest.raises(ValueError, match="exact sealed confirmatory plan"):
        _rebuild_study(study, exposure_inventory=foreign_calibration_source)

    contrast = context.final_seal.primary_contrasts[0]
    attacked_candidates = list(prior.inventory.screening_manifest.candidates)
    attacked_index = next(
        index
        for index, candidate in enumerate(attacked_candidates)
        if candidate.configuration_checksum == contrast.control_configuration_checksum
    )
    attacked_candidate = copy.copy(attacked_candidates[attacked_index])
    object.__setattr__(  # noqa: PLC2801
        attacked_candidate,
        "resource_stratum_id",
        "foreign_resource_stratum",
    )
    attacked_candidates[attacked_index] = attacked_candidate
    attacked_screening_manifest = copy.copy(prior.inventory.screening_manifest)
    object.__setattr__(  # noqa: PLC2801
        attacked_screening_manifest,
        "candidates",
        tuple(attacked_candidates),
    )
    attacked_screening_plan = replace(
        prior.inventory.screening_plan,
        screening_manifest_checksum=attacked_screening_manifest.content_checksum,
    )
    mismatched_strata_inventory = _rebuild_inventory(
        prior.inventory,
        screening_plan=attacked_screening_plan,
        screening_manifest=attacked_screening_manifest,
    )
    with pytest.raises(ValueError, match="same exact resource_stratum_id"):
        _rebuild_study(study, exposure_inventory=mismatched_strata_inventory)


def test_terminal_rows_require_authenticated_outcome_and_custody_and_fixed_counts(
    prior: PriorExposureFixture,
    confirmation: ConfirmationContextFixture,
) -> None:
    """Success and failure rows retain exact first-attempt and partial custody."""
    context = confirmation.context
    success_job, failure_job = context.plan.jobs[:2]
    success_outcome, success_reopened = terminal_confirmation_custody(success_job, context, "success")
    failure_outcome, failure_reopened = terminal_confirmation_custody(failure_job, context, "failure")
    outcomes = {
        success_job.content_checksum: success_outcome,
        failure_job.content_checksum: failure_outcome,
    }
    reopened_results = {
        success_job.content_checksum: success_reopened,
        failure_job.content_checksum: failure_reopened,
    }
    study = _operational_study(
        context,
        prior.inventory,
        outcomes_by_job=outcomes,
        reopened_results_by_job=reopened_results,
    )
    success, failure = study.rows[:2]
    assert success.terminal_state == "success"
    assert success.observed_test_trajectory_count == success.fixed_test_trajectory_count == 256
    assert success.raw_trajectory_document_checksum is not None
    assert success.partial_artifact_root is None
    assert failure.terminal_state == "failure"
    assert failure.observed_test_trajectory_count == 0
    assert failure.raw_trajectory_document_checksum is None
    assert failure.production_result_reference is not None
    assert failure.partial_artifact_root == failure.production_result_reference.manifest_content_checksum
    assert study.terminal_job_count == 2
    assert study.successful_job_count == study.failed_job_count == 1
    assert study.observed_test_trajectory_count == 256

    with pytest.raises(ValueError, match="derived from the embedded typed custody members"):
        _rebuild_row(
            success,
            production_custody_checksum=canonical_checksum({"caller": "authored custody summary"}),
        )

    with pytest.raises(ValueError, match="both outcome and authenticated reopened result"):
        _operational_study(
            context,
            prior.inventory,
            outcomes_by_job={success_job.content_checksum: success_outcome},
        )
    with pytest.raises(ValueError, match="both outcome and authenticated reopened result"):
        _operational_study(
            context,
            prior.inventory,
            reopened_results_by_job={success_job.content_checksum: success_reopened},
        )

    short_outcome, short_reopened = terminal_confirmation_custody(
        success_job,
        context,
        "success",
        trajectory_count=255,
    )
    with pytest.raises(ValueError, match="exact fixed test trajectory count"):
        _operational_study(
            context,
            prior.inventory,
            outcomes_by_job={success_job.content_checksum: short_outcome},
            reopened_results_by_job={success_job.content_checksum: short_reopened},
        )

    with pytest.raises(ValueError, match="contiguous plan-order prefix"):
        _operational_study(
            context,
            prior.inventory,
            outcomes_by_job={failure_job.content_checksum: failure_outcome},
            reopened_results_by_job={failure_job.content_checksum: failure_reopened},
        )


def test_resource_limit_status_is_derived_and_forbids_later_terminal_rows(
    prior: PriorExposureFixture,
    confirmation: ConfirmationContextFixture,
) -> None:
    """Authenticated resource-limit evidence marks custody and stops later jobs."""
    context = confirmation.context
    stopped_job, later_job = context.plan.jobs[:2]
    stopped_outcome, stopped_reopened = terminal_confirmation_custody(
        stopped_job,
        context,
        "failure",
        failure_evidence_exception_type="NormalizedComputeCapError",
    )
    stopped = _operational_study(
        context,
        prior.inventory,
        outcomes_by_job={stopped_job.content_checksum: stopped_outcome},
        reopened_results_by_job={stopped_job.content_checksum: stopped_reopened},
    )
    assert stopped.status == "incomplete_resource_limit"
    roundtripped_stop = LockedConfirmatoryStudyRow.from_dict(stopped.rows[0].to_dict())
    assert (
        LockedConfirmatoryStudyManifest._derive_status(  # noqa: SLF001 -- focused derivation test
            (roundtripped_stop, *stopped.rows[1:]),
            context.final_seal,
        )
        == "incomplete_resource_limit"
    )

    mismatched_outcome, mismatched_reopened = terminal_confirmation_custody(
        stopped_job,
        context,
        "failure",
        failure_evidence_exception_type="ConfirmationResourceLimitError",
    )
    with pytest.raises(ValueError, match="typed failure exception family"):
        _operational_study(
            context,
            prior.inventory,
            outcomes_by_job={stopped_job.content_checksum: mismatched_outcome},
            reopened_results_by_job={stopped_job.content_checksum: mismatched_reopened},
        )

    later_outcome, later_reopened = terminal_confirmation_custody(later_job, context, "failure")
    with pytest.raises(ValueError, match="may follow authenticated resource-limit"):
        _operational_study(
            context,
            prior.inventory,
            outcomes_by_job={
                stopped_job.content_checksum: stopped_outcome,
                later_job.content_checksum: later_outcome,
            },
            reopened_results_by_job={
                stopped_job.content_checksum: stopped_reopened,
                later_job.content_checksum: later_reopened,
            },
        )


def test_study_roundtrip_rejects_missing_duplicate_and_count_tamper(
    prior: PriorExposureFixture,
    confirmation: ConfirmationContextFixture,
) -> None:
    """Study decoding rederives plan rows, counts, pairing records, and roots."""
    study = LockedConfirmatoryStudyManifest.create(
        context=confirmation.context,
        exposure_inventory=prior.inventory,
    )
    assert LockedConfirmatoryStudyManifest.from_dict(study.to_dict()) == study

    with pytest.raises(ValueError, match="exactly one row"):
        _rebuild_study(study, rows=study.rows[:-1])

    rows = list(study.rows)
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="duplicate"):
        _rebuild_study(study, rows=tuple(rows))

    with pytest.raises(ValueError, match="planned_test_trajectory_count"):
        _rebuild_study(study, planned_test_trajectory_count=study.planned_test_trajectory_count - 1)
