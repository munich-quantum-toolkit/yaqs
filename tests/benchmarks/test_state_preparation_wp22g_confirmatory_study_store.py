# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Filesystem and orchestration custody tests for the locked WP22G study."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation import training_runner
from benchmarks.state_preparation.phase2 import confirmatory_study_store as study_store_module
from benchmarks.state_preparation.phase2 import production_executors as production_module
from benchmarks.state_preparation.phase2 import training_orchestration as orchestration_module
from benchmarks.state_preparation.phase2.canonical import canonical_checksum, canonical_json
from benchmarks.state_preparation.phase2.confirmatory_study_store import (
    CONFIRMATORY_STUDY_DIRECTORY_NAME,
    LockedConfirmatoryStudySnapshot,
    confirmation_output_has_interrupted_attempt,
    publish_locked_confirmatory_study_snapshot,
    validate_initial_locked_confirmatory_study_snapshot,
    validate_locked_confirmatory_study_output,
)
from benchmarks.state_preparation.phase2.production_executors import (
    PersistedProductionAttemptError,
    ProductionAttemptStore,
    ProductionConfirmationExecutor,
    create_default_training_executor_registry,
    initialize_confirmation_plan_session,
)
from benchmarks.state_preparation.phase2.training_orchestration import (
    JOB_ATTEMPTS_DIRECTORY_NAME,
    JobExecutionControls,
    TrainingJobOutcome,
    execute_training_plan,
)
from tests.benchmarks.wp22_confirmation_test_support import build_confirmation_context_fixture
from tests.benchmarks.wp22g_confirmatory_study_test_support import prior_exposure_fixture

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.state_preparation.phase2.confirmatory_study import PriorTargetExposureInventory
    from benchmarks.state_preparation.phase2.execution_context import ConfirmationExecutionContext
    from tests.benchmarks.wp22_confirmation_test_support import ConfirmationContextFixture


def _study_fixture(
    tmp_path: Path,
    template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory]:
    """Build one exact context and prior-exposure inventory.

    Returns:
        The source fixture, rebound confirmation context, and exact inventory.
    """
    fixture, base, inventory = template
    context = replace(
        base,
        authorized_output_root=(tmp_path / "confirmation-output").resolve(),
        locked_study_head_custody_path=(tmp_path / "confirmation-study-head.json").resolve(),
    )
    return fixture, context, inventory


@pytest.fixture(scope="module")
def study_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory]:
    """Build the expensive sealed study universe once for filesystem tests.

    Returns:
        The reusable source fixture, base context, and exact exposure inventory.
    """
    fixture = build_confirmation_context_fixture(tmp_path_factory.mktemp("wp22g-study-store"))
    base = fixture.context
    prior = prior_exposure_fixture(
        resource_calibration_checksum=cast(
            "str",
            base.final_seal.primary_resource_budget["reachable_stratum_manifest_checksum"],
        ),
        execution_source_manifest_checksum=base.final_seal.execution_source_checksum,
    )
    context = replace(
        base,
        prior_target_exposure_inventory_checksum=prior.inventory.content_checksum,
    )
    return fixture, context, prior.inventory


def _failure_outcome(job_checksum: str) -> TrainingJobOutcome:
    """Return the sole deterministic outer failure projection."""
    return TrainingJobOutcome(
        job_checksum=job_checksum,
        status="failure",
        result_artifact_checksum=None,
        exception_type="executor_failure",
        message="executor failed; secret-bearing diagnostics are intentionally not persisted",
        attempt=1,
    )


def test_initial_snapshot_is_single_link_idempotent_and_session_bound(
    tmp_path: Path,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """Snapshot zero closes the all-unattempted universe before dispatch."""
    _fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)

    reference = publish_locked_confirmatory_study_snapshot(context, inventory)
    path = context.authorized_output_root / reference.relative_path
    snapshot = LockedConfirmatoryStudySnapshot.from_json(path.read_text(encoding="utf-8"))

    assert path.stat().st_nlink == 1
    assert snapshot.ordinal == 0
    assert snapshot.study_manifest.terminal_job_count == 0
    assert snapshot.study_manifest.unattempted_job_count == len(context.plan.jobs)
    external_head = context.locked_study_head_custody_path
    assert not external_head.is_relative_to(context.authorized_output_root)
    assert external_head.read_text(encoding="utf-8") == f"{reference.to_json()}\n"
    assert external_head.stat().st_nlink == 1
    assert validate_initial_locked_confirmatory_study_snapshot(context) == reference
    assert validate_locked_confirmatory_study_output(context, inventory) == reference
    foreign_head = replace(reference, file_checksum="sha256:" + "f" * 64)
    with pytest.raises(ValueError, match=r"externally retained|trusted head"):
        validate_locked_confirmatory_study_output(context, inventory, foreign_head)
    assert publish_locked_confirmatory_study_snapshot(context, inventory) == reference
    assert not tuple(context.authorized_output_root.parent.glob(".wp22-confirmatory-study-*.tmp"))
    assert not tuple(external_head.parent.glob(f".{external_head.name}.*.tmp"))


def test_external_head_recovers_snapshot_publication_crash_and_accepts_verified_ancestor(
    tmp_path: Path,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """External custody advances after an internal append without rejecting its prior head."""
    _fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)
    first = publish_locked_confirmatory_study_snapshot(context, inventory)
    first_path = context.authorized_output_root / first.relative_path
    first_snapshot = LockedConfirmatoryStudySnapshot.from_json(first_path.read_text(encoding="utf-8"))

    context.locked_study_head_custody_path.unlink()
    with pytest.raises(ValueError, match=r"externally published|head custody"):
        validate_initial_locked_confirmatory_study_snapshot(context)
    assert publish_locked_confirmatory_study_snapshot(context, inventory) == first

    second_snapshot = replace(first_snapshot, ordinal=1, previous_snapshot=first)
    second = study_store_module._write_snapshot(  # noqa: SLF001 -- exact publication-gap fixture
        context.authorized_output_root,
        second_snapshot,
    )
    assert context.locked_study_head_custody_path.read_text(encoding="utf-8") == f"{first.to_json()}\n"

    assert validate_locked_confirmatory_study_output(context, inventory, first) == second
    assert validate_initial_locked_confirmatory_study_snapshot(context) == first
    assert publish_locked_confirmatory_study_snapshot(context, inventory) == second
    assert context.locked_study_head_custody_path.read_text(encoding="utf-8") == f"{second.to_json()}\n"


def test_orchestration_repairs_penultimate_head_before_orphan_resume_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """A validated one-publication gap is repaired before orphan recovery dispatch."""
    fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)
    first = publish_locked_confirmatory_study_snapshot(context, inventory)
    initial_path = context.authorized_output_root / first.relative_path
    initial = LockedConfirmatoryStudySnapshot.from_json(initial_path.read_text(encoding="utf-8"))
    second_snapshot = replace(initial, ordinal=1, previous_snapshot=first)
    second = study_store_module._write_snapshot(  # noqa: SLF001 -- exact publication-gap fixture
        context.authorized_output_root,
        second_snapshot,
    )
    first_job = context.plan.jobs[0]
    request = first_job.confirm_execution_request
    assert request is not None
    store = ProductionAttemptStore(
        context.authorized_output_root / first_job.output_path,
        request.content_checksum,
        1,
    )
    (store.job_directory / store.relative_attempt_directory).mkdir(parents=True)
    assert context.locked_study_head_custody_path.read_text(encoding="utf-8") == f"{first.to_json()}\n"
    dispatches: list[object] = []

    def stop_at_dispatch(*values: object) -> tuple[TrainingJobOutcome, bool]:
        dispatches.append(values)
        assert context.locked_study_head_custody_path.read_text(encoding="utf-8") == f"{second.to_json()}\n"
        msg = "bounded dispatch observation"
        raise RuntimeError(msg)

    monkeypatch.setattr(orchestration_module, "_dispatch_job_outcome", stop_at_dispatch)
    registry = create_default_training_executor_registry(context)
    with pytest.raises(RuntimeError, match="bounded dispatch observation"):
        execute_training_plan(
            context.plan,
            context.authorized_output_root,
            registry,
            resume=True,
            context=context,
            repository_root=fixture.repository_root,
            prior_target_exposure_inventory=inventory,
            expected_locked_study_head=first,
        )
    assert len(dispatches) == 1

    third_snapshot = replace(second_snapshot, ordinal=2, previous_snapshot=second)
    study_store_module._write_snapshot(  # noqa: SLF001 -- invalid two-gap fixture
        context.authorized_output_root,
        third_snapshot,
    )
    context.locked_study_head_custody_path.write_text(f"{first.to_json()}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"lags beyond|publication gap"):
        execute_training_plan(
            context.plan,
            context.authorized_output_root,
            registry,
            resume=True,
            context=context,
            repository_root=fixture.repository_root,
            prior_target_exposure_inventory=inventory,
            expected_locked_study_head=first,
        )
    assert len(dispatches) == 1


def test_prior_cli_summary_head_wrapper_is_accepted_and_normalized(
    tmp_path: Path,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """Resume accepts strict prior CLI output and republishes one raw reference."""
    _fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)
    reference = publish_locked_confirmatory_study_snapshot(context, inventory)
    wrapper = {
        "planned": len(context.plan.jobs),
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "locked_study_snapshot_reference": reference.to_dict(),
        "external_study_head_custody_required": True,
    }
    context.locked_study_head_custody_path.write_text(f"{canonical_json(wrapper)}\n", encoding="utf-8")

    assert validate_initial_locked_confirmatory_study_snapshot(context) == reference
    assert publish_locked_confirmatory_study_snapshot(context, inventory) == reference
    assert context.locked_study_head_custody_path.read_text(encoding="utf-8") == f"{reference.to_json()}\n"


def test_first_unattempted_job_ancestor_prefix_is_recoverable(
    tmp_path: Path,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """An empty canonical job-ancestor prefix remains explicit resume state."""
    _fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)
    reference = publish_locked_confirmatory_study_snapshot(context, inventory)
    job_directory = context.authorized_output_root / context.plan.jobs[0].output_path

    job_directory.parent.mkdir(parents=True)

    assert validate_locked_confirmatory_study_output(context, inventory, reference) == reference
    assert confirmation_output_has_interrupted_attempt(context, inventory)


def test_first_unattempted_job_empty_nested_evaluation_directory_is_recoverable(
    tmp_path: Path,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """A crash after creating a known nested attempt directory can resume."""
    _fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)
    reference = publish_locked_confirmatory_study_snapshot(context, inventory)
    job = context.plan.jobs[0]
    request = job.confirm_execution_request
    assert request is not None
    store = ProductionAttemptStore(
        context.authorized_output_root / job.output_path,
        request.content_checksum,
        1,
    )

    (store.job_directory / store.relative_attempt_directory / "evaluation").mkdir(parents=True)

    assert validate_locked_confirmatory_study_output(context, inventory, reference) == reference
    assert confirmation_output_has_interrupted_attempt(context, inventory)


def test_first_unattempted_job_empty_outer_history_directory_is_recoverable(
    tmp_path: Path,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """An empty immutable outer-history directory remains explicit resume state."""
    _fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)
    reference = publish_locked_confirmatory_study_snapshot(context, inventory)
    job_directory = context.authorized_output_root / context.plan.jobs[0].output_path

    (job_directory / JOB_ATTEMPTS_DIRECTORY_NAME).mkdir(parents=True)

    assert validate_locked_confirmatory_study_output(context, inventory, reference) == reference
    assert confirmation_output_has_interrupted_attempt(context, inventory)


def test_marker_and_empty_study_directory_can_finish_snapshot_zero_initialization(
    tmp_path: Path,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """A crash immediately after study-directory creation can publish snapshot zero."""
    _fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)
    (context.authorized_output_root / CONFIRMATORY_STUDY_DIRECTORY_NAME).mkdir(mode=0o700)

    assert validate_locked_confirmatory_study_output(context, inventory) is None

    reference = publish_locked_confirmatory_study_snapshot(context, inventory)
    assert validate_locked_confirmatory_study_output(context, inventory, reference) == reference


def test_pre_reveal_snapshot_zero_gap_binds_public_roots_before_held_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """A valid-looking foreign snapshot zero fails before target or entropy access."""
    _fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)
    reference = publish_locked_confirmatory_study_snapshot(context, inventory)
    context.locked_study_head_custody_path.unlink()
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--resume",
            "--expected-locked-study-head",
            str(context.locked_study_head_custody_path),
            "--output",
            str(context.authorized_output_root),
        ])
    )
    accesses: list[str] = []

    def forbidden_target(*_arguments: object) -> object:
        accesses.append("target")
        pytest.fail("snapshot-zero preflight opened the held target")

    def forbidden_entropy(*_arguments: object) -> object:
        accesses.append("entropy")
        pytest.fail("snapshot-zero preflight opened held entropy")

    monkeypatch.setattr(training_runner, "_load_targets", forbidden_target)
    monkeypatch.setattr(training_runner.ExternalEntropyKeyring, "from_files", forbidden_entropy)
    training_runner._preflight_expected_locked_study_head_before_reveal(  # noqa: SLF001 -- security boundary
        options,
        None,
        final_seal=context.final_seal,
        configuration_execution_manifest=context.configuration_execution_manifest,
        execution_manifest=context.execution_source_manifest,
        analysis_manifest=context.analysis_source_manifest,
        exposure_inventory=inventory,
    )

    snapshot_path = context.authorized_output_root / reference.relative_path
    initial = LockedConfirmatoryStudySnapshot.from_json(snapshot_path.read_text(encoding="utf-8"))
    marker_entry = initial.output_entries[0]
    assert marker_entry.relative_path == ".wp22-confirmation-session.json"
    forged_entries = (
        replace(marker_entry, file_checksum="sha256:" + "f" * 64),
        *initial.output_entries[1:],
    )
    wrong_marker_receipt = replace(
        initial,
        output_entries=forged_entries,
        filesystem_inventory_root=canonical_checksum({
            "entry_checksums": [entry.content_checksum for entry in forged_entries],
        }),
    )
    wrong_output_root = replace(
        initial,
        authorized_output_root=str((tmp_path / "foreign-confirmation-output").resolve()),
    )
    for forged in (wrong_marker_receipt, wrong_output_root):
        snapshot_path.unlink()
        snapshot_path = (
            snapshot_path.parent / f"snapshot_{0:08d}_{forged.content_checksum.removeprefix('sha256:')}.json"
        )
        snapshot_path.write_text(f"{forged.to_json()}\n", encoding="utf-8")
        with pytest.raises(training_runner.TrainingRunnerConfigurationError, match=r"public-root-bound|snapshot-zero"):
            training_runner._preflight_expected_locked_study_head_before_reveal(  # noqa: SLF001 -- security boundary
                options,
                None,
                final_seal=context.final_seal,
                configuration_execution_manifest=context.configuration_execution_manifest,
                execution_manifest=context.execution_source_manifest,
                analysis_manifest=context.analysis_source_manifest,
                exposure_inventory=inventory,
            )
    assert accesses == []


def test_missing_study_directory_cannot_hide_confirmation_owned_state(
    tmp_path: Path,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """A marker plus scientific subtree is not mistaken for fresh output."""
    _fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)
    (context.authorized_output_root / "roles").mkdir()

    with pytest.raises(ValueError, match=r"owned state|snapshot custody"):
        validate_locked_confirmatory_study_output(context, inventory)


def test_exact_output_inventory_rejects_foreign_and_hardlinked_members(
    tmp_path: Path,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """Rogue bytes and external aliases cannot enter immutable study custody."""
    _fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)
    publish_locked_confirmatory_study_snapshot(context, inventory)
    foreign = context.authorized_output_root / "rogue.json"
    foreign.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"extra_files|universe"):
        validate_locked_confirmatory_study_output(context, inventory)
    foreign.unlink()
    fifo = context.authorized_output_root / "rogue.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="special member"):
        validate_locked_confirmatory_study_output(context, inventory)
    fifo.unlink()
    marker = context.authorized_output_root / ".wp22-confirmation-session.json"
    alias = tmp_path / "external-marker-alias.json"
    alias.hardlink_to(marker)
    with pytest.raises(ValueError, match=r"linked|session marker|unsafe"):
        validate_locked_confirmatory_study_output(context, inventory)


def test_terminal_results_cannot_backfill_missing_snapshot_zero(
    tmp_path: Path,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """Deleting snapshot custody cannot create a post-result initial branch."""
    _fixture, context, inventory = _study_fixture(tmp_path, study_template)
    initialize_confirmation_plan_session(context)
    reference = publish_locked_confirmatory_study_snapshot(context, inventory)
    job = context.plan.jobs[0]
    request = job.confirm_execution_request
    assert request is not None
    directory = context.authorized_output_root / job.output_path
    store = ProductionAttemptStore(directory, request.content_checksum, 1)
    store.write_json_blob(
        "structural_prefix/stage_000.json",
        production_module._typed_document(  # noqa: SLF001 -- closed partial-member fixture
            "closed_partial_stage",
            {"request_checksum": request.content_checksum},
        ),
        role="structural_stage",
    )
    with pytest.raises(PersistedProductionAttemptError, match="Recovered interrupted"):
        ProductionConfirmationExecutor(context).execute(
            request,
            directory,
            JobExecutionControls(resume=True, overwrite=False),
        )
    outcome = _failure_outcome(job.content_checksum)
    orchestration_module._write_outcome_attempt(  # noqa: SLF001 -- exact outer crash-boundary fixture
        directory,
        outcome,
        staging_directory=context.authorized_output_root.parent,
    )
    (context.authorized_output_root / reference.relative_path).unlink()

    with pytest.raises(ValueError, match=r"(?i)initial.*precede|terminal result|external.*before.*chain"):
        publish_locked_confirmatory_study_snapshot(context, inventory)


def test_orchestration_rejects_fail_fast_and_implicit_interrupted_recovery(
    tmp_path: Path,
    study_template: tuple[ConfirmationContextFixture, ConfirmationExecutionContext, PriorTargetExposureInventory],
) -> None:
    """Direct callers cannot adaptively stop or recover without explicit resume."""
    fixture, context, inventory = _study_fixture(tmp_path, study_template)
    registry = create_default_training_executor_registry(context)
    with pytest.raises(ValueError, match="fail_fast"):
        execute_training_plan(
            context.plan,
            context.authorized_output_root,
            registry,
            fail_fast=True,
            context=context,
            repository_root=fixture.repository_root,
            prior_target_exposure_inventory=inventory,
        )

    initialize_confirmation_plan_session(context)
    publish_locked_confirmatory_study_snapshot(context, inventory)
    first = context.plan.jobs[0]
    request = first.confirm_execution_request
    assert request is not None
    store = ProductionAttemptStore(
        context.authorized_output_root / first.output_path,
        request.content_checksum,
        1,
    )
    (store.job_directory / store.relative_attempt_directory).mkdir(parents=True)
    with pytest.raises(ValueError, match=r"(?i)resume=True|explicit resume|external.*custody"):
        execute_training_plan(
            context.plan,
            context.authorized_output_root,
            registry,
            resume=False,
            context=context,
            repository_root=fixture.repository_root,
            prior_target_exposure_inventory=inventory,
        )
    with pytest.raises(ValueError, match=r"externally retained|head reference"):
        execute_training_plan(
            context.plan,
            context.authorized_output_root,
            registry,
            resume=True,
            context=context,
            repository_root=fixture.repository_root,
            prior_target_exposure_inventory=inventory,
        )
