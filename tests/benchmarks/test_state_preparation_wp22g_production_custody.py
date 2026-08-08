# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Adversarial first-attempt and resource custody tests for WP22G."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation.phase2 import production_executors as production_module
from benchmarks.state_preparation.phase2.confirmatory_study_store import (
    publish_locked_confirmatory_study_snapshot,
)
from benchmarks.state_preparation.phase2.layerwise_bmpd import create_bmpd_circuit_binding
from benchmarks.state_preparation.phase2.production_executors import (
    ArtifactBlobRef,
    ConfirmationResourceLimitError,
    ConfirmationResourceLimitProof,
    PersistedConfirmationResourceLimitError,
    PersistedProductionAttemptError,
    ProductionAttemptStore,
    ProductionConfirmationAuthority,
    ProductionConfirmationExecutor,
    ProductionNumericalEvidence,
    authenticated_confirmation_resource_limit_proof,
    initialize_confirmation_plan_session,
    is_authenticated_confirmation_resource_limit_stop,
    reopen_result_artifact,
    validate_existing_confirmation_outcome,
)
from benchmarks.state_preparation.phase2.scheduled_execution import NormalizedComputeCapError
from benchmarks.state_preparation.phase2.training_orchestration import (
    JobExecutionControls,
    TrainingJobOutcome,
    confirmatory_evaluation_policy_checksum,
    training_job_attempt_path,
)
from tests.benchmarks.wp22_confirmation_test_support import build_confirmation_context_fixture
from tests.benchmarks.wp22g_confirmatory_study_test_support import prior_exposure_fixture

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.state_preparation.phase2.confirmatory_study import PriorTargetExposureInventory
    from benchmarks.state_preparation.phase2.execution_context import ConfirmationExecutionContext
    from benchmarks.state_preparation.phase2.production_executors import ResolvedProductionJob
    from benchmarks.state_preparation.phase2.training_orchestration import (
        ConfirmExecutionRequest,
        TrainingJob,
    )


def _first_job(context: ConfirmationExecutionContext) -> tuple[TrainingJob, ConfirmExecutionRequest, Path]:
    """Return the first job, request, and sole authorized output path."""
    job = context.plan.jobs[0]
    request = cast("ConfirmExecutionRequest", job.confirm_execution_request)
    return job, request, context.authorized_output_root / job.output_path


def _locked_confirmation_context(tmp_path: Path) -> ConfirmationExecutionContext:
    """Create session and all-unattempted snapshot custody required for dispatch.

    Returns:
        A context rebound to the exact snapshotted prior-exposure inventory.
    """
    base = build_confirmation_context_fixture(tmp_path).context
    inventory = _prior_exposure_inventory(base)
    context = replace(
        base,
        prior_target_exposure_inventory_checksum=inventory.content_checksum,
    )
    initialize_confirmation_plan_session(context)
    publish_locked_confirmatory_study_snapshot(context, inventory)
    return context


def _prior_exposure_inventory(context: ConfirmationExecutionContext) -> PriorTargetExposureInventory:
    """Rebuild the deterministic exact prior-exposure inventory for a context.

    Returns:
        The exact inventory embedded into locked-study snapshots.
    """
    return prior_exposure_fixture(
        resource_calibration_checksum=cast(
            "str",
            context.final_seal.primary_resource_budget["reachable_stratum_manifest_checksum"],
        ),
        execution_source_manifest_checksum=context.final_seal.execution_source_checksum,
    ).inventory


def _failure_outcome(job_checksum: str) -> TrainingJobOutcome:
    """Return the outer projection for an authoritative structured failure."""
    return TrainingJobOutcome(
        job_checksum=job_checksum,
        status="failure",
        result_artifact_checksum=None,
        exception_type="executor_failure",
        message="executor failed; secret-bearing diagnostics are intentionally not persisted",
        attempt=1,
    )


def test_confirmation_session_marker_is_single_link_atomic_publication(tmp_path: Path) -> None:
    """The canonical marker is a lone rename target with no in-root staging alias."""
    context = build_confirmation_context_fixture(tmp_path).context
    reference = initialize_confirmation_plan_session(context)
    assert reference.marker_path.stat().st_nlink == 1
    assert reference.locked_study_head_custody_path == context.locked_study_head_custody_path
    assert not tuple(context.authorized_output_root.glob(".wp22-confirmation-session-*.tmp"))
    assert not tuple(
        context.authorized_output_root.parent.glob(
            f".wp22-confirmation-session-{context.authorized_output_root.name}-*.tmp"
        )
    )
    assert initialize_confirmation_plan_session(context) == reference


def test_confirmation_terminal_kill_window_stages_only_outside_scientific_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure immediately before rename leaves no in-root temp or hard link."""
    output_root = (tmp_path / "confirmation-output").resolve()
    job_directory = output_root / "roles/confirmatory/test-job"
    store = ProductionAttemptStore(
        job_directory,
        "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        1,
        confirmation_output_root=output_root,
    )
    payload = b"complete terminal payload\n"
    observed_staging_paths: list[Path] = []

    def interrupt_before_rename(
        _source_directory_descriptor: int,
        source_name: str,
        _destination_directory_descriptor: int,
        _destination_name: str,
    ) -> None:
        """Model process death after the off-tree member was durably filled.

        Raises:
            RuntimeError: Always, at the exclusive-rename kill window.
        """
        staged = output_root.parent / source_name
        observed_staging_paths.append(staged)
        assert staged.is_file()
        assert staged.read_bytes() == payload
        assert not staged.is_relative_to(output_root)
        msg = "simulated kill immediately before terminal rename"
        raise RuntimeError(msg)

    monkeypatch.setattr(production_module, "_atomic_rename_no_replace", interrupt_before_rename)
    with pytest.raises(RuntimeError, match="simulated kill"):
        store._publish_terminal_payload(payload)  # noqa: SLF001 - exact publication kill window

    assert len(observed_staging_paths) == 1
    assert not observed_staging_paths[0].exists()
    assert not (job_directory / store.manifest_relative_path).exists()
    assert not tuple(path for path in output_root.rglob("*") if path.is_file())


def test_confirmation_requires_initial_locked_snapshot_before_target_reveal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session marker alone cannot authorize direct subset dispatch."""
    context = build_confirmation_context_fixture(tmp_path).context
    initialize_confirmation_plan_session(context)
    _job, request, directory = _first_job(context)
    reveals: list[object] = []

    def forbidden_reveal(*values: object) -> object:
        """Record an invalid target reveal."""
        reveals.append(values)
        pytest.fail("held targets were materialized without locked snapshot zero")

    monkeypatch.setattr(type(context), "materialize_targets", forbidden_reveal)
    with pytest.raises(ValueError, match=r"initial|snapshot|locked|study directory"):
        ProductionConfirmationExecutor(context).execute(
            request,
            directory,
            JobExecutionControls(resume=False, overwrite=False),
        )
    assert not reveals
    assert not (directory / "production_attempts").exists()


def test_confirmation_rejects_arbitrary_directory_and_skipped_cell_before_target_reveal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither a second directory nor a skipped plan cell can reveal held targets."""
    context = _locked_confirmation_context(tmp_path)
    executor = ProductionConfirmationExecutor(context)
    job, request, directory = _first_job(context)
    reveals: list[object] = []

    def forbidden_reveal(*values: object) -> object:
        """Record an invalid target reveal."""
        reveals.append(values)
        pytest.fail("held targets were materialized before output/plan custody")

    monkeypatch.setattr(type(context), "materialize_targets", forbidden_reveal)
    arbitrary = tmp_path / "second-confirmation-root"
    with pytest.raises(ValueError, match="exact authorized output path"):
        executor.execute(request, arbitrary, JobExecutionControls(resume=False, overwrite=False))
    with pytest.raises(ValueError, match=r"unavailable|missing|unsafe"):
        validate_existing_confirmation_outcome(context, job, _failure_outcome(job.content_checksum), directory)
    later = context.plan.jobs[1]
    later_request = cast("ConfirmExecutionRequest", later.confirm_execution_request)
    with pytest.raises(ValueError, match="skipped a prior canonical unattempted plan cell"):
        executor.execute(
            later_request,
            context.authorized_output_root / later.output_path,
            JobExecutionControls(resume=False, overwrite=False),
        )
    assert not reveals
    assert not arbitrary.exists()


def test_direct_second_dispatch_requires_prior_outer_outcome_and_aggregate_advance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A production terminal alone cannot authorize the next scientific cell."""
    context = _locked_confirmation_context(tmp_path)
    executor = ProductionConfirmationExecutor(context)
    _job, request, directory = _first_job(context)

    def fail_first_dispatch(*_values: object) -> object:
        """Create an authenticated generic first-cell failure terminal.

        Raises:
            RuntimeError: Always, after the executor has resolved the first cell.
        """
        msg = "bounded first-cell failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(production_module, "_dispatch_production_attempt", fail_first_dispatch)
    with pytest.raises(RuntimeError, match="bounded first-cell failure"):
        executor.execute(request, directory, JobExecutionControls(resume=False, overwrite=False))

    resolves: list[object] = []

    def forbidden_resolve(*values: object) -> object:
        """Fail if stale aggregate custody reaches target resolution."""
        resolves.append(values)
        pytest.fail("stale aggregate custody reached target resolution")

    monkeypatch.setattr(executor.authority, "resolve", forbidden_resolve)
    later = context.plan.jobs[1]
    later_request = cast("ConfirmExecutionRequest", later.confirm_execution_request)
    with pytest.raises(ValueError, match="aggregate head does not authorize"):
        executor.execute(
            later_request,
            context.authorized_output_root / later.output_path,
            JobExecutionControls(resume=False, overwrite=False),
        )
    assert not resolves
    assert not (context.authorized_output_root / later.output_path / "production_attempts").exists()


def test_cached_direct_gate_cannot_survive_external_head_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every call reopens external custody even after a prior snapshot check."""
    context = _locked_confirmation_context(tmp_path)
    executor = ProductionConfirmationExecutor(context)
    _job, request, directory = _first_job(context)
    later = context.plan.jobs[1]
    later_request = cast("ConfirmExecutionRequest", later.confirm_execution_request)

    # This rejected call populated the obsolete in-memory snapshot-zero cache.
    with pytest.raises(ValueError, match="skipped a prior canonical unattempted plan cell"):
        executor.execute(
            later_request,
            context.authorized_output_root / later.output_path,
            JobExecutionControls(resume=False, overwrite=False),
        )
    context.locked_study_head_custody_path.unlink()

    resolves: list[object] = []

    def forbidden_resolve(*values: object) -> object:
        """Fail if deleted external custody reaches target resolution."""
        resolves.append(values)
        pytest.fail("deleted external custody reached target resolution")

    monkeypatch.setattr(executor.authority, "resolve", forbidden_resolve)
    with pytest.raises(ValueError, match="External locked-study head custody is missing"):
        executor.execute(request, directory, JobExecutionControls(resume=False, overwrite=False))
    assert not resolves
    assert not (directory / "production_attempts").exists()


def test_direct_dispatch_rejects_one_snapshot_external_head_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A penultimate external reference cannot authorize another target use."""
    context = _locked_confirmation_context(tmp_path)
    initial_head_payload = context.locked_study_head_custody_path.read_bytes()
    executor = ProductionConfirmationExecutor(context)
    job, request, directory = _first_job(context)

    def fail_first_dispatch(*_values: object) -> object:
        """Create one bounded generic terminal failure.

        Raises:
            RuntimeError: Always, after exact first-cell resolution.
        """
        msg = "bounded aggregate-advance failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(production_module, "_dispatch_production_attempt", fail_first_dispatch)
    with pytest.raises(RuntimeError, match="bounded aggregate-advance failure"):
        executor.execute(request, directory, JobExecutionControls(resume=False, overwrite=False))
    outcome = _failure_outcome(job.content_checksum)
    outcome_path = training_job_attempt_path(directory, 1)
    outcome_path.parent.mkdir(parents=True)
    outcome_path.write_bytes(production_module._json_bytes(outcome.to_dict()))  # noqa: SLF001 - custody fixture
    publish_locked_confirmatory_study_snapshot(context, _prior_exposure_inventory(context))
    assert context.locked_study_head_custody_path.read_bytes() != initial_head_payload

    context.locked_study_head_custody_path.write_bytes(initial_head_payload)
    resumed_executor = ProductionConfirmationExecutor(context)
    resolves: list[object] = []

    def forbidden_resolve(*values: object) -> object:
        """Fail if rolled-back external custody reaches target resolution."""
        resolves.append(values)
        pytest.fail("rolled-back aggregate custody reached target resolution")

    monkeypatch.setattr(resumed_executor.authority, "resolve", forbidden_resolve)
    later = context.plan.jobs[1]
    later_request = cast("ConfirmExecutionRequest", later.confirm_execution_request)
    with pytest.raises(ValueError, match="rolled back behind the immutable aggregate chain"):
        resumed_executor.execute(
            later_request,
            context.authorized_output_root / later.output_path,
            JobExecutionControls(resume=True, overwrite=False),
        )
    assert not resolves
    assert not (context.authorized_output_root / later.output_path / "production_attempts").exists()


def test_interrupted_first_attempt_is_terminalized_once_and_resume_custody_is_mandatory(
    tmp_path: Path,
) -> None:
    """Closed partial members become one immutable failure, never attempt two."""
    context = _locked_confirmation_context(tmp_path)
    job, request, directory = _first_job(context)
    store = ProductionAttemptStore(directory, request.content_checksum, 1)
    partial = production_module._typed_document(  # noqa: SLF001 -- bounded crash-boundary fixture
        "closed_partial_stage",
        {"request_checksum": request.content_checksum},
    )
    partial_ref = store.write_json_blob(
        "structural_prefix/stage_000.json",
        partial,
        role="structural_stage",
    )
    partial_path = directory / partial_ref.path
    original_partial_bytes = partial_path.read_bytes()

    executor = ProductionConfirmationExecutor(context)
    with pytest.raises(PersistedProductionAttemptError, match="Recovered interrupted"):
        executor.execute(request, directory, JobExecutionControls(resume=True, overwrite=False))
    reference = ProductionAttemptStore(directory, request.content_checksum, 1).derive_existing_ref()
    reopened = reopen_result_artifact(reference, directory)
    assert reference.status == "failure"
    assert reopened.evidence.status == "failure"
    assert partial_path.read_bytes() == original_partial_bytes
    assert tuple(path.name for path in (directory / "production_attempts").iterdir()) == ("attempt_000001",)
    validate_existing_confirmation_outcome(context, job, _failure_outcome(job.content_checksum), directory)

    manifest_path = directory / reference.manifest_path
    terminal_bytes = manifest_path.read_bytes()
    assert manifest_path.stat().st_nlink == 1
    assert not tuple(context.authorized_output_root.rglob(".wp22g-terminal-*.tmp"))
    assert not tuple(context.authorized_output_root.parent.glob(".wp22g-terminal-*.tmp"))
    with pytest.raises(PersistedProductionAttemptError, match="structured failure"):
        executor.execute(request, directory, JobExecutionControls(resume=True, overwrite=False))
    assert manifest_path.read_bytes() == terminal_bytes
    for exception_type, message in (
        (
            "arbitrary_resealed_failure",
            "executor failed; secret-bearing diagnostics are intentionally not persisted",
        ),
        ("executor_failure", "resealed arbitrary diagnostic message"),
    ):
        with pytest.raises(ValueError, match="deterministic redacted outer failure projection"):
            validate_existing_confirmation_outcome(
                context,
                job,
                TrainingJobOutcome(
                    job_checksum=job.content_checksum,
                    status="failure",
                    result_artifact_checksum=None,
                    exception_type=exception_type,
                    message=message,
                    attempt=1,
                ),
                directory,
            )
    manifest_path.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match=r"manifest|checksum|sealed"):
        validate_existing_confirmation_outcome(
            context,
            job,
            _failure_outcome(job.content_checksum),
            directory,
        )


@pytest.mark.parametrize(
    ("relative", "partial_bytes"),
    [
        ("production_attempts/attempt_000001/evaluation/raw_trajectory_fidelities.json", b'{"schema_version":'),
        ("production_attempts/attempt_000001/production_evidence.json", b""),
    ],
)
def test_torn_known_path_member_is_preserved_as_opaque_first_attempt_custody(
    tmp_path: Path,
    relative: str,
    partial_bytes: bytes,
) -> None:
    """Truncated and zero-byte writes remain immutable, authenticated evidence."""
    context = _locked_confirmation_context(tmp_path)
    job, request, directory = _first_job(context)
    partial_path = directory / relative
    partial_path.parent.mkdir(parents=True)
    partial_path.write_bytes(partial_bytes)

    with pytest.raises(PersistedProductionAttemptError, match="Recovered interrupted"):
        ProductionConfirmationExecutor(context).execute(
            request,
            directory,
            JobExecutionControls(resume=True, overwrite=False),
        )
    reference = ProductionAttemptStore(directory, request.content_checksum, 1).derive_existing_ref()
    reopened = reopen_result_artifact(reference, directory)
    opaque = tuple(ref for ref in reopened.manifest.blobs if ref.role == "opaque_partial_member")
    assert len(opaque) == 1
    assert opaque[0].path == relative
    assert opaque[0].media_type == "application/octet-stream"
    assert opaque[0].byte_count == len(partial_bytes)
    assert partial_path.read_bytes() == partial_bytes
    validate_existing_confirmation_outcome(context, job, _failure_outcome(job.content_checksum), directory)
    partial_path.write_bytes(partial_bytes + b"x")
    with pytest.raises(ValueError, match=r"byte|checksum|size"):
        reopen_result_artifact(reference, directory)


def test_nonterminal_inventory_rejects_foreign_member_path_without_mutation(tmp_path: Path) -> None:
    """Opaque recovery is available only at repository-owned production paths."""
    store = ProductionAttemptStore(
        tmp_path / "foreign-member-job",
        "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        1,
    )
    foreign = store.job_directory / store.relative_attempt_directory / "foreign.bin"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"torn foreign bytes")
    with pytest.raises(ValueError, match="foreign member path"):
        store.inventory_closed_members()
    assert foreign.read_bytes() == b"torn foreign bytes"
    assert not (store.job_directory / store.manifest_relative_path).exists()


def test_recovery_after_closed_success_evidence_uses_a_distinct_failure_record(
    tmp_path: Path,
) -> None:
    """A crash after the normal evidence write never overwrites that member."""
    context = _locked_confirmation_context(tmp_path)
    job, request, directory = _first_job(context)
    resolved = ProductionConfirmationAuthority(context).resolve(request)
    store = ProductionAttemptStore(directory, request.content_checksum, 1)
    placeholder_checksum = request.execution_source_checksum

    def placeholder(role: str, path: str) -> ArtifactBlobRef:
        return ArtifactBlobRef(
            role=role,
            media_type="application/json",
            path=f"production_attempts/attempt_000001/{path}",
            byte_count=1,
            file_checksum=placeholder_checksum,
            logical_checksum=placeholder_checksum,
        )

    evidence = ProductionNumericalEvidence(
        job_checksum=request.content_checksum,
        attempt=1,
        artifact_kind="pipeline",
        status="success",
        execution_source_manifest_checksum=request.execution_source_checksum,
        source_fingerprint_checksum=request.execution_source_checksum,
        executable_binding_checksum=request.executable_binding_checksum,
        scheduled_program_checksum=resolved.scheduled_program.content_checksum,
        target_identity=resolved.target.identity_dict(),
        evaluation_policy_checksum=confirmatory_evaluation_policy_checksum(request),
        structural_prefix_checksums=(),
        schedule_snapshot_ref=placeholder("schedule_snapshot", "schedule/snapshot.json"),
        map_evidence_refs=(),
        diagnostic_refs=(),
        raw_trajectory_ref=placeholder("raw_trajectory_sidecar", "evaluation/raw_trajectory_fidelities.json"),
        resource_ref=placeholder("runtime_resources", "runtime/resources.json"),
        derived_metrics={
            "execution_preset": "paper-confirm",
            "scheduled_noisy_training": False,
            "pilot_diagnostic_required": False,
            "strategy_schedule_checksum": request.hyperparameters_checksum,
        },
        failure=None,
    )
    evidence_ref = store.write_json_blob(
        "production_evidence.json",
        evidence.to_dict(),
        role="production_evidence",
    )
    evidence_path = directory / evidence_ref.path
    original_evidence_bytes = evidence_path.read_bytes()

    with pytest.raises(PersistedProductionAttemptError, match="Recovered interrupted"):
        ProductionConfirmationExecutor(context).execute(
            request,
            directory,
            JobExecutionControls(resume=True, overwrite=False),
        )
    reference = ProductionAttemptStore(directory, request.content_checksum, 1).derive_existing_ref()
    reopened = reopen_result_artifact(reference, directory)
    assert reopened.evidence.status == "failure"
    assert evidence_path.read_bytes() == original_evidence_bytes
    assert (directory / "production_attempts/attempt_000001/recovery_failure_evidence.json").is_file()
    assert sum(ref.role == "production_evidence" for ref in reopened.manifest.blobs) == 2
    validate_existing_confirmation_outcome(context, job, _failure_outcome(job.content_checksum), directory)


@pytest.mark.parametrize("breach", ["normalized_work", "native_edge"])
def test_internally_measured_over_cap_resources_publish_structured_first_attempt_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    breach: str,
) -> None:
    """Both sealed cap dimensions stop success before terminal publication."""
    context = _locked_confirmation_context(tmp_path)
    job, request, directory = _first_job(context)
    dispatches: list[str] = []

    def over_cap_dispatch(
        resolved: ResolvedProductionJob,
        store: ProductionAttemptStore,
        _artifact_kind: str,
    ) -> object:
        """Persist a mechanically measured but over-cap resource record.

        Returns:
            No value; the success-publication resource gate always raises.
        """
        dispatches.append(resolved.evidence_identity_checksum)
        binding = create_bmpd_circuit_binding(6, 5 if breach == "native_edge" else 4)
        normalized_work = (
            cast("float", request.primary_resource_budget["normalized_compute_cap"]) + 1.0
            if breach == "normalized_work"
            else 1.0
        )
        resource_ref = store.write_json_blob(
            "runtime/resources.json",
            production_module._runtime_resource_document(  # noqa: SLF001 -- bounded cap fixture
                resolved=resolved,
                circuit_binding=binding,
                wall_time_seconds=0.0,
                peak_memory_bytes=0,
                normalized_work=normalized_work,
            ),
            role="runtime_resources",
        )
        return production_module._publish_attempt(  # noqa: SLF001 -- assert the publication gate
            store=store,
            resolved=resolved,
            artifact_kind="pipeline",
            status="success",
            blobs=(resource_ref,),
            prefix_checksums=(),
            schedule_snapshot_ref=None,
            map_evidence_refs=(),
            raw_trajectory_ref=None,
            resource_ref=resource_ref,
            derived_metrics={},
        )

    monkeypatch.setattr(production_module, "_dispatch_production_attempt", over_cap_dispatch)
    with pytest.raises(ConfirmationResourceLimitError, match="sealed"):
        ProductionConfirmationExecutor(context).execute(
            request,
            directory,
            JobExecutionControls(resume=False, overwrite=False),
        )
    reference = ProductionAttemptStore(directory, request.content_checksum, 1).derive_existing_ref()
    reopened = reopen_result_artifact(reference, directory)
    assert reference.status == "failure"
    assert reopened.evidence.failure is not None
    assert reopened.evidence.failure["exception_type"] == "ConfirmationResourceLimitError"
    proof = authenticated_confirmation_resource_limit_proof(request, reopened)
    assert proof is not None
    assert proof.proof_kind == "measured_confirmation_resources"
    assert proof.request_checksum == request.content_checksum
    assert proof.normalized_compute_cap == request.primary_resource_budget["normalized_compute_cap"]
    assert proof.native_edge_gate_cap == request.primary_resource_budget["cap_per_chain_edge"]
    assert proof.measured_normalized_work is not None
    assert len(proof.measured_native_edge_gate_counts) == request.qubit_count - 1
    assert proof.exceeded_dimensions == (breach,)
    assert is_authenticated_confirmation_resource_limit_stop(request, reopened)
    assert not (directory / "production_attempts/attempt_000002").exists()
    validate_existing_confirmation_outcome(context, job, _failure_outcome(job.content_checksum), directory)
    later = context.plan.jobs[1]
    later_request = cast("ConfirmExecutionRequest", later.confirm_execution_request)
    with pytest.raises(PersistedConfirmationResourceLimitError, match="terminally stopped"):
        ProductionConfirmationExecutor(context).execute(
            later_request,
            context.authorized_output_root / later.output_path,
            JobExecutionControls(resume=False, overwrite=False),
        )
    assert dispatches == [request.content_checksum]


def test_normalized_compute_cap_failure_preserves_typed_prospective_work_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scheduler cap stop preserves exact operands instead of trusting its class name."""
    context = _locked_confirmation_context(tmp_path)
    job, request, directory = _first_job(context)
    cap = cast("float", request.primary_resource_budget["normalized_compute_cap"])
    completed = cap - 1.0
    prospective = 2.0

    def prospective_cap_dispatch(*_values: object) -> object:
        """Raise the exact structured scheduler cap error.

        Raises:
            NormalizedComputeCapError: Always, with the sealed cap operands.
        """
        raise NormalizedComputeCapError(
            cap=cap,
            completed_work=completed,
            prospective_update_work=prospective,
        )

    monkeypatch.setattr(production_module, "_dispatch_production_attempt", prospective_cap_dispatch)
    with pytest.raises(NormalizedComputeCapError):
        ProductionConfirmationExecutor(context).execute(
            request,
            directory,
            JobExecutionControls(resume=False, overwrite=False),
        )
    reference = ProductionAttemptStore(directory, request.content_checksum, 1).derive_existing_ref()
    reopened = reopen_result_artifact(reference, directory)
    proof = authenticated_confirmation_resource_limit_proof(request, reopened)
    assert proof is not None
    assert proof.proof_kind == "prospective_normalized_work"
    assert proof.completed_normalized_work == completed
    assert proof.prospective_normalized_work == prospective
    assert proof.measured_normalized_work is None
    assert proof.measured_native_edge_gate_counts == ()
    assert proof.measurement_resource_ref is None
    assert proof.exceeded_dimensions == ("normalized_work",)
    assert reopened.evidence.failure is not None
    assert reopened.evidence.failure["exception_type"] == "NormalizedComputeCapError"
    validate_existing_confirmation_outcome(context, job, _failure_outcome(job.content_checksum), directory)

    tampered = proof.to_dict()
    tampered["completed_normalized_work"] = completed - 1.0
    with pytest.raises(ValueError, match="checksum"):
        ConfirmationResourceLimitProof.from_dict(tampered)


def test_resource_limit_class_name_without_typed_proof_is_not_a_study_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spoofed exception class name remains an ordinary structured failure."""
    context = _locked_confirmation_context(tmp_path)
    job, request, directory = _first_job(context)
    fake_error_type = cast(
        "type[RuntimeError]",
        type("ConfirmationResourceLimitError", (RuntimeError,), {}),
    )

    def spoofed_dispatch(*_values: object) -> object:
        """Raise an unrelated error whose class name copies the resource stop."""
        msg = "class-name-only spoof"
        raise fake_error_type(msg)

    monkeypatch.setattr(production_module, "_dispatch_production_attempt", spoofed_dispatch)
    with pytest.raises(fake_error_type):
        ProductionConfirmationExecutor(context).execute(
            request,
            directory,
            JobExecutionControls(resume=False, overwrite=False),
        )
    reference = ProductionAttemptStore(directory, request.content_checksum, 1).derive_existing_ref()
    reopened = reopen_result_artifact(reference, directory)
    assert reopened.evidence.failure is not None
    assert reopened.evidence.failure["exception_type"] == "ConfirmationResourceLimitError"
    assert authenticated_confirmation_resource_limit_proof(request, reopened) is None
    assert not is_authenticated_confirmation_resource_limit_stop(request, reopened)
    validate_existing_confirmation_outcome(context, job, _failure_outcome(job.content_checksum), directory)


def test_recovery_rebuilds_measured_proof_from_closed_over_cap_resources(tmp_path: Path) -> None:
    """A crash after durable measurement retains exact overage in terminal failure custody."""
    context = _locked_confirmation_context(tmp_path)
    job, request, directory = _first_job(context)
    resolved = ProductionConfirmationAuthority(context).resolve(request)
    store = ProductionAttemptStore(directory, request.content_checksum, 1)
    normalized_cap = cast("float", request.primary_resource_budget["normalized_compute_cap"])
    resource_ref = store.write_json_blob(
        "runtime/resources.json",
        production_module._runtime_resource_document(  # noqa: SLF001 -- exact crash-boundary fixture
            resolved=resolved,
            circuit_binding=create_bmpd_circuit_binding(request.qubit_count, 4),
            wall_time_seconds=1.0,
            peak_memory_bytes=1024,
            normalized_work=normalized_cap + 1.0,
        ),
        role="runtime_resources",
    )
    resource_path = directory / resource_ref.path
    original_resource_bytes = resource_path.read_bytes()

    with pytest.raises(PersistedProductionAttemptError, match="Recovered interrupted"):
        ProductionConfirmationExecutor(context).execute(
            request,
            directory,
            JobExecutionControls(resume=True, overwrite=False),
        )
    reference = ProductionAttemptStore(directory, request.content_checksum, 1).derive_existing_ref()
    reopened = reopen_result_artifact(reference, directory)
    proof = authenticated_confirmation_resource_limit_proof(request, reopened)
    assert proof is not None
    assert proof.proof_kind == "measured_confirmation_resources"
    assert proof.measurement_resource_ref == resource_ref
    assert proof.measured_normalized_work == pytest.approx(normalized_cap + 1.0)
    assert proof.exceeded_dimensions == ("normalized_work",)
    assert resource_path.read_bytes() == original_resource_bytes
    validate_existing_confirmation_outcome(context, job, _failure_outcome(job.content_checksum), directory)
