# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Real-but-bounded dormant confirmation execution tests for WP22F."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation import training_runner
from benchmarks.state_preparation.phase2 import production_executors as production_executor_module
from benchmarks.state_preparation.phase2.confirmatory_study import PriorTargetExposureInventory
from benchmarks.state_preparation.phase2.execution_context import ConfirmationExecutionContext
from benchmarks.state_preparation.phase2.production_executors import (
    ProductionAttemptStore,
    ProductionConfirmationAuthority,
    ProductionConfirmationExecutor,
    SyntheticConfirmationFixture,
    create_default_training_executor_registry,
    initialize_confirmation_plan_session,
    reopen_result_artifact,
)
from benchmarks.state_preparation.phase2.protocol import load_initial_preregistration
from benchmarks.state_preparation.phase2.training_orchestration import (
    ConfirmExecutionRequest,
    JobExecutionControls,
    TrainingExecutorRegistry,
    confirmatory_evaluation_policy_checksum,
)
from benchmarks.state_preparation.training_runner import TrainingRunnerConfigurationError
from tests.benchmarks.wp22_confirmation_test_support import build_confirmation_context_fixture

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.state_preparation.phase2.production_executors import ResolvedProductionJob
    from benchmarks.state_preparation.phase2.targets import MaterializedTarget


def test_source_locked_confirmation_authority_compiles_then_dispatches_one_bounded_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real registry route closes source, target, schedule, and first-attempt custody."""
    fixture = build_confirmation_context_fixture(tmp_path)
    context = fixture.context
    output_root = tmp_path / "confirmation-output"
    context.preflight(fixture.repository_root, output_root)
    assert not output_root.exists()

    job = context.plan.jobs[0]
    request = cast("ConfirmExecutionRequest", job.confirm_execution_request)
    detached_request = ConfirmExecutionRequest.from_json(request.to_json())
    assert detached_request == request
    with pytest.raises(ValueError, match="exact context-owned request"):
        context.scheduled_program_checksum(detached_request)
    with pytest.raises(ValueError, match="exact context-owned request"):
        context.artifact_kind(replace(request, hyperparameters_checksum=context.final_seal.content_checksum))

    materialization_calls: list[ConfirmationExecutionContext] = []
    original_materialize = ConfirmationExecutionContext.materialize_targets

    def record_materialization(
        bound_context: ConfirmationExecutionContext,
    ) -> tuple[MaterializedTarget, ...]:
        """Record the first authorized test-population materialization.

        Returns:
            The materialized test-only target vectors.
        """
        materialization_calls.append(bound_context)
        return original_materialize(bound_context)

    monkeypatch.setattr(ConfirmationExecutionContext, "materialize_targets", record_materialization)
    authority = ProductionConfirmationAuthority(context)
    assert not materialization_calls
    resolved = authority.resolve(request)
    assert materialization_calls == [context]
    assert resolved.confirm_request is request
    assert resolved.evidence_identity_checksum == request.content_checksum
    assert resolved.scheduled_program.content_checksum == context.scheduled_program_checksum(request)
    assert resolved.evaluation_policy_checksum == confirmatory_evaluation_policy_checksum(request)
    assert context.artifact_kind(request) == "pipeline"

    dispatched: list[ResolvedProductionJob] = []

    def bounded_stop(
        dispatched_job: ResolvedProductionJob,
        _store: ProductionAttemptStore,
        artifact_kind: str,
    ) -> object:
        """Stop before numerical training while proving the real dispatch identity.

        Raises:
            RuntimeError: Always, after validating the resolved production job.
        """
        assert dispatched_job.confirm_request is request
        assert dispatched_job.scheduled_program.content_checksum == context.scheduled_program_checksum(request)
        assert artifact_kind == context.artifact_kind(request)
        dispatched.append(dispatched_job)
        msg = "bounded confirmation dispatch stop"
        raise RuntimeError(msg)

    monkeypatch.setattr(production_executor_module, "_dispatch_production_attempt", bounded_stop)
    monkeypatch.setattr(
        production_executor_module,
        "_validate_confirmation_aggregate_plan_position",
        lambda _context, _request: None,
    )
    directory = output_root / job.output_path
    initialize_confirmation_plan_session(context)
    executor = ProductionConfirmationExecutor(context)
    with pytest.raises(RuntimeError, match="bounded confirmation dispatch stop"):
        executor.execute(request, directory, JobExecutionControls(resume=False, overwrite=False))
    assert len(dispatched) == 1

    reference = ProductionAttemptStore(directory, request.content_checksum, 1).derive_existing_ref()
    reopened = reopen_result_artifact(reference, directory)
    assert reopened.evidence.status == "failure"
    assert reopened.evidence.job_checksum == request.content_checksum
    assert reopened.evidence.artifact_kind == context.artifact_kind(request)
    assert reopened.evidence.scheduled_program_checksum == context.scheduled_program_checksum(request)
    assert reopened.evidence.evaluation_policy_checksum == confirmatory_evaluation_policy_checksum(request)


def test_real_registry_and_runner_reject_synthetic_or_programmatic_injection(
    tmp_path: Path,
) -> None:
    """Real confirmation is available only through the repository-owned CLI route."""
    context = build_confirmation_context_fixture(tmp_path).context
    registry = create_default_training_executor_registry(context)
    assert registry.confirm_executor is not None
    request = cast("ConfirmExecutionRequest", context.plan.jobs[0].confirm_execution_request)
    fixture = SyntheticConfirmationFixture(
        request.content_checksum,
        (0.5,) * request.fixed_test_trajectory_count,
    )
    with pytest.raises(ValueError, match="cannot accept a synthetic fixture"):
        create_default_training_executor_registry(context, synthetic_confirmation_fixture=fixture)

    options = training_runner.resolve_options(
        training_runner.parse_arguments(["--preset", "paper-confirm", "--execute-expensive"])
    )
    with pytest.raises(TrainingRunnerConfigurationError, match="programmatic context or executor injection"):
        training_runner.run(options, context=context)
    with pytest.raises(TrainingRunnerConfigurationError, match="programmatic context or executor injection"):
        training_runner.run(
            options,
            executor=TrainingExecutorRegistry(
                confirm_executor=lambda _request, _path, _controls: request.content_checksum,
            ),
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--dry-run",), "execute-expensive"),
        ((), "execute-expensive"),
    ],
)
def test_confirmation_public_entry_points_do_not_load_held_inputs_without_execution_opt_in(
    arguments: tuple[str, ...],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run and exported builders reject before context, entropy, or target access."""
    options = training_runner.resolve_options(
        training_runner.parse_arguments(["--preset", "paper-confirm", *arguments])
    )
    context_loads: list[object] = []
    target_loads: list[object] = []

    def forbidden_context_load(*values: object) -> object:
        """Record an invalid attempt to enter confirmation artifact loading."""
        context_loads.append(values)
        pytest.fail("confirmation context loader was reached without execution opt-in")

    def forbidden_target_load(value: object) -> object:
        """Record an invalid attempt to open the held target path."""
        target_loads.append(value)
        pytest.fail("held target loader was reached without execution opt-in")

    monkeypatch.setattr(training_runner, "_load_confirmation_execution_context", forbidden_context_load)
    monkeypatch.setattr(training_runner, "_load_targets", forbidden_target_load)
    for entry_point in (
        training_runner.run,
        training_runner.build_training_plan,
        training_runner.build_confirmation_execution_context,
    ):
        with pytest.raises(TrainingRunnerConfigurationError, match=message):
            entry_point(options)
    assert not context_loads
    assert not target_loads


@pytest.mark.parametrize("arguments", [(), ("--dry-run",)])
def test_confirmation_context_loader_itself_guards_before_any_held_path(
    arguments: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The central private loader is fail-closed even when called directly."""
    options = training_runner.resolve_options(
        training_runner.parse_arguments(["--preset", "paper-confirm", *arguments])
    )
    path_accesses: list[object] = []

    def forbidden_path_access(*values: object) -> object:
        """Record an invalid attempt to inspect any confirmation path."""
        path_accesses.append(values)
        pytest.fail("confirmation path was inspected before execution opt-in")

    monkeypatch.setattr(training_runner, "_require_paths", forbidden_path_access)
    monkeypatch.setattr(training_runner, "_load_targets", forbidden_path_access)
    with pytest.raises(TrainingRunnerConfigurationError, match=r"dry-run|execute-expensive"):
        training_runner._load_confirmation_execution_context(  # noqa: SLF001 -- direct central-guard regression
            options,
            load_initial_preregistration(),
        )
    assert not path_accesses


def test_opted_in_confirmation_dry_run_preflights_plan_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit reveal consent permits dry-run validation but no numerical work."""
    output = tmp_path / "opted-in-confirmation-dry-run"
    fixture = build_confirmation_context_fixture(tmp_path, authorized_output_root=output)
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--dry-run",
            "--execute-expensive",
            "--repository-root",
            str(fixture.repository_root),
            "--expected-locked-study-head",
            str(tmp_path / "confirmation-study-head.json"),
            "--output",
            str(output),
        ])
    )
    context_loads: list[object] = []

    def load_authorized_context(*values: object) -> ConfirmationExecutionContext:
        """Return the already built context after recording intentional reveal."""
        context_loads.append(values)
        return fixture.context

    def forbidden_dispatch(*_values: object, **_keywords: object) -> object:
        """Fail if an opted-in dry run reaches real numerical dispatch."""
        pytest.fail("opted-in confirmation dry-run attempted numerical dispatch")

    monkeypatch.setattr(training_runner, "_load_confirmation_execution_context", load_authorized_context)
    inventory = object.__new__(PriorTargetExposureInventory)
    object.__setattr__(  # noqa: PLC2801 -- narrow checksum-only dry-run test seam
        inventory,
        "_content_checksum",
        fixture.context.prior_target_exposure_inventory_checksum,
    )
    monkeypatch.setattr(training_runner, "_load_prior_target_exposure_inventory", lambda _options: inventory)
    monkeypatch.setattr(ProductionConfirmationExecutor, "execute", forbidden_dispatch)
    plan = training_runner.run(options)
    assert plan == fixture.context.plan
    assert len(context_loads) == 1
    assert not output.exists()
