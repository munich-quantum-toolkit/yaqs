# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for WP22 sealed plan fan-out and durable dispatch."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation import training_runner as training_cli
from benchmarks.state_preparation.phase2 import targets as phase2_targets
from benchmarks.state_preparation.phase2.canonical import load_canonical_json_object
from benchmarks.state_preparation.phase2.execution_bindings import PILOT_METHOD_IDS, SMOKE_METHOD_IDS
from benchmarks.state_preparation.phase2.layerwise_bmpd import build_layerwise_bmpd_crn_v2_template
from benchmarks.state_preparation.phase2.protocol import (
    PRIMARY_TARGET_FAMILIES,
    FinalComparatorRef,
    FinalConfigurationExecutionManifest,
    FinalConfigurationExecutionRef,
    FinalConfirmationSeal,
    InitialPreregistration,
    PrimaryContrastBinding,
    load_initial_preregistration,
)
from benchmarks.state_preparation.phase2.screening_design import (
    ADAPT_STYLE_PUBLICATION_METHOD_ID,
    IMPACT_PRUNING_PUBLICATION_METHOD_ID,
    WP22_PUBLICATION_PRUNING_MAPPING_VERSION,
    WP22CandidateConfiguration,
    build_screening_manifest,
)
from benchmarks.state_preparation.phase2.targets import (
    TargetPopulationManifest,
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.training_orchestration import (
    JOB_ATTEMPTS_DIRECTORY_NAME,
    JOB_RESULT_NAME,
    PILOT_OPTIMIZATION_SEED_COUNT,
    ConfirmExecutionRequest,
    JobExecutionControls,
    TrainingExecutorRegistry,
    TrainingJob,
    TrainingJobOutcome,
    TrainingRunPlan,
    build_confirm_execution_context,
    build_historical_reproduction_plan,
    build_paper_confirm_plan,
    build_paper_pilot_plan,
    build_paper_screen_plan,
    build_training_smoke_plan,
    derive_pilot_optimization_seeds,
    execute_training_plan,
    load_training_job_outcome_history,
    training_job_attempt_path,
    validate_confirm_execution_request,
    validate_job_pipeline_binding,
)
from benchmarks.state_preparation.phase2.training_schedules import (
    CheckpointValidationPolicy,
    LimitedMultistartPlan,
    NoiselessPretrainNoisyFinetune,
    NoiseMixtureComponent,
    NoiseStrengthContinuation,
    StandardNoiseMixture,
    TrainingStrategySchedule,
    TrajectoryCountCurriculum,
    TrajectoryCountStep,
    TrajectorySamplingPolicy,
)

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineTemplate

_DEVELOPMENT_MASTER = bytes(range(32))
_SCREENING_MASTER = bytes(reversed(range(32)))
_SECONDARY_MASTER = bytes((index * 11) % 256 for index in range(32))
_CONFIRMATORY_MASTER = bytes((index * 7) % 256 for index in range(32))


def _checksum(label: str) -> str:
    """Return one deterministic prefixed SHA-256 checksum."""
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


@pytest.fixture(scope="module")
def preregistration() -> InitialPreregistration:
    """Return the trusted Phase II protocol.

    Returns:
        The checked-in preregistration.
    """
    return load_initial_preregistration()


@pytest.fixture(scope="module")
def development_manifest(preregistration: InitialPreregistration) -> TargetPopulationManifest:
    """Build the deterministic development target population.

    Returns:
        The seed-bearing development manifest.
    """
    config = build_target_population_config(
        preregistration,
        "development",
        role_master_entropy_commitment=role_master_entropy_commitment(_DEVELOPMENT_MASTER),
    )
    return create_target_population_manifest(config, preregistration, _DEVELOPMENT_MASTER)


@pytest.fixture(scope="module")
def screening_manifest_targets(preregistration: InitialPreregistration) -> TargetPopulationManifest:
    """Build the deterministic primary q6 screening target population.

    Returns:
        The seed-bearing screening manifest.
    """
    config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(_SCREENING_MASTER),
        population_scope="primary_q6",
    )
    return create_target_population_manifest(config, preregistration, _SCREENING_MASTER)


@pytest.fixture
def secondary_q12_manifest(
    preregistration: InitialPreregistration,
    monkeypatch: pytest.MonkeyPatch,
) -> TargetPopulationManifest:
    """Build the deterministic representative secondary q12 population.

    Returns:
        The seed-bearing secondary-q12 screening-role manifest.
    """

    def cheap_tfim_parameters(
        _master: bytes,
        _target_instance_id: str,
        stratum_id: str,
        qubit_count: int,
    ) -> dict[str, object]:
        """Return shape-valid spectral metadata without dense q12 diagonalization."""
        ratio = {"ferromagnetic": 0.5, "critical": 1.0, "paramagnetic": 1.5}[stratum_id]
        return {
            "attempt_index": 0,
            "couplings": [1.0] * (qubit_count - 1),
            "fields": [ratio] * qubit_count,
            "ground_energy": -float(qubit_count),
            "ground_state_gap": 1.0,
            "gap_threshold": 1e-10 * float(qubit_count),
            "spectral_norm": float(qubit_count),
        }

    monkeypatch.setattr(phase2_targets, "_tfim_parameter_record", cheap_tfim_parameters)
    config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(_SECONDARY_MASTER),
        population_scope="secondary_q12",
    )
    return create_target_population_manifest(config, preregistration, _SECONDARY_MASTER)


def _strategy_schedule(
    method_id: str,
    *,
    noisy: bool = True,
    direct_noisy: bool = False,
) -> TrainingStrategySchedule:
    """Return one complete method-authorizing noisy strategy schedule."""
    continuation = (
        NoiseStrengthContinuation(
            start_update=0,
            end_update=7,
            start_strength_scale=1.0,
            target_strength_scale=1.0,
            interpolation="constant",
        )
        if direct_noisy
        else NoiseStrengthContinuation(start_update=3, end_update=7, target_strength_scale=1.0)
        if noisy
        else NoiseStrengthContinuation(
            start_update=0,
            end_update=7,
            start_strength_scale=0.0,
            target_strength_scale=0.0,
            interpolation="constant",
        )
    )
    return TrainingStrategySchedule(
        schedule_id=f"schedule_{method_id}",
        noise_continuation=continuation,
        trajectory_curriculum=(
            TrajectoryCountCurriculum((TrajectoryCountStep(0, 4), TrajectoryCountStep(4, 8)))
            if noisy
            else TrajectoryCountCurriculum((TrajectoryCountStep(0, 0),))
        ),
        sampling_policy=TrajectorySamplingPolicy("fixed_crn"),
        checkpoint_validation=CheckpointValidationPolicy(patience=3, min_delta=0.01),
        phase_boundary=NoiselessPretrainNoisyFinetune(
            noiseless_pretrain_updates=0 if direct_noisy else 3 if noisy else 8,
            noisy_finetune_updates=8 if direct_noisy else 5 if noisy else 0,
        ),
        multistart=LimitedMultistartPlan(start_count=2, declared_cap=3),
        training_noise=(
            StandardNoiseMixture(
                "matched",
                (NoiseMixtureComponent("depolarizing_1s_all", 1.0),),
            )
            if noisy
            else StandardNoiseMixture("noiseless", ())
        ),
    )


def _all_candidates(preregistration: InitialPreregistration) -> tuple[WP22CandidateConfiguration, ...]:
    """Return the complete nine-method synthetic configuration registry.

    Returns:
        One strict candidate per family-wide preregistered method.
    """
    result: list[WP22CandidateConfiguration] = []
    matching = _checksum("matching")
    for policy in preregistration.candidate_methods:
        if policy["scope"] != "all_families":
            continue
        method = cast("str", policy["method_id"])
        implementation = "topdown_impact_iterative" if method == IMPACT_PRUNING_PUBLICATION_METHOD_ID else method
        mapping: dict[str, object] = {}
        if method == IMPACT_PRUNING_PUBLICATION_METHOD_ID:
            mapping = {
                "mapping_version": WP22_PUBLICATION_PRUNING_MAPPING_VERSION,
                "publication_method_id": method,
                "implementation_method_id": implementation,
                "pruning_rule": "impact_iterative",
                "minimum_pruning_rounds": 2,
                "required_final_finetune_sampling": "crn_fixed",
            }
        schedule = _strategy_schedule(method, noisy=cast("bool", policy["noisy_training"]))
        result.append(
            WP22CandidateConfiguration(
                method_id=method,
                implementation_kind="operator_growth"
                if method == ADAPT_STYLE_PUBLICATION_METHOD_ID
                else "phase2_pipeline",
                implementation_method_id=implementation,
                implementation_schema_version="test_implementation.v1",
                implementation_checksum=_checksum(f"implementation {method}"),
                strategy_schedule_checksum=schedule.content_checksum,
                resource_stratum_id="primary_cap_12",
                noisy_training=cast("bool", policy["noisy_training"]),
                matching_projection_checksum=(
                    matching if method in {"layerwise_bmpd_crn_v2", "layerwise_bmpd_noiseless"} else None
                ),
                publication_mapping=mapping,
            )
        )
    return tuple(result)


def _smoke_candidates(preregistration: InitialPreregistration) -> tuple[WP22CandidateConfiguration, ...]:
    """Return the exact ten-method smoke registry, including analytic Energy-ADAPT.

    Returns:
        The nine family-wide methods plus the TFIM-only analytic smoke route.
    """
    candidates = list(_all_candidates(preregistration))
    energy_schedule = _strategy_schedule("energy_adapt_vqe", noisy=False)
    candidates.append(
        WP22CandidateConfiguration(
            method_id="energy_adapt_vqe",
            implementation_kind="phase2_pipeline",
            implementation_method_id="energy_adapt_vqe",
            implementation_schema_version="test_implementation.v1",
            implementation_checksum=_checksum("implementation energy_adapt_vqe"),
            strategy_schedule_checksum=energy_schedule.content_checksum,
            resource_stratum_id="primary_cap_12",
            noisy_training=False,
            matching_projection_checksum=None,
            publication_mapping={},
        )
    )
    assert {candidate.method_id for candidate in candidates} == set(SMOKE_METHOD_IDS)
    return tuple(candidates)


def _schedules_for(candidates: tuple[WP22CandidateConfiguration, ...]) -> tuple[TrainingStrategySchedule, ...]:
    """Return synthetic schedules matching an exact candidate sequence.

    Returns:
        One schedule per candidate in caller order.
    """
    return tuple(_strategy_schedule(candidate.method_id, noisy=candidate.noisy_training) for candidate in candidates)


def _v2_candidate() -> tuple[TrainingPipelineTemplate, WP22CandidateConfiguration, TrainingStrategySchedule]:
    """Return a genuine v2 template and its WP22 publication wrapper.

    Returns:
        The typed template, candidate wrapper, and complete strategy schedule.
    """
    template = build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=1,
        checkpoint_validation_trajectory_count=1,
    )
    schedule = _strategy_schedule("layerwise_bmpd_crn_v2")
    candidate = WP22CandidateConfiguration.from_pipeline(
        template,
        strategy_schedule_checksum=schedule.content_checksum,
    )
    return template, candidate, schedule


def test_smoke_plan_roundtrip_and_concrete_pipeline_binding(
    preregistration: InitialPreregistration,
    development_manifest: TargetPopulationManifest,
) -> None:
    """The exact ten-cell smoke plan binds its genuine v2 pipeline job."""
    template, candidate, schedule = _v2_candidate()
    candidates = tuple(
        candidate if item.method_id == candidate.method_id else item for item in _smoke_candidates(preregistration)
    )
    schedules = tuple(
        schedule
        if item.method_id == candidate.method_id
        else _strategy_schedule(item.method_id, noisy=item.noisy_training)
        for item in candidates
    )
    plan = build_training_smoke_plan(
        preregistration_checksum=preregistration.content_checksum,
        target_manifest=development_manifest,
        candidates=candidates,
        schedules=schedules,
    )
    assert len(plan.jobs) == 10
    smoke_target = next(
        target
        for target in development_manifest.instances
        if target.qubit_count == 6 and target.family_id == "tfim_ground_state"
    )
    assert {item.target_instance_id for item in plan.jobs} == {smoke_target.target_instance_id}
    assert {item.family_id for item in plan.jobs} == {"tfim_ground_state"}
    assert TrainingRunPlan.from_json(plan.to_json()) == plan
    job = next(item for item in plan.jobs if item.method_id == candidate.method_id)
    assert job.strategy_schedule == schedule
    pipeline = template.resolve(
        target_namespace="phase2",
        target_manifest=development_manifest,
        target_instance_id=job.target_instance_id,
        target_population_manifest_checksum=job.target_manifest_checksum,
        target_instance_spec_checksum=job.target_spec_checksum,
        target_family_id=job.family_id,
        target_stratum_id=job.stratum_id,
        qubit_count=job.qubit_count,
        optimization_block_id=job.optimization_block_id,
        optimization_seed=job.optimization_seed,
        data_role="development",
    )
    validate_job_pipeline_binding(job, candidate, template, pipeline)


def test_smoke_plan_rejects_a_manifest_without_a_q6_tfim_target(
    preregistration: InitialPreregistration,
    development_manifest: TargetPopulationManifest,
) -> None:
    """The common smoke target is mandatory because Energy-ADAPT is TFIM-only."""
    tampered_manifest = TargetPopulationManifest.from_json(development_manifest.to_json())
    object.__setattr__(  # noqa: PLC2801 - deliberately corrupt a frozen record for fail-closed testing
        tampered_manifest,
        "instances",
        tuple(target for target in tampered_manifest.instances if target.family_id != "tfim_ground_state"),
    )

    candidates = _smoke_candidates(preregistration)
    with pytest.raises(ValueError, match="q6 tfim_ground_state"):
        build_training_smoke_plan(
            preregistration_checksum=preregistration.content_checksum,
            target_manifest=tampered_manifest,
            candidates=candidates,
            schedules=_schedules_for(candidates),
        )


def test_strategy_candidates_require_typed_direct_noisy_and_noiseless_schedules(
    preregistration: InitialPreregistration,
    development_manifest: TargetPopulationManifest,
) -> None:
    """Complete direct-control modes survive planning and missing schedules fail."""
    candidates = _smoke_candidates(preregistration)
    noisy = next(candidate for candidate in candidates if candidate.noisy_training)
    direct_noisy = _strategy_schedule(noisy.method_id, noisy=True, direct_noisy=True)
    changed = tuple(
        replace(candidate, strategy_schedule_checksum=direct_noisy.content_checksum)
        if candidate.method_id == noisy.method_id
        else candidate
        for candidate in candidates
    )
    schedules = tuple(
        direct_noisy
        if candidate.method_id == noisy.method_id
        else _strategy_schedule(candidate.method_id, noisy=candidate.noisy_training)
        for candidate in changed
    )
    plan = build_training_smoke_plan(
        preregistration_checksum=preregistration.content_checksum,
        target_manifest=development_manifest,
        candidates=changed,
        schedules=schedules,
    )

    modes = {job.strategy_schedule.phase_boundary.mode for job in plan.jobs if job.strategy_schedule is not None}
    assert {"noiseless_only", "noisy_only"} <= modes
    assert TrainingRunPlan.from_json(plan.to_json()) == plan
    with pytest.raises(TypeError, match="TrainingStrategySchedule"):
        build_training_smoke_plan(
            preregistration_checksum=preregistration.content_checksum,
            target_manifest=development_manifest,
            candidates=changed,
            schedules=(),
        )


def test_dry_run_resume_overwrite_and_failure_ledgers_are_safe(
    tmp_path: Path,
    preregistration: InitialPreregistration,
    development_manifest: TargetPopulationManifest,
) -> None:
    """Attempts are append-only and stale projections never control resume."""
    _, candidate, schedule = _v2_candidate()
    candidates = tuple(
        candidate if item.method_id == candidate.method_id else item for item in _smoke_candidates(preregistration)
    )
    schedules = tuple(
        schedule
        if item.method_id == candidate.method_id
        else _strategy_schedule(item.method_id, noisy=item.noisy_training)
        for item in candidates
    )
    complete_plan = build_training_smoke_plan(
        preregistration_checksum=preregistration.content_checksum,
        target_manifest=development_manifest,
        candidates=candidates,
        schedules=schedules,
    )
    selected_job = next(item for item in complete_plan.jobs if item.method_id == candidate.method_id)
    plan = TrainingRunPlan(
        plan_id=complete_plan.plan_id,
        preset=complete_plan.preset,
        preregistration_checksum=complete_plan.preregistration_checksum,
        target_manifest_checksums=complete_plan.target_manifest_checksums,
        screening_manifest_checksum=None,
        final_confirmation_seal_checksum=None,
        execution_source_checksum=None,
        jobs=(selected_job,),
    )
    output = tmp_path / "wp22"
    calls: list[tuple[str, bool, bool, int]] = []

    def executor(job: TrainingJob, _directory: Path, controls: JobExecutionControls) -> str:
        assert job.strategy_schedule == schedule
        assert controls.schedule_resume_state is not None
        calls.append((
            job.job_id,
            controls.resume,
            controls.overwrite,
            controls.schedule_resume_state.prior_attempt,
        ))
        return _checksum(f"result {len(calls)}")

    summary = execute_training_plan(plan, output, executor, dry_run=True)
    assert summary.planned == 1
    assert summary.attempted == 0
    assert calls == []
    assert not output.exists()

    first = execute_training_plan(plan, output, executor)
    assert (first.succeeded, first.failed, first.skipped) == (1, 0, 0)
    assert calls == [(plan.jobs[0].job_id, False, False, 0)]
    result_path = output / plan.jobs[0].output_path / JOB_RESULT_NAME
    job_directory = output / plan.jobs[0].output_path
    first_outcome = TrainingJobOutcome.from_dict(load_canonical_json_object(result_path.read_text(encoding="utf-8")))
    assert first_outcome.attempt == 1
    assert training_job_attempt_path(job_directory, 1).read_text(encoding="utf-8") == result_path.read_text(
        encoding="utf-8"
    )

    resumed = execute_training_plan(plan, output, executor, resume=True)
    assert resumed.skipped == 1
    assert len(calls) == 1
    overwritten = execute_training_plan(plan, output, executor, overwrite=True)
    assert overwritten.succeeded == 1
    assert len(calls) == 2
    assert calls[-1] == (plan.jobs[0].job_id, False, True, 1)
    second_outcome = TrainingJobOutcome.from_dict(load_canonical_json_object(result_path.read_text(encoding="utf-8")))
    assert second_outcome.attempt == 2
    assert tuple(sorted(path.name for path in (job_directory / JOB_ATTEMPTS_DIRECTORY_NAME).iterdir())) == (
        "attempt_00000001.json",
        "attempt_00000002.json",
    )
    assert load_training_job_outcome_history(job_directory, plan.jobs[0]) == (first_outcome, second_outcome)

    failure_output = tmp_path / "failure"
    private_diagnostic = _DEVELOPMENT_MASTER.hex()

    def fail(_job: TrainingJob, _directory: Path, _controls: JobExecutionControls) -> str:
        msg = private_diagnostic
        raise RuntimeError(msg)

    failed = execute_training_plan(plan, failure_output, fail, fail_fast=True)
    assert failed.failed == 1
    assert failed.succeeded == 0
    failed_path = failure_output / plan.jobs[0].output_path / JOB_RESULT_NAME
    failure = TrainingJobOutcome.from_dict(load_canonical_json_object(failed_path.read_text(encoding="utf-8")))
    assert failure.status == "failure"
    assert failure.exception_type == "executor_failure"
    assert private_diagnostic not in failed_path.read_text(encoding="utf-8")

    recovered = execute_training_plan(plan, failure_output, executor, resume=True)
    assert recovered.succeeded == 1
    failed_again = execute_training_plan(plan, failure_output, fail, overwrite=True)
    assert failed_again.failed == 1
    failure_job_directory = failure_output / plan.jobs[0].output_path
    history = load_training_job_outcome_history(failure_job_directory, plan.jobs[0])
    assert tuple(outcome.status for outcome in history) == ("failure", "success", "failure")

    # Simulate an interrupted latest-projection refresh.  Attempt history is
    # authoritative, so stale success cannot hide the newer failure on resume.
    failed_path.write_bytes(training_job_attempt_path(failure_job_directory, 2).read_bytes())
    retried = execute_training_plan(plan, failure_output, executor, resume=True)
    assert retried.attempted == 1
    assert retried.skipped == 0
    final_history = load_training_job_outcome_history(failure_job_directory, plan.jobs[0])
    assert tuple(outcome.status for outcome in final_history) == ("failure", "success", "failure", "success")
    assert failed_path.read_bytes() == training_job_attempt_path(failure_job_directory, 4).read_bytes()


def test_whole_plan_preflight_rejects_late_unsafe_output_before_dispatch(tmp_path: Path) -> None:
    """A bad final job path cannot cause an earlier job directory or attempt."""
    plan = build_historical_reproduction_plan(
        preregistration_checksum=load_initial_preregistration().content_checksum,
    )
    output = tmp_path / "whole-plan-preflight"
    last_directory = output / plan.jobs[-1].output_path
    last_directory.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    last_directory.symlink_to(outside, target_is_directory=True)
    before = tuple(sorted(path.relative_to(output) for path in output.rglob("*")))
    calls: list[str] = []

    def executor(job: TrainingJob, _directory: Path, _controls: JobExecutionControls) -> str:
        calls.append(job.job_id)
        return _checksum("must not execute")

    with pytest.raises(ValueError, match="non-symlink"):
        execute_training_plan(plan, output, executor)

    assert calls == []
    assert not (output / plan.jobs[0].output_path).exists()
    assert tuple(sorted(path.relative_to(output) for path in output.rglob("*"))) == before


def test_pilot_screen_and_historical_cardinalities_are_deterministic(
    preregistration: InitialPreregistration,
    development_manifest: TargetPopulationManifest,
    screening_manifest_targets: TargetPopulationManifest,
    secondary_q12_manifest: TargetPopulationManifest,
) -> None:
    """Every WP22 fan-out has stable cardinality and canonical ordering."""
    candidates = _all_candidates(preregistration)
    schedules = _schedules_for(candidates)
    pilot_candidates = tuple(
        next(candidate for candidate in candidates if candidate.method_id == method_id)
        for method_id in PILOT_METHOD_IDS
    )
    pilot_schedules = _schedules_for(pilot_candidates)
    pilot_seeds = derive_pilot_optimization_seeds(
        preregistration.content_checksum,
        PILOT_OPTIMIZATION_SEED_COUNT,
    )
    pilot = build_paper_pilot_plan(
        preregistration_checksum=preregistration.content_checksum,
        target_manifests=(secondary_q12_manifest, development_manifest),
        candidates=pilot_candidates,
        schedules=pilot_schedules,
        optimization_seeds=pilot_seeds,
    )
    expected_pilot_targets = len(development_manifest.instances) + len(secondary_q12_manifest.instances)
    assert len(pilot.jobs) == expected_pilot_targets * PILOT_OPTIMIZATION_SEED_COUNT * len(PILOT_METHOD_IDS)
    assert {job.qubit_count for job in pilot.jobs} == {6, 12}
    assert {job.data_role for job in pilot.jobs if job.qubit_count == 6} == {"development"}
    q12_jobs = tuple(job for job in pilot.jobs if job.qubit_count == 12)
    assert {job.data_role for job in q12_jobs} == {"secondary_benchmark"}
    assert all(job.output_path.startswith("roles/secondary_benchmark/") for job in q12_jobs)
    assert {job.target_manifest_checksum for job in q12_jobs} == {secondary_q12_manifest.content_checksum}
    with pytest.raises(ValueError, match="secondary-q12 benchmark"):
        replace(
            q12_jobs[0],
            data_role="screening_selection",
            output_path=q12_jobs[0].output_path.replace("secondary_benchmark", "screening_selection", 1),
        )
    assert pilot.target_manifest_checksums == (
        development_manifest.content_checksum,
        secondary_q12_manifest.content_checksum,
    )
    with pytest.raises(ValueError, match="exactly 5"):
        build_paper_pilot_plan(
            preregistration_checksum=preregistration.content_checksum,
            target_manifests=(development_manifest, secondary_q12_manifest),
            candidates=pilot_candidates,
            schedules=pilot_schedules,
            optimization_seeds=pilot_seeds[:1],
        )
    with pytest.raises(ValueError, match="secondary_q12"):
        build_paper_pilot_plan(
            preregistration_checksum=preregistration.content_checksum,
            target_manifests=(development_manifest,),
            candidates=pilot_candidates,
            schedules=pilot_schedules,
            optimization_seeds=pilot_seeds,
        )

    screening = build_screening_manifest(
        preregistration,
        screening_manifest_targets,
        candidates,
        optimization_seeds=(10, 20, 30),
        screening_seed_root=42,
    )
    screen = build_paper_screen_plan(
        preregistration_checksum=preregistration.content_checksum,
        target_manifest=screening_manifest_targets,
        screening_manifest=screening,
        candidates=tuple(reversed(candidates)),
        schedules=tuple(reversed(schedules)),
    )
    assert len(screen.jobs) == 1_296
    assert TrainingRunPlan.from_json(screen.to_json()) == screen
    repeated = build_paper_screen_plan(
        preregistration_checksum=preregistration.content_checksum,
        target_manifest=screening_manifest_targets,
        screening_manifest=screening,
        candidates=candidates,
        schedules=schedules,
    )
    assert repeated.content_checksum == screen.content_checksum
    assert repeated.jobs == screen.jobs

    historical = build_historical_reproduction_plan(preregistration_checksum=preregistration.content_checksum)
    assert len(historical.jobs) == 5
    assert tuple(job.optimization_seed for job in historical.jobs) == (100, 200, 300, 400, 500)


def _final_seal(
    preregistration: InitialPreregistration,
    target_manifest: TargetPopulationManifest,
) -> tuple[FinalConfirmationSeal, FinalConfigurationExecutionManifest]:
    """Build a strict synthetic seal bound to an actual target manifest.

    Returns:
        The checksum-sealed confirmatory design and its executable configuration manifest.
    """
    promoted = _checksum("promoted")
    v2 = _checksum("v2")
    noiseless = _checksum("noiseless")
    matching = _checksum("matching confirm")
    execution_manifest = FinalConfigurationExecutionManifest(
        manifest_id="wp22_runner_final_configuration_execution",
        entries=tuple(
            sorted(
                (
                    FinalConfigurationExecutionRef(
                        method_id="spsa_layerwise",
                        configuration_schema_version="test_configuration.v1",
                        configuration_checksum=promoted,
                        strategy_schedule=_strategy_schedule("spsa_layerwise"),
                        implementation_checksum=_checksum("implementation promoted"),
                        scoped_binding_checksum=_checksum("scoped promoted"),
                        executable_binding_checksum=_checksum("executable promoted"),
                    ),
                    FinalConfigurationExecutionRef(
                        method_id="layerwise_bmpd_crn_v2",
                        configuration_schema_version="test_configuration.v1",
                        configuration_checksum=v2,
                        strategy_schedule=_strategy_schedule("layerwise_bmpd_crn_v2"),
                        implementation_checksum=_checksum("implementation v2"),
                        scoped_binding_checksum=_checksum("scoped v2"),
                        executable_binding_checksum=_checksum("executable v2"),
                    ),
                    FinalConfigurationExecutionRef(
                        method_id="layerwise_bmpd_noiseless",
                        configuration_schema_version="test_configuration.v1",
                        configuration_checksum=noiseless,
                        strategy_schedule=_strategy_schedule("layerwise_bmpd_noiseless", noisy=False),
                        implementation_checksum=_checksum("implementation noiseless"),
                        scoped_binding_checksum=_checksum("scoped noiseless"),
                        executable_binding_checksum=_checksum("executable noiseless"),
                    ),
                ),
                key=lambda item: (item.configuration_checksum, item.method_id),
            )
        ),
    )
    seal = FinalConfirmationSeal(
        seal_id="wp22-runner-confirm-test",
        preregistration_checksum=preregistration.content_checksum,
        promotion_decision_checksum=_checksum("promotion"),
        promoted_method_id="spsa_layerwise",
        promoted_configuration_checksum=promoted,
        comparators=(
            FinalComparatorRef(
                role="layerwise_v2_reference",
                method_id="layerwise_bmpd_crn_v2",
                configuration_schema_version="test_configuration.v1",
                configuration_checksum=v2,
                matched_to_configuration_checksum=noiseless,
                matching_projection_checksum=matching,
            ),
            FinalComparatorRef(
                role="matched_noiseless_control",
                method_id="layerwise_bmpd_noiseless",
                configuration_schema_version="test_configuration.v1",
                configuration_checksum=noiseless,
                matched_to_configuration_checksum=v2,
                matching_projection_checksum=matching,
            ),
        ),
        primary_contrasts=(
            PrimaryContrastBinding(
                contrast_id="noisy_vs_noiseless",
                treatment_configuration_checksum=v2,
                control_configuration_checksum=noiseless,
                paired_block_policy_checksum=preregistration.paired_block_policy_checksum,
                matching_projection_checksum=matching,
            ),
            PrimaryContrastBinding(
                contrast_id="promoted_vs_layerwise_v2_if_distinct",
                treatment_configuration_checksum=promoted,
                control_configuration_checksum=v2,
                paired_block_policy_checksum=preregistration.paired_block_policy_checksum,
                matching_projection_checksum=None,
            ),
        ),
        confirmatory_target_manifest_checksum=target_manifest.content_checksum,
        target_count_by_family=dict.fromkeys(PRIMARY_TARGET_FAMILIES, 24),
        optimization_seed_count=3,
        fixed_test_trajectory_count=256,
        primary_noise_condition=preregistration.primary_noise_condition,
        primary_resource_budget={
            "metric": preregistration.primary_resource_constraint["metric"],
            "cap_per_chain_edge": preregistration.primary_resource_constraint["cap_per_chain_edge"],
            "normalized_compute_cap": 1_000_000.0,
            "reachable_stratum_manifest_checksum": _checksum("reachable"),
        },
        hyperparameters_checksum=execution_manifest.content_checksum,
        execution_source_checksum=_checksum("source"),
        analysis_template_checksum=preregistration.analysis_template_checksum,
        analysis_source_manifest_checksum=_checksum("analysis source"),
        sample_size_design_checksum=_checksum("sample size"),
        failure_policy_checksum=preregistration.failure_policy_checksum,
    )
    return seal, execution_manifest


def test_confirm_plan_uses_only_sealed_configurations_and_counts(
    tmp_path: Path,
    preregistration: InitialPreregistration,
) -> None:
    """The dormant confirm path expands only the target and method roots in its seal."""
    config = build_target_population_config(
        preregistration,
        "confirmatory",
        role_master_entropy_commitment=role_master_entropy_commitment(_CONFIRMATORY_MASTER),
        confirmatory_target_count_by_family=dict.fromkeys(PRIMARY_TARGET_FAMILIES, 24),
    )
    manifest = create_target_population_manifest(config, preregistration, _CONFIRMATORY_MASTER)
    seal, execution_manifest = _final_seal(preregistration, manifest)
    plan = build_paper_confirm_plan(
        seal=seal,
        target_manifest=manifest,
        configuration_execution_manifest=execution_manifest,
    )
    assert len(plan.jobs) == 24 * 4 * 3 * 3
    assert {job.candidate_configuration_checksum for job in plan.jobs} == {
        seal.promoted_configuration_checksum,
        *(item.configuration_checksum for item in seal.comparators),
    }
    assert {job.implementation_kind for job in plan.jobs} == {"sealed_configuration"}
    execution_by_configuration = {item.configuration_checksum: item for item in execution_manifest.entries}
    for sealed_job in plan.jobs:
        execution = execution_by_configuration[sealed_job.candidate_configuration_checksum]
        assert sealed_job.implementation_checksum == execution.implementation_checksum
        assert sealed_job.strategy_schedule_checksum == execution.strategy_schedule_checksum
        sealed_request = sealed_job.confirm_execution_request
        assert sealed_request is not None
        assert sealed_request.configuration_execution_manifest_checksum == execution_manifest.content_checksum
        assert sealed_request.hyperparameters_checksum == execution.strategy_schedule_checksum
        assert sealed_request.implementation_checksum == execution.implementation_checksum
        assert sealed_request.scoped_binding_checksum == execution.scoped_binding_checksum
        assert sealed_request.executable_binding_checksum == execution.executable_binding_checksum
        assert sealed_request.hyperparameters_checksum != seal.hyperparameters_checksum
    assert plan.final_confirmation_seal_checksum == seal.content_checksum
    assert plan.execution_source_checksum == seal.execution_source_checksum
    request = plan.jobs[0].confirm_execution_request
    assert isinstance(request, ConfirmExecutionRequest)
    assert ConfirmExecutionRequest.from_json(request.to_json()) == request
    assert request.fixed_test_trajectory_count == seal.fixed_test_trajectory_count
    assert request.primary_noise_condition == seal.primary_noise_condition
    assert request.primary_resource_budget == seal.primary_resource_budget
    assert request.analysis_template_checksum == seal.analysis_template_checksum
    assert request.analysis_source_manifest_checksum == seal.analysis_source_manifest_checksum
    context = build_confirm_execution_context(seal, manifest, execution_manifest)
    validate_confirm_execution_request(request, context)
    for field_name in (
        "configuration_execution_manifest_checksum",
        "implementation_checksum",
        "hyperparameters_checksum",
        "scoped_binding_checksum",
        "executable_binding_checksum",
    ):
        with pytest.raises(ValueError, match="exact final-seal cell"):
            validate_confirm_execution_request(
                replace(request, **{field_name: _checksum(f"changed {field_name}")}),
                context,
            )
    changed_entry = replace(
        execution_manifest.entries[0],
        implementation_checksum=_checksum("changed final implementation"),
    )
    changed_execution_manifest = FinalConfigurationExecutionManifest(
        manifest_id=execution_manifest.manifest_id,
        entries=(changed_entry, *execution_manifest.entries[1:]),
    )
    with pytest.raises(ValueError, match="hyperparameters root"):
        build_confirm_execution_context(seal, manifest, changed_execution_manifest)

    job = plan.jobs[0]
    single = TrainingRunPlan(
        plan_id=plan.plan_id,
        preset=plan.preset,
        preregistration_checksum=plan.preregistration_checksum,
        target_manifest_checksums=plan.target_manifest_checksums,
        screening_manifest_checksum=plan.screening_manifest_checksum,
        final_confirmation_seal_checksum=plan.final_confirmation_seal_checksum,
        execution_source_checksum=plan.execution_source_checksum,
        jobs=(job,),
    )
    output = tmp_path / "typed-confirm"
    received: list[ConfirmExecutionRequest] = []

    def confirm_executor(
        typed_request: ConfirmExecutionRequest,
        _directory: Path,
        controls: JobExecutionControls,
    ) -> str:
        """Record one seal-complete low-cost confirm dispatch.

        Returns:
            A deterministic result checksum.
        """
        assert controls.schedule_resume_state is None
        received.append(typed_request)
        return _checksum("confirm result")

    with pytest.raises(TypeError, match="typed confirm_executor"):
        execute_training_plan(single, output, lambda _job, _directory, _controls: _checksum("untyped"))
    assert not output.exists()
    summary = execute_training_plan(
        single,
        output,
        TrainingExecutorRegistry(confirm_executor=confirm_executor),
    )
    assert summary.succeeded == 1
    assert received == [request]
    with pytest.raises(ValueError, match="first terminal attempt"):
        execute_training_plan(
            single,
            tmp_path / "confirm-overwrite-must-not-exist",
            TrainingExecutorRegistry(confirm_executor=confirm_executor),
            overwrite=True,
        )
    assert not (tmp_path / "confirm-overwrite-must-not-exist").exists()

    failed_output = tmp_path / "typed-confirm-failure"

    def failing_confirm_executor(
        _typed_request: ConfirmExecutionRequest,
        _directory: Path,
        _controls: JobExecutionControls,
    ) -> str:
        """Fail one terminal confirm attempt.

        Raises:
            RuntimeError: Always, to exercise frozen failure semantics.
        """
        msg = "terminal confirm failure"
        raise RuntimeError(msg)

    failed = execute_training_plan(
        single,
        failed_output,
        TrainingExecutorRegistry(confirm_executor=failing_confirm_executor),
    )
    assert failed.failed == 1
    preserved = execute_training_plan(
        single,
        failed_output,
        TrainingExecutorRegistry(confirm_executor=confirm_executor),
        resume=True,
    )
    assert (preserved.attempted, preserved.skipped) == (0, 1)
    failure_history = load_training_job_outcome_history(failed_output / job.output_path, job)
    assert len(failure_history) == 1
    assert failure_history[0].status == "failure"

    cli_output = tmp_path / "cli-confirm"
    cli_options = training_cli.resolve_options(
        training_cli.parse_arguments([
            "--preset",
            "paper-confirm",
            "--target-manifest",
            "externally-custodied.json",
            "--final-seal",
            "seal.json",
            "--configuration-execution-manifest",
            "execution-configurations.json",
            "--execution-source-manifest",
            "execution.json",
            "--analysis-source-manifest",
            "analysis.json",
            "--execute-expensive",
            "--output",
            str(cli_output),
        ])
    )
    with pytest.raises(
        training_cli.TrainingRunnerConfigurationError,
        match="programmatic context or executor injection",
    ):
        training_cli.run(
            cli_options,
            executor=TrainingExecutorRegistry(confirm_executor=confirm_executor),
        )
    assert not cli_output.exists()
    assert received == [request]
