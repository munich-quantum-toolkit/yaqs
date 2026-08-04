# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for checksum-sealed WP22 pilot and sample-size evidence."""

from __future__ import annotations

import math
from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.pilot import (
    FROZEN_CONTRAST_IDS,
    PilotContrastBinding,
    PilotDesignInfeasibleError,
    PilotEvaluationEvidence,
    PilotJobResult,
    PilotNuisanceSummary,
    PilotObservation,
    build_cluster_aware_paired_difference_v1,
    build_pilot_nuisance_summary,
    reestimate_cluster_aware_paired_difference_v1,
)
from benchmarks.state_preparation.phase2.protocol import (
    PRIMARY_FAMILY_STRATA,
    PRIMARY_TARGET_FAMILIES,
    SampleSizeDesign,
    load_initial_preregistration,
)
from benchmarks.state_preparation.phase2.result_custody import production_noisy_fidelity
from benchmarks.state_preparation.phase2.screening import WP22CandidateConfiguration
from benchmarks.state_preparation.phase2.targets import (
    TargetPopulationManifest,
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.training_orchestration import (
    PILOT_OPTIMIZATION_SEED_COUNT,
    TrainingJob,
    TrainingJobOutcome,
    TrainingRunPlan,
    build_paper_pilot_plan,
    derive_pilot_optimization_seeds,
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
from tests.benchmarks.wp22_pilot_test_support import (
    production_pilot_custody_fixture,
    production_pilot_record,
)

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.state_preparation.phase2.protocol import InitialPreregistration

_DEVELOPMENT_MASTER = bytes(range(32))
_SECONDARY_MASTER = bytes(reversed(range(32)))


def _synthetic_tfim_parameters(
    _master: bytes,
    _target_id: str,
    stratum_id: str,
    qubit_count: int,
) -> dict[str, object]:
    """Return inexpensive schema-valid TFIM metadata for q12 custody tests."""
    ratio = {"ferromagnetic": 0.5, "critical": 1.0, "paramagnetic": 1.5}[stratum_id]
    return {
        "attempt_index": 0,
        "couplings": [1.0] * (qubit_count - 1),
        "fields": [ratio] * qubit_count,
        "ground_energy": -1.0,
        "ground_state_gap": 1.0,
        "gap_threshold": 1e-10,
        "spectral_norm": 1.0,
    }


def _strategy_schedule(method_id: str, *, noisy: bool) -> TrainingStrategySchedule:
    """Return one complete direct-mode schedule for a synthetic pilot candidate."""
    return TrainingStrategySchedule(
        schedule_id=f"pilot_schedule_{method_id}",
        noise_continuation=NoiseStrengthContinuation(
            start_update=0,
            end_update=7,
            start_strength_scale=1.0 if noisy else 0.0,
            target_strength_scale=1.0 if noisy else 0.0,
            interpolation="constant",
        ),
        trajectory_curriculum=TrajectoryCountCurriculum(
            (TrajectoryCountStep(0, 8 if noisy else 0),),
        ),
        sampling_policy=TrajectorySamplingPolicy("fixed_crn"),
        checkpoint_validation=CheckpointValidationPolicy(patience=3, min_delta=0.01),
        phase_boundary=NoiselessPretrainNoisyFinetune(
            noiseless_pretrain_updates=0 if noisy else 8,
            noisy_finetune_updates=8 if noisy else 0,
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


def _candidate(
    method_id: str,
    schedule: TrainingStrategySchedule,
    *,
    noisy: bool,
) -> WP22CandidateConfiguration:
    """Return one exact executable pilot candidate configuration."""
    matching = canonical_checksum({"matching_projection": "layerwise_pair"})
    return WP22CandidateConfiguration(
        method_id=method_id,
        implementation_kind="phase2_pipeline",
        implementation_method_id=method_id,
        implementation_schema_version="test_pilot_implementation.v1",
        implementation_checksum=canonical_checksum({"implementation": method_id}),
        strategy_schedule_checksum=schedule.content_checksum,
        resource_stratum_id="primary_cap_12",
        noisy_training=noisy,
        matching_projection_checksum=(
            matching if method_id in {"layerwise_bmpd_crn_v2", "layerwise_bmpd_noiseless"} else None
        ),
        publication_mapping={},
    )


@cache
def _pilot_context_with_secondary_master(
    secondary_master: bytes,
) -> tuple[
    InitialPreregistration,
    TargetPopulationManifest,
    TargetPopulationManifest,
    TrainingRunPlan,
    tuple[PilotContrastBinding, ...],
]:
    """Return one plan whose q12 archive uses the supplied entropy.

    Raises:
        ValueError: If the secondary entropy does not contain 32 bytes.
    """
    if len(secondary_master) != 32:
        msg = "secondary_master must contain exactly 32 bytes."
        raise ValueError(msg)
    preregistration = load_initial_preregistration()
    config = build_target_population_config(
        preregistration,
        "development",
        role_master_entropy_commitment=role_master_entropy_commitment(_DEVELOPMENT_MASTER),
    )
    target_manifest = create_target_population_manifest(config, preregistration, _DEVELOPMENT_MASTER)
    supplemental_config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(secondary_master),
        population_scope="secondary_q12",
    )
    with patch(
        "benchmarks.state_preparation.phase2.targets._tfim_parameter_record",
        side_effect=_synthetic_tfim_parameters,
    ):
        supplemental_target_manifest = create_target_population_manifest(
            supplemental_config,
            preregistration,
            secondary_master,
        )
    method_ids = (
        "layerwise_bmpd_crn_v2",
        "layerwise_bmpd_noiseless",
        "fixed_depth_bmpd_crn",
    )
    schedules = tuple(
        _strategy_schedule(method_id, noisy=method_id != "layerwise_bmpd_noiseless") for method_id in method_ids
    )
    candidates = tuple(
        _candidate(method_id, schedule, noisy=method_id != "layerwise_bmpd_noiseless")
        for method_id, schedule in zip(method_ids, schedules, strict=True)
    )
    seeds = derive_pilot_optimization_seeds(
        preregistration.content_checksum,
        PILOT_OPTIMIZATION_SEED_COUNT,
    )
    pilot_plan = build_paper_pilot_plan(
        preregistration_checksum=preregistration.content_checksum,
        target_manifests=(target_manifest, supplemental_target_manifest),
        candidates=candidates,
        schedules=schedules,
        optimization_seeds=seeds,
    )
    promoted = next(candidate for candidate in candidates if candidate.method_id == "fixed_depth_bmpd_crn")
    bindings = (
        PilotContrastBinding.noisy_vs_noiseless(pilot_plan),
        PilotContrastBinding.promoted_vs_layerwise_v2(
            pilot_plan,
            treatment_method_id=promoted.method_id,
            treatment_configuration_checksum=promoted.content_checksum,
        ),
    )
    return preregistration, target_manifest, supplemental_target_manifest, pilot_plan, bindings


@cache
def _pilot_context() -> tuple[
    InitialPreregistration,
    TargetPopulationManifest,
    TargetPopulationManifest,
    TrainingRunPlan,
    tuple[PilotContrastBinding, ...],
]:
    """Return one exact manifest, plan, and prescribed contrast universe."""
    return _pilot_context_with_secondary_master(_SECONDARY_MASTER)


def _symmetric_samples(mean: float, variance: float, *, count: int) -> tuple[float, ...]:
    """Return a symmetric raw sample with the requested mean and sample variance."""
    assert count >= 2
    assert count % 2 == 0
    radius = math.sqrt(variance * (count - 1) / count)
    return (mean - radius,) * (count // 2) + (mean + radius,) * (count // 2)


def _pilot_job_evidence(
    job: TrainingJob,
    *,
    fidelity: float,
    failed: bool,
    gradient_variance: float,
    trajectory_mc_variance: float,
    wall_time_seconds: float,
    tracemalloc_peak_bytes: int,
) -> tuple[TrainingJobOutcome, PilotJobResult]:
    """Return linked typed outcome/result evidence for one synthetic job."""
    if failed:
        outcome = TrainingJobOutcome(
            job_checksum=job.content_checksum,
            status="failure",
            result_artifact_checksum=None,
            exception_type="synthetic_failure",
            message="Synthetic pilot failure.",
            attempt=1,
        )
        result = PilotJobResult.failure(
            job,
            outcome,
            gradient_variance=gradient_variance,
            trajectory_mc_variance=trajectory_mc_variance,
            trajectory_count=1024,
            wall_time_seconds=wall_time_seconds,
            tracemalloc_peak_bytes=tracemalloc_peak_bytes,
        )
        return outcome, result
    result = PilotJobResult.success(
        job,
        evaluation_evidence=PilotEvaluationEvidence(
            job_checksum=job.content_checksum,
            fresh_test_trajectory_fidelities=_symmetric_samples(
                fidelity,
                trajectory_mc_variance,
                count=1024,
            ),
            gradient_samples=tuple((value,) for value in _symmetric_samples(0.0, gradient_variance, count=32)),
        ),
        wall_time_seconds=wall_time_seconds,
        tracemalloc_peak_bytes=tracemalloc_peak_bytes,
    )
    outcome = TrainingJobOutcome(
        job_checksum=job.content_checksum,
        status="success",
        result_artifact_checksum=result.content_checksum,
        exception_type=None,
        message=None,
        attempt=1,
    )
    return outcome, result


@cache
def _pilot_observations(
    *,
    difference_scale: float = 0.0001,
    trajectory_mc_variance: float = 0.0025,
    failed_cell: tuple[str, str, int, str] | None = None,
) -> tuple[PilotObservation, ...]:
    """Return complete balanced synthetic clustered pilot evidence."""
    _preregistration, target_manifest, _supplemental, pilot_plan, bindings = _pilot_context()
    seeds = tuple(sorted({job.optimization_seed for job in pilot_plan.jobs}))
    jobs = {(job.target_instance_id, job.optimization_seed, job.method_id): job for job in pilot_plan.jobs}
    binding_by_id = {binding.contrast_id: binding for binding in bindings}
    failed_job_key: tuple[str, int, str] | None = None
    if failed_cell is not None:
        target_id, contrast_id, optimization_seed, _family_id = failed_cell
        failed_job_key = (
            target_id,
            optimization_seed,
            binding_by_id[contrast_id].treatment_method_id,
        )
    evidence_by_job: dict[str, tuple[TrainingJobOutcome, PilotJobResult]] = {}
    for target_ordinal, target in enumerate(target_manifest.instances):
        target_effect = difference_scale * (-1.0 if target_ordinal % 2 == 0 else 1.0)
        family_index = PRIMARY_TARGET_FAMILIES.index(target.family_id)
        for seed_index, optimization_seed in enumerate(seeds):
            seed_effect = difference_scale * (seed_index - 2) / 4
            fidelities = {
                "layerwise_bmpd_noiseless": 0.5,
                "layerwise_bmpd_crn_v2": 0.53 + target_effect + seed_effect,
                "fixed_depth_bmpd_crn": 0.545 + 2 * target_effect + 2 * seed_effect,
            }
            for method_id, fidelity in fidelities.items():
                job = jobs[target.target_instance_id, optimization_seed, method_id]
                evidence_by_job[job.content_checksum] = _pilot_job_evidence(
                    job,
                    fidelity=fidelity,
                    failed=(target.target_instance_id, optimization_seed, method_id) == failed_job_key,
                    gradient_variance=0.01 + 0.001 * family_index,
                    trajectory_mc_variance=trajectory_mc_variance / 2,
                    wall_time_seconds=(0.1 + target_ordinal / 1_000) / 2,
                    tracemalloc_peak_bytes=1_000_000 + target_ordinal,
                )
    observations: list[PilotObservation] = []
    for target in target_manifest.instances:
        for optimization_seed in seeds:
            for binding in bindings:
                treatment_job = jobs[
                    target.target_instance_id,
                    optimization_seed,
                    binding.treatment_method_id,
                ]
                comparator_job = jobs[
                    target.target_instance_id,
                    optimization_seed,
                    binding.comparator_method_id,
                ]
                treatment_outcome, treatment_result = evidence_by_job[treatment_job.content_checksum]
                comparator_outcome, comparator_result = evidence_by_job[comparator_job.content_checksum]
                observations.append(
                    PilotObservation.from_paired_job_evidence(
                        contrast_id=binding.contrast_id,
                        treatment_job=treatment_job,
                        treatment_outcome=treatment_outcome,
                        treatment_result=treatment_result,
                        comparator_job=comparator_job,
                        comparator_outcome=comparator_outcome,
                        comparator_result=comparator_result,
                    )
                )
    return tuple(observations)


def _build_summary(
    observations: tuple[PilotObservation, ...],
    *,
    summary_id: str = "phase2_pilot_nuisance_v1",
) -> PilotNuisanceSummary:
    """Bind observations to the exact cached pilot custody artifacts.

    Returns:
        The complete checksum-sealed nuisance summary.
    """
    preregistration, target_manifest, supplemental, pilot_plan, bindings = _pilot_context()
    return build_pilot_nuisance_summary(
        preregistration,
        target_manifest,
        supplemental,
        pilot_plan,
        bindings,
        observations,
        summary_id=summary_id,
    )


@pytest.fixture
def preregistration() -> InitialPreregistration:
    """Return the trusted Phase II protocol."""
    return load_initial_preregistration()


@pytest.fixture
def pilot_summary(preregistration: InitialPreregistration) -> PilotNuisanceSummary:
    """Return a feasible complete pilot summary."""
    assert preregistration == _pilot_context()[0]
    return _build_summary(_pilot_observations())


def test_pilot_summary_binds_exact_two_manifest_plan_seeds_and_contrasts(
    pilot_summary: PilotNuisanceSummary,
) -> None:
    """Only exact q6 jobs enter inference while q12 and contrast custody remain sealed."""
    preregistration, target_manifest, supplemental, pilot_plan, bindings = _pilot_context()
    expected_seeds = derive_pilot_optimization_seeds(
        preregistration.content_checksum,
        PILOT_OPTIMIZATION_SEED_COUNT,
    )
    assert pilot_summary.target_manifest == target_manifest
    assert pilot_summary.supplemental_target_manifest == supplemental
    assert pilot_summary.pilot_plan == pilot_plan
    assert pilot_plan.target_manifest_checksums == (
        target_manifest.content_checksum,
        supplemental.content_checksum,
    )
    assert {observation.optimization_seed for observation in pilot_summary.observations} == set(expected_seeds)
    assert all(
        observation.qubit_count == 6
        and observation.treatment_job.target_manifest_checksum == target_manifest.content_checksum
        for observation in pilot_summary.observations
    )
    assert any(
        job.qubit_count == 12
        and job.target_manifest_checksum == supplemental.content_checksum
        and supplemental.data_role == "screening_selection"
        and supplemental.population_scope == "secondary_q12"
        and job.data_role == "secondary_benchmark"
        and job.output_path.startswith("roles/secondary_benchmark/")
        for job in pilot_plan.jobs
    )
    for contrast_id in FROZEN_CONTRAST_IDS:
        for family_id in PRIMARY_TARGET_FAMILIES:
            component = cast(
                "dict[str, object]",
                cast("dict[str, object]", pilot_summary.nuisance_by_contrast[contrast_id])[family_id],
            )
            assert component["target_count"] == 12
            assert component["optimization_seed_count"] == 5

    noisy, promoted = bindings
    assert (noisy.treatment_method_id, noisy.comparator_method_id) == (
        "layerwise_bmpd_crn_v2",
        "layerwise_bmpd_noiseless",
    )
    forged_binding = replace(
        promoted,
        treatment_configuration_checksum=noisy.comparator_configuration_checksum,
    )
    with pytest.raises(ValueError, match="only and all contrast-bound configurations"):
        replace(pilot_summary, contrast_bindings=(noisy, forged_binding))

    noisy_observation = next(
        observation for observation in pilot_summary.observations if observation.contrast_id == "noisy_vs_noiseless"
    )
    with pytest.raises(ValueError, match="promoted planning rows"):
        replace(noisy_observation, contrast_id="promoted_vs_layerwise_v2_if_distinct")


def test_q12_archive_identity_cannot_change_q6_inference_or_design(
    preregistration: InitialPreregistration,
    pilot_summary: PilotNuisanceSummary,
) -> None:
    """Changed q12 manifest and raw-archive bytes remain audit-only."""
    _, primary_manifest, supplemental, pilot_plan, bindings = _pilot_context_with_secondary_master(
        bytes((index + 17) % 256 for index in range(32)),
    )
    changed = build_pilot_nuisance_summary(
        preregistration,
        primary_manifest,
        supplemental,
        pilot_plan,
        bindings,
        pilot_summary.observations,
        summary_id=pilot_summary.summary_id,
    )
    original_archive = canonical_checksum({
        "secondary_manifest": pilot_summary.supplemental_target_manifest.content_checksum,
        "q12_raw_trajectory_fidelities": [0.125] * 256,
    })
    changed_archive = canonical_checksum({
        "secondary_manifest": supplemental.content_checksum,
        "q12_raw_trajectory_fidelities": [0.875] * 256,
    })

    assert original_archive != changed_archive
    assert pilot_summary.content_checksum != changed.content_checksum
    assert pilot_summary.inference_projection == changed.inference_projection
    assert pilot_summary.inference_checksum == changed.inference_checksum
    original_design = build_cluster_aware_paired_difference_v1(preregistration, pilot_summary)
    changed_design = build_cluster_aware_paired_difference_v1(preregistration, changed)
    assert original_design.to_json() == changed_design.to_json()
    assert original_design.pilot_nuisance_summary_checksum == pilot_summary.inference_checksum


def test_production_pilot_records_reopen_q6_success_failure_and_q12_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Representative pilot rows originate in immutable typed WP22E attempts."""
    monkeypatch.setattr(
        "benchmarks.state_preparation.phase2.production_executors.os.fsync",
        lambda _descriptor: None,
    )
    _preregistration, _primary, _supplemental, plan, bindings = _pilot_context()
    primary_jobs = [job for job in plan.jobs if job.qubit_count == 6]
    secondary_job = next(job for job in plan.jobs if job.qubit_count == 12)
    binding = bindings[0]
    anchor = primary_jobs[0]
    successful_source = next(
        job
        for job in primary_jobs
        if job.target_instance_id == anchor.target_instance_id
        and job.optimization_seed == anchor.optimization_seed
        and job.method_id == binding.comparator_method_id
    )
    failed_source = next(
        job
        for job in primary_jobs
        if job.target_instance_id == anchor.target_instance_id
        and job.optimization_seed == anchor.optimization_seed
        and job.method_id == binding.treatment_method_id
    )

    successful = production_pilot_record(successful_source, tmp_path / "q6-success", status="success")
    assert successful.outcome.status == "success"
    assert successful.result_custody.trajectory_fidelities == (0.75,) * 1024
    assert len(successful.result_custody.pilot_diagnostics) == 1
    diagnostic = successful.result_custody.pilot_diagnostics[0]
    assert len(diagnostic.member_seeds) == len(diagnostic.pathwise_update_vectors) == 32
    assert successful.pilot_result is not None
    assert successful.pilot_result.status == "success"
    assert (
        successful.pilot_result.source_result_reference_checksum == successful.result_custody.reference.content_checksum
    )

    with pytest.raises(ValueError, match="frozen seeds"):
        production_pilot_record(
            successful_source,
            tmp_path / "q6-self-consistent-wrong-seed",
            status="success",
            diagnostic_seed_offset=1,
        )

    failed = production_pilot_record(failed_source, tmp_path / "q6-failure", status="failure")
    assert failed.outcome.status == "failure"
    assert failed.result_custody.trajectory_fidelities is None
    assert failed.result_custody.pilot_diagnostics == ()
    assert failed.pilot_result is not None
    assert failed.pilot_result.status == "failure"
    assert failed.pilot_result.fresh_test_noisy_fidelity is None
    observation = PilotObservation.from_paired_job_evidence(
        contrast_id=binding.contrast_id,
        treatment_job=failed.job,
        treatment_outcome=failed.outcome,
        treatment_result=failed.pilot_result,
        comparator_job=successful.job,
        comparator_outcome=successful.outcome,
        comparator_result=successful.pilot_result,
    )
    assert observation.treatment_intention_to_treat_fidelity == pytest.approx(0.0, abs=0.0)
    assert observation.fidelity_difference == pytest.approx(
        -observation.comparator_intention_to_treat_fidelity,
    )

    secondary = production_pilot_record(secondary_job, tmp_path / "q12-success", status="success")
    assert secondary.outcome.status == "success"
    assert secondary.result_custody.trajectory_fidelities == (0.75,) * 256
    assert secondary.result_custody.pilot_diagnostics == ()
    assert secondary.pilot_result is None


def test_complete_production_pilot_custody_uses_all_1080_jobs_but_only_q6_for_inference(
    tmp_path: Path,
) -> None:
    """The aggregate replay has the exact split and keeps q12 evidence audit-only."""
    _preregistration, _primary, _supplemental, _plan, bindings = _pilot_context()
    observations = _pilot_observations()
    first = production_pilot_custody_fixture(
        tmp_path,
        secondary_archive_marker="secondary-archive-a",
    )
    second = production_pilot_custody_fixture(
        tmp_path,
        secondary_archive_marker="secondary-archive-b",
    )
    assert len(first.records) == 1_080
    assert sum(record.job.qubit_count == 6 for record in first.records) == 720
    assert sum(record.job.qubit_count == 12 for record in first.records) == 360
    assert first.secondary_archive_checksum != second.secondary_archive_checksum
    expected = _build_summary(observations)
    assert first.build_nuisance_summary(bindings).to_json() == expected.to_json()
    assert second.build_nuisance_summary(bindings).to_json() == expected.to_json()


def test_pilot_observation_round_trip_and_strict_validation() -> None:
    """Nested job/result evidence round-trips and retains strict diagnostics."""
    observation = _pilot_observations()[0]
    assert PilotObservation.from_json(observation.to_json()) == observation
    assert PilotJobResult.from_json(observation.treatment_result.to_json()) == observation.treatment_result
    assert observation.fidelity_difference == pytest.approx(
        cast("float", observation.treatment_result.fresh_test_noisy_fidelity)
        - cast("float", observation.comparator_result.fresh_test_noisy_fidelity)
    )

    assert observation.treatment_result.evaluation_evidence is not None
    with pytest.raises(ValueError, match="must be finite"):
        replace(
            observation.treatment_result.evaluation_evidence,
            gradient_samples=((0.0,), (math.inf,)),
        )
    with pytest.raises(ValueError, match=r"q=6|primary-q6"):
        replace(observation, treatment_job=replace(observation.treatment_job, qubit_count=12))
    with pytest.raises(ValueError, match="first randomized attempt"):
        replace(observation, treatment_outcome=replace(observation.treatment_outcome, attempt=2))


def _rebuild_success_evidence(
    job: TrainingJob,
    result: PilotJobResult,
) -> tuple[TrainingJobOutcome, PilotJobResult]:
    """Rebind one successful result to a changed job for mismatch tests.

    Returns:
        The changed job's linked outcome and result.
    """
    assert result.status == "success"
    assert result.evaluation_evidence is not None
    rebound = PilotJobResult.success(
        job,
        evaluation_evidence=replace(result.evaluation_evidence, job_checksum=job.content_checksum),
        wall_time_seconds=result.wall_time_seconds,
        tracemalloc_peak_bytes=result.tracemalloc_peak_bytes,
    )
    outcome = TrainingJobOutcome(
        job_checksum=job.content_checksum,
        status="success",
        result_artifact_checksum=rebound.content_checksum,
        exception_type=None,
        message=None,
        attempt=1,
    )
    return outcome, rebound


def _mismatched_observation(
    observation: PilotObservation,
    field: str,
    replacement: object,
) -> PilotObservation:
    """Attempt to rebuild an observation with one changed comparator field.

    Returns:
        The changed observation when the mismatch is unexpectedly accepted.
    """
    changed_job = replace(observation.comparator_job, **{field: replacement})
    changed_outcome, changed_result = _rebuild_success_evidence(changed_job, observation.comparator_result)
    return replace(
        observation,
        comparator_job=changed_job,
        comparator_outcome=changed_outcome,
        comparator_result=changed_result,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("target_manifest_checksum", canonical_checksum({"manifest": "other"})),
        ("target_spec_checksum", canonical_checksum({"target": "other"})),
        ("target_instance_id", "other-pilot-target"),
        ("family_id", "haar_random"),
        ("stratum_id", "uniform"),
        ("qubit_count", 12),
        ("optimization_block_id", "other-pilot-block"),
        ("optimization_seed", 999),
    ],
)
def test_pilot_adapter_rejects_every_paired_cell_mismatch(field: str, replacement: object) -> None:
    """Target, stratum, q6, block, and optimization seed must agree mechanically."""
    observation = _pilot_observations()[0]
    with pytest.raises(ValueError, match=r"q=6|primary-q6|not registered|disagree on required cell fields"):
        _mismatched_observation(observation, field, replacement)


def test_pilot_adapter_derives_itt_and_rejects_resealed_fidelity() -> None:
    """Failure imputation comes from the typed outcome and success artifacts stay linked."""
    successful = _pilot_observations()[0]
    failed_key = (
        successful.target_instance_id,
        successful.contrast_id,
        successful.optimization_seed,
        successful.family_id,
    )
    failed = _pilot_observations(failed_cell=failed_key)[0]
    assert failed.treatment_failed is True
    assert failed.comparator_failed is False
    assert failed.treatment_intention_to_treat_fidelity == pytest.approx(0.0, abs=0.0)
    assert failed.fidelity_difference == pytest.approx(-failed.comparator_intention_to_treat_fidelity)

    comparator_outcome, comparator_result = _pilot_job_evidence(
        successful.comparator_job,
        fidelity=0.0,
        failed=True,
        gradient_variance=successful.comparator_result.gradient_variance,
        trajectory_mc_variance=successful.comparator_result.trajectory_mc_variance,
        wall_time_seconds=successful.comparator_result.wall_time_seconds,
        tracemalloc_peak_bytes=successful.comparator_result.tracemalloc_peak_bytes,
    )
    comparator_failed = replace(
        successful,
        comparator_outcome=comparator_outcome,
        comparator_result=comparator_result,
    )
    assert comparator_failed.treatment_failed is False
    assert comparator_failed.comparator_failed is True
    assert comparator_failed.fidelity_difference == pytest.approx(
        comparator_failed.treatment_intention_to_treat_fidelity
    )

    resealed_scalar = replace(successful.treatment_result, fresh_test_noisy_fidelity=0.99)
    assert resealed_scalar == successful.treatment_result
    assert resealed_scalar.fresh_test_noisy_fidelity == pytest.approx(
        successful.treatment_result.fresh_test_noisy_fidelity,
    )

    assert successful.treatment_result.evaluation_evidence is not None
    forged_evidence = replace(
        successful.treatment_result.evaluation_evidence,
        fresh_test_trajectory_fidelities=(0.98, 0.99),
    )
    forged_result = replace(successful.treatment_result, evaluation_evidence=forged_evidence)
    with pytest.raises(ValueError, match="does not checksum-address"):
        replace(successful, treatment_result=forged_result)


def test_pilot_fidelity_replays_the_production_float64_reducer_exactly() -> None:
    """A 1-ULP summation edge case cannot change pilot inference bytes."""
    fidelities = (0.1,) * 1_024
    evidence = PilotEvaluationEvidence(
        job_checksum=canonical_checksum({"job": "float64-reducer"}),
        fresh_test_trajectory_fidelities=fidelities,
        gradient_samples=((0.0,),) * 32,
    )

    assert float(evidence.fresh_test_noisy_fidelity).hex() == float(production_noisy_fidelity(fidelities)).hex()
    assert float(evidence.fresh_test_noisy_fidelity).hex() != float(math.fsum(fidelities) / len(fidelities)).hex()


def test_pilot_summary_is_order_independent_and_preserves_failures() -> None:
    """Summary identity depends on cells, not input ordering, and keeps failures."""
    observations = list(_pilot_observations())
    failed_observation = next(
        item for item in observations if item.contrast_id == "promoted_vs_layerwise_v2_if_distinct"
    )
    failed_key = (
        failed_observation.target_instance_id,
        failed_observation.contrast_id,
        failed_observation.optimization_seed,
        failed_observation.family_id,
    )
    observations = list(_pilot_observations(failed_cell=failed_key))
    first = _build_summary(tuple(observations))
    observations.reverse()
    shuffled = _build_summary(tuple(observations))

    assert shuffled.content_checksum == first.content_checksum
    assert shuffled.observations == first.observations
    assert shuffled.runtime_summary["failure_count"] == 1
    assert shuffled.runtime_summary["method_observation_count"] == 2 * len(shuffled.observations)
    component = cast(
        "dict[str, object]",
        cast("dict[str, object]", shuffled.nuisance_by_contrast[failed_key[1]])[failed_key[3]],
    )
    assert component["failure_observation_count"] == 2 * cast("int", component["observation_count"])
    assert component["failure_rate"] == pytest.approx(1 / cast("int", component["failure_observation_count"]))
    assert PilotNuisanceSummary.from_json(first.to_json()) == first


def test_pilot_summary_rejects_duplicate_and_incomplete_cluster_grids() -> None:
    """Duplicate cells and missing family/stratum evidence are fatal."""
    observations = _pilot_observations()
    with pytest.raises(ValueError, match="must not duplicate"):
        _build_summary((*observations, observations[0]))

    missing_stratum = tuple(
        item for item in observations if not (item.family_id == "tfim_ground_state" and item.stratum_id == "critical")
    )
    with pytest.raises(ValueError, match=r"exact manifest target|every primary family and stratum"):
        _build_summary(missing_stratum)


def test_pilot_summary_rejects_resealed_caller_supplied_statistics(
    pilot_summary: PilotNuisanceSummary,
) -> None:
    """A self-consistent seal cannot substitute caller-authored nuisance values."""
    document = pilot_summary.to_dict()
    nuisance = cast("dict[str, object]", document["nuisance_by_contrast"])
    assert isinstance(nuisance, dict)
    contrast = cast("dict[str, object]", nuisance[FROZEN_CONTRAST_IDS[0]])
    assert isinstance(contrast, dict)
    family = cast("dict[str, object]", contrast[PRIMARY_TARGET_FAMILIES[0]])
    assert isinstance(family, dict)
    family["target_cluster_variance"] = 0.99
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })

    with pytest.raises(ValueError, match="not derived"):
        PilotNuisanceSummary.from_dict(document)


def test_cluster_aware_design_meets_every_frozen_bound(
    preregistration: InitialPreregistration,
    pilot_summary: PilotNuisanceSummary,
) -> None:
    """The selected design is balanced, bounded, powered, and fixed-sample."""
    design = build_cluster_aware_paired_difference_v1(preregistration, pilot_summary)
    policy = preregistration.sample_size_policy

    assert SampleSizeDesign.from_json(design.to_json()) == design
    assert set(design.target_count_by_family) == set(PRIMARY_TARGET_FAMILIES)
    assert len(set(design.target_count_by_family.values())) == 1
    targets = cast("int", next(iter(design.target_count_by_family.values())))
    assert (
        cast("int", policy["minimum_targets_per_family"])
        <= targets
        <= cast("int", policy["maximum_targets_per_family"])
    )
    assert targets % cast("int", policy["target_count_increment"]) == 0
    assert design.optimization_seed_count in cast("tuple[int, ...]", policy["allowed_optimization_seed_counts"])
    assert design.fixed_test_trajectory_count & (design.fixed_test_trajectory_count - 1) == 0
    assert (
        cast("int", policy["trajectory_count_min"])
        <= design.fixed_test_trajectory_count
        <= cast("int", policy["trajectory_count_max"])
    )
    assert all(
        cast("float", power) >= cast("float", policy["power"]) for power in design.achieved_power_by_contrast.values()
    )
    assert design.expected_primary_mean_half_width <= cast("float", policy["target_mean_half_width"])
    assert design.expected_overall_failure_rate_half_width <= cast("float", policy["failure_rate_half_width"])
    assert design.expected_trajectory_mcse <= cast("float", policy["trajectory_mcse_target"])
    for family_id, strata in PRIMARY_FAMILY_STRATA.items():
        family_allocations = [item for item in design.allocations if item.family_id == family_id]
        assert tuple(item.stratum_id for item in family_allocations) == strata
        assert len({item.target_count for item in family_allocations}) == 1


def test_cluster_aware_design_returns_typed_infeasibility(
    preregistration: InitialPreregistration,
) -> None:
    """Variance that exceeds every bounded allocation has a typed outcome."""
    summary = _build_summary(_pilot_observations(difference_scale=0.12))
    with pytest.raises(PilotDesignInfeasibleError) as error:
        build_cluster_aware_paired_difference_v1(preregistration, summary)
    assert error.value.reason_code == "sample_size_bounds_exhausted"


def test_halfway_reestimation_is_parent_linked_nondecreasing_and_single_use(
    preregistration: InitialPreregistration,
    pilot_summary: PilotNuisanceSummary,
) -> None:
    """The one nuisance-only update cannot shrink or alter fixed trajectories."""
    parent = build_cluster_aware_paired_difference_v1(preregistration, pilot_summary)
    updated = reestimate_cluster_aware_paired_difference_v1(preregistration, pilot_summary, parent)

    assert updated.reestimation_kind == "blinded_nuisance_only"
    assert updated.reestimation_parent_checksum == parent.content_checksum
    assert updated.fixed_test_trajectory_count == parent.fixed_test_trajectory_count
    assert updated.optimization_seed_count >= parent.optimization_seed_count
    assert all(
        cast("int", updated.target_count_by_family[family_id]) >= cast("int", parent.target_count_by_family[family_id])
        for family_id in PRIMARY_TARGET_FAMILIES
    )
    with pytest.raises(PilotDesignInfeasibleError) as second_error:
        reestimate_cluster_aware_paired_difference_v1(preregistration, pilot_summary, updated)
    assert second_error.value.reason_code == "reestimation_limit_exceeded"
    with pytest.raises(ValueError, match="halfway"):
        reestimate_cluster_aware_paired_difference_v1(
            preregistration,
            pilot_summary,
            parent,
            information_fraction=0.75,
        )


def test_halfway_reestimation_cannot_optionally_increase_trajectory_count(
    preregistration: InitialPreregistration,
    pilot_summary: PilotNuisanceSummary,
) -> None:
    """Updated Monte Carlo variance cannot trigger outcome-dependent trajectories."""
    parent = build_cluster_aware_paired_difference_v1(preregistration, pilot_summary)
    higher_mc_summary = _build_summary(_pilot_observations(trajectory_mc_variance=0.02))
    with pytest.raises(PilotDesignInfeasibleError) as error:
        reestimate_cluster_aware_paired_difference_v1(preregistration, higher_mc_summary, parent)
    assert error.value.reason_code == "fixed_trajectory_budget_inadequate"
