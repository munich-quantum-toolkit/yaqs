# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared exact-plan pilot custody fixtures for WP22 tests."""

from __future__ import annotations

import math
from dataclasses import replace
from functools import cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast
from unittest.mock import patch

import numpy as np

from benchmarks.state_preparation.phase2.canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
)
from benchmarks.state_preparation.phase2.execution_context import TrainingExecutionContext
from benchmarks.state_preparation.phase2.execution_protocol import PilotDiagnosticPolicy
from benchmarks.state_preparation.phase2.pilot import (
    PilotContrastBinding,
    PilotEvaluationEvidence,
    PilotJobResult,
    PilotNuisanceSummary,
    PilotObservation,
    ProductionPilotCustody,
    ProductionPilotJobRecord,
    build_pilot_nuisance_summary,
)
from benchmarks.state_preparation.phase2.production_executors import (
    PilotDiagnosticEvidence,
    ProductionAttemptStore,
    ProductionNumericalEvidence,
)
from benchmarks.state_preparation.phase2.protocol import (
    PRIMARY_TARGET_FAMILIES,
    InitialPreregistration,
    load_initial_preregistration,
)
from benchmarks.state_preparation.phase2.result_custody import ProductionResultCustody
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
    training_job_attempt_path,
)
from benchmarks.state_preparation.phase2.training_schedules import (
    PILOT_DIAGNOSTIC_SEED_POLICY_ID,
    CheckpointValidationPolicy,
    ExecutionSeedPolicySuite,
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
from benchmarks.state_preparation.phase2.wp20_resources import CircuitResourceMetrics, LogicalEventSignature
from mqt.yaqs.optimization import KrotovFixedMapEnsemble, KrotovNoiseMap

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.state_preparation.phase2.source_lock import ExecutionSourceManifest

_DEVELOPMENT_MASTER = bytes(range(32))
_SECONDARY_MASTER = bytes(reversed(range(32)))


class _BoundedScheduledExecutionSnapshot:
    """Minimal typed schedule snapshot used only by the manifest-backed F seam."""

    schema_version = "yaqs.state_preparation.phase2.wp22f_bounded_schedule_snapshot.v1"

    def __init__(self, program_checksum: str) -> None:
        self.program_checksum = program_checksum
        self.states = ()
        self.multistart_evidence = SimpleNamespace(
            selected_start_index=0,
            selected_update=0,
            selected_parameter_checksum=canonical_checksum({
                "bounded fixture selected parameters": program_checksum,
            }),
            selected_parameter_artifact=SimpleNamespace(parameters=()),
            total_normalized_work=0.0,
        )

    @property
    def content_checksum(self) -> str:
        """Checksum the bounded snapshot's exact scheduled program identity."""
        return canonical_checksum({
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
        })

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json({
            "schema_version": self.schema_version,
            "program_checksum": self.program_checksum,
            "content_checksum": self.content_checksum,
        })

    @classmethod
    def from_json(cls, payload: str) -> _BoundedScheduledExecutionSnapshot:
        """Decode the narrow bounded snapshot used by the test seam.

        Returns:
            The verified bounded snapshot.

        Raises:
            ValueError: If its schema or checksum changed.
        """
        mapping = load_canonical_json_object(payload)
        if mapping.get("schema_version") != cls.schema_version:
            msg = "Bounded schedule snapshot uses the wrong schema."
            raise ValueError(msg)
        snapshot = cls(cast("str", mapping["program_checksum"]))
        if mapping.get("content_checksum") != snapshot.content_checksum:
            msg = "Bounded schedule snapshot checksum changed."
            raise ValueError(msg)
        return snapshot


def _set_frozen_slot(instance: object, name: str, value: object) -> None:
    """Set one slot while constructing a deliberately bounded typed test seam."""
    object.__setattr__(instance, name, value)  # noqa: PLC2801 -- frozen init bypass for test-only seams


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
def pilot_context_with_secondary_master(
    secondary_master: bytes,
) -> tuple[
    InitialPreregistration,
    TargetPopulationManifest,
    TargetPopulationManifest,
    TrainingRunPlan,
    tuple[PilotContrastBinding, ...],
]:
    """Return a two-manifest plan with caller-selected q12 archive entropy.

    Returns:
        The preregistration, q6 manifest, q12 manifest, plan, and bindings.

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
def pilot_context() -> tuple[
    InitialPreregistration,
    TargetPopulationManifest,
    TargetPopulationManifest,
    TrainingRunPlan,
    tuple[PilotContrastBinding, ...],
]:
    """Return the default exact two-manifest pilot custody fixture.

    Returns:
        The preregistration, q6 manifest, q12 manifest, plan, and bindings.
    """
    return pilot_context_with_secondary_master(_SECONDARY_MASTER)


def _symmetric_samples(mean: float, variance: float, *, count: int) -> tuple[float, ...]:
    """Return a symmetric raw sample with the requested mean and variance.

    Raises:
        ValueError: If ``count`` is odd or below two.
    """
    if count < 2 or count % 2:
        msg = "count must be an even integer of at least two."
        raise ValueError(msg)
    radius = math.sqrt(variance * (count - 1) / count)
    return (mean - radius,) * (count // 2) + (mean + radius,) * (count // 2)


def _production_document(document_type: str, payload: dict[str, object]) -> dict[str, object]:
    """Return one WP22E-compatible checksum-sealed typed document."""
    content = {
        "schema_version": "yaqs.state_preparation.phase2.production_document.v1",
        "document_type": document_type,
        "payload": payload,
    }
    return {**content, "content_checksum": canonical_checksum(content)}


def _failure_partial_receipts(normalized_work: float) -> dict[str, object]:
    """Return bounded structured partial-work custody for a failed fixture."""
    return {
        "closed_artifact_count": 0,
        "closed_artifact_bytes": 0,
        "closed_role_counts": {},
        "normalized_work_components": [{"kind": "bounded_fixture", "work": normalized_work}],
        "normalized_work_unavailable": False,
        "normalized_work_is_lower_bound": False,
        "unavailable_partial_work_roles": [],
    }


def production_pilot_job(job: TrainingJob) -> TrainingJob:
    """Bind complete production fingerprints to one exact pilot fixture job.

    Returns:
        A pipeline-backed job suitable for bounded E custody tests.
    """
    return replace(
        job,
        execution_profile_checksum=canonical_checksum({"pilot": "execution profile"}),
        scoped_binding_checksum=canonical_checksum({"pilot": "scoped binding"}),
        executable_binding_checksum=canonical_checksum({"pilot": "executable binding"}),
        evaluation_policy_checksum=canonical_checksum({"pilot": "evaluation policy"}),
        target_configuration_checksum=canonical_checksum({"pilot": "target configuration"}),
        source_fingerprint_checksum=canonical_checksum({"pilot": "source fingerprint"}),
        scheduled_execution_program_checksum=canonical_checksum({"pilot": "scheduled program"}),
    )


def _production_pilot_context(job: TrainingJob, diagnostic_policy: PilotDiagnosticPolicy) -> TrainingExecutionContext:
    """Create the narrow real-type context seam used by one-record tests.

    Returns:
        A typed context exposing only the fields read by production replay.
    """
    context = object.__new__(TrainingExecutionContext)
    _set_frozen_slot(
        context,
        "execution_source_manifest",
        SimpleNamespace(content_checksum=canonical_checksum({"pilot": "execution source manifest"})),
    )
    _set_frozen_slot(context, "plan", SimpleNamespace(preset="paper-pilot", jobs=(job,)))
    _set_frozen_slot(
        context,
        "scoped_bindings",
        (
            SimpleNamespace(
                content_checksum=job.executable_binding_checksum,
                binding=SimpleNamespace(pilot_diagnostic_policy=diagnostic_policy),
            ),
        ),
    )
    return context


def production_pilot_record(
    source_job: TrainingJob,
    job_directory: Path,
    *,
    status: Literal["success", "failure"],
    diagnostic_seed_offset: int = 0,
) -> ProductionPilotJobRecord:
    """Publish and reopen one representative q6/q12 WP22E pilot attempt.

    Returns:
        A production pilot record initialized through immutable E manifests.

    Raises:
        ValueError: If status is not ``success`` or ``failure``.
    """
    if status not in {"success", "failure"}:
        msg = "status must be success or failure."
        raise ValueError(msg)
    job = production_pilot_job(source_job)
    diagnostic_enabled = job.qubit_count == 6
    diagnostic_policy = (
        PilotDiagnosticPolicy.primary_q6() if diagnostic_enabled else PilotDiagnosticPolicy.secondary_q12()
    )
    policy_checksum = diagnostic_policy.content_checksum
    context = _production_pilot_context(job, diagnostic_policy)
    store = ProductionAttemptStore(job_directory, job.content_checksum, 1)
    source_manifest_checksum = context.execution_source_manifest.content_checksum
    circuit_checksum = canonical_checksum({"diagnostic": "circuit", "job": job.content_checksum})
    trajectory_count = 1024 if job.qubit_count == 6 else 256
    resource_normalized_work = (
        float(trajectory_count + (diagnostic_policy.trajectory_count if diagnostic_enabled else 0))
        if status == "success"
        else 2.0
    )
    compiled_resources = CircuitResourceMetrics(
        qubit_count=job.qubit_count,
        trainable_parameter_count=0,
        logical_events=(
            LogicalEventSignature(
                ordinal=0,
                logical_gate_id=0,
                name="rx",
                sites=(0,),
                parameter_index=None,
                angle_scale=1.0,
                angle_offset=0.0,
                fixed_parameters=(0.0,),
                noise_enabled=True,
            ),
        ),
        native_events=(),
    )
    resource_ref = store.write_json_blob(
        "runtime/resources.json",
        _production_document(
            "runtime_resources",
            {
                "job_checksum": job.content_checksum,
                "source_fingerprint_checksum": job.source_fingerprint_checksum,
                "wall_time_seconds": 0.25,
                "peak_memory_bytes": 1024,
                "normalized_work": resource_normalized_work,
                "failure_phase": None if status == "success" else "operator_growth_execution",
                "partial_receipts": (
                    None if status == "success" else _failure_partial_receipts(resource_normalized_work)
                ),
                "circuit": {
                    "circuit_binding_checksum": circuit_checksum,
                    "topology_id": "wp22_fixture_topology",
                    "qubit_count": compiled_resources.qubit_count,
                    "parameter_count": compiled_resources.trainable_parameter_count,
                    "logical_gate_count": len(compiled_resources.logical_events),
                    "logical_two_qubit_gate_count": compiled_resources.logical_two_qubit_gates,
                    "noisy_gate_indices": [],
                    "compiled_resources": compiled_resources.to_dict(),
                    "compiled_resources_checksum": compiled_resources.content_checksum,
                    "native_two_qubit_gates_per_chain_edge": list(
                        compiled_resources.native_two_qubit_gates_per_chain_edge,
                    ),
                },
            },
        ),
        role="runtime_resources",
    )
    blobs = [resource_ref]
    raw_ref = None
    schedule_snapshot_ref = None
    diagnostic_refs = ()
    derived_metrics: dict[str, object] = {
        "execution_preset": "paper-pilot",
        "promotion_eligible": False,
        "pilot_diagnostic_required": diagnostic_enabled,
    }
    if status == "success":
        snapshot = _BoundedScheduledExecutionSnapshot(
            cast("str", job.scheduled_execution_program_checksum),
        )
        schedule_snapshot_ref = store.write_blob(
            "schedule/snapshot.json",
            f"{snapshot.to_json()}\n".encode(),
            role="schedule_snapshot",
            logical_checksum=snapshot.content_checksum,
        )
        blobs.append(schedule_snapshot_ref)
        values = tuple(0.75 for _ in range(trajectory_count))
        evaluation_seed_domain = "pilot_evaluation"
        evaluation_configuration_checksum = canonical_checksum({
            "job_checksum": job.content_checksum,
            "evaluation_policy_checksum": job.evaluation_policy_checksum,
            "circuit_checksum": circuit_checksum,
            "parameter_checksum": canonical_checksum({"parameters": []}),
        })
        provider_checksum = canonical_checksum({"bounded fixture provider": job.content_checksum})
        fresh_ensemble = KrotovFixedMapEnsemble(
            role="pilot_evaluation",
            resolved_seed=job.evaluation_seed,
            stage_index=0,
            stage_id="fresh_evaluation",
            stage_configuration_checksum=evaluation_configuration_checksum,
            circuit_checksum=circuit_checksum,
            provider_checksum=provider_checksum,
            ensemble_index=0,
            refresh_index=0,
            global_iteration_start=0,
            trajectory_maps=tuple(
                (KrotovNoiseMap(source_gate_index=0, is_identity=True),) for _ in range(trajectory_count)
            ),
        )
        fresh_ensemble_ref = store.write_blob(
            "evaluation/fresh_fixed_map_ensemble.json",
            f"{fresh_ensemble.to_json()}\n".encode(),
            role="fixed_map_ensemble",
            logical_checksum=fresh_ensemble.content_checksum,
        )
        blobs.append(fresh_ensemble_ref)
        raw_ref = store.write_json_blob(
            "evaluation/raw_trajectory_fidelities.json",
            _production_document(
                "raw_trajectory_fidelities",
                {
                    "job_checksum": job.content_checksum,
                    "evaluation_policy_checksum": job.evaluation_policy_checksum,
                    "evaluation_configuration_checksum": evaluation_configuration_checksum,
                    "data_role": job.data_role,
                    "seed_domain": evaluation_seed_domain,
                    "evaluation_seed": job.evaluation_seed,
                    "trajectory_count": trajectory_count,
                    "trajectory_fidelities": list(values),
                    "fixed_map_ensemble_checksum": fresh_ensemble.content_checksum,
                    "sampled_nonidentity_events": fresh_ensemble.nonidentity_event_count,
                },
            ),
            role="raw_trajectory_sidecar",
        )
        blobs.append(raw_ref)
        derived_metrics.update({
            "noisy_fidelity": float(np.mean(np.asarray(values, dtype=np.float64))),
            "trajectory_count": trajectory_count,
            "evaluation_configuration_checksum": evaluation_configuration_checksum,
            "evaluation_seed": job.evaluation_seed,
            "evaluation_data_role": job.data_role,
            "evaluation_seed_domain": evaluation_seed_domain,
            "reporting_prefixes": [trajectory_count],
            "prefix_mean_fidelities": {str(trajectory_count): 0.75},
            "fresh_ensemble_checksum": fresh_ensemble.content_checksum,
            "provider_checksum": provider_checksum,
            "sampled_nonidentity_events": fresh_ensemble.nonidentity_event_count,
            "total_normalized_training_work": 0.0,
            "selected_start_index": snapshot.multistart_evidence.selected_start_index,
            "selected_update": snapshot.multistart_evidence.selected_update,
            "selected_parameter_checksum": snapshot.multistart_evidence.selected_parameter_checksum,
        })
        if diagnostic_enabled:
            map_refs = []
            member_seeds = tuple(
                ExecutionSeedPolicySuite.frozen().derive(
                    PILOT_DIAGNOSTIC_SEED_POLICY_ID,
                    {
                        "target_manifest_checksum": job.target_manifest_checksum,
                        "target_instance_spec_checksum": job.target_spec_checksum,
                        "optimization_seed": job.optimization_seed,
                        "publication_candidate_checksum": job.candidate_configuration_checksum,
                        "repetition": repetition,
                    },
                )
                for repetition in range(diagnostic_policy.trajectory_count)
            )
            if diagnostic_seed_offset:
                member_seeds = (member_seeds[0] + diagnostic_seed_offset, *member_seeds[1:])
            provider_identity = cast("dict[str, object]", diagnostic_policy.provider_identity)
            provider_checksum = cast("str", provider_identity["content_checksum"])
            for index, seed in enumerate(member_seeds):
                ensemble = KrotovFixedMapEnsemble(
                    role="pilot_evaluation",
                    resolved_seed=seed,
                    stage_index=0,
                    stage_id="post_training_diagnostic",
                    stage_configuration_checksum=canonical_checksum({"diagnostic": "stage"}),
                    circuit_checksum=circuit_checksum,
                    provider_checksum=provider_checksum,
                    ensemble_index=index,
                    refresh_index=0,
                    global_iteration_start=0,
                    trajectory_maps=((KrotovNoiseMap(source_gate_index=0, is_identity=True),),),
                )
                map_ref = store.write_blob(
                    f"diagnostics/maps/ensemble_{index:05d}.json",
                    f"{ensemble.to_json()}\n".encode(),
                    role="fixed_map_ensemble",
                    logical_checksum=ensemble.content_checksum,
                )
                map_refs.append(map_ref)
                blobs.append(map_ref)
            selected_checksum = snapshot.multistart_evidence.selected_parameter_checksum
            estimator_checksum = canonical_checksum({
                "endpoint": diagnostic_policy.endpoint,
                "checkpoint_rule": diagnostic_policy.checkpoint_rule,
                "estimator_id": diagnostic_policy.estimator_id,
                "estimator_version": diagnostic_policy.estimator_version,
                "parameter_ordering": diagnostic_policy.parameter_ordering,
                "coordinate_variance_rule": diagnostic_policy.coordinate_variance_rule,
                "summary_statistics": list(diagnostic_policy.summary_statistics),
                "provider_checksum": provider_checksum,
            })
            diagnostic = PilotDiagnosticEvidence(
                job_checksum=job.content_checksum,
                policy_checksum=policy_checksum,
                checkpoint_parameter_checksum=selected_checksum,
                parameter_vector_checksum=selected_checksum,
                circuit_checksum=circuit_checksum,
                provider_checksum=provider_checksum,
                estimator_checksum=estimator_checksum,
                member_seeds=member_seeds,
                ensemble_refs=tuple(map_refs),
                pathwise_update_vectors=tuple((index / 100.0,) for index in range(32)),
            )
            diagnostic_ref = store.write_json_blob(
                "diagnostics/pathwise_update_vectors.json",
                diagnostic.to_dict(),
                role="pilot_diagnostic_sidecar",
            )
            blobs.append(diagnostic_ref)
            diagnostic_refs = (diagnostic_ref,)
    evidence = ProductionNumericalEvidence(
        job_checksum=job.content_checksum,
        attempt=1,
        artifact_kind="pipeline",
        status=status,
        execution_source_manifest_checksum=source_manifest_checksum,
        source_fingerprint_checksum=cast("str", job.source_fingerprint_checksum),
        executable_binding_checksum=cast("str", job.executable_binding_checksum),
        scheduled_program_checksum=cast("str", job.scheduled_execution_program_checksum),
        target_identity={
            "target_instance_id": job.target_instance_id,
            "target_instance_spec_checksum": job.target_spec_checksum,
            "population_config_checksum": job.target_configuration_checksum,
            "target_manifest_checksum": job.target_manifest_checksum,
            "family_id": job.family_id,
            "stratum_id": job.stratum_id,
            "qubit_count": job.qubit_count,
            "parameter_checksum": canonical_checksum({"bounded fixture parameters": job.target_instance_id}),
            "vector_checksum": canonical_checksum({"bounded fixture target": job.target_instance_id}),
            "norm": 1.0,
        },
        evaluation_policy_checksum=cast("str", job.evaluation_policy_checksum),
        structural_prefix_checksums=(canonical_checksum({"pilot prefix": job.content_checksum}),),
        schedule_snapshot_ref=schedule_snapshot_ref,
        map_evidence_refs=(),
        diagnostic_refs=diagnostic_refs,
        raw_trajectory_ref=raw_ref,
        resource_ref=resource_ref,
        derived_metrics=derived_metrics,
        failure=(
            None
            if status == "success"
            else {
                "phase": "operator_growth_execution",
                "exception_type": "SyntheticFixtureFailure",
                "message": "bounded manifest-backed pilot failure",
            }
        ),
    )
    evidence_ref = store.write_json_blob(
        "production_evidence.json",
        evidence.to_dict(),
        role="production_evidence",
    )
    blobs.append(evidence_ref)
    reference = store.publish(
        artifact_kind="pipeline",
        status=evidence.status,
        execution_source_manifest_checksum=source_manifest_checksum,
        source_fingerprint_checksum=cast("str", job.source_fingerprint_checksum),
        blobs=blobs,
        evidence_ref=evidence_ref,
    )
    outcome = TrainingJobOutcome(
        job_checksum=job.content_checksum,
        status=evidence.status,
        result_artifact_checksum=(reference.content_checksum if status == "success" else None),
        exception_type=(None if status == "success" else "executor_failure"),
        message=(None if status == "success" else "bounded manifest-backed pilot failure"),
        attempt=1,
    )
    outcome_path = training_job_attempt_path(job_directory, 1)
    outcome_path.parent.mkdir(parents=True, exist_ok=True)
    outcome_path.write_text(f"{canonical_json(outcome.to_dict())}\n", encoding="utf-8")
    with patch(
        "benchmarks.state_preparation.phase2.production_executors.ScheduledExecutionSnapshot",
        _BoundedScheduledExecutionSnapshot,
    ):
        return ProductionPilotJobRecord(context, job, job_directory)


def pilot_job_evidence(
    job: TrainingJob,
    *,
    fidelity: float,
    failed: bool,
    gradient_variance: float,
    trajectory_mc_variance: float,
    wall_time_seconds: float,
    tracemalloc_peak_bytes: int,
) -> tuple[TrainingJobOutcome, PilotJobResult]:
    """Return linked typed outcome/result evidence for one synthetic job.

    Returns:
        The authoritative first-attempt outcome and typed pilot result.
    """
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
def pilot_observations(
    *,
    difference_scale: float = 0.0001,
    trajectory_mc_variance: float = 0.0025,
    failed_cell: tuple[str, str, int, str] | None = None,
) -> tuple[PilotObservation, ...]:
    """Return the exact q6 inferential observations from the two-manifest plan.

    Returns:
        Complete plan-linked paired pilot evidence.
    """
    _preregistration, target_manifest, _supplemental, pilot_plan, bindings = pilot_context()
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
                evidence_by_job[job.content_checksum] = pilot_job_evidence(
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


def production_pilot_custody_fixture(
    output_root: Path,
    *,
    execution_source_manifest: ExecutionSourceManifest | None = None,
    secondary_archive_marker: str = "secondary-archive",
) -> ProductionPilotCustody:
    """Return a full typed aggregate replay backed by exact q6 evidence.

    Representative :class:`ProductionPilotJobRecord` tests reopen actual WP22E
    manifests.  This bounded aggregate seam substitutes only the expensive
    1,080 record-opening loop while exercising the real custody projection and
    exact q6 nuisance reconstruction.

    Returns:
        A typed 720-q6 plus 360-q12 production pilot custody fixture.
    """
    preregistration, primary, supplemental, plan, _bindings = pilot_context()
    primary_evidence: dict[str, tuple[TrainingJobOutcome, PilotJobResult]] = {}
    for observation in pilot_observations():
        primary_evidence[observation.treatment_job.content_checksum] = (
            observation.treatment_outcome,
            observation.treatment_result,
        )
        primary_evidence[observation.comparator_job.content_checksum] = (
            observation.comparator_outcome,
            observation.comparator_result,
        )
    context = object.__new__(TrainingExecutionContext)
    source = (
        SimpleNamespace(content_checksum=canonical_checksum({"pilot": "bounded execution source"}))
        if execution_source_manifest is None
        else execution_source_manifest
    )
    _set_frozen_slot(context, "plan", plan)
    _set_frozen_slot(context, "preregistration", preregistration)
    _set_frozen_slot(context, "target_manifests", (primary, supplemental))
    _set_frozen_slot(context, "execution_source_manifest", source)

    def fake_record(
        _context: TrainingExecutionContext,
        job: TrainingJob,
        _job_directory: Path,
    ) -> SimpleNamespace:
        marker = secondary_archive_marker if job.qubit_count == 12 else "primary"
        reference_checksum = canonical_checksum({
            "marker": marker,
            "job_checksum": job.content_checksum,
        })
        if job.qubit_count == 6:
            outcome, result = primary_evidence[job.content_checksum]
        else:
            outcome = TrainingJobOutcome(
                job_checksum=job.content_checksum,
                status="success",
                result_artifact_checksum=reference_checksum,
                exception_type=None,
                message=None,
                attempt=1,
            )
            result = None
        reference_checksum = (
            outcome.result_artifact_checksum
            if outcome.status == "success" and outcome.result_artifact_checksum is not None
            else reference_checksum
        )
        fidelity_values = (
            (
                result.evaluation_evidence.fresh_test_trajectory_fidelities
                if result is not None and result.evaluation_evidence is not None
                else (0.75,) * 256
            )
            if outcome.status == "success"
            else None
        )
        custody = object.__new__(ProductionResultCustody)
        reference = SimpleNamespace(
            content_checksum=reference_checksum,
            attempt=1,
            job_checksum=job.content_checksum,
            status=outcome.status,
            execution_source_manifest_checksum=source.content_checksum,
        )
        production_evidence = SimpleNamespace(
            artifact_kind=("operator_growth" if job.implementation_kind == "operator_growth" else "pipeline"),
            execution_source_manifest_checksum=source.content_checksum,
            structural_prefix_checksums=(canonical_checksum({"prefix": job.content_checksum}),),
            derived_metrics={"execution_preset": "paper-pilot"},
            source_fingerprint_checksum=job.source_fingerprint_checksum,
            executable_binding_checksum=job.executable_binding_checksum,
            scheduled_program_checksum=job.scheduled_execution_program_checksum,
            evaluation_policy_checksum=job.evaluation_policy_checksum,
            target_identity={
                "target_instance_id": job.target_instance_id,
                "target_instance_spec_checksum": job.target_spec_checksum,
                "population_config_checksum": job.target_configuration_checksum,
                "target_manifest_checksum": job.target_manifest_checksum,
                "family_id": job.family_id,
                "stratum_id": job.stratum_id,
                "qubit_count": job.qubit_count,
            },
        )
        compiled_resources = CircuitResourceMetrics(
            qubit_count=job.qubit_count,
            trainable_parameter_count=0,
            logical_events=(),
            native_events=(),
        )
        raw_payload = (
            None
            if fidelity_values is None
            else {
                "data_role": job.data_role,
                "evaluation_seed": job.evaluation_seed,
                "trajectory_count": len(fidelity_values),
                "trajectory_fidelities": fidelity_values,
            }
        )
        _set_frozen_slot(custody, "reference", reference)
        _set_frozen_slot(custody, "production_evidence", production_evidence)
        _set_frozen_slot(custody, "result_evidence_checksum", canonical_checksum({"evidence": reference_checksum}))
        _set_frozen_slot(custody, "raw_trajectory_payload", raw_payload)
        _set_frozen_slot(
            custody,
            "raw_trajectory_document_checksum",
            None if raw_payload is None else canonical_checksum({"raw": reference_checksum}),
        )
        _set_frozen_slot(
            custody,
            "resource_payload",
            {
                "normalized_work": 1_000.0 if job.qubit_count == 6 else 2_000.0,
                "circuit": {
                    "circuit_binding_checksum": canonical_checksum({"circuit": job.content_checksum}),
                    "compiled_resources": compiled_resources.to_dict(),
                    "compiled_resources_checksum": compiled_resources.content_checksum,
                    "native_two_qubit_gates_per_chain_edge": (compiled_resources.native_two_qubit_gates_per_chain_edge),
                },
            },
        )
        _set_frozen_slot(custody, "resource_document_checksum", canonical_checksum({"resource": reference_checksum}))
        _set_frozen_slot(custody, "pilot_diagnostics", ())
        _set_frozen_slot(
            custody,
            "schema_version",
            "yaqs.state_preparation.phase2.production_result_custody.v1",
        )
        return SimpleNamespace(
            job=job,
            outcome=outcome,
            pilot_result=result,
            result_custody=custody,
            content_checksum=canonical_checksum({
                "record": job.content_checksum,
                "marker": marker,
            }),
        )

    with patch(
        "benchmarks.state_preparation.phase2.pilot.ProductionPilotJobRecord",
        side_effect=fake_record,
    ):
        return ProductionPilotCustody(context, output_root)


def build_pilot_summary(
    observations: tuple[PilotObservation, ...],
    *,
    summary_id: str = "phase2_pilot_nuisance_v1",
) -> PilotNuisanceSummary:
    """Bind observations to the exact cached pilot custody artifacts.

    Returns:
        The complete checksum-sealed nuisance summary.
    """
    preregistration, target_manifest, supplemental, pilot_plan, bindings = pilot_context()
    return build_pilot_nuisance_summary(
        preregistration,
        target_manifest,
        supplemental,
        pilot_plan,
        bindings,
        observations,
        summary_id=summary_id,
    )


__all__ = [
    "build_pilot_summary",
    "pilot_context",
    "pilot_context_with_secondary_master",
    "pilot_job_evidence",
    "pilot_observations",
    "production_pilot_custody_fixture",
    "production_pilot_job",
    "production_pilot_record",
]
