# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end acceptance tests for the WP18 Phase II artifact engine."""

from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from filelock import FileLock

import benchmarks.state_preparation.phase2.artifacts as artifact_module
from benchmarks.state_preparation.phase2.artifacts import (
    Phase2ArtifactStore,
    Phase2ArtifactVerificationError,
    Phase2ConcurrentMutationError,
    Phase2ResumeMismatchError,
    StageExecutionEvidence,
)
from benchmarks.state_preparation.phase2.canonical import (
    canonical_checksum,
    load_canonical_json_object,
)
from benchmarks.state_preparation.phase2.evaluator import (
    MaterializedCircuitPayload,
    ParallelPhase2Evaluator,
)
from benchmarks.state_preparation.phase2.execution import (
    Phase2PipelineExecutor,
    PipelineExecutionFailure,
)
from benchmarks.state_preparation.phase2.noisy_krotov import (
    NoisyKrotovCircuitBinding,
    NoisyKrotovObjectiveBinding,
    NoisyKrotovStageExecution,
    execute_fixed_rate_krotov_stage,
)
from benchmarks.state_preparation.phase2.pipeline import (
    ExternalCheckpointRef,
    PipelineBenchmarkFailure,
    PipelineBenchmarkResult,
    PipelineEvaluationConfig,
    TrainingPipelineResult,
    TrainingPipelineTemplate,
    TrainingStageTemplate,
    pipeline_benchmark_record_from_csv_row,
    pipeline_benchmark_record_from_json,
)
from benchmarks.state_preparation.phase2.protocol import (
    TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
    load_initial_preregistration,
)
from benchmarks.state_preparation.phase2.resumability import (
    ExecutionSourceEntry,
    NonScientificResumeOverride,
    ResumabilityFingerprint,
)
from benchmarks.state_preparation.phase2.targets import (
    authorize_target_materialization,
    build_target_population_config,
    materialize_target_population,
    role_master_entropy_commitment,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovFixedMapEnsemble,
    KrotovNoiseMap,
    ParameterizedCircuit,
    ParameterizedGate,
)
from tests.benchmarks.test_state_preparation_phase2_pipeline import (
    _checksum,
    _materialization_policy,
    _pipeline,
    _screening_target_manifest,
    _seed_domains,
    _stage_template,
    _template,
    _work,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from benchmarks.state_preparation.phase2.evaluator import (
        PipelineEvaluationMeasurement,
    )
    from benchmarks.state_preparation.phase2.pipeline import (
        TrainingPipelineConfig,
        TrainingStageConfig,
    )
    from benchmarks.state_preparation.phase2.targets import (
        MaterializedTarget,
    )
    from mqt.yaqs.optimization import (
        KrotovMapRole,
    )


def _pipeline_with_every_stage_kind() -> TrainingPipelineConfig:
    """Return a valid optimize/grow/optimize/prune pipeline.

    Returns:
        A resolved screening pipeline containing every Phase II stage kind.
    """
    base = _template(
        method_id="impact_pruning_crn",
        template_id="wp18_resume_all_stage_kinds",
    )
    generic_stages = []
    for stage_template in base.stages:
        policy = dict(stage_template.stage_policy)
        if policy["optimizer_id"] == "krotov":
            policy["optimizer_id"] = "wp18_test_optimizer"
        generic_stages.append(replace(stage_template, stage_policy=policy))
    base = replace(base, stages=tuple(generic_stages))
    pruning = _stage_template(
        index=3,
        stage_id="prune_d2",
        kind="prune",
        input_topology="bmpd_d2",
        output_topology="bmpd_d2_pruned",
        input_parameters=108,
        output_parameters=90,
        transfer="apply_pruning_mask",
        iterations=0,
        pruning_rule="random",
        pruning_threshold=0.1,
    )
    return _pipeline(replace(base, stages=(*base.stages, pruning)))


def _fingerprint(pipeline: TrainingPipelineConfig) -> ResumabilityFingerprint:
    """Build a compact complete resumability fingerprint for a test pipeline.

    Returns:
        A fingerprint covering source, lockfile, and sealed-input roles.
    """
    entries = tuple(
        ExecutionSourceEntry(
            role=role,
            repository_path=path,
            starting_git_blob_id=character * 40,
            content_checksum=_checksum(character),
        )
        for role, path, character in (
            ("execution_source", "src/runner.py", "a"),
            ("lockfile", "uv.lock", "b"),
            ("sealed_input", "protocol.json", "c"),
        )
    )
    return ResumabilityFingerprint(
        starting_commit="d" * 40,
        pipeline_prefix_id=pipeline.prefix_id(len(pipeline.stages) - 1),
        dependency_versions={"numpy": "2.4.6", "yaqs": "0.5.0"},
        entries=entries,
    )


@lru_cache(maxsize=1)
def _screening_materialized_targets() -> tuple[MaterializedTarget, ...]:
    """Materialize the genuine typed targets used by objective-publication tests.

    Returns:
        The authorized screening target vectors in manifest order.
    """
    master_entropy = bytes(reversed(range(32)))
    preregistration = load_initial_preregistration()
    config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(master_entropy),
        population_scope="primary_q6",
    )
    manifest = _screening_target_manifest()
    authorization = authorize_target_materialization(
        preregistration,
        config,
        manifest,
        master_entropy,
    )
    return materialize_target_population(
        config,
        preregistration,
        manifest,
        master_entropy,
        authorization,
    ).targets


def _objective_pipeline() -> TrainingPipelineConfig:
    """Resolve a one-step pipeline for focused genuine-WP17 publication tests.

    Returns:
        A typed pipeline bound to the first materialized screening target.
    """
    stage = _stage_template(
        index=0,
        stage_id="objective_stage",
        kind="optimize",
        input_topology=None,
        output_topology="wp18_objective_q6",
        input_parameters=0,
        output_parameters=1,
        transfer="initialize_random_normal",
        iterations=1,
    )
    template = TrainingPipelineTemplate(
        template_id="wp18_objective_binding",
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id="primary_q6",
        ansatz_family="bmpd_brickwall",
        method_id="fixed_depth_bmpd_crn",
        method_version="1",
        resource_stratum_id="primary_cap_12",
        stages=(stage,),
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )
    manifest = _screening_target_manifest()
    target = _screening_materialized_targets()[0]
    spec = next(item for item in manifest.instances if item.target_instance_id == target.target_instance_id)
    return template.resolve(
        target_namespace="phase2",
        target_manifest=manifest,
        target_instance_id=spec.target_instance_id,
        target_population_manifest_checksum=manifest.content_checksum,
        target_instance_spec_checksum=spec.content_checksum,
        target_family_id=spec.family_id,
        target_stratum_id=spec.stratum_id,
        qubit_count=spec.qubit_count,
        optimization_block_id="wp18_objective_binding",
        optimization_seed=123,
        data_role="screening_selection",
    )


def _objective_evidence(
    pipeline: TrainingPipelineConfig,
    target: MaterializedTarget,
    *,
    initial_state: MPS | None = None,
) -> StageExecutionEvidence:
    """Execute and translate one genuine WP17 objective into stage evidence.

    Returns:
        Complete genuine-WP17 persistence evidence.
    """
    stage = pipeline.stages[0]
    circuit = ParameterizedCircuit(
        pipeline.qubit_count,
        [ParameterizedGate("ry", (0,), param_index=0, logical_gate_id="ry_0")],
        num_params=1,
    )
    binding = NoisyKrotovCircuitBinding(circuit, stage.output_topology_id)
    execution = execute_fixed_rate_krotov_stage(
        stage,
        binding,
        target,
        np.array([0.1], dtype=np.float64),
        initial_state=initial_state,
    )
    assert isinstance(execution, NoisyKrotovStageExecution), getattr(execution, "message", "")
    return StageExecutionEvidence.from_noisy_krotov(
        stage,
        execution,
        source_parameters=None,
        circuit_statistics={
            "topology_id": stage.output_topology_id,
            "parameter_count": stage.output_parameter_count,
        },
    )


def _ensemble(
    stage: TrainingStageConfig,
    *,
    role: KrotovMapRole,
    seed: int,
    trajectory_count: int,
) -> KrotovFixedMapEnsemble:
    """Create one exact role-bound identity-map ensemble.

    Returns:
        A deterministic fixed-map ensemble for ``stage``.
    """
    return KrotovFixedMapEnsemble(
        role=role,
        resolved_seed=seed,
        stage_index=stage.stage_index,
        stage_id=stage.stage_id,
        stage_configuration_checksum=stage.configuration_checksum,
        circuit_checksum=cast("str", _circuit_topology(stage)["content_checksum"]),
        provider_checksum=_checksum("f"),
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
        trajectory_maps=[[KrotovNoiseMap(source_gate_index=0, is_identity=True)] for _ in range(trajectory_count)],
    )


def _circuit_topology(stage: TrainingStageConfig) -> dict[str, object]:
    """Return a complete checksum-sealed synthetic circuit topology."""
    payload: dict[str, object] = {
        "schema_version": "test.phase2.circuit_binding.v1",
        "topology_id": stage.output_topology_id,
        "num_qubits": 6,
        "num_params": stage.output_parameter_count,
        "gates": [
            {
                "name": "ry",
                "sites": [0],
                "param_index": 0,
            }
        ],
    }
    return {**payload, "content_checksum": canonical_checksum(payload)}


def _stage_evidence(
    stage: TrainingStageConfig,
    predecessor: NDArray[np.float64] | None,
) -> StageExecutionEvidence:
    """Create deterministic complete evidence without running an optimizer.

    Returns:
        Evidence satisfying the exact configured stage and fixed-map policy.
    """
    output = np.linspace(
        0.01 * (stage.stage_index + 1),
        0.01 * (stage.stage_index + 2),
        stage.output_parameter_count,
        dtype=np.float64,
    )
    circuit_topology = _circuit_topology(stage)
    circuit_binding_checksum = cast("str", circuit_topology["content_checksum"])
    if stage.stage_kind == "prune":
        assert predecessor is not None
        return StageExecutionEvidence.for_parameter_transform(
            stage,
            initial_parameters=predecessor,
            output_parameters=output,
            circuit_binding_checksum=circuit_binding_checksum,
            circuit_topology=circuit_topology,
            circuit_statistics={
                "topology_id": stage.output_topology_id,
                "parameter_count": stage.output_parameter_count,
            },
            summary={"event": "deterministic_pruning", "retained_parameters": output.size},
        )

    if predecessor is None:
        initial = output.copy()
    elif stage.stage_kind == "grow":
        initial = np.concatenate((predecessor, output[stage.input_parameter_count :]))
    else:
        initial = predecessor.copy()

    training_maps: tuple[KrotovFixedMapEnsemble, ...] = ()
    validation_maps: tuple[KrotovFixedMapEnsemble, ...] = ()
    if stage.sampling_policy == "crn_fixed":
        assert stage.training_seed is not None
        training_maps = (
            _ensemble(
                stage,
                role="training_trajectory",
                seed=stage.training_seed,
                trajectory_count=stage.trajectory_count,
            ),
        )
    if stage.checkpoint_validation.enabled:
        assert stage.checkpoint_validation.seed is not None
        validation_maps = (
            _ensemble(
                stage,
                role="checkpoint_validation",
                seed=stage.checkpoint_validation.seed,
                trajectory_count=stage.checkpoint_validation.trajectory_count,
            ),
        )
    selected_iteration = stage.iteration_budget
    selected = output.copy()
    selected_fidelity: float | None = None
    validation_summary: Mapping[str, object] | None = None
    if stage.checkpoint_validation.enabled:
        cadence = stage.checkpoint_validation.cadence
        assert cadence is not None
        selected_iteration -= cadence
        selected = output + 0.001
        selected_fidelity = 0.91
        validation_summary = {
            "evaluation_count": 2,
            "selected_iteration": selected_iteration,
            "selected_fidelity": selected_fidelity,
        }
    training_trajectories = stage.iteration_budget * stage.trajectory_count
    validation_trajectories = sum(item.trajectory_count for item in validation_maps)
    return StageExecutionEvidence(
        stage=stage,
        source_parameters=predecessor,
        initial_parameters=initial,
        final_parameters=output,
        selected_parameters=selected,
        selected_global_iteration=selected_iteration,
        completed_global_iteration=stage.iteration_budget,
        selected_checkpoint_validation_fidelity=selected_fidelity,
        circuit_binding_checksum=circuit_binding_checksum,
        provider_checksum=(_checksum("f") if stage.training_noise_id != "noiseless" else None),
        objective_checksum=_checksum("1"),
        trace=(
            {
                "global_iteration": stage.iteration_budget,
                "selected": selected_iteration,
            },
        ),
        training_ensembles=training_maps,
        checkpoint_validation_ensembles=validation_maps,
        normalized_work=_work(
            objective=stage.iteration_budget,
            gradient=stage.iteration_budget,
            training=training_trajectories,
            validation=validation_trajectories,
            gates=training_trajectories,
        ),
        training_summary={
            "completed_iterations": stage.iteration_budget,
            "final_objective": 0.1 + 0.01 * stage.stage_index,
        },
        checkpoint_validation_summary=validation_summary,
        circuit_topology=circuit_topology,
        circuit_statistics={
            "topology_id": stage.output_topology_id,
            "parameter_count": stage.output_parameter_count,
        },
    )


def _complete_store(output: Path) -> Phase2ArtifactStore:
    """Create a complete deterministic test store with explicit stage timings.

    Returns:
        A verified store containing all four configured stages.
    """
    pipeline = _pipeline_with_every_stage_kind()
    store = Phase2ArtifactStore(output, pipeline, _fingerprint(pipeline))
    predecessor: NDArray[np.float64] | None = None
    for stage in pipeline.stages:
        evidence = _stage_evidence(stage, predecessor)
        store.publish_stage(
            evidence,
            wall_time_seconds=float(stage.stage_index + 1),
            peak_memory_bytes=1000 + stage.stage_index,
        )
        predecessor = evidence.selected_parameters
    assert store.pipeline_result is not None
    return store


def _external_consumer_pipeline(
    producer: TrainingPipelineResult,
    checkpoint_index: int,
    checkpoint_path: str,
) -> TrainingPipelineConfig:
    """Resolve a one-stage pipeline consuming one exact producer checkpoint.

    Returns:
        A target-compatible pipeline whose input path is non-identifying.
    """
    reference = ExternalCheckpointRef.from_pipeline_result(producer, checkpoint_index)
    base_stage = _stage_template(
        index=0,
        stage_id="resume_d2",
        kind="optimize",
        input_topology="bmpd_d2",
        output_topology="bmpd_d2",
        input_parameters=108,
        output_parameters=108,
        transfer="copy",
        iterations=2,
    )
    stage_policy = dict(base_stage.stage_policy)
    stage_policy["parameter_transfer_rule"] = "load_checkpoint"
    stage_policy["optimizer_id"] = "wp18_test_optimizer"
    stage = TrainingStageTemplate(stage_policy=stage_policy, seed_bindings=base_stage.seed_bindings)
    template = TrainingPipelineTemplate(
        template_id="wp18_external_checkpoint_consumer",
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        target_scope_id="primary_q6",
        ansatz_family="bmpd_brickwall",
        method_id="fixed_depth_bmpd_crn",
        method_version="1",
        resource_stratum_id="primary_cap_12",
        stages=(stage,),
        seed_domains=_seed_domains(),
        final_materialization_policy=_materialization_policy(),
    )
    manifest = _screening_target_manifest()
    producer_config = producer.config
    target_ref = producer_config.target_ref
    assert target_ref is not None
    target_spec = target_ref.target_spec
    return template.resolve(
        target_namespace="phase2",
        target_manifest=manifest,
        target_instance_id=target_spec.target_instance_id,
        target_population_manifest_checksum=manifest.content_checksum,
        target_instance_spec_checksum=target_spec.content_checksum,
        target_family_id=target_spec.family_id,
        target_stratum_id=target_spec.stratum_id,
        qubit_count=target_spec.qubit_count,
        optimization_block_id=producer_config.optimization_block_id,
        optimization_seed=producer_config.optimization_seed,
        data_role="screening_selection",
        input_checkpoint_path=checkpoint_path,
        input_checkpoint_ref=reference,
    )


@pytest.mark.parametrize(
    ("interrupted_index", "stage_kind"),
    [(0, "optimize"), (1, "grow"), (3, "prune")],
)
def test_interruption_resumes_each_stage_kind_without_replay(
    tmp_path: Path,
    interrupted_index: int,
    stage_kind: str,
) -> None:
    """Failures retain the prefix and resume only the unfinished suffix."""
    pipeline = _pipeline_with_every_stage_kind()
    fingerprint = _fingerprint(pipeline)
    output = tmp_path / stage_kind
    store = Phase2ArtifactStore(output, pipeline, fingerprint)
    initial_calls: list[int] = []

    def interrupt(
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> StageExecutionEvidence:
        initial_calls.append(stage.stage_index)
        if stage.stage_index == interrupted_index:
            msg = f"interrupted {stage.stage_kind}"
            raise RuntimeError(msg)
        return _stage_evidence(stage, predecessor_parameters)

    stopped = Phase2PipelineExecutor(store).execute(interrupt)
    assert isinstance(stopped, PipelineExecutionFailure)
    assert stopped.completed_stage_count == interrupted_index
    assert stopped.failure.stage_index == interrupted_index
    assert stopped.failure.message == f"interrupted {stage_kind}"

    orphan = output / "checkpoints" / "uncommitted-stage.npz"
    orphan.write_bytes(b"partial checkpoint")
    temporary = output / "traces" / ".interrupted.tmp"
    temporary.write_bytes(b"partial atomic write")
    managed_root_temporary = output / ".manifest.json.interrupted.tmp"
    managed_root_temporary.write_bytes(b"partial root atomic write")
    unrelated_temporary = output / "unrelated" / ".draft.tmp"
    unrelated_temporary.parent.mkdir()
    unrelated_temporary.write_bytes(b"user-owned temporary data")

    resumed = Phase2ArtifactStore(output, pipeline, fingerprint, resume=True)
    assert resumed.completed_stage_count == interrupted_index
    assert len(resumed.stage_failures) == 1
    assert not orphan.exists()
    assert not temporary.exists()
    assert not managed_root_temporary.exists()
    assert unrelated_temporary.read_bytes() == b"user-owned temporary data"
    resumed_calls: list[int] = []

    def finish(
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> StageExecutionEvidence:
        resumed_calls.append(stage.stage_index)
        return _stage_evidence(stage, predecessor_parameters)

    completed = Phase2PipelineExecutor(resumed).execute(finish)
    assert not isinstance(completed, PipelineExecutionFailure)
    assert resumed_calls == list(range(interrupted_index, len(pipeline.stages)))
    assert resumed.completed_stage_count == len(pipeline.stages)
    assert len(resumed.stage_failures) == 1

    replay_calls: list[int] = []

    def unexpected_replay(
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> StageExecutionEvidence:
        del predecessor_parameters
        replay_calls.append(stage.stage_index)
        msg = "A completed stage was replayed."
        raise AssertionError(msg)

    replayed = Phase2PipelineExecutor(Phase2ArtifactStore(output, pipeline, fingerprint, resume=True)).execute(
        unexpected_replay
    )
    assert not isinstance(replayed, PipelineExecutionFailure)
    assert replay_calls == []


def test_managed_artifact_symlinks_are_never_followed_or_target_deleted(tmp_path: Path) -> None:
    """Referenced aliases fail closed while orphan cleanup unlinks only the alias."""
    completed = _complete_store(tmp_path / "referenced_alias")
    artifact = completed.stage_artifacts[0]
    checkpoint_path = completed.output_directory / artifact.stage_result.produced_checkpoint_path
    checkpoint_payload = checkpoint_path.read_bytes()
    backing = tmp_path / "referenced_backing.npz"
    backing.write_bytes(checkpoint_payload)
    checkpoint_path.unlink()
    checkpoint_path.symlink_to(backing)

    with pytest.raises(Phase2ArtifactVerificationError, match="symbolic links"):
        Phase2ArtifactStore(
            completed.output_directory,
            completed.pipeline,
            completed.fingerprint,
            resume=True,
        )
    assert checkpoint_path.is_symlink()
    assert backing.read_bytes() == checkpoint_payload

    pipeline = _pipeline_with_every_stage_kind()
    output = tmp_path / "orphan_alias"
    Phase2ArtifactStore(output, pipeline, _fingerprint(pipeline))
    orphan_backing = tmp_path / "orphan_backing.bin"
    orphan_backing.write_bytes(b"must survive")
    orphan_alias = output / "checkpoints" / "orphan.npz"
    orphan_alias.symlink_to(orphan_backing)
    Phase2ArtifactStore(output, pipeline, _fingerprint(pipeline), resume=True)
    assert not orphan_alias.exists()
    assert orphan_backing.read_bytes() == b"must survive"


def test_non_exception_abort_is_preserved_and_re_raised(tmp_path: Path) -> None:
    """Process-control BaseExceptions are never converted into normal outcomes."""
    pipeline = _pipeline_with_every_stage_kind()
    store = Phase2ArtifactStore(tmp_path / "abort", pipeline, _fingerprint(pipeline))

    def abort(
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> StageExecutionEvidence:
        del stage, predecessor_parameters
        raise GeneratorExit

    with pytest.raises(GeneratorExit):
        Phase2PipelineExecutor(store).execute(abort)

    assert store.completed_stage_count == 0
    assert len(store.stage_failures) == 1
    assert store.stage_failures[0].exception_type == "GeneratorExit"


def test_executor_rejects_invalid_statistics_provider_before_running(tmp_path: Path) -> None:
    """Invalid callback wiring fails before any expensive stage work starts."""
    pipeline = _pipeline_with_every_stage_kind()
    store = Phase2ArtifactStore(tmp_path / "statistics_provider", pipeline, _fingerprint(pipeline))
    calls: list[int] = []

    def runner(
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> StageExecutionEvidence:
        calls.append(stage.stage_index)
        return _stage_evidence(stage, predecessor_parameters)

    with pytest.raises(TypeError, match="circuit_statistics must be callable or None"):
        Phase2PipelineExecutor(store).execute(
            runner,
            circuit_statistics=cast("Callable[..., Mapping[str, object]]", {}),
        )

    assert calls == []
    assert store.completed_stage_count == 0


def test_atomic_stage_commit_recovers_orphans_and_rejects_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ledger commit is retryable, but committed byte damage is fatal."""
    pipeline = _pipeline_with_every_stage_kind()
    fingerprint = _fingerprint(pipeline)
    output = tmp_path / "atomic"
    store = Phase2ArtifactStore(output, pipeline, fingerprint)
    evidence = _stage_evidence(pipeline.stages[0], None)
    real_atomic_write = artifact_module.atomic_write_bytes

    def fail_stage_ledger(path: Path, payload: bytes) -> None:
        if path == store.stage_result_stream_path:
            msg = "simulated ledger interruption"
            raise OSError(msg)
        real_atomic_write(path, payload)

    with monkeypatch.context() as context:
        context.setattr(artifact_module, "atomic_write_bytes", fail_stage_ledger)
        with pytest.raises(OSError, match="ledger interruption"):
            store.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=10)
    assert store.completed_stage_count == 0
    assert any(path.is_file() for path in store.checkpoint_directory.rglob("*"))

    recovered = Phase2ArtifactStore(output, pipeline, fingerprint, resume=True)
    assert recovered.completed_stage_count == 0
    assert not any(path.is_file() for path in recovered.checkpoint_directory.rglob("*"))
    artifact = recovered.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=10)
    checkpoint_path = output / artifact.stage_result.produced_checkpoint_path
    checkpoint_path.write_bytes(b"corrupt committed checkpoint")
    with pytest.raises(Phase2ArtifactVerificationError, match="checksum"):
        Phase2ArtifactStore(output, pipeline, fingerprint, resume=True)


def test_stale_store_handle_cannot_replace_a_committed_stage(tmp_path: Path) -> None:
    """A writer opened before another commit must reopen instead of overwriting it."""
    pipeline = _pipeline_with_every_stage_kind()
    fingerprint = _fingerprint(pipeline)
    output = tmp_path / "stale_writer"
    writer = Phase2ArtifactStore(output, pipeline, fingerprint)
    stale = Phase2ArtifactStore(output, pipeline, fingerprint, resume=True)
    evidence = _stage_evidence(pipeline.stages[0], None)

    committed = writer.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=10)
    stage_stream = writer.stage_result_stream_path.read_bytes()
    manifest = writer.manifest_path.read_bytes()
    with pytest.raises(Phase2ConcurrentMutationError, match="advanced after this handle opened"):
        stale.publish_stage(evidence, wall_time_seconds=2.0, peak_memory_bytes=20)

    assert writer.stage_result_stream_path.read_bytes() == stage_stream
    assert writer.manifest_path.read_bytes() == manifest
    reopened = Phase2ArtifactStore(output, pipeline, fingerprint, resume=True)
    assert reopened.stage_artifacts == (committed,)


def test_executor_propagates_stale_writer_error_without_recording_failure(tmp_path: Path) -> None:
    """The executor never converts a failed ownership check into stage evidence."""
    pipeline = _pipeline_with_every_stage_kind()
    fingerprint = _fingerprint(pipeline)
    output = tmp_path / "stale_executor"
    writer = Phase2ArtifactStore(output, pipeline, fingerprint)
    stale = Phase2ArtifactStore(output, pipeline, fingerprint, resume=True)
    evidence = _stage_evidence(pipeline.stages[0], None)
    writer.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=10)
    runner_calls: list[int] = []

    def stale_runner(
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> StageExecutionEvidence:
        del predecessor_parameters
        runner_calls.append(stage.stage_index)
        return evidence

    with pytest.raises(Phase2ConcurrentMutationError, match="advanced after this handle opened"):
        Phase2PipelineExecutor(stale).execute(stale_runner)

    reopened = Phase2ArtifactStore(output, pipeline, fingerprint, resume=True)
    assert runner_calls == []
    assert reopened.completed_stage_count == 1
    assert reopened.stage_failures == ()


def test_evaluator_propagates_stale_writer_error_without_recursive_failure(tmp_path: Path) -> None:
    """The evaluator does not publish failure rows after losing store ownership."""
    writer = _complete_store(tmp_path / "stale_evaluator")
    stale = Phase2ArtifactStore(
        writer.output_directory,
        writer.pipeline,
        writer.fingerprint,
        resume=True,
    )
    payload = b"stale evaluator materialization"
    config = _evaluation_config(writer, payload, repetition=0)
    writer.record_materialization_failure(
        config=config,
        exception=RuntimeError("advance the retained manifest"),
        phase="materialization",
        wall_time_seconds=0.1,
    )
    materialization_calls: list[int] = []

    def stale_materialization(
        pipeline_result: TrainingPipelineResult,
        parameters: NDArray[np.float64],
    ) -> MaterializedCircuitPayload:
        del pipeline_result, parameters
        materialization_calls.append(1)
        return MaterializedCircuitPayload(
            serialized_bytes=payload,
            wall_time_seconds=0.2,
            peak_memory_bytes=20,
        )

    def unexpected_evaluation(
        evaluation_config: PipelineEvaluationConfig,
        runtime_circuit: object,
    ) -> PipelineEvaluationMeasurement:
        del evaluation_config, runtime_circuit
        msg = "evaluation must not run before materialization publication"
        raise AssertionError(msg)

    evaluator = ParallelPhase2Evaluator(stale, lambda serialized: serialized)
    with pytest.raises(Phase2ConcurrentMutationError, match="advanced after this handle opened"):
        evaluator.evaluate(
            (config,),
            stale_materialization,
            unexpected_evaluation,
            max_workers=1,
        )

    reopened = Phase2ArtifactStore(
        writer.output_directory,
        writer.pipeline,
        writer.fingerprint,
        resume=True,
    )
    assert materialization_calls == []
    assert len(reopened.materialization_attempts) == 1
    assert reopened.records == ()
    assert reopened.evaluation_failures == ()


def test_store_lock_reports_a_concurrent_owner_without_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two handles cannot enter a mutation while the OS-level writer lock is held."""
    monkeypatch.setattr(artifact_module, "_STORE_LOCK_TIMEOUT_SECONDS", 0.0)
    pipeline = _pipeline_with_every_stage_kind()
    fingerprint = _fingerprint(pipeline)
    output = tmp_path / "owned_lock"
    owner = Phase2ArtifactStore(output, pipeline, fingerprint)
    contender = Phase2ArtifactStore(output, pipeline, fingerprint, resume=True)
    evidence = _stage_evidence(pipeline.stages[0], None)
    stage_stream = owner.stage_result_stream_path.read_bytes()
    manifest = owner.manifest_path.read_bytes()

    external_owner = FileLock(output / ".phase2-artifact-store.lock", timeout=0.0)
    with external_owner, pytest.raises(Phase2ConcurrentMutationError, match="currently owns"):
        contender.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=10)

    assert owner.stage_result_stream_path.read_bytes() == stage_stream
    assert owner.manifest_path.read_bytes() == manifest


def test_torn_manifest_commit_poison_stale_handles_until_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forward ledger row is recovered once; no stale handle may write over it."""
    pipeline = _pipeline_with_every_stage_kind()
    fingerprint = _fingerprint(pipeline)
    output = tmp_path / "torn_manifest"
    interrupted = Phase2ArtifactStore(output, pipeline, fingerprint)
    stale = Phase2ArtifactStore(output, pipeline, fingerprint, resume=True)
    evidence = _stage_evidence(pipeline.stages[0], None)
    prior_manifest = interrupted.manifest_path.read_bytes()
    real_atomic_write = artifact_module.atomic_write_bytes

    def fail_manifest(path: Path, payload: bytes) -> None:
        if path == interrupted.manifest_path:
            msg = "simulated manifest interruption"
            raise OSError(msg)
        real_atomic_write(path, payload)

    with monkeypatch.context() as context:
        context.setattr(artifact_module, "atomic_write_bytes", fail_manifest)
        with pytest.raises(OSError, match="manifest interruption"):
            interrupted.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=10)

    assert interrupted.manifest_path.read_bytes() == prior_manifest
    with pytest.raises(Phase2ConcurrentMutationError, match="must be reopened"):
        interrupted.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=10)
    with pytest.raises(Phase2ConcurrentMutationError, match="needs recovery; reopen"):
        stale.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=10)

    recovered = Phase2ArtifactStore(output, pipeline, fingerprint, resume=True)
    assert recovered.completed_stage_count == 1


def _evaluation_config(
    store: Phase2ArtifactStore,
    payload: bytes,
    *,
    repetition: int,
) -> PipelineEvaluationConfig:
    """Create one small fixed-sample screening row.

    Returns:
        A valid evaluation config linked to the store's complete pipeline.
    """
    pipeline = store.pipeline_result
    assert pipeline is not None
    return PipelineEvaluationConfig.for_pipeline(
        pipeline=pipeline,
        materialized_circuit_checksum=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        test_noise_id="depolarizing_1s_all",
        noise_definition_version=pipeline.config.stages[-2].noise_definition_version,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        evaluation_seed=909,
        evaluation_seed_domain="screening_selection",
        repetition=repetition,
        trajectory_budget=3,
        evaluation_policy="fixed_sample",
        confidence_level=None,
        confidence_interval_method=None,
        sidecar_storage_policy="trajectory_fidelities",
        max_bond_dimension=64,
        svd_threshold=0.0,
        truncation_mode="discarded_weight",
        min_bond_dimension=1,
    )


def _evaluation_ensemble(
    config: PipelineEvaluationConfig,
    *,
    circuit_checksum: str | None = None,
    provider_checksum: str | None = None,
) -> KrotovFixedMapEnsemble:
    """Create fresh screening maps for one final-evaluation row.

    Returns:
        A role- and seed-bound ensemble covering the complete row budget.
    """
    assert config.evaluation_seed is not None
    return KrotovFixedMapEnsemble(
        role="screening_selection",
        resolved_seed=config.evaluation_seed,
        stage_index=0,
        stage_id="final_evaluation",
        stage_configuration_checksum=config.configuration_checksum,
        circuit_checksum=config.materialized_circuit_checksum if circuit_checksum is None else circuit_checksum,
        provider_checksum=_checksum("2") if provider_checksum is None else provider_checksum,
        ensemble_index=config.repetition,
        refresh_index=0,
        global_iteration_start=0,
        trajectory_maps=[
            [KrotovNoiseMap(source_gate_index=0, is_identity=True)] for _ in range(config.trajectory_budget)
        ],
    )


def test_evaluation_rejects_wrong_map_bindings_and_event_count(tmp_path: Path) -> None:
    """Final evaluation rejects row, circuit, provider, and event-count mismatches."""
    store = _complete_store(tmp_path / "invalid_evaluation_maps")
    payload = b"deterministic materialized circuit"
    config = _evaluation_config(store, payload, repetition=0)
    other_config = _evaluation_config(store, payload, repetition=1)
    materialization = store.publish_materialized_circuit(
        config=config,
        payload=payload,
        wall_time_seconds=0.5,
        peak_memory_bytes=1024,
    )
    valid_maps = _evaluation_ensemble(config)
    invalid_bindings = (
        _evaluation_ensemble(other_config),
        _evaluation_ensemble(config, circuit_checksum=_checksum("3")),
        _evaluation_ensemble(config, provider_checksum=_checksum("4")),
    )
    for invalid_maps in invalid_bindings:
        with pytest.raises(ValueError, match="bind the planned row"):
            store.write_evaluation_success(
                config=config,
                materialization=materialization,
                test_noiseless_fidelity=0.95,
                trajectory_fidelities=(0.8, 0.9, 1.0),
                sampled_nonidentity_events=0,
                normalized_work=_work(objective=1, test=3, gates=3),
                evaluation_wall_time_seconds=0.75,
                peak_memory_bytes=2048,
                evaluation_provider_checksum=_checksum("2"),
                evaluation_ensembles=(invalid_maps,),
            )

    with pytest.raises(ValueError, match="sampled_nonidentity_events"):
        store.write_evaluation_success(
            config=config,
            materialization=materialization,
            test_noiseless_fidelity=0.95,
            trajectory_fidelities=(0.8, 0.9, 1.0),
            sampled_nonidentity_events=1,
            normalized_work=_work(objective=1, test=3, gates=3),
            evaluation_wall_time_seconds=0.75,
            peak_memory_bytes=2048,
            evaluation_provider_checksum=_checksum("2"),
            evaluation_ensembles=(valid_maps,),
        )
    assert store.records == ()


def test_result_store_links_rows_preserves_failures_and_accounts_time(tmp_path: Path) -> None:
    """JSONL, CSV, manifest, maps, sidecars, failures, and timings agree."""
    output = tmp_path / "results"
    store = _complete_store(output)
    payload = b"deterministic materialized circuit"
    success_config = _evaluation_config(store, payload, repetition=0)
    failure_config = _evaluation_config(store, payload, repetition=1)
    materialization = store.publish_materialized_circuit(
        config=success_config,
        payload=payload,
        wall_time_seconds=2.5,
        peak_memory_bytes=4096,
    )
    evaluation_maps = _evaluation_ensemble(success_config)
    success = store.write_evaluation_success(
        config=success_config,
        materialization=materialization,
        test_noiseless_fidelity=0.95,
        trajectory_fidelities=(0.8, 0.9, 1.0),
        sampled_nonidentity_events=0,
        normalized_work=_work(objective=1, test=3, gates=3),
        evaluation_wall_time_seconds=1.25,
        peak_memory_bytes=2048,
        evaluation_provider_checksum=_checksum("2"),
        evaluation_ensembles=(evaluation_maps,),
    )
    assert isinstance(success, PipelineBenchmarkResult)

    failure = store.write_evaluation_failure(
        config=failure_config,
        exception=RuntimeError("evaluation worker stopped"),
        phase="evaluation",
        wall_time_seconds=0.75,
        materialization=materialization,
        retryable=True,
    )
    assert isinstance(failure, PipelineBenchmarkFailure)
    assert failure.config.pipeline_result_checksum == success.config.pipeline_result_checksum
    assert failure.materialized_circuit_checksum == materialization.payload_checksum

    reopened = Phase2ArtifactStore(
        output,
        store.pipeline,
        store.fingerprint,
        resume=True,
    )
    assert reopened.records == (success, failure)
    assert reopened.evaluation_failures == (failure,)
    jsonl_records = tuple(
        pipeline_benchmark_record_from_json(line)
        for line in reopened.results_jsonl_path.read_text(encoding="utf-8").splitlines()
    )
    with reopened.results_csv_path.open(newline="", encoding="utf-8") as csv_file:
        csv_records = tuple(pipeline_benchmark_record_from_csv_row(row) for row in csv.DictReader(csv_file))
    assert jsonl_records == reopened.records
    assert csv_records == reopened.records

    manifest = dict(load_canonical_json_object(reopened.manifest_path.read_text(encoding="utf-8")))
    manifest_checksum = cast("str", manifest.pop("content_checksum"))
    assert manifest_checksum == canonical_checksum(manifest)
    assert manifest["record_count"] == 2
    assert manifest["successful_evaluation_row_ids"] == (success.evaluation_row_id,)
    assert manifest["failed_evaluation_row_ids"] == (failure.evaluation_row_id,)
    timing = cast("dict[str, float]", manifest["timing"])
    assert timing == {
        "stage_execution_seconds": 10.0,
        "circuit_materialization_seconds": 2.5,
        "row_evaluation_seconds": 2.0,
        "total_wall_time_seconds": 14.5,
    }
    assert success.trajectory_sidecar_path is not None
    assert (output / success.trajectory_sidecar_path).is_file()


def test_materialization_failure_is_one_shared_timed_attempt(tmp_path: Path) -> None:
    """A failed shared build is timed once while every planned row remains explicit."""
    output = tmp_path / "materialization_failure"
    store = _complete_store(output)
    payload = b"planned materialized circuit"
    configs = tuple(_evaluation_config(store, payload, repetition=index) for index in range(3))
    evaluation_called = False

    def fail_materialization(
        _pipeline: TrainingPipelineResult,
        _parameters: NDArray[np.float64],
    ) -> MaterializedCircuitPayload:
        msg = "deterministic compiler failure"
        raise RuntimeError(msg)

    def unexpected_evaluation(
        _config: PipelineEvaluationConfig,
        _circuit: object,
    ) -> PipelineEvaluationMeasurement:
        nonlocal evaluation_called
        evaluation_called = True
        msg = "evaluation cannot run after materialization failure"
        raise AssertionError(msg)

    records = ParallelPhase2Evaluator(store, lambda payload: payload).evaluate(
        configs,
        fail_materialization,
        unexpected_evaluation,
        max_workers=2,
    )
    assert not evaluation_called
    assert all(isinstance(record, PipelineBenchmarkFailure) for record in records)
    assert all(cast("PipelineBenchmarkFailure", record).wall_time_seconds == pytest.approx(0.0) for record in records)
    assert len(store.materialization_attempts) == 1
    attempt = store.materialization_attempts[0]
    assert attempt.status == "failure"
    assert attempt.phase == "materialization"
    assert attempt.message == "deterministic compiler failure"

    manifest = dict(load_canonical_json_object(store.manifest_path.read_text(encoding="utf-8")))
    timing = cast("dict[str, float]", manifest["timing"])
    assert timing["stage_execution_seconds"] == pytest.approx(10.0)
    assert timing["circuit_materialization_seconds"] == pytest.approx(attempt.wall_time_seconds)
    assert timing["row_evaluation_seconds"] == pytest.approx(0.0)
    assert timing["total_wall_time_seconds"] == pytest.approx(10.0 + attempt.wall_time_seconds)

    reopened = Phase2ArtifactStore(output, store.pipeline, store.fingerprint, resume=True)
    assert reopened.materialization_attempts == (attempt,)
    assert reopened.records == records


def test_resume_recovers_torn_cross_stream_commits(tmp_path: Path) -> None:
    """Authoritative ledgers repair interrupted attempt and current-row projections."""
    materialization_store = _complete_store(tmp_path / "recover_materialization")
    payload = b"recoverable deterministic circuit"
    config = _evaluation_config(materialization_store, payload, repetition=0)
    prior_manifest = materialization_store.manifest_path.read_bytes()
    prior_attempts = materialization_store.materialization_attempt_stream_path.read_bytes()
    artifact = materialization_store.publish_materialized_circuit(
        config=config,
        payload=payload,
        wall_time_seconds=1.75,
        peak_memory_bytes=512,
    )
    materialization_store.materialization_attempt_stream_path.write_bytes(prior_attempts)
    materialization_store.manifest_path.write_bytes(prior_manifest)

    recovered_materialization = Phase2ArtifactStore(
        materialization_store.output_directory,
        materialization_store.pipeline,
        materialization_store.fingerprint,
        resume=True,
    )
    assert recovered_materialization.materializations == (artifact,)
    assert len(recovered_materialization.materialization_attempts) == 1
    repaired_attempt = recovered_materialization.materialization_attempts[0]
    assert repaired_attempt.status == "success"
    assert repaired_attempt.payload_checksum == artifact.payload_checksum
    assert repaired_attempt.wall_time_seconds == pytest.approx(artifact.wall_time_seconds)

    failure_store = _complete_store(tmp_path / "recover_failure_projection")
    failure_config = _evaluation_config(failure_store, payload, repetition=0)
    materialization = failure_store.publish_materialized_circuit(
        config=failure_config,
        payload=payload,
        wall_time_seconds=0.5,
        peak_memory_bytes=128,
    )
    first_failure = failure_store.write_evaluation_failure(
        config=failure_config,
        exception=RuntimeError("first attempt"),
        phase="evaluation",
        wall_time_seconds=0.25,
        materialization=materialization,
    )
    prior_results = failure_store.results_jsonl_path.read_bytes()
    prior_failure_manifest = failure_store.manifest_path.read_bytes()
    second_failure = failure_store.write_evaluation_failure(
        config=failure_config,
        exception=RuntimeError("second attempt"),
        phase="evaluation",
        wall_time_seconds=0.5,
        materialization=materialization,
    )
    failure_store.results_jsonl_path.write_bytes(prior_results)
    failure_store.manifest_path.write_bytes(prior_failure_manifest)

    recovered_failure = Phase2ArtifactStore(
        failure_store.output_directory,
        failure_store.pipeline,
        failure_store.fingerprint,
        resume=True,
    )
    assert recovered_failure.evaluation_failures == (first_failure, second_failure)
    assert recovered_failure.records == (second_failure,)


def test_manifest_baseline_rejects_missing_truncated_and_removed_rows(tmp_path: Path) -> None:
    """A committed manifest makes canonical stream deletion and rollback detectable."""
    missing = _complete_store(tmp_path / "missing_stream")
    missing.stage_failure_stream_path.unlink()
    with pytest.raises(Phase2ArtifactVerificationError, match="missing"):
        Phase2ArtifactStore(missing.output_directory, missing.pipeline, missing.fingerprint, resume=True)

    truncated = _complete_store(tmp_path / "truncated_stream")
    stage_bytes = truncated.stage_result_stream_path.read_bytes()
    truncated.stage_result_stream_path.write_bytes(stage_bytes[:-1])
    with pytest.raises(Phase2ArtifactVerificationError, match="not terminated"):
        Phase2ArtifactStore(truncated.output_directory, truncated.pipeline, truncated.fingerprint, resume=True)

    rolled_back = _complete_store(tmp_path / "removed_row")
    rows = rolled_back.stage_result_stream_path.read_bytes().splitlines(keepends=True)
    rolled_back.stage_result_stream_path.write_bytes(b"".join(rows[:-1]))
    with pytest.raises(Phase2ArtifactVerificationError, match="committed prefix"):
        Phase2ArtifactStore(rolled_back.output_directory, rolled_back.pipeline, rolled_back.fingerprint, resume=True)


def test_resume_override_is_idempotent_and_failed_validation_is_nonmutating(tmp_path: Path) -> None:
    """An exact recorded override is reusable while unrelated overrides leave no trace."""
    store = _complete_store(tmp_path / "override")
    changed = replace(store.fingerprint, starting_commit="e" * 40)
    override = NonScientificResumeOverride(
        stored_fingerprint=store.fingerprint,
        current_fingerprint=changed,
        reason="Diagnostic recovery only; excluded from scientific analysis.",
    )
    resumed = Phase2ArtifactStore(
        store.output_directory,
        store.pipeline,
        changed,
        resume=True,
        resume_override=override,
    )
    repeated = Phase2ArtifactStore(
        store.output_directory,
        store.pipeline,
        changed,
        resume=True,
        resume_override=override,
    )
    assert resumed.runtime_fingerprint_checksum == changed.content_checksum
    assert repeated.runtime_fingerprint_checksum == changed.content_checksum
    override_rows = repeated.resume_override_stream_path.read_text(encoding="utf-8").splitlines()
    assert len(override_rows) == 1

    fingerprint_before = repeated.fingerprint_path.read_bytes()
    overrides_before = repeated.resume_override_stream_path.read_bytes()
    unrelated = replace(changed, dependency_versions={**changed.dependency_versions, "numpy": "99.0"})
    invalid_override = NonScientificResumeOverride(
        stored_fingerprint=changed,
        current_fingerprint=unrelated,
        reason="Does not bind this matching resume pair.",
    )
    with pytest.raises(Phase2ResumeMismatchError):
        Phase2ArtifactStore(
            store.output_directory,
            store.pipeline,
            changed,
            resume=True,
            resume_override=invalid_override,
        )
    assert repeated.fingerprint_path.read_bytes() == fingerprint_before
    assert repeated.resume_override_stream_path.read_bytes() == overrides_before


def test_manifest_runtime_rejects_rollback_and_recovers_torn_forward_override(tmp_path: Path) -> None:
    """The manifest permits only directionally recorded runtime-fingerprint transitions."""
    rolled_back = _complete_store(tmp_path / "runtime_rollback")
    original = rolled_back.fingerprint
    changed = replace(original, starting_commit="e" * 40)
    override = NonScientificResumeOverride(
        stored_fingerprint=original,
        current_fingerprint=changed,
        reason="Diagnostic recovery only; excluded from scientific analysis.",
    )
    transitioned = Phase2ArtifactStore(
        rolled_back.output_directory,
        rolled_back.pipeline,
        changed,
        resume=True,
        resume_override=override,
    )
    transitioned_manifest = transitioned.manifest_path.read_bytes()
    transitioned.fingerprint_path.write_bytes(f"{original.to_json()}\n".encode())

    with pytest.raises(Phase2ArtifactVerificationError, match="rolls back the manifest's active runtime"):
        Phase2ArtifactStore(
            transitioned.output_directory,
            transitioned.pipeline,
            original,
            resume=True,
        )

    assert transitioned.manifest_path.read_bytes() == transitioned_manifest
    repaired = Phase2ArtifactStore(
        transitioned.output_directory,
        transitioned.pipeline,
        changed,
        resume=True,
    )
    assert repaired.runtime_fingerprint_checksum == changed.content_checksum
    assert ResumabilityFingerprint.from_json(repaired.fingerprint_path.read_text(encoding="utf-8")) == changed

    torn = _complete_store(tmp_path / "torn_runtime_override")
    prior_manifest = torn.manifest_path.read_bytes()
    prior_fingerprint = torn.fingerprint_path.read_bytes()
    torn_changed = replace(torn.fingerprint, starting_commit="f" * 40)
    torn_override = NonScientificResumeOverride(
        stored_fingerprint=torn.fingerprint,
        current_fingerprint=torn_changed,
        reason="Diagnostic recovery only; excluded from scientific analysis.",
    )
    transitioned_torn = Phase2ArtifactStore(
        torn.output_directory,
        torn.pipeline,
        torn_changed,
        resume=True,
        resume_override=torn_override,
    )
    transitioned_torn.manifest_path.write_bytes(prior_manifest)
    transitioned_torn.fingerprint_path.write_bytes(prior_fingerprint)

    recovered_torn = Phase2ArtifactStore(
        torn.output_directory,
        torn.pipeline,
        torn_changed,
        resume=True,
    )
    recovered_manifest = load_canonical_json_object(recovered_torn.manifest_path.read_text(encoding="utf-8"))
    assert recovered_manifest["active_runtime_fingerprint_checksum"] == torn_changed.content_checksum


def test_resume_override_stream_is_one_contiguous_acyclic_chain(tmp_path: Path) -> None:
    """Ordered A-to-B-to-C transitions are valid, but a later fork is corruption."""
    store = _complete_store(tmp_path / "override_chain")
    fingerprint_a = store.fingerprint
    fingerprint_b = replace(fingerprint_a, starting_commit="e" * 40)
    override_ab = NonScientificResumeOverride(
        stored_fingerprint=fingerprint_a,
        current_fingerprint=fingerprint_b,
        reason="Diagnostic transition A to B; excluded from scientific analysis.",
    )
    store_b = Phase2ArtifactStore(
        store.output_directory,
        store.pipeline,
        fingerprint_b,
        resume=True,
        resume_override=override_ab,
    )
    manifest_b = store_b.manifest_path.read_bytes()
    fingerprint_file_b = store_b.fingerprint_path.read_bytes()
    fingerprint_c = replace(fingerprint_b, starting_commit="f" * 40)
    override_bc = NonScientificResumeOverride(
        stored_fingerprint=fingerprint_b,
        current_fingerprint=fingerprint_c,
        reason="Diagnostic transition B to C; excluded from scientific analysis.",
    )
    store_c = Phase2ArtifactStore(
        store.output_directory,
        store.pipeline,
        fingerprint_c,
        resume=True,
        resume_override=override_bc,
    )
    assert len(store_c.resume_override_stream_path.read_text(encoding="utf-8").splitlines()) == 2
    store_c.manifest_path.write_bytes(manifest_b)
    store_c.fingerprint_path.write_bytes(fingerprint_file_b)
    with pytest.raises(Phase2ArtifactVerificationError, match="Only the final torn-forward"):
        Phase2ArtifactStore(
            store.output_directory,
            store.pipeline,
            fingerprint_b,
            resume=True,
        )
    store_c = Phase2ArtifactStore(
        store.output_directory,
        store.pipeline,
        fingerprint_c,
        resume=True,
    )

    fingerprint_d = replace(fingerprint_c, starting_commit="a" * 40)
    fork = NonScientificResumeOverride(
        stored_fingerprint=fingerprint_b,
        current_fingerprint=fingerprint_d,
        reason="Invalid fork from B; excluded from scientific analysis.",
    )
    with store_c.resume_override_stream_path.open("ab") as stream:
        stream.write(f"{fork.to_json()}\n".encode())

    with pytest.raises(Phase2ArtifactVerificationError, match="disconnected from or forks"):
        Phase2ArtifactStore(
            store.output_directory,
            store.pipeline,
            fingerprint_c,
            resume=True,
        )


def test_checkpoint_selection_and_identities_survive_store_reopen(tmp_path: Path) -> None:
    """Selected/final vectors and every stable identity survive exact decoding."""
    store = _complete_store(tmp_path / "stable")
    final_training_index = 2
    checkpoint = store.load_stage_checkpoint(final_training_index)
    assert checkpoint.selected_global_iteration < checkpoint.completed_global_iteration
    assert checkpoint.selected_checkpoint_validation_fidelity == pytest.approx(0.91)
    assert not np.array_equal(checkpoint.selected_theta, checkpoint.final_theta)
    expected_ids = tuple(artifact.stage_result.pipeline_prefix_id for artifact in store.stage_artifacts)
    expected_checkpoint_bytes = tuple(
        (store.output_directory / artifact.stage_result.produced_checkpoint_path).read_bytes()
        for artifact in store.stage_artifacts
    )

    reopened = Phase2ArtifactStore(
        store.output_directory,
        store.pipeline,
        store.fingerprint,
        resume=True,
    )
    assert tuple(artifact.stage_result.pipeline_prefix_id for artifact in reopened.stage_artifacts) == expected_ids
    assert (
        tuple(
            (reopened.output_directory / artifact.stage_result.produced_checkpoint_path).read_bytes()
            for artifact in reopened.stage_artifacts
        )
        == expected_checkpoint_bytes
    )
    np.testing.assert_array_equal(reopened.load_final_parameters(), reopened.load_stage_checkpoint(3).selected_theta)


def test_genuine_wp17_objective_binding_publishes_and_verifies_on_reopen(tmp_path: Path) -> None:
    """Persisted WP17 evidence retains its typed target and zero-state objective."""
    pipeline = _objective_pipeline()
    evidence = _objective_evidence(pipeline, _screening_materialized_targets()[0])
    binding = evidence.objective_binding
    assert isinstance(binding, NoisyKrotovObjectiveBinding)
    assert binding.materialized_target_identity is not None

    store = Phase2ArtifactStore(tmp_path / "genuine_wp17", pipeline, _fingerprint(pipeline))
    artifact = store.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=100)
    metadata_path = store.output_directory / cast("str", artifact.stage_result.diagnostic_sidecar_path)
    metadata = load_canonical_json_object(metadata_path.read_text(encoding="utf-8"))
    assert NoisyKrotovObjectiveBinding.from_dict(metadata["objective_binding"]) == binding

    reopened = Phase2ArtifactStore(store.output_directory, pipeline, store.fingerprint, resume=True)
    assert reopened.completed_stage_count == 1
    metadata_path.write_bytes(metadata_path.read_bytes() + b" ")
    with pytest.raises(Phase2ArtifactVerificationError, match="checksum mismatch"):
        Phase2ArtifactStore(store.output_directory, pipeline, store.fingerprint, resume=True)


def test_wp17_wrong_target_and_custom_initial_state_reject_before_artifact_writes(tmp_path: Path) -> None:
    """Publication accepts only the pipeline target and computational-zero input."""
    pipeline = _objective_pipeline()
    configured_target, wrong_target = _screening_materialized_targets()[:2]
    cases = (
        (
            "wrong_target",
            _objective_evidence(pipeline, wrong_target),
            "does not match the configured pipeline target",
        ),
        (
            "custom_initial",
            _objective_evidence(pipeline, configured_target, initial_state=MPS(6, state="x+")),
            "computational-zero initial-state policy",
        ),
    )
    for name, evidence, message in cases:
        store = Phase2ArtifactStore(tmp_path / name, pipeline, _fingerprint(pipeline))
        before = {
            path.relative_to(store.output_directory).as_posix(): path.read_bytes()
            for path in store.output_directory.rglob("*")
            if path.is_file()
        }
        with pytest.raises(ValueError, match=message):
            store.publish_stage(evidence, wall_time_seconds=1.0, peak_memory_bytes=100)
        after = {
            path.relative_to(store.output_directory).as_posix(): path.read_bytes()
            for path in store.output_directory.rglob("*")
            if path.is_file()
        }
        assert after == before


def test_cross_stage_objective_mismatch_is_rejected_before_commit(tmp_path: Path) -> None:
    """Every optimized stage must retain one exact target/objective identity."""
    pipeline = _pipeline_with_every_stage_kind()
    store = Phase2ArtifactStore(tmp_path / "objective_binding", pipeline, _fingerprint(pipeline))
    first_evidence = _stage_evidence(pipeline.stages[0], None)
    store.publish_stage(first_evidence, wall_time_seconds=1.0, peak_memory_bytes=10)
    predecessor = store.load_stage_checkpoint(0).selected_theta
    expected = _stage_evidence(pipeline.stages[1], predecessor)
    mismatched = StageExecutionEvidence(
        stage=expected.stage,
        source_parameters=predecessor,
        initial_parameters=expected.initial_parameters,
        final_parameters=expected.final_parameters,
        selected_parameters=expected.selected_parameters,
        selected_global_iteration=expected.selected_global_iteration,
        completed_global_iteration=expected.completed_global_iteration,
        selected_checkpoint_validation_fidelity=expected.selected_checkpoint_validation_fidelity,
        circuit_binding_checksum=expected.circuit_binding_checksum,
        provider_checksum=expected.provider_checksum,
        objective_checksum=_checksum("2"),
        trace=expected.trace,
        training_ensembles=expected.training_ensembles,
        checkpoint_validation_ensembles=expected.checkpoint_validation_ensembles,
        normalized_work=expected.normalized_work,
        training_summary=expected.training_summary,
        checkpoint_validation_summary=expected.checkpoint_validation_summary,
        circuit_topology=expected.circuit_topology,
        circuit_statistics=expected.circuit_statistics,
        optimizer_state=expected.optimizer_state,
        cumulative_cross_trajectory_pairings=expected.cumulative_cross_trajectory_pairings,
    )
    stage_stream_before = store.stage_result_stream_path.read_bytes()

    with pytest.raises(Phase2ArtifactVerificationError, match="one exact target/objective binding"):
        store.publish_stage(mismatched, wall_time_seconds=2.0, peak_memory_bytes=20)

    assert store.completed_stage_count == 1
    assert store.stage_result_stream_path.read_bytes() == stage_stream_before
    reopened = Phase2ArtifactStore(store.output_directory, pipeline, store.fingerprint, resume=True)
    assert reopened.completed_stage_count == 1


def test_external_checkpoint_is_sealed_portable_and_hands_off_selected_theta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External inputs are preflighted, sealed, relocatable, and selection-aware."""
    producer_store = _complete_store(tmp_path / "producer")
    producer = producer_store.pipeline_result
    assert producer is not None
    checkpoint_index = 2
    checkpoint = producer_store.load_stage_checkpoint(checkpoint_index)
    assert not np.array_equal(checkpoint.selected_theta, checkpoint.final_theta)

    launch_directory = tmp_path / "launch"
    source_path = launch_directory / "inputs" / "source.npz"
    source_path.parent.mkdir(parents=True)
    producer_checkpoint_path = (
        producer_store.output_directory / producer.stage_results[checkpoint_index].produced_checkpoint_path
    )
    shutil.copyfile(producer_checkpoint_path, source_path)
    consumer = _external_consumer_pipeline(producer, checkpoint_index, "inputs/source.npz")
    output_directory = tmp_path / "consumer"

    monkeypatch.chdir(launch_directory)
    store = Phase2ArtifactStore(output_directory, consumer, _fingerprint(consumer))
    sealed = store.load_external_checkpoint()
    np.testing.assert_array_equal(sealed.selected_theta, checkpoint.selected_theta)
    source_path.unlink()

    relocated_root = tmp_path / "relocated"
    relocated_root.mkdir()
    monkeypatch.chdir(relocated_root)
    relocated = _external_consumer_pipeline(producer, checkpoint_index, "missing/relocated.npz")
    assert relocated.configuration_checksum == consumer.configuration_checksum
    reopened = Phase2ArtifactStore(output_directory, relocated, _fingerprint(relocated), resume=True)
    captured_sources: list[NDArray[np.float64]] = []

    def capture_source(
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray[np.float64] | None,
    ) -> StageExecutionEvidence:
        assert predecessor_parameters is not None
        captured_sources.append(predecessor_parameters.copy())
        return _stage_evidence(stage, predecessor_parameters)

    result = Phase2PipelineExecutor(reopened).execute(capture_source)
    assert isinstance(result, TrainingPipelineResult)
    assert len(captured_sources) == 1
    np.testing.assert_array_equal(captured_sources[0], checkpoint.selected_theta)
    assert not np.array_equal(captured_sources[0], checkpoint.final_theta)

    invalid_new_output = tmp_path / "invalid_new"
    with pytest.raises(Phase2ArtifactVerificationError, match="could not be read"):
        Phase2ArtifactStore(invalid_new_output, relocated, _fingerprint(relocated))
    assert not invalid_new_output.exists()

    manifest_before = reopened.manifest_path.read_bytes()
    pipeline_before = reopened.pipeline_config_path.read_bytes()
    stages_before = reopened.stage_result_stream_path.read_bytes()
    with pytest.raises(Phase2ArtifactVerificationError, match="could not be read"):
        Phase2ArtifactStore(output_directory, relocated, _fingerprint(relocated), overwrite=True)
    assert reopened.manifest_path.read_bytes() == manifest_before
    assert reopened.pipeline_config_path.read_bytes() == pipeline_before
    assert reopened.stage_result_stream_path.read_bytes() == stages_before
