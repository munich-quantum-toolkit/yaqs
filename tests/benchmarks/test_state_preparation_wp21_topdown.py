# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused pipeline-path tests for the WP21 top-down orchestrator."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.phase2.artifacts import (
    Phase2ArtifactStore,
    StageExecutionEvidence,
)
from benchmarks.state_preparation.phase2.layerwise_bmpd import build_layerwise_bmpd_crn_v2_template
from benchmarks.state_preparation.phase2.noisy_krotov import (
    NoisyKrotovCircuitBinding,
    NoisyKrotovStageExecution,
    NoisyKrotovStageFailure,
)
from benchmarks.state_preparation.phase2.pruning import (
    TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
    TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID,
    TOPDOWN_MAGNITUDE_METHOD_ID,
    TOPDOWN_RANDOM_METHOD_ID,
    PruningRoundResult,
    PruningStagePolicy,
    PruningStageSpec,
    run_pruning_round,
)
from benchmarks.state_preparation.phase2.topdown_pruning import (
    TopDownPruningPathExecution,
    TopDownPruningStageExecution,
    TopDownPruningStageRunner,
    build_topdown_impact_iterative_template,
    build_topdown_impact_one_shot_template,
    build_topdown_magnitude_template,
    build_topdown_random_template,
    topdown_pruning_stage_evidence,
    topdown_reachable_resource_strata,
)
from benchmarks.state_preparation.phase2.wp20_resources import (
    InfeasibleResourceBudget,
    ResourceBudget,
    SelectedResourceStratum,
    WP20WorkLedger,
)
from mqt.yaqs.optimization import KrotovFixedMapEnsemble, ParameterizedCircuit, ParameterizedGate
from tests.benchmarks.test_state_preparation_phase2_pipeline import _screening_target_manifest
from tests.benchmarks.test_state_preparation_wp18_artifacts import (
    _fingerprint,
    _screening_materialized_targets,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from benchmarks.state_preparation.phase2.pipeline import (
        TrainingPipelineConfig,
        TrainingPipelineTemplate,
        TrainingStageConfig,
    )
    from benchmarks.state_preparation.phase2.targets import MaterializedTarget


def _magnitude_round() -> PruningRoundResult:
    """Build one genuine compiler-counted entangler pruning round.

    Returns:
        The checksum-sealed pruning round used by path-accounting tests.
    """
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rxx", (0, 1), param_index=0, logical_gate_id="first-entangler"),
            ParameterizedGate("rzz", (0, 1), param_index=1, logical_gate_id="second-entangler"),
            ParameterizedGate("ry", (0,), param_index=2, logical_gate_id="retained-local"),
        ],
        num_params=3,
    )
    binding = NoisyKrotovCircuitBinding(circuit, "topdown_path_input")
    policy = PruningStagePolicy(
        pruning_unit="compiled_entangler_group",
        scoring_objective_kind="none",
        removal_schedule="fixed_count",
        removal_count=1,
        removal_fraction=None,
        relax_after_round=False,
    )
    return run_pruning_round(
        binding,
        np.array([0.1, 0.5, 0.2], dtype=np.float64),
        PruningStageSpec(
            method_id=TOPDOWN_MAGNITUDE_METHOD_ID,
            score_rule="magnitude",
            policy=policy,
            random_seed=None,
        ),
        round_index=0,
        output_topology_id="topdown_path_output",
    )


def _resolve(
    template: TrainingPipelineTemplate,
    *,
    suffix: str,
) -> tuple[TrainingPipelineConfig, MaterializedTarget]:
    """Resolve a WP21 template against one authorized q6 target.

    Returns:
        The target-bound pipeline and its exact materialized target.
    """
    manifest = _screening_target_manifest()
    target = _screening_materialized_targets()[0]
    spec = next(item for item in manifest.instances if item.target_instance_id == target.target_instance_id)
    pipeline = template.resolve(
        target_namespace="phase2",
        target_manifest=manifest,
        target_instance_id=spec.target_instance_id,
        target_population_manifest_checksum=manifest.content_checksum,
        target_instance_spec_checksum=spec.content_checksum,
        target_family_id=spec.family_id,
        target_stratum_id=spec.stratum_id,
        qubit_count=spec.qubit_count,
        optimization_block_id=f"wp21_topdown_{suffix}",
        optimization_seed=321,
        data_role="screening_selection",
    )
    return pipeline, target


def _adapt(
    runner: TopDownPruningStageRunner,
    stage: TrainingStageConfig,
    outcome: StageExecutionEvidence | NoisyKrotovStageExecution,
    predecessor: NDArray[np.float64] | None,
) -> StageExecutionEvidence:
    """Adapt a supported runner outcome to publishable stage evidence.

    Returns:
        The exact optimizer-independent stage record.
    """
    if isinstance(outcome, StageExecutionEvidence):
        return outcome
    return StageExecutionEvidence.from_noisy_krotov(
        stage,
        outcome,
        source_parameters=predecessor,
        circuit_statistics=runner.circuit_statistics(stage),
    )


def _run_and_publish(
    store: Phase2ArtifactStore,
    runner: TopDownPruningStageRunner,
    stage: TrainingStageConfig,
    predecessor: NDArray[np.float64] | None,
) -> StageExecutionEvidence:
    """Execute and persist one exact stage boundary.

    Returns:
        The published stage evidence.
    """
    outcome = runner(stage, predecessor)
    assert not isinstance(outcome, NoisyKrotovStageFailure)
    evidence = _adapt(runner, stage, outcome, predecessor)
    store.publish_stage(evidence, wall_time_seconds=0.001, peak_memory_bytes=1)
    return evidence


def _assert_paths_equal(left: TopDownPruningPathExecution, right: TopDownPruningPathExecution) -> None:
    """Require complete round, work, runtime, and reachable-stratum equality.

    Args:
        left: Live path after publishing its current prefix.
        right: Path reconstructed after reopening the same artifacts.
    """
    assert [item.to_dict() for item in left.rounds] == [item.to_dict() for item in right.rounds]
    assert left.root_prefix_work.to_dict() == right.root_prefix_work.to_dict()
    assert [item.to_dict() for item in left.post_round_work] == [item.to_dict() for item in right.post_round_work]
    assert [item.to_dict() for item in left.reachable_strata] == [item.to_dict() for item in right.reachable_strata]


def test_four_named_templates_freeze_distinct_pruning_methods() -> None:
    """Named builders preserve score and iteration distinctions."""
    templates = (
        build_topdown_random_template(deep_depth=1, pretrain_iterations=1),
        build_topdown_magnitude_template(deep_depth=1, pretrain_iterations=1),
        build_topdown_impact_one_shot_template(deep_depth=1, pretrain_iterations=1),
        build_topdown_impact_iterative_template(
            deep_depth=1,
            pretrain_iterations=1,
            relaxation_iterations=1,
        ),
    )
    assert [template.method_id for template in templates] == [
        TOPDOWN_RANDOM_METHOD_ID,
        TOPDOWN_MAGNITUDE_METHOD_ID,
        TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID,
        TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
    ]
    assert [template.stages[1].stage_policy["pruning_rule"] for template in templates] == [
        "random",
        "magnitude",
        "impact_one_shot",
        "impact_iterative",
    ]
    assert [len(template.stages) for template in templates] == [2, 2, 2, 4]


def test_iterative_template_alternates_pruning_and_relaxation() -> None:
    """Every nonterminal impact round has one active relaxation boundary."""
    template = build_topdown_impact_iterative_template(
        deep_depth=1,
        round_count=3,
        pretrain_iterations=1,
        relaxation_iterations=1,
    )
    assert [stage.stage_id for stage in template.stages] == [
        "deep_pretrain",
        "prune_round_1",
        "relax_round_1",
        "prune_round_2",
        "relax_round_2",
        "prune_round_3",
    ]


def test_iterative_template_rejects_single_round_degeneration() -> None:
    """The iterative builder cannot produce a one-shot impact stage graph."""
    with pytest.raises(ValueError, match="at least two pruning rounds"):
        build_topdown_impact_iterative_template(
            deep_depth=1,
            pretrain_iterations=1,
            round_count=1,
        )


@pytest.mark.parametrize(
    ("noisy", "expected_map_count"),
    [(False, 0), (True, 1)],
)
def test_genuine_impact_stage_seals_noiseless_or_one_ensemble_crn_evidence(
    *,
    noisy: bool,
    expected_map_count: int,
) -> None:
    """Impact evidence binds its target, input circuit, and exact score maps."""
    template = build_topdown_impact_one_shot_template(
        deep_depth=1,
        pretrain_iterations=1,
        scoring_objective_kind=("fixed_map_sample_average_fidelity" if noisy else "noiseless_fidelity"),
        scoring_trajectory_count=1 if noisy else 0,
    )
    pipeline, target = _resolve(template, suffix=f"impact_{'noisy' if noisy else 'noiseless'}")
    runner = TopDownPruningStageRunner(pipeline, target)
    pretrain = runner(pipeline.stages[0], None)
    assert isinstance(pretrain, NoisyKrotovStageExecution)
    pruning = runner(pipeline.stages[1], pretrain.selected_theta)
    assert isinstance(pruning, StageExecutionEvidence)
    execution = TopDownPruningStageExecution.from_dict(pruning.training_summary["pruning_execution_document"])

    assert execution.objective_binding is not None
    assert len(pruning.training_ensembles) == expected_map_count
    assert len(execution.training_ensemble_checksums) == expected_map_count
    assert (execution.provider_checksum is not None) is noisy
    assert pruning.map_circuit_binding_checksum == execution.round.input_circuit_binding.content_checksum
    assert pruning.circuit_binding_checksum == execution.round.output_circuit_binding.content_checksum
    assert pruning.map_circuit_binding_checksum != pruning.circuit_binding_checksum
    assert pruning.objective_binding == execution.objective_binding
    assert pruning.normalized_work["gradient_evaluations"] == 1
    if noisy:
        assert cast("int", pruning.normalized_work["training_trajectories"]) > 1
        ensemble = pruning.training_ensembles[0]
        assert ensemble.circuit_checksum == execution.round.input_circuit_binding.content_checksum
        assert ensemble.content_checksum == execution.training_ensemble_checksums[0]
        incomplete = KrotovFixedMapEnsemble(
            role=ensemble.role,
            resolved_seed=ensemble.resolved_seed,
            stage_index=ensemble.stage_index,
            stage_id=ensemble.stage_id,
            stage_configuration_checksum=ensemble.stage_configuration_checksum,
            circuit_checksum=ensemble.circuit_checksum,
            provider_checksum=ensemble.provider_checksum,
            ensemble_index=ensemble.ensemble_index,
            refresh_index=ensemble.refresh_index,
            global_iteration_start=ensemble.global_iteration_start,
            trajectory_maps=[maps[:1] for maps in ensemble.replay_maps()],
        )
        forged_execution = replace(
            execution,
            training_ensemble_checksums=(incomplete.content_checksum,),
        )
        with pytest.raises(ValueError, match="cover every input-circuit gate"):
            topdown_pruning_stage_evidence(
                pipeline.stages[1],
                forged_execution,
                (incomplete,),
            )
    else:
        assert pruning.normalized_work["training_trajectories"] == 0


def test_iterative_resume_restores_prune_and_relax_boundaries_with_exact_work(tmp_path: Path) -> None:
    """Reopening never replays a round and reconstructs cumulative path work."""
    pipeline, target = _resolve(
        build_topdown_impact_iterative_template(
            deep_depth=1,
            pretrain_iterations=1,
            round_count=2,
            relaxation_iterations=1,
        ),
        suffix="iterative_resume",
    )
    fingerprint = _fingerprint(pipeline)
    store = Phase2ArtifactStore(tmp_path, pipeline, fingerprint)
    runner = TopDownPruningStageRunner(pipeline, target, store)

    pretrain = _run_and_publish(store, runner, pipeline.stages[0], None)
    store = Phase2ArtifactStore(tmp_path, pipeline, fingerprint, resume=True)
    runner = TopDownPruningStageRunner(pipeline, target, store)
    assert runner.pruning_path is None

    first_prune = _run_and_publish(store, runner, pipeline.stages[1], pretrain.selected_parameters)
    live_after_prune = runner.pruning_path
    assert live_after_prune is not None
    store = Phase2ArtifactStore(tmp_path, pipeline, fingerprint, resume=True)
    runner = TopDownPruningStageRunner(pipeline, target, store)
    restored_after_prune = runner.pruning_path
    assert restored_after_prune is not None
    assert len(restored_after_prune.rounds) == 1
    _assert_paths_equal(live_after_prune, restored_after_prune)

    relaxation = _run_and_publish(store, runner, pipeline.stages[2], first_prune.selected_parameters)
    live_after_relax = runner.pruning_path
    assert live_after_relax is not None
    assert live_after_relax.post_round_work[0].backward_circuit_evaluations > 0
    store = Phase2ArtifactStore(tmp_path, pipeline, fingerprint, resume=True)
    runner = TopDownPruningStageRunner(pipeline, target, store)
    restored_after_relax = runner.pruning_path
    assert restored_after_relax is not None
    _assert_paths_equal(live_after_relax, restored_after_relax)

    second_prune = _run_and_publish(store, runner, pipeline.stages[3], relaxation.selected_parameters)
    assert second_prune.selected_parameters.size == pipeline.stages[3].output_parameter_count
    live_complete = runner.pruning_path
    assert live_complete is not None
    assert len(live_complete.rounds) == 2
    store = Phase2ArtifactStore(tmp_path, pipeline, fingerprint, resume=True)
    restored_complete = TopDownPruningStageRunner(pipeline, target, store).pruning_path
    assert restored_complete is not None
    assert len(restored_complete.rounds) == 2
    _assert_paths_equal(live_complete, restored_complete)
    assert store.pipeline_result is not None


def test_optional_fixed_crn_finetune_is_counted_after_pruning(tmp_path: Path) -> None:
    """Noisy fine-tuning remains a separate map-bound post-round treatment."""
    pipeline, target = _resolve(
        build_topdown_magnitude_template(
            deep_depth=1,
            pretrain_iterations=1,
            fine_tune_mode="fixed_crn",
            fine_tune_iterations=1,
            fine_tune_trajectory_count=1,
        ),
        suffix="noisy_finetune",
    )
    fingerprint = _fingerprint(pipeline)
    store = Phase2ArtifactStore(tmp_path, pipeline, fingerprint)
    runner = TopDownPruningStageRunner(pipeline, target, store)
    pretrain = _run_and_publish(store, runner, pipeline.stages[0], None)
    pruning = _run_and_publish(store, runner, pipeline.stages[1], pretrain.selected_parameters)
    finetune = _run_and_publish(store, runner, pipeline.stages[2], pruning.selected_parameters)
    assert len(finetune.training_ensembles) == 1
    assert finetune.provider_checksum is not None
    assert finetune.stage.stage_id == "final_finetune"
    live_path = runner.pruning_path
    assert live_path is not None
    assert live_path.post_round_work[0].training_trajectories > 0
    assert live_path.reachable_strata[-1].normalized_compute > live_path.reachable_strata[0].normalized_compute

    reopened_store = Phase2ArtifactStore(tmp_path, pipeline, fingerprint, resume=True)
    restored_path = TopDownPruningStageRunner(pipeline, target, reopened_store).pruning_path
    assert restored_path is not None
    _assert_paths_equal(live_path, restored_path)


def test_fixed_crn_finetune_matches_corrected_bottom_up_scientific_treatment() -> None:
    """Top-down pruning reuses the corrected bottom-up fine-tune treatment exactly."""
    training_trajectories = 3
    validation_trajectories = 5
    bottom_up = build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=training_trajectories,
        checkpoint_validation_trajectory_count=validation_trajectories,
    ).stages[-1]
    top_down = build_topdown_magnitude_template(
        fine_tune_mode="fixed_crn",
        fine_tune_trajectory_count=training_trajectories,
        checkpoint_validation_trajectory_count=validation_trajectories,
    ).stages[-1]
    structural_fields = {
        "stage_index",
        "input_topology_id",
        "output_topology_id",
        "input_parameter_count",
        "output_parameter_count",
    }

    assert {key: value for key, value in top_down.stage_policy.items() if key not in structural_fields} == {
        key: value for key, value in bottom_up.stage_policy.items() if key not in structural_fields
    }
    assert set(top_down.seed_bindings) == set(bottom_up.seed_bindings)
    assert {key: value is None for key, value in top_down.seed_bindings.items()} == {
        key: value is None for key, value in bottom_up.seed_bindings.items()
    }


def test_reachable_path_requires_and_accumulates_exact_prefix_and_post_round_work() -> None:
    """Resource strata cannot silently omit pretraining or relaxation work."""
    pruning_round = _magnitude_round()
    root_work = WP20WorkLedger(forward_circuit_evaluations=5)
    relaxation_work = WP20WorkLedger(
        forward_circuit_evaluations=7,
        backward_circuit_evaluations=3,
    )
    path = TopDownPruningPathExecution(
        rounds=(pruning_round,),
        root_prefix_work=root_work,
        post_round_work=(relaxation_work,),
    )

    assert [item.normalized_compute for item in path.reachable_strata] == [5.0, 15.0]
    selected = path.select(
        ResourceBudget(
            native_two_qubit_gate_cap_per_chain_edge=1,
            normalized_compute_cap=15.0,
        )
    )
    assert isinstance(selected, SelectedResourceStratum)
    assert selected.selected.stratum_id == "topdown_round_1"
    assert selected.selected.circuit_resources.native_two_qubit_gates == 1

    infeasible = path.select(
        ResourceBudget(
            native_two_qubit_gate_cap_per_chain_edge=0,
            normalized_compute_cap=15.0,
        )
    )
    assert isinstance(infeasible, InfeasibleResourceBudget)
    assert len(infeasible.attempted_strata) == 2


def test_reachable_resource_projection_rejects_missing_round_work() -> None:
    """Callers must explicitly account for every post-pruning training stage."""
    with pytest.raises(TypeError, match="one exact WP20 ledger per attempted round"):
        topdown_reachable_resource_strata(
            (_magnitude_round(),),
            root_prefix_work=WP20WorkLedger(),
            post_round_work=(),
        )


@pytest.mark.parametrize(
    ("round_index", "message"),
    [(1, "same registered method"), (99, "contiguous from zero")],
)
def test_public_resource_projection_rejects_mixed_or_skipped_paths(
    round_index: int,
    message: str,
) -> None:
    """The standalone selector enforces the same path identity as its wrapper."""
    first = _magnitude_round()
    policy = PruningStagePolicy(
        pruning_unit="gate",
        scoring_objective_kind="none",
        removal_schedule="fixed_count",
        removal_count=1,
        removal_fraction=None,
        relax_after_round=False,
    )
    second = run_pruning_round(
        first.output_circuit_binding,
        first.output_theta,
        PruningStageSpec(
            method_id=TOPDOWN_RANDOM_METHOD_ID,
            score_rule="random",
            policy=policy,
            random_seed=17,
        ),
        round_index=round_index,
        output_topology_id="topdown_invalid_mixed_path",
    )
    with pytest.raises(ValueError, match=message):
        topdown_reachable_resource_strata(
            (first, second),
            root_prefix_work=WP20WorkLedger(),
            post_round_work=(WP20WorkLedger(), WP20WorkLedger()),
        )
