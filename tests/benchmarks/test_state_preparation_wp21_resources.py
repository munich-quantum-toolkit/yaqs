# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Training-randomness and reachable-resource tests for WP21 top-down methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation.phase2.artifacts import StageExecutionEvidence
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.noisy_krotov import NoisyKrotovStageExecution
from benchmarks.state_preparation.phase2.protocol import load_initial_preregistration
from benchmarks.state_preparation.phase2.pruning import (
    TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID,
    TOPDOWN_METHOD_IDS,
)
from benchmarks.state_preparation.phase2.targets import (
    authorize_target_materialization,
    build_target_population_config,
    materialize_target_population,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.topdown_pruning import (
    TopDownPruningStageRunner,
    build_topdown_magnitude_template,
    build_topdown_pruning_template,
)
from benchmarks.state_preparation.phase2.wp20_resources import (
    InfeasibleResourceBudget,
    PairedBlockIdentity,
    ReachableResourceStratum,
    ResourceBudget,
    SelectedResourceStratum,
    TrainingRandomnessRecord,
    TrainingRandomnessStageEvidence,
    WP20WorkLedger,
    measure_circuit_resources,
    resource_selection_outcome_from_dict,
    select_reachable_resource_stratum,
)
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate
from tests.benchmarks.test_state_preparation_phase2_pipeline import _screening_target_manifest

if TYPE_CHECKING:
    from benchmarks.state_preparation.phase2.pipeline import (
        TrainingPipelineConfig,
        TrainingPipelineTemplate,
    )
    from benchmarks.state_preparation.phase2.targets import (
        MaterializedTarget,
        TargetPopulationManifest,
    )


_TARGET_MASTER_ENTROPY = bytes(reversed(range(32)))


def _checksum(label: str) -> str:
    """Return a deterministic valid checksum for one focused identity."""
    return canonical_checksum({"wp21_resource_test": label})


@pytest.fixture(scope="module")
def target_case() -> tuple[MaterializedTarget, TargetPopulationManifest]:
    """Materialize one authorized q6 screening target.

    Returns:
        The target and its seed-bearing population manifest.
    """
    preregistration = load_initial_preregistration()
    population = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(_TARGET_MASTER_ENTROPY),
        population_scope="primary_q6",
    )
    manifest = _screening_target_manifest()
    authorization = authorize_target_materialization(
        preregistration,
        population,
        manifest,
        _TARGET_MASTER_ENTROPY,
    )
    target = materialize_target_population(
        population,
        preregistration,
        manifest,
        _TARGET_MASTER_ENTROPY,
        authorization,
    ).targets[0]
    return target, manifest


def _resolve_template(
    template: TrainingPipelineTemplate,
    target_case: tuple[MaterializedTarget, TargetPopulationManifest],
    *,
    suffix: str,
) -> TrainingPipelineConfig:
    """Resolve one top-down template for the focused target.

    Returns:
        The exact target/optimization-bound pipeline.
    """
    target, manifest = target_case
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
        optimization_block_id=f"wp21_resources_{suffix}",
        optimization_seed=123,
        data_role="screening_selection",
    )


def _randomness_record(pipeline: TrainingPipelineConfig) -> TrainingRandomnessRecord:
    """Build complete stage-schedule evidence for one resolved pipeline.

    Returns:
        A strict method-wide randomness record.
    """
    stages = tuple(
        TrainingRandomnessStageEvidence(
            stage=stage,
            execution_checksum=_checksum(f"{pipeline.method_id}-execution-{stage.stage_index}"),
            training_ensemble_checksums=(
                ()
                if stage.trajectory_count == 0
                else (_checksum(f"{pipeline.method_id}-training-map-{stage.stage_index}"),)
            ),
            checkpoint_validation_ensemble_checksums=(
                ()
                if not stage.checkpoint_validation.enabled
                else (_checksum(f"{pipeline.method_id}-validation-map-{stage.stage_index}"),)
            ),
        )
        for stage in pipeline.stages
    )
    return TrainingRandomnessRecord(
        paired_block_checksum=_checksum(f"{pipeline.method_id}-paired-block"),
        method_id=pipeline.method_id,
        training_id=pipeline.training_id,
        pipeline_configuration_checksum=pipeline.configuration_checksum,
        stages=stages,
    )


def test_zero_iteration_noisy_impact_requires_exactly_one_fixed_crn_ensemble(
    target_case: tuple[MaterializedTarget, TargetPopulationManifest],
) -> None:
    """A scoring round consumes one map despite having no optimizer iteration."""
    pipeline = _resolve_template(
        build_topdown_pruning_template(
            TOPDOWN_IMPACT_ONE_SHOT_METHOD_ID,
            deep_depth=1,
            pretrain_iterations=1,
            pruning_unit="parameter",
            removal_count=1,
            scoring_objective_kind="fixed_map_sample_average_fidelity",
            scoring_trajectory_count=2,
        ),
        target_case,
        suffix="noisy_impact",
    )
    stage = pipeline.stages[1]
    assert stage.stage_kind == "prune"
    assert stage.iteration_budget == 0
    assert stage.sampling_policy == "crn_fixed"
    ensemble_checksum = _checksum("noisy-impact-map")

    evidence = TrainingRandomnessStageEvidence(
        stage=stage,
        execution_checksum=_checksum("noisy-impact-execution"),
        training_ensemble_checksums=(ensemble_checksum,),
    )
    assert evidence.training_ensemble_checksums == (ensemble_checksum,)
    assert TrainingRandomnessStageEvidence.from_dict(evidence.to_dict()) == evidence
    for incomplete in ((), (ensemble_checksum, _checksum("surplus-impact-map"))):
        with pytest.raises(ValueError, match="incomplete for the sealed stage"):
            TrainingRandomnessStageEvidence(
                stage=stage,
                execution_checksum=_checksum("noisy-impact-execution"),
                training_ensemble_checksums=incomplete,
            )


@pytest.mark.parametrize("method_id", sorted(TOPDOWN_METHOD_IDS))
@pytest.mark.parametrize("noisy_finetune", [False, True])
def test_all_topdown_methods_accept_explicit_noiseless_or_noisy_training_treatments(
    target_case: tuple[MaterializedTarget, TargetPopulationManifest],
    method_id: str,
    *,
    noisy_finetune: bool,
) -> None:
    """Pruning identity is independent of the optional fine-tuning treatment."""
    template = build_topdown_pruning_template(
        method_id,
        deep_depth=1,
        pretrain_iterations=1,
        pruning_unit="parameter",
        removal_count=1,
        fine_tune_mode="fixed_crn" if noisy_finetune else "none",
        fine_tune_iterations=1,
        fine_tune_trajectory_count=1 if noisy_finetune else 0,
    )
    pipeline = _resolve_template(
        template,
        target_case,
        suffix=f"{method_id}_{'noisy' if noisy_finetune else 'noiseless'}",
    )
    record = _randomness_record(pipeline)

    assert record.method_id == method_id
    assert record.training_noise_active is noisy_finetune
    assert bool(record.training_ensemble_checksums) is noisy_finetune
    assert TrainingRandomnessRecord.from_dict(record.to_dict()) == record


@pytest.fixture(scope="module")
def magnitude_execution_case(
    target_case: tuple[MaterializedTarget, TargetPopulationManifest],
) -> tuple[TrainingPipelineConfig, tuple[StageExecutionEvidence, StageExecutionEvidence]]:
    """Execute one small genuine pretrain-and-prune pipeline.

    Returns:
        The resolved pipeline and its two artifact-neutral stage records.
    """
    target, _manifest = target_case
    pipeline = _resolve_template(
        build_topdown_magnitude_template(
            deep_depth=1,
            pretrain_iterations=1,
            pruning_unit="parameter",
            removal_count=1,
        ),
        target_case,
        suffix="execution_binding",
    )
    runner = TopDownPruningStageRunner(pipeline, target)
    pretrain_execution = runner(pipeline.stages[0], None)
    assert isinstance(pretrain_execution, NoisyKrotovStageExecution)
    pretrain_evidence = StageExecutionEvidence.from_noisy_krotov(
        pipeline.stages[0],
        pretrain_execution,
        source_parameters=None,
        circuit_statistics=runner.circuit_statistics(pipeline.stages[0]),
    )
    pruning_evidence = runner(pipeline.stages[1], pretrain_evidence.selected_parameters)
    assert isinstance(pruning_evidence, StageExecutionEvidence)
    return pipeline, (pretrain_evidence, pruning_evidence)


def test_randomness_record_binds_the_exact_pruning_execution_checksum(
    magnitude_execution_case: tuple[
        TrainingPipelineConfig,
        tuple[StageExecutionEvidence, StageExecutionEvidence],
    ],
) -> None:
    """Method-wide evidence derives the pruning checksum from validated execution."""
    pipeline, evidence = magnitude_execution_case
    block = PairedBlockIdentity(
        target_instance_id=pipeline.target_instance_id,
        target_manifest_checksum=pipeline.target_population_manifest_checksum,
        target_spec_checksum=pipeline.target_instance_spec_checksum,
        optimization_block_id=pipeline.optimization_block_id,
        optimization_seed=pipeline.optimization_seed,
        test_noise_id="depolarizing_1s_all",
        test_protocol_checksum=_checksum("fresh-test-protocol"),
        resource_stratum_id=pipeline.template.resource_stratum_id,
    )
    record = TrainingRandomnessRecord.from_stage_evidence(block, pipeline, evidence)
    pruning_checksum = cast("str", evidence[1].training_summary["pruning_execution_checksum"])

    assert record.stages[1].execution_checksum == pruning_checksum
    assert record.source_execution_checksums == (
        cast("str", evidence[0].training_summary["adapter_execution_checksum"]),
        pruning_checksum,
    )
    assert TrainingRandomnessRecord.from_dict(record.to_dict()) == record


def _stratum(stratum_id: str, native_two_qubit_count: int, work: int) -> ReachableResourceStratum:
    """Build one observed q2 resource stratum.

    Returns:
        Mechanically counted circuit and work evidence.
    """
    circuit = ParameterizedCircuit(
        num_qubits=2,
        gates=[
            ParameterizedGate("rzz", (0, 1), param_index=index, logical_gate_id=f"rzz-{index}")
            for index in range(native_two_qubit_count)
        ],
        num_params=native_two_qubit_count,
    )
    return ReachableResourceStratum(
        stratum_id=stratum_id,
        circuit_resources=measure_circuit_resources(circuit),
        work=WP20WorkLedger(forward_circuit_evaluations=work),
    )


def test_nonmonotonic_topdown_attempts_select_largest_observed_count_under_budget() -> None:
    """Selection searches observed strata instead of assuming monotonic pruning."""
    attempts = (
        _stratum("topdown-round-0", 6, 1),
        _stratum("topdown-round-1", 2, 2),
        _stratum("topdown-round-2", 4, 3),
        _stratum("topdown-round-3", 3, 4),
        _stratum("topdown-round-4", 5, 5),
    )
    budget = ResourceBudget(
        native_two_qubit_gate_cap_per_chain_edge=4,
        normalized_compute_cap=10.0,
    )
    outcome = select_reachable_resource_stratum(attempts, budget)

    assert isinstance(outcome, SelectedResourceStratum)
    assert outcome.selected.stratum_id == "topdown-round-2"
    assert outcome.selected.circuit_resources.native_two_qubit_gates == 4
    assert outcome.exact_native_match


def test_unreachable_topdown_budget_returns_typed_infeasibility() -> None:
    """No attempted round under both caps yields a structured negative result."""
    attempts = (
        _stratum("topdown-round-0", 3, 2),
        _stratum("topdown-round-1", 1, 3),
    )
    outcome = select_reachable_resource_stratum(
        attempts,
        ResourceBudget(
            native_two_qubit_gate_cap_per_chain_edge=0,
            normalized_compute_cap=1.0,
        ),
    )

    assert isinstance(outcome, InfeasibleResourceBudget)
    assert outcome.status == "infeasible"
    assert outcome.reason == "no_reachable_stratum_within_joint_caps"
    assert resource_selection_outcome_from_dict(outcome.to_dict()) == outcome
