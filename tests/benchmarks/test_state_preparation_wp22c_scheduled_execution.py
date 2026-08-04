# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Golden schedule, restart, and multistart tests for WP22C."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.phase2.binding_catalog import ExecutableScopedBinding
from benchmarks.state_preparation.phase2.canonical import canonical_checksum, canonical_json
from benchmarks.state_preparation.phase2.competitor_optimizers import (
    ParameterShiftAdamConfig,
    SPSAConfig,
    build_parameter_shift_adam_layerwise_template,
)
from benchmarks.state_preparation.phase2.execution_bindings import (
    BindingResourcePolicy,
    ControlledTrainingStage,
    ExecutionBudget,
    ExecutionImplementationArtifact,
    QubitTreatmentProjection,
    ScopedImplementationBinding,
)
from benchmarks.state_preparation.phase2.execution_protocol import (
    CHECKPOINT_VALIDATION_UPDATES,
    FreshEvaluationPolicy,
    OperatorGrowthExecutionSpec,
    PilotDiagnosticPolicy,
)
from benchmarks.state_preparation.phase2.fair_controls import build_fixed_depth_bmpd_crn_template
from benchmarks.state_preparation.phase2.implementation_catalog import (
    ExecutableImplementationEntry,
    RepositoryRunnerAdapter,
)
from benchmarks.state_preparation.phase2.operator_growth import (
    CandidateGradient,
    OperatorGrowthSpec,
    PoolOperator,
    build_projector_operator_pool,
)
from benchmarks.state_preparation.phase2.scheduled_execution import (
    AdamOptimizerPayload,
    KrotovOptimizerPayload,
    KrotovScheduledUpdateAdapter,
    MultistartWorkEvidence,
    NormalizedComputeCapError,
    OperatorGrowthAdamScheduledUpdateAdapter,
    OperatorGrowthOptimizerPayload,
    OperatorGrowthSegmentedObjectiveEvidence,
    OperatorGrowthSegmentedObjectiveExecutor,
    OperatorGrowthSegmentedObjectiveRequest,
    OperatorGrowthSegmentedObjectiveResult,
    OperatorGrowthSegmentedSnapshot,
    OperatorGrowthSelectionRequest,
    OperatorGrowthSelectionResult,
    OptimizerInitialization,
    ParameterShiftAdamScheduledUpdateAdapter,
    ScheduledExecutionProgram,
    ScheduledExecutionSnapshot,
    ScheduledJobSeedSet,
    ScheduledOptimizerState,
    ScheduledTrainingGradientResult,
    ScheduledTrainingObjectiveResult,
    ScheduledValidationResult,
    SPSAOptimizerPayload,
    SPSAScheduledUpdateAdapter,
    compile_development_schedule,
    compile_frozen_schedule_trace,
    execute_operator_growth_segmented_program,
    execute_scheduled_program,
    initialize_scheduled_execution,
)
from benchmarks.state_preparation.phase2.training_schedules import (
    FROZEN_TRAINING_POLICY_IDS,
    CheckpointValidationPolicy,
    FrozenTrainingPolicyUniverse,
    LimitedMultistartPlan,
    NoiselessPretrainNoisyFinetune,
    NoiseStrengthContinuation,
    TrajectoryCountCurriculum,
    TrajectoryCountStep,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization import (
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    forward_tjm_trajectory,
    noisy_state_preparation_contribution,
    noisy_state_preparation_loss,
    state_preparation_contribution,
    state_preparation_loss,
)

if TYPE_CHECKING:
    from benchmarks.state_preparation.phase2.implementation_catalog import (
        CatalogPreset,
    )
    from benchmarks.state_preparation.phase2.scheduled_execution import (
        ScheduledTrainingGradientRequest,
        ScheduledTrainingObjectiveRequest,
        ScheduledValidationRequest,
    )
    from benchmarks.state_preparation.phase2.training_schedules import (
        TrainingStrategySchedule,
    )
    from mqt.yaqs.optimization import (
        KrotovNoiseMap,
    )

_SEED_SET = ScheduledJobSeedSet(123)


def _schedule(schedule_id: str) -> TrainingStrategySchedule:
    """Return one exact member of the frozen policy universe."""
    return next(
        schedule for schedule in FrozenTrainingPolicyUniverse.frozen().schedules if schedule.schedule_id == schedule_id
    )


def _binding(
    *,
    operator_growth: bool = False,
    method_id: str = "fixed_depth_bmpd_crn",
    operator_growth_compute_cap: float = 1.0,
) -> ExecutableScopedBinding:
    """Build one valid q6 pilot or operator-growth screen binding.

    Returns:
        The exact synthetic scoped binding closed to its repository runner.
    """
    schedule = _schedule("direct_matched_fixed_crn")
    method_id = "adapt_style_state_preparation" if operator_growth else method_id
    screen = operator_growth or method_id == "parameter_shift_adam_layerwise"
    preset: CatalogPreset = "paper-screen" if screen else "paper-pilot"
    candidate_checksum = canonical_checksum({"wp22c_candidate": method_id})
    operator_spec = OperatorGrowthExecutionSpec.for_screening(256) if operator_growth else None
    implementation_payload = (
        operator_spec
        if operator_spec is not None
        else (
            build_parameter_shift_adam_layerwise_template(
                training_trajectory_count=8,
                checkpoint_validation_trajectory_count=256,
            )
            if method_id == "parameter_shift_adam_layerwise"
            else build_fixed_depth_bmpd_crn_template(
                iteration_budget=200,
                training_trajectory_count=8,
                checkpoint_validation_trajectory_count=256,
            )
        )
    )
    artifact = ExecutionImplementationArtifact(
        artifact_id=f"wp22c_{preset}_{method_id}",
        preset=preset,
        publication_method_id=method_id,
        implementation_kind="operator_growth" if operator_growth else "phase2_pipeline",
        implementation_method_id=method_id,
        target_scope_id="primary_q6",
        strategy_schedule_checksum=schedule.content_checksum,
        implementation_payload=implementation_payload,
    )
    evaluation_policies = (
        FreshEvaluationPolicy.checkpoint_validation(),
        FreshEvaluationPolicy.screening(256) if screen else FreshEvaluationPolicy.primary_q6_pilot(),
    )
    binding = ScopedImplementationBinding(
        binding_id=f"wp22c_binding_{preset}_{method_id}",
        preset=preset,
        publication_candidate_schema_version="test.wp22c_candidate.v1",
        publication_candidate_checksum=candidate_checksum,
        publication_method_id=method_id,
        target_scope_id="primary_q6",
        qubit_count=6,
        manifest_data_role="screening_selection" if screen else "development",
        execution_data_role="screening_selection" if screen else "development",
        implementation_artifact=artifact,
        strategy_schedule=schedule,
        controlled_stage=ControlledTrainingStage.complete_schedule(schedule, artifact),
        evaluation_policies=evaluation_policies,
        pilot_diagnostic_policy=None if screen else PilotDiagnosticPolicy.primary_q6(),
        execution_budget=ExecutionBudget(
            total_update_count=200,
            maximum_training_trajectory_count=8,
            checkpoint_validation_trajectory_count=256,
            multistart_count=1,
            normalized_compute_cap=(
                operator_growth_compute_cap if operator_growth else 1_000_000.0 if screen else None
            ),
        ),
        resource_policy=BindingResourcePolicy(),
        treatment_projection=QubitTreatmentProjection(
            publication_candidate_checksum=candidate_checksum,
            publication_method_id=method_id,
            target_scope_id="primary_q6",
            primary_q6_implementation_checksum=artifact.content_checksum,
            inference_role="primary",
            screening_eligible=screen,
            promotion_eligible=screen,
        ),
        operator_growth_spec=operator_spec,
    )
    runner = RepositoryRunnerAdapter.for_artifact(artifact)
    entry = ExecutableImplementationEntry(
        preset=preset,
        publication_method_id=method_id,
        target_scope_id="primary_q6",
        strategy_schedule=schedule,
        implementation_artifact=artifact,
        runner_adapter=runner,
    )
    return ExecutableScopedBinding.close(binding, entry)


def _trace_checksum(program: ScheduledExecutionProgram) -> str:
    """Return a compact golden checksum of all schedule-varying policy fields."""
    return canonical_checksum({
        "schedule_id": program.schedule.schedule_id,
        "trace": [
            {
                "start": policy.start_index,
                "update": policy.update,
                "phase": policy.phase,
                "noise": policy.noise_strength_scale,
                "count": policy.trajectory_count,
                "epoch": policy.sampling_epoch,
                "retained": (
                    None if policy.training_membership is None else policy.training_membership.retained_member_count
                ),
                "members": (None if policy.training_membership is None else policy.training_membership.member_seeds),
                "component_counts": (
                    () if policy.mixture_allocation is None else policy.mixture_allocation.component_counts
                ),
                "checkpoint": policy.checkpoint_due,
            }
            for policy in program.update_policies
        ],
    })


_GOLDEN_TRACE_CHECKSUMS = {
    "direct_matched_fixed_crn": "sha256:0752b4d32b7fa0833e3ad24af93b99ca63e3fd66c879df8e8c4544c195dd0c16",
    "continuation_fixed_crn": "sha256:4c89d028b5bd3c3398db778bea73a242d631fd3c9354eac77bfbe9f5f66f11a4",
    "curriculum_fixed_crn": "sha256:cdc524041777a436ab6ce020c6ffcb7a2dd9154ede9fe74bb887fc1bf85baf88",
    "periodic_refresh_20": "sha256:573d55694227095f47a376282b7fb85adb2ed6fcc776ca8d243c317189c98895",
    "rolling_half_refresh_20": "sha256:f5be593b6058256f8752ff231cbfd2836de94a29f491004e8931ec86cf1818c1",
    "resampled_each_update": "sha256:9f173696398426820fcd2e2d3e0f1240350609acca5f196f3a300abcbde2184e",
    "frozen_half_depolarizing_half_dephasing": (
        "sha256:b78e0dca403216e06c59c4bde260aa9b04b97aaa8ebe39486658cdc68830453c"
    ),
    "limited_multistart_3": "sha256:36571dea6371b2b86511d603873e31b1f9054ddfd520b0364d0e14d922fed9dd",
    "direct_noiseless_control": "sha256:07353a88ffe9f9842d3bde6ffda5ced22241b66cb8329449580700b16a0d9458",
}


def test_every_frozen_schedule_compiles_to_its_golden_update_trace() -> None:
    """All continuation, sampling, mixture, phase, and multistart variants are exact."""
    programs = {
        schedule_id: compile_frozen_schedule_trace(_schedule(schedule_id), _SEED_SET)
        for schedule_id in FROZEN_TRAINING_POLICY_IDS
    }
    assert {schedule_id: _trace_checksum(program) for schedule_id, program in programs.items()} == (
        _GOLDEN_TRACE_CHECKSUMS
    )
    assert all(program.checkpoint_updates == CHECKPOINT_VALIDATION_UPDATES for program in programs.values())
    assert all(
        tuple(policy.update for policy in program.update_policies[:200] if policy.checkpoint_due)
        == CHECKPOINT_VALIDATION_UPDATES
        for program in programs.values()
    )
    assert (
        ScheduledExecutionProgram.from_json(programs["rolling_half_refresh_20"].to_json())
        == programs["rolling_half_refresh_20"]
    )


def test_golden_boundaries_membership_and_component_local_assignment() -> None:
    """Inclusive boundaries, refresh retention, and frozen-mixture partitions are literal."""
    continuation = compile_frozen_schedule_trace(_schedule("continuation_fixed_crn"), _SEED_SET)
    assert tuple(continuation.policy(0, update).noise_strength_scale for update in (0, 1, 49, 50, 199)) == (
        0.0,
        1.0 / 49.0,
        1.0,
        1.0,
        1.0,
    )
    curriculum = compile_frozen_schedule_trace(_schedule("curriculum_fixed_crn"), _SEED_SET)
    assert tuple(curriculum.policy(0, update).trajectory_count for update in (0, 49, 50, 99, 100, 199)) == (
        2,
        2,
        4,
        4,
        8,
        8,
    )
    direct = compile_frozen_schedule_trace(_schedule("direct_matched_fixed_crn"), _SEED_SET)
    direct_membership = direct.policy(0, 199).training_membership
    assert direct_membership is not None
    assert direct_membership.retained_member_count == 8
    periodic = compile_frozen_schedule_trace(_schedule("periodic_refresh_20"), _SEED_SET)
    assert periodic.policy(0, 19).sampling_epoch == 0
    assert periodic.policy(0, 20).sampling_epoch == 1
    periodic_membership = periodic.policy(0, 20).training_membership
    assert periodic_membership is not None
    assert periodic_membership.retained_member_count == 0
    rolling = compile_frozen_schedule_trace(_schedule("rolling_half_refresh_20"), _SEED_SET)
    rolling_membership = rolling.policy(0, 20).training_membership
    assert rolling_membership is not None
    assert rolling_membership.retained_member_count == 4
    resampled = compile_frozen_schedule_trace(_schedule("resampled_each_update"), _SEED_SET)
    assert resampled.policy(0, 20).sampling_epoch == 20
    resampled_membership = resampled.policy(0, 20).training_membership
    assert resampled_membership is not None
    assert resampled_membership.retained_member_count == 0

    mixture = compile_frozen_schedule_trace(
        _schedule("frozen_half_depolarizing_half_dephasing"),
        _SEED_SET,
    )
    first = mixture.policy(0, 0)
    second = mixture.policy(0, 1)
    assert first.mixture_allocation is not None
    assert first.training_membership is not None
    assert first.mixture_allocation.component_counts == (4, 4)
    assert tuple(seed for component in first.component_memberships for seed in component.member_seeds) == (
        first.training_membership.member_seeds
    )
    assert tuple(component.predecessor_checksum for component in first.component_memberships) == (None, None)
    assert tuple(component.predecessor_checksum for component in second.component_memberships) == tuple(
        component.content_checksum for component in first.component_memberships
    )
    noiseless = compile_frozen_schedule_trace(_schedule("direct_noiseless_control"), _SEED_SET)
    assert {policy.phase for policy in noiseless.update_policies} == {"noiseless_pretrain"}
    assert all(policy.training_membership is None for policy in noiseless.update_policies)


_CIRCUIT = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
_TARGET = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
_OBJECTIVE_CHECKSUM = canonical_checksum({"wp22c_objective": "one_qubit_plus_state"})
_TRUNCATION = KrotovTruncation()
_ADAM_CONFIG = ParameterShiftAdamConfig(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
_SPSA_CONFIG = SPSAConfig(a=0.1, stability_constant=10.0, alpha=0.602, c=0.1, gamma=0.101)


def _sample_scheduled_maps(
    request: ScheduledTrainingGradientRequest | ScheduledTrainingObjectiveRequest,
) -> tuple[NoiseModel, list[list[KrotovNoiseMap]]]:
    """Sample each declared trajectory member independently at its exact seed.

    Returns:
        The scaled noise model and member-ordered realized map lists.
    """
    membership = request.policy.training_membership
    assert membership is not None
    model = NoiseModel([
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": 0.08 * request.policy.noise_strength_scale,
        }
    ])
    parameters = np.asarray(request.parameters, dtype=np.float64)
    maps: list[list[KrotovNoiseMap]] = []
    for member_seed in membership.member_seeds:
        trajectory = forward_tjm_trajectory(
            _CIRCUIT,
            parameters,
            np.array([], dtype=np.float64),
            MPS(1),
            _TRUNCATION,
            model,
            KrotovTJMOptions(num_trajectories=1, random_seed=member_seed),
            np.random.Generator(np.random.PCG64(member_seed)),
        )
        maps.append(trajectory.noise_maps)
    return model, maps


class _RealKrotovGradient:
    """One-qubit repository Krotov contribution using scheduled member seeds."""

    def __init__(self, *, cross_trajectory: bool = False) -> None:
        self.cross_trajectory = cross_trajectory
        self.requests: list[ScheduledTrainingGradientRequest] = []

    def __call__(self, request: ScheduledTrainingGradientRequest) -> ScheduledTrainingGradientResult:
        """Evaluate and seal one noiseless or fixed-map noisy contribution.

        Returns:
            The contribution sealed to its exact training request.
        """
        self.requests.append(request)
        parameters = np.asarray(request.parameters, dtype=np.float64)
        if request.policy.trajectory_count == 0 or np.isclose(request.policy.noise_strength_scale, 0.0):
            contribution = state_preparation_contribution(
                _CIRCUIT,
                parameters,
                _TARGET,
                MPS(1),
                _TRUNCATION,
            )[0]
        else:
            model, maps = _sample_scheduled_maps(request)
            contribution = noisy_state_preparation_contribution(
                _CIRCUIT,
                parameters,
                _TARGET,
                model,
                KrotovTJMOptions(
                    num_trajectories=len(maps),
                    trajectory_update="cross" if self.cross_trajectory else "independent",
                ),
                MPS(1),
                _TRUNCATION,
                fixed_noise_maps=maps,
            )[0]
        return ScheduledTrainingGradientResult.for_request(request, tuple(float(value) for value in contribution))


class _RealObjective:
    """One-qubit repository infidelity executor for Adam, SPSA, and growth."""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, request: ScheduledTrainingObjectiveRequest) -> ScheduledTrainingObjectiveResult:
        """Evaluate one request using its scheduled fixed-map ensemble.

        Returns:
            The infidelity sealed to its exact training request.
        """
        self.call_count += 1
        parameters = np.asarray(request.parameters, dtype=np.float64)
        if request.policy.trajectory_count == 0 or np.isclose(request.policy.noise_strength_scale, 0.0):
            loss = state_preparation_loss(_CIRCUIT, parameters, _TARGET)
        else:
            model, maps = _sample_scheduled_maps(request)
            loss = noisy_state_preparation_loss(
                _CIRCUIT,
                parameters,
                _TARGET,
                model,
                KrotovTJMOptions(num_trajectories=len(maps)),
                fixed_noise_maps=maps,
            )
        return ScheduledTrainingObjectiveResult.for_request(request, loss)


def _real_validation(request: ScheduledValidationRequest) -> ScheduledValidationResult:
    """Evaluate bounded post-update one-qubit fidelity from recoverable parameters.

    Returns:
        The fidelity sealed to the validation-only request.
    """
    loss = state_preparation_loss(_CIRCUIT, np.asarray(request.parameter_artifact.parameters), _TARGET)
    return ScheduledValidationResult.for_request(request, min(1.0, max(0.0, 1.0 - loss)))


def _favored_validation(request: ScheduledValidationRequest) -> ScheduledValidationResult:
    """Exercise maximum, earliest-checkpoint, and lowest-start tie rules.

    Returns:
        A deterministic validation-only score.
    """
    favored = (request.start_index == 0 and request.update == 10) or (
        request.start_index in {1, 2} and request.update == 0
    )
    return ScheduledValidationResult.for_request(request, 0.9 if favored else 0.1)


def _krotov_payload(
    program: ScheduledExecutionProgram,
    start_index: int,
    parameter: float,
    *,
    learning_rate: float = 0.04,
) -> KrotovOptimizerPayload:
    """Build one development warm-start Krotov payload.

    Returns:
        The typed update-zero Krotov state.
    """
    initialization = OptimizerInitialization.warm_start(
        program.start_seed_bundles[start_index],
        (parameter,),
        source_checksum=canonical_checksum({"development_start": start_index}),
    )
    return KrotovOptimizerPayload.initialize(initialization, learning_rate=learning_rate)


def _initial_krotov_snapshot(program: ScheduledExecutionProgram) -> ScheduledExecutionSnapshot:
    """Initialize every start with a distinct real one-qubit parameter.

    Returns:
        The complete all-start update-zero snapshot.
    """
    return initialize_scheduled_execution(
        program,
        tuple(_krotov_payload(program, index, 0.11 + 0.03 * index) for index in range(program.start_count)),
    )


def _adapter(gradient: _RealKrotovGradient | None = None) -> KrotovScheduledUpdateAdapter:
    """Build the fixed-contract repository Krotov update adapter.

    Returns:
        The repository-owned Krotov update adapter.
    """
    return KrotovScheduledUpdateAdapter(_OBJECTIVE_CHECKSUM, gradient or _RealKrotovGradient())


def _reseal(mapping: dict[str, object]) -> None:
    """Recompute one test fixture's canonical content checksum in place."""
    mapping["content_checksum"] = canonical_checksum({
        key: value for key, value in mapping.items() if key != "content_checksum"
    })


def test_interrupted_resume_is_byte_identical_and_preserves_optimizer_state() -> None:
    """Real Krotov restart reproduces all receipts, work, state, and selection byte for byte."""
    program = compile_frozen_schedule_trace(_schedule("limited_multistart_3"), ScheduledJobSeedSet(9123))
    uninterrupted = execute_scheduled_program(
        program,
        _initial_krotov_snapshot(program),
        _adapter(),
        validation_executor=_favored_validation,
    )
    partial = execute_scheduled_program(
        program,
        _initial_krotov_snapshot(program),
        _adapter(),
        validation_executor=_favored_validation,
        stop_after_updates=217,
    )
    reloaded = ScheduledExecutionSnapshot.from_json(partial.to_json())
    resumed = execute_scheduled_program(
        program,
        reloaded,
        _adapter(),
        validation_executor=_favored_validation,
    )
    assert resumed.to_json() == uninterrupted.to_json()
    assert resumed.multistart_evidence is not None
    assert resumed.multistart_evidence.selected_start_index == 1
    assert resumed.multistart_evidence.selected_update == 0
    assert resumed.multistart_evidence.total_normalized_work == pytest.approx(25_728.0)
    assert tuple(
        cast("KrotovOptimizerPayload", state.optimizer_payload).completed_updates for state in resumed.states
    ) == (
        200,
        200,
        200,
    )
    assert ScheduledOptimizerState.from_json(resumed.states[0].to_json()) == resumed.states[0]
    assert MultistartWorkEvidence.from_dict(resumed.multistart_evidence.to_dict()) == resumed.multistart_evidence
    with pytest.raises(ValueError, match="exact ordered start universe"):
        MultistartWorkEvidence.from_states(program, resumed.states[:1])


@pytest.mark.parametrize("alias", ["request_checksum", "result_checksum"])
def test_resealed_historical_request_and_result_alias_tampering_is_rejected(alias: str) -> None:
    """Standalone restart decoding re-derives both checksum-shaped receipt aliases."""
    program = compile_frozen_schedule_trace(_schedule("direct_matched_fixed_crn"), ScheduledJobSeedSet(71))
    partial = execute_scheduled_program(
        program,
        _initial_krotov_snapshot(program),
        _adapter(),
        validation_executor=_real_validation,
        stop_after_updates=1,
    ).to_dict()
    states = cast("list[dict[str, object]]", partial["states"])
    receipts = cast("list[dict[str, object]]", states[0]["receipts"])
    receipts[0][alias] = canonical_checksum({"forged_alias": alias})
    _reseal(receipts[0])
    _reseal(states[0])
    _reseal(partial)
    with pytest.raises(ValueError, match="request, result, optimizer, or parameter alias"):
        ScheduledExecutionSnapshot.from_json(canonical_json(partial))


def test_resealed_historical_work_is_checked_against_adapter_before_resume_callback() -> None:
    """A self-consistent lower historical work claim cannot evade the current adapter quote."""
    program = compile_frozen_schedule_trace(_schedule("direct_matched_fixed_crn"), ScheduledJobSeedSet(72))
    partial = execute_scheduled_program(
        program,
        _initial_krotov_snapshot(program),
        _adapter(),
        validation_executor=_real_validation,
        stop_after_updates=1,
    ).to_dict()
    states = cast("list[dict[str, object]]", partial["states"])
    receipts = cast("list[dict[str, object]]", states[0]["receipts"])
    result = cast("dict[str, object]", receipts[0]["result"])
    result["normalized_work"] = 0.0
    _reseal(result)
    receipts[0]["result_checksum"] = result["content_checksum"]
    receipts[0]["normalized_work"] = 256.0
    _reseal(receipts[0])
    states[0]["total_normalized_work"] = 256.0
    _reseal(states[0])
    _reseal(partial)
    reloaded = ScheduledExecutionSnapshot.from_json(canonical_json(partial))
    gradient = _RealKrotovGradient()
    with pytest.raises(ValueError, match="currently bound repository adapter"):
        execute_scheduled_program(program, reloaded, _adapter(gradient), validation_executor=_real_validation)
    assert gradient.requests == []


def test_constant_schedule_matches_direct_fixed_seed_noisy_krotov_update() -> None:
    """One scheduled update equals direct fixed-map noisy Krotov at the same member seeds."""
    program = compile_frozen_schedule_trace(_schedule("direct_matched_fixed_crn"), ScheduledJobSeedSet(44))
    initial = _krotov_payload(program, 0, 0.23, learning_rate=0.05)
    snapshot = initialize_scheduled_execution(program, (initial,))
    gradient = _RealKrotovGradient()
    result = execute_scheduled_program(
        program,
        snapshot,
        _adapter(gradient),
        validation_executor=_real_validation,
        stop_after_updates=1,
    )
    request = gradient.requests[0]
    model, maps = _sample_scheduled_maps(request)
    direct_gradient = noisy_state_preparation_contribution(
        _CIRCUIT,
        np.array([0.23]),
        _TARGET,
        model,
        KrotovTJMOptions(num_trajectories=8),
        MPS(1),
        _TRUNCATION,
        fixed_noise_maps=maps,
    )[0]
    expected = 0.23 - 0.05 * float(direct_gradient[0])
    state = cast("KrotovOptimizerPayload", result.states[0].optimizer_payload)
    assert state.parameters[0] == pytest.approx(expected, abs=1e-14)
    assert request.policy.training_membership is not None
    assert not hasattr(request.policy, "checkpoint_due")
    assert result.states[0].receipts[0].validation_request is not None
    assert result.states[0].receipts[0].validation_request.parameter_artifact.parameters == state.parameters


def test_all_repository_optimizer_adapters_advance_real_one_qubit_objectives() -> None:
    """Krotov, Adam, SPSA, and growth Adam all perform genuine numerical updates."""
    program = compile_frozen_schedule_trace(_schedule("direct_matched_fixed_crn"), ScheduledJobSeedSet(81))
    bundle = program.start_seed_bundles[0]
    source = canonical_checksum({"one_qubit_prefix": 1})

    adam_initialization = OptimizerInitialization.warm_start(bundle, (0.23,), source_checksum=source)
    adam = AdamOptimizerPayload.initialize(adam_initialization, _ADAM_CONFIG)
    adam_result = execute_scheduled_program(
        program,
        initialize_scheduled_execution(program, (adam,)),
        ParameterShiftAdamScheduledUpdateAdapter(_OBJECTIVE_CHECKSUM, (1.0,), _RealObjective()),
        validation_executor=_real_validation,
        stop_after_updates=1,
    )
    adam_state = cast("AdamOptimizerPayload", adam_result.states[0].optimizer_payload)
    assert adam_state.completed_updates == 1
    assert adam_state.parameters != adam.parameters
    assert adam_state.first_moment != (0.0,)

    spsa_initialization = OptimizerInitialization.warm_start(bundle, (0.23,), source_checksum=source)
    spsa = SPSAOptimizerPayload.initialize(spsa_initialization, _SPSA_CONFIG)
    spsa_adapter = SPSAScheduledUpdateAdapter(_OBJECTIVE_CHECKSUM, _RealObjective())
    first_spsa = execute_scheduled_program(
        program,
        initialize_scheduled_execution(program, (spsa,)),
        spsa_adapter,
        validation_executor=_real_validation,
        stop_after_updates=1,
    )
    repeated_spsa = execute_scheduled_program(
        program,
        initialize_scheduled_execution(program, (spsa,)),
        SPSAScheduledUpdateAdapter(_OBJECTIVE_CHECKSUM, _RealObjective()),
        validation_executor=_real_validation,
        stop_after_updates=1,
    )
    spsa_state = cast("SPSAOptimizerPayload", first_spsa.states[0].optimizer_payload)
    assert spsa_state.completed_updates == 1
    assert spsa_state.parameters != spsa.parameters
    assert first_spsa.to_json() == repeated_spsa.to_json()

    pool = build_projector_operator_pool(1)
    spec = OperatorGrowthSpec.for_pool(pool, max_operators=1, reoptimization_steps=2)
    growth_initialization = OptimizerInitialization.warm_start(bundle, (0.23,), source_checksum=source)
    growth = OperatorGrowthOptimizerPayload.initialize(
        growth_initialization,
        spec,
        growth_step_index=1,
        selected_operator_ids=(pool.operators[0].operator_id,),
        structural_state_checksum=source,
        initial_objective=state_preparation_loss(_CIRCUIT, np.array([0.23]), _TARGET),
    )
    growth_result = execute_scheduled_program(
        program,
        initialize_scheduled_execution(program, (growth,)),
        OperatorGrowthAdamScheduledUpdateAdapter(_OBJECTIVE_CHECKSUM, _RealObjective()),
        validation_executor=_real_validation,
        stop_after_updates=1,
    )
    growth_state = cast("OperatorGrowthOptimizerPayload", growth_result.states[0].optimizer_payload)
    assert growth_state.completed_updates == 1
    assert growth_state.parameters != growth.parameters
    assert growth_state.first_moment != (0.0,)
    assert growth_state.best_objective <= growth.best_objective


def test_development_phase_boundary_runs_genuine_noiseless_then_noisy_updates() -> None:
    """The engine crosses the pretrain/fine-tune boundary without resetting optimizer state."""
    base = _schedule("continuation_fixed_crn")
    schedule = replace(
        base,
        schedule_id="wp22c_three_update_phase_test",
        noise_continuation=NoiseStrengthContinuation(
            start_update=1,
            end_update=2,
            start_strength_scale=0.0,
            target_strength_scale=1.0,
            interpolation="linear_clamped",
        ),
        trajectory_curriculum=TrajectoryCountCurriculum((TrajectoryCountStep(0, 0), TrajectoryCountStep(1, 2))),
        checkpoint_validation=CheckpointValidationPolicy(patience=None),
        phase_boundary=NoiselessPretrainNoisyFinetune(1, 2),
        multistart=LimitedMultistartPlan(1, 1),
    )
    program = compile_development_schedule(schedule, ScheduledJobSeedSet(91))
    gradient = _RealKrotovGradient()
    result = execute_scheduled_program(program, _initial_krotov_snapshot(program), _adapter(gradient))
    state = cast("KrotovOptimizerPayload", result.states[0].optimizer_payload)
    assert state.completed_updates == 3
    assert tuple(request.policy.phase for request in gradient.requests) == (
        "noiseless_pretrain",
        "noisy_finetune",
        "noisy_finetune",
    )
    assert tuple(request.policy.noise_strength_scale for request in gradient.requests) == (0.0, 0.0, 1.0)
    assert gradient.requests[0].policy.training_membership is None
    assert gradient.requests[2].policy.training_membership is not None
    assert all(
        receipt.request.optimizer_payload.completed_updates == receipt.update for receipt in result.states[0].receipts
    )


def test_binding_owned_hyperparameters_scales_and_cross_mode_reject_tampering() -> None:
    """Same-family state and adapter drift fail before any numerical callback."""

    class _KrotovAdapterSubclass(KrotovScheduledUpdateAdapter):
        pass

    executable = _binding()
    program = ScheduledExecutionProgram.compile(
        executable,
        executable.binding.strategy_schedule,
        ScheduledJobSeedSet(8),
    )
    initialization = OptimizerInitialization.normal(program.start_seed_bundles[0], 198, scale=0.05)
    correct = KrotovOptimizerPayload.initialize(
        initialization,
        learning_rate=0.2,
        learning_rate_schedule="exp",
        decay=0.01,
    )
    snapshot = initialize_scheduled_execution(program, (correct,))
    wrong_rate = replace(correct, learning_rate=0.21)
    with pytest.raises(ValueError, match="Krotov hyperparameters"):
        initialize_scheduled_execution(program, (wrong_rate,))
    with pytest.raises(ValueError, match="cross-trajectory adapter mode"):
        execute_scheduled_program(
            program,
            snapshot,
            KrotovScheduledUpdateAdapter(_OBJECTIVE_CHECKSUM, _RealKrotovGradient(), cross_trajectory=True),
            validation_executor=_real_validation,
        )
    with pytest.raises(TypeError, match="concrete repository scheduled-update adapter"):
        execute_scheduled_program(
            program,
            snapshot,
            _KrotovAdapterSubclass(_OBJECTIVE_CHECKSUM, _RealKrotovGradient()),
            validation_executor=_real_validation,
        )

    adam_executable = _binding(method_id="parameter_shift_adam_layerwise")
    adam_program = ScheduledExecutionProgram.compile(
        adam_executable,
        adam_executable.binding.strategy_schedule,
        ScheduledJobSeedSet(9),
    )
    adam_initialization = OptimizerInitialization.warm_start(
        adam_program.start_seed_bundles[0],
        (0.0,) * 198,
        source_checksum=canonical_checksum({"layerwise_prefix": "final_finetune"}),
    )
    adam = AdamOptimizerPayload.initialize(adam_initialization, _ADAM_CONFIG)
    adam_snapshot = initialize_scheduled_execution(adam_program, (adam,))
    objective = _RealObjective()
    with pytest.raises(ValueError, match="parameter-shift scales"):
        execute_scheduled_program(
            adam_program,
            adam_snapshot,
            ParameterShiftAdamScheduledUpdateAdapter(_OBJECTIVE_CHECKSUM, (1.0,) * 198, objective),
            validation_executor=_real_validation,
        )
    assert objective.call_count == 0


def test_job_wide_compute_cap_is_atomic_before_operator_growth_callback() -> None:
    """A complete quoted growth update is rejected before any objective call."""
    executable = _binding(operator_growth=True)
    program = ScheduledExecutionProgram.compile(
        executable,
        executable.binding.strategy_schedule,
        ScheduledJobSeedSet(101),
    )
    execution_spec = cast("OperatorGrowthExecutionSpec", executable.binding.operator_growth_spec)
    source = canonical_checksum({"operator_prefix": 1})
    initialization = OptimizerInitialization.warm_start(
        program.start_seed_bundles[0],
        (0.23,),
        source_checksum=source,
    )
    forged_prefix = OperatorGrowthOptimizerPayload.initialize(
        initialization,
        execution_spec.growth_spec,
        growth_step_index=1,
        selected_operator_ids=("forged_projector",),
        structural_state_checksum=source,
        initial_objective=0.5,
    )
    with pytest.raises(ValueError, match="complete selected prefix"):
        initialize_scheduled_execution(program, (forged_prefix,))
    payload = OperatorGrowthOptimizerPayload.initialize(
        initialization,
        execution_spec.growth_spec,
        growth_step_index=1,
        selected_operator_ids=(execution_spec.pool.operators[0].operator_id,),
        structural_state_checksum=source,
        initial_objective=0.5,
    )
    objective = _RealObjective()
    with pytest.raises(NormalizedComputeCapError) as caught:
        execute_scheduled_program(
            program,
            initialize_scheduled_execution(program, (payload,)),
            OperatorGrowthAdamScheduledUpdateAdapter(_OBJECTIVE_CHECKSUM, objective),
            validation_executor=_real_validation,
        )
    assert caught.value.completed_work == pytest.approx(0.0)
    assert objective.call_count == 0


def test_validation_requirement_schedule_substitution_and_binding_closure_fail_closed() -> None:
    """Validation absence, schedule substitution, and incomplete bindings abort before mutation."""
    executable = _binding()
    program = ScheduledExecutionProgram.compile(
        executable,
        executable.binding.strategy_schedule,
        ScheduledJobSeedSet(8),
    )
    initialization = OptimizerInitialization.normal(program.start_seed_bundles[0], 198, scale=0.05)
    payload = KrotovOptimizerPayload.initialize(
        initialization,
        learning_rate=0.2,
        learning_rate_schedule="exp",
        decay=0.01,
    )
    gradient = _RealKrotovGradient()
    with pytest.raises(ValueError, match="requires a validation executor"):
        execute_scheduled_program(program, initialize_scheduled_execution(program, (payload,)), _adapter(gradient))
    assert gradient.requests == []
    with pytest.raises(ValueError, match="not the binding's exact"):
        ScheduledExecutionProgram.compile(executable, _schedule("continuation_fixed_crn"), ScheduledJobSeedSet(8))
    with pytest.raises(TypeError, match="ExecutableScopedBinding"):
        ScheduledExecutionProgram.compile(
            cast("ExecutableScopedBinding", executable.binding),
            executable.binding.strategy_schedule,
            ScheduledJobSeedSet(8),
        )
    changed = replace(_schedule("direct_matched_fixed_crn"), schedule_id="unreviewed_schedule")
    with pytest.raises(ValueError, match="exact members"):
        compile_frozen_schedule_trace(changed, ScheduledJobSeedSet(8))


def test_binding_compilation_includes_pipeline_and_operator_reoptimization() -> None:
    """Pipeline and operator-growth programs retain their exact executable closure."""
    direct_binding = _binding()
    direct = ScheduledExecutionProgram.compile(
        direct_binding,
        direct_binding.binding.strategy_schedule,
        ScheduledJobSeedSet(6),
    )
    assert direct.controlled_stage_id == "direct_depth4_noisy_training"
    assert direct.executable_binding == direct_binding
    assert ScheduledExecutionProgram.from_json(direct.to_json()) == direct
    operator_binding = _binding(operator_growth=True)
    operator = ScheduledExecutionProgram.compile(
        operator_binding,
        operator_binding.binding.strategy_schedule,
        ScheduledJobSeedSet(6),
    )
    assert operator.controlled_stage_id == "operator_growth_reoptimization"
    assert operator.to_dict()["structural_stage_semantics"] == "outside_engine_independently_sealed"
    assert "confirmatory" not in operator.to_json()
    assert direct.checkpoint_updates == CHECKPOINT_VALIDATION_UPDATES
    assert operator.checkpoint_updates == (99, 199)


class _SegmentedQuadraticObjective:
    """Deterministic operator-aware quadratic for segmented protocol tests."""

    def __init__(self, program: ScheduledExecutionProgram) -> None:
        assert program.executable_binding is not None
        execution_spec = cast(
            "OperatorGrowthExecutionSpec",
            program.executable_binding.binding.operator_growth_spec,
        )
        self.targets = {
            operator.operator_id: 0.01 * (index + 1) for index, operator in enumerate(execution_spec.pool.operators)
        }
        self.requests: list[OperatorGrowthSegmentedObjectiveRequest] = []

    def __call__(
        self,
        request: OperatorGrowthSegmentedObjectiveRequest,
    ) -> OperatorGrowthSegmentedObjectiveResult:
        """Evaluate one exact request and retain it for role/membership assertions.

        Returns:
            The checksum-linked quadratic objective result.
        """
        self.requests.append(request)
        objective = sum(
            (parameter - self.targets[operator_id]) ** 2
            for parameter, operator_id in zip(
                request.parameters,
                request.selected_operator_ids,
                strict=True,
            )
        )
        return OperatorGrowthSegmentedObjectiveResult.for_request(request, objective)


class _ExactSegmentedSelection:
    """Test structural callback implementing the frozen pool-order selection rule."""

    def __init__(self, program: ScheduledExecutionProgram) -> None:
        assert program.executable_binding is not None
        execution_spec = cast(
            "OperatorGrowthExecutionSpec",
            program.executable_binding.binding.operator_growth_spec,
        )
        self.program = program
        self.pool = execution_spec.pool
        self.spec = execution_spec.growth_spec
        self.quote_count = 0
        self.requests: list[OperatorGrowthSelectionRequest] = []

    def _metadata(
        self,
        request: OperatorGrowthSelectionRequest,
    ) -> tuple[tuple[int, PoolOperator, bool], ...]:
        """Return remaining pool records with native-cap feasibility.

        Returns:
            Ordered pool index, operator, and feasibility triples.
        """
        selected = set(request.selected_operator_ids)
        edge_counts = [0] * (self.pool.num_qubits - 1)
        by_id = {operator.operator_id: operator for operator in self.pool.operators}
        for operator_id in request.selected_operator_ids:
            operator = by_id[operator_id]
            if len(operator.sites) == 2:
                edge_counts[operator.sites[0]] += operator.native_two_qubit_gates
        metadata: list[tuple[int, PoolOperator, bool]] = []
        for index, operator in enumerate(self.pool.operators):
            if operator.operator_id in selected:
                continue
            feasible = True
            if len(operator.sites) == 2 and self.spec.native_two_qubit_cap_per_edge is not None:
                feasible = (
                    edge_counts[operator.sites[0]] + operator.native_two_qubit_gates
                    <= self.spec.native_two_qubit_cap_per_edge
                )
            metadata.append((index, operator, feasible))
        return tuple(metadata)

    def quote_normalized_work(self, request: OperatorGrowthSelectionRequest) -> float:
        """Quote every feasible shift pair and the chosen appended-zero objective.

        Returns:
            Exact normalized structural-selection work.
        """
        self.quote_count += 1
        feasible = sum(item[2] for item in self._metadata(request))
        return float((2 * feasible + 1) * 8)

    def __call__(
        self,
        request: OperatorGrowthSelectionRequest,
        objective_executor: OperatorGrowthSegmentedObjectiveExecutor,
    ) -> OperatorGrowthSelectionResult:
        """Evaluate the full candidate universe and select the largest gradient.

        Returns:
            Complete structural-selection evidence.
        """
        self.requests.append(request)
        candidates: list[CandidateGradient] = []
        evidence: list[OperatorGrowthSegmentedObjectiveEvidence] = []
        for pool_index, operator, feasible in self._metadata(request):
            operator_id = operator.operator_id
            native_increment = operator.native_two_qubit_gates
            if not feasible:
                candidates.append(
                    CandidateGradient(
                        operator_id=operator_id,
                        pool_index=pool_index,
                        gradient=None,
                        absolute_gradient=None,
                        native_two_qubit_increment=native_increment,
                        native_cap_feasible=False,
                    )
                )
                continue
            plus_request = OperatorGrowthSegmentedObjectiveRequest(
                program_checksum=self.program.content_checksum,
                structural_state_checksum=request.content_checksum,
                selected_operator_ids=(*request.selected_operator_ids, operator_id),
                prefix_index=request.prefix_index,
                global_update=request.global_update_start,
                local_update=0,
                evaluation_stage="structural_selection",
                evaluation_kind="gradient_plus",
                parameter_index=request.prefix_index,
                parameters=(*request.parameters, np.pi / 2.0),
                policy=request.policy,
            )
            minus_request = replace(
                plus_request,
                evaluation_kind="gradient_minus",
                parameters=(*request.parameters, -np.pi / 2.0),
            )
            plus_result = objective_executor(plus_request)
            minus_result = objective_executor(minus_request)
            evidence.extend((
                OperatorGrowthSegmentedObjectiveEvidence(plus_request, plus_result),
                OperatorGrowthSegmentedObjectiveEvidence(minus_request, minus_result),
            ))
            gradient = 0.5 * (plus_result.objective - minus_result.objective)
            candidates.append(
                CandidateGradient(
                    operator_id=operator_id,
                    pool_index=pool_index,
                    gradient=gradient,
                    absolute_gradient=abs(gradient),
                    native_two_qubit_increment=native_increment,
                    native_cap_feasible=True,
                )
            )
        chosen = max(
            (candidate for candidate in candidates if candidate.native_cap_feasible),
            key=lambda candidate: cast("float", candidate.absolute_gradient),
        )
        baseline_request = OperatorGrowthSegmentedObjectiveRequest(
            program_checksum=self.program.content_checksum,
            structural_state_checksum=request.content_checksum,
            selected_operator_ids=(*request.selected_operator_ids, chosen.operator_id),
            prefix_index=request.prefix_index,
            global_update=request.global_update_start,
            local_update=0,
            evaluation_stage="structural_selection",
            evaluation_kind="post_update",
            parameter_index=request.prefix_index + 1,
            parameters=(*request.parameters, 0.0),
            policy=request.policy,
        )
        baseline_result = objective_executor(baseline_request)
        evidence.append(OperatorGrowthSegmentedObjectiveEvidence(baseline_request, baseline_result))
        return OperatorGrowthSelectionResult(
            request_checksum=request.content_checksum,
            candidate_gradients=tuple(candidates),
            selected_operator_id=chosen.operator_id,
            selected_gradient=cast("float", chosen.gradient),
            objective_before_reoptimization=baseline_result.objective,
            objective_evidence=tuple(evidence),
            normalized_work=float(len(evidence) * 8),
        )


class _PrefixValidation:
    """Deterministic equal-score prefix validator exercising earliest tie-break."""

    def __init__(self) -> None:
        self.requests: list[ScheduledValidationRequest] = []

    def __call__(self, request: ScheduledValidationRequest) -> ScheduledValidationResult:
        """Return the same score for both completed prefixes.

        Returns:
            The checksum-linked validation result.
        """
        self.requests.append(request)
        return ScheduledValidationResult.for_request(request, 0.75)


class _ZeroSegmentedObjective:
    """Degenerate objective exercising frozen two-prefix fail-closed semantics."""

    def __init__(self) -> None:
        self.requests: list[OperatorGrowthSegmentedObjectiveRequest] = []

    def __call__(
        self,
        request: OperatorGrowthSegmentedObjectiveRequest,
    ) -> OperatorGrowthSegmentedObjectiveResult:
        """Return zero for every pool candidate.

        Returns:
            A checksum-linked zero objective.
        """
        self.requests.append(request)
        return OperatorGrowthSegmentedObjectiveResult.for_request(request, 0.0)


def _segmented_program(cap: float) -> ScheduledExecutionProgram:
    """Compile the exact operator-growth program with a test compute cap.

    Returns:
        The complete q6 segmented execution program.
    """
    executable = _binding(operator_growth=True, operator_growth_compute_cap=cap)
    return ScheduledExecutionProgram.compile(
        executable,
        executable.binding.strategy_schedule,
        ScheduledJobSeedSet(701),
    )


def test_segmented_operator_growth_executes_two_exact_prefixes_and_selects_earliest_tie() -> None:
    """Two append/reset/100-update segments retain all work and prefix validation evidence."""
    program = _segmented_program(10_000.0)
    objective = _SegmentedQuadraticObjective(program)
    selector = _ExactSegmentedSelection(program)
    validation = _PrefixValidation()
    result = execute_operator_growth_segmented_program(
        program,
        OperatorGrowthSegmentedSnapshot.initialize(program),
        selector,
        objective,
        validation,
    )
    assert result.complete
    assert result.terminal_reason == "update_budget"
    assert len(result.transitions) == 2
    assert len(result.receipts) == 200
    assert tuple(item.request.update for item in result.prefix_validations) == (99, 199)
    assert tuple(item.request.membership.trajectory_count for item in result.prefix_validations) == (256, 256)
    assert result.selected_prefix_index == 0
    assert result.selected_operator_ids == result.transitions[0].selected_operator_ids
    assert result.selected_parameters == result.prefix_validations[0].request.parameter_artifact.parameters
    assert result.active_operator_ids == result.transitions[1].selected_operator_ids
    assert result.receipts[100].local_update == 0
    assert result.receipts[100].first_moment_before == (0.0, 0.0)
    assert result.receipts[100].second_moment_before == (0.0, 0.0)
    assert result.transitions[1].initial_parameters == (
        *result.prefix_validations[0].request.parameter_artifact.parameters,
        0.0,
    )
    assert tuple(len(item.result.candidate_gradients) for item in result.transitions) == (33, 32)
    assert (
        len({
            request.policy.training_membership.member_seeds
            for request in objective.requests
            if request.policy.training_membership is not None
        })
        == 1
    )
    assert tuple(request.update for request in validation.requests) == (99, 199)
    assert result.total_normalized_work == pytest.approx(7_968.0)
    assert 'final_test_access":"forbidden' in result.to_json()
    assert OperatorGrowthSegmentedSnapshot.from_json(result.to_json()) == result


def test_segmented_operator_growth_restart_is_byte_identical() -> None:
    """Canonical restart after an interior Adam update reproduces every byte."""
    program = _segmented_program(10_000.0)
    uninterrupted = execute_operator_growth_segmented_program(
        program,
        OperatorGrowthSegmentedSnapshot.initialize(program),
        _ExactSegmentedSelection(program),
        _SegmentedQuadraticObjective(program),
        _PrefixValidation(),
    )
    partial = execute_operator_growth_segmented_program(
        program,
        OperatorGrowthSegmentedSnapshot.initialize(program),
        _ExactSegmentedSelection(program),
        _SegmentedQuadraticObjective(program),
        _PrefixValidation(),
        stop_after_updates=73,
    )
    resumed = execute_operator_growth_segmented_program(
        program,
        OperatorGrowthSegmentedSnapshot.from_json(partial.to_json()),
        _ExactSegmentedSelection(program),
        _SegmentedQuadraticObjective(program),
        _PrefixValidation(),
    )
    assert resumed.to_json() == uninterrupted.to_json()


def test_segmented_prefix_start_cap_preflight_precedes_every_numerical_callback() -> None:
    """A cap allowing selection alone but not its first update leaves no unrecorded callback work."""
    program = _segmented_program(559.0)
    selector = _ExactSegmentedSelection(program)
    objective = _SegmentedQuadraticObjective(program)
    validation = _PrefixValidation()
    with pytest.raises(NormalizedComputeCapError) as caught:
        execute_operator_growth_segmented_program(
            program,
            OperatorGrowthSegmentedSnapshot.initialize(program),
            selector,
            objective,
            validation,
        )
    assert caught.value.prospective_update_work == pytest.approx(560.0)
    assert selector.quote_count == 1
    assert selector.requests == []
    assert objective.requests == []
    assert validation.requests == []


def test_segmented_prefix_validation_cap_is_checked_before_update_objectives() -> None:
    """The mandatory 256-member validation is included before the 100th update callback."""
    program = _segmented_program(3_191.0)
    partial = execute_operator_growth_segmented_program(
        program,
        OperatorGrowthSegmentedSnapshot.initialize(program),
        _ExactSegmentedSelection(program),
        _SegmentedQuadraticObjective(program),
        _PrefixValidation(),
        stop_after_updates=99,
    )
    assert partial.next_global_update == 99
    objective = _SegmentedQuadraticObjective(program)
    validation = _PrefixValidation()
    with pytest.raises(NormalizedComputeCapError) as caught:
        execute_operator_growth_segmented_program(
            program,
            OperatorGrowthSegmentedSnapshot.from_json(partial.to_json()),
            _ExactSegmentedSelection(program),
            objective,
            validation,
        )
    assert caught.value.completed_work == pytest.approx(2_912.0)
    assert caught.value.prospective_update_work == pytest.approx(280.0)
    assert objective.requests == []
    assert validation.requests == []


def test_segmented_schedule_fails_closed_on_structural_convergence_before_two_prefixes() -> None:
    """The fixed 2x100 paper schedule rejects, rather than silently truncates, a zero-gradient prefix."""
    program = _segmented_program(10_000.0)
    initial = OperatorGrowthSegmentedSnapshot.initialize(program)
    objective = _ZeroSegmentedObjective()
    with pytest.raises(ValueError, match="largest-gradient"):
        execute_operator_growth_segmented_program(
            program,
            initial,
            _ExactSegmentedSelection(program),
            objective,
            _PrefixValidation(),
        )
    assert len(objective.requests) == 67
    assert initial.next_global_update == 0
    assert initial.transitions == ()
