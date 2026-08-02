# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for the benchmark-grade WP17 noisy Krotov stage."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest

from benchmarks.state_preparation.constants import BALLARIN_NOISE_ID, STANDARD_NOISE_IDS
from benchmarks.state_preparation.noise import (
    FIXED_RATE_NOISE_DEFINITION_VERSION,
    HISTORICAL_FIXED_RATE_NOISE_ID,
    HistoricalFixedRateNoiseProvider,
    ScaledStandardNoiseProvider,
)
from benchmarks.state_preparation.phase2.noisy_krotov import (
    FixedRateNoisyKrotovStageAdapter,
    NoisyKrotovCircuitBinding,
    NoisyKrotovResumeState,
    NoisyKrotovStageExecution,
    NoisyKrotovStageFailure,
    execute_fixed_rate_krotov_stage,
    translate_fixed_rate_krotov_stage,
)
from benchmarks.state_preparation.phase2.pipeline import (
    CheckpointValidationConfig,
    TrainingStageConfig,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovFixedMapEnsemble,
    KrotovOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    train_krotov_state_preparation_batch,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_TOPOLOGY_ID = "wp17_toy_d1"
_TARGET = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
_INITIAL_THETA = np.array([0.23, -0.41, 0.17], dtype=np.float64)


def _circuit(*, include_fixed_gate: bool = False) -> ParameterizedCircuit:
    """Build a small exact logical state-preparation circuit.

    Returns:
        A two-qubit circuit with three trainable parameters.
    """
    gates = []
    if include_fixed_gate:
        gates.append(ParameterizedGate("h", (0,), noise_enabled=True))
    gates.extend([
        ParameterizedGate("rx", (0,), param_index=0, logical_gate_id="rx_0"),
        ParameterizedGate("ry", (1,), param_index=1, logical_gate_id="ry_1"),
        ParameterizedGate("rzz", (0, 1), param_index=2, logical_gate_id="rzz_0_1"),
    ])
    return ParameterizedCircuit(2, gates, num_params=3)


def _validation(
    *,
    enabled: bool = False,
    sampling_policy: str = "crn_fixed",
    refresh_interval: int | None = None,
    cadence: int = 1,
) -> CheckpointValidationConfig:
    """Build a canonical disabled or enabled checkpoint policy.

    Returns:
        The requested checkpoint-validation configuration.
    """
    if not enabled:
        return CheckpointValidationConfig.disabled()
    return CheckpointValidationConfig(
        noise_id="depolarizing_1s_all",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=100.0,
        tjm_dt=1.0,
        trajectory_count=3,
        seed=333,
        sampling_policy=cast(
            "Literal['none', 'resampled', 'crn_fixed', 'crn_refresh']",
            sampling_policy,
        ),
        ensemble_refresh_interval=refresh_interval,
        cadence=cadence,
        selection_rule="best_validation_fidelity",
        tie_breaker="earliest_iteration",
    )


def _stage(
    *,
    noise_id: str = "depolarizing_1s_all",
    scale: float = 100.0,
    dt: float = 1.0,
    trajectories: int = 3,
    training_seed: int = 222,
    optimizer_seed: int = 111,
    iterations: int = 4,
    update: str = "independent",
    sampling_policy: str = "crn_fixed",
    refresh_interval: int | None = None,
    validation: CheckpointValidationConfig | None = None,
) -> TrainingStageConfig:
    """Build a fully resolved WP17 training-stage fixture.

    Returns:
        A valid noisy or noiseless stage configuration.
    """
    noiseless = noise_id == "noiseless"
    return TrainingStageConfig(
        stage_index=2,
        stage_id="noisy_fine_tune",
        stage_kind="optimize",
        input_topology_id=_TOPOLOGY_ID,
        output_topology_id=_TOPOLOGY_ID,
        input_parameter_count=3,
        output_parameter_count=3,
        parameter_transfer_rule="copy",
        initialization_seed=None,
        optimizer_id="krotov",
        optimizer_hyperparameters={"learning_rate": 0.025, "schedule": "constant", "decay": 0.0},
        optimizer_seed=optimizer_seed,
        iteration_budget=iterations,
        training_noise_id=noise_id,
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=None if noiseless else scale,
        tjm_dt=None if noiseless else dt,
        trajectory_count=0 if noiseless else trajectories,
        training_seed=None if noiseless else training_seed,
        trajectory_update=(None if noiseless else cast("Literal['independent', 'cross']", update)),
        sampling_policy=cast(
            "Literal['none', 'resampled', 'crn_fixed', 'crn_refresh']",
            "none" if noiseless else sampling_policy,
        ),
        crn_refresh_interval=None if noiseless else refresh_interval,
        checkpoint_validation=validation or CheckpointValidationConfig.disabled(),
        pruning_rule="none",
        pruning_threshold=None,
        max_bond_dimension=None,
        svd_threshold=0.0,
        truncation_mode="discarded_weight",
        min_bond_dimension=1,
    )


def _binding(*, include_fixed_gate: bool = False) -> NoisyKrotovCircuitBinding:
    """Bind the toy circuit to the frozen logical training placement.

    Returns:
        A validated circuit binding.
    """
    return NoisyKrotovCircuitBinding(_circuit(include_fixed_gate=include_fixed_gate), _TOPOLOGY_ID)


def _successful(
    stage: TrainingStageConfig,
    *,
    binding: NoisyKrotovCircuitBinding | None = None,
    theta: np.ndarray = _INITIAL_THETA,
    global_iteration_offset: int = 0,
    iteration_count: int | None = None,
    replay_training_ensembles: Sequence[KrotovFixedMapEnsemble] = (),
    replay_validation_ensembles: Sequence[KrotovFixedMapEnsemble] = (),
    resume_state: NoisyKrotovResumeState | None = None,
) -> NoisyKrotovStageExecution:
    """Execute a stage and narrow the result to success.

    Returns:
        The successful stage execution.
    """
    result = FixedRateNoisyKrotovStageAdapter.execute(
        stage,
        binding or _binding(),
        _TARGET,
        theta,
        global_iteration_offset=global_iteration_offset,
        iteration_count=iteration_count,
        replay_training_ensembles=replay_training_ensembles,
        replay_validation_ensembles=replay_validation_ensembles,
        resume_state=resume_state,
    )
    assert isinstance(result, NoisyKrotovStageExecution), getattr(result, "message", "")
    return result


@pytest.mark.parametrize("noise_id", STANDARD_NOISE_IDS)
def test_adapter_translates_all_ten_standard_profiles(noise_id: str) -> None:
    """Every canonical fixed-rate profile reaches the stage adapter."""
    stage = _stage(noise_id=noise_id, trajectories=1, iterations=1)
    translation = translate_fixed_rate_krotov_stage(stage, _binding())
    assert isinstance(translation.noise_provider, ScaledStandardNoiseProvider)
    assert translation.noise_provider.base_noise_id == noise_id
    assert translation.noise_provider.strength_scale == pytest.approx(100.0)
    assert translation.options.seed == stage.optimizer_seed
    assert translation.options.batch_step_size == pytest.approx(0.025)
    assert translation.tjm_options is not None
    assert translation.tjm_options.random_seed == stage.training_seed
    assert translation.tjm_options.noisy_gate_indices == (0, 1, 2)
    assert translation.tjm_options.dt == pytest.approx(1.0)
    execution = _successful(stage)
    assert len(execution.trace) == 2
    assert len(execution.training_ensembles) == 1


def test_historical_profile_is_exact_and_rejects_scaling_or_dt_changes() -> None:
    """The historical reproduction profile is frozen at scale and dt one."""
    stage = _stage(noise_id=HISTORICAL_FIXED_RATE_NOISE_ID, scale=1.0, dt=1.0)
    translation = translate_fixed_rate_krotov_stage(stage, _binding())
    assert isinstance(translation.noise_provider, HistoricalFixedRateNoiseProvider)

    with pytest.raises(ValueError, match=r"noise_strength_scale=1\.0"):
        translate_fixed_rate_krotov_stage(replace(stage, noise_strength_scale=2.0), _binding())
    with pytest.raises(ValueError, match=r"tjm_dt=1\.0"):
        translate_fixed_rate_krotov_stage(replace(stage, tjm_dt=0.5), _binding())


def test_zero_noise_execution_matches_phase_i_batch_krotov() -> None:
    """The additive adapter preserves exact Phase I noiseless Krotov behavior."""
    stage = _stage(noise_id="noiseless", iterations=3)
    binding = _binding()
    actual = _successful(stage, binding=binding)
    expected = train_krotov_state_preparation_batch(
        binding.circuit,
        _TARGET,
        initial_theta=_INITIAL_THETA,
        options=KrotovOptions(
            max_iterations=3,
            batch_step_size=0.025,
            batch_schedule="constant",
            batch_decay=0.0,
            seed=111,
            truncation=KrotovTruncation(),
        ),
    )
    np.testing.assert_allclose(actual.final_theta, expected.theta, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray([row.monitoring_fidelity for row in actual.trace], dtype=np.float64),
        np.asarray(expected.trace["fidelity"], dtype=np.float64),
        atol=1e-12,
        rtol=0.0,
    )
    assert actual.training_ensembles == ()


def test_fixed_crn_is_reproducible_and_optimizer_seed_does_not_change_maps() -> None:
    """Optimizer ordering is absent from the trajectory-map identity and draws."""
    first = _successful(_stage(optimizer_seed=111))
    repeated = _successful(_stage(optimizer_seed=111))
    changed_optimizer = _successful(_stage(optimizer_seed=999))

    np.testing.assert_array_equal(first.final_theta, repeated.final_theta)
    assert first.content_checksum == repeated.content_checksum
    assert len(first.training_ensembles) == 1
    assert first.training_ensembles[0].ensemble_id == changed_optimizer.training_ensembles[0].ensemble_id
    assert (
        first.training_ensembles[0].to_dict()["trajectory_maps"]
        == changed_optimizer.training_ensembles[0].to_dict()["trajectory_maps"]
    )
    assert first.training_ensembles[0].content_checksum != changed_optimizer.training_ensembles[0].content_checksum


@pytest.mark.parametrize(
    ("policy", "refresh_interval", "expected_ensembles"),
    [("crn_fixed", None, 1), ("resampled", None, 5), ("crn_refresh", 2, 3)],
)
def test_sampling_policy_uses_exact_ensemble_cadence(
    policy: str,
    refresh_interval: int | None,
    expected_ensembles: int,
) -> None:
    """Fixed, resampled, and refreshed policies use the sealed global schedule."""
    result = _successful(
        _stage(
            iterations=5,
            sampling_policy=policy,
            refresh_interval=refresh_interval,
        )
    )
    assert len(result.training_ensembles) == expected_ensembles
    assert [ensemble.ensemble_index for ensemble in result.training_ensembles] == list(range(expected_ensembles))


def test_fixed_crn_resume_replays_active_ensemble_and_matches_uninterrupted_run() -> None:
    """A split fixed-CRN stage keeps map and learning-rate schedule continuity."""
    stage = _stage(iterations=4)
    uninterrupted = _successful(stage)
    first_chunk = _successful(stage, iteration_count=2)
    second_chunk = _successful(
        stage,
        theta=first_chunk.final_theta,
        global_iteration_offset=2,
        replay_training_ensembles=first_chunk.training_ensembles,
        resume_state=first_chunk.resume_state,
    )
    np.testing.assert_allclose(second_chunk.final_theta, uninterrupted.final_theta, atol=1e-12, rtol=0.0)
    assert second_chunk.training_ensemble_checksums == first_chunk.training_ensemble_checksums
    assert [row.global_iteration for row in second_chunk.trace] == [2, 3, 4]
    assert (
        second_chunk.normalized_work["objective_evaluations"]
        == cast("int", uninterrupted.normalized_work["objective_evaluations"]) + 1
    )
    assert (
        second_chunk.normalized_work["training_trajectories"]
        == cast("int", uninterrupted.normalized_work["training_trajectories"]) + stage.trajectory_count
    )
    assert second_chunk.resume_state.to_dict()["content_checksum"] == second_chunk.resume_state.content_checksum

    failure = execute_fixed_rate_krotov_stage(
        stage,
        _binding(),
        _TARGET,
        first_chunk.final_theta,
        global_iteration_offset=2,
        resume_state=first_chunk.resume_state,
    )
    assert isinstance(failure, NoisyKrotovStageFailure)
    assert "active replay ensemble" in failure.message


def test_checkpoint_validation_has_a_disjoint_seed_domain_and_selects_a_checkpoint() -> None:
    """Checkpoint maps include iteration zero and stay disjoint from training."""
    stage = _stage(iterations=3, validation=_validation(enabled=True, cadence=1))
    result = _successful(stage)
    assert len(result.checkpoint_validation_ensembles) == 1
    training = result.training_ensembles[0]
    validation = result.checkpoint_validation_ensembles[0]
    assert training.resolved_seed == 222
    assert validation.resolved_seed == 333
    assert training.role == "training_trajectory"
    assert validation.role == "checkpoint_validation"
    assert training.ensemble_id != validation.ensemble_id
    assert all(row.checkpoint_validation_fidelity is not None for row in result.trace)
    assert result.selected_global_iteration in {0, 1, 2, 3}
    assert result.selected_checkpoint_validation_fidelity is not None
    assert result.checkpoint_selection is not None


def test_validation_resume_preserves_an_earlier_best_checkpoint() -> None:
    """A split run retains a best checkpoint that predates the resume boundary."""
    stage = replace(
        _stage(iterations=4, validation=_validation(enabled=True, cadence=1)),
        optimizer_hyperparameters={"learning_rate": 20.0, "schedule": "constant", "decay": 0.0},
    )
    uninterrupted = _successful(stage)
    first_chunk = _successful(stage, iteration_count=2)
    selection = first_chunk.checkpoint_selection
    assert selection is not None
    assert selection.global_iteration == 1
    detached_theta = selection.theta
    detached_theta[:] = 99.0
    assert not np.allclose(selection.theta, 99.0)

    resumed = _successful(
        stage,
        theta=first_chunk.final_theta,
        global_iteration_offset=2,
        replay_training_ensembles=first_chunk.training_ensembles,
        replay_validation_ensembles=first_chunk.checkpoint_validation_ensembles,
        resume_state=first_chunk.resume_state,
    )
    np.testing.assert_allclose(resumed.final_theta, uninterrupted.final_theta, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(resumed.selected_theta, uninterrupted.selected_theta, atol=1e-12, rtol=0.0)
    assert resumed.selected_global_iteration == uninterrupted.selected_global_iteration == 1
    assert resumed.selected_checkpoint_validation_fidelity == pytest.approx(
        uninterrupted.selected_checkpoint_validation_fidelity
    )

    missing_selection = execute_fixed_rate_krotov_stage(
        stage,
        _binding(),
        _TARGET,
        first_chunk.final_theta,
        global_iteration_offset=2,
        replay_training_ensembles=first_chunk.training_ensembles,
        replay_validation_ensembles=first_chunk.checkpoint_validation_ensembles,
        resume_state=replace(first_chunk.resume_state, checkpoint_selection=None),
    )
    assert isinstance(missing_selection, NoisyKrotovStageFailure)
    assert "prior best checkpoint selection" in missing_selection.message


def test_resume_state_rejects_a_foreign_target_or_initial_state() -> None:
    """Resume provenance seals both operands of the optimized fidelity objective."""
    stage = _stage(iterations=3, validation=_validation(enabled=True, cadence=1))
    first_chunk = _successful(stage, iteration_count=1)
    foreign_target = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    wrong_target = execute_fixed_rate_krotov_stage(
        stage,
        _binding(),
        foreign_target,
        first_chunk.final_theta,
        global_iteration_offset=1,
        replay_training_ensembles=first_chunk.training_ensembles,
        replay_validation_ensembles=first_chunk.checkpoint_validation_ensembles,
        resume_state=first_chunk.resume_state,
    )
    assert isinstance(wrong_target, NoisyKrotovStageFailure)
    assert "target, or initial state" in wrong_target.message

    wrong_initial_state = execute_fixed_rate_krotov_stage(
        stage,
        _binding(),
        _TARGET,
        first_chunk.final_theta,
        initial_state=MPS(2, state="x+"),
        global_iteration_offset=1,
        replay_training_ensembles=first_chunk.training_ensembles,
        replay_validation_ensembles=first_chunk.checkpoint_validation_ensembles,
        resume_state=first_chunk.resume_state,
    )
    assert isinstance(wrong_initial_state, NoisyKrotovStageFailure)
    assert "target, or initial state" in wrong_initial_state.message


def test_cross_update_is_never_labelled_as_an_objective_gradient() -> None:
    """Cross dense-sum diagnostics stay separate from monitoring fidelity."""
    result = _successful(_stage(update="cross", iterations=2))
    for index, row in enumerate(result.trace[1:], start=1):
        assert row.update_signal_kind == "cross_dense_sum_update"
        assert row.gradient_norm is None
        assert row.cross_dense_sum_norm is not None
        assert row.update_signal_norm == pytest.approx(row.cross_dense_sum_norm)
        assert row.monitoring_loss == pytest.approx(1.0 - row.monitoring_fidelity)
        assert row.cross_trajectory_pairings == 27
        assert row.cumulative_cross_trajectory_pairings == 27 * index
    assert result.normalized_work["gradient_evaluations"] == 0
    assert result.cross_trajectory_pairings == 54

    first_chunk = _successful(_stage(update="cross", iterations=2), iteration_count=1)
    resumed = _successful(
        _stage(update="cross", iterations=2),
        theta=first_chunk.final_theta,
        global_iteration_offset=1,
        replay_training_ensembles=first_chunk.training_ensembles,
        resume_state=first_chunk.resume_state,
    )
    assert resumed.trace[0].cumulative_cross_trajectory_pairings == 27
    assert resumed.cross_trajectory_pairings == 54


def test_truncated_pathwise_update_is_not_overlabelled_as_a_gradient() -> None:
    """A finite bond cap turns the signal into an explicitly approximate update."""
    stage = replace(_stage(iterations=1), max_bond_dimension=1)
    result = _successful(stage)
    row = result.trace[1]
    assert row.update_signal_kind == "independent_pathwise_update"
    assert row.gradient_norm is None
    assert row.cross_dense_sum_norm is None
    assert row.update_signal_norm > 0.0


def test_logical_placement_excludes_fixed_and_native_gates() -> None:
    """Only logical parameterized gates are eligible for WP17 training noise."""
    binding = _binding(include_fixed_gate=True)
    assert binding.noisy_gate_indices == (1, 2, 3)
    result = _successful(_stage(iterations=1), binding=binding)
    maps = result.training_ensembles[0].replay_maps()
    assert all(trajectory[0].channel_id is None for trajectory in maps)
    assert all(trajectory[0].operators == () for trajectory in maps)

    native = _circuit()
    native.gates[0].native_gate_id = "native_rx"
    with pytest.raises(ValueError, match="compiled native"):
        NoisyKrotovCircuitBinding(native, _TOPOLOGY_ID)

    long_range = ParameterizedCircuit(
        3,
        [ParameterizedGate("rzz", (0, 2), param_index=0)],
        num_params=1,
    )
    with pytest.raises(ValueError, match="nearest-neighbor"):
        NoisyKrotovCircuitBinding(long_range, "long_range")


def test_circuit_binding_defensively_detaches_source_and_public_copies() -> None:
    """Caller mutations cannot alter a sealed binding or subsequent execution."""
    source = _circuit()
    binding = NoisyKrotovCircuitBinding(source, _TOPOLOGY_ID)
    stage = _stage(iterations=1, trajectories=1)
    checksum = binding.content_checksum
    payload = binding.to_dict()
    expected_execution = _successful(stage, binding=binding)

    source.gates[0].angle_offset = 0.75
    source.gates[1].noise_enabled = False
    exposed = binding.circuit
    exposed.gates[0].angle_offset = -1.25
    exposed.gates.clear()

    assert binding.content_checksum == checksum
    assert binding.to_dict() == payload
    assert binding.circuit.gates[0].angle_offset == pytest.approx(0.0)
    assert binding.circuit.gates[1].noise_enabled is True
    actual_execution = _successful(stage, binding=binding)
    np.testing.assert_array_equal(actual_execution.final_theta, expected_execution.final_theta)
    assert actual_execution.content_checksum == expected_execution.content_checksum


def test_work_ledger_counts_sampling_updates_monitoring_and_gate_applications() -> None:
    """The frozen normalized-work convention matches actual core executions."""
    result = _successful(_stage(iterations=2, trajectories=2))
    assert dict(result.normalized_work) == {
        "objective_evaluations": 5,
        "gradient_evaluations": 2,
        "training_trajectories": 12,
        "checkpoint_validation_trajectories": 0,
        "test_trajectories": 0,
        "trajectory_gate_applications": 36,
    }
    assert result.trace[-1].cumulative_work.to_dict() == dict(result.normalized_work)


def test_adapter_has_no_final_test_input_and_repeated_training_is_isolated() -> None:
    """Final-test settings have no route into the WP17 training call graph."""
    signature = inspect.signature(FixedRateNoisyKrotovStageAdapter.execute)
    assert "evaluation_config" not in signature.parameters
    assert "test_config" not in signature.parameters
    first = _successful(_stage(iterations=2))
    second = _successful(_stage(iterations=2))
    assert first.selected_parameter_checksum == second.selected_parameter_checksum
    assert first.training_ensemble_checksums == second.training_ensemble_checksums


def test_unsupported_training_profiles_become_structured_failures_without_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ballarin and unknown profiles fail before provider or map construction."""
    stage = _stage(iterations=1)
    # Defense-in-depth test: bypass the frozen constructor to emulate a forged
    # or deserialized object that was not validated by WP16.
    object.__setattr__(stage, "training_noise_id", BALLARIN_NOISE_ID)  # ruff: ignore[unnecessary-dunder-call]

    def forbidden_provider(*_args: object, **_kwargs: object) -> None:
        pytest.fail("provider construction must not be reached")

    monkeypatch.setattr(
        "benchmarks.state_preparation.phase2.noisy_krotov.create_scaled_standard_noise_provider",
        forbidden_provider,
    )
    result = execute_fixed_rate_krotov_stage(stage, _binding(), _TARGET, _INITIAL_THETA)
    assert isinstance(result, NoisyKrotovStageFailure)
    assert result.phase == "validation"
    assert result.exception_type == "ValueError"
    assert "evaluation-only" in result.message
    assert result.partial_work["objective_evaluations"] == 0
    assert result.to_dict()["content_checksum"] == result.content_checksum


def test_result_vectors_are_defensively_detached() -> None:
    """Callers cannot mutate stored initial, final, or selected parameters."""
    result = _successful(_stage(iterations=1))
    final = result.final_theta
    selected = result.selected_theta
    final[:] = 99.0
    selected[:] = -99.0
    assert not np.allclose(result.final_theta, 99.0)
    assert not np.allclose(result.selected_theta, -99.0)
