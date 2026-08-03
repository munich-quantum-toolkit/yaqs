# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reference and provenance tests for WP20 Adam and SPSA adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
import pytest

from benchmarks.state_preparation.noise import FIXED_RATE_NOISE_DEFINITION_VERSION
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.competitor_optimizers import (
    PARAMETER_SHIFT_POLICY_ID,
    SPSA_PERTURBATION_DISTRIBUTION_ID,
    CompetitorObjectiveRequest,
    CompetitorWorkBudget,
    FixedRateNoisyCompetitorObjective,
    ParameterShiftAdamStageAdapter,
    SPSAStageAdapter,
    build_parameter_shift_adam_fixed_template,
    build_parameter_shift_adam_layerwise_template,
    build_spsa_fixed_template,
    build_spsa_layerwise_template,
)
from benchmarks.state_preparation.phase2.legacy_targets import load_legacy_target_collection
from benchmarks.state_preparation.phase2.noisy_krotov import NoisyKrotovCircuitBinding
from benchmarks.state_preparation.phase2.pipeline import (
    CheckpointValidationConfig,
    TrainingPipelineTemplate,
    TrainingStageConfig,
)
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray


def _stage(
    optimizer_id: Literal["parameter_shift_adam", "spsa"],
    parameter_count: int,
    *,
    iteration_budget: int = 1,
) -> TrainingStageConfig:
    """Return a small resolved noisy competitor stage."""
    if optimizer_id == "parameter_shift_adam":
        hyperparameters = {
            "learning_rate": 0.1,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "parameter_shift": PARAMETER_SHIFT_POLICY_ID,
            "gradient_trajectory_count": 2,
            "sampling_policy": "crn_fixed",
        }
        sampling_policy = "crn_fixed"
    else:
        hyperparameters = {
            "a": 0.2,
            "A": 1.0,
            "alpha": 0.602,
            "c": 0.1,
            "gamma": 0.101,
            "perturbation_distribution": SPSA_PERTURBATION_DISTRIBUTION_ID,
            "gradient_trajectory_count": 2,
            "sampling_policy": "resampled",
        }
        sampling_policy = "resampled"
    return TrainingStageConfig(
        stage_index=0,
        stage_id=f"toy_{optimizer_id}",
        stage_kind="optimize",
        input_topology_id=None,
        output_topology_id=f"toy_p{parameter_count}",
        input_parameter_count=0,
        output_parameter_count=parameter_count,
        parameter_transfer_rule="initialize_zeros",
        initialization_seed=None,
        optimizer_id=optimizer_id,
        optimizer_hyperparameters=hyperparameters,
        optimizer_seed=23,
        iteration_budget=iteration_budget,
        training_noise_id="depolarizing_1s_all",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        trajectory_count=2,
        training_seed=29,
        trajectory_update="independent",
        sampling_policy=sampling_policy,
        crn_refresh_interval=None,
        checkpoint_validation=CheckpointValidationConfig(
            noise_id="depolarizing_1s_all",
            noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
            noise_strength_scale=1.0,
            tjm_dt=1.0,
            trajectory_count=2,
            seed=31,
            sampling_policy="crn_fixed",
            ensemble_refresh_interval=None,
            cadence=1,
            selection_rule="best_validation_fidelity",
            tie_breaker="earliest_iteration",
        ),
        pruning_rule="none",
        pruning_threshold=None,
        max_bond_dimension=None,
        svd_threshold=0.0,
        truncation_mode="discarded_weight",
        min_bond_dimension=1,
    )


def _circuit(scales: tuple[float, ...]) -> NoisyKrotovCircuitBinding:
    """Return one independently parameterized, policy-bound Pauli circuit."""
    circuit = ParameterizedCircuit(
        max(1, len(scales)),
        [ParameterizedGate("ry", (index,), param_index=index, angle_scale=scale) for index, scale in enumerate(scales)],
        num_params=len(scales),
    )
    return NoisyKrotovCircuitBinding(circuit, f"toy_p{len(scales)}")


class _AdapterKwargs(TypedDict):
    """Typed shared adapter provenance keywords."""

    objective_checksum: str
    provider_checksum: str


def _adapter_kwargs(label: str) -> _AdapterKwargs:
    """Return stable circuit/objective/noise provenance."""
    return {
        "objective_checksum": canonical_checksum({"objective": label}),
        "provider_checksum": canonical_checksum({"provider": label}),
    }


def _trigonometric_loss(target: NDArray[np.float64]) -> Callable[[NDArray[np.float64], object], float]:
    """Return a bounded sum of one-parameter Pauli expectation losses."""

    def objective(theta: NDArray[np.float64], request: object) -> float:
        assert isinstance(request, CompetitorObjectiveRequest)
        return float(np.mean(0.5 * (1.0 - np.cos(theta - target))))

    return objective


def test_parameter_shift_adam_matches_one_step_reference_and_accounts_all_work() -> None:
    """The exact shift rule and bias-corrected first Adam update match a closed form."""
    stage = _stage("parameter_shift_adam", 1)
    objective = _trigonometric_loss(np.array([0.4]))
    execution = ParameterShiftAdamStageAdapter(
        stage,
        _circuit((1.0,)),
        **_adapter_kwargs("adam"),
    ).execute(np.zeros(1), objective, checkpoint_objective=objective)

    expected_gradient = 0.5 * np.sin(-0.4)
    expected_update = 0.1 * expected_gradient / (abs(expected_gradient) + 1e-8)
    assert execution.trace[1].gradient == pytest.approx((expected_gradient,))
    np.testing.assert_allclose(execution.final_theta, [-expected_update], rtol=1e-13, atol=1e-13)
    assert execution.work.objective_calls == 6
    assert execution.work.gradient_calls == 1
    assert execution.work.forward_circuit_evaluations == 12
    assert execution.work.training_trajectories == 8
    assert execution.work.checkpoint_validation_trajectories == 4
    assert execution.work.trajectory_gate_applications == 12

    row = execution.trace[1]
    assert row.objective_stream_checksums[0] == row.objective_stream_checksums[1]
    assert len(set(row.objective_stream_checksums[:3])) == 1
    assert row.objective_stream_checksums[-1] != row.objective_stream_checksums[0]
    assert execution.content_checksum == execution.content_checksum


def test_parameter_shift_honors_signed_angle_scale() -> None:
    """A signed nonunit angle map uses the corresponding theta-space shift and chain rule."""
    stage = _stage("parameter_shift_adam", 1)
    circuit = _circuit((-2.0,))
    target = 0.35

    def objective(theta: NDArray[np.float64], request: CompetitorObjectiveRequest) -> float:
        del request
        return float(0.5 * (1.0 - np.cos(-2.0 * theta[0] - target)))

    execution = ParameterShiftAdamStageAdapter(stage, circuit, **_adapter_kwargs("signed")).execute(
        np.zeros(1),
        objective,
        checkpoint_objective=objective,
    )
    assert execution.trace[1].gradient == pytest.approx((np.sin(target),))


def test_spsa_matches_two_evaluation_reference_and_is_globally_rng_independent() -> None:
    """SPSA uses one paired Rademacher direction and one-based gain schedules."""
    stage = _stage("spsa", 2)
    circuit = _circuit((1.0, 1.0))
    objective = _trigonometric_loss(np.array([0.3, -0.2]))
    adapter = SPSAStageAdapter(stage, circuit, **_adapter_kwargs("spsa"))

    np.random.Generator(np.random.PCG64(1)).standard_normal(100)
    first = adapter.execute(np.zeros(2), objective, checkpoint_objective=objective)
    np.random.Generator(np.random.PCG64(999)).standard_normal(100)
    second = adapter.execute(np.zeros(2), objective, checkpoint_objective=objective)
    np.testing.assert_array_equal(first.final_theta, second.final_theta)
    assert first.trace == second.trace

    row = first.trace[1]
    coefficient = (row.plus_losses[0] - row.minus_losses[0]) / (2.0 * cast("float", row.perturbation_scale))
    assert np.abs(np.asarray(row.gradient)) == pytest.approx(np.full(2, abs(coefficient)))
    expected_rate = 0.2 / (1.0 + 1.0) ** 0.602
    np.testing.assert_allclose(first.final_theta, -expected_rate * np.asarray(row.gradient))
    assert row.objective_stream_checksums[0] == row.objective_stream_checksums[1]
    assert row.objective_stream_checksums[2] == row.objective_stream_checksums[0]
    assert first.work.objective_calls == 6
    assert first.work.gradient_calls == 1


def test_resampled_spsa_generates_each_map_at_the_unperturbed_center() -> None:
    """The second SPSA window is sampled at row one, never at its plus arm."""
    stage = _stage("spsa", 8, iteration_budget=2)
    binding = _circuit((1.0,) * 8)
    target = load_legacy_target_collection().target("legacy_tfim_seed_100")
    objective = FixedRateNoisyCompetitorObjective(stage, binding, target)
    execution = SPSAStageAdapter(
        stage,
        binding,
        objective_checksum=objective.objective_checksum,
        provider_checksum=objective.provider_checksum,
    ).execute(np.zeros(8), objective, checkpoint_objective=objective)

    expected_centers = (execution.trace[0].parameter_checksum, execution.trace[1].parameter_checksum)
    assert execution.training_ensemble_sampling_parameter_checksums == expected_centers
    assert (
        tuple(cast("Sequence[str]", execution.to_dict()["training_ensemble_sampling_parameter_checksums"]))
        == expected_centers
    )


def test_equal_work_budget_stops_before_an_entire_unaffordable_update() -> None:
    """No callback or parameter mutation occurs for a partially affordable Adam update."""
    stage = _stage("parameter_shift_adam", 1, iteration_budget=3)
    calls: list[str] = []

    def objective(theta: NDArray[np.float64], request: CompetitorObjectiveRequest) -> float:
        calls.append(request.content_checksum)
        return float(0.5 * (1.0 - np.cos(theta[0] - 0.4)))

    execution = ParameterShiftAdamStageAdapter(
        stage,
        _circuit((1.0,)),
        work_budget=CompetitorWorkBudget(objective_calls=6, gradient_calls=1),
        **_adapter_kwargs("budget"),
    ).execute(np.zeros(1), objective, checkpoint_objective=objective)
    assert execution.stop_reason == "work_budget_exhausted"
    assert execution.completed_iterations == 1
    assert execution.work.objective_calls == 6
    assert len(calls) == 6
    with pytest.raises(ValueError, match="resource-truncated"):
        execution.to_stage_evidence(
            source_parameters=None,
            circuit_statistics={},
        )


def test_parameter_shift_rejects_shared_parameters_instead_of_mislabeling_gradient() -> None:
    """The two-call rule is never applied to a parameter reused by multiple gates."""
    stage = _stage("parameter_shift_adam", 1)
    circuit = NoisyKrotovCircuitBinding(
        ParameterizedCircuit(
            2,
            [
                ParameterizedGate("ry", (0,), param_index=0),
                ParameterizedGate("rz", (1,), param_index=0),
            ],
            num_params=1,
        ),
        stage.output_topology_id,
    )
    with pytest.raises(ValueError, match="exactly one Pauli rotation"):
        ParameterShiftAdamStageAdapter(stage, circuit, **_adapter_kwargs("shared"))


def test_layerwise_and_fixed_competitor_templates_keep_distinct_honest_identities() -> None:
    """Adam/SPSA templates retain the reference ansatz but never share method streams."""
    adam_layerwise = build_parameter_shift_adam_layerwise_template(
        training_trajectory_count=2,
        checkpoint_validation_trajectory_count=3,
    )
    spsa_layerwise = build_spsa_layerwise_template(
        training_trajectory_count=2,
        checkpoint_validation_trajectory_count=3,
    )
    adam_fixed = build_parameter_shift_adam_fixed_template(
        iteration_budget=7,
        training_trajectory_count=2,
        checkpoint_validation_trajectory_count=3,
    )
    spsa_fixed = build_spsa_fixed_template(
        iteration_budget=7,
        training_trajectory_count=2,
        checkpoint_validation_trajectory_count=3,
    )

    assert adam_layerwise.method_id == "parameter_shift_adam_layerwise"
    assert spsa_layerwise.method_id == "spsa_layerwise"
    assert adam_fixed.method_id == "parameter_shift_adam_fixed"
    assert spsa_fixed.method_id == "spsa_fixed"
    assert all(stage.stage_policy["optimizer_id"] == "parameter_shift_adam" for stage in adam_layerwise.stages)
    assert all(stage.stage_policy["optimizer_id"] == "spsa" for stage in spsa_layerwise.stages)
    assert all(stage.stage_policy["sampling_policy"] == "none" for stage in adam_layerwise.stages[:-1])
    assert all(stage.stage_policy["sampling_policy"] == "none" for stage in spsa_layerwise.stages[:-1])
    assert adam_layerwise.stages[-1].stage_policy["sampling_policy"] == "crn_fixed"
    assert spsa_layerwise.stages[-1].stage_policy["sampling_policy"] == "resampled"
    assert len(adam_layerwise.stages) == len(spsa_layerwise.stages) == 5
    assert len(adam_fixed.stages) == len(spsa_fixed.stages) == 1

    adam_bindings = {
        value for stage in adam_layerwise.stages for value in stage.seed_bindings.values() if value is not None
    }
    spsa_bindings = {
        value for stage in spsa_layerwise.stages for value in stage.seed_bindings.values() if value is not None
    }
    assert adam_bindings.isdisjoint(spsa_bindings)
    for template in (adam_layerwise, spsa_layerwise, adam_fixed, spsa_fixed):
        assert TrainingPipelineTemplate.from_json(template.to_json()) == template


def test_generic_callback_cannot_forge_publishable_competitor_evidence() -> None:
    """Math-test callbacks cannot invent target, provider, map, or topology provenance."""
    stage = _stage("parameter_shift_adam", 1)
    objective = _trigonometric_loss(np.array([0.4]))

    def validation(theta: NDArray[np.float64], request: CompetitorObjectiveRequest) -> float:
        del theta
        return 0.0 if request.global_iteration == 0 else 0.5

    execution = ParameterShiftAdamStageAdapter(
        stage,
        _circuit((1.0,)),
        objective_checksum=canonical_checksum({"objective": "artifact"}),
        provider_checksum=canonical_checksum({"provider": "artifact"}),
    ).execute(np.zeros(1), objective, checkpoint_objective=validation)
    with pytest.raises(ValueError, match="target-bound"):
        execution.to_stage_evidence(source_parameters=None, circuit_statistics={})
