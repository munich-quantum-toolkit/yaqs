# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for independent state-preparation benchmark evaluation."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import benchmarks.state_preparation.evaluation as evaluation_module
from benchmarks.state_preparation import (
    BALLARIN_NOISE_ID,
    AnsatzConfig,
    BenchmarkConfig,
    EvaluationConfig,
    InitializationConfig,
    KrotovStatePreparationMethod,
    NoiseConfig,
    OptimizerConfig,
    SeedDomain,
    TargetSelection,
    count_nonidentity_events,
    derive_seed_sequence,
    evaluate_state_preparation_artifact,
    load_target_collection,
    state_preparation_training_id,
    train_state_preparation_method,
)
from mqt.yaqs.optimization import KrotovNoiseMap, RandomUnitaryInstruction

if TYPE_CHECKING:
    from benchmarks.state_preparation import StatePreparationTrainingArtifact, TargetCollection
    from mqt.yaqs.optimization import GateNoiseContext


def _optimizer() -> OptimizerConfig:
    """Return a deterministic zero-iteration optimizer configuration."""
    return OptimizerConfig(
        optimizer_id="krotov",
        max_iterations=0,
        optimizer_seed=17,
        hyperparameters={"step_size": 0.1, "schedule": {"kind": "constant"}},
        train_trajectories_or_shots=0,
        training_seed=None,
    )


def _config(
    targets: TargetCollection,
    *,
    ansatz: AnsatzConfig | None = None,
    noise_id: str = "dephasing_1s_1q",
    test_seed: int | None = 23,
    trajectory_count: int = 5,
    sidecar: bool = True,
    confidence_level: float | None = 0.95,
) -> BenchmarkConfig:
    """Return a small fully resolved benchmark configuration."""
    target = targets.load_target(6, "gaussian_mu0p5_sigma0p1")
    noiseless = noise_id == "noiseless"
    return BenchmarkConfig(
        method_id="krotov",
        method_version="1",
        target=TargetSelection(
            num_qubits=target.num_qubits,
            target_id=target.target_id,
            target_seed=target.seed,
            fixture_format=targets.fixture_format,
            fixture_checksum=targets.fixture_checksum,
        ),
        ansatz=ansatz or AnsatzConfig(0, initial_single_qubit_layer=True),
        initialization=InitializationConfig(rule="random_normal", seed=11, scale=0.1),
        optimizer=_optimizer(),
        evaluation=EvaluationConfig(
            test_trajectories_or_shots=0 if noiseless else trajectory_count,
            test_seed=None if noiseless else test_seed,
            store_trajectory_sidecar=False if noiseless else sidecar,
            confidence_level=None if noiseless else confidence_level,
            confidence_interval_method=None if noiseless or confidence_level is None else "normal_clipped",
        ),
        training_noise=NoiseConfig("noiseless"),
        test_noise=NoiseConfig(
            noise_id,
            tjm_dt=1.0 if noise_id not in {"noiseless", BALLARIN_NOISE_ID} else None,
        ),
    )


@pytest.fixture
def trained_problem() -> tuple[
    KrotovStatePreparationMethod,
    StatePreparationTrainingArtifact,
    BenchmarkConfig,
    TargetCollection,
]:
    """Train one inexpensive artifact reusable across evaluation cells.

    Returns:
        The method, trained artifact, evaluation config, and target collection.
    """
    targets = load_target_collection()
    config = _config(targets)
    method = KrotovStatePreparationMethod()
    artifact = train_state_preparation_method(method, config, targets)
    return method, artifact, config, targets


def test_seed_domains_are_stable_and_disjoint() -> None:
    """Domain tags, repetitions, and samples must produce distinct streams."""
    states = {
        tuple(derive_seed_sequence(123, domain, repetition=repetition, sample_index=sample).generate_state(4))
        for domain in SeedDomain
        for repetition, sample in ((0, 0), (1, 0), (0, 1))
    }
    assert len(states) == len(SeedDomain) * 3
    assert tuple(derive_seed_sequence(123, SeedDomain.TEST_TRAJECTORY).generate_state(4)) == tuple(
        derive_seed_sequence(123, SeedDomain.TEST_TRAJECTORY).generate_state(4)
    )


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("resolved_seed", True, TypeError),
        ("resolved_seed", -1, ValueError),
        ("repetition", -1, ValueError),
        ("sample_index", 2**64, ValueError),
    ],
)
def test_seed_derivation_rejects_invalid_coordinates(argument: str, value: int, error: type[Exception]) -> None:
    """Seed derivation must not silently coerce invalid coordinates."""
    kwargs = {"resolved_seed": 1, "repetition": 0, "sample_index": 0}
    kwargs[argument] = value
    with pytest.raises(error):
        derive_seed_sequence(
            kwargs["resolved_seed"],
            SeedDomain.TEST_TRAJECTORY,
            repetition=kwargs["repetition"],
            sample_index=kwargs["sample_index"],
        )


def test_fixed_evaluation_is_exactly_reproducible_and_uses_full_budget(
    trained_problem: tuple[
        KrotovStatePreparationMethod,
        StatePreparationTrainingArtifact,
        BenchmarkConfig,
        TargetCollection,
    ],
) -> None:
    """A repeated call must reproduce every sampled and aggregate value."""
    method, artifact, config, targets = trained_problem
    first = evaluate_state_preparation_artifact(method, artifact, config, targets)
    second = evaluate_state_preparation_artifact(method, artifact, config, targets)

    assert first == second
    assert first.trajectory_fidelities is not None
    assert len(first.trajectory_fidelities) == config.evaluation.test_trajectories_or_shots
    assert first.test_noisy_fidelity == pytest.approx(float(np.mean(first.trajectory_fidelities)))
    assert first.noisy_fidelity_standard_deviation == pytest.approx(float(np.std(first.trajectory_fidelities, ddof=1)))
    assert first.noisy_fidelity_standard_error == pytest.approx(
        first.noisy_fidelity_standard_deviation / np.sqrt(len(first.trajectory_fidelities))
    )
    assert first.confidence_interval_lower is not None
    assert first.confidence_interval_upper is not None


def test_changed_test_seed_does_not_change_training_or_noiseless_evaluation(
    trained_problem: tuple[
        KrotovStatePreparationMethod,
        StatePreparationTrainingArtifact,
        BenchmarkConfig,
        TargetCollection,
    ],
) -> None:
    """Test randomness must not participate in optimization or noiseless metrics."""
    method, artifact, config, targets = trained_problem
    changed = replace(config, evaluation=replace(config.evaluation, test_seed=29))

    assert state_preparation_training_id(method, changed) == artifact.training_id
    original_result = evaluate_state_preparation_artifact(method, artifact, config, targets)
    changed_result = evaluate_state_preparation_artifact(method, artifact, changed, targets)
    assert changed_result.training_id == original_result.training_id
    assert changed_result.train_fidelity == original_result.train_fidelity
    assert changed_result.logical_test_noiseless_fidelity == original_result.logical_test_noiseless_fidelity
    assert changed_result.test_noiseless_fidelity == original_result.test_noiseless_fidelity


def test_noiseless_evaluation_has_no_sampling_fields(
    trained_problem: tuple[
        KrotovStatePreparationMethod,
        StatePreparationTrainingArtifact,
        BenchmarkConfig,
        TargetCollection,
    ],
) -> None:
    """Noiseless evaluation must be seed-independent and bypass trajectories."""
    method, artifact, _config_unused, targets = trained_problem
    noiseless_config = _config(targets, noise_id="noiseless")
    assert state_preparation_training_id(method, noiseless_config) == artifact.training_id

    result = evaluate_state_preparation_artifact(method, artifact, noiseless_config, targets)
    assert result.test_noisy_fidelity == result.test_noiseless_fidelity
    assert result.sampled_nonidentity_events == 0
    assert result.noisy_fidelity_standard_deviation is None
    assert result.noisy_fidelity_standard_error is None
    assert result.trajectory_fidelities is None
    assert result.logical_test_noiseless_fidelity == result.test_noiseless_fidelity


def test_training_noise_maps_cannot_be_passed_to_evaluation() -> None:
    """The public evaluator must have no path for replaying training maps."""
    parameters = inspect.signature(evaluate_state_preparation_artifact).parameters
    assert "noise_maps" not in parameters
    assert "fixed_noise_maps" not in parameters


def test_nonidentity_event_count_distinguishes_channel_semantics() -> None:
    """Ballarin local outcomes and TJM jumps must use their natural event units."""
    maps = [
        KrotovNoiseMap(
            channel_id=BALLARIN_NOISE_ID,
            outcome_labels=("X", "Z"),
            is_identity=False,
        ),
        KrotovNoiseMap(
            channel_id="dephasing_1s_1q",
            jump_process_index=0,
            is_identity=False,
        ),
        KrotovNoiseMap(
            channel_id="dephasing_1s_1q",
            jump_process_index=None,
            is_identity=True,
        ),
    ]
    assert count_nonidentity_events(maps) == 3


class _RecordingBallarinProvider:
    """Ballarin-shaped provider recording fresh trajectory RNG draws."""

    def __init__(self) -> None:
        self.draws: list[float] = []

    def __call__(
        self,
        context: GateNoiseContext,
        rng: np.random.Generator,
        /,
    ) -> RandomUnitaryInstruction | None:
        """Record one draw for every retained noisy native rotation.

        Returns:
            An identity Ballarin outcome for native rotations, otherwise
            ``None``.
        """
        if context.gate_name != "rzz":
            return None
        self.draws.append(float(rng.random()))
        return RandomUnitaryInstruction(
            (),
            channel_id=BALLARIN_NOISE_ID,
            outcome_labels=("I", "I"),
        )


def test_ballarin_outcomes_refresh_for_independent_repetitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated evaluation must derive new Ballarin streams without retraining."""
    targets = load_target_collection()
    config = _config(
        targets,
        ansatz=AnsatzConfig(1, initial_single_qubit_layer=False),
        noise_id=BALLARIN_NOISE_ID,
        trajectory_count=2,
        confidence_level=None,
    )
    method = KrotovStatePreparationMethod()
    artifact = train_state_preparation_method(method, config, targets)
    provider = _RecordingBallarinProvider()
    monkeypatch.setattr(evaluation_module, "create_ballarin_noise_provider", lambda: provider)

    first_result = evaluate_state_preparation_artifact(method, artifact, config, targets, repetition=0)
    first_draws = tuple(provider.draws)
    provider.draws.clear()
    evaluate_state_preparation_artifact(method, artifact, config, targets, repetition=1)
    second_draws = tuple(provider.draws)

    expected_draws = config.evaluation.test_trajectories_or_shots * first_result.circuit_statistics.native_rzz_count
    assert len(first_draws) == expected_draws
    assert len(second_draws) == expected_draws
    assert first_draws != second_draws
