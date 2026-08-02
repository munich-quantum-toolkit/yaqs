# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Structural and numerical regression tests for WP19 layerwise BMPD profiles."""

from __future__ import annotations

import inspect
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import benchmarks.state_preparation.phase2.layerwise_bmpd as layerwise_bmpd_module
import benchmarks.state_preparation.phase2.noisy_krotov as noisy_krotov_module
from benchmarks.state_preparation.noise import (
    FIXED_RATE_NOISE_DEFINITION_VERSION,
    HISTORICAL_FIXED_RATE_NOISE_ID,
    create_historical_fixed_rate_noise_provider,
)
from benchmarks.state_preparation.phase2.layerwise_bmpd import (
    LAYERWISE_BMPD_APPEND_SCALE,
    LAYERWISE_BMPD_CRN_LEGACY_METHOD_ID,
    LAYERWISE_BMPD_CRN_V2_METHOD_ID,
    LAYERWISE_BMPD_INITIAL_SCALE,
    LEGACY_EVALUATION_SEED,
    LEGACY_EVALUATION_TRAJECTORY_COUNT,
    LEGACY_TRAINING_TRAJECTORY_COUNT,
    LayerwiseBMPDStageRunner,
    bmpd_parameter_count,
    build_layerwise_bmpd_crn_legacy_v1_template,
    build_layerwise_bmpd_crn_v2_template,
    create_bmpd_circuit_binding,
    initialize_layerwise_stage_parameters,
    resolve_layerwise_bmpd_crn_legacy_v1_pipeline,
)
from benchmarks.state_preparation.phase2.legacy_targets import load_legacy_target_collection
from benchmarks.state_preparation.phase2.noisy_krotov import NoisyKrotovStageExecution, execute_fixed_rate_krotov_stage
from benchmarks.state_preparation.phase2.protocol import load_initial_preregistration
from benchmarks.state_preparation.phase2.targets import (
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization import (
    KrotovFixedMapEnsemble,
    KrotovOptions,
    KrotovTJMOptions,
    KrotovTruncation,
    derive_legacy_krotov_trajectory_seed,
    forward_tjm_trajectory,
    noisy_state_preparation_metrics,
    sample_krotov_fixed_map_ensemble,
    state_preparation_metrics,
    train_krotov_noisy_state_preparation_batch,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineConfig
    from benchmarks.state_preparation.phase2.targets import TargetPopulationManifest


_MASTER_ENTROPY = bytes(range(32))
_LEGACY_UPDATE_ATOL = 5e-15


def _historical_global_noise_model(num_qubits: int) -> NoiseModel:
    """Build the exact archived global IBM-inspired Pauli model.

    Returns:
        The pre-provider global ``NoiseModel`` in its archived process order.
    """
    processes = [
        {"name": name, "sites": [site], "strength": 3.0e-4 / 3.0}
        for site in range(num_qubits)
        for name in ("pauli_x", "pauli_y", "pauli_z")
    ]
    processes.extend(
        {"name": name, "sites": [site, site + 1], "strength": 1.5e-3}
        for site in range(num_qubits - 1)
        for name in ("crosstalk_xx", "crosstalk_zz")
    )
    return NoiseModel(processes)


@lru_cache(maxsize=1)
def _screening_manifest() -> TargetPopulationManifest:
    """Return a genuine typed primary-q6 manifest for corrected-profile tests.

    Returns:
        A seed-bearing primary-q6 screening manifest.
    """
    preregistration = load_initial_preregistration()
    config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(_MASTER_ENTROPY),
        population_scope="primary_q6",
    )
    return create_target_population_manifest(config, preregistration, _MASTER_ENTROPY)


def _resolve_v2(*, optimization_seed: int = 73) -> TrainingPipelineConfig:
    """Resolve the corrected template against one genuine target specification.

    Returns:
        A target-bound corrected pipeline with explicit trajectory counts.
    """
    template = build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=3,
        checkpoint_validation_trajectory_count=5,
    )
    manifest = _screening_manifest()
    target = manifest.instances[0]
    return template.resolve(
        target_namespace="phase2",
        target_manifest=manifest,
        target_instance_id=target.target_instance_id,
        target_population_manifest_checksum=manifest.content_checksum,
        target_instance_spec_checksum=target.content_checksum,
        target_family_id=target.family_id,
        target_stratum_id=target.stratum_id,
        qubit_count=target.qubit_count,
        optimization_block_id="wp19_v2_seed_audit",
        optimization_seed=optimization_seed,
        data_role="screening_selection",
    )


def _assert_global_randomstate_unchanged(before: object, after: object) -> None:
    """Compare the process-global legacy RNG state without consuming it."""
    assert isinstance(before, tuple)
    assert isinstance(after, tuple)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_layerwise_parameter_counts_and_profiles_have_exact_five_stage_topologies() -> None:
    """Both profiles grow through depths one to four and fine-tune depth four."""
    legacy = build_layerwise_bmpd_crn_legacy_v1_template()
    corrected = build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=7,
        checkpoint_validation_trajectory_count=11,
    )

    assert [bmpd_parameter_count(8, depth) for depth in range(1, 5)] == [87, 150, 213, 276]
    assert [bmpd_parameter_count(6, depth) for depth in range(1, 5)] == [63, 108, 153, 198]
    assert [stage.stage_policy["output_parameter_count"] for stage in legacy.stages] == [87, 150, 213, 276, 276]
    assert [stage.stage_policy["output_parameter_count"] for stage in corrected.stages] == [63, 108, 153, 198, 198]
    assert [stage.stage_policy["stage_id"] for stage in legacy.stages] == [
        "grow_d1",
        "grow_d2",
        "grow_d3",
        "grow_d4",
        "final_finetune",
    ]
    assert legacy.method_id == LAYERWISE_BMPD_CRN_LEGACY_METHOD_ID
    assert corrected.method_id == LAYERWISE_BMPD_CRN_V2_METHOD_ID
    assert legacy.configuration_checksum != corrected.configuration_checksum


def test_legacy_profile_freezes_historical_stage_noise_and_seed_arithmetic() -> None:
    """The isolated legacy profile retains every audited update and seed choice."""
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(100)
    growth = pipeline.stages[:4]
    final = pipeline.stages[-1]

    assert [stage.iteration_budget for stage in growth] == [100] * 4
    assert [stage.optimizer_hyperparameters["learning_rate"] for stage in growth] == [1.0] * 4
    assert [stage.optimizer_hyperparameters["schedule"] for stage in growth] == ["constant"] * 4
    assert [stage.initialization_seed for stage in growth] == [2000, 2002, 2003, 2004]
    assert [stage.optimizer_seed for stage in growth] == [3000] * 4
    assert final.iteration_budget == 200
    assert dict(final.optimizer_hyperparameters) == {
        "decay": 0.01,
        "learning_rate": 0.2,
        "schedule": "exp",
    }
    assert final.training_noise_id == HISTORICAL_FIXED_RATE_NOISE_ID
    assert final.noise_definition_version == FIXED_RATE_NOISE_DEFINITION_VERSION
    assert final.noise_strength_scale is not None
    assert final.tjm_dt is not None
    assert float(final.noise_strength_scale).hex() == (1.0).hex()
    assert float(final.tjm_dt).hex() == (1.0).hex()
    assert final.trajectory_count == LEGACY_TRAINING_TRAJECTORY_COUNT
    assert final.trajectory_update == "cross"
    assert final.sampling_policy == "crn_fixed"
    assert final.optimizer_seed == final.training_seed == 4000
    assert LEGACY_EVALUATION_TRAJECTORY_COUNT == 500
    assert LEGACY_EVALUATION_SEED == 0


def test_legacy_seed_policy_is_bound_to_the_target_seed_and_reserved_method() -> None:
    """Historical seed reuse cannot leak into corrected methods or another target row."""
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(200)
    with pytest.raises(ValueError, match="target seed as its exact outer seed"):
        pipeline.template.resolve(
            target_namespace="legacy_reproduction",
            target_manifest=None,
            target_instance_id=pipeline.target_instance_id,
            target_population_manifest_checksum=pipeline.target_population_manifest_checksum,
            target_instance_spec_checksum=pipeline.target_instance_spec_checksum,
            target_family_id=pipeline.target_family_id,
            target_stratum_id=pipeline.target_stratum_id,
            qubit_count=pipeline.qubit_count,
            optimization_block_id=pipeline.optimization_block_id,
            optimization_seed=201,
            data_role="secondary_benchmark",
        )
    with pytest.raises(ValueError, match="one of 100, 200, 300, 400, or 500"):
        resolve_layerwise_bmpd_crn_legacy_v1_pipeline(101)

    legacy = build_layerwise_bmpd_crn_legacy_v1_template()
    with pytest.raises(ValueError, match="reserved"):
        replace(legacy, method_id=LAYERWISE_BMPD_CRN_V2_METHOD_ID, target_scope_id="primary_q6")


def test_legacy_randomstate_initialization_and_growth_are_exact_and_process_local() -> None:
    """RandomState draws match the archived formulas without touching global RNG state."""
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(100)
    global_before = np.random.get_state()  # noqa: NPY002 -- compatibility invariant
    initial = initialize_layerwise_stage_parameters(pipeline.stages[0], None)
    predecessor = np.linspace(-0.2, 0.2, pipeline.stages[1].input_parameter_count)
    grown = initialize_layerwise_stage_parameters(pipeline.stages[1], predecessor)
    global_after = np.random.get_state()  # noqa: NPY002 -- compatibility invariant

    expected_initial = (
        np.random.RandomState(2000).standard_normal(pipeline.stages[0].output_parameter_count)
        * LAYERWISE_BMPD_INITIAL_SCALE
    )
    tail_count = pipeline.stages[1].output_parameter_count - pipeline.stages[1].input_parameter_count
    expected_tail = np.random.RandomState(2002).standard_normal(tail_count) * LAYERWISE_BMPD_APPEND_SCALE
    np.testing.assert_array_equal(initial, expected_initial)
    np.testing.assert_array_equal(grown[: predecessor.size], predecessor)
    np.testing.assert_array_equal(grown[predecessor.size :], expected_tail)
    _assert_global_randomstate_unchanged(global_before, global_after)


def test_corrected_initialization_uses_pcg64_and_disjoint_resolved_streams() -> None:
    """The v2 profile uses independent hash-derived streams and the declared PCG64 API."""
    pipeline = _resolve_v2()
    seeds = [
        seed
        for stage in pipeline.stages
        for seed in (
            stage.initialization_seed,
            stage.optimizer_seed,
            stage.training_seed,
            stage.checkpoint_validation.seed,
        )
        if seed is not None
    ]
    assert len(seeds) == len(set(seeds))

    first = pipeline.stages[0]
    actual = initialize_layerwise_stage_parameters(first, None)
    expected = (
        np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(cast("int", first.initialization_seed)))
        ).standard_normal(first.output_parameter_count)
        * LAYERWISE_BMPD_INITIAL_SCALE
    )
    np.testing.assert_array_equal(actual, expected)


def test_zero_initialized_growth_tail_preserves_the_predecessor_fidelity() -> None:
    """Appending an identity layer leaves the represented state unchanged before training."""
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(100)
    shallow = create_bmpd_circuit_binding(8, 1).circuit
    deeper = create_bmpd_circuit_binding(8, 2).circuit
    shallow_theta = np.linspace(-0.31, 0.29, shallow.num_params)
    deeper_theta = initialize_layerwise_stage_parameters(
        pipeline.stages[1],
        shallow_theta,
        appended_values=np.zeros(deeper.num_params - shallow.num_params),
    )
    target = np.arange(1, 257, dtype=np.float64).astype(np.complex128)
    target /= np.linalg.norm(target)

    shallow_fidelity = state_preparation_metrics(shallow, shallow_theta, target)[1]
    deeper_fidelity = state_preparation_metrics(deeper, deeper_theta, target)[1]
    assert deeper_fidelity == pytest.approx(shallow_fidelity, rel=0.0, abs=2e-15)


def test_one_legacy_noiseless_update_matches_the_compatibility_path_within_roundoff() -> None:
    """The staged adapter retains the archived empty-noise batch update semantics."""
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(100)
    stage = replace(pipeline.stages[0], iteration_budget=1)
    binding = create_bmpd_circuit_binding(8, 1)
    target = load_legacy_target_collection().target("legacy_tfim_seed_100")
    initial = initialize_layerwise_stage_parameters(stage, None)

    staged = execute_fixed_rate_krotov_stage(stage, binding, target, initial)
    assert isinstance(staged, NoisyKrotovStageExecution)
    compatibility = train_krotov_noisy_state_preparation_batch(
        binding.circuit,
        target.state_vector_copy(),
        NoiseModel([]),
        KrotovTJMOptions(num_trajectories=1, use_crn=False),
        initial_theta=initial,
        options=KrotovOptions(
            max_iterations=1,
            batch_step_size=1.0,
            batch_schedule="constant",
            seed=3000,
        ),
    )

    np.testing.assert_allclose(staged.final_theta, compatibility.theta, rtol=0.0, atol=5e-15)
    assert staged.trace[-1].monitoring_fidelity == pytest.approx(
        compatibility.trace["fidelity"][-1],
        rel=0.0,
        abs=5e-15,
    )


def test_legacy_layerwise_runner_passes_its_method_identity_to_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the identity-bearing legacy runner opts into compatibility sampling."""
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(100)
    target = load_legacy_target_collection().target("legacy_tfim_seed_100")
    captured: dict[str, object] = {}
    marker = cast("NoisyKrotovStageExecution", object())

    def execute(*args: object, **kwargs: object) -> NoisyKrotovStageExecution:
        captured["args"] = args
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(layerwise_bmpd_module, "execute_fixed_rate_krotov_stage", execute)
    result = LayerwiseBMPDStageRunner(pipeline, target)(pipeline.stages[0], None)

    assert result is marker
    assert captured["compatibility_method_id"] == LAYERWISE_BMPD_CRN_LEGACY_METHOD_ID


def test_one_q8_legacy_noisy_update_matches_the_archived_global_noise_model() -> None:
    """The provider adapter must reproduce one archived global-model CRN update."""
    pipeline = resolve_layerwise_bmpd_crn_legacy_v1_pipeline(100)
    stage = replace(pipeline.stages[-1], iteration_budget=1)
    binding = create_bmpd_circuit_binding(8, 4)
    target = load_legacy_target_collection().target("legacy_tfim_seed_100")
    initial = np.linspace(-0.23, 0.19, binding.circuit.num_params, dtype=np.float64)

    modern_provider_result = execute_fixed_rate_krotov_stage(stage, binding, target, initial)
    assert isinstance(modern_provider_result, NoisyKrotovStageExecution)
    modern_maps = modern_provider_result.training_ensembles[0].replay_maps()
    assert all(noise_map.normalized for trajectory in modern_maps for noise_map in trajectory)

    provider_result = execute_fixed_rate_krotov_stage(
        stage,
        binding,
        target,
        initial,
        compatibility_method_id=LAYERWISE_BMPD_CRN_LEGACY_METHOD_ID,
    )
    assert isinstance(provider_result, NoisyKrotovStageExecution)
    assert (
        provider_result.training_ensembles[0].content_checksum
        != modern_provider_result.training_ensembles[0].content_checksum
    )
    assert provider_result.training_ensembles[0].nonidentity_event_count > 0
    replay_maps = provider_result.training_ensembles[0].replay_maps()
    assert all(not noise_map.normalized for trajectory in replay_maps for noise_map in trajectory)
    assert all(not noise_map.normalization_checkpoints for trajectory in replay_maps for noise_map in trajectory)

    archived_result = train_krotov_noisy_state_preparation_batch(
        binding.circuit,
        target.state_vector_copy(),
        _historical_global_noise_model(8),
        KrotovTJMOptions(
            num_trajectories=3,
            random_seed=0,
            dt=1.0,
            apply_noise_to="all",
            noisy_gate_indices=binding.noisy_gate_indices,
            trajectory_update="cross",
            use_crn=True,
        ),
        initial_theta=initial,
        options=KrotovOptions(
            max_iterations=1,
            batch_step_size=0.2,
            batch_schedule="exp",
            batch_decay=0.01,
            seed=4000,
            truncation=KrotovTruncation(
                max_bond_dim=stage.max_bond_dimension,
                svd_threshold=stage.svd_threshold,
                trunc_mode=stage.truncation_mode,
                min_bond_dim=stage.min_bond_dimension,
            ),
        ),
    )

    parameter_drift = float(np.max(np.abs(provider_result.final_theta - archived_result.theta)))
    assert parameter_drift <= _LEGACY_UPDATE_ATOL
    np.testing.assert_allclose(
        provider_result.final_theta,
        archived_result.theta,
        rtol=0.0,
        atol=_LEGACY_UPDATE_ATOL,
    )
    assert provider_result.trace[-1].monitoring_fidelity == pytest.approx(
        archived_result.trace["fidelity"][-1],
        rel=0.0,
        abs=_LEGACY_UPDATE_ATOL,
    )


def test_q8_legacy_evaluation_replay_matches_archived_direct_sampling() -> None:
    """Evaluation keeps live normalization while reproducing legacy trajectory seeds."""
    binding = create_bmpd_circuit_binding(8, 4)
    target = load_legacy_target_collection().target("legacy_tfim_seed_100")
    theta = np.linspace(-0.23, 0.19, binding.circuit.num_params, dtype=np.float64)
    truncation = KrotovTruncation()
    provider = create_historical_fixed_rate_noise_provider()
    options = KrotovTJMOptions(
        num_trajectories=3,
        random_seed=0,
        dt=1.0,
        apply_noise_to="all",
        noisy_gate_indices=binding.noisy_gate_indices,
        trajectory_update="independent",
        use_crn=False,
    )
    checksum = "sha256:" + "2" * 64
    ensemble = sample_krotov_fixed_map_ensemble(
        binding.circuit,
        theta,
        None,
        truncation,
        provider,
        options,
        role="pilot_evaluation",
        resolved_seed=0,
        stage_index=5,
        stage_id="legacy_evaluation",
        stage_configuration_checksum=checksum,
        circuit_checksum=binding.content_checksum,
        provider_checksum=provider.content_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
        legacy_linear_seed=True,
        legacy_compact_replay=False,
    )
    replay_maps = ensemble.replay_maps()
    assert ensemble.nonidentity_event_count > 0
    assert all(noise_map.normalized for trajectory in replay_maps for noise_map in trajectory)

    _, replay_fidelity, replay_trajectories = noisy_state_preparation_metrics(
        binding.circuit,
        theta,
        target.state_vector_copy(),
        None,
        options,
        truncation=truncation,
        fixed_noise_maps=replay_maps,
        noise_provider=provider,
    )
    _, archived_fidelity, archived_trajectories = noisy_state_preparation_metrics(
        binding.circuit,
        theta,
        target.state_vector_copy(),
        _historical_global_noise_model(8),
        options,
        truncation=truncation,
        iteration=0,
    )

    np.testing.assert_allclose(
        replay_trajectories,
        archived_trajectories,
        rtol=0.0,
        atol=_LEGACY_UPDATE_ATOL,
    )
    assert replay_fidelity == pytest.approx(
        archived_fidelity,
        rel=0.0,
        abs=_LEGACY_UPDATE_ATOL,
    )


def test_historical_noise_provider_has_exact_process_placement_and_strengths() -> None:
    """The legacy simulated profile reproduces local Pauli and adjacent XX/ZZ rates."""
    provider = create_historical_fixed_rate_noise_provider()
    document = provider.to_dict()
    one_qubit = cast("list[dict[str, object]]", document["one_qubit_gate_processes"])
    two_qubit = cast("list[dict[str, object]]", document["two_qubit_gate_processes"])

    assert document["noise_id"] == HISTORICAL_FIXED_RATE_NOISE_ID
    assert document["is_hardware_execution"] is False
    assert [item["name"] for item in one_qubit] == ["pauli_x", "pauli_y", "pauli_z"]
    one_qubit_strengths = [item["strength"] for item in one_qubit]
    assert all(isinstance(strength, float) for strength in one_qubit_strengths)
    assert {strength.hex() for strength in one_qubit_strengths if isinstance(strength, float)} == {
        "0x1.a36e2eb1c432cp-14"
    }
    assert [item["name"] for item in two_qubit][-2:] == ["crosstalk_xx", "crosstalk_zz"]
    assert {item["strength"] for item in two_qubit[-2:]} == {0.0015}


def test_legacy_linear_trajectory_seeds_reproduce_the_archived_formula_and_maps() -> None:
    """The compatibility sampler realizes the exact old iteration/trajectory seed map."""
    assert [
        derive_legacy_krotov_trajectory_seed(optimizer_iteration_seed=4000, trajectory_index=index)
        for index in range(3)
    ] == [4_000_012_000, 4_000_012_001, 4_000_012_002]

    binding = create_bmpd_circuit_binding(2, 1)
    circuit = binding.circuit
    theta = np.zeros(circuit.num_params, dtype=np.float64)
    truncation = KrotovTruncation()
    provider = create_historical_fixed_rate_noise_provider()
    options = KrotovTJMOptions(
        num_trajectories=3,
        random_seed=0,
        dt=1.0,
        apply_noise_to="all",
        noisy_gate_indices=binding.noisy_gate_indices,
        trajectory_update="cross",
        use_crn=False,
    )
    checksum = "sha256:" + "1" * 64
    sampled = sample_krotov_fixed_map_ensemble(
        circuit,
        theta,
        None,
        truncation,
        provider,
        options,
        role="training_trajectory",
        resolved_seed=4000,
        stage_index=4,
        stage_id="final_finetune",
        stage_configuration_checksum=checksum,
        circuit_checksum=binding.content_checksum,
        provider_checksum=provider.content_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
        legacy_linear_seed=True,
        legacy_compact_replay=True,
    )
    modern = sample_krotov_fixed_map_ensemble(
        circuit,
        theta,
        None,
        truncation,
        provider,
        options,
        role="training_trajectory",
        resolved_seed=4000,
        stage_index=4,
        stage_id="final_finetune",
        stage_configuration_checksum=checksum,
        circuit_checksum=binding.content_checksum,
        provider_checksum=provider.content_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
        legacy_linear_seed=True,
    )
    modern_provider_maps = [
        noise_map
        for trajectory in modern.replay_maps()
        for noise_map in trajectory
        if noise_map.channel_id == HISTORICAL_FIXED_RATE_NOISE_ID
    ]
    assert modern_provider_maps
    assert all(noise_map.normalized for noise_map in modern_provider_maps)

    expected_maps = []
    for index in range(3):
        seed = derive_legacy_krotov_trajectory_seed(optimizer_iteration_seed=4000, trajectory_index=index)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        trajectory = forward_tjm_trajectory(
            circuit,
            theta,
            np.empty(0, dtype=np.float64),
            MPS(2),
            truncation,
            None,
            options,
            rng,
            noise_provider=provider,
        )
        assert all(not noise_map.normalization_checkpoints for noise_map in trajectory.noise_maps)
        expected_maps.append([replace(noise_map, normalized=False) for noise_map in trajectory.noise_maps])
    expected = KrotovFixedMapEnsemble(
        role="training_trajectory",
        resolved_seed=4000,
        stage_index=4,
        stage_id="final_finetune",
        stage_configuration_checksum=checksum,
        circuit_checksum=binding.content_checksum,
        provider_checksum=provider.content_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
        trajectory_maps=expected_maps,
    )
    assert sampled.content_checksum == expected.content_checksum
    assert all(not noise_map.normalized for trajectory in sampled.replay_maps() for noise_map in trajectory)


def test_v2_requires_pilot_frozen_counts_and_freezes_checkpoint_selection() -> None:
    """There is no unpiloted count default and the full candidate cadence is fixed."""
    signature = inspect.signature(build_layerwise_bmpd_crn_v2_template)
    assert signature.parameters["training_trajectory_count"].default is inspect.Parameter.empty
    assert signature.parameters["checkpoint_validation_trajectory_count"].default is inspect.Parameter.empty

    pipeline = _resolve_v2()
    final = pipeline.stages[-1]
    validation = final.checkpoint_validation
    assert final.training_noise_id == "depolarizing_1s_all"
    assert final.trajectory_update == "independent"
    assert final.sampling_policy == "crn_fixed"
    assert final.trajectory_count == 3
    assert validation.trajectory_count == 5
    assert validation.sampling_policy == "crn_fixed"
    assert validation.cadence == 10
    assert validation.selection_rule == "best_validation_fidelity"
    assert validation.tie_breaker == "earliest_iteration"
    assert [0, *range(10, final.iteration_budget + 1, 10)][-1] == 200
    assert len([0, *range(10, final.iteration_budget + 1, 10)]) == 21
    assert final.training_seed != validation.seed


def test_v2_equal_checkpoint_fidelities_select_iteration_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal validation candidates exercise the v2 earliest-iteration rule."""
    pipeline = _resolve_v2()
    final = replace(
        pipeline.stages[-1],
        iteration_budget=2,
        checkpoint_validation=replace(pipeline.stages[-1].checkpoint_validation, cadence=1),
    )
    binding = create_bmpd_circuit_binding(6, 4)
    initial = np.linspace(-0.1, 0.1, binding.circuit.num_params, dtype=np.float64)
    target = np.zeros(2**6, dtype=np.complex128)
    target[0] = 1.0

    def constant_contribution(*args: object, **kwargs: object) -> tuple[NDArray[np.float64], float, float, list[float]]:
        del args, kwargs
        return np.zeros(binding.circuit.num_params, dtype=np.float64), 0.5, 0.5, [0.5]

    def constant_metrics(*args: object, **kwargs: object) -> tuple[float, float, list[float]]:
        del args, kwargs
        return 0.5, 0.5, [0.5]

    monkeypatch.setattr(noisy_krotov_module, "noisy_state_preparation_contribution", constant_contribution)
    monkeypatch.setattr(noisy_krotov_module, "noisy_state_preparation_metrics", constant_metrics)
    result = execute_fixed_rate_krotov_stage(final, binding, target, initial)

    assert isinstance(result, NoisyKrotovStageExecution)
    candidates = [row for row in result.trace if row.checkpoint_validation_fidelity is not None]
    assert [row.global_iteration for row in candidates] == [0, 1, 2]
    assert [row.checkpoint_validation_fidelity for row in candidates] == [0.5, 0.5, 0.5]
    assert result.selected_global_iteration == 0
    assert result.checkpoint_selection is not None
    assert result.checkpoint_selection.global_iteration == 0
    np.testing.assert_array_equal(result.selected_theta, initial)
