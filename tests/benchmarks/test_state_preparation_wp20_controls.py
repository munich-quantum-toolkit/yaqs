# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Structural, identity, and isolation tests for WP20 fair controls."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache, partial
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import benchmarks.state_preparation.phase2.fair_controls as fair_controls_module
import benchmarks.state_preparation.phase2.layerwise_bmpd as layerwise_bmpd_module
from benchmarks.state_preparation.constants import NOISELESS_NOISE_ID
from benchmarks.state_preparation.noise import (
    FIXED_RATE_NOISE_DEFINITION_VERSION,
    HISTORICAL_FIXED_RATE_NOISE_ID,
)
from benchmarks.state_preparation.phase2.fair_controls import (
    FIXED_DEPTH_BMPD_CRN_METHOD_ID,
    LAYERWISE_BMPD_CROSS_CRN_METHOD_ID,
    LAYERWISE_BMPD_NOISELESS_METHOD_ID,
    LAYERWISE_BMPD_RESAMPLED_METHOD_ID,
    PHASE1_NOISELESS_CHECKPOINT_CONTROL_METHOD_ID,
    UNPRUNED_DEEP_BMPD_METHOD_ID,
    FixedDepthBMPDStageRunner,
    NativeBudgetDescriptor,
    SecondaryControlDescriptor,
    build_fixed_depth_bmpd_crn_template,
    build_independent_fixed_crn_control_template,
    build_layerwise_bmpd_cross_crn_template,
    build_layerwise_bmpd_noiseless_template,
    build_layerwise_bmpd_resampled_template,
    build_phase1_noiseless_test_control,
    build_unpruned_deep_control,
)
from benchmarks.state_preparation.phase2.layerwise_bmpd import (
    LAYERWISE_BMPD_CRN_V2_METHOD_ID,
    LayerwiseBMPDStageRunner,
    bmpd_parameter_count,
    build_layerwise_bmpd_crn_v2_template,
    create_bmpd_circuit_binding,
)
from benchmarks.state_preparation.phase2.noisy_krotov import (
    NoisyKrotovStageExecution,
    NoisyKrotovStageFailure,
    execute_fixed_rate_krotov_stage,
)
from benchmarks.state_preparation.phase2.pipeline import (
    LEGACY_LAYERWISE_SEED_BINDINGS,
    TrainingPipelineTemplate,
    TrainingStageTemplate,
)
from benchmarks.state_preparation.phase2.protocol import load_initial_preregistration
from benchmarks.state_preparation.phase2.targets import (
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from benchmarks.state_preparation.phase2.noisy_krotov import NoisyKrotovCircuitBinding
    from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineConfig
    from benchmarks.state_preparation.phase2.targets import MaterializedTarget, TargetPopulationManifest

_MASTER_ENTROPY = bytes(reversed(range(32)))


@lru_cache(maxsize=1)
def _development_manifest() -> TargetPopulationManifest:
    """Return a genuine q6 manifest used to resolve deterministic seed domains.

    Returns:
        A seed-bearing Phase-II development manifest.
    """
    preregistration = load_initial_preregistration()
    config = build_target_population_config(
        preregistration,
        "development",
        role_master_entropy_commitment=role_master_entropy_commitment(_MASTER_ENTROPY),
        population_scope="primary_q6",
    )
    return create_target_population_manifest(config, preregistration, _MASTER_ENTROPY)


def _resolve(template: TrainingPipelineTemplate, *, optimization_seed: int = 91) -> TrainingPipelineConfig:
    """Resolve one primary q6 template against the same target and outer block.

    Returns:
        The deterministic target-bound pipeline.
    """
    manifest = _development_manifest()
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
        optimization_block_id="wp20_common_paired_block",
        optimization_seed=optimization_seed,
        data_role="development",
    )


def _runtime_seeds(pipeline: TrainingPipelineConfig) -> set[int]:
    """Return every active concrete seed in one resolved pipeline."""
    return {
        seed
        for stage in pipeline.stages
        for seed in (
            stage.initialization_seed,
            stage.optimizer_seed,
            stage.training_seed,
            stage.checkpoint_validation.seed,
        )
        if seed is not None
    }


def _assert_global_randomstate_unchanged(before: object, after: object) -> None:
    """Assert that template construction and resolution never consume global RNG."""
    assert isinstance(before, tuple)
    assert isinstance(after, tuple)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def _capture_stage_execution(
    captured: dict[str, object],
    marker: NoisyKrotovStageExecution,
    *args: object,
    **kwargs: object,
) -> NoisyKrotovStageExecution:
    """Capture a stage-adapter call and return its opaque test marker.

    Returns:
        The supplied opaque execution marker.
    """
    captured["args"] = args
    captured.update(kwargs)
    return marker


def _return_target_identity(identity: dict[str, object], _target: object) -> dict[str, object]:
    """Return a caller-bound target identity for a runner contract test.

    Returns:
        The supplied target identity mapping.
    """
    return identity


def test_noiseless_control_is_the_exact_v2_treatment_match() -> None:
    """Noisy training alone differs while growth, work, and validation remain matched."""
    v2 = build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=7,
        checkpoint_validation_trajectory_count=11,
    )
    noiseless = build_layerwise_bmpd_noiseless_template(
        checkpoint_validation_trajectory_count=11,
    )

    assert v2.method_id == LAYERWISE_BMPD_CRN_V2_METHOD_ID
    assert noiseless.method_id == LAYERWISE_BMPD_NOISELESS_METHOD_ID
    assert noiseless.stages[:-1] == v2.stages[:-1]
    assert [stage.stage_id for stage in noiseless.stages] == [
        "grow_d1",
        "grow_d2",
        "grow_d3",
        "grow_d4",
        "final_finetune",
    ]
    assert [stage.stage_policy["iteration_budget"] for stage in noiseless.stages] == [100, 100, 100, 100, 200]
    final = noiseless.stages[-1]
    assert final.stage_policy["training_noise_id"] == NOISELESS_NOISE_ID
    assert final.stage_policy["trajectory_count"] == 0
    assert final.stage_policy["trajectory_update"] is None
    assert final.stage_policy["sampling_policy"] == "none"
    assert final.seed_bindings["training"] is None
    assert (
        final.stage_policy["checkpoint_validation_policy"] == v2.stages[-1].stage_policy["checkpoint_validation_policy"]
    )
    assert noiseless.matching_projection_checksum == v2.matching_projection_checksum
    assert noiseless.configuration_checksum != v2.configuration_checksum


def test_noisy_layerwise_controls_change_only_the_declared_training_policy() -> None:
    """Fixed CRN, resampling, and cross CRN have exact and distinct semantics."""
    fixed = build_independent_fixed_crn_control_template(
        training_trajectory_count=5,
        checkpoint_validation_trajectory_count=9,
    )
    resampled = build_layerwise_bmpd_resampled_template(
        training_trajectory_count=5,
        checkpoint_validation_trajectory_count=9,
    )
    cross = build_layerwise_bmpd_cross_crn_template(
        training_trajectory_count=5,
        checkpoint_validation_trajectory_count=9,
    )

    assert fixed.method_id == LAYERWISE_BMPD_CRN_V2_METHOD_ID
    assert resampled.method_id == LAYERWISE_BMPD_RESAMPLED_METHOD_ID
    assert cross.method_id == LAYERWISE_BMPD_CROSS_CRN_METHOD_ID
    assert len(fixed.stages) == len(resampled.stages) == len(cross.stages) == 5
    assert [stage.stage_policy["iteration_budget"] for stage in resampled.stages] == [100, 100, 100, 100, 200]
    assert [stage.stage_policy["output_parameter_count"] for stage in resampled.stages] == [63, 108, 153, 198, 198]

    fixed_final = fixed.stages[-1]
    resampled_final = resampled.stages[-1]
    cross_final = cross.stages[-1]
    for final in (fixed_final, resampled_final, cross_final):
        validation = cast("Mapping[str, object]", final.stage_policy["checkpoint_validation_policy"])
        assert final.stage_policy["training_noise_id"] == "depolarizing_1s_all"
        assert final.stage_policy["noise_definition_version"] == FIXED_RATE_NOISE_DEFINITION_VERSION
        assert final.stage_policy["trajectory_count"] == 5
        assert validation["trajectory_count"] == 9
        assert validation["sampling_policy"] == "crn_fixed"
    assert (fixed_final.stage_policy["trajectory_update"], fixed_final.stage_policy["sampling_policy"]) == (
        "independent",
        "crn_fixed",
    )
    assert (
        resampled_final.stage_policy["trajectory_update"],
        resampled_final.stage_policy["sampling_policy"],
    ) == ("independent", "resampled")
    assert (cross_final.stage_policy["trajectory_update"], cross_final.stage_policy["sampling_policy"]) == (
        "cross",
        "crn_fixed",
    )
    assert len({fixed.configuration_checksum, resampled.configuration_checksum, cross.configuration_checksum}) == 3


def test_fixed_depth_control_is_direct_and_requires_an_explicit_budget() -> None:
    """The fixed-depth candidate receives exactly the caller-frozen noisy work."""
    template = build_fixed_depth_bmpd_crn_template(
        iteration_budget=600,
        training_trajectory_count=7,
        checkpoint_validation_trajectory_count=13,
    )

    assert template.method_id == FIXED_DEPTH_BMPD_CRN_METHOD_ID
    assert template.target_scope_id == "primary_q6"
    assert len(template.stages) == 1
    stage = template.stages[0]
    assert stage.stage_policy["stage_id"] == "direct_depth4_noisy_training"
    assert stage.stage_policy["input_topology_id"] is None
    assert stage.stage_policy["input_parameter_count"] == 0
    assert stage.stage_policy["output_topology_id"] == "bmpd_q6_d4"
    assert stage.stage_policy["output_parameter_count"] == 198
    assert stage.stage_policy["parameter_transfer_rule"] == "initialize_random_normal"
    assert stage.stage_policy["iteration_budget"] == 600
    assert stage.stage_policy["trajectory_count"] == 7
    assert stage.stage_policy["trajectory_update"] == "independent"
    assert stage.stage_policy["sampling_policy"] == "crn_fixed"
    validation = cast("Mapping[str, object]", stage.stage_policy["checkpoint_validation_policy"])
    assert validation["trajectory_count"] == 13

    with pytest.raises(ValueError, match="iteration_budget"):
        build_fixed_depth_bmpd_crn_template(
            iteration_budget=0,
            training_trajectory_count=7,
            checkpoint_validation_trajectory_count=13,
        )


def test_fixed_depth_runner_executes_the_topology_bound_depth_four_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primary direct baseline binds q6 depth four and complete native statistics."""
    pipeline = _resolve(
        build_fixed_depth_bmpd_crn_template(
            iteration_budget=2,
            training_trajectory_count=2,
            checkpoint_validation_trajectory_count=3,
        ),
    )
    target = cast("MaterializedTarget", object())
    identity: dict[str, object] = {
        "target_instance_id": pipeline.target_instance_id,
        "target_instance_spec_checksum": pipeline.target_instance_spec_checksum,
        "target_manifest_checksum": pipeline.target_population_manifest_checksum,
        "family_id": pipeline.target_family_id,
        "stratum_id": pipeline.target_stratum_id,
        "qubit_count": pipeline.qubit_count,
    }
    captured: dict[str, object] = {}
    marker = cast("NoisyKrotovStageExecution", object())
    monkeypatch.setattr(
        fair_controls_module,
        "_fixed_depth_target_identity",
        partial(_return_target_identity, identity),
    )
    monkeypatch.setattr(
        fair_controls_module,
        "execute_fixed_rate_krotov_stage",
        partial(_capture_stage_execution, captured, marker),
    )

    runner = FixedDepthBMPDStageRunner(pipeline, target)
    stage = pipeline.stages[0]
    result = runner(stage, None)

    assert result is marker
    args = cast("tuple[object, ...]", captured["args"])
    assert args[0] == stage
    binding = cast("NoisyKrotovCircuitBinding", args[1])
    assert binding.topology_id == "bmpd_q6_d4"
    assert binding.circuit.num_params == 198
    assert args[2] is target
    initial_theta = cast("np.ndarray", args[3])
    assert initial_theta.shape == (198,)
    assert np.all(np.isfinite(initial_theta))
    assert captured["compatibility_method_id"] is None

    statistics = runner.circuit_statistics(stage)
    assert statistics["topology_id"] == "bmpd_q6_d4"
    assert statistics["bmpd_depth"] == 4
    assert statistics["native_two_qubit_gate_count"] == 60
    assert statistics["native_two_qubit_gates_per_chain_edge"] == [12, 12, 12, 12, 12]
    assert isinstance(statistics["circuit_resource_metrics"], dict)


def test_fixed_depth_runner_rejects_a_same_named_depth_five_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The method identity cannot be reused with a different direct topology depth."""
    original = build_fixed_depth_bmpd_crn_template(
        iteration_budget=2,
        training_trajectory_count=2,
        checkpoint_validation_trajectory_count=3,
    )
    policy = dict(original.stages[0].stage_policy)
    policy["output_topology_id"] = "bmpd_q6_d5"
    policy["output_parameter_count"] = bmpd_parameter_count(6, 5)
    altered_stage = TrainingStageTemplate(
        stage_policy=policy,
        seed_bindings=dict(original.stages[0].seed_bindings),
    )
    pipeline = _resolve(replace(original, stages=(altered_stage,)))
    identity: dict[str, object] = {
        "target_instance_id": pipeline.target_instance_id,
        "target_instance_spec_checksum": pipeline.target_instance_spec_checksum,
        "target_manifest_checksum": pipeline.target_population_manifest_checksum,
        "family_id": pipeline.target_family_id,
        "stratum_id": pipeline.target_stratum_id,
        "qubit_count": pipeline.qubit_count,
    }
    monkeypatch.setattr(
        fair_controls_module,
        "_fixed_depth_target_identity",
        partial(_return_target_identity, identity),
    )

    with pytest.raises(ValueError, match="q6 depth-four"):
        FixedDepthBMPDStageRunner(pipeline, cast("MaterializedTarget", object()))


def test_pilot_trajectory_counts_are_mandatory_and_positive() -> None:
    """No control silently invents a training or validation trajectory budget."""
    with pytest.raises(ValueError, match="checkpoint_validation_trajectory_count"):
        build_layerwise_bmpd_noiseless_template(checkpoint_validation_trajectory_count=0)
    with pytest.raises(ValueError, match="training_trajectory_count"):
        build_layerwise_bmpd_resampled_template(
            training_trajectory_count=0,
            checkpoint_validation_trajectory_count=3,
        )
    with pytest.raises(ValueError, match="checkpoint_validation_trajectory_count"):
        build_layerwise_bmpd_cross_crn_template(
            training_trajectory_count=3,
            checkpoint_validation_trajectory_count=0,
        )


def test_control_resolution_uses_disjoint_hash_domains_without_global_rng() -> None:
    """Matched streams align only where intended; other methods share no runtime seed."""
    before = np.random.get_state()  # noqa: NPY002
    v2 = _resolve(
        build_independent_fixed_crn_control_template(
            training_trajectory_count=3,
            checkpoint_validation_trajectory_count=5,
        ),
    )
    noiseless = _resolve(
        build_layerwise_bmpd_noiseless_template(
            checkpoint_validation_trajectory_count=5,
        ),
    )
    resampled = _resolve(
        build_layerwise_bmpd_resampled_template(
            training_trajectory_count=3,
            checkpoint_validation_trajectory_count=5,
        ),
    )
    cross = _resolve(
        build_layerwise_bmpd_cross_crn_template(
            training_trajectory_count=3,
            checkpoint_validation_trajectory_count=5,
        ),
    )
    after = np.random.get_state()  # noqa: NPY002

    for noisy_stage, control_stage in zip(v2.stages, noiseless.stages, strict=True):
        assert noisy_stage.initialization_seed == control_stage.initialization_seed
        assert noisy_stage.optimizer_seed == control_stage.optimizer_seed
        assert noisy_stage.checkpoint_validation.seed == control_stage.checkpoint_validation.seed
    assert v2.stages[-1].training_seed is not None
    assert noiseless.stages[-1].training_seed is None
    assert _runtime_seeds(v2).isdisjoint(_runtime_seeds(resampled))
    assert _runtime_seeds(v2).isdisjoint(_runtime_seeds(cross))
    assert _runtime_seeds(resampled).isdisjoint(_runtime_seeds(cross))
    _assert_global_randomstate_unchanged(before, after)


def test_cross_crn_control_cannot_enable_legacy_compatibility() -> None:
    """Modern cross updates use standard noise, normalized replay, and no legacy seed opt-in."""
    pipeline = _resolve(
        build_layerwise_bmpd_cross_crn_template(
            training_trajectory_count=2,
            checkpoint_validation_trajectory_count=3,
        ),
    )
    final = pipeline.stages[-1]
    active_bindings = {
        value for stage in pipeline.template.stages for value in stage.seed_bindings.values() if value is not None
    }
    assert final.training_noise_id != HISTORICAL_FIXED_RATE_NOISE_ID
    assert final.training_noise_id == "depolarizing_1s_all"
    assert final.trajectory_update == "cross"
    assert final.sampling_policy == "crn_fixed"
    assert active_bindings.isdisjoint(LEGACY_LAYERWISE_SEED_BINDINGS)
    assert final.optimizer_seed != final.training_seed

    binding = create_bmpd_circuit_binding(6, 4)
    rejected = execute_fixed_rate_krotov_stage(
        final,
        binding,
        np.eye(1, 2**6, dtype=np.complex128)[0],
        np.zeros(binding.circuit.num_params, dtype=np.float64),
        compatibility_method_id=LAYERWISE_BMPD_CROSS_CRN_METHOD_ID,
    )
    assert isinstance(rejected, NoisyKrotovStageFailure)
    assert rejected.phase == "validation"
    assert "isolated WP19 legacy method" in rejected.message


def test_layerwise_runner_accepts_modern_controls_but_only_legacy_identity_opts_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every modern layerwise template is executable without the legacy identity switch."""
    for template in (
        build_layerwise_bmpd_noiseless_template(checkpoint_validation_trajectory_count=3),
        build_layerwise_bmpd_resampled_template(
            training_trajectory_count=2,
            checkpoint_validation_trajectory_count=3,
        ),
        build_layerwise_bmpd_cross_crn_template(
            training_trajectory_count=2,
            checkpoint_validation_trajectory_count=3,
        ),
    ):
        pipeline = _resolve(template)
        target = cast("MaterializedTarget", object())
        target_identity: dict[str, object] = {
            "target_instance_id": pipeline.target_instance_id,
            "target_instance_spec_checksum": pipeline.target_instance_spec_checksum,
            "target_manifest_checksum": pipeline.target_population_manifest_checksum,
            "family_id": pipeline.target_family_id,
            "stratum_id": pipeline.target_stratum_id,
            "qubit_count": pipeline.qubit_count,
        }
        captured: dict[str, object] = {}
        marker = cast("NoisyKrotovStageExecution", object())

        monkeypatch.setattr(
            layerwise_bmpd_module,
            "_target_identity",
            partial(_return_target_identity, target_identity),
        )
        monkeypatch.setattr(
            layerwise_bmpd_module,
            "execute_fixed_rate_krotov_stage",
            partial(_capture_stage_execution, captured, marker),
        )
        runner = LayerwiseBMPDStageRunner(pipeline, target)
        result = runner(pipeline.stages[0], None)

        assert result is marker
        assert captured["compatibility_method_id"] is None
        statistics = runner.circuit_statistics(pipeline.stages[0])
        assert statistics["native_two_qubit_gates_per_chain_edge"] == [3, 3, 3, 3, 3]
        assert isinstance(statistics["circuit_resource_metrics"], dict)


def test_secondary_controls_have_distinct_ids_and_truthful_native_budget_status() -> None:
    """Exact-cap and deeper circuits remain separate and cannot enter sealed roles."""
    phase1 = build_phase1_noiseless_test_control(depth=4, iteration_budget=100)
    matched = build_unpruned_deep_control(depth=4, iteration_budget=100)
    unmatched = build_unpruned_deep_control(depth=8, iteration_budget=100)

    assert phase1.template.method_id == PHASE1_NOISELESS_CHECKPOINT_CONTROL_METHOD_ID
    assert phase1.template.target_scope_id == "phase1_fixture"
    assert phase1.template.stages[0].stage_policy["training_noise_id"] == NOISELESS_NOISE_ID
    assert phase1.evaluation_policy_id == "fresh_independent_standard_noise_v1"
    assert matched.template.method_id == UNPRUNED_DEEP_BMPD_METHOD_ID
    assert matched.resource.match_status == "exact_match"
    assert matched.resource.attained_per_chain_edge == pytest.approx(12.0)
    assert matched.resource.residual_gap == pytest.approx(0.0)
    assert matched.resource.resource_excess == pytest.approx(0.0)
    assert unmatched.resource.match_status == "above_cap_unmatched"
    assert unmatched.resource.attained_per_chain_edge == pytest.approx(24.0)
    assert unmatched.resource.residual_gap == pytest.approx(0.0)
    assert unmatched.resource.resource_excess == pytest.approx(12.0)
    assert matched.template.configuration_checksum != unmatched.template.configuration_checksum
    assert len({phase1.content_checksum, matched.content_checksum, unmatched.content_checksum}) == 3

    with pytest.raises(ValueError, match="compiler-derived from its declared depth"):
        SecondaryControlDescriptor(
            template=unmatched.template,
            resource=NativeBudgetDescriptor(12.0),
            depth=unmatched.depth,
            iteration_budget=unmatched.iteration_budget,
        )

    preregistered = {method["method_id"] for method in load_initial_preregistration().candidate_methods}
    assert phase1.template.method_id not in preregistered
    assert matched.template.method_id not in preregistered
    for control in (phase1, matched, unmatched):
        assert control.screening_eligible is False
        with pytest.raises(ValueError, match="Secondary-only control"):
            control.require_data_role("screening_selection")
        with pytest.raises(ValueError, match="Secondary-only control"):
            control.require_data_role("confirmatory")
        control.require_data_role("development")


def test_every_control_round_trips_and_has_a_separate_training_identity() -> None:
    """Control method, treatment, work, and resource choices are identity-bearing."""
    templates = (
        build_independent_fixed_crn_control_template(
            training_trajectory_count=3,
            checkpoint_validation_trajectory_count=5,
        ),
        build_layerwise_bmpd_noiseless_template(checkpoint_validation_trajectory_count=5),
        build_fixed_depth_bmpd_crn_template(
            iteration_budget=600,
            training_trajectory_count=3,
            checkpoint_validation_trajectory_count=5,
        ),
        build_layerwise_bmpd_resampled_template(
            training_trajectory_count=3,
            checkpoint_validation_trajectory_count=5,
        ),
        build_layerwise_bmpd_cross_crn_template(
            training_trajectory_count=3,
            checkpoint_validation_trajectory_count=5,
        ),
        build_phase1_noiseless_test_control(depth=4, iteration_budget=100).template,
        build_unpruned_deep_control(depth=8, iteration_budget=100).template,
    )

    assert len({template.configuration_checksum for template in templates}) == len(templates)
    assert len({_resolve(template).training_id for template in templates[:5]}) == 5
    for template in templates:
        assert TrainingPipelineTemplate.from_json(template.to_json()) == template
