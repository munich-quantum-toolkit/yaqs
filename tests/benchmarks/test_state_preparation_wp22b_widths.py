# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Golden and structural tests for WP22B q6/q12 BMPD builders."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import benchmarks.state_preparation.phase2.targets as targets_module
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.competitor_optimizers import (
    build_parameter_shift_adam_fixed_template,
    build_parameter_shift_adam_layerwise_template,
    build_spsa_fixed_template,
    build_spsa_layerwise_template,
)
from benchmarks.state_preparation.phase2.fair_controls import (
    FixedDepthBMPDStageRunner,
    build_fixed_depth_bmpd_crn_template,
    build_layerwise_bmpd_noiseless_template,
)
from benchmarks.state_preparation.phase2.layerwise_bmpd import (
    LayerwiseBMPDStageRunner,
    bmpd_parameter_count,
    bmpd_topology_id,
    build_layerwise_bmpd_crn_v2_template,
    create_bmpd_circuit_binding,
)
from benchmarks.state_preparation.phase2.noisy_krotov import (
    NoisyKrotovStageExecution,
    execute_fixed_rate_krotov_stage,
)
from benchmarks.state_preparation.phase2.protocol import load_initial_preregistration
from benchmarks.state_preparation.phase2.targets import (
    authorize_target_materialization,
    build_target_population_config,
    create_target_population_manifest,
    materialize_target_population,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.wp20_resources import measure_circuit_resources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from benchmarks.state_preparation.phase2.pipeline import (
        TrainingPipelineConfig,
        TrainingPipelineTemplate,
    )
    from benchmarks.state_preparation.phase2.targets import MaterializedTarget, TargetPopulationManifest

_MASTER_ENTROPY = bytes(range(32))
_Q6_DOCUMENT_GOLDENS = {
    False: (
        "sha256:92756f9b6bdac7347fc8bbc1fe3f5c85cc6aaedde9e0b383f09436f31ff170c4",
        "sha256:191fbce9cc4ac3e323bb27dbc1184502947b8fe833bec8c236c7ead7d02958dd",
        "sha256:3c6285f1fe93667956c6edc6612d5d095075cc0a9ddd976ddf7d4abbae6d4a04",
        "sha256:bfcb0f564d095c4c7034f0517a19544feef8d85282cbf21b66783d9c9805a8c5",
        "sha256:7ad71f919639c76b263c41a19532431dbdfc50b6566c75dae937e83f8a0c0621",
    ),
    True: (
        "sha256:aab5024538ad72eabb21ad596d1e1df0e15be540220c08c473b278cae3ad84bf",
        "sha256:83a7202ca81da33ed12311d8e1befb84644d47442046e788fab4590f6666424e",
        "sha256:955668827ec78e53e250c1d7feefe8031dbc0f0640c000d14b9dc5a1f0034222",
        "sha256:4b7868805f89f2ffa19b180b5eb5b41643118d659b0976c35014d5f6d711681f",
        "sha256:966e52e773e54cb9201c166cb64c1f482c1137d3327c52b90468df049b12f00d",
    ),
}
_Q6_CONFIGURATION_GOLDENS = {
    False: (
        "sha256:e5a18b2b64e3beb2bf9f99be21c17c540771c4d2f7eee04cf9b11c148bc43ab7",
        "sha256:80340a35aabea730da83c1b88f6fc4dcf2c8c09654b9406b558061bc0c039b6c",
        "sha256:c1314dc24b3aa1b021be2f9e26cde15da09c3564f8239c496e4051cd16c01af5",
        "sha256:c3045c442b7c91962e22206d4a95cf62303d384421cb332a18fb81c1a6d52f25",
        "sha256:074328c34fc8a7b1a8124e870b9e7828f2dbf5b0cf2d67dbdbc79b83d86483f8",
    ),
    True: (
        "sha256:cbaf747d3165d2fd7c0bacf1fe9a1398e6ee6402f9a0c8cbc4d9a446d02b328c",
        "sha256:a5fbdca2f3801677d8649892199c592a5c14f22cc4665eb4ec93e10bb202b604",
        "sha256:72d933b378bebb428b8b3121b6a8c8fb2e10a751474c52088b2917598b44d78c",
        "sha256:12605ca38aaf9383b794d0131b988687b3c95844721e2362833e98ee2526db17",
        "sha256:77c7bfe65c6351660adc17b52e2f1fba9f090872fe84a71e69fa54ba6a4e025b",
    ),
}
_Q12_DOCUMENT_GOLDENS = (
    "sha256:2834dd6e2b9e98e316959d56089ef959799fe7f9a22a8b436f5348f92abc4dad",
    "sha256:72d8a3fa786a56715ea598aa4450d572bdc2478ca340b55190ab1878b9935b7f",
    "sha256:fdff6668c5632ed6b0173c29125f376da9858ba92f82c55f5e19ea17562ab8a7",
    "sha256:fdbad746b81df103ddbd2fdcd527f95645486e70f5395e007860bdecbc2b45ff",
    "sha256:2ecee10d93a224551754256377f36e50ff889f98b372af1290307ca08bcc08de",
)
_Q12_CONFIGURATION_GOLDENS = (
    "sha256:4c1aadbad1cc9d11df0c5a77544f63708c321157580242ab2eb6411a2756914f",
    "sha256:d4a8a9902550c86765ac733009bdb6e300893369395fd9b9db5c411c94c8ebb8",
    "sha256:ac2f385ee291973ffa6575665d90d606f316a0c95e7435ff8789a3c8bbdc4773",
    "sha256:f734bd38c3fd53e5cafbd4a48aa339fef8fd26adb08f81da4d14f8f17b4642eb",
    "sha256:31908f4f0d533a34c1b8e7e79822088e84b3f05456fb35056a84c631f1fc0c2e",
)
_Q6_FIXED_COMPETITOR_GOLDENS = {
    False: (
        (
            "sha256:956c82b4d38e2626a644e13177aefe49adc724090a3845e3978d47f97a2605e5",
            "sha256:8106b5f7f2ba3db5e606aad7ea90e54d05773b5e8b37e604d8ebce5ab2dbf875",
        ),
        (
            "sha256:565262d959b7d82d820a0ddce61251f49a0fbbe36c391b7230126f1f5b4e0038",
            "sha256:0d0e9fc3e6267588dcde39f5e848f1d01ea1803eb11a77369f91e6cf586e9745",
        ),
    ),
    True: (
        (
            "sha256:2cb804c70705bb3d7659299383709bd85d78e13969d36a42b6200707b47e1235",
            "sha256:e152d40fb133d794e4c86f64c4fb770d2a099c9dff4f717d3b50a3879a61ae51",
        ),
        (
            "sha256:e9f13c5e6ef3f76fff4967d0686e60fd0b741f865f80848ab64ed976d25f2b4a",
            "sha256:2b9371646e21191d992ae6f4606d00b1067358350df92065c562e070d3cf91bc",
        ),
    ),
}


def _paper_templates(qubit_count: int, *, smoke: bool = False) -> tuple[TrainingPipelineTemplate, ...]:
    """Build the five width-generalized pipeline families.

    Returns:
        Layerwise, noiseless, fixed-depth, Adam, and SPSA templates.
    """
    training_count = 1 if smoke else 8
    validation_count = 1 if smoke else 256
    iteration_budget = 1 if smoke else 200
    return (
        build_layerwise_bmpd_crn_v2_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
            qubit_count=qubit_count,
        ),
        build_layerwise_bmpd_noiseless_template(
            checkpoint_validation_trajectory_count=validation_count,
            qubit_count=qubit_count,
        ),
        build_fixed_depth_bmpd_crn_template(
            iteration_budget=iteration_budget,
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
            qubit_count=qubit_count,
        ),
        build_parameter_shift_adam_layerwise_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
            qubit_count=qubit_count,
        ),
        build_spsa_layerwise_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
            qubit_count=qubit_count,
        ),
    )


def _default_q6_templates(*, smoke: bool) -> tuple[TrainingPipelineTemplate, ...]:
    """Build the same five families without passing the new width argument.

    Returns:
        The backward-compatible default q6 templates.
    """
    training_count = 1 if smoke else 8
    validation_count = 1 if smoke else 256
    iteration_budget = 1 if smoke else 200
    return (
        build_layerwise_bmpd_crn_v2_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        ),
        build_layerwise_bmpd_noiseless_template(
            checkpoint_validation_trajectory_count=validation_count,
        ),
        build_fixed_depth_bmpd_crn_template(
            iteration_budget=iteration_budget,
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        ),
        build_parameter_shift_adam_layerwise_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        ),
        build_spsa_layerwise_template(
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        ),
    )


@pytest.mark.parametrize("profile", ["production", "smoke"])
def test_q6_defaults_remain_byte_identical_to_the_pre_wp22b_goldens(profile: str) -> None:
    """Adding an explicit width cannot alter production or smoke q6 bytes."""
    smoke = profile == "smoke"
    defaults = _default_q6_templates(smoke=smoke)
    explicit = _paper_templates(6, smoke=smoke)
    assert [template.to_dict() for template in defaults] == [template.to_dict() for template in explicit]
    assert tuple(canonical_checksum(template.to_dict()) for template in defaults) == _Q6_DOCUMENT_GOLDENS[smoke]
    assert tuple(template.configuration_checksum for template in defaults) == _Q6_CONFIGURATION_GOLDENS[smoke]


def test_q12_projection_changes_only_width_derived_pipeline_fields() -> None:
    """q12 retains q6 treatment semantics while deriving topology and size."""
    q6_templates = _paper_templates(6)
    q12_templates = _paper_templates(12)
    assert tuple(canonical_checksum(template.to_dict()) for template in q12_templates) == _Q12_DOCUMENT_GOLDENS
    assert tuple(template.configuration_checksum for template in q12_templates) == _Q12_CONFIGURATION_GOLDENS

    for q6, q12 in zip(q6_templates, q12_templates, strict=True):
        assert q12.template_id == f"{q6.template_id}_q12_projection"
        assert (q6.target_scope_id, q12.target_scope_id) == ("primary_q6", "secondary_q12")
        assert q12.method_id == q6.method_id
        assert q12.method_version == q6.method_version
        assert q12.resource_stratum_id == q6.resource_stratum_id
        assert q12.seed_domains == q6.seed_domains
        assert q12.final_materialization_policy == q6.final_materialization_policy
        assert len(q12.stages) == len(q6.stages)
        for q6_stage, q12_stage in zip(q6.stages, q12.stages, strict=True):
            q6_policy = dict(q6_stage.stage_policy)
            q12_policy = dict(q12_stage.stage_policy)
            q6_output_depth = int(cast("str", q6_policy["output_topology_id"]).rsplit("_d", maxsplit=1)[1])
            assert q12_policy["output_topology_id"] == bmpd_topology_id(12, q6_output_depth)
            assert q12_policy["output_parameter_count"] == bmpd_parameter_count(12, q6_output_depth)
            if q6_policy["input_topology_id"] is not None:
                q6_input_depth = int(cast("str", q6_policy["input_topology_id"]).rsplit("_d", maxsplit=1)[1])
                assert q12_policy["input_topology_id"] == bmpd_topology_id(12, q6_input_depth)
                assert q12_policy["input_parameter_count"] == bmpd_parameter_count(12, q6_input_depth)
            for field_name in (
                "input_topology_id",
                "input_parameter_count",
                "output_topology_id",
                "output_parameter_count",
            ):
                q6_policy.pop(field_name)
                q12_policy.pop(field_name)
            assert q12_policy == q6_policy
            assert q12_stage.seed_bindings == q6_stage.seed_bindings


def test_q12_fixed_competitor_builders_preserve_optimizer_policies() -> None:
    """The fixed Adam/SPSA projections alter width-derived fields only."""
    for builder in (build_parameter_shift_adam_fixed_template, build_spsa_fixed_template):
        q6 = builder(
            iteration_budget=7,
            training_trajectory_count=2,
            checkpoint_validation_trajectory_count=3,
            qubit_count=6,
        )
        q12 = builder(
            iteration_budget=7,
            training_trajectory_count=2,
            checkpoint_validation_trajectory_count=3,
            qubit_count=12,
        )
        assert q12.target_scope_id == "secondary_q12"
        assert q12.template_id == f"{q6.template_id}_q12_projection"
        assert q12.method_id == q6.method_id
        assert q12.method_version == q6.method_version
        assert q12.resource_stratum_id == q6.resource_stratum_id
        assert q12.seed_domains == q6.seed_domains
        assert q12.final_materialization_policy == q6.final_materialization_policy
        q6_policy = dict(q6.stages[0].stage_policy)
        q12_policy = dict(q12.stages[0].stage_policy)
        assert q12_policy["output_topology_id"] == "bmpd_q12_d4"
        assert q12_policy["output_parameter_count"] == 432
        for field_name in ("output_topology_id", "output_parameter_count"):
            q6_policy.pop(field_name)
            q12_policy.pop(field_name)
        assert q12_policy == q6_policy
        assert q12.stages[0].seed_bindings == q6.stages[0].seed_bindings


@pytest.mark.parametrize("profile", ["production", "smoke"])
def test_fixed_competitor_q6_defaults_retain_pre_wp22b_bytes(profile: str) -> None:
    """The generalized exploratory Adam/SPSA defaults also remain unchanged."""
    smoke = profile == "smoke"
    training_count = 1 if smoke else 8
    validation_count = 1 if smoke else 256
    iteration_budget = 1 if smoke else 200
    for builder, golden in zip(
        (build_parameter_shift_adam_fixed_template, build_spsa_fixed_template),
        _Q6_FIXED_COMPETITOR_GOLDENS[smoke],
        strict=True,
    ):
        default = builder(
            iteration_budget=iteration_budget,
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
        )
        explicit = builder(
            iteration_budget=iteration_budget,
            training_trajectory_count=training_count,
            checkpoint_validation_trajectory_count=validation_count,
            qubit_count=6,
        )
        assert default.to_dict() == explicit.to_dict()
        assert (canonical_checksum(default.to_dict()), default.configuration_checksum) == golden


@pytest.mark.parametrize("qubit_count", [True, 5, 8, 13])
def test_width_generalized_builders_reject_unregistered_widths(qubit_count: int) -> None:
    """No builder can silently create an unregistered publication width."""
    with pytest.raises(ValueError, match="exactly 6 or 12"):
        build_layerwise_bmpd_crn_v2_template(
            training_trajectory_count=1,
            checkpoint_validation_trajectory_count=1,
            qubit_count=qubit_count,
        )
    with pytest.raises(ValueError, match="exactly 6 or 12"):
        build_layerwise_bmpd_noiseless_template(
            checkpoint_validation_trajectory_count=1,
            qubit_count=qubit_count,
        )
    with pytest.raises(ValueError, match="exactly 6 or 12"):
        build_fixed_depth_bmpd_crn_template(
            iteration_budget=1,
            training_trajectory_count=1,
            checkpoint_validation_trajectory_count=1,
            qubit_count=qubit_count,
        )
    for builder in (build_parameter_shift_adam_layerwise_template, build_spsa_layerwise_template):
        with pytest.raises(ValueError, match="exactly 6 or 12"):
            builder(
                training_trajectory_count=1,
                checkpoint_validation_trajectory_count=1,
                qubit_count=qubit_count,
            )
    for builder in (build_parameter_shift_adam_fixed_template, build_spsa_fixed_template):
        with pytest.raises(ValueError, match="exactly 6 or 12"):
            builder(
                iteration_budget=1,
                training_trajectory_count=1,
                checkpoint_validation_trajectory_count=1,
                qubit_count=qubit_count,
            )


def test_q12_circuit_derives_parameter_and_uniform_per_edge_resources() -> None:
    """The native q12 circuit retains the q6 site-major and per-edge rules."""
    for qubit_count in (6, 12):
        binding = create_bmpd_circuit_binding(qubit_count, 4)
        assert binding.circuit.num_params == bmpd_parameter_count(qubit_count, 4)
        initial_layer = tuple((gate.name, gate.sites) for gate in binding.circuit.gates[: 3 * qubit_count])
        assert initial_layer == tuple((name, (site,)) for site in range(qubit_count) for name in ("rz", "ry", "rz"))
        resources = measure_circuit_resources(binding.circuit)
        assert resources.logical_two_qubit_gates == 12 * (qubit_count - 1)
        assert resources.native_two_qubit_gates_per_chain_edge == (12,) * (qubit_count - 1)


def _secondary_q12_manifest(monkeypatch: pytest.MonkeyPatch) -> TargetPopulationManifest:
    """Build a deterministic q12 manifest without dense TFIM diagonalization.

    Returns:
        A genuine checksum-bearing secondary-q12 target manifest.
    """

    def cheap_tfim_parameters(
        _master: bytes,
        _target_instance_id: str,
        stratum_id: str,
        qubit_count: int,
    ) -> dict[str, object]:
        """Return shape-valid q12 TFIM spectral metadata."""
        ratio = {"ferromagnetic": 0.5, "critical": 1.0, "paramagnetic": 1.5}[stratum_id]
        return {
            "attempt_index": 0,
            "couplings": [1.0] * (qubit_count - 1),
            "fields": [ratio] * qubit_count,
            "ground_energy": -float(qubit_count),
            "ground_state_gap": 1.0,
            "gap_threshold": 1e-10 * float(qubit_count),
            "spectral_norm": float(qubit_count),
        }

    monkeypatch.setattr(targets_module, "_tfim_parameter_record", cheap_tfim_parameters)
    preregistration = load_initial_preregistration()
    config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(_MASTER_ENTROPY),
        population_scope="secondary_q12",
    )
    return create_target_population_manifest(config, preregistration, _MASTER_ENTROPY)


def _resolve_secondary(
    template: TrainingPipelineTemplate, manifest: TargetPopulationManifest
) -> TrainingPipelineConfig:
    """Resolve a q12 template against one authorized secondary target.

    Returns:
        The exact secondary target-bound pipeline.
    """
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
        optimization_block_id="wp22b_secondary_q12_width_test",
        optimization_seed=71,
        data_role="secondary_benchmark",
    )


def _materialize_secondary_target(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TargetPopulationManifest, MaterializedTarget]:
    """Materialize a cheap but genuine checksum-bound q12 target.

    Returns:
        The secondary manifest and its first typed materialized target.
    """
    manifest = _secondary_q12_manifest(monkeypatch)
    preregistration = load_initial_preregistration()
    config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(_MASTER_ENTROPY),
        population_scope="secondary_q12",
    )

    def cheap_tfim_vector(
        _parameters: Mapping[str, object],
        *,
        spectrum_agreement_rtol: float,
        spectrum_agreement_atol: float,
    ) -> np.ndarray:
        """Return a normalized q12 vector for every manifest member."""
        assert spectrum_agreement_rtol >= 0.0
        assert spectrum_agreement_atol >= 0.0
        vector = np.zeros(2**12, dtype=np.complex128)
        vector[0] = 1.0
        return vector

    monkeypatch.setattr(targets_module, "_tfim_ground_state_vector", cheap_tfim_vector)
    authorization = authorize_target_materialization(
        preregistration,
        config,
        manifest,
        _MASTER_ENTROPY,
    )
    target = materialize_target_population(
        config,
        preregistration,
        manifest,
        _MASTER_ENTROPY,
        authorization,
    ).targets[0]
    return manifest, target


def test_layerwise_profiles_complete_real_q12_noisy_and_noiseless_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both exact layerwise treatments complete one genuine terminal q12 update."""
    manifest, target = _materialize_secondary_target(monkeypatch)
    templates = (
        build_layerwise_bmpd_crn_v2_template(
            training_trajectory_count=1,
            checkpoint_validation_trajectory_count=1,
            qubit_count=12,
        ),
        build_layerwise_bmpd_noiseless_template(
            checkpoint_validation_trajectory_count=1,
            qubit_count=12,
        ),
    )

    for template in templates:
        pipeline = _resolve_secondary(template, manifest)
        stage = pipeline.stages[-1]
        runner = LayerwiseBMPDStageRunner(pipeline, target)
        binding = create_bmpd_circuit_binding(12, 4)
        initial_theta = np.zeros(stage.input_parameter_count, dtype=np.float64)
        execution = execute_fixed_rate_krotov_stage(
            stage,
            binding,
            target,
            initial_theta,
            iteration_count=1,
        )
        assert isinstance(execution, NoisyKrotovStageExecution)
        assert len(execution.trace) == 2
        assert execution.trace[-1].global_iteration == 1
        assert execution.final_theta.shape == (432,)
        assert len(execution.training_ensembles) == stage.trajectory_count
        assert len(execution.checkpoint_validation_ensembles) == 1
        assert execution.normalized_work["gradient_evaluations"] == 1
        assert execution.normalized_work["training_trajectories"] == 4 * stage.trajectory_count
        statistics = runner.circuit_statistics(stage)
        assert statistics["topology_id"] == "bmpd_q12_d4"
        assert statistics["parameter_count"] == 432
        assert statistics["native_two_qubit_gates_per_chain_edge"] == [12] * 11


def test_fixed_depth_runner_accepts_q12_and_derives_resource_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixed-depth runner completes one real q12 update and resource audit."""
    manifest, target = _materialize_secondary_target(monkeypatch)
    template = build_fixed_depth_bmpd_crn_template(
        iteration_budget=1,
        training_trajectory_count=1,
        checkpoint_validation_trajectory_count=1,
        qubit_count=12,
    )
    pipeline = _resolve_secondary(template, manifest)
    runner = FixedDepthBMPDStageRunner(pipeline, target)
    execution = runner(pipeline.stages[0], None)
    assert isinstance(execution, NoisyKrotovStageExecution)
    assert len(execution.trace) == 2
    assert execution.trace[-1].global_iteration == 1
    assert len(execution.training_ensembles) == 1
    assert execution.training_ensembles[0].trajectory_count == 1
    statistics = runner.circuit_statistics(pipeline.stages[0])
    assert statistics["topology_id"] == "bmpd_q12_d4"
    assert statistics["parameter_count"] == 432
    assert statistics["qubit_count"] == 12
    assert statistics["native_two_qubit_gates_per_chain_edge"] == [12] * 11
