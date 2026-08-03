# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""WP20 paired-block isolation and safe test-coupling tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.state_preparation import phase2
from benchmarks.state_preparation.noise import FIXED_RATE_NOISE_DEFINITION_VERSION
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.legacy_targets import load_legacy_target_collection
from benchmarks.state_preparation.phase2.operator_growth import (
    OperatorGrowthResult,
    adapt_style_state_preparation,
    run_standard_fixed_rate_noisy_operator_growth,
)
from benchmarks.state_preparation.phase2.pipeline import PipelineEvaluationConfig
from benchmarks.state_preparation.phase2.wp20_resources import (
    CircuitResourceMetrics,
    EventLevelTestCoupling,
    OperatorGrowthRandomnessEvidence,
    PairedBlockIdentity,
    TrainingRandomnessRecord,
    TrainingRandomnessStageEvidence,
    decide_event_level_test_coupling,
    measure_circuit_resources,
    validate_training_randomness_isolation,
)
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate
from tests.benchmarks.test_state_preparation_wp17_noisy_krotov import _stage

_MANIFEST = "sha256:" + "1" * 64
_SPEC = "sha256:" + "2" * 64
_ENSEMBLE_A = "sha256:" + "a" * 64
_ENSEMBLE_B = "sha256:" + "b" * 64
_ENSEMBLE_C = "sha256:" + "c" * 64


def _checksum(character: str) -> str:
    """Return a compact valid checksum fixture."""
    return "sha256:" + character * 64


def _materialized_circuit_id(
    training_id: str,
    checkpoint_checksum: str,
    materialization_checksum: str,
    circuit_checksum: str,
) -> str:
    """Reproduce the public materialized-circuit identity contract.

    Returns:
        The derived circuit identifier.
    """
    checksum = canonical_checksum({
        "identity_version": "yaqs.state_preparation.phase2.materialized_circuit_identity.v1",
        "pipeline_training_id": training_id,
        "final_checkpoint_checksum": checkpoint_checksum,
        "final_materialization_policy_checksum": materialization_checksum,
        "materialized_circuit_checksum": circuit_checksum,
    })
    return "phase2_circuit_" + checksum.removeprefix("sha256:")


def _evaluation(side: str, *, noise_scale: float = 1.0) -> PipelineEvaluationConfig:
    """Return a complete final-test configuration for one compared method."""
    character = "4" if side == "left" else "5"
    training_id = "phase2_training_" + character * 64
    final_checkpoint = _checksum("6")
    materialization = _checksum("7")
    circuit = _checksum(character)
    return PipelineEvaluationConfig(
        pipeline_training_id=training_id,
        pipeline_configuration_checksum=_checksum(character),
        pipeline_result_checksum=_checksum("8" if side == "left" else "9"),
        final_checkpoint_checksum=final_checkpoint,
        final_materialization_policy_checksum=materialization,
        data_role="screening_selection",
        materialized_circuit_id=_materialized_circuit_id(
            training_id,
            final_checkpoint,
            materialization,
            circuit,
        ),
        materialized_circuit_checksum=circuit,
        test_noise_id="depolarizing_1s_all",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=noise_scale,
        tjm_dt=1.0,
        evaluation_seed=901,
        evaluation_seed_domain="screening_selection",
        repetition=0,
        trajectory_budget=32,
        evaluation_policy="fixed_sample",
        confidence_level=None,
        confidence_interval_method=None,
        sidecar_storage_policy="trajectory_fidelities",
        max_bond_dimension=64,
        svd_threshold=0.0,
        truncation_mode="discarded_weight",
        min_bond_dimension=1,
    )


def _test_protocol_checksum() -> str:
    """Return the paired commitment to the complete method-independent test protocol."""
    evaluation = _evaluation("left")
    return canonical_checksum({
        "evaluation_schema_version": evaluation.schema_version,
        "data_role": evaluation.data_role,
        "test_noise_id": evaluation.test_noise_id,
        "noise_definition_version": evaluation.noise_definition_version,
        "noise_strength_scale": evaluation.noise_strength_scale,
        "tjm_dt": evaluation.tjm_dt,
        "evaluation_seed": evaluation.evaluation_seed,
        "evaluation_seed_domain": evaluation.evaluation_seed_domain,
        "repetition": evaluation.repetition,
        "trajectory_budget": evaluation.trajectory_budget,
        "evaluation_policy": evaluation.evaluation_policy,
        "confidence_level": evaluation.confidence_level,
        "confidence_interval_method": evaluation.confidence_interval_method,
        "sidecar_storage_policy": evaluation.sidecar_storage_policy,
        "max_bond_dimension": evaluation.max_bond_dimension,
        "svd_threshold": evaluation.svd_threshold,
        "truncation_mode": evaluation.truncation_mode,
        "min_bond_dimension": evaluation.min_bond_dimension,
    })


def _reseal(document: dict[str, object]) -> None:
    """Refresh a top-level checksum after deliberate test tampering."""
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })


def _block(*, optimization_seed: int = 17) -> PairedBlockIdentity:
    """Return one exact target/optimization/resource comparison block."""
    return PairedBlockIdentity(
        target_instance_id="phase2_target_" + "3" * 64,
        target_manifest_checksum=_MANIFEST,
        target_spec_checksum=_SPEC,
        optimization_block_id="target-3-seed-17",
        optimization_seed=optimization_seed,
        test_noise_id="depolarizing_1s_all",
        test_protocol_checksum=_test_protocol_checksum(),
        resource_stratum_id="native-rzz-12",
    )


def _operator_growth_case(
    *,
    trajectory_seed: int = 701,
) -> tuple[PairedBlockIdentity, OperatorGrowthResult, TrainingRandomnessRecord]:
    """Return matching block, noisy operator result, and randomness record.

    Returns:
        The exact paired block, completed operator result, and derived record.
    """
    target = load_legacy_target_collection().target("legacy_tfim_seed_100")
    result = run_standard_fixed_rate_noisy_operator_growth(
        target,
        optimization_block_id="legacy-target-operator-growth",
        optimization_seed=17,
        resource_stratum_id="native-rzz-12",
        noise_id="dephasing_1s_1q",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        trajectory_count=2,
        trajectory_seed=trajectory_seed,
        max_operators=0,
    )
    block = PairedBlockIdentity(
        target_instance_id=target.target_instance_id,
        target_manifest_checksum=target.target_manifest_checksum,
        target_spec_checksum=target.target_instance_spec_checksum,
        optimization_block_id="legacy-target-operator-growth",
        optimization_seed=17,
        test_noise_id="depolarizing_1s_all",
        test_protocol_checksum=_test_protocol_checksum(),
        resource_stratum_id="native-rzz-12",
    )
    return block, result, TrainingRandomnessRecord.from_operator_growth_result(block, result)


def _randomness(
    block: PairedBlockIdentity,
    method_id: str,
    training_id: str,
    *,
    seeds: tuple[int, ...],
    ensembles: tuple[str, ...],
) -> TrainingRandomnessRecord:
    """Build one method's training-only randomness evidence.

    Returns:
        The immutable randomness record.

    Raises:
        ValueError: If the compact test inputs contain surplus ensembles.
    """
    stages: list[TrainingRandomnessStageEvidence] = []
    optimizer_seed = int(canonical_checksum({"optimizer": training_id})[7:23], 16)
    if seeds:
        for index, seed in enumerate(seeds):
            ensemble = () if index >= len(ensembles) else (ensembles[index],)
            stage = replace(
                _stage(
                    trajectories=1,
                    training_seed=seed,
                    optimizer_seed=(optimizer_seed + index) % (2**64),
                    iterations=1,
                ),
                stage_index=index,
                stage_id=f"training_stage_{index}",
            )
            stages.append(
                TrainingRandomnessStageEvidence(
                    stage=stage,
                    execution_checksum=canonical_checksum({"training_id": training_id, "stage": index}),
                    training_ensemble_checksums=ensemble,
                )
            )
    else:
        stage = replace(
            _stage(noise_id="noiseless", optimizer_seed=optimizer_seed, iterations=1),
            stage_index=0,
            stage_id="noiseless_stage",
        )
        stages.append(
            TrainingRandomnessStageEvidence(
                stage=stage,
                execution_checksum=canonical_checksum({"training_id": training_id, "stage": 0}),
                training_ensemble_checksums=(),
            )
        )
    if len(ensembles) > len(seeds):
        msg = "Test helper received more ensembles than noisy stages."
        raise ValueError(msg)
    return TrainingRandomnessRecord(
        paired_block_checksum=block.content_checksum,
        method_id=method_id,
        training_id=training_id,
        pipeline_configuration_checksum=canonical_checksum({
            "method_id": method_id,
            "training_id": training_id,
            "stages": [stage.stage.configuration_checksum for stage in stages],
        }),
        stages=tuple(stages),
    )


def _single_gate_circuit(
    name: str,
    sites: tuple[int, ...],
    *,
    logical_gate_id: str = "logical-0",
) -> ParameterizedCircuit:
    """Return one trainable event with stable source identity."""
    return ParameterizedCircuit(
        num_qubits=2,
        gates=[ParameterizedGate(name, sites, param_index=0, logical_gate_id=logical_gate_id)],
        num_params=1,
    )


def _coupling(
    left: CircuitResourceMetrics,
    right: CircuitResourceMetrics,
    *,
    block: PairedBlockIdentity | None = None,
) -> EventLevelTestCoupling:
    """Return a coupling decision bound to the standard paired noise block."""
    resolved_block = _block() if block is None else block
    return decide_event_level_test_coupling(
        resolved_block,
        left,
        right,
        left_method_id="method-left",
        right_method_id="method-right",
        left_evaluation=_evaluation("left"),
        right_evaluation=_evaluation("right"),
    )


def test_paired_block_is_strict_sealed_and_uses_every_pairing_coordinate() -> None:
    """Changing target, seed, noise, or stratum changes the paired identity."""
    block = _block()

    assert PairedBlockIdentity.from_dict(block.to_dict()) == block
    assert replace(block, optimization_seed=18).content_checksum != block.content_checksum
    assert replace(block, test_noise_id="depolarizing_2s_all").content_checksum != block.content_checksum
    assert replace(block, test_protocol_checksum=_checksum("0")).content_checksum != block.content_checksum
    assert replace(block, resource_stratum_id="native-rzz-10").content_checksum != block.content_checksum

    document = block.to_dict()
    document["optimization_seed"] = 18
    with pytest.raises(ValueError, match="content checksum mismatch"):
        PairedBlockIdentity.from_dict(document)


def test_paired_methods_are_sorted_but_never_share_training_randomness() -> None:
    """Pairing shares the outer block, not trajectory seeds or map ensembles."""
    block = _block()
    adam = _randomness(
        block,
        "parameter_shift_adam_layerwise",
        "training-adam",
        seeds=(101, 102),
        ensembles=(_ENSEMBLE_A, _ENSEMBLE_C),
    )
    krotov = _randomness(
        block,
        "layerwise_bmpd_crn_v2",
        "training-krotov",
        seeds=(201,),
        ensembles=(_ENSEMBLE_B,),
    )
    noiseless = _randomness(
        block,
        "layerwise_bmpd_noiseless",
        "training-noiseless",
        seeds=(),
        ensembles=(),
    )

    validated = validate_training_randomness_isolation(block, (noiseless, krotov, adam))
    assert tuple(record.method_id for record in validated) == (
        "layerwise_bmpd_crn_v2",
        "layerwise_bmpd_noiseless",
        "parameter_shift_adam_layerwise",
    )
    assert TrainingRandomnessRecord.from_dict(adam.to_dict()) == adam


def test_operator_growth_randomness_embeds_and_derives_the_complete_noisy_result() -> None:
    """Standalone target-bound growth participates through strict derived evidence."""
    block, result, record = _operator_growth_case()
    evidence = record.operator_growth_evidence
    provenance = result.training_provenance

    assert isinstance(evidence, OperatorGrowthRandomnessEvidence)
    assert phase2.OperatorGrowthRandomnessEvidence is OperatorGrowthRandomnessEvidence
    assert evidence.schema_version == phase2.WP20_OPERATOR_GROWTH_RANDOMNESS_SCHEMA_VERSION
    assert provenance is not None
    assert evidence.result == result
    assert evidence.method_id == result.method_id == record.method_id
    assert evidence.target_instance_id == block.target_instance_id
    assert evidence.target_spec_checksum == block.target_spec_checksum
    assert evidence.target_manifest_checksum == block.target_manifest_checksum
    assert evidence.optimization_block_id == block.optimization_block_id
    assert evidence.optimization_seed == block.optimization_seed
    assert evidence.resource_stratum_id == block.resource_stratum_id
    assert evidence.training_seed == provenance.trajectory_seed
    assert evidence.training_ensemble_checksum == provenance.trajectory_ensemble_checksum
    assert record.stages == ()
    assert record.training_noise_active is True
    assert record.training_seeds == (evidence.training_seed,)
    assert record.training_ensemble_checksums == (evidence.training_ensemble_checksum,)
    assert record.source_execution_checksums == (result.content_checksum,)
    assert OperatorGrowthRandomnessEvidence.from_dict(evidence.to_dict()) == evidence
    assert TrainingRandomnessRecord.from_dict(record.to_dict()) == record

    tampered = evidence.to_dict()
    tampered["training_seed"] = evidence.training_seed + 1
    _reseal(tampered)
    with pytest.raises(ValueError, match="aliases are not derived"):
        OperatorGrowthRandomnessEvidence.from_dict(tampered)


def test_operator_growth_randomness_requires_exact_block_and_exclusive_evidence_mode() -> None:
    """Target coordinates and evidence representation cannot be caller substituted."""
    block, result, record = _operator_growth_case()
    mismatches = (
        replace(block, target_instance_id="different-target"),
        replace(block, target_spec_checksum=_checksum("d")),
        replace(block, target_manifest_checksum=_checksum("e")),
        replace(block, optimization_block_id="different-optimization-block"),
        replace(block, optimization_seed=18),
        replace(block, resource_stratum_id="native-rzz-13"),
    )
    for mismatch in mismatches:
        with pytest.raises(ValueError, match="does not belong to the paired block"):
            TrainingRandomnessRecord.from_operator_growth_result(mismatch, result)

    staged = _randomness(block, "method-b", "training-b", seeds=(811,), ensembles=(_ENSEMBLE_A,))
    with pytest.raises(ValueError, match="exactly one"):
        replace(record, stages=staged.stages)
    with pytest.raises(ValueError, match="exactly one"):
        replace(record, operator_growth_evidence=None)

    target = load_legacy_target_collection().target("legacy_tfim_seed_100")
    analytic = adapt_style_state_preparation(target.state_vector_copy(), family_id=target.family_id, max_operators=0)
    with pytest.raises(ValueError, match="promotion-eligible"):
        TrainingRandomnessRecord.from_operator_growth_result(block, analytic)


def test_operator_growth_seed_and_ensemble_are_enforced_by_pairing_isolation() -> None:
    """Operator-growth CRN identities cannot leak into another paired method."""
    block, _result, operator_record = _operator_growth_case()
    seed = operator_record.training_seeds[0]
    ensemble = operator_record.training_ensemble_checksums[0]
    shared_seed = _randomness(block, "method-b", "training-b", seeds=(seed,), ensembles=(_ENSEMBLE_A,))
    shared_map = _randomness(block, "method-b", "training-b", seeds=(812,), ensembles=(ensemble,))
    independent = _randomness(block, "method-b", "training-b", seeds=(813,), ensembles=(_ENSEMBLE_B,))

    with pytest.raises(ValueError, match="share training trajectory seeds"):
        validate_training_randomness_isolation(block, (operator_record, shared_seed))
    with pytest.raises(ValueError, match="share sampled training or checkpoint-map ensembles"):
        validate_training_randomness_isolation(block, (operator_record, shared_map))
    assert {
        record.method_id for record in validate_training_randomness_isolation(block, (operator_record, independent))
    } == {
        operator_record.method_id,
        independent.method_id,
    }


def test_training_seed_ensemble_identity_and_block_reuse_are_rejected() -> None:
    """Every paired method receives an independent training random stream."""
    block = _block()
    first = _randomness(block, "method-a", "training-a", seeds=(10,), ensembles=(_ENSEMBLE_A,))
    shared_seed = _randomness(block, "method-b", "training-b", seeds=(10,), ensembles=(_ENSEMBLE_B,))
    shared_map = _randomness(block, "method-b", "training-b", seeds=(20,), ensembles=(_ENSEMBLE_A,))
    duplicate_training = _randomness(block, "method-b", "training-a", seeds=(20,), ensembles=(_ENSEMBLE_B,))
    foreign = _randomness(
        _block(optimization_seed=18),
        "method-b",
        "training-b",
        seeds=(20,),
        ensembles=(_ENSEMBLE_B,),
    )

    with pytest.raises(ValueError, match="share training trajectory seeds"):
        validate_training_randomness_isolation(block, (first, shared_seed))
    with pytest.raises(ValueError, match="share sampled training or checkpoint-map ensembles"):
        validate_training_randomness_isolation(block, (first, shared_map))
    with pytest.raises(ValueError, match="distinct training identities"):
        validate_training_randomness_isolation(block, (first, duplicate_training))
    with pytest.raises(ValueError, match="supplied paired block"):
        validate_training_randomness_isolation(block, (first, foreign))


def test_paired_methods_cannot_share_optimizer_or_initialization_streams() -> None:
    """Noiseless controls still carry stochastic optimizer and initialization roles."""
    block = _block()
    first = _randomness(
        block,
        "layerwise_bmpd_noiseless",
        "training-noiseless-a",
        seeds=(),
        ensembles=(),
    )
    second = _randomness(
        block,
        "phase1_noiseless_checkpoint_control",
        "training-noiseless-b",
        seeds=(),
        ensembles=(),
    )
    shared_optimizer_stage = replace(
        second.stages[0].stage,
        optimizer_seed=first.stages[0].stage.optimizer_seed,
    )
    shared_optimizer = replace(
        second,
        stages=(
            replace(
                second.stages[0],
                stage=shared_optimizer_stage,
            ),
        ),
    )
    with pytest.raises(ValueError, match="share optimizer seeds"):
        validate_training_randomness_isolation(block, (first, shared_optimizer))

    initialized: list[TrainingRandomnessRecord] = []
    for record in (first, second):
        stage = replace(
            record.stages[0].stage,
            input_topology_id=None,
            input_parameter_count=0,
            parameter_transfer_rule="initialize_random_normal",
            initialization_seed=1234,
        )
        initialized.append(replace(record, stages=(replace(record.stages[0], stage=stage),)))
    with pytest.raises(ValueError, match="share initialization seeds"):
        validate_training_randomness_isolation(block, tuple(initialized))


def test_noisy_training_randomness_cannot_certify_omitted_seeds_or_maps() -> None:
    """Complete stage schedules, rather than caller counts, determine required maps."""
    block = _block()
    noisy_stage = _stage(trajectories=1, training_seed=101, iterations=1)
    with pytest.raises(ValueError, match="incomplete for the sealed stage"):
        TrainingRandomnessStageEvidence(
            stage=noisy_stage,
            execution_checksum=canonical_checksum({"execution": "missing-map"}),
            training_ensemble_checksums=(),
        )
    resampled = replace(noisy_stage, sampling_policy="resampled", iteration_budget=2)
    with pytest.raises(ValueError, match="incomplete for the sealed stage"):
        TrainingRandomnessStageEvidence(
            stage=resampled,
            execution_checksum=canonical_checksum({"execution": "omitted-resample"}),
            training_ensemble_checksums=(_ENSEMBLE_A,),
        )
    noiseless_stage = replace(
        _stage(noise_id="noiseless", iterations=1),
        stage_index=0,
        stage_id="noiseless_stage",
    )
    stage_evidence = TrainingRandomnessStageEvidence(
        stage=noiseless_stage,
        execution_checksum=canonical_checksum({"execution": "noiseless"}),
        training_ensemble_checksums=(),
    )
    with pytest.raises(ValueError, match="not a registered noiseless-training control"):
        TrainingRandomnessRecord(
            paired_block_checksum=block.content_checksum,
            method_id="layerwise_bmpd_crn_v2",
            training_id="training-krotov",
            pipeline_configuration_checksum=canonical_checksum({"pipeline": "noisy-claim"}),
            stages=(stage_evidence,),
        )


def test_identical_full_native_signatures_authorize_event_level_coupling() -> None:
    """Independent compilations couple only after complete event comparison."""
    left = measure_circuit_resources(_single_gate_circuit("rxx", (0, 1)))
    right = measure_circuit_resources(_single_gate_circuit("rxx", (0, 1)))
    decision = _coupling(left, right)

    assert decision.mode == "event_level_coupled"
    assert decision.reason == "identical_full_native_event_signatures"
    assert decision.aligned_event_count == 5
    assert decision.alignment_checksum is not None
    assert EventLevelTestCoupling.from_dict(decision.to_dict()) == decision


def test_coincident_native_ids_and_counts_do_not_authorize_coupling() -> None:
    """Locally numbered RXX and RYY compilations have different full events."""
    rxx = measure_circuit_resources(_single_gate_circuit("rxx", (0, 1)))
    ryy = measure_circuit_resources(_single_gate_circuit("ryy", (0, 1)))

    assert tuple(event.native_gate_id for event in rxx.native_events) == tuple(
        event.native_gate_id for event in ryy.native_events
    )
    assert rxx.native_one_qubit_gates == ryy.native_one_qubit_gates == 4
    assert rxx.native_two_qubit_gates == ryy.native_two_qubit_gates == 1
    decision = _coupling(rxx, ryy)
    assert decision.mode == "independent"
    assert decision.reason == "native_event_signature_mismatch"
    assert decision.alignment_checksum is None


def test_source_identity_count_mismatch_and_empty_circuits_force_independence() -> None:
    """Every unsafe alignment path records an explicit independent reason."""
    original = measure_circuit_resources(_single_gate_circuit("rzz", (0, 1), logical_gate_id="source-a"))
    renamed = measure_circuit_resources(_single_gate_circuit("rzz", (0, 1), logical_gate_id="source-b"))
    extra = measure_circuit_resources(
        ParameterizedCircuit(
            2,
            [
                ParameterizedGate("rzz", (0, 1), param_index=0, logical_gate_id="source-a"),
                ParameterizedGate("rx", (0,), param_index=1, logical_gate_id="extra"),
            ],
            num_params=2,
        )
    )
    empty = measure_circuit_resources(ParameterizedCircuit(2, [], num_params=0))

    assert _coupling(original, renamed).reason == "native_event_signature_mismatch"
    count_decision = _coupling(original, extra)
    assert count_decision.reason == "native_event_count_mismatch"
    assert count_decision.aligned_event_count == 1
    assert _coupling(empty, empty).reason == "no_stable_native_events"


def test_coupling_rejects_noise_identity_mismatch_with_paired_block() -> None:
    """Native equality cannot couple methods evaluated under different noise identities."""
    block = _block()
    resources = measure_circuit_resources(_single_gate_circuit("rxx", (0, 1)))

    with pytest.raises(ValueError, match="test-noise identities must match"):
        decide_event_level_test_coupling(
            block,
            resources,
            resources,
            left_method_id="method-left",
            right_method_id="method-right",
            left_evaluation=replace(_evaluation("left"), test_noise_id="depolarizing_2s_all"),
            right_evaluation=_evaluation("right"),
        )

    document = _coupling(resources, resources, block=block).to_dict()
    document["right_test_noise_id"] = "depolarizing_2s_all"
    _reseal(document)
    with pytest.raises(ValueError, match="test-noise identities must match"):
        EventLevelTestCoupling.from_dict(document)


def test_coupling_rejects_different_complete_test_protocols() -> None:
    """A shared noise slug cannot hide a different strength or evaluation channel."""
    block = _block()
    resources = measure_circuit_resources(_single_gate_circuit("rxx", (0, 1)))

    with pytest.raises(ValueError, match="identical complete final-test protocols"):
        decide_event_level_test_coupling(
            block,
            resources,
            resources,
            left_method_id="method-left",
            right_method_id="method-right",
            left_evaluation=_evaluation("left"),
            right_evaluation=_evaluation("right", noise_scale=2.0),
        )


def test_coupling_decision_rejects_resealed_mode_or_alignment_forgery() -> None:
    """A caller cannot upgrade independent streams to coupled evidence."""
    left = measure_circuit_resources(_single_gate_circuit("rxx", (0, 1)))
    right = measure_circuit_resources(_single_gate_circuit("ryy", (0, 1)))
    document = _coupling(left, right).to_dict()
    document["mode"] = "event_level_coupled"
    document["reason"] = "identical_full_native_event_signatures"
    document["alignment_checksum"] = "sha256:" + "f" * 64
    _reseal(document)

    with pytest.raises(ValueError, match="mechanically derived"):
        EventLevelTestCoupling.from_dict(document)


def test_coupling_cannot_be_forged_from_equal_checksums_without_event_evidence() -> None:
    """Equal aliases and a resealed alignment hash cannot replace native events."""
    resources = measure_circuit_resources(_single_gate_circuit("rxx", (0, 1)))
    decision = _coupling(resources, resources)
    document = decision.to_dict()
    document["aligned_native_events"] = []
    document["aligned_event_count"] = 1
    document["alignment_checksum"] = "sha256:" + "f" * 64
    _reseal(document)
    with pytest.raises(ValueError, match="every aligned native event"):
        EventLevelTestCoupling.from_dict(document)


def test_coupling_cannot_be_relabelled_for_another_method_or_evaluation() -> None:
    """Alignment evidence is ordered and bound to both compared executions."""
    resources = measure_circuit_resources(_single_gate_circuit("rxx", (0, 1)))
    method_document = _coupling(resources, resources).to_dict()
    method_document["left_method_id"] = "relabelled-method"
    _reseal(method_document)

    with pytest.raises(ValueError, match="alignment checksum is not derived"):
        EventLevelTestCoupling.from_dict(method_document)

    evaluation_document = _coupling(resources, resources).to_dict()
    evaluation_document["left_evaluation_id"] = "relabelled-evaluation"
    _reseal(evaluation_document)
    with pytest.raises(ValueError, match="aliases must be derived"):
        EventLevelTestCoupling.from_dict(evaluation_document)
