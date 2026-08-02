# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for bounded deterministic Phase II artifact codecs."""

from __future__ import annotations

import io
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.state_preparation.noise import FIXED_RATE_NOISE_DEFINITION_VERSION
from benchmarks.state_preparation.phase2 import (
    decode_noisy_krotov_circuit_binding_document,
    validate_noisy_krotov_execution_trace,
)
from benchmarks.state_preparation.phase2.artifact_codecs import (
    MAX_TRAJECTORY_FIDELITY_COUNT,
    StageParameterCheckpoint,
    artifact_checksum,
    create_phase2_trajectory_sidecar,
    read_phase2_trajectory_sidecar,
)
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.noisy_krotov import (
    LOGICAL_PARAMETERIZED_GATE_PLACEMENT,
    NOISY_KROTOV_CIRCUIT_BINDING_SCHEMA_VERSION,
    PRIMARY_COMPILER_POLICY_ID,
    PRIMARY_CONNECTIVITY,
    PRIMARY_COUNTING_POLICY_ID,
    PRIMARY_ROUTING_POLICY_ID,
    KrotovWorkLedger,
    NoisyKrotovIterationRecord,
    NoisyKrotovObjectiveBinding,
    NoisyKrotovStageExecution,
    translate_fixed_rate_krotov_stage,
)
from benchmarks.state_preparation.phase2.pipeline import CheckpointValidationConfig, TrainingStageConfig
from mqt.yaqs.optimization import KrotovFixedMapEnsemble, KrotovNoiseMap

_TRAINING_ID = "phase2_training_" + "1" * 64
_PREFIX_ID = "phase2_pipeline_prefix_" + "2" * 64
_EVALUATION_ID = "phase2_evaluation_" + "3" * 64
_CIRCUIT_PAYLOAD = {
    "schema_version": NOISY_KROTOV_CIRCUIT_BINDING_SCHEMA_VERSION,
    "topology_id": "toy_d1",
    "placement": LOGICAL_PARAMETERIZED_GATE_PLACEMENT,
    "compiler_policy_id": PRIMARY_COMPILER_POLICY_ID,
    "connectivity": PRIMARY_CONNECTIVITY,
    "routing_policy_id": PRIMARY_ROUTING_POLICY_ID,
    "counting_policy_id": PRIMARY_COUNTING_POLICY_ID,
    "num_qubits": 2,
    "num_params": 3,
    "noisy_gate_indices": [0],
    "gates": [
        {
            "name": "ry",
            "sites": [0],
            "param_index": 0,
            "angle_scale": 1.0,
            "angle_offset": 0.0,
            "fixed_params": [],
            "logical_gate_id": None,
            "native_gate_id": None,
            "noise_enabled": True,
        }
    ],
}
_CIRCUIT_CHECKSUM = canonical_checksum(_CIRCUIT_PAYLOAD)
_CIRCUIT_DOCUMENT = {**_CIRCUIT_PAYLOAD, "content_checksum": _CIRCUIT_CHECKSUM}
_OBJECTIVE_BINDING = NoisyKrotovObjectiveBinding(
    target_state_checksum=canonical_checksum({"target": "toy"}),
    initial_state_policy="custom_state_v1",
    initial_state_checksum=canonical_checksum({"initial_state": "toy"}),
    materialized_target_identity=None,
)
_OBJECTIVE_CHECKSUM = _OBJECTIVE_BINDING.objective_checksum
_CHECKSUM = canonical_checksum({"map": "toy"})


def _parameter_checksum(values: np.ndarray) -> str:
    """Return the WP17 checksum of one exact float64 parameter vector."""
    return artifact_checksum(np.ascontiguousarray(values, dtype=np.dtype("<f8")).tobytes())


def _resealed_circuit_document(**gate_updates: object) -> dict[str, object]:
    """Return a binding document with one changed gate and a matching seal."""
    gates = _CIRCUIT_PAYLOAD["gates"]
    assert isinstance(gates, list)
    gate = gates[0]
    assert isinstance(gate, dict)
    payload = {**_CIRCUIT_PAYLOAD, "gates": [{**gate, **gate_updates}]}
    return {**payload, "content_checksum": canonical_checksum(payload)}


def _stage() -> TrainingStageConfig:
    """Create a compact validation-selecting noisy stage.

    Returns:
        A complete resolved training-stage configuration.
    """
    validation = CheckpointValidationConfig(
        noise_id="depolarizing_1s_all",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        trajectory_count=2,
        seed=41,
        sampling_policy="crn_fixed",
        ensemble_refresh_interval=None,
        cadence=1,
        selection_rule="best_validation_fidelity",
        tie_breaker="earliest_iteration",
    )
    return TrainingStageConfig(
        stage_index=1,
        stage_id="noisy_fine_tune",
        stage_kind="optimize",
        input_topology_id="toy_d1",
        output_topology_id="toy_d1",
        input_parameter_count=3,
        output_parameter_count=3,
        parameter_transfer_rule="copy",
        initialization_seed=None,
        optimizer_id="krotov",
        optimizer_hyperparameters={"learning_rate": 0.05, "schedule": "constant", "decay": 0.0},
        optimizer_seed=17,
        iteration_budget=2,
        training_noise_id="depolarizing_1s_all",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        trajectory_count=2,
        training_seed=29,
        trajectory_update="independent",
        sampling_policy="crn_fixed",
        crn_refresh_interval=None,
        checkpoint_validation=validation,
        pruning_rule="none",
        pruning_threshold=None,
        max_bond_dimension=None,
        svd_threshold=0.0,
        truncation_mode="discarded_weight",
        min_bond_dimension=1,
    )


def _execution(*, selected_fidelity: float = 0.8) -> NoisyKrotovStageExecution:
    """Create a successful execution whose selected and final states differ.

    Returns:
        A synthetic but internally valid WP17 execution.
    """
    stage = _stage()
    binding = decode_noisy_krotov_circuit_binding_document(_CIRCUIT_DOCUMENT)
    provider_checksum = translate_fixed_rate_krotov_stage(stage, binding).provider_checksum
    assert provider_checksum is not None
    training_ensemble = KrotovFixedMapEnsemble(
        role="training_trajectory",
        resolved_seed=29,
        stage_index=stage.stage_index,
        stage_id=stage.stage_id,
        stage_configuration_checksum=stage.configuration_checksum,
        circuit_checksum=_CIRCUIT_CHECKSUM,
        provider_checksum=provider_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
        trajectory_maps=[[KrotovNoiseMap(source_gate_index=0, is_identity=True)] for _ in range(2)],
    )
    validation_ensemble = KrotovFixedMapEnsemble(
        role="checkpoint_validation",
        resolved_seed=41,
        stage_index=stage.stage_index,
        stage_id=stage.stage_id,
        stage_configuration_checksum=stage.configuration_checksum,
        circuit_checksum=_CIRCUIT_CHECKSUM,
        provider_checksum=provider_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
        trajectory_maps=[[KrotovNoiseMap(source_gate_index=0, is_identity=True)] for _ in range(2)],
    )
    initial_work = KrotovWorkLedger(
        objective_evaluations=2,
        training_trajectories=4,
        checkpoint_validation_trajectories=4,
        trajectory_gate_applications=8,
    )
    selected_work = KrotovWorkLedger(
        objective_evaluations=5,
        gradient_evaluations=1,
        training_trajectories=8,
        checkpoint_validation_trajectories=6,
        trajectory_gate_applications=14,
    )
    work = KrotovWorkLedger(
        objective_evaluations=8,
        gradient_evaluations=2,
        training_trajectories=12,
        checkpoint_validation_trajectories=8,
        trajectory_gate_applications=20,
    )
    initial = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    final = np.array([0.4, 0.5, 0.6], dtype=np.float64)
    selected = np.array([0.7, 0.8, 0.9], dtype=np.float64)
    normalized_selected_fidelity = min(1.0, max(0.0, selected_fidelity))
    selects_initial = normalized_selected_fidelity <= 0.0
    other_fidelity = 0.0 if selects_initial else 0.6
    final_fidelity = 0.0 if selects_initial else 0.7
    final_record = NoisyKrotovIterationRecord(
        local_iteration=2,
        global_iteration=2,
        parameter_checksum=_parameter_checksum(final),
        learning_rate=0.05,
        monitoring_loss=0.4,
        monitoring_fidelity=0.6,
        checkpoint_validation_fidelity=final_fidelity,
        update_signal=(0.0, 0.0, 0.0),
        update_signal_kind="independent_pathwise_gradient",
        update_signal_norm=0.0,
        gradient_norm=0.0,
        cross_dense_sum_norm=None,
        update_norm=0.0,
        trajectory_count=2,
        nonidentity_events=0,
        training_ensemble_id=training_ensemble.ensemble_id,
        training_ensemble_checksum=training_ensemble.content_checksum,
        checkpoint_validation_ensemble_checksum=validation_ensemble.content_checksum,
        cumulative_work=work,
    )
    initial_record = replace(
        final_record,
        local_iteration=0,
        global_iteration=0,
        parameter_checksum=_parameter_checksum(initial),
        learning_rate=0.0,
        checkpoint_validation_fidelity=other_fidelity,
        update_signal=(),
        update_signal_kind="none",
        gradient_norm=None,
        cumulative_work=initial_work,
        training_ensemble_sampled=True,
        checkpoint_validation_ensemble_sampled=True,
    )
    selected_record = replace(
        final_record,
        local_iteration=1,
        global_iteration=1,
        parameter_checksum=_parameter_checksum(selected),
        checkpoint_validation_fidelity=selected_fidelity,
        cumulative_work=selected_work,
    )
    return NoisyKrotovStageExecution(
        stage=stage,
        circuit_binding_checksum=_CIRCUIT_CHECKSUM,
        circuit_binding_document=_CIRCUIT_DOCUMENT,
        provider_checksum=provider_checksum,
        objective_binding=_OBJECTIVE_BINDING,
        initial_theta=initial,
        final_theta=final,
        selected_theta=initial if selects_initial else selected,
        selected_global_iteration=0 if selects_initial else 1,
        selected_checkpoint_validation_fidelity=selected_fidelity,
        trace=(initial_record, selected_record, final_record),
        training_ensembles=(training_ensemble,),
        checkpoint_validation_ensembles=(validation_ensemble,),
        normalized_work=work,
    )


def _validate_test_execution(
    execution: NoisyKrotovStageExecution,
    *,
    trace: tuple[NoisyKrotovIterationRecord, ...] | None = None,
    provider_checksum: str | None = None,
    validation_ensembles: tuple[KrotovFixedMapEnsemble, ...] | None = None,
) -> None:
    """Run exact trace validation with selected test evidence replaced."""
    validate_noisy_krotov_execution_trace(
        stage=_stage(),
        circuit_binding_document=execution.circuit_binding_document,
        provider_checksum=execution.provider_checksum if provider_checksum is None else provider_checksum,
        trace=execution.trace if trace is None else trace,
        training_ensembles=execution.training_ensembles,
        validation_ensembles=(
            execution.checkpoint_validation_ensembles if validation_ensembles is None else validation_ensembles
        ),
        normalized_work=execution.trace[-1].cumulative_work,
        input_resume_state=None,
    )


def _replace_zip_member(payload: bytes, name: str, replacement: bytes) -> bytes:
    """Return an archive with one member replaced for corruption tests."""
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        members = {info.filename: source.read(info) for info in source.infolist()}
    members[name] = replacement
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as target:
        for member_name in sorted(members):
            target.writestr(member_name, members[member_name])
    return buffer.getvalue()


def _add_zip_member(payload: bytes, name: str, value: bytes) -> bytes:
    """Return an archive with one unsupported member."""
    with zipfile.ZipFile(io.BytesIO(payload)) as source:
        members = {info.filename: source.read(info) for info in source.infolist()}
    members[name] = value
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as target:
        for member_name in sorted(members):
            target.writestr(member_name, members[member_name])
    return buffer.getvalue()


def _npy_bytes(values: np.ndarray) -> bytes:
    """Serialize one test array as NPY bytes.

    Returns:
        The generated NPY payload.
    """
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=True)
    return buffer.getvalue()


@pytest.mark.parametrize("param_index", [-1, 3])
def test_circuit_binding_decoder_rejects_resealed_parameter_indices(param_index: int) -> None:
    """A fresh checksum cannot legitimize an index outside the parameter vector."""
    document = _resealed_circuit_document(param_index=param_index)
    with pytest.raises(ValueError, match="outside the parameter vector"):
        decode_noisy_krotov_circuit_binding_document(document)


def test_circuit_binding_decoder_rejects_resealed_gate_arity() -> None:
    """A fresh checksum cannot legitimize a gate whose matrix and sites disagree."""
    document = _resealed_circuit_document(name="cx", sites=[0], param_index=None)
    with pytest.raises(ValueError, match="matrix arity does not match"):
        decode_noisy_krotov_circuit_binding_document(document)


def test_trace_validator_rejects_tampered_learning_rate() -> None:
    """Recorded learning rates remain derived from the sealed stage schedule."""
    execution = _execution()
    trace = (
        execution.trace[0],
        replace(execution.trace[1], learning_rate=0.04),
        execution.trace[2],
    )
    with pytest.raises(ValueError, match="learning rate does not match"):
        _validate_test_execution(execution, trace=trace)


def test_trace_validator_rejects_tampered_update_signal_dimension() -> None:
    """Update signals must retain the stage's exact parameter dimension."""
    execution = _execution()
    trace = (
        execution.trace[0],
        replace(
            execution.trace[1],
            update_signal=(0.0, 0.0),
            update_signal_norm=0.0,
            gradient_norm=0.0,
        ),
        execution.trace[2],
    )
    with pytest.raises(ValueError, match="output parameter count"):
        _validate_test_execution(execution, trace=trace)


def test_trace_validator_rejects_resampling_a_fixed_crn_ensemble() -> None:
    """A fixed CRN ensemble cannot be reported as newly sampled on reuse."""
    execution = _execution()
    trace = (
        execution.trace[0],
        replace(execution.trace[1], training_ensemble_sampled=True),
        execution.trace[2],
    )
    with pytest.raises(ValueError, match="cannot be sampled after its first use"):
        _validate_test_execution(execution, trace=trace)


def test_trace_validator_rejects_forged_stage_training_provider() -> None:
    """The trace-level provider must equal the provider derived from the stage."""
    execution = _execution()
    with pytest.raises(ValueError, match="stage-derived training-noise provider"):
        _validate_test_execution(execution, provider_checksum=_CHECKSUM)


def test_trace_validator_rejects_forged_stage_validation_provider() -> None:
    """Validation ensembles must use the provider derived from validation config."""
    execution = _execution()
    source = execution.checkpoint_validation_ensembles[0]
    forged_ensemble = KrotovFixedMapEnsemble(
        role=source.role,
        resolved_seed=source.resolved_seed,
        stage_index=source.stage_index,
        stage_id=source.stage_id,
        stage_configuration_checksum=source.stage_configuration_checksum,
        circuit_checksum=source.circuit_checksum,
        provider_checksum=_CHECKSUM,
        ensemble_index=source.ensemble_index,
        refresh_index=source.refresh_index,
        global_iteration_start=source.global_iteration_start,
        trajectory_maps=source.replay_maps(),
    )
    trace = tuple(
        replace(row, checkpoint_validation_ensemble_checksum=forged_ensemble.content_checksum)
        for row in execution.trace
    )
    with pytest.raises(ValueError, match="validation ensemble does not match"):
        _validate_test_execution(execution, trace=trace, validation_ensembles=(forged_ensemble,))


def test_noisy_checkpoint_is_deterministic_detached_and_resume_complete() -> None:
    """Selected/final states and exact WP17 resume evidence survive round-trip."""
    execution = _execution()
    checkpoint = StageParameterCheckpoint.from_noisy_krotov(
        pipeline_training_id=_TRAINING_ID,
        pipeline_prefix_id=_PREFIX_ID,
        execution=execution,
    )
    payload = checkpoint.to_bytes()
    assert payload == checkpoint.to_bytes()
    assert checkpoint.content_checksum == artifact_checksum(payload)
    assert checkpoint.selected_parameter_checksum != checkpoint.final_parameter_checksum

    restored = StageParameterCheckpoint.from_bytes(
        payload,
        expected_checksum=checkpoint.content_checksum,
        expected_pipeline_training_id=_TRAINING_ID,
        expected_pipeline_prefix_id=_PREFIX_ID,
        expected_stage_configuration_checksum=execution.stage_configuration_checksum,
    )
    assert restored.to_bytes() == payload
    np.testing.assert_array_equal(restored.selected_theta, execution.selected_theta)
    np.testing.assert_array_equal(restored.final_theta, execution.final_theta)

    detached = restored.selected_theta
    detached[0] = -99.0
    np.testing.assert_array_equal(restored.selected_theta, execution.selected_theta)

    resume = restored.to_noisy_krotov_resume_state()
    assert resume.content_checksum == execution.resume_state.content_checksum
    assert resume.checkpoint_selection is not None
    np.testing.assert_array_equal(resume.checkpoint_selection.theta, execution.selected_theta)
    assert resume.final_parameter_checksum == restored.final_parameter_checksum
    assert resume.cumulative_work == execution.resume_state.cumulative_work


def test_generic_transform_checkpoint_round_trips_without_noisy_resume() -> None:
    """Transform stages retain selected/final parameters without fake noisy fields."""
    checkpoint = StageParameterCheckpoint(
        pipeline_training_id=_TRAINING_ID,
        pipeline_prefix_id=_PREFIX_ID,
        stage_index=2,
        stage_id="magnitude_prune",
        stage_configuration_checksum=canonical_checksum({"stage": "prune"}),
        selected_theta=np.array([0.1, 0.2], dtype=np.float64),
        final_theta=np.array([0.3, 0.4], dtype=np.float64),
        circuit_binding_checksum=_CIRCUIT_CHECKSUM,
        stage_execution_checksum=canonical_checksum({"mask": [True, False]}),
    )
    restored = StageParameterCheckpoint.from_bytes(checkpoint.to_bytes())
    assert restored.resume_state_checksum is None
    assert restored.circuit_binding_checksum == _CIRCUIT_CHECKSUM
    assert restored.stage_execution_checksum == checkpoint.stage_execution_checksum
    np.testing.assert_array_equal(restored.selected_theta, [0.1, 0.2])
    np.testing.assert_array_equal(restored.final_theta, [0.3, 0.4])
    with pytest.raises(ValueError, match="generic transform-stage"):
        restored.to_noisy_krotov_resume_state()


@pytest.mark.parametrize("selected_fidelity", [-5e-11, 1.0 + 5e-11])
def test_checkpoint_normalizes_wp17_roundoff_fidelity(selected_fidelity: float) -> None:
    """Accepted numerical roundoff is normalized before checkpoint identity."""
    checkpoint = StageParameterCheckpoint.from_noisy_krotov(
        pipeline_training_id=_TRAINING_ID,
        pipeline_prefix_id=_PREFIX_ID,
        execution=_execution(selected_fidelity=selected_fidelity),
    )
    restored = StageParameterCheckpoint.from_bytes(checkpoint.to_bytes())
    expected = min(1.0, max(0.0, selected_fidelity))
    assert restored.selected_checkpoint_validation_fidelity == expected


def test_checkpoint_rejects_corruption_foreign_identity_and_unsafe_npy() -> None:
    """Raw checksum, exact members, vector digests, and safe dtype are enforced."""
    checkpoint = StageParameterCheckpoint.from_noisy_krotov(
        pipeline_training_id=_TRAINING_ID,
        pipeline_prefix_id=_PREFIX_ID,
        execution=_execution(),
    )
    payload = checkpoint.to_bytes()
    with pytest.raises(ValueError, match="checksum mismatch"):
        StageParameterCheckpoint.from_bytes(payload, expected_checksum=canonical_checksum({"wrong": True}))
    with pytest.raises(ValueError, match="pipeline_prefix_id"):
        StageParameterCheckpoint.from_bytes(
            payload,
            expected_pipeline_prefix_id="phase2_pipeline_prefix_" + "9" * 64,
        )

    corrupted_vector = _replace_zip_member(
        payload,
        "selected_theta.npy",
        _npy_bytes(np.array([9.0, 8.0, 7.0], dtype=np.float64)),
    )
    with pytest.raises(ValueError, match="Selected parameter checksum"):
        StageParameterCheckpoint.from_bytes(corrupted_vector)
    with pytest.raises(ValueError, match="members"):
        StageParameterCheckpoint.from_bytes(_add_zip_member(payload, "unexpected.bin", b"x"))
    with pytest.raises(ValueError, match="trailing data"):
        StageParameterCheckpoint.from_bytes(payload + b"uncommitted-tail")

    object_vector = _replace_zip_member(
        payload,
        "selected_theta.npy",
        _npy_bytes(np.array([object()], dtype=object)),
    )
    with pytest.raises(ValueError, match=r"NPY envelope|float64"):
        StageParameterCheckpoint.from_bytes(object_vector)


def test_artifact_encoders_reject_metadata_larger_than_the_decoder_limit() -> None:
    """Encoders never emit checkpoint or sidecar metadata their decoders reject."""
    checkpoint = StageParameterCheckpoint(
        pipeline_training_id=_TRAINING_ID,
        pipeline_prefix_id=_PREFIX_ID,
        stage_index=0,
        stage_id="a" * 70_000,
        stage_configuration_checksum=canonical_checksum({"stage": "large"}),
        selected_theta=np.array([0.1], dtype=np.float64),
        final_theta=np.array([0.1], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="metadata exceeds"):
        checkpoint.to_bytes()

    partitions = ({"ensemble_id": "a" * 70_000, "content_checksum": _CHECKSUM, "trajectory_count": 1},)
    with pytest.raises(ValueError, match="metadata exceeds"):
        create_phase2_trajectory_sidecar(
            evaluation_row_id=_EVALUATION_ID,
            pipeline_training_id=_TRAINING_ID,
            map_role="confirmatory_test",
            map_partitions=partitions,
            fidelities=(0.5,),
        )


def test_trajectory_sidecar_is_deterministic_identity_bound_and_bounded() -> None:
    """Phase II trajectory sidecars preserve exact role-separated evidence."""
    partitions = ({"ensemble_id": "maps_confirmatory", "content_checksum": _CHECKSUM, "trajectory_count": 3},)
    kwargs = {
        "evaluation_row_id": _EVALUATION_ID,
        "pipeline_training_id": _TRAINING_ID,
        "map_role": "confirmatory_test",
        "map_partitions": partitions,
        "fidelities": (0.1, 0.5, 0.9),
    }
    payload = create_phase2_trajectory_sidecar(**kwargs)
    assert payload == create_phase2_trajectory_sidecar(**kwargs)
    assert read_phase2_trajectory_sidecar(
        payload,
        expected_evaluation_row_id=_EVALUATION_ID,
        expected_pipeline_training_id=_TRAINING_ID,
        expected_map_role="confirmatory_test",
        expected_map_partitions=partitions,
        expected_count=3,
        expected_checksum=artifact_checksum(payload),
    ) == pytest.approx((0.1, 0.5, 0.9))

    with pytest.raises(ValueError, match="map_role"):
        read_phase2_trajectory_sidecar(
            payload,
            expected_evaluation_row_id=_EVALUATION_ID,
            expected_pipeline_training_id=_TRAINING_ID,
            expected_map_role="screening_selection",
            expected_map_partitions=partitions,
            expected_count=3,
        )
    with pytest.raises(ValueError, match="exceeds"):
        read_phase2_trajectory_sidecar(
            payload,
            expected_evaluation_row_id=_EVALUATION_ID,
            expected_pipeline_training_id=_TRAINING_ID,
            expected_map_role="confirmatory_test",
            expected_map_partitions=partitions,
            expected_count=MAX_TRAJECTORY_FIDELITY_COUNT + 1,
        )


def test_trajectory_sidecar_rejects_corrupt_values_and_unsupported_members() -> None:
    """Fidelity sidecars reject checksum changes, unsafe values, and schema drift."""
    partitions = ({"ensemble_id": "maps_screening", "content_checksum": _CHECKSUM, "trajectory_count": 2},)
    payload = create_phase2_trajectory_sidecar(
        evaluation_row_id=_EVALUATION_ID,
        pipeline_training_id=_TRAINING_ID,
        map_role="screening_selection",
        map_partitions=partitions,
        fidelities=(0.2, 0.4),
    )
    corrupted = _replace_zip_member(
        payload,
        "fidelities.npy",
        _npy_bytes(np.array([0.2, 1.4], dtype=np.float64)),
    )
    with pytest.raises(ValueError, match=r"checksum|lie in"):
        read_phase2_trajectory_sidecar(
            corrupted,
            expected_evaluation_row_id=_EVALUATION_ID,
            expected_pipeline_training_id=_TRAINING_ID,
            expected_map_role="screening_selection",
            expected_map_partitions=partitions,
            expected_count=2,
        )
    with pytest.raises(ValueError, match="members"):
        read_phase2_trajectory_sidecar(
            _add_zip_member(payload, "unexpected.bin", b"x"),
            expected_evaluation_row_id=_EVALUATION_ID,
            expected_pipeline_training_id=_TRAINING_ID,
            expected_map_role="screening_selection",
            expected_map_partitions=partitions,
            expected_count=2,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        create_phase2_trajectory_sidecar(
            evaluation_row_id=_EVALUATION_ID,
            pipeline_training_id=_TRAINING_ID,
            map_role="screening_selection",
            map_partitions=partitions,
            fidelities=(0.2, float("nan")),
        )
