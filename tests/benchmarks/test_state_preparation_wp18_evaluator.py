# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused tests for the WP18 parallel Phase II evaluator."""

from __future__ import annotations

import hashlib
import threading
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from typing_extensions import override

import benchmarks.state_preparation.phase2.artifacts as artifact_module
from benchmarks.state_preparation.phase2.artifacts import (
    MaterializationAttemptArtifact,
    MaterializedCircuitArtifact,
    Phase2ArtifactStore,
)
from benchmarks.state_preparation.phase2.evaluator import (
    MaterializedCircuitPayload,
    ParallelPhase2Evaluator,
    PipelineEvaluationMeasurement,
)
from benchmarks.state_preparation.phase2.pipeline import (
    PipelineBenchmarkFailure,
    PipelineBenchmarkResult,
    PipelineEvaluationConfig,
    TrainingPipelineResult,
)
from mqt.yaqs.optimization import KrotovFixedMapEnsemble, KrotovNoiseMap

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal

    from benchmarks.state_preparation.phase2.pipeline import (
        PipelineBenchmarkRecord,
    )
    from mqt.yaqs.optimization import KrotovMapRole

_CHECKSUM = "sha256:" + "a" * 64
_RUNTIME_CHECKSUM = "sha256:" + "b" * 64
_PIPELINE_ID = "phase2_training_" + "1" * 64
_CIRCUIT_ID = "phase2_circuit_" + "2" * 64
_CIRCUIT_BYTES = b"deterministic final circuit"
_CIRCUIT_CHECKSUM = f"sha256:{hashlib.sha256(_CIRCUIT_BYTES).hexdigest()}"


def _work(*, test_trajectories: int) -> dict[str, object]:
    """Return an evaluation-only normalized-work ledger."""
    return {
        "objective_evaluations": 1,
        "gradient_evaluations": 0,
        "training_trajectories": 0,
        "checkpoint_validation_trajectories": 0,
        "test_trajectories": test_trajectories,
        "trajectory_gate_applications": test_trajectories,
    }


class _PipelineResult(TrainingPipelineResult):
    """Minimal typed complete-pipeline token for evaluator isolation tests."""


class _EvaluationConfig(PipelineEvaluationConfig):
    """Minimal typed final-test configuration with stable test identities."""

    def __init__(self, repetition: int, *, noisy: bool = True, data_role: str = "screening_selection") -> None:
        """Populate only fields observed across the evaluator/store boundary."""
        object.__setattr__(self, "repetition", repetition)
        object.__setattr__(self, "test_noise_id", "depolarizing_1s_all" if noisy else "noiseless")
        object.__setattr__(self, "trajectory_budget", 2 if noisy else 0)
        object.__setattr__(self, "data_role", data_role)
        object.__setattr__(self, "materialized_circuit_id", _CIRCUIT_ID)
        object.__setattr__(self, "materialized_circuit_checksum", _CIRCUIT_CHECKSUM)
        object.__setattr__(self, "final_materialization_policy_checksum", _CHECKSUM)
        object.__setattr__(self, "sidecar_storage_policy", "none")
        object.__setattr__(self, "evaluation_policy", "fixed_sample")
        object.__setattr__(self, "confidence_level", None)
        object.__setattr__(self, "confidence_interval_method", None)

    @property
    @override
    def evaluation_row_id(self) -> str:
        """Stable identity derived from the test repetition."""
        return f"phase2_evaluation_{self.repetition:064x}"

    @property
    @override
    def configuration_checksum(self) -> str:
        """Stable planned-row checksum used by map bindings."""
        return _CHECKSUM

    @override
    def validate_against_pipeline(self, pipeline: TrainingPipelineResult) -> None:
        """Accept the deliberately minimal typed pipeline test token."""
        assert self.repetition >= 0
        assert isinstance(pipeline, TrainingPipelineResult)


def _ensemble(
    config: _EvaluationConfig,
    *,
    role: str = "screening_selection",
) -> KrotovFixedMapEnsemble:
    """Create exact identity-map evidence covering a noisy test budget.

    Returns:
        A checksum-sealed ensemble with one map per planned trajectory.
    """
    noise_map = KrotovNoiseMap(source_gate_index=0, is_identity=True)
    return KrotovFixedMapEnsemble(
        role=cast("KrotovMapRole", role),
        resolved_seed=101 + config.repetition,
        stage_index=0,
        stage_id="final_evaluation",
        stage_configuration_checksum=_CHECKSUM,
        circuit_checksum=_CIRCUIT_CHECKSUM,
        provider_checksum=_RUNTIME_CHECKSUM,
        ensemble_index=config.repetition,
        refresh_index=0,
        global_iteration_start=0,
        trajectory_maps=[[noise_map] for _ in range(config.trajectory_budget)],
    )


def _measurement(
    config: _EvaluationConfig,
    *,
    role: str = "screening_selection",
) -> PipelineEvaluationMeasurement:
    """Build one valid noisy final-evaluation measurement.

    Returns:
        Exact final-test evidence for ``config``.
    """
    ensemble = _ensemble(config, role=role)
    return PipelineEvaluationMeasurement(
        noiseless_fidelity=0.95,
        trajectory_fidelities=(0.8, 0.9),
        sampled_nonidentity_events=0,
        provider_checksum=_RUNTIME_CHECKSUM,
        normalized_work=_work(test_trajectories=2),
        fixed_map_ensembles=(ensemble,),
        wall_time_seconds=1.5,
        peak_memory_bytes=2048,
    )


class _RecordingStore(Phase2ArtifactStore):
    """In-memory typed store double recording all evaluator publications."""

    def __init__(self, *, complete: bool = True) -> None:
        """Create a complete or intentionally incomplete store double."""
        self._test_pipeline_result = object.__new__(_PipelineResult) if complete else None
        self._test_parameters = np.array([0.1, 0.2], dtype=np.float64)
        self.materialization_calls: list[tuple[str, bytes, float, int]] = []
        self.publication_order: list[str] = []
        self.failure_phases: list[str] = []
        self.materialization_failure_attempts = 0
        self._test_records: list[PipelineBenchmarkRecord] = []

    @property
    @override
    def pipeline_result(self) -> TrainingPipelineResult | None:
        """Configured complete-pipeline token."""
        return self._test_pipeline_result

    @property
    @override
    def records(self) -> tuple[PipelineBenchmarkRecord, ...]:
        """Current in-memory evaluation rows."""
        return tuple(self._test_records)

    @override
    def load_final_parameters(self) -> np.ndarray:
        """Return detached selected parameters."""
        return self._test_parameters.copy()

    @override
    def require_fresh_handle(self) -> None:
        """Model a store double that retains its in-memory commit baseline."""

    def publish_materialized_circuit(
        self,
        *,
        config: PipelineEvaluationConfig,
        payload: bytes,
        wall_time_seconds: float,
        peak_memory_bytes: int,
    ) -> MaterializedCircuitArtifact:
        """Record and return one typed materialization artifact.

        Returns:
            The linked in-memory materialization record.
        """
        self.materialization_calls.append((config.evaluation_row_id, payload, wall_time_seconds, peak_memory_bytes))
        return MaterializedCircuitArtifact(
            materialized_circuit_id=config.materialized_circuit_id,
            pipeline_training_id=_PIPELINE_ID,
            pipeline_result_checksum=_CHECKSUM,
            final_checkpoint_checksum=_CHECKSUM,
            materialization_policy_checksum=config.final_materialization_policy_checksum,
            path=f"circuits/{config.materialized_circuit_id}.bin",
            payload_checksum=config.materialized_circuit_checksum,
            wall_time_seconds=wall_time_seconds,
            peak_memory_bytes=peak_memory_bytes,
            runtime_fingerprint_checksum=_RUNTIME_CHECKSUM,
        )

    def record_materialization_failure(
        self,
        *,
        config: PipelineEvaluationConfig,
        exception: BaseException,
        phase: Literal["materialization", "serialization"],
        wall_time_seconds: float,
        peak_memory_bytes: int = 0,
    ) -> MaterializationAttemptArtifact:
        """Record exactly one shared failed materialization attempt.

        Returns:
            A typed in-memory attempt artifact.
        """
        self.materialization_failure_attempts += 1
        return MaterializationAttemptArtifact(
            materialized_circuit_id=config.materialized_circuit_id,
            pipeline_training_id=_PIPELINE_ID,
            pipeline_result_checksum=_CHECKSUM,
            attempt=self.materialization_failure_attempts,
            status="failure",
            phase=phase,
            payload_checksum=None,
            exception_type=type(exception).__name__,
            message=str(exception),
            wall_time_seconds=wall_time_seconds,
            peak_memory_bytes=peak_memory_bytes,
            runtime_fingerprint_checksum=_RUNTIME_CHECKSUM,
        )

    def write_evaluation_success(
        self,
        *,
        config: PipelineEvaluationConfig,
        materialization: MaterializedCircuitArtifact,
        test_noiseless_fidelity: float,
        trajectory_fidelities: Sequence[float],
        sampled_nonidentity_events: int,
        normalized_work: Mapping[str, object],
        evaluation_wall_time_seconds: float,
        peak_memory_bytes: int,
        evaluation_provider_checksum: str | None,
        evaluation_ensembles: Sequence[KrotovFixedMapEnsemble] = (),
    ) -> PipelineBenchmarkResult:
        """Record one success in canonical input order.

        Returns:
            A typed successful benchmark row.
        """
        del trajectory_fidelities, evaluation_provider_checksum, evaluation_ensembles
        self.publication_order.append(config.evaluation_row_id)
        record = PipelineBenchmarkResult(
            config=config,
            materialized_circuit_path=materialization.path,
            test_noiseless_fidelity=test_noiseless_fidelity,
            test_noisy_fidelity=0.85,
            noisy_fidelity_standard_deviation=0.05,
            noisy_fidelity_standard_error=0.025,
            confidence_interval_lower=None,
            confidence_interval_upper=None,
            sampled_nonidentity_events=sampled_nonidentity_events,
            trajectory_sidecar_path=None,
            trajectory_sidecar_checksum=None,
            evaluation_wall_time_seconds=evaluation_wall_time_seconds,
            peak_memory_bytes=peak_memory_bytes,
            normalized_work=normalized_work,
            runtime_fingerprint_checksum=_RUNTIME_CHECKSUM,
        )
        self._test_records = [item for item in self._test_records if item.evaluation_row_id != record.evaluation_row_id]
        self._test_records.append(record)
        return record

    def write_evaluation_failure(
        self,
        *,
        config: PipelineEvaluationConfig,
        exception: BaseException,
        phase: Literal["pipeline_loading", "materialization", "evaluation", "serialization"],
        wall_time_seconds: float,
        materialization: MaterializedCircuitArtifact | None = None,
        retryable: bool = False,
    ) -> PipelineBenchmarkFailure:
        """Record one structured failure in canonical input order.

        Returns:
            A typed failed benchmark row.
        """
        del retryable
        self.publication_order.append(config.evaluation_row_id)
        self.failure_phases.append(phase)
        record = PipelineBenchmarkFailure.from_exception(
            config=config,
            failure_phase=phase,
            exception=exception,
            runtime_fingerprint_checksum=_RUNTIME_CHECKSUM,
            materialized_circuit_path=None if materialization is None else materialization.path,
            materialized_circuit_checksum=(None if materialization is None else materialization.payload_checksum),
            wall_time_seconds=wall_time_seconds,
        )
        self._test_records = [item for item in self._test_records if item.evaluation_row_id != record.evaluation_row_id]
        self._test_records.append(record)
        return record


def _materialized_payload() -> MaterializedCircuitPayload:
    """Return deterministic circuit bytes and explicit resource observations."""
    return MaterializedCircuitPayload(
        serialized_bytes=_CIRCUIT_BYTES,
        wall_time_seconds=2.5,
        peak_memory_bytes=4096,
    )


def test_measurement_is_immutable_and_enforces_budget_work_and_map_role() -> None:
    """Exact row evidence rejects cross-role maps and mismatched trajectory work."""
    config = _EvaluationConfig(1)
    mutable_work = _work(test_trajectories=2)
    measurement = PipelineEvaluationMeasurement(
        noiseless_fidelity=0.95,
        trajectory_fidelities=(0.8, 0.9),
        sampled_nonidentity_events=0,
        provider_checksum=_RUNTIME_CHECKSUM,
        normalized_work=mutable_work,
        fixed_map_ensembles=(_ensemble(config),),
        wall_time_seconds=1.0,
        peak_memory_bytes=10,
    )
    mutable_work["test_trajectories"] = 999
    assert measurement.normalized_work["test_trajectories"] == 2
    measurement.validate_against_config(config)

    wrong_role = _measurement(config, role="training_trajectory")
    with pytest.raises(ValueError, match="reserved 'screening_selection' role"):
        wrong_role.validate_against_config(config)
    with pytest.raises(TypeError, match="immutable bytes"):
        MaterializedCircuitPayload(
            serialized_bytes=cast("bytes", bytearray(b"mutable")),
            wall_time_seconds=0.0,
            peak_memory_bytes=0,
        )


def test_parallel_evaluator_materializes_once_and_publishes_in_input_order() -> None:
    """Reverse worker completion cannot reorder the canonical result stream."""
    store = _RecordingStore()
    runtime_circuit = object()
    decoder_payloads: list[bytes] = []

    def deserialize(payload: bytes) -> object:
        decoder_payloads.append(payload)
        return runtime_circuit

    evaluator = ParallelPhase2Evaluator(store, deserialize)
    configs = tuple(_EvaluationConfig(index) for index in range(3))
    barrier = threading.Barrier(3)
    second_finished = threading.Event()
    first_finished = threading.Event()
    callback_completion: list[int] = []
    materialization_count = 0

    def materialize(
        pipeline: TrainingPipelineResult,
        selected_parameters: np.ndarray,
    ) -> MaterializedCircuitPayload:
        nonlocal materialization_count
        assert pipeline is store.pipeline_result
        assert selected_parameters == pytest.approx([0.1, 0.2])
        selected_parameters[0] = 99.0
        materialization_count += 1
        return _materialized_payload()

    def evaluate(config: PipelineEvaluationConfig, circuit: object) -> PipelineEvaluationMeasurement:
        assert circuit is runtime_circuit
        barrier.wait(timeout=5.0)
        if config.repetition == 0:
            assert first_finished.wait(timeout=5.0)
        elif config.repetition == 1:
            assert second_finished.wait(timeout=5.0)
            first_finished.set()
        else:
            second_finished.set()
        callback_completion.append(config.repetition)
        return _measurement(cast("_EvaluationConfig", config))

    records = evaluator.evaluate(configs, materialize, evaluate, max_workers=3)
    assert materialization_count == 1
    assert decoder_payloads == [_CIRCUIT_BYTES]
    assert len(store.materialization_calls) == 1
    assert store.load_final_parameters() == pytest.approx([0.1, 0.2])
    assert callback_completion == [2, 1, 0]
    assert store.publication_order == [config.evaluation_row_id for config in configs]
    assert [record.evaluation_row_id for record in records] == store.publication_order
    assert all(isinstance(record, PipelineBenchmarkResult) for record in records)


def test_row_exception_and_role_leak_become_ordered_linked_failures() -> None:
    """One failed callback and one leaked training map do not cancel peer rows."""
    store = _RecordingStore()
    evaluator = ParallelPhase2Evaluator(store, lambda payload: payload)
    configs = tuple(_EvaluationConfig(index) for index in range(3))

    def evaluate(config: PipelineEvaluationConfig, _circuit: object) -> PipelineEvaluationMeasurement:
        typed = cast("_EvaluationConfig", config)
        if typed.repetition == 1:
            msg = "trajectory worker stopped"
            raise RuntimeError(msg)
        if typed.repetition == 2:
            return _measurement(typed, role="checkpoint_validation")
        return _measurement(typed)

    records = evaluator.evaluate(
        configs,
        lambda _pipeline, _parameters: _materialized_payload(),
        evaluate,
        max_workers=3,
    )
    assert [type(record) for record in records] == [
        PipelineBenchmarkResult,
        PipelineBenchmarkFailure,
        PipelineBenchmarkFailure,
    ]
    assert store.publication_order == [config.evaluation_row_id for config in configs]
    assert store.failure_phases == ["evaluation", "evaluation"]
    assert cast("PipelineBenchmarkFailure", records[1]).message == "trajectory worker stopped"
    assert "reserved 'screening_selection' role" in cast("PipelineBenchmarkFailure", records[2]).message


def test_materialization_failure_creates_one_failure_per_planned_row() -> None:
    """A shared materialization failure retains the complete planned fan-out."""
    store = _RecordingStore()
    evaluator = ParallelPhase2Evaluator(store, lambda payload: payload)
    configs = tuple(_EvaluationConfig(index) for index in range(3))
    evaluation_called = False

    def fail_materialization(
        _pipeline: TrainingPipelineResult,
        _parameters: np.ndarray,
    ) -> MaterializedCircuitPayload:
        msg = "compiler stopped"
        raise RuntimeError(msg)

    def evaluate(_config: PipelineEvaluationConfig, _circuit: object) -> PipelineEvaluationMeasurement:
        nonlocal evaluation_called
        evaluation_called = True
        return _measurement(configs[0])

    records = evaluator.evaluate(configs, fail_materialization, evaluate, max_workers=2)
    assert not evaluation_called
    assert not store.materialization_calls
    assert all(isinstance(record, PipelineBenchmarkFailure) for record in records)
    assert store.publication_order == [config.evaluation_row_id for config in configs]
    assert store.failure_phases == ["materialization"] * len(configs)
    assert all(cast("PipelineBenchmarkFailure", record).message == "compiler stopped" for record in records)
    assert store.materialization_failure_attempts == 1
    assert all(cast("PipelineBenchmarkFailure", record).wall_time_seconds == pytest.approx(0.0) for record in records)


def test_oversized_materialization_is_rejected_before_the_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared store bound protects the trusted decoder and one fan-out attempt."""
    monkeypatch.setattr(artifact_module, "_MAX_CIRCUIT_SIZE", len(_CIRCUIT_BYTES) - 1)
    store = _RecordingStore()
    decoded = False

    def unexpected_decoder(_payload: bytes) -> object:
        nonlocal decoded
        decoded = True
        return object()

    evaluator = ParallelPhase2Evaluator(store, unexpected_decoder)
    configs = tuple(_EvaluationConfig(index) for index in range(2))
    records = evaluator.evaluate(
        configs,
        lambda _pipeline, _parameters: _materialized_payload(),
        lambda config, _circuit: _measurement(cast("_EvaluationConfig", config)),
        max_workers=2,
    )

    assert not decoded
    assert not store.materialization_calls
    assert store.materialization_failure_attempts == 1
    assert store.failure_phases == ["materialization", "materialization"]
    assert all(isinstance(record, PipelineBenchmarkFailure) for record in records)
    assert all("verification limit" in cast("PipelineBenchmarkFailure", record).message for record in records)


def test_trusted_decoder_failure_is_one_serialization_attempt() -> None:
    """Only verified bytes reach the trusted decoder and decode failure fans out safely."""
    store = _RecordingStore()
    decoded_payloads: list[bytes] = []

    def fail_decode(payload: bytes) -> object:
        decoded_payloads.append(payload)
        msg = "unsupported deterministic circuit encoding"
        raise ValueError(msg)

    evaluator = ParallelPhase2Evaluator(store, fail_decode)
    configs = tuple(_EvaluationConfig(index) for index in range(2))
    evaluation_called = False

    def unexpected_evaluation(
        _config: PipelineEvaluationConfig,
        _circuit: object,
    ) -> PipelineEvaluationMeasurement:
        nonlocal evaluation_called
        evaluation_called = True
        return _measurement(configs[0])

    records = evaluator.evaluate(
        configs,
        lambda _pipeline, _parameters: _materialized_payload(),
        unexpected_evaluation,
        max_workers=2,
    )
    assert decoded_payloads == [_CIRCUIT_BYTES]
    assert not evaluation_called
    assert store.materialization_failure_attempts == 1
    assert store.failure_phases == ["serialization", "serialization"]
    assert all(isinstance(record, PipelineBenchmarkFailure) for record in records)


def test_resume_skips_successful_rows_and_retries_failed_and_missing_rows() -> None:
    """Resume leaves successes untouched while retrying only unfinished rows."""
    store = _RecordingStore()
    evaluator = ParallelPhase2Evaluator(store, lambda payload: payload)
    configs = tuple(_EvaluationConfig(index) for index in range(3))
    first_evaluation_calls: list[int] = []

    def initial_evaluation(
        config: PipelineEvaluationConfig,
        _circuit: object,
    ) -> PipelineEvaluationMeasurement:
        first_evaluation_calls.append(config.repetition)
        if config.repetition == 1:
            msg = "retry this row"
            raise RuntimeError(msg)
        return _measurement(cast("_EvaluationConfig", config))

    initial_records = evaluator.evaluate(
        configs[:2],
        lambda _pipeline, _parameters: _materialized_payload(),
        initial_evaluation,
        max_workers=2,
    )
    retained_success = initial_records[0]
    assert sorted(first_evaluation_calls) == [0, 1]
    assert isinstance(retained_success, PipelineBenchmarkResult)
    assert isinstance(initial_records[1], PipelineBenchmarkFailure)

    materialization_calls = 0
    retried_repetitions: list[int] = []

    def resume_materialization(
        _pipeline: TrainingPipelineResult,
        _parameters: np.ndarray,
    ) -> MaterializedCircuitPayload:
        nonlocal materialization_calls
        materialization_calls += 1
        return _materialized_payload()

    def resume_evaluation(
        config: PipelineEvaluationConfig,
        _circuit: object,
    ) -> PipelineEvaluationMeasurement:
        retried_repetitions.append(config.repetition)
        return _measurement(cast("_EvaluationConfig", config))

    resumed = evaluator.evaluate(
        configs,
        resume_materialization,
        resume_evaluation,
        max_workers=2,
    )
    assert materialization_calls == 1
    assert sorted(retried_repetitions) == [1, 2]
    assert resumed[0] is retained_success
    assert all(isinstance(record, PipelineBenchmarkResult) for record in resumed)
    assert [record.evaluation_row_id for record in resumed] == [config.evaluation_row_id for config in configs]

    callbacks_after_completion = 0

    def unexpected_materialization(
        _pipeline: TrainingPipelineResult,
        _parameters: np.ndarray,
    ) -> MaterializedCircuitPayload:
        nonlocal callbacks_after_completion
        callbacks_after_completion += 1
        msg = "completed rows must not be replayed"
        raise AssertionError(msg)

    def unexpected_evaluation(
        _config: PipelineEvaluationConfig,
        _circuit: object,
    ) -> PipelineEvaluationMeasurement:
        nonlocal callbacks_after_completion
        callbacks_after_completion += 1
        msg = "completed rows must not be replayed"
        raise AssertionError(msg)

    replayed = evaluator.evaluate(
        configs,
        unexpected_materialization,
        unexpected_evaluation,
        max_workers=2,
    )
    assert replayed == resumed
    assert callbacks_after_completion == 0


def test_evaluator_requires_a_complete_pipeline_before_any_callback() -> None:
    """Final-test access is impossible before the full stage prefix verifies."""
    store = _RecordingStore(complete=False)
    evaluator = ParallelPhase2Evaluator(store, lambda payload: payload)
    callback_called = False

    def materialize(
        _pipeline: TrainingPipelineResult,
        _parameters: np.ndarray,
    ) -> MaterializedCircuitPayload:
        nonlocal callback_called
        callback_called = True
        return _materialized_payload()

    with pytest.raises(RuntimeError, match="complete verified training pipeline"):
        evaluator.evaluate(
            (_EvaluationConfig(0),),
            materialize,
            lambda config, _circuit: _measurement(cast("_EvaluationConfig", config)),
            max_workers=1,
        )
    assert not callback_called
