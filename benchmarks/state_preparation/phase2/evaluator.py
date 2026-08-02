# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Parallel final evaluation for completed Phase II training pipelines.

The evaluator deliberately exposes only the selected final parameters and a
caller-owned materialized circuit to callbacks.  Training and
checkpoint-validation fixed maps therefore cannot cross the final-test API
boundary.  Worker completion order is also kept separate from publication
order: canonical rows are always committed in the order supplied by the
caller.
"""

# The private validators below share the strict public contracts documented on
# the immutable value types; repeating every propagated exception and return
# description would obscure the scientific API.
# ruff: noqa: DOC201, DOC501

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from benchmarks.state_preparation.constants import NOISELESS_NOISE_ID
from mqt.yaqs.optimization import KrotovFixedMapEnsemble

from .artifacts import (
    MaterializedCircuitArtifact,
    Phase2ArtifactStore,
    Phase2ConcurrentMutationError,
    Phase2DerivedArtifactError,
    validate_materialized_circuit_payload,
)
from .pipeline import (
    PipelineBenchmarkResult,
    PipelineEvaluationConfig,
    TrainingPipelineResult,
)
from .validation import require_checksum

if TYPE_CHECKING:
    from .pipeline import (
        PipelineBenchmarkFailure,
        PipelineBenchmarkRecord,
    )

_WORK_FIELDS = frozenset({
    "objective_evaluations",
    "gradient_evaluations",
    "training_trajectories",
    "checkpoint_validation_trajectories",
    "test_trajectories",
    "trajectory_gate_applications",
})
_EVALUATION_ROLE_BY_DATA_ROLE: Mapping[str, str] = MappingProxyType({
    "development": "pilot_evaluation",
    "checkpoint_validation": "checkpoint_validation",
    "screening_selection": "screening_selection",
    "secondary_benchmark": "pilot_evaluation",
    "confirmatory": "confirmatory_test",
})


def _require_nonnegative_float(value: object, name: str) -> float:
    """Return one finite nonnegative built-in float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"{name} must be a real number."
        raise TypeError(msg)
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        msg = f"{name} must be finite and nonnegative."
        raise ValueError(msg)
    return result


def _require_nonnegative_int(value: object, name: str) -> int:
    """Return one nonnegative built-in integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        msg = f"{name} must be an integer."
        raise TypeError(msg)
    result = int(value)
    if result < 0:
        msg = f"{name} must be nonnegative."
        raise ValueError(msg)
    return result


def _require_fidelity(value: object, name: str) -> float:
    """Return one finite fidelity in the closed unit interval."""
    result = _require_nonnegative_float(value, name)
    if result > 1.0:
        msg = f"{name} must lie in [0, 1]."
        raise ValueError(msg)
    return result


def _freeze_work(value: object) -> Mapping[str, int]:
    """Validate and freeze the exact Phase II normalized-work ledger."""
    if not isinstance(value, Mapping):
        msg = "normalized_work must be a mapping."
        raise TypeError(msg)
    if set(value) != _WORK_FIELDS:
        msg = "normalized_work fields do not match the Phase II work ledger."
        raise ValueError(msg)
    typed_value = cast("Mapping[str, object]", value)
    return MappingProxyType({
        name: _require_nonnegative_int(typed_value[name], f"normalized_work.{name}") for name in sorted(_WORK_FIELDS)
    })


@dataclass(frozen=True, slots=True)
class MaterializedCircuitPayload:
    """Deterministic persisted circuit representation and resource evidence."""

    serialized_bytes: bytes = field(repr=False)
    wall_time_seconds: float
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        """Validate immutable bytes and explicit resource observations."""
        if type(self.serialized_bytes) is not bytes:
            msg = "serialized_bytes must be immutable bytes."
            raise TypeError(msg)
        if not self.serialized_bytes:
            msg = "serialized_bytes must not be empty."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "wall_time_seconds",
            _require_nonnegative_float(self.wall_time_seconds, "wall_time_seconds"),
        )
        object.__setattr__(
            self,
            "peak_memory_bytes",
            _require_nonnegative_int(self.peak_memory_bytes, "peak_memory_bytes"),
        )

    @property
    def payload_checksum(self) -> str:
        """Checksum of the exact deterministic circuit bytes."""
        return f"sha256:{hashlib.sha256(self.serialized_bytes).hexdigest()}"


@dataclass(frozen=True, slots=True)
class PipelineEvaluationMeasurement:
    """Exact in-memory evidence returned by one final-evaluation callback."""

    noiseless_fidelity: float
    trajectory_fidelities: tuple[float, ...]
    sampled_nonidentity_events: int
    provider_checksum: str | None
    normalized_work: Mapping[str, object]
    fixed_map_ensembles: tuple[KrotovFixedMapEnsemble, ...]
    wall_time_seconds: float
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        """Validate and defensively freeze all row evidence."""
        object.__setattr__(
            self,
            "noiseless_fidelity",
            _require_fidelity(self.noiseless_fidelity, "noiseless_fidelity"),
        )
        fidelities = tuple(
            _require_fidelity(value, f"trajectory_fidelities[{index}]")
            for index, value in enumerate(self.trajectory_fidelities)
        )
        object.__setattr__(self, "trajectory_fidelities", fidelities)
        object.__setattr__(
            self,
            "sampled_nonidentity_events",
            _require_nonnegative_int(self.sampled_nonidentity_events, "sampled_nonidentity_events"),
        )
        if self.provider_checksum is not None:
            object.__setattr__(self, "provider_checksum", require_checksum(self.provider_checksum, "provider_checksum"))
        object.__setattr__(self, "normalized_work", _freeze_work(self.normalized_work))
        ensembles = tuple(self.fixed_map_ensembles)
        if not all(isinstance(ensemble, KrotovFixedMapEnsemble) for ensemble in ensembles):
            msg = "fixed_map_ensembles must contain only KrotovFixedMapEnsemble values."
            raise TypeError(msg)
        if len({ensemble.ensemble_id for ensemble in ensembles}) != len(ensembles):
            msg = "fixed_map_ensembles must not repeat an ensemble identity."
            raise ValueError(msg)
        if len({ensemble.content_checksum for ensemble in ensembles}) != len(ensembles):
            msg = "fixed_map_ensembles must not repeat ensemble content."
            raise ValueError(msg)
        object.__setattr__(self, "fixed_map_ensembles", ensembles)
        object.__setattr__(
            self,
            "wall_time_seconds",
            _require_nonnegative_float(self.wall_time_seconds, "wall_time_seconds"),
        )
        object.__setattr__(
            self,
            "peak_memory_bytes",
            _require_nonnegative_int(self.peak_memory_bytes, "peak_memory_bytes"),
        )

    def validate_against_config(self, config: PipelineEvaluationConfig) -> None:
        """Verify trajectory counts, map roles, and work against one row.

        Args:
            config: Planned final-evaluation cell.

        Raises:
            TypeError: If ``config`` has the wrong type.
            ValueError: If the evidence does not implement that cell exactly.
        """
        if not isinstance(config, PipelineEvaluationConfig):
            msg = "config must be a PipelineEvaluationConfig."
            raise TypeError(msg)
        noisy = config.test_noise_id != NOISELESS_NOISE_ID
        expected_count = config.trajectory_budget if noisy else 0
        if len(self.trajectory_fidelities) != expected_count:
            msg = "trajectory_fidelities must contain exactly the configured final-test trajectories."
            raise ValueError(msg)
        if self.normalized_work["test_trajectories"] != expected_count:
            msg = "normalized test-trajectory work must equal the configured final-test budget."
            raise ValueError(msg)
        if self.normalized_work["training_trajectories"] != 0:
            msg = "Final evaluation cannot report training-trajectory work."
            raise ValueError(msg)
        if self.normalized_work["checkpoint_validation_trajectories"] != 0:
            msg = "Final evaluation cannot report checkpoint-validation work."
            raise ValueError(msg)
        if not noisy:
            if self.fixed_map_ensembles or self.sampled_nonidentity_events != 0 or self.provider_checksum is not None:
                msg = "Noiseless evaluation cannot contain fixed maps, a provider, or sampled noise events."
                raise ValueError(msg)
            return
        if self.provider_checksum is None:
            msg = "Noisy final evaluation requires an exact provider checksum."
            raise ValueError(msg)
        if not self.fixed_map_ensembles:
            msg = "Noisy final evaluation requires persisted fixed-map evidence."
            raise ValueError(msg)
        expected_role = _EVALUATION_ROLE_BY_DATA_ROLE[config.data_role]
        if any(ensemble.role != expected_role for ensemble in self.fixed_map_ensembles):
            msg = f"Final-evaluation maps must use the reserved {expected_role!r} role."
            raise ValueError(msg)
        if any(
            ensemble.stage_configuration_checksum != config.configuration_checksum
            or ensemble.circuit_checksum != config.materialized_circuit_checksum
            or ensemble.provider_checksum != self.provider_checksum
            for ensemble in self.fixed_map_ensembles
        ):
            msg = "Final-evaluation maps must bind the planned row, circuit, and noise provider."
            raise ValueError(msg)
        if sum(ensemble.trajectory_count for ensemble in self.fixed_map_ensembles) != expected_count:
            msg = "Fixed-map trajectory counts must equal the configured final-test budget."
            raise ValueError(msg)
        if sum(ensemble.nonidentity_event_count for ensemble in self.fixed_map_ensembles) != (
            self.sampled_nonidentity_events
        ):
            msg = "sampled_nonidentity_events must equal the persisted fixed-map evidence."
            raise ValueError(msg)


MaterializeCallback = Callable[
    [TrainingPipelineResult, NDArray[np.float64]],
    MaterializedCircuitPayload,
]
DeserializeCircuitCallback = Callable[[bytes], object]
EvaluateCallback = Callable[
    [PipelineEvaluationConfig, object],
    PipelineEvaluationMeasurement,
]


@dataclass(frozen=True, slots=True)
class _WorkerOutcome:
    """Private result of one concurrent row callback."""

    measurement: PipelineEvaluationMeasurement | None
    exception: BaseException | None
    observed_wall_time_seconds: float


class ParallelPhase2Evaluator:
    """Materialize once and evaluate final-test rows concurrently."""

    def __init__(
        self,
        store: Phase2ArtifactStore,
        deserialize_circuit: DeserializeCircuitCallback,
    ) -> None:
        """Bind the evaluator to a store and trusted canonical circuit decoder."""
        if not isinstance(store, Phase2ArtifactStore):
            msg = "store must be a Phase2ArtifactStore."
            raise TypeError(msg)
        if not callable(deserialize_circuit):
            msg = "deserialize_circuit must be callable."
            raise TypeError(msg)
        self.store = store
        self.deserialize_circuit = deserialize_circuit

    @staticmethod
    def _validate_configs(
        configs: Sequence[PipelineEvaluationConfig],
        pipeline: TrainingPipelineResult,
    ) -> tuple[PipelineEvaluationConfig, ...]:
        """Validate one ordered fan-out sharing an exact materialization."""
        if isinstance(configs, (str, bytes)) or not isinstance(configs, Sequence):
            msg = "configs must be a sequence of PipelineEvaluationConfig values."
            raise TypeError(msg)
        ordered = tuple(configs)
        if not ordered:
            msg = "configs must contain at least one evaluation row."
            raise ValueError(msg)
        if not all(isinstance(config, PipelineEvaluationConfig) for config in ordered):
            msg = "configs must contain only PipelineEvaluationConfig values."
            raise TypeError(msg)
        for config in ordered:
            config.validate_against_pipeline(pipeline)
        row_ids = [config.evaluation_row_id for config in ordered]
        if len(row_ids) != len(set(row_ids)):
            msg = "configs must not repeat an evaluation-row identity."
            raise ValueError(msg)
        materializations = {
            (config.materialized_circuit_id, config.materialized_circuit_checksum) for config in ordered
        }
        if len(materializations) != 1:
            msg = "All evaluation rows must identify the same materialized circuit."
            raise ValueError(msg)
        return ordered

    @staticmethod
    def _checked_measurement(
        value: object,
        config: PipelineEvaluationConfig,
    ) -> PipelineEvaluationMeasurement:
        """Require a typed measurement implementing the planned row."""
        if not isinstance(value, PipelineEvaluationMeasurement):
            msg = "evaluate callback must return a PipelineEvaluationMeasurement."
            raise TypeError(msg)
        value.validate_against_config(config)
        return value

    @staticmethod
    def _evaluate_one(
        callback: EvaluateCallback,
        config: PipelineEvaluationConfig,
        runtime_circuit: object,
    ) -> _WorkerOutcome:
        """Execute and validate one callback without publishing from a worker."""
        started = time.perf_counter()
        try:
            measurement = ParallelPhase2Evaluator._checked_measurement(
                callback(config, runtime_circuit),
                config,
            )
        except Exception as error:  # noqa: BLE001 - row failures are the public contract
            return _WorkerOutcome(
                measurement=None,
                exception=error,
                observed_wall_time_seconds=time.perf_counter() - started,
            )
        return _WorkerOutcome(
            measurement=measurement,
            exception=None,
            observed_wall_time_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _checked_materialization(
        value: object,
        expected_checksum: str,
    ) -> MaterializedCircuitPayload:
        """Require typed materialization bytes matching the planned circuit."""
        if not isinstance(value, MaterializedCircuitPayload):
            msg = "materialize callback must return a MaterializedCircuitPayload."
            raise TypeError(msg)
        validate_materialized_circuit_payload(value.serialized_bytes)
        if value.payload_checksum != expected_checksum:
            msg = "Materialized circuit bytes do not match the planned circuit checksum."
            raise ValueError(msg)
        return value

    @staticmethod
    def _checked_materialization_artifact(value: object) -> MaterializedCircuitArtifact:
        """Require the store's typed materialization artifact."""
        if not isinstance(value, MaterializedCircuitArtifact):
            msg = "publish_materialized_circuit must return a MaterializedCircuitArtifact."
            raise TypeError(msg)
        return value

    def _write_materialization_failures(
        self,
        configs: Sequence[PipelineEvaluationConfig],
        error: BaseException,
        *,
        wall_time_seconds: float,
        phase: Literal["materialization", "serialization"],
    ) -> tuple[PipelineBenchmarkFailure, ...]:
        """Persist one linked failure per planned row in input order."""
        self.store.record_materialization_failure(
            config=configs[0],
            exception=error,
            phase=phase,
            wall_time_seconds=wall_time_seconds,
        )
        return tuple(
            self.store.write_evaluation_failure(
                config=config,
                exception=error,
                phase=phase,
                wall_time_seconds=0.0,
                materialization=None,
            )
            for config in configs
        )

    @staticmethod
    def _merge_records(
        ordered: Sequence[PipelineEvaluationConfig],
        existing: Mapping[str, PipelineBenchmarkRecord],
        new_records: Sequence[PipelineBenchmarkRecord],
    ) -> tuple[PipelineBenchmarkRecord, ...]:
        """Merge skipped successes and newly published rows in request order."""
        by_id = dict(existing)
        by_id.update({record.evaluation_row_id: record for record in new_records})
        return tuple(by_id[config.evaluation_row_id] for config in ordered)

    def evaluate(
        self,
        configs: Sequence[PipelineEvaluationConfig],
        materialize: MaterializeCallback,
        evaluate: EvaluateCallback,
        *,
        max_workers: int,
    ) -> tuple[PipelineBenchmarkRecord, ...]:
        """Materialize once, evaluate concurrently, and publish in input order.

        Args:
            configs: Ordered final-evaluation fan-out for one circuit.
            materialize: Callback receiving the complete pipeline and a
                detached selected-parameter vector.
            evaluate: Callback receiving only one final-test config and the
                caller-owned materialized runtime circuit.
            max_workers: Positive bounded worker count.

        Returns:
            Successful and structured-failure rows in ``configs`` order.

        Raises:
            RuntimeError: If training is not complete.
            TypeError: If callbacks or arguments have unsupported types.
            ValueError: If rows do not form one valid fan-out.
        """
        if not callable(materialize) or not callable(evaluate):
            msg = "materialize and evaluate must be callable."
            raise TypeError(msg)
        workers = _require_nonnegative_int(max_workers, "max_workers")
        if workers == 0:
            msg = "max_workers must be positive."
            raise ValueError(msg)
        pipeline = self.store.pipeline_result
        if pipeline is None:
            msg = "Final evaluation requires a complete verified training pipeline."
            raise RuntimeError(msg)
        ordered = self._validate_configs(configs, pipeline)
        existing_successes = {
            record.evaluation_row_id: record
            for record in self.store.records
            if isinstance(record, PipelineBenchmarkResult)
            and record.evaluation_row_id in {config.evaluation_row_id for config in ordered}
        }
        pending = tuple(config for config in ordered if config.evaluation_row_id not in existing_successes)
        if not pending:
            return self._merge_records(ordered, existing_successes, ())
        self.store.require_fresh_handle()
        selected_parameters = self.store.load_final_parameters()

        materialization_started = time.perf_counter()
        try:
            payload = self._checked_materialization(
                materialize(pipeline, selected_parameters),
                pending[0].materialized_circuit_checksum,
            )
        except Exception as error:  # noqa: BLE001 - materialization failures become linked rows
            failures = self._write_materialization_failures(
                pending, error, wall_time_seconds=time.perf_counter() - materialization_started, phase="materialization"
            )
            return self._merge_records(ordered, existing_successes, failures)

        decoding_started = time.perf_counter()
        try:
            runtime_circuit = self.deserialize_circuit(payload.serialized_bytes)
        except Exception as error:  # noqa: BLE001 - decode failures become linked rows
            failures = self._write_materialization_failures(
                pending,
                error,
                wall_time_seconds=payload.wall_time_seconds + time.perf_counter() - decoding_started,
                phase="serialization",
            )
            return self._merge_records(ordered, existing_successes, failures)
        materialization_wall_time = payload.wall_time_seconds + time.perf_counter() - decoding_started

        try:
            materialization = self._checked_materialization_artifact(
                self.store.publish_materialized_circuit(
                    config=pending[0],
                    payload=payload.serialized_bytes,
                    wall_time_seconds=materialization_wall_time,
                    peak_memory_bytes=payload.peak_memory_bytes,
                )
            )
        except (Phase2ConcurrentMutationError, Phase2DerivedArtifactError):
            raise
        except Exception as error:  # noqa: BLE001 - serialization failures become linked rows
            failures = self._write_materialization_failures(
                pending, error, wall_time_seconds=materialization_wall_time, phase="serialization"
            )
            return self._merge_records(ordered, existing_successes, failures)

        records: list[PipelineBenchmarkRecord] = []
        self.store.require_fresh_handle()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="phase2-evaluation") as executor:
            futures = [executor.submit(self._evaluate_one, evaluate, config, runtime_circuit) for config in pending]
            for config, future in zip(pending, futures, strict=True):
                outcome = future.result()
                if outcome.exception is not None:
                    records.append(
                        self.store.write_evaluation_failure(
                            config=config,
                            exception=outcome.exception,
                            phase="evaluation",
                            wall_time_seconds=outcome.observed_wall_time_seconds,
                            materialization=materialization,
                        )
                    )
                    continue
                measurement = cast("PipelineEvaluationMeasurement", outcome.measurement)
                try:
                    record = self.store.write_evaluation_success(
                        config=config,
                        materialization=materialization,
                        test_noiseless_fidelity=measurement.noiseless_fidelity,
                        trajectory_fidelities=measurement.trajectory_fidelities,
                        sampled_nonidentity_events=measurement.sampled_nonidentity_events,
                        normalized_work=measurement.normalized_work,
                        evaluation_wall_time_seconds=measurement.wall_time_seconds,
                        peak_memory_bytes=measurement.peak_memory_bytes,
                        evaluation_provider_checksum=measurement.provider_checksum,
                        evaluation_ensembles=measurement.fixed_map_ensembles,
                    )
                except (Phase2ConcurrentMutationError, Phase2DerivedArtifactError):
                    raise
                except Exception as error:  # noqa: BLE001 - serialization failures become linked rows
                    record = self.store.write_evaluation_failure(
                        config=config,
                        exception=error,
                        phase="serialization",
                        wall_time_seconds=measurement.wall_time_seconds,
                        materialization=materialization,
                    )
                records.append(record)
        return self._merge_records(ordered, existing_successes, records)


__all__ = [
    "DeserializeCircuitCallback",
    "EvaluateCallback",
    "MaterializeCallback",
    "MaterializedCircuitPayload",
    "ParallelPhase2Evaluator",
    "PipelineEvaluationMeasurement",
]
