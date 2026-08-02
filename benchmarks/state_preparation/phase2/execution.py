# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sequential, interruption-safe execution of Phase II training pipelines."""

# Callback errors are normalized into canonical stage failures, so private
# helpers intentionally share the public executor's documented error contract.
# ruff: noqa: DOC201, DOC501, DOC502

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .artifacts import (
    Phase2ArtifactStore,
    Phase2ConcurrentMutationError,
    StageExecutionEvidence,
    StageFailureArtifact,
)
from .noisy_krotov import NoisyKrotovStageExecution, NoisyKrotovStageFailure

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from .pipeline import TrainingPipelineResult, TrainingStageConfig


StageOutcome = StageExecutionEvidence | NoisyKrotovStageExecution | NoisyKrotovStageFailure


class _MeasuredStageError(BaseException):
    """Internal carrier preserving callback resource observations."""

    def __init__(self, original: BaseException, wall_time_seconds: float, peak_memory_bytes: int) -> None:
        """Store the original callback failure and measurements."""
        super().__init__(str(original))
        self.original = original
        self.wall_time_seconds = wall_time_seconds
        self.peak_memory_bytes = peak_memory_bytes


class StageRunner(Protocol):
    """Optimizer/materializer callback used by :class:`Phase2PipelineExecutor`."""

    def __call__(
        self,
        stage: TrainingStageConfig,
        predecessor_parameters: NDArray | None,
    ) -> StageOutcome:
        """Execute exactly one stage without performing filesystem mutation."""


@dataclass(frozen=True, slots=True)
class PipelineExecutionFailure:
    """Stopped pipeline outcome retaining the canonical structured failure."""

    failure: StageFailureArtifact
    completed_stage_count: int

    def __post_init__(self) -> None:
        """Validate the failure and completed-prefix length."""
        if not isinstance(self.failure, StageFailureArtifact):
            msg = "failure must be a StageFailureArtifact."
            raise TypeError(msg)
        if type(self.completed_stage_count) is not int or self.completed_stage_count < 0:
            msg = "completed_stage_count must be a nonnegative int."
            raise ValueError(msg)


class Phase2PipelineExecutor:
    """Run only unfinished stages and commit each one as an immutable prefix."""

    def __init__(self, store: Phase2ArtifactStore) -> None:
        """Bind the executor to one already verified artifact store."""
        if not isinstance(store, Phase2ArtifactStore):
            msg = "store must be a Phase2ArtifactStore."
            raise TypeError(msg)
        self.store = store

    @staticmethod
    def _default_circuit_statistics(stage: TrainingStageConfig) -> Mapping[str, object]:
        """Return minimum topology statistics when a runner supplies none."""
        return {
            "topology_id": stage.output_topology_id,
            "parameter_count": stage.output_parameter_count,
            "stage_kind": stage.stage_kind,
        }

    @staticmethod
    def _adapt_stage_outcome(
        outcome: StageOutcome,
        stage: TrainingStageConfig,
        predecessor: NDArray | None,
        statistics_provider: Callable[[TrainingStageConfig], Mapping[str, object]],
    ) -> StageExecutionEvidence | NoisyKrotovStageFailure:
        """Convert one supported runner result into persistence evidence."""
        if isinstance(outcome, NoisyKrotovStageExecution):
            return StageExecutionEvidence.from_noisy_krotov(
                stage,
                outcome,
                source_parameters=predecessor,
                circuit_statistics=statistics_provider(stage),
            )
        if isinstance(outcome, (StageExecutionEvidence, NoisyKrotovStageFailure)):
            return outcome
        msg = "Stage runner must return StageExecutionEvidence, NoisyKrotovStageExecution, or NoisyKrotovStageFailure."
        raise TypeError(msg)

    @staticmethod
    def _measure_stage_call(
        runner: StageRunner,
        stage: TrainingStageConfig,
        predecessor: NDArray | None,
        statistics_provider: Callable[[TrainingStageConfig], Mapping[str, object]],
    ) -> tuple[StageExecutionEvidence | NoisyKrotovStageFailure, float, int]:
        """Execute and adapt one callback while measuring scientific stage work."""
        owns_tracing = not tracemalloc.is_tracing()
        if owns_tracing:
            tracemalloc.start()
        before_current, _before_peak = tracemalloc.get_traced_memory()
        if owns_tracing:
            tracemalloc.reset_peak()
            before_current = 0
        start = time.perf_counter()
        try:
            outcome = runner(stage, predecessor)
            resolved = Phase2PipelineExecutor._adapt_stage_outcome(
                outcome,
                stage,
                predecessor,
                statistics_provider,
            )
        except BaseException as error:
            elapsed = time.perf_counter() - start
            current, peak = tracemalloc.get_traced_memory()
            measured_peak = max(0, peak - before_current, current - before_current)
            if owns_tracing:
                tracemalloc.stop()
            raise _MeasuredStageError(error, elapsed, measured_peak) from error
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        measured_peak = max(0, peak - before_current, current - before_current)
        if owns_tracing:
            tracemalloc.stop()
        return resolved, elapsed, measured_peak

    def execute(
        self,
        runner: StageRunner,
        *,
        circuit_statistics: Callable[[TrainingStageConfig], Mapping[str, object]] | None = None,
    ) -> TrainingPipelineResult | PipelineExecutionFailure:
        """Execute the unfinished suffix without replaying a completed stage.

        Args:
            runner: Stage-local callback. Its signature deliberately contains
                no final-test configuration or trajectory evidence.
            circuit_statistics: Optional deterministic topology/statistics
                provider used when adapting a WP17 execution.

        Returns:
            A complete pipeline result or a structured stopped outcome.

        Raises:
            KeyboardInterrupt: After first preserving an interruption failure.
            SystemExit: After first preserving an interruption failure.
        """
        if not callable(runner):
            msg = "runner must be callable."
            raise TypeError(msg)
        if circuit_statistics is not None and not callable(circuit_statistics):
            msg = "circuit_statistics must be callable or None."
            raise TypeError(msg)
        statistics_provider = circuit_statistics if circuit_statistics is not None else self._default_circuit_statistics
        result = self.store.pipeline_result
        if result is not None:
            return result
        if self.store.completed_stage_count:
            predecessor = self.store.load_stage_checkpoint(self.store.completed_stage_count - 1).selected_theta
        elif self.store.pipeline.stages[0].input_checkpoint_checksum is not None:
            predecessor = self.store.load_external_checkpoint().selected_theta
        else:
            predecessor = None
        for stage in self.store.pipeline.stages[self.store.completed_stage_count :]:
            self.store.require_fresh_handle()
            try:
                outcome, wall_time, peak_memory = self._measure_stage_call(
                    runner,
                    stage,
                    predecessor,
                    statistics_provider,
                )
            except _MeasuredStageError as measured:
                error = measured.original
                failure = self.store.write_stage_failure(
                    stage,
                    error,
                    wall_time_seconds=measured.wall_time_seconds,
                    retryable=False,
                )
                if not isinstance(error, Exception):
                    raise error from measured
                return PipelineExecutionFailure(
                    failure=failure,
                    completed_stage_count=self.store.completed_stage_count,
                )
            if isinstance(outcome, NoisyKrotovStageFailure):
                failure = self.store.write_stage_failure(
                    stage,
                    outcome,
                    wall_time_seconds=wall_time,
                )
                return PipelineExecutionFailure(
                    failure=failure,
                    completed_stage_count=self.store.completed_stage_count,
                )
            evidence = outcome
            try:
                self.store.publish_stage(
                    evidence,
                    wall_time_seconds=wall_time,
                    peak_memory_bytes=peak_memory,
                )
            except BaseException as error:
                if isinstance(error, Phase2ConcurrentMutationError) or self.store.is_stage_completed(stage.stage_index):
                    raise
                failure = self.store.write_stage_failure(
                    stage,
                    error,
                    wall_time_seconds=wall_time,
                    retryable=False,
                )
                if not isinstance(error, Exception):
                    raise
                return PipelineExecutionFailure(
                    failure=failure,
                    completed_stage_count=self.store.completed_stage_count,
                )
            predecessor = evidence.selected_parameters
        completed = self.store.pipeline_result
        if completed is None:
            msg = "Pipeline executor ended without a complete result."
            raise RuntimeError(msg)
        return completed


__all__ = [
    "Phase2PipelineExecutor",
    "PipelineExecutionFailure",
    "StageOutcome",
    "StageRunner",
]
