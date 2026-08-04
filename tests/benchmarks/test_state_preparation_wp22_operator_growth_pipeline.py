# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for durable standalone WP22 operator-growth orchestration."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation.noise import FIXED_RATE_NOISE_DEFINITION_VERSION
from benchmarks.state_preparation.phase2.canonical import canonical_checksum, canonical_json, seal_mapping
from benchmarks.state_preparation.phase2.operator_growth import (
    OperatorGrowthResult,
    adapt_style_state_preparation,
    run_standard_fixed_rate_noisy_operator_growth,
)
from benchmarks.state_preparation.phase2.operator_growth_pipeline import (
    OPERATOR_GROWTH_ARTIFACT_NAME,
    OperatorGrowthPipelineArtifact,
    OperatorGrowthPipelineRequest,
    OuterScreeningEvaluation,
    execute_operator_growth_pipeline,
)
from benchmarks.state_preparation.phase2.protocol import ScreeningCell, load_initial_preregistration
from benchmarks.state_preparation.phase2.screening import (
    OperatorGrowthScreeningTemplate,
    ScreeningSourceRecord,
    WP22CandidateConfiguration,
)
from benchmarks.state_preparation.phase2.targets import (
    MaterializedTarget,
    authorize_target_materialization,
    build_target_population_config,
    materialize_target_population,
    role_master_entropy_commitment,
)
from tests.benchmarks.test_state_preparation_phase2_pipeline import _screening_target_manifest

if TYPE_CHECKING:
    from pathlib import Path


_OPTIMIZATION_BLOCK_ID = "wp22-op-growth-block"
_OPTIMIZATION_SEED = 17
_SCREENING_SEED = 29
_RESOURCE_STRATUM_ID = "native-rzz-12"
_SCHEDULE_CHECKSUM = canonical_checksum({"schedule": "wp22-test"})
_EVALUATION_POLICY_CHECKSUM = canonical_checksum({"evaluation": "outer-primary-noise"})
_SOURCE_CHECKSUM = canonical_checksum({"source": "execution"})
_TRACKED_SOURCE_CHECKSUM = canonical_checksum({"source": "tracked"})
_SOURCE_COMMIT = "1" * 40


def _checksum(label: str) -> str:
    """Return a deterministic valid checksum for one test identity."""
    return canonical_checksum({"label": label})


@pytest.fixture(scope="module")
def operator_growth_case() -> tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest]:
    """Return one authorized q6 target, exact noisy result, and bound request.

    Returns:
        The target, precomputed promotion-eligible result, and exact request.
    """
    master_entropy = bytes(reversed(range(32)))
    preregistration = load_initial_preregistration()
    population = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(master_entropy),
        population_scope="primary_q6",
    )
    manifest = _screening_target_manifest()
    authorization = authorize_target_materialization(
        preregistration,
        population,
        manifest,
        master_entropy,
    )
    materialization = materialize_target_population(
        population,
        preregistration,
        manifest,
        master_entropy,
        authorization,
    )
    spec = manifest.instances[0]
    target = materialization.target(spec.target_instance_id)
    result = run_standard_fixed_rate_noisy_operator_growth(
        target,
        optimization_block_id=_OPTIMIZATION_BLOCK_ID,
        optimization_seed=_OPTIMIZATION_SEED,
        resource_stratum_id=_RESOURCE_STRATUM_ID,
        noise_id="dephasing_1s_1q",
        noise_definition_version=FIXED_RATE_NOISE_DEFINITION_VERSION,
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        trajectory_count=1,
        trajectory_seed=19,
        gradient_tolerance=1e6,
        max_operators=1,
        native_two_qubit_cap_per_edge=12,
        reoptimization_steps=0,
    )
    assert result.pool is not None
    assert result.growth_spec is not None
    template = OperatorGrowthScreeningTemplate(
        pool_policy_id="nearest_neighbor_pool",
        growth_policy_id="largest_projector_gradient",
        max_operators=1,
        reoptimization_steps=0,
        gradient_threshold=1e6,
        training_trajectory_count=1,
        native_two_qubit_cap_per_edge=12.0,
    )
    candidate = WP22CandidateConfiguration.from_operator_growth(
        template,
        strategy_schedule_checksum=_SCHEDULE_CHECKSUM,
        resource_stratum_id=_RESOURCE_STRATUM_ID,
    )
    cell = ScreeningCell(
        cell_id="operator-growth-cell",
        family_id=target.family_id,
        stratum_id=target.stratum_id,
        qubit_count=target.qubit_count,
        target_instance_id=target.target_instance_id,
        optimization_seed=_OPTIMIZATION_SEED,
        screening_seed=_SCREENING_SEED,
    )
    request = OperatorGrowthPipelineRequest(
        request_id="operator-growth-request",
        template=template,
        candidate=candidate,
        cell=cell,
        target_manifest_checksum=target.target_manifest_checksum,
        target_spec_checksum=target.target_instance_spec_checksum,
        target_vector_checksum=target.vector_checksum,
        strategy_schedule_checksum=_SCHEDULE_CHECKSUM,
        screening_evaluation_policy_checksum=_EVALUATION_POLICY_CHECKSUM,
        execution_source_manifest_checksum=_SOURCE_CHECKSUM,
        tracked_source_manifest_checksum=_TRACKED_SOURCE_CHECKSUM,
        source_commit=_SOURCE_COMMIT,
        optimization_block_id=_OPTIMIZATION_BLOCK_ID,
        outer_evaluation_trajectory_count=5,
    )
    return target, result, request


def _outer_evaluation(
    request: OperatorGrowthPipelineRequest,
    result: OperatorGrowthResult,
) -> OuterScreeningEvaluation:
    """Return fresh aggregate outer-screening evidence.

    Returns:
        A role- and seed-bound evaluation record.
    """
    scratch = bytearray(20_000)
    scratch[0] = 1
    return OuterScreeningEvaluation.create(
        request,
        result,
        trajectory_fidelities=(0.73,) * request.outer_evaluation_trajectory_count,
    )


def test_request_is_strict_sealed_and_binds_candidate_schedule_and_source(
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """Request roundtrip and nested tamper detection retain every exact identity."""
    _, _, request = operator_growth_case
    assert OperatorGrowthPipelineRequest.from_json(request.to_json()) == request

    tampered = request.to_dict()
    nested = cast("dict[str, object]", tampered["template"])
    nested["max_operators"] = 2
    tampered.pop("content_checksum")
    with pytest.raises(ValueError, match="checksum mismatch"):
        OperatorGrowthPipelineRequest.from_dict(seal_mapping(tampered))

    with pytest.raises(ValueError, match="strategy_schedule_checksum"):
        replace(request, strategy_schedule_checksum=_checksum("other-schedule"))

    changed_source = replace(request, execution_source_manifest_checksum=_checksum("other-source"))
    changed_policy = replace(request, screening_evaluation_policy_checksum=_checksum("other-policy"))
    assert changed_source.content_checksum != request.content_checksum
    assert changed_policy.content_checksum != request.content_checksum

    unregistered_template = replace(request.template, pool_policy_id="unregistered_pool")
    unregistered_candidate = WP22CandidateConfiguration.from_operator_growth(
        unregistered_template,
        strategy_schedule_checksum=_SCHEDULE_CHECKSUM,
        resource_stratum_id=_RESOURCE_STRATUM_ID,
    )
    with pytest.raises(ValueError, match="registered projector-pool"):
        replace(request, template=unregistered_template, candidate=unregistered_candidate)


def test_success_attaches_authoritative_runtime_memory_work_and_promotion_row(
    tmp_path: Path,
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """Successful execution measures both callbacks and emits the shared verified row."""
    _, result, request = operator_growth_case
    calls: list[str] = []

    def growth(actual: OperatorGrowthPipelineRequest) -> OperatorGrowthResult:
        calls.append(actual.content_checksum)
        scratch = bytearray(80_000)
        scratch[0] = 1
        return result

    artifact = execute_operator_growth_pipeline(request, tmp_path, growth, _outer_evaluation)

    assert artifact is not None
    assert artifact.status == "success"
    assert calls == [request.content_checksum]
    assert artifact.training_work.wall_time_seconds > 0.0
    assert artifact.training_work.peak_memory_bytes > 0
    assert artifact.evaluation_work.wall_time_seconds > 0.0
    assert artifact.evaluation_work.peak_memory_bytes > 0
    assert artifact.training_work.objective_calls == result.wp20_work.objective_calls
    assert artifact.training_work.training_trajectories == result.wp20_work.training_trajectories
    assert artifact.evaluation_work.test_trajectories == request.outer_evaluation_trajectory_count
    assert artifact.total_work == artifact.verified_outcome.work_ledger
    assert artifact.verified_outcome.status == "success"
    assert artifact.verified_outcome.noisy_fidelity == pytest.approx(0.73)
    assert artifact.circuit_resources == result.circuit_resources
    source = ScreeningSourceRecord.from_operator_growth_artifact(artifact)
    assert source.verified_outcome() == artifact.verified_outcome
    assert ScreeningSourceRecord.from_json(source.to_json()) == source
    assert OperatorGrowthPipelineArtifact.from_json(artifact.to_json()) == artifact
    persisted = OperatorGrowthPipelineArtifact.from_json(
        (tmp_path / OPERATOR_GROWTH_ARTIFACT_NAME).read_text(encoding="utf-8").removesuffix("\n")
    )
    assert persisted == artifact


def test_evaluation_callbacks_cannot_claim_runtime_or_final_test_role(
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """Only wrapper measurements and the outer screening-selection role are accepted."""
    _, result, request = operator_growth_case
    valid = _outer_evaluation(request, result)
    claimed = replace(valid.work_ledger, wall_time_seconds=10.0, peak_memory_bytes=9)
    with pytest.raises(ValueError, match="cannot claim runtime"):
        replace(valid, work_ledger=claimed)

    document = valid.to_dict()
    document.pop("content_checksum")
    document["data_role"] = "confirmatory_test"
    with pytest.raises(ValueError, match="never final-test"):
        OuterScreeningEvaluation.from_dict(seal_mapping(document))

    with pytest.raises(ValueError, match="screening role and seed"):
        OuterScreeningEvaluation.create(
            request,
            result,
            trajectory_fidelities=(0.73,) * request.outer_evaluation_trajectory_count,
            trajectory_ensemble_checksum=_checksum("caller-selected-ensemble"),
        )


def test_wrong_outer_seed_is_a_durable_screening_failure(
    tmp_path: Path,
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """A well-formed evaluation from another seed cannot enter promotion evidence."""
    _, result, request = operator_growth_case

    def wrong_seed(
        actual_request: OperatorGrowthPipelineRequest,
        actual_result: OperatorGrowthResult,
    ) -> OuterScreeningEvaluation:
        return replace(_outer_evaluation(actual_request, actual_result), evaluation_seed=_SCREENING_SEED + 1)

    artifact = execute_operator_growth_pipeline(request, tmp_path, lambda _: result, wrong_seed)

    assert artifact is not None
    assert artifact.status == "failure"
    assert artifact.failure_phase == "screening_evaluation"
    assert artifact.verified_outcome.status == "failure"
    assert "seed" in cast("str", artifact.message)


def test_nonnoisy_or_wrong_callback_result_is_a_durable_training_failure(
    tmp_path: Path,
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """Analytic reference output can never masquerade as promotion-eligible noisy growth."""
    _, _, request = operator_growth_case
    analytic = adapt_style_state_preparation([1.0, 0.0], max_operators=1, reoptimization_steps=0)

    artifact = execute_operator_growth_pipeline(request, tmp_path, lambda _: analytic, _outer_evaluation)

    assert artifact is not None
    assert artifact.status == "failure"
    assert artifact.failure_phase == "operator_growth_execution"
    assert artifact.operator_growth_result == analytic
    assert artifact.training_work.normalized_compute() == pytest.approx(
        analytic.wp20_work.normalized_compute(),
        rel=0.0,
        abs=0.0,
    )
    assert artifact.verified_outcome.normalized_work == pytest.approx(
        artifact.total_work.normalized_compute(),
        rel=0.0,
        abs=0.0,
    )
    assert OperatorGrowthPipelineArtifact.from_json(artifact.to_json()) == artifact


def test_first_terminal_failure_is_immutable_and_resume_only_reopens(
    tmp_path: Path,
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """Resume reopens the first failure and overwrite can never replace it."""
    _, result, request = operator_growth_case
    calls = 0

    def fails_then_succeeds(_: OperatorGrowthPipelineRequest) -> OperatorGrowthResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            msg = "synthetic growth failure"
            raise RuntimeError(msg)
        return result

    failed = execute_operator_growth_pipeline(request, tmp_path, fails_then_succeeds, _outer_evaluation)
    assert failed is not None
    assert failed.status == "failure"
    assert failed.attempt == 1

    resumed = execute_operator_growth_pipeline(
        request,
        tmp_path,
        fails_then_succeeds,
        _outer_evaluation,
        resume=True,
    )
    assert resumed == failed
    assert calls == 1

    with pytest.raises(ValueError, match="first-terminal artifact already exists"):
        execute_operator_growth_pipeline(request, tmp_path, lambda _: result, _outer_evaluation)

    with pytest.raises(ValueError, match="immutable; overwrite is not supported"):
        execute_operator_growth_pipeline(
            request,
            tmp_path,
            lambda _: result,
            _outer_evaluation,
            overwrite=True,
        )
    assert calls == 1


def test_dry_run_does_not_mutate_or_call_callbacks(
    tmp_path: Path,
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """Dry-run validation returns before directory, lock, artifact, or callback creation."""
    _, result, request = operator_growth_case
    output = tmp_path / "absent"
    calls = 0

    def called(_: OperatorGrowthPipelineRequest) -> OperatorGrowthResult:
        nonlocal calls
        calls += 1
        return result

    artifact = execute_operator_growth_pipeline(request, output, called, _outer_evaluation, dry_run=True)

    assert artifact is None
    assert calls == 0
    assert not output.exists()


def test_interruption_leaves_no_partial_artifact_and_can_recover(
    tmp_path: Path,
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """BaseException interruption publishes nothing and a later exact resume starts cleanly."""
    _, result, request = operator_growth_case

    def interrupted(_: OperatorGrowthPipelineRequest) -> OperatorGrowthResult:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_operator_growth_pipeline(request, tmp_path, interrupted, _outer_evaluation)
    assert not (tmp_path / OPERATOR_GROWTH_ARTIFACT_NAME).exists()

    recovered = execute_operator_growth_pipeline(
        request,
        tmp_path,
        lambda _: result,
        _outer_evaluation,
        resume=True,
    )
    assert recovered is not None
    assert recovered.status == "success"
    assert recovered.attempt == 1


def test_file_lock_serializes_concurrent_resume_to_one_execution(
    tmp_path: Path,
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """Two concurrent resume callers share one atomically committed successful artifact."""
    _, result, request = operator_growth_case
    calls = 0
    calls_lock = threading.Lock()
    artifacts: list[OperatorGrowthPipelineArtifact | None] = []
    errors: list[BaseException] = []

    def growth(_: OperatorGrowthPipelineRequest) -> OperatorGrowthResult:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.03)
        return result

    def worker() -> None:
        try:
            artifacts.append(
                execute_operator_growth_pipeline(
                    request,
                    tmp_path,
                    growth,
                    _outer_evaluation,
                    resume=True,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = (threading.Thread(target=worker), threading.Thread(target=worker))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert calls == 1
    assert len(artifacts) == 2
    assert artifacts[0] == artifacts[1]
    assert artifacts[0] is not None
    assert artifacts[0].status == "success"


def test_existing_artifact_cannot_be_reused_for_a_different_request(
    tmp_path: Path,
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """Resume rejects foreign identity and overwrite is unavailable for every request."""
    _, result, request = operator_growth_case
    artifact = execute_operator_growth_pipeline(request, tmp_path, lambda _: result, _outer_evaluation)
    assert artifact is not None
    changed = replace(request, request_id="different-request")

    with pytest.raises(ValueError, match="different exact request"):
        execute_operator_growth_pipeline(
            changed,
            tmp_path,
            lambda _: result,
            _outer_evaluation,
            resume=True,
        )
    with pytest.raises(ValueError, match="overwrite is not supported"):
        execute_operator_growth_pipeline(
            changed,
            tmp_path,
            lambda _: result,
            _outer_evaluation,
            overwrite=True,
        )


def test_resealed_later_attempt_cannot_enter_first_terminal_custody(
    tmp_path: Path,
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """A canonical attempt-two artifact is rejected both directly and on resume."""
    _, result, request = operator_growth_case
    artifact = execute_operator_growth_pipeline(request, tmp_path, lambda _: result, _outer_evaluation)
    assert artifact is not None
    forged_content = artifact.to_dict()
    forged_content["attempt"] = 2
    forged = seal_mapping({key: value for key, value in forged_content.items() if key != "content_checksum"})

    with pytest.raises(ValueError, match="authoritative first attempt"):
        OperatorGrowthPipelineArtifact.from_dict(forged)

    artifact_path = tmp_path / OPERATOR_GROWTH_ARTIFACT_NAME
    artifact_path.write_text(f"{canonical_json(forged)}\n")
    with pytest.raises(ValueError, match="authoritative first attempt"):
        execute_operator_growth_pipeline(
            request,
            tmp_path,
            lambda _: result,
            _outer_evaluation,
            resume=True,
        )


def test_evaluation_callback_exception_persists_source_linked_failure(
    tmp_path: Path,
    operator_growth_case: tuple[MaterializedTarget, OperatorGrowthResult, OperatorGrowthPipelineRequest],
) -> None:
    """Outer simulation exceptions preserve exact training work and a failure promotion row."""
    _, result, request = operator_growth_case

    def fail_evaluation(
        _: OperatorGrowthPipelineRequest,
        __: OperatorGrowthResult,
    ) -> OuterScreeningEvaluation:
        msg = "outer simulator unavailable"
        raise RuntimeError(msg)

    artifact = execute_operator_growth_pipeline(request, tmp_path, lambda _: result, fail_evaluation)

    assert artifact is not None
    assert artifact.status == "failure"
    assert artifact.failure_phase == "screening_evaluation"
    assert artifact.operator_growth_result == result
    assert artifact.materialization_checksum is not None
    assert artifact.training_work.objective_calls == result.wp20_work.objective_calls
    assert artifact.verified_outcome.failure_code == "screening_evaluation"
