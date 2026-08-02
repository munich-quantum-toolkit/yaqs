# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Structural tests for the explicitly opt-in WP19 reproduction job."""

# Typed evaluator doubles must initialize the frozen slotted configuration.
# ruff: noqa: PLC2801, SLF001

from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import benchmarks.state_preparation.phase2.run_historical_reproduction as runner_module
from benchmarks.state_preparation.phase2.artifacts import PIPELINE_CONFIG_NAME
from benchmarks.state_preparation.phase2.canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
)
from benchmarks.state_preparation.phase2.historical_reproduction import (
    LEGACY_REPRODUCTION_TARGET_SEEDS,
    LayerwiseMaterializedCircuit,
    LegacyReproductionOutcome,
    LegacyReproductionReport,
    compare_legacy_reproduction,
    load_archived_layerwise_reference,
)
from benchmarks.state_preparation.phase2.layerwise_bmpd import resolve_layerwise_bmpd_crn_legacy_v1_pipeline
from benchmarks.state_preparation.phase2.noisy_krotov import NoisyKrotovCircuitBinding
from benchmarks.state_preparation.phase2.pipeline import (
    PipelineBenchmarkResult,
    PipelineEvaluationConfig,
    TrainingPipelineResult,
)
from benchmarks.state_preparation.phase2.run_historical_reproduction import (
    HISTORICAL_REPRODUCTION_DISCREPANCY_EXIT_CODE,
    HISTORICAL_REPRODUCTION_EVALUATION_WORKERS,
    HISTORICAL_REPRODUCTION_FAILURE_EXIT_CODE,
    HISTORICAL_REPRODUCTION_LOCK_NAME,
    HISTORICAL_REPRODUCTION_REPORT_NAME,
    HISTORICAL_REPRODUCTION_RUNTIME_NAME,
    HISTORICAL_REPRODUCTION_SOURCE_MANIFEST_NAME,
    HISTORICAL_REPRODUCTION_SUCCESS_EXIT_CODE,
    HISTORICAL_REPRODUCTION_THREAD_LIMIT,
    HISTORICAL_REPRODUCTION_TOLERANCE,
    HistoricalReproductionConcurrentExecutionError,
    HistoricalReproductionSourceManifest,
    run_historical_reproduction_job,
    verify_historical_reproduction_artifacts,
)
from mqt.yaqs.optimization import (
    KrotovFixedMapEnsemble,
    KrotovNoiseMap,
    ParameterizedCircuit,
    ParameterizedGate,
)

if TYPE_CHECKING:
    from benchmarks.state_preparation.phase2.legacy_targets import LegacyMaterializedTarget
    from benchmarks.state_preparation.phase2.run_historical_reproduction import (
        HistoricalTargetRequest,
    )
    from mqt.yaqs.optimization import (
        KrotovMapRole,
        KrotovTJMOptions,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CHECKSUM = "sha256:" + "a" * 64


def _success(request: HistoricalTargetRequest, fidelity: float | None = None) -> LegacyReproductionOutcome:
    """Return a synthetic outcome derived only from its requested seed."""
    value = request.target_seed / 1000 if fidelity is None else fidelity
    return LegacyReproductionOutcome(
        target_seed=request.target_seed,
        status="success",
        computed_fidelity=value,
        source_record_id=f"phase2_evaluation_seed_{request.target_seed}",
        source_record_checksum=canonical_checksum({"seed": request.target_seed, "fidelity": value}),
        runtime_fingerprint_checksum=request.resumability_fingerprint.content_checksum,
    )


def _run_git(repository: Path, *arguments: str) -> str:
    """Run Git in an isolated test repository.

    Returns:
        Stripped command stdout.
    """
    executable = shutil.which("git")
    assert executable is not None
    completed = subprocess.run(  # noqa: S603 -- resolved executable; no shell interpretation
        (executable, "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def source_repository(tmp_path: Path) -> Path:
    """Create the minimal tracked source universe required by the WP19 job.

    Returns:
        The isolated committed repository root.
    """
    repository = tmp_path / "source_repository"
    source = repository / "src" / "mqt" / "yaqs" / "method.py"
    benchmark = repository / "benchmarks" / "state_preparation" / "method.py"
    data = repository / "benchmarks" / "state_preparation" / "phase2" / "data"
    source.parent.mkdir(parents=True)
    benchmark.parent.mkdir(parents=True)
    data.mkdir(parents=True)
    source.write_text("SOURCE_VERSION = 1\n", encoding="utf-8")
    benchmark.write_text("BENCHMARK_VERSION = 1\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname = 'wp19-test'\nversion = '0'\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    for name in (
        "initial_preregistration_v1.json",
        "legacy_evidence_audit_v1.json",
        "legacy_tfim_targets_v1.json",
    ):
        (data / name).write_text('{"test":true}\n', encoding="utf-8")
    _run_git(repository, "init", "--quiet")
    _run_git(repository, "config", "user.name", "WP19 Test")
    _run_git(repository, "config", "user.email", "wp19@example.invalid")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "--quiet", "-m", "initial")
    return repository


def _reseal(document: dict[str, object]) -> None:
    """Recompute one mutable test document's outer checksum."""
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })


def test_job_is_inert_without_the_explicit_expensive_opt_in(tmp_path: Path) -> None:
    """Importing or accidentally calling the job cannot launch q8 work."""
    calls: list[int] = []

    def would_execute(request: HistoricalTargetRequest) -> LegacyReproductionOutcome:
        calls.append(request.target_seed)
        return _success(request)

    with pytest.raises(ValueError, match="explicit execute_expensive=True"):
        run_historical_reproduction_job(
            tmp_path / "not_started",
            execute_expensive=False,
            repository_root=REPOSITORY_ROOT,
            target_executor=would_execute,
        )

    assert calls == []
    assert not (tmp_path / "not_started").exists()


def test_job_runs_exactly_five_targets_serially_and_writes_sealed_outputs(
    tmp_path: Path,
    source_repository: Path,
) -> None:
    """The injected CI path observes the production seed order and output layout."""
    requests: list[HistoricalTargetRequest] = []

    def execute(request: HistoricalTargetRequest) -> LegacyReproductionOutcome:
        requests.append(request)
        return _success(request)

    output_root = tmp_path / "wp19"
    report = run_historical_reproduction_job(
        output_root,
        execute_expensive=True,
        repository_root=source_repository,
        target_executor=execute,
    )

    assert tuple(request.target_seed for request in requests) == LEGACY_REPRODUCTION_TARGET_SEEDS
    assert tuple(request.output_directory.name for request in requests) == tuple(
        f"seed_{seed}" for seed in LEGACY_REPRODUCTION_TARGET_SEEDS
    )
    assert all(not request.resume and not request.overwrite for request in requests)
    assert report.classification == "discrepant"
    assert report.computed_mean == pytest.approx(0.3)
    assert report.comparison_tolerance == HISTORICAL_REPRODUCTION_TOLERANCE

    report_payload = (output_root / HISTORICAL_REPRODUCTION_REPORT_NAME).read_text(encoding="utf-8")
    assert report_payload.endswith("\n")
    assert LegacyReproductionReport.from_json(report_payload) == report
    manifest = HistoricalReproductionSourceManifest.from_json(
        (output_root / HISTORICAL_REPRODUCTION_SOURCE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    runtime = load_canonical_json_object(
        (output_root / HISTORICAL_REPRODUCTION_RUNTIME_NAME).read_text(encoding="utf-8")
    )
    assert runtime["target_seeds"] == LEGACY_REPRODUCTION_TARGET_SEEDS
    assert runtime["target_execution"] == "serial"
    assert runtime["evaluation_workers"] == HISTORICAL_REPRODUCTION_EVALUATION_WORKERS == 1
    assert runtime["numerical_thread_limit"] == HISTORICAL_REPRODUCTION_THREAD_LIMIT == 1
    assert runtime["evaluation_trajectory_count"] == 500
    assert runtime["evaluation_seed"] == 0
    assert runtime["source_manifest_checksum"] == manifest.content_checksum == report.source_manifest_checksum
    assert runtime["content_checksum"] == report.runtime_checksum
    assert verify_historical_reproduction_artifacts(output_root) == report
    assert all(request.source_manifest == manifest for request in requests)
    assert len({request.resumability_fingerprint.content_checksum for request in requests}) == 5
    for request, comparison in zip(requests, report.target_comparisons, strict=True):
        assert (
            request.source_manifest.fingerprint_for_pipeline(request.resumability_fingerprint.pipeline_prefix_id)
            == request.resumability_fingerprint
        )
        assert comparison.outcome.runtime_fingerprint_checksum == request.resumability_fingerprint.content_checksum


def test_source_mutation_between_targets_aborts_without_a_mixed_report(
    tmp_path: Path,
    source_repository: Path,
) -> None:
    """A changed tracked byte is job-fatal before the next target can enter."""
    calls: list[int] = []
    tracked_source = source_repository / "src" / "mqt" / "yaqs" / "method.py"

    def mutate_after_first(request: HistoricalTargetRequest) -> LegacyReproductionOutcome:
        calls.append(request.target_seed)
        outcome = _success(request)
        if request.target_seed == 100:
            tracked_source.write_text("SOURCE_VERSION = 2\n", encoding="utf-8")
        return outcome

    output_root = tmp_path / "mutated_source"
    with pytest.raises(ValueError, match="launch snapshot changed"):
        run_historical_reproduction_job(
            output_root,
            execute_expensive=True,
            repository_root=source_repository,
            target_executor=mutate_after_first,
        )

    assert calls == [100]
    assert not (output_root / HISTORICAL_REPRODUCTION_REPORT_NAME).exists()


def test_resealed_runtime_or_report_cannot_break_the_persisted_binding_chain(
    tmp_path: Path,
    source_repository: Path,
) -> None:
    """Cross-artifact checks reject individually valid but incorrectly linked roots."""
    output_root = tmp_path / "binding_tamper"
    run_historical_reproduction_job(
        output_root,
        execute_expensive=True,
        repository_root=source_repository,
        target_executor=_success,
    )
    runtime_path = output_root / HISTORICAL_REPRODUCTION_RUNTIME_NAME
    report_path = output_root / HISTORICAL_REPRODUCTION_REPORT_NAME
    original_runtime = runtime_path.read_text(encoding="utf-8")
    original_report = report_path.read_text(encoding="utf-8")

    runtime_document = dict(load_canonical_json_object(original_runtime))
    runtime_document["source_manifest_checksum"] = _CHECKSUM
    _reseal(runtime_document)
    runtime_path.write_text(f"{canonical_json(runtime_document)}\n", encoding="utf-8")
    runner_module._verified_runtime_document(runtime_document)
    with pytest.raises(ValueError, match="does not bind the active source manifest"):
        verify_historical_reproduction_artifacts(output_root)

    runtime_path.write_text(original_runtime, encoding="utf-8")
    report_document = dict(load_canonical_json_object(original_report))
    report_document["runtime_checksum"] = _CHECKSUM
    _reseal(report_document)
    report_path.write_text(f"{canonical_json(report_document)}\n", encoding="utf-8")
    LegacyReproductionReport.from_json(report_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="does not bind the active runtime"):
        verify_historical_reproduction_artifacts(output_root)
    with pytest.raises(ValueError, match="retained historical report"):
        run_historical_reproduction_job(
            output_root,
            execute_expensive=True,
            resume=True,
            repository_root=source_repository,
            target_executor=_success,
        )


def test_second_job_cannot_enter_while_the_root_lock_is_owned(
    tmp_path: Path,
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One lock serializes prepare, every target, and final report publication."""
    monkeypatch.setattr(runner_module, "HISTORICAL_REPRODUCTION_LOCK_TIMEOUT_SECONDS", 0.0)
    first_entered = Event()
    release_first = Event()
    second_calls: list[int] = []
    output_root = tmp_path / "contended"

    def blocking_executor(request: HistoricalTargetRequest) -> LegacyReproductionOutcome:
        if request.target_seed == 100:
            first_entered.set()
            assert release_first.wait(timeout=10.0)
        return _success(request)

    def second_executor(request: HistoricalTargetRequest) -> LegacyReproductionOutcome:
        second_calls.append(request.target_seed)
        return _success(request)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            run_historical_reproduction_job,
            output_root,
            execute_expensive=True,
            repository_root=source_repository,
            target_executor=blocking_executor,
        )
        assert first_entered.wait(timeout=10.0)
        try:
            with pytest.raises(HistoricalReproductionConcurrentExecutionError, match="currently owns"):
                run_historical_reproduction_job(
                    output_root,
                    execute_expensive=True,
                    resume=True,
                    repository_root=source_repository,
                    target_executor=second_executor,
                )
        finally:
            release_first.set()
        assert first.result(timeout=20.0).classification == "discrepant"

    assert second_calls == []
    lock_path = output_root / HISTORICAL_REPRODUCTION_LOCK_NAME
    if lock_path.exists():
        assert lock_path.is_file()
        assert not lock_path.is_symlink()


def test_job_preserves_one_failure_and_continues_later_targets(
    tmp_path: Path,
    source_repository: Path,
) -> None:
    """A failed target remains a failure row and never receives a copied reference."""
    calls: list[int] = []

    def execute(request: HistoricalTargetRequest) -> LegacyReproductionOutcome:
        calls.append(request.target_seed)
        if request.target_seed == 300:
            msg = "synthetic optimizer failure"
            raise RuntimeError(msg)
        return _success(request, 0.25)

    report = run_historical_reproduction_job(
        tmp_path / "failure",
        execute_expensive=True,
        repository_root=source_repository,
        target_executor=execute,
    )

    assert tuple(calls) == LEGACY_REPRODUCTION_TARGET_SEEDS
    assert len(report.target_comparisons) == 5
    failed = report.target_comparisons[2].outcome
    assert failed.target_seed == 300
    assert failed.status == "failure"
    assert failed.computed_fidelity is None
    assert failed.failure_type == "RuntimeError"
    assert failed.failure_message == "synthetic optimizer failure"
    assert report.classification == "discrepant"
    assert report.computed_mean is None


def test_new_resume_and_overwrite_modes_are_explicit_and_target_local(
    tmp_path: Path,
    source_repository: Path,
) -> None:
    """Resume detects started targets while overwrite remains a separate request."""
    output_root = tmp_path / "modes"

    def first(request: HistoricalTargetRequest) -> LegacyReproductionOutcome:
        request.output_directory.mkdir(parents=True)
        (request.output_directory / PIPELINE_CONFIG_NAME).write_text("test marker", encoding="utf-8")
        return _success(request)

    run_historical_reproduction_job(
        output_root,
        execute_expensive=True,
        repository_root=source_repository,
        target_executor=first,
    )
    with pytest.raises(ValueError, match="requires resume=True or overwrite=True"):
        run_historical_reproduction_job(
            output_root,
            execute_expensive=True,
            repository_root=source_repository,
            target_executor=_success,
        )

    resumed: list[HistoricalTargetRequest] = []
    run_historical_reproduction_job(
        output_root,
        execute_expensive=True,
        resume=True,
        repository_root=source_repository,
        target_executor=lambda request: (resumed.append(request), _success(request))[1],
    )
    assert len(resumed) == 5
    assert all(request.resume and not request.overwrite for request in resumed)

    overwritten: list[HistoricalTargetRequest] = []
    run_historical_reproduction_job(
        output_root,
        execute_expensive=True,
        overwrite=True,
        repository_root=source_repository,
        target_executor=lambda request: (overwritten.append(request), _success(request))[1],
    )
    assert len(overwritten) == 5
    assert all(request.overwrite and not request.resume for request in overwritten)


def test_job_rejects_a_broad_or_repository_output_root() -> None:
    """Even explicit overwrite cannot target the checkout itself."""
    with pytest.raises(ValueError, match="dedicated output directory"):
        run_historical_reproduction_job(
            REPOSITORY_ROOT,
            execute_expensive=True,
            overwrite=True,
            repository_root=REPOSITORY_ROOT,
            target_executor=_success,
        )


class _TrainingResult(TrainingPipelineResult):
    """Minimal complete-result token for evaluation-binding tests."""

    @property
    def content_checksum(self) -> str:
        """Fixed synthetic pipeline-result checksum."""
        return _CHECKSUM


def _training_result(target_seed: int = 100) -> _TrainingResult:
    """Build a typed pipeline-result double around the real resolved profile.

    Returns:
        The synthetic completed training result.
    """
    result = object.__new__(_TrainingResult)
    object.__setattr__(result, "config", resolve_layerwise_bmpd_crn_legacy_v1_pipeline(target_seed))
    object.__setattr__(result, "final_checkpoint_checksum", _CHECKSUM)
    return result


def _historical_config(pipeline: TrainingPipelineResult) -> PipelineEvaluationConfig:
    """Build the exact historical evaluation row used by the production job.

    Returns:
        The pinned historical evaluation configuration.
    """
    stage = pipeline.config.stages[-1]
    return PipelineEvaluationConfig.for_pipeline(
        pipeline=pipeline,
        materialized_circuit_checksum=_CHECKSUM,
        test_noise_id="ibm_inspired_pauli_legacy_v1",
        noise_definition_version="yaqs.state_preparation.noise.v1",
        noise_strength_scale=1.0,
        tjm_dt=1.0,
        evaluation_seed=0,
        evaluation_seed_domain=cast("str", pipeline.config.seed_domains["pilot_evaluation"]),
        repetition=0,
        trajectory_budget=500,
        evaluation_policy="fixed_sample",
        confidence_level=None,
        confidence_interval_method=None,
        sidecar_storage_policy="trajectory_fidelities",
        max_bond_dimension=stage.max_bond_dimension,
        svd_threshold=stage.svd_threshold,
        truncation_mode=stage.truncation_mode,
        min_bond_dimension=stage.min_bond_dimension,
    )


def _evaluation_record(
    config: PipelineEvaluationConfig,
    runtime_fingerprint_checksum: str = _CHECKSUM,
) -> PipelineBenchmarkResult:
    """Create a typed record token exposing its immutable planned config.

    Returns:
        A result token carrying ``config`` for binding validation.
    """
    record = object.__new__(PipelineBenchmarkResult)
    object.__setattr__(record, "config", config)
    object.__setattr__(record, "runtime_fingerprint_checksum", runtime_fingerprint_checksum)
    return record


def test_projection_accepts_only_the_exact_planned_row_for_the_current_target() -> None:
    """A same-pipeline row with another identity cannot be relabelled as seed 100."""
    pipeline = _training_result()
    planned = _historical_config(pipeline)
    record = _evaluation_record(planned)

    assert (
        runner_module.validate_historical_evaluation_record(
            target_seed=100,
            record=record,
            planned_config=planned,
            pipeline=pipeline,
            expected_runtime_fingerprint_checksum=_CHECKSUM,
        )
        is record
    )

    miswired = replace(planned, repetition=1)
    with pytest.raises(ValueError, match="other than the exact planned"):
        runner_module.validate_historical_evaluation_record(
            target_seed=100,
            record=_evaluation_record(miswired),
            planned_config=planned,
            pipeline=pipeline,
            expected_runtime_fingerprint_checksum=_CHECKSUM,
        )
    with pytest.raises(ValueError, match="requested historical target"):
        runner_module.validate_historical_evaluation_record(
            target_seed=200,
            record=record,
            planned_config=planned,
            pipeline=pipeline,
            expected_runtime_fingerprint_checksum=_CHECKSUM,
        )
    with pytest.raises(ValueError, match="does not bind the verified target WP18 fingerprint"):
        runner_module.validate_historical_evaluation_record(
            target_seed=100,
            record=_evaluation_record(planned, "sha256:" + "b" * 64),
            planned_config=planned,
            pipeline=pipeline,
            expected_runtime_fingerprint_checksum=_CHECKSUM,
        )


def test_projection_rejects_nonhistorical_noise_seed_or_budget() -> None:
    """Even an internally consistent planned row must implement the frozen policy."""
    pipeline = _training_result()
    historical = _historical_config(pipeline)
    nonhistorical_configs = (
        replace(historical, test_noise_id="depolarizing_1s_all"),
        replace(historical, evaluation_seed=1),
        replace(historical, trajectory_budget=499),
    )

    for planned in nonhistorical_configs:
        with pytest.raises(ValueError, match="fixed historical noise"):
            runner_module.validate_historical_evaluation_record(
                target_seed=100,
                record=_evaluation_record(planned),
                planned_config=planned,
                pipeline=pipeline,
                expected_runtime_fingerprint_checksum=_CHECKSUM,
            )


def _comparison_report(*, discrepant: bool = False, failed: bool = False) -> LegacyReproductionReport:
    """Build a typed successful, discrepant, or failed five-row report.

    Returns:
        The requested synthetic report variant.
    """
    reference = load_archived_layerwise_reference()
    fidelities = (0.1,) * 5 if discrepant else reference.fidelities
    outcomes = [
        LegacyReproductionOutcome(
            target_seed=seed,
            status="success",
            computed_fidelity=fidelity,
            source_record_id=f"phase2_evaluation_cli_seed_{seed}",
            source_record_checksum=canonical_checksum({"seed": seed, "fidelity": fidelity}),
            runtime_fingerprint_checksum=canonical_checksum({"seed": seed, "runtime": "cli"}),
        )
        for seed, fidelity in zip(LEGACY_REPRODUCTION_TARGET_SEEDS, fidelities, strict=True)
    ]
    if failed:
        outcomes[2] = LegacyReproductionOutcome(
            target_seed=300,
            status="failure",
            computed_fidelity=None,
            source_record_id="phase2_evaluation_cli_seed_300_failure",
            source_record_checksum=canonical_checksum({"seed": 300, "failure": "test"}),
            runtime_fingerprint_checksum=canonical_checksum({"seed": 300, "runtime": "cli"}),
            failure_type="RuntimeError",
            failure_message="synthetic failure",
        )
    return compare_legacy_reproduction(
        outcomes,
        tolerance=1.0e-6,
        tolerance_rationale="Pinned structural CLI status test.",
        source_manifest_checksum=canonical_checksum({"manifest": "cli"}),
        runtime_checksum=canonical_checksum({"runtime": "cli"}),
    )


def test_cli_exit_code_distinguishes_success_discrepancy_and_failed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automation receives nonzero status for scientific mismatch or incompleteness."""
    reports = (
        (_comparison_report(), HISTORICAL_REPRODUCTION_SUCCESS_EXIT_CODE),
        (_comparison_report(discrepant=True), HISTORICAL_REPRODUCTION_DISCREPANCY_EXIT_CODE),
        (_comparison_report(failed=True), HISTORICAL_REPRODUCTION_FAILURE_EXIT_CODE),
    )
    for report, expected in reports:
        monkeypatch.setattr(
            runner_module,
            "run_historical_reproduction_job",
            lambda *_args, _report=report, **_kwargs: _report,
        )
        assert runner_module.main(["--output-root", str(tmp_path / "cli"), "--execute-expensive"]) == expected


def test_cli_maps_setup_exceptions_to_failure_status_and_concise_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Setup/integrity exceptions cannot escape or masquerade as discrepancies."""

    def fail_job(*_args: object, **_kwargs: object) -> LegacyReproductionReport:
        msg = "source snapshot changed"
        raise ValueError(msg)

    monkeypatch.setattr(runner_module, "run_historical_reproduction_job", fail_job)
    status = runner_module.main(["--output-root", str(tmp_path / "cli_failure"), "--execute-expensive"])

    assert status == HISTORICAL_REPRODUCTION_FAILURE_EXIT_CODE
    assert capsys.readouterr().err == "Historical reproduction failed: source snapshot changed\n"


class _EvaluationConfig(PipelineEvaluationConfig):
    """Minimal typed configuration exposing only fields used by the callback."""

    @property
    def configuration_checksum(self) -> str:
        """Fixed synthetic row checksum."""
        return _CHECKSUM


class _Target:
    """Small target-vector provider used after numerical calls are stubbed."""

    @staticmethod
    def state_vector_copy() -> np.ndarray:
        """Return a normalized two-qubit target."""
        vector = np.zeros(4, dtype=np.complex128)
        vector[0] = 1.0
        return vector


def _small_materialized_circuit() -> LayerwiseMaterializedCircuit:
    """Return a three-gate deterministic circuit payload for callback isolation."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rx", (0,), param_index=0, logical_gate_id="rx_0"),
            ParameterizedGate("ry", (1,), param_index=1, logical_gate_id="ry_1"),
            ParameterizedGate("rzz", (0, 1), param_index=2, logical_gate_id="rzz_0_1"),
        ],
        num_params=3,
    )
    return LayerwiseMaterializedCircuit(
        NoisyKrotovCircuitBinding(circuit, "wp19_evaluation_stub"),
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )


def test_evaluation_callback_freezes_legacy_seed_and_500_trajectory_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stubbed tiny circuit verifies policy arguments without running q8 TJM."""
    config = object.__new__(_EvaluationConfig)
    object.__setattr__(config, "max_bond_dimension", None)
    object.__setattr__(config, "svd_threshold", 0.0)
    object.__setattr__(config, "truncation_mode", "discarded_weight")
    object.__setattr__(config, "min_bond_dimension", 1)
    object.__setattr__(config, "materialized_circuit_checksum", _CHECKSUM)
    captured: dict[str, object] = {}

    def sample(*args: object, **kwargs: object) -> KrotovFixedMapEnsemble:
        captured["tjm_options"] = args[5]
        captured.update(kwargs)
        identity = KrotovNoiseMap(source_gate_index=0, is_identity=True)
        return KrotovFixedMapEnsemble(
            role=cast("KrotovMapRole", kwargs["role"]),
            resolved_seed=cast("int", kwargs["resolved_seed"]),
            stage_index=cast("int", kwargs["stage_index"]),
            stage_id=cast("str", kwargs["stage_id"]),
            stage_configuration_checksum=cast("str", kwargs["stage_configuration_checksum"]),
            circuit_checksum=cast("str", kwargs["circuit_checksum"]),
            provider_checksum=cast("str", kwargs["provider_checksum"]),
            ensemble_index=cast("int", kwargs["ensemble_index"]),
            refresh_index=cast("int", kwargs["refresh_index"]),
            global_iteration_start=cast("int", kwargs["global_iteration_start"]),
            trajectory_maps=[[identity] for _ in range(500)],
        )

    monkeypatch.setattr(runner_module, "sample_krotov_fixed_map_ensemble", sample)

    def noiseless_metrics(*_: object, **__: object) -> tuple[float, float]:
        return 0.1, 0.9

    def noisy_metrics(*_: object, **__: object) -> tuple[float, float, list[float]]:
        return 0.2, 0.8, [0.8] * 500

    monkeypatch.setattr(runner_module, "state_preparation_metrics", noiseless_metrics)
    monkeypatch.setattr(
        runner_module,
        "noisy_state_preparation_metrics",
        noisy_metrics,
    )

    measurement = runner_module.evaluate_historical_materialized_circuit(
        config,
        _small_materialized_circuit(),
        cast("LegacyMaterializedTarget", _Target()),
    )

    options = cast("KrotovTJMOptions", captured["tjm_options"])
    assert options.num_trajectories == 500
    assert options.random_seed == 0
    assert captured["role"] == "pilot_evaluation"
    assert captured["resolved_seed"] == 0
    assert captured["legacy_linear_seed"] is True
    assert captured["legacy_compact_replay"] is False
    assert measurement.noiseless_fidelity == pytest.approx(0.9)
    assert measurement.trajectory_fidelities == (0.8,) * 500
    assert measurement.normalized_work["test_trajectories"] == 500
    assert measurement.normalized_work["trajectory_gate_applications"] == 3000
    assert len(measurement.fixed_map_ensembles) == 1
