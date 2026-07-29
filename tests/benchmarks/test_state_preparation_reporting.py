# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for atomic state-preparation benchmark reporting."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

import benchmarks.state_preparation.reporting as reporting_module
from benchmarks.state_preparation import (
    REPORT_MANIFEST_FORMAT,
    AnsatzConfig,
    ArtifactVerificationError,
    BenchmarkConfig,
    BenchmarkFailure,
    BenchmarkReportStore,
    BenchmarkResult,
    DuplicateRunError,
    EvaluationConfig,
    InitializationConfig,
    KrotovStatePreparationMethod,
    NoiseConfig,
    OptimizerConfig,
    ProvenanceMismatchError,
    RunProvenance,
    TargetSelection,
    atomic_write_bytes,
    capture_run_provenance,
    create_trajectory_sidecar,
    evaluate_state_preparation_artifact,
    load_target_collection,
    read_csv_records,
    read_trajectory_sidecar,
    train_state_preparation_method,
)

if TYPE_CHECKING:
    import os
    from pathlib import Path

    from benchmarks.state_preparation import (
        IndependentEvaluation,
        StatePreparationTrainingArtifact,
        TargetCollection,
    )


def _provenance(*, commit: str = "1" * 40, dirty: bool = False) -> RunProvenance:
    """Return compact fixed provenance for reporting tests."""
    return RunProvenance(
        software_versions={
            "yaqs": "0.0.0",
            "python": "3.11.0",
            "numpy": "2.3.0",
            "scipy": "1.16.0",
        },
        git_commit=commit,
        git_dirty=dirty,
        git_diff_checksum=f"sha256:{'a' * 64}" if dirty else None,
    )


def _optimizer() -> OptimizerConfig:
    """Return a deterministic zero-iteration optimizer."""
    return OptimizerConfig(
        optimizer_id="krotov",
        max_iterations=0,
        optimizer_seed=17,
        hyperparameters={"step_size": 0.1, "schedule": {"kind": "constant"}},
        train_trajectories_or_shots=0,
        training_seed=None,
    )


def _config(
    targets: TargetCollection,
    *,
    test_seed: int = 23,
    sidecar: bool = True,
) -> BenchmarkConfig:
    """Return a small resolved noisy benchmark cell."""
    target = targets.load_target(6, "gaussian_mu0p5_sigma0p1")
    return BenchmarkConfig(
        method_id="krotov",
        method_version="1",
        target=TargetSelection(
            num_qubits=target.num_qubits,
            target_id=target.target_id,
            target_seed=target.seed,
            fixture_format=targets.fixture_format,
            fixture_checksum=targets.fixture_checksum,
        ),
        ansatz=AnsatzConfig(0, initial_single_qubit_layer=True),
        initialization=InitializationConfig(rule="random_normal", seed=11, scale=0.1),
        optimizer=_optimizer(),
        evaluation=EvaluationConfig(
            test_trajectories_or_shots=3,
            test_seed=test_seed,
            store_trajectory_sidecar=sidecar,
            confidence_level=0.95,
            confidence_interval_method="normal_clipped",
        ),
        training_noise=NoiseConfig("noiseless"),
        test_noise=NoiseConfig("dephasing_1s_1q", tjm_dt=1.0),
    )


@pytest.fixture(scope="module")
def evaluated_problem() -> tuple[
    KrotovStatePreparationMethod,
    StatePreparationTrainingArtifact,
    IndependentEvaluation,
    BenchmarkConfig,
    TargetCollection,
]:
    """Train and evaluate one inexpensive reporting fixture.

    Returns:
        Method, artifact, evaluation, configuration, and targets.
    """
    targets = load_target_collection()
    config = _config(targets)
    method = KrotovStatePreparationMethod()
    artifact = train_state_preparation_method(method, config, targets)
    evaluation = evaluate_state_preparation_artifact(method, artifact, config, targets)
    return method, artifact, evaluation, config, targets


def _write_success(
    store: BenchmarkReportStore,
    artifact: StatePreparationTrainingArtifact,
    evaluation: IndependentEvaluation,
    config: BenchmarkConfig,
    *,
    replace_existing: bool = False,
) -> BenchmarkResult:
    """Write a standard successful fixture row.

    Returns:
        The published result.
    """
    return store.write_success(
        config=config,
        artifact=artifact,
        evaluation=evaluation,
        optimization_wall_time_seconds=1.25,
        evaluation_wall_time_seconds=0.75,
        replace=replace_existing,
    )


def _run_test_git(repository: Path, *arguments: str) -> None:
    """Run a trusted Git command for a temporary test repository."""
    git_executable = shutil.which("git")
    if git_executable is None:
        pytest.skip("Git is required for provenance tests.")
    subprocess.run(  # noqa: S603 -- resolved executable and fixed test arguments
        (git_executable, "-C", repository, *arguments),
        check=True,
        shell=False,
    )


def test_trajectory_sidecar_is_deterministic_and_strict() -> None:
    """Sidecars must be reproducible, compressed, and identity-bound."""
    kwargs = {
        "run_id": "spr-v1-test",
        "training_id": "spt-v1-test",
        "repetition": 2,
        "fidelities": (0.1, 0.5, 0.9),
    }
    first = create_trajectory_sidecar(**kwargs)
    second = create_trajectory_sidecar(**kwargs)
    assert first == second
    assert read_trajectory_sidecar(
        first,
        expected_run_id="spr-v1-test",
        expected_training_id="spt-v1-test",
        expected_repetition=2,
        expected_count=3,
    ) == pytest.approx((0.1, 0.5, 0.9))
    with pytest.raises(ValueError, match="identity"):
        read_trajectory_sidecar(
            first,
            expected_run_id="different",
            expected_training_id="spt-v1-test",
            expected_repetition=2,
            expected_count=3,
        )


def test_success_writes_consistent_jsonl_csv_manifest_and_artifacts(
    tmp_path: Path,
    evaluated_problem: tuple[
        KrotovStatePreparationMethod,
        StatePreparationTrainingArtifact,
        IndependentEvaluation,
        BenchmarkConfig,
        TargetCollection,
    ],
) -> None:
    """One success must publish every versioned output consistently."""
    _method, artifact, evaluation, config, _targets = evaluated_problem
    store = BenchmarkReportStore(tmp_path / "run", _provenance())
    result = _write_success(store, artifact, evaluation, config)

    json_lines = store.results_jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(json_lines) == 1
    assert BenchmarkResult.from_json(json_lines[0]) == result
    assert read_csv_records(store.results_csv_path) == (result,)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_format"] == REPORT_MANIFEST_FORMAT
    assert manifest["record_count"] == 1
    assert manifest["successful_run_ids"] == [config.run_id]
    assert result.wall_time_seconds == pytest.approx(2.0)
    assert (store.output_directory / result.parameter_checkpoint_path).read_bytes() == artifact.checkpoint_payload
    assert store.load_trajectory_fidelities(result) == pytest.approx(evaluation.trajectory_fidelities)
    assert not tuple(store.output_directory.rglob(".*.tmp"))


def test_duplicate_prevention_resume_and_explicit_replacement(
    tmp_path: Path,
    evaluated_problem: tuple[
        KrotovStatePreparationMethod,
        StatePreparationTrainingArtifact,
        IndependentEvaluation,
        BenchmarkConfig,
        TargetCollection,
    ],
) -> None:
    """Completed rows must resume, reject duplicates, and replace explicitly."""
    _method, artifact, evaluation, config, _targets = evaluated_problem
    output = tmp_path / "run"
    store = BenchmarkReportStore(output, _provenance())
    first = _write_success(store, artifact, evaluation, config)
    assert store.is_completed(config)
    with pytest.raises(DuplicateRunError):
        _write_success(store, artifact, evaluation, config)

    resumed = BenchmarkReportStore(output, _provenance())
    assert resumed.records == (first,)
    replacement = resumed.write_success(
        config=config,
        artifact=artifact,
        evaluation=evaluation,
        optimization_wall_time_seconds=2.0,
        evaluation_wall_time_seconds=3.0,
        replace=True,
    )
    assert len(resumed.records) == 1
    assert replacement.wall_time_seconds == pytest.approx(5.0)


def test_success_rejects_evaluation_from_another_run(
    tmp_path: Path,
    evaluated_problem: tuple[
        KrotovStatePreparationMethod,
        StatePreparationTrainingArtifact,
        IndependentEvaluation,
        BenchmarkConfig,
        TargetCollection,
    ],
) -> None:
    """Reporting must bind reused training artifacts to the full evaluation cell."""
    _method, artifact, evaluation, config, _targets = evaluated_problem
    changed_config = replace(config, evaluation=replace(config.evaluation, test_seed=29))
    store = BenchmarkReportStore(tmp_path / "run", _provenance())

    with pytest.raises(ValueError, match="run identities"):
        _write_success(store, artifact, evaluation, changed_config)
    assert store.records == ()


def test_provenance_mismatch_requires_explicit_override(
    tmp_path: Path,
    evaluated_problem: tuple[
        KrotovStatePreparationMethod,
        StatePreparationTrainingArtifact,
        IndependentEvaluation,
        BenchmarkConfig,
        TargetCollection,
    ],
) -> None:
    """Scientific result reuse must compare Git and software fingerprints."""
    _method, artifact, evaluation, config, _targets = evaluated_problem
    output = tmp_path / "run"
    store = BenchmarkReportStore(output, _provenance())
    _write_success(store, artifact, evaluation, config)
    changed = _provenance(commit="2" * 40)
    with pytest.raises(ProvenanceMismatchError):
        BenchmarkReportStore(output, changed)
    resumed = BenchmarkReportStore(output, changed, allow_provenance_mismatch=True)
    assert resumed.is_completed(config)
    manifest = json.loads(resumed.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["provenance_history"]) == 2


def test_failure_rows_preserve_previous_success(
    tmp_path: Path,
    evaluated_problem: tuple[
        KrotovStatePreparationMethod,
        StatePreparationTrainingArtifact,
        IndependentEvaluation,
        BenchmarkConfig,
        TargetCollection,
    ],
) -> None:
    """A later failed cell must not disturb an earlier successful row."""
    _method, artifact, evaluation, config, _targets = evaluated_problem
    store = BenchmarkReportStore(tmp_path / "run", _provenance())
    success = _write_success(store, artifact, evaluation, config)
    failed_config = replace(config, evaluation=replace(config.evaluation, test_seed=29))
    failure = store.write_failure(
        config=failed_config,
        failure_phase="evaluation",
        exception=RuntimeError("trajectory worker stopped"),
        wall_time_seconds=4.0,
        retryable=True,
    )
    assert store.records == (success, failure)
    assert isinstance(store.records[0], BenchmarkResult)
    assert isinstance(store.records[1], BenchmarkFailure)
    assert read_csv_records(store.results_csv_path) == store.records


def test_interrupted_tail_and_abandoned_temporary_file_are_recovered(
    tmp_path: Path,
    evaluated_problem: tuple[
        KrotovStatePreparationMethod,
        StatePreparationTrainingArtifact,
        IndependentEvaluation,
        BenchmarkConfig,
        TargetCollection,
    ],
) -> None:
    """Resume must discard only an unterminated partial tail and stale temp."""
    _method, artifact, evaluation, config, _targets = evaluated_problem
    output = tmp_path / "run"
    store = BenchmarkReportStore(output, _provenance())
    result = _write_success(store, artifact, evaluation, config)
    with store.results_jsonl_path.open("ab") as stream:
        stream.write(b'{"status":"success"')
    abandoned = output / ".results.jsonl.interrupted.tmp"
    abandoned.write_bytes(b"partial replacement")

    resumed = BenchmarkReportStore(output, _provenance())
    assert resumed.records == (result,)
    assert resumed.results_jsonl_path.read_bytes().endswith(b"\n")
    assert not abandoned.exists()
    assert read_csv_records(resumed.results_csv_path) == (result,)


def test_checkpoint_corruption_blocks_resume(
    tmp_path: Path,
    evaluated_problem: tuple[
        KrotovStatePreparationMethod,
        StatePreparationTrainingArtifact,
        IndependentEvaluation,
        BenchmarkConfig,
        TargetCollection,
    ],
) -> None:
    """Stored checkpoints must be checksum-verified before reuse."""
    _method, artifact, evaluation, config, _targets = evaluated_problem
    output = tmp_path / "run"
    store = BenchmarkReportStore(output, _provenance())
    result = _write_success(store, artifact, evaluation, config)
    checkpoint = output / result.parameter_checkpoint_path
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ArtifactVerificationError, match="checksum mismatch"):
        BenchmarkReportStore(output, _provenance())


def test_overwrite_removes_only_managed_outputs(
    tmp_path: Path,
    evaluated_problem: tuple[
        KrotovStatePreparationMethod,
        StatePreparationTrainingArtifact,
        IndependentEvaluation,
        BenchmarkConfig,
        TargetCollection,
    ],
) -> None:
    """Explicit overwrite must preserve foreign files in the output root."""
    _method, artifact, evaluation, config, _targets = evaluated_problem
    output = tmp_path / "run"
    store = BenchmarkReportStore(output, _provenance())
    _write_success(store, artifact, evaluation, config)
    foreign = output / "notes.txt"
    foreign.write_text("keep me", encoding="utf-8")

    overwritten = BenchmarkReportStore(output, _provenance(commit="2" * 40), overwrite=True)
    assert overwritten.records == ()
    assert foreign.read_text(encoding="utf-8") == "keep me"
    assert not tuple(overwritten.checkpoint_directory.glob("*"))


def test_atomic_write_preserves_old_file_when_replace_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed atomic replacement must retain the previous destination."""
    destination = tmp_path / "result.json"
    destination.write_bytes(b"old")

    def fail_replace(_source: os.PathLike[str], _destination: os.PathLike[str]) -> None:
        msg = "simulated interruption"
        raise OSError(msg)

    monkeypatch.setattr(reporting_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        atomic_write_bytes(destination, b"new")
    assert destination.read_bytes() == b"old"
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_capture_run_provenance_tracks_dirty_content(tmp_path: Path) -> None:
    """Git provenance must distinguish clean, tracked, and untracked content."""
    repository = tmp_path / "repository"
    repository.mkdir()
    _run_test_git(repository, "init", "-q")
    _run_test_git(repository, "config", "user.email", "test@example.com")
    _run_test_git(repository, "config", "user.name", "Test User")
    tracked = repository / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _run_test_git(repository, "add", "tracked.txt")
    _run_test_git(repository, "commit", "-q", "-m", "Initial")

    clean = capture_run_provenance(repository)
    assert not clean.git_dirty
    tracked.write_text("changed\n", encoding="utf-8")
    dirty_tracked = capture_run_provenance(repository)
    assert dirty_tracked.git_dirty
    assert dirty_tracked.git_diff_checksum is not None
    untracked = repository / "untracked.txt"
    untracked.write_text("one\n", encoding="utf-8")
    dirty_untracked = capture_run_provenance(repository)
    untracked.write_text("two\n", encoding="utf-8")
    changed_untracked = capture_run_provenance(repository)
    assert dirty_tracked.git_diff_checksum != dirty_untracked.git_diff_checksum
    assert dirty_untracked.git_diff_checksum != changed_untracked.git_diff_checksum

    excluded = capture_run_provenance(repository, excluded_paths=(untracked,))
    untracked.write_text("three\n", encoding="utf-8")
    changed_excluded = capture_run_provenance(repository, excluded_paths=(untracked,))
    assert excluded.git_diff_checksum == changed_excluded.git_diff_checksum
