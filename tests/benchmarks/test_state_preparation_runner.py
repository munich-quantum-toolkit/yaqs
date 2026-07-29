# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the state-preparation benchmark command-line runner."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

import benchmarks.state_preparation.runner as runner_module
from benchmarks.state_preparation import (
    NOISE_IDS,
    BenchmarkConfig,
    BenchmarkFailure,
    KrotovStatePreparationMethod,
    StatePreparationTrainingArtifact,
    TargetCollection,
    read_csv_records,
)
from benchmarks.state_preparation.runner import (
    MINIMUM_NOISE_IDS,
    RunnerConfigurationError,
    build_benchmark_matrix,
    execute_benchmark_matrix,
    parse_arguments,
    resolve_options,
    run,
)


def _options(*arguments: str) -> runner_module.RunnerOptions:
    """Resolve compact test CLI arguments.

    Returns:
        Validated runner options.
    """
    return resolve_options(parse_arguments(arguments))


def _repository_root() -> Path:
    """Return the repository containing this test module."""
    return Path(__file__).parents[2]


def test_configuration_precedence_and_boolean_override(tmp_path: Path) -> None:
    """CLI values must override JSON values, which override preset defaults."""
    configuration = tmp_path / "runner.json"
    configuration.write_text(
        json.dumps({
            "preset": "minimum",
            "num_qubits": 6,
            "target_id": ["tfim_critical", "gaussian_mu0p5_sigma0p1"],
            "optimizer_iterations": 7,
            "output_dir": str(tmp_path / "from-json"),
            "overwrite": True,
        }),
        encoding="utf-8",
    )

    options = _options(
        "--config",
        str(configuration),
        "--preset",
        "full",
        "--optimizer-iterations",
        "3",
        "--resume",
    )
    assert options.preset == "full"
    assert options.num_qubits == (6,)
    assert options.target_ids == ("gaussian_mu0p5_sigma0p1", "tfim_critical")
    assert options.optimizer_iterations == 3
    assert options.output_dir == (tmp_path / "from-json").resolve()
    assert options.resume
    assert not options.overwrite
    assert options.noise_ids == NOISE_IDS


@pytest.mark.parametrize(("preset", "expected_rows"), [("minimum", 108), ("full", 216)])
def test_canonical_matrix_cardinality(preset: str, expected_rows: int) -> None:
    """Canonical presets must expand to their frozen per-method cardinality."""
    options = _options("--preset", preset, "--dry-run")
    matrix = build_benchmark_matrix(options)
    assert len(matrix) == expected_rows
    assert len({config.run_id for config in matrix}) == expected_rows
    expected_noise_ids = MINIMUM_NOISE_IDS if preset == "minimum" else NOISE_IDS
    assert {config.test_noise.noise_id for config in matrix} == set(expected_noise_ids)


def test_filters_are_canonical_and_unknown_identifiers_fail() -> None:
    """Repeated filters must deduplicate canonically and reject unknown names."""
    options = _options(
        "--preset",
        "full",
        "--num-qubits",
        "12",
        "--target-id",
        "tfim_critical",
        "--target-id",
        "gaussian_mu0p5_sigma0p1",
        "--noise-id",
        "dephasing_1s_1q",
        "--noise-id",
        "noiseless",
        "--num-layers",
        "4",
        "--initialization-seed",
        "99",
        "--optimizer-iterations",
        "0",
    )
    matrix = build_benchmark_matrix(options)
    assert len(matrix) == 4
    assert [config.target.target_id for config in matrix] == [
        "gaussian_mu0p5_sigma0p1",
        "gaussian_mu0p5_sigma0p1",
        "tfim_critical",
        "tfim_critical",
    ]
    assert [config.test_noise.noise_id for config in matrix[:2]] == ["noiseless", "dephasing_1s_1q"]

    with pytest.raises(RunnerConfigurationError, match="Unknown target_ids"):
        _options("--target-id", "not-a-target")
    with pytest.raises(RunnerConfigurationError, match="must be even"):
        _options("--num-layers", "3")
    with pytest.raises(RunnerConfigurationError, match="test_trajectories"):
        _options("--test-trajectories", "0")


def test_dry_run_is_deterministic_and_does_not_mutate_output(tmp_path: Path) -> None:
    """Dry-run output must be stable and leave the output path untouched."""
    output = tmp_path / "must-not-exist"
    arguments = (
        "--preset",
        "smoke",
        "--noise-id",
        "noiseless",
        "--output-dir",
        str(output),
        "--dry-run",
    )
    first_stdout = io.StringIO()
    second_stdout = io.StringIO()
    assert run(arguments, stdout=first_stdout, stderr=io.StringIO()) == 0
    assert run(arguments, stdout=second_stdout, stderr=io.StringIO()) == 0
    assert first_stdout.getvalue() == second_stdout.getvalue()
    assert '"result_rows":1' in first_stdout.getvalue()
    assert "DRY RUN" in first_stdout.getvalue()
    assert not output.exists()


def test_smoke_run_and_partial_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial smoke run must resume to all noise implementations safely."""
    output = tmp_path / "smoke"
    training_calls = 0
    real_train = runner_module.train_state_preparation_method

    def counted_train(
        method: KrotovStatePreparationMethod,
        config: BenchmarkConfig,
        targets: TargetCollection,
        *,
        checkpoint_root: Path | None = None,
    ) -> StatePreparationTrainingArtifact:
        nonlocal training_calls
        training_calls += 1
        return real_train(method, config, targets, checkpoint_root=checkpoint_root)

    monkeypatch.setattr(runner_module, "train_state_preparation_method", counted_train)
    partial_stdout = io.StringIO()
    common = (
        "--preset",
        "smoke",
        "--test-trajectories",
        "1",
        "--output-dir",
        str(output),
    )
    assert (
        run(
            (*common, "--noise-id", "noiseless", "--noise-id", "ballarin_coupled"),
            stdout=partial_stdout,
            stderr=io.StringIO(),
        )
        == 0
    )
    first_payload = (output / "results.jsonl").read_text(encoding="utf-8")
    assert len(first_payload.splitlines()) == 2

    resumed_stdout = io.StringIO()
    assert run((*common, "--resume"), stdout=resumed_stdout, stderr=io.StringIO()) == 0
    records = read_csv_records(output / "results.csv")
    assert len(records) == len(NOISE_IDS)
    assert {record.config.test_noise.noise_id for record in records} == set(NOISE_IDS)
    assert (output / "results.jsonl").read_text(encoding="utf-8").startswith(first_payload)
    assert "skipped=2" in resumed_stdout.getvalue()
    assert training_calls == 2


def test_failure_rows_and_fail_fast_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training failures must become rows, with fail-fast stopping after one."""
    real_train = runner_module.train_state_preparation_method

    def fail_training(*_args: object, **_kwargs: object) -> object:
        msg = "simulated optimizer failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(runner_module, "train_state_preparation_method", fail_training)
    base = _options(
        "--preset",
        "smoke",
        "--noise-id",
        "noiseless",
        "--noise-id",
        "ballarin_coupled",
        "--noise-id",
        "dephasing_1s_1q",
        "--optimizer-iterations",
        "0",
        "--output-dir",
        str(tmp_path / "continue"),
    )
    matrix = build_benchmark_matrix(base)
    continued = execute_benchmark_matrix(
        base,
        matrix,
        repository_root=_repository_root(),
        stream=io.StringIO(),
    )
    assert continued.failed == 3
    assert continued.attempted == 3
    assert all(isinstance(record, BenchmarkFailure) for record in read_csv_records(base.output_dir / "results.csv"))

    fail_fast = replace(base, output_dir=tmp_path / "fail-fast", fail_fast=True)
    stopped = execute_benchmark_matrix(
        fail_fast,
        matrix,
        repository_root=_repository_root(),
        stream=io.StringIO(),
    )
    assert stopped.failed == 1
    assert stopped.attempted == 1
    assert len(read_csv_records(fail_fast.output_dir / "results.csv")) == 1

    def fail_evaluation(*_args: object, **_kwargs: object) -> object:
        msg = "simulated trajectory failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(runner_module, "train_state_preparation_method", real_train)
    monkeypatch.setattr(runner_module, "evaluate_state_preparation_artifact", fail_evaluation)
    evaluation_failure = replace(base, output_dir=tmp_path / "evaluation-failure")
    evaluated = execute_benchmark_matrix(
        evaluation_failure,
        matrix,
        repository_root=_repository_root(),
        stream=io.StringIO(),
    )
    evaluation_records = read_csv_records(evaluation_failure.output_dir / "results.csv")
    assert evaluated.failed == 3
    assert {record.failure_phase for record in evaluation_records if isinstance(record, BenchmarkFailure)} == {
        "evaluation"
    }


def test_existing_output_requires_explicit_resume_or_overwrite(tmp_path: Path) -> None:
    """A canonical stream must never be reused implicitly."""
    output = tmp_path / "existing"
    output.mkdir()
    (output / "results.jsonl").write_text("", encoding="utf-8")
    options = replace(_options("--preset", "smoke"), output_dir=output)
    matrix = build_benchmark_matrix(options)
    with pytest.raises(RunnerConfigurationError, match="--resume or --overwrite"):
        execute_benchmark_matrix(
            options,
            matrix,
            repository_root=_repository_root(),
            stream=io.StringIO(),
        )
