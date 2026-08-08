# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Configuration, authorization, and custody tests for the WP22 runner."""

# These tests intentionally exercise the module's narrow validation seams so
# target-access ordering can be proven without opening a confirmatory artifact.

from __future__ import annotations

import json
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation import runner as phase1_runner
from benchmarks.state_preparation import training_runner
from benchmarks.state_preparation.constants import BALLARIN_NOISE_ID, NOISELESS_NOISE_ID
from benchmarks.state_preparation.phase2.layerwise_bmpd import build_layerwise_bmpd_crn_v2_template
from benchmarks.state_preparation.phase2.protocol import (
    DEFAULT_PREREGISTRATION_PATH,
    TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
)
from benchmarks.state_preparation.phase2.targets import (
    build_target_population_config,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.training_orchestration import TrainingRunPlan
from benchmarks.state_preparation.training_runner import (
    TRAINING_RUNNER_CONFIGURATION_FORMAT,
    TrainingRunnerConfigurationError,
    load_configuration_file,
    parse_arguments,
    resolve_options,
)
from tests.benchmarks.test_state_preparation_wp22_execution_context import _context

if TYPE_CHECKING:
    from benchmarks.state_preparation.phase2.training_schedules import TrainingStrategySchedule

_CHECKSUM_A = f"sha256:{'a' * 64}"
_CHECKSUM_B = f"sha256:{'b' * 64}"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            f'{{"format":"{TRAINING_RUNNER_CONFIGURATION_FORMAT}","nested":{{"key":1,"key":2}}}}',
            "Duplicate",
        ),
        (f'{{"format":"{TRAINING_RUNNER_CONFIGURATION_FORMAT}","normalized_compute_cap":NaN}}', "Non-finite"),
        (
            f'{{"format":"{TRAINING_RUNNER_CONFIGURATION_FORMAT}","normalized_compute_cap":Infinity}}',
            "Non-finite",
        ),
        (f'{{"format":"{TRAINING_RUNNER_CONFIGURATION_FORMAT}","normalized_compute_cap":1e999}}', "non-finite"),
        (f'{{"format":"{TRAINING_RUNNER_CONFIGURATION_FORMAT}","surprise":1}}', "Unknown"),
    ],
)
def test_configuration_json_is_strict_and_finite(tmp_path: Path, payload: str, message: str) -> None:
    """Duplicate keys, non-finite numbers, and unknown fields are rejected."""
    configuration = tmp_path / "runner.json"
    configuration.write_text(payload, encoding="utf-8")

    with pytest.raises(TrainingRunnerConfigurationError, match=message):
        load_configuration_file(configuration)


def test_configuration_aliases_cannot_shadow_canonical_fields(tmp_path: Path) -> None:
    """A scalar alias cannot silently replace its canonical configuration field."""
    configuration = tmp_path / "runner.json"
    configuration.write_text(
        json.dumps({
            "format": TRAINING_RUNNER_CONFIGURATION_FORMAT,
            "method": "alias",
            "method_id": "canonical",
        }),
        encoding="utf-8",
    )

    with pytest.raises(TrainingRunnerConfigurationError, match="both"):
        load_configuration_file(configuration)


def test_external_entropy_references_are_forbidden_in_json(tmp_path: Path) -> None:
    """Secret-custody paths are accepted only at the explicit CLI boundary."""
    configuration = tmp_path / "runner.json"
    configuration.write_text(
        json.dumps({
            "format": TRAINING_RUNNER_CONFIGURATION_FORMAT,
            "external_entropy_file_specs": ["development/primary_q6=private.key"],
        }),
        encoding="utf-8",
    )

    with pytest.raises(TrainingRunnerConfigurationError, match="Unknown"):
        load_configuration_file(configuration)


def test_cli_overrides_json_and_every_wp22_option_is_resolved(tmp_path: Path) -> None:
    """CLI values win as a whole over JSON values across all WP22 option families."""
    configuration = tmp_path / "runner.json"
    configuration.write_text(
        json.dumps({
            "format": TRAINING_RUNNER_CONFIGURATION_FORMAT,
            "preset": "training-smoke",
            "pipeline_path": "file-pipeline.json",
            "method_id": "file_method",
            "stage_depths": [1, 1],
            "stage_budgets": [3, 3],
            "training_noise_id": "dephasing_1s_all",
            "training_noise_strength": 0.5,
            "trajectory_update": "cross",
            "sampling_policy": "resampled",
            "training_trajectory_count": 4,
            "checkpoint_validation_trajectory_count": 5,
            "crn_refresh_interval": 2,
            "checkpoint_rule": "last_iteration",
            "target_manifest_paths": ["file-target.json"],
            "target_manifest_checksums": [_CHECKSUM_A],
            "data_role": "development",
            "native_two_qubit_cap_per_edge": 10.0,
            "normalized_compute_cap": 100.0,
            "preregistration_path": str(DEFAULT_PREREGISTRATION_PATH),
            "preregistration_checksum": _CHECKSUM_A,
            "resume": True,
            "overwrite": False,
            "dry_run": False,
            "fail_fast": True,
            "legacy_reproduction": False,
            "execute_expensive": False,
            "executor_factory": "file.executors:build_registry",
            "candidate_paths": ["file-candidate.json"],
            "schedule_paths": ["file-schedule.json"],
            "screening_manifest_path": "file-screen.json",
            "screening_evidence_path": "file-screen-evidence.json",
            "promotion_decision_path": "file-promotion.json",
            "final_seal_path": "file-seal.json",
            "configuration_execution_manifest_path": "file-configuration-execution.json",
            "execution_source_manifest_path": "file-execution.json",
            "analysis_source_manifest_path": "file-analysis.json",
            "execution_profile_path": "file-profile.json",
            "binding_catalog_path": "file-binding-catalog.json",
            "target_configuration_paths": ["file-target-config.json"],
            "sample_size_design_path": "file-sample-size.json",
            "resource_calibration_path": "file-resource-calibration.json",
            "resumability_fingerprint_paths": ["file-fingerprint.json"],
            "repository_root": "file-repository",
            "pilot_optimization_seeds": [3],
            "output_dir": "file-output",
        }),
        encoding="utf-8",
    )
    options = resolve_options(
        parse_arguments([
            "--config",
            str(configuration),
            "--preset",
            "paper-pilot",
            "--pipeline",
            "cli-pipeline.json",
            "--method",
            "cli_method",
            "--stage-depth",
            "2",
            "--stage-depth",
            "2",
            "--stage-budget",
            "7",
            "--stage-budget",
            "7",
            "--training-noise-id",
            "depolarizing_1s_all",
            "--training-noise-strength",
            "1.0",
            "--trajectory-update",
            "independent",
            "--sampling-policy",
            "crn_refresh",
            "--training-trajectories",
            "8",
            "--validation-trajectories",
            "9",
            "--crn-refresh-interval",
            "4",
            "--checkpoint-rule",
            "best_validation_fidelity",
            "--target-manifest",
            "cli-target.json",
            "--target-manifest-checksum",
            _CHECKSUM_B,
            "--data-role",
            "secondary_benchmark",
            "--native-two-qubit-cap-per-edge",
            "12",
            "--normalized-compute-cap",
            "250",
            "--preregistration",
            "cli-preregistration.json",
            "--preregistration-checksum",
            _CHECKSUM_B,
            "--no-resume",
            "--overwrite",
            "--dry-run",
            "--no-fail-fast",
            "--executor-factory",
            "cli.executors:build_registry",
            "--candidate",
            "cli-candidate.json",
            "--schedule",
            "cli-schedule.json",
            "--screening-manifest",
            "cli-screen.json",
            "--screening-evidence",
            "cli-screen-evidence.json",
            "--promotion-decision",
            "cli-promotion.json",
            "--final-seal",
            "cli-seal.json",
            "--configuration-execution-manifest",
            "cli-configuration-execution.json",
            "--execution-source-manifest",
            "cli-execution.json",
            "--analysis-source-manifest",
            "cli-analysis.json",
            "--execution-profile",
            "cli-profile.json",
            "--binding-catalog",
            "cli-binding-catalog.json",
            "--target-configuration",
            "cli-target-config.json",
            "--sample-size-design",
            "cli-sample-size.json",
            "--resource-calibration",
            "cli-resource-calibration.json",
            "--resumability-fingerprint",
            "cli-fingerprint.json",
            "--external-entropy-file",
            "development/primary_q6=development.key",
            "--repository-root",
            "cli-repository",
            "--pilot-optimization-seed",
            "17",
            "--pilot-optimization-seed",
            "18",
            "--pilot-optimization-seed",
            "19",
            "--pilot-optimization-seed",
            "20",
            "--pilot-optimization-seed",
            "21",
            "--output",
            str(tmp_path / "cli-output"),
        ])
    )

    assert options.preset == "paper-pilot"
    assert options.pipeline_path == Path("cli-pipeline.json")
    assert options.method_id == "cli_method"
    assert options.stage_depths == (2, 2)
    assert options.stage_budgets == (7, 7)
    assert options.training_noise_id == "depolarizing_1s_all"
    assert options.training_noise_strength == pytest.approx(1.0, abs=0.0)
    assert options.trajectory_update == "independent"
    assert options.sampling_policy == "crn_refresh"
    assert options.training_trajectory_count == 8
    assert options.checkpoint_validation_trajectory_count == 9
    assert options.crn_refresh_interval == 4
    assert options.checkpoint_rule == "best_validation_fidelity"
    assert options.target_manifest_paths == (Path("cli-target.json"),)
    assert options.target_manifest_checksums == (_CHECKSUM_B,)
    assert options.data_role == "secondary_benchmark"
    assert options.native_two_qubit_cap_per_edge == pytest.approx(12.0, abs=0.0)
    assert options.normalized_compute_cap == pytest.approx(250.0, abs=0.0)
    assert options.preregistration_path == Path("cli-preregistration.json")
    assert options.preregistration_checksum == _CHECKSUM_B
    assert (options.resume, options.overwrite, options.dry_run, options.fail_fast) == (False, True, True, False)
    assert not options.legacy_reproduction
    assert not options.execute_expensive
    assert options.executor_factory == "cli.executors:build_registry"
    assert options.candidate_paths == (Path("cli-candidate.json"),)
    assert options.schedule_paths == (Path("cli-schedule.json"),)
    assert options.screening_manifest_path == Path("cli-screen.json")
    assert options.screening_evidence_path == Path("cli-screen-evidence.json")
    assert options.promotion_decision_path == Path("cli-promotion.json")
    assert options.final_seal_path == Path("cli-seal.json")
    assert options.configuration_execution_manifest_path == Path("cli-configuration-execution.json")
    assert options.execution_source_manifest_path == Path("cli-execution.json")
    assert options.analysis_source_manifest_path == Path("cli-analysis.json")
    assert options.execution_profile_path == Path("cli-profile.json")
    assert options.binding_catalog_path == Path("cli-binding-catalog.json")
    assert options.target_configuration_paths == (Path("cli-target-config.json"),)
    assert options.sample_size_design_path == Path("cli-sample-size.json")
    assert options.resource_calibration_path == Path("cli-resource-calibration.json")
    assert options.resumability_fingerprint_paths == (Path("cli-fingerprint.json"),)
    assert options.external_entropy_file_specs == ("development/primary_q6=development.key",)
    assert options.repository_root == Path("cli-repository")
    assert options.pilot_optimization_seeds == (17, 18, 19, 20, 21)
    assert options.output_dir == (tmp_path / "cli-output").resolve()


def test_parser_exposes_only_the_five_exact_presets() -> None:
    """The public preset registry is exact and argparse rejects abbreviations."""
    assert training_runner.TRAINING_PRESETS == (
        "training-smoke",
        "historical-layerwise-reproduction",
        "paper-pilot",
        "paper-screen",
        "paper-confirm",
    )
    assert tuple(parse_arguments(["--preset", preset]).preset for preset in training_runner.TRAINING_PRESETS) == (
        training_runner.TRAINING_PRESETS
    )
    with pytest.raises(SystemExit):
        parse_arguments(["--preset", "paper"])


def test_paper_pilot_uses_five_deterministic_preregistered_seed_slots() -> None:
    """The pilot default is a five-seed schedule and one-seed overrides fail early."""
    options = resolve_options(parse_arguments(["--preset", "paper-pilot", "--dry-run"]))

    assert len(options.pilot_optimization_seeds) == 5
    assert len(set(options.pilot_optimization_seeds)) == 5
    with pytest.raises(TrainingRunnerConfigurationError, match="exactly 5"):
        resolve_options(
            parse_arguments([
                "--preset",
                "paper-pilot",
                "--pilot-optimization-seed",
                "17",
            ])
        )


@pytest.mark.parametrize(
    "pilot_seeds",
    [
        (1, 2, 3, 4, 5),
        tuple(reversed(training_runner.DEFAULT_PILOT_OPTIMIZATION_SEEDS)),
    ],
)
def test_paper_pilot_cli_rejects_noncanonical_five_seed_schedule_before_output(
    tmp_path: Path,
    pilot_seeds: tuple[int, ...],
) -> None:
    """Five distinct or reordered seeds cannot replace the checksum-derived schedule."""
    output = tmp_path / "pilot-output"
    arguments = [
        "--preset",
        "paper-pilot",
        "--target-manifest",
        "development.json",
        "--target-manifest",
        "secondary-q12.json",
        "--candidate",
        "candidate.json",
        "--schedule",
        "schedule.json",
        "--output",
        str(output),
    ]
    for seed in pilot_seeds:
        arguments.extend(("--pilot-optimization-seed", str(seed)))
    options = resolve_options(parse_arguments(arguments))

    with pytest.raises(TrainingRunnerConfigurationError, match="exact checksum-derived"):
        training_runner.run(options, executor=lambda *_args: _CHECKSUM_A)
    assert not output.exists()


def test_help_is_inert_and_documents_the_opt_in_entry_point(capsys: pytest.CaptureFixture[str]) -> None:
    """Requesting help exits before configuration or output paths are touched."""
    with pytest.raises(SystemExit) as error:
        parse_arguments(["--help"])

    assert error.value.code == 0
    assert "python -m benchmarks.state_preparation.training_runner" in capsys.readouterr().out


def test_historical_dry_run_is_canonical_deterministic_and_inert(tmp_path: Path) -> None:
    """Dry-run emits the exact same sealed five-job plan without creating output."""
    output = tmp_path / "must-not-exist"
    arguments = [
        "--preset",
        "historical-layerwise-reproduction",
        "--legacy-reproduction",
        "--dry-run",
        "--output",
        str(output),
    ]
    first_stdout = StringIO()
    second_stdout = StringIO()

    assert training_runner.main(arguments, stdout=first_stdout, stderr=StringIO()) == 0
    assert training_runner.main(arguments, stdout=second_stdout, stderr=StringIO()) == 0
    assert first_stdout.getvalue() == second_stdout.getvalue()
    plan = TrainingRunPlan.from_json(first_stdout.getvalue())
    assert plan.preset == "historical-layerwise-reproduction"
    assert len(plan.jobs) == 5
    assert not output.exists()


def test_historical_execution_requires_both_opt_ins_before_delegation(tmp_path: Path) -> None:
    """WP19 is delegated only after legacy mode and expensive execution are explicit."""
    output = tmp_path / "historical"
    base = [
        "--preset",
        "historical-layerwise-reproduction",
        "--legacy-reproduction",
        "--output",
        str(output),
    ]
    without_expensive = resolve_options(parse_arguments(base))
    with pytest.raises(TrainingRunnerConfigurationError, match="execute-expensive"):
        training_runner.run(without_expensive)
    assert not output.exists()

    calls: list[tuple[Path, dict[str, object]]] = []

    def fake_historical_runner(path: Path, **kwargs: object) -> str:
        """Record the exact safe delegation boundary.

        Returns:
            A sentinel result.
        """
        calls.append((path, kwargs))
        return "delegated"

    authorized = resolve_options(parse_arguments([*base, "--execute-expensive", "--resume"]))
    assert training_runner.run(authorized, historical_runner=fake_historical_runner) == "delegated"
    assert calls == [
        (
            output.resolve(),
            {
                "execute_expensive": True,
                "resume": True,
                "overwrite": False,
                "repository_root": training_runner.DEFAULT_REPOSITORY_ROOT,
            },
        )
    ]
    assert not output.exists()


def test_generic_execution_rejects_missing_context_before_output(tmp_path: Path) -> None:
    """An ordinary preset cannot fall back to candidate-only planning."""
    output = tmp_path / "generic"
    generic = resolve_options(parse_arguments(["--preset", "training-smoke", "--output", str(output)]))

    with pytest.raises(TrainingRunnerConfigurationError, match="execution-profile"):
        training_runner.run(generic)
    assert not output.exists()


def test_cli_executor_factory_loads_typed_registry_without_runner_source_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source-locked factory receives the complete context and returns a registry."""
    module_name = "wp22_executor_factory_test"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "from benchmarks.state_preparation.phase2.training_orchestration import TrainingExecutorRegistry\n"
        "RESULT = 'sha256:' + 'd' * 64\n"
        "SEEN_CONTEXT = None\n"
        "def execute(job, directory, controls):\n"
        "    return RESULT\n"
        "def build_registry(context):\n"
        "    global SEEN_CONTEXT\n"
        "    SEEN_CONTEXT = context\n"
        "    return TrainingExecutorRegistry(legacy_delegate_executor=execute)\n"
        "def invalid_registry(context):\n"
        "    return execute\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    context = _context()
    monkeypatch.setattr(
        training_runner,
        "_executor_factory_source",
        lambda *_arguments: ("runner.py", module_path.resolve()),
    )
    monkeypatch.setattr(training_runner, "verify_execution_source_manifest", lambda *_arguments: ("runner.py",))
    registry = training_runner.load_executor_registry(
        f"{module_name}:build_registry",
        tmp_path,
        context,
    )
    imported = training_runner.importlib.import_module(module_name)
    assert registry.legacy_delegate_executor is imported.execute
    assert imported.SEEN_CONTEXT is context

    with pytest.raises(TrainingRunnerConfigurationError, match="TrainingExecutorRegistry"):
        training_runner.load_executor_registry(
            f"{module_name}:invalid_registry",
            tmp_path,
            context,
        )


def test_executor_factory_never_receives_context_before_complete_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source or fingerprint failure stops extension import and invocation."""
    output = tmp_path / "factory-must-not-run"
    options = resolve_options(
        parse_arguments([
            "--preset",
            "training-smoke",
            "--executor-factory",
            "extension.factory:build_registry",
            "--output",
            str(output),
        ])
    )
    context = _context()
    calls: list[str] = []

    def reject_preflight(_context: object, _repository: Path, _output: Path) -> None:
        msg = "source fingerprint mismatch"
        raise ValueError(msg)

    def forbidden_factory(*_arguments: object) -> object:
        calls.append("factory")
        pytest.fail("factory received the secret-bearing context before preflight")

    monkeypatch.setattr(training_runner.TrainingExecutionContext, "preflight", reject_preflight)
    monkeypatch.setattr(training_runner, "load_executor_registry", forbidden_factory)

    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        training_runner.run(options, context=context)
    assert calls == []
    assert not output.exists()


def test_wrong_external_entropy_stops_target_manifest_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A role-key commitment mismatch preserves the target-manifest custody boundary."""
    preregistration = training_runner.load_initial_preregistration()
    correct_entropy = bytes(range(32))
    config = build_target_population_config(
        preregistration,
        "development",
        role_master_entropy_commitment=role_master_entropy_commitment(correct_entropy),
    )
    wrong_key_path = tmp_path / "wrong.key"
    wrong_key_path.write_bytes(b"z" * 32)
    profile = SimpleNamespace(preset="training-smoke")
    catalog = SimpleNamespace(profile=profile, bindings=())
    fingerprint = SimpleNamespace(
        pipeline_prefix_id=f"phase2_pipeline_prefix_{'a' * 64}",
        content_checksum=_CHECKSUM_A,
    )
    options = replace(
        resolve_options(parse_arguments(["--preset", "training-smoke"])),
        execution_profile_path=Path("profile.json"),
        binding_catalog_path=Path("catalog.json"),
        execution_source_manifest_path=Path("source.json"),
        target_manifest_paths=(Path("target.json"),),
        target_configuration_paths=(Path("config.json"),),
        resumability_fingerprint_paths=(Path("fingerprint.json"),),
        external_entropy_file_specs=(f"development/primary_q6={wrong_key_path}",),
    )

    def fake_decode(_path: Path, name: str, _decoder: object) -> object:
        return {
            "execution-source manifest": object(),
            "training execution profile": profile,
            "repository binding catalog": catalog,
            "target-population configuration": config,
            "resumability fingerprint": fingerprint,
        }[name]

    def forbidden_target_load(_options: object) -> object:
        pytest.fail("target manifest was opened before external entropy matched its commitment")

    monkeypatch.setattr(training_runner, "_decode_artifact", fake_decode)
    monkeypatch.setattr(training_runner, "verify_execution_source_manifest", lambda *_arguments: ())
    monkeypatch.setattr(training_runner, "validate_resumability_source_fingerprints", lambda *_arguments: None)
    monkeypatch.setattr(training_runner, "_load_targets", forbidden_target_load)

    with pytest.raises(TrainingRunnerConfigurationError, match="External entropy"):
        training_runner.build_training_execution_context(options)


@pytest.mark.parametrize("value", ["module", "module:", ":factory", "module:factory.extra", "module-path:factory"])
def test_executor_factory_syntax_is_strict(value: str) -> None:
    """Ambiguous dynamic import spellings are rejected during pure resolution."""
    with pytest.raises(TrainingRunnerConfigurationError, match=r"module\.path:function_name"):
        resolve_options(parse_arguments(["--executor-factory", value]))


def test_ballarin_training_is_rejected_before_artifact_or_output_access(tmp_path: Path) -> None:
    """The evaluation-only Ballarin condition fails during pure option resolution."""
    output = tmp_path / "output"
    missing_pipeline = tmp_path / "must-not-be-opened.json"

    with pytest.raises(TrainingRunnerConfigurationError, match="evaluation-only"):
        resolve_options(
            parse_arguments([
                "--training-noise-id",
                BALLARIN_NOISE_ID,
                "--pipeline",
                str(missing_pipeline),
                "--output",
                str(output),
            ])
        )
    assert not missing_pipeline.exists()
    assert not output.exists()


def test_pipeline_scientific_values_are_assertions_not_overrides(tmp_path: Path) -> None:
    """Matching assertions preserve exact pipeline bytes and mismatches are rejected."""
    template = build_layerwise_bmpd_crn_v2_template(
        training_trajectory_count=8,
        checkpoint_validation_trajectory_count=6,
    )
    pipeline_path = tmp_path / "pipeline.json"
    original = template.to_json()
    pipeline_path.write_text(original, encoding="utf-8")
    policies = tuple(stage.stage_policy for stage in template.stages)
    arguments = ["--pipeline", str(pipeline_path)]
    for policy in policies:
        depth = int(str(policy["output_topology_id"]).rsplit("_d", maxsplit=1)[1])
        arguments.extend(("--stage-depth", str(depth)))
        arguments.extend(("--stage-budget", str(policy["iteration_budget"])))
    arguments.extend((
        "--training-noise-id",
        "depolarizing_1s_all",
        "--training-noise-strength",
        "1",
        "--trajectory-update",
        "independent",
        "--sampling-policy",
        "crn_fixed",
        "--training-trajectories",
        "8",
        "--validation-trajectories",
        "6",
        "--checkpoint-rule",
        "best_validation_fidelity",
    ))
    options = resolve_options(parse_arguments(arguments))
    pipeline = training_runner._load_pipeline(pipeline_path)  # noqa: SLF001 -- focused assertion-helper test

    training_runner._validate_pipeline_assertions(options, pipeline)  # noqa: SLF001 -- focused helper test
    assert pipeline.template == template
    assert pipeline_path.read_text(encoding="utf-8") == original

    mismatch = resolve_options(parse_arguments(["--pipeline", str(pipeline_path), "--stage-budget", "999"]))
    with pytest.raises(TrainingRunnerConfigurationError, match="stage_budgets"):
        training_runner._validate_pipeline_assertions(mismatch, pipeline)  # noqa: SLF001 -- focused helper test
    assert pipeline_path.read_text(encoding="utf-8") == original


def test_paper_confirm_missing_seal_never_loads_target_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing final seal fails before the target-manifest loader is called."""
    options = resolve_options(
        parse_arguments([
            "--preset",
            "paper-confirm",
            "--target-manifest",
            "custodied-target.json",
            "--execute-expensive",
            "--output",
            str(tmp_path / "confirm"),
        ])
    )

    def forbidden_target_load(_options: object) -> object:
        """Fail if the custody boundary is violated."""
        pytest.fail("confirmatory target was accessed before authorization")

    monkeypatch.setattr(training_runner, "_load_targets", forbidden_target_load)
    with pytest.raises(TrainingRunnerConfigurationError, match="final-seal"):
        training_runner.build_training_plan(options)


def test_paper_confirm_missing_configuration_execution_manifest_never_loads_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-configuration execution root is mandatory before held-target access."""
    options = resolve_options(
        parse_arguments([
            "--preset",
            "paper-confirm",
            "--target-manifest",
            "custodied-target.json",
            "--final-seal",
            "seal.json",
            "--execution-source-manifest",
            "execution.json",
            "--analysis-source-manifest",
            "analysis.json",
            "--prior-target-exposure-inventory",
            "prior-exposure.json",
            "--expected-locked-study-head",
            str(tmp_path / "confirmation-study-head.json"),
            "--execute-expensive",
            "--output",
            str(tmp_path / "confirm"),
        ])
    )

    def forbidden_target_load(_options: object) -> object:
        """Fail if the custody boundary is violated."""
        pytest.fail("confirmatory target was accessed without its exact execution manifest")

    monkeypatch.setattr(training_runner, "_load_targets", forbidden_target_load)
    with pytest.raises(TrainingRunnerConfigurationError, match="configuration-execution-manifest"):
        training_runner.build_training_plan(options)


def test_noiseless_training_assertion_matches_explicit_noop_schedule() -> None:
    """The noiseless identifier addresses the typed no-component schedule mode."""
    options = resolve_options(parse_arguments(["--training-noise-id", NOISELESS_NOISE_ID]))
    schedule = cast(
        "TrainingStrategySchedule",
        SimpleNamespace(training_noise=SimpleNamespace(mode="noiseless", components=())),
    )

    training_runner._validate_schedule_assertions(  # noqa: SLF001 -- focused assertion-helper test
        options,
        (schedule,),
        pipeline_present=False,
    )


def test_paper_confirm_verifies_source_lock_before_target_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid execution/source lock prevents any confirmatory target read."""
    options = resolve_options(
        parse_arguments([
            "--preset",
            "paper-confirm",
            "--target-manifest",
            "custodied-target.json",
            "--final-seal",
            "seal.json",
            "--configuration-execution-manifest",
            "configuration-execution.json",
            "--execution-source-manifest",
            "execution.json",
            "--analysis-source-manifest",
            "analysis.json",
            "--execute-expensive",
            "--output",
            str(tmp_path / "confirm"),
        ])
    )
    decoded: list[str] = []
    seal = SimpleNamespace(
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        confirmatory_target_manifest_checksum=_CHECKSUM_A,
    )

    def fake_decode(_path: Path, name: str, _decoder: object) -> object:
        """Return harmless source records while recording artifact order."""
        decoded.append(name)
        return {
            "final confirmation seal": seal,
            "final configuration execution manifest": object(),
            "execution-source manifest": object(),
            "analysis-source manifest": object(),
        }[name]

    def reject_source_lock(*_arguments: object) -> None:
        """Simulate a clean-checkout/source identity mismatch.

        Raises:
            ValueError: Always, to reject the simulated source lock.
        """
        msg = "source mismatch"
        raise ValueError(msg)

    def forbidden_target_load(_options: object) -> object:
        """Fail if the target is accessed after a rejected source lock."""
        pytest.fail("confirmatory target was accessed before source authorization")

    monkeypatch.setattr(training_runner, "_decode_artifact", fake_decode)
    monkeypatch.setattr(training_runner, "verify_final_seal_source_lock", reject_source_lock)
    monkeypatch.setattr(training_runner, "_load_targets", forbidden_target_load)

    with pytest.raises(TrainingRunnerConfigurationError, match="source custody"):
        training_runner.build_training_plan(options)
    assert decoded == [
        "final confirmation seal",
        "final configuration execution manifest",
        "execution-source manifest",
        "analysis-source manifest",
    ]


def test_paper_confirm_rejects_ungoverned_executor_factory_before_target_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirmation rejects every caller-selected executor before target access."""
    options = resolve_options(
        parse_arguments([
            "--preset",
            "paper-confirm",
            "--target-manifest",
            "custodied-target.json",
            "--final-seal",
            "seal.json",
            "--configuration-execution-manifest",
            "configuration-execution.json",
            "--execution-source-manifest",
            "execution.json",
            "--analysis-source-manifest",
            "analysis.json",
            "--executor-factory",
            "rogue.executors:build_registry",
            "--execute-expensive",
            "--output",
            str(tmp_path / "confirm"),
        ])
    )

    def forbidden_target_load(_options: object) -> object:
        """Fail if the ungoverned factory did not stop target access."""
        pytest.fail("confirmatory target was accessed before executor source authorization")

    monkeypatch.setattr(training_runner, "_load_targets", forbidden_target_load)

    with pytest.raises(TrainingRunnerConfigurationError, match="executor_factory"):
        training_runner.build_training_plan(options)


def test_paper_confirm_authorizes_final_artifacts_before_target_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected final authorization cannot open the revealed target path."""
    options = resolve_options(
        parse_arguments([
            "--preset",
            "paper-confirm",
            "--target-manifest",
            "custodied-target.json",
            "--target-configuration",
            "confirmatory-target-config.json",
            "--external-entropy-file",
            "confirmatory/primary_q6=confirmatory.key",
            "--screening-manifest",
            "screening.json",
            "--screening-evidence",
            "screening-evidence.json",
            "--promotion-decision",
            "promotion.json",
            "--sample-size-design",
            "sample-size.json",
            "--resource-calibration",
            "resource-calibration.json",
            "--binding-catalog",
            "binding-catalog.json",
            "--final-seal",
            "seal.json",
            "--configuration-execution-manifest",
            "configuration-execution.json",
            "--execution-source-manifest",
            "execution.json",
            "--analysis-source-manifest",
            "analysis.json",
            "--prior-target-exposure-inventory",
            "prior-exposure.json",
            "--expected-locked-study-head",
            str(tmp_path / "confirmation-study-head.json"),
            "--execute-expensive",
            "--output",
            str(tmp_path / "confirm"),
        ])
    )
    seal = SimpleNamespace(
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        confirmatory_target_manifest_checksum=_CHECKSUM_A,
    )
    screening_manifest = SimpleNamespace(screening_target_manifest_checksum=_CHECKSUM_B)
    resource_calibration = SimpleNamespace(
        content_checksum=_CHECKSUM_A,
        execution_source_manifest_checksum=_CHECKSUM_B,
        pilot_plan_checksum=_CHECKSUM_A,
        screening_plan_checksum=_CHECKSUM_B,
        pilot_custody_checksum=_CHECKSUM_A,
        pilot_calibration_checksum=_CHECKSUM_B,
        screening_custody_checksum=_CHECKSUM_A,
    )
    execution_manifest = SimpleNamespace(content_checksum=_CHECKSUM_B)
    prior_exposure = SimpleNamespace(
        preregistration_checksum=TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        screening_manifest=screening_manifest,
        screening_target_manifest=SimpleNamespace(content_checksum=_CHECKSUM_B),
        resource_calibration_checksum=resource_calibration.content_checksum,
        resource_calibration_execution_source_checksum=execution_manifest.content_checksum,
        pilot_plan=SimpleNamespace(content_checksum=resource_calibration.pilot_plan_checksum),
        screening_plan=SimpleNamespace(content_checksum=resource_calibration.screening_plan_checksum),
        pilot_custody_checksum=resource_calibration.pilot_custody_checksum,
        pilot_calibration_checksum=resource_calibration.pilot_calibration_checksum,
        screening_custody_checksum=resource_calibration.screening_custody_checksum,
    )
    decoded = {
        "final confirmation seal": seal,
        "final configuration execution manifest": object(),
        "execution-source manifest": execution_manifest,
        "analysis-source manifest": object(),
        "screening manifest": screening_manifest,
        "screening evidence": object(),
        "promotion decision": object(),
        "sample-size design": object(),
        "production resource calibration": resource_calibration,
        "repository binding catalog": object(),
        "prior-target exposure inventory": prior_exposure,
    }

    def fake_decode(_path: Path, name: str, _decoder: object) -> object:
        """Return typed-enough records up to the opaque authorization call."""
        return decoded[name]

    def reject_authorization(*_arguments: object) -> object:
        """Reject the final artifact universe before target access.

        Raises:
            ValueError: Always, to simulate an invalid promotion/calibration root.
        """
        msg = "promotion or calibration mismatch"
        raise ValueError(msg)

    def forbidden_target_load(_options: object) -> object:
        """Fail if the target is accessed before final authorization."""
        pytest.fail("confirmatory target was accessed before final authorization")

    monkeypatch.setattr(training_runner, "_decode_artifact", fake_decode)
    monkeypatch.setattr(training_runner, "verify_final_seal_source_lock", lambda *_arguments: None)
    monkeypatch.setattr(
        training_runner,
        "validate_final_configuration_execution_manifest",
        lambda *_arguments: None,
    )
    monkeypatch.setattr(training_runner, "authorize_confirmation", reject_authorization)
    monkeypatch.setattr(training_runner, "_load_targets", forbidden_target_load)

    with pytest.raises(TrainingRunnerConfigurationError, match="Final confirmation authorization"):
        training_runner.build_training_plan(options)


def test_phase_one_runner_registry_and_smoke_cardinality_are_unchanged() -> None:
    """Importing the separate WP22 runner leaves Phase I behavior untouched."""
    assert phase1_runner.RUNNER_CONFIGURATION_FORMAT == "yaqs.state_preparation.runner_config.v1"
    assert phase1_runner.PRESET_NAMES == ("smoke", "minimum", "full")
    phase1_options = phase1_runner.resolve_options(phase1_runner.parse_arguments([]))
    assert phase1_options.preset == "smoke"
    assert len(phase1_runner.build_benchmark_matrix(phase1_options)) == 12
