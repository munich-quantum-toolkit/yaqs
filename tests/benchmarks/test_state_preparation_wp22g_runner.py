# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Pre-reveal runner-hardening tests for prospective WP22G confirmation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from benchmarks.state_preparation import training_runner
from benchmarks.state_preparation.phase2.canonical import canonical_checksum, canonical_json
from benchmarks.state_preparation.phase2.confirmatory_study_store import LockedConfirmatoryStudySnapshotRef
from benchmarks.state_preparation.phase2.training_orchestration import TrainingRunSummary
from benchmarks.state_preparation.training_runner import (
    TRAINING_RUNNER_CONFIGURATION_FORMAT,
    TrainingRunnerConfigurationError,
)


def _install_held_input_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Install sentinels for every held-input entry point used by the runner.

    Returns:
        The mutable list recording any forbidden access.
    """
    accesses: list[str] = []

    def forbidden_preregistration(*_arguments: object) -> object:
        accesses.append("preregistration")
        pytest.fail("paper-confirm opened the preregistration before its early guard")

    def forbidden_entropy(*_arguments: object) -> object:
        accesses.append("entropy")
        pytest.fail("paper-confirm opened external entropy before its early guard")

    def forbidden_targets(*_arguments: object) -> object:
        accesses.append("target")
        pytest.fail("paper-confirm opened the held target before its early guard")

    monkeypatch.setattr(training_runner, "_load_preregistration", forbidden_preregistration)
    monkeypatch.setattr(training_runner.ExternalEntropyKeyring, "from_files", forbidden_entropy)
    monkeypatch.setattr(training_runner, "_load_targets", forbidden_targets)
    return accesses


@pytest.mark.parametrize(
    ("arguments", "option_name"),
    [
        (("--method", "layerwise_bmpd_crn_v2"), "method_id"),
        (("--pipeline", "pipeline.json"), "pipeline_path"),
        (("--stage-depth", "4"), "stage_depths"),
        (("--stage-budget", "100"), "stage_budgets"),
        (("--training-noise-id", "dephasing_1s_all"), "training_noise_id"),
        (("--training-noise-strength", "1"), "training_noise_strength"),
        (("--trajectory-update", "cross"), "trajectory_update"),
        (("--sampling-policy", "crn_fixed"), "sampling_policy"),
        (("--training-trajectories", "32"), "training_trajectory_count"),
        (("--validation-trajectories", "16"), "checkpoint_validation_trajectory_count"),
        (("--crn-refresh-interval", "5"), "crn_refresh_interval"),
        (("--checkpoint-rule", "best_validation_fidelity"), "checkpoint_rule"),
        (("--data-role", "development"), "data_role"),
        (("--native-two-qubit-cap-per-edge", "12"), "native_two_qubit_cap_per_edge"),
        (("--normalized-compute-cap", "250"), "normalized_compute_cap"),
        (("--overwrite",), "overwrite"),
        (("--no-fail-fast",), "fail_fast"),
        (("--executor-factory", "rogue.module:factory"), "executor_factory"),
        (("--candidate", "candidate.json"), "candidate_paths"),
        (("--schedule", "schedule.json"), "schedule_paths"),
        (("--execution-profile", "profile.json"), "execution_profile_path"),
        (("--resumability-fingerprint", "resume.json"), "resumability_fingerprint_paths"),
        (("--pilot-optimization-seed", "17"), "pilot_optimization_seeds"),
    ],
)
def test_paper_confirm_rejects_scientific_options_before_held_input_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
    option_name: str,
) -> None:
    """Every caller-selectable scientific or redundant control fails early."""
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--output",
            str(tmp_path / "confirm"),
            *arguments,
        ])
    )
    accesses = _install_held_input_sentinels(monkeypatch)

    with pytest.raises(TrainingRunnerConfigurationError, match=option_name):
        training_runner.build_training_plan(options)
    assert accesses == []


def test_paper_confirm_rejects_explicit_false_scientific_config_before_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JSON knob is forbidden even when its value equals an operational default."""
    configuration = tmp_path / "paper-confirm.json"
    configuration.write_text(
        json.dumps({
            "format": TRAINING_RUNNER_CONFIGURATION_FORMAT,
            "preset": "paper-confirm",
            "fail_fast": False,
        }),
        encoding="utf-8",
    )
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--config",
            str(configuration),
            "--execute-expensive",
            "--output",
            str(tmp_path / "confirm"),
        ])
    )
    accesses = _install_held_input_sentinels(monkeypatch)

    with pytest.raises(TrainingRunnerConfigurationError, match="fail_fast"):
        training_runner.build_training_plan(options)
    assert accesses == []


@pytest.mark.parametrize("entry_point", ["build_training_plan", "build_confirmation_execution_context", "run"])
def test_paper_confirm_requires_explicit_cli_output_before_held_reads(
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    """No repository-relative preset default can authorize confirmation output."""
    options = training_runner.resolve_options(
        training_runner.parse_arguments(["--preset", "paper-confirm", "--execute-expensive"])
    )
    accesses = _install_held_input_sentinels(monkeypatch)

    with pytest.raises(TrainingRunnerConfigurationError, match="explicit CLI --output"):
        getattr(training_runner, entry_point)(options)
    assert accesses == []


def test_paper_confirm_does_not_accept_config_only_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reveal-bearing invocation itself must name its dedicated output root."""
    configuration = tmp_path / "paper-confirm.json"
    configuration.write_text(
        json.dumps({
            "format": TRAINING_RUNNER_CONFIGURATION_FORMAT,
            "preset": "paper-confirm",
            "output_dir": str(tmp_path / "confirm"),
        }),
        encoding="utf-8",
    )
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--config",
            str(configuration),
            "--execute-expensive",
        ])
    )
    accesses = _install_held_input_sentinels(monkeypatch)

    with pytest.raises(TrainingRunnerConfigurationError, match="explicit CLI --output"):
        training_runner.build_training_plan(options)
    assert accesses == []


def test_paper_confirm_rejects_repository_owned_output_before_held_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirmation output cannot alter the source tree whose bytes were sealed."""
    repository = tmp_path / "repository"
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--repository-root",
            str(repository),
            "--output",
            str(repository / "confirm"),
        ])
    )
    accesses = _install_held_input_sentinels(monkeypatch)

    with pytest.raises(TrainingRunnerConfigurationError, match="outside repository_root"):
        training_runner.build_training_plan(options)
    assert accesses == []


def test_paper_confirm_rejects_output_that_contains_repository_before_held_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broad output ancestor cannot place locks or artifacts around the checkout."""
    repository = tmp_path / "repository"
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--repository-root",
            str(repository),
            "--output",
            str(tmp_path),
        ])
    )
    accesses = _install_held_input_sentinels(monkeypatch)

    with pytest.raises(TrainingRunnerConfigurationError, match="disjoint"):
        training_runner.build_training_plan(options)
    assert accesses == []


@pytest.mark.parametrize(
    "unsafe_kind",
    ["mixed-role", "roles-file", "confirmatory-file", "roles-symlink", "output-symlink"],
)
def test_paper_confirm_rejects_mixed_or_unsafe_output_roles_before_held_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    """Existing output role layouts fail closed before any scientific artifact opens."""
    output = tmp_path / "confirm"
    roles = output / "roles"
    if unsafe_kind == "output-symlink":
        target = tmp_path / "foreign-output"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    else:
        output.mkdir()
    if unsafe_kind == "mixed-role":
        (roles / "development").mkdir(parents=True)
    elif unsafe_kind == "roles-file":
        roles.write_text("not a directory", encoding="utf-8")
    elif unsafe_kind == "confirmatory-file":
        roles.mkdir()
        (roles / "confirmatory").write_text("not a directory", encoding="utf-8")
    elif unsafe_kind == "roles-symlink":
        target = tmp_path / "foreign-roles"
        target.mkdir()
        roles.symlink_to(target, target_is_directory=True)
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--output",
            str(output),
        ])
    )
    accesses = _install_held_input_sentinels(monkeypatch)

    with pytest.raises(TrainingRunnerConfigurationError, match=r"share|roles|confirmatory|symlink"):
        training_runner.build_training_plan(options)
    assert accesses == []


def test_paper_confirm_accepts_exposure_inventory_as_custody_artifact(tmp_path: Path) -> None:
    """The reserved exposure-inventory path is allowed through the early option gate."""
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--output",
            str(tmp_path / "confirm"),
            "--target-exposure-inventory",
            "custody/exposure-inventory.json",
        ])
    )

    assert options.prior_target_exposure_inventory_path == Path("custody/exposure-inventory.json")
    training_runner._preflight_paper_confirm_request(  # noqa: SLF001 - focused early-gate regression
        options
    )


def test_paper_confirm_normalizes_exposure_inventory_config_alias(tmp_path: Path) -> None:
    """A supported JSON custody alias is recorded under its allowed canonical name."""
    configuration = tmp_path / "paper-confirm.json"
    configuration.write_text(
        json.dumps({
            "format": TRAINING_RUNNER_CONFIGURATION_FORMAT,
            "preset": "paper-confirm",
            "prior_target_exposure_inventory": "custody/exposure-inventory.json",
        }),
        encoding="utf-8",
    )
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--config",
            str(configuration),
            "--execute-expensive",
            "--output",
            str(tmp_path / "confirm"),
        ])
    )

    assert options.explicit_option_names >= {
        "preset",
        "prior_target_exposure_inventory_path",
    }
    training_runner._preflight_paper_confirm_request(  # noqa: SLF001 - focused alias regression
        options
    )


def test_paper_confirm_requires_writable_off_tree_staging_before_held_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crash-safe publication prerequisites fail before scientific inputs open."""
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--output",
            str(tmp_path / "confirm"),
        ])
    )
    accesses = _install_held_input_sentinels(monkeypatch)
    monkeypatch.setattr(training_runner.os, "access", lambda *_arguments: False)

    with pytest.raises(TrainingRunnerConfigurationError, match=r"staging|writable"):
        training_runner.build_training_plan(options)
    assert accesses == []


def test_paper_confirm_resume_requires_external_head_before_held_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume cannot select the highest mutable on-disk snapshot as authority."""
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--resume",
            "--output",
            str(tmp_path / "confirm"),
        ])
    )
    accesses = _install_held_input_sentinels(monkeypatch)

    with pytest.raises(TrainingRunnerConfigurationError, match=r"expected-locked-study-head|external"):
        training_runner.build_training_plan(options)
    assert accesses == []


def test_fresh_confirmation_reserves_absent_external_head_custody(tmp_path: Path) -> None:
    """The first invocation requires a new external path and never overwrites it."""
    output = tmp_path / "confirm"
    missing_options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--output",
            str(output),
        ])
    )
    with pytest.raises(TrainingRunnerConfigurationError, match=r"expected-locked-study-head|external custody"):
        training_runner._load_expected_locked_study_head(missing_options)  # noqa: SLF001

    retained = tmp_path / "confirmation-study-head.json"
    retained.write_text("{}\n", encoding="utf-8")
    existing_options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--expected-locked-study-head",
            str(retained),
            "--output",
            str(output),
        ])
    )
    with pytest.raises(TrainingRunnerConfigurationError, match=r"Fresh|overwrite"):
        training_runner._load_expected_locked_study_head(existing_options)  # noqa: SLF001


def test_runner_loads_retained_head_from_prior_cli_output(tmp_path: Path) -> None:
    """The exact sealed nested reference printed by the CLI is resume-ready."""
    reference = LockedConfirmatoryStudySnapshotRef(
        relative_path=f"confirmation_study/snapshot_{0:08d}_{'1' * 64}.json",
        ordinal=0,
        file_checksum="sha256:" + "2" * 64,
        snapshot_content_checksum="sha256:" + "1" * 64,
    )
    retained = tmp_path / "retained-head.json"
    retained.write_text(
        training_runner._render_result(  # noqa: SLF001 - exact CLI handoff regression
            TrainingRunSummary(
                planned=576,
                attempted=1,
                succeeded=0,
                failed=1,
                skipped=0,
                locked_study_snapshot_path=reference.relative_path,
                locked_study_snapshot_ordinal=reference.ordinal,
                locked_study_snapshot_file_checksum=reference.file_checksum,
                locked_study_snapshot_content_checksum=reference.snapshot_content_checksum,
                locked_study_snapshot_reference_checksum=reference.content_checksum,
            )
        ),
        encoding="utf-8",
    )
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--resume",
            "--expected-locked-study-head",
            str(retained),
            "--output",
            str(tmp_path / "confirm"),
        ])
    )

    assert training_runner._load_expected_locked_study_head(options) == reference  # noqa: SLF001


def test_retained_resume_head_must_be_outside_mutable_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An on-tree head copy cannot authenticate deletion of that same tree."""
    output = tmp_path / "confirm"
    output.mkdir()
    retained = output / "head.json"
    retained.write_text("{}\n", encoding="utf-8")
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "paper-confirm",
            "--execute-expensive",
            "--resume",
            "--expected-locked-study-head",
            str(retained),
            "--output",
            str(output),
        ])
    )
    accesses = _install_held_input_sentinels(monkeypatch)

    with pytest.raises(TrainingRunnerConfigurationError, match=r"outside.*output|output.*custody"):
        training_runner.build_training_plan(options)
    assert accesses == []


def _session_document(output_root: Path) -> dict[str, object]:
    """Return one checksum-sealed public session marker fixture."""
    content: dict[str, object] = {
        "schema_version": "yaqs.state_preparation.phase2.confirmation_plan_session.v1",
        "plan_checksum": "sha256:" + "1" * 64,
        "final_confirmation_seal_checksum": "sha256:" + "2" * 64,
        "execution_source_manifest_checksum": "sha256:" + "3" * 64,
        "analysis_source_manifest_checksum": "sha256:" + "4" * 64,
        "prior_target_exposure_inventory_checksum": "sha256:" + "5" * 64,
        "authorized_output_root": str(output_root),
        "locked_study_head_custody_path": str(output_root.parent / "confirmation-study-head.json"),
        "job_count": 576,
    }
    return {**content, "content_checksum": canonical_checksum(content)}


def test_static_session_header_requires_canonical_single_link_bytes(tmp_path: Path) -> None:
    """Pre-reveal session inspection authenticates checksum, bytes, and link count."""
    output = tmp_path / "confirm"
    output.mkdir()
    marker = output / ".wp22-confirmation-session.json"
    document = _session_document(output)
    marker.write_text(f"{canonical_json(document)}\n", encoding="utf-8")

    assert training_runner._read_confirmation_session_header(marker) == document  # noqa: SLF001
    alias = tmp_path / "marker-alias.json"
    os.link(marker, alias)
    with pytest.raises(ValueError, match=r"single-link|linked"):
        training_runner._read_confirmation_session_header(marker)  # noqa: SLF001
    alias.unlink()
    changed = dict(document)
    changed["job_count"] = 575
    marker.write_text(f"{canonical_json(changed)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"checksum|sealed"):
        training_runner._read_confirmation_session_header(marker)  # noqa: SLF001


def test_target_novelty_failure_precedes_external_entropy_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused held identifier or seed fails before its entropy is consumed."""
    entropy_reads: list[object] = []

    class RejectingInventory:
        @staticmethod
        def validate_confirmatory_novelty(_manifest: object) -> None:
            msg = "reused target seed"
            raise ValueError(msg)

    def record_entropy(*values: object) -> object:
        entropy_reads.append(values)
        pytest.fail("entropy was read after novelty already failed")

    monkeypatch.setattr(training_runner.ExternalEntropyKeyring, "from_files", record_entropy)
    with pytest.raises(ValueError, match="reused target seed"):
        training_runner._authorize_revealed_confirmatory_target(  # noqa: SLF001
            preregistration=cast("Any", object()),
            target_configuration=cast("Any", object()),
            target_manifest=cast("Any", object()),
            entropy_files={},
            confirmation_authorization=cast("Any", object()),
            exposure_inventory=cast("Any", RejectingInventory()),
        )
    assert entropy_reads == []
