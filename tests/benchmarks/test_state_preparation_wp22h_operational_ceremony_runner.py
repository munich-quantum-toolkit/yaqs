# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused custody and CLI tests for the path-oriented WP22H runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import pytest

from benchmarks.state_preparation import training_runner
from benchmarks.state_preparation.phase2 import operational_ceremony_runner as runner
from benchmarks.state_preparation.phase2 import targets as phase2_targets
from benchmarks.state_preparation.phase2.binding_catalog import ExecutableScopedBinding, RepositoryBindingCatalog
from benchmarks.state_preparation.phase2.canonical import canonical_checksum, canonical_json
from benchmarks.state_preparation.phase2.ceremony_store import (
    CeremonyBundleMember,
    ReopenedCeremonyBundle,
    read_ceremony_bundle_member,
)
from benchmarks.state_preparation.phase2.execution_bindings import (
    ScopedImplementationBinding,
    TrainingExecutionProfile,
)
from benchmarks.state_preparation.phase2.execution_registry import (
    build_paper_pilot_execution_registry,
    build_paper_screen_execution_registry,
)
from benchmarks.state_preparation.phase2.protocol import AnalysisSourceManifest, InitialPreregistration
from benchmarks.state_preparation.phase2.resumability import ResumabilityFingerprint
from benchmarks.state_preparation.phase2.source_lock import ExecutionSourceManifest
from benchmarks.state_preparation.phase2.targets import (
    TargetPopulationConfig,
    TargetPopulationManifest,
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.training_orchestration import TrainingRunPlan
from benchmarks.state_preparation.phase2.training_schedules import TrainingStrategySchedule
from tests.benchmarks.test_state_preparation_wp22h_execution_registry import (
    _pilot_calibration,
    _sample_size_design,
)
from tests.benchmarks.wp22_confirmation_test_support import _create_source_repository

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class _JSONArtifact(Protocol):
    """Structural type for a canonical ceremony test artifact."""

    def to_json(self) -> str:
        """Return canonical JSON."""


_ArtifactT = TypeVar("_ArtifactT", bound=_JSONArtifact)


def _checksum(label: str) -> str:
    """Return one stable valid checksum."""
    return canonical_checksum({"wp22h_runner_test": label})


def _option_paths(tmp_path: Path) -> dict[str, Path]:
    """Return absolute placeholder paths for option-schema tests."""
    return {
        "repository_root": (tmp_path / "repository").absolute(),
        "ceremony_root": (tmp_path / "ceremony").absolute(),
        "pilot_primary_target_config_path": (tmp_path / "pilot-primary-config.json").absolute(),
        "pilot_primary_target_manifest_path": (tmp_path / "pilot-primary-manifest.json").absolute(),
        "pilot_secondary_target_config_path": (tmp_path / "pilot-secondary-config.json").absolute(),
        "pilot_secondary_target_manifest_path": (tmp_path / "pilot-secondary-manifest.json").absolute(),
        "pilot_primary_entropy_path": (tmp_path / "pilot-primary.key").absolute(),
        "pilot_secondary_entropy_path": (tmp_path / "pilot-secondary.key").absolute(),
    }


def _prepare_options(tmp_path: Path) -> runner.PreparePilotOptions:
    """Return one type-valid stage-zero option object."""
    return runner.PreparePilotOptions(**_option_paths(tmp_path))


def _members_for(stage: runner.WP22HCeremonyStage) -> tuple[CeremonyBundleMember, ...]:
    """Return generic bytes with the runner's exact fixed path/role shape."""
    shape = runner._STAGE_SPEC[stage][2]  # noqa: SLF001 -- fixed inventory contract
    return tuple(CeremonyBundleMember(path, role, f"{path}:{role}\n".encode()) for path, role in sorted(shape))


def _with_artifacts(
    members: tuple[CeremonyBundleMember, ...],
    artifacts: dict[str, _JSONArtifact],
) -> tuple[CeremonyBundleMember, ...]:
    """Replace selected generic members with canonical typed artifacts.

    Returns:
        The exact stage inventory with typed replacement payloads.
    """
    by_path = {member.relative_path: member for member in members}
    for path, artifact in artifacts.items():
        original = by_path[path]
        by_path[path] = CeremonyBundleMember(path, original.role, f"{artifact.to_json()}\n".encode())
    return tuple(by_path[path] for path in sorted(by_path))


def _decode_member(
    bundle: ReopenedCeremonyBundle,
    relative_path: str,
    loader: Callable[[str], _ArtifactT],
) -> _ArtifactT:
    """Decode one store-authenticated member with its strict public loader.

    Returns:
        The strict typed member artifact.
    """
    return loader(read_ceremony_bundle_member(bundle, relative_path).decode())


def _memoize_string_property(
    monkeypatch: pytest.MonkeyPatch,
    owner: type[object],
    name: str,
) -> None:
    """Memoize one immutable checksum property for bounded integration tests."""
    descriptor = vars(owner)[name]
    assert isinstance(descriptor, property)
    assert descriptor.fget is not None
    getter = cast("Callable[[object], str]", descriptor.fget)
    cache: dict[int, tuple[object, str]] = {}

    def cached(instance: object) -> str:
        """Return the once-derived immutable property value."""
        key = id(instance)
        stored = cache.get(key)
        if stored is None or stored[0] is not instance:
            stored = (instance, getter(instance))
            cache[key] = stored
        return stored[1]

    monkeypatch.setattr(owner, name, property(cached))


def test_operational_schema_roundtrips_and_handoff_has_no_held_path(tmp_path: Path) -> None:
    """Path, run-custody, and handoff schemas are strict and canonical."""
    repository = (tmp_path / "repository").absolute()
    ceremony = (tmp_path / "ceremony").absolute()
    paths = runner.WP22HOperationalPaths(repository, ceremony, (tmp_path / "pilot").absolute())
    assert runner.WP22HOperationalPaths.from_json(paths.to_json()) == paths

    stage_receipt = runner.WP22HStageRunReceipt(
        stage=runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
        bundle_directory=(ceremony / "01-close-pilot-prepare-screen").absolute(),
        stage_manifest_checksum=_checksum("stage manifest"),
        bundle_index_checksum=_checksum("bundle index"),
        predecessor_stage_manifest_checksum=_checksum("predecessor"),
    )
    assert runner.WP22HStageRunReceipt.from_json(stage_receipt.to_json()) == stage_receipt
    with pytest.raises(ValueError, match="null predecessor"):
        replace(stage_receipt, predecessor_stage_manifest_checksum=None)
    with pytest.raises(ValueError, match="exact canonical directory"):
        replace(stage_receipt, bundle_directory=ceremony / "foreign-stage")
    prepare_receipt = runner.WP22HStageRunReceipt(
        stage=runner.WP22HCeremonyStage.PREPARE_PILOT,
        bundle_directory=ceremony / "00-prepare-pilot",
        stage_manifest_checksum=_checksum("prepare manifest"),
        bundle_index_checksum=_checksum("prepare index"),
        predecessor_stage_manifest_checksum=None,
    )
    with pytest.raises(ValueError, match="null predecessor"):
        replace(prepare_receipt, predecessor_stage_manifest_checksum=_checksum("forbidden parent"))

    checksums = {
        role: _checksum(role)
        for role in runner._HANDOFF_ARTIFACT_PATHS  # noqa: SLF001 -- fixed handoff contract
    }
    handoff = runner.WP22HPaperConfirmHandoff(checksums)
    assert runner.WP22HPaperConfirmHandoff.from_json(handoff.to_json()) == handoff
    assert "held" not in handoff.to_json()
    assert "screening_evidence" in handoff.artifact_checksums

    changed = handoff.to_dict()
    assert isinstance(changed["artifacts"], dict)
    artifacts = dict(changed["artifacts"])
    assert isinstance(artifacts["screening_evidence"], dict)
    screening = dict(artifacts["screening_evidence"])
    screening["relative_path"] = "foreign.json"
    artifacts["screening_evidence"] = screening
    changed["artifacts"] = artifacts
    changed["content_checksum"] = canonical_checksum({
        key: value for key, value in changed.items() if key != "content_checksum"
    })
    with pytest.raises(ValueError, match="changed path"):
        runner.WP22HPaperConfirmHandoff.from_json(canonical_json(changed))


def test_execution_profiles_are_custodied_but_not_confirmatory_overrides() -> None:
    """Pilot and screen bundles expose profiles while paper-confirm keeps the catalog only."""
    prepare_shape = runner._STAGE_SPEC[runner.WP22HCeremonyStage.PREPARE_PILOT][2]  # noqa: SLF001
    pilot_shape = runner._STAGE_SPEC[runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN][2]  # noqa: SLF001

    assert ("pilot/execution_profile.json", "pilot-execution-profile") in prepare_shape
    assert ("screen/execution_profile.json", "screen-execution-profile") in pilot_shape
    assert "execution_profile" not in runner._HANDOFF_ARTIFACT_PATHS  # noqa: SLF001


def test_external_artifact_reader_requires_canonical_single_link_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External ceremony artifacts require canonical bytes and one stable inode."""
    artifact = runner.WP22HOperationalPaths(
        (tmp_path / "repository").absolute(),
        (tmp_path / "ceremony").absolute(),
    )
    artifact_path = tmp_path / "operational-paths.json"
    artifact_path.write_text(f"{artifact.to_json()}\n")
    opened_flags: list[int] = []
    original_open = runner.os.open

    def recording_open(path: Path, flags: int) -> int:
        """Record the nofollow/nonblocking policy and open the real fixture.

        Returns:
            The real read-only file descriptor.
        """
        opened_flags.append(flags)
        return original_open(path, flags)

    monkeypatch.setattr(runner.os, "open", recording_open)

    assert (
        runner._load_external_artifact(  # noqa: SLF001 -- secure reader regression
            artifact_path,
            runner.WP22HOperationalPaths.from_json,
            "operational paths",
        )
        == artifact
    )
    nofollow = getattr(runner.os, "O_NOFOLLOW", 0)
    nonblocking = getattr(runner.os, "O_NONBLOCK", 0)
    assert not nofollow or opened_flags[-1] & nofollow
    assert not nonblocking or opened_flags[-1] & nonblocking

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(artifact.to_json())
    with pytest.raises(ValueError, match="canonical newline encoding"):
        runner._load_external_artifact(  # noqa: SLF001 -- secure reader regression
            noncanonical,
            runner.WP22HOperationalPaths.from_json,
            "operational paths",
        )

    linked = tmp_path / "linked.json"
    linked.symlink_to(artifact_path)
    with pytest.raises(ValueError, match="single-link regular file"):
        runner._load_external_artifact(  # noqa: SLF001 -- secure reader regression
            linked,
            runner.WP22HOperationalPaths.from_json,
            "operational paths",
        )

    rewrite_path = tmp_path / "same-size-rewrite.json"
    original_payload = f"{artifact.to_json()}\n".encode()
    rewrite_path.write_bytes(original_payload)
    before_rewrite = rewrite_path.stat()

    def rewrite_after_open(path: Path, flags: int) -> int:
        """Rewrite the opened inode in place before descriptor validation.

        Returns:
            The descriptor that now observes the same-size rewrite.
        """
        descriptor = original_open(path, flags)
        if path == rewrite_path:
            rewrite_path.write_bytes(b"x" * len(original_payload))
            runner.os.utime(  # force a distinct timestamp even on coarse filesystems
                rewrite_path,
                ns=(before_rewrite.st_atime_ns, before_rewrite.st_mtime_ns + 1_000_000_000),
            )
        return descriptor

    monkeypatch.setattr(runner.os, "open", rewrite_after_open)
    with pytest.raises(ValueError, match="changed while it was opened"):
        runner._load_external_artifact(  # noqa: SLF001 -- in-place rewrite regression
            rewrite_path,
            runner.WP22HOperationalPaths.from_json,
            "operational paths",
        )

    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(artifact_path)
    with pytest.raises(ValueError, match="single-link regular file"):
        runner._load_external_artifact(  # noqa: SLF001 -- secure reader regression
            artifact_path,
            runner.WP22HOperationalPaths.from_json,
            "operational paths",
        )


def test_stage_zero_and_one_bundle_required_runner_artifacts_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stored authorities reopen and the public runner reconstructs the pilot plan."""

    def cheap_tfim_parameters(
        _master: bytes,
        _target_instance_id: str,
        stratum_id: str,
        qubit_count: int,
    ) -> dict[str, object]:
        """Return shape-valid metadata without dense target diagonalization."""
        ratio = {"ferromagnetic": 0.5, "critical": 1.0, "paramagnetic": 1.5}[stratum_id]
        return {
            "attempt_index": 0,
            "couplings": [1.0] * (qubit_count - 1),
            "fields": [ratio] * qubit_count,
            "ground_energy": -float(qubit_count),
            "ground_state_gap": 1.0,
            "gap_threshold": 1e-10 * float(qubit_count),
            "spectral_norm": float(qubit_count),
        }

    monkeypatch.setattr(phase2_targets, "_tfim_parameter_record", cheap_tfim_parameters)
    for owner, name in (
        (ScopedImplementationBinding, "implementation_checksum"),
        (ScopedImplementationBinding, "content_checksum"),
        (ExecutableScopedBinding, "content_checksum"),
        (TrainingExecutionProfile, "content_checksum"),
        (TrainingStrategySchedule, "content_checksum"),
    ):
        _memoize_string_property(monkeypatch, owner, name)
    repository = (tmp_path / "source").absolute()
    execution_source, analysis_source = _create_source_repository(repository)
    ceremony = (tmp_path / "ceremony").absolute()
    ceremony.mkdir()
    preregistration = InitialPreregistration.from_json(
        (Path(runner.__file__).parent / "data/initial_preregistration_v1.json").read_text()
    )

    primary_entropy = bytes(range(32))
    secondary_entropy = bytes(range(32, 64))
    primary_config = build_target_population_config(
        preregistration,
        "development",
        role_master_entropy_commitment=role_master_entropy_commitment(primary_entropy),
    )
    secondary_config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(secondary_entropy),
        population_scope="secondary_q12",
    )
    primary_targets = create_target_population_manifest(primary_config, preregistration, primary_entropy)
    secondary_targets = create_target_population_manifest(secondary_config, preregistration, secondary_entropy)
    _, pilot_catalog = build_paper_pilot_execution_registry(preregistration)
    pilot_fingerprint = runner.build_ceremony_resumability_fingerprint(execution_source, pilot_catalog)
    primary_entropy_path = (tmp_path / "primary.key").absolute()
    secondary_entropy_path = (tmp_path / "secondary.key").absolute()
    for path, entropy in ((primary_entropy_path, primary_entropy), (secondary_entropy_path, secondary_entropy)):
        path.write_bytes(entropy)
        path.chmod(0o600)
    pilot_keyring = runner.ExternalEntropyKeyring.from_files({
        (primary_config.data_role, primary_config.population_scope): primary_entropy_path,
        (secondary_config.data_role, secondary_config.population_scope): secondary_entropy_path,
    })
    pilot_context = runner.build_ceremony_training_context(
        preregistration=preregistration,
        catalog=pilot_catalog,
        execution_source_manifest=execution_source,
        target_configurations=(primary_config, secondary_config),
        target_manifests=(primary_targets, secondary_targets),
        external_entropy_keyring=pilot_keyring,
        resumability_fingerprint=pilot_fingerprint,
    )
    prepare_paths = runner.WP22HOperationalPaths(repository, ceremony)
    prepare_members = _with_artifacts(
        _members_for(runner.WP22HCeremonyStage.PREPARE_PILOT),
        {
            "operational/paths.json": prepare_paths,
            "source/preregistration.json": preregistration,
            "source/execution_source_manifest.json": execution_source,
            "source/analysis_source_manifest.json": analysis_source,
            "pilot/execution_catalog.json": pilot_catalog,
            "pilot/execution_profile.json": pilot_catalog.profile,
            "pilot/primary_target_config.json": primary_config,
            "pilot/primary_target_manifest.json": primary_targets,
            "pilot/secondary_target_config.json": secondary_config,
            "pilot/secondary_target_manifest.json": secondary_targets,
            "pilot/resumability_fingerprint.json": pilot_fingerprint,
            "pilot/training_plan.json": pilot_context.plan,
        },
    )
    prepare_receipt = runner._publish_stage(  # noqa: SLF001 -- exact stage integration
        ceremony,
        runner.WP22HCeremonyStage.PREPARE_PILOT,
        prepare_members,
        None,
    )
    (prepare_bundle,) = runner._reopen_chain(  # noqa: SLF001 -- exact stage integration
        ceremony,
        runner.WP22HCeremonyStage.PREPARE_PILOT,
        prepare_receipt.bundle_index_checksum,
    )

    screen_entropy = bytes(reversed(range(32)))
    screen_config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(screen_entropy),
    )
    screen_targets = create_target_population_manifest(screen_config, preregistration, screen_entropy)
    sample_size_design = _sample_size_design()
    pilot_calibration = _pilot_calibration()
    screen_candidates, screen_catalog = build_paper_screen_execution_registry(
        preregistration,
        sample_size_design,
        pilot_calibration,
    )
    screen_fingerprint = runner.build_ceremony_resumability_fingerprint(execution_source, screen_catalog)
    screening_manifest = runner.build_screening_manifest(
        preregistration,
        screen_targets,
        screen_candidates,
        optimization_seeds=runner.derive_screening_optimization_seeds(preregistration),
        screening_seed_root=runner.derive_screening_seed_root(
            preregistration,
            screen_catalog.profile,
            screen_targets,
        ),
    )
    screen_entropy_path = (tmp_path / "screen.key").absolute()
    screen_entropy_path.write_bytes(screen_entropy)
    screen_entropy_path.chmod(0o600)
    screen_keyring = runner.ExternalEntropyKeyring.from_files({
        (screen_config.data_role, screen_config.population_scope): screen_entropy_path,
    })
    screen_context = runner.build_ceremony_training_context(
        preregistration=preregistration,
        catalog=screen_catalog,
        execution_source_manifest=execution_source,
        target_configurations=(screen_config,),
        target_manifests=(screen_targets,),
        external_entropy_keyring=screen_keyring,
        resumability_fingerprint=screen_fingerprint,
        screening_manifest=screening_manifest,
        sample_size_design=sample_size_design,
    )
    pilot_paths = runner.WP22HOperationalPaths(repository, ceremony, (tmp_path / "pilot-output").absolute())
    pilot_members = _with_artifacts(
        _members_for(runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN),
        {
            "operational/paths.json": pilot_paths,
            "screen/target_config.json": screen_config,
            "screen/target_manifest.json": screen_targets,
            "screen/execution_catalog.json": screen_catalog,
            "screen/execution_profile.json": screen_catalog.profile,
            "screen/resumability_fingerprint.json": screen_fingerprint,
            "screen/screening_manifest.json": screening_manifest,
            "screen/training_plan.json": screen_context.plan,
            "pilot/sample_size_design.json": sample_size_design,
            "pilot/compute_calibration.json": pilot_calibration,
        },
    )
    pilot_receipt = runner._publish_stage(  # noqa: SLF001 -- exact stage integration
        ceremony,
        runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
        pilot_members,
        prepare_bundle.manifest,
    )
    _, pilot_bundle = runner._reopen_chain(  # noqa: SLF001 -- exact stage integration
        ceremony,
        runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
        pilot_receipt.bundle_index_checksum,
    )

    reopened_execution = _decode_member(
        prepare_bundle,
        "source/execution_source_manifest.json",
        ExecutionSourceManifest.from_json,
    )
    reopened_analysis = _decode_member(
        prepare_bundle,
        "source/analysis_source_manifest.json",
        AnalysisSourceManifest.from_json,
    )
    reopened_pilot_catalog = _decode_member(
        prepare_bundle,
        "pilot/execution_catalog.json",
        RepositoryBindingCatalog.from_json,
    )
    reopened_pilot_profile = _decode_member(
        prepare_bundle,
        "pilot/execution_profile.json",
        TrainingExecutionProfile.from_json,
    )
    reopened_primary_config = _decode_member(
        prepare_bundle,
        "pilot/primary_target_config.json",
        TargetPopulationConfig.from_json,
    )
    reopened_primary_targets = _decode_member(
        prepare_bundle,
        "pilot/primary_target_manifest.json",
        TargetPopulationManifest.from_json,
    )
    reopened_pilot_fingerprint = _decode_member(
        prepare_bundle,
        "pilot/resumability_fingerprint.json",
        ResumabilityFingerprint.from_json,
    )
    reopened_pilot_plan = _decode_member(
        prepare_bundle,
        "pilot/training_plan.json",
        TrainingRunPlan.from_json,
    )
    reopened_screen_catalog = _decode_member(
        pilot_bundle,
        "screen/execution_catalog.json",
        RepositoryBindingCatalog.from_json,
    )
    reopened_screen_profile = _decode_member(
        pilot_bundle,
        "screen/execution_profile.json",
        TrainingExecutionProfile.from_json,
    )
    reopened_screen_config = _decode_member(
        pilot_bundle,
        "screen/target_config.json",
        TargetPopulationConfig.from_json,
    )
    reopened_screen_targets = _decode_member(
        pilot_bundle,
        "screen/target_manifest.json",
        TargetPopulationManifest.from_json,
    )
    reopened_screen_fingerprint = _decode_member(
        pilot_bundle,
        "screen/resumability_fingerprint.json",
        ResumabilityFingerprint.from_json,
    )
    reopened_screen_plan = _decode_member(
        pilot_bundle,
        "screen/training_plan.json",
        TrainingRunPlan.from_json,
    )

    assert reopened_execution == execution_source
    assert reopened_analysis == analysis_source
    assert reopened_pilot_profile == reopened_pilot_catalog.profile
    assert reopened_pilot_profile.preset == "paper-pilot"
    assert reopened_primary_config == primary_config
    assert reopened_primary_targets == primary_targets
    assert reopened_pilot_fingerprint == pilot_fingerprint
    assert reopened_pilot_plan == pilot_context.plan
    assert reopened_screen_profile == reopened_screen_catalog.profile
    assert reopened_screen_profile.preset == "paper-screen"
    assert reopened_screen_config == screen_config
    assert reopened_screen_targets == screen_targets
    assert reopened_screen_fingerprint == screen_fingerprint
    assert reopened_screen_plan == screen_context.plan

    pilot_directory = prepare_bundle.bundle_directory
    argv = (
        "--preset",
        "paper-pilot",
        "--preregistration",
        str(pilot_directory / "source/preregistration.json"),
        "--preregistration-checksum",
        preregistration.content_checksum,
        "--execution-source-manifest",
        str(pilot_directory / "source/execution_source_manifest.json"),
        "--execution-profile",
        str(pilot_directory / "pilot/execution_profile.json"),
        "--binding-catalog",
        str(pilot_directory / "pilot/execution_catalog.json"),
        "--target-configuration",
        str(pilot_directory / "pilot/primary_target_config.json"),
        "--target-configuration",
        str(pilot_directory / "pilot/secondary_target_config.json"),
        "--target-manifest",
        str(pilot_directory / "pilot/primary_target_manifest.json"),
        "--target-manifest",
        str(pilot_directory / "pilot/secondary_target_manifest.json"),
        "--resumability-fingerprint",
        str(pilot_directory / "pilot/resumability_fingerprint.json"),
        "--external-entropy-file",
        f"development/primary_q6={primary_entropy_path}",
        "--external-entropy-file",
        f"screening_selection/secondary_q12={secondary_entropy_path}",
        "--repository-root",
        str(repository),
        "--output",
        str((tmp_path / "unexecuted-pilot-output").absolute()),
    )
    loaded_context = training_runner.build_training_execution_context(
        training_runner.resolve_options(training_runner.parse_arguments(argv))
    )
    assert loaded_context.plan == reopened_pilot_plan

    screen_directory = pilot_bundle.bundle_directory
    screen_argv = (
        "--preset",
        "paper-screen",
        "--preregistration",
        str(pilot_directory / "source/preregistration.json"),
        "--preregistration-checksum",
        preregistration.content_checksum,
        "--execution-source-manifest",
        str(pilot_directory / "source/execution_source_manifest.json"),
        "--execution-profile",
        str(screen_directory / "screen/execution_profile.json"),
        "--binding-catalog",
        str(screen_directory / "screen/execution_catalog.json"),
        "--target-configuration",
        str(screen_directory / "screen/target_config.json"),
        "--target-manifest",
        str(screen_directory / "screen/target_manifest.json"),
        "--screening-manifest",
        str(screen_directory / "screen/screening_manifest.json"),
        "--sample-size-design",
        str(screen_directory / "pilot/sample_size_design.json"),
        "--resumability-fingerprint",
        str(screen_directory / "screen/resumability_fingerprint.json"),
        "--external-entropy-file",
        f"screening_selection/primary_q6={screen_entropy_path}",
        "--repository-root",
        str(repository),
        "--output",
        str((tmp_path / "unexecuted-screen-output").absolute()),
    )
    loaded_screen_context = training_runner.build_training_execution_context(
        training_runner.resolve_options(training_runner.parse_arguments(screen_argv))
    )
    assert loaded_screen_context.plan == reopened_screen_plan


def test_operational_paths_reject_nested_pilot_and_screen_outputs(tmp_path: Path) -> None:
    """Stage-two custody cannot place either production root inside the other."""
    repository = (tmp_path / "repository").absolute()
    ceremony = (tmp_path / "ceremony").absolute()
    pilot = (tmp_path / "production" / "pilot").absolute()

    with pytest.raises(ValueError, match="pairwise disjoint"):
        runner.WP22HOperationalPaths(
            repository,
            ceremony,
            pilot_output_root=pilot,
            screen_output_root=pilot / "screen",
        )
    with pytest.raises(ValueError, match="pairwise disjoint"):
        runner.WP22HOperationalPaths(
            repository,
            ceremony,
            pilot_output_root=pilot / "nested",
            screen_output_root=pilot,
        )


def test_option_schemas_reject_non_path_and_normalize_predecessor(tmp_path: Path) -> None:
    """Programmatic options cannot carry loosely typed paths or malformed custody."""
    values = _option_paths(tmp_path)
    values["pilot_primary_entropy_path"] = cast("Path", "secret.key")
    with pytest.raises(TypeError, match="pilot_primary_entropy_path"):
        runner.PreparePilotOptions(**values)

    options = runner.VerifyReadyOptions(
        repository_root=(tmp_path / "repository").absolute(),
        ceremony_root=(tmp_path / "ceremony").absolute(),
        expected_predecessor_index_checksum=_checksum("head"),
        pilot_primary_entropy_path=(tmp_path / "primary.key").absolute(),
        pilot_secondary_entropy_path=(tmp_path / "secondary.key").absolute(),
        screening_entropy_path=(tmp_path / "screen.key").absolute(),
    )
    with pytest.raises(ValueError, match="checksum"):
        replace(options, expected_predecessor_index_checksum="changed")


def test_fixed_store_chain_requires_external_terminal_index_custody(tmp_path: Path) -> None:
    """Successors use reopened manifests and reject a retained-head mismatch."""
    ceremony = (tmp_path / "ceremony").absolute()
    ceremony.mkdir()
    first = runner._publish_stage(  # noqa: SLF001 -- focused custody integration
        ceremony,
        runner.WP22HCeremonyStage.PREPARE_PILOT,
        _members_for(runner.WP22HCeremonyStage.PREPARE_PILOT),
        None,
    )
    (reopened_first,) = runner._reopen_chain(  # noqa: SLF001 -- focused custody integration
        ceremony,
        runner.WP22HCeremonyStage.PREPARE_PILOT,
        first.bundle_index_checksum,
    )
    second = runner._publish_stage(  # noqa: SLF001 -- focused custody integration
        ceremony,
        runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
        _members_for(runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN),
        reopened_first.manifest,
    )
    chain = runner._reopen_chain(  # noqa: SLF001 -- focused custody integration
        ceremony,
        runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
        second.bundle_index_checksum,
    )

    assert tuple(item.manifest.stage_ordinal for item in chain) == (0, 1)
    assert chain[1].manifest.predecessor_stage_manifest_checksum == chain[0].manifest.content_checksum
    with pytest.raises(ValueError, match="expected terminal custody"):
        runner._reopen_chain(  # noqa: SLF001 -- retained-head regression
            ceremony,
            runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
            _checksum("rollback or foreign index"),
        )


def test_whole_ceremony_lock_rejects_foreign_root_before_stage_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreign root state fails before any source, entropy, or custody seam."""
    options = _prepare_options(tmp_path)
    options.repository_root.mkdir()
    options.ceremony_root.mkdir()
    (options.ceremony_root / "foreign.json").write_text("{}")
    called = False

    def forbidden(_: runner.PreparePilotOptions) -> runner.WP22HStageRunReceipt:
        """Record forbidden stage entry if root validation is bypassed.

        Raises:
            AssertionError: Always, because this seam must remain unreachable.
        """
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(runner, "_prepare_pilot_locked", forbidden)
    with pytest.raises(ValueError, match="foreign, future, or special"):
        runner.prepare_pilot(options)

    assert not called
    assert not any("wp22h-operational.lock" in item.name for item in options.ceremony_root.iterdir())


def test_source_validation_precedes_target_and_entropy_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dirty source root fails before registry, target, or entropy access."""
    repository = (tmp_path / "repository").absolute()
    ceremony = (tmp_path / "ceremony").absolute()
    paths = runner.WP22HOperationalPaths(repository, ceremony)
    member_reads: list[str] = []
    entropy_opened = False

    def load_member(
        _bundle: object,
        relative_path: str,
        _loader: object,
    ) -> object:
        """Return only the roots needed to reach source verification."""
        member_reads.append(relative_path)
        if relative_path == "operational/paths.json":
            return paths
        return object()

    def dirty_source(*_args: object, **_kwargs: object) -> tuple[str, ...]:
        """Model source drift detected at the first governed verification.

        Raises:
            ValueError: Always, to stop before target or entropy access.
        """
        msg = "dirty source sentinel"
        raise ValueError(msg)

    def forbidden_entropy(*_args: object, **_kwargs: object) -> object:
        """Record any forbidden external entropy access.

        Returns:
            An unreachable sentinel object.
        """
        nonlocal entropy_opened
        entropy_opened = True
        return object()

    monkeypatch.setattr(runner, "_load_bundle_artifact", load_member)
    monkeypatch.setattr(runner, "verify_governed_execution_source_manifest", dirty_source)
    monkeypatch.setattr(runner.ExternalEntropyKeyring, "from_files", forbidden_entropy)
    with pytest.raises(ValueError, match="dirty source sentinel"):
        runner._load_pilot_context(  # noqa: SLF001 -- source-order regression
            cast("ReopenedCeremonyBundle", object()),
            repository,
            ceremony,
            tmp_path / "primary.key",
            tmp_path / "secondary.key",
        )

    assert "pilot/execution_catalog.json" not in member_reads
    assert not entropy_opened


def test_close_screen_rejects_nested_output_before_screen_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage two rejects overlapping production roots before screen reopening."""
    repository = (tmp_path / "repository").absolute()
    ceremony = (tmp_path / "ceremony").absolute()
    repository.mkdir()
    ceremony.mkdir()
    pilot_output = (tmp_path / "production" / "pilot").absolute()
    screen_output = pilot_output / "screen"
    screen_opened = False
    options = runner.CloseScreenSealOptions(
        repository_root=repository,
        ceremony_root=ceremony,
        expected_predecessor_index_checksum=_checksum("pilot stage index"),
        screen_output_root=screen_output,
        pilot_primary_entropy_path=tmp_path / "primary.key",
        pilot_secondary_entropy_path=tmp_path / "secondary.key",
        screening_entropy_path=tmp_path / "screen.key",
        confirmatory_target_config_path=tmp_path / "confirm-config.json",
        confirmatory_target_commitment_path=tmp_path / "confirm-commitment.json",
    )
    pilot_state = SimpleNamespace(preregistration=object())
    pilot_closure = SimpleNamespace(sample_size_design=object())
    screen_state = SimpleNamespace(
        paths=runner.WP22HOperationalPaths(repository, ceremony, pilot_output),
        context=object(),
    )

    def reopen_chain(*_args: object, **_kwargs: object) -> tuple[object, object]:
        """Return bounded predecessor sentinels."""
        return object(), object()

    def load_public(*_args: object, **_kwargs: object) -> object:
        """Return one bounded public-artifact sentinel."""
        return object()

    def rebuild(*_args: object, **_kwargs: object) -> tuple[object, object, object]:
        """Return bounded pre-screen context sentinels."""
        return pilot_state, pilot_closure, screen_state

    def validate_public(*_args: object, **_kwargs: object) -> None:
        """Accept the bounded public-artifact sentinels."""

    def nested_output(*_args: object, **_kwargs: object) -> Path:
        """Return the unsafe screen root after ordinary path validation."""
        return screen_output

    def forbidden_screen(*_args: object, **_kwargs: object) -> object:
        """Record forbidden production screening custody access.

        Returns:
            An unreachable sentinel object.
        """
        nonlocal screen_opened
        screen_opened = True
        return object()

    monkeypatch.setattr(runner, "_reopen_chain", reopen_chain)
    monkeypatch.setattr(runner, "_load_external_artifact", load_public)
    monkeypatch.setattr(runner, "_rebuild_through_screen_context", rebuild)
    monkeypatch.setattr(runner, "_validate_public_confirmation", validate_public)
    monkeypatch.setattr(runner, "_validate_output_root", nested_output)
    monkeypatch.setattr(runner, "close_production_screen", forbidden_screen)
    with pytest.raises(ValueError, match="pairwise disjoint"):
        runner._close_screen_seal_locked(options)  # noqa: SLF001 -- pre-custody ordering regression

    assert not screen_opened


@pytest.mark.parametrize(
    ("stage", "arguments", "option_type"),
    [
        (
            runner.WP22HCeremonyStage.PREPARE_PILOT,
            (
                "--pilot-primary-target-config",
                "/inputs/pilot-primary-config.json",
                "--pilot-primary-target-manifest",
                "/inputs/pilot-primary-manifest.json",
                "--pilot-secondary-target-config",
                "/inputs/pilot-secondary-config.json",
                "--pilot-secondary-target-manifest",
                "/inputs/pilot-secondary-manifest.json",
                "--pilot-primary-entropy",
                "/secrets/pilot-primary.key",
                "--pilot-secondary-entropy",
                "/secrets/pilot-secondary.key",
            ),
            runner.PreparePilotOptions,
        ),
        (
            runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
            (
                "--expected-predecessor-index-checksum",
                _checksum("prepare index"),
                "--pilot-output-root",
                "/outputs/pilot",
                "--pilot-primary-entropy",
                "/secrets/pilot-primary.key",
                "--pilot-secondary-entropy",
                "/secrets/pilot-secondary.key",
                "--screening-target-config",
                "/inputs/screen-config.json",
                "--screening-target-manifest",
                "/inputs/screen-manifest.json",
                "--screening-entropy",
                "/secrets/screen.key",
            ),
            runner.ClosePilotPrepareScreenOptions,
        ),
        (
            runner.WP22HCeremonyStage.CLOSE_SCREEN_SEAL,
            (
                "--expected-predecessor-index-checksum",
                _checksum("pilot index"),
                "--screen-output-root",
                "/outputs/screen",
                "--pilot-primary-entropy",
                "/secrets/pilot-primary.key",
                "--pilot-secondary-entropy",
                "/secrets/pilot-secondary.key",
                "--screening-entropy",
                "/secrets/screen.key",
                "--confirmatory-target-config",
                "/public/confirm-config.json",
                "--confirmatory-target-commitment",
                "/public/confirm-commitment.json",
            ),
            runner.CloseScreenSealOptions,
        ),
        (
            runner.WP22HCeremonyStage.VERIFY_READY,
            (
                "--expected-predecessor-index-checksum",
                _checksum("screen index"),
                "--pilot-primary-entropy",
                "/secrets/pilot-primary.key",
                "--pilot-secondary-entropy",
                "/secrets/pilot-secondary.key",
                "--screening-entropy",
                "/secrets/screen.key",
            ),
            runner.VerifyReadyOptions,
        ),
    ],
)
def test_cli_dispatches_every_typed_stage_and_prints_custody(
    stage: runner.WP22HCeremonyStage,
    arguments: Sequence[str],
    option_type: type[runner.CeremonyOptions],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every CLI stage dispatches once and emits the new retained index head."""
    seen: list[runner.CeremonyOptions] = []
    expected = runner.WP22HStageRunReceipt(
        stage=stage,
        bundle_directory=Path("/ceremony") / runner._STAGE_DIRECTORY[stage],  # noqa: SLF001
        stage_manifest_checksum=_checksum(f"{stage.value} manifest"),
        bundle_index_checksum=_checksum(f"{stage.value} index"),
        predecessor_stage_manifest_checksum=(
            None if stage is runner.WP22HCeremonyStage.PREPARE_PILOT else _checksum("parent")
        ),
    )

    def dispatch(options: runner.CeremonyOptions) -> runner.WP22HStageRunReceipt:
        """Record the exact typed CLI options and return fixed custody.

        Returns:
            The fixed stage receipt.
        """
        seen.append(options)
        return expected

    monkeypatch.setattr(runner, "run_operational_ceremony", dispatch)
    argv = (
        stage.value,
        "--repository-root",
        "/repository",
        "--ceremony-root",
        "/ceremony",
        *arguments,
    )

    assert runner.main(argv) == 0
    assert len(seen) == 1
    assert isinstance(seen[0], option_type)
    assert capsys.readouterr().out == f"{expected.to_json()}\n"


def test_cli_has_no_held_or_scientific_override_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown held/scientific flags fail in parsing before stage dispatch."""
    called = False

    def forbidden(_: runner.CeremonyOptions) -> runner.WP22HStageRunReceipt:
        """Record any forbidden dispatch after parser rejection.

        Raises:
            AssertionError: Always, because this seam must remain unreachable.
        """
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(runner, "run_operational_ceremony", forbidden)
    parser_help = runner.build_argument_parser().format_help()
    assert set(runner.WP22HCeremonyStage) == {
        runner.WP22HCeremonyStage.PREPARE_PILOT,
        runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
        runner.WP22HCeremonyStage.CLOSE_SCREEN_SEAL,
        runner.WP22HCeremonyStage.VERIFY_READY,
    }
    assert "held-target" not in parser_help
    assert "confirmatory-entropy" not in parser_help
    with pytest.raises(SystemExit):
        runner.main((
            "verify-ready",
            "--repository-root",
            "/repository",
            "--ceremony-root",
            "/ceremony",
            "--expected-predecessor-index-checksum",
            _checksum("screen"),
            "--pilot-primary-entropy",
            "/secrets/primary",
            "--pilot-secondary-entropy",
            "/secrets/secondary",
            "--screening-entropy",
            "/secrets/screen",
            "--held-target-manifest",
            "/forbidden/held.json",
        ))
    assert not called
