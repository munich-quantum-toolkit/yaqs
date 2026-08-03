# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused fail-closed tests for WP22D execution context and entropy custody."""

from __future__ import annotations

import json
import pickle  # noqa: S403 - verifies that secret-bearing state rejects pickle
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation import phase2
from benchmarks.state_preparation.phase2.binding_catalog import RepositoryBindingCatalog
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.execution_context import (
    AuthorizedTargetMaterialization,
    ExternalEntropyKeyring,
    TrainingExecutionContext,
    bind_training_plan_fingerprints,
    candidate_refs_from_bindings,
    parse_entropy_file_specs,
    schedules_from_bindings,
)
from benchmarks.state_preparation.phase2.implementation_catalog import RepositoryImplementationCatalog
from benchmarks.state_preparation.phase2.protocol import InitialPreregistration, load_initial_preregistration
from benchmarks.state_preparation.phase2.resumability import ExecutionSourceEntry, ResumabilityFingerprint
from benchmarks.state_preparation.phase2.source_lock import (
    ExecutionSourceFileRef,
    ExecutionSourceManifest,
)
from benchmarks.state_preparation.phase2.targets import (
    TargetPopulationConfig,
    TargetPopulationManifest,
    authorize_target_materialization,
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.training_orchestration import (
    TrainingRunPlan,
    build_training_smoke_plan,
)
from tests.benchmarks import test_state_preparation_wp22a_execution_bindings as wp22a_support

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch

_DEVELOPMENT_ENTROPY = bytes(range(32))
_SOURCE_COMMIT = "1" * 40


def _source_manifest() -> ExecutionSourceManifest:
    """Return a complete four-role source lock for structural preflight tests."""
    specs = (
        ("analysis_source", "analysis.py", "2" * 40, canonical_checksum({"source": "analysis"})),
        ("sealed_input", "preregistration.json", "3" * 40, canonical_checksum({"source": "input"})),
        ("execution_source", "runner.py", "4" * 40, canonical_checksum({"source": "runner"})),
        ("dependency_lock", "uv.lock", "5" * 40, canonical_checksum({"source": "lock"})),
    )
    source_files = tuple(
        ExecutionSourceFileRef(
            role=role,
            repo_path=path,
            git_blob_id=blob,
            source_checksum=checksum,
        )
        for role, path, blob, checksum in specs
    )
    return ExecutionSourceManifest(
        manifest_id="wp22d_test_sources",
        source_commit=_SOURCE_COMMIT,
        entry_point="runner.py",
        source_files=source_files,
        environment_lock_checksum=canonical_checksum({
            "dependency_locks": [source_files[-1].to_dict()],
        }),
        tracked_source_manifest_checksum=canonical_checksum({
            "source_files": [source.to_dict() for source in source_files],
        }),
    )


def _resumability_fingerprint(source_manifest: ExecutionSourceManifest) -> ResumabilityFingerprint:
    """Return one fingerprint whose byte roots are a subset of the source lock."""
    source_by_path = {source.repo_path: source for source in source_manifest.source_files}
    entries = tuple(
        ExecutionSourceEntry(
            role=role,
            repository_path=path,
            starting_git_blob_id=source_by_path[path].git_blob_id,
            content_checksum=source_by_path[path].source_checksum,
        )
        for role, path in (
            ("execution_source", "runner.py"),
            ("lockfile", "uv.lock"),
            ("sealed_input", "preregistration.json"),
        )
    )
    return ResumabilityFingerprint(
        starting_commit=_SOURCE_COMMIT,
        pipeline_prefix_id=f"phase2_pipeline_prefix_{'6' * 64}",
        dependency_versions={"mqt-yaqs": "test"},
        entries=entries,
    )


def _target_authority(
    preregistration: InitialPreregistration,
) -> tuple[TargetPopulationConfig, TargetPopulationManifest, AuthorizedTargetMaterialization]:
    """Return an externally custodied development config, manifest, and token."""
    config = build_target_population_config(
        preregistration,
        "development",
        role_master_entropy_commitment=role_master_entropy_commitment(_DEVELOPMENT_ENTROPY),
    )
    manifest = create_target_population_manifest(config, preregistration, _DEVELOPMENT_ENTROPY)
    authorization = authorize_target_materialization(
        preregistration,
        config,
        manifest,
        _DEVELOPMENT_ENTROPY,
    )
    return config, manifest, AuthorizedTargetMaterialization(config, manifest, authorization)


def _context() -> TrainingExecutionContext:
    """Build one exact ten-job smoke context from the WP22A/B test catalog.

    Returns:
        A structurally complete non-serializable execution context.
    """
    preregistration = load_initial_preregistration()
    bindings = wp22a_support._smoke_bindings()  # noqa: SLF001 - reuses the frozen WP22A test catalog
    profile = wp22a_support._profile(bindings)  # noqa: SLF001 - reuses the frozen WP22A profile fixture
    implementation_catalog = RepositoryImplementationCatalog.frozen(
        screening_outer_trajectory_count=256,
        smoke_evaluation_trajectory_count=2,
    )
    catalog = RepositoryBindingCatalog.from_profile(profile, implementation_catalog)
    config, manifest, authorized = _target_authority(preregistration)
    source_manifest = _source_manifest()
    fingerprint = _resumability_fingerprint(source_manifest)
    plan = build_training_smoke_plan(
        preregistration_checksum=preregistration.content_checksum,
        target_manifest=manifest,
        executable_bindings=catalog.bindings,
    )
    bound_plan = bind_training_plan_fingerprints(
        plan,
        execution_profile=profile,
        executable_bindings=catalog.bindings,
        target_configurations=(config,),
        target_manifests=(manifest,),
        execution_source_manifest=source_manifest,
        resumability_fingerprints=(fingerprint,),
        required_sample_size_design=None,
    )
    return TrainingExecutionContext(
        plan=bound_plan,
        execution_profile=profile,
        preregistration=preregistration,
        candidates=candidate_refs_from_bindings(catalog.bindings),
        schedules=schedules_from_bindings(catalog.bindings),
        scoped_bindings=catalog.bindings,
        target_configurations=(config,),
        target_manifests=(manifest,),
        authorized_materializations=(authorized,),
        screening_manifest=None,
        screening_cells=(),
        required_sample_size_design=None,
        execution_source_manifest=source_manifest,
        resumability_fingerprints=(fingerprint,),
        external_entropy_keyring=ExternalEntropyKeyring({
            ("development", "primary_q6"): _DEVELOPMENT_ENTROPY,
        }),
    )


def test_external_entropy_is_exact_nonserializable_and_redacted(tmp_path: Path) -> None:
    """Opaque entropy accepts exact files but cannot leak through ordinary serializers."""
    raw_path = tmp_path / "development.key"
    raw_path.write_bytes(_DEVELOPMENT_ENTROPY)
    keyring = ExternalEntropyKeyring.from_files({("development", "primary_q6"): raw_path})

    assert keyring.entropy_for("development", "primary_q6") == _DEVELOPMENT_ENTROPY
    assert repr(keyring) == "ExternalEntropyKeyring(<redacted>)"
    assert _DEVELOPMENT_ENTROPY.hex() not in repr(keyring)
    with pytest.raises(TypeError, match="non-serializable"):
        pickle.dumps(keyring)
    with pytest.raises(TypeError):
        json.dumps(keyring)

    linked_path = tmp_path / "linked.key"
    linked_path.symlink_to(raw_path)
    with pytest.raises(ValueError, match="unavailable or unsafe"):
        ExternalEntropyKeyring.from_files({("development", "primary_q6"): linked_path})
    with pytest.raises(ValueError, match="reuse"):
        ExternalEntropyKeyring({
            ("development", "primary_q6"): _DEVELOPMENT_ENTROPY,
            ("screening_selection", "primary_q6"): _DEVELOPMENT_ENTROPY,
        })


def test_entropy_specs_are_cli_only_exact_and_nonprobing(tmp_path: Path) -> None:
    """CLI references parse exact slots without reading or requiring their paths."""
    missing = tmp_path / "does-not-exist.key"
    specs = parse_entropy_file_specs((f"development/primary_q6={missing}",))
    assert specs == {("development", "primary_q6"): missing}
    assert not missing.exists()
    with pytest.raises(ValueError, match="only once"):
        parse_entropy_file_specs((
            f"development/primary_q6={missing}",
            f"development/primary_q6={tmp_path / 'other.key'}",
        ))


def test_bound_context_roundtrips_plan_and_preflights_without_mutation(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """All ten programs and job fingerprints validate before output creation."""
    context = _context()
    output = tmp_path / "prospective-output"
    monkeypatch.setattr(
        "benchmarks.state_preparation.phase2.execution_context.verify_execution_source_manifest",
        lambda _manifest, _root: ("runner.py",),
    )

    report = context.preflight(tmp_path, output)
    assert report.job_count == 10
    assert report.compiled_program_count == 10
    assert report.target_population_count == 1
    assert {job.family_id for job in context.plan.jobs} == {"tfim_ground_state"}
    assert len({job.target_instance_id for job in context.plan.jobs}) == 1
    assert not output.exists()
    assert TrainingRunPlan.from_json(context.plan.to_json()) == context.plan
    assert all(
        job.execution_profile_checksum == context.execution_profile.content_checksum for job in context.plan.jobs
    )
    assert all(job.scheduled_execution_program_checksum is not None for job in context.plan.jobs)
    with pytest.raises(TypeError, match="non-serializable"):
        pickle.dumps(context)
    assert _DEVELOPMENT_ENTROPY.hex() not in repr(context)
    assert _DEVELOPMENT_ENTROPY.hex() not in context.plan.to_json()


def test_preflight_rejects_source_schedule_and_entropy_before_output(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Late whole-context failures leave the prospective output root absent."""
    context = _context()
    output = tmp_path / "must-remain-absent"

    def reject_source(_manifest: ExecutionSourceManifest, _root: Path) -> tuple[str, ...]:
        msg = "sealed source byte mismatch"
        raise ValueError(msg)

    monkeypatch.setattr(
        "benchmarks.state_preparation.phase2.execution_context.verify_execution_source_manifest",
        reject_source,
    )
    with pytest.raises(ValueError, match="source byte mismatch"):
        context.preflight(tmp_path, output)
    assert not output.exists()

    monkeypatch.setattr(
        "benchmarks.state_preparation.phase2.execution_context.verify_execution_source_manifest",
        lambda _manifest, _root: ("runner.py",),
    )
    wrong_program = canonical_checksum({"wrong": "program"})
    changed_job = replace(
        context.plan.jobs[0],
        scheduled_execution_program_checksum=wrong_program,
    )
    changed_jobs = tuple(sorted((changed_job, *context.plan.jobs[1:]), key=lambda job: job.sort_key))
    changed_plan = replace(
        context.plan,
        jobs=changed_jobs,
        scheduled_execution_program_checksums=tuple(
            sorted({cast("str", job.scheduled_execution_program_checksum) for job in changed_jobs})
        ),
    )
    changed_context = replace(context, plan=changed_plan)
    with pytest.raises(ValueError, match="fingerprint"):
        changed_context.preflight(tmp_path, output)
    assert not output.exists()

    wrong_entropy = b"z" * 32
    wrong_context = replace(
        context,
        external_entropy_keyring=ExternalEntropyKeyring({
            ("development", "primary_q6"): wrong_entropy,
        }),
    )
    with pytest.raises(ValueError, match="commitment") as error:
        wrong_context.preflight(tmp_path, output)
    assert wrong_entropy.hex() not in str(error.value)
    assert not output.exists()


def test_wp22d_public_package_exports_are_available() -> None:
    """The Phase II package exposes the context, plan, and source-lock records."""
    assert phase2.TrainingExecutionContext is TrainingExecutionContext
    assert phase2.TrainingRunPlan is TrainingRunPlan
    assert phase2.ExecutionSourceManifest is ExecutionSourceManifest
    assert phase2.WP22CandidateConfiguration.__module__.endswith("screening_design")
