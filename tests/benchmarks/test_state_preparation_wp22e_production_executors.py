# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Acceptance tests for WP22E production executors and result custody."""

# The adversarial custody tests deliberately exercise private verification
# helpers, and the clean-checkout test invokes fixed, resolved executables.
# ruff: noqa: S603, SLF001

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation import phase2, training_runner
from benchmarks.state_preparation.phase2 import execution_context as execution_context_module
from benchmarks.state_preparation.phase2 import production_executors as production_executor_module
from benchmarks.state_preparation.phase2.binding_catalog import RepositoryBindingCatalog
from benchmarks.state_preparation.phase2.canonical import canonical_checksum, canonical_json
from benchmarks.state_preparation.phase2.execution_bindings import SCREEN_METHOD_IDS, SMOKE_METHOD_IDS, Preset
from benchmarks.state_preparation.phase2.execution_context import (
    TrainingExecutionContext,
    bind_training_plan_fingerprints,
)
from benchmarks.state_preparation.phase2.implementation_catalog import RepositoryImplementationCatalog
from benchmarks.state_preparation.phase2.layerwise_bmpd import create_bmpd_circuit_binding
from benchmarks.state_preparation.phase2.production_executors import (
    PersistedProductionAttemptError,
    PilotDiagnosticEvidence,
    ProductionAttemptStore,
    ProductionTrainingExecutor,
    ReopenedProductionResult,
    ResultArtifactRef,
    SyntheticConfirmationExecutor,
    SyntheticConfirmationFixture,
    create_default_training_executor_registry,
    derive_result_artifact_ref,
    reopen_result_artifact,
)
from benchmarks.state_preparation.phase2.scheduled_execution import (
    KrotovScheduledUpdateAdapter,
    OperatorGrowthSegmentedSnapshot,
    ScheduledExecutionProgram,
    ScheduledJobSeedSet,
    ScheduledTrainingGradientResult,
    ScheduledValidationResult,
    compile_frozen_schedule_trace,
    execute_operator_growth_segmented_program,
    execute_scheduled_program,
)
from benchmarks.state_preparation.phase2.source_lock import ExecutionSourceFileRef, ExecutionSourceManifest
from benchmarks.state_preparation.phase2.training_orchestration import (
    ConfirmExecutionRequest,
    JobExecutionControls,
    TrainingJob,
    TrainingRunSummary,
    TrainingScheduleResumeState,
    load_training_job_outcome_history,
)
from mqt.yaqs.optimization import KrotovFixedMapEnsemble, KrotovNoiseMap
from tests.benchmarks import test_state_preparation_wp22a_execution_bindings as wp22a_support
from tests.benchmarks import test_state_preparation_wp22c_scheduled_execution as wp22c_support
from tests.benchmarks.test_state_preparation_wp22_execution_context import _context

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import NoReturn

    from _pytest.tmpdir import TempPathFactory

    from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineConfig


@dataclass(frozen=True)
class _SmokeResult:
    """One real repository smoke execution and its reopened artifacts."""

    job: TrainingJob
    directory: Path
    reference: ResultArtifactRef
    reopened: ReopenedProductionResult


@dataclass(frozen=True)
class _SmokeSuite:
    """Shared context, executor, runner summary, and ten immutable results."""

    context: TrainingExecutionContext
    executor: ProductionTrainingExecutor
    summary: TrainingRunSummary
    results: tuple[_SmokeResult, ...]


_FRESH_CONTROLS = JobExecutionControls(resume=False, overwrite=False)
_SKIP = cast("Callable[[str], NoReturn]", pytest.skip)


def _checksum(label: str) -> str:
    """Return a stable checksum for one synthetic identity.

    Returns:
        A canonical prefixed checksum.
    """
    return canonical_checksum({"wp22e_test_identity": label})


def _reseal_mapping(value: dict[str, object]) -> dict[str, object]:
    """Recompute one canonical top-level content checksum after an adversarial edit.

    Returns:
        The detached checksum-sealed mapping.
    """
    payload = {key: item for key, item in value.items() if key != "content_checksum"}
    return {**payload, "content_checksum": canonical_checksum(payload)}


def _replace_nested_value(value: object, old: object, new: object) -> object:
    """Replace one exact nested JSON-native value without changing unrelated aliases.

    Returns:
        The recursively detached value with exact matches replaced.
    """
    if value == old:
        return new
    if isinstance(value, dict):
        return {key: _replace_nested_value(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_nested_value(item, old, new) for item in value]
    return value


def _reseal_attempt_member(
    job_directory: Path,
    reference: ResultArtifactRef,
    member_path: str,
    changed_document: dict[str, object],
) -> ResultArtifactRef:
    """Reseal one nested member, evidence aliases, manifest refs, and the result address.

    Returns:
        The newly derived result reference for the fully resealed attempt.
    """
    manifest_path = job_directory / reference.manifest_path
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    blobs = cast("list[dict[str, object]]", manifest["blobs"])
    old_member_ref = next(item for item in blobs if item["path"] == member_path)
    member_payload = f"{canonical_json(changed_document)}\n".encode()
    (job_directory / member_path).write_bytes(member_payload)
    new_member_ref = _reseal_mapping({
        **old_member_ref,
        "byte_count": len(member_payload),
        "file_checksum": f"sha256:{hashlib.sha256(member_payload).hexdigest()}",
        "logical_checksum": changed_document["content_checksum"],
    })

    old_evidence_ref = cast("dict[str, object]", manifest["evidence_ref"])
    evidence_path = job_directory / cast("str", old_evidence_ref["path"])
    evidence = cast("dict[str, object]", json.loads(evidence_path.read_text(encoding="utf-8")))
    replaced = cast("dict[str, object]", _replace_nested_value(evidence, old_member_ref, new_member_ref))
    changed_evidence = _reseal_mapping(replaced)
    evidence_payload = f"{canonical_json(changed_evidence)}\n".encode()
    evidence_path.write_bytes(evidence_payload)
    new_evidence_ref = _reseal_mapping({
        **old_evidence_ref,
        "byte_count": len(evidence_payload),
        "file_checksum": f"sha256:{hashlib.sha256(evidence_payload).hexdigest()}",
        "logical_checksum": changed_evidence["content_checksum"],
    })
    manifest["blobs"] = [
        new_member_ref if item == old_member_ref else new_evidence_ref if item == old_evidence_ref else item
        for item in blobs
    ]
    manifest["evidence_ref"] = new_evidence_ref
    changed_manifest = _reseal_mapping(manifest)
    manifest_payload = f"{canonical_json(changed_manifest)}\n".encode()
    manifest_path.write_bytes(manifest_payload)
    return ResultArtifactRef(
        job_checksum=reference.job_checksum,
        attempt=reference.attempt,
        artifact_kind=reference.artifact_kind,
        status=reference.status,
        execution_source_manifest_checksum=reference.execution_source_manifest_checksum,
        source_fingerprint_checksum=reference.source_fingerprint_checksum,
        manifest_path=reference.manifest_path,
        manifest_file_checksum=f"sha256:{hashlib.sha256(manifest_payload).hexdigest()}",
        manifest_content_checksum=cast("str", changed_manifest["content_checksum"]),
        evidence_checksum=cast("str", new_evidence_ref["logical_checksum"]),
    )


def _confirm_request() -> ConfirmExecutionRequest:
    """Return one complete synthetic-only confirmatory execution request.

    Returns:
        A schema-valid request that does not reveal or address held target bytes.
    """
    return ConfirmExecutionRequest(
        final_confirmation_seal_checksum=_checksum("seal"),
        preregistration_checksum=_checksum("preregistration"),
        promotion_decision_checksum=_checksum("promotion"),
        execution_source_checksum=_checksum("execution source"),
        analysis_source_manifest_checksum=_checksum("analysis source"),
        analysis_template_checksum=_checksum("analysis template"),
        configuration_execution_manifest_checksum=_checksum("configuration execution manifest"),
        hyperparameters_checksum=_checksum("hyperparameters"),
        implementation_checksum=_checksum("implementation"),
        scoped_binding_checksum=_checksum("scoped binding"),
        executable_binding_checksum=_checksum("executable binding"),
        sample_size_design_checksum=_checksum("sample size"),
        failure_policy_checksum=_checksum("failure policy"),
        fixed_test_trajectory_count=2,
        primary_noise_condition={
            "noise_id": "depolarizing_1s_all",
            "definition_version": "fixed_rate_noise_v1",
            "strength_scale": 1.0,
            "tjm_dt": 0.1,
            "training_placement": "after_native_gates",
            "test_placement": "after_native_gates",
        },
        primary_resource_budget={
            "metric": "native_two_qubit_gates_per_chain_edge",
            "cap_per_chain_edge": 12.0,
            "normalized_compute_cap": 1_000.0,
            "reachable_stratum_manifest_checksum": _checksum("reachable resources"),
        },
        method_id="synthetic_confirm_method",
        configuration_checksum=_checksum("configuration"),
        target_manifest_checksum=_checksum("held manifest commitment"),
        target_instance_id="confirm_target_0001",
        target_spec_checksum=_checksum("held target commitment"),
        family_id="tfim_ground_state",
        stratum_id="critical",
        qubit_count=6,
        optimization_block_id="confirm_target_0001_seed_0",
        optimization_seed_index=0,
        optimization_seed=101,
        evaluation_seed=202,
    )


def _context_with_production_source() -> TrainingExecutionContext:
    """Add WP22E's default factory to the synthetic source-lock fixture.

    Returns:
        The rebound ten-job context whose source inventory includes WP22E.
    """
    context = _context()
    production_path = "benchmarks/state_preparation/phase2/production_executors.py"
    production_source = ExecutionSourceFileRef(
        role="execution_source",
        repo_path=production_path,
        git_blob_id="6" * 40,
        source_checksum=canonical_checksum({"source": production_path}),
    )
    source_files = tuple(
        sorted(
            (*context.execution_source_manifest.source_files, production_source),
            key=lambda source: source.repo_path,
        )
    )
    manifest = ExecutionSourceManifest(
        manifest_id="wp22e_test_sources",
        source_commit=context.execution_source_manifest.source_commit,
        entry_point=context.execution_source_manifest.entry_point,
        source_files=source_files,
        environment_lock_checksum=canonical_checksum({
            "dependency_locks": [source.to_dict() for source in source_files if source.role == "dependency_lock"],
        }),
        tracked_source_manifest_checksum=canonical_checksum({
            "source_files": [source.to_dict() for source in source_files],
        }),
    )
    plan = bind_training_plan_fingerprints(
        context.plan,
        execution_profile=context.execution_profile,
        executable_bindings=context.scoped_bindings,
        target_configurations=context.target_configurations,
        target_manifests=context.target_manifests,
        execution_source_manifest=manifest,
        resumability_fingerprints=context.resumability_fingerprints,
        required_sample_size_design=None,
    )
    return TrainingExecutionContext(
        plan=plan,
        execution_profile=context.execution_profile,
        preregistration=context.preregistration,
        candidates=context.candidates,
        schedules=context.schedules,
        scoped_bindings=context.scoped_bindings,
        target_configurations=context.target_configurations,
        target_manifests=context.target_manifests,
        authorized_materializations=context.authorized_materializations,
        screening_manifest=None,
        screening_cells=(),
        required_sample_size_design=None,
        execution_source_manifest=manifest,
        resumability_fingerprints=context.resumability_fingerprints,
        external_entropy_keyring=context.external_entropy_keyring,
    )


@pytest.fixture(scope="module")
def production_smoke_suite(tmp_path_factory: TempPathFactory) -> _SmokeSuite:
    """Execute all ten families through the repository default runner factory.

    Returns:
        One shared default-runner summary and its ten reopened typed results.
    """
    context = _context_with_production_source()
    root = tmp_path_factory.mktemp("wp22e-production-smoke")
    options = training_runner.resolve_options(
        training_runner.parse_arguments([
            "--preset",
            "training-smoke",
            "--output",
            str(root),
        ])
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        verified_paths = tuple(source.repo_path for source in context.execution_source_manifest.source_files)
        monkeypatch.setattr(
            training_runner,
            "verify_execution_source_manifest",
            lambda *_arguments: verified_paths,
        )
        monkeypatch.setattr(
            execution_context_module,
            "verify_execution_source_manifest",
            lambda *_arguments: verified_paths,
        )
        run_result = training_runner.run(options, context=context, executor=None)
    assert isinstance(run_result, TrainingRunSummary)
    executor = ProductionTrainingExecutor(context)
    results: list[_SmokeResult] = []
    for job in context.plan.jobs:
        directory = root / job.output_path
        outcomes = load_training_job_outcome_history(directory, job)
        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.result_artifact_checksum is not None
        reference = derive_result_artifact_ref(
            directory,
            job.content_checksum,
            1,
            expected_reference_checksum=outcome.result_artifact_checksum,
        )
        results.append(
            _SmokeResult(
                job=job,
                directory=directory,
                reference=reference,
                reopened=reopen_result_artifact(reference, directory),
            )
        )
    return _SmokeSuite(context, executor, run_result, tuple(results))


def test_all_ten_smoke_families_execute_through_typed_custody(
    production_smoke_suite: _SmokeSuite,
) -> None:
    """The default factory produces ten typed, self-verifying terminal attempts."""
    results = production_smoke_suite.results
    assert production_smoke_suite.summary == TrainingRunSummary(10, 10, 10, 0, 0)
    assert len(results) == 10
    assert {result.job.method_id for result in results} == set(SMOKE_METHOD_IDS)
    assert all(result.reopened.evidence.status == "success" for result in results)
    assert all(result.reopened.resources for result in results)
    pipelines = tuple(result for result in results if result.job.implementation_kind == "phase2_pipeline")
    operators = tuple(result for result in results if result.job.implementation_kind == "operator_growth")
    assert len(pipelines) == 8
    assert len(operators) == 2
    assert all(result.reopened.raw_trajectory is not None for result in pipelines)
    assert all(result.reopened.raw_trajectory is None for result in operators)
    assert all(result.reopened.evidence.structural_prefix_checksums for result in operators)
    assert all(result.reopened.evidence.derived_metrics["promotion_eligible"] is False for result in results)


def test_default_runner_uses_real_source_lock_in_a_clean_checkout(tmp_path: Path) -> None:
    """A clean temporary Git checkout reaches the default registry through the real verifier."""
    repository_root = Path(production_executor_module.__file__).resolve().parents[3]
    checkout = tmp_path / "clean-checkout"
    checkout.mkdir()
    git = shutil.which("git")
    if git is None:
        _SKIP("Git is required for the source-custody acceptance test.")

    listed = subprocess.run(
        (git, "-C", str(repository_root), "ls-files", "-z"),
        check=True,
        capture_output=True,
    )
    for raw_path in listed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(os.fsdecode(raw_path))
        source = repository_root / relative
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for source in (repository_root / "benchmarks/state_preparation/phase2").glob("*.py"):
        relative = source.relative_to(repository_root)
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    generated_version = repository_root / "src/mqt/yaqs/_version.py"
    if generated_version.is_file():
        destination = checkout / generated_version.relative_to(repository_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated_version, destination)

    def run_git(*arguments: str) -> None:
        """Run one fixed Git setup operation in the isolated checkout."""
        subprocess.run(
            (git, "-C", str(checkout), *arguments),
            check=True,
            capture_output=True,
        )

    run_git("init", "--quiet")
    run_git("config", "user.name", "WP22E Test")
    run_git("config", "user.email", "wp22e-test@example.invalid")
    run_git("add", "--all")
    run_git("commit", "--quiet", "-m", "source custody fixture")

    output = checkout / "source-locked-output"
    script = r"""
import sys
from dataclasses import replace
from pathlib import Path

from benchmarks.state_preparation import training_runner
from benchmarks.state_preparation.phase2.execution_context import (
    TrainingExecutionContext,
    bind_training_plan_fingerprints,
)
from benchmarks.state_preparation.phase2.resumability import ExecutionSourceEntry, ResumabilityFingerprint
from benchmarks.state_preparation.phase2.source_lock import capture_execution_source_manifest
from benchmarks.state_preparation.phase2.training_orchestration import TrainingRunSummary
from tests.benchmarks.test_state_preparation_wp22_execution_context import _context

root = Path(sys.argv[1])
output = Path(sys.argv[2])
manifest = capture_execution_source_manifest(
    root,
    manifest_id="wp22e_clean_checkout_sources",
    entry_point="benchmarks/state_preparation/phase2/production_executors.py",
    execution_source_paths=("benchmarks/state_preparation/phase2/production_executors.py",),
    analysis_source_paths=("benchmarks/state_preparation/phase2/source_lock.py",),
    dependency_lock_paths=("pyproject.toml",),
    sealed_input_paths=("benchmarks/state_preparation/phase2/data/initial_preregistration_v1.json",),
)
base = _context()
source_by_path = {source.repo_path: source for source in manifest.source_files}
fingerprint = ResumabilityFingerprint(
    starting_commit=manifest.source_commit,
    pipeline_prefix_id=base.resumability_fingerprints[0].pipeline_prefix_id,
    dependency_versions=base.resumability_fingerprints[0].dependency_versions,
    entries=tuple(
        ExecutionSourceEntry(
            role=role,
            repository_path=path,
            starting_git_blob_id=source_by_path[path].git_blob_id,
            content_checksum=source_by_path[path].source_checksum,
        )
        for role, path in (
            ("execution_source", "benchmarks/state_preparation/phase2/production_executors.py"),
            ("lockfile", "pyproject.toml"),
            ("sealed_input", "benchmarks/state_preparation/phase2/data/initial_preregistration_v1.json"),
        )
    ),
)
plan = bind_training_plan_fingerprints(
    base.plan,
    execution_profile=base.execution_profile,
    executable_bindings=base.scoped_bindings,
    target_configurations=base.target_configurations,
    target_manifests=base.target_manifests,
    execution_source_manifest=manifest,
    resumability_fingerprints=(fingerprint,),
    required_sample_size_design=None,
)
context = TrainingExecutionContext(
    plan=plan,
    execution_profile=base.execution_profile,
    preregistration=base.preregistration,
    candidates=base.candidates,
    schedules=base.schedules,
    scoped_bindings=base.scoped_bindings,
    target_configurations=base.target_configurations,
    target_manifests=base.target_manifests,
    authorized_materializations=base.authorized_materializations,
    screening_manifest=None,
    screening_cells=(),
    required_sample_size_design=None,
    execution_source_manifest=manifest,
    resumability_fingerprints=(fingerprint,),
    external_entropy_keyring=base.external_entropy_keyring,
)
options = training_runner.resolve_options(
    training_runner.parse_arguments([
        "--preset", "training-smoke",
        "--repository-root", str(root),
        "--output", str(output),
    ])
)
result = training_runner.run(options, context=context, executor=None)
assert result == TrainingRunSummary(10, 10, 10, 0, 0)
print("REAL_SOURCE_LOCK_DEFAULT_REGISTRY_OK")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(checkout), str(checkout / "src")))
    completed = subprocess.run(
        (sys.executable, "-c", script, str(checkout), str(output)),
        cwd=checkout,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert "REAL_SOURCE_LOCK_DEFAULT_REGISTRY_OK" in completed.stdout


def test_phase2_exports_the_typed_production_custody_boundary() -> None:
    """The package API exposes typed custody while retaining the registry adapter."""
    assert phase2.OperatorGrowthSegmentedSnapshot is OperatorGrowthSegmentedSnapshot
    assert phase2.ProductionAttemptStore is ProductionAttemptStore
    assert phase2.ProductionTrainingExecutor is ProductionTrainingExecutor
    assert phase2.ResultArtifactRef is ResultArtifactRef
    assert phase2.SyntheticConfirmationExecutor is SyntheticConfirmationExecutor
    assert phase2.SyntheticConfirmationFixture is SyntheticConfirmationFixture
    assert phase2.create_default_training_executor_registry is create_default_training_executor_registry
    assert phase2.derive_result_artifact_ref is derive_result_artifact_ref
    assert phase2.execute_operator_growth_segmented_program is execute_operator_growth_segmented_program
    assert phase2.reopen_result_artifact is reopen_result_artifact


def test_reopen_rejects_a_byte_tamper(
    production_smoke_suite: _SmokeSuite,
    tmp_path: Path,
) -> None:
    """Reopening rejects mutation of any manifest-enumerated result byte."""
    source = next(result for result in production_smoke_suite.results if result.reopened.raw_trajectory is not None)
    copied = tmp_path / "copied-job"
    shutil.copytree(source.directory, copied)
    raw_ref = source.reopened.evidence.raw_trajectory_ref
    assert raw_ref is not None
    raw_path = copied / raw_ref.path
    raw_path.write_bytes(raw_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match=r"checksum|byte"):
        reopen_result_artifact(source.reference, copied)


def test_reopen_rejects_a_fully_resealed_execution_source_alias(
    production_smoke_suite: _SmokeSuite,
    tmp_path: Path,
) -> None:
    """Evidence cannot substitute the direct execution-source manifest behind sealed bytes."""
    source = production_smoke_suite.results[0]
    copied = tmp_path / "resealed-source-alias"
    shutil.copytree(source.directory, copied)
    evidence_ref = source.reopened.manifest.evidence_ref
    document = cast(
        "dict[str, object]",
        json.loads((copied / evidence_ref.path).read_text(encoding="utf-8")),
    )
    document["execution_source_manifest_checksum"] = _checksum("foreign execution source manifest")
    changed_reference = _reseal_attempt_member(
        copied,
        source.reference,
        evidence_ref.path,
        _reseal_mapping(document),
    )
    with pytest.raises(ValueError, match=r"source|aliases"):
        reopen_result_artifact(changed_reference, copied)


@pytest.mark.parametrize("mutation", ["job", "policy", "seed", "fidelity", "event_count"])
def test_reopen_rejects_fully_resealed_raw_trajectory_semantic_substitution(
    production_smoke_suite: _SmokeSuite,
    tmp_path: Path,
    mutation: str,
) -> None:
    """Nested resealing cannot detach raw values from job, policy, maps, or metrics."""
    source = next(
        result for result in production_smoke_suite.results if result.job.implementation_kind == "phase2_pipeline"
    )
    raw_ref = source.reopened.evidence.raw_trajectory_ref
    assert raw_ref is not None
    copied = tmp_path / f"resealed-raw-{mutation}"
    shutil.copytree(source.directory, copied)
    document = cast("dict[str, object]", json.loads((copied / raw_ref.path).read_text(encoding="utf-8")))
    payload = cast("dict[str, object]", document["payload"])
    if mutation == "job":
        payload["job_checksum"] = _checksum("foreign raw job")
    elif mutation == "policy":
        payload["evaluation_policy_checksum"] = _checksum("foreign raw evaluation policy")
    elif mutation == "seed":
        payload["evaluation_seed"] = cast("int", payload["evaluation_seed"]) + 1
    elif mutation == "fidelity":
        values = cast("list[float]", payload["trajectory_fidelities"])
        values[0] = 0.0 if not math.isclose(values[0], 0.0, rel_tol=0.0, abs_tol=0.0) else 1.0
    else:
        payload["sampled_nonidentity_events"] = cast("int", payload["sampled_nonidentity_events"]) + 1
    changed_reference = _reseal_attempt_member(
        copied,
        source.reference,
        raw_ref.path,
        _reseal_mapping(document),
    )
    with pytest.raises(ValueError, match=r"Raw trajectory|raw fidelity|Derived fresh|Fresh manifest"):
        reopen_result_artifact(changed_reference, copied)


@pytest.mark.parametrize("mutation", ["job", "source", "normalized_work"])
def test_reopen_rejects_fully_resealed_runtime_identity_or_work_substitution(
    production_smoke_suite: _SmokeSuite,
    tmp_path: Path,
    mutation: str,
) -> None:
    """Nested resealing cannot rewrite runtime identity or normalized work."""
    source = next(
        result for result in production_smoke_suite.results if result.job.implementation_kind == "phase2_pipeline"
    )
    resource_ref = source.reopened.evidence.resource_ref
    copied = tmp_path / f"resealed-resource-{mutation}"
    shutil.copytree(source.directory, copied)
    document = cast("dict[str, object]", json.loads((copied / resource_ref.path).read_text(encoding="utf-8")))
    payload = cast("dict[str, object]", document["payload"])
    if mutation == "job":
        payload["job_checksum"] = _checksum("foreign resource job")
    elif mutation == "source":
        payload["source_fingerprint_checksum"] = _checksum("foreign resource source")
    else:
        payload["normalized_work"] = cast("float", payload["normalized_work"]) + 1.0
    changed_reference = _reseal_attempt_member(
        copied,
        source.reference,
        resource_ref.path,
        _reseal_mapping(document),
    )
    with pytest.raises(ValueError, match=r"Runtime resource|normalized work"):
        reopen_result_artifact(changed_reference, copied)


@pytest.mark.parametrize("mutation", ["drop", "swap", "symlink", "foreign", "nested_manifest"])
def test_reopen_rejects_incomplete_swapped_linked_or_foreign_members(
    production_smoke_suite: _SmokeSuite,
    tmp_path: Path,
    mutation: str,
) -> None:
    """The manifest is an exact closed member universe, not a best-effort index."""
    source = next(result for result in production_smoke_suite.results if result.reopened.raw_trajectory is not None)
    copied = tmp_path / f"copied-{mutation}"
    shutil.copytree(source.directory, copied)
    raw_ref = source.reopened.evidence.raw_trajectory_ref
    assert raw_ref is not None
    raw_path = copied / raw_ref.path
    resource_path = copied / source.reopened.evidence.resource_ref.path
    if mutation == "drop":
        resource_path.unlink()
    elif mutation == "swap":
        raw_payload = raw_path.read_bytes()
        resource_payload = resource_path.read_bytes()
        raw_path.write_bytes(resource_payload)
        resource_path.write_bytes(raw_payload)
    elif mutation == "symlink":
        external = tmp_path / "external-raw.json"
        external.write_bytes(raw_path.read_bytes())
        raw_path.unlink()
        try:
            raw_path.symlink_to(external)
        except OSError:
            _SKIP("This Windows host does not grant symlink creation privileges.")
    elif mutation == "foreign":
        attempt_root = copied / source.reference.manifest_path
        (attempt_root.parent / "foreign-member.json").write_text("{}\n", encoding="utf-8")
    else:
        attempt_root = copied / source.reference.manifest_path
        nested = attempt_root.parent / "nested" / "attempt_manifest.json"
        nested.parent.mkdir()
        nested.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"member set|symlink|checksum|byte"):
        reopen_result_artifact(source.reference, copied)


def test_manifest_publication_is_exclusive_and_cannot_overwrite_a_member(tmp_path: Path) -> None:
    """Exclusive-create custody retains the first byte sequence at an exact path."""
    store = ProductionAttemptStore(tmp_path / "job", _checksum("exclusive job"), 1)
    first = b"first immutable payload\n"
    store.write_blob(
        "raw/value.json",
        first,
        role="raw_trajectory_sidecar",
        logical_checksum=_checksum("first logical value"),
    )
    with pytest.raises(ValueError, match="already exists"):
        store.write_blob(
            "raw/value.json",
            b"replacement payload\n",
            role="raw_trajectory_sidecar",
            logical_checksum=_checksum("replacement logical value"),
        )
    assert (tmp_path / "job" / "production_attempts" / "attempt_000001" / "raw" / "value.json").read_bytes() == first


def test_member_enumeration_rejects_nested_manifest_and_intermediate_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Descriptor-anchored enumeration includes nested names and detects a swapped parent."""
    store = ProductionAttemptStore(tmp_path / "job", _checksum("descriptor enumeration"), 1)
    nested_manifest = store.write_blob(
        "nested/attempt_manifest.json",
        b"nested foreign terminal name\n",
        role="raw_trajectory_sidecar",
        logical_checksum=_checksum("nested terminal bytes"),
        media_type="application/octet-stream",
    )
    assert store.member_paths() == (nested_manifest.path,)

    nested = store.job_directory / store.relative_attempt_directory / "nested"
    detached = store.job_directory / "detached-nested"
    if not store._descriptor_creation_supported():
        nested.rename(detached)
        try:
            nested.symlink_to(detached, target_is_directory=True)
        except OSError:
            _SKIP("This host does not grant symlink creation privileges.")
        with pytest.raises(ValueError, match=r"symlink|changed|unsafe"):
            store.member_paths()
        return
    real_listdir = os.listdir
    swapped = False

    def swapping_listdir(path: int) -> list[str]:
        """Swap the listed intermediate directory before its descriptor is opened.

        Returns:
            The names captured from the still-pinned parent descriptor.
        """
        nonlocal swapped
        names = real_listdir(path)
        if not swapped and "nested" in names:
            swapped = True
            nested.rename(detached)
            try:
                nested.symlink_to(detached, target_is_directory=True)
            except OSError:
                _SKIP("This host does not grant symlink creation privileges.")
        return names

    monkeypatch.setattr(production_executor_module.os, "listdir", swapping_listdir)
    with pytest.raises(ValueError, match=r"symlink|changed|unsafe"):
        store.member_paths()
    assert swapped


def test_atomic_terminal_publication_never_exposes_staging_or_overwrites_partial_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed link has no terminal path, and hostile partial terminal bytes remain non-authoritative."""
    store = ProductionAttemptStore(tmp_path / "job", _checksum("atomic terminal"), 1)
    payload = b'{"complete":true}\n'
    descriptor_support = store._descriptor_creation_supported()
    if descriptor_support:
        monkeypatch.setattr(
            ProductionAttemptStore,
            "_descriptor_creation_supported",
            staticmethod(lambda: True),
        )

    def interrupted_link(*_arguments: object, **_keywords: object) -> None:
        """Simulate interruption after the fully synced staging write.

        Raises:
            OSError: Always, to model interruption before the atomic link.
        """
        msg = "injected publication interruption"
        raise OSError(msg)

    monkeypatch.setattr(production_executor_module.os, "link", interrupted_link)
    with pytest.raises(OSError, match="injected publication interruption"):
        store._publish_terminal_payload(payload)
    manifest_path = store.job_directory / store.manifest_relative_path
    assert not manifest_path.exists()
    assert tuple(store.job_directory.glob(".wp22e-terminal-*.tmp")) == ()

    monkeypatch.undo()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    hostile_partial = b'{"schema_version":'
    manifest_path.write_bytes(hostile_partial)
    with pytest.raises((TypeError, ValueError)):
        store.derive_existing_ref()
    with pytest.raises(ValueError, match="already exists"):
        store._publish_terminal_payload(payload)
    assert manifest_path.read_bytes() == hostile_partial


def test_reopen_rejects_fully_resealed_scheduled_map_alias_substitution(tmp_path: Path) -> None:
    """Nested resealing cannot detach a scheduled member seed from its fixed-map ensemble."""
    job_checksum = _checksum("resealed scheduled map job")
    source_checksum = _checksum("resealed scheduled map source")
    circuit_checksum = _checksum("resealed scheduled map circuit")
    provider_checksum = _checksum("resealed scheduled map provider")
    store = ProductionAttemptStore(tmp_path / "scheduled-map", job_checksum, 1)
    ensemble = production_executor_module.KrotovFixedMapEnsemble(
        role="training_trajectory",
        resolved_seed=100,
        stage_index=0,
        stage_id="resealed_map_test",
        stage_configuration_checksum=_checksum("resealed map stage"),
        circuit_checksum=circuit_checksum,
        provider_checksum=provider_checksum,
        ensemble_index=0,
        refresh_index=0,
        global_iteration_start=0,
        trajectory_maps=((KrotovNoiseMap(source_gate_index=0, is_identity=True),),),
    )
    ensemble_ref = store.write_blob(
        "maps/request_00000000_component_000.json",
        f"{ensemble.to_json()}\n".encode(),
        role="fixed_map_ensemble",
        logical_checksum=ensemble.content_checksum,
    )
    map_evidence = production_executor_module.ScheduledMapEvidence(
        request_checksum=_checksum("resealed map request"),
        policy_checksum=_checksum("resealed map policy"),
        membership_checksum=_checksum("resealed map membership"),
        component_membership_checksums=(),
        member_seeds=(200,),
        component_member_seeds=((200,),),
        map_role="training_trajectory",
        resolved_seeds=(100,),
        circuit_checksum=circuit_checksum,
        provider_checksums=(provider_checksum,),
        ensemble_refs=(ensemble_ref,),
    )
    map_ref = store.write_json_blob(
        "map_evidence/request_00000000.json",
        map_evidence.to_dict(),
        role="scheduled_map_evidence",
    )
    resource_ref = store.write_json_blob(
        "runtime/resources.json",
        production_executor_module._typed_document(
            "runtime_resources",
            {
                "job_checksum": job_checksum,
                "source_fingerprint_checksum": source_checksum,
                "wall_time_seconds": 0.0,
                "peak_memory_bytes": 0,
                "normalized_work": 1.0,
                "failure_phase": "test",
                "partial_receipts": {},
                "circuit": None,
            },
        ),
        role="runtime_resources",
    )
    evidence = production_executor_module.ProductionNumericalEvidence(
        job_checksum=job_checksum,
        attempt=1,
        artifact_kind="pipeline",
        status="failure",
        execution_source_manifest_checksum=source_checksum,
        source_fingerprint_checksum=source_checksum,
        executable_binding_checksum=_checksum("resealed map executable"),
        scheduled_program_checksum=_checksum("resealed map program"),
        target_identity={"target": "resealed_map_test"},
        evaluation_policy_checksum=_checksum("resealed map evaluation"),
        structural_prefix_checksums=(),
        schedule_snapshot_ref=None,
        map_evidence_refs=(map_ref,),
        diagnostic_refs=(),
        raw_trajectory_ref=None,
        resource_ref=resource_ref,
        derived_metrics={"execution_preset": "paper-screen"},
        failure={"phase": "test"},
    )
    evidence_ref = store.write_json_blob(
        "production_evidence.json",
        evidence.to_dict(),
        role="production_evidence",
    )
    reference = store.publish(
        artifact_kind="pipeline",
        status="failure",
        execution_source_manifest_checksum=source_checksum,
        source_fingerprint_checksum=source_checksum,
        blobs=(ensemble_ref, map_ref, resource_ref, evidence_ref),
        evidence_ref=evidence_ref,
    )
    reopen_result_artifact(reference, store.job_directory)

    document = cast(
        "dict[str, object]",
        json.loads((store.job_directory / map_ref.path).read_text(encoding="utf-8")),
    )
    document["resolved_seeds"] = [101]
    changed_reference = _reseal_attempt_member(
        store.job_directory,
        reference,
        map_ref.path,
        _reseal_mapping(document),
    )
    with pytest.raises(ValueError, match=r"resolved/member seeds"):
        reopen_result_artifact(changed_reference, store.job_directory)


def test_reopen_replays_pipeline_request_identity_after_full_nested_reseal(tmp_path: Path) -> None:
    """A fully resealed pipeline map request must still equal its snapshot callback."""
    context = _context_with_production_source()
    base_job = next(job for job in context.plan.jobs if job.implementation_kind == "phase2_pipeline")
    resolved = ProductionTrainingExecutor(context).authority.resolve(base_job)
    target_identity = resolved.target.identity_dict()
    program = compile_frozen_schedule_trace(
        wp22c_support._schedule("direct_matched_fixed_crn"),
        ScheduledJobSeedSet(407),
    )
    objective_checksum = production_executor_module._pipeline_objective_checksum(target_identity)

    def gradient(request: object) -> ScheduledTrainingGradientResult:
        typed = cast("production_executor_module.ScheduledTrainingGradientRequest", request)
        return ScheduledTrainingGradientResult.for_request(typed, (0.0,) * len(typed.parameters))

    def validate(request: object) -> ScheduledValidationResult:
        typed = cast("production_executor_module.ScheduledValidationRequest", request)
        return ScheduledValidationResult.for_request(typed, 0.5)

    snapshot = execute_scheduled_program(
        program,
        wp22c_support._initial_krotov_snapshot(program),
        KrotovScheduledUpdateAdapter(objective_checksum, gradient),
        validation_executor=validate,
        stop_after_updates=1,
    )
    circuit_binding = create_bmpd_circuit_binding(6, 1)
    gate_count = len(circuit_binding.circuit.gates)
    links = production_executor_module._pipeline_snapshot_numerical_links(
        snapshot,
        target_identity=target_identity,
        circuit_checksum=circuit_binding.content_checksum,
        circuit_gate_count=gate_count,
    )
    assert len(links) == 2

    store = ProductionAttemptStore(tmp_path / "pipeline-request-replay", base_job.content_checksum, 1)
    snapshot_ref = store.write_blob(
        "schedule/snapshot.json",
        f"{snapshot.to_json()}\n".encode(),
        role="schedule_snapshot",
        logical_checksum=snapshot.content_checksum,
    )
    map_blobs = []
    map_refs = []
    identity_path = tuple(KrotovNoiseMap(source_gate_index=index, is_identity=True) for index in range(gate_count))
    for request_index, link in enumerate(links):
        ensemble_refs = []
        for component_index, member_seeds in enumerate(link.component_member_seeds):
            ensemble = KrotovFixedMapEnsemble(
                role=link.map_role,
                resolved_seed=link.resolved_seeds[component_index],
                stage_index=0,
                stage_id="pipeline_request_replay",
                stage_configuration_checksum=_checksum(f"pipeline replay stage {request_index}"),
                circuit_checksum=link.circuit_checksum,
                provider_checksum=link.provider_checksums[component_index],
                ensemble_index=request_index,
                refresh_index=component_index,
                global_iteration_start=0,
                trajectory_maps=(identity_path,) * len(member_seeds),
            )
            ensemble_ref = store.write_blob(
                f"maps/request_{request_index:08d}_component_{component_index:03d}.json",
                f"{ensemble.to_json()}\n".encode(),
                role="fixed_map_ensemble",
                logical_checksum=ensemble.content_checksum,
            )
            map_blobs.append(ensemble_ref)
            ensemble_refs.append(ensemble_ref)
        map_evidence = production_executor_module.ScheduledMapEvidence(
            request_checksum=link.request_checksum,
            policy_checksum=link.policy_checksum,
            membership_checksum=link.membership_checksum,
            component_membership_checksums=link.component_membership_checksums,
            member_seeds=link.member_seeds,
            component_member_seeds=link.component_member_seeds,
            map_role=link.map_role,
            resolved_seeds=link.resolved_seeds,
            circuit_checksum=link.circuit_checksum,
            provider_checksums=link.provider_checksums,
            ensemble_refs=tuple(ensemble_refs),
        )
        map_refs.append(
            store.write_json_blob(
                f"map_evidence/request_{request_index:08d}.json",
                map_evidence.to_dict(),
                role="scheduled_map_evidence",
            )
        )
    resource_ref = store.write_json_blob(
        "runtime/resources.json",
        production_executor_module._runtime_resource_document(
            resolved=resolved,
            circuit_binding=circuit_binding,
            wall_time_seconds=0.0,
            peak_memory_bytes=0,
            normalized_work=1.0,
            failure_phase="test",
            partial_receipts={},
        ),
        role="runtime_resources",
    )
    evidence = production_executor_module.ProductionNumericalEvidence(
        job_checksum=base_job.content_checksum,
        attempt=1,
        artifact_kind="pipeline",
        status="failure",
        execution_source_manifest_checksum=resolved.execution_source_manifest_checksum,
        source_fingerprint_checksum=cast("str", base_job.source_fingerprint_checksum),
        executable_binding_checksum=cast("str", base_job.executable_binding_checksum),
        scheduled_program_checksum=program.content_checksum,
        target_identity=target_identity,
        evaluation_policy_checksum=resolved.evaluation_policy.content_checksum,
        structural_prefix_checksums=(),
        schedule_snapshot_ref=snapshot_ref,
        map_evidence_refs=tuple(map_refs),
        diagnostic_refs=(),
        raw_trajectory_ref=None,
        resource_ref=resource_ref,
        derived_metrics={"execution_preset": "paper-screen"},
        failure={"phase": "test"},
    )
    evidence_ref = store.write_json_blob(
        "production_evidence.json",
        evidence.to_dict(),
        role="production_evidence",
    )
    reference = store.publish(
        artifact_kind="pipeline",
        status="failure",
        execution_source_manifest_checksum=resolved.execution_source_manifest_checksum,
        source_fingerprint_checksum=cast("str", base_job.source_fingerprint_checksum),
        blobs=(*map_blobs, *map_refs, snapshot_ref, resource_ref, evidence_ref),
        evidence_ref=evidence_ref,
    )
    reopen_result_artifact(reference, store.job_directory)

    first_map_ref = map_refs[0]
    document = cast(
        "dict[str, object]",
        json.loads((store.job_directory / first_map_ref.path).read_text(encoding="utf-8")),
    )
    document["request_checksum"] = _checksum("foreign pipeline callback")
    changed_reference = _reseal_attempt_member(
        store.job_directory,
        reference,
        first_map_ref.path,
        _reseal_mapping(document),
    )
    with pytest.raises(ValueError, match="exact snapshot callback"):
        reopen_result_artifact(changed_reference, store.job_directory)


def test_terminal_manifest_crash_recovery_and_resume_reopen_the_same_attempt(
    production_smoke_suite: _SmokeSuite,
) -> None:
    """A published manifest is enough to recover and resume without attempt two."""
    result = production_smoke_suite.results[0]
    recovered = production_smoke_suite.executor.execute(result.job, result.directory, _FRESH_CONTROLS)
    outcomes = load_training_job_outcome_history(result.directory, result.job)
    assert len(outcomes) == 1
    schedule = result.job.strategy_schedule
    assert schedule is not None
    resume_state = TrainingScheduleResumeState(
        strategy_schedule_checksum=result.job.strategy_schedule_checksum,
        schedule_id=schedule.schedule_id,
        resume_requested=True,
        overwrite_requested=False,
        prior_attempt=1,
        prior_outcome_checksum=outcomes[0].content_checksum,
        prior_status="success",
    )
    resumed = production_smoke_suite.executor.execute(
        result.job,
        result.directory,
        JobExecutionControls(resume=True, overwrite=False, schedule_resume_state=resume_state),
    )
    assert recovered == resumed == result.reference
    assert not (result.directory / "production_attempts" / "attempt_000002").exists()


def test_structured_failure_is_redacted_terminal_and_not_silently_retried(
    production_smoke_suite: _SmokeSuite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary execution errors publish one redacted, reopenable failure attempt."""
    job = production_smoke_suite.results[0].job
    directory = tmp_path / "failed-job"

    def fail_dispatch(*_arguments: object) -> ResultArtifactRef:
        retained_allocation = bytearray(64 * 1024)
        assert retained_allocation
        msg = "secret-bearing scientific diagnostic"
        raise RuntimeError(msg)

    monkeypatch.setattr(production_executor_module, "_dispatch_production_attempt", fail_dispatch)
    with pytest.raises(RuntimeError, match="secret-bearing"):
        production_smoke_suite.executor.execute(job, directory, _FRESH_CONTROLS)
    reference = derive_result_artifact_ref(directory, job.content_checksum, 1)
    reopened = reopen_result_artifact(reference, directory)
    assert reference.status == "failure"
    assert reopened.evidence.failure == {
        "phase": "production_execution",
        "exception_type": "RuntimeError",
        "message": "production executor failed; diagnostics are intentionally redacted",
    }
    failure_resources = cast("Mapping[str, object]", reopened.resources["payload"])
    partial = cast("Mapping[str, object]", failure_resources["partial_receipts"])
    assert cast("float", failure_resources["wall_time_seconds"]) >= 0.0
    assert cast("int", failure_resources["peak_memory_bytes"]) > 0
    assert failure_resources["normalized_work"] == pytest.approx(0.0)
    assert partial["closed_artifact_count"] == 0
    assert partial["normalized_work_unavailable"] is True
    assert "secret-bearing" not in str(reopened.evidence.to_dict())
    with pytest.raises(PersistedProductionAttemptError, match="structured failure"):
        production_smoke_suite.executor.execute(job, directory, _FRESH_CONTROLS)
    assert not (directory / "production_attempts" / "attempt_000002").exists()


def test_late_failure_retains_raw_bytes_without_claiming_success_evidence(
    production_smoke_suite: _SmokeSuite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Late raw and diagnostic files remain manifest members but not failure aliases."""
    job = production_smoke_suite.results[0].job
    directory = tmp_path / "late-failed-job"

    def late_failure(
        _resolved: object,
        store: ProductionAttemptStore,
        _artifact_kind: object,
    ) -> ResultArtifactRef:
        raw = production_executor_module._typed_document(
            "raw_trajectory_fidelities",
            {"trajectory_count": 2, "trajectory_fidelities": [0.4, 0.6]},
        )
        diagnostic = production_executor_module._typed_document(
            "pilot_diagnostic",
            {"pathwise_update_vectors": [[0.1], [0.2]]},
        )
        store.write_json_blob(
            "evaluation/raw_trajectory_fidelities.json",
            raw,
            role="raw_trajectory_sidecar",
        )
        store.write_json_blob(
            "diagnostics/pathwise_update_vectors.json",
            diagnostic,
            role="pilot_diagnostic_sidecar",
        )
        msg = "late failure after numerical sidecars"
        raise RuntimeError(msg)

    monkeypatch.setattr(production_executor_module, "_dispatch_production_attempt", late_failure)
    with pytest.raises(RuntimeError, match="late failure"):
        production_smoke_suite.executor.execute(job, directory, _FRESH_CONTROLS)
    reference = derive_result_artifact_ref(directory, job.content_checksum, 1)
    reopened = reopen_result_artifact(reference, directory)
    assert reopened.evidence.status == "failure"
    assert reopened.evidence.raw_trajectory_ref is None
    assert reopened.evidence.diagnostic_refs == ()
    assert reopened.raw_trajectory is None
    assert reopened.pilot_diagnostics == ()
    failure_resources = cast("Mapping[str, object]", reopened.resources["payload"])
    partial = cast("Mapping[str, object]", failure_resources["partial_receipts"])
    assert failure_resources["normalized_work"] == pytest.approx(2.0)
    assert cast("float", failure_resources["wall_time_seconds"]) >= 0.0
    assert cast("int", failure_resources["peak_memory_bytes"]) >= 0
    assert partial["closed_artifact_count"] == 2
    assert partial["closed_role_counts"] == {
        "pilot_diagnostic_sidecar": 1,
        "raw_trajectory_sidecar": 1,
    }
    assert partial["normalized_work_is_lower_bound"] is True
    assert partial["unavailable_partial_work_roles"] == ("pilot_diagnostic_sidecar",)
    roles = {ref.role for ref in reopened.manifest.blobs}
    assert {"raw_trajectory_sidecar", "pilot_diagnostic_sidecar"} <= roles
    retained = tuple(
        directory / ref.path
        for ref in reopened.manifest.blobs
        if ref.role in {"raw_trajectory_sidecar", "pilot_diagnostic_sidecar"}
    )
    assert len(retained) == 2
    assert all(path.read_bytes() for path in retained)


def test_forged_method_schedule_pair_rejects_before_output_mutation(
    production_smoke_suite: _SmokeSuite,
    tmp_path: Path,
) -> None:
    """Only an exact context-owned job object may reach target materialization or optimization."""
    pipeline_jobs = tuple(
        result.job for result in production_smoke_suite.results if result.job.implementation_kind == "phase2_pipeline"
    )
    original, foreign = pipeline_jobs[:2]
    forged = replace(
        original,
        strategy_schedule=foreign.strategy_schedule,
        strategy_schedule_checksum=foreign.strategy_schedule_checksum,
    )
    output = tmp_path / "forged-pair"
    with pytest.raises(ValueError, match="exact TrainingJob object"):
        production_smoke_suite.executor.execute(forged, output, _FRESH_CONTROLS)
    assert not output.exists()


def test_ballarin_rejection_precedes_runner_resolution(tmp_path: Path) -> None:
    """Evaluation-only Ballarin noise cannot instantiate an optimization runner."""
    store = ProductionAttemptStore(tmp_path / "ballarin", _checksum("ballarin job"), 1)
    pipeline = SimpleNamespace(stages=(SimpleNamespace(training_noise_id="ballarin_coupled"),))
    with pytest.raises(ValueError, match="evaluation-only"):
        production_executor_module._run_structural_prefix(
            cast("production_executor_module.ResolvedProductionJob", object()),
            cast("TrainingPipelineConfig", pipeline),
            store,
        )
    assert not (tmp_path / "ballarin").exists()


@pytest.mark.parametrize(
    ("preset", "purpose", "data_role", "trajectory_count"),
    [
        ("paper-pilot", "pilot_fresh_evaluation", "development", 1024),
        ("paper-screen", "screening_outer", "screening_selection", 256),
    ],
)
def test_bounded_pipeline_paper_presets_publish_typed_raw_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preset: Preset,
    purpose: str,
    data_role: str,
    trajectory_count: int,
) -> None:
    """Exact pilot and screen closures retain typed publication around bounded numerics."""
    method_id = "layerwise_bmpd_noiseless"
    context = _context_with_production_source()
    base_job = next(job for job in context.plan.jobs if job.method_id == method_id)
    base = ProductionTrainingExecutor(context).authority.resolve(base_job)
    if preset == "paper-pilot":
        bindings = wp22a_support._pilot_bindings()
    else:
        bindings = tuple(
            wp22a_support._binding(
                candidate,
                "primary_q6",
                preset="paper-screen",
                normalized_compute_cap=1_000.0,
            )
            for candidate in SCREEN_METHOD_IDS
        )
    profile = wp22a_support._profile(bindings)
    catalog = RepositoryBindingCatalog.from_profile(
        profile,
        RepositoryImplementationCatalog.frozen(
            screening_outer_trajectory_count=256,
            smoke_evaluation_trajectory_count=2,
        ),
    )
    binding = next(item for item in bindings if item.publication_method_id == method_id and item.qubit_count == 6)
    link = catalog.resolve(preset, binding.publication_candidate_checksum, "primary_q6")
    policy = next(item for item in link.binding.evaluation_policies if item.purpose == purpose)
    program = ScheduledExecutionProgram.compile(
        link,
        link.binding.strategy_schedule,
        ScheduledJobSeedSet(base_job.optimization_seed),
    )
    paper_job = replace(
        base_job,
        preset=preset,
        candidate_configuration_checksum=link.binding.publication_candidate_checksum,
        implementation_checksum=link.binding.implementation_checksum,
        strategy_schedule_checksum=link.binding.strategy_schedule.content_checksum,
        strategy_schedule=link.binding.strategy_schedule,
        data_role=data_role,
        output_path=base_job.output_path.replace("roles/development/", f"roles/{data_role}/", 1),
        execution_profile_checksum=profile.content_checksum,
        scoped_binding_checksum=link.binding.content_checksum,
        executable_binding_checksum=link.content_checksum,
        evaluation_policy_checksum=policy.content_checksum,
        scheduled_execution_program_checksum=program.content_checksum,
    )
    resolved = production_executor_module.ResolvedProductionJob(
        paper_job,
        link,
        base.target_configuration,
        base.target_manifest,
        base.target_spec,
        base.target,
        policy,
        program,
        context.execution_source_manifest.content_checksum,
        None,
    )
    circuit_binding = create_bmpd_circuit_binding(6, 1)
    selected = (0.0,) * circuit_binding.circuit.num_params
    parameter_checksum = canonical_checksum({
        "dtype": "float64",
        "parameters": list(selected),
    })
    snapshot_checksum = _checksum(f"{preset} bounded schedule snapshot")

    class _BoundedSnapshot:
        """Minimal completed snapshot codec used around the skipped 200 updates."""

        content_checksum = snapshot_checksum
        program_checksum = program.content_checksum
        multistart_evidence = SimpleNamespace(
            selected_parameter_checksum=parameter_checksum,
            selected_parameter_artifact=SimpleNamespace(parameters=selected),
            selected_start_index=0,
            selected_update=200,
            total_normalized_work=0.0,
        )

        def to_json(self) -> str:
            """Return a checksum-addressed bounded snapshot document.

            Returns:
                Canonical snapshot JSON accepted by the patched bounded codec.
            """
            return canonical_json({
                "schema_version": "wp22e.bounded_schedule_snapshot.v1",
                "content_checksum": self.content_checksum,
            })

        @classmethod
        def from_json(cls, _payload: str) -> _BoundedSnapshot:
            """Reopen the bounded snapshot during full manifest verification.

            Returns:
                The deterministic bounded snapshot.
            """
            return cls()

    snapshot = _BoundedSnapshot()

    def bounded_schedule(
        _resolved: object,
        _store: ProductionAttemptStore,
    ) -> tuple[object, object, tuple[float, ...], tuple[str, ...], list[object], float, int]:
        """Skip optimizer work while retaining its exact closure and publication path.

        Returns:
            A completed bounded snapshot, materialized circuit, parameters, and work.
        """
        return snapshot, circuit_binding, selected, (_checksum(f"{preset} structural prefix"),), [], 0.0, 0

    def noiseless_metrics(*_arguments: object, **_keywords: object) -> tuple[float, float]:
        """Return one deterministic reference metric pair.

        Returns:
            Loss and noiseless fidelity.
        """
        return 0.1, 0.9

    def noisy_metrics_with_maps(
        circuit: object,
        *_arguments: object,
        **_keywords: object,
    ) -> SimpleNamespace:
        """Return one bounded member fidelity and one map per circuit gate."""
        gates = cast("Sequence[SimpleNamespace]", cast("SimpleNamespace", circuit).gates)
        maps = tuple(KrotovNoiseMap(source_gate_index=index, is_identity=True) for index in range(len(gates)))
        return SimpleNamespace(
            loss=0.2,
            mean_fidelity=0.8,
            trajectory_fidelities=(0.8,),
            realized_noise_maps=(maps,),
        )

    def bounded_diagnostic(
        *,
        store: ProductionAttemptStore,
        checkpoint_parameter_checksum: str,
        **_keywords: object,
    ) -> tuple[tuple[object, ...], object] | None:
        """Persist a typed q6 pilot diagnostic or the screen's exact absence.

        Returns:
            One typed diagnostic reference for pilot and ``None`` for screening.
        """
        if preset != "paper-pilot":
            return None
        diagnostic_policy = link.binding.pilot_diagnostic_policy
        assert diagnostic_policy is not None
        fresh_ref = next(
            ref for ref in store.written_refs if ref.path.endswith("/evaluation/fresh_fixed_map_ensemble.json")
        )
        fresh_ensemble = production_executor_module.KrotovFixedMapEnsemble.from_json(
            (store.job_directory / fresh_ref.path).read_text(encoding="utf-8")
        )
        trajectory_maps = fresh_ensemble.replay_maps()[0]
        provider_checksum = _checksum("bounded pilot provider")
        ensemble_refs = []
        for index in range(32):
            ensemble = production_executor_module.KrotovFixedMapEnsemble(
                role="pilot_evaluation",
                resolved_seed=index,
                stage_index=0,
                stage_id="bounded_pilot_diagnostic",
                stage_configuration_checksum=_checksum("bounded pilot diagnostic stage"),
                circuit_checksum=circuit_binding.content_checksum,
                provider_checksum=provider_checksum,
                ensemble_index=index,
                refresh_index=0,
                global_iteration_start=0,
                trajectory_maps=(trajectory_maps,),
            )
            ensemble_refs.append(
                store.write_blob(
                    f"diagnostics/maps/pathwise_{index:03d}.json",
                    f"{ensemble.to_json()}\n".encode(),
                    role="fixed_map_ensemble",
                    logical_checksum=ensemble.content_checksum,
                )
            )
        evidence = PilotDiagnosticEvidence(
            job_checksum=paper_job.content_checksum,
            policy_checksum=diagnostic_policy.content_checksum,
            checkpoint_parameter_checksum=checkpoint_parameter_checksum,
            parameter_vector_checksum=checkpoint_parameter_checksum,
            circuit_checksum=circuit_binding.content_checksum,
            provider_checksum=provider_checksum,
            estimator_checksum=_checksum("bounded pilot estimator"),
            member_seeds=tuple(range(32)),
            ensemble_refs=tuple(ensemble_refs),
            pathwise_update_vectors=(selected,) * 32,
        )
        ref = store.write_json_blob(
            "diagnostics/pathwise_update_vectors.json",
            evidence.to_dict(),
            role="pilot_diagnostic_sidecar",
        )
        return (*ensemble_refs, ref), ref

    monkeypatch.setattr(production_executor_module, "ScheduledExecutionSnapshot", _BoundedSnapshot)
    monkeypatch.setattr(production_executor_module, "_pipeline_snapshot_numerical_links", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(production_executor_module, "_scheduled_pipeline_execution", bounded_schedule)
    monkeypatch.setattr(production_executor_module, "state_preparation_metrics", noiseless_metrics)
    monkeypatch.setattr(
        production_executor_module,
        "noisy_state_preparation_metrics_with_maps",
        noisy_metrics_with_maps,
    )
    monkeypatch.setattr(production_executor_module, "_pilot_diagnostic", bounded_diagnostic)
    directory = tmp_path / preset
    reference = production_executor_module._execute_pipeline_attempt(
        resolved,
        ProductionAttemptStore(directory, paper_job.content_checksum, 1),
    )
    reopened = reopen_result_artifact(reference, directory)
    assert isinstance(reference, ResultArtifactRef)
    assert reopened.evidence.schedule_snapshot_ref is not None
    assert reopened.raw_trajectory is not None
    raw = cast("Mapping[str, object]", reopened.raw_trajectory["payload"])
    assert raw["data_role"] == data_role
    assert raw["trajectory_count"] == trajectory_count
    assert len(cast("tuple[float, ...]", raw["trajectory_fidelities"])) == trajectory_count
    assert any(ref.role == "fixed_map_ensemble" for ref in reopened.manifest.blobs)
    assert len(reopened.pilot_diagnostics) == (1 if preset == "paper-pilot" else 0)
    raw_ref = reopened.evidence.raw_trajectory_ref
    assert raw_ref is not None
    raw_alias_directory = tmp_path / f"{preset}-resealed-raw-map-alias"
    shutil.copytree(directory, raw_alias_directory)
    raw_document = cast(
        "dict[str, object]",
        json.loads((raw_alias_directory / raw_ref.path).read_text(encoding="utf-8")),
    )
    raw_payload = cast("dict[str, object]", raw_document["payload"])
    raw_payload["fixed_map_ensemble_checksum"] = _checksum(f"{preset} foreign fresh map")
    changed_raw_reference = _reseal_attempt_member(
        raw_alias_directory,
        reference,
        raw_ref.path,
        _reseal_mapping(raw_document),
    )
    with pytest.raises(ValueError, match=r"Fresh raw sidecar"):
        reopen_result_artifact(changed_raw_reference, raw_alias_directory)
    if preset == "paper-pilot":
        diagnostic_ref = reopened.evidence.diagnostic_refs[0]
        resealed_directory = tmp_path / "paper-pilot-resealed-member-alias"
        shutil.copytree(directory, resealed_directory)
        diagnostic_document = cast(
            "dict[str, object]",
            json.loads((resealed_directory / diagnostic_ref.path).read_text(encoding="utf-8")),
        )
        member_seeds = cast("list[int]", diagnostic_document["member_seeds"])
        member_seeds[0] = 10_000
        changed_reference = _reseal_attempt_member(
            resealed_directory,
            reference,
            diagnostic_ref.path,
            _reseal_mapping(diagnostic_document),
        )
        with pytest.raises(ValueError, match=r"member seed|path order"):
            reopen_result_artifact(changed_reference, resealed_directory)


def _operator_screen_resolved() -> production_executor_module.ResolvedProductionJob:
    """Build one exact context-compatible paper-screen operator job.

    Returns:
        The authorized target closed over the real segmented program.
    """
    context = _context_with_production_source()
    base_job = next(job for job in context.plan.jobs if job.method_id == "adapt_style_state_preparation")
    base = ProductionTrainingExecutor(context).authority.resolve(base_job)
    screen_bindings = tuple(
        wp22a_support._binding(
            method_id,
            "primary_q6",
            preset="paper-screen",
            normalized_compute_cap=1_000_000.0,
        )
        for method_id in SCREEN_METHOD_IDS
    )
    profile = wp22a_support._profile(
        screen_bindings,
    )
    catalog = RepositoryBindingCatalog.from_profile(
        profile,
        RepositoryImplementationCatalog.frozen(
            screening_outer_trajectory_count=256,
            smoke_evaluation_trajectory_count=2,
        ),
    )
    binding = next(item for item in screen_bindings if item.publication_method_id == "adapt_style_state_preparation")
    link = catalog.resolve("paper-screen", binding.publication_candidate_checksum, "primary_q6")
    policy = next(item for item in link.binding.evaluation_policies if item.purpose == "screening_outer")
    program = ScheduledExecutionProgram.compile(
        link,
        link.binding.strategy_schedule,
        ScheduledJobSeedSet(base_job.optimization_seed),
    )
    screen_job = replace(
        base_job,
        preset="paper-screen",
        method_id=link.binding.publication_method_id,
        implementation_kind="operator_growth",
        candidate_configuration_checksum=link.binding.publication_candidate_checksum,
        implementation_checksum=link.binding.implementation_checksum,
        strategy_schedule_checksum=link.binding.strategy_schedule.content_checksum,
        strategy_schedule=link.binding.strategy_schedule,
        data_role="screening_selection",
        output_path=base_job.output_path.replace("roles/development/", "roles/screening_selection/", 1),
        execution_profile_checksum=profile.content_checksum,
        scoped_binding_checksum=link.binding.content_checksum,
        executable_binding_checksum=link.content_checksum,
        evaluation_policy_checksum=policy.content_checksum,
        scheduled_execution_program_checksum=program.content_checksum,
    )
    return production_executor_module.ResolvedProductionJob(
        screen_job,
        link,
        base.target_configuration,
        base.target_manifest,
        base.target_spec,
        base.target,
        policy,
        program,
        context.execution_source_manifest.content_checksum,
        None,
    )


def _install_bounded_operator_numerics(
    resolved: production_executor_module.ResolvedProductionJob,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_after_member_calls: int | None = None,
) -> list[int]:
    """Install deterministic one-pass numerics while retaining exact callback counts.

    Returns:
        A mutable one-item list containing the actual member-evolution count.
    """
    validation_scores: dict[int, float] = {}
    for update, score in ((99, 0.9), (199, 0.8)):
        membership = resolved.scheduled_program.policy(0, update).checkpoint_membership
        assert membership is not None
        validation_scores.update(dict.fromkeys(membership.member_seeds, score))
    member_calls = [0]

    def one_pass_metrics(
        circuit: object,
        theta: object,
        _target: object,
        _noise_model: object,
        options: object,
        **_keywords: object,
    ) -> SimpleNamespace:
        """Return one fidelity and its maps from one synthetic trajectory pass.

        Raises:
            RuntimeError: If the requested injected failure boundary was crossed.
        """
        member_calls[0] += 1
        if fail_after_member_calls is not None and member_calls[0] > fail_after_member_calls:
            msg = "injected operator member failure"
            raise RuntimeError(msg)
        seed = cast("int", cast("SimpleNamespace", options).random_seed)
        parameters = tuple(float(value) for value in np.asarray(theta, dtype=np.float64))
        if seed in validation_scores:
            fidelity = validation_scores[seed]
        else:
            gates = cast("Sequence[SimpleNamespace]", cast("SimpleNamespace", circuit).gates)
            trainable_ids = tuple(cast("str", gate.logical_gate_id) for gate in gates if gate.param_index is not None)
            coefficients = tuple((sum(map(ord, operator_id)) % 17 + 1) * 0.002 for operator_id in trainable_ids)
            loss = 0.4 + sum(
                coefficient * math.sin(parameter)
                for coefficient, parameter in zip(coefficients, parameters, strict=True)
            )
            fidelity = 1.0 - loss
        gates = cast("Sequence[SimpleNamespace]", cast("SimpleNamespace", circuit).gates)
        maps = tuple(KrotovNoiseMap(source_gate_index=index, is_identity=True) for index in range(len(gates)))
        return SimpleNamespace(
            loss=1.0 - fidelity,
            mean_fidelity=fidelity,
            trajectory_fidelities=(fidelity,),
            realized_noise_maps=(maps,),
        )

    def noiseless_metrics(*_arguments: object, **_keywords: object) -> tuple[float, float]:
        """Return bounded deterministic fresh noiseless metrics."""
        return 0.05, 0.95

    def bounded_measure(callback: Callable[[], object]) -> production_executor_module._MeasuredCall:
        """Run synthetic callbacks without retaining tracemalloc's full object graph.

        Returns:
            A deterministic zero-resource measurement around the callback value.
        """
        return production_executor_module._MeasuredCall(
            callback(),
            0.0,
            0,
        )

    monkeypatch.setattr(production_executor_module, "noisy_state_preparation_metrics_with_maps", one_pass_metrics)
    monkeypatch.setattr(production_executor_module, "state_preparation_metrics", noiseless_metrics)
    monkeypatch.setattr(production_executor_module, "_measure_call", bounded_measure)
    return member_calls


def test_operator_screen_persists_segmented_schedule_and_fresh_selected_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paper-screen growth closes every callback and fresh-evaluates the selected prefix."""
    resolved = _operator_screen_resolved()
    member_calls = _install_bounded_operator_numerics(resolved, monkeypatch)
    directory = tmp_path / "operator-screen"
    reference = production_executor_module._execute_operator_attempt(
        resolved,
        ProductionAttemptStore(directory, resolved.job.content_checksum, 1),
    )
    reopened = reopen_result_artifact(reference, directory)
    snapshot_ref = reopened.evidence.schedule_snapshot_ref
    assert snapshot_ref is not None
    snapshot = OperatorGrowthSegmentedSnapshot.from_json((directory / snapshot_ref.path).read_text(encoding="utf-8"))
    assert snapshot.complete
    assert len(snapshot.receipts) == 200
    assert len(snapshot.transitions) == 2
    assert len(snapshot.prefix_validations) == 2
    assert snapshot.selected_prefix_index == 0
    assert len(snapshot.selected_operator_ids) == 1
    assert len(snapshot.active_operator_ids) == 2
    objective_count = sum(len(transition.result.objective_evidence) for transition in snapshot.transitions) + sum(
        len(receipt.objective_evidence) for receipt in snapshot.receipts
    )
    assert len(reopened.scheduled_map_evidence) == objective_count + 2
    assert sum(item.map_role == "training_trajectory" for item in reopened.scheduled_map_evidence) == objective_count
    assert sum(item.map_role == "checkpoint_validation" for item in reopened.scheduled_map_evidence) == 2
    assert all(item.numerical_result_checksum is not None for item in reopened.scheduled_map_evidence)
    assert all(
        len(item.trajectory_fidelities) == (8 if item.map_role == "training_trajectory" else 256)
        for item in reopened.scheduled_map_evidence
    )
    assert member_calls[0] == snapshot.total_normalized_work + resolved.evaluation_policy.trajectory_count
    assert reopened.evidence.structural_prefix_checksums == (snapshot.content_checksum,)
    assert reopened.evidence.derived_metrics["selected_operator_ids"] == snapshot.selected_operator_ids
    assert reopened.evidence.derived_metrics["active_operator_ids"] == snapshot.active_operator_ids
    assert reopened.raw_trajectory is not None
    raw = cast("Mapping[str, object]", reopened.raw_trajectory["payload"])
    assert raw["trajectory_count"] == 256

    first_map = reopened.scheduled_map_evidence[0]
    expected_link = production_executor_module._operator_snapshot_numerical_links(snapshot)[0]
    ensembles = tuple(
        KrotovFixedMapEnsemble.from_json((directory / ensemble_ref.path).read_text(encoding="utf-8"))
        for ensemble_ref in first_map.ensemble_refs
    )
    with pytest.raises(ValueError, match="snapshot callback"):
        production_executor_module._validate_operator_numerical_link(
            replace(first_map, request_checksum=_checksum("foreign operator request")),
            expected_link,
            ensembles,
        )

    assert len(ensembles) == 1
    original = ensembles[0]
    assert original.gate_count > 0
    truncated = KrotovFixedMapEnsemble(
        role=original.role,
        resolved_seed=original.resolved_seed,
        stage_index=original.stage_index,
        stage_id=original.stage_id,
        stage_configuration_checksum=original.stage_configuration_checksum,
        circuit_checksum=original.circuit_checksum,
        provider_checksum=original.provider_checksum,
        ensemble_index=original.ensemble_index,
        refresh_index=original.refresh_index,
        global_iteration_start=original.global_iteration_start,
        trajectory_maps=tuple(tuple(maps[:-1]) for maps in original.replay_maps()),
    )
    with pytest.raises(ValueError, match="snapshot callback"):
        production_executor_module._validate_operator_numerical_link(
            first_map,
            expected_link,
            (truncated,),
        )


def test_partial_operator_failure_accounts_completed_map_fidelity_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed later objective retains completed one-pass trajectory work as a lower bound."""
    resolved = _operator_screen_resolved()
    _install_bounded_operator_numerics(resolved, monkeypatch, fail_after_member_calls=8)
    store = ProductionAttemptStore(tmp_path / "operator-partial", resolved.job.content_checksum, 1)
    with pytest.raises(RuntimeError, match="injected operator member failure"):
        production_executor_module._execute_operator_attempt(
            resolved,
            store,
        )
    normalized_work, receipt = production_executor_module._failure_partial_receipts(store)
    assert normalized_work == pytest.approx(8.0)
    assert receipt["normalized_work_is_lower_bound"] is True
    components = cast("list[Mapping[str, object]]", receipt["normalized_work_components"])
    assert components == [
        {
            "kind": "scheduled_map_fidelity_vector",
            "path": next(ref.path for ref in store.written_refs if ref.role == "scheduled_map_evidence"),
            "work": 8.0,
        }
    ]


def test_synthetic_confirmation_is_typed_first_attempt_without_target_materialization(
    production_smoke_suite: _SmokeSuite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic confirmation completes custody without opening a held target manifest."""
    request = _confirm_request()
    fixture = SyntheticConfirmationFixture(request.content_checksum, (0.25, 0.75))
    executor = SyntheticConfirmationExecutor(fixture)

    def forbidden_materialization(*_arguments: object) -> object:
        msg = "synthetic confirmation attempted held target materialization"
        raise AssertionError(msg)

    monkeypatch.setattr(
        production_executor_module,
        "materialize_target_population",
        forbidden_materialization,
    )
    directory = tmp_path / "synthetic-confirm"
    reference = executor.execute(request, directory, _FRESH_CONTROLS)
    reopened = reopen_result_artifact(reference, directory)
    assert reference.artifact_kind == "synthetic_confirmation"
    assert reference.attempt == 1
    assert reopened.evidence.executable_binding_checksum == request.executable_binding_checksum
    assert reopened.evidence.scheduled_program_checksum == request.hyperparameters_checksum
    assert reopened.evidence.derived_metrics["noisy_fidelity"] == pytest.approx(0.5)
    assert reopened.raw_trajectory is not None
    raw_payload = cast("Mapping[str, object]", reopened.raw_trajectory["payload"])
    assert raw_payload["trajectory_fidelities"] == (0.25, 0.75)
    assert executor.execute(request, directory, _FRESH_CONTROLS) == reference
    with pytest.raises(ValueError, match="forbids overwrite"):
        executor.execute(request, directory, JobExecutionControls(resume=False, overwrite=True))
    assert not (directory / "production_attempts" / "attempt_000002").exists()

    registry = create_default_training_executor_registry(
        production_smoke_suite.context,
        synthetic_confirmation_fixture=fixture,
    )
    assert registry.confirm_executor is not None
    assert registry.confirm_executor(request, directory, _FRESH_CONTROLS) == reference.content_checksum


def test_default_confirmation_slot_rejects_without_an_exact_fixture(tmp_path: Path) -> None:
    """The held confirmation slot is present but remains dormant by default."""
    context = _context_with_production_source()
    registry = create_default_training_executor_registry(context)
    assert registry.confirm_executor is not None
    output = tmp_path / "unheld-confirmation"
    with pytest.raises(ValueError, match="dormant"):
        registry.confirm_executor(_confirm_request(), output, _FRESH_CONTROLS)
    assert not output.exists()
