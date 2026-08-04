# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Execution-source locking and final-seal linkage tests for WP22."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.pilot import (
    PilotNuisanceSummary,
    build_cluster_aware_paired_difference_v1,
    reestimate_cluster_aware_paired_difference_v1,
)
from benchmarks.state_preparation.phase2.protocol import (
    AnalysisSourceManifest,
    FinalComparatorRef,
    FinalConfirmationSeal,
    PrimaryContrastBinding,
    load_initial_preregistration,
)
from benchmarks.state_preparation.phase2.screening import (
    PilotNormalizedComputeCalibration,
    ProductionResourceCalibration,
    ProductionResourceProjection,
    build_final_configuration_execution_manifest,
    build_pilot_normalized_compute_calibration,
    build_production_resource_calibration,
    build_screening_manifest,
    create_final_confirmation_seal,
)
from benchmarks.state_preparation.phase2.source_lock import (
    EXECUTION_SOURCE_ROLES,
    WP22_GOVERNED_ANALYSIS_ENTRY_POINT,
    WP22_GOVERNED_ENTRY_POINT,
    ExecutionSourceFileRef,
    ExecutionSourceManifest,
    build_analysis_source_manifest,
    capture_execution_source_manifest,
    capture_governed_execution_source_manifest,
    verify_analysis_source_bridge,
    verify_execution_source_manifest,
    verify_final_seal_source_lock,
)
from benchmarks.state_preparation.phase2.targets import (
    TargetPopulationCommitment,
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from tests.benchmarks.wp22_pilot_test_support import (
    build_pilot_summary,
    pilot_observations,
    production_pilot_custody_fixture,
)
from tests.benchmarks.wp22_screening_test_support import (
    candidate_configurations,
    production_screening_custody_fixture,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from benchmarks.state_preparation.phase2.protocol import InitialPreregistration


def _run_git(repository: Path, *arguments: str) -> str:
    """Run Git in an isolated test repository and return stripped stdout.

    Args:
        repository: Exact test repository root.
        arguments: Git subcommand and arguments.

    Returns:
        Stripped standard output.
    """
    executable = shutil.which("git")
    assert executable is not None
    completed = subprocess.run(  # noqa: S603 -- resolved executable
        (executable, "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def source_repository(tmp_path: Path) -> Path:
    """Create one clean repository covering every final source-lock role.

    Args:
        tmp_path: Pytest-managed temporary directory.

    Returns:
        Clean committed repository root.
    """
    repository = tmp_path / "repository"
    (repository / "src").mkdir(parents=True)
    (repository / "analysis").mkdir()
    (repository / "config").mkdir()
    (repository / "benchmarks" / "state_preparation" / "phase2" / "data").mkdir(parents=True)
    (repository / "src" / "mqt" / "yaqs").mkdir(parents=True)
    (repository / "src" / "runner.py").write_text("RUNNER_VERSION = 1\n", encoding="utf-8")
    (repository / "src" / "method.py").write_text("METHOD_VERSION = 1\n", encoding="utf-8")
    (repository / "analysis" / "primary.py").write_text("ANALYSIS_VERSION = 1\n", encoding="utf-8")
    (repository / "analysis" / "statistics.py").write_text("STATISTICS_VERSION = 1\n", encoding="utf-8")
    (repository / "config" / "protocol.json").write_text('{"protocol":1}\n', encoding="utf-8")
    (repository / WP22_GOVERNED_ENTRY_POINT).write_text("RUNNER_VERSION = 1\n", encoding="utf-8")
    (repository / WP22_GOVERNED_ANALYSIS_ENTRY_POINT).write_text("ANALYSIS_VERSION = 1\n", encoding="utf-8")
    (repository / "src" / "mqt" / "yaqs" / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    (
        repository / "benchmarks" / "state_preparation" / "phase2" / "data" / "initial_preregistration_v1.json"
    ).write_text('{"protocol":1}\n', encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname = 'wp22-test'\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _run_git(repository, "init", "--quiet")
    _run_git(repository, "config", "user.name", "WP22 Test")
    _run_git(repository, "config", "user.email", "wp22@example.invalid")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "--quiet", "-m", "initial")
    return repository


def _capture(repository: Path) -> ExecutionSourceManifest:
    """Capture the standard clean test source universe.

    Args:
        repository: Clean test repository root.

    Returns:
        Complete execution-source manifest.
    """
    return capture_governed_execution_source_manifest(
        repository,
        manifest_id="phase2_execution_v1",
    )


def _capture_generic(repository: Path) -> ExecutionSourceManifest:
    """Capture the caller-selected source universe used by low-level API tests.

    Returns:
        The generic source manifest.
    """
    return capture_execution_source_manifest(
        repository,
        manifest_id="phase2_execution_v1",
        entry_point="src/runner.py",
        execution_source_paths=("src/runner.py", "src/method.py"),
        analysis_source_paths=("analysis/statistics.py", "analysis/primary.py"),
        dependency_lock_paths=("uv.lock",),
        sealed_input_paths=("config/protocol.json",),
    )


def _analysis_manifest(execution_manifest: ExecutionSourceManifest) -> AnalysisSourceManifest:
    """Build the existing protocol analysis manifest from one source lock.

    Args:
        execution_manifest: Complete WP22 source lock.

    Returns:
        Exact analysis-source projection.
    """
    return build_analysis_source_manifest(
        execution_manifest,
        manifest_id="phase2_primary_analysis_v1",
        preregistration_checksum=canonical_checksum({"preregistration": 1}),
        analysis_template_checksum=canonical_checksum({"analysis_template": 1}),
        analysis_entry_point=WP22_GOVERNED_ANALYSIS_ENTRY_POINT,
    )


def _final_seal(
    execution_manifest: ExecutionSourceManifest,
    analysis_manifest: AnalysisSourceManifest,
) -> FinalConfirmationSeal:
    """Create a schema-valid final seal carrying the exact source links.

    Args:
        execution_manifest: Complete WP22 source lock.
        analysis_manifest: Existing protocol analysis manifest.

    Returns:
        Final seal suitable for focused source-link verification.
    """
    preregistration = load_initial_preregistration()
    promoted = canonical_checksum({"configuration": "promoted"})
    baseline = canonical_checksum({"configuration": "baseline"})
    noiseless = canonical_checksum({"configuration": "noiseless"})
    matching = canonical_checksum({"matching": "projection"})
    return FinalConfirmationSeal(
        seal_id="phase2_confirmation_v1",
        preregistration_checksum=preregistration.content_checksum,
        promotion_decision_checksum=canonical_checksum({"promotion": 1}),
        promoted_method_id="fixed_depth_bmpd_crn",
        promoted_configuration_checksum=promoted,
        comparators=(
            FinalComparatorRef(
                role="layerwise_v2_reference",
                method_id="layerwise_bmpd_crn_v2",
                configuration_schema_version="phase2_pipeline_v1",
                configuration_checksum=baseline,
                matched_to_configuration_checksum=noiseless,
                matching_projection_checksum=matching,
            ),
            FinalComparatorRef(
                role="matched_noiseless_control",
                method_id="layerwise_bmpd_noiseless",
                configuration_schema_version="phase2_pipeline_v1",
                configuration_checksum=noiseless,
                matched_to_configuration_checksum=baseline,
                matching_projection_checksum=matching,
            ),
        ),
        primary_contrasts=(
            PrimaryContrastBinding(
                contrast_id="promoted_vs_layerwise_v2_if_distinct",
                treatment_configuration_checksum=promoted,
                control_configuration_checksum=baseline,
                paired_block_policy_checksum=preregistration.paired_block_policy_checksum,
                matching_projection_checksum=None,
            ),
        ),
        confirmatory_target_manifest_checksum=canonical_checksum({"confirmatory_targets": 1}),
        target_count_by_family={
            "gaussian_amplitude": 24,
            "tfim_ground_state": 24,
            "haar_random": 24,
            "random_mps": 24,
        },
        optimization_seed_count=3,
        fixed_test_trajectory_count=256,
        primary_noise_condition=preregistration.primary_noise_condition,
        primary_resource_budget={
            "metric": "native_two_qubit_gates_per_chain_edge",
            "cap_per_chain_edge": 12.0,
            "normalized_compute_cap": 1000.0,
            "reachable_stratum_manifest_checksum": canonical_checksum({"resources": 1}),
        },
        hyperparameters_checksum=canonical_checksum({"hyperparameters": 1}),
        execution_source_checksum=execution_manifest.content_checksum,
        analysis_template_checksum=analysis_manifest.analysis_template_checksum,
        analysis_source_manifest_checksum=analysis_manifest.content_checksum,
        sample_size_design_checksum=canonical_checksum({"sample_size": 1}),
        failure_policy_checksum=preregistration.failure_policy_checksum,
    )


def test_clean_capture_records_exact_head_blobs_roles_and_derived_checksums(
    source_repository: Path,
) -> None:
    """Clean capture must bind every requested file to exact HEAD bytes."""
    manifest = _capture(source_repository)

    assert manifest.source_commit == _run_git(source_repository, "rev-parse", "HEAD")
    assert tuple(source_file.repo_path for source_file in manifest.source_files) == tuple(
        sorted(source_file.repo_path for source_file in manifest.source_files)
    )
    assert {source_file.role for source_file in manifest.source_files} == set(EXECUTION_SOURCE_ROLES)
    assert manifest.entry_point == WP22_GOVERNED_ENTRY_POINT
    assert manifest.clean_worktree is True
    for source_file in manifest.source_files:
        path = source_repository / source_file.repo_path
        assert source_file.git_blob_id == _run_git(
            source_repository,
            "rev-parse",
            f"HEAD:{source_file.repo_path}",
        )
        assert source_file.source_checksum == f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    assert verify_execution_source_manifest(manifest, source_repository) == tuple(
        source_file.repo_path for source_file in manifest.source_files
    )


def test_capture_and_nested_file_documents_round_trip_canonically_and_deterministically(
    source_repository: Path,
) -> None:
    """Input order cannot perturb strict source-lock identities or JSON bytes."""
    first = _capture_generic(source_repository)
    reordered = capture_execution_source_manifest(
        source_repository,
        manifest_id="phase2_execution_v1",
        entry_point=Path("src/runner.py"),
        execution_source_paths=(Path("src/method.py"), Path("src/runner.py")),
        analysis_source_paths=(Path("analysis/primary.py"), Path("analysis/statistics.py")),
        dependency_lock_paths=(Path("uv.lock"),),
        sealed_input_paths=(Path("config/protocol.json"),),
    )

    assert reordered == first
    assert reordered.to_json() == first.to_json()
    assert ExecutionSourceManifest.from_json(first.to_json()) == first
    for source_file in first.source_files:
        assert ExecutionSourceFileRef.from_json(source_file.to_json()) == source_file

    tampered = first.to_dict()
    tampered["entry_point"] = "src/method.py"
    with pytest.raises(ValueError, match="content checksum mismatch"):
        ExecutionSourceManifest.from_dict(tampered)


@pytest.mark.parametrize("dirty_kind", ["modified", "untracked"])
def test_dirty_or_untracked_checkout_rejects_before_any_governed_file_read(
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    dirty_kind: str,
) -> None:
    """The pre-read cleanliness guard must run before science bytes are opened."""
    if dirty_kind == "modified":
        (source_repository / "config" / "protocol.json").write_text('{"protocol":2}\n', encoding="utf-8")
    else:
        (source_repository / "untracked.txt").write_text("not sealed\n", encoding="utf-8")
    reads: list[Path] = []
    original_read_bytes: Callable[[Path], bytes] = Path.read_bytes

    def record_read(path: Path) -> bytes:
        reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)

    with pytest.raises(ValueError, match="exactly clean worktree"):
        _capture(source_repository)
    assert reads == []


@pytest.mark.parametrize(
    "invalid_path",
    ["../outside.py", "/absolute.py", "src/../runner.py", "src\\runner.py"],
)
def test_capture_rejects_path_traversal_and_noncanonical_spelling(
    source_repository: Path,
    invalid_path: str,
) -> None:
    """Only normalized repository-relative POSIX paths may enter the lock."""
    with pytest.raises(ValueError, match="normalized relative POSIX path"):
        capture_execution_source_manifest(
            source_repository,
            manifest_id="phase2_execution_v1",
            entry_point="src/runner.py",
            execution_source_paths=("src/runner.py", invalid_path),
            analysis_source_paths=("analysis/primary.py",),
            dependency_lock_paths=("uv.lock",),
            sealed_input_paths=("config/protocol.json",),
        )


def test_capture_rejects_untracked_missing_duplicate_and_nested_repository_root(
    source_repository: Path,
) -> None:
    """Every source must have one role and one exact tracked HEAD tree entry."""
    with pytest.raises(ValueError, match="tracked file at HEAD"):
        capture_execution_source_manifest(
            source_repository,
            manifest_id="phase2_execution_v1",
            entry_point="src/runner.py",
            execution_source_paths=("src/runner.py", "src/missing.py"),
            analysis_source_paths=("analysis/primary.py",),
            dependency_lock_paths=("uv.lock",),
            sealed_input_paths=("config/protocol.json",),
        )
    with pytest.raises(ValueError, match="exactly one final-lock role"):
        capture_execution_source_manifest(
            source_repository,
            manifest_id="phase2_execution_v1",
            entry_point="src/runner.py",
            execution_source_paths=("src/runner.py",),
            analysis_source_paths=("analysis/primary.py",),
            dependency_lock_paths=("uv.lock",),
            sealed_input_paths=("config/protocol.json", "uv.lock"),
        )
    with pytest.raises(ValueError, match="exact top level"):
        _capture(source_repository / "src")


def test_capture_rejects_a_clean_committed_symbolic_link(
    source_repository: Path,
) -> None:
    """A Git symlink blob is not a tracked regular executable source file."""
    link = source_repository / "analysis" / "linked.py"
    link.symlink_to("primary.py")
    _run_git(source_repository, "add", "analysis/linked.py")
    _run_git(source_repository, "commit", "--quiet", "-m", "add link")

    with pytest.raises(ValueError, match="tracked regular file"):
        capture_execution_source_manifest(
            source_repository,
            manifest_id="phase2_execution_v1",
            entry_point="src/runner.py",
            execution_source_paths=("src/runner.py",),
            analysis_source_paths=("analysis/linked.py",),
            dependency_lock_paths=("uv.lock",),
            sealed_input_paths=("config/protocol.json",),
        )


def test_verification_rejects_worktree_mutation_and_new_committed_head(
    source_repository: Path,
) -> None:
    """Neither dirty bytes nor a different clean commit may replay a source lock."""
    manifest = _capture(source_repository)
    method = source_repository / "src" / "method.py"
    method.write_text("METHOD_VERSION = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly clean worktree"):
        verify_execution_source_manifest(manifest, source_repository)

    _run_git(source_repository, "add", "src/method.py")
    _run_git(source_repository, "commit", "--quiet", "-m", "change method")
    with pytest.raises(ValueError, match="HEAD differs"):
        verify_execution_source_manifest(manifest, source_repository)


def test_capture_rechecks_cleanliness_after_reading_the_source_universe(
    source_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent mutation during capture must be caught by the closing guard."""
    original_read_bytes: Callable[[Path], bytes] = Path.read_bytes
    calls = 0

    def mutate_after_first(path: Path) -> bytes:
        nonlocal calls
        result = original_read_bytes(path)
        calls += 1
        if calls == 1:
            (source_repository / "late-untracked.txt").write_text("late mutation\n", encoding="utf-8")
        return result

    monkeypatch.setattr(Path, "read_bytes", mutate_after_first)

    with pytest.raises(ValueError, match="exactly clean worktree"):
        _capture(source_repository)


def test_analysis_bridge_is_an_exact_projection_and_verifies_committed_bytes(
    source_repository: Path,
) -> None:
    """The WP15 analysis manifest must be derived only from locked analysis files."""
    execution_manifest = _capture(source_repository)
    analysis_manifest = _analysis_manifest(execution_manifest)

    assert analysis_manifest.source_commit == execution_manifest.source_commit
    assert analysis_manifest.environment_lock_checksum == execution_manifest.environment_lock_checksum
    assert analysis_manifest.execution_source_manifest_checksum == execution_manifest.content_checksum
    assert tuple(source_file.repo_path for source_file in analysis_manifest.source_files) == (
        WP22_GOVERNED_ANALYSIS_ENTRY_POINT,
    )
    assert verify_analysis_source_bridge(
        execution_manifest,
        analysis_manifest,
        source_repository,
    ) == (WP22_GOVERNED_ANALYSIS_ENTRY_POINT,)

    changed_link = replace(
        analysis_manifest,
        execution_source_manifest_checksum=canonical_checksum({"other_execution": 1}),
    )
    with pytest.raises(ValueError, match="exact analysis_source projection"):
        verify_analysis_source_bridge(execution_manifest, changed_link, source_repository)


def test_existing_final_seal_fields_bind_both_source_manifests(
    source_repository: Path,
) -> None:
    """Final-seal source checksums must close the execution/analysis linkage chain."""
    execution_manifest = _capture(source_repository)
    analysis_manifest = _analysis_manifest(execution_manifest)
    seal = _final_seal(execution_manifest, analysis_manifest)

    assert seal.execution_source_checksum == execution_manifest.content_checksum
    assert seal.analysis_source_manifest_checksum == analysis_manifest.content_checksum
    assert verify_final_seal_source_lock(
        seal,
        execution_manifest,
        analysis_manifest,
        source_repository,
    ) == (WP22_GOVERNED_ANALYSIS_ENTRY_POINT,)

    wrong_execution = replace(
        seal,
        execution_source_checksum=canonical_checksum({"other_execution": 1}),
    )
    with pytest.raises(ValueError, match="exact execution-source manifest"):
        verify_final_seal_source_lock(
            wrong_execution,
            execution_manifest,
            analysis_manifest,
            source_repository,
        )
    wrong_analysis = replace(
        seal,
        analysis_source_manifest_checksum=canonical_checksum({"other_analysis": 1}),
    )
    with pytest.raises(ValueError, match="exact analysis-source manifest"):
        verify_final_seal_source_lock(
            wrong_analysis,
            execution_manifest,
            analysis_manifest,
            source_repository,
        )


def _pilot_nuisance_summary(preregistration: InitialPreregistration) -> PilotNuisanceSummary:
    """Build balanced five-seed pilot evidence for final-seal derivation.

    Returns:
        The checksum-sealed nuisance summary.
    """
    assert preregistration == load_initial_preregistration()
    return build_pilot_summary(pilot_observations())


def test_final_seal_factory_closes_promotion_pilot_target_and_clean_source_roots(
    source_repository: Path,
) -> None:
    """The WP22 factory creates one fully authorized, reproducible seal checksum."""
    preregistration = load_initial_preregistration()
    execution_manifest = _capture(source_repository)
    analysis_manifest = build_analysis_source_manifest(
        execution_manifest,
        manifest_id="phase2_primary_analysis_v1",
        preregistration_checksum=preregistration.content_checksum,
        analysis_template_checksum=preregistration.analysis_template_checksum,
        analysis_entry_point=WP22_GOVERNED_ANALYSIS_ENTRY_POINT,
    )
    candidate_sources = candidate_configurations(preregistration)
    screening_master = bytes(reversed(range(32)))
    screening_target_config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(screening_master),
        population_scope="primary_q6",
    )
    screening_targets = create_target_population_manifest(
        screening_target_config,
        preregistration,
        screening_master,
    )
    screening_manifest = build_screening_manifest(
        preregistration,
        screening_targets,
        candidate_sources,
        optimization_seeds=(101, 202, 303),
        screening_seed_root=10_000,
        manifest_id="wp22_final_seal_factory_screen",
    )
    pilot_summary = _pilot_nuisance_summary(preregistration)
    design = build_cluster_aware_paired_difference_v1(
        preregistration,
        pilot_summary,
        design_id="wp22_final_seal_factory_design",
    )
    target_commitment = TargetPopulationCommitment(
        target_manifest_checksum=canonical_checksum({"targets": "confirmatory"}),
        target_count_by_family=design.target_count_by_family,
    )
    production_custody = production_pilot_custody_fixture(
        source_repository.parent / "pilot-custody",
        execution_source_manifest=execution_manifest,
    )
    pilot_calibration = build_pilot_normalized_compute_calibration(production_custody)
    assert PilotNormalizedComputeCalibration.from_json(pilot_calibration.to_json()) == pilot_calibration
    assert pilot_calibration.normalized_compute_cap == pytest.approx(1_000.0)
    screening_custody = production_screening_custody_fixture(
        preregistration,
        screening_manifest,
        screening_targets,
        design,
        execution_manifest,
        normalized_compute_cap=pilot_calibration.normalized_compute_cap,
    )
    _production_evidence, production_promotion = screening_custody.build_evidence()
    configuration_execution_manifest = build_final_configuration_execution_manifest(
        screening_custody,
        production_promotion,
    )
    production_records = screening_custody.records
    production_seal_arguments = {
        "preregistration": preregistration,
        "screening_manifest": screening_manifest,
        "promotion_decision": production_promotion,
        "pilot_nuisance_summary": pilot_summary,
        "sample_size_design": design,
        "confirmatory_target_commitment": target_commitment,
        "analysis_source_manifest": analysis_manifest,
        "execution_source_manifest": execution_manifest,
        "configuration_execution_manifest": configuration_execution_manifest,
        "repository_root": source_repository,
        "production_screening_custody": screening_custody,
        "production_pilot_custody": production_custody,
    }
    factory = cast("Callable[..., FinalConfirmationSeal]", create_final_confirmation_seal)
    production_seal = factory(**production_seal_arguments)
    resource_calibration = build_production_resource_calibration(production_custody, screening_custody)
    resource_calibration_json = resource_calibration.to_json()
    decoded_resource_calibration = ProductionResourceCalibration.from_json(resource_calibration_json)
    assert decoded_resource_calibration == resource_calibration
    assert (
        ProductionResourceProjection.from_dict(resource_calibration.pilot_q6_resources[0].to_dict())
        == resource_calibration.pilot_q6_resources[0]
    )
    forged_calibration = resource_calibration.to_dict()
    forged_calibration["normalized_compute_cap"] = 999.0
    forged_calibration["content_checksum"] = canonical_checksum({
        key: value for key, value in forged_calibration.items() if key != "content_checksum"
    })
    with pytest.raises(ValueError, match="maximum verified primary-q6 pilot work"):
        ProductionResourceCalibration.from_dict(forged_calibration)
    wrong_cap_screening_custody = production_screening_custody_fixture(
        preregistration,
        screening_manifest,
        screening_targets,
        design,
        execution_manifest,
        normalized_compute_cap=pilot_calibration.normalized_compute_cap - 1.0,
    )
    with pytest.raises(ValueError, match="unique pilot-calibrated normalized compute cap"):
        build_production_resource_calibration(production_custody, wrong_cap_screening_custody)
    assert production_seal.hyperparameters_checksum == configuration_execution_manifest.content_checksum
    assert production_seal.primary_resource_budget["normalized_compute_cap"] == pytest.approx(1_000.0)
    assert (
        production_seal.primary_resource_budget["reachable_stratum_manifest_checksum"]
        == resource_calibration.content_checksum
    )
    assert production_seal.sample_size_design_checksum == design.content_checksum

    with pytest.raises(TypeError, match="screening_source_records"):
        factory(**{**production_seal_arguments, "screening_source_records": (object(),)})
    with pytest.raises(TypeError, match="screening_source_records"):
        factory(**{
            **production_seal_arguments,
            "screening_source_records": (object(), production_records[0]),
        })
    with pytest.raises(TypeError, match="screening_source_records"):
        factory(**{**production_seal_arguments, "screening_source_records": production_records})

    changed_q12_custody = production_pilot_custody_fixture(
        source_repository.parent / "pilot-custody-q12-change",
        execution_source_manifest=execution_manifest,
        secondary_archive_marker="changed-secondary-archive",
    )
    assert changed_q12_custody.secondary_archive_checksum != production_custody.secondary_archive_checksum
    changed_q12_seal = factory(**{
        **production_seal_arguments,
        "production_pilot_custody": changed_q12_custody,
    })
    assert changed_q12_seal.to_json() == production_seal.to_json()

    forged_designs = (
        replace(
            design,
            expected_overall_failure_rate_half_width=design.expected_overall_failure_rate_half_width / 2,
        ),
        replace(design, calculation_source_checksum=canonical_checksum({"calculator": "forged"})),
    )
    for forged_design in forged_designs:
        with pytest.raises(ValueError, match=r"final-seal protocol, manifest, or design"):
            factory(**{**production_seal_arguments, "sample_size_design": forged_design})

    with pytest.raises(ValueError, match="initial sample-size design cannot have a parent"):
        factory(**{**production_seal_arguments, "parent_sample_size_design": design})

    reestimated_design = reestimate_cluster_aware_paired_difference_v1(
        preregistration,
        pilot_summary,
        design,
        information_fraction=0.5,
        design_id="wp22_final_seal_factory_reestimated_design",
    )
    reestimated_commitment = TargetPopulationCommitment(
        target_manifest_checksum=target_commitment.target_manifest_checksum,
        target_count_by_family=reestimated_design.target_count_by_family,
    )
    reestimated_screening_custody = production_screening_custody_fixture(
        preregistration,
        screening_manifest,
        screening_targets,
        reestimated_design,
        execution_manifest,
        normalized_compute_cap=pilot_calibration.normalized_compute_cap,
    )
    _reestimated_evidence, reestimated_promotion = reestimated_screening_custody.build_evidence()
    reestimated_execution_manifest = build_final_configuration_execution_manifest(
        reestimated_screening_custody,
        reestimated_promotion,
    )
    reestimated_arguments = {
        **production_seal_arguments,
        "production_screening_custody": reestimated_screening_custody,
        "promotion_decision": reestimated_promotion,
        "configuration_execution_manifest": reestimated_execution_manifest,
        "sample_size_design": reestimated_design,
        "confirmatory_target_commitment": reestimated_commitment,
        "parent_sample_size_design": design,
    }
    reestimated_seal = factory(**reestimated_arguments)
    assert reestimated_seal.sample_size_design_checksum == reestimated_design.content_checksum

    without_parent = {key: value for key, value in reestimated_arguments.items() if key != "parent_sample_size_design"}
    with pytest.raises(ValueError, match="requires its initial parent design"):
        factory(**without_parent)
    other_parent = replace(design, design_id="wp22_other_initial_design")
    with pytest.raises(ValueError, match=r"exact result.*pilot nuisance evidence and frozen source"):
        factory(**{**reestimated_arguments, "parent_sample_size_design": other_parent})

    first_observation = next(
        observation
        for observation in pilot_summary.observations
        if observation.contrast_id == "promoted_vs_layerwise_v2_if_distinct"
    )
    assert first_observation.treatment_result.evaluation_evidence is not None
    evaluation_evidence = first_observation.treatment_result.evaluation_evidence
    fidelity = evaluation_evidence.fresh_test_noisy_fidelity
    changed_evidence = replace(
        evaluation_evidence,
        fresh_test_trajectory_fidelities=tuple(
            fidelity + 2 * (value - fidelity) for value in evaluation_evidence.fresh_test_trajectory_fidelities
        ),
    )
    changed_treatment_result = replace(
        first_observation.treatment_result,
        evaluation_evidence=changed_evidence,
    )
    changed_treatment_outcome = replace(
        first_observation.treatment_outcome,
        result_artifact_checksum=changed_treatment_result.content_checksum,
    )
    changed_observation = replace(
        first_observation,
        treatment_result=changed_treatment_result,
        treatment_outcome=changed_treatment_outcome,
    )
    changed_pilot = build_pilot_summary(
        tuple(
            changed_observation if observation.identity == first_observation.identity else observation
            for observation in pilot_summary.observations
        ),
        summary_id=pilot_summary.summary_id,
    )
    with pytest.raises(ValueError, match="exact q6 inference projection of production custody"):
        factory(**{**production_seal_arguments, "pilot_nuisance_summary": changed_pilot})

    changed_entries = list(configuration_execution_manifest.entries)
    changed_entries[0] = replace(
        changed_entries[0],
        strategy_schedule=replace(
            changed_entries[0].strategy_schedule,
            schedule_id="forged_final_configuration_schedule",
        ),
    )
    changed_entries.sort(key=lambda item: (item.configuration_checksum, item.method_id))
    forged_execution_manifest = replace(
        configuration_execution_manifest,
        entries=tuple(changed_entries),
    )
    with pytest.raises(ValueError, match="exact screened bindings and schedules"):
        factory(**{
            **production_seal_arguments,
            "configuration_execution_manifest": forged_execution_manifest,
        })

    foreign_execution = replace(execution_manifest, manifest_id="foreign_execution_source")
    foreign_pilot = production_pilot_custody_fixture(
        source_repository.parent / "foreign-pilot-custody",
        execution_source_manifest=foreign_execution,
    )
    foreign_pilot_calibration = build_pilot_normalized_compute_calibration(foreign_pilot)
    foreign_screening = production_screening_custody_fixture(
        preregistration,
        screening_manifest,
        screening_targets,
        design,
        foreign_execution,
        normalized_compute_cap=foreign_pilot_calibration.normalized_compute_cap,
    )
    foreign_configuration_manifest = build_final_configuration_execution_manifest(
        foreign_screening,
        production_promotion,
    )
    with pytest.raises(ValueError, match="foreign execution-source manifest"):
        factory(**{
            **production_seal_arguments,
            "production_screening_custody": foreign_screening,
            "production_pilot_custody": foreign_pilot,
            "configuration_execution_manifest": foreign_configuration_manifest,
        })

    with pytest.raises(TypeError, match="screening_evidence"):
        factory(**{**production_seal_arguments, "screening_evidence": object()})

    assert production_seal.promoted_method_id == "fixed_depth_bmpd_crn"
    assert production_seal.confirmatory_target_manifest_checksum == target_commitment.target_manifest_checksum
    assert production_seal.execution_source_checksum == execution_manifest.content_checksum
    assert production_seal.analysis_source_manifest_checksum == analysis_manifest.content_checksum
    assert production_seal.sample_size_design_checksum == design.content_checksum
    assert FinalConfirmationSeal.from_json(production_seal.to_json()) == production_seal
