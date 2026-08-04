# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bounded real-confirmation context support for WP22F tests.

The helper creates an actual clean source-locked repository, production
repository bindings, a revealed test-only confirmatory population, and the
complete immutable confirmation plan.  Opaque authorization records are
initialized through a narrow test seam because their full factories are
already covered by the protocol/final-seal tests and would replay 2,376
pilot/screening records for every consumer of this fixture.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from benchmarks.state_preparation.phase2.binding_catalog import RepositoryBindingCatalog
from benchmarks.state_preparation.phase2.execution_bindings import SCREEN_METHOD_IDS
from benchmarks.state_preparation.phase2.execution_context import ConfirmationExecutionContext, ExternalEntropyKeyring
from benchmarks.state_preparation.phase2.execution_protocol import OperatorGrowthExecutionSpec
from benchmarks.state_preparation.phase2.implementation_catalog import RepositoryImplementationCatalog
from benchmarks.state_preparation.phase2.pipeline import TrainingPipelineTemplate
from benchmarks.state_preparation.phase2.protocol import (
    AnalysisSourceManifest,
    ConfirmationAuthorization,
    FinalComparatorRef,
    FinalConfigurationExecutionManifest,
    FinalConfigurationExecutionRef,
    FinalConfirmationSeal,
    PrimaryContrastBinding,
    load_initial_preregistration,
)
from benchmarks.state_preparation.phase2.screening_design import WP22CandidateConfiguration
from benchmarks.state_preparation.phase2.source_lock import (
    WP22_GOVERNED_ANALYSIS_ENTRY_POINT,
    WP22_GOVERNED_ENTRY_POINT,
    ExecutionSourceManifest,
    build_analysis_source_manifest,
    capture_governed_execution_source_manifest,
)
from benchmarks.state_preparation.phase2.targets import (
    TargetMaterializationAuthorization,
    TargetPopulationConfig,
    TargetPopulationManifest,
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.training_orchestration import build_paper_confirm_plan
from tests.benchmarks import test_state_preparation_wp22a_execution_bindings as wp22a_support

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.state_preparation.phase2.execution_bindings import ScopedImplementationBinding

_CONFIRMATORY_ENTROPY = bytes(range(32))
_NORMALIZED_COMPUTE_CAP = 1_000.0


@dataclass(frozen=True, slots=True)
class ConfirmationContextFixture:
    """One bounded, source-locked real-confirmation test universe."""

    repository_root: Path
    context: ConfirmationExecutionContext


def _run_git(repository: Path, *arguments: str) -> None:
    """Run one noninteractive Git command in the bounded test repository."""
    executable = shutil.which("git")
    assert executable is not None
    subprocess.run(  # noqa: S603 -- resolved executable and test-owned arguments
        (executable, "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )


def _set_frozen_slot(instance: object, name: str, value: object) -> None:
    """Set one slot while constructing a deliberately bounded test seam."""
    object.__setattr__(instance, name, value)  # noqa: PLC2801 -- frozen test-only initialization


def _create_source_repository(repository: Path) -> tuple[ExecutionSourceManifest, AnalysisSourceManifest]:
    """Create and capture the smallest complete governed clean checkout.

    Returns:
        The execution- and analysis-source manifests captured from the checkout.
    """
    files = {
        WP22_GOVERNED_ENTRY_POINT: "RUNNER_VERSION = 1\n",
        WP22_GOVERNED_ANALYSIS_ENTRY_POINT: "ANALYSIS_VERSION = 1\n",
        "src/mqt/yaqs/__init__.py": "VERSION = 1\n",
        "benchmarks/state_preparation/phase2/data/initial_preregistration_v1.json": "{}\n",
        "pyproject.toml": "[project]\nname = 'wp22-confirm-test'\nversion = '0'\n",
        "uv.lock": "version = 1\n",
    }
    for relative_path, payload in files.items():
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    _run_git(repository, "init", "--quiet")
    _run_git(repository, "config", "user.name", "WP22 Confirmation Test")
    _run_git(repository, "config", "user.email", "wp22-confirm@example.invalid")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "--quiet", "-m", "initial")
    preregistration = load_initial_preregistration()
    execution = capture_governed_execution_source_manifest(
        repository,
        manifest_id="wp22_confirmation_test_source",
    )
    analysis = build_analysis_source_manifest(
        execution,
        manifest_id="wp22_confirmation_test_analysis",
        preregistration_checksum=preregistration.content_checksum,
        analysis_template_checksum=preregistration.analysis_template_checksum,
        analysis_entry_point=WP22_GOVERNED_ANALYSIS_ENTRY_POINT,
    )
    return execution, analysis


def _candidate_for(binding: ScopedImplementationBinding) -> WP22CandidateConfiguration:
    """Project one real screen binding onto its publication configuration.

    Returns:
        The typed screening configuration addressed by the repository binding.
    """
    artifact = binding.implementation_artifact
    payload = artifact.implementation_payload
    if artifact.implementation_kind == "operator_growth":
        assert isinstance(payload, OperatorGrowthExecutionSpec)
        return WP22CandidateConfiguration(
            method_id=binding.publication_method_id,
            implementation_kind="operator_growth",
            implementation_method_id=binding.publication_method_id,
            implementation_schema_version=payload.schema_version,
            implementation_checksum=payload.content_checksum,
            strategy_schedule_checksum=binding.strategy_schedule.content_checksum,
            resource_stratum_id="primary_cap_12",
            noisy_training=True,
            matching_projection_checksum=None,
            publication_mapping={},
        )
    assert isinstance(payload, TrainingPipelineTemplate)
    return WP22CandidateConfiguration.from_pipeline(
        payload,
        strategy_schedule_checksum=binding.strategy_schedule.content_checksum,
        publication_method_id=binding.publication_method_id,
    )


def _catalog() -> tuple[RepositoryBindingCatalog, tuple[WP22CandidateConfiguration, ...]]:
    """Build the real paper-screen catalog with exact publication roots.

    Returns:
        The executable catalog and its ordered publication configurations.
    """
    candidates: list[WP22CandidateConfiguration] = []
    bindings: list[ScopedImplementationBinding] = []
    for method_id in SCREEN_METHOD_IDS:
        original = wp22a_support._binding(  # noqa: SLF001 -- shared frozen repository-binding fixture
            method_id,
            "primary_q6",
            preset="paper-screen",
            normalized_compute_cap=_NORMALIZED_COMPUTE_CAP,
        )
        candidate = _candidate_for(original)
        binding = replace(
            original,
            publication_candidate_schema_version=candidate.schema_version,
            publication_candidate_checksum=candidate.content_checksum,
            treatment_projection=replace(
                original.treatment_projection,
                publication_candidate_checksum=candidate.content_checksum,
            ),
        )
        candidates.append(candidate)
        bindings.append(binding)
    profile = wp22a_support._profile(tuple(bindings))  # noqa: SLF001 -- shared frozen profile fixture
    implementation_catalog = RepositoryImplementationCatalog.frozen(
        screening_outer_trajectory_count=256,
        smoke_evaluation_trajectory_count=2,
    )
    return RepositoryBindingCatalog.from_profile(profile, implementation_catalog), tuple(candidates)


def _opaque_confirmation_authorization(
    preregistration_checksum: str,
    seal: FinalConfirmationSeal,
    target_manifest: TargetPopulationManifest,
) -> ConfirmationAuthorization:
    """Initialize a checksum-bound token through the explicit test-only seam.

    Returns:
        The exact-type opaque authorization record used by the bounded context.
    """
    authorization = object.__new__(ConfirmationAuthorization)
    _set_frozen_slot(authorization, "preregistration_checksum", preregistration_checksum)
    _set_frozen_slot(authorization, "final_seal_checksum", seal.content_checksum)
    _set_frozen_slot(authorization, "target_manifest_checksum", target_manifest.content_checksum)
    _set_frozen_slot(authorization, "execution_source_checksum", seal.execution_source_checksum)
    return authorization


def _opaque_target_authorization(
    preregistration_checksum: str,
    configuration: TargetPopulationConfig,
    manifest: TargetPopulationManifest,
) -> TargetMaterializationAuthorization:
    """Initialize the exact target token through the explicit test-only seam.

    Returns:
        The exact-type opaque materialization record used by the bounded context.
    """
    authorization = object.__new__(TargetMaterializationAuthorization)
    _set_frozen_slot(authorization, "preregistration_checksum", preregistration_checksum)
    _set_frozen_slot(authorization, "population_config_checksum", configuration.content_checksum)
    _set_frozen_slot(authorization, "target_manifest_checksum", manifest.content_checksum)
    _set_frozen_slot(authorization, "data_role", "confirmatory")
    return authorization


def build_confirmation_context_fixture(tmp_path: Path) -> ConfirmationContextFixture:
    """Build one bounded source-locked context without empirical execution.

    Returns:
        A real repository-bound confirmation context and its clean source root.
    """
    repository = tmp_path / "confirmation-source"
    execution_source, analysis_source = _create_source_repository(repository)
    preregistration = load_initial_preregistration()
    catalog, candidates = _catalog()
    candidate_by_method = {candidate.method_id: candidate for candidate in candidates}
    baseline = candidate_by_method["layerwise_bmpd_crn_v2"]
    noiseless = candidate_by_method["layerwise_bmpd_noiseless"]
    binding_by_configuration = {binding.binding.publication_candidate_checksum: binding for binding in catalog.bindings}
    selected = (baseline, noiseless)
    configuration_execution_manifest = FinalConfigurationExecutionManifest(
        manifest_id="wp22_confirmation_test_configurations",
        entries=tuple(
            sorted(
                (
                    FinalConfigurationExecutionRef(
                        method_id=candidate.method_id,
                        configuration_schema_version=candidate.schema_version,
                        configuration_checksum=candidate.content_checksum,
                        strategy_schedule=binding_by_configuration[
                            candidate.content_checksum
                        ].binding.strategy_schedule,
                        implementation_checksum=binding_by_configuration[
                            candidate.content_checksum
                        ].binding.implementation_checksum,
                        scoped_binding_checksum=binding_by_configuration[
                            candidate.content_checksum
                        ].binding.content_checksum,
                        executable_binding_checksum=binding_by_configuration[
                            candidate.content_checksum
                        ].content_checksum,
                    )
                    for candidate in selected
                ),
                key=lambda item: (item.configuration_checksum, item.method_id),
            )
        ),
    )
    target_configuration = build_target_population_config(
        preregistration,
        "confirmatory",
        role_master_entropy_commitment=role_master_entropy_commitment(_CONFIRMATORY_ENTROPY),
        confirmatory_target_count_by_family={
            "gaussian_amplitude": 24,
            "tfim_ground_state": 24,
            "haar_random": 24,
            "random_mps": 24,
        },
        population_scope="primary_q6",
    )
    target_manifest = create_target_population_manifest(
        target_configuration,
        preregistration,
        _CONFIRMATORY_ENTROPY,
    )
    matching = cast("str", baseline.matching_projection_checksum)
    seal = FinalConfirmationSeal(
        seal_id="wp22_confirmation_test_seal",
        preregistration_checksum=preregistration.content_checksum,
        promotion_decision_checksum=baseline.content_checksum,
        promoted_method_id=baseline.method_id,
        promoted_configuration_checksum=baseline.content_checksum,
        comparators=(
            FinalComparatorRef(
                role="matched_noiseless_control",
                method_id=noiseless.method_id,
                configuration_schema_version=noiseless.schema_version,
                configuration_checksum=noiseless.content_checksum,
                matched_to_configuration_checksum=baseline.content_checksum,
                matching_projection_checksum=matching,
            ),
        ),
        primary_contrasts=(
            PrimaryContrastBinding(
                contrast_id="noisy_vs_noiseless",
                treatment_configuration_checksum=baseline.content_checksum,
                control_configuration_checksum=noiseless.content_checksum,
                paired_block_policy_checksum=preregistration.paired_block_policy_checksum,
                matching_projection_checksum=matching,
            ),
        ),
        confirmatory_target_manifest_checksum=target_manifest.content_checksum,
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
            "metric": preregistration.primary_resource_constraint["metric"],
            "cap_per_chain_edge": preregistration.primary_resource_constraint["cap_per_chain_edge"],
            "normalized_compute_cap": _NORMALIZED_COMPUTE_CAP,
            "reachable_stratum_manifest_checksum": baseline.implementation_checksum,
        },
        hyperparameters_checksum=configuration_execution_manifest.content_checksum,
        execution_source_checksum=execution_source.content_checksum,
        analysis_template_checksum=preregistration.analysis_template_checksum,
        analysis_source_manifest_checksum=analysis_source.content_checksum,
        sample_size_design_checksum=noiseless.implementation_checksum,
        failure_policy_checksum=preregistration.failure_policy_checksum,
    )
    confirmation_authorization = _opaque_confirmation_authorization(
        preregistration.content_checksum,
        seal,
        target_manifest,
    )
    target_authorization = _opaque_target_authorization(
        preregistration.content_checksum,
        target_configuration,
        target_manifest,
    )
    plan = build_paper_confirm_plan(
        seal=seal,
        target_manifest=target_manifest,
        configuration_execution_manifest=configuration_execution_manifest,
    )
    context = ConfirmationExecutionContext(
        plan=plan,
        preregistration=preregistration,
        final_seal=seal,
        configuration_execution_manifest=configuration_execution_manifest,
        execution_source_manifest=execution_source,
        analysis_source_manifest=analysis_source,
        repository_binding_catalog=catalog,
        target_configuration=target_configuration,
        target_manifest=target_manifest,
        confirmation_authorization=confirmation_authorization,
        target_materialization_authorization=target_authorization,
        external_entropy_keyring=ExternalEntropyKeyring({
            ("confirmatory", "primary_q6"): _CONFIRMATORY_ENTROPY,
        }),
    )
    return ConfirmationContextFixture(repository_root=repository, context=context)


__all__ = ["ConfirmationContextFixture", "build_confirmation_context_fixture"]
