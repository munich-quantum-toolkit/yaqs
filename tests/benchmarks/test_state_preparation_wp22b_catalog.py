# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused tests for the executable WP22B implementation catalog."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from benchmarks.state_preparation import phase2
from benchmarks.state_preparation.phase2.artifacts import StageExecutionEvidence
from benchmarks.state_preparation.phase2.execution_bindings import (
    PILOT_METHOD_IDS,
    SCREEN_METHOD_IDS,
    SMOKE_METHOD_IDS,
)
from benchmarks.state_preparation.phase2.implementation_catalog import (
    IMPLEMENTATION_CATALOG_ENTRY_COUNT,
    PILOT_ENTRY_COUNT,
    SCREEN_ENTRY_COUNT,
    SMOKE_ENTRY_COUNT,
    ExecutableImplementationEntry,
    OperatorGrowthSmokeExecution,
    OperatorGrowthSmokeRuntimeProgram,
    PipelineSmokeRuntimeProgram,
    RepositoryImplementationCatalog,
    RepositoryRunnerAdapter,
)
from benchmarks.state_preparation.phase2.noisy_krotov import NoisyKrotovStageExecution
from benchmarks.state_preparation.phase2.protocol import load_initial_preregistration
from benchmarks.state_preparation.phase2.targets import (
    TargetPopulationManifest,
    TargetPopulationMaterialization,
    authorize_target_materialization,
    build_target_population_config,
    create_target_population_manifest,
    materialize_target_population,
    role_master_entropy_commitment,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from benchmarks.state_preparation.phase2.targets import MaterializedTarget, TargetInstanceSpec
    from benchmarks.state_preparation.phase2.training_schedules import TrainingStrategySchedule


_DEVELOPMENT_ENTROPY = bytes(range(32))


@pytest.fixture(scope="module")
def catalog() -> RepositoryImplementationCatalog:
    """Return one complete fixed-count repository catalog."""
    return RepositoryImplementationCatalog.frozen(screening_outer_trajectory_count=256)


@pytest.fixture(scope="module")
def development_population() -> tuple[TargetPopulationManifest, TargetPopulationMaterialization]:
    """Materialize one genuine development population for runtime dispatch.

    Returns:
        The exact manifest and its authorized target materialization.
    """
    preregistration = load_initial_preregistration()
    config = build_target_population_config(
        preregistration,
        "development",
        role_master_entropy_commitment=role_master_entropy_commitment(_DEVELOPMENT_ENTROPY),
        population_scope="primary_q6",
    )
    manifest = create_target_population_manifest(
        config,
        preregistration,
        _DEVELOPMENT_ENTROPY,
    )
    authorization = authorize_target_materialization(
        preregistration,
        config,
        manifest,
        _DEVELOPMENT_ENTROPY,
    )
    materialization = materialize_target_population(
        config,
        preregistration,
        manifest,
        _DEVELOPMENT_ENTROPY,
        authorization,
    )
    return manifest, materialization


def test_catalog_has_literal_cardinality_operational_routes_and_roundtrip(
    catalog: RepositoryImplementationCatalog,
) -> None:
    """All 25 exact entries carry typed artifacts and concrete callables."""
    by_preset = {
        preset: tuple(entry for entry in catalog.entries if entry.preset == preset)
        for preset in ("training-smoke", "paper-pilot", "paper-screen")
    }

    assert len(catalog.entries) == IMPLEMENTATION_CATALOG_ENTRY_COUNT == 25
    assert len(by_preset["training-smoke"]) == SMOKE_ENTRY_COUNT == 10
    assert len(by_preset["paper-pilot"]) == PILOT_ENTRY_COUNT == 6
    assert len(by_preset["paper-screen"]) == SCREEN_ENTRY_COUNT == 9
    assert {entry.publication_method_id for entry in by_preset["training-smoke"]} == set(SMOKE_METHOD_IDS)
    assert {entry.publication_method_id for entry in by_preset["paper-screen"]} == set(SCREEN_METHOD_IDS)
    assert all(callable(entry.resolve_callable()) for entry in catalog.entries)
    assert all(entry.implementation_artifact.implementation_payload is not None for entry in catalog.entries)
    assert RepositoryImplementationCatalog.from_json(catalog.to_json()) == catalog


def test_wp22b_catalogs_are_public_phase2_apis() -> None:
    """Both repository catalog layers and runtime envelopes are exported."""
    assert phase2.RepositoryImplementationCatalog is RepositoryImplementationCatalog
    assert phase2.RepositoryRunnerAdapter is RepositoryRunnerAdapter
    assert phase2.PipelineSmokeRuntimeProgram is PipelineSmokeRuntimeProgram
    assert phase2.OperatorGrowthSmokeRuntimeProgram is OperatorGrowthSmokeRuntimeProgram


def test_pilot_widths_screen_scope_and_confirmation_alias_are_exact(
    catalog: RepositoryImplementationCatalog,
) -> None:
    """q12 stays pilot-only and confirmation reuses the q6 screen object."""
    pilot_keys = {
        (entry.publication_method_id, entry.target_scope_id)
        for entry in catalog.entries
        if entry.preset == "paper-pilot"
    }
    assert pilot_keys == {(method, scope) for method in PILOT_METHOD_IDS for scope in ("primary_q6", "secondary_q12")}
    assert all(
        entry.target_scope_id == "primary_q6"
        for entry in catalog.entries
        if entry.preset in {"training-smoke", "paper-screen"}
    )

    screened = catalog.resolve("paper-screen", "layerwise_bmpd_crn_v2", "primary_q6")
    confirmation = catalog.resolve("paper-confirm", "layerwise_bmpd_crn_v2", "primary_q6")
    assert confirmation is screened
    assert confirmation.implementation_artifact is screened.implementation_artifact
    assert confirmation.implementation_artifact.content_checksum == screened.implementation_artifact.content_checksum
    assert confirmation.preset == "paper-screen"

    with pytest.raises(KeyError, match="promotion-eligible q6"):
        catalog.resolve("paper-confirm", "layerwise_bmpd_noiseless", "primary_q6")
    with pytest.raises(KeyError, match="promotion-eligible q6"):
        catalog.resolve("paper-confirm", PILOT_METHOD_IDS[0], "secondary_q12")
    with pytest.raises(KeyError, match="No unique executable"):
        catalog.resolve("paper-screen", PILOT_METHOD_IDS[0], "secondary_q12")


def test_catalog_rejects_missing_duplicate_cross_preset_and_scope_substitution(
    catalog: RepositoryImplementationCatalog,
) -> None:
    """Malformed universes fail during construction, before resolving a runner."""
    with pytest.raises(ValueError, match="exactly 10 smoke, 6 pilot, and 9"):
        replace(catalog, entries=catalog.entries[:-1])
    with pytest.raises(ValueError, match="keys must be unique"):
        replace(catalog, entries=(*catalog.entries, catalog.entries[0]))

    pilot_q6 = catalog.resolve("paper-pilot", PILOT_METHOD_IDS[0], "primary_q6")
    pilot_q12 = catalog.resolve("paper-pilot", PILOT_METHOD_IDS[0], "secondary_q12")
    with pytest.raises(ValueError, match="Catalog key, strategy schedule, and implementation artifact disagree"):
        replace(pilot_q6, preset="paper-screen")
    with pytest.raises(ValueError, match="Catalog key, strategy schedule, and implementation artifact disagree"):
        replace(
            pilot_q12,
            implementation_artifact=pilot_q6.implementation_artifact,
            runner_adapter=pilot_q6.runner_adapter,
        )


def test_catalog_rejects_an_internally_resealed_schedule_treatment_substitution(
    catalog: RepositoryImplementationCatalog,
) -> None:
    """Canonical universe closure rejects a noiseless schedule on a noisy runtime."""
    noisy = catalog.resolve(
        "training-smoke",
        "layerwise_bmpd_crn_v2",
        "primary_q6",
    )
    noiseless_schedule = catalog.resolve(
        "training-smoke",
        "layerwise_bmpd_noiseless",
        "primary_q6",
    ).strategy_schedule
    resealed_artifact = replace(
        noisy.implementation_artifact,
        strategy_schedule_checksum=noiseless_schedule.content_checksum,
    )
    resealed_entry = replace(
        noisy,
        strategy_schedule=noiseless_schedule,
        implementation_artifact=resealed_artifact,
        runner_adapter=RepositoryRunnerAdapter.for_artifact(resealed_artifact),
    )
    forged_entries = tuple(resealed_entry if entry.key == noisy.key else entry for entry in catalog.entries)

    assert resealed_entry.strategy_schedule.training_noise.mode == "noiseless"
    assert resealed_entry.smoke_runtime_program().training_trajectory_count == 1
    with pytest.raises(ValueError, match="exact ordered canonical"):
        replace(catalog, entries=forged_entries)


def test_operator_and_runner_substitutions_fail_closed(catalog: RepositoryImplementationCatalog) -> None:
    """Operator artifacts and runner-family routes cannot cross publication methods."""
    operator = catalog.resolve("paper-screen", "adapt_style_state_preparation", "primary_q6")
    pipeline = catalog.resolve("paper-screen", "layerwise_bmpd_crn_v2", "primary_q6")

    with pytest.raises(ValueError, match="Catalog key, strategy schedule, and implementation artifact disagree"):
        replace(
            operator,
            implementation_artifact=pipeline.implementation_artifact,
            runner_adapter=pipeline.runner_adapter,
        )
    with pytest.raises(ValueError, match="Runner adapter does not bind"):
        replace(operator, runner_adapter=pipeline.runner_adapter)

    forged_adapter = RepositoryRunnerAdapter(
        publication_method_id=operator.publication_method_id,
        target_scope_id=operator.target_scope_id,
        implementation_kind=operator.implementation_artifact.implementation_kind,
        implementation_payload_checksum=pipeline.implementation_artifact.implementation_payload_checksum,
    )
    with pytest.raises(ValueError, match="Runner adapter does not bind"):
        replace(operator, runner_adapter=forged_adapter)


def test_pipeline_runtime_identity_and_tiny_trajectory_limit_fail_closed(
    catalog: RepositoryImplementationCatalog,
) -> None:
    """A direct runtime forgery cannot widen work or relabel its method."""
    program = catalog.resolve(
        "training-smoke",
        "fixed_depth_bmpd_crn",
        "primary_q6",
    ).smoke_runtime_program()
    assert isinstance(program, PipelineSmokeRuntimeProgram)

    with pytest.raises(ValueError, match="trajectory limit differs"):
        replace(program, training_trajectory_count=2)
    forged_template = replace(
        program.runtime_template,
        template_id="wp22b_smoke_runtime_fixed_depth_bmpd_crn_forged",
    )
    with pytest.raises(ValueError, match="identity, or trajectory limit differs"):
        replace(program, runtime_template=forged_template)

    layerwise = catalog.resolve(
        "training-smoke",
        "layerwise_bmpd_crn_v2",
        "primary_q6",
    ).smoke_runtime_program()
    assert isinstance(layerwise, PipelineSmokeRuntimeProgram)
    policy = dict(layerwise.runtime_template.stages[0].stage_policy)
    optimizer = dict(cast("Mapping[str, object]", policy["optimizer_hyperparameters"]))
    optimizer["learning_rate"] = 0.123
    policy["optimizer_hyperparameters"] = optimizer
    forged_stage = replace(layerwise.runtime_template.stages[0], stage_policy=policy)
    forged_runtime = replace(layerwise.runtime_template, stages=(forged_stage,))
    with pytest.raises(ValueError, match="method, scope, identity, or trajectory limit differs"):
        replace(layerwise, runtime_template=forged_runtime)


def test_entry_requires_typed_schedule_artifact_and_adapter(catalog: RepositoryImplementationCatalog) -> None:
    """Identifier-only stand-ins are rejected at the executable entry boundary."""
    entry = catalog.entries[0]
    with pytest.raises(TypeError, match="strategy_schedule"):
        ExecutableImplementationEntry(
            preset=entry.preset,
            publication_method_id=entry.publication_method_id,
            target_scope_id=entry.target_scope_id,
            strategy_schedule=cast("TrainingStrategySchedule", "checksum-only"),
            implementation_artifact=entry.implementation_artifact,
            runner_adapter=entry.runner_adapter,
        )


def test_all_pipeline_smoke_entries_execute_their_genuine_one_update_runner(
    catalog: RepositoryImplementationCatalog,
    development_population: tuple[TargetPopulationManifest, TargetPopulationMaterialization],
) -> None:
    """Every pipeline family performs one real update with no validation work."""
    manifest, materialization = development_population
    target = materialization.targets[0]
    observed: dict[str, str] = {}
    for entry in catalog.entries:
        if entry.preset != "training-smoke":
            continue
        program = entry.smoke_runtime_program()
        if not isinstance(program, PipelineSmokeRuntimeProgram):
            continue
        bound = program.bind(manifest, target, optimization_seed=17)
        outcome = bound.execute()
        observed[entry.publication_method_id] = type(outcome).__name__

        assert bound.stage.iteration_budget == 1
        assert bound.stage.trajectory_count == program.training_trajectory_count
        assert bound.stage.checkpoint_validation.enabled is False
        assert isinstance(outcome, (NoisyKrotovStageExecution, StageExecutionEvidence))
        if isinstance(outcome, NoisyKrotovStageExecution):
            assert tuple(row.global_iteration for row in outcome.trace) == (0, 1)

    assert observed == {
        "fixed_depth_bmpd_crn": "NoisyKrotovStageExecution",
        "impact_pruning_crn": "NoisyKrotovStageExecution",
        "layerwise_bmpd_crn_v2": "NoisyKrotovStageExecution",
        "layerwise_bmpd_cross_crn": "NoisyKrotovStageExecution",
        "layerwise_bmpd_noiseless": "NoisyKrotovStageExecution",
        "layerwise_bmpd_resampled": "NoisyKrotovStageExecution",
        "parameter_shift_adam_layerwise": "StageExecutionEvidence",
        "spsa_layerwise": "StageExecutionEvidence",
    }


def _tfim_case(
    manifest: TargetPopulationManifest,
    materialization: TargetPopulationMaterialization,
) -> tuple[MaterializedTarget, TargetInstanceSpec]:
    """Return one target-bound TFIM development cell.

    Returns:
        The exact materialized target and its seed-bearing specification.
    """
    spec = next(candidate for candidate in manifest.instances if candidate.family_id == "tfim_ground_state")
    return materialization.target(spec.target_instance_id), spec


def test_operator_smoke_entries_execute_noisy_and_analytic_paths_without_promotion(
    catalog: RepositoryImplementationCatalog,
    development_population: tuple[TargetPopulationManifest, TargetPopulationMaterialization],
) -> None:
    """Both growth callbacks run, but their smoke envelopes remain evidence-excluded."""
    manifest, materialization = development_population
    projector_program = catalog.resolve(
        "training-smoke",
        "adapt_style_state_preparation",
        "primary_q6",
    ).smoke_runtime_program()
    energy_program = catalog.resolve(
        "training-smoke",
        "energy_adapt_vqe",
        "primary_q6",
    ).smoke_runtime_program()
    assert isinstance(projector_program, OperatorGrowthSmokeRuntimeProgram)
    assert isinstance(energy_program, OperatorGrowthSmokeRuntimeProgram)

    projector = projector_program.execute_projector(
        materialization.targets[0],
        optimization_block_id="wp22b_projector_smoke",
        optimization_seed=17,
        resource_stratum_id="primary_cap_12",
        trajectory_seed=23,
    )
    tfim_target, tfim_spec = _tfim_case(manifest, materialization)
    energy = energy_program.execute_energy(tfim_target, tfim_spec)

    assert isinstance(projector, OperatorGrowthSmokeExecution)
    assert projector.execution_mode == "noisy_training"
    assert projector.promotion_eligible is False
    assert len(projector.selected_operator_ids) <= 1
    assert projector.trace_count <= 1
    assert projector.work.total_sampled_trajectories > 0
    assert projector.objective_request_trajectory_counts
    assert set(projector.objective_request_trajectory_counts) == {1}
    assert "result" not in projector.to_dict()

    assert isinstance(energy, OperatorGrowthSmokeExecution)
    assert energy.execution_mode == "analytic_reference"
    assert energy.promotion_eligible is False
    assert len(energy.selected_operator_ids) <= 1
    assert energy.trace_count <= 1
    assert energy.work.total_sampled_trajectories == 0
    assert energy.objective_request_trajectory_counts == ()
    assert energy_program.training_trajectory_count == 0

    with pytest.raises(ValueError, match="Only the Energy-ADAPT"):
        projector_program.execute_energy(tfim_target, tfim_spec)
    with pytest.raises(ValueError, match="Only the projector"):
        energy_program.execute_projector(
            materialization.targets[0],
            optimization_block_id="wrong_route",
            optimization_seed=17,
            resource_stratum_id="primary_cap_12",
            trajectory_seed=23,
        )
