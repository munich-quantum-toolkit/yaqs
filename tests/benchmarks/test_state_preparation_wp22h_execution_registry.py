# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the deterministic WP22H production execution registry compiler."""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import pytest

from benchmarks.state_preparation.phase2.binding_catalog import RepositoryBindingCatalog
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.execution_bindings import (
    PILOT_METHOD_IDS,
    SCREEN_METHOD_IDS,
    TrainingExecutionProfile,
)
from benchmarks.state_preparation.phase2.execution_protocol import OperatorGrowthExecutionSpec
from benchmarks.state_preparation.phase2.execution_registry import (
    build_paper_pilot_contrast_bindings,
    build_paper_pilot_execution_registry,
    build_paper_screen_execution_registry,
    derive_screening_optimization_seeds,
    derive_screening_seed_root,
)
from benchmarks.state_preparation.phase2.pilot import PILOT_CALCULATION_SOURCE_CHECKSUM
from benchmarks.state_preparation.phase2.protocol import (
    PRIMARY_FAMILY_STRATA,
    PRIMARY_TARGET_FAMILIES,
    SampleAllocation,
    SampleSizeDesign,
    load_initial_preregistration,
)
from benchmarks.state_preparation.phase2.screening import (
    PilotNormalizedComputeCalibration,
    ProductionResourceProjection,
)
from benchmarks.state_preparation.phase2.screening_design import (
    WP22CandidateConfiguration,
    build_screening_manifest,
)
from benchmarks.state_preparation.phase2.targets import (
    TargetPopulationManifest,
    build_target_population_config,
    create_target_population_manifest,
    role_master_entropy_commitment,
)
from benchmarks.state_preparation.phase2.training_orchestration import (
    TrainingJob,
    TrainingRunPlan,
)

_SCREEN_ENTROPY = b"wp22h-screening-target-entropy!!"
_SOURCE_CHECKSUM = canonical_checksum({"wp22h": "execution source"})


@cache
def _sample_size_design(fixed_test_trajectory_count: int = 512) -> SampleSizeDesign:
    """Return one typed pilot-derived design seam for registry compilation.

    Returns:
        A checksum-sealed design with balanced primary-family allocations.
    """
    preregistration = load_initial_preregistration()
    allocations = tuple(
        SampleAllocation(
            family_id=family_id,
            stratum_id=stratum_id,
            qubit_count=6,
            target_count=24 // len(PRIMARY_FAMILY_STRATA[family_id]),
        )
        for family_id in PRIMARY_TARGET_FAMILIES
        for stratum_id in PRIMARY_FAMILY_STRATA[family_id]
    )
    return SampleSizeDesign(
        design_id="wp22h_registry_test_design",
        preregistration_checksum=preregistration.content_checksum,
        pilot_nuisance_summary_checksum=canonical_checksum({"wp22h": "pilot nuisance"}),
        calculation_method_id="cluster_aware_paired_difference_v1",
        calculation_source_checksum=PILOT_CALCULATION_SOURCE_CHECKSUM,
        contrast_set_checksum=preregistration.contrast_set_checksum,
        target_population_configuration_checksum=preregistration.target_population_configuration_checksum,
        allocations=allocations,
        optimization_seed_count=3,
        fixed_test_trajectory_count=fixed_test_trajectory_count,
        achieved_power_by_contrast={
            "noisy_vs_noiseless": 0.91,
            "promoted_vs_layerwise_v2_if_distinct": 0.9,
        },
        expected_primary_mean_half_width=0.01,
        expected_overall_failure_rate_half_width=0.05,
        expected_trajectory_mcse=0.005,
        reestimation_kind="initial",
        reestimation_parent_checksum=None,
    )


@cache
def _pilot_calibration() -> PilotNormalizedComputeCalibration:
    """Return a complete typed 720-row pilot cap calibration seam.

    Returns:
        A valid calibration whose maximum successful normalized work is 10.
    """
    preregistration = load_initial_preregistration()
    method_ids = PILOT_METHOD_IDS
    candidate_checksums = {
        method_id: canonical_checksum({"wp22h pilot candidate": method_id}) for method_id in method_ids
    }
    resources = tuple(
        ProductionResourceProjection(
            job_checksum=canonical_checksum({"wp22h pilot job": index}),
            result_reference_checksum=canonical_checksum({"wp22h pilot result": index}),
            resource_document_checksum=canonical_checksum({"wp22h pilot resource": index}),
            execution_source_manifest_checksum=_SOURCE_CHECKSUM,
            method_id=method_ids[index % len(method_ids)],
            candidate_configuration_checksum=candidate_checksums[method_ids[index % len(method_ids)]],
            data_role="development",
            family_id="tfim_ground_state",
            stratum_id="critical",
            qubit_count=6,
            status="success",
            normalized_work=float(index % 10 + 1),
            structural_prefix_checksums=(),
            circuit_binding_checksum=canonical_checksum({"wp22h circuit": index}),
            compiled_resources_checksum=canonical_checksum({"wp22h compiled": index}),
            native_two_qubit_gates_per_chain_edge=(),
        )
        for index in range(720)
    )
    return PilotNormalizedComputeCalibration(
        preregistration_checksum=preregistration.content_checksum,
        execution_source_manifest_checksum=_SOURCE_CHECKSUM,
        pilot_plan_checksum=canonical_checksum({"wp22h": "pilot plan"}),
        pilot_custody_checksum=canonical_checksum({"wp22h": "pilot custody"}),
        calculation_rule_id="maximum_successful_q6_pilot_normalized_work_v1",
        normalized_compute_cap=10.0,
        pilot_q6_resources=resources,
    )


@cache
def _screening_target_manifest() -> TargetPopulationManifest:
    """Return the exact 48-target primary-q6 screening population.

    Returns:
        A deterministic typed screening target manifest.
    """
    preregistration = load_initial_preregistration()
    config = build_target_population_config(
        preregistration,
        "screening_selection",
        role_master_entropy_commitment=role_master_entropy_commitment(_SCREEN_ENTROPY),
    )
    return create_target_population_manifest(config, preregistration, _SCREEN_ENTROPY)


def _minimal_pilot_plan() -> TrainingRunPlan:
    """Return a small typed plan containing the three pilot configurations.

    Returns:
        A canonical plan sufficient to test deterministic contrast projection.
    """
    preregistration = load_initial_preregistration()
    _candidates, catalog = build_paper_pilot_execution_registry(preregistration)
    target_manifest_checksum = canonical_checksum({"wp22h": "minimal pilot targets"})
    target_spec_checksum = canonical_checksum({"wp22h": "minimal pilot target"})
    jobs = []
    for index, method_id in enumerate(PILOT_METHOD_IDS):
        link = next(
            item
            for item in catalog.bindings
            if item.binding.publication_method_id == method_id and item.binding.target_scope_id == "primary_q6"
        )
        binding = link.binding
        job_id = f"wp22h_minimal_pilot_{method_id}"
        jobs.append(
            TrainingJob(
                job_id=job_id,
                preset="paper-pilot",
                method_id=method_id,
                implementation_kind="phase2_pipeline",
                candidate_configuration_checksum=binding.publication_candidate_checksum,
                implementation_checksum=binding.implementation_checksum,
                strategy_schedule_checksum=binding.strategy_schedule.content_checksum,
                strategy_schedule=binding.strategy_schedule,
                target_manifest_checksum=target_manifest_checksum,
                target_instance_id="wp22h_minimal_pilot_target",
                target_spec_checksum=target_spec_checksum,
                family_id="tfim_ground_state",
                stratum_id="critical",
                qubit_count=6,
                data_role="development",
                optimization_block_id="wp22h_minimal_pilot_block",
                optimization_seed=index,
                evaluation_seed=100 + index,
                output_path=f"roles/development/tfim_ground_state/wp22h_minimal_pilot_target/{job_id}",
            )
        )
    return TrainingRunPlan(
        plan_id="wp22h_minimal_pilot_plan",
        preset="paper-pilot",
        preregistration_checksum=preregistration.content_checksum,
        target_manifest_checksums=(target_manifest_checksum,),
        screening_manifest_checksum=None,
        final_confirmation_seal_checksum=None,
        execution_source_checksum=None,
        jobs=tuple(sorted(jobs, key=lambda item: item.sort_key)),
    )


def test_pilot_registry_is_canonical_and_round_trips() -> None:
    """Pilot compilation closes real q6/q12 entries without test factories."""
    preregistration = load_initial_preregistration()
    candidates, catalog = build_paper_pilot_execution_registry(preregistration)
    repeated_candidates, repeated_catalog = build_paper_pilot_execution_registry(preregistration)

    assert tuple(candidate.method_id for candidate in candidates) == PILOT_METHOD_IDS
    assert len(catalog.bindings) == 6
    assert catalog.profile.preset == "paper-pilot"
    assert catalog.implementation_catalog.screening_outer_trajectory_count == 256
    assert candidates == repeated_candidates
    assert catalog.to_json() == repeated_catalog.to_json()
    assert all(WP22CandidateConfiguration.from_json(candidate.to_json()) == candidate for candidate in candidates)
    assert TrainingExecutionProfile.from_json(catalog.profile.to_json()) == catalog.profile
    assert RepositoryBindingCatalog.from_json(catalog.to_json()) == catalog

    for method_id in PILOT_METHOD_IDS:
        method_bindings = tuple(
            item.binding for item in catalog.bindings if item.binding.publication_method_id == method_id
        )
        assert len(method_bindings) == 2
        assert len({item.publication_candidate_checksum for item in method_bindings}) == 1
        assert {item.target_scope_id for item in method_bindings} == {"primary_q6", "secondary_q12"}
        assert all(item.execution_budget.normalized_compute_cap is None for item in method_bindings)


def test_screen_registry_uses_only_typed_pilot_derived_counts_and_cap() -> None:
    """Screen compilation binds the design count and pilot-only compute cap."""
    preregistration = load_initial_preregistration()
    design = _sample_size_design()
    calibration = _pilot_calibration()
    candidates, catalog = build_paper_screen_execution_registry(
        preregistration,
        design,
        calibration,
    )

    assert tuple(candidate.method_id for candidate in candidates) == SCREEN_METHOD_IDS
    assert len(catalog.bindings) == 9
    assert catalog.profile.preset == "paper-screen"
    assert catalog.implementation_catalog.screening_outer_trajectory_count == design.fixed_test_trajectory_count
    assert {item.binding.execution_budget.normalized_compute_cap for item in catalog.bindings} == {
        calibration.normalized_compute_cap
    }
    assert {
        policy.trajectory_count
        for item in catalog.bindings
        for policy in item.binding.evaluation_policies
        if policy.purpose == "screening_outer"
    } == {design.fixed_test_trajectory_count}
    assert RepositoryBindingCatalog.from_json(catalog.to_json()) == catalog

    operator_candidate = next(item for item in candidates if item.method_id == "adapt_style_state_preparation")
    operator_entry = catalog.implementation_catalog.resolve(
        "paper-screen",
        "adapt_style_state_preparation",
        "primary_q6",
    )
    assert isinstance(operator_entry.implementation_artifact.implementation_payload, OperatorGrowthExecutionSpec)
    assert (
        operator_candidate.implementation_checksum
        == operator_entry.implementation_artifact.implementation_payload.content_checksum
    )

    with pytest.raises(TypeError, match="sample_size_design"):
        build_paper_screen_execution_registry(
            preregistration,
            512,  # ty: ignore[invalid-argument-type] - runtime type rejection is under test
            calibration,
        )
    foreign_calibration = replace(
        calibration,
        preregistration_checksum=canonical_checksum({"wp22h": "foreign preregistration"}),
    )
    with pytest.raises(ValueError, match="different preregistration"):
        build_paper_screen_execution_registry(preregistration, design, foreign_calibration)


def test_screen_seeds_close_the_reviewed_manifest_derivation() -> None:
    """Screen optimization and outer seeds are derived from frozen authorities."""
    preregistration = load_initial_preregistration()
    candidates, catalog = build_paper_screen_execution_registry(
        preregistration,
        _sample_size_design(),
        _pilot_calibration(),
    )
    target_manifest = _screening_target_manifest()
    seeds = derive_screening_optimization_seeds(preregistration)
    root = derive_screening_seed_root(preregistration, catalog.profile, target_manifest)
    manifest = build_screening_manifest(
        preregistration,
        target_manifest,
        candidates,
        optimization_seeds=seeds,
        screening_seed_root=root,
    )

    assert seeds == (7329660033858372585, 4524389697880734114, 4579802874124897325)
    assert len(manifest.candidates) == 9
    assert len(manifest.cells) == 144
    assert len({cell.screening_seed for cell in manifest.cells}) == 144
    with pytest.raises(ValueError, match="primary-q6 screen"):
        derive_screening_seed_root(
            preregistration,
            build_paper_pilot_execution_registry(preregistration)[1].profile,
            target_manifest,
        )


def test_pilot_contrasts_are_mechanical_and_ordered() -> None:
    """The pilot contrast helper introduces no caller-selected treatment."""
    plan = _minimal_pilot_plan()
    contrasts = build_paper_pilot_contrast_bindings(plan)

    assert tuple(item.contrast_id for item in contrasts) == (
        "noisy_vs_noiseless",
        "promoted_vs_layerwise_v2_if_distinct",
    )
    assert contrasts[0].treatment_method_id == "layerwise_bmpd_crn_v2"
    assert contrasts[0].comparator_method_id == "layerwise_bmpd_noiseless"
    assert contrasts[1].treatment_method_id == "fixed_depth_bmpd_crn"
    assert contrasts[1].comparator_method_id == "layerwise_bmpd_crn_v2"
    assert all(item.pilot_plan_checksum == plan.content_checksum for item in contrasts)
