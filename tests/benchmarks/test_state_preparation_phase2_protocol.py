# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the sealed Phase II protocol and mechanical promotion rule."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from benchmarks.state_preparation.phase2 import (
    PRIMARY_FAMILY_STRATA,
    TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
    AnalysisSourceFileRef,
    AnalysisSourceManifest,
    CandidateSummary,
    ConfirmationAuthorization,
    FinalComparatorRef,
    FinalConfirmationSeal,
    InitialPreregistration,
    PrimaryContrastBinding,
    PromotionDecision,
    PromotionObservation,
    SampleAllocation,
    SampleSizeDesign,
    ScreeningCandidateRef,
    ScreeningCell,
    ScreeningEvidence,
    ScreeningManifest,
    authorize_confirmation,
    canonical_checksum,
    canonical_json,
    load_initial_preregistration,
    select_promoted_candidate,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


CONFIGURATION_SCHEMA_VERSION = "phase2_test_configuration_v1"
RESULT_SCHEMA_VERSION = "phase2_test_result_v1"
OPTIMIZATION_SEEDS = (101, 102, 103)


def _checksum(label: str) -> str:
    """Return a deterministic test checksum.

    Returns:
        A valid SHA-256 checksum string.
    """
    return f"sha256:{sha256(label.encode()).hexdigest()}"


MATCHING_PROJECTION_CHECKSUM = _checksum("matched layerwise v2/noiseless projection")
EXECUTION_SOURCE_MANIFEST_CHECKSUM = _checksum("frozen execution source manifest")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ENTRY_POINT = "scripts/plot_paper_figures.py"
ANALYSIS_SOURCE_BLOB = "415af0927728f60c8780905a786f7263d80969b6"
ANALYSIS_SOURCE_CHECKSUM = "sha256:e0445701e3e5d29962966da036364c8976b5f928d79e1e9dd424999a359e8e4e"


@pytest.fixture(scope="module")
def preregistration() -> InitialPreregistration:
    """Load the checked-in initial protocol.

    Returns:
        The validated Phase II preregistration.
    """
    return load_initial_preregistration()


def _candidate_checksum(method_id: str) -> str:
    """Return the deterministic configuration checksum for a method.

    Returns:
        The method's test configuration checksum.
    """
    return _checksum(f"configuration:{method_id}")


@pytest.fixture(scope="module")
def screening_candidates(
    preregistration: InitialPreregistration,
) -> tuple[ScreeningCandidateRef, ...]:
    """Create one screening configuration per family-wide method.

    Returns:
        The complete preregistered family-wide candidate set.
    """
    candidates = []
    for policy in preregistration.candidate_methods:
        if policy["scope"] != "all_families":
            continue
        method_id = cast("str", policy["method_id"])
        matching_projection_checksum = (
            MATCHING_PROJECTION_CHECKSUM if method_id in {"layerwise_bmpd_crn_v2", "layerwise_bmpd_noiseless"} else None
        )
        candidates.append(
            ScreeningCandidateRef(
                configuration_schema_version=CONFIGURATION_SCHEMA_VERSION,
                configuration_checksum=_candidate_checksum(method_id),
                method_id=method_id,
                noisy_training=cast("bool", policy["noisy_training"]),
                resource_stratum_id="depth4_equivalent",
                matching_projection_checksum=matching_projection_checksum,
            )
        )
    return tuple(candidates)


@pytest.fixture(scope="module")
def screening_cells() -> tuple[ScreeningCell, ...]:
    """Create the complete balanced q6 screening allocation.

    Returns:
        Twelve targets per family and three optimization seeds per target.
    """
    cells = []
    screening_seed = 500_000
    for family_id, strata in PRIMARY_FAMILY_STRATA.items():
        targets_per_stratum = 12 // len(strata)
        for stratum_id in strata:
            for target_index in range(targets_per_stratum):
                target_instance_id = f"{family_id}_{stratum_id}_q6_target_{target_index + 1:02d}"
                for optimization_index, optimization_seed in enumerate(
                    OPTIMIZATION_SEEDS,
                    start=1,
                ):
                    screening_seed += 1
                    cells.append(
                        ScreeningCell(
                            cell_id=f"{target_instance_id}_optimization_{optimization_index}",
                            family_id=family_id,
                            stratum_id=stratum_id,
                            qubit_count=6,
                            target_instance_id=target_instance_id,
                            optimization_seed=optimization_seed,
                            screening_seed=screening_seed,
                        )
                    )
    return tuple(cells)


@pytest.fixture(scope="module")
def screening_manifest(
    preregistration: InitialPreregistration,
    screening_candidates: tuple[ScreeningCandidateRef, ...],
    screening_cells: tuple[ScreeningCell, ...],
) -> ScreeningManifest:
    """Create the complete checksum-sealed screening universe.

    Returns:
        A valid screening manifest.
    """
    return ScreeningManifest(
        manifest_id="phase2_screening_manifest_v1",
        preregistration_checksum=preregistration.content_checksum,
        screening_target_manifest_checksum=_checksum("screening target manifest"),
        evaluation_policy_checksum=canonical_checksum({
            "endpoint": preregistration.primary_endpoint,
            "failure_policy": preregistration.failure_policy,
            "noise": preregistration.primary_noise_condition,
        }),
        resource_policy_checksum=canonical_checksum(preregistration.primary_resource_constraint),
        baseline_configuration_checksum=_candidate_checksum("layerwise_bmpd_crn_v2"),
        candidates=screening_candidates,
        cells=screening_cells,
    )


def _screening_evidence(
    manifest: ScreeningManifest,
    *,
    preregistration_checksum: str,
    fidelity_by_method: Mapping[str, float] | None = None,
    resource_by_configuration: Mapping[str, float] | None = None,
    failed_pairs: frozenset[tuple[str, str]] = frozenset(),
    violations_by_configuration: Mapping[str, tuple[str, ...]] | None = None,
    evidence_id: str = "phase2_screening_evidence_v1",
) -> ScreeningEvidence:
    """Create one complete source-addressed screening ledger.

    Returns:
        Exactly one observation for every sealed candidate/cell pair.
    """
    fidelities = {
        "layerwise_bmpd_crn_v2": 0.80,
        "layerwise_bmpd_noiseless": 0.81,
        "fixed_depth_bmpd_crn": 0.82,
    }
    if fidelity_by_method is not None:
        fidelities.update(fidelity_by_method)
    resources = {} if resource_by_configuration is None else resource_by_configuration
    violations = {} if violations_by_configuration is None else violations_by_configuration
    observations = []
    for candidate in manifest.candidates:
        fidelity = fidelities.get(candidate.method_id, 0.79)
        resource_value = resources.get(candidate.configuration_checksum, 12.0)
        candidate_violations = violations.get(candidate.configuration_checksum, ())
        for cell in manifest.cells:
            pair = (candidate.configuration_checksum, cell.cell_id)
            failed = pair in failed_pairs
            result_identity = canonical_checksum({
                "candidate": candidate.configuration_checksum,
                "cell": cell.cell_id,
                "failed": failed,
                "fidelity": None if failed else fidelity,
                "resource": None if failed else resource_value,
                "violations": candidate_violations,
            })
            observations.append(
                PromotionObservation(
                    configuration_checksum=candidate.configuration_checksum,
                    cell_id=cell.cell_id,
                    result_schema_version=RESULT_SCHEMA_VERSION,
                    result_record_checksum=result_identity,
                    status="failure" if failed else "success",
                    noisy_fidelity=None if failed else fidelity,
                    resource_value=None if failed else resource_value,
                    normalized_work=1.0,
                    failure_code="optimizer_failed" if failed else None,
                    protocol_violations=candidate_violations,
                )
            )
    return ScreeningEvidence(
        evidence_id=evidence_id,
        preregistration_checksum=preregistration_checksum,
        screening_manifest_checksum=manifest.content_checksum,
        observations=tuple(observations),
    )


@pytest.fixture(scope="module")
def screening_evidence(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
) -> ScreeningEvidence:
    """Create standard evidence that promotes fixed-depth CRN.

    Returns:
        The complete raw screening evidence.
    """
    return _screening_evidence(
        screening_manifest,
        preregistration_checksum=preregistration.content_checksum,
    )


@pytest.fixture(scope="module")
def promotion_decision(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
) -> PromotionDecision:
    """Apply the preregistered promotion rule.

    Returns:
        The standard decision promoting fixed-depth CRN.
    """
    return select_promoted_candidate(
        preregistration,
        screening_manifest,
        screening_evidence,
    )


def _sample_allocations(targets_per_family: int) -> tuple[SampleAllocation, ...]:
    """Create a balanced q6 allocation.

    Returns:
        One allocation for every primary family/stratum pair.
    """
    return tuple(
        SampleAllocation(
            family_id=family_id,
            stratum_id=stratum_id,
            qubit_count=6,
            target_count=targets_per_family // len(strata),
        )
        for family_id, strata in PRIMARY_FAMILY_STRATA.items()
        for stratum_id in strata
    )


def _sample_size_design(
    preregistration: InitialPreregistration,
    *,
    targets_per_family: int = 24,
    optimization_seed_count: int = 3,
    fixed_test_trajectory_count: int = 512,
) -> SampleSizeDesign:
    """Create a pilot-derived confirmatory sample-size design.

    Returns:
        A checksum-sealed balanced design.
    """
    return SampleSizeDesign(
        design_id=f"phase2_sample_size_{targets_per_family}_v1",
        preregistration_checksum=preregistration.content_checksum,
        pilot_nuisance_summary_checksum=_checksum("pilot nuisance summary"),
        calculation_method_id=cast("str", preregistration.sample_size_policy["method"]),
        calculation_source_checksum=_checksum("sample-size calculation source"),
        contrast_set_checksum=preregistration.contrast_set_checksum,
        target_population_configuration_checksum=(preregistration.target_population_configuration_checksum),
        allocations=_sample_allocations(targets_per_family),
        optimization_seed_count=optimization_seed_count,
        fixed_test_trajectory_count=fixed_test_trajectory_count,
        achieved_power_by_contrast={
            "noisy_vs_noiseless": 0.91,
            "promoted_vs_layerwise_v2_if_distinct": 0.91,
        },
        expected_primary_mean_half_width=0.009,
        expected_overall_failure_rate_half_width=0.049,
        expected_trajectory_mcse=0.004,
        reestimation_kind="initial",
        reestimation_parent_checksum=None,
    )


@pytest.fixture(scope="module")
def sample_size_design(
    preregistration: InitialPreregistration,
) -> SampleSizeDesign:
    """Create the standard valid sample-size design.

    Returns:
        The valid balanced q6 design.
    """
    return _sample_size_design(preregistration)


def _candidate(
    manifest: ScreeningManifest,
    method_id: str,
) -> ScreeningCandidateRef:
    """Resolve the unique candidate for a method.

    Returns:
        The method's screening candidate.
    """
    return next(candidate for candidate in manifest.candidates if candidate.method_id == method_id)


def _comparators(
    manifest: ScreeningManifest,
) -> tuple[FinalComparatorRef, ...]:
    """Create the exact screened v2 and noiseless comparator references.

    Returns:
        The two required typed comparator references.
    """
    baseline = _candidate(manifest, "layerwise_bmpd_crn_v2")
    noiseless = _candidate(manifest, "layerwise_bmpd_noiseless")
    return (
        FinalComparatorRef(
            role="layerwise_v2_reference",
            method_id=baseline.method_id,
            configuration_schema_version=baseline.configuration_schema_version,
            configuration_checksum=baseline.configuration_checksum,
            matched_to_configuration_checksum=noiseless.configuration_checksum,
            matching_projection_checksum=baseline.matching_projection_checksum,
        ),
        FinalComparatorRef(
            role="matched_noiseless_control",
            method_id=noiseless.method_id,
            configuration_schema_version=noiseless.configuration_schema_version,
            configuration_checksum=noiseless.configuration_checksum,
            matched_to_configuration_checksum=baseline.configuration_checksum,
            matching_projection_checksum=noiseless.matching_projection_checksum,
        ),
    )


def _primary_contrasts(
    preregistration: InitialPreregistration,
    manifest: ScreeningManifest,
    decision: PromotionDecision,
) -> tuple[PrimaryContrastBinding, ...]:
    """Bind the applicable primary contrasts to exact configurations.

    Returns:
        The noisy-control and promoted-v2 contrast bindings.
    """
    baseline = _candidate(manifest, "layerwise_bmpd_crn_v2")
    noiseless = _candidate(manifest, "layerwise_bmpd_noiseless")
    return (
        PrimaryContrastBinding(
            contrast_id="noisy_vs_noiseless",
            treatment_configuration_checksum=baseline.configuration_checksum,
            control_configuration_checksum=noiseless.configuration_checksum,
            paired_block_policy_checksum=preregistration.paired_block_policy_checksum,
            matching_projection_checksum=baseline.matching_projection_checksum,
        ),
        PrimaryContrastBinding(
            contrast_id="promoted_vs_layerwise_v2_if_distinct",
            treatment_configuration_checksum=decision.promoted_configuration_checksum,
            control_configuration_checksum=baseline.configuration_checksum,
            paired_block_policy_checksum=preregistration.paired_block_policy_checksum,
            matching_projection_checksum=None,
        ),
    )


def _analysis_source_manifest(
    preregistration: InitialPreregistration,
) -> AnalysisSourceManifest:
    """Create a manifest for one tracked executable analysis source.

    Returns:
        A commit- and blob-addressed analysis-source manifest.
    """
    return AnalysisSourceManifest(
        manifest_id="phase2_analysis_source_v1",
        preregistration_checksum=preregistration.content_checksum,
        analysis_template_checksum=preregistration.analysis_template_checksum,
        source_commit=preregistration.implementation_plan_commit,
        entry_point=ANALYSIS_ENTRY_POINT,
        source_files=(
            AnalysisSourceFileRef(
                repo_path=ANALYSIS_ENTRY_POINT,
                git_blob_id=ANALYSIS_SOURCE_BLOB,
                content_checksum=ANALYSIS_SOURCE_CHECKSUM,
            ),
        ),
        environment_lock_checksum=_checksum("analysis environment lock"),
        execution_source_manifest_checksum=EXECUTION_SOURCE_MANIFEST_CHECKSUM,
        clean_worktree=True,
    )


def _final_seal(
    preregistration: InitialPreregistration,
    manifest: ScreeningManifest,
    decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
) -> FinalConfirmationSeal:
    """Create a confirmation seal linked to every prior frozen object.

    Returns:
        A complete valid final-confirmation seal.
    """
    analysis_source_manifest = _analysis_source_manifest(preregistration)
    return FinalConfirmationSeal(
        seal_id="phase2_confirmation_v1",
        preregistration_checksum=preregistration.content_checksum,
        promotion_decision_checksum=decision.content_checksum,
        promoted_method_id=decision.promoted_method_id,
        promoted_configuration_checksum=decision.promoted_configuration_checksum,
        comparators=_comparators(manifest),
        primary_contrasts=_primary_contrasts(
            preregistration,
            manifest,
            decision,
        ),
        confirmatory_target_manifest_checksum=_checksum("confirmatory target manifest"),
        target_count_by_family=sample_size_design.target_count_by_family,
        optimization_seed_count=sample_size_design.optimization_seed_count,
        fixed_test_trajectory_count=sample_size_design.fixed_test_trajectory_count,
        primary_noise_condition=preregistration.primary_noise_condition,
        primary_resource_budget={
            "metric": "native_two_qubit_gates_per_chain_edge",
            "cap_per_chain_edge": 12.0,
            "normalized_compute_cap": 1_000_000.0,
            "reachable_stratum_manifest_checksum": _checksum("reachable resource strata"),
        },
        hyperparameters_checksum=_checksum("final hyperparameters"),
        execution_source_checksum=(analysis_source_manifest.execution_source_manifest_checksum),
        analysis_template_checksum=preregistration.analysis_template_checksum,
        analysis_source_manifest_checksum=analysis_source_manifest.content_checksum,
        sample_size_design_checksum=sample_size_design.content_checksum,
        failure_policy_checksum=preregistration.failure_policy_checksum,
    )


def _authorize(
    preregistration: InitialPreregistration,
    manifest: ScreeningManifest,
    evidence: ScreeningEvidence,
    decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
    final_seal: FinalConfirmationSeal,
    *,
    analysis_source_manifest: AnalysisSourceManifest | None = None,
) -> ConfirmationAuthorization:
    """Authorize using the tracked test analysis source.

    Returns:
        The opaque confirmation authorization.
    """
    source_manifest = (
        _analysis_source_manifest(preregistration) if analysis_source_manifest is None else analysis_source_manifest
    )
    return authorize_confirmation(
        preregistration,
        manifest,
        evidence,
        decision,
        sample_size_design,
        source_manifest,
        final_seal,
        REPOSITORY_ROOT,
    )


@pytest.fixture(scope="module")
def final_seal(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    promotion_decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
) -> FinalConfirmationSeal:
    """Create the standard valid final seal.

    Returns:
        The fully linked confirmation seal.
    """
    return _final_seal(
        preregistration,
        screening_manifest,
        promotion_decision,
        sample_size_design,
    )


def test_checked_in_preregistration_freezes_scientific_and_rng_decisions(
    preregistration: InitialPreregistration,
) -> None:
    """The trusted protocol must pin scientific, RNG, and sample floors."""
    assert (
        preregistration.content_checksum
        == TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM
        == "sha256:f87aeef22069fbf01c1c5f6957a629f9599e08c0c61e1b0976f85c3151a6ab3f"
    )
    assert preregistration.phase_i_baseline_commit == "fcf0a65f0f9d2c6d2c50131f100ab530e6ceab11"
    assert preregistration.implementation_plan_commit == "3c74e22cba0301f7d44c6472f6cf65bd5a7e43da"
    assert preregistration.legacy_evidence_audit_checksum == (
        "sha256:a294080bf54a62b2bad0df85faa2f75ade5098b6a9afd84dc81fbb29bafdda1c"
    )
    assert preregistration.primary_noise_condition["noise_id"] == ("depolarizing_1s_all")
    assert preregistration.primary_resource_constraint["cap_per_chain_edge"] == (pytest.approx(12.0))

    target_policy = preregistration.target_population_policy
    rng_policy = cast("Mapping[str, object]", target_policy["rng_policy"])
    numeric_policy = cast("Mapping[str, object]", target_policy["numeric_policy"])
    allocation = cast("Mapping[str, object]", target_policy["role_allocation_policy"])
    assert rng_policy["bit_generator"] == "PCG64"
    assert rng_policy["seed_sequence"] == "SeedSequence"
    assert rng_policy["derivation"] == ("first_16_bytes_hmac_sha256_external_master_and_canonical_identity")
    assert rng_policy["instance_entropy_bits"] == 128
    assert rng_policy["master_entropy_bits"] == 256
    assert rng_policy["role_master_domains"] == (
        "development",
        "screening_selection",
        "confirmatory",
    )
    assert rng_policy["forbidden_apis"] == (
        "default_rng",
        "global_numpy_rng",
        "randomstate",
        "python_hash",
    )
    assert numeric_policy["generation_numpy_version"] == "2.4.6"
    assert allocation["development_targets_per_family"] == 12
    assert allocation["screening_targets_per_family"] == 12
    assert allocation["screening_optimizer_seed_count"] == 3
    assert allocation["confirmatory_min_targets_per_family"] == 24
    assert allocation["confirmatory_max_targets_per_family"] == 96
    assert allocation["confirmatory_target_increment"] == 6

    sample_policy = preregistration.sample_size_policy
    assert sample_policy["minimum_targets_per_family"] == 24
    assert sample_policy["minimum_optimization_seed_count"] == 3
    assert sample_policy["trajectory_count_min"] == 256
    assert sample_policy["power"] == pytest.approx(0.9)
    assert sample_policy["planning_alpha"] == pytest.approx(0.025)
    assert sample_policy["minimum_relevant_noisy_gain"] == pytest.approx(0.02)
    assert sample_policy["maximum_reestimations"] == 1
    assert sample_policy["reestimation_trigger_fraction"] == pytest.approx(0.5)
    assert sample_policy["variance_model"] == (
        "target_over_n_plus_optimizer_over_n_s_plus_mc_over_n_s_ntraj_then_squared_family_weights"
    )
    assert sample_policy["power_calculation"] == ("normal_approximation_one_sided_worst_holm_alpha")


def test_preregistration_is_immutable_canonical_and_tamper_evident(
    preregistration: InitialPreregistration,
) -> None:
    """Nested protocol content must be immutable and checksum verified."""
    assert InitialPreregistration.from_json(preregistration.to_json()) == preregistration
    assert canonical_checksum(preregistration.promotion_rule) == preregistration.promotion_rule_checksum
    with pytest.raises(TypeError):
        cast("dict[str, object]", preregistration.primary_noise_condition)["strength_scale"] = 0.5
    with pytest.raises(FrozenInstanceError):
        cast("Any", preregistration).protocol_id = "changed"

    tampered = preregistration.to_dict()
    cast("list[str]", tampered["scientific_questions"])[0] = "A post hoc question."
    with pytest.raises(ValueError, match="content checksum mismatch"):
        InitialPreregistration.from_dict(tampered)


def test_screening_manifest_is_complete_balanced_and_round_trips(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
) -> None:
    """The sealed universe must contain every method and all 144 q6 cells."""
    assert ScreeningManifest.from_json(screening_manifest.to_json()) == screening_manifest
    expected_methods = {
        cast("str", policy["method_id"])
        for policy in preregistration.candidate_methods
        if policy["scope"] == "all_families"
    }
    assert {candidate.method_id for candidate in screening_manifest.candidates} == (expected_methods)
    method_counts = Counter(candidate.method_id for candidate in screening_manifest.candidates)
    assert method_counts["layerwise_bmpd_crn_v2"] == 1
    assert method_counts["layerwise_bmpd_noiseless"] == 1
    assert len(screening_manifest.cells) == 144
    assert {cell.qubit_count for cell in screening_manifest.cells} == {6}
    assert len({cell.screening_seed for cell in screening_manifest.cells}) == 144

    targets_by_family: dict[str, set[str]] = defaultdict(set)
    targets_by_stratum: Counter[tuple[str, str]] = Counter()
    seeds_by_target: dict[str, set[int]] = defaultdict(set)
    for cell in screening_manifest.cells:
        targets_by_family[cell.family_id].add(cell.target_instance_id)
        seeds_by_target[cell.target_instance_id].add(cell.optimization_seed)
    for family_id, targets in targets_by_family.items():
        assert len(targets) == 12, family_id
        for target_id in targets:
            cell = next(item for item in screening_manifest.cells if item.target_instance_id == target_id)
            targets_by_stratum[cell.family_id, cell.stratum_id] += 1
    assert all(seeds == set(OPTIMIZATION_SEEDS) for seeds in seeds_by_target.values())
    for family_id, strata in PRIMARY_FAMILY_STRATA.items():
        assert {targets_by_stratum[family_id, stratum_id] for stratum_id in strata} == {12 // len(strata)}


def test_screening_manifest_rejects_omissions_and_changed_matching(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
) -> None:
    """Cross-validation must enforce method coverage and the matched projection."""
    omitted = replace(
        screening_manifest,
        candidates=tuple(
            candidate for candidate in screening_manifest.candidates if candidate.method_id != "spsa_layerwise"
        ),
    )
    with pytest.raises(ValueError, match="omits preregistered family-wide methods"):
        select_promoted_candidate(preregistration, omitted, screening_evidence)

    changed_candidates = tuple(
        replace(
            candidate,
            matching_projection_checksum=_checksum("changed projection"),
        )
        if candidate.method_id == "layerwise_bmpd_noiseless"
        else candidate
        for candidate in screening_manifest.candidates
    )
    changed_matching = replace(
        screening_manifest,
        candidates=changed_candidates,
    )
    with pytest.raises(ValueError, match="share one matching projection"):
        select_promoted_candidate(preregistration, changed_matching, screening_evidence)


def test_screening_evidence_is_exact_cartesian_product_and_round_trips(
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
) -> None:
    """Every candidate/cell result must be present once and source-addressed."""
    assert ScreeningEvidence.from_json(screening_evidence.to_json()) == screening_evidence
    expected_pairs = {
        (candidate.configuration_checksum, cell.cell_id)
        for candidate in screening_manifest.candidates
        for cell in screening_manifest.cells
    }
    actual_pairs = {
        (observation.configuration_checksum, observation.cell_id) for observation in screening_evidence.observations
    }
    assert actual_pairs == expected_pairs
    assert len(actual_pairs) == 9 * 144
    assert len({observation.result_record_checksum for observation in screening_evidence.observations}) == len(
        actual_pairs
    )


def test_screening_evidence_rejects_omitted_and_extra_pairs(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
) -> None:
    """The raw ledger must equal, rather than approximate, the sealed universe."""
    omitted = replace(
        screening_evidence,
        observations=screening_evidence.observations[:-1],
    )
    with pytest.raises(ValueError, match="does not match the sealed universe"):
        select_promoted_candidate(preregistration, screening_manifest, omitted)

    exemplar = screening_evidence.observations[0]
    extra_observation = replace(
        exemplar,
        configuration_checksum=_checksum("unsealed configuration"),
        result_record_checksum=_checksum("unsealed result"),
    )
    extra = replace(
        screening_evidence,
        observations=(*screening_evidence.observations, extra_observation),
    )
    with pytest.raises(ValueError, match="does not match the sealed universe"):
        select_promoted_candidate(preregistration, screening_manifest, extra)


def test_screening_evidence_rejects_fake_seal_linkage(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
) -> None:
    """Evidence must link to the exact preregistration and screening manifest."""
    wrong_preregistration = replace(
        screening_evidence,
        preregistration_checksum=_checksum("wrong preregistration"),
    )
    with pytest.raises(ValueError, match="does not reference the supplied preregistration"):
        select_promoted_candidate(
            preregistration,
            screening_manifest,
            wrong_preregistration,
        )

    wrong_manifest = replace(
        screening_evidence,
        screening_manifest_checksum=_checksum("wrong screening manifest"),
    )
    with pytest.raises(ValueError, match="does not reference the supplied screening manifest"):
        select_promoted_candidate(
            preregistration,
            screening_manifest,
            wrong_manifest,
        )


def test_promotion_uses_family_weighted_itt_and_round_trips(
    promotion_decision: PromotionDecision,
    screening_evidence: ScreeningEvidence,
) -> None:
    """A qualifying noisy candidate must be promoted mechanically."""
    assert promotion_decision.promoted_method_id == "fixed_depth_bmpd_crn"
    assert promotion_decision.promoted_configuration_checksum == _candidate_checksum("fixed_depth_bmpd_crn")
    assert promotion_decision.screening_evidence_checksum == (screening_evidence.content_checksum)
    assert promotion_decision.null_fallback is False
    promoted = next(
        summary
        for summary in promotion_decision.candidate_summaries
        if summary.configuration_checksum == promotion_decision.promoted_configuration_checksum
    )
    assert promoted.weighted_itt_fidelity == pytest.approx(0.82)
    assert PromotionDecision.from_json(promotion_decision.to_json()) == (promotion_decision)


def test_failures_contribute_zero_and_can_force_null_promotion(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
) -> None:
    """Failure of one complete family must count as zero and exceed the cap."""
    challenger_checksum = _candidate_checksum("fixed_depth_bmpd_crn")
    gaussian_cells = {cell.cell_id for cell in screening_manifest.cells if cell.family_id == "gaussian_amplitude"}
    evidence = _screening_evidence(
        screening_manifest,
        preregistration_checksum=preregistration.content_checksum,
        fidelity_by_method={"fixed_depth_bmpd_crn": 1.0},
        failed_pairs=frozenset((challenger_checksum, cell_id) for cell_id in gaussian_cells),
    )
    decision = select_promoted_candidate(
        preregistration,
        screening_manifest,
        evidence,
    )
    challenger = next(
        summary for summary in decision.candidate_summaries if summary.configuration_checksum == challenger_checksum
    )

    assert challenger.weighted_itt_fidelity == pytest.approx(0.75)
    assert challenger.failure_rate == pytest.approx(0.25)
    assert "failure_rate_exceeded" in challenger.ineligibility_reasons
    assert decision.promoted_configuration_checksum == (screening_manifest.baseline_configuration_checksum)
    assert decision.null_fallback is True


@pytest.mark.parametrize(
    ("fidelity", "resource", "violation", "reason"),
    [
        (0.804, 12.0, (), "minimum_gain_not_met"),
        (0.82, 12.01, (), "resource_cap_exceeded"),
        (0.82, 12.0, ("leaked_holdout",), "leaked_holdout"),
    ],
)
def test_promotion_rejects_ineligible_challengers(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    fidelity: float,
    resource: float,
    violation: tuple[str, ...],
    reason: str,
) -> None:
    """Every sealed eligibility constraint must be applied before ranking."""
    challenger_checksum = _candidate_checksum("fixed_depth_bmpd_crn")
    evidence = _screening_evidence(
        screening_manifest,
        preregistration_checksum=preregistration.content_checksum,
        fidelity_by_method={"fixed_depth_bmpd_crn": fidelity},
        resource_by_configuration={challenger_checksum: resource},
        violations_by_configuration={challenger_checksum: violation},
    )
    decision = select_promoted_candidate(
        preregistration,
        screening_manifest,
        evidence,
    )
    challenger = next(
        summary for summary in decision.candidate_summaries if summary.configuration_checksum == challenger_checksum
    )

    assert challenger.eligible is False
    assert reason in challenger.ineligibility_reasons
    assert decision.null_fallback is True


def test_noiseless_control_is_reported_but_not_promotion_eligible(
    promotion_decision: PromotionDecision,
) -> None:
    """The matched noiseless method is evidence, not a promotion candidate."""
    noiseless = next(
        summary for summary in promotion_decision.candidate_summaries if summary.method_id == "layerwise_bmpd_noiseless"
    )
    assert noiseless.eligible is False
    assert set(noiseless.ineligibility_reasons) >= {
        "method_not_promotion_eligible",
        "no_noisy_training",
    }


def test_baseline_integrity_violation_is_fatal(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
) -> None:
    """The corrected v2 baseline cannot silently fall back from itself."""
    baseline_checksum = screening_manifest.baseline_configuration_checksum
    evidence = _screening_evidence(
        screening_manifest,
        preregistration_checksum=preregistration.content_checksum,
        violations_by_configuration={baseline_checksum: ("leaked_holdout",)},
    )
    with pytest.raises(
        ValueError,
        match="corrected v2 baseline failed fatal screening-integrity",
    ):
        select_promoted_candidate(preregistration, screening_manifest, evidence)


def test_promotion_ties_are_deterministic(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
) -> None:
    """The lexicographically smaller checksum must break a complete metric tie."""
    tied_methods = ("fixed_depth_bmpd_crn", "layerwise_bmpd_resampled")
    evidence = _screening_evidence(
        screening_manifest,
        preregistration_checksum=preregistration.content_checksum,
        fidelity_by_method=dict.fromkeys(tied_methods, 0.82),
    )
    decision = select_promoted_candidate(
        preregistration,
        screening_manifest,
        evidence,
    )
    assert decision.promoted_configuration_checksum == min(_candidate_checksum(method_id) for method_id in tied_methods)


def test_promotion_decision_rejects_nonmechanical_choice() -> None:
    """A sealed decision cannot nominate a lower-ranked eligible challenger."""
    baseline = CandidateSummary(
        configuration_checksum=_checksum("summary baseline"),
        method_id="layerwise_bmpd_crn_v2",
        weighted_itt_fidelity=0.8,
        failure_rate=0.0,
        max_resource_excess=0.0,
        mean_normalized_work=1.0,
        eligible=True,
        ineligibility_reasons=(),
    )
    winner = replace(
        baseline,
        configuration_checksum=_checksum("summary winner"),
        method_id="fixed_depth_bmpd_crn",
        weighted_itt_fidelity=0.83,
    )
    lower = replace(
        baseline,
        configuration_checksum=_checksum("summary lower"),
        method_id="spsa_layerwise",
        weighted_itt_fidelity=0.82,
    )
    with pytest.raises(ValueError, match="not the mechanically ranked"):
        PromotionDecision(
            preregistration_checksum=_checksum("summary preregistration"),
            screening_manifest_checksum=_checksum("summary manifest"),
            screening_evidence_checksum=_checksum("summary evidence"),
            baseline_configuration_checksum=baseline.configuration_checksum,
            promoted_method_id=lower.method_id,
            promoted_configuration_checksum=lower.configuration_checksum,
            null_fallback=False,
            rule_checksum=_checksum("summary rule"),
            candidate_summaries=(baseline, winner, lower),
        )


def test_sample_size_design_round_trips_and_derives_counts(
    sample_size_design: SampleSizeDesign,
) -> None:
    """The sample calculation must be sealed with balanced derived totals."""
    assert SampleSizeDesign.from_json(sample_size_design.to_json()) == (sample_size_design)
    assert sample_size_design.target_count_by_family == {
        "gaussian_amplitude": 24,
        "tfim_ground_state": 24,
        "haar_random": 24,
        "random_mps": 24,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimization_seed_count", 1, "optimization_seed_count"),
        ("fixed_test_trajectory_count", 1, "fixed_test_trajectory_count"),
    ],
)
def test_sample_size_design_rejects_single_replicates(
    sample_size_design: SampleSizeDesign,
    field: str,
    value: int,
    message: str,
) -> None:
    """A sample-size seal cannot encode n=1 seed or trajectory estimates."""
    with pytest.raises(ValueError, match=message):
        replace(sample_size_design, **{field: value})


def test_confirmation_authorization_requires_all_independent_seals(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
    promotion_decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
    final_seal: FinalConfirmationSeal,
) -> None:
    """Matching prior, evidence, sample, and final seals authorize confirmation."""
    authorization = _authorize(
        preregistration,
        screening_manifest,
        screening_evidence,
        promotion_decision,
        sample_size_design,
        final_seal,
    )

    assert authorization.preregistration_checksum == preregistration.content_checksum
    assert authorization.final_seal_checksum == final_seal.content_checksum
    assert authorization.target_manifest_checksum == (final_seal.confirmatory_target_manifest_checksum)
    assert authorization.execution_source_checksum == (final_seal.execution_source_checksum)
    assert FinalConfirmationSeal.from_json(final_seal.to_json()) == final_seal
    assert final_seal.analysis_template_checksum != (final_seal.analysis_source_manifest_checksum)
    with pytest.raises(ValueError, match="only be created"):
        ConfirmationAuthorization(
            _checksum("authorization preregistration"),
            _checksum("authorization final seal"),
            _checksum("authorization target"),
            _checksum("authorization source"),
            object(),
        )


def test_confirmation_rejects_forged_high_promotion_summary(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
    promotion_decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
) -> None:
    """Authorization must recompute raw evidence rather than trust summaries."""
    forged_summaries = tuple(
        replace(summary, weighted_itt_fidelity=0.99)
        if summary.configuration_checksum == promotion_decision.promoted_configuration_checksum
        else summary
        for summary in promotion_decision.candidate_summaries
    )
    forged_decision = replace(
        promotion_decision,
        candidate_summaries=forged_summaries,
    )
    forged_seal = _final_seal(
        preregistration,
        screening_manifest,
        forged_decision,
        sample_size_design,
    )
    with pytest.raises(ValueError, match="not the exact result"):
        _authorize(
            preregistration,
            screening_manifest,
            screening_evidence,
            forged_decision,
            sample_size_design,
            forged_seal,
        )


def test_confirmation_requires_exact_v2_and_matched_noiseless_comparators(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
    promotion_decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
    final_seal: FinalConfirmationSeal,
) -> None:
    """Comparator roles must retain screened identities and matching projection."""
    v2_reference, noiseless_control = final_seal.comparators
    unknown_v2 = replace(
        v2_reference,
        configuration_checksum=_checksum("unscreened v2 configuration"),
    )
    unknown_v2_seal = replace(
        final_seal,
        comparators=(unknown_v2, noiseless_control),
    )
    with pytest.raises(ValueError, match="is not the exact screened configuration"):
        _authorize(
            preregistration,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            sample_size_design,
            unknown_v2_seal,
        )

    changed_projection = replace(
        noiseless_control,
        matching_projection_checksum=_checksum("changed noiseless projection"),
    )
    changed_projection_seal = replace(
        final_seal,
        comparators=(v2_reference, changed_projection),
    )
    with pytest.raises(ValueError, match="uses a changed matching projection"):
        _authorize(
            preregistration,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            sample_size_design,
            changed_projection_seal,
        )


def test_confirmation_requires_exact_primary_contrast_bindings(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
    promotion_decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
    final_seal: FinalConfirmationSeal,
) -> None:
    """Primary treatment, control, pairing, and projection must stay frozen."""
    noisy_contrast, promoted_contrast = final_seal.primary_contrasts
    changed_noisy = replace(
        noisy_contrast,
        paired_block_policy_checksum=_checksum("changed paired block"),
    )
    changed_noisy_seal = replace(
        final_seal,
        primary_contrasts=(changed_noisy, promoted_contrast),
    )
    with pytest.raises(ValueError, match="noisy-versus-noiseless contrast"):
        _authorize(
            preregistration,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            sample_size_design,
            changed_noisy_seal,
        )

    noiseless = _candidate(screening_manifest, "layerwise_bmpd_noiseless")
    changed_promoted = replace(
        promoted_contrast,
        control_configuration_checksum=noiseless.configuration_checksum,
    )
    changed_promoted_seal = replace(
        final_seal,
        primary_contrasts=(noisy_contrast, changed_promoted),
    )
    with pytest.raises(ValueError, match="promoted-versus-v2 contrast"):
        _authorize(
            preregistration,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            sample_size_design,
            changed_promoted_seal,
        )


def test_confirmation_rejects_below_floor_sample_allocation(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
    promotion_decision: PromotionDecision,
) -> None:
    """Balanced allocations below the preregistered family floor must fail."""
    undersized = _sample_size_design(
        preregistration,
        targets_per_family=18,
    )
    undersized_seal = _final_seal(
        preregistration,
        screening_manifest,
        promotion_decision,
        undersized,
    )
    with pytest.raises(ValueError, match="violates the balanced frozen bounds"):
        _authorize(
            preregistration,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            undersized,
            undersized_seal,
        )


def test_confirmation_rejects_denormalized_sample_count_mismatch(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
    promotion_decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
    final_seal: FinalConfirmationSeal,
) -> None:
    """The final seal cannot restate different counts than its sample design."""
    changed_counts = dict(final_seal.target_count_by_family)
    changed_counts["gaussian_amplitude"] = 30
    changed_seal = replace(
        final_seal,
        target_count_by_family=changed_counts,
    )
    with pytest.raises(ValueError, match="denormalized sample sizes differ"):
        _authorize(
            preregistration,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            sample_size_design,
            changed_seal,
        )


@pytest.mark.parametrize(
    ("field", "replacement_value", "message"),
    [
        (
            "analysis_template_checksum",
            _checksum("changed analysis template"),
            "analysis-template checksum",
        ),
        (
            "failure_policy_checksum",
            _checksum("changed failure policy"),
            "failure policy",
        ),
    ],
)
def test_confirmation_rejects_changed_locked_values(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
    promotion_decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
    final_seal: FinalConfirmationSeal,
    field: str,
    replacement_value: object,
    message: str,
) -> None:
    """Changing a preregistered final-study value must prevent authorization."""
    changed_seal = replace(final_seal, **{field: replacement_value})
    with pytest.raises(ValueError, match=message):
        _authorize(
            preregistration,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            sample_size_design,
            changed_seal,
        )


def test_analysis_source_manifest_round_trips_and_links_distinct_seals(
    preregistration: InitialPreregistration,
    final_seal: FinalConfirmationSeal,
) -> None:
    """Analysis intent, source inventory, and execution source stay distinct."""
    source_manifest = _analysis_source_manifest(preregistration)
    assert AnalysisSourceManifest.from_json(source_manifest.to_json()) == (source_manifest)
    assert source_manifest.source_files == (
        AnalysisSourceFileRef(
            repo_path=ANALYSIS_ENTRY_POINT,
            git_blob_id=ANALYSIS_SOURCE_BLOB,
            content_checksum=ANALYSIS_SOURCE_CHECKSUM,
        ),
    )
    assert final_seal.analysis_template_checksum != (final_seal.analysis_source_manifest_checksum)
    assert source_manifest.analysis_template_checksum == (preregistration.analysis_template_checksum)
    assert final_seal.analysis_source_manifest_checksum == (source_manifest.content_checksum)
    assert final_seal.execution_source_checksum == (source_manifest.execution_source_manifest_checksum)


def test_confirmation_rejects_changed_analysis_source_links(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
    promotion_decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
    final_seal: FinalConfirmationSeal,
) -> None:
    """The executable analysis must link to its plan and final execution seal."""
    source_manifest = _analysis_source_manifest(preregistration)
    changed_template = replace(
        source_manifest,
        analysis_template_checksum=_checksum("changed source analysis template"),
    )
    with pytest.raises(ValueError, match="changed primary-analysis template"):
        _authorize(
            preregistration,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            sample_size_design,
            final_seal,
            analysis_source_manifest=changed_template,
        )

    changed_manifest_link = replace(
        final_seal,
        analysis_source_manifest_checksum=_checksum("changed analysis manifest link"),
    )
    with pytest.raises(ValueError, match="does not reference the supplied executable"):
        _authorize(
            preregistration,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            sample_size_design,
            changed_manifest_link,
            analysis_source_manifest=source_manifest,
        )

    changed_execution_link = replace(
        final_seal,
        execution_source_checksum=_checksum("changed execution source link"),
    )
    with pytest.raises(ValueError, match="not linked to the final execution-source"):
        _authorize(
            preregistration,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            sample_size_design,
            changed_execution_link,
            analysis_source_manifest=source_manifest,
        )


def test_authorization_enforces_preregistration_root_of_trust(
    preregistration: InitialPreregistration,
    screening_manifest: ScreeningManifest,
    screening_evidence: ScreeningEvidence,
    promotion_decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
    final_seal: FinalConfirmationSeal,
) -> None:
    """Authorization must reject a validly resealed but untrusted protocol."""
    untrusted = replace(
        preregistration,
        protocol_id="untrusted_phase2_protocol",
    )
    with pytest.raises(ValueError, match="trusted checked-in protocol digest"):
        _authorize(
            untrusted,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            sample_size_design,
            final_seal,
        )


def test_loader_rejects_validly_resealed_untrusted_preregistration(
    preregistration: InitialPreregistration,
    tmp_path: Path,
) -> None:
    """Loading must compare normalized content with the runtime trust anchor."""
    payload = preregistration.to_dict()
    payload["protocol_id"] = "validly_resealed_but_untrusted"
    payload["content_checksum"] = canonical_checksum({
        key: value for key, value in payload.items() if key != "content_checksum"
    })
    path = tmp_path / "untrusted_preregistration.json"
    path.write_text(canonical_json(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="trusted runtime constant"):
        load_initial_preregistration(path)
