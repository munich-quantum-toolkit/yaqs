# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sealed Phase II study protocol, promotion, and confirmation authorization."""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    read_canonical_json_object,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .training_schedules import TrainingStrategySchedule
from .validation import (
    require_bool,
    require_checksum,
    require_exact_keys,
    require_float,
    require_git_blob,
    require_git_commit,
    require_int,
    require_mapping,
    require_nonempty_text,
    require_relative_path,
    require_slug,
    require_string_sequence,
)

PREREGISTRATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.preregistration.v1"
PROMOTION_RULE_VERSION = "yaqs.state_preparation.phase2.promotion_rule.v1"
SCREENING_MANIFEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.screening_manifest.v1"
SCREENING_EVIDENCE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.screening_evidence.v1"
PROMOTION_DECISION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.promotion_decision.v1"
SAMPLE_SIZE_DESIGN_SCHEMA_VERSION = "yaqs.state_preparation.phase2.sample_size_design.v1"
ANALYSIS_SOURCE_MANIFEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.analysis_source_manifest.v1"
CONFIRMATION_SEAL_SCHEMA_VERSION = "yaqs.state_preparation.phase2.confirmation_seal.v1"
FINAL_CONFIGURATION_EXECUTION_REF_SCHEMA_VERSION = "yaqs.state_preparation.phase2.final_configuration_execution_ref.v1"
FINAL_CONFIGURATION_EXECUTION_MANIFEST_SCHEMA_VERSION = (
    "yaqs.state_preparation.phase2.final_configuration_execution_manifest.v1"
)

DEFAULT_PREREGISTRATION_PATH = Path(__file__).with_name("data") / "initial_preregistration_v1.json"

DATA_ROLES = (
    "development",
    "checkpoint_validation",
    "screening_selection",
    "confirmatory",
    "secondary_benchmark",
)
PRIMARY_TARGET_FAMILIES = ("gaussian_amplitude", "tfim_ground_state", "haar_random", "random_mps")
PRIMARY_FAMILY_STRATA = {
    "gaussian_amplitude": ("interior",),
    "tfim_ground_state": ("ferromagnetic", "critical", "paramagnetic"),
    "haar_random": ("dense_complex",),
    "random_mps": ("bond2", "bond3"),
}

# This literal is the runtime root of trust for the checked-in preregistration.
# It is updated only when the governing protocol is deliberately resealed.
TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM = "sha256:fa8a4efac484e5b426a76cea15b727becb04fc65814312aec0d8649668054b50"

_PREREGISTRATION_KEYS = frozenset({
    "schema_version",
    "protocol_id",
    "phase_i_baseline_commit",
    "implementation_plan_commit",
    "legacy_evidence_audit_checksum",
    "scientific_questions",
    "confirmatory_questions",
    "candidate_methods",
    "primary_endpoint",
    "primary_noise_condition",
    "primary_resource_constraint",
    "target_population_policy",
    "target_family_weights",
    "data_role_policy",
    "failure_policy",
    "promotion_rule",
    "sample_size_policy",
    "multiplicity_policy",
    "target_access_policy",
    "analysis_template",
    "final_confirmation_schema_version",
    "content_checksum",
})
_CANDIDATE_METHOD_KEYS = frozenset({"method_id", "scope", "promotion_eligible", "noisy_training", "role"})
_PRIMARY_ENDPOINT_KEYS = frozenset({"metric", "aggregation", "data_role", "higher_is_better"})
_PRIMARY_NOISE_KEYS = frozenset({
    "noise_id",
    "definition_version",
    "strength_scale",
    "tjm_dt",
    "training_placement",
    "test_placement",
})
_PRIMARY_RESOURCE_KEYS = frozenset({
    "metric",
    "cap_per_chain_edge",
    "comparison_rule",
    "compiler_policy_id",
    "connectivity",
    "routing_policy",
    "residual_gap_reporting",
    "normalized_compute_cap_source",
})
_TARGET_POPULATION_KEYS = frozenset({
    "generator_schema_version",
    "families",
    "primary_qubit_counts",
    "secondary_qubit_counts",
    "instance_id_policy",
    "rng_policy",
    "numeric_policy",
    "role_allocation_policy",
})
_TARGET_FAMILY_KEYS = frozenset({"family_id", "parameter_distribution", "strata"})
_TARGET_NUMERIC_POLICY_KEYS = frozenset({
    "authorized_state_solver",
    "basis_bit_order",
    "complex_precision",
    "generation_numpy_version",
    "global_phase_rule",
    "manifest_spectrum_solver",
    "real_precision",
    "spectrum_agreement_atol",
    "spectrum_agreement_rtol",
    "standard_normal_api",
    "uniform_interval",
})
_DATA_ROLE_KEYS = frozenset({
    "roles",
    "target_instance_domains",
    "random_stream_domains",
    "screening_nesting",
    "phase_i_fixture_role",
})
_FAILURE_POLICY_KEYS = frozenset({
    "failed_fidelity",
    "include_all_expected_cells",
    "failure_rate_endpoint",
    "structural_not_applicable_is_failure",
    "conditional_success_analysis",
})
_PROMOTION_RULE_KEYS = frozenset({
    "version",
    "baseline_method_id",
    "minimum_weighted_itt_gain",
    "maximum_failure_rate",
    "require_no_resource_violation",
    "max_promoted",
    "tie_breakers",
    "null_action",
})
_SAMPLE_SIZE_KEYS = frozenset({
    "method",
    "power",
    "familywise_alpha",
    "planning_alpha",
    "minimum_relevant_noisy_gain",
    "planned_noninferiority_true_difference",
    "target_mean_half_width",
    "failure_rate_half_width",
    "failure_rate_precision_scope",
    "minimum_targets_per_family",
    "maximum_targets_per_family",
    "target_count_increment",
    "minimum_optimization_seed_count",
    "allowed_optimization_seed_counts",
    "allocation_rule",
    "variance_bound_method",
    "variance_model",
    "power_calculation",
    "infeasible_action",
    "trajectory_mcse_target",
    "trajectory_count_min",
    "trajectory_count_max",
    "trajectory_count_rule",
    "trajectory_optional_stopping",
    "blinded_reestimation",
    "maximum_reestimations",
    "reestimation_trigger_fraction",
})
_MULTIPLICITY_KEYS = frozenset({
    "method",
    "familywise_alpha",
    "applicable_primary_contrasts",
    "contrast_definitions",
})
_CONTRAST_DEFINITION_KEYS = frozenset({
    "contrast_id",
    "estimand",
    "hypothesis",
    "margin",
    "applicability",
})
_TARGET_ACCESS_KEYS = frozenset({
    "manifest_visibility_before_final_seal",
    "seed_custody",
    "materialization_before_final_seal",
    "cryptographic_blinding_claim",
    "authorization_model",
})
_ANALYSIS_TEMPLATE_KEYS = frozenset({
    "primary_estimator",
    "cluster_unit",
    "paired_block",
    "uncertainty_components",
    "failure_analysis",
    "claim_rule",
})
_PROMOTION_DECISION_KEYS = frozenset({
    "schema_version",
    "preregistration_checksum",
    "screening_manifest_checksum",
    "screening_evidence_checksum",
    "baseline_configuration_checksum",
    "promoted_method_id",
    "promoted_configuration_checksum",
    "null_fallback",
    "rule_checksum",
    "candidate_summaries",
    "content_checksum",
})
_SCREENING_CANDIDATE_KEYS = frozenset({
    "configuration_schema_version",
    "configuration_checksum",
    "method_id",
    "noisy_training",
    "resource_stratum_id",
    "matching_projection_checksum",
})
_SCREENING_CELL_KEYS = frozenset({
    "cell_id",
    "family_id",
    "stratum_id",
    "qubit_count",
    "target_instance_id",
    "optimization_seed",
    "screening_seed",
})
_SCREENING_MANIFEST_KEYS = frozenset({
    "schema_version",
    "manifest_id",
    "preregistration_checksum",
    "screening_target_manifest_checksum",
    "evaluation_policy_checksum",
    "resource_policy_checksum",
    "baseline_configuration_checksum",
    "candidates",
    "cells",
    "content_checksum",
})
_PROMOTION_OBSERVATION_KEYS = frozenset({
    "configuration_checksum",
    "cell_id",
    "result_schema_version",
    "result_record_checksum",
    "status",
    "noisy_fidelity",
    "resource_value",
    "normalized_work",
    "failure_code",
    "protocol_violations",
})
_SCREENING_EVIDENCE_KEYS = frozenset({
    "schema_version",
    "evidence_id",
    "preregistration_checksum",
    "screening_manifest_checksum",
    "observations",
    "content_checksum",
})
_CANDIDATE_SUMMARY_KEYS = frozenset({
    "configuration_checksum",
    "method_id",
    "weighted_itt_fidelity",
    "failure_rate",
    "max_resource_excess",
    "mean_normalized_work",
    "eligible",
    "ineligibility_reasons",
})
_FINAL_SEAL_KEYS = frozenset({
    "schema_version",
    "seal_id",
    "preregistration_checksum",
    "promotion_decision_checksum",
    "promoted_method_id",
    "promoted_configuration_checksum",
    "comparators",
    "primary_contrasts",
    "confirmatory_target_manifest_checksum",
    "target_count_by_family",
    "optimization_seed_count",
    "fixed_test_trajectory_count",
    "primary_noise_condition",
    "primary_resource_budget",
    "hyperparameters_checksum",
    "execution_source_checksum",
    "analysis_template_checksum",
    "analysis_source_manifest_checksum",
    "sample_size_design_checksum",
    "failure_policy_checksum",
    "content_checksum",
})
_COMPARATOR_KEYS = frozenset({
    "role",
    "method_id",
    "configuration_schema_version",
    "configuration_checksum",
    "matched_to_configuration_checksum",
    "matching_projection_checksum",
})
_PRIMARY_CONTRAST_BINDING_KEYS = frozenset({
    "contrast_id",
    "treatment_configuration_checksum",
    "control_configuration_checksum",
    "paired_block_policy_checksum",
    "matching_projection_checksum",
})
_FINAL_CONFIGURATION_EXECUTION_REF_KEYS = frozenset({
    "schema_version",
    "method_id",
    "configuration_schema_version",
    "configuration_checksum",
    "strategy_schedule",
    "strategy_schedule_checksum",
    "implementation_checksum",
    "scoped_binding_checksum",
    "executable_binding_checksum",
    "content_checksum",
})
_FINAL_CONFIGURATION_EXECUTION_MANIFEST_KEYS = frozenset({
    "schema_version",
    "manifest_id",
    "entries",
    "entry_count",
    "content_checksum",
})
_RESOURCE_BUDGET_KEYS = frozenset({
    "metric",
    "cap_per_chain_edge",
    "normalized_compute_cap",
    "reachable_stratum_manifest_checksum",
})
_SAMPLE_ALLOCATION_KEYS = frozenset({"family_id", "stratum_id", "qubit_count", "target_count"})
_SAMPLE_SIZE_DESIGN_KEYS = frozenset({
    "schema_version",
    "design_id",
    "preregistration_checksum",
    "pilot_nuisance_summary_checksum",
    "calculation_method_id",
    "calculation_source_checksum",
    "contrast_set_checksum",
    "target_population_configuration_checksum",
    "allocations",
    "optimization_seed_count",
    "fixed_test_trajectory_count",
    "achieved_power_by_contrast",
    "expected_primary_mean_half_width",
    "expected_overall_failure_rate_half_width",
    "expected_trajectory_mcse",
    "reestimation_kind",
    "reestimation_parent_checksum",
    "content_checksum",
})
_ANALYSIS_SOURCE_FILE_KEYS = frozenset({"repo_path", "git_blob_id", "content_checksum"})
_ANALYSIS_SOURCE_MANIFEST_KEYS = frozenset({
    "schema_version",
    "manifest_id",
    "preregistration_checksum",
    "analysis_template_checksum",
    "source_commit",
    "entry_point",
    "source_files",
    "environment_lock_checksum",
    "execution_source_manifest_checksum",
    "clean_worktree",
    "content_checksum",
})

_AUTHORIZATION_SENTINEL = object()


def _sequence_of_mappings(value: object, name: str) -> tuple[Mapping[str, object], ...]:
    """Validate and freeze a sequence of JSON mappings.

    Args:
        value: Candidate sequence.
        name: Human-readable location.

    Returns:
        Frozen mappings in source order.

    Raises:
        TypeError: If ``value`` is not a sequence.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of mappings."
        raise TypeError(msg)
    return tuple(freeze_json_mapping(item, f"{name}[{index}]") for index, item in enumerate(value))


def _validate_candidate_methods(value: object) -> tuple[Mapping[str, object], ...]:
    """Validate the preregistered method family.

    Returns:
        The frozen method policies in preregistered order.

    Raises:
        ValueError: If the method set is empty, duplicated, incomplete, or internally inconsistent.
    """
    methods = _sequence_of_mappings(value, "candidate_methods")
    if not methods:
        msg = "candidate_methods must not be empty."
        raise ValueError(msg)
    method_ids: list[str] = []
    for index, method in enumerate(methods):
        name = f"candidate_methods[{index}]"
        require_exact_keys(method, _CANDIDATE_METHOD_KEYS, name)
        method_id = require_slug(method["method_id"], f"{name}.method_id")
        scope = require_slug(method["scope"], f"{name}.scope")
        if scope not in {"all_families", "tfim_only"}:
            msg = f"{name}.scope must be 'all_families' or 'tfim_only'."
            raise ValueError(msg)
        promotion_eligible = require_bool(method["promotion_eligible"], f"{name}.promotion_eligible")
        require_bool(method["noisy_training"], f"{name}.noisy_training")
        role = require_slug(method["role"], f"{name}.role")
        if role not in {"candidate", "required_comparator", "exploratory"}:
            msg = f"{name}.role is unsupported."
            raise ValueError(msg)
        if scope == "tfim_only" and promotion_eligible:
            msg = f"{name} cannot be family-wide promotion eligible when scope is tfim_only."
            raise ValueError(msg)
        method_ids.append(method_id)
    if len(method_ids) != len(set(method_ids)):
        msg = "candidate_methods must have unique method_id values."
        raise ValueError(msg)
    if "layerwise_bmpd_crn_v2" not in method_ids:
        msg = "candidate_methods must contain layerwise_bmpd_crn_v2."
        raise ValueError(msg)
    if "layerwise_bmpd_noiseless" not in method_ids:
        msg = "candidate_methods must contain the otherwise matched noiseless comparator."
        raise ValueError(msg)
    if "adapt_style_state_preparation" not in method_ids:
        msg = "candidate_methods must contain the family-wide operator-growth comparator."
        raise ValueError(msg)
    policies = {cast("str", method["method_id"]): method for method in methods}
    baseline = policies["layerwise_bmpd_crn_v2"]
    if (
        baseline["scope"] != "all_families"
        or baseline["role"] != "required_comparator"
        or baseline["promotion_eligible"] is not True
        or baseline["noisy_training"] is not True
    ):
        msg = "layerwise_bmpd_crn_v2 must be a family-wide, noisy, promotion-eligible required comparator."
        raise ValueError(msg)
    noiseless = policies["layerwise_bmpd_noiseless"]
    if (
        noiseless["scope"] != "all_families"
        or noiseless["role"] != "required_comparator"
        or noiseless["promotion_eligible"] is not False
        or noiseless["noisy_training"] is not False
    ):
        msg = "layerwise_bmpd_noiseless must be a family-wide, non-promotable noiseless required comparator."
        raise ValueError(msg)
    return methods


def _validate_primary_endpoint(value: object) -> Mapping[str, object]:
    """Validate the primary endpoint definition.

    Returns:
        The frozen primary endpoint policy.

    Raises:
        ValueError: If the endpoint differs from the frozen confirmatory definition.
    """
    endpoint = freeze_json_mapping(value, "primary_endpoint")
    require_exact_keys(endpoint, _PRIMARY_ENDPOINT_KEYS, "primary_endpoint")
    if endpoint["metric"] != "fresh_test_noisy_fidelity":
        msg = "primary_endpoint.metric must be 'fresh_test_noisy_fidelity'."
        raise ValueError(msg)
    if endpoint["aggregation"] != "family_weighted_intention_to_treat_mean":
        msg = "primary_endpoint.aggregation must use the family-weighted intention-to-treat mean."
        raise ValueError(msg)
    if endpoint["data_role"] != "confirmatory":
        msg = "primary_endpoint.data_role must be 'confirmatory'."
        raise ValueError(msg)
    if require_bool(endpoint["higher_is_better"], "primary_endpoint.higher_is_better") is not True:
        msg = "primary_endpoint.higher_is_better must be true."
        raise ValueError(msg)
    return endpoint


def _validate_primary_noise(value: object) -> Mapping[str, object]:
    """Validate the fixed primary training and test noise condition.

    Returns:
        The frozen primary noise policy.

    Raises:
        ValueError: If the noise definition or placement is unsupported.
    """
    noise = freeze_json_mapping(value, "primary_noise_condition")
    require_exact_keys(noise, _PRIMARY_NOISE_KEYS, "primary_noise_condition")
    if noise["noise_id"] != "depolarizing_1s_all":
        msg = "primary_noise_condition.noise_id must be 'depolarizing_1s_all'."
        raise ValueError(msg)
    if noise["definition_version"] != "yaqs.state_preparation.noise.v1":
        msg = "primary_noise_condition.definition_version must preserve the Phase I fixed-rate definition."
        raise ValueError(msg)
    strength_scale = require_float(
        noise["strength_scale"],
        "primary_noise_condition.strength_scale",
        minimum=0.0,
    )
    tjm_dt = require_float(noise["tjm_dt"], "primary_noise_condition.tjm_dt", minimum=0.0)
    if strength_scale <= 0.0 or tjm_dt <= 0.0:
        msg = "The primary noisy condition requires positive strength_scale and tjm_dt."
        raise ValueError(msg)
    for field_name in ("training_placement", "test_placement"):
        placement = require_slug(noise[field_name], f"primary_noise_condition.{field_name}")
        if placement not in {"logical_parameterized_gates", "compiled_native_gates"}:
            msg = f"primary_noise_condition.{field_name} has unsupported placement {placement!r}."
            raise ValueError(msg)
    return noise


def _validate_primary_resource(value: object) -> Mapping[str, object]:
    """Validate the one primary resource constraint.

    Returns:
        The frozen primary resource policy.

    Raises:
        ValueError: If the resource definition differs from the frozen policy.
    """
    resource = freeze_json_mapping(value, "primary_resource_constraint")
    require_exact_keys(resource, _PRIMARY_RESOURCE_KEYS, "primary_resource_constraint")
    if resource["metric"] != "native_two_qubit_gates_per_chain_edge":
        msg = "primary_resource_constraint.metric must be 'native_two_qubit_gates_per_chain_edge'."
        raise ValueError(msg)
    require_float(resource["cap_per_chain_edge"], "primary_resource_constraint.cap_per_chain_edge", minimum=0.0)
    if resource["comparison_rule"] != "largest_reachable_at_or_below_cap":
        msg = "primary_resource_constraint.comparison_rule must use reachable strata."
        raise ValueError(msg)
    require_slug(resource["compiler_policy_id"], "primary_resource_constraint.compiler_policy_id")
    require_slug(resource["connectivity"], "primary_resource_constraint.connectivity")
    require_slug(resource["routing_policy"], "primary_resource_constraint.routing_policy")
    if (
        require_bool(
            resource["residual_gap_reporting"],
            "primary_resource_constraint.residual_gap_reporting",
        )
        is not True
    ):
        msg = "primary_resource_constraint.residual_gap_reporting must be true."
        raise ValueError(msg)
    if resource["normalized_compute_cap_source"] != "pilot_final_seal":
        msg = "primary_resource_constraint.normalized_compute_cap_source must be 'pilot_final_seal'."
        raise ValueError(msg)
    return resource


def _validate_target_population(value: object) -> tuple[Mapping[str, object], tuple[str, ...]]:
    """Validate target-family distributions and return their identifiers.

    Returns:
        The frozen population policy and its ordered family identifiers.

    Raises:
        ValueError: If families, qubit strata, or the instance identity policy are inconsistent.
    """
    policy = freeze_json_mapping(value, "target_population_policy")
    require_exact_keys(policy, _TARGET_POPULATION_KEYS, "target_population_policy")
    generator_schema = require_slug(
        policy["generator_schema_version"],
        "target_population_policy.generator_schema_version",
    )
    if generator_schema != "yaqs.state_preparation.phase2.targets.v2":
        msg = "target_population_policy.generator_schema_version must be the corrected targets.v2 policy."
        raise ValueError(msg)
    families = _sequence_of_mappings(policy["families"], "target_population_policy.families")
    family_ids: list[str] = []
    for index, family in enumerate(families):
        name = f"target_population_policy.families[{index}]"
        require_exact_keys(family, _TARGET_FAMILY_KEYS, name)
        family_id = require_slug(family["family_id"], f"{name}.family_id")
        freeze_json_mapping(family["parameter_distribution"], f"{name}.parameter_distribution")
        require_string_sequence(family["strata"], f"{name}.strata", minimum_length=1, unique=True, slugs=True)
        family_ids.append(family_id)
    if tuple(family_ids) != PRIMARY_TARGET_FAMILIES:
        msg = f"Target families must be ordered as {PRIMARY_TARGET_FAMILIES!r}."
        raise ValueError(msg)
    actual_strata = {
        cast("str", family["family_id"]): require_string_sequence(
            family["strata"],
            f"target_population_policy.{family['family_id']}.strata",
            minimum_length=1,
            unique=True,
            slugs=True,
        )
        for family in families
    }
    if actual_strata != PRIMARY_FAMILY_STRATA:
        msg = "Target-family strata do not match the frozen primary allocation."
        raise ValueError(msg)
    tfim_distribution = cast(
        "Mapping[str, object]",
        next(family for family in families if family["family_id"] == "tfim_ground_state")["parameter_distribution"],
    )
    if "eigensolver" in tfim_distribution:
        msg = "The corrected target policy must not retain the ambiguous TFIM eigensolver field."
        raise ValueError(msg)
    primary_qubits = cast("Sequence[object]", policy["primary_qubit_counts"])
    secondary_qubits = cast("Sequence[object]", policy["secondary_qubit_counts"])
    primary_values = tuple(require_int(item, "primary_qubit_counts item", minimum=2) for item in primary_qubits)
    secondary_values = tuple(require_int(item, "secondary_qubit_counts item", minimum=2) for item in secondary_qubits)
    if not primary_values or len(primary_values) != len(set(primary_values)):
        msg = "primary_qubit_counts must be nonempty and unique."
        raise ValueError(msg)
    if set(primary_values) & set(secondary_values):
        msg = "primary and secondary qubit counts must be disjoint."
        raise ValueError(msg)
    if policy["instance_id_policy"] != "hash_of_population_family_stratum_qubits_and_seed":
        msg = "target_population_policy.instance_id_policy is not the frozen v1 policy."
        raise ValueError(msg)
    freeze_json_mapping(policy["rng_policy"], "target_population_policy.rng_policy")
    numeric_policy = freeze_json_mapping(
        policy["numeric_policy"],
        "target_population_policy.numeric_policy",
    )
    require_exact_keys(
        numeric_policy,
        _TARGET_NUMERIC_POLICY_KEYS,
        "target_population_policy.numeric_policy",
    )
    expected_numeric_text = {
        "authorized_state_solver": "numpy_linalg_eigh_dense_hermitian",
        "basis_bit_order": "little_endian_site_i_is_bit_i",
        "complex_precision": "complex128",
        "generation_numpy_version": "2.4.6",
        "global_phase_rule": "largest_magnitude_lowest_index_component_real_nonnegative",
        "manifest_spectrum_solver": "numpy_linalg_eigvalsh_dense_hermitian",
        "real_precision": "float64",
        "standard_normal_api": "Generator.standard_normal",
        "uniform_interval": "half_open",
    }
    for field_name, expected_value in expected_numeric_text.items():
        if numeric_policy[field_name] != expected_value:
            msg = f"target_population_policy.numeric_policy.{field_name} differs from the corrected policy."
            raise ValueError(msg)
    for field_name in ("spectrum_agreement_rtol", "spectrum_agreement_atol"):
        tolerance = require_float(
            numeric_policy[field_name],
            f"target_population_policy.numeric_policy.{field_name}",
            minimum=0.0,
        )
        if not math.isclose(tolerance, 1e-13, rel_tol=0.0, abs_tol=0.0):
            msg = f"target_population_policy.numeric_policy.{field_name} must be exactly 1e-13."
            raise ValueError(msg)
    freeze_json_mapping(
        policy["role_allocation_policy"],
        "target_population_policy.role_allocation_policy",
    )
    return policy, tuple(family_ids)


def _validate_family_weights(value: object, family_ids: tuple[str, ...]) -> Mapping[str, object]:
    """Validate exact normalized target-family weights.

    Returns:
        The frozen family-weight mapping.

    Raises:
        ValueError: If keys differ from the target families or weights are not positive and normalized.
    """
    weights = freeze_json_mapping(value, "target_family_weights")
    if frozenset(weights) != frozenset(family_ids):
        msg = "target_family_weights keys must exactly match the target families."
        raise ValueError(msg)
    normalized = [
        require_float(weights[family_id], f"target_family_weights.{family_id}", minimum=0.0, maximum=1.0)
        for family_id in family_ids
    ]
    if not math.isclose(math.fsum(normalized), 1.0, rel_tol=0.0, abs_tol=1e-12):
        msg = "target_family_weights must sum to one."
        raise ValueError(msg)
    if any(weight <= 0.0 for weight in normalized):
        msg = "Every primary target family must have positive weight."
        raise ValueError(msg)
    return weights


def _validate_data_roles(value: object) -> Mapping[str, object]:
    """Validate target-instance and random-stream separation rules.

    Returns:
        The frozen data-role policy.

    Raises:
        ValueError: If role ordering, domain separation, or the Phase I fixture role is inconsistent.
    """
    policy = freeze_json_mapping(value, "data_role_policy")
    require_exact_keys(policy, _DATA_ROLE_KEYS, "data_role_policy")
    roles = require_string_sequence(policy["roles"], "data_role_policy.roles", unique=True, slugs=True)
    if roles != DATA_ROLES:
        msg = f"data_role_policy.roles must be ordered as {DATA_ROLES!r}."
        raise ValueError(msg)
    target_domains = require_string_sequence(
        policy["target_instance_domains"],
        "data_role_policy.target_instance_domains",
        unique=True,
        slugs=True,
    )
    if target_domains != ("development", "screening_selection", "confirmatory"):
        msg = "target_instance_domains must separate development, screening selection, and confirmation."
        raise ValueError(msg)
    stream_domains = require_string_sequence(
        policy["random_stream_domains"],
        "data_role_policy.random_stream_domains",
        unique=True,
        slugs=True,
    )
    expected_stream_domains = (
        "target_generation",
        "initialization",
        "optimizer_ordering",
        "training_trajectory",
        "checkpoint_validation",
        "pilot_evaluation",
        "screening_selection",
        "confirmatory_test",
    )
    if stream_domains != expected_stream_domains:
        msg = "random_stream_domains do not match the frozen separation policy."
        raise ValueError(msg)
    require_nonempty_text(policy["screening_nesting"], "data_role_policy.screening_nesting")
    if policy["phase_i_fixture_role"] != "secondary_benchmark":
        msg = "data_role_policy.phase_i_fixture_role must be 'secondary_benchmark'."
        raise ValueError(msg)
    return policy


def _validate_failure_policy(value: object) -> Mapping[str, object]:
    """Validate unconditional failure handling.

    Returns:
        The frozen failure policy.

    Raises:
        ValueError: If failures could be omitted or used only conditionally.
    """
    policy = freeze_json_mapping(value, "failure_policy")
    require_exact_keys(policy, _FAILURE_POLICY_KEYS, "failure_policy")
    require_float(policy["failed_fidelity"], "failure_policy.failed_fidelity", minimum=0.0, maximum=1.0)
    if require_bool(policy["include_all_expected_cells"], "failure_policy.include_all_expected_cells") is not True:
        msg = "failure_policy.include_all_expected_cells must be true."
        raise ValueError(msg)
    if require_bool(policy["failure_rate_endpoint"], "failure_policy.failure_rate_endpoint") is not True:
        msg = "failure_policy.failure_rate_endpoint must be true."
        raise ValueError(msg)
    if require_bool(
        policy["structural_not_applicable_is_failure"],
        "failure_policy.structural_not_applicable_is_failure",
    ):
        msg = "Structurally inapplicable method-family cells must not count as failures."
        raise ValueError(msg)
    if policy["conditional_success_analysis"] != "secondary_only":
        msg = "failure_policy.conditional_success_analysis must be 'secondary_only'."
        raise ValueError(msg)
    return policy


def _validate_promotion_rule(value: object) -> Mapping[str, object]:
    """Validate the mechanical one-candidate promotion rule.

    Returns:
        The frozen promotion rule.

    Raises:
        ValueError: If the baseline, thresholds, tie breakers, or null action differ.
    """
    rule = freeze_json_mapping(value, "promotion_rule")
    require_exact_keys(rule, _PROMOTION_RULE_KEYS, "promotion_rule")
    if rule["version"] != PROMOTION_RULE_VERSION:
        msg = f"promotion_rule.version must be {PROMOTION_RULE_VERSION!r}."
        raise ValueError(msg)
    if rule["baseline_method_id"] != "layerwise_bmpd_crn_v2":
        msg = "promotion_rule.baseline_method_id must be 'layerwise_bmpd_crn_v2'."
        raise ValueError(msg)
    require_float(rule["minimum_weighted_itt_gain"], "promotion_rule.minimum_weighted_itt_gain", minimum=0.0)
    require_float(rule["maximum_failure_rate"], "promotion_rule.maximum_failure_rate", minimum=0.0, maximum=1.0)
    if (
        require_bool(
            rule["require_no_resource_violation"],
            "promotion_rule.require_no_resource_violation",
        )
        is not True
    ):
        msg = "promotion_rule.require_no_resource_violation must be true."
        raise ValueError(msg)
    if require_int(rule["max_promoted"], "promotion_rule.max_promoted", minimum=1) != 1:
        msg = "promotion_rule.max_promoted must be one."
        raise ValueError(msg)
    tie_breakers = require_string_sequence(
        rule["tie_breakers"],
        "promotion_rule.tie_breakers",
        minimum_length=1,
        unique=True,
        slugs=True,
    )
    expected_ties = ("failure_rate", "resource_excess", "normalized_work", "configuration_checksum")
    if tie_breakers != expected_ties:
        msg = f"promotion_rule.tie_breakers must be {expected_ties!r}."
        raise ValueError(msg)
    if rule["null_action"] != "promote_baseline":
        msg = "promotion_rule.null_action must be 'promote_baseline'."
        raise ValueError(msg)
    return rule


def _validate_sample_size_policy(value: object) -> Mapping[str, object]:
    """Validate cluster-aware sample-size and fixed-trajectory rules.

    Returns:
        The frozen sample-size policy.

    Raises:
        ValueError: If the policy permits optional stopping or has inconsistent trajectory bounds.
    """
    policy = freeze_json_mapping(value, "sample_size_policy")
    require_exact_keys(policy, _SAMPLE_SIZE_KEYS, "sample_size_policy")
    if policy["method"] != "cluster_aware_paired_difference_v1":
        msg = "sample_size_policy.method must be 'cluster_aware_paired_difference_v1'."
        raise ValueError(msg)
    for name in ("power", "familywise_alpha", "planning_alpha"):
        value_as_float = require_float(policy[name], f"sample_size_policy.{name}", minimum=0.0, maximum=1.0)
        if value_as_float in {0.0, 1.0}:
            msg = f"sample_size_policy.{name} must be strictly between zero and one."
            raise ValueError(msg)
    for name in (
        "minimum_relevant_noisy_gain",
        "target_mean_half_width",
        "failure_rate_half_width",
        "trajectory_mcse_target",
    ):
        positive_value = require_float(policy[name], f"sample_size_policy.{name}", minimum=0.0, maximum=1.0)
        if positive_value <= 0.0:
            msg = f"sample_size_policy.{name} must be positive."
            raise ValueError(msg)
    require_float(
        policy["planned_noninferiority_true_difference"],
        "sample_size_policy.planned_noninferiority_true_difference",
        minimum=-1.0,
        maximum=1.0,
    )
    minimum_targets = require_int(
        policy["minimum_targets_per_family"],
        "sample_size_policy.minimum_targets_per_family",
        minimum=2,
    )
    maximum_targets = require_int(
        policy["maximum_targets_per_family"],
        "sample_size_policy.maximum_targets_per_family",
        minimum=minimum_targets,
    )
    increment = require_int(
        policy["target_count_increment"],
        "sample_size_policy.target_count_increment",
        minimum=1,
    )
    if minimum_targets % increment != 0 or maximum_targets % increment != 0:
        msg = "Target-count bounds must be exact multiples of target_count_increment."
        raise ValueError(msg)
    minimum_seeds = require_int(
        policy["minimum_optimization_seed_count"],
        "sample_size_policy.minimum_optimization_seed_count",
        minimum=2,
    )
    allowed_seeds = tuple(
        require_int(item, "sample_size_policy.allowed_optimization_seed_counts item", minimum=minimum_seeds)
        for item in cast("Sequence[object]", policy["allowed_optimization_seed_counts"])
    )
    if not allowed_seeds or allowed_seeds != tuple(sorted(set(allowed_seeds))) or allowed_seeds[0] != minimum_seeds:
        msg = "allowed_optimization_seed_counts must be sorted, unique, and begin at the minimum."
        raise ValueError(msg)
    if policy["allocation_rule"] != "equal_family_and_within_family_strata_primary_q6":
        msg = "sample_size_policy.allocation_rule differs from the frozen balanced design."
        raise ValueError(msg)
    if policy["variance_bound_method"] != "upper_confidence_bound_target_cluster_components":
        msg = "sample_size_policy.variance_bound_method differs from the frozen conservative rule."
        raise ValueError(msg)
    if (
        policy["variance_model"]
        != "target_over_n_plus_optimizer_over_n_s_plus_mc_over_n_s_ntraj_then_squared_family_weights"
    ):
        msg = "sample_size_policy.variance_model differs from the frozen cluster variance formula."
        raise ValueError(msg)
    if policy["power_calculation"] != "normal_approximation_one_sided_worst_holm_alpha":
        msg = "sample_size_policy.power_calculation differs from the frozen planning formula."
        raise ValueError(msg)
    if policy["failure_rate_precision_scope"] != "overall_family_weighted_method_marginal_descriptive":
        msg = "sample_size_policy.failure_rate_precision_scope differs from the frozen estimand."
        raise ValueError(msg)
    if policy["infeasible_action"] != "abort_before_final_seal":
        msg = "sample_size_policy.infeasible_action must abort an underpowered design."
        raise ValueError(msg)
    minimum = require_int(policy["trajectory_count_min"], "sample_size_policy.trajectory_count_min", minimum=2)
    maximum = require_int(policy["trajectory_count_max"], "sample_size_policy.trajectory_count_max", minimum=minimum)
    if maximum < minimum:
        msg = "trajectory_count_max must not be below trajectory_count_min."
        raise ValueError(msg)
    if require_bool(
        policy["trajectory_optional_stopping"],
        "sample_size_policy.trajectory_optional_stopping",
    ):
        msg = "Outcome-dependent trajectory optional stopping is forbidden."
        raise ValueError(msg)
    if policy["trajectory_count_rule"] != "next_power_of_two_from_pilot_variance_upper_bound":
        msg = "sample_size_policy.trajectory_count_rule differs from the frozen fixed-count rule."
        raise ValueError(msg)
    if policy["blinded_reestimation"] != "at_most_once_halfway_nuisance_only_non_decreasing":
        msg = "sample_size_policy.blinded_reestimation has an unsupported scope."
        raise ValueError(msg)
    if require_int(policy["maximum_reestimations"], "sample_size_policy.maximum_reestimations") != 1:
        msg = "sample_size_policy.maximum_reestimations must be one."
        raise ValueError(msg)
    trigger_fraction = require_float(
        policy["reestimation_trigger_fraction"],
        "sample_size_policy.reestimation_trigger_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if not math.isclose(trigger_fraction, 0.5, rel_tol=0.0, abs_tol=0.0):
        msg = "sample_size_policy.reestimation_trigger_fraction must be 0.5."
        raise ValueError(msg)
    return policy


def _validate_multiplicity_policy(value: object) -> Mapping[str, object]:
    """Validate multiplicity control for applicable primary contrasts.

    Returns:
        The frozen multiplicity policy.

    Raises:
        ValueError: If the method or primary contrast set differs from the frozen policy.
    """
    policy = freeze_json_mapping(value, "multiplicity_policy")
    require_exact_keys(policy, _MULTIPLICITY_KEYS, "multiplicity_policy")
    if policy["method"] != "holm":
        msg = "multiplicity_policy.method must be 'holm'."
        raise ValueError(msg)
    require_float(policy["familywise_alpha"], "multiplicity_policy.familywise_alpha", minimum=0.0, maximum=1.0)
    contrasts = require_string_sequence(
        policy["applicable_primary_contrasts"],
        "multiplicity_policy.applicable_primary_contrasts",
        minimum_length=1,
        unique=True,
        slugs=True,
    )
    if contrasts != ("noisy_vs_noiseless", "promoted_vs_layerwise_v2_if_distinct"):
        msg = "multiplicity_policy.applicable_primary_contrasts does not match the frozen protocol."
        raise ValueError(msg)
    definitions = _sequence_of_mappings(
        policy["contrast_definitions"],
        "multiplicity_policy.contrast_definitions",
    )
    if len(definitions) != len(contrasts):
        msg = "multiplicity_policy must define every applicable primary contrast exactly once."
        raise ValueError(msg)
    expected_definitions = (
        (
            "noisy_vs_noiseless",
            "family_weighted_itt_mean_difference",
            "superiority",
            0.0,
            "always",
        ),
        (
            "promoted_vs_layerwise_v2_if_distinct",
            "family_weighted_itt_mean_difference",
            "noninferiority",
            -0.01,
            "promoted_method_is_distinct_from_layerwise_v2",
        ),
    )
    for index, (definition, expected) in enumerate(zip(definitions, expected_definitions, strict=True)):
        name = f"multiplicity_policy.contrast_definitions[{index}]"
        require_exact_keys(definition, _CONTRAST_DEFINITION_KEYS, name)
        actual = (
            require_slug(definition["contrast_id"], f"{name}.contrast_id"),
            require_slug(definition["estimand"], f"{name}.estimand"),
            require_slug(definition["hypothesis"], f"{name}.hypothesis"),
            require_float(definition["margin"], f"{name}.margin"),
            require_slug(definition["applicability"], f"{name}.applicability"),
        )
        if actual != expected:
            msg = f"{name} does not match the frozen estimand, hypothesis, margin, and applicability."
            raise ValueError(msg)
    return policy


def _validate_target_access_policy(value: object) -> Mapping[str, object]:
    """Validate external custody and pre-seal materialization prohibition.

    Returns:
        The frozen target-access policy.

    Raises:
        ValueError: If the policy exposes confirmatory information or overstates the access guard.
    """
    policy = freeze_json_mapping(value, "target_access_policy")
    require_exact_keys(policy, _TARGET_ACCESS_KEYS, "target_access_policy")
    if policy["manifest_visibility_before_final_seal"] != "checksum_only":
        msg = "Only the confirmatory manifest checksum may be visible before the final seal."
        raise ValueError(msg)
    if policy["seed_custody"] != "independent_external_custodian":
        msg = "Confirmatory target seeds must remain with an independent external custodian."
        raise ValueError(msg)
    if require_bool(
        policy["materialization_before_final_seal"],
        "target_access_policy.materialization_before_final_seal",
    ):
        msg = "Confirmatory targets must not be materialized before the final seal."
        raise ValueError(msg)
    if require_bool(
        policy["cryptographic_blinding_claim"],
        "target_access_policy.cryptographic_blinding_claim",
    ):
        msg = "The in-process authorization guard must not be called cryptographic blinding."
        raise ValueError(msg)
    if policy["authorization_model"] != "opaque_in_process_accidental_access_guard":
        msg = "target_access_policy.authorization_model is not the frozen guard model."
        raise ValueError(msg)
    return policy


def _validate_analysis_template(value: object) -> Mapping[str, object]:
    """Validate the frozen primary analysis template.

    Returns:
        The frozen primary analysis template.
    """
    template = freeze_json_mapping(value, "analysis_template")
    require_exact_keys(template, _ANALYSIS_TEMPLATE_KEYS, "analysis_template")
    for name in ("primary_estimator", "cluster_unit", "paired_block", "failure_analysis", "claim_rule"):
        require_nonempty_text(template[name], f"analysis_template.{name}")
    require_string_sequence(
        template["uncertainty_components"],
        "analysis_template.uncertainty_components",
        minimum_length=1,
        unique=True,
        slugs=True,
    )
    return template


@dataclass(frozen=True, slots=True)
class InitialPreregistration:
    """Immutable initial Phase II scientific protocol seal."""

    protocol_id: str
    phase_i_baseline_commit: str
    implementation_plan_commit: str
    legacy_evidence_audit_checksum: str
    scientific_questions: tuple[str, ...]
    confirmatory_questions: tuple[str, ...]
    candidate_methods: tuple[Mapping[str, object], ...]
    primary_endpoint: Mapping[str, object]
    primary_noise_condition: Mapping[str, object]
    primary_resource_constraint: Mapping[str, object]
    target_population_policy: Mapping[str, object]
    target_family_weights: Mapping[str, object]
    data_role_policy: Mapping[str, object]
    failure_policy: Mapping[str, object]
    promotion_rule: Mapping[str, object]
    sample_size_policy: Mapping[str, object]
    multiplicity_policy: Mapping[str, object]
    target_access_policy: Mapping[str, object]
    analysis_template: Mapping[str, object]
    schema_version: str = field(default=PREREGISTRATION_SCHEMA_VERSION, init=False)
    final_confirmation_schema_version: str = field(default=CONFIRMATION_SEAL_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate every frozen scientific and access-control decision."""
        object.__setattr__(self, "protocol_id", require_slug(self.protocol_id, "protocol_id"))
        object.__setattr__(
            self,
            "phase_i_baseline_commit",
            require_git_commit(self.phase_i_baseline_commit, "phase_i_baseline_commit"),
        )
        object.__setattr__(
            self,
            "implementation_plan_commit",
            require_git_commit(self.implementation_plan_commit, "implementation_plan_commit"),
        )
        object.__setattr__(
            self,
            "legacy_evidence_audit_checksum",
            require_checksum(self.legacy_evidence_audit_checksum, "legacy_evidence_audit_checksum"),
        )
        object.__setattr__(
            self,
            "scientific_questions",
            require_string_sequence(
                self.scientific_questions,
                "scientific_questions",
                minimum_length=1,
                unique=True,
            ),
        )
        object.__setattr__(
            self,
            "confirmatory_questions",
            require_string_sequence(
                self.confirmatory_questions,
                "confirmatory_questions",
                minimum_length=1,
                unique=True,
            ),
        )
        object.__setattr__(self, "candidate_methods", _validate_candidate_methods(self.candidate_methods))
        object.__setattr__(self, "primary_endpoint", _validate_primary_endpoint(self.primary_endpoint))
        object.__setattr__(
            self,
            "primary_noise_condition",
            _validate_primary_noise(self.primary_noise_condition),
        )
        object.__setattr__(
            self,
            "primary_resource_constraint",
            _validate_primary_resource(self.primary_resource_constraint),
        )
        target_policy, family_ids = _validate_target_population(self.target_population_policy)
        object.__setattr__(self, "target_population_policy", target_policy)
        object.__setattr__(
            self,
            "target_family_weights",
            _validate_family_weights(self.target_family_weights, family_ids),
        )
        object.__setattr__(self, "data_role_policy", _validate_data_roles(self.data_role_policy))
        object.__setattr__(self, "failure_policy", _validate_failure_policy(self.failure_policy))
        object.__setattr__(self, "promotion_rule", _validate_promotion_rule(self.promotion_rule))
        object.__setattr__(
            self,
            "sample_size_policy",
            _validate_sample_size_policy(self.sample_size_policy),
        )
        object.__setattr__(
            self,
            "multiplicity_policy",
            _validate_multiplicity_policy(self.multiplicity_policy),
        )
        object.__setattr__(
            self,
            "target_access_policy",
            _validate_target_access_policy(self.target_access_policy),
        )
        object.__setattr__(self, "analysis_template", _validate_analysis_template(self.analysis_template))

    @property
    def content_checksum(self) -> str:
        """Checksum of all preregistered content."""
        return canonical_checksum(self._content_dict())

    @property
    def promotion_rule_checksum(self) -> str:
        """Independently addressable promotion-rule checksum."""
        return canonical_checksum(self.promotion_rule)

    @property
    def failure_policy_checksum(self) -> str:
        """Independently addressable failure-policy checksum."""
        return canonical_checksum(self.failure_policy)

    @property
    def analysis_template_checksum(self) -> str:
        """Independently addressable primary-analysis checksum."""
        return canonical_checksum(self.analysis_template)

    @property
    def paired_block_policy_checksum(self) -> str:
        """Checksum of the primary paired-block identity."""
        return canonical_checksum({"paired_block": self.analysis_template["paired_block"]})

    @property
    def contrast_set_checksum(self) -> str:
        """Checksum of the complete preregistered primary contrast set."""
        return canonical_checksum(self.multiplicity_policy["contrast_definitions"])

    @property
    def target_population_configuration_checksum(self) -> str:
        """Checksum of the complete target-population construction policy."""
        return canonical_checksum(self.target_population_policy)

    def method_policy(self, method_id: str) -> Mapping[str, object]:
        """Return one preregistered method policy.

        Args:
            method_id: Method identifier to resolve.

        Returns:
            The immutable method policy.

        Raises:
            KeyError: If the identifier is not preregistered.
        """
        normalized = require_slug(method_id, "method_id")
        for method in self.candidate_methods:
            if method["method_id"] == normalized:
                return method
        raise KeyError(normalized)

    def _content_dict(self) -> dict[str, object]:
        """Return the checksum-covered preregistration payload."""
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "phase_i_baseline_commit": self.phase_i_baseline_commit,
            "implementation_plan_commit": self.implementation_plan_commit,
            "legacy_evidence_audit_checksum": self.legacy_evidence_audit_checksum,
            "scientific_questions": list(self.scientific_questions),
            "confirmatory_questions": list(self.confirmatory_questions),
            "candidate_methods": [thaw_json_mapping(method) for method in self.candidate_methods],
            "primary_endpoint": thaw_json_mapping(self.primary_endpoint),
            "primary_noise_condition": thaw_json_mapping(self.primary_noise_condition),
            "primary_resource_constraint": thaw_json_mapping(self.primary_resource_constraint),
            "target_population_policy": thaw_json_mapping(self.target_population_policy),
            "target_family_weights": thaw_json_mapping(self.target_family_weights),
            "data_role_policy": thaw_json_mapping(self.data_role_policy),
            "failure_policy": thaw_json_mapping(self.failure_policy),
            "promotion_rule": thaw_json_mapping(self.promotion_rule),
            "sample_size_policy": thaw_json_mapping(self.sample_size_policy),
            "multiplicity_policy": thaw_json_mapping(self.multiplicity_policy),
            "target_access_policy": thaw_json_mapping(self.target_access_policy),
            "analysis_template": thaw_json_mapping(self.analysis_template),
            "final_confirmation_schema_version": self.final_confirmation_schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the sealed JSON-native preregistration."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> InitialPreregistration:
        """Construct and checksum-verify a preregistration.

        Args:
            data: Sealed preregistration mapping.

        Returns:
            The validated immutable preregistration.

        Raises:
            ValueError: If a schema version or normalized checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_PREREGISTRATION_KEYS, name="initial preregistration")
        if mapping["schema_version"] != PREREGISTRATION_SCHEMA_VERSION:
            msg = f"schema_version must be {PREREGISTRATION_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        if mapping["final_confirmation_schema_version"] != CONFIRMATION_SEAL_SCHEMA_VERSION:
            msg = f"final_confirmation_schema_version must be {CONFIRMATION_SEAL_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        preregistration = cls(
            protocol_id=cast("str", mapping["protocol_id"]),
            phase_i_baseline_commit=cast("str", mapping["phase_i_baseline_commit"]),
            implementation_plan_commit=cast("str", mapping["implementation_plan_commit"]),
            legacy_evidence_audit_checksum=cast("str", mapping["legacy_evidence_audit_checksum"]),
            scientific_questions=cast("tuple[str, ...]", mapping["scientific_questions"]),
            confirmatory_questions=cast("tuple[str, ...]", mapping["confirmatory_questions"]),
            candidate_methods=cast("tuple[Mapping[str, object], ...]", mapping["candidate_methods"]),
            primary_endpoint=cast("Mapping[str, object]", mapping["primary_endpoint"]),
            primary_noise_condition=cast("Mapping[str, object]", mapping["primary_noise_condition"]),
            primary_resource_constraint=cast("Mapping[str, object]", mapping["primary_resource_constraint"]),
            target_population_policy=cast("Mapping[str, object]", mapping["target_population_policy"]),
            target_family_weights=cast("Mapping[str, object]", mapping["target_family_weights"]),
            data_role_policy=cast("Mapping[str, object]", mapping["data_role_policy"]),
            failure_policy=cast("Mapping[str, object]", mapping["failure_policy"]),
            promotion_rule=cast("Mapping[str, object]", mapping["promotion_rule"]),
            sample_size_policy=cast("Mapping[str, object]", mapping["sample_size_policy"]),
            multiplicity_policy=cast("Mapping[str, object]", mapping["multiplicity_policy"]),
            target_access_policy=cast("Mapping[str, object]", mapping["target_access_policy"]),
            analysis_template=cast("Mapping[str, object]", mapping["analysis_template"]),
        )
        supplied = cast("str", mapping["content_checksum"])
        if preregistration.content_checksum != supplied:
            msg = (
                "Initial preregistration checksum changed during normalization: "
                f"expected {supplied}, got {preregistration.content_checksum}."
            )
            raise ValueError(msg)
        return preregistration

    @classmethod
    def from_json(cls, payload: str) -> InitialPreregistration:
        """Construct a preregistration from canonical sealed JSON.

        Args:
            payload: Canonical JSON text.

        Returns:
            The validated immutable preregistration.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class ScreeningCandidateRef:
    """One configuration in the sealed screening candidate universe."""

    configuration_schema_version: str
    configuration_checksum: str
    method_id: str
    noisy_training: bool
    resource_stratum_id: str
    matching_projection_checksum: str | None

    def __post_init__(self) -> None:
        """Validate candidate identity and its optional matching projection."""
        object.__setattr__(
            self,
            "configuration_schema_version",
            require_slug(self.configuration_schema_version, "configuration_schema_version"),
        )
        object.__setattr__(
            self,
            "configuration_checksum",
            require_checksum(self.configuration_checksum, "configuration_checksum"),
        )
        object.__setattr__(self, "method_id", require_slug(self.method_id, "method_id"))
        object.__setattr__(self, "noisy_training", require_bool(self.noisy_training, "noisy_training"))
        object.__setattr__(
            self,
            "resource_stratum_id",
            require_slug(self.resource_stratum_id, "resource_stratum_id"),
        )
        if self.matching_projection_checksum is not None:
            object.__setattr__(
                self,
                "matching_projection_checksum",
                require_checksum(self.matching_projection_checksum, "matching_projection_checksum"),
            )

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native candidate reference."""
        return {
            "configuration_schema_version": self.configuration_schema_version,
            "configuration_checksum": self.configuration_checksum,
            "method_id": self.method_id,
            "noisy_training": self.noisy_training,
            "resource_stratum_id": self.resource_stratum_id,
            "matching_projection_checksum": self.matching_projection_checksum,
        }

    @classmethod
    def from_dict(cls, data: object) -> ScreeningCandidateRef:
        """Construct a candidate reference from an exact JSON object.

        Returns:
            The validated immutable candidate reference.
        """
        mapping = require_mapping(data, "screening candidate")
        require_exact_keys(mapping, _SCREENING_CANDIDATE_KEYS, "screening candidate")
        return cls(
            configuration_schema_version=cast("str", mapping["configuration_schema_version"]),
            configuration_checksum=cast("str", mapping["configuration_checksum"]),
            method_id=cast("str", mapping["method_id"]),
            noisy_training=cast("bool", mapping["noisy_training"]),
            resource_stratum_id=cast("str", mapping["resource_stratum_id"]),
            matching_projection_checksum=cast("str | None", mapping["matching_projection_checksum"]),
        )


@dataclass(frozen=True, slots=True)
class ScreeningCell:
    """One fully identified outer screening-selection cell."""

    cell_id: str
    family_id: str
    stratum_id: str
    qubit_count: int
    target_instance_id: str
    optimization_seed: int
    screening_seed: int
    data_role: str = field(default="screening_selection", init=False)

    def __post_init__(self) -> None:
        """Validate family, stratum, target, qubit, and seed identities.

        Raises:
            ValueError: If the family or stratum is not in the primary population.
        """
        object.__setattr__(self, "cell_id", require_slug(self.cell_id, "cell_id"))
        family_id = require_slug(self.family_id, "family_id")
        if family_id not in PRIMARY_TARGET_FAMILIES:
            msg = f"family_id must be one of {PRIMARY_TARGET_FAMILIES!r}."
            raise ValueError(msg)
        object.__setattr__(self, "family_id", family_id)
        stratum_id = require_slug(self.stratum_id, "stratum_id")
        if stratum_id not in PRIMARY_FAMILY_STRATA[family_id]:
            msg = f"stratum_id {stratum_id!r} is not registered for family {family_id!r}."
            raise ValueError(msg)
        object.__setattr__(self, "stratum_id", stratum_id)
        object.__setattr__(self, "qubit_count", require_int(self.qubit_count, "qubit_count", minimum=2))
        object.__setattr__(
            self,
            "target_instance_id",
            require_slug(self.target_instance_id, "target_instance_id"),
        )
        object.__setattr__(
            self,
            "optimization_seed",
            require_int(self.optimization_seed, "optimization_seed"),
        )
        object.__setattr__(self, "screening_seed", require_int(self.screening_seed, "screening_seed"))

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native screening cell."""
        return {
            "cell_id": self.cell_id,
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "qubit_count": self.qubit_count,
            "target_instance_id": self.target_instance_id,
            "optimization_seed": self.optimization_seed,
            "screening_seed": self.screening_seed,
        }

    @classmethod
    def from_dict(cls, data: object) -> ScreeningCell:
        """Construct a screening cell from an exact JSON object.

        Returns:
            The validated immutable screening cell.
        """
        mapping = require_mapping(data, "screening cell")
        require_exact_keys(mapping, _SCREENING_CELL_KEYS, "screening cell")
        return cls(
            cell_id=cast("str", mapping["cell_id"]),
            family_id=cast("str", mapping["family_id"]),
            stratum_id=cast("str", mapping["stratum_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            target_instance_id=cast("str", mapping["target_instance_id"]),
            optimization_seed=cast("int", mapping["optimization_seed"]),
            screening_seed=cast("int", mapping["screening_seed"]),
        )


@dataclass(frozen=True, slots=True)
class ScreeningManifest:
    """Checksum-sealed complete screening candidate and cell universe."""

    manifest_id: str
    preregistration_checksum: str
    screening_target_manifest_checksum: str
    evaluation_policy_checksum: str
    resource_policy_checksum: str
    baseline_configuration_checksum: str
    candidates: tuple[ScreeningCandidateRef, ...]
    cells: tuple[ScreeningCell, ...]
    schema_version: str = field(default=SCREENING_MANIFEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate immutable identities and complete primary-family coverage.

        Raises:
            TypeError: If candidates or cells contain unsupported record types.
            ValueError: If identities are duplicated, incomplete, or inconsistent.
        """
        object.__setattr__(self, "manifest_id", require_slug(self.manifest_id, "manifest_id"))
        for name in (
            "preregistration_checksum",
            "screening_target_manifest_checksum",
            "evaluation_policy_checksum",
            "resource_policy_checksum",
            "baseline_configuration_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        candidates = tuple(self.candidates)
        cells = tuple(self.cells)
        if not candidates or not all(isinstance(candidate, ScreeningCandidateRef) for candidate in candidates):
            msg = "candidates must contain ScreeningCandidateRef values."
            raise TypeError(msg)
        if not cells or not all(isinstance(cell, ScreeningCell) for cell in cells):
            msg = "cells must contain ScreeningCell values."
            raise TypeError(msg)
        candidate_checksums = tuple(candidate.configuration_checksum for candidate in candidates)
        cell_ids = tuple(cell.cell_id for cell in cells)
        if len(candidate_checksums) != len(set(candidate_checksums)):
            msg = "Screening candidates must have unique configuration checksums."
            raise ValueError(msg)
        if len(cell_ids) != len(set(cell_ids)):
            msg = "Screening cells must have unique cell identifiers."
            raise ValueError(msg)
        if self.baseline_configuration_checksum not in candidate_checksums:
            msg = "baseline_configuration_checksum must reference a screening candidate."
            raise ValueError(msg)
        baseline = next(
            candidate
            for candidate in candidates
            if candidate.configuration_checksum == self.baseline_configuration_checksum
        )
        if baseline.method_id != "layerwise_bmpd_crn_v2" or not baseline.noisy_training:
            msg = "The screening baseline must be a noisy layerwise_bmpd_crn_v2 configuration."
            raise ValueError(msg)
        represented = {cell.family_id for cell in cells}
        if represented != set(PRIMARY_TARGET_FAMILIES):
            msg = "Screening cells must represent every primary target family."
            raise ValueError(msg)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "cells", cells)

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete screening universe."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered manifest content."""
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "preregistration_checksum": self.preregistration_checksum,
            "screening_target_manifest_checksum": self.screening_target_manifest_checksum,
            "evaluation_policy_checksum": self.evaluation_policy_checksum,
            "resource_policy_checksum": self.resource_policy_checksum,
            "baseline_configuration_checksum": self.baseline_configuration_checksum,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "cells": [cell.to_dict() for cell in self.cells],
        }

    def to_dict(self) -> dict[str, object]:
        """Return the sealed JSON-native screening manifest."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    def candidate(self, configuration_checksum: str) -> ScreeningCandidateRef:
        """Resolve one screening candidate by configuration checksum.

        Args:
            configuration_checksum: Configuration digest to resolve.

        Returns:
            The matching candidate reference.

        Raises:
            KeyError: If the checksum is not in the sealed universe.
        """
        normalized = require_checksum(configuration_checksum, "configuration_checksum")
        for candidate in self.candidates:
            if candidate.configuration_checksum == normalized:
                return candidate
        raise KeyError(normalized)

    @classmethod
    def from_dict(cls, data: object) -> ScreeningManifest:
        """Construct and checksum-verify a screening manifest.

        Returns:
            The validated immutable screening manifest.

        Raises:
            ValueError: If the schema or checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_SCREENING_MANIFEST_KEYS, name="screening manifest")
        if mapping["schema_version"] != SCREENING_MANIFEST_SCHEMA_VERSION:
            msg = f"schema_version must be {SCREENING_MANIFEST_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        candidates = cast("Sequence[object]", mapping["candidates"])
        cells = cast("Sequence[object]", mapping["cells"])
        manifest = cls(
            manifest_id=cast("str", mapping["manifest_id"]),
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            screening_target_manifest_checksum=cast("str", mapping["screening_target_manifest_checksum"]),
            evaluation_policy_checksum=cast("str", mapping["evaluation_policy_checksum"]),
            resource_policy_checksum=cast("str", mapping["resource_policy_checksum"]),
            baseline_configuration_checksum=cast("str", mapping["baseline_configuration_checksum"]),
            candidates=tuple(ScreeningCandidateRef.from_dict(item) for item in candidates),
            cells=tuple(ScreeningCell.from_dict(item) for item in cells),
        )
        supplied = cast("str", mapping["content_checksum"])
        if manifest.content_checksum != supplied:
            msg = f"Screening manifest checksum changed during normalization: expected {supplied}."
            raise ValueError(msg)
        return manifest

    @classmethod
    def from_json(cls, payload: str) -> ScreeningManifest:
        """Construct a screening manifest from canonical JSON.

        Returns:
            The validated immutable screening manifest.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class PromotionObservation:
    """One source-addressed outcome on a sealed screening cell."""

    configuration_checksum: str
    cell_id: str
    result_schema_version: str
    result_record_checksum: str
    status: Literal["success", "failure"]
    noisy_fidelity: float | None
    resource_value: float | None
    normalized_work: float
    failure_code: str | None = None
    protocol_violations: tuple[str, ...] = ()
    data_role: str = field(default="screening_selection", init=False)

    def __post_init__(self) -> None:
        """Validate a source-addressed success or failure observation.

        Raises:
            ValueError: If status-dependent measurements or provenance are inconsistent.
        """
        object.__setattr__(
            self,
            "configuration_checksum",
            require_checksum(self.configuration_checksum, "configuration_checksum"),
        )
        object.__setattr__(self, "cell_id", require_slug(self.cell_id, "cell_id"))
        object.__setattr__(
            self,
            "result_schema_version",
            require_slug(self.result_schema_version, "result_schema_version"),
        )
        object.__setattr__(
            self,
            "result_record_checksum",
            require_checksum(self.result_record_checksum, "result_record_checksum"),
        )
        if self.status not in {"success", "failure"}:
            msg = "status must be 'success' or 'failure'."
            raise ValueError(msg)
        if self.status == "success":
            fidelity = require_float(self.noisy_fidelity, "noisy_fidelity", minimum=0.0, maximum=1.0)
            resource = require_float(self.resource_value, "resource_value", minimum=0.0)
            if self.failure_code is not None:
                msg = "Successful observations must not carry a failure_code."
                raise ValueError(msg)
            object.__setattr__(self, "noisy_fidelity", fidelity)
            object.__setattr__(self, "resource_value", resource)
        else:
            if self.noisy_fidelity is not None or self.resource_value is not None:
                msg = "Failure observations must have null fidelity and resource values."
                raise ValueError(msg)
            if self.failure_code is None:
                msg = "Failure observations require a failure_code."
                raise ValueError(msg)
            object.__setattr__(self, "failure_code", require_slug(self.failure_code, "failure_code"))
        object.__setattr__(
            self,
            "normalized_work",
            require_float(self.normalized_work, "normalized_work", minimum=0.0),
        )
        object.__setattr__(
            self,
            "protocol_violations",
            require_string_sequence(
                self.protocol_violations,
                "protocol_violations",
                unique=True,
                slugs=True,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native screening observation."""
        return {
            "configuration_checksum": self.configuration_checksum,
            "cell_id": self.cell_id,
            "result_schema_version": self.result_schema_version,
            "result_record_checksum": self.result_record_checksum,
            "status": self.status,
            "noisy_fidelity": self.noisy_fidelity,
            "resource_value": self.resource_value,
            "normalized_work": self.normalized_work,
            "failure_code": self.failure_code,
            "protocol_violations": list(self.protocol_violations),
        }

    @classmethod
    def from_dict(cls, data: object) -> PromotionObservation:
        """Construct an observation from an exact JSON object.

        Returns:
            The validated immutable observation.
        """
        mapping = require_mapping(data, "promotion observation")
        require_exact_keys(mapping, _PROMOTION_OBSERVATION_KEYS, "promotion observation")
        return cls(
            configuration_checksum=cast("str", mapping["configuration_checksum"]),
            cell_id=cast("str", mapping["cell_id"]),
            result_schema_version=cast("str", mapping["result_schema_version"]),
            result_record_checksum=cast("str", mapping["result_record_checksum"]),
            status=cast("Literal['success', 'failure']", mapping["status"]),
            noisy_fidelity=cast("float | None", mapping["noisy_fidelity"]),
            resource_value=cast("float | None", mapping["resource_value"]),
            normalized_work=cast("float", mapping["normalized_work"]),
            failure_code=cast("str | None", mapping["failure_code"]),
            protocol_violations=cast("tuple[str, ...]", mapping["protocol_violations"]),
        )


@dataclass(frozen=True, slots=True)
class ScreeningEvidence:
    """Checksum-sealed raw outcome ledger for one screening manifest."""

    evidence_id: str
    preregistration_checksum: str
    screening_manifest_checksum: str
    observations: tuple[PromotionObservation, ...]
    schema_version: str = field(default=SCREENING_EVIDENCE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate source records and reject duplicate candidate/cell pairs.

        Raises:
            TypeError: If observations contain unsupported record types.
            ValueError: If result or candidate/cell identities are duplicated.
        """
        object.__setattr__(self, "evidence_id", require_slug(self.evidence_id, "evidence_id"))
        object.__setattr__(
            self,
            "preregistration_checksum",
            require_checksum(self.preregistration_checksum, "preregistration_checksum"),
        )
        object.__setattr__(
            self,
            "screening_manifest_checksum",
            require_checksum(self.screening_manifest_checksum, "screening_manifest_checksum"),
        )
        observations = tuple(self.observations)
        if not observations or not all(isinstance(item, PromotionObservation) for item in observations):
            msg = "observations must contain PromotionObservation values."
            raise TypeError(msg)
        pairs = tuple((item.configuration_checksum, item.cell_id) for item in observations)
        if len(pairs) != len(set(pairs)):
            msg = "Screening evidence must not duplicate candidate/cell pairs."
            raise ValueError(msg)
        result_checksums = tuple(item.result_record_checksum for item in observations)
        if len(result_checksums) != len(set(result_checksums)):
            msg = "Every screening observation must reference a unique result record."
            raise ValueError(msg)
        object.__setattr__(self, "observations", observations)

    @property
    def content_checksum(self) -> str:
        """Checksum of every raw screening outcome."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered evidence content."""
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "preregistration_checksum": self.preregistration_checksum,
            "screening_manifest_checksum": self.screening_manifest_checksum,
            "observations": [observation.to_dict() for observation in self.observations],
        }

    def to_dict(self) -> dict[str, object]:
        """Return the sealed JSON-native evidence ledger."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ScreeningEvidence:
        """Construct and checksum-verify a screening evidence ledger.

        Returns:
            The validated immutable evidence ledger.

        Raises:
            ValueError: If the schema or checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_SCREENING_EVIDENCE_KEYS, name="screening evidence")
        if mapping["schema_version"] != SCREENING_EVIDENCE_SCHEMA_VERSION:
            msg = f"schema_version must be {SCREENING_EVIDENCE_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        observations = cast("Sequence[object]", mapping["observations"])
        evidence = cls(
            evidence_id=cast("str", mapping["evidence_id"]),
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            screening_manifest_checksum=cast("str", mapping["screening_manifest_checksum"]),
            observations=tuple(PromotionObservation.from_dict(item) for item in observations),
        )
        supplied = cast("str", mapping["content_checksum"])
        if evidence.content_checksum != supplied:
            msg = f"Screening evidence checksum changed during normalization: expected {supplied}."
            raise ValueError(msg)
        return evidence

    @classmethod
    def from_json(cls, payload: str) -> ScreeningEvidence:
        """Construct a screening evidence ledger from canonical JSON.

        Returns:
            The validated immutable evidence ledger.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """Family-weighted screening summary used by the mechanical rule."""

    configuration_checksum: str
    method_id: str
    weighted_itt_fidelity: float
    failure_rate: float
    max_resource_excess: float
    mean_normalized_work: float
    eligible: bool
    ineligibility_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate summary metrics and deterministic eligibility reasons.

        Raises:
            ValueError: If eligibility disagrees with its recorded reasons.
        """
        object.__setattr__(
            self,
            "configuration_checksum",
            require_checksum(self.configuration_checksum, "configuration_checksum"),
        )
        object.__setattr__(self, "method_id", require_slug(self.method_id, "method_id"))
        object.__setattr__(
            self,
            "weighted_itt_fidelity",
            require_float(self.weighted_itt_fidelity, "weighted_itt_fidelity", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "failure_rate",
            require_float(self.failure_rate, "failure_rate", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "max_resource_excess",
            require_float(self.max_resource_excess, "max_resource_excess", minimum=0.0),
        )
        object.__setattr__(
            self,
            "mean_normalized_work",
            require_float(self.mean_normalized_work, "mean_normalized_work", minimum=0.0),
        )
        object.__setattr__(self, "eligible", require_bool(self.eligible, "eligible"))
        reasons = require_string_sequence(
            self.ineligibility_reasons,
            "ineligibility_reasons",
            unique=True,
            slugs=True,
        )
        if self.eligible == bool(reasons):
            msg = "eligible must be true exactly when ineligibility_reasons is empty."
            raise ValueError(msg)
        object.__setattr__(self, "ineligibility_reasons", reasons)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native summary."""
        return {
            "configuration_checksum": self.configuration_checksum,
            "method_id": self.method_id,
            "weighted_itt_fidelity": self.weighted_itt_fidelity,
            "failure_rate": self.failure_rate,
            "max_resource_excess": self.max_resource_excess,
            "mean_normalized_work": self.mean_normalized_work,
            "eligible": self.eligible,
            "ineligibility_reasons": list(self.ineligibility_reasons),
        }

    @classmethod
    def from_dict(cls, data: object) -> CandidateSummary:
        """Construct a candidate summary from an exact JSON object.

        Returns:
            The validated immutable candidate summary.
        """
        mapping = require_mapping(data, "candidate summary")
        require_exact_keys(mapping, _CANDIDATE_SUMMARY_KEYS, "candidate summary")
        return cls(
            configuration_checksum=cast("str", mapping["configuration_checksum"]),
            method_id=cast("str", mapping["method_id"]),
            weighted_itt_fidelity=cast("float", mapping["weighted_itt_fidelity"]),
            failure_rate=cast("float", mapping["failure_rate"]),
            max_resource_excess=cast("float", mapping["max_resource_excess"]),
            mean_normalized_work=cast("float", mapping["mean_normalized_work"]),
            eligible=cast("bool", mapping["eligible"]),
            ineligibility_reasons=cast("tuple[str, ...]", mapping["ineligibility_reasons"]),
        )


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Checksum-locked result of the mechanical screening rule."""

    preregistration_checksum: str
    screening_manifest_checksum: str
    screening_evidence_checksum: str
    baseline_configuration_checksum: str
    promoted_method_id: str
    promoted_configuration_checksum: str
    null_fallback: bool
    rule_checksum: str
    candidate_summaries: tuple[CandidateSummary, ...]
    schema_version: str = field(default=PROMOTION_DECISION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate checksums, promotion identity, and summary uniqueness.

        Raises:
            TypeError: If candidate summaries contain an unsupported record type.
            ValueError: If summaries are duplicated or disagree with the promoted identity.
        """
        for name in (
            "preregistration_checksum",
            "screening_manifest_checksum",
            "screening_evidence_checksum",
            "baseline_configuration_checksum",
            "promoted_configuration_checksum",
            "rule_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(self, "promoted_method_id", require_slug(self.promoted_method_id, "promoted_method_id"))
        object.__setattr__(self, "null_fallback", require_bool(self.null_fallback, "null_fallback"))
        summaries = tuple(self.candidate_summaries)
        if not summaries or not all(isinstance(summary, CandidateSummary) for summary in summaries):
            msg = "candidate_summaries must contain CandidateSummary values."
            raise TypeError(msg)
        checksums = tuple(summary.configuration_checksum for summary in summaries)
        if len(checksums) != len(set(checksums)):
            msg = "candidate_summaries must have unique configuration checksums."
            raise ValueError(msg)
        if self.baseline_configuration_checksum not in checksums:
            msg = "baseline_configuration_checksum must reference a candidate summary."
            raise ValueError(msg)
        baseline = next(
            summary for summary in summaries if summary.configuration_checksum == self.baseline_configuration_checksum
        )
        if baseline.method_id != "layerwise_bmpd_crn_v2":
            msg = "The baseline candidate summary must use layerwise_bmpd_crn_v2."
            raise ValueError(msg)
        if self.promoted_configuration_checksum not in checksums:
            msg = "promoted_configuration_checksum must reference a candidate summary."
            raise ValueError(msg)
        promoted = next(
            summary for summary in summaries if summary.configuration_checksum == self.promoted_configuration_checksum
        )
        if promoted.method_id != self.promoted_method_id:
            msg = "Promoted method and configuration summary disagree."
            raise ValueError(msg)
        if self.null_fallback != (self.promoted_configuration_checksum == self.baseline_configuration_checksum):
            msg = "null_fallback must identify exactly a baseline promotion."
            raise ValueError(msg)
        eligible_challengers = [
            summary
            for summary in summaries
            if summary.configuration_checksum != self.baseline_configuration_checksum and summary.eligible
        ]
        expected = (
            min(
                eligible_challengers,
                key=lambda summary: (
                    -summary.weighted_itt_fidelity,
                    summary.failure_rate,
                    summary.max_resource_excess,
                    summary.mean_normalized_work,
                    summary.configuration_checksum,
                ),
            )
            if eligible_challengers
            else baseline
        )
        if promoted.configuration_checksum != expected.configuration_checksum:
            msg = "Promoted configuration is not the mechanically ranked eligible candidate or null baseline."
            raise ValueError(msg)
        object.__setattr__(self, "candidate_summaries", summaries)

    @property
    def content_checksum(self) -> str:
        """Promotion-decision checksum."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return the checksum-covered decision payload."""
        return {
            "schema_version": self.schema_version,
            "preregistration_checksum": self.preregistration_checksum,
            "screening_manifest_checksum": self.screening_manifest_checksum,
            "screening_evidence_checksum": self.screening_evidence_checksum,
            "baseline_configuration_checksum": self.baseline_configuration_checksum,
            "promoted_method_id": self.promoted_method_id,
            "promoted_configuration_checksum": self.promoted_configuration_checksum,
            "null_fallback": self.null_fallback,
            "rule_checksum": self.rule_checksum,
            "candidate_summaries": [summary.to_dict() for summary in self.candidate_summaries],
        }

    def to_dict(self) -> dict[str, object]:
        """Return the sealed JSON-native decision."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> PromotionDecision:
        """Construct and checksum-verify a promotion decision.

        Returns:
            The validated immutable promotion decision.

        Raises:
            ValueError: If the schema version or normalized checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_PROMOTION_DECISION_KEYS, name="promotion decision")
        if mapping["schema_version"] != PROMOTION_DECISION_SCHEMA_VERSION:
            msg = f"schema_version must be {PROMOTION_DECISION_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        summaries = cast("Sequence[object]", mapping["candidate_summaries"])
        decision = cls(
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            screening_manifest_checksum=cast("str", mapping["screening_manifest_checksum"]),
            screening_evidence_checksum=cast("str", mapping["screening_evidence_checksum"]),
            baseline_configuration_checksum=cast("str", mapping["baseline_configuration_checksum"]),
            promoted_method_id=cast("str", mapping["promoted_method_id"]),
            promoted_configuration_checksum=cast("str", mapping["promoted_configuration_checksum"]),
            null_fallback=cast("bool", mapping["null_fallback"]),
            rule_checksum=cast("str", mapping["rule_checksum"]),
            candidate_summaries=tuple(CandidateSummary.from_dict(summary) for summary in summaries),
        )
        supplied = cast("str", mapping["content_checksum"])
        if decision.content_checksum != supplied:
            msg = f"Promotion decision checksum changed during normalization: expected {supplied}."
            raise ValueError(msg)
        return decision

    @classmethod
    def from_json(cls, payload: str) -> PromotionDecision:
        """Construct a decision from canonical sealed JSON.

        Returns:
            The validated immutable promotion decision.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _validate_screening_manifest(
    preregistration: InitialPreregistration,
    manifest: ScreeningManifest,
) -> None:
    """Cross-validate a screening universe against the governing protocol.

    Raises:
        ValueError: If method identities, matching, qubits, or strata disagree
            with the preregistration.
    """
    if manifest.preregistration_checksum != preregistration.content_checksum:
        msg = "Screening manifest does not reference the supplied preregistration."
        raise ValueError(msg)
    expected_evaluation_policy_checksum = canonical_checksum({
        "endpoint": preregistration.primary_endpoint,
        "failure_policy": preregistration.failure_policy,
        "noise": preregistration.primary_noise_condition,
    })
    if manifest.evaluation_policy_checksum != expected_evaluation_policy_checksum:
        msg = "Screening manifest uses an evaluation policy not frozen by the preregistration."
        raise ValueError(msg)
    if manifest.resource_policy_checksum != canonical_checksum(preregistration.primary_resource_constraint):
        msg = "Screening manifest uses a resource policy not frozen by the preregistration."
        raise ValueError(msg)
    method_ids: list[str] = []
    for candidate in manifest.candidates:
        try:
            policy = preregistration.method_policy(candidate.method_id)
        except KeyError as error:
            msg = f"Screening manifest contains unregistered method {candidate.method_id!r}."
            raise ValueError(msg) from error
        if policy["scope"] != "all_families":
            msg = f"Screening candidate {candidate.method_id!r} is not registered for every primary family."
            raise ValueError(msg)
        if candidate.noisy_training is not policy["noisy_training"]:
            msg = f"Screening candidate {candidate.method_id!r} contradicts its noisy-training policy."
            raise ValueError(msg)
        method_ids.append(candidate.method_id)
    for required_method in ("layerwise_bmpd_crn_v2", "layerwise_bmpd_noiseless"):
        if method_ids.count(required_method) != 1:
            msg = f"Screening manifest must contain exactly one {required_method!r} configuration."
            raise ValueError(msg)
    required_screening_methods = {
        cast("str", policy["method_id"])
        for policy in preregistration.candidate_methods
        if policy["scope"] == "all_families"
    }
    missing_methods = sorted(required_screening_methods - set(method_ids))
    if missing_methods:
        msg = f"Screening manifest omits preregistered family-wide methods {missing_methods!r}."
        raise ValueError(msg)
    baseline = manifest.candidate(manifest.baseline_configuration_checksum)
    noiseless = next(
        candidate for candidate in manifest.candidates if candidate.method_id == "layerwise_bmpd_noiseless"
    )
    if (
        baseline.matching_projection_checksum is None
        or noiseless.matching_projection_checksum != baseline.matching_projection_checksum
    ):
        msg = "The noisy v2 baseline and noiseless comparator must share one matching projection checksum."
        raise ValueError(msg)

    primary_qubits = frozenset(cast("Sequence[int]", preregistration.target_population_policy["primary_qubit_counts"]))
    if any(cell.qubit_count not in primary_qubits for cell in manifest.cells):
        msg = "Screening promotion cells must use only preregistered primary qubit counts."
        raise ValueError(msg)
    family_stratum_targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    target_identities: dict[str, tuple[str, str, int]] = {}
    target_optimization_seeds: dict[str, set[int]] = defaultdict(set)
    target_seed_pairs: set[tuple[str, int]] = set()
    screening_seeds: set[int] = set()
    for cell in manifest.cells:
        family_stratum_targets[cell.family_id, cell.stratum_id].add(cell.target_instance_id)
        identity = (cell.family_id, cell.stratum_id, cell.qubit_count)
        previous = target_identities.setdefault(cell.target_instance_id, identity)
        if previous != identity:
            msg = f"Target instance {cell.target_instance_id!r} has inconsistent screening identity."
            raise ValueError(msg)
        target_seed_pair = (cell.target_instance_id, cell.optimization_seed)
        if target_seed_pair in target_seed_pairs:
            msg = f"Screening target/optimization-seed pair {target_seed_pair!r} is duplicated."
            raise ValueError(msg)
        target_seed_pairs.add(target_seed_pair)
        target_optimization_seeds[cell.target_instance_id].add(cell.optimization_seed)
        if cell.screening_seed in screening_seeds:
            msg = f"Screening seed {cell.screening_seed} is reused across outer cells."
            raise ValueError(msg)
        screening_seeds.add(cell.screening_seed)
    allocation_policy = cast(
        "Mapping[str, object]",
        preregistration.target_population_policy["role_allocation_policy"],
    )
    expected_targets_per_family = cast("int", allocation_policy["screening_targets_per_family"])
    expected_optimization_seeds = cast("int", allocation_policy["screening_optimizer_seed_count"])
    targets_by_family: dict[str, set[str]] = defaultdict(set)
    for target_instance_id, (family_id, _stratum_id, _qubit_count) in target_identities.items():
        targets_by_family[family_id].add(target_instance_id)
        if len(target_optimization_seeds[target_instance_id]) != expected_optimization_seeds:
            msg = (
                f"Screening target {target_instance_id!r} must have exactly "
                f"{expected_optimization_seeds} optimization seeds."
            )
            raise ValueError(msg)
    for family_id, strata in PRIMARY_FAMILY_STRATA.items():
        counts = tuple(len(family_stratum_targets[family_id, stratum_id]) for stratum_id in strata)
        if (
            len(targets_by_family[family_id]) != expected_targets_per_family
            or not counts
            or any(count == 0 for count in counts)
            or len(set(counts)) != 1
        ):
            msg = f"Screening cells must allocate family {family_id!r} equally across its strata."
            raise ValueError(msg)


def _validate_screening_evidence(
    preregistration: InitialPreregistration,
    manifest: ScreeningManifest,
    evidence: ScreeningEvidence,
) -> None:
    """Require exactly one raw result for every sealed candidate/cell pair.

    Raises:
        ValueError: If the ledger is unlinked, incomplete, or contains extras.
    """
    if evidence.preregistration_checksum != preregistration.content_checksum:
        msg = "Screening evidence does not reference the supplied preregistration."
        raise ValueError(msg)
    if evidence.screening_manifest_checksum != manifest.content_checksum:
        msg = "Screening evidence does not reference the supplied screening manifest."
        raise ValueError(msg)
    expected_pairs = {
        (candidate.configuration_checksum, cell.cell_id) for candidate in manifest.candidates for cell in manifest.cells
    }
    actual_pairs = {(observation.configuration_checksum, observation.cell_id) for observation in evidence.observations}
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        msg = f"Screening evidence does not match the sealed universe: missing={missing!r}, extra={extra!r}."
        raise ValueError(msg)


def _candidate_summary(
    preregistration: InitialPreregistration,
    candidate: ScreeningCandidateRef,
    observations: Sequence[PromotionObservation],
    cells: Mapping[str, ScreeningCell],
    *,
    baseline_score: float | None,
) -> CandidateSummary:
    """Compute one family-weighted intention-to-treat candidate summary.

    Returns:
        The validated screening summary for the candidate configuration.

    Raises:
        ValueError: If an observation disagrees with its screening cell or omits a required family.
    """
    policy = preregistration.method_policy(candidate.method_id)
    failed_fidelity = cast("float", preregistration.failure_policy["failed_fidelity"])
    family_values: dict[str, list[float]] = defaultdict(list)
    family_failure_counts: dict[str, int] = defaultdict(int)
    resource_values: list[float] = []
    work_values: list[float] = []
    violations: set[str] = set()
    for observation in observations:
        cell = cells[observation.cell_id]
        if observation.status == "failure":
            family_values[cell.family_id].append(failed_fidelity)
            family_failure_counts[cell.family_id] += 1
        else:
            assert observation.noisy_fidelity is not None
            assert observation.resource_value is not None
            family_values[cell.family_id].append(observation.noisy_fidelity)
            resource_values.append(observation.resource_value)
        work_values.append(observation.normalized_work)
        violations.update(observation.protocol_violations)

    family_scores: dict[str, float] = {}
    for family_id in PRIMARY_TARGET_FAMILIES:
        values = family_values.get(family_id)
        if not values:
            msg = f"Screening manifest contains no observations for required family {family_id!r}."
            raise ValueError(msg)
        family_scores[family_id] = math.fsum(values) / len(values)
    weighted_score = math.fsum(
        cast("float", preregistration.target_family_weights[family_id]) * family_scores[family_id]
        for family_id in PRIMARY_TARGET_FAMILIES
    )
    failure_rate = math.fsum(
        cast("float", preregistration.target_family_weights[family_id])
        * family_failure_counts[family_id]
        / len(family_values[family_id])
        for family_id in PRIMARY_TARGET_FAMILIES
    )
    cap = cast("float", preregistration.primary_resource_constraint["cap_per_chain_edge"])
    max_excess = max((max(0.0, value - cap) for value in resource_values), default=0.0)
    mean_work = math.fsum(work_values) / len(work_values)

    reasons = set(violations)
    if not cast("bool", policy["promotion_eligible"]):
        reasons.add("method_not_promotion_eligible")
    if not candidate.noisy_training:
        reasons.add("no_noisy_training")
    maximum_failure = cast("float", preregistration.promotion_rule["maximum_failure_rate"])
    if failure_rate > maximum_failure:
        reasons.add("failure_rate_exceeded")
    if max_excess > 0.0:
        reasons.add("resource_cap_exceeded")
    if baseline_score is not None:
        minimum_gain = cast("float", preregistration.promotion_rule["minimum_weighted_itt_gain"])
        if weighted_score < baseline_score + minimum_gain:
            reasons.add("minimum_gain_not_met")

    return CandidateSummary(
        configuration_checksum=candidate.configuration_checksum,
        method_id=candidate.method_id,
        weighted_itt_fidelity=weighted_score,
        failure_rate=failure_rate,
        max_resource_excess=max_excess,
        mean_normalized_work=mean_work,
        eligible=not reasons,
        ineligibility_reasons=tuple(sorted(reasons)),
    )


def select_promoted_candidate(
    preregistration: InitialPreregistration,
    manifest: ScreeningManifest,
    evidence: ScreeningEvidence,
) -> PromotionDecision:
    """Apply the sealed family-weighted one-candidate promotion rule.

    Failures contribute the preregistered zero fidelity, families receive their
    preregistered weights, final-test rows cannot be represented by the input
    type, and a null screen deterministically promotes the corrected v2
    baseline.

    Args:
        preregistration: Initial protocol governing selection.
        manifest: Sealed complete candidate and screening-cell universe.
        evidence: Sealed raw result ledger containing the complete Cartesian product.

    Returns:
        A checksum-locked decision promoting exactly one configuration.

    Raises:
        TypeError: If values have unsupported types.
        ValueError: If candidates or cells are missing, duplicated, inconsistent, or unknown.
    """
    if not isinstance(preregistration, InitialPreregistration):
        msg = f"preregistration must be an InitialPreregistration, got {type(preregistration).__name__}."
        raise TypeError(msg)
    if not isinstance(manifest, ScreeningManifest):
        msg = f"manifest must be a ScreeningManifest, got {type(manifest).__name__}."
        raise TypeError(msg)
    if not isinstance(evidence, ScreeningEvidence):
        msg = f"evidence must be a ScreeningEvidence, got {type(evidence).__name__}."
        raise TypeError(msg)
    _validate_screening_manifest(preregistration, manifest)
    _validate_screening_evidence(preregistration, manifest, evidence)

    baseline_checksum = manifest.baseline_configuration_checksum
    cell_map = {cell.cell_id: cell for cell in manifest.cells}
    grouped: dict[str, list[PromotionObservation]] = defaultdict(list)
    for observation in evidence.observations:
        grouped[observation.configuration_checksum].append(observation)
    candidate_map = {candidate.configuration_checksum: candidate for candidate in manifest.candidates}
    baseline_candidate = candidate_map[baseline_checksum]
    baseline_summary = _candidate_summary(
        preregistration,
        baseline_candidate,
        grouped[baseline_checksum],
        cell_map,
        baseline_score=None,
    )
    if not baseline_summary.eligible:
        msg = (
            "The corrected v2 baseline failed fatal screening-integrity requirements: "
            f"{baseline_summary.ineligibility_reasons!r}."
        )
        raise ValueError(msg)
    summaries = [
        _candidate_summary(
            preregistration,
            candidate,
            grouped[candidate.configuration_checksum],
            cell_map,
            baseline_score=(
                None
                if candidate.configuration_checksum == baseline_checksum
                else baseline_summary.weighted_itt_fidelity
            ),
        )
        for candidate in manifest.candidates
    ]
    ranked = sorted(
        summaries,
        key=lambda summary: (
            not summary.eligible,
            -summary.weighted_itt_fidelity,
            summary.failure_rate,
            summary.max_resource_excess,
            summary.mean_normalized_work,
            summary.configuration_checksum,
        ),
    )
    promoted = next(
        (summary for summary in ranked if summary.configuration_checksum != baseline_checksum and summary.eligible),
        baseline_summary,
    )
    ordered_summaries = tuple(
        sorted(
            summaries,
            key=lambda summary: (
                summary.configuration_checksum != promoted.configuration_checksum,
                summary.configuration_checksum,
            ),
        )
    )
    return PromotionDecision(
        preregistration_checksum=preregistration.content_checksum,
        screening_manifest_checksum=manifest.content_checksum,
        screening_evidence_checksum=evidence.content_checksum,
        baseline_configuration_checksum=baseline_checksum,
        promoted_method_id=promoted.method_id,
        promoted_configuration_checksum=promoted.configuration_checksum,
        null_fallback=promoted.configuration_checksum == baseline_checksum,
        rule_checksum=preregistration.promotion_rule_checksum,
        candidate_summaries=ordered_summaries,
    )


@dataclass(frozen=True, slots=True)
class FinalComparatorRef:
    """One typed comparator configuration in the final confirmatory design."""

    role: Literal["layerwise_v2_reference", "matched_noiseless_control", "additional"]
    method_id: str
    configuration_schema_version: str
    configuration_checksum: str
    matched_to_configuration_checksum: str | None
    matching_projection_checksum: str | None

    def __post_init__(self) -> None:
        """Validate comparator role, identity, and matching metadata.

        Raises:
            ValueError: If the role, method, or matching metadata is inconsistent.
        """
        if self.role not in {"layerwise_v2_reference", "matched_noiseless_control", "additional"}:
            msg = f"Unsupported comparator role {self.role!r}."
            raise ValueError(msg)
        object.__setattr__(self, "method_id", require_slug(self.method_id, "method_id"))
        object.__setattr__(
            self,
            "configuration_schema_version",
            require_slug(self.configuration_schema_version, "configuration_schema_version"),
        )
        object.__setattr__(
            self,
            "configuration_checksum",
            require_checksum(self.configuration_checksum, "configuration_checksum"),
        )
        if self.matched_to_configuration_checksum is not None:
            object.__setattr__(
                self,
                "matched_to_configuration_checksum",
                require_checksum(self.matched_to_configuration_checksum, "matched_to_configuration_checksum"),
            )
        if self.matching_projection_checksum is not None:
            object.__setattr__(
                self,
                "matching_projection_checksum",
                require_checksum(self.matching_projection_checksum, "matching_projection_checksum"),
            )
        if self.role == "layerwise_v2_reference" and self.method_id != "layerwise_bmpd_crn_v2":
            msg = "The layerwise_v2_reference role requires layerwise_bmpd_crn_v2."
            raise ValueError(msg)
        if self.role == "matched_noiseless_control" and self.method_id != "layerwise_bmpd_noiseless":
            msg = "The matched_noiseless_control role requires layerwise_bmpd_noiseless."
            raise ValueError(msg)
        if self.role in {"layerwise_v2_reference", "matched_noiseless_control"}:
            if self.matched_to_configuration_checksum is None or self.matching_projection_checksum is None:
                msg = f"Comparator role {self.role!r} requires exact matching metadata."
                raise ValueError(msg)
        elif self.matched_to_configuration_checksum is not None or self.matching_projection_checksum is not None:
            msg = "Additional comparators must not claim a matched primary contrast."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native comparator reference."""
        return {
            "role": self.role,
            "method_id": self.method_id,
            "configuration_schema_version": self.configuration_schema_version,
            "configuration_checksum": self.configuration_checksum,
            "matched_to_configuration_checksum": self.matched_to_configuration_checksum,
            "matching_projection_checksum": self.matching_projection_checksum,
        }

    @classmethod
    def from_dict(cls, data: object) -> FinalComparatorRef:
        """Construct a comparator reference from an exact JSON object.

        Returns:
            The validated immutable comparator reference.
        """
        mapping = require_mapping(data, "final comparator")
        require_exact_keys(mapping, _COMPARATOR_KEYS, "final comparator")
        return cls(
            role=cast(
                "Literal['layerwise_v2_reference', 'matched_noiseless_control', 'additional']",
                mapping["role"],
            ),
            method_id=cast("str", mapping["method_id"]),
            configuration_schema_version=cast("str", mapping["configuration_schema_version"]),
            configuration_checksum=cast("str", mapping["configuration_checksum"]),
            matched_to_configuration_checksum=cast("str | None", mapping["matched_to_configuration_checksum"]),
            matching_projection_checksum=cast("str | None", mapping["matching_projection_checksum"]),
        )


@dataclass(frozen=True, slots=True)
class PrimaryContrastBinding:
    """Exact treatment, control, pairing, and matching identities for one contrast."""

    contrast_id: str
    treatment_configuration_checksum: str
    control_configuration_checksum: str
    paired_block_policy_checksum: str
    matching_projection_checksum: str | None

    def __post_init__(self) -> None:
        """Validate contrast and configuration identities.

        Raises:
            ValueError: If a checksum is invalid or the contrast is a self-comparison.
        """
        object.__setattr__(self, "contrast_id", require_slug(self.contrast_id, "contrast_id"))
        for name in (
            "treatment_configuration_checksum",
            "control_configuration_checksum",
            "paired_block_policy_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        if self.treatment_configuration_checksum == self.control_configuration_checksum:
            msg = "A primary contrast must use distinct treatment and control configurations."
            raise ValueError(msg)
        if self.matching_projection_checksum is not None:
            object.__setattr__(
                self,
                "matching_projection_checksum",
                require_checksum(self.matching_projection_checksum, "matching_projection_checksum"),
            )

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native primary-contrast binding."""
        return {
            "contrast_id": self.contrast_id,
            "treatment_configuration_checksum": self.treatment_configuration_checksum,
            "control_configuration_checksum": self.control_configuration_checksum,
            "paired_block_policy_checksum": self.paired_block_policy_checksum,
            "matching_projection_checksum": self.matching_projection_checksum,
        }

    @classmethod
    def from_dict(cls, data: object) -> PrimaryContrastBinding:
        """Construct a contrast binding from an exact JSON object.

        Returns:
            The validated immutable contrast binding.
        """
        mapping = require_mapping(data, "primary contrast binding")
        require_exact_keys(mapping, _PRIMARY_CONTRAST_BINDING_KEYS, "primary contrast binding")
        return cls(
            contrast_id=cast("str", mapping["contrast_id"]),
            treatment_configuration_checksum=cast("str", mapping["treatment_configuration_checksum"]),
            control_configuration_checksum=cast("str", mapping["control_configuration_checksum"]),
            paired_block_policy_checksum=cast("str", mapping["paired_block_policy_checksum"]),
            matching_projection_checksum=cast("str | None", mapping["matching_projection_checksum"]),
        )


@dataclass(frozen=True, slots=True)
class FinalConfigurationExecutionRef:
    """Exact screened execution identity for one final configuration."""

    method_id: str
    configuration_schema_version: str
    configuration_checksum: str
    strategy_schedule: TrainingStrategySchedule
    implementation_checksum: str
    scoped_binding_checksum: str
    executable_binding_checksum: str
    schema_version: str = field(default=FINAL_CONFIGURATION_EXECUTION_REF_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the complete configuration-to-executable identity chain.

        Raises:
            TypeError: If the embedded schedule has the wrong typed schema.
        """
        object.__setattr__(self, "method_id", require_slug(self.method_id, "method_id"))
        object.__setattr__(
            self,
            "configuration_schema_version",
            require_slug(self.configuration_schema_version, "configuration_schema_version"),
        )
        for name in (
            "configuration_checksum",
            "implementation_checksum",
            "scoped_binding_checksum",
            "executable_binding_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        if not isinstance(self.strategy_schedule, TrainingStrategySchedule):
            msg = "strategy_schedule must be a TrainingStrategySchedule."
            raise TypeError(msg)

    @property
    def strategy_schedule_checksum(self) -> str:
        """Checksum of the embedded exact configuration-specific schedule."""
        return self.strategy_schedule.content_checksum

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete final execution reference."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return every configuration-specific executable field."""
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "configuration_schema_version": self.configuration_schema_version,
            "configuration_checksum": self.configuration_checksum,
            "strategy_schedule": self.strategy_schedule.to_dict(),
            "strategy_schedule_checksum": self.strategy_schedule_checksum,
            "implementation_checksum": self.implementation_checksum,
            "scoped_binding_checksum": self.scoped_binding_checksum,
            "executable_binding_checksum": self.executable_binding_checksum,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed JSON-native execution reference."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> FinalConfigurationExecutionRef:
        """Decode and verify one final configuration execution reference.

        Returns:
            The validated exact execution reference.

        Raises:
            ValueError: If the schema or a derived checksum differs.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_FINAL_CONFIGURATION_EXECUTION_REF_KEYS,
            name="final configuration execution reference",
        )
        if mapping["schema_version"] != FINAL_CONFIGURATION_EXECUTION_REF_SCHEMA_VERSION:
            msg = "Final configuration execution reference uses an unsupported schema version."
            raise ValueError(msg)
        reference = cls(
            method_id=cast("str", mapping["method_id"]),
            configuration_schema_version=cast("str", mapping["configuration_schema_version"]),
            configuration_checksum=cast("str", mapping["configuration_checksum"]),
            strategy_schedule=TrainingStrategySchedule.from_dict(mapping["strategy_schedule"]),
            implementation_checksum=cast("str", mapping["implementation_checksum"]),
            scoped_binding_checksum=cast("str", mapping["scoped_binding_checksum"]),
            executable_binding_checksum=cast("str", mapping["executable_binding_checksum"]),
        )
        if mapping["strategy_schedule_checksum"] != reference.strategy_schedule_checksum:
            msg = "Final configuration schedule checksum is not derived from its embedded schedule."
            raise ValueError(msg)
        if mapping["content_checksum"] != reference.content_checksum:
            msg = "Final configuration execution reference checksum changed during normalization."
            raise ValueError(msg)
        return reference


@dataclass(frozen=True, slots=True)
class FinalConfigurationExecutionManifest:
    """Aggregate root of every configuration-specific final execution identity."""

    manifest_id: str
    entries: tuple[FinalConfigurationExecutionRef, ...]
    schema_version: str = field(default=FINAL_CONFIGURATION_EXECUTION_MANIFEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require a canonical, unique, nonempty final configuration universe.

        Raises:
            TypeError: If entries do not use the exact reference schema.
            ValueError: If ordering or any required identity is duplicated.
        """
        object.__setattr__(self, "manifest_id", require_slug(self.manifest_id, "manifest_id"))
        entries = tuple(self.entries)
        if not entries or not all(isinstance(item, FinalConfigurationExecutionRef) for item in entries):
            msg = "entries must contain FinalConfigurationExecutionRef values."
            raise TypeError(msg)
        expected_order = tuple(sorted(entries, key=lambda item: (item.configuration_checksum, item.method_id)))
        if entries != expected_order:
            msg = "Final configuration execution entries must use canonical configuration order."
            raise ValueError(msg)
        method_ids = tuple(item.method_id for item in entries)
        configuration_checksums = tuple(item.configuration_checksum for item in entries)
        scoped_checksums = tuple(item.scoped_binding_checksum for item in entries)
        executable_checksums = tuple(item.executable_binding_checksum for item in entries)
        if (
            len(method_ids) != len(set(method_ids))
            or len(configuration_checksums) != len(set(configuration_checksums))
            or len(scoped_checksums) != len(set(scoped_checksums))
            or len(executable_checksums) != len(set(executable_checksums))
        ):
            msg = "Final configuration execution entries must have unique method, configuration, and bindings."
            raise ValueError(msg)
        object.__setattr__(self, "entries", entries)

    @property
    def content_checksum(self) -> str:
        """Aggregate root stored in ``FinalConfirmationSeal.hyperparameters_checksum``."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return the exact ordered executable configuration universe."""
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "entries": [item.to_dict() for item in self.entries],
            "entry_count": len(self.entries),
        }

    def to_dict(self) -> dict[str, object]:
        """Return the checksum-sealed JSON-native execution manifest."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical checksum-sealed execution-manifest JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> FinalConfigurationExecutionManifest:
        """Decode and verify the final configuration execution manifest.

        Returns:
            The validated exact final execution universe.

        Raises:
            TypeError: If serialized entries do not form a sequence.
            ValueError: If the schema, count, or a derived checksum differs.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_FINAL_CONFIGURATION_EXECUTION_MANIFEST_KEYS,
            name="final configuration execution manifest",
        )
        if mapping["schema_version"] != FINAL_CONFIGURATION_EXECUTION_MANIFEST_SCHEMA_VERSION:
            msg = "Final configuration execution manifest uses an unsupported schema version."
            raise ValueError(msg)
        raw_entries = mapping["entries"]
        if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, Sequence):
            msg = "Final configuration execution entries must be a sequence."
            raise TypeError(msg)
        manifest = cls(
            manifest_id=cast("str", mapping["manifest_id"]),
            entries=tuple(FinalConfigurationExecutionRef.from_dict(item) for item in raw_entries),
        )
        if mapping["entry_count"] != len(manifest.entries):
            msg = "Final configuration execution entry_count is not derived from its entries."
            raise ValueError(msg)
        if mapping["content_checksum"] != manifest.content_checksum:
            msg = "Final configuration execution manifest checksum changed during normalization."
            raise ValueError(msg)
        return manifest

    @classmethod
    def from_json(cls, payload: str) -> FinalConfigurationExecutionManifest:
        """Decode canonical final configuration execution-manifest JSON.

        Returns:
            The validated exact final execution universe.
        """
        return cls.from_dict(load_canonical_json_object(payload))

    def entry(self, configuration_checksum: str) -> FinalConfigurationExecutionRef:
        """Return the exact execution reference for one final configuration.

        Returns:
            The unique configuration-specific execution reference.

        Raises:
            KeyError: If the configuration is absent from the final universe.
        """
        checksum = require_checksum(configuration_checksum, "configuration_checksum")
        for item in self.entries:
            if item.configuration_checksum == checksum:
                return item
        raise KeyError(checksum)


class FinalResourceCalibrationManifest(ABC):
    """Protocol-owned base for a typed production resource calibration.

    Concrete custody-aware implementations live downstream of this foundational
    protocol module.  Inheriting from this base does not confer authority:
    confirmation authorization accepts only the exact repository-owned
    :class:`~benchmarks.state_preparation.phase2.screening.ProductionResourceCalibration`
    type, whose constructor revalidates the complete pilot and screening
    projection universes.
    """

    preregistration_checksum: str
    execution_source_manifest_checksum: str
    screening_manifest_checksum: str
    normalized_compute_cap: float

    @property
    @abstractmethod
    def content_checksum(self) -> str:
        """Checksum of the complete typed resource calibration."""
        raise NotImplementedError


def _validate_comparators(value: object) -> tuple[FinalComparatorRef, ...]:
    """Validate a de-duplicated typed primary comparator set.

    Returns:
        The validated comparator tuple.

    Raises:
        TypeError: If the value is not a supported comparator sequence.
        ValueError: If method, configuration, or role identities are duplicated.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = "comparators must be a sequence of FinalComparatorRef values."
        raise TypeError(msg)
    raw_comparators = tuple(value)
    if not 1 <= len(raw_comparators) <= 3 or not all(isinstance(item, FinalComparatorRef) for item in raw_comparators):
        msg = "comparators must contain between one and three FinalComparatorRef values."
        raise TypeError(msg)
    comparators = cast("tuple[FinalComparatorRef, ...]", raw_comparators)
    method_ids = tuple(comparator.method_id for comparator in comparators)
    checksums = tuple(comparator.configuration_checksum for comparator in comparators)
    roles = tuple(comparator.role for comparator in comparators)
    if (
        len(method_ids) != len(set(method_ids))
        or len(checksums) != len(set(checksums))
        or len(roles) != len(set(roles))
    ):
        msg = "comparators must not duplicate method, configuration, or role identities."
        raise ValueError(msg)
    return comparators


def _validate_primary_contrasts(value: object) -> tuple[PrimaryContrastBinding, ...]:
    """Validate unique primary contrast bindings.

    Returns:
        The validated contrast-binding tuple.

    Raises:
        TypeError: If the value is not a supported binding sequence.
        ValueError: If contrast identities are duplicated.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = "primary_contrasts must be a sequence of PrimaryContrastBinding values."
        raise TypeError(msg)
    raw_bindings = tuple(value)
    if not raw_bindings or not all(isinstance(item, PrimaryContrastBinding) for item in raw_bindings):
        msg = "primary_contrasts must contain PrimaryContrastBinding values."
        raise TypeError(msg)
    bindings = cast("tuple[PrimaryContrastBinding, ...]", raw_bindings)
    contrast_ids = tuple(binding.contrast_id for binding in bindings)
    if len(contrast_ids) != len(set(contrast_ids)):
        msg = "primary_contrasts must have unique contrast identifiers."
        raise ValueError(msg)
    return bindings


def _validate_target_counts(value: object) -> Mapping[str, object]:
    """Validate positive confirmatory target counts for every primary family.

    Returns:
        The frozen target-count mapping.

    Raises:
        ValueError: If the mapping does not cover every primary target family.
    """
    counts = freeze_json_mapping(value, "target_count_by_family")
    if frozenset(counts) != frozenset(PRIMARY_TARGET_FAMILIES):
        msg = "target_count_by_family keys must exactly match primary target families."
        raise ValueError(msg)
    for family_id in PRIMARY_TARGET_FAMILIES:
        require_int(counts[family_id], f"target_count_by_family.{family_id}", minimum=1)
    return counts


def _validate_resource_budget(value: object) -> Mapping[str, object]:
    """Validate the pilot-sealed reachable primary resource budget.

    Returns:
        The frozen primary resource budget.

    Raises:
        ValueError: If the primary metric differs from the preregistered metric.
    """
    budget = freeze_json_mapping(value, "primary_resource_budget")
    require_exact_keys(budget, _RESOURCE_BUDGET_KEYS, "primary_resource_budget")
    if budget["metric"] != "native_two_qubit_gates_per_chain_edge":
        msg = "primary_resource_budget.metric must match the preregistered primary metric."
        raise ValueError(msg)
    require_float(budget["cap_per_chain_edge"], "primary_resource_budget.cap_per_chain_edge", minimum=0.0)
    require_float(budget["normalized_compute_cap"], "primary_resource_budget.normalized_compute_cap", minimum=0.0)
    require_checksum(
        budget["reachable_stratum_manifest_checksum"],
        "primary_resource_budget.reachable_stratum_manifest_checksum",
    )
    return budget


@dataclass(frozen=True, slots=True)
class SampleAllocation:
    """One target allocation for a primary family, stratum, and qubit count."""

    family_id: str
    stratum_id: str
    qubit_count: int
    target_count: int

    def __post_init__(self) -> None:
        """Validate a primary allocation cell.

        Raises:
            ValueError: If the family, stratum, qubit count, or target count is invalid.
        """
        family_id = require_slug(self.family_id, "family_id")
        if family_id not in PRIMARY_TARGET_FAMILIES:
            msg = f"family_id must be one of {PRIMARY_TARGET_FAMILIES!r}."
            raise ValueError(msg)
        object.__setattr__(self, "family_id", family_id)
        stratum_id = require_slug(self.stratum_id, "stratum_id")
        if stratum_id not in PRIMARY_FAMILY_STRATA[family_id]:
            msg = f"stratum_id {stratum_id!r} is not registered for family {family_id!r}."
            raise ValueError(msg)
        object.__setattr__(self, "stratum_id", stratum_id)
        object.__setattr__(self, "qubit_count", require_int(self.qubit_count, "qubit_count", minimum=2))
        object.__setattr__(self, "target_count", require_int(self.target_count, "target_count", minimum=1))

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native allocation."""
        return {
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "qubit_count": self.qubit_count,
            "target_count": self.target_count,
        }

    @classmethod
    def from_dict(cls, data: object) -> SampleAllocation:
        """Construct an allocation from an exact JSON object.

        Returns:
            The validated immutable allocation.
        """
        mapping = require_mapping(data, "sample allocation")
        require_exact_keys(mapping, _SAMPLE_ALLOCATION_KEYS, "sample allocation")
        return cls(
            family_id=cast("str", mapping["family_id"]),
            stratum_id=cast("str", mapping["stratum_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            target_count=cast("int", mapping["target_count"]),
        )


@dataclass(frozen=True, slots=True)
class SampleSizeDesign:
    """Pilot-derived, checksum-sealed confirmatory sample-size design."""

    design_id: str
    preregistration_checksum: str
    pilot_nuisance_summary_checksum: str
    calculation_method_id: str
    calculation_source_checksum: str
    contrast_set_checksum: str
    target_population_configuration_checksum: str
    allocations: tuple[SampleAllocation, ...]
    optimization_seed_count: int
    fixed_test_trajectory_count: int
    achieved_power_by_contrast: Mapping[str, object]
    expected_primary_mean_half_width: float
    expected_overall_failure_rate_half_width: float
    expected_trajectory_mcse: float
    reestimation_kind: Literal["initial", "blinded_nuisance_only"]
    reestimation_parent_checksum: str | None
    schema_version: str = field(default=SAMPLE_SIZE_DESIGN_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate allocations, operating characteristics, and provenance.

        Raises:
            TypeError: If allocations contain unsupported record types.
            ValueError: If identities, metrics, or reestimation provenance are inconsistent.
        """
        object.__setattr__(self, "design_id", require_slug(self.design_id, "design_id"))
        for name in (
            "preregistration_checksum",
            "pilot_nuisance_summary_checksum",
            "calculation_source_checksum",
            "contrast_set_checksum",
            "target_population_configuration_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(
            self,
            "calculation_method_id",
            require_slug(self.calculation_method_id, "calculation_method_id"),
        )
        allocations = tuple(self.allocations)
        if not allocations or not all(isinstance(item, SampleAllocation) for item in allocations):
            msg = "allocations must contain SampleAllocation values."
            raise TypeError(msg)
        allocation_keys = tuple((item.family_id, item.stratum_id, item.qubit_count) for item in allocations)
        if len(allocation_keys) != len(set(allocation_keys)):
            msg = "Sample allocations must have unique family/stratum/qubit identities."
            raise ValueError(msg)
        object.__setattr__(self, "allocations", allocations)
        object.__setattr__(
            self,
            "optimization_seed_count",
            require_int(self.optimization_seed_count, "optimization_seed_count", minimum=2),
        )
        object.__setattr__(
            self,
            "fixed_test_trajectory_count",
            require_int(self.fixed_test_trajectory_count, "fixed_test_trajectory_count", minimum=2),
        )
        achieved_power = freeze_json_mapping(self.achieved_power_by_contrast, "achieved_power_by_contrast")
        if not achieved_power:
            msg = "achieved_power_by_contrast must not be empty."
            raise ValueError(msg)
        for contrast_id, power in achieved_power.items():
            require_slug(contrast_id, "achieved_power_by_contrast key")
            require_float(power, f"achieved_power_by_contrast.{contrast_id}", minimum=0.0, maximum=1.0)
        object.__setattr__(self, "achieved_power_by_contrast", achieved_power)
        for name in (
            "expected_primary_mean_half_width",
            "expected_overall_failure_rate_half_width",
            "expected_trajectory_mcse",
        ):
            object.__setattr__(
                self,
                name,
                require_float(getattr(self, name), name, minimum=0.0, maximum=1.0),
            )
        if self.reestimation_kind not in {"initial", "blinded_nuisance_only"}:
            msg = "reestimation_kind must be 'initial' or 'blinded_nuisance_only'."
            raise ValueError(msg)
        if self.reestimation_kind == "initial":
            if self.reestimation_parent_checksum is not None:
                msg = "An initial sample-size design must not have a reestimation parent."
                raise ValueError(msg)
        elif self.reestimation_parent_checksum is None:
            msg = "A blinded nuisance-only reestimation requires its parent design checksum."
            raise ValueError(msg)
        else:
            object.__setattr__(
                self,
                "reestimation_parent_checksum",
                require_checksum(self.reestimation_parent_checksum, "reestimation_parent_checksum"),
            )

    @property
    def target_count_by_family(self) -> Mapping[str, object]:
        """Return immutable totals derived from stratum allocations."""
        totals = dict.fromkeys(PRIMARY_TARGET_FAMILIES, 0)
        for allocation in self.allocations:
            totals[allocation.family_id] += allocation.target_count
        return freeze_json_mapping(totals, "target_count_by_family")

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete sample-size design."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered sample-size design content."""
        return {
            "schema_version": self.schema_version,
            "design_id": self.design_id,
            "preregistration_checksum": self.preregistration_checksum,
            "pilot_nuisance_summary_checksum": self.pilot_nuisance_summary_checksum,
            "calculation_method_id": self.calculation_method_id,
            "calculation_source_checksum": self.calculation_source_checksum,
            "contrast_set_checksum": self.contrast_set_checksum,
            "target_population_configuration_checksum": self.target_population_configuration_checksum,
            "allocations": [allocation.to_dict() for allocation in self.allocations],
            "optimization_seed_count": self.optimization_seed_count,
            "fixed_test_trajectory_count": self.fixed_test_trajectory_count,
            "achieved_power_by_contrast": thaw_json_mapping(self.achieved_power_by_contrast),
            "expected_primary_mean_half_width": self.expected_primary_mean_half_width,
            "expected_overall_failure_rate_half_width": self.expected_overall_failure_rate_half_width,
            "expected_trajectory_mcse": self.expected_trajectory_mcse,
            "reestimation_kind": self.reestimation_kind,
            "reestimation_parent_checksum": self.reestimation_parent_checksum,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the sealed JSON-native sample-size design."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> SampleSizeDesign:
        """Construct and checksum-verify a sample-size design.

        Returns:
            The validated immutable sample-size design.

        Raises:
            ValueError: If the schema or checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_SAMPLE_SIZE_DESIGN_KEYS, name="sample-size design")
        if mapping["schema_version"] != SAMPLE_SIZE_DESIGN_SCHEMA_VERSION:
            msg = f"schema_version must be {SAMPLE_SIZE_DESIGN_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        allocations = cast("Sequence[object]", mapping["allocations"])
        design = cls(
            design_id=cast("str", mapping["design_id"]),
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            pilot_nuisance_summary_checksum=cast("str", mapping["pilot_nuisance_summary_checksum"]),
            calculation_method_id=cast("str", mapping["calculation_method_id"]),
            calculation_source_checksum=cast("str", mapping["calculation_source_checksum"]),
            contrast_set_checksum=cast("str", mapping["contrast_set_checksum"]),
            target_population_configuration_checksum=cast(
                "str",
                mapping["target_population_configuration_checksum"],
            ),
            allocations=tuple(SampleAllocation.from_dict(item) for item in allocations),
            optimization_seed_count=cast("int", mapping["optimization_seed_count"]),
            fixed_test_trajectory_count=cast("int", mapping["fixed_test_trajectory_count"]),
            achieved_power_by_contrast=cast("Mapping[str, object]", mapping["achieved_power_by_contrast"]),
            expected_primary_mean_half_width=cast("float", mapping["expected_primary_mean_half_width"]),
            expected_overall_failure_rate_half_width=cast(
                "float",
                mapping["expected_overall_failure_rate_half_width"],
            ),
            expected_trajectory_mcse=cast("float", mapping["expected_trajectory_mcse"]),
            reestimation_kind=cast(
                "Literal['initial', 'blinded_nuisance_only']",
                mapping["reestimation_kind"],
            ),
            reestimation_parent_checksum=cast("str | None", mapping["reestimation_parent_checksum"]),
        )
        supplied = cast("str", mapping["content_checksum"])
        if design.content_checksum != supplied:
            msg = f"Sample-size design checksum changed during normalization: expected {supplied}."
            raise ValueError(msg)
        return design

    @classmethod
    def from_json(cls, payload: str) -> SampleSizeDesign:
        """Construct a sample-size design from canonical JSON.

        Returns:
            The validated immutable sample-size design.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class AnalysisSourceFileRef:
    """One commit-addressed executable analysis source file."""

    repo_path: str
    git_blob_id: str
    content_checksum: str

    def __post_init__(self) -> None:
        """Validate source path and immutable content identifiers."""
        object.__setattr__(self, "repo_path", require_relative_path(self.repo_path, "repo_path"))
        object.__setattr__(self, "git_blob_id", require_git_blob(self.git_blob_id, "git_blob_id"))
        object.__setattr__(
            self,
            "content_checksum",
            require_checksum(self.content_checksum, "content_checksum"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native source-file reference."""
        return {
            "repo_path": self.repo_path,
            "git_blob_id": self.git_blob_id,
            "content_checksum": self.content_checksum,
        }

    @classmethod
    def from_dict(cls, data: object) -> AnalysisSourceFileRef:
        """Construct a source-file reference from an exact JSON object.

        Returns:
            The validated immutable source-file reference.
        """
        mapping = require_mapping(data, "analysis source file")
        require_exact_keys(mapping, _ANALYSIS_SOURCE_FILE_KEYS, "analysis source file")
        return cls(
            repo_path=cast("str", mapping["repo_path"]),
            git_blob_id=cast("str", mapping["git_blob_id"]),
            content_checksum=cast("str", mapping["content_checksum"]),
        )


@dataclass(frozen=True, slots=True)
class AnalysisSourceManifest:
    """Commit-addressed executable primary-analysis source manifest."""

    manifest_id: str
    preregistration_checksum: str
    analysis_template_checksum: str
    source_commit: str
    entry_point: str
    source_files: tuple[AnalysisSourceFileRef, ...]
    environment_lock_checksum: str
    execution_source_manifest_checksum: str
    clean_worktree: bool
    schema_version: str = field(default=ANALYSIS_SOURCE_MANIFEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate source identities and clean-worktree provenance.

        Raises:
            TypeError: If source_files contains unsupported record types.
            ValueError: If paths are unsorted, duplicated, incomplete, or not clean.
        """
        object.__setattr__(self, "manifest_id", require_slug(self.manifest_id, "manifest_id"))
        object.__setattr__(
            self,
            "preregistration_checksum",
            require_checksum(self.preregistration_checksum, "preregistration_checksum"),
        )
        object.__setattr__(
            self,
            "analysis_template_checksum",
            require_checksum(self.analysis_template_checksum, "analysis_template_checksum"),
        )
        object.__setattr__(self, "source_commit", require_git_commit(self.source_commit, "source_commit"))
        object.__setattr__(self, "entry_point", require_relative_path(self.entry_point, "entry_point"))
        source_files = tuple(self.source_files)
        if not source_files or not all(isinstance(item, AnalysisSourceFileRef) for item in source_files):
            msg = "source_files must contain AnalysisSourceFileRef values."
            raise TypeError(msg)
        paths = tuple(source_file.repo_path for source_file in source_files)
        if paths != tuple(sorted(set(paths))):
            msg = "source_files must have unique paths in lexical order."
            raise ValueError(msg)
        if self.entry_point not in paths:
            msg = "entry_point must reference a file in source_files."
            raise ValueError(msg)
        object.__setattr__(self, "source_files", source_files)
        object.__setattr__(
            self,
            "environment_lock_checksum",
            require_checksum(self.environment_lock_checksum, "environment_lock_checksum"),
        )
        object.__setattr__(
            self,
            "execution_source_manifest_checksum",
            require_checksum(
                self.execution_source_manifest_checksum,
                "execution_source_manifest_checksum",
            ),
        )
        if require_bool(self.clean_worktree, "clean_worktree") is not True:
            msg = "Analysis source must be sealed from a clean worktree."
            raise ValueError(msg)

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete executable analysis manifest."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered analysis-source content."""
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "preregistration_checksum": self.preregistration_checksum,
            "analysis_template_checksum": self.analysis_template_checksum,
            "source_commit": self.source_commit,
            "entry_point": self.entry_point,
            "source_files": [source_file.to_dict() for source_file in self.source_files],
            "environment_lock_checksum": self.environment_lock_checksum,
            "execution_source_manifest_checksum": self.execution_source_manifest_checksum,
            "clean_worktree": self.clean_worktree,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the sealed JSON-native analysis-source manifest."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> AnalysisSourceManifest:
        """Construct and checksum-verify an analysis-source manifest.

        Returns:
            The validated immutable analysis-source manifest.

        Raises:
            ValueError: If the schema or checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_ANALYSIS_SOURCE_MANIFEST_KEYS,
            name="analysis source manifest",
        )
        if mapping["schema_version"] != ANALYSIS_SOURCE_MANIFEST_SCHEMA_VERSION:
            msg = f"schema_version must be {ANALYSIS_SOURCE_MANIFEST_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        source_files = cast("Sequence[object]", mapping["source_files"])
        manifest = cls(
            manifest_id=cast("str", mapping["manifest_id"]),
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            analysis_template_checksum=cast("str", mapping["analysis_template_checksum"]),
            source_commit=cast("str", mapping["source_commit"]),
            entry_point=cast("str", mapping["entry_point"]),
            source_files=tuple(AnalysisSourceFileRef.from_dict(item) for item in source_files),
            environment_lock_checksum=cast("str", mapping["environment_lock_checksum"]),
            execution_source_manifest_checksum=cast("str", mapping["execution_source_manifest_checksum"]),
            clean_worktree=cast("bool", mapping["clean_worktree"]),
        )
        supplied = cast("str", mapping["content_checksum"])
        if manifest.content_checksum != supplied:
            msg = f"Analysis-source manifest checksum changed during normalization: expected {supplied}."
            raise ValueError(msg)
        return manifest

    @classmethod
    def from_json(cls, payload: str) -> AnalysisSourceManifest:
        """Construct an analysis-source manifest from canonical JSON.

        Returns:
            The validated immutable analysis-source manifest.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def verify_analysis_source_files(
    manifest: AnalysisSourceManifest,
    repository_root: Path,
) -> tuple[str, ...]:
    """Verify every executable source file against its commit and digest.

    Args:
        manifest: Analysis-source manifest to verify.
        repository_root: Git worktree containing the sealed source commit.

    Returns:
        Verified source paths in manifest order.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If Git is unavailable or a source identity differs.
    """
    if not isinstance(manifest, AnalysisSourceManifest):
        msg = f"manifest must be an AnalysisSourceManifest, got {type(manifest).__name__}."
        raise TypeError(msg)
    if not isinstance(repository_root, Path):
        msg = f"repository_root must be a pathlib.Path, got {type(repository_root).__name__}."
        raise TypeError(msg)
    git_executable = shutil.which("git")
    if git_executable is None:
        msg = "Git is required to verify executable analysis sources."
        raise ValueError(msg)
    verified: list[str] = []
    for source_file in manifest.source_files:
        revision_path = f"{manifest.source_commit}:{source_file.repo_path}"
        blob_result = subprocess.run(  # noqa: S603 -- executable, commit, and path are strictly validated
            [git_executable, "-C", str(repository_root), "rev-parse", revision_path],
            check=False,
            capture_output=True,
            text=True,
        )
        if blob_result.returncode != 0:
            detail = blob_result.stderr.strip() or blob_result.stdout.strip()
            msg = f"Could not resolve analysis source {revision_path!r}: {detail}."
            raise ValueError(msg)
        actual_blob = blob_result.stdout.strip()
        if actual_blob != source_file.git_blob_id:
            msg = (
                f"Analysis source {source_file.repo_path!r} blob mismatch: "
                f"expected {source_file.git_blob_id}, got {actual_blob}."
            )
            raise ValueError(msg)
        content_result = subprocess.run(  # noqa: S603 -- executable and blob identifier are strictly validated
            [git_executable, "-C", str(repository_root), "cat-file", "blob", actual_blob],
            check=False,
            capture_output=True,
        )
        if content_result.returncode != 0:
            detail = content_result.stderr.decode(errors="replace").strip()
            msg = f"Could not read analysis source blob {actual_blob}: {detail}."
            raise ValueError(msg)
        actual_checksum = f"sha256:{hashlib.sha256(content_result.stdout).hexdigest()}"
        if actual_checksum != source_file.content_checksum:
            msg = (
                f"Analysis source {source_file.repo_path!r} checksum mismatch: "
                f"expected {source_file.content_checksum}, got {actual_checksum}."
            )
            raise ValueError(msg)
        verified.append(source_file.repo_path)
    return tuple(verified)


@dataclass(frozen=True, slots=True)
class FinalConfirmationSeal:
    """Final checksum-locked confirmation design instantiated after screening."""

    seal_id: str
    preregistration_checksum: str
    promotion_decision_checksum: str
    promoted_method_id: str
    promoted_configuration_checksum: str
    comparators: tuple[FinalComparatorRef, ...]
    primary_contrasts: tuple[PrimaryContrastBinding, ...]
    confirmatory_target_manifest_checksum: str
    target_count_by_family: Mapping[str, object]
    optimization_seed_count: int
    fixed_test_trajectory_count: int
    primary_noise_condition: Mapping[str, object]
    primary_resource_budget: Mapping[str, object]
    hyperparameters_checksum: str
    execution_source_checksum: str
    analysis_template_checksum: str
    analysis_source_manifest_checksum: str
    sample_size_design_checksum: str
    failure_policy_checksum: str
    schema_version: str = field(default=CONFIRMATION_SEAL_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate all independently sealed confirmation inputs.

        Raises:
            ValueError: If a comparator duplicates the promoted method or configuration.
        """
        object.__setattr__(self, "seal_id", require_slug(self.seal_id, "seal_id"))
        for name in (
            "preregistration_checksum",
            "promotion_decision_checksum",
            "promoted_configuration_checksum",
            "confirmatory_target_manifest_checksum",
            "hyperparameters_checksum",
            "execution_source_checksum",
            "analysis_template_checksum",
            "analysis_source_manifest_checksum",
            "sample_size_design_checksum",
            "failure_policy_checksum",
        ):
            object.__setattr__(self, name, require_checksum(getattr(self, name), name))
        object.__setattr__(self, "promoted_method_id", require_slug(self.promoted_method_id, "promoted_method_id"))
        comparators = _validate_comparators(self.comparators)
        if self.promoted_method_id in {item.method_id for item in comparators}:
            msg = "comparators must not repeat the promoted method."
            raise ValueError(msg)
        if self.promoted_configuration_checksum in {item.configuration_checksum for item in comparators}:
            msg = "comparators must not repeat the promoted configuration."
            raise ValueError(msg)
        object.__setattr__(self, "comparators", comparators)
        object.__setattr__(self, "primary_contrasts", _validate_primary_contrasts(self.primary_contrasts))
        object.__setattr__(self, "target_count_by_family", _validate_target_counts(self.target_count_by_family))
        object.__setattr__(
            self,
            "optimization_seed_count",
            require_int(self.optimization_seed_count, "optimization_seed_count", minimum=1),
        )
        object.__setattr__(
            self,
            "fixed_test_trajectory_count",
            require_int(self.fixed_test_trajectory_count, "fixed_test_trajectory_count", minimum=2),
        )
        object.__setattr__(
            self,
            "primary_noise_condition",
            _validate_primary_noise(self.primary_noise_condition),
        )
        object.__setattr__(
            self,
            "primary_resource_budget",
            _validate_resource_budget(self.primary_resource_budget),
        )

    @property
    def content_checksum(self) -> str:
        """Complete final-seal checksum."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return the checksum-covered final-seal payload."""
        return {
            "schema_version": self.schema_version,
            "seal_id": self.seal_id,
            "preregistration_checksum": self.preregistration_checksum,
            "promotion_decision_checksum": self.promotion_decision_checksum,
            "promoted_method_id": self.promoted_method_id,
            "promoted_configuration_checksum": self.promoted_configuration_checksum,
            "comparators": [comparator.to_dict() for comparator in self.comparators],
            "primary_contrasts": [binding.to_dict() for binding in self.primary_contrasts],
            "confirmatory_target_manifest_checksum": self.confirmatory_target_manifest_checksum,
            "target_count_by_family": thaw_json_mapping(self.target_count_by_family),
            "optimization_seed_count": self.optimization_seed_count,
            "fixed_test_trajectory_count": self.fixed_test_trajectory_count,
            "primary_noise_condition": thaw_json_mapping(self.primary_noise_condition),
            "primary_resource_budget": thaw_json_mapping(self.primary_resource_budget),
            "hyperparameters_checksum": self.hyperparameters_checksum,
            "execution_source_checksum": self.execution_source_checksum,
            "analysis_template_checksum": self.analysis_template_checksum,
            "analysis_source_manifest_checksum": self.analysis_source_manifest_checksum,
            "sample_size_design_checksum": self.sample_size_design_checksum,
            "failure_policy_checksum": self.failure_policy_checksum,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the sealed JSON-native confirmation design."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> FinalConfirmationSeal:
        """Construct and checksum-verify a final confirmation seal.

        Returns:
            The validated immutable final confirmation seal.

        Raises:
            ValueError: If the schema version or normalized checksum is inconsistent.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_FINAL_SEAL_KEYS, name="final confirmation seal")
        if mapping["schema_version"] != CONFIRMATION_SEAL_SCHEMA_VERSION:
            msg = f"schema_version must be {CONFIRMATION_SEAL_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        comparators = cast("Sequence[object]", mapping["comparators"])
        primary_contrasts = cast("Sequence[object]", mapping["primary_contrasts"])
        seal = cls(
            seal_id=cast("str", mapping["seal_id"]),
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            promotion_decision_checksum=cast("str", mapping["promotion_decision_checksum"]),
            promoted_method_id=cast("str", mapping["promoted_method_id"]),
            promoted_configuration_checksum=cast("str", mapping["promoted_configuration_checksum"]),
            comparators=tuple(FinalComparatorRef.from_dict(item) for item in comparators),
            primary_contrasts=tuple(PrimaryContrastBinding.from_dict(item) for item in primary_contrasts),
            confirmatory_target_manifest_checksum=cast("str", mapping["confirmatory_target_manifest_checksum"]),
            target_count_by_family=cast("Mapping[str, object]", mapping["target_count_by_family"]),
            optimization_seed_count=cast("int", mapping["optimization_seed_count"]),
            fixed_test_trajectory_count=cast("int", mapping["fixed_test_trajectory_count"]),
            primary_noise_condition=cast("Mapping[str, object]", mapping["primary_noise_condition"]),
            primary_resource_budget=cast("Mapping[str, object]", mapping["primary_resource_budget"]),
            hyperparameters_checksum=cast("str", mapping["hyperparameters_checksum"]),
            execution_source_checksum=cast("str", mapping["execution_source_checksum"]),
            analysis_template_checksum=cast("str", mapping["analysis_template_checksum"]),
            analysis_source_manifest_checksum=cast("str", mapping["analysis_source_manifest_checksum"]),
            sample_size_design_checksum=cast("str", mapping["sample_size_design_checksum"]),
            failure_policy_checksum=cast("str", mapping["failure_policy_checksum"]),
        )
        supplied = cast("str", mapping["content_checksum"])
        if seal.content_checksum != supplied:
            msg = f"Final confirmation checksum changed during normalization: expected {supplied}."
            raise ValueError(msg)
        return seal

    @classmethod
    def from_json(cls, payload: str) -> FinalConfirmationSeal:
        """Construct a final seal from canonical JSON text.

        Returns:
            The validated immutable final confirmation seal.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def validate_final_configuration_execution_manifest(
    seal: FinalConfirmationSeal,
    manifest: FinalConfigurationExecutionManifest,
) -> None:
    """Authenticate the exact executable configuration set against a final seal.

    Raises:
        TypeError: If either artifact has the wrong typed schema.
        ValueError: If the manifest root, method/configuration set, or comparator schema differs.
    """
    if not isinstance(seal, FinalConfirmationSeal):
        msg = "seal must be a FinalConfirmationSeal."
        raise TypeError(msg)
    if not isinstance(manifest, FinalConfigurationExecutionManifest):
        msg = "manifest must be a FinalConfigurationExecutionManifest."
        raise TypeError(msg)
    if manifest.content_checksum != seal.hyperparameters_checksum:
        msg = "Final configuration execution manifest does not reproduce the seal hyperparameters root."
        raise ValueError(msg)
    expected_methods = {
        seal.promoted_configuration_checksum: seal.promoted_method_id,
        **{item.configuration_checksum: item.method_id for item in seal.comparators},
    }
    actual_methods = {item.configuration_checksum: item.method_id for item in manifest.entries}
    if actual_methods != expected_methods:
        msg = "Final configuration execution manifest differs from the exact promoted-plus-comparator set."
        raise ValueError(msg)
    schema_by_configuration = {
        item.configuration_checksum: item.configuration_schema_version for item in manifest.entries
    }
    for comparator in seal.comparators:
        if schema_by_configuration[comparator.configuration_checksum] != comparator.configuration_schema_version:
            msg = "Final comparator configuration schema differs from its execution manifest reference."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ConfirmationAuthorization:
    """Opaque in-process guard proving that all confirmation seals agree."""

    preregistration_checksum: str
    final_seal_checksum: str
    target_manifest_checksum: str
    execution_source_checksum: str
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject direct construction outside the authorization function.

        Raises:
            ValueError: If construction did not originate from :func:`authorize_confirmation`.
        """
        if self._marker is not _AUTHORIZATION_SENTINEL:
            msg = "ConfirmationAuthorization may only be created by authorize_confirmation."
            raise ValueError(msg)


def authorize_confirmation(
    preregistration: InitialPreregistration,
    manifest: ScreeningManifest,
    evidence: ScreeningEvidence,
    decision: PromotionDecision,
    sample_size_design: SampleSizeDesign,
    analysis_source_manifest: AnalysisSourceManifest,
    final_seal: FinalConfirmationSeal,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
    resource_calibration: FinalResourceCalibrationManifest,
    repository_root: Path,
) -> ConfirmationAuthorization:
    """Authorize post-seal confirmatory target materialization.

    This object prevents accidental in-process access only. External custody of
    target seeds provides the actual pre-seal information boundary.

    Args:
        preregistration: Initial protocol seal.
        manifest: Complete sealed screening universe.
        evidence: Complete sealed raw screening outcome ledger.
        decision: Mechanical screening decision.
        sample_size_design: Pilot-derived confirmatory sample-size design.
        analysis_source_manifest: Commit-addressed executable primary-analysis source.
        final_seal: Pilot- and screening-instantiated confirmation design.
        configuration_execution_manifest: Exact per-configuration executable identities.
        resource_calibration: Typed pilot/screen production resource calibration.
        repository_root: Git worktree containing the sealed analysis source commit.

    Returns:
        An opaque authorization token for the source-locked WP23 materializer.

    Raises:
        TypeError: If a supplied object has the wrong record type.
        ValueError: If any independently sealed identity disagrees.
    """
    if not isinstance(preregistration, InitialPreregistration):
        msg = "preregistration must be an InitialPreregistration."
        raise TypeError(msg)
    if preregistration.content_checksum != TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM:
        msg = "The supplied preregistration does not match the trusted checked-in protocol digest."
        raise ValueError(msg)
    if not isinstance(manifest, ScreeningManifest):
        msg = "manifest must be a ScreeningManifest."
        raise TypeError(msg)
    if not isinstance(evidence, ScreeningEvidence):
        msg = "evidence must be a ScreeningEvidence."
        raise TypeError(msg)
    if not isinstance(decision, PromotionDecision):
        msg = "decision must be a PromotionDecision."
        raise TypeError(msg)
    if not isinstance(sample_size_design, SampleSizeDesign):
        msg = "sample_size_design must be a SampleSizeDesign."
        raise TypeError(msg)
    if not isinstance(analysis_source_manifest, AnalysisSourceManifest):
        msg = "analysis_source_manifest must be an AnalysisSourceManifest."
        raise TypeError(msg)
    if not isinstance(final_seal, FinalConfirmationSeal):
        msg = "final_seal must be a FinalConfirmationSeal."
        raise TypeError(msg)
    if not isinstance(configuration_execution_manifest, FinalConfigurationExecutionManifest):
        msg = "configuration_execution_manifest must be a FinalConfigurationExecutionManifest."
        raise TypeError(msg)
    # Imported here because the concrete custody implementation depends on the
    # foundational protocol records in this module.  An open ABC check is not
    # sufficient at this boundary: a caller-authored subclass could otherwise
    # echo the known seal roots without carrying the required 720 pilot and
    # 1,296 screening projections.
    from .screening import ProductionResourceCalibration  # noqa: PLC0415

    if type(resource_calibration) is not ProductionResourceCalibration:
        msg = "resource_calibration must be the exact ProductionResourceCalibration type."
        raise TypeError(msg)
    if not isinstance(repository_root, Path):
        msg = "repository_root must be a pathlib.Path."
        raise TypeError(msg)
    recomputed_decision = select_promoted_candidate(preregistration, manifest, evidence)
    if recomputed_decision.content_checksum != decision.content_checksum:
        msg = "Promotion decision is not the exact result of the sealed screening manifest and raw evidence."
        raise ValueError(msg)
    if final_seal.preregistration_checksum != preregistration.content_checksum:
        msg = "Final seal does not reference the supplied preregistration."
        raise ValueError(msg)
    if final_seal.promotion_decision_checksum != decision.content_checksum:
        msg = "Final seal does not reference the supplied promotion decision."
        raise ValueError(msg)
    if (
        final_seal.promoted_method_id != decision.promoted_method_id
        or final_seal.promoted_configuration_checksum != decision.promoted_configuration_checksum
    ):
        msg = "Final seal promoted configuration differs from the mechanical decision."
        raise ValueError(msg)
    candidate_by_checksum = {candidate.configuration_checksum: candidate for candidate in manifest.candidates}
    for comparator in final_seal.comparators:
        candidate = candidate_by_checksum.get(comparator.configuration_checksum)
        if candidate is None or candidate.method_id != comparator.method_id:
            msg = f"Comparator {comparator.configuration_checksum!r} is not the exact screened configuration."
            raise ValueError(msg)
        if candidate.configuration_schema_version != comparator.configuration_schema_version:
            msg = f"Comparator {comparator.configuration_checksum!r} uses a changed configuration schema."
            raise ValueError(msg)
        if candidate.matching_projection_checksum != comparator.matching_projection_checksum:
            msg = f"Comparator {comparator.configuration_checksum!r} uses a changed matching projection."
            raise ValueError(msg)

    baseline_checksum = decision.baseline_configuration_checksum
    baseline_candidate = candidate_by_checksum[baseline_checksum]
    noiseless_candidates = [
        candidate for candidate in manifest.candidates if candidate.method_id == "layerwise_bmpd_noiseless"
    ]
    assert len(noiseless_candidates) == 1
    noiseless_candidate = noiseless_candidates[0]
    comparator_by_role = {comparator.role: comparator for comparator in final_seal.comparators}
    noiseless_comparator = comparator_by_role.get("matched_noiseless_control")
    if (
        noiseless_comparator is None
        or noiseless_comparator.configuration_checksum != noiseless_candidate.configuration_checksum
    ):
        msg = "Final seal omits the exact screened matched noiseless comparator."
        raise ValueError(msg)
    if (
        noiseless_comparator.matched_to_configuration_checksum != baseline_checksum
        or noiseless_comparator.matching_projection_checksum != baseline_candidate.matching_projection_checksum
    ):
        msg = "The noiseless comparator is not bound to the exact noisy v2 baseline projection."
        raise ValueError(msg)
    v2_reference = comparator_by_role.get("layerwise_v2_reference")
    if decision.promoted_configuration_checksum == baseline_checksum:
        if v2_reference is not None:
            msg = "The promoted v2 baseline must not be duplicated as a comparator."
            raise ValueError(msg)
    elif (
        v2_reference is None
        or v2_reference.configuration_checksum != baseline_checksum
        or v2_reference.matched_to_configuration_checksum != noiseless_candidate.configuration_checksum
        or v2_reference.matching_projection_checksum != baseline_candidate.matching_projection_checksum
    ):
        msg = "The corrected v2 comparator is not the exact screened baseline and matched pair."
        raise ValueError(msg)

    contrast_by_id = {binding.contrast_id: binding for binding in final_seal.primary_contrasts}
    expected_contrast_ids = {"noisy_vs_noiseless"}
    if decision.promoted_configuration_checksum != baseline_checksum:
        expected_contrast_ids.add("promoted_vs_layerwise_v2_if_distinct")
    if set(contrast_by_id) != expected_contrast_ids:
        msg = "Final seal primary contrasts do not match the applicable preregistered contrast set."
        raise ValueError(msg)
    noisy_contrast = contrast_by_id["noisy_vs_noiseless"]
    if (
        noisy_contrast.treatment_configuration_checksum != baseline_checksum
        or noisy_contrast.control_configuration_checksum != noiseless_candidate.configuration_checksum
        or noisy_contrast.paired_block_policy_checksum != preregistration.paired_block_policy_checksum
        or noisy_contrast.matching_projection_checksum != baseline_candidate.matching_projection_checksum
    ):
        msg = "The noisy-versus-noiseless contrast is not bound to the exact matched configurations."
        raise ValueError(msg)
    promoted_contrast = contrast_by_id.get("promoted_vs_layerwise_v2_if_distinct")
    if promoted_contrast is not None and (
        promoted_contrast.treatment_configuration_checksum != decision.promoted_configuration_checksum
        or promoted_contrast.control_configuration_checksum != baseline_checksum
        or promoted_contrast.paired_block_policy_checksum != preregistration.paired_block_policy_checksum
        or promoted_contrast.matching_projection_checksum is not None
    ):
        msg = "The promoted-versus-v2 contrast is not bound to the exact screened configurations."
        raise ValueError(msg)

    if canonical_checksum(final_seal.primary_noise_condition) != canonical_checksum(
        preregistration.primary_noise_condition
    ):
        msg = "Final seal primary noise condition differs from the preregistration."
        raise ValueError(msg)
    preregistered_resource = preregistration.primary_resource_constraint
    if (
        final_seal.primary_resource_budget["metric"] != preregistered_resource["metric"]
        or final_seal.primary_resource_budget["cap_per_chain_edge"] != preregistered_resource["cap_per_chain_edge"]
    ):
        msg = "Final seal primary resource differs from the preregistration."
        raise ValueError(msg)
    if final_seal.failure_policy_checksum != preregistration.failure_policy_checksum:
        msg = "Final seal failure policy checksum differs from the preregistration."
        raise ValueError(msg)
    if final_seal.analysis_template_checksum != preregistration.analysis_template_checksum:
        msg = "Final seal analysis-template checksum differs from the preregistration."
        raise ValueError(msg)
    if analysis_source_manifest.preregistration_checksum != preregistration.content_checksum:
        msg = "Analysis-source manifest does not reference the supplied preregistration."
        raise ValueError(msg)
    if analysis_source_manifest.analysis_template_checksum != preregistration.analysis_template_checksum:
        msg = "Analysis-source manifest uses a changed primary-analysis template."
        raise ValueError(msg)
    if final_seal.analysis_source_manifest_checksum != analysis_source_manifest.content_checksum:
        msg = "Final seal does not reference the supplied executable analysis-source manifest."
        raise ValueError(msg)
    if final_seal.execution_source_checksum != analysis_source_manifest.execution_source_manifest_checksum:
        msg = "Executable analysis is not linked to the final execution-source manifest."
        raise ValueError(msg)
    verify_analysis_source_files(analysis_source_manifest, repository_root)

    sample_policy = preregistration.sample_size_policy
    if sample_size_design.preregistration_checksum != preregistration.content_checksum:
        msg = "Sample-size design does not reference the supplied preregistration."
        raise ValueError(msg)
    if final_seal.sample_size_design_checksum != sample_size_design.content_checksum:
        msg = "Final seal does not reference the supplied sample-size design."
        raise ValueError(msg)
    if sample_size_design.calculation_method_id != sample_policy["method"]:
        msg = "Sample-size design uses a calculation method not frozen by the preregistration."
        raise ValueError(msg)
    if sample_size_design.contrast_set_checksum != preregistration.contrast_set_checksum:
        msg = "Sample-size design uses a changed primary contrast set."
        raise ValueError(msg)
    if (
        sample_size_design.target_population_configuration_checksum
        != preregistration.target_population_configuration_checksum
    ):
        msg = "Sample-size design uses a changed target-population configuration."
        raise ValueError(msg)
    primary_qubits = tuple(cast("Sequence[int]", preregistration.target_population_policy["primary_qubit_counts"]))
    if len(primary_qubits) != 1:
        msg = "The v1 sample-size validator requires one preregistered primary qubit count."
        raise ValueError(msg)
    expected_allocation_keys = {
        (family_id, stratum_id, primary_qubits[0])
        for family_id, strata in PRIMARY_FAMILY_STRATA.items()
        for stratum_id in strata
    }
    actual_allocation_keys = {
        (allocation.family_id, allocation.stratum_id, allocation.qubit_count)
        for allocation in sample_size_design.allocations
    }
    if actual_allocation_keys != expected_allocation_keys:
        msg = "Sample-size design does not allocate every primary family and stratum exactly once."
        raise ValueError(msg)
    family_counts = sample_size_design.target_count_by_family
    minimum_targets = cast("int", sample_policy["minimum_targets_per_family"])
    maximum_targets = cast("int", sample_policy["maximum_targets_per_family"])
    increment = cast("int", sample_policy["target_count_increment"])
    for family_id, strata in PRIMARY_FAMILY_STRATA.items():
        stratum_counts = [
            allocation.target_count
            for allocation in sample_size_design.allocations
            if allocation.family_id == family_id
        ]
        total = cast("int", family_counts[family_id])
        if (
            len(stratum_counts) != len(strata)
            or len(set(stratum_counts)) != 1
            or not minimum_targets <= total <= maximum_targets
            or total % increment != 0
        ):
            msg = f"Sample-size allocation for family {family_id!r} violates the balanced frozen bounds."
            raise ValueError(msg)
    allowed_seed_counts = cast("Sequence[int]", sample_policy["allowed_optimization_seed_counts"])
    if sample_size_design.optimization_seed_count not in allowed_seed_counts:
        msg = "Sample-size design uses an unsupported optimization-seed count."
        raise ValueError(msg)
    expected_power_ids = {
        cast("str", item["contrast_id"])
        for item in cast("Sequence[Mapping[str, object]]", preregistration.multiplicity_policy["contrast_definitions"])
    }
    if set(sample_size_design.achieved_power_by_contrast) != expected_power_ids or any(
        cast("float", power) < cast("float", sample_policy["power"])
        for power in sample_size_design.achieved_power_by_contrast.values()
    ):
        msg = "Sample-size design does not meet the frozen power requirement for every primary contrast."
        raise ValueError(msg)
    if (
        sample_size_design.expected_primary_mean_half_width > cast("float", sample_policy["target_mean_half_width"])
        or sample_size_design.expected_overall_failure_rate_half_width
        > cast("float", sample_policy["failure_rate_half_width"])
        or sample_size_design.expected_trajectory_mcse > cast("float", sample_policy["trajectory_mcse_target"])
    ):
        msg = "Sample-size design does not meet the frozen precision requirements."
        raise ValueError(msg)
    minimum_trajectories = cast("int", sample_policy["trajectory_count_min"])
    maximum_trajectories = cast("int", sample_policy["trajectory_count_max"])
    if not minimum_trajectories <= sample_size_design.fixed_test_trajectory_count <= maximum_trajectories:
        msg = "Final test trajectory count is outside the preregistered fixed range."
        raise ValueError(msg)
    if (
        final_seal.target_count_by_family != family_counts
        or final_seal.optimization_seed_count != sample_size_design.optimization_seed_count
        or final_seal.fixed_test_trajectory_count != sample_size_design.fixed_test_trajectory_count
    ):
        msg = "Final seal denormalized sample sizes differ from the sealed sample-size design."
        raise ValueError(msg)
    validate_final_configuration_execution_manifest(final_seal, configuration_execution_manifest)
    for execution in configuration_execution_manifest.entries:
        candidate = candidate_by_checksum.get(execution.configuration_checksum)
        if candidate is None or candidate.method_id != execution.method_id:
            msg = f"Final execution {execution.configuration_checksum!r} is not the exact screened configuration."
            raise ValueError(msg)
        if candidate.configuration_schema_version != execution.configuration_schema_version:
            msg = f"Final execution {execution.configuration_checksum!r} uses a changed configuration schema."
            raise ValueError(msg)
    calibration_cap = require_float(
        resource_calibration.normalized_compute_cap,
        "resource_calibration.normalized_compute_cap",
        minimum=0.0,
    )
    if (
        resource_calibration.preregistration_checksum != preregistration.content_checksum
        or resource_calibration.screening_manifest_checksum != manifest.content_checksum
        or resource_calibration.execution_source_manifest_checksum != final_seal.execution_source_checksum
        or resource_calibration.content_checksum
        != final_seal.primary_resource_budget["reachable_stratum_manifest_checksum"]
        or float(calibration_cap).hex()
        != float(cast("float", final_seal.primary_resource_budget["normalized_compute_cap"])).hex()
    ):
        msg = "Final resource budget is not the exact typed pilot/screen production calibration."
        raise ValueError(msg)
    return ConfirmationAuthorization(
        preregistration_checksum=preregistration.content_checksum,
        final_seal_checksum=final_seal.content_checksum,
        target_manifest_checksum=final_seal.confirmatory_target_manifest_checksum,
        execution_source_checksum=final_seal.execution_source_checksum,
        _marker=_AUTHORIZATION_SENTINEL,
    )


def load_initial_preregistration(path: Path = DEFAULT_PREREGISTRATION_PATH) -> InitialPreregistration:
    """Load the checked-in canonical initial preregistration.

    Args:
        path: Canonical preregistration document.

    Returns:
        The validated immutable protocol.

    Raises:
        ValueError: If the document differs from the trusted checked-in digest.
    """
    preregistration = InitialPreregistration.from_dict(read_canonical_json_object(path))
    if preregistration.content_checksum != TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM:
        msg = (
            "Checked-in preregistration digest differs from the trusted runtime constant: "
            f"expected {TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM}, got {preregistration.content_checksum}."
        )
        raise ValueError(msg)
    return preregistration


__all__ = [
    "ANALYSIS_SOURCE_MANIFEST_SCHEMA_VERSION",
    "CONFIRMATION_SEAL_SCHEMA_VERSION",
    "DATA_ROLES",
    "DEFAULT_PREREGISTRATION_PATH",
    "FINAL_CONFIGURATION_EXECUTION_MANIFEST_SCHEMA_VERSION",
    "FINAL_CONFIGURATION_EXECUTION_REF_SCHEMA_VERSION",
    "PREREGISTRATION_SCHEMA_VERSION",
    "PRIMARY_FAMILY_STRATA",
    "PRIMARY_TARGET_FAMILIES",
    "PROMOTION_DECISION_SCHEMA_VERSION",
    "PROMOTION_RULE_VERSION",
    "SAMPLE_SIZE_DESIGN_SCHEMA_VERSION",
    "SCREENING_EVIDENCE_SCHEMA_VERSION",
    "SCREENING_MANIFEST_SCHEMA_VERSION",
    "TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM",
    "AnalysisSourceFileRef",
    "AnalysisSourceManifest",
    "CandidateSummary",
    "ConfirmationAuthorization",
    "FinalComparatorRef",
    "FinalConfigurationExecutionManifest",
    "FinalConfigurationExecutionRef",
    "FinalConfirmationSeal",
    "FinalResourceCalibrationManifest",
    "InitialPreregistration",
    "PrimaryContrastBinding",
    "PromotionDecision",
    "PromotionObservation",
    "SampleAllocation",
    "SampleSizeDesign",
    "ScreeningCandidateRef",
    "ScreeningCell",
    "ScreeningEvidence",
    "ScreeningManifest",
    "authorize_confirmation",
    "load_initial_preregistration",
    "select_promoted_candidate",
    "validate_final_configuration_execution_manifest",
    "verify_analysis_source_files",
]
