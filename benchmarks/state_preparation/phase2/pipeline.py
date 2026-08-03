# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Versioned staged-training and evaluation records for Phase II.

The records in this module deliberately do not depend on a concrete optimizer
or target generator.  They freeze the scientific inputs needed by the later
Phase II runners while keeping filesystem paths and observed outcomes out of
the stable training and evaluation identities.
"""

# The module's many tiny strict validators inherit detailed error contracts
# from ``validation.py``; repeating those contracts in every private helper
# would obscure the scientific schema documentation.
# ruff: noqa: DOC201, DOC501, DOC502

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, NoReturn, Protocol, cast

from benchmarks.state_preparation.constants import (
    BALLARIN_NOISE_ID,
    NOISELESS_NOISE_ID,
    SUPPORTED_QUBIT_COUNTS,
)
from benchmarks.state_preparation.constants import (
    TARGET_IDS as PHASE1_FIXTURE_TARGET_IDS,
)

from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .protocol import (
    DATA_ROLES,
    PRIMARY_FAMILY_STRATA,
    TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
    ScreeningCandidateRef,
    ScreeningCell,
    ScreeningManifest,
    load_initial_preregistration,
)
from .targets import TargetInstanceSpec, TargetPopulationManifest, verify_screening_target_population
from .validation import (
    require_bool,
    require_checksum,
    require_exact_keys,
    require_float,
    require_int,
    require_mapping,
    require_nonempty_text,
    require_relative_path,
    require_slug,
    require_string,
)

TRAINING_STAGE_CONFIG_SCHEMA_VERSION = "yaqs.state_preparation.phase2.training_stage_config.v1"
TRAINING_STAGE_TEMPLATE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.training_stage_template.v1"
CHECKPOINT_VALIDATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.checkpoint_validation_config.v1"
TRAINING_PIPELINE_TEMPLATE_SCHEMA_VERSION = "yaqs.state_preparation.phase2.training_pipeline_template.v1"
PHASE2_TARGET_REF_SCHEMA_VERSION = "yaqs.state_preparation.phase2.target_ref.v1"
EXTERNAL_CHECKPOINT_REF_SCHEMA_VERSION = "yaqs.state_preparation.phase2.external_checkpoint_ref.v1"
TRAINING_PIPELINE_CONFIG_SCHEMA_VERSION = "yaqs.state_preparation.phase2.training_pipeline_config.v1"
TRAINING_STAGE_RESULT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.training_stage_result.v1"
TRAINING_PIPELINE_RESULT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.training_pipeline_result.v1"
PIPELINE_EVALUATION_CONFIG_SCHEMA_VERSION = "yaqs.state_preparation.phase2.pipeline_evaluation_config.v1"
PIPELINE_BENCHMARK_RESULT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.pipeline_benchmark_result.v1"
PIPELINE_TRAINING_IDENTITY_VERSION = "yaqs.state_preparation.phase2.training_identity.v1"
PIPELINE_EVALUATION_IDENTITY_VERSION = "yaqs.state_preparation.phase2.evaluation_identity.v1"
PIPELINE_PREFIX_IDENTITY_VERSION = "yaqs.state_preparation.phase2.pipeline_prefix_identity.v1"

TRAINING_ID_PREFIX = "phase2_training_"
EVALUATION_ROW_ID_PREFIX = "phase2_evaluation_"
PIPELINE_PREFIX = "phase2_pipeline_prefix_"
PHASE2_TARGET_ID_PREFIX = "phase2_target_"
MATERIALIZED_CIRCUIT_ID_PREFIX = "phase2_circuit_"

STAGE_KINDS = ("optimize", "grow", "prune")
PARAMETER_TRANSFER_RULES = (
    "initialize_zeros",
    "initialize_random_uniform",
    "initialize_random_normal",
    "load_checkpoint",
    "copy",
    "append_zeros",
    "append_random_uniform",
    "append_random_normal",
    "apply_pruning_mask",
)
PRUNING_RULES = (
    "none",
    "random",
    "magnitude",
    "impact_one_shot",
    "impact_iterative",
    "operator_selection",
)
TRAJECTORY_UPDATES = ("independent", "cross")
TRAINING_SAMPLING_POLICIES = ("none", "resampled", "crn_fixed", "crn_refresh")
CHECKPOINT_SELECTION_RULES = ("last_iteration", "best_validation_fidelity")
CHECKPOINT_TIE_BREAKERS = ("earliest_iteration", "latest_iteration")
TRUNCATION_MODES = ("discarded_weight", "relative")
TARGET_NAMESPACES = ("phase2", "phase1_fixture", "legacy_reproduction")
TARGET_SCOPE_IDS = (
    "primary_q6",
    "secondary_q12",
    "phase1_fixture",
    "legacy_reproduction",
)
LEGACY_REPRODUCTION_TARGET_IDS = tuple(f"legacy_tfim_seed_{seed}" for seed in (100, 200, 300, 400, 500))
PHASE1_FIXTURE_MANIFEST_CHECKSUM = "sha256:49948fe4e63f652169c603e5e03f32f8a66ad70daa25091ee7cdf83644287d11"
LEGACY_REPRODUCTION_MANIFEST_CHECKSUM = "sha256:a294080bf54a62b2bad0df85faa2f75ade5098b6a9afd84dc81fbb29bafdda1c"
LEGACY_LAYERWISE_SEED_BINDINGS = (
    "legacy_layerwise_depth1_initialization",
    "legacy_layerwise_depth2_initialization",
    "legacy_layerwise_depth3_initialization",
    "legacy_layerwise_depth4_initialization",
    "legacy_layerwise_growth_optimizer",
    "legacy_layerwise_final_optimizer",
    "legacy_layerwise_fixed_crn",
)
SEED_DOMAIN_ROLES = (
    "initialization",
    "optimizer_ordering",
    "training_trajectory",
    "checkpoint_validation",
    "pilot_evaluation",
    "screening_selection",
    "confirmatory_test",
)
EVALUATION_POLICIES = ("fixed_sample", "confidence_interval")
SIDECAR_STORAGE_POLICIES = ("none", "trajectory_fidelities")
FAILURE_PHASES = ("pipeline_loading", "materialization", "evaluation", "serialization")

DataRole = Literal[
    "development",
    "checkpoint_validation",
    "screening_selection",
    "secondary_benchmark",
    "confirmatory",
]


class _WP21PruningStageSpec(Protocol):
    """Fields consumed from the independently validated pruning-stage schema."""

    scoring_objective_kind: str
    removal_schedule: str
    removal_count: int | None
    removal_fraction: float | None
    relax_after_round: bool


_UINT64_MAX = 2**64 - 1
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_PHASE2_TARGET_PATTERN = re.compile(r"^phase2_target_[0-9a-f]{64}$")
_PHASE2_TARGET_MANIFEST_PATTERN = re.compile(r"^phase2_target_population_[0-9a-f]{64}$")
_TRAINING_ID_PATTERN = re.compile(r"^phase2_training_[0-9a-f]{64}$")
_EVALUATION_ROW_ID_PATTERN = re.compile(r"^phase2_evaluation_[0-9a-f]{64}$")
_PIPELINE_PREFIX_PATTERN = re.compile(r"^phase2_pipeline_prefix_[0-9a-f]{64}$")
_MATERIALIZED_CIRCUIT_ID_PATTERN = re.compile(r"^phase2_circuit_[0-9a-f]{64}$")

_METHOD_FAMILY_BY_METHOD_ID = {
    "phase1_noiseless_checkpoint_control": "fixed_depth_bmpd",
    "unpruned_deep_bmpd": "fixed_depth_bmpd",
    "layerwise_bmpd_crn_v2": "layerwise_bmpd",
    "layerwise_bmpd_noiseless": "layerwise_bmpd",
    "fixed_depth_bmpd_crn": "fixed_depth_bmpd",
    "layerwise_bmpd_resampled": "layerwise_bmpd",
    "layerwise_bmpd_cross_crn": "layerwise_bmpd",
    "parameter_shift_adam_layerwise": "parameter_shift_adam_layerwise",
    "parameter_shift_adam_fixed": "parameter_shift_adam_fixed",
    "spsa_layerwise": "spsa_layerwise",
    "spsa_fixed": "spsa_fixed",
    "adapt_style_state_preparation": "adapt_style_state_preparation",
    "impact_pruning_crn": "impact_pruning",
    "topdown_random": "topdown_pruning",
    "topdown_magnitude": "topdown_pruning",
    "topdown_impact_one_shot": "topdown_pruning",
    "topdown_impact_iterative": "topdown_pruning",
    "energy_adapt_vqe": "energy_adapt_vqe",
    "layerwise_bmpd_crn_legacy_v1": "layerwise_bmpd_legacy",
    "standard_vqa": "standard_vqa",
    "topdown_magnitude_pruning_legacy_v1": "topdown_magnitude_pruning_legacy",
    "krotov": "krotov",
}
_WP21_PRUNING_RULE_BY_METHOD_ID = {
    "topdown_random": "random",
    "topdown_magnitude": "magnitude",
    "topdown_impact_one_shot": "impact_one_shot",
    "topdown_impact_iterative": "impact_iterative",
}
_WP21_METHOD_ID_BY_PRUNING_RULE = {rule: method for method, rule in _WP21_PRUNING_RULE_BY_METHOD_ID.items()}
_PHASE1_FIXTURE_METADATA = {
    "gaussian_mu0p5_sigma0p1": ("gaussian_amplitude", "interior"),
    "tfim_ferro": ("tfim_ground_state", "ferromagnetic"),
    "tfim_critical": ("tfim_ground_state", "critical"),
    "tfim_para": ("tfim_ground_state", "paramagnetic"),
    "haar_random_1": ("haar_random", "dense_complex"),
    "haar_random_2": ("haar_random", "dense_complex"),
    "haar_random_3": ("haar_random", "dense_complex"),
    "random_mps_bond2": ("random_mps", "bond2"),
    "random_mps_bond3": ("random_mps", "bond3"),
}
_LEGACY_METHOD_IDS = frozenset({
    "layerwise_bmpd_crn_legacy_v1",
    "standard_vqa",
    "topdown_magnitude_pruning_legacy_v1",
})
_LEGACY_LAYERWISE_SEED_BINDING_SET = frozenset(LEGACY_LAYERWISE_SEED_BINDINGS)

_CHECKPOINT_VALIDATION_KEYS = frozenset({
    "schema_version",
    "noise_id",
    "noise_definition_version",
    "noise_strength_scale",
    "tjm_dt",
    "trajectory_count",
    "seed",
    "sampling_policy",
    "ensemble_refresh_interval",
    "cadence",
    "selection_rule",
    "tie_breaker",
})
_STAGE_CONFIG_KEYS = frozenset({
    "schema_version",
    "stage_index",
    "stage_id",
    "stage_kind",
    "input_topology_id",
    "output_topology_id",
    "input_parameter_count",
    "output_parameter_count",
    "parameter_transfer_rule",
    "initialization_seed",
    "optimizer_id",
    "optimizer_hyperparameters",
    "optimizer_seed",
    "iteration_budget",
    "training_noise_id",
    "noise_definition_version",
    "noise_strength_scale",
    "tjm_dt",
    "trajectory_count",
    "training_seed",
    "trajectory_update",
    "sampling_policy",
    "crn_refresh_interval",
    "checkpoint_validation",
    "pruning_rule",
    "pruning_threshold",
    "max_bond_dimension",
    "svd_threshold",
    "truncation_mode",
    "min_bond_dimension",
    "input_checkpoint_path",
    "input_checkpoint_checksum",
    "input_checkpoint_provenance_checksum",
    "input_checkpoint_pipeline_prefix",
    "input_checkpoint_ref_checksum",
    "configuration_checksum",
})
_STAGE_TEMPLATE_POLICY_KEYS = frozenset(
    _STAGE_CONFIG_KEYS
    - {
        "schema_version",
        "initialization_seed",
        "optimizer_seed",
        "training_seed",
        "checkpoint_validation",
        "input_checkpoint_path",
        "input_checkpoint_checksum",
        "input_checkpoint_provenance_checksum",
        "input_checkpoint_pipeline_prefix",
        "input_checkpoint_ref_checksum",
        "configuration_checksum",
    }
    | {"checkpoint_validation_policy"}
)
_STAGE_TEMPLATE_KEYS = frozenset({
    "schema_version",
    "stage_policy",
    "seed_bindings",
    "configuration_checksum",
})
_PIPELINE_TEMPLATE_KEYS = frozenset({
    "schema_version",
    "template_id",
    "preregistration_checksum",
    "target_scope_id",
    "ansatz_family",
    "method_family_id",
    "method_id",
    "method_version",
    "resource_stratum_id",
    "stages",
    "seed_domains",
    "final_materialization_policy",
    "matching_projection_checksum",
    "configuration_checksum",
})
_PIPELINE_CONFIG_KEYS = frozenset({
    "schema_version",
    "template",
    "template_checksum",
    "target_ref",
    "target_namespace",
    "target_instance_id",
    "target_population_manifest_checksum",
    "target_instance_spec_checksum",
    "target_family_id",
    "target_stratum_id",
    "qubit_count",
    "optimization_block_id",
    "optimization_seed",
    "stages",
    "data_role",
    "training_id",
    "matching_projection_checksum",
    "configuration_checksum",
})
_PHASE2_TARGET_REF_KEYS = frozenset({
    "schema_version",
    "target_manifest_checksum",
    "preregistration_checksum",
    "population_config_checksum",
    "data_role",
    "population_scope",
    "target_spec",
    "content_checksum",
})
_EXTERNAL_CHECKPOINT_REF_KEYS = frozenset({
    "schema_version",
    "producer_result",
    "producer_stage_index",
    "provenance_ref_checksum",
    "content_checksum",
})
_STAGE_RESULT_KEYS = frozenset({
    "schema_version",
    "pipeline_training_id",
    "pipeline_prefix_id",
    "stage_index",
    "stage_id",
    "stage_configuration_checksum",
    "input_checkpoint_checksum",
    "input_checkpoint_provenance_checksum",
    "produced_checkpoint_path",
    "produced_checkpoint_checksum",
    "checkpoint_provenance_checksum",
    "output_topology_id",
    "output_parameter_count",
    "training_summary",
    "checkpoint_validation_summary",
    "training_ensemble_checksums",
    "checkpoint_validation_ensemble_checksum",
    "optimizer_trace_path",
    "optimizer_trace_checksum",
    "diagnostic_sidecar_path",
    "diagnostic_sidecar_checksum",
    "wall_time_seconds",
    "peak_memory_bytes",
    "normalized_work",
    "content_checksum",
})
_PIPELINE_RESULT_KEYS = frozenset({
    "schema_version",
    "config",
    "stage_results",
    "final_checkpoint_path",
    "final_checkpoint_checksum",
    "final_checkpoint_provenance_checksum",
    "wall_time_seconds",
    "peak_memory_bytes",
    "normalized_work",
    "content_checksum",
})
_EVALUATION_CONFIG_KEYS = frozenset({
    "schema_version",
    "pipeline_training_id",
    "pipeline_configuration_checksum",
    "pipeline_result_checksum",
    "final_checkpoint_checksum",
    "final_materialization_policy_checksum",
    "data_role",
    "materialized_circuit_id",
    "materialized_circuit_checksum",
    "test_noise_id",
    "noise_definition_version",
    "noise_strength_scale",
    "tjm_dt",
    "evaluation_seed",
    "evaluation_seed_domain",
    "repetition",
    "trajectory_budget",
    "evaluation_policy",
    "confidence_level",
    "confidence_interval_method",
    "sidecar_storage_policy",
    "max_bond_dimension",
    "svd_threshold",
    "truncation_mode",
    "min_bond_dimension",
    "evaluation_row_id",
    "configuration_checksum",
})
_BENCHMARK_RESULT_KEYS = frozenset({
    "schema_version",
    "status",
    "evaluation_row_id",
    "config",
    "materialized_circuit_path",
    "test_noiseless_fidelity",
    "test_noisy_fidelity",
    "noisy_fidelity_standard_deviation",
    "noisy_fidelity_standard_error",
    "confidence_interval_lower",
    "confidence_interval_upper",
    "sampled_nonidentity_events",
    "trajectory_sidecar_path",
    "trajectory_sidecar_checksum",
    "evaluation_wall_time_seconds",
    "peak_memory_bytes",
    "normalized_work",
    "runtime_fingerprint_checksum",
    "content_checksum",
})
_BENCHMARK_FAILURE_KEYS = frozenset({
    "schema_version",
    "status",
    "evaluation_row_id",
    "config",
    "failure_phase",
    "exception_type",
    "message",
    "traceback",
    "retryable",
    "attempt",
    "materialized_circuit_path",
    "materialized_circuit_checksum",
    "wall_time_seconds",
    "runtime_fingerprint_checksum",
    "content_checksum",
})

PIPELINE_CSV_COLUMNS = (
    "schema_version",
    "status",
    "evaluation_row_id",
    "pipeline_training_id",
    "data_role",
    "materialized_circuit_id",
    "test_noise_id",
    "evaluation_seed",
    "repetition",
    "trajectory_budget",
    "test_noiseless_fidelity",
    "test_noisy_fidelity",
    "noisy_fidelity_standard_deviation",
    "noisy_fidelity_standard_error",
    "confidence_interval_lower",
    "confidence_interval_upper",
    "sampled_nonidentity_events",
    "evaluation_wall_time_seconds",
    "peak_memory_bytes",
    "normalized_work",
    "runtime_fingerprint_checksum",
    "materialized_circuit_path",
    "trajectory_sidecar_path",
    "trajectory_sidecar_checksum",
    "failure_phase",
    "exception_type",
    "message",
    "traceback",
    "retryable",
    "attempt",
    "config",
)
_CSV_INTEGER_COLUMNS = frozenset({
    "evaluation_seed",
    "repetition",
    "trajectory_budget",
    "sampled_nonidentity_events",
    "peak_memory_bytes",
    "attempt",
})
_CSV_FLOAT_COLUMNS = frozenset({
    "test_noiseless_fidelity",
    "test_noisy_fidelity",
    "noisy_fidelity_standard_deviation",
    "noisy_fidelity_standard_error",
    "confidence_interval_lower",
    "confidence_interval_upper",
    "evaluation_wall_time_seconds",
})
_CSV_BOOLEAN_COLUMNS = frozenset({"retryable"})
_CSV_JSON_COLUMNS = frozenset({"normalized_work", "config"})
_CSV_COMMON_COLUMNS = frozenset({
    "schema_version",
    "status",
    "evaluation_row_id",
    "pipeline_training_id",
    "data_role",
    "materialized_circuit_id",
    "test_noise_id",
    "evaluation_seed",
    "repetition",
    "trajectory_budget",
    "evaluation_wall_time_seconds",
    "runtime_fingerprint_checksum",
    "materialized_circuit_path",
    "config",
})
_CSV_SUCCESS_COLUMNS = _CSV_COMMON_COLUMNS | frozenset({
    "test_noiseless_fidelity",
    "test_noisy_fidelity",
    "noisy_fidelity_standard_deviation",
    "noisy_fidelity_standard_error",
    "confidence_interval_lower",
    "confidence_interval_upper",
    "sampled_nonidentity_events",
    "peak_memory_bytes",
    "normalized_work",
    "trajectory_sidecar_path",
    "trajectory_sidecar_checksum",
})
_CSV_FAILURE_COLUMNS = _CSV_COMMON_COLUMNS | frozenset({
    "failure_phase",
    "exception_type",
    "message",
    "traceback",
    "retryable",
    "attempt",
})


def _raise_type_error(name: str, expected: str, value: object) -> NoReturn:
    """Raise a consistently formatted strict type error.

    Raises:
        TypeError: Always.
    """
    msg = f"{name} must be {expected}; received {type(value).__name__}."
    raise TypeError(msg)


def _require_optional_slug(value: object, name: str) -> str | None:
    """Validate an optional stable identifier."""
    return None if value is None else require_slug(value, name)


def _require_optional_checksum(value: object, name: str) -> str | None:
    """Validate an optional prefixed SHA-256 digest."""
    return None if value is None else require_checksum(value, name)


def _require_optional_path(value: object, name: str) -> str | None:
    """Validate an optional normalized relative POSIX path."""
    return None if value is None else require_relative_path(value, name)


def _require_seed(value: object, name: str, *, allow_none: bool = True) -> int | None:
    """Validate an optional unsigned 64-bit seed."""
    if value is None and allow_none:
        return None
    seed = require_int(value, name, minimum=0)
    if seed > _UINT64_MAX:
        msg = f"{name} must be at most {_UINT64_MAX}."
        raise ValueError(msg)
    return seed


def _require_optional_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Validate an optional strict finite float."""
    return None if value is None else require_float(value, name, minimum=minimum, maximum=maximum)


def _require_identifier(value: object, name: str, pattern: re.Pattern[str]) -> str:
    """Validate one prefixed stable content identifier."""
    text = require_string(value, name)
    if pattern.fullmatch(text) is None:
        msg = f"{name} has an invalid stable identifier."
        raise ValueError(msg)
    return text


def _validate_pair(
    path: object,
    checksum: object,
    *,
    path_name: str,
    checksum_name: str,
    required: bool = False,
) -> tuple[str | None, str | None]:
    """Validate an optional or required relative-path/checksum pair."""
    normalized_path = _require_optional_path(path, path_name)
    normalized_checksum = _require_optional_checksum(checksum, checksum_name)
    if (normalized_path is None) != (normalized_checksum is None):
        msg = f"{path_name} and {checksum_name} must either both be present or both be absent."
        raise ValueError(msg)
    if required and normalized_path is None:
        msg = f"{path_name} and {checksum_name} are required."
        raise ValueError(msg)
    return normalized_path, normalized_checksum


def _validate_truncation(
    max_bond_dimension: object,
    svd_threshold: object,
    truncation_mode: object,
    min_bond_dimension: object,
) -> tuple[int | None, float, str, int]:
    """Validate one resolved tensor-network truncation policy."""
    maximum = (
        None
        if max_bond_dimension is None
        else require_int(
            max_bond_dimension,
            "max_bond_dimension",
            minimum=1,
        )
    )
    threshold = require_float(svd_threshold, "svd_threshold", minimum=0.0)
    mode = require_string(truncation_mode, "truncation_mode")
    if mode not in TRUNCATION_MODES:
        msg = f"truncation_mode must be one of {TRUNCATION_MODES!r}."
        raise ValueError(msg)
    minimum = require_int(min_bond_dimension, "min_bond_dimension", minimum=1)
    if maximum is not None and minimum > maximum:
        msg = "min_bond_dimension cannot exceed max_bond_dimension."
        raise ValueError(msg)
    return maximum, threshold, mode, minimum


def _validate_noise(
    *,
    noise_id: object,
    definition_version: object,
    strength_scale: object,
    tjm_dt: object,
    name: str,
    allow_ballarin: bool,
) -> tuple[str, str, float | None, float | None]:
    """Validate a fully resolved noiseless or noisy condition."""
    normalized_id = require_slug(noise_id, f"{name}.noise_id")
    normalized_version = require_slug(definition_version, f"{name}.noise_definition_version")
    scale = _require_optional_float(strength_scale, f"{name}.noise_strength_scale", minimum=0.0)
    dt = _require_optional_float(tjm_dt, f"{name}.tjm_dt", minimum=0.0)
    if normalized_id == NOISELESS_NOISE_ID:
        if scale is not None or dt is not None:
            msg = f"{name} noiseless condition cannot specify noise strength or TJM dt."
            raise ValueError(msg)
    elif normalized_id == BALLARIN_NOISE_ID:
        if not allow_ballarin:
            msg = "ballarin_coupled is evaluation-only and cannot be used during training or checkpoint selection."
            raise ValueError(msg)
        if scale is not None or dt is not None:
            msg = "ballarin_coupled does not use a strength scale or TJM dt."
            raise ValueError(msg)
    elif scale is None or scale <= 0.0 or dt is None or dt <= 0.0:
        msg = f"{name} noisy condition requires an explicit positive strength scale and TJM dt."
        raise ValueError(msg)
    return normalized_id, normalized_version, scale, dt


def _validate_summary(value: object, name: str) -> Mapping[str, object]:
    """Freeze a nonempty, finite JSON summary."""
    summary = freeze_json_mapping(value, name)
    if not summary:
        msg = f"{name} must not be empty."
        raise ValueError(msg)
    return summary


def _validate_normalized_work(value: object, name: str) -> Mapping[str, object]:
    """Validate strict normalized computational-work counters."""
    work = freeze_json_mapping(value, name)
    expected = frozenset({
        "objective_evaluations",
        "gradient_evaluations",
        "training_trajectories",
        "checkpoint_validation_trajectories",
        "test_trajectories",
        "trajectory_gate_applications",
    })
    require_exact_keys(work, expected, name)
    for key in expected:
        require_int(work[key], f"{name}.{key}", minimum=0)
    return work


def _sum_work(items: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Return a frozen component-wise sum of work ledgers."""
    if not items:
        return _validate_normalized_work(
            {
                "objective_evaluations": 0,
                "gradient_evaluations": 0,
                "training_trajectories": 0,
                "checkpoint_validation_trajectories": 0,
                "test_trajectories": 0,
                "trajectory_gate_applications": 0,
            },
            "normalized_work",
        )
    totals = dict.fromkeys(items[0], 0)
    for item in items:
        for key in totals:
            totals[key] += cast("int", item[key])
    return _validate_normalized_work(totals, "normalized_work")


@dataclass(frozen=True, slots=True)
class CheckpointValidationConfig:
    """Identity-bearing policy used to select a stage checkpoint."""

    noise_id: str
    noise_definition_version: str
    noise_strength_scale: float | None
    tjm_dt: float | None
    trajectory_count: int
    seed: int | None
    sampling_policy: Literal["none", "resampled", "crn_fixed", "crn_refresh"]
    ensemble_refresh_interval: int | None
    cadence: int | None
    selection_rule: Literal["last_iteration", "best_validation_fidelity"]
    tie_breaker: Literal["earliest_iteration", "latest_iteration"]
    schema_version: str = field(default=CHECKPOINT_VALIDATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate noise, sampling, cadence, and selection together."""
        noise_id, version, scale, dt = _validate_noise(
            noise_id=self.noise_id,
            definition_version=self.noise_definition_version,
            strength_scale=self.noise_strength_scale,
            tjm_dt=self.tjm_dt,
            name="checkpoint_validation",
            allow_ballarin=False,
        )
        object.__setattr__(self, "noise_id", noise_id)
        object.__setattr__(self, "noise_definition_version", version)
        object.__setattr__(self, "noise_strength_scale", scale)
        object.__setattr__(self, "tjm_dt", dt)
        count = require_int(self.trajectory_count, "checkpoint_validation.trajectory_count", minimum=0)
        seed = _require_seed(self.seed, "checkpoint_validation.seed")
        if self.sampling_policy not in TRAINING_SAMPLING_POLICIES:
            msg = f"checkpoint_validation.sampling_policy must be one of {TRAINING_SAMPLING_POLICIES!r}."
            raise ValueError(msg)
        refresh = (
            None
            if self.ensemble_refresh_interval is None
            else require_int(
                self.ensemble_refresh_interval,
                "checkpoint_validation.ensemble_refresh_interval",
                minimum=1,
            )
        )
        cadence = (
            None if self.cadence is None else require_int(self.cadence, "checkpoint_validation.cadence", minimum=1)
        )
        if self.selection_rule not in CHECKPOINT_SELECTION_RULES:
            msg = f"checkpoint_validation.selection_rule must be one of {CHECKPOINT_SELECTION_RULES!r}."
            raise ValueError(msg)
        if self.tie_breaker not in CHECKPOINT_TIE_BREAKERS:
            msg = f"checkpoint_validation.tie_breaker must be one of {CHECKPOINT_TIE_BREAKERS!r}."
            raise ValueError(msg)

        disabled = count == 0
        if disabled and (
            noise_id != NOISELESS_NOISE_ID
            or seed is not None
            or self.sampling_policy != "none"
            or refresh is not None
            or cadence is not None
            or self.selection_rule != "last_iteration"
        ):
            msg = "Disabled checkpoint validation requires the canonical noiseless/none/last configuration."
            raise ValueError(msg)
        if not disabled and (
            noise_id == NOISELESS_NOISE_ID
            or seed is None
            or self.sampling_policy == "none"
            or cadence is None
            or self.selection_rule != "best_validation_fidelity"
        ):
            msg = (
                "Enabled checkpoint validation requires noisy sampling, a seed, cadence, and best-validation selection."
            )
            raise ValueError(msg)
        if self.sampling_policy == "crn_refresh" and refresh is None:
            msg = "crn_refresh checkpoint validation requires a positive ensemble_refresh_interval."
            raise ValueError(msg)
        if self.sampling_policy != "crn_refresh" and refresh is not None:
            msg = "ensemble_refresh_interval is valid only for crn_refresh checkpoint validation."
            raise ValueError(msg)
        object.__setattr__(self, "trajectory_count", count)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "ensemble_refresh_interval", refresh)
        object.__setattr__(self, "cadence", cadence)

    @classmethod
    def disabled(cls) -> CheckpointValidationConfig:
        """Return the one canonical disabled validation policy."""
        return cls(
            noise_id=NOISELESS_NOISE_ID,
            noise_definition_version="yaqs.state_preparation.noise.v1",
            noise_strength_scale=None,
            tjm_dt=None,
            trajectory_count=0,
            seed=None,
            sampling_policy="none",
            ensemble_refresh_interval=None,
            cadence=None,
            selection_rule="last_iteration",
            tie_breaker="earliest_iteration",
        )

    @property
    def enabled(self) -> bool:
        """Whether checkpoint outcomes can select the produced artifact."""
        return self.trajectory_count > 0

    def to_dict(self) -> dict[str, object]:
        """Return a detached exact-schema representation."""
        return {
            "schema_version": self.schema_version,
            "noise_id": self.noise_id,
            "noise_definition_version": self.noise_definition_version,
            "noise_strength_scale": self.noise_strength_scale,
            "tjm_dt": self.tjm_dt,
            "trajectory_count": self.trajectory_count,
            "seed": self.seed,
            "sampling_policy": self.sampling_policy,
            "ensemble_refresh_interval": self.ensemble_refresh_interval,
            "cadence": self.cadence,
            "selection_rule": self.selection_rule,
            "tie_breaker": self.tie_breaker,
        }

    @classmethod
    def from_dict(cls, data: object) -> CheckpointValidationConfig:
        """Construct a strict validation policy from JSON-native data."""
        mapping = require_mapping(data, "checkpoint validation config")
        require_exact_keys(mapping, _CHECKPOINT_VALIDATION_KEYS, "checkpoint validation config")
        if mapping["schema_version"] != CHECKPOINT_VALIDATION_SCHEMA_VERSION:
            msg = f"schema_version must be {CHECKPOINT_VALIDATION_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        return cls(
            noise_id=cast("str", mapping["noise_id"]),
            noise_definition_version=cast("str", mapping["noise_definition_version"]),
            noise_strength_scale=cast("float | None", mapping["noise_strength_scale"]),
            tjm_dt=cast("float | None", mapping["tjm_dt"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            seed=cast("int | None", mapping["seed"]),
            sampling_policy=cast(
                "Literal['none', 'resampled', 'crn_fixed', 'crn_refresh']",
                mapping["sampling_policy"],
            ),
            ensemble_refresh_interval=cast("int | None", mapping["ensemble_refresh_interval"]),
            cadence=cast("int | None", mapping["cadence"]),
            selection_rule=cast(
                "Literal['last_iteration', 'best_validation_fidelity']",
                mapping["selection_rule"],
            ),
            tie_breaker=cast(
                "Literal['earliest_iteration', 'latest_iteration']",
                mapping["tie_breaker"],
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainingStageConfig:
    """One fully resolved, versioned training, growth, or pruning stage."""

    stage_index: int
    stage_id: str
    stage_kind: Literal["optimize", "grow", "prune"]
    input_topology_id: str | None
    output_topology_id: str
    input_parameter_count: int
    output_parameter_count: int
    parameter_transfer_rule: str
    initialization_seed: int | None
    optimizer_id: str
    optimizer_hyperparameters: Mapping[str, object]
    optimizer_seed: int | None
    iteration_budget: int
    training_noise_id: str
    noise_definition_version: str
    noise_strength_scale: float | None
    tjm_dt: float | None
    trajectory_count: int
    training_seed: int | None
    trajectory_update: Literal["independent", "cross"] | None
    sampling_policy: Literal["none", "resampled", "crn_fixed", "crn_refresh"]
    crn_refresh_interval: int | None
    checkpoint_validation: CheckpointValidationConfig
    pruning_rule: Literal[
        "none",
        "random",
        "magnitude",
        "impact_one_shot",
        "impact_iterative",
        "operator_selection",
    ]
    pruning_threshold: float | None
    max_bond_dimension: int | None
    svd_threshold: float
    truncation_mode: Literal["discarded_weight", "relative"]
    min_bond_dimension: int
    input_checkpoint_path: str | None = None
    input_checkpoint_checksum: str | None = None
    input_checkpoint_provenance_checksum: str | None = None
    input_checkpoint_pipeline_prefix: str | None = None
    input_checkpoint_ref_checksum: str | None = None
    schema_version: str = field(default=TRAINING_STAGE_CONFIG_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate stage-local scientific and artifact constraints."""
        object.__setattr__(self, "stage_index", require_int(self.stage_index, "stage_index", minimum=0))
        object.__setattr__(self, "stage_id", require_slug(self.stage_id, "stage_id"))
        if self.stage_kind not in STAGE_KINDS:
            msg = f"stage_kind must be one of {STAGE_KINDS!r}."
            raise ValueError(msg)
        input_topology = _require_optional_slug(self.input_topology_id, "input_topology_id")
        output_topology = require_slug(self.output_topology_id, "output_topology_id")
        input_count = require_int(self.input_parameter_count, "input_parameter_count", minimum=0)
        output_count = require_int(self.output_parameter_count, "output_parameter_count", minimum=1)
        if input_topology is None and input_count != 0:
            msg = "A missing input topology requires zero input parameters."
            raise ValueError(msg)
        if input_topology is not None and input_count == 0:
            msg = "An input topology requires at least one input parameter."
            raise ValueError(msg)
        transfer = require_string(self.parameter_transfer_rule, "parameter_transfer_rule")
        if transfer not in PARAMETER_TRANSFER_RULES:
            msg = f"parameter_transfer_rule must be one of {PARAMETER_TRANSFER_RULES!r}."
            raise ValueError(msg)
        initialization_seed = _require_seed(self.initialization_seed, "initialization_seed")
        random_transfer = transfer in {
            "initialize_random_uniform",
            "initialize_random_normal",
            "append_random_uniform",
            "append_random_normal",
        }
        if random_transfer != (initialization_seed is not None):
            msg = "An initialization seed is required exactly for random parameter-transfer rules."
            raise ValueError(msg)

        optimizer_id = require_slug(self.optimizer_id, "optimizer_id")
        hyperparameters = freeze_json_mapping(self.optimizer_hyperparameters, "optimizer_hyperparameters")
        optimizer_seed = _require_seed(self.optimizer_seed, "optimizer_seed")
        iteration_budget = require_int(self.iteration_budget, "iteration_budget", minimum=0)
        random_pruning = self.stage_kind == "prune" and self.pruning_rule == "random"
        pruning_stage_spec: _WP21PruningStageSpec | None = None
        if optimizer_id == "none":
            if iteration_budget != 0 or random_pruning != (optimizer_seed is not None):
                msg = (
                    "optimizer_id 'none' requires zero iterations; "
                    "an optimizer-ordering seed is present exactly for random pruning."
                )
                raise ValueError(msg)
            if hyperparameters:
                if self.stage_kind != "prune":
                    msg = "Only a WP21 pruning transform may attach policy data to optimizer_id 'none'."
                    raise ValueError(msg)
                policy_method = _WP21_METHOD_ID_BY_PRUNING_RULE.get(self.pruning_rule)
                if policy_method is None:
                    msg = "A WP21 pruning policy requires one of the four registered top-down pruning rules."
                    raise ValueError(msg)
                pruning_stage_spec = _validate_wp21_pruning_stage_policy(
                    hyperparameters,
                    method_id=policy_method,
                    score_rule=self.pruning_rule,
                    random_seed=optimizer_seed,
                )
        elif iteration_budget == 0 or optimizer_seed is None:
            msg = "An active optimizer requires a positive iteration budget and optimizer seed."
            raise ValueError(msg)

        noise_id, noise_version, scale, dt = _validate_noise(
            noise_id=self.training_noise_id,
            definition_version=self.noise_definition_version,
            strength_scale=self.noise_strength_scale,
            tjm_dt=self.tjm_dt,
            name="training",
            allow_ballarin=False,
        )
        trajectory_count = require_int(self.trajectory_count, "trajectory_count", minimum=0)
        training_seed = _require_seed(self.training_seed, "training_seed")
        if self.trajectory_update is not None and self.trajectory_update not in TRAJECTORY_UPDATES:
            msg = f"trajectory_update must be one of {TRAJECTORY_UPDATES!r} or None."
            raise ValueError(msg)
        if self.sampling_policy not in TRAINING_SAMPLING_POLICIES:
            msg = f"sampling_policy must be one of {TRAINING_SAMPLING_POLICIES!r}."
            raise ValueError(msg)
        refresh = (
            None
            if self.crn_refresh_interval is None
            else require_int(self.crn_refresh_interval, "crn_refresh_interval", minimum=1)
        )
        noiseless = noise_id == NOISELESS_NOISE_ID
        if noiseless and (
            trajectory_count != 0
            or self.trajectory_update is not None
            or self.sampling_policy != "none"
            or refresh is not None
            or training_seed is not None
        ):
            msg = "Noiseless training requires zero trajectories, no update, no sampling, no refresh, and no seed."
            raise ValueError(msg)
        if not noiseless and (
            trajectory_count == 0
            or self.trajectory_update is None
            or self.sampling_policy == "none"
            or training_seed is None
        ):
            msg = "Noisy training requires positive trajectories, an update, sampling policy, and training seed."
            raise ValueError(msg)
        if self.sampling_policy == "crn_refresh":
            if refresh is None:
                msg = "crn_refresh training requires a positive crn_refresh_interval."
                raise ValueError(msg)
            if iteration_budget and refresh > iteration_budget:
                msg = "crn_refresh_interval cannot exceed the iteration budget."
                raise ValueError(msg)
        elif refresh is not None:
            msg = "crn_refresh_interval is valid only with crn_refresh sampling."
            raise ValueError(msg)

        if not isinstance(self.checkpoint_validation, CheckpointValidationConfig):
            _raise_type_error(
                "checkpoint_validation",
                "a CheckpointValidationConfig",
                self.checkpoint_validation,
            )
        if pruning_stage_spec is not None and self.checkpoint_validation.enabled:
            msg = "A WP21 pruning transform cannot perform checkpoint validation."
            raise ValueError(msg)
        if self.checkpoint_validation.cadence is not None and self.checkpoint_validation.cadence > iteration_budget:
            msg = "Checkpoint-validation cadence cannot exceed the stage iteration budget."
            raise ValueError(msg)

        if self.pruning_rule not in PRUNING_RULES:
            msg = f"pruning_rule must be one of {PRUNING_RULES!r}."
            raise ValueError(msg)
        pruning_threshold = _require_optional_float(
            self.pruning_threshold,
            "pruning_threshold",
            minimum=0.0,
        )
        if self.stage_kind == "prune":
            if (
                self.pruning_rule == "none"
                or pruning_threshold is None
                or transfer != "apply_pruning_mask"
                or input_topology is None
                or output_count >= input_count
                or output_topology == input_topology
            ):
                msg = (
                    "Pruning stages require an explicit rule/threshold, pruning transfer, "
                    "and a smaller distinct topology."
                )
                raise ValueError(msg)
            if pruning_stage_spec is not None:
                expected_threshold = (
                    float(cast("int", pruning_stage_spec.removal_count))
                    if pruning_stage_spec.removal_schedule == "fixed_count"
                    else pruning_stage_spec.removal_fraction
                )
                if pruning_threshold != expected_threshold:
                    msg = "pruning_threshold must exactly match the WP21 removal schedule."
                    raise ValueError(msg)
                fixed_map_score = pruning_stage_spec.scoring_objective_kind == "fixed_map_sample_average_fidelity"
                if fixed_map_score and (
                    noiseless
                    or self.sampling_policy != "crn_fixed"
                    or self.trajectory_update != "independent"
                    or training_seed is None
                ):
                    msg = "Fixed-map impact scoring requires noisy independent CRN-fixed sampling and a training seed."
                    raise ValueError(msg)
                if not fixed_map_score and not noiseless:
                    msg = "Only fixed-map impact scoring may activate pruning-stage noise or a training seed."
                    raise ValueError(msg)
        elif self.pruning_rule != "none" or pruning_threshold is not None:
            msg = "Non-pruning stages cannot specify a pruning rule or threshold."
            raise ValueError(msg)
        if self.stage_kind == "grow" and (
            transfer not in {"append_zeros", "append_random_uniform", "append_random_normal"}
            or input_topology is None
            or output_count <= input_count
            or output_topology == input_topology
        ):
            msg = "Growth stages require append transfer and a larger distinct topology."
            raise ValueError(msg)
        if (
            self.stage_kind == "optimize"
            and input_topology is not None
            and (
                transfer not in {"copy", "load_checkpoint"}
                or output_count != input_count
                or output_topology != input_topology
            )
        ):
            msg = "Optimize stages with an input topology must preserve topology and parameters."
            raise ValueError(msg)
        if (
            self.stage_kind == "optimize"
            and input_topology is None
            and transfer
            not in {
                "initialize_zeros",
                "initialize_random_uniform",
                "initialize_random_normal",
            }
        ):
            msg = "An initial optimize stage requires an initialization transfer."
            raise ValueError(msg)

        maximum, threshold, mode, minimum = _validate_truncation(
            self.max_bond_dimension,
            self.svd_threshold,
            self.truncation_mode,
            self.min_bond_dimension,
        )
        checkpoint_path, checkpoint_checksum = _validate_pair(
            self.input_checkpoint_path,
            self.input_checkpoint_checksum,
            path_name="input_checkpoint_path",
            checksum_name="input_checkpoint_checksum",
        )
        checkpoint_prefix = (
            None
            if self.input_checkpoint_pipeline_prefix is None
            else _require_identifier(
                self.input_checkpoint_pipeline_prefix,
                "input_checkpoint_pipeline_prefix",
                _PIPELINE_PREFIX_PATTERN,
            )
        )
        checkpoint_provenance = _require_optional_checksum(
            self.input_checkpoint_provenance_checksum,
            "input_checkpoint_provenance_checksum",
        )
        checkpoint_ref_checksum = _require_optional_checksum(
            self.input_checkpoint_ref_checksum,
            "input_checkpoint_ref_checksum",
        )
        supplied = (
            checkpoint_path is not None,
            checkpoint_prefix is not None,
            checkpoint_provenance is not None,
            checkpoint_ref_checksum is not None,
        )
        if len(set(supplied)) != 1:
            msg = (
                "An input checkpoint path, checksum, prefix, provenance, and typed reference must be supplied together."
            )
            raise ValueError(msg)
        if (transfer == "load_checkpoint") != (checkpoint_ref_checksum is not None):
            msg = "load_checkpoint transfer is required exactly when an external input checkpoint is supplied."
            raise ValueError(msg)

        object.__setattr__(self, "input_topology_id", input_topology)
        object.__setattr__(self, "output_topology_id", output_topology)
        object.__setattr__(self, "input_parameter_count", input_count)
        object.__setattr__(self, "output_parameter_count", output_count)
        object.__setattr__(self, "parameter_transfer_rule", transfer)
        object.__setattr__(self, "initialization_seed", initialization_seed)
        object.__setattr__(self, "optimizer_id", optimizer_id)
        object.__setattr__(self, "optimizer_hyperparameters", hyperparameters)
        object.__setattr__(self, "optimizer_seed", optimizer_seed)
        object.__setattr__(self, "iteration_budget", iteration_budget)
        object.__setattr__(self, "training_noise_id", noise_id)
        object.__setattr__(self, "noise_definition_version", noise_version)
        object.__setattr__(self, "noise_strength_scale", scale)
        object.__setattr__(self, "tjm_dt", dt)
        object.__setattr__(self, "trajectory_count", trajectory_count)
        object.__setattr__(self, "training_seed", training_seed)
        object.__setattr__(self, "crn_refresh_interval", refresh)
        object.__setattr__(self, "pruning_threshold", pruning_threshold)
        object.__setattr__(self, "max_bond_dimension", maximum)
        object.__setattr__(self, "svd_threshold", threshold)
        object.__setattr__(self, "truncation_mode", mode)
        object.__setattr__(self, "min_bond_dimension", minimum)
        object.__setattr__(self, "input_checkpoint_path", checkpoint_path)
        object.__setattr__(self, "input_checkpoint_checksum", checkpoint_checksum)
        object.__setattr__(self, "input_checkpoint_provenance_checksum", checkpoint_provenance)
        object.__setattr__(self, "input_checkpoint_pipeline_prefix", checkpoint_prefix)
        object.__setattr__(self, "input_checkpoint_ref_checksum", checkpoint_ref_checksum)

    def identity_payload(self) -> dict[str, object]:
        """Return stage identity fields, excluding path spelling."""
        optimizer_hyperparameters = thaw_json_mapping(self.optimizer_hyperparameters)
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "stage_index": self.stage_index,
            "stage_id": self.stage_id,
            "stage_kind": self.stage_kind,
            "input_topology_id": self.input_topology_id,
            "output_topology_id": self.output_topology_id,
            "input_parameter_count": self.input_parameter_count,
            "output_parameter_count": self.output_parameter_count,
            "parameter_transfer_rule": self.parameter_transfer_rule,
            "initialization_seed": self.initialization_seed,
            "optimizer_id": self.optimizer_id,
            "optimizer_hyperparameters": optimizer_hyperparameters,
            "optimizer_seed": self.optimizer_seed,
            "iteration_budget": self.iteration_budget,
            "checkpoint_validation": self.checkpoint_validation.to_dict(),
            "pruning_rule": self.pruning_rule,
            "pruning_threshold": self.pruning_threshold,
            "max_bond_dimension": self.max_bond_dimension,
            "svd_threshold": self.svd_threshold,
            "truncation_mode": self.truncation_mode,
            "min_bond_dimension": self.min_bond_dimension,
            "input_checkpoint_checksum": self.input_checkpoint_checksum,
            "input_checkpoint_provenance_checksum": self.input_checkpoint_provenance_checksum,
            "input_checkpoint_pipeline_prefix": self.input_checkpoint_pipeline_prefix,
            "input_checkpoint_ref_checksum": self.input_checkpoint_ref_checksum,
        }
        payload.update({
            "training_noise_id": self.training_noise_id,
            "noise_definition_version": self.noise_definition_version,
            "noise_strength_scale": self.noise_strength_scale,
            "tjm_dt": self.tjm_dt,
            "trajectory_count": self.trajectory_count,
            "training_seed": self.training_seed,
            "trajectory_update": self.trajectory_update,
            "sampling_policy": self.sampling_policy,
            "crn_refresh_interval": self.crn_refresh_interval,
        })
        return payload

    @property
    def configuration_checksum(self) -> str:
        """Stable stage identity, excluding input checkpoint path spelling."""
        return canonical_checksum(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached exact-schema stage configuration."""
        return {
            "schema_version": self.schema_version,
            "stage_index": self.stage_index,
            "stage_id": self.stage_id,
            "stage_kind": self.stage_kind,
            "input_topology_id": self.input_topology_id,
            "output_topology_id": self.output_topology_id,
            "input_parameter_count": self.input_parameter_count,
            "output_parameter_count": self.output_parameter_count,
            "parameter_transfer_rule": self.parameter_transfer_rule,
            "initialization_seed": self.initialization_seed,
            "optimizer_id": self.optimizer_id,
            "optimizer_hyperparameters": thaw_json_mapping(self.optimizer_hyperparameters),
            "optimizer_seed": self.optimizer_seed,
            "iteration_budget": self.iteration_budget,
            "training_noise_id": self.training_noise_id,
            "noise_definition_version": self.noise_definition_version,
            "noise_strength_scale": self.noise_strength_scale,
            "tjm_dt": self.tjm_dt,
            "trajectory_count": self.trajectory_count,
            "training_seed": self.training_seed,
            "trajectory_update": self.trajectory_update,
            "sampling_policy": self.sampling_policy,
            "crn_refresh_interval": self.crn_refresh_interval,
            "checkpoint_validation": self.checkpoint_validation.to_dict(),
            "pruning_rule": self.pruning_rule,
            "pruning_threshold": self.pruning_threshold,
            "max_bond_dimension": self.max_bond_dimension,
            "svd_threshold": self.svd_threshold,
            "truncation_mode": self.truncation_mode,
            "min_bond_dimension": self.min_bond_dimension,
            "input_checkpoint_path": self.input_checkpoint_path,
            "input_checkpoint_checksum": self.input_checkpoint_checksum,
            "input_checkpoint_provenance_checksum": self.input_checkpoint_provenance_checksum,
            "input_checkpoint_pipeline_prefix": self.input_checkpoint_pipeline_prefix,
            "input_checkpoint_ref_checksum": self.input_checkpoint_ref_checksum,
            "configuration_checksum": self.configuration_checksum,
        }

    @classmethod
    def from_dict(cls, data: object) -> TrainingStageConfig:
        """Construct and identity-verify a strict stage configuration."""
        mapping = require_mapping(data, "training stage config")
        require_exact_keys(mapping, _STAGE_CONFIG_KEYS, "training stage config")
        if mapping["schema_version"] != TRAINING_STAGE_CONFIG_SCHEMA_VERSION:
            msg = f"schema_version must be {TRAINING_STAGE_CONFIG_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        stage = cls(
            stage_index=cast("int", mapping["stage_index"]),
            stage_id=cast("str", mapping["stage_id"]),
            stage_kind=cast("Literal['optimize', 'grow', 'prune']", mapping["stage_kind"]),
            input_topology_id=cast("str | None", mapping["input_topology_id"]),
            output_topology_id=cast("str", mapping["output_topology_id"]),
            input_parameter_count=cast("int", mapping["input_parameter_count"]),
            output_parameter_count=cast("int", mapping["output_parameter_count"]),
            parameter_transfer_rule=cast("str", mapping["parameter_transfer_rule"]),
            initialization_seed=cast("int | None", mapping["initialization_seed"]),
            optimizer_id=cast("str", mapping["optimizer_id"]),
            optimizer_hyperparameters=cast("Mapping[str, object]", mapping["optimizer_hyperparameters"]),
            optimizer_seed=cast("int | None", mapping["optimizer_seed"]),
            iteration_budget=cast("int", mapping["iteration_budget"]),
            training_noise_id=cast("str", mapping["training_noise_id"]),
            noise_definition_version=cast("str", mapping["noise_definition_version"]),
            noise_strength_scale=cast("float | None", mapping["noise_strength_scale"]),
            tjm_dt=cast("float | None", mapping["tjm_dt"]),
            trajectory_count=cast("int", mapping["trajectory_count"]),
            training_seed=cast("int | None", mapping["training_seed"]),
            trajectory_update=cast("Literal['independent', 'cross'] | None", mapping["trajectory_update"]),
            sampling_policy=cast(
                "Literal['none', 'resampled', 'crn_fixed', 'crn_refresh']",
                mapping["sampling_policy"],
            ),
            crn_refresh_interval=cast("int | None", mapping["crn_refresh_interval"]),
            checkpoint_validation=CheckpointValidationConfig.from_dict(mapping["checkpoint_validation"]),
            pruning_rule=cast(
                "Literal['none', 'random', 'magnitude', 'impact_one_shot', 'impact_iterative', 'operator_selection']",
                mapping["pruning_rule"],
            ),
            pruning_threshold=cast("float | None", mapping["pruning_threshold"]),
            max_bond_dimension=cast("int | None", mapping["max_bond_dimension"]),
            svd_threshold=cast("float", mapping["svd_threshold"]),
            truncation_mode=cast("Literal['discarded_weight', 'relative']", mapping["truncation_mode"]),
            min_bond_dimension=cast("int", mapping["min_bond_dimension"]),
            input_checkpoint_path=cast("str | None", mapping["input_checkpoint_path"]),
            input_checkpoint_checksum=cast("str | None", mapping["input_checkpoint_checksum"]),
            input_checkpoint_provenance_checksum=cast("str | None", mapping["input_checkpoint_provenance_checksum"]),
            input_checkpoint_pipeline_prefix=cast("str | None", mapping["input_checkpoint_pipeline_prefix"]),
            input_checkpoint_ref_checksum=cast("str | None", mapping["input_checkpoint_ref_checksum"]),
        )
        supplied = require_checksum(mapping["configuration_checksum"], "configuration_checksum")
        if stage.configuration_checksum != supplied:
            msg = "Training stage configuration checksum changed during normalization."
            raise ValueError(msg)
        return stage

    def to_json(self) -> str:
        """Return canonical JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> TrainingStageConfig:
        """Construct a stage from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))


def _validate_seed_domains(value: object) -> Mapping[str, object]:
    """Validate the six explicit and disjoint random-stream domains."""
    domains = freeze_json_mapping(value, "seed_domains")
    if frozenset(domains) != frozenset(SEED_DOMAIN_ROLES):
        msg = f"seed_domains keys must exactly match {SEED_DOMAIN_ROLES!r}."
        raise ValueError(msg)
    domain_ids = tuple(require_slug(domains[role], f"seed_domains.{role}") for role in SEED_DOMAIN_ROLES)
    if domain_ids != SEED_DOMAIN_ROLES:
        msg = "seed_domains must use the exact preregistered role-to-identical-domain mapping."
        raise ValueError(msg)
    return domains


def _validate_materialization_policy(value: object) -> Mapping[str, object]:
    """Validate the final, identity-bearing materialization policy."""
    policy = freeze_json_mapping(value, "final_materialization_policy")
    expected = frozenset({
        "policy_id",
        "compiler_policy_id",
        "connectivity_id",
        "routing_policy_id",
        "optimization_level",
        "noise_placement",
        "parameter_source",
    })
    require_exact_keys(policy, expected, "final_materialization_policy")
    for key in expected - {"optimization_level"}:
        require_slug(policy[key], f"final_materialization_policy.{key}")
    require_int(policy["optimization_level"], "final_materialization_policy.optimization_level", minimum=0)
    return policy


def _method_family(method_id: object) -> str:
    """Return the frozen comparison family for one preregistered method."""
    normalized = require_slug(method_id, "method_id")
    try:
        return _METHOD_FAMILY_BY_METHOD_ID[normalized]
    except KeyError as error:
        msg = f"method_id {normalized!r} has no frozen Phase II comparison-family mapping."
        raise ValueError(msg) from error


def _validate_wp21_pruning_stage_policy(
    value: Mapping[str, object],
    *,
    method_id: str,
    score_rule: str,
    random_seed: int | None,
) -> _WP21PruningStageSpec:
    """Validate one embedded WP21 pruning policy without an import cycle."""
    # ``pruning`` imports pipeline records for its executable adapter.  Keeping
    # this import at the validation boundary lets both modules remain usable
    # during module initialization while still making the pruning schema the
    # single authority for the opaque optimizer-hyperparameter payload.
    from .pruning import PruningStageSpec  # noqa: PLC0415

    return cast(
        "_WP21PruningStageSpec",
        PruningStageSpec.from_mapping(
            value,
            method_id=method_id,
            score_rule=score_rule,
            random_seed=random_seed,
        ),
    )


def fixture_target_spec_checksum(
    target_namespace: Literal["phase1_fixture", "legacy_reproduction"],
    target_instance_id: str,
    qubit_count: int,
) -> str:
    """Return the frozen per-key commitment for an immutable fixture target."""
    if target_namespace not in {"phase1_fixture", "legacy_reproduction"}:
        msg = "fixture target commitments require a Phase I or legacy namespace."
        raise ValueError(msg)
    target_id = require_string(target_instance_id, "target_instance_id")
    qubits = require_int(qubit_count, "qubit_count", minimum=2)
    manifest_checksum = (
        PHASE1_FIXTURE_MANIFEST_CHECKSUM
        if target_namespace == "phase1_fixture"
        else LEGACY_REPRODUCTION_MANIFEST_CHECKSUM
    )
    return canonical_checksum({
        "binding_version": "yaqs.state_preparation.phase2.fixture_target_binding.v1",
        "target_namespace": target_namespace,
        "target_manifest_checksum": manifest_checksum,
        "target_instance_id": target_id,
        "qubit_count": qubits,
    })


def _derive_resolved_seed(
    *,
    optimization_seed: int,
    domain_id: str,
    binding: str,
    resolution_context_checksum: str,
) -> int:
    """Derive one deterministic, domain-separated unsigned 64-bit seed."""
    payload = canonical_json({
        "derivation_version": "yaqs.state_preparation.phase2.stage_seed_derivation.v1",
        "optimization_seed": optimization_seed,
        "domain_id": domain_id,
        "binding": binding,
        "resolution_context_checksum": resolution_context_checksum,
    }).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def _derive_legacy_layerwise_seed(*, optimization_seed: int, binding: str) -> int:
    """Reproduce the archived target-seed arithmetic for one legacy stream.

    The historical experiment used the target seed as its only outer random
    identifier.  This deliberately isolated rule must not be used by corrected
    Phase II methods, whose streams remain hash-derived and domain separated.
    """
    seed = cast("int", _require_seed(optimization_seed, "optimization_seed", allow_none=False))
    formulas = {
        "legacy_layerwise_depth1_initialization": 20 * seed,
        "legacy_layerwise_depth2_initialization": 20 * seed + 2,
        "legacy_layerwise_depth3_initialization": 20 * seed + 3,
        "legacy_layerwise_depth4_initialization": 20 * seed + 4,
        "legacy_layerwise_growth_optimizer": 30 * seed,
        "legacy_layerwise_final_optimizer": 40 * seed,
        "legacy_layerwise_fixed_crn": 40 * seed,
    }
    try:
        resolved = formulas[binding]
    except KeyError as error:
        msg = f"Unknown historical layerwise seed binding {binding!r}."
        raise ValueError(msg) from error
    return cast("int", _require_seed(resolved, f"resolved seed for {binding}", allow_none=False))


@dataclass(frozen=True, slots=True)
class TrainingStageTemplate:
    """Target-independent stage policy with symbolic random-stream bindings."""

    stage_policy: Mapping[str, object]
    seed_bindings: Mapping[str, object]
    schema_version: str = field(default=TRAINING_STAGE_TEMPLATE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact policy fields and required symbolic seed bindings."""
        policy = freeze_json_mapping(self.stage_policy, "stage_policy")
        require_exact_keys(policy, _STAGE_TEMPLATE_POLICY_KEYS, "stage_policy")
        bindings = freeze_json_mapping(self.seed_bindings, "seed_bindings")
        binding_keys = frozenset({"initialization", "optimizer", "training", "checkpoint_validation"})
        require_exact_keys(bindings, binding_keys, "seed_bindings")
        normalized_bindings: dict[str, object] = {}
        for role in binding_keys:
            value = bindings[role]
            normalized_bindings[role] = None if value is None else require_slug(value, f"seed_bindings.{role}")
        present = [value for value in normalized_bindings.values() if value is not None]
        if len(present) != len(set(present)):
            msg = "Stage seed bindings must be distinct."
            raise ValueError(msg)

        validation_policy = require_mapping(
            policy["checkpoint_validation_policy"],
            "stage_policy.checkpoint_validation_policy",
        )
        require_exact_keys(
            validation_policy,
            _CHECKPOINT_VALIDATION_KEYS - {"seed"},
            "stage_policy.checkpoint_validation_policy",
        )
        required = {
            "initialization": policy["parameter_transfer_rule"]
            in {
                "initialize_random_uniform",
                "initialize_random_normal",
                "append_random_uniform",
                "append_random_normal",
            },
            "optimizer": policy["optimizer_id"] != "none"
            or (policy["stage_kind"] == "prune" and policy["pruning_rule"] == "random"),
            "training": policy["training_noise_id"] != NOISELESS_NOISE_ID,
            "checkpoint_validation": validation_policy["trajectory_count"] != 0,
        }
        for role, is_required in required.items():
            if is_required != (normalized_bindings[role] is not None):
                msg = f"seed_bindings.{role} is required exactly when that random stream is active."
                raise ValueError(msg)

        object.__setattr__(self, "stage_policy", policy)
        object.__setattr__(
            self,
            "seed_bindings",
            freeze_json_mapping(normalized_bindings, "seed_bindings"),
        )
        # Resolve once with harmless probe seeds so all stage-local semantics
        # are checked by the canonical concrete schema.
        self._resolve_with_seeds(
            seed_values={
                role: (index + 1 if value is not None else None)
                for index, (role, value) in enumerate(normalized_bindings.items())
            },
            input_checkpoint_path=None,
            input_checkpoint_checksum=None,
            input_checkpoint_provenance_checksum=None,
            input_checkpoint_pipeline_prefix=None,
            input_checkpoint_ref_checksum=None,
            template_probe=True,
        )

    @property
    def stage_index(self) -> int:
        """Zero-based stage position."""
        return cast("int", self.stage_policy["stage_index"])

    @property
    def stage_id(self) -> str:
        """Stable stage identifier."""
        return cast("str", self.stage_policy["stage_id"])

    def _resolve_with_seeds(
        self,
        *,
        seed_values: Mapping[str, int | None],
        input_checkpoint_path: str | None,
        input_checkpoint_checksum: str | None,
        input_checkpoint_provenance_checksum: str | None,
        input_checkpoint_pipeline_prefix: str | None,
        input_checkpoint_ref_checksum: str | None,
        template_probe: bool = False,
    ) -> TrainingStageConfig:
        """Construct the concrete stage after symbolic seeds are resolved."""
        policy = self.stage_policy
        transfer = cast("str", policy["parameter_transfer_rule"])
        if template_probe and transfer == "load_checkpoint":
            transfer = "copy"
        validation_data = thaw_json_mapping(cast("Mapping[str, object]", policy["checkpoint_validation_policy"]))
        validation_data["seed"] = seed_values["checkpoint_validation"]
        return TrainingStageConfig(
            stage_index=cast("int", policy["stage_index"]),
            stage_id=cast("str", policy["stage_id"]),
            stage_kind=cast("Literal['optimize', 'grow', 'prune']", policy["stage_kind"]),
            input_topology_id=cast("str | None", policy["input_topology_id"]),
            output_topology_id=cast("str", policy["output_topology_id"]),
            input_parameter_count=cast("int", policy["input_parameter_count"]),
            output_parameter_count=cast("int", policy["output_parameter_count"]),
            parameter_transfer_rule=transfer,
            initialization_seed=seed_values["initialization"],
            optimizer_id=cast("str", policy["optimizer_id"]),
            optimizer_hyperparameters=cast("Mapping[str, object]", policy["optimizer_hyperparameters"]),
            optimizer_seed=seed_values["optimizer"],
            iteration_budget=cast("int", policy["iteration_budget"]),
            training_noise_id=cast("str", policy["training_noise_id"]),
            noise_definition_version=cast("str", policy["noise_definition_version"]),
            noise_strength_scale=cast("float | None", policy["noise_strength_scale"]),
            tjm_dt=cast("float | None", policy["tjm_dt"]),
            trajectory_count=cast("int", policy["trajectory_count"]),
            training_seed=seed_values["training"],
            trajectory_update=cast("Literal['independent', 'cross'] | None", policy["trajectory_update"]),
            sampling_policy=cast(
                "Literal['none', 'resampled', 'crn_fixed', 'crn_refresh']",
                policy["sampling_policy"],
            ),
            crn_refresh_interval=cast("int | None", policy["crn_refresh_interval"]),
            checkpoint_validation=CheckpointValidationConfig.from_dict(validation_data),
            pruning_rule=cast(
                "Literal['none', 'random', 'magnitude', 'impact_one_shot', 'impact_iterative', 'operator_selection']",
                policy["pruning_rule"],
            ),
            pruning_threshold=cast("float | None", policy["pruning_threshold"]),
            max_bond_dimension=cast("int | None", policy["max_bond_dimension"]),
            svd_threshold=cast("float", policy["svd_threshold"]),
            truncation_mode=cast("Literal['discarded_weight', 'relative']", policy["truncation_mode"]),
            min_bond_dimension=cast("int", policy["min_bond_dimension"]),
            input_checkpoint_path=input_checkpoint_path,
            input_checkpoint_checksum=input_checkpoint_checksum,
            input_checkpoint_provenance_checksum=input_checkpoint_provenance_checksum,
            input_checkpoint_pipeline_prefix=input_checkpoint_pipeline_prefix,
            input_checkpoint_ref_checksum=input_checkpoint_ref_checksum,
        )

    def _resolved_seed_values(
        self,
        *,
        optimization_seed: int,
        seed_domains: Mapping[str, object],
        resolution_context_checksum: str,
    ) -> dict[str, int | None]:
        """Resolve only this stage's symbolic random streams."""
        outer = cast("int", _require_seed(optimization_seed, "optimization_seed", allow_none=False))
        domains = _validate_seed_domains(seed_domains)
        context_checksum = require_checksum(
            resolution_context_checksum,
            "resolution_context_checksum",
        )
        values: dict[str, int | None] = {}
        for role, binding_value in self.seed_bindings.items():
            if binding_value is None:
                values[role] = None
                continue
            binding = cast("str", binding_value)
            if binding in _LEGACY_LAYERWISE_SEED_BINDING_SET:
                values[role] = _derive_legacy_layerwise_seed(
                    optimization_seed=outer,
                    binding=binding,
                )
                continue
            domain_role = {
                "initialization": "initialization",
                "optimizer": "optimizer_ordering",
                "training": "training_trajectory",
                "checkpoint_validation": "checkpoint_validation",
            }[role]
            values[role] = _derive_resolved_seed(
                optimization_seed=outer,
                domain_id=cast("str", domains[domain_role]),
                binding=binding,
                resolution_context_checksum=context_checksum,
            )
        return values

    def resolve(
        self,
        *,
        optimization_seed: int,
        seed_domains: Mapping[str, object],
        resolution_context_checksum: str,
        input_checkpoint_path: str | None = None,
        input_checkpoint_ref: ExternalCheckpointRef | None = None,
    ) -> TrainingStageConfig:
        """Resolve symbolic stage streams deterministically from one outer seed."""
        if input_checkpoint_ref is not None:
            if not isinstance(input_checkpoint_ref, ExternalCheckpointRef):
                _raise_type_error("input_checkpoint_ref", "an ExternalCheckpointRef", input_checkpoint_ref)
            if (
                self.stage_policy["input_topology_id"] != input_checkpoint_ref.output_topology_id
                or self.stage_policy["input_parameter_count"] != input_checkpoint_ref.output_parameter_count
            ):
                msg = "External checkpoint topology and parameter count do not match the consuming stage."
                raise ValueError(msg)
        return self._resolve_with_seeds(
            seed_values=self._resolved_seed_values(
                optimization_seed=optimization_seed,
                seed_domains=seed_domains,
                resolution_context_checksum=resolution_context_checksum,
            ),
            input_checkpoint_path=input_checkpoint_path,
            input_checkpoint_checksum=(
                None if input_checkpoint_ref is None else input_checkpoint_ref.produced_checkpoint_checksum
            ),
            input_checkpoint_provenance_checksum=(
                None if input_checkpoint_ref is None else input_checkpoint_ref.checkpoint_provenance_checksum
            ),
            input_checkpoint_pipeline_prefix=(
                None if input_checkpoint_ref is None else input_checkpoint_ref.producer_pipeline_prefix_id
            ),
            input_checkpoint_ref_checksum=(
                None if input_checkpoint_ref is None else input_checkpoint_ref.provenance_ref_checksum
            ),
        )

    def resolve_recorded(
        self,
        *,
        optimization_seed: int,
        seed_domains: Mapping[str, object],
        resolution_context_checksum: str,
        recorded_stage: TrainingStageConfig | None,
    ) -> TrainingStageConfig:
        """Reconstruct a stage from compact checkpoint fields already sealed in a pipeline config."""
        return self._resolve_with_seeds(
            seed_values=self._resolved_seed_values(
                optimization_seed=optimization_seed,
                seed_domains=seed_domains,
                resolution_context_checksum=resolution_context_checksum,
            ),
            input_checkpoint_path=None if recorded_stage is None else recorded_stage.input_checkpoint_path,
            input_checkpoint_checksum=None if recorded_stage is None else recorded_stage.input_checkpoint_checksum,
            input_checkpoint_provenance_checksum=(
                None if recorded_stage is None else recorded_stage.input_checkpoint_provenance_checksum
            ),
            input_checkpoint_pipeline_prefix=(
                None if recorded_stage is None else recorded_stage.input_checkpoint_pipeline_prefix
            ),
            input_checkpoint_ref_checksum=(
                None if recorded_stage is None else recorded_stage.input_checkpoint_ref_checksum
            ),
        )

    def identity_payload(self) -> dict[str, object]:
        """Return every target-independent stage-template field."""
        return {
            "schema_version": self.schema_version,
            "stage_policy": thaw_json_mapping(self.stage_policy),
            "seed_bindings": thaw_json_mapping(self.seed_bindings),
        }

    @property
    def configuration_checksum(self) -> str:
        """Stable target-independent stage-template identity."""
        return canonical_checksum(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached exact-schema stage template."""
        return {**self.identity_payload(), "configuration_checksum": self.configuration_checksum}

    @classmethod
    def from_dict(cls, data: object) -> TrainingStageTemplate:
        """Construct and identity-verify a strict stage template."""
        mapping = require_mapping(data, "training stage template")
        require_exact_keys(mapping, _STAGE_TEMPLATE_KEYS, "training stage template")
        if mapping["schema_version"] != TRAINING_STAGE_TEMPLATE_SCHEMA_VERSION:
            msg = f"schema_version must be {TRAINING_STAGE_TEMPLATE_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        template = cls(
            stage_policy=cast("Mapping[str, object]", mapping["stage_policy"]),
            seed_bindings=cast("Mapping[str, object]", mapping["seed_bindings"]),
        )
        if mapping["configuration_checksum"] != template.configuration_checksum:
            msg = "Training stage template checksum does not match its content."
            raise ValueError(msg)
        return template


_TARGET_REF_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class Phase2TargetRef:
    """Compact sealed reference derived from one typed Phase II target manifest."""

    target_manifest_checksum: str
    preregistration_checksum: str
    population_config_checksum: str
    data_role: str
    population_scope: str
    target_spec: TargetInstanceSpec
    _marker: object = field(repr=False, compare=False)
    schema_version: str = field(default=PHASE2_TARGET_REF_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the factory boundary and every compact manifest/spec alias."""
        if self._marker is not _TARGET_REF_SENTINEL:
            msg = "Phase2TargetRef records must be created from a typed target manifest or sealed JSON."
            raise ValueError(msg)
        manifest_checksum = require_checksum(self.target_manifest_checksum, "target_manifest_checksum")
        preregistration_checksum = require_checksum(self.preregistration_checksum, "preregistration_checksum")
        if preregistration_checksum != TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM:
            msg = "Phase2TargetRef must reference the trusted Phase II preregistration."
            raise ValueError(msg)
        population_checksum = require_checksum(self.population_config_checksum, "population_config_checksum")
        role = require_slug(self.data_role, "data_role")
        scope = require_slug(self.population_scope, "population_scope")
        if not isinstance(self.target_spec, TargetInstanceSpec):
            _raise_type_error("target_spec", "a TargetInstanceSpec", self.target_spec)
        expected_qubits = 6 if scope == "primary_q6" else 12 if scope == "secondary_q12" else None
        if (
            role != self.target_spec.data_role
            or population_checksum != self.target_spec.population_config_checksum
            or expected_qubits != self.target_spec.qubit_count
        ):
            msg = "Phase2TargetRef manifest aliases do not match its typed target specification."
            raise ValueError(msg)
        object.__setattr__(self, "target_manifest_checksum", manifest_checksum)
        object.__setattr__(self, "preregistration_checksum", preregistration_checksum)
        object.__setattr__(self, "population_config_checksum", population_checksum)
        object.__setattr__(self, "data_role", role)
        object.__setattr__(self, "population_scope", scope)

    @classmethod
    def from_manifest(cls, manifest: TargetPopulationManifest, target_instance_id: str) -> Phase2TargetRef:
        """Create an exact compact reference from a typed manifest and member spec."""
        if not isinstance(manifest, TargetPopulationManifest):
            _raise_type_error("manifest", "a TargetPopulationManifest", manifest)
        target_id = require_string(target_instance_id, "target_instance_id")
        try:
            spec = next(item for item in manifest.instances if item.target_instance_id == target_id)
        except StopIteration as error:
            msg = "target_instance_id is absent from the supplied typed target manifest."
            raise ValueError(msg) from error
        return cls(
            target_manifest_checksum=manifest.content_checksum,
            preregistration_checksum=manifest.preregistration_checksum,
            population_config_checksum=manifest.population_config_checksum,
            data_role=manifest.data_role,
            population_scope=manifest.population_scope,
            target_spec=spec,
            _marker=_TARGET_REF_SENTINEL,
        )

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered compact target provenance."""
        return {
            "schema_version": self.schema_version,
            "target_manifest_checksum": self.target_manifest_checksum,
            "preregistration_checksum": self.preregistration_checksum,
            "population_config_checksum": self.population_config_checksum,
            "data_role": self.data_role,
            "population_scope": self.population_scope,
            "target_spec": self.target_spec.to_dict(),
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the typed manifest/spec binding."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return a sealed JSON-native compact target reference."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> Phase2TargetRef:
        """Construct and checksum-verify a compact target reference."""
        mapping = verify_sealed_mapping(data, expected_keys=_PHASE2_TARGET_REF_KEYS, name="Phase II target reference")
        if mapping["schema_version"] != PHASE2_TARGET_REF_SCHEMA_VERSION:
            msg = f"schema_version must be {PHASE2_TARGET_REF_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        reference = cls(
            target_manifest_checksum=cast("str", mapping["target_manifest_checksum"]),
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            population_config_checksum=cast("str", mapping["population_config_checksum"]),
            data_role=cast("str", mapping["data_role"]),
            population_scope=cast("str", mapping["population_scope"]),
            target_spec=TargetInstanceSpec.from_dict(mapping["target_spec"]),
            _marker=_TARGET_REF_SENTINEL,
        )
        if mapping["content_checksum"] != reference.content_checksum:
            msg = "Phase II target reference checksum changed during normalization."
            raise ValueError(msg)
        return reference


@dataclass(frozen=True, slots=True)
class TrainingPipelineTemplate:
    """One target-independent candidate configuration used by screening."""

    template_id: str
    preregistration_checksum: str
    target_scope_id: str
    ansatz_family: str
    method_id: str
    method_version: str
    resource_stratum_id: str
    stages: tuple[TrainingStageTemplate, ...]
    seed_domains: Mapping[str, object]
    final_materialization_policy: Mapping[str, object]
    schema_version: str = field(default=TRAINING_PIPELINE_TEMPLATE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate candidate identity, method family, and ordered stage chain."""
        object.__setattr__(self, "template_id", require_slug(self.template_id, "template_id"))
        preregistration_checksum = require_checksum(
            self.preregistration_checksum,
            "preregistration_checksum",
        )
        if preregistration_checksum != TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM:
            msg = "Training pipeline templates must reference the trusted Phase II preregistration."
            raise ValueError(msg)
        object.__setattr__(self, "preregistration_checksum", preregistration_checksum)
        target_scope = require_slug(self.target_scope_id, "target_scope_id")
        if target_scope not in TARGET_SCOPE_IDS:
            msg = f"target_scope_id must be one of {TARGET_SCOPE_IDS!r}."
            raise ValueError(msg)
        object.__setattr__(self, "target_scope_id", target_scope)
        object.__setattr__(self, "ansatz_family", require_slug(self.ansatz_family, "ansatz_family"))
        object.__setattr__(self, "method_id", require_slug(self.method_id, "method_id"))
        _method_family(self.method_id)
        object.__setattr__(self, "method_version", require_string(self.method_version, "method_version"))
        object.__setattr__(
            self,
            "resource_stratum_id",
            require_slug(self.resource_stratum_id, "resource_stratum_id"),
        )
        if (self.method_id in _LEGACY_METHOD_IDS) != (target_scope == "legacy_reproduction"):
            msg = "Legacy method identities and the legacy_reproduction target scope must be used together."
            raise ValueError(msg)
        stages = tuple(self.stages)
        if not stages or not all(isinstance(stage, TrainingStageTemplate) for stage in stages):
            msg = "stages must contain at least one TrainingStageTemplate."
            raise TypeError(msg)
        if tuple(stage.stage_index for stage in stages) != tuple(range(len(stages))):
            msg = "Template stage indices must be contiguous and agree with tuple order."
            raise ValueError(msg)
        if len({stage.stage_id for stage in stages}) != len(stages):
            msg = "Template stage identifiers must be unique."
            raise ValueError(msg)
        first_policy = stages[0].stage_policy
        if (
            first_policy["input_topology_id"] is not None
            and first_policy["parameter_transfer_rule"] != "load_checkpoint"
        ):
            msg = "A pipeline with an existing first-stage topology must load a typed external checkpoint."
            raise ValueError(msg)
        if any(stage.stage_policy["parameter_transfer_rule"] == "load_checkpoint" for stage in stages[1:]):
            msg = "Only the first pipeline stage may load an external checkpoint; later stages use their predecessor."
            raise ValueError(msg)
        for predecessor, stage in itertools.pairwise(stages):
            if (
                stage.stage_policy["input_topology_id"] != predecessor.stage_policy["output_topology_id"]
                or stage.stage_policy["input_parameter_count"] != predecessor.stage_policy["output_parameter_count"]
            ):
                msg = "Template stage topology and parameter chains must be contiguous."
                raise ValueError(msg)
        pruning_stages = tuple(stage for stage in stages if stage.stage_policy["stage_kind"] == "prune")
        wp21_rule = _WP21_PRUNING_RULE_BY_METHOD_ID.get(self.method_id)
        policy_pruning_stages = tuple(
            stage
            for stage in pruning_stages
            if cast("Mapping[str, object]", stage.stage_policy["optimizer_hyperparameters"])
        )
        if wp21_rule is not None:
            iterative = self.method_id == "topdown_impact_iterative"
            valid_count = (
                len(pruning_stages) >= 2 and len(policy_pruning_stages) == len(pruning_stages)
                if iterative
                else len(pruning_stages) == len(policy_pruning_stages) == 1
            )
            if not valid_count:
                qualifier = "at least two" if iterative else "exactly one"
                msg = f"A WP21 top-down method requires {qualifier} schema-validated pruning transform(s)."
                raise ValueError(msg)
            if any(stage.stage_policy["pruning_rule"] != wp21_rule for stage in policy_pruning_stages):
                msg = "The pipeline method_id must agree with its WP21 pruning policy and rule."
                raise ValueError(msg)
            if iterative:
                pruning_indices = tuple(stage.stage_index for stage in policy_pruning_stages)
                for first_index, second_index in itertools.pairwise(pruning_indices):
                    first_policy = cast(
                        "Mapping[str, object]",
                        stages[first_index].stage_policy["optimizer_hyperparameters"],
                    )
                    relaxation = stages[first_index + 1] if first_index + 1 < len(stages) else None
                    if (
                        second_index != first_index + 2
                        or first_policy["relax_after_round"] is not True
                        or relaxation is None
                        or relaxation.stage_policy["stage_kind"] != "optimize"
                        or relaxation.stage_policy["optimizer_id"] == "none"
                        or not relaxation.stage_id.startswith("relax_round_")
                    ):
                        msg = "Iterative impact-pruning rounds must alternate with active relaxation stages."
                        raise ValueError(msg)
                terminal_index = pruning_indices[-1]
                terminal_policy = cast(
                    "Mapping[str, object]",
                    stages[terminal_index].stage_policy["optimizer_hyperparameters"],
                )
                terminal_successor = stages[terminal_index + 1] if terminal_index + 1 < len(stages) else None
                has_terminal_relaxation = (
                    terminal_successor is not None
                    and terminal_successor.stage_policy["stage_kind"] == "optimize"
                    and terminal_successor.stage_policy["optimizer_id"] != "none"
                    and terminal_successor.stage_id.startswith("relax_round_")
                )
                if (terminal_policy["relax_after_round"] is True) != has_terminal_relaxation:
                    msg = "A terminal iterative pruning relaxation must be declared by relax_after_round."
                    raise ValueError(msg)
        elif policy_pruning_stages:
            msg = "Embedded WP21 pruning policies are reserved for top-down pruning pipeline methods."
            raise ValueError(msg)
        reserved_bindings = {
            binding
            for stage in stages
            for binding in stage.seed_bindings.values()
            if binding in _LEGACY_LAYERWISE_SEED_BINDING_SET
        }
        if self.method_id == "layerwise_bmpd_crn_legacy_v1":
            expected_bindings = (
                {
                    "initialization": LEGACY_LAYERWISE_SEED_BINDINGS[0],
                    "optimizer": LEGACY_LAYERWISE_SEED_BINDINGS[4],
                    "training": None,
                    "checkpoint_validation": None,
                },
                {
                    "initialization": LEGACY_LAYERWISE_SEED_BINDINGS[1],
                    "optimizer": LEGACY_LAYERWISE_SEED_BINDINGS[4],
                    "training": None,
                    "checkpoint_validation": None,
                },
                {
                    "initialization": LEGACY_LAYERWISE_SEED_BINDINGS[2],
                    "optimizer": LEGACY_LAYERWISE_SEED_BINDINGS[4],
                    "training": None,
                    "checkpoint_validation": None,
                },
                {
                    "initialization": LEGACY_LAYERWISE_SEED_BINDINGS[3],
                    "optimizer": LEGACY_LAYERWISE_SEED_BINDINGS[4],
                    "training": None,
                    "checkpoint_validation": None,
                },
                {
                    "initialization": None,
                    "optimizer": LEGACY_LAYERWISE_SEED_BINDINGS[5],
                    "training": LEGACY_LAYERWISE_SEED_BINDINGS[6],
                    "checkpoint_validation": None,
                },
            )
            if len(stages) != len(expected_bindings) or any(
                dict(stage.seed_bindings) != expected for stage, expected in zip(stages, expected_bindings, strict=True)
            ):
                msg = "The historical layerwise method requires its exact five-stage legacy seed policy."
                raise ValueError(msg)
        elif reserved_bindings:
            msg = "Historical layerwise seed bindings are reserved for layerwise_bmpd_crn_legacy_v1."
            raise ValueError(msg)
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "seed_domains", _validate_seed_domains(self.seed_domains))
        object.__setattr__(
            self,
            "final_materialization_policy",
            _validate_materialization_policy(self.final_materialization_policy),
        )

    @property
    def method_family_id(self) -> str:
        """Frozen, non-caller-controlled method comparison family."""
        return _method_family(self.method_id)

    def identity_payload(self) -> dict[str, object]:
        """Return every target-independent candidate identity field."""
        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "preregistration_checksum": self.preregistration_checksum,
            "target_scope_id": self.target_scope_id,
            "ansatz_family": self.ansatz_family,
            "method_family_id": self.method_family_id,
            "method_id": self.method_id,
            "method_version": self.method_version,
            "resource_stratum_id": self.resource_stratum_id,
            "stages": [stage.to_dict() for stage in self.stages],
            "seed_domains": thaw_json_mapping(self.seed_domains),
            "final_materialization_policy": thaw_json_mapping(self.final_materialization_policy),
        }

    @property
    def configuration_checksum(self) -> str:
        """Candidate checksum shared across every screening target/seed cell."""
        return canonical_checksum(self.identity_payload())

    def matching_projection(self) -> Mapping[str, object]:
        """Return the mechanically derived matched-treatment projection.

        For the preregistered v2 noisy/noiseless pair, only the final
        ``final_finetune`` training-treatment bundle is replaced.  Growth,
        optimizer, validation, topology, truncation, and materialization fields
        remain byte-identical requirements.
        """
        payload = self.identity_payload()
        if self.method_id not in {"layerwise_bmpd_crn_v2", "layerwise_bmpd_noiseless"}:
            return freeze_json_mapping(payload, "matching_projection")
        if self.method_version != "1" or self.stages[-1].stage_id != "final_finetune":
            msg = "The matched layerwise pair requires version 1 and a final 'final_finetune' stage."
            raise ValueError(msg)
        projected_stages = cast("list[dict[str, object]]", payload["stages"])
        for stage_index, stage in enumerate(projected_stages[:-1]):
            policy = cast("dict[str, object]", stage["stage_policy"])
            bindings = cast("dict[str, object]", stage["seed_bindings"])
            if (
                policy["training_noise_id"] != NOISELESS_NOISE_ID
                or policy["noise_definition_version"] != "yaqs.state_preparation.noise.v1"
                or policy["noise_strength_scale"] is not None
                or policy["tjm_dt"] is not None
                or policy["trajectory_count"] != 0
                or policy["trajectory_update"] is not None
                or policy["sampling_policy"] != "none"
                or policy["crn_refresh_interval"] is not None
                or bindings["training"] is not None
            ):
                msg = (
                    "The preregistered matched layerwise pair requires canonical noiseless "
                    f"training in every pre-final stage; stage {stage_index} differs."
                )
                raise ValueError(msg)
        final_stage = projected_stages[-1]
        final_policy = cast("dict[str, object]", final_stage["stage_policy"])
        final_bindings = cast("dict[str, object]", final_stage["seed_bindings"])
        treatment_keys = {
            "training_noise_id",
            "noise_definition_version",
            "noise_strength_scale",
            "tjm_dt",
            "trajectory_count",
            "trajectory_update",
            "sampling_policy",
            "crn_refresh_interval",
        }
        if self.method_id == "layerwise_bmpd_crn_v2":
            if (
                final_policy["training_noise_id"] != "depolarizing_1s_all"
                or final_policy["noise_definition_version"] != "yaqs.state_preparation.noise.v1"
                or type(final_policy["noise_strength_scale"]) is not float
                or float(cast("float", final_policy["noise_strength_scale"])).hex() != float(1).hex()
                or type(final_policy["tjm_dt"]) is not float
                or float(cast("float", final_policy["tjm_dt"])).hex() != float(1).hex()
                or final_policy["trajectory_count"] == 0
                or final_policy["trajectory_update"] != "independent"
                or final_policy["sampling_policy"] != "crn_fixed"
                or final_policy["crn_refresh_interval"] is not None
                or final_bindings["training"] is None
            ):
                msg = "layerwise_bmpd_crn_v2 does not contain the preregistered noisy final treatment."
                raise ValueError(msg)
        elif (
            final_policy["training_noise_id"] != NOISELESS_NOISE_ID
            or final_policy["noise_definition_version"] != "yaqs.state_preparation.noise.v1"
            or final_policy["noise_strength_scale"] is not None
            or final_policy["tjm_dt"] is not None
            or final_policy["trajectory_count"] != 0
            or final_policy["trajectory_update"] is not None
            or final_policy["sampling_policy"] != "none"
            or final_policy["crn_refresh_interval"] is not None
            or final_bindings["training"] is not None
        ):
            msg = "layerwise_bmpd_noiseless does not contain the exact noiseless final treatment."
            raise ValueError(msg)
        if final_policy["iteration_budget"] != 200:
            msg = "The preregistered matched final-finetuning budget is 200 iterations."
            raise ValueError(msg)
        validation_policy = cast("dict[str, object]", final_policy["checkpoint_validation_policy"])
        if (
            validation_policy["noise_id"] != "depolarizing_1s_all"
            or validation_policy["noise_definition_version"] != "yaqs.state_preparation.noise.v1"
            or canonical_json(validation_policy["noise_strength_scale"]) != "1.0"
            or canonical_json(validation_policy["tjm_dt"]) != "1.0"
            or validation_policy["trajectory_count"] == 0
            or validation_policy["sampling_policy"] != "crn_fixed"
            or validation_policy["ensemble_refresh_interval"] is not None
            or validation_policy["cadence"] != 10
            or validation_policy["selection_rule"] != "best_validation_fidelity"
            or validation_policy["tie_breaker"] != "earliest_iteration"
        ):
            msg = "The matched pair requires noisy CRN checkpoint validation with earliest-iteration ties."
            raise ValueError(msg)
        for key in treatment_keys:
            final_policy[key] = "__matched_training_treatment__"
        final_bindings["training"] = "__matched_training_treatment__"
        payload["template_id"] = "layerwise_bmpd_matched_v2"
        payload["method_family_id"] = "layerwise_bmpd_matched_v2"
        payload["method_id"] = "layerwise_bmpd_matched_v2"
        payload["method_version"] = "matched"
        for stage in projected_stages:
            stage.pop("configuration_checksum")
        return freeze_json_mapping(payload, "matching_projection")

    @property
    def matching_projection_checksum(self) -> str:
        """Checksum used by WP15 comparator pairing."""
        return canonical_checksum(self.matching_projection())

    def resolution_context_checksum(
        self,
        *,
        stage_index: int,
        target_instance_id: str,
        optimization_block_id: str,
    ) -> str:
        """Bind stage streams only to the compatible prefix and outer cell."""
        index = require_int(stage_index, "stage_index", minimum=0)
        if index >= len(self.stages):
            msg = "stage_index does not exist in this template."
            raise ValueError(msg)
        prefix_projection = thaw_json_mapping(self.matching_projection())
        projected_stages = cast("list[object]", prefix_projection["stages"])
        prefix_projection["stages"] = projected_stages[: index + 1]
        # Final materialization is downstream of every training-stage prefix.
        # Excluding it prevents suffix-only changes from invalidating reusable
        # checkpoints or perturbing earlier random streams.
        prefix_projection.pop("final_materialization_policy")
        return canonical_checksum({
            "derivation_version": "yaqs.state_preparation.phase2.pipeline_resolution_context.v2",
            "template_prefix_projection": prefix_projection,
            "target_instance_id": require_string(target_instance_id, "target_instance_id"),
            "optimization_block_id": require_slug(optimization_block_id, "optimization_block_id"),
        })

    def resolve(
        self,
        *,
        target_namespace: Literal["phase2", "phase1_fixture", "legacy_reproduction"],
        target_manifest: TargetPopulationManifest | None,
        target_instance_id: str,
        target_population_manifest_checksum: str,
        target_instance_spec_checksum: str,
        target_family_id: str,
        target_stratum_id: str,
        qubit_count: int,
        optimization_block_id: str,
        optimization_seed: int,
        data_role: Literal[
            "development",
            "checkpoint_validation",
            "screening_selection",
            "secondary_benchmark",
            "confirmatory",
        ],
        input_checkpoint_path: str | None = None,
        input_checkpoint_ref: ExternalCheckpointRef | None = None,
    ) -> TrainingPipelineConfig:
        """Resolve this candidate deterministically for one target/outer seed."""
        if input_checkpoint_ref is not None:
            producer = input_checkpoint_ref.producer_result.config
            if (
                producer.target_namespace != target_namespace
                or producer.target_instance_id != target_instance_id
                or producer.target_population_manifest_checksum != target_population_manifest_checksum
                or producer.target_instance_spec_checksum != target_instance_spec_checksum
                or producer.target_family_id != target_family_id
                or producer.target_stratum_id != target_stratum_id
                or producer.qubit_count != qubit_count
                or producer.optimization_block_id != optimization_block_id
                or producer.optimization_seed != optimization_seed
                or producer.data_role != data_role
            ):
                msg = "External checkpoint producer does not belong to the same target and optimization cell."
                raise ValueError(msg)
        stages = tuple(
            stage.resolve(
                optimization_seed=optimization_seed,
                seed_domains=self.seed_domains,
                resolution_context_checksum=self.resolution_context_checksum(
                    stage_index=stage.stage_index,
                    target_instance_id=target_instance_id,
                    optimization_block_id=optimization_block_id,
                ),
                input_checkpoint_path=input_checkpoint_path if stage.stage_index == 0 else None,
                input_checkpoint_ref=input_checkpoint_ref if stage.stage_index == 0 else None,
            )
            for stage in self.stages
        )
        return TrainingPipelineConfig(
            template=self,
            target_ref=(
                None if target_manifest is None else Phase2TargetRef.from_manifest(target_manifest, target_instance_id)
            ),
            target_namespace=target_namespace,
            target_instance_id=target_instance_id,
            target_population_manifest_checksum=target_population_manifest_checksum,
            target_instance_spec_checksum=target_instance_spec_checksum,
            target_family_id=target_family_id,
            target_stratum_id=target_stratum_id,
            qubit_count=qubit_count,
            optimization_block_id=optimization_block_id,
            optimization_seed=optimization_seed,
            stages=stages,
            data_role=data_role,
        )

    def to_dict(self) -> dict[str, object]:
        """Return an exact-schema target-independent candidate template."""
        return {
            **self.identity_payload(),
            "matching_projection_checksum": self.matching_projection_checksum,
            "configuration_checksum": self.configuration_checksum,
        }

    @classmethod
    def from_dict(cls, data: object) -> TrainingPipelineTemplate:
        """Construct and identity-verify a strict candidate template."""
        mapping = require_mapping(data, "training pipeline template")
        require_exact_keys(mapping, _PIPELINE_TEMPLATE_KEYS, "training pipeline template")
        if mapping["schema_version"] != TRAINING_PIPELINE_TEMPLATE_SCHEMA_VERSION:
            msg = f"schema_version must be {TRAINING_PIPELINE_TEMPLATE_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        stage_data = mapping["stages"]
        if isinstance(stage_data, (str, bytes)) or not isinstance(stage_data, Sequence):
            _raise_type_error("stages", "a sequence", stage_data)
        template = cls(
            template_id=cast("str", mapping["template_id"]),
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            target_scope_id=cast("str", mapping["target_scope_id"]),
            ansatz_family=cast("str", mapping["ansatz_family"]),
            method_id=cast("str", mapping["method_id"]),
            method_version=cast("str", mapping["method_version"]),
            resource_stratum_id=cast("str", mapping["resource_stratum_id"]),
            stages=tuple(TrainingStageTemplate.from_dict(item) for item in stage_data),
            seed_domains=cast("Mapping[str, object]", mapping["seed_domains"]),
            final_materialization_policy=cast("Mapping[str, object]", mapping["final_materialization_policy"]),
        )
        expected = {
            "method_family_id": template.method_family_id,
            "matching_projection_checksum": template.matching_projection_checksum,
            "configuration_checksum": template.configuration_checksum,
        }
        for name, value in expected.items():
            if mapping[name] != value:
                msg = f"Serialized template {name} does not match its derived value."
                raise ValueError(msg)
        return template

    def to_json(self) -> str:
        """Return canonical JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> TrainingPipelineTemplate:
        """Construct a template from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class TrainingPipelineConfig:
    """Concrete target/optimization-cell resolution of one candidate template."""

    template: TrainingPipelineTemplate
    target_ref: Phase2TargetRef | None
    target_namespace: Literal["phase2", "phase1_fixture", "legacy_reproduction"]
    target_instance_id: str
    target_population_manifest_checksum: str
    target_instance_spec_checksum: str
    target_family_id: str
    target_stratum_id: str
    qubit_count: int
    optimization_block_id: str
    optimization_seed: int
    stages: tuple[TrainingStageConfig, ...]
    data_role: Literal[
        "development",
        "checkpoint_validation",
        "screening_selection",
        "secondary_benchmark",
        "confirmatory",
    ]
    schema_version: str = field(default=TRAINING_PIPELINE_CONFIG_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate template resolution, target role, and concrete seed chain."""
        if not isinstance(self.template, TrainingPipelineTemplate):
            _raise_type_error("template", "a TrainingPipelineTemplate", self.template)
        if self.target_namespace not in TARGET_NAMESPACES:
            msg = f"target_namespace must be one of {TARGET_NAMESPACES!r}."
            raise ValueError(msg)
        target_id = require_string(self.target_instance_id, "target_instance_id")
        if _IDENTIFIER_PATTERN.fullmatch(target_id) is None:
            msg = "target_instance_id must be a stable lowercase identifier."
            raise ValueError(msg)
        if self.target_namespace == "phase2" and _PHASE2_TARGET_PATTERN.fullmatch(target_id) is None:
            msg = f"Phase II target identifiers must use {PHASE2_TARGET_ID_PREFIX!r} plus 64 lowercase hex digits."
            raise ValueError(msg)
        manifest_checksum = require_checksum(
            self.target_population_manifest_checksum,
            "target_population_manifest_checksum",
        )
        spec_checksum = require_checksum(
            self.target_instance_spec_checksum,
            "target_instance_spec_checksum",
        )
        family = require_slug(self.target_family_id, "target_family_id")
        stratum = require_slug(self.target_stratum_id, "target_stratum_id")
        qubit_count = require_int(self.qubit_count, "qubit_count", minimum=2)
        block_id = require_slug(self.optimization_block_id, "optimization_block_id")
        optimization_seed = cast(
            "int",
            _require_seed(self.optimization_seed, "optimization_seed", allow_none=False),
        )
        if self.data_role not in DATA_ROLES:
            msg = f"data_role must be one of {DATA_ROLES!r}."
            raise ValueError(msg)
        if self.target_namespace == "phase2":
            if not isinstance(self.target_ref, Phase2TargetRef):
                _raise_type_error(
                    "target_ref",
                    "a Phase2TargetRef derived from the exact typed manifest",
                    self.target_ref,
                )
            target_spec = self.target_ref.target_spec
            expected_manifest_role, expected_population_scope = {
                "development": ("development", "primary_q6"),
                "checkpoint_validation": ("development", "primary_q6"),
                "screening_selection": ("screening_selection", "primary_q6"),
                "secondary_benchmark": ("screening_selection", "secondary_q12"),
                "confirmatory": ("confirmatory", "primary_q6"),
            }[self.data_role]
            if (
                self.target_ref.preregistration_checksum != self.template.preregistration_checksum
                or self.target_ref.data_role != expected_manifest_role
                or self.target_ref.population_scope != expected_population_scope
                or manifest_checksum != self.target_ref.target_manifest_checksum
                or target_id != target_spec.target_instance_id
                or spec_checksum != target_spec.content_checksum
                or family != target_spec.family_id
                or stratum != target_spec.stratum_id
                or qubit_count != target_spec.qubit_count
            ):
                msg = "Phase II target aliases do not match the exact typed target manifest and specification."
                raise ValueError(msg)
        elif self.target_ref is not None:
            msg = "Phase I and legacy fixture namespaces cannot carry a Phase II target manifest."
            raise ValueError(msg)
        elif self.target_namespace == "phase1_fixture":
            expected_metadata = _PHASE1_FIXTURE_METADATA.get(target_id)
            if (
                target_id not in PHASE1_FIXTURE_TARGET_IDS
                or expected_metadata is None
                or qubit_count not in SUPPORTED_QUBIT_COUNTS
                or (family, stratum) != expected_metadata
                or manifest_checksum != PHASE1_FIXTURE_MANIFEST_CHECKSUM
                or spec_checksum != fixture_target_spec_checksum("phase1_fixture", target_id, qubit_count)
            ):
                msg = "Phase I fixtures must use one of the exact immutable 18 target/qubit records."
                raise ValueError(msg)
        elif (
            target_id not in LEGACY_REPRODUCTION_TARGET_IDS
            or qubit_count != 8
            or family != "tfim_ground_state"
            or stratum != "legacy_disordered"
            or manifest_checksum != LEGACY_REPRODUCTION_MANIFEST_CHECKSUM
            or spec_checksum != fixture_target_spec_checksum("legacy_reproduction", target_id, qubit_count)
        ):
            msg = "Legacy reproduction targets must use one of the exact five immutable q8 TFIM fixtures."
            raise ValueError(msg)
        if self.data_role != "secondary_benchmark" and self.target_namespace != "phase2":
            msg = "Phase I and legacy targets are allowed only for secondary benchmarks."
            raise ValueError(msg)
        if self.target_namespace == "phase2" and (
            family not in PRIMARY_FAMILY_STRATA or stratum not in PRIMARY_FAMILY_STRATA[family]
        ):
            msg = "Phase II target family/stratum must match the preregistered population."
            raise ValueError(msg)
        if self.data_role in {"screening_selection", "confirmatory"} and qubit_count != 6:
            msg = "Primary screening-selection and confirmatory pipelines require q=6 targets."
            raise ValueError(msg)
        if self.data_role == "secondary_benchmark" and self.target_namespace == "phase2" and qubit_count != 12:
            msg = "Phase II secondary-benchmark pipelines require q=12 targets."
            raise ValueError(msg)
        scope = self.template.target_scope_id
        if scope == "primary_q6" and (self.target_namespace != "phase2" or qubit_count != 6):
            msg = "primary_q6 templates resolve only to q=6 Phase II targets."
            raise ValueError(msg)
        if scope == "secondary_q12" and (
            self.target_namespace != "phase2" or qubit_count != 12 or self.data_role != "secondary_benchmark"
        ):
            msg = "secondary_q12 templates resolve only to Phase II secondary q=12 targets."
            raise ValueError(msg)
        if scope == "legacy_reproduction" and (
            self.target_namespace != "legacy_reproduction" or self.data_role != "secondary_benchmark"
        ):
            msg = "legacy_reproduction templates resolve only to isolated secondary fixtures."
            raise ValueError(msg)
        if scope == "phase1_fixture" and (
            self.target_namespace != "phase1_fixture" or self.data_role != "secondary_benchmark"
        ):
            msg = "phase1_fixture templates resolve only to immutable Phase I secondary fixtures."
            raise ValueError(msg)
        if self.target_namespace == "phase1_fixture" and scope != "phase1_fixture":
            msg = "Phase I fixtures require the distinct phase1_fixture target scope."
            raise ValueError(msg)
        if self.target_namespace == "legacy_reproduction" and scope != "legacy_reproduction":
            msg = "Legacy fixtures require the distinct legacy_reproduction target scope."
            raise ValueError(msg)

        stages = tuple(self.stages)
        if len(stages) != len(self.template.stages) or not all(
            isinstance(stage, TrainingStageConfig) for stage in stages
        ):
            msg = "stages must contain exactly one TrainingStageConfig per template stage."
            raise TypeError(msg)
        expected_stages = tuple(
            stage_template.resolve_recorded(
                optimization_seed=optimization_seed,
                seed_domains=self.template.seed_domains,
                resolution_context_checksum=self.template.resolution_context_checksum(
                    stage_index=stage_template.stage_index,
                    target_instance_id=target_id,
                    optimization_block_id=block_id,
                ),
                recorded_stage=stages[0] if stage_template.stage_index == 0 else None,
            )
            for stage_template in self.template.stages
        )
        if stages != expected_stages:
            msg = "Concrete stages are not the deterministic resolution of the candidate template."
            raise ValueError(msg)
        runtime_seeds = [
            seed
            for stage in stages
            for seed in (
                stage.initialization_seed,
                stage.optimizer_seed,
                stage.training_seed,
                stage.checkpoint_validation.seed,
            )
            if seed is not None
        ]
        if self.method_id == "layerwise_bmpd_crn_legacy_v1":
            try:
                historical_target_seed = int(target_id.removeprefix("legacy_tfim_seed_"))
            except ValueError as error:
                msg = "Historical layerwise target identifiers must end in their decimal target seed."
                raise ValueError(msg) from error
            if optimization_seed != historical_target_seed:
                msg = "Historical layerwise reproduction uses the target seed as its exact outer seed."
                raise ValueError(msg)
            expected_runtime_seeds = [
                20 * historical_target_seed,
                30 * historical_target_seed,
                20 * historical_target_seed + 2,
                30 * historical_target_seed,
                20 * historical_target_seed + 3,
                30 * historical_target_seed,
                20 * historical_target_seed + 4,
                30 * historical_target_seed,
                40 * historical_target_seed,
                40 * historical_target_seed,
            ]
            if runtime_seeds != expected_runtime_seeds:
                msg = "Resolved historical layerwise seeds differ from the archived target-seed arithmetic."
                raise ValueError(msg)
        elif len(runtime_seeds) != len(set(runtime_seeds)):
            msg = "Resolved initialization, optimizer, training, and validation seeds must be distinct."
            raise ValueError(msg)
        object.__setattr__(self, "target_instance_id", target_id)
        object.__setattr__(self, "target_population_manifest_checksum", manifest_checksum)
        object.__setattr__(self, "target_instance_spec_checksum", spec_checksum)
        object.__setattr__(self, "target_family_id", family)
        object.__setattr__(self, "target_stratum_id", stratum)
        object.__setattr__(self, "qubit_count", qubit_count)
        object.__setattr__(self, "optimization_block_id", block_id)
        object.__setattr__(self, "optimization_seed", optimization_seed)
        object.__setattr__(self, "stages", stages)

    @property
    def template_checksum(self) -> str:
        """Target-independent candidate checksum used by WP15 screening."""
        return self.template.configuration_checksum

    @property
    def ansatz_family(self) -> str:
        """Resolved ansatz family from the immutable candidate template."""
        return self.template.ansatz_family

    @property
    def method_id(self) -> str:
        """Resolved method identity from the immutable candidate template."""
        return self.template.method_id

    @property
    def method_version(self) -> str:
        """Resolved method version from the immutable candidate template."""
        return self.template.method_version

    @property
    def method_family_id(self) -> str:
        """Derived method comparison family."""
        return self.template.method_family_id

    @property
    def seed_domains(self) -> Mapping[str, object]:
        """Frozen random-stream domains from the candidate template."""
        return self.template.seed_domains

    @property
    def final_materialization_policy(self) -> Mapping[str, object]:
        """Final materialization policy from the candidate template."""
        return self.template.final_materialization_policy

    def identity_payload(self) -> dict[str, object]:
        """Return all concrete training fields and no output or final-test field."""
        return {
            "identity_version": PIPELINE_TRAINING_IDENTITY_VERSION,
            "config_schema_version": self.schema_version,
            "template_schema_version": self.template.schema_version,
            "template_checksum": self.template_checksum,
            "target_namespace": self.target_namespace,
            "target_instance_id": self.target_instance_id,
            "target_population_manifest_checksum": self.target_population_manifest_checksum,
            "target_instance_spec_checksum": self.target_instance_spec_checksum,
            "target_family_id": self.target_family_id,
            "target_stratum_id": self.target_stratum_id,
            "qubit_count": self.qubit_count,
            "optimization_block_id": self.optimization_block_id,
            "optimization_seed": self.optimization_seed,
            "stages": [stage.identity_payload() for stage in self.stages],
            "data_role": self.data_role,
        }

    @property
    def training_id(self) -> str:
        """Stable identity shared by all final-test fan-out evaluations."""
        digest = hashlib.sha256(canonical_json(self.identity_payload()).encode()).hexdigest()
        return f"{TRAINING_ID_PREFIX}{digest}"

    @property
    def configuration_checksum(self) -> str:
        """Checksum of the concrete target/optimization training configuration."""
        return canonical_checksum(self.identity_payload())

    @property
    def matching_projection_checksum(self) -> str:
        """Target-independent matching projection from the candidate template."""
        return self.template.matching_projection_checksum

    def prefix_id(self, stage_index: int) -> str:
        """Return the compatible pipeline-prefix identity through one stage."""
        index = require_int(stage_index, "stage_index", minimum=0)
        if index >= len(self.stages):
            msg = "stage_index does not exist in this pipeline."
            raise ValueError(msg)
        matched_pair = self.method_id in {
            "layerwise_bmpd_crn_v2",
            "layerwise_bmpd_noiseless",
        }
        payload = {
            "identity_version": PIPELINE_PREFIX_IDENTITY_VERSION,
            "config_schema_version": self.schema_version,
            "preregistration_checksum": self.template.preregistration_checksum,
            "target_namespace": self.target_namespace,
            "target_instance_id": self.target_instance_id,
            "target_population_manifest_checksum": self.target_population_manifest_checksum,
            "target_instance_spec_checksum": self.target_instance_spec_checksum,
            "target_family_id": self.target_family_id,
            "target_stratum_id": self.target_stratum_id,
            "qubit_count": self.qubit_count,
            "optimization_block_id": self.optimization_block_id,
            "optimization_seed": self.optimization_seed,
            "ansatz_family": self.ansatz_family,
            "method_id": ("layerwise_bmpd_matched_v2" if matched_pair else self.method_id),
            "method_version": "matched" if matched_pair else self.method_version,
            "resource_stratum_id": self.template.resource_stratum_id,
            "stages": [stage.identity_payload() for stage in self.stages[: index + 1]],
            "data_role": self.data_role,
        }
        digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        return f"{PIPELINE_PREFIX}{digest}"

    def to_dict(self) -> dict[str, object]:
        """Return a detached exact-schema concrete training configuration."""
        return {
            "schema_version": self.schema_version,
            "template": self.template.to_dict(),
            "template_checksum": self.template_checksum,
            "target_ref": None if self.target_ref is None else self.target_ref.to_dict(),
            "target_namespace": self.target_namespace,
            "target_instance_id": self.target_instance_id,
            "target_population_manifest_checksum": self.target_population_manifest_checksum,
            "target_instance_spec_checksum": self.target_instance_spec_checksum,
            "target_family_id": self.target_family_id,
            "target_stratum_id": self.target_stratum_id,
            "qubit_count": self.qubit_count,
            "optimization_block_id": self.optimization_block_id,
            "optimization_seed": self.optimization_seed,
            "stages": [stage.to_dict() for stage in self.stages],
            "data_role": self.data_role,
            "training_id": self.training_id,
            "matching_projection_checksum": self.matching_projection_checksum,
            "configuration_checksum": self.configuration_checksum,
        }

    @classmethod
    def from_dict(cls, data: object) -> TrainingPipelineConfig:
        """Construct and identity-verify a strict concrete pipeline."""
        mapping = require_mapping(data, "training pipeline config")
        require_exact_keys(mapping, _PIPELINE_CONFIG_KEYS, "training pipeline config")
        if mapping["schema_version"] != TRAINING_PIPELINE_CONFIG_SCHEMA_VERSION:
            msg = f"schema_version must be {TRAINING_PIPELINE_CONFIG_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        stage_data = mapping["stages"]
        if isinstance(stage_data, (str, bytes)) or not isinstance(stage_data, Sequence):
            _raise_type_error("stages", "a sequence", stage_data)
        pipeline = cls(
            template=TrainingPipelineTemplate.from_dict(mapping["template"]),
            target_ref=None if mapping["target_ref"] is None else Phase2TargetRef.from_dict(mapping["target_ref"]),
            target_namespace=cast(
                "Literal['phase2', 'phase1_fixture', 'legacy_reproduction']",
                mapping["target_namespace"],
            ),
            target_instance_id=cast("str", mapping["target_instance_id"]),
            target_population_manifest_checksum=cast("str", mapping["target_population_manifest_checksum"]),
            target_instance_spec_checksum=cast("str", mapping["target_instance_spec_checksum"]),
            target_family_id=cast("str", mapping["target_family_id"]),
            target_stratum_id=cast("str", mapping["target_stratum_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            optimization_block_id=cast("str", mapping["optimization_block_id"]),
            optimization_seed=cast("int", mapping["optimization_seed"]),
            stages=tuple(TrainingStageConfig.from_dict(item) for item in stage_data),
            data_role=cast("DataRole", mapping["data_role"]),
        )
        expected = {
            "template_checksum": pipeline.template_checksum,
            "training_id": pipeline.training_id,
            "matching_projection_checksum": pipeline.matching_projection_checksum,
            "configuration_checksum": pipeline.configuration_checksum,
        }
        for name, value in expected.items():
            if mapping[name] != value:
                msg = f"Serialized {name} does not match the derived pipeline identity."
                raise ValueError(msg)
        return pipeline

    def to_json(self) -> str:
        """Return canonical JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> TrainingPipelineConfig:
        """Construct a concrete pipeline from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))


def _checkpoint_provenance_checksum(
    *,
    pipeline_prefix_id: str,
    stage_id: str,
    stage_configuration_checksum: str,
    input_checkpoint_checksum: str | None,
    input_checkpoint_provenance_checksum: str | None,
    produced_checkpoint_checksum: str,
) -> str:
    """Derive immutable checkpoint provenance from its complete stage chain."""
    return canonical_checksum({
        "pipeline_prefix_id": pipeline_prefix_id,
        "stage_id": stage_id,
        "stage_configuration_checksum": stage_configuration_checksum,
        "input_checkpoint_checksum": input_checkpoint_checksum,
        "input_checkpoint_provenance_checksum": input_checkpoint_provenance_checksum,
        "produced_checkpoint_checksum": produced_checkpoint_checksum,
    })


@dataclass(frozen=True, slots=True)
class TrainingStageResult:
    """Lossless result and artifact ledger for one configured stage."""

    pipeline_training_id: str
    pipeline_prefix_id: str
    stage_index: int
    stage_id: str
    stage_configuration_checksum: str
    input_checkpoint_checksum: str | None
    input_checkpoint_provenance_checksum: str | None
    produced_checkpoint_path: str
    produced_checkpoint_checksum: str
    checkpoint_provenance_checksum: str
    output_topology_id: str
    output_parameter_count: int
    training_summary: Mapping[str, object]
    checkpoint_validation_summary: Mapping[str, object] | None
    training_ensemble_checksums: tuple[str, ...]
    checkpoint_validation_ensemble_checksum: str | None
    optimizer_trace_path: str | None
    optimizer_trace_checksum: str | None
    diagnostic_sidecar_path: str | None
    diagnostic_sidecar_checksum: str | None
    wall_time_seconds: float
    peak_memory_bytes: int
    normalized_work: Mapping[str, object]
    schema_version: str = field(default=TRAINING_STAGE_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate stage artifacts, summaries, identities, and work."""
        training_id = _require_identifier(
            self.pipeline_training_id,
            "pipeline_training_id",
            _TRAINING_ID_PATTERN,
        )
        prefix = _require_identifier(self.pipeline_prefix_id, "pipeline_prefix_id", _PIPELINE_PREFIX_PATTERN)
        index = require_int(self.stage_index, "stage_index", minimum=0)
        stage_id = require_slug(self.stage_id, "stage_id")
        stage_checksum = require_checksum(self.stage_configuration_checksum, "stage_configuration_checksum")
        input_checksum = _require_optional_checksum(
            self.input_checkpoint_checksum,
            "input_checkpoint_checksum",
        )
        input_provenance = _require_optional_checksum(
            self.input_checkpoint_provenance_checksum,
            "input_checkpoint_provenance_checksum",
        )
        if (input_checksum is None) != (input_provenance is None):
            msg = "Input checkpoint checksum and provenance must either both be present or both be absent."
            raise ValueError(msg)
        checkpoint_path, checkpoint_checksum = _validate_pair(
            self.produced_checkpoint_path,
            self.produced_checkpoint_checksum,
            path_name="produced_checkpoint_path",
            checksum_name="produced_checkpoint_checksum",
            required=True,
        )
        provenance = require_checksum(self.checkpoint_provenance_checksum, "checkpoint_provenance_checksum")
        expected_provenance = _checkpoint_provenance_checksum(
            pipeline_prefix_id=prefix,
            stage_id=stage_id,
            stage_configuration_checksum=stage_checksum,
            input_checkpoint_checksum=input_checksum,
            input_checkpoint_provenance_checksum=input_provenance,
            produced_checkpoint_checksum=cast("str", checkpoint_checksum),
        )
        if provenance != expected_provenance:
            msg = "Checkpoint provenance checksum does not match its pipeline prefix and content."
            raise ValueError(msg)
        output_topology = require_slug(self.output_topology_id, "output_topology_id")
        output_count = require_int(self.output_parameter_count, "output_parameter_count", minimum=1)
        training_summary = _validate_summary(self.training_summary, "training_summary")
        validation_summary = (
            None
            if self.checkpoint_validation_summary is None
            else _validate_summary(self.checkpoint_validation_summary, "checkpoint_validation_summary")
        )
        ensembles = tuple(
            require_checksum(value, f"training_ensemble_checksums[{index}]")
            for index, value in enumerate(self.training_ensemble_checksums)
        )
        if len(ensembles) != len(set(ensembles)):
            msg = "training_ensemble_checksums must not contain duplicates."
            raise ValueError(msg)
        validation_ensemble = _require_optional_checksum(
            self.checkpoint_validation_ensemble_checksum,
            "checkpoint_validation_ensemble_checksum",
        )
        trace_path, trace_checksum = _validate_pair(
            self.optimizer_trace_path,
            self.optimizer_trace_checksum,
            path_name="optimizer_trace_path",
            checksum_name="optimizer_trace_checksum",
        )
        diagnostic_path, diagnostic_checksum = _validate_pair(
            self.diagnostic_sidecar_path,
            self.diagnostic_sidecar_checksum,
            path_name="diagnostic_sidecar_path",
            checksum_name="diagnostic_sidecar_checksum",
        )
        wall_time = require_float(self.wall_time_seconds, "wall_time_seconds", minimum=0.0)
        peak_memory = require_int(self.peak_memory_bytes, "peak_memory_bytes", minimum=0)
        work = _validate_normalized_work(self.normalized_work, "normalized_work")

        object.__setattr__(self, "pipeline_training_id", training_id)
        object.__setattr__(self, "pipeline_prefix_id", prefix)
        object.__setattr__(self, "stage_index", index)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "stage_configuration_checksum", stage_checksum)
        object.__setattr__(self, "input_checkpoint_checksum", input_checksum)
        object.__setattr__(self, "input_checkpoint_provenance_checksum", input_provenance)
        object.__setattr__(self, "produced_checkpoint_path", cast("str", checkpoint_path))
        object.__setattr__(self, "produced_checkpoint_checksum", cast("str", checkpoint_checksum))
        object.__setattr__(self, "checkpoint_provenance_checksum", provenance)
        object.__setattr__(self, "output_topology_id", output_topology)
        object.__setattr__(self, "output_parameter_count", output_count)
        object.__setattr__(self, "training_summary", training_summary)
        object.__setattr__(self, "checkpoint_validation_summary", validation_summary)
        object.__setattr__(self, "training_ensemble_checksums", ensembles)
        object.__setattr__(self, "checkpoint_validation_ensemble_checksum", validation_ensemble)
        object.__setattr__(self, "optimizer_trace_path", trace_path)
        object.__setattr__(self, "optimizer_trace_checksum", trace_checksum)
        object.__setattr__(self, "diagnostic_sidecar_path", diagnostic_path)
        object.__setattr__(self, "diagnostic_sidecar_checksum", diagnostic_checksum)
        object.__setattr__(self, "wall_time_seconds", wall_time)
        object.__setattr__(self, "peak_memory_bytes", peak_memory)
        object.__setattr__(self, "normalized_work", work)

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete stage-result record, including artifact paths."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered stage-result content."""
        return {
            "schema_version": self.schema_version,
            "pipeline_training_id": self.pipeline_training_id,
            "pipeline_prefix_id": self.pipeline_prefix_id,
            "stage_index": self.stage_index,
            "stage_id": self.stage_id,
            "stage_configuration_checksum": self.stage_configuration_checksum,
            "input_checkpoint_checksum": self.input_checkpoint_checksum,
            "input_checkpoint_provenance_checksum": self.input_checkpoint_provenance_checksum,
            "produced_checkpoint_path": self.produced_checkpoint_path,
            "produced_checkpoint_checksum": self.produced_checkpoint_checksum,
            "checkpoint_provenance_checksum": self.checkpoint_provenance_checksum,
            "output_topology_id": self.output_topology_id,
            "output_parameter_count": self.output_parameter_count,
            "training_summary": thaw_json_mapping(self.training_summary),
            "checkpoint_validation_summary": (
                None
                if self.checkpoint_validation_summary is None
                else thaw_json_mapping(self.checkpoint_validation_summary)
            ),
            "training_ensemble_checksums": list(self.training_ensemble_checksums),
            "checkpoint_validation_ensemble_checksum": self.checkpoint_validation_ensemble_checksum,
            "optimizer_trace_path": self.optimizer_trace_path,
            "optimizer_trace_checksum": self.optimizer_trace_checksum,
            "diagnostic_sidecar_path": self.diagnostic_sidecar_path,
            "diagnostic_sidecar_checksum": self.diagnostic_sidecar_checksum,
            "wall_time_seconds": self.wall_time_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "normalized_work": thaw_json_mapping(self.normalized_work),
        }

    def to_dict(self) -> dict[str, object]:
        """Return a sealed stage-result mapping."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> TrainingStageResult:
        """Construct and seal-verify a strict stage result."""
        mapping = verify_sealed_mapping(data, expected_keys=_STAGE_RESULT_KEYS, name="training stage result")
        if mapping["schema_version"] != TRAINING_STAGE_RESULT_SCHEMA_VERSION:
            msg = f"schema_version must be {TRAINING_STAGE_RESULT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        ensembles = mapping["training_ensemble_checksums"]
        if isinstance(ensembles, (str, bytes)) or not isinstance(ensembles, Sequence):
            _raise_type_error("training_ensemble_checksums", "a sequence", ensembles)
        result = cls(
            pipeline_training_id=cast("str", mapping["pipeline_training_id"]),
            pipeline_prefix_id=cast("str", mapping["pipeline_prefix_id"]),
            stage_index=cast("int", mapping["stage_index"]),
            stage_id=cast("str", mapping["stage_id"]),
            stage_configuration_checksum=cast("str", mapping["stage_configuration_checksum"]),
            input_checkpoint_checksum=cast("str | None", mapping["input_checkpoint_checksum"]),
            input_checkpoint_provenance_checksum=cast(
                "str | None",
                mapping["input_checkpoint_provenance_checksum"],
            ),
            produced_checkpoint_path=cast("str", mapping["produced_checkpoint_path"]),
            produced_checkpoint_checksum=cast("str", mapping["produced_checkpoint_checksum"]),
            checkpoint_provenance_checksum=cast("str", mapping["checkpoint_provenance_checksum"]),
            output_topology_id=cast("str", mapping["output_topology_id"]),
            output_parameter_count=cast("int", mapping["output_parameter_count"]),
            training_summary=cast("Mapping[str, object]", mapping["training_summary"]),
            checkpoint_validation_summary=cast(
                "Mapping[str, object] | None",
                mapping["checkpoint_validation_summary"],
            ),
            training_ensemble_checksums=cast("tuple[str, ...]", ensembles),
            checkpoint_validation_ensemble_checksum=cast(
                "str | None",
                mapping["checkpoint_validation_ensemble_checksum"],
            ),
            optimizer_trace_path=cast("str | None", mapping["optimizer_trace_path"]),
            optimizer_trace_checksum=cast("str | None", mapping["optimizer_trace_checksum"]),
            diagnostic_sidecar_path=cast("str | None", mapping["diagnostic_sidecar_path"]),
            diagnostic_sidecar_checksum=cast("str | None", mapping["diagnostic_sidecar_checksum"]),
            wall_time_seconds=cast("float", mapping["wall_time_seconds"]),
            peak_memory_bytes=cast("int", mapping["peak_memory_bytes"]),
            normalized_work=cast("Mapping[str, object]", mapping["normalized_work"]),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Training stage result checksum changed during normalization."
            raise ValueError(msg)
        return result

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> TrainingStageResult:
        """Construct a stage result from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))


def _validate_stage_result_against_config(
    *,
    result: TrainingStageResult,
    stage: TrainingStageConfig,
    pipeline: TrainingPipelineConfig,
    predecessor: TrainingStageResult | None,
) -> None:
    """Cross-check one stage result against its pipeline and predecessor."""
    expected = {
        "pipeline_training_id": pipeline.training_id,
        "pipeline_prefix_id": pipeline.prefix_id(stage.stage_index),
        "stage_index": stage.stage_index,
        "stage_id": stage.stage_id,
        "stage_configuration_checksum": stage.configuration_checksum,
        "output_topology_id": stage.output_topology_id,
        "output_parameter_count": stage.output_parameter_count,
    }
    for name, value in expected.items():
        if getattr(result, name) != value:
            msg = f"Stage result {name} does not match its configured pipeline."
            raise ValueError(msg)
    if predecessor is None:
        expected_input_checksum = stage.input_checkpoint_checksum
        expected_input_provenance = stage.input_checkpoint_provenance_checksum
        if stage.input_checkpoint_pipeline_prefix is not None and (
            stage.input_checkpoint_pipeline_prefix == result.pipeline_prefix_id
        ):
            msg = "An external checkpoint cannot claim the pipeline prefix that consumes it."
            raise ValueError(msg)
    else:
        expected_input_checksum = predecessor.produced_checkpoint_checksum
        expected_input_provenance = predecessor.checkpoint_provenance_checksum
    if (
        result.input_checkpoint_checksum != expected_input_checksum
        or result.input_checkpoint_provenance_checksum != expected_input_provenance
    ):
        msg = "Stage result input checkpoint checksum or provenance does not match its predecessor."
        raise ValueError(msg)

    expected_training_ensembles = 0
    if stage.sampling_policy == "crn_fixed":
        expected_training_ensembles = 1
    elif stage.sampling_policy == "crn_refresh":
        expected_training_ensembles = math.ceil(stage.iteration_budget / cast("int", stage.crn_refresh_interval))
    if len(result.training_ensemble_checksums) != expected_training_ensembles:
        msg = "Training ensemble checksums do not match the configured CRN policy."
        raise ValueError(msg)
    validation_requires_ensemble = (
        stage.checkpoint_validation.enabled
        and stage.checkpoint_validation.sampling_policy in {"crn_fixed", "crn_refresh"}
    )
    if validation_requires_ensemble != (result.checkpoint_validation_ensemble_checksum is not None):
        msg = "Checkpoint-validation ensemble checksum does not match the configured sampling policy."
        raise ValueError(msg)
    if stage.checkpoint_validation.enabled != (result.checkpoint_validation_summary is not None):
        msg = "Checkpoint-validation summary presence does not match the configured policy."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TrainingPipelineResult:
    """Complete, provenance-checked result of one staged training pipeline."""

    config: TrainingPipelineConfig
    stage_results: tuple[TrainingStageResult, ...]
    final_checkpoint_path: str
    final_checkpoint_checksum: str
    final_checkpoint_provenance_checksum: str
    wall_time_seconds: float
    peak_memory_bytes: int
    normalized_work: Mapping[str, object]
    schema_version: str = field(default=TRAINING_PIPELINE_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate exact stage order, checkpoint chain, and aggregate aliases."""
        if not isinstance(self.config, TrainingPipelineConfig):
            _raise_type_error("config", "a TrainingPipelineConfig", self.config)
        results = tuple(self.stage_results)
        if len(results) != len(self.config.stages) or not all(
            isinstance(result, TrainingStageResult) for result in results
        ):
            msg = "stage_results must contain exactly one TrainingStageResult per configured stage."
            raise TypeError(msg)
        predecessor: TrainingStageResult | None = None
        for stage, result in zip(self.config.stages, results, strict=True):
            _validate_stage_result_against_config(
                result=result,
                stage=stage,
                pipeline=self.config,
                predecessor=predecessor,
            )
            predecessor = result
        final = results[-1]
        final_path = require_relative_path(self.final_checkpoint_path, "final_checkpoint_path")
        final_checksum = require_checksum(self.final_checkpoint_checksum, "final_checkpoint_checksum")
        final_provenance = require_checksum(
            self.final_checkpoint_provenance_checksum,
            "final_checkpoint_provenance_checksum",
        )
        if (
            final_path != final.produced_checkpoint_path
            or final_checksum != final.produced_checkpoint_checksum
            or final_provenance != final.checkpoint_provenance_checksum
        ):
            msg = "Final checkpoint aliases must exactly identify the last stage checkpoint."
            raise ValueError(msg)
        wall_time = require_float(self.wall_time_seconds, "wall_time_seconds", minimum=0.0)
        expected_wall_time = sum(result.wall_time_seconds for result in results)
        if wall_time != expected_wall_time:
            msg = "Pipeline wall time must equal the sum of stage wall times."
            raise ValueError(msg)
        peak_memory = require_int(self.peak_memory_bytes, "peak_memory_bytes", minimum=0)
        if peak_memory != max(result.peak_memory_bytes for result in results):
            msg = "Pipeline peak memory must equal the maximum stage peak memory."
            raise ValueError(msg)
        work = _validate_normalized_work(self.normalized_work, "normalized_work")
        if work != _sum_work([result.normalized_work for result in results]):
            msg = "Pipeline normalized work must equal the component-wise sum of stage work."
            raise ValueError(msg)
        object.__setattr__(self, "stage_results", results)
        object.__setattr__(self, "final_checkpoint_path", final_path)
        object.__setattr__(self, "final_checkpoint_checksum", final_checksum)
        object.__setattr__(self, "final_checkpoint_provenance_checksum", final_provenance)
        object.__setattr__(self, "wall_time_seconds", wall_time)
        object.__setattr__(self, "peak_memory_bytes", peak_memory)
        object.__setattr__(self, "normalized_work", work)

    @property
    def training_id(self) -> str:
        """Stable scientific training identity."""
        return self.config.training_id

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete pipeline result and artifact ledger."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered pipeline-result content."""
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "stage_results": [result.to_dict() for result in self.stage_results],
            "final_checkpoint_path": self.final_checkpoint_path,
            "final_checkpoint_checksum": self.final_checkpoint_checksum,
            "final_checkpoint_provenance_checksum": self.final_checkpoint_provenance_checksum,
            "wall_time_seconds": self.wall_time_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "normalized_work": thaw_json_mapping(self.normalized_work),
        }

    def to_dict(self) -> dict[str, object]:
        """Return a sealed pipeline-result mapping."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> TrainingPipelineResult:
        """Construct and seal-verify a strict pipeline result."""
        mapping = verify_sealed_mapping(data, expected_keys=_PIPELINE_RESULT_KEYS, name="training pipeline result")
        if mapping["schema_version"] != TRAINING_PIPELINE_RESULT_SCHEMA_VERSION:
            msg = f"schema_version must be {TRAINING_PIPELINE_RESULT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        stage_data = mapping["stage_results"]
        if isinstance(stage_data, (str, bytes)) or not isinstance(stage_data, Sequence):
            _raise_type_error("stage_results", "a sequence", stage_data)
        result = cls(
            config=TrainingPipelineConfig.from_dict(mapping["config"]),
            stage_results=tuple(TrainingStageResult.from_dict(item) for item in stage_data),
            final_checkpoint_path=cast("str", mapping["final_checkpoint_path"]),
            final_checkpoint_checksum=cast("str", mapping["final_checkpoint_checksum"]),
            final_checkpoint_provenance_checksum=cast(
                "str",
                mapping["final_checkpoint_provenance_checksum"],
            ),
            wall_time_seconds=cast("float", mapping["wall_time_seconds"]),
            peak_memory_bytes=cast("int", mapping["peak_memory_bytes"]),
            normalized_work=cast("Mapping[str, object]", mapping["normalized_work"]),
        )
        if result.content_checksum != mapping["content_checksum"]:
            msg = "Training pipeline result checksum changed during normalization."
            raise ValueError(msg)
        return result

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> TrainingPipelineResult:
        """Construct a pipeline result from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class ExternalCheckpointRef:
    """Typed, producer-verified reference to one completed stage checkpoint."""

    producer_result: TrainingPipelineResult
    producer_stage_index: int
    schema_version: str = field(default=EXTERNAL_CHECKPOINT_REF_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the producer result and select an existing stage mechanically."""
        if not isinstance(self.producer_result, TrainingPipelineResult):
            _raise_type_error("producer_result", "a TrainingPipelineResult", self.producer_result)
        index = require_int(self.producer_stage_index, "producer_stage_index", minimum=0)
        if index >= len(self.producer_result.stage_results):
            msg = "producer_stage_index does not exist in the producer pipeline result."
            raise ValueError(msg)
        object.__setattr__(self, "producer_stage_index", index)

    @classmethod
    def from_pipeline_result(
        cls,
        producer_result: TrainingPipelineResult,
        producer_stage_index: int,
    ) -> ExternalCheckpointRef:
        """Derive a reference whose prefix and provenance come from the producer."""
        return cls(producer_result=producer_result, producer_stage_index=producer_stage_index)

    @property
    def _stage_result(self) -> TrainingStageResult:
        """Selected validated producer-stage result."""
        return self.producer_result.stage_results[self.producer_stage_index]

    @property
    def producer_pipeline_prefix_id(self) -> str:
        """Mechanically recomputed compatible producer prefix."""
        return self.producer_result.config.prefix_id(self.producer_stage_index)

    @property
    def produced_checkpoint_checksum(self) -> str:
        """Content checksum of the selected checkpoint artifact."""
        return self._stage_result.produced_checkpoint_checksum

    @property
    def checkpoint_provenance_checksum(self) -> str:
        """Path-independent provenance checksum of the selected checkpoint."""
        return self._stage_result.checkpoint_provenance_checksum

    @property
    def output_topology_id(self) -> str:
        """Topology produced by the referenced stage."""
        return self._stage_result.output_topology_id

    @property
    def output_parameter_count(self) -> int:
        """Parameter count produced by the referenced stage."""
        return self._stage_result.output_parameter_count

    def _identity_dict(self) -> dict[str, object]:
        """Return path-, outcome-, and future-suffix-free producer provenance."""
        stage = self._stage_result
        return {
            "schema_version": self.schema_version,
            "producer_pipeline_prefix_id": self.producer_pipeline_prefix_id,
            "producer_stage_index": self.producer_stage_index,
            "producer_stage_id": stage.stage_id,
            "producer_stage_configuration_checksum": stage.stage_configuration_checksum,
            "produced_checkpoint_checksum": self.produced_checkpoint_checksum,
            "checkpoint_provenance_checksum": self.checkpoint_provenance_checksum,
            "output_topology_id": self.output_topology_id,
            "output_parameter_count": self.output_parameter_count,
        }

    @property
    def provenance_ref_checksum(self) -> str:
        """Stable path-free identity used by a consuming training stage."""
        return canonical_checksum(self._identity_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return the complete producer-backed artifact reference."""
        return {
            "schema_version": self.schema_version,
            "producer_result": self.producer_result.to_dict(),
            "producer_stage_index": self.producer_stage_index,
            "provenance_ref_checksum": self.provenance_ref_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete reference including its producer audit root."""
        return canonical_checksum(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        """Return a sealed producer-backed reference."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> ExternalCheckpointRef:
        """Construct and checksum-verify a producer-backed reference."""
        mapping = verify_sealed_mapping(
            data,
            expected_keys=_EXTERNAL_CHECKPOINT_REF_KEYS,
            name="external checkpoint reference",
        )
        if mapping["schema_version"] != EXTERNAL_CHECKPOINT_REF_SCHEMA_VERSION:
            msg = f"schema_version must be {EXTERNAL_CHECKPOINT_REF_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        reference = cls(
            producer_result=TrainingPipelineResult.from_dict(mapping["producer_result"]),
            producer_stage_index=cast("int", mapping["producer_stage_index"]),
        )
        if mapping["provenance_ref_checksum"] != reference.provenance_ref_checksum:
            msg = "External checkpoint provenance reference does not match its producer."
            raise ValueError(msg)
        if mapping["content_checksum"] != reference.content_checksum:
            msg = "External checkpoint reference checksum changed during normalization."
            raise ValueError(msg)
        return reference


def _derive_materialized_circuit_id(
    *,
    pipeline_training_id: str,
    final_checkpoint_checksum: str,
    final_materialization_policy_checksum: str,
    materialized_circuit_checksum: str,
) -> str:
    """Derive the stable identity of one materialized trained circuit."""
    digest = hashlib.sha256(
        canonical_json({
            "identity_version": "yaqs.state_preparation.phase2.materialized_circuit_identity.v1",
            "pipeline_training_id": pipeline_training_id,
            "final_checkpoint_checksum": final_checkpoint_checksum,
            "final_materialization_policy_checksum": final_materialization_policy_checksum,
            "materialized_circuit_checksum": materialized_circuit_checksum,
        }).encode()
    ).hexdigest()
    return f"{MATERIALIZED_CIRCUIT_ID_PREFIX}{digest}"


@dataclass(frozen=True, slots=True)
class PipelineEvaluationConfig:
    """One fixed final-test fan-out cell for a trained pipeline."""

    pipeline_training_id: str
    pipeline_configuration_checksum: str
    pipeline_result_checksum: str
    final_checkpoint_checksum: str
    final_materialization_policy_checksum: str
    data_role: Literal[
        "development",
        "checkpoint_validation",
        "screening_selection",
        "secondary_benchmark",
        "confirmatory",
    ]
    materialized_circuit_id: str
    materialized_circuit_checksum: str
    test_noise_id: str
    noise_definition_version: str
    noise_strength_scale: float | None
    tjm_dt: float | None
    evaluation_seed: int | None
    evaluation_seed_domain: str | None
    repetition: int
    trajectory_budget: int
    evaluation_policy: Literal["fixed_sample", "confidence_interval"]
    confidence_level: float | None
    confidence_interval_method: str | None
    sidecar_storage_policy: Literal["none", "trajectory_fidelities"]
    max_bond_dimension: int | None
    svd_threshold: float
    truncation_mode: Literal["discarded_weight", "relative"]
    min_bond_dimension: int
    schema_version: str = field(default=PIPELINE_EVALUATION_CONFIG_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate final-test policy without introducing optional stopping."""
        training_id = _require_identifier(
            self.pipeline_training_id,
            "pipeline_training_id",
            _TRAINING_ID_PATTERN,
        )
        pipeline_checksum = require_checksum(
            self.pipeline_configuration_checksum,
            "pipeline_configuration_checksum",
        )
        result_checksum = require_checksum(self.pipeline_result_checksum, "pipeline_result_checksum")
        final_checkpoint_checksum = require_checksum(
            self.final_checkpoint_checksum,
            "final_checkpoint_checksum",
        )
        materialization_checksum = require_checksum(
            self.final_materialization_policy_checksum,
            "final_materialization_policy_checksum",
        )
        if self.data_role not in DATA_ROLES:
            msg = f"data_role must be one of {DATA_ROLES!r}."
            raise ValueError(msg)
        circuit_checksum = require_checksum(
            self.materialized_circuit_checksum,
            "materialized_circuit_checksum",
        )
        circuit_id = _require_identifier(
            self.materialized_circuit_id,
            "materialized_circuit_id",
            _MATERIALIZED_CIRCUIT_ID_PATTERN,
        )
        expected_circuit_id = _derive_materialized_circuit_id(
            pipeline_training_id=training_id,
            final_checkpoint_checksum=final_checkpoint_checksum,
            final_materialization_policy_checksum=materialization_checksum,
            materialized_circuit_checksum=circuit_checksum,
        )
        if circuit_id != expected_circuit_id:
            msg = "materialized_circuit_id is not derived from the training, checkpoint, policy, and circuit."
            raise ValueError(msg)
        noise_id, noise_version, scale, dt = _validate_noise(
            noise_id=self.test_noise_id,
            definition_version=self.noise_definition_version,
            strength_scale=self.noise_strength_scale,
            tjm_dt=self.tjm_dt,
            name="evaluation",
            allow_ballarin=True,
        )
        evaluation_seed = _require_seed(self.evaluation_seed, "evaluation_seed")
        seed_domain = _require_optional_slug(self.evaluation_seed_domain, "evaluation_seed_domain")
        repetition = require_int(self.repetition, "repetition", minimum=0)
        budget = require_int(self.trajectory_budget, "trajectory_budget", minimum=0)
        if self.evaluation_policy not in EVALUATION_POLICIES:
            msg = f"evaluation_policy must be one of {EVALUATION_POLICIES!r}."
            raise ValueError(msg)
        confidence_level = _require_optional_float(
            self.confidence_level,
            "confidence_level",
            minimum=0.0,
            maximum=1.0,
        )
        confidence_method = (
            None
            if self.confidence_interval_method is None
            else require_slug(self.confidence_interval_method, "confidence_interval_method")
        )
        if self.sidecar_storage_policy not in SIDECAR_STORAGE_POLICIES:
            msg = f"sidecar_storage_policy must be one of {SIDECAR_STORAGE_POLICIES!r}."
            raise ValueError(msg)
        noiseless = noise_id == NOISELESS_NOISE_ID
        if noiseless and (
            budget != 0
            or evaluation_seed is not None
            or seed_domain is not None
            or self.evaluation_policy != "fixed_sample"
            or confidence_level is not None
            or confidence_method is not None
            or self.sidecar_storage_policy != "none"
        ):
            msg = "Noiseless evaluation requires zero trajectories, no seed/domain/CI, fixed sample, and no sidecar."
            raise ValueError(msg)
        if not noiseless and (budget == 0 or evaluation_seed is None or seed_domain is None):
            msg = "Noisy evaluation requires a positive fixed trajectory budget, seed, and seed domain."
            raise ValueError(msg)
        if self.evaluation_policy == "fixed_sample" and (confidence_level is not None or confidence_method is not None):
            msg = "Fixed-sample evaluation cannot request a confidence interval."
            raise ValueError(msg)
        if self.evaluation_policy == "confidence_interval" and (
            budget < 2 or confidence_level is None or not 0.0 < confidence_level < 1.0 or confidence_method is None
        ):
            msg = "Confidence-interval evaluation requires a fixed budget of at least two and a complete CI policy."
            raise ValueError(msg)
        maximum, threshold, mode, minimum = _validate_truncation(
            self.max_bond_dimension,
            self.svd_threshold,
            self.truncation_mode,
            self.min_bond_dimension,
        )
        object.__setattr__(self, "pipeline_training_id", training_id)
        object.__setattr__(self, "pipeline_configuration_checksum", pipeline_checksum)
        object.__setattr__(self, "pipeline_result_checksum", result_checksum)
        object.__setattr__(self, "final_checkpoint_checksum", final_checkpoint_checksum)
        object.__setattr__(
            self,
            "final_materialization_policy_checksum",
            materialization_checksum,
        )
        object.__setattr__(self, "materialized_circuit_id", circuit_id)
        object.__setattr__(self, "materialized_circuit_checksum", circuit_checksum)
        object.__setattr__(self, "test_noise_id", noise_id)
        object.__setattr__(self, "noise_definition_version", noise_version)
        object.__setattr__(self, "noise_strength_scale", scale)
        object.__setattr__(self, "tjm_dt", dt)
        object.__setattr__(self, "evaluation_seed", evaluation_seed)
        object.__setattr__(self, "evaluation_seed_domain", seed_domain)
        object.__setattr__(self, "repetition", repetition)
        object.__setattr__(self, "trajectory_budget", budget)
        object.__setattr__(self, "confidence_level", confidence_level)
        object.__setattr__(self, "confidence_interval_method", confidence_method)
        object.__setattr__(self, "max_bond_dimension", maximum)
        object.__setattr__(self, "svd_threshold", threshold)
        object.__setattr__(self, "truncation_mode", mode)
        object.__setattr__(self, "min_bond_dimension", minimum)

    def identity_payload(self) -> dict[str, object]:
        """Return evaluation-row identity fields, excluding every output path."""
        return {
            "identity_version": PIPELINE_EVALUATION_IDENTITY_VERSION,
            "config_schema_version": self.schema_version,
            "pipeline_training_id": self.pipeline_training_id,
            "data_role": self.data_role,
            "materialized_circuit_id": self.materialized_circuit_id,
            "materialized_circuit_checksum": self.materialized_circuit_checksum,
            "test_noise_id": self.test_noise_id,
            "noise_definition_version": self.noise_definition_version,
            "noise_strength_scale": self.noise_strength_scale,
            "tjm_dt": self.tjm_dt,
            "evaluation_seed": self.evaluation_seed,
            "evaluation_seed_domain": self.evaluation_seed_domain,
            "repetition": self.repetition,
            "trajectory_budget": self.trajectory_budget,
            "evaluation_policy": self.evaluation_policy,
            "confidence_level": self.confidence_level,
            "confidence_interval_method": self.confidence_interval_method,
            "sidecar_storage_policy": self.sidecar_storage_policy,
            "max_bond_dimension": self.max_bond_dimension,
            "svd_threshold": self.svd_threshold,
            "truncation_mode": self.truncation_mode,
            "min_bond_dimension": self.min_bond_dimension,
        }

    @property
    def evaluation_row_id(self) -> str:
        """Stable identifier of one repeated final-test evaluation."""
        digest = hashlib.sha256(canonical_json(self.identity_payload()).encode()).hexdigest()
        return f"{EVALUATION_ROW_ID_PREFIX}{digest}"

    @property
    def configuration_checksum(self) -> str:
        """Checksum of path-free evaluation policy and exact pipeline provenance."""
        return canonical_checksum({
            **self.identity_payload(),
            "pipeline_configuration_checksum": self.pipeline_configuration_checksum,
            "pipeline_result_checksum": self.pipeline_result_checksum,
            "final_checkpoint_checksum": self.final_checkpoint_checksum,
            "final_materialization_policy_checksum": self.final_materialization_policy_checksum,
        })

    def validate_against_pipeline(self, pipeline: TrainingPipelineResult) -> None:
        """Verify pipeline/result identity and seed-domain separation.

        Args:
            pipeline: Trained pipeline result to evaluate.

        Raises:
            ValueError: If identity, role, or random-stream provenance differs.
        """
        if not isinstance(pipeline, TrainingPipelineResult):
            _raise_type_error("pipeline", "a TrainingPipelineResult", pipeline)
        if (
            self.pipeline_training_id != pipeline.training_id
            or self.pipeline_configuration_checksum != pipeline.config.configuration_checksum
            or self.pipeline_result_checksum != pipeline.content_checksum
            or self.final_checkpoint_checksum != pipeline.final_checkpoint_checksum
            or self.final_materialization_policy_checksum
            != canonical_checksum(pipeline.config.final_materialization_policy)
            or self.data_role != pipeline.config.data_role
        ):
            msg = "Evaluation configuration does not identify the supplied pipeline result and data role."
            raise ValueError(msg)
        if self.evaluation_seed is None:
            return
        role = {
            "confirmatory": "confirmatory_test",
            "screening_selection": "screening_selection",
            "checkpoint_validation": "checkpoint_validation",
            "development": "pilot_evaluation",
            "secondary_benchmark": "pilot_evaluation",
        }[self.data_role]
        expected_domain = pipeline.config.seed_domains[role]
        if self.evaluation_seed_domain != expected_domain:
            msg = f"Evaluation seed domain must match the pipeline's {role} domain."
            raise ValueError(msg)
        stage_seeds = {
            seed
            for stage in pipeline.config.stages
            for seed in (
                stage.initialization_seed,
                stage.optimizer_seed,
                stage.training_seed,
                stage.checkpoint_validation.seed,
            )
            if seed is not None
        }
        if self.evaluation_seed in stage_seeds:
            msg = "Final evaluation seed must be disjoint from every training and validation seed."
            raise ValueError(msg)

    @classmethod
    def for_pipeline(
        cls,
        *,
        pipeline: TrainingPipelineResult,
        materialized_circuit_checksum: str,
        test_noise_id: str,
        noise_definition_version: str,
        noise_strength_scale: float | None,
        tjm_dt: float | None,
        evaluation_seed: int | None,
        evaluation_seed_domain: str | None,
        repetition: int,
        trajectory_budget: int,
        evaluation_policy: Literal["fixed_sample", "confidence_interval"],
        confidence_level: float | None,
        confidence_interval_method: str | None,
        sidecar_storage_policy: Literal["none", "trajectory_fidelities"],
        max_bond_dimension: int | None,
        svd_threshold: float,
        truncation_mode: Literal["discarded_weight", "relative"],
        min_bond_dimension: int,
    ) -> PipelineEvaluationConfig:
        """Create an evaluation whose circuit identity is mechanically derived."""
        if not isinstance(pipeline, TrainingPipelineResult):
            _raise_type_error("pipeline", "a TrainingPipelineResult", pipeline)
        circuit_checksum = require_checksum(
            materialized_circuit_checksum,
            "materialized_circuit_checksum",
        )
        materialization_checksum = canonical_checksum(pipeline.config.final_materialization_policy)
        config = cls(
            pipeline_training_id=pipeline.training_id,
            pipeline_configuration_checksum=pipeline.config.configuration_checksum,
            pipeline_result_checksum=pipeline.content_checksum,
            final_checkpoint_checksum=pipeline.final_checkpoint_checksum,
            final_materialization_policy_checksum=materialization_checksum,
            data_role=pipeline.config.data_role,
            materialized_circuit_id=_derive_materialized_circuit_id(
                pipeline_training_id=pipeline.training_id,
                final_checkpoint_checksum=pipeline.final_checkpoint_checksum,
                final_materialization_policy_checksum=materialization_checksum,
                materialized_circuit_checksum=circuit_checksum,
            ),
            materialized_circuit_checksum=circuit_checksum,
            test_noise_id=test_noise_id,
            noise_definition_version=noise_definition_version,
            noise_strength_scale=noise_strength_scale,
            tjm_dt=tjm_dt,
            evaluation_seed=evaluation_seed,
            evaluation_seed_domain=evaluation_seed_domain,
            repetition=repetition,
            trajectory_budget=trajectory_budget,
            evaluation_policy=evaluation_policy,
            confidence_level=confidence_level,
            confidence_interval_method=confidence_interval_method,
            sidecar_storage_policy=sidecar_storage_policy,
            max_bond_dimension=max_bond_dimension,
            svd_threshold=svd_threshold,
            truncation_mode=truncation_mode,
            min_bond_dimension=min_bond_dimension,
        )
        config.validate_against_pipeline(pipeline)
        return config

    def to_dict(self) -> dict[str, object]:
        """Return a detached exact-schema evaluation configuration."""
        return {
            "schema_version": self.schema_version,
            "pipeline_training_id": self.pipeline_training_id,
            "pipeline_configuration_checksum": self.pipeline_configuration_checksum,
            "pipeline_result_checksum": self.pipeline_result_checksum,
            "final_checkpoint_checksum": self.final_checkpoint_checksum,
            "final_materialization_policy_checksum": self.final_materialization_policy_checksum,
            "data_role": self.data_role,
            "materialized_circuit_id": self.materialized_circuit_id,
            "materialized_circuit_checksum": self.materialized_circuit_checksum,
            "test_noise_id": self.test_noise_id,
            "noise_definition_version": self.noise_definition_version,
            "noise_strength_scale": self.noise_strength_scale,
            "tjm_dt": self.tjm_dt,
            "evaluation_seed": self.evaluation_seed,
            "evaluation_seed_domain": self.evaluation_seed_domain,
            "repetition": self.repetition,
            "trajectory_budget": self.trajectory_budget,
            "evaluation_policy": self.evaluation_policy,
            "confidence_level": self.confidence_level,
            "confidence_interval_method": self.confidence_interval_method,
            "sidecar_storage_policy": self.sidecar_storage_policy,
            "max_bond_dimension": self.max_bond_dimension,
            "svd_threshold": self.svd_threshold,
            "truncation_mode": self.truncation_mode,
            "min_bond_dimension": self.min_bond_dimension,
            "evaluation_row_id": self.evaluation_row_id,
            "configuration_checksum": self.configuration_checksum,
        }

    @classmethod
    def from_dict(cls, data: object) -> PipelineEvaluationConfig:
        """Construct and identity-verify a strict evaluation configuration."""
        mapping = require_mapping(data, "pipeline evaluation config")
        require_exact_keys(mapping, _EVALUATION_CONFIG_KEYS, "pipeline evaluation config")
        if mapping["schema_version"] != PIPELINE_EVALUATION_CONFIG_SCHEMA_VERSION:
            msg = f"schema_version must be {PIPELINE_EVALUATION_CONFIG_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        config = cls(
            pipeline_training_id=cast("str", mapping["pipeline_training_id"]),
            pipeline_configuration_checksum=cast("str", mapping["pipeline_configuration_checksum"]),
            pipeline_result_checksum=cast("str", mapping["pipeline_result_checksum"]),
            final_checkpoint_checksum=cast("str", mapping["final_checkpoint_checksum"]),
            final_materialization_policy_checksum=cast(
                "str",
                mapping["final_materialization_policy_checksum"],
            ),
            data_role=cast("DataRole", mapping["data_role"]),
            materialized_circuit_id=cast("str", mapping["materialized_circuit_id"]),
            materialized_circuit_checksum=cast("str", mapping["materialized_circuit_checksum"]),
            test_noise_id=cast("str", mapping["test_noise_id"]),
            noise_definition_version=cast("str", mapping["noise_definition_version"]),
            noise_strength_scale=cast("float | None", mapping["noise_strength_scale"]),
            tjm_dt=cast("float | None", mapping["tjm_dt"]),
            evaluation_seed=cast("int | None", mapping["evaluation_seed"]),
            evaluation_seed_domain=cast("str | None", mapping["evaluation_seed_domain"]),
            repetition=cast("int", mapping["repetition"]),
            trajectory_budget=cast("int", mapping["trajectory_budget"]),
            evaluation_policy=cast("Literal['fixed_sample', 'confidence_interval']", mapping["evaluation_policy"]),
            confidence_level=cast("float | None", mapping["confidence_level"]),
            confidence_interval_method=cast("str | None", mapping["confidence_interval_method"]),
            sidecar_storage_policy=cast(
                "Literal['none', 'trajectory_fidelities']",
                mapping["sidecar_storage_policy"],
            ),
            max_bond_dimension=cast("int | None", mapping["max_bond_dimension"]),
            svd_threshold=cast("float", mapping["svd_threshold"]),
            truncation_mode=cast("Literal['discarded_weight', 'relative']", mapping["truncation_mode"]),
            min_bond_dimension=cast("int", mapping["min_bond_dimension"]),
        )
        if mapping["evaluation_row_id"] != config.evaluation_row_id:
            msg = "Serialized evaluation_row_id does not match the derived identity."
            raise ValueError(msg)
        if mapping["configuration_checksum"] != config.configuration_checksum:
            msg = "Serialized evaluation configuration checksum does not match the derived identity."
            raise ValueError(msg)
        return config

    def to_json(self) -> str:
        """Return canonical JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> PipelineEvaluationConfig:
        """Construct an evaluation configuration from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))


def _validate_fidelity(value: object, name: str) -> float:
    """Validate a finite fidelity in the closed unit interval."""
    return require_float(value, name, minimum=0.0, maximum=1.0)


def _validate_ci(
    *,
    config: PipelineEvaluationConfig,
    standard_deviation: object,
    standard_error: object,
    lower: object,
    upper: object,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Validate noisy dispersion and optional confidence bounds."""
    sd = _require_optional_float(standard_deviation, "noisy_fidelity_standard_deviation", minimum=0.0)
    se = _require_optional_float(standard_error, "noisy_fidelity_standard_error", minimum=0.0)
    lo = _require_optional_float(lower, "confidence_interval_lower", minimum=0.0, maximum=1.0)
    hi = _require_optional_float(upper, "confidence_interval_upper", minimum=0.0, maximum=1.0)
    noisy = config.test_noise_id != NOISELESS_NOISE_ID
    if noisy and (sd is None or se is None):
        msg = "Noisy evaluation results require standard deviation and standard error."
        raise ValueError(msg)
    if not noisy and any(value is not None for value in (sd, se, lo, hi)):
        msg = "Noiseless evaluation cannot report Monte Carlo dispersion or confidence bounds."
        raise ValueError(msg)
    requests_ci = config.evaluation_policy == "confidence_interval"
    if requests_ci != (lo is not None and hi is not None):
        msg = "Confidence bounds are required exactly for confidence-interval evaluation."
        raise ValueError(msg)
    if (lo is None) != (hi is None):
        msg = "Confidence bounds must be an ordered pair."
        raise ValueError(msg)
    if lo is not None and hi is not None and lo > hi:
        msg = "Confidence bounds must be an ordered pair."
        raise ValueError(msg)
    return sd, se, lo, hi


@dataclass(frozen=True, slots=True)
class PipelineBenchmarkResult:
    """Successful fixed-sample Phase II pipeline evaluation."""

    config: PipelineEvaluationConfig
    materialized_circuit_path: str
    test_noiseless_fidelity: float
    test_noisy_fidelity: float | None
    noisy_fidelity_standard_deviation: float | None
    noisy_fidelity_standard_error: float | None
    confidence_interval_lower: float | None
    confidence_interval_upper: float | None
    sampled_nonidentity_events: int
    trajectory_sidecar_path: str | None
    trajectory_sidecar_checksum: str | None
    evaluation_wall_time_seconds: float
    peak_memory_bytes: int
    normalized_work: Mapping[str, object]
    runtime_fingerprint_checksum: str
    schema_version: str = field(default=PIPELINE_BENCHMARK_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate success metrics and artifact policy."""
        if not isinstance(self.config, PipelineEvaluationConfig):
            _raise_type_error("config", "a PipelineEvaluationConfig", self.config)
        circuit_path = require_relative_path(self.materialized_circuit_path, "materialized_circuit_path")
        noiseless_fidelity = _validate_fidelity(self.test_noiseless_fidelity, "test_noiseless_fidelity")
        noisy = self.config.test_noise_id != NOISELESS_NOISE_ID
        noisy_fidelity = (
            None
            if self.test_noisy_fidelity is None
            else _validate_fidelity(self.test_noisy_fidelity, "test_noisy_fidelity")
        )
        if noisy != (noisy_fidelity is not None):
            msg = "test_noisy_fidelity is required exactly for noisy evaluation."
            raise ValueError(msg)
        sd, se, lower, upper = _validate_ci(
            config=self.config,
            standard_deviation=self.noisy_fidelity_standard_deviation,
            standard_error=self.noisy_fidelity_standard_error,
            lower=self.confidence_interval_lower,
            upper=self.confidence_interval_upper,
        )
        events = require_int(self.sampled_nonidentity_events, "sampled_nonidentity_events", minimum=0)
        sidecar_path, sidecar_checksum = _validate_pair(
            self.trajectory_sidecar_path,
            self.trajectory_sidecar_checksum,
            path_name="trajectory_sidecar_path",
            checksum_name="trajectory_sidecar_checksum",
        )
        wants_sidecar = self.config.sidecar_storage_policy == "trajectory_fidelities"
        if wants_sidecar != (sidecar_path is not None):
            msg = "Trajectory sidecar presence must match sidecar_storage_policy."
            raise ValueError(msg)
        wall_time = require_float(
            self.evaluation_wall_time_seconds,
            "evaluation_wall_time_seconds",
            minimum=0.0,
        )
        peak_memory = require_int(self.peak_memory_bytes, "peak_memory_bytes", minimum=0)
        work = _validate_normalized_work(self.normalized_work, "normalized_work")
        if work["test_trajectories"] != self.config.trajectory_budget:
            msg = "Result test-trajectory work must equal the fixed evaluation budget."
            raise ValueError(msg)
        runtime = require_checksum(self.runtime_fingerprint_checksum, "runtime_fingerprint_checksum")
        object.__setattr__(self, "materialized_circuit_path", circuit_path)
        object.__setattr__(self, "test_noiseless_fidelity", noiseless_fidelity)
        object.__setattr__(self, "test_noisy_fidelity", noisy_fidelity)
        object.__setattr__(self, "noisy_fidelity_standard_deviation", sd)
        object.__setattr__(self, "noisy_fidelity_standard_error", se)
        object.__setattr__(self, "confidence_interval_lower", lower)
        object.__setattr__(self, "confidence_interval_upper", upper)
        object.__setattr__(self, "sampled_nonidentity_events", events)
        object.__setattr__(self, "trajectory_sidecar_path", sidecar_path)
        object.__setattr__(self, "trajectory_sidecar_checksum", sidecar_checksum)
        object.__setattr__(self, "evaluation_wall_time_seconds", wall_time)
        object.__setattr__(self, "peak_memory_bytes", peak_memory)
        object.__setattr__(self, "normalized_work", work)
        object.__setattr__(self, "runtime_fingerprint_checksum", runtime)

    @property
    def status(self) -> str:
        """Result-stream discriminator."""
        return "success"

    @property
    def evaluation_row_id(self) -> str:
        """Stable planned evaluation identity."""
        return self.config.evaluation_row_id

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete successful result."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered successful result content."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "evaluation_row_id": self.evaluation_row_id,
            "config": self.config.to_dict(),
            "materialized_circuit_path": self.materialized_circuit_path,
            "test_noiseless_fidelity": self.test_noiseless_fidelity,
            "test_noisy_fidelity": self.test_noisy_fidelity,
            "noisy_fidelity_standard_deviation": self.noisy_fidelity_standard_deviation,
            "noisy_fidelity_standard_error": self.noisy_fidelity_standard_error,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "sampled_nonidentity_events": self.sampled_nonidentity_events,
            "trajectory_sidecar_path": self.trajectory_sidecar_path,
            "trajectory_sidecar_checksum": self.trajectory_sidecar_checksum,
            "evaluation_wall_time_seconds": self.evaluation_wall_time_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "normalized_work": thaw_json_mapping(self.normalized_work),
            "runtime_fingerprint_checksum": self.runtime_fingerprint_checksum,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a sealed successful result mapping."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> PipelineBenchmarkResult:
        """Construct and seal-verify a successful result."""
        mapping = verify_sealed_mapping(data, expected_keys=_BENCHMARK_RESULT_KEYS, name="pipeline benchmark result")
        if mapping["schema_version"] != PIPELINE_BENCHMARK_RESULT_SCHEMA_VERSION:
            msg = f"schema_version must be {PIPELINE_BENCHMARK_RESULT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        if mapping["status"] != "success":
            msg = "PipelineBenchmarkResult status must be 'success'."
            raise ValueError(msg)
        result = cls(
            config=PipelineEvaluationConfig.from_dict(mapping["config"]),
            materialized_circuit_path=cast("str", mapping["materialized_circuit_path"]),
            test_noiseless_fidelity=cast("float", mapping["test_noiseless_fidelity"]),
            test_noisy_fidelity=cast("float | None", mapping["test_noisy_fidelity"]),
            noisy_fidelity_standard_deviation=cast(
                "float | None",
                mapping["noisy_fidelity_standard_deviation"],
            ),
            noisy_fidelity_standard_error=cast("float | None", mapping["noisy_fidelity_standard_error"]),
            confidence_interval_lower=cast("float | None", mapping["confidence_interval_lower"]),
            confidence_interval_upper=cast("float | None", mapping["confidence_interval_upper"]),
            sampled_nonidentity_events=cast("int", mapping["sampled_nonidentity_events"]),
            trajectory_sidecar_path=cast("str | None", mapping["trajectory_sidecar_path"]),
            trajectory_sidecar_checksum=cast("str | None", mapping["trajectory_sidecar_checksum"]),
            evaluation_wall_time_seconds=cast("float", mapping["evaluation_wall_time_seconds"]),
            peak_memory_bytes=cast("int", mapping["peak_memory_bytes"]),
            normalized_work=cast("Mapping[str, object]", mapping["normalized_work"]),
            runtime_fingerprint_checksum=cast("str", mapping["runtime_fingerprint_checksum"]),
        )
        if mapping["evaluation_row_id"] != result.evaluation_row_id:
            msg = "Serialized evaluation_row_id does not match the typed evaluation config."
            raise ValueError(msg)
        if mapping["content_checksum"] != result.content_checksum:
            msg = "Pipeline benchmark result checksum changed during normalization."
            raise ValueError(msg)
        return result

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> PipelineBenchmarkResult:
        """Construct a result from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))

    def to_csv_row(self) -> dict[str, object]:
        """Flatten the result into the stable Phase II union CSV schema."""
        return _record_to_csv_row(self.to_dict())


@dataclass(frozen=True, slots=True)
class PipelineBenchmarkFailure:
    """Structured failure for one planned Phase II evaluation."""

    config: PipelineEvaluationConfig
    failure_phase: Literal["pipeline_loading", "materialization", "evaluation", "serialization"]
    exception_type: str
    message: str
    traceback: str | None
    retryable: bool
    attempt: int
    materialized_circuit_path: str | None
    materialized_circuit_checksum: str | None
    wall_time_seconds: float
    runtime_fingerprint_checksum: str
    schema_version: str = field(default=PIPELINE_BENCHMARK_RESULT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate failure provenance without fabricating scientific outcomes."""
        if not isinstance(self.config, PipelineEvaluationConfig):
            _raise_type_error("config", "a PipelineEvaluationConfig", self.config)
        if self.failure_phase not in FAILURE_PHASES:
            msg = f"failure_phase must be one of {FAILURE_PHASES!r}."
            raise ValueError(msg)
        exception_type = require_string(self.exception_type, "exception_type")
        message = require_nonempty_text(self.message, "message")
        traceback = None if self.traceback is None else require_nonempty_text(self.traceback, "traceback")
        retryable = require_bool(self.retryable, "retryable")
        attempt = require_int(self.attempt, "attempt", minimum=1)
        circuit_path, circuit_checksum = _validate_pair(
            self.materialized_circuit_path,
            self.materialized_circuit_checksum,
            path_name="materialized_circuit_path",
            checksum_name="materialized_circuit_checksum",
        )
        if circuit_checksum is not None and circuit_checksum != self.config.materialized_circuit_checksum:
            msg = "Failure materialized-circuit checksum differs from the planned circuit."
            raise ValueError(msg)
        wall_time = require_float(self.wall_time_seconds, "wall_time_seconds", minimum=0.0)
        runtime = require_checksum(self.runtime_fingerprint_checksum, "runtime_fingerprint_checksum")
        object.__setattr__(self, "exception_type", exception_type)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "traceback", traceback)
        object.__setattr__(self, "retryable", retryable)
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "materialized_circuit_path", circuit_path)
        object.__setattr__(self, "materialized_circuit_checksum", circuit_checksum)
        object.__setattr__(self, "wall_time_seconds", wall_time)
        object.__setattr__(self, "runtime_fingerprint_checksum", runtime)

    @property
    def status(self) -> str:
        """Result-stream discriminator."""
        return "failure"

    @property
    def evaluation_row_id(self) -> str:
        """Stable identity of the failed planned evaluation."""
        return self.config.evaluation_row_id

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete failure record."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return checksum-covered failure content."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "evaluation_row_id": self.evaluation_row_id,
            "config": self.config.to_dict(),
            "failure_phase": self.failure_phase,
            "exception_type": self.exception_type,
            "message": self.message,
            "traceback": self.traceback,
            "retryable": self.retryable,
            "attempt": self.attempt,
            "materialized_circuit_path": self.materialized_circuit_path,
            "materialized_circuit_checksum": self.materialized_circuit_checksum,
            "wall_time_seconds": self.wall_time_seconds,
            "runtime_fingerprint_checksum": self.runtime_fingerprint_checksum,
        }

    def to_dict(self) -> dict[str, object]:
        """Return a sealed failure mapping."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    @classmethod
    def from_dict(cls, data: object) -> PipelineBenchmarkFailure:
        """Construct and seal-verify a strict failure."""
        mapping = verify_sealed_mapping(data, expected_keys=_BENCHMARK_FAILURE_KEYS, name="pipeline benchmark failure")
        if mapping["schema_version"] != PIPELINE_BENCHMARK_RESULT_SCHEMA_VERSION:
            msg = f"schema_version must be {PIPELINE_BENCHMARK_RESULT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        if mapping["status"] != "failure":
            msg = "PipelineBenchmarkFailure status must be 'failure'."
            raise ValueError(msg)
        failure = cls(
            config=PipelineEvaluationConfig.from_dict(mapping["config"]),
            failure_phase=cast(
                "Literal['pipeline_loading', 'materialization', 'evaluation', 'serialization']",
                mapping["failure_phase"],
            ),
            exception_type=cast("str", mapping["exception_type"]),
            message=cast("str", mapping["message"]),
            traceback=cast("str | None", mapping["traceback"]),
            retryable=cast("bool", mapping["retryable"]),
            attempt=cast("int", mapping["attempt"]),
            materialized_circuit_path=cast("str | None", mapping["materialized_circuit_path"]),
            materialized_circuit_checksum=cast("str | None", mapping["materialized_circuit_checksum"]),
            wall_time_seconds=cast("float", mapping["wall_time_seconds"]),
            runtime_fingerprint_checksum=cast("str", mapping["runtime_fingerprint_checksum"]),
        )
        if mapping["evaluation_row_id"] != failure.evaluation_row_id:
            msg = "Serialized evaluation_row_id does not match the typed evaluation config."
            raise ValueError(msg)
        if mapping["content_checksum"] != failure.content_checksum:
            msg = "Pipeline benchmark failure checksum changed during normalization."
            raise ValueError(msg)
        return failure

    @classmethod
    def from_exception(
        cls,
        *,
        config: PipelineEvaluationConfig,
        failure_phase: Literal["pipeline_loading", "materialization", "evaluation", "serialization"],
        exception: BaseException,
        runtime_fingerprint_checksum: str,
        traceback: str | None = None,
        retryable: bool = False,
        attempt: int = 1,
        materialized_circuit_path: str | None = None,
        materialized_circuit_checksum: str | None = None,
        wall_time_seconds: float = 0.0,
    ) -> PipelineBenchmarkFailure:
        """Create a structured failure from an exception."""
        if not isinstance(exception, BaseException):
            _raise_type_error("exception", "a BaseException", exception)
        return cls(
            config=config,
            failure_phase=failure_phase,
            exception_type=type(exception).__name__,
            message=str(exception) or type(exception).__name__,
            traceback=traceback,
            retryable=retryable,
            attempt=attempt,
            materialized_circuit_path=materialized_circuit_path,
            materialized_circuit_checksum=materialized_circuit_checksum,
            wall_time_seconds=wall_time_seconds,
            runtime_fingerprint_checksum=runtime_fingerprint_checksum,
        )

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> PipelineBenchmarkFailure:
        """Construct a failure from canonical JSON text."""
        return cls.from_dict(load_canonical_json_object(payload))

    def to_csv_row(self) -> dict[str, object]:
        """Flatten the failure into the stable Phase II union CSV schema."""
        return _record_to_csv_row(self.to_dict())


PipelineBenchmarkRecord = PipelineBenchmarkResult | PipelineBenchmarkFailure


def pipeline_benchmark_record_from_dict(data: object) -> PipelineBenchmarkRecord:
    """Deserialize a Phase II evaluation record by status."""
    mapping = require_mapping(data, "pipeline benchmark record")
    if mapping.get("status") == "success":
        return PipelineBenchmarkResult.from_dict(mapping)
    if mapping.get("status") == "failure":
        return PipelineBenchmarkFailure.from_dict(mapping)
    msg = "Pipeline benchmark record status must be 'success' or 'failure'."
    raise ValueError(msg)


def pipeline_benchmark_record_from_json(payload: str) -> PipelineBenchmarkRecord:
    """Deserialize a canonical Phase II evaluation record."""
    return pipeline_benchmark_record_from_dict(load_canonical_json_object(payload))


def _record_to_csv_row(data: Mapping[str, object]) -> dict[str, object]:
    """Flatten one success or failure into the stable union CSV schema."""
    config = cast("Mapping[str, object]", data["config"])
    aliases: dict[str, object] = {
        "pipeline_training_id": config["pipeline_training_id"],
        "data_role": config["data_role"],
        "materialized_circuit_id": config["materialized_circuit_id"],
        "test_noise_id": config["test_noise_id"],
        "evaluation_seed": config["evaluation_seed"],
        "repetition": config["repetition"],
        "trajectory_budget": config["trajectory_budget"],
        "evaluation_wall_time_seconds": data.get("wall_time_seconds"),
    }
    row: dict[str, object] = {}
    for column in PIPELINE_CSV_COLUMNS:
        value = data.get(column, aliases.get(column))
        row[column] = canonical_json(value) if column in _CSV_JSON_COLUMNS and value is not None else value
    return row


def _csv_optional(value: object) -> object | None:
    """Normalize a blank CSV cell to ``None``."""
    return None if value is None or (type(value) is str and not value) else value


def _csv_int(value: object, name: str) -> int | None:
    """Decode an optional canonical base-ten CSV integer."""
    normalized = _csv_optional(value)
    if normalized is None:
        return None
    if type(normalized) is int:
        return normalized
    if type(normalized) is not str or re.fullmatch(r"-?(0|[1-9][0-9]*)", normalized) is None:
        msg = f"CSV column {name!r} must contain a canonical base-ten integer."
        raise ValueError(msg)
    return int(normalized)


def _csv_float(value: object, name: str) -> float | None:
    """Decode an optional finite CSV float."""
    normalized = _csv_optional(value)
    if normalized is None:
        return None
    if type(normalized) is float:
        result = normalized
    elif type(normalized) is str:
        try:
            result = float(normalized)
        except ValueError as error:
            msg = f"CSV column {name!r} must contain a float."
            raise ValueError(msg) from error
    else:
        _raise_type_error(f"CSV column {name!r}", "a float string", normalized)
    if not math.isfinite(result):
        msg = f"CSV column {name!r} must contain a finite float."
        raise ValueError(msg)
    return result


def _csv_bool(value: object, name: str) -> bool | None:
    """Decode an optional strict CSV Boolean."""
    normalized = _csv_optional(value)
    if normalized is None:
        return None
    if type(normalized) is bool:
        return normalized
    if normalized == "True":
        return True
    if normalized == "False":
        return False
    msg = f"CSV column {name!r} must contain 'True' or 'False'."
    raise ValueError(msg)


def pipeline_benchmark_record_from_csv_row(row: object) -> PipelineBenchmarkRecord:
    """Deserialize one strict Phase II union CSV row."""
    mapping = require_mapping(row, "pipeline benchmark CSV row")
    require_exact_keys(mapping, frozenset(PIPELINE_CSV_COLUMNS), "pipeline benchmark CSV row")
    decoded: dict[str, object] = {}
    for column in PIPELINE_CSV_COLUMNS:
        value = mapping[column]
        if column in _CSV_INTEGER_COLUMNS:
            decoded[column] = _csv_int(value, column)
        elif column in _CSV_FLOAT_COLUMNS:
            decoded[column] = _csv_float(value, column)
        elif column in _CSV_BOOLEAN_COLUMNS:
            decoded[column] = _csv_bool(value, column)
        elif column in _CSV_JSON_COLUMNS:
            normalized = _csv_optional(value)
            if normalized is None:
                decoded[column] = None
            elif type(normalized) is str:
                decoded[column] = load_canonical_json_object(normalized)
            elif isinstance(normalized, Mapping):
                decoded[column] = normalized
            else:
                _raise_type_error(f"CSV column {column!r}", "a JSON object string", normalized)
        else:
            decoded[column] = _csv_optional(value)
    config = PipelineEvaluationConfig.from_dict(decoded["config"])
    aliases = {
        "evaluation_row_id": config.evaluation_row_id,
        "pipeline_training_id": config.pipeline_training_id,
        "data_role": config.data_role,
        "materialized_circuit_id": config.materialized_circuit_id,
        "test_noise_id": config.test_noise_id,
        "evaluation_seed": config.evaluation_seed,
        "repetition": config.repetition,
        "trajectory_budget": config.trajectory_budget,
    }
    if any(decoded[name] != value for name, value in aliases.items()):
        msg = "CSV reporting aliases do not match the typed evaluation configuration."
        raise ValueError(msg)
    if decoded["status"] == "success":
        unexpected = sorted(
            column for column in frozenset(PIPELINE_CSV_COLUMNS) - _CSV_SUCCESS_COLUMNS if decoded[column] is not None
        )
        if unexpected:
            msg = f"Success CSV rows must leave failure-only columns blank: {unexpected!r}."
            raise ValueError(msg)
        data = {
            "schema_version": decoded["schema_version"],
            "status": "success",
            "evaluation_row_id": decoded["evaluation_row_id"],
            "config": config.to_dict(),
            "materialized_circuit_path": decoded["materialized_circuit_path"],
            "test_noiseless_fidelity": decoded["test_noiseless_fidelity"],
            "test_noisy_fidelity": decoded["test_noisy_fidelity"],
            "noisy_fidelity_standard_deviation": decoded["noisy_fidelity_standard_deviation"],
            "noisy_fidelity_standard_error": decoded["noisy_fidelity_standard_error"],
            "confidence_interval_lower": decoded["confidence_interval_lower"],
            "confidence_interval_upper": decoded["confidence_interval_upper"],
            "sampled_nonidentity_events": decoded["sampled_nonidentity_events"],
            "trajectory_sidecar_path": decoded["trajectory_sidecar_path"],
            "trajectory_sidecar_checksum": decoded["trajectory_sidecar_checksum"],
            "evaluation_wall_time_seconds": decoded["evaluation_wall_time_seconds"],
            "peak_memory_bytes": decoded["peak_memory_bytes"],
            "normalized_work": decoded["normalized_work"],
            "runtime_fingerprint_checksum": decoded["runtime_fingerprint_checksum"],
        }
        data["content_checksum"] = canonical_checksum(data)
        return PipelineBenchmarkResult.from_dict(data)
    if decoded["status"] == "failure":
        unexpected = sorted(
            column for column in frozenset(PIPELINE_CSV_COLUMNS) - _CSV_FAILURE_COLUMNS if decoded[column] is not None
        )
        if unexpected:
            msg = f"Failure CSV rows must leave success-only columns blank: {unexpected!r}."
            raise ValueError(msg)
        data = {
            "schema_version": decoded["schema_version"],
            "status": "failure",
            "evaluation_row_id": decoded["evaluation_row_id"],
            "config": config.to_dict(),
            "failure_phase": decoded["failure_phase"],
            "exception_type": decoded["exception_type"],
            "message": decoded["message"],
            "traceback": decoded["traceback"],
            "retryable": decoded["retryable"],
            "attempt": decoded["attempt"],
            "materialized_circuit_path": decoded["materialized_circuit_path"],
            "materialized_circuit_checksum": (
                config.materialized_circuit_checksum if decoded["materialized_circuit_path"] is not None else None
            ),
            "wall_time_seconds": decoded["evaluation_wall_time_seconds"],
            "runtime_fingerprint_checksum": decoded["runtime_fingerprint_checksum"],
        }
        data["content_checksum"] = canonical_checksum(data)
        return PipelineBenchmarkFailure.from_dict(data)
    msg = "CSV status must be 'success' or 'failure'."
    raise ValueError(msg)


def validate_screening_resolution(
    *,
    screening_manifest: ScreeningManifest,
    target_manifest: TargetPopulationManifest,
    candidate: object,
    cell: object,
    template: TrainingPipelineTemplate,
    pipeline: TrainingPipelineConfig,
    pipeline_result: TrainingPipelineResult | None = None,
    evaluation: PipelineEvaluationConfig | None = None,
) -> None:
    """Bind one WP15 candidate/cell to its target-independent and resolved records.

    This validator is intentionally mechanical: caller-supplied method,
    matching, target, and seed aliases cannot substitute for the typed WP15
    records and their derived WP16 identities.

    Args:
        candidate: A :class:`~.protocol.ScreeningCandidateRef`.
        cell: A :class:`~.protocol.ScreeningCell`.
        screening_manifest: Complete sealed WP15 candidate/cell universe.
        target_manifest: Exact seed-bearing WP16 screening population.
        template: Target-independent candidate configuration.
        pipeline: Concrete target/optimization resolution.
        pipeline_result: Optional completed training result.
        evaluation: Optional screening evaluation using the cell's screening seed.

    Raises:
        TypeError: If a record has the wrong typed schema.
        ValueError: If any candidate, target, seed, or result binding differs.
    """
    if not isinstance(candidate, ScreeningCandidateRef):
        _raise_type_error("candidate", "a ScreeningCandidateRef", candidate)
    if not isinstance(cell, ScreeningCell):
        _raise_type_error("cell", "a ScreeningCell", cell)
    if not isinstance(template, TrainingPipelineTemplate):
        _raise_type_error("template", "a TrainingPipelineTemplate", template)
    if not isinstance(pipeline, TrainingPipelineConfig):
        _raise_type_error("pipeline", "a TrainingPipelineConfig", pipeline)
    if not isinstance(screening_manifest, ScreeningManifest):
        _raise_type_error("screening_manifest", "a ScreeningManifest", screening_manifest)
    if not isinstance(target_manifest, TargetPopulationManifest):
        _raise_type_error("target_manifest", "a TargetPopulationManifest", target_manifest)
    verify_screening_target_population(screening_manifest, target_manifest)
    preregistration = load_initial_preregistration()
    expected_evaluation_policy_checksum = canonical_checksum({
        "endpoint": preregistration.primary_endpoint,
        "failure_policy": preregistration.failure_policy,
        "noise": preregistration.primary_noise_condition,
    })
    if screening_manifest.evaluation_policy_checksum != expected_evaluation_policy_checksum:
        msg = "WP15 screening manifest does not carry the trusted primary evaluation policy."
        raise ValueError(msg)
    if screening_manifest.resource_policy_checksum != canonical_checksum(preregistration.primary_resource_constraint):
        msg = "WP15 screening manifest does not carry the trusted primary resource policy."
        raise ValueError(msg)
    resource = preregistration.primary_resource_constraint
    noise = preregistration.primary_noise_condition
    expected_materialization = {
        "policy_id": "native_chain_v1",
        "compiler_policy_id": resource["compiler_policy_id"],
        "connectivity_id": resource["connectivity"],
        "routing_policy_id": resource["routing_policy"],
        "optimization_level": 0,
        "noise_placement": noise["test_placement"],
        "parameter_source": "selected_checkpoint",
    }
    for name, value in expected_materialization.items():
        if template.final_materialization_policy[name] != value:
            msg = f"WP16 template materialization {name} does not match the trusted primary policy."
            raise ValueError(msg)
    if candidate not in screening_manifest.candidates or cell not in screening_manifest.cells:
        msg = "Candidate and cell must belong to the supplied complete WP15 screening manifest."
        raise ValueError(msg)
    try:
        target_spec = next(
            spec for spec in target_manifest.instances if spec.target_instance_id == cell.target_instance_id
        )
    except StopIteration as error:
        msg = "WP15 screening target is absent from the supplied WP16 target manifest."
        raise ValueError(msg) from error
    noisy_training = any(stage.stage_policy["training_noise_id"] != NOISELESS_NOISE_ID for stage in template.stages)
    expected_candidate = {
        "configuration_schema_version": template.schema_version,
        "configuration_checksum": template.configuration_checksum,
        "method_id": template.method_id,
        "noisy_training": noisy_training,
        "resource_stratum_id": template.resource_stratum_id,
        "matching_projection_checksum": (
            template.matching_projection_checksum
            if template.method_id in {"layerwise_bmpd_crn_v2", "layerwise_bmpd_noiseless"}
            else None
        ),
    }
    for name, value in expected_candidate.items():
        if getattr(candidate, name) != value:
            msg = f"WP15 screening candidate {name} does not match the WP16 template."
            raise ValueError(msg)
    if (
        pipeline.template != template
        or pipeline.template_checksum != candidate.configuration_checksum
        or pipeline.target_ref != Phase2TargetRef.from_manifest(target_manifest, cell.target_instance_id)
        or pipeline.data_role != "screening_selection"
        or pipeline.target_population_manifest_checksum != target_manifest.content_checksum
        or pipeline.target_instance_id != cell.target_instance_id
        or pipeline.target_instance_spec_checksum != target_spec.content_checksum
        or pipeline.target_family_id != target_spec.family_id
        or pipeline.target_stratum_id != target_spec.stratum_id
        or pipeline.qubit_count != target_spec.qubit_count
        or pipeline.optimization_block_id != cell.cell_id
        or pipeline.optimization_seed != cell.optimization_seed
    ):
        msg = "Resolved WP16 pipeline does not match the complete WP15 screening cell."
        raise ValueError(msg)
    if pipeline_result is not None:
        if not isinstance(pipeline_result, TrainingPipelineResult):
            _raise_type_error("pipeline_result", "a TrainingPipelineResult", pipeline_result)
        if pipeline_result.config != pipeline:
            msg = "Training result does not belong to the resolved screening pipeline."
            raise ValueError(msg)
    if evaluation is not None:
        if not isinstance(evaluation, PipelineEvaluationConfig):
            _raise_type_error("evaluation", "a PipelineEvaluationConfig", evaluation)
        if pipeline_result is None:
            msg = "A screening evaluation requires its exact training pipeline result."
            raise ValueError(msg)
        evaluation.validate_against_pipeline(pipeline_result)
        if (
            evaluation.evaluation_seed != cell.screening_seed
            or evaluation.evaluation_seed_domain != template.seed_domains["screening_selection"]
        ):
            msg = "Screening evaluation seed or domain does not match the WP15 screening cell."
            raise ValueError(msg)
        if (
            preregistration.primary_endpoint["metric"] != "fresh_test_noisy_fidelity"
            or evaluation.test_noise_id != noise["noise_id"]
            or evaluation.noise_definition_version != noise["definition_version"]
            or canonical_json(evaluation.noise_strength_scale) != canonical_json(noise["strength_scale"])
            or canonical_json(evaluation.tjm_dt) != canonical_json(noise["tjm_dt"])
            or template.final_materialization_policy["noise_placement"] != noise["test_placement"]
        ):
            msg = "Screening evaluation does not implement the trusted fresh-test noisy-fidelity endpoint."
            raise ValueError(msg)


__all__ = [
    "CHECKPOINT_SELECTION_RULES",
    "CHECKPOINT_TIE_BREAKERS",
    "CHECKPOINT_VALIDATION_SCHEMA_VERSION",
    "DATA_ROLES",
    "EVALUATION_POLICIES",
    "EVALUATION_ROW_ID_PREFIX",
    "EXTERNAL_CHECKPOINT_REF_SCHEMA_VERSION",
    "FAILURE_PHASES",
    "LEGACY_LAYERWISE_SEED_BINDINGS",
    "LEGACY_REPRODUCTION_MANIFEST_CHECKSUM",
    "LEGACY_REPRODUCTION_TARGET_IDS",
    "MATERIALIZED_CIRCUIT_ID_PREFIX",
    "PARAMETER_TRANSFER_RULES",
    "PHASE1_FIXTURE_MANIFEST_CHECKSUM",
    "PHASE1_FIXTURE_TARGET_IDS",
    "PHASE2_TARGET_ID_PREFIX",
    "PHASE2_TARGET_REF_SCHEMA_VERSION",
    "PIPELINE_BENCHMARK_RESULT_SCHEMA_VERSION",
    "PIPELINE_CSV_COLUMNS",
    "PIPELINE_EVALUATION_CONFIG_SCHEMA_VERSION",
    "PIPELINE_EVALUATION_IDENTITY_VERSION",
    "PIPELINE_PREFIX",
    "PIPELINE_PREFIX_IDENTITY_VERSION",
    "PIPELINE_TRAINING_IDENTITY_VERSION",
    "PRUNING_RULES",
    "SEED_DOMAIN_ROLES",
    "SIDECAR_STORAGE_POLICIES",
    "STAGE_KINDS",
    "TARGET_NAMESPACES",
    "TARGET_SCOPE_IDS",
    "TRAINING_ID_PREFIX",
    "TRAINING_PIPELINE_CONFIG_SCHEMA_VERSION",
    "TRAINING_PIPELINE_RESULT_SCHEMA_VERSION",
    "TRAINING_PIPELINE_TEMPLATE_SCHEMA_VERSION",
    "TRAINING_SAMPLING_POLICIES",
    "TRAINING_STAGE_CONFIG_SCHEMA_VERSION",
    "TRAINING_STAGE_RESULT_SCHEMA_VERSION",
    "TRAINING_STAGE_TEMPLATE_SCHEMA_VERSION",
    "TRAJECTORY_UPDATES",
    "TRUNCATION_MODES",
    "CheckpointValidationConfig",
    "ExternalCheckpointRef",
    "Phase2TargetRef",
    "PipelineBenchmarkFailure",
    "PipelineBenchmarkRecord",
    "PipelineBenchmarkResult",
    "PipelineEvaluationConfig",
    "TrainingPipelineConfig",
    "TrainingPipelineResult",
    "TrainingPipelineTemplate",
    "TrainingStageConfig",
    "TrainingStageResult",
    "TrainingStageTemplate",
    "fixture_target_spec_checksum",
    "pipeline_benchmark_record_from_csv_row",
    "pipeline_benchmark_record_from_dict",
    "pipeline_benchmark_record_from_json",
    "validate_screening_resolution",
]
