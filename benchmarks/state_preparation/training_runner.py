# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Opt-in command-line boundary for WP22 Phase II training plans.

Scientific command-line values in this module are assertions about immutable,
checksum-bearing artifacts.  They never rewrite a pipeline, schedule, target
manifest, screening manifest, or final seal.  In particular, confirmation
source custody is verified before the confirmatory target-manifest path is
opened.
"""

# The private parsing/validation helpers deliberately centralize contextual
# errors; repeating their transitively raised exceptions would obscure the API.

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, NoReturn, cast

from filelock import FileLock

from benchmarks.state_preparation.constants import BALLARIN_NOISE_ID, NOISELESS_NOISE_ID
from benchmarks.state_preparation.phase2.binding_catalog import RepositoryBindingCatalog
from benchmarks.state_preparation.phase2.canonical import (
    canonical_json,
    load_canonical_json_object,
    verify_sealed_mapping,
)
from benchmarks.state_preparation.phase2.confirmatory_study import PriorTargetExposureInventory
from benchmarks.state_preparation.phase2.confirmatory_study_store import (
    LockedConfirmatoryStudySnapshot,
    LockedConfirmatoryStudySnapshotRef,
)
from benchmarks.state_preparation.phase2.execution_bindings import TrainingExecutionProfile
from benchmarks.state_preparation.phase2.execution_context import (
    AuthorizedTargetMaterialization,
    ConfirmationExecutionContext,
    ExternalEntropyKeyring,
    TrainingExecutionContext,
    bind_training_plan_fingerprints,
    candidate_refs_from_bindings,
    parse_entropy_file_specs,
    schedules_from_bindings,
    validate_resumability_source_fingerprints,
)
from benchmarks.state_preparation.phase2.pipeline import (
    TRAINING_PIPELINE_CONFIG_SCHEMA_VERSION,
    TRAINING_PIPELINE_TEMPLATE_SCHEMA_VERSION,
    TrainingPipelineConfig,
    TrainingPipelineTemplate,
)
from benchmarks.state_preparation.phase2.protocol import (
    DEFAULT_PREREGISTRATION_PATH,
    TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
    AnalysisSourceManifest,
    ConfirmationAuthorization,
    FinalConfigurationExecutionManifest,
    FinalConfirmationSeal,
    InitialPreregistration,
    PromotionDecision,
    SampleSizeDesign,
    ScreeningEvidence,
    ScreeningManifest,
    authorize_confirmation,
    load_initial_preregistration,
    validate_final_configuration_execution_manifest,
)
from benchmarks.state_preparation.phase2.resumability import ResumabilityFingerprint
from benchmarks.state_preparation.phase2.run_historical_reproduction import run_historical_reproduction_job
from benchmarks.state_preparation.phase2.screening import ProductionResourceCalibration
from benchmarks.state_preparation.phase2.screening_design import WP22CandidateConfiguration
from benchmarks.state_preparation.phase2.source_lock import (
    ExecutionSourceManifest,
    verify_execution_source_manifest,
    verify_final_seal_source_lock,
)
from benchmarks.state_preparation.phase2.targets import (
    TargetMaterializationAuthorization,
    TargetPopulationConfig,
    TargetPopulationManifest,
    authorize_target_materialization,
)
from benchmarks.state_preparation.phase2.training_orchestration import (
    PILOT_OPTIMIZATION_SEED_COUNT,
    RUNNABLE_DATA_ROLES,
    TRAINING_PRESETS,
    TrainingExecutorRegistry,
    TrainingJobExecutor,
    TrainingRunPlan,
    TrainingRunSummary,
    build_historical_reproduction_plan,
    build_paper_confirm_plan,
    build_paper_pilot_plan,
    build_paper_screen_plan,
    build_training_smoke_plan,
    derive_pilot_optimization_seeds,
    execute_training_plan,
    reject_ballarin_training,
)
from benchmarks.state_preparation.phase2.training_schedules import TrainingStrategySchedule

if TYPE_CHECKING:
    from argparse import Namespace


TRAINING_RUNNER_CONFIGURATION_FORMAT = "yaqs.state_preparation.phase2.training_runner_config.v2"
DEFAULT_TRAINING_OUTPUT_ROOT = Path("state_preparation_phase2_results")
DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXECUTOR_FACTORY = (
    "benchmarks.state_preparation.phase2.production_executors:create_default_training_executor_registry"
)
DEFAULT_PILOT_OPTIMIZATION_SEEDS = derive_pilot_optimization_seeds(
    TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
    PILOT_OPTIMIZATION_SEED_COUNT,
)

_CHECKSUM_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOPOLOGY_DEPTH_PATTERN = re.compile(r"(?:^|_)d(?P<depth>[0-9]+)$")
_EXECUTOR_FACTORY_PATTERN = re.compile(
    r"^(?P<module>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*):"
    r"(?P<attribute>[A-Za-z_][A-Za-z0-9_]*)$"
)
_CONFIRMATION_SESSION_HEADER_KEYS = frozenset({
    "schema_version",
    "plan_checksum",
    "final_confirmation_seal_checksum",
    "execution_source_manifest_checksum",
    "analysis_source_manifest_checksum",
    "prior_target_exposure_inventory_checksum",
    "authorized_output_root",
    "locked_study_head_custody_path",
    "job_count",
    "content_checksum",
})
_CONFIGURATION_KEYS = frozenset({
    "format",
    "preset",
    "pipeline",
    "pipeline_path",
    "method",
    "method_id",
    "stage_depth",
    "stage_depths",
    "stage_budget",
    "stage_budgets",
    "training_noise_id",
    "training_noise_strength",
    "trajectory_update",
    "sampling_policy",
    "training_trajectories",
    "training_trajectory_count",
    "validation_trajectories",
    "checkpoint_validation_trajectory_count",
    "crn_refresh_interval",
    "checkpoint_rule",
    "target_manifest",
    "target_manifest_path",
    "target_manifest_paths",
    "target_manifest_checksum",
    "target_manifest_checksums",
    "data_role",
    "native_two_qubit_cap_per_edge",
    "normalized_compute_cap",
    "preregistration",
    "preregistration_path",
    "preregistration_checksum",
    "resume",
    "overwrite",
    "dry_run",
    "fail_fast",
    "legacy_reproduction",
    "execute_expensive",
    "executor_factory",
    "candidate",
    "candidate_path",
    "candidate_paths",
    "schedule",
    "schedule_path",
    "schedule_paths",
    "screening_manifest",
    "screening_manifest_path",
    "screening_evidence",
    "screening_evidence_path",
    "promotion_decision",
    "promotion_decision_path",
    "final_seal",
    "final_seal_path",
    "configuration_execution_manifest",
    "configuration_execution_manifest_path",
    "execution_source_manifest",
    "execution_source_manifest_path",
    "analysis_source_manifest",
    "analysis_source_manifest_path",
    "execution_profile",
    "execution_profile_path",
    "binding_catalog",
    "binding_catalog_path",
    "target_configuration",
    "target_configuration_path",
    "target_configuration_paths",
    "sample_size_design",
    "sample_size_design_path",
    "resource_calibration",
    "resource_calibration_path",
    "prior_target_exposure_inventory",
    "prior_target_exposure_inventory_path",
    "expected_locked_study_head",
    "expected_locked_study_head_path",
    "target_exposure_inventory",
    "resumability_fingerprint",
    "resumability_fingerprint_path",
    "resumability_fingerprint_paths",
    "repository_root",
    "pilot_optimization_seed",
    "pilot_optimization_seeds",
    "output",
    "output_dir",
})
_CONFIGURATION_ALIASES = {
    "pipeline": "pipeline_path",
    "method": "method_id",
    "stage_depth": "stage_depths",
    "stage_budget": "stage_budgets",
    "training_trajectories": "training_trajectory_count",
    "validation_trajectories": "checkpoint_validation_trajectory_count",
    "target_manifest": "target_manifest_paths",
    "target_manifest_path": "target_manifest_paths",
    "target_manifest_checksum": "target_manifest_checksums",
    "preregistration": "preregistration_path",
    "candidate": "candidate_paths",
    "candidate_path": "candidate_paths",
    "schedule": "schedule_paths",
    "schedule_path": "schedule_paths",
    "screening_manifest": "screening_manifest_path",
    "screening_evidence": "screening_evidence_path",
    "promotion_decision": "promotion_decision_path",
    "final_seal": "final_seal_path",
    "configuration_execution_manifest": "configuration_execution_manifest_path",
    "execution_source_manifest": "execution_source_manifest_path",
    "analysis_source_manifest": "analysis_source_manifest_path",
    "execution_profile": "execution_profile_path",
    "binding_catalog": "binding_catalog_path",
    "target_configuration": "target_configuration_paths",
    "target_configuration_path": "target_configuration_paths",
    "sample_size_design": "sample_size_design_path",
    "resource_calibration": "resource_calibration_path",
    "prior_target_exposure_inventory": "prior_target_exposure_inventory_path",
    "target_exposure_inventory": "prior_target_exposure_inventory_path",
    "expected_locked_study_head": "expected_locked_study_head_path",
    "resumability_fingerprint": "resumability_fingerprint_paths",
    "resumability_fingerprint_path": "resumability_fingerprint_paths",
    "pilot_optimization_seed": "pilot_optimization_seeds",
    "output": "output_dir",
}
_CLI_DESTINATIONS = (
    "pipeline_path",
    "method_id",
    "stage_depths",
    "stage_budgets",
    "training_noise_id",
    "training_noise_strength",
    "trajectory_update",
    "sampling_policy",
    "training_trajectory_count",
    "checkpoint_validation_trajectory_count",
    "crn_refresh_interval",
    "checkpoint_rule",
    "target_manifest_paths",
    "target_manifest_checksums",
    "data_role",
    "native_two_qubit_cap_per_edge",
    "normalized_compute_cap",
    "preregistration_path",
    "preregistration_checksum",
    "resume",
    "overwrite",
    "dry_run",
    "fail_fast",
    "legacy_reproduction",
    "execute_expensive",
    "executor_factory",
    "candidate_paths",
    "schedule_paths",
    "screening_manifest_path",
    "screening_evidence_path",
    "promotion_decision_path",
    "final_seal_path",
    "configuration_execution_manifest_path",
    "execution_source_manifest_path",
    "analysis_source_manifest_path",
    "execution_profile_path",
    "binding_catalog_path",
    "target_configuration_paths",
    "sample_size_design_path",
    "resource_calibration_path",
    "prior_target_exposure_inventory_path",
    "expected_locked_study_head_path",
    "resumability_fingerprint_paths",
    "external_entropy_file_specs",
    "repository_root",
    "pilot_optimization_seeds",
    "output_dir",
)

_PAPER_CONFIRM_ALLOWED_EXPLICIT_OPTIONS = frozenset({
    "preset",
    "target_manifest_paths",
    "target_manifest_checksums",
    "preregistration_path",
    "preregistration_checksum",
    "resume",
    "dry_run",
    "execute_expensive",
    "screening_manifest_path",
    "screening_evidence_path",
    "promotion_decision_path",
    "final_seal_path",
    "configuration_execution_manifest_path",
    "execution_source_manifest_path",
    "analysis_source_manifest_path",
    "binding_catalog_path",
    "target_configuration_paths",
    "sample_size_design_path",
    "resource_calibration_path",
    "prior_target_exposure_inventory_path",
    "expected_locked_study_head_path",
    "external_entropy_file_specs",
    "repository_root",
    "output_dir",
})


class TrainingRunnerConfigurationError(ValueError):
    """Raised when an opt-in WP22 runner request is not safely authorized."""


# Short compatibility spelling for callers that already distinguish the module.
RunnerConfigurationError = TrainingRunnerConfigurationError


@dataclass(frozen=True, slots=True)
class TrainingRunnerOptions:
    """Resolved controls and artifact assertions for one WP22 invocation."""

    preset: str
    pipeline_path: Path | None
    method_id: str | None
    stage_depths: tuple[int, ...]
    stage_budgets: tuple[int, ...]
    training_noise_id: str | None
    training_noise_strength: float | None
    trajectory_update: str | None
    sampling_policy: str | None
    training_trajectory_count: int | None
    checkpoint_validation_trajectory_count: int | None
    crn_refresh_interval: int | None
    checkpoint_rule: str | None
    target_manifest_paths: tuple[Path, ...]
    target_manifest_checksums: tuple[str, ...]
    data_role: str | None
    native_two_qubit_cap_per_edge: float | None
    normalized_compute_cap: float | None
    preregistration_path: Path
    preregistration_checksum: str
    resume: bool
    overwrite: bool
    dry_run: bool
    fail_fast: bool
    legacy_reproduction: bool
    execute_expensive: bool
    executor_factory: str | None
    candidate_paths: tuple[Path, ...]
    schedule_paths: tuple[Path, ...]
    screening_manifest_path: Path | None
    screening_evidence_path: Path | None
    promotion_decision_path: Path | None
    final_seal_path: Path | None
    configuration_execution_manifest_path: Path | None
    execution_source_manifest_path: Path | None
    analysis_source_manifest_path: Path | None
    execution_profile_path: Path | None
    binding_catalog_path: Path | None
    target_configuration_paths: tuple[Path, ...]
    sample_size_design_path: Path | None
    resource_calibration_path: Path | None
    prior_target_exposure_inventory_path: Path | None
    resumability_fingerprint_paths: tuple[Path, ...]
    external_entropy_file_specs: tuple[str, ...]
    repository_root: Path
    pilot_optimization_seeds: tuple[int, ...]
    output_dir: Path
    expected_locked_study_head_path: Path | None = None
    explicit_option_names: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)
    output_was_cli_explicit: bool = field(default=False, repr=False, compare=False)
    requested_output_dir: Path | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PipelineView:
    """Uniform immutable view of a pipeline template or concrete config."""

    template: TrainingPipelineTemplate
    concrete: TrainingPipelineConfig | None


def _reject_json_constant(value: str) -> NoReturn:
    """Reject one nonstandard JSON numeric constant.

    Raises:
        TrainingRunnerConfigurationError: Always, because JSON constants must
            be finite standard numbers.
    """
    msg = f"Non-finite JSON constant {value!r} is not supported."
    raise TrainingRunnerConfigurationError(msg)


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build an object while rejecting duplicate keys at every nesting level.

    Returns:
        The detached mapping represented by ``pairs``.

    Raises:
        TrainingRunnerConfigurationError: If a key occurs more than once.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"Duplicate JSON configuration key {key!r}."
            raise TrainingRunnerConfigurationError(msg)
        result[key] = value
    return result


def _reject_nonfinite_json(value: object, path: str = "configuration") -> None:
    """Reject finite-looking JSON numbers that overflowed during decoding.

    Raises:
        TrainingRunnerConfigurationError: If a decoded float is non-finite.
    """
    if type(value) is float and not math.isfinite(value):
        msg = f"{path} contains a non-finite number."
        raise TrainingRunnerConfigurationError(msg)
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_json(item, f"{path}[{index}]")


def _decode_json_object(payload: str, name: str) -> dict[str, object]:
    """Decode a strict finite JSON object.

    Returns:
        The decoded string-keyed object.

    Raises:
        TrainingRunnerConfigurationError: If the payload is invalid, non-finite,
            duplicated, or not an object.
    """
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        msg = f"Could not decode {name}: {error}."
        raise TrainingRunnerConfigurationError(msg) from error
    if not isinstance(decoded, dict):
        msg = f"{name} must be a JSON object."
        raise TrainingRunnerConfigurationError(msg)
    _reject_nonfinite_json(decoded, name)
    return cast("dict[str, object]", decoded)


def load_configuration_file(path: Path) -> dict[str, object]:
    """Load a strict partial WP22 configuration.

    Returns:
        Normalized top-level configuration overrides.

    Raises:
        TypeError: If ``path`` is not a :class:`pathlib.Path`.
        TrainingRunnerConfigurationError: If the file cannot be read or its
            strict configuration schema is invalid.
    """
    if not isinstance(path, Path):
        msg = "path must be a pathlib.Path."
        raise TypeError(msg)
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        msg = f"Could not read JSON configuration {path}: {error}."
        raise TrainingRunnerConfigurationError(msg) from error
    decoded = _decode_json_object(payload, "training runner configuration")
    unknown = sorted(set(decoded) - _CONFIGURATION_KEYS)
    if unknown:
        msg = f"Unknown training runner configuration fields: {unknown}."
        raise TrainingRunnerConfigurationError(msg)
    configuration_format = decoded.get("format", TRAINING_RUNNER_CONFIGURATION_FORMAT)
    if configuration_format != TRAINING_RUNNER_CONFIGURATION_FORMAT:
        msg = f"format must be {TRAINING_RUNNER_CONFIGURATION_FORMAT!r}."
        raise TrainingRunnerConfigurationError(msg)
    for alias, canonical in _CONFIGURATION_ALIASES.items():
        if alias in decoded and canonical in decoded:
            msg = f"Configuration cannot contain both {alias!r} and {canonical!r}."
            raise TrainingRunnerConfigurationError(msg)
        if alias in decoded:
            decoded[canonical] = decoded.pop(alias)
    return decoded


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the public, non-abbreviating WP22 argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.state_preparation.training_runner",
        description="Validate and run checksum-sealed WP22 Phase II training plans.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, help="Strict JSON configuration file.")
    parser.add_argument("--preset", choices=TRAINING_PRESETS)
    parser.add_argument("--pipeline", "--training-pipeline", dest="pipeline_path", type=Path)
    parser.add_argument("--method", dest="method_id")
    parser.add_argument("--stage-depth", "--stage-depths", dest="stage_depths", type=int, action="append")
    parser.add_argument("--stage-budget", "--stage-budgets", dest="stage_budgets", type=int, action="append")
    parser.add_argument("--training-noise-id")
    parser.add_argument("--training-noise-strength", "--training-noise-strength-scale", type=float)
    parser.add_argument("--trajectory-update")
    parser.add_argument("--sampling-policy")
    parser.add_argument(
        "--training-trajectories",
        "--training-trajectory-count",
        dest="training_trajectory_count",
        type=int,
    )
    parser.add_argument(
        "--validation-trajectories",
        "--checkpoint-validation-trajectory-count",
        dest="checkpoint_validation_trajectory_count",
        type=int,
    )
    parser.add_argument("--crn-refresh-interval", type=int)
    parser.add_argument("--checkpoint-rule", "--checkpoint-selection-rule", dest="checkpoint_rule")
    parser.add_argument(
        "--target-manifest",
        "--target-population-manifest",
        dest="target_manifest_paths",
        type=Path,
        action="append",
    )
    parser.add_argument("--target-manifest-checksum", dest="target_manifest_checksums", action="append")
    parser.add_argument("--data-role", choices=RUNNABLE_DATA_ROLES)
    parser.add_argument("--native-two-qubit-cap-per-edge", type=float)
    parser.add_argument("--normalized-compute-cap", type=float)
    parser.add_argument("--preregistration", "--preregistration-path", dest="preregistration_path", type=Path)
    parser.add_argument("--preregistration-checksum")
    for option in ("resume", "overwrite", "dry-run", "fail-fast", "legacy-reproduction", "execute-expensive"):
        parser.add_argument(
            f"--{option}",
            dest=option.replace("-", "_"),
            action=argparse.BooleanOptionalAction,
            default=None,
        )
    parser.add_argument(
        "--executor-factory",
        help="Importable MODULE:FUNCTION returning a typed TrainingExecutorRegistry.",
    )
    parser.add_argument("--candidate", dest="candidate_paths", type=Path, action="append")
    parser.add_argument("--schedule", dest="schedule_paths", type=Path, action="append")
    parser.add_argument("--screening-manifest", dest="screening_manifest_path", type=Path)
    parser.add_argument("--screening-evidence", dest="screening_evidence_path", type=Path)
    parser.add_argument("--promotion-decision", dest="promotion_decision_path", type=Path)
    parser.add_argument("--final-seal", dest="final_seal_path", type=Path)
    parser.add_argument(
        "--configuration-execution-manifest",
        dest="configuration_execution_manifest_path",
        type=Path,
    )
    parser.add_argument("--execution-source-manifest", dest="execution_source_manifest_path", type=Path)
    parser.add_argument("--analysis-source-manifest", dest="analysis_source_manifest_path", type=Path)
    parser.add_argument("--execution-profile", dest="execution_profile_path", type=Path)
    parser.add_argument("--binding-catalog", dest="binding_catalog_path", type=Path)
    parser.add_argument("--target-configuration", dest="target_configuration_paths", type=Path, action="append")
    parser.add_argument("--sample-size-design", dest="sample_size_design_path", type=Path)
    parser.add_argument("--resource-calibration", dest="resource_calibration_path", type=Path)
    parser.add_argument(
        "--prior-target-exposure-inventory",
        "--target-exposure-inventory",
        dest="prior_target_exposure_inventory_path",
        type=Path,
    )
    parser.add_argument(
        "--expected-locked-study-head",
        dest="expected_locked_study_head_path",
        type=Path,
        help="External checksum-sealed head custody written before dispatch and reused for paper-confirm resume.",
    )
    parser.add_argument(
        "--resumability-fingerprint",
        dest="resumability_fingerprint_paths",
        type=Path,
        action="append",
    )
    parser.add_argument(
        "--external-entropy-file",
        dest="external_entropy_file_specs",
        action="append",
        help="CLI-only opaque ROLE/SCOPE=PATH target-entropy source; forbidden in JSON.",
    )
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--pilot-optimization-seed", dest="pilot_optimization_seeds", type=int, action="append")
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path)
    return parser


def parse_arguments(arguments: Sequence[str] | None = None) -> Namespace:
    """Parse CLI arguments without loading files or resolving defaults.

    Returns:
        The unresolved argument namespace.
    """
    return create_argument_parser().parse_args(arguments)


def _preset_defaults(preset: str) -> dict[str, object]:
    """Return detached operational defaults for one exact WP22 preset.

    Raises:
        TrainingRunnerConfigurationError: If ``preset`` is unsupported.
    """
    if preset not in TRAINING_PRESETS:
        msg = f"preset must be one of {TRAINING_PRESETS!r}."
        raise TrainingRunnerConfigurationError(msg)
    return {
        "preset": preset,
        "pipeline_path": None,
        "method_id": None,
        "stage_depths": [],
        "stage_budgets": [],
        "training_noise_id": None,
        "training_noise_strength": None,
        "trajectory_update": None,
        "sampling_policy": None,
        "training_trajectory_count": None,
        "checkpoint_validation_trajectory_count": None,
        "crn_refresh_interval": None,
        "checkpoint_rule": None,
        "target_manifest_paths": [],
        "target_manifest_checksums": [],
        "data_role": None,
        "native_two_qubit_cap_per_edge": None,
        "normalized_compute_cap": None,
        "preregistration_path": DEFAULT_PREREGISTRATION_PATH,
        "preregistration_checksum": TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
        "resume": False,
        "overwrite": False,
        "dry_run": False,
        "fail_fast": False,
        "legacy_reproduction": False,
        "execute_expensive": False,
        "executor_factory": None,
        "candidate_paths": [],
        "schedule_paths": [],
        "screening_manifest_path": None,
        "screening_evidence_path": None,
        "promotion_decision_path": None,
        "final_seal_path": None,
        "configuration_execution_manifest_path": None,
        "execution_source_manifest_path": None,
        "analysis_source_manifest_path": None,
        "execution_profile_path": None,
        "binding_catalog_path": None,
        "target_configuration_paths": [],
        "sample_size_design_path": None,
        "resource_calibration_path": None,
        "prior_target_exposure_inventory_path": None,
        "expected_locked_study_head_path": None,
        "resumability_fingerprint_paths": [],
        "external_entropy_file_specs": [],
        "repository_root": DEFAULT_REPOSITORY_ROOT,
        "pilot_optimization_seeds": list(DEFAULT_PILOT_OPTIMIZATION_SEEDS) if preset == "paper-pilot" else [],
        "output_dir": DEFAULT_TRAINING_OUTPUT_ROOT / preset,
    }


def _boolean(value: object, name: str) -> bool:
    """Return an exact Boolean configuration value.

    Raises:
        TrainingRunnerConfigurationError: If ``value`` is not a Boolean.
    """
    if type(value) is not bool:
        msg = f"{name} must be a boolean."
        raise TrainingRunnerConfigurationError(msg)
    return value


def _optional_text(value: object, name: str) -> str | None:
    """Return an optional nonempty string.

    Raises:
        TrainingRunnerConfigurationError: If a supplied value is not nonempty text.
    """
    if value is None:
        return None
    if type(value) is not str or not value:
        msg = f"{name} must be a nonempty string."
        raise TrainingRunnerConfigurationError(msg)
    return value


def _optional_executor_factory(value: object) -> str | None:
    """Return a syntactically explicit executor-factory reference.

    Raises:
        TrainingRunnerConfigurationError: If a supplied reference does not use
            the exact module-and-function syntax.
    """
    factory = _optional_text(value, "executor_factory")
    if factory is not None and _EXECUTOR_FACTORY_PATTERN.fullmatch(factory) is None:
        msg = "executor_factory must use the exact 'module.path:function_name' syntax."
        raise TrainingRunnerConfigurationError(msg)
    return factory


def _exact_int(value: object, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    """Return an exact bounded integer.

    Raises:
        TrainingRunnerConfigurationError: If ``value`` is not in the requested range.
    """
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        msg = f"{name} must be an integer in the configured range."
        raise TrainingRunnerConfigurationError(msg)
    return value


def _optional_int(value: object, name: str, *, minimum: int = 0) -> int | None:
    """Return an optional exact bounded integer."""
    return None if value is None else _exact_int(value, name, minimum=minimum)


def _optional_float(value: object, name: str, *, strictly_positive: bool = False) -> float | None:
    """Return an optional finite built-in number.

    Raises:
        TrainingRunnerConfigurationError: If a supplied value is nonnumeric,
            non-finite, negative, or not strictly positive when required.
    """
    if value is None:
        return None
    if type(value) not in {int, float}:
        msg = f"{name} must be a finite number."
        raise TrainingRunnerConfigurationError(msg)
    result = float(cast("int | float", value))
    invalid = not math.isfinite(result) or result < 0.0 or (strictly_positive and result <= 0.0)
    if invalid:
        msg = f"{name} must be {'positive' if strictly_positive else 'nonnegative'} and finite."
        raise TrainingRunnerConfigurationError(msg)
    return result


def _items(value: object, name: str, *, optional: bool = False) -> list[object]:
    """Normalize one scalar or JSON array into a detached list.

    Returns:
        A mutable detached list containing the supplied values.

    Raises:
        TrainingRunnerConfigurationError: If an explicitly required array is empty.
    """
    if optional and (value is None or value in ([], ())):
        return []
    result = list(value) if isinstance(value, (list, tuple)) else [value]
    if not result:
        msg = f"{name} must not be an empty array."
        raise TrainingRunnerConfigurationError(msg)
    return result


def _integer_tuple(
    value: object,
    name: str,
    *,
    maximum: int | None = None,
    unique: bool = False,
) -> tuple[int, ...]:
    """Return an ordered integer tuple with optional uniqueness.

    Raises:
        TrainingRunnerConfigurationError: If uniqueness is required and a value repeats.
    """
    values = tuple(_exact_int(item, name, maximum=maximum) for item in _items(value, name, optional=True))
    if unique and len(values) != len(set(values)):
        msg = f"{name} must not contain duplicates."
        raise TrainingRunnerConfigurationError(msg)
    return values


def _path(value: object, name: str) -> Path:
    """Return a path without probing or resolving its filesystem target.

    Raises:
        TrainingRunnerConfigurationError: If ``value`` is not a nonempty path.
    """
    if isinstance(value, Path):
        path = value
    elif type(value) is str and value:
        path = Path(value)
    else:
        msg = f"{name} must be a nonempty path string."
        raise TrainingRunnerConfigurationError(msg)
    return path.expanduser()


def _optional_path(value: object, name: str) -> Path | None:
    """Return an optional non-probed path."""
    return None if value is None else _path(value, name)


def _path_tuple(value: object, name: str) -> tuple[Path, ...]:
    """Return ordered, duplicate-free artifact paths without probing them.

    Raises:
        TrainingRunnerConfigurationError: If a path occurs more than once.
    """
    paths = tuple(_path(item, name) for item in _items(value, name, optional=True))
    if len(paths) != len(set(paths)):
        msg = f"{name} must not contain duplicate paths."
        raise TrainingRunnerConfigurationError(msg)
    return paths


def _checksum(value: object, name: str) -> str:
    """Return a syntactically valid SHA-256 checksum.

    Raises:
        TrainingRunnerConfigurationError: If ``value`` is not a canonical checksum.
    """
    if type(value) is not str or _CHECKSUM_PATTERN.fullmatch(value) is None:
        msg = f"{name} must use the 'sha256:' prefix and 64 lowercase hexadecimal digits."
        raise TrainingRunnerConfigurationError(msg)
    return value


def _checksum_tuple(value: object, name: str) -> tuple[str, ...]:
    """Return ordered, duplicate-free checksums.

    Raises:
        TrainingRunnerConfigurationError: If a checksum occurs more than once.
    """
    values = tuple(_checksum(item, name) for item in _items(value, name, optional=True))
    if len(values) != len(set(values)):
        msg = f"{name} must not contain duplicates."
        raise TrainingRunnerConfigurationError(msg)
    return values


def resolve_options(namespace: Namespace) -> TrainingRunnerOptions:
    """Apply preset defaults, JSON values, and finally explicit CLI values.

    Returns:
        Fully validated options that have not opened scientific artifacts.

    Raises:
        TrainingRunnerConfigurationError: If JSON, CLI, preset, or operational
            control values are invalid or conflict.
    """
    file_values = load_configuration_file(namespace.config) if namespace.config is not None else {}
    explicit_option_names = {_CONFIGURATION_ALIASES.get(name, name) for name in file_values if name != "format"}
    if namespace.preset is not None:
        explicit_option_names.add("preset")
    explicit_option_names.update(key for key in _CLI_DESTINATIONS if getattr(namespace, key) is not None)
    preset_value = namespace.preset if namespace.preset is not None else file_values.get("preset", "training-smoke")
    if type(preset_value) is not str or preset_value not in TRAINING_PRESETS:
        msg = f"preset must be one of {TRAINING_PRESETS!r}."
        raise TrainingRunnerConfigurationError(msg)
    resolved = _preset_defaults(preset_value)
    resolved.update({key: value for key, value in file_values.items() if key not in {"format", "preset"}})
    for key in _CLI_DESTINATIONS:
        cli_value = getattr(namespace, key)
        if cli_value is not None:
            resolved[key] = cli_value

    training_noise_id = _optional_text(resolved["training_noise_id"], "training_noise_id")
    if training_noise_id == BALLARIN_NOISE_ID:
        msg = "Ballarin noise is evaluation-only and cannot be a WP22 training objective."
        raise TrainingRunnerConfigurationError(msg)
    target_paths = _path_tuple(resolved["target_manifest_paths"], "target_manifest_paths")
    target_checksums = _checksum_tuple(resolved["target_manifest_checksums"], "target_manifest_checksums")
    if target_checksums and len(target_checksums) != len(target_paths):
        msg = "target_manifest_checksums must pair one-to-one with target_manifest_paths."
        raise TrainingRunnerConfigurationError(msg)
    resume = _boolean(resolved["resume"], "resume")
    overwrite = _boolean(resolved["overwrite"], "overwrite")
    if resume and overwrite:
        msg = "resume and overwrite are mutually exclusive."
        raise TrainingRunnerConfigurationError(msg)
    legacy = _boolean(resolved["legacy_reproduction"], "legacy_reproduction")
    expensive = _boolean(resolved["execute_expensive"], "execute_expensive")
    executor_factory = _optional_executor_factory(resolved["executor_factory"])
    historical = preset_value == "historical-layerwise-reproduction"
    if historical != legacy:
        msg = "The historical preset and explicit legacy_reproduction mode must be selected together."
        raise TrainingRunnerConfigurationError(msg)
    if expensive and preset_value not in {"historical-layerwise-reproduction", "paper-confirm"}:
        msg = "execute_expensive is accepted only for historical reproduction or sealed confirmation."
        raise TrainingRunnerConfigurationError(msg)
    if historical and executor_factory is not None:
        msg = "executor_factory is not used by historical-layerwise-reproduction."
        raise TrainingRunnerConfigurationError(msg)

    data_role = _optional_text(resolved["data_role"], "data_role")
    if data_role is not None and data_role not in RUNNABLE_DATA_ROLES:
        msg = f"data_role must be one of {RUNNABLE_DATA_ROLES!r}."
        raise TrainingRunnerConfigurationError(msg)
    pilot_seeds = _integer_tuple(
        resolved["pilot_optimization_seeds"],
        "pilot_optimization_seeds",
        maximum=2**64 - 1,
        unique=True,
    )
    if preset_value == "paper-pilot" and len(pilot_seeds) != PILOT_OPTIMIZATION_SEED_COUNT:
        msg = f"paper-pilot requires exactly {PILOT_OPTIMIZATION_SEED_COUNT} distinct optimization seeds."
        raise TrainingRunnerConfigurationError(msg)
    entropy_specs = tuple(
        cast("str", _optional_text(item, "external_entropy_file_specs item"))
        for item in _items(resolved["external_entropy_file_specs"], "external_entropy_file_specs", optional=True)
    )
    try:
        parse_entropy_file_specs(entropy_specs)
    except (TypeError, ValueError) as error:
        msg = f"Invalid external entropy file reference: {error}"
        raise TrainingRunnerConfigurationError(msg) from None
    requested_output_dir = _path(resolved["output_dir"], "output_dir")
    return TrainingRunnerOptions(
        preset=preset_value,
        pipeline_path=_optional_path(resolved["pipeline_path"], "pipeline_path"),
        method_id=_optional_text(resolved["method_id"], "method_id"),
        stage_depths=_integer_tuple(resolved["stage_depths"], "stage_depths"),
        stage_budgets=_integer_tuple(resolved["stage_budgets"], "stage_budgets"),
        training_noise_id=training_noise_id,
        training_noise_strength=_optional_float(
            resolved["training_noise_strength"],
            "training_noise_strength",
            strictly_positive=True,
        ),
        trajectory_update=_optional_text(resolved["trajectory_update"], "trajectory_update"),
        sampling_policy=_optional_text(resolved["sampling_policy"], "sampling_policy"),
        training_trajectory_count=_optional_int(
            resolved["training_trajectory_count"],
            "training_trajectory_count",
        ),
        checkpoint_validation_trajectory_count=_optional_int(
            resolved["checkpoint_validation_trajectory_count"],
            "checkpoint_validation_trajectory_count",
        ),
        crn_refresh_interval=_optional_int(
            resolved["crn_refresh_interval"],
            "crn_refresh_interval",
            minimum=1,
        ),
        checkpoint_rule=_optional_text(resolved["checkpoint_rule"], "checkpoint_rule"),
        target_manifest_paths=target_paths,
        target_manifest_checksums=target_checksums,
        data_role=data_role,
        native_two_qubit_cap_per_edge=_optional_float(
            resolved["native_two_qubit_cap_per_edge"],
            "native_two_qubit_cap_per_edge",
        ),
        normalized_compute_cap=_optional_float(
            resolved["normalized_compute_cap"],
            "normalized_compute_cap",
        ),
        preregistration_path=_path(resolved["preregistration_path"], "preregistration_path"),
        preregistration_checksum=_checksum(resolved["preregistration_checksum"], "preregistration_checksum"),
        resume=resume,
        overwrite=overwrite,
        dry_run=_boolean(resolved["dry_run"], "dry_run"),
        fail_fast=_boolean(resolved["fail_fast"], "fail_fast"),
        legacy_reproduction=legacy,
        execute_expensive=expensive,
        executor_factory=executor_factory,
        candidate_paths=_path_tuple(resolved["candidate_paths"], "candidate_paths"),
        schedule_paths=_path_tuple(resolved["schedule_paths"], "schedule_paths"),
        screening_manifest_path=_optional_path(resolved["screening_manifest_path"], "screening_manifest_path"),
        screening_evidence_path=_optional_path(resolved["screening_evidence_path"], "screening_evidence_path"),
        promotion_decision_path=_optional_path(resolved["promotion_decision_path"], "promotion_decision_path"),
        final_seal_path=_optional_path(resolved["final_seal_path"], "final_seal_path"),
        configuration_execution_manifest_path=_optional_path(
            resolved["configuration_execution_manifest_path"],
            "configuration_execution_manifest_path",
        ),
        execution_source_manifest_path=_optional_path(
            resolved["execution_source_manifest_path"],
            "execution_source_manifest_path",
        ),
        analysis_source_manifest_path=_optional_path(
            resolved["analysis_source_manifest_path"],
            "analysis_source_manifest_path",
        ),
        execution_profile_path=_optional_path(resolved["execution_profile_path"], "execution_profile_path"),
        binding_catalog_path=_optional_path(resolved["binding_catalog_path"], "binding_catalog_path"),
        target_configuration_paths=_path_tuple(
            resolved["target_configuration_paths"],
            "target_configuration_paths",
        ),
        sample_size_design_path=_optional_path(resolved["sample_size_design_path"], "sample_size_design_path"),
        resource_calibration_path=_optional_path(
            resolved["resource_calibration_path"],
            "resource_calibration_path",
        ),
        prior_target_exposure_inventory_path=_optional_path(
            resolved["prior_target_exposure_inventory_path"],
            "prior_target_exposure_inventory_path",
        ),
        resumability_fingerprint_paths=_path_tuple(
            resolved["resumability_fingerprint_paths"],
            "resumability_fingerprint_paths",
        ),
        external_entropy_file_specs=entropy_specs,
        repository_root=_path(resolved["repository_root"], "repository_root"),
        pilot_optimization_seeds=pilot_seeds,
        output_dir=requested_output_dir.resolve(),
        expected_locked_study_head_path=_optional_path(
            resolved["expected_locked_study_head_path"],
            "expected_locked_study_head_path",
        ),
        explicit_option_names=frozenset(explicit_option_names),
        output_was_cli_explicit=namespace.output_dir is not None,
        requested_output_dir=requested_output_dir,
    )


def _executor_factory_source(
    executor_factory: str,
    repository_root: Path,
) -> tuple[str, Path]:
    """Resolve a factory module to one repository-owned Python source file.

    Returns:
        The repository-relative POSIX path and resolved absolute source path.

    Raises:
        TrainingRunnerConfigurationError: If the reference is invalid, cannot
            resolve to source, or resolves outside the repository.
    """
    match = _EXECUTOR_FACTORY_PATTERN.fullmatch(executor_factory)
    if match is None:
        msg = "executor_factory must use the exact 'module.path:function_name' syntax."
        raise TrainingRunnerConfigurationError(msg)
    module_name = match.group("module")
    try:
        module_spec = importlib.util.find_spec(module_name)
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        msg = f"Could not resolve executor-factory module {module_name!r} without exposing private diagnostics."
        raise TrainingRunnerConfigurationError(msg) from None
    if module_spec is None or module_spec.origin is None or module_spec.origin in {"built-in", "frozen"}:
        msg = f"Executor-factory module {module_name!r} must resolve to a Python source file."
        raise TrainingRunnerConfigurationError(msg)
    source_path = Path(module_spec.origin).resolve()
    root = repository_root.resolve()
    try:
        relative_path = source_path.relative_to(root)
    except ValueError as error:
        msg = f"Executor-factory module {module_name!r} is outside repository_root {root}."
        raise TrainingRunnerConfigurationError(msg) from error
    if source_path.suffix != ".py" or source_path.is_symlink() or not source_path.is_file():
        msg = f"Executor-factory module {module_name!r} must be a non-symlink Python source file."
        raise TrainingRunnerConfigurationError(msg)
    return relative_path.as_posix(), source_path


def load_executor_registry(
    executor_factory: str,
    repository_root: Path,
    context: TrainingExecutionContext | ConfirmationExecutionContext,
) -> TrainingExecutorRegistry:
    """Load a production executor registry through an explicit source module.

    Factory modules provide the real method-specific scientific executors; the
    WP22 runner does not manufacture placeholder results.  The selected module
    must resolve inside ``repository_root`` so it can be included by the later
    governed execution-source lock.

    Returns:
        The typed registry returned by the selected ``factory(context)`` seam.

    Raises:
        TypeError: If ``context`` is not a complete execution context.
        TrainingRunnerConfigurationError: If source resolution, import, factory
            lookup, invocation, or the returned registry is invalid.
    """
    if not isinstance(context, (TrainingExecutionContext, ConfirmationExecutionContext)):
        msg = "context must be a TrainingExecutionContext or ConfirmationExecutionContext."
        raise TypeError(msg)
    try:
        verify_execution_source_manifest(context.execution_source_manifest, repository_root)
    except (OSError, TypeError, ValueError):
        msg = "Executor-factory source custody failed without exposing private diagnostics."
        raise TrainingRunnerConfigurationError(msg) from None
    relative_path, expected_source = _executor_factory_source(executor_factory, repository_root)
    source_roles = {
        source_file.repo_path: source_file.role for source_file in context.execution_source_manifest.source_files
    }
    if source_roles.get(relative_path) != "execution_source":
        msg = "Executor-factory source is absent from the context execution-source manifest."
        raise TrainingRunnerConfigurationError(msg)
    match = cast("re.Match[str]", _EXECUTOR_FACTORY_PATTERN.fullmatch(executor_factory))
    module_name = match.group("module")
    attribute_name = match.group("attribute")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError, RuntimeError, ValueError):
        msg = f"Could not import executor-factory module {module_name!r} without exposing private diagnostics."
        raise TrainingRunnerConfigurationError(msg) from None
    imported_source = getattr(module, "__file__", None)
    if not isinstance(imported_source, str) or Path(imported_source).resolve() != expected_source:
        msg = f"Imported executor-factory module {module_name!r} differs from its verified source path."
        raise TrainingRunnerConfigurationError(msg)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        msg = f"Executor factory {executor_factory!r} is not callable."
        raise TrainingRunnerConfigurationError(msg)
    try:
        registry = factory(context)
    except Exception:  # noqa: BLE001 - extension failures are redacted at the trust boundary
        msg = f"Executor factory {executor_factory!r} failed without exposing private diagnostics."
        raise TrainingRunnerConfigurationError(msg) from None
    if not isinstance(registry, TrainingExecutorRegistry):
        msg = f"Executor factory {executor_factory!r} must return a TrainingExecutorRegistry."
        raise TrainingRunnerConfigurationError(msg)
    return registry


def _verify_confirm_executor_factory_source(
    options: TrainingRunnerOptions,
    execution_manifest: ExecutionSourceManifest,
) -> None:
    """Require a selected extension factory in the verified execution source lock.

    Raises:
        TrainingRunnerConfigurationError: If the factory source is outside the
            verified execution-source universe.
    """
    if options.executor_factory is None:
        return
    relative_path, _source_path = _executor_factory_source(options.executor_factory, options.repository_root)
    source_roles = {source_file.repo_path: source_file.role for source_file in execution_manifest.source_files}
    if source_roles.get(relative_path) != "execution_source":
        msg = f"Executor-factory source {relative_path!r} is not an execution_source in the verified governed manifest."
        raise TrainingRunnerConfigurationError(msg)


def _read_artifact(path: Path, name: str) -> str:
    """Read one artifact with contextual configuration errors.

    Returns:
        The exact UTF-8 artifact text.

    Raises:
        TrainingRunnerConfigurationError: If the artifact cannot be read as UTF-8.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        msg = f"Could not read {name} {path}: {error}."
        raise TrainingRunnerConfigurationError(msg) from error


def _decode_artifact(path: Path, name: str, decoder: Callable[[str], object]) -> object:
    """Read and decode one immutable artifact.

    Returns:
        The typed artifact returned by ``decoder``.

    Raises:
        TrainingRunnerConfigurationError: If reading or decoding fails.
    """
    payload = _read_artifact(path, name)
    try:
        return decoder(payload)
    except (TypeError, ValueError) as error:
        msg = f"Invalid {name} {path}: {error}."
        raise TrainingRunnerConfigurationError(msg) from error


def _load_pipeline(path: Path) -> _PipelineView:
    """Load a sealed template or concrete pipeline without normalizing it.

    Returns:
        A uniform sealed pipeline view.

    Raises:
        TrainingRunnerConfigurationError: If the artifact or schema is invalid.
    """
    payload = _read_artifact(path, "training pipeline")
    mapping = _decode_json_object(payload, "training pipeline")
    schema = mapping.get("schema_version")
    if schema == TRAINING_PIPELINE_TEMPLATE_SCHEMA_VERSION:
        try:
            template = TrainingPipelineTemplate.from_dict(mapping)
            reject_ballarin_training(template)
            return _PipelineView(template=template, concrete=None)
        except (TypeError, ValueError) as error:
            msg = f"Invalid training pipeline {path}: {error}."
            raise TrainingRunnerConfigurationError(msg) from error
    if schema == TRAINING_PIPELINE_CONFIG_SCHEMA_VERSION:
        try:
            concrete = TrainingPipelineConfig.from_dict(mapping)
            reject_ballarin_training(concrete.template)
            return _PipelineView(template=concrete.template, concrete=concrete)
        except (TypeError, ValueError) as error:
            msg = f"Invalid training pipeline {path}: {error}."
            raise TrainingRunnerConfigurationError(msg) from error
    msg = "training pipeline uses an unsupported schema_version."
    raise TrainingRunnerConfigurationError(msg)


def _load_preregistration(options: TrainingRunnerOptions) -> InitialPreregistration:
    """Load and verify the immutable preregistration assertion.

    Returns:
        The trusted immutable preregistration.

    Raises:
        TrainingRunnerConfigurationError: If loading or checksum verification fails.
    """
    try:
        preregistration = load_initial_preregistration(options.preregistration_path)
    except (OSError, TypeError, ValueError) as error:
        msg = f"Invalid preregistration {options.preregistration_path}: {error}."
        raise TrainingRunnerConfigurationError(msg) from error
    if preregistration.content_checksum != options.preregistration_checksum:
        msg = "preregistration_checksum does not match the sealed preregistration."
        raise TrainingRunnerConfigurationError(msg)
    return preregistration


def _require_paths(paths: tuple[Path, ...], name: str, *, count: int | None = None) -> None:
    """Require a nonempty or exact artifact-path cardinality.

    Raises:
        TrainingRunnerConfigurationError: If the path cardinality is invalid.
    """
    invalid = not paths if count is None else len(paths) != count
    if invalid:
        expected = "at least one" if count is None else f"exactly {count}"
        msg = f"{name} requires {expected} path(s)."
        raise TrainingRunnerConfigurationError(msg)


def _load_targets(options: TrainingRunnerOptions) -> tuple[TargetPopulationManifest, ...]:
    """Load target manifests after the caller has passed any custody gate.

    Returns:
        The ordered typed target manifests.

    Raises:
        TrainingRunnerConfigurationError: If a manifest or its asserted identity
            is invalid.
    """
    manifests = tuple(
        cast(
            "TargetPopulationManifest",
            _decode_artifact(path, "target-population manifest", TargetPopulationManifest.from_json),
        )
        for path in options.target_manifest_paths
    )
    if options.target_manifest_checksums:
        for expected, actual in zip(options.target_manifest_checksums, manifests, strict=True):
            if expected != actual.content_checksum:
                msg = "target_manifest_checksum does not match its sealed manifest."
                raise TrainingRunnerConfigurationError(msg)
    if any(manifest.preregistration_checksum != options.preregistration_checksum for manifest in manifests):
        msg = "A target manifest belongs to a different preregistration."
        raise TrainingRunnerConfigurationError(msg)
    if options.data_role is not None and any(manifest.data_role != options.data_role for manifest in manifests):
        msg = "data_role does not match every supplied target manifest."
        raise TrainingRunnerConfigurationError(msg)
    return manifests


def _load_candidates(options: TrainingRunnerOptions) -> tuple[WP22CandidateConfiguration, ...]:
    """Load ordered, distinct candidate configuration artifacts.

    Returns:
        The ordered typed candidate configurations.

    Raises:
        TrainingRunnerConfigurationError: If an artifact is invalid or duplicated.
    """
    candidates = tuple(
        cast(
            "WP22CandidateConfiguration",
            _decode_artifact(path, "candidate configuration", WP22CandidateConfiguration.from_json),
        )
        for path in options.candidate_paths
    )
    if len({item.content_checksum for item in candidates}) != len(candidates):
        msg = "candidate_paths must identify distinct candidate configurations."
        raise TrainingRunnerConfigurationError(msg)
    return candidates


def _load_schedules(options: TrainingRunnerOptions) -> tuple[TrainingStrategySchedule, ...]:
    """Load ordered, distinct strategy schedules.

    Returns:
        The ordered typed strategy schedules.

    Raises:
        TrainingRunnerConfigurationError: If an artifact is invalid or duplicated.
    """
    schedules = tuple(
        cast(
            "TrainingStrategySchedule",
            _decode_artifact(path, "training strategy schedule", TrainingStrategySchedule.from_json),
        )
        for path in options.schedule_paths
    )
    checksums = {item.content_checksum for item in schedules}
    if len(checksums) != len(schedules):
        msg = "schedule_paths must identify distinct strategy schedules."
        raise TrainingRunnerConfigurationError(msg)
    return schedules


def _require_schedule_bindings(
    candidates: tuple[WP22CandidateConfiguration, ...],
    schedules: tuple[TrainingStrategySchedule, ...],
) -> None:
    """Require supplied schedules to be the exact candidate schedule universe.

    Raises:
        TrainingRunnerConfigurationError: If the candidate and schedule universes differ.
    """
    if not candidates and not schedules:
        return
    if not schedules:
        msg = "Every strategy candidate requires its complete sealed TrainingStrategySchedule."
        raise TrainingRunnerConfigurationError(msg)
    expected = {candidate.strategy_schedule_checksum for candidate in candidates}
    actual = {schedule.content_checksum for schedule in schedules}
    if expected != actual:
        msg = "Candidate and strategy-schedule checksum universes differ."
        raise TrainingRunnerConfigurationError(msg)


def _pipeline_policies(template: TrainingPipelineTemplate) -> tuple[Mapping[str, object], ...]:
    """Return immutable template policies in declared stage order."""
    return tuple(stage.stage_policy for stage in template.stages)


def _noisy_policies(template: TrainingPipelineTemplate) -> tuple[Mapping[str, object], ...]:
    """Return only stages with a genuine training-noise objective."""
    return tuple(
        policy
        for policy in _pipeline_policies(template)
        if cast("str", policy["training_noise_id"]) != NOISELESS_NOISE_ID
    )


def _assert_equal(actual: object, expected: object, name: str) -> None:
    """Require an assertion value to match a sealed artifact exactly.

    Raises:
        TrainingRunnerConfigurationError: If the assertion differs.
    """
    if actual != expected:
        msg = f"{name}={expected!r} does not match sealed artifact value {actual!r}."
        raise TrainingRunnerConfigurationError(msg)


def _validate_pipeline_assertions(options: TrainingRunnerOptions, pipeline: _PipelineView | None) -> None:
    """Check every pipeline-scoped CLI value without changing the pipeline.

    Raises:
        TrainingRunnerConfigurationError: If an assertion is unsupported or differs.
    """
    pipeline_assertions = any((
        options.stage_depths,
        options.stage_budgets,
        options.trajectory_update is not None,
        options.training_trajectory_count is not None,
        options.checkpoint_validation_trajectory_count is not None,
        options.checkpoint_rule is not None,
    ))
    if pipeline is None:
        if pipeline_assertions:
            msg = "Stage, trajectory-update, and checkpoint assertions require --pipeline."
            raise TrainingRunnerConfigurationError(msg)
        return
    template = pipeline.template
    policies = _pipeline_policies(template)
    if options.stage_depths:
        depths: list[int] = []
        for policy in policies:
            match = _TOPOLOGY_DEPTH_PATTERN.search(cast("str", policy["output_topology_id"]))
            if match is None:
                msg = "stage_depths can be asserted only for depth-addressed pipeline topologies."
                raise TrainingRunnerConfigurationError(msg)
            depths.append(int(match.group("depth")))
        _assert_equal(tuple(depths), options.stage_depths, "stage_depths")
    if options.stage_budgets:
        _assert_equal(
            tuple(cast("int", policy["iteration_budget"]) for policy in policies),
            options.stage_budgets,
            "stage_budgets",
        )

    relevant = _noisy_policies(template)
    if options.training_noise_id is not None:
        selected = policies if options.training_noise_id == NOISELESS_NOISE_ID else relevant
        if not selected:
            msg = "training_noise_id asserts noisy training, but the sealed pipeline has no noisy stage."
            raise TrainingRunnerConfigurationError(msg)
        _assert_equal(
            tuple(cast("str", policy["training_noise_id"]) for policy in selected),
            (options.training_noise_id,) * len(selected),
            "training_noise_id",
        )
    for expected, key, name in (
        (options.training_noise_strength, "noise_strength_scale", "training_noise_strength"),
        (options.trajectory_update, "trajectory_update", "trajectory_update"),
        (options.sampling_policy, "sampling_policy", "sampling_policy"),
        (options.training_trajectory_count, "trajectory_count", "training_trajectory_count"),
        (options.crn_refresh_interval, "crn_refresh_interval", "crn_refresh_interval"),
    ):
        if expected is not None:
            if not relevant:
                msg = f"{name} requires at least one sealed noisy-training stage."
                raise TrainingRunnerConfigurationError(msg)
            _assert_equal(tuple(policy[key] for policy in relevant), (expected,) * len(relevant), name)
    validation_policies = tuple(
        cast("Mapping[str, object]", policy["checkpoint_validation_policy"])
        for policy in policies
        if cast("Mapping[str, object]", policy["checkpoint_validation_policy"])["trajectory_count"] != 0
    )
    if options.checkpoint_validation_trajectory_count is not None:
        if not validation_policies and options.checkpoint_validation_trajectory_count != 0:
            msg = "The sealed pipeline has no enabled checkpoint validation."
            raise TrainingRunnerConfigurationError(msg)
        actual = tuple(policy["trajectory_count"] for policy in validation_policies)
        expected = (options.checkpoint_validation_trajectory_count,) * len(validation_policies)
        _assert_equal(actual, expected, "checkpoint_validation_trajectory_count")
    if options.checkpoint_rule is not None:
        if not validation_policies:
            msg = "checkpoint_rule requires an enabled sealed checkpoint policy."
            raise TrainingRunnerConfigurationError(msg)
        _assert_equal(
            tuple(policy["selection_rule"] for policy in validation_policies),
            (options.checkpoint_rule,) * len(validation_policies),
            "checkpoint_rule",
        )


def _validate_schedule_assertions(
    options: TrainingRunnerOptions,
    schedules: tuple[TrainingStrategySchedule, ...],
    *,
    pipeline_present: bool,
) -> None:
    """Check schedule-scoped assertions not already grounded in a pipeline.

    Raises:
        TrainingRunnerConfigurationError: If an assertion is unsupported or differs.
    """
    if not schedules:
        if not pipeline_present and any((
            options.training_noise_id is not None,
            options.training_noise_strength is not None,
            options.sampling_policy is not None,
            options.crn_refresh_interval is not None,
        )):
            msg = "Noise and sampling assertions require a sealed pipeline or strategy schedule."
            raise TrainingRunnerConfigurationError(msg)
        return
    if options.training_noise_id is not None:
        for schedule in schedules:
            identities = (
                {NOISELESS_NOISE_ID}
                if schedule.training_noise.mode == "noiseless"
                else {component.noise_id for component in schedule.training_noise.components}
            )
            if options.training_noise_id not in identities:
                msg = "training_noise_id is absent from a sealed strategy schedule."
                raise TrainingRunnerConfigurationError(msg)
    if options.training_noise_strength is not None:
        for schedule in schedules:
            _assert_equal(
                schedule.noise_continuation.target_strength_scale,
                options.training_noise_strength,
                "training_noise_strength",
            )
    if options.sampling_policy is not None:
        schedule_kind = {
            "crn_fixed": "fixed_crn",
            "crn_refresh": "periodic_full_refresh",
        }.get(options.sampling_policy, options.sampling_policy)
        for schedule in schedules:
            _assert_equal(schedule.sampling_policy.kind, schedule_kind, "sampling_policy")
    if options.crn_refresh_interval is not None:
        for schedule in schedules:
            _assert_equal(
                schedule.sampling_policy.refresh_interval,
                options.crn_refresh_interval,
                "crn_refresh_interval",
            )


def _validate_resource_assertions(
    options: TrainingRunnerOptions,
    preregistration: InitialPreregistration,
    final_seal: FinalConfirmationSeal | None,
) -> None:
    """Validate resource values against their authoritative sealed records.

    Raises:
        TrainingRunnerConfigurationError: If a resource assertion is unsealed or differs.
    """
    if options.native_two_qubit_cap_per_edge is not None:
        _assert_equal(
            preregistration.primary_resource_constraint["cap_per_chain_edge"],
            options.native_two_qubit_cap_per_edge,
            "native_two_qubit_cap_per_edge",
        )
        if final_seal is not None:
            _assert_equal(
                final_seal.primary_resource_budget["cap_per_chain_edge"],
                options.native_two_qubit_cap_per_edge,
                "native_two_qubit_cap_per_edge",
            )
    if options.normalized_compute_cap is not None:
        if final_seal is None:
            msg = "normalized_compute_cap is authoritative only in a valid final seal."
            raise TrainingRunnerConfigurationError(msg)
        _assert_equal(
            final_seal.primary_resource_budget["normalized_compute_cap"],
            options.normalized_compute_cap,
            "normalized_compute_cap",
        )


def _validate_method_assertion(
    options: TrainingRunnerOptions,
    pipeline: _PipelineView | None,
    candidates: tuple[WP22CandidateConfiguration, ...],
    final_seal: FinalConfirmationSeal | None,
) -> None:
    """Require a method assertion to exist in the sealed method universe.

    Raises:
        TrainingRunnerConfigurationError: If the method assertion is ungrounded or absent.
    """
    if options.method_id is None:
        return
    if candidates:
        identities = {candidate.method_id for candidate in candidates}
    elif final_seal is not None:
        identities = {final_seal.promoted_method_id, *(item.method_id for item in final_seal.comparators)}
    elif pipeline is not None:
        identities = {pipeline.template.method_id}
    else:
        msg = "method_id requires a pipeline, candidate configuration, or final seal."
        raise TrainingRunnerConfigurationError(msg)
    if options.method_id not in identities:
        msg = f"method_id={options.method_id!r} is absent from the sealed method universe."
        raise TrainingRunnerConfigurationError(msg)


def _validate_pipeline_candidate_binding(
    pipeline: _PipelineView | None,
    candidates: tuple[WP22CandidateConfiguration, ...],
) -> None:
    """Require an asserted pipeline to be implemented by a supplied candidate.

    Raises:
        TrainingRunnerConfigurationError: If the pipeline has no candidate binding.
    """
    if pipeline is None or not candidates:
        return
    checksum = pipeline.template.configuration_checksum
    if not any(candidate.implementation_checksum == checksum for candidate in candidates):
        msg = "The supplied pipeline is not the implementation of any supplied candidate configuration."
        raise TrainingRunnerConfigurationError(msg)


def _load_nonconfirm_artifacts(
    options: TrainingRunnerOptions,
) -> tuple[
    _PipelineView | None,
    tuple[WP22CandidateConfiguration, ...],
    tuple[TrainingStrategySchedule, ...],
    tuple[TargetPopulationManifest, ...],
]:
    """Load ordinary planning artifacts after basic option validation.

    Returns:
        The optional pipeline, candidates, schedules, and target manifests.
    """
    pipeline = None if options.pipeline_path is None else _load_pipeline(options.pipeline_path)
    candidates = _load_candidates(options)
    schedules = _load_schedules(options)
    targets = _load_targets(options)
    _require_schedule_bindings(candidates, schedules)
    _validate_pipeline_candidate_binding(pipeline, candidates)
    _validate_pipeline_assertions(options, pipeline)
    _validate_schedule_assertions(options, schedules, pipeline_present=pipeline is not None)
    _validate_method_assertion(options, pipeline, candidates, None)
    return pipeline, candidates, schedules, targets


def _require_optional_artifact(path: Path | None, name: str) -> Path:
    """Return a required optional path without accessing it.

    Raises:
        TrainingRunnerConfigurationError: If the required path is absent.
    """
    if path is None:
        msg = f"paper-confirm requires {name}."
        raise TrainingRunnerConfigurationError(msg)
    return path


def _require_confirmation_reveal_opt_in(options: TrainingRunnerOptions) -> None:
    """Reject confirmation planning that could consume held inputs without execution consent.

    Raises:
        TrainingRunnerConfigurationError: If confirmation lacks the explicit
            held-input/expensive-execution opt-in.
    """
    if not options.execute_expensive:
        msg = "Real confirmation execution requires the additional --execute-expensive opt-in."
        raise TrainingRunnerConfigurationError(msg)


def _validate_paper_confirm_output_path(requested: Path, resolved: Path) -> None:
    """Validate one explicit output path without following its lexical symlinks.

    Raises:
        TrainingRunnerConfigurationError: If the path is linked, inconsistent,
            or not an absent or existing directory.
    """
    requested_absolute = requested.absolute()
    if requested_absolute.resolve() != resolved:
        msg = "paper-confirm output resolution differs from the explicit CLI --output."
        raise TrainingRunnerConfigurationError(msg)
    current = Path(requested_absolute.anchor)
    for component in requested_absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            msg = "paper-confirm output cannot contain a symlink component."
            raise TrainingRunnerConfigurationError(msg)
        if current != requested_absolute and current.exists() and not current.is_dir():
            msg = "paper-confirm output has a non-directory parent component."
            raise TrainingRunnerConfigurationError(msg)
    if resolved.exists() and not resolved.is_dir():
        msg = "paper-confirm output must be absent or an existing directory."
        raise TrainingRunnerConfigurationError(msg)


def _validate_paper_confirm_output_roles(output_root: Path) -> None:
    """Require a confirmation-only, non-symlink output-role namespace.

    Raises:
        TrainingRunnerConfigurationError: If the role namespace is linked,
            mixed with another role, or not a directory.
    """
    roles = output_root / "roles"
    if roles.is_symlink() or (roles.exists() and not roles.is_dir()):
        msg = "paper-confirm output roles must be absent or a non-symlink directory."
        raise TrainingRunnerConfigurationError(msg)
    if roles.exists() and any(path.name != "confirmatory" for path in roles.iterdir()):
        msg = "Development, screening, and confirmation outputs cannot share one output root."
        raise TrainingRunnerConfigurationError(msg)
    confirmatory = roles / "confirmatory"
    if confirmatory.is_symlink() or (confirmatory.exists() and not confirmatory.is_dir()):
        msg = "paper-confirm confirmatory output must be absent or a non-symlink directory."
        raise TrainingRunnerConfigurationError(msg)


def _validate_confirmation_staging_parent(output_root: Path) -> None:
    """Require crash-safe off-tree staging on the output filesystem.

    Raises:
        TrainingRunnerConfigurationError: If the staging parent cannot support
            same-filesystem atomic publication before held inputs are read.
    """
    staging_parent = output_root.parent
    if not staging_parent.exists() or (
        staging_parent.is_symlink() or not staging_parent.is_dir() or not os.access(staging_parent, os.W_OK | os.X_OK)
    ):
        msg = "paper-confirm output parent must be a writable non-symlink directory."
        raise TrainingRunnerConfigurationError(msg)
    if output_root.exists() and output_root.stat().st_dev != staging_parent.stat().st_dev:
        msg = "paper-confirm output and off-tree staging parent must share one filesystem."
        raise TrainingRunnerConfigurationError(msg)


def _preflight_paper_confirm_request(options: TrainingRunnerOptions) -> None:
    """Reject caller-selected science and unsafe output custody before held reads.

    The confirmation route accepts artifact locations and the minimum operational
    controls needed to execute those artifacts. It does not accept redundant
    scientific assertions that could become an unblinding-time choice.

    Raises:
        TrainingRunnerConfigurationError: If a forbidden option is present, no
            explicit CLI output was supplied, or the output root is not safely
            isolated from the repository and nonconfirmatory roles.
    """
    forbidden = set(options.explicit_option_names - _PAPER_CONFIRM_ALLOWED_EXPLICIT_OPTIONS)
    nondefault_scientific_options = (
        ("pipeline_path", options.pipeline_path is not None),
        ("method_id", options.method_id is not None),
        ("stage_depths", bool(options.stage_depths)),
        ("stage_budgets", bool(options.stage_budgets)),
        ("training_noise_id", options.training_noise_id is not None),
        ("training_noise_strength", options.training_noise_strength is not None),
        ("trajectory_update", options.trajectory_update is not None),
        ("sampling_policy", options.sampling_policy is not None),
        ("training_trajectory_count", options.training_trajectory_count is not None),
        (
            "checkpoint_validation_trajectory_count",
            options.checkpoint_validation_trajectory_count is not None,
        ),
        ("crn_refresh_interval", options.crn_refresh_interval is not None),
        ("checkpoint_rule", options.checkpoint_rule is not None),
        ("data_role", options.data_role is not None),
        ("native_two_qubit_cap_per_edge", options.native_two_qubit_cap_per_edge is not None),
        ("normalized_compute_cap", options.normalized_compute_cap is not None),
        ("overwrite", options.overwrite),
        ("fail_fast", options.fail_fast),
        ("legacy_reproduction", options.legacy_reproduction),
        ("executor_factory", options.executor_factory is not None),
        ("candidate_paths", bool(options.candidate_paths)),
        ("schedule_paths", bool(options.schedule_paths)),
        ("execution_profile_path", options.execution_profile_path is not None),
        ("resumability_fingerprint_paths", bool(options.resumability_fingerprint_paths)),
        ("pilot_optimization_seeds", bool(options.pilot_optimization_seeds)),
    )
    forbidden.update(name for name, present in nondefault_scientific_options if present)
    if forbidden:
        msg = f"paper-confirm forbids caller-selected or redundant options: {sorted(forbidden)}."
        raise TrainingRunnerConfigurationError(msg)
    if options.resume and options.expected_locked_study_head_path is None:
        msg = "paper-confirm --resume requires --expected-locked-study-head from external custody."
        raise TrainingRunnerConfigurationError(msg)
    if not options.output_was_cli_explicit or options.requested_output_dir is None:
        msg = "paper-confirm requires an explicit CLI --output outside repository_root."
        raise TrainingRunnerConfigurationError(msg)

    output_root = options.output_dir
    try:
        _validate_paper_confirm_output_path(options.requested_output_dir, output_root)
    except OSError:
        msg = "paper-confirm output custody could not be validated safely."
        raise TrainingRunnerConfigurationError(msg) from None

    try:
        repository_root = options.repository_root.resolve()
    except OSError:
        msg = "paper-confirm repository_root could not be resolved safely."
        raise TrainingRunnerConfigurationError(msg) from None
    output_inside_repository = output_root.is_relative_to(repository_root)
    repository_inside_output = repository_root.is_relative_to(output_root)
    if output_inside_repository or repository_inside_output:
        msg = "paper-confirm --output must be disjoint from and outside repository_root."
        raise TrainingRunnerConfigurationError(msg)
    if options.expected_locked_study_head_path is not None:
        try:
            retained_head = options.expected_locked_study_head_path.resolve()
        except OSError:
            msg = "Externally retained study head path could not be resolved safely."
            raise TrainingRunnerConfigurationError(msg) from None
        if retained_head.is_relative_to(output_root) or retained_head.is_relative_to(repository_root):
            msg = "Externally retained study head must be outside both output and repository custody."
            raise TrainingRunnerConfigurationError(msg)
        parent = retained_head.parent
        if parent.is_symlink() or not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            msg = "Externally retained study head requires a writable non-symlink parent directory."
            raise TrainingRunnerConfigurationError(msg)
        if retained_head.is_symlink():
            msg = "Externally retained study head cannot be a symlink."
            raise TrainingRunnerConfigurationError(msg)
        if retained_head.exists():
            metadata = retained_head.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                msg = "Externally retained study head must be a single-link regular file."
                raise TrainingRunnerConfigurationError(msg)

    try:
        _validate_paper_confirm_output_roles(output_root)
        _validate_confirmation_staging_parent(output_root)
    except OSError:
        msg = "paper-confirm output roles could not be validated safely."
        raise TrainingRunnerConfigurationError(msg) from None


def _pilot_optimization_seeds(
    options: TrainingRunnerOptions,
    preregistration: InitialPreregistration,
) -> tuple[int, ...]:
    """Require the exact checksum-derived preregistered pilot seed schedule.

    Returns:
        The exact checksum-derived pilot seed tuple.

    Raises:
        TrainingRunnerConfigurationError: If the declaration or supplied seeds differ.
    """
    policy = preregistration.target_population_policy
    allocation = cast("Mapping[str, object]", policy["role_allocation_policy"])
    declared = allocation["pilot_optimizer_seed_count"]
    if type(declared) is not int or declared != PILOT_OPTIMIZATION_SEED_COUNT:
        msg = f"Preregistration must declare exactly {PILOT_OPTIMIZATION_SEED_COUNT} pilot optimization seeds."
        raise TrainingRunnerConfigurationError(msg)
    expected = derive_pilot_optimization_seeds(preregistration.content_checksum, declared)
    if options.pilot_optimization_seeds != expected:
        msg = "paper-pilot optimization seeds must equal the exact checksum-derived preregistered schedule."
        raise TrainingRunnerConfigurationError(msg)
    return expected


def _required_context_path(path: Path | None, option: str) -> Path:
    """Return one required WP22D artifact path without opening it.

    Raises:
        TrainingRunnerConfigurationError: If the context artifact path is absent.
    """
    if path is None:
        msg = f"A complete WP22D execution context requires {option}."
        raise TrainingRunnerConfigurationError(msg)
    return path


def _load_training_execution_context(options: TrainingRunnerOptions) -> TrainingExecutionContext:
    """Load, cross-bind, and authorize one complete nonconfirmatory context.

    Returns:
        The non-serializable context ready for source-byte preflight.

    Raises:
        TrainingRunnerConfigurationError: If any required artifact, external
            entropy slot, authorization, or WP22A--C identity differs.
    """
    if options.preset not in {"training-smoke", "paper-pilot", "paper-screen"}:
        msg = "WP22D execution profiles cover training-smoke, paper-pilot, and paper-screen only."
        raise TrainingRunnerConfigurationError(msg)
    preregistration = _load_preregistration(options)
    pilot_seeds = _pilot_optimization_seeds(options, preregistration) if options.preset == "paper-pilot" else ()
    if options.preset != "paper-screen" and (
        options.screening_manifest_path is not None or options.sample_size_design_path is not None
    ):
        msg = "Screening manifests and sample-size designs are accepted only by paper-screen."
        raise TrainingRunnerConfigurationError(msg)
    profile_path = _required_context_path(options.execution_profile_path, "--execution-profile")
    catalog_path = _required_context_path(options.binding_catalog_path, "--binding-catalog")
    source_path = _required_context_path(
        options.execution_source_manifest_path,
        "--execution-source-manifest",
    )
    _require_paths(options.target_manifest_paths, options.preset, count=2 if options.preset == "paper-pilot" else 1)
    _require_paths(
        options.target_configuration_paths,
        "--target-configuration",
        count=2 if options.preset == "paper-pilot" else 1,
    )
    _require_paths(options.resumability_fingerprint_paths, "--resumability-fingerprint")
    source_manifest = cast(
        "ExecutionSourceManifest",
        _decode_artifact(source_path, "execution-source manifest", ExecutionSourceManifest.from_json),
    )
    try:
        verify_execution_source_manifest(source_manifest, options.repository_root)
    except (OSError, TypeError, ValueError):
        msg = "WP22D execution-source custody failed before executable bindings were loaded."
        raise TrainingRunnerConfigurationError(msg) from None
    profile = cast(
        "TrainingExecutionProfile",
        _decode_artifact(profile_path, "training execution profile", TrainingExecutionProfile.from_json),
    )
    catalog = cast(
        "RepositoryBindingCatalog",
        _decode_artifact(catalog_path, "repository binding catalog", RepositoryBindingCatalog.from_json),
    )
    if catalog.profile != profile or profile.preset != options.preset:
        msg = "Execution profile and repository binding catalog differ or belong to another preset."
        raise TrainingRunnerConfigurationError(msg)
    configs = tuple(
        cast(
            "TargetPopulationConfig",
            _decode_artifact(path, "target-population configuration", TargetPopulationConfig.from_json),
        )
        for path in options.target_configuration_paths
    )
    config_by_checksum = {config.content_checksum: config for config in configs}
    if len(config_by_checksum) != len(configs):
        msg = "Target-population configurations must be checksum-distinct."
        raise TrainingRunnerConfigurationError(msg)
    _verify_confirm_executor_factory_source(options, source_manifest)
    fingerprints = tuple(
        sorted(
            (
                cast(
                    "ResumabilityFingerprint",
                    _decode_artifact(path, "resumability fingerprint", ResumabilityFingerprint.from_json),
                )
                for path in options.resumability_fingerprint_paths
            ),
            key=lambda item: (item.pipeline_prefix_id, item.content_checksum),
        )
    )
    try:
        validate_resumability_source_fingerprints(source_manifest, fingerprints)
    except (TypeError, ValueError):
        msg = "Resumability fingerprints differ from the verified execution-source manifest."
        raise TrainingRunnerConfigurationError(msg) from None
    try:
        entropy_files = parse_entropy_file_specs(options.external_entropy_file_specs)
    except (TypeError, ValueError):
        msg = "External entropy file references are invalid."
        raise TrainingRunnerConfigurationError(msg) from None
    expected_slots = {(config.data_role, config.population_scope) for config in configs}
    if set(entropy_files) != expected_slots:
        msg = "External entropy files must cover exactly the target configuration role/scope slots."
        raise TrainingRunnerConfigurationError(msg)
    try:
        keyring = ExternalEntropyKeyring.from_files(entropy_files)
        entropy_commitment_mismatch = any(
            keyring.commitment_for(config.data_role, config.population_scope) != config.role_master_entropy_commitment
            for config in configs
        )
    except (KeyError, OSError, TypeError, ValueError):
        msg = "External entropy or its target-population commitment is invalid."
        raise TrainingRunnerConfigurationError(msg) from None
    if entropy_commitment_mismatch:
        msg = "External entropy or its target-population commitment is invalid."
        raise TrainingRunnerConfigurationError(msg)

    # Target seed manifests remain unopened until the exact independent role
    # keys have reproduced every public target-configuration commitment.
    manifests = _load_targets(options)
    try:
        ordered_configs = tuple(config_by_checksum[manifest.population_config_checksum] for manifest in manifests)
    except KeyError:
        msg = "A target manifest has no exact supplied target-population configuration."
        raise TrainingRunnerConfigurationError(msg) from None
    try:
        authorized = tuple(
            AuthorizedTargetMaterialization(
                target_configuration=config,
                target_manifest=manifest,
                authorization=authorize_target_materialization(
                    preregistration,
                    config,
                    manifest,
                    keyring.entropy_for(config.data_role, config.population_scope),
                ),
            )
            for config, manifest in zip(ordered_configs, manifests, strict=True)
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        msg = "Target-materialization authorization is invalid."
        raise TrainingRunnerConfigurationError(msg) from None

    # Legacy candidate/pipeline inputs remain assertion-only.  They cannot
    # replace the execution profile or executable catalog.
    pipeline = None if options.pipeline_path is None else _load_pipeline(options.pipeline_path)
    asserted_candidates = _load_candidates(options)
    asserted_schedules = _load_schedules(options)
    _require_schedule_bindings(asserted_candidates, asserted_schedules)
    _validate_pipeline_candidate_binding(pipeline, asserted_candidates)
    _validate_pipeline_assertions(options, pipeline)
    profile_candidates = candidate_refs_from_bindings(catalog.bindings)
    if asserted_candidates and {item.content_checksum for item in asserted_candidates} != {
        item.candidate_checksum for item in profile_candidates
    }:
        msg = "Candidate assertions differ from the complete execution profile."
        raise TrainingRunnerConfigurationError(msg)
    profile_schedules = schedules_from_bindings(catalog.bindings)
    if asserted_schedules and {item.content_checksum for item in asserted_schedules} != {
        item.content_checksum for item in profile_schedules
    }:
        msg = "Schedule assertions differ from the complete execution profile."
        raise TrainingRunnerConfigurationError(msg)
    _validate_schedule_assertions(options, profile_schedules, pipeline_present=pipeline is not None)
    if options.method_id is not None and options.method_id not in {item.method_id for item in profile_candidates}:
        msg = "method_id is absent from the complete execution profile."
        raise TrainingRunnerConfigurationError(msg)

    screening: ScreeningManifest | None = None
    sample_size_design: SampleSizeDesign | None = None
    if options.preset == "training-smoke":
        unbound_plan = build_training_smoke_plan(
            preregistration_checksum=preregistration.content_checksum,
            target_manifest=manifests[0],
            executable_bindings=catalog.bindings,
        )
    elif options.preset == "paper-pilot":
        unbound_plan = build_paper_pilot_plan(
            preregistration_checksum=preregistration.content_checksum,
            target_manifests=manifests,
            optimization_seeds=pilot_seeds,
            executable_bindings=catalog.bindings,
        )
        # The plan normalizes the two custody populations; mirror that exact
        # order throughout the non-serializable context.
        manifest_by_checksum = {manifest.content_checksum: manifest for manifest in manifests}
        manifests = tuple(manifest_by_checksum[checksum] for checksum in unbound_plan.target_manifest_checksums)
        config_by_checksum = {config.content_checksum: config for config in ordered_configs}
        ordered_configs = tuple(config_by_checksum[manifest.population_config_checksum] for manifest in manifests)
        authorized_by_manifest = {item.target_manifest.content_checksum: item for item in authorized}
        authorized = tuple(authorized_by_manifest[manifest.content_checksum] for manifest in manifests)
    else:
        screen_path = _required_context_path(options.screening_manifest_path, "--screening-manifest")
        design_path = _required_context_path(options.sample_size_design_path, "--sample-size-design")
        screening = cast(
            "ScreeningManifest",
            _decode_artifact(screen_path, "screening manifest", ScreeningManifest.from_json),
        )
        sample_size_design = cast(
            "SampleSizeDesign",
            _decode_artifact(design_path, "sample-size design", SampleSizeDesign.from_json),
        )
        unbound_plan = build_paper_screen_plan(
            preregistration_checksum=preregistration.content_checksum,
            target_manifest=manifests[0],
            screening_manifest=screening,
            executable_bindings=catalog.bindings,
        )
    try:
        bound_plan = bind_training_plan_fingerprints(
            unbound_plan,
            execution_profile=profile,
            executable_bindings=catalog.bindings,
            target_configurations=ordered_configs,
            target_manifests=manifests,
            execution_source_manifest=source_manifest,
            resumability_fingerprints=fingerprints,
            required_sample_size_design=sample_size_design,
        )
        return TrainingExecutionContext(
            plan=bound_plan,
            execution_profile=profile,
            preregistration=preregistration,
            candidates=profile_candidates,
            schedules=profile_schedules,
            scoped_bindings=catalog.bindings,
            target_configurations=ordered_configs,
            target_manifests=manifests,
            authorized_materializations=authorized,
            screening_manifest=screening,
            screening_cells=() if screening is None else screening.cells,
            required_sample_size_design=sample_size_design,
            execution_source_manifest=source_manifest,
            resumability_fingerprints=fingerprints,
            external_entropy_keyring=keyring,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        msg = "WP22D execution context failed complete binding validation."
        raise TrainingRunnerConfigurationError(msg) from None


def build_training_execution_context(options: TrainingRunnerOptions) -> TrainingExecutionContext:
    """Build the complete non-serializable WP22D execution authority.

    Returns:
        The structurally verified context. Source bytes are rechecked by
        :meth:`TrainingExecutionContext.preflight` immediately before dispatch.

    Raises:
        TypeError: If ``options`` is not a :class:`TrainingRunnerOptions`.
    """
    if not isinstance(options, TrainingRunnerOptions):
        msg = "options must be TrainingRunnerOptions."
        raise TypeError(msg)
    return _load_training_execution_context(options)


def build_confirmation_execution_context(options: TrainingRunnerOptions) -> ConfirmationExecutionContext:
    """Build the complete source-locked authority for real confirmation.

    Returns:
        The narrow non-serializable confirmation execution context.

    Raises:
        TypeError: If ``options`` is not resolved runner options.
        TrainingRunnerConfigurationError: If the preset is not paper-confirm
            or any authorization input differs.
    """
    if not isinstance(options, TrainingRunnerOptions):
        msg = "options must be TrainingRunnerOptions."
        raise TypeError(msg)
    if options.preset != "paper-confirm":
        msg = "A ConfirmationExecutionContext requires the paper-confirm preset."
        raise TrainingRunnerConfigurationError(msg)
    _require_confirmation_reveal_opt_in(options)
    _preflight_paper_confirm_request(options)
    return _load_confirmation_execution_context(options, _load_preregistration(options))


def _load_prior_target_exposure_inventory(
    options: TrainingRunnerOptions,
) -> PriorTargetExposureInventory:
    """Load the required public novelty ledger for real confirmation.

    Returns:
        The strict checksum-sealed pilot, screening, Phase-I, and legacy
        exposure inventory.
    """
    path = _require_optional_artifact(
        options.prior_target_exposure_inventory_path,
        "--prior-target-exposure-inventory",
    )
    return cast(
        "PriorTargetExposureInventory",
        _decode_artifact(
            path,
            "prior-target exposure inventory",
            PriorTargetExposureInventory.from_json,
        ),
    )


def _load_expected_locked_study_head(
    options: TrainingRunnerOptions,
) -> LockedConfirmatoryStudySnapshotRef | None:
    """Load the externally retained exact head required for confirmation resume.

    The path may contain either the strict reference itself or prior CLI output
    with a ``locked_study_snapshot_reference`` member.

    Returns:
        The verified retained head, or ``None`` for a fresh invocation.

    Raises:
        TrainingRunnerConfigurationError: If the reference is absent on resume
            or its schema/checksum is invalid.
    """
    path = options.expected_locked_study_head_path
    if path is None:
        msg = "paper-confirm requires --expected-locked-study-head as external custody before dispatch."
        raise TrainingRunnerConfigurationError(msg)
    if not path.exists():
        return None
    if not options.resume:
        msg = "Fresh paper-confirm execution refuses to overwrite existing external head custody."
        raise TrainingRunnerConfigurationError(msg)
    try:
        document = load_canonical_json_object(_read_single_link_custody_payload(path).decode())
        raw_reference = document.get("locked_study_snapshot_reference", document)
        return LockedConfirmatoryStudySnapshotRef.from_dict(raw_reference)
    except (OSError, TypeError, ValueError):
        msg = "Externally retained locked-study head reference is invalid."
        raise TrainingRunnerConfigurationError(msg) from None


def _preflight_expected_locked_study_head_before_reveal(
    options: TrainingRunnerOptions,
    expected_head: LockedConfirmatoryStudySnapshotRef | None,
    *,
    final_seal: FinalConfirmationSeal,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
    execution_manifest: ExecutionSourceManifest,
    analysis_manifest: AnalysisSourceManifest,
    exposure_inventory: PriorTargetExposureInventory,
) -> None:
    """Authenticate external rollback custody before opening held inputs.

    Raises:
        TrainingRunnerConfigurationError: If fresh execution sees prior state,
            resume lacks its externally retained snapshot, or the retained
            snapshot bytes differ from the mutable output tree.
    """
    output_root = options.output_dir
    entries = tuple(output_root.iterdir()) if output_root.exists() else ()
    if not options.resume:
        if entries:
            msg = "Existing confirmation-owned output requires explicit resume and external head custody."
            raise TrainingRunnerConfigurationError(msg)
        if expected_head is not None:
            msg = "Fresh paper-confirm execution cannot reuse an existing external study head."
            raise TrainingRunnerConfigurationError(msg)
        return
    if expected_head is None:
        allowed = {".wp22-confirmation-session.json", "confirmation_study"}
        if not entries or any(path.name not in allowed for path in entries):
            msg = "paper-confirm resume lacks external head custody for non-initial output state."
            raise TrainingRunnerConfigurationError(msg)
        study_directory = output_root / "confirmation_study"
        if study_directory.exists():
            if study_directory.is_symlink() or not study_directory.is_dir():
                msg = "paper-confirm resume found an unsafe initial snapshot directory."
                raise TrainingRunnerConfigurationError(msg)
            snapshots = tuple(study_directory.iterdir())
            if snapshots:
                if len(snapshots) != 1:
                    msg = "paper-confirm resume lost external head custody for an established snapshot chain."
                    raise TrainingRunnerConfigurationError(msg)
                try:
                    snapshot_payload = _read_single_link_custody_payload(snapshots[0])
                    initial = LockedConfirmatoryStudySnapshot.from_json(snapshot_payload.decode())
                except (OSError, TypeError, UnicodeError, ValueError):
                    msg = "paper-confirm resume found invalid snapshot-zero custody."
                    raise TrainingRunnerConfigurationError(msg) from None
                _validate_snapshot_zero_publication_gap_before_reveal(
                    options,
                    snapshots[0],
                    snapshot_payload,
                    initial,
                    final_seal=final_seal,
                    configuration_execution_manifest=configuration_execution_manifest,
                    execution_manifest=execution_manifest,
                    analysis_manifest=analysis_manifest,
                    exposure_inventory=exposure_inventory,
                )
        return
    snapshot_path = output_root / expected_head.relative_path
    try:
        payload = _read_single_link_custody_payload(snapshot_path)
        file_checksum = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        snapshot = LockedConfirmatoryStudySnapshot.from_json(payload.decode())
    except (OSError, TypeError, UnicodeError, ValueError):
        msg = "Externally retained study head is absent or invalid in the confirmation output tree."
        raise TrainingRunnerConfigurationError(msg) from None
    if (
        file_checksum != expected_head.file_checksum
        or snapshot.ordinal != expected_head.ordinal
        or snapshot.content_checksum != expected_head.snapshot_content_checksum
    ):
        msg = "Externally retained study head differs from its exact on-disk snapshot."
        raise TrainingRunnerConfigurationError(msg)


def _validate_snapshot_zero_publication_gap_before_reveal(
    options: TrainingRunnerOptions,
    snapshot_path: Path,
    snapshot_payload: bytes,
    initial: LockedConfirmatoryStudySnapshot,
    *,
    final_seal: FinalConfirmationSeal,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
    execution_manifest: ExecutionSourceManifest,
    analysis_manifest: AnalysisSourceManifest,
    exposure_inventory: PriorTargetExposureInventory,
) -> None:
    """Bind the sole missing-external-head state to every public sealed root.

    This static recovery check reads only public session and snapshot custody.
    It deliberately neither opens the held target path nor decodes held entropy.

    Raises:
        TrainingRunnerConfigurationError: If snapshot zero is not the exact
            all-unattempted initial state committed by the public session.
    """
    output_root = options.output_dir
    marker_path = output_root / ".wp22-confirmation-session.json"
    try:
        marker = _read_confirmation_session_header(marker_path)
    except (OSError, TypeError, ValueError) as error:
        msg = "Snapshot-zero recovery cannot authenticate its exact public session marker."
        raise TrainingRunnerConfigurationError(msg) from error
    marker_payload = f"{canonical_json(marker)}\n".encode()
    expected_snapshot_path = (
        output_root / "confirmation_study" / f"snapshot_{0:08d}_{initial.content_checksum.removeprefix('sha256:')}.json"
    )
    custody_path = options.expected_locked_study_head_path
    manifest = initial.study_manifest
    expected_plan = build_paper_confirm_plan(
        seal=final_seal,
        target_manifest=manifest.target_manifest,
        configuration_execution_manifest=configuration_execution_manifest,
    )
    session_matches = (
        marker["content_checksum"] == initial.session_marker_content_checksum
        and marker["plan_checksum"] == expected_plan.content_checksum
        and marker["final_confirmation_seal_checksum"] == final_seal.content_checksum
        and marker["execution_source_manifest_checksum"] == execution_manifest.content_checksum
        and marker["analysis_source_manifest_checksum"] == analysis_manifest.content_checksum
        and marker["prior_target_exposure_inventory_checksum"] == exposure_inventory.content_checksum
        and marker["authorized_output_root"] == str(output_root)
        and custody_path is not None
        and marker["locked_study_head_custody_path"] == str(custody_path.resolve())
        and marker["job_count"] == len(expected_plan.jobs)
    )
    snapshot_matches = (
        snapshot_path == expected_snapshot_path
        and snapshot_payload == f"{initial.to_json()}\n".encode()
        and initial.ordinal == 0
        and initial.previous_snapshot is None
        and initial.authorized_output_root == str(output_root)
        and _snapshot_zero_has_exact_initial_inventory(initial, marker_payload)
    )
    public_roots_match = (
        manifest.plan == expected_plan
        and manifest.final_seal == final_seal
        and manifest.configuration_execution_manifest == configuration_execution_manifest
        and manifest.exposure_inventory == exposure_inventory
        and manifest.execution_source_manifest_checksum == execution_manifest.content_checksum
        and manifest.analysis_source_manifest_checksum == analysis_manifest.content_checksum
        and manifest.target_manifest.content_checksum == final_seal.confirmatory_target_manifest_checksum
    )
    all_unattempted = (
        manifest.status == "incomplete"
        and manifest.planned_job_count == len(expected_plan.jobs)
        and manifest.terminal_job_count == 0
        and manifest.successful_job_count == 0
        and manifest.failed_job_count == 0
        and manifest.unattempted_job_count == len(expected_plan.jobs)
        and manifest.observed_test_trajectory_count == 0
        and all(row.terminal_state == "unattempted" for row in manifest.rows)
    )
    root_names = {path.name for path in output_root.iterdir()}
    if not (
        root_names == {".wp22-confirmation-session.json", "confirmation_study"}
        and session_matches
        and snapshot_matches
        and public_roots_match
        and all_unattempted
    ):
        msg = (
            "Only the exact public-root-bound all-unattempted snapshot-zero publication gap may lack external custody."
        )
        raise TrainingRunnerConfigurationError(msg)


def _snapshot_zero_has_exact_initial_inventory(
    initial: LockedConfirmatoryStudySnapshot,
    marker_payload: bytes,
) -> bool:
    """Return whether snapshot zero receipts exactly the on-disk marker and study directory."""
    marker_checksum = f"sha256:{hashlib.sha256(marker_payload).hexdigest()}"
    actual = tuple(
        (
            entry.relative_path,
            entry.entry_kind,
            entry.byte_count,
            entry.file_checksum,
        )
        for entry in initial.output_entries
    )
    expected = (
        (".wp22-confirmation-session.json", "file", len(marker_payload), marker_checksum),
        ("confirmation_study", "directory", None, None),
    )
    return actual == expected


def _preflight_confirmation_session_header(
    options: TrainingRunnerOptions,
    final_seal: FinalConfirmationSeal,
    configuration_execution_manifest: FinalConfigurationExecutionManifest,
    execution_manifest: ExecutionSourceManifest,
    analysis_manifest: AnalysisSourceManifest,
    exposure_inventory: PriorTargetExposureInventory,
) -> None:
    """Reject foreign root state or a mismatched session before held reads.

    This static pass intentionally leaves the target-dependent plan checksum
    for the complete context preflight.  Every root that is already public is
    nevertheless checked before the held manifest or entropy file is opened.

    Raises:
        TrainingRunnerConfigurationError: If existing output has no valid
            source-, seal-, exposure-, and root-bound whole-plan marker.
    """
    output_root = options.output_dir
    if not output_root.exists():
        return
    allowed_names = {
        ".wp22-confirmation-session.json",
        "confirmation_study",
        "roles",
    }
    entries = tuple(output_root.iterdir())
    foreign = sorted(path.name for path in entries if path.name not in allowed_names)
    if foreign:
        msg = f"paper-confirm output contains foreign root members before reveal: {foreign!r}."
        raise TrainingRunnerConfigurationError(msg)
    if not entries:
        return
    marker = output_root / ".wp22-confirmation-session.json"
    if marker.is_symlink() or not marker.is_file():
        msg = "Existing confirmation-owned output lacks a regular whole-plan session marker."
        raise TrainingRunnerConfigurationError(msg)
    try:
        document = _read_confirmation_session_header(marker)
    except (OSError, TypeError, ValueError) as error:
        msg = "Existing confirmation whole-plan session marker is invalid before reveal."
        raise TrainingRunnerConfigurationError(msg) from error
    expected_job_count = (
        sum(cast("int", count) for count in final_seal.target_count_by_family.values())
        * final_seal.optimization_seed_count
        * len(configuration_execution_manifest.entries)
    )
    expected_custody_path = (
        None
        if options.expected_locked_study_head_path is None
        else str(options.expected_locked_study_head_path.resolve())
    )
    if (
        document["schema_version"] != "yaqs.state_preparation.phase2.confirmation_plan_session.v1"
        or type(document["plan_checksum"]) is not str
        or _CHECKSUM_PATTERN.fullmatch(document["plan_checksum"]) is None
        or document["final_confirmation_seal_checksum"] != final_seal.content_checksum
        or document["execution_source_manifest_checksum"] != execution_manifest.content_checksum
        or document["analysis_source_manifest_checksum"] != analysis_manifest.content_checksum
        or document["prior_target_exposure_inventory_checksum"] != exposure_inventory.content_checksum
        or document["authorized_output_root"] != str(output_root)
        or document["locked_study_head_custody_path"] != expected_custody_path
        or type(document["job_count"]) is not int
        or document["job_count"] != expected_job_count
    ):
        msg = "Existing confirmation whole-plan session header differs from the sealed public roots."
        raise TrainingRunnerConfigurationError(msg)


def _read_confirmation_session_header(marker: Path) -> dict[str, object]:
    """Read and checksum-verify a single-link session marker without following links.

    Returns:
        The strictly decoded and checksum-verified public session document.

    Raises:
        ValueError: If the marker changes, is linked, or is not canonical.

    """
    payload = _read_single_link_custody_payload(marker)
    document = dict(
        verify_sealed_mapping(
            load_canonical_json_object(payload.decode()),
            expected_keys=_CONFIRMATION_SESSION_HEADER_KEYS,
            name="confirmation whole-plan session marker",
        )
    )
    if payload != f"{canonical_json(document)}\n".encode():
        msg = "Confirmation session marker bytes are not canonical."
        raise ValueError(msg)
    return document


def _read_single_link_custody_payload(path: Path) -> bytes:
    """Open and read one public custody file without following links.

    Returns:
        Exact stable file bytes.

    Raises:
        ValueError: If the file is non-regular, linked, or changes while read.
    """
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        msg = "Public confirmation custody must be a single-link regular file."
        raise ValueError(msg)
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(path, flags)
    try:
        return _read_pinned_custody_payload(descriptor, metadata)
    finally:
        os.close(descriptor)


def _read_pinned_custody_payload(descriptor: int, expected: os.stat_result) -> bytes:
    """Read one custody descriptor after validating its immutable identity.

    Returns:
        Exact marker bytes.

    Raises:
        ValueError: If the descriptor is unsafe or changes while read.
    """
    opened = os.fstat(descriptor)
    identity = (opened.st_dev, opened.st_ino, opened.st_size)
    expected_identity = (expected.st_dev, expected.st_ino, expected.st_size)
    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or identity != expected_identity:
        msg = "Public confirmation custody changed during open."
        raise ValueError(msg)
    with os.fdopen(descriptor, "rb", closefd=False) as handle:
        payload = handle.read()
    closed = os.fstat(descriptor)
    if (closed.st_dev, closed.st_ino, closed.st_size, closed.st_nlink) != (*identity, 1):
        msg = "Public confirmation custody changed during read."
        raise ValueError(msg)
    return payload


def _load_confirmation_execution_context(
    options: TrainingRunnerOptions,
    preregistration: InitialPreregistration,
) -> ConfirmationExecutionContext:
    """Load real confirmation while holding its off-tree whole-run lock.

    Returns:
        The non-serializable source- and final-seal-bound execution context.
    """
    _require_confirmation_reveal_opt_in(options)
    _preflight_paper_confirm_request(options)
    lock_path = options.output_dir.parent / f".{options.output_dir.name}.wp22-confirmation-runner.lock"
    with FileLock(str(lock_path)):
        return _load_confirmation_execution_context_under_lock(options, preregistration)


def _load_confirmation_execution_context_under_lock(
    options: TrainingRunnerOptions,
    preregistration: InitialPreregistration,
) -> ConfirmationExecutionContext:
    """Authorize and close real confirmation before opening its target path.

    Returns:
        The non-serializable source- and final-seal-bound execution context.

    Raises:
        TrainingRunnerConfigurationError: If custody, source-lock, artifact, or
            assertion verification fails.
    """
    if any((
        options.execution_profile_path is not None,
        options.resumability_fingerprint_paths,
        options.pipeline_path is not None,
        options.candidate_paths,
        options.schedule_paths,
    )):
        msg = "paper-confirm cannot accept an execution profile, resumability state, or caller-selected code."
        raise TrainingRunnerConfigurationError(msg)
    _require_paths(options.target_manifest_paths, "paper-confirm", count=1)
    final_path = _require_optional_artifact(options.final_seal_path, "--final-seal")
    configuration_execution_path = _require_optional_artifact(
        options.configuration_execution_manifest_path,
        "--configuration-execution-manifest",
    )
    execution_path = _require_optional_artifact(
        options.execution_source_manifest_path,
        "--execution-source-manifest",
    )
    analysis_path = _require_optional_artifact(
        options.analysis_source_manifest_path,
        "--analysis-source-manifest",
    )
    final_seal = cast(
        "FinalConfirmationSeal",
        _decode_artifact(final_path, "final confirmation seal", FinalConfirmationSeal.from_json),
    )
    configuration_execution_manifest = cast(
        "FinalConfigurationExecutionManifest",
        _decode_artifact(
            configuration_execution_path,
            "final configuration execution manifest",
            FinalConfigurationExecutionManifest.from_json,
        ),
    )
    execution_manifest = cast(
        "ExecutionSourceManifest",
        _decode_artifact(execution_path, "execution-source manifest", ExecutionSourceManifest.from_json),
    )
    analysis_manifest = cast(
        "AnalysisSourceManifest",
        _decode_artifact(analysis_path, "analysis-source manifest", AnalysisSourceManifest.from_json),
    )
    if final_seal.preregistration_checksum != preregistration.content_checksum:
        msg = "Final seal belongs to a different preregistration."
        raise TrainingRunnerConfigurationError(msg)
    if options.target_manifest_checksums and (
        options.target_manifest_checksums != (final_seal.confirmatory_target_manifest_checksum,)
    ):
        msg = "target_manifest_checksum does not match the still-sealed confirmatory target commitment."
        raise TrainingRunnerConfigurationError(msg)
    try:
        verify_final_seal_source_lock(
            final_seal,
            execution_manifest,
            analysis_manifest,
            options.repository_root,
        )
    except (OSError, TypeError, ValueError) as error:
        msg = f"Final confirmation source custody is invalid: {error}."
        raise TrainingRunnerConfigurationError(msg) from error
    try:
        validate_final_configuration_execution_manifest(final_seal, configuration_execution_manifest)
    except (TypeError, ValueError) as error:
        msg = f"Final configuration execution custody is invalid: {error}."
        raise TrainingRunnerConfigurationError(msg) from error
    if options.executor_factory is not None:
        msg = "paper-confirm uses only the repository-owned default executor registry."
        raise TrainingRunnerConfigurationError(msg)
    screening_path = _require_optional_artifact(options.screening_manifest_path, "--screening-manifest")
    screening_evidence_path = _require_optional_artifact(
        options.screening_evidence_path,
        "--screening-evidence",
    )
    promotion_path = _require_optional_artifact(options.promotion_decision_path, "--promotion-decision")
    sample_size_path = _require_optional_artifact(options.sample_size_design_path, "--sample-size-design")
    calibration_path = _require_optional_artifact(
        options.resource_calibration_path,
        "--resource-calibration",
    )
    catalog_path = _require_optional_artifact(options.binding_catalog_path, "--binding-catalog")
    _require_paths(options.target_configuration_paths, "--target-configuration", count=1)
    screening_manifest = cast(
        "ScreeningManifest",
        _decode_artifact(screening_path, "screening manifest", ScreeningManifest.from_json),
    )
    screening_evidence = cast(
        "ScreeningEvidence",
        _decode_artifact(screening_evidence_path, "screening evidence", ScreeningEvidence.from_json),
    )
    promotion_decision = cast(
        "PromotionDecision",
        _decode_artifact(promotion_path, "promotion decision", PromotionDecision.from_json),
    )
    sample_size_design = cast(
        "SampleSizeDesign",
        _decode_artifact(sample_size_path, "sample-size design", SampleSizeDesign.from_json),
    )
    resource_calibration = cast(
        "ProductionResourceCalibration",
        _decode_artifact(
            calibration_path,
            "production resource calibration",
            ProductionResourceCalibration.from_json,
        ),
    )
    catalog = cast(
        "RepositoryBindingCatalog",
        _decode_artifact(catalog_path, "repository binding catalog", RepositoryBindingCatalog.from_json),
    )
    exposure_inventory = _load_prior_target_exposure_inventory(options)
    if (
        exposure_inventory.preregistration_checksum != preregistration.content_checksum
        or exposure_inventory.screening_manifest != screening_manifest
        or exposure_inventory.screening_target_manifest.content_checksum
        != screening_manifest.screening_target_manifest_checksum
        or exposure_inventory.resource_calibration_checksum != resource_calibration.content_checksum
        or exposure_inventory.resource_calibration_execution_source_checksum
        != resource_calibration.execution_source_manifest_checksum
        or exposure_inventory.resource_calibration_execution_source_checksum != execution_manifest.content_checksum
        or exposure_inventory.pilot_plan.content_checksum != resource_calibration.pilot_plan_checksum
        or exposure_inventory.screening_plan.content_checksum != resource_calibration.screening_plan_checksum
        or exposure_inventory.pilot_custody_checksum != resource_calibration.pilot_custody_checksum
        or exposure_inventory.pilot_calibration_checksum != resource_calibration.pilot_calibration_checksum
        or exposure_inventory.screening_custody_checksum != resource_calibration.screening_custody_checksum
    ):
        msg = "Prior-target exposure inventory differs from the authorized pilot and screening custody."
        raise TrainingRunnerConfigurationError(msg)
    _preflight_confirmation_session_header(
        options,
        final_seal,
        configuration_execution_manifest,
        execution_manifest,
        analysis_manifest,
        exposure_inventory,
    )
    expected_locked_study_head = _load_expected_locked_study_head(options)
    _preflight_expected_locked_study_head_before_reveal(
        options,
        expected_locked_study_head,
        final_seal=final_seal,
        configuration_execution_manifest=configuration_execution_manifest,
        execution_manifest=execution_manifest,
        analysis_manifest=analysis_manifest,
        exposure_inventory=exposure_inventory,
    )
    try:
        confirmation_authorization = authorize_confirmation(
            preregistration,
            screening_manifest,
            screening_evidence,
            promotion_decision,
            sample_size_design,
            analysis_manifest,
            final_seal,
            configuration_execution_manifest,
            resource_calibration,
            options.repository_root,
        )
    except (OSError, TypeError, ValueError) as error:
        msg = f"Final confirmation authorization is invalid: {error}."
        raise TrainingRunnerConfigurationError(msg) from error

    if (
        catalog.profile.preset != "paper-screen"
        or catalog.profile.preregistration_checksum != preregistration.content_checksum
    ):
        msg = "paper-confirm requires the exact preregistration-bound paper-screen binding catalog."
        raise TrainingRunnerConfigurationError(msg)
    for execution in configuration_execution_manifest.entries:
        matches = tuple(
            link
            for link in catalog.bindings
            if link.binding.publication_candidate_checksum == execution.configuration_checksum
        )
        try:
            alias = catalog.implementation_catalog.resolve("paper-confirm", execution.method_id, "primary_q6")
        except (KeyError, TypeError, ValueError):
            msg = "A final configuration lacks its dormant repository confirmation alias."
            raise TrainingRunnerConfigurationError(msg) from None
        if len(matches) != 1:
            msg = "A final configuration lacks one exact paper-screen executable binding."
            raise TrainingRunnerConfigurationError(msg)
        link = matches[0]
        if (
            link.binding.publication_method_id != execution.method_id
            or link.binding.strategy_schedule != execution.strategy_schedule
            or link.binding.implementation_checksum != execution.implementation_checksum
            or link.binding.content_checksum != execution.scoped_binding_checksum
            or link.content_checksum != execution.executable_binding_checksum
            or link.implementation_entry != alias
        ):
            msg = "A final configuration differs from its exact dormant repository confirmation alias."
            raise TrainingRunnerConfigurationError(msg)

    target_configuration = cast(
        "TargetPopulationConfig",
        _decode_artifact(
            options.target_configuration_paths[0],
            "target-population configuration",
            TargetPopulationConfig.from_json,
        ),
    )
    try:
        entropy_files = parse_entropy_file_specs(options.external_entropy_file_specs)
    except (TypeError, ValueError):
        msg = "Confirmatory external entropy file references are invalid."
        raise TrainingRunnerConfigurationError(msg) from None
    if set(entropy_files) != {("confirmatory", "primary_q6")}:
        msg = "paper-confirm requires exactly the confirmatory/primary_q6 external entropy slot."
        raise TrainingRunnerConfigurationError(msg)
    if (
        target_configuration.data_role != "confirmatory"
        or target_configuration.population_scope != "primary_q6"
        or target_configuration.preregistration_checksum != preregistration.content_checksum
    ):
        msg = "Confirmatory target configuration is invalid."
        raise TrainingRunnerConfigurationError(msg)

    # This is deliberately the first operation that opens the confirmatory target path.
    targets = _load_targets(options)
    target_manifest = targets[0]
    if target_manifest.content_checksum != final_seal.confirmatory_target_manifest_checksum:
        msg = "Revealed confirmatory target manifest differs from the final seal."
        raise TrainingRunnerConfigurationError(msg)
    try:
        materialization_authorization, keyring = _authorize_revealed_confirmatory_target(
            preregistration=preregistration,
            target_configuration=target_configuration,
            target_manifest=target_manifest,
            entropy_files=entropy_files,
            confirmation_authorization=confirmation_authorization,
            exposure_inventory=exposure_inventory,
        )
    except (OSError, TypeError, ValueError):
        msg = "Revealed confirmatory target novelty or materialization authorization is invalid."
        raise TrainingRunnerConfigurationError(msg) from None
    _validate_method_assertion(options, None, (), final_seal)
    _validate_resource_assertions(options, preregistration, final_seal)
    plan = build_paper_confirm_plan(
        seal=final_seal,
        target_manifest=target_manifest,
        configuration_execution_manifest=configuration_execution_manifest,
    )
    try:
        return ConfirmationExecutionContext(
            plan=plan,
            preregistration=preregistration,
            final_seal=final_seal,
            configuration_execution_manifest=configuration_execution_manifest,
            execution_source_manifest=execution_manifest,
            analysis_source_manifest=analysis_manifest,
            repository_binding_catalog=catalog,
            target_configuration=target_configuration,
            target_manifest=target_manifest,
            authorized_output_root=options.output_dir,
            locked_study_head_custody_path=cast(
                "Path",
                options.expected_locked_study_head_path,
            ).resolve(),
            prior_target_exposure_inventory_checksum=exposure_inventory.content_checksum,
            confirmation_authorization=confirmation_authorization,
            target_materialization_authorization=materialization_authorization,
            external_entropy_keyring=keyring,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        msg = "Real confirmation context failed complete final-seal binding validation."
        raise TrainingRunnerConfigurationError(msg) from None


def _authorize_revealed_confirmatory_target(
    *,
    preregistration: InitialPreregistration,
    target_configuration: TargetPopulationConfig,
    target_manifest: TargetPopulationManifest,
    entropy_files: Mapping[tuple[str, str], Path],
    confirmation_authorization: ConfirmationAuthorization,
    exposure_inventory: PriorTargetExposureInventory,
) -> tuple[TargetMaterializationAuthorization, ExternalEntropyKeyring]:
    """Verify novelty and held entropy, then authorize target materialization.

    Returns:
        The exact sealed target-materialization authorization and held keyring.

    """
    exposure_inventory.validate_confirmatory_novelty(target_manifest)
    keyring = ExternalEntropyKeyring.from_files(entropy_files)
    _validate_confirmatory_entropy_commitment(keyring, target_configuration)
    return (
        authorize_target_materialization(
            preregistration,
            target_configuration,
            target_manifest,
            keyring.entropy_for("confirmatory", "primary_q6"),
            confirmation_authorization,
        ),
        keyring,
    )


def _validate_confirmatory_entropy_commitment(
    keyring: ExternalEntropyKeyring,
    target_configuration: TargetPopulationConfig,
) -> None:
    """Require revealed entropy to match the confirmatory population seal.

    Raises:
        ValueError: If the entropy commitment differs from the target configuration.
    """
    if keyring.commitment_for("confirmatory", "primary_q6") != target_configuration.role_master_entropy_commitment:
        msg = "Confirmatory external entropy differs from its sealed commitment."
        raise ValueError(msg)


def build_training_plan(options: TrainingRunnerOptions) -> TrainingRunPlan:
    """Load immutable artifacts and build one deterministic WP22 plan.

    Returns:
        The checksum-sealed plan in canonical job order.

    Raises:
        TypeError: If ``options`` is not a :class:`TrainingRunnerOptions`.
        TrainingRunnerConfigurationError: If the preset, artifacts, or controls
            are invalid.
    """
    if not isinstance(options, TrainingRunnerOptions):
        msg = "options must be TrainingRunnerOptions."
        raise TypeError(msg)
    if options.preset == "paper-confirm":
        _require_confirmation_reveal_opt_in(options)
        _preflight_paper_confirm_request(options)
    preregistration = _load_preregistration(options)
    if options.preset == "paper-confirm":
        if any((
            options.execution_profile_path is not None,
            options.resumability_fingerprint_paths,
            options.pipeline_path is not None,
            options.candidate_paths,
            options.schedule_paths,
        )):
            msg = "paper-confirm cannot accept an execution profile, resumability state, or caller-selected code."
            raise TrainingRunnerConfigurationError(msg)
        return _load_confirmation_execution_context(options, preregistration).plan
    _validate_resource_assertions(options, preregistration, None)
    if options.preset == "historical-layerwise-reproduction":
        scientific_artifacts = any((
            options.pipeline_path is not None,
            options.candidate_paths,
            options.schedule_paths,
            options.target_manifest_paths,
            options.screening_manifest_path is not None,
            options.screening_evidence_path is not None,
            options.promotion_decision_path is not None,
            options.execution_profile_path is not None,
            options.binding_catalog_path is not None,
            options.target_configuration_paths,
            options.sample_size_design_path is not None,
            options.resource_calibration_path is not None,
            options.resumability_fingerprint_paths,
            options.external_entropy_file_specs,
            options.execution_source_manifest_path is not None,
            options.analysis_source_manifest_path is not None,
            options.final_seal_path is not None,
            options.configuration_execution_manifest_path is not None,
        ))
        if scientific_artifacts:
            msg = "Historical reproduction reads only the exact frozen WP19 pipeline and targets."
            raise TrainingRunnerConfigurationError(msg)
        if options.fail_fast:
            msg = "fail_fast is not part of the exact WP19 historical delegate."
            raise TrainingRunnerConfigurationError(msg)
        return build_historical_reproduction_plan(preregistration_checksum=preregistration.content_checksum)
    if (
        options.final_seal_path is not None
        or options.configuration_execution_manifest_path is not None
        or options.analysis_source_manifest_path is not None
        or options.screening_evidence_path is not None
        or options.promotion_decision_path is not None
        or options.resource_calibration_path is not None
    ):
        msg = "Final authorization artifacts require paper-confirm."
        raise TrainingRunnerConfigurationError(msg)
    if options.preset in {"training-smoke", "paper-pilot", "paper-screen"}:
        return build_training_execution_context(options).plan
    msg = f"Unsupported training preset {options.preset!r}."
    raise TrainingRunnerConfigurationError(msg)


# Compatibility spelling matching the lower-level orchestration factory.
build_run_plan = build_training_plan


def _preflight_context_plan(
    options: TrainingRunnerOptions,
    context: TrainingExecutionContext | ConfirmationExecutionContext,
    prior_target_exposure_inventory: PriorTargetExposureInventory | None = None,
    expected_locked_study_head: LockedConfirmatoryStudySnapshotRef | None = None,
) -> None:
    """Validate the complete context and output universe without dispatch."""

    def unreachable_executor(*_arguments: object) -> str:
        msg = "Dry preflight attempted to dispatch an executor."
        raise RuntimeError(msg)

    preflight_executor: TrainingJobExecutor | TrainingExecutorRegistry = unreachable_executor
    if isinstance(context, ConfirmationExecutionContext):
        preflight_executor = TrainingExecutorRegistry(confirm_executor=unreachable_executor)
    execute_training_plan(
        context.plan,
        options.output_dir,
        preflight_executor,
        resume=options.resume,
        overwrite=options.overwrite,
        dry_run=True,
        fail_fast=options.fail_fast,
        context=context,
        repository_root=options.repository_root,
        prior_target_exposure_inventory=prior_target_exposure_inventory,
        expected_locked_study_head=expected_locked_study_head,
    )


def run(
    options: TrainingRunnerOptions,
    *,
    context: TrainingExecutionContext | ConfirmationExecutionContext | None = None,
    executor: TrainingJobExecutor | TrainingExecutorRegistry | None = None,
    historical_runner: Callable[..., object] = run_historical_reproduction_job,
) -> TrainingRunPlan | TrainingRunSummary | object:
    """Dry-run, dispatch, or explicitly delegate one validated plan.

    Returns:
        The dry-run plan, generic execution summary, or historical report.

    Raises:
        TypeError: If a programmatically supplied context or runner is invalid.
        TrainingRunnerConfigurationError: If execution authority, artifacts,
            controls, or executor selection is invalid.
    """
    if context is not None and not isinstance(context, (TrainingExecutionContext, ConfirmationExecutionContext)):
        msg = "context must be a TrainingExecutionContext or ConfirmationExecutionContext."
        raise TypeError(msg)
    if options.preset == "paper-confirm" and (context is not None or executor is not None):
        msg = "paper-confirm forbids programmatic context or executor injection; use the sealed CLI artifacts."
        raise TrainingRunnerConfigurationError(msg)
    if options.preset == "paper-confirm":
        _require_confirmation_reveal_opt_in(options)
        _preflight_paper_confirm_request(options)
    if executor is not None and options.executor_factory is not None:
        msg = "Select either the configured executor_factory or a programmatically injected executor, not both."
        raise TrainingRunnerConfigurationError(msg)
    selected_context = context
    exposure_inventory: PriorTargetExposureInventory | None = None
    expected_locked_study_head: LockedConfirmatoryStudySnapshotRef | None = None
    if selected_context is None and options.preset in {"training-smoke", "paper-pilot", "paper-screen"}:
        selected_context = build_training_execution_context(options)
    elif selected_context is None and options.preset == "paper-confirm":
        selected_context = _load_confirmation_execution_context(options, _load_preregistration(options))
    if selected_context is None:
        plan = build_training_plan(options)
    else:
        plan = selected_context.plan
        if plan.preset != options.preset or plan.preregistration_checksum != options.preregistration_checksum:
            msg = "Programmatic execution context differs from the resolved CLI preset or preregistration assertion."
            raise TrainingRunnerConfigurationError(msg)
    if isinstance(selected_context, ConfirmationExecutionContext):
        expected_locked_study_head = _load_expected_locked_study_head(options)
        exposure_inventory = _load_prior_target_exposure_inventory(options)
        if exposure_inventory.content_checksum != selected_context.prior_target_exposure_inventory_checksum:
            msg = "Prior-target exposure inventory differs from the authorized confirmation context."
            raise TrainingRunnerConfigurationError(msg)
    if options.dry_run:
        if selected_context is not None:
            _preflight_context_plan(
                options,
                selected_context,
                exposure_inventory,
                expected_locked_study_head,
            )
        return plan
    if options.preset == "historical-layerwise-reproduction":
        if not options.execute_expensive:
            msg = "Historical execution requires the additional --execute-expensive opt-in."
            raise TrainingRunnerConfigurationError(msg)
        if not callable(historical_runner):
            msg = "historical_runner must be callable."
            raise TypeError(msg)
        return historical_runner(
            options.output_dir,
            execute_expensive=True,
            resume=options.resume,
            overwrite=options.overwrite,
            repository_root=options.repository_root,
        )
    if selected_context is not None:
        # No extension or repository-owned executor code receives the secret-bearing
        # context until every source byte and scientific fingerprint has passed.
        _preflight_context_plan(
            options,
            selected_context,
            exposure_inventory,
            expected_locked_study_head,
        )
    selected_executor = executor
    if selected_executor is None and options.executor_factory is not None:
        if selected_context is None:
            msg = "An executor factory requires a complete execution context."
            raise TrainingRunnerConfigurationError(msg)
        selected_executor = load_executor_registry(
            options.executor_factory,
            options.repository_root,
            selected_context,
        )
    if selected_executor is None:
        if selected_context is None:
            msg = "Repository execution requires a complete source-locked execution context."
            raise TrainingRunnerConfigurationError(msg)
        selected_executor = load_executor_registry(
            DEFAULT_EXECUTOR_FACTORY,
            options.repository_root,
            selected_context,
        )
    return execute_training_plan(
        plan,
        options.output_dir,
        selected_executor,
        resume=options.resume,
        overwrite=options.overwrite,
        dry_run=False,
        fail_fast=options.fail_fast,
        context=selected_context,
        repository_root=options.repository_root if selected_context is not None else None,
        prior_target_exposure_inventory=exposure_inventory,
        expected_locked_study_head=expected_locked_study_head,
    )


def _render_result(result: object) -> str:
    """Render one CLI result without guessing scientific content.

    Returns:
        Canonical JSON where available, otherwise the object's string form.

    Raises:
        ValueError: If a locked-study summary contains an inconsistent reference.
    """
    if isinstance(result, TrainingRunPlan):
        return result.to_json()
    to_json = getattr(result, "to_json", None)
    if callable(to_json):
        rendered = to_json()
        if isinstance(rendered, str):
            return rendered
    if isinstance(result, TrainingRunSummary):
        summary: dict[str, object] = {
            "planned": result.planned,
            "attempted": result.attempted,
            "succeeded": result.succeeded,
            "failed": result.failed,
            "skipped": result.skipped,
        }
        if result.locked_study_snapshot_path is not None:
            reference = LockedConfirmatoryStudySnapshotRef(
                relative_path=result.locked_study_snapshot_path,
                ordinal=cast("int", result.locked_study_snapshot_ordinal),
                file_checksum=cast("str", result.locked_study_snapshot_file_checksum),
                snapshot_content_checksum=cast("str", result.locked_study_snapshot_content_checksum),
            )
            if reference.content_checksum != result.locked_study_snapshot_reference_checksum:
                msg = "Training summary contains an inconsistent locked-study head reference."
                raise ValueError(msg)
            summary["locked_study_snapshot_reference"] = reference.to_dict()
            summary["external_study_head_custody_required"] = True
        return json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
        )
    return str(result)


def main(
    arguments: Sequence[str] | None = None,
    *,
    stdout: IO[str] = sys.stdout,
    stderr: IO[str] = sys.stderr,
    context: TrainingExecutionContext | ConfirmationExecutionContext | None = None,
    executor: TrainingJobExecutor | TrainingExecutorRegistry | None = None,
) -> int:
    """Run the opt-in WP22 CLI and return a process exit status.

    Returns:
        Zero on success and two for rejected configuration or authorization.
    """
    try:
        options = resolve_options(parse_arguments(arguments))
        result = run(options, context=context, executor=executor)
    except (OSError, TypeError, ValueError) as error:
        print(f"training runner error: {error}", file=stderr)
        return 2
    print(_render_result(result), file=stdout)
    return 0


__all__ = [
    "DEFAULT_PILOT_OPTIMIZATION_SEEDS",
    "DEFAULT_TRAINING_OUTPUT_ROOT",
    "TRAINING_PRESETS",
    "TRAINING_RUNNER_CONFIGURATION_FORMAT",
    "RunnerConfigurationError",
    "TrainingRunnerConfigurationError",
    "TrainingRunnerOptions",
    "build_confirmation_execution_context",
    "build_run_plan",
    "build_training_execution_context",
    "build_training_plan",
    "create_argument_parser",
    "load_configuration_file",
    "load_executor_registry",
    "main",
    "parse_arguments",
    "resolve_options",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
