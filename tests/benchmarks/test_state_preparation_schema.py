# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the state-preparation benchmark configuration and result schemas."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import traceback as traceback_module
from dataclasses import FrozenInstanceError, replace
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, cast

import pytest

from benchmarks.state_preparation.schema import (
    CONFIDENCE_INTERVAL_METHODS,
    CONFIG_SCHEMA_VERSION,
    CSV_COLUMNS,
    NOISE_DEFINITION_VERSION,
    NOISE_IDS,
    RESULT_SCHEMA_VERSION,
    STANDARD_NOISE_IDS,
    SUPPORTED_QUBIT_COUNTS,
    TARGET_FIXTURE_FORMAT,
    TARGET_GENERATION_SEEDS,
    TARGET_IDS,
    AnsatzConfig,
    BenchmarkConfig,
    BenchmarkFailure,
    BenchmarkResult,
    CircuitStatistics,
    EvaluationConfig,
    InitializationConfig,
    NoiseConfig,
    OptimizerConfig,
    TargetSelection,
    benchmark_record_from_csv_row,
    benchmark_record_from_dict,
    benchmark_record_from_json,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

SHA_A = f"sha256:{'a' * 64}"
SHA_B = f"sha256:{'b' * 64}"
GIT_COMMIT = "c" * 40

EXPECTED_TARGET_IDS = {
    "gaussian_mu0p5_sigma0p1",
    "tfim_ferro",
    "tfim_critical",
    "tfim_para",
    "haar_random_1",
    "haar_random_2",
    "haar_random_3",
    "random_mps_bond2",
    "random_mps_bond3",
}
EXPECTED_STANDARD_NOISE_IDS = {
    "dephasing_1s_1q",
    "dephasing_1s_2q",
    "dephasing_1s_all",
    "dephasing_2s_2q",
    "dephasing_1s2s_all",
    "depolarizing_1s_1q",
    "depolarizing_1s_2q",
    "depolarizing_1s_all",
    "depolarizing_2s_2q",
    "depolarizing_1s2s_all",
}


def _target(**changes: object) -> TargetSelection:
    target_id = changes.get("target_id", "gaussian_mu0p5_sigma0p1")
    values: dict[str, object] = {
        "num_qubits": 6,
        "target_id": target_id,
        "target_seed": TARGET_GENERATION_SEEDS.get(cast("str", target_id)),
        "fixture_format": TARGET_FIXTURE_FORMAT,
        "fixture_checksum": SHA_A,
    }
    values.update(changes)
    return TargetSelection(**cast("Any", values))


def _ansatz(**changes: object) -> AnsatzConfig:
    values: dict[str, object] = {
        "configured_bmpd_depth": 2,
        "initial_single_qubit_layer": True,
        "ansatz_id": "bmpd_brickwall",
    }
    values.update(changes)
    return AnsatzConfig(**cast("Any", values))


def _initialization(**changes: object) -> InitializationConfig:
    values: dict[str, object] = {
        "rule": "random_normal",
        "seed": 11,
        "scale": 0.1,
        "warm_start_path": None,
        "warm_start_checksum": None,
    }
    values.update(changes)
    return InitializationConfig(**cast("Any", values))


def _optimizer(**changes: object) -> OptimizerConfig:
    values: dict[str, object] = {
        "optimizer_id": "krotov",
        "max_iterations": 4,
        "optimizer_seed": 17,
        "hyperparameters": {"step_size": 0.2, "schedule": {"kind": "constant"}},
        "train_trajectories_or_shots": 0,
        "training_seed": None,
        "max_bond_dimension": None,
        "svd_threshold": 0.0,
        "truncation_mode": "discarded_weight",
        "min_bond_dimension": 1,
    }
    values.update(changes)
    return OptimizerConfig(**cast("Any", values))


def _evaluation(*, noisy: bool = True, **changes: object) -> EvaluationConfig:
    confidence_level = changes.get("confidence_level")
    values: dict[str, object] = {
        "test_trajectories_or_shots": 32 if noisy else 0,
        "test_seed": 23 if noisy else None,
        "max_bond_dimension": None,
        "svd_threshold": 0.0,
        "truncation_mode": "discarded_weight",
        "min_bond_dimension": 1,
        "store_trajectory_sidecar": False,
        "confidence_level": confidence_level,
        "confidence_interval_method": "normal_clipped" if confidence_level is not None else None,
    }
    values.update(changes)
    return EvaluationConfig(**cast("Any", values))


def _noise(noise_id: str = "dephasing_1s_all", **changes: object) -> NoiseConfig:
    values: dict[str, object] = {
        "noise_id": noise_id,
        "tjm_dt": 1.0 if noise_id in STANDARD_NOISE_IDS else None,
        "definition_version": NOISE_DEFINITION_VERSION,
    }
    values.update(changes)
    return NoiseConfig(**cast("Any", values))


def _config(*, noisy: bool = True, **changes: object) -> BenchmarkConfig:
    values: dict[str, object] = {
        "method_id": "krotov",
        "method_version": "1",
        "target": _target(),
        "ansatz": _ansatz(),
        "initialization": _initialization(),
        "optimizer": _optimizer(),
        "evaluation": _evaluation(noisy=noisy),
        "training_noise": _noise("noiseless"),
        "test_noise": _noise("dephasing_1s_all") if noisy else _noise("noiseless"),
        "schema_version": CONFIG_SCHEMA_VERSION,
    }
    values.update(changes)
    return BenchmarkConfig(**cast("Any", values))


def _statistics(**changes: object) -> CircuitStatistics:
    values: dict[str, object] = {
        "configured_bmpd_depth": 2,
        "num_parameters": 108,
        "logical_depth": 10,
        "logical_num_1q_gates": 12,
        "logical_num_2q_gates": 8,
        "native_depth": 14,
        "native_num_1q_gates": 20,
        "native_num_2q_gates": 8,
        "native_rzz_count": 8,
        "pruned_native_rzz_count": 0,
        "evaluated_representation": "logical",
        "logical_gate_counts": {"rx": 6, "ry": 6, "rxx": 4, "ryy": 4},
        "native_gate_counts": {"rx": 10, "ry": 10, "rzz": 8},
    }
    values.update(changes)
    return CircuitStatistics(**cast("Any", values))


def _software_versions() -> dict[str, str]:
    return {
        "yaqs": "0.0.dev0",
        "numpy": "2.3.0",
        "python": "3.13.5",
        "scipy": "1.16.0",
    }


def _normal_interval(mean: float, standard_error: float, level: float = 0.95) -> tuple[float, float]:
    z_score = NormalDist().inv_cdf((1.0 + level) / 2.0)
    return max(0.0, mean - z_score * standard_error), min(1.0, mean + z_score * standard_error)


def _result(*, noisy: bool = True, **changes: object) -> BenchmarkResult:
    config = cast("BenchmarkConfig", changes.get("config", _config(noisy=noisy)))
    is_noisy = config.test_noise.noise_id != "noiseless"
    trajectory_count = config.evaluation.test_trajectories_or_shots
    standard_deviation = 0.04 if is_noisy and trajectory_count >= 2 else None
    standard_error = standard_deviation / math.sqrt(trajectory_count) if standard_deviation is not None else None
    evaluated_representation = "native" if config.test_noise.noise_id == "ballarin_coupled" else "logical"
    num_bmpd_blocks = config.ansatz.configured_bmpd_depth * (config.target.num_qubits - 1)
    initial_parameters = 3 * config.target.num_qubits if config.ansatz.initial_single_qubit_layer else 0
    num_parameters = 9 * num_bmpd_blocks + initial_parameters
    values: dict[str, object] = {
        "config": config,
        "circuit_statistics": _statistics(
            configured_bmpd_depth=config.ansatz.configured_bmpd_depth,
            num_parameters=num_parameters,
            evaluated_representation=evaluated_representation,
        ),
        "train_fidelity": 0.81,
        "logical_test_noiseless_fidelity": 0.80,
        "native_pre_pruning_noiseless_fidelity": 0.795 if evaluated_representation == "native" else None,
        "test_noiseless_fidelity": 0.80,
        "test_noisy_fidelity": 0.74 if is_noisy else 0.80,
        "noisy_fidelity_standard_deviation": standard_deviation,
        "noisy_fidelity_standard_error": standard_error,
        "confidence_interval_lower": None,
        "confidence_interval_upper": None,
        "sampled_nonidentity_events": 7 if is_noisy else 0,
        "optimization_wall_time_seconds": 1.25,
        "evaluation_wall_time_seconds": 0.75,
        "software_versions": _software_versions(),
        "git_commit": GIT_COMMIT,
        "git_dirty": False,
        "git_diff_checksum": None,
        "parameter_checkpoint_path": "parameters.npz",
        "parameter_checkpoint_checksum": SHA_A,
        "trajectory_sidecar_path": None,
        "trajectory_sidecar_checksum": None,
        "notes": "",
        "schema_version": RESULT_SCHEMA_VERSION,
    }
    values.update(changes)
    return BenchmarkResult(**cast("Any", values))


def _failure(**changes: object) -> BenchmarkFailure:
    values: dict[str, object] = {
        "config": _config(),
        "failure_phase": "optimization",
        "exception_type": "RuntimeError",
        "message": "optimizer failed",
        "traceback": "Traceback (most recent call last):\nRuntimeError: optimizer failed",
        "retryable": False,
        "attempt": 1,
        "wall_time_seconds": 0.5,
        "software_versions": _software_versions(),
        "git_commit": GIT_COMMIT,
        "git_dirty": False,
        "git_diff_checksum": None,
        "parameter_checkpoint_path": None,
        "parameter_checkpoint_checksum": None,
        "notes": "",
        "schema_version": RESULT_SCHEMA_VERSION,
    }
    values.update(changes)
    return BenchmarkFailure(**cast("Any", values))


def _assert_rejected(factory: Callable[..., object], field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        factory(**{field: value})


def _raise_runtime_error(message: str) -> None:
    """Raise a runtime error for exception-capture tests.

    Raises:
        RuntimeError: Unconditionally, with the supplied message.
    """
    raise RuntimeError(message)


def test_schema_constants_cover_the_frozen_benchmark_definition() -> None:
    """Schema registries should exactly cover the benchmark-defined identifiers."""
    assert tuple(SUPPORTED_QUBIT_COUNTS) == (6, 12)
    assert set(TARGET_IDS) == EXPECTED_TARGET_IDS
    assert set(STANDARD_NOISE_IDS) == EXPECTED_STANDARD_NOISE_IDS
    assert set(NOISE_IDS) == EXPECTED_STANDARD_NOISE_IDS | {"noiseless", "ballarin_coupled"}
    assert CONFIG_SCHEMA_VERSION
    assert RESULT_SCHEMA_VERSION
    assert NOISE_DEFINITION_VERSION
    assert CONFIDENCE_INTERVAL_METHODS == ("normal_clipped",)
    assert len(CSV_COLUMNS) == len(set(CSV_COLUMNS))
    assert {"schema_version", "status", "run_id"}.issubset(CSV_COLUMNS)


def test_valid_minimal_config_has_stable_derived_values() -> None:
    """A valid minimal config should expose its derived layer count and run identifier."""
    config = _config()
    assert config.ansatz.num_layers == 2 * config.ansatz.configured_bmpd_depth
    assert re.fullmatch(r"spr-v1-[0-9a-f]{64}", config.run_id)
    assert config.identity_payload() == BenchmarkConfig.from_dict(config.to_dict()).identity_payload()


@pytest.mark.parametrize("num_qubits", SUPPORTED_QUBIT_COUNTS)
@pytest.mark.parametrize("target_id", TARGET_IDS)
def test_all_target_and_qubit_identifiers_are_accepted(num_qubits: int, target_id: str) -> None:
    """Every frozen target identifier should be valid for both benchmark sizes."""
    target = _target(num_qubits=num_qubits, target_id=target_id)
    assert target.num_qubits == num_qubits
    assert target.target_id == target_id
    assert target.target_seed == TARGET_GENERATION_SEEDS[target_id]
    assert target.fixture_format == TARGET_FIXTURE_FORMAT


@pytest.mark.parametrize("noise_id", NOISE_IDS)
def test_all_noise_identifiers_are_accepted_with_their_required_tjm_step(noise_id: str) -> None:
    """Every noise identifier should accept exactly its documented TJM-step form."""
    noise = _noise(noise_id)
    assert noise.noise_id == noise_id
    assert noise.tjm_dt == (1.0 if noise_id in STANDARD_NOISE_IDS else None)


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (_target, "num_qubits", 4),
        (_target, "num_qubits", True),
        (_target, "target_id", "TFIM_FERRO"),
        (_target, "target_id", " tfim_ferro"),
        (_target, "target_id", "../tfim_ferro"),
        (_target, "target_seed", -1),
        (_target, "target_seed", True),
        (_target, "target_seed", 2**64),
        (_target, "fixture_format", ""),
        (_target, "fixture_format", "state-preparation-targets-v1"),
        (_target, "fixture_checksum", "a" * 64),
        (_target, "fixture_checksum", f"sha256:{'A' * 64}"),
        (_target, "fixture_checksum", f"sha256:{'a' * 63}"),
        (_ansatz, "configured_bmpd_depth", -1),
        (_ansatz, "configured_bmpd_depth", True),
        (_ansatz, "initial_single_qubit_layer", 1),
        (_ansatz, "ansatz_id", "other"),
        (_initialization, "rule", ""),
        (_initialization, "seed", -1),
        (_initialization, "seed", True),
        (_initialization, "seed", 2**64),
        (_initialization, "scale", math.nan),
        (_initialization, "scale", math.inf),
        (_optimizer, "optimizer_id", "Krotov"),
        (_optimizer, "optimizer_id", "krotov optimizer"),
        (_optimizer, "max_iterations", -1),
        (_optimizer, "max_iterations", True),
        (_optimizer, "optimizer_seed", -1),
        (_optimizer, "optimizer_seed", True),
        (_optimizer, "optimizer_seed", 2**64),
        (_optimizer, "train_trajectories_or_shots", -1),
        (_optimizer, "train_trajectories_or_shots", True),
        (_optimizer, "training_seed", -1),
        (_optimizer, "svd_threshold", -1.0),
        (_optimizer, "svd_threshold", math.inf),
        (_optimizer, "truncation_mode", "unknown"),
        (_optimizer, "min_bond_dimension", 0),
        (_optimizer, "min_bond_dimension", True),
        (_evaluation, "test_trajectories_or_shots", -1),
        (_evaluation, "test_trajectories_or_shots", True),
        (_evaluation, "test_seed", -1),
        (_evaluation, "test_seed", True),
        (_evaluation, "svd_threshold", math.nan),
        (_evaluation, "truncation_mode", "unknown"),
        (_evaluation, "min_bond_dimension", 0),
        (_evaluation, "store_trajectory_sidecar", 1),
        (_evaluation, "confidence_level", 0.0),
        (_evaluation, "confidence_level", 1.0),
        (_evaluation, "confidence_level", math.nan),
        (_evaluation, "confidence_interval_method", "bootstrap"),
    ],
)
def test_leaf_config_records_reject_invalid_values(
    factory: Callable[..., object],
    field: str,
    value: object,
) -> None:
    """Leaf configuration records should reject invalid types, ranges, and identifiers."""
    _assert_rejected(factory, field, value)


@pytest.mark.parametrize(
    ("noise_id", "tjm_dt"),
    [
        ("noiseless", 1.0),
        ("ballarin_coupled", 1.0),
        ("dephasing_1s_all", None),
        ("dephasing_1s_all", 0.0),
        ("depolarizing_2s_2q", -1.0),
    ],
)
def test_noise_config_rejects_incompatible_tjm_steps(noise_id: str, tjm_dt: float | None) -> None:
    """Standard and discrete channels should enforce their distinct TJM-step conventions."""
    with pytest.raises(ValueError, match=r"TJM|tjm_dt"):
        _noise(noise_id, tjm_dt=tjm_dt)


@pytest.mark.parametrize("tjm_dt", [0.25, 0.5, 1.0, 2.0])
def test_standard_noise_accepts_any_explicit_positive_tjm_step(tjm_dt: float) -> None:
    """Nondefault TJM parameterizations should be explicit, valid run-identity inputs."""
    noise = _noise("dephasing_1s_all", tjm_dt=tjm_dt)
    assert noise.tjm_dt == tjm_dt


def test_target_selection_rejects_seed_provenance_from_another_target() -> None:
    """A resolved target must carry the exact generation seed frozen by its identifier."""
    with pytest.raises(ValueError, match="target_seed"):
        _target(target_seed=4001)
    with pytest.raises(ValueError, match="target_seed"):
        _target(target_id="haar_random_1", target_seed=None)


def test_confidence_level_and_method_are_an_atomic_policy() -> None:
    """A confidence request must identify its frozen interval construction."""
    with pytest.raises(ValueError, match="both"):
        _evaluation(confidence_level=0.95, confidence_interval_method=None)
    with pytest.raises(ValueError, match="both"):
        _evaluation(confidence_interval_method="normal_clipped")
    with pytest.raises(TypeError, match="confidence_interval_method"):
        _evaluation(confidence_level=0.95, confidence_interval_method=1)


@pytest.mark.parametrize(
    "noise_id",
    ["", "Noiseless", "noiseless ", "../noiseless", "dephasing_1s", "gaussian_mu0p5_sigma0p1"],
)
def test_noise_config_rejects_unknown_or_malformed_identifiers(noise_id: str) -> None:
    """Noise identifiers should use the exact frozen registry spelling."""
    with pytest.raises(ValueError, match="noise_id"):
        _noise(noise_id, tjm_dt=None)


def test_initialization_requires_warm_start_path_and_checksum_together() -> None:
    """Warm-start provenance should never contain only one member of the path/checksum pair."""
    with pytest.raises(ValueError, match="warm_start"):
        _initialization(rule="warm_start", seed=None, scale=None, warm_start_path="warm.npy")
    with pytest.raises(ValueError, match="warm_start"):
        _initialization(rule="warm_start", seed=None, scale=None, warm_start_checksum=SHA_A)

    initialization = _initialization(
        rule="warm_start",
        seed=None,
        scale=None,
        warm_start_path="warm.npy",
        warm_start_checksum=SHA_A,
    )
    assert initialization.warm_start_path == "warm.npy"
    assert initialization.warm_start_checksum == SHA_A


@pytest.mark.parametrize("factory", [_optimizer, _evaluation])
def test_truncation_config_rejects_a_maximum_below_the_minimum(factory: Callable[..., object]) -> None:
    """A truncation maximum should not be smaller than its mandatory minimum."""
    with pytest.raises(ValueError, match="bond"):
        factory(max_bond_dimension=2, min_bond_dimension=3)


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {"step_size": math.nan},
        {"nested": {"value": math.inf}},
        {1: "non-string-key"},
        {"unsupported": object()},
    ],
)
def test_optimizer_rejects_noncanonical_json_hyperparameters(hyperparameters: Mapping[object, object]) -> None:
    """Optimizer metadata should be finite, string-keyed, and JSON-native recursively."""
    with pytest.raises((TypeError, ValueError), match="hyperparameters"):
        _optimizer(hyperparameters=hyperparameters)


def test_nested_configuration_mappings_are_defensively_frozen() -> None:
    """Caller mutation should not change serialized configuration or its run identifier."""
    schedule = {"kind": "constant"}
    source: dict[str, object] = {"step_size": 0.2, "schedule": schedule}
    optimizer = _optimizer(hyperparameters=source)
    config = _config(optimizer=optimizer)
    original_payload = config.to_dict()
    original_run_id = config.run_id

    source["step_size"] = 99.0
    schedule["kind"] = "exp"

    assert config.to_dict() == original_payload
    assert config.run_id == original_run_id
    with pytest.raises(TypeError):
        cast("dict[str, object]", optimizer.hyperparameters)["new"] = "value"
    with pytest.raises(TypeError):
        cast("dict[str, object]", optimizer.hyperparameters["schedule"])["kind"] = "exp"


def test_frozen_configuration_records_reject_attribute_assignment() -> None:
    """Configuration records should be immutable after validation."""
    config = _config()
    with pytest.raises(FrozenInstanceError):
        cast("Any", config).method_id = "other"
    with pytest.raises(FrozenInstanceError):
        cast("Any", config.target).num_qubits = 12


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"method_id": ""}, "method_id"),
        ({"method_id": "Krotov"}, "method_id"),
        ({"method_id": "krotov/method"}, "method_id"),
        ({"method_version": ""}, "method_version"),
        ({"schema_version": "unknown"}, "schema_version"),
    ],
)
def test_benchmark_config_rejects_invalid_top_level_fields(changes: dict[str, object], match: str) -> None:
    """Top-level benchmark identifiers and schema versions should be strict."""
    with pytest.raises((TypeError, ValueError), match=match):
        _config(**cast("Any", changes))


def test_benchmark_config_enforces_noiseless_v1_training() -> None:
    """Version-one benchmark training should be noiseless and use no trajectory stream."""
    with pytest.raises(ValueError, match="noiseless optimization"):
        _config(training_noise=_noise("dephasing_1s_all"))
    with pytest.raises(ValueError, match="training trajectories"):
        _config(optimizer=_optimizer(train_trajectories_or_shots=1))
    with pytest.raises(ValueError, match="training seed"):
        _config(optimizer=_optimizer(training_seed=29))


@pytest.mark.parametrize(
    ("evaluation", "test_noise"),
    [
        (_evaluation(noisy=False, test_trajectories_or_shots=1), _noise("noiseless")),
        (_evaluation(noisy=False, test_seed=23), _noise("noiseless")),
        (_evaluation(noisy=False), _noise("dephasing_1s_all")),
        (_evaluation(noisy=True, test_seed=None), _noise("dephasing_1s_all")),
    ],
)
def test_benchmark_config_enforces_test_noise_budget_and_seed(
    evaluation: EvaluationConfig,
    test_noise: NoiseConfig,
) -> None:
    """Noiseless and noisy testing should use disjoint budget/seed conventions."""
    with pytest.raises(ValueError, match="test"):
        _config(evaluation=evaluation, test_noise=test_noise)


def test_benchmark_config_rejects_unexecutable_evaluation_policies() -> None:
    """Every accepted resolved configuration should admit a successful result record."""
    with pytest.raises(ValueError, match="confidence interval"):
        _config(noisy=False, evaluation=_evaluation(noisy=False, confidence_level=0.95))
    with pytest.raises(ValueError, match="trajectory sidecar"):
        _config(noisy=False, evaluation=_evaluation(noisy=False, store_trajectory_sidecar=True))
    with pytest.raises(ValueError, match="at least two"):
        _config(evaluation=_evaluation(test_trajectories_or_shots=1, confidence_level=0.95))


@pytest.mark.parametrize(
    "evaluation",
    [
        _evaluation(test_seed=17),
        _evaluation(test_seed=11),
    ],
)
def test_benchmark_config_requires_distinct_resolved_runtime_seeds(evaluation: EvaluationConfig) -> None:
    """Initialization, optimizer, and evaluation streams must not reuse the same seed."""
    with pytest.raises(ValueError, match="distinct resolved seeds"):
        _config(evaluation=evaluation)


def test_benchmark_config_json_and_dict_round_trips_are_lossless_and_deterministic() -> None:
    """Config codecs should preserve typed equality and canonical bytes."""
    config = _config()
    dictionary = config.to_dict()
    encoded = config.to_json()

    assert BenchmarkConfig.from_dict(dictionary) == config
    assert BenchmarkConfig.from_json(encoded) == config
    assert BenchmarkConfig.from_json(encoded).to_json() == encoded
    assert json.loads(encoded) == dictionary


def test_json_decoders_normalize_integer_spelling_for_real_valued_fields() -> None:
    """JSON's integer spelling should be accepted for schema fields in the real-number domain."""
    config_payload = _config().to_dict()
    initialization = cast("dict[str, object]", config_payload["initialization"])
    optimizer = cast("dict[str, object]", config_payload["optimizer"])
    evaluation = cast("dict[str, object]", config_payload["evaluation"])
    test_noise = cast("dict[str, object]", config_payload["test_noise"])
    initialization["scale"] = 1
    optimizer["svd_threshold"] = 0
    evaluation["svd_threshold"] = 0
    test_noise["tjm_dt"] = 1

    decoded_config = BenchmarkConfig.from_json(json.dumps(config_payload))
    assert math.isclose(cast("float", decoded_config.initialization.scale), 1.0, rel_tol=0.0, abs_tol=0.0)
    assert type(decoded_config.initialization.scale) is float
    assert type(decoded_config.optimizer.svd_threshold) is float
    assert type(decoded_config.evaluation.svd_threshold) is float
    assert type(decoded_config.test_noise.tjm_dt) is float

    result_payload = _result(noisy=False).to_dict()
    for field in (
        "train_fidelity",
        "logical_test_noiseless_fidelity",
        "test_noiseless_fidelity",
        "test_noisy_fidelity",
        "optimization_wall_time_seconds",
        "evaluation_wall_time_seconds",
    ):
        result_payload[field] = 0
    result_payload["wall_time_seconds"] = 0.0
    decoded_result = BenchmarkResult.from_json(json.dumps(result_payload))
    assert math.isclose(decoded_result.train_fidelity, 0.0, rel_tol=0.0, abs_tol=0.0)
    assert type(decoded_result.train_fidelity) is float

    failure_payload = _failure().to_dict()
    failure_payload["wall_time_seconds"] = 0
    decoded_failure = BenchmarkFailure.from_json(json.dumps(failure_payload))
    assert math.isclose(decoded_failure.wall_time_seconds, 0.0, rel_tol=0.0, abs_tol=0.0)
    assert type(decoded_failure.wall_time_seconds) is float


def test_json_real_number_normalization_rejects_booleans_and_overflowed_integers() -> None:
    """Boolean and non-finite coercions must not enter real-valued schema fields."""
    payload = _config().to_dict()
    cast("dict[str, object]", payload["optimizer"])["svd_threshold"] = True
    with pytest.raises(TypeError, match="svd_threshold"):
        BenchmarkConfig.from_dict(payload)

    payload = _config().to_dict()
    cast("dict[str, object]", payload["optimizer"])["svd_threshold"] = 10**10000
    with pytest.raises(ValueError, match="finite"):
        BenchmarkConfig.from_dict(payload)

    result_payload = _result().to_dict()
    result_payload["train_fidelity"] = math.inf
    with pytest.raises(ValueError, match="finite"):
        BenchmarkResult.from_dict(result_payload)


def test_config_dict_decoder_rejects_unknown_and_missing_fields() -> None:
    """Dictionary decoders should not silently accept typos or incomplete records."""
    payload = _config().to_dict()
    unknown = dict(payload)
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        BenchmarkConfig.from_dict(unknown)

    missing = dict(payload)
    del missing["method_id"]
    with pytest.raises(ValueError, match="method_id"):
        BenchmarkConfig.from_dict(missing)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_config_json_decoder_rejects_nonfinite_nested_numbers(value: float) -> None:
    """JSON decoding should reject nonstandard and overflowed non-finite numbers."""
    payload = _config().to_dict()
    optimizer = cast("dict[str, object]", payload["optimizer"])
    optimizer["hyperparameters"] = {"bad": value}
    encoded = json.dumps(payload)
    with pytest.raises((TypeError, ValueError), match=r"finite|hyperparameters|Nonstandard"):
        BenchmarkConfig.from_json(encoded)


def test_config_json_decoder_rejects_overflowed_json_number() -> None:
    """A standards-shaped JSON exponent that overflows to infinity should still fail validation."""
    encoded = _config().to_json().replace('"step_size":0.2', '"step_size":1e309')
    with pytest.raises((TypeError, ValueError), match=r"finite|hyperparameters"):
        BenchmarkConfig.from_json(encoded)


def test_run_id_is_the_sha256_of_the_canonical_identity_payload() -> None:
    """The run ID should be content-addressed by a canonical identity document."""
    config = _config()
    canonical = json.dumps(
        config.identity_payload(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert config.run_id == f"spr-v1-{hashlib.sha256(canonical).hexdigest()}"


def test_run_id_ignores_recursive_mapping_insertion_order() -> None:
    """Semantically equal hyperparameter mappings should produce the same run ID."""
    first = _optimizer(
        hyperparameters={
            "alpha": 0.1,
            "schedule": {"kind": "exp", "decay": 0.2},
            "tags": ["one", "two"],
        }
    )
    second = _optimizer(
        hyperparameters={
            "tags": ["one", "two"],
            "schedule": {"decay": 0.2, "kind": "exp"},
            "alpha": 0.1,
        }
    )
    assert _config(optimizer=first).run_id == _config(optimizer=second).run_id


@pytest.mark.parametrize(
    "replacement_config",
    [
        _config(method_id="other_method"),
        _config(method_version="2"),
        _config(target=_target(num_qubits=12)),
        _config(target=_target(target_id="tfim_ferro")),
        _config(target=_target(fixture_checksum=SHA_B)),
        _config(ansatz=_ansatz(configured_bmpd_depth=3)),
        _config(ansatz=_ansatz(initial_single_qubit_layer=False)),
        _config(initialization=_initialization(seed=12)),
        _config(optimizer=_optimizer(max_iterations=5)),
        _config(optimizer=_optimizer(optimizer_seed=18)),
        _config(optimizer=_optimizer(hyperparameters={"step_size": 0.3})),
        _config(evaluation=_evaluation(test_trajectories_or_shots=33)),
        _config(evaluation=_evaluation(test_seed=24)),
        _config(test_noise=_noise("depolarizing_1s_all")),
    ],
)
def test_each_run_identity_input_changes_the_run_id(replacement_config: BenchmarkConfig) -> None:
    """Every benchmark-cell input should participate in the stable run identity."""
    assert replacement_config.run_id != _config().run_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_fidelity", 0.82),
        ("test_noisy_fidelity", 0.73),
        ("optimization_wall_time_seconds", 2.0),
        ("evaluation_wall_time_seconds", 3.0),
        ("notes", "different"),
        ("git_dirty", True),
        ("git_diff_checksum", SHA_B),
        ("parameter_checkpoint_path", "params.npz"),
        ("parameter_checkpoint_checksum", SHA_B),
    ],
)
def test_result_only_fields_do_not_change_the_run_id(field: str, value: object) -> None:
    """Measurements and output provenance should not redefine the planned experiment cell."""
    result = _result()
    changes: dict[str, object] = {field: value}
    if field == "parameter_checkpoint_path":
        changes["parameter_checkpoint_checksum"] = SHA_A
    elif field == "parameter_checkpoint_checksum":
        changes["parameter_checkpoint_path"] = "params.npz"
    elif field == "git_dirty":
        changes["git_diff_checksum"] = SHA_A
    elif field == "git_diff_checksum":
        changes["git_dirty"] = True
    changed = replace(result, **cast("Any", changes))
    assert changed.run_id == result.run_id


def test_complete_target_noise_matrix_has_unique_run_ids() -> None:
    """Every target, size, and test-noise cell should have a unique content address."""
    run_ids: set[str] = set()
    for num_qubits in SUPPORTED_QUBIT_COUNTS:
        for target_id in TARGET_IDS:
            for noise_id in NOISE_IDS:
                noisy = noise_id != "noiseless"
                config = _config(
                    noisy=noisy,
                    target=_target(num_qubits=num_qubits, target_id=target_id),
                    evaluation=_evaluation(noisy=noisy),
                    test_noise=_noise(noise_id),
                )
                assert config.run_id not in run_ids
                run_ids.add(config.run_id)
    assert len(run_ids) == len(SUPPORTED_QUBIT_COUNTS) * len(TARGET_IDS) * len(NOISE_IDS)


def test_circuit_statistics_derive_counts_from_the_evaluated_representation() -> None:
    """Reported gate counts and depth should come from the selected representation."""
    logical = _statistics(evaluated_representation="logical")
    assert logical.num_layers == 4
    assert logical.evaluated_depth == logical.logical_depth
    assert logical.num_1q_gates == logical.logical_num_1q_gates
    assert logical.num_2q_gates == logical.logical_num_2q_gates

    native = _statistics(evaluated_representation="native")
    assert native.evaluated_depth == native.native_depth
    assert native.num_1q_gates == native.native_num_1q_gates
    assert native.num_2q_gates == native.native_num_2q_gates
    assert CircuitStatistics.from_dict(native.to_dict()) == native


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("configured_bmpd_depth", -1),
        ("configured_bmpd_depth", True),
        ("num_parameters", -1),
        ("logical_depth", -1),
        ("logical_num_1q_gates", True),
        ("logical_num_2q_gates", -1),
        ("native_depth", -1),
        ("native_num_1q_gates", -1),
        ("native_num_2q_gates", -1),
        ("native_rzz_count", -1),
        ("pruned_native_rzz_count", -1),
        ("evaluated_representation", "compiled"),
        ("logical_gate_counts", {"rx": -1}),
        ("logical_gate_counts", {"rx": True}),
        ("native_gate_counts", {1: 2}),
    ],
)
def test_circuit_statistics_reject_invalid_values(field: str, value: object) -> None:
    """Circuit statistics should reject negative counts, booleans, and unknown representations."""
    _assert_rejected(_statistics, field, value)


def test_circuit_statistics_reject_inconsistent_native_counts() -> None:
    """Native RZZ and pruning counts should be physically consistent."""
    with pytest.raises(ValueError, match="native_rzz_count"):
        _statistics(native_num_2q_gates=7)
    with pytest.raises(ValueError, match="Retained and pruned"):
        _statistics(pruned_native_rzz_count=1)
    with pytest.raises(ValueError, match="logical_gate_counts"):
        _statistics(logical_gate_counts={"rx": 12})
    with pytest.raises(ValueError, match="native_gate_counts"):
        _statistics(native_gate_counts={"rx": 20, "rzz": 7})


def test_circuit_gate_count_mappings_are_defensively_frozen() -> None:
    """Mutating source count mappings should not modify stored statistics."""
    source = {"rx": 12, "rxx": 4, "ryy": 4}
    statistics = _statistics(logical_gate_counts=source)
    source["rx"] = 99
    assert statistics.logical_gate_counts["rx"] == 12
    with pytest.raises(TypeError):
        cast("dict[str, int]", statistics.logical_gate_counts)["rx"] = 99


def test_valid_result_exposes_derived_status_identifier_and_wall_time() -> None:
    """A success record should derive stable identity and total wall time from its inputs."""
    result = _result()
    assert result.status == "success"
    assert result.run_id == result.config.run_id
    assert result.wall_time_seconds == pytest.approx(2.0)
    assert result.to_dict()["circuit_depth"] == result.circuit_statistics.evaluated_depth


@pytest.mark.parametrize(
    "field",
    [
        "train_fidelity",
        "logical_test_noiseless_fidelity",
        "test_noiseless_fidelity",
        "test_noisy_fidelity",
    ],
)
@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf, -math.inf, True, "0.5", None])
def test_result_rejects_invalid_fidelities(field: str, value: object) -> None:
    """All completed-result fidelities should be finite floats in the closed unit interval."""
    _assert_rejected(_result, field, value)


@pytest.mark.parametrize(
    "field",
    [
        "native_pre_pruning_noiseless_fidelity",
        "noisy_fidelity_standard_deviation",
        "noisy_fidelity_standard_error",
        "confidence_interval_lower",
        "confidence_interval_upper",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "0.1"])
def test_result_rejects_invalid_optional_statistics(field: str, value: object) -> None:
    """Optional uncertainty statistics should still use strict finite numeric types."""
    _assert_rejected(_result, field, value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("native_pre_pruning_noiseless_fidelity", -0.01),
        ("noisy_fidelity_standard_deviation", -0.01),
        ("noisy_fidelity_standard_error", -0.01),
        ("confidence_interval_lower", -0.01),
        ("confidence_interval_upper", 1.01),
        ("sampled_nonidentity_events", -1),
        ("sampled_nonidentity_events", True),
        ("optimization_wall_time_seconds", -0.01),
        ("optimization_wall_time_seconds", math.inf),
        ("evaluation_wall_time_seconds", -0.01),
        ("evaluation_wall_time_seconds", math.nan),
        ("git_dirty", 0),
        ("schema_version", "unknown"),
    ],
)
def test_result_rejects_invalid_ranges_and_types(field: str, value: object) -> None:
    """Result counts, timings, flags, and schema version should be strictly validated."""
    _assert_rejected(_result, field, value)


def test_result_accepts_closed_fidelity_boundaries_and_high_precision_float() -> None:
    """Fidelity validation should include both mathematical boundaries without rounding interiors."""
    result = _result(
        train_fidelity=0.0,
        logical_test_noiseless_fidelity=0.12345678901234566,
        test_noiseless_fidelity=0.12345678901234566,
        test_noisy_fidelity=1.0,
    )
    assert math.isclose(result.train_fidelity, 0.0, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(result.test_noisy_fidelity, 1.0, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(result.test_noiseless_fidelity, 0.12345678901234566, rel_tol=0.0, abs_tol=0.0)


def test_noiseless_result_requires_equal_fidelities_and_no_stochastic_metadata() -> None:
    """A noiseless test row should not claim stochastic uncertainty or sampled noise events."""
    with pytest.raises(ValueError, match="noiseless"):
        _result(noisy=False, test_noisy_fidelity=0.79)
    with pytest.raises(ValueError, match="noiseless"):
        _result(noisy=False, sampled_nonidentity_events=1)
    with pytest.raises(ValueError, match="noiseless"):
        _result(noisy=False, noisy_fidelity_standard_deviation=0.0)
    with pytest.raises(ValueError, match="noiseless"):
        _result(noisy=False, noisy_fidelity_standard_error=0.0)
    with pytest.raises(ValueError, match="noiseless"):
        _result(noisy=False, confidence_interval_lower=0.7, confidence_interval_upper=0.9)


def test_result_preserves_the_three_compilation_fidelity_checkpoints() -> None:
    """Logical rows reuse one fidelity while Ballarin rows retain the pre-pruning diagnostic."""
    with pytest.raises(ValueError, match="same logical noiseless fidelity"):
        _result(logical_test_noiseless_fidelity=0.79)
    with pytest.raises(ValueError, match="forbids"):
        _result(native_pre_pruning_noiseless_fidelity=0.79)

    ballarin_config = _config(test_noise=_noise("ballarin_coupled"))
    with pytest.raises(ValueError, match="requires a pre-pruning"):
        _result(config=ballarin_config, native_pre_pruning_noiseless_fidelity=None)
    result = _result(
        config=ballarin_config,
        logical_test_noiseless_fidelity=0.81,
        native_pre_pruning_noiseless_fidelity=0.805,
        test_noiseless_fidelity=0.80,
    )
    assert result.native_pre_pruning_noiseless_fidelity == pytest.approx(0.805)


def test_confidence_interval_fields_form_a_valid_pair() -> None:
    """Confidence interval bounds should be supplied together and ordered within the unit interval."""
    with pytest.raises(ValueError, match=r"[Cc]onfidence"):
        _result(confidence_interval_lower=0.7)
    with pytest.raises(ValueError, match=r"[Cc]onfidence"):
        _result(confidence_interval_upper=0.8)
    with pytest.raises(ValueError, match="confidence"):
        _result(confidence_interval_lower=0.8, confidence_interval_upper=0.7)

    lower, upper = _normal_interval(0.74, 0.04 / math.sqrt(32))
    result = _result(
        config=_config(evaluation=_evaluation(confidence_level=0.95)),
        confidence_interval_lower=lower,
        confidence_interval_upper=upper,
    )
    assert result.confidence_interval_lower == pytest.approx(lower)
    assert result.confidence_interval_upper == pytest.approx(upper)


def test_success_result_requires_a_checkpoint_and_optional_sidecars_are_atomic() -> None:
    """Successful runs must be resumable; optional trajectory artifacts remain all-or-nothing."""
    with pytest.raises((TypeError, ValueError), match="parameter_checkpoint_path"):
        _result(parameter_checkpoint_path=None)
    with pytest.raises((TypeError, ValueError), match="parameter_checkpoint_checksum"):
        _result(parameter_checkpoint_checksum=None)
    with pytest.raises(ValueError, match=r"checksum|path"):
        _result(trajectory_sidecar_path="artifact.bin")
    with pytest.raises(ValueError, match=r"checksum|path"):
        _result(trajectory_sidecar_checksum=SHA_A)


def test_result_requires_statistics_for_the_configured_depth() -> None:
    """Result circuit statistics should describe the ansatz stored in the same run config."""
    with pytest.raises(ValueError, match="BMPD depths"):
        _result(circuit_statistics=_statistics(configured_bmpd_depth=3))
    with pytest.raises(ValueError, match="parameter count"):
        _result(circuit_statistics=_statistics(num_parameters=107))


def test_result_json_and_dict_round_trips_are_lossless_and_deterministic() -> None:
    """Success codecs should preserve all nested typed records and stable JSON bytes."""
    result = _result(
        config=_config(evaluation=_evaluation(store_trajectory_sidecar=True)),
        notes='comma, quote " and unicode λ\nsecond line',
        parameter_checkpoint_path="parameters.npz",
        parameter_checkpoint_checksum=SHA_A,
        trajectory_sidecar_path="trajectory data.npz",
        trajectory_sidecar_checksum=SHA_B,
    )
    dictionary = result.to_dict()
    encoded = result.to_json()

    assert BenchmarkResult.from_dict(dictionary) == result
    assert BenchmarkResult.from_json(encoded) == result
    assert BenchmarkResult.from_json(encoded).to_json() == encoded
    assert benchmark_record_from_dict(dictionary) == result
    assert benchmark_record_from_json(encoded) == result


def test_result_software_versions_are_defensively_frozen() -> None:
    """Runtime provenance should not change after a result has been constructed."""
    versions = _software_versions()
    result = _result(software_versions=versions)
    versions["numpy"] = "changed"
    assert result.software_versions["numpy"] == "2.3.0"
    with pytest.raises(TypeError):
        cast("dict[str, str]", result.software_versions)["numpy"] = "changed"


def test_failure_exposes_derived_status_and_run_identifier() -> None:
    """A failure row should preserve the planned run identity without inventing metrics."""
    failure = _failure()
    assert failure.status == "failure"
    assert failure.run_id == failure.config.run_id
    assert "fidelity" not in failure.to_dict()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("failure_phase", ""),
        ("failure_phase", "unknown"),
        ("exception_type", ""),
        ("message", ""),
        ("retryable", 1),
        ("attempt", 0),
        ("attempt", True),
        ("wall_time_seconds", -0.1),
        ("wall_time_seconds", math.inf),
        ("git_dirty", 0),
        ("schema_version", "unknown"),
    ],
)
def test_failure_rejects_invalid_values(field: str, value: object) -> None:
    """Failure records should retain strict phase, diagnostic, count, and provenance types."""
    _assert_rejected(_failure, field, value)


def test_failure_requires_checkpoint_path_and_checksum_together() -> None:
    """A partial failure must not claim an unverifiable parameter checkpoint."""
    with pytest.raises(ValueError, match=r"checksum|path"):
        _failure(parameter_checkpoint_path="partial.npz")
    with pytest.raises(ValueError, match=r"checksum|path"):
        _failure(parameter_checkpoint_checksum=SHA_A)


def test_failure_from_exception_preserves_type_message_traceback_and_metadata() -> None:
    """The exception helper should create serializable diagnostics from an active exception."""
    try:
        msg = "failure with unicode λ"
        _raise_runtime_error(msg)
    except RuntimeError as error:
        formatted_traceback = traceback_module.format_exc()
        failure = BenchmarkFailure.from_exception(
            config=_config(),
            failure_phase="evaluation",
            exception=error,
            retryable=True,
            attempt=2,
            wall_time_seconds=0.25,
            software_versions=_software_versions(),
            git_commit=GIT_COMMIT,
            git_dirty=True,
            git_diff_checksum=SHA_A,
            traceback=formatted_traceback,
            notes="retry later",
        )

    assert failure.exception_type == "RuntimeError"
    assert failure.message == "failure with unicode λ"
    assert failure.traceback is not None
    assert "RuntimeError: failure with unicode λ" in failure.traceback
    assert failure.retryable is True
    assert failure.attempt == 2


def test_failure_json_and_dict_round_trips_are_lossless_and_deterministic() -> None:
    """Failure codecs and generic discriminated decoders should preserve diagnostics."""
    failure = _failure(
        message='comma, quote " and unicode λ',
        parameter_checkpoint_path="partial parameters.npz",
        parameter_checkpoint_checksum=SHA_A,
        notes="first line\nsecond line",
    )
    dictionary = failure.to_dict()
    encoded = failure.to_json()

    assert BenchmarkFailure.from_dict(dictionary) == failure
    assert BenchmarkFailure.from_json(encoded) == failure
    assert BenchmarkFailure.from_json(encoded).to_json() == encoded
    assert benchmark_record_from_dict(dictionary) == failure
    assert benchmark_record_from_json(encoded) == failure


@pytest.mark.parametrize("record", [_result(), _failure()])
def test_generic_decoder_rejects_unknown_record_status(record: BenchmarkResult | BenchmarkFailure) -> None:
    """The record union should use an explicit and closed status discriminant."""
    payload = record.to_dict()
    payload["status"] = "partial"
    with pytest.raises(ValueError, match="status"):
        benchmark_record_from_dict(payload)


@pytest.mark.parametrize("record", [_result(), _failure()])
def test_csv_rows_use_the_stable_union_schema(record: BenchmarkResult | BenchmarkFailure) -> None:
    """Success and failure records should emit exactly the common stable CSV columns."""
    row = record.to_csv_row()
    assert tuple(row) == tuple(CSV_COLUMNS)
    assert row["status"] == record.status
    assert row["run_id"] == record.run_id


@pytest.mark.parametrize(
    "record",
    [
        _result(
            notes='comma, quote " and unicode λ\nsecond line',
            parameter_checkpoint_path="parameters.npz",
            parameter_checkpoint_checksum=SHA_A,
        ),
        _failure(message='comma, quote " and unicode λ', notes="first line\nsecond line"),
    ],
)
def test_csv_round_trip_is_lossless_for_success_and_failure(
    record: BenchmarkResult | BenchmarkFailure,
) -> None:
    """CSV should preserve nested JSON, precise numbers, optional values, and escaped text."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerow(record.to_csv_row())
    stream.seek(0)
    row = next(csv.DictReader(stream))

    decoded = benchmark_record_from_csv_row(row)
    assert decoded == record
    assert type(decoded) is type(record)


def test_csv_union_preserves_mixed_record_order() -> None:
    """A shared CSV stream should preserve adjacent successes and failures without ambiguity."""
    records: list[BenchmarkResult | BenchmarkFailure] = [_result(), _failure(), _result(noisy=False)]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(record.to_csv_row() for record in records)
    stream.seek(0)

    decoded = [benchmark_record_from_csv_row(row) for row in csv.DictReader(stream)]
    assert decoded == records


@pytest.mark.parametrize("record", [_result(), _failure()])
def test_csv_decoder_rejects_missing_and_unknown_columns(record: BenchmarkResult | BenchmarkFailure) -> None:
    """CSV decoding should not silently accept schema drift or truncated rows."""
    row = record.to_csv_row()
    missing = dict(row)
    del missing[CSV_COLUMNS[-1]]
    with pytest.raises(ValueError, match=CSV_COLUMNS[-1]):
        benchmark_record_from_csv_row(missing)

    unknown = dict(row)
    unknown["unexpected"] = ""
    with pytest.raises(ValueError, match="unexpected"):
        benchmark_record_from_csv_row(unknown)


def test_high_precision_csv_float_round_trip() -> None:
    """CSV conversion should use enough precision to recover binary floating-point values."""
    value = 0.12345678901234566
    result = _result(test_noisy_fidelity=value)
    decoded = benchmark_record_from_csv_row(result.to_csv_row())
    assert isinstance(decoded, BenchmarkResult)
    assert decoded.test_noisy_fidelity == value


def test_deterministic_property_sweep_round_trips_configs_and_records() -> None:
    """A deterministic matrix should preserve equality and identity through every public codec."""
    cases = [
        (6, "gaussian_mu0p5_sigma0p1", "noiseless"),
        (6, "tfim_critical", "ballarin_coupled"),
        (12, "haar_random_2", "dephasing_2s_2q"),
        (12, "random_mps_bond3", "depolarizing_1s2s_all"),
    ]
    for index, (num_qubits, target_id, noise_id) in enumerate(cases):
        noisy = noise_id != "noiseless"
        config = _config(
            noisy=noisy,
            target=_target(num_qubits=num_qubits, target_id=target_id),
            initialization=_initialization(seed=index),
            optimizer=_optimizer(optimizer_seed=100 + index),
            evaluation=_evaluation(noisy=noisy, test_seed=200 + index if noisy else None),
            test_noise=_noise(noise_id),
        )
        config_round_trip = BenchmarkConfig.from_json(config.to_json())
        assert config_round_trip == config
        assert config_round_trip.run_id == config.run_id

        result = _result(
            noisy=noisy,
            config=config,
            test_noisy_fidelity=0.74 if noisy else 0.80,
            sampled_nonidentity_events=index if noisy else 0,
        )
        assert benchmark_record_from_json(result.to_json()) == result
        assert benchmark_record_from_csv_row(result.to_csv_row()) == result


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (_target, "target_id", 1),
        (_target, "fixture_format", 1),
        (_target, "fixture_checksum", 1),
        (_ansatz, "ansatz_id", 1),
        (_initialization, "rule", 1),
        (_optimizer, "truncation_mode", 1),
        (_evaluation, "truncation_mode", 1),
        (_config, "method_id", 1),
        (_config, "method_version", 1),
    ],
)
def test_schema_identifiers_reject_non_string_values(
    factory: Callable[..., object],
    field: str,
    value: object,
) -> None:
    """Identifiers should never be coerced from non-string scalar values."""
    with pytest.raises(TypeError, match=field):
        factory(**cast("Any", {field: value}))


def test_noise_identifiers_and_definition_version_use_strict_frozen_values() -> None:
    """Noise identifiers should reject non-strings and stale definition versions."""
    with pytest.raises(TypeError, match="noise_id"):
        _noise(cast("Any", 1), tjm_dt=None)
    with pytest.raises(ValueError, match="definition_version"):
        _noise(definition_version="yaqs.state_preparation.noise.v0")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _initialization(seed=0),
        lambda: _initialization(seed=2**64 - 1),
        lambda: _optimizer(optimizer_seed=0),
        lambda: _optimizer(optimizer_seed=2**64 - 1),
        lambda: _optimizer(training_seed=2**64 - 1),
        lambda: _evaluation(test_seed=0),
        lambda: _evaluation(test_seed=2**64 - 1),
    ],
    ids=[
        "initialization-zero",
        "initialization-max",
        "optimizer-zero",
        "optimizer-max",
        "training-max",
        "evaluation-zero",
        "evaluation-max",
    ],
)
def test_seed_fields_accept_the_complete_uint64_boundary(factory: Callable[[], object]) -> None:
    """Every seed field should include zero and the maximum unsigned 64-bit value."""
    assert factory() is not None


def test_initialization_rules_enforce_all_rule_specific_fields() -> None:
    """Each initialization rule should accept only its documented parameter source."""
    with pytest.raises(ValueError, match="Random initialization"):
        _initialization(seed=None)
    with pytest.raises(ValueError, match="Random initialization"):
        _initialization(scale=None)
    with pytest.raises(ValueError, match="cannot specify a warm start"):
        _initialization(warm_start_path="warm.npy", warm_start_checksum=SHA_A)
    with pytest.raises(ValueError, match="requires a path"):
        _initialization(rule="warm_start", seed=None, scale=None)
    with pytest.raises(ValueError, match="cannot specify seed or scale"):
        _initialization(rule="warm_start", warm_start_path="warm.npy", warm_start_checksum=SHA_A)
    with pytest.raises(ValueError, match="cannot specify seed or scale"):
        _initialization(rule="zeros")
    with pytest.raises(ValueError, match="cannot specify a warm start"):
        _initialization(
            rule="zeros",
            seed=None,
            scale=None,
            warm_start_path="warm.npy",
            warm_start_checksum=SHA_A,
        )

    assert _initialization(rule="zeros", seed=None, scale=None).rule == "zeros"
    assert _initialization(rule="random_uniform").rule == "random_uniform"


@pytest.mark.parametrize("path", ["../artifact.npz", "dir//artifact.npz", r"dir\artifact.npz", "/artifact.npz"])
def test_artifact_paths_must_be_normalized_relative_posix_paths(path: str) -> None:
    """Serialized artifact locations should be portable and traversal-free."""
    with pytest.raises(ValueError, match="parameter_checkpoint_path"):
        _failure(parameter_checkpoint_path=path, parameter_checkpoint_checksum=SHA_A)


def test_optimizer_hyperparameters_require_a_mapping_and_freeze_sequences() -> None:
    """JSON hyperparameters should require an object and detach nested sequences."""
    with pytest.raises(TypeError, match="hyperparameters"):
        _optimizer(hyperparameters=["not", "a", "mapping"])

    sequence = [1, {"nested": [2, -0.0]}]
    optimizer = _optimizer(hyperparameters={"sequence": sequence})
    sequence.append(3)
    assert optimizer.to_dict()["hyperparameters"] == {"sequence": [1, {"nested": [2, 0.0]}]}


def test_software_version_provenance_requires_complete_string_metadata() -> None:
    """Result provenance should require all mandatory tools and string versions."""
    missing = _software_versions()
    del missing["yaqs"]
    with pytest.raises(ValueError, match="yaqs"):
        _result(software_versions=missing)

    non_string = cast("dict[str, object]", _software_versions())
    non_string["numpy"] = 2
    with pytest.raises(TypeError, match="numpy"):
        _failure(software_versions=non_string)

    with pytest.raises(TypeError, match="software_versions"):
        _result(software_versions=[])


def test_json_decoder_rejects_non_strings_non_objects_and_duplicate_keys() -> None:
    """Strict JSON entry points should reject ambiguous or structurally invalid documents."""
    with pytest.raises(TypeError, match="payload"):
        BenchmarkConfig.from_json(cast("Any", 1))
    with pytest.raises(TypeError, match="top level"):
        BenchmarkConfig.from_json("[]")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        BenchmarkConfig.from_json('{"method_id":"krotov","method_id":"adam"}')


def test_config_decoder_rejects_non_mapping_nested_records() -> None:
    """Typed nested configuration records should not be accepted as scalar JSON values."""
    payload = _config().to_dict()
    payload["target"] = "not-an-object"
    with pytest.raises(TypeError, match="target"):
        BenchmarkConfig.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    ["target", "ansatz", "initialization", "optimizer", "evaluation", "training_noise", "test_noise"],
)
def test_benchmark_config_requires_typed_nested_records(field: str) -> None:
    """Direct construction should reject objects of the wrong nested schema type."""
    with pytest.raises(TypeError, match=field):
        _config(**cast("Any", {field: object()}))


def test_leaf_decoders_verify_derived_aliases() -> None:
    """Deserialization should reject reporting aliases that disagree with typed sources."""
    ansatz = _ansatz().to_dict()
    ansatz["num_layers"] = 999
    with pytest.raises(ValueError, match="num_layers"):
        AnsatzConfig.from_dict(ansatz)
    ansatz["num_layers"] = True
    with pytest.raises(ValueError, match="num_layers"):
        AnsatzConfig.from_dict(ansatz)

    for alias in ("num_layers", "evaluated_depth", "num_1q_gates", "num_2q_gates"):
        statistics = _statistics().to_dict()
        statistics[alias] = cast("int", statistics[alias]) + 1
        with pytest.raises(ValueError, match=alias):
            CircuitStatistics.from_dict(statistics)

    statistics = _statistics().to_dict()
    statistics["num_1q_gates"] = True
    with pytest.raises(ValueError, match="num_1q_gates"):
        CircuitStatistics.from_dict(statistics)


def test_circuit_statistics_reject_additional_representation_and_gate_detail_errors() -> None:
    """Circuit statistics should strictly type their representation and cross-check RZZ detail."""
    with pytest.raises(TypeError, match="evaluated_representation"):
        _statistics(evaluated_representation=1)
    with pytest.raises(ValueError, match="native_gate_counts"):
        _statistics(native_gate_counts={"rzz": 7})


def test_result_requires_typed_config_and_circuit_statistics() -> None:
    """Successful results should embed validated typed config and statistics records."""
    result = _result()
    with pytest.raises(TypeError, match="config"):
        replace(result, config=cast("Any", object()))
    with pytest.raises(TypeError, match="circuit_statistics"):
        replace(result, circuit_statistics=cast("Any", object()))


def test_result_representation_must_match_the_noise_model() -> None:
    """Standard noise uses logical statistics while Ballarin noise uses native statistics."""
    with pytest.raises(ValueError, match="logical"):
        _result(circuit_statistics=_statistics(evaluated_representation="native"))

    ballarin = _config(test_noise=_noise("ballarin_coupled"))
    with pytest.raises(ValueError, match="native"):
        _result(config=ballarin, circuit_statistics=_statistics(evaluated_representation="logical"))


@pytest.mark.parametrize("factory", [_result, _failure])
def test_result_records_validate_git_and_optional_text_provenance(factory: Callable[..., object]) -> None:
    """Success and failure records should enforce Git-dirty and checksum consistency."""
    with pytest.raises(ValueError, match="git_commit"):
        factory(git_commit="NOT-A-COMMIT")
    with pytest.raises(ValueError, match="git_commit"):
        factory(git_commit="abcdef0")
    assert factory(git_commit="d" * 64) is not None
    with pytest.raises(ValueError, match="git_diff_checksum"):
        factory(git_dirty=True)
    with pytest.raises(ValueError, match="git_diff_checksum"):
        factory(git_diff_checksum=SHA_A)
    with pytest.raises(TypeError, match="notes"):
        factory(notes=1)


def test_noisy_result_sampling_uncertainty_matches_the_evaluation_budget() -> None:
    """Noisy rows should report mathematically consistent uncertainty for their sample count."""
    single_sample = _config(evaluation=_evaluation(test_trajectories_or_shots=1))
    assert _result(config=single_sample).noisy_fidelity_standard_deviation is None
    with pytest.raises(ValueError, match="single trajectory"):
        _result(
            config=single_sample,
            noisy_fidelity_standard_deviation=0.1,
            noisy_fidelity_standard_error=None,
        )

    with pytest.raises(ValueError, match="requires standard deviation"):
        _result(noisy_fidelity_standard_deviation=None)
    with pytest.raises(ValueError, match="requires standard deviation"):
        _result(noisy_fidelity_standard_error=None)
    with pytest.raises(ValueError, match="divided by sqrt"):
        _result(noisy_fidelity_standard_error=0.1)


def test_result_confidence_interval_matches_value_and_requested_setting() -> None:
    """Confidence bounds should use the requested frozen method and exist exactly when requested."""
    configured = _config(evaluation=_evaluation(confidence_level=0.95))
    with pytest.raises(ValueError, match="cannot exceed"):
        _result(config=configured, confidence_interval_lower=0.80, confidence_interval_upper=0.70)
    with pytest.raises(ValueError, match="clipped normal"):
        _result(config=configured, confidence_interval_lower=0.80, confidence_interval_upper=0.90)
    with pytest.raises(ValueError, match="confidence-level setting"):
        _result(config=configured)
    with pytest.raises(ValueError, match="confidence-level setting"):
        _result(confidence_interval_lower=0.70, confidence_interval_upper=0.78)


def test_result_sidecar_artifacts_match_the_evaluation_storage_setting() -> None:
    """Trajectory sidecars should be present if and only if storage was requested."""
    configured = _config(evaluation=_evaluation(store_trajectory_sidecar=True))
    with pytest.raises(ValueError, match="store_trajectory_sidecar"):
        _result(config=configured)
    with pytest.raises(ValueError, match="store_trajectory_sidecar"):
        _result(trajectory_sidecar_path="samples.npz", trajectory_sidecar_checksum=SHA_A)


def test_result_decoder_rejects_wrong_status_and_reporting_aliases() -> None:
    """The success decoder should validate its own discriminator and derived aliases."""
    payload = _result().to_dict()
    payload["status"] = "failure"
    with pytest.raises(ValueError, match="status"):
        BenchmarkResult.from_dict(payload)

    payload = _result().to_dict()
    payload["run_id"] = "spr-v1-" + "0" * 64
    with pytest.raises(ValueError, match="run_id"):
        BenchmarkResult.from_dict(payload)

    payload = _result().to_dict()
    payload["circuit_depth"] = True
    with pytest.raises(ValueError, match="circuit_depth"):
        BenchmarkResult.from_dict(payload)

    sequence_config = _config(
        optimizer=_optimizer(hyperparameters={"labels": ["first", "second"]}),
    )
    payload = _result(config=sequence_config).to_dict()
    budget = cast("dict[str, object]", payload["optimizer_budget"])
    budget["hyperparameters"] = {"labels": ["first"]}
    with pytest.raises(ValueError, match="optimizer_budget"):
        BenchmarkResult.from_dict(payload)

    payload = _result(config=sequence_config).to_dict()
    budget = cast("dict[str, object]", payload["optimizer_budget"])
    budget["hyperparameters"] = {"labels": ["first", True]}
    with pytest.raises(ValueError, match="optimizer_budget"):
        BenchmarkResult.from_dict(payload)

    payload = _result(config=sequence_config).to_dict()
    budget = cast("dict[str, object]", payload["optimizer_budget"])
    budget["unexpected"] = True
    with pytest.raises(ValueError, match="optimizer_budget"):
        BenchmarkResult.from_dict(payload)


def test_failure_rejects_additional_strict_types_and_accepts_multiline_messages() -> None:
    """Failure diagnostics should preserve free-form messages while strictly typing metadata."""
    assert _failure(message="line one\nline two").message == "line one\nline two"
    with pytest.raises(TypeError, match="config"):
        _failure(config=object())
    with pytest.raises(TypeError, match="failure_phase"):
        _failure(failure_phase=1)
    with pytest.raises(TypeError, match="message"):
        _failure(message=1)
    with pytest.raises(TypeError, match="traceback"):
        _failure(traceback=1)


def test_failure_decoder_rejects_wrong_status_and_reporting_aliases() -> None:
    """The failure decoder should validate its own discriminator and derived aliases."""
    payload = _failure().to_dict()
    payload["status"] = "success"
    with pytest.raises(ValueError, match="status"):
        BenchmarkFailure.from_dict(payload)

    payload = _failure().to_dict()
    payload["method"] = "other"
    with pytest.raises(ValueError, match="method"):
        BenchmarkFailure.from_dict(payload)

    payload = _failure().to_dict()
    payload["num_layers"] = True
    with pytest.raises(ValueError, match="num_layers"):
        BenchmarkFailure.from_dict(payload)


def test_failure_from_exception_rejects_non_exceptions_and_handles_empty_messages() -> None:
    """Exception conversion should require exceptions and supply a useful fallback message."""
    arguments: dict[str, object] = {
        "config": _config(),
        "failure_phase": "evaluation",
        "software_versions": _software_versions(),
        "git_commit": GIT_COMMIT,
        "git_dirty": False,
    }
    with pytest.raises(TypeError, match="exception"):
        BenchmarkFailure.from_exception(exception=cast("Any", object()), **cast("Any", arguments))

    failure = BenchmarkFailure.from_exception(exception=RuntimeError(), **cast("Any", arguments))
    assert failure.message == "RuntimeError"


@pytest.mark.parametrize("value", ["01", "+1", "1.0", "not-an-integer", object()])
def test_csv_decoder_rejects_noncanonical_integer_cells(value: object) -> None:
    """CSV integer columns should use an unambiguous canonical base-ten spelling."""
    row = _result().to_csv_row()
    row["num_qubits"] = value
    with pytest.raises(ValueError, match="num_qubits"):
        benchmark_record_from_csv_row(row)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ("not-a-float", ValueError),
        ("inf", ValueError),
        (object(), TypeError),
    ],
)
def test_csv_decoder_rejects_invalid_float_cells(value: object, error: type[Exception]) -> None:
    """CSV float columns should reject malformed, non-finite, and non-scalar values."""
    row = _result().to_csv_row()
    row["train_fidelity"] = value
    with pytest.raises(error, match="train_fidelity"):
        benchmark_record_from_csv_row(row)


def test_csv_decoder_validates_boolean_and_json_cells() -> None:
    """CSV Boolean and JSON cells should support documented forms without coercing others."""
    dirty_result = _result(git_dirty=True, git_diff_checksum=SHA_A)
    dirty_row = dirty_result.to_csv_row()
    dirty_row["git_dirty"] = "True"
    assert benchmark_record_from_csv_row(dirty_row) == dirty_result

    row = _result().to_csv_row()
    row["git_dirty"] = "not-a-bool"
    with pytest.raises(ValueError, match="git_dirty"):
        benchmark_record_from_csv_row(row)

    result = _result()
    row = result.to_csv_row()
    row["config"] = result.config.to_dict()
    assert benchmark_record_from_csv_row(row) == result

    row = result.to_csv_row()
    row["config"] = object()
    with pytest.raises(TypeError, match="config"):
        benchmark_record_from_csv_row(row)


def test_csv_decoder_rejects_an_unknown_status_after_cell_decoding() -> None:
    """CSV decoding should keep the success/failure discriminator closed."""
    row = _result().to_csv_row()
    row["status"] = "partial"
    with pytest.raises(ValueError, match="CSV status"):
        benchmark_record_from_csv_row(row)


def test_non_identity_output_policies_do_not_change_run_identity() -> None:
    """Artifact spelling, sidecar storage, and CI post-processing should not split scientific cells."""
    first_warm_start = _config(
        initialization=_initialization(
            rule="warm_start",
            seed=None,
            scale=None,
            warm_start_path="first/warm.npy",
            warm_start_checksum=SHA_A,
        )
    )
    second_warm_start = _config(
        initialization=_initialization(
            rule="warm_start",
            seed=None,
            scale=None,
            warm_start_path="second/warm.npy",
            warm_start_checksum=SHA_A,
        )
    )
    assert first_warm_start.run_id == second_warm_start.run_id
    assert _config().run_id == _config(evaluation=_evaluation(store_trajectory_sidecar=True)).run_id
    assert _config().run_id == _config(evaluation=_evaluation(confidence_level=0.95)).run_id
    assert (
        _config(optimizer=_optimizer(hyperparameters={"zero": -0.0})).run_id
        == _config(optimizer=_optimizer(hyperparameters={"zero": 0.0})).run_id
    )
