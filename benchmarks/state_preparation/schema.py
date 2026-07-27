# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Typed configuration and result records for state-preparation benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import starmap
from pathlib import PurePosixPath
from statistics import NormalDist
from types import MappingProxyType
from typing import NoReturn, cast

from .constants import (
    BALLARIN_NOISE_ID,
    NOISE_IDS,
    NOISELESS_NOISE_ID,
    STANDARD_NOISE_IDS,
    SUPPORTED_QUBIT_COUNTS,
    TARGET_FIXTURE_FORMAT,
    TARGET_GENERATION_SEEDS,
    TARGET_IDS,
)

CONFIG_SCHEMA_VERSION = "yaqs.state_preparation.config.v1"
RESULT_SCHEMA_VERSION = "yaqs.state_preparation.result.v1"
RUN_IDENTITY_VERSION = "yaqs.state_preparation.run_identity.v1"
NOISE_DEFINITION_VERSION = "yaqs.state_preparation.noise.v1"
RUN_ID_PREFIX = "spr-v1-"
ANSATZ_ID = "bmpd_brickwall"

INITIALIZATION_RULES = ("zeros", "random_uniform", "random_normal", "warm_start")
TRUNCATION_MODES = ("discarded_weight", "relative")
CONFIDENCE_INTERVAL_METHODS = ("normal_clipped",)
EVALUATED_REPRESENTATIONS = ("logical", "native")
FAILURE_PHASES = (
    "target_loading",
    "ansatz",
    "initialization",
    "optimization",
    "compilation",
    "checkpoint",
    "evaluation",
    "serialization",
)

_UINT64_MAX = 2**64 - 1
_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REQUIRED_SOFTWARE_VERSIONS = frozenset({"yaqs", "python", "numpy", "scipy"})


def _raise_type_error(name: str, expected: str, value: object) -> NoReturn:
    """Raise a consistent strict-type validation error.

    Raises:
        TypeError: Always, with a consistently formatted validation message.
    """
    msg = f"{name} must be {expected}; received {type(value).__name__}."
    raise TypeError(msg)


def _validate_string(value: object, name: str) -> str:
    """Validate a nonempty string without leading, trailing, or control whitespace.

    Returns:
        The validated string.

    Raises:
        ValueError: If the string is empty or contains invalid whitespace.
    """
    if type(value) is not str:
        _raise_type_error(name, "a string", value)
    if not value or value != value.strip() or any(character.isspace() and character != " " for character in value):
        msg = f"{name} must be a nonempty string without surrounding or control whitespace."
        raise ValueError(msg)
    return value


def _validate_nonempty_text(value: object, name: str) -> str:
    """Validate nonempty free-form text while preserving internal whitespace.

    Returns:
        The validated text.

    Raises:
        ValueError: If the text contains only whitespace.
    """
    if type(value) is not str:
        _raise_type_error(name, "a string", value)
    if not value.strip():
        msg = f"{name} must contain non-whitespace text."
        raise ValueError(msg)
    return value


def _validate_slug(value: object, name: str) -> str:
    """Validate a stable lowercase identifier.

    Returns:
        The validated identifier.

    Raises:
        ValueError: If the identifier does not follow the slug format.
    """
    text = _validate_string(value, name)
    if _SLUG_PATTERN.fullmatch(text) is None:
        msg = f"{name} must be a lowercase identifier containing only letters, digits, '.', '_', or '-'."
        raise ValueError(msg)
    return text


def _validate_bool(value: object, name: str) -> bool:
    """Validate an exact Boolean value.

    Returns:
        The validated Boolean value.
    """
    if type(value) is not bool:
        _raise_type_error(name, "a bool", value)
    return value


def _validate_count(value: object, name: str, *, minimum: int = 0) -> int:
    """Validate an exact integer count.

    Returns:
        The validated count.

    Raises:
        ValueError: If the count is below ``minimum``.
    """
    if type(value) is not int:
        _raise_type_error(name, "an int", value)
    if value < minimum:
        msg = f"{name} must be at least {minimum}."
        raise ValueError(msg)
    return value


def _validate_seed(value: object, name: str, *, allow_none: bool = False) -> int | None:
    """Validate a seed in NumPy SeedSequence's unsigned 64-bit input range.

    Returns:
        The validated seed, or ``None`` when allowed.

    Raises:
        ValueError: If the seed is outside the unsigned 64-bit range.
    """
    if value is None and allow_none:
        return None
    if type(value) is not int:
        _raise_type_error(name, "an int", value)
    if not 0 <= value <= _UINT64_MAX:
        msg = f"{name} must be between 0 and {_UINT64_MAX}."
        raise ValueError(msg)
    return value


def _validate_float(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    """Validate an exact finite float with optional bounds.

    Returns:
        The validated finite float.

    Raises:
        ValueError: If the value is non-finite or outside the requested bounds.
    """
    if type(value) is not float:
        _raise_type_error(name, "a float", value)
    if not math.isfinite(value):
        msg = f"{name} must be finite."
        raise ValueError(msg)
    below_minimum = value < minimum if minimum_inclusive else value <= minimum
    if below_minimum:
        relation = "at least" if minimum_inclusive else "greater than"
        msg = f"{name} must be {relation} {minimum}."
        raise ValueError(msg)
    if maximum is not None:
        above_maximum = value > maximum if maximum_inclusive else value >= maximum
        if above_maximum:
            relation = "at most" if maximum_inclusive else "less than"
            msg = f"{name} must be {relation} {maximum}."
            raise ValueError(msg)
    return value


def _validate_optional_float(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float | None:
    """Validate an optional finite float.

    Returns:
        The validated finite float, or ``None``.
    """
    if value is None:
        return None
    return _validate_float(
        value,
        name,
        minimum=minimum,
        maximum=maximum,
        minimum_inclusive=minimum_inclusive,
        maximum_inclusive=maximum_inclusive,
    )


def _json_float(value: object, name: str) -> float:
    """Normalize a finite JSON number without accepting Boolean values.

    Returns:
        The value normalized to a Python float.

    Raises:
        ValueError: If conversion overflows or produces a non-finite value.
    """
    if type(value) not in {int, float}:
        _raise_type_error(name, "a JSON number", value)
    try:
        result = float(cast("int | float", value))
    except OverflowError as error:
        msg = f"{name} must be finite."
        raise ValueError(msg) from error
    if not math.isfinite(result):
        msg = f"{name} must be finite."
        raise ValueError(msg)
    return result


def _json_optional_float(value: object, name: str) -> float | None:
    """Normalize an optional finite JSON number.

    Returns:
        The value normalized to a Python float, or ``None``.
    """
    return None if value is None else _json_float(value, name)


def _validate_fidelity(value: object, name: str) -> float:
    """Validate a fidelity in the closed unit interval.

    Returns:
        The validated fidelity.
    """
    return _validate_float(value, name, minimum=0.0, maximum=1.0)


def _validate_checksum(value: object, name: str, *, allow_none: bool = False) -> str | None:
    """Validate a prefixed lowercase SHA-256 checksum.

    Returns:
        The validated checksum, or ``None`` when allowed.

    Raises:
        ValueError: If the checksum does not use the required format.
    """
    if value is None and allow_none:
        return None
    text = _validate_string(value, name)
    if _SHA256_PATTERN.fullmatch(text) is None:
        msg = f"{name} must have the form 'sha256:' followed by 64 lowercase hexadecimal characters."
        raise ValueError(msg)
    return text


def _validate_git_commit(value: object) -> str:
    """Validate an abbreviated or full lowercase hexadecimal Git commit.

    Returns:
        The validated commit identifier.

    Raises:
        ValueError: If the commit identifier has an invalid format.
    """
    text = _validate_string(value, "git_commit")
    if _GIT_COMMIT_PATTERN.fullmatch(text) is None:
        msg = "git_commit must be a complete 40- or 64-character lowercase hexadecimal object ID."
        raise ValueError(msg)
    return text


def _validate_relative_path(value: object, name: str, *, allow_none: bool = False) -> str | None:
    """Validate a portable relative artifact path.

    Returns:
        The validated relative path, or ``None`` when allowed.

    Raises:
        ValueError: If the path is absolute, unnormalized, or contains traversal.
    """
    if value is None and allow_none:
        return None
    text = _validate_string(value, name)
    path = PurePosixPath(text)
    if path.is_absolute():
        msg = f"{name} must be relative."
        raise ValueError(msg)
    if "\\" in text or any(part in {"", ".", ".."} for part in text.split("/")):
        msg = f"{name} must be a normalized relative POSIX path without traversal."
        raise ValueError(msg)
    return text


def _freeze_json_value(value: object, name: str) -> object:
    """Validate and recursively freeze a JSON-native value.

    Returns:
        The recursively frozen JSON-native value.

    Raises:
        TypeError: If a value or mapping key has an unsupported type.
        ValueError: If the value contains a non-finite float.
    """
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            msg = f"{name} contains a non-finite float."
            raise ValueError(msg)
        return value or 0.0
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                _raise_type_error(f"{name} key", "a string", key)
            normalized[key] = _freeze_json_value(item, f"{name}.{key}")
        return MappingProxyType(dict(sorted(normalized.items())))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item, f"{name}[{index}]") for index, item in enumerate(value))
    msg = f"{name} contains a non-JSON-native value of type {type(value).__name__}."
    raise TypeError(msg)


def _freeze_json_mapping(value: object, name: str) -> Mapping[str, object]:
    """Validate and recursively freeze a string-keyed JSON object.

    Returns:
        The recursively frozen mapping.
    """
    if not isinstance(value, Mapping):
        _raise_type_error(name, "a mapping", value)
    frozen = _freeze_json_value(value, name)
    return cast("Mapping[str, object]", frozen)


def _thaw_json_value(value: object) -> object:
    """Convert an internal immutable JSON value to mutable JSON-native containers.

    Returns:
        The detached JSON-native value.
    """
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw_json_value(item) for item in value]
    return value


def _thaw_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return a detached JSON-native dictionary."""
    return cast("dict[str, object]", _thaw_json_value(value))


def _canonical_json(value: object) -> str:
    """Serialize a JSON-native value deterministically.

    Returns:
        The canonical JSON string.
    """
    normalized = _thaw_json_value(_freeze_json_value(value, "JSON payload"))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _strict_json_equal(left: object, right: object) -> bool:
    """Return whether JSON-native values have equal values and exact scalar types."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_mapping = cast("Mapping[str, object]", left)
        right_mapping = cast("Mapping[str, object]", right)
        if left_mapping.keys() != right_mapping.keys():
            return False
        return all(_strict_json_equal(left_mapping[key], right_mapping[key]) for key in left_mapping)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        left_sequence = cast("Sequence[object]", left)
        right_sequence = cast("Sequence[object]", right)
        if len(left_sequence) != len(right_sequence):
            return False
        return all(starmap(_strict_json_equal, zip(left_sequence, right_sequence, strict=True)))
    return left == right


def _reject_json_constant(value: str) -> object:
    """Reject nonstandard JSON constants such as NaN and Infinity.

    Raises:
        ValueError: Always, because the supplied constant is unsupported.
    """
    msg = f"Nonstandard JSON constant {value!r} is not supported."
    raise ValueError(msg)


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys.

    Returns:
        The decoded JSON object.

    Raises:
        ValueError: If a key occurs more than once.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"Duplicate JSON key {key!r}."
            raise ValueError(msg)
        result[key] = value
    return result


def _load_json_object(payload: str) -> Mapping[str, object]:
    """Decode a strict JSON object.

    Returns:
        The decoded top-level mapping.

    Raises:
        TypeError: If the decoded top-level value is not an object.
    """
    if type(payload) is not str:
        _raise_type_error("payload", "a string", payload)
    decoded = json.loads(
        payload,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(decoded, Mapping):
        msg = "The JSON payload must contain an object at the top level."
        raise TypeError(msg)
    return cast("Mapping[str, object]", decoded)


def _validate_exact_keys(data: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    """Reject missing and unknown fields in one schema object.

    Raises:
        ValueError: If required fields are missing or unknown fields are present.
    """
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {missing}")
        if unknown:
            details.append(f"unknown fields: {unknown}")
        msg = f"Invalid {name}: {'; '.join(details)}."
        raise ValueError(msg)


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    """Return a runtime-validated mapping."""
    if not isinstance(value, Mapping):
        _raise_type_error(name, "a mapping", value)
    return cast("Mapping[str, object]", value)


def _validate_optional_pair(
    path: object,
    checksum: object,
    *,
    path_name: str,
    checksum_name: str,
) -> tuple[str | None, str | None]:
    """Validate that an artifact path and checksum are present together.

    Returns:
        The normalized path and checksum pair.

    Raises:
        ValueError: If exactly one member of the pair is present.
    """
    normalized_path = _validate_relative_path(path, path_name, allow_none=True)
    normalized_checksum = _validate_checksum(checksum, checksum_name, allow_none=True)
    if (normalized_path is None) != (normalized_checksum is None):
        msg = f"{path_name} and {checksum_name} must either both be present or both be absent."
        raise ValueError(msg)
    return normalized_path, normalized_checksum


def _validate_software_versions(value: object) -> Mapping[str, object]:
    """Validate required software-version provenance and freeze it.

    Returns:
        The validated immutable software-version mapping.

    Raises:
        ValueError: If a required version is missing or a value is invalid.
    """
    versions = _freeze_json_mapping(value, "software_versions")
    missing = sorted(_REQUIRED_SOFTWARE_VERSIONS - set(versions))
    if missing:
        msg = f"software_versions is missing required keys: {missing}."
        raise ValueError(msg)
    for key, version in versions.items():
        _validate_slug(key, f"software_versions key {key!r}")
        _validate_string(version, f"software_versions[{key!r}]")
    return versions


def _freeze_gate_counts(value: object, name: str) -> Mapping[str, object]:
    """Validate and freeze a gate-name-to-count mapping.

    Returns:
        The validated immutable gate-count mapping.
    """
    counts = _freeze_json_mapping(value, name)
    for gate_name, count in counts.items():
        _validate_slug(gate_name, f"{name} key {gate_name!r}")
        _validate_count(count, f"{name}[{gate_name!r}]")
    return counts


@dataclass(frozen=True, slots=True)
class TargetSelection:
    """One resolved target-state fixture entry."""

    num_qubits: int
    target_id: str
    target_seed: int | None
    fixture_format: str
    fixture_checksum: str

    def __post_init__(self) -> None:
        """Validate the resolved target selection.

        Raises:
            ValueError: If a target field is unsupported or invalid.
        """
        if type(self.num_qubits) is not int:
            _raise_type_error("num_qubits", "an int", self.num_qubits)
        if self.num_qubits not in SUPPORTED_QUBIT_COUNTS:
            msg = f"num_qubits must be one of {SUPPORTED_QUBIT_COUNTS}."
            raise ValueError(msg)
        if type(self.target_id) is not str:
            _raise_type_error("target_id", "a string", self.target_id)
        if self.target_id not in TARGET_IDS:
            msg = f"Unknown target_id {self.target_id!r}."
            raise ValueError(msg)
        _validate_seed(self.target_seed, "target_seed", allow_none=True)
        expected_seed = TARGET_GENERATION_SEEDS[self.target_id]
        if self.target_seed != expected_seed:
            msg = f"target_seed for {self.target_id!r} must be {expected_seed!r}."
            raise ValueError(msg)
        if type(self.fixture_format) is not str:
            _raise_type_error("fixture_format", "a string", self.fixture_format)
        if self.fixture_format != TARGET_FIXTURE_FORMAT:
            msg = f"fixture_format must be {TARGET_FIXTURE_FORMAT!r}."
            raise ValueError(msg)
        _validate_checksum(self.fixture_checksum, "fixture_checksum")

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native representation."""
        return {
            "num_qubits": self.num_qubits,
            "target_id": self.target_id,
            "target_seed": self.target_seed,
            "fixture_format": self.fixture_format,
            "fixture_checksum": self.fixture_checksum,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TargetSelection:
        """Construct a target selection from a strict dictionary.

        Returns:
            The validated target selection.
        """
        expected = frozenset({"num_qubits", "target_id", "target_seed", "fixture_format", "fixture_checksum"})
        _validate_exact_keys(data, expected, "TargetSelection")
        return cls(
            num_qubits=cast("int", data["num_qubits"]),
            target_id=cast("str", data["target_id"]),
            target_seed=cast("int | None", data["target_seed"]),
            fixture_format=cast("str", data["fixture_format"]),
            fixture_checksum=cast("str", data["fixture_checksum"]),
        )


@dataclass(frozen=True, slots=True)
class AnsatzConfig:
    """Shared brickwall ansatz configuration."""

    configured_bmpd_depth: int
    initial_single_qubit_layer: bool = True
    ansatz_id: str = ANSATZ_ID

    def __post_init__(self) -> None:
        """Validate the ansatz configuration.

        Raises:
            ValueError: If the ansatz identifier or depth is invalid.
        """
        _validate_count(self.configured_bmpd_depth, "configured_bmpd_depth")
        _validate_bool(self.initial_single_qubit_layer, "initial_single_qubit_layer")
        if type(self.ansatz_id) is not str:
            _raise_type_error("ansatz_id", "a string", self.ansatz_id)
        if self.ansatz_id != ANSATZ_ID:
            msg = f"ansatz_id must be {ANSATZ_ID!r}."
            raise ValueError(msg)

    @property
    def num_layers(self) -> int:
        """Number of brickwall layers generated by the BMPD depth."""
        return 2 * self.configured_bmpd_depth

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native representation."""
        return {
            "ansatz_id": self.ansatz_id,
            "configured_bmpd_depth": self.configured_bmpd_depth,
            "num_layers": self.num_layers,
            "initial_single_qubit_layer": self.initial_single_qubit_layer,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AnsatzConfig:
        """Construct an ansatz configuration from a strict dictionary.

        Returns:
            The validated ansatz configuration.

        Raises:
            ValueError: If a serialized derived layer count is inconsistent.
        """
        expected = frozenset({"ansatz_id", "configured_bmpd_depth", "num_layers", "initial_single_qubit_layer"})
        _validate_exact_keys(data, expected, "AnsatzConfig")
        result = cls(
            ansatz_id=cast("str", data["ansatz_id"]),
            configured_bmpd_depth=cast("int", data["configured_bmpd_depth"]),
            initial_single_qubit_layer=cast("bool", data["initial_single_qubit_layer"]),
        )
        if not _strict_json_equal(data["num_layers"], result.num_layers):
            msg = "num_layers must equal twice configured_bmpd_depth."
            raise ValueError(msg)
        return result


@dataclass(frozen=True, slots=True)
class InitializationConfig:
    """Parameter-initialization rule and reproducibility metadata."""

    rule: str
    seed: int | None = None
    scale: float | None = None
    warm_start_path: str | None = None
    warm_start_checksum: str | None = None

    def __post_init__(self) -> None:
        """Validate rule-specific initialization fields.

        Raises:
            ValueError: If the initialization fields are inconsistent with the rule.
        """
        if type(self.rule) is not str:
            _raise_type_error("rule", "a string", self.rule)
        if self.rule not in INITIALIZATION_RULES:
            msg = f"rule must be one of {INITIALIZATION_RULES}."
            raise ValueError(msg)
        _validate_seed(self.seed, "seed", allow_none=True)
        _validate_optional_float(self.scale, "scale", minimum=0.0, minimum_inclusive=False)
        path, checksum = _validate_optional_pair(
            self.warm_start_path,
            self.warm_start_checksum,
            path_name="warm_start_path",
            checksum_name="warm_start_checksum",
        )
        object.__setattr__(self, "warm_start_path", path)
        object.__setattr__(self, "warm_start_checksum", checksum)

        random_rule = self.rule in {"random_uniform", "random_normal"}
        if random_rule and (self.seed is None or self.scale is None):
            msg = "Random initialization requires both seed and positive scale."
            raise ValueError(msg)
        if random_rule and path is not None:
            msg = "Random initialization cannot specify a warm start."
            raise ValueError(msg)
        if self.rule == "warm_start" and path is None:
            msg = "Warm-start initialization requires a path and checksum."
            raise ValueError(msg)
        if self.rule in {"zeros", "warm_start"} and (self.seed is not None or self.scale is not None):
            msg = f"{self.rule!r} initialization cannot specify seed or scale."
            raise ValueError(msg)
        if self.rule == "zeros" and path is not None:
            msg = "Zero initialization cannot specify a warm start."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native representation."""
        return {
            "rule": self.rule,
            "seed": self.seed,
            "scale": self.scale,
            "warm_start_path": self.warm_start_path,
            "warm_start_checksum": self.warm_start_checksum,
        }

    def identity_dict(self) -> dict[str, object]:
        """Return identity-bearing fields, excluding the warm-start path spelling."""
        return {
            "rule": self.rule,
            "seed": self.seed,
            "scale": self.scale,
            "warm_start_checksum": self.warm_start_checksum,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> InitializationConfig:
        """Construct initialization metadata from a strict dictionary.

        Returns:
            The validated initialization configuration.
        """
        expected = frozenset({"rule", "seed", "scale", "warm_start_path", "warm_start_checksum"})
        _validate_exact_keys(data, expected, "InitializationConfig")
        return cls(
            rule=cast("str", data["rule"]),
            seed=cast("int | None", data["seed"]),
            scale=_json_optional_float(data["scale"], "scale"),
            warm_start_path=cast("str | None", data["warm_start_path"]),
            warm_start_checksum=cast("str | None", data["warm_start_checksum"]),
        )


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Optimizer budget, random stream, truncation, and implementation parameters."""

    optimizer_id: str
    max_iterations: int
    optimizer_seed: int
    hyperparameters: Mapping[str, object] = field(default_factory=dict)
    train_trajectories_or_shots: int = 0
    training_seed: int | None = None
    max_bond_dimension: int | None = None
    svd_threshold: float = 0.0
    truncation_mode: str = "discarded_weight"
    min_bond_dimension: int = 1

    def __post_init__(self) -> None:
        """Validate and freeze optimizer configuration.

        Raises:
            ValueError: If an optimizer, budget, or truncation field is invalid.
        """
        _validate_slug(self.optimizer_id, "optimizer_id")
        _validate_count(self.max_iterations, "max_iterations")
        _validate_seed(self.optimizer_seed, "optimizer_seed")
        object.__setattr__(self, "hyperparameters", _freeze_json_mapping(self.hyperparameters, "hyperparameters"))
        _validate_count(self.train_trajectories_or_shots, "train_trajectories_or_shots")
        _validate_seed(self.training_seed, "training_seed", allow_none=True)
        if self.max_bond_dimension is not None:
            _validate_count(self.max_bond_dimension, "max_bond_dimension", minimum=1)
        _validate_float(self.svd_threshold, "svd_threshold", minimum=0.0)
        if type(self.truncation_mode) is not str:
            _raise_type_error("truncation_mode", "a string", self.truncation_mode)
        if self.truncation_mode not in TRUNCATION_MODES:
            msg = f"truncation_mode must be one of {TRUNCATION_MODES}."
            raise ValueError(msg)
        _validate_count(self.min_bond_dimension, "min_bond_dimension", minimum=1)
        if self.max_bond_dimension is not None and self.min_bond_dimension > self.max_bond_dimension:
            msg = "min_bond_dimension cannot exceed max_bond_dimension."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native representation."""
        return {
            "optimizer_id": self.optimizer_id,
            "max_iterations": self.max_iterations,
            "optimizer_seed": self.optimizer_seed,
            "hyperparameters": _thaw_json_mapping(self.hyperparameters),
            "train_trajectories_or_shots": self.train_trajectories_or_shots,
            "training_seed": self.training_seed,
            "max_bond_dimension": self.max_bond_dimension,
            "svd_threshold": self.svd_threshold,
            "truncation_mode": self.truncation_mode,
            "min_bond_dimension": self.min_bond_dimension,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> OptimizerConfig:
        """Construct optimizer configuration from a strict dictionary.

        Returns:
            The validated optimizer configuration.
        """
        expected = frozenset({
            "optimizer_id",
            "max_iterations",
            "optimizer_seed",
            "hyperparameters",
            "train_trajectories_or_shots",
            "training_seed",
            "max_bond_dimension",
            "svd_threshold",
            "truncation_mode",
            "min_bond_dimension",
        })
        _validate_exact_keys(data, expected, "OptimizerConfig")
        return cls(
            optimizer_id=cast("str", data["optimizer_id"]),
            max_iterations=cast("int", data["max_iterations"]),
            optimizer_seed=cast("int", data["optimizer_seed"]),
            hyperparameters=_as_mapping(data["hyperparameters"], "hyperparameters"),
            train_trajectories_or_shots=cast("int", data["train_trajectories_or_shots"]),
            training_seed=cast("int | None", data["training_seed"]),
            max_bond_dimension=cast("int | None", data["max_bond_dimension"]),
            svd_threshold=_json_float(data["svd_threshold"], "svd_threshold"),
            truncation_mode=cast("str", data["truncation_mode"]),
            min_bond_dimension=cast("int", data["min_bond_dimension"]),
        )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Independent test-evaluation budget, seed, and truncation settings."""

    test_trajectories_or_shots: int
    test_seed: int | None
    max_bond_dimension: int | None = None
    svd_threshold: float = 0.0
    truncation_mode: str = "discarded_weight"
    min_bond_dimension: int = 1
    store_trajectory_sidecar: bool = False
    confidence_level: float | None = None
    confidence_interval_method: str | None = None

    def __post_init__(self) -> None:
        """Validate evaluation configuration.

        Raises:
            ValueError: If an evaluation budget or truncation field is invalid.
        """
        _validate_count(self.test_trajectories_or_shots, "test_trajectories_or_shots")
        _validate_seed(self.test_seed, "test_seed", allow_none=True)
        if self.max_bond_dimension is not None:
            _validate_count(self.max_bond_dimension, "max_bond_dimension", minimum=1)
        _validate_float(self.svd_threshold, "svd_threshold", minimum=0.0)
        if type(self.truncation_mode) is not str:
            _raise_type_error("truncation_mode", "a string", self.truncation_mode)
        if self.truncation_mode not in TRUNCATION_MODES:
            msg = f"truncation_mode must be one of {TRUNCATION_MODES}."
            raise ValueError(msg)
        _validate_count(self.min_bond_dimension, "min_bond_dimension", minimum=1)
        if self.max_bond_dimension is not None and self.min_bond_dimension > self.max_bond_dimension:
            msg = "min_bond_dimension cannot exceed max_bond_dimension."
            raise ValueError(msg)
        _validate_bool(self.store_trajectory_sidecar, "store_trajectory_sidecar")
        _validate_optional_float(
            self.confidence_level,
            "confidence_level",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
            maximum_inclusive=False,
        )
        if self.confidence_interval_method is not None:
            if type(self.confidence_interval_method) is not str:
                _raise_type_error("confidence_interval_method", "a string or None", self.confidence_interval_method)
            if self.confidence_interval_method not in CONFIDENCE_INTERVAL_METHODS:
                msg = f"confidence_interval_method must be one of {CONFIDENCE_INTERVAL_METHODS}."
                raise ValueError(msg)
        if (self.confidence_level is None) != (self.confidence_interval_method is None):
            msg = "confidence_level and confidence_interval_method must either both be set or both be omitted."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native representation."""
        return {
            "test_trajectories_or_shots": self.test_trajectories_or_shots,
            "test_seed": self.test_seed,
            "max_bond_dimension": self.max_bond_dimension,
            "svd_threshold": self.svd_threshold,
            "truncation_mode": self.truncation_mode,
            "min_bond_dimension": self.min_bond_dimension,
            "store_trajectory_sidecar": self.store_trajectory_sidecar,
            "confidence_level": self.confidence_level,
            "confidence_interval_method": self.confidence_interval_method,
        }

    def identity_dict(self) -> dict[str, object]:
        """Return identity-bearing evaluation fields, excluding output and CI policy."""
        return {
            key: value
            for key, value in self.to_dict().items()
            if key not in {"store_trajectory_sidecar", "confidence_level", "confidence_interval_method"}
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvaluationConfig:
        """Construct evaluation configuration from a strict dictionary.

        Returns:
            The validated evaluation configuration.
        """
        expected = frozenset({
            "test_trajectories_or_shots",
            "test_seed",
            "max_bond_dimension",
            "svd_threshold",
            "truncation_mode",
            "min_bond_dimension",
            "store_trajectory_sidecar",
            "confidence_level",
            "confidence_interval_method",
        })
        _validate_exact_keys(data, expected, "EvaluationConfig")
        return cls(
            test_trajectories_or_shots=cast("int", data["test_trajectories_or_shots"]),
            test_seed=cast("int | None", data["test_seed"]),
            max_bond_dimension=cast("int | None", data["max_bond_dimension"]),
            svd_threshold=_json_float(data["svd_threshold"], "svd_threshold"),
            truncation_mode=cast("str", data["truncation_mode"]),
            min_bond_dimension=cast("int", data["min_bond_dimension"]),
            store_trajectory_sidecar=cast("bool", data["store_trajectory_sidecar"]),
            confidence_level=_json_optional_float(data["confidence_level"], "confidence_level"),
            confidence_interval_method=cast("str | None", data["confidence_interval_method"]),
        )


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """One versioned benchmark noise configuration."""

    noise_id: str
    tjm_dt: float | None = None
    definition_version: str = NOISE_DEFINITION_VERSION

    def __post_init__(self) -> None:
        """Validate identifier-specific noise configuration.

        Raises:
            ValueError: If the noise identifier, version, or time step is invalid.
        """
        if type(self.noise_id) is not str:
            _raise_type_error("noise_id", "a string", self.noise_id)
        if self.noise_id not in NOISE_IDS:
            msg = f"Unknown noise_id {self.noise_id!r}."
            raise ValueError(msg)
        if self.definition_version != NOISE_DEFINITION_VERSION:
            msg = f"definition_version must be {NOISE_DEFINITION_VERSION!r}."
            raise ValueError(msg)
        dt = _validate_optional_float(self.tjm_dt, "tjm_dt", minimum=0.0, minimum_inclusive=False)
        if self.noise_id in STANDARD_NOISE_IDS and dt is None:
            msg = "Standard noise configurations require an explicit positive tjm_dt."
            raise ValueError(msg)
        if self.noise_id in {NOISELESS_NOISE_ID, BALLARIN_NOISE_ID} and dt is not None:
            msg = f"{self.noise_id!r} does not use a TJM time step."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native representation."""
        return {
            "noise_id": self.noise_id,
            "tjm_dt": self.tjm_dt,
            "definition_version": self.definition_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> NoiseConfig:
        """Construct noise configuration from a strict dictionary.

        Returns:
            The validated noise configuration.
        """
        expected = frozenset({"noise_id", "tjm_dt", "definition_version"})
        _validate_exact_keys(data, expected, "NoiseConfig")
        return cls(
            noise_id=cast("str", data["noise_id"]),
            tjm_dt=_json_optional_float(data["tjm_dt"], "tjm_dt"),
            definition_version=cast("str", data["definition_version"]),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """One fully resolved state-preparation benchmark result cell."""

    method_id: str
    method_version: str
    target: TargetSelection
    ansatz: AnsatzConfig
    initialization: InitializationConfig
    optimizer: OptimizerConfig
    evaluation: EvaluationConfig
    training_noise: NoiseConfig
    test_noise: NoiseConfig
    schema_version: str = CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the resolved benchmark configuration and cross-field rules.

        Raises:
            ValueError: If a schema or train/test noise invariant is violated.
        """
        _validate_slug(self.method_id, "method_id")
        _validate_string(self.method_version, "method_version")
        if not isinstance(self.target, TargetSelection):
            _raise_type_error("target", "a TargetSelection", self.target)
        if not isinstance(self.ansatz, AnsatzConfig):
            _raise_type_error("ansatz", "an AnsatzConfig", self.ansatz)
        if not isinstance(self.initialization, InitializationConfig):
            _raise_type_error("initialization", "an InitializationConfig", self.initialization)
        if not isinstance(self.optimizer, OptimizerConfig):
            _raise_type_error("optimizer", "an OptimizerConfig", self.optimizer)
        if not isinstance(self.evaluation, EvaluationConfig):
            _raise_type_error("evaluation", "an EvaluationConfig", self.evaluation)
        if not isinstance(self.training_noise, NoiseConfig):
            _raise_type_error("training_noise", "a NoiseConfig", self.training_noise)
        if not isinstance(self.test_noise, NoiseConfig):
            _raise_type_error("test_noise", "a NoiseConfig", self.test_noise)
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            msg = f"schema_version must be {CONFIG_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        if self.training_noise.noise_id != NOISELESS_NOISE_ID:
            msg = "The v1 benchmark requires noiseless optimization."
            raise ValueError(msg)
        if self.optimizer.train_trajectories_or_shots != 0 or self.optimizer.training_seed is not None:
            msg = "Noiseless optimization requires zero training trajectories or shots and no training seed."
            raise ValueError(msg)

        noiseless_test = self.test_noise.noise_id == NOISELESS_NOISE_ID
        if noiseless_test and (
            self.evaluation.test_trajectories_or_shots != 0 or self.evaluation.test_seed is not None
        ):
            msg = "Noiseless evaluation requires zero trajectories or shots and no test seed."
            raise ValueError(msg)
        if not noiseless_test and (
            self.evaluation.test_trajectories_or_shots == 0 or self.evaluation.test_seed is None
        ):
            msg = "Noisy evaluation requires a positive trajectory or shot count and a test seed."
            raise ValueError(msg)
        if noiseless_test and self.evaluation.confidence_level is not None:
            msg = "Noiseless evaluation cannot request a confidence interval."
            raise ValueError(msg)
        if noiseless_test and self.evaluation.store_trajectory_sidecar:
            msg = "Noiseless evaluation cannot store a trajectory sidecar."
            raise ValueError(msg)
        if self.evaluation.confidence_level is not None and self.evaluation.test_trajectories_or_shots < 2:
            msg = "Confidence intervals require at least two trajectories or shots."
            raise ValueError(msg)

        runtime_seeds = tuple(
            seed
            for seed in (
                self.initialization.seed,
                self.optimizer.optimizer_seed,
                self.optimizer.training_seed,
                self.evaluation.test_seed,
            )
            if seed is not None
        )
        if len(runtime_seeds) != len(set(runtime_seeds)):
            msg = "Initialization, optimizer, training, and test random streams require distinct resolved seeds."
            raise ValueError(msg)

    def identity_payload(self) -> dict[str, object]:
        """Return the scientific run identity, excluding output and result metadata."""
        return {
            "identity_version": RUN_IDENTITY_VERSION,
            "config_schema_version": self.schema_version,
            "method_id": self.method_id,
            "method_version": self.method_version,
            "target": self.target.to_dict(),
            "ansatz": self.ansatz.to_dict(),
            "initialization": self.initialization.identity_dict(),
            "optimizer": self.optimizer.to_dict(),
            "evaluation": self.evaluation.identity_dict(),
            "training_noise": self.training_noise.to_dict(),
            "test_noise": self.test_noise.to_dict(),
        }

    @property
    def run_id(self) -> str:
        """Stable SHA-256 identifier for this resolved run."""
        digest = hashlib.sha256(_canonical_json(self.identity_payload()).encode()).hexdigest()
        return f"{RUN_ID_PREFIX}{digest}"

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native representation."""
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "method_version": self.method_version,
            "target": self.target.to_dict(),
            "ansatz": self.ansatz.to_dict(),
            "initialization": self.initialization.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "training_noise": self.training_noise.to_dict(),
            "test_noise": self.test_noise.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BenchmarkConfig:
        """Construct a resolved benchmark configuration from a strict dictionary.

        Returns:
            The validated benchmark configuration.
        """
        expected = frozenset({
            "schema_version",
            "method_id",
            "method_version",
            "target",
            "ansatz",
            "initialization",
            "optimizer",
            "evaluation",
            "training_noise",
            "test_noise",
        })
        _validate_exact_keys(data, expected, "BenchmarkConfig")
        return cls(
            schema_version=cast("str", data["schema_version"]),
            method_id=cast("str", data["method_id"]),
            method_version=cast("str", data["method_version"]),
            target=TargetSelection.from_dict(_as_mapping(data["target"], "target")),
            ansatz=AnsatzConfig.from_dict(_as_mapping(data["ansatz"], "ansatz")),
            initialization=InitializationConfig.from_dict(_as_mapping(data["initialization"], "initialization")),
            optimizer=OptimizerConfig.from_dict(_as_mapping(data["optimizer"], "optimizer")),
            evaluation=EvaluationConfig.from_dict(_as_mapping(data["evaluation"], "evaluation")),
            training_noise=NoiseConfig.from_dict(_as_mapping(data["training_noise"], "training_noise")),
            test_noise=NoiseConfig.from_dict(_as_mapping(data["test_noise"], "test_noise")),
        )

    def to_json(self) -> str:
        """Serialize the configuration to deterministic JSON.

        Returns:
            The canonical JSON representation.
        """
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> BenchmarkConfig:
        """Deserialize a strict JSON configuration.

        Returns:
            The validated benchmark configuration.
        """
        return cls.from_dict(_load_json_object(payload))


@dataclass(frozen=True, slots=True)
class CircuitStatistics:
    """Logical, native, and actually evaluated final-circuit statistics."""

    configured_bmpd_depth: int
    num_parameters: int
    logical_depth: int
    logical_num_1q_gates: int
    logical_num_2q_gates: int
    native_depth: int
    native_num_1q_gates: int
    native_num_2q_gates: int
    native_rzz_count: int
    pruned_native_rzz_count: int
    evaluated_representation: str
    logical_gate_counts: Mapping[str, object] = field(default_factory=dict)
    native_gate_counts: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze circuit statistics.

        Raises:
            ValueError: If a count or evaluated representation is inconsistent.
        """
        for field_name in (
            "configured_bmpd_depth",
            "num_parameters",
            "logical_depth",
            "logical_num_1q_gates",
            "logical_num_2q_gates",
            "native_depth",
            "native_num_1q_gates",
            "native_num_2q_gates",
            "native_rzz_count",
            "pruned_native_rzz_count",
        ):
            _validate_count(getattr(self, field_name), field_name)
        if type(self.evaluated_representation) is not str:
            _raise_type_error("evaluated_representation", "a string", self.evaluated_representation)
        if self.evaluated_representation not in EVALUATED_REPRESENTATIONS:
            msg = f"evaluated_representation must be one of {EVALUATED_REPRESENTATIONS}."
            raise ValueError(msg)
        if self.native_rzz_count != self.native_num_2q_gates:
            msg = "native_rzz_count must equal native_num_2q_gates because every native two-qubit gate is RZZ."
            raise ValueError(msg)
        if self.native_rzz_count + self.pruned_native_rzz_count != self.logical_num_2q_gates:
            msg = "Retained and pruned native RZZ counts must account for every logical two-qubit gate."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "logical_gate_counts",
            _freeze_gate_counts(self.logical_gate_counts, "logical_gate_counts"),
        )
        object.__setattr__(
            self,
            "native_gate_counts",
            _freeze_gate_counts(self.native_gate_counts, "native_gate_counts"),
        )
        native_rzz_detail = self.native_gate_counts.get("rzz")
        if native_rzz_detail is not None and native_rzz_detail != self.native_rzz_count:
            msg = "native_gate_counts['rzz'] must equal native_rzz_count."
            raise ValueError(msg)
        detailed_totals = (
            ("logical_gate_counts", self.logical_gate_counts, self.logical_num_1q_gates + self.logical_num_2q_gates),
            ("native_gate_counts", self.native_gate_counts, self.native_num_1q_gates + self.native_num_2q_gates),
        )
        for name, counts, expected_total in detailed_totals:
            if counts and sum(cast("int", count) for count in counts.values()) != expected_total:
                msg = f"{name} must sum to the corresponding aggregate gate counts."
                raise ValueError(msg)

    @property
    def num_layers(self) -> int:
        """Reported brickwall layer count."""
        return 2 * self.configured_bmpd_depth

    @property
    def evaluated_depth(self) -> int:
        """Depth of the circuit used for this row's evaluation."""
        return self.logical_depth if self.evaluated_representation == "logical" else self.native_depth

    @property
    def num_1q_gates(self) -> int:
        """Reported one-qubit count of the evaluated circuit."""
        return self.logical_num_1q_gates if self.evaluated_representation == "logical" else self.native_num_1q_gates

    @property
    def num_2q_gates(self) -> int:
        """Reported two-qubit count of the evaluated circuit."""
        return self.logical_num_2q_gates if self.evaluated_representation == "logical" else self.native_num_2q_gates

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native representation including reporting aliases."""
        return {
            "configured_bmpd_depth": self.configured_bmpd_depth,
            "num_layers": self.num_layers,
            "num_parameters": self.num_parameters,
            "logical_depth": self.logical_depth,
            "logical_num_1q_gates": self.logical_num_1q_gates,
            "logical_num_2q_gates": self.logical_num_2q_gates,
            "native_depth": self.native_depth,
            "native_num_1q_gates": self.native_num_1q_gates,
            "native_num_2q_gates": self.native_num_2q_gates,
            "native_rzz_count": self.native_rzz_count,
            "pruned_native_rzz_count": self.pruned_native_rzz_count,
            "evaluated_representation": self.evaluated_representation,
            "evaluated_depth": self.evaluated_depth,
            "num_1q_gates": self.num_1q_gates,
            "num_2q_gates": self.num_2q_gates,
            "logical_gate_counts": _thaw_json_mapping(self.logical_gate_counts),
            "native_gate_counts": _thaw_json_mapping(self.native_gate_counts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CircuitStatistics:
        """Construct circuit statistics and verify all derived reporting aliases.

        Returns:
            The validated circuit statistics.

        Raises:
            ValueError: If a serialized reporting alias is inconsistent.
        """
        expected = frozenset({
            "configured_bmpd_depth",
            "num_layers",
            "num_parameters",
            "logical_depth",
            "logical_num_1q_gates",
            "logical_num_2q_gates",
            "native_depth",
            "native_num_1q_gates",
            "native_num_2q_gates",
            "native_rzz_count",
            "pruned_native_rzz_count",
            "evaluated_representation",
            "evaluated_depth",
            "num_1q_gates",
            "num_2q_gates",
            "logical_gate_counts",
            "native_gate_counts",
        })
        _validate_exact_keys(data, expected, "CircuitStatistics")
        result = cls(
            configured_bmpd_depth=cast("int", data["configured_bmpd_depth"]),
            num_parameters=cast("int", data["num_parameters"]),
            logical_depth=cast("int", data["logical_depth"]),
            logical_num_1q_gates=cast("int", data["logical_num_1q_gates"]),
            logical_num_2q_gates=cast("int", data["logical_num_2q_gates"]),
            native_depth=cast("int", data["native_depth"]),
            native_num_1q_gates=cast("int", data["native_num_1q_gates"]),
            native_num_2q_gates=cast("int", data["native_num_2q_gates"]),
            native_rzz_count=cast("int", data["native_rzz_count"]),
            pruned_native_rzz_count=cast("int", data["pruned_native_rzz_count"]),
            evaluated_representation=cast("str", data["evaluated_representation"]),
            logical_gate_counts=_as_mapping(data["logical_gate_counts"], "logical_gate_counts"),
            native_gate_counts=_as_mapping(data["native_gate_counts"], "native_gate_counts"),
        )
        aliases = {
            "num_layers": result.num_layers,
            "evaluated_depth": result.evaluated_depth,
            "num_1q_gates": result.num_1q_gates,
            "num_2q_gates": result.num_2q_gates,
        }
        for name, expected_value in aliases.items():
            if not _strict_json_equal(data[name], expected_value):
                msg = f"{name} does not match the derived circuit statistic."
                raise ValueError(msg)
        return result


def _reporting_ansatz(config: BenchmarkConfig) -> str:
    """Return a stable human-readable ansatz description."""
    initial_layer = "true" if config.ansatz.initial_single_qubit_layer else "false"
    return (
        f"{config.ansatz.ansatz_id}"
        f"(configured_bmpd_depth={config.ansatz.configured_bmpd_depth},"
        f"num_layers={config.ansatz.num_layers},"
        f"initial_single_qubit_layer={initial_layer})"
    )


def _reporting_optimizer_budget(config: BenchmarkConfig) -> dict[str, object]:
    """Return the structured optimizer budget used by the reporting schema."""
    return {
        "optimizer_id": config.optimizer.optimizer_id,
        "max_iterations": config.optimizer.max_iterations,
        "hyperparameters": _thaw_json_mapping(config.optimizer.hyperparameters),
    }


def _validate_result_config_statistics(config: BenchmarkConfig, statistics: CircuitStatistics) -> None:
    """Validate cross-record circuit and noise representation invariants.

    Raises:
        ValueError: If circuit statistics conflict with the benchmark configuration.
    """
    if statistics.configured_bmpd_depth != config.ansatz.configured_bmpd_depth:
        msg = "Circuit statistics and ansatz configuration use different BMPD depths."
        raise ValueError(msg)
    num_bmpd_blocks = config.ansatz.configured_bmpd_depth * (config.target.num_qubits - 1)
    initial_parameters = 3 * config.target.num_qubits if config.ansatz.initial_single_qubit_layer else 0
    expected_num_parameters = 9 * num_bmpd_blocks + initial_parameters
    if statistics.num_parameters != expected_num_parameters:
        msg = "Circuit statistics contain a parameter count inconsistent with the configured BMPD ansatz."
        raise ValueError(msg)
    expected_representation = "native" if config.test_noise.noise_id == BALLARIN_NOISE_ID else "logical"
    if statistics.evaluated_representation != expected_representation:
        msg = (
            f"Noise configuration {config.test_noise.noise_id!r} requires "
            f"{expected_representation!r} circuit statistics."
        )
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Successful benchmark result with complete metrics and provenance."""

    config: BenchmarkConfig
    circuit_statistics: CircuitStatistics
    train_fidelity: float
    logical_test_noiseless_fidelity: float
    test_noiseless_fidelity: float
    test_noisy_fidelity: float
    software_versions: Mapping[str, object]
    git_commit: str
    git_dirty: bool
    parameter_checkpoint_path: str
    parameter_checkpoint_checksum: str
    native_pre_pruning_noiseless_fidelity: float | None = None
    noisy_fidelity_standard_deviation: float | None = None
    noisy_fidelity_standard_error: float | None = None
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None
    sampled_nonidentity_events: int = 0
    optimization_wall_time_seconds: float = 0.0
    evaluation_wall_time_seconds: float = 0.0
    git_diff_checksum: str | None = None
    trajectory_sidecar_path: str | None = None
    trajectory_sidecar_checksum: str | None = None
    notes: str = ""
    schema_version: str = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate metrics, uncertainty, artifacts, and provenance.

        Raises:
            ValueError: If a metric, artifact, or provenance invariant is violated.
        """
        if not isinstance(self.config, BenchmarkConfig):
            _raise_type_error("config", "a BenchmarkConfig", self.config)
        if not isinstance(self.circuit_statistics, CircuitStatistics):
            _raise_type_error("circuit_statistics", "CircuitStatistics", self.circuit_statistics)
        _validate_result_config_statistics(self.config, self.circuit_statistics)
        for field_name in (
            "train_fidelity",
            "logical_test_noiseless_fidelity",
            "test_noiseless_fidelity",
            "test_noisy_fidelity",
        ):
            _validate_fidelity(getattr(self, field_name), field_name)
        pre_pruning_fidelity = (
            None
            if self.native_pre_pruning_noiseless_fidelity is None
            else _validate_fidelity(
                self.native_pre_pruning_noiseless_fidelity,
                "native_pre_pruning_noiseless_fidelity",
            )
        )
        standard_deviation = _validate_optional_float(
            self.noisy_fidelity_standard_deviation,
            "noisy_fidelity_standard_deviation",
            minimum=0.0,
        )
        standard_error = _validate_optional_float(
            self.noisy_fidelity_standard_error,
            "noisy_fidelity_standard_error",
            minimum=0.0,
        )
        lower = _validate_optional_float(
            self.confidence_interval_lower,
            "confidence_interval_lower",
            minimum=0.0,
            maximum=1.0,
        )
        upper = _validate_optional_float(
            self.confidence_interval_upper,
            "confidence_interval_upper",
            minimum=0.0,
            maximum=1.0,
        )
        _validate_count(self.sampled_nonidentity_events, "sampled_nonidentity_events")
        _validate_float(self.optimization_wall_time_seconds, "optimization_wall_time_seconds", minimum=0.0)
        _validate_float(self.evaluation_wall_time_seconds, "evaluation_wall_time_seconds", minimum=0.0)
        object.__setattr__(self, "software_versions", _validate_software_versions(self.software_versions))
        _validate_git_commit(self.git_commit)
        _validate_bool(self.git_dirty, "git_dirty")
        diff_checksum = _validate_checksum(self.git_diff_checksum, "git_diff_checksum", allow_none=True)
        if self.git_dirty != (diff_checksum is not None):
            msg = "git_diff_checksum is required exactly when git_dirty is true."
            raise ValueError(msg)
        checkpoint_path = cast(
            "str",
            _validate_relative_path(self.parameter_checkpoint_path, "parameter_checkpoint_path"),
        )
        checkpoint_checksum = cast(
            "str",
            _validate_checksum(self.parameter_checkpoint_checksum, "parameter_checkpoint_checksum"),
        )
        sidecar_path, sidecar_checksum = _validate_optional_pair(
            self.trajectory_sidecar_path,
            self.trajectory_sidecar_checksum,
            path_name="trajectory_sidecar_path",
            checksum_name="trajectory_sidecar_checksum",
        )
        object.__setattr__(self, "git_diff_checksum", diff_checksum)
        object.__setattr__(self, "parameter_checkpoint_path", checkpoint_path)
        object.__setattr__(self, "parameter_checkpoint_checksum", checkpoint_checksum)
        object.__setattr__(self, "trajectory_sidecar_path", sidecar_path)
        object.__setattr__(self, "trajectory_sidecar_checksum", sidecar_checksum)
        if type(self.notes) is not str:
            _raise_type_error("notes", "a string", self.notes)
        if self.schema_version != RESULT_SCHEMA_VERSION:
            msg = f"schema_version must be {RESULT_SCHEMA_VERSION!r}."
            raise ValueError(msg)

        trajectory_count = self.config.evaluation.test_trajectories_or_shots
        noiseless = self.config.test_noise.noise_id == NOISELESS_NOISE_ID
        native = self.circuit_statistics.evaluated_representation == "native"
        if native != (pre_pruning_fidelity is not None):
            msg = "Native evaluation requires a pre-pruning noiseless fidelity; logical evaluation forbids it."
            raise ValueError(msg)
        if not native and self.logical_test_noiseless_fidelity != self.test_noiseless_fidelity:
            msg = "Logical evaluation must reuse the same logical noiseless fidelity."
            raise ValueError(msg)
        if noiseless:
            if self.test_noisy_fidelity != self.test_noiseless_fidelity:
                msg = "A noiseless result must report identical noisy and noiseless test fidelities."
                raise ValueError(msg)
            if any(value is not None for value in (standard_deviation, standard_error, lower, upper)):
                msg = "A noiseless result cannot report sampling uncertainty."
                raise ValueError(msg)
            if self.sampled_nonidentity_events != 0:
                msg = "A noiseless result cannot report non-identity noise events."
                raise ValueError(msg)
        elif trajectory_count == 1:
            if standard_deviation is not None or standard_error is not None:
                msg = "Sampling uncertainty is undefined for a single trajectory or shot."
                raise ValueError(msg)
        elif standard_deviation is None or standard_error is None:
            msg = "Noisy evaluation with at least two samples requires standard deviation and standard error."
            raise ValueError(msg)

        if standard_deviation is not None and standard_error is not None:
            expected_standard_error = standard_deviation / math.sqrt(trajectory_count)
            if not math.isclose(standard_error, expected_standard_error, rel_tol=1e-12, abs_tol=1e-15):
                msg = "noisy_fidelity_standard_error must equal standard deviation divided by sqrt(sample count)."
                raise ValueError(msg)
        if (lower is None) != (upper is None):
            msg = "Confidence-interval bounds must either both be present or both be absent."
            raise ValueError(msg)
        confidence_requested = self.config.evaluation.confidence_level is not None
        if confidence_requested != (lower is not None):
            msg = "Confidence-interval bounds must match the evaluation confidence-level setting."
            raise ValueError(msg)
        if lower is not None and upper is not None:
            if lower > upper:
                msg = "confidence_interval_lower cannot exceed confidence_interval_upper."
                raise ValueError(msg)
            confidence_level = cast("float", self.config.evaluation.confidence_level)
            z_score = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
            expected_lower = max(0.0, self.test_noisy_fidelity - z_score * cast("float", standard_error))
            expected_upper = min(1.0, self.test_noisy_fidelity + z_score * cast("float", standard_error))
            if not (
                math.isclose(lower, expected_lower, rel_tol=1e-12, abs_tol=1e-15)
                and math.isclose(upper, expected_upper, rel_tol=1e-12, abs_tol=1e-15)
            ):
                msg = "Confidence bounds must equal the requested clipped normal interval."
                raise ValueError(msg)
        if self.config.evaluation.store_trajectory_sidecar != (sidecar_path is not None):
            msg = "Trajectory-sidecar artifacts must match store_trajectory_sidecar."
            raise ValueError(msg)

    @property
    def status(self) -> str:
        """Result-stream discriminator."""
        return "success"

    @property
    def run_id(self) -> str:
        """Stable identity of the planned result cell."""
        return self.config.run_id

    @property
    def wall_time_seconds(self) -> float:
        """Optimization plus row-specific evaluation wall time."""
        return self.optimization_wall_time_seconds + self.evaluation_wall_time_seconds

    def to_dict(self) -> dict[str, object]:
        """Return the canonical result record with reporting aliases."""
        statistics = self.circuit_statistics
        config = self.config
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "run_id": self.run_id,
            "method": config.method_id,
            "num_qubits": config.target.num_qubits,
            "target_id": config.target.target_id,
            "noise_id": config.test_noise.noise_id,
            "seed": config.target.target_seed,
            "ansatz": _reporting_ansatz(config),
            "num_layers": statistics.num_layers,
            "num_parameters": statistics.num_parameters,
            "circuit_depth": statistics.evaluated_depth,
            "num_1q_gates": statistics.num_1q_gates,
            "num_2q_gates": statistics.num_2q_gates,
            "optimizer_budget": _reporting_optimizer_budget(config),
            "train_trajectories_or_shots": config.optimizer.train_trajectories_or_shots,
            "train_fidelity": self.train_fidelity,
            "logical_test_noiseless_fidelity": self.logical_test_noiseless_fidelity,
            "native_pre_pruning_noiseless_fidelity": self.native_pre_pruning_noiseless_fidelity,
            "test_noiseless_fidelity": self.test_noiseless_fidelity,
            "test_noisy_fidelity": self.test_noisy_fidelity,
            "test_trajectories_or_shots": config.evaluation.test_trajectories_or_shots,
            "noisy_fidelity_standard_deviation": self.noisy_fidelity_standard_deviation,
            "noisy_fidelity_standard_error": self.noisy_fidelity_standard_error,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "sampled_nonidentity_events": self.sampled_nonidentity_events,
            "optimization_wall_time_seconds": self.optimization_wall_time_seconds,
            "evaluation_wall_time_seconds": self.evaluation_wall_time_seconds,
            "wall_time_seconds": self.wall_time_seconds,
            "software_versions": _thaw_json_mapping(self.software_versions),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "git_diff_checksum": self.git_diff_checksum,
            "parameter_checkpoint_path": self.parameter_checkpoint_path,
            "parameter_checkpoint_checksum": self.parameter_checkpoint_checksum,
            "trajectory_sidecar_path": self.trajectory_sidecar_path,
            "trajectory_sidecar_checksum": self.trajectory_sidecar_checksum,
            "notes": self.notes,
            "config": config.to_dict(),
            "circuit_statistics": statistics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BenchmarkResult:
        """Construct a result and verify all serialized reporting aliases.

        Returns:
            The validated successful benchmark result.

        Raises:
            ValueError: If the status or a serialized reporting alias is inconsistent.
        """
        _validate_exact_keys(data, _SUCCESS_RECORD_KEYS, "BenchmarkResult")
        if data["status"] != "success":
            msg = "BenchmarkResult status must be 'success'."
            raise ValueError(msg)
        config = BenchmarkConfig.from_dict(_as_mapping(data["config"], "config"))
        statistics = CircuitStatistics.from_dict(_as_mapping(data["circuit_statistics"], "circuit_statistics"))
        result = cls(
            schema_version=cast("str", data["schema_version"]),
            config=config,
            circuit_statistics=statistics,
            train_fidelity=_json_float(data["train_fidelity"], "train_fidelity"),
            logical_test_noiseless_fidelity=_json_float(
                data["logical_test_noiseless_fidelity"],
                "logical_test_noiseless_fidelity",
            ),
            native_pre_pruning_noiseless_fidelity=_json_optional_float(
                data["native_pre_pruning_noiseless_fidelity"],
                "native_pre_pruning_noiseless_fidelity",
            ),
            test_noiseless_fidelity=_json_float(data["test_noiseless_fidelity"], "test_noiseless_fidelity"),
            test_noisy_fidelity=_json_float(data["test_noisy_fidelity"], "test_noisy_fidelity"),
            noisy_fidelity_standard_deviation=_json_optional_float(
                data["noisy_fidelity_standard_deviation"],
                "noisy_fidelity_standard_deviation",
            ),
            noisy_fidelity_standard_error=_json_optional_float(
                data["noisy_fidelity_standard_error"],
                "noisy_fidelity_standard_error",
            ),
            confidence_interval_lower=_json_optional_float(
                data["confidence_interval_lower"],
                "confidence_interval_lower",
            ),
            confidence_interval_upper=_json_optional_float(
                data["confidence_interval_upper"],
                "confidence_interval_upper",
            ),
            sampled_nonidentity_events=cast("int", data["sampled_nonidentity_events"]),
            optimization_wall_time_seconds=_json_float(
                data["optimization_wall_time_seconds"],
                "optimization_wall_time_seconds",
            ),
            evaluation_wall_time_seconds=_json_float(
                data["evaluation_wall_time_seconds"],
                "evaluation_wall_time_seconds",
            ),
            software_versions=_as_mapping(data["software_versions"], "software_versions"),
            git_commit=cast("str", data["git_commit"]),
            git_dirty=cast("bool", data["git_dirty"]),
            git_diff_checksum=cast("str | None", data["git_diff_checksum"]),
            parameter_checkpoint_path=cast("str", data["parameter_checkpoint_path"]),
            parameter_checkpoint_checksum=cast("str", data["parameter_checkpoint_checksum"]),
            trajectory_sidecar_path=cast("str | None", data["trajectory_sidecar_path"]),
            trajectory_sidecar_checksum=cast("str | None", data["trajectory_sidecar_checksum"]),
            notes=cast("str", data["notes"]),
        )
        result._verify_reporting_aliases(data)
        return result

    def _verify_reporting_aliases(self, data: Mapping[str, object]) -> None:
        """Verify values that are derived from nested typed records.

        Raises:
            ValueError: If a serialized alias differs from its typed source.
        """
        expected = {
            "run_id": self.run_id,
            "method": self.config.method_id,
            "num_qubits": self.config.target.num_qubits,
            "target_id": self.config.target.target_id,
            "noise_id": self.config.test_noise.noise_id,
            "seed": self.config.target.target_seed,
            "ansatz": _reporting_ansatz(self.config),
            "num_layers": self.circuit_statistics.num_layers,
            "num_parameters": self.circuit_statistics.num_parameters,
            "circuit_depth": self.circuit_statistics.evaluated_depth,
            "num_1q_gates": self.circuit_statistics.num_1q_gates,
            "num_2q_gates": self.circuit_statistics.num_2q_gates,
            "optimizer_budget": _reporting_optimizer_budget(self.config),
            "train_trajectories_or_shots": self.config.optimizer.train_trajectories_or_shots,
            "test_trajectories_or_shots": self.config.evaluation.test_trajectories_or_shots,
            "wall_time_seconds": self.wall_time_seconds,
        }
        for name, expected_value in expected.items():
            if not _strict_json_equal(data[name], expected_value):
                msg = f"Serialized reporting field {name!r} does not match its typed source."
                raise ValueError(msg)

    def to_json(self) -> str:
        """Serialize the successful result to deterministic JSON.

        Returns:
            The canonical JSON representation.
        """
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> BenchmarkResult:
        """Deserialize a strict successful-result JSON object.

        Returns:
            The validated successful benchmark result.
        """
        return cls.from_dict(_load_json_object(payload))

    def to_csv_row(self) -> dict[str, object]:
        """Flatten this result into the stable union CSV schema.

        Returns:
            The flattened CSV row.
        """
        return _record_dict_to_csv_row(self.to_dict())


@dataclass(frozen=True, slots=True)
class BenchmarkFailure:
    """Failed planned result cell without fabricated fidelity values."""

    config: BenchmarkConfig
    failure_phase: str
    exception_type: str
    message: str
    software_versions: Mapping[str, object]
    git_commit: str
    git_dirty: bool
    traceback: str | None = None
    retryable: bool = False
    attempt: int = 1
    wall_time_seconds: float = 0.0
    git_diff_checksum: str | None = None
    parameter_checkpoint_path: str | None = None
    parameter_checkpoint_checksum: str | None = None
    notes: str = ""
    schema_version: str = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate failure details and available provenance.

        Raises:
            ValueError: If failure metadata, artifacts, or provenance are invalid.
        """
        if not isinstance(self.config, BenchmarkConfig):
            _raise_type_error("config", "a BenchmarkConfig", self.config)
        if type(self.failure_phase) is not str:
            _raise_type_error("failure_phase", "a string", self.failure_phase)
        if self.failure_phase not in FAILURE_PHASES:
            msg = f"failure_phase must be one of {FAILURE_PHASES}."
            raise ValueError(msg)
        _validate_string(self.exception_type, "exception_type")
        _validate_nonempty_text(self.message, "message")
        if self.traceback is not None and type(self.traceback) is not str:
            _raise_type_error("traceback", "a string or None", self.traceback)
        _validate_bool(self.retryable, "retryable")
        _validate_count(self.attempt, "attempt", minimum=1)
        _validate_float(self.wall_time_seconds, "wall_time_seconds", minimum=0.0)
        object.__setattr__(self, "software_versions", _validate_software_versions(self.software_versions))
        _validate_git_commit(self.git_commit)
        _validate_bool(self.git_dirty, "git_dirty")
        diff_checksum = _validate_checksum(self.git_diff_checksum, "git_diff_checksum", allow_none=True)
        if self.git_dirty != (diff_checksum is not None):
            msg = "git_diff_checksum is required exactly when git_dirty is true."
            raise ValueError(msg)
        checkpoint_path, checkpoint_checksum = _validate_optional_pair(
            self.parameter_checkpoint_path,
            self.parameter_checkpoint_checksum,
            path_name="parameter_checkpoint_path",
            checksum_name="parameter_checkpoint_checksum",
        )
        object.__setattr__(self, "git_diff_checksum", diff_checksum)
        object.__setattr__(self, "parameter_checkpoint_path", checkpoint_path)
        object.__setattr__(self, "parameter_checkpoint_checksum", checkpoint_checksum)
        if type(self.notes) is not str:
            _raise_type_error("notes", "a string", self.notes)
        if self.schema_version != RESULT_SCHEMA_VERSION:
            msg = f"schema_version must be {RESULT_SCHEMA_VERSION!r}."
            raise ValueError(msg)

    @property
    def status(self) -> str:
        """Result-stream discriminator."""
        return "failure"

    @property
    def run_id(self) -> str:
        """Stable identity of the failed planned result cell."""
        return self.config.run_id

    def to_dict(self) -> dict[str, object]:
        """Return the canonical failure record."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "run_id": self.run_id,
            "method": self.config.method_id,
            "num_qubits": self.config.target.num_qubits,
            "target_id": self.config.target.target_id,
            "noise_id": self.config.test_noise.noise_id,
            "seed": self.config.target.target_seed,
            "ansatz": _reporting_ansatz(self.config),
            "num_layers": self.config.ansatz.num_layers,
            "optimizer_budget": _reporting_optimizer_budget(self.config),
            "train_trajectories_or_shots": self.config.optimizer.train_trajectories_or_shots,
            "test_trajectories_or_shots": self.config.evaluation.test_trajectories_or_shots,
            "failure_phase": self.failure_phase,
            "exception_type": self.exception_type,
            "message": self.message,
            "traceback": self.traceback,
            "retryable": self.retryable,
            "attempt": self.attempt,
            "wall_time_seconds": self.wall_time_seconds,
            "software_versions": _thaw_json_mapping(self.software_versions),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "git_diff_checksum": self.git_diff_checksum,
            "parameter_checkpoint_path": self.parameter_checkpoint_path,
            "parameter_checkpoint_checksum": self.parameter_checkpoint_checksum,
            "notes": self.notes,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BenchmarkFailure:
        """Construct a failure and verify all serialized reporting aliases.

        Returns:
            The validated benchmark failure.

        Raises:
            ValueError: If the status or a serialized reporting alias is inconsistent.
        """
        _validate_exact_keys(data, _FAILURE_RECORD_KEYS, "BenchmarkFailure")
        if data["status"] != "failure":
            msg = "BenchmarkFailure status must be 'failure'."
            raise ValueError(msg)
        config = BenchmarkConfig.from_dict(_as_mapping(data["config"], "config"))
        result = cls(
            schema_version=cast("str", data["schema_version"]),
            config=config,
            failure_phase=cast("str", data["failure_phase"]),
            exception_type=cast("str", data["exception_type"]),
            message=cast("str", data["message"]),
            traceback=cast("str | None", data["traceback"]),
            retryable=cast("bool", data["retryable"]),
            attempt=cast("int", data["attempt"]),
            wall_time_seconds=_json_float(data["wall_time_seconds"], "wall_time_seconds"),
            software_versions=_as_mapping(data["software_versions"], "software_versions"),
            git_commit=cast("str", data["git_commit"]),
            git_dirty=cast("bool", data["git_dirty"]),
            git_diff_checksum=cast("str | None", data["git_diff_checksum"]),
            parameter_checkpoint_path=cast("str | None", data["parameter_checkpoint_path"]),
            parameter_checkpoint_checksum=cast("str | None", data["parameter_checkpoint_checksum"]),
            notes=cast("str", data["notes"]),
        )
        result._verify_reporting_aliases(data)
        return result

    def _verify_reporting_aliases(self, data: Mapping[str, object]) -> None:
        """Verify failure metadata derived from the typed configuration.

        Raises:
            ValueError: If a serialized alias differs from its typed source.
        """
        expected = {
            "run_id": self.run_id,
            "method": self.config.method_id,
            "num_qubits": self.config.target.num_qubits,
            "target_id": self.config.target.target_id,
            "noise_id": self.config.test_noise.noise_id,
            "seed": self.config.target.target_seed,
            "ansatz": _reporting_ansatz(self.config),
            "num_layers": self.config.ansatz.num_layers,
            "optimizer_budget": _reporting_optimizer_budget(self.config),
            "train_trajectories_or_shots": self.config.optimizer.train_trajectories_or_shots,
            "test_trajectories_or_shots": self.config.evaluation.test_trajectories_or_shots,
        }
        for name, expected_value in expected.items():
            if not _strict_json_equal(data[name], expected_value):
                msg = f"Serialized reporting field {name!r} does not match its typed source."
                raise ValueError(msg)

    @classmethod
    def from_exception(
        cls,
        *,
        config: BenchmarkConfig,
        failure_phase: str,
        exception: BaseException,
        software_versions: Mapping[str, object],
        git_commit: str,
        git_dirty: bool,
        traceback: str | None = None,
        retryable: bool = False,
        attempt: int = 1,
        wall_time_seconds: float = 0.0,
        git_diff_checksum: str | None = None,
        parameter_checkpoint_path: str | None = None,
        parameter_checkpoint_checksum: str | None = None,
        notes: str = "",
    ) -> BenchmarkFailure:
        """Create a serializable failure record from an exception.

        Returns:
            The validated benchmark failure.
        """
        if not isinstance(exception, BaseException):
            _raise_type_error("exception", "a BaseException", exception)
        message = str(exception) or type(exception).__name__
        return cls(
            config=config,
            failure_phase=failure_phase,
            exception_type=type(exception).__name__,
            message=message,
            traceback=traceback,
            retryable=retryable,
            attempt=attempt,
            wall_time_seconds=wall_time_seconds,
            software_versions=software_versions,
            git_commit=git_commit,
            git_dirty=git_dirty,
            git_diff_checksum=git_diff_checksum,
            parameter_checkpoint_path=parameter_checkpoint_path,
            parameter_checkpoint_checksum=parameter_checkpoint_checksum,
            notes=notes,
        )

    def to_json(self) -> str:
        """Serialize the failure to deterministic JSON.

        Returns:
            The canonical JSON representation.
        """
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> BenchmarkFailure:
        """Deserialize a strict failure JSON object.

        Returns:
            The validated benchmark failure.
        """
        return cls.from_dict(_load_json_object(payload))

    def to_csv_row(self) -> dict[str, object]:
        """Flatten this failure into the stable union CSV schema.

        Returns:
            The flattened CSV row.
        """
        return _record_dict_to_csv_row(self.to_dict())


_SUCCESS_RECORD_KEYS = frozenset({
    "schema_version",
    "status",
    "run_id",
    "method",
    "num_qubits",
    "target_id",
    "noise_id",
    "seed",
    "ansatz",
    "num_layers",
    "num_parameters",
    "circuit_depth",
    "num_1q_gates",
    "num_2q_gates",
    "optimizer_budget",
    "train_trajectories_or_shots",
    "train_fidelity",
    "logical_test_noiseless_fidelity",
    "native_pre_pruning_noiseless_fidelity",
    "test_noiseless_fidelity",
    "test_noisy_fidelity",
    "test_trajectories_or_shots",
    "noisy_fidelity_standard_deviation",
    "noisy_fidelity_standard_error",
    "confidence_interval_lower",
    "confidence_interval_upper",
    "sampled_nonidentity_events",
    "optimization_wall_time_seconds",
    "evaluation_wall_time_seconds",
    "wall_time_seconds",
    "software_versions",
    "git_commit",
    "git_dirty",
    "git_diff_checksum",
    "parameter_checkpoint_path",
    "parameter_checkpoint_checksum",
    "trajectory_sidecar_path",
    "trajectory_sidecar_checksum",
    "notes",
    "config",
    "circuit_statistics",
})

_FAILURE_RECORD_KEYS = frozenset({
    "schema_version",
    "status",
    "run_id",
    "method",
    "num_qubits",
    "target_id",
    "noise_id",
    "seed",
    "ansatz",
    "num_layers",
    "optimizer_budget",
    "train_trajectories_or_shots",
    "test_trajectories_or_shots",
    "failure_phase",
    "exception_type",
    "message",
    "traceback",
    "retryable",
    "attempt",
    "wall_time_seconds",
    "software_versions",
    "git_commit",
    "git_dirty",
    "git_diff_checksum",
    "parameter_checkpoint_path",
    "parameter_checkpoint_checksum",
    "notes",
    "config",
})

CSV_COLUMNS = (
    "schema_version",
    "status",
    "run_id",
    "method",
    "num_qubits",
    "target_id",
    "noise_id",
    "seed",
    "ansatz",
    "num_layers",
    "num_parameters",
    "circuit_depth",
    "num_1q_gates",
    "num_2q_gates",
    "optimizer_budget",
    "train_trajectories_or_shots",
    "train_fidelity",
    "logical_test_noiseless_fidelity",
    "native_pre_pruning_noiseless_fidelity",
    "test_noiseless_fidelity",
    "test_noisy_fidelity",
    "test_trajectories_or_shots",
    "noisy_fidelity_standard_deviation",
    "noisy_fidelity_standard_error",
    "confidence_interval_lower",
    "confidence_interval_upper",
    "sampled_nonidentity_events",
    "optimization_wall_time_seconds",
    "evaluation_wall_time_seconds",
    "wall_time_seconds",
    "software_versions",
    "git_commit",
    "git_dirty",
    "git_diff_checksum",
    "parameter_checkpoint_path",
    "parameter_checkpoint_checksum",
    "trajectory_sidecar_path",
    "trajectory_sidecar_checksum",
    "failure_phase",
    "exception_type",
    "message",
    "traceback",
    "retryable",
    "attempt",
    "notes",
    "config",
    "circuit_statistics",
)

_CSV_JSON_COLUMNS = frozenset({"optimizer_budget", "software_versions", "config", "circuit_statistics"})
_CSV_INTEGER_COLUMNS = frozenset({
    "num_qubits",
    "seed",
    "num_layers",
    "num_parameters",
    "circuit_depth",
    "num_1q_gates",
    "num_2q_gates",
    "train_trajectories_or_shots",
    "test_trajectories_or_shots",
    "sampled_nonidentity_events",
    "attempt",
})
_CSV_FLOAT_COLUMNS = frozenset({
    "train_fidelity",
    "logical_test_noiseless_fidelity",
    "native_pre_pruning_noiseless_fidelity",
    "test_noiseless_fidelity",
    "test_noisy_fidelity",
    "noisy_fidelity_standard_deviation",
    "noisy_fidelity_standard_error",
    "confidence_interval_lower",
    "confidence_interval_upper",
    "optimization_wall_time_seconds",
    "evaluation_wall_time_seconds",
    "wall_time_seconds",
})
_CSV_BOOLEAN_COLUMNS = frozenset({"git_dirty", "retryable"})


def _record_dict_to_csv_row(data: Mapping[str, object]) -> dict[str, object]:
    """Convert a typed record dictionary to the stable union CSV row.

    Returns:
        The flattened CSV row.
    """
    row: dict[str, object] = {}
    for column in CSV_COLUMNS:
        value = data.get(column)
        row[column] = _canonical_json(value) if column in _CSV_JSON_COLUMNS and value is not None else value
    return row


def _csv_optional_value(value: object) -> object | None:
    """Normalize a blank CSV cell to ``None``.

    Returns:
        ``None`` for a blank cell, otherwise the original value.
    """
    return None if value is None or (type(value) is str and not value) else value


def _csv_integer(value: object, name: str) -> int | None:
    """Decode an optional strict base-10 CSV integer.

    Returns:
        The decoded integer, or ``None`` for a blank cell.

    Raises:
        ValueError: If the cell is not a strict base-10 integer.
    """
    normalized = _csv_optional_value(value)
    if normalized is None:
        return None
    if type(normalized) is int:
        return normalized
    if type(normalized) is not str or re.fullmatch(r"-?(0|[1-9][0-9]*)", normalized) is None:
        msg = f"CSV column {name!r} must contain a base-10 integer."
        raise ValueError(msg)
    return int(normalized)


def _csv_float(value: object, name: str) -> float | None:
    """Decode an optional finite CSV float.

    Returns:
        The decoded finite float, or ``None`` for a blank cell.

    Raises:
        ValueError: If the cell is not a finite float.
    """
    normalized = _csv_optional_value(value)
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
    """Decode an optional CSV Boolean.

    Returns:
        The decoded Boolean, or ``None`` for a blank cell.

    Raises:
        ValueError: If the cell is not ``True`` or ``False``.
    """
    normalized = _csv_optional_value(value)
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


def _csv_json(value: object, name: str) -> object | None:
    """Decode an optional JSON-valued CSV cell.

    Returns:
        The decoded JSON object, or ``None`` for a blank cell.
    """
    normalized = _csv_optional_value(value)
    if normalized is None:
        return None
    if type(normalized) is str:
        return _load_json_object(normalized)
    if isinstance(normalized, Mapping):
        return normalized
    _raise_type_error(f"CSV column {name!r}", "a JSON object string", normalized)


def benchmark_record_from_dict(data: Mapping[str, object]) -> BenchmarkResult | BenchmarkFailure:
    """Deserialize a result-stream record using its status discriminator.

    Returns:
        The validated success or failure record.

    Raises:
        ValueError: If the status discriminator is unsupported.
    """
    status = data.get("status")
    if status == "success":
        return BenchmarkResult.from_dict(data)
    if status == "failure":
        return BenchmarkFailure.from_dict(data)
    msg = "Benchmark record status must be 'success' or 'failure'."
    raise ValueError(msg)


def benchmark_record_from_json(payload: str) -> BenchmarkResult | BenchmarkFailure:
    """Deserialize a strict JSON result-stream record.

    Returns:
        The validated success or failure record.
    """
    return benchmark_record_from_dict(_load_json_object(payload))


def benchmark_record_from_csv_row(row: Mapping[str, object]) -> BenchmarkResult | BenchmarkFailure:
    """Deserialize one strict row from the stable union CSV schema.

    Returns:
        The validated success or failure record.

    Raises:
        ValueError: If the row schema or status discriminator is invalid.
    """
    _validate_exact_keys(row, frozenset(CSV_COLUMNS), "benchmark CSV row")
    decoded: dict[str, object] = {}
    for column in CSV_COLUMNS:
        value = row[column]
        if column in _CSV_JSON_COLUMNS:
            decoded[column] = _csv_json(value, column)
        elif column in _CSV_INTEGER_COLUMNS:
            decoded[column] = _csv_integer(value, column)
        elif column in _CSV_FLOAT_COLUMNS:
            decoded[column] = _csv_float(value, column)
        elif column in _CSV_BOOLEAN_COLUMNS:
            decoded[column] = _csv_bool(value, column)
        elif column == "notes":
            decoded[column] = value
        else:
            decoded[column] = _csv_optional_value(value)

    status = decoded["status"]
    if status == "success":
        success_data = {key: decoded[key] for key in _SUCCESS_RECORD_KEYS}
        return BenchmarkResult.from_dict(success_data)
    if status == "failure":
        failure_data = {key: decoded[key] for key in _FAILURE_RECORD_KEYS}
        return BenchmarkFailure.from_dict(failure_data)
    msg = "CSV status must be 'success' or 'failure'."
    raise ValueError(msg)


__all__ = [
    "ANSATZ_ID",
    "BALLARIN_NOISE_ID",
    "CONFIDENCE_INTERVAL_METHODS",
    "CONFIG_SCHEMA_VERSION",
    "CSV_COLUMNS",
    "EVALUATED_REPRESENTATIONS",
    "FAILURE_PHASES",
    "INITIALIZATION_RULES",
    "NOISELESS_NOISE_ID",
    "NOISE_DEFINITION_VERSION",
    "NOISE_IDS",
    "RESULT_SCHEMA_VERSION",
    "RUN_IDENTITY_VERSION",
    "STANDARD_NOISE_IDS",
    "SUPPORTED_QUBIT_COUNTS",
    "TARGET_FIXTURE_FORMAT",
    "TARGET_GENERATION_SEEDS",
    "TARGET_IDS",
    "TRUNCATION_MODES",
    "AnsatzConfig",
    "BenchmarkConfig",
    "BenchmarkFailure",
    "BenchmarkResult",
    "CircuitStatistics",
    "EvaluationConfig",
    "InitializationConfig",
    "NoiseConfig",
    "OptimizerConfig",
    "TargetSelection",
    "benchmark_record_from_csv_row",
    "benchmark_record_from_dict",
    "benchmark_record_from_json",
]
