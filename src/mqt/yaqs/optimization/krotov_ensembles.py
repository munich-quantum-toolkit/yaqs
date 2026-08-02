# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Deterministic fixed-map ensembles for noisy Krotov optimization.

The objects in this module are deliberately independent of optimizer-ordering
randomness.  A trajectory seed is derived from a reserved scientific role, a
resolved root seed, and explicit stage/ensemble/trajectory/refresh indices.  A
logical ensemble identifier uses the same coordinates, while a separate content
checksum additionally seals stage provenance, circuit/provider identities, and
the exact realized replay maps.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal, NoReturn, cast

import numpy as np

from ..core.data_structures.mps import MPS
from .krotov import KrotovNoiseMap, KrotovTJMOptions, KrotovTruncation, forward_tjm_trajectory
from .parameterized_circuit import ParameterizedCircuit

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .gate_noise import GateNoiseProvider

KROTOV_FIXED_MAP_ENSEMBLE_SCHEMA_VERSION = "mqt.yaqs.optimization.krotov_fixed_map_ensemble.v1"
KROTOV_FIXED_MAP_ENSEMBLE_IDENTITY_VERSION = "mqt.yaqs.optimization.krotov_fixed_map_identity.v1"
KROTOV_TRAJECTORY_SEED_DERIVATION_VERSION = "mqt.yaqs.optimization.krotov_trajectory_seed.v1"
KROTOV_LEGACY_TRAJECTORY_SEED_DERIVATION_VERSION = "mqt.yaqs.optimization.krotov_legacy_linear_trajectory_seed.v1"
KROTOV_TRAJECTORY_RNG_ALGORITHM = "numpy.random.Generator(PCG64(SeedSequence(uint64)))"

KrotovMapRole = Literal[
    "training_trajectory",
    "checkpoint_validation",
    "pilot_evaluation",
    "screening_selection",
    "confirmatory_test",
]
KrotovMapSamplingPolicy = Literal["resampled", "crn_fixed", "crn_refresh"]

KROTOV_MAP_ROLES: tuple[KrotovMapRole, ...] = (
    "training_trajectory",
    "checkpoint_validation",
    "pilot_evaluation",
    "screening_selection",
    "confirmatory_test",
)
KROTOV_MAP_SAMPLING_POLICIES: tuple[KrotovMapSamplingPolicy, ...] = ("resampled", "crn_fixed", "crn_refresh")
_KROTOV_MAP_ROLE_BY_NAME: Mapping[str, KrotovMapRole] = {role: role for role in KROTOV_MAP_ROLES}

_UINT64_MAX = 2**64 - 1
_CHECKSUM_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_ENSEMBLE_ID_PREFIX = "krotov_map_ensemble_"
_COMPLEX128_LE = np.dtype("<c16")

_LOCAL_OPERATOR_KEYS = frozenset({"data_base64", "dtype", "shape", "sites"})
_NOISE_MAP_KEYS = frozenset({
    "channel_id",
    "is_identity",
    "jump_process_index",
    "normalization_checkpoints",
    "normalized",
    "operators",
    "outcome_labels",
    "resolved_native_angle",
    "source_gate_index",
})
_ENSEMBLE_KEYS = frozenset({
    "circuit_checksum",
    "content_checksum",
    "ensemble_id",
    "ensemble_index",
    "gate_count",
    "global_iteration_start",
    "nonidentity_event_count",
    "provider_checksum",
    "refresh_index",
    "resolved_seed",
    "role",
    "schema_version",
    "stage_configuration_checksum",
    "stage_id",
    "stage_index",
    "trajectory_count",
    "trajectory_indices",
    "trajectory_maps",
})


def _canonical_json(value: object) -> str:
    """Serialize an already validated JSON-native tree deterministically.

    Returns:
        Canonical compact JSON text with lexicographically sorted object keys.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _canonical_checksum(value: object) -> str:
    """Return a prefixed SHA-256 checksum of canonical JSON content.

    Returns:
        A lowercase ``sha256:``-prefixed digest.
    """
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _require_exact_keys(value: object, expected: frozenset[str], name: str) -> Mapping[str, object]:
    """Validate an object's type and exact versioned field set.

    Returns:
        The input narrowed to a string-keyed mapping.

    Raises:
        TypeError: If the value is not a string-keyed mapping.
        ValueError: If the mapping does not have the exact expected fields.
    """
    if not isinstance(value, Mapping):
        msg = f"{name} must be a mapping, got {type(value).__name__}."
        raise TypeError(msg)
    raw_keys = set(value)
    if any(type(key) is not str for key in raw_keys):
        msg = f"{name} keys must be strings."
        raise TypeError(msg)
    keys = {cast("str", key) for key in raw_keys}
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(keys - expected)
        msg = f"{name} fields differ: missing={missing!r}, unexpected={unexpected!r}."
        raise ValueError(msg)
    return cast("Mapping[str, object]", value)


def _require_uint64(value: object, name: str) -> int:
    """Validate and normalize a nonnegative 64-bit integer.

    Returns:
        A built-in integer.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is outside the unsigned 64-bit range.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        msg = f"{name} must be an integer."
        raise TypeError(msg)
    normalized = int(value)
    if normalized < 0 or normalized > _UINT64_MAX:
        msg = f"{name} must be in [0, {_UINT64_MAX}], got {normalized}."
        raise ValueError(msg)
    return normalized


def _require_optional_nonnegative_int(value: object, name: str) -> int | None:
    """Validate an optional nonnegative integer.

    Returns:
        A built-in integer or ``None``.
    """
    if value is None:
        return None
    return _require_uint64(value, name)


def _require_role(value: object) -> KrotovMapRole:
    """Validate a reserved fixed-map role.

    Returns:
        The validated role.

    Raises:
        TypeError: If the role is not a string.
        ValueError: If the role is not reserved for fixed-map use.
    """
    if type(value) is not str:
        msg = f"role must be a string, got {type(value).__name__}."
        raise TypeError(msg)
    if value not in KROTOV_MAP_ROLES:
        msg = f"role must be one of {KROTOV_MAP_ROLES!r}, got {value!r}."
        raise ValueError(msg)
    return _KROTOV_MAP_ROLE_BY_NAME[value]


def _require_identifier(value: object, name: str) -> str:
    """Validate a stable lowercase identifier.

    Returns:
        The validated identifier.

    Raises:
        TypeError: If the value is not a string.
        ValueError: If the value is not a canonical lowercase slug.
    """
    if type(value) is not str:
        msg = f"{name} must be a string, got {type(value).__name__}."
        raise TypeError(msg)
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        msg = f"{name} must be a lowercase slug, got {value!r}."
        raise ValueError(msg)
    return value


def _require_checksum(value: object, name: str) -> str:
    """Validate a canonical SHA-256 checksum string.

    Returns:
        The validated checksum.

    Raises:
        TypeError: If the value is not a string.
        ValueError: If the value is not a canonical SHA-256 checksum.
    """
    if type(value) is not str:
        msg = f"{name} must be a string, got {type(value).__name__}."
        raise TypeError(msg)
    if _CHECKSUM_PATTERN.fullmatch(value) is None:
        msg = f"{name} must have form 'sha256:' followed by 64 lowercase hexadecimal digits."
        raise ValueError(msg)
    return value


def _require_optional_label(value: object, name: str) -> str | None:
    """Validate an optional nonempty provider label.

    Returns:
        The validated label or ``None``.

    Raises:
        TypeError: If the label is not a string or ``None``.
        ValueError: If the label is empty or padded by whitespace.
    """
    if value is None:
        return None
    if type(value) is not str:
        msg = f"{name} must be a string or None, got {type(value).__name__}."
        raise TypeError(msg)
    if not value or value != value.strip():
        msg = f"{name} must be nonempty and have no surrounding whitespace."
        raise ValueError(msg)
    return value


def _require_optional_finite_float(value: object, name: str) -> float | None:
    """Validate and normalize an optional finite real number.

    Returns:
        A built-in finite float or ``None``.

    Raises:
        TypeError: If the value is not a real number or ``None``.
        ValueError: If the value is not finite.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        msg = f"{name} must be a real number or None."
        raise TypeError(msg)
    normalized = float(value)
    if not math.isfinite(normalized):
        msg = f"{name} must be finite, got {normalized!r}."
        raise ValueError(msg)
    return normalized or 0.0


def _require_bool(value: object, name: str) -> bool:
    """Validate a built-in Boolean.

    Returns:
        The validated Boolean.

    Raises:
        TypeError: If the value is not a built-in Boolean.
    """
    if type(value) is not bool:
        msg = f"{name} must be a bool, got {type(value).__name__}."
        raise TypeError(msg)
    return value


def _require_optional_bool(value: object, name: str) -> bool | None:
    """Validate an optional built-in Boolean.

    Returns:
        The validated Boolean or ``None``.
    """
    if value is None:
        return None
    return _require_bool(value, name)


def _require_string_tuple(value: object, name: str, *, serialized: bool = False) -> tuple[str, ...]:
    """Validate a tuple, or serialized list, of provider labels.

    Returns:
        An immutable tuple of labels.

    Raises:
        TypeError: If the collection has the wrong representation.
    """
    expected_type = list if serialized else tuple
    if type(value) is not expected_type:
        msg = f"{name} must be a {expected_type.__name__}."
        raise TypeError(msg)
    labels = []
    for index, label in enumerate(cast("Sequence[object]", value)):
        validated = _require_optional_label(label, f"{name}[{index}]")
        assert validated is not None
        labels.append(validated)
    return tuple(labels)


def _require_sites(value: object, name: str, *, serialized: bool = False) -> tuple[int, ...]:
    """Validate one- or two-site local support.

    Returns:
        An immutable tuple of site indices.

    Raises:
        TypeError: If the collection has the wrong representation.
        ValueError: If support is not one or two distinct nonnegative sites.
    """
    expected_type = list if serialized else tuple
    if type(value) is not expected_type:
        msg = f"{name} must be a {expected_type.__name__}."
        raise TypeError(msg)
    items = cast("Sequence[object]", value)
    if len(items) not in {1, 2}:
        msg = f"{name} must contain one or two sites, got {len(items)}."
        raise ValueError(msg)
    sites = tuple(_require_uint64(site, f"{name}[{index}]") for index, site in enumerate(items))
    if len(set(sites)) != len(sites):
        msg = f"{name} contains duplicate sites {sites!r}."
        raise ValueError(msg)
    return sites


def _require_checkpoints(
    value: object,
    operator_count: int,
    name: str,
    *,
    serialized: bool = False,
) -> tuple[int, ...]:
    """Validate canonical normalization checkpoints.

    Returns:
        Strictly increasing operator counts.

    Raises:
        TypeError: If the collection has the wrong representation.
        ValueError: If checkpoints are noncanonical or outside the map.
    """
    expected_type = list if serialized else tuple
    if type(value) is not expected_type:
        msg = f"{name} must be a {expected_type.__name__}."
        raise TypeError(msg)
    checkpoints = tuple(
        _require_uint64(checkpoint, f"{name}[{index}]")
        for index, checkpoint in enumerate(cast("Sequence[object]", value))
    )
    if any(checkpoint > operator_count for checkpoint in checkpoints):
        msg = f"{name} values must not exceed operator count {operator_count}, got {checkpoints!r}."
        raise ValueError(msg)
    if tuple(sorted(set(checkpoints))) != checkpoints:
        msg = f"{name} must be strictly increasing without duplicates, got {checkpoints!r}."
        raise ValueError(msg)
    return checkpoints


def _immutable_matrix_bytes(matrix: object, sites: tuple[int, ...], name: str) -> tuple[bytes, tuple[int, int]]:
    """Validate and copy a local matrix into canonical binary form.

    Returns:
        Little-endian complex128 C-order bytes and the square matrix shape.

    Raises:
        TypeError: If the matrix cannot be converted to complex128.
        ValueError: If its shape or entries are invalid.
    """
    try:
        candidate = np.asarray(matrix)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be convertible to a complex matrix."
        raise TypeError(msg) from error
    if candidate.ndim != 2:
        msg = f"{name} must be two-dimensional, got shape {candidate.shape}."
        raise ValueError(msg)
    try:
        canonical = np.ascontiguousarray(candidate, dtype=_COMPLEX128_LE)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be convertible to complex128."
        raise TypeError(msg) from error
    expected_dimension = 2 ** len(sites)
    expected_shape = (expected_dimension, expected_dimension)
    if canonical.shape != expected_shape:
        msg = f"{name} on {len(sites)} site(s) must have shape {expected_shape}, got {canonical.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(canonical)):
        msg = f"{name} must contain only finite entries."
        raise ValueError(msg)
    return canonical.tobytes(order="C"), expected_shape


@dataclass(frozen=True, slots=True)
class _FrozenLocalOperator:
    """Defensively copied local replay operator."""

    matrix_bytes: bytes
    shape: tuple[int, int]
    sites: tuple[int, ...]

    @classmethod
    def from_operator(cls, value: object, name: str) -> _FrozenLocalOperator:
        """Freeze one ``(matrix, sites)`` pair.

        Returns:
            An immutable local operator.

        Raises:
            TypeError: If the operator is not a two-item tuple.
        """
        if type(value) is not tuple or len(value) != 2:
            msg = f"{name} must be a two-item tuple (matrix, sites)."
            raise TypeError(msg)
        matrix, raw_sites = value
        sites = _require_sites(raw_sites, f"{name}.sites")
        matrix_bytes, shape = _immutable_matrix_bytes(matrix, sites, f"{name}.matrix")
        return cls(matrix_bytes=matrix_bytes, shape=shape, sites=sites)

    @classmethod
    def from_dict(cls, value: object, name: str) -> _FrozenLocalOperator:
        """Decode one strict serialized local operator.

        Returns:
            The verified immutable operator.

        Raises:
            TypeError: If a serialized field has the wrong type.
            ValueError: If matrix metadata or binary content is invalid.
        """
        mapping = _require_exact_keys(value, _LOCAL_OPERATOR_KEYS, name)
        if mapping["dtype"] != "<c16" or type(mapping["dtype"]) is not str:
            msg = f"{name}.dtype must be '<c16'."
            raise ValueError(msg)
        sites = _require_sites(mapping["sites"], f"{name}.sites", serialized=True)
        shape_value = mapping["shape"]
        if type(shape_value) is not list or len(shape_value) != 2:
            msg = f"{name}.shape must be a two-item list."
            raise TypeError(msg)
        shape = tuple(
            _require_uint64(item, f"{name}.shape[{index}]")
            for index, item in enumerate(cast("list[object]", shape_value))
        )
        expected_dimension = 2 ** len(sites)
        if shape != (expected_dimension, expected_dimension):
            msg = f"{name}.shape must be {(expected_dimension, expected_dimension)}, got {shape}."
            raise ValueError(msg)
        encoded = mapping["data_base64"]
        if type(encoded) is not str:
            msg = f"{name}.data_base64 must be a string."
            raise TypeError(msg)
        try:
            matrix_bytes = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as error:
            msg = f"{name}.data_base64 is not canonical base64."
            raise ValueError(msg) from error
        expected_bytes = expected_dimension * expected_dimension * _COMPLEX128_LE.itemsize
        if len(matrix_bytes) != expected_bytes:
            msg = f"{name}.data_base64 decodes to {len(matrix_bytes)} bytes; expected {expected_bytes}."
            raise ValueError(msg)
        if base64.b64encode(matrix_bytes).decode("ascii") != encoded:
            msg = f"{name}.data_base64 is not canonical base64."
            raise ValueError(msg)
        matrix = np.frombuffer(matrix_bytes, dtype=_COMPLEX128_LE).reshape(shape)
        if not np.all(np.isfinite(matrix)):
            msg = f"{name} matrix must contain only finite entries."
            raise ValueError(msg)
        return cls(matrix_bytes=matrix_bytes, shape=cast("tuple[int, int]", shape), sites=sites)

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-native representation.

        Returns:
            A detached serialized operator.
        """
        return {
            "data_base64": base64.b64encode(self.matrix_bytes).decode("ascii"),
            "dtype": "<c16",
            "shape": list(self.shape),
            "sites": list(self.sites),
        }

    def thaw(self) -> tuple[NDArray[np.complex128], tuple[int, ...]]:
        """Return a fresh writable replay operator.

        Returns:
            A detached matrix and its immutable site tuple.
        """
        matrix = np.frombuffer(self.matrix_bytes, dtype=_COMPLEX128_LE).reshape(self.shape).copy(order="C")
        return cast("NDArray[np.complex128]", matrix), self.sites


@dataclass(frozen=True, slots=True)
class _FrozenNoiseMap:
    """Defensively copied fixed noise map."""

    operators: tuple[_FrozenLocalOperator, ...]
    normalized: bool
    jump_process_index: int | None
    channel_id: str | None
    outcome_labels: tuple[str, ...]
    source_gate_index: int | None
    resolved_native_angle: float | None
    is_identity: bool | None
    normalization_checkpoints: tuple[int, ...]

    @classmethod
    def from_noise_map(cls, value: object, name: str) -> _FrozenNoiseMap:
        """Defensively freeze one public replay map.

        Returns:
            A validated immutable map.

        Raises:
            TypeError: If the value or its operator container has the wrong type.
        """
        if not isinstance(value, KrotovNoiseMap):
            msg = f"{name} must be a KrotovNoiseMap, got {type(value).__name__}."
            raise TypeError(msg)
        if type(value.operators) is not tuple:
            msg = f"{name}.operators must be a tuple."
            raise TypeError(msg)
        operators = tuple(
            _FrozenLocalOperator.from_operator(operator, f"{name}.operators[{index}]")
            for index, operator in enumerate(value.operators)
        )
        return cls(
            operators=operators,
            normalized=_require_bool(value.normalized, f"{name}.normalized"),
            jump_process_index=_require_optional_nonnegative_int(
                value.jump_process_index,
                f"{name}.jump_process_index",
            ),
            channel_id=_require_optional_label(value.channel_id, f"{name}.channel_id"),
            outcome_labels=_require_string_tuple(value.outcome_labels, f"{name}.outcome_labels"),
            source_gate_index=_require_optional_nonnegative_int(
                value.source_gate_index,
                f"{name}.source_gate_index",
            ),
            resolved_native_angle=_require_optional_finite_float(
                value.resolved_native_angle,
                f"{name}.resolved_native_angle",
            ),
            is_identity=_require_optional_bool(value.is_identity, f"{name}.is_identity"),
            normalization_checkpoints=_require_checkpoints(
                value.normalization_checkpoints,
                len(operators),
                f"{name}.normalization_checkpoints",
            ),
        )

    @classmethod
    def from_dict(cls, value: object, name: str) -> _FrozenNoiseMap:
        """Decode one strict serialized fixed map.

        Returns:
            A verified immutable map.

        Raises:
            TypeError: If a serialized field has the wrong type.
        """
        mapping = _require_exact_keys(value, _NOISE_MAP_KEYS, name)
        raw_operators = mapping["operators"]
        if type(raw_operators) is not list:
            msg = f"{name}.operators must be a list."
            raise TypeError(msg)
        operators = tuple(
            _FrozenLocalOperator.from_dict(operator, f"{name}.operators[{index}]")
            for index, operator in enumerate(cast("list[object]", raw_operators))
        )
        angle = mapping["resolved_native_angle"]
        if angle is not None and type(angle) is not float:
            msg = f"{name}.resolved_native_angle must be a float or None."
            raise TypeError(msg)
        return cls(
            operators=operators,
            normalized=_require_bool(mapping["normalized"], f"{name}.normalized"),
            jump_process_index=_require_optional_nonnegative_int(
                mapping["jump_process_index"],
                f"{name}.jump_process_index",
            ),
            channel_id=_require_optional_label(mapping["channel_id"], f"{name}.channel_id"),
            outcome_labels=_require_string_tuple(
                mapping["outcome_labels"],
                f"{name}.outcome_labels",
                serialized=True,
            ),
            source_gate_index=_require_optional_nonnegative_int(
                mapping["source_gate_index"],
                f"{name}.source_gate_index",
            ),
            resolved_native_angle=_require_optional_finite_float(angle, f"{name}.resolved_native_angle"),
            is_identity=_require_optional_bool(mapping["is_identity"], f"{name}.is_identity"),
            normalization_checkpoints=_require_checkpoints(
                mapping["normalization_checkpoints"],
                len(operators),
                f"{name}.normalization_checkpoints",
                serialized=True,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a detached canonical JSON-native map.

        Returns:
            The complete fixed-map representation.
        """
        return {
            "channel_id": self.channel_id,
            "is_identity": self.is_identity,
            "jump_process_index": self.jump_process_index,
            "normalization_checkpoints": list(self.normalization_checkpoints),
            "normalized": self.normalized,
            "operators": [operator.to_dict() for operator in self.operators],
            "outcome_labels": list(self.outcome_labels),
            "resolved_native_angle": self.resolved_native_angle,
            "source_gate_index": self.source_gate_index,
        }

    def thaw(self) -> KrotovNoiseMap:
        """Return a fully detached public replay map.

        Returns:
            A fresh :class:`KrotovNoiseMap` and fresh operator arrays.
        """
        return KrotovNoiseMap(
            operators=tuple(operator.thaw() for operator in self.operators),
            normalized=self.normalized,
            jump_process_index=self.jump_process_index,
            channel_id=self.channel_id,
            outcome_labels=self.outcome_labels,
            source_gate_index=self.source_gate_index,
            resolved_native_angle=self.resolved_native_angle,
            is_identity=self.is_identity,
            normalization_checkpoints=self.normalization_checkpoints,
        )

    @property
    def is_nonidentity_event(self) -> bool:
        """Whether this map records one sampled non-identity channel event."""
        return self.is_identity is False or (self.jump_process_index is not None and self.is_identity is None)


def derive_krotov_trajectory_seed(
    *,
    role: KrotovMapRole,
    resolved_seed: int,
    stage_index: int,
    ensemble_index: int,
    trajectory_index: int,
    refresh_index: int,
) -> int:
    """Derive a stable trajectory seed from explicit scientific coordinates.

    The derivation uses SHA-256 over canonical JSON and returns the first 64 bits
    in network byte order.  It neither consumes shared generator state nor uses
    Python's process-randomized ``hash()`` function.

    Args:
        role: Reserved random-stream role.
        resolved_seed: Resolved root seed for that role.
        stage_index: Zero-based training-stage index.
        ensemble_index: Zero-based ensemble index selected by the schedule.
        trajectory_index: Zero-based trajectory index within the ensemble.
        refresh_index: Zero-based CRN refresh index.

    Returns:
        A deterministic unsigned 64-bit integer suitable for ``PCG64``.
    """
    payload = {
        "ensemble_index": _require_uint64(ensemble_index, "ensemble_index"),
        "refresh_index": _require_uint64(refresh_index, "refresh_index"),
        "resolved_seed": _require_uint64(resolved_seed, "resolved_seed"),
        "role": _require_role(role),
        "stage_index": _require_uint64(stage_index, "stage_index"),
        "trajectory_index": _require_uint64(trajectory_index, "trajectory_index"),
        "version": KROTOV_TRAJECTORY_SEED_DERIVATION_VERSION,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def derive_legacy_krotov_trajectory_seed(
    *,
    optimizer_iteration_seed: int,
    trajectory_index: int,
    base_seed: int = 0,
) -> int:
    """Reproduce the pre-Phase-II linear circuit-TJM trajectory seed.

    This function exists only for isolated historical reproduction. Corrected
    methods use :func:`derive_krotov_trajectory_seed`.

    Args:
        optimizer_iteration_seed: Historical Krotov ``options.seed`` value at
            which the fixed CRN ensemble was sampled.
        trajectory_index: Zero-based trajectory position.
        base_seed: Historical ``KrotovTJMOptions.random_seed`` value.

    Returns:
        The exact legacy seed ``base + 1_000_003 * iteration + trajectory``.
    """
    iteration = _require_uint64(optimizer_iteration_seed, "optimizer_iteration_seed")
    trajectory = _require_uint64(trajectory_index, "trajectory_index")
    base = _require_uint64(base_seed, "base_seed")
    resolved = base + 1_000_003 * iteration + trajectory
    return _require_uint64(resolved, "legacy trajectory seed")


@dataclass(frozen=True, slots=True)
class KrotovMapSchedulePoint:
    """Resolved fixed-map coordinates for one optimizer iteration."""

    local_iteration: int
    global_iteration: int
    ensemble_index: int
    refresh_index: int
    is_refresh_boundary: bool


@dataclass(frozen=True, slots=True)
class KrotovMapSchedule:
    """Deterministic ensemble schedule with resume-continuous global offsets.

    Args:
        policy: Resample every iteration, keep one fixed CRN ensemble, or refresh
            a CRN ensemble periodically.
        refresh_interval: Positive interval required exactly for ``crn_refresh``.
        global_iteration_offset: Stage-global index corresponding to local
            iteration zero for resumed execution. It is not cumulative across
            distinct pipeline stages.
    """

    policy: KrotovMapSamplingPolicy
    refresh_interval: int | None = None
    global_iteration_offset: int = 0

    def __post_init__(self) -> None:
        """Validate and normalize schedule settings.

        Raises:
            ValueError: If policy and refresh settings are inconsistent.
        """
        if type(self.policy) is not str or self.policy not in KROTOV_MAP_SAMPLING_POLICIES:
            msg = f"policy must be one of {KROTOV_MAP_SAMPLING_POLICIES!r}, got {self.policy!r}."
            raise ValueError(msg)
        offset = _require_uint64(self.global_iteration_offset, "global_iteration_offset")
        object.__setattr__(self, "global_iteration_offset", offset)
        if self.policy == "crn_refresh":
            if self.refresh_interval is None:
                msg = "crn_refresh requires a positive refresh_interval."
                raise ValueError(msg)
            interval = _require_uint64(self.refresh_interval, "refresh_interval")
            if interval == 0:
                msg = "refresh_interval must be positive."
                raise ValueError(msg)
            object.__setattr__(self, "refresh_interval", interval)
        elif self.refresh_interval is not None:
            msg = "refresh_interval is valid only for the crn_refresh policy."
            raise ValueError(msg)

    def point(self, local_iteration: int) -> KrotovMapSchedulePoint:
        """Resolve one local optimizer iteration to fixed-map coordinates.

        Args:
            local_iteration: Zero-based iteration within the current execution.

        Returns:
            Global iteration, ensemble/refresh indices, and boundary status.

        Raises:
            ValueError: If the resulting global iteration exceeds uint64.
        """
        local = _require_uint64(local_iteration, "local_iteration")
        global_iteration = self.global_iteration_offset + local
        if global_iteration > _UINT64_MAX:
            msg = "global_iteration_offset + local_iteration exceeds uint64."
            raise ValueError(msg)
        if self.policy == "resampled":
            ensemble_index = global_iteration
            refresh_index = global_iteration
            boundary = True
        elif self.policy == "crn_fixed":
            ensemble_index = 0
            refresh_index = 0
            boundary = global_iteration == 0
        else:
            assert self.refresh_interval is not None
            refresh_index = global_iteration // self.refresh_interval
            ensemble_index = refresh_index
            boundary = global_iteration % self.refresh_interval == 0
        return KrotovMapSchedulePoint(
            local_iteration=local,
            global_iteration=global_iteration,
            ensemble_index=ensemble_index,
            refresh_index=refresh_index,
            is_refresh_boundary=boundary,
        )

    def indices_for_iteration(self, local_iteration: int) -> tuple[int, int]:
        """Return only the ensemble and refresh indices for one iteration.

        Returns:
            ``(ensemble_index, refresh_index)``.
        """
        point = self.point(local_iteration)
        return point.ensemble_index, point.refresh_index


@dataclass(frozen=True, slots=True, eq=False, init=False)
class KrotovFixedMapEnsemble:
    """Immutable, checksum-sealed ensemble of exact Krotov replay maps.

    Source matrices are copied immediately into canonical little-endian
    complex128 bytes.  :meth:`replay_maps` always constructs fresh arrays, so
    neither construction-time nor replay-time mutation can alter the ensemble.
    """

    role: KrotovMapRole
    resolved_seed: int
    stage_index: int
    stage_id: str
    stage_configuration_checksum: str
    circuit_checksum: str
    provider_checksum: str
    ensemble_index: int
    refresh_index: int
    global_iteration_start: int
    _trajectory_maps: tuple[tuple[_FrozenNoiseMap, ...], ...] = field(repr=False)
    schema_version: str = field(default=KROTOV_FIXED_MAP_ENSEMBLE_SCHEMA_VERSION, init=False)
    __hash__ = None

    def __init__(
        self,
        *,
        role: KrotovMapRole,
        resolved_seed: int,
        stage_index: int,
        stage_id: str,
        stage_configuration_checksum: str,
        circuit_checksum: str,
        provider_checksum: str,
        ensemble_index: int,
        refresh_index: int,
        global_iteration_start: int,
        trajectory_maps: Sequence[Sequence[KrotovNoiseMap]],
    ) -> None:
        """Validate metadata and defensively freeze exact replay maps.

        Args:
            role: Reserved random-stream role.
            resolved_seed: Resolved root seed for map sampling.
            stage_index: Zero-based pipeline stage index.
            stage_id: Stable pipeline stage identifier.
            stage_configuration_checksum: Checksum of the resolved stage config.
            circuit_checksum: Checksum of the exact sampled circuit.
            provider_checksum: Checksum of the exact resolved noise provider.
            ensemble_index: Schedule-derived ensemble index.
            refresh_index: Schedule-derived refresh index.
            global_iteration_start: First global iteration using this ensemble.
            trajectory_maps: One complete gate-map sequence per trajectory.

        Raises:
            TypeError: If a collection, map, or metadata value has the wrong type.
            ValueError: If trajectories differ in gate count or metadata is invalid.
        """
        if isinstance(trajectory_maps, (str, bytes)) or not isinstance(trajectory_maps, Sequence):
            msg = "trajectory_maps must be a nonempty sequence of trajectory map sequences."
            raise TypeError(msg)
        if len(trajectory_maps) == 0:
            msg = "trajectory_maps must contain at least one trajectory."
            raise ValueError(msg)
        frozen_trajectories: list[tuple[_FrozenNoiseMap, ...]] = []
        for trajectory_index, raw_maps in enumerate(trajectory_maps):
            if isinstance(raw_maps, (str, bytes)) or not isinstance(raw_maps, Sequence):
                msg = f"trajectory_maps[{trajectory_index}] must be a sequence of KrotovNoiseMap objects."
                raise TypeError(msg)
            frozen_trajectories.append(
                tuple(
                    _FrozenNoiseMap.from_noise_map(
                        noise_map,
                        f"trajectory_maps[{trajectory_index}][{gate_index}]",
                    )
                    for gate_index, noise_map in enumerate(raw_maps)
                )
            )
        gate_counts = {len(maps) for maps in frozen_trajectories}
        if len(gate_counts) != 1:
            msg = f"Every trajectory must contain the same gate-map count, got {sorted(gate_counts)!r}."
            raise ValueError(msg)
        for trajectory_index, maps in enumerate(frozen_trajectories):
            for gate_index, noise_map in enumerate(maps):
                if noise_map.source_gate_index not in {None, gate_index}:
                    msg = (
                        f"trajectory_maps[{trajectory_index}][{gate_index}].source_gate_index "
                        "must be None or match its circuit-gate position."
                    )
                    raise ValueError(msg)

        object.__setattr__(self, "role", _require_role(role))
        object.__setattr__(self, "resolved_seed", _require_uint64(resolved_seed, "resolved_seed"))
        object.__setattr__(self, "stage_index", _require_uint64(stage_index, "stage_index"))
        object.__setattr__(self, "stage_id", _require_identifier(stage_id, "stage_id"))
        object.__setattr__(
            self,
            "stage_configuration_checksum",
            _require_checksum(stage_configuration_checksum, "stage_configuration_checksum"),
        )
        object.__setattr__(self, "circuit_checksum", _require_checksum(circuit_checksum, "circuit_checksum"))
        object.__setattr__(self, "provider_checksum", _require_checksum(provider_checksum, "provider_checksum"))
        object.__setattr__(self, "ensemble_index", _require_uint64(ensemble_index, "ensemble_index"))
        object.__setattr__(self, "refresh_index", _require_uint64(refresh_index, "refresh_index"))
        object.__setattr__(
            self,
            "global_iteration_start",
            _require_uint64(global_iteration_start, "global_iteration_start"),
        )
        object.__setattr__(self, "_trajectory_maps", tuple(frozen_trajectories))
        object.__setattr__(self, "schema_version", KROTOV_FIXED_MAP_ENSEMBLE_SCHEMA_VERSION)

    @property
    def trajectory_count(self) -> int:
        """Number of fixed trajectories in the ensemble."""
        return len(self._trajectory_maps)

    @property
    def trajectory_indices(self) -> tuple[int, ...]:
        """Canonical zero-based trajectory indices participating in identity."""
        return tuple(range(self.trajectory_count))

    @property
    def gate_count(self) -> int:
        """Number of post-gate maps in every trajectory."""
        return len(self._trajectory_maps[0])

    def _identity_dict(self) -> dict[str, object]:
        """Return the logical seed/index identity payload."""
        return {
            "ensemble_index": self.ensemble_index,
            "refresh_index": self.refresh_index,
            "resolved_seed": self.resolved_seed,
            "role": self.role,
            "stage_index": self.stage_index,
            "trajectory_indices": list(self.trajectory_indices),
            "version": KROTOV_FIXED_MAP_ENSEMBLE_IDENTITY_VERSION,
        }

    @property
    def ensemble_id(self) -> str:
        """Logical identifier derived only from seed-domain coordinates."""
        digest = hashlib.sha256(_canonical_json(self._identity_dict()).encode("utf-8")).hexdigest()
        return f"{_ENSEMBLE_ID_PREFIX}{digest}"

    @property
    def nonidentity_event_count(self) -> int:
        """Total sampled non-identity events across all trajectory maps."""
        return sum(
            noise_map.is_nonidentity_event for trajectory_maps in self._trajectory_maps for noise_map in trajectory_maps
        )

    def _content_dict(self) -> dict[str, object]:
        """Return every content-bound field except the content checksum."""
        return {
            "circuit_checksum": self.circuit_checksum,
            "ensemble_id": self.ensemble_id,
            "ensemble_index": self.ensemble_index,
            "gate_count": self.gate_count,
            "global_iteration_start": self.global_iteration_start,
            "nonidentity_event_count": self.nonidentity_event_count,
            "provider_checksum": self.provider_checksum,
            "refresh_index": self.refresh_index,
            "resolved_seed": self.resolved_seed,
            "role": self.role,
            "schema_version": self.schema_version,
            "stage_configuration_checksum": self.stage_configuration_checksum,
            "stage_id": self.stage_id,
            "stage_index": self.stage_index,
            "trajectory_count": self.trajectory_count,
            "trajectory_indices": list(self.trajectory_indices),
            "trajectory_maps": [
                [noise_map.to_dict() for noise_map in trajectory_maps] for trajectory_maps in self._trajectory_maps
            ],
        }

    @property
    def content_checksum(self) -> str:
        """Checksum sealing provenance, circuit/provider bindings, and maps."""
        return _canonical_checksum(self._content_dict())

    def replay_maps(self) -> list[list[KrotovNoiseMap]]:
        """Return caller-owned maps for an exact forward/backward replay.

        Returns:
            A mutable outer structure containing immutable map records whose
            operator matrices are fresh writable copies.
        """
        return [[noise_map.thaw() for noise_map in trajectory_maps] for trajectory_maps in self._trajectory_maps]

    def verify_bindings(
        self,
        *,
        stage_configuration_checksum: str | None = None,
        circuit_checksum: str | None = None,
        provider_checksum: str | None = None,
    ) -> None:
        """Reject replay under caller-supplied incompatible content bindings.

        Args:
            stage_configuration_checksum: Expected stage checksum, when known.
            circuit_checksum: Expected exact circuit checksum, when known.
            provider_checksum: Expected exact provider checksum, when known.

        Raises:
            ValueError: If any supplied binding differs from this ensemble.
        """
        for name, supplied, actual in (
            ("stage_configuration_checksum", stage_configuration_checksum, self.stage_configuration_checksum),
            ("circuit_checksum", circuit_checksum, self.circuit_checksum),
            ("provider_checksum", provider_checksum, self.provider_checksum),
        ):
            if supplied is not None and _require_checksum(supplied, name) != actual:
                msg = f"{name} does not match this fixed-map ensemble."
                raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic checksum-sealed JSON-native record.

        Returns:
            A detached mapping safe for serialization.
        """
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return strict canonical JSON.

        Returns:
            Deterministically ordered compact JSON text.
        """
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> KrotovFixedMapEnsemble:
        """Decode and verify a checksum-sealed serialized ensemble.

        Args:
            value: JSON-native ensemble mapping.

        Returns:
            A defensively copied verified ensemble.

        Raises:
            TypeError: If the serialized record has an invalid field type.
            ValueError: If derived fields, identity, or checksum do not verify.
        """
        mapping = _require_exact_keys(value, _ENSEMBLE_KEYS, "fixed-map ensemble")
        if mapping["schema_version"] != KROTOV_FIXED_MAP_ENSEMBLE_SCHEMA_VERSION:
            msg = f"schema_version must be {KROTOV_FIXED_MAP_ENSEMBLE_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        raw_trajectory_maps = mapping["trajectory_maps"]
        if type(raw_trajectory_maps) is not list or not raw_trajectory_maps:
            msg = "trajectory_maps must be a nonempty list."
            raise TypeError(msg)
        frozen_maps: list[tuple[_FrozenNoiseMap, ...]] = []
        for trajectory_index, raw_maps in enumerate(cast("list[object]", raw_trajectory_maps)):
            if type(raw_maps) is not list:
                msg = f"trajectory_maps[{trajectory_index}] must be a list."
                raise TypeError(msg)
            frozen_maps.append(
                tuple(
                    _FrozenNoiseMap.from_dict(
                        noise_map,
                        f"trajectory_maps[{trajectory_index}][{gate_index}]",
                    )
                    for gate_index, noise_map in enumerate(cast("list[object]", raw_maps))
                )
            )
        trajectory_maps = tuple(tuple(noise_map.thaw() for noise_map in maps) for maps in frozen_maps)
        ensemble = cls(
            role=_require_role(mapping["role"]),
            resolved_seed=_require_uint64(mapping["resolved_seed"], "resolved_seed"),
            stage_index=_require_uint64(mapping["stage_index"], "stage_index"),
            stage_id=_require_identifier(mapping["stage_id"], "stage_id"),
            stage_configuration_checksum=_require_checksum(
                mapping["stage_configuration_checksum"],
                "stage_configuration_checksum",
            ),
            circuit_checksum=_require_checksum(mapping["circuit_checksum"], "circuit_checksum"),
            provider_checksum=_require_checksum(mapping["provider_checksum"], "provider_checksum"),
            ensemble_index=_require_uint64(mapping["ensemble_index"], "ensemble_index"),
            refresh_index=_require_uint64(mapping["refresh_index"], "refresh_index"),
            global_iteration_start=_require_uint64(
                mapping["global_iteration_start"],
                "global_iteration_start",
            ),
            trajectory_maps=trajectory_maps,
        )
        for name, supplied, expected in (
            ("trajectory_count", mapping["trajectory_count"], ensemble.trajectory_count),
            ("gate_count", mapping["gate_count"], ensemble.gate_count),
            ("nonidentity_event_count", mapping["nonidentity_event_count"], ensemble.nonidentity_event_count),
        ):
            if _require_uint64(supplied, name) != expected:
                msg = f"{name} does not match the serialized trajectory maps."
                raise ValueError(msg)
        indices = mapping["trajectory_indices"]
        if type(indices) is not list:
            msg = "trajectory_indices must be a list."
            raise TypeError(msg)
        normalized_indices = tuple(
            _require_uint64(index, f"trajectory_indices[{position}]")
            for position, index in enumerate(cast("list[object]", indices))
        )
        if normalized_indices != ensemble.trajectory_indices:
            msg = "trajectory_indices must be the canonical zero-based trajectory sequence."
            raise ValueError(msg)
        if type(mapping["ensemble_id"]) is not str or mapping["ensemble_id"] != ensemble.ensemble_id:
            msg = "ensemble_id does not match the seed/index-derived logical identity."
            raise ValueError(msg)
        supplied_checksum = _require_checksum(mapping["content_checksum"], "content_checksum")
        if supplied_checksum != ensemble.content_checksum:
            msg = (
                "fixed-map ensemble content checksum mismatch: "
                f"expected {supplied_checksum}, computed {ensemble.content_checksum}."
            )
            raise ValueError(msg)
        return ensemble

    @classmethod
    def from_json(cls, payload: str) -> KrotovFixedMapEnsemble:
        """Decode and verify strict canonical JSON with one optional newline.

        Args:
            payload: Canonical serialized ensemble.

        Returns:
            A verified fixed-map ensemble.

        Raises:
            TypeError: If the payload or top-level JSON value has the wrong type.
            ValueError: If JSON is malformed, noncanonical, or fails verification.
        """
        if type(payload) is not str:
            msg = f"payload must be a string, got {type(payload).__name__}."
            raise TypeError(msg)
        normalized = payload.removesuffix("\n")
        try:
            decoded = json.loads(
                normalized,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as error:
            msg = f"Could not decode canonical JSON: {error}."
            raise ValueError(msg) from error
        if not isinstance(decoded, Mapping):
            msg = f"Canonical JSON top level must be an object, got {type(decoded).__name__}."
            raise TypeError(msg)
        if normalized != _canonical_json(decoded):
            msg = "JSON document is not in canonical form."
            raise ValueError(msg)
        return cls.from_dict(decoded)


def _reject_json_constant(value: str) -> NoReturn:
    """Reject nonstandard JSON numeric constants.

    Raises:
        ValueError: Always, because canonical JSON permits only finite numbers.
    """
    msg = f"Nonstandard JSON constant {value!r} is not supported."
    raise ValueError(msg)


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate member names.

    Returns:
        A mapping containing each source member exactly once.

    Raises:
        ValueError: If a source member occurs more than once.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"Duplicate JSON key {key!r}."
            raise ValueError(msg)
        result[key] = value
    return result


def _legacy_compact_replay_maps(
    noise_maps: Sequence[KrotovNoiseMap],
    *,
    trajectory_index: int,
) -> list[KrotovNoiseMap]:
    """Restore the archived compact-Pauli replay normalization metadata.

    The pre-Phase-II global ``NoiseModel`` sampler normalized its live forward
    state after every Pauli outcome, but omitted that normalization from the
    compact map persisted for CRN replay.  The historical provider sampler adds
    the modern final-normalization marker, so the isolated legacy boundary must
    remove it before sealing the ensemble.

    Returns:
        Fresh maps preserving every operator and diagnostic except ``normalized``.

    Raises:
        ValueError: If a map contains intermediate normalization checkpoints,
            which cannot be represented by the archived compact-map convention.
    """
    compact_maps: list[KrotovNoiseMap] = []
    for gate_index, noise_map in enumerate(noise_maps):
        if noise_map.normalization_checkpoints:
            msg = (
                "Legacy compact-map compatibility requires no normalization checkpoints, "
                f"but trajectory {trajectory_index}, gate {gate_index} has "
                f"{noise_map.normalization_checkpoints!r}."
            )
            raise ValueError(msg)
        compact_maps.append(replace(noise_map, normalized=False))
    return compact_maps


def sample_krotov_fixed_map_ensemble(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    initial_state: MPS | None,
    truncation: KrotovTruncation,
    noise_provider: GateNoiseProvider,
    tjm_options: KrotovTJMOptions,
    *,
    role: KrotovMapRole,
    resolved_seed: int,
    stage_index: int,
    stage_id: str,
    stage_configuration_checksum: str,
    circuit_checksum: str,
    provider_checksum: str,
    ensemble_index: int,
    refresh_index: int,
    global_iteration_start: int,
    legacy_linear_seed: bool = False,
    legacy_compact_replay: bool = False,
) -> KrotovFixedMapEnsemble:
    """Sample and seal one deterministic provider-backed fixed-map ensemble.

    Each trajectory gets a new :class:`numpy.random.PCG64` generator seeded by
    :func:`derive_krotov_trajectory_seed`.  ``tjm_options.random_seed`` is not
    consulted, which keeps optimizer/order configuration outside map identity.

    Args:
        circuit: Exact parameterized circuit at which maps are sampled.
        theta: Parameter vector resolving all native gate angles.
        initial_state: Fixed state-preparation input, or ``None`` for all zeros.
        truncation: Forward-sampling truncation settings.
        noise_provider: Gate-local provider sampled by the Krotov forward path.
        tjm_options: Trajectory count, time step, placement, and update settings.
        role: Reserved random-stream role.
        resolved_seed: Resolved root seed for this role.
        stage_index: Zero-based pipeline stage index.
        stage_id: Stable pipeline stage identifier.
        stage_configuration_checksum: Resolved stage checksum.
        circuit_checksum: Exact circuit checksum to bind into the ensemble.
        provider_checksum: Exact provider checksum to bind into the ensemble.
        ensemble_index: Schedule-derived ensemble index.
        refresh_index: Schedule-derived refresh index.
        global_iteration_start: First global iteration using this ensemble.
        legacy_linear_seed: Reproduce the archived linear seed formula. This is
            reserved for the isolated WP19 legacy-reproduction profile.
        legacy_compact_replay: Persist the archived CRN training maps without
            the provider's modern final-normalization marker. Historical direct
            evaluation leaves this false because it used normalized live paths.

    Returns:
        A defensively immutable, serializable fixed-map ensemble.

    Raises:
        TypeError: If a core object or provider has the wrong type.
        ValueError: If the parameter/state dimensions or metadata are invalid.
    """
    if not isinstance(circuit, ParameterizedCircuit):
        msg = f"circuit must be a ParameterizedCircuit, got {type(circuit).__name__}."
        raise TypeError(msg)
    if not isinstance(truncation, KrotovTruncation):
        msg = f"truncation must be a KrotovTruncation, got {type(truncation).__name__}."
        raise TypeError(msg)
    if not isinstance(tjm_options, KrotovTJMOptions):
        msg = f"tjm_options must be KrotovTJMOptions, got {type(tjm_options).__name__}."
        raise TypeError(msg)
    if not callable(noise_provider):
        msg = "noise_provider must be callable."
        raise TypeError(msg)
    if type(legacy_linear_seed) is not bool:
        msg = "legacy_linear_seed must be a bool."
        raise TypeError(msg)
    if type(legacy_compact_replay) is not bool:
        msg = "legacy_compact_replay must be a bool."
        raise TypeError(msg)
    if legacy_compact_replay and not legacy_linear_seed:
        msg = "legacy_compact_replay requires legacy_linear_seed=True."
        raise ValueError(msg)
    if any(gate.data_map is not None for gate in circuit.gates):
        msg = "Fixed state-preparation ensembles do not support sample-dependent gate data maps."
        raise ValueError(msg)
    theta_input = np.asarray(theta)
    if np.iscomplexobj(theta_input):
        msg = "theta must contain real values."
        raise TypeError(msg)
    try:
        resolved_theta = np.asarray(theta_input, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "theta must be convertible to float64."
        raise TypeError(msg) from error
    if resolved_theta.shape != (circuit.num_params,):
        msg = f"theta must have shape ({circuit.num_params},), got {resolved_theta.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(resolved_theta)):
        msg = "theta must contain only finite values."
        raise ValueError(msg)
    if initial_state is not None and not isinstance(initial_state, MPS):
        msg = f"initial_state must be an MPS or None, got {type(initial_state).__name__}."
        raise TypeError(msg)
    if initial_state is not None and initial_state.length != circuit.num_qubits:
        msg = f"Initial state has {initial_state.length} qubits, but the circuit has {circuit.num_qubits}."
        raise ValueError(msg)

    # Validate all externally supplied identity and binding metadata before the
    # provider is invoked, so malformed records cannot consume random streams.
    validated_role = _require_role(role)
    validated_seed = _require_uint64(resolved_seed, "resolved_seed")
    validated_stage_index = _require_uint64(stage_index, "stage_index")
    validated_stage_id = _require_identifier(stage_id, "stage_id")
    validated_stage_checksum = _require_checksum(stage_configuration_checksum, "stage_configuration_checksum")
    validated_circuit_checksum = _require_checksum(circuit_checksum, "circuit_checksum")
    validated_provider_checksum = _require_checksum(provider_checksum, "provider_checksum")
    validated_ensemble_index = _require_uint64(ensemble_index, "ensemble_index")
    validated_refresh_index = _require_uint64(refresh_index, "refresh_index")
    validated_iteration_start = _require_uint64(global_iteration_start, "global_iteration_start")

    x = np.empty(0, dtype=np.float64)
    base_state = copy.deepcopy(initial_state) if initial_state is not None else MPS(circuit.num_qubits)
    trajectory_maps: list[list[KrotovNoiseMap]] = []
    for trajectory_index in range(tjm_options.num_trajectories):
        seed = (
            derive_legacy_krotov_trajectory_seed(
                optimizer_iteration_seed=validated_seed,
                trajectory_index=trajectory_index,
            )
            if legacy_linear_seed
            else derive_krotov_trajectory_seed(
                role=validated_role,
                resolved_seed=validated_seed,
                stage_index=validated_stage_index,
                ensemble_index=validated_ensemble_index,
                trajectory_index=trajectory_index,
                refresh_index=validated_refresh_index,
            )
        )
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        trajectory = forward_tjm_trajectory(
            circuit,
            resolved_theta,
            x,
            copy.deepcopy(base_state),
            truncation,
            None,
            tjm_options,
            rng,
            noise_provider=noise_provider,
        )
        trajectory_maps.append(
            _legacy_compact_replay_maps(
                trajectory.noise_maps,
                trajectory_index=trajectory_index,
            )
            if legacy_compact_replay
            else trajectory.noise_maps
        )

    return KrotovFixedMapEnsemble(
        role=validated_role,
        resolved_seed=validated_seed,
        stage_index=validated_stage_index,
        stage_id=validated_stage_id,
        stage_configuration_checksum=validated_stage_checksum,
        circuit_checksum=validated_circuit_checksum,
        provider_checksum=validated_provider_checksum,
        ensemble_index=validated_ensemble_index,
        refresh_index=validated_refresh_index,
        global_iteration_start=validated_iteration_start,
        trajectory_maps=trajectory_maps,
    )


__all__ = [
    "KROTOV_FIXED_MAP_ENSEMBLE_IDENTITY_VERSION",
    "KROTOV_FIXED_MAP_ENSEMBLE_SCHEMA_VERSION",
    "KROTOV_LEGACY_TRAJECTORY_SEED_DERIVATION_VERSION",
    "KROTOV_MAP_ROLES",
    "KROTOV_MAP_SAMPLING_POLICIES",
    "KROTOV_TRAJECTORY_RNG_ALGORITHM",
    "KROTOV_TRAJECTORY_SEED_DERIVATION_VERSION",
    "KrotovFixedMapEnsemble",
    "KrotovMapRole",
    "KrotovMapSamplingPolicy",
    "KrotovMapSchedule",
    "KrotovMapSchedulePoint",
    "derive_krotov_trajectory_seed",
    "derive_legacy_krotov_trajectory_seed",
    "sample_krotov_fixed_map_ensemble",
]
