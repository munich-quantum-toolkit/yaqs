# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Canonical JSON and content checksums for Phase II benchmark evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, cast

from .validation import require_checksum, require_exact_keys

CONTENT_CHECKSUM_FIELD = "content_checksum"


def _reject_json_constant(value: str) -> NoReturn:
    """Reject nonstandard JSON constants.

    Raises:
        ValueError: Always, because canonical JSON contains only finite numbers.
    """
    msg = f"Nonstandard JSON constant {value!r} is not supported."
    raise ValueError(msg)


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Construct a JSON object while rejecting duplicate member names.

    Args:
        pairs: Decoded key-value pairs in source order.

    Returns:
        A mapping containing each member exactly once.

    Raises:
        ValueError: If a member name occurs more than once.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"Duplicate JSON key {key!r}."
            raise ValueError(msg)
        result[key] = value
    return result


def freeze_json(value: object, name: str = "JSON value") -> object:
    """Validate and recursively freeze a JSON-native value.

    Args:
        value: JSON-native value to validate.
        name: Human-readable location used in validation messages.

    Returns:
        An immutable detached representation.

    Raises:
        TypeError: If the tree contains an unsupported scalar, key, or container.
        ValueError: If a float is non-finite.
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
                msg = f"{name} contains a non-string mapping key of type {type(key).__name__}."
                raise TypeError(msg)
            normalized[key] = freeze_json(item, f"{name}.{key}")
        return MappingProxyType(dict(sorted(normalized.items())))
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, f"{name}[{index}]") for index, item in enumerate(value))
    msg = f"{name} contains unsupported type {type(value).__name__}."
    raise TypeError(msg)


def thaw_json(value: object) -> object:
    """Return a detached JSON-native representation of a frozen value.

    Args:
        value: Frozen JSON tree.

    Returns:
        A tree containing dictionaries, lists, and JSON scalar values.
    """
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [thaw_json(item) for item in value]
    return value


def freeze_json_mapping(value: object, name: str) -> Mapping[str, object]:
    """Validate and freeze a JSON object.

    Args:
        value: Candidate top-level mapping.
        name: Human-readable location used in validation messages.

    Returns:
        A recursively frozen string-keyed mapping.

    Raises:
        TypeError: If ``value`` is not a mapping.
    """
    if not isinstance(value, Mapping):
        msg = f"{name} must be a mapping, got {type(value).__name__}."
        raise TypeError(msg)
    return cast("Mapping[str, object]", freeze_json(value, name))


def thaw_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return a detached mutable dictionary from a frozen JSON mapping.

    Args:
        value: Frozen mapping.

    Returns:
        A detached JSON-native dictionary.
    """
    return cast("dict[str, object]", thaw_json(value))


def canonical_json(value: object) -> str:
    """Serialize a JSON-native value deterministically.

    Args:
        value: JSON-native value to serialize.

    Returns:
        UTF-8-compatible canonical JSON text.
    """
    normalized = thaw_json(freeze_json(value))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_checksum(value: object) -> str:
    """Return the prefixed SHA-256 digest of canonical JSON content.

    Args:
        value: JSON-native value whose content is identified.

    Returns:
        A ``sha256:``-prefixed lowercase digest.
    """
    digest = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return f"sha256:{digest}"


def seal_mapping(payload: Mapping[str, object]) -> dict[str, object]:
    """Attach a checksum covering every supplied field.

    Args:
        payload: Unsealed document payload.

    Returns:
        A detached mapping with ``content_checksum`` appended.

    Raises:
        ValueError: If the payload already contains ``content_checksum``.
    """
    if CONTENT_CHECKSUM_FIELD in payload:
        msg = f"{CONTENT_CHECKSUM_FIELD!r} must not be present before sealing."
        raise ValueError(msg)
    detached = thaw_json_mapping(freeze_json_mapping(payload, "payload"))
    detached[CONTENT_CHECKSUM_FIELD] = canonical_checksum(detached)
    return detached


def verify_sealed_mapping(
    document: object,
    *,
    expected_keys: frozenset[str],
    name: str,
) -> Mapping[str, object]:
    """Validate exact fields and verify a sealed document checksum.

    Args:
        document: Candidate sealed document.
        expected_keys: Complete versioned field set, including the checksum.
        name: Human-readable document name.

    Returns:
        A recursively frozen verified mapping.

    Raises:
        ValueError: If fields or the checksum differ.
    """
    frozen = freeze_json_mapping(document, name)
    require_exact_keys(frozen, expected_keys, name)
    expected_checksum = require_checksum(frozen[CONTENT_CHECKSUM_FIELD], f"{name}.{CONTENT_CHECKSUM_FIELD}")
    payload = {key: thaw_json(value) for key, value in frozen.items() if key != CONTENT_CHECKSUM_FIELD}
    actual_checksum = canonical_checksum(payload)
    if expected_checksum != actual_checksum:
        msg = f"{name} content checksum mismatch: expected {expected_checksum}, computed {actual_checksum}."
        raise ValueError(msg)
    return frozen


def load_canonical_json_object(payload: str) -> Mapping[str, object]:
    """Decode a canonical JSON object with at most one trailing newline.

    Args:
        payload: Canonical JSON text.

    Returns:
        A recursively frozen mapping.

    Raises:
        TypeError: If ``payload`` is not text or the top level is not an object.
        ValueError: If parsing fails, duplicate keys exist, or the text is not canonical.
    """
    if type(payload) is not str:
        msg = f"payload must be a string, got {type(payload).__name__}."
        raise TypeError(msg)
    normalized_payload = payload.removesuffix("\n")
    try:
        decoded = json.loads(
            normalized_payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        msg = f"Could not decode canonical JSON: {error}."
        raise ValueError(msg) from error
    if not isinstance(decoded, Mapping):
        msg = f"Canonical JSON top level must be an object, got {type(decoded).__name__}."
        raise TypeError(msg)
    frozen = freeze_json_mapping(decoded, "canonical JSON document")
    if normalized_payload != canonical_json(frozen):
        msg = "JSON document is not in canonical form."
        raise ValueError(msg)
    return frozen


def read_canonical_json_object(path: Path) -> Mapping[str, object]:
    """Read one canonical JSON object from disk.

    Args:
        path: File to read as UTF-8.

    Returns:
        A recursively frozen mapping.

    Raises:
        TypeError: If ``path`` is not a :class:`~pathlib.Path`.
        ValueError: If the file is unreadable, invalid UTF-8, or noncanonical.
    """
    if not isinstance(path, Path):
        msg = f"path must be a pathlib.Path, got {type(path).__name__}."
        raise TypeError(msg)
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        msg = f"Could not read canonical JSON document {path}: {error}."
        raise ValueError(msg) from error
    return load_canonical_json_object(payload)


__all__ = [
    "CONTENT_CHECKSUM_FIELD",
    "canonical_checksum",
    "canonical_json",
    "freeze_json",
    "freeze_json_mapping",
    "load_canonical_json_object",
    "read_canonical_json_object",
    "seal_mapping",
    "thaw_json",
    "thaw_json_mapping",
    "verify_sealed_mapping",
]
