# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Strict scalar and structural validators for Phase II evidence schemas."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import NoReturn, cast

_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_GIT_BLOB_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _raise_type_error(name: str, expected: str, value: object) -> NoReturn:
    """Raise a consistently formatted strict type error.

    Raises:
        TypeError: Always.
    """
    msg = f"{name} must be {expected}; received {type(value).__name__}."
    raise TypeError(msg)


def require_exact_keys(value: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    """Require a mapping to contain exactly a versioned field set.

    Args:
        value: Mapping to inspect.
        expected: Complete expected key set.
        name: Human-readable mapping name.

    Raises:
        ValueError: If fields are missing or unsupported.
    """
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        msg = f"{name} fields do not match the schema: missing={missing!r}, extra={extra!r}."
        raise ValueError(msg)


def require_mapping(value: object, name: str) -> Mapping[str, object]:
    """Require a string-keyed mapping.

    Args:
        value: Candidate mapping.
        name: Human-readable argument name.

    Returns:
        The validated mapping.

    Raises:
        TypeError: If ``value`` is not a mapping or contains a non-string key.
    """
    if not isinstance(value, Mapping):
        _raise_type_error(name, "a mapping", value)
    if any(type(key) is not str for key in value):
        msg = f"{name} keys must all be strings."
        raise TypeError(msg)
    return cast("Mapping[str, object]", value)


def require_string(value: object, name: str) -> str:
    """Require nonempty text without surrounding or control whitespace.

    Args:
        value: Candidate string.
        name: Human-readable argument name.

    Returns:
        The validated text.

    Raises:
        ValueError: If the text is empty or contains surrounding or control whitespace.
    """
    if type(value) is not str:
        _raise_type_error(name, "a string", value)
    if not value or value != value.strip() or any(character.isspace() and character != " " for character in value):
        msg = f"{name} must be nonempty text without surrounding or control whitespace."
        raise ValueError(msg)
    return value


def require_nonempty_text(value: object, name: str) -> str:
    """Require text containing at least one non-whitespace character.

    Args:
        value: Candidate text.
        name: Human-readable argument name.

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


def require_slug(value: object, name: str) -> str:
    """Require a stable lowercase identifier.

    Args:
        value: Candidate identifier.
        name: Human-readable argument name.

    Returns:
        The validated identifier.

    Raises:
        ValueError: If the identifier does not follow the stable slug format.
    """
    text = require_string(value, name)
    if _SLUG_PATTERN.fullmatch(text) is None:
        msg = f"{name} must be a lowercase identifier containing only letters, digits, '.', '_', or '-'."
        raise ValueError(msg)
    return text


def require_bool(value: object, name: str) -> bool:
    """Require an exact Boolean value.

    Args:
        value: Candidate Boolean.
        name: Human-readable argument name.

    Returns:
        The validated Boolean.
    """
    if type(value) is not bool:
        _raise_type_error(name, "a bool", value)
    return value


def require_int(value: object, name: str, *, minimum: int = 0) -> int:
    """Require an exact integer at or above a minimum.

    Args:
        value: Candidate integer.
        name: Human-readable argument name.
        minimum: Inclusive lower bound.

    Returns:
        The validated integer.

    Raises:
        ValueError: If the integer is below ``minimum``.
    """
    if type(value) is not int:
        _raise_type_error(name, "an int", value)
    if value < minimum:
        msg = f"{name} must be at least {minimum}."
        raise ValueError(msg)
    return value


def require_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Require an exact finite float within optional inclusive bounds.

    Args:
        value: Candidate float.
        name: Human-readable argument name.
        minimum: Optional inclusive lower bound.
        maximum: Optional inclusive upper bound.

    Returns:
        The validated float, normalizing negative zero.

    Raises:
        ValueError: If the float is non-finite or outside the requested bounds.
    """
    if type(value) is not float:
        _raise_type_error(name, "a float", value)
    if not math.isfinite(value):
        msg = f"{name} must be finite."
        raise ValueError(msg)
    if minimum is not None and value < minimum:
        msg = f"{name} must be at least {minimum}."
        raise ValueError(msg)
    if maximum is not None and value > maximum:
        msg = f"{name} must be at most {maximum}."
        raise ValueError(msg)
    return value or 0.0


def require_checksum(value: object, name: str) -> str:
    """Require a prefixed lowercase SHA-256 digest.

    Args:
        value: Candidate checksum.
        name: Human-readable argument name.

    Returns:
        The validated checksum.

    Raises:
        ValueError: If the checksum is not a prefixed lowercase SHA-256 digest.
    """
    text = require_string(value, name)
    if _SHA256_PATTERN.fullmatch(text) is None:
        msg = f"{name} must have the form 'sha256:' followed by 64 lowercase hexadecimal characters."
        raise ValueError(msg)
    return text


def require_git_commit(value: object, name: str = "git_commit") -> str:
    """Require a complete lowercase SHA-1 Git commit identifier.

    Args:
        value: Candidate object identifier.
        name: Human-readable argument name.

    Returns:
        The validated commit identifier.

    Raises:
        ValueError: If the value is not a complete lowercase SHA-1 commit identifier.
    """
    text = require_string(value, name)
    if _GIT_COMMIT_PATTERN.fullmatch(text) is None:
        msg = f"{name} must be a complete 40-character lowercase hexadecimal Git commit."
        raise ValueError(msg)
    return text


def require_git_blob(value: object, name: str = "git_blob_id") -> str:
    """Require a complete lowercase SHA-1 Git blob identifier.

    Args:
        value: Candidate object identifier.
        name: Human-readable argument name.

    Returns:
        The validated blob identifier.

    Raises:
        ValueError: If the value is not a complete lowercase SHA-1 blob identifier.
    """
    text = require_string(value, name)
    if _GIT_BLOB_PATTERN.fullmatch(text) is None:
        msg = f"{name} must be a complete 40-character lowercase hexadecimal Git blob."
        raise ValueError(msg)
    return text


def require_relative_path(value: object, name: str) -> str:
    """Require a normalized relative POSIX path without traversal.

    Args:
        value: Candidate path spelling.
        name: Human-readable argument name.

    Returns:
        The validated path.

    Raises:
        ValueError: If the path is absolute, non-normalized, or contains traversal.
    """
    text = require_string(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or "\\" in text or any(part in {"", ".", ".."} for part in text.split("/")):
        msg = f"{name} must be a normalized relative POSIX path without traversal."
        raise ValueError(msg)
    return text


def require_string_sequence(
    value: object,
    name: str,
    *,
    minimum_length: int = 0,
    unique: bool = False,
    slugs: bool = False,
) -> tuple[str, ...]:
    """Require an immutable sequence of validated strings.

    Args:
        value: Candidate sequence.
        name: Human-readable argument name.
        minimum_length: Minimum number of members.
        unique: Whether duplicate values are forbidden.
        slugs: Whether each member must be a stable identifier.

    Returns:
        The validated tuple.

    Raises:
        ValueError: If the sequence is too short, contains duplicates, or has an invalid member.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _raise_type_error(name, "a sequence", value)
    validator = require_slug if slugs else require_string
    result = tuple(validator(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(result) < minimum_length:
        msg = f"{name} must contain at least {minimum_length} items."
        raise ValueError(msg)
    if unique and len(result) != len(set(result)):
        msg = f"{name} must not contain duplicates."
        raise ValueError(msg)
    return result


__all__ = [
    "require_bool",
    "require_checksum",
    "require_exact_keys",
    "require_float",
    "require_git_blob",
    "require_git_commit",
    "require_int",
    "require_mapping",
    "require_nonempty_text",
    "require_relative_path",
    "require_slug",
    "require_string",
    "require_string_sequence",
]
