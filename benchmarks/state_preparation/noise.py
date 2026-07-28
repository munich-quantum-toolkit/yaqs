# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact state-independent noise samplers for state-preparation benchmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np

from mqt.yaqs.core.libraries.noise_library import PauliX, PauliY, PauliZ
from mqt.yaqs.optimization import LocalOperator

PauliDistribution: TypeAlias = Mapping[str, Real]

_PAULI_LABELS = ("I", "X", "Y", "Z")
_PAULI_MATRICES = {
    "X": PauliX.matrix,
    "Y": PauliY.matrix,
    "Z": PauliZ.matrix,
}
_NORMALIZATION_ATOL = 1e-12


def _validate_site(site: object, name: str) -> int:
    """Validate and normalize one qubit site.

    Args:
        site: Site index to validate.
        name: Argument name used in validation errors.

    Returns:
        The normalized built-in integer.

    Raises:
        TypeError: If the site is not an integer or is a Boolean.
        ValueError: If the site is negative.
    """
    if isinstance(site, (bool, np.bool_)) or not isinstance(site, Integral):
        msg = f"{name} must be a nonnegative integer, got {type(site).__name__}."
        raise TypeError(msg)
    normalized = int(site)
    if normalized < 0:
        msg = f"{name} must be nonnegative, got {normalized}."
        raise ValueError(msg)
    return normalized


def _validate_distribution(distribution: object, name: str) -> tuple[float, float, float, float]:
    """Validate a local Pauli probability distribution.

    Missing canonical labels represent zero-probability outcomes. Labels are
    otherwise strict and case-sensitive so configuration mistakes cannot
    silently change a channel.

    Args:
        distribution: Mapping from ``I``, ``X``, ``Y``, and ``Z`` to probabilities.
        name: Argument name used in validation errors.

    Returns:
        Probabilities in canonical ``I, X, Y, Z`` order.

    Raises:
        TypeError: If the distribution or one of its probabilities has an
            unsupported type.
        ValueError: If a label or probability is invalid, or probabilities do
            not sum to one.
    """
    if not isinstance(distribution, Mapping):
        msg = f"{name} must be a mapping from Pauli labels to probabilities."
        raise TypeError(msg)

    labels = tuple(distribution)
    non_string_labels = tuple(label for label in labels if not isinstance(label, str))
    if non_string_labels:
        msg = f"{name} labels must be strings, got {non_string_labels!r}."
        raise TypeError(msg)
    unknown_labels = tuple(label for label in labels if label not in _PAULI_LABELS)
    if unknown_labels:
        msg = f"{name} contains unknown Pauli labels {unknown_labels!r}; expected only I, X, Y, and Z."
        raise ValueError(msg)

    probabilities: list[float] = []
    for label in _PAULI_LABELS:
        value = distribution.get(label, 0.0)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            msg = f"{name}[{label!r}] must be a finite real probability, got {type(value).__name__}."
            raise TypeError(msg)
        try:
            probability = float(value)
        except OverflowError as error:
            msg = f"{name}[{label!r}] must lie in [0, 1], got an overflowing real value."
            raise ValueError(msg) from error
        if not np.isfinite(probability):
            msg = f"{name}[{label!r}] must be finite, got {probability!r}."
            raise ValueError(msg)
        if probability < 0.0 or probability > 1.0:
            msg = f"{name}[{label!r}] must lie in [0, 1], got {probability!r}."
            raise ValueError(msg)
        probabilities.append(probability)

    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_NORMALIZATION_ATOL):
        msg = f"{name} probabilities must sum to one, got {total!r}."
        raise ValueError(msg)
    return probabilities[0], probabilities[1], probabilities[2], probabilities[3]


def _draw_uniform(rng: np.random.Generator) -> float:
    """Draw and validate one scalar from a NumPy generator.

    Args:
        rng: Random-number generator providing a scalar ``random`` draw.

    Returns:
        A finite draw in the half-open interval ``[0, 1)``.

    Raises:
        TypeError: If the generator returns a non-real or Boolean value.
        ValueError: If the draw is non-finite or outside ``[0, 1)``.
    """
    value = rng.random()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        msg = f"rng.random() must return a real scalar, got {type(value).__name__}."
        raise TypeError(msg)
    try:
        draw = float(value)
    except OverflowError as error:
        msg = "rng.random() must return a finite value in [0, 1), got an overflowing real value."
        raise ValueError(msg) from error
    if not np.isfinite(draw) or draw < 0.0 or draw >= 1.0:
        msg = f"rng.random() must return a finite value in [0, 1), got {draw!r}."
        raise ValueError(msg)
    return draw


def sample_local_pauli(
    distribution: PauliDistribution,
    site: int,
    rng: np.random.Generator,
) -> LocalOperator | None:
    """Sample one state-independent local Pauli outcome.

    The four half-open sampling intervals follow canonical ``I, X, Y, Z``
    order, independent of mapping insertion order. Exactly one random number is
    consumed even for a deterministic distribution. Identity outcomes return
    ``None``; non-identity outcomes contain a bare unitary Pauli matrix without
    a probability weight.

    Args:
        distribution: Local Pauli probabilities. Missing canonical labels have
            probability zero.
        site: Nonnegative qubit site.
        rng: Trajectory-local NumPy random-number generator.

    Returns:
        A one-site Pauli operator, or ``None`` for the identity outcome.
    """
    probabilities = _validate_distribution(distribution, "distribution")
    validated_site = _validate_site(site, "site")
    draw = _draw_uniform(rng)

    cumulative = 0.0
    selected_label: str | None = None
    for label, probability in zip(_PAULI_LABELS, probabilities, strict=True):
        cumulative = math.fsum((cumulative, probability))
        if draw < cumulative:
            selected_label = label
            break

    if selected_label is None:
        # A distribution accepted within the normalization tolerance may end a
        # few ulps below one. Preserve zero-probability branches by assigning
        # that numerical gap to the final strictly positive outcome.
        selected_label = next(
            label
            for label, probability in reversed(tuple(zip(_PAULI_LABELS, probabilities, strict=True)))
            if probability > 0.0
        )

    if selected_label == "I":
        return None
    return LocalOperator(_PAULI_MATRICES[selected_label], (validated_site,), label=selected_label)


def sample_product_pauli_channel(
    first_site: int,
    second_site: int,
    first_distribution: PauliDistribution,
    second_distribution: PauliDistribution,
    rng: np.random.Generator,
) -> tuple[LocalOperator, ...]:
    """Sample two independent local Pauli channels.

    Both inputs are validated before the generator is consumed. The two local
    samplers then draw sequentially from the same trajectory-local generator.
    Identity outcomes are omitted, while non-identity operators retain site-call
    order.

    Args:
        first_site: Site of the first local channel.
        second_site: Distinct site of the second local channel.
        first_distribution: Pauli probabilities for ``first_site``.
        second_distribution: Pauli probabilities for ``second_site``.
        rng: Trajectory-local NumPy random-number generator.

    Returns:
        Zero, one, or two one-site Pauli operators.

    Raises:
        ValueError: If both channels target the same site.
    """
    validated_first_site = _validate_site(first_site, "first_site")
    validated_second_site = _validate_site(second_site, "second_site")
    if validated_first_site == validated_second_site:
        msg = f"Product-Pauli channel sites must be distinct, got {validated_first_site} twice."
        raise ValueError(msg)
    _validate_distribution(first_distribution, "first_distribution")
    _validate_distribution(second_distribution, "second_distribution")

    first_operator = sample_local_pauli(first_distribution, validated_first_site, rng)
    second_operator = sample_local_pauli(second_distribution, validated_second_site, rng)

    operators: list[LocalOperator] = []
    if first_operator is not None:
        operators.append(first_operator)
    if second_operator is not None:
        operators.append(second_operator)
    return tuple(operators)


__all__ = [
    "PauliDistribution",
    "sample_local_pauli",
    "sample_product_pauli_channel",
]
