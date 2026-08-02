# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""State-independent noise definitions and samplers for state-preparation benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import starmap
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

import numpy as np

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.libraries.noise_library import PauliX, PauliY, PauliZ
from mqt.yaqs.optimization import GateNoiseContext, LocalOperator, TJMNoiseInstruction

from .constants import STANDARD_NOISE_IDS

PauliDistribution: TypeAlias = Mapping[str, Real]
_StandardNoiseFamily: TypeAlias = Literal["dephasing", "depolarizing"]
_StandardNoiseSiteSupport: TypeAlias = Literal["single_site", "two_site", "single_site_and_two_site"]
_StandardNoiseGatePlacement: TypeAlias = Literal["single_qubit_gates", "multi_qubit_gates", "all_gates"]
_ProcessTemplate: TypeAlias = tuple[str, tuple[int, ...], float]

_PAULI_LABELS = ("I", "X", "Y", "Z")
_PAULI_MATRICES = {
    "X": PauliX.matrix,
    "Y": PauliY.matrix,
    "Z": PauliZ.matrix,
}
_NORMALIZATION_ATOL = 1e-12

STANDARD_ONE_QUBIT_GATE_STRENGTH = 6.4e-4
STANDARD_TWO_QUBIT_GATE_STRENGTH = 5.1e-3
STANDARD_NOISE_STRENGTH_INTERPRETATION = "per_jump_operator"
FIXED_RATE_NOISE_DEFINITION_VERSION = "yaqs.state_preparation.noise.v1"
HISTORICAL_FIXED_RATE_NOISE_ID = "ibm_inspired_pauli_legacy_v1"
_LOGICAL_PARAMETERIZED_GATE_PLACEMENT = "logical_parameterized_gates"
_HISTORICAL_ONE_QUBIT_PROCESS_STRENGTH = 3.0e-4 / 3.0
_HISTORICAL_TWO_QUBIT_PROCESS_STRENGTH = 1.5e-3
_HISTORICAL_TJM_DT = 1.0
TWO_SITE_DEPOLARIZING_OPERATORS = (
    "XX",
    "XY",
    "XZ",
    "YX",
    "YY",
    "YZ",
    "ZX",
    "ZY",
    "ZZ",
)


@dataclass(frozen=True, slots=True)
class _StandardNoiseShape:
    """Internal structural axes for one standard-noise identifier."""

    family: _StandardNoiseFamily
    site_support: _StandardNoiseSiteSupport
    gate_placement: _StandardNoiseGatePlacement


_STANDARD_NOISE_SHAPES: Mapping[str, _StandardNoiseShape] = MappingProxyType({
    "dephasing_1s_1q": _StandardNoiseShape("dephasing", "single_site", "single_qubit_gates"),
    "dephasing_1s_2q": _StandardNoiseShape("dephasing", "single_site", "multi_qubit_gates"),
    "dephasing_1s_all": _StandardNoiseShape("dephasing", "single_site", "all_gates"),
    "dephasing_2s_2q": _StandardNoiseShape("dephasing", "two_site", "multi_qubit_gates"),
    "dephasing_1s2s_all": _StandardNoiseShape("dephasing", "single_site_and_two_site", "all_gates"),
    "depolarizing_1s_1q": _StandardNoiseShape("depolarizing", "single_site", "single_qubit_gates"),
    "depolarizing_1s_2q": _StandardNoiseShape("depolarizing", "single_site", "multi_qubit_gates"),
    "depolarizing_1s_all": _StandardNoiseShape("depolarizing", "single_site", "all_gates"),
    "depolarizing_2s_2q": _StandardNoiseShape("depolarizing", "two_site", "multi_qubit_gates"),
    "depolarizing_1s2s_all": _StandardNoiseShape("depolarizing", "single_site_and_two_site", "all_gates"),
})
if tuple(_STANDARD_NOISE_SHAPES) != STANDARD_NOISE_IDS:
    msg = "Standard-noise registry shapes must exactly follow STANDARD_NOISE_IDS."
    raise RuntimeError(msg)


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
    distribution_mapping = cast("Mapping[object, object]", distribution)

    labels = tuple(distribution_mapping)
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
        value = distribution_mapping.get(label, 0.0)
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


def _strict_serialized_equal(left: object, right: object) -> bool:
    """Compare serialized values without scalar type coercion.

    Returns:
        Whether both JSON-native trees have identical container and scalar
        types as well as equal values.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_mapping = cast("Mapping[str, object]", left)
        right_mapping = cast("Mapping[str, object]", right)
        if left_mapping.keys() != right_mapping.keys():
            return False
        return all(_strict_serialized_equal(left_mapping[key], right_mapping[key]) for key in left_mapping)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(starmap(_strict_serialized_equal, zip(left, right, strict=True)))
    if left is None:
        return True
    if type(left) in {bool, int, float, str}:
        return bool(left == right)
    return False


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


@dataclass(frozen=True, slots=True)
class StandardNoiseDefinition:
    """One immutable standard state-preparation noise definition.

    The definition freezes the noise family, support axis, and gate-placement
    axis associated with its benchmark identifier. Its serialized process
    templates use relative gate-site positions, so they remain independent of
    any particular circuit.

    Attributes:
        noise_id: Standard benchmark noise identifier.
        family: Dephasing or depolarizing process family.
        site_support: Single-site, two-site, or combined support.
        gate_placement: One-qubit, two-qubit, or all-gate placement.
    """

    noise_id: str
    family: _StandardNoiseFamily
    site_support: _StandardNoiseSiteSupport
    gate_placement: _StandardNoiseGatePlacement

    def __post_init__(self) -> None:
        """Validate that all axes exactly match the frozen identifier.

        Raises:
            TypeError: If a field is not a string.
            ValueError: If the identifier is not standard or its axes do not
                match the registry.
        """
        fields = {
            "noise_id": self.noise_id,
            "family": self.family,
            "site_support": self.site_support,
            "gate_placement": self.gate_placement,
        }
        for name, value in fields.items():
            if type(value) is not str:
                msg = f"{name} must be a string, got {type(value).__name__}."
                raise TypeError(msg)

        shape = _STANDARD_NOISE_SHAPES.get(self.noise_id)
        if shape is None:
            msg = f"Unknown standard noise identifier {self.noise_id!r}."
            raise ValueError(msg)
        actual = (self.family, self.site_support, self.gate_placement)
        expected = (shape.family, shape.site_support, shape.gate_placement)
        if actual != expected:
            msg = (
                f"Definition axes {actual!r} do not match standard noise identifier "
                f"{self.noise_id!r}, which requires {expected!r}."
            )
            raise ValueError(msg)

    @property
    def single_site_operators(self) -> tuple[str, ...]:
        """The frozen one-site Pauli labels for this family."""
        if self.site_support == "two_site":
            return ()
        if self.family == "dephasing":
            return ("Z",)
        return ("X", "Y", "Z")

    @property
    def two_site_operators(self) -> tuple[str, ...]:
        """The frozen strictly two-sided Pauli labels."""
        if self.site_support == "single_site":
            return ()
        if self.family == "dephasing":
            return ("ZZ",)
        return TWO_SITE_DEPOLARIZING_OPERATORS

    @property
    def applies_after_one_qubit_gates(self) -> bool:
        """Whether this definition has a one-qubit-gate component."""
        return self.gate_placement in {"single_qubit_gates", "all_gates"}

    @property
    def applies_after_two_qubit_gates(self) -> bool:
        """Whether this definition has a two-qubit-gate component."""
        return self.gate_placement in {"multi_qubit_gates", "all_gates"}

    def process_templates(self, arity: int) -> tuple[_ProcessTemplate, ...]:
        """Return ordered process templates for one supported gate arity.

        Returns:
            Process name, relative support, and per-operator strength tuples.

        Raises:
            TypeError: If ``arity`` is not an integer or is a Boolean.
            ValueError: If ``arity`` is not one or two.
        """
        if isinstance(arity, (bool, np.bool_)) or not isinstance(arity, Integral):
            msg = f"arity must be an integer, got {type(arity).__name__}."
            raise TypeError(msg)
        normalized_arity = int(arity)
        if normalized_arity == 1:
            if not self.applies_after_one_qubit_gates:
                return ()
            return tuple(
                (
                    f"pauli_{label.lower()}",
                    (0,),
                    STANDARD_ONE_QUBIT_GATE_STRENGTH,
                )
                for label in self.single_site_operators
            )
        if normalized_arity == 2:
            if not self.applies_after_two_qubit_gates:
                return ()
            single_site = tuple(
                (
                    f"pauli_{label.lower()}",
                    (relative_site,),
                    STANDARD_TWO_QUBIT_GATE_STRENGTH,
                )
                for relative_site in (0, 1)
                for label in self.single_site_operators
            )
            two_site = tuple(
                (
                    f"crosstalk_{label.lower()}",
                    (0, 1),
                    STANDARD_TWO_QUBIT_GATE_STRENGTH,
                )
                for label in self.two_site_operators
            )
            return single_site + two_site
        msg = f"Standard noise definitions support only gate arities one and two, got {normalized_arity}."
        raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-native registry definition."""

        def serialize_templates(arity: int) -> list[dict[str, object]]:
            return [
                {
                    "name": name,
                    "relative_sites": list(relative_sites),
                    "strength": strength,
                }
                for name, relative_sites, strength in self.process_templates(arity)
            ]

        return {
            "noise_id": self.noise_id,
            "family": self.family,
            "site_support": self.site_support,
            "gate_placement": self.gate_placement,
            "strength_interpretation": STANDARD_NOISE_STRENGTH_INTERPRETATION,
            "one_qubit_gate_processes": serialize_templates(1),
            "two_qubit_gate_processes": serialize_templates(2),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> StandardNoiseDefinition:
        """Restore and strictly verify a serialized registry definition.

        Returns:
            The canonical immutable registry entry.

        Raises:
            TypeError: If ``data`` is not a mapping or an axis has the wrong
                scalar type.
            ValueError: If keys or derived metadata differ from the registry.
        """
        if not isinstance(data, Mapping):
            msg = "StandardNoiseDefinition data must be a mapping."
            raise TypeError(msg)
        expected_keys = {
            "noise_id",
            "family",
            "site_support",
            "gate_placement",
            "strength_interpretation",
            "one_qubit_gate_processes",
            "two_qubit_gate_processes",
        }
        if set(data) != expected_keys:
            missing = sorted(expected_keys - set(data), key=repr)
            extra = sorted(set(data) - expected_keys, key=repr)
            msg = f"StandardNoiseDefinition keys mismatch: missing={missing!r}, extra={extra!r}."
            raise ValueError(msg)

        candidate = cls(
            noise_id=cast("str", data["noise_id"]),
            family=cast("_StandardNoiseFamily", data["family"]),
            site_support=cast("_StandardNoiseSiteSupport", data["site_support"]),
            gate_placement=cast("_StandardNoiseGatePlacement", data["gate_placement"]),
        )
        if not _strict_serialized_equal(dict(data), candidate.to_dict()):
            msg = f"Serialized definition for {candidate.noise_id!r} does not match the frozen registry."
            raise ValueError(msg)
        return get_standard_noise_definition(candidate.noise_id)


STANDARD_NOISE_REGISTRY: Mapping[str, StandardNoiseDefinition] = MappingProxyType({
    noise_id: StandardNoiseDefinition(
        noise_id=noise_id,
        family=shape.family,
        site_support=shape.site_support,
        gate_placement=shape.gate_placement,
    )
    for noise_id, shape in _STANDARD_NOISE_SHAPES.items()
})


def get_standard_noise_definition(noise_id: str) -> StandardNoiseDefinition:
    """Look up one validated standard-noise definition.

    Args:
        noise_id: Exact standard benchmark identifier.

    Returns:
        The immutable registry definition.

    Raises:
        TypeError: If ``noise_id`` is not a string.
        ValueError: If it is unknown or belongs to a non-standard channel.
    """
    if type(noise_id) is not str:
        msg = f"noise_id must be a string, got {type(noise_id).__name__}."
        raise TypeError(msg)
    try:
        return STANDARD_NOISE_REGISTRY[noise_id]
    except KeyError as error:
        msg = f"Unknown standard noise identifier {noise_id!r}."
        raise ValueError(msg) from error


@dataclass(frozen=True, slots=True)
class StandardNoiseProvider:
    """Callable gate-local factory for one standard-noise definition."""

    definition: StandardNoiseDefinition

    def __post_init__(self) -> None:
        """Validate and canonicalize the immutable provider definition.

        Raises:
            TypeError: If ``definition`` is not a standard-noise definition.
        """
        if type(self.definition) is not StandardNoiseDefinition:
            msg = f"definition must be a StandardNoiseDefinition, got {type(self.definition).__name__}."
            raise TypeError(msg)
        canonical = get_standard_noise_definition(self.definition.noise_id)
        object.__setattr__(self, "definition", canonical)

    @property
    def noise_id(self) -> str:
        """The selected standard benchmark identifier."""
        return self.definition.noise_id

    def to_dict(self) -> dict[str, object]:
        """Return the selected standard-noise definition as JSON-native data."""
        return self.definition.to_dict()

    def __call__(
        self,
        context: GateNoiseContext,
        rng: np.random.Generator,
        /,
    ) -> TJMNoiseInstruction | None:
        """Build a fresh gate-local TJM instruction.

        The provider itself consumes no random numbers. Circuit-TJM samples the
        returned model afterward using the same trajectory-local generator.

        Args:
            context: Immutable metadata for the current one- or two-qubit gate.
            rng: Trajectory-local generator reserved for downstream TJM
                sampling.

        Returns:
            A tagged fresh local model, or ``None`` when this identifier excludes
            the current gate arity.

        Raises:
            TypeError: If ``context`` is not a gate-noise context.
        """
        del rng
        if not isinstance(context, GateNoiseContext):
            msg = f"context must be a GateNoiseContext, got {type(context).__name__}."
            raise TypeError(msg)

        templates = self.definition.process_templates(context.arity)
        if not templates:
            return None
        processes = [
            {
                "name": name,
                "sites": [context.sites[index] for index in relative_sites],
                "strength": strength,
            }
            for name, relative_sites, strength in templates
        ]
        noise_model = NoiseModel(processes)
        for process in noise_model.processes:
            if "matrix" in process:
                process["matrix"] = np.array(process["matrix"], dtype=np.complex128, copy=True)
            if "factors" in process:
                process["factors"] = tuple(
                    np.array(factor, dtype=np.complex128, copy=True) for factor in process["factors"]
                )
        return TJMNoiseInstruction(
            noise_model,
            channel_id=self.noise_id,
        )


def create_standard_noise_provider(noise_id: str) -> StandardNoiseProvider:
    """Create a validated provider for one standard benchmark identifier.

    Args:
        noise_id: Exact standard-noise registry key.

    Returns:
        An immutable provider that constructs fresh gate-local models.
    """
    return StandardNoiseProvider(get_standard_noise_definition(noise_id))


def _validate_positive_finite_scale(value: object) -> float:
    """Return one canonical positive finite noise-strength scale.

    Returns:
        The scale as a built-in float.

    Raises:
        TypeError: If the scale is not a real scalar or is a Boolean.
        ValueError: If the scale is non-finite or not strictly positive.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        msg = f"strength_scale must be a real scalar, got {type(value).__name__}."
        raise TypeError(msg)
    try:
        scale = float(value)
    except OverflowError as error:
        msg = "strength_scale must be finite."
        raise ValueError(msg) from error
    if not math.isfinite(scale) or scale <= 0.0:
        msg = f"strength_scale must be positive and finite, got {scale!r}."
        raise ValueError(msg)
    return scale


def _provider_content_checksum(payload: Mapping[str, object]) -> str:
    """Return a deterministic checksum for JSON-native provider metadata.

    Returns:
        A ``sha256:``-prefixed checksum.
    """
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def _fresh_scaled_instruction(
    templates: tuple[_ProcessTemplate, ...],
    context: GateNoiseContext,
    strength_scale: float,
    channel_id: str,
) -> TJMNoiseInstruction | None:
    """Build one detached gate-local model from relative process templates.

    Returns:
        A fresh TJM instruction, or ``None`` for an excluded gate arity.
    """
    if not templates:
        return None
    processes = [
        {
            "name": name,
            "sites": [context.sites[index] for index in relative_sites],
            "strength": strength * strength_scale,
        }
        for name, relative_sites, strength in templates
    ]
    noise_model = NoiseModel(processes)
    for process in noise_model.processes:
        if "matrix" in process:
            process["matrix"] = np.array(process["matrix"], dtype=np.complex128, copy=True)
        if "factors" in process:
            process["factors"] = tuple(
                np.array(factor, dtype=np.complex128, copy=True) for factor in process["factors"]
            )
    return TJMNoiseInstruction(noise_model, channel_id=channel_id)


@dataclass(frozen=True, slots=True)
class ScaledStandardNoiseProvider:
    """Immutable logical-gate provider for a scaled standard fixed-rate profile.

    Scaling creates fresh gate-local models and never mutates the canonical
    :data:`STANDARD_NOISE_REGISTRY`. The time step remains a separate stage
    setting; this provider scales only the per-jump-operator strengths.

    Attributes:
        definition: Canonical base standard-noise definition.
        strength_scale: Strictly positive finite multiplier for every strength.
    """

    definition: StandardNoiseDefinition
    strength_scale: float

    def __post_init__(self) -> None:
        """Canonicalize the base definition and validate the scale.

        Raises:
            TypeError: If the base definition or scale has an invalid type.
        """
        if type(self.definition) is not StandardNoiseDefinition:
            msg = f"definition must be a StandardNoiseDefinition, got {type(self.definition).__name__}."
            raise TypeError(msg)
        object.__setattr__(self, "definition", get_standard_noise_definition(self.definition.noise_id))
        object.__setattr__(self, "strength_scale", _validate_positive_finite_scale(self.strength_scale))

    @property
    def base_noise_id(self) -> str:
        """The canonical unscaled registry identifier."""
        return self.definition.noise_id

    @property
    def noise_id(self) -> str:
        """Alias for the canonical base registry identifier."""
        return self.base_noise_id

    @property
    def noise_definition_version(self) -> str:
        """The fixed-rate noise-definition version."""
        return FIXED_RATE_NOISE_DEFINITION_VERSION

    @property
    def gate_placement(self) -> str:
        """The benchmark representation on which this provider acts."""
        return _LOGICAL_PARAMETERIZED_GATE_PLACEMENT

    @property
    def identity(self) -> tuple[str, str, float, str]:
        """The complete immutable provider identity."""
        return (
            self.base_noise_id,
            self.noise_definition_version,
            self.strength_scale,
            self.gate_placement,
        )

    def identity_payload(self) -> dict[str, object]:
        """Return the JSON-native fields that define provider identity."""
        return {
            "base_noise_id": self.base_noise_id,
            "noise_definition_version": self.noise_definition_version,
            "strength_scale": self.strength_scale,
            "gate_placement": self.gate_placement,
        }

    @property
    def content_checksum(self) -> str:
        """A deterministic checksum of the provider identity."""
        return _provider_content_checksum(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        """Return a detached deterministic JSON-native identity mapping."""
        return self.identity_payload()

    def __call__(
        self,
        context: GateNoiseContext,
        rng: np.random.Generator,
        /,
    ) -> TJMNoiseInstruction | None:
        """Build one fresh scaled gate-local TJM instruction.

        Returns:
            A fresh instruction, or ``None`` if the base profile excludes the
            gate arity.

        Raises:
            TypeError: If ``context`` is not a gate-noise context.
        """
        del rng
        if not isinstance(context, GateNoiseContext):
            msg = f"context must be a GateNoiseContext, got {type(context).__name__}."
            raise TypeError(msg)
        return _fresh_scaled_instruction(
            self.definition.process_templates(context.arity),
            context,
            self.strength_scale,
            self.base_noise_id,
        )


def create_scaled_standard_noise_provider(
    base_noise_id: str,
    strength_scale: float,
) -> ScaledStandardNoiseProvider:
    """Create an immutable scaled provider for one standard profile.

    Returns:
        A provider bound to the canonical base definition and requested scale.
    """
    return ScaledStandardNoiseProvider(
        get_standard_noise_definition(base_noise_id),
        strength_scale,
    )


def _historical_process_templates(arity: int) -> tuple[_ProcessTemplate, ...]:
    """Return the archived logical fixed-rate process templates for one gate.

    Returns:
        Ordered relative process templates for a one- or two-qubit gate.

    Raises:
        ValueError: If the gate arity is unsupported.
    """
    if arity not in {1, 2}:
        msg = f"Historical fixed-rate noise supports only gate arities one and two, got {arity}."
        raise ValueError(msg)
    single_site = tuple(
        (f"pauli_{label.lower()}", (relative_site,), _HISTORICAL_ONE_QUBIT_PROCESS_STRENGTH)
        for relative_site in range(arity)
        for label in ("X", "Y", "Z")
    )
    if arity == 1:
        return single_site
    return (
        *single_site,
        ("crosstalk_xx", (0, 1), _HISTORICAL_TWO_QUBIT_PROCESS_STRENGTH),
        ("crosstalk_zz", (0, 1), _HISTORICAL_TWO_QUBIT_PROCESS_STRENGTH),
    )


@dataclass(frozen=True, slots=True)
class HistoricalFixedRateNoiseProvider:
    """Frozen historical logical fixed-rate TJM simulation profile.

    The identifier preserves the archived hardware-inspired label solely for
    reproducibility. This provider represents a logical-gate TJM simulation: it
    is neither a Ballarin channel nor evidence of IBM Heron hardware execution.
    """

    noise_id: str = field(default=HISTORICAL_FIXED_RATE_NOISE_ID, init=False)
    noise_definition_version: str = field(default=FIXED_RATE_NOISE_DEFINITION_VERSION, init=False)
    gate_placement: str = field(default=_LOGICAL_PARAMETERIZED_GATE_PLACEMENT, init=False)
    tjm_dt: float = field(default=_HISTORICAL_TJM_DT, init=False)

    @property
    def identity(self) -> tuple[str, str, str, float]:
        """The complete immutable historical profile identity."""
        return (
            self.noise_id,
            self.noise_definition_version,
            self.gate_placement,
            self.tjm_dt,
        )

    def identity_payload(self) -> dict[str, object]:
        """Return the JSON-native fields that define historical identity."""
        return {
            "noise_id": self.noise_id,
            "noise_definition_version": self.noise_definition_version,
            "gate_placement": self.gate_placement,
            "tjm_dt": self.tjm_dt,
        }

    @property
    def content_checksum(self) -> str:
        """A deterministic checksum of the full frozen profile."""
        return _provider_content_checksum(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return complete deterministic historical-profile metadata."""
        return {
            **self.identity_payload(),
            "channel_semantics": "fixed_rate_logical_tjm_simulation",
            "is_ballarin": False,
            "is_hardware_execution": False,
            "two_qubit_crosstalk_connectivity": "adjacent_linear_chain_only",
            "one_qubit_gate_processes": [
                {
                    "name": name,
                    "relative_sites": list(relative_sites),
                    "strength": strength,
                }
                for name, relative_sites, strength in _historical_process_templates(1)
            ],
            "two_qubit_gate_processes": [
                {
                    "name": name,
                    "relative_sites": list(relative_sites),
                    "strength": strength,
                }
                for name, relative_sites, strength in _historical_process_templates(2)
            ],
        }

    def __call__(
        self,
        context: GateNoiseContext,
        rng: np.random.Generator,
        /,
    ) -> TJMNoiseInstruction:
        """Build a fresh archived logical gate-local TJM instruction.

        Returns:
            A fresh model containing three processes per participating site and,
            for two-qubit gates, the archived ``XX`` and ``ZZ`` processes.

        Raises:
            TypeError: If ``context`` is not a gate-noise context.
        """
        del rng
        if not isinstance(context, GateNoiseContext):
            msg = f"context must be a GateNoiseContext, got {type(context).__name__}."
            raise TypeError(msg)
        templates = _historical_process_templates(context.arity)
        if context.arity == 2 and abs(context.sites[0] - context.sites[1]) != 1:
            templates = tuple(template for template in templates if len(template[1]) == 1)
        instruction = _fresh_scaled_instruction(
            templates,
            context,
            1.0,
            self.noise_id,
        )
        assert instruction is not None
        return instruction


def create_historical_fixed_rate_noise_provider() -> HistoricalFixedRateNoiseProvider:
    """Create the frozen historical logical fixed-rate simulation provider.

    Returns:
        A stateless immutable provider for the exact archived profile.
    """
    return HistoricalFixedRateNoiseProvider()


__all__ = [
    "FIXED_RATE_NOISE_DEFINITION_VERSION",
    "HISTORICAL_FIXED_RATE_NOISE_ID",
    "STANDARD_NOISE_REGISTRY",
    "STANDARD_NOISE_STRENGTH_INTERPRETATION",
    "STANDARD_ONE_QUBIT_GATE_STRENGTH",
    "STANDARD_TWO_QUBIT_GATE_STRENGTH",
    "TWO_SITE_DEPOLARIZING_OPERATORS",
    "HistoricalFixedRateNoiseProvider",
    "PauliDistribution",
    "ScaledStandardNoiseProvider",
    "StandardNoiseDefinition",
    "StandardNoiseProvider",
    "create_historical_fixed_rate_noise_provider",
    "create_scaled_standard_noise_provider",
    "create_standard_noise_provider",
    "get_standard_noise_definition",
    "sample_local_pauli",
    "sample_product_pauli_channel",
]
