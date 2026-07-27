# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Gate-local noise-provider records for parameterized-circuit trajectories.

The types in this module deliberately describe only the two noise mechanisms
needed by circuit-TJM optimization:

* a gate-local :class:`~mqt.yaqs.core.data_structures.noise_model.NoiseModel`;
* a realized, state-independent sequence of local unitary operators.

The two mechanisms may be sequenced only through an explicit
:class:`CompositeGateNoiseInstruction`.

Providers receive immutable gate metadata and a trajectory-local random-number
generator. They never receive the evolving state. This keeps random-unitary
sampling independent of MPS normalization and truncation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np

from ..core.data_structures.noise_model import NoiseModel

if TYPE_CHECKING:
    from numpy.typing import NDArray

_UNITARY_ATOL = 1e-10


def _validate_identifier(value: int | str, name: str) -> int | str:
    """Validate a logical or native gate identifier.

    Args:
        value: Identifier to validate.
        name: Field name used in validation errors.

    Returns:
        The normalized identifier.

    Raises:
        TypeError: If the identifier is not an integer or string.
        ValueError: If the identifier is negative or an empty string.
    """
    if isinstance(value, (bool, np.bool_)):
        msg = f"{name} must be an integer or nonempty string, not a Boolean."
        raise TypeError(msg)
    if isinstance(value, Integral):
        identifier = int(value)
        if identifier < 0:
            msg = f"{name} must be nonnegative, got {identifier}."
            raise ValueError(msg)
        return identifier
    if isinstance(value, str):
        if not value or value != value.strip():
            msg = f"{name} must be a nonempty string without surrounding whitespace."
            raise ValueError(msg)
        return value
    msg = f"{name} must be an integer or nonempty string, got {type(value).__name__}."
    raise TypeError(msg)


def _validate_optional_label(value: str | None, name: str) -> str | None:
    """Validate an optional diagnostic label.

    Args:
        value: Label to validate.
        name: Field name used in validation errors.

    Returns:
        The validated label.

    Raises:
        TypeError: If the label is not a string or ``None``.
        ValueError: If the label is empty or has surrounding whitespace.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{name} must be a string or None, got {type(value).__name__}."
        raise TypeError(msg)
    if not value or value != value.strip():
        msg = f"{name} must be nonempty and have no surrounding whitespace."
        raise ValueError(msg)
    return value


def _validate_sites(
    sites: object,
    name: str,
) -> tuple[int, ...]:
    """Validate one- or two-site support in ascending order.

    Args:
        sites: Site sequence to validate.
        name: Field name used in validation errors.

    Returns:
        Validated sites as a tuple of built-in integers.

    Raises:
        TypeError: If the support is not an appropriate integer sequence.
        ValueError: If the support is empty, duplicated, too large, or unordered.
    """
    if isinstance(sites, (str, bytes)) or not isinstance(sites, Sequence):
        msg = f"{name} must be a sequence of one or two site indices."
        raise TypeError(msg)
    if len(sites) not in {1, 2}:
        msg = f"{name} must contain one or two sites, got {len(sites)}."
        raise ValueError(msg)

    normalized: list[int] = []
    for site in sites:
        if isinstance(site, (bool, np.bool_)):
            msg = f"{name} must not contain Boolean site indices."
            raise TypeError(msg)
        if not isinstance(site, Integral):
            msg = f"{name} must contain only integer site indices, got {type(site).__name__}."
            raise TypeError(msg)
        site_index = int(site)
        if site_index < 0:
            msg = f"{name} must contain only nonnegative site indices, got {site_index}."
            raise ValueError(msg)
        normalized.append(site_index)

    site_tuple = tuple(normalized)
    if len(set(site_tuple)) != len(site_tuple):
        msg = f"{name} contains duplicate sites {site_tuple!r}."
        raise ValueError(msg)
    if site_tuple != tuple(sorted(site_tuple)):
        msg = f"{name} must be in ascending order, got {site_tuple!r}."
        raise ValueError(msg)
    return site_tuple


def _immutable_complex_matrix(matrix: object, name: str) -> NDArray[np.complex128]:
    """Return a finite, defensively copied, irreversibly read-only complex matrix.

    Args:
        matrix: Matrix-like input.
        name: Field name used in validation errors.

    Returns:
        An immutable complex matrix backed by a read-only byte buffer.

    Raises:
        TypeError: If the input cannot be converted to a complex matrix.
        ValueError: If the input is not two-dimensional or has non-finite entries.
    """
    try:
        converted = np.asarray(matrix, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        msg = f"{name} must be convertible to a complex matrix."
        raise TypeError(msg) from error
    if converted.ndim != 2:
        msg = f"{name} must be two-dimensional, got shape {converted.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(converted)):
        msg = f"{name} must contain only finite entries."
        raise ValueError(msg)

    contiguous = np.ascontiguousarray(converted)
    return np.frombuffer(contiguous.tobytes(), dtype=np.complex128).reshape(contiguous.shape)


def _validate_unitary_matrix(matrix: NDArray[np.complex128], sites: tuple[int, ...], name: str) -> None:
    """Validate the dimension and unitarity of a local operator matrix.

    Args:
        matrix: Matrix to validate.
        sites: Operator support.
        name: Field name used in validation errors.

    Raises:
        ValueError: If the matrix has the wrong shape or is not unitary.
    """
    dimension = 2 ** len(sites)
    expected_shape = (dimension, dimension)
    if matrix.shape != expected_shape:
        msg = f"{name} on {len(sites)} site(s) must have shape {expected_shape}, got {matrix.shape}."
        raise ValueError(msg)
    identity = np.eye(dimension, dtype=np.complex128)
    if not np.allclose(matrix.conj().T @ matrix, identity, atol=_UNITARY_ATOL, rtol=_UNITARY_ATOL):
        msg = f"{name} must be unitary within tolerance {_UNITARY_ATOL}."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GateNoiseContext:
    """Immutable metadata for one post-gate noise-provider request.

    Attributes:
        gate_index: Zero-based index of the gate in the evaluated circuit.
        gate_name: Gate-library name.
        sites: Gate support in ascending-site order.
        arity: Number of sites on which the gate acts.
        resolved_angle: Evaluated gate angle, or ``None`` for non-parametric gates.
        logical_gate_id: Stable identifier of the source logical gate.
        native_gate_id: Stable identifier of the evaluated native gate.
        parameter_index: Trainable parameter index, or ``None``.
    """

    gate_index: int
    gate_name: str
    sites: tuple[int, ...]
    arity: int
    resolved_angle: float | None
    logical_gate_id: int | str
    native_gate_id: int | str
    parameter_index: int | None

    def __post_init__(self) -> None:
        """Validate and normalize gate metadata.

        Raises:
            TypeError: If a field has an unsupported type.
            ValueError: If a value is inconsistent or outside its valid range.
        """
        if isinstance(self.gate_index, (bool, np.bool_)) or not isinstance(self.gate_index, Integral):
            msg = "gate_index must be an integer."
            raise TypeError(msg)
        gate_index = int(self.gate_index)
        if gate_index < 0:
            msg = f"gate_index must be nonnegative, got {gate_index}."
            raise ValueError(msg)
        object.__setattr__(self, "gate_index", gate_index)

        if not isinstance(self.gate_name, str):
            msg = f"gate_name must be a string, got {type(self.gate_name).__name__}."
            raise TypeError(msg)
        if not self.gate_name or self.gate_name != self.gate_name.strip():
            msg = "gate_name must be nonempty and have no surrounding whitespace."
            raise ValueError(msg)

        sites = _validate_sites(self.sites, "sites")
        object.__setattr__(self, "sites", sites)

        if isinstance(self.arity, (bool, np.bool_)) or not isinstance(self.arity, Integral):
            msg = "arity must be an integer."
            raise TypeError(msg)
        arity = int(self.arity)
        if arity != len(sites):
            msg = f"arity={arity} does not match the {len(sites)} gate sites."
            raise ValueError(msg)
        object.__setattr__(self, "arity", arity)

        if self.resolved_angle is not None:
            if isinstance(self.resolved_angle, (bool, np.bool_)) or not isinstance(self.resolved_angle, Real):
                msg = "resolved_angle must be a real number or None."
                raise TypeError(msg)
            angle = float(self.resolved_angle)
            if not np.isfinite(angle):
                msg = f"resolved_angle must be finite, got {angle!r}."
                raise ValueError(msg)
            object.__setattr__(self, "resolved_angle", angle)

        object.__setattr__(
            self,
            "logical_gate_id",
            _validate_identifier(self.logical_gate_id, "logical_gate_id"),
        )
        object.__setattr__(
            self,
            "native_gate_id",
            _validate_identifier(self.native_gate_id, "native_gate_id"),
        )

        if self.parameter_index is not None:
            if isinstance(self.parameter_index, (bool, np.bool_)) or not isinstance(self.parameter_index, Integral):
                msg = "parameter_index must be an integer or None."
                raise TypeError(msg)
            parameter_index = int(self.parameter_index)
            if parameter_index < 0:
                msg = f"parameter_index must be nonnegative, got {parameter_index}."
                raise ValueError(msg)
            object.__setattr__(self, "parameter_index", parameter_index)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class LocalOperator:
    """One immutable local unitary in a realized random-unitary instruction.

    Attributes:
        matrix: Dense ``2 x 2`` or ``4 x 4`` unitary.
        sites: One- or two-site support in ascending order.
        label: Optional diagnostic outcome label.
    """

    sites: tuple[int, ...]
    label: str | None = None
    _matrix_bytes: bytes = field(init=False, repr=False)
    _matrix_shape: tuple[int, int] = field(init=False, repr=False)
    __hash__ = None

    def __init__(
        self,
        matrix: NDArray[np.complex128],
        sites: tuple[int, ...],
        label: str | None = None,
    ) -> None:
        """Validate and defensively freeze a local operator.

        Args:
            matrix: Dense local unitary.
            sites: One- or two-site support in ascending order.
            label: Optional diagnostic outcome label.
        """
        validated_sites = _validate_sites(sites, "LocalOperator.sites")
        immutable_matrix = _immutable_complex_matrix(matrix, "LocalOperator.matrix")
        _validate_unitary_matrix(immutable_matrix, validated_sites, "LocalOperator.matrix")
        object.__setattr__(self, "sites", validated_sites)
        object.__setattr__(self, "label", _validate_optional_label(label, "LocalOperator.label"))
        object.__setattr__(self, "_matrix_bytes", immutable_matrix.tobytes())
        object.__setattr__(self, "_matrix_shape", (immutable_matrix.shape[0], immutable_matrix.shape[1]))

    @property
    def matrix(self) -> NDArray[np.complex128]:
        """A fresh read-only matrix view.

        Returns:
            A complex matrix whose data and metadata cannot mutate this record.
        """
        return np.frombuffer(self._matrix_bytes, dtype=np.complex128).reshape(self._matrix_shape)

    def __eq__(self, other: object) -> bool:
        """Compare two local operators without NumPy's ambiguous array equality.

        Returns:
            Whether both records contain the same support, label, and matrix.
        """
        if not isinstance(other, LocalOperator):
            return NotImplemented
        return self.sites == other.sites and self.label == other.label and np.array_equal(self.matrix, other.matrix)


@dataclass(frozen=True, slots=True)
class TJMNoiseInstruction:
    """A gate-local noise model evaluated by the existing circuit-TJM machinery.

    Attributes:
        noise_model: Concrete gate-local YAQS noise model.
        channel_id: Optional diagnostic channel identifier.
    """

    noise_model: NoiseModel
    channel_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the instruction's immediate fields.

        Raises:
            TypeError: If ``noise_model`` is not a :class:`NoiseModel`.
        """
        if not isinstance(self.noise_model, NoiseModel):
            msg = f"noise_model must be a NoiseModel, got {type(self.noise_model).__name__}."
            raise TypeError(msg)
        object.__setattr__(self, "channel_id", _validate_optional_label(self.channel_id, "channel_id"))


@dataclass(frozen=True, slots=True)
class RandomUnitaryInstruction:
    """A realized state-independent random-unitary channel outcome.

    Operators are replayed in tuple order. An empty tuple represents an identity
    realization while ``outcome_labels`` may still record sampled identity
    branches.

    Attributes:
        operators: Ordered local unitaries comprising the realized outcome.
        channel_id: Optional diagnostic channel identifier.
        outcome_labels: Optional labels for the sampled channel branches.
    """

    operators: tuple[LocalOperator, ...] = ()
    channel_id: str | None = None
    outcome_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the realized channel outcome.

        Raises:
            TypeError: If the operators or outcome-label collections are not tuples.
        """
        if not isinstance(self.operators, tuple):
            msg = "operators must be a tuple of LocalOperator objects."
            raise TypeError(msg)
        if not all(isinstance(operator, LocalOperator) for operator in self.operators):
            msg = "operators must contain only LocalOperator objects."
            raise TypeError(msg)

        object.__setattr__(self, "channel_id", _validate_optional_label(self.channel_id, "channel_id"))

        if not isinstance(self.outcome_labels, tuple):
            msg = "outcome_labels must be a tuple of strings."
            raise TypeError(msg)
        labels: list[str] = []
        for index, label in enumerate(self.outcome_labels):
            validated = _validate_optional_label(label, f"outcome_labels[{index}]")
            assert validated is not None
            labels.append(validated)
        object.__setattr__(self, "outcome_labels", tuple(labels))


AtomicGateNoiseInstruction: TypeAlias = TJMNoiseInstruction | RandomUnitaryInstruction


@dataclass(frozen=True, slots=True)
class CompositeGateNoiseInstruction:
    """An explicit ordered composition of supported gate-local mechanisms.

    Raw :class:`NoiseModel` objects are deliberately not accepted as children:
    wrap them in :class:`TJMNoiseInstruction` so mixed TJM/random-unitary
    composition is always visible at the provider boundary.

    Attributes:
        instructions: Atomic instructions applied in tuple order.
        channel_id: Optional diagnostic identifier for the composite channel.
    """

    instructions: tuple[AtomicGateNoiseInstruction, ...]
    channel_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the explicit composition.

        Raises:
            TypeError: If the instruction collection is not an atomic tuple.
        """
        if not isinstance(self.instructions, tuple):
            msg = "instructions must be a tuple of TJMNoiseInstruction or RandomUnitaryInstruction objects."
            raise TypeError(msg)
        if not all(
            isinstance(instruction, (TJMNoiseInstruction, RandomUnitaryInstruction))
            for instruction in self.instructions
        ):
            msg = "instructions must contain only TJMNoiseInstruction or RandomUnitaryInstruction objects."
            raise TypeError(msg)
        object.__setattr__(self, "channel_id", _validate_optional_label(self.channel_id, "channel_id"))


ValidatedGateNoiseInstruction: TypeAlias = AtomicGateNoiseInstruction | CompositeGateNoiseInstruction
GateNoiseInstruction: TypeAlias = NoiseModel | ValidatedGateNoiseInstruction


class GateNoiseProvider(Protocol):
    """Callable protocol for state-independent post-gate noise providers."""

    def __call__(
        self,
        context: GateNoiseContext,
        rng: np.random.Generator,
        /,
    ) -> GateNoiseInstruction | None:
        """Return a gate-local instruction sampled with ``rng``."""
        ...


def _validate_noise_model(noise_model: NoiseModel, context: GateNoiseContext) -> None:
    """Validate a concrete gate-local TJM model against the current gate.

    Args:
        noise_model: Model returned by a gate-local provider.
        context: Current gate context.

    Raises:
        TypeError: If process support or strength has an unsupported type.
        ValueError: If scheduled jumps, strengths, or process support are invalid.
    """
    if noise_model.scheduled_jumps:
        msg = "Gate-local NoiseModel instructions do not support scheduled jumps."
        raise ValueError(msg)

    gate_support = set(context.sites)
    for process_index, process in enumerate(noise_model.processes):
        if not isinstance(process, Mapping):
            msg = f"noise_model.processes[{process_index}] must be a mapping."
            raise TypeError(msg)
        if "name" not in process:
            msg = f"noise_model.processes[{process_index}] is missing 'name'."
            raise ValueError(msg)
        process_name = process["name"]
        if not isinstance(process_name, str):
            msg = f"noise_model.processes[{process_index}]['name'] must be a string."
            raise TypeError(msg)
        if not process_name or process_name != process_name.strip():
            msg = f"noise_model.processes[{process_index}]['name'] must be nonempty without surrounding whitespace."
            raise ValueError(msg)
        if "strength" not in process:
            msg = f"noise_model.processes[{process_index}] is missing 'strength'."
            raise ValueError(msg)
        strength = process["strength"]
        if isinstance(strength, (bool, np.bool_)) or not isinstance(strength, Real):
            msg = (
                f"noise_model.processes[{process_index}]['strength'] must be a concrete real number, "
                f"got {type(strength).__name__}."
            )
            raise TypeError(msg)
        concrete_strength = float(strength)
        if not np.isfinite(concrete_strength):
            msg = f"noise_model.processes[{process_index}]['strength'] must be finite."
            raise ValueError(msg)
        if concrete_strength < 0.0:
            msg = f"noise_model.processes[{process_index}]['strength'] must be nonnegative."
            raise ValueError(msg)

        if "sites" not in process:
            msg = f"noise_model.processes[{process_index}] is missing 'sites'."
            raise ValueError(msg)
        sites = _validate_sites(process["sites"], f"noise_model.processes[{process_index}]['sites']")
        if not set(sites).issubset(gate_support):
            msg = f"noise_model.processes[{process_index}] acts on {sites!r}, outside gate support {context.sites!r}."
            raise ValueError(msg)
        _validate_noise_process_payload(process, process_index, sites)


def _validate_random_unitary(
    instruction: RandomUnitaryInstruction,
    context: GateNoiseContext,
) -> None:
    """Validate realized local operators against the current gate support.

    Args:
        instruction: Realized random-unitary instruction.
        context: Current gate context.

    Raises:
        ValueError: If an operator acts outside the current gate support.
    """
    gate_support = set(context.sites)
    for operator_index, operator in enumerate(instruction.operators):
        matrix = _immutable_complex_matrix(operator.matrix, f"operators[{operator_index}].matrix")
        _validate_unitary_matrix(matrix, operator.sites, f"operators[{operator_index}].matrix")
        if not set(operator.sites).issubset(gate_support):
            msg = f"operators[{operator_index}] acts on {operator.sites!r}, outside gate support {context.sites!r}."
            raise ValueError(msg)


def _validate_noise_process_payload(
    process: Mapping[str, object],
    process_index: int,
    sites: tuple[int, ...],
) -> None:
    """Validate the local operator payload of one TJM process.

    Args:
        process: Noise-process mapping.
        process_index: Index used in validation errors.
        sites: Validated process support.

    Raises:
        TypeError: If a matrix or factor payload has an unsupported type.
        ValueError: If a payload has the wrong shape or non-finite entries.
    """
    prefix = f"noise_model.processes[{process_index}]"
    if "matrix" in process:
        matrix = _immutable_complex_matrix(process["matrix"], f"{prefix}['matrix']")
        dimension = 2 ** len(sites)
        expected_shape = (dimension, dimension)
        if matrix.shape != expected_shape:
            msg = f"{prefix}['matrix'] must have shape {expected_shape}, got {matrix.shape}."
            raise ValueError(msg)
        return

    if len(sites) == 1:
        msg = f"{prefix} must provide a matrix for its one-site operator."
        raise ValueError(msg)

    if "factors" not in process:
        msg = f"{prefix} must provide either a two-site matrix or two local factors."
        raise ValueError(msg)
    factors = process["factors"]
    if isinstance(factors, (str, bytes)) or not isinstance(factors, Sequence):
        msg = f"{prefix}['factors'] must be a sequence of two matrices."
        raise TypeError(msg)
    if len(factors) != 2:
        msg = f"{prefix}['factors'] must contain exactly two matrices."
        raise ValueError(msg)
    for factor_index, factor in enumerate(factors):
        factor_matrix = _immutable_complex_matrix(
            factor,
            f"{prefix}['factors'][{factor_index}]",
        )
        if factor_matrix.shape != (2, 2):
            msg = f"{prefix}['factors'][{factor_index}] must have shape (2, 2), got {factor_matrix.shape}."
            raise ValueError(msg)


def validate_gate_noise_instruction(
    instruction: GateNoiseInstruction | None,
    context: GateNoiseContext,
) -> ValidatedGateNoiseInstruction | None:
    """Validate and normalize one provider result for a gate.

    A raw :class:`NoiseModel` is accepted as shorthand for
    :class:`TJMNoiseInstruction`. Explicit composites contain only tagged atomic
    instructions. No categorical, Kraus, or state-dependent output forms are
    accepted.

    Args:
        instruction: Provider output to validate.
        context: Context supplied to the provider.

    Returns:
        ``None`` or a normalized tagged instruction.

    Raises:
        TypeError: If ``context`` or ``instruction`` has an unsupported type.
    """
    if not isinstance(context, GateNoiseContext):
        msg = f"context must be a GateNoiseContext, got {type(context).__name__}."
        raise TypeError(msg)
    if instruction is None:
        return None
    if isinstance(instruction, NoiseModel):
        normalized = TJMNoiseInstruction(instruction)
        _validate_noise_model(normalized.noise_model, context)
        return normalized
    if isinstance(instruction, TJMNoiseInstruction):
        _validate_noise_model(instruction.noise_model, context)
        return instruction
    if isinstance(instruction, RandomUnitaryInstruction):
        _validate_random_unitary(instruction, context)
        return instruction
    if isinstance(instruction, CompositeGateNoiseInstruction):
        for child in instruction.instructions:
            if isinstance(child, TJMNoiseInstruction):
                _validate_noise_model(child.noise_model, context)
            else:
                _validate_random_unitary(child, context)
        return instruction
    msg = (
        "Gate-noise providers must return None, NoiseModel, "
        "TJMNoiseInstruction, RandomUnitaryInstruction, or CompositeGateNoiseInstruction; "
        f"got {type(instruction).__name__}."
    )
    raise TypeError(msg)


__all__ = [
    "CompositeGateNoiseInstruction",
    "GateNoiseContext",
    "GateNoiseInstruction",
    "GateNoiseProvider",
    "LocalOperator",
    "RandomUnitaryInstruction",
    "TJMNoiseInstruction",
    "ValidatedGateNoiseInstruction",
    "validate_gate_noise_instruction",
]
