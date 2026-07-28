# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Final-circuit materialization and Ballarin noise for state preparation.

The Ballarin/Quantinuum benchmark is evaluated on a fully bound native circuit.
This module owns that final boundary: native ``RZZ`` angles are canonicalized,
small entanglers and their compilation-only basis changes are removed, safe
basis-change inverses are cancelled, and the resulting gate specification is
made immutable before either noiseless or noisy evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, field
from decimal import Decimal
from fractions import Fraction
from numbers import Integral, Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np

from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.optimization import (
    GateNoiseContext,
    ParameterizedCircuit,
    ParameterizedGate,
    RandomUnitaryInstruction,
)
from mqt.yaqs.optimization.parameterized_circuit import SINGLE_ANGLE_GATES

from .circuits import BasisChangeRelationship, LogicalToNativeMapping, NativeAngleExpression, NativeCompilation
from .constants import BALLARIN_NOISE_ID
from .noise import PauliDistribution, sample_product_pauli_channel

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

NativeGateIdentifier: TypeAlias = int | str
_BasisRole: TypeAlias = Literal["before", "after"]

BALLARIN_PRUNING_THRESHOLD = 1e-4
BALLARIN_EPSILON_INTERCEPT = 2.1e-4
BALLARIN_EPSILON_SLOPE = 1.43e-3
BALLARIN_MAX_EPSILON = 4.0 / 5.0

_TWO_PI = 2.0 * math.pi
_DOMAIN_ROUNDOFF = 8.0 * math.ulp(BALLARIN_MAX_EPSILON)


def _validated_integer_count(value: object, name: str, *, minimum: int) -> int:
    """Validate and normalize an integer size.

    Args:
        value: Count to validate.
        name: Argument name used in validation errors.
        minimum: Smallest accepted value.

    Returns:
        The normalized built-in integer.

    Raises:
        TypeError: If ``value`` is not a non-Boolean integer.
        ValueError: If ``value`` is smaller than ``minimum``.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        msg = f"{name} must be an integer, got {type(value).__name__}."
        raise TypeError(msg)
    normalized = int(value)
    if normalized < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        msg = f"{name} must be {qualifier}, got {normalized}."
        raise ValueError(msg)
    return normalized


def _validated_sites(sites: object, name: str) -> tuple[int, ...]:
    """Validate and normalize one immutable gate support tuple.

    Args:
        sites: Site tuple to validate.
        name: Field name used in validation errors.

    Returns:
        Built-in integer site indices in source order.

    Raises:
        TypeError: If the support or an index has an invalid type.
        ValueError: If the support has invalid arity or duplicate indices.
    """
    if not isinstance(sites, tuple):
        msg = f"{name} must be a tuple, got {type(sites).__name__}."
        raise TypeError(msg)
    if len(sites) not in {1, 2}:
        msg = f"{name} must contain one or two sites, got {sites!r}."
        raise ValueError(msg)
    normalized: list[int] = []
    for position, site in enumerate(sites):
        if isinstance(site, (bool, np.bool_)) or not isinstance(site, Integral):
            msg = f"{name}[{position}] must be an integer."
            raise TypeError(msg)
        normalized_site = int(site)
        if normalized_site < 0:
            msg = f"{name}[{position}] must be nonnegative."
            raise ValueError(msg)
        normalized.append(normalized_site)
    if len(set(normalized)) != len(normalized):
        msg = f"{name} contains duplicate sites."
        raise ValueError(msg)
    return tuple(normalized)


def _validated_name(value: object, name: str) -> str:
    """Validate and normalize a gate or provenance name.

    Returns:
        An exact built-in string detached from any subclass state.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If ``value`` is empty.
    """
    if not isinstance(value, str):
        msg = f"{name} must be a string, got {type(value).__name__}."
        raise TypeError(msg)
    normalized = str(value)
    if not normalized:
        msg = f"{name} must not be empty."
        raise ValueError(msg)
    return normalized


def _validated_optional_index(value: object, name: str) -> int | None:
    """Validate and normalize an optional nonnegative integer index.

    Returns:
        A built-in integer or ``None``.
    """
    if value is None:
        return None
    return _validated_integer_count(value, name, minimum=0)


def _validated_identifier(
    value: object,
    name: str,
    *,
    allow_none: bool,
) -> NativeGateIdentifier | None:
    """Validate and normalize one stable logical or native identifier.

    Returns:
        A built-in nonnegative integer, an exact trimmed string, or ``None``.

    Raises:
        TypeError: If the identifier has an invalid type.
        ValueError: If its integer or string value is invalid.
    """
    if value is None:
        if allow_none:
            return None
        msg = f"{name} must not be None."
        raise ValueError(msg)
    if isinstance(value, (bool, np.bool_)):
        msg = f"{name} must be a nonnegative integer or a nonempty string."
        raise TypeError(msg)
    if isinstance(value, Integral):
        return _validated_integer_count(value, name, minimum=0)
    if isinstance(value, str):
        normalized = str(value)
        if not normalized or normalized != normalized.strip():
            msg = f"{name} must be a nonempty string without surrounding whitespace."
            raise ValueError(msg)
        return normalized
    msg = f"{name} must be a nonnegative integer or a nonempty string."
    raise TypeError(msg)


def _finite_real(value: object, name: str) -> float:
    """Validate and normalize one finite real scalar.

    Args:
        value: Scalar to validate.
        name: Argument name used in validation errors.

    Returns:
        The normalized built-in float.

    Raises:
        TypeError: If ``value`` is not a real scalar or is a Boolean.
        ValueError: If ``value`` is non-finite or overflows a float.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        msg = f"{name} must be a finite real number, got {type(value).__name__}."
        raise TypeError(msg)
    try:
        normalized = float(value)
    except OverflowError as error:
        msg = f"{name} must be finite, got an overflowing real value."
        raise ValueError(msg) from error
    if not math.isfinite(normalized):
        msg = f"{name} must be finite, got {normalized!r}."
        raise ValueError(msg)
    return normalized


def canonicalize_rzz_angle(angle: object) -> float:
    """Canonicalize a native ``RZZ`` angle to the interval ``[-pi, pi)``.

    ``math.remainder`` avoids the rounding failure of the common
    ``(angle + pi) % (2*pi) - pi`` expression immediately below ``pi``.
    Exact positive half turns are mapped to ``-pi``, and signed zero is
    normalized to positive zero.

    Args:
        angle: Finite angle in radians.

    Returns:
        The canonical angle in ``[-pi, pi)``.
    """
    normalized = _finite_real(angle, "angle")
    canonical = math.remainder(normalized, _TWO_PI)
    if canonical == math.pi:
        canonical = -math.pi
    if not canonical:
        return 0.0
    return canonical


def canonicalize_native_rzz_angle(angle: object) -> float:
    """Alias spelling that emphasizes the native-circuit boundary.

    Args:
        angle: Finite angle in radians.

    Returns:
        The canonical angle in ``[-pi, pi)``.
    """
    return canonicalize_rzz_angle(angle)


def _angle_magnitude(value: object) -> float:
    """Validate one nonnegative finite canonical-angle magnitude.

    Returns:
        The normalized magnitude.

    Raises:
        ValueError: If ``value`` is negative or non-finite.
    """
    magnitude = _finite_real(value, "angle_magnitude")
    if magnitude < 0.0:
        msg = f"angle_magnitude must be nonnegative, got {magnitude!r}."
        raise ValueError(msg)
    return magnitude


def ballarin_epsilon(angle_magnitude: object) -> float:
    """Return the Ballarin native-gate error fit ``epsilon(a)``.

    Args:
        angle_magnitude: Nonnegative native ``RZZ`` angle magnitude.

    Returns:
        The fitted two-qubit error probability.

    Raises:
        ValueError: If the magnitude is negative or non-finite, or the result
            overflows.
    """
    magnitude = _angle_magnitude(angle_magnitude)
    epsilon = BALLARIN_EPSILON_INTERCEPT + BALLARIN_EPSILON_SLOPE * magnitude
    if not math.isfinite(epsilon):
        msg = f"Ballarin epsilon must be finite, got {epsilon!r}."
        raise ValueError(msg)
    return epsilon


def ballarin_local_pauli_rate(angle_magnitude: object) -> float:
    """Return the probability of each local ``X``, ``Y``, or ``Z`` outcome.

    Args:
        angle_magnitude: Nonnegative native ``RZZ`` angle magnitude.

    Returns:
        The probability ``r(a)`` of each non-identity local Pauli.

    Raises:
        ValueError: If the fitted epsilon lies outside the square-root domain.
    """
    epsilon = ballarin_epsilon(angle_magnitude)
    if epsilon > BALLARIN_MAX_EPSILON:
        if epsilon - BALLARIN_MAX_EPSILON <= _DOMAIN_ROUNDOFF:
            epsilon = BALLARIN_MAX_EPSILON
        else:
            msg = (
                "Ballarin epsilon lies outside the probability domain: "
                f"epsilon={epsilon!r} exceeds {BALLARIN_MAX_EPSILON!r}."
            )
            raise ValueError(msg)
    radicand = 1.0 - 1.25 * epsilon
    if radicand < 0.0:
        if abs(radicand) <= _DOMAIN_ROUNDOFF:
            radicand = 0.0
        else:
            msg = f"Ballarin rate has a negative square-root radicand {radicand!r}."
            raise ValueError(msg)
    return (1.0 - math.sqrt(radicand)) / 3.0


def ballarin_local_pauli_probability(angle_magnitude: object) -> float:
    """Return ``r(a)`` using an explicit probability-oriented alias.

    Args:
        angle_magnitude: Nonnegative native ``RZZ`` angle magnitude.

    Returns:
        The probability of each non-identity local Pauli.
    """
    return ballarin_local_pauli_rate(angle_magnitude)


@dataclass(frozen=True, slots=True)
class _FrozenArrayParameter:
    """Private byte-backed snapshot of one array-valued gate parameter."""

    dtype: np.dtype
    shape: tuple[int, ...]
    data: bytes

    def to_array(self, *, mutable: bool) -> NDArray[np.generic]:
        """Reconstruct the captured array without exposing stored metadata.

        Args:
            mutable: Whether to return a writable detached copy.

        Returns:
            The reconstructed array.
        """
        if self.dtype.itemsize == 0:
            view = np.empty(self.shape, dtype=self.dtype)
        else:
            view = np.frombuffer(self.data, dtype=self.dtype).reshape(self.shape)
        if mutable:
            return np.array(view, copy=True)
        view.setflags(write=False)
        return view


@dataclass(frozen=True, slots=True)
class _FrozenNumpyScalarParameter:
    """Byte-backed snapshot of one non-object NumPy scalar."""

    dtype: np.dtype
    data: bytes

    def to_scalar(self, *, mutable: bool) -> np.generic:
        """Reconstruct the captured scalar.

        Args:
            mutable: Whether its backing array should be a detached writable copy.

        Returns:
            A scalar with the captured NumPy dtype.
        """
        if self.dtype.itemsize == 0:
            array = np.zeros(1, dtype=self.dtype)
        else:
            array = np.frombuffer(self.data, dtype=self.dtype, count=1)
        if mutable:
            array = np.array(array, copy=True)
        else:
            array.setflags(write=False)
        return array[0]


@dataclass(frozen=True, slots=True)
class _FrozenObjectArrayParameter:
    """Element-backed snapshot of an object-dtype array."""

    shape: tuple[int, ...]
    values: tuple[object, ...]

    def to_array(self, *, mutable: bool) -> NDArray[np.generic]:
        """Reconstruct the captured object array.

        Args:
            mutable: Whether nested values should be detached for mutation.

        Returns:
            A fresh array with the captured shape and values.
        """
        flat = np.empty(len(self.values), dtype=object)
        for index, value in enumerate(self.values):
            flat[index] = _thaw_fixed_parameter(value, mutable=mutable)
        array = flat.reshape(self.shape)
        if not mutable:
            array.setflags(write=False)
        return array


@dataclass(frozen=True, slots=True)
class _FrozenMappingParameter:
    """Tuple-backed snapshot of a mapping-valued parameter."""

    items: tuple[tuple[object, object], ...]

    def to_mapping(self, *, mutable: bool) -> Mapping[object, object]:
        """Reconstruct the captured mapping.

        Args:
            mutable: Whether to return a detached mutable dictionary.

        Returns:
            A dictionary for mutable clones or a read-only mapping proxy.
        """
        mapping = {
            _thaw_fixed_parameter(key, mutable=False): _thaw_fixed_parameter(value, mutable=mutable)
            for key, value in self.items
        }
        return mapping if mutable else MappingProxyType(mapping)


@dataclass(frozen=True, slots=True)
class _FrozenSequenceParameter:
    """Tuple-backed snapshot that retains list-versus-tuple provenance."""

    values: tuple[object, ...]
    source_was_list: bool

    def to_sequence(
        self,
        *,
        mutable: bool,
    ) -> list[object] | tuple[object, ...]:
        """Reconstruct the captured sequence.

        Args:
            mutable: Whether source lists should become mutable lists again.

        Returns:
            A detached list for mutable clones of source lists, otherwise a
            tuple.
        """
        values = tuple(_thaw_fixed_parameter(value, mutable=mutable) for value in self.values)
        return list(values) if mutable and self.source_was_list else values


@dataclass(frozen=True, slots=True)
class _FrozenSetParameter:
    """Tuple-backed snapshot of a set-valued parameter."""

    values: tuple[object, ...]

    def to_set(self, *, mutable: bool) -> set[object] | frozenset[object]:
        """Reconstruct the captured set.

        Args:
            mutable: Whether to return a detached mutable set.

        Returns:
            A set for mutable clones or a frozen set for read-only views.
        """
        values = (_thaw_fixed_parameter(value, mutable=False) for value in self.values)
        return set(values) if mutable else frozenset(values)


def _dtype_contains_metadata(dtype: np.dtype) -> bool:
    """Return whether a dtype or any nested field carries metadata."""
    if dtype.metadata is not None:
        return True
    if dtype.subdtype is not None:
        return _dtype_contains_metadata(dtype.subdtype[0])
    if dtype.fields is not None:
        return any(_dtype_contains_metadata(field[0]) for field in dtype.fields.values())
    return False


def _metadata_free_dtype(dtype: np.dtype) -> np.dtype:
    """Return a detached metadata-free dtype suitable for byte reconstruction.

    Returns:
        ``dtype`` itself when it has no metadata, otherwise the equivalent
        scalar dtype without semantically irrelevant metadata.

    Raises:
        TypeError: If metadata is nested in a structured or subarray dtype.
    """
    if not _dtype_contains_metadata(dtype):
        return dtype
    if dtype.fields is None and dtype.subdtype is None:
        return np.dtype(dtype.str)
    msg = "fixed_params does not support metadata on structured or subarray dtypes."
    raise TypeError(msg)


def _freeze_fixed_parameter(value: object, active_ids: set[int] | None = None) -> object:
    """Return an immutable detached snapshot of one fixed gate parameter.

    Returns:
        An immutable internal representation of ``value``.

    Raises:
        TypeError: If an object cannot be safely represented.
        ValueError: If a container recursively contains itself.
    """
    if active_ids is None:
        active_ids = set()
    if isinstance(value, memoryview):
        return _freeze_fixed_parameter(np.asarray(value), active_ids)
    if isinstance(value, np.ndarray):
        contiguous = np.array(value, copy=True, order="C", subok=False)
        if contiguous.dtype.hasobject:
            if contiguous.dtype != np.dtype(object):
                msg = "fixed_params does not support structured arrays with object fields."
                raise TypeError(msg)
            identity = id(value)
            if identity in active_ids:
                msg = "fixed_params must not contain recursive object arrays."
                raise ValueError(msg)
            active_ids.add(identity)
            try:
                return _FrozenObjectArrayParameter(
                    shape=tuple(contiguous.shape),
                    values=tuple(_freeze_fixed_parameter(item, active_ids) for item in contiguous.flat),
                )
            finally:
                active_ids.remove(identity)
        return _FrozenArrayParameter(
            dtype=_metadata_free_dtype(contiguous.dtype),
            shape=tuple(contiguous.shape),
            data=contiguous.tobytes(),
        )
    if isinstance(value, (tuple, list, Mapping, set, frozenset)):
        identity = id(value)
        if identity in active_ids:
            msg = "fixed_params must not contain recursive containers."
            raise ValueError(msg)
        active_ids.add(identity)
        try:
            if isinstance(value, Mapping):
                return _FrozenMappingParameter(
                    tuple(
                        (
                            _freeze_fixed_parameter(key, active_ids),
                            _freeze_fixed_parameter(item, active_ids),
                        )
                        for key, item in value.items()
                    )
                )
            if isinstance(value, (set, frozenset)):
                return _FrozenSetParameter(tuple(_freeze_fixed_parameter(item, active_ids) for item in value))
            return _FrozenSequenceParameter(
                values=tuple(_freeze_fixed_parameter(item, active_ids) for item in value),
                source_was_list=isinstance(value, list),
            )
        finally:
            active_ids.remove(identity)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.bytes_):
        return bytes(value)
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            msg = "fixed_params does not support NumPy scalars with object fields."
            raise TypeError(msg)
        return _FrozenNumpyScalarParameter(dtype=_metadata_free_dtype(value.dtype), data=value.tobytes())
    if value is None:
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bytes):
        return bytes(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, complex):
        return complex(value)
    if isinstance(value, Decimal):
        return Decimal(str(value))
    if isinstance(value, Fraction):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, bytearray):
        return bytes(value)
    msg = f"fixed_params contains unsupported mutable value of type {type(value).__name__}."
    raise TypeError(msg)


def _thaw_fixed_parameter(value: object, *, mutable: bool) -> object:
    """Reconstruct one private fixed-parameter snapshot.

    Returns:
        A read-only view or detached mutable parameter.
    """
    if isinstance(value, _FrozenArrayParameter):
        return value.to_array(mutable=mutable)
    if isinstance(value, _FrozenNumpyScalarParameter):
        return value.to_scalar(mutable=mutable)
    if isinstance(value, _FrozenObjectArrayParameter):
        return value.to_array(mutable=mutable)
    if isinstance(value, _FrozenMappingParameter):
        return value.to_mapping(mutable=mutable)
    if isinstance(value, _FrozenSequenceParameter):
        return value.to_sequence(mutable=mutable)
    if isinstance(value, _FrozenSetParameter):
        return value.to_set(mutable=mutable)
    if isinstance(value, tuple):
        return tuple(_thaw_fixed_parameter(item, mutable=mutable) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class _NativeGateSnapshot:
    """Detached unresolved gate metadata used across callback evaluation."""

    name: str
    sites: tuple[int, ...]
    param_index: int | None
    angle_scale: float
    angle_offset: float
    data_map: Callable[[NDArray[np.float64]], float] | None
    logical_gate_id: NativeGateIdentifier | None
    native_gate_id: NativeGateIdentifier | None
    noise_enabled: bool
    _fixed_params: tuple[object, ...] = field(repr=False)

    @classmethod
    def from_gate(cls, gate: ParameterizedGate, gate_index: int) -> _NativeGateSnapshot:
        """Capture one mutable source gate without invoking its data map.

        Args:
            gate: Mutable compiled gate to detach.
            gate_index: Native position used in validation errors.

        Returns:
            A detached immutable metadata snapshot.

        Raises:
            TypeError: If callback or Boolean metadata has an invalid type.
        """
        data_map = gate.data_map
        if data_map is not None and not callable(data_map):
            msg = f"gate[{gate_index}].data_map must be callable."
            raise TypeError(msg)
        if type(gate.noise_enabled) is not bool:
            msg = f"gate[{gate_index}].noise_enabled must be a bool."
            raise TypeError(msg)
        return cls(
            name=_validated_name(gate.name, f"gate[{gate_index}].name"),
            sites=_validated_sites(gate.sites, f"gate[{gate_index}].sites"),
            param_index=_validated_optional_index(gate.param_index, f"gate[{gate_index}].param_index"),
            angle_scale=_finite_real(gate.angle_scale, f"gate[{gate_index}].angle_scale"),
            angle_offset=_finite_real(gate.angle_offset, f"gate[{gate_index}].angle_offset"),
            data_map=data_map,
            logical_gate_id=_validated_identifier(
                gate.logical_gate_id,
                f"gate[{gate_index}].logical_gate_id",
                allow_none=True,
            ),
            native_gate_id=_validated_identifier(
                gate.native_gate_id,
                f"gate[{gate_index}].native_gate_id",
                allow_none=True,
            ),
            noise_enabled=gate.noise_enabled,
            _fixed_params=tuple(_freeze_fixed_parameter(value) for value in gate.fixed_params),
        )

    @property
    def fixed_params(self) -> tuple[object, ...]:
        """Fresh read-only views of all fixed parameters."""
        return tuple(_thaw_fixed_parameter(value, mutable=False) for value in self._fixed_params)

    def detached_mutable_fixed_params(self) -> tuple[object, ...]:
        """Return detached values while retaining mutable-container provenance."""
        return tuple(_thaw_fixed_parameter(value, mutable=True) for value in self._fixed_params)

    @property
    def is_parametric(self) -> bool:
        """Whether this gate is constructed from one resolved angle."""
        return self.name in SINGLE_ANGLE_GATES


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FrozenNativeGate:
    """One immutable, fully resolved gate in the final native circuit.

    The attribute surface deliberately mirrors :class:`ParameterizedGate` so
    the immutable circuit can be evaluated directly by YAQS. Trainable indices
    and data maps are absent; a parametric gate's resolved angle is stored in
    ``angle_offset``.
    """

    name: str
    sites: tuple[int, ...]
    param_index: None = field(default=None, init=False)
    angle_scale: float = field(default=0.0, init=False)
    angle_offset: float
    data_map: None = field(default=None, init=False)
    logical_gate_id: NativeGateIdentifier | None
    native_gate_id: NativeGateIdentifier | None
    noise_enabled: bool
    _fixed_params: tuple[object, ...] = field(repr=False)

    def __init__(
        self,
        name: str,
        sites: tuple[int, ...],
        *,
        angle_offset: float = 0.0,
        fixed_params: tuple[object, ...] = (),
        logical_gate_id: NativeGateIdentifier | None = None,
        native_gate_id: NativeGateIdentifier | None = None,
        noise_enabled: bool = False,
    ) -> None:
        """Create a detached immutable gate specification.

        Args:
            name: Gate-library name.
            sites: Gate support in source order.
            angle_offset: Fully resolved angle for a parametric gate.
            fixed_params: Parameters of a nonparametric gate.
            logical_gate_id: Stable source logical identifier.
            native_gate_id: Stable pre-pruning native identifier.
            noise_enabled: Whether a provider may act after the gate.

        Raises:
            TypeError: If ``fixed_params`` is not a tuple.
        """
        if not isinstance(fixed_params, tuple):
            msg = f"fixed_params must be a tuple, got {type(fixed_params).__name__}."
            raise TypeError(msg)
        if type(noise_enabled) is not bool:
            msg = f"noise_enabled must be a bool, got {type(noise_enabled).__name__}."
            raise TypeError(msg)
        object.__setattr__(self, "name", _validated_name(name, "name"))
        object.__setattr__(self, "sites", _validated_sites(sites, "sites"))
        object.__setattr__(self, "param_index", None)
        object.__setattr__(self, "angle_scale", 0.0)
        object.__setattr__(self, "angle_offset", _finite_real(angle_offset, "angle_offset"))
        object.__setattr__(self, "data_map", None)
        object.__setattr__(
            self,
            "logical_gate_id",
            _validated_identifier(logical_gate_id, "logical_gate_id", allow_none=True),
        )
        object.__setattr__(
            self,
            "native_gate_id",
            _validated_identifier(native_gate_id, "native_gate_id", allow_none=True),
        )
        object.__setattr__(self, "noise_enabled", noise_enabled)
        object.__setattr__(
            self,
            "_fixed_params",
            tuple(_freeze_fixed_parameter(value) for value in fixed_params),
        )

    @property
    def fixed_params(self) -> tuple[object, ...]:
        """Fresh read-only views of all fixed parameters."""
        return tuple(_thaw_fixed_parameter(value, mutable=False) for value in self._fixed_params)

    @property
    def is_trainable(self) -> bool:
        """Whether this fully bound gate carries a trainable parameter."""
        return False

    @property
    def is_parametric(self) -> bool:
        """Whether this gate is constructed from one resolved angle."""
        return self.name in SINGLE_ANGLE_GATES

    @property
    def resolved_angle(self) -> float | None:
        """The bound angle, or ``None`` for a nonparametric gate."""
        return self.angle_offset if self.is_parametric else None

    def to_parameterized_gate(self) -> ParameterizedGate:
        """Return a detached mutable YAQS gate with the same resolved operation.

        Returns:
            A zero-parameter gate safe for a caller to mutate independently.
        """
        return ParameterizedGate(
            name=self.name,
            sites=self.sites,
            angle_scale=0.0,
            angle_offset=self.angle_offset,
            fixed_params=cast(
                "tuple[float, ...]",
                tuple(_thaw_fixed_parameter(value, mutable=True) for value in self._fixed_params),
            ),
            logical_gate_id=self.logical_gate_id,
            native_gate_id=self.native_gate_id,
            noise_enabled=self.noise_enabled,
        )


class FrozenNativeCircuit(ParameterizedCircuit):
    """Executable immutable snapshot of a fully materialized native circuit."""

    __slots__ = ("_frozen",)

    gates: tuple[FrozenNativeGate, ...]

    def __init__(self, num_qubits: int, gates: tuple[FrozenNativeGate, ...]) -> None:
        """Validate and freeze a zero-parameter native gate sequence.

        Args:
            num_qubits: Number of circuit qubits.
            gates: Immutable resolved native gates in evaluation order.

        Raises:
            TypeError: If ``gates`` is not an immutable tuple of resolved
                native gates.
            ValueError: If a gate's matrix arity disagrees with its sites.
        """
        if not isinstance(gates, tuple):
            msg = f"gates must be a tuple, got {type(gates).__name__}."
            raise TypeError(msg)
        if not all(type(gate) is FrozenNativeGate for gate in gates):
            msg = "gates must contain only exact FrozenNativeGate objects."
            raise TypeError(msg)
        validated_num_qubits = _validated_integer_count(num_qubits, "num_qubits", minimum=1)
        object.__setattr__(self, "_frozen", False)
        super().__init__(
            num_qubits=validated_num_qubits,
            gates=cast("list[ParameterizedGate]", list(gates)),
            num_params=0,
        )
        object.__setattr__(self, "gates", tuple(cast("list[FrozenNativeGate]", self.gates)))
        for gate_index, gate in enumerate(self.gates):
            semantic_arity = _semantic_gate_arity(gate, gate_index)
            if semantic_arity != len(gate.sites):
                msg = (
                    f"Frozen native gate {gate_index} has semantic arity {semantic_arity}, "
                    f"but sites {gate.sites!r} specify arity {len(gate.sites)}."
                )
                raise ValueError(msg)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        """Reject mutation after construction.

        Raises:
            FrozenInstanceError: If a caller changes the frozen snapshot.
        """
        if getattr(self, "_frozen", False):
            msg = f"cannot assign to field {name!r}"
            raise FrozenInstanceError(msg)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion after construction.

        Raises:
            FrozenInstanceError: If a caller deletes frozen snapshot state.
        """
        if getattr(self, "_frozen", False):
            msg = f"cannot delete field {name!r}"
            raise FrozenInstanceError(msg)
        object.__delattr__(self, name)

    def __getattribute__(self, name: str) -> object:
        """Prevent mutation through the mutable dictionary inherited by YAQS.

        Returns:
            The requested attribute, with the inherited dictionary exposed as
            a read-only mapping.
        """
        if name == "__dict__":
            attributes = object.__getattribute__(self, "__dict__")
            return MappingProxyType(attributes)
        return object.__getattribute__(self, name)

    def __copy__(self) -> FrozenNativeCircuit:
        """Return this immutable snapshot."""
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> FrozenNativeCircuit:
        """Return this immutable snapshot without traversing read-only state."""
        memo[id(self)] = self
        return self

    def to_parameterized_circuit(self) -> ParameterizedCircuit:
        """Return a detached mutable zero-parameter circuit.

        Returns:
            A fresh circuit whose gates and array-valued fixed parameters do
            not alias this authoritative snapshot.
        """
        return ParameterizedCircuit(
            num_qubits=self.num_qubits,
            gates=[gate.to_parameterized_gate() for gate in self.gates],
            num_params=0,
        )


@dataclass(frozen=True, slots=True)
class BallarinLogicalToNativeMapping:
    """Final disposition of one logical gate's native factors."""

    source_logical_gate_index: int
    logical_gate_id: NativeGateIdentifier
    source_gate_name: str
    source_sites: tuple[int, ...]
    source_parameter_index: int | None
    pre_pruning_native_gate_ids: tuple[NativeGateIdentifier, ...]
    retained_native_gate_ids: tuple[NativeGateIdentifier, ...]
    final_native_gate_indices: tuple[int, ...]
    native_rotation_gate_id: NativeGateIdentifier | None
    final_native_rotation_gate_index: int | None
    resolved_native_angle: float | None
    canonical_rzz_angle: float | None
    canonical_rzz_magnitude: float | None
    rotation_pruned: bool
    omitted_basis_change_native_gate_ids: tuple[NativeGateIdentifier, ...]
    cancelled_basis_change_native_gate_ids: tuple[NativeGateIdentifier, ...]


@dataclass(frozen=True, slots=True)
class BallarinCircuitMaterialization:
    """Immutable final native circuit and complete pruning provenance."""

    circuit: FrozenNativeCircuit
    mapping: tuple[BallarinLogicalToNativeMapping, ...]
    pre_pruning_to_final_indices: tuple[int | None, ...]
    pruned_native_rotation_ids: tuple[NativeGateIdentifier, ...]
    omitted_basis_change_native_gate_ids: tuple[NativeGateIdentifier, ...]
    cancelled_basis_change_native_gate_ids: tuple[NativeGateIdentifier, ...]

    @property
    def pruned_native_rzz_count(self) -> int:
        """Number of native rotations removed by the inclusive threshold."""
        return len(self.pruned_native_rotation_ids)

    @property
    def retained_native_rzz_count(self) -> int:
        """Number of native ``RZZ`` gates in the final circuit."""
        return sum(gate.name == "rzz" for gate in self.circuit.gates)

    @property
    def cancelled_basis_change_count(self) -> int:
        """Number of redundant compilation-only one-qubit gates removed."""
        return len(self.cancelled_basis_change_native_gate_ids)

    def to_parameterized_circuit(self) -> ParameterizedCircuit:
        """Return a detached mutable copy of the final ideal circuit.

        Returns:
            A fully bound, zero-parameter YAQS circuit.
        """
        return self.circuit.to_parameterized_circuit()


@dataclass(frozen=True, slots=True)
class _ValidatedNativeCompilation:
    """Detached validated state used while data-map callbacks execute."""

    num_qubits: int
    num_params: int
    gates: tuple[_NativeGateSnapshot, ...]
    mapping: tuple[LogicalToNativeMapping, ...]
    native_gate_ids: tuple[NativeGateIdentifier, ...]


def _validated_parameter_vector(
    theta: NDArray[np.float64],
    expected_size: int,
) -> NDArray[np.float64]:
    """Return a detached finite one-dimensional parameter vector.

    Returns:
        The normalized parameter vector.

    Raises:
        TypeError: If the input cannot be represented as real numbers.
        ValueError: If its shape, length, or values are invalid.
    """
    try:
        raw_vector = np.asarray(theta)
    except (TypeError, ValueError) as error:
        msg = "theta must be a one-dimensional finite real parameter vector."
        raise TypeError(msg) from error
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in raw_vector.flat):
        msg = "theta must contain only real non-Boolean values."
        raise TypeError(msg)
    try:
        vector = np.asarray(raw_vector, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        msg = "theta must be a one-dimensional finite real parameter vector."
        raise TypeError(msg) from error
    if vector.ndim != 1:
        msg = f"theta must be one-dimensional, got shape {vector.shape!r}."
        raise ValueError(msg)
    if vector.size != expected_size:
        msg = f"theta must contain exactly {expected_size} parameters, got {vector.size}."
        raise ValueError(msg)
    if not np.all(np.isfinite(vector)):
        msg = "theta must contain only finite values."
        raise ValueError(msg)
    return np.array(vector, dtype=np.float64, copy=True)


def _validate_indices(indices: object, name: str, gate_count: int) -> tuple[int, ...]:
    """Validate one immutable collection of pre-pruning gate indices.

    Returns:
        The normalized index tuple.

    Raises:
        TypeError: If the collection or an index has an invalid type.
        ValueError: If an index is duplicated or outside the circuit.
    """
    if not isinstance(indices, tuple):
        msg = f"{name} must be a tuple."
        raise TypeError(msg)
    normalized: list[int] = []
    for position, index in enumerate(indices):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            msg = f"{name}[{position}] must be an integer."
            raise TypeError(msg)
        value = int(index)
        if value < 0 or value >= gate_count:
            msg = f"{name}[{position}]={value} is outside range(0, {gate_count})."
            raise ValueError(msg)
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        msg = f"{name} contains duplicate gate indices."
        raise ValueError(msg)
    return tuple(normalized)


def _semantic_gate_arity(
    gate: ParameterizedGate | _NativeGateSnapshot | FrozenNativeGate,
    gate_index: int,
) -> int:
    """Return one native gate's matrix arity without resolving its angle map.

    Returns:
        The number of qubits on which the gate matrix acts.

    Raises:
        ValueError: If the gate name or fixed parameters no longer construct a
            valid library gate.
    """
    try:
        gate_factory = getattr(GateLibrary, gate.name)
    except AttributeError as error:
        msg = f"Native gate {gate_index} no longer constructs a valid gate-library operation."
        raise ValueError(msg) from error
    if gate.is_parametric:
        parameters: list[object] | None = [0.0]
    elif gate.fixed_params:
        parameters = list(gate.fixed_params)
    else:
        parameters = None
    try:
        library_gate = gate_factory() if parameters is None else gate_factory(parameters)
    except (TypeError, ValueError) as error:
        msg = f"Native gate {gate_index} no longer constructs a valid gate-library operation."
        raise ValueError(msg) from error
    return int(library_gate.interaction)


def _validate_compilation(compilation: NativeCompilation) -> _ValidatedNativeCompilation:
    """Detach and validate WP6 layout before invoking any angle callbacks.

    Returns:
        A detached gate snapshot with validated provenance.

    Raises:
        TypeError: If compilation records have invalid types.
        ValueError: If their provenance or native layout is inconsistent.
    """
    if not isinstance(compilation, NativeCompilation):
        msg = f"compilation must be a NativeCompilation, got {type(compilation).__name__}."
        raise TypeError(msg)
    circuit = compilation.circuit
    if not isinstance(circuit, ParameterizedCircuit):
        msg = f"compilation.circuit must be a ParameterizedCircuit, got {type(circuit).__name__}."
        raise TypeError(msg)
    if not isinstance(compilation.mapping, tuple):
        msg = "compilation.mapping must be a tuple."
        raise TypeError(msg)

    num_qubits = _validated_integer_count(circuit.num_qubits, "compilation.circuit.num_qubits", minimum=1)
    num_params = _validated_integer_count(circuit.num_params, "compilation.circuit.num_params", minimum=0)
    source_gates = tuple(circuit.gates)
    for native_index, gate in enumerate(source_gates):
        if not isinstance(gate, ParameterizedGate):
            msg = f"compilation.circuit.gates[{native_index}] must be a ParameterizedGate."
            raise TypeError(msg)
    gates = tuple(_NativeGateSnapshot.from_gate(gate, index) for index, gate in enumerate(source_gates))
    mapping = compilation.mapping

    gate_count = len(gates)
    covered_indices: list[int] = []
    native_ids: list[NativeGateIdentifier] = []
    for native_index, gate in enumerate(gates):
        if any(site >= num_qubits for site in gate.sites):
            msg = f"Native gate {native_index} acts on sites {gate.sites!r} outside range(0, {num_qubits})."
            raise ValueError(msg)
        if gate.param_index is not None and gate.param_index >= num_params:
            msg = (
                f"Native gate {native_index} has param_index={gate.param_index} "
                f"outside the declared parameter vector of length {num_params}."
            )
            raise ValueError(msg)
        semantic_arity = _semantic_gate_arity(gate, native_index)
        if semantic_arity != len(gate.sites):
            msg = (
                f"Native gate {native_index} has semantic arity {semantic_arity}, "
                f"but sites {gate.sites!r} specify arity {len(gate.sites)}."
            )
            raise ValueError(msg)

    validated_mapping: list[LogicalToNativeMapping] = []
    for mapping_index, record in enumerate(mapping):
        if not isinstance(record, LogicalToNativeMapping):
            msg = f"compilation.mapping[{mapping_index}] must be a LogicalToNativeMapping."
            raise TypeError(msg)
        source_logical_gate_index = _validated_integer_count(
            record.source_logical_gate_index,
            f"compilation.mapping[{mapping_index}].source_logical_gate_index",
            minimum=0,
        )
        if source_logical_gate_index != mapping_index:
            msg = (
                "Compilation mappings must remain in source logical-gate order; "
                f"entry {mapping_index} records source index {source_logical_gate_index}."
            )
            raise ValueError(msg)
        logical_gate_id = cast(
            "NativeGateIdentifier",
            _validated_identifier(
                record.logical_gate_id,
                f"compilation.mapping[{mapping_index}].logical_gate_id",
                allow_none=False,
            ),
        )
        source_gate_name = _validated_name(
            record.source_gate_name,
            f"compilation.mapping[{mapping_index}].source_gate_name",
        )
        source_sites = _validated_sites(
            record.source_sites,
            f"compilation.mapping[{mapping_index}].source_sites",
        )
        if any(site >= num_qubits for site in source_sites):
            msg = f"Compilation mapping {mapping_index} has source sites outside range(0, {num_qubits})."
            raise ValueError(msg)
        source_parameter_index = _validated_optional_index(
            record.source_parameter_index,
            f"compilation.mapping[{mapping_index}].source_parameter_index",
        )
        relationship = _validated_name(
            record.basis_change_relationship,
            f"compilation.mapping[{mapping_index}].basis_change_relationship",
        )
        indices = _validate_indices(
            record.native_gate_indices,
            f"compilation.mapping[{mapping_index}].native_gate_indices",
            gate_count,
        )
        if not indices:
            msg = f"compilation.mapping[{mapping_index}].native_gate_indices must not be empty."
            raise ValueError(msg)
        before = _validate_indices(
            record.basis_change_before_indices,
            f"compilation.mapping[{mapping_index}].basis_change_before_indices",
            gate_count,
        )
        after = _validate_indices(
            record.basis_change_after_indices,
            f"compilation.mapping[{mapping_index}].basis_change_after_indices",
            gate_count,
        )
        if any(index not in indices for index in (*before, *after)):
            msg = f"Compilation mapping {mapping_index} has a basis index outside its native group."
            raise ValueError(msg)

        rotation_index = _validated_optional_index(
            record.native_rotation_gate_index,
            f"compilation.mapping[{mapping_index}].native_rotation_gate_index",
        )
        normalized_expression: NativeAngleExpression | None = None
        if rotation_index is not None:
            if rotation_index not in indices:
                msg = f"Compilation mapping {mapping_index} has a rotation outside its native group."
                raise ValueError(msg)
            rotation_gate = gates[rotation_index]
            expression = record.native_angle_expression
            if not rotation_gate.is_parametric or not isinstance(expression, NativeAngleExpression):
                msg = f"Compilation mapping {mapping_index} has inconsistent native-angle provenance."
                raise ValueError(msg)
            expression_param_index = _validated_optional_index(
                expression.param_index,
                f"compilation.mapping[{mapping_index}].native_angle_expression.param_index",
            )
            expression_scale = _finite_real(
                expression.angle_scale,
                f"compilation.mapping[{mapping_index}].native_angle_expression.angle_scale",
            )
            expression_offset = _finite_real(
                expression.angle_offset,
                f"compilation.mapping[{mapping_index}].native_angle_expression.angle_offset",
            )
            if (
                rotation_gate.param_index != expression_param_index
                or rotation_gate.angle_scale != expression_scale
                or rotation_gate.angle_offset != expression_offset
                or rotation_gate.data_map is not expression.data_map
                or source_parameter_index != rotation_gate.param_index
            ):
                msg = f"Compilation mapping {mapping_index} has mutated native-angle metadata."
                raise ValueError(msg)
            normalized_expression = NativeAngleExpression(
                param_index=expression_param_index,
                angle_scale=expression_scale,
                angle_offset=expression_offset,
                data_map=rotation_gate.data_map,
            )
        elif record.native_angle_expression is not None or source_parameter_index is not None:
            msg = f"Compilation mapping {mapping_index} has parameter provenance without a native rotation."
            raise ValueError(msg)

        source_arity = len(source_sites)
        if source_arity not in {1, 2}:
            msg = f"Compilation mapping {mapping_index} has unsupported source sites {source_sites!r}."
            raise ValueError(msg)
        if source_arity == 2:
            if rotation_index is None:
                msg = f"Two-qubit compilation mapping {mapping_index} has no native rotation."
                raise ValueError(msg)
            rotation = gates[rotation_index]
            if rotation.name != "rzz" or len(rotation.sites) != 2:
                msg = f"Two-qubit compilation mapping {mapping_index} does not identify a native RZZ."
                raise ValueError(msg)
            expected_relationship = {
                "rxx": "rxx_h",
                "ryy": "ryy_rx_pi_over_2",
                "rzz": "none",
            }.get(source_gate_name)
            if expected_relationship is None or relationship != expected_relationship:
                msg = f"Compilation mapping {mapping_index} has an inconsistent basis-change relationship."
                raise ValueError(msg)
            expected_group = (*before, rotation_index, *after)
            if indices != expected_group:
                msg = f"Compilation mapping {mapping_index} does not preserve its native block order."
                raise ValueError(msg)
            if tuple(rotation.sites) != source_sites:
                msg = f"Compilation mapping {mapping_index} changed the native rotation sites."
                raise ValueError(msg)
            if source_gate_name == "rxx":
                expected_basis = (
                    ("h", (source_sites[0],), 0.0),
                    ("h", (source_sites[1],), 0.0),
                    ("h", (source_sites[1],), 0.0),
                    ("h", (source_sites[0],), 0.0),
                )
            elif source_gate_name == "ryy":
                half_pi = math.pi / 2.0
                expected_basis = (
                    ("rx", (source_sites[0],), half_pi),
                    ("rx", (source_sites[1],), half_pi),
                    ("rx", (source_sites[1],), -half_pi),
                    ("rx", (source_sites[0],), -half_pi),
                )
            else:
                expected_basis = ()
            actual_basis_indices = (*before, *after)
            actual_basis = tuple(gates[index] for index in actual_basis_indices)
            if len(actual_basis) != len(expected_basis) or any(
                gate.name != expected_name
                or tuple(gate.sites) != expected_sites
                or gate.param_index is not None
                or gate.data_map is not None
                or gate.fixed_params
                or gate.angle_offset != expected_angle
                or gate.noise_enabled
                for gate, (expected_name, expected_sites, expected_angle) in zip(
                    actual_basis,
                    expected_basis,
                    strict=True,
                )
            ):
                msg = f"Compilation mapping {mapping_index} has mutated or malformed basis changes."
                raise ValueError(msg)
        elif relationship != "none" or before or after:
            msg = f"One-qubit compilation mapping {mapping_index} cannot contain basis changes."
            raise ValueError(msg)
        elif len(indices) != 1:
            msg = f"One-qubit compilation mapping {mapping_index} must contain exactly one native gate."
            raise ValueError(msg)
        elif gates[indices[0]].name != source_gate_name or tuple(gates[indices[0]].sites) != source_sites:
            msg = f"One-qubit compilation mapping {mapping_index} no longer matches its source gate."
            raise ValueError(msg)
        elif gates[indices[0]].noise_enabled:
            msg = f"One-qubit compilation mapping {mapping_index} must remain noiseless."
            raise ValueError(msg)
        elif rotation_index is None and (
            gates[indices[0]].param_index is not None
            or gates[indices[0]].data_map is not None
            or not np.isclose(gates[indices[0]].angle_offset, 0.0)
        ):
            msg = f"One-qubit compilation mapping {mapping_index} has malformed passthrough metadata."
            raise ValueError(msg)

        for native_index in indices:
            gate = gates[native_index]
            if gate.logical_gate_id != logical_gate_id:
                msg = (
                    f"Native gate {native_index} logical ID {gate.logical_gate_id!r} "
                    f"does not match mapping ID {logical_gate_id!r}."
                )
                raise ValueError(msg)
            native_id = gate.native_gate_id
            if native_id is None:
                msg = f"Native gate {native_index} has no stable native_gate_id."
                raise ValueError(msg)
            native_ids.append(native_id)
        covered_indices.extend(indices)
        validated_mapping.append(
            LogicalToNativeMapping(
                source_logical_gate_index=source_logical_gate_index,
                logical_gate_id=logical_gate_id,
                source_gate_name=source_gate_name,
                source_sites=source_sites,
                source_parameter_index=source_parameter_index,
                native_gate_indices=indices,
                native_rotation_gate_index=rotation_index,
                native_angle_expression=normalized_expression,
                basis_change_before_indices=before,
                basis_change_after_indices=after,
                basis_change_relationship=cast("BasisChangeRelationship", relationship),
            )
        )

    expected_indices = list(range(gate_count))
    if covered_indices != expected_indices:
        msg = "Compilation mappings must cover every native gate exactly once in circuit order."
        raise ValueError(msg)
    if len(set(native_ids)) != len(native_ids):
        msg = "Compilation native_gate_id values must be unique."
        raise ValueError(msg)
    return _ValidatedNativeCompilation(
        num_qubits=num_qubits,
        num_params=num_params,
        gates=gates,
        mapping=tuple(validated_mapping),
        native_gate_ids=tuple(native_ids),
    )


def _resolve_angle(
    gate: ParameterizedGate | _NativeGateSnapshot,
    theta: NDArray[np.float64],
    x: NDArray[np.float64] | None,
    gate_index: int,
) -> float:
    """Resolve one native angle exactly once with strict scalar validation.

    Returns:
        The finite resolved angle.

    Raises:
        TypeError: If parameter or data-map metadata has an invalid type.
        ValueError: If angle metadata is non-finite or a required sample is
            absent.
    """
    offset = _finite_real(gate.angle_offset, f"gate[{gate_index}].angle_offset")
    angle = offset
    if gate.param_index is not None:
        if isinstance(gate.param_index, (bool, np.bool_)) or not isinstance(gate.param_index, Integral):
            msg = f"gate[{gate_index}].param_index must be an integer."
            raise TypeError(msg)
        parameter_index = int(gate.param_index)
        if parameter_index < 0 or parameter_index >= theta.size:
            msg = (
                f"gate[{gate_index}].param_index={parameter_index} is outside "
                f"the materialized parameter vector of length {theta.size}."
            )
            raise ValueError(msg)
        scale = _finite_real(gate.angle_scale, f"gate[{gate_index}].angle_scale")
        angle += scale * float(theta[parameter_index])
    if gate.data_map is not None:
        if x is None:
            msg = f"Gate '{gate.name}' at native index {gate_index} has a data map but no input sample was provided."
            raise ValueError(msg)
        if not callable(gate.data_map):
            msg = f"gate[{gate_index}].data_map must be callable."
            raise TypeError(msg)
        contribution = _finite_real(gate.data_map(x), f"gate[{gate_index}].data_map(x)")
        angle += contribution
    return _finite_real(angle, f"gate[{gate_index}] resolved angle")


def _resolved_gate(
    source: _NativeGateSnapshot,
    resolved_angle: float | None,
) -> FrozenNativeGate:
    """Create one detached immutable final-gate candidate.

    Returns:
        The bound immutable gate.
    """
    angle_offset = source.angle_offset if resolved_angle is None else resolved_angle
    return FrozenNativeGate(
        name=source.name,
        sites=source.sites,
        angle_offset=angle_offset,
        fixed_params=source.detached_mutable_fixed_params(),
        logical_gate_id=source.logical_gate_id,
        native_gate_id=source.native_gate_id,
        noise_enabled=source.name == "rzz" and len(source.sites) == 2,
    )


def _inverse_basis_pair(left: FrozenNativeGate, right: FrozenNativeGate) -> bool:
    """Return whether two compiler basis gates are exact inverses."""
    if left.sites != right.sites or len(left.sites) != 1:
        return False
    if left.name == right.name == "h":
        return True
    half_pi = math.pi / 2.0
    return (
        left.name == right.name == "rx"
        and abs(left.angle_offset) == half_pi
        and right.angle_offset == -left.angle_offset
    )


def _cancel_basis_run(
    run: list[int],
    gates: tuple[FrozenNativeGate, ...],
    basis_roles: Mapping[int, _BasisRole],
) -> set[int]:
    """Find inverse after/before pairs in one contiguous basis-only run.

    Returns:
        Pre-pruning indices of gates safe to cancel.
    """
    cancelled: set[int] = set()
    site_stacks: dict[int, list[int]] = {}
    for native_index in run:
        gate = gates[native_index]
        site = gate.sites[0]
        stack = site_stacks.setdefault(site, [])
        if (
            stack
            and basis_roles[stack[-1]] == "after"
            and basis_roles[native_index] == "before"
            and _inverse_basis_pair(gates[stack[-1]], gate)
        ):
            cancelled.add(stack.pop())
            cancelled.add(native_index)
        else:
            stack.append(native_index)
    return cancelled


def _cancel_redundant_basis_changes(
    retained_indices: tuple[int, ...],
    gates: tuple[FrozenNativeGate, ...],
    basis_roles: Mapping[int, _BasisRole],
) -> set[int]:
    """Cancel only exact compiler-basis inverses without crossing other gates.

    Returns:
        Pre-pruning indices of cancelled basis gates.
    """
    cancelled: set[int] = set()
    run: list[int] = []
    for native_index in retained_indices:
        if native_index in basis_roles:
            run.append(native_index)
            continue
        if run:
            cancelled.update(_cancel_basis_run(run, gates, basis_roles))
            run = []
    if run:
        cancelled.update(_cancel_basis_run(run, gates, basis_roles))
    return cancelled


def materialize_ballarin_circuit(
    compilation: NativeCompilation,
    theta: NDArray[np.float64],
    x: NDArray[np.float64] | None = None,
) -> BallarinCircuitMaterialization:
    """Resolve, canonicalize, prune, and freeze a Ballarin native circuit.

    Every parametric native gate is evaluated exactly once. Native ``RZZ``
    angles are then canonicalized before the inclusive pruning decision.
    Pruned compiled ``RXX``/``RYY`` rotations lose their entire basis-change
    round trip. Finally, exact inverse compilation basis gates may cancel
    within basis-only runs; logical one-qubit gates and native rotations are
    never cancelled or fused.

    Args:
        compilation: Traceable WP6 Quantinuum-native compilation.
        theta: Final trainable parameter vector.
        x: Optional sample for data-dependent angle maps.

    Returns:
        An immutable, fully bound final circuit and its pruning provenance.
    """
    validated = _validate_compilation(compilation)
    native_ids = validated.native_gate_ids
    vector = _validated_parameter_vector(theta, validated.num_params)

    resolved_angles: list[float | None] = []
    canonical_angles: list[float | None] = []
    resolved_gates: list[FrozenNativeGate] = []
    for native_index, source in enumerate(validated.gates):
        resolved_angle = _resolve_angle(source, vector, x, native_index) if source.is_parametric else None
        canonical_angle = canonicalize_rzz_angle(cast("Real", resolved_angle)) if source.name == "rzz" else None
        final_angle = canonical_angle if canonical_angle is not None else resolved_angle
        resolved_angles.append(resolved_angle)
        canonical_angles.append(canonical_angle)
        resolved_gates.append(_resolved_gate(source, final_angle))
    frozen_candidates = tuple(resolved_gates)

    pruned_rotation_indices: set[int] = set()
    omitted_basis_indices: set[int] = set()
    basis_roles: dict[int, _BasisRole] = {}
    for record in validated.mapping:
        for native_index in record.basis_change_before_indices:
            basis_roles[native_index] = "before"
        for native_index in record.basis_change_after_indices:
            basis_roles[native_index] = "after"
        if len(record.source_sites) != 2:
            continue
        rotation_index = cast("int", record.native_rotation_gate_index)
        canonical_angle = cast("float", canonical_angles[rotation_index])
        if abs(canonical_angle) <= BALLARIN_PRUNING_THRESHOLD:
            pruned_rotation_indices.add(rotation_index)
            omitted_basis_indices.update(record.basis_change_before_indices)
            omitted_basis_indices.update(record.basis_change_after_indices)

    threshold_removed = pruned_rotation_indices | omitted_basis_indices
    retained_before_cancellation = tuple(
        native_index for native_index in range(len(frozen_candidates)) if native_index not in threshold_removed
    )
    retained_basis_roles = {
        native_index: role for native_index, role in basis_roles.items() if native_index not in threshold_removed
    }
    cancelled_basis_indices = _cancel_redundant_basis_changes(
        retained_before_cancellation,
        frozen_candidates,
        retained_basis_roles,
    )
    retained_indices = tuple(
        native_index for native_index in retained_before_cancellation if native_index not in cancelled_basis_indices
    )
    final_index_by_pre_index = {native_index: final_index for final_index, native_index in enumerate(retained_indices)}
    pre_pruning_to_final = tuple(
        final_index_by_pre_index.get(native_index) for native_index in range(len(frozen_candidates))
    )
    final_circuit = FrozenNativeCircuit(
        num_qubits=validated.num_qubits,
        gates=tuple(frozen_candidates[native_index] for native_index in retained_indices),
    )

    final_mapping: list[BallarinLogicalToNativeMapping] = []
    for record in validated.mapping:
        group_indices = record.native_gate_indices
        retained_group_indices = tuple(index for index in group_indices if index in final_index_by_pre_index)
        rotation_index = record.native_rotation_gate_index
        native_rotation_id = native_ids[rotation_index] if rotation_index is not None else None
        final_rotation_index = final_index_by_pre_index.get(rotation_index) if rotation_index is not None else None
        canonical_angle = (
            canonical_angles[rotation_index] if rotation_index is not None and len(record.source_sites) == 2 else None
        )
        final_mapping.append(
            BallarinLogicalToNativeMapping(
                source_logical_gate_index=record.source_logical_gate_index,
                logical_gate_id=record.logical_gate_id,
                source_gate_name=record.source_gate_name,
                source_sites=record.source_sites,
                source_parameter_index=record.source_parameter_index,
                pre_pruning_native_gate_ids=tuple(native_ids[index] for index in group_indices),
                retained_native_gate_ids=tuple(native_ids[index] for index in retained_group_indices),
                final_native_gate_indices=tuple(final_index_by_pre_index[index] for index in retained_group_indices),
                native_rotation_gate_id=native_rotation_id,
                final_native_rotation_gate_index=final_rotation_index,
                resolved_native_angle=(resolved_angles[rotation_index] if rotation_index is not None else None),
                canonical_rzz_angle=canonical_angle,
                canonical_rzz_magnitude=abs(canonical_angle) if canonical_angle is not None else None,
                rotation_pruned=rotation_index in pruned_rotation_indices,
                omitted_basis_change_native_gate_ids=tuple(
                    native_ids[index]
                    for index in (*record.basis_change_before_indices, *record.basis_change_after_indices)
                    if index in omitted_basis_indices
                ),
                cancelled_basis_change_native_gate_ids=tuple(
                    native_ids[index]
                    for index in (*record.basis_change_before_indices, *record.basis_change_after_indices)
                    if index in cancelled_basis_indices
                ),
            )
        )

    return BallarinCircuitMaterialization(
        circuit=final_circuit,
        mapping=tuple(final_mapping),
        pre_pruning_to_final_indices=pre_pruning_to_final,
        pruned_native_rotation_ids=tuple(
            native_ids[index] for index in range(len(native_ids)) if index in pruned_rotation_indices
        ),
        omitted_basis_change_native_gate_ids=tuple(
            native_ids[index] for index in range(len(native_ids)) if index in omitted_basis_indices
        ),
        cancelled_basis_change_native_gate_ids=tuple(
            native_ids[index] for index in range(len(native_ids)) if index in cancelled_basis_indices
        ),
    )


@dataclass(frozen=True, slots=True)
class BallarinNoiseProvider:
    """Exact independent product-Pauli provider for retained native rotations."""

    @property
    def noise_id(self) -> str:
        """The fixed benchmark noise identifier."""
        return BALLARIN_NOISE_ID

    @staticmethod
    def to_dict() -> dict[str, object]:
        """Return the fixed provider definition as deterministic JSON data."""
        return {
            "noise_id": BALLARIN_NOISE_ID,
            "gate_name": "rzz",
            "gate_placement": "post_gate",
            "angle_convention": "canonical_magnitude",
            "canonical_interval": "[-pi, pi)",
            "pruning_threshold": BALLARIN_PRUNING_THRESHOLD,
            "epsilon_intercept": BALLARIN_EPSILON_INTERCEPT,
            "epsilon_slope": BALLARIN_EPSILON_SLOPE,
            "channel": "independent_product_pauli",
            "single_qubit_gates": "noiseless",
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> BallarinNoiseProvider:
        """Restore and strictly validate the fixed provider definition.

        Args:
            data: Serialized provider definition.

        Returns:
            A validated stateless provider.

        Raises:
            TypeError: If ``data`` is not a mapping.
            ValueError: If any key, scalar type, or value differs.
        """
        if not isinstance(data, Mapping):
            msg = "BallarinNoiseProvider data must be a mapping."
            raise TypeError(msg)
        provider = cls()
        expected = provider.to_dict()
        if set(data) != set(expected):
            missing = sorted(set(expected) - set(data), key=repr)
            extra = sorted(set(data) - set(expected), key=repr)
            msg = f"BallarinNoiseProvider keys mismatch: missing={missing!r}, extra={extra!r}."
            raise ValueError(msg)
        for key, expected_value in expected.items():
            actual = data[key]
            if type(actual) is not type(expected_value) or actual != expected_value:
                msg = f"BallarinNoiseProvider field {key!r} does not match the fixed definition."
                raise ValueError(msg)
        return provider

    def __call__(
        self,
        context: GateNoiseContext,
        rng: np.random.Generator,
        /,
    ) -> RandomUnitaryInstruction | None:
        """Sample one post-``RZZ`` product channel from two independent draws.

        Args:
            context: Metadata for the resolved circuit gate.
            rng: Trajectory-local random-number generator.

        Returns:
            A tagged realized channel, or ``None`` for a non-entangling gate or
            a rotation that should have been pruned.

        Raises:
            TypeError: If ``context`` is not a gate-noise context.
            ValueError: If a two-qubit context is not a resolved native
                ``RZZ``.
        """
        if not isinstance(context, GateNoiseContext):
            msg = f"context must be a GateNoiseContext, got {type(context).__name__}."
            raise TypeError(msg)
        if context.arity == 1 and context.gate_name != "rzz":
            return None
        if context.arity != 2 or context.gate_name != "rzz":
            msg = (
                "Ballarin noise requires a two-qubit native RZZ context, got "
                f"gate_name={context.gate_name!r}, arity={context.arity}."
            )
            raise ValueError(msg)
        if context.resolved_angle is None:
            msg = "Ballarin noise requires a resolved native RZZ angle."
            raise ValueError(msg)

        canonical_angle = canonicalize_rzz_angle(cast("Real", context.resolved_angle))
        magnitude = abs(canonical_angle)
        if magnitude <= BALLARIN_PRUNING_THRESHOLD:
            return None
        rate = ballarin_local_pauli_rate(cast("Real", magnitude))
        distribution = cast(
            "PauliDistribution",
            {
                "I": 1.0 - 3.0 * rate,
                "X": rate,
                "Y": rate,
                "Z": rate,
            },
        )
        first_site, second_site = context.sites
        operators = sample_product_pauli_channel(
            first_site,
            second_site,
            distribution,
            distribution,
            rng,
        )
        labels_by_site = {
            first_site: "I",
            second_site: "I",
        }
        for operator in operators:
            assert operator.label is not None
            labels_by_site[operator.sites[0]] = operator.label
        return RandomUnitaryInstruction(
            operators=operators,
            channel_id=BALLARIN_NOISE_ID,
            outcome_labels=(labels_by_site[first_site], labels_by_site[second_site]),
        )


def create_ballarin_noise_provider() -> BallarinNoiseProvider:
    """Create the fixed Ballarin benchmark noise provider.

    Returns:
        A stateless validated product-Pauli provider.
    """
    return BallarinNoiseProvider()


__all__ = [
    "BALLARIN_EPSILON_INTERCEPT",
    "BALLARIN_EPSILON_SLOPE",
    "BALLARIN_MAX_EPSILON",
    "BALLARIN_PRUNING_THRESHOLD",
    "BallarinCircuitMaterialization",
    "BallarinLogicalToNativeMapping",
    "BallarinNoiseProvider",
    "FrozenNativeCircuit",
    "FrozenNativeGate",
    "ballarin_epsilon",
    "ballarin_local_pauli_probability",
    "ballarin_local_pauli_rate",
    "canonicalize_native_rzz_angle",
    "canonicalize_rzz_angle",
    "create_ballarin_noise_provider",
    "materialize_ballarin_circuit",
]
