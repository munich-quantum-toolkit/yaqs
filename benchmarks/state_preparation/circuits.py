# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit compilation for the state-preparation benchmarks.

The Ballarin/Quantinuum benchmark convention uses ``RZZ`` as its only native
two-qubit rotation.  This module converts the shared logical ansatz into that
basis while retaining an explicit logical-to-native provenance map.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

BasisChangeRelationship: TypeAlias = Literal["none", "rxx_h", "ryy_rx_pi_over_2"]

_SUPPORTED_TWO_QUBIT_GATES = frozenset({"rxx", "ryy", "rzz"})
_NO_BASIS_CHANGE: BasisChangeRelationship = "none"
_RXX_BASIS_CHANGE: BasisChangeRelationship = "rxx_h"
_RYY_BASIS_CHANGE: BasisChangeRelationship = "ryy_rx_pi_over_2"


@dataclass(frozen=True, slots=True)
class NativeAngleExpression:
    """An affine native-gate angle retained without binding its inputs.

    Attributes:
        param_index: Index into the trainable parameter vector, or ``None``.
        angle_scale: Multiplicative factor of the trainable parameter.
        angle_offset: Constant contribution to the angle.
        data_map: Optional sample-dependent contribution to the angle.
    """

    param_index: int | None
    angle_scale: float
    angle_offset: float
    data_map: Callable[[NDArray[np.float64]], float] | None

    def evaluate(self, theta: NDArray[np.float64], x: NDArray[np.float64] | None = None) -> float:
        """Evaluate the retained angle expression.

        Args:
            theta: Trainable parameter vector.
            x: Input sample, required when this expression has a data map.

        Returns:
            The resolved rotation angle.

        Raises:
            ValueError: If the expression has a data map but no sample is
                provided.
        """
        angle = self.angle_offset
        if self.param_index is not None:
            angle += self.angle_scale * float(theta[self.param_index])
        if self.data_map is not None:
            if x is None:
                msg = "Native angle expression has a data map but no input sample was provided."
                raise ValueError(msg)
            angle += float(self.data_map(x))
        return angle


@dataclass(frozen=True, slots=True)
class LogicalToNativeMapping:
    """Trace one source gate to the native factors emitted for it.

    ``native_gate_indices`` contains every emitted factor in compile-time
    circuit order. For compiled entanglers, ``native_rotation_gate_index``
    identifies the single angle-preserving ``RZZ`` and the two basis-index
    tuples identify the complete round trip that must be removed with it if
    WP7 prunes the rotation. These are pre-pruning positions; stable
    ``native_gate_id`` values on the emitted gates remain the durable identity
    after downstream materialization.

    Attributes:
        source_logical_gate_index: Zero-based index in the source gate list.
        logical_gate_id: Effective logical identifier copied to every emitted
            gate. This is the source identifier when supplied, otherwise the
            source gate index.
        source_gate_name: Gate-library name of the source factor.
        source_sites: Source sites in their original order.
        source_parameter_index: Source trainable-parameter index, or ``None``.
        native_gate_indices: Compile-time indices of all emitted native
            factors.
        native_rotation_gate_index: Compile-time index of the corresponding
            native single-angle rotation, or ``None`` for a nonparametric
            passthrough.
        native_angle_expression: Unbound native angle expression, when present.
        basis_change_before_indices: Compile-time indices of native basis
            gates before the ``RZZ``.
        basis_change_after_indices: Compile-time indices of native basis gates
            after the ``RZZ``.
        basis_change_relationship: The exact basis-change convention used.
    """

    source_logical_gate_index: int
    logical_gate_id: int | str
    source_gate_name: str
    source_sites: tuple[int, ...]
    source_parameter_index: int | None
    native_gate_indices: tuple[int, ...]
    native_rotation_gate_index: int | None
    native_angle_expression: NativeAngleExpression | None
    basis_change_before_indices: tuple[int, ...]
    basis_change_after_indices: tuple[int, ...]
    basis_change_relationship: BasisChangeRelationship


@dataclass(frozen=True, slots=True)
class NativeCompilation:
    """A native circuit and its immutable logical-to-native mapping.

    The contained :class:`ParameterizedCircuit` follows its normal mutable API;
    the compilation records and their index tuples are immutable. Downstream
    pruning must build a new circuit and mapping atomically rather than mutate
    this gate list in place, because the recorded positions describe the
    pre-pruning circuit. Stable ``native_gate_id`` values provide the durable
    cross-stage identity.

    Attributes:
        circuit: Compiled circuit containing only ``RZZ`` two-qubit gates.
        mapping: One provenance record for every source logical gate.
    """

    circuit: ParameterizedCircuit
    mapping: tuple[LogicalToNativeMapping, ...]


def _copy_gate(
    source: ParameterizedGate,
    *,
    name: str,
    logical_gate_id: int | str,
    native_gate_id: int,
    noise_enabled: bool,
) -> ParameterizedGate:
    """Return a detached gate carrying the source angle and noise metadata."""
    return ParameterizedGate(
        name=name,
        sites=tuple(source.sites),
        param_index=source.param_index,
        angle_scale=source.angle_scale,
        angle_offset=source.angle_offset,
        data_map=source.data_map,
        fixed_params=tuple(source.fixed_params),
        logical_gate_id=logical_gate_id,
        native_gate_id=native_gate_id,
        noise_enabled=noise_enabled,
    )


def _basis_gate(
    name: str,
    site: int,
    *,
    logical_gate_id: int | str,
    native_gate_id: int,
    angle_offset: float = 0.0,
) -> ParameterizedGate:
    """Construct one noiseless compilation-only basis gate.

    Returns:
        The fixed one-qubit basis gate.
    """
    return ParameterizedGate(
        name=name,
        sites=(site,),
        angle_offset=angle_offset,
        logical_gate_id=logical_gate_id,
        native_gate_id=native_gate_id,
        noise_enabled=False,
    )


def _native_angle_expression(gate: ParameterizedGate) -> NativeAngleExpression | None:
    """Return the unbound angle expression of a native gate, if any."""
    if not gate.is_parametric:
        return None
    return NativeAngleExpression(
        param_index=gate.param_index,
        angle_scale=gate.angle_scale,
        angle_offset=gate.angle_offset,
        data_map=gate.data_map,
    )


def _semantic_gate_arity(gate: ParameterizedGate) -> int:
    """Return a source gate's matrix arity without resolving its angle map.

    Returns:
        The number of qubits on which the gate matrix acts.
    """
    gate_factory = getattr(GateLibrary, gate.name)
    if gate.is_parametric:
        library_gate = gate_factory([0.0])
    elif gate.fixed_params:
        library_gate = gate_factory(list(gate.fixed_params))
    else:
        library_gate = gate_factory()
    return int(library_gate.interaction)


def _validate_source_parameter_index(gate: ParameterizedGate, logical_gate_index: int) -> None:
    """Reject parameter indices that cannot safely index a parameter vector.

    Raises:
        TypeError: If the index is not an integer or is a Boolean.
        ValueError: If the index is negative.
    """
    param_index = gate.param_index
    if param_index is None:
        return
    if isinstance(param_index, (bool, np.bool_)) or not isinstance(param_index, Integral):
        msg = (
            f"Gate '{gate.name}' at logical gate index {logical_gate_index} has "
            f"a non-integer param_index {param_index!r}."
        )
        raise TypeError(msg)
    if param_index < 0:
        msg = (
            f"Gate '{gate.name}' at logical gate index {logical_gate_index} has a negative param_index {param_index!r}."
        )
        raise ValueError(msg)


def _append_basis_gate(
    native_gates: list[ParameterizedGate],
    name: str,
    site: int,
    *,
    logical_gate_id: int | str,
    angle_offset: float = 0.0,
) -> int:
    """Append one basis gate and return its stable native index.

    Returns:
        The appended gate's zero-based native index.
    """
    native_gate_index = len(native_gates)
    native_gates.append(
        _basis_gate(
            name,
            site,
            logical_gate_id=logical_gate_id,
            native_gate_id=native_gate_index,
            angle_offset=angle_offset,
        )
    )
    return native_gate_index


def _append_native_rotation(
    native_gates: list[ParameterizedGate],
    source: ParameterizedGate,
    *,
    name: str,
    logical_gate_id: int | str,
    noise_enabled: bool | None = None,
) -> int:
    """Append an angle-preserving native rotation and return its index.

    Returns:
        The appended gate's zero-based native index.
    """
    native_gate_index = len(native_gates)
    native_gates.append(
        _copy_gate(
            source,
            name=name,
            logical_gate_id=logical_gate_id,
            native_gate_id=native_gate_index,
            noise_enabled=source.noise_enabled if noise_enabled is None else noise_enabled,
        )
    )
    return native_gate_index


def _compile_rxx(
    native_gates: list[ParameterizedGate],
    source: ParameterizedGate,
    logical_gate_id: int | str,
) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    """Compile ``RXX`` as an exact Hadamard-conjugated ``RZZ``.

    Returns:
        The before-basis indices, central ``RZZ`` index, and after-basis
        indices.
    """
    first_site, second_site = source.sites
    before = (
        _append_basis_gate(native_gates, "h", first_site, logical_gate_id=logical_gate_id),
        _append_basis_gate(native_gates, "h", second_site, logical_gate_id=logical_gate_id),
    )
    rotation = _append_native_rotation(native_gates, source, name="rzz", logical_gate_id=logical_gate_id)
    after = (
        _append_basis_gate(native_gates, "h", second_site, logical_gate_id=logical_gate_id),
        _append_basis_gate(native_gates, "h", first_site, logical_gate_id=logical_gate_id),
    )
    return before, rotation, after


def _compile_ryy(
    native_gates: list[ParameterizedGate],
    source: ParameterizedGate,
    logical_gate_id: int | str,
) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    """Compile ``RYY`` using inverse ``RX(pi/2)`` basis changes.

    Returns:
        The before-basis indices, central ``RZZ`` index, and after-basis
        indices.
    """
    first_site, second_site = source.sites
    before = (
        _append_basis_gate(
            native_gates,
            "rx",
            first_site,
            logical_gate_id=logical_gate_id,
            angle_offset=np.pi / 2,
        ),
        _append_basis_gate(
            native_gates,
            "rx",
            second_site,
            logical_gate_id=logical_gate_id,
            angle_offset=np.pi / 2,
        ),
    )
    rotation = _append_native_rotation(native_gates, source, name="rzz", logical_gate_id=logical_gate_id)
    after = (
        _append_basis_gate(
            native_gates,
            "rx",
            second_site,
            logical_gate_id=logical_gate_id,
            angle_offset=-np.pi / 2,
        ),
        _append_basis_gate(
            native_gates,
            "rx",
            first_site,
            logical_gate_id=logical_gate_id,
            angle_offset=-np.pi / 2,
        ),
    )
    return before, rotation, after


def compile_quantinuum_native(circuit: ParameterizedCircuit) -> NativeCompilation:
    """Compile a logical circuit to the benchmark's Quantinuum-native basis.

    One-qubit gates are copied with their angle metadata intact and marked
    noiseless, as required by the Ballarin benchmark convention. Native ``RZZ``
    gates retain their noise eligibility. ``RXX`` and ``RYY`` each become an
    exact noiseless one-qubit basis-change round trip around exactly one
    ``RZZ`` carrying the original angle expression. No angle is bound,
    canonicalized, or pruned.

    Args:
        circuit: Logical parameterized circuit to compile.

    Returns:
        The detached native circuit and a complete provenance mapping.

    Raises:
        TypeError: If ``circuit`` is not a :class:`ParameterizedCircuit` or a
            source parameter index is not an integer.
        ValueError: If a parameter index is negative, a gate's semantic arity
            disagrees with its sites, or a two-qubit gate has no verified
            native decomposition.
    """
    if not isinstance(circuit, ParameterizedCircuit):
        msg = f"circuit must be a ParameterizedCircuit, got {type(circuit).__name__}."
        raise TypeError(msg)

    native_gates: list[ParameterizedGate] = []
    mapping: list[LogicalToNativeMapping] = []

    for logical_gate_index, source in enumerate(circuit.gates):
        logical_gate_id = source.logical_gate_id if source.logical_gate_id is not None else logical_gate_index
        _validate_source_parameter_index(source, logical_gate_index)
        semantic_arity = _semantic_gate_arity(source)
        if semantic_arity != len(source.sites):
            msg = (
                f"Gate '{source.name}' at logical gate index {logical_gate_index} has "
                f"semantic arity {semantic_arity}, but sites {source.sites!r} specify "
                f"arity {len(source.sites)}."
            )
            raise ValueError(msg)

        if semantic_arity == 1:
            native_gate_index = _append_native_rotation(
                native_gates,
                source,
                name=source.name,
                logical_gate_id=logical_gate_id,
                noise_enabled=False,
            )
            native_gate = native_gates[native_gate_index]
            mapping.append(
                LogicalToNativeMapping(
                    source_logical_gate_index=logical_gate_index,
                    logical_gate_id=logical_gate_id,
                    source_gate_name=source.name,
                    source_sites=tuple(source.sites),
                    source_parameter_index=source.param_index,
                    native_gate_indices=(native_gate_index,),
                    native_rotation_gate_index=native_gate_index if native_gate.is_parametric else None,
                    native_angle_expression=_native_angle_expression(native_gate),
                    basis_change_before_indices=(),
                    basis_change_after_indices=(),
                    basis_change_relationship=_NO_BASIS_CHANGE,
                )
            )
            continue

        if source.name not in _SUPPORTED_TWO_QUBIT_GATES:
            msg = (
                f"Unsupported two-qubit gate '{source.name}' at logical gate index "
                f"{logical_gate_index} on sites {source.sites!r}; only rxx, ryy, and rzz "
                "have verified Quantinuum-native rewrites."
            )
            raise ValueError(msg)

        if source.name == "rxx":
            before, native_rotation_gate_index, after = _compile_rxx(native_gates, source, logical_gate_id)
            relationship = _RXX_BASIS_CHANGE
        elif source.name == "ryy":
            before, native_rotation_gate_index, after = _compile_ryy(native_gates, source, logical_gate_id)
            relationship = _RYY_BASIS_CHANGE
        else:
            before = ()
            native_rotation_gate_index = _append_native_rotation(
                native_gates,
                source,
                name="rzz",
                logical_gate_id=logical_gate_id,
            )
            after = ()
            relationship = _NO_BASIS_CHANGE

        native_gate_indices = (*before, native_rotation_gate_index, *after)
        native_rotation = native_gates[native_rotation_gate_index]
        mapping.append(
            LogicalToNativeMapping(
                source_logical_gate_index=logical_gate_index,
                logical_gate_id=logical_gate_id,
                source_gate_name=source.name,
                source_sites=tuple(source.sites),
                source_parameter_index=source.param_index,
                native_gate_indices=native_gate_indices,
                native_rotation_gate_index=native_rotation_gate_index,
                native_angle_expression=_native_angle_expression(native_rotation),
                basis_change_before_indices=before,
                basis_change_after_indices=after,
                basis_change_relationship=relationship,
            )
        )

    native_circuit = ParameterizedCircuit(
        num_qubits=circuit.num_qubits,
        gates=native_gates,
        num_params=circuit.num_params,
    )
    return NativeCompilation(circuit=native_circuit, mapping=tuple(mapping))


__all__ = [
    "BasisChangeRelationship",
    "LogicalToNativeMapping",
    "NativeAngleExpression",
    "NativeCompilation",
    "compile_quantinuum_native",
]
