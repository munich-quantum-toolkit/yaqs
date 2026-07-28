# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Final-circuit statistics for the state-preparation benchmarks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast

from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

from .ballarin import BallarinCircuitMaterialization, FrozenNativeGate
from .circuits import NativeCompilation, compile_quantinuum_native
from .schema import AnsatzConfig, CircuitStatistics

EvaluatedRepresentation: TypeAlias = Literal["logical", "native"]
NativeCircuitSource: TypeAlias = NativeCompilation | BallarinCircuitMaterialization
_CircuitGate: TypeAlias = ParameterizedGate | FrozenNativeGate


@dataclass(frozen=True, slots=True)
class _CircuitCounts:
    """Internal aggregate counts for one circuit representation."""

    depth: int
    num_1q_gates: int
    num_2q_gates: int
    by_name: dict[str, int]


class _GateWithInteraction(Protocol):
    """Structural type needed from a gate-library instance."""

    interaction: int


def _instantiate_library_gate(gate: _CircuitGate) -> _GateWithInteraction:
    """Instantiate a library gate only to inspect its interaction arity.

    Returns:
        The corresponding gate-library object.
    """
    gate_factory = getattr(GateLibrary, gate.name)
    if gate.is_parametric:
        return cast("_GateWithInteraction", gate_factory([0.0]))
    if gate.fixed_params:
        return cast("_GateWithInteraction", gate_factory(list(gate.fixed_params)))
    return cast("_GateWithInteraction", gate_factory())


def _semantic_gate_arity(gate: _CircuitGate, gate_index: int, representation: str) -> int:
    """Return and validate the matrix arity of one circuit gate.

    Returns:
        The number of qubits on which the gate matrix acts.

    Raises:
        ValueError: If the gate factory cannot be resolved or its matrix arity
            disagrees with the recorded sites.
    """
    try:
        library_gate = _instantiate_library_gate(gate)
    except (AttributeError, TypeError, ValueError) as error:
        msg = f"Cannot inspect {representation} gate {gate_index} named {gate.name!r}: {error}."
        raise ValueError(msg) from error

    arity = int(library_gate.interaction)
    if arity != len(gate.sites):
        msg = (
            f"{representation.capitalize()} gate {gate_index} named {gate.name!r} "
            f"has semantic arity {arity}, but sites {gate.sites!r} specify "
            f"arity {len(gate.sites)}."
        )
        raise ValueError(msg)
    if arity not in {1, 2}:
        msg = (
            f"{representation.capitalize()} gate {gate_index} named {gate.name!r} "
            f"has unsupported arity {arity}; only one- and two-qubit gates can "
            "be reported."
        )
        raise ValueError(msg)
    return arity


def _count_circuit(circuit: ParameterizedCircuit, representation: str) -> _CircuitCounts:
    """Count gates and dependency layers without binding any gate angles.

    Gates on disjoint qubits occupy the same layer. A gate acting on sites
    ``S`` is assigned to one layer after the latest layer already used by any
    site in ``S``. This is the same dependency-depth convention used by
    Qiskit's circuit depth calculation for unitary circuits.

    Returns:
        The depth, aggregate arity counts, and counts by gate name.
    """
    site_depths = [0] * circuit.num_qubits
    num_1q_gates = 0
    num_2q_gates = 0
    by_name: Counter[str] = Counter()

    for gate_index, gate in enumerate(cast("list[_CircuitGate]", circuit.gates)):
        arity = _semantic_gate_arity(gate, gate_index, representation)
        gate_depth = 1 + max(site_depths[site] for site in gate.sites)
        for site in gate.sites:
            site_depths[site] = gate_depth
        if arity == 1:
            num_1q_gates += 1
        else:
            num_2q_gates += 1
        by_name[gate.name] += 1

    return _CircuitCounts(
        depth=max(site_depths, default=0),
        num_1q_gates=num_1q_gates,
        num_2q_gates=num_2q_gates,
        by_name=dict(sorted(by_name.items())),
    )


def _validate_native_source(
    logical_circuit: ParameterizedCircuit,
    native_source: NativeCircuitSource,
) -> None:
    """Ensure native provenance describes the supplied logical circuit.

    Raises:
        ValueError: If the native source belongs to a different logical
            circuit.
    """
    native_circuit = native_source.circuit
    if native_circuit.num_qubits != logical_circuit.num_qubits:
        msg = "Logical and native circuits must use the same number of qubits."
        raise ValueError(msg)
    if len(native_source.mapping) != len(logical_circuit.gates):
        msg = "Native provenance must contain exactly one record per logical gate."
        raise ValueError(msg)

    for expected_index, (logical_gate, record) in enumerate(
        zip(logical_circuit.gates, native_source.mapping, strict=True)
    ):
        if (
            record.source_logical_gate_index != expected_index
            or record.source_gate_name != logical_gate.name
            or record.source_sites != tuple(logical_gate.sites)
            or record.source_parameter_index != logical_gate.param_index
        ):
            msg = f"Native provenance record {expected_index} does not match the supplied logical circuit."
            raise ValueError(msg)


def collect_circuit_statistics(
    logical_circuit: ParameterizedCircuit,
    ansatz: AnsatzConfig,
    *,
    native_source: NativeCircuitSource | None = None,
    evaluated_representation: EvaluatedRepresentation = "logical",
) -> CircuitStatistics:
    """Collect all logical, native, and row-selected circuit statistics.

    If ``native_source`` is omitted, the logical circuit is compiled to the
    benchmark's Quantinuum-native basis for extended metadata. Pass a
    :class:`BallarinCircuitMaterialization` to count the final resolved,
    pruned, and basis-change-cancelled circuit. Passing a
    :class:`NativeCompilation` counts its unpruned native circuit.

    Standard-noise result rows should select ``"logical"``. Ballarin rows
    should pass their final materialization and select ``"native"``. In both
    cases the returned record preserves counts for both representations.

    Args:
        logical_circuit: Final logical primitive circuit.
        ansatz: Configuration that produced the logical circuit.
        native_source: Optional compiled or final materialized native circuit.
        evaluated_representation: Representation used for the result row.

    Returns:
        A validated :class:`CircuitStatistics` record.

    Raises:
        TypeError: If an argument has an unsupported type.
        ValueError: If the circuits or their provenance are inconsistent.
    """
    if not isinstance(logical_circuit, ParameterizedCircuit):
        msg = f"logical_circuit must be a ParameterizedCircuit, got {type(logical_circuit).__name__}."
        raise TypeError(msg)
    if not isinstance(ansatz, AnsatzConfig):
        msg = f"ansatz must be an AnsatzConfig, got {type(ansatz).__name__}."
        raise TypeError(msg)
    if evaluated_representation not in {"logical", "native"}:
        msg = "evaluated_representation must be either 'logical' or 'native'."
        raise ValueError(msg)
    if native_source is None:
        native_source = compile_quantinuum_native(logical_circuit)
    elif not isinstance(native_source, (NativeCompilation, BallarinCircuitMaterialization)):
        msg = (
            "native_source must be a NativeCompilation, a "
            "BallarinCircuitMaterialization, or None; "
            f"got {type(native_source).__name__}."
        )
        raise TypeError(msg)

    _validate_native_source(logical_circuit, native_source)
    logical = _count_circuit(logical_circuit, "logical")
    native = _count_circuit(native_source.circuit, "native")
    pruned_native_rzz_count = (
        native_source.pruned_native_rzz_count if isinstance(native_source, BallarinCircuitMaterialization) else 0
    )

    return CircuitStatistics(
        configured_bmpd_depth=ansatz.configured_bmpd_depth,
        num_parameters=logical_circuit.num_params,
        logical_depth=logical.depth,
        logical_num_1q_gates=logical.num_1q_gates,
        logical_num_2q_gates=logical.num_2q_gates,
        native_depth=native.depth,
        native_num_1q_gates=native.num_1q_gates,
        native_num_2q_gates=native.num_2q_gates,
        native_rzz_count=native.by_name.get("rzz", 0),
        pruned_native_rzz_count=pruned_native_rzz_count,
        evaluated_representation=evaluated_representation,
        logical_gate_counts=logical.by_name,
        native_gate_counts=native.by_name,
    )


__all__ = [
    "EvaluatedRepresentation",
    "NativeCircuitSource",
    "collect_circuit_statistics",
]
