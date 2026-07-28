# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for final Ballarin native-circuit materialization."""

from __future__ import annotations

import copy
import math
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.ballarin import (
    BALLARIN_PRUNING_THRESHOLD,
    FrozenNativeCircuit,
    FrozenNativeGate,
    canonicalize_rzz_angle,
    materialize_ballarin_circuit,
)
from benchmarks.state_preparation.circuits import NativeCompilation, compile_quantinuum_native
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_PAULI_BY_ROTATION = {
    "rxx": _X,
    "ryy": _Y,
    "rzz": _Z,
}


def _embed_one_site(
    matrix: NDArray[np.complex128],
    site: int,
    num_qubits: int,
) -> NDArray[np.complex128]:
    """Embed a one-site matrix with site zero as the least-significant bit.

    Returns:
        The full-register matrix.
    """
    embedded = np.array([[1.0]], dtype=np.complex128)
    for qubit in range(num_qubits - 1, -1, -1):
        embedded = np.kron(embedded, matrix if qubit == site else _I2)
    return np.asarray(embedded, dtype=np.complex128)


def _embed_two_site(
    matrix: NDArray[np.complex128],
    first_site: int,
    second_site: int,
    num_qubits: int,
) -> NDArray[np.complex128]:
    """Embed an ascending-site YAQS two-qubit matrix into a full register.

    Returns:
        The full-register matrix.
    """
    dimension = 2**num_qubits
    embedded = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        bits = [(column >> qubit) & 1 for qubit in range(num_qubits)]
        local_column = 2 * bits[first_site] + bits[second_site]
        for local_row in range(4):
            output_bits = bits.copy()
            output_bits[first_site] = (local_row >> 1) & 1
            output_bits[second_site] = local_row & 1
            row = sum(bit << qubit for qubit, bit in enumerate(output_bits))
            embedded[row, column] += matrix[local_row, local_column]
    return embedded


def _dense_unitary(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64] | None = None,
) -> NDArray[np.complex128]:
    """Return the full dense unitary of a parameterized circuit."""
    dimension = 2**circuit.num_qubits
    unitary = np.eye(dimension, dtype=np.complex128)
    for gate in circuit.gates:
        matrix, sites = circuit.gate_matrix(gate, theta, x)
        if len(sites) == 1:
            embedded = _embed_one_site(matrix, sites[0], circuit.num_qubits)
        else:
            embedded = _embed_two_site(matrix, sites[0], sites[1], circuit.num_qubits)
        unitary = embedded @ unitary
    return unitary


def _rotation_oracle(name: str, angle: float) -> NDArray[np.complex128]:
    """Return an independent dense Pauli-product rotation."""
    generator = np.kron(_PAULI_BY_ROTATION[name], _PAULI_BY_ROTATION[name])
    return np.asarray(
        np.cos(angle / 2.0) * np.eye(4, dtype=np.complex128) - 1.0j * np.sin(angle / 2.0) * generator,
        dtype=np.complex128,
    )


def _assert_unitaries_equal_up_to_global_phase(
    actual: NDArray[np.complex128],
    expected: NDArray[np.complex128],
    *,
    atol: float = 1e-10,
) -> complex:
    """Assert dense-unitary equality up to one scalar global phase.

    Returns:
        The phase multiplying ``expected`` to produce ``actual``.
    """
    overlap = np.trace(expected.conj().T @ actual)
    assert abs(overlap) > 1e-12
    phase = complex(overlap / abs(overlap))
    np.testing.assert_allclose(actual, phase * expected, atol=atol, rtol=atol)
    return phase


def _gate_snapshot(gate: ParameterizedGate) -> tuple[object, ...]:
    """Return all mutable source-gate fields as an immutable snapshot."""
    return (
        gate.name,
        gate.sites,
        gate.param_index,
        gate.angle_scale,
        gate.angle_offset,
        gate.data_map,
        gate.fixed_params,
        gate.logical_gate_id,
        gate.native_gate_id,
        gate.noise_enabled,
    )


def _assign_attribute(instance: object, name: str, value: object) -> None:
    """Attempt a dynamic assignment used to exercise frozen public records."""
    setattr(instance, name, value)


def _delete_attribute(instance: object, name: str) -> None:
    """Attempt dynamic deletion used to exercise frozen public records."""
    delattr(instance, name)


def _assign_mapping_item(instance: object, key: object, value: object) -> None:
    """Attempt dynamic mutation used to exercise read-only parameter mappings."""
    cast("dict[object, object]", instance)[key] = value


def _constant_data_map(_: NDArray[np.float64]) -> float:
    """Return a fixed contribution for mutation-validation tests."""
    return 0.0


_PI_BELOW = float(np.nextafter(math.pi, -math.inf))


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (-0.0, 0.0),
        (math.pi, -math.pi),
        (-math.pi, -math.pi),
        (3.0 * math.pi, -math.pi),
        (-3.0 * math.pi, -math.pi),
        (2.0 * math.pi, 0.0),
        (-2.0 * math.pi, 0.0),
        (_PI_BELOW, _PI_BELOW),
        (float(np.nextafter(math.pi, math.inf)), -_PI_BELOW),
        (float(np.nextafter(-math.pi, -math.inf)), _PI_BELOW),
        (float(np.nextafter(-math.pi, math.inf)), -_PI_BELOW),
    ],
)
def test_canonicalize_rzz_angle_has_exact_half_open_boundary(
    angle: float,
    expected: float,
) -> None:
    """Canonicalization should wrap robustly at half turns and their ULP neighbors."""
    actual = canonicalize_rzz_angle(angle)

    assert actual == expected
    assert -math.pi <= actual < math.pi
    if not expected:
        assert not np.signbit(actual)


@pytest.mark.parametrize(
    ("angle", "expected_pruned"),
    [
        (float(np.nextafter(BALLARIN_PRUNING_THRESHOLD, 0.0)), True),
        (BALLARIN_PRUNING_THRESHOLD, True),
        (float(np.nextafter(BALLARIN_PRUNING_THRESHOLD, math.inf)), False),
        (float(np.nextafter(-BALLARIN_PRUNING_THRESHOLD, 0.0)), True),
        (-BALLARIN_PRUNING_THRESHOLD, True),
        (float(np.nextafter(-BALLARIN_PRUNING_THRESHOLD, -math.inf)), False),
    ],
)
def test_materialization_uses_exact_inclusive_pruning_threshold(
    angle: float,
    *,
    expected_pruned: bool,
) -> None:
    """The exact threshold should prune inclusively without an isclose band."""
    logical = ParameterizedCircuit(2, [ParameterizedGate("rzz", (1, 0), angle_offset=angle)])
    materialization = materialize_ballarin_circuit(
        compile_quantinuum_native(logical),
        np.array([], dtype=np.float64),
    )
    mapping = materialization.mapping[0]

    assert mapping.canonical_rzz_angle == angle
    assert mapping.canonical_rzz_magnitude == abs(angle)
    assert mapping.rotation_pruned is expected_pruned
    assert materialization.pruned_native_rotation_ids == ((0,) if expected_pruned else ())
    if expected_pruned:
        assert materialization.circuit.gates == ()
        assert materialization.pre_pruning_to_final_indices == (None,)
    else:
        assert len(materialization.circuit.gates) == 1
        assert materialization.circuit.gates[0].angle_offset == angle
        assert materialization.circuit.gates[0].noise_enabled is True


@pytest.mark.parametrize(
    ("angle", "expected_pruned"),
    [
        (2.0 * math.pi, True),
        (2.0 * math.pi + BALLARIN_PRUNING_THRESHOLD / 2.0, True),
        (-2.0 * math.pi - 2.0 * BALLARIN_PRUNING_THRESHOLD, False),
        (3.0 * math.pi, False),
    ],
)
def test_materialization_prunes_only_after_wrapping(
    angle: float,
    *,
    expected_pruned: bool,
) -> None:
    """Wrapped angles should be canonicalized before threshold comparison."""
    logical = ParameterizedCircuit(2, [ParameterizedGate("rzz", (0, 1), angle_offset=angle)])
    materialization = materialize_ballarin_circuit(
        compile_quantinuum_native(logical),
        np.array([], dtype=np.float64),
    )
    mapping = materialization.mapping[0]
    expected_canonical = canonicalize_rzz_angle(angle)

    assert mapping.resolved_native_angle == angle
    assert mapping.canonical_rzz_angle == expected_canonical
    assert mapping.canonical_rzz_magnitude == abs(expected_canonical)
    assert mapping.rotation_pruned is expected_pruned


@pytest.mark.parametrize(
    ("gate_name", "angle"),
    [
        ("rxx", BALLARIN_PRUNING_THRESHOLD),
        ("ryy", 2.0 * math.pi + BALLARIN_PRUNING_THRESHOLD / 2.0),
    ],
)
@pytest.mark.parametrize("sites", [(0, 1), (1, 0)])
def test_pruned_compiled_entangler_omits_its_whole_native_block(
    gate_name: str,
    angle: float,
    sites: tuple[int, int],
) -> None:
    """Pruning RXX or RYY must remove its rotation and complete basis round trip."""
    logical = ParameterizedCircuit(2, [ParameterizedGate(gate_name, sites, angle_offset=angle)])
    compilation = compile_quantinuum_native(logical)
    pre_pruning_snapshot = tuple(_gate_snapshot(gate) for gate in compilation.circuit.gates)

    materialization = materialize_ballarin_circuit(compilation, np.array([], dtype=np.float64))
    mapping = materialization.mapping[0]

    assert materialization.circuit.gates == ()
    assert materialization.pruned_native_rotation_ids == (2,)
    assert materialization.omitted_basis_change_native_gate_ids == (0, 1, 3, 4)
    assert materialization.cancelled_basis_change_native_gate_ids == ()
    assert materialization.pre_pruning_to_final_indices == (None, None, None, None, None)
    assert mapping.pre_pruning_native_gate_ids == (0, 1, 2, 3, 4)
    assert mapping.retained_native_gate_ids == ()
    assert mapping.final_native_gate_indices == ()
    assert mapping.native_rotation_gate_id == 2
    assert mapping.final_native_rotation_gate_index is None
    assert mapping.rotation_pruned is True
    assert mapping.omitted_basis_change_native_gate_ids == (0, 1, 3, 4)
    assert tuple(_gate_snapshot(gate) for gate in compilation.circuit.gates) == pre_pruning_snapshot


@pytest.mark.parametrize("gate_name", ["rxx", "ryy"])
@pytest.mark.parametrize("second_sites", [(0, 1), (1, 0)])
def test_consecutive_compiled_entanglers_cancel_only_inverse_basis_boundaries(
    gate_name: str,
    second_sites: tuple[int, int],
) -> None:
    """Safe basis cancellation should work for equal and reversed source site orders."""
    logical = ParameterizedCircuit(
        2,
        [
            ParameterizedGate(gate_name, (0, 1), angle_offset=0.31),
            ParameterizedGate(gate_name, second_sites, angle_offset=-0.47),
        ],
    )
    compilation = compile_quantinuum_native(logical)

    materialization = materialize_ballarin_circuit(compilation, np.array([], dtype=np.float64))

    assert materialization.cancelled_basis_change_native_gate_ids == (3, 4, 5, 6)
    assert materialization.pre_pruning_to_final_indices == (0, 1, 2, None, None, None, None, 3, 4, 5)
    assert tuple(gate.native_gate_id for gate in materialization.circuit.gates) == (0, 1, 2, 7, 8, 9)
    assert tuple(gate.name for gate in materialization.circuit.gates) == (
        ("h", "h", "rzz", "rzz", "h", "h") if gate_name == "rxx" else ("rx", "rx", "rzz", "rzz", "rx", "rx")
    )
    assert materialization.mapping[0].cancelled_basis_change_native_gate_ids == (3, 4)
    assert materialization.mapping[1].cancelled_basis_change_native_gate_ids == (5, 6)
    assert materialization.mapping[0].retained_native_gate_ids == (0, 1, 2)
    assert materialization.mapping[1].retained_native_gate_ids == (7, 8, 9)
    assert materialization.mapping[1].final_native_rotation_gate_index == 3

    logical_dense = _dense_unitary(logical, np.array([], dtype=np.float64))
    final_dense = _dense_unitary(materialization.circuit, np.array([], dtype=np.float64))
    phase = _assert_unitaries_equal_up_to_global_phase(final_dense, logical_dense)
    assert phase == pytest.approx(1.0 + 0.0j)


def test_logical_and_mismatched_basis_gates_are_never_cancelled() -> None:
    """Logical H gates and unlike RXX/RYY basis boundaries must remain explicit."""
    logical = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("h", (0,), logical_gate_id="logical-h-0"),
            ParameterizedGate("h", (0,), logical_gate_id="logical-h-1"),
            ParameterizedGate("rxx", (0, 1), angle_offset=0.31),
            ParameterizedGate("ryy", (0, 1), angle_offset=-0.47),
        ],
    )

    materialization = materialize_ballarin_circuit(
        compile_quantinuum_native(logical),
        np.array([], dtype=np.float64),
    )

    assert materialization.cancelled_basis_change_native_gate_ids == ()
    assert materialization.pre_pruning_to_final_indices == tuple(range(12))
    assert tuple(gate.native_gate_id for gate in materialization.circuit.gates) == tuple(range(12))
    assert tuple(gate.logical_gate_id for gate in materialization.circuit.gates[:2]) == (
        "logical-h-0",
        "logical-h-1",
    )
    logical_dense = _dense_unitary(logical, np.array([], dtype=np.float64))
    final_dense = _dense_unitary(materialization.circuit, np.array([], dtype=np.float64))
    _assert_unitaries_equal_up_to_global_phase(final_dense, logical_dense)


def test_intervening_logical_gate_blocks_compiler_basis_cancellation() -> None:
    """Basis gates must not be commuted through a logical operation on their site."""
    logical = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rxx", (0, 1), angle_offset=0.31),
            ParameterizedGate("h", (0,), logical_gate_id="barrier"),
            ParameterizedGate("rxx", (0, 1), angle_offset=-0.47),
        ],
    )

    materialization = materialize_ballarin_circuit(
        compile_quantinuum_native(logical),
        np.array([], dtype=np.float64),
    )

    assert materialization.cancelled_basis_change_native_gate_ids == ()
    assert len(materialization.circuit.gates) == 11
    assert sum(gate.logical_gate_id == "barrier" for gate in materialization.circuit.gates) == 1
    logical_dense = _dense_unitary(logical, np.array([], dtype=np.float64))
    final_dense = _dense_unitary(materialization.circuit, np.array([], dtype=np.float64))
    _assert_unitaries_equal_up_to_global_phase(final_dense, logical_dense)


@pytest.mark.parametrize("gate_name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("sites", [(0, 1), (1, 0)])
def test_positive_pi_materialization_matches_independent_dense_reference_up_to_minus_phase(
    gate_name: str,
    sites: tuple[int, int],
) -> None:
    """Mapping positive pi to negative pi should change only the global phase."""
    logical = ParameterizedCircuit(
        2,
        [ParameterizedGate(gate_name, sites, angle_offset=math.pi)],
    )

    materialization = materialize_ballarin_circuit(
        compile_quantinuum_native(logical),
        np.array([], dtype=np.float64),
    )
    actual = _dense_unitary(materialization.circuit, np.array([], dtype=np.float64))
    expected = _rotation_oracle(gate_name, math.pi)

    assert materialization.mapping[0].canonical_rzz_angle == -math.pi
    np.testing.assert_allclose(actual, -expected, atol=1e-12, rtol=1e-12)
    phase = _assert_unitaries_equal_up_to_global_phase(actual, expected)
    assert phase == pytest.approx(-1.0 + 0.0j)


def test_frozen_snapshot_and_mutable_clones_are_deeply_isolated() -> None:
    """The final snapshot should remain authoritative across independent mutable clones."""
    source_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    logical = ParameterizedCircuit(
        2,
        [
            ParameterizedGate(
                "unitary",
                (0,),
                fixed_params=cast("tuple[float, ...]", (source_matrix,)),
                logical_gate_id="fixed",
            ),
            ParameterizedGate("rx", (1,), angle_offset=0.2, logical_gate_id="rotation"),
            ParameterizedGate("rzz", (0, 1), angle_offset=0.3, logical_gate_id="native"),
        ],
    )
    materialization = materialize_ballarin_circuit(
        compile_quantinuum_native(logical),
        np.array([], dtype=np.float64),
    )

    assert isinstance(materialization.circuit, FrozenNativeCircuit)
    assert isinstance(materialization.circuit.gates, tuple)
    assert all(isinstance(gate, FrozenNativeGate) for gate in materialization.circuit.gates)
    assert materialization.circuit.num_params == 0
    assert copy.copy(materialization.circuit) is materialization.circuit
    assert copy.deepcopy(materialization.circuit) is materialization.circuit
    frozen_matrix = cast("NDArray[np.complex128]", materialization.circuit.gates[0].fixed_params[0])
    assert not frozen_matrix.flags["W"]
    assert not np.shares_memory(frozen_matrix, source_matrix)
    np.testing.assert_array_equal(frozen_matrix, source_matrix)

    with pytest.raises(FrozenInstanceError):
        _assign_attribute(materialization.circuit, "num_qubits", 7)
    with pytest.raises(FrozenInstanceError):
        _assign_attribute(materialization.circuit.gates[1], "angle_offset", 9.0)
    with pytest.raises(FrozenInstanceError):
        _assign_attribute(materialization.mapping[0], "source_gate_name", "changed")
    for attribute in ("_frozen", "gates", "num_params"):
        with pytest.raises(FrozenInstanceError):
            _delete_attribute(materialization.circuit, attribute)
    with pytest.raises(ValueError, match="read-only"):
        frozen_matrix[0, 0] = 4.0
    with pytest.raises(TypeError):
        materialization.circuit.__dict__["num_qubits"] = 7

    reshaped_view = cast("NDArray[np.complex128]", materialization.circuit.gates[0].fixed_params[0])
    reshaped_view.shape = (4,)
    assert cast("NDArray[np.complex128]", materialization.circuit.gates[0].fixed_params[0]).shape == (2, 2)

    first_clone = materialization.to_parameterized_circuit()
    second_clone = materialization.to_parameterized_circuit()
    first_matrix = cast("NDArray[np.complex128]", first_clone.gates[0].fixed_params[0])
    second_matrix = cast("NDArray[np.complex128]", second_clone.gates[0].fixed_params[0])
    assert first_clone is not second_clone
    assert first_clone.gates[0] is not second_clone.gates[0]
    assert not np.shares_memory(first_matrix, second_matrix)
    assert not np.shares_memory(first_matrix, frozen_matrix)

    first_matrix[0, 0] = 8.0
    first_clone.gates[1].angle_offset = -5.0
    first_clone.gates.append(ParameterizedGate("h", (0,)))

    np.testing.assert_array_equal(frozen_matrix, source_matrix)
    np.testing.assert_array_equal(second_matrix, source_matrix)
    assert materialization.circuit.gates[1].angle_offset == pytest.approx(0.2)
    assert second_clone.gates[1].angle_offset == pytest.approx(0.2)
    assert len(materialization.circuit.gates) == len(second_clone.gates) == 3


def test_frozen_gate_deeply_detaches_container_fixed_parameters() -> None:
    """Nested mutable containers must not alias a frozen gate or its views."""
    angles = [0.1]
    labels = {"original"}
    payload = {"angles": angles, "labels": labels}
    gate = FrozenNativeGate("u", (0,), fixed_params=(payload,))

    angles.append(0.2)
    labels.add("source-only")
    frozen_payload = cast("Mapping[str, object]", gate.fixed_params[0])

    assert frozen_payload["angles"] == (0.1,)
    assert frozen_payload["labels"] == frozenset({"original"})
    with pytest.raises(TypeError):
        _assign_mapping_item(frozen_payload, "new", 3)

    mutable_payload = cast("dict[str, object]", gate.to_parameterized_gate().fixed_params[0])
    mutable_angles = cast("list[float]", mutable_payload["angles"])
    assert isinstance(mutable_angles, list)
    mutable_angles.append(0.3)
    mutable_payload["new"] = 3
    assert cast("Mapping[str, object]", gate.fixed_params[0])["angles"] == (0.1,)
    assert "new" not in cast("Mapping[str, object]", gate.fixed_params[0])


def test_frozen_gate_snapshots_mutable_scalar_representations() -> None:
    """Structured scalars and built-in subclasses must not alias their sources."""
    mutable_string_type = type("_MutableString", (str,), {})
    source_text = mutable_string_type("value")
    source_payload: list[str] = []
    _assign_attribute(source_text, "payload", source_payload)
    source_scalar = np.zeros((), dtype=[("count", "i8"), ("weight", "f8")])[()]
    gate = FrozenNativeGate(
        "u",
        (0,),
        fixed_params=(
            source_text,
            source_scalar,
            np.str_(""),
            np.bytes_(b""),
            np.void(b""),
            np.empty((0,), dtype="V0"),
        ),
    )

    source_payload.append("mutated")
    source_scalar["count"] = 7
    frozen_text, frozen_scalar_value, frozen_string, frozen_bytes, frozen_void_value, zero_width_array_value = (
        gate.fixed_params
    )
    frozen_scalar = cast("np.void", frozen_scalar_value)
    frozen_void = cast("np.void", frozen_void_value)
    zero_width_array = cast("NDArray[np.void]", zero_width_array_value)

    assert type(frozen_text) is str
    assert frozen_text == "value"
    assert not hasattr(frozen_text, "payload")
    assert frozen_scalar["count"] == 0
    assert type(frozen_string) is str
    assert type(frozen_bytes) is bytes
    assert frozen_void.dtype.itemsize == 0
    assert zero_width_array.shape == (0,)
    assert not zero_width_array.flags["W"]
    with pytest.raises(ValueError, match="read-only"):
        frozen_scalar["count"] = 9


def test_zero_dimensional_fixed_parameter_arrays_preserve_gate_semantics() -> None:
    """Freezing must preserve scalar-array shape for fixed gate parameters."""
    fixed_params = tuple(np.array(value) for value in (0.1, 0.2, 0.3))
    logical = ParameterizedCircuit(
        1,
        [
            ParameterizedGate(
                "u",
                (0,),
                fixed_params=cast("tuple[float, ...]", fixed_params),
            )
        ],
    )
    expected = _dense_unitary(logical, np.array([], dtype=np.float64))

    materialization = materialize_ballarin_circuit(
        compile_quantinuum_native(logical),
        np.array([], dtype=np.float64),
    )
    actual = _dense_unitary(materialization.circuit, np.array([], dtype=np.float64))

    assert all(
        cast("NDArray[np.generic]", value).shape == () for value in materialization.circuit.gates[0].fixed_params
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_numeric_array_dtype_metadata_is_detached_and_stripped() -> None:
    """Mutable dtype metadata must not alias the authoritative snapshot."""
    metadata_payload: list[str] = []
    source_dtype = np.dtype(np.float64, metadata={"payload": metadata_payload})
    source_matrix = np.eye(2, dtype=source_dtype)
    gate = FrozenNativeGate(
        "unitary",
        (0,),
        fixed_params=cast("tuple[float, ...]", (source_matrix,)),
    )

    metadata_payload.append("source-only")
    frozen_matrix = cast("NDArray[np.float64]", gate.fixed_params[0])

    assert frozen_matrix.dtype.metadata is None
    np.testing.assert_array_equal(frozen_matrix, np.eye(2, dtype=np.float64))


def test_object_dtype_unitary_remains_executable_after_materialization() -> None:
    """Object-dtype numeric matrices need an element-backed immutable snapshot."""
    source_matrix = np.array(
        [
            [Decimal(1), Fraction(0, 1)],
            [Decimal(0), Fraction(1, 1)],
        ],
        dtype=object,
    )
    logical = ParameterizedCircuit(
        1,
        [
            ParameterizedGate(
                "unitary",
                (0,),
                fixed_params=cast("tuple[float, ...]", (source_matrix,)),
            )
        ],
    )

    materialization = materialize_ballarin_circuit(
        compile_quantinuum_native(logical),
        np.array([], dtype=np.float64),
    )
    frozen_matrix = cast("NDArray[np.object_]", materialization.circuit.gates[0].fixed_params[0])

    assert frozen_matrix.dtype == np.dtype(object)
    assert not frozen_matrix.flags["W"]
    np.testing.assert_array_equal(
        _dense_unitary(materialization.circuit, np.array([], dtype=np.float64)),
        np.eye(2, dtype=np.complex128),
    )

    source_matrix[0, 0] = 0
    np.testing.assert_array_equal(
        cast("NDArray[np.object_]", materialization.circuit.gates[0].fixed_params[0]),
        np.eye(2, dtype=object),
    )


@pytest.mark.parametrize(
    "source_matrix",
    [
        [[1.0, 0.0], [0.0, 1.0]],
        np.eye(2, dtype=np.float64).data,
    ],
    ids=("nested-lists", "multidimensional-memoryview"),
)
def test_fixed_unitary_container_semantics_survive_freezing(source_matrix: object) -> None:
    """Valid sequence and buffer parameters must remain executable and detachable."""
    logical = ParameterizedCircuit(
        1,
        [
            ParameterizedGate(
                "unitary",
                (0,),
                fixed_params=cast("tuple[float, ...]", (source_matrix,)),
            )
        ],
    )
    materialization = materialize_ballarin_circuit(
        compile_quantinuum_native(logical),
        np.array([], dtype=np.float64),
    )

    np.testing.assert_allclose(
        _dense_unitary(materialization.circuit, np.array([], dtype=np.float64)),
        np.eye(2, dtype=np.complex128),
        atol=1e-12,
        rtol=1e-12,
    )
    mutable_matrix = materialization.to_parameterized_circuit().gates[0].fixed_params[0]
    if isinstance(source_matrix, list):
        assert isinstance(mutable_matrix, list)
        assert all(isinstance(row, list) for row in mutable_matrix)
        cast("list[list[float]]", mutable_matrix)[0][0] = 0.0
    else:
        assert isinstance(mutable_matrix, np.ndarray)
        assert mutable_matrix.shape == (2, 2)
        mutable_matrix[0, 0] = 0.0
    np.testing.assert_allclose(
        _dense_unitary(materialization.circuit, np.array([], dtype=np.float64)),
        np.eye(2, dtype=np.complex128),
        atol=1e-12,
        rtol=1e-12,
    )


def test_frozen_circuit_constructor_rejects_semantic_arity_mismatch() -> None:
    """The public constructor must reject a two-qubit matrix on one site."""
    gate = FrozenNativeGate(
        "cx",
        (0,),
        logical_gate_id=0,
        native_gate_id=0,
    )

    with pytest.raises(ValueError, match="semantic arity 2"):
        FrozenNativeCircuit(1, (gate,))


def test_frozen_circuit_constructor_rejects_gate_subclasses() -> None:
    """Executable snapshots must not retain overridable subclass behavior."""
    gate_subclass = type("_GateSubclass", (FrozenNativeGate,), {})
    subclass_gate = gate_subclass(
        "h",
        (0,),
        logical_gate_id=0,
        native_gate_id=0,
    )

    with pytest.raises(TypeError, match="exact FrozenNativeGate"):
        FrozenNativeCircuit(1, (subclass_gate,))


@pytest.mark.parametrize(
    ("invalid_num_qubits", "expected_error"),
    [
        (True, TypeError),
        (np.zeros((), dtype=np.bool_)[()], TypeError),
        (1.5, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_frozen_circuit_constructor_rejects_invalid_qubit_counts(
    invalid_num_qubits: object,
    expected_error: type[Exception],
) -> None:
    """A frozen executable circuit requires a positive integral qubit count."""
    with pytest.raises(expected_error, match="num_qubits"):
        FrozenNativeCircuit(cast("int", invalid_num_qubits), ())


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_error"),
    [
        ("num_qubits", True, TypeError),
        ("num_qubits", 1.5, TypeError),
        ("num_qubits", 0, ValueError),
        ("num_params", True, TypeError),
        ("num_params", 0.0, TypeError),
        ("num_params", -1, ValueError),
    ],
)
def test_materialization_rejects_invalid_mutated_circuit_counts(
    field: str,
    invalid_value: object,
    expected_error: type[Exception],
) -> None:
    """Mutable WP6 count fields must be revalidated at the freeze boundary."""
    compilation = compile_quantinuum_native(ParameterizedCircuit(1, [], num_params=0))
    _assign_attribute(compilation.circuit, field, invalid_value)

    with pytest.raises(expected_error, match=field):
        materialize_ballarin_circuit(compilation, np.array([], dtype=np.float64))


def test_frozen_circuit_normalizes_numpy_integer_counts_and_sites() -> None:
    """Accepted NumPy integer metadata should become built-in integers."""
    gate = FrozenNativeGate(
        "h",
        cast("tuple[int, ...]", (np.int64(0),)),
        logical_gate_id=0,
        native_gate_id=0,
    )
    circuit = FrozenNativeCircuit(cast("int", np.int64(1)), (gate,))

    assert type(circuit.num_qubits) is int
    assert type(circuit.gates[0].sites[0]) is int
    assert circuit.num_qubits == 1
    assert circuit.gates[0].sites == (0,)


def test_materialization_normalizes_string_subclass_names_and_identifiers() -> None:
    """Mutable string-subclass state must not survive in executable metadata."""

    def mutable_hash(value: object) -> int:
        return cast("int", vars(value)["hash_value"])

    mutable_string_type = type("_MutableString", (str,), {"__hash__": mutable_hash})
    source_name = mutable_string_type("rx")
    source_identifier = mutable_string_type("logical")
    _assign_attribute(source_name, "hash_value", hash("rx"))
    _assign_attribute(source_identifier, "hash_value", hash("logical"))
    logical = ParameterizedCircuit(
        1,
        [
            ParameterizedGate(
                cast("str", source_name),
                (0,),
                angle_offset=0.2,
                logical_gate_id=cast("str", source_identifier),
            )
        ],
    )

    materialization = materialize_ballarin_circuit(
        compile_quantinuum_native(logical),
        np.array([], dtype=np.float64),
    )
    _assign_attribute(source_name, "hash_value", 0)
    _assign_attribute(source_identifier, "hash_value", 0)

    frozen_gate = materialization.circuit.gates[0]
    assert type(frozen_gate.name) is str
    assert type(frozen_gate.logical_gate_id) is str
    assert type(materialization.mapping[0].source_gate_name) is str
    assert type(materialization.mapping[0].logical_gate_id) is str
    assert frozen_gate.is_parametric
    assert frozen_gate.resolved_angle == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("invalid_sites", "expected_error"),
    [
        ((True,), TypeError),
        ((0.5,), TypeError),
        ((), ValueError),
        ((0, 0), ValueError),
        ((0, 1, 2), ValueError),
    ],
)
def test_frozen_gate_rejects_invalid_site_metadata(
    invalid_sites: tuple[object, ...],
    expected_error: type[Exception],
) -> None:
    """Frozen gate sites must be distinct non-Boolean integers of valid arity."""
    with pytest.raises(expected_error, match="sites"):
        FrozenNativeGate("h", cast("tuple[int, ...]", invalid_sites))


@pytest.mark.parametrize("invalid_sites", [(True,), (0.5,)])
def test_materialization_rejects_invalid_mutated_native_sites(
    invalid_sites: tuple[object, ...],
) -> None:
    """The detached WP6 snapshot must validate every native site index."""
    compilation = compile_quantinuum_native(
        ParameterizedCircuit(2, [ParameterizedGate("h", (0,))]),
    )
    _assign_attribute(
        compilation.circuit.gates[0],
        "sites",
        cast("tuple[int, ...]", invalid_sites),
    )

    with pytest.raises(TypeError, match=r"gate\[0\]\.sites"):
        materialize_ballarin_circuit(compilation, np.array([], dtype=np.float64))


@pytest.mark.parametrize(
    ("target", "field", "invalid_value"),
    [
        ("circuit", "num_qubits", 1),
        ("callback_gate", "noise_enabled", True),
        ("later_gate", "native_gate_id", -1),
        ("later_gate", "param_index", 0),
        ("later_gate", "data_map", _constant_data_map),
    ],
)
def test_mutated_compilation_is_rejected_before_any_data_callback(
    target: str,
    field: str,
    invalid_value: object,
) -> None:
    """All detached metadata invariants must be checked before callbacks run."""
    callback_count = 0

    def count_callback(_: NDArray[np.float64]) -> float:
        nonlocal callback_count
        callback_count += 1
        return 0.2

    compilation = compile_quantinuum_native(
        ParameterizedCircuit(
            2,
            [
                ParameterizedGate("rx", (0,), data_map=count_callback),
                ParameterizedGate("h", (1,)),
            ],
        )
    )
    if target == "circuit":
        mutation_target: object = compilation.circuit
    elif target == "callback_gate":
        mutation_target = compilation.circuit.gates[0]
    else:
        mutation_target = compilation.circuit.gates[1]
    _assign_attribute(mutation_target, field, invalid_value)

    with pytest.raises((TypeError, ValueError)):
        materialize_ballarin_circuit(
            compilation,
            np.array([], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )
    assert callback_count == 0


def test_near_zero_nonparametric_offset_remains_compatible() -> None:
    """Materialization should retain the base circuit's near-zero tolerance."""
    compilation = compile_quantinuum_native(
        ParameterizedCircuit(
            1,
            [ParameterizedGate("h", (0,), angle_offset=1e-12)],
        )
    )

    materialization = materialize_ballarin_circuit(
        compilation,
        np.array([], dtype=np.float64),
    )

    assert materialization.circuit.gates[0].name == "h"
    assert materialization.circuit.gates[0].angle_offset == pytest.approx(1e-12)


def test_materialization_snapshots_all_gates_before_data_callbacks() -> None:
    """An early callback must not mutate a later gate after validation."""
    compilation: NativeCompilation | None = None

    def current_float(value: object) -> float:
        return cast("float", vars(value)["current_value"])

    mutable_float_type = type("_MutableFloat", (float,), {"__float__": current_float})
    source_angle = mutable_float_type(0.3)
    _assign_attribute(source_angle, "current_value", 0.3)

    def mutate_later_gate(_: NDArray[np.float64]) -> float:
        assert compilation is not None
        compilation.circuit.gates[1].name = "x"
        _assign_attribute(source_angle, "current_value", 0.9)
        return 0.2

    logical = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rx", (0,), data_map=mutate_later_gate),
            ParameterizedGate("rxx", (0, 1), angle_offset=cast("float", source_angle)),
        ],
    )
    compilation = compile_quantinuum_native(logical)

    materialization = materialize_ballarin_circuit(
        compilation,
        np.array([], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
    )

    assert compilation.circuit.gates[1].name == "x"
    assert [gate.name for gate in materialization.circuit.gates] == [
        "rx",
        "h",
        "h",
        "rzz",
        "h",
        "h",
    ]
    assert materialization.mapping[1].resolved_native_angle == pytest.approx(0.3)


@pytest.mark.parametrize(
    "invalid_theta",
    [
        np.array([True]),
        np.array(["0.2"]),
        np.array([0.2 + 4.0j]),
    ],
    ids=("boolean", "numeric-string", "complex"),
)
def test_materialization_rejects_non_real_parameter_vector_values(
    invalid_theta: NDArray[np.generic],
) -> None:
    """Binding must not silently coerce Boolean, textual, or complex values."""
    logical = ParameterizedCircuit(
        1,
        [ParameterizedGate("rx", (0,), param_index=0)],
        num_params=1,
    )

    with pytest.raises(TypeError, match="real non-Boolean"):
        materialize_ballarin_circuit(compile_quantinuum_native(logical), cast("NDArray[np.float64]", invalid_theta))


def test_materialization_rejects_mutated_compilation_semantic_arity() -> None:
    """A mutable WP6 circuit must be revalidated before it becomes authoritative."""
    compilation = compile_quantinuum_native(
        ParameterizedCircuit(1, [ParameterizedGate("h", (0,))]),
    )
    compilation.circuit.gates[0].name = "cx"

    with pytest.raises(ValueError, match="semantic arity"):
        materialize_ballarin_circuit(compilation, np.array([], dtype=np.float64))


def test_materialization_rejects_mutated_compilation_basis_block() -> None:
    """Pruning must not trust basis-change provenance after gate mutation."""
    compilation = compile_quantinuum_native(
        ParameterizedCircuit(2, [ParameterizedGate("rxx", (0, 1), angle_offset=0.2)]),
    )
    compilation.circuit.gates[0].name = "x"

    with pytest.raises(ValueError, match="malformed basis changes"):
        materialize_ballarin_circuit(compilation, np.array([], dtype=np.float64))


def test_materialization_rejects_mutated_passthrough_and_angle_provenance() -> None:
    """Source names and unbound native-angle metadata must remain traceable."""
    passthrough = compile_quantinuum_native(
        ParameterizedCircuit(1, [ParameterizedGate("h", (0,))]),
    )
    passthrough.circuit.gates[0].name = "x"
    with pytest.raises(ValueError, match="no longer matches its source"):
        materialize_ballarin_circuit(passthrough, np.array([], dtype=np.float64))

    rotation = compile_quantinuum_native(
        ParameterizedCircuit(
            2,
            [ParameterizedGate("rzz", (0, 1), param_index=0, angle_offset=0.2)],
            num_params=1,
        ),
    )
    rotation.circuit.gates[0].angle_offset = 0.4
    with pytest.raises(ValueError, match="mutated native-angle metadata"):
        materialize_ballarin_circuit(rotation, np.array([0.1], dtype=np.float64))


def test_pruning_remap_preserves_sparse_native_ids_and_logical_provenance() -> None:
    """Final positions should compact while retained stable native IDs keep their origin."""
    logical = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("h", (0,), logical_gate_id="logical-h"),
            ParameterizedGate("rxx", (0, 1), angle_offset=0.0, logical_gate_id="pruned-x"),
            ParameterizedGate("rzz", (1, 0), angle_offset=0.3, logical_gate_id="retained-z"),
            ParameterizedGate("ry", (1,), angle_offset=-0.2, logical_gate_id="logical-y"),
        ],
    )

    materialization = materialize_ballarin_circuit(
        compile_quantinuum_native(logical),
        np.array([], dtype=np.float64),
    )

    assert tuple(gate.native_gate_id for gate in materialization.circuit.gates) == (0, 6, 7)
    assert tuple(gate.logical_gate_id for gate in materialization.circuit.gates) == (
        "logical-h",
        "retained-z",
        "logical-y",
    )
    assert materialization.pre_pruning_to_final_indices == (0, None, None, None, None, None, 1, 2)
    assert materialization.pruned_native_rotation_ids == (3,)
    assert materialization.omitted_basis_change_native_gate_ids == (1, 2, 4, 5)

    pruned = materialization.mapping[1]
    retained = materialization.mapping[2]
    assert pruned.pre_pruning_native_gate_ids == (1, 2, 3, 4, 5)
    assert pruned.retained_native_gate_ids == ()
    assert pruned.final_native_gate_indices == ()
    assert pruned.native_rotation_gate_id == 3
    assert pruned.final_native_rotation_gate_index is None
    assert retained.pre_pruning_native_gate_ids == (6,)
    assert retained.retained_native_gate_ids == (6,)
    assert retained.final_native_gate_indices == (1,)
    assert retained.native_rotation_gate_id == 6
    assert retained.final_native_rotation_gate_index == 1


def test_materialization_resolves_data_map_once_without_mutating_any_source() -> None:
    """Each native angle should be resolved once and all caller-owned inputs preserved."""
    calls = 0

    def data_map(x: NDArray[np.float64]) -> float:
        nonlocal calls
        calls += 1
        return float(x[0])

    source_gate = ParameterizedGate(
        "rxx",
        (1, 0),
        param_index=0,
        angle_scale=-0.5,
        angle_offset=0.2,
        data_map=data_map,
        logical_gate_id="mapped",
        native_gate_id="old-native",
        noise_enabled=False,
    )
    logical = ParameterizedCircuit(2, [source_gate], num_params=2)
    source_snapshot = tuple(_gate_snapshot(gate) for gate in logical.gates)
    compilation = compile_quantinuum_native(logical)
    compiled_snapshot = tuple(_gate_snapshot(gate) for gate in compilation.circuit.gates)
    theta = np.array([0.4, 9.0], dtype=np.float64)
    sample = np.array([0.3], dtype=np.float64)
    theta_snapshot = theta.copy()
    sample_snapshot = sample.copy()

    assert calls == 0
    materialization = materialize_ballarin_circuit(compilation, theta, sample)
    mapping = materialization.mapping[0]

    assert calls == 1
    assert mapping.resolved_native_angle == pytest.approx(0.3)
    assert mapping.canonical_rzz_angle == pytest.approx(0.3)
    assert mapping.canonical_rzz_magnitude == pytest.approx(0.3)
    assert mapping.rotation_pruned is False
    rotation = materialization.circuit.gates[cast("int", mapping.final_native_rotation_gate_index)]
    assert rotation.angle_offset == pytest.approx(0.3)
    assert rotation.param_index is None
    assert rotation.data_map is None
    assert rotation.noise_enabled is True
    assert tuple(_gate_snapshot(gate) for gate in logical.gates) == source_snapshot
    assert tuple(_gate_snapshot(gate) for gate in compilation.circuit.gates) == compiled_snapshot
    np.testing.assert_array_equal(theta, theta_snapshot)
    np.testing.assert_array_equal(sample, sample_snapshot)
