# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Quantinuum-native state-preparation circuit compilation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
from qiskit.quantum_info import Operator

from benchmarks.state_preparation.circuits import (
    LogicalToNativeMapping,
    NativeAngleExpression,
    NativeCompilation,
    compile_quantinuum_native,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    GateNoiseContext,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
)
from mqt.yaqs.optimization.krotov import forward_states, forward_tjm_trajectory

if TYPE_CHECKING:
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
_QISKIT_ROTATION = {
    "rxx": RXXGate,
    "ryy": RYYGate,
    "rzz": RZZGate,
}
_ANGLE_CASES = (
    0.0,
    np.nextafter(0.0, 1.0),
    np.nextafter(0.0, -1.0),
    np.pi / 2,
    -np.pi / 2,
    np.nextafter(np.pi, 0.0),
    np.nextafter(-np.pi, 0.0),
    np.pi,
    -np.pi,
    3 * np.pi,
    -3 * np.pi,
    0.713,
    -1.291,
)


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
    """Embed a matrix given in ascending two-site YAQS ordering.

    Returns:
        The full-register matrix.
    """
    dimension = 2**num_qubits
    embedded = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        bits = [(column >> qubit) & 1 for qubit in range(num_qubits)]
        local_column = 2 * bits[first_site] + bits[second_site]
        for local_row in range(4):
            amplitude = matrix[local_row, local_column]
            output_bits = bits.copy()
            output_bits[first_site] = (local_row >> 1) & 1
            output_bits[second_site] = local_row & 1
            row = sum(bit << qubit for qubit, bit in enumerate(output_bits))
            embedded[row, column] += amplitude
    return embedded


def _embed_local_matrix(
    matrix: NDArray[np.complex128],
    sites: tuple[int, ...],
    num_qubits: int,
) -> NDArray[np.complex128]:
    """Embed a one- or two-site matrix into the full register.

    Returns:
        The full-register matrix.
    """
    if len(sites) == 1:
        return _embed_one_site(matrix, sites[0], num_qubits)
    return _embed_two_site(matrix, sites[0], sites[1], num_qubits)


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
        unitary = _embed_local_matrix(matrix, sites, circuit.num_qubits) @ unitary
    return unitary


def _dense_unitary_and_derivative(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    parameter_index: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return a dense unitary and its analytic derivative for one parameter."""
    dimension = 2**circuit.num_qubits
    unitary = np.eye(dimension, dtype=np.complex128)
    derivative = np.zeros((dimension, dimension), dtype=np.complex128)
    for gate in circuit.gates:
        local_matrix, sites = circuit.gate_matrix(gate, theta)
        matrix = _embed_local_matrix(local_matrix, sites, circuit.num_qubits)
        local_derivative = np.zeros_like(local_matrix)
        if gate.param_index == parameter_index:
            derivative_operator, derivative_sites = circuit.derivative_operator(gate)
            assert derivative_sites == sites
            local_derivative = gate.angle_scale * derivative_operator @ local_matrix
        embedded_derivative = _embed_local_matrix(local_derivative, sites, circuit.num_qubits)
        derivative = embedded_derivative @ unitary + matrix @ derivative
        unitary = matrix @ unitary
    return unitary, derivative


def _rotation_oracle(name: str, angle: float) -> NDArray[np.complex128]:
    """Return an independent dense Pauli-product rotation."""
    pauli = _PAULI_BY_ROTATION[name]
    generator = np.kron(pauli, pauli)
    return np.asarray(
        np.cos(angle / 2) * np.eye(4, dtype=np.complex128) - 1.0j * np.sin(angle / 2) * generator,
        dtype=np.complex128,
    )


def _assert_unitaries_equal_up_to_global_phase(
    actual: NDArray[np.complex128],
    expected: NDArray[np.complex128],
    *,
    atol: float = 1e-10,
) -> complex:
    """Assert unitary equality up to global phase.

    Returns:
        The phase multiplying ``expected`` to produce ``actual``.
    """
    overlap = np.trace(expected.conj().T @ actual)
    assert abs(overlap) > 1e-12
    phase = complex(overlap / abs(overlap))
    np.testing.assert_allclose(actual, phase * expected, atol=atol, rtol=atol)
    return phase


def _gate_snapshot(gate: ParameterizedGate) -> tuple[object, ...]:
    """Return all caller-owned gate fields as an immutable snapshot."""
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


@pytest.mark.parametrize("gate_name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("sites", [(0, 1), (1, 0)])
@pytest.mark.parametrize("angle", _ANGLE_CASES)
def test_compiled_rotations_match_explicit_and_qiskit_conventions(
    gate_name: str,
    sites: tuple[int, int],
    angle: float,
) -> None:
    """Every signed and boundary rotation should compile without pruning."""
    logical = ParameterizedCircuit(
        2,
        [ParameterizedGate(gate_name, sites, angle_offset=angle)],
    )

    compilation = compile_quantinuum_native(logical)
    actual = _dense_unitary(compilation.circuit, np.array([], dtype=np.float64))
    expected = _rotation_oracle(gate_name, angle)
    qiskit_expected = np.asarray(Operator(_QISKIT_ROTATION[gate_name](angle)).data, dtype=np.complex128)

    assert isinstance(compilation, NativeCompilation)
    assert sum(gate.name == "rzz" for gate in compilation.circuit.gates) == 1
    assert all(gate.name not in {"rxx", "ryy"} for gate in compilation.circuit.gates)
    np.testing.assert_allclose(expected, qiskit_expected, atol=1e-12, rtol=1e-12)
    _assert_unitaries_equal_up_to_global_phase(actual, expected)

    mapping = compilation.mapping[0]
    assert mapping.source_sites == sites
    assert mapping.source_gate_name == gate_name
    assert mapping.native_rotation_gate_index is not None
    native_rotation = compilation.circuit.gates[mapping.native_rotation_gate_index]
    assert native_rotation.name == "rzz"
    assert native_rotation.angle_offset == angle


@pytest.mark.parametrize(
    ("gate_name", "relationship", "expected_names"),
    [
        ("rxx", "rxx_h", ("h", "h", "rzz", "h", "h")),
        ("ryy", "ryy_rx_pi_over_2", ("rx", "rx", "rzz", "rx", "rx")),
        ("rzz", "none", ("rzz",)),
    ],
)
@pytest.mark.parametrize("sites", [(0, 1), (1, 0)])
def test_compiler_emits_exact_native_blocks_and_noise_flags(
    gate_name: str,
    relationship: str,
    expected_names: tuple[str, ...],
    sites: tuple[int, int],
) -> None:
    """The native block structure and basis-change metadata should be exact."""
    source = ParameterizedGate(
        gate_name,
        sites,
        param_index=2,
        angle_scale=-1.25,
        angle_offset=0.17,
        logical_gate_id="logical",
        native_gate_id="stale-native",
        noise_enabled=True,
    )
    logical = ParameterizedCircuit(2, [source], num_params=4)

    compilation = compile_quantinuum_native(logical)
    native = compilation.circuit
    mapping = compilation.mapping[0]

    assert tuple(gate.name for gate in native.gates) == expected_names
    assert mapping.source_logical_gate_index == 0
    assert mapping.logical_gate_id == "logical"
    assert mapping.source_parameter_index == 2
    assert mapping.native_gate_indices == tuple(range(len(expected_names)))
    assert mapping.native_rotation_gate_index == expected_names.index("rzz")
    assert mapping.basis_change_relationship == relationship
    assert all(gate.logical_gate_id == "logical" for gate in native.gates)
    assert [gate.native_gate_id for gate in native.gates] == list(range(len(native.gates)))

    rotation_index = cast("int", mapping.native_rotation_gate_index)
    rotation = native.gates[rotation_index]
    assert rotation.name == "rzz"
    assert rotation.sites == sites
    assert rotation.param_index == source.param_index
    assert rotation.angle_scale == source.angle_scale
    assert rotation.angle_offset == source.angle_offset
    assert rotation.data_map is source.data_map
    assert rotation.noise_enabled is True

    basis_indices = mapping.basis_change_before_indices + mapping.basis_change_after_indices
    assert all(native.gates[index].param_index is None for index in basis_indices)
    assert all(native.gates[index].noise_enabled is False for index in basis_indices)
    if gate_name == "rxx":
        assert mapping.basis_change_before_indices == (0, 1)
        assert mapping.basis_change_after_indices == (3, 4)
        assert [native.gates[index].sites for index in range(5)] == [
            (sites[0],),
            (sites[1],),
            sites,
            (sites[1],),
            (sites[0],),
        ]
    elif gate_name == "ryy":
        assert mapping.basis_change_before_indices == (0, 1)
        assert mapping.basis_change_after_indices == (3, 4)
        assert [native.gates[index].angle_offset for index in (0, 1, 3, 4)] == [
            np.pi / 2,
            np.pi / 2,
            -np.pi / 2,
            -np.pi / 2,
        ]
        assert [native.gates[index].sites for index in range(5)] == [
            (sites[0],),
            (sites[1],),
            sites,
            (sites[1],),
            (sites[0],),
        ]
    else:
        assert mapping.basis_change_before_indices == ()
        assert mapping.basis_change_after_indices == ()


def test_mapping_preserves_affine_metadata_and_source_isolation() -> None:
    """Compilation should retain complete provenance without evaluating or aliasing the source."""
    data_map_calls = 0

    def data_map(x: NDArray[np.float64]) -> float:
        nonlocal data_map_calls
        data_map_calls += 1
        return float(x[0])

    gates = [
        ParameterizedGate(
            "h",
            (0,),
            logical_gate_id="logical-h",
            native_gate_id="old-h",
            noise_enabled=False,
        ),
        ParameterizedGate(
            "rxx",
            (2, 0),
            param_index=3,
            angle_scale=-1.75,
            angle_offset=0.31,
            data_map=data_map,
        ),
        ParameterizedGate(
            "ryy",
            (0, 1),
            angle_offset=-0.4,
            logical_gate_id=19,
            noise_enabled=False,
        ),
        ParameterizedGate(
            "rzz",
            (2, 1),
            param_index=1,
            angle_scale=0.75,
            angle_offset=-0.2,
            logical_gate_id="logical-z",
        ),
    ]
    logical = ParameterizedCircuit(3, gates, num_params=6)
    source_snapshots = tuple(_gate_snapshot(gate) for gate in logical.gates)

    compilation = compile_quantinuum_native(logical)

    assert data_map_calls == 0
    assert compilation.circuit.num_params == 6
    assert tuple(_gate_snapshot(gate) for gate in logical.gates) == source_snapshots
    assert tuple(mapping.source_logical_gate_index for mapping in compilation.mapping) == (0, 1, 2, 3)
    assert tuple(mapping.logical_gate_id for mapping in compilation.mapping) == ("logical-h", 1, 19, "logical-z")
    assert tuple(mapping.source_gate_name for mapping in compilation.mapping) == ("h", "rxx", "ryy", "rzz")
    assert tuple(mapping.source_sites for mapping in compilation.mapping) == ((0,), (2, 0), (0, 1), (2, 1))
    assert tuple(mapping.source_parameter_index for mapping in compilation.mapping) == (None, 3, None, 1)
    assert tuple(index for mapping in compilation.mapping for index in mapping.native_gate_indices) == tuple(
        range(len(compilation.circuit.gates))
    )
    assert all(
        compilation.circuit.gates[index].native_gate_id == index for index in range(len(compilation.circuit.gates))
    )
    assert compilation.circuit.gates[0] is not logical.gates[0]

    rxx_mapping = compilation.mapping[1]
    expression = rxx_mapping.native_angle_expression
    assert isinstance(expression, NativeAngleExpression)
    assert expression.param_index == 3
    assert expression.angle_scale == pytest.approx(-1.75)
    assert expression.angle_offset == pytest.approx(0.31)
    assert expression.data_map is data_map
    theta = np.array([0.0, 0.0, 0.0, 0.4, 0.0, 0.0], dtype=np.float64)
    x = np.array([0.23], dtype=np.float64)
    assert expression.evaluate(theta, x) == pytest.approx(-1.75 * 0.4 + 0.31 + 0.23)
    assert data_map_calls == 1

    for mapping in compilation.mapping:
        assert isinstance(mapping, LogicalToNativeMapping)
    with pytest.raises(FrozenInstanceError):
        cast("Any", rxx_mapping).source_gate_name = "changed"

    compilation.circuit.gates[0].noise_enabled = True
    compilation.circuit.gates[cast("int", rxx_mapping.native_rotation_gate_index)].angle_scale = 99.0
    assert logical.gates[0].noise_enabled is False
    assert logical.gates[1].angle_scale == pytest.approx(-1.75)
    assert tuple(_gate_snapshot(gate) for gate in logical.gates) == source_snapshots


@pytest.mark.parametrize("seed", [17, 91])
def test_whole_compiled_circuit_matches_logical_unitary(seed: int) -> None:
    """An interleaved long-range circuit should compile up to global phase."""
    logical = ParameterizedCircuit(
        3,
        [
            ParameterizedGate("h", (0,), logical_gate_id="h"),
            ParameterizedGate("rxx", (2, 0), param_index=0, angle_scale=-1.3, angle_offset=0.2),
            ParameterizedGate("ry", (1,), param_index=1, angle_scale=0.7, angle_offset=-0.1),
            ParameterizedGate("ryy", (0, 2), param_index=2, angle_scale=0.6, angle_offset=-0.4),
            ParameterizedGate("rz", (2,), angle_offset=0.33),
            ParameterizedGate("rzz", (2, 1), param_index=3, angle_scale=-0.5, angle_offset=0.7),
        ],
        num_params=5,
    )
    theta = np.random.default_rng(seed).normal(scale=0.8, size=logical.num_params)

    compilation = compile_quantinuum_native(logical)
    logical_unitary = _dense_unitary(logical, theta)
    native_unitary = _dense_unitary(compilation.circuit, theta)

    _assert_unitaries_equal_up_to_global_phase(native_unitary, logical_unitary, atol=2e-10)
    logical_state = forward_states(logical, theta, np.array([]), MPS(3), KrotovTruncation())[-1].to_vec()
    native_state = forward_states(
        compilation.circuit,
        theta,
        np.array([]),
        MPS(3),
        KrotovTruncation(),
    )[-1].to_vec()
    assert abs(np.vdot(logical_state, native_state)) == pytest.approx(1.0, abs=2e-10)


@pytest.mark.parametrize("gate_name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("sites", [(0, 1), (1, 0)])
@pytest.mark.parametrize("angle_scale", [-1.7, 0.65])
def test_compiled_analytic_derivative_matches_logical_and_finite_difference(
    gate_name: str,
    sites: tuple[int, int],
    angle_scale: float,
) -> None:
    """The preserved affine rotation should retain its exact signed derivative."""
    parameter_index = 2
    gate = ParameterizedGate(
        gate_name,
        sites,
        param_index=parameter_index,
        angle_scale=angle_scale,
        angle_offset=0.23,
    )
    logical = ParameterizedCircuit(2, [gate], num_params=4)
    compilation = compile_quantinuum_native(logical)
    theta = np.array([0.0, 0.0, 0.37, 0.0], dtype=np.float64)

    logical_unitary, logical_derivative = _dense_unitary_and_derivative(logical, theta, parameter_index)
    native_unitary, native_derivative = _dense_unitary_and_derivative(
        compilation.circuit,
        theta,
        parameter_index,
    )
    phase = _assert_unitaries_equal_up_to_global_phase(native_unitary, logical_unitary)

    angle = angle_scale * theta[parameter_index] + gate.angle_offset
    pauli = _PAULI_BY_ROTATION[gate_name]
    expected_logical_derivative = (
        angle_scale
        * (-0.5j * np.kron(pauli, pauli))
        @ _rotation_oracle(
            gate_name,
            angle,
        )
    )
    np.testing.assert_allclose(logical_derivative, expected_logical_derivative, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(native_derivative, phase * expected_logical_derivative, atol=2e-10, rtol=2e-10)

    epsilon = 1e-7
    plus = theta.copy()
    minus = theta.copy()
    plus[parameter_index] += epsilon
    minus[parameter_index] -= epsilon
    finite_difference = (_dense_unitary(compilation.circuit, plus) - _dense_unitary(compilation.circuit, minus)) / (
        2 * epsilon
    )
    np.testing.assert_allclose(native_derivative, finite_difference, atol=2e-8, rtol=2e-8)


def test_shared_bmpd_ansatz_compiles_equivalently_with_exact_native_rotation_count() -> None:
    """The shared logical BMPD ansatz should retain its parameters, signs, and state."""
    logical = create_brickwall_matrix_product_disentangler_parameterized_circuit(4, 1)
    compilation = compile_quantinuum_native(logical)
    theta = np.random.default_rng(123).normal(scale=0.2, size=logical.num_params)
    logical_entangler_indices = [
        index for index, gate in enumerate(logical.gates) if gate.name in {"rxx", "ryy", "rzz"}
    ]

    assert logical.num_params == compilation.circuit.num_params == 27
    assert len(logical_entangler_indices) == 9
    assert sum(gate.name == "rzz" for gate in compilation.circuit.gates) == len(logical_entangler_indices)
    assert all(gate.name not in {"rxx", "ryy"} for gate in compilation.circuit.gates)

    for logical_index in logical_entangler_indices:
        source = logical.gates[logical_index]
        mapping = compilation.mapping[logical_index]
        assert mapping.source_logical_gate_index == logical_index
        assert mapping.native_rotation_gate_index is not None
        native = compilation.circuit.gates[mapping.native_rotation_gate_index]
        assert native.name == "rzz"
        assert native.param_index == source.param_index
        assert native.angle_scale == pytest.approx(source.angle_scale)
        assert source.angle_scale == pytest.approx(-1.0)
        assert native.angle_offset == source.angle_offset
        assert all(
            compilation.circuit.gates[index].noise_enabled is False
            for index in mapping.basis_change_before_indices + mapping.basis_change_after_indices
        )

    logical_state = forward_states(logical, theta, np.array([]), MPS(4), KrotovTruncation())[-1].to_vec()
    native_state = forward_states(
        compilation.circuit,
        theta,
        np.array([]),
        MPS(4),
        KrotovTruncation(),
    )[-1].to_vec()
    assert abs(np.vdot(logical_state, native_state)) == pytest.approx(1.0, abs=5e-10)


def test_compilation_basis_changes_are_excluded_before_provider_invocation() -> None:
    """Only central enabled RZZ gates should produce gate-local provider contexts."""
    logical = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), angle_offset=0.31),
            ParameterizedGate(
                "rxx",
                (1, 0),
                param_index=0,
                angle_scale=-1.0,
                angle_offset=0.1,
                logical_gate_id="logical-x",
            ),
            ParameterizedGate(
                "ryy",
                (0, 1),
                param_index=1,
                angle_scale=0.5,
                angle_offset=-0.2,
            ),
            ParameterizedGate(
                "rzz",
                (1, 0),
                param_index=2,
                angle_scale=1.25,
                angle_offset=0.05,
                logical_gate_id="logical-z",
            ),
        ],
        num_params=3,
    )
    compilation = compile_quantinuum_native(logical)
    contexts: list[GateNoiseContext] = []

    def provider(context: GateNoiseContext, rng: np.random.Generator) -> None:
        del rng
        contexts.append(context)

    theta = np.array([0.2, -0.4, 0.3], dtype=np.float64)
    forward_tjm_trajectory(
        compilation.circuit,
        theta,
        np.array([], dtype=np.float64),
        MPS(2),
        KrotovTruncation(),
        None,
        KrotovTJMOptions(apply_noise_to="all"),
        np.random.default_rng(4),
        noise_provider=provider,
    )

    assert compilation.circuit.gates[0].noise_enabled is False
    assert [context.gate_index for context in contexts] == [3, 8, 11]
    assert [context.gate_name for context in contexts] == ["rzz", "rzz", "rzz"]
    assert [context.sites for context in contexts] == [(0, 1), (0, 1), (0, 1)]
    assert [context.parameter_index for context in contexts] == [0, 1, 2]
    assert [context.logical_gate_id for context in contexts] == ["logical-x", 2, "logical-z"]
    assert [context.native_gate_id for context in contexts] == [3, 8, 11]
    assert [context.resolved_angle for context in contexts] == pytest.approx([-0.1, -0.4, 0.425])


@pytest.mark.parametrize(
    "unsupported_gate",
    [
        ParameterizedGate("cx", (0, 1)),
        ParameterizedGate("cz", (0, 1)),
        ParameterizedGate("swap", (0, 1)),
        ParameterizedGate("cp", (0, 1), param_index=0),
    ],
    ids=lambda gate: gate.name,
)
def test_unsupported_two_qubit_gates_fail_clearly_and_atomically(
    unsupported_gate: ParameterizedGate,
) -> None:
    """Unsupported two-qubit gates should name the gate and logical index."""
    logical = ParameterizedCircuit(
        2,
        [ParameterizedGate("h", (0,)), unsupported_gate],
        num_params=1 if unsupported_gate.param_index is not None else 0,
    )
    snapshots = tuple(_gate_snapshot(gate) for gate in logical.gates)

    with pytest.raises(ValueError, match=unsupported_gate.name) as error:
        compile_quantinuum_native(logical)

    message = str(error.value)
    assert unsupported_gate.name in message
    assert "1" in message
    assert tuple(_gate_snapshot(gate) for gate in logical.gates) == snapshots


@pytest.mark.parametrize(
    "mismatched_gate",
    [
        ParameterizedGate("rxx", (0,), angle_offset=0.2),
        ParameterizedGate("ryy", (0,), angle_offset=-0.3),
        ParameterizedGate("cx", (0,)),
        ParameterizedGate("h", (0, 1)),
        ParameterizedGate(
            "unitary",
            (0,),
            fixed_params=cast("Any", (np.eye(4, dtype=np.complex128),)),
        ),
    ],
    ids=lambda gate: gate.name,
)
def test_semantic_gate_arity_must_match_declared_sites(mismatched_gate: ParameterizedGate) -> None:
    """Malformed arity metadata must not bypass native compilation checks."""
    logical = ParameterizedCircuit(2, [mismatched_gate])

    with pytest.raises(ValueError, match=rf"{mismatched_gate.name}.*semantic arity"):
        compile_quantinuum_native(logical)


@pytest.mark.parametrize("invalid_index", [-1, True, 0.5])
def test_compiler_rejects_invalid_source_parameter_indices(invalid_index: object) -> None:
    """Malformed indices must not retain Python negative-index semantics."""
    logical = ParameterizedCircuit(
        1,
        [ParameterizedGate("rx", (0,), param_index=cast("Any", invalid_index))],
        num_params=2,
    )

    expected_error = ValueError if invalid_index == -1 else TypeError
    with pytest.raises(expected_error, match="param_index"):
        compile_quantinuum_native(logical)


def test_empty_and_one_qubit_only_circuits_are_fresh_passthroughs() -> None:
    """Trivial supported circuits should preserve dimensions and metadata."""
    empty = ParameterizedCircuit(2, [], num_params=3)
    empty_compilation = compile_quantinuum_native(empty)
    assert empty_compilation.circuit.num_qubits == 2
    assert empty_compilation.circuit.num_params == 3
    assert empty_compilation.circuit.gates == []
    assert empty_compilation.mapping == ()

    source = ParameterizedGate(
        "ry",
        (1,),
        param_index=2,
        angle_scale=-0.4,
        angle_offset=0.3,
        logical_gate_id="one-qubit",
        native_gate_id="old",
        noise_enabled=True,
    )
    logical = ParameterizedCircuit(2, [source], num_params=4)
    compilation = compile_quantinuum_native(logical)
    native = compilation.circuit.gates[0]

    assert native is not source
    assert native.name == source.name
    assert native.sites == source.sites
    assert native.param_index == source.param_index
    assert native.angle_scale == source.angle_scale
    assert native.angle_offset == source.angle_offset
    assert native.data_map is source.data_map
    assert native.fixed_params == source.fixed_params
    assert native.logical_gate_id == source.logical_gate_id
    assert native.native_gate_id == 0
    assert native.noise_enabled is False
    assert source.noise_enabled is True
    assert compilation.mapping[0].native_gate_indices == (0,)
    assert compilation.mapping[0].basis_change_relationship == "none"
