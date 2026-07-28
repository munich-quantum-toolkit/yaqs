# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for state-preparation circuit statistics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from qiskit.circuit import Gate, QuantumCircuit

from benchmarks.state_preparation import (
    AnsatzConfig,
    CircuitStatistics,
    EvaluatedRepresentation,
    NativeCircuitSource,
    NativeCompilation,
    collect_circuit_statistics,
    compile_quantinuum_native,
    materialize_ballarin_circuit,
)
from mqt.yaqs.optimization import (
    ParameterizedCircuit,
    ParameterizedGate,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
)


def _bmpd_statistics(
    num_qubits: int,
    depth: int,
    *,
    initial_single_qubit_layer: bool,
) -> tuple[ParameterizedCircuit, CircuitStatistics]:
    """Build the shared ansatz and collect its standard-row statistics.

    Returns:
        The logical circuit and its collected statistics.
    """
    circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(
        num_qubits,
        depth,
        initial_single_qubit_layer=initial_single_qubit_layer,
    )
    ansatz = AnsatzConfig(
        configured_bmpd_depth=depth,
        initial_single_qubit_layer=initial_single_qubit_layer,
    )
    return circuit, collect_circuit_statistics(circuit, ansatz)


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("initial_single_qubit_layer", [False, True])
def test_bmpd_closed_form_gate_and_parameter_counts(
    num_qubits: int,
    depth: int,
    initial_single_qubit_layer: object,
) -> None:
    """Even, odd, zero-depth, and product-layer counts follow the BMPD form."""
    assert isinstance(initial_single_qubit_layer, bool)
    _, statistics = _bmpd_statistics(
        num_qubits,
        depth,
        initial_single_qubit_layer=initial_single_qubit_layer,
    )
    num_blocks = depth * (num_qubits - 1)
    initial_gate_count = 3 * num_qubits if initial_single_qubit_layer else 0

    assert statistics.configured_bmpd_depth == depth
    assert statistics.num_layers == 2 * depth
    assert statistics.num_parameters == 9 * num_blocks + initial_gate_count
    assert statistics.logical_num_1q_gates == 6 * num_blocks + initial_gate_count
    assert statistics.logical_num_2q_gates == 3 * num_blocks
    assert statistics.logical_gate_counts == {
        **(
            {"ry": num_blocks * 2 + num_qubits, "rz": num_blocks * 4 + 2 * num_qubits}
            if initial_single_qubit_layer
            else ({"ry": num_blocks * 2, "rz": num_blocks * 4} if num_blocks else {})
        ),
        **({"rxx": num_blocks, "ryy": num_blocks, "rzz": num_blocks} if num_blocks else {}),
    }
    assert statistics.evaluated_representation == "logical"
    assert statistics.num_1q_gates == statistics.logical_num_1q_gates
    assert statistics.num_2q_gates == statistics.logical_num_2q_gates
    assert statistics.evaluated_depth == statistics.logical_depth


@pytest.mark.parametrize(
    ("num_qubits", "depth", "initial_single_qubit_layer"),
    [(2, 2, False), (3, 1, True), (4, 2, True), (5, 1, False)],
)
def test_dependency_depth_matches_independent_qiskit_calculation(
    num_qubits: int,
    depth: int,
    initial_single_qubit_layer: object,
) -> None:
    """Logical and native dependency depths agree with Qiskit's scheduler."""
    assert isinstance(initial_single_qubit_layer, bool)
    logical, statistics = _bmpd_statistics(
        num_qubits,
        depth,
        initial_single_qubit_layer=initial_single_qubit_layer,
    )
    native = compile_quantinuum_native(logical).circuit

    def qiskit_depth(circuit: ParameterizedCircuit) -> int:
        qiskit_circuit = QuantumCircuit(circuit.num_qubits)
        for gate in circuit.gates:
            qiskit_circuit.append(
                Gate(gate.name, len(gate.sites), []),
                [qiskit_circuit.qubits[site] for site in gate.sites],
            )
        return qiskit_circuit.depth()

    assert statistics.logical_depth == qiskit_depth(logical)
    assert statistics.native_depth == qiskit_depth(native)


def test_native_compilation_counts_every_entangler_as_one_rzz() -> None:
    """Native metadata includes compilation basis gates and exact RZZ totals."""
    _, statistics = _bmpd_statistics(4, 1, initial_single_qubit_layer=True)

    assert statistics.logical_num_1q_gates == 30
    assert statistics.logical_num_2q_gates == 9
    assert statistics.native_num_1q_gates == 54
    assert statistics.native_num_2q_gates == 9
    assert statistics.native_rzz_count == 9
    assert statistics.pruned_native_rzz_count == 0
    assert statistics.native_gate_counts == {
        "h": 12,
        "rx": 12,
        "ry": 10,
        "rz": 20,
        "rzz": 9,
    }


def test_ballarin_counts_use_final_pruning_and_safe_cancellation() -> None:
    """Ballarin rows count the frozen circuit after both final simplifications."""
    logical = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rxx", (0, 1), param_index=0),
            ParameterizedGate("rxx", (0, 1), param_index=1),
            ParameterizedGate("ryy", (0, 1), param_index=2),
        ],
        num_params=3,
    )
    compilation = compile_quantinuum_native(logical)
    materialization = materialize_ballarin_circuit(
        compilation,
        np.array([0.2, 0.3, 1e-4], dtype=np.float64),
    )

    statistics = collect_circuit_statistics(
        logical,
        AnsatzConfig(0, initial_single_qubit_layer=False),
        native_source=materialization,
        evaluated_representation="native",
    )

    assert materialization.cancelled_basis_change_count == 4
    assert statistics.logical_gate_counts == {"rxx": 2, "ryy": 1}
    assert statistics.logical_num_2q_gates == 3
    assert statistics.native_gate_counts == {"h": 4, "rzz": 2}
    assert statistics.native_num_1q_gates == 4
    assert statistics.native_num_2q_gates == 2
    assert statistics.native_rzz_count == 2
    assert statistics.pruned_native_rzz_count == 1
    assert statistics.num_1q_gates == 4
    assert statistics.num_2q_gates == 2
    assert statistics.evaluated_depth == statistics.native_depth


def test_shared_parameters_are_not_counted_once_per_gate() -> None:
    """Trainable parameter count comes from the vector, not gate occurrences."""
    logical = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("rx", (0,), param_index=0),
            ParameterizedGate("ry", (1,), param_index=0),
            ParameterizedGate("rzz", (0, 1), param_index=0),
        ],
        num_params=1,
    )

    statistics = collect_circuit_statistics(
        logical,
        AnsatzConfig(0, initial_single_qubit_layer=False),
    )

    assert statistics.num_parameters == 1
    assert statistics.logical_num_1q_gates == 2
    assert statistics.logical_num_2q_gates == 1


def test_standard_and_ballarin_row_selection_changes_only_reporting_aliases() -> None:
    """Both row types preserve identical extended logical/native metadata."""
    logical = ParameterizedCircuit(
        2,
        [ParameterizedGate("rxx", (0, 1), param_index=0)],
        num_params=1,
    )
    source = compile_quantinuum_native(logical)
    ansatz = AnsatzConfig(0, initial_single_qubit_layer=False)

    standard = collect_circuit_statistics(logical, ansatz, native_source=source)
    ballarin = collect_circuit_statistics(
        logical,
        ansatz,
        native_source=source,
        evaluated_representation="native",
    )

    for field in (
        "logical_depth",
        "logical_num_1q_gates",
        "logical_num_2q_gates",
        "native_depth",
        "native_num_1q_gates",
        "native_num_2q_gates",
        "native_rzz_count",
        "pruned_native_rzz_count",
        "logical_gate_counts",
        "native_gate_counts",
    ):
        assert getattr(standard, field) == getattr(ballarin, field)
    assert standard.num_1q_gates == 0
    assert standard.num_2q_gates == 1
    assert ballarin.num_1q_gates == 4
    assert ballarin.num_2q_gates == 1


def test_collector_rejects_native_provenance_from_another_circuit() -> None:
    """Mixing independently compiled logical and native circuits fails early."""
    first = ParameterizedCircuit(2, [ParameterizedGate("rxx", (0, 1), param_index=0)])
    second = ParameterizedCircuit(2, [ParameterizedGate("ryy", (0, 1), param_index=0)])
    source = compile_quantinuum_native(first)

    with pytest.raises(ValueError, match="does not match"):
        collect_circuit_statistics(
            second,
            AnsatzConfig(0, initial_single_qubit_layer=False),
            native_source=source,
        )


def test_collector_rejects_an_unsupported_native_source() -> None:
    """An unsupported native source receives a clear error."""
    logical = ParameterizedCircuit(1, [])
    with pytest.raises(TypeError, match="native_source"):
        collect_circuit_statistics(
            logical,
            AnsatzConfig(0),
            native_source=cast("NativeCircuitSource", object()),
        )


def test_collector_rejects_an_unsupported_evaluated_representation() -> None:
    """An unsupported evaluated representation receives a clear error."""
    logical = ParameterizedCircuit(1, [])
    with pytest.raises(ValueError, match="evaluated_representation"):
        collect_circuit_statistics(
            logical,
            AnsatzConfig(0),
            evaluated_representation=cast("EvaluatedRepresentation", "compiled"),
        )


def test_native_compilation_type_is_publicly_supported() -> None:
    """The collector accepts the documented unpruned native source type."""
    logical = ParameterizedCircuit(1, [ParameterizedGate("h", (0,))])
    source = compile_quantinuum_native(logical)

    assert isinstance(source, NativeCompilation)
    assert collect_circuit_statistics(logical, AnsatzConfig(0), native_source=source).native_gate_counts == {"h": 1}
