# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the DAG utility functions.

This module contains unit tests for the DAG utility functions used in the conversion and processing
of quantum circuits. It verifies that the functions correctly extract gate operations from a DAGCircuit,
group nodes into single-qubit and two-qubit (even/odd) categories, ignore unsupported operations (such as
measure and barrier), and compute properties like the longest gate distance and appropriate starting point
ranges for gate application.
"""

from __future__ import annotations

import copy
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import (
    CRZGate,
    CXGate,
    IGate,
    RZXGate,
    SdgGate,
    SGate,
    SXdgGate,
    TdgGate,
    TGate,
    U1Gate,
    U3Gate,
    UnitaryGate,
)
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.qasm2 import loads
from qiskit.quantum_info import Operator, Statevector

from mqt.yaqs import EquivalenceChecker, State, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import GateLibrary, Rx
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate
from mqt.yaqs.digital.utils.dag_utils import (
    SUPPORTED_QISKIT_GATE_NAMES,
    check_longest_gate,
    convert_dag_to_tensor_algorithm,
    get_temporal_zone,
    select_starting_point,
)
from tests.core.methods.tdvp.conftest import _fidelity
from tests.digital.conftest import _run_strong_noiseless


def test_supported_qiskit_gate_names_exact() -> None:
    """The documented Qiskit gate-name list must match ``GateLibrary`` exactly."""
    assert len(SUPPORTED_QISKIT_GATE_NAMES) == 28
    for name in SUPPORTED_QISKIT_GATE_NAMES:
        assert hasattr(GateLibrary, name), f"GateLibrary missing hardcoded alias '{name}'"
        assert getattr(GateLibrary, name) is not GateLibrary.custom


def _qiskit_gate_matrix(gate_cls: type, *params: float) -> np.ndarray:
    """Return the Qiskit dense matrix for a standard gate class."""
    gate = gate_cls(*params) if params else gate_cls()
    matrix = gate.to_matrix()
    assert matrix is not None
    return np.asarray(matrix, dtype=np.complex128)


@pytest.mark.parametrize(
    ("library_name", "qiskit_gate_cls", "params"),
    [
        ("s", SGate, ()),
        ("sdg", SdgGate, ()),
        ("t", TGate, ()),
        ("tdg", TdgGate, ()),
        ("sxdg", SXdgGate, ()),
        ("i", IGate, ()),
        ("iden", IGate, ()),
        ("u1", U1Gate, (0.37,)),
        ("u3", U3Gate, (0.2, 0.3, 0.4)),
    ],
)
def test_qiskit_gate_aliases_match_qiskit_matrices(
    library_name: str,
    qiskit_gate_cls: type,
    params: tuple[float, ...],
) -> None:
    """Hardcoded Qiskit aliases should match standard Qiskit gate matrices."""
    gate_cls = getattr(GateLibrary, library_name)
    yaqs_gate = gate_cls(list(params)) if params else gate_cls()
    expected = _qiskit_gate_matrix(qiskit_gate_cls, *params)
    assert_allclose(yaqs_gate.matrix, expected, atol=1e-12)


def test_custom_one_qubit_unitary_gate_translation() -> None:
    """Unknown 1-qubit UnitaryGate nodes should translate from the Qiskit matrix."""
    unitary = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    qc = QuantumCircuit(1)
    qc.append(UnitaryGate(unitary), [0])
    dag = circuit_to_dag(qc)
    node = dag.op_nodes()[0]
    assert isinstance(node, DAGOpNode)

    gate = convert_dag_to_tensor_algorithm(node)[0]
    assert gate.name == "unitary"
    assert gate.interaction == 1
    assert gate.sites == [0]
    assert_allclose(gate.matrix, unitary, atol=1e-12)
    assert not hasattr(gate, "generator")


def test_qasm_defined_gate_translates_when_to_matrix_raises() -> None:
    """QASM custom gates without ``to_matrix`` (older Qiskit) still translate via ``Operator``."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    gate prep a,b { h a; cx a,b; }
    qreg q[2];
    prep q[0], q[1];
    """
    qc = loads(qasm)
    dag = circuit_to_dag(qc)
    node = next(n for n in dag.op_nodes() if n.op.name == "prep")
    reference = convert_dag_to_tensor_algorithm(node)[0]

    with patch.object(node.op, "to_matrix", side_effect=CircuitError("to_matrix unavailable")):
        gate = convert_dag_to_tensor_algorithm(node)[0]

    assert gate.name == "prep"
    assert_allclose(gate.matrix, reference.matrix, atol=1e-12)
    assert not hasattr(gate, "generator")


def test_custom_two_qubit_unitary_gate_translation() -> None:
    """Unknown 2-qubit UnitaryGate nodes should build tensor/MPO data like other 2Q gates."""
    unitary = np.asarray(CXGate().to_matrix(), dtype=np.complex128)
    qc = QuantumCircuit(2)
    qc.append(UnitaryGate(unitary), [0, 1])
    dag = circuit_to_dag(qc)

    gate = convert_dag_to_tensor_algorithm(dag)[0]
    expected = GateLibrary.cx()
    expected.set_sites(0, 1)

    assert gate.name == "unitary"
    assert gate.interaction == 2
    assert gate.sites == [0, 1]
    assert_allclose(gate.matrix, expected.matrix, atol=1e-12)
    assert_allclose(gate.tensor, expected.tensor, atol=1e-12)
    assert len(gate.mpo_tensors) == 2


def test_custom_two_qubit_unitary_gate_reversed_qargs() -> None:
    """Reversed two-qubit qargs should store the Qiskit operator matrix for those sites."""
    unitary = np.asarray(CXGate().to_matrix(), dtype=np.complex128)
    qc = QuantumCircuit(2)
    qc.append(UnitaryGate(unitary), [1, 0])
    dag = circuit_to_dag(qc)

    gate = convert_dag_to_tensor_algorithm(dag)[0]
    expected_op = Operator(qc).data

    assert gate.sites == [1, 0]
    assert_allclose(gate.matrix, expected_op, atol=1e-12)


def test_unbound_parameterized_gate_raises() -> None:
    """Unbound symbolic parameters should be rejected with a clear error."""
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    dag = circuit_to_dag(qc)
    node = dag.op_nodes()[0]

    with pytest.raises(ValueError, match="unbound parameters"):
        convert_dag_to_tensor_algorithm(node)


def test_unsupported_reset_instruction_raises() -> None:
    """Reset instructions should be rejected with a clear error."""
    qc = QuantumCircuit(1)
    qc.reset(0)
    dag = circuit_to_dag(qc)
    node = dag.op_nodes()[0]

    with pytest.raises(ValueError, match="reset is not supported"):
        convert_dag_to_tensor_algorithm(node)


def test_unsupported_measure_instruction_raises() -> None:
    """Measure instructions should be rejected during DAG-to-gate conversion."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    dag = circuit_to_dag(qc)
    node = next(n for n in dag.op_nodes() if n.op.name == "measure")

    with pytest.raises(ValueError, match="mid-circuit measurements are not supported"):
        convert_dag_to_tensor_algorithm(node)


def test_unsupported_measure_in_full_dag_raises() -> None:
    """A DAG containing measure nodes should fail conversion."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    dag = circuit_to_dag(qc)

    with pytest.raises(ValueError, match="mid-circuit measurements are not supported"):
        convert_dag_to_tensor_algorithm(dag)


def test_unsupported_control_flow_raises() -> None:
    """Control-flow instructions should be rejected with a clear error."""
    qc = QuantumCircuit(1, 1)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
    dag = circuit_to_dag(qc)
    node = dag.op_nodes()[0]

    with pytest.raises(ValueError, match="control-flow"):
        convert_dag_to_tensor_algorithm(node)


def _haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random unitary matrix of size ``dim x dim``.

    Returns:
        A ``dim x dim`` unitary matrix.
    """
    z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(z)
    phases = np.diagonal(r) / np.abs(np.diagonal(r))
    return np.asarray(q * phases, dtype=np.complex128)


# Fixed Haar-random 2-qubit unitary (seed 42); not symmetric under qubit interchange.
_FIXED_NONSYMMETRIC_2Q = _haar_unitary(4, np.random.default_rng(42))


def test_custom_one_qubit_unitary_matches_qiskit_statevector() -> None:
    """Custom 1-qubit UnitaryGate circuits should match Qiskit statevectors."""
    unitary = _haar_unitary(2, np.random.default_rng(0))
    qc = QuantumCircuit(2)
    qc.append(UnitaryGate(unitary), [1])
    qc.h(0)

    vec = _run_strong_noiseless(qc, gate_mode="tdvp", get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


def test_custom_two_qubit_unitary_matches_qiskit_statevector() -> None:
    """Custom 2-qubit UnitaryGate circuits should match Qiskit statevectors."""
    unitary = _haar_unitary(4, np.random.default_rng(1))
    qc = QuantumCircuit(2)
    qc.append(UnitaryGate(unitary), [0, 1])
    qc.h(0)

    vec = _run_strong_noiseless(qc, gate_mode="tdvp", get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


def test_custom_two_qubit_unitary_reversed_qargs_matches_qiskit() -> None:
    """Reversed custom two-qubit qargs on a long-range pair should match Qiskit."""
    unitary = _haar_unitary(4, np.random.default_rng(2))
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.append(UnitaryGate(unitary), [2, 0])

    vec = _run_strong_noiseless(qc, gate_mode="mpo", get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


def test_custom_long_range_two_qubit_unitary_matches_qiskit() -> None:
    """Long-range custom two-qubit gates should work via the MPO application path."""
    unitary = _haar_unitary(4, np.random.default_rng(3))
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.append(UnitaryGate(unitary), [0, 2])

    vec = _run_strong_noiseless(qc, gate_mode="mpo", get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize(
    ("gate_name", "qargs", "gate_mode"),
    [
        ("swap", (0, 1), "tdvp"),
        ("swap", (1, 0), "tdvp"),
        ("cz", (0, 1), "tdvp"),
        ("cz", (1, 0), "tdvp"),
        ("cx", (0, 1), "tdvp"),
        ("cx", (1, 0), "tdvp"),
    ],
)
def test_builtin_two_qubit_gate_reversed_qargs_match_qiskit(
    gate_name: str,
    qargs: tuple[int, int],
    gate_mode: Literal["tdvp"],
) -> None:
    """Built-in two-qubit gates should match Qiskit for forward and reversed qargs."""
    qc = QuantumCircuit(2)
    qc.h(0)
    getattr(qc, gate_name)(*qargs)

    vec = _run_strong_noiseless(qc, gate_mode=gate_mode, get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize(
    ("num_qubits", "qargs", "gate_mode"),
    [
        (2, (0, 1), "tdvp"),
        (2, (1, 0), "tdvp"),
        (3, (0, 2), "mpo"),
        (3, (2, 0), "mpo"),
    ],
)
def test_fixed_nonsymmetric_two_qubit_unitary_qarg_ordering(
    num_qubits: int,
    qargs: tuple[int, int],
    gate_mode: Literal["tdvp", "mpo"],
) -> None:
    """A fixed non-symmetric custom unitary should match Qiskit for every qarg ordering."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.append(UnitaryGate(_FIXED_NONSYMMETRIC_2Q), list(qargs))

    vec = _run_strong_noiseless(qc, gate_mode=gate_mode, get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


class _CustomNamedUnitary(Gate):
    """Qiskit gate whose ``name`` collides with a non-alias ``GateLibrary`` attribute."""

    def __init__(self) -> None:
        super().__init__("custom", 1, [])

    def to_matrix(self) -> np.ndarray:  # noqa: PLR6301
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def test_translate_unknown_name_skips_gate_library_attributes() -> None:
    """Names outside ``SUPPORTED_QISKIT_GATE_NAMES`` must not bind ``GateLibrary`` helpers."""
    assert "custom" not in SUPPORTED_QISKIT_GATE_NAMES
    assert hasattr(GateLibrary, "custom")

    qc = QuantumCircuit(1)
    qc.append(_CustomNamedUnitary(), [0])
    dag = circuit_to_dag(qc)
    gate = convert_dag_to_tensor_algorithm(dag)[0]

    assert gate.name == "custom"
    assert_allclose(gate.matrix, _CustomNamedUnitary().to_matrix())


@pytest.mark.parametrize(
    ("qiskit_gate_cls", "params", "qargs"),
    [
        (CRZGate, (0.37,), (0, 1)),
        (RZXGate, (0.42,), (0, 1)),
    ],
)
def test_qiskit_standard_gate_matrix_fallback_matches_qiskit(
    qiskit_gate_cls: type,
    params: tuple[float, ...],
    qargs: tuple[int, int],
) -> None:
    """Unsupported standard Qiskit gates should use the matrix fallback and match Qiskit."""
    gate_name = qiskit_gate_cls(*params).name
    assert gate_name not in SUPPORTED_QISKIT_GATE_NAMES

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.append(qiskit_gate_cls(*params), list(qargs))
    dag = circuit_to_dag(qc)
    gate = next(g for g in convert_dag_to_tensor_algorithm(dag) if g.name == gate_name)

    assert gate.name == gate_name
    assert not hasattr(gate, "generator")

    vec = _run_strong_noiseless(qc, gate_mode="tdvp", get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


def test_unitary_gate_equivalent_to_decomposition() -> None:
    """A custom ``UnitaryGate`` circuit should be equivalent to an equivalent decomposition."""
    theta = 0.31
    unitary = np.asarray(GateLibrary.rx([theta]).matrix, dtype=np.complex128)

    qc_unitary = QuantumCircuit(1)
    qc_unitary.append(UnitaryGate(unitary), [0])

    qc_decomposed = QuantumCircuit(1)
    qc_decomposed.rx(theta, 0)

    result = EquivalenceChecker(representation="matrix", fidelity=1 - 1e-12).check(qc_unitary, qc_decomposed)
    assert result["equivalent"] is True


def _haar_unitary_2q(seed: int) -> np.ndarray:
    """Return a Haar-random ``4 x 4`` unitary for generator-less routing tests."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    q, r = np.linalg.qr(z)
    phases = np.diagonal(r) / np.abs(np.diagonal(r))
    return np.asarray(q * phases, dtype=np.complex128)


def test_generator_less_nn_custom_gate_routes_tebd() -> None:
    """Custom NN gates without a generator should use TEBD in ``gate_mode='tdvp'``."""
    unitary = _haar_unitary_2q(11)
    qc = QuantumCircuit(2)
    qc.append(UnitaryGate(unitary), [0, 1])
    dag = circuit_to_dag(qc)
    node = dag.op_nodes()[0]
    gate = convert_dag_to_tensor_algorithm(node)[0]
    assert not hasattr(gate, "generator")

    out = copy.deepcopy(State(2, initial="zeros").mps)
    params = StrongSimParams(
        observables=[],
        gate_mode="tdvp",
        preset="exact",
        svd_threshold=1e-12,
        tdvp_sweeps=1,
    )

    with (
        patch("mqt.yaqs.digital.digital_tjm.apply_two_qubit_gate_tebd") as mock_tebd,
        patch("mqt.yaqs.digital.digital_tjm.apply_two_qubit_gate_tdvp") as mock_tdvp,
        patch("mqt.yaqs.digital.digital_tjm.apply_long_range_gate_mpo") as mock_mpo,
    ):
        mock_tebd.return_value = (0, 1)
        apply_two_qubit_gate(out, node, params)
        mock_tebd.assert_called_once()
        mock_tdvp.assert_not_called()
        mock_mpo.assert_not_called()


def test_generator_less_lr_custom_gate_routes_mpo() -> None:
    """Custom LR gates without a generator should use the MPO path in ``gate_mode='tdvp'``."""
    unitary = _haar_unitary_2q(13)
    length = 3
    qc = QuantumCircuit(length)
    qc.append(UnitaryGate(unitary), [0, length - 1])
    dag = circuit_to_dag(qc)
    node = dag.op_nodes()[0]
    gate = convert_dag_to_tensor_algorithm(node)[0]
    assert not hasattr(gate, "generator")

    out = copy.deepcopy(State(length, initial="zeros").mps)
    params = StrongSimParams(
        observables=[],
        gate_mode="tdvp",
        preset="exact",
        svd_threshold=1e-12,
        tdvp_sweeps=1,
    )

    with (
        patch("mqt.yaqs.digital.digital_tjm.apply_two_qubit_gate_tebd") as mock_tebd,
        patch("mqt.yaqs.digital.digital_tjm.apply_two_qubit_gate_tdvp") as mock_tdvp,
        patch("mqt.yaqs.digital.digital_tjm.apply_long_range_gate_mpo") as mock_mpo,
    ):
        mock_mpo.return_value = (0, length - 1)
        apply_two_qubit_gate(out, node, params)
        mock_mpo.assert_called_once()
        mock_tdvp.assert_not_called()
        mock_tebd.assert_not_called()


def test_convert_dag_to_tensor_algorithm_single_qubit_gate() -> None:
    """Test converting a DAGCircuit with a single-qubit X gate.

    This test creates a quantum circuit with one qubit and applies an X gate.
    It then converts the circuit to a DAG and uses convert_dag_to_tensor_algorithm
    to extract the gate. The test verifies that exactly one gate is returned,
    the gate's name matches 'x' (case-insensitive), and it acts on qubit 0.
    """
    qc = QuantumCircuit(1)
    qc.x(0)
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Should have one gate from a single X."
    gate = gates[0]
    assert gate.name.lower() == "x", "Gate name should match 'x'."
    assert gate.sites == [0], "Gate should act on qubit 0."


def test_convert_dag_to_tensor_algorithm_two_qubit_gate() -> None:
    """Test converting a DAGCircuit with a two-qubit CX gate.

    This test creates a two-qubit circuit with a controlled-NOT gate (CX) and converts it
    to a DAG. It then verifies that the conversion produces one two-qubit gate with a name
    matching 'cx' or 'cnot' (case-insensitive) and that the gate acts on qubits 0 and 1.
    """
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Should have one 2-qubit gate (CX)."
    gate = gates[0]
    assert gate.name.lower() in {"cx", "cnot"}, "Gate name should match CX/CNOT."
    assert gate.sites == [0, 1], "Gate should act on qubits 0 and 1."


def test_convert_dag_to_tensor_algorithm_two_qubit_gate_flipped() -> None:
    """Test converting a DAGCircuit with a two-qubit CX gate where control and target are flipped.

    This test creates a two-qubit circuit with a CX gate applied from qubit 1 to qubit 0,
    converts the circuit to a DAG, and extracts the gate. It verifies that the gate's name
    matches 'cx' or 'cnot' and that the gate correctly reflects the flipped qubit order.
    """
    qc = QuantumCircuit(2)
    qc.cx(1, 0)
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Should have one 2-qubit gate (CX)."
    gate = gates[0]
    assert gate.name.lower() in {"cx", "cnot"}, "Gate name should match CX/CNOT."
    assert gate.sites == [1, 0], "Gate should act on qubits 1 and 0."


def test_convert_dag_to_tensor_algorithm_single_dagopnode() -> None:
    """Test converting a single DAGOpNode representing a single-qubit RX gate.

    This test creates a circuit with a single RX gate, converts it to a DAG,
    and then extracts the first op node. The conversion function should return one gate
    with name 'rx', capture the rotation parameter (theta) correctly, and specify that it acts on qubit 0.
    """
    qc = QuantumCircuit(1)
    qc.rx(np.pi / 4, 0)
    dag = circuit_to_dag(qc)
    node = dag.op_nodes()[0]
    assert isinstance(node, DAGOpNode)

    gates = convert_dag_to_tensor_algorithm(node)
    assert len(gates) == 1, "Expected one gate from a single DAGOpNode."
    gate = gates[0]
    assert gate.name.lower() == "rx", "Gate name should match 'rx'."
    assert isinstance(gate, Rx), "Gate should be an Rx instance."
    assert gate.theta == pytest.approx(np.pi / 4), "Gate should capture the rotation parameter (pi/4)."
    assert gate.sites == [0], "Gate should act on qubit 0."


def test_convert_dag_to_tensor_algorithm_ignores_barrier() -> None:
    """Test that convert_dag_to_tensor_algorithm ignores barrier nodes but rejects measure.

    This test constructs a two-qubit circuit that includes an X gate and a barrier.
    After converting the circuit to a DAG and extracting gates, the function should
    return only the X gate. Measurements must be rejected explicitly.
    """
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.barrier()
    dag = circuit_to_dag(qc)

    gates = convert_dag_to_tensor_algorithm(dag)
    assert len(gates) == 1, "Only the X gate should be extracted."
    gate = gates[0]
    assert gate.name.lower() == "x", "The extracted gate should be an X gate."

    qc_measure = QuantumCircuit(2, 2)
    qc_measure.x(0)
    qc_measure.barrier()
    qc_measure.measure_all()
    with pytest.raises(ValueError, match="mid-circuit measurements are not supported"):
        convert_dag_to_tensor_algorithm(circuit_to_dag(qc_measure))


def test_get_temporal_zone_simple() -> None:
    """Test extracting the temporal zone for a subset of qubits from a DAGCircuit.

    This test creates a three-qubit circuit with two X gates and one CX gate.
    It then extracts the temporal zone for qubits 0 and 1, expecting only the two single-qubit X gates
    to be present in the temporal zone.
    """
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.cx(1, 2)
    dag = circuit_to_dag(qc)

    new_dag = get_temporal_zone(dag, [0, 1])
    new_nodes = new_dag.op_nodes()
    assert len(new_nodes) == 2, "Should only have the 2 single-qubit gates in the temporal zone."


def test_check_longest_gate() -> None:
    """Test the computation of the longest gate distance in the first layer of a DAGCircuit.

    This test creates a three-qubit circuit with two CX gates: one between qubits 0 and 2 and one
    between qubits 0 and 1. The function check_longest_gate should return 3, indicating that the maximum distance
    between involved qubits is 3.
    """
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)

    dist = check_longest_gate(dag)
    assert dist == 3, f"Longest distance should be 3, got {dist}"


def test_select_starting_point_even_odd() -> None:
    """Test selecting starting points for gate application.

    This test creates a 4-qubit circuit with a CX gate starting at qubit 0.
    The function select_starting_point should return ranges corresponding to even-odd pairings:
    first_iterator = range(0, 3, 2) and second_iterator = range(1, 3, 2).
    """
    num_qubits = 4
    qc = QuantumCircuit(num_qubits)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)

    first_iter, second_iter = select_starting_point(num_qubits, dag)
    assert first_iter == range(0, 3, 2), "Expected the default even qubits first."
    assert second_iter == range(1, 3, 2), "Then the odd qubit pairs."


def test_select_starting_point_odd() -> None:
    """Test selecting starting points when the first two-qubit gate starts at an odd qubit.

    This test creates a 4-qubit circuit with a CX gate starting at qubit 1.
    The function select_starting_point should return ranges with odd qubits first:
    first_iterator = range(1, 3, 2) and second_iterator = range(0, 3, 2).
    """
    num_qubits = 4
    qc = QuantumCircuit(num_qubits)
    qc.cx(1, 2)

    dag = circuit_to_dag(qc)
    first_iter, second_iter = select_starting_point(num_qubits, dag)

    assert first_iter == range(1, 3, 2), "Expected odd qubits first."
    assert second_iter == range(0, 3, 2), "Then the even qubit pairs."
