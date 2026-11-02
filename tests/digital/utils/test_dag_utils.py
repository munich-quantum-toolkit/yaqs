# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff:file-ignore[import-private-name] -- white-box tests import the private gate builders of dag_utils

"""Tests for the DAG utility functions.

This module contains unit tests for the DAG utility functions used in the conversion and processing
of quantum circuits. It verifies that the functions correctly extract gate operations from a DAGCircuit,
group nodes into single-qubit and two-qubit (even/odd) categories, ignore unsupported operations (such as
measure and barrier), and compute properties like the longest gate distance and appropriate starting point
ranges for gate application.
"""

from __future__ import annotations

import copy
import pickle  # ruff: ignore[suspicious-pickle-import]  # controlled test round-trips; no untrusted input
from itertools import permutations
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qiskit import QuantumCircuit
from qiskit.circuit import (
    AnnotatedOperation,
    Barrier,
    ControlledGate,
    ControlModifier,
    Gate,
    InverseModifier,
    Operation,
    Parameter,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import (
    CCXGate,
    CPhaseGate,
    CRZGate,
    CSwapGate,
    CUGate,
    CXGate,
    GlobalPhaseGate,
    HGate,
    IGate,
    MCMTGate,
    MCXGate,
    PauliEvolutionGate,
    PauliGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    SXdgGate,
    TdgGate,
    TGate,
    U1Gate,
    U3Gate,
    UnitaryGate,
    XGate,
    ZGate,
)
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.qasm2 import loads
from qiskit.quantum_info import Operator, PauliList, SparsePauliOp, Statevector
from scipy.linalg import expm

from mqt.yaqs import DigitalSimParams, EquivalenceChecker, State
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.libraries.gate_library import GateLibrary, Rx
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate
from mqt.yaqs.digital.utils.dag_utils import (
    SUPPORTED_QISKIT_GATE_NAMES,
    _ControlledGate,
    _PauliSumGate,
    _target_factors,
    _translate_pauli_evolution,
    _translate_pauli_string,
    check_longest_gate,
    convert_dag_to_tensor_algorithm,
    get_temporal_zone,
    is_digital_noise_opportunity,
    select_starting_point,
)
from tests.conftest import YAQS_TEST_SEED
from tests.core.methods.tdvp.conftest import _fidelity
from tests.digital.conftest import _run_digital_observables_noiseless


def test_supported_qiskit_gate_names_exact() -> None:
    """The documented Qiskit gate-name list must match ``GateLibrary`` exactly."""
    assert len(SUPPORTED_QISKIT_GATE_NAMES) == 31
    for name in SUPPORTED_QISKIT_GATE_NAMES:
        assert hasattr(GateLibrary, name), f"GateLibrary missing hardcoded alias '{name}'"
        assert getattr(GateLibrary, name) is not GateLibrary.custom


def _wrapped_cx() -> Operation:
    """Return a unitary two-qubit Instruction that is not a Gate."""
    definition = QuantumCircuit(2, name="wrapped_cx")
    definition.cx(0, 1)
    return definition.to_instruction()


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        pytest.param(HGate(), "skip", id="qiskit-h"),
        pytest.param(CXGate(), "noise", id="qiskit-cx"),
        pytest.param(_wrapped_cx(), "noise", id="unitary-instruction"),
        pytest.param(Barrier(2), "skip", id="barrier"),
        pytest.param(
            Gate("reset", 2, []),
            "skip",
            id="rejected-reset",
        ),
        pytest.param(Gate("opaque", 2, []), "skip", id="non-matrix-operation"),
        pytest.param(
            SimpleNamespace(name="cx", num_qubits=2, condition=object()),
            "skip",
            id="conditional-cx",
        ),
    ],
)
def test_digital_noise_opportunity(operation: Operation, expected: str) -> None:
    """Only supported two-qubit unitary operations are opportunities."""
    assert is_digital_noise_opportunity(operation) is (expected == "noise")


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(CPhaseGate(Parameter("theta")), id="cp"),
        pytest.param(RXXGate(Parameter("theta")), id="rxx"),
        pytest.param(RYYGate(Parameter("theta")), id="ryy"),
        pytest.param(RZZGate(Parameter("theta")), id="rzz"),
    ],
)
def test_unbound_supported_gate_is_not_digital_noise_opportunity(operation: Operation) -> None:
    """Supported two-qubit gates require bound parameters to be noise opportunities."""
    assert operation.name in SUPPORTED_QISKIT_GATE_NAMES
    assert is_digital_noise_opportunity(operation) is False


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


def test_wide_matrix_fallback_gate_raises() -> None:
    """Matrix-fallback gates wider than the documented maximum should be rejected."""
    qc = QuantumCircuit(9)
    qc.append(Gate("wide", 9, []), range(9))
    dag = circuit_to_dag(qc)

    with pytest.raises(ValueError, match="matrix fallback supports at most"):
        convert_dag_to_tensor_algorithm(dag)


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

    vec = _run_digital_observables_noiseless(qc, gate_mode="tdvp", get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


def test_custom_two_qubit_unitary_matches_qiskit_statevector() -> None:
    """Custom 2-qubit UnitaryGate circuits should match Qiskit statevectors."""
    unitary = _haar_unitary(4, np.random.default_rng(1))
    qc = QuantumCircuit(2)
    qc.append(UnitaryGate(unitary), [0, 1])
    qc.h(0)

    vec = _run_digital_observables_noiseless(qc, gate_mode="tdvp", get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


def test_custom_two_qubit_unitary_reversed_qargs_matches_qiskit() -> None:
    """Reversed custom two-qubit qargs on a long-range pair should match Qiskit."""
    unitary = _haar_unitary(4, np.random.default_rng(2))
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.append(UnitaryGate(unitary), [2, 0])

    vec = _run_digital_observables_noiseless(qc, gate_mode="mpo", get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


def test_custom_long_range_two_qubit_unitary_matches_qiskit() -> None:
    """Long-range custom two-qubit gates should work via the MPO application path."""
    unitary = _haar_unitary(4, np.random.default_rng(3))
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.append(UnitaryGate(unitary), [0, 2])

    vec = _run_digital_observables_noiseless(qc, gate_mode="mpo", get_state=True)
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

    vec = _run_digital_observables_noiseless(qc, gate_mode=gate_mode, get_state=True)
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

    vec = _run_digital_observables_noiseless(qc, gate_mode=gate_mode, get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


def test_ccx_translates_to_gate_library_class() -> None:
    """CCX nodes should translate to the hardcoded ``GateLibrary.ccx`` class with a generator."""
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    dag = circuit_to_dag(qc)

    gate = convert_dag_to_tensor_algorithm(dag)[0]
    assert gate.name == "ccx"
    assert gate.interaction == 3
    assert gate.sites == [0, 1, 2]
    assert len(gate.generator) == 3
    assert len(gate.mpo_tensors) == 3


def test_cswap_translates_without_generator() -> None:
    """CSWAP nodes should translate to the hardcoded ``GateLibrary.cswap`` class without a generator."""
    qc = QuantumCircuit(3)
    qc.cswap(0, 1, 2)
    dag = circuit_to_dag(qc)

    gate = convert_dag_to_tensor_algorithm(dag)[0]
    assert gate.name == "cswap"
    assert gate.interaction == 3
    assert gate.sites == [0, 1, 2]
    assert not hasattr(gate, "generator")
    assert len(gate.mpo_tensors) == 3


def _seeded_diagonal_unitary(dimension: int) -> np.ndarray:
    """Return a diagonal unitary that is not a tensor product of one-qubit factors."""
    rng = np.random.default_rng(20261031)
    return np.diag(np.exp(2j * np.pi * rng.random(dimension)))


def _controlled(base: Gate, num_ctrl_qubits: int, ctrl_state: int | None = None) -> ControlledGate:
    """Controlled gate built directly, without the decomposition that ``UnitaryGate.control`` synthesizes.

    Returns:
        The controlled gate with ``base`` as its target operator.
    """
    return ControlledGate(
        f"c{num_ctrl_qubits}{base.name}",
        num_ctrl_qubits + base.num_qubits,
        base.params,
        num_ctrl_qubits=num_ctrl_qubits,
        ctrl_state=ctrl_state,
        base_gate=base,
    )


def _dense_controlled_gate(
    num_qubits: int, controls: dict[int, int], targets: list[int], target_matrix: np.ndarray
) -> np.ndarray:
    """Dense ``|b><b| (x) U + (I - |b><b|) (x) I`` with site 0 least significant and ``U`` in Qiskit order.

    Returns:
        The ``2**num_qubits`` square matrix of the controlled gate.
    """
    out = np.eye(2**num_qubits, dtype=np.complex128)
    for col in range(2**num_qubits):
        if all((col >> site) & 1 == value for site, value in controls.items()):
            target_in = sum(((col >> site) & 1) << j for j, site in enumerate(targets))
            rest = col & ~sum(1 << site for site in targets)
            out[col, col] = 0.0
            for target_out in range(2 ** len(targets)):
                row = rest | sum(((target_out >> j) & 1) << site for j, site in enumerate(targets))
                out[row, col] = target_matrix[target_out, target_in]
    return out


def test_wide_controlled_gate_builds_its_mpo_without_a_matrix() -> None:
    """A controlled gate on more than eight qubits is built from its controls and target operator."""
    qc = QuantumCircuit(9)
    qc.mcx(list(range(8)), 8)

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert gate.interaction == 9
    assert gate.sites == list(range(9))
    assert not hasattr(gate, "matrix")
    assert not hasattr(gate, "generator")
    assert len(gate.mpo_tensors) == 9


def test_annotated_wide_controlled_gate_builds_its_mpo() -> None:
    """A control given as an annotation takes the same path as a plain controlled gate."""
    qc = QuantumCircuit(12)
    qc.append(ZGate().control(11, annotated=True), range(12))

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert not hasattr(gate, "matrix")
    assert [tensor.shape[3] for tensor in gate.mpo_tensors[:-1]] == [2] * 11


def test_annotated_wide_controlled_gate_matches_the_dense_gate() -> None:
    """The control state of an annotated controlled gate is read from its modifier."""
    ctrl_state = 5
    controls = {site: (ctrl_state >> site) & 1 for site in range(9)}
    qc = QuantumCircuit(10)
    qc.append(ZGate().control(9, annotated=True, ctrl_state=ctrl_state), range(10))

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    expected = _dense_controlled_gate(10, controls, [9], np.asarray(ZGate().to_matrix(), dtype=np.complex128))
    assert_allclose(MPO.from_gate(gate, 10).to_matrix(), expected, atol=1e-12)


def test_annotated_operation_that_is_not_only_controls_falls_back() -> None:
    """An annotation carrying more than control modifiers keeps the fallback limit."""
    qc = QuantumCircuit(10)
    qc.append(AnnotatedOperation(ZGate(), [ControlModifier(9), InverseModifier()]), range(10))

    with pytest.raises(ValueError, match="matrix fallback supports at most"):
        convert_dag_to_tensor_algorithm(circuit_to_dag(qc))


@pytest.mark.parametrize(
    ("gate", "qubits", "bonds"),
    [
        # The identity string and one controlled string differ on every cut.
        pytest.param(
            MCXGate(9, ctrl_state=0b101101011),
            [0, 1, 2, 4, 6, 7, 9, 10, 11, 5],
            [2] * 11,
            id="mcx-controls-both-sides",
        ),
        # A phase on the target acts on the controls alone, so the bond is 1 after the last control.
        pytest.param(
            _controlled(UnitaryGate(np.exp(0.3j) * np.eye(2)), 9),
            [*range(9), 10],
            [2] * 8 + [1, 1],
            id="phase-target",
        ),
        # The controls lie left of SWAP, so the cut between its targets carries the rank 4 of SWAP - I.
        pytest.param(SwapGate().control(8, annotated=False), list(range(10)), [2] * 8 + [4], id="controlled-swap"),
        # The same with the controls right of SWAP: the reduction acts on the cut before the first control.
        pytest.param(
            SwapGate().control(8, annotated=False),
            [*range(2, 10), 0, 1],
            [4] + [2] * 8,
            id="controlled-swap-targets-first",
        ),
        # A phase on two targets also acts on the controls alone, on either side of them.
        pytest.param(_controlled(UnitaryGate(-np.eye(4)), 8), list(range(10)), [2] * 7 + [1, 1], id="two-target-phase"),
        pytest.param(
            _controlled(UnitaryGate(-np.eye(4)), 8),
            [*range(2, 10), 0, 1],
            [1, 1] + [2] * 7,
            id="two-target-phase-targets-first",
        ),
        # U - I has rank 4 across its targets: 4 past the outer controls, 1 + 4 between them.
        pytest.param(
            _controlled(UnitaryGate(_haar_unitary(4, np.random.default_rng(YAQS_TEST_SEED))), 8),
            [*range(1, 9), 0, 9],
            [4] + [5] * 7 + [4],
            id="targets-outside-the-controls",
        ),
    ],
)
def test_wide_controlled_gate_mpo_has_the_minimal_bond(gate: Gate, qubits: list[int], bonds: list[int]) -> None:
    """Every cut of a wide controlled gate carries the operator-Schmidt rank of the gate on that cut."""
    qc = QuantumCircuit(max(qubits) + 1)
    qc.append(gate, qubits)

    (yaqs_gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert [tensor.shape[3] for tensor in yaqs_gate.mpo_tensors[:-1]] == bonds


@pytest.mark.parametrize(
    ("num_qubits", "controls", "targets", "target_matrix"),
    [
        pytest.param(
            9,
            {0: 1, 1: 0, 2: 1, 3: 1, 5: 0, 6: 1, 7: 0, 8: 1},
            [4],
            _haar_unitary(2, np.random.default_rng(YAQS_TEST_SEED)),
            id="one-target",
        ),
        pytest.param(
            10,
            {0: 1, 1: 0, 2: 1, 3: 1, 5: 0, 6: 1, 7: 1},
            [9, 4],
            _haar_unitary(4, np.random.default_rng(YAQS_TEST_SEED)),
            id="two-targets-descending",
        ),
        pytest.param(
            10,
            {1: 1, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1},
            [9, 0],
            _haar_unitary(4, np.random.default_rng(YAQS_TEST_SEED)),
            id="targets-at-both-ends",
        ),
        pytest.param(
            10,
            {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 6: 1, 8: 0},
            [9, 5, 7],
            _haar_unitary(8, np.random.default_rng(YAQS_TEST_SEED)),
            id="three-targets-cycled",
        ),
        pytest.param(
            10,
            {0: 1, 1: 0, 2: 1, 3: 1, 5: 0, 6: 1, 7: 0, 8: 1, 9: 1},
            [4],
            np.exp(0.3j) * np.eye(2),
            id="phase-target",
        ),
        pytest.param(9, {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}, [8], np.eye(2), id="identity-target"),
    ],
)
def test_wide_controlled_gate_matches_the_dense_gate(
    num_qubits: int, controls: dict[int, int], targets: list[int], target_matrix: np.ndarray
) -> None:
    """A wide controlled gate equals its definition for any control pattern and target order."""
    ctrl_state = sum(value << i for i, value in enumerate(controls.values()))
    qc = QuantumCircuit(num_qubits)
    qc.append(_controlled(UnitaryGate(target_matrix), len(controls), ctrl_state), [*controls, *targets])

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    expected = _dense_controlled_gate(num_qubits, controls, targets, target_matrix)
    assert_allclose(MPO.from_gate(gate, num_qubits).to_matrix(), expected, atol=1e-12)


@pytest.mark.parametrize(
    ("gate", "targets", "target_in", "target_out", "expected"),
    [
        pytest.param(_controlled(SwapGate(), 100), [100, 101], [0, 1], [1, 0], 1.0, id="swap-after-100-controls"),
        pytest.param(
            _controlled(UnitaryGate(np.diag(np.exp(-1e-6j * np.array([1, -1, -1, 1])))), 40),
            [0, 1],
            [0, 0],
            [0, 0],
            np.exp(-1e-6j),
            id="near-identity-before-40-controls",
        ),
    ],
)
def test_wide_controlled_gate_keeps_its_action_under_many_controls(
    gate: ControlledGate, targets: list[int], target_in: list[int], target_out: list[int], expected: complex
) -> None:
    """The bond reduction keeps the controlled part, which the identity outweighs by 2**(m/2) in norm."""
    num_qubits = gate.num_qubits
    qc = QuantumCircuit(num_qubits)
    qc.append(gate, [*(site for site in range(num_qubits) if site not in targets), *targets])

    (yaqs_gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    bits_in, bits_out = [1] * num_qubits, [1] * num_qubits
    for site, bit_in, bit_out in zip(targets, target_in, target_out, strict=True):
        bits_in[site], bits_out[site] = bit_in, bit_out
    amplitude = np.linalg.multi_dot([
        tensor[bit_out, bit_in]
        for tensor, bit_in, bit_out in zip(yaqs_gate.mpo_tensors, bits_in, bits_out, strict=True)
    ])
    assert amplitude[0, 0] == pytest.approx(expected, abs=1e-12)


def test_phase_on_several_targets_needs_no_svd() -> None:
    """A phase on several targets is built from its strings, so the SVD fallback never runs."""
    qc = QuantumCircuit(6)
    qc.append(_controlled(UnitaryGate(-np.eye(4)), 4), range(6))

    with patch("mqt.yaqs.digital.utils.dag_utils._control_window_mpo", side_effect=AssertionError("SVD fallback")):
        (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert [tensor.shape[3] for tensor in gate.mpo_tensors[:-1]] == [2, 2, 2, 1, 1]


@pytest.mark.parametrize(
    "gate",
    [
        pytest.param(
            ControlledGate("c8x", 11, [], num_ctrl_qubits=8, base_gate=XGate()),
            id="target-with-ancillas",
        ),
        pytest.param(
            _controlled(UnitaryGate(_seeded_diagonal_unitary(512), check_input=False), 2),
            id="nine-dense-targets",
        ),
        pytest.param(CUGate(0.1, 0.2, 0.3, 0.4).control(7, annotated=False), id="base-without-matrix"),
        pytest.param(GlobalPhaseGate(0.3).control(9, annotated=False), id="no-target"),
        pytest.param(_controlled(UnitaryGate(np.diag([1.0, 2.0]), check_input=False), 8), id="non-unitary-base"),
    ],
)
def test_wide_controlled_gate_without_a_single_target_operator_falls_back(gate: Gate) -> None:
    """A wide controlled gate that is not controls plus one unitary target operator keeps the fallback limit."""
    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, range(gate.num_qubits))

    with pytest.raises(ValueError, match="matrix fallback supports at most"):
        convert_dag_to_tensor_algorithm(circuit_to_dag(qc))


def test_wide_controlled_gate_survives_pickling() -> None:
    """A wide controlled gate keeps its MPO through pickling, as trajectory workers receive it."""
    qc = QuantumCircuit(10)
    qc.mcx(list(range(9)), 9)
    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    restored = pickle.loads(pickle.dumps(gate))  # ruff: ignore[suspicious-pickle-usage]  # controlled round-trip

    assert restored.sites == gate.sites
    for restored_tensor, tensor in zip(restored.mpo_tensors, gate.mpo_tensors, strict=True):
        assert_allclose(restored_tensor, tensor)


def test_custom_three_qubit_unitary_gate_translation() -> None:
    """Unknown 3-qubit UnitaryGate nodes should build tensor/MPO data via the matrix fallback."""
    unitary = _haar_unitary(8, np.random.default_rng(4))
    qc = QuantumCircuit(3)
    qc.append(UnitaryGate(unitary), [0, 1, 2])
    dag = circuit_to_dag(qc)

    gate = convert_dag_to_tensor_algorithm(dag)[0]
    assert gate.name == "unitary"
    assert gate.interaction == 3
    assert gate.sites == [0, 1, 2]
    assert gate.tensor.shape == (2,) * 6
    assert len(gate.mpo_tensors) == 3
    assert not hasattr(gate, "generator")


def test_custom_three_qubit_unitary_reversed_qargs() -> None:
    """Fully reversed three-qubit qargs should store the Qiskit operator matrix for those sites."""
    unitary = _haar_unitary(8, np.random.default_rng(5))
    qc = QuantumCircuit(3)
    qc.append(UnitaryGate(unitary), [2, 1, 0])
    dag = circuit_to_dag(qc)

    gate = convert_dag_to_tensor_algorithm(dag)[0]
    expected_op = Operator(qc).data

    assert gate.sites == [2, 1, 0]
    assert_allclose(gate.matrix, expected_op, atol=1e-12)


# Fixed Haar-random 3-qubit unitary (seed 43); not symmetric under qubit interchange.
_FIXED_NONSYMMETRIC_3Q = _haar_unitary(8, np.random.default_rng(43))


@pytest.mark.parametrize("qargs", list(permutations((0, 1, 2))))
def test_fixed_nonsymmetric_three_qubit_unitary_qarg_ordering(qargs: tuple[int, int, int]) -> None:
    """A fixed non-symmetric three-qubit unitary should match Qiskit for every qarg ordering."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.append(UnitaryGate(_FIXED_NONSYMMETRIC_3Q), list(qargs))

    vec = _run_digital_observables_noiseless(qc, gate_mode="mpo", get_state=True)
    assert isinstance(vec, np.ndarray)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    assert _fidelity(ref, vec) == pytest.approx(1.0, abs=1e-10)


class _CustomNamedUnitary(Gate):
    """Qiskit gate whose ``name`` collides with a non-alias ``GateLibrary`` attribute."""

    def __init__(self) -> None:
        super().__init__("custom", 1, [])

    def to_matrix(self) -> np.ndarray:  # ruff:ignore[no-self-use]
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

    vec = _run_digital_observables_noiseless(qc, gate_mode="tdvp", get_state=True)
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
    params = DigitalSimParams(
        get_state=True,
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
    params = DigitalSimParams(
        get_state=True,
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


@pytest.mark.parametrize("qargs", [(0, 1, 2), (0, 2, 4)])
@pytest.mark.parametrize("gate_kind", ["ccx", "unitary"])
@pytest.mark.parametrize("gate_mode", ["mpo", "tdvp", "full-tdvp", "swaps"])
def test_multi_qubit_gate_routing(
    gate_mode: Literal["mpo", "tdvp", "full-tdvp", "swaps"],
    gate_kind: Literal["ccx", "unitary"],
    qargs: tuple[int, int, int],
) -> None:
    """Gates on three or more qubits route to the TDVP window or the gate-MPO path, never TEBD.

    The TDVP modes use the generator window for gates with a generator (``ccx``); all other
    combinations, including ``gate_mode="swaps"`` and matrix-backed gates, use the gate-MPO path.
    """
    length = 5
    qc = QuantumCircuit(length)
    if gate_kind == "ccx":
        qc.ccx(*qargs)
    else:
        qc.append(UnitaryGate(_FIXED_NONSYMMETRIC_3Q), list(qargs))
    dag = circuit_to_dag(qc)
    node = dag.op_nodes()[0]

    out = copy.deepcopy(State(length, initial="zeros").mps)
    params = DigitalSimParams(
        get_state=True,
        gate_mode=gate_mode,
        preset="exact",
        svd_threshold=1e-12,
        tdvp_sweeps=1,
    )

    expect_tdvp = gate_mode in {"tdvp", "full-tdvp"} and gate_kind == "ccx"
    with (
        patch("mqt.yaqs.digital.digital_tjm.apply_two_qubit_gate_tebd") as mock_tebd,
        patch("mqt.yaqs.digital.digital_tjm.apply_two_qubit_gate_tdvp") as mock_tdvp,
        patch("mqt.yaqs.digital.digital_tjm.apply_long_range_gate_mpo") as mock_mpo,
    ):
        mock_tdvp.return_value = (min(qargs), max(qargs))
        mock_mpo.return_value = (min(qargs), max(qargs))
        apply_two_qubit_gate(out, node, params)
        if expect_tdvp:
            mock_tdvp.assert_called_once()
            mock_mpo.assert_not_called()
        else:
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


def _four_qubit_circuit(register_sizes: tuple[int, ...] | None) -> QuantumCircuit:
    """Return an empty four-qubit circuit with the given register sizes, or without registers for ``None``."""
    if register_sizes is None:
        return QuantumCircuit([Qubit() for _ in range(4)])
    return QuantumCircuit(*(QuantumRegister(size, f"r{position}") for position, size in enumerate(register_sizes)))


@pytest.mark.parametrize("register_sizes", [(1, 3), (2, 2), None], ids=["1+3", "2+2", "register-less"])
def test_check_longest_gate_counts_circuit_positions_across_registers(register_sizes: tuple[int, ...] | None) -> None:
    """A gate between the first and last qubit spans four sites, whatever the registers."""
    qc = _four_qubit_circuit(register_sizes)
    qc.cx(qc.qubits[0], qc.qubits[3])

    assert check_longest_gate(circuit_to_dag(qc)) == 4


@pytest.mark.parametrize("register_sizes", [(1, 3), None], ids=["1+3", "register-less"])
def test_select_starting_point_follows_circuit_positions_across_registers(
    register_sizes: tuple[int, ...] | None,
) -> None:
    """The starting parity follows the circuit position of the first gate qubit, whatever the registers."""
    qc = _four_qubit_circuit(register_sizes)
    qc.cx(qc.qubits[1], qc.qubits[2])

    first_iter, second_iter = select_starting_point(4, circuit_to_dag(qc))

    assert first_iter == range(1, 3, 2), "Expected odd qubits first."
    assert second_iter == range(0, 3, 2), "Then the even qubit pairs."


def _full_pauli_label(num_qubits: int, qubits: list[int], label: str) -> str:
    """Return the register-wide Qiskit label of a gate ``label`` placed on ``qubits``."""
    full = ["I"] * num_qubits
    for factor, single in enumerate(reversed(label)):
        full[qubits[factor]] = single
    return "".join(reversed(full))


def _dense_pauli_rotation(
    num_qubits: int, qubits: list[int], labels: list[str], coeffs: list[float], time: float
) -> np.ndarray:
    """Return ``exp(-i time sum_k coeffs_k P_k)`` on the whole register, site zero least significant."""
    generator = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
    for label, coeff in zip(labels, coeffs, strict=True):
        generator += coeff * SparsePauliOp([_full_pauli_label(num_qubits, qubits, label)]).to_matrix()
    return expm(-1j * time * generator)


def test_wide_pauli_rotation_builds_its_mpo_without_a_matrix() -> None:
    """A Pauli-string rotation on more than eight qubits is built from its two strings."""
    qc = QuantumCircuit(12)
    qc.append(PauliEvolutionGate(SparsePauliOp(["X" + "Z" * 10 + "X"]), time=0.37), range(12))

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert gate.interaction == 12
    assert gate.sites == list(range(12))
    assert not hasattr(gate, "matrix")
    assert not hasattr(gate, "generator")
    # cos(t) I - i sin(t) P are two product strings, so every cut of the support carries two.
    assert [tensor.shape[3] for tensor in gate.mpo_tensors[:-1]] == [2] * 11


def test_wide_pauli_string_builds_its_mpo_without_a_matrix() -> None:
    """A Pauli string on more than eight qubits is one product string, so every bond is one."""
    qc = QuantumCircuit(12)
    qc.append(PauliGate("XYZXYZXYZXYZ"), range(12))

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert not hasattr(gate, "matrix")
    assert [tensor.shape[3] for tensor in gate.mpo_tensors[:-1]] == [1] * 11


@pytest.mark.parametrize(
    ("num_qubits", "qubits", "labels", "coeffs", "time", "bonds"),
    [
        pytest.param(10, list(range(10)), ["X" + "Z" * 8 + "X"], [1.0], 0.37, [2] * 9, id="one-term"),
        pytest.param(
            10,
            [8, 1, 6, 3, 0, 9, 2, 7, 4],
            ["XYZXZYXZZ"],
            [1.0],
            0.31,
            None,
            id="unsorted-with-a-spectator",
        ),
        pytest.param(
            10,
            list(range(10)),
            ["X" + "Z" * 8 + "X", "Y" + "Z" * 8 + "Y"],
            [0.3, -0.2],
            0.25,
            [4] * 9,
            id="jordan-wigner-hopping",
        ),
        pytest.param(
            10,
            list(range(10)),
            ["ZZIIIIIIII", "IIZZIIIIII", "ZZZZIIIIII"],
            [0.4, 0.5, 0.6],
            0.2,
            None,
            id="three-commuting-terms",
        ),
        pytest.param(9, list(range(9)), ["Z" * 9], [1.0], 0.0, [1] * 8, id="zero-angle"),
        pytest.param(9, list(range(9)), ["Z" * 9], [1.0], np.pi / 2, [1] * 8, id="quarter-turn"),
        pytest.param(9, list(range(9)), ["XIIIIIIIX"], [1.0], 0.4, [2] * 8, id="support-at-the-ends"),
    ],
)
def test_wide_pauli_rotation_matches_the_dense_gate(
    num_qubits: int,
    qubits: list[int],
    labels: list[str],
    coeffs: list[float],
    time: float,
    bonds: list[int] | None,
) -> None:
    """A wide Pauli rotation equals its dense definition for any support, order and angle."""
    qc = QuantumCircuit(num_qubits)
    qc.append(PauliEvolutionGate(SparsePauliOp(labels, coeffs=np.asarray(coeffs)), time=time), qubits)

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    expected = _dense_pauli_rotation(num_qubits, qubits, labels, coeffs, time)
    assert_allclose(MPO.from_gate(gate, num_qubits).to_matrix(), expected, atol=1e-10)
    if bonds is not None:
        assert [tensor.shape[3] for tensor in gate.mpo_tensors[:-1]] == bonds


def test_wide_pauli_string_matches_the_dense_gate() -> None:
    """A wide Pauli string equals its dense definition on unsorted qubits."""
    qubits = [8, 1, 6, 3, 0, 9, 2, 7, 4]
    qc = QuantumCircuit(10)
    qc.append(PauliGate("XYZXYZXYZ"), qubits)

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    expected = SparsePauliOp([_full_pauli_label(10, qubits, "XYZXYZXYZ")]).to_matrix()
    assert_allclose(MPO.from_gate(gate, 10).to_matrix(), expected, atol=1e-10)


def test_wide_pauli_rotation_keeps_its_bond_as_the_span_grows() -> None:
    """The bond of a Pauli rotation follows its term count, not the distance it reaches over."""
    bonds = []
    for span in (12, 24, 40):
        qc = QuantumCircuit(span)
        qc.append(PauliEvolutionGate(SparsePauliOp(["X" + "Z" * (span - 2) + "X"]), time=0.3), range(span))
        (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))
        bonds.append(max(tensor.shape[3] for tensor in gate.mpo_tensors[:-1]))

    assert bonds == [2, 2, 2]


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(SparsePauliOp(["XXIIIIIIII", "ZIIIIIIIIZ"]), id="non-commuting-terms"),
        pytest.param(
            SparsePauliOp(["Z" * 10] * 40 + ["IIIIIIIIIX"], coeffs=np.full(41, 0.1)),
            id="one-non-commuting-term-among-many",
        ),
        pytest.param(
            SparsePauliOp([("I" * i + "ZZ" + "I" * (8 - i)) for i in range(7)], coeffs=np.full(7, 0.3)),
            id="too-many-terms",
        ),
        pytest.param(
            SparsePauliOp(PauliList(["-iX" + "Y" * 9]), np.array([1.0]), ignore_pauli_phase=True),
            id="phase-on-the-label",
        ),
    ],
)
def test_wide_pauli_rotation_outside_the_closed_form_falls_back(operator: SparsePauliOp) -> None:
    """A rotation that is not a short commuting Pauli sum keeps the fallback limit."""
    qc = QuantumCircuit(10)
    qc.append(PauliEvolutionGate(operator, time=0.3), range(10))

    with pytest.raises(ValueError, match="matrix fallback supports at most"):
        convert_dag_to_tensor_algorithm(circuit_to_dag(qc))


def test_wide_pauli_rotation_with_many_repeated_terms_matches_the_dense_gate() -> None:
    """Thousands of repeated terms are checked for commutation at once and add up to one rotation."""
    qc = QuantumCircuit(10)
    qc.append(PauliEvolutionGate(SparsePauliOp(["Z" * 10] * 2000, coeffs=np.full(2000, 1e-4)), time=0.3), range(10))

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    expected = _dense_pauli_rotation(10, list(range(10)), ["Z" * 10], [0.2], 0.3)
    assert_allclose(MPO.from_gate(gate, 10).to_matrix(), expected, atol=1e-10)


def test_wide_pauli_rotation_over_several_operators_falls_back() -> None:
    """A Pauli evolution carrying a list of operators is not one Pauli sum, so it falls back."""
    qc = QuantumCircuit(10)
    operators = [SparsePauliOp(["Z" * 10]), SparsePauliOp(["X" * 10])]
    qc.append(PauliEvolutionGate(operators, time=0.3), range(10))

    with pytest.raises(ValueError, match="matrix fallback supports at most"):
        convert_dag_to_tensor_algorithm(circuit_to_dag(qc))


def test_wide_pauli_rotation_survives_pickling() -> None:
    """A wide Pauli rotation keeps its MPO through pickling, as trajectory workers receive it."""
    qc = QuantumCircuit(10)
    qc.append(PauliEvolutionGate(SparsePauliOp(["X" + "Z" * 8 + "X"]), time=0.3), range(10))
    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    restored = pickle.loads(pickle.dumps(gate))  # ruff: ignore[suspicious-pickle-usage]  # controlled round-trip

    assert restored.sites == gate.sites
    for original, copied in zip(gate.mpo_tensors, restored.mpo_tensors, strict=True):
        assert_allclose(copied, original, atol=0.0)


def test_wide_multi_target_gate_builds_its_mpo_without_a_matrix() -> None:
    """A controlled gate with more than eight targets is built from its one-qubit factors."""
    qc = QuantumCircuit(12)
    qc.append(MCMTGate(XGate(), 3, 9), range(12))

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert gate.interaction == 12
    assert not hasattr(gate, "matrix")
    assert not hasattr(gate, "generator")
    # The difference of a product telescopes into two states, so the channel carries bond two.
    assert [tensor.shape[3] for tensor in gate.mpo_tensors[:-1]] == [2] * 11


@pytest.mark.parametrize(
    ("num_qubits", "gate", "qubits", "bonds"),
    [
        pytest.param(9, MCMTGate(XGate(), 3, 6), list(range(9)), [2] * 8, id="fan-out-contiguous"),
        pytest.param(
            9,
            MCMTGate(XGate(), 3, 6),
            [0, 4, 8, 1, 2, 3, 5, 6, 7],
            [2, 3, 3, 3, 3, 3, 3, 2],
            id="fan-out-with-controls-spread-out",
        ),
        pytest.param(
            11,
            MCMTGate(XGate(), 3, 6),
            [0, 1, 2, 4, 5, 6, 8, 9, 10],
            [2] * 10,
            id="fan-out-with-spectators",
        ),
        pytest.param(9, MCMTGate(RYGate(0.3), 2, 7), list(range(9)), [2] * 8, id="fan-out-of-a-rotation"),
        pytest.param(9, MCMTGate(XGate(), 7, 2), list(range(9)), [2] * 8, id="several-targets-of-one-base"),
        pytest.param(9, MCMTGate(PauliGate("X"), 3, 6), list(range(9)), [2] * 8, id="fan-out-of-a-pauli-gate"),
        pytest.param(
            9, MCMTGate(IGate(), 3, 6), [0, 4, 8, 1, 2, 3, 5, 6, 7], [1] * 8, id="identity-target-controls-spread-out"
        ),
        pytest.param(
            9,
            MCMTGate(UnitaryGate(np.exp(0.3j) * np.eye(2)), 3, 6),
            [0, 4, 8, 1, 2, 3, 5, 6, 7],
            [2] * 8,
            id="phase-target-controls-spread-out",
        ),
        pytest.param(5, MCMTGate(RZGate(2 * np.pi), 2, 3), list(range(5)), [2, 1, 1, 1], id="phase-up-to-rounding"),
        pytest.param(
            11,
            PauliGate("IIIIIIIIX").control(2),
            [0, 10, *range(1, 10)],
            [2] * 10,
            id="identity-letters-around-one-factor",
        ),
        pytest.param(
            11,
            PauliGate("IIXYZXYII").control(2),
            [0, 10, *range(1, 10)],
            [2, 2, 2, 3, 3, 3, 3, 2, 2, 2],
            id="identity-letters-at-the-edges",
        ),
        pytest.param(
            10,
            PauliGate("XYZXYZX").control(3),
            list(range(10)),
            [2] * 9,
            id="controlled-pauli-string",
        ),
        pytest.param(
            10,
            PauliGate("XYZXYZX").control(3, ctrl_state=0b010),
            [9, 0, 5, 1, 2, 3, 4, 6, 7, 8],
            [2, 3, 3, 3, 3, 3, 3, 3, 2],
            id="controlled-pauli-string-unsorted",
        ),
    ],
)
def test_wide_multi_target_gate_mpo_has_the_minimal_bond(
    num_qubits: int, gate: Gate, qubits: list[int], bonds: list[int]
) -> None:
    """A controlled product target carries the minimal bond, also when some or all of its factors are phases."""
    qc = QuantumCircuit(num_qubits)
    qc.append(gate, qubits)

    (yaqs_gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert [tensor.shape[3] for tensor in yaqs_gate.mpo_tensors[:-1]] == bonds


@pytest.mark.parametrize(
    ("num_qubits", "gate", "qubits"),
    [
        pytest.param(9, MCMTGate(XGate(), 3, 6), list(range(9)), id="fan-out-of-x"),
        pytest.param(9, MCMTGate(HGate(), 3, 6), list(range(9)), id="fan-out-of-h"),
        pytest.param(9, MCMTGate(RYGate(0.3), 2, 7), list(range(9)), id="fan-out-of-a-rotation"),
        pytest.param(9, MCMTGate(XGate(), 7, 2), list(range(9)), id="two-targets-of-one-base"),
        pytest.param(9, MCMTGate(IGate(), 3, 6), list(range(9)), id="identity-target"),
        pytest.param(9, MCMTGate(PauliGate("Y"), 3, 6), list(range(9)), id="fan-out-of-a-pauli-gate"),
        pytest.param(
            9,
            MCMTGate(UnitaryGate(np.exp(0.3j) * np.eye(2)), 3, 6),
            [0, 4, 8, 1, 2, 3, 5, 6, 7],
            id="phase-target-controls-spread-out",
        ),
        pytest.param(11, PauliGate("IIXYZXYII").control(2), [0, 10, *range(1, 10)], id="identity-letters-at-the-edges"),
        pytest.param(10, PauliGate("XYZXYZXYZ").control(1), list(range(10)), id="one-control-nine-targets"),
        pytest.param(
            10,
            PauliGate("XYZXYZX").control(3, ctrl_state=0b010),
            [9, 0, 5, 1, 2, 3, 4, 6, 7, 8],
            id="controlled-pauli-string-unsorted",
        ),
    ],
)
def test_wide_multi_target_gate_matches_the_dense_gate(num_qubits: int, gate: Gate, qubits: list[int]) -> None:
    """A controlled product target equals its definition for any base, order and control pattern."""
    qc = QuantumCircuit(num_qubits)
    qc.append(gate, qubits)

    (yaqs_gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert_allclose(MPO.from_gate(yaqs_gate, num_qubits).to_matrix(), Operator(qc).data, atol=1e-12)


def test_wide_multi_target_gate_survives_pickling() -> None:
    """A wide multi-target gate keeps its MPO through pickling, as trajectory workers receive it."""
    qc = QuantumCircuit(12)
    qc.append(MCMTGate(XGate(), 3, 9), range(12))
    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    restored = pickle.loads(pickle.dumps(gate))  # ruff: ignore[suspicious-pickle-usage]  # controlled round-trip

    assert restored.sites == gate.sites
    for original, copied in zip(gate.mpo_tensors, restored.mpo_tensors, strict=True):
        assert_allclose(copied, original, atol=0.0)


@pytest.mark.parametrize(
    ("gate", "qubits", "bonds"),
    [
        pytest.param(MCXGate(3), [0, 1, 2, 3], [2, 2, 2], id="mcx-on-four-qubits"),
        pytest.param(MCXGate(7, ctrl_state=0b1011001), [0, 1, 2, 3, 5, 6, 7, 4], [2] * 7, id="mcx-target-inside"),
        pytest.param(ZGate().control(7, annotated=True), list(range(8)), [2] * 7, id="annotated-z-on-eight-qubits"),
        # Open controls give names outside the gate library, such as ``ccx_o1`` and ``cswap_o0``.
        pytest.param(CCXGate(ctrl_state=1), [0, 1, 2], [2, 2], id="toffoli-with-an-open-control"),
        pytest.param(CSwapGate(ctrl_state=0), [0, 1, 2], [2, 4], id="fredkin-with-an-open-control"),
        pytest.param(SwapGate().control(2, annotated=False), [0, 1, 2, 3], [2, 2, 4], id="swap-with-two-controls"),
        pytest.param(MCMTGate(XGate(), 3, 5), list(range(8)), [2] * 7, id="multi-target-on-eight-qubits"),
        pytest.param(MCMTGate(XGate(), 1, 2), [1, 0, 2], [2, 2], id="multi-target-control-between"),
        pytest.param(
            PauliGate("XYZ").control(2, annotated=False), list(range(5)), [2] * 4, id="controlled-pauli-string"
        ),
        pytest.param(PauliGate("XYZXYZXY"), list(range(8)), [1] * 7, id="pauli-string"),
    ],
)
def test_gate_from_three_qubits_is_built_from_its_parts(gate: Gate, qubits: list[int], bonds: list[int]) -> None:
    """From three qubits on, a gate whose parts Qiskit states is built from them at the minimal bond."""
    qc = QuantumCircuit(len(qubits))
    qc.append(gate, qubits)

    (yaqs_gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert not hasattr(yaqs_gate, "matrix")
    assert_allclose(MPO.from_gate(yaqs_gate, len(qubits)).to_matrix(), Operator(qc).data, atol=1e-12)
    assert [tensor.shape[3] for tensor in yaqs_gate.mpo_tensors[:-1]] == bonds


@pytest.mark.parametrize(
    ("labels", "coeffs", "bonds"),
    [
        pytest.param(["XZZX"], [1.0], [2, 2, 2], id="one-string"),
        pytest.param(["IXZXI"], [1.0], [1, 2, 2, 1], id="identity-letters-at-the-ends"),
        pytest.param(["XZZZX", "YZZZY"], [0.3, -0.2], [4] * 4, id="jordan-wigner-hopping"),
    ],
)
def test_pauli_rotation_from_three_qubits_is_built_from_its_terms(
    labels: list[str], coeffs: list[float], bonds: list[int]
) -> None:
    """A Pauli rotation on three to eight qubits is built from its strings, never from Qiskit's matrix."""
    num_qubits = len(labels[0])
    qc = QuantumCircuit(num_qubits)
    qc.append(PauliEvolutionGate(SparsePauliOp(labels, coeffs=np.asarray(coeffs)), time=0.3), range(num_qubits))

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert not hasattr(gate, "matrix")
    expected = _dense_pauli_rotation(num_qubits, list(range(num_qubits)), labels, coeffs, 0.3)
    assert_allclose(MPO.from_gate(gate, num_qubits).to_matrix(), expected, atol=1e-12)
    assert [tensor.shape[3] for tensor in gate.mpo_tensors[:-1]] == bonds


@pytest.mark.parametrize(
    "gate",
    [
        pytest.param(CUGate(0.1, 0.2, 0.3, 0.4).control(1, annotated=False), id="base-without-matrix"),
        pytest.param(GlobalPhaseGate(0.3).control(3, annotated=False), id="no-target"),
        pytest.param(AnnotatedOperation(CCXGate(), [ControlModifier(0)]), id="no-control"),
    ],
)
def test_gate_from_three_qubits_outside_the_builders_keeps_its_matrix(gate: Operation) -> None:
    """A gate on three to eight qubits that no builder takes keeps the dense fallback."""
    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, range(gate.num_qubits))

    (yaqs_gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert hasattr(yaqs_gate, "matrix")
    assert_allclose(MPO.from_gate(yaqs_gate, gate.num_qubits).to_matrix(), Operator(qc).data, atol=1e-12)


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_non_commuting_pauli_rotation_from_three_qubits_keeps_its_matrix() -> None:
    """A Pauli rotation whose terms do not commute keeps the dense fallback up to eight qubits.

    The reference is the gate's own dense matrix rather than the exact exponential of the generator,
    because for non-commuting terms the two differ by the Trotter error of the gate's definition on
    the Qiskit versions that give no matrix for ``PauliEvolutionGate``.
    """
    qc = QuantumCircuit(4)
    qc.append(PauliEvolutionGate(SparsePauliOp(["XXII", "ZIIZ"]), time=0.3), range(4))

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    assert hasattr(gate, "matrix")
    assert_allclose(MPO.from_gate(gate, 4).to_matrix(), Operator(qc).data, atol=1e-12)


def test_pauli_rotation_keeps_a_small_angle_exactly() -> None:
    """At a small angle the hopping term keeps its weak string, so the MPO equals the gate to round-off."""
    labels = ["XZZZZX", "YZZZZY"]
    qc = QuantumCircuit(6)
    qc.append(PauliEvolutionGate(SparsePauliOp(labels), time=1e-7), range(6))

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    expected = _dense_pauli_rotation(6, list(range(6)), labels, [1.0, 1.0], 1e-7)
    assert_allclose(MPO.from_gate(gate, 6).to_matrix(), expected, atol=1e-15)
    assert [tensor.shape[3] for tensor in gate.mpo_tensors[:-1]] == [4] * 5


def test_commuting_rotations_that_factorize_have_bond_one_between_them() -> None:
    """Rotations on disjoint qubits multiply to a product, so the cuts between them carry bond one."""
    labels = ["IIIIZZ", "ZZIIII"]
    qc = QuantumCircuit(6)
    qc.append(PauliEvolutionGate(SparsePauliOp(labels, coeffs=np.array([0.4, 0.7])), time=1.0), range(6))

    (gate,) = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))

    expected = _dense_pauli_rotation(6, list(range(6)), labels, [0.4, 0.7], 1.0)
    assert_allclose(MPO.from_gate(gate, 6).to_matrix(), expected, atol=1e-12)
    assert [tensor.shape[3] for tensor in gate.mpo_tensors[:-1]] == [2, 1, 1, 1, 2]


def test_controlled_gate_binds_sites_given_as_lists() -> None:
    """``set_sites`` accepts a list of indices as well as single indices."""
    gate = _ControlledGate(np.eye(2, dtype=np.complex128), [1, 1])

    gate.set_sites([0, 1], 2)

    assert gate.sites == [0, 1, 2]
    assert len(gate.mpo_tensors) == 3


def test_controlled_gate_rejects_a_site_count_below_its_interaction() -> None:
    """A gate bound to fewer or more sites than it acts on must raise."""
    gate = _ControlledGate(np.eye(2, dtype=np.complex128), [1])

    with pytest.raises(ValueError, match="Number of sites 3 must be equal to the interaction level 2"):
        gate.set_sites(0, 1, 2)


def test_pauli_sum_gate_binds_sites_given_as_lists() -> None:
    """``set_sites`` of the Pauli-sum gate accepts a list of indices as well as single indices."""
    gate = _PauliSumGate([(1.0 + 0j, {0: "X", 2: "Z"})], 3)

    gate.set_sites([0, 1, 2])

    assert gate.sites == [0, 1, 2]
    assert len(gate.mpo_tensors) == 3


def test_pauli_sum_gate_rejects_a_site_count_below_its_interaction() -> None:
    """A Pauli-sum gate bound to fewer or more sites than it acts on must raise."""
    gate = _PauliSumGate([(1.0 + 0j, {0: "X"})], 2)

    with pytest.raises(ValueError, match="Number of sites 3 must be equal to the interaction level 2"):
        gate.set_sites(0, 1, 2)


def test_pauli_string_that_does_not_span_its_sites_falls_back() -> None:
    """A Pauli label that does not cover every site is not built from the label."""
    assert _translate_pauli_string(PauliGate("XYZ"), name="pauli", sites=[0, 1]) is None


def test_pauli_evolution_whose_time_is_not_a_number_falls_back() -> None:
    """An evolution time that does not convert to a float is not built from the terms."""
    gate = PauliEvolutionGate(SparsePauliOp(["XZZX"]), time=Parameter("t"))

    assert _translate_pauli_evolution(gate, name="pauli-evolution", sites=[0, 1, 2, 3]) is None


def test_target_without_a_matrix_is_not_a_product_of_factors() -> None:
    """A target the gate gives no matrix for cannot be split into one-qubit factors."""
    assert _target_factors(Gate("opaque", 1, []), 2, name="opaque") is None


def test_non_unitary_target_is_not_a_product_of_factors() -> None:
    """A target that is not unitary cannot be split into one-qubit factors."""
    base = UnitaryGate(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128), check_input=False)

    assert _target_factors(base, 2, name="nonunitary") is None
