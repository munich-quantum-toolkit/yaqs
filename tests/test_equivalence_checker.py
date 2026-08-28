# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for circuit equivalence checking.

This module provides unit tests for :class:`~mqt.yaqs.EquivalenceChecker`. It verifies
the MPO and dense matrix backends by comparing quantum circuits, including automatic
backend selection, global-phase equivalence, and regression coverage for QASM custom gates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ECRGate, U1Gate, U3Gate
from qiskit.converters import circuit_to_dag
from qiskit.qasm2 import load, loads
from qiskit.quantum_info import Operator

from mqt.yaqs import EquivalenceChecker, NoiseModel
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.utils import matrix_utils
from mqt.yaqs.digital.utils.contraction_utils import MIN_QUBITS_FOR_MPO_PARALLEL
from mqt.yaqs.digital.utils.dag_utils import (
    SUPPORTED_QISKIT_GATE_NAMES,
    convert_dag_to_tensor_algorithm,
)
from mqt.yaqs.equivalence_checker import DEFAULT_MATRIX_MAX_QUBITS
from tests.conftest import LARGE_QASM2_STRING, SAMPLE_QASM3_STRING, requires_qasm3_import, write_qasm_file

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.yaqs.equivalence_checker import Representation


@pytest.mark.parametrize(("threshold", "fidelity"), [(1e-13, 1 - 1e-13), (1e-1, 1 - 1e-3)])
def test_identity_vs_identity(threshold: float, fidelity: float) -> None:
    """Test that two empty (no-gate) circuits on the same number of qubits are equivalent.

    This test creates two quantum circuits with no gates (which effectively implement the identity)
    on 2 qubits, and then checks that the MPO-based equivalence algorithm returns True and that
    the elapsed time is non-negative.

    Args:
        threshold (float): The SVD truncation threshold to be used.
        fidelity (float): The fidelity threshold for determining equivalence.
    """
    num_qubits = 2
    qc1 = QuantumCircuit(num_qubits)
    qc2 = QuantumCircuit(num_qubits)

    checker = EquivalenceChecker(threshold=threshold, fidelity=fidelity)
    result = checker.check(qc1, qc2)
    assert result["equivalent"] is True, "Empty circuits (identities) should be equivalent."
    assert float(result["elapsed_time"]) >= 0


def test_two_qubit_equivalence() -> None:
    """Test that two-qubit circuits implementing the same logical operation are equivalent.

    This test creates two circuits that prepare the same Bell state using H and CX gates
    on a 2-qubit system, and verifies that the equivalence check returns True.
    """
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)

    checker = EquivalenceChecker(threshold=1e-13, fidelity=1 - 1e-13)
    result = checker.check(qc1, qc2)
    assert result["equivalent"] is True, "Identical 2-qubit circuits must be equivalent."


def test_two_qubit_non_equivalence() -> None:
    """Test that two-qubit circuits differing by an extra gate are not equivalent.

    This test creates two circuits on 2 qubits where the second circuit has an extra X gate applied
    after the entangling operation. The equivalence check should return False.
    """
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.x(1)  # An extra gate after entangling

    checker = EquivalenceChecker(threshold=1e-13, fidelity=1 - 1e-13)
    result = checker.check(qc1, qc2)
    assert result["equivalent"] is False, "Extra gate should break equivalence."


def test_long_range_equivalence() -> None:
    """Test that long-range circuits implementing the same operation are equivalent.

    This test creates two 3-qubit circuits with an identical long-range CX gate (acting between qubits 0 and 2)
    and verifies that the equivalence check returns True.
    """
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.cx(0, 2)

    qc2 = QuantumCircuit(3)
    qc2.h(0)
    qc2.cx(0, 2)

    checker = EquivalenceChecker(threshold=1e-13, fidelity=1 - 1e-13)
    result = checker.check(qc1, qc2)
    assert result["equivalent"] is True, "Long-range circuits with identical operations must be equivalent."


def test_long_range_non_equivalence() -> None:
    """Test that long-range circuits differing by an extra gate are not equivalent.

    This test creates two 3-qubit circuits where the second circuit has an extra X gate after the long-range
    CX gate. The equivalence check should return False.
    """
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.cx(0, 2)

    qc2 = QuantumCircuit(3)
    qc2.h(0)
    qc2.cx(0, 2)
    qc2.x(1)  # An extra gate after entangling

    checker = EquivalenceChecker(threshold=1e-13, fidelity=1 - 1e-13)
    result = checker.check(qc1, qc2)
    assert result["equivalent"] is False, "Extra gate should break equivalence."


def test_large_equivalence(tmp_path: Path) -> None:
    """Test large-scale equivalence.

    This test creates a large quantum circuit with multiple CNOT gates, Ry gates, and an Rzz gate.
    This should verify nearly all parts of the equivalence checking algorithm.
    """
    qasm_path = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    qc = load(filename=str(qasm_path))

    checker = EquivalenceChecker(representation="mpo")
    result = checker.check(qc, qc)
    assert result["equivalent"] is True, "Large scale test fails. Circuits should be equivalent."
    assert result["representation"] == "mpo"


ISSUE_QASM_WITH_MEASURES = """
OPENQASM 2.0;
include "qelib1.inc";

gate bellprep a,b {
  h a;
  cx a,b;
}

gate phase_kick(theta) q {
  rz(theta) q;
  x q;
  rz(-theta) q;
  x q;
}

qreg q[3];
creg c[3];

bellprep q[0], q[1];
phase_kick(pi/4) q[2];

cx q[1], q[2];
h q[0];

measure q -> c;
"""

ISSUE_QASM_CUSTOM = """
OPENQASM 2.0;
include "qelib1.inc";

gate bellprep a,b {
  h a;
  cx a,b;
}

gate phase_kick(theta) q {
  rz(theta) q;
  x q;
  rz(-theta) q;
  x q;
}

qreg q[3];

bellprep q[0], q[1];
phase_kick(pi/4) q[2];
cx q[1], q[2];
h q[0];
"""

ISSUE_QASM_EXPANDED = """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];

h q[0];
cx q[0], q[1];
rz(pi/4) q[2];
x q[2];
rz(-pi/4) q[2];
x q[2];
cx q[1], q[2];
h q[0];
"""


def _issue_checker(*, representation: Literal["mpo", "matrix", "auto"] = "mpo") -> EquivalenceChecker:
    """Return an equivalence checker configured for issue regression tests."""
    return EquivalenceChecker(
        threshold=1e-13,
        fidelity=1 - 1e-13,
        representation=representation,
    )


def test_issue_qasm_self_equivalence_with_final_measurements() -> None:
    """The exact issue QASM circuit with custom gates and final measurements is self-equivalent."""
    qc = loads(ISSUE_QASM_WITH_MEASURES)
    result = _issue_checker(representation="mpo").check(qc, qc)
    assert result["equivalent"] is True
    assert result["representation"] == "mpo"


def test_issue_qasm_custom_vs_expanded_equivalence() -> None:
    """QASM custom gates should be equivalent to their manually expanded decomposition."""
    qc_custom = loads(ISSUE_QASM_CUSTOM)
    qc_expanded = loads(ISSUE_QASM_EXPANDED)
    result = _issue_checker(representation="mpo").check(qc_custom, qc_expanded)
    assert result["equivalent"] is True
    assert result["representation"] == "mpo"


@pytest.mark.parametrize("gate_name", ["u1", "u3", "ecr"])
def test_u1_u3_ecr_self_equivalence(gate_name: str) -> None:
    """Legacy Qiskit gate names from the issue should self-equivalence-check on the MPO path."""
    if gate_name == "u1":
        qc = QuantumCircuit(2)
        qc.append(U1Gate(0.37), [0])
    elif gate_name == "u3":
        qc = QuantumCircuit(2)
        qc.append(U3Gate(0.2, 0.3, 0.4), [0])
    else:
        qc = QuantumCircuit(2)
        qc.append(ECRGate(), [0, 1])

    result = _issue_checker(representation="mpo").check(qc, qc)
    assert result["equivalent"] is True
    assert result["representation"] == "mpo"


def test_ecr_has_no_hardcoded_gate_library_path() -> None:
    """``ecr`` must not use a hardcoded GateLibrary entry and should translate via matrix fallback."""
    assert "ecr" not in SUPPORTED_QISKIT_GATE_NAMES
    assert not hasattr(GateLibrary, "ecr")

    qc = QuantumCircuit(2)
    qc.append(ECRGate(), [0, 1])
    gates = convert_dag_to_tensor_algorithm(circuit_to_dag(qc))
    assert len(gates) == 1
    assert gates[0].name == "ecr"


def test_equivalence_checker_rejects_mid_circuit_measurements() -> None:
    """Mid-circuit measurements must be rejected clearly by the equivalence checker."""
    qc1 = QuantumCircuit(2, 1)
    qc1.x(0)
    qc1.measure(0, 0)
    qc1.x(0)

    qc2 = QuantumCircuit(2, 1)
    qc2.x(0)
    qc2.measure(0, 0)
    qc2.x(0)

    with pytest.raises(ValueError, match="Mid-circuit measurements are not supported"):
        _issue_checker(representation="mpo").check(qc1, qc2)
    with pytest.raises(ValueError, match="Mid-circuit measurements are not supported"):
        _issue_checker(representation="matrix").check(qc1, qc2)


def test_matrix_backend_descending_cx_equivalence() -> None:
    """The matrix backend accepts the H-conjugation identity for a descending cx.

    ``(H ⊗ H) · cx(1, 2) · (H ⊗ H)`` equals ``cx(2, 1)``; both backends must agree.
    """
    qa = QuantumCircuit(3)
    qa.cx(2, 1)

    qb = QuantumCircuit(3)
    qb.h(1)
    qb.h(2)
    qb.cx(1, 2)
    qb.h(1)
    qb.h(2)

    assert EquivalenceChecker(representation="matrix").check(qa, qb)["equivalent"] is True
    assert EquivalenceChecker(representation="mpo").check(qa, qb)["equivalent"] is True


def test_equivalence_checker_matrix_backend_strips_measurements_once() -> None:
    """The matrix backend should strip final measurements only inside ``compose_operator_tensor``."""
    qc1 = QuantumCircuit(1, 1)
    qc1.x(0)
    qc1.measure(0, 0)
    qc2 = qc1.copy()

    with patch.object(matrix_utils, "strip_final_measurements", wraps=matrix_utils.strip_final_measurements) as strip:
        result = _issue_checker(representation="matrix").check(qc1, qc2)

    assert result["equivalent"] is True
    assert strip.call_count == 2


@pytest.mark.parametrize("representation", ["matrix", "mpo"])
def test_matrix_and_mpo_agree_on_small_circuits(representation: Literal["matrix", "mpo"]) -> None:
    """Matrix and MPO backends agree on equivalent and non-equivalent small circuits."""
    qc_equal_a = QuantumCircuit(2)
    qc_equal_a.h(0)
    qc_equal_a.cx(0, 1)
    qc_equal_b = qc_equal_a.copy()

    qc_diff_b = QuantumCircuit(2)
    qc_diff_b.h(0)
    qc_diff_b.cx(0, 1)
    qc_diff_b.x(1)

    checker = EquivalenceChecker(
        threshold=1e-13,
        fidelity=1 - 1e-13,
        representation=cast("Representation", representation),
    )
    equal_result = checker.check(qc_equal_a, qc_equal_b)
    diff_result = checker.check(qc_equal_a, qc_diff_b)
    assert equal_result["equivalent"] is True
    assert diff_result["equivalent"] is False
    assert equal_result["representation"] == representation


def test_matrix_and_mpo_return_same_relative_operator_orientation() -> None:
    """Both backends construct the documented ``U1 U2†`` relative operator."""
    circuit1 = QuantumCircuit(3)
    circuit1.h(0)
    circuit1.cx(0, 1)
    circuit1.ry(0.37, 2)
    circuit1.cx(1, 2)

    circuit2 = QuantumCircuit(3)
    circuit2.rz(0.29, 0)
    circuit2.cx(1, 2)
    circuit2.rx(-0.41, 1)
    circuit2.cx(0, 1)

    unitary1 = np.asarray(Operator(circuit1.reverse_bits()).data, dtype=np.complex128)
    unitary2 = np.asarray(Operator(circuit2.reverse_bits()).data, dtype=np.complex128)
    expected = unitary1 @ unitary2.conj().T
    reverse_order = unitary2.conj().T @ unitary1
    assert not np.allclose(expected, reverse_order, atol=1e-10)

    matrix = EquivalenceChecker(representation="matrix").check(circuit1, circuit2)["matrix"]
    mpo = EquivalenceChecker(representation="mpo", parallel=False).check(circuit1, circuit2)["mpo"]

    assert matrix is not None
    assert mpo is not None
    np.testing.assert_allclose(matrix, expected, atol=1e-12)
    np.testing.assert_allclose(mpo.to_matrix(), expected, atol=1e-12)


@pytest.mark.parametrize("representation", ["matrix", "mpo"])
def test_global_phase_equivalence(representation: str) -> None:
    """Circuits differing by global phase are equivalent on both backends."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = qc1.copy()
    qc2.global_phase = np.pi / 3

    checker = EquivalenceChecker(
        representation=cast("Representation", representation),
        fidelity=1 - 1e-13,
    )
    result = checker.check(qc1, qc2)
    assert result["equivalent"] is True
    assert result["representation"] == representation


def test_mpo_backend_rejects_multi_qubit_gates() -> None:
    """The MPO backend rejects circuits containing gates on more than two qubits."""
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)

    checker = EquivalenceChecker(representation="mpo")
    with pytest.raises(ValueError, match="more than two qubits"):
        checker.check(qc, qc)


def test_matrix_backend_supports_multi_qubit_gates() -> None:
    """The matrix backend checks equivalence of circuits containing three-qubit gates."""
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    decomposed = transpile(qc, basis_gates=["cx", "u"], optimization_level=0)
    assert all(len(instruction.qubits) <= 2 for instruction in decomposed.data)

    checker = EquivalenceChecker(representation="matrix")
    result = checker.check(qc, decomposed)
    assert result["equivalent"] is True
    assert result["representation"] == "matrix"


def test_auto_representation_selects_by_qubit_count() -> None:
    """``representation='auto'`` uses matrix at or below the cutover and MPO above it."""
    small = QuantumCircuit(2)
    large = QuantumCircuit(DEFAULT_MATRIX_MAX_QUBITS + 1)

    auto_small = EquivalenceChecker(representation="auto")
    assert auto_small.check(small, small)["representation"] == "matrix"

    auto_large = EquivalenceChecker(representation="auto")
    assert auto_large.check(large, large)["representation"] == "mpo"


def test_matrix_max_qubits_override() -> None:
    """``matrix_max_qubits`` controls the auto cutover."""
    qc = QuantumCircuit(3)
    checker = EquivalenceChecker(representation="auto", matrix_max_qubits=2)
    assert checker.check(qc, qc)["representation"] == "mpo"

    checker_wide = EquivalenceChecker(representation="auto", matrix_max_qubits=4)
    assert checker_wide.check(qc, qc)["representation"] == "matrix"


@pytest.mark.parametrize("fidelity", [-0.1, 1.1, np.nan, np.inf])
def test_checker_rejects_invalid_fidelity_threshold(fidelity: float) -> None:
    """The root-overlap threshold must be finite and inside the fidelity range."""
    with pytest.raises(ValueError, match="fidelity must be finite and between 0 and 1"):
        EquivalenceChecker(fidelity=fidelity)


@pytest.mark.parametrize("fidelity", [True, "0.9"])
def test_checker_rejects_non_real_fidelity_threshold(fidelity: object) -> None:
    """Booleans and non-real values are not valid fidelity thresholds."""
    with pytest.raises(TypeError, match="fidelity must be a real number"):
        EquivalenceChecker(fidelity=fidelity)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("max_workers", [0, -1])
def test_checker_rejects_non_positive_max_workers(max_workers: int) -> None:
    """``max_workers`` must be positive when provided."""
    with pytest.raises(ValueError, match="positive"):
        EquivalenceChecker(max_workers=max_workers)


def test_checker_rejects_bool_max_workers() -> None:
    """``max_workers=False`` is rejected (booleans are not valid thread caps)."""
    with pytest.raises(TypeError, match="max_workers"):
        EquivalenceChecker(max_workers=False)


def test_checker_rejects_non_int_max_workers() -> None:
    """Non-integer ``max_workers`` values are rejected."""
    with pytest.raises(TypeError, match="max_workers"):
        EquivalenceChecker(max_workers=1.5)  # ty: ignore[invalid-argument-type]


def test_equivalence_checker_defaults_parallel_true() -> None:
    """``parallel`` defaults to ``True`` (MPO thread pool still gated by qubit count)."""
    assert EquivalenceChecker().parallel is True


def _make_n_by_n_circuit(num_qubits: int) -> QuantumCircuit:
    """Build an ``n`` x ``n`` layered circuit (``n`` qubits, ``n`` repetitions).

    Returns:
        A layered circuit with all-qubit ``h`` gates and linear ``cx`` chains.
    """
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_qubits):
        for q in range(num_qubits):
            qc.h(q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    return qc


@pytest.mark.parametrize("parallel", [False, True])
def test_mpo_checker_serial_vs_parallel_small(*, parallel: bool) -> None:
    """MPO equivalence on small circuits (serial path even when parallel=True)."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)

    checker = EquivalenceChecker(representation="mpo", parallel=parallel, max_workers=2)
    result = checker.check(qc1, qc2)
    assert result["equivalent"] is True


@pytest.mark.parametrize("num_qubits", [MIN_QUBITS_FOR_MPO_PARALLEL, MIN_QUBITS_FOR_MPO_PARALLEL + 2])
def test_wide_mpo_serial_vs_parallel_equivalent(num_qubits: int) -> None:
    """Wide ``n`` x ``n`` circuits agree between serial and parallel MPO checking."""
    qc = _make_n_by_n_circuit(num_qubits)
    serial = EquivalenceChecker(representation="mpo", parallel=False, threshold=1e-6).check(qc, qc)
    parallel = EquivalenceChecker(
        representation="mpo",
        parallel=True,
        max_workers=2,
        threshold=1e-6,
    ).check(qc, qc)

    assert serial["equivalent"] is True
    assert parallel["equivalent"] is True
    assert serial["equivalent"] == parallel["equivalent"]


def test_wide_mpo_serial_vs_parallel_non_equivalent() -> None:
    """Serial and parallel MPO paths agree on non-equivalent wide circuits."""
    num_qubits = MIN_QUBITS_FOR_MPO_PARALLEL
    qc1 = _make_n_by_n_circuit(num_qubits)
    qc2 = qc1.copy()
    qc2.x(0)

    serial = EquivalenceChecker(representation="mpo", parallel=False, threshold=1e-6).check(qc1, qc2)
    parallel = EquivalenceChecker(
        representation="mpo",
        parallel=True,
        max_workers=2,
        threshold=1e-6,
    ).check(qc1, qc2)

    assert serial["equivalent"] is False
    assert serial["equivalent"] == parallel["equivalent"]


def test_mpo_parallel_max_workers_one_uses_in_process_path() -> None:
    """``max_workers=1`` still runs through the parallel sweep with a thread pool."""
    num_qubits = MIN_QUBITS_FOR_MPO_PARALLEL
    qc = _make_n_by_n_circuit(num_qubits)
    result = EquivalenceChecker(
        representation="mpo",
        parallel=True,
        max_workers=1,
        threshold=1e-6,
    ).check(qc, qc)
    assert result["equivalent"] is True


def test_long_range_mpo_parallel() -> None:
    """Long-range circuits agree between serial and parallel MPO checking."""
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.cx(0, 2)

    qc2 = qc1.copy()

    serial = EquivalenceChecker(representation="mpo", parallel=False).check(qc1, qc2)
    parallel = EquivalenceChecker(representation="mpo", parallel=True, max_workers=2).check(qc1, qc2)
    assert serial["equivalent"] == parallel["equivalent"]


def test_check_accepts_qasm2_path_object(tmp_path: Path) -> None:
    """Check that a QASM 2 file given as a Path object is accepted and returns equivalent."""
    qasm_path = write_qasm_file(tmp_path, LARGE_QASM2_STRING)

    checker = EquivalenceChecker(representation="mpo")
    result = checker.check(qasm_path, qasm_path)
    assert result["equivalent"] is True


def test_check_accepts_qasm2_str_path(tmp_path: Path) -> None:
    """Check that a QASM 2 file given as a str path is accepted and returns equivalent."""
    qasm_path = str(write_qasm_file(tmp_path, LARGE_QASM2_STRING))

    checker = EquivalenceChecker(representation="mpo")
    result = checker.check(qasm_path, qasm_path)
    assert result["equivalent"] is True


def test_check_qasm_path_vs_quantumcircuit_agree(tmp_path: Path) -> None:
    """Verify that loading via path and via QuantumCircuit gives the same equivalence result."""
    qasm_path = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    qc = load(filename=str(qasm_path))
    checker = EquivalenceChecker(representation="mpo")
    result_path = checker.check(qasm_path, qasm_path)
    result_qc = checker.check(qc, qc)
    assert result_path["equivalent"] == result_qc["equivalent"]


@requires_qasm3_import
def test_check_accepts_qasm3_path_object(tmp_path: Path) -> None:
    """Check that a QASM 3 file given as a Path object is accepted and returns equivalent."""
    qasm_file = write_qasm_file(tmp_path, SAMPLE_QASM3_STRING, filename="circuit3.qasm")

    checker = EquivalenceChecker(representation="matrix")
    result = checker.check(qasm_file, qasm_file)
    assert result["equivalent"] is True


@requires_qasm3_import
def test_check_accepts_qasm3_str_path(tmp_path: Path) -> None:
    """Check that a QASM 3 file given as a str path is accepted and returns equivalent."""
    qasm_file = str(write_qasm_file(tmp_path, SAMPLE_QASM3_STRING, filename="circuit3.qasm"))

    checker = EquivalenceChecker(representation="matrix")
    result = checker.check(qasm_file, qasm_file)
    assert result["equivalent"] is True


def test_check_accepts_qasm2_raw_string() -> None:
    """Check that a raw QASM 2 string (not a file path) is accepted and returns equivalent."""
    checker = EquivalenceChecker(representation="mpo")
    result = checker.check(LARGE_QASM2_STRING, LARGE_QASM2_STRING)
    assert result["equivalent"] is True


@requires_qasm3_import
def test_check_accepts_qasm3_raw_string() -> None:
    """Check that a raw QASM 3 string (not a file path) is accepted and returns equivalent."""
    checker = EquivalenceChecker(representation="matrix")
    result = checker.check(SAMPLE_QASM3_STRING, SAMPLE_QASM3_STRING)
    assert result["equivalent"] is True


def test_check_issue_qasm_raw_strings_custom_vs_expanded() -> None:
    """Raw OpenQASM strings with custom gates are equivalent to their expanded form."""
    result = _issue_checker(representation="mpo").check(ISSUE_QASM_CUSTOM, ISSUE_QASM_EXPANDED)
    assert result["equivalent"] is True
    assert result["representation"] == "mpo"


def test_check_mixed_qasm_path_and_quantumcircuit(tmp_path: Path) -> None:
    """Mixed OpenQASM path and QuantumCircuit inputs agree with path-only checking."""
    qasm_path = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    qc = load(filename=str(qasm_path))
    checker = EquivalenceChecker(representation="mpo")
    assert checker.check(qasm_path, qc)["equivalent"] is True
    assert checker.check(qc, qasm_path)["equivalent"] is True


def test_check_mixed_qasm_raw_string_and_quantumcircuit(tmp_path: Path) -> None:
    """Raw OpenQASM string mixed with a QuantumCircuit matches path-based checking."""
    qasm_path = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    qc = load(filename=str(qasm_path))
    checker = EquivalenceChecker(representation="mpo")
    assert checker.check(LARGE_QASM2_STRING, qc)["equivalent"] is True
    assert checker.check(qc, LARGE_QASM2_STRING)["equivalent"] is True


def test_check_issue_qasm_raw_strings_with_final_measurements() -> None:
    """Raw OpenQASM with custom gates and final measurements is self-equivalent on MPO."""
    result = _issue_checker(representation="mpo").check(ISSUE_QASM_WITH_MEASURES, ISSUE_QASM_WITH_MEASURES)
    assert result["equivalent"] is True
    assert result["representation"] == "mpo"


def test_check_qasm2_self_equivalence_uses_matrix_backend(tmp_path: Path) -> None:
    """OpenQASM 2 self-equivalence can run on the explicit matrix backend."""
    qasm_path = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    checker = EquivalenceChecker(representation="matrix")
    result = checker.check(qasm_path, qasm_path)
    assert result["equivalent"] is True
    assert result["representation"] == "matrix"


def test_check_qasm3_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """EquivalenceChecker propagates ImportError when OpenQASM 3 importer is missing."""
    monkeypatch.setattr("mqt.yaqs.digital.utils.qasm_utils.HAS_QASM3_IMPORT", False)
    with pytest.raises(ImportError, match="mqt-yaqs\\[qasm3\\]"):
        EquivalenceChecker(representation="matrix").check(SAMPLE_QASM3_STRING, SAMPLE_QASM3_STRING)


def test_check_mpo_path_returns_operator_diagnostics() -> None:
    """MPO backend returns composed-operator diagnostics and measured fidelity."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)
    qc2 = qc1.copy()

    result = EquivalenceChecker(representation="mpo").check(qc1, qc2)

    assert result["equivalent"] is True
    assert result["representation"] == "mpo"
    assert isinstance(result["fidelity"], float)
    assert float(result["fidelity"]) == pytest.approx(1.0, abs=1e-10)

    mpo = result["mpo"]
    assert mpo is not None
    assert result["matrix"] is None
    assert mpo.length == 2

    schmidt = result["schmidt_values"]
    assert schmidt is not None
    assert schmidt.ndim == 1
    assert schmidt.dtype == np.float64

    center_entropy = result["center_cut_entanglement_entropy"]
    global_entropy = result["global_entanglement_entropy"]
    assert center_entropy is not None
    assert global_entropy is not None
    assert float(center_entropy) >= 0.0
    assert float(global_entropy) >= 0.0
    assert float(center_entropy) == pytest.approx(0.0, abs=1e-10)


def test_check_matrix_path_returns_fidelity_and_matrix() -> None:
    """Matrix backend returns measured fidelity and the dense composed operator."""
    qc = QuantumCircuit(2)
    qc.h(0)

    result = EquivalenceChecker(representation="matrix").check(qc, qc)

    assert result["representation"] == "matrix"
    assert isinstance(result["fidelity"], float)
    assert float(result["fidelity"]) == pytest.approx(1.0, abs=1e-10)
    matrix = result["matrix"]
    assert matrix is not None
    assert matrix.shape == (4, 4)
    assert matrix.dtype == np.complex128
    assert result["mpo"] is None
    assert result["schmidt_values"] is None
    assert result["center_cut_entanglement_entropy"] is None
    assert result["global_entanglement_entropy"] is None


def test_check_non_equivalent_pair_still_returns_diagnostics() -> None:
    """Diagnostics describe the composed operator even when circuits differ."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.x(1)

    result = EquivalenceChecker(representation="mpo").check(qc1, qc2)

    assert result["equivalent"] is False
    assert float(result["fidelity"]) < 1.0
    assert result["matrix"] is None
    assert result["mpo"] is not None
    assert result["schmidt_values"] is not None
    assert result["center_cut_entanglement_entropy"] is not None
    assert result["global_entanglement_entropy"] is not None


def _pauli_x_noise(num_qubits: int, strength: float) -> NoiseModel:
    """Build a local Pauli-X noise model on every qubit.

    Args:
        num_qubits: Number of qubits to cover.
        strength: Direct per-opportunity branch probability on each site.

    Returns:
        A :class:`NoiseModel` of one-qubit ``pauli_x`` processes.
    """
    return NoiseModel([{"name": "pauli_x", "sites": [q], "strength": strength} for q in range(num_qubits)])


def test_zero_strength_noise_matches_noiseless_check() -> None:
    """A Pauli model with zero probabilities leaves the second circuit unchanged."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    noise = _pauli_x_noise(2, 0.0)
    checker = EquivalenceChecker(representation="mpo")

    noiseless = checker.check(qc, qc)
    ensemble = checker.check(qc, qc, noise_model=noise, num_traj=4, random_seed=0)

    assert ensemble["num_traj"] == 4
    assert len(ensemble["trajectories"]) == 4
    assert ensemble["equivalent"] is True
    assert float(ensemble["fidelity"]) == pytest.approx(float(noiseless["fidelity"]) ** 2, abs=1e-12)
    standard_error = ensemble["fidelity_error"]
    assert standard_error is not None
    assert standard_error == pytest.approx(0.0, abs=1e-12)
    assert ensemble["mpo"] is None
    assert ensemble["matrix"] is None
    for traj in ensemble["trajectories"]:
        assert traj["mpo"] is None
        assert traj["matrix"] is None
        assert float(traj["fidelity"]) == pytest.approx(1.0, abs=1e-12)


def test_finite_pauli_noise_on_circuit2_reduces_fidelity() -> None:
    """Sampling Pauli errors onto the second circuit lowers process fidelity."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    noise = _pauli_x_noise(2, 1.0)
    checker = EquivalenceChecker(representation="mpo", fidelity=1 - 1e-8)

    ensemble = checker.check(qc, qc, noise_model=noise, num_traj=8, random_seed=1)

    assert ensemble["equivalent"] is False
    assert float(ensemble["fidelity"]) < 1.0 - 1e-6
    assert all(isinstance(traj["fidelity"], float) for traj in ensemble["trajectories"])


def test_noise_is_applied_only_to_circuit2() -> None:
    """An empty second circuit has no noise sites, so a strong model is a no-op."""
    qc_cx = QuantumCircuit(2)
    qc_cx.cx(0, 1)
    qc_id = QuantumCircuit(2)
    noise = _pauli_x_noise(2, 1.0)
    checker = EquivalenceChecker(representation="mpo")

    noiseless = checker.check(qc_cx, qc_id)
    noisy_empty = checker.check(qc_cx, qc_id, noise_model=noise, num_traj=5, random_seed=0)
    noisy_cx = checker.check(qc_id, qc_cx, noise_model=noise, num_traj=5, random_seed=0)

    assert float(noisy_empty["fidelity"]) == pytest.approx(float(noiseless["fidelity"]) ** 2, abs=1e-12)
    assert float(noisy_cx["fidelity"]) != pytest.approx(float(checker.check(qc_id, qc_cx)["fidelity"]) ** 2, abs=1e-6)


def test_noisy_check_is_seeded_reproducible() -> None:
    """The same ``(random_seed, num_traj)`` pair reproduces trajectory fidelities."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    noise = _pauli_x_noise(2, 0.2)
    checker = EquivalenceChecker(representation="mpo")

    first = checker.check(qc, qc, noise_model=noise, num_traj=6, random_seed=42)
    second = checker.check(qc, qc, noise_model=noise, num_traj=6, random_seed=42)
    third = checker.check(qc, qc, noise_model=noise, num_traj=6, random_seed=43)

    assert [traj["fidelity"] for traj in first["trajectories"]] == [traj["fidelity"] for traj in second["trajectories"]]
    assert [traj["fidelity"] for traj in first["trajectories"]] != [traj["fidelity"] for traj in third["trajectories"]]


def test_noisy_check_precomputes_instruction_noise_plan_once() -> None:
    """Gate opportunities are classified once and reused by every trajectory."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    noise = _pauli_x_noise(2, 0.2)

    with patch(
        "mqt.yaqs.equivalence_checker.is_digital_noise_opportunity",
        return_value=True,
    ) as classify_opportunity:
        EquivalenceChecker(representation="matrix", parallel=False).check(
            circuit,
            circuit,
            noise_model=noise,
            num_traj=4,
            random_seed=0,
        )

    assert classify_opportunity.call_count == len(circuit.data)


@pytest.mark.parametrize(
    ("draw", "error_gate"),
    [
        pytest.param(0.1, "x", id="first-branch"),
        pytest.param(0.2, "y", id="second-branch-boundary"),
        pytest.param(0.499, "y", id="second-branch"),
        pytest.param(0.5, None, id="identity-remainder"),
    ],
)
def test_noisy_check_uses_direct_categorical_probabilities(
    draw: float,
    error_gate: Literal["x", "y"] | None,
) -> None:
    """One direct draw selects a same-support branch or its identity remainder."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    expected = circuit.copy()
    if error_gate is not None:
        getattr(expected, error_gate)(0)
    noise = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.2},
        {"name": "pauli_y", "sites": [0], "strength": 0.3},
    ])

    with patch("mqt.yaqs.equivalence_checker.make_trajectory_rng") as make_rng:
        make_rng.return_value.random.return_value = draw
        result = EquivalenceChecker(representation="matrix").check(
            expected,
            circuit,
            noise_model=noise,
        )

    assert result["equivalent"] is True
    assert float(result["fidelity"]) == pytest.approx(1.0, abs=1e-12)
    assert make_rng.return_value.random.call_count == 1


def test_noisy_check_samples_distinct_supports_independently() -> None:
    """Distinct and overlapping supports may all contribute after one gate."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    expected = circuit.copy()
    expected.x(0)
    expected.z(1)
    expected.y(0)
    expected.x(1)
    noise = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 1.0},
        {"name": "pauli_z", "sites": [1], "strength": 1.0},
        {"name": "crosstalk_yx", "sites": [0, 1], "strength": 1.0},
    ])

    result = EquivalenceChecker(representation="matrix").check(
        expected,
        circuit,
        noise_model=noise,
        random_seed=0,
    )

    assert result["equivalent"] is True
    assert float(result["fidelity"]) == pytest.approx(1.0, abs=1e-12)


def test_noisy_check_rejects_probability_sum_above_one_per_support() -> None:
    """Mutually exclusive branches on one support cannot exceed unit probability."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    noise = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.6},
        {"name": "pauli_y", "sites": [0], "strength": 0.5},
    ])

    with pytest.raises(ValueError, match=r"sharing support \(0,\) to sum to at most 1"):
        EquivalenceChecker(representation="matrix").check(circuit, circuit, noise_model=noise)


def test_noisy_check_accepts_nine_branches_summing_to_one() -> None:
    """Nine two-site Pauli branches of probability one ninth form a valid group."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    noise = NoiseModel([
        {"name": f"crosstalk_{left}{right}", "sites": [0, 1], "strength": 1 / 9} for left in "xyz" for right in "xyz"
    ])

    result = EquivalenceChecker(representation="matrix").check(circuit, circuit, noise_model=noise, random_seed=0)

    assert float(result["fidelity"]) == pytest.approx(0.0, abs=1e-12)


def test_noisy_check_validates_probability_after_distribution_sampling() -> None:
    """A resolved distribution draw must satisfy the checker's probability bound."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    noise = NoiseModel([
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "normal", "mean": 1.1, "std": 0.0},
        }
    ])

    with pytest.raises(ValueError, match=r"sharing support \(0,\) to sum to at most 1"):
        EquivalenceChecker(representation="matrix").check(circuit, circuit, noise_model=noise)


@pytest.mark.parametrize("representation", ["mpo", "matrix"])
def test_noisy_check_accepts_mpo_and_matrix_backends(representation: Literal["mpo", "matrix"]) -> None:
    """Both backends accept a Pauli ``noise_model`` and return an ensemble."""
    qc = QuantumCircuit(2)
    qc.h(0)
    noise = _pauli_x_noise(2, 0.5)
    result = EquivalenceChecker(representation=representation).check(
        qc, qc, noise_model=noise, num_traj=3, random_seed=0
    )

    assert result["representation"] == representation
    assert result["num_traj"] == 3
    assert len(result["trajectories"]) == 3
    assert isinstance(result["equivalent"], bool)
    assert isinstance(result["fidelity"], float)
    assert result["fidelity_error"] is not None
    if representation == "matrix":
        assert result["schmidt_values"] is None
        assert all(trajectory["schmidt_values"] is None for trajectory in result["trajectories"])
        assert result["center_cut_entanglement_entropy"] is None
        assert result["global_entanglement_entropy"] is None
    else:
        trajectory_schmidt_values = [
            values for trajectory in result["trajectories"] if (values := trajectory["schmidt_values"]) is not None
        ]
        ensemble_schmidt_values = result["schmidt_values"]
        assert ensemble_schmidt_values is not None
        assert len(trajectory_schmidt_values) == result["num_traj"]
        np.testing.assert_allclose(ensemble_schmidt_values, np.concatenate(trajectory_schmidt_values))
        assert result["center_cut_entanglement_entropy"] is not None
        assert result["global_entanglement_entropy"] is not None


@pytest.mark.parametrize("max_workers", [8, None])
def test_noisy_ensemble_caps_workers_and_reassembles_by_index(max_workers: int | None) -> None:
    """A noisy pool is capped by trajectory count and preserves trajectory order."""
    reference = QuantumCircuit(2)
    checker = EquivalenceChecker(representation="matrix", parallel=True, max_workers=max_workers)
    trajectory_results = []
    for overlap in (0.2, 0.8):
        candidate = QuantumCircuit(2)
        candidate.ry(2 * np.arccos(overlap), 0)
        trajectory_results.append(checker.check(reference, candidate))
    indexed_results = [(1, trajectory_results[1]), (0, trajectory_results[0])]

    with (
        patch("mqt.yaqs.equivalence_checker.available_cpus", return_value=9) as available_cpus,
        patch(
            "mqt.yaqs.equivalence_checker.run_backend_parallel",
            return_value=iter(indexed_results),
        ) as run_parallel,
    ):
        result = checker.check(
            reference,
            reference,
            noise_model=_pauli_x_noise(2, 0.0),
            num_traj=2,
            random_seed=0,
        )

    run_parallel.assert_called_once()
    assert run_parallel.call_args.kwargs["n_jobs"] == 2
    assert run_parallel.call_args.kwargs["max_workers"] == 2
    if max_workers is None:
        available_cpus.assert_called_once_with()
    else:
        available_cpus.assert_not_called()
    np.testing.assert_allclose([trajectory["fidelity"] for trajectory in result["trajectories"]], [0.2, 0.8])


def test_parallel_false_keeps_noisy_ensemble_serial() -> None:
    """Disabling parallelism prevents noisy process-pool dispatch."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)

    with patch("mqt.yaqs.equivalence_checker.run_backend_parallel") as run_parallel:
        result = EquivalenceChecker(representation="matrix", parallel=False, max_workers=8).check(
            circuit,
            circuit,
            noise_model=_pauli_x_noise(2, 0.0),
            num_traj=3,
            random_seed=0,
        )

    run_parallel.assert_not_called()
    assert len(result["trajectories"]) == 3


def test_noisy_ensemble_averages_squared_overlaps_and_reports_standard_error() -> None:
    """The channel estimate averages squared overlaps and reports their sample SEM."""
    reference = QuantumCircuit(2)
    overlaps = np.asarray([0.2, 0.5, 0.9])
    checker = EquivalenceChecker(representation="matrix", fidelity=0.6, max_workers=1)
    trajectory_results = []
    for overlap in overlaps:
        candidate = QuantumCircuit(2)
        candidate.ry(2 * np.arccos(overlap), 0)
        trajectory_results.append(checker.check(reference, candidate))

    with patch(
        "mqt.yaqs.equivalence_checker._run_noisy_check_trajectory",
        side_effect=trajectory_results,
    ):
        result = checker.check(
            reference,
            reference,
            noise_model=_pauli_x_noise(2, 0.0),
            num_traj=len(overlaps),
            random_seed=0,
        )

    squared_overlaps = np.square(overlaps)
    assert result["fidelity"] == pytest.approx(float(np.mean(squared_overlaps)), abs=1e-12)
    assert result["fidelity"] != pytest.approx(float(np.mean(overlaps)), abs=1e-12)
    assert result["fidelity"] != pytest.approx(float(np.mean(overlaps) ** 2), abs=1e-12)
    assert result["fidelity_error"] == pytest.approx(
        float(np.std(squared_overlaps, ddof=1) / np.sqrt(len(squared_overlaps))), abs=1e-12
    )
    assert [trajectory["equivalent"] for trajectory in result["trajectories"]] == [False, False, True]
    assert result["equivalent"] is True


def test_single_trajectory_process_fidelity_has_no_standard_error() -> None:
    """One trajectory gives a point estimate but cannot estimate sampling uncertainty."""
    reference = QuantumCircuit(2)
    candidate = QuantumCircuit(2)
    candidate.ry(2 * np.arccos(0.6), 0)
    checker = EquivalenceChecker(representation="matrix", fidelity=0.7, max_workers=1)
    trajectory_result = checker.check(reference, candidate)

    with patch(
        "mqt.yaqs.equivalence_checker._run_noisy_check_trajectory",
        return_value=trajectory_result,
    ):
        result = checker.check(
            reference,
            reference,
            noise_model=_pauli_x_noise(2, 0.0),
            num_traj=1,
            random_seed=0,
        )

    assert result["fidelity"] == pytest.approx(0.6**2, abs=1e-12)
    assert result["fidelity_error"] is None
    assert result["equivalent"] is False


@pytest.mark.parametrize("representation", ["matrix", "mpo"])
def test_noiseless_check_keeps_root_overlap_semantics(representation: Literal["matrix", "mpo"]) -> None:
    """Noiseless checks retain root-overlap fidelity and threshold semantics."""
    reference = QuantumCircuit(2)
    candidate = QuantumCircuit(2)
    candidate.ry(2 * np.pi / 3, 0)

    result = EquivalenceChecker(representation=representation, fidelity=0.4).check(reference, candidate)

    assert result["fidelity"] == pytest.approx(0.5, abs=1e-12)
    assert result["equivalent"] is True
    assert "fidelity_error" not in result


def test_num_traj_without_noise_model_raises() -> None:
    """``num_traj`` is only meaningful together with a noise model."""
    qc = QuantumCircuit(1)
    with pytest.raises(ValueError, match="num_traj must be 1"):
        EquivalenceChecker(representation="mpo").check(qc, qc, num_traj=4)


def test_negative_random_seed_raises() -> None:
    """Negative checker seeds are rejected consistently on both execution paths."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    checker = EquivalenceChecker(representation="matrix")

    with pytest.raises(ValueError, match="random_seed must be non-negative, got -1"):
        checker.check(circuit, circuit, random_seed=-1)
    with pytest.raises(ValueError, match="random_seed must be non-negative, got -1"):
        checker.check(circuit, circuit, noise_model=_pauli_x_noise(2, 0.1), random_seed=-1)


def test_non_pauli_noise_is_rejected() -> None:
    """Dissipative processes cannot be materialized as stochastic circuits."""
    qc = QuantumCircuit(1)
    qc.x(0)
    noise = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.1}])
    with pytest.raises(ValueError, match="process that is not supported for circuit sampling"):
        EquivalenceChecker(representation="mpo").check(qc, qc, noise_model=noise)


def test_zero_strength_non_pauli_noise_is_rejected() -> None:
    """Unsupported operators are rejected even when their probability is zero."""
    qc = QuantumCircuit(1)
    noise = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.0}])

    with pytest.raises(ValueError, match="process that is not supported for circuit sampling"):
        EquivalenceChecker(representation="matrix").check(qc, qc, noise_model=noise)


def test_pauli_name_does_not_hide_unsupported_matrix() -> None:
    """A recognized process name cannot override a non-Pauli normalized operator."""
    qc = QuantumCircuit(1)
    noise = NoiseModel([
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": 1.0,
            "matrix": 2 * NoiseModel.get_operator("x"),
        }
    ])

    with pytest.raises(ValueError, match="process that is not supported for circuit sampling"):
        EquivalenceChecker(representation="matrix").check(qc, qc, noise_model=noise)


@pytest.mark.parametrize("name", ["pauli_x", "custom_z"])
def test_noisy_check_uses_custom_pauli_matrix_override(name: str) -> None:
    """The normalized process matrix, rather than its name, selects the Pauli gate."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    expected = circuit.copy()
    expected.z(0)
    noise = NoiseModel([
        {
            "name": name,
            "sites": [0],
            "strength": 1.0,
            "matrix": np.exp(0.37j) * np.diag([1.0, -1.0]),
        }
    ])

    result = EquivalenceChecker(representation="matrix").check(expected, circuit, noise_model=noise, random_seed=0)

    assert result["equivalent"] is True
    assert float(result["fidelity"]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    ("gate_name", "num_qubits", "noise_gate"),
    [
        pytest.param("h", 1, None, id="h"),
        pytest.param("cx", 2, "x", id="cx"),
        pytest.param("ccx", 3, None, id="ccx"),
    ],
)
def test_noisy_check_uses_two_qubit_gate_opportunities(
    gate_name: str,
    num_qubits: int,
    noise_gate: Literal["x"] | None,
) -> None:
    """Only supported two-qubit gates are noise opportunities for the checker."""
    circuit = QuantumCircuit(num_qubits)
    getattr(circuit, gate_name)(*range(num_qubits))
    expected = circuit.copy()
    if noise_gate is not None:
        expected.x(0)
    noise = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1.0}])

    result = EquivalenceChecker(representation="matrix").check(
        expected,
        circuit,
        noise_model=noise,
        random_seed=0,
    )

    assert result["equivalent"] is True
    assert float(result["fidelity"]) == pytest.approx(1.0, abs=1e-12)


def test_noisy_check_treats_unitary_instruction_as_opportunity() -> None:
    """A matrix-backed two-qubit unitary Instruction is a noise opportunity."""
    definition = QuantumCircuit(2, name="wrapped_cx")
    definition.cx(0, 1)
    circuit = QuantumCircuit(2)
    circuit.append(definition.to_instruction(), [0, 1])
    expected = circuit.copy()
    expected.x(0)
    noise = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1.0}])

    result = EquivalenceChecker(representation="matrix").check(
        expected,
        circuit,
        noise_model=noise,
        random_seed=0,
    )

    assert result["equivalent"] is True
    assert float(result["fidelity"]) == pytest.approx(1.0, abs=1e-12)


def test_noisy_check_preserves_descending_crosstalk_order() -> None:
    """Normalized crosstalk matrices retain the caller's descending-site operator order."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    expected = circuit.copy()
    expected.y(0)
    expected.x(1)
    noise = NoiseModel([{"name": "crosstalk_xy", "sites": [1, 0], "strength": 1.0}])

    result = EquivalenceChecker(representation="matrix").check(expected, circuit, noise_model=noise, random_seed=0)

    assert result["equivalent"] is True
    assert float(result["fidelity"]) == pytest.approx(1.0, abs=1e-12)


def test_noisy_check_accepts_normalized_long_range_crosstalk() -> None:
    """Long-range Pauli factors are decoded independently of the process alias."""
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    expected = circuit.copy()
    expected.x(0)
    expected.y(2)
    noise = NoiseModel([{"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 1.0}])

    result = EquivalenceChecker(representation="matrix").check(expected, circuit, noise_model=noise, random_seed=0)

    assert result["equivalent"] is True
    assert float(result["fidelity"]) == pytest.approx(1.0, abs=1e-12)


def test_noisy_check_rejects_out_of_range_process_site() -> None:
    """Noise sites are validated against the circuit width before sampling."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    noise = NoiseModel([{"name": "pauli_x", "sites": [1], "strength": 1.0}])

    with pytest.raises(ValueError, match="Process site index 1 is out of range for length 1"):
        EquivalenceChecker(representation="matrix").check(circuit, circuit, noise_model=noise)


def test_noisy_check_rejects_wrong_process_dimension() -> None:
    """Noise operator dimensions are validated for qubit circuits before sampling."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    noise = NoiseModel([
        {
            "name": "custom",
            "sites": [0],
            "strength": 1.0,
            "matrix": np.eye(3),
        }
    ])

    with pytest.raises(ValueError, match=r"Process matrix shape \(3, 3\).*expected \(2, 2\)"):
        EquivalenceChecker(representation="matrix").check(circuit, circuit, noise_model=noise)


def test_scheduled_jumps_are_rejected_by_noisy_check() -> None:
    """Scheduled jumps are not part of the explicit circuit-sampling path."""
    qc = QuantumCircuit(1)
    qc.x(0)
    noise = NoiseModel(scheduled_jumps=[{"time": 0.0, "sites": [0], "name": "x"}])
    with pytest.raises(ValueError, match="Scheduled jumps are not supported for circuit-sampled equivalence checks"):
        EquivalenceChecker(representation="mpo").check(qc, qc, noise_model=noise)


def test_example_ideal_versus_noisy_compiled_circuit() -> None:
    """The docs example: noiseless compiled circuit matches; Pauli noise lowers process fidelity."""
    ideal = QuantumCircuit(4)
    for qubit in range(4):
        ideal.ry(0.4 * (qubit + 1), qubit)
    for qubit in range(3):
        ideal.cx(qubit, qubit + 1)
    compiled = transpile(ideal, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    noise = NoiseModel([{"name": "pauli_x", "sites": [qubit], "strength": 0.02} for qubit in range(4)])
    checker = EquivalenceChecker(representation="mpo", threshold=1e-6)

    noiseless = checker.check(ideal, compiled)
    noisy = checker.check(ideal, compiled, noise_model=noise, num_traj=8, random_seed=0)

    assert noiseless["equivalent"] is True
    assert float(noiseless["fidelity"]) == pytest.approx(1.0, abs=1e-10)
    assert noisy["num_traj"] == 8
    assert float(noisy["fidelity"]) < float(noiseless["fidelity"]) ** 2


def test_seeded_serial_and_process_pool_ensembles_agree() -> None:
    """Serial and process-pool workers return the same seeded MPO ensemble."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    noise = _pauli_x_noise(2, 0.5)
    kwargs = {"noise_model": noise, "num_traj": 6, "random_seed": 0}

    serial = EquivalenceChecker(representation="mpo", parallel=False).check(qc, qc, **kwargs)
    pooled = EquivalenceChecker(representation="mpo", parallel=True, max_workers=2).check(qc, qc, **kwargs)
    serial_fidelities = [traj["fidelity"] for traj in serial["trajectories"]]
    pooled_fidelities = [traj["fidelity"] for traj in pooled["trajectories"]]

    assert len(set(serial_fidelities)) > 1, "Parity fixture must sample both noisy and clean trajectories."
    np.testing.assert_allclose(serial_fidelities, pooled_fidelities, atol=1e-12)
    assert serial["fidelity"] == pytest.approx(pooled["fidelity"], abs=1e-12)
    assert serial["fidelity_error"] == pytest.approx(pooled["fidelity_error"], abs=1e-12)
    assert serial["equivalent"] is pooled["equivalent"]
    assert serial["schmidt_values"] is not None
    assert pooled["schmidt_values"] is not None
    np.testing.assert_allclose(serial["schmidt_values"], pooled["schmidt_values"], atol=1e-12)
