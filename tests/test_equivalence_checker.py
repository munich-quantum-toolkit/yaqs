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
from qiskit import QuantumCircuit
from qiskit.circuit.library import ECRGate, U1Gate, U3Gate
from qiskit.converters import circuit_to_dag
from qiskit.qasm2 import load, loads

from mqt.yaqs import EquivalenceChecker
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.utils import matrix_utils
from mqt.yaqs.digital.utils.contraction_utils import MIN_QUBITS_FOR_MPO_PARALLEL
from mqt.yaqs.digital.utils.dag_utils import SUPPORTED_QISKIT_GATE_NAMES, convert_dag_to_tensor_algorithm
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
