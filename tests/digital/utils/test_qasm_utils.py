# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QASM loading utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from mqt.yaqs.digital.utils.qasm_utils import (
    _first_non_comment_line,  # noqa: PLC2701
    load_circuit,
)

QASM2_STRING = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
"""

QASM3_STRING = """\
OPENQASM 3.0;
qubit[1] q;
h q[0];
"""


def test_first_non_comment_line_skips_comments() -> None:
    """Lines starting with // are skipped; the first real line is returned."""
    text = "// comment\n// another\nOPENQASM 2.0;"
    assert _first_non_comment_line(text) == "OPENQASM 2.0;"


def test_first_non_comment_line_empty_returns_empty() -> None:
    """An all-comment or blank text returns an empty string."""
    assert not _first_non_comment_line("// only comments\n// still comments")
    assert not _first_non_comment_line("")


def test_load_circuit_passthrough_quantum_circuit() -> None:
    """A QuantumCircuit is returned unchanged."""
    qc = QuantumCircuit(1)
    qc.h(0)
    result = load_circuit(qc)
    assert result is qc


def test_load_circuit_qasm2_string() -> None:
    """A raw QASM 2 string is parsed and returned as a QuantumCircuit."""
    qc = load_circuit(QASM2_STRING)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 1


def test_load_circuit_qasm2_path_object() -> None:
    """A QASM 2 file given as a Path is loaded and returned as a QuantumCircuit."""
    qasm_file = Path(__file__).parent.parent.parent / "circuit.qasm"
    qc = load_circuit(qasm_file)
    assert isinstance(qc, QuantumCircuit)


def test_load_circuit_qasm2_str_path() -> None:
    """A QASM 2 file given as a str path is loaded and returned as a QuantumCircuit."""
    qasm_file = str(Path(__file__).parent.parent.parent / "circuit.qasm")
    qc = load_circuit(qasm_file)
    assert isinstance(qc, QuantumCircuit)


def test_load_circuit_qasm3_string() -> None:
    """A raw QASM 3 string is parsed and returned as a QuantumCircuit."""
    pytest.importorskip("qiskit_qasm3_import")
    qc = load_circuit(QASM3_STRING)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 1


def test_load_circuit_qasm3_path_object() -> None:
    """A QASM 3 file given as a Path is loaded and returned as a QuantumCircuit."""
    pytest.importorskip("qiskit_qasm3_import")
    qasm_file = Path(__file__).parent.parent.parent / "circuit3.qasm"
    qc = load_circuit(qasm_file)
    assert isinstance(qc, QuantumCircuit)


def test_load_circuit_qasm3_str_path() -> None:
    """A QASM 3 file given as a str path is loaded and returned as a QuantumCircuit."""
    pytest.importorskip("qiskit_qasm3_import")
    qasm_file = str(Path(__file__).parent.parent.parent / "circuit3.qasm")
    qc = load_circuit(qasm_file)
    assert isinstance(qc, QuantumCircuit)
