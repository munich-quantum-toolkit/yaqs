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
from qiskit.qasm2 import loads

from mqt.yaqs import EquivalenceChecker
from mqt.yaqs.digital.utils.qasm_utils import (
    _parse_qasm_version,  # noqa: PLC2701 — unit tests target the private parser directly
    load_circuit,
)
from tests.conftest import LARGE_QASM2_STRING, SAMPLE_QASM3_STRING, requires_qasm3_import, write_qasm_file

QASM2_STRING = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
"""

QASM3_STRING = """\
OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
h q[0];
"""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("// comment\n// another\nOPENQASM 2.0;", "2"),
        ("/* preamble */\nOPENQASM 2.0;", "2"),
        ("/*\n multiline\n */\nOPENQASM 3.0;", "3"),
        ("OPENQASM;", "2"),
        ("OPENQASM 3.0;", "3"),
        ("openqasm 3.0;", "3"),
        ("/* x */ OPENQASM 2.0;", "2"),
        ("// only comments\n// still comments", None),
        ("", None),
    ],
)
def test_parse_qasm_version(text: str, expected: str | None) -> None:
    """Parse OpenQASM major versions from source text, skipping comments."""
    assert _parse_qasm_version(text) == expected


def test_load_circuit_passthrough_quantum_circuit() -> None:
    """A QuantumCircuit is returned unchanged."""
    qc = QuantumCircuit(1)
    qc.h(0)
    result = load_circuit(qc)
    assert result is qc


def test_load_circuit_qasm2_string() -> None:
    """A raw OpenQASM 2 string is parsed and returned as a QuantumCircuit."""
    qc = load_circuit(QASM2_STRING)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 1


def test_load_circuit_qasm2_path_object(tmp_path: Path) -> None:
    """An OpenQASM 2 file given as a Path is loaded and returned as a QuantumCircuit."""
    qasm_file = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    qc = load_circuit(qasm_file)
    assert isinstance(qc, QuantumCircuit)


def test_load_circuit_qasm2_str_path(tmp_path: Path) -> None:
    """An OpenQASM 2 file given as a str path is loaded and returned as a QuantumCircuit."""
    qasm_file = str(write_qasm_file(tmp_path, LARGE_QASM2_STRING))
    qc = load_circuit(qasm_file)
    assert isinstance(qc, QuantumCircuit)


def test_load_circuit_qasm2_string_with_leading_block_comment() -> None:
    """Block comments before the OPENQASM header do not prevent parsing."""
    text = f"/* header */\n{QASM2_STRING}"
    qc = load_circuit(text)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 1


@requires_qasm3_import
def test_load_circuit_qasm3_string() -> None:
    """A raw OpenQASM 3 string is parsed and returned as a QuantumCircuit."""
    qc = load_circuit(QASM3_STRING)
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == 1


@requires_qasm3_import
def test_load_circuit_qasm3_path_object(tmp_path: Path) -> None:
    """An OpenQASM 3 file given as a Path is loaded and returned as a QuantumCircuit."""
    qasm_file = write_qasm_file(tmp_path, SAMPLE_QASM3_STRING, filename="circuit3.qasm")
    qc = load_circuit(qasm_file)
    assert isinstance(qc, QuantumCircuit)


@requires_qasm3_import
def test_load_circuit_qasm3_str_path(tmp_path: Path) -> None:
    """An OpenQASM 3 file given as a str path is loaded and returned as a QuantumCircuit."""
    qasm_file = str(write_qasm_file(tmp_path, SAMPLE_QASM3_STRING, filename="circuit3.qasm"))
    qc = load_circuit(qasm_file)
    assert isinstance(qc, QuantumCircuit)


def test_load_circuit_rejects_invalid_string() -> None:
    """A str that is neither OpenQASM nor a valid path raises ValueError."""
    with pytest.raises(ValueError, match="Expected a QuantumCircuit"):
        load_circuit("not-a-qasm-program")


def test_load_circuit_rejects_missing_path() -> None:
    """A str path to a missing file raises ValueError."""
    with pytest.raises(ValueError, match="Expected a QuantumCircuit"):
        load_circuit("definitely/missing/file.qasm")


def test_load_circuit_rejects_missing_path_object() -> None:
    """A Path to a missing file raises ValueError, not FileNotFoundError."""
    with pytest.raises(ValueError, match="Expected a QuantumCircuit"):
        load_circuit(Path("definitely/missing/file.qasm"))


def test_load_circuit_rejects_file_without_openqasm_header(tmp_path: Path) -> None:
    """A file that exists but lacks an OPENQASM header raises ValueError."""
    bad_file = tmp_path / "bad.qasm"
    bad_file.write_text("qreg q[1];\nh q[0];", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a QuantumCircuit"):
        load_circuit(bad_file)
    with pytest.raises(ValueError, match="Expected a QuantumCircuit"):
        load_circuit(str(bad_file))


def test_load_circuit_qasm2_path_and_string_match(tmp_path: Path) -> None:
    """Path and raw-string loading produce structurally equivalent QuantumCircuits."""
    qasm_file = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    from_path = load_circuit(qasm_file)
    from_string = load_circuit(LARGE_QASM2_STRING)
    expected = loads(LARGE_QASM2_STRING)
    assert from_path.num_qubits == expected.num_qubits == from_string.num_qubits
    assert len(from_path.data) == len(expected.data) == len(from_string.data)
    checker = EquivalenceChecker(representation="mpo")
    assert checker.check(from_path, from_string)["equivalent"] is True


def test_load_circuit_qasm3_missing_importer(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenQASM 3 without qiskit-qasm3-import raises ImportError with install hint."""
    monkeypatch.setattr("mqt.yaqs.digital.utils.qasm_utils.HAS_QASM3_IMPORT", False)
    with pytest.raises(ImportError, match="mqt-yaqs\\[qasm3\\]"):
        load_circuit(QASM3_STRING)


def test_load_circuit_qasm3_missing_importer_on_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OpenQASM 3 file paths also raise ImportError when the importer is absent."""
    qasm3_file = tmp_path / "prog.qasm"
    qasm3_file.write_text(QASM3_STRING, encoding="utf-8")
    monkeypatch.setattr("mqt.yaqs.digital.utils.qasm_utils.HAS_QASM3_IMPORT", False)
    with pytest.raises(ImportError, match="mqt-yaqs\\[qasm3\\]"):
        load_circuit(qasm3_file)
