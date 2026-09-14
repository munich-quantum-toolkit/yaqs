# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for DITQASM 2.0 (qudit QASM dialect) loading utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mqt.yaqs.digital.utils.qudit_qasm_utils import is_ditqasm_source, load_qudit_circuit

if TYPE_CHECKING:
    from pathlib import Path

DITQASM_STRING = """\
DITQASM 2.0;

qreg field [2][2,3];

h field[0];
cx field[0], field[1];
"""

OPENQASM_STRING = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
"""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (DITQASM_STRING, True),
        ("\n\n  \nDITQASM 2.0;", True),
        ("// comment\n// another\nDITQASM 2.0;", True),
        ("/* preamble */\nDITQASM 2.0;", True),
        ("/* x */ DITQASM 2.0;", True),
        ("/*\n multiline\n */\nDITQASM 2.0;", True),
        ("/*\n multiline\n */ DITQASM 2.0;", True),
        ("ditqasm 2.0;", True),
        (OPENQASM_STRING, False),
        ("// only comments\n// still comments", False),
        ("", False),
    ],
)
def test_is_ditqasm_source_raw_text(text: str, *, expected: bool) -> None:
    """``is_ditqasm_source`` correctly classifies raw source text."""
    assert is_ditqasm_source(text) is expected


def test_is_ditqasm_source_path(tmp_path: Path) -> None:
    """``is_ditqasm_source`` sniffs file content for both ``Path`` and path-string input."""
    ditqasm_file = tmp_path / "circuit.qasm"
    ditqasm_file.write_text(DITQASM_STRING, encoding="utf-8")
    assert is_ditqasm_source(ditqasm_file) is True
    assert is_ditqasm_source(str(ditqasm_file)) is True

    openqasm_file = tmp_path / "qubit_circuit.qasm"
    openqasm_file.write_text(OPENQASM_STRING, encoding="utf-8")
    assert is_ditqasm_source(openqasm_file) is False


def test_is_ditqasm_source_nonexistent_path_returns_false(tmp_path: Path) -> None:
    """A ``Path`` that does not exist is never DITQASM."""
    assert is_ditqasm_source(tmp_path / "does_not_exist.qasm") is False


def test_load_qudit_circuit_from_text() -> None:
    """``load_qudit_circuit`` parses raw DITQASM text into an mqt.qudits QuantumCircuit."""
    pytest.importorskip("mqt.qudits")
    circuit = load_qudit_circuit(DITQASM_STRING)
    assert getattr(circuit, "num_qudits") == 2  # noqa: B009
    assert list(getattr(circuit, "dimensions")) == [2, 3]  # noqa: B009


def test_load_qudit_circuit_from_path(tmp_path: Path) -> None:
    """``load_qudit_circuit`` parses a DITQASM file into an mqt.qudits QuantumCircuit."""
    pytest.importorskip("mqt.qudits")
    ditqasm_file = tmp_path / "circuit.qasm"
    ditqasm_file.write_text(DITQASM_STRING, encoding="utf-8")

    circuit = load_qudit_circuit(ditqasm_file)
    assert getattr(circuit, "num_qudits") == 2  # noqa: B009
    assert list(getattr(circuit, "dimensions")) == [2, 3]  # noqa: B009
