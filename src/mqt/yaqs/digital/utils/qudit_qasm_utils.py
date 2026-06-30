# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""DITQASM 2.0 (mqt-qudits qudit QASM dialect) loading utilities."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

_DITQASM_HEADER_RE = re.compile(r"DITQASM\s+(\d+(?:\.\d+)?)", re.IGNORECASE)


def _first_substantive_line(text: str) -> str | None:
    """Return the first non-blank, non-comment line of QASM-like source text."""
    in_block_comment = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
                remainder = stripped.split("*/", maxsplit=1)[1].strip()
                if not remainder or remainder.startswith("//"):
                    continue
                stripped = remainder
            else:
                continue
        if stripped.startswith("//"):
            continue
        if "/*" in stripped:
            if "*/" in stripped:
                remainder = stripped.split("*/", maxsplit=1)[1].strip()
                if not remainder or remainder.startswith("//"):
                    continue
                stripped = remainder
            else:
                in_block_comment = True
                continue
        return stripped
    return None


def is_ditqasm_source(operator: str | Path) -> bool:
    """Return True if ``operator`` is (or points to) DITQASM 2.0 source text.

    Accepts raw DITQASM text, or a filesystem path (existing file) whose content
    is sniffed. Never raises -- returns False for anything that can't be read.
    """
    if isinstance(operator, Path):
        if not operator.is_file():
            return False
        text = operator.read_text(encoding="utf-8")
    else:
        path = Path(operator)
        text = path.read_text(encoding="utf-8") if path.is_file() else operator

    line = _first_substantive_line(text)
    return line is not None and bool(_DITQASM_HEADER_RE.match(line))


def load_qudit_circuit(operator: str | Path) -> object:  # pragma: no cover - requires optional mqt-qudits
    """Load an mqt.qudits QuantumCircuit from DITQASM source text or a file path.

    ``mqt.qudits`` is loaded dynamically via :mod:`importlib`: this module must never
    be statically imported from a ``ty``-checked caller without going through it.

    Args:
        operator: Raw DITQASM source text, or a filesystem path to a ``.qasm`` file.

    Returns:
        The corresponding ``mqt.qudits.quantum_circuit.QuantumCircuit``.
    """
    quantum_circuit_module = importlib.import_module("mqt.qudits.quantum_circuit.circuit")
    qudit_circuit = quantum_circuit_module.QuantumCircuit()

    if isinstance(operator, Path):
        qudit_circuit.load_from_file(str(operator))
        return qudit_circuit

    path = Path(operator)
    if path.is_file():
        qudit_circuit.load_from_file(str(path))
    else:
        qudit_circuit.from_qasm(operator)
    return qudit_circuit
