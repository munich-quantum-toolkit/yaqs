# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QASM loading utilities shared by the Simulator and EquivalenceChecker."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qiskit import qasm2, qasm3

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def _first_non_comment_line(text: str) -> str:
    """Return the first non-empty, non-comment line from QASM-like text.

    Args:
        text: Multiline string to scan.

    Returns:
        The first line that is not empty and does not start with ``//``,
        or an empty string if no such line exists.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("//"):
            return stripped
    return ""


def load_circuit(circuit: QuantumCircuit | str | Path) -> QuantumCircuit:
    """Load a QuantumCircuit from a QASM string, file path, or return it unchanged.

    Args:
        circuit: A ``QuantumCircuit``, a raw QASM string, or a path to a ``.qasm`` file.

    Returns:
        The corresponding ``QuantumCircuit``.
    """
    if not isinstance(circuit, (str, Path)):
        return circuit

    if isinstance(circuit, str):
        header = _first_non_comment_line(circuit)
        if header.startswith("OPENQASM"):
            if header.startswith("OPENQASM 3"):  # pragma: no cover
                return qasm3.loads(circuit)
            return qasm2.loads(circuit)

    path = Path(circuit)
    content = path.read_text(encoding="utf-8")
    if _first_non_comment_line(content).startswith("OPENQASM 3"):  # pragma: no cover
        return qasm3.load(str(path))
    return qasm2.load(str(path))
