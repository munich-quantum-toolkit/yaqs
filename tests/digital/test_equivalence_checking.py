# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for MPO-based equivalence checking implementation.

This module provides unit tests for the equivalence checker implemented in
mqt.yaqs.circuits.equivalence_checker. It verifies the correctness of the
MPO-based equivalence algorithm by comparing quantum circuits. Tests include
checks for:
  - Identity circuits (empty circuits) which should be equivalent.
  - Two-qubit circuits that implement the same operation (e.g., Bell state preparation).
  - Two-qubit circuits that differ by an extra gate, which should be non-equivalent.
  - Long-range circuits, ensuring that circuits with identical long-range interactions
    are equivalent, while those with additional gates are not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qiskit import QuantumCircuit
from qiskit.qasm2 import load

from mqt.yaqs.digital.equivalence_checker import run


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

    result = run(qc1, qc2, threshold=threshold, fidelity=fidelity)
    assert result["equivalent"] is True, "Empty circuits (identities) should be equivalent."
    assert result["elapsed_time"] >= 0


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

    result = run(qc1, qc2, threshold=1e-13, fidelity=1 - 1e-13)
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

    result = run(qc1, qc2, threshold=1e-13, fidelity=1 - 1e-13)
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

    result = run(qc1, qc2, threshold=1e-13, fidelity=1 - 1e-13)
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

    result = run(qc1, qc2, threshold=1e-13, fidelity=1 - 1e-13)
    assert result["equivalent"] is False, "Extra gate should break equivalence."


def test_large_equivalence() -> None:
    """Test large-scale equivalence.

    This test creates a large quantum circuit with multiple CNOT gates, Ry gates, and an Rzz gate.
    This should verify nearly all parts of the equivalence checking algorithm.
    """
    qasm_path = Path(__file__).parent / "circuit.qasm"
    qc = load(filename=str(qasm_path))

    result = run(qc, qc)
    assert result["equivalent"] is True, "Large scale test fails. Circuits should be equivalent."
