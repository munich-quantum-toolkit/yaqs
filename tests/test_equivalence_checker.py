# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for circuit equivalence checking.

This module provides unit tests for :class:`~mqt.yaqs.EquivalenceChecker`. It verifies
the MPO and dense matrix backends by comparing quantum circuits, including automatic
backend selection and global-phase equivalence.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.qasm2 import load

from mqt.yaqs import DEFAULT_MATRIX_MAX_QUBITS, EquivalenceChecker

if TYPE_CHECKING:
    from mqt.yaqs.equivalence_checker import EquivalenceRepresentation


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


def test_large_equivalence() -> None:
    """Test large-scale equivalence.

    This test creates a large quantum circuit with multiple CNOT gates, Ry gates, and an Rzz gate.
    This should verify nearly all parts of the equivalence checking algorithm.
    """
    qasm_path = Path(__file__).parent / "circuit.qasm"
    qc = load(filename=str(qasm_path))

    checker = EquivalenceChecker(representation="mpo")
    result = checker.check(qc, qc)
    assert result["equivalent"] is True, "Large scale test fails. Circuits should be equivalent."
    assert result["representation"] == "mpo"


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
        representation=cast("EquivalenceRepresentation", representation),
    )
    equal_result = checker.check(qc_equal_a, qc_equal_b)
    diff_result = checker.check(qc_equal_a, qc_diff_b)
    assert equal_result["equivalent"] is True
    assert diff_result["equivalent"] is False
    assert equal_result["representation"] == representation


def test_global_phase_equivalence_matrix() -> None:
    """Circuits differing by global phase are equivalent under the matrix backend."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = qc1.copy()
    qc2.global_phase = np.pi / 3

    checker = EquivalenceChecker(representation="matrix", fidelity=1 - 1e-13)
    result = checker.check(qc1, qc2)
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
