# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from mqt.yaqs.circuits.equivalence_checker import run


@pytest.mark.parametrize(("threshold", "fidelity"), [(1e-13, 1 - 1e-13), (1e-1, 1 - 1e-3)])
def test_identity_vs_identity(threshold, fidelity) -> None:
    """Two empty (no-gate) circuits on the same number of qubits
    should be equivalent.
    """
    num_qubits = 2
    qc1 = QuantumCircuit(num_qubits)
    qc2 = QuantumCircuit(num_qubits)

    result = run(qc1, qc2, threshold=threshold, fidelity=fidelity)
    assert result["equivalent"] is True, "Empty circuits (identities) should be equivalent."
    assert result["elapsed_time"] >= 0


# # def test_single_qubit_equivalence():
# #     """
# #     Single-qubit test: applying the same gate sequence
# #     on each circuit should yield equivalence.
# #     """
# #     qc1 = QuantumCircuit(1)
# #     qc1.h(0)
# #     qc1.x(0)

# #     qc2 = QuantumCircuit(1)
# #     qc2.h(0)
# #     qc2.x(0)

# #     result = run(qc1, qc2, threshold=1e-13, fidelity=1-1e-13)
# #     assert result['equivalent'] is True, "Identical single-qubit circuits must be equivalent."


# # def test_single_qubit_non_equivalence():
# #     """
# #     Single-qubit test: applying different gates should
# #     yield non-equivalent circuits.
# #     """
# #     qc1 = QuantumCircuit(1)
# #     qc1.h(0)

# #     qc2 = QuantumCircuit(1)
# #     qc2.x(0)

# #     result = run(qc1, qc2, threshold=1e-13, fidelity=1-1e-13)
# #     assert result['equivalent'] is False, "Different single-qubit gates (H vs X) are not equivalent."


def test_two_qubit_equivalence() -> None:
    """Two-qubit circuits that implement the same logical operation
    should be equivalent. Here we create a simple entangling circuit
    in two different ways but ensuring the final unitary is the same.
    """
    qc1 = QuantumCircuit(2)
    # Method 1: Prepare a Bell state
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(2)
    # Method 2: The same sequence (identical in this case)
    qc2.h(0)
    qc2.cx(0, 1)

    result = run(qc1, qc2, threshold=1e-13, fidelity=1 - 1e-13)
    assert result["equivalent"] is True, "Identical 2-qubit circuits must be equivalent."


def test_two_qubit_non_equivalence() -> None:
    """Two-qubit circuits that differ by an extra gate
    or a different gate location are not equivalent.
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
    """Two-qubit circuits that differ by an extra gate
    or a different gate location are not equivalent.
    """
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.cx(0, 2)

    qc2 = QuantumCircuit(3)
    qc2.h(0)
    qc2.cx(0, 2)

    result = run(qc1, qc2, threshold=1e-13, fidelity=1 - 1e-13)
    assert result["equivalent"] is True, "Long-range test not equivalent."


def test_long_range_non_equivalence() -> None:
    """Two-qubit circuits that differ by an extra gate
    or a different gate location are not equivalent.
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
