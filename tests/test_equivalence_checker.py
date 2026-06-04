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
from mqt.yaqs.digital.utils.mpo_utils import MIN_QUBITS_FOR_MPO_PARALLEL

if TYPE_CHECKING:
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


def test_check_accepts_qasm2_path_object() -> None:
    qasm_path = Path(__file__).parent / "circuit.qasm"

    checker = EquivalenceChecker(representation="mpo")
    result = checker.check(qasm_path, qasm_path)
    assert result["equivalent"] is True


def test_check_accepts_qasm2_str_path() -> None:
    qasm_path = str(Path(__file__).parent / "circuit.qasm")

    checker = EquivalenceChecker(representation="mpo")
    result = checker.check(qasm_path, qasm_path)
    assert result["equivalent"] is True


def test_check_qasm_path_vs_quantumcircuit_agree() -> None:
    qasm_path = Path(__file__).parent / "circuit.qasm"
    qc = load(filename=str(qasm_path))
    checker = EquivalenceChecker(representation="mpo")
    result_path = checker.check(qasm_path, qasm_path)
    result_qc = checker.check(qc, qc)
    assert result_path["equivalent"] == result_qc["equivalent"]


def test_check_accepts_qasm3_path_object() -> None:
    pytest.importorskip("qiskit_qasm3_import")
    qasm_file = Path(__file__).parent / "circuit3.qasm"

    checker = EquivalenceChecker(representation="matrix")
    result = checker.check(qasm_file, qasm_file)
    assert result["equivalent"] is True


def test_check_accepts_qasm3_str_path() -> None:
    pytest.importorskip("qiskit_qasm3_import")
    qasm_file = str(Path(__file__).parent / "circuit3.qasm")

    checker = EquivalenceChecker(representation="matrix")
    result = checker.check(qasm_file, qasm_file)
    assert result["equivalent"] is True
