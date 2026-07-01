# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for matrix equivalence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit

from mqt.yaqs import EquivalenceChecker
from mqt.yaqs.core.libraries.gate_library import BaseGate
from mqt.yaqs.digital.utils.matrix_utils import (
    apply_gate_left,
    check_matrix_equivalence,
    compose_operator_tensor,
    is_identity_tensor,
    make_identity_tensor,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_compose_operator_tensor_rejects_mismatched_widths() -> None:
    """Different qubit counts raise before tensor construction."""
    with pytest.raises(ValueError, match="same number of qubits"):
        compose_operator_tensor(QuantumCircuit(2), QuantumCircuit(3))


def test_mid_circuit_measurement_rejected() -> None:
    """Measurements before the final layer are rejected."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)

    with pytest.raises(ValueError, match="Mid-circuit measurements"):
        compose_operator_tensor(qc, qc)


def test_final_measurements_are_stripped() -> None:
    """Terminal measurements are removed before building the composed operator."""
    qc1 = QuantumCircuit(2, 2)
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.measure([0, 1], [0, 1])

    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)

    assert check_matrix_equivalence(qc1, qc2, fidelity=1 - 1e-12) is True


def test_barriers_are_ignored() -> None:
    """Barrier instructions do not affect equivalence."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.barrier()
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)

    assert check_matrix_equivalence(qc1, qc2, fidelity=1 - 1e-12) is True


def test_check_matrix_equivalence_empty_circuits() -> None:
    """Empty circuits compose to the identity."""
    qc = QuantumCircuit(3)
    assert check_matrix_equivalence(qc, qc, fidelity=1 - 1e-12) is True


def test_check_matrix_equivalence_identical_circuits() -> None:
    """Identical non-trivial circuits are equivalent."""
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.rz(np.pi / 4, 2)
    qc2 = qc1.copy()

    assert check_matrix_equivalence(qc1, qc2, fidelity=1 - 1e-12) is True


def test_check_matrix_equivalence_non_equivalent() -> None:
    """An extra gate breaks equivalence."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.x(1)

    assert check_matrix_equivalence(qc1, qc2, fidelity=1 - 1e-12) is False


def test_check_matrix_equivalence_global_phase() -> None:
    """Circuits differing only by global phase are equivalent."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)

    qc2 = qc1.copy()
    qc2.global_phase = np.pi / 5

    assert check_matrix_equivalence(qc1, qc2, fidelity=1 - 1e-12) is True


def test_check_matrix_equivalence_long_range_cx() -> None:
    """Long-range two-qubit gates are supported."""
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.cx(0, 2)

    qc2 = qc1.copy()
    assert check_matrix_equivalence(qc1, qc2, fidelity=1 - 1e-12) is True


def test_check_matrix_equivalence_different_single_qubit_gates() -> None:
    """Distinct single-qubit circuits are not equivalent."""
    qc1 = QuantumCircuit(1)
    qc1.h(0)

    qc2 = QuantumCircuit(1)
    qc2.x(0)

    assert check_matrix_equivalence(qc1, qc2, fidelity=1 - 1e-12) is False


def test_check_matrix_equivalence_disjoint_single_qubit_layer() -> None:
    """A wide single-qubit layer applies all gates in one front layer."""
    qc1 = QuantumCircuit(4)
    qc1.h(0)
    qc1.h(1)
    qc1.h(2)
    qc1.h(3)
    qc1.cx(0, 1)
    qc1.cx(2, 3)

    qc2 = qc1.copy()
    assert check_matrix_equivalence(qc1, qc2, fidelity=1 - 1e-12) is True


def test_check_matrix_equivalence_cx_with_swapped_sites() -> None:
    """Two-qubit gates with reversed site order compose correctly."""
    qc = QuantumCircuit(2)
    qc.cx(1, 0)

    assert check_matrix_equivalence(qc, qc, fidelity=1 - 1e-12) is True


def testapply_gate_left_three_qubit_embedding() -> None:
    """Multi-qubit gates use the dense embedding path in ``apply_gate_left``."""
    swap = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
        ],
        dtype=np.complex128,
    )
    gate = BaseGate(swap)
    gate.interaction = 3
    gate.set_sites(0, 1, 2)

    op = make_identity_tensor(3)
    updated = apply_gate_left(op, gate, 3, dagger=False)
    round_trip = apply_gate_left(updated, gate, 3, dagger=True)
    assert is_identity_tensor(round_trip, fidelity=1 - 1e-12) is True


def test_compose_operator_tensor_is_identity_for_matching_circuits() -> None:
    """Matching circuits produce an identity-like composed operator tensor."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    composed = compose_operator_tensor(qc, qc)
    assert composed.shape == (2, 2, 2, 2)
    assert is_identity_tensor(composed, fidelity=1 - 1e-12) is True


def test_is_identity_tensor_without_premature_rounding() -> None:
    """Near-identity overlap is not lost to one-decimal rounding."""
    op = np.diag([1.0, 0.99995]).astype(np.complex128)
    tensor = op.reshape((2, 2))
    assert is_identity_tensor(tensor, fidelity=0.9999) is True
    assert is_identity_tensor(tensor, fidelity=1.0) is False


def _bell_pair() -> tuple[QuantumCircuit, QuantumCircuit]:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc, qc.copy()


def _bell_with_extra_x() -> tuple[QuantumCircuit, QuantumCircuit]:
    qc1, qc2 = _bell_pair()
    qc2.x(1)
    return qc1, qc2


def _empty_three_qubit() -> tuple[QuantumCircuit, QuantumCircuit]:
    qc = QuantumCircuit(3)
    return qc, qc.copy()


def _long_range_pair() -> tuple[QuantumCircuit, QuantumCircuit]:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 2)
    return qc, qc.copy()


def _global_phase_pair() -> tuple[QuantumCircuit, QuantumCircuit]:
    qc1, qc2 = _bell_pair()
    qc2.global_phase = np.pi / 7
    return qc1, qc2


@dataclass(frozen=True)
class _AgreementCase:
    factory: Callable[[], tuple[QuantumCircuit, QuantumCircuit]]
    expected_equivalent: bool
    name: str


@pytest.mark.parametrize(
    "case",
    [
        _AgreementCase(_empty_three_qubit, expected_equivalent=True, name="empty"),
        _AgreementCase(_bell_pair, expected_equivalent=True, name="bell"),
        _AgreementCase(_bell_with_extra_x, expected_equivalent=False, name="bell_extra_x"),
        _AgreementCase(_long_range_pair, expected_equivalent=True, name="long_range"),
        _AgreementCase(_global_phase_pair, expected_equivalent=True, name="global_phase"),
    ],
    ids=lambda case: case.name,
)
def test_matrix_and_mpo_backends_agree(case: _AgreementCase) -> None:
    """Matrix and MPO backends return the same equivalence verdict."""
    qc1, qc2 = case.factory()
    fidelity = 1 - 1e-12

    matrix = EquivalenceChecker(representation="matrix", fidelity=fidelity).check(qc1, qc2)
    mpo = EquivalenceChecker(representation="mpo", fidelity=fidelity).check(qc1, qc2)

    assert matrix["equivalent"] is case.expected_equivalent
    assert mpo["equivalent"] is case.expected_equivalent
    assert matrix["equivalent"] == mpo["equivalent"]
