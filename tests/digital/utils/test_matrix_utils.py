# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for matrix equivalence utilities."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from mqt.yaqs.digital.utils.matrix_utils import (
    build_composed_operator_tensor,
    check_operator_is_identity,
)


def test_build_composed_operator_tensor_rejects_mismatched_widths() -> None:
    """Different qubit counts raise before tensor construction."""
    with pytest.raises(ValueError, match="same number of qubits"):
        build_composed_operator_tensor(QuantumCircuit(2), QuantumCircuit(3))


def test_mid_circuit_measurement_rejected() -> None:
    """Measurements before the final layer are rejected."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)

    with pytest.raises(ValueError, match="Mid-circuit measurements"):
        build_composed_operator_tensor(qc, qc)


def test_check_operator_is_identity_without_premature_rounding() -> None:
    """Near-identity overlap is not lost to one-decimal rounding."""
    op = np.diag([1.0, 0.99995]).astype(np.complex128)
    tensor = op.reshape((2, 2))
    assert check_operator_is_identity(tensor, fidelity=0.9999) is True
    assert check_operator_is_identity(tensor, fidelity=1.0) is False
