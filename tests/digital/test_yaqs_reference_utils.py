# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for YAQS vs Qiskit reference convention helpers."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit

from scripts.benchmark_long_range_tdvp_regimes import validate_reference_convention
from scripts.yaqs_reference_utils import bit_reverse_vec, compare_statevectors, qiskit_reference_vec


def test_bit_reverse_is_involution() -> None:
    n = 3
    vec = np.arange(2**n, dtype=np.complex128)
    np.testing.assert_allclose(bit_reverse_vec(bit_reverse_vec(vec, n), n), vec)


def test_validate_reference_convention_passes() -> None:
    validate_reference_convention(chi=64)


def test_plus_state_fidelity_direct() -> None:
    from scripts.benchmark_utils import _prep_initial_state
    from scripts.benchmark_long_range_tdvp_regimes import _run_validation_circuit

    n = 4
    prep = QuantumCircuit(n)
    _prep_initial_state(prep, "plus", seed=0)
    ref = qiskit_reference_vec(prep)
    mps = _run_validation_circuit(QuantumCircuit(n), n, chi=32)
    vec = np.asarray(mps.to_vec(), dtype=np.complex128)
    f_dir, f_rev, conv = compare_statevectors(vec, ref, n=n)
    assert conv == "direct"
    assert f_dir > 1.0 - 1e-10
