# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for dense Hilbert-space embedding helpers in state_utils."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Pauli

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.data_structures.state_utils import (
    embed_adjacent_two_site_operator,
    embed_one_site_operator,
    embed_two_site_factors,
)
from mqt.yaqs.digital.utils.matrix_utils import embed_unitary


@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("site", [0, 1, 2])
@pytest.mark.parametrize("local", ["X", "Y", "Z"])
def test_embed_one_site_matches_qiskit(length: int, site: int, local: str) -> None:
    """One-site embedding agrees with Qiskit ``Operator`` layout."""
    if site >= length:
        pytest.skip("site out of range for chain length")

    local_mat = np.asarray(Pauli(local).to_matrix(), dtype=np.complex128)
    yaqs = embed_one_site_operator(local_mat, length, site)
    qiskit = embed_unitary(local_mat, [site], length)
    np.testing.assert_allclose(yaqs, qiskit, atol=1e-12)


@pytest.mark.parametrize("length", [3, 4])
@pytest.mark.parametrize("site_left", [0, 1, 2])
def test_embed_adjacent_two_site_matches_qiskit(length: int, site_left: int) -> None:
    """Adjacent two-site embedding agrees with Qiskit ``Operator`` layout."""
    if site_left + 1 >= length:
        pytest.skip("pair out of range for chain length")

    local_mat = np.asarray(CXGate().to_matrix(), dtype=np.complex128)
    yaqs = embed_adjacent_two_site_operator(local_mat, length, site_left)
    qiskit = embed_unitary(local_mat, [site_left, site_left + 1], length)
    np.testing.assert_allclose(yaqs, qiskit, atol=1e-12)


def test_embed_matches_mps_expect_on_haar() -> None:
    """Embedded Pauli expectations match ``MPS.expect`` on an entangled state."""
    length = 3
    mps = MPS(length, state="haar-random", pad=4)
    psi = mps.to_vec()
    for site in range(length):
        for name in ("x", "z"):
            obs = Observable(name, site)
            mps_val = mps.expect(obs)
            op = embed_one_site_operator(np.asarray(obs.gate.matrix, dtype=np.complex128), length, site)
            embed_val = float(np.real(np.vdot(psi, op @ psi)))
            assert mps_val == pytest.approx(embed_val, abs=1e-9)


def test_embed_two_site_factors_non_adjacent() -> None:
    """Non-adjacent factor embedding matches sequential one-site products."""
    length = 3
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    full = embed_two_site_factors(x, z, length, 0, 2)
    expected = embed_one_site_operator(z, length, 2) @ embed_one_site_operator(x, length, 0)
    np.testing.assert_allclose(full, expected, atol=1e-12)
