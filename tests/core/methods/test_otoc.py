# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for dense clean-circuit OTOC helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.methods.otoc import (
    averaged_pair_otoc,
    cross_cut_otoc,
    cross_cut_otoc_by_depth,
    embed_single_site_operator,
    pair_pauli_otoc,
    pauli_matrix,
)


def test_pauli_matrices_are_valid() -> None:
    """Test Pauli matrix shapes, involutions, Hermiticity, and validation."""
    identity = pauli_matrix("I")

    for label in ("X", "Y", "Z"):
        pauli = pauli_matrix(label)
        assert pauli.shape == (2, 2)
        np.testing.assert_allclose(pauli @ pauli, identity)
        np.testing.assert_allclose(pauli.conj().T, pauli)

    with pytest.raises(ValueError, match="Unsupported Pauli label"):
        pauli_matrix("A")


def test_embed_single_site_operator_uses_leftmost_site_convention() -> None:
    """Test single-site embedding with site 0 as the leftmost Kronecker factor."""
    identity = pauli_matrix("I")
    x_matrix = pauli_matrix("X")

    left_x = embed_single_site_operator(x_matrix, site=0, num_qubits=2)
    right_x = embed_single_site_operator(x_matrix, site=1, num_qubits=2)

    np.testing.assert_allclose(left_x, np.kron(x_matrix, identity))
    np.testing.assert_allclose(right_x, np.kron(identity, x_matrix))
    assert left_x.shape == (4, 4)
    assert right_x.shape == (4, 4)
    np.testing.assert_allclose(left_x.conj().T, left_x)
    np.testing.assert_allclose(right_x.conj().T, right_x)

    with pytest.raises(ValueError, match="site must satisfy"):
        embed_single_site_operator(x_matrix, site=2, num_qubits=2)

    with pytest.raises(ValueError, match="local_op must have shape"):
        embed_single_site_operator(np.eye(3), site=0, num_qubits=2)


def test_pair_otoc_identity_unitary_commutes_across_different_sites() -> None:
    """Test that distinct-site Pauli operators commute under identity evolution."""
    unitary = np.eye(4, dtype=np.complex128)

    xz_otoc = pair_pauli_otoc(unitary, 0, 1, "X", "Z", num_qubits=2)
    xx_otoc = pair_pauli_otoc(unitary, 0, 1, "X", "X", num_qubits=2)

    assert xz_otoc == pytest.approx(0)
    assert xx_otoc == pytest.approx(0)


def test_same_site_noncommuting_otoc_is_nonzero() -> None:
    """Test same-site noncommuting Paulis and raw versus bounded normalization."""
    unitary = np.eye(4, dtype=np.complex128)

    bounded = pair_pauli_otoc(unitary, 0, 0, "X", "Z", num_qubits=2, normalization="bounded")
    raw = pair_pauli_otoc(unitary, 0, 0, "X", "Z", num_qubits=2, normalization="raw")

    assert bounded > 0
    assert bounded == pytest.approx(raw / 2)


def test_cross_cut_otoc_returns_expected_keys() -> None:
    """Test cross-cut return keys and symmetric average definition."""
    unitary = np.eye(4, dtype=np.complex128)

    result = cross_cut_otoc(unitary, left_sites=[0], right_sites=[1], num_qubits=2)

    assert set(result) == {"left_to_right", "right_to_left", "symmetric"}
    assert result["symmetric"] == pytest.approx(0.5 * (result["left_to_right"] + result["right_to_left"]))


def test_entangling_unitary_produces_cross_site_otoc() -> None:
    """Test that a simple CNOT-like unitary creates a nonzero cross-site OTOC."""
    cnot_left_control = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.complex128,
    )

    otoc = pair_pauli_otoc(cnot_left_control, 0, 1, "X", "Z", num_qubits=2)
    cross_cut = cross_cut_otoc(cnot_left_control, left_sites=[0], right_sites=[1], num_qubits=2)

    assert otoc > 0
    assert cross_cut["symmetric"] > 0


def test_input_validation() -> None:
    """Test OTOC input validation errors."""
    unitary = np.eye(4, dtype=np.complex128)

    with pytest.raises(ValueError, match="Unsupported Pauli label"):
        pair_pauli_otoc(unitary, 0, 1, "A", "Z", num_qubits=2)

    with pytest.raises(ValueError, match="normalization must be"):
        pair_pauli_otoc(unitary, 0, 1, "X", "Z", num_qubits=2, normalization="scaled")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="site must satisfy"):
        pair_pauli_otoc(unitary, 2, 1, "X", "Z", num_qubits=2)

    with pytest.raises(ValueError, match="unitary must have shape"):
        pair_pauli_otoc(np.eye(2), 0, 1, "X", "Z", num_qubits=2)

    with pytest.raises(ValueError, match="left_sites must be non-empty"):
        cross_cut_otoc(unitary, left_sites=[], right_sites=[1], num_qubits=2)


def test_averaged_pair_and_depth_series_helpers() -> None:
    """Test averaged pair OTOC and deterministic depth-series output."""
    identity = np.eye(4, dtype=np.complex128)
    cnot_left_control = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.complex128,
    )

    assert averaged_pair_otoc(identity, 0, 1, num_qubits=2) == pytest.approx(0)

    rows = cross_cut_otoc_by_depth(
        unitary_by_depth={2: cnot_left_control, 1: identity},
        left_sites=range(1),
        right_sites=range(1, 2),
        num_qubits=2,
    )

    assert [row["depth"] for row in rows] == [1, 2]
    assert rows[0]["symmetric"] == pytest.approx(0)
    assert rows[1]["symmetric"] > 0
