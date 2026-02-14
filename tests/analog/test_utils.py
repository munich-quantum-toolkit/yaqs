# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for analog solver utility functions."""

import numpy as np
import pytest

from mqt.yaqs.analog.utils import _embed_observable, _embed_operator, _kron_all
from mqt.yaqs.core.data_structures.simulation_parameters import Observable


def test_kron_all() -> None:
    """Test Kronecker product of multiple matrices."""
    i = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    z = np.array([[1, 0], [0, -1]])

    # I x X
    res = _kron_all([i, x])
    expected = np.kron(i, x)
    assert np.allclose(res, expected)

    # X x Z x I
    res = _kron_all([x, z, i])
    expected = np.kron(np.kron(x, z), i)
    assert np.allclose(res, expected)


def test_embed_operator_matrix_1site() -> None:
    """Test embedding a 1-site matrix operator."""
    num_sites = 3
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    process = {"sites": [1], "matrix": sigma_x}

    # Expected: I x X x I
    op = _embed_operator(process, num_sites)
    expected = np.kron(np.eye(2), np.kron(sigma_x, np.eye(2)))

    assert np.allclose(op, expected)


def test_embed_operator_matrix_2site_adjacent() -> None:
    """Test embedding a 2-site adjacent matrix operator."""
    num_sites = 4
    # CNOT on 1, 2
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    process = {"sites": [1, 2], "matrix": cnot}

    # Expected: I x CNOT x I
    op = _embed_operator(process, num_sites)
    expected = np.kron(np.eye(2), np.kron(cnot, np.eye(2)))

    assert np.allclose(op, expected)


def test_embed_operator_factors() -> None:
    """Test embedding a factor-based operator (e.g. X_0 Z_2)."""
    num_sites = 3
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    process = {"sites": [0, 2], "factors": (x, z)}

    # Expected: X x I x Z
    op = _embed_operator(process, num_sites)
    expected = np.kron(x, np.kron(np.eye(2), z))

    assert np.allclose(op, expected)


def test_embed_operator_errors() -> None:
    """Test error handling in _embed_operator."""
    num_sites = 3

    # Unknown process type
    with pytest.raises(NotImplementedError, match="Cannot embed operator"):
        _embed_operator({"sites": [0], "unknown": "value"}, num_sites)

    # 2-site matrix non-adjacent
    with pytest.raises(AssertionError, match="must be adjacent"):
        cnot = np.eye(4)
        _embed_operator({"sites": [0, 2], "matrix": cnot}, num_sites)


def test_embed_observable_1site() -> None:
    """Test embedding a 1-site observable."""
    num_sites = 3
    obs = Observable("z", sites=[1])

    # Expected: I x Z x I
    op = _embed_observable(obs, num_sites)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    expected = np.kron(np.eye(2), np.kron(z, np.eye(2)))

    assert np.allclose(op, expected)


class DummyGate:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix


class DummyObservable:
    def __init__(self, sites: int | list[int], matrix: np.ndarray) -> None:
        self.sites = sites
        self.gate = DummyGate(matrix)


def test_embed_observable_2site_adjacent() -> None:
    """Test embedding a 2-site adjacent observable."""
    num_sites = 3
    # Use dummy observable to avoid GateLibrary validation
    sites = [0, 1]
    matrix = np.eye(4, dtype=complex)
    obs = DummyObservable(sites, matrix)

    op = _embed_observable(obs, num_sites)  # type: ignore[arg-type]
    # Expected: I4 x I
    expected = np.kron(np.eye(4), np.eye(2))
    assert np.allclose(op, expected)


def test_embed_observable_errors() -> None:
    """Test error handling in _embed_observable."""
    num_sites = 3

    # Non-adjacent 2-site
    with pytest.raises(NotImplementedError, match="Non-adjacent"):
        obs = DummyObservable(sites=[0, 2], matrix=np.eye(4))
        _embed_observable(obs, num_sites)  # type: ignore[arg-type]

    # >2 sites
    with pytest.raises(NotImplementedError, match="Unsupported observable site count"):
        obs = DummyObservable(sites=[0, 1, 2], matrix=np.eye(8))
        _embed_observable(obs, num_sites)  # type: ignore[arg-type]
