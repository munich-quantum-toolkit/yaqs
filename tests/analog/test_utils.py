# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for analog solver utility functions."""

from typing import Any, cast

import numpy as np
import pytest
import scipy.sparse

from mqt.yaqs.analog.utils import (
    _embed_observable_dense,  # noqa: PLC2701
    _embed_observable_sparse,  # noqa: PLC2701
    _embed_operator_dense,  # noqa: PLC2701
    _embed_operator_sparse,  # noqa: PLC2701
    _kron_all_dense,  # noqa: PLC2701
    _kron_all_sparse,  # noqa: PLC2701
)
from mqt.yaqs.core.data_structures.simulation_parameters import Observable


def test_kron_all_dense() -> None:
    """Test Kronecker product of multiple dense matrices."""
    i = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)

    # I x X
    res = _kron_all_dense([i, x])
    expected = np.kron(i, x)
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, expected)

    # X x Z x I
    res = _kron_all_dense([x, z, i])
    expected = np.kron(np.kron(x, z), i)
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, expected)


def test_kron_all_sparse() -> None:
    """Test Kronecker product of sparse matrices."""
    i = scipy.sparse.eye(2, format="csr", dtype=complex)
    x = scipy.sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)

    # I x X
    res = _kron_all_sparse([i, x])
    expected = scipy.sparse.kron(i, x, format="csr")
    assert scipy.sparse.issparse(res)
    assert cast("Any", (res != expected)).nnz == 0


def test_embed_operator_dense_1site() -> None:
    """Test embedding a 1-site matrix operator (dense)."""
    num_sites = 3
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    process = {"sites": [1], "matrix": sigma_x}

    op = _embed_operator_dense(process, num_sites)
    expected = np.kron(np.eye(2), np.kron(sigma_x, np.eye(2)))

    assert isinstance(op, np.ndarray)
    assert np.allclose(op, expected)


def test_embed_operator_sparse_1site() -> None:
    """Test embedding a 1-site matrix operator (sparse)."""
    num_sites = 3
    sigma_x = scipy.sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
    process = {"sites": [1], "matrix": sigma_x}

    op = _embed_operator_sparse(process, num_sites)
    expected = scipy.sparse.kron(scipy.sparse.eye(2), scipy.sparse.kron(sigma_x, scipy.sparse.eye(2)))

    assert scipy.sparse.issparse(op)
    assert cast("Any", (op != expected)).nnz == 0


def test_embed_operator_dense_2site() -> None:
    """Test embedding a 2-site adjacent matrix operator (dense)."""
    num_sites = 4
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    process = {"sites": [1, 2], "matrix": cnot}

    op = _embed_operator_dense(process, num_sites)
    expected = np.kron(np.eye(2), np.kron(cnot, np.eye(2)))

    assert isinstance(op, np.ndarray)
    assert np.allclose(op, expected)


def test_embed_operator_sparse_2site() -> None:
    """Test embedding a 2-site adjacent matrix operator (sparse)."""
    num_sites = 4
    cnot = scipy.sparse.eye(4, format="csr", dtype=complex)
    cnot[2, 2] = 0
    cnot[2, 3] = 1
    cnot[3, 3] = 0
    cnot[3, 2] = 1
    process = {"sites": [1, 2], "matrix": cnot}

    op = _embed_operator_sparse(process, num_sites)
    # Correct construction: I(2) x CNOT x I(2)
    # _embed_operator logic:
    # left_id = eye(2**1) = eye(2)
    # right_id = eye(2**(4-1-2)) = eye(2**1) = eye(2)
    expected = scipy.sparse.kron(scipy.sparse.eye(2), scipy.sparse.kron(cnot, scipy.sparse.eye(2)))

    assert scipy.sparse.issparse(op)
    assert cast("Any", (op != expected)).nnz == 0


def test_embed_operator_errors() -> None:
    """Test error handling."""
    num_sites = 3
    with pytest.raises(NotImplementedError, match="Cannot embed operator"):
        _embed_operator_dense({"sites": [0], "unknown": "value"}, num_sites)


def test_embed_observable_dense_1site() -> None:
    """Test embedding a 1-site observable (dense)."""
    num_sites = 3
    obs = Observable("z", sites=[1])

    op = _embed_observable_dense(obs, num_sites)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    expected = np.kron(np.eye(2), np.kron(z, np.eye(2)))

    assert isinstance(op, np.ndarray)
    assert np.allclose(op, expected)


def test_embed_observable_sparse_1site() -> None:
    """Test embedding a 1-site observable (sparse)."""
    num_sites = 3
    obs = Observable("z", sites=[1])

    op = _embed_observable_sparse(obs, num_sites)
    z = scipy.sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
    expected = scipy.sparse.kron(scipy.sparse.eye(2), scipy.sparse.kron(z, scipy.sparse.eye(2)))

    assert scipy.sparse.issparse(op)
    assert cast("Any", (op != expected)).nnz == 0
