# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: PLC2701 -- white-box tests import private analog.utils helpers

"""Tests for analog solver utility functions."""

from typing import Any, cast

import numpy as np
import pytest
import scipy.sparse

from mqt.yaqs.analog.utils import (
    _embed_observable_dense,
    _embed_observable_sparse,
    _embed_operator_dense,
    _embed_operator_sparse,
    _kron_all_dense,
    _kron_all_sparse,
)
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.data_structures.state_utils import (
    embed_adjacent_two_site_operator,
    embed_one_site_operator,
)


def test_kron_all_dense() -> None:
    """Test Kronecker product of multiple dense matrices."""
    i = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)

    # I then X (site-0 LSB order)
    res = _kron_all_dense([i, x])
    expected = np.kron(x, i)
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, expected)

    # X, Z, I
    res = _kron_all_dense([x, z, i])
    expected = np.kron(i, np.kron(z, x))
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, expected)


def test_kron_all_sparse() -> None:
    """Test Kronecker product of sparse matrices."""
    i = scipy.sparse.eye(2, format="csr", dtype=complex)
    x = scipy.sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)

    # I x X
    res = _kron_all_sparse([i, x])
    expected = scipy.sparse.kron(x, i, format="csr")
    assert scipy.sparse.issparse(res)
    assert cast("Any", (res != expected)).nnz == 0


def test_embed_operator_dense_1site() -> None:
    """Test embedding a 1-site matrix operator (dense)."""
    num_sites = 3
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    process = {"sites": [1], "matrix": sigma_x}

    op = _embed_operator_dense(process, num_sites)
    expected = embed_one_site_operator(sigma_x, num_sites, 1)

    assert isinstance(op, np.ndarray)
    assert np.allclose(op, expected)


def test_embed_operator_sparse_1site() -> None:
    """Test embedding a 1-site matrix operator (sparse)."""
    num_sites = 3
    sigma_x = scipy.sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
    process = {"sites": [1], "matrix": sigma_x}

    op = _embed_operator_sparse(process, num_sites)
    expected = scipy.sparse.csr_matrix(embed_one_site_operator(np.asarray(sigma_x.toarray()), num_sites, 1))

    assert scipy.sparse.issparse(op)
    assert cast("Any", (op != expected)).nnz == 0


def test_embed_operator_dense_2site() -> None:
    """Test embedding a 2-site adjacent matrix operator (dense)."""
    num_sites = 4
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    process = {"sites": [1, 2], "matrix": cnot}

    op = _embed_operator_dense(process, num_sites)
    expected = embed_adjacent_two_site_operator(cnot, num_sites, 1)

    assert isinstance(op, np.ndarray)
    assert np.allclose(op, expected)


def test_embed_operator_sparse_2site() -> None:
    """Test embedding a 2-site adjacent matrix operator (sparse)."""
    num_sites = 4
    cnot_dense = np.eye(4, dtype=complex)
    cnot_dense[2, 2] = 0
    cnot_dense[2, 3] = 1
    cnot_dense[3, 3] = 0
    cnot_dense[3, 2] = 1
    cnot = scipy.sparse.csr_matrix(cnot_dense)
    process = {"sites": [1, 2], "matrix": cnot}

    op = _embed_operator_sparse(process, num_sites)
    expected = scipy.sparse.csr_matrix(embed_adjacent_two_site_operator(cnot_dense, num_sites, 1))

    assert scipy.sparse.issparse(op)
    assert cast("Any", (op != expected)).nnz == 0


def test_embed_operator_errors() -> None:
    """Test error handling."""
    num_sites = 3
    with pytest.raises(NotImplementedError, match="Cannot embed operator"):
        _embed_operator_dense({"sites": [0], "unknown": "value"}, num_sites)

    with pytest.raises(NotImplementedError, match="Cannot embed operator"):
        _embed_operator_sparse({"sites": [0], "unknown": "value"}, num_sites)


def test_embed_observable_dense_1site() -> None:
    """Test embedding a 1-site observable (dense)."""
    num_sites = 3
    obs = Observable("z", sites=[1])

    op = _embed_observable_dense(obs, num_sites)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    expected = embed_one_site_operator(z, num_sites, 1)

    assert isinstance(op, np.ndarray)
    assert np.allclose(op, expected)


def test_embed_observable_sparse_1site() -> None:
    """Test embedding a 1-site observable (sparse)."""
    num_sites = 3
    obs = Observable("z", sites=[1])

    op = _embed_observable_sparse(obs, num_sites)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    expected = scipy.sparse.csr_matrix(embed_one_site_operator(z, num_sites, 1))

    assert scipy.sparse.issparse(op)
    assert cast("Any", (op != expected)).nnz == 0


def test_embed_operator_dense_adjacent_site_order() -> None:
    """Reversed adjacent site lists transpose the local matrix before embedding."""
    num_sites = 4
    rng = np.random.default_rng(0)
    local = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    forward = _embed_operator_dense({"sites": [0, 1], "matrix": local}, num_sites)
    reversed_sites = _embed_operator_dense({"sites": [1, 0], "matrix": local}, num_sites)
    expected_forward = embed_adjacent_two_site_operator(local, num_sites, 0)
    expected_reversed = embed_adjacent_two_site_operator(
        np.asarray(
            local.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4),
            dtype=np.complex128,
        ),
        num_sites,
        0,
    )
    assert np.allclose(forward, expected_forward)
    assert np.allclose(reversed_sites, expected_reversed)
    assert not np.allclose(forward, reversed_sites)


def test_embed_operator_sparse_adjacent_site_order() -> None:
    """Sparse reversed adjacent site lists transpose the local matrix before embedding."""
    num_sites = 4
    rng = np.random.default_rng(1)
    local = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    forward = _embed_operator_sparse({"sites": [0, 1], "matrix": local}, num_sites)
    reversed_sites = _embed_operator_sparse({"sites": [1, 0], "matrix": local}, num_sites)
    expected_forward = scipy.sparse.csr_matrix(embed_adjacent_two_site_operator(local, num_sites, 0))
    expected_reversed = scipy.sparse.csr_matrix(
        embed_adjacent_two_site_operator(
            np.asarray(
                local.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4),
                dtype=np.complex128,
            ),
            num_sites,
            0,
        )
    )
    assert cast("Any", (forward != expected_forward)).nnz == 0
    assert cast("Any", (reversed_sites != expected_reversed)).nnz == 0
    assert cast("Any", (forward != reversed_sites)).nnz > 0


def test_embed_operator_dense_rejects_non_adjacent_pair() -> None:
    """Matrix-based two-site embedding requires neighboring sites."""
    num_sites = 4
    with pytest.raises(ValueError, match="adjacent"):
        _embed_operator_dense({"sites": [0, 2], "matrix": np.eye(4, dtype=complex)}, num_sites)
