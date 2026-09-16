# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff:file-ignore[import-private-name] -- white-box tests import private analog.utils helpers

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
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.observable import Observable
from mqt.yaqs.core.data_structures.state_utils import (
    embed_one_site_operator,
)
from tests.site_order_reference import embed_local_factors, embed_local_operator


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
    expected = embed_local_operator(cnot, (1, 2), (2,) * num_sites)

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
    expected = scipy.sparse.csr_matrix(embed_local_operator(cnot_dense, (1, 2), (2,) * num_sites))

    assert scipy.sparse.issparse(op)
    assert cast("Any", (op != expected)).nnz == 0


def test_embed_descending_longrange_factors_agrees_for_dense_and_sparse() -> None:
    """Long-range factors retain their original site assignment after sorting."""
    factor_on_site_3 = np.array([[0, 1 + 1j], [2, 0]], dtype=np.complex128)
    factor_on_site_1 = np.array([[1, 2j], [0, -1]], dtype=np.complex128)
    original_sites = (3, 1)
    original_factors = (factor_on_site_3, factor_on_site_1)
    noise_model = NoiseModel([
        {
            "name": "custom_longrange",
            "sites": list(original_sites),
            "strength": 0.2,
            "factors": original_factors,
        }
    ])
    process = noise_model.processes[0]
    expected = embed_local_factors(original_factors, original_sites, (2, 2, 2, 2))

    dense = _embed_operator_dense(process, 4)
    sparse = _embed_operator_sparse(process, 4)

    np.testing.assert_allclose(dense, expected, atol=1e-12)
    np.testing.assert_allclose(cast("Any", sparse).toarray(), expected, atol=1e-12)


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
    obs = Observable("z", sites=1)

    op = _embed_observable_dense(obs, num_sites)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    expected = embed_one_site_operator(z, num_sites, 1)

    assert isinstance(op, np.ndarray)
    assert np.allclose(op, expected)


def test_embed_observable_sparse_1site() -> None:
    """Test embedding a 1-site observable (sparse)."""
    num_sites = 3
    obs = Observable("z", sites=1)

    op = _embed_observable_sparse(obs, num_sites)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    expected = scipy.sparse.csr_matrix(embed_one_site_operator(z, num_sites, 1))

    assert scipy.sparse.issparse(op)
    assert cast("Any", (op != expected)).nnz == 0


@pytest.mark.parametrize("sites", [[0, 1], [1, 0], [2, 0], [0, 2]])
def test_embed_asymmetric_observable_agrees_for_dense_and_sparse(sites: list[int]) -> None:
    """Adjacent and periodic observable factors retain their listed sites."""
    pauli_z = np.diag([1, -1]).astype(np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    local = np.kron(pauli_z, identity)
    observable = Observable(local, sites=sites)
    expected = embed_local_operator(local, tuple(sites), (2, 2, 2))

    dense = _embed_observable_dense(observable, 3)
    sparse = _embed_observable_sparse(observable, 3)

    np.testing.assert_allclose(dense, expected, atol=1e-12)
    np.testing.assert_allclose(cast("Any", sparse).toarray(), expected, atol=1e-12)


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_embed_periodic_observable_rejects_nonqubit_chain(kind: str) -> None:
    """Periodic observables have the same all-qubit limit on each backend."""
    embed = _embed_observable_dense if kind == "dense" else _embed_observable_sparse
    observable = Observable(np.eye(4, dtype=np.complex128), sites=[2, 0])

    with pytest.raises(ValueError, match="require qubit sites throughout the system"):
        embed(observable, 3, physical_dimensions=[2, 3, 2])


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_embed_observable_requires_local_operator(kind: str) -> None:
    """Observable embedding rejects requests without a local matrix and sites."""
    embed = _embed_observable_dense if kind == "dense" else _embed_observable_sparse
    with pytest.raises(ValueError, match="requires an operator with explicit sites"):
        embed(Observable("000"), 3)


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_embed_observable_rejects_more_than_two_sites(kind: str) -> None:
    """Dense and sparse backends reject unsupported three-site observables."""
    embed = _embed_observable_dense if kind == "dense" else _embed_observable_sparse
    with pytest.raises(NotImplementedError, match="Unsupported observable site count: 3"):
        embed(Observable(np.eye(8), [0, 1, 2]), 3)


def test_embed_operator_dense_rejects_non_adjacent_pair() -> None:
    """Matrix-based two-site embedding requires neighboring sites."""
    num_sites = 4
    with pytest.raises(ValueError, match="nearest neighbors or the periodic wrap"):
        _embed_operator_dense({"sites": [0, 2], "matrix": np.eye(4, dtype=complex)}, num_sites)


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_embed_periodic_operator_rejects_invalid_matrix_shape(kind: str) -> None:
    """Periodic-wrap matrices must act on two qubits."""
    embed = _embed_operator_dense if kind == "dense" else _embed_operator_sparse

    with pytest.raises(ValueError, match=r"Periodic-wrap matrix must have shape \(4, 4\)"):
        embed({"sites": [0, 2], "matrix": np.eye(2, dtype=np.complex128)}, 3)


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_embed_operator_rejects_duplicate_matrix_sites(kind: str) -> None:
    """A two-site matrix requires two distinct sites."""
    embed = _embed_operator_dense if kind == "dense" else _embed_operator_sparse

    with pytest.raises(ValueError, match="Two-site matrix sites must be distinct"):
        embed({"sites": [1, 1], "matrix": np.eye(4, dtype=np.complex128)}, 3)


def test_embed_operator_sparse_rejects_out_of_range_matrix_sites() -> None:
    """Sparse matrix embedding rejects invalid adjacent site indices."""
    num_sites = 4
    op4 = scipy.sparse.eye(4, format="csc", dtype=complex)
    with pytest.raises(ValueError, match="site -1 out of range"):
        _embed_operator_sparse({"sites": [-1, 0], "matrix": op4}, num_sites)
    with pytest.raises(ValueError, match="site 4 out of range"):
        _embed_operator_sparse({"sites": [3, 4], "matrix": op4}, num_sites)


def test_embed_operator_sparse_rejects_invalid_factor_sites() -> None:
    """Sparse factor embedding rejects duplicate and out-of-range sites."""
    num_sites = 3
    op = scipy.sparse.eye(2, format="coo", dtype=complex)
    with pytest.raises(ValueError, match="site1 and site2 must differ"):
        _embed_operator_sparse({"sites": [1, 1], "factors": (op, op)}, num_sites)
    with pytest.raises(ValueError, match="site 3 out of range"):
        _embed_operator_sparse({"sites": [0, 3], "factors": (op, op)}, num_sites)
