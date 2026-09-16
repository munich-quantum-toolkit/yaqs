# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for analog solvers (Lindblad and MCWF)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.sparse
from scipy.sparse import issparse

from ..core.data_structures.state_utils import (
    _swap_two_site_factor_order,
    embed_adjacent_two_site_operator,
    embed_one_site_operator,
    embed_two_site_factors,
    resolve_physical_dimensions,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.observable import Observable


def _kron_all_dense(
    ops: list[NDArray[np.complex128]],
) -> NDArray[np.complex128]:
    """Kronecker product of site-local factors in MPS ``to_vec`` order (site 0 = LSB).

    Args:
        ops: Local operators ``[op_0, op_1, ..., op_{L-1}]``.

    Returns:
        The resulting dense matrix.
    """
    res = ops[0]
    for op in ops[1:]:
        res = np.kron(op, res)
    return np.asarray(res, dtype=complex)


def _kron_all_sparse(
    ops: list[NDArray[np.complex128] | scipy.sparse.spmatrix],
) -> scipy.sparse.spmatrix:
    """Sparse Kronecker product in MPS ``to_vec`` order (site 0 = LSB).

    Args:
        ops: Local operators for each site.

    Returns:
        The resulting sparse matrix (CSR format).
    """
    res = ops[0] if issparse(ops[0]) else scipy.sparse.csr_matrix(ops[0])

    for op in ops[1:]:
        op_csr = op if issparse(op) else scipy.sparse.csr_matrix(op)
        res = scipy.sparse.kron(op_csr, res, format="csr")

    return cast("scipy.sparse.spmatrix", res)


def _to_dense(op: NDArray[np.complex128] | scipy.sparse.spmatrix) -> NDArray[np.complex128]:
    """Convert a sparse or dense matrix to a dense NumPy array.

    Args:
        op: Input operator as a dense array or SciPy sparse matrix.

    Returns:
        Dense ``complex128`` array with the same entries as ``op``.
    """
    if issparse(op):
        return np.asarray(cast("Any", op).toarray(), dtype=np.complex128)
    return np.asarray(op, dtype=np.complex128)


def _to_sparse_csr(op: NDArray[np.complex128] | scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    """Convert a dense or sparse matrix to CSR sparse format.

    Args:
        op: Input operator as a dense array or SciPy sparse matrix.

    Returns:
        CSR sparse matrix with the same entries as ``op``.
    """
    if issparse(op):
        return cast("Any", op).tocsr()
    return scipy.sparse.csr_matrix(op)


def _sparse_identity(dim: int) -> scipy.sparse.spmatrix:
    """Return a CSR identity matrix of the given local dimension.

    Args:
        dim: Hilbert-space dimension of a single site.

    Returns:
        CSR sparse identity of shape ``(dim, dim)``.
    """
    return scipy.sparse.identity(dim, format="csr", dtype=np.complex128)


def _embed_one_site_sparse(
    op: NDArray[np.complex128] | scipy.sparse.spmatrix,
    num_sites: int,
    site: int,
    dims: list[int],
) -> scipy.sparse.spmatrix:
    """Embed a one-site operator into the full Hilbert space without densifying.

    Args:
        op: Local operator on ``site``.
        num_sites: Total number of sites in the chain.
        site: Site index on which ``op`` acts.
        dims: Per-site Hilbert-space dimensions.

    Returns:
        CSR sparse embedded operator on the full space.

    Raises:
        ValueError: If ``op`` does not have shape ``(dims[site], dims[site])``.
    """
    op_csr = _to_sparse_csr(op)
    site_dim = dims[site]
    if op_csr.shape != (site_dim, site_dim):
        msg = f"op must have shape ({site_dim}, {site_dim}), got {op_csr.shape}."
        raise ValueError(msg)
    res = scipy.sparse.csr_matrix([[1.0]], dtype=np.complex128)
    for k in range(num_sites):
        local = op_csr if k == site else _sparse_identity(dims[k])
        res = scipy.sparse.kron(local, res, format="csr")
    return cast("scipy.sparse.spmatrix", res)


def _embed_adjacent_two_site_sparse(
    op: NDArray[np.complex128] | scipy.sparse.spmatrix,
    num_sites: int,
    site_left: int,
    dims: list[int],
) -> scipy.sparse.spmatrix:
    """Embed an adjacent two-site operator without densifying the full Hilbert space.

    The first tensor factor of ``op`` acts on ``site_left`` and the second
    factor acts on ``site_left + 1``. The full matrix uses site-0-LSB order.

    Args:
        op: Local operator on the pair ``(site_left, site_left + 1)``.
        num_sites: Total number of sites in the chain.
        site_left: Left site index of the adjacent pair.
        dims: Per-site Hilbert-space dimensions.

    Returns:
        CSR sparse embedded operator on the full space.

    Raises:
        ValueError: If ``op`` does not have the expected pair shape.
    """
    site_right = site_left + 1
    pair_dim = dims[site_left] * dims[site_right]
    op_csr = _to_sparse_csr(op)
    if op_csr.shape != (pair_dim, pair_dim):
        msg = f"op4 must have shape ({pair_dim}, {pair_dim}), got {op_csr.shape}."
        raise ValueError(msg)
    pair_op = scipy.sparse.csr_matrix(_swap_two_site_factor_order(_to_dense(op_csr), dims[site_left], dims[site_right]))
    res = scipy.sparse.csr_matrix([[1.0]], dtype=np.complex128)
    site = 0
    while site < num_sites:
        if site == site_left:
            res = scipy.sparse.kron(pair_op, res, format="csr")
            site += 2
        else:
            res = scipy.sparse.kron(_sparse_identity(dims[site]), res, format="csr")
            site += 1
    return cast("scipy.sparse.spmatrix", res)


def _embed_two_site_factors_sparse(
    op1: NDArray[np.complex128] | scipy.sparse.spmatrix,
    op2: NDArray[np.complex128] | scipy.sparse.spmatrix,
    num_sites: int,
    site1: int,
    site2: int,
    dims: list[int],
) -> scipy.sparse.spmatrix:
    """Embed a product of local operators on two sites without densifying.

    Args:
        op1: Local operator on ``site1``.
        op2: Local operator on ``site2``.
        num_sites: Total number of sites in the chain.
        site1: First site index.
        site2: Second site index.
        dims: Per-site Hilbert-space dimensions.

    Returns:
        CSR sparse embedded operator on the full space.

    Raises:
        ValueError: If either local operator does not match its site dimension.
    """
    op1_csr = _to_sparse_csr(op1)
    op2_csr = _to_sparse_csr(op2)
    if op1_csr.shape != (dims[site1], dims[site1]) or op2_csr.shape != (dims[site2], dims[site2]):
        msg = (
            f"local operators must match site dimensions "
            f"({dims[site1]}, {dims[site1]}) and ({dims[site2]}, {dims[site2]})."
        )
        raise ValueError(msg)
    res = scipy.sparse.csr_matrix([[1.0]], dtype=np.complex128)
    for k in range(num_sites):
        if k == site1:
            local = op1_csr
        elif k == site2:
            local = op2_csr
        else:
            local = _sparse_identity(dims[k])
        res = scipy.sparse.kron(local, res, format="csr")
    return cast("scipy.sparse.spmatrix", res)


def _embed_periodic_two_site_matrix(
    op: NDArray[np.complex128] | scipy.sparse.spmatrix,
    num_sites: int,
    site1: int,
    site2: int,
    dims: list[int],
    *,
    sparse: bool,
) -> NDArray[np.complex128] | scipy.sparse.spmatrix:
    """Embed a periodic-wrap two-qubit matrix in its listed-site order.

    The matrix-unit expansion keeps the first local tensor factor on ``site1``
    and the second on ``site2``. It reuses the arbitrary-site factor embedding
    without constructing a full-space permutation matrix.

    Args:
        op: Local two-qubit matrix in ``(site1, site2)`` tensor-factor order.
        num_sites: Total number of sites in the chain.
        site1: Site for the first matrix tensor factor.
        site2: Site for the second matrix tensor factor.
        dims: Per-site Hilbert-space dimensions.
        sparse: Whether to return a sparse matrix.

    Returns:
        Embedded matrix in site-0-LSB order.

    Raises:
        ValueError: If ``op`` is not a two-qubit matrix.
    """
    op_array = _to_dense(op)
    if op_array.shape != (4, 4):
        msg = f"Periodic-wrap matrix must have shape (4, 4), got {op_array.shape}."
        raise ValueError(msg)

    hilbert_dimension = int(np.prod(dims))
    if sparse:
        result: NDArray[np.complex128] | scipy.sparse.spmatrix = scipy.sparse.csr_matrix(
            (hilbert_dimension, hilbert_dimension), dtype=np.complex128
        )
    else:
        result = np.zeros((hilbert_dimension, hilbert_dimension), dtype=np.complex128)

    op_tensor = op_array.reshape(2, 2, 2, 2)
    for output1, output2, input1, input2 in np.ndindex(2, 2, 2, 2):
        coefficient = op_tensor[output1, output2, input1, input2]
        if coefficient == 0:
            continue
        factor1 = np.zeros((2, 2), dtype=np.complex128)
        factor2 = np.zeros((2, 2), dtype=np.complex128)
        factor1[output1, input1] = 1.0
        factor2[output2, input2] = 1.0
        if sparse:
            embedded = _embed_two_site_factors_sparse(factor1, factor2, num_sites, site1, site2, dims)
        else:
            embedded = embed_two_site_factors(
                factor1,
                factor2,
                num_sites,
                site1,
                site2,
                physical_dimensions=dims,
            )
        result += coefficient * embedded
    return result


def _embed_generic(
    sites: list[int],
    num_sites: int,
    *,
    op_matrix: NDArray[np.complex128] | scipy.sparse.spmatrix | None = None,
    op_factors: tuple[NDArray[np.complex128] | scipy.sparse.spmatrix, NDArray[np.complex128] | scipy.sparse.spmatrix]
    | None = None,
    sparse: bool = False,
    physical_dimensions: list[int] | int | None = None,
) -> NDArray[np.complex128] | scipy.sparse.spmatrix:
    """Embed a local operator into the full Hilbert space (MPS / Qiskit indexing).

    Args:
        sites: Site indices the operator acts on.
        num_sites: Total number of sites.
        op_matrix: Optional matrix for 1-site, adjacent 2-site, or periodic-wrap
            two-qubit embedding.
        op_factors: Optional non-adjacent two-site product factors.
        sparse: If True, return a CSR sparse matrix.
        physical_dimensions: Per-site Hilbert-space dimensions (defaults to qubits).

    Returns:
        Embedded operator on the full space.

    Raises:
        ValueError: If a 2-site matrix is not nearest-neighbor or periodic-wrap,
            or if factors are not 2-site.
        NotImplementedError: If neither matrix nor factors provided.
    """
    dims = resolve_physical_dimensions(num_sites, physical_dimensions)
    if op_matrix is not None:
        if len(sites) == 1:
            if sites[0] < 0 or sites[0] >= num_sites:
                msg = f"site {sites[0]} out of range for length {num_sites}."
                raise ValueError(msg)
            if sparse:
                return _embed_one_site_sparse(op_matrix, num_sites, sites[0], dims)
            return embed_one_site_operator(
                _to_dense(op_matrix),
                num_sites,
                sites[0],
                physical_dimensions=dims,
            )

        if len(sites) == 2:
            s1, s2 = sites[0], sites[1]
            if s1 == s2:
                msg = "Two-site matrix sites must be distinct."
                raise ValueError(msg)
            for site in (s1, s2):
                if site < 0 or site >= num_sites:
                    msg = f"site {site} out of range for length {num_sites}."
                    raise ValueError(msg)
            is_neighbor = abs(s1 - s2) == 1
            is_periodic_wrap = num_sites > 2 and {s1, s2} == {0, num_sites - 1}
            if not is_neighbor and not is_periodic_wrap:
                msg = "Matrix-based 2-site op must act on nearest neighbors or the periodic wrap."
                raise ValueError(msg)
            if is_periodic_wrap:
                if any(dimension != 2 for dimension in dims):
                    msg = "Periodic-wrap two-site matrices currently require qubit sites throughout the system."
                    raise ValueError(msg)
                return _embed_periodic_two_site_matrix(
                    op_matrix,
                    num_sites,
                    s1,
                    s2,
                    dims,
                    sparse=sparse,
                )
            site_left = min(s1, s2)
            pair_op = op_matrix
            if s1 > s2:
                pair_op = _swap_two_site_factor_order(_to_dense(op_matrix), dims[s1], dims[s2])
            if sparse:
                return _embed_adjacent_two_site_sparse(pair_op, num_sites, site_left, dims)
            return embed_adjacent_two_site_operator(
                _to_dense(pair_op),
                num_sites,
                site_left,
                physical_dimensions=dims,
            )

    if op_factors is not None:
        op1, op2 = op_factors
        if len(sites) != 2:
            msg = f"Factors require exactly 2 sites, got {len(sites)}"
            raise ValueError(msg)
        s1, s2 = sites
        if s1 == s2:
            msg = "site1 and site2 must differ."
            raise ValueError(msg)
        for site in (s1, s2):
            if site < 0 or site >= num_sites:
                msg = f"site {site} out of range for length {num_sites}."
                raise ValueError(msg)
        if sparse:
            return _embed_two_site_factors_sparse(op1, op2, num_sites, s1, s2, dims)
        return embed_two_site_factors(
            _to_dense(op1),
            _to_dense(op2),
            num_sites,
            s1,
            s2,
            physical_dimensions=dims,
        )

    msg = "Invalid embedding request: neither matrix nor factors provided."
    raise NotImplementedError(msg)


# --- Dense Embedding ---


def _embed_operator_dense(
    process: dict,
    num_sites: int,
    *,
    physical_dimensions: list[int] | int | None = None,
) -> NDArray[np.complex128]:
    """Embeds a local noise process operator into the full Hilbert space (dense).

    Args:
        process: Dictionary containing "sites" (list of ints) and either "matrix" (operator matrix)
                 or "factors" (tuple of operators).
        num_sites: Total number of sites in the system.
        physical_dimensions: Per-site Hilbert-space dimensions (defaults to qubits).

    Returns:
        The embedded operator as a dense matrix.

    Raises:
        NotImplementedError: If the process dictionary is missing required keys.
    """
    sites = process["sites"]
    params: dict[str, Any] = {}
    if "matrix" in process:
        params["op_matrix"] = process["matrix"]
    elif "factors" in process:
        params["op_factors"] = process["factors"]
    else:
        msg = f"Cannot embed operator for process: {process}"
        raise NotImplementedError(msg)

    result = _embed_generic(
        sites=sites,
        num_sites=num_sites,
        sparse=False,
        physical_dimensions=physical_dimensions,
        **params,
    )
    return cast("NDArray[np.complex128]", result)


def _embed_observable_dense(
    obs: Observable,
    num_sites: int,
    *,
    physical_dimensions: list[int] | int | None = None,
) -> NDArray[np.complex128]:
    """Embeds an observable into the full Hilbert space (dense).

    Args:
        obs: Observable object containing sites and the operator definition.
        num_sites: Total number of sites in the system.
        physical_dimensions: Per-site Hilbert-space dimensions (defaults to qubits).

    Returns:
        The embedded observable as a dense matrix.

    Raises:
        ValueError: If the observable does not define a local operator.
        NotImplementedError: If the observable involves more than 2 sites.
    """
    sites = obs.sites
    if isinstance(sites, int):
        sites = [sites]
    if sites is None or obs.matrix is None:
        msg = "Observable embedding requires an operator with explicit sites."
        raise ValueError(msg)

    if len(sites) > 2:
        msg = f"Unsupported observable site count: {len(sites)}"
        raise NotImplementedError(msg)

    result = _embed_generic(
        sites=sites,
        num_sites=num_sites,
        op_matrix=obs.matrix,
        sparse=False,
        physical_dimensions=physical_dimensions,
    )
    return cast("NDArray[np.complex128]", result)


# --- Sparse Embedding ---


def _embed_operator_sparse(
    process: dict,
    num_sites: int,
    *,
    physical_dimensions: list[int] | int | None = None,
) -> scipy.sparse.spmatrix:
    """Embeds a local noise process operator into the full Hilbert space (sparse).

    Args:
        process: Dictionary containing "sites" (list of ints) and either "matrix" (operator matrix)
                 or "factors" (tuple of operators).
        num_sites: Total number of sites in the system.
        physical_dimensions: Per-site Hilbert-space dimensions (defaults to qubits).

    Returns:
        The embedded operator as a sparse matrix.

    Raises:
        NotImplementedError: If the process dictionary is missing required keys.
    """
    sites = process["sites"]
    params: dict[str, Any] = {}
    if "matrix" in process:
        params["op_matrix"] = _to_sparse_csr(process["matrix"])
    elif "factors" in process:
        op1, op2 = process["factors"]
        params["op_factors"] = (_to_sparse_csr(op1), _to_sparse_csr(op2))
    else:
        msg = f"Cannot embed operator for process: {process}"
        raise NotImplementedError(msg)

    result = _embed_generic(
        sites=sites,
        num_sites=num_sites,
        sparse=True,
        physical_dimensions=physical_dimensions,
        **params,
    )
    return cast("scipy.sparse.spmatrix", result)


def _embed_observable_sparse(
    obs: Observable,
    num_sites: int,
    *,
    physical_dimensions: list[int] | int | None = None,
) -> scipy.sparse.spmatrix:
    """Embeds an observable into the full Hilbert space (sparse).

    Args:
        obs: Observable object containing sites and the operator definition.
        num_sites: Total number of sites in the system.
        physical_dimensions: Per-site Hilbert-space dimensions (defaults to qubits).

    Returns:
        The embedded observable as a sparse matrix.

    Raises:
        ValueError: If the observable does not define a local operator.
        NotImplementedError: If the observable involves more than 2 sites.
    """
    sites = obs.sites
    if isinstance(sites, int):
        sites = [sites]
    if sites is None or obs.matrix is None:
        msg = "Observable embedding requires an operator with explicit sites."
        raise ValueError(msg)

    if len(sites) > 2:
        msg = f"Unsupported observable site count: {len(sites)}"
        raise NotImplementedError(msg)

    result = _embed_generic(
        sites=sites,
        num_sites=num_sites,
        op_matrix=_to_sparse_csr(obs.matrix),
        sparse=True,
        physical_dimensions=physical_dimensions,
    )
    return cast("scipy.sparse.spmatrix", result)
