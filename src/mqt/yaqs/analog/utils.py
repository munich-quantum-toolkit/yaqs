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
    embed_adjacent_two_site_operator,
    embed_one_site_operator,
    embed_two_site_factors,
    resolve_physical_dimensions,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.simulation_parameters import Observable


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
    if issparse(op):
        return np.asarray(cast("Any", op).toarray(), dtype=np.complex128)
    return np.asarray(op, dtype=np.complex128)


def _to_sparse_csr(op: NDArray[np.complex128] | scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    if issparse(op):
        return cast("Any", op)
    return scipy.sparse.csr_matrix(op)


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
        op_matrix: Optional matrix for 1-site or adjacent 2-site embedding.
        op_factors: Optional non-adjacent two-site product factors.
        sparse: If True, return a CSR sparse matrix.
        physical_dimensions: Per-site Hilbert-space dimensions (defaults to qubits).

    Returns:
        Embedded operator on the full space.

    Raises:
        ValueError: If 2-site matrix is not adjacent or factors are not 2-site.
        NotImplementedError: If neither matrix nor factors provided.
    """
    dims = resolve_physical_dimensions(num_sites, physical_dimensions)
    if op_matrix is not None:
        if len(sites) == 1:
            dense = embed_one_site_operator(
                _to_dense(op_matrix),
                num_sites,
                sites[0],
                physical_dimensions=dims,
            )
            return scipy.sparse.csr_matrix(dense) if sparse else dense

        if len(sites) == 2:
            s1, s2 = sorted(sites)
            if s2 != s1 + 1:
                msg = "Matrix-based 2-site op must be adjacent"
                raise ValueError(msg)
            dense = embed_adjacent_two_site_operator(
                _to_dense(op_matrix),
                num_sites,
                s1,
                physical_dimensions=dims,
            )
            return scipy.sparse.csr_matrix(dense) if sparse else dense

    if op_factors is not None:
        op1, op2 = op_factors
        if len(sites) != 2:
            msg = f"Factors require exactly 2 sites, got {len(sites)}"
            raise ValueError(msg)
        s1, s2 = sites
        dense = embed_two_site_factors(
            _to_dense(op1),
            _to_dense(op2),
            num_sites,
            s1,
            s2,
            physical_dimensions=dims,
        )
        return scipy.sparse.csr_matrix(dense) if sparse else dense

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
        obs: Observable object containing sites and the gate/operator definition.
        num_sites: Total number of sites in the system.
        physical_dimensions: Per-site Hilbert-space dimensions (defaults to qubits).

    Returns:
        The embedded observable as a dense matrix.

    Raises:
        NotImplementedError: If the observable involves more than 2 sites.
    """
    sites = obs.sites
    if isinstance(sites, int):
        sites = [sites]

    if len(sites) > 2:
        msg = f"Unsupported observable site count: {len(sites)}"
        raise NotImplementedError(msg)

    result = _embed_generic(
        sites=sites,
        num_sites=num_sites,
        op_matrix=obs.gate.matrix,
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
        obs: Observable object containing sites and the gate/operator definition.
        num_sites: Total number of sites in the system.
        physical_dimensions: Per-site Hilbert-space dimensions (defaults to qubits).

    Returns:
        The embedded observable as a sparse matrix.

    Raises:
        NotImplementedError: If the observable involves more than 2 sites.
    """
    sites = obs.sites
    if isinstance(sites, int):
        sites = [sites]

    if len(sites) > 2:
        msg = f"Unsupported observable site count: {len(sites)}"
        raise NotImplementedError(msg)

    result = _embed_generic(
        sites=sites,
        num_sites=num_sites,
        op_matrix=_to_sparse_csr(obs.gate.matrix),
        sparse=True,
        physical_dimensions=physical_dimensions,
    )
    return cast("scipy.sparse.spmatrix", result)
