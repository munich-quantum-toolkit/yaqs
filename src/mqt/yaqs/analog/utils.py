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

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ..core.data_structures.simulation_parameters import Observable


def _kron_all_dense(
    ops: list[NDArray[np.complex128]],
) -> NDArray[np.complex128]:
    """Compute the Kronecker product of a list of dense matrices.

    Args:
        ops: A list of dense numpy arrays to tensor product together.

    Returns:
        The resulting dense matrix as a numpy array.
    """
    res = ops[0]
    for op in ops[1:]:
        res = np.kron(res, op)
    return np.asarray(res, dtype=complex)


def _kron_all_sparse(
    ops: list[NDArray[np.complex128] | scipy.sparse.spmatrix],
) -> scipy.sparse.spmatrix:
    """Compute the Kronecker product of a list of matrices (sparse or dense) returning sparse.

    Args:
        ops: A list of matrices (dense or sparse) to tensor product together.

    Returns:
        The resulting sparse matrix (CSR format).
    """
    res = ops[0] if issparse(ops[0]) else scipy.sparse.csr_matrix(ops[0])

    for op in ops[1:]:
        op_csr = op if issparse(op) else scipy.sparse.csr_matrix(op)

        # Blanket ignore for sparse kron to satisfy strict type checkers
        res = scipy.sparse.kron(res, op_csr, format="csr")

    return cast("scipy.sparse.spmatrix", res)


# --- Generic Embedding Driver ---


def _embed_generic(
    sites: list[int],
    num_sites: int,
    eye_fn: Callable[[int], Any],
    kron_two_fn: Callable[[Any, Any], Any],
    kron_all_fn: Callable[[list[Any]], Any],
    op_matrix: Any | None = None,  # noqa: ANN401
    op_factors: tuple[Any, Any] | None = None,
) -> Any:  # noqa: ANN401
    """Generic driver for embedding operators into the full Hilbert space.

    Handles the logic for 1-site, 2-site adjacent, and factorized embeddings
    using provided primitive functions for identity and kroenecker products.

    Args:
        sites: List of site indices the operator acts on.
        num_sites: Total number of sites in the system.
        eye_fn: Function to create an identity matrix of size N.
        kron_two_fn: Function to compute the Kronecker product of two matrices.
        kron_all_fn: Function to compute the Kronecker product of a list of matrices.
        op_matrix: Optional matrix operator to embed (for 1-site or 2-site adjacent).
        op_factors: Optional tuple of operators to embed (for 2-site non-adjacent).

    Returns:
        The embedded operator/observable (type depends on generic functions).

    Raises:
        ValueError: If 2-site matrix is not adjacent or factors are not 2-site.
        NotImplementedError: If neither matrix nor factors provided.
    """
    if op_matrix is not None:
        if len(sites) == 1:
            site = sites[0]
            ops = [eye_fn(2) for _ in range(num_sites)]
            ops[site] = op_matrix
            return kron_all_fn(ops)

        if len(sites) == 2:
            s1, s2 = sorted(sites)
            if s2 != s1 + 1:
                msg = "Matrix-based 2-site op must be adjacent"
                raise ValueError(msg)

            left_id = eye_fn(2**s1) if s1 > 0 else eye_fn(1)
            right_id = eye_fn(2 ** (num_sites - 1 - s2)) if s2 < num_sites - 1 else eye_fn(1)

            # Construct: left_id (x) op_matrix (x) right_id
            res = kron_two_fn(left_id, op_matrix)
            return kron_two_fn(res, right_id)

    if op_factors is not None:
        op1, op2 = op_factors
        if len(sites) != 2:
            msg = f"Factors require exactly 2 sites, got {len(sites)}"
            raise ValueError(msg)
        s1, s2 = sites  # factors assumes sites order matches ops order
        ops = [eye_fn(2) for _ in range(num_sites)]
        ops[s1] = op1
        ops[s2] = op2
        return kron_all_fn(ops)

    msg = "Invalid embedding request: neither matrix nor factors provided."
    raise NotImplementedError(msg)


# --- Dense Embedding ---


def _embed_operator_dense(process: dict, num_sites: int) -> NDArray[np.complex128]:
    """Embeds a local noise process operator into the full Hilbert space (dense).

    Args:
        process: Dictionary containing "sites" (list of ints) and either "matrix" (operator matrix)
                 or "factors" (tuple of operators).
        num_sites: Total number of sites in the system.

    Returns:
        The embedded operator as a dense matrix.

    Raises:
        NotImplementedError: If the process dictionary is missing required keys.
    """
    sites = process["sites"]

    def eye(n: int) -> NDArray[np.complex128]:
        return np.eye(n, dtype=complex)

    params: dict[str, Any] = {}
    if "matrix" in process:
        params["op_matrix"] = process["matrix"]
    elif "factors" in process:
        params["op_factors"] = process["factors"]
    else:
        msg = f"Cannot embed operator for process: {process}"
        raise NotImplementedError(msg)

    return _embed_generic(
        sites=sites, num_sites=num_sites, eye_fn=eye, kron_two_fn=np.kron, kron_all_fn=_kron_all_dense, **params
    )


def _embed_observable_dense(obs: Observable, num_sites: int) -> NDArray[np.complex128]:
    """Embeds an observable into the full Hilbert space (dense).

    Args:
        obs: Observable object containing sites and the gate/operator definition.
        num_sites: Total number of sites in the system.

    Returns:
        The embedded observable as a dense matrix.

    Raises:
        NotImplementedError: If the observable involves more than 2 sites.
    """
    sites = obs.sites
    if isinstance(sites, int):
        sites = [sites]

    def eye(n: int) -> NDArray[np.complex128]:
        return np.eye(n, dtype=complex)

    if len(sites) > 2:
        msg = f"Unsupported observable site count: {len(sites)}"
        raise NotImplementedError(msg)

    # Observables are always matrix-based in this context
    return _embed_generic(
        sites=sites,
        num_sites=num_sites,
        eye_fn=eye,
        kron_two_fn=np.kron,
        kron_all_fn=_kron_all_dense,
        op_matrix=obs.gate.matrix,
    )


# --- Sparse Embedding ---


def _embed_operator_sparse(process: dict, num_sites: int) -> scipy.sparse.spmatrix:
    """Embeds a local noise process operator into the full Hilbert space (sparse).

    Args:
        process: Dictionary containing "sites" (list of ints) and either "matrix" (operator matrix)
                 or "factors" (tuple of operators).
        num_sites: Total number of sites in the system.

    Returns:
        The embedded operator as a sparse matrix.

    Raises:
        NotImplementedError: If the process dictionary is missing required keys.
    """
    sites = process["sites"]

    def eye(n: int) -> scipy.sparse.spmatrix:
        return scipy.sparse.eye(n, format="csr", dtype=complex)

    def to_sparse(op: NDArray[np.complex128] | scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
        if issparse(op):
            return cast("Any", op)
        return scipy.sparse.csr_matrix(op)

    def sparse_kron(a: scipy.sparse.spmatrix, b: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
        return scipy.sparse.kron(a, b, format="csr")

    params: dict[str, Any] = {}
    if "matrix" in process:
        params["op_matrix"] = to_sparse(process["matrix"])
    elif "factors" in process:
        op1, op2 = process["factors"]
        params["op_factors"] = (to_sparse(op1), to_sparse(op2))
    else:
        msg = f"Cannot embed operator for process: {process}"
        raise NotImplementedError(msg)

    return _embed_generic(
        sites=sites, num_sites=num_sites, eye_fn=eye, kron_two_fn=sparse_kron, kron_all_fn=_kron_all_sparse, **params
    )


def _embed_observable_sparse(obs: Observable, num_sites: int) -> scipy.sparse.spmatrix:
    """Embeds an observable into the full Hilbert space (sparse).

    Args:
        obs: Observable object containing sites and the gate/operator definition.
        num_sites: Total number of sites in the system.

    Returns:
        The embedded observable as a sparse matrix.

    Raises:
        NotImplementedError: If the observable involves more than 2 sites.
    """
    sites = obs.sites
    if isinstance(sites, int):
        sites = [sites]

    def eye(n: int) -> scipy.sparse.spmatrix:
        return scipy.sparse.eye(n, format="csr", dtype=complex)

    def to_sparse(op: NDArray[np.complex128] | scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
        if issparse(op):
            return cast("Any", op)
        return scipy.sparse.csr_matrix(op)

    def sparse_kron(a: scipy.sparse.spmatrix, b: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
        return scipy.sparse.kron(a, b, format="csr")

    if len(sites) > 2:
        msg = f"Unsupported observable site count: {len(sites)}"
        raise NotImplementedError(msg)

    return _embed_generic(
        sites=sites,
        num_sites=num_sites,
        eye_fn=eye,
        kron_two_fn=sparse_kron,
        kron_all_fn=_kron_all_sparse,
        op_matrix=to_sparse(obs.gate.matrix),
    )
