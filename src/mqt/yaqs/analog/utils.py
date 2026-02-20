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


# --- Dense Embedding ---


def _embed_operator_dense(process: dict, num_sites: int) -> NDArray[np.complex128]:
    """Embeds a local noise process operator into the full Hilbert space (dense).

    Args:
        process: A dictionary defining the noise process. Must contain "sites" and either
            "matrix" (for 1-site or adjacent 2-site ops) or "factors" (for tensor factors).
        num_sites: The total number of sites in the system.

    Returns:
        The full-system size operator as a dense numpy array.

    Raises:
        NotImplementedError: If the process definition is invalid or unsupported.
    """
    sites = process["sites"]

    def eye(n: int) -> NDArray[np.complex128]:
        return np.eye(n, dtype=complex)

    if "matrix" in process:
        op_local = process["matrix"]
        if len(sites) == 1:
            site = sites[0]
            ops = [eye(2) for _ in range(num_sites)]
            ops[site] = op_local
            return _kron_all_dense(ops)

        if len(sites) == 2:
            s1, s2 = sorted(sites)
            assert s2 == s1 + 1, "Matrix-based 2-site op must be adjacent"
            left_id = eye(2**s1) if s1 > 0 else eye(1)
            right_id = eye(2 ** (num_sites - 1 - s2)) if s2 < num_sites - 1 else eye(1)
            res = np.kron(left_id, op_local)
            return np.kron(res, right_id)

    if "factors" in process:
        op1, op2 = process["factors"]
        s1, s2 = sites
        ops = [eye(2) for _ in range(num_sites)]
        ops[s1] = op1
        ops[s2] = op2
        return _kron_all_dense(ops)

    msg = f"Cannot embed operator for process: {process}"
    raise NotImplementedError(msg)


def _embed_observable_dense(obs: Observable, num_sites: int) -> NDArray[np.complex128]:
    """Embeds an observable into the full Hilbert space (dense).

    Args:
        obs: The observable to embed.
        num_sites: The total number of sites in the system.

    Returns:
        The full-system size observable operator as a dense numpy array.

    Raises:
        NotImplementedError: If the observable targets non-adjacent 2-site operations
            or an unsupported number of sites (>2).
    """
    sites = obs.sites
    if isinstance(sites, int):
        sites = [sites]

    def eye(n: int) -> NDArray[np.complex128]:
        return np.eye(n, dtype=complex)

    if len(sites) == 1:
        site = sites[0]
        ops = [eye(2) for _ in range(num_sites)]
        ops[site] = obs.gate.matrix
        return _kron_all_dense(ops)

    if len(sites) == 2:
        s1, s2 = sorted(sites)
        if s2 == s1 + 1:
            op_local = obs.gate.matrix
            left_id = eye(2**s1) if s1 > 0 else eye(1)
            right_id = eye(2 ** (num_sites - 1 - s2)) if s2 < num_sites - 1 else eye(1)
            res = np.kron(left_id, op_local)
            return np.asarray(np.kron(res, right_id), dtype=complex)
        msg = "Non-adjacent 2-site observables not yet supported in exact solver."
        raise NotImplementedError(msg)

    msg = f"Unsupported observable site count: {len(sites)}"
    raise NotImplementedError(msg)


# --- Sparse Embedding ---


def _embed_operator_sparse(process: dict, num_sites: int) -> scipy.sparse.spmatrix:
    """Embeds a local noise process operator into the full Hilbert space (sparse).

    Args:
        process: A dictionary defining the noise process. Must contain "sites" and either
            "matrix" (for 1-site or adjacent 2-site ops) or "factors" (for tensor factors).
        num_sites: The total number of sites in the system.

    Returns:
        The full-system size operator as a sparse matrix (CSR).

    Raises:
        NotImplementedError: If the process definition is invalid or unsupported.
    """
    sites = process["sites"]

    def eye(n: int) -> scipy.sparse.spmatrix:
        return scipy.sparse.eye(n, format="csr", dtype=complex)

    def to_sparse(op: NDArray[np.complex128] | scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
        if issparse(op):
            return cast("Any", op)
        return scipy.sparse.csr_matrix(op)

    if "matrix" in process:
        op_local = to_sparse(process["matrix"])
        if len(sites) == 1:
            site = sites[0]
            ops: list[NDArray[np.complex128] | scipy.sparse.spmatrix] = [eye(2) for _ in range(num_sites)]
            ops[site] = op_local
            return _kron_all_sparse(ops)

        if len(sites) == 2:
            s1, s2 = sorted(sites)
            assert s2 == s1 + 1, "Matrix-based 2-site op must be adjacent"
            left_id = eye(2**s1) if s1 > 0 else eye(1)
            right_id = eye(2 ** (num_sites - 1 - s2)) if s2 < num_sites - 1 else eye(1)

            res = scipy.sparse.kron(left_id, op_local, format="csr")
            final_res = scipy.sparse.kron(res, right_id, format="csr")
            return cast("Any", final_res)

    if "factors" in process:
        op1, op2 = process["factors"]
        s1, s2 = sites
        ops = [eye(2) for _ in range(num_sites)]
        ops[s1] = to_sparse(op1)
        ops[s2] = to_sparse(op2)
        return _kron_all_sparse(ops)

    msg = f"Cannot embed operator for process: {process}"
    raise NotImplementedError(msg)


def _embed_observable_sparse(obs: Observable, num_sites: int) -> scipy.sparse.spmatrix:
    """Embeds an observable into the full Hilbert space (sparse).

    Args:
        obs: The observable to embed.
        num_sites: The total number of sites in the system.

    Returns:
        The full-system size observable operator as a sparse matrix (CSR).

    Raises:
        NotImplementedError: If the observable targets non-adjacent 2-site operations
            or an unsupported number of sites (>2).
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

    if len(sites) == 1:
        site = sites[0]
        ops: list[NDArray[np.complex128] | scipy.sparse.spmatrix] = [eye(2) for _ in range(num_sites)]
        ops[site] = to_sparse(obs.gate.matrix)
        return _kron_all_sparse(ops)

    if len(sites) == 2:
        s1, s2 = sorted(sites)
        if s2 == s1 + 1:
            op_local = to_sparse(obs.gate.matrix)
            left_id = eye(2**s1) if s1 > 0 else eye(1)
            right_id = eye(2 ** (num_sites - 1 - s2)) if s2 < num_sites - 1 else eye(1)

            res = scipy.sparse.kron(left_id, op_local, format="csr")
            final_res = scipy.sparse.kron(res, right_id, format="csr")
            return cast("Any", final_res)
        msg = "Non-adjacent 2-site observables not yet supported in exact solver."
        raise NotImplementedError(msg)

    msg = f"Unsupported observable site count: {len(sites)}"
    raise NotImplementedError(msg)
