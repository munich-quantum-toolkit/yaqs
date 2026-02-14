# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for analog solvers (Lindblad and MCWF)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.simulation_parameters import Observable


def _kron_all(ops: list[NDArray[np.complex128]]) -> NDArray[np.complex128]:
    """Compute the Kronecker product of a list of matrices.

    Args:
        ops: A list of matrices (2D numpy arrays) to compute the product of.

    Returns:
        The resulting matrix from the Kronecker product of all input matrices.
    """
    res = ops[0]
    for op in ops[1:]:
        res = np.kron(res, op)
    return res.astype(np.complex128)


def _embed_operator(process: dict, num_sites: int) -> NDArray[np.complex128]:
    """Embeds a local noise process operator into the full Hilbert space (size 2^N).

    Args:
        process: A dictionary defining the noise process, including its matrix and target sites.
        num_sites: The total number of sites in the system.

    Returns:
        The operator embedded into the full Hilbert space.

    Raises:
        NotImplementedError: If the process definition involves unsupported structures.
    """
    sites = process["sites"]

    # If it's a "matrix" based process (1-site or adjacent 2-site)
    if "matrix" in process:
        op_local = process["matrix"]

        # If 1-site
        if len(sites) == 1:
            site = sites[0]
            # Construct I x ... x op x ... x I
            ops = [np.eye(2, dtype=complex) for _ in range(num_sites)]
            ops[site] = op_local
            return _kron_all(ops)

        # If 2-site (adjacent)
        if len(sites) == 2:
            s1, s2 = sorted(sites)
            assert s2 == s1 + 1, "Matrix-based 2-site op must be adjacent for simple kron construction"

            # Construct I x ... x op_2site x ... x I
            # Note: The 4x4 matrix `op_local` corresponds to sites (s1, s2)

            # Divide system into: [0...s1-1], [s1, s2], [s2+1...N-1]
            left_id = np.eye(2**s1, dtype=complex) if s1 > 0 else np.eye(1, dtype=complex)
            right_id = (
                np.eye(2 ** (num_sites - 1 - s2), dtype=complex) if s2 < num_sites - 1 else np.eye(1, dtype=complex)
            )

            # Full op = L x Op x R
            res = np.kron(left_id, op_local)
            return np.kron(res, right_id)

    # If it's a factors-based process (non-adjacent or crosstalk)
    if "factors" in process:
        # factors is tuple (op1, op2) acting on (site1, site2)
        op1, op2 = process["factors"]
        s1, s2 = sites  # Keep original order if crucial, but factors usually map 1-1 to sites

        ops = [np.eye(2, dtype=complex) for _ in range(num_sites)]
        ops[s1] = op1
        ops[s2] = op2
        return _kron_all(ops)

    msg = f"Cannot embed operator for process: {process}"
    raise NotImplementedError(msg)


def _embed_observable(obs: Observable, num_sites: int) -> NDArray[np.complex128]:
    """Embeds an observable into the full Hilbert space.

    Args:
        obs: The observable to embed.
        num_sites: The total number of sites in the system.

    Returns:
        The observable operator embedded into the full Hilbert space.

    Raises:
        NotImplementedError: If the observable targets non-adjacent sites or >2 sites.
    """
    sites = obs.sites
    if isinstance(sites, int):
        sites = [sites]

    # If 1-site
    if len(sites) == 1:
        site = sites[0]
        ops = [np.eye(2, dtype=complex) for _ in range(num_sites)]
        ops[site] = obs.gate.matrix
        return _kron_all(ops)

    # If 2-site
    if len(sites) == 2:
        s1, s2 = sorted(sites)
        if s2 == s1 + 1:
            # Adjacent 2-site observable (local gate matrix is 4x4)
            op_local = obs.gate.matrix
            left_id = np.eye(2**s1, dtype=complex) if s1 > 0 else np.eye(1, dtype=complex)
            right_id = (
                np.eye(2 ** (num_sites - 1 - s2), dtype=complex) if s2 < num_sites - 1 else np.eye(1, dtype=complex)
            )

            res = np.kron(left_id, op_local)
            return np.kron(res, right_id)
        msg = "Non-adjacent 2-site observables not yet supported in exact solver."
        raise NotImplementedError(msg)

    msg = f"Unsupported observable site count: {len(sites)}"
    raise NotImplementedError(msg)
