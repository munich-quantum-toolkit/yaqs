# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Internal helpers for :class:`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import scipy.sparse

if TYPE_CHECKING:
    from .hamiltonian import Hamiltonian
    from .mpo import MPO

Representation = Literal["mpo", "sparse", "dense"]

_ALLOWED_REPRESENTATIONS = frozenset({"mpo", "sparse", "dense"})


def validate_representation(value: str) -> Representation:
    """Validate and return a Hamiltonian representation label.

    Returns:
        A valid ``"mpo"``, ``"sparse"``, or ``"dense"`` label.

    Raises:
        ValueError: If ``value`` is not ``"mpo"``, ``"sparse"``, or ``"dense"``.
    """
    if value not in _ALLOWED_REPRESENTATIONS:
        msg = f"Invalid representation {value!r}. Allowed values are 'mpo', 'sparse', or 'dense'."
        raise ValueError(msg)
    return cast("Representation", value)


def sparse_to_csr(matrix: scipy.sparse.spmatrix) -> scipy.sparse.csr_matrix:
    """Return ``matrix`` as CSR (copies only when needed)."""
    if isinstance(matrix, scipy.sparse.csr_matrix):
        return matrix
    return scipy.sparse.csr_matrix(matrix)


def attach_mpo(wrapped: Hamiltonian, mpo: MPO) -> None:
    """Initialize ``wrapped`` from an existing MPO (factory helper for :meth:`Hamiltonian.from_mpo`)."""
    wrapped.length = mpo.length
    wrapped.physical_dimension = mpo.physical_dimension
    wrapped.representation = "mpo"
    # Private fields: wrapped is a fresh Hamiltonian from __new__; attach_mpo is the sole initializer.
    wrapped._tensors = None  # noqa: SLF001
    wrapped._matrix = None  # noqa: SLF001
    wrapped._sparse_matrix = None  # noqa: SLF001
    wrapped._mpo = mpo  # noqa: SLF001
    wrapped._encoded_as = "mpo"  # noqa: SLF001
