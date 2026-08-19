# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Internal helpers for :class:`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.sparse

if TYPE_CHECKING:
    from .hamiltonian import Hamiltonian
    from .mpo import MPO


def sparse_to_csr(matrix: scipy.sparse.spmatrix) -> scipy.sparse.csr_matrix:
    """Return ``matrix`` as CSR (copies only when needed)."""
    if isinstance(matrix, scipy.sparse.csr_matrix):
        return matrix
    return scipy.sparse.csr_matrix(matrix)


def attach_mpo(wrapped: Hamiltonian, mpo: MPO) -> None:
    """Initialize ``wrapped`` from an existing MPO (factory helper for :meth:`Hamiltonian.from_mpo`)."""
    wrapped.length = mpo.length
    wrapped.physical_dimension = mpo.physical_dimension
    # Private fields: wrapped is a fresh Hamiltonian from __new__; attach_mpo is the sole initializer.
    wrapped._tensors = None  # ruff:ignore[private-member-access]
    wrapped._matrix = None  # ruff:ignore[private-member-access]
    wrapped._sparse_matrix = None  # ruff:ignore[private-member-access]
    wrapped._mpo = mpo  # ruff:ignore[private-member-access]
    wrapped._pieces = None  # ruff:ignore[private-member-access]


def attach_piecewise(
    wrapped: Hamiltonian,
    pieces: tuple[tuple[Hamiltonian, float], ...],
    length: int,
) -> None:
    """Initialize ``wrapped`` from static Hamiltonian pieces (factory helper for :meth:`Hamiltonian.piecewise`)."""
    wrapped.length = length
    wrapped.physical_dimension = pieces[0][0].physical_dimension
    wrapped._tensors = None  # ruff:ignore[private-member-access]
    wrapped._matrix = None  # ruff:ignore[private-member-access]
    wrapped._sparse_matrix = None  # ruff:ignore[private-member-access]
    wrapped._mpo = None  # ruff:ignore[private-member-access]
    wrapped._pieces = pieces  # ruff:ignore[private-member-access]
