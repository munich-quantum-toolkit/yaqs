# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing observable definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ..libraries.observable_library import ObservableLibrary

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

ObservableType = Literal["operator", "bitstring", "diagnostic"]
_HERMITIAN_RTOL = 1e-10
_HERMITIAN_ATOL = 1e-12


def _validate_matrix(operator: ArrayLike) -> NDArray[np.complex128]:
    """Return an independent, finite, Hermitian operator matrix.

    Args:
        operator: Matrix-like operator data.

    Returns:
        A complex copy of the validated matrix.

    Raises:
        TypeError: If ``operator`` cannot be read as numeric matrix data.
        ValueError: If the matrix is empty, non-square, non-finite, or non-Hermitian.
    """
    try:
        matrix = np.array(operator, dtype=np.complex128, copy=True)
    except (TypeError, ValueError) as exc:
        msg = "operator must be a named observable or a numeric matrix."
        raise TypeError(msg) from exc
    if matrix.ndim != 2:
        msg = "Observable matrix must be a 2-D array."
        raise ValueError(msg)
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        msg = "Observable matrix must be non-empty and square."
        raise ValueError(msg)
    if not np.all(np.isfinite(matrix)):
        msg = "Observable matrix must contain only finite values."
        raise ValueError(msg)
    if not np.allclose(matrix, matrix.conj().T, rtol=_HERMITIAN_RTOL, atol=_HERMITIAN_ATOL):
        msg = "Observable matrix must be Hermitian."
        raise ValueError(msg)
    return matrix


def _site_count(sites: int | list[int]) -> int:
    """Validate sites and return their count.

    Args:
        sites: One site or a list of sites.

    Returns:
        The number of sites.

    Raises:
        TypeError: If a site is not an integer.
        ValueError: If a list of sites is empty.
    """
    if isinstance(sites, bool):
        msg = "sites must be an int or a list of ints."
        raise TypeError(msg)
    if isinstance(sites, int):
        return 1
    if not isinstance(sites, list) or any(isinstance(site, bool) or not isinstance(site, int) for site in sites):
        msg = "sites must be an int or a list of ints."
        raise TypeError(msg)
    if not sites:
        msg = "sites must not be empty."
        raise ValueError(msg)
    return len(sites)


class Observable:
    """A Hermitian operator or state diagnostic requested from a simulation.

    Named observables use :class:`~mqt.yaqs.core.libraries.observable_library.ObservableLibrary`.
    Numeric matrices define custom local operators. Binary strings request the
    probability of a computational-basis state.

    Attributes:
        name: Canonical observable or diagnostic name.
        matrix: Local operator matrix, or ``None`` for diagnostics and bitstrings.
        sites: Site or sites for a local operator or diagnostic.
        interaction: Number of sites used by a local operator.
        type: ``"operator"``, ``"bitstring"``, or ``"diagnostic"``.
        bitstring: Computational-basis state for a bitstring request, otherwise ``None``.
    """

    def __init__(
        self,
        operator: str | ArrayLike,
        sites: int | list[int] | None = None,
        **operator_kwargs: object,
    ) -> None:
        """Create an observable.

        Args:
            operator: Named observable, computational-basis bitstring, or local matrix.
            sites: Site indices on which a named or custom local operator acts.
            **operator_kwargs: Arguments for a configurable named observable.

        Raises:
            TypeError: If matrix data, sites, or factory arguments have invalid types.
            ValueError: If a name, matrix, or number of sites is invalid.
        """
        self.bitstring: str | None = None
        if isinstance(operator, str) and operator and set(operator) <= {"0", "1"}:
            if sites is not None:
                msg = "Bitstring observables do not accept sites."
                raise TypeError(msg)
            if operator_kwargs:
                msg = "Bitstring observables do not accept operator parameters."
                raise TypeError(msg)
            self.name = "pvm"
            self.type: ObservableType = "bitstring"
            self.matrix: NDArray[np.complex128] | None = None
            self.sites = None
            self.interaction = 0
            self.bitstring = operator
            return

        if isinstance(operator, str):
            definition = ObservableLibrary.resolve(operator, **operator_kwargs)
            if sites is None:
                msg = "sites are required for named observables."
                raise ValueError(msg)
            count = _site_count(sites)
            if definition.type == "diagnostic":
                self.name = definition.name
                self.type = "diagnostic"
                self.matrix = None
                self.sites = sites
                self.interaction = 0
                return
            if count != definition.interaction:
                msg = (
                    f"Observable {definition.name!r} acts on {definition.interaction} site(s), "
                    f"but {count} site(s) were given."
                )
                raise ValueError(msg)
            assert definition.matrix is not None
            matrix = _validate_matrix(definition.matrix)
            name = definition.name
            interaction = definition.interaction
        else:
            if sites is None:
                msg = "sites are required for matrix observables."
                raise ValueError(msg)
            count = _site_count(sites)
            if operator_kwargs:
                msg = "Observable parameters are only supported for named observables."
                raise TypeError(msg)
            matrix = _validate_matrix(operator)
            interaction = count
            if interaction > 1 and matrix.shape != (2**interaction, 2**interaction):
                msg = f"A {interaction}-site custom observable must have shape {(2**interaction, 2**interaction)}."
                raise ValueError(msg)
            name = "local"

        self.name = name
        self.type = "operator"
        self.matrix = matrix
        self.sites = sites
        self.interaction = interaction
