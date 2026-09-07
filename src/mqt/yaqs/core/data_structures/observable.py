# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing observable definitions."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..libraries.observable_library import ObservableLibrary
from .mpo import MPO
from .state_utils import resolve_physical_dimensions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

ObservableType = Literal["operator", "bitstring", "diagnostic"]
_HERMITIAN_RTOL = 1e-10
_HERMITIAN_ATOL = 1e-12

__all__ = ["Observable"]


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
    scale = max(float(np.max(np.abs(matrix.real))), float(np.max(np.abs(matrix.imag))))
    if scale <= 0.0:
        return matrix
    scaled_matrix = matrix / scale
    matrix_norm = float(np.linalg.norm(scaled_matrix, ord="fro"))
    residual_norm = float(np.linalg.norm(scaled_matrix - scaled_matrix.conj().T, ord="fro"))
    scaled_tolerance = _HERMITIAN_ATOL / scale + _HERMITIAN_RTOL * matrix_norm
    if residual_norm > scaled_tolerance:
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


def _validate_sites_for_length(sites: int | list[int] | None, length: int) -> list[int]:
    """Return explicit sites after bounds and duplicate validation.

    Args:
        sites: One site, several sites, or ``None``.
        length: Full state length.

    Returns:
        Sites in user-supplied order.

    Raises:
        ValueError: If sites are absent, duplicated, or outside the state.
    """
    if sites is None:
        msg = "Observable requires explicit sites."
        raise ValueError(msg)
    site_list = [sites] if isinstance(sites, int) else list(sites)
    if len(set(site_list)) != len(site_list):
        msg = f"Observable sites must be distinct; got {site_list}."
        raise ValueError(msg)
    if any(site < 0 or site >= length for site in site_list):
        msg = f"Observable sites {site_list} are outside the state of length {length}."
        raise ValueError(msg)
    return site_list


def _copy_mpo(operator: MPO) -> MPO:
    """Return an independent copy of a structurally valid MPO.

    Args:
        operator: Source MPO.

    Returns:
        An MPO that owns its tensor arrays.
    """
    operator.validate()
    copied = copy.deepcopy(operator)
    copied.tensors = [np.array(tensor, dtype=np.complex128, copy=True) for tensor in operator.tensors]
    copied.validate()
    return copied


def _matrix_in_chain_order(
    matrix: NDArray[np.complex128],
    sites: list[int],
    dimensions: tuple[int, ...],
) -> tuple[NDArray[np.complex128], list[int], tuple[int, ...]]:
    """Permute an operator from user site order to ascending chain order.

    Args:
        matrix: Matrix whose tensor factors follow ``sites``.
        sites: User-supplied site order.
        dimensions: Full-chain physical dimensions.

    Returns:
        The permuted matrix, sorted sites, and their ordered dimensions.
    """
    permutation = sorted(range(len(sites)), key=sites.__getitem__)
    sorted_sites = [sites[index] for index in permutation]
    user_dimensions = tuple(dimensions[site] for site in sites)
    sorted_dimensions = tuple(dimensions[site] for site in sorted_sites)
    if permutation == list(range(len(sites))):
        return matrix.copy(), sorted_sites, sorted_dimensions
    num_sites = len(sites)
    tensor = matrix.reshape((*user_dimensions, *user_dimensions))
    axes = tuple(permutation) + tuple(num_sites + index for index in permutation)
    return np.transpose(tensor, axes).reshape(matrix.shape), sorted_sites, sorted_dimensions


def _build_local_mpo(
    matrix: NDArray[np.complex128],
    sites: list[int],
    dimensions: tuple[int, ...],
    factors: tuple[NDArray[np.complex128], ...] | None,
) -> tuple[MPO, tuple[int, ...]]:
    """Build an untruncated MPO on the smallest contiguous support interval.

    Args:
        matrix: Operator in user ``sites`` order.
        sites: Distinct, valid full-chain site indices.
        dimensions: Full-chain physical dimensions.
        factors: Optional product factors in user site-list order.

    Returns:
        The support MPO and the full-chain sites represented by its tensors.

    Raises:
        ValueError: If named product factors do not match the state dimensions.
    """
    ordered_matrix, sorted_sites, active_dimensions = _matrix_in_chain_order(matrix, sites, dimensions)
    if factors is None:
        active_mpo = MPO.from_matrix_with_dimensions(ordered_matrix, active_dimensions)
    else:
        permutation = sorted(range(len(sites)), key=sites.__getitem__)
        ordered_factors = [factors[index] for index in permutation]
        if any(
            factor.shape != (dimension, dimension)
            for factor, dimension in zip(ordered_factors, active_dimensions, strict=True)
        ):
            msg = f"Observable product factors do not match site dimensions {active_dimensions}."
            raise ValueError(msg)
        active_mpo = MPO.from_local_ops(ordered_factors)
    first_site = sorted_sites[0]
    last_site = sorted_sites[-1]
    active_index = 0
    tensors: list[NDArray[np.complex128]] = []
    for site in range(first_site, last_site + 1):
        if site == sorted_sites[active_index]:
            tensors.append(active_mpo.tensors[active_index].copy())
            active_index += 1
            continue
        bond_dimension = tensors[-1].shape[3]
        physical_identity = np.eye(dimensions[site], dtype=np.complex128)
        bond_identity = np.eye(bond_dimension, dtype=np.complex128)
        tensors.append(np.einsum("ij,ab->ijab", physical_identity, bond_identity))

    support_mpo = MPO()
    support_mpo.tensors = tensors
    support_mpo.length = len(tensors)
    support_mpo.physical_dimension = dimensions[first_site]
    support_mpo.validate()
    return support_mpo, tuple(range(first_site, last_site + 1))


class Observable:
    """A Hermitian operator or state diagnostic requested from a simulation.

    Named observables use :class:`~mqt.yaqs.core.libraries.observable_library.ObservableLibrary`.
    Numeric matrices define custom local operators. Binary strings request the
    probability of a computational-basis state. An
    :class:`~mqt.yaqs.core.data_structures.mpo.MPO` defines a full-chain
    operator.

    Matrix tensor factors follow the order in :attr:`sites`, with the first
    listed site as the most-significant Kronecker factor. Prepared MPO tensors
    follow ascending chain order and use ``(phys_out, phys_in, left_bond,
    right_bond)`` axes.

    YAQS copies matrix, MPO, and site-list inputs. Treat the public definition
    attributes and prepared MPO tensors as read-only. Create a new observable
    to change an operator or its support. An unprepared observable can be reused
    with different compatible state layouts. Matrix and MPO inputs are accepted
    when ``||O - O†||_F <= 1e-12 + 1e-10 * ||O||_F``.

    Attributes:
        name: Canonical observable or diagnostic name.
        matrix: Local operator matrix, or ``None`` for diagnostics and bitstrings.
        sites: Site or sites for a local operator or diagnostic.
        interaction: Number of sites used by a local operator.
        type: ``"operator"``, ``"bitstring"``, or ``"diagnostic"``.
        bitstring: Computational-basis state for a bitstring request, otherwise ``None``.
        mpo: Supplied or prepared MPO, otherwise ``None``.
        mpo_sites: Full-chain sites represented by the prepared MPO tensors.
        full_chain: Whether a supplied MPO defines the full chain.
        prepared_length: State length used for preparation, otherwise ``None``.
        prepared_dimensions: State dimensions used for preparation, otherwise ``None``.
    """

    def __init__(
        self,
        operator: str | ArrayLike | MPO,
        sites: int | list[int] | None = None,
        **operator_kwargs: object,
    ) -> None:
        """Create an observable.

        Args:
            operator: Named observable, computational-basis bitstring, local
                matrix, or full-chain MPO.
            sites: Site indices on which a named or custom local operator acts.
            **operator_kwargs: Arguments for a configurable named observable.

        Raises:
            TypeError: If matrix data, sites, or factory arguments have invalid types.
            ValueError: If a name, matrix, or number of sites is invalid.
        """
        self.bitstring: str | None = None
        self.mpo: MPO | None = None
        self.mpo_sites: tuple[int, ...] | None = None
        self.local_factors: tuple[NDArray[np.complex128], ...] | None = None
        self.full_chain = False
        self.prepared_length: int | None = None
        self.prepared_dimensions: tuple[int, ...] | None = None

        if isinstance(operator, MPO):
            if sites is not None:
                msg = "Full-chain MPO observables do not accept sites."
                raise TypeError(msg)
            if operator_kwargs:
                msg = "MPO observables do not accept operator parameters."
                raise TypeError(msg)
            copied_mpo = _copy_mpo(operator)
            if not copied_mpo.is_hermitian(rtol=_HERMITIAN_RTOL, atol=_HERMITIAN_ATOL):
                msg = "Observable MPO must be Hermitian."
                raise ValueError(msg)
            self.name = "mpo"
            self.type: ObservableType = "operator"
            self.matrix: NDArray[np.complex128] | None = None
            self.sites = None
            self.interaction = copied_mpo.length
            self.mpo = copied_mpo
            self.full_chain = True
            return

        if isinstance(operator, str) and operator and set(operator) <= {"0", "1"}:
            if sites is not None:
                msg = "Bitstring observables do not accept sites."
                raise TypeError(msg)
            if operator_kwargs:
                msg = "Bitstring observables do not accept operator parameters."
                raise TypeError(msg)
            self.name = "pvm"
            self.type = "bitstring"
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
                self.sites = list(sites) if isinstance(sites, list) else sites
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
            self.local_factors = definition.factors
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
            name = "local"

        self.name = name
        self.type = "operator"
        self.matrix = matrix
        self.sites = list(sites) if isinstance(sites, list) else sites
        self.interaction = interaction

    @classmethod
    def from_pauli_sum(
        cls,
        *,
        terms: Sequence[tuple[complex | float, str]],
        length: int,
    ) -> Observable:
        """Create a full-chain observable from a sum of Pauli products.

        Construction uses the MPO finite-state-machine builder with no SVD
        compression, cutoff, or bond-dimension cap.

        Args:
            terms: ``(coefficient, specification)`` pairs such as
                ``(0.5, "Z0 Z3")``.
            length: Number of qubit sites in the full operator.

        Returns:
            A Hermitian Pauli-sum observable.
        """
        mpo = MPO()
        mpo.from_pauli_sum(
            terms=list(terms),
            length=length,
            tol=0.0,
            max_bond_dim=None,
            n_sweeps=0,
        )
        observable = cls(mpo)
        observable.name = "pauli_sum"
        return observable

    def prepare(
        self,
        length: int,
        physical_dimensions: list[int] | int | None = None,
    ) -> Observable:
        """Validate this observable for a state and prepare its support MPO.

        Matrix tensor factors follow the user-supplied ``sites`` order. The
        prepared MPO follows ascending chain order and spans only from the
        smallest to the largest target site. Identity transport tensors carry
        its virtual bond across gaps. A supplied MPO represents the full chain.

        Args:
            length: Full state length.
            physical_dimensions: Per-site dimensions. ``None`` means qubits.

        Returns:
            An independent prepared observable. Calling this method on an
            already compatible prepared copy returns that copy.

        Raises:
            ValueError: If sites, dimensions, matrix shape, bitstring length,
                MPO structure, or MPO physical dimensions are incompatible.
        """
        dimensions = tuple(resolve_physical_dimensions(length, physical_dimensions))
        if self.prepared_length == length and self.prepared_dimensions == dimensions:
            return self

        prepared = copy.deepcopy(self)
        if prepared.type == "bitstring":
            assert prepared.bitstring is not None
            if len(prepared.bitstring) != length:
                msg = f"Bitstring length {len(prepared.bitstring)} does not match state length {length}."
                raise ValueError(msg)
        elif prepared.type == "diagnostic":
            _validate_sites_for_length(prepared.sites, length)
        elif prepared.full_chain:
            assert prepared.mpo is not None
            mpo_dimensions = prepared.mpo.validate()
            if prepared.mpo.length != length:
                msg = f"Observable MPO length {prepared.mpo.length} does not match state length {length}."
                raise ValueError(msg)
            if mpo_dimensions != dimensions:
                msg = f"Observable MPO physical dimensions {mpo_dimensions} do not match state dimensions {dimensions}."
                raise ValueError(msg)
            prepared.mpo_sites = tuple(range(length))
        else:
            sites = _validate_sites_for_length(prepared.sites, length)
            assert prepared.matrix is not None
            if len(sites) != prepared.interaction:
                if prepared.interaction == 1:
                    msg = f"One-site local observable requires one site, got {sites}."
                elif prepared.interaction == 2:
                    msg = f"Two-site local observable requires two sites, got {sites}."
                else:
                    msg = f"{prepared.interaction}-site observable requires {prepared.interaction} sites, got {sites}."
                raise ValueError(msg)
            expected_dimension = math.prod(dimensions[site] for site in sites)
            if prepared.matrix.shape != (expected_dimension, expected_dimension):
                site_dimensions = tuple(dimensions[site] for site in sites)
                if len(sites) == 1:
                    msg = (
                        f"Observable matrix shape {prepared.matrix.shape} does not match "
                        f"site {sites[0]} dimension {site_dimensions[0]}."
                    )
                elif len(sites) == 2:
                    msg = (
                        f"Observable matrix shape {prepared.matrix.shape} does not match "
                        f"site dimensions {site_dimensions[0]} and {site_dimensions[1]}."
                    )
                else:
                    msg = (
                        f"Observable matrix shape {prepared.matrix.shape} does not match "
                        f"site dimensions {site_dimensions}."
                    )
                raise ValueError(msg)
            prepared.mpo, prepared.mpo_sites = _build_local_mpo(
                prepared.matrix,
                sites,
                dimensions,
                prepared.local_factors,
            )
            if not prepared.mpo.is_hermitian(rtol=_HERMITIAN_RTOL, atol=_HERMITIAN_ATOL):
                msg = "Prepared observable MPO must be Hermitian."
                raise ValueError(msg)

        prepared.prepared_length = length
        prepared.prepared_dimensions = dimensions
        return prepared

    def to_mpo(
        self,
        length: int,
        physical_dimensions: list[int] | int | None = None,
    ) -> MPO:
        """Return an independent MPO prepared for a state layout.

        Args:
            length: Full state length.
            physical_dimensions: Per-site dimensions. ``None`` means qubits.

        Returns:
            An independent MPO for the prepared operator. Local operators use
            the smallest contiguous interval from the lowest to the highest
            target site. Tensor index zero therefore corresponds to the lowest
            target site. Supplied MPOs represent the full chain.

        Raises:
            ValueError: If this object is a diagnostic or bitstring request.
        """
        prepared = self.prepare(length, physical_dimensions)
        if prepared.mpo is None:
            msg = f"Observable type {prepared.type!r} does not define an MPO."
            raise ValueError(msg)
        return _copy_mpo(prepared.mpo)


def prepare_observables(
    observables: Sequence[Observable],
    length: int,
    physical_dimensions: list[int] | int | None = None,
) -> list[Observable]:
    """Prepare independent observable copies for one state layout.

    Args:
        observables: User observable definitions.
        length: Full state length.
        physical_dimensions: Per-site dimensions. ``None`` means qubits.

    Returns:
        Prepared observables in the input order.
    """
    prepared_observables: list[Observable] = []
    for observable in observables:
        prepared = observable.prepare(length, physical_dimensions)
        prepared_observables.append(copy.deepcopy(prepared) if prepared is observable else prepared)
    return prepared_observables
