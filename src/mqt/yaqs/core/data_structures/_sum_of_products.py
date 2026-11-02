# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact MPO of a sum of product operators, with the operator-Schmidt rank as the bond on every cut."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


class ExactComplex:
    """Complex number with rational real and imaginary parts."""

    __slots__ = ("im", "re")

    def __init__(self, re: Fraction, im: Fraction | None = None) -> None:
        """Store the real and imaginary parts.

        Args:
            re: Real part.
            im: Imaginary part, zero when omitted.
        """
        self.re = re
        self.im = Fraction(0) if im is None else im

    @classmethod
    def of(cls, value: complex | ExactComplex) -> ExactComplex:
        """Return ``value`` as an exact number; a float converts without rounding.

        Args:
            value: A Python or NumPy number, or an exact number.

        Returns:
            The exact number.
        """
        if isinstance(value, ExactComplex):
            return value
        number = complex(value)
        return cls(Fraction(number.real), Fraction(number.imag))

    def __add__(self, other: ExactComplex) -> ExactComplex:
        """Return the exact sum."""
        return ExactComplex(self.re + other.re, self.im + other.im)

    def __sub__(self, other: ExactComplex) -> ExactComplex:
        """Return the exact difference."""
        return ExactComplex(self.re - other.re, self.im - other.im)

    def __mul__(self, other: ExactComplex) -> ExactComplex:
        """Return the exact product."""
        return ExactComplex(self.re * other.re - self.im * other.im, self.re * other.im + self.im * other.re)

    def inverse(self) -> ExactComplex:
        """Return the exact reciprocal of a nonzero number.

        Returns:
            ``1 / self``.
        """
        norm = self.re * self.re + self.im * self.im
        return ExactComplex(self.re / norm, -self.im / norm)

    def __bool__(self) -> bool:
        """Return whether the number is nonzero."""
        return bool(self.re) or bool(self.im)

    def __complex__(self) -> complex:
        """Return the nearest double-precision number."""
        return complex(float(self.re), float(self.im))

    def __repr__(self) -> str:
        """Return a readable form."""
        return f"ExactComplex({self.re}, {self.im})"


_ZERO = ExactComplex(Fraction(0))
_ONE = ExactComplex(Fraction(1))
# A coefficient this small against the largest is round-off, such as cos(pi / 2) = 6e-17.
_ROUND_OFF = 8 * float(np.finfo(np.float64).eps)


def _reduced_row_echelon(matrix: list[list[ExactComplex]]) -> tuple[list[list[ExactComplex]], list[int]]:
    """Return the nonzero rows of the reduced row echelon form of ``matrix`` and its pivot columns.

    Args:
        matrix: Rows of exact numbers, all of one length.

    Returns:
        The reduced rows, each with a leading one, and the column of each leading one.
    """
    rows = [list(row) for row in matrix]
    num_cols = len(rows[0]) if rows else 0
    pivots: list[int] = []
    rank = 0
    for col in range(num_cols):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col].inverse()
        rows[rank] = [entry * scale for entry in rows[rank]]
        for i, row in enumerate(rows):
            if i != rank and row[col]:
                factor = row[col]
                rows[i] = [entry - factor * lead for entry, lead in zip(row, rows[rank], strict=True)]
        pivots.append(col)
        rank += 1
        if rank == len(rows):
            break
    return rows[:rank], pivots


def _site_basis(
    operators: list[NDArray[np.complex128]],
) -> tuple[list[NDArray[np.complex128]], list[dict[int, ExactComplex]]]:
    """Return independent operators spanning ``operators``, and each operator's exact coefficients on them.

    Args:
        operators: Distinct operators of one site.

    Returns:
        The independent operators, a subset of the input, and the coefficients of every input operator.
    """
    basis: list[NDArray[np.complex128]] = []
    vectors: list[list[ExactComplex]] = []
    coordinates: list[dict[int, ExactComplex]] = []
    for operator in operators:
        vector = [ExactComplex.of(entry) for entry in operator.ravel()]
        if vectors:
            # Solve vector = sum_b a_b vectors[b] exactly by reducing [vectors^T | vector].
            system = [[*(basis_vector[i] for basis_vector in vectors), vector[i]] for i in range(len(vector))]
            reduced, pivots = _reduced_row_echelon(system)
            if len(vectors) not in pivots:
                coordinates.append({pivot: row[-1] for pivot, row in zip(pivots, reduced, strict=True) if row[-1]})
                continue
        coordinates.append({len(basis): _ONE})
        basis.append(operator)
        vectors.append(vector)
    return basis, coordinates


def _independent_strings(
    terms: Sequence[tuple[complex | ExactComplex, Mapping[int, NDArray[np.complex128]]]], num_sites: int
) -> tuple[list[list[NDArray[np.complex128]]], dict[tuple[int, ...], ExactComplex]]:
    """Rewrite the terms over independent operators on each site, with exact coefficients.

    Args:
        terms: ``(coefficient, {site: operator})`` pairs, the identity on every absent site.
        num_sites: Number of sites.

    Returns:
        The independent operators of each site, and ``{operator index per site: coefficient}``.
    """
    identity = np.eye(2, dtype=np.complex128)
    keyed = [
        (coeff, [np.asarray(operators.get(site, identity), dtype=np.complex128) for site in range(num_sites)])
        for coeff, operators in terms
    ]
    distinct: list[dict[bytes, NDArray[np.complex128]]] = [{} for _ in range(num_sites)]
    for _, operators in keyed:
        for site, operator in enumerate(operators):
            distinct[site].setdefault(operator.tobytes(), operator)
    # Sites with the same operators, such as the controls of a gate, share one basis.
    solved: dict[tuple[bytes, ...], tuple[list[NDArray[np.complex128]], list[dict[int, ExactComplex]]]] = {}
    bases, coordinates = [], []
    for site_operators in distinct:
        signature = tuple(site_operators)
        if signature not in solved:
            solved[signature] = _site_basis(list(site_operators.values()))
        basis, coords = solved[signature]
        bases.append(basis)
        coordinates.append(dict(zip(site_operators, coords, strict=True)))

    strings: dict[tuple[int, ...], ExactComplex] = {}
    for coeff, operators in keyed:
        choices = [coordinates[site][operator.tobytes()].items() for site, operator in enumerate(operators)]
        for combination in product(*choices):
            value = ExactComplex.of(coeff)
            for _, weight in combination:
                if weight is not _ONE:
                    value *= weight
            string = tuple(part for part, _ in combination)
            strings[string] = strings.get(string, _ZERO) + value
    floor = _ROUND_OFF * max((abs(complex(value)) for value in strings.values()), default=0.0)
    return bases, {string: value for string, value in strings.items() if value and abs(complex(value)) > floor}


def _factorize(
    strings: dict[tuple[int, ...], ExactComplex], bases: list[list[NDArray[np.complex128]]]
) -> list[NDArray[np.complex128]]:
    """Return the MPO tensors of a sum of strings by an exact rank factorization at every site.

    Args:
        strings: ``{operator index per site: coefficient}``.
        bases: Independent operators of each site.

    Returns:
        Tensors ``(out, in, left, right)`` with the operator-Schmidt rank as the bond on every cut.
    """
    # Each tail of a string is stored once as (operator on its first site, rest of the tail).
    tails: list[tuple[int, int]] = [(-1, -1)]
    interned: dict[tuple[int, int], int] = {}
    starts: dict[int, ExactComplex] = {}
    for string, value in strings.items():
        tail = 0
        for part in reversed(string):
            key = (part, tail)
            if key not in interned:
                interned[key] = len(tails)
                tails.append(key)
            tail = interned[key]
        starts[tail] = value
    rows: list[dict[int, ExactComplex]] = [starts]
    reductions: dict[tuple, tuple[list[list[ExactComplex]], list[int], NDArray[np.complex128]]] = {}
    tensors = []
    for basis in bases:
        # Rows pair a bond state with this site's operator, columns are the tails to the right.
        blocks: dict[tuple[int, int], dict[int, ExactComplex]] = {}
        for state, row in enumerate(rows):
            for tail, value in row.items():
                part, rest = tails[tail]
                block = blocks.setdefault((state, part), {})
                block[rest] = block.get(rest, _ZERO) + value
        row_keys = list(blocks)
        col_keys = list(dict.fromkeys(rest for block in blocks.values() for rest in block))
        matrix = [[blocks[key].get(rest, _ZERO) for rest in col_keys] for key in row_keys]
        signature = (len(rows), len(basis), tuple(row_keys), tuple(tuple((e.re, e.im) for e in row) for row in matrix))
        if signature not in reductions:
            reduced, pivots = _reduced_row_echelon(matrix)
            # M = M[:, pivots] R, and the site tensor is sum_p basis[p] (x) M[(state, p), pivots].
            weights = np.zeros((len(basis), len(rows), len(pivots)), dtype=np.complex128)
            for (state, part), entries in zip(row_keys, matrix, strict=True):
                weights[part, state] = [complex(entries[col]) for col in pivots]
            reductions[signature] = (reduced, pivots, weights)
        reduced, pivots, weights = reductions[signature]
        tensors.append(np.einsum("pab,pkl->abkl", np.asarray(basis), weights))
        rows = [{rest: value for rest, value in zip(col_keys, row, strict=True) if value} for row in reduced]
    return tensors


def sum_of_products_tensors(
    terms: Sequence[tuple[complex | ExactComplex, Mapping[int, NDArray[np.complex128]]]],
    num_sites: int,
) -> list[NDArray[np.complex128]]:
    """Return the MPO tensors of ``sum_t c_t (x)_s o_{t,s}``, exact and at the minimal bond.

    Args:
        terms: ``(coefficient, {site: 2 x 2 operator})`` pairs, the identity on every absent site.
        num_sites: Number of sites the MPO spans.

    Returns:
        Tensors ``(out, in, left, right)``, all zero at bond one if the terms add up to zero.
    """
    bases, strings = _independent_strings(terms, num_sites)
    if not strings:
        return [np.zeros((2, 2, 1, 1), dtype=np.complex128) for _ in range(num_sites)]
    return _factorize(strings, bases)
