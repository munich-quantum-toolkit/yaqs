# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the exact MPO construction of a sum of product operators."""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs import MPO
from mqt.yaqs.core.data_structures._sum_of_products import (  # ruff: ignore[import-private-name]  # module-private builder
    sum_of_products_tensors,
)
from mqt.yaqs.core.libraries.operator_matrices import (
    IDENTITY,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PROJECTOR_ONE,
    PROJECTOR_ZERO,
)
from tests.conftest import YAQS_TEST_SEED

if TYPE_CHECKING:
    from numpy.typing import NDArray

_PAULIS = (IDENTITY, PAULI_X, PAULI_Y, PAULI_Z)


def _dense(terms: list[tuple[complex, dict[int, NDArray[np.complex128]]]], num_sites: int) -> NDArray[np.complex128]:
    """Return the sum of the terms, with site 0 least significant as in ``MPO.to_matrix``."""
    total = np.zeros((2**num_sites, 2**num_sites), dtype=np.complex128)
    for coeff, operators in terms:
        total += coeff * reduce(np.kron, [operators.get(site, IDENTITY) for site in reversed(range(num_sites))])
    return total


def _to_matrix(tensors: list[NDArray[np.complex128]]) -> NDArray[np.complex128]:
    """Return the dense operator of MPO tensors."""
    mpo = MPO()
    mpo.custom(tensors, transpose=False)
    return mpo.to_matrix()


def _ranks(dense: NDArray[np.complex128], num_sites: int) -> list[int]:
    """Return the operator-Schmidt rank of ``dense`` on every cut of the chain."""
    tensor = dense.reshape((2,) * (2 * num_sites))
    ranks = []
    for cut in range(1, num_sites):
        left = [num_sites - 1 - site for site in range(cut)] + [2 * num_sites - 1 - site for site in range(cut)]
        right = [axis for axis in range(2 * num_sites) if axis not in left]
        singular = np.linalg.svd(tensor.transpose(left + right).reshape(4**cut, -1), compute_uv=False)
        ranks.append(int((singular > 1e-10 * singular[0]).sum()))
    return ranks


@pytest.mark.parametrize("seed", range(5))
def test_random_pauli_sum_has_the_rank_on_every_cut(seed: int) -> None:
    """A sum of random Pauli strings is exact, with the operator-Schmidt rank as its bond on every cut."""
    rng = np.random.default_rng(YAQS_TEST_SEED + seed)
    num_sites = 5
    terms = [
        (
            complex(rng.integers(1, 4) * rng.choice([-1, 1])),
            {site: _PAULIS[rng.integers(0, 4)] for site in range(num_sites)},
        )
        for _ in range(8)
    ]

    tensors = sum_of_products_tensors(terms, num_sites)

    dense = _dense(terms, num_sites)
    np.testing.assert_allclose(_to_matrix(tensors), dense, atol=1e-12)
    assert [tensor.shape[3] for tensor in tensors[:-1]] == _ranks(dense, num_sites)


def test_dependent_operators_on_a_site_are_combined() -> None:
    """``I X - P0 X - P1 X + P0 Z`` is ``P0 Z``, since ``P0 + P1 = I`` is recognised, so its bond is one."""
    terms = [
        (1.0, {1: PAULI_X}),
        (-1.0, {0: PROJECTOR_ZERO, 1: PAULI_X}),
        (-1.0, {0: PROJECTOR_ONE, 1: PAULI_X}),
        (1.0, {0: PROJECTOR_ZERO, 1: PAULI_Z}),
    ]

    tensors = sum_of_products_tensors(terms, 2)

    np.testing.assert_allclose(_to_matrix(tensors), _dense([(1.0, {0: PROJECTOR_ZERO, 1: PAULI_Z})], 2), atol=1e-12)
    assert tensors[0].shape[3] == 1


def test_rank_deficient_coefficients_give_the_rank() -> None:
    """``(X + Y) (x) (X + Y)`` given as four terms is one product, which counting terms cannot see."""
    terms = [(1.0, {0: left, 1: right}) for left in (PAULI_X, PAULI_Y) for right in (PAULI_X, PAULI_Y)]

    tensors = sum_of_products_tensors(terms, 2)

    np.testing.assert_allclose(_to_matrix(tensors), _dense(terms, 2), atol=1e-12)
    assert tensors[0].shape[3] == 1


def test_small_exact_coefficient_is_kept() -> None:
    """A coefficient far above round-off keeps its string, however small against the others."""
    terms = [(1.0, {}), (1e-14, {0: PAULI_X, 2: PAULI_X})]

    tensors = sum_of_products_tensors(terms, 3)

    assert [tensor.shape[3] for tensor in tensors[:-1]] == [2, 2]
    # X on sites 0 and 2 maps |000> to |101>, an entry the identity does not touch.
    assert _to_matrix(tensors)[5, 0] == pytest.approx(1e-14, rel=1e-12)


def test_coefficient_at_round_off_counts_as_zero() -> None:
    """The float value of a vanishing coefficient, such as cos(pi / 2), drops its string."""
    terms = [(np.cos(np.pi / 2), {}), (1.0, {0: PAULI_Z, 1: PAULI_Z})]

    tensors = sum_of_products_tensors(terms, 2)

    assert tensors[0].shape[3] == 1


def test_cancelling_terms_give_the_zero_mpo() -> None:
    """Terms that add up to zero give the zero operator at bond one."""
    tensors = sum_of_products_tensors([(1.0, {0: PAULI_X}), (-1.0, {0: PAULI_X})], 3)

    assert [tensor.shape for tensor in tensors] == [(2, 2, 1, 1)] * 3
    assert not any(tensor.any() for tensor in tensors)
