# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent mixed-radix references for physical-site ordering tests."""

from __future__ import annotations

from itertools import product

import numpy as np


def mixed_radix_index(digits: tuple[int, ...], dimensions: tuple[int, ...]) -> int:
    """Return the flat index when site 0 is the least-significant subsystem.

    Args:
        digits: One computational-basis digit per physical site.
        dimensions: Local dimension of each physical site.

    Returns:
        Flat site-0-LSB basis index.
    """
    index = 0
    stride = 1
    for digit, dimension in zip(digits, dimensions, strict=True):
        index += digit * stride
        stride *= dimension
    return index


def embed_local_factors(
    factors: tuple[np.ndarray, ...],
    sites: tuple[int, ...],
    dimensions: tuple[int, ...],
) -> np.ndarray:
    """Embed local factors by enumerating physical basis digits.

    This reference deliberately does not use Kronecker products or YAQS
    conversion helpers.

    Args:
        factors: One local matrix for each entry in ``sites``.
        sites: Physical site for each local matrix.
        dimensions: Local dimension of every physical site.

    Returns:
        Full operator in site-0-LSB order.

    Raises:
        ValueError: If a factor shape does not match its physical site.
    """
    factor_by_site = dict(zip(sites, factors, strict=True))
    hilbert_dimension = int(np.prod(dimensions))
    embedded = np.zeros((hilbert_dimension, hilbert_dimension), dtype=np.complex128)
    basis = tuple(product(*(range(dimension) for dimension in dimensions)))
    for output_digits in basis:
        row = mixed_radix_index(output_digits, dimensions)
        for input_digits in basis:
            column = mixed_radix_index(input_digits, dimensions)
            element = 1.0 + 0.0j
            for site, dimension in enumerate(dimensions):
                factor = factor_by_site.get(site)
                if factor is None:
                    if output_digits[site] != input_digits[site]:
                        element = 0.0
                        break
                    continue
                if factor.shape != (dimension, dimension):
                    msg = f"factor on site {site} has shape {factor.shape}, expected {(dimension, dimension)}."
                    raise ValueError(msg)
                element *= factor[output_digits[site], input_digits[site]]
            embedded[row, column] = element
    return embedded


def embed_local_operator(
    operator: np.ndarray,
    sites: tuple[int, ...],
    dimensions: tuple[int, ...],
) -> np.ndarray:
    """Embed a local matrix by enumerating input and output basis digits.

    The first tensor factor of ``operator`` acts on ``sites[0]``, the second
    factor acts on ``sites[1]``, and so on. This reference deliberately does
    not use Kronecker products or YAQS conversion helpers.

    Args:
        operator: Local operator in the tensor-factor order given by ``sites``.
        sites: Ordered physical sites on which the operator acts.
        dimensions: Local dimension of every physical site.

    Returns:
        Full operator in site-0-LSB order.

    Raises:
        ValueError: If sites repeat, a site is out of range, or the operator
            shape does not match the selected site dimensions.
    """
    if len(set(sites)) != len(sites):
        msg = f"sites must be distinct, got {sites}."
        raise ValueError(msg)
    if any(site < 0 or site >= len(dimensions) for site in sites):
        msg = f"sites {sites} are invalid for {len(dimensions)} physical sites."
        raise ValueError(msg)

    local_dimensions = tuple(dimensions[site] for site in sites)
    local_size = int(np.prod(local_dimensions))
    operator_array = np.asarray(operator, dtype=np.complex128)
    if operator_array.shape != (local_size, local_size):
        msg = f"operator has shape {operator_array.shape}, expected {(local_size, local_size)}."
        raise ValueError(msg)

    hilbert_dimension = int(np.prod(dimensions))
    embedded = np.zeros((hilbert_dimension, hilbert_dimension), dtype=np.complex128)
    full_basis = tuple(product(*(range(dimension) for dimension in dimensions)))
    local_basis = tuple(product(*(range(dimension) for dimension in local_dimensions)))

    for input_digits in full_basis:
        column = mixed_radix_index(input_digits, dimensions)
        local_input = tuple(input_digits[site] for site in sites)
        local_column = mixed_radix_index(tuple(reversed(local_input)), tuple(reversed(local_dimensions)))
        for local_output in local_basis:
            output_digits = list(input_digits)
            for site, digit in zip(sites, local_output, strict=True):
                output_digits[site] = digit
            row = mixed_radix_index(tuple(output_digits), dimensions)
            local_row = mixed_radix_index(tuple(reversed(local_output)), tuple(reversed(local_dimensions)))
            embedded[row, column] = operator_array[local_row, local_column]
    return embedded
