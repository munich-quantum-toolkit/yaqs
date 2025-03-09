# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""General tensor network methods.

This module implements functions for computing expectation values and performing measurements
on quantum states represented as Matrix Product States (MPS). It provides routines for calculating
the scalar (inner) product between MPS objects, evaluating local expectation values of operators,
and simulating projective measurements via single-shot and multi-shot strategies. Parallel execution
of measurements is supported using a ProcessPoolExecutor with a progress bar via tqdm.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..data_structures.networks import MPS


def scalar_product(A: MPS, B: MPS, site: int | None = None) -> np.complex128:
    """Compute the scalar (inner) product between two Matrix Product States (MPS).

    The function contracts the corresponding tensors of two MPS objects. If no specific site is
    provided, the contraction is performed sequentially over all sites to yield the overall inner
    product. When a site is specified, only the tensors at that site are contracted.

    Args:
        A (MPS): The first Matrix Product State.
        B (MPS): The second Matrix Product State.
        site (int | None): Optional site index at which to compute the contraction. If None, the
            contraction is performed over all sites.

    Returns:
        np.complex128: The resulting scalar product as a complex number.
    """
    A_copy = copy.deepcopy(A)
    B_copy = copy.deepcopy(B)
    for i, tensor in enumerate(A_copy.tensors):
        A_copy.tensors[i] = np.conj(tensor)

    result = np.array(np.inf)
    if site is None:
        for idx in range(A.length):
            tensor = oe.contract("abc, ade->bdce", A_copy.tensors[idx], B_copy.tensors[idx])
            result = tensor if idx == 0 else oe.contract("abcd, cdef->abef", result, tensor)
    else:
        result = oe.contract("ijk, ijk", A_copy.tensors[site], B_copy.tensors[site])
    return np.complex128(np.squeeze(result))


def local_expval(state: MPS, operator: NDArray[np.complex128], site: int) -> np.complex128:
    """Compute the local expectation value of an operator on an MPS.

    The function applies the given operator to the tensor at the specified site of a deep copy of the
    input MPS, then computes the scalar product between the original and the modified state at that site.
    This effectively calculates the expectation value of the operator at the specified site.

    Args:
        state (MPS): The Matrix Product State representing the quantum state.
        operator (NDArray[np.complex128]): The local operator (matrix) to be applied.
        site (int): The index of the site at which to evaluate the expectation value.

    Returns:
        np.complex128: The computed expectation value (typically, its real part is of interest).

    Notes:
        A deep copy of the state is used to prevent modifications to the original MPS.
    """
    temp_state = copy.deepcopy(state)
    temp_state.tensors[site] = oe.contract("ab, bcd->acd", operator, temp_state.tensors[site])
    return scalar_product(state, temp_state, site)
