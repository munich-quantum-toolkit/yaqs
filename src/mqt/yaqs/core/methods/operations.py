# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module implements functions for computing expectation values and performing measurements
on quantum states represented as Matrix Product States (MPS). It provides routines for calculating
the scalar (inner) product between MPS objects, evaluating local expectation values of operators,
and simulating projective measurements via single-shot and multi-shot strategies. Parallel execution
of measurements is supported using a ProcessPoolExecutor with a progress bar via tqdm.
"""

from __future__ import annotations

import concurrent.futures
import copy
import multiprocessing
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from tqdm import tqdm

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
        for site in range(A.length):
            tensor = oe.contract("abc, ade->bdce", A_copy.tensors[site], B_copy.tensors[site])
            result = tensor if site == 0 else oe.contract("abcd, cdef->abef", result, tensor)
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
    # TODO(Aaron): Could be more memory-efficient by not copying state
    temp_state = copy.deepcopy(state)
    temp_state.tensors[site] = oe.contract("ab, bcd->acd", operator, temp_state.tensors[site])
    return scalar_product(state, temp_state, site)


def measure_single_shot(state: MPS) -> int:
    """Perform a single-shot measurement on a Matrix Product State (MPS).

    This function simulates a projective measurement on an MPS. For each site, it computes the
    local reduced density matrix from the site's tensor, derives the probability distribution over
    basis states, and randomly selects an outcome. The overall measurement result is encoded as an
    integer corresponding to the measured bitstring.

    Args:
        state (MPS): The MPS state to be measured.

    Returns:
        int: The measurement outcome represented as an integer.
    """
    temp_state = copy.deepcopy(state)
    bitstring = []
    for site, tensor in enumerate(temp_state.tensors):
        reduced_density_matrix = oe.contract("abc, dbc->ad", tensor, np.conj(tensor))
        probabilities = np.diag(reduced_density_matrix).real
        rng = np.random.default_rng()
        chosen_index = rng.choice(len(probabilities), p=probabilities)
        bitstring.append(chosen_index)
        selected_state = np.zeros(len(probabilities))
        selected_state[chosen_index] = 1
        # Multiply state: project the tensor onto the selected state.
        tensor = oe.contract("a, acd->cd", selected_state, tensor)
        # Propagate the measurement to the next site.
        if site != state.length - 1:
            temp_state.tensors[site + 1] = (
                1
                / np.sqrt(probabilities[chosen_index])
                * oe.contract("ab, cbd->cad", tensor, temp_state.tensors[site + 1])
            )
    return sum(c << i for i, c in enumerate(bitstring))


def measure(state: MPS, shots: int) -> dict[int, int]:
    """Perform multiple single-shot measurements on an MPS and aggregate the results.

    This function executes a specified number of measurement shots on the given MPS. For each shot,
    a single-shot measurement is performed, and the outcomes are aggregated into a histogram (dictionary)
    mapping basis states (represented as integers) to the number of times they were observed.

    Args:
        state (MPS): The Matrix Product State to be measured.
        shots (int): The number of measurement shots to perform.

    Returns:
        dict[int, int]: A dictionary where keys are measured basis states (as integers) and values are
        the corresponding counts.

    Notes:
        - When more than one shot is requested, measurements are parallelized using a ProcessPoolExecutor.
        - A progress bar (via tqdm) displays the progress of the measurement process.
    """
    results: dict[int, int] = {}
    if shots > 1:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=shots, desc="Measuring shots", ncols=80) as pbar:
                futures = [executor.submit(measure_single_shot, copy.deepcopy(state)) for _ in range(shots)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results[result] = results.get(result, 0) + 1
                    except Exception:
                        pass
                    finally:
                        pbar.update(1)
        return results
    basis_state = measure_single_shot(state)
    results[basis_state] = results.get(basis_state, 0) + 1
    return results
