# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dissipative sweep of the Tensor Jump Method.

This module implements a function to apply dissipation to a quantum state represented as an MPS.
The dissipative operator is computed from a noise model by exponentiating a weighted sum of jump operators,
and is then applied to each tensor in the MPS via tensor contraction. If no noise is present or if all
noise strengths are zero, the MPS is simply shifted to its canonical form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from scipy.linalg import expm
from collections import defaultdict

from ..libraries.noise_library import NoiseLibrary
from ..methods.tdvp import split_mps_tensor, merge_mps_tensors

if TYPE_CHECKING:
    from ..data_structures.networks import MPS
    from ..data_structures.noise_model import NoiseModel
    


# def apply_dissipation(state: MPS, noise_model: NoiseModel | None, dt: float) -> None:
#     """Apply dissipation to the system state using a given noise model and time step.

#     This function modifies the state tensors of an MPS by applying a dissipative operator
#     that is calculated from the noise model's jump operators and strengths. The operator is
#     computed by exponentiating a matrix derived from these jump operators, and then applied to
#     each tensor in the state using an Einstein summation contraction.

#     Args:
#         state (MPS): The Matrix Product State representing the current state of the system.
#         noise_model (NoiseModel | None): The noise model containing jump operators and their
#             corresponding strengths. If None or if all strengths are zero, no dissipation is applied.
#         dt (float): The time step for the evolution, used in the exponentiation of the dissipative operator.



#     Notes:
#         - If no noise is present (i.e. `noise_model` is None or all noise strengths are zero),
#           the function shifts the orthogonality center of the MPS tensors and returns early.
#         - The dissipation operator A is calculated as a sum over each jump operator, where each
#           term is given by (noise strength) * (conjugate transpose of the jump operator) multiplied
#           by the jump operator.
#         - The dissipative operator is computed using the matrix exponential `expm(-0.5 * dt * A)`.
#         - The operator is then applied to each tensor in the MPS via a contraction using `opt_einsum`.
#     """
#     # Check if noise is absent or has zero strength; if so, simply shift the state to canonical form.
#     if (noise_model is None or all(all(gamma == 0 for gamma in site_gammas) for site_gammas in noise_model.strengths)):
#         for i in reversed(range(state.length)):
#             state.shift_orthogonality_center_left(current_orthogonality_center=i, decomposition="QR")
#         return

#     # # Calculate the dissipation matrix from the noise model.
#     # mat = sum(
#     #     noise_model.strengths[i] * np.conj(jump_operator).T @ jump_operator
#     #     for i, jump_operator in enumerate(noise_model.jump_operators)
#     # )

#     # # Compute the dissipative operator by exponentiating -0.5 * dt * A.
#     # dissipative_operator = expm(-0.5 * dt * mat)

#     # Apply the dissipative operator to each tensor in the MPS.
#     # The contraction "ab, bcd->acd" applies the operator on the physical indices.
#     for i in reversed(range(state.length)):
#         mat = 0
#         #print('Processing site:', i)
#         for j, process in enumerate(noise_model.processes[i]):
#             jump_operator = getattr(NoiseLibrary, process)().matrix
#             mat += noise_model.strengths[i][j] * np.conj(jump_operator).T @ jump_operator
#             #print('mat:', mat)
#             #print('mat shape:', mat.shape)
#         # if mat == 0:
#         #     if i != 0:
#         #         state.shift_orthogonality_center_left(current_orthogonality_center=i, decomposition="SVD")
#         #         return

#         # Compute the dissipative operator by exponentiating -0.5 * dt * A.
#         dissipative_operator = expm(-0.5 * dt * mat)
#         #print('state.tensors[i].shape, dissipative_operator.shape):', state.tensors[i].shape, dissipative_operator.shape)
#         state.tensors[i] = oe.contract("ab, bcd->acd", dissipative_operator, state.tensors[i])
#         #print('state.tensors[i].shape after contraction:', state.tensors[i].shape)
#         # Prepare the state for probability calculation by shifting the orthogonality center.
#         # Shifting during the sweep is more efficient than setting it only once at the end.
#         if i != 0:
#             state.shift_orthogonality_center_left(current_orthogonality_center=i, decomposition="SVD")


def apply_dissipation(state: MPS, noise_model: NoiseModel, dt: float, sim_params) -> None:
    """
    Dissipative sweep: right-to-left, compatible with left-canonical MPS.
    Assumes state is left-canonical at start.
    """
    n_sites = state.length


    # Prepare: For each bond, collect all 2-site processes acting on that bond
    two_site_on_bond = defaultdict(list)
    for process in noise_model.processes:
        if len(process["sites"]) == 2:
            bond = tuple(sorted(process["sites"]))  # e.g. (i-1, i)
            two_site_on_bond[bond].append(process)


    for i in reversed(range(n_sites)):
        # 1. Apply all 1-site dissipators on site i
        for process in noise_model.processes:
            if len(process["sites"]) == 1 and process["sites"][0] == i:
                gamma = process["strength"]
                L = process["jump_operator"]
                mat = np.conj(L).T @ L
                dissipative_operator = expm(-0.5 * dt * gamma * mat)
                state.tensors[i] = oe.contract("ab, bcd->acd", dissipative_operator, state.tensors[i])

            bond = (i-1, i)
            processes_here = two_site_on_bond.get(bond, [])
            n_procs = len(processes_here)

        # 2. Apply all 2-site dissipators acting on sites (i-1, i)
        if i!=0:
            for idx, process in enumerate(processes_here):
                gamma = process["strength"]
                L = process["jump_operator"]
                mat = np.conj(L).T @ L
                dissipative_operator = expm(-0.5 * dt * gamma * mat)

                merged_tensor = merge_mps_tensors(state.tensors[i-1], state.tensors[i])
                merged_tensor = oe.contract("ab, bcd->acd", dissipative_operator, merged_tensor)

                # singular values always contracted right
                # since ortho center is shifter to the left after loop
                A, B = split_mps_tensor(merged_tensor, 'right', sim_params, dynamic=False)
                state.tensors[i-1], state.tensors[i] = A, B
                

        # Shift orthogonality center
        if i != 0:
            state.shift_orthogonality_center_left(current_orthogonality_center=i, decomposition="SVD")