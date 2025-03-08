# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from scipy.linalg import expm

if TYPE_CHECKING:
    from ..data_structures.networks import MPS
    from ..data_structures.noise_model import NoiseModel


# TODO(Aaron): Assumes noise is same at all sites
#       Could be sped-up by pre-calculating exponential somewhere else
#       Likely not a problem since it's only exponentiating small matrices
def apply_dissipation(state: MPS, noise_model: NoiseModel | None, dt: float) -> None:
    """Apply dissipation to the system state using a given noise model and time step.

    This function modifies the state tensors of an MPS by applying a dissipative operator
    that is calculated from the noise model's jump operators and strengths. The operator is
    computed by exponentiating a matrix derived from these jump operators, and then applied to
    each tensor in the state using an Einstein summation contraction.

    Args:
        state (MPS): The Matrix Product State representing the current state of the system.
        noise_model (NoiseModel | None): The noise model containing jump operators and their
            corresponding strengths. If None or if all strengths are zero, no dissipation is applied.
        dt (float): The time step for the evolution, used in the exponentiation of the dissipative operator.

    Returns:
        None

    Notes:
        - If no noise is present (i.e. `noise_model` is None or all noise strengths are zero),
          the function shifts the orthogonality center of the MPS tensors and returns early.
        - The dissipation operator A is calculated as a sum over each jump operator, where each
          term is given by (noise strength) * (conjugate transpose of the jump operator) multiplied
          by the jump operator.
        - The dissipative operator is computed using the matrix exponential `expm(-0.5 * dt * A)`.
        - The operator is then applied to each tensor in the MPS via a contraction using `opt_einsum`.
    """
    # Check if noise is absent or has zero strength; if so, simply shift the state to canonical form.
    if noise_model is None or all(gamma == 0 for gamma in noise_model.strengths):
        for i in reversed(range(state.length)):
            state.shift_orthogonality_center_left(current_orthogonality_center=i)
        return

    # Calculate the dissipation operator A from the noise model.
    A = sum(
        noise_model.strengths[i] * np.conj(jump_operator).T @ jump_operator
        for i, jump_operator in enumerate(noise_model.jump_operators)
    )

    # Compute the dissipative operator by exponentiating -0.5 * dt * A.
    dissipative_operator = expm(-0.5 * dt * A)

    # Apply the dissipative operator to each tensor in the MPS.
    # The contraction "ab, bcd->acd" applies the operator on the physical indices.
    for i in reversed(range(state.length)):
        state.tensors[i] = oe.contract("ab, bcd->acd", dissipative_operator, state.tensors[i])
        # Prepare the state for probability calculation by shifting the orthogonality center.
        # Shifting during the sweep is more efficient than setting it only once at the end.
        if i != 0:
            state.shift_orthogonality_center_left(current_orthogonality_center=i)
