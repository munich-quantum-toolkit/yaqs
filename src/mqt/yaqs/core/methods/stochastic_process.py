# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Stochastic Process of the Tensor Jump Method.

This module implements stochastic processes for quantum systems represented as Matrix Product States (MPS).
It provides functions to compute the stochastic factor, generate a probability distribution for quantum jumps
based on a noise model, and perform a stochastic (quantum jump) process on the state. These tools are used
to simulate noise-induced evolution in quantum many-body systems.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..data_structures.networks import MPS
    from ..data_structures.noise_model import NoiseModel


def calculate_stochastic_factor(state: MPS) -> NDArray[np.float64]:
    """Calculate the stochastic factor for a given state.

    This factor is used to determine the probability that a quantum jump will occur
    during the stochastic evolution. It is defined as 1 minus the norm of the state
    at site 0.

    Args:
        state (MPS): The Matrix Product State representing the current state of the system.
                     The state should be in mixed canonical form at site 0 or B normalized.

    Returns:
        NDArray[np.float64]: The calculated stochastic factor as a float.
    """
    return 1 - state.norm(0)


def create_probability_distribution(
    state: MPS, noise_model: NoiseModel | None, dt: float
) -> dict[str, list[NDArray[np.complex128] | np.float64 | float | int]]:
    """Create a probability distribution for potential quantum jumps in the system.

    For each site in the MPS and each jump operator in the noise model, this function calculates
    the probability that a quantum jump will occur at that site. The probability is computed as
    the product of the time step, the jump strength, and the scalar product of the state after the jump
    with itself. The resulting probabilities are normalized and returned along with the associated jump
    operators and site indices.

    Args:
        state (MPS): The Matrix Product State representing the current state of the system.
                     It must be in mixed canonical form at site 0 (i.e. not normalized).
        noise_model (NoiseModel | None): The noise model containing jump operators and their corresponding strengths.
            If None, an empty probability distribution is returned.
        dt (float): The time step for the evolution, used to scale the jump probabilities.

    Returns:
        dict[str, list]: A dictionary with the following keys:
            - "jumps": List of jump operator tensors.
            - "strengths": Corresponding jump strengths.
            - "sites": Site indices where each jump operator is applied.
            - "probabilities": Normalized probabilities for each jump.
    """
    # Ordered as [Jump 0 Site 0, Jump 1 Site 0, Jump 0 Site 1, Jump 1 Site 1, ...]
    jump_dict: dict[str, list[NDArray[np.complex128] | np.float64 | float | int]] = {
        "jumps": [],
        "strengths": [],
        "sites": [],
        "probabilities": [],
    }
    if noise_model is None:
        return jump_dict

    dp_m_list = []

    # Dissipative sweep should always result in a mixed canonical form at site L.
    for site, _ in enumerate(state.tensors):
        if site not in {0, state.length}:
            state.shift_orthogonality_center_right(site - 1)

        for j, jump_operator in enumerate(noise_model.jump_operators):
            jumped_state = copy.deepcopy(state)
            jumped_state.tensors[site] = oe.contract("ab, bcd->acd", jump_operator, state.tensors[site])
            dp_m = dt * noise_model.strengths[j] * jumped_state.norm(site)
            dp_m_list.append(dp_m.real)
            jump_dict["jumps"].append(jump_operator)
            jump_dict["strengths"].append(noise_model.strengths[j])
            jump_dict["sites"].append(site)

    # Normalize the probabilities.
    dp: np.float64 = np.sum(dp_m_list)
    jump_dict["probabilities"] = (dp_m_list / dp).astype(float)
    return jump_dict


def stochastic_process(state: MPS, noise_model: NoiseModel | None, dt: float) -> MPS:
    """Perform a stochastic process on the given state, simulating a quantum jump.

    The function calculates the stochastic factor for the state and, based on a random draw,
    determines whether a quantum jump should occur. If a jump is to occur, a jump operator is
    selected according to the probability distribution derived from the noise model and applied to
    the state. Otherwise, the state is simply normalized.

    Args:
        state (MPS): The current Matrix Product State, which should be in mixed canonical form at site 0.
        noise_model (NoiseModel | None): The noise model containing jump operators and their corresponding strengths.
            If None, no jump is performed.
        dt (float): The time step for the evolution, used to compute jump probabilities.

    Returns:
        MPS: The updated Matrix Product State after the stochastic process.
    """
    dp = calculate_stochastic_factor(state)
    rng = np.random.default_rng()
    if noise_model is None or rng.random() >= dp:
        # No jump occurs; shift the state to canonical form at site 0.
        state.shift_orthogonality_center_left(0)
        return state

    # A jump occurs: create the probability distribution and select a jump operator.
    jump_dict = create_probability_distribution(state, noise_model, dt)
    choices = list(range(len(jump_dict["probabilities"])))
    choice = rng.choice(choices, p=jump_dict["probabilities"])
    jump_operator = jump_dict["jumps"][choice]
    state.tensors[jump_dict["sites"][choice]] = oe.contract(
        "ab, bcd->acd", jump_operator, state.tensors[jump_dict["sites"][choice]]
    )
    state.normalize("B", decomposition="SVD")
    return state
