# Copyright (c) 2025 Chair for Design Automation, TUM
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
from ..methods.tdvp import split_mps_tensor, merge_mps_tensors

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


# def create_probability_distribution(
#     state: MPS, noise_model: NoiseModel | None, dt: float
# ) -> dict[str, list[NDArray[np.complex128] | np.float64 | float | int]]:
#     """Create a probability distribution for potential quantum jumps in the system.

#     For each site in the MPS and each jump operator in the noise model, this function calculates
#     the probability that a quantum jump will occur at that site. The probability is computed as
#     the product of the time step, the jump strength, and the scalar product of the state after the jump
#     with itself. The resulting probabilities are normalized and returned along with the associated jump
#     operators and site indices.

#     Args:
#         state (MPS): The Matrix Product State representing the current state of the system.
#                      It must be in mixed canonical form at site 0 (i.e. not normalized).
#         noise_model (NoiseModel | None): The noise model containing jump operators and their corresponding strengths.
#             If None, an empty probability distribution is returned.
#         dt (float): The time step for the evolution, used to scale the jump probabilities.

#     Returns:
#         dict[str, list]: A dictionary with the following keys:
#             - "jumps": List of jump operator tensors.
#             - "strengths": Corresponding jump strengths.
#             - "sites": Site indices where each jump operator is applied.
#             - "probabilities": Normalized probabilities for each jump.
#     """
#     # Ordered as [Jump 0 Site 0, Jump 1 Site 0, Jump 0 Site 1, Jump 1 Site 1, ...]
#     jump_dict: dict[str, list[NDArray[np.complex128] | np.float64 | float | int]] = {
#         "jumps": [],
#         "strengths": [],
#         "sites": [],
#         "probabilities": [],
#     }
#     if noise_model is None:
#         return jump_dict

#     dp_m_list = []
#     print(f'state ortho center {state.check_canonical_form()}')

#     # Dissipative sweep should always result in a mixed canonical form at site L.
#     for site, _ in enumerate(state.tensors):
#         if site not in {0, state.length}:
#             state.shift_orthogonality_center_right(site - 1)

#         for j, jump_operator in enumerate(noise_model.jump_operators):
#             jumped_state = copy.deepcopy(state)
#             jumped_state.tensors[site] = oe.contract("ab, bcd->acd", jump_operator, state.tensors[site])
#             dp_m = dt * noise_model.strengths[j] * jumped_state.norm(site)
#             dp_m_list.append(dp_m.real)
#             jump_dict["jumps"].append(jump_operator)
#             jump_dict["strengths"].append(noise_model.strengths[j])
#             jump_dict["sites"].append(site)

#     # Normalize the probabilities.
#     dp: np.float64 = np.sum(dp_m_list)
#     jump_dict["probabilities"] = (dp_m_list / dp).astype(float)
#     return jump_dict



def create_probability_distribution(
    state: MPS, noise_model: NoiseModel | None, dt: float, sim_params=None
) -> dict[str, list]:
    """
    Create a probability distribution for potential quantum jumps in the system,
    supporting both 1-site and 2-site jump operators.

    The function sweeps from left to right over the sites of the MPS. For each site,
    it shifts the orthogonality center to that site if necessary and then considers all
    relevant jump operators in the noise model:
      - For each 1-site jump operator acting on the current site, it constructs a candidate
        post-jump state, computes the corresponding quantum jump probability (proportional to the
        time step, jump strength, and post-jump norm at that site), and records the operator and
        site.
      - For each 2-site jump operator acting on the current site and its right neighbor,
        it merges the two tensors, applies the operator, splits the result, computes the probability,
        and records the operator and the site pair.

    After all possible jumps are considered, the probabilities are normalized and returned along with
    the associated jump operators and their target site(s).

    Args:
        state (MPS): The Matrix Product State, assumed left-canonical at site 0 on entry.
        noise_model (NoiseModel | None): The noise model as a list of process dicts, each with keys
            "jump_operator", "strength", and "sites" (list of length 1 or 2).
        dt (float): Time step for the evolution, used to scale the jump probabilities.
        sim_params: Simulation parameters, needed for splitting merged tensors (e.g., SVD threshold, bond dimension).

    Returns:
        dict[str, list]: A dictionary with the following keys:
            - "jumps": List of jump operator tensors.
            - "strengths": Corresponding jump strengths.
            - "sites": Site indices (list of 1 or 2 ints) where each jump operator is applied.
            - "probabilities": Normalized probabilities for each possible jump.
    """

    jump_dict = {"jumps": [], "strengths": [], "sites": [], "probabilities": []}
    if noise_model is None or not noise_model.processes:
        return jump_dict

    dp_m_list = []
    n_sites = state.length

    for site in range(n_sites):
        # Shift ortho center to the right as needed (no shift for site 0)
        if site not in {0, n_sites}:
            state.shift_orthogonality_center_right(site - 1)

        # --- 1-site jumps at this site ---
        for process in noise_model.processes:
            if len(process["sites"]) == 1 and process["sites"][0] == site:
                gamma = process["strength"]
                L = process["jump_operator"]

                jumped_state = copy.deepcopy(state)
                jumped_state.tensors[site] = oe.contract("ab, bcd->acd", L, state.tensors[site])
                dp_m = dt * gamma * jumped_state.norm(site)
                dp_m_list.append(dp_m.real)
                jump_dict["jumps"].append(L)
                jump_dict["strengths"].append(gamma)
                jump_dict["sites"].append([site])

        # --- 2-site jumps starting at [site, site+1] ---
        if site < n_sites - 1:
            for process in noise_model.processes:
                if len(process["sites"]) == 2 and process["sites"][0] == site and process["sites"][1] == site + 1:
                    gamma = process["strength"]
                    L = process["jump_operator"]

                    jumped_state = copy.deepcopy(state)
                    # merge the tensors at site and site+1
                    A = jumped_state.tensors[site]
                    B = jumped_state.tensors[site + 1]
                    merged = merge_mps_tensors(A, B)
                    # apply the 2-site jump operator
                    merged = oe.contract("ab, bcd->acd", L, merged)
                    dp_m = dt * gamma * jumped_state.norm(site)
                    # split the tensor (always contract singular values right for probabilities)
                    A_new, B_new = split_mps_tensor(merged, "right", sim_params, dynamic=False)
                    jumped_state.tensors[site], jumped_state.tensors[site + 1] = A_new, B_new
                    # compute the norm at `site`
                    
                    dp_m_list.append(dp_m.real)
                    jump_dict["jumps"].append(L)
                    jump_dict["strengths"].append(gamma)
                    jump_dict["sites"].append([site, site + 1])

    # Normalize the probabilities
    dp = np.sum(dp_m_list)
    jump_dict["probabilities"] = (np.array(dp_m_list) / dp).tolist() if dp > 0 else [0.0] * len(dp_m_list)
    return jump_dict



# def stochastic_process(state: MPS, noise_model: NoiseModel | None, dt: float) -> MPS:
#     """Perform a stochastic process on the given state, simulating a quantum jump.

#     The function calculates the stochastic factor for the state and, based on a random draw,
#     determines whether a quantum jump should occur. If a jump is to occur, a jump operator is
#     selected according to the probability distribution derived from the noise model and applied to
#     the state. Otherwise, the state is simply normalized.

#     Args:
#         state (MPS): The current Matrix Product State, which should be in mixed canonical form at site 0.
#         noise_model (NoiseModel | None): The noise model containing jump operators and their corresponding strengths.
#             If None, no jump is performed.
#         dt (float): The time step for the evolution, used to compute jump probabilities.

#     Returns:
#         MPS: The updated Matrix Product State after the stochastic process.
#     """
#     dp = calculate_stochastic_factor(state)
#     rng = np.random.default_rng()
#     if noise_model is None or rng.random() >= dp:
#         # No jump occurs; shift the state to canonical form at site 0.
#         state.shift_orthogonality_center_left(0)
#         return state

#     # A jump occurs: create the probability distribution and select a jump operator.
#     jump_dict = create_probability_distribution(state, noise_model, dt)
#     choices = list(range(len(jump_dict["probabilities"])))
#     choice = rng.choice(choices, p=jump_dict["probabilities"])
#     jump_operator = jump_dict["jumps"][choice]
#     state.tensors[jump_dict["sites"][choice]] = oe.contract(
#         "ab, bcd->acd", jump_operator, state.tensors[jump_dict["sites"][choice]]
#     )
#     state.normalize("B", decomposition="SVD")
#     return state

def stochastic_process(state: MPS, noise_model: NoiseModel | None, dt: float, sim_params=None) -> MPS:
    """
    Perform a stochastic process on the given state, simulating a quantum jump.
    Supports both 1-site and 2-site jump operators.
    
    Args:
        state (MPS): The current Matrix Product State, left-canonical at site 0.
        noise_model (NoiseModel | None): The noise model, or None for no jumps.
        dt (float): The time step for the evolution.
        sim_params: Simulation parameters (for splitting tensors, required for 2-site jumps).
    
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
    jump_dict = create_probability_distribution(state, noise_model, dt, sim_params)
    choices = list(range(len(jump_dict["probabilities"])))
    choice = rng.choice(choices, p=jump_dict["probabilities"])
    jump_operator = jump_dict["jumps"][choice]
    sites = jump_dict["sites"][choice]

    if len(sites) == 1:
        # 1-site jump
        site = sites[0]
        state.tensors[site] = oe.contract(
            "ab, bcd->acd", jump_operator, state.tensors[site]
        )
    elif len(sites) == 2:
        # 2-site jump: merge, apply, split
        i, j = sites
        # Ensure j == i+1
        if j != i + 1:
            raise ValueError(f"Only nearest-neighbor 2-site jumps are supported (got sites {i}, {j})")
        merged = merge_mps_tensors(state.tensors[i], state.tensors[j])
        merged = oe.contract("ab, bcd->acd", jump_operator, merged)
        # For stochastic jumps, always contract singular values to the right
        A_new, B_new = split_mps_tensor(merged, "right", sim_params, dynamic=False)
        state.tensors[i], state.tensors[j] = A_new, B_new
    else:
        raise ValueError("Jump operator must act on 1 or 2 sites.")

    # Normalize MPS after jump
    state.normalize("B", decomposition="SVD")
    return state

