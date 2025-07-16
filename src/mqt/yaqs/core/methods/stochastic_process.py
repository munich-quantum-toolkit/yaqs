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

from ..methods.tdvp import merge_mps_tensors, split_mps_tensor

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from ..data_structures.networks import MPS
    from ..data_structures.noise_model import NoiseModel
    from ..data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams


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
    state: MPS,
    noise_model: NoiseModel | None,
    dt: float,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> dict[str, list[Any]]:
    """Create a probability distribution for potential quantum jumps in the system.

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
        state: The Matrix Product State, assumed left-canonical at site 0 on entry.
        noise_model: The noise model as a list of process dicts, each with keys
        "name", "strength", "sites", and "matrix" (list of length 1 or 2).
        dt : Time step for the evolution, used to scale the jump probabilities.
        sim_params: Simulation parameters, needed for splitting merged tensors (e.g., SVD threshold, bond dimension).

    Returns:
        dict[str, list]: A dictionary with the following keys:
            - "jumps": List of jump operator tensors.
            - "strengths": Corresponding jump strengths.
            - "sites": Site indices (list of 1 or 2 ints) where each jump operator is applied.
            - "probabilities": Normalized probabilities for each possible jump.
    """
    jump_dict: dict[str, list[Any]] = {"jumps": [], "strengths": [], "sites": [], "probabilities": []}

    if noise_model is None or not noise_model.processes:
        return jump_dict

    dp_m_list = []

    for site in range(state.length):
        # Shift ortho center to the right as needed (no shift for site 0)
        if site not in {0, state.length}:
            state.shift_orthogonality_center_right(site - 1)

        # --- 1-site jumps at this site ---
        for process in noise_model.processes:
            if len(process["sites"]) == 1 and process["sites"][0] == site:
                gamma = process["strength"]
                jump_op = process["matrix"]

                jumped_state = copy.deepcopy(state)
                jumped_state.tensors[site] = oe.contract("ab, bcd->acd", jump_op, state.tensors[site])
                dp_m = dt * gamma * jumped_state.norm(site)
                dp_m_list.append(dp_m.real)
                jump_dict["jumps"].append(jump_op)
                jump_dict["strengths"].append(gamma)
                jump_dict["sites"].append([site])

        # --- 2-site jumps starting at [site, site+1] ---
        if site < state.length - 1:
            for process in noise_model.processes:
                if len(process["sites"]) == 2 and process["sites"][0] == site and process["sites"][1] == site + 1:
                    gamma = process["strength"]
                    jump_op = process["matrix"]

                    jumped_state = copy.deepcopy(state)
                    # merge the tensors at site and site+1
                    tensor_left = jumped_state.tensors[site]
                    tensor_right = jumped_state.tensors[site + 1]
                    merged = merge_mps_tensors(tensor_left, tensor_right)
                    # apply the 2-site jump operator
                    merged = oe.contract("ab, bcd->acd", jump_op, merged)
                    dp_m = dt * gamma * jumped_state.norm(site)
                    # split the tensor (always contract singular values right for probabilities)
                    tensor_left_new, tensor_right_new = split_mps_tensor(
                        merged,
                        "right",
                        sim_params,
                        [state.physical_dimensions[site], state.physical_dimensions[site + 1]],
                        dynamic=False,
                    )
                    jumped_state.tensors[site], jumped_state.tensors[site + 1] = tensor_left_new, tensor_right_new
                    # compute the norm at `site`

                    dp_m_list.append(dp_m.real)
                    jump_dict["jumps"].append(jump_op)
                    jump_dict["strengths"].append(gamma)
                    jump_dict["sites"].append([site, site + 1])

    # Normalize the probabilities
    dp: float = np.sum(dp_m_list)
    jump_dict["probabilities"] = (np.array(dp_m_list) / dp).tolist() if dp > 0 else [0.0] * len(dp_m_list)
    return jump_dict


def stochastic_process(
    state: MPS,
    noise_model: NoiseModel | None,
    dt: float,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> MPS:
    """Perform a stochastic process on the given state, simulating a quantum jump.

    This function randomly determines whether a quantum jump occurs in the given
    timestep based on the system state and noise model. If a jump is triggered,
    the function samples the specific jump process according to the calculated
    probability distribution and applies the corresponding operator to the MPS.
    Both single-site and nearest-neighbor two-site jump processes are supported,
    with appropriate tensor contractions and normalization to ensure physical validity.

    Args:
        state: The current Matrix Product State, left-canonical at site 0.
        noise_model: The noise model, or None for no jumps.
        dt: The time step for the evolution.
        sim_params: Simulation parameters (for splitting tensors, required for 2-site jumps).

    Returns:
        MPS: The updated Matrix Product State after the stochastic process.

    Raises:
        ValueError: If a 2-site jump is not nearest-neighbor, or if the jump operator does not act on 1 or 2 sites.
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
    jump_op = jump_dict["jumps"][choice]
    sites = jump_dict["sites"][choice]

    if len(sites) == 1:
        # 1-site jump
        site = sites[0]
        state.tensors[site] = oe.contract("ab, bcd->acd", jump_op, state.tensors[site])
    elif len(sites) == 2:
        # 2-site jump: merge, apply, split
        i, j = sites
        # Ensure j == i+1
        if j != i + 1:
            msg = f"Only nearest-neighbor 2-site jumps are supported (got sites {i}, {j})"
            raise ValueError(msg)
        merged = merge_mps_tensors(state.tensors[i], state.tensors[j])
        merged = oe.contract("ab, bcd->acd", jump_op, merged)
        # For stochastic jumps, always contract singular values to the right
        tensor_left_new, tensor_right_new = split_mps_tensor(
            merged, "right", sim_params, [state.physical_dimensions[i], state.physical_dimensions[j]], dynamic=False
        )
        state.tensors[i], state.tensors[j] = tensor_left_new, tensor_right_new
    else:
        msg = "Jump operator must act on 1 or 2 sites."
        raise ValueError(msg)

    # Normalize MPS after jump
    state.normalize("B", decomposition="SVD")
    return state
