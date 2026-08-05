# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
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
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import opt_einsum as oe

from mqt.yaqs.core.methods.dissipation import is_longrange, is_pauli

from ..methods.decompositions import merge_two_site, split_two_site

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..data_structures.mps import MPS
    from ..data_structures.noise_model import NoiseModel
    from ..data_structures.simulation_parameters import AnalogSimParams, DigitalSimParams
    from ..methods.decompositions import TruncMode


def calculate_stochastic_factor(state: MPS) -> NDArray[np.float64]:
    """Calculate the stochastic factor for a given state.

    This factor is used to determine the probability that a quantum jump will occur
    during the stochastic evolution. It is defined as 1 minus the norm of the state
    at site 0.

    Args:
        state: The Matrix Product MPS representing the current state of the system.
            The state should be in mixed canonical form at site 0 or B normalized.

    Returns:
        The calculated stochastic factor as a float.
    """
    return np.asarray(1 - state.norm(0), dtype=np.float64)


def _adjacent_jump_weight(
    state: MPS,
    site: int,
    jump_op: NDArray[np.complex128],
    sim_params: AnalogSimParams | DigitalSimParams,
) -> float:
    """Return ``||L|psi>||^2`` for an adjacent two-site jump without truncating.

    When the MPS is mixed-canonical at ``site``, the Frobenius weight of the
    untruncated post-jump block equals the global squared norm. With unknown gauge,
    an untruncated split is written and the global norm is used. Truncation for
    bond-dimension control belongs in the jump-application path, not PDF weights.
    """
    merged = merge_two_site(state.tensors[site], state.tensors[site + 1])
    merged = oe.contract("ab, bcd->acd", jump_op, merged)
    if state.orthogonality_center is not None:
        return float(np.vdot(merged, merged).real)

    jumped_state = copy.deepcopy(state)
    tensor_left_new, tensor_right_new = split_two_site(
        merged,
        [state.physical_dimensions[site], state.physical_dimensions[site + 1]],
        svd_distribution="right",
        trunc_mode=cast("TruncMode", sim_params.trunc_mode),
        threshold=0.0,
        max_bond_dim=None,
    )
    jumped_state.tensors[site] = tensor_left_new
    jumped_state.tensors[site + 1] = tensor_right_new
    jumped_state.set_center(None)
    return float(jumped_state.norm())


def create_probability_distribution(
    state: MPS,
    noise_model: NoiseModel | None,
    dt: float,
    sim_params: AnalogSimParams | DigitalSimParams,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Create a probability distribution for potential quantum jumps in the system.

    The function sweeps from left to right over the sites of the MPS. For each
    site, it shifts the orthogonality center to that site if necessary and then
    considers all relevant jump operators in the noise model:

    - For each 1-site jump operator acting on the current site, it constructs a
      candidate post-jump state, computes the corresponding quantum jump
      probability (proportional to the time step, jump strength, and post-jump
      norm at that site), and records the operator and site.
    - For each 2-site jump operator acting on the current site and its right
      neighbor, it merges the two tensors, applies the operator, and computes
      the probability from the untruncated post-jump block (truncation is
      deferred until a channel is selected), then records the operator and site
      pair.

    After all possible jumps are considered, the per-process probabilities are
    normalized and returned together with the corresponding processes in the same
    site-sweep order. ``stochastic_process`` must index both lists with the same
    ``choice_idx``.

    Args:
        state: The Matrix Product MPS, assumed left-canonical at site 0 on entry.
        noise_model: The noise model as a list of process dicts, each with keys
            "name", "strength", "sites", and "matrix" (for 1-site and adjacent 2-site processes)
            or "factors" (for long-range 2-site processes).
        dt: Time step for the evolution, used to scale the jump probabilities.
        sim_params: Simulation parameters, needed for splitting merged tensors (e.g., SVD threshold, bond dimension).

    Returns:
        A tuple ``(ordered_processes, probabilities)`` where ``ordered_processes`` are
        the applicable jump processes in site-sweep order and ``probabilities`` are
        the corresponding normalized jump probabilities.

    Raises:
        ValueError: If a non-Pauli long-range two-site process is present.
    """
    if noise_model is None or not noise_model.processes:
        return [], []

    if state.orthogonality_center is not None:
        state.assert_center(0, context="create_probability_distribution")

    ordered_processes: list[dict[str, Any]] = []
    dp_m_list: list[float] = []

    for site in range(state.length):
        if site != 0 and state.orthogonality_center is not None:
            state.shift_center_to(site)

        # --- 1-site jumps at this site ---
        for process in noise_model.processes:
            if len(process["sites"]) == 1 and process["sites"][0] == site:
                gamma = process["strength"]
                jump_op = process["matrix"]

                jumped_state = copy.deepcopy(state)
                jumped_state.tensors[site] = oe.contract("ab, bcd->acd", jump_op, state.tensors[site])
                dp_m = dt * gamma * jumped_state.norm(site)
                ordered_processes.append(process)
                dp_m_list.append(float(dp_m.real))

        # --- 2-site jumps starting at [site, site+1] ---
        if site < state.length - 1:
            for process in noise_model.processes:
                if len(process["sites"]) == 2 and process["sites"][0] == site:
                    if is_pauli(process):
                        gamma = process["strength"]
                        dp_m = dt * gamma * state.norm(site)
                        ordered_processes.append(process)
                        dp_m_list.append(float(dp_m.real))

                    elif process["sites"][1] == site + 1:
                        gamma = process["strength"]
                        weight = _adjacent_jump_weight(state, site, process["matrix"], sim_params)
                        dp_m = dt * gamma * weight
                        ordered_processes.append(process)
                        dp_m_list.append(float(dp_m.real))
                    else:
                        msg = (
                            "Non-Pauli long-range two-site jumps are not supported "
                            f"(process '{process['name']}' on sites {process['sites']})."
                        )
                        raise ValueError(msg)

    # Normalize the probabilities
    dp: float = float(np.sum(dp_m_list))
    if not np.isfinite(dp) or dp <= 0.0:
        msg = (
            "Jump probability weights are zero or non-finite. "
            "Reduce process strengths and/or the timestep dt so that "
            "dt * strength * ||L|psi>||^2 remains representable."
        )
        raise ValueError(msg)
    return ordered_processes, [val / dp for val in dp_m_list]


def stochastic_process(
    state: MPS,
    noise_model: NoiseModel | None,
    dt: float,
    sim_params: AnalogSimParams | DigitalSimParams,
    rng: np.random.Generator | None = None,
) -> MPS:
    """Perform a stochastic process on the given state, simulating a quantum jump.

    This function randomly determines whether a quantum jump occurs in the given
    timestep based on the system state and noise model. If a jump is triggered,
    the function samples the specific jump process according to the calculated
    probability distribution and applies the corresponding operator to the MPS.
    Both single-site and nearest-neighbor two-site jump processes are supported,
    with appropriate tensor contractions and normalization to ensure physical validity.

    Args:
        state: The current Matrix Product MPS, left-canonical at site 0.
        noise_model: The noise model, or None for no jumps.
        dt: The time step for the evolution.
        sim_params: Simulation parameters (for splitting tensors, required for 2-site jumps).
        rng: The random number generator to use. If None, valid global rng or new generator is used.

    Returns:
        MPS: The updated Matrix Product MPS after the stochastic process.

    Raises:
        ValueError: If a 2-site jump is not nearest-neighbor, or if the jump operator does not act on 1 or 2 sites.
    """
    if rng is None:
        rng = np.random.default_rng()

    if state.orthogonality_center is not None:
        state.assert_center(0, context="stochastic_process")

    dp = calculate_stochastic_factor(state)
    if noise_model is None or rng.random() >= dp:
        if state.orthogonality_center is not None:
            state.shift_orthogonality_center_left(0)
        else:
            state.set_canonical_form(0)
        return state

    # A jump occurs: create the probability distribution and select a jump operator.
    ordered_processes, probabilities = create_probability_distribution(state, noise_model, dt, sim_params)

    if len(probabilities) == 0:
        if state.orthogonality_center is not None:
            if state.orthogonality_center != 0:
                state.shift_center_to(0)
            state.shift_orthogonality_center_left(0)
        else:
            state.set_canonical_form(0)
        return state

    choice_idx = rng.choice(len(ordered_processes), p=probabilities)
    chosen_process = ordered_processes[choice_idx]

    # Extract information from chosen process
    sites = chosen_process["sites"]

    if len(sites) == 1:
        # 1-site jump
        site = sites[0]
        jump_op = chosen_process["matrix"]
        state.tensors[site] = oe.contract("ab, bcd->acd", jump_op, state.tensors[site])
        if state.orthogonality_center is not None and state.orthogonality_center != site:
            state.set_center(None)

    else:
        # 2-site jump: check if long-range or adjacent
        i, j = sites

        if is_pauli(chosen_process) and is_longrange(chosen_process):
            jump_op_0, jump_op_1 = chosen_process["factors"][0], chosen_process["factors"][1]
            state.tensors[i] = oe.contract("ab, bcd->acd", jump_op_0, state.tensors[i])
            state.tensors[j] = oe.contract("ab, bcd->acd", jump_op_1, state.tensors[j])
            state.set_center(None)
        else:
            # Adjacent 2-site process: use matrix
            if np.abs(i - j) > 1:
                msg = f"Only nearest-neighbor 2-site jumps are supported for non-Pauli processes (got sites {i}, {j})"
                raise ValueError(msg)

            jump_op = chosen_process["matrix"]
            merged = merge_two_site(state.tensors[i], state.tensors[j])
            merged = oe.contract("ab, bcd->acd", jump_op, merged)
            # For stochastic jumps, always contract singular values to the right
            tensor_left_new, tensor_right_new = split_two_site(
                merged,
                [state.physical_dimensions[i], state.physical_dimensions[j]],
                svd_distribution="right",
                trunc_mode=cast("TruncMode", sim_params.trunc_mode),
                threshold=sim_params.svd_threshold,
                max_bond_dim=sim_params.max_bond_dim,
            )
            state.tensors[i], state.tensors[j] = tensor_left_new, tensor_right_new
            left_site, right_site = min(i, j), max(i, j)
            state.update_center_after_split(left_site, right_site, "right")

    # Normalize MPS after jump
    state.normalize("B", decomposition="SVD")
    return state
