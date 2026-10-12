# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Jump probabilities, channel sampling, and MPS updates for the Tensor Jump Method."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import opt_einsum as oe

from mqt.yaqs.core.methods.dissipation import is_longrange, is_pauli

from ..methods.decompositions import merge_two_site, split_two_site

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..data_structures.mps import MPS
    from ..data_structures.noise_model import NoiseModel
    from ..data_structures.simulation_parameters import AnalogSimParams, DigitalSimParams


def calculate_stochastic_factor(state: MPS) -> NDArray[np.float64]:
    """Compute the jump probability ``1 - ||psi||**2`` after dissipation.

    Args:
        state: Unnormalized MPS after the no-jump dissipative update.

    Returns:
        Zero-dimensional float64 array containing the jump probability.
    """
    return np.asarray(1.0 - state.norm(0) ** 2, dtype=np.float64)


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
        trunc_mode=sim_params.trunc_mode,
        threshold=0.0,
        max_bond_dim=None,
    )
    jumped_state.tensors[site] = tensor_left_new
    jumped_state.tensors[site + 1] = tensor_right_new
    jumped_state.set_center(None)
    return float(jumped_state.norm() ** 2)


def create_probability_distribution(
    state: MPS,
    noise_model: NoiseModel | None,
    dt: float,
    sim_params: AnalogSimParams | DigitalSimParams,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Compute normalized jump-channel weights without truncating candidate states.

    Weights are ``dt * gamma * ||L psi||**2``. Processes follow ascending first
    site, with one-site channels before two-site channels at each site and
    original model order within each group. Use matching indices for processes
    and probabilities when selecting a jump.

    Args:
        state: Post-dissipation MPS. A known center must be at site zero on entry;
            the sweep moves it in place.
        noise_model: Concrete one-site, adjacent two-site, or long-range Pauli
            processes. ``None`` or an empty model returns empty lists.
        dt: Interval used to scale channel weights.
        sim_params: Settings passed to the adjacent jump-weight helper.

    Returns:
        ``(ordered_processes, probabilities)`` with matching channel order.

    Raises:
        NotImplementedError: If a long-range two-site process is not a Pauli product.
        ValueError: If the total jump weight is zero or non-finite.
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
                dp_m = dt * gamma * jumped_state.norm(site) ** 2
                ordered_processes.append(process)
                dp_m_list.append(float(dp_m.real))

        # --- 2-site jumps starting at [site, site+1] ---
        if site < state.length - 1:
            for process in noise_model.processes:
                if len(process["sites"]) == 2 and process["sites"][0] == site:
                    if is_pauli(process):
                        gamma = process["strength"]
                        dp_m = dt * gamma * state.norm(site) ** 2
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
                        raise NotImplementedError(msg)

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


def _normalize_at_zero(state: MPS) -> None:
    """Normalize ``state`` at site zero without a redundant full gauge sweep.

    Args:
        state: MPS normalized in place. Its norm must be nonzero and finite.
    """
    if state.orthogonality_center is None:
        state.set_canonical_form(0)
    center = state.orthogonality_center
    assert center is not None
    state.normalize_center()
    if center != 0:
        state.shift_center_to(0)


def stochastic_process(
    state: MPS,
    noise_model: NoiseModel | None,
    dt: float,
    sim_params: AnalogSimParams | DigitalSimParams,
    rng: np.random.Generator | None = None,
) -> MPS:
    """Sample a quantum jump, then normalize the MPS in place at site zero.

    The jump probability is ``1 - ||psi||**2`` after dissipation. Conditional
    channel probabilities come from :func:`create_probability_distribution`.
    Supports one-site, adjacent two-site, and long-range Pauli jumps.
    Non-Pauli long-range channels are rejected by :func:`create_probability_distribution`.
    Uses the configured truncation when splitting a selected adjacent jump.

    Args:
        state: Post-dissipation MPS; a known center must be at site zero.
        noise_model: Concrete noise model, or ``None`` for normalization only.
        dt: Interval used to scale channel weights.
        sim_params: Truncation settings for the selected adjacent jump.
        rng: Jump generator. ``None`` creates a new generator without using global RNG state.

    Returns:
        The same MPS after the jump or no-jump update.

    Raises:
        ValueError: If normalization or jump weights are invalid, or a selected
            operator has unsupported sites.
    """
    if rng is None:
        rng = np.random.default_rng()

    if state.orthogonality_center is not None:
        state.assert_center(0, context="stochastic_process")

    if noise_model is None:
        _normalize_at_zero(state)
        return state

    dp = calculate_stochastic_factor(state)
    if rng.random() >= dp:
        _normalize_at_zero(state)
        return state

    # A jump occurs: create the probability distribution and select a jump operator.
    ordered_processes, probabilities = create_probability_distribution(state, noise_model, dt, sim_params)

    if len(probabilities) == 0:
        _normalize_at_zero(state)
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
            # NoiseModel stores each full matrix in ascending physical-site order.
            left_site, right_site = sorted((i, j))
            if right_site - left_site != 1:
                msg = f"Only nearest-neighbor 2-site jumps are supported for non-Pauli processes (got sites {i}, {j})"
                raise ValueError(msg)

            # A split can record a pair center only when the surrounding gauge is canonical.
            if not state.check_covers_sites([left_site, right_site]):
                if state.orthogonality_center is None:
                    state.set_canonical_form(left_site)
                else:
                    target = left_site if state.orthogonality_center < left_site else right_site
                    state.shift_center_to(target)

            jump_op = chosen_process["matrix"]
            merged = merge_two_site(state.tensors[left_site], state.tensors[right_site])
            merged = oe.contract("ab, bcd->acd", jump_op, merged)
            # For stochastic jumps, always contract singular values to the right
            tensor_left_new, tensor_right_new = split_two_site(
                merged,
                [state.physical_dimensions[left_site], state.physical_dimensions[right_site]],
                svd_distribution="right",
                trunc_mode=sim_params.trunc_mode,
                threshold=sim_params.svd_threshold,
                max_bond_dim=sim_params.max_bond_dim,
            )
            state.tensors[left_site], state.tensors[right_site] = tensor_left_new, tensor_right_new
            state.update_center_after_split(left_site, right_site, "right")

    # Normalize MPS after jump
    state.normalize("B", decomposition="SVD")
    return state
