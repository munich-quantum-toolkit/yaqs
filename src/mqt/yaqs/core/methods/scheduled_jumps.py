# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Scheduled Noise Jumps.

This module implements functions for applying scheduled noise jumps to a Matrix Product State (MPS)
during an analog simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe

from ..methods.decompositions import merge_two_site, split_two_site

if TYPE_CHECKING:
    from ..data_structures.mps import MPS
    from ..data_structures.noise_model import NoiseModel
    from ..data_structures.simulation_parameters import AnalogSimParams


def _scheduled_jump_matches_time(jump_time: float, time: float, dt: float) -> bool:
    """Return True if ``jump_time`` matches ``time`` within ``atol=dt * 1e-3``."""
    return bool(np.isclose(jump_time, time, atol=dt * 1e-3, rtol=0.0))


def has_scheduled_jump(noise_model: NoiseModel | None, time: float, dt: float) -> bool:
    """Check if there is a scheduled jump at the given time.

    Args:
        noise_model: The noise model containing scheduled jumps.
        time: The current simulation time.
        dt: The time step size.

    Returns:
        True if a jump is scheduled at the given time, False otherwise.
    """
    if noise_model is None or not noise_model.scheduled_jumps:
        return False

    return any(_scheduled_jump_matches_time(jump["time"], time, dt) for jump in noise_model.scheduled_jumps)


def apply_scheduled_jumps(
    state: MPS,
    noise_model: NoiseModel | None,
    time: float,
    sim_params: AnalogSimParams,
) -> MPS:
    """Apply scheduled jumps to the state.

    Args:
        state: The current Matrix Product State (MPS).
        noise_model: The noise model containing scheduled jumps.
        time: The current simulation time.
        sim_params: Simulation parameters.

    Returns:
        The updated Matrix Product State (MPS).

    Raises:
        ValueError: If a two-site jump acts on non-adjacent sites, or if a matched
            jump produces a zero or non-finite squared norm.
    """
    if noise_model is None or not noise_model.scheduled_jumps:
        return state

    applied = False
    for jump in noise_model.scheduled_jumps:
        if _scheduled_jump_matches_time(jump["time"], time, sim_params.dt):
            applied = True
            sites = jump["sites"]
            jump_op = jump["matrix"]

            if len(sites) == 1:
                site = sites[0]
                state.tensors[site] = oe.contract("ab, bcd->acd", jump_op, state.tensors[site])
                if state.orthogonality_center is not None and state.orthogonality_center != site:
                    state.set_center(None)
            elif len(sites) == 2:
                i, j = sorted(sites)
                if abs(i - j) != 1:
                    msg = (
                        f"Scheduled jump acts on non-adjacent sites {sites}. Only nearest-neighbor jumps are supported."
                    )
                    raise ValueError(msg)

                merged = merge_two_site(state.tensors[i], state.tensors[j])
                merged = oe.contract("ab, bcd->acd", jump_op, merged)
                tensor_left_new, tensor_right_new = split_two_site(
                    merged,
                    [state.physical_dimensions[i], state.physical_dimensions[j]],
                    svd_distribution="right",
                    trunc_mode=sim_params.trunc_mode,
                    threshold=sim_params.svd_threshold,
                    max_bond_dim=sim_params.max_bond_dim,
                )
                state.tensors[i], state.tensors[j] = tensor_left_new, tensor_right_new
                state.update_center_after_split(i, j, "right")

    if not applied:
        return state

    post_squared_norm = float(state.norm())
    if not np.isfinite(post_squared_norm) or post_squared_norm <= 0.0:
        msg = (
            "Scheduled jump produced a zero or non-finite squared norm "
            f"(squared_norm={post_squared_norm}). The jump operator annihilates the current state."
        )
        raise ValueError(msg)
    state.normalize("B")
    return state
