# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe

from ..methods.tdvp import merge_mps_tensors, split_mps_tensor

if TYPE_CHECKING:
    from ..data_structures.networks import MPS
    from ..data_structures.noise_model import NoiseModel
    from ..data_structures.simulation_parameters import AnalogSimParams


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

    for jump in noise_model.scheduled_jumps:
        if np.isclose(jump["time"], time, atol=dt * 1e-3):
            return True
    return False


def apply_scheduled_jumps(
    state: MPS,
    noise_model: NoiseModel,
    time: float,
    sim_params: AnalogSimParams,
) -> MPS:
    """Apply scheduled jumps to the state.

    Args:
        state: The current Matrix Product State.
        noise_model: The noise model containing scheduled jumps.
        time: The current simulation time.
        sim_params: Simulation parameters.

    Returns:
        The updated Matrix Product State.
    """
    if noise_model is None or not noise_model.scheduled_jumps:
        return state

    for jump in noise_model.scheduled_jumps:
        if np.isclose(jump["time"], time, atol=sim_params.dt * 1e-3):
            sites = jump["sites"]
            jump_op = jump["matrix"]

            if len(sites) == 1:
                site = sites[0]
                state.tensors[site] = oe.contract("ab, bcd->acd", jump_op, state.tensors[site])
            elif len(sites) == 2:
                i, j = sites[0], sites[1]
                # Assuming adjacent for now based on NoiseModel constraints or generic apply logic
                if abs(i - j) == 1:
                     merged = merge_mps_tensors(state.tensors[i], state.tensors[j])
                     merged = oe.contract("ab, bcd->acd", jump_op, merged)
                     tensor_left_new, tensor_right_new = split_mps_tensor(
                        merged, "right", sim_params, [state.physical_dimensions[i], state.physical_dimensions[j]], dynamic=False
                    )
                     state.tensors[i], state.tensors[j] = tensor_left_new, tensor_right_new
                else:
                    pass

    state.normalize("B")
    return state
