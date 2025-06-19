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

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from scipy.linalg import expm

from ..methods.tdvp import merge_mps_tensors, split_mps_tensor

if TYPE_CHECKING:
    from ..data_structures.networks import MPS
    from ..data_structures.noise_model import NoiseModel
    from ..data_structures.simulation_parameters import PhysicsSimParams, StrongSimParams, WeakSimParams


def apply_dissipation(
    state: MPS,
    noise_model: NoiseModel | None,
    dt: float,
    sim_params: PhysicsSimParams | StrongSimParams | WeakSimParams | None,
) -> None:
    """Dissipative sweep: right-to-left, compatible with left-canonical MPS. Assumes state is left-canonical at start.

    This function applies dissipative evolution to an MPS state
    by exponentiating weighted sums of jump operators derived from
    the provided noise model. Both one-site and two-site dissipators
    are handled, and the corresponding operators are applied to the
    appropriate tensors via efficient tensor contractions.
    The function iterates from right to left, updating the
    MPS tensors and shifting the orthogonality center as needed.
    """
    if noise_model is None or sim_params is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        for i in reversed(range(state.length)):
            state.shift_orthogonality_center_left(current_orthogonality_center=i, decomposition="QR")
        return

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
                jump_operator = process["jump_operator"]
                mat = np.conj(jump_operator).T @ jump_operator
                dissipative_operator = expm(-0.5 * dt * gamma * mat)
                state.tensors[i] = oe.contract("ab, bcd->acd", dissipative_operator, state.tensors[i])

            bond = (i - 1, i)
            processes_here = two_site_on_bond.get(bond, [])
            len(processes_here)

        # 2. Apply all 2-site dissipators acting on sites (i-1, i)
        if i != 0:
            for process in processes_here:
                gamma = process["strength"]
                jump_operator = process["jump_operator"]
                mat = np.conj(jump_operator).T @ jump_operator
                dissipative_operator = expm(-0.5 * dt * gamma * mat)

                merged_tensor = merge_mps_tensors(state.tensors[i - 1], state.tensors[i])
                merged_tensor = oe.contract("ab, bcd->acd", dissipative_operator, merged_tensor)

                # singular values always contracted right
                # since ortho center is shifter to the left after loop
                tensor_right, tensor_left = split_mps_tensor(merged_tensor, "right", sim_params, dynamic=False)
                state.tensors[i - 1], state.tensors[i] = tensor_right, tensor_left

        # Shift orthogonality center
        if i != 0:
            state.shift_orthogonality_center_left(current_orthogonality_center=i, decomposition="SVD")
