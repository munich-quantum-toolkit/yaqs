# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dynamic Time-Dependent Variational Principle (TDVP).

This module implements dynamic time evolution of Matrix Product States (MPS)
using the Time-Dependent Variational Principle (TDVP). The dynamic_TDVP function
chooses between a two-site TDVP (2TDVP) and a single-site TDVP (1TDVP) evolution based
on the current maximum bond dimension of the state relative to a specified maximum
in the simulation parameters. This adaptive approach enables efficient evolution
of the state under a Hamiltonian represented as a Matrix Product Operator (MPO).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ..data_structures.simulation_parameters import PhysicsSimParams, StrongSimParams, WeakSimParams
from .tdvp import single_site_tdvp, two_site_tdvp

if TYPE_CHECKING:
    from ..data_structures.networks import MPO, MPS


def dynamic_tdvp(state: MPS, hamiltonian: MPO, sim_params: PhysicsSimParams | StrongSimParams | WeakSimParams) -> None:
    """Perform a dynamic Time-Dependent Variational Principle (TDVP) evolution of the system state.

    This function evolves the state by choosing between a two-site TDVP (2TDVP) and a single-site TDVP (1TDVP)
    based on the current maximum bond dimension of the MPS. The decision is made by comparing the state's bond
    dimension (obtained via `state.write_max_bond_dim()`) to the maximum allowed bond dimension specified in
    `sim_params`.

    Args:
        state (MPS): The Matrix Product State representing the current state of the system.
        hamiltonian (MPO): The Matrix Product Operator representing the Hamiltonian of the system.
        sim_params (PhysicsSimParams | StrongSimParams | WeakSimParams): Simulation parameters containing settings
            such as the maximum allowable bond dimension for the MPS.
    """
    current_max_bond_dim = state.write_max_bond_dim()

    # Manages long-range gates which require low SVD threshold
    save = copy.deepcopy(sim_params.threshold)
    if isinstance(sim_params, (StrongSimParams, WeakSimParams)) and hamiltonian.length > 2:
        sim_params.threshold = 0

    if current_max_bond_dim < sim_params.max_bond_dim:
        # Perform 2TDVP when the current bond dimension is within the allowed limit
        two_site_tdvp(state, hamiltonian, sim_params)
    else:
        # Perform 1TDVP when the bond dimension exceeds the allowed limit
        single_site_tdvp(state, hamiltonian, sim_params)

    sim_params.threshold = save
