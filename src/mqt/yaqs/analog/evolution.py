# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared unitary evolution dispatch for analog TJM and ensemble paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.data_structures.simulation_parameters import EvolutionMode
from ..core.methods.bug import bug
from ..core.methods.tdvp import tdvp

if TYPE_CHECKING:
    from ..core.data_structures.mpo import MPO
    from ..core.data_structures.mps import MPS
    from ..core.data_structures.simulation_parameters import AnalogSimParams


def apply_unitary_evolution(state: MPS, hamiltonian: MPO, sim_params: AnalogSimParams) -> None:
    """Advance one unitary time step according to ``sim_params.evolution_mode``.

    Args:
        state: MPS to evolve in place.
        hamiltonian: Hamiltonian as an MPO.
        sim_params: Analog simulation parameters (time step, bond limits, etc.).

    Raises:
        ValueError: If ``evolution_mode`` is not supported.
    """
    if sim_params.evolution_mode == EvolutionMode.TDVP:
        tdvp(state, hamiltonian, sim_params)
    elif sim_params.evolution_mode == EvolutionMode.BUG:
        bug(state, hamiltonian, sim_params)
    else:
        msg = f"Unsupported evolution_mode: {sim_params.evolution_mode!r}"
        raise ValueError(msg)
