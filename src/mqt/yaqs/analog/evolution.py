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


def apply_unitary_evolution(
    state: MPS,
    hamiltonian: MPO,
    sim_params: AnalogSimParams,
    *,
    normalize: bool = True,
) -> None:
    """Advance one unitary time step according to ``sim_params.evolution_mode``.

    Args:
        state: MPS to evolve in place.
        hamiltonian: Hamiltonian as an MPO.
        sim_params: Analog simulation parameters (time step, bond limits, etc.).
        normalize: When ``evolution_mode`` is BUG, renormalize after compression.
            Ordinary physical states keep the default ``True``. Auxiliary correlator
            states (``B|ψ⟩``) should pass ``False`` so non-unitary probe amplitudes
            are preserved. TDVP ignores this flag.

    Raises:
        ValueError: If ``evolution_mode`` is not supported.
    """
    if sim_params.evolution_mode == EvolutionMode.TDVP:
        tdvp(state, hamiltonian, sim_params)
    elif sim_params.evolution_mode == EvolutionMode.BUG:
        if state.orthogonality_center is None:
            state.set_canonical_form(0, decomposition="QR")
        elif state.orthogonality_center != 0:
            state.shift_center_to(0, decomposition="QR")
        bug(state, hamiltonian, sim_params, normalize=normalize)
    else:
        msg = f"Unsupported evolution_mode: {sim_params.evolution_mode!r}"
        raise ValueError(msg)
