# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Guards for the documented ``mqt.yaqs`` public surface."""

from __future__ import annotations

from mqt import yaqs
from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State

# Intentional contract: update when the top-level API changes (see UPGRADING.md).
EXPECTED_PUBLIC_API = frozenset({
    "DenseComb",
    "MPO",
    "MPS",
    "MPOComb",
    "SIMULATION_PRESETS",
    "AnalogSimParams",
    "EquivalenceChecker",
    "Hamiltonian",
    "NoiseModel",
    "Observable",
    "Result",
    "Simulator",
    "State",
    "StrongSimParams",
    "TransformerComb",
    "WeakSimParams",
    "__version__",
    "characterizer",
    "construct_process_tensor",
    "create_surrogate",
    "generate_data",
    "simulator",
    "version_info",
})


def test_public_api_all_matches_documented_surface() -> None:
    """``__all__`` matches the documented top-level export list."""
    assert frozenset(yaqs.__all__) == EXPECTED_PUBLIC_API


def test_top_level_import_smoke() -> None:
    """Exercise the documented import path without ``core.data_structures``."""
    state = State(2, initial="zeros")
    hamiltonian = Hamiltonian.ising(2, J=1.0, g=0.5)
    params = AnalogSimParams(
        observables=[Observable("z", sites=0)],
        elapsed_time=0.1,
        dt=0.05,
        num_traj=1,
        max_bond_dim=4,
        sample_timesteps=False,
    )

    result = Simulator(show_progress=False).run(state, hamiltonian, params)

    assert len(result.expectation_values) == 1
    assert result.observables[0].gate.name == "z"
    assert result.sim_params is params
