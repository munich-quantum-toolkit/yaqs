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
    "MPO",
    "MPS",
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
    "WeakSimParams",
    "__version__",
    "version_info",
})

# Symbols that must not appear on the package root (use submodules instead).
INTERNAL_AT_PACKAGE_ROOT = frozenset({
    "DEFAULT_MATRIX_MAX_QUBITS",
    "GateLibrary",
    "tdvp",
})


def test_public_api_all_matches_documented_surface() -> None:
    """``__all__`` matches the documented top-level export list."""
    assert frozenset(yaqs.__all__) == EXPECTED_PUBLIC_API


def test_internal_symbols_not_exposed_at_package_root() -> None:
    """Implementation details are not re-exported on ``mqt.yaqs``."""
    for name in INTERNAL_AT_PACKAGE_ROOT:
        assert not hasattr(yaqs, name), name


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
