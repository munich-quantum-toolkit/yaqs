# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Guards for the documented ``mqt.yaqs`` public surface."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

from qiskit.circuit import QuantumCircuit

from mqt import yaqs
from mqt.yaqs import (
    AnalogSimParams,
    DigitalSimParams,
    Hamiltonian,
    MemoryCharacterizer,
    NoiseCharacterizer,
    Observable,
    Result,
    SimulationProgram,
    Simulator,
    State,
)

if TYPE_CHECKING:
    from pathlib import Path

# Intentional contract: update when the top-level API changes (see UPGRADING.md).
EXPECTED_PUBLIC_API = frozenset({
    "MPO",
    "MPS",
    "SIMULATION_PRESETS",
    "AnalogSimParams",
    "DigitalSimParams",
    "EquivalenceChecker",
    "EvolutionMode",
    "Hamiltonian",
    "MemoryCharacterizer",
    "NoiseCharacterizer",
    "NoiseModel",
    "Observable",
    "Result",
    "Simulator",
    "SimulationProgram",
    "State",
    "__version__",
    "simulator",
    "version_info",
})


def test_public_api_all_matches_documented_surface() -> None:
    """``__all__`` matches the documented top-level export list."""
    assert frozenset(yaqs.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(yaqs, name) for name in EXPECTED_PUBLIC_API)


def test_characterization_result_not_top_level() -> None:
    """CharacterizationResult is returned by MemoryCharacterizer, not a top-level import."""
    assert "CharacterizationResult" not in yaqs.__all__
    assert "ProbeResult" not in yaqs.__all__
    assert "_AnalogSegment" not in yaqs.__all__
    assert "_DigitalSegment" not in yaqs.__all__


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

    assert isinstance(result, Result)
    assert len(result.expectation_values) == 1
    assert result.observables[0].name == "z"
    assert result.sim_params is params

    assert MemoryCharacterizer is not None
    assert NoiseCharacterizer is not None


def test_program_specifications_are_available_from_top_level() -> None:
    """Mixed program specifications use the documented top-level imports."""
    hamiltonian = Hamiltonian.ising(2, J=1.0, g=0.5)
    circuit = QuantumCircuit(2)
    program = SimulationProgram(
        [(hamiltonian, AnalogSimParams()), (circuit, DigitalSimParams())],
        observables=[Observable("z", 0)],
        num_traj=8,
        get_state=True,
    )

    assert tuple(program) == program.segments
    assert program.num_traj == 8
    assert program.get_state


def test_top_level_import_needs_no_optional_matplotlib(tmp_path: Path) -> None:
    """A warnings-as-errors import works when Matplotlib is unavailable."""
    code = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def import_without_matplotlib(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and (name == "matplotlib" or name.startswith("matplotlib.")):
                raise ImportError("Matplotlib is intentionally unavailable")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_without_matplotlib
        import mqt.yaqs
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - sys.executable is trusted.
        [sys.executable, "-W", "error", "-c", code],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not completed.stdout
    assert not completed.stderr
