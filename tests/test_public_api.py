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

import pytest
from qiskit.circuit import QuantumCircuit

import mqt.yaqs.simulator as simulator_module
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
from mqt.yaqs.characterization import memory, noise
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import DenseProcessTensor, MPOProcessTensor
from mqt.yaqs.characterization.memory.operational_memory.results import CharacterizationResult
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet
from mqt.yaqs.characterization.noise.optimization.results import NoiseCharacterizationResult

if TYPE_CHECKING:
    from pathlib import Path

# Intentional contract: update when the top-level API changes (see UPGRADING.md).
EXPECTED_PUBLIC_API = frozenset({
    "MPO",
    "MPS",
    "AnalogSimParams",
    "DigitalSimParams",
    "EquivalenceChecker",
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
    "version_info",
})


def test_public_api_all_matches_documented_surface() -> None:
    """``__all__`` matches the documented top-level export list."""
    assert frozenset(yaqs.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(yaqs, name) for name in EXPECTED_PUBLIC_API)
    assert not hasattr(yaqs, "SIMULATION_PRESETS")
    assert not hasattr(yaqs, "EvolutionMode")


def test_simulator_module_exports_execution_facade() -> None:
    """The simulator module supports the facade without a CPU-discovery shortcut."""
    assert simulator_module.__all__ == ["Simulator"]
    assert simulator_module.Simulator is Simulator
    assert not hasattr(simulator_module, "available_cpus")


def test_characterization_result_not_top_level() -> None:
    """CharacterizationResult is returned by MemoryCharacterizer, not a top-level import."""
    assert "CharacterizationResult" not in yaqs.__all__
    assert "ProbeResult" not in yaqs.__all__
    assert "_AnalogSegment" not in yaqs.__all__
    assert "_DigitalSegment" not in yaqs.__all__


def test_memory_result_imports_preserve_type_identity() -> None:
    """Supported memory imports identify the actual result and model classes."""
    assert set(memory.__all__) == {"CharacterizationResult", "DenseProcessTensor", "MPOProcessTensor", "ProbeSet"}
    assert memory.CharacterizationResult is CharacterizationResult
    assert memory.DenseProcessTensor is DenseProcessTensor
    assert memory.MPOProcessTensor is MPOProcessTensor
    assert memory.ProbeSet is ProbeSet


def test_noise_result_import_preserves_type_identity() -> None:
    """The supported noise import identifies the class returned by NoiseCharacterizer."""
    assert noise.__all__ == ["NoiseCharacterizationResult"]
    assert noise.NoiseCharacterizationResult is NoiseCharacterizationResult


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


@pytest.mark.parametrize(
    "imports",
    [
        "import mqt.yaqs",
        (
            "from mqt.yaqs.characterization.memory import "
            "CharacterizationResult, DenseProcessTensor, MPOProcessTensor, ProbeSet\n"
            "from mqt.yaqs.characterization.noise import NoiseCharacterizationResult"
        ),
        (
            "from mqt.yaqs.characterization.noise import NoiseCharacterizationResult\n"
            "from mqt.yaqs.characterization.memory import "
            "CharacterizationResult, DenseProcessTensor, MPOProcessTensor, ProbeSet"
        ),
    ],
    ids=["top-level", "memory-first", "noise-first"],
)
def test_public_imports_need_no_optional_dependencies(tmp_path: Path, imports: str) -> None:
    """Public imports work without Torch, plotting, or loading an optimization runner."""
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        attempted = []

        class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"torch", "matplotlib"}:
                    attempted.append(fullname)
                    raise ModuleNotFoundError(f"{fullname} is intentionally unavailable")

        sys.meta_path.insert(0, BlockOptionalDependencies())
        """
    )
    code += imports + "\n"
    code += textwrap.dedent(
        """
        assert not attempted, attempted
        assert "cma" not in sys.modules
        assert "mqt.yaqs.characterization.noise.optimization.run" not in sys.modules
        assert "mqt.yaqs.characterization.memory.backends.surrogates.model" not in sys.modules
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
