# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""YAQS init file.

Yet Another Quantum Simulator (YAQS), a part of the Munich Quantum Toolkit (MQT),
is a package to facilitate simulation and process tomography for the exploration
of noise in quantum systems.
"""

from __future__ import annotations

from . import simulator
from ._version import version as __version__
from ._version import version_tuple as version_info
from .core.data_structures.hamiltonian import Hamiltonian
from .core.data_structures.mpo import MPO
from .core.data_structures.mps import MPS
from .core.data_structures.noise_model import NoiseModel
from .core.data_structures.result import Result
from .core.data_structures.simulation_parameters import (
    SIMULATION_PRESETS,
    AnalogSimParams,
    Observable,
    StrongSimParams,
    WeakSimParams,
)
from .core.data_structures.state import State
from .equivalence_checker import EquivalenceChecker
from .memory_characterizer import MemoryCharacterizer
from .simulator import Simulator

__all__ = [
    "MPO",
    "MPS",
    "SIMULATION_PRESETS",
    "AnalogSimParams",
    "EquivalenceChecker",
    "Hamiltonian",
    "MemoryCharacterizer",
    "NoiseModel",
    "Observable",
    "Result",
    "Simulator",
    "State",
    "StrongSimParams",
    "WeakSimParams",
    "__version__",
    "simulator",
    "version_info",
]
