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

from typing import Any

from . import characterizer, simulator
from ._version import version as __version__
from ._version import version_tuple as version_info
from .characterizer import construct_process_tensor
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
from .simulator import Simulator

_LAZY_EXPORTS = {
    "TransformerComb": ("mqt.yaqs.characterizer", "TransformerComb"),
    "create_surrogate": ("mqt.yaqs.characterizer", "create_surrogate"),
    "generate_data": ("mqt.yaqs.characterizer", "generate_data"),
}

__all__ = [
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
    "TransformerComb",
    "WeakSimParams",
    "__version__",
    "characterizer",
    "construct_process_tensor",
    "create_surrogate",
    "generate_data",
    "simulator",
    "version_info",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        import importlib

        return getattr(importlib.import_module(module_path), attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
