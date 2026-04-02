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

Public entry points:
    - ``simulate``: high-level simulation interface (from ``simulator.run``)
    - ``generate_data``, ``create_surrogate``, ``TransformerComb``: surrogate tools (from ``mqt.yaqs.characterizer``)
    - ``construct``: exhaustive process-tensor tomography (from ``mqt.yaqs.characterizer``)

Module-level namespaces:
    - ``simulator``
    - ``characterizer``
"""

from __future__ import annotations

from . import characterizer, simulator
from ._version import version as __version__
from ._version import version_tuple as version_info
from .simulator import run as simulate
from .characterizer import (
    DenseComb,
    MPOComb,
    SequenceData,
    TransformerComb,
    construct,
    create_surrogate,
    generate_data,
)

__all__ = [
    "__version__",
    "version_info",
    "simulator",
    "characterizer",
    "simulate",
    "generate_data",
    "create_surrogate",
    "TransformerComb",
    "construct",
    "SequenceData",
    "DenseComb",
    "MPOComb",
]
