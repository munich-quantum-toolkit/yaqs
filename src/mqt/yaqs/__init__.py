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
    - ``generate_data``, ``create_surrogate``, ``TransformerComb``: surrogate tomography (from ``mqt.yaqs.tomography``)
    - ``run_exhaustive`` / ``run_estimate``: process tomography (re-exported from ``mqt.yaqs``)

Module-level namespaces:
    - ``simulator``
    - ``tomography``
"""

from __future__ import annotations

from . import simulator, tomography
from ._version import version as __version__
from ._version import version_tuple as version_info
from .simulator import run as simulate
from .tomography import (
    DenseComb,
    MPOComb,
    TomographyEstimate,
    TransformerComb,
    create_surrogate,
    generate_data,
    run_estimate,
    run_exhaustive,
)

__all__ = [
    "__version__",
    "version_info",
    "simulator",
    "tomography",
    "simulate",
    "generate_data",
    "create_surrogate",
    "TransformerComb",
    "run_exhaustive",
    "run_estimate",
    "TomographyEstimate",
    "DenseComb",
    "MPOComb",
]
