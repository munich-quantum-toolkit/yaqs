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
    - ``run_tomography``: process tomography interface (from ``tomography.run``)

Module-level namespaces:
    - ``simulator``
    - ``tomography``
"""

from __future__ import annotations

from . import simulator, tomography
from ._version import version as __version__
from ._version import version_tuple as version_info
from .simulator import run as simulate
from .tomography import DenseComb, MPOComb, TomographyEstimate, run as run_tomography

__all__ = [
    "__version__",
    "version_info",
    "simulator",
    "tomography",
    "simulate",
    "run_tomography",
    "TomographyEstimate",
    "DenseComb",
    "MPOComb",
]
