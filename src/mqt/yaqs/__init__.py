# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""YAQS init file.

Yet Another Quantum Simulator (YAQS), a part of the Munich Quantum Toolkit (MQT),
is a package to facilitate simulation for the exploration of noise in quantum systems.
"""

from __future__ import annotations

from ._version import version as __version__
from ._version import version_tuple as version_info

__all__ = ["__version__", "version_info"]
