# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Optimization and surrogate backends for Markovian noise characterization.

Submodules:

- :mod:`.cma` — CMA-ES wrapper for the analytical optimization pipeline
"""

from .cma import cma_opt

__all__ = ["cma_opt"]
