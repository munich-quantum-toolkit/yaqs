# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Analytical optimization pipeline for Markovian noise fitting.

Fits jump rates by matching simulated observable trajectories to reference data
using a physics forward model and gradient-free optimization (CMA-ES).

Submodules:

- :mod:`.trajectories` — reference simulation, loss assembly, and simulation helpers
- :mod:`.loss` — trajectory-mismatch objective for the optimizer
- :mod:`.run` — :func:`run_optimization_characterization`
- :mod:`.results` — :class:`NoiseCharacterizationResult`

User code should use :class:`~mqt.yaqs.noise_characterizer.NoiseCharacterizer` only.
"""

from .results import NoiseCharacterizationResult
from .run import run_optimization_characterization

__all__ = ["NoiseCharacterizationResult", "run_optimization_characterization"]
