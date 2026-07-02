# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Markovian noise-parameter characterization via trajectory matching.

Package layout (internal; user entry point is :class:`~mqt.yaqs.noise_characterizer.NoiseCharacterizer`):

- :mod:`.trajectory_matching` — reference trajectories, orchestration, typed results
- :mod:`.shared` — propagation, loss, and representation helpers shared by backends
- :mod:`.backends` — gradient-free optimizers (CMA-ES)
"""
