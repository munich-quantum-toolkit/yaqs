# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Markovian noise-parameter characterization.

Package layout (internal; user entry point is :class:`~mqt.yaqs.noise_characterizer.NoiseCharacterizer`):

- :mod:`.optimization` — analytical trajectory-matching pipeline (CMA-ES)
- :mod:`.shared` — propagation and representation helpers shared across pipelines
- :mod:`.backends` — optimizers and future surrogate/ML backends
"""
