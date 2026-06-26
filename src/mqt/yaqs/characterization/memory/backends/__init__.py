# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Process backends for operational-memory characterization.

Subpackages:

- :mod:`.sequences` — :func:`~mqt.yaqs.characterization.memory.backends.sequences.simulate_sequences` and pool workers
- :mod:`.exact` — :class:`~mqt.yaqs.characterization.memory.backends.exact.ExactBackend` and
  :func:`~mqt.yaqs.characterization.memory.backends.exact.simulate_exact`
- :mod:`.tomography` — reference dense/MPO combs via exhaustive process-tensor tomography
- :mod:`.surrogates` — :class:`~mqt.yaqs.characterization.memory.backends.surrogates.model.TransformerComb`
  and training-data workflow
"""
