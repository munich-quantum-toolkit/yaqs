# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Split-cut operational memory protocol and helpers.

Submodules:

- :mod:`.samples` — :class:`ProbeSet`, :func:`sample_probes`
- :mod:`.grid` — :func:`assemble_probe_sequence`, :func:`assemble_probe_grid`
- :mod:`.branch_weights` — :func:`compute_branch_weights`, :func:`compute_branch_weights_from_simulation`
- :mod:`.response_matrix` — weighted assembly and :func:`compute_spectrum`
- :mod:`.run` — :func:`run_memory_characterization`, :class:`OperationalMemoryBackend`
- :mod:`.results` — :class:`CharacterizationResult`

User code should use :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer` only.
Intervention encoding lives in :mod:`~mqt.yaqs.characterization.memory.shared.interventions`.
"""

from .results import CharacterizationResult

__all__ = ["CharacterizationResult"]
