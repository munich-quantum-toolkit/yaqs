# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Non-Markovian memory characterization via split-cut operational memory.

Package layout (internal; user entry point is :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer`):

- :mod:`.operational_memory` — split-cut probes, branch weights, memory matrix, orchestration
- :mod:`.shared` — encoding, metrics, and MCWF/TJM site-0 helpers shared by backends
- :mod:`.backends` — exact Hamiltonian simulation, comb-schedule sequences, tomography, neural surrogates

Public helpers use compact verb-first names (``sample_probes``, ``assemble_probe_grid``,
``compute_trace_weights``, ``simulate_sequences`` in :mod:`.backends.sequences`, …).
"""
